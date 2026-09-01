//! A 1:1 port of Just-a-Sample's `CustomSamplerVoice` in BASIC playback mode.
//!
//! Structure, naming and evaluation order follow `CustomSamplerVoice.cpp`
//! closely so the two can be diffed side by side. Where the reference relies on
//! C++ implicit conversions the port spells the conversion out, because the
//! gold tests compare the two sample by sample.
//!
//! One deliberate departure: the Lanczos window is tabulated rather than
//! evaluated per sample (see [`WINDOW_TABLE`]). That is the same window
//! reassociated, so the output moves by f32 rounding — a few tenths of an
//! ulp — and every gold scenario pins its own divergence.

use crate::juce::{
    approximately_equal_absolute, decibels_to_gain, jmap, IirCoefficients, IirFilter,
};
use crate::params::{SamplerParams, WAVETABLE_CUTOFF_HZ};
use crate::SampleBuffer;

/// `juce::MathConstants<float>::pi`.
const PI_F: f32 = core::f32::consts::PI;
const LANCZOS_WINDOW_SIZE: i32 = 5;
const INVERSE_SIN_SQUARED: f32 = 1.0 / (PI_F * PI_F);
const TAIL_OFF: usize = 50;

/// Bloop is stereo end to end, so a voice never renders more than two
/// channels; imported audio is folded to at most two on decode.
pub const MAX_CHANNELS: usize = 2;

/// The states a voice moves through, from the reference's `VoiceState`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VoiceState {
    /// Before or during the loop.
    Playing,
    /// Continuing after the loop.
    PlayingEnd,
    #[default]
    Stopped,
}

/// The reference's `VoiceContext`: the per-sample processing state, kept in its
/// own struct so each channel can replay the same transitions.
#[derive(Clone, Copy, Debug, Default)]
struct VoiceContext {
    state: VoiceState,
    /// Fractional positions are necessary.
    current_position: f64,

    /// The initial attack curve.
    is_smoothing_attack: bool,
    is_crossfading_loop: bool,
    /// Crossfading between looping and the end part of the sample.
    is_crossfading_end: bool,
    /// Active when the note is released or when it nears the end of the sample.
    is_releasing: bool,

    crossfade_end_position: f64,
    /// Times the attack envelope. Note this is time passed, not position.
    speed_moved_since_start: f32,
    /// Times the release envelope.
    speed_moved_since_release: f32,
}

/// Stores the lowpass state for one channel, with a circular buffer of recent
/// output large enough for the Lanczos window.
#[derive(Clone, Debug)]
struct LowpassStream {
    filters: [IirFilter; 4],
    intermediate: [f32; (2 * LANCZOS_WINDOW_SIZE) as usize],
    start_sample: i32,
    buffer_loc: usize,
    next_sample: i32,
}

impl LowpassStream {
    fn new() -> Self {
        Self {
            filters: [IirFilter::default(); 4],
            intermediate: [0.0; (2 * LANCZOS_WINDOW_SIZE) as usize],
            start_sample: 0,
            buffer_loc: 0,
            next_sample: 0,
        }
    }

    /// Reset the processing state of the stream to a new sample position.
    fn reset_processing(&mut self, next_sample_to_process: i32) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.buffer_loc = 0;
        self.start_sample = next_sample_to_process;
        self.next_sample = next_sample_to_process;
    }

    fn process_samples(&mut self, samples: &[f32]) {
        for &input in samples {
            let mut processed = input;
            for filter in &mut self.filters {
                processed = filter.process_single_sample_raw(processed);
            }
            self.intermediate[self.buffer_loc] = processed;
            self.next_sample += 1;
            self.buffer_loc = (self.buffer_loc + 1) % self.intermediate.len();
        }
    }

    fn processed_sample(&self, sample_index: i32) -> f32 {
        let len = self.intermediate.len() as i32;
        debug_assert!(
            sample_index >= self.start_sample
                && sample_index < self.next_sample
                && sample_index >= self.next_sample - len
        );
        self.intermediate[((sample_index - self.start_sample) % len) as usize]
    }

    /// Following `juce::dsp::FilterDesign::designIIRLowpassHighOrderButterworthMethod`,
    /// this is theoretically -48 dB above 20 kHz.
    fn set_coefficients(&mut self, sample_rate: i32, frequency: f32) {
        let order = 8.0f32;
        for (index, filter) in self.filters.iter_mut().enumerate() {
            let odd = (2 * index + 1) as f32;
            let q = 1.0 / (2.0 * libm::cosf(odd * PI_F / (order * 2.0)));
            filter.set_coefficients(IirCoefficients::make_low_pass(
                f64::from(sample_rate),
                f64::from(frequency),
                f64::from(q),
            ));
        }
    }
}

/// Which of a voice's three lowpass sets a fetch reads through. The reference
/// keeps three `vector<unique_ptr<LowpassStream>>` members and swaps elements
/// between them at loop and end transitions; the port keeps one array of three
/// and swaps the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamSet {
    Main = 0,
    Loop = 1,
    End = 2,
}

/// The main DSP logic: pitch shifting with antialiasing, looping modes, attack
/// and release, and smooth crossfading.
///
/// Unlike the reference this voice does not own an FX chain, and only BASIC
/// playback mode is implemented (see the crate docs).
pub struct SamplerVoice {
    application_sample_rate: f64,
    max_block_size: usize,

    // Per-note state captured at `note_on`, mirroring the reference's members.
    playing: bool,
    current_note: u8,
    sample_rate_conversion: f32,
    /// Frames of source sample advanced per output sample.
    speed: f32,
    effective_start: i32,
    effective_end: i32,

    /// "Wavetable mode" activates when the bounds are very short and can act as
    /// a waveform cycle.
    wavetable_mode: bool,

    tuning: f32,
    pitch_wheel: i32,
    speed_factor: f32,
    note_velocity: f32,

    play_until_end: bool,
    is_looping: bool,
    looping_has_start: bool,
    looping_has_end: bool,
    sample_start: i32,
    sample_end: i32,
    loop_start: i32,
    loop_end: i32,

    /// The reference calls this "smoothing" but it is a normal attack/release
    /// envelope.
    attack_smoothing: f32,
    release_smoothing: f32,
    attack_shape: f32,
    release_shape: f32,
    crossfade: f32,

    vc: VoiceContext,
    midi_released: bool,

    temp_output: Vec<Vec<f32>>,
    envelope: Vec<Vec<f32>>,

    /// Rendered on voice stealing so a stolen note fades instead of clicking.
    tail_off_buffer: Vec<Vec<f32>>,
    /// Scratch the steal fade is rendered into before it is kept.
    tail_scratch: Vec<Vec<f32>>,
    /// The reference resizes its tail buffer to the sample's channel count; the
    /// rows stay allocated here and this is how many of them are live.
    tail_channels: usize,
    tail_off: usize,

    do_lowpass: bool,
    /// Three lowpass sets (one per channel each), addressed through `assignment`
    /// so "swapping main and loop" is an index swap.
    lowpass: [Vec<LowpassStream>; 3],
}

impl SamplerVoice {
    /// Creates a voice for a host running at `application_sample_rate` with
    /// blocks of at most `max_block_size` frames, playing samples of at most
    /// `max_channels` channels.
    ///
    /// All buffers are allocated here so `render` never allocates. The
    /// interpolation table is built here too, for the same reason.
    pub fn new(application_sample_rate: f64, max_block_size: usize, max_channels: usize) -> Self {
        warm_window_table();
        let block = max_block_size.max(1) * 2;
        let channels = max_channels.clamp(2, MAX_CHANNELS);
        Self {
            application_sample_rate,
            max_block_size: block,

            playing: false,
            current_note: 0,
            sample_rate_conversion: 0.0,
            speed: 0.0,
            effective_start: 0,
            effective_end: 0,
            wavetable_mode: false,
            tuning: 0.0,
            pitch_wheel: 0,
            speed_factor: 0.0,
            note_velocity: 0.0,
            play_until_end: false,
            is_looping: false,
            looping_has_start: false,
            looping_has_end: false,
            sample_start: 0,
            sample_end: 0,
            loop_start: 0,
            loop_end: 0,
            attack_smoothing: 0.0,
            release_smoothing: 0.0,
            attack_shape: 0.0,
            release_shape: 0.0,
            crossfade: 0.0,
            vc: VoiceContext::default(),
            midi_released: false,

            temp_output: vec![vec![0.0; block]; channels],
            envelope: vec![vec![0.0; block]; channels],
            tail_off_buffer: vec![vec![0.0; TAIL_OFF]; channels],
            tail_scratch: vec![vec![0.0; TAIL_OFF]; channels],
            // The reference sizes this buffer to two channels up front and
            // resizes it to the sample's channel count on the next render.
            tail_channels: 2.min(channels),
            tail_off: TAIL_OFF,

            do_lowpass: false,
            lowpass: [
                vec![LowpassStream::new(); channels],
                vec![LowpassStream::new(); channels],
                vec![LowpassStream::new(); channels],
            ],
        }
    }

    /// Whether the voice is actively playing (not stopped).
    pub fn is_playing(&self) -> bool {
        self.playing && self.vc.state != VoiceState::Stopped
    }

    /// Where the voice is in its playback state machine.
    pub fn state(&self) -> VoiceState {
        self.vc.state
    }

    /// Whether the note's bounds were short enough to be played as a single
    /// waveform cycle.
    pub fn is_wavetable_mode(&self) -> bool {
        self.wavetable_mode
    }

    /// Whether the antialiasing lowpass is currently in the signal path. It
    /// engages only when the voice reads the sample faster than 1:1.
    pub fn is_antialiasing(&self) -> bool {
        self.do_lowpass
    }

    /// The effective position within the source sample, for visualisation.
    pub fn position(&self) -> f64 {
        self.vc.current_position
    }

    pub fn current_note(&self) -> u8 {
        self.current_note
    }

    /// The playback-speed multiplier after the octave-tracking adjustment. Only
    /// the reference's ADVANCED mode feeds this into playback; BASIC mode
    /// derives `speed` from the tuning instead, so this is for display.
    pub fn speed_factor(&self) -> f32 {
        self.speed_factor
    }

    /// The condition for wavetable mode.
    pub fn is_wavetable_mode_available(
        sample_rate: f32,
        sample_start: i32,
        sample_end: i32,
    ) -> bool {
        sample_rate / (sample_end - sample_start + 1) as f32 > WAVETABLE_CUTOFF_HZ
    }

    /// `x` should be in `[0, 1]`.
    pub fn exponential_curve(a: f32, x: f32) -> f32 {
        if approximately_equal_absolute(a, 0.0, 0.001) {
            x
        } else {
            (libm::expf(a * x) - 1.0) / (libm::expf(a) - 1.0)
        }
    }

    /// The current envelope gain, for visualisation.
    pub fn envelope_gain(&self) -> f32 {
        let mut gain = 1.0;
        if self.vc.is_smoothing_attack {
            gain *= Self::exponential_curve(
                self.attack_shape,
                self.vc.speed_moved_since_start / self.attack_smoothing,
            );
        }
        if self.vc.is_releasing {
            gain *= Self::exponential_curve(
                self.release_shape,
                1.0 - self.vc.speed_moved_since_release / self.release_smoothing,
            );
        }
        gain
    }

    /// Starts a note. Returns false — leaving the voice stopped — when the note
    /// falls outside the sound's MIDI range, as the reference does.
    pub fn note_on(
        &mut self,
        sample: &SampleBuffer,
        params: &SamplerParams,
        midi_note: u8,
        velocity: f32,
        pitch_wheel: i32,
    ) -> bool {
        if midi_note < params.midi_start || midi_note > params.midi_end {
            return false;
        }
        if sample.num_channels() == 0 {
            return false;
        }

        self.note_velocity = if params.disable_velocity {
            1.0
        } else {
            velocity
        };
        self.playing = true;
        self.current_note = midi_note;

        self.sample_rate_conversion =
            (f64::from(sample.sample_rate()) / self.application_sample_rate) as f32;

        self.sample_start = params.sample_start;
        self.sample_end = params.sample_end;

        self.wavetable_mode = Self::is_wavetable_mode_available(
            sample.sample_rate() as f32,
            self.sample_start,
            self.sample_end,
        ) && !params.disable_wavetable_mode;

        self.play_until_end = params.play_until_end;
        self.is_looping = params.is_looping || self.wavetable_mode;
        self.looping_has_start = self.is_looping
            && params.looping_has_start
            && params.loop_start < params.sample_start
            && !self.wavetable_mode;
        self.loop_start = params.loop_start;
        self.looping_has_end = self.is_looping
            && params.looping_has_end
            && params.loop_end > params.sample_end
            && !self.wavetable_mode;
        self.loop_end = params.loop_end;

        self.effective_start = if self.looping_has_start {
            self.loop_start
        } else {
            self.sample_start
        };
        self.effective_end = if self.looping_has_end {
            self.loop_end
        } else {
            self.sample_end
        };

        let rate = self.application_sample_rate as f32;
        self.attack_smoothing = params.attack_ms * rate / 1000.0;
        self.release_smoothing = params.release_ms * rate / 1000.0;
        self.attack_shape = params.attack_shape;
        self.release_shape = params.release_shape;
        if self.looping_has_end {
            // Keep release smoothing within the end portion.
            self.release_smoothing = self
                .release_smoothing
                .min((self.loop_end - self.sample_end) as f32);
        }
        self.crossfade = (params.crossfade_samples as f32)
            .min((self.sample_end - self.sample_start + 1) as f32 / 2.0 + 1.0);

        self.vc = VoiceContext::default();
        self.midi_released = false;

        self.do_lowpass = false;
        self.vc.current_position = f64::from(self.effective_start);
        self.update_speed_and_pitch(sample, params, midi_note, pitch_wheel);

        self.vc.state = VoiceState::Playing;
        self.vc.is_smoothing_attack = self.attack_smoothing > 0.0;
        true
    }

    /// Releases the note, letting the envelope (or the sample's end portion)
    /// play out.
    pub fn note_off(&mut self) {
        if !self.play_until_end || self.is_looping {
            self.midi_released = true;
        }
    }

    /// Steals the voice: renders a short fade so the cut does not click, then
    /// stops. The fade is mixed into the next block this voice renders.
    pub fn steal(&mut self, sample: &SampleBuffer, params: &SamplerParams) {
        let channels = self.tail_channels;
        let mut scratch = core::mem::take(&mut self.tail_scratch);
        for row in scratch.iter_mut().take(channels) {
            row.fill(0.0);
        }
        {
            let (first, rest) = scratch.split_at_mut(1);
            let mut views: [&mut [f32]; MAX_CHANNELS] =
                [first[0].as_mut_slice(), rest[0].as_mut_slice()];
            self.render_into(sample, params, &mut views[..channels], 0, TAIL_OFF);
        }
        for (kept, rendered) in self.tail_off_buffer.iter_mut().zip(&scratch).take(channels) {
            kept.copy_from_slice(rendered);
        }
        self.tail_scratch = scratch;
        self.tail_off = 0;

        self.vc.state = VoiceState::Stopped;
        self.playing = false;
    }

    /// Point the voice at a new host sample rate and silence it.
    ///
    /// Allocation-free, so a host can call this on the audio thread when a
    /// sampler is swapped onto a track. Buffers keep the size they were built
    /// with, so construct the voice for the largest block the host will ask
    /// for.
    pub fn prepare(&mut self, application_sample_rate: f64) {
        self.application_sample_rate = application_sample_rate;
        self.immediate_halt();
        self.tail_off = TAIL_OFF;
        for set in &mut self.lowpass {
            for stream in set.iter_mut() {
                stream.reset_processing(0);
            }
        }
    }

    /// Stops the voice immediately, with no fade.
    pub fn immediate_halt(&mut self) {
        self.vc.state = VoiceState::Stopped;
        self.playing = false;
    }

    pub fn pitch_wheel_moved(
        &mut self,
        sample: &SampleBuffer,
        params: &SamplerParams,
        new_value: i32,
    ) {
        let note = self.current_note;
        self.update_speed_and_pitch(sample, params, note, new_value);
    }

    /// Updates the speed and pitch, setting the filter cutoffs correctly.
    fn update_speed_and_pitch(
        &mut self,
        sample: &SampleBuffer,
        params: &SamplerParams,
        current_note: u8,
        pitch_wheel_position: i32,
    ) {
        self.pitch_wheel = pitch_wheel_position;

        // Account for tuning adjustments.
        let mut tuning_ratio = 1.0f32;

        if self.wavetable_mode {
            tuning_ratio *= libm::powf(
                2.0,
                (params.waveform_semitone_tuning as f32
                    + params.waveform_cent_tuning as f32 / 100.0)
                    / 12.0,
            );
        } else {
            tuning_ratio *= libm::powf(
                2.0,
                (params.semitone_tuning as f32 + params.cent_tuning as f32 / 100.0) / 12.0,
            );
        }

        let wheel_range = params.pitch_wheel_range;
        tuning_ratio *= libm::powf(
            2.0,
            jmap(
                pitch_wheel_position as f32,
                0.0,
                16383.0,
                -wheel_range,
                wheel_range,
            ) / 12.0,
        );

        tuning_ratio *= libm::powf(2.0, params.wide_tuning / 12.0);

        // Account for the MIDI note.
        if params.follow_midi_pitch {
            let root_note = i32::from(params.midi_root);
            tuning_ratio *= libm::powf(2.0, (i32::from(current_note) - root_note) as f32 / 12.0);
        }

        self.tuning = tuning_ratio;
        self.speed_factor = params.speed_factor;
        self.speed_factor *= 1.0 + (self.tuning - 1.0) * params.octave_speed_factor;

        // In wavetable mode the sample is treated as a single cycle; otherwise
        // it is simply sped up according to the tuning.
        let a4_hz = params.a4_hz;
        self.speed = if self.wavetable_mode {
            (self.tuning * a4_hz)
                * ((self.sample_end - self.sample_start + 1) as f32 - self.crossfade)
                / (self.application_sample_rate as f32)
                / 2.0
        } else {
            self.tuning * self.sample_rate_conversion
        };

        // Configure the filters.
        let sample_rate = sample.sample_rate() as i32;
        let frequency = sample_rate as f32 / 2.0 / self.speed;
        // The filter misbehaves too close to the Nyquist frequency.
        let filter_limit = sample_rate as f32 / 2.0 - 10.0;

        let was_lowpass = self.do_lowpass;
        self.do_lowpass = self.speed > 1.0 && frequency < filter_limit;

        if !was_lowpass && self.do_lowpass {
            let position = self.vc.current_position as i32;
            for ch in 0..sample.num_channels() {
                self.stream_mut(StreamSet::Main, ch)
                    .reset_processing(position);
            }
        }

        if self.do_lowpass {
            for ch in 0..sample.num_channels() {
                for set in [StreamSet::Main, StreamSet::Loop, StreamSet::End] {
                    self.stream_mut(set, ch)
                        .set_coefficients(sample_rate, frequency);
                }
            }
        }
    }

    fn stream_mut(&mut self, set: StreamSet, channel: usize) -> &mut LowpassStream {
        &mut self.lowpass[set as usize][channel]
    }

    /// The reference's `std::swap(mainLowpass[ch], otherLowpass[ch])`.
    fn swap_streams(&mut self, other: StreamSet, channel: usize) {
        let (main, rest) = self.lowpass.split_at_mut(other as usize);
        core::mem::swap(
            &mut main[StreamSet::Main as usize][channel],
            &mut rest[0][channel],
        );
    }

    /// Renders `num_samples` frames into `output` starting at `start_sample`,
    /// adding to whatever is already there.
    pub fn render(
        &mut self,
        sample: &SampleBuffer,
        params: &SamplerParams,
        output: &mut [&mut [f32]],
        start_sample: usize,
        num_samples: usize,
    ) {
        self.render_into(sample, params, output, start_sample, num_samples);
    }

    fn render_into(
        &mut self,
        sample: &SampleBuffer,
        params: &SamplerParams,
        output: &mut [&mut [f32]],
        start_sample: usize,
        num_samples: usize,
    ) {
        // The reference clears the note and bails as soon as the voice stops;
        // without an FX chain there is never a tail to keep rendering for.
        if self.vc.state == VoiceState::Stopped {
            self.playing = false;
            return;
        }
        assert!(
            num_samples <= self.max_block_size,
            "block of {num_samples} frames exceeds the size the voice was prepared for"
        );

        let num_channels = sample.num_channels();
        for ch in 0..num_channels {
            self.temp_output[ch][..num_samples].fill(0.0);
            self.envelope[ch][..num_samples].fill(0.0);
        }

        let note = self.current_note;
        let wheel = self.pitch_wheel;
        self.update_speed_and_pitch(sample, params, note, wheel);

        // Main processing loop.
        let mut con = self.vc;
        for ch in 0..num_channels {
            // Each channel replays the same transitions from the entry state.
            con = self.vc;
            for i in 0..num_samples {
                if con.state == VoiceState::Stopped {
                    continue;
                }

                let mut value =
                    self.fetch_sample(sample, params, ch, con.current_position, StreamSet::Main);

                // Crossfading.
                if con.is_crossfading_loop {
                    let crossfade_position = con.current_position - f64::from(self.sample_start);
                    if crossfade_position >= f64::from(self.crossfade) {
                        con.is_crossfading_loop = false;
                    } else {
                        let (increase, decrease) =
                            Self::crossfade_gains(crossfade_position, self.crossfade);
                        let next = self.fetch_sample(
                            sample,
                            params,
                            ch,
                            con.current_position + f64::from(self.sample_end - self.sample_start)
                                - f64::from(self.crossfade),
                            StreamSet::Loop,
                        );
                        value = value * increase + next * decrease;
                    }
                }

                if con.is_crossfading_end {
                    let crossfade_position = con.current_position - f64::from(self.sample_end);
                    if crossfade_position >= f64::from(self.crossfade) {
                        con.is_crossfading_end = false;
                    } else {
                        let (increase, decrease) =
                            Self::crossfade_gains(crossfade_position, self.crossfade);
                        let next = self.fetch_sample(
                            sample,
                            params,
                            ch,
                            con.crossfade_end_position,
                            StreamSet::End,
                        );
                        value = value * increase + next * decrease;
                        con.crossfade_end_position += f64::from(self.speed);
                    }
                }

                // Attack and release envelopes.
                self.envelope[ch][i] = 1.0;

                if con.is_smoothing_attack {
                    if con.speed_moved_since_start >= self.attack_smoothing {
                        con.is_smoothing_attack = false;
                    } else {
                        self.envelope[ch][i] = Self::exponential_curve(
                            self.attack_shape,
                            con.speed_moved_since_start / self.attack_smoothing,
                        );
                    }
                }

                if con.is_releasing {
                    self.envelope[ch][i] *= Self::exponential_curve(
                        self.release_shape,
                        1.0 - con.speed_moved_since_release / self.release_smoothing,
                    );
                }

                // Update the position.
                con.current_position += f64::from(self.speed);
                con.speed_moved_since_start += 1.0;
                if con.is_releasing {
                    con.speed_moved_since_release += 1.0;
                }

                // Handle transitions.
                if con.state == VoiceState::Playing
                    && self.is_looping
                    && con.current_position >= f64::from(self.sample_end as f32 - self.crossfade)
                {
                    // Loop crossfade.
                    con.current_position -= f64::from(
                        (self.sample_end - self.sample_start + 1) as f32 - self.crossfade,
                    );
                    con.is_crossfading_loop = true;

                    self.swap_streams(StreamSet::Loop, ch);
                    let position = con.current_position as i32;
                    self.stream_mut(StreamSet::Main, ch)
                        .reset_processing(position);
                }

                if self.midi_released && !con.is_releasing && con.state == VoiceState::Playing {
                    // MIDI release, end crossfade.
                    if self.looping_has_end {
                        con.crossfade_end_position = con.current_position;
                        con.current_position = f64::from(self.sample_end + 1);
                        con.state = VoiceState::PlayingEnd;
                        con.is_crossfading_end = true;

                        self.swap_streams(StreamSet::End, ch);
                        let position = con.current_position as i32;
                        self.stream_mut(StreamSet::Main, ch)
                            .reset_processing(position);
                    } else {
                        con.is_releasing = true;
                    }
                }

                if ((con.state == VoiceState::Playing && !self.is_looping)
                    || con.state == VoiceState::PlayingEnd)
                    && !con.is_releasing
                    && con.current_position
                        > f64::from(self.effective_end as f32 - self.release_smoothing * self.speed)
                {
                    // Release smoothing.
                    con.is_releasing = true;
                }

                if con.current_position > f64::from(self.effective_end)
                    || (con.is_releasing && con.speed_moved_since_release >= self.release_smoothing)
                {
                    // End of playback reached.
                    con.state = VoiceState::Stopped;
                }

                // Scale with the standard velocity curve. `40 * log10(v)` picks
                // C++'s double `log10`, so the whole velocity gain is computed
                // in double and only then narrowed — the gold tests catch the
                // one-ULP difference if this is done in f32.
                value = (f64::from(value)
                    * decibels_to_gain_f64(40.0 * libm::log10(f64::from(self.note_velocity))))
                    as f32;
                value *= decibels_to_gain(params.gain_db);

                self.temp_output[ch][i] = value;
            }
        }
        self.vc = con;

        for ch in 0..num_channels {
            for i in 0..num_samples {
                self.temp_output[ch][i] *= self.envelope[ch][i];
            }
        }

        mix_to_buffer(
            &self.temp_output,
            num_channels,
            output,
            start_sample,
            num_samples,
            params.mono_output,
        );

        // Add the previous tail-off samples to the output buffer. The reference
        // resizes its tail buffer here, keeping existing content and clearing
        // any channels the resize adds.
        let new_tail_channels = num_channels.max(1);
        for row in self
            .tail_off_buffer
            .iter_mut()
            .take(new_tail_channels)
            .skip(self.tail_channels)
        {
            row.fill(0.0);
        }
        self.tail_channels = new_tail_channels;

        let mut i = 0;
        while self.tail_off < TAIL_OFF {
            if i >= num_samples {
                break;
            }
            for ch in 0..self.tail_channels.min(output.len()) {
                let faded = self.tail_off_buffer[ch][self.tail_off]
                    * (TAIL_OFF - self.tail_off) as f32
                    / TAIL_OFF as f32;
                output[ch][start_sample + i] += faded;
            }
            self.tail_off += 1;
            i += 1;
        }
    }

    /// The power-preserving crossfade gain pair used for both crossfades.
    fn crossfade_gains(crossfade_position: f64, crossfade: f32) -> (f32, f32) {
        let phase = f64::from(PI_F) * crossfade_position / f64::from(crossfade);
        let increase = (0.5 + 0.5 * libm::cos(phase + f64::from(PI_F))).sqrt() as f32;
        let decrease = (0.5 + 0.5 * libm::cos(phase)).sqrt() as f32;
        (increase, decrease)
    }

    /// Fetch a sample at a given position, in BASIC mode.
    fn fetch_sample(
        &mut self,
        sample: &SampleBuffer,
        params: &SamplerParams,
        channel: usize,
        position: f64,
        set: StreamSet,
    ) -> f32 {
        if !(0.0..f64::from(sample.num_samples() as f32)).contains(&position) {
            return 0.0;
        }

        if params.skip_antialiasing {
            sample.sample(channel, position as usize)
        } else {
            self.lanczos_interpolate(sample, channel, position, set)
        }
    }

    /// Use a Lanczos kernel to calculate fractional sample indices, applying the
    /// lowpass filter beforehand if `do_lowpass`.
    fn lanczos_interpolate(
        &mut self,
        sample: &SampleBuffer,
        channel: usize,
        position: f64,
        set: StreamSet,
    ) -> f32 {
        let num_samples = sample.num_samples() as i32;

        // First, process the lowpass filter.
        if self.do_lowpass {
            let next = self.stream_mut(set, channel).next_sample;
            if next < num_samples {
                let last_window_sample =
                    (position.floor() as i32 + LANCZOS_WINDOW_SIZE).min(num_samples - 1);
                let count = last_window_sample - next + 1;
                if count > 0 {
                    let start = next.max(0) as usize;
                    let end = start + count as usize;
                    let slice = &sample.channel(channel)[start..end];
                    self.lowpass[set as usize][channel].process_samples(slice);
                }
            }
        }

        // Then, interpolate.
        let floor_index = position.floor() as i32;
        let stream = &self.lowpass[set as usize][channel];

        let window = lanczos_taps(position - f64::from(floor_index));

        let mut result = 0.0f32;
        for i in (-LANCZOS_WINDOW_SIZE + 1)..=LANCZOS_WINDOW_SIZE {
            let i_plus = i + floor_index;

            let mut value = 0.0f32;
            if 0 <= i_plus && i_plus < num_samples {
                if self.do_lowpass {
                    if i_plus >= stream.start_sample {
                        value = stream.processed_sample(i_plus);
                    }
                } else {
                    value = sample.sample(channel, i_plus as usize);
                }
            }

            result += value * window[(i + LANCZOS_WINDOW_SIZE - 1) as usize];
        }
        result
    }
}

/// `juce::Decibels::decibelsToGain<double>`.
fn decibels_to_gain_f64(decibels: f64) -> f64 {
    if decibels > -100.0 {
        libm::pow(10.0, decibels * 0.05)
    } else {
        0.0
    }
}

fn lanczos_window(x: f64) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        let pi = f64::from(PI_F);
        (f64::from(LANCZOS_WINDOW_SIZE)
            * libm::sin(pi * x)
            * libm::sin(pi * x / f64::from(LANCZOS_WINDOW_SIZE))
            * f64::from(INVERSE_SIN_SQUARED)
            / (x * x)) as f32
    }
}

/// How many source samples each output sample is built from.
const TAPS: usize = (2 * LANCZOS_WINDOW_SIZE) as usize;

/// Fractional positions the window is tabulated at. Values between rows are
/// linearly interpolated; see `the_table_matches_the_analytic_window` for
/// what that costs.
const PHASES: usize = 1024;

/// The Lanczos window, precomputed at every table phase.
///
/// The reference evaluates [`lanczos_window`] afresh for all ten taps of
/// every output sample of every channel — twenty `sin` calls per sample —
/// which measured as roughly 70% of Bloop's entire audio-thread cost with
/// the default kit's five sample pads. That is affordable for a desktop
/// plugin holding one or two voices and is not affordable on a phone.
///
/// The window depends only on the fractional part of the position, so it is
/// tabulated once instead. This is the standard resampler construction
/// (libsamplerate and r8brain both do it); the reference's per-sample
/// evaluation is a direct transcription of the textbook formula rather than
/// a considered choice — its own comment says as much.
///
/// Row `PHASES` is the window at a fraction of exactly 1, which no position
/// reaches but which interpolation approaches from below.
static WINDOW_TABLE: std::sync::LazyLock<[[f32; TAPS]; PHASES + 1]> =
    std::sync::LazyLock::new(|| {
        core::array::from_fn(|phase| {
            let frac = phase as f64 / PHASES as f64;
            core::array::from_fn(|k| {
                let i = k as i32 - LANCZOS_WINDOW_SIZE + 1;
                lanczos_window(frac - f64::from(i))
            })
        })
    });

/// Build the window table. Call once off the audio thread — the first
/// [`lanczos_taps`] would otherwise pay for it inside a callback.
fn warm_window_table() {
    std::sync::LazyLock::force(&WINDOW_TABLE);
}

/// The ten window coefficients for a position `frac` of the way between two
/// source samples.
fn lanczos_taps(frac: f64) -> [f32; TAPS] {
    let scaled = frac * PHASES as f64;
    let row = (scaled as usize).min(PHASES - 1);
    let blend = (scaled - row as f64) as f32;
    let (low, high) = (&WINDOW_TABLE[row], &WINDOW_TABLE[row + 1]);
    core::array::from_fn(|k| low[k] + blend * (high[k] - low[k]))
}

/// The reference's `mixToBuffer`: fan a voice's channels onto however many
/// channels the host asked for.
///
/// The channel arithmetic is index-based to stay line-for-line with the
/// reference, which is what the gold tests diff against.
#[allow(
    clippy::needless_range_loop,
    reason = "mirrors the reference's channel indexing"
)]
fn mix_to_buffer(
    src: &[Vec<f32>],
    src_channels: usize,
    dest: &mut [&mut [f32]],
    start_sample: usize,
    num_samples: usize,
    mix_mono: bool,
) {
    let dest_channels = dest.len();
    if src_channels == 0 || dest_channels == 0 {
        return;
    }

    if mix_mono {
        let gain = 1.0 / src_channels as f32;
        for ch in 0..src_channels {
            for output_ch in 0..dest_channels {
                for i in 0..num_samples {
                    dest[output_ch][start_sample + i] += src[ch][i] * gain;
                }
            }
        }
    } else if dest_channels < src_channels {
        let ratio = src_channels as f32 / dest_channels as f32;
        for output_ch in 0..dest_channels {
            let start_ch = (output_ch as f32 * ratio) as usize;
            let end_ch = ((output_ch + 1) as f32 * ratio) as usize;
            let gain = 1.0 / (end_ch - start_ch) as f32;
            for ch in start_ch..end_ch {
                for i in 0..num_samples {
                    dest[output_ch][start_sample + i] += src[ch][i] * gain;
                }
            }
        }
    } else {
        let ratio = dest_channels as f32 / src_channels as f32;
        for input_ch in 0..src_channels {
            let start_ch = (input_ch as f32 * ratio) as usize;
            let end_ch = ((input_ch + 1) as f32 * ratio) as usize;
            for output_ch in start_ch..end_ch {
                for i in 0..num_samples {
                    dest[output_ch][start_sample + i] += src[input_ch][i];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// The analytic window, evaluated the way the reference evaluates it.
    fn exact_taps(frac: f64) -> [f32; TAPS] {
        core::array::from_fn(|k| {
            let i = k as i32 - LANCZOS_WINDOW_SIZE + 1;
            lanczos_window(frac - f64::from(i))
        })
    }

    /// How far the table's window strays from the analytic one, sampled at
    /// fractions that do not land on table rows.
    ///
    /// `worst_output` is the bound that matters: the sum of the per-tap
    /// errors is the largest output error possible for a signal that never
    /// leaves ±1, so it converts directly to a noise floor in dBFS.
    fn check(phases_per_step: f64, expect: Expect) {
        let (mut worst_tap, mut worst_output, mut at) = (0.0f32, 0.0f32, 0.0f64);
        let mut frac = 0.0f64;
        while frac < 1.0 {
            let (table, exact) = (lanczos_taps(frac), exact_taps(frac));
            let mut sum = 0.0f32;
            for k in 0..TAPS {
                let error = (table[k] - exact[k]).abs();
                worst_tap = worst_tap.max(error);
                sum += error;
            }
            if sum > worst_output {
                (worst_output, at) = (sum, frac);
            }
            frac += phases_per_step / PHASES as f64;
        }
        expect.assert_eq(&format!(
            "worst tap error   {worst_tap:.3e}\n\
             worst output error {worst_output:.3e} at frac={at:.6} ({:.1} dBFS)",
            20.0 * worst_output.log10()
        ));
    }

    /// Halfway between rows is where linear interpolation is furthest from
    /// the curve it is chasing, so this is the true worst case.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_table_matches_the_analytic_window() {
        check(0.5, expect![[r#"
            worst tap error   4.768e-7
            worst output error 1.245e-6 at frac=0.005371 (-118.1 dBFS)"#]]);
    }

    /// An irrational-ish step visits phases with no relationship to the
    /// table's spacing, in case the aligned sweep flatters it.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_table_matches_at_unaligned_fractions() {
        check(core::f64::consts::E / 7.0, expect![[r#"
            worst tap error   4.172e-7
            worst output error 1.184e-6 at frac=0.010239 (-118.5 dBFS)"#]]);
    }

    /// Rows are hit exactly, so nothing is interpolated and the only error
    /// is the f32 the table stores.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_table_is_exact_on_its_own_rows() {
        check(1.0, expect![[r#"
            worst tap error   0.000e0
            worst output error 0.000e0 at frac=0.000000 (-inf dBFS)"#]]);
    }

    /// A fraction of exactly zero must reproduce the reference's special
    /// case: the sample passes through untouched, no neighbours mixed in.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_whole_position_passes_the_sample_through() {
        let taps = lanczos_taps(0.0);
        let report: Vec<String> = taps.iter().map(|w| format!("{w:.6}")).collect();
        expect!["0.000000 -0.000000 0.000000 -0.000000 1.000000 -0.000000 0.000000 -0.000000 0.000000 0.000000"].assert_eq(&report.join(" "));
    }

    /// The table must come out identical on every platform, or the same
    /// project renders differently in the browser than on the desktop and
    /// the shared demo gold hashes cannot hold. Native `sin` is the C
    /// library's and WASM's is a Rust reimplementation, so this cannot be
    /// taken on trust.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_table_is_identical_on_every_platform() {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for row in WINDOW_TABLE.iter() {
            for coefficient in row {
                hash ^= u64::from(coefficient.to_bits());
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        }
        expect!["ffa5fd68acba7e9b"].assert_eq(&format!("{hash:016x}"));
    }

    /// The window is symmetric about the midpoint: the taps at frac and at
    /// 1-frac must be mirror images. A table built with an off-by-one in the
    /// phase index would still be smooth, but would lose this.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_table_is_symmetric() {
        let mut worst = 0.0f32;
        for step in 1..PHASES {
            let frac = step as f64 / PHASES as f64;
            let (left, right) = (lanczos_taps(frac), lanczos_taps(1.0 - frac));
            for k in 0..TAPS {
                worst = worst.max((left[k] - right[TAPS - 1 - k]).abs());
            }
        }
        expect!["worst asymmetry 0.000e0"].assert_eq(&format!("worst asymmetry {worst:.3e}"));
    }
}
