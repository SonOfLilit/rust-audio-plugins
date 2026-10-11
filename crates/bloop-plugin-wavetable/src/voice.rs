//! One voice: `WavetableVoice` (`WavetableVoice.{h,cpp}`) — two wavetable
//! oscillators, a sub, noise, the filter and its envelope, three
//! modulation envelopes, three LFOs and a step LFO, and the amplifier —
//! with the note it plays as JUCE's `MPESynthesiserVoice` keeps it.

use crate::dsp::env::{Adsr, AnalogAdsr, AnalogState};
use crate::dsp::lfo::{Lfo, LfoParams, Shape, StepLfo};
use crate::dsp::osc::{StereoOscillator, StereoParams, VoicedParams, VoicedWtOscillator, MAX_UNISON};
use crate::dsp::random::JuceRandom;
use crate::dsp::tables::{AnalogTables, NoiseSource, Table, Wave};
use crate::modmatrix::{src, Matrix, VoiceMod, DESTINATION_OF};
use crate::params::{id, Values, FILTER_MAX_NOTE};
use bloop_plugin_socalabs::shared::{
    approximately_equal, jlimit, math, midi_note_in_hertz, EasedValueSmoother, Filter,
    FilterType, Slope, DURATIONS,
};

/// The longest stretch a voice renders at once: Wavetable's 32-sample
/// slices.
pub const MAX_BLOCK: usize = 32;

/// `gin::Q`, Butterworth.
const Q: f32 = 0.707_106_77;

/// `juce::MPEValue`: a 14-bit value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpeValue(pub i32);

impl MpeValue {
    /// `from7BitInt`: 0 to 64 by sevenfold shift, 65 to 127 stretched to
    /// the top.
    #[must_use]
    pub fn from_7bit(value: i32) -> Self {
        if value <= 64 {
            Self(value << 7)
        } else {
            let mapped = 0.0 + ((8191.0f32 - 0.0) * ((value - 64) as f32 - 0.0)) / (63.0 - 0.0);
            Self(mapped as i32 + 8192)
        }
    }

    pub const MIN: Self = Self(0);
    pub const CENTRE: Self = Self(64 << 7);

    #[must_use]
    pub fn unsigned(self) -> f32 {
        0.0 + ((1.0f32 - 0.0) * (self.0 as f32 - 0.0)) / (16383.0 - 0.0)
    }

    #[must_use]
    pub fn signed(self) -> f32 {
        if self.0 < 8192 {
            -1.0 + ((0.0f32 - -1.0) * (self.0 as f32 - 0.0)) / (8192.0 - 0.0)
        } else {
            0.0 + ((1.0f32 - 0.0) * (self.0 as f32 - 8192.0)) / (16383.0 - 8192.0)
        }
    }
}

/// `MPENote::KeyState`, as far as a voice needs it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KeyState {
    #[default]
    Off,
    Down,
}

/// `juce::MPENote`: one note as the synth tracks it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Note {
    /// 0 for no note (`isValid` false); otherwise `(channel << 7) + note`.
    pub id: u16,
    pub channel: u8,
    #[expect(clippy::struct_field_names, reason = "juce::MPENote's own name for it")]
    pub initial_note: u8,
    pub velocity: MpeValue,
    pub pitchbend: MpeValue,
    pub pressure: MpeValue,
    pub timbre: MpeValue,
    pub key: KeyState,
    pub total_pitchbend: f64,
}

impl Note {
    #[must_use]
    pub fn new(channel: u8, note: u8, velocity: MpeValue) -> Self {
        Self {
            id: (u16::from(channel) << 7) + u16::from(note),
            channel,
            initial_note: note,
            velocity,
            pitchbend: MpeValue::CENTRE,
            pressure: MpeValue::MIN,
            timbre: MpeValue::CENTRE,
            key: KeyState::Down,
            total_pitchbend: 0.0,
        }
    }

    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.channel > 0 && self.channel <= 16
    }
}

/// `gin::GlideInfo`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlideInfo {
    pub from_note: f32,
    pub glissando: bool,
    pub portamento: bool,
    pub rate: f32,
}

impl Default for GlideInfo {
    fn default() -> Self {
        Self {
            from_note: -1.0,
            glissando: false,
            portamento: false,
            rate: 0.0,
        }
    }
}

/// What a voice reads of the synth around it.
pub struct Context<'a> {
    pub values: &'a Values,
    pub matrix: &'a Matrix,
    pub tables: [Option<&'a Table>; 2],
    pub analog: Option<&'a AnalogTables>,
    pub noise: &'a mut NoiseSource,
    /// `proc.rng`.
    pub rng: &'a mut JuceRandom,
    /// `juce::Random::getSystemRandom()`.
    pub system: &'a mut JuceRandom,
    pub bpm: f32,
}

impl Context<'_> {
    /// The seconds a Beat knob's note lasts.
    fn beat_seconds(&self, index: usize) -> f32 {
        DURATIONS[index.min(DURATIONS.len() - 1)].to_seconds(self.bpm)
    }
}

#[derive(Clone, Debug)]
pub struct Voice {
    pub note: Note,
    pub note_on_time: u32,
    pub glide: GlideInfo,
    pub fast_kill: bool,
    pub modulation: VoiceMod,
    sample_rate: f64,
    oscillators: [VoicedWtOscillator; 2],
    noise: StereoOscillator,
    sub: StereoOscillator,
    filter: Filter,
    filter_adsr: Adsr,
    mod_adsrs: [Adsr; 3],
    lfos: [Lfo; 3],
    step_lfo: StepLfo,
    adsr: AnalogAdsr,
    current_notes: [f32; 2],
    osc_params: [VoicedParams; 2],
    sub_note: f32,
    sub_params: StereoParams,
    noise_params: StereoParams,
    note_smoother: EasedValueSmoother,
    amp_key_track: f32,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            note: Note::default(),
            note_on_time: 0,
            glide: GlideInfo::default(),
            fast_kill: false,
            modulation: VoiceMod::default(),
            sample_rate: 0.0,
            oscillators: [VoicedWtOscillator::default(); 2],
            noise: StereoOscillator::default(),
            sub: StereoOscillator::default(),
            filter: Filter::new(FilterType::LoPass),
            filter_adsr: Adsr::default(),
            mod_adsrs: [Adsr::default(); 3],
            lfos: [Lfo::default(); 3],
            step_lfo: StepLfo::default(),
            adsr: AnalogAdsr::default(),
            current_notes: [0.0; 2],
            osc_params: [VoicedParams::default(); 2],
            sub_note: 0.0,
            sub_params: StereoParams::default(),
            noise_params: StereoParams::default(),
            note_smoother: EasedValueSmoother::default(),
            amp_key_track: 1.0,
        }
    }
}

impl Voice {
    /// `isActive`: holding a note, down or ringing out.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.note.is_valid()
    }

    #[must_use]
    pub fn is_playing_but_released(&self) -> bool {
        self.is_active() && self.note.key == KeyState::Off
    }

    #[must_use]
    pub fn is_playing(&self, note: &Note) -> bool {
        self.is_active() && self.note.id == note.id
    }

    /// `getCurrentNote`: where the glide has got to.
    #[must_use]
    pub fn current_note(&self) -> f32 {
        self.note_smoother.current() * 127.0
    }

    /// `clearCurrentNote`.
    pub fn clear_note(&mut self) {
        self.note = Note::default();
    }

    /// `setCurrentSampleRate`.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for o in &mut self.oscillators {
            o.set_sample_rate(sample_rate);
        }
        self.noise.set_sample_rate(sample_rate);
        self.sub.set_sample_rate(sample_rate);
        self.filter.set_sample_rate(sample_rate);
        self.filter_adsr.set_sample_rate(sample_rate);
        for l in &mut self.lfos {
            l.set_sample_rate(sample_rate);
        }
        for a in &mut self.mod_adsrs {
            a.set_sample_rate(sample_rate);
        }
        self.step_lfo.set_sample_rate(sample_rate);
        self.note_smoother.set_sample_rate(sample_rate);
        self.adsr.set_sample_rate(sample_rate);
    }

    /// `getValue (p)` through the matrix, for this voice.
    fn value(&mut self, ctx: &Context, param: usize) -> f32 {
        let dst = DESTINATION_OF[param];
        let spec = &crate::params::PARAMS[param];
        let base = ctx.values.normalised(param);
        let turn = |v: f32| spec.conversion.apply(spec.range.convert_from_0to1(v));
        if dst >= 0 {
            let dst = dst as usize;
            let v = ctx.matrix.value_voice(&mut self.modulation, dst, base);
            self.modulation.turned(dst, v, turn)
        } else {
            turn(base)
        }
    }

    fn osc_value(&mut self, ctx: &Context, osc: usize, offset: usize) -> f32 {
        self.value(ctx, id::OSC[osc] + offset)
    }

    /// `noteStarted`.
    pub fn note_started(&mut self, ctx: &mut Context) {
        self.fast_kill = false;
        let note = self.note;
        if self.glide.from_note != -1.0 && (self.glide.glissando || self.glide.portamento) {
            self.note_smoother.set_time(f64::from(self.glide.rate));
            self.note_smoother
                .set_value_unsmoothed(self.glide.from_note / 127.0);
            self.note_smoother
                .set_value(f32::from(note.initial_note) / 127.0);
        } else {
            self.note_smoother
                .set_value_unsmoothed(f32::from(note.initial_note) / 127.0);
        }
        self.modulation.values[src::VELOCITY] = note.velocity.unsigned();
        self.modulation.values[src::TIMBRE] = note.timbre.unsigned();
        self.modulation.values[src::MPE_BEND] = note.pitchbend.unsigned();
        self.modulation.values[src::PRESSURE] = note.pressure.unsigned();

        self.modulation.disable_smoothing = true;
        self.filter.reset();
        self.filter_adsr.reset();
        for a in &mut self.mod_adsrs {
            a.reset();
        }
        for l in &mut self.lfos {
            l.reset();
        }
        self.update_params(ctx, 0);
        self.modulation.snap();
        self.update_params(ctx, 0);
        self.modulation.snap();

        for index in 0..2 {
            let base = id::OSC[index];
            let retrig = ctx.values.is_on(base + id::osc::RETRIG);
            let voices = ctx.values.user(base + id::osc::UNISON) as i32;
            if voices == 1 {
                let phase = if retrig { 0.0 } else { ctx.rng.next_float() };
                self.oscillators[index].note_on_all(phase);
            } else if retrig {
                let mut phases = [0.0f32; MAX_UNISON];
                for (i, p) in phases.iter_mut().enumerate().take(voices as usize) {
                    *p = 1.0 / voices as f32 * i as f32;
                }
                self.oscillators[index].note_on(&phases);
            } else {
                let mut phases = [0.0f32; MAX_UNISON];
                for p in &mut phases {
                    *p = ctx.rng.next_float();
                }
                self.oscillators[index].note_on(&phases);
            }
        }

        self.noise.note_on(ctx.system.next_float());
        let sub_phase = if ctx.values.is_on(id::SUB_RETRIG) {
            0.0
        } else {
            ctx.rng.next_float()
        };
        self.sub.note_on(sub_phase);

        self.filter_adsr.note_on();
        for a in &mut self.mod_adsrs {
            a.note_on();
        }
        for (i, l) in self.lfos.iter_mut().enumerate() {
            let phase = if ctx.values.is_on(id::LFO[i] + id::lfo::RETRIG) {
                -1.0
            } else {
                ctx.rng.next_float()
            };
            l.note_on(phase, ctx.system);
        }
        self.step_lfo.reset();
        let step_phase = if ctx.values.is_on(id::SLFO_RETRIG) {
            -1.0
        } else {
            ctx.rng.next_float()
        };
        self.step_lfo.note_on(step_phase);

        self.adsr.reset();
        self.adsr.note_on();
        self.modulation.disable_smoothing = false;
    }

    /// `noteRetriggered`: a mono voice handed a new note.
    pub fn note_retriggered(&mut self, ctx: &mut Context) {
        let note = self.note;
        if self.glide.from_note != -1.0 && (self.glide.glissando || self.glide.portamento) {
            self.note_smoother.set_time(f64::from(self.glide.rate));
            self.note_smoother
                .set_value(f32::from(note.initial_note) / 127.0);
        } else {
            self.note_smoother
                .set_value_unsmoothed(f32::from(note.initial_note) / 127.0);
        }
        self.modulation.values[src::VELOCITY] = note.velocity.unsigned();
        self.modulation.values[src::TIMBRE] = note.timbre.unsigned();
        self.modulation.values[src::PRESSURE] = note.pressure.unsigned();
        self.modulation.values[src::MPE_BEND] = note.pitchbend.unsigned();

        self.update_params(ctx, 0);

        if ctx.values.is_on(id::FLT_RETRIG) {
            self.filter_adsr.note_on();
        }
        for (i, a) in self.mod_adsrs.iter_mut().enumerate() {
            if ctx.values.is_on(id::ENV[i] + id::env::RETRIG) {
                a.note_on();
            }
        }
        if ctx.values.is_on(id::AMP_RETRIG) {
            self.adsr.note_on();
        }
    }

    /// `noteStopped (allowTailOff)`: the envelopes released; without a
    /// tail, the voice freed at once. Returns whether it was freed.
    pub fn note_stopped(&mut self, allow_tail_off: bool) -> bool {
        self.adsr.note_off();
        self.filter_adsr.note_off();
        for a in &mut self.mod_adsrs {
            a.note_off();
        }
        if allow_tail_off {
            false
        } else {
            self.clear_note();
            true
        }
    }

    /// `notePitchbendChanged`.
    pub fn pitchbend_changed(&mut self) {
        self.modulation.values[src::MPE_BEND] = self.note.pitchbend.unsigned();
    }

    /// `updateParams (blockSize)`.
    #[expect(
        clippy::too_many_lines,
        reason = "the original's one function, read against it line by line"
    )]
    pub fn update_params(&mut self, ctx: &mut Context, block: usize) {
        let note = self.note;
        self.modulation.values[src::NOTE] = f32::from(note.initial_note) / 127.0;

        for i in 0..2 {
            let base = id::OSC[i];
            if !ctx.values.is_on(base + id::osc::ENABLE) {
                continue;
            }
            let mut n = self.note_smoother.current() * 127.0;
            if self.glide.glissando {
                n = round_to_int(n) as f32;
            }
            n += note.total_pitchbend as f32;
            n += self.osc_value(ctx, i, id::osc::TUNE)
                + self.osc_value(ctx, i, id::osc::FINETUNE) / 100.0;
            self.current_notes[i] = n;

            let voices = ctx.values.user(base + id::osc::UNISON) as i32;
            let position = self.osc_value(ctx, i, id::osc::POS) / 100.0;
            let pan = self.osc_value(ctx, i, id::osc::PAN);
            let spread = self.osc_value(ctx, i, id::osc::SPREAD) / 100.0;
            let detune = self.osc_value(ctx, i, id::osc::DETUNE);
            let gain = self.osc_value(ctx, i, id::osc::LEVEL) * 0.5;
            let formant = self.osc_value(ctx, i, id::osc::FORMANT);
            let bend = self.osc_value(ctx, i, id::osc::BEND);
            self.osc_params[i] = VoicedParams {
                voices,
                pan,
                spread,
                detune,
                gain,
                position,
                bend,
                formant,
            };
        }

        if ctx.values.is_on(id::SUB_ENABLE) {
            let mut n = self.note_smoother.current() * 127.0;
            if self.glide.glissando {
                n = round_to_int(n) as f32;
            }
            n += note.total_pitchbend as f32;
            n += self.value(ctx, id::SUB_TUNE);
            self.sub_note = n;
            let (wave, pw) = match ctx.values.user(id::SUB_WAVE) as i32 {
                0 => (Wave::Sine, self.sub_params.pw),
                1 => (Wave::Triangle, self.sub_params.pw),
                2 => (Wave::SawUp, self.sub_params.pw),
                3 => (Wave::Pulse, 0.5),
                4 => (Wave::Pulse, 0.25),
                _ => (Wave::Pulse, 0.125),
            };
            self.sub_params.wave = wave;
            self.sub_params.pw = pw;
            self.sub_params.left_gain =
                self.value(ctx, id::SUB_LEVEL) * (1.0 - self.value(ctx, id::SUB_PAN));
            self.sub_params.right_gain =
                self.value(ctx, id::SUB_LEVEL) * (1.0 + self.value(ctx, id::SUB_PAN));
        }

        if ctx.values.is_on(id::NOISE_ENABLE) {
            self.noise_params.wave = if ctx.values.user(id::NOISE_TYPE) as i32 == 0 {
                Wave::WhiteNoise
            } else {
                Wave::PinkNoise
            };
            self.noise_params.left_gain =
                self.value(ctx, id::NOISE_LEVEL) * (1.0 - self.value(ctx, id::NOISE_PAN));
            self.noise_params.right_gain =
                self.value(ctx, id::NOISE_LEVEL) * (1.0 + self.value(ctx, id::NOISE_PAN));
        }

        self.amp_key_track = self.value(ctx, id::AMP_VEL);

        if ctx.values.is_on(id::FLT_ENABLE) {
            let attack = self.value(ctx, id::FLT_ATTACK);
            self.filter_adsr.set_attack(attack);
            let sustain = self.value(ctx, id::FLT_SUSTAIN);
            self.filter_adsr.set_sustain_level(sustain);
            let decay = self.value(ctx, id::FLT_DECAY);
            self.filter_adsr.set_decay(decay);
            let release = self.value(ctx, id::FLT_RELEASE);
            self.filter_adsr.set_release(release);
            self.filter_adsr.process(block);

            let filter_width = FILTER_MAX_NOTE;
            let filter_env = self.filter_adsr.output();
            let mut sens = self.value(ctx, id::FLT_VEL);
            sens = note.velocity.unsigned() * sens + 1.0 - sens;

            let mut n = self.value(ctx, id::FLT_FREQ);
            n += (i32::from(note.initial_note) - 60) as f32 * self.value(ctx, id::FLT_KEY);
            n += filter_env * sens * self.value(ctx, id::FLT_AMOUNT) * filter_width;

            let f = midi_note_in_hertz(n);
            let max_freq = 20000.0f32.min((self.sample_rate / 2.0) as f32);
            let f = jlimit(4.0, max_freq, f);
            let q = Q / (1.0 - (self.value(ctx, id::FLT_RES) / 100.0) * 0.99);

            let (kind, slope) = match ctx.values.user(id::FLT_TYPE) as i32 {
                0 => (FilterType::LoPass, Slope::Db12),
                1 => (FilterType::LoPass, Slope::Db24),
                2 => (FilterType::HiPass, Slope::Db12),
                3 => (FilterType::HiPass, Slope::Db24),
                4 => (FilterType::BandPass, Slope::Db12),
                5 => (FilterType::BandPass, Slope::Db24),
                6 => (FilterType::Notch, Slope::Db12),
                _ => (FilterType::Notch, Slope::Db24),
            };
            self.filter.set_type(kind);
            self.filter.set_slope(slope);
            self.filter.set_params(f, q, 0.0);
            self.modulation.values[src::FILTER_ENV] = self.filter_adsr.output();
        } else {
            self.modulation.values[src::FILTER_ENV] = 0.0;
        }

        for i in 0..3 {
            let base = id::ENV[i];
            if ctx.values.is_on(base + id::env::ENABLE) {
                let attack = self.value(ctx, base + id::env::ATTACK);
                self.mod_adsrs[i].set_attack(attack);
                let sustain = self.value(ctx, base + id::env::SUSTAIN);
                self.mod_adsrs[i].set_sustain_level(sustain);
                let decay = self.value(ctx, base + id::env::DECAY);
                self.mod_adsrs[i].set_decay(decay);
                let release = self.value(ctx, base + id::env::RELEASE);
                self.mod_adsrs[i].set_release(release);
                self.modulation.values[src::ENV[i]] = self.mod_adsrs[i].output();
                self.mod_adsrs[i].process(block);
            } else {
                self.modulation.values[src::ENV[i]] = 0.0;
            }
        }

        for i in 0..3 {
            let base = id::LFO[i];
            if ctx.values.is_on(base + id::lfo::ENABLE) {
                let frequency = if ctx.values.user(base + id::lfo::SYNC) > 0.0 {
                    1.0 / ctx.beat_seconds(ctx.values.user(base + id::lfo::BEAT) as usize)
                } else {
                    self.value(ctx, base + id::lfo::RATE)
                };
                let shape = Shape::from_index(ctx.values.user(base + id::lfo::WAVE) as i32);
                let phase = self.value(ctx, base + id::lfo::PHASE);
                let offset = self.value(ctx, base + id::lfo::OFFSET);
                let depth = self.value(ctx, base + id::lfo::DEPTH);
                let delay = self.value(ctx, base + id::lfo::DELAY);
                let fade = self.value(ctx, base + id::lfo::FADE);
                let params = LfoParams {
                    shape,
                    frequency,
                    phase,
                    offset,
                    depth,
                    delay,
                    fade,
                };
                self.lfos[i].set_parameters(params, ctx.system);
                self.lfos[i].process(block);
                self.modulation.values[src::LFO[i]] = self.lfos[i].output();
            } else {
                self.modulation.values[src::LFO[i]] = 0.0;
            }
        }

        if ctx.values.is_on(id::SLFO_ENABLE) {
            let frequency = 1.0 / ctx.beat_seconds(ctx.values.user(id::SLFO_BEAT) as usize);
            self.step_lfo.set_freq(frequency);
            let n = ctx.values.user(id::SLFO_LENGTH) as usize;
            self.step_lfo.set_num_points(n);
            for i in (0..n).rev() {
                self.step_lfo
                    .set_point(i, ctx.values.user(id::SLFO_STEP1 + i));
            }
            self.step_lfo.process(block);
            self.modulation.values[src::STEP] = self.step_lfo.output();
        } else {
            self.modulation.values[src::STEP] = 0.0;
        }

        let attack = self.value(ctx, id::AMP_ATTACK);
        self.adsr.set_attack(attack);
        let decay = self.value(ctx, id::AMP_DECAY);
        self.adsr.set_decay(decay);
        let sustain = self.value(ctx, id::AMP_SUSTAIN);
        self.adsr.set_sustain_level(sustain);
        let release = if self.fast_kill {
            0.01
        } else {
            self.value(ctx, id::AMP_RELEASE)
        };
        self.adsr.set_release(release);

        self.note_smoother.process(block);
    }

    /// `renderNextBlock`: this voice added into `left`/`right`. Returns
    /// whether it fell silent and was freed (`stopVoice`).
    pub fn render(&mut self, ctx: &mut Context, left: &mut [f32], right: &mut [f32]) -> bool {
        let n = left.len().min(MAX_BLOCK);
        self.update_params(ctx, n);

        let mut pre_l = [0.0f32; MAX_BLOCK];
        let mut pre_r = [0.0f32; MAX_BLOCK];
        let mut post_l = [0.0f32; MAX_BLOCK];
        let mut post_r = [0.0f32; MAX_BLOCK];
        let (pre_l, pre_r) = (&mut pre_l[..n], &mut pre_r[..n]);
        let (post_l, post_r) = (&mut post_l[..n], &mut post_r[..n]);

        for i in 0..2 {
            if !ctx.values.is_on(id::OSC[i] + id::osc::ENABLE) {
                continue;
            }
            let filtered = ctx.values.is_on(if i == 0 { id::FLT_WT1 } else { id::FLT_WT2 });
            let (l, r) = if filtered {
                (&mut *pre_l, &mut *pre_r)
            } else {
                (&mut *post_l, &mut *post_r)
            };
            self.oscillators[i].process_adding(
                ctx.tables[i],
                self.current_notes[i],
                &self.osc_params[i],
                l,
                r,
            );
        }

        if let Some(analog) = ctx.analog {
            if ctx.values.is_on(id::SUB_ENABLE) {
                let (l, r) = if ctx.values.is_on(id::FLT_SUB) {
                    (&mut *pre_l, &mut *pre_r)
                } else {
                    (&mut *post_l, &mut *post_r)
                };
                self.sub
                    .process_adding(analog, ctx.noise, self.sub_note, &self.sub_params, l, r);
            }
            if ctx.values.is_on(id::NOISE_ENABLE) {
                let (l, r) = if ctx.values.is_on(id::FLT_NOISE) {
                    (&mut *pre_l, &mut *pre_r)
                } else {
                    (&mut *post_l, &mut *post_r)
                };
                self.noise
                    .process_adding(analog, ctx.noise, 60.0, &self.noise_params, l, r);
            }
        }

        let gain = velocity_to_gain(self.note.velocity.unsigned(), self.amp_key_track);
        apply_gain(pre_l, pre_r, gain);
        apply_gain(post_l, post_r, gain);

        if ctx.values.is_on(id::FLT_ENABLE) {
            for (l, r) in pre_l.iter_mut().zip(pre_r.iter_mut()) {
                *l = self.filter.process(0, *l);
                *r = self.filter.process(1, *r);
            }
        }

        for i in 0..n {
            post_l[i] += pre_l[i];
            post_r[i] += pre_r[i];
        }

        self.adsr.process_multiplying(post_l, post_r);

        let freed = self.adsr.state() == AnalogState::Idle;
        if freed {
            self.clear_note();
        }

        for i in 0..n {
            left[i] += post_l[i];
            right[i] += post_r[i];
        }

        self.modulation.finish_block(n);
        freed
    }
}

/// `juce::roundToInt` for a float.
fn round_to_int(value: f32) -> i32 {
    (f64::from(value) + 6_755_399_441_055_744.0).to_bits() as u32 as i32
}

/// `gin::velocityToGain`.
fn velocity_to_gain(velocity: f32, sensitivity: f32) -> f32 {
    let v = velocity * sensitivity + 1.0 - sensitivity;
    jlimit(0.0, 1.0, v * math::powf(25.0, v) * 0.04)
}

/// `juce::AudioBuffer::applyGain`: nothing at 1, silence at 0.
fn apply_gain(left: &mut [f32], right: &mut [f32], gain: f32) {
    if approximately_equal(gain, 1.0) {
        return;
    }
    for channel in [left, right] {
        if approximately_equal(gain, 0.0) {
            channel.fill(0.0);
        } else {
            for x in channel.iter_mut() {
                *x *= gain;
            }
        }
    }
}
