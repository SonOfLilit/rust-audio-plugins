//! The voice's parameter set — the BASIC-mode subset of Just-a-Sample's
//! `SamplerParameters`, with the reference's defaults and ranges.

/// Range metadata for one parameter, mirroring the reference's APVTS layout.
#[derive(Clone, Copy, Debug)]
pub struct ParamRange {
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

impl ParamRange {
    const fn new(min: f32, max: f32, default: f32) -> Self {
        Self { min, max, default }
    }

    pub fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }
}

/// `PluginParameters::WAVETABLE_CUTOFF_HZ`.
pub const WAVETABLE_CUTOFF_HZ: f32 = 20.0;

/// Reference ranges, in the order they appear in `createParameterLayout`.
/// Where a steady lowpass ([`SamplerParams::steady_lowpass`]) cuts at most,
/// as a share of the Sample's Nyquist: 21.6 kHz of a 48 kHz Sample's 24.
pub const STEADY_LOWPASS_OF_NYQUIST: f32 = 0.9;

pub mod ranges {
    use super::ParamRange;

    pub const SEMITONE_TUNING: ParamRange = ParamRange::new(-18.0, 18.0, 0.0);
    pub const CENT_TUNING: ParamRange = ParamRange::new(-100.0, 100.0, 0.0);
    pub const PITCH_WHEEL_RANGE: ParamRange = ParamRange::new(0.0, 12.0, 1.0);
    pub const WIDE_TUNING: ParamRange = ParamRange::new(-48.0, 48.0, 0.0);
    pub const A4_HZ: ParamRange = ParamRange::new(400.0, 480.0, 440.0);
    pub const WAVEFORM_SEMITONE_TUNING: ParamRange = ParamRange::new(-24.0, 24.0, 0.0);
    pub const WAVEFORM_CENT_TUNING: ParamRange = ParamRange::new(-100.0, 100.0, 0.0);
    pub const SPEED_FACTOR: ParamRange = ParamRange::new(0.01, 5.0, 1.0);
    pub const OCTAVE_SPEED_FACTOR: ParamRange = ParamRange::new(0.0, 0.6, 0.0);
    pub const SAMPLE_GAIN: ParamRange = ParamRange::new(-32.0, 16.0, 0.0);
    pub const MIDI_NOTE: ParamRange = ParamRange::new(0.0, 127.0, 0.0);
    pub const ATTACK: ParamRange = ParamRange::new(0.0, 5000.0, 1.0);
    pub const RELEASE: ParamRange = ParamRange::new(0.0, 5000.0, 1.0);
    pub const ATTACK_SHAPE: ParamRange = ParamRange::new(-10.0, 10.0, 0.0);
    pub const RELEASE_SHAPE: ParamRange = ParamRange::new(-10.0, 10.0, 2.0);
    pub const CROSSFADE_SAMPLES: ParamRange = ParamRange::new(0.0, 50000.0, 1000.0);
}

/// What the voice's amplitude does between the Attack and the Release: the
/// Maschine Sampler's envelope TYPE (plugins#amplitude-envelope). The Attack
/// and the Release themselves are the reference's (`attack_ms`,
/// `release_ms`, with their curves); this is the part the reference has no
/// equivalent of, and it is bloop's.
///
/// Every time is in ms of output, as the reference's Attack and Release
/// are. A Decay falls along the Release's curve (`release_shape`), so a
/// Decay to silence sounds like a Release of the same length.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AmpEnvelope {
    /// Nothing between: the reference's voice, bit for bit, whose note-off
    /// `play_until_end` may ignore.
    #[default]
    Reference,
    /// Attack, Hold, Decay: full for `hold_ms` once the Attack is done,
    /// then down to silence over `decay_ms`, which ends the note. Not gated:
    /// a note-off changes nothing, and neither does how long it was held.
    Ahd { hold_ms: f32, decay_ms: f32 },
    /// Attack, Decay, Sustain, Release: down to `sustain` (a level, 0..1)
    /// over `decay_ms` once the Attack is done, and there until the
    /// note-off, which always releases — `play_until_end` or not.
    Adsr { decay_ms: f32, sustain: f32 },
}

/// Everything `CustomSamplerVoice` reads in BASIC playback mode.
///
/// The reference reads these atomically from the APVTS on every block; here the
/// host owns the struct and the voice borrows it, which keeps the read pattern
/// (and therefore the output) identical while staying allocation-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerParams {
    // Tuning
    pub semitone_tuning: i32,
    pub cent_tuning: i32,
    pub pitch_wheel_range: f32,
    pub wide_tuning: f32,
    pub a4_hz: f32,
    pub waveform_semitone_tuning: i32,
    pub waveform_cent_tuning: i32,

    // Playback
    pub speed_factor: f32,
    pub octave_speed_factor: f32,
    pub skip_antialiasing: bool,
    pub disable_wavetable_mode: bool,
    pub play_until_end: bool,
    pub is_looping: bool,
    pub looping_has_start: bool,
    pub looping_has_end: bool,

    // Bounds, in frames of the source sample
    pub sample_start: i32,
    pub sample_end: i32,
    pub loop_start: i32,
    pub loop_end: i32,

    // Envelope
    pub attack_ms: f32,
    pub release_ms: f32,
    pub attack_shape: f32,
    pub release_shape: f32,
    pub crossfade_samples: i32,
    /// Bloop's, not the reference's: Hold, Decay and Sustain after the
    /// reference's Attack. [`AmpEnvelope::Reference`] leaves the voice
    /// exactly the reference's.
    pub amp_envelope: AmpEnvelope,
    /// Bloop's, not the reference's: keep the antialiasing lowpass running
    /// whatever the speed, its cutoff never above [`STEADY_LOWPASS_OF_NYQUIST`]
    /// of the Sample's Nyquist, for a voice whose pitch is modulated. The
    /// reference switches it on only once the speed passes 1, and a pitch
    /// that rises through 1 mid-note then starts an eighth-order filter at
    /// a hair under Nyquist under a sounding note, which clicks. Off, the
    /// voice is the reference's.
    pub steady_lowpass: bool,

    // Output
    pub gain_db: f32,
    pub mono_output: bool,
    pub disable_velocity: bool,

    // MIDI
    pub midi_start: u8,
    pub midi_end: u8,
    pub midi_root: u8,
    pub follow_midi_pitch: bool,
}

impl Default for SamplerParams {
    /// The reference's parameter defaults.
    fn default() -> Self {
        Self {
            semitone_tuning: 0,
            cent_tuning: 0,
            pitch_wheel_range: 1.0,
            wide_tuning: 0.0,
            a4_hz: 440.0,
            waveform_semitone_tuning: 0,
            waveform_cent_tuning: 0,

            speed_factor: 1.0,
            octave_speed_factor: 0.0,
            skip_antialiasing: false,
            disable_wavetable_mode: false,
            play_until_end: false,
            is_looping: false,
            looping_has_start: false,
            looping_has_end: false,

            sample_start: 0,
            sample_end: 0,
            loop_start: 0,
            loop_end: 0,

            attack_ms: 1.0,
            release_ms: 1.0,
            attack_shape: 0.0,
            release_shape: 2.0,
            crossfade_samples: 1000,
            amp_envelope: AmpEnvelope::Reference,
            steady_lowpass: false,

            gain_db: 0.0,
            mono_output: false,
            disable_velocity: false,

            midi_start: 0,
            midi_end: 127,
            midi_root: 69,
            follow_midi_pitch: true,
        }
    }
}

impl SamplerParams {
    /// Defaults with the playback bounds spanning a whole sample.
    pub fn spanning(num_samples: usize) -> Self {
        let last = num_samples.saturating_sub(1) as i32;
        Self {
            sample_end: last,
            loop_end: last,
            ..Self::default()
        }
    }
}
