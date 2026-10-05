//! The oscillators: Gin's `WTOscillator` and `WTVoicedStereoOscillator`
//! (`gin_wtoscillators.h`), `StereoOscillator` (`gin_oscillators.h`) and
//! the `DCBlocker` a wavetable oscillator runs (`gin_dcblocker.h`).

use super::tables::{linear_in, AnalogTables, NoiseSource, Wave, Wavetable};
use bloop_plugin_socalabs::shared::{jlimit, math};

/// `std::numeric_limits<float>::epsilon()` below 1.
const ALMOST_ONE: f32 = 1.0 - f32::EPSILON;

/// `gin::DCBlocker`: a one-pole low-pass whose output is the DC to
/// subtract.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DcBlocker {
    a: f32,
    b: f32,
    z: f32,
}

impl DcBlocker {
    /// `setSampleRate` then `setCutoff (10)`, as `WTOscillator` does.
    pub fn prepare(&mut self, sample_rate: f32) {
        self.b = math::expf(-2.0 * core::f32::consts::PI * 10.0 / sample_rate);
        self.a = 1.0 - self.b;
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.z = x * self.a + self.z * self.b;
        self.z
    }
}

/// `440 * 2^((note - 69) / 12)`, held below Nyquist, in double, then the
/// phase step it makes.
#[inline]
fn phase_delta(note: f32, sample_rate: f64) -> f32 {
    let freq =
        (sample_rate / 2.0).min(440.0 * math::pow(2.0, (f64::from(note) - 69.0) / 12.0)) as f32;
    1.0 / (((1.0 / freq) as f64 * sample_rate) as f32)
}

/// `WTOscillator::Params`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WtParams {
    pub left_gain: f32,
    pub right_gain: f32,
    pub position: f32,
    pub bend: f32,
    pub formant: f32,
}

/// `gin::WTOscillator`: one wavetable voice, moving to a new frame only at
/// the end of a cycle and crossfading the cycle after a move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WtOscillator {
    sample_rate: f64,
    phase: f32,
    table_index: i32,
    last_table_index: i32,
    blocker_l: DcBlocker,
    blocker_r: DcBlocker,
}

impl Default for WtOscillator {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            phase: 0.0,
            table_index: 0,
            last_table_index: -1,
            blocker_l: DcBlocker::default(),
            blocker_r: DcBlocker::default(),
        }
    }
}

/// `bendDistortion`: the phase pushed toward the end of the cycle (bend
/// above 0) or its start (below).
#[inline]
fn bend_distortion(phase: f32, bend: f32) -> f32 {
    let add_dist = bend.clamp(0.0, 1.0);
    let sub_dist = bend.clamp(-1.0, 0.0);
    let sub = pow8(phase);
    let add = pow8(1.0 - phase);
    let add_mix = lerp(phase, 1.0 - add, add_dist);
    let sub_mix = lerp(phase, sub, -sub_dist);
    ALMOST_ONE.min(add_mix + sub_mix - phase)
}

/// `ln 5` as the original writes it, `1.60943791243f`.
#[expect(
    clippy::excessive_precision,
    reason = "the original's literal, digit for digit, so the float is its float"
)]
const LN_5: f32 = 1.609_437_912_43;

/// `formantDistortion`: the cycle read faster, up to five times, and held
/// at its end. `speed` is `exp (formant * ln 5)`, which the original
/// works out every sample and the port once a block: the same float.
#[inline]
fn formant_distortion(phase: f32, speed: f32) -> f32 {
    ALMOST_ONE.min(phase * speed)
}

/// How much faster `formantDistortion` reads the cycle at `formant`.
fn formant_speed(formant: f32) -> f32 {
    math::expf(formant * LN_5)
}

#[inline]
fn phase_distortion(mut phase: f32, bend: f32, formant: f32, speed: f32) -> f32 {
    if bend != 0.0 {
        phase = bend_distortion(phase, bend);
    }
    if formant != 0.0 {
        phase = formant_distortion(phase, speed);
    }
    phase
}

/// `gin::math::lerp`: `a + t * (b - a)`.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// `gin::math::pow8`: seven multiplications in a row, not three squarings.
#[inline]
fn pow8(x: f32) -> f32 {
    x * x * x * x * x * x * x * x
}

impl WtOscillator {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.blocker_l.prepare(sample_rate as f32);
        self.blocker_r.prepare(sample_rate as f32);
    }

    /// `noteOn (p)`: start at phase `p`, on a fresh frame.
    pub fn note_on(&mut self, phase: f32) {
        self.phase = phase;
        self.table_index = -1;
        self.last_table_index = -1;
    }

    fn frame_for(table: &Wavetable, position: f32) -> i32 {
        (table.len() as i32 - 1).min((table.len() as f32 * position) as i32)
    }

    /// `processAdding (note, params, buffer)`: added into `left`/`right`,
    /// then the DC blocked from the whole buffer.
    pub fn process_adding(
        &mut self,
        table: Option<&Wavetable>,
        note: f32,
        params: &WtParams,
        left: &mut [f32],
        right: &mut [f32],
    ) {
        if let Some(table) = table.filter(|t| !t.is_empty()) {
            if self.table_index != self.last_table_index && self.last_table_index >= 0 {
                self.run(table, note, params, left, right, Mode::Crossfade);
            } else if params.bend == 0.0 && params.formant == 0.0 {
                self.run(table, note, params, left, right, Mode::Simple);
            } else {
                self.run(table, note, params, left, right, Mode::Complex);
            }
        }
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            *l -= self.blocker_l.process(*l);
            *r -= self.blocker_r.process(*r);
        }
    }

    fn run(
        &mut self,
        table: &Wavetable,
        note: f32,
        params: &WtParams,
        left: &mut [f32],
        right: &mut [f32],
        mode: Mode,
    ) {
        let len = table.len() as i32;
        if self.table_index == -1 || self.table_index >= len {
            self.table_index = Self::frame_for(table, params.position);
        }
        if mode == Mode::Crossfade && (self.last_table_index == -1 || self.last_table_index >= len)
        {
            self.last_table_index = Self::frame_for(table, params.position);
        }

        let delta = phase_delta(note, self.sample_rate);
        let speed = formant_speed(params.formant);
        // The copy a note reads is the same all block: looked up once a
        // frame, not once a sample.
        let mut frame1 = table.frame(self.table_index as usize).copy_for_note(note);
        let mut frame2 = if mode == Mode::Crossfade {
            table.frame(self.last_table_index as usize).copy_for_note(note)
        } else {
            frame1
        };

        let mut at = 0;
        let mut samps = left.len() as i32;
        while samps > 0 {
            let mut todo = samps.min(((1.0 - self.phase) / delta) as i32 + 1);
            samps -= todo;
            // Four at a time, as Gin's SIMD path runs wherever it is built
            // with it (x86 and ARM, so every desktop build): each lane's
            // phase is the group's plus a multiple of the step, the group
            // moves on by four steps at once, and a crossfade weighs all
            // four by the group's phase.
            while todo >= 4 {
                let base = self.phase;
                for lane in 0..4 {
                    let phase = base + lane as f32 * delta;
                    let s = Self::sample(mode, frame1, frame2, params, speed, phase, base);
                    left[at] += s * params.left_gain;
                    right[at] += s * params.right_gain;
                    at += 1;
                }
                self.phase += delta * 4.0;
                todo -= 4;
            }
            for _ in 0..todo {
                let s = Self::sample(mode, frame1, frame2, params, speed, self.phase, self.phase);
                left[at] += s * params.left_gain;
                right[at] += s * params.right_gain;
                at += 1;
                self.phase += delta;
            }
            while self.phase >= 1.0 {
                self.phase -= 1.0;
                self.last_table_index = self.table_index;
                self.table_index = Self::frame_for(table, params.position);
                frame1 = table.frame(self.table_index as usize).copy_for_note(note);
                if mode == Mode::Crossfade {
                    frame2 = frame1;
                }
            }
        }
    }

    /// One sample at `phase`, a crossfade weighed by `weight`.
    #[inline]
    fn sample(
        mode: Mode,
        frame1: &[f32],
        frame2: &[f32],
        params: &WtParams,
        speed: f32,
        phase: f32,
        weight: f32,
    ) -> f32 {
        let phase = ALMOST_ONE.min(phase);
        match mode {
            Mode::Simple => linear_in(frame1, phase),
            Mode::Complex => {
                linear_in(frame1, phase_distortion(phase, params.bend, params.formant, speed))
            }
            Mode::Crossfade => {
                let p = phase_distortion(phase, params.bend, params.formant, speed);
                let s1 = linear_in(frame1, p);
                let s2 = linear_in(frame2, p);
                s1 * weight + s2 * (1.0 - weight)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Simple,
    Complex,
    Crossfade,
}

/// `WTVoicedStereoOscillatorParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoicedParams {
    pub voices: i32,
    pub pan: f32,
    pub spread: f32,
    pub detune: f32,
    pub gain: f32,
    pub position: f32,
    pub bend: f32,
    pub formant: f32,
}

impl Default for VoicedParams {
    fn default() -> Self {
        Self {
            voices: 1,
            pan: 0.0,
            spread: 0.0,
            detune: 0.0,
            gain: 1.0,
            position: 0.5,
            bend: 0.0,
            formant: 0.0,
        }
    }
}

/// The most unison voices an oscillator runs.
pub const MAX_UNISON: usize = 8;

/// `gin::WTVoicedStereoOscillator`: up to eight copies, detuned across
/// `detune` semitones and panned across `spread`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoicedWtOscillator {
    oscillators: [WtOscillator; MAX_UNISON],
}

impl VoicedWtOscillator {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        for o in &mut self.oscillators {
            o.set_sample_rate(sample_rate);
        }
    }

    /// `noteOn (phases)`, one phase per copy.
    pub fn note_on(&mut self, phases: &[f32; MAX_UNISON]) {
        for (o, p) in self.oscillators.iter_mut().zip(phases) {
            o.note_on(*p);
        }
    }

    /// `noteOn (phase)`, the same phase for every copy.
    pub fn note_on_all(&mut self, phase: f32) {
        for o in &mut self.oscillators {
            o.note_on(phase);
        }
    }

    pub fn process_adding(
        &mut self,
        table: Option<&Wavetable>,
        note: f32,
        params: &VoicedParams,
        left: &mut [f32],
        right: &mut [f32],
    ) {
        let mut p = WtParams {
            position: params.position,
            bend: params.bend,
            formant: params.formant,
            ..WtParams::default()
        };
        if params.voices == 1 {
            p.left_gain = params.gain * (1.0 - params.pan);
            p.right_gain = params.gain * (1.0 + params.pan);
            self.oscillators[0].process_adding(table, note, &p, left, right);
        } else {
            let voices = params.voices as f32;
            let base_note = note - params.detune / 2.0;
            let note_delta = params.detune / (voices - 1.0);
            let base_pan = params.pan - params.spread;
            let pan_delta = (params.spread * 2.0) / (voices - 1.0);
            let norm = f64::from(params.voices).sqrt() as f32;
            for (i, osc) in self
                .oscillators
                .iter_mut()
                .enumerate()
                .take(params.voices as usize)
            {
                let pan = jlimit(-1.0, 1.0, base_pan + pan_delta * i as f32);
                p.left_gain = params.gain * (1.0 - pan) / norm;
                p.right_gain = params.gain * (1.0 + pan) / norm;
                osc.process_adding(table, base_note + note_delta * i as f32, &p, left, right);
            }
        }
    }
}

/// `StereoOscillator::Params`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoParams {
    pub wave: Wave,
    pub left_gain: f32,
    pub right_gain: f32,
    pub pw: f32,
}

impl Default for StereoParams {
    fn default() -> Self {
        Self {
            wave: Wave::SawUp,
            left_gain: 1.0,
            right_gain: 1.0,
            pw: 0.5,
        }
    }
}

/// `gin::StereoOscillator`: a band-limited analog shape or noise, for the
/// sub oscillator and the noise source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoOscillator {
    sample_rate: f64,
    phase: f32,
}

impl Default for StereoOscillator {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            phase: 0.0,
        }
    }
}

impl StereoOscillator {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn note_on(&mut self, phase: f32) {
        self.phase = phase;
    }

    pub fn process_adding(
        &mut self,
        tables: &AnalogTables,
        noise: &mut NoiseSource,
        note: f32,
        params: &StereoParams,
        left: &mut [f32],
        right: &mut [f32],
    ) {
        let delta = phase_delta(note, self.sample_rate);
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let s = match params.wave {
                Wave::WhiteNoise => noise.white.next_sample(),
                Wave::PinkNoise => noise.pink.next_sample(),
                wave => tables.process(wave, note, self.phase, params.pw),
            };
            *l += s * params.left_gain;
            *r += s * params.right_gain;
            self.phase += delta;
            while self.phase >= 1.0 {
                self.phase -= 1.0;
            }
        }
    }
}
