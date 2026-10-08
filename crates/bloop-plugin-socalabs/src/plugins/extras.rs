//! The Gin effects slPlugins does not ship as plug-ins, each behind knobs
//! of bloop's choosing — or, for the Chorus, of SocaLabs Wavetable's.
//!
//! Every wrapper here has a C++ twin in `tests/cpp_harness/extras/`, knob
//! for knob, which is what the gold tests run the port against. The
//! ranges are the ones each class documents, short of the ends that are
//! only good for hurting someone (each wrapper says which); percentages
//! and milliseconds are converted where the knobs are read, once a block,
//! on both sides. A frequency knob prints the way slPlugins' own do, in Hz
//! below a thousand and kHz above (`Text::Frequency`); the text reaches no
//! audio, so the C++ twins do not carry it.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::distortion::AirWindowsDistortion;
use crate::gin::exciter::Exciter;
use crate::gin::maximizer::Maximizer;
use crate::gin::modulation::Modulation;
use crate::gin::param::{Conversion, Param, ParamSpec, Smoothing};
use crate::gin::platereverb::PlateReverb;
use crate::gin::softclipper::SoftClipper;
use crate::gin::text::Text;
use crate::gin::transientshaper::TransientShaper;
use crate::juce::{decibels_to_gain, jlimit, NormalisableRange as Range};
use bloop_plugin_api::meter::{peak_of, DynamicsMeter, LevelTracker};

const SMOOTH: Smoothing = Smoothing::Linear(0.1);

const fn knob(name: &'static str, unit: &'static str, range: Range, default: f32) -> ParamSpec {
    ParamSpec::new(name, unit, range, default, SMOOTH, Conversion::Identity)
}

const fn stepped(name: &'static str, unit: &'static str, range: Range, default: f32) -> ParamSpec {
    ParamSpec::new(
        name,
        unit,
        range,
        default,
        Smoothing::None,
        Conversion::Identity,
    )
}

fn params(specs: &[ParamSpec]) -> Vec<Param> {
    specs.iter().copied().map(Param::new).collect()
}

/// Each knob's value for this block, in the order they were added.
fn read<const N: usize>(params: &mut [Param], samples: usize) -> [f32; N] {
    core::array::from_fn(|i| params[i].proc_value(samples))
}

/// Decay stops at 95%: at the class's 100% the plate never decays, and
/// every note adds to what it holds. Size starts at 10%, below which the
/// tanks are too short to be a plate.
pub const PLATE_REVERB: [ParamSpec; 6] = [
    knob("Mix", "%", Range::new(0.0, 100.0, 0.0, 1.0), 25.0),
    knob("Predelay", "ms", Range::new(0.0, 100.0, 0.0, 1.0), 0.0),
    knob(
        "Lowpass",
        "Hz",
        Range::new(16.0, 20000.0, 0.0, 0.3),
        10000.0,
    )
    .with_text(Text::Frequency),
    knob("Decay", "%", Range::new(0.0, 95.0, 0.0, 1.0), 50.0),
    knob("Size", "%", Range::new(10.0, 200.0, 0.0, 1.0), 100.0),
    knob(
        "Damping",
        "Hz",
        Range::new(16.0, 20000.0, 0.0, 0.3),
        10000.0,
    )
    .with_text(Text::Frequency),
];

#[derive(Clone, Debug)]
pub struct PlateReverbProcessor {
    params: Vec<Param>,
    reverb: PlateReverb,
}

impl Default for PlateReverbProcessor {
    fn default() -> Self {
        Self {
            params: params(&PLATE_REVERB),
            reverb: PlateReverb::new(44100.0),
        }
    }
}

impl Processor for PlateReverbProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.reverb.set_sample_rate(sample_rate as f32);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.reverb.reset();
    }

    fn clear(&mut self) {
        reset_params(&mut self.params);
        self.reverb.clear();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [mix, predelay, lowpass, decay, size, damping] = read(&mut self.params, samples);
        self.reverb.set_mix(mix / 100.0);
        self.reverb.set_predelay(predelay / 1000.0);
        self.reverb.set_lowpass(lowpass);
        self.reverb.set_decay(decay / 100.0);
        self.reverb.set_size(size / 100.0);
        self.reverb.set_damping(damping);
        self.reverb
            .process(&mut left[..samples], &mut right[..samples]);
    }
}

pub const DISTORTION: [ParamSpec; 4] = [
    knob("Density", "", Range::new(-1.0, 4.0, 0.0, 1.0), 0.0),
    knob("Highpass", "%", Range::new(0.0, 100.0, 0.0, 1.0), 0.0),
    knob("Output", "%", Range::new(0.0, 100.0, 0.0, 1.0), 100.0),
    knob("Mix", "%", Range::new(0.0, 100.0, 0.0, 1.0), 100.0),
];

#[derive(Clone, Debug)]
pub struct DistortionProcessor {
    params: Vec<Param>,
    distortion: AirWindowsDistortion,
}

impl Default for DistortionProcessor {
    fn default() -> Self {
        Self {
            params: params(&DISTORTION),
            distortion: AirWindowsDistortion::default(),
        }
    }
}

impl Processor for DistortionProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.distortion.set_sample_rate(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.distortion.reset();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [density, highpass, output, mix] = read(&mut self.params, samples);
        self.distortion.set_params(
            (density + 1.0) / 5.0,
            highpass / 100.0,
            output / 100.0,
            mix / 100.0,
        );
        self.distortion
            .process(&mut left[..samples], &mut right[..samples]);
    }
}

/// Low Cut is off at 0 Hz and High Cut at 20 kHz, the two ends of the
/// band. Output stops at +12 dB: the shaping itself can add 36.
pub const TRANSIENT_SHAPER: [ParamSpec; 8] = [
    knob("Attack", "", Range::new(-4.0, 4.0, 0.0, 1.0), 0.0),
    knob("Sustain", "", Range::new(-8.0, 8.0, 0.0, 1.0), 0.0),
    knob("Smoothing", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    stepped("Low Cut", "Hz", Range::new(0.0, 2000.0, 0.0, 0.5), 0.0).with_text(Text::Frequency),
    stepped(
        "High Cut",
        "Hz",
        Range::new(20.0, 20000.0, 0.0, 0.3),
        20000.0,
    )
    .with_text(Text::Frequency),
    knob("Detector", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    knob("Process", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    knob("Output", "dB", Range::new(-24.0, 12.0, 0.0, 1.0), 0.0),
];

#[derive(Clone, Debug)]
pub struct TransientShaperProcessor {
    params: Vec<Param>,
    shaper: TransientShaper,
}

impl Default for TransientShaperProcessor {
    fn default() -> Self {
        Self {
            params: params(&TRANSIENT_SHAPER),
            shaper: TransientShaper::default(),
        }
    }
}

impl TransientShaperProcessor {
    /// The last block's trace ([`TransientShaper::trace`]).
    #[must_use]
    pub fn trace(&self) -> &bloop_plugin_api::Trace {
        self.shaper.trace()
    }
}

impl Processor for TransientShaperProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.shaper.set_sample_rate(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.shaper.reset();
    }

    fn clear(&mut self) {
        reset_params(&mut self.params);
        self.shaper.clear();
    }

    fn settle(&mut self) {
        reset_params(&mut self.params);
        self.shaper.settle();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [attack, sustain, smoothing, low_cut, high_cut, detector, balance, output] =
            read(&mut self.params, samples);
        self.shaper.set_attack(attack);
        self.shaper.set_sustain(sustain);
        self.shaper.set_smoothing(smoothing / 100.0);
        let high_cut = if high_cut >= 20000.0 { 0.0 } else { high_cut };
        self.shaper.set_band(low_cut, high_cut);
        self.shaper.set_detector_balance(detector / 100.0);
        self.shaper.set_process_balance(balance / 100.0);
        self.shaper.set_output_gain(output);
        self.shaper
            .process(&mut left[..samples], &mut right[..samples]);
    }
}

/// What comes out is held to the ceiling: a half cycle longer than the
/// longest the class holds (DC, anything under 10 Hz) is never scaled, and
/// would otherwise get the whole makeup gain.
pub const MAXIMIZER: [ParamSpec; 3] = [
    knob("Threshold", "dB", Range::new(-24.0, 0.0, 0.0, 1.0), -6.0),
    knob("Ceiling", "dB", Range::new(-24.0, 0.0, 0.0, 1.0), -0.3),
    knob("Release", "ms", Range::new(0.0, 1000.0, 0.0, 0.5), 100.0),
];

#[derive(Clone, Debug)]
pub struct MaximizerProcessor {
    params: Vec<Param>,
    maximizer: Maximizer,
    /// The level in, the level out — after the ceiling — and the least
    /// gain, as Gin's dynamics meter theirs: held 50 ms, then falling (or
    /// rising) 30 dB a second. They touch no audio.
    input_tracker: LevelTracker,
    output_tracker: LevelTracker,
    reduction_tracker: LevelTracker,
}

/// Gin's trackers' fall, in dB a second.
const TRACKER_FALL: f32 = 30.0;

impl Default for MaximizerProcessor {
    fn default() -> Self {
        let mut maximizer = Maximizer::default();
        maximizer.set_num_channels(2);
        Self {
            params: params(&MAXIMIZER),
            maximizer,
            input_tracker: LevelTracker::new(TRACKER_FALL),
            output_tracker: LevelTracker::new(TRACKER_FALL),
            reduction_tracker: LevelTracker::new(-TRACKER_FALL),
        }
    }
}

impl MaximizerProcessor {
    /// The last block's trace ([`Maximizer::trace`]).
    #[must_use]
    pub fn trace(&self) -> &bloop_plugin_api::Trace {
        self.maximizer.trace()
    }

    /// What its screen's meters read: in, out and the gain reduction.
    #[must_use]
    pub fn meter(&self) -> DynamicsMeter {
        DynamicsMeter {
            input_db: self.input_tracker.level(),
            output_db: self.output_tracker.level(),
            reduction_db: [self.reduction_tracker.level(), 0.0, 0.0],
            bands: 1,
        }
    }
}

impl Processor for MaximizerProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.maximizer.set_sample_rate(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.maximizer.reset();
        for tracker in [
            &mut self.input_tracker,
            &mut self.output_tracker,
            &mut self.reduction_tracker,
        ] {
            tracker.reset();
        }
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        #[expect(
            clippy::cast_precision_loss,
            reason = "a block's length, far inside f32"
        )]
        let seconds = samples as f32 / self.maximizer.sample_rate() as f32;
        let [threshold, ceiling, release] = read(&mut self.params, samples);
        self.maximizer.set_threshold(threshold);
        self.maximizer.set_ceiling(ceiling);
        self.maximizer.set_release(release / 1000.0);
        self.maximizer
            .process(&mut left[..samples], &mut right[..samples]);
        let limit = decibels_to_gain(ceiling);
        for x in left[..samples].iter_mut().chain(&mut right[..samples]) {
            *x = jlimit(-limit, limit, *x);
        }
        self.output_tracker.track(
            peak_of(&left[..samples]).max(peak_of(&right[..samples])),
            seconds,
        );
        // In and reduction from the trace, which reads each sample as it
        // goes out: the three meters in step, the lookahead behind the
        // input. A block too short to finish a chunk reads silence and no
        // reduction, and only moves the meters' clocks on.
        let points = self.maximizer.trace().points();
        let peak = points.iter().fold(0.0f32, |m, [p, _]| m.max(*p));
        let least = points.iter().fold(1.0f32, |g, [_, gain]| g.min(*gain));
        self.input_tracker.track(peak, seconds);
        self.reduction_tracker.track(least, seconds);
    }
}

/// Mix stops at 100%, where the class goes to 200%, and Harmonics is
/// skewed toward its bottom: its top quarter turns anything in the band
/// into a square wave.
pub const EXCITER: [ParamSpec; 4] = [
    knob(
        "Frequency",
        "Hz",
        Range::new(1000.0, 16000.0, 0.0, 0.5),
        7000.0,
    )
    .with_text(Text::Frequency),
    knob("Drive", "", Range::new(1.0, 8.0, 0.0, 1.0), 1.0),
    knob("Harmonics", "%", Range::new(0.0, 100.0, 0.0, 0.5), 0.0),
    knob("Mix", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
];

#[derive(Clone, Debug)]
pub struct ExciterProcessor {
    params: Vec<Param>,
    exciter: Exciter,
}

impl Default for ExciterProcessor {
    fn default() -> Self {
        Self {
            params: params(&EXCITER),
            exciter: Exciter::default(),
        }
    }
}

impl Processor for ExciterProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.exciter.set_sample_rate(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.exciter.reset();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [frequency, drive, harmonics, mix] = read(&mut self.params, samples);
        self.exciter.set_frequency(frequency);
        self.exciter.set_drive(drive);
        self.exciter.set_harmonics(harmonics / 100.0);
        self.exciter.set_mix(mix / 100.0);
        self.exciter
            .process(&mut left[..samples], &mut right[..samples]);
    }
}

pub const SOFT_CLIPPER: [ParamSpec; 3] = [
    knob("Drive", "dB", Range::new(0.0, 24.0, 0.0, 1.0), 0.0),
    knob("Knee", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    knob("Ceiling", "dB", Range::new(-24.0, 0.0, 0.0, 1.0), 0.0),
];

#[derive(Clone, Debug)]
pub struct SoftClipperProcessor {
    params: Vec<Param>,
    clipper: SoftClipper,
}

impl Default for SoftClipperProcessor {
    fn default() -> Self {
        Self {
            params: params(&SOFT_CLIPPER),
            clipper: SoftClipper::default(),
        }
    }
}

impl Processor for SoftClipperProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [drive, knee, ceiling] = read(&mut self.params, samples);
        self.clipper.set_drive(drive);
        self.clipper.set_knee(knee / 100.0);
        self.clipper.set_ceiling(ceiling);
        self.clipper
            .process(&mut left[..samples], &mut right[..samples]);
    }
}

/// Wavetable's chorus (`ChorusParams::setup`): its five knobs, ranges and
/// defaults, with Width and Mix as percentages where Wavetable has 0 to 1,
/// smoothed as the other extras' are. The twin is
/// `tests/cpp_harness/extras/Chorus.cpp`.
///
/// Width is how far apart the two ears' sweeps are, as a share of a
/// cycle: at 50% they go opposite ways, which is as wide as it gets, and
/// at 100% they are a whole cycle apart — in step, as at 0%. That is the
/// class's arithmetic, and Wavetable's knob.
pub const CHORUS: [ParamSpec; 5] = [
    knob("Delay", "ms", Range::new(0.1, 30.0, 0.0, 1.0), 1.0),
    knob("Depth", "ms", Range::new(0.1, 20.0, 0.0, 1.0), 1.0),
    knob("Speed", "Hz", Range::new(0.1, 10.0, 0.0, 1.0), 3.0),
    knob("Width", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    knob("Mix", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
];

/// The longest delay the line is built for, in seconds: Wavetable's
/// `gin::Modulation chorus { 0.5f }`. The knobs reach 50 ms of it.
const CHORUS_MAX_DELAY: f32 = 0.5;

#[derive(Clone, Debug)]
pub struct ChorusProcessor {
    params: Vec<Param>,
    chorus: Modulation,
}

impl Default for ChorusProcessor {
    fn default() -> Self {
        Self {
            params: params(&CHORUS),
            chorus: Modulation::new(CHORUS_MAX_DELAY),
        }
    }
}

impl Processor for ChorusProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    /// The line is sized from the class's defaults, not the knobs, so a
    /// reset at the same rate does not build it again (`Modulation::prepare`).
    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.chorus.prepare(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.chorus.reset();
    }

    /// The echoes go and the sweep runs on, so a cleared Chorus is one
    /// that only ever heard silence.
    fn clear(&mut self) {
        reset_params(&mut self.params);
        self.chorus.clear();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [delay, depth, speed, width, mix] = read(&mut self.params, samples);
        self.chorus.set_params(
            delay / 1000.0,
            speed,
            depth / 1000.0,
            width / 100.0,
            mix / 100.0,
        );
        self.chorus
            .process(&mut left[..samples], &mut right[..samples]);
    }
}
