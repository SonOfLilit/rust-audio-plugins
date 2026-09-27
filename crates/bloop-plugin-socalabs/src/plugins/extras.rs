//! The Gin effects slPlugins does not ship as plug-ins, each behind knobs
//! of bloop's choosing.
//!
//! Every wrapper here has a C++ twin in `tests/cpp_harness/extras/`, knob
//! for knob, which is what the gold tests run the port against. The
//! ranges are the ones each class documents; percentages and milliseconds
//! are converted where the knobs are read, once a block, on both sides.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::distortion::AirWindowsDistortion;
use crate::gin::exciter::Exciter;
use crate::gin::maximizer::Maximizer;
use crate::gin::param::{Conversion, Param, ParamSpec, Smoothing};
use crate::gin::platereverb::PlateReverb;
use crate::gin::softclipper::SoftClipper;
use crate::gin::transientshaper::TransientShaper;
use crate::juce::NormalisableRange as Range;

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

pub const PLATE_REVERB: [ParamSpec; 6] = [
    knob("Mix", "%", Range::new(0.0, 100.0, 0.0, 1.0), 25.0),
    stepped("Predelay", "ms", Range::new(0.0, 100.0, 0.0, 1.0), 0.0),
    knob(
        "Lowpass",
        "Hz",
        Range::new(16.0, 20000.0, 0.0, 0.3),
        10000.0,
    ),
    knob("Decay", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    stepped("Size", "%", Range::new(0.0, 200.0, 0.0, 1.0), 100.0),
    knob(
        "Damping",
        "Hz",
        Range::new(16.0, 20000.0, 0.0, 0.3),
        10000.0,
    ),
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

pub const TRANSIENT_SHAPER: [ParamSpec; 8] = [
    knob("Attack", "", Range::new(-4.0, 4.0, 0.0, 1.0), 0.0),
    knob("Sustain", "", Range::new(-8.0, 8.0, 0.0, 1.0), 0.0),
    knob("Smoothing", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    stepped("Low Cut", "Hz", Range::new(0.0, 2000.0, 0.0, 0.5), 0.0),
    stepped("High Cut", "Hz", Range::new(0.0, 20000.0, 0.0, 0.3), 0.0),
    knob("Detector", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    knob("Process", "%", Range::new(0.0, 100.0, 0.0, 1.0), 50.0),
    knob("Output", "dB", Range::new(-24.0, 24.0, 0.0, 1.0), 0.0),
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

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [attack, sustain, smoothing, low_cut, high_cut, detector, balance, output] =
            read(&mut self.params, samples);
        self.shaper.set_attack(attack);
        self.shaper.set_sustain(sustain);
        self.shaper.set_smoothing(smoothing / 100.0);
        self.shaper.set_band(low_cut, high_cut);
        self.shaper.set_detector_balance(detector / 100.0);
        self.shaper.set_process_balance(balance / 100.0);
        self.shaper.set_output_gain(output);
        self.shaper
            .process(&mut left[..samples], &mut right[..samples]);
    }
}

pub const MAXIMIZER: [ParamSpec; 3] = [
    knob("Threshold", "dB", Range::new(-24.0, 0.0, 0.0, 1.0), -6.0),
    knob("Ceiling", "dB", Range::new(-24.0, 0.0, 0.0, 1.0), -0.3),
    knob("Release", "ms", Range::new(0.0, 1000.0, 0.0, 0.5), 100.0),
];

#[derive(Clone, Debug)]
pub struct MaximizerProcessor {
    params: Vec<Param>,
    maximizer: Maximizer,
}

impl Default for MaximizerProcessor {
    fn default() -> Self {
        let mut maximizer = Maximizer::default();
        maximizer.set_num_channels(2);
        Self {
            params: params(&MAXIMIZER),
            maximizer,
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
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let [threshold, ceiling, release] = read(&mut self.params, samples);
        self.maximizer.set_threshold(threshold);
        self.maximizer.set_ceiling(ceiling);
        self.maximizer.set_release(release / 1000.0);
        self.maximizer
            .process(&mut left[..samples], &mut right[..samples]);
    }
}

pub const EXCITER: [ParamSpec; 4] = [
    knob(
        "Frequency",
        "Hz",
        Range::new(1000.0, 16000.0, 0.0, 0.5),
        7000.0,
    ),
    knob("Drive", "", Range::new(1.0, 8.0, 0.0, 1.0), 1.0),
    knob("Harmonics", "%", Range::new(0.0, 100.0, 0.0, 1.0), 0.0),
    knob("Mix", "%", Range::new(0.0, 200.0, 0.0, 1.0), 50.0),
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
