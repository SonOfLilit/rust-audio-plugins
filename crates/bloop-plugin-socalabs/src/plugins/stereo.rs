//! slPlugins' StereoEnhancer and StereoProcessor: faders straight onto
//! `gin::StereoEnhancer` and `gin::StereoProcessor`.
//!
//! Ported from `plugins/{StereoEnhancer,StereoProcessor}/Source/
//! PluginProcessor.cpp`. The StereoEnhancer never tells its DSP the sample
//! rate, so its crossover is where 44.1kHz puts it at any rate; the port
//! keeps that.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::param::{any_smoothing, Conversion, Param, ParamSpec, Smoothing};
use crate::gin::stereo::{StereoEnhancer, StereoProcessor};
use crate::gin::text::Text;
use crate::juce::NormalisableRange as Range;

const SMOOTH: Smoothing = Smoothing::Linear(0.1);
const UNIT: Range = Range::new(0.0, 1.0, 0.0, 1.0);
const BIPOLAR: Range = Range::new(-1.0, 1.0, 0.0, 1.0);

const fn fader(name: &'static str, range: Range, default: f32) -> ParamSpec {
    ParamSpec::new(name, "%", range, default, SMOOTH, Conversion::Identity).with_text(Text::Percent)
}

pub const STEREO_ENHANCER: [ParamSpec; 3] = [
    fader("Width", UNIT, 0.5),
    fader("Freq HP", UNIT, 0.5),
    fader("Width HP", UNIT, 0.0),
];

/// The StereoProcessor's knobs, in the order its constructor adds them.
/// The original names the two ends of the chain alike — its editor sets
/// them apart by where they sit — so here they are numbered for the stage
/// they belong to.
pub const STEREO_PROCESSOR: [ParamSpec; 8] = [
    fader("Width 1", UNIT, 0.5),
    fader("Center 1", UNIT, 0.5),
    fader("Pan 1", BIPOLAR, 0.0),
    fader("Rotation", UNIT, 0.5),
    fader("Pan 2", BIPOLAR, 0.0),
    fader("Center 2", UNIT, 0.5),
    fader("Width 2", UNIT, 0.5),
    fader("Output", UNIT, 0.5),
];

#[derive(Clone, Debug)]
pub struct StereoEnhancerProcessor {
    params: Vec<Param>,
    proc: StereoEnhancer,
}

impl Default for StereoEnhancerProcessor {
    fn default() -> Self {
        Self {
            params: STEREO_ENHANCER.iter().copied().map(Param::new).collect(),
            proc: StereoEnhancer::default(),
        }
    }
}

impl StereoEnhancerProcessor {
    fn apply(&mut self, steps: usize) {
        let p = &mut self.params;
        let width_lp = p[0].proc_value(steps);
        let freq_hp = p[1].proc_value(steps);
        let width_hp = p[2].proc_value(steps);
        self.proc.set_parameters(width_lp, freq_hp, width_hp);
    }
}

impl Processor for StereoEnhancerProcessor {
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
        if any_smoothing(&self.params) {
            for i in 0..samples {
                self.apply(1);
                self.proc.process(&mut left[i..=i], &mut right[i..=i]);
            }
        } else {
            self.apply(samples);
            self.proc
                .process(&mut left[..samples], &mut right[..samples]);
        }
    }
}

#[derive(Clone, Debug)]
pub struct StereoProcessorProcessor {
    params: Vec<Param>,
    proc: StereoProcessor,
}

impl Default for StereoProcessorProcessor {
    fn default() -> Self {
        Self {
            params: STEREO_PROCESSOR.iter().copied().map(Param::new).collect(),
            proc: StereoProcessor::default(),
        }
    }
}

impl StereoProcessorProcessor {
    fn apply(&mut self, steps: usize) {
        let mut v = [0.0f32; 8];
        for (value, param) in v.iter_mut().zip(&mut self.params) {
            *value = param.proc_value(steps);
        }
        self.proc
            .set_parameters(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    }
}

impl Processor for StereoProcessorProcessor {
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
        if any_smoothing(&self.params) {
            for i in 0..samples {
                self.apply(1);
                self.proc.process(&mut left[i..=i], &mut right[i..=i]);
            }
        } else {
            self.apply(samples);
            self.proc
                .process(&mut left[..samples], &mut right[..samples]);
        }
    }
}
