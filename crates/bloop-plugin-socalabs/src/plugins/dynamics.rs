//! slPlugins' Compressor, Limiter, Gate and Expander: four processors over
//! one `gin::Dynamics`, differing in their knobs and in how those reach it.
//!
//! Ported from `plugins/{Compressor,Limiter,Gate,Expander}/Source/
//! PluginProcessor.cpp`. The MIDI learn, preset files and scope FIFO are
//! left behind; none of them reaches audio.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::dynamics::{Dynamics, DynamicsType};
use crate::gin::param::{any_smoothing, Conversion, Param, ParamSpec, Smoothing};
use crate::juce::NormalisableRange as Range;

const SMOOTH: Smoothing = Smoothing::Linear(0.1);

const fn ms(name: &'static str, range: Range, default: f32) -> ParamSpec {
    ParamSpec::new(name, "ms", range, default, SMOOTH, Conversion::MsToSeconds)
}

const fn db(name: &'static str, unit: &'static str, range: Range, default: f32) -> ParamSpec {
    ParamSpec::new(
        name,
        unit,
        range,
        default,
        SMOOTH,
        Conversion::DecibelsToGain,
    )
}

const fn plain(name: &'static str, unit: &'static str, range: Range, default: f32) -> ParamSpec {
    ParamSpec::new(name, unit, range, default, SMOOTH, Conversion::Identity)
}

/// The Compressor's knobs, in the order its constructor adds them. The
/// original labels the Compressor's and Expander's decibel knobs with no
/// unit, where the Limiter's and Gate's say dB; here all four say dB.
pub const COMPRESSOR: [ParamSpec; 7] = [
    ms("Attack", Range::new(1.0, 200.0, 0.0, 0.4), 1.0),
    ms("Release", Range::new(1.0, 2000.0, 0.0, 0.4), 5.0),
    plain("Ratio", "", Range::new(1.0, 30.0, 0.0, 0.4), 5.0),
    plain("Threshold", "dB", Range::new(-60.0, 0.0, 0.0, 1.0), -30.0),
    plain("Knee", "dB", Range::new(0.0, 60.0, 0.0, 1.0), 5.0),
    db("Input", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
    db("Output", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
];

pub const LIMITER: [ParamSpec; 5] = [
    ms("Attack", Range::new(0.0, 5.0, 0.0, 1.0), 0.0),
    ms("Release", Range::new(1.0, 500.0, 0.0, 0.3), 5.0),
    plain("Threshold", "dB", Range::new(-60.0, 0.0, 0.0, 1.0), -30.0),
    db("Input", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
    db("Output", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
];

pub const GATE: [ParamSpec; 7] = [
    ms("Attack", Range::new(0.0, 5.0, 0.0, 1.0), 0.0),
    ms("Hold", Range::new(0.0, 500.0, 0.0, 0.3), 0.0),
    ms("Release", Range::new(1.0, 500.0, 0.0, 0.3), 5.0),
    plain("Threshold", "dB", Range::new(-100.0, 0.0, 0.0, 1.0), -30.0),
    plain("Knee", "dB", Range::new(0.0, 60.0, 0.0, 1.0), 0.0),
    db("Input", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
    db("Output", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
];

pub const EXPANDER: [ParamSpec; 7] = [
    ms("Attack", Range::new(1.0, 200.0, 0.0, 0.4), 1.0),
    ms("Release", Range::new(1.0, 2000.0, 0.0, 0.4), 5.0),
    plain("Ratio", "", Range::new(1.0, 4.0, 0.0, 1.0), 1.0),
    plain("Threshold", "dB", Range::new(-60.0, 0.0, 0.0, 1.0), -30.0),
    plain("Knee", "dB", Range::new(0.0, 60.0, 0.0, 1.0), 5.0),
    db("Input", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
    db("Output", "dB", Range::new(-30.0, 30.0, 0.0, 1.0), 0.0),
];

/// Which of the four processors this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavour {
    Compressor,
    Limiter,
    Gate,
    Expander,
}

impl Flavour {
    #[must_use]
    pub fn specs(self) -> &'static [ParamSpec] {
        match self {
            Self::Compressor => &COMPRESSOR,
            Self::Limiter => &LIMITER,
            Self::Gate => &GATE,
            Self::Expander => &EXPANDER,
        }
    }
}

/// One of the four dynamics processors.
#[derive(Clone, Debug)]
pub struct DynamicsProcessor {
    flavour: Flavour,
    params: Vec<Param>,
    dynamics: Dynamics,
}

impl DynamicsProcessor {
    #[must_use]
    pub fn new(flavour: Flavour) -> Self {
        Self {
            flavour,
            params: flavour.specs().iter().copied().map(Param::new).collect(),
            dynamics: Dynamics::default(),
        }
    }

    /// Hand the knobs to the detector: the body of the processor's
    /// `processBlock` that runs once per block, or once per sample while
    /// any knob is still smoothing. `knee_steps` is how far the Gate's knee
    /// advances, which its unsmoothed branch has as 1 rather than the block
    /// length.
    fn apply(&mut self, steps: usize, knee_steps: usize) {
        let p = &mut self.params;
        let d = &mut self.dynamics;
        match self.flavour {
            Flavour::Compressor | Flavour::Expander => {
                d.set_input_gain(p[5].proc_value(steps));
                d.set_output_gain(p[6].proc_value(steps));
                let attack = p[0].proc_value(steps);
                let release = p[1].proc_value(steps);
                let threshold = p[3].proc_value(steps);
                let ratio = p[2].proc_value(steps);
                let knee = p[4].proc_value(steps);
                d.set_params(attack, 0.0, release, threshold, ratio, knee);
            }
            Flavour::Limiter => {
                d.set_input_gain(p[3].proc_value(steps));
                d.set_output_gain(p[4].proc_value(steps));
                let attack = p[0].proc_value(steps);
                let release = p[1].proc_value(steps);
                let threshold = p[2].proc_value(steps);
                d.set_params(attack, 0.0, release, threshold, 1000.0, 0.0);
            }
            Flavour::Gate => {
                d.set_input_gain(p[5].proc_value(steps));
                d.set_output_gain(p[6].proc_value(steps));
                let attack = p[0].proc_value(steps);
                let hold = p[1].proc_value(steps);
                let release = p[2].proc_value(steps);
                let threshold = p[3].proc_value(steps);
                let knee = p[4].proc_value(knee_steps);
                d.set_params(attack, hold, release, threshold, 1000.0, knee);
            }
        }
    }
}

impl Processor for DynamicsProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        let kind = match self.flavour {
            Flavour::Compressor => None,
            Flavour::Limiter => Some(DynamicsType::Limiter),
            Flavour::Gate => Some(DynamicsType::Gate),
            Flavour::Expander => Some(DynamicsType::Expander),
        };
        // The Expander sets its mode before the sample rate and the others
        // after the reset; neither reads it, so one order serves all four.
        if let Some(kind) = kind {
            self.dynamics.set_mode(kind);
        }
        self.dynamics.set_sample_rate(sample_rate);
        self.dynamics.reset();
        self.dynamics.set_num_channels(2);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.dynamics.reset();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        if any_smoothing(&self.params) {
            for i in 0..samples {
                self.apply(1, 1);
                self.dynamics
                    .process(&mut [&mut left[i..=i], &mut right[i..=i]]);
            }
        } else {
            self.apply(samples, 1);
            self.dynamics
                .process(&mut [&mut left[..samples], &mut right[..samples]]);
        }
    }
}

impl DynamicsProcessor {
    /// `process`, the detector listening to `key` — a side-chain's two
    /// channels — rather than to the signal. The knobs are handed on as
    /// `process` hands them.
    pub fn process_keyed(&mut self, left: &mut [f32], right: &mut [f32], key: [&[f32]; 2]) {
        let samples = left.len().min(right.len());
        if key.iter().any(|k| k.len() < samples) {
            self.process(left, right, Host { bpm: 120.0 });
            return;
        }
        if any_smoothing(&self.params) {
            for i in 0..samples {
                self.apply(1, 1);
                self.dynamics.process_keyed(
                    &mut [&mut left[i..=i], &mut right[i..=i]],
                    &[&key[0][i..=i], &key[1][i..=i]],
                );
            }
        } else {
            self.apply(samples, 1);
            self.dynamics.process_keyed(
                &mut [&mut left[..samples], &mut right[..samples]],
                &[&key[0][..samples], &key[1][..samples]],
            );
        }
    }
}
