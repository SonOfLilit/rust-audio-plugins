//! slPlugins' BitCrusher: `gin::BitCrusher` behind three knobs.
//!
//! Ported from `plugins/BitCrusher/Source/PluginProcessor.cpp`. The
//! original reports the crusher's latency to its host; bloop's Plug-ins
//! have no latency to report, and nothing here needs it.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::bitcrusher::BitCrusher;
use crate::gin::param::{Conversion, Param, ParamSpec, Smoothing};
use crate::juce::NormalisableRange as Range;

const SMOOTH: Smoothing = Smoothing::Linear(0.1);

/// The BitCrusher's knobs, in the order its constructor adds them.
pub const BITCRUSHER: [ParamSpec; 3] = [
    ParamSpec::new(
        "Bits",
        "",
        Range::new(1.0, 16.0, 0.0, 1.0),
        16.0,
        SMOOTH,
        Conversion::Identity,
    ),
    ParamSpec::new(
        "Rate",
        "Hz",
        Range::new(500.0, 48000.0, 0.0, 0.3),
        48000.0,
        SMOOTH,
        Conversion::Identity,
    ),
    ParamSpec::new(
        "Dirty",
        "",
        Range::new(0.0, 1.0, 1.0, 1.0),
        0.0,
        Smoothing::None,
        Conversion::Identity,
    ),
];

#[derive(Clone, Debug)]
pub struct BitCrusherProcessor {
    params: Vec<Param>,
    crusher: BitCrusher,
}

impl Default for BitCrusherProcessor {
    fn default() -> Self {
        Self {
            params: BITCRUSHER.iter().copied().map(Param::new).collect(),
            crusher: BitCrusher::default(),
        }
    }
}

impl Processor for BitCrusherProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.crusher.set_sample_rate(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.crusher.reset();
    }

    /// The knobs are read once a block, smoothing or not: the crusher
    /// smooths its own.
    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        let bits = self.params[0].proc_value(samples);
        let rate = self.params[1].proc_value(samples);
        let dirty = self.params[2].proc_value(samples) > 0.5;
        self.crusher.set_params(bits, rate, dirty);
        self.crusher
            .process(&mut left[..samples], &mut right[..samples]);
    }
}
