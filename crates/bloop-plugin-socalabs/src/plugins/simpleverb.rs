//! slPlugins' SimpleVerb: `gin::SimpleVerb` behind seven faders.
//!
//! Ported from `plugins/SimpleVerb/Source/PluginProcessor.cpp`.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::param::{any_smoothing, Conversion, Param, ParamSpec, Smoothing};
use crate::gin::simpleverb::SimpleVerb;
use crate::juce::NormalisableRange as Range;

const UNIT: Range = Range::new(0.0, 1.0, 0.0, 1.0);

const fn fader(name: &'static str, default: f32, smoothing: Smoothing) -> ParamSpec {
    ParamSpec::new(name, "%", UNIT, default, smoothing, Conversion::Identity)
}

const SMOOTH: Smoothing = Smoothing::Linear(0.1);

/// The SimpleVerb's knobs, in the order its constructor adds them. Room and
/// pre-delay are not smoothed: each change of either clears the buffers.
pub const SIMPLEVERB: [ParamSpec; 7] = [
    fader("Room", 0.3, Smoothing::None),
    fader("Damp", 0.6, SMOOTH),
    fader("PreDelay", 0.3, Smoothing::None),
    fader("LP", 0.9, SMOOTH),
    fader("HP", 0.1, SMOOTH),
    fader("Wet", 0.4, SMOOTH),
    fader("Dry", 0.3, SMOOTH),
];

#[derive(Clone, Debug)]
pub struct SimpleVerbProcessor {
    params: Vec<Param>,
    reverb: SimpleVerb,
}

impl Default for SimpleVerbProcessor {
    fn default() -> Self {
        Self {
            params: SIMPLEVERB.iter().copied().map(Param::new).collect(),
            reverb: SimpleVerb::default(),
        }
    }
}

impl SimpleVerbProcessor {
    fn apply(&mut self, steps: usize) {
        let p = &mut self.params;
        let room = p[0].proc_value(steps);
        let damp = p[1].proc_value(steps);
        let pre_delay = p[2].proc_value(steps);
        let lp = p[3].proc_value(steps);
        let hp = p[4].proc_value(steps);
        let wet = p[5].proc_value(steps);
        let dry = p[6].proc_value(steps);
        self.reverb
            .set_parameters(room, damp, pre_delay, lp, hp, wet, dry);
    }
}

impl Processor for SimpleVerbProcessor {
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

    /// Only the knobs: the original leaves the reverb's state alone.
    fn reset(&mut self) {
        reset_params(&mut self.params);
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        if any_smoothing(&self.params) {
            for i in 0..samples {
                self.apply(1);
                self.reverb.process(&mut left[i..=i], &mut right[i..=i]);
            }
        } else {
            self.apply(samples);
            self.reverb
                .process(&mut left[..samples], &mut right[..samples]);
        }
    }
}
