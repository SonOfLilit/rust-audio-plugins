//! slPlugins' Delay: `gin::StereoDelay`, its time free or synced to the
//! tempo.
//!
//! Ported from `plugins/Delay/Source/PluginProcessor.cpp`.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::delay::StereoDelay;
use crate::gin::noteduration::DURATIONS;
use crate::gin::param::{any_smoothing, Conversion, Param, ParamSpec, Smoothing};
use crate::juce::NormalisableRange as Range;

const SMOOTH: Smoothing = Smoothing::Linear(0.1);

/// The Delay's knobs, in the order its constructor adds them.
pub const DELAY: [ParamSpec; 6] = [
    ParamSpec::new(
        "Sync",
        "",
        Range::new(0.0, 1.0, 1.0, 1.0),
        0.0,
        Smoothing::None,
        Conversion::Identity,
    ),
    ParamSpec::new(
        "Delay",
        "s",
        Range::new(0.0, 120.0, 0.0, 0.3),
        1.0,
        Smoothing::None,
        Conversion::Identity,
    ),
    ParamSpec::new(
        "Beat",
        "",
        Range::new(0.0, (DURATIONS.len() - 1) as f32, 1.0, 1.0),
        13.0,
        Smoothing::None,
        Conversion::Identity,
    ),
    ParamSpec::new(
        "Feedback",
        "dB",
        Range::new(-100.0, 0.0, 0.0, 5.0),
        -10.0,
        SMOOTH,
        Conversion::DecibelsToGain,
    ),
    ParamSpec::new(
        "Crossfeed",
        "dB",
        Range::new(-100.0, 0.0, 0.0, 5.0),
        -100.0,
        SMOOTH,
        Conversion::DecibelsToGain,
    ),
    ParamSpec::new(
        "Mix",
        "%",
        Range::new(0.0, 100.0, 0.0, 1.0),
        0.0,
        SMOOTH,
        Conversion::PercentToFraction,
    ),
];

/// The delay the other knobs resolve to: an internal parameter, eased.
const DELAY_TIME: ParamSpec = ParamSpec::new(
    "Delay",
    "s",
    Range::new(0.0, 120.0, 0.0, 1.0),
    1.0,
    Smoothing::Eased(0.2),
    Conversion::Identity,
);

const SYNC: usize = 0;
const TIME: usize = 1;
const BEAT: usize = 2;
const FEEDBACK: usize = 3;
const CROSSFEED: usize = 4;
const MIX: usize = 5;
/// Where the internal delay time sits after the six knobs.
const DELAY_PARAM: usize = 6;

#[derive(Clone, Debug)]
pub struct DelayProcessor {
    /// The six knobs, then the internal delay time: `allParameters` order,
    /// which is the order smoothing is checked in.
    params: Vec<Param>,
    delay: StereoDelay,
}

impl Default for DelayProcessor {
    fn default() -> Self {
        let mut params: Vec<Param> = DELAY.iter().copied().map(Param::new).collect();
        params.push(Param::new(DELAY_TIME));
        Self {
            params,
            delay: StereoDelay::new(5.1),
        }
    }
}

impl DelayProcessor {
    /// `updateInternalParams`: the delay time from the free time or from
    /// the note the Beat knob names at `bpm`.
    fn update_internal_params(&mut self, bpm: f32) {
        let seconds = if self.params[SYNC].is_on() {
            let beat = self.params[BEAT].user_value_int() as usize;
            DURATIONS[beat.min(DURATIONS.len() - 1)].to_seconds(bpm)
        } else {
            self.params[TIME].user_value()
        };
        self.params[DELAY_PARAM].set_user_value(seconds);
    }

    fn apply(&mut self, steps: usize) {
        let delay = self.params[DELAY_PARAM].proc_value(steps);
        let mix = self.params[MIX].proc_value(steps);
        let feedback = self.params[FEEDBACK].proc_value(steps);
        let crossfeed = self.params[CROSSFEED].proc_value(steps);
        self.delay.set_params(delay, mix, feedback, crossfeed);
    }

    fn process_block(&mut self, left: &mut [f32], right: &mut [f32], host: Host) {
        self.update_internal_params(host.bpm);
        let samples = left.len().min(right.len());
        if any_smoothing(&self.params) {
            for i in 0..samples {
                self.apply(1);
                self.delay.process(&mut left[i..=i], &mut right[i..=i]);
            }
        } else {
            self.apply(samples);
            self.delay
                .process(&mut left[..samples], &mut right[..samples]);
        }
    }
}

impl Processor for DelayProcessor {
    /// The knobs; the internal delay time is not one.
    fn params(&self) -> &[Param] {
        &self.params[..DELAY_PARAM]
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params[..DELAY_PARAM]
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.delay.set_sample_rate(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.delay.reset();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], host: Host) {
        self.process_block(left, right, host);
    }
}
