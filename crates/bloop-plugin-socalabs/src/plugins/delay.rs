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

/// The Delay's knobs, in the order its constructor adds them. The original
/// calls both the free time and the synced note "Delay"; here they are
/// Time and Beat.
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
        "Time",
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

/// How long the original's line is, in seconds: `stereoDelay { 120.1 }`,
/// enough for the Time knob's two minutes. At 48kHz that is 46 MB of line
/// per Delay, allocated when the Delay is made — which is why a host can
/// build one with less room ([`DelayProcessor::with_room`]) and build it
/// again with more when the time outgrows it.
pub const MAX_DELAY: f64 = 120.1;

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
        Self::with_room(MAX_DELAY)
    }
}

/// The delay a Delay's knobs ask for, in seconds: the Time knob's, or with
/// Sync on the note the Beat knob names, at `bpm`.
#[must_use]
pub fn delay_seconds(sync: bool, time: f32, beat: f32, bpm: f32) -> f32 {
    if sync {
        let beat = beat.round().clamp(0.0, (DURATIONS.len() - 1) as f32) as usize;
        DURATIONS[beat].to_seconds(bpm)
    } else {
        time
    }
}

impl DelayProcessor {
    /// A Delay whose line holds `room` seconds. The algorithm is the
    /// original's at any room: what the room changes is how far back it
    /// can reach, a time past it reading from the far end, and — the line
    /// is indexed in `f32` — how finely a fractional delay is resolved,
    /// which only gets better with less.
    #[must_use]
    pub fn with_room(room: f64) -> Self {
        let mut params: Vec<Param> = DELAY.iter().copied().map(Param::new).collect();
        params.push(Param::new(DELAY_TIME));
        Self {
            params,
            delay: StereoDelay::new(room),
        }
    }

    /// `updateInternalParams`: the delay time from the free time or from
    /// the note the Beat knob names at `bpm`.
    fn update_internal_params(&mut self, bpm: f32) {
        let seconds = delay_seconds(
            self.params[SYNC].is_on(),
            self.params[TIME].user_value(),
            self.params[BEAT].user_value_int() as f32,
            bpm,
        );
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
