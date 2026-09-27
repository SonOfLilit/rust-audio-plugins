//! The few JUCE helpers the originals lean on, written from JUCE's
//! documented behaviour rather than ported from its source: JUCE is not
//! under a licence this crate could carry. The gold tests link the real
//! JUCE, so they hold every one of these to the original.

use crate::math;

/// `juce::Decibels`'s floor: anything at or below it is silence.
const MINUS_INFINITY_DB: f32 = -100.0;

/// `juce::Decibels::decibelsToGain`, float, default floor.
#[must_use]
pub fn decibels_to_gain(decibels: f32) -> f32 {
    if decibels > MINUS_INFINITY_DB {
        math::powf(10.0, decibels * 0.05)
    } else {
        0.0
    }
}

/// `juce::Decibels::gainToDecibels`, float, default floor.
#[must_use]
pub fn gain_to_decibels(gain: f32) -> f32 {
    if gain > 0.0 {
        MINUS_INFINITY_DB.max(math::log10f(gain) * 20.0)
    } else {
        MINUS_INFINITY_DB
    }
}

/// `juce::approximatelyEqual` for floats, at its default tolerance: equal
/// within the smallest normal float, or within one epsilon of the larger.
#[must_use]
pub fn approximately_equal(a: f32, b: f32) -> bool {
    if !(a.is_finite() && b.is_finite()) {
        return a == b;
    }
    let diff = (a - b).abs();
    diff <= f32::MIN_POSITIVE || diff <= f32::EPSILON * a.abs().max(b.abs())
}

/// `juce::jlimit`: `value` held to `[lo, hi]`, compared the way JUCE does
/// (so a NaN passes through rather than being clamped).
#[must_use]
pub fn jlimit(lo: f32, hi: f32, value: f32) -> f32 {
    if value < lo {
        lo
    } else if hi < value {
        hi
    } else {
        value
    }
}

/// `juce::NormalisableRange<float>` as a parameter uses it: a span with an
/// optional step and a power-law skew, and no custom mapping functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalisableRange {
    pub start: f32,
    pub end: f32,
    pub interval: f32,
    pub skew: f32,
}

impl NormalisableRange {
    #[must_use]
    pub const fn new(start: f32, end: f32, interval: f32, skew: f32) -> Self {
        Self {
            start,
            end,
            interval,
            skew,
        }
    }

    /// Where `value` sits in the range, 0 to 1, after the skew.
    #[must_use]
    pub fn convert_to_0to1(&self, value: f32) -> f32 {
        let proportion = ((value - self.start) / (self.end - self.start)).clamp(0.0, 1.0);
        if self.skew == 1.0 {
            proportion
        } else {
            math::powf(proportion, self.skew)
        }
    }

    /// The value at `proportion` of the range, undoing the skew.
    #[must_use]
    pub fn convert_from_0to1(&self, proportion: f32) -> f32 {
        let mut proportion = proportion.clamp(0.0, 1.0);
        if self.skew != 1.0 && proportion > 0.0 {
            proportion = math::expf(math::logf(proportion) / self.skew);
        }
        self.start + (self.end - self.start) * proportion
    }

    /// `value` moved to the nearest step, and into the range.
    #[must_use]
    pub fn snap_to_legal_value(&self, value: f32) -> f32 {
        let mut v = value;
        if self.interval > 0.0 {
            v = self.start + self.interval * ((v - self.start) / self.interval + 0.5).floor();
        }
        if v <= self.start || self.end <= self.start {
            self.start
        } else if v >= self.end {
            self.end
        } else {
            v
        }
    }
}

/// `juce::LinearSmoothedValue<float>`: a ramp of a fixed number of steps
/// to each new target.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinearSmoothedValue {
    current: f32,
    target: f32,
    countdown: i32,
    steps_to_target: i32,
    step: f32,
}

impl LinearSmoothedValue {
    /// `reset (sampleRate, rampLengthInSeconds)`: ramps that long from now
    /// on, and no ramp in progress.
    pub fn reset(&mut self, sample_rate: f64, ramp_seconds: f64) {
        self.steps_to_target = (ramp_seconds * sample_rate).floor() as i32;
        self.set_current_and_target(self.target);
    }

    pub fn set_current_and_target(&mut self, value: f32) {
        self.target = value;
        self.current = value;
        self.countdown = 0;
    }

    pub fn set_target(&mut self, value: f32) {
        if approximately_equal(value, self.target) {
            return;
        }
        if self.steps_to_target <= 0 {
            self.set_current_and_target(value);
            return;
        }
        self.target = value;
        self.countdown = self.steps_to_target;
        self.step = (self.target - self.current) / self.countdown as f32;
    }

    #[must_use]
    pub fn is_smoothing(&self) -> bool {
        self.countdown > 0
    }

    #[must_use]
    pub fn target(&self) -> f32 {
        self.target
    }

    pub fn next_value(&mut self) -> f32 {
        if !self.is_smoothing() {
            return self.target;
        }
        self.countdown -= 1;
        if self.is_smoothing() {
            self.current += self.step;
        } else {
            self.current = self.target;
        }
        self.current
    }
}
