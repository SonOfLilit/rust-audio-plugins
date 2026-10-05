//! `juce::NormalisableRange<float>` as a Wavetable parameter uses it: a span
//! with a step, a power-law skew, and for the LFOs' Fade a skew mirrored
//! about the middle.
//!
//! Written from JUCE's documented behaviour; the gold tests link the real
//! one.

use bloop_plugin_socalabs::shared::math;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    pub start: f32,
    pub end: f32,
    pub interval: f32,
    pub skew: f32,
    /// The skew bends both halves away from the middle rather than the
    /// whole span away from `start`.
    pub symmetric: bool,
}

impl Range {
    #[must_use]
    pub const fn new(start: f32, end: f32, interval: f32, skew: f32) -> Self {
        Self {
            start,
            end,
            interval,
            skew,
            symmetric: false,
        }
    }

    #[must_use]
    pub const fn symmetric(start: f32, end: f32, interval: f32, skew: f32) -> Self {
        Self {
            start,
            end,
            interval,
            skew,
            symmetric: true,
        }
    }

    /// `convertTo0to1`.
    #[must_use]
    pub fn to_0to1(&self, value: f32) -> f32 {
        let proportion = clamp01((value - self.start) / (self.end - self.start));
        if self.skew == 1.0 {
            return proportion;
        }
        if !self.symmetric {
            return math::powf(proportion, self.skew);
        }
        let distance = 2.0 * proportion - 1.0;
        let sign = if distance < 0.0 { -1.0 } else { 1.0 };
        (1.0 + math::powf(distance.abs(), self.skew) * sign) / 2.0
    }

    /// `convertFrom0to1`.
    #[must_use]
    pub fn from_0to1(&self, proportion: f32) -> f32 {
        let mut proportion = clamp01(proportion);
        if !self.symmetric {
            if self.skew != 1.0 && proportion > 0.0 {
                proportion = math::expf(math::logf(proportion) / self.skew);
            }
            return self.start + (self.end - self.start) * proportion;
        }
        let mut distance = 2.0 * proportion - 1.0;
        if self.skew != 1.0 && distance != 0.0 {
            let sign = if distance < 0.0 { -1.0 } else { 1.0 };
            distance = math::expf(math::logf(distance.abs()) / self.skew) * sign;
        }
        self.start + (self.end - self.start) / 2.0 * (1.0 + distance)
    }

    /// `snapToLegalValue`: onto the nearest step, and into the span.
    #[must_use]
    pub fn snap(&self, value: f32) -> f32 {
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

    /// `jlimit (start, end, v)`, as a parameter clamps what it stores.
    #[must_use]
    pub fn clamp(&self, value: f32) -> f32 {
        bloop_plugin_socalabs::shared::jlimit(self.start, self.end, value)
    }
}

fn clamp01(value: f32) -> f32 {
    bloop_plugin_socalabs::shared::jlimit(0.0, 1.0, value)
}
