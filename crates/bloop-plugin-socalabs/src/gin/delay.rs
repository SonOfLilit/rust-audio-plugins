//! `gin::DelayLine`, `gin::Lagrange` and `gin::StereoDelay`.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_delayline.h`, `gin_stereodelay.h`
//! and `gin/utilities/gin_lagrange.h`.

use super::audioutil::WetDryMix;
use crate::juce::approximately_equal;
use crate::math;

/// `gin::Lagrange::interpolate (xArr, yArr, num, x)`: the polynomial
/// through the points, at `x`.
#[must_use]
pub fn lagrange(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    let mut res = 0.0f32;
    for i in 0..xs.len() {
        let mut term = ys[i];
        for j in 0..xs.len() {
            if i != j {
                let d = xs[i] - xs[j];
                if approximately_equal(d, 0.0) {
                    term = 0.0;
                } else {
                    term = term * (x - xs[j]) / d;
                }
            }
        }
        res += term;
    }
    res
}

/// `gin::DelayLine`: a ring per channel, read back at a fractional delay.
///
/// Sized when it is made, on the UI thread; nothing after that allocates.
#[derive(Clone, Debug)]
pub struct DelayLine {
    channels: Vec<Vec<f32>>,
    sample_rate: f64,
    write_pos: usize,
}

impl DelayLine {
    #[must_use]
    pub fn new(channels: usize, maximum_delay: f64, sample_rate: f64) -> Self {
        let mut line = Self {
            channels: Vec::new(),
            sample_rate: 44100.0,
            write_pos: 0,
        };
        line.set_size(channels, maximum_delay, sample_rate);
        line
    }

    /// `setSize`. Allocates.
    pub fn set_size(&mut self, channels: usize, maximum_delay: f64, sample_rate: f64) {
        self.sample_rate = if sample_rate > 0.0 {
            sample_rate
        } else {
            44100.0
        };
        let samples = ((maximum_delay * self.sample_rate).ceil() as i32).max(1) as usize;
        self.channels = vec![vec![0.0; samples]; channels.max(1)];
        self.write_pos = 0;
    }

    /// Whether this line already is what `set_size` would make it, so a
    /// caller on the audio thread can clear it rather than rebuild it.
    #[must_use]
    pub fn has_size(&self, channels: usize, maximum_delay: f64, sample_rate: f64) -> bool {
        let rate = if sample_rate > 0.0 {
            sample_rate
        } else {
            44100.0
        };
        let samples = ((maximum_delay * rate).ceil() as i32).max(1) as usize;
        self.sample_rate == rate
            && self.channels.len() == channels.max(1)
            && self.channels.first().map_or(0, Vec::len) == samples
    }

    pub fn clear(&mut self) {
        self.write_pos = 0;
        for c in &mut self.channels {
            c.fill(0.0);
        }
    }

    fn len(&self) -> usize {
        self.channels.first().map_or(1, Vec::len)
    }

    /// Where a read `t` seconds back lands, as a fractional sample index.
    fn read_pos(&self, t: f64) -> f32 {
        let n = self.len() as i32;
        let t = t.clamp(0.0, f64::from(n - 1) / self.sample_rate);
        ((f64::from(self.write_pos as i32 - 1 + n) - t * self.sample_rate) as f32) % (n as f32)
    }

    /// `readLagrange`: four-point Lagrange interpolation, `t` seconds back.
    #[must_use]
    pub fn read_lagrange(&self, channel: usize, t: f64) -> f32 {
        let n = self.len();
        let read_pos = self.read_pos(t);
        let i1 = read_pos.floor() as usize;
        let data = &self.channels[channel];
        let fraction = read_pos - i1 as f32;
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [
            data[i1],
            data[(i1 + 1) % n],
            data[(i1 + 2) % n],
            data[(i1 + 3) % n],
        ];
        lagrange(&xs, &ys, fraction)
    }

    pub fn write(&mut self, channel: usize, input: f32) {
        let pos = self.write_pos;
        self.channels[channel][pos] = input;
    }

    pub fn write_finished(&mut self) {
        self.write_pos += 1;
        if self.write_pos >= self.len() {
            self.write_pos = 0;
        }
    }
}

/// `gin::StereoDelay`, as the Delay plug-in drives it.
///
/// The original keeps its four settings in `juce::SmoothedValue`s, but the
/// `process` the plug-in calls reads only their targets — it smooths the
/// knobs itself, a sample at a time — so the targets are all there is to
/// port. For the same reason the wet/dry pass is folded into the delay
/// pass: the mix is one value per call either way.
#[derive(Clone, Debug)]
pub struct StereoDelay {
    line: DelayLine,
    max_delay: f64,
    delay: f32,
    mix: f32,
    feedback: f32,
    crossfeed: f32,
}

impl StereoDelay {
    #[must_use]
    pub fn new(max_delay: f64) -> Self {
        Self {
            line: DelayLine::new(2, max_delay, 44100.0),
            max_delay,
            delay: 0.0,
            mix: 0.0,
            feedback: 0.0,
            crossfeed: 0.0,
        }
    }

    /// `setSampleRate`. Resizes the line, which allocates, unless it is
    /// already the size this rate needs.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if self.line.has_size(2, self.max_delay, sample_rate) {
            self.line.clear();
        } else {
            self.line.set_size(2, self.max_delay, sample_rate);
        }
    }

    /// Each setting moves only if it moves by more than float rounding:
    /// `SmoothedValue::setTargetValue` ignores a target approximately
    /// equal to the one it has, and a knob smoothing a sample at a time
    /// asks for exactly such steps.
    pub fn set_params(&mut self, delay: f32, mix: f32, feedback: f32, crossfeed: f32) {
        let retarget = |target: &mut f32, value: f32| {
            if !approximately_equal(value, *target) {
                *target = value;
            }
        };
        retarget(&mut self.mix, mix);
        retarget(&mut self.delay, delay);
        retarget(&mut self.feedback, math::std_min(0.99, feedback));
        retarget(&mut self.crossfeed, math::std_min(0.99, crossfeed));
    }

    pub fn reset(&mut self) {
        self.line.clear();
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let wet_dry = WetDryMix::new(self.mix);
        let delay = f64::from(self.delay);
        for (out_l, out_r) in left.iter_mut().zip(right.iter_mut()) {
            let (in_l, in_r) = (*out_l, *out_r);
            let l = self.line.read_lagrange(0, delay);
            let r = self.line.read_lagrange(1, delay);
            self.line
                .write(0, in_l + (self.feedback * l) + (self.crossfeed * r));
            self.line
                .write(1, in_r + (self.feedback * r) + (self.crossfeed * l));
            self.line.write_finished();
            *out_l = (wet_dry.wet_gain * l) + (wet_dry.dry_gain * in_l);
            *out_r = (wet_dry.wet_gain * r) + (wet_dry.dry_gain * in_r);
        }
    }
}
