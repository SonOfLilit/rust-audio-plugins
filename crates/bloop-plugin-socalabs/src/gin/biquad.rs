//! `gin::Biquad`: one RBJ cookbook section, a sample at a time.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_biquad.h`: the lowpass and
//! highpass the TransientShaper uses.

use crate::math;
use core::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Default for Biquad {
    fn default() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
}

/// `w0`, `alpha` and `a0`.
fn common(freq: f64, q: f64, sample_rate: f64) -> (f64, f64, f64) {
    let w0 = 2.0 * PI * freq / sample_rate;
    let alpha = math::sin(w0) / (2.0 * q);
    (w0, alpha, 1.0 + alpha)
}

impl Biquad {
    fn set(&mut self, b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) {
        self.b0 = b0 as f32;
        self.b1 = b1 as f32;
        self.b2 = b2 as f32;
        self.a1 = a1 as f32;
        self.a2 = a2 as f32;
    }

    pub fn set_lowpass(&mut self, freq: f64, q: f64, sample_rate: f64) {
        let (w0, alpha, a0) = common(freq, q, sample_rate);
        let cosw0 = math::cos(w0);
        self.set(
            ((1.0 - cosw0) / 2.0) / a0,
            (1.0 - cosw0) / a0,
            ((1.0 - cosw0) / 2.0) / a0,
            (-2.0 * cosw0) / a0,
            (1.0 - alpha) / a0,
        );
    }

    pub fn set_highpass(&mut self, freq: f64, q: f64, sample_rate: f64) {
        let (w0, alpha, a0) = common(freq, q, sample_rate);
        let cosw0 = math::cos(w0);
        self.set(
            ((1.0 + cosw0) / 2.0) / a0,
            (-(1.0 + cosw0)) / a0,
            ((1.0 + cosw0) / 2.0) / a0,
            (-2.0 * cosw0) / a0,
            (1.0 - alpha) / a0,
        );
    }

    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}
