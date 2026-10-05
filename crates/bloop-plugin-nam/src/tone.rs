//! What NAM's plug-in puts after the model (NeuralAmpModelerPlugin's
//! `ToneStack.cpp` and AudioDSPTools' `RecursiveLinearFilter`, both MIT):
//! a three-band tone stack and a 5 Hz DC blocker. In `f64`, as there.

use core::f64::consts::PI;

/// One biquad, Direct Form I, its coefficients normalised by `a0`.
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b: [f64; 3],
    /// `-a1/a0`, `-a2/a0`: added, as AudioDSPTools adds them.
    a: [f64; 2],
    x: [f64; 2],
    y: [f64; 2],
}

/// Which of the Audio EQ Cookbook's shapes.
#[derive(Clone, Copy)]
enum Shape {
    LowShelf,
    Peaking,
    HighShelf,
}

impl Biquad {
    fn set(&mut self, shape: Shape, rate: f64, frequency: f64, quality: f64, gain_db: f64) {
        let a = libm::pow(10.0, gain_db / 40.0);
        let w0 = 2.0 * PI * frequency / rate;
        let alpha = libm::sin(w0) / (2.0 * quality);
        let cosw = libm::cos(w0);
        let (a0, a1, a2, b0, b1, b2) = match shape {
            Shape::LowShelf => {
                let (ap, am, r) = (a + 1.0, a - 1.0, 2.0 * libm::sqrt(a) * alpha);
                (
                    ap + am * cosw + r,
                    -2.0 * (am + ap * cosw),
                    ap + am * cosw - r,
                    a * (ap - am * cosw + r),
                    2.0 * a * (am - ap * cosw),
                    a * (ap - am * cosw - r),
                )
            }
            Shape::Peaking => (
                1.0 + alpha / a,
                -2.0 * cosw,
                1.0 - alpha / a,
                1.0 + alpha * a,
                -2.0 * cosw,
                1.0 - alpha * a,
            ),
            Shape::HighShelf => {
                let (ap, am, r) = (a + 1.0, a - 1.0, 2.0 * libm::sqrt(a) * alpha);
                (
                    ap - am * cosw + r,
                    2.0 * (am - ap * cosw),
                    ap - am * cosw - r,
                    a * (ap + am * cosw + r),
                    -2.0 * a * (am + ap * cosw),
                    a * (ap + am * cosw - r),
                )
            }
        };
        self.b = [b0 / a0, b1 / a0, b2 / a0];
        self.a = [-a1 / a0, -a2 / a0];
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let mut y = self.b[0] * x + self.b[1] * self.x[0] + self.b[2] * self.x[1];
        y += self.a[0] * self.y[0] + self.a[1] * self.y[1];
        // AudioDSPTools: "Prevent a NaN from jamming the filter!"
        if y.is_nan() {
            y = 0.0;
        }
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }

    fn clear(&mut self) {
        self.x = [0.0; 2];
        self.y = [0.0; 2];
    }
}

/// Bass, Middle and Treble, each 0 to 10 with 5 flat, as NAM's plug-in
/// sets them: a low shelf at 150 Hz (±20 dB), a peak at 425 Hz (±15 dB,
/// wider when it cuts) and a high shelf at 1.8 kHz (±10 dB).
#[derive(Clone, Debug)]
pub struct ToneStack {
    rate: f64,
    knobs: [f64; 3],
    bands: [Biquad; 3],
}

impl ToneStack {
    #[must_use]
    pub fn new(rate: f64) -> Self {
        let mut stack = Self {
            rate,
            knobs: [5.0; 3],
            bands: [Biquad::default(); 3],
        };
        for band in 0..3 {
            stack.set(band, 5.0);
        }
        stack
    }

    /// Knob `band` (0 bass, 1 middle, 2 treble) to `value`, 0 to 10.
    pub fn set(&mut self, band: usize, value: f64) {
        let rate = self.rate;
        let Some(filter) = self.bands.get_mut(band) else {
            return;
        };
        self.knobs[band] = value;
        match band {
            0 => filter.set(Shape::LowShelf, rate, 150.0, 0.707, 4.0 * (value - 5.0)),
            1 => {
                let gain = 3.0 * (value - 5.0);
                let quality = if gain < 0.0 { 1.5 } else { 0.7 };
                filter.set(Shape::Peaking, rate, 425.0, quality, gain);
            }
            _ => filter.set(Shape::HighShelf, rate, 1800.0, 0.707, 2.0 * (value - 5.0)),
        }
    }

    /// Prepared for `rate`, its knobs kept.
    pub fn set_rate(&mut self, rate: f64) {
        self.rate = rate;
        for band in 0..3 {
            self.set(band, self.knobs[band]);
        }
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let x = self.bands[0].process(x);
        let x = self.bands[1].process(x);
        self.bands[2].process(x)
    }

    pub fn clear(&mut self) {
        for band in &mut self.bands {
            band.clear();
        }
    }
}

/// The one-pole high-pass NAM's plug-in ends with, against a model's DC
/// offset: `y[n] = α·y[n-1] + α·(x[n] - x[n-1])`, at 5 Hz.
#[derive(Clone, Copy, Debug)]
pub struct DcBlocker {
    alpha: f64,
    x: f64,
    y: f64,
}

impl DcBlocker {
    /// The frequency NAM's plug-in blocks below.
    pub const FREQUENCY: f64 = 5.0;

    #[must_use]
    pub fn new(rate: f64) -> Self {
        let c = 2.0 * PI * Self::FREQUENCY / rate;
        Self {
            alpha: 1.0 / (c + 1.0),
            x: 0.0,
            y: 0.0,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let mut y = self.alpha * x - self.alpha * self.x + self.alpha * self.y;
        if y.is_nan() {
            y = 0.0;
        }
        self.x = x;
        self.y = y;
        y
    }

    pub fn clear(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}
