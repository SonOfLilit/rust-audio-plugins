//! A complex FFT over a power-of-two size, for band-limiting wavetables.
//!
//! The original calls `juce::dsp::FFT::perform`, forward unscaled and
//! inverse scaled by `1 / size`. That is JUCE's code, which this crate
//! cannot carry, so this is an iterative radix-2 transform of bloop's own
//! with the same scaling. Its rounding is not JUCE's, which is why a
//! wavetable's band-limited frames match the original's only to within
//! a few ulps (`tests/gold.rs` measures how far); everything downstream
//! of a frame is held bit for bit.
//!
//! Twiddles come from `libm` in double, so the transform rounds the same
//! way on every target.

/// A complex number, `juce::dsp::Complex<float>`'s layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

/// A transform of one size, its twiddles worked out once.
pub struct Fft {
    size: usize,
    twiddles: Vec<Complex>,
}

impl Fft {
    /// A transform of `2^order` points.
    #[must_use]
    pub fn new(order: u32) -> Self {
        let size = 1usize << order;
        let twiddles = (0..size / 2)
            .map(|k| {
                let angle = -2.0 * core::f64::consts::PI * k as f64 / size as f64;
                Complex {
                    re: libm::cos(angle) as f32,
                    im: libm::sin(angle) as f32,
                }
            })
            .collect();
        Self { size, twiddles }
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// `input` transformed into `output`, both `size` long; the inverse is
    /// scaled by `1 / size`.
    pub fn perform(&self, input: &[Complex], output: &mut [Complex], inverse: bool) {
        let n = self.size;
        debug_assert!(input.len() >= n && output.len() >= n);
        let bits = n.trailing_zeros();
        for (i, x) in input.iter().take(n).enumerate() {
            let j = if bits == 0 {
                0
            } else {
                i.reverse_bits() >> (usize::BITS - bits)
            };
            output[j] = *x;
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let w = self.twiddles[k * stride];
                    let w = if inverse {
                        Complex { re: w.re, im: -w.im }
                    } else {
                        w
                    };
                    let a = output[start + k];
                    let b = output[start + k + half];
                    let t = Complex {
                        re: b.re * w.re - b.im * w.im,
                        im: b.re * w.im + b.im * w.re,
                    };
                    output[start + k] = Complex {
                        re: a.re + t.re,
                        im: a.im + t.im,
                    };
                    output[start + k + half] = Complex {
                        re: a.re - t.re,
                        im: a.im - t.im,
                    };
                }
            }
            len *= 2;
        }
        if inverse {
            let scale = 1.0f32 / n as f32;
            for x in output.iter_mut().take(n) {
                x.re *= scale;
                x.im *= scale;
            }
        }
    }
}
