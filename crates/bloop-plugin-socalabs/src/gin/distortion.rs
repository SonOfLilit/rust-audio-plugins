//! `gin::AirWindowsDistortion`: Chris Johnson's Airwindows Density (MIT;
//! see LICENSE) — sine waveshaping, from starved to many passes of boost.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_distortion.h`, with two things the
//! original leaves to its platform pinned down. It works in `long double`,
//! which is 80 bits on x86 Linux and 64 on arm64 and Windows; here it is
//! 64. And its dither draws from the C library's `rand()`; here that is
//! glibc's generator, seeded as glibc seeds it, so a run means the same
//! thing everywhere. Its denormal guard keeps a counter per channel in a
//! function-level static shared by every instance; here each instance has
//! its own.

use crate::math;

/// glibc's `rand()`: the additive feedback generator `random()` uses, with
/// its default seed of 1.
#[derive(Clone, Debug)]
pub struct GlibcRand {
    state: [u32; 31],
    front: usize,
    rear: usize,
}

impl GlibcRand {
    /// `srand (seed)`.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 31];
        let mut word = i64::from(seed as i32);
        state[0] = word as u32;
        for s in state.iter_mut().skip(1) {
            let hi = word / 127_773;
            let lo = word % 127_773;
            word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            *s = word as u32;
        }
        let mut rand = Self {
            state,
            front: 3,
            rear: 0,
        };
        for _ in 0..310 {
            rand.next();
        }
        rand
    }

    /// `rand()`.
    pub fn next(&mut self) -> i32 {
        self.state[self.front] = self.state[self.front].wrapping_add(self.state[self.rear]);
        let result = (self.state[self.front] >> 1) as i32;
        self.front = (self.front + 1) % 31;
        self.rear = (self.rear + 1) % 31;
        result
    }
}

const RAND_MAX: i32 = 2_147_483_647;
/// Airwindows' own half pi, which is not quite `FRAC_PI_2`.
#[expect(
    clippy::approx_constant,
    reason = "the original's literal: FRAC_PI_2 would move every output"
)]
const HALF_PI: f64 = 1.570_796_33;

/// The denormal guard: a tiny positive number from a hashed counter, in
/// place of digital silence.
///
/// The original squares `int`s past their range, which C leaves undefined;
/// GCC, reasoning that a square cannot be negative, does it all unsigned,
/// and so does this. That is also the arithmetic the original's comment
/// describes: noise that is "actually all positive".
fn residue(counter: &mut u32) -> f64 {
    *counter %= 1_700_021;
    *counter += 1;
    let mut residue = counter.wrapping_mul(*counter);
    for modulus in [170_003, 17011, 1709, 173, 17] {
        residue %= modulus;
        if modulus != 17 {
            residue = residue.wrapping_mul(residue);
        }
    }
    let mut apply = f64::from(residue);
    apply *= 0.000_000_01;
    apply *= 0.000_000_01;
    apply
}

/// One pass of the sine shaper.
fn shape(x: f64) -> f64 {
    let mut bridge = x.abs() * HALF_PI;
    if bridge > HALF_PI {
        bridge = HALF_PI;
    }
    bridge
}

#[derive(Clone, Debug)]
pub struct AirWindowsDistortion {
    sample_rate: f64,
    fp_nshape_l: f64,
    fp_nshape_r: f64,
    iir_al: f64,
    iir_bl: f64,
    iir_ar: f64,
    iir_br: f64,
    fp_flip: bool,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    noise_l: u32,
    noise_r: u32,
    rand: GlibcRand,
}

impl Default for AirWindowsDistortion {
    fn default() -> Self {
        let mut d = Self {
            sample_rate: 44100.0,
            fp_nshape_l: 0.0,
            fp_nshape_r: 0.0,
            iir_al: 0.0,
            iir_bl: 0.0,
            iir_ar: 0.0,
            iir_br: 0.0,
            fp_flip: true,
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            noise_l: 0,
            noise_r: 0,
            rand: GlibcRand::new(1),
        };
        d.reset();
        d
    }
}

/// The dither the original adds: a fraction of the output's own last bit,
/// noise-shaped against the last one added.
fn dither(rand: &mut GlibcRand, x: f64) -> f64 {
    let (_, expon) = libm::frexpf(x as f32);
    (f64::from(rand.next()) / (f64::from(RAND_MAX) * 7.737_125_245_533_627e25))
        * math::pow(2.0, f64::from(expon + 62))
}

impl AirWindowsDistortion {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn reset(&mut self) {
        self.a = 0.2;
        self.b = 0.0;
        self.c = 1.0;
        self.d = 1.0;
        self.iir_al = 0.0;
        self.iir_bl = 0.0;
        self.iir_ar = 0.0;
        self.iir_br = 0.0;
        self.fp_flip = true;
        self.fp_nshape_l = 0.0;
        self.fp_nshape_r = 0.0;
    }

    pub fn set_params(&mut self, density: f32, highpass: f32, output: f32, mix: f32) {
        self.a = density;
        self.b = highpass;
        self.c = output;
        self.d = mix;
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let mut overallscale = 1.0;
        overallscale /= 44100.0;
        overallscale *= self.sample_rate;
        let mut density = (f64::from(self.a) * 5.0) - 1.0;
        let iir_amount = math::pow(f64::from(self.b), 3.0) / overallscale;
        let output = f64::from(self.c);
        let wet = f64::from(self.d);
        let dry = 1.0 - wet;
        let mut out = density.abs();
        density *= density.abs();

        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mut in_l = f64::from(*l);
            let mut in_r = f64::from(*r);
            if in_l < 1.2e-38 && -in_l < 1.2e-38 {
                in_l = residue(&mut self.noise_l);
            }
            if in_r < 1.2e-38 && -in_r < 1.2e-38 {
                in_r = residue(&mut self.noise_r);
            }
            let dry_l = in_l;
            let dry_r = in_r;

            if self.fp_flip {
                self.iir_al = (self.iir_al * (1.0 - iir_amount)) + (in_l * iir_amount);
                in_l -= self.iir_al;
                self.iir_ar = (self.iir_ar * (1.0 - iir_amount)) + (in_r * iir_amount);
                in_r -= self.iir_ar;
            } else {
                self.iir_bl = (self.iir_bl * (1.0 - iir_amount)) + (in_l * iir_amount);
                in_l -= self.iir_bl;
                self.iir_br = (self.iir_br * (1.0 - iir_amount)) + (in_r * iir_amount);
                in_r -= self.iir_br;
            }
            self.fp_flip = !self.fp_flip;

            let mut count = density;
            while count > 1.0 {
                let bridge = math::sin(shape(in_l));
                in_l = if in_l > 0.0 { bridge } else { -bridge };
                let bridge = math::sin(shape(in_r));
                in_r = if in_r > 0.0 { bridge } else { -bridge };
                count -= 1.0;
            }

            while out > 1.0 {
                out -= 1.0;
            }

            let starve = |x: f64| {
                let bridge = shape(x);
                if density > 0.0 {
                    math::sin(bridge)
                } else {
                    1.0 - math::cos(bridge)
                }
            };
            let bridge = starve(in_l);
            in_l = if in_l > 0.0 {
                (in_l * (1.0 - out)) + (bridge * out)
            } else {
                (in_l * (1.0 - out)) - (bridge * out)
            };
            let bridge = starve(in_r);
            in_r = if in_r > 0.0 {
                (in_r * (1.0 - out)) + (bridge * out)
            } else {
                (in_r * (1.0 - out)) - (bridge * out)
            };

            if output < 1.0 {
                in_l *= output;
                in_r *= output;
            }
            if wet < 1.0 {
                in_l = (dry_l * dry) + (in_l * wet);
                in_r = (dry_r * dry) + (in_r * wet);
            }

            let dither_l = dither(&mut self.rand, in_l);
            in_l += dither_l - self.fp_nshape_l;
            self.fp_nshape_l = dither_l;
            let dither_r = dither(&mut self.rand, in_r);
            in_r += dither_r - self.fp_nshape_r;
            self.fp_nshape_r = dither_r;

            *l = in_l as f32;
            *r = in_r as f32;
        }
    }
}
