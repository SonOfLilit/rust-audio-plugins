//! Console7Channel and Console7Buss: Airwindows' console pair. A channel
//! saturates on a sine curve behind its fader, a buss on an arcsine, each
//! between 20 kHz lowpasses; the fader is chased smoothly, faster for a
//! while after it moves.

use crate::math::{asin, dither, guard, pow, sin, tan, SEEDS};
use core::f64::consts::PI;

/// `biquadA[15]`: frequency, resonance, five coefficients, then the left
/// and right direct-form-I histories.
type Biquad = [f64; 15];

/// The coefficients the originals compute at the top of every block: a
/// lowpass at `biquad[0]` with resonance `biquad[1]`.
fn lowpass(biquad: &mut Biquad, k: f64) {
    let norm = 1.0 / (1.0 + k / biquad[1] + k * k);
    biquad[2] = k * k * norm;
    biquad[3] = 2.0 * biquad[2];
    biquad[4] = biquad[2];
    biquad[5] = 2.0 * (k * k - 1.0) * norm;
    biquad[6] = (1.0 - k / biquad[1] + k * k) * norm;
}

/// One direct-form-I step, left (`history` 7) or right (11).
fn run(b: &mut Biquad, x: f64, h: usize) -> f64 {
    let out = b[2] * x + b[3] * b[h] + b[4] * b[h + 1] - b[5] * b[h + 2] - b[6] * b[h + 3];
    b[h + 1] = b[h];
    b[h] = x;
    b[h + 3] = b[h + 2];
    b[h + 2] = out;
    out
}

/// The fader chase both share: `chasespeed` doubles when the fader moves,
/// held to the block length, and relaxes back to 64 sample by sample.
#[derive(Clone, Debug)]
struct Chase {
    gainchase: f64,
    chasespeed: f64,
}

impl Chase {
    const NEW: Self = Self {
        gainchase: -1.0,
        chasespeed: 64.0,
    };

    fn block(&mut self, inputgain: f64, frames: usize) {
        if self.gainchase != inputgain {
            self.chasespeed *= 2.0;
        }
        let frames = frames as i32 as f64;
        if self.chasespeed > frames {
            self.chasespeed = frames;
        }
        if self.gainchase < 0.0 {
            self.gainchase = inputgain;
        }
    }

    fn sample(&mut self, inputgain: f64) -> f64 {
        self.chasespeed *= 0.9999;
        self.chasespeed -= 0.01;
        if self.chasespeed < 64.0 {
            self.chasespeed = 64.0;
        }
        self.gainchase = ((self.gainchase * self.chasespeed) + inputgain) / (self.chasespeed + 1.0);
        self.gainchase
    }
}

/// `x / ((fabs(x) == 0.0) ? 1 : fabs(x))`'s divisor.
fn nonzero_abs(x: f64) -> f64 {
    if x.abs() == 0.0 {
        1.0
    } else {
        x.abs()
    }
}

#[derive(Clone, Debug)]
pub struct Console7Channel {
    pub params: [f32; 1],
    pub sample_rate: f32,
    chase: Chase,
    biquad_a: Biquad,
    fpd_l: u32,
    fpd_r: u32,
}

impl Console7Channel {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            params: [0.772_f64 as f32],
            sample_rate,
            chase: Chase::NEW,
            biquad_a: [0.0; 15],
            fpd_l: SEEDS.0,
            fpd_r: SEEDS.1,
        }
    }

    pub fn clear(&mut self) {
        *self = Self {
            params: self.params,
            ..Self::new(self.sample_rate)
        };
    }

    fn saturate(x: f64) -> f64 {
        let x = x.clamp(-1.097, 1.097);
        ((sin(x * x.abs()) / nonzero_abs(x)) * 0.8) + (sin(x) * 0.2)
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let inputgain = f64::from(self.params[0]) * 1.272019649514069;
        self.chase.block(inputgain, left.len());
        let b = &mut self.biquad_a;
        b[0] = 20000.0 / f64::from(self.sample_rate);
        b[1] = 1.618033988749894848204586;
        lowpass(b, tan(PI * b[0]));

        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mut sl = guard(*l, self.fpd_l);
            let mut sr = guard(*r, self.fpd_r);
            sl = run(b, sl, 7);
            sr = run(b, sr, 11);
            let gainchase = self.chase.sample(inputgain);
            if 1.0 != gainchase {
                sl *= pow(gainchase, 3.0);
                sr *= pow(gainchase, 3.0);
            }
            sl = Self::saturate(sl);
            sr = Self::saturate(sr);
            if 1.0 != gainchase && 0.0 != gainchase {
                sl /= gainchase;
                sr /= gainchase;
            }
            *l = dither(sl, &mut self.fpd_l) as f32;
            *r = dither(sr, &mut self.fpd_r) as f32;
        }
    }
}

#[derive(Clone, Debug)]
pub struct Console7Buss {
    pub params: [f32; 1],
    pub sample_rate: f32,
    chase: Chase,
    biquad_a: Biquad,
    biquad_b: Biquad,
    fpd_l: u32,
    fpd_r: u32,
}

impl Console7Buss {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            params: [1.0],
            sample_rate,
            chase: Chase::NEW,
            biquad_a: [0.0; 15],
            biquad_b: [0.0; 15],
            fpd_l: SEEDS.0,
            fpd_r: SEEDS.1,
        }
    }

    pub fn clear(&mut self) {
        *self = Self {
            params: self.params,
            ..Self::new(self.sample_rate)
        };
    }

    fn saturate(x: f64) -> f64 {
        let x = x.clamp(-1.0, 1.0);
        ((asin(x * x.abs()) / nonzero_abs(x)) * 0.618033988749894848204586)
            + (asin(x) * 0.381966011250105)
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let inputgain = f64::from(self.params[0]) * 1.03;
        self.chase.block(inputgain, left.len());
        let (a, b) = (&mut self.biquad_a, &mut self.biquad_b);
        a[0] = 20000.0 / f64::from(self.sample_rate);
        b[0] = a[0];
        a[1] = 0.618033988749894848204586;
        b[1] = 0.5;
        lowpass(a, tan(PI * a[0]));
        lowpass(b, tan(PI * a[0]));

        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mut sl = guard(*l, self.fpd_l);
            let mut sr = guard(*r, self.fpd_r);
            sl = run(a, sl, 7);
            sr = run(a, sr, 11);
            let gainchase = self.chase.sample(inputgain);
            if 1.0 != gainchase {
                sl *= gainchase.sqrt();
                sr *= gainchase.sqrt();
            }
            sl = Self::saturate(sl);
            sr = Self::saturate(sr);
            sl = run(b, sl, 7);
            sr = run(b, sr, 11);
            if 1.0 != gainchase {
                sl *= gainchase.sqrt();
                sr *= gainchase.sqrt();
            }
            *l = dither(sl, &mut self.fpd_l) as f32;
            *r = dither(sr, &mut self.fpd_r) as f32;
        }
    }
}
