//! A 4x oversampler for the BitCrusher: two 2x stages of polyphase IIR
//! half-band filters.
//!
//! This is bloop's, not Gin's. Gin's BitCrusher runs its quantiser inside
//! `juce::dsp::Oversampling`, which is JUCE's code and under a licence this
//! crate cannot carry, so it is replaced here with the classic alternative:
//! each 2x stage is a pair of allpass chains, one per polyphase branch,
//! designed as an elliptic half-band after Valenzuela and Constantinides —
//! the design Laurent de Soras' HIIR library (WTFPL) popularised. It
//! differs from JUCE's in the last bits and in its latency, so the
//! BitCrusher's anti-aliased mode has behaviour tests where the rest of the
//! crate has gold ones.

use crate::math;
use core::f64::consts::PI;

/// `(k, q)` for a half-band with this transition width (a fraction of the
/// sample rate).
fn transition_param(transition: f64) -> (f64, f64) {
    let mut k = math::tan((1.0 - transition * 2.0) * PI / 4.0);
    k *= k;
    let kksqrt = math::pow(1.0 - k * k, 0.25);
    let e = 0.5 * (1.0 - kksqrt) / (1.0 + kksqrt);
    let e2 = e * e;
    let e4 = e2 * e2;
    let q = e * (1.0 + e4 * (2.0 + e4 * (15.0 + 150.0 * e4)));
    (k, q)
}

fn acc_num(q: f64, order: f64, c: f64) -> f64 {
    let mut i = 0i32;
    let mut sign = 1.0;
    let mut acc = 0.0;
    loop {
        let term = q.powi(i * (i + 1)) * math::sin(f64::from(i * 2 + 1) * c * PI / order) * sign;
        acc += term;
        sign = -sign;
        i += 1;
        if term.abs() <= 1e-100 {
            return acc;
        }
    }
}

fn acc_den(q: f64, order: f64, c: f64) -> f64 {
    let mut i = 1i32;
    let mut sign = -1.0;
    let mut acc = 0.0;
    loop {
        let term = q.powi(i * i) * math::cos(f64::from(i * 2) * c * PI / order) * sign;
        acc += term;
        sign = -sign;
        i += 1;
        if term.abs() <= 1e-100 {
            return acc;
        }
    }
}

/// The allpass coefficients of a half-band with `N` of them.
fn coefficients<const N: usize>(transition: f64) -> [f32; N] {
    let (k, q) = transition_param(transition);
    let order = (N * 2 + 1) as f64;
    core::array::from_fn(|index| {
        let c = (index + 1) as f64;
        let num = acc_num(q, order, c) * math::pow(q, 0.25);
        let den = acc_den(q, order, c) + 0.5;
        let ww = num / den;
        let wwsq = ww * ww;
        let x = ((1.0 - wwsq * k) * (1.0 - wwsq / k)).sqrt() / (1.0 + wwsq);
        ((1.0 - x) / (1.0 + x)) as f32
    })
}

/// Both allpass chains of one polyphase half-band, run a sample at a time:
/// the even coefficients on one branch, the odd ones on the other.
#[derive(Clone, Copy, Debug)]
struct HalfBand<const N: usize> {
    coefs: [f32; N],
    x: [f32; N],
    y: [f32; N],
}

impl<const N: usize> HalfBand<N> {
    fn new(transition: f64) -> Self {
        Self {
            coefs: coefficients::<N>(transition),
            x: [0.0; N],
            y: [0.0; N],
        }
    }

    fn reset(&mut self) {
        self.x = [0.0; N];
        self.y = [0.0; N];
    }

    fn branches(&mut self, mut a: f32, mut b: f32) -> (f32, f32) {
        for i in (0..N).step_by(2) {
            let t0 = (a - self.y[i]) * self.coefs[i] + self.x[i];
            self.x[i] = a;
            self.y[i] = t0;
            a = t0;
            if i + 1 < N {
                let t1 = (b - self.y[i + 1]) * self.coefs[i + 1] + self.x[i + 1];
                self.x[i + 1] = b;
                self.y[i + 1] = t1;
                b = t1;
            }
        }
        (a, b)
    }

    /// One sample in, two out.
    fn up(&mut self, x: f32) -> [f32; 2] {
        let (a, b) = self.branches(x, x);
        [a, b]
    }

    /// Two samples in, one out.
    fn down(&mut self, pair: [f32; 2]) -> f32 {
        let (a, b) = self.branches(pair[1], pair[0]);
        0.5 * (a + b)
    }
}

/// The first stage filters the full band and has to be steep; the second
/// sees only what the first let through and can be gentle.
const FIRST: (usize, f64) = (8, 0.04);
const SECOND: (usize, f64) = (4, 0.24);

/// One channel of 4x up- and down-sampling.
#[derive(Clone, Copy, Debug)]
struct Channel {
    up1: HalfBand<{ FIRST.0 }>,
    up2: HalfBand<{ SECOND.0 }>,
    down2: HalfBand<{ SECOND.0 }>,
    down1: HalfBand<{ FIRST.0 }>,
}

impl Channel {
    fn new() -> Self {
        Self {
            up1: HalfBand::new(FIRST.1),
            up2: HalfBand::new(SECOND.1),
            down2: HalfBand::new(SECOND.1),
            down1: HalfBand::new(FIRST.1),
        }
    }

    fn reset(&mut self) {
        self.up1.reset();
        self.up2.reset();
        self.down2.reset();
        self.down1.reset();
    }

    fn up(&mut self, x: f32) -> [f32; 4] {
        let [a, b] = self.up1.up(x);
        let [a0, a1] = self.up2.up(a);
        let [b0, b1] = self.up2.up(b);
        [a0, a1, b0, b1]
    }

    fn down(&mut self, four: [f32; 4]) -> f32 {
        let a = self.down2.down([four[0], four[1]]);
        let b = self.down2.down([four[2], four[3]]);
        self.down1.down([a, b])
    }
}

/// 4x oversampling of a stereo pair.
#[derive(Clone, Debug)]
pub struct Oversampler {
    channels: [Channel; 2],
}

impl Default for Oversampler {
    fn default() -> Self {
        Self {
            channels: [Channel::new(), Channel::new()],
        }
    }
}

impl Oversampler {
    pub fn reset(&mut self) {
        for c in &mut self.channels {
            c.reset();
        }
    }

    /// One sample of `channel` up to four.
    pub fn up(&mut self, channel: usize, x: f32) -> [f32; 4] {
        self.channels[channel].up(x)
    }

    /// Four samples of `channel` back down to one.
    pub fn down(&mut self, channel: usize, four: [f32; 4]) -> f32 {
        self.channels[channel].down(four)
    }
}
