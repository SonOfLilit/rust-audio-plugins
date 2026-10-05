//! ToTape8: tape. A Dolby-like encode ("Dubly"), flutter from a wandering
//! delay, bias as a chain of slew limiters, the ToTape saturation (sine on
//! the lows, a thinning cosine on the highs), a resonant head bump, the
//! matching decode, and ClipOnly2 at the output.

use crate::math::{cos, dither, guard, log, pow, sin, tan, xorshift, SEEDS};
use core::f64::consts::PI;

/// How long ToTape8 rings after its input stops, at most. The flutter
/// line is under a thousand samples and ClipOnly2 holds sixteen; what
/// lasts is the head bump, a resonant bandpass at 25 to 200 Hz fed by an
/// integrator whose only leak is cubic. tests/behaviour.rs measures it
/// falling below -90 dBFS within a third of a second at the knobs that
/// stretch it most.
pub const TAIL_SECONDS: f32 = 0.5;

// `gslew`'s layout: a left and right history and a threshold, nine times.
const GSLEW_TOTAL: usize = 27;
const fn threshold(n: usize) -> usize {
    3 * (n - 1) + 2
}

// `hdbA` and `hdbB`'s layout.
const HDB_FREQ: usize = 0;
const HDB_RESO: usize = 1;
const HDB_A0: usize = 2;
const HDB_A1: usize = 3;
const HDB_A2: usize = 4;
const HDB_B1: usize = 5;
const HDB_B2: usize = 6;
const HDB_SL1: usize = 7;
const HDB_SL2: usize = 8;
const HDB_SR1: usize = 9;
const HDB_SR2: usize = 10;
const HDB_TOTAL: usize = 11;

/// Dubly's encode or decode, one channel's state.
#[derive(Clone, Copy, Debug)]
struct Dubly {
    iir: f64,
    comp: f64,
    avg: f64,
}

impl Dubly {
    const NEW: Self = Self {
        iir: 0.0,
        comp: 1.0,
        avg: 0.0,
    };

    /// `x` with its highs pushed by `amount`; `split` is how the original
    /// divides the highs between now (`split.0`) and the next sample.
    fn run(&mut self, mut x: f64, freq: f64, split: (f64, f64), amount: f64) -> f64 {
        self.iir = (self.iir * (1.0 - freq)) + (x * freq);
        let mut high_part = (x - self.iir) * split.0;
        high_part += self.avg;
        self.avg = (x - self.iir) * split.1;
        high_part = high_part.clamp(-1.0, 1.0);
        let mut dubly = high_part.abs();
        if dubly > 0.0 {
            let adjust = log(1.0 + (255.0 * dubly)) / 2.40823996531;
            if adjust > 0.0 {
                dubly /= adjust;
            }
            self.comp = (self.comp * (1.0 - freq)) + (dubly * freq);
            x += (high_part * self.comp) * amount;
        }
        x
    }
}

/// One channel's flutter: a 1000-sample line read at a wandering offset.
#[derive(Clone, Debug)]
struct Flutter {
    d: [f64; 1002],
    sweep: f64,
    nextmax: f64,
}

impl Flutter {
    const NEW: Self = Self {
        d: [0.0; 1002],
        sweep: PI,
        nextmax: 0.5,
    };
}

/// One channel's ClipOnly2.
#[derive(Clone, Copy, Debug)]
struct Clip {
    last_sample: f64,
    intermediate: [f64; 16],
    was_pos: bool,
    was_neg: bool,
}

impl Clip {
    const NEW: Self = Self {
        last_sample: 0.0,
        intermediate: [0.0; 16],
        was_pos: false,
        was_neg: false,
    };

    fn run(&mut self, mut x: f64, spacing: usize) -> f64 {
        x = x.clamp(-4.0, 4.0);
        if self.was_pos {
            if x < self.last_sample {
                self.last_sample = 0.7058208 + (x * 0.2609148);
            } else {
                self.last_sample = 0.2491717 + (self.last_sample * 0.7390851);
            }
        }
        self.was_pos = false;
        if x > 0.9549925859 {
            self.was_pos = true;
            x = 0.7058208 + (self.last_sample * 0.2609148);
        }
        if self.was_neg {
            if x > self.last_sample {
                self.last_sample = -0.7058208 + (x * 0.2609148);
            } else {
                self.last_sample = -0.2491717 + (self.last_sample * 0.7390851);
            }
        }
        self.was_neg = false;
        if x < -0.9549925859 {
            self.was_neg = true;
            x = -0.7058208 + (self.last_sample * 0.2609148);
        }
        self.intermediate[spacing] = x;
        let out = self.last_sample;
        for i in (1..=spacing).rev() {
            self.intermediate[i - 1] = self.intermediate[i];
        }
        self.last_sample = self.intermediate[0];
        out
    }
}

/// ToTape's saturation on one channel: the lows (below the mid roller,
/// less the sub highpass) through a sine, the highs thinned by a cosine.
#[derive(Clone, Copy, Debug)]
struct Tape {
    iir_mid_roller: f64,
    iir_low_cutoff: f64,
}

impl Tape {
    const NEW: Self = Self {
        iir_mid_roller: 0.0,
        iir_low_cutoff: 0.0,
    };

    /// The saturated (lows, highs).
    fn run(&mut self, x: f64, k: &Block) -> (f64, f64) {
        self.iir_mid_roller = (self.iir_mid_roller * (1.0 - k.iir_mid_freq)) + (x * k.iir_mid_freq);
        let mut highs = x - self.iir_mid_roller;
        let mut lows = self.iir_mid_roller;
        if k.iir_sub_freq > 0.0 {
            self.iir_low_cutoff =
                (self.iir_low_cutoff * (1.0 - k.iir_sub_freq)) + (lows * k.iir_sub_freq);
            lows -= self.iir_low_cutoff;
        }
        lows = sin(lows.clamp(-1.57079633, 1.57079633));
        let mut thinned = highs.abs() * 1.57079633;
        if thinned > 1.57079633 {
            thinned = 1.57079633;
        }
        thinned = 1.0 - cos(thinned);
        if highs < 0.0 {
            thinned = -thinned;
        }
        highs -= thinned;
        (lows, highs)
    }
}

/// The per-block values every sample reads.
struct Block {
    overallscale: f64,
    spacing: usize,
    input_gain: f64,
    dubly_amount: f64,
    outly_amount: f64,
    iir_enc_freq: f64,
    iir_dec_freq: f64,
    iir_mid_freq: f64,
    flut_depth: f64,
    flut_frequency: f64,
    bias: f64,
    under_bias: f64,
    head_bump_drive: f64,
    head_bump_mix: f64,
    iir_sub_freq: f64,
    output_gain: f64,
}

#[derive(Clone, Debug)]
pub struct ToTape8 {
    pub params: [f32; 9],
    pub sample_rate: f32,
    enc_l: Dubly,
    dec_l: Dubly,
    enc_r: Dubly,
    dec_r: Dubly,
    flutter_l: Flutter,
    flutter_r: Flutter,
    gcount: i32,
    gslew: [f64; GSLEW_TOTAL],
    tape_l: Tape,
    tape_r: Tape,
    head_bump_l: f64,
    head_bump_r: f64,
    hdb_a: [f64; HDB_TOTAL],
    hdb_b: [f64; HDB_TOTAL],
    clip_l: Clip,
    clip_r: Clip,
    fpd_l: u32,
    fpd_r: u32,
}

/// The bandpass `hdbA`/`hdbB` compute at the top of every block.
fn bandpass(hdb: &mut [f64; HDB_TOTAL]) {
    let k = tan(PI * hdb[HDB_FREQ]);
    let norm = 1.0 / (1.0 + k / hdb[HDB_RESO] + k * k);
    hdb[HDB_A0] = k / hdb[HDB_RESO] * norm;
    hdb[HDB_A2] = -hdb[HDB_A0];
    hdb[HDB_B1] = 2.0 * (k * k - 1.0) * norm;
    hdb[HDB_B2] = (1.0 - k / hdb[HDB_RESO] + k * k) * norm;
}

/// One transposed-direct-form-II step, on the state at `s1` and `s2`.
fn biquad(hdb: &mut [f64; HDB_TOTAL], x: f64, (s1, s2): (usize, usize)) -> f64 {
    let y = (x * hdb[HDB_A0]) + hdb[s1];
    hdb[s1] = (x * hdb[HDB_A1]) - (y * hdb[HDB_B1]) + hdb[s2];
    hdb[s2] = (x * hdb[HDB_A2]) - (y * hdb[HDB_B2]);
    y
}

impl ToTape8 {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            params: [0.5; 9],
            sample_rate,
            enc_l: Dubly::NEW,
            dec_l: Dubly::NEW,
            enc_r: Dubly::NEW,
            dec_r: Dubly::NEW,
            flutter_l: Flutter::NEW,
            flutter_r: Flutter::NEW,
            gcount: 0,
            gslew: [0.0; GSLEW_TOTAL],
            tape_l: Tape::NEW,
            tape_r: Tape::NEW,
            head_bump_l: 0.0,
            head_bump_r: 0.0,
            hdb_a: [0.0; HDB_TOTAL],
            hdb_b: [0.0; HDB_TOTAL],
            clip_l: Clip::NEW,
            clip_r: Clip::NEW,
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

    fn block(&mut self) -> Block {
        let [a, b, c, d, e, f, g, h, i] = self.params;
        let [a, b, c, d, e, f, g, i] = [a, b, c, d, e, f, g, i].map(f64::from);
        let rate = f64::from(self.sample_rate);
        let mut overallscale = 1.0;
        overallscale /= 44100.0;
        overallscale *= rate;

        let spacing = (overallscale.floor() as i32).clamp(1, 16) as usize;
        let mut outly_amount = (1.0 - b) * -2.0;
        if outly_amount < -1.0 {
            outly_amount = -1.0;
        }
        let mut flut_depth = pow(d, 6.0) * overallscale * 50.0;
        if flut_depth > 498.0 {
            flut_depth = 498.0;
        }
        let bias = (f * 2.0) - 1.0;
        let mut under_bias = (pow(bias, 4.0) * 0.25) / overallscale;
        let mut over_bias = pow(1.0 - bias, 3.0) / overallscale;
        if bias > 0.0 {
            under_bias = 0.0;
        }
        if bias < 0.0 {
            over_bias = 1.0 / overallscale;
        }
        for n in (1..=9).rev() {
            self.gslew[threshold(n)] = over_bias;
            over_bias *= 1.618033988749894848204586;
        }

        let sub_curve = sin(g * PI);

        // `(H*H)` is a float times a float, rounded to float.
        self.hdb_a[HDB_FREQ] = ((f64::from(h * h) * 175.0) + 25.0) / rate;
        self.hdb_b[HDB_FREQ] = self.hdb_a[HDB_FREQ] * 0.9375;
        self.hdb_a[HDB_RESO] = 0.618033988749894848204586;
        self.hdb_b[HDB_RESO] = 0.618033988749894848204586;
        self.hdb_a[HDB_A1] = 0.0;
        self.hdb_b[HDB_A1] = 0.0;
        bandpass(&mut self.hdb_a);
        bandpass(&mut self.hdb_b);

        Block {
            overallscale,
            spacing,
            input_gain: pow(a * 2.0, 2.0),
            dubly_amount: b * 2.0,
            outly_amount,
            iir_enc_freq: (1.0 - c) / overallscale,
            iir_dec_freq: c / overallscale,
            iir_mid_freq: ((c * 0.618) + 0.382) / overallscale,
            flut_depth,
            flut_frequency: (0.02 * pow(e, 3.0)) / overallscale,
            bias,
            under_bias,
            head_bump_drive: (g * 0.1) / overallscale,
            head_bump_mix: g * 0.5,
            iir_sub_freq: (sub_curve * 0.008) / overallscale,
            output_gain: i * 2.0,
        }
    }

    /// Flutter on both channels. The two sweeps steer each other: when
    /// one wraps, it picks its next speed by the other's position.
    fn flutter(&mut self, l: f64, r: f64, k: &Block) -> (f64, f64) {
        if self.gcount < 0 || self.gcount > 999 {
            self.gcount = 999;
        }
        let gcount = self.gcount as usize;
        let l = Self::flutter_one(
            &mut self.flutter_l,
            &mut self.fpd_l,
            self.flutter_r.sweep,
            self.flutter_r.nextmax,
            l,
            gcount,
            k,
        );
        let r = Self::flutter_one(
            &mut self.flutter_r,
            &mut self.fpd_r,
            self.flutter_l.sweep,
            self.flutter_l.nextmax,
            r,
            gcount,
            k,
        );
        self.gcount -= 1;
        (l, r)
    }

    fn flutter_one(
        f: &mut Flutter,
        fpd: &mut u32,
        other_sweep: f64,
        other_nextmax: f64,
        x: f64,
        gcount: usize,
        k: &Block,
    ) -> f64 {
        f.d[gcount] = x;
        let mut count = gcount as i32;
        let offset = k.flut_depth + (k.flut_depth * sin(f.sweep));
        f.sweep += f.nextmax * k.flut_frequency;
        if f.sweep > (PI * 2.0) {
            f.sweep -= PI * 2.0;
            let flut_a = 0.24 + (f64::from(*fpd) / f64::from(u32::MAX) * 0.74);
            xorshift(fpd);
            let flut_b = 0.24 + (f64::from(*fpd) / f64::from(u32::MAX) * 0.74);
            let target = sin(other_sweep + other_nextmax);
            f.nextmax = if (flut_a - target).abs() < (flut_b - target).abs() {
                flut_a
            } else {
                flut_b
            };
        }
        count += offset.floor() as i32;
        let frac = offset - offset.floor();
        let at = |c: i32| (c - if c > 999 { 1000 } else { 0 }) as usize;
        let mut y = f.d[at(count)] * (1.0 - frac);
        y += f.d[at(count + 1)] * frac;
        y
    }

    /// The bias routine: nine slew limiters in a row, and below the
    /// middle of the knob, a stickiness around each one's last value.
    fn bias(&mut self, mut l: f64, mut r: f64, k: &Block) -> (f64, f64) {
        let g = &mut self.gslew;
        for x in (0..GSLEW_TOTAL).step_by(3) {
            if k.under_bias > 0.0 {
                let stuck = (l - (g[x] / 0.975)).abs() / k.under_bias;
                if stuck < 1.0 {
                    l = (l * stuck) + ((g[x] / 0.975) * (1.0 - stuck));
                }
                let stuck = (r - (g[x + 1] / 0.975)).abs() / k.under_bias;
                if stuck < 1.0 {
                    r = (r * stuck) + ((g[x + 1] / 0.975) * (1.0 - stuck));
                }
            }
            if (l - g[x]) > g[x + 2] {
                l = g[x] + g[x + 2];
            }
            if -(l - g[x]) > g[x + 2] {
                l = g[x] - g[x + 2];
            }
            g[x] = l * 0.975;
            if (r - g[x + 1]) > g[x + 2] {
                r = g[x + 1] + g[x + 2];
            }
            if -(r - g[x + 1]) > g[x + 2] {
                r = g[x + 1] - g[x + 2];
            }
            g[x + 1] = r * 0.975;
        }
        (l, r)
    }

    /// The head bump on both channels: the lows integrated (with a cubic
    /// leak) and run through two bandpasses in series.
    fn head_bump(&mut self, lows_l: f64, lows_r: f64, k: &Block) -> (f64, f64) {
        let leak = 0.0618 / k.overallscale.sqrt();
        self.head_bump_l += lows_l * k.head_bump_drive;
        self.head_bump_l -= self.head_bump_l * self.head_bump_l * self.head_bump_l * leak;
        self.head_bump_r += lows_r * k.head_bump_drive;
        self.head_bump_r -= self.head_bump_r * self.head_bump_r * self.head_bump_r * leak;
        let biq_l = biquad(&mut self.hdb_a, self.head_bump_l, (HDB_SL1, HDB_SL2));
        let out_l = biquad(&mut self.hdb_b, biq_l, (HDB_SL1, HDB_SL2));
        let biq_r = biquad(&mut self.hdb_a, self.head_bump_r, (HDB_SR1, HDB_SR2));
        let out_r = biquad(&mut self.hdb_b, biq_r, (HDB_SR1, HDB_SR2));
        (out_l, out_r)
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let k = self.block();
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mut sl = guard(*l, self.fpd_l);
            let mut sr = guard(*r, self.fpd_r);
            if k.input_gain != 1.0 {
                sl *= k.input_gain;
                sr *= k.input_gain;
            }
            sl = self
                .enc_l
                .run(sl, k.iir_enc_freq, (2.848, 1.152), k.dubly_amount);
            sr = self
                .enc_r
                .run(sr, k.iir_enc_freq, (2.848, 1.152), k.dubly_amount);
            if k.flut_depth > 0.0 {
                (sl, sr) = self.flutter(sl, sr, &k);
            }
            if k.bias.abs() > 0.001 {
                (sl, sr) = self.bias(sl, sr, &k);
            }
            let (lows_l, highs_l) = self.tape_l.run(sl, &k);
            let (lows_r, highs_r) = self.tape_r.run(sr, &k);
            let (bump_l, bump_r) = if k.head_bump_mix > 0.0 {
                self.head_bump(lows_l, lows_r, &k)
            } else {
                (0.0, 0.0)
            };
            sl = lows_l + highs_l + (bump_l * k.head_bump_mix);
            sr = lows_r + highs_r + (bump_r * k.head_bump_mix);
            sl = self
                .dec_l
                .run(sl, k.iir_dec_freq, (2.628, 1.372), k.outly_amount);
            sr = self
                .dec_r
                .run(sr, k.iir_dec_freq, (2.628, 1.372), k.outly_amount);
            if k.output_gain != 1.0 {
                sl *= k.output_gain;
                sr *= k.output_gain;
            }
            sl = self.clip_l.run(sl, k.spacing);
            sr = self.clip_r.run(sr, k.spacing);
            *l = dither(sl, &mut self.fpd_l) as f32;
            *r = dither(sr, &mut self.fpd_r) as f32;
        }
    }
}
