//! `gin::PlateReverb<float, int>`: Jon Dattorro's plate, after Mike
//! Jarmy's implementation (MIT; see LICENSE).
//!
//! Ported from Gin's `gin_dsp/dsp/gin_platereverb.h`, instantiated as
//! float samples and int indices.

use crate::math;
use core::f32::consts::PI;

/// `F (0.1)`: C's double literal, rounded to float. (Rust would type a
/// bare `0.1` straight as an f32, which is not always the same float.)
const MAX_PREDELAY: f32 = 0.1_f64 as f32;
const MAX_SIZE: f32 = 2.0;
const TAPS: usize = 7;

/// `std::min (std::max (val, low), high)`.
fn clamp(val: f32, low: f32, high: f32) -> f32 {
    math::std_min(math::std_max(val, low), high)
}

/// `FastMath<float>::fastSin`: a parabola, sharpened, over -π to π.
fn fast_sin(x: f32) -> f32 {
    const B: f32 = (4.0 / core::f64::consts::PI) as f32;
    const C: f32 = (-4.0 / (core::f64::consts::PI * core::f64::consts::PI)) as f32;
    const P: f32 = 0.225_f64 as f32;
    let y = B * x + C * x * x.abs();
    P * (y * y.abs() - y) + y
}

#[derive(Clone, Copy, Debug)]
struct OnePole {
    sample_rate: f32,
    cutoff: f32,
    a: f32,
    b: f32,
    z: f32,
}

impl Default for OnePole {
    fn default() -> Self {
        Self {
            sample_rate: 1.0,
            cutoff: 0.0,
            a: 0.0,
            b: 0.0,
            z: 0.0,
        }
    }
}

impl OnePole {
    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.recalc();
    }

    fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff;
        self.recalc();
    }

    fn recalc(&mut self) {
        self.b = math::expf(-2.0 * PI * self.cutoff / self.sample_rate);
        self.a = 1.0 - self.b;
    }

    fn process(&mut self, x: f32) -> f32 {
        self.z = x * self.a + self.z * self.b;
        self.z
    }

    /// The filter's memory only, its coefficients kept (bloop's `clear`).
    fn clear(&mut self) {
        self.z = 0.0;
    }

    /// The original's reset clears the coefficients too: nothing passes
    /// until the cutoff is set again.
    fn reset(&mut self) {
        self.a = 0.0;
        self.b = 0.0;
        self.z = 0.0;
    }
}

/// `ceilPowerOfTwo`, computed as the original computes it.
fn ceil_power_of_two(n: i32) -> i32 {
    math::pow(2.0, (math::log(f64::from(n)) / math::log(2.0)).ceil()) as i32
}

#[derive(Clone, Debug)]
struct DelayLine {
    size: i32,
    buffer: Vec<f32>,
    mask: i32,
    write: i32,
}

impl DelayLine {
    fn new(size: i32) -> Self {
        let buffer_size = ceil_power_of_two(size);
        Self {
            size,
            buffer: vec![0.0; buffer_size.max(1) as usize],
            mask: buffer_size - 1,
            write: 0,
        }
    }

    /// A line of `size`, reusing this one's memory if it is already that
    /// size, as a fresh one from `new` would be.
    fn remake(&mut self, size: i32) {
        if self.size == size {
            self.reset();
        } else {
            *self = Self::new(size);
        }
    }

    fn push(&mut self, value: f32) {
        self.buffer[self.write as usize] = value;
        self.write += 1;
        self.write &= self.mask;
    }

    fn tap(&self, delay: f32) -> f32 {
        let d = delay as i32;
        let frac = 1.0 - (delay - d as f32);
        let read = (self.write - 1) - d;
        let a = self.buffer[((read - 1) & self.mask) as usize];
        let b = self.buffer[(read & self.mask) as usize];
        a + (b - a) * frac
    }

    fn tap_and_push(&mut self, delay: f32, value: f32) -> f32 {
        let out = self.tap(delay);
        self.push(value);
        out
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write = 0;
    }
}

#[derive(Clone, Debug)]
struct DelayAllpass {
    line: DelayLine,
    gain: f32,
}

impl DelayAllpass {
    fn new(size: i32, gain: f32) -> Self {
        Self {
            line: DelayLine::new(size),
            gain,
        }
    }

    fn remake(&mut self, size: i32, gain: f32) {
        self.line.remake(size);
        self.gain = gain;
    }

    fn process(&mut self, x: f32, delay: f32) -> f32 {
        let wd = self.line.tap(delay);
        let w = x + self.gain * wd;
        let y = -self.gain * w + wd;
        self.line.push(w);
        y
    }

    fn size(&self) -> i32 {
        self.line.size
    }
}

#[derive(Clone, Copy, Debug)]
struct Lfo {
    sample_rate: f32,
    freq: f32,
    phase_inc: f32,
    phase: f32,
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            sample_rate: 1.0,
            freq: 0.0,
            phase_inc: 0.0,
            phase: -PI,
        }
    }
}

impl Lfo {
    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.recalc();
    }

    fn set_frequency(&mut self, freq: f32) {
        self.freq = freq;
        self.recalc();
    }

    fn recalc(&mut self) {
        self.phase_inc = self.freq / self.sample_rate;
        self.phase_inc *= 2.0 * PI;
    }

    fn process(&mut self) -> f32 {
        let out = -fast_sin(self.phase);
        self.phase += self.phase_inc;
        if self.phase > PI {
            self.phase = -PI;
        }
        out
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

#[derive(Clone, Debug)]
struct Tank {
    out: f32,
    apf1: DelayAllpass,
    apf2: DelayAllpass,
    del1: DelayLine,
    del2: DelayLine,
    damping: OnePole,
    lfo: Lfo,
    apf1_size: i32,
    max_mod_depth: f32,
    mod_depth: f32,
    apf1_delay: f32,
    apf2_delay: f32,
    del1_delay: f32,
    del2_delay: f32,
    decay_rate: f32,
    size_ratio: f32,
}

/// One tank's delay sizes: `resetDelayLines`' arguments.
struct TankSizes {
    apf1: i32,
    apf1_gain: f32,
    max_mod_depth: f32,
    del1: i32,
    apf2: i32,
    apf2_gain: f32,
    del2: i32,
}

impl Tank {
    fn new(sizes: &TankSizes) -> Self {
        let mut tank = Self {
            out: 0.0,
            apf1: DelayAllpass::new(1, 0.0),
            apf2: DelayAllpass::new(1, 0.0),
            del1: DelayLine::new(1),
            del2: DelayLine::new(1),
            damping: OnePole::default(),
            lfo: Lfo::default(),
            apf1_size: 0,
            max_mod_depth: 0.0,
            mod_depth: 0.0,
            apf1_delay: 0.0,
            apf2_delay: 0.0,
            del1_delay: 0.0,
            del2_delay: 0.0,
            decay_rate: 0.0,
            size_ratio: 0.0,
        };
        tank.reset_delay_lines(sizes);
        tank
    }

    fn reset_delay_lines(&mut self, sizes: &TankSizes) {
        self.apf1_size = sizes.apf1;
        self.max_mod_depth = sizes.max_mod_depth;
        let max_apf1_size = self.apf1_size as f32 + self.max_mod_depth + 1.0;
        self.apf1.remake(max_apf1_size as i32, sizes.apf1_gain);
        self.del1.remake(sizes.del1);
        self.apf2.remake(sizes.apf2, sizes.apf2_gain);
        self.del2.remake(sizes.del2);
        self.recalc_size_ratio();
    }

    fn set_decay(&mut self, decay_rate: f32) {
        self.decay_rate = decay_rate;
        self.apf2.gain = clamp((f64::from(decay_rate) + 0.15) as f32, 0.25, 0.5);
    }

    fn set_size_ratio(&mut self, ratio: f32) {
        self.size_ratio = ratio;
        self.recalc_size_ratio();
    }

    fn recalc_size_ratio(&mut self) {
        self.apf1_delay = self.apf1_size as f32 * self.size_ratio;
        self.mod_depth = self.max_mod_depth * self.size_ratio;
        self.apf2_delay = self.apf2.size() as f32 * self.size_ratio;
        self.del1_delay = self.del1.size as f32 * self.size_ratio;
        self.del2_delay = self.del2.size as f32 * self.size_ratio;
    }

    fn process(&mut self, value: f32) {
        let mut val = self
            .apf1
            .process(value, self.apf1_delay + self.lfo.process() * self.mod_depth);
        val = self.del1.tap_and_push(self.del1_delay, val);
        val = self.damping.process(val);
        val *= self.decay_rate;
        val = self.apf2.process(val, self.apf2_delay);
        val = self.del2.tap_and_push(self.del2_delay, val);
        self.out = val;
    }

    fn reset(&mut self) {
        self.apf1.line.reset();
        self.apf2.line.reset();
        self.del1.reset();
        self.del2.reset();
        self.damping.reset();
        self.lfo.reset();
    }

    /// The tank's audio, and nothing else: the lines, the damping
    /// filter's memory and what it last fed across to the other tank.
    /// The LFO runs on, as it would have through silence.
    fn clear(&mut self) {
        self.apf1.line.reset();
        self.apf2.line.reset();
        self.del1.reset();
        self.del2.reset();
        self.damping.clear();
        self.out = 0.0;
    }
}

/// `(I) std::ceil (x)`.
fn ceil_i(x: f32) -> i32 {
    x.ceil() as i32
}

#[derive(Clone, Debug)]
pub struct PlateReverb {
    sample_rate: f32,
    mix: f32,
    predelay: f32,
    decay_rate: f32,
    predelay_line: DelayLine,
    lowpass: OnePole,
    diffusers: [DelayAllpass; 4],
    left: Tank,
    right: Tank,
    base_left_taps: [f32; TAPS],
    base_right_taps: [f32; TAPS],
    left_taps: [f32; TAPS],
    right_taps: [f32; TAPS],
}

impl PlateReverb {
    /// Built and sized for `sample_rate`: the original is unusable until
    /// `setSampleRate`, so here the two are one step.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let one = TankSizes {
            apf1: 1,
            apf1_gain: 0.0,
            max_mod_depth: 0.0,
            del1: 1,
            apf2: 1,
            apf2_gain: 0.0,
            del2: 1,
        };
        let mut plate = Self {
            sample_rate: 1.0,
            mix: 0.0,
            predelay: 0.0,
            decay_rate: 0.0,
            predelay_line: DelayLine::new(1),
            lowpass: OnePole::default(),
            diffusers: [
                DelayAllpass::new(1, 0.0),
                DelayAllpass::new(1, 0.0),
                DelayAllpass::new(1, 0.0),
                DelayAllpass::new(1, 0.0),
            ],
            left: Tank::new(&one),
            right: Tank::new(&one),
            base_left_taps: [0.0; TAPS],
            base_right_taps: [0.0; TAPS],
            left_taps: [0.0; TAPS],
            right_taps: [0.0; TAPS],
        };
        plate.set_sample_rate(sample_rate);
        plate
    }

    /// `setSampleRate`: every line remade for the rate. Remaking a line at
    /// the size it already has clears it without allocating.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        let r = (f64::from(sample_rate) / 29761.0) as f32;

        self.predelay_line
            .remake(ceil_i(sample_rate * MAX_PREDELAY));

        self.lowpass.set_sample_rate(sample_rate);
        self.left.damping.set_sample_rate(sample_rate);
        self.right.damping.set_sample_rate(sample_rate);

        for (d, (len, gain)) in self.diffusers.iter_mut().zip([
            (142.0, 0.75),
            (107.0, 0.75),
            (379.0, 0.625),
            (277.0, 0.625),
        ]) {
            d.remake(ceil_i(len * r), gain);
        }

        let max_mod_depth = (8.0 * f64::from(MAX_SIZE) * f64::from(r)) as f32;
        self.left.reset_delay_lines(&TankSizes {
            apf1: ceil_i(MAX_SIZE * 672.0 * r),
            apf1_gain: -0.7_f64 as f32,
            max_mod_depth,
            del1: ceil_i(MAX_SIZE * 4453.0 * r),
            apf2: ceil_i(MAX_SIZE * 1800.0 * r),
            apf2_gain: 0.5,
            del2: ceil_i(MAX_SIZE * 3720.0 * r),
        });
        self.right.reset_delay_lines(&TankSizes {
            apf1: ceil_i(MAX_SIZE * 908.0 * r),
            apf1_gain: -0.7_f64 as f32,
            max_mod_depth,
            del1: ceil_i(MAX_SIZE * 4217.0 * r),
            apf2: ceil_i(MAX_SIZE * 2656.0 * r),
            apf2_gain: 0.5,
            del2: ceil_i(MAX_SIZE * 3163.0 * r),
        });

        self.left.lfo.set_sample_rate(sample_rate);
        self.right.lfo.set_sample_rate(sample_rate);
        self.left.lfo.set_frequency(1.0);
        self.right.lfo.set_frequency(0.95_f64 as f32);

        self.base_left_taps = [266.0, 2974.0, 1913.0, 1996.0, 1990.0, 187.0, 1066.0].map(|t| t * r);
        self.base_right_taps = [353.0, 3627.0, 1228.0, 2673.0, 2111.0, 335.0, 121.0].map(|t| t * r);
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = clamp(mix, 0.0, 1.0);
    }

    pub fn set_predelay(&mut self, seconds: f32) {
        self.predelay = clamp(seconds, 0.0, MAX_PREDELAY) * self.sample_rate;
    }

    pub fn set_lowpass(&mut self, cutoff: f32) {
        self.lowpass.set_cutoff(clamp(cutoff, 16.0, 20000.0));
    }

    pub fn set_decay(&mut self, decay: f32) {
        self.decay_rate = clamp(decay, 0.0, 0.999_999_9_f64 as f32);
        self.left.set_decay(self.decay_rate);
        self.right.set_decay(self.decay_rate);
    }

    pub fn set_size(&mut self, size: f32) {
        let ratio = clamp(size, 0.0, MAX_SIZE) / MAX_SIZE;
        self.left.set_size_ratio(ratio);
        self.right.set_size_ratio(ratio);
        for i in 0..TAPS {
            self.left_taps[i] = self.base_left_taps[i] * ratio;
            self.right_taps[i] = self.base_right_taps[i] * ratio;
        }
    }

    pub fn set_damping(&mut self, cutoff: f32) {
        let cutoff = clamp(cutoff, 16.0, 20000.0);
        self.left.damping.set_cutoff(cutoff);
        self.right.damping.set_cutoff(cutoff);
    }

    fn process_sample(&mut self, dry_left: f32, dry_right: f32) -> (f32, f32) {
        let mut sum = dry_left + dry_right;
        sum = self.predelay_line.tap_and_push(self.predelay, sum);
        sum = self.lowpass.process(sum);
        for d in &mut self.diffusers {
            let size = d.size() as f32;
            sum = d.process(sum, size);
        }

        let left_in = sum + self.right.out * self.decay_rate;
        let right_in = sum + self.left.out * self.decay_rate;
        self.left.process(left_in);
        self.right.process(right_in);

        let (l, r) = (&self.left, &self.right);
        let lt = &self.left_taps;
        let rt = &self.right_taps;
        let wet_left = r.del1.tap(lt[0]) + r.del1.tap(lt[1]) - r.apf2.line.tap(lt[2])
            + r.del2.tap(lt[3])
            - l.del1.tap(lt[4])
            - l.apf2.line.tap(lt[5])
            - l.del2.tap(lt[6]);
        let wet_right = l.del1.tap(rt[0]) + l.del1.tap(rt[1]) - l.apf2.line.tap(rt[2])
            + l.del2.tap(rt[3])
            - r.del1.tap(rt[4])
            - r.apf2.line.tap(rt[5])
            - r.del2.tap(rt[6]);

        (
            dry_left * (1.0 - self.mix) + wet_left * self.mix,
            dry_right * (1.0 - self.mix) + wet_right * self.mix,
        )
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            (*l, *r) = self.process_sample(*l, *r);
        }
    }

    pub fn reset(&mut self) {
        self.predelay_line.reset();
        self.lowpass.reset();
        for d in &mut self.diffusers {
            d.line.reset();
        }
        self.left.reset();
        self.right.reset();
    }

    /// Silence the reverb and leave it as it would be after silence:
    /// bloop's Choke All Notes. The original's `reset` also zeroes the
    /// filters' coefficients, which only a changed knob sets again, so a
    /// reverb reset mid-stream stays wrong until someone turns one.
    pub fn clear(&mut self) {
        self.predelay_line.reset();
        self.lowpass.clear();
        for d in &mut self.diffusers {
            d.line.reset();
        }
        self.left.clear();
        self.right.clear();
    }
}
