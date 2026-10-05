//! Airwindows DeRez2 (MIT), the distortion's Bitcrusher mode: sample-rate
//! and bit-depth reduction through a µ-law curve, with "hard" mixing in
//! the dry transitions. Ported from Wavetable's `FX/DeRez2Proc.cpp`, the
//! single-precision `processReplacing`, which works in double.

use bloop_plugin_socalabs::shared::math;

/// What Wavetable keeps between blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeRez2 {
    /// `A` to `D`, the four knobs, 0 to 1.
    pub params: [f32; 4],
    last_sample_l: f64,
    held_sample_l: f64,
    last_dry_sample_l: f64,
    last_output_sample_l: f64,
    last_sample_r: f64,
    held_sample_r: f64,
    last_dry_sample_r: f64,
    last_output_sample_r: f64,
    position: f64,
    increment_a: f64,
    increment_b: f64,
    /// The dither seeds, drawn and never used: the dither is commented out
    /// in the original, but the draws move `rand()` for the effects built
    /// after this one.
    fpd: [u32; 2],
}

/// `fpd = 1.0; while (fpd < 16386) fpd = rand()*UINT32_MAX;`: a dither
/// seed, from as many `rand()`s as it takes.
pub fn seed_fpd(rand: &mut dyn FnMut() -> i32) -> u32 {
    let mut fpd = 1u32;
    while fpd < 16386 {
        fpd = (rand() as u32).wrapping_mul(u32::MAX);
    }
    fpd
}

impl Default for DeRez2 {
    fn default() -> Self {
        Self {
            params: [1.0; 4],
            last_sample_l: 0.0,
            held_sample_l: 0.0,
            last_dry_sample_l: 0.0,
            last_output_sample_l: 0.0,
            last_sample_r: 0.0,
            held_sample_r: 0.0,
            last_dry_sample_r: 0.0,
            last_output_sample_r: 0.0,
            position: 0.0,
            increment_a: 0.0,
            increment_b: 0.0,
            fpd: [1; 2],
        }
    }
}

/// `log(1 + 255|x|) / log(256)`, signed: µ-law encoding.
fn ulaw_encode(x: f64) -> f64 {
    if x > 0.0 {
        math::log(1.0 + (255.0 * x.abs())) / math::log(256.0)
    } else if x < 0.0 {
        -math::log(1.0 + (255.0 * x.abs())) / math::log(256.0)
    } else {
        x
    }
}

/// `(256^|x| - 1) / 255`, signed: µ-law decoding.
fn ulaw_decode(x: f64) -> f64 {
    if x > 0.0 {
        (math::pow(256.0, x.abs()) - 1.0) / 255.0
    } else if x < 0.0 {
        -(math::pow(256.0, x.abs()) - 1.0) / 255.0
    } else {
        x
    }
}

/// One channel's dry-transition step: a sample unlike the last output
/// takes `hard` of itself and the rest of the last dry sample.
fn soften_transition(input: f64, last_output: &mut f64, last_dry: f64, hard: f64) -> f64 {
    if input == *last_output {
        *last_output = input;
        input
    } else {
        let temp = input;
        *last_output = temp;
        (input * hard) + (last_dry * (1.0 - hard))
    }
}

/// The bit-depth reduction: `x` down to a multiple of `step`, by
/// subtracting what is left over, then scaled.
fn quantise(x: f64, step: f64) -> f64 {
    let mut out = x;
    if out > 0.0 {
        let mut temp = out;
        while temp > 0.0 {
            temp -= step;
        }
        out -= temp;
    }
    if out < 0.0 {
        let mut temp = out;
        while temp < 0.0 {
            temp += step;
        }
        out -= temp;
    }
    out
}

fn clip(x: f64) -> f64 {
    if x > 1.0 {
        1.0
    } else if x < -1.0 {
        -1.0
    } else {
        x
    }
}

impl DeRez2 {
    /// As constructed, drawing its dither seeds from `rand`.
    pub fn new(rand: &mut dyn FnMut() -> i32) -> Self {
        let mut d = Self::default();
        d.fpd = [seed_fpd(rand), seed_fpd(rand)];
        d
    }

    /// `setParameter (index, value)`.
    pub fn set_parameter(&mut self, index: usize, value: f32) {
        if let Some(p) = self.params.get_mut(index) {
            *p = value;
        }
    }

    /// `processReplacing`, in place.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32], sample_rate: f64) {
        let [a, b, c, d] = self.params.map(f64::from);
        let mut target_a = math::pow(a, 3.0) + 0.0005;
        if target_a > 1.0 {
            target_a = 1.0;
        }
        let soften = (1.0 + target_a) / 2.0;
        let target_b = math::pow(1.0 - b, 3.0) / 3.0;
        let hard = c;
        let wet = d;
        let mut overallscale = 1.0;
        overallscale /= 44100.0;
        overallscale *= sample_rate;
        target_a /= overallscale;

        for (out_l, out_r) in left.iter_mut().zip(right.iter_mut()) {
            let mut input_l = f64::from(*out_l);
            let mut input_r = f64::from(*out_r);
            let dry_l = input_l;
            let dry_r = input_r;

            self.increment_a = ((self.increment_a * 999.0) + target_a) / 1000.0;
            self.increment_b = ((self.increment_b * 999.0) + target_b) / 1000.0;
            self.position += self.increment_a;

            let mut output_l = self.held_sample_l;
            let mut output_r = self.held_sample_r;
            if self.position > 1.0 {
                self.position -= 1.0;
                self.held_sample_l =
                    (self.last_sample_l * self.position) + (input_l * (1.0 - self.position));
                output_l = (output_l * (1.0 - soften)) + (self.held_sample_l * soften);
                self.held_sample_r =
                    (self.last_sample_r * self.position) + (input_r * (1.0 - self.position));
                output_r = (output_r * (1.0 - soften)) + (self.held_sample_r * soften);
            }
            input_l = output_l;
            input_r = output_r;

            input_l = soften_transition(
                input_l,
                &mut self.last_output_sample_l,
                self.last_dry_sample_l,
                hard,
            );
            input_r = soften_transition(
                input_r,
                &mut self.last_output_sample_r,
                self.last_dry_sample_r,
                hard,
            );
            self.last_dry_sample_l = dry_l;
            self.last_dry_sample_r = dry_r;

            let temp_l = input_l;
            let temp_r = input_r;
            input_l = ulaw_encode(clip(input_l));
            input_r = ulaw_encode(clip(input_r));
            input_l = (temp_l * hard) + (input_l * (1.0 - hard));
            input_r = (temp_r * hard) + (input_r * (1.0 - hard));

            if self.increment_b > 0.0005 {
                input_l = quantise(input_l, self.increment_b) * (1.0 - self.increment_b);
                input_r = quantise(input_r, self.increment_b) * (1.0 - self.increment_b);
            }

            let temp_l = input_l;
            let temp_r = input_r;
            input_l = ulaw_decode(clip(input_l));
            input_r = ulaw_decode(clip(input_r));
            input_l = (temp_l * hard) + (input_l * (1.0 - hard));
            input_r = (temp_r * hard) + (input_r * (1.0 - hard));

            if wet != 1.0 {
                input_l = (input_l * wet) + (dry_l * (1.0 - wet));
                input_r = (input_r * wet) + (dry_r * (1.0 - wet));
            }

            self.last_sample_l = dry_l;
            self.last_sample_r = dry_r;

            *out_l = input_l as f32;
            *out_r = input_r as f32;
        }
    }

    /// `reset`: back to as built, knobs included, with new dither seeds.
    pub fn reset(&mut self, rand: &mut dyn FnMut() -> i32) {
        *self = Self::new(rand);
    }
}
