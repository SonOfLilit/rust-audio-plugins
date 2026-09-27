//! `gin::BitCrusher`: bit depth and sample rate reduction, the one
//! anti-aliased by 4x oversampling, the other by polyBLEP steps; or, in
//! dirty mode, both done naively.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_bitcrusher.h`, except for the
//! oversampling, which in the original is `juce::dsp::Oversampling` and
//! here is `super::oversampler` (see there for why). Dirty mode never
//! touches it and is bit-exact against the original; the anti-aliased mode
//! is the original's algorithm around a different oversampler.

use super::oversampler::Oversampler;
use super::param::ValueSmoother;
use crate::juce::jlimit;
use crate::math;

const OVER_ORDER: u32 = 2;
const MAX_BITS: f32 = 16.0;
/// The largest block processed in one go: longer ones are taken in pieces
/// of this, as the original does.
const MAX_BLOCK: usize = 4096;

#[derive(Clone, Debug)]
pub struct BitCrusher {
    sample_rate: f64,
    target_bits: f32,
    target_rate: f32,
    dirty: bool,
    last_dirty: bool,
    oversampler: Oversampler,
    bits: ValueSmoother,
    bits_os: ValueSmoother,
    rate: ValueSmoother,
    phase: f32,
    held: [f32; 2],
    prev_input: [f32; 2],
    delayed: [f32; 2],
}

impl Default for BitCrusher {
    fn default() -> Self {
        let mut crusher = Self {
            sample_rate: 44100.0,
            target_bits: 16.0,
            target_rate: 44100.0,
            dirty: false,
            last_dirty: false,
            oversampler: Oversampler::default(),
            bits: ValueSmoother::default(),
            bits_os: ValueSmoother::default(),
            rate: ValueSmoother::default(),
            phase: 1.0,
            held: [0.0; 2],
            prev_input: [0.0; 2],
            delayed: [0.0; 2],
        };
        crusher.bits.set_time(0.02);
        crusher.bits_os.set_time(0.02);
        crusher.rate.set_time(0.02);
        crusher.set_sample_rate(44100.0);
        crusher
    }
}

/// `quantise`: `v` on a grid of `2^(bits-1)` steps a side.
fn quantise(v: f32, bits: f32) -> f32 {
    let scale = math::exp2f(bits - 1.0);
    (v * scale).round() / scale
}

impl BitCrusher {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.bits.set_sample_rate(sample_rate);
        self.bits_os
            .set_sample_rate(sample_rate * f64::from(1u32 << OVER_ORDER));
        self.rate.set_sample_rate(sample_rate);
        self.set_params(self.target_bits, self.target_rate, self.dirty);
        self.reset();
    }

    pub fn set_params(&mut self, bit_depth: f32, rate_hz: f32, dirty: bool) {
        self.target_bits = bit_depth;
        self.target_rate = rate_hz;
        self.dirty = dirty;
        self.bits.set_value(bit_depth / MAX_BITS);
        self.bits_os.set_value(bit_depth / MAX_BITS);
        self.rate
            .set_value((f64::from(rate_hz) / self.sample_rate) as f32);
    }

    pub fn reset(&mut self) {
        self.oversampler.reset();
        self.phase = 1.0;
        self.held = [0.0; 2];
        self.prev_input = [0.0; 2];
        self.delayed = [0.0; 2];
        self.bits.snap_to_value();
        self.bits_os.snap_to_value();
        self.rate.snap_to_value();
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        if self.dirty != self.last_dirty {
            self.last_dirty = self.dirty;
            self.oversampler.reset();
            self.delayed = self.held;
            self.prev_input = self.held;
        }
        for (l, r) in left.chunks_mut(MAX_BLOCK).zip(right.chunks_mut(MAX_BLOCK)) {
            if self.dirty {
                self.process_naive([l, r]);
            } else {
                self.process_anti_aliased([l, r]);
            }
        }
    }

    fn process_naive(&mut self, channels: [&mut [f32]; 2]) {
        let [left, right] = channels;
        let samples = left.len().min(right.len());
        // Kept in step so a switch out of dirty mode does not jump.
        self.bits_os.process(samples << OVER_ORDER);
        for i in 0..samples {
            let bits = self.bits.next_value() * MAX_BITS;
            let inc = self.rate.next_value();
            self.phase += inc;
            let step = self.phase >= 1.0;
            if step {
                self.phase -= self.phase.floor();
            }
            for (ch, d) in [&mut *left, &mut *right].into_iter().enumerate() {
                if step {
                    self.held[ch] = quantise(d[i], bits);
                }
                d[i] = self.held[ch];
            }
        }
    }

    fn process_anti_aliased(&mut self, channels: [&mut [f32]; 2]) {
        let [left, right] = channels;
        let samples = left.len().min(right.len());

        // Quantise at the oversampled rate, one oversampled sample of both
        // channels at a time as the original's loop over the upsampled block
        // does.
        for i in 0..samples {
            let mut four = [
                self.oversampler.up(0, left[i]),
                self.oversampler.up(1, right[i]),
            ];
            for j in 0..4 {
                let bits = self.bits_os.next_value() * MAX_BITS;
                for channel in &mut four {
                    channel[j] = quantise(channel[j], bits);
                }
            }
            left[i] = self.oversampler.down(0, four[0]);
            right[i] = self.oversampler.down(1, four[1]);
        }
        self.bits.process(samples);

        // The polyBLEP-corrected sample and hold, at the project rate.
        for i in 0..samples {
            let inc = self.rate.next_value();
            if inc >= 1.0 {
                // At or above the project rate: straight through, one
                // sample late so the latency does not change.
                self.phase = 1.0;
                for (ch, d) in [&mut *left, &mut *right].into_iter().enumerate() {
                    core::mem::swap(&mut d[i], &mut self.delayed[ch]);
                    let input = self.delayed[ch];
                    self.held[ch] = input;
                    self.prev_input[ch] = input;
                }
                continue;
            }
            self.phase += inc;
            let step = self.phase >= 1.0;
            let mut t = 0.0f32;
            if step {
                self.phase -= 1.0;
                t = jlimit(0.0, 1.0, self.phase / inc);
            }
            for (ch, d) in [&mut *left, &mut *right].into_iter().enumerate() {
                let input = d[i];
                let mut out = self.held[ch];
                if step {
                    let new_held = self.prev_input[ch] + (1.0 - t) * (input - self.prev_input[ch]);
                    let amt = new_held - self.held[ch];
                    self.delayed[ch] += 0.5 * amt * t * t;
                    out = new_held - 0.5 * amt * (1.0 - t) * (1.0 - t);
                    self.held[ch] = new_held;
                }
                d[i] = self.delayed[ch];
                self.delayed[ch] = out;
                self.prev_input[ch] = input;
            }
        }
    }
}
