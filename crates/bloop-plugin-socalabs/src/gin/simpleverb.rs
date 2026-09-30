//! `gin::SimpleVerb`: twelve damped combs into three allpasses a side,
//! after Michael "LOSER" Gruhn's reverb (ISC; see LICENSE).
//!
//! Ported from Gin's `gin_dsp/dsp/gin_simpleverb.{h,cpp}`.
//!
//! Two things differ from the letter of the original, both where it
//! reads memory it never wrote. Its constructor compares the first
//! settings against faders it has not initialised; here they start at
//! zero, which is what a fresh allocation holds and what the gold harness
//! constructs the original into. And a sample-rate drop shrinks the
//! buffers without shortening the comb and pre-delay lengths set at the old
//! rate, so the original then reads past their ends; here the lengths are
//! held to the buffers, which changes nothing a defined run can reach.

use crate::juce::approximately_equal;
use crate::math;

const CDC: f32 = 1e-30;
const COMB_SCALES: [f32; 12] = [
    1.00, 1.09, 1.16, 1.23, 1.32, 1.41, 1.45, 1.56, 1.66, 1.71, 1.80, 1.90,
];
const COMB_GAINS: [f32; 12] = [
    0.49, 0.76, 1.00, 0.91, 0.79, 0.71, 0.59, 0.51, 0.42, 0.38, 0.35, 0.30,
];
const SW: f32 = 1.0;
/// The allpasses' lengths in milliseconds, left then right: the stereo
/// width (`SW`) is added to alternate ones.
const ALLPASS_MS: [f32; 6] = [1.0, 2.5 + SW, 5.0, 1.0 + SW, 2.5, 5.0 + SW];
const AP_FBQ: f32 = 0.6;

#[derive(Clone, Debug, Default)]
struct Ring {
    buffer: Vec<f32>,
    pos: u32,
    len: u32,
}

impl Ring {
    /// `++pos >= length ? 0 : pos`.
    fn advance(&mut self) {
        self.pos += 1;
        if self.pos >= self.len {
            self.pos = 0;
        }
    }

    fn at(&mut self) -> &mut f32 {
        &mut self.buffer[self.pos as usize]
    }

    /// Resize to `samples`, as `std::vector::resize` does: a shrink keeps
    /// the front and never allocates; a grow within capacity does not
    /// either.
    fn resize(&mut self, samples: u32) {
        self.buffer.resize(samples as usize, 0.0);
        self.len = self.len.min(samples);
    }
}

/// The allpass stage: write, read the value just written, move on.
fn allpass(ring: &mut Ring, input: f32) -> f32 {
    let stored = input + *ring.at() * AP_FBQ;
    *ring.at() = stored;
    let out = input - stored * AP_FBQ;
    ring.advance();
    out
}

#[derive(Clone, Debug)]
pub struct SimpleVerb {
    pre_delay: Ring,
    pre_delay_fader: f32,
    combs: [Ring; 12],
    /// L1, L2, L3, R1, R2, R3.
    allpasses: [Ring; 6],
    damp: f32,
    dry: f32,
    wet: f32,
    room_size: f32,
    room_size_fader: f32,
    damp_fader: f32,
    dry_fader: f32,
    wet_fader: f32,
    sample_rate: f32,
    freq_lp_fader: f32,
    freq_hp_fader: f32,
    a0_lp: f32,
    b1_lp: f32,
    tmp1_lp: f32,
    tmp2_lp: f32,
    a0_hp: f32,
    b1_hp: f32,
    tmp1_hp: f32,
    tmp2_hp: f32,
}

impl Default for SimpleVerb {
    fn default() -> Self {
        let mut verb = Self {
            pre_delay: Ring::default(),
            pre_delay_fader: 0.0,
            combs: Default::default(),
            allpasses: Default::default(),
            damp: 0.0,
            dry: 0.0,
            wet: 0.0,
            room_size: -1.0,
            room_size_fader: -1.0,
            damp_fader: -1.0,
            dry_fader: -1.0,
            wet_fader: -1.0,
            sample_rate: 44100.0,
            freq_lp_fader: 0.0,
            freq_hp_fader: 0.0,
            a0_lp: 0.0,
            b1_lp: 0.0,
            tmp1_lp: 0.0,
            tmp2_lp: 0.0,
            a0_hp: 0.0,
            b1_hp: 0.0,
            tmp1_hp: 0.0,
            tmp2_hp: 0.0,
        };
        verb.set_sample_rate(44100.0);
        verb.set_parameters(0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 1.0);
        verb
    }
}

impl SimpleVerb {
    /// `setSampleRate`: the buffers sized for the largest room and every
    /// state cleared. Allocates only if the buffers have to grow past what
    /// they have held before.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        const ROOM_MAX_SIZE: f32 = 100.0;
        self.sample_rate = sample_rate;
        for (comb, scale) in self.combs.iter_mut().zip(COMB_SCALES) {
            comb.resize((scale * ROOM_MAX_SIZE * sample_rate / 1000.0) as u32);
        }
        for (ring, ms) in self.allpasses.iter_mut().zip(ALLPASS_MS) {
            let len = (ms * sample_rate / 1000.0) as u32;
            ring.buffer.resize(len as usize, 0.0);
            ring.len = len;
        }
        self.pre_delay.resize((250.0 * sample_rate / 1000.0) as u32);
        self.flush_pre_delay();
        self.flush_buffers();
        for ring in &mut self.allpasses {
            ring.pos = 0;
        }
        self.tmp1_lp = 0.0;
        self.tmp2_lp = 0.0;
        self.tmp1_hp = 0.0;
        self.tmp2_hp = 0.0;
        for comb in &mut self.combs {
            comb.pos = 0;
        }
        self.pre_delay.pos = 0;
    }

    /// Silence the reverb — its pre-delay, combs, allpasses and filters —
    /// and keep its parameters. bloop's: the original has no call that
    /// does this short of `setSampleRate`, which may allocate.
    pub fn clear(&mut self) {
        self.flush_pre_delay();
        self.flush_buffers();
        self.tmp1_lp = 0.0;
        self.tmp2_lp = 0.0;
        self.tmp1_hp = 0.0;
        self.tmp2_hp = 0.0;
    }

    fn flush_pre_delay(&mut self) {
        self.pre_delay.buffer.fill(0.0);
    }

    fn flush_buffers(&mut self) {
        for ring in self.combs.iter_mut().chain(&mut self.allpasses) {
            ring.buffer.fill(0.0);
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the original's signature: seven faders, each its own knob"
    )]
    pub fn set_parameters(
        &mut self,
        room: f32,
        damp: f32,
        pre_delay: f32,
        lp_fader: f32,
        hp_fader: f32,
        wet: f32,
        dry: f32,
    ) {
        if !approximately_equal(room, self.room_size_fader) {
            self.room_size_fader = room;
            self.room_size = 5.0 + self.room_size_fader * self.room_size_fader * 95.0;
            for (comb, scale) in self.combs.iter_mut().zip(COMB_SCALES) {
                let len = (scale * self.room_size * self.sample_rate / 1000.0) as u32;
                comb.len = len.min(comb.buffer.len() as u32);
                comb.pos = 0;
            }
            self.flush_buffers();
        }
        if !approximately_equal(damp, self.damp_fader) {
            self.damp_fader = damp;
            self.damp = math::std_min(1.0 - self.damp_fader * self.damp_fader, 0.95);
        }
        if !approximately_equal(pre_delay, self.pre_delay_fader) {
            self.pre_delay_fader = pre_delay;
            let len = (self.pre_delay_fader * self.pre_delay_fader * 250.0 * self.sample_rate
                / 1000.0) as u32;
            self.pre_delay.len = len.min(self.pre_delay.buffer.len() as u32);
            self.pre_delay.pos = 0;
            self.flush_pre_delay();
        }
        if !approximately_equal(lp_fader, self.freq_lp_fader) {
            self.freq_lp_fader = lp_fader;
            let freq = self.freq_lp_fader * self.freq_lp_fader * self.freq_lp_fader * 24000.0;
            self.b1_lp = -math::expf(-2.0 * core::f32::consts::PI * freq / self.sample_rate);
            self.a0_lp = 1.0 + self.b1_lp;
        }
        if !approximately_equal(hp_fader, self.freq_hp_fader) {
            self.freq_hp_fader = hp_fader;
            let freq = self.freq_hp_fader * self.freq_hp_fader * self.freq_hp_fader * 24000.0;
            self.b1_hp = -math::expf(-2.0 * core::f32::consts::PI * freq / self.sample_rate);
            self.a0_hp = 1.0 + self.b1_hp;
        }
        if !approximately_equal(dry, self.dry_fader) {
            self.dry_fader = dry;
            self.dry = self.dry_fader * 2.0;
        }
        if !approximately_equal(wet, self.wet_fader) {
            self.wet_fader = wet;
            self.wet = self.wet_fader * 2.0;
        }
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (out1, out2) in left.iter_mut().zip(right.iter_mut()) {
            let (in1, in2) = (*out1, *out2);
            let input = (in1 + in2) / (1.0 + self.damp) + CDC;
            let mut reverb = if self.pre_delay.len <= 1 {
                input
            } else {
                *self.pre_delay.at() = input;
                self.pre_delay.advance();
                *self.pre_delay.at()
            };
            for (comb, gain) in self.combs.iter_mut().zip(COMB_GAINS) {
                let v = reverb * gain + *comb.at() * self.damp;
                *comb.at() = v;
            }
            for comb in &mut self.combs {
                comb.advance();
            }
            // Summed from the first comb, not from zero: 0 + -0 is +0, and
            // the original's sum keeps the sign.
            let [first, rest @ ..] = &mut self.combs;
            reverb = rest
                .iter_mut()
                .fold(*first.at(), |sum, comb| sum + *comb.at());

            let [l1, l2, l3, r1, r2, r3] = &mut self.allpasses;
            let mut l = allpass(l1, reverb);
            l = allpass(l2, l);
            l = allpass(l3, l);
            let mut r = allpass(r1, reverb);
            r = allpass(r2, r);
            r = allpass(r3, r);

            if !approximately_equal(self.freq_hp_fader, 0.0) {
                self.tmp1_hp = self.a0_hp * l - self.b1_hp * self.tmp1_hp + CDC;
                l -= self.tmp1_hp - CDC;
                self.tmp2_hp = self.a0_hp * r - self.b1_hp * self.tmp2_hp + CDC;
                r -= self.tmp2_hp - CDC;
            }
            if !approximately_equal(self.freq_lp_fader, 1.0) {
                self.tmp1_lp = self.a0_lp * l - self.b1_lp * self.tmp1_lp + CDC;
                l = self.tmp1_lp - CDC;
                self.tmp2_lp = self.a0_lp * r - self.b1_lp * self.tmp2_lp + CDC;
                r = self.tmp2_lp - CDC;
            }
            *out1 = in1 * self.dry + l * self.wet;
            *out2 = in2 * self.dry + r * self.wet;
        }
    }
}
