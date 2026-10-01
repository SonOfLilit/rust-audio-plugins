//! `gin::Modulation`: a delay line read back at a time a sine sweeps — a
//! chorus, a flanger or a vibrato, depending on how it is set.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_modulation.h`. SocaLabs' Wavetable
//! runs its chorus through this class (`PluginProcessor.cpp`, `chorus
//! { 0.5f }`), which is where bloop's Chorus takes it from.

use super::audioutil::WetDryMix;
use super::delay::DelayLine;
use crate::math;

/// The settings the class is constructed with, before `setParams`.
const DEFAULT_DELAY: f32 = 0.02;
const DEFAULT_RATE: f32 = 1.0;
const DEFAULT_DEPTH: f32 = 0.03;
const DEFAULT_WIDTH: f32 = 0.5;
const DEFAULT_MIX: f32 = 0.0;

#[derive(Clone, Debug)]
pub struct Modulation {
    max_delay: f32,
    line: DelayLine,
    sample_rate: f64,
    phase: f32,
    delay: f32,
    rate: f32,
    depth: f32,
    width: f32,
    mix: f32,
}

impl Modulation {
    /// `Modulation (maxDelay)`. The line starts at the class's own
    /// default, a millisecond; `set_sample_rate` sizes it.
    #[must_use]
    pub fn new(max_delay: f32) -> Self {
        Self {
            max_delay,
            line: DelayLine::new(2, f64::from(0.001f32), 44100.0),
            sample_rate: 44100.0,
            phase: 0.0,
            delay: DEFAULT_DELAY,
            rate: DEFAULT_RATE,
            depth: DEFAULT_DEPTH,
            width: DEFAULT_WIDTH,
            mix: DEFAULT_MIX,
        }
    }

    /// `setParams` with the class's own settings, then `setSampleRate`.
    ///
    /// The class sizes its line from whatever depth was set last, so the
    /// defaults go back first: the line is then the size a fresh one's
    /// first `setSampleRate` makes it — Wavetable's, which prepares before
    /// it sets anything — whatever the knobs did since, and preparing again
    /// at the same rate does not build it again. The next `set_params`,
    /// once a block, puts the knobs' values back.
    pub fn prepare(&mut self, sample_rate: f64) {
        self.set_params(
            DEFAULT_DELAY,
            DEFAULT_RATE,
            DEFAULT_DEPTH,
            DEFAULT_WIDTH,
            DEFAULT_MIX,
        );
        self.set_sample_rate(sample_rate);
    }

    /// `setSampleRate`: the line sized for the longest delay plus whatever
    /// depth is set now, and a tenth of a second more. Allocates, unless
    /// the line is already that size.
    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        let room = f64::from(self.max_delay + self.depth) + 0.1;
        if self.line.has_size(2, room, sample_rate) {
            self.line.clear();
        } else {
            self.line.set_size(2, room, sample_rate);
        }
        self.phase = 0.0;
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.line.clear();
    }

    /// The line emptied and the sweep left where it is: what silence
    /// would have left behind. Not in the original, whose `reset` also
    /// starts the sweep again.
    pub fn clear(&mut self) {
        self.line.silence();
    }

    /// `setParams`: seconds, Hz, seconds, 0 to 1, 0 to 1.
    pub fn set_params(&mut self, delay: f32, rate: f32, depth: f32, width: f32, mix: f32) {
        self.delay = delay;
        self.rate = rate;
        self.depth = depth;
        self.width = width;
        self.mix = mix;
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let wet_dry = WetDryMix::new(self.mix);
        let delta = (f64::from(self.rate) / self.sample_rate) as f32;
        let turn = core::f32::consts::PI * 2.0;
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let swing_l = math::sinf(turn * (self.phase + self.width / 2.0));
            let swing_r = math::sinf(turn * (self.phase - self.width / 2.0));
            let total_l = self.delay + self.depth / 2.0 + self.depth * swing_l / 2.0;
            let total_r = self.delay + self.depth / 2.0 + self.depth * swing_r / 2.0;

            self.line.write(0, *l);
            self.line.write(1, *r);
            self.line.write_finished();

            *l = self.line.read_lagrange(0, f64::from(total_l)) * wet_dry.wet_gain
                + *l * wet_dry.dry_gain;
            *r = self.line.read_lagrange(1, f64::from(total_r)) * wet_dry.wet_gain
                + *r * wet_dry.dry_gain;

            self.phase += delta;
            if self.phase >= 1.0 {
                self.phase = 0.0;
            }
        }
    }
}
