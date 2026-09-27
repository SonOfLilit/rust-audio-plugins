//! `gin::Exciter`: what is above a crossover, driven into a soft
//! saturator, highpassed again, and added back.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_exciter.h`.

use crate::juce::jlimit;
use crate::math::{self, std_max};
use core::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Channel {
    lp1: f32,
    lp2: f32,
}

#[derive(Clone, Debug)]
pub struct Exciter {
    sample_rate: f64,
    frequency: f32,
    drive: f32,
    harmonics: f32,
    mix: f32,
    a: f32,
    b: f32,
    chans: [Channel; 2],
}

impl Default for Exciter {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            frequency: 7000.0,
            drive: 1.0,
            harmonics: 0.0,
            mix: 0.5,
            a: 0.0,
            b: 0.0,
            chans: [Channel::default(); 2],
        }
    }
}

impl Exciter {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.recalc();
        self.reset();
    }

    pub fn set_frequency(&mut self, hz: f32) {
        self.frequency = hz;
        self.recalc();
    }

    pub fn set_drive(&mut self, drive: f32) {
        self.drive = std_max(1.0, drive);
    }

    pub fn set_harmonics(&mut self, harmonics: f32) {
        let f = jlimit(0.0, 1.0, harmonics) * 0.99;
        self.harmonics = 2.0 * f / std_max(1.0e-6, 1.0 - f.sqrt());
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix;
    }

    pub fn reset(&mut self) {
        self.chans = [Channel::default(); 2];
    }

    fn recalc(&mut self) {
        self.b = math::exp(-2.0 * PI * f64::from(self.frequency) / self.sample_rate) as f32;
        self.a = 1.0 - self.b;
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (c, data) in self.chans.iter_mut().zip([left, right]) {
            for x in data.iter_mut() {
                let dry = *x;
                c.lp1 = self.a * dry + self.b * c.lp1;
                let mut s = dry - c.lp1;
                s = jlimit(-1.0, 1.0, s * self.drive);
                s = (1.0 + self.harmonics) * s / (1.0 + self.harmonics * s.abs());
                c.lp2 = self.a * s + self.b * c.lp2;
                s -= c.lp2;
                *x = dry + s * self.mix;
            }
        }
    }
}
