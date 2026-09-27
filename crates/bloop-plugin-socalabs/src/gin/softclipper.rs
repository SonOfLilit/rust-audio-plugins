//! `gin::SoftClipper`: drive into a quadratic knee that flattens out at a
//! ceiling.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_softclipper.h`.

use crate::juce::{decibels_to_gain, gain_to_decibels, jlimit};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftClipper {
    drive_db: f32,
    drive_gain: f32,
    knee: f32,
    ceiling_gain: f32,
    knee_start: f32,
    knee_end: f32,
}

impl Default for SoftClipper {
    fn default() -> Self {
        Self {
            drive_db: 0.0,
            drive_gain: 1.0,
            knee: 0.5,
            ceiling_gain: 1.0,
            knee_start: 0.0,
            knee_end: 0.0,
        }
    }
}

impl SoftClipper {
    pub fn set_drive(&mut self, db: f32) {
        self.drive_db = db;
        self.drive_gain = decibels_to_gain(db);
        self.recalc();
    }

    pub fn set_knee(&mut self, knee: f32) {
        self.knee = jlimit(0.0, 1.0, knee);
        self.recalc();
    }

    pub fn set_ceiling(&mut self, db: f32) {
        self.ceiling_gain = decibels_to_gain(db);
    }

    fn recalc(&mut self) {
        self.knee_start = -self.knee * self.drive_db;
        self.knee_end = self.knee * self.drive_db;
    }

    #[must_use]
    pub fn process_sample(&self, x: f32) -> f32 {
        let driven = x * self.drive_gain;
        let mag = driven.abs();
        if mag < 1.0e-10 {
            return 0.0;
        }
        let db = gain_to_decibels(mag);
        if db <= self.knee_start {
            return driven * self.ceiling_gain;
        }
        let gain_db = if db < self.knee_end && self.knee_end > self.knee_start {
            let t = 1.0 - (db - self.knee_start) / (self.knee_end - self.knee_start);
            self.knee_start * t * t - db
        } else {
            -db
        };
        driven * decibels_to_gain(gain_db) * self.ceiling_gain
    }

    pub fn process(&self, left: &mut [f32], right: &mut [f32]) {
        for x in left.iter_mut().chain(right.iter_mut()) {
            *x = self.process_sample(*x);
        }
    }
}
