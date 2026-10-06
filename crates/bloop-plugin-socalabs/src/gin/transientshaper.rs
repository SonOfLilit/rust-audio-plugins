//! `gin::TransientShaper`: level-independent attack and sustain shaping,
//! from how far a fast and a slow envelope run ahead of and behind a
//! baseline.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_transientshaper.h`. bloop adds the
//! trace its screen draws (`bloop_plugin_api::trace`): the detected
//! envelope and the gain applied to the mid, read as it goes, which changes
//! nothing that comes out.

use bloop_plugin_api::trace::{Fold, Trace};

use super::biquad::Biquad;
use crate::juce::{decibels_to_gain, jlimit};
use crate::math::{self, std_max, std_min};
use core::f64::consts::PI;

/// The original's `0.70710678118654752440`, which is this as an `f64`.
const BUTTERWORTH_Q: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// `onePole`: the coefficients of a one-pole lowpass at `freq`.
fn one_pole(freq: f64, sample_rate: f64) -> (f32, f32) {
    let b = math::exp(-2.0 * PI * freq / sample_rate) as f32;
    (1.0 - b, b)
}

#[derive(Clone, Debug)]
pub struct TransientShaper {
    sample_rate: f64,
    attack: f32,
    sustain: f32,
    det_mid: f32,
    det_side: f32,
    proc_mid: f32,
    proc_side: f32,
    smoothing: f32,
    output_gain: f32,
    low_cut: f32,
    high_cut: f32,
    low_cut_active: bool,
    high_cut_active: bool,
    env: (f32, f32, f32),
    env_att: (f32, f32, f32),
    env_rel: (f32, f32, f32),
    gain: (f32, f32, f32),
    low_cut_filter: [Biquad; 2],
    high_cut_filter: [Biquad; 2],
    /// The detected envelope and the mid's gain.
    trace: Trace,
}

impl Default for TransientShaper {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            attack: 0.0,
            sustain: 0.0,
            det_mid: 0.5,
            det_side: 0.5,
            proc_mid: 0.5,
            proc_side: 0.5,
            smoothing: 0.5,
            output_gain: 1.0,
            low_cut: 0.0,
            high_cut: 0.0,
            low_cut_active: false,
            high_cut_active: false,
            env: (0.0, 0.0, 0.0),
            env_att: (0.0, 0.0, 0.0),
            env_rel: (0.0, 0.0, 0.0),
            gain: (0.0, 0.0, 0.0),
            low_cut_filter: [Biquad::default(); 2],
            high_cut_filter: [Biquad::default(); 2],
            trace: Trace::new([Fold::Max, Fold::FarthestFromOne]),
        }
    }
}

impl TransientShaper {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.recalc();
        self.reset();
    }

    pub fn set_attack(&mut self, attack: f32) {
        self.attack = attack;
    }

    pub fn set_sustain(&mut self, sustain: f32) {
        self.sustain = sustain;
    }

    pub fn set_detector_balance(&mut self, balance: f32) {
        self.det_side = std_min(balance, 0.5);
        self.det_mid = std_min(1.0 - balance, 0.5);
    }

    pub fn set_process_balance(&mut self, balance: f32) {
        self.proc_side = std_min(balance, 0.5);
        self.proc_mid = std_min(1.0 - balance, 0.5);
    }

    pub fn set_band(&mut self, low_cut: f32, high_cut: f32) {
        self.low_cut = low_cut;
        self.high_cut = high_cut;
        self.recalc();
    }

    pub fn set_smoothing(&mut self, smoothing: f32) {
        self.smoothing = jlimit(0.0, 1.0, smoothing);
        self.recalc();
    }

    pub fn set_output_gain(&mut self, db: f32) {
        self.output_gain = decibels_to_gain(db);
    }

    /// Silence the shaper and leave it as it would be after silence:
    /// bloop's Choke All Notes. The same as `reset` but for the gain, which
    /// silence brings to unity and `reset` leaves at nothing, so the next
    /// sound would swell in from zero.
    pub fn clear(&mut self) {
        self.reset();
        self.gain.2 = 1.0;
    }

    pub fn reset(&mut self) {
        self.env.2 = 0.0;
        self.env_att.2 = 0.0;
        self.env_rel.2 = 0.0;
        self.gain.2 = 0.0;
        for f in self
            .low_cut_filter
            .iter_mut()
            .chain(&mut self.high_cut_filter)
        {
            f.reset();
        }
        self.trace.reset();
    }

    /// The block's trace: for every chunk, the detected envelope's peak —
    /// the 10 Hz follower of the rectified detector that the attack and
    /// sustain are measured against, as an amplitude — and the gain applied
    /// to the mid farthest from unity.
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    fn recalc(&mut self) {
        let sr = self.sample_rate;
        (self.env.0, self.env.1) = one_pole(10.0, sr);
        (self.env_att.0, self.env_att.1) = one_pole(100.0, sr);
        (self.env_rel.0, self.env_rel.1) = one_pole(1.0, sr);
        let s = 1.0 - self.smoothing;
        let smooth_hz = math::powf(9999.9, s * s) + 0.1;
        (self.gain.0, self.gain.1) = one_pole(f64::from(smooth_hz), sr);

        let nyquist = (sr / 2.1) as f32;
        self.low_cut_active = self.low_cut > 0.0 && self.low_cut < nyquist;
        if self.low_cut_active {
            for f in &mut self.low_cut_filter {
                f.set_highpass(f64::from(self.low_cut), BUTTERWORTH_Q, sr);
            }
        }
        self.high_cut_active = self.high_cut > 0.0 && self.high_cut < nyquist;
        if self.high_cut_active {
            for f in &mut self.high_cut_filter {
                f.set_lowpass(f64::from(self.high_cut), BUTTERWORTH_Q, sr);
            }
        }
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        self.trace.begin();
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (mut in_l, mut in_r) = (*l, *r);
            let (mut dry_l, mut dry_r) = (0.0f32, 0.0f32);

            if self.high_cut_active {
                let wet_l = self.high_cut_filter[0].process(in_l);
                let wet_r = self.high_cut_filter[1].process(in_r);
                dry_l += in_l - wet_l;
                dry_r += in_r - wet_r;
                in_l = wet_l;
                in_r = wet_r;
            }
            if self.low_cut_active {
                let wet_l = self.low_cut_filter[0].process(in_l);
                let wet_r = self.low_cut_filter[1].process(in_r);
                dry_l += in_l - wet_l;
                dry_r += in_r - wet_r;
                in_l = wet_l;
                in_r = wet_r;
            }

            let mut mid = (in_l + in_r) * 0.5;
            let mut side = (in_l - in_r) * 0.5;

            let detector =
                std_max((mid * self.det_mid).abs(), (side * self.det_side).abs()) + 1.0e-5;

            self.env.2 = self.env.0 * detector + self.env.1 * self.env.2;
            self.env_att.2 = self.env_att.0 * detector + self.env_att.1 * self.env_att.2;
            self.env_rel.2 = self.env_rel.0 * detector + self.env_rel.1 * self.env_rel.2;

            let env = self.env.2.sqrt();
            let env_att = self.env_att.2.sqrt();
            let env_rel = self.env_rel.2.sqrt();

            let mut gain = math::expf(
                math::logf(std_max(env_att / env, 1.0)) * self.attack
                    + math::logf(std_max(env_rel / env, 1.0)) * self.sustain,
            );

            self.gain.2 = self.gain.0 * gain + self.gain.1 * self.gain.2;
            gain = std_min(std_max(self.gain.2, 0.0).sqrt(), gain);
            gain = std_min(gain, 64.0);

            let mid_gain = math::powf(gain, self.proc_mid);
            mid *= mid_gain;
            side *= math::powf(gain, self.proc_side);
            self.trace.sample(self.env.2, mid_gain);

            *l = (dry_l + (mid + side)) * self.output_gain;
            *r = (dry_r + (mid - side)) * self.output_gain;
        }
    }
}
