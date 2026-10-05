//! `gin::EnvelopeDetector` and `gin::Dynamics`: the one engine behind the
//! Compressor, Limiter, Expander and Gate.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_dynamics.{h,cpp}`, which is itself
//! after Will Pirkle's "Designing Audio Effect Plug-ins in C++". What the
//! four processors use is ported: a peak detector with digital time
//! constants, reporting in decibels, and the stereo pair linked. The
//! class's other switches — mean-square and RMS detection, analog time
//! constants, unlinked channels, lookahead, side-chain filters,
//! automatic makeup gain, the compressor-expander curve — are left behind
//! until something turns one on. So are the level trackers, which feed
//! meters and do not touch audio.

use crate::juce::{decibels_to_gain, gain_to_decibels};
use crate::math;

/// 1%: the digital time constant reaches 99% of the target in the time given.
const DIGITAL_TC: f32 = -2.0;

/// `gin::EnvelopeDetector` in peak mode, with its log detector on: attack,
/// hold and release over the rectified signal, in decibels.
#[derive(Clone, Debug)]
pub struct EnvelopeDetector {
    sample_rate: f64,
    attack_time: f32,
    release_time: f32,
    envelope: f32,
    hold_time: f32,
    hold_remaining: f32,
}

impl Default for EnvelopeDetector {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            attack_time: 0.0,
            release_time: 0.0,
            envelope: 0.0,
            hold_time: 0.0,
            hold_remaining: 0.0,
        }
    }
}

impl EnvelopeDetector {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn set_params(&mut self, attack: f32, hold: f32, release: f32) {
        self.attack_time = self.time_constant(attack);
        self.hold_time = hold;
        self.release_time = self.time_constant(release);
    }

    fn time_constant(&self, seconds: f32) -> f32 {
        math::exp(f64::from(DIGITAL_TC) / (f64::from(seconds) * self.sample_rate)) as f32
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let input = input.abs();
        if input > self.envelope {
            self.envelope = self.attack_time * (self.envelope - input) + input;
            self.hold_remaining = self.hold_time;
        } else if self.hold_time > 0.0 && self.hold_remaining > 0.0 {
            self.hold_remaining -= 1.0 / self.sample_rate as f32;
        } else {
            self.envelope = self.release_time * (self.envelope - input) + input;
        }
        self.envelope = math::std_max(0.0, self.envelope);
        if self.envelope == 0.0 {
            return -100.0;
        }
        gain_to_decibels(self.envelope)
    }
}

/// Which way the gain curve bends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicsType {
    Compressor,
    Limiter,
    Expander,
    Gate,
}

/// The square GCC makes of `std::pow (x, 2.0f)`: it rewrites the call as
/// `x * x`, which is the correctly rounded square either way.
fn squared(x: f32) -> f32 {
    x * x
}

/// `gin::Dynamics` on a linked stereo pair.
#[derive(Clone, Debug)]
pub struct Dynamics {
    envelopes: [EnvelopeDetector; 2],
    sample_rate: f64,
    curve: Curve,
    channels: usize,
    input_gain: f32,
    output_gain: f32,
    attack: f32,
    hold: f32,
    release: f32,
}

impl Default for Dynamics {
    fn default() -> Self {
        Self {
            envelopes: [EnvelopeDetector::default(), EnvelopeDetector::default()],
            sample_rate: 44100.0,
            curve: Curve {
                kind: DynamicsType::Compressor,
                threshold: 0.0,
                ratio: 0.0,
                knee: 0.0,
            },
            channels: 0,
            input_gain: 1.0,
            output_gain: 1.0,
            attack: 0.0,
            hold: 0.0,
            release: 0.0,
        }
    }
}

impl Dynamics {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for e in &mut self.envelopes[..self.channels] {
            e.set_sample_rate(sample_rate);
        }
        self.update_detector_params();
        self.reset();
    }

    /// One or two channels. A detector joining the set starts at the
    /// current sample rate, as the original's freshly allocated one does.
    pub fn set_num_channels(&mut self, channels: usize) {
        let channels = channels.min(2);
        for e in &mut self.envelopes[self.channels.min(channels)..channels] {
            *e = EnvelopeDetector::default();
            e.set_sample_rate(self.sample_rate);
        }
        self.channels = channels;
        self.update_detector_params();
    }

    pub fn set_mode(&mut self, kind: DynamicsType) {
        self.curve.kind = kind;
    }

    pub fn set_params(
        &mut self,
        attack: f32,
        hold: f32,
        release: f32,
        threshold: f32,
        ratio: f32,
        knee_width: f32,
    ) {
        self.attack = attack;
        self.hold = hold;
        self.release = release;
        self.update_detector_params();
        self.curve.threshold = threshold;
        self.curve.ratio = ratio;
        self.curve.knee = knee_width;
    }

    pub fn set_input_gain(&mut self, gain: f32) {
        self.input_gain = gain;
    }

    pub fn set_output_gain(&mut self, gain: f32) {
        self.output_gain = gain;
    }

    fn update_detector_params(&mut self) {
        for e in &mut self.envelopes[..self.channels] {
            e.set_params(self.attack, self.hold, self.release);
        }
    }

    pub fn reset(&mut self) {
        for e in &mut self.envelopes[..self.channels] {
            e.reset();
        }
    }

    /// Process `channels` in place, the detector following the signal.
    pub fn process(&mut self, channels: &mut [&mut [f32]]) {
        self.run(channels, None);
    }

    /// Process `channels` in place, the detector following `key` instead:
    /// `process (buffer, sidechain)`, the newer Gin's side-chain keying.
    /// As there, the input gain lands on the key as on the signal, and a
    /// key with fewer channels than the signal has its last drive the
    /// rest; one shorter than the block is not used at all.
    pub fn process_keyed(&mut self, channels: &mut [&mut [f32]], key: &[&[f32]]) {
        let samples = channels.first().map_or(0, |c| c.len());
        let usable = !key.is_empty() && key.iter().all(|k| k.len() >= samples);
        self.run(channels, usable.then_some(key));
    }

    fn run(&mut self, channels: &mut [&mut [f32]], key: Option<&[&[f32]]>) {
        let n = self.channels.min(channels.len());
        let Some(samples) = channels.first().map(|c| c.len()) else {
            return;
        };
        let input_gain = self.input_gain;
        let output_gain = self.output_gain;
        for i in 0..samples {
            let mut linked = 0.0f32;
            for (c, (channel, envelope)) in channels
                .iter()
                .zip(&mut self.envelopes)
                .take(n)
                .enumerate()
            {
                let heard = key.map_or(channel[i], |key| key[c.min(key.len() - 1)][i]);
                linked += decibels_to_gain(envelope.process(input_gain * heard));
            }
            linked /= self.channels as f32;
            let linked = gain_to_decibels(linked);
            let gain = decibels_to_gain(self.curve.calc(linked) - linked);
            for channel in channels.iter_mut().take(n) {
                // `* autoMakeup` too, in the original, which is 1 unless
                // automatic makeup gain is on.
                channel[i] = input_gain * gain * channel[i] * output_gain;
            }
        }
    }
}

/// The settings `calcCurve` reads.
#[derive(Clone, Copy, Debug)]
struct Curve {
    kind: DynamicsType,
    threshold: f32,
    ratio: f32,
    knee: f32,
}

impl Curve {
    /// `calcCurve`: the level out, in decibels, for a level in.
    fn calc(self, db_in: f32) -> f32 {
        let Self {
            kind,
            threshold,
            ratio,
            knee,
        } = self;
        let in_knee =
            knee > 0.0 && db_in >= (threshold - knee / 2.0) && db_in <= threshold + knee / 2.0;
        match kind {
            DynamicsType::Compressor => {
                if in_knee {
                    db_in
                        + ((1.0 / ratio - 1.0) * squared(db_in - threshold + knee / 2.0)
                            / (2.0 * knee))
                } else if f64::from(db_in) > f64::from(threshold) + f64::from(knee) / 2.0 {
                    threshold + (db_in - threshold) / ratio
                } else {
                    db_in
                }
            }
            DynamicsType::Limiter => {
                if in_knee {
                    db_in + (1.0 * squared(db_in - threshold + knee / 2.0) / (2.0 * knee))
                } else if db_in > threshold + knee / 2.0 {
                    threshold
                } else {
                    db_in
                }
            }
            DynamicsType::Expander => {
                if in_knee {
                    db_in
                        - ((ratio - 1.0) * squared(db_in - threshold - (knee / 2.0))) / (2.0 * knee)
                } else if f64::from(db_in) < f64::from(threshold) + f64::from(knee) / 2.0 {
                    threshold + (db_in - threshold) * ratio
                } else {
                    db_in
                }
            }
            DynamicsType::Gate => {
                if in_knee {
                    db_in
                        - ((100.0 - 1.0) * squared(db_in - threshold - (knee / 2.0))) / (2.0 * knee)
                } else if db_in < threshold - knee / 2.0 {
                    -1000.0
                } else {
                    db_in
                }
            }
        }
    }
}
