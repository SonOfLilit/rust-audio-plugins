//! `gin::EnvelopeDetector` and `gin::Dynamics`: the one engine behind the
//! Compressor, Limiter, Expander and Gate.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_dynamics.{h,cpp}`, which is itself
//! after Will Pirkle's "Designing Audio Effect Plug-ins in C++". The
//! processors that use it drive the detector from the signal itself, in
//! peak mode, with channels linked; lookahead, side-chain keying and the
//! detector's filters are features of the class those processors never
//! switch on, and are not ported. Nor are the level trackers, which feed
//! meters and do not touch audio.

use crate::juce::{decibels_to_gain, gain_to_decibels};
use crate::math;

/// 1%: the digital time constant reaches 99% of the target in the time given.
const DIGITAL_TC: f32 = -2.0;
/// 36.7%: the analog one reaches 63%, as a capacitor does.
const ANALOG_TC: f32 = -0.435_333_94;

/// How the detector reads level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorMode {
    Peak,
    MeanSquare,
    Rms,
}

/// `gin::EnvelopeDetector`: attack, hold and release over the rectified
/// signal, reported in decibels when `log_detector` is set.
#[derive(Clone, Debug)]
pub struct EnvelopeDetector {
    sample_rate: f64,
    mode: DetectorMode,
    attack_time: f32,
    release_time: f32,
    envelope: f32,
    hold_time: f32,
    hold_remaining: f32,
    analog_tc: bool,
    log_detector: bool,
    rms_window: f32,
    rms_a: f32,
    rms_b: f32,
    rms_state: f32,
}

impl Default for EnvelopeDetector {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            mode: DetectorMode::Peak,
            attack_time: 0.0,
            release_time: 0.0,
            envelope: 0.0,
            hold_time: 0.0,
            hold_remaining: 0.0,
            analog_tc: false,
            log_detector: false,
            rms_window: 0.010,
            rms_a: 1.0,
            rms_b: 0.0,
            rms_state: 0.0,
        }
    }
}

impl EnvelopeDetector {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.set_rms_window(self.rms_window);
    }

    pub fn set_rms_window(&mut self, seconds: f32) {
        self.rms_window = seconds;
        if seconds <= 0.0 {
            self.rms_a = 1.0;
            self.rms_b = 0.0;
        } else {
            self.rms_b = math::exp(-1.0 / (f64::from(seconds) * self.sample_rate)) as f32;
            self.rms_a = 1.0 - self.rms_b;
        }
    }

    pub fn set_params(
        &mut self,
        attack: f32,
        hold: f32,
        release: f32,
        analog_tc: bool,
        mode: DetectorMode,
        log_detector: bool,
    ) {
        self.analog_tc = analog_tc;
        self.mode = mode;
        self.log_detector = log_detector;
        self.attack_time = self.time_constant(attack);
        self.hold_time = hold;
        self.release_time = self.time_constant(release);
    }

    fn time_constant(&self, seconds: f32) -> f32 {
        let tc = if self.analog_tc {
            ANALOG_TC
        } else {
            DIGITAL_TC
        };
        math::exp(f64::from(tc) / (f64::from(seconds) * self.sample_rate)) as f32
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.rms_state = 0.0;
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let input = match self.mode {
            DetectorMode::Peak => input.abs(),
            DetectorMode::MeanSquare => {
                self.rms_state = self.rms_a * (input * input) + self.rms_b * self.rms_state;
                self.rms_state
            }
            DetectorMode::Rms => {
                self.rms_state = self.rms_a * (input * input) + self.rms_b * self.rms_state;
                self.rms_state.sqrt()
            }
        };
        if input > self.envelope {
            self.envelope = self.attack_time * (self.envelope - input) + input;
            self.hold_remaining = self.hold_time;
        } else if self.hold_time > 0.0 && self.hold_remaining > 0.0 {
            self.hold_remaining -= 1.0 / self.sample_rate as f32;
        } else {
            self.envelope = self.release_time * (self.envelope - input) + input;
        }
        self.envelope = math::std_max(0.0, self.envelope);
        if self.log_detector {
            if self.envelope == 0.0 {
                return -100.0;
            }
            return gain_to_decibels(self.envelope);
        }
        self.envelope
    }
}

/// Which way the gain curve bends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicsType {
    Compressor,
    Limiter,
    Expander,
    Gate,
    /// Downward compression above the threshold and upward below its
    /// mirror image, at one ratio.
    CompExp,
}

/// The square GCC makes of `std::pow (x, 2.0f)`: it rewrites the call as
/// `x * x`, which is the correctly rounded square either way.
fn squared(x: f32) -> f32 {
    x * x
}

/// `gin::Dynamics` for a stereo pair, linked or not.
#[derive(Clone, Debug)]
pub struct Dynamics {
    envelopes: [EnvelopeDetector; 2],
    sample_rate: f64,
    kind: DynamicsType,
    channels: usize,
    linked: bool,
    auto_makeup_gain: bool,
    input_gain: f32,
    output_gain: f32,
    threshold: f32,
    ratio: f32,
    knee_width: f32,
    attack: f32,
    hold: f32,
    release: f32,
    detector_mode: DetectorMode,
    rms_window: f32,
    analog_tc: bool,
}

impl Default for Dynamics {
    fn default() -> Self {
        Self {
            envelopes: [EnvelopeDetector::default(), EnvelopeDetector::default()],
            sample_rate: 44100.0,
            kind: DynamicsType::Compressor,
            channels: 0,
            linked: true,
            auto_makeup_gain: false,
            input_gain: 1.0,
            output_gain: 1.0,
            threshold: 0.0,
            ratio: 0.0,
            knee_width: 0.0,
            attack: 0.0,
            hold: 0.0,
            release: 0.0,
            detector_mode: DetectorMode::Peak,
            rms_window: 0.010,
            analog_tc: false,
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
        self.kind = kind;
    }

    pub fn set_linked(&mut self, linked: bool) {
        self.linked = linked;
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
        self.threshold = threshold;
        self.ratio = ratio;
        self.knee_width = knee_width;
    }

    pub fn set_input_gain(&mut self, gain: f32) {
        self.input_gain = gain;
    }

    pub fn set_output_gain(&mut self, gain: f32) {
        self.output_gain = gain;
    }

    pub fn set_detector_mode(&mut self, mode: DetectorMode) {
        self.detector_mode = mode;
        self.update_detector_params();
    }

    pub fn set_rms_window(&mut self, seconds: f32) {
        self.rms_window = seconds;
        self.update_detector_params();
    }

    pub fn set_analog_tc(&mut self, analog: bool) {
        self.analog_tc = analog;
        self.update_detector_params();
    }

    pub fn set_auto_makeup_gain(&mut self, enabled: bool) {
        self.auto_makeup_gain = enabled;
    }

    fn update_detector_params(&mut self) {
        for e in &mut self.envelopes[..self.channels] {
            e.set_rms_window(self.rms_window);
            e.set_params(
                self.attack,
                self.hold,
                self.release,
                self.analog_tc,
                self.detector_mode,
                true,
            );
        }
    }

    pub fn reset(&mut self) {
        for e in &mut self.envelopes[..self.channels] {
            e.reset();
        }
    }

    /// Process `channels` in place, the detector following the signal.
    pub fn process(&mut self, channels: &mut [&mut [f32]]) {
        let n = self.channels.min(channels.len());
        let Some(samples) = channels.first().map(|c| c.len()) else {
            return;
        };
        let auto_makeup = if self.auto_makeup_gain
            && matches!(self.kind, DynamicsType::Compressor | DynamicsType::Limiter)
        {
            let makeup_db = if self.kind == DynamicsType::Limiter {
                -self.threshold
            } else if self.ratio > 1.0 {
                -self.threshold * (1.0 - 1.0 / self.ratio)
            } else {
                0.0
            };
            decibels_to_gain(makeup_db)
        } else {
            1.0
        };
        let input_gain = self.input_gain;
        let output_gain = self.output_gain;
        for i in 0..samples {
            if self.linked {
                let mut linked = 0.0f32;
                for (channel, envelope) in channels.iter().zip(&mut self.envelopes).take(n) {
                    linked += decibels_to_gain(envelope.process(input_gain * channel[i]));
                }
                linked /= self.channels as f32;
                let linked = gain_to_decibels(linked);
                let gain = decibels_to_gain(self.curve().calc(linked) - linked);
                for channel in channels.iter_mut().take(n) {
                    channel[i] = input_gain * gain * channel[i] * output_gain * auto_makeup;
                }
            } else {
                let curve = self.curve();
                for (channel, envelope) in channels.iter_mut().zip(&mut self.envelopes).take(n) {
                    let level = envelope.process(input_gain * channel[i]);
                    let gain = decibels_to_gain(curve.calc(level) - level);
                    channel[i] = input_gain * gain * channel[i] * output_gain * auto_makeup;
                }
            }
        }
    }

    fn curve(&self) -> Curve {
        Curve {
            kind: self.kind,
            threshold: self.threshold,
            ratio: self.ratio,
            knee: self.knee_width,
        }
    }

    /// `calcCurve`: the level out, in decibels, for a level in.
    #[must_use]
    pub fn calc_curve(&self, db_in: f32) -> f32 {
        self.curve().calc(db_in)
    }
}

/// The settings `calcCurve` reads, apart from the detectors it runs beside.
#[derive(Clone, Copy, Debug)]
struct Curve {
    kind: DynamicsType,
    threshold: f32,
    ratio: f32,
    knee: f32,
}

impl Curve {
    fn calc(self, db_in: f32) -> f32 {
        let Self {
            kind,
            threshold,
            ratio,
            knee,
        } = self;
        let in_knee = |t: f32| knee > 0.0 && db_in >= (t - knee / 2.0) && db_in <= t + knee / 2.0;
        match kind {
            DynamicsType::Compressor => {
                if in_knee(threshold) {
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
                if in_knee(threshold) {
                    db_in + (1.0 * squared(db_in - threshold + knee / 2.0) / (2.0 * knee))
                } else if db_in > threshold + knee / 2.0 {
                    threshold
                } else {
                    db_in
                }
            }
            DynamicsType::Expander => {
                if in_knee(threshold) {
                    db_in
                        - ((ratio - 1.0) * squared(db_in - threshold - (knee / 2.0))) / (2.0 * knee)
                } else if f64::from(db_in) < f64::from(threshold) + f64::from(knee) / 2.0 {
                    threshold + (db_in - threshold) * ratio
                } else {
                    db_in
                }
            }
            DynamicsType::Gate => {
                if in_knee(threshold) {
                    db_in
                        - ((100.0 - 1.0) * squared(db_in - threshold - (knee / 2.0))) / (2.0 * knee)
                } else if db_in < threshold - knee / 2.0 {
                    -1000.0
                } else {
                    db_in
                }
            }
            DynamicsType::CompExp => {
                let exp_threshold = -100.0 - threshold;
                if in_knee(threshold) {
                    db_in
                        + ((1.0 / ratio - 1.0) * squared(db_in - threshold + knee / 2.0)
                            / (2.0 * knee))
                } else if db_in > threshold + knee / 2.0 {
                    threshold + (db_in - threshold) / ratio
                } else if in_knee(exp_threshold) {
                    db_in
                        - ((ratio - 1.0) * squared(db_in - exp_threshold - (knee / 2.0)))
                            / (2.0 * knee)
                } else if db_in < exp_threshold - knee / 2.0 {
                    exp_threshold + (db_in - exp_threshold) * ratio
                } else {
                    db_in
                }
            }
        }
    }
}
