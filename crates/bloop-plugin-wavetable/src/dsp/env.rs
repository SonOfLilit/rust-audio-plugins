//! The envelopes: Gin's linear `ADSR` (`gin_adsr.{h,cpp}`), which drives
//! the filter and the three modulation envelopes, and its exponential
//! `AnalogADSR` (`gin_analogadsr.{h,cpp}`), which drives the amplifier and
//! the gate effect.

use bloop_plugin_socalabs::shared::{approximately_equal, math};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdsrState {
    #[default]
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
    Finished,
}

/// `gin::ADSR`: straight lines at a fixed rate per stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adsr {
    state: AdsrState,
    sample_rate: f64,
    output: f32,
    attack_delta: f32,
    decay_delta: f32,
    release_delta: f32,
    sustain_level: f32,
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            state: AdsrState::Idle,
            sample_rate: 44100.0,
            output: 0.0,
            attack_delta: 0.0,
            decay_delta: 0.0,
            release_delta: 0.0,
            sustain_level: 0.0,
        }
    }
}

fn delta(seconds: f32, sample_rate: f64) -> f32 {
    if seconds > 0.0 {
        1.0 / (f64::from(seconds) * sample_rate) as f32
    } else {
        0.0
    }
}

impl Adsr {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn set_attack(&mut self, seconds: f32) {
        self.attack_delta = delta(seconds, self.sample_rate);
    }

    pub fn set_decay(&mut self, seconds: f32) {
        self.decay_delta = delta(seconds, self.sample_rate);
    }

    pub fn set_release(&mut self, seconds: f32) {
        self.release_delta = delta(seconds, self.sample_rate);
    }

    pub fn set_sustain_level(&mut self, level: f32) {
        self.sustain_level = level;
    }

    pub fn note_on(&mut self) {
        self.state = AdsrState::Attack;
    }

    pub fn note_off(&mut self) {
        self.state = AdsrState::Release;
    }

    pub fn reset(&mut self) {
        self.state = AdsrState::Idle;
        self.output = 0.0;
    }

    #[must_use]
    pub fn state(&self) -> AdsrState {
        self.state
    }

    /// `getOutput`: an envelope never struck reads full if it has no
    /// attack.
    #[must_use]
    pub fn output(&self) -> f32 {
        if self.state == AdsrState::Idle {
            return if self.attack_delta == 0.0 { 1.0 } else { 0.0 };
        }
        self.output
    }

    fn step(&mut self) -> f32 {
        match self.state {
            AdsrState::Idle | AdsrState::Sustain | AdsrState::Finished => {}
            AdsrState::Attack => {
                if self.attack_delta > 0.0 {
                    self.output += self.attack_delta;
                } else {
                    self.output = 1.0;
                }
                if self.output >= 1.0 {
                    self.output = 1.0;
                    self.state = AdsrState::Decay;
                }
            }
            AdsrState::Decay => {
                if self.decay_delta > 0.0 {
                    self.output -= self.decay_delta;
                } else {
                    self.output = self.sustain_level;
                }
                if self.output <= self.sustain_level {
                    self.output = self.sustain_level;
                    self.state = AdsrState::Sustain;
                }
            }
            AdsrState::Release => {
                if self.release_delta > 0.0 {
                    self.output -= self.release_delta;
                } else {
                    self.output = 0.0;
                }
                if self.output <= 0.0 {
                    self.output = 0.0;
                    self.state = AdsrState::Finished;
                }
            }
        }
        self.output
    }

    /// `process (num)`: `num` samples on.
    pub fn process(&mut self, num: usize) -> f32 {
        if self.state == AdsrState::Idle {
            return if self.attack_delta == 0.0 { 1.0 } else { 0.0 };
        }
        if self.state == AdsrState::Finished {
            return 0.0;
        }
        for _ in 0..num {
            self.step();
        }
        self.output
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnalogState {
    #[default]
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// `gin::AnalogADSR`: one-pole curves that overshoot their targets, so
/// each stage ends in finite time.
///
/// A stage's curve is worked out when its time is set to something new
/// (approximately), at the sample rate of that moment; setting the rate
/// alone recomputes nothing, as in the original.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalogAdsr {
    state: AnalogState,
    sample_rate: f64,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    attack_coeff: f32,
    decay_coeff: f32,
    release_coeff: f32,
    attack_offset: f32,
    decay_offset: f32,
    release_offset: f32,
    output: f32,
}

impl Default for AnalogAdsr {
    fn default() -> Self {
        let mut adsr = Self {
            state: AnalogState::Idle,
            sample_rate: 44100.0,
            attack: 0.0,
            decay: 0.0,
            sustain: 0.0,
            release: 0.0,
            attack_coeff: 0.0,
            decay_coeff: 0.0,
            release_coeff: 0.0,
            attack_offset: 0.0,
            decay_offset: 0.0,
            release_offset: 0.0,
            output: 0.0,
        };
        adsr.set_attack(0.2);
        adsr.set_decay(0.2);
        adsr.set_release(0.2);
        adsr.set_sustain_level(0.8);
        adsr.reset();
        adsr
    }
}

/// The coefficient and offset of a one-pole stage `seconds` long that
/// aims `tco` past its target.
fn curve(sample_rate: f64, seconds: f32, tco: f32) -> f32 {
    let samples = (sample_rate * f64::from(seconds)) as f32;
    math::expf(-math::logf((1.0 + tco) / tco) / samples)
}

impl AnalogAdsr {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn set_attack(&mut self, seconds: f32) {
        if !approximately_equal(self.attack, seconds) {
            self.attack = seconds;
            self.calculate_attack();
        }
    }

    pub fn set_decay(&mut self, seconds: f32) {
        if !approximately_equal(self.decay, seconds) {
            self.decay = seconds;
            self.calculate_decay();
        }
    }

    pub fn set_release(&mut self, seconds: f32) {
        if !approximately_equal(self.release, seconds) {
            self.release = seconds;
            self.calculate_release();
        }
    }

    pub fn set_sustain_level(&mut self, level: f32) {
        if !approximately_equal(self.sustain, level) {
            self.sustain = level;
            self.calculate_decay();
            if self.state != AnalogState::Release {
                self.calculate_release();
            }
        }
    }

    fn calculate_attack(&mut self) {
        let tco = math::expf(-0.5);
        self.attack_coeff = curve(self.sample_rate, self.attack, tco);
        self.attack_offset = (1.0 + tco) * (1.0 - self.attack_coeff);
    }

    fn calculate_decay(&mut self) {
        let tco = math::expf(-5.0);
        self.decay_coeff = curve(self.sample_rate, self.decay, tco);
        self.decay_offset = (self.sustain - tco) * (1.0 - self.decay_coeff);
    }

    fn calculate_release(&mut self) {
        let tco = math::expf(-5.0);
        self.release_coeff = curve(self.sample_rate, self.release, tco);
        self.release_offset = -tco * (1.0 - self.release_coeff);
    }

    pub fn note_on(&mut self) {
        self.calculate_release();
        let was = self.state;
        if self.attack == 0.0 {
            self.state = AnalogState::Decay;
            self.output = 1.0;
        } else {
            self.state = AnalogState::Attack;
            if was == AnalogState::Idle {
                self.output = 0.0;
            }
        }
    }

    pub fn note_off(&mut self) {
        self.state = if self.output > 0.0 {
            AnalogState::Release
        } else {
            AnalogState::Idle
        };
    }

    pub fn reset(&mut self) {
        self.state = AnalogState::Idle;
        self.output = 0.0;
    }

    #[must_use]
    pub fn state(&self) -> AnalogState {
        self.state
    }

    #[must_use]
    pub fn output(&self) -> f32 {
        self.output
    }

    /// `process()`: one sample on.
    #[inline]
    pub fn process(&mut self) -> f32 {
        match self.state {
            AnalogState::Idle | AnalogState::Sustain => {}
            AnalogState::Attack => {
                self.output = self.attack_offset + self.output * self.attack_coeff;
                if self.output >= 1.0 || self.attack == 0.0 {
                    self.output = 1.0;
                    self.state = AnalogState::Decay;
                }
            }
            AnalogState::Decay => {
                self.output = self.decay_offset + self.output * self.decay_coeff;
                if self.output <= self.sustain {
                    self.state = AnalogState::Sustain;
                }
            }
            AnalogState::Release => {
                self.output = self.release_offset + self.output * self.release_coeff;
                if self.output <= 0.0 || self.release == 0.0 {
                    self.output = 0.0;
                    self.state = AnalogState::Idle;
                }
            }
        }
        self.output
    }

    /// `processMultiplying`: each frame of `left`/`right` scaled by the
    /// envelope.
    pub fn process_multiplying(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let gain = self.process();
            *l *= gain;
            *r *= gain;
        }
    }
}
