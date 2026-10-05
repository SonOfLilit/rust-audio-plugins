//! Gin's `LFO` (`gin_lfo.h`) and `StepLFO` (`gin_steplfo.h`).

use super::random::JuceRandom;
use bloop_plugin_socalabs::shared::{jlimit, math};

/// `gin::LFO::WaveShape`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shape {
    None,
    #[default]
    Sine,
    Triangle,
    SawUp,
    SawDown,
    Square,
    SquarePos,
    SampleAndHold,
    Noise,
    StepUp3,
    StepUp4,
    StepUp8,
    StepDown3,
    StepDown4,
    StepDown8,
    Pyramid3,
    Pyramid5,
    Pyramid9,
}

impl Shape {
    /// The shape a Wave knob's value names.
    #[must_use]
    pub fn from_index(index: i32) -> Self {
        const ALL: [Shape; 18] = [
            Shape::None,
            Shape::Sine,
            Shape::Triangle,
            Shape::SawUp,
            Shape::SawDown,
            Shape::Square,
            Shape::SquarePos,
            Shape::SampleAndHold,
            Shape::Noise,
            Shape::StepUp3,
            Shape::StepUp4,
            Shape::StepUp8,
            Shape::StepDown3,
            Shape::StepDown4,
            Shape::StepDown8,
            Shape::Pyramid3,
            Shape::Pyramid5,
            Shape::Pyramid9,
        ];
        ALL.get(usize::try_from(index).unwrap_or(0))
            .copied()
            .unwrap_or(Shape::None)
    }

    fn is_random(self) -> bool {
        matches!(self, Self::SampleAndHold | Self::Noise)
    }
}

/// `LFO::Parameters`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LfoParams {
    pub shape: Shape,
    pub frequency: f32,
    pub phase: f32,
    pub offset: f32,
    pub depth: f32,
    pub delay: f32,
    pub fade: f32,
}

/// `maxRandomPhase`: how many random points a sample-and-hold walks.
const MAX_RANDOM_PHASE: usize = 1000;

/// The thousand points every LFO's sample-and-hold and noise walk,
/// `juce::Random { 1 }`'s first draws. Worked out once, on first use,
/// which [`Lfo::warm`] makes happen off the audio thread.
fn random_points() -> &'static [f32; MAX_RANDOM_PHASE] {
    static POINTS: std::sync::OnceLock<[f32; MAX_RANDOM_PHASE]> = std::sync::OnceLock::new();
    POINTS.get_or_init(|| {
        let mut rnd = JuceRandom::new(1);
        let mut points = [0.0; MAX_RANDOM_PHASE];
        for p in &mut points {
            *p = rnd.next_float() * 2.0 - 1.0;
        }
        points
    })
}

/// A random point, or 0 past the last, as `juce::Array::operator[]` reads.
fn point(index: i32) -> f32 {
    usize::try_from(index)
        .ok()
        .and_then(|i| random_points().get(i).copied())
        .unwrap_or(0.0)
}

/// `juce::roundToInt` for a double: the low word of `value + 1.5 · 2^52`,
/// which rounds half to even.
fn round_to_int(value: f64) -> i32 {
    (value + 6_755_399_441_055_744.0).to_bits() as u32 as i32
}

/// `gin::LFO`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lfo {
    params: LfoParams,
    sample_rate: f64,
    phase: f32,
    cur_phase: f32,
    output: f32,
    fade_delta: f32,
    cur_fade: f32,
    delay_steps: i32,
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            params: LfoParams::default(),
            sample_rate: 0.0,
            phase: 0.0,
            cur_phase: 0.0,
            output: 0.0,
            fade_delta: 0.0,
            cur_fade: 1.0,
            delay_steps: 0,
        }
    }
}

impl Lfo {
    /// Work out the random points now, so the audio thread never does.
    pub fn warm() {
        let _ = random_points();
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    /// `setParameters`. Switching to a random shape starts it somewhere
    /// random, drawn from `system` (`juce::Random::getSystemRandom()`).
    pub fn set_parameters(&mut self, params: LfoParams, system: &mut JuceRandom) {
        let old = self.params.shape;
        self.params = params;
        if !old.is_random() && params.shape.is_random() {
            self.phase = system.next_int_below(MAX_RANDOM_PHASE as i32) as f32;
        }
    }

    pub fn reset(&mut self) {
        self.output = 0.0;
        self.phase = 0.0;
        self.cur_phase = 0.0;
        self.cur_fade = 1.0;
        self.fade_delta = 0.0;
        self.delay_steps = 0;
    }

    fn max_phase(&self) -> f32 {
        if self.params.shape.is_random() {
            MAX_RANDOM_PHASE as f32
        } else {
            1.0
        }
    }

    /// `noteOn (phase)`: from `phase`, or with a negative phase from the
    /// start (a random point for the random shapes, from `system`).
    pub fn note_on(&mut self, phase: f32, system: &mut JuceRandom) {
        let max_phase = self.max_phase();
        self.cur_fade = if self.params.fade <= 0.0 { 1.0 } else { 0.0 };
        self.cur_phase = 0.0;
        self.phase = if self.params.shape.is_random() {
            if phase < 0.0 {
                system.next_int_below(MAX_RANDOM_PHASE as i32) as f32
            } else {
                phase
            }
        } else if phase < 0.0 {
            0.0
        } else {
            phase
        };
        self.fade_delta = (1.0 / (self.sample_rate * f64::from(self.params.fade))) as f32;
        self.delay_steps = round_to_int(self.sample_rate * f64::from(self.params.delay));
        let mut cur = (self.phase + self.params.phase) % max_phase;
        if cur < 0.0 {
            cur += max_phase;
        }
        self.cur_phase = cur;
        self.update();
    }

    /// `process (numSamples)`: that many samples on, and the output there.
    pub fn process(&mut self, samples: usize) -> f32 {
        let step = if self.params.frequency > 0.0001 {
            (f64::from(self.params.frequency) / self.sample_rate) as f32
        } else {
            0.0
        };
        for _ in 0..samples {
            if self.delay_steps > 0 {
                self.delay_steps -= 1;
            } else {
                self.cur_fade = jlimit(0.0, 1.0, self.cur_fade + self.fade_delta);
                let max_phase = self.max_phase();
                self.phase += step;
                while self.phase >= max_phase {
                    self.phase -= max_phase;
                }
                let mut cur = (self.phase + self.params.phase) % max_phase;
                if cur < 0.0 {
                    cur += max_phase;
                }
                self.cur_phase = cur;
            }
        }
        self.update()
    }

    /// `getOutput`.
    #[must_use]
    pub fn output(&self) -> f32 {
        jlimit(
            -1.0,
            1.0,
            self.cur_fade * self.output * self.params.depth + self.params.offset,
        )
    }

    fn update(&mut self) -> f32 {
        if self.delay_steps == 0 {
            let p = self.cur_phase;
            self.output = match self.params.shape {
                Shape::None => 0.0,
                Shape::Sine => math::sinf(p * core::f32::consts::PI * 2.0),
                Shape::Triangle => {
                    let t = (p + 0.25) % 1.0;
                    if t < 0.5 {
                        4.0 * t - 1.0
                    } else {
                        -4.0 * t + 3.0
                    }
                }
                Shape::SawUp => p * 2.0 - 1.0,
                Shape::SawDown => (1.0 - p) * 2.0 - 1.0,
                Shape::Square => {
                    if p < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Shape::SquarePos => {
                    if p < 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                }
                Shape::SampleAndHold => point(p as i32),
                Shape::Noise => {
                    let i = p as i32;
                    let t = p - i as f32;
                    let (a, b) = (point(i), point(i + 1));
                    a + t * (b - a)
                }
                Shape::StepUp3 => (p * 3.0) as i32 as f32 / 3.0 * 6.0 / 2.0 - 1.0,
                Shape::StepUp4 => (p * 4.0) as i32 as f32 / 4.0 * 8.0 / 3.0 - 1.0,
                Shape::StepUp8 => (p * 8.0) as i32 as f32 / 8.0 * 16.0 / 7.0 - 1.0,
                Shape::StepDown3 => -((p * 3.0) as i32 as f32 / 3.0 * 6.0 / 2.0 - 1.0),
                Shape::StepDown4 => -((p * 4.0) as i32 as f32 / 4.0 * 8.0 / 3.0 - 1.0),
                Shape::StepDown8 => -((p * 8.0) as i32 as f32 / 8.0 * 16.0 / 7.0 - 1.0),
                // The pyramids read `phase`, not the offset `curPhase`.
                Shape::Pyramid3 => table(&[0.0, 1.0, 0.0, -1.0], self.phase),
                Shape::Pyramid5 => table(&[0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5], self.phase),
                Shape::Pyramid9 => table(
                    &[
                        0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0,
                        -0.75, -0.5, -0.25,
                    ],
                    self.phase,
                ),
            };
        }
        self.output()
    }
}

fn table(values: &[f32], phase: f32) -> f32 {
    values[((phase * values.len() as f32) as usize).min(values.len() - 1)]
}

/// The most steps a Step LFO has.
pub const MAX_STEPS: usize = 32;

/// `gin::StepLFO`: a cycle of up to 32 levels, held a step each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepLfo {
    sample_rate: f64,
    frequency: f32,
    phase: f32,
    output: f32,
    points: [f32; MAX_STEPS],
    len: usize,
}

impl Default for StepLfo {
    fn default() -> Self {
        Self {
            sample_rate: 0.0,
            frequency: 0.0,
            phase: 0.0,
            output: 0.0,
            points: [0.0; MAX_STEPS],
            len: 0,
        }
    }
}

impl StepLfo {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn set_freq(&mut self, frequency: f32) {
        self.frequency = frequency;
    }

    /// `setNumPoints`: steps added read 0 until set.
    pub fn set_num_points(&mut self, n: usize) {
        let n = n.min(MAX_STEPS);
        for p in self.points.iter_mut().take(n).skip(self.len) {
            *p = 0.0;
        }
        self.len = n;
    }

    pub fn set_point(&mut self, n: usize, value: f32) {
        if n < self.len {
            self.points[n] = value;
        }
    }

    pub fn reset(&mut self) {
        self.output = 0.0;
        self.phase = 0.0;
    }

    pub fn note_on(&mut self, phase: f32) {
        self.output = 0.0;
        let size = self.len as f32;
        self.phase = if phase < 0.0 {
            0.0
        } else {
            (phase * size) as i32 as f32 / size
        };
    }

    pub fn process(&mut self, samples: usize) -> f32 {
        let step = if self.frequency > 0.0001 {
            (f64::from(self.frequency) / self.sample_rate) as f32 / self.len as f32
        } else {
            0.0
        };
        for _ in 0..samples {
            self.phase += step;
            while self.phase >= 1.0 {
                self.phase -= 1.0;
            }
        }
        self.output = usize::try_from((self.phase * self.len as f32) as i32)
            .ok()
            .filter(|&i| i < self.len)
            .map_or(0.0, |i| self.points[i]);
        jlimit(-1.0, 1.0, self.output)
    }

    #[must_use]
    pub fn output(&self) -> f32 {
        jlimit(-1.0, 1.0, self.output)
    }
}
