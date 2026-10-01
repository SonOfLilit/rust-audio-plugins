//! `gin::Parameter` and its smoothed variants, as a processor reads them.
//!
//! Ported from Gin's `gin_plugin/plugin/gin_parameter.h`,
//! `gin_smoothedparameter.h` and `gin_processor.cpp` (`createParam`), and
//! the smoothers in `gin_dsp/dsp/gin_valuesmoother.h` and
//! `gin_easedvaluesmoother.h`. What a JUCE host needs of a parameter —
//! notification, text, automation — is left behind; what is kept is the
//! value a processor sees, step by step, which is all that reaches audio.

use crate::gin::text::Text;
use crate::juce::{approximately_equal, jlimit, NormalisableRange};
use crate::math::{std_max, std_min};

/// `gin::ValueSmoother<float>`: a constant-rate ramp, one full 0-to-1
/// sweep per `time` seconds.
#[derive(Clone, Debug)]
pub struct ValueSmoother {
    time: f64,
    sample_rate: f64,
    delta: f64,
    target: f32,
    current: f32,
}

impl Default for ValueSmoother {
    fn default() -> Self {
        Self {
            time: 0.1,
            sample_rate: 44100.0,
            delta: 0.0,
            target: 0.0,
            current: 0.0,
        }
    }
}

impl ValueSmoother {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.reset();
    }

    pub fn set_time(&mut self, time: f64) {
        self.time = time;
        self.reset();
    }

    #[must_use]
    pub fn is_smoothing(&self) -> bool {
        f64::from((self.current - self.target).abs()) > self.delta * 0.001
    }

    fn reset(&mut self) {
        self.delta = 1.0 / (self.sample_rate * self.time);
    }

    #[must_use]
    pub fn current(&self) -> f32 {
        self.current
    }

    pub fn set_value(&mut self, value: f32) {
        self.target = value;
    }

    pub fn snap_to_value(&mut self) {
        self.current = self.target;
    }

    pub fn process(&mut self, steps: usize) {
        if !approximately_equal(self.target, self.current) {
            for _ in 0..steps {
                self.update_value();
            }
        }
    }

    fn update_value(&mut self) {
        if self.current < self.target {
            self.current = std_min(self.target, (f64::from(self.current) + self.delta) as f32);
        } else if self.current > self.target {
            self.current = std_max(self.target, (f64::from(self.current) - self.delta) as f32);
        }
    }

    /// `getNextValue`: one step on, and where that lands.
    pub fn next_value(&mut self) -> f32 {
        self.update_value();
        self.current
    }
}

/// `gin::EasedValueSmoother<float, QuadraticInOutEasing>`: the same ramp
/// through an ease-in-out curve, restarted from wherever it stands each
/// time the target moves.
#[derive(Clone, Debug)]
pub struct EasedValueSmoother {
    time: f64,
    sample_rate: f64,
    position: f64,
    target: f32,
    orig: f32,
    current: f32,
}

impl Default for EasedValueSmoother {
    fn default() -> Self {
        Self {
            time: 0.1,
            sample_rate: 44100.0,
            position: 1.0,
            target: 0.0,
            orig: 0.0,
            current: 0.0,
        }
    }
}

/// `gin::easeQuadraticInOut<double>`.
fn ease_quadratic_in_out(p: f64) -> f64 {
    if p < 0.5 {
        2.0 * p * p
    } else {
        (-2.0 * p * p) + (4.0 * p) - 1.0
    }
}

impl EasedValueSmoother {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    #[must_use]
    pub fn is_smoothing(&self) -> bool {
        self.position < 1.0
    }

    #[must_use]
    pub fn current(&self) -> f32 {
        self.current
    }

    pub fn snap_to_value(&mut self) {
        self.current = self.target;
        self.position = 1.0;
    }

    pub fn process(&mut self, steps: usize) {
        if !approximately_equal(self.target, self.current) {
            for _ in 0..steps {
                self.update_value();
            }
        }
    }

    pub fn set_value(&mut self, value: f32) {
        self.orig = self.current;
        self.target = value;
        self.position = 0.0;
    }

    fn update_value(&mut self) {
        if self.position < 1.0 {
            let delta = 1.0 / (self.sample_rate * self.time);
            self.position = crate::math::std_min_f64(self.position + delta, 1.0);
            let f = ease_quadratic_in_out(self.position);
            self.current = (f64::from(self.target) * f + f64::from(self.orig) * (1.0 - f)) as f32;
        }
    }
}

/// How a parameter moves between the values it is set to:
/// `gin::SmoothingType`, where a zero time means not at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Smoothing {
    None,
    Linear(f32),
    Eased(f32),
}

#[derive(Clone, Debug)]
enum Smoother {
    None,
    Linear(ValueSmoother),
    Eased(EasedValueSmoother),
}

/// What a parameter's user value turns into before the DSP sees it — the
/// original's `conversionFunction`, one per kind the plug-ins use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    /// Handed over as it stands.
    Identity,
    /// Milliseconds to seconds, through a double: `in / 1000.0`.
    MsToSeconds,
    /// Decibels to linear gain.
    DecibelsToGain,
    /// Percent to a fraction, in float: `in / 100.0f`.
    PercentToFraction,
    /// A MIDI note number to its frequency: `gin::getMidiNoteInHertz`.
    MidiNoteToHertz,
}

impl Conversion {
    #[must_use]
    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::Identity => value,
            Self::MsToSeconds => (f64::from(value) / 1000.0) as f32,
            Self::DecibelsToGain => crate::juce::decibels_to_gain(value),
            Self::PercentToFraction => value / 100.0,
            Self::MidiNoteToHertz => midi_note_in_hertz(value),
        }
    }
}

/// `gin::getMidiNoteInHertz<float>`, A at 440 Hz.
#[must_use]
pub fn midi_note_in_hertz(note: f32) -> f32 {
    (f64::from(440.0f32) * crate::math::pow(2.0, (f64::from(note) - 69.0) / 12.0)) as f32
}

/// `gin::getMidiNoteFromHertz<float>`, A at 440 Hz.
#[must_use]
pub fn midi_note_from_hertz(freq: f32) -> f32 {
    (12.0 * crate::math::log2(f64::from(freq) / f64::from(440.0f32)) + 69.0) as f32
}

/// The constant part of a parameter: what it is called and how it runs,
/// as each plug-in's constructor declares it (`addExtParam`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamSpec {
    pub name: &'static str,
    pub unit: &'static str,
    pub range: NormalisableRange,
    pub default: f32,
    pub smoothing: Smoothing,
    pub conversion: Conversion,
    /// How the value prints: `addExtParam`'s text function, or Gin's own.
    pub text: Text,
}

impl ParamSpec {
    #[must_use]
    pub const fn new(
        name: &'static str,
        unit: &'static str,
        range: NormalisableRange,
        default: f32,
        smoothing: Smoothing,
        conversion: Conversion,
    ) -> Self {
        Self {
            name,
            unit,
            range,
            default,
            smoothing,
            conversion,
            text: Text::Number,
        }
    }

    /// The same parameter, printed with one of the originals' text
    /// functions rather than Gin's own.
    #[must_use]
    pub const fn with_text(self, text: Text) -> Self {
        Self { text, ..self }
    }
}

/// One live parameter: `gin::Parameter`, or `gin::SmoothedParameter` over
/// either smoother, as `createParam` picks.
#[derive(Clone, Debug)]
pub struct Param {
    spec: ParamSpec,
    value: f32,
    smoother: Smoother,
}

impl Param {
    #[must_use]
    pub fn new(spec: ParamSpec) -> Self {
        let mut param = Self {
            spec,
            value: spec.default,
            smoother: match spec.smoothing {
                Smoothing::Linear(time) if time > 0.0 => {
                    let mut s = ValueSmoother::default();
                    s.set_time(f64::from(time));
                    Smoother::Linear(s)
                }
                Smoothing::Eased(time) if time > 0.0 => {
                    let mut s = EasedValueSmoother::default();
                    s.set_time(f64::from(time));
                    Smoother::Eased(s)
                }
                _ => Smoother::None,
            },
        };
        let normalised = param.spec.range.convert_to_0to1(param.value);
        match &mut param.smoother {
            Smoother::None => {}
            Smoother::Linear(s) => {
                s.set_value(normalised);
                s.snap_to_value();
            }
            Smoother::Eased(s) => {
                s.set_value(normalised);
                s.snap_to_value();
            }
        }
        param
    }

    #[must_use]
    pub fn spec(&self) -> &ParamSpec {
        &self.spec
    }

    /// `prepareToPlay`: the smoother learns the sample rate.
    pub fn prepare(&mut self, sample_rate: f64) {
        match &mut self.smoother {
            Smoother::None => {}
            Smoother::Linear(s) => s.set_sample_rate(sample_rate),
            Smoother::Eased(s) => s.set_sample_rate(sample_rate),
        }
    }

    /// `reset`: a smoothed parameter jumps to where it is heading.
    pub fn reset(&mut self) {
        match &mut self.smoother {
            Smoother::None => {}
            Smoother::Linear(s) => s.snap_to_value(),
            Smoother::Eased(s) => s.snap_to_value(),
        }
    }

    /// `setUserValue`: snapped to a legal value, and the smoother told to
    /// head there if it moved.
    pub fn set_user_value(&mut self, value: f32) {
        let range = self.spec.range;
        let v = jlimit(range.start, range.end, range.snap_to_legal_value(value));
        if !approximately_equal(self.value, v) {
            self.value = v;
            let normalised = range.convert_to_0to1(v);
            match &mut self.smoother {
                Smoother::None => {}
                Smoother::Linear(s) => s.set_value(normalised),
                Smoother::Eased(s) => s.set_value(normalised),
            }
        }
    }

    /// `getUserValue`: where the parameter was last set.
    #[must_use]
    pub fn user_value(&self) -> f32 {
        let range = self.spec.range;
        jlimit(range.start, range.end, self.value)
    }

    /// `getUserValueInt`.
    #[must_use]
    pub fn user_value_int(&self) -> i32 {
        self.user_value() as i32
    }

    /// `isOn`: anywhere but the bottom of the range.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.spec.range.start != self.user_value()
    }

    #[must_use]
    pub fn is_smoothing(&self) -> bool {
        match &self.smoother {
            Smoother::None => false,
            Smoother::Linear(s) => s.is_smoothing(),
            Smoother::Eased(s) => s.is_smoothing(),
        }
    }

    /// `getProcValue (stepSize)`: the value the DSP should use now, and the
    /// smoother moved on by `steps` samples.
    pub fn proc_value(&mut self, steps: usize) -> f32 {
        let range = self.spec.range;
        let smoothed = match &mut self.smoother {
            Smoother::Linear(s) if s.is_smoothing() => {
                let v = range.convert_from_0to1(s.current());
                s.process(steps);
                Some(v)
            }
            Smoother::Eased(s) if s.is_smoothing() => {
                let v = range.convert_from_0to1(s.current());
                s.process(steps);
                Some(v)
            }
            _ => None,
        };
        self.spec
            .conversion
            .apply(smoothed.unwrap_or_else(|| self.user_value()))
    }
}

/// `gin::Processor::isSmoothing`: whether any parameter is still moving.
#[must_use]
pub fn any_smoothing(params: &[Param]) -> bool {
    params.iter().any(Param::is_smoothing)
}
