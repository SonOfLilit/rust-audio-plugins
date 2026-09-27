//! The SocaLabs effects as bloop Plug-ins.
//!
//! A Rust port of the effect processors in Roland Rabien's slPlugins and of
//! the Gin DSP they are built from. [`Effect`] is the Plug-in; `plugins`
//! holds the ported processors and `gin` the ported DSP, each module named
//! for the file it came from.
//!
//! Every processor is held bit-exact against the C++ it was ported from:
//! `cargo test -p bloop-plugin-socalabs --features gold` builds the
//! originals and runs both over the same programs (tests/gold.rs).

mod gin;
mod juce;
mod math;
mod plugins;

use bloop_plugin_api::{ParamInfo, Plugin, PluginCategory, PluginInfo, ProcessContext};
use plugins::crossfeed::CrossfeedProcessor;
use plugins::delay::DelayProcessor;
use plugins::dynamics::{DynamicsProcessor, Flavour};
use plugins::simpleverb::SimpleVerbProcessor;
use plugins::stereo::{StereoEnhancerProcessor, StereoProcessorProcessor};
use plugins::{Host, Processor};

/// The vendor every effect here reports.
pub const VENDOR: &str = "SocaLabs";

/// Which effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Compressor,
    Limiter,
    Gate,
    Expander,
    Delay,
    SimpleVerb,
    StereoEnhancer,
    StereoProcessor,
    Crossfeed,
}

impl Kind {
    /// Every effect, in the order the Browser lists them.
    pub const ALL: [Self; 9] = [
        Self::Compressor,
        Self::Crossfeed,
        Self::Delay,
        Self::Expander,
        Self::Gate,
        Self::Limiter,
        Self::SimpleVerb,
        Self::StereoEnhancer,
        Self::StereoProcessor,
    ];

    /// The effect's name, as SocaLabs publishes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Compressor => "Compressor",
            Self::Limiter => "Limiter",
            Self::Gate => "Gate",
            Self::Expander => "Expander",
            Self::Delay => "Delay",
            Self::SimpleVerb => "SimpleVerb",
            Self::StereoEnhancer => "StereoEnhancer",
            Self::StereoProcessor => "StereoProcessor",
            Self::Crossfeed => "Crossfeed",
        }
    }

    /// The effect called `name`, if there is one.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// The ported processor behind an [`Effect`], by enum rather than trait
/// object so that nothing on the audio thread dispatches through a vtable.
/// Boxed, so an `Effect` is small whichever effect it holds.
#[derive(Clone, Debug)]
enum Engine {
    Dynamics(Box<DynamicsProcessor>),
    Delay(Box<DelayProcessor>),
    SimpleVerb(Box<SimpleVerbProcessor>),
    StereoEnhancer(Box<StereoEnhancerProcessor>),
    StereoProcessor(Box<StereoProcessorProcessor>),
    Crossfeed(Box<CrossfeedProcessor>),
}

macro_rules! dispatch {
    ($engine:expr, $p:ident => $body:expr) => {
        match $engine {
            Engine::Dynamics($p) => $body,
            Engine::Delay($p) => $body,
            Engine::SimpleVerb($p) => $body,
            Engine::StereoEnhancer($p) => $body,
            Engine::StereoProcessor($p) => $body,
            Engine::Crossfeed($p) => $body,
        }
    };
}

impl Engine {
    fn processor(&self) -> &dyn Processor {
        dispatch!(self, p => &**p)
    }

    fn processor_mut(&mut self) -> &mut dyn Processor {
        dispatch!(self, p => &mut **p)
    }
}

/// One SocaLabs effect, as a bloop Plug-in.
#[derive(Clone)]
pub struct Effect {
    kind: Kind,
    engine: Engine,
}

impl Effect {
    /// The effect at its defaults, prepared for 44.1kHz until the host
    /// says otherwise with [`Plugin::reset`].
    #[must_use]
    pub fn new(kind: Kind) -> Self {
        let mut engine = match kind {
            Kind::Compressor => {
                Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Compressor)))
            }
            Kind::Limiter => Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Limiter))),
            Kind::Gate => Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Gate))),
            Kind::Expander => Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Expander))),
            Kind::Delay => Engine::Delay(Box::default()),
            Kind::SimpleVerb => Engine::SimpleVerb(Box::default()),
            Kind::StereoEnhancer => Engine::StereoEnhancer(Box::default()),
            Kind::StereoProcessor => Engine::StereoProcessor(Box::default()),
            Kind::Crossfeed => Engine::Crossfeed(Box::default()),
        };
        engine.processor_mut().prepare(44100.0);
        Self { kind, engine }
    }

    #[must_use]
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The unit a parameter's value is in, as the original labels it.
    #[must_use]
    pub fn param_unit(&self, index: usize) -> &'static str {
        self.engine
            .processor()
            .params()
            .get(index)
            .map_or("", |p| p.spec().unit)
    }

    /// The skew of a parameter's range: a knob's position `p`, 0 to 1,
    /// stands for `min + (max - min) * p^(1/skew)`.
    #[must_use]
    pub fn param_skew(&self, index: usize) -> f32 {
        self.engine
            .processor()
            .params()
            .get(index)
            .map_or(1.0, |p| p.spec().range.skew)
    }
}

impl core::fmt::Debug for Effect {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Effect")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Plugin for Effect {
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: self.kind.name(),
            vendor: VENDOR,
            category: PluginCategory::Effect,
            num_audio_inputs: 2,
            num_audio_outputs: 2,
        }
    }

    fn param_count(&self) -> usize {
        self.engine.processor().params().len()
    }

    fn param_info(&self, index: usize) -> ParamInfo {
        self.engine.processor().params().get(index).map_or(
            ParamInfo {
                name: "",
                default: 0.0,
                min: 0.0,
                max: 0.0,
            },
            |p| {
                let spec = p.spec();
                ParamInfo {
                    name: spec.name,
                    default: spec.default,
                    min: spec.range.start,
                    max: spec.range.end,
                }
            },
        )
    }

    /// The knob at `index` to `value`, snapped and held to its range as
    /// the original's `setUserValue` does. A value that is not a number is
    /// dropped: the original would store it, and every sample after would
    /// be NaN.
    fn set_param(&mut self, index: usize, value: f32) {
        if !value.is_finite() {
            return;
        }
        if let Some(p) = self.engine.processor_mut().params_mut().get_mut(index) {
            p.set_user_value(value);
        }
    }

    fn get_param(&self, index: usize) -> f32 {
        self.engine
            .processor()
            .params()
            .get(index)
            .map_or(0.0, gin::param::Param::user_value)
    }

    /// In place: the outputs arrive holding the input.
    fn process(&mut self, context: &mut ProcessContext) {
        let host = Host {
            bpm: context.tempo as f32,
        };
        if let [left, right, ..] = &mut context.outputs[..] {
            dispatch!(&mut self.engine, p => p.process(left, right, host));
        }
    }

    fn reset(&mut self, sample_rate: f32, _max_buffer_size: usize) {
        dispatch!(&mut self.engine, p => p.prepare(f64::from(sample_rate)));
    }
}
