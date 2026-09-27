//! The SocaLabs effects as bloop Plug-ins.
//!
//! A Rust port of the effect processors in Roland Rabien's slPlugins, of
//! the Gin DSP they are built from, and of the Gin effects slPlugins does
//! not ship as plug-ins (PlateReverb, Distortion, TransientShaper,
//! Maximizer, Exciter, SoftClipper), behind knobs of bloop's choosing. [`Effect`] is the Plug-in; `plugins`
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
use plugins::bitcrusher::BitCrusherProcessor;
use plugins::crossfeed::CrossfeedProcessor;
use plugins::delay::DelayProcessor;
use plugins::dynamics::{DynamicsProcessor, Flavour};
use plugins::extras::{
    DistortionProcessor, ExciterProcessor, MaximizerProcessor, PlateReverbProcessor,
    SoftClipperProcessor, TransientShaperProcessor,
};
use plugins::graphiceq::GraphicEqProcessor;
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
    GraphicEq,
    BitCrusher,
    PlateReverb,
    Distortion,
    TransientShaper,
    Maximizer,
    Exciter,
    SoftClipper,
}

impl Kind {
    /// Every effect, in the order the Browser lists them.
    pub const ALL: [Self; 17] = [
        Self::BitCrusher,
        Self::Compressor,
        Self::Crossfeed,
        Self::Delay,
        Self::Distortion,
        Self::Exciter,
        Self::Expander,
        Self::Gate,
        Self::GraphicEq,
        Self::Limiter,
        Self::Maximizer,
        Self::PlateReverb,
        Self::SimpleVerb,
        Self::SoftClipper,
        Self::StereoEnhancer,
        Self::StereoProcessor,
        Self::TransientShaper,
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
            Self::GraphicEq => "GraphicEQ",
            Self::BitCrusher => "BitCrusher",
            Self::PlateReverb => "PlateReverb",
            Self::Distortion => "Distortion",
            Self::TransientShaper => "TransientShaper",
            Self::Maximizer => "Maximizer",
            Self::Exciter => "Exciter",
            Self::SoftClipper => "SoftClipper",
        }
    }

    /// The effect called `name`, if there is one.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// The lowest rate an effect is prepared for.
pub const MIN_SAMPLE_RATE: f32 = 8000.0;

fn playable(sample_rate: f32) -> f32 {
    sample_rate.max(MIN_SAMPLE_RATE)
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
    GraphicEq(Box<GraphicEqProcessor>),
    BitCrusher(Box<BitCrusherProcessor>),
    PlateReverb(Box<PlateReverbProcessor>),
    Distortion(Box<DistortionProcessor>),
    TransientShaper(Box<TransientShaperProcessor>),
    Maximizer(Box<MaximizerProcessor>),
    Exciter(Box<ExciterProcessor>),
    SoftClipper(Box<SoftClipperProcessor>),
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
            Engine::GraphicEq($p) => $body,
            Engine::BitCrusher($p) => $body,
            Engine::PlateReverb($p) => $body,
            Engine::Distortion($p) => $body,
            Engine::TransientShaper($p) => $body,
            Engine::Maximizer($p) => $body,
            Engine::Exciter($p) => $body,
            Engine::SoftClipper($p) => $body,
        }
    };
}

impl Engine {
    fn params(&self) -> &[gin::param::Param] {
        dispatch!(self, p => p.params())
    }

    fn params_mut(&mut self) -> &mut [gin::param::Param] {
        dispatch!(self, p => p.params_mut())
    }

    fn prepare(&mut self, sample_rate: f32) {
        dispatch!(self, p => p.prepare(f64::from(sample_rate)));
    }
}

/// One SocaLabs effect, as a bloop Plug-in.
#[derive(Clone)]
pub struct Effect {
    kind: Kind,
    engine: Engine,
    sample_rate: f32,
}

impl Effect {
    /// The effect at its defaults, prepared for `sample_rate` as a host
    /// prepares a processor before its first block. This is where an
    /// effect allocates — the Delay's line, the reverb's combs — so it
    /// belongs on the UI thread, at the rate the engine runs at; a later
    /// [`Plugin::reset`] at that same rate allocates nothing.
    ///
    /// A rate below [`MIN_SAMPLE_RATE`], or not a number, is taken as that
    /// minimum: the delay lines are sized from it, and none can be empty.
    #[must_use]
    pub fn new(kind: Kind, sample_rate: f32) -> Self {
        let sample_rate = playable(sample_rate);
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
            Kind::GraphicEq => Engine::GraphicEq(Box::default()),
            Kind::BitCrusher => Engine::BitCrusher(Box::default()),
            Kind::PlateReverb => Engine::PlateReverb(Box::default()),
            Kind::Distortion => Engine::Distortion(Box::default()),
            Kind::TransientShaper => Engine::TransientShaper(Box::default()),
            Kind::Maximizer => Engine::Maximizer(Box::default()),
            Kind::Exciter => Engine::Exciter(Box::default()),
            Kind::SoftClipper => Engine::SoftClipper(Box::default()),
        };
        engine.prepare(sample_rate);
        Self {
            kind,
            engine,
            sample_rate,
        }
    }

    #[must_use]
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The rate the effect is prepared for: the last of [`Self::new`] and
    /// [`Plugin::reset`].
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// The unit a parameter's value is in, as the original labels it.
    #[must_use]
    pub fn param_unit(&self, index: usize) -> &'static str {
        self.engine
            .params()
            .get(index)
            .map_or("", |p| p.spec().unit)
    }

    /// The skew of a parameter's range: a knob's position `p`, 0 to 1,
    /// stands for `min + (max - min) * p^(1/skew)`.
    #[must_use]
    pub fn param_skew(&self, index: usize) -> f32 {
        self.engine
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
        self.engine.params().len()
    }

    fn param_info(&self, index: usize) -> ParamInfo {
        self.engine.params().get(index).map_or(
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
    /// the original's `setUserValue` does. A value that is not finite is
    /// dropped: the original would store a NaN, and every sample after
    /// would be NaN.
    fn set_param(&mut self, index: usize, value: f32) {
        if !value.is_finite() {
            return;
        }
        if let Some(p) = self.engine.params_mut().get_mut(index) {
            p.set_user_value(value);
        }
    }

    fn get_param(&self, index: usize) -> f32 {
        self.engine
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
        let sample_rate = playable(sample_rate);
        self.sample_rate = sample_rate;
        self.engine.prepare(sample_rate);
    }
}
