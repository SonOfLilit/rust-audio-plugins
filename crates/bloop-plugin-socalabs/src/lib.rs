//! The SocaLabs effects as bloop Plug-ins.
//!
//! A Rust port of the effect processors in Roland Rabien's slPlugins, of
//! the Gin DSP they are built from, and of the Gin effects slPlugins does
//! not ship as plug-ins (PlateReverb, Distortion, TransientShaper,
//! Maximizer, Exciter, SoftClipper), behind knobs of bloop's choosing, and
//! of the chorus SocaLabs Wavetable builds from Gin, behind Wavetable's
//! knobs. [`Effect`] is the Plug-in; `plugins`
//! holds the ported processors and `gin` the ported DSP, each module named
//! for the file it came from.
//!
//! Every processor is held bit-exact against the C++ it was ported from:
//! `cargo test -p bloop-plugin-socalabs --features gold` builds the
//! originals and runs both over the same programs (tests/gold.rs).

mod gin;
mod juce;
pub mod math;
mod plugins;
pub mod view;

/// The parts of this port SocaLabs' Wavetable synth is built from too
/// (`bloop-plugin-wavetable`): Gin's filter, chorus, delay line, plate
/// reverb, note durations and smoothers, the JUCE helpers, and the switch
/// between `libm` and the C library (`gold`). Shared rather than copied,
/// so that one gold harness holding either port to its C++ holds both.
pub mod shared {
    pub use crate::gin::audioutil::WetDryMix;
    pub use crate::gin::delay::DelayLine;
    pub use crate::gin::distortion::GlibcRand;
    pub use crate::gin::filter::{Filter, FilterType, Slope};
    pub use crate::gin::modulation::Modulation;
    pub use crate::gin::noteduration::{NoteDuration, DURATIONS};
    pub use crate::gin::param::{
        midi_note_from_hertz, midi_note_in_hertz, EasedValueSmoother, ValueSmoother,
    };
    pub use crate::gin::platereverb::PlateReverb;
    pub use crate::juce::{
        approximately_equal, decibels_to_gain, gain_to_decibels, jlimit, LinearSmoothedValue,
    };
    pub use crate::math;
}

use bloop_plugin_api::{ParamInfo, Plugin, PluginCategory, PluginInfo, ProcessContext};
use plugins::bitcrusher::BitCrusherProcessor;
use plugins::crossfeed::CrossfeedProcessor;
use plugins::delay::DelayProcessor;
use plugins::dynamics::{DynamicsProcessor, Flavour};
use plugins::extras::{
    ChorusProcessor, DistortionProcessor, ExciterProcessor, MaximizerProcessor,
    PlateReverbProcessor, SoftClipperProcessor, TransientShaperProcessor,
};
use plugins::graphiceq::GraphicEqProcessor;
use plugins::scopes::{Analyser, ScopeProcessor};
pub use plugins::scopes::oscilloscope;
use plugins::simpleverb::SimpleVerbProcessor;
use plugins::stereo::{StereoEnhancerProcessor, StereoProcessorProcessor};
use plugins::{Host, Processor};

/// The vendor every effect here reports.
pub const VENDOR: &str = "SocaLabs";

/// The slPlugins revision the effect processors are ported from: the one
/// `vendor/slplugins` pins, and the one the License pages cite.
pub const PORTED_FROM_SLPLUGINS_REV: &str = "034facd9cf74fa4a7a6312a8ebf4cbde1d357dac";
/// The Gin revision the DSP is ported from: the one `vendor/gin` pins.
pub const PORTED_FROM_GIN_REV: &str = "c983c0db166c6482b94d6227dacf9a35b8b9d59c";
/// The Wavetable revision whose chorus knobs the Chorus has. Its Gin
/// submodule is pinned at [`PORTED_FROM_GIN_REV`], so the `gin::Modulation`
/// it runs is the one ported here.
pub const PORTED_FROM_WAVETABLE_REV: &str = "5f932ff87c453d32cebee233af1fde946f53ac89";

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
    Chorus,
    Oscilloscope,
    SpectrumAnalyzer,
    XyScope,
}

impl Kind {
    /// Every effect, in the order the Browser lists them.
    pub const ALL: [Self; 21] = [
        Self::BitCrusher,
        Self::Chorus,
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
        Self::Oscilloscope,
        Self::PlateReverb,
        Self::SimpleVerb,
        Self::SoftClipper,
        Self::SpectrumAnalyzer,
        Self::StereoEnhancer,
        Self::StereoProcessor,
        Self::TransientShaper,
        Self::XyScope,
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
            Self::Chorus => "Chorus",
            Self::Oscilloscope => "Oscilloscope",
            Self::SpectrumAnalyzer => "SpectrumAnalyzer",
            Self::XyScope => "XYScope",
        }
    }

    /// What kind of effect it is, as the Browser's TYPE filter sorts them.
    #[must_use]
    pub fn category(self) -> &'static str {
        match self {
            Self::Compressor
            | Self::Limiter
            | Self::Gate
            | Self::Expander
            | Self::Maximizer
            | Self::TransientShaper => "Dynamics",
            Self::SimpleVerb | Self::PlateReverb => "Reverb",
            Self::Delay => "Delay",
            Self::GraphicEq => "EQ",
            Self::Distortion | Self::BitCrusher | Self::SoftClipper | Self::Exciter => "Distortion",
            Self::StereoEnhancer | Self::StereoProcessor | Self::Crossfeed => "Stereo",
            Self::Chorus => "Modulation",
            Self::Oscilloscope | Self::SpectrumAnalyzer | Self::XyScope => "Analysis",
        }
    }

    /// Whether it listens to a side-chain input: the Compressor, the
    /// Limiter and the Gate, as the Maschine's do
    /// (effects#the-side-chain-parameter-page) — Gin's dynamics can key
    /// off another signal. The Expander, which the Maschine has not, does
    /// not.
    #[must_use]
    pub fn takes_key(self) -> bool {
        matches!(self, Self::Compressor | Self::Limiter | Self::Gate)
    }

    /// The effect called `name`, if there is one.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// What an effect is built from: which one, and for a Delay how many
/// seconds its line holds.
///
/// The original Delay holds two minutes, 46 MB at 48kHz, whatever its
/// time is set to. A host that builds one with [`Recipe::new`] gets
/// [`Recipe::MIN_DELAY_ROOM`], and builds it again with
/// [`Recipe::room_for`] the time it is turned to, when that is more.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Recipe {
    pub kind: Kind,
    /// Seconds of delay line, for a Delay; nothing else has one.
    pub delay_room: f32,
}

impl Recipe {
    /// The least room a Delay is built with.
    pub const MIN_DELAY_ROOM: f32 = 2.0;
    /// The original's room, which is all a Delay can use: its Time knob
    /// stops at 120 s.
    pub const MAX_DELAY_ROOM: f32 = plugins::delay::MAX_DELAY as f32;

    /// `kind`, with the least room.
    #[must_use]
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            delay_room: Self::MIN_DELAY_ROOM,
        }
    }

    /// The room a Delay needs to reach `seconds` back: doubled from the
    /// least until it does, so a knob turned slowly up does not build a
    /// Delay per step, and never more than the original's.
    #[must_use]
    pub fn room_for(seconds: f32) -> f32 {
        let mut room = Self::MIN_DELAY_ROOM;
        while room < seconds + 0.1 && room < Self::MAX_DELAY_ROOM {
            room *= 2.0;
        }
        room.min(Self::MAX_DELAY_ROOM)
    }
}

/// The delay a Delay's knobs ask for, in seconds: the Time knob's, or with
/// Sync on the note the Beat knob names, at `bpm`.
#[must_use]
pub fn delay_seconds(sync: bool, time: f32, beat: f32, bpm: f32) -> f32 {
    plugins::delay::delay_seconds(sync, time, beat, bpm)
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
    Chorus(Box<ChorusProcessor>),
    Scope(Box<ScopeProcessor>),
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
            Engine::Chorus($p) => $body,
            Engine::Scope($p) => $body,
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

    fn clear(&mut self) {
        dispatch!(self, p => p.clear());
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
    ///
    /// This is the original: a Delay holds two minutes. [`Self::build`]
    /// makes one with the room a [`Recipe`] says.
    #[must_use]
    pub fn new(kind: Kind, sample_rate: f32) -> Self {
        Self::build(
            Recipe {
                kind,
                delay_room: Recipe::MAX_DELAY_ROOM,
            },
            sample_rate,
        )
    }

    /// The effect a recipe describes, prepared for `sample_rate` as
    /// [`Self::new`] is.
    #[must_use]
    pub fn build(recipe: Recipe, sample_rate: f32) -> Self {
        let Recipe { kind, delay_room } = recipe;
        let sample_rate = playable(sample_rate);
        let room = f64::from(delay_room.clamp(0.1, Recipe::MAX_DELAY_ROOM));
        let mut engine = match kind {
            Kind::Compressor => {
                Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Compressor)))
            }
            Kind::Limiter => Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Limiter))),
            Kind::Gate => Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Gate))),
            Kind::Expander => Engine::Dynamics(Box::new(DynamicsProcessor::new(Flavour::Expander))),
            Kind::Delay => Engine::Delay(Box::new(DelayProcessor::with_room(room))),
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
            Kind::Chorus => Engine::Chorus(Box::default()),
            Kind::Oscilloscope => Engine::Scope(Box::new(ScopeProcessor::new(Analyser::Oscilloscope))),
            Kind::SpectrumAnalyzer => {
                Engine::Scope(Box::new(ScopeProcessor::new(Analyser::SpectrumAnalyzer)))
            }
            Kind::XyScope => Engine::Scope(Box::new(ScopeProcessor::new(Analyser::XyScope))),
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

    /// The longest the effect's output can be silent while it still holds
    /// something it will play later, in seconds at `bpm`: a Delay's time,
    /// when its echo is heard at all (Mix above zero) — the gap between a
    /// note and its first echo. Zero for every other effect, whose tails
    /// sound without a gap. A host that stops running an effect after a
    /// silence must wait at least this long, or the echo is lost.
    #[must_use]
    pub fn quiet_gap_seconds(&self, bpm: f32) -> f32 {
        use plugins::delay::{BEAT, MIX, SYNC, TIME};
        let Engine::Delay(delay) = &self.engine else {
            return 0.0;
        };
        let value = |index: usize| {
            delay
                .params()
                .get(index)
                .map_or(0.0, gin::param::Param::user_value)
        };
        if value(MIX) <= 0.0 {
            return 0.0;
        }
        delay_seconds(value(SYNC) >= 0.5, value(TIME), value(BEAT), bpm)
    }

    /// Silence what the effect is still holding — a Delay's echoes, a
    /// reverb's tank, an envelope follower — and keep its knobs where they
    /// are: `reset` without `prepareToPlay`. Allocates nothing, at any
    /// rate, so the audio thread can call it (Choke All Notes).
    pub fn clear(&mut self) {
        self.engine.clear();
    }

    /// The rate the effect is prepared for: the last of [`Self::new`] and
    /// [`Plugin::reset`].
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// One block in place, the detector listening to `key` — a side-chain
    /// input's two channels, at least as long as the block — for an effect
    /// that has one ([`Kind::takes_key`]); any other processes as it
    /// always does.
    pub fn process_keyed(&mut self, context: &mut ProcessContext, key: [&[f32]; 2]) {
        if let (true, Engine::Dynamics(dynamics), [left, right, ..]) = (
            self.kind.takes_key(),
            &mut self.engine,
            &mut context.outputs[..],
        ) {
            dynamics.process_keyed(left, right, key);
            return;
        }
        self.process(context);
    }

    /// What a dynamics processor's meters read — the level in, the level
    /// out, the gain reduction — for the Compressor, Limiter, Gate,
    /// Expander and Maximizer; `None` for any other. Allocation-free: the engine reads it
    /// every block for the effect a screen is watching.
    #[must_use]
    pub fn meter(&self) -> Option<bloop_plugin_api::DynamicsMeter> {
        match &self.engine {
            Engine::Dynamics(dynamics) => Some(dynamics.meter()),
            Engine::Maximizer(maximizer) => Some(maximizer.meter()),
            _ => None,
        }
    }

    /// What the Maximizer and the TransientShaper traced over the last
    /// block, for their screens (`bloop_plugin_api::trace`): empty for any
    /// other effect. Allocation-free, read after every block for the effect
    /// a screen is watching.
    #[must_use]
    pub fn trace(&self) -> &[[f32; 2]] {
        match &self.engine {
            Engine::Maximizer(maximizer) => maximizer.trace().points(),
            Engine::TransientShaper(shaper) => shaper.trace().points(),
            _ => &[],
        }
    }

    /// The unit a parameter's value is in, as the original labels it.
    #[must_use]
    pub fn param_unit(&self, index: usize) -> &'static str {
        self.engine
            .params()
            .get(index)
            .map_or("", |p| p.spec().unit)
    }

    /// A parameter at `value`, as the original prints it over its knob:
    /// its own text function where it has one ("On", "1/8d", "12.0 kHz"),
    /// Gin's number format and its unit where it does not.
    #[must_use]
    pub fn param_text(&self, index: usize, value: f32) -> String {
        self.engine
            .params()
            .get(index)
            .map_or_else(String::new, |p| {
                let spec = p.spec();
                spec.text.show(value, spec.unit, spec.range.interval)
            })
    }

    /// How many values a parameter that steps can take, ends included —
    /// a switch is two, the Delay's Beat a note length each — and `None`
    /// for one that turns continuously.
    ///
    /// Also `None` for steps through a skewed range: they are not evenly
    /// spread along the knob, and a host stopping at even positions would
    /// land between them. No effect has one.
    #[must_use]
    pub fn param_steps(&self, index: usize) -> Option<usize> {
        let range = self.engine.params().get(index)?.spec().range;
        #[expect(clippy::float_cmp, reason = "a skew of exactly one is no skew")]
        let even = range.skew == 1.0;
        (range.interval > 0.0 && even)
            .then(|| ((range.end - range.start) / range.interval).round() as usize + 1)
    }

    /// Where a knob turned to `fraction` of its travel (0 to 1) puts the
    /// parameter: its range's skew applied, and snapped to the steps a
    /// switch or a list takes, as the original's `setValue` does.
    #[must_use]
    pub fn knob_value(&self, index: usize, fraction: f32) -> f32 {
        self.engine.params().get(index).map_or(0.0, |p| {
            let range = p.spec().range;
            range.snap_to_legal_value(range.convert_from_0to1(fraction))
        })
    }

    /// How far along its knob's travel (0 to 1) a parameter at `value`
    /// is: the inverse of [`Self::knob_value`].
    #[must_use]
    pub fn knob_fraction(&self, index: usize, value: f32) -> f32 {
        self.engine
            .params()
            .get(index)
            .map_or(0.0, |p| p.spec().range.convert_to_0to1(value))
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
