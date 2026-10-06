//! Neural Amp Modeler: a Rust port of the WaveNet and LSTM networks of
//! NeuralAmpModelerCore (MIT, Steven Atkinson), which run `.nam` captures
//! of amps and pedals, and of the tone stack NAM's plug-in puts after them.
//!
//! [`Model`] reads a `.nam` file and runs it, a block at a time,
//! allocation-free. [`bundled`] is the captures bloop ships, [`tone`] the
//! plug-in's tone stack and DC blocker. The effect a Sound loads, with its
//! knobs, is `bloop_plugin_effects::nam`.
//!
//! What runs: the WaveNet as NeuralAmpModelerCore reads it — the
//! trainer's A1 sizes (standard, lite, feather, nano), its A2 ones (Full
//! and Lite, in a `SlimmableContainer`), and every A2 feature the core
//! has (per-layer kernels, bottlenecks, grouped convolutions, gating and
//! blending, head 1x1s and head convolutions, `FiLM`, a condition DSP, a
//! post-stack head, slimmable WaveNets) — and the LSTM. What does not,
//! and is refused with a message saying so: the old `ConvNet` and
//! `Linear` architectures, and models of more than one input or output.

pub mod act;
pub mod bundled;
pub mod config;
pub mod lstm;
pub mod model;
pub mod preset;
pub mod slim;
pub mod tone;
pub mod wavenet;

pub use act::Tanh;
pub use model::{Metadata, Model, Size};

/// The NeuralAmpModelerCore revision the port follows, which `vendor/nam`
/// pins and the gold tests compile: v0.6.0, which has A2 as released.
pub const PORTED_FROM_NAM_REV: &str = "0b3d3c97b0859a3a8c92a8628c4dd89a25eb5842";
/// The NeuralAmpModelerPlugin revision the plug-in's chain (tone stack,
/// normalized output) and the Obsidian capture come from.
pub const NAM_PLUGIN_REV: &str = "16be869746b8915885c7a35bafdcc0061faeb50e";
/// The AudioDSPTools revision that plug-in pins, whose filters the tone
/// stack and the DC blocker are.
pub const AUDIO_DSP_TOOLS_REV: &str = "e19ef4b5b3bf2171c431847563acde29eedf85c0";
