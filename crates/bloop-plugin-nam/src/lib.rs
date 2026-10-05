//! Neural Amp Modeler: a Rust port of the WaveNet and LSTM networks of
//! NeuralAmpModelerCore (MIT, Steven Atkinson), which run `.nam` captures
//! of amps and pedals, and of the tone stack NAM's plug-in puts after them.
//!
//! [`Model`] reads a `.nam` file and runs it, a block at a time,
//! allocation-free. [`bundled`] is the captures bloop ships, [`tone`] the
//! plug-in's tone stack and DC blocker. The effect a Sound loads, with its
//! knobs, is `bloop_plugin_effects::nam`.
//!
//! What runs: the WaveNet of the trainer's A1 sizes (standard, lite,
//! feather, nano), gated or not, any activation NAM names, and the LSTM.
//! What does not, yet, and is refused with a message saying so: NAM's
//! A2 WaveNets (`FiLM`, bottlenecks, grouped convolutions, per-layer
//! kernels, slimmable containers, a condition DSP, a post-stack head) and
//! the old `ConvNet` and `Linear` architectures.

pub mod act;
pub mod bundled;
pub mod lstm;
pub mod model;
pub mod preset;
pub mod tone;
pub mod wavenet;

pub use act::Tanh;
pub use model::{Metadata, Model};

/// The NeuralAmpModelerCore revision the port follows, which `vendor/nam`
/// pins and the gold tests compile.
pub const PORTED_FROM_NAM_REV: &str = "0b3d3c97b0859a3a8c92a8628c4dd89a25eb5842";
/// The NeuralAmpModelerPlugin revision the plug-in's chain (tone stack,
/// normalized output) and the Obsidian capture come from.
pub const NAM_PLUGIN_REV: &str = "16be869746b8915885c7a35bafdcc0061faeb50e";
/// The AudioDSPTools revision that plug-in pins, whose filters the tone
/// stack and the DC blocker are.
pub const AUDIO_DSP_TOOLS_REV: &str = "e19ef4b5b3bf2171c431847563acde29eedf85c0";
