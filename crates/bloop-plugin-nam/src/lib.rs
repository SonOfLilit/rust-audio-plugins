//! Neural Amp Modeler, as a bloop Plug-in: a Rust port of the WaveNet and
//! LSTM networks of NeuralAmpModelerCore (MIT), which run `.nam` captures
//! of amps and pedals.

pub mod act;
pub mod lstm;
pub mod model;
pub mod preset;
pub mod wavenet;

pub use act::Tanh;
pub use model::{Metadata, Model};
