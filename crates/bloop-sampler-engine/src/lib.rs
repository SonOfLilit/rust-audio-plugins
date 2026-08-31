//! bloop-sampler-engine: sample playback, ported from Just-a-Sample.
//!
//! The voice is a 1:1 Rust port of `CustomSamplerVoice` from
//! [Just-a-Sample](https://github.com/BOBONA/Just-a-Sample) (MIT, © 2024
//! Binyamin Friedman), pinned at [`PORTED_FROM_JAS_REV`]. Bit-exactness against
//! the C++ original is enforced by the gold tests in `tests/gold.rs`, which
//! compile the reference voice and compare every rendered sample.
//!
//! Two parts of the reference are deliberately out of scope:
//!
//! - **The FX chain** (reverb, distortion, EQ, chorus). Bloop routes effects
//!   through its own mixer, and the reference's chain depends on Gin (GPL).
//! - **ADVANCED (Bungee) playback mode**, the independent time/pitch stretcher.
//!   Bungee is a separate C++ project on the scale of the port itself; BASIC
//!   mode covers everything the sampler needs to slice and play. See the
//!   roadmap.
//!
//! MTS-ESP microtuning is also absent — the reference is a no-op unless an
//! MTS-ESP master is connected, which is the behaviour reproduced here.

mod buffer;
mod decode;
mod edit;
mod juce;
mod onsets;
mod params;
mod record;
mod voice;

pub use buffer::{SampleBuffer, SharedSample};
pub use decode::{decode, ImportError, SampleDecoder, IMPORTABLE_EXTENSIONS};
pub use edit::{fade, normalize, reverse, trim};
pub use onsets::detect_onsets;
pub use params::{ranges, ParamRange, SamplerParams, WAVETABLE_CUTOFF_HZ};
pub use record::{RecordState, Recorder};
pub use voice::{SamplerVoice, VoiceState, MAX_CHANNELS};

/// The Just-a-Sample revision this crate's voice is a port of.
///
/// The manual's License page cites this exact revision — keep them in sync (see
/// the `license_page_cites_the_pinned_jas_revision` test in bloop-menu-pages).
pub const PORTED_FROM_JAS_REV: &str = "bc97747a514755ca686f8b1e5cd223e5b44866b4";
