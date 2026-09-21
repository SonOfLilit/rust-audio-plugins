//! bloop-sampler-engine: sample playback, ported from Just-a-Sample.
//!
//! The voice is a 1:1 Rust port of `CustomSamplerVoice` from
//! [Just-a-Sample](https://github.com/BOBONA/Just-a-Sample) (MIT, © 2024
//! Binyamin Friedman), pinned at [`PORTED_FROM_JAS_REV`]. Agreement with the
//! C++ original is enforced by the gold tests in `tests/gold.rs`, which compile
//! the reference voice and compare every rendered sample. The port matched bit
//! for bit until the interpolation window was tabulated for speed; what is left
//! is f32 rounding, and each scenario pins how much.
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
pub mod wave;

pub use buffer::{SampleBuffer, SharedSample};
pub use decode::{decode, ImportError, SampleDecoder, IMPORTABLE_EXTENSIONS};
pub use edit::{reverse, trim};
pub use onsets::detect_onsets;
pub use params::{ranges, ParamRange, SamplerParams, WAVETABLE_CUTOFF_HZ};
pub use record::{RecordState, Recorder};
pub use voice::{SamplerVoice, VoiceState, MAX_CHANNELS};
pub use wave::{MarkKind, WaveMark, WaveView, WAVE_COLUMNS};

/// The Just-a-Sample revision this crate's voice is a port of.
///
/// The manual's License page cites this exact revision — keep them in sync (see
/// the `license_page_cites_the_pinned_jas_revision` test in bloop-menu-pages).
pub const PORTED_FROM_JAS_REV: &str = "bc97747a514755ca686f8b1e5cd223e5b44866b4";

/// Encode interleaved 16-bit PCM stereo WAV bytes.
///
/// The pair of [`decode`], and the one encoder in the tree: both the
/// legacy app's export and the new manual's gold runner name their audio
/// by a hash of these bytes, so a second implementation would make the
/// same sound hash differently and break the comparison the port rests
/// on.
pub fn encode_wav16(samples_l: &[f32], samples_r: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples_l.len();
    let data_size = (num_samples * 2 * 2) as u32; // 2 channels × 2 bytes

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&2u16.to_le_bytes()); // stereo
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
    buf.extend_from_slice(&4u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
                                                 // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for i in 0..num_samples {
        let l = (samples_l[i].clamp(-1.0, 1.0) * 32767.0) as i16;
        let r = (samples_r[i].clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&l.to_le_bytes());
        buf.extend_from_slice(&r.to_le_bytes());
    }
    buf
}
