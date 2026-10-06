//! bloop-sampler-engine: sample playback, ported from Just-a-Sample.
//!
//! The voice is a 1:1 Rust port of `CustomSamplerVoice` from
//! [Just-a-Sample](https://github.com/BOBONA/Just-a-Sample) (MIT, © 2024
//! Binyamin Friedman), pinned at [`PORTED_FROM_JAS_REV`]. Agreement with the
//! C++ original is enforced by the gold tests in `tests/it/gold.rs`, which compile
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
pub mod edit;
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
pub use params::{
    ranges, AmpEnvelope, ParamRange, SamplerParams, STEADY_LOWPASS_OF_NYQUIST, WAVETABLE_CUTOFF_HZ,
};
pub use record::{RecordState, Recorder};
pub use voice::{SamplerVoice, VoiceState, MAX_CHANNELS};
pub use wave::{MarkKind, WaveMark, WaveView, WAVE_COLUMNS};

/// The Just-a-Sample revision this crate's voice is a port of.
///
/// The manual's License page cites this exact revision — keep them in sync (see
/// `every_notice_cites_its_pinned_revision` in bloop-synths).
pub const PORTED_FROM_JAS_REV: &str = "bc97747a514755ca686f8b1e5cd223e5b44866b4";

/// The left and right gains for a Zone's pan, -1 (left) to 1 (right).
///
/// The mixer's equal-power law (`bloop_mixer::pan_law`), restated here
/// because a plug-in sits below the mixer, and scaled so the centre is
/// unity rather than -3 dB: an unpanned Zone plays exactly as it
/// did before it had a pan, and a panned one keeps its power. `libm`, not
/// `std`, so a panned Zone renders the same on every target.
#[must_use]
pub fn pan_gains(pan: f32) -> (f32, f32) {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * core::f32::consts::FRAC_PI_4;
    (
        (libm::cosf(angle) * core::f32::consts::SQRT_2).max(0.0),
        (libm::sinf(angle) * core::f32::consts::SQRT_2).max(0.0),
    )
}

/// Encode interleaved 16-bit PCM stereo WAV bytes.
///
/// The pair of [`decode`], and the one encoder in the tree: both the
/// legacy app's export and the new manual's gold runner name their audio
/// by a hash of these bytes, so a second implementation would make the
/// same sound hash differently and break the comparison the port rests
/// on.
pub fn encode_wav16(samples_l: &[f32], samples_r: &[f32], sample_rate: u32) -> Vec<u8> {
    encode_wav(samples_l, samples_r, sample_rate, WavFormat::Int16)
}

/// How a WAV stores its samples: the Maschine's three Export Audio bit
/// depths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavFormat {
    /// 16-bit PCM, what a CD holds; full scale is ±32767.
    Int16,
    /// 24-bit PCM, for mastering; full scale is ±8388607.
    Int24,
    /// 32-bit IEEE float, the engine's own samples as they are — nothing
    /// above full scale is clipped.
    Float32,
}

impl WavFormat {
    const fn bytes(self) -> u16 {
        match self {
            WavFormat::Int16 => 2,
            WavFormat::Int24 => 3,
            WavFormat::Float32 => 4,
        }
    }
}

/// Encode stereo WAV bytes in `format`.
///
/// PCM is clamped to full scale. 16-bit is truncated toward zero, as it
/// always has been: every gold hash in the manual is a hash of those
/// bytes. 24-bit, which nothing hashes, is rounded to the nearest step,
/// so its error is half a step rather than one. A float file is `WAVE_FORMAT_IEEE_FLOAT`,
/// with the `cbSize` and `fact` chunk the format asks of anything that
/// is not PCM.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a WAV's sizes are 32-bit by definition; a file over 4 GB is not one"
)]
pub fn encode_wav(
    samples_l: &[f32],
    samples_r: &[f32],
    sample_rate: u32,
    format: WavFormat,
) -> Vec<u8> {
    let frames = samples_l.len().min(samples_r.len());
    let bytes = format.bytes();
    let align = 2 * bytes;
    let data_size = (frames * usize::from(align)) as u32;
    let float = format == WavFormat::Float32;
    // fmt is 16 bytes for PCM; 18, with a cbSize of 0, for float, which
    // also has a 12-byte fact chunk.
    let fmt_size: u32 = if float { 18 } else { 16 };
    let fact_size: u32 = if float { 12 } else { 0 };
    let mut buf = Vec::with_capacity(44 + (fmt_size - 16 + fact_size + data_size) as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(4 + 8 + fmt_size + fact_size + 8 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&fmt_size.to_le_bytes());
    buf.extend_from_slice(&(if float { 3u16 } else { 1u16 }).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * u32::from(align)).to_le_bytes());
    buf.extend_from_slice(&align.to_le_bytes());
    buf.extend_from_slice(&(bytes * 8).to_le_bytes());
    if float {
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(b"fact");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(frames as u32).to_le_bytes());
    }
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for (&l, &r) in samples_l.iter().zip(samples_r).take(frames) {
        for sample in [l, r] {
            match format {
                WavFormat::Int16 => {
                    buf.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
                }
                WavFormat::Int24 => {
                    let value = (f64::from(sample.clamp(-1.0, 1.0)) * 8_388_607.0).round() as i32;
                    buf.extend_from_slice(&value.to_le_bytes()[..3]);
                }
                WavFormat::Float32 => buf.extend_from_slice(&sample.to_le_bytes()),
            }
        }
    }
    buf
}

