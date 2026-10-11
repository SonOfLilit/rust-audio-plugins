//! One test binary for the crate.
//!
//! Native only: on wasm32 no plug-in is compiled, so there is nothing
//! here to test.
#![cfg(not(target_arch = "wasm32"))]
#![cfg_attr(
    not(target_arch = "wasm32"),
    expect(
        clippy::needless_raw_string_hashes,
        reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
    )
)]

mod behaviour;
mod plugins;

use bloop_events::EventBuffer;
use bloop_plugin_airwindows_cpp::Effect;
use bloop_plugin_api::{Plugin, ProcessContext};

pub(crate) const RATE: f32 = 48_000.0;
pub(crate) const BLOCK: usize = 256;

/// Allocations this thread has made, Rust's and the C++'s: the test binary
/// wraps the allocator's entry points on Linux (build.rs). Elsewhere,
/// `None`.
#[cfg_attr(
    target_os = "linux",
    expect(
        clippy::unnecessary_wraps,
        reason = "None where the binary counts nothing: not on Linux"
    )
)]
pub(crate) fn allocations() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn bloop_aw_test_allocations() -> u64;
        }
        // Safety: reads a thread-local counter.
        Some(unsafe { bloop_aw_test_allocations() })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Uniform noise, -0.5 to 0.5, from `seed`.
pub(crate) fn noise(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "24 bits into a float, exactly")]
            let unit = (s >> 8) as f32 / (1u32 << 24) as f32;
            unit - 0.5
        })
        .collect()
}

/// A sine at `hz`, `len` frames at RATE, peak `level`.
pub(crate) fn sine(hz: f32, len: usize, level: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "a frame index as a time")]
            let t = i as f32 / RATE;
            level * (std::f32::consts::TAU * hz * t).sin()
        })
        .collect()
}

/// `left` and `right` through `effect`, BLOCK frames at a time, in place.
/// Allocates nothing.
pub(crate) fn run(effect: &mut Effect, events: &EventBuffer, left: &mut [f32], right: &mut [f32]) {
    for (l, r) in left.chunks_mut(BLOCK).zip(right.chunks_mut(BLOCK)) {
        let mut outputs: [&mut [f32]; 2] = [l, r];
        effect.process(&mut ProcessContext {
            outputs: &mut outputs,
            input_events: events,
            sample_rate: RATE,
            tempo: 120.0,
            is_playing: true,
        });
    }
}

/// Root-mean-square level in dB full scale.
pub(crate) fn rms_db(signal: &[f32]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "a length as a divisor")]
    let mean = signal.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / signal.len() as f64;
    10.0 * mean.max(1e-30).log10()
}

/// The largest magnitude.
pub(crate) fn peak(signal: &[f32]) -> f32 {
    signal.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}
