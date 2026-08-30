//! Value-level tests that run on every target, including WASM.
//!
//! The gold suite (`gold.rs`) is the specification, but it shells out to a C++
//! compiler and so is native-only. These snapshots pin the port's actual output
//! samples, so a divergence between native and WASM — different `exp`, `sin` or
//! `powf` implementations, say — shows up as a failing snapshot rather than as
//! a synth that quietly detunes itself in the browser.

use bloop_sampler_engine::{SampleBuffer, SamplerParams, SamplerVoice};
use expect_test::{expect, Expect};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

/// A short ramp, whose exact values make interpolation errors obvious.
fn ramp(frames: usize) -> SampleBuffer {
    let data = (0..frames)
        .map(|i| (i as f32 / (frames - 1) as f32) * 2.0 - 1.0)
        .collect();
    SampleBuffer::new(vec![data], 48_000)
}

/// Renders `frames` frames of one note and snapshots every output sample.
fn check(params: SamplerParams, note: u8, frames: usize, expect: Expect) {
    let sample = ramp(4800);
    let rendered: Vec<String> = render_once(&sample, &params, note, 1.0, frames)
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect();
    expect.assert_eq(&rendered.join(" "));
}

/// Envelope-free playback of `sample_start..=sample_end`.
fn flat(sample_start: i32, sample_end: i32) -> SamplerParams {
    SamplerParams {
        sample_start,
        sample_end,
        loop_end: sample_end,
        attack_ms: 0.0,
        release_ms: 0.0,
        ..SamplerParams::default()
    }
}

/// At the root note the voice reads consecutive frames, so the output is the
/// ramp itself: even steps of 2/4799 ≈ 0.000417 per frame.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn root_note_reproduces_the_ramp() {
    check(flat(2400, 4799), 69, 16, expect!["0.000208 0.000625 0.001042 0.001459 0.001875 0.002292 0.002709 0.003126 0.003542 0.003959 0.004376 0.004793 0.005209 0.005626 0.006043 0.006460"]);
}

/// An octave down reads every other frame, so odd outputs land halfway between
/// ramp steps — the Lanczos kernel interpolating, well away from the buffer's
/// edges.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn an_octave_down_interpolates_between_frames() {
    check(flat(2400, 4799), 57, 16, expect!["0.000208 0.000416 0.000625 0.000832 0.001042 0.001249 0.001459 0.001665 0.001875 0.002081 0.002292 0.002497 0.002709 0.002914 0.003126 0.003330"]);
}

/// Near the start of the buffer the kernel's window runs off the front, and the
/// reference reads those taps as zero rather than clamping or mirroring. That
/// rings for the first few frames — audible as a click on a slice whose start
/// lands mid-waveform, and the reason slice starts want a short attack.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn interpolating_at_the_buffer_edge_rings() {
    check(flat(0, 4799), 57, 12, expect!["-1.000000 -1.125401 -0.999582 -0.942776 -0.999167 -1.023439 -0.998749 -0.989555 -0.998333 -0.996873 -0.997916 -0.996457"]);
}

/// Lo-fi mode truncates to the nearest frame instead, so the same run repeats
/// each ramp value twice and never rings.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn lo_fi_mode_repeats_frames_instead_of_interpolating() {
    check(
        SamplerParams { skip_antialiasing: true, ..flat(0, 4799) },
        57,
        16,
        expect!["-1.000000 -1.000000 -0.999583 -0.999583 -0.999166 -0.999166 -0.998750 -0.998750 -0.998333 -0.998333 -0.997916 -0.997916 -0.997499 -0.997499 -0.997083 -0.997083"],
    );
}

/// A linear attack scales the opening frames by `i / attack_frames`, which the
/// ratio against an attack-free render shows directly.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_linear_attack_ramps_the_gain_from_zero() {
    let sample = ramp(4800);
    let dry = SamplerParams {
        sample_start: 2400,
        ..flat(2400, 4799)
    };
    // 0.25 ms at 48 kHz is 12 frames, so the ramp completes inside the block.
    let shaped = SamplerParams {
        attack_ms: 0.25,
        attack_shape: 0.0,
        ..dry
    };

    let gains = gain_ratio(&sample, &dry, &shaped, 16);
    expect!["0.0000 0.0833 0.1667 0.2500 0.3333 0.4167 0.5000 0.5833 0.6667 0.7500 0.8333 0.9167 1.0000 1.0000 1.0000 1.0000"].assert_eq(&gains);
}

/// Every sample scales by the velocity curve, which works out to velocity
/// squared.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_velocity_curve_is_velocity_squared() {
    let sample = ramp(4800);
    let params = flat(2400, 4799);

    let full = render_once(&sample, &params, 69, 1.0, 4);
    let half = render_once(&sample, &params, 69, 0.5, 4);

    let ratios: Vec<String> = (0..4)
        .map(|i| format!("{:.4}", half[i] / full[i]))
        .collect();
    expect!["0.2500 0.2500 0.2500 0.2500"].assert_eq(&ratios.join(" "));
}

fn render_once(
    sample: &SampleBuffer,
    params: &SamplerParams,
    note: u8,
    velocity: f32,
    frames: usize,
) -> Vec<f32> {
    let mut voice = SamplerVoice::new(48_000.0, frames, sample.num_channels());
    let mut output = vec![vec![0.0f32; frames]; 2];
    assert!(
        voice.note_on(sample, params, note, velocity, 8192),
        "note must start"
    );
    {
        let (left, right) = output.split_at_mut(1);
        let mut out: [&mut [f32]; 2] = [&mut left[0], &mut right[0]];
        voice.render(sample, params, &mut out, 0, frames);
    }
    output.swap_remove(0)
}

/// The per-frame gain `shaped` applies relative to `dry`.
fn gain_ratio(
    sample: &SampleBuffer,
    dry: &SamplerParams,
    shaped: &SamplerParams,
    frames: usize,
) -> String {
    let plain = render_once(sample, dry, 69, 1.0, frames);
    let enveloped = render_once(sample, shaped, 69, 1.0, frames);
    (0..frames)
        .map(|i| format!("{:.4}", enveloped[i] / plain[i]))
        .collect::<Vec<_>>()
        .join(" ")
}
