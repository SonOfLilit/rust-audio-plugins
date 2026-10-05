//! The port run on real signals: what a sine at a few levels comes out as
//! (its level and harmonics), what silence settles to, how the fast and
//! exact `tanh` differ, and what the reader refuses.

#![expect(
    clippy::unwrap_used,
    reason = "a test fails loudly, and that is the point"
)]
#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

use std::fmt::Write as _;

use bloop_plugin_nam::preset::Shape;
use bloop_plugin_nam::{Model, Tanh};
use expect_test::{expect, Expect};

const RATE: f32 = 48_000.0;
/// 100 Hz: 480 samples a period at 48 kHz.
const F0: f32 = 100.0;
const PERIOD: usize = 480;

const STANDARD: &str = include_str!("../models/example.nam");
const TINY_WAVENET: &str = include_str!("models/wavenet.nam");
const TINY_LSTM: &str = include_str!("models/lstm.nam");

fn sine(amplitude: f32, frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|i| amplitude * libm::sinf(2.0 * core::f32::consts::PI * F0 * i as f32 / RATE))
        .collect()
}

fn db(x: f32) -> f32 {
    20.0 * libm::log10f(x.max(1e-12))
}

/// The amplitude of harmonic `k` of a signal a whole number of periods
/// long.
fn harmonic(x: &[f32], k: usize) -> f32 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &v) in x.iter().enumerate() {
        let phase = 2.0 * core::f64::consts::PI * (k * i) as f64 / PERIOD as f64;
        re += f64::from(v) * libm::cos(phase);
        im += f64::from(v) * libm::sin(phase);
    }
    (2.0 * libm::sqrt(re * re + im * im) / x.len() as f64) as f32
}

/// A second of a 100 Hz sine at each of `levels` through the model, from
/// fresh: the output's peak, its DC, and its first five harmonics in dB
/// (H1 absolute, the rest relative to it), over the last half second.
fn check_sines(text: &str, tanh: Tanh, levels: &[f32], expect: &Expect) {
    let mut out = String::new();
    for &level in levels {
        let mut model = Model::parse(text).unwrap();
        model.reset(tanh);
        let input = sine(level, 48_000);
        let mut output = vec![0.0; input.len()];
        model.process(&input, &mut output, tanh);
        let tail = &output[24_000..];
        let peak = tail.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let dc = tail.iter().sum::<f32>() / tail.len() as f32;
        let h1 = harmonic(tail, 1);
        let rel: Vec<String> = (2..=5)
            .map(|k| format!("{:6.1}", db(harmonic(tail, k)) - db(h1)))
            .collect();
        let _ = writeln!(
            out,
            "in {:5.1} dB: peak {:6.1} dB  dc {dc:+.4}  H1 {:6.1} dB  H2..H5 {}",
            db(level),
            db(peak),
            db(h1),
            rel.join(" ")
        );
    }
    expect.assert_eq(&out);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_example_standard_capture_distorts_more_as_it_is_driven_harder() {
    check_sines(
        STANDARD,
        Tanh::Exact,
        &[0.01, 0.1, 0.3, 1.0],
        &expect![[r#"
        in -40.0 dB: peak   -3.7 dB  dc +0.0068  H1   -3.4 dB  H2..H5  -51.3  -25.9  -42.9  -48.7
        in -20.0 dB: peak   -2.9 dB  dc -0.0052  H1   -2.4 dB  H2..H5  -32.9  -22.4  -34.8  -32.6
        in -10.5 dB: peak   -3.0 dB  dc -0.0091  H1   -2.5 dB  H2..H5  -27.1  -22.5  -29.8  -34.9
        in   0.0 dB: peak   -3.7 dB  dc -0.0119  H1   -3.3 dB  H2..H5  -28.1  -27.0  -39.5  -39.1
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_fast_tanh_sounds_the_same_to_a_tenth_of_a_db() {
    check_sines(
        STANDARD,
        Tanh::Fast,
        &[0.01, 0.1, 0.3, 1.0],
        &expect![[r#"
        in -40.0 dB: peak   -3.7 dB  dc +0.0069  H1   -3.4 dB  H2..H5  -49.5  -25.9  -43.1  -48.7
        in -20.0 dB: peak   -2.9 dB  dc -0.0052  H1   -2.4 dB  H2..H5  -32.8  -22.4  -34.7  -32.7
        in -10.5 dB: peak   -3.0 dB  dc -0.0091  H1   -2.5 dB  H2..H5  -27.1  -22.6  -29.8  -34.9
        in   0.0 dB: peak   -3.7 dB  dc -0.0119  H1   -3.3 dB  H2..H5  -28.1  -27.0  -39.5  -39.1
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_tiny_examples_run() {
    check_sines(
        TINY_WAVENET,
        Tanh::Exact,
        &[0.1, 1.0],
        &expect![[r#"
        in -20.0 dB: peak  -38.7 dB  dc -0.0005  H1  -39.2 dB  H2..H5  -38.9  -38.8  -76.7  -75.7
        in   0.0 dB: peak  -23.5 dB  dc -0.0010  H1  -29.0 dB  H2..H5  -19.1   -3.2  -26.2  -11.4
    "#]],
    );
    check_sines(
        TINY_LSTM,
        Tanh::Exact,
        &[0.1, 1.0],
        &expect![[r#"
        in -20.0 dB: peak  -41.5 dB  dc -0.0009  H1  -42.5 dB  H2..H5  -53.7  -84.4 -120.3 -130.5
        in   0.0 dB: peak  -22.4 dB  dc +0.0005  H1  -22.5 dB  H2..H5  -35.5  -44.3  -65.5  -75.2
    "#]],
    );
}

/// How far NAM's fast `tanh` is from the exact one, over the range a
/// layer's sums reach, and where.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_fast_tanh_is_within_half_a_thousandth() {
    let (mut worst, mut at) = (0.0f32, 0.0f32);
    for i in -80_000..=80_000 {
        let x = i as f32 / 10_000.0;
        let d = (bloop_plugin_nam::act::fast_tanh(x) - libm::tanhf(x)).abs();
        if d > worst {
            (worst, at) = (d, x);
        }
    }
    expect!["largest difference 4.4e-4 at x = -0.580"]
        .assert_eq(&format!("largest difference {worst:.1e} at x = {at:.3}"));
}

/// The provenance comment in Cargo.toml and the shipped captures' notes
/// name the revisions the License page cites, so they cannot drift apart.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn cargo_toml_and_the_captures_cite_the_ported_revisions() {
    let manifest = include_str!("../Cargo.toml");
    let bundled = include_str!("../src/bundled.rs");
    for rev in [
        bloop_plugin_nam::PORTED_FROM_NAM_REV,
        bloop_plugin_nam::NAM_PLUGIN_REV,
        bloop_plugin_nam::AUDIO_DSP_TOOLS_REV,
    ] {
        assert!(manifest.contains(rev), "Cargo.toml must cite {rev}");
    }
    assert!(bundled.contains(bloop_plugin_nam::NAM_PLUGIN_REV));
}

/// Every shipped capture reads, and is what its notes say it is.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn every_shipped_capture_reads() {
    use bloop_plugin_nam::bundled::{models, CAPTURES};
    let mut out = String::new();
    for (capture, model) in CAPTURES.iter().zip(models()) {
        let m = &model.metadata;
        let _ = writeln!(
            out,
            "{:<9} {} at {} Hz, {} MAC/sample, loudness {:?}, {:?} by {:?}",
            capture.name,
            model.architecture,
            model.sample_rate,
            model.macs_per_sample(),
            m.loudness.map(|l| (l * 10.0).round() / 10.0),
            m.gear_model,
            m.modeled_by,
        );
    }
    let _ = write!(out, "read: {} of {}", models().len(), CAPTURES.len());
    expect![[r#"
        Obsidian  WaveNet at 48000 Hz, 13320 MAC/sample, loudness Some(-28.2), Some("Obsidian") by Some("Steven Atkinson")
        Example   WaveNet at 48000 Hz, 13320 MAC/sample, loudness None, None by None
        read: 2 of 2"#]].assert_eq(&out);
}

/// What the reader refuses, and says why.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn what_the_reader_refuses() {
    let a2 = r#"{"architecture": "WaveNet", "config": {"layers": [{"input_size": 1, "condition_size": 1, "head_size": 1, "channels": 3, "kernel_sizes": [3], "dilations": [1], "activation": "Tanh", "head_bias": false}], "head": null, "head_scale": 0.02}, "weights": []}"#;
    let short = Shape::Nano.nam(1).replace(", 0.02]}", "]}");
    let cases = [
        ("not JSON", "hello".to_string()),
        ("ConvNet", r#"{"architecture": "ConvNet", "config": {}, "weights": []}"#.to_string()),
        ("an A2 layer", a2.to_string()),
        ("a weight short", short),
        ("a weight over", Shape::Nano.nam(1).replace("0.02]}", "0.02, 1.0]}")),
        ("stereo LSTM", r#"{"architecture": "LSTM", "config": {"input_size": 2, "hidden_size": 3, "num_layers": 1}, "weights": []}"#.to_string()),
    ];
    let mut out = String::new();
    for (label, text) in cases {
        let said = match Model::parse(&text) {
            Ok(_) => "read".to_string(),
            Err(e) => e.chars().take(90).collect(),
        };
        let _ = writeln!(out, "{label:<15} {said}");
    }
    expect![[r#"
        not JSON        not a .nam file: expected value at line 1 column 1
        ConvNet         bloop does not run ConvNet models yet
        an A2 layer     layer array 0 uses "kernel_sizes", which bloop does not run yet
        a weight short  the model has 841 weights, fewer than its layers need
        a weight over   the model has 843 weights, but its layers use 842
        stereo LSTM     the LSTM takes 2 inputs; bloop runs mono models
    "#]]
    .assert_eq(&out);
}
