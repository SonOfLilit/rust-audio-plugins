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
#[ignore = "heavy: cargo test -p bloop-plugin-nam -- --include-ignored"]
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
        Clean     WaveNet at 48000 Hz, 11776 MAC/sample, loudness Some(-21.9), None by None
        Blues     WaveNet at 48000 Hz, 11776 MAC/sample, loudness Some(-23.0), None by None
        Crunch    WaveNet at 48000 Hz, 11776 MAC/sample, loudness Some(-22.9), None by None
        High Gain WaveNet at 48000 Hz, 11776 MAC/sample, loudness Some(-21.6), None by None
        read: 6 of 6"#]].assert_eq(&out);
}

/// What the reader refuses, and says why.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn what_the_reader_refuses() {
    let nano = Shape::Nano.nam(1);
    let a2 = Shape::A2Full.nam(1);
    let short = nano.replace(", 0.02]}", "]}");
    let first = |from: &str, to: &str| nano.replacen(from, to, 1);
    let activation = |to: &str| first(r#""activation": "Tanh""#, &format!(r#""activation": {to}"#));
    let container = |models: &[(f64, &str)]| {
        let held: Vec<String> = models
            .iter()
            .map(|(max, m)| format!(r#"{{"max_value": {max}, "model": {m}}}"#))
            .collect();
        format!(
            r#"{{"architecture": "SlimmableContainer", "config": {{"submodels": [{}]}}, "weights": []}}"#,
            held.join(", ")
        )
    };
    let cases = [
        ("not JSON", "hello".to_string()),
        ("JSON, not a model", "[1, 2]".to_string()),
        ("ConvNet", r#"{"architecture": "ConvNet", "config": {}, "weights": []}"#.to_string()),
        ("A2-Full", a2.clone()),
        ("a slimmed A2", a2.replace(r#""slimmable": null"#, r#""slimmable": {"method": "slice_channels_uniform"}"#)),
        ("slimmed otherwise", a2.replace(r#""slimmable": null"#, r#""slimmable": {"method": "prune"}"#)),
        ("a container", container(&[(0.5, &Shape::A2Lite.nam(1)), (1.0, &a2)])),
        ("out of order", container(&[(1.0, &a2), (0.5, &Shape::A2Lite.nam(1))])),
        ("short of 1", container(&[(0.5, &a2)])),
        ("in a container", container(&[(1.0, &container(&[(1.0, &a2)]))])),
        ("empty container", container(&[])),
        ("a weight short", short),
        ("a weight over", Shape::Nano.nam(1).replace("0.02]}", "0.02, 1.0]}")),
        ("stereo LSTM", r#"{"architecture": "LSTM", "config": {"input_size": 2, "hidden_size": 3, "num_layers": 1}, "weights": []}"#.to_string()),
        ("kernel 0", nano.replacen(r#""kernel_size": 3"#, r#""kernel_size": 0"#, 1)),
        ("dilation 0", nano.replacen("[1, 2, 4", "[0, 2, 4", 1)),
        ("kernel 300", nano.replacen(r#""kernel_size": 3"#, r#""kernel_size": 300"#, 1)),
        ("dilation 2^20", nano.replacen("[1, 2, 4", "[1048576, 2, 4", 1)),
        ("history of 40x16k", nano.replacen("[1, 2, 4", &format!("[{}1, 2, 4", "8000, ".repeat(40)), 1)),
        ("a broken head", nano.replace(r#""head": null"#, r#""head": {"out_channels": 1}"#)),
        ("a stereo head", nano.replace(r#""head": null"#, r#""head": {"channels": 2, "out_channels": 2, "kernel_sizes": [1], "activation": "Tanh"}"#)),
        ("an LSTM condition", nano.replace(r#""head": null"#, r#""head": null, "condition_dsp": {"architecture": "LSTM"}"#)),
        ("65 channels", first(r#""channels": 4"#, r#""channels": 65"#)),
        ("stereo out", nano.replace(r#""head_size": 1"#, r#""head_size": 2"#)),
        ("unknown activation", activation(r#""Swish7""#)),
        ("activation object", activation(r#"{"type": "LeakyReLU", "negative_slope": 0.2}"#)),
        ("activation, no type", activation(r#"{"kind": "Tanh"}"#)),
        ("PReLU of 3 for 4", activation(r#"{"type": "PReLU", "negative_slopes": [0.1, 0.2, 0.3]}"#)),
        ("gated PReLU of 1", first(r#""gated": false"#, r#""gated": true"#).replacen(r#""activation": "Tanh""#, r#""activation": {"type": "PReLU", "negative_slope": 0.1}"#, 1)),
        ("unknown gating", first(r#""gated": false"#, r#""gating_mode": "sideways""#)),
        ("two kernel sizes", first(r#""kernel_size": 3"#, r#""kernel_size": 3, "kernel_sizes": [3]"#)),
        ("no layer 1x1", first(r#""gated": false"#, r#""gated": false, "bottleneck": 2, "layer1x1": {"active": false, "groups": 1}"#)),
        ("3 groups of 4", first(r#""gated": false"#, r#""gated": false, "groups_input": 3"#)),
        ("mismatched arrays", first(r#""input_size": 4"#, r#""input_size": 3"#)),
    ];
    let mut out = String::new();
    for (label, text) in cases {
        let said = match Model::parse(&text) {
            Ok(_) => "read".to_string(),
            Err(e) => e.chars().take(100).collect(),
        };
        let _ = writeln!(out, "{label:<19} {said}");
    }
    expect![[r#"
        not JSON            not a .nam file: expected value at line 1 column 1
        JSON, not a model   not a .nam file: it is JSON, but not a model
        ConvNet             bloop does not run ConvNet models yet
        A2-Full             read
        a slimmed A2        read
        slimmed otherwise   layer array 0 slims by "prune", which bloop does not run
        a container         read
        out of order        the container's models are not in ascending max_value
        short of 1          the container's last model does not reach a max_value of 1
        in a container      bloop does not run a container inside a container
        empty container     the container holds no models
        a weight short      the model has 841 weights, fewer than its layers need
        a weight over       the model has 843 weights, but its layers use 842
        stereo LSTM         the LSTM takes 2 inputs; bloop runs mono models
        kernel 0            layer array 0 has a kernel of 0
        dilation 0          layer array 0 has a dilation of 0
        kernel 300          a convolution has a kernel of 300; bloop runs up to 256
        dilation 2^20       the WaveNet's layer array 0's dilations is 1048576, more than bloop runs
        history of 40x16k   the model would hold more history than bloop runs
        a broken head       the head's kernel_sizes is not a list
        a stereo head       the model outputs 2 channels; bloop runs mono models
        an LSTM condition   bloop runs only WaveNets as a condition DSP
        65 channels         layer array 0 has 65 channels; bloop runs 1 to 64
        stereo out          the model outputs 2 channels; bloop runs mono models
        unknown activation  bloop does not run the Swish7 activation yet
        activation object   read
        activation, no type an activation is neither a name nor an object with a type
        PReLU of 3 for 4    a PReLU has 3 negative slopes for 4 channels
        gated PReLU of 1    a PReLU has 1 negative slopes for 4 channels
        unknown gating      layer array 0 has a gating mode NAM does not have
        two kernel sizes    layer array 0 gives both a kernel_size and kernel_sizes
        no layer 1x1        layer array 0 has no layer 1x1, so its bottleneck must be its channels
        3 groups of 4       a convolution of 4 channels into 4 cannot be split into 3 groups
        mismatched arrays   layer array 1 takes 3 channels, but the array before it outputs 4
    "#]]
    .assert_eq(&out);
}

/// The sizes a capture comes in: what each costs, how loud it says it
/// is, and which slim value NAM's core would be given for it. A model of
/// one size runs it whichever is asked for.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn every_size_of_a_capture() {
    use bloop_plugin_nam::Size;
    let models = [
        ("a2-container", Shape::A2Container.nam(1)),
        ("a2-full", Shape::A2Full.nam(1)),
        ("slimmable wavenet", include_str!("models/slimmable_wavenet.nam").to_string()),
        ("slimmable container", include_str!("models/slimmable_container.nam").to_string()),
        ("standard", Shape::Standard.nam(1)),
    ];
    let mut out = String::new();
    for (name, text) in models {
        let mut model = Model::parse(&text).unwrap();
        let _ = writeln!(out, "{name}: has a light size: {}", model.has_light());
        for size in [Size::Full, Size::Light] {
            model.set_size(size);
            let _ = writeln!(
                out,
                "  {size:?}: runs {:?} at slim {:.3}, {} MAC/sample, prewarm {}, loudness {:?}",
                model.size(),
                model.slim_value(),
                model.macs_per_sample(),
                model.prewarm_samples(),
                model.loudness().map(|l| (l * 10.0).round() / 10.0),
            );
        }
    }
    expect![[r#"
        a2-container: has a light size: true
          Full: runs Full at slim 1.000, 11776 MAC/sample, prewarm 6347, loudness Some(-20.0)
          Light: runs Light at slim 0.250, 1731 MAC/sample, prewarm 6347, loudness Some(-20.0)
        a2-full: has a light size: false
          Full: runs Full at slim 1.000, 11776 MAC/sample, prewarm 6347, loudness Some(-20.0)
          Light: runs Full at slim 1.000, 11776 MAC/sample, prewarm 6347, loudness Some(-20.0)
        slimmable wavenet: has a light size: true
          Full: runs Full at slim 1.000, 396 MAC/sample, prewarm 2047, loudness None
          Light: runs Light at slim 0.500, 184 MAC/sample, prewarm 2047, loudness None
        slimmable container: has a light size: true
          Full: runs Full at slim 1.000, 732 MAC/sample, prewarm 4093, loudness Some(-25.5)
          Light: runs Light at slim 0.495, 113 MAC/sample, prewarm 23, loudness Some(-20.0)
        standard: has a light size: false
          Full: runs Full at slim 1.000, 13320 MAC/sample, prewarm 4093, loudness None
          Light: runs Full at slim 1.000, 13320 MAC/sample, prewarm 4093, loudness None
    "#]].assert_eq(&out);
}
