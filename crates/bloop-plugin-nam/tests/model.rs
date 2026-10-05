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
