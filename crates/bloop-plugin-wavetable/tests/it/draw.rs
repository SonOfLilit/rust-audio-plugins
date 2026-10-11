//! What a screen draws a Wavetable from (`bloop_plugin_wavetable::draw`):
//! a table's frames as the oscillator reads them, an LFO's curve, the
//! filter's response.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use std::fmt::Write as _;

use bloop_plugin_wavetable::draw::{self, LfoShape, SAMPLES};
use bloop_plugin_wavetable::dsp::tables::{Table, Wavetable};
use bloop_plugin_wavetable::params::id;
use expect_test::expect;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

/// A cycle's samples at eight places through it.
fn eighths(cycle: &[f32]) -> String {
    (0..8)
        .map(|i| format!("{:+.2}", cycle[i * cycle.len() / 8]))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Three frames, 256 samples each: a sine, a saw, silence.
fn table() -> Table {
    let size = 256;
    let samples: Vec<f32> = (0..3)
        .flat_map(|frame| {
            (0..size).map(move |i| {
                let t = i as f32 / size as f32;
                match frame {
                    0 => libm::sinf(t * core::f32::consts::TAU),
                    1 => 1.0 - 2.0 * t,
                    _ => 0.0,
                }
            })
        })
        .collect();
    Table::Exact(Wavetable::load(48_000.0, &samples, 48_000.0, size))
}

/// Every frame, one cycle of it, as the oscillator reads it: a sine a
/// sine, a saw falling, silence silent; Bend up reads the cycle ahead of
/// itself (the sine's trough comes early), Bend down behind (its peak
/// late), and Formant reads it faster and holds it at its end.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn each_frame_is_a_cycle_as_the_oscillator_reads_it() {
    let table = table();
    let mut out = String::new();
    for (what, bend, formant) in [
        ("plain", 0.0, 0.0),
        ("bend +0.8", 0.8, 0.0),
        ("bend -0.8", -0.8, 0.0),
        ("formant 0.5", 0.0, 0.5),
    ] {
        let cycles = draw::cycles(&table, bend, formant);
        let _ = writeln!(out, "{what}: {} frames of {SAMPLES}", cycles.len());
        for (frame, cycle) in cycles.iter().enumerate() {
            let _ = writeln!(out, "  {frame}: {}", eighths(cycle));
        }
    }
    let empty = Table::Exact(Wavetable::from_frames(Vec::new()));
    let _ = write!(
        out,
        "an empty table: {} frames",
        draw::cycles(&empty, 0.0, 0.0).len()
    );
    expect![[r#"
        plain: 3 frames of 64
          0: +0.00 +0.70 +0.98 +0.68 -0.03 -0.73 -1.01 -0.71
          1: +0.25 +0.71 +0.47 +0.22 -0.03 -0.27 -0.51 -0.74
          2: +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
        bend +0.8: 3 frames of 64
          0: +0.00 -0.32 -0.99 -0.77 -0.58 -0.43 -0.28 -0.12
          1: +0.25 -0.10 -0.53 -0.73 -0.76 -0.84 -0.93 -0.95
          2: +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
        bend -0.8: 3 frames of 64
          0: +0.00 +0.16 +0.31 +0.45 +0.59 +0.76 +0.96 +0.27
          1: +0.25 +0.89 +0.92 +0.85 +0.76 +0.68 +0.49 +0.04
          2: +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
        formant 0.5: 3 frames of 64
          0: +0.00 +0.97 -0.37 -0.85 +0.00 +0.00 +0.00 +0.00
          1: +0.25 +0.42 -0.12 -0.66 +0.25 +0.24 +0.24 +0.24
          2: +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
        an empty table: 0 frames"#]]
    .assert_eq(&out);
}

/// An LFO's curve, through the port's own LFO: a sine and a square at one
/// cycle and at two, at half depth, offset, and started a quarter in.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn an_lfo_is_drawn_as_it_runs() {
    let shape = |wave: f32| LfoShape {
        wave,
        cycles: 1.0,
        phase: 0.0,
        offset: 0.0,
        depth: 1.0,
    };
    let sine = 1.0;
    let square = 5.0;
    let mut out = String::new();
    for (what, s) in [
        ("sine", shape(sine)),
        (
            "sine, two cycles",
            LfoShape {
                cycles: 2.0,
                ..shape(sine)
            },
        ),
        (
            "sine at half depth",
            LfoShape {
                depth: 0.5,
                ..shape(sine)
            },
        ),
        (
            "sine offset +0.5",
            LfoShape {
                offset: 0.5,
                ..shape(sine)
            },
        ),
        (
            "sine from a quarter in",
            LfoShape {
                phase: 0.25,
                ..shape(sine)
            },
        ),
        ("square", shape(square)),
        ("saw up", shape(3.0)),
        (
            "triangle, a cycle and a half",
            LfoShape {
                cycles: 1.5,
                ..shape(2.0)
            },
        ),
    ] {
        let _ = writeln!(out, "{what:<24} {}", eighths(&draw::lfo(s, 16)));
    }
    expect![[r#"
        sine                     +0.00 +0.71 +1.00 +0.71 -0.00 -0.71 -1.00 -0.71
        sine, two cycles         +0.00 +1.00 -0.00 -1.00 +0.00 +1.00 -0.00 -1.00
        sine at half depth       +0.00 +0.35 +0.50 +0.35 -0.00 -0.35 -0.50 -0.35
        sine offset +0.5         +0.50 +1.00 +1.00 +1.00 +0.50 -0.21 -0.50 -0.21
        sine from a quarter in   +1.00 +0.71 -0.00 -0.71 -1.00 -0.71 +0.00 +0.71
        square                   +1.00 +1.00 +1.00 +1.00 -1.00 -1.00 -1.00 -1.00
        saw up                   -1.00 -0.75 -0.50 -0.25 +0.00 +0.25 +0.50 +0.75
        triangle, a cycle and a half +0.00 +0.75 +0.50 -0.25 -1.00 -0.25 +0.50 +0.75
    "#]]
    .assert_eq(&out);
}

/// The filter's response, as a factor at 50 Hz, the cutoff and 5 kHz:
/// each type at C4's cutoff (261.6 Hz), a low-pass with resonance peaking
/// there; and two knobs' values as the DSP sees them.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_filter_responds_as_it_filters() {
    let freqs = [50.0, 261.6, 5000.0];
    let mut out = String::new();
    for (what, kind, res) in [
        ("LP 12", 0.0, 0.0),
        ("LP 24", 1.0, 0.0),
        ("HP 12", 2.0, 0.0),
        ("BP 12", 4.0, 0.0),
        ("NT 12", 6.0, 0.0),
        ("LP 12, resonance 80%", 0.0, 80.0),
    ] {
        let gains = draw::filter_response(kind, 60.0, res, &freqs);
        let _ = writeln!(
            out,
            "{what:<22} {}",
            gains
                .iter()
                .map(|g| format!("{g:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    let _ = write!(
        out,
        "the amp's Sustain knob at 50% is {}; OSC 1's Level at -6 dB is a gain of {:.3}",
        draw::converted(id::AMP_SUSTAIN, 50.0),
        draw::converted(id::OSC[0] + id::osc::LEVEL, -6.0)
    );
    expect![[r#"
        LP 12                  0.999 0.707 0.003
        LP 24                  0.999 0.500 0.000
        HP 12                  0.036 0.707 1.000
        BP 12                  0.270 1.000 0.074
        NT 12                  0.963 0.000 0.997
        LP 12, resonance 80%   1.036 3.400 0.003
        the amp's Sustain knob at 50% is 0.5; OSC 1's Level at -6 dB is a gain of 0.501"#]]
    .assert_eq(&out);
}
