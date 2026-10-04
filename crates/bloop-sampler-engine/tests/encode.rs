//! Export's encoder: stereo WAV at 16-bit, 24-bit and 32-bit float, read
//! back through the same decoder an import goes through.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use bloop_sampler_engine::{decode, encode_wav, WavFormat};
use expect_test::{expect, Expect};
use std::fmt::Write as _;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

/// A second of a 1 kHz sine at half scale on the left and a 3 kHz one at
/// a tenth on the right, then the edges: full scale both ways, past it
/// both ways, the smallest 24-bit step, and silence.
fn signal() -> (Vec<f32>, Vec<f32>) {
    let rate = 48_000.0f32;
    let mut left: Vec<f32> = (0..48_000)
        .map(|i| 0.5 * libm::sinf(2.0 * core::f32::consts::PI * 1_000.0 * i as f32 / rate))
        .collect();
    let mut right: Vec<f32> = (0..48_000)
        .map(|i| 0.1 * libm::sinf(2.0 * core::f32::consts::PI * 3_000.0 * i as f32 / rate))
        .collect();
    for (l, r) in [
        (1.0, -1.0),
        (1.5, -1.5),
        (1.0 / 8_388_608.0, 0.0),
        (0.0, 0.0),
    ] {
        left.push(l);
        right.push(r);
    }
    (left, right)
}

/// The header as the file says it; what came back, and how far each side
/// is from what went in over the sine, in the format's own steps; then
/// the four edge frames as decoded.
fn check(format: WavFormat, expect: Expect) {
    let (left, right) = signal();
    let wav = encode_wav(&left, &right, 48_000, format);
    let word = |at: usize| u16::from_le_bytes([wav[at], wav[at + 1]]);
    let long = |at: usize| u32::from_le_bytes([wav[at], wav[at + 1], wav[at + 2], wav[at + 3]]);
    let fmt_len = long(16) as usize;
    let mut out = format!(
        "{} bytes, fmt chunk {fmt_len} bytes: tag {}, {} ch, {} Hz, {} bytes/s, align {}, {} bits\n",
        wav.len(),
        word(20),
        word(22),
        long(24),
        long(28),
        word(32),
        word(34),
    );
    let mut at = 12;
    let mut chunks = Vec::new();
    while at + 8 <= wav.len() {
        let id = String::from_utf8_lossy(&wav[at..at + 4]).to_string();
        let len = long(at + 4) as usize;
        chunks.push(format!("{id}({len})"));
        at += 8 + len + (len & 1);
    }
    let _ = writeln!(out, "chunks: {}, RIFF says {}", chunks.join(" "), long(4));
    let decoded = decode(wav, Some("wav")).expect("the decoder reads what the encoder wrote");
    let step = match format {
        WavFormat::Int16 => 1.0 / 32_767.0,
        WavFormat::Int24 => 1.0 / 8_388_607.0,
        WavFormat::Float32 => f32::EPSILON,
    };
    let error = |sent: &[f32], got: &[f32]| {
        sent.iter()
            .zip(got)
            .take(48_000)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / step
    };
    let _ = writeln!(
        out,
        "decoded: {} ch, {} Hz, {} frames; sine error L {:.2} R {:.2} steps",
        decoded.num_channels(),
        decoded.sample_rate(),
        decoded.num_samples(),
        error(&left, decoded.channel(0)),
        error(&right, decoded.channel(1)),
    );
    for frame in 48_000..48_004 {
        let _ = writeln!(
            out,
            "  in ({:e}, {:e}) out ({:e}, {:e})",
            left[frame],
            right[frame],
            decoded.channel(0)[frame],
            decoded.channel(1)[frame]
        );
    }
    expect.assert_eq(&out);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn sixteen_bit() {
    check(WavFormat::Int16, expect![[r#"
        192060 bytes, fmt chunk 16 bytes: tag 1, 2 ch, 48000 Hz, 192000 bytes/s, align 4, 16 bits
        chunks: fmt (16) data(192016), RIFF says 192052
        decoded: 2 ch, 48000 Hz, 48004 frames; sine error L 1.50 R 1.09 steps
          in (1e0, -1e0) out (9.999695e-1, -9.999695e-1)
          in (1.5e0, -1.5e0) out (9.999695e-1, -9.999695e-1)
          in (1.1920929e-7, 0e0) out (0e0, 0e0)
          in (0e0, 0e0) out (0e0, 0e0)
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn twenty_four_bit() {
    check(WavFormat::Int24, expect![[r#"
        288068 bytes, fmt chunk 16 bytes: tag 1, 2 ch, 48000 Hz, 288000 bytes/s, align 6, 24 bits
        chunks: fmt (16) data(288024), RIFF says 288060
        decoded: 2 ch, 48000 Hz, 48004 frames; sine error L 0.75 R 0.56 steps
          in (1e0, -1e0) out (9.999999e-1, -9.999999e-1)
          in (1.5e0, -1.5e0) out (9.999999e-1, -9.999999e-1)
          in (1.1920929e-7, 0e0) out (1.1920929e-7, 0e0)
          in (0e0, 0e0) out (0e0, 0e0)
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn thirty_two_bit_float() {
    check(WavFormat::Float32, expect![[r#"
        384090 bytes, fmt chunk 18 bytes: tag 3, 2 ch, 48000 Hz, 384000 bytes/s, align 8, 32 bits
        chunks: fmt (18) fact(4) data(384032), RIFF says 384082
        decoded: 2 ch, 48000 Hz, 48004 frames; sine error L 0.00 R 0.00 steps
          in (1e0, -1e0) out (1e0, -1e0)
          in (1.5e0, -1.5e0) out (1.5e0, -1.5e0)
          in (1.1920929e-7, 0e0) out (1.1920929e-7, 0e0)
          in (0e0, 0e0) out (0e0, 0e0)
    "#]]);
}
