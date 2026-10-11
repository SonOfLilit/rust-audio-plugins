//! EXPORT's and SHARE's MP3s, read back through the decoder an import
//! goes through (symphonia's, not the encoder's own).

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use bloop_sampler_engine::{decode, encode_mp3, Mp3Encoding, MP3_KBPS};
use expect_test::{expect, Expect};
use std::fmt::Write as _;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

const RATE: u32 = 48_000;

fn sine(hz: f32, level: f32, frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "a test signal")]
            let t = i as f32 / RATE as f32;
            level * libm::sinf(2.0 * core::f32::consts::PI * hz * t)
        })
        .collect()
}

/// The first frame's header, as the bytes say it.
fn first_header(mp3: &[u8]) -> String {
    let Some(at) = mp3
        .windows(2)
        .position(|w| w[0] == 0xFF && w[1] & 0xE0 == 0xE0)
    else {
        return "no frame sync".to_string();
    };
    let h = &mp3[at..at + 4];
    let version = match (h[1] >> 3) & 3 {
        3 => "MPEG-1",
        2 => "MPEG-2",
        0 => "MPEG-2.5",
        _ => "reserved",
    };
    let layer = match (h[1] >> 1) & 3 {
        1 => "Layer III",
        2 => "Layer II",
        3 => "Layer I",
        _ => "reserved",
    };
    const V1_L3: [u32; 16] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    let kbps = V1_L3[usize::from(h[2] >> 4)];
    let rate = match (h[2] >> 2) & 3 {
        0 => 44_100,
        1 => 48_000,
        2 => 32_000,
        _ => 0,
    };
    let mode = match h[3] >> 6 {
        0 => "stereo",
        1 => "joint stereo",
        2 => "dual channel",
        _ => "mono",
    };
    let tag = ["Xing", "Info"]
        .into_iter()
        .find(|tag| mp3.windows(4).take(200).any(|w| w == tag.as_bytes()))
        .unwrap_or("no Xing/Info");
    format!("at byte {at}: {version} {layer}, {kbps} kbps, {rate} Hz, {mode}; {tag}")
}

/// Where the decoded side best lines up with what went in, within the
/// first `most` frames, and the error left there, in dB under the signal.
fn aligned(sent: &[f32], got: &[f32], most: usize) -> (usize, f32) {
    // Over the same frames whatever the lag: what went in, short of the
    // longest lag.
    let most = most.min(got.len().saturating_sub(sent.len()) + 1);
    let span = sent.len().min(got.len().saturating_sub(most - 1));
    let error_at = |lag: usize| -> f64 {
        sent.iter()
            .zip(&got[lag..])
            .take(span)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum()
    };
    let lag = (0..most.min(got.len()))
        .min_by(|a, b| error_at(*a).total_cmp(&error_at(*b)))
        .unwrap_or(0);
    let power: f64 = sent.iter().take(span).map(|a| f64::from(*a).powi(2)).sum();
    if power == 0.0 {
        return (lag, f32::NAN);
    }
    #[expect(clippy::cast_possible_truncation, reason = "a dB figure")]
    let db = (10.0 * (error_at(lag) / power).log10()) as f32;
    (lag, db)
}

/// Zero crossings upward over the middle second: the frequency heard.
fn frequency(side: &[f32]) -> usize {
    let middle = &side[side.len() / 4..side.len() / 4 + RATE as usize];
    middle
        .windows(2)
        .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
        .count()
}

fn peak(side: &[f32]) -> f32 {
    side.iter().fold(0.0f32, |p, s| p.max(s.abs()))
}

/// The file as the bytes say it, then as symphonia decodes it: its
/// shape, each side's peak, the frequency it plays, and how far it is
/// from what went in once lined up (the encoder's delay).
fn check(left: &[f32], right: &[f32], expect: Expect) {
    let mp3 = encode_mp3(left, right, RATE).expect("48 kHz is an MP3 rate");
    #[expect(clippy::cast_precision_loss, reason = "a rate")]
    let seconds = left.len() as f32 / RATE as f32;
    #[expect(clippy::cast_precision_loss, reason = "a rate")]
    let kbps = mp3.len() as f32 * 8.0 / seconds / 1000.0;
    let mut out = format!(
        "{} bytes, {kbps:.0} kbps over the audio\n{}\n",
        mp3.len(),
        first_header(&mp3)
    );
    let decoded = decode(mp3, Some("mp3")).expect("the decoder reads what the encoder wrote");
    let _ = writeln!(
        out,
        "decoded: {} ch, {} Hz, {} frames (sent {})",
        decoded.num_channels(),
        decoded.sample_rate(),
        decoded.num_samples(),
        left.len()
    );
    for (name, sent, got) in [
        ("L", left, decoded.channel(0)),
        ("R", right, decoded.channel(1)),
    ] {
        let (lag, db) = aligned(sent, got, 3_000);
        let _ = writeln!(
            out,
            "{name}: peak {:.3} (sent {:.3}), {} Hz, delay {lag} frames, error {db:.1} dB",
            peak(got),
            peak(sent),
            if got.len() >= RATE as usize * 2 {
                frequency(got).to_string()
            } else {
                "-".to_string()
            },
        );
    }
    expect.assert_eq(&out);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_sine_each_side() {
    check(
        &sine(1_000.0, 0.5, 96_000),
        &sine(3_000.0, 0.1, 96_000),
        expect![[r#"
            66048 bytes, 264 kbps over the audio
            at byte 0: MPEG-1 Layer III, 256 kbps, 48000 Hz, stereo; Info
            decoded: 2 ch, 48000 Hz, 97920 frames (sent 96000)
            L: peak 0.502 (sent 0.500), 1000 Hz, delay 1057 frames, error -71.6 dB
            R: peak 0.101 (sent 0.100), 3000 Hz, delay 1057 frames, error -63.2 dB
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn silence() {
    check(
        &[0.0; 96_000],
        &[0.0; 96_000],
        expect![[r#"
        66048 bytes, 264 kbps over the audio
        at byte 0: MPEG-1 Layer III, 256 kbps, 48000 Hz, stereo; Info
        decoded: 2 ch, 48000 Hz, 97920 frames (sent 96000)
        L: peak 0.000 (sent 0.000), 0 Hz, delay 0 frames, error NaN dB
        R: peak 0.000 (sent 0.000), 0 Hz, delay 0 frames, error NaN dB
    "#]],
    );
}

/// Past full scale the file holds what it is given, as a float WAV
/// does; a player clips it.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn past_full_scale() {
    check(
        &sine(440.0, 1.5, 96_000),
        &sine(440.0, -1.5, 96_000),
        expect![[r#"
            66048 bytes, 264 kbps over the audio
            at byte 0: MPEG-1 Layer III, 256 kbps, 48000 Hz, stereo; Info
            decoded: 2 ch, 48000 Hz, 97920 frames (sent 96000)
            L: peak 1.501 (sent 1.500), 440 Hz, delay 1057 frames, error -75.6 dB
            R: peak 1.501 (sent 1.500), 440 Hz, delay 1057 frames, error -75.6 dB
        "#]],
    );
}

/// Shorter than one frame of 1152.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_hundred_frames() {
    check(
        &sine(1_000.0, 0.5, 100),
        &sine(1_000.0, 0.5, 100),
        expect![[r#"
            2304 bytes, 8847 kbps over the audio
            at byte 0: MPEG-1 Layer III, 256 kbps, 48000 Hz, stereo; Info
            decoded: 2 ch, 48000 Hz, 2304 frames (sent 100)
            L: peak 0.501 (sent 0.500), - Hz, delay 1057 frames, error -42.7 dB
            R: peak 0.501 (sent 0.500), - Hz, delay 1057 frames, error -42.7 dB
        "#]],
    );
}

/// Nothing at all is still a file a player opens.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn nothing() {
    let mp3 = encode_mp3(&[], &[], RATE).expect("48 kHz is an MP3 rate");
    expect!["1536 bytes; at byte 0: MPEG-1 Layer III, 256 kbps, 48000 Hz, stereo; Info"]
        .assert_eq(&format!("{} bytes; {}", mp3.len(), first_header(&mp3)));
}

/// A slice at a time writes the same bytes as all at once, and counts
/// its way there; the gain is applied on the way in.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_slice_at_a_time() {
    let left = sine(1_000.0, 0.5, 30_000);
    let right = sine(2_000.0, 0.25, 30_000);
    let whole = encode_mp3(&left, &right, RATE).expect("48 kHz is an MP3 rate");
    let mut encoding =
        Mp3Encoding::new(left.clone(), right.clone(), RATE, 1.0).expect("48 kHz is an MP3 rate");
    let mut progress = Vec::new();
    let sliced = loop {
        let step = encoding.step(7_000).expect("encodes");
        progress.push(format!("{:.2}", encoding.progress()));
        if let Some(bytes) = step {
            break bytes;
        }
    };
    let halved = Mp3Encoding::new(left.clone(), right.clone(), RATE, 0.5)
        .and_then(|mut encoding| {
            std::iter::repeat_with(|| encoding.step(10_000))
                .find_map(Result::transpose)
                .unwrap_or_else(|| Err("never finished".to_string()))
        })
        .expect("encodes");
    let halved = decode(halved, Some("mp3")).expect("decodes");
    expect!["256 kbps; sliced == the whole; progress 0.23 0.47 0.70 0.93 1.00; at gain 0.5 the left peaks at 0.251"].assert_eq(&format!(
        "{MP3_KBPS} kbps; sliced {} the whole; progress {}; at gain 0.5 the left peaks at {:.3}",
        if sliced == whole { "==" } else { "!=" },
        progress.join(" "),
        peak(halved.channel(0)),
    ));
}

/// MP3 has 32, 44.1 and 48 kHz, and their halves and quarters.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn rates() {
    let said: Vec<String> = [48_000, 44_100, 32_000, 24_000, 96_000, 0]
        .into_iter()
        .map(|rate| match encode_mp3(&[0.0; 10], &[0.0; 10], rate) {
            Ok(bytes) => format!("{rate}: {} bytes", bytes.len()),
            Err(said) => format!("{rate}: {said}"),
        })
        .collect();
    expect![[r#"
        48000: 2304 bytes
        44100: 2505 bytes
        32000: 3456 bytes
        24000: 1920 bytes
        96000: MP3 has no sample rate of 96000 Hz
        0: MP3 has no sample rate of 0 Hz"#]]
    .assert_eq(&said.join("\n"));
}

/// Clicks at the start, the middle and the last frame: where each comes
/// out says the encoder's delay, and whether the end survives it.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn where_a_click_lands() {
    let mut left = vec![0.0f32; 96_000];
    for at in [0, 48_000, 95_999] {
        left[at] = 0.9;
    }
    let mp3 = encode_mp3(&left, &left, RATE).expect("48 kHz is an MP3 rate");
    let decoded = decode(mp3, Some("mp3")).expect("decodes");
    let got = decoded.channel(0);
    let said: Vec<String> = [0usize, 48_000, 95_999]
        .into_iter()
        .map(|at| {
            let window = &got[at.min(got.len())..(at + 3_000).min(got.len())];
            match window
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            {
                Some((lag, s)) if s.abs() > 0.05 => {
                    format!("{at}: out at +{lag}, {s:.2}")
                }
                _ => format!("{at}: not heard"),
            }
        })
        .collect();
    expect![[r#"
        97920 frames out
        0: out at +1057, 0.87
        48000: out at +1057, 0.82
        95999: out at +1057, 0.92"#]]
    .assert_eq(&format!("{} frames out\n{}", got.len(), said.join("\n")));
}

/// The version the License page cites is the one Cargo.toml pins.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_pinned_encoder_is_the_one_cited() {
    let manifest = include_str!("../../Cargo.toml");
    let pin = format!(
        "rusty_mp3 = \"={}\"",
        bloop_sampler_engine::RUSTY_MP3_VERSION
    );
    assert!(
        manifest.contains(&pin),
        "Cargo.toml must pin {pin}, as RUSTY_MP3_VERSION says"
    );
}
