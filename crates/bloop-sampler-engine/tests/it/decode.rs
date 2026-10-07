//! Import: turning encoded audio into the buffer pads share.
//!
//! Fixtures are encoded here rather than committed, so what the decoder is fed
//! is visible in the test. WAV is written by hand (the format is small enough
//! to be obvious); FLAC goes through `flacenc`, which is also what the sample
//! pack will be transcoded with.

use bloop_sampler_engine::{decode, SampleBuffer, SampleDecoder};
use expect_test::{expect, Expect};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A deterministic test signal: `channels` channels of `frames` frames, where
/// channel `c` frame `i` is a distinct value that stays inside [-0.5, 0.5).
///
/// Every value is an exact multiple of 1/32768, so 16-bit PCM stores it without
/// rounding and the snapshots can be read as exact.
fn signal(channels: usize, frames: usize) -> Vec<Vec<f32>> {
    (0..channels)
        .map(|c| {
            (0..frames)
                .map(|i| {
                    let step = ((i + 1) * (c + 1) * 1024) % 32768;
                    (step as i32 - 16384) as f32 / 32768.0
                })
                .collect()
        })
        .collect()
}

/// Writes a PCM WAV file. `bits` may be 16 or 24.
fn wav(channels: &[Vec<f32>], rate: u32, bits: u16) -> Vec<u8> {
    let num_channels = channels.len() as u16;
    let frames = channels.first().map_or(0, Vec::len);
    let bytes_per_sample = u32::from(bits) / 8;
    let block_align = u32::from(num_channels) * bytes_per_sample;
    let data_len = frames as u32 * block_align;

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // format: PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * block_align).to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());

    for frame in 0..frames {
        for channel in channels {
            let value = f64::from(channel[frame]);
            match bits {
                16 => {
                    let scaled = (value * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
                    out.extend_from_slice(&scaled.to_le_bytes());
                }
                24 => {
                    let scaled = (value * 8_388_608.0)
                        .round()
                        .clamp(-8_388_608.0, 8_388_607.0) as i32;
                    out.extend_from_slice(&scaled.to_le_bytes()[..3]);
                }
                other => panic!("unsupported test bit depth {other}"),
            }
        }
    }
    out
}

/// The FLAC fixture's block size. Fixture lengths must be a multiple of it:
/// `flacenc` mis-signals a trailing partial block, which is a fixture-side
/// limitation, not something the decoder should paper over. Truncated input is
/// covered separately by `a_truncated_file_imports_what_survived`.
const FLAC_BLOCK: usize = 256;

/// Encodes the same signal as 16-bit FLAC.
fn flac(channels: &[Vec<f32>], rate: u32) -> Vec<u8> {
    use flacenc::bitsink::MemSink;
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let num_channels = channels.len();
    let frames = channels.first().map_or(0, Vec::len);
    assert_eq!(
        frames % FLAC_BLOCK,
        0,
        "FLAC fixtures must be a whole number of {FLAC_BLOCK}-frame blocks"
    );
    let mut interleaved = Vec::with_capacity(frames * num_channels);
    for frame in 0..frames {
        for channel in channels {
            interleaved.push((f64::from(channel[frame]) * 32768.0).round() as i32);
        }
    }

    let source =
        flacenc::source::MemSource::from_samples(&interleaved, num_channels, 16, rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(
        &flacenc::config::Encoder::default()
            .into_verified()
            .expect("default encoder config is valid"),
        source,
        FLAC_BLOCK,
    )
    .expect("encoding the fixture must succeed");

    let mut sink = MemSink::<u8>::new();
    stream.write(&mut sink).expect("writing the fixture");
    sink.into_inner()
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Describes a decoded buffer: shape, rate, and the opening frames of every
/// channel.
fn describe(buffer: &SampleBuffer) -> String {
    let mut out = format!(
        "{} ch, {} frames @{} Hz",
        buffer.num_channels(),
        buffer.num_samples(),
        buffer.sample_rate()
    );
    for channel in 0..buffer.num_channels() {
        let values: Vec<String> = buffer
            .channel(channel)
            .iter()
            .take(6)
            .map(|v| format!("{v:.6}"))
            .collect();
        out.push_str(&format!("\nch{channel}: {}", values.join(" ")));
    }
    out
}

fn check(bytes: Vec<u8>, extension: Option<&str>, expect: Expect) {
    match decode(bytes, extension) {
        Ok(buffer) => expect.assert_eq(&describe(&buffer)),
        // Both the variant (what UI code matches on) and the message (what it
        // shows) are pinned.
        Err(error) => expect.assert_eq(&format!("rejected: {error:?} — {error}")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A mono 16-bit WAV decodes to exactly the values that were written, at the
/// file's own rate — import never resamples, because the voice does.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_mono_wav_decodes_to_the_values_it_stored() {
    check(
        wav(&signal(1, 8), 44_100, 16),
        Some("wav"),
        expect![[r#"
            1 ch, 8 frames @44100 Hz
            ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500"#]],
    );
}

/// Stereo keeps its channels apart and in order.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_stereo_wav_keeps_its_channels_in_order() {
    check(
        wav(&signal(2, 8), 48_000, 16),
        Some("wav"),
        expect![[r#"
            2 ch, 8 frames @48000 Hz
            ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
            ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000"#]],
    );
}

/// 24-bit files decode at their full precision rather than being narrowed to
/// 16 — half the sample pack is 24-bit. The same value at both depths shows
/// the detail 16 bits rounds away and 24 bits keeps.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_24_bit_wav_keeps_precision_that_16_bits_loses() {
    // Half-way between two 16-bit codes, exactly representable in 24.
    let value = 1000.5 / 32768.0;
    let fine = vec![vec![value; 4]];

    let narrow = decode(wav(&fine, 44_100, 16), Some("wav")).expect("16-bit decodes");
    let wide = decode(wav(&fine, 44_100, 24), Some("wav")).expect("24-bit decodes");

    expect![[r#"
        wrote:  0.030532837
        16-bit: 0.030548096
        24-bit: 0.030532837"#]]
    .assert_eq(&format!(
        "wrote:  {value:.9}\n16-bit: {:.9}\n24-bit: {:.9}",
        narrow.sample(0, 0),
        wide.sample(0, 0),
    ));
}

/// FLAC is lossless, so the same signal must decode to the same samples as the
/// WAV of it. This is the format the bundled pack ships in.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn flac_decodes_identically_to_the_wav_of_the_same_signal() {
    let source = signal(2, 512);
    let from_wav = decode(wav(&source, 44_100, 16), Some("wav")).expect("wav decodes");
    let from_flac = decode(flac(&source, 44_100), Some("flac")).expect("flac decodes");

    let identical = (0..from_wav.num_channels()).all(|c| {
        from_wav.num_channels() == from_flac.num_channels()
            && from_wav.channel(c) == from_flac.channel(c)
    });
    expect![[r#"
        wav:  2 ch, 512 frames @44100 Hz
        flac: 2 ch, 512 frames @44100 Hz
        sample for sample identical: true"#]]
    .assert_eq(&format!(
        "wav:  {}\nflac: {}\nsample for sample identical: {identical}",
        describe(&from_wav).lines().next().unwrap_or_default(),
        describe(&from_flac).lines().next().unwrap_or_default(),
    ));
}

/// The extension is only a hint — the bytes decide. A FLAC misnamed `.wav`
/// still imports, which matters because file pickers lie.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_extension_is_only_a_hint() {
    check(
        flac(&signal(1, 256), 44_100),
        Some("wav"),
        expect![[r#"
        1 ch, 256 frames @44100 Hz
        ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500"#]],
    );
}

/// Bloop is stereo end to end, so anything wider is folded: the front half of
/// the channels to the left, the back half to the right.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn more_than_two_channels_fold_down_to_stereo() {
    check(
        wav(&signal(4, 8), 48_000, 16),
        Some("wav"),
        expect![[r#"
            2 ch, 8 frames @48000 Hz
            ch0: -0.453125 -0.406250 -0.359375 -0.312500 -0.265625 -0.218750
            ch1: -0.390625 -0.281250 -0.171875 -0.062500 0.046875 0.156250"#]],
    );
}

/// A file that is not audio is refused with something the UI can show.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_file_that_is_not_audio_is_refused() {
    check(
        b"this is a text file, not a sample".to_vec(),
        Some("wav"),
        expect!["rejected: UnrecognisedFormat — not an audio file Bloop can read"],
    );
}

/// So is a valid header with no audio behind it.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_wav_with_no_samples_is_refused() {
    check(
        wav(&[vec![]], 44_100, 16),
        Some("wav"),
        expect!["rejected: Empty — this file contains no audio"],
    );
}

/// A file truncated part-way through still imports what survived, rather than
/// throwing away a mostly-good recording.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_truncated_file_imports_what_survived() {
    let mut bytes = wav(&signal(1, 4096), 44_100, 16);
    bytes.truncate(bytes.len() / 2);
    check(
        bytes,
        Some("wav"),
        expect![[r#"
        1 ch, 2037 frames @44100 Hz
        ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500"#]],
    );
}

/// An import with no extension to go on probes the bytes instead.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn an_import_with_no_extension_probes_the_bytes() {
    check(
        wav(&signal(1, 8), 22_050, 16),
        None,
        expect![[r#"
        1 ch, 8 frames @22050 Hz
        ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500"#]],
    );
}

/// Decoding a packet at a time must produce exactly what decoding all at once
/// does — the resumable path exists so a long import does not stall the
/// interface, not to be a second implementation.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_resumed_decode_matches_an_all_at_once_one() {
    let source = signal(2, 4096);
    let bytes = flac(&source, 44_100);

    let at_once = decode(bytes.clone(), Some("flac")).expect("decodes");

    let mut decoder = SampleDecoder::new(bytes, Some("flac")).expect("opens");
    let mut steps = 0;
    let resumed = loop {
        steps += 1;
        if let Some(buffer) = decoder.step(1).expect("decodes") {
            break buffer;
        }
        assert!(steps < 10_000, "the decode never finished");
    };

    assert_eq!(at_once, resumed, "the two paths must agree");
    expect!["17 steps, 4096 frames, progress ends at 1.0"].assert_eq(&format!(
        "{steps} steps, {} frames, progress ends at {:.1}",
        resumed.num_samples(),
        decoder.progress(),
    ));
}

/// A file that is not audio fails when it is opened, before any stepping —
/// so a caller pumping across frames learns immediately.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_resumable_decode_rejects_rubbish_up_front() {
    let error = SampleDecoder::new(b"not audio at all".to_vec(), Some("wav"))
        .err()
        .expect("must fail");
    expect!["UnrecognisedFormat"].assert_eq(&format!("{error:?}"));
}

// ---------------------------------------------------------------------------
// The fmt chunk's length
// ---------------------------------------------------------------------------

/// How the samples of a [`wav_with_fmt`] file are stored.
#[derive(Clone, Copy)]
enum Encoding {
    /// `WAVE_FORMAT_PCM`, 16-bit.
    Pcm16,
    /// `WAVE_FORMAT_IEEE_FLOAT`, 32-bit.
    Float32,
    /// `WAVE_FORMAT_EXTENSIBLE` around 16-bit PCM.
    Extensible16,
}

/// A stereo 8-frame WAV whose fmt chunk is the standard 16 bytes followed
/// by `extension`, its length field saying `16 + extension.len()`, and a pad
/// byte after it when that is odd, as RIFF asks.
fn wav_with_fmt(encoding: Encoding, extension: &[u8]) -> Vec<u8> {
    wav_with_fmt_len(encoding, extension, 16 + extension.len() as u32)
}

/// [`wav_with_fmt`], with the fmt chunk's length field set to `declared`
/// whatever the chunk really holds.
fn wav_with_fmt_len(encoding: Encoding, extension: &[u8], declared: u32) -> Vec<u8> {
    let channels = signal(2, 8);
    let (tag, bits): (u16, u16) = match encoding {
        Encoding::Pcm16 => (1, 16),
        Encoding::Float32 => (3, 32),
        Encoding::Extensible16 => (0xFFFE, 16),
    };
    let block_align = 2 * bits / 8;
    let rate = 44_100u32;

    let mut fmt = Vec::new();
    fmt.extend_from_slice(&tag.to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes());
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    fmt.extend_from_slice(extension);
    if fmt.len() % 2 == 1 {
        fmt.push(0);
    }

    let mut data = Vec::new();
    for frame in 0..8 {
        for channel in &channels {
            let value = channel[frame];
            match encoding {
                Encoding::Pcm16 | Encoding::Extensible16 => {
                    data.extend_from_slice(&((value * 32768.0) as i16).to_le_bytes());
                }
                Encoding::Float32 => data.extend_from_slice(&value.to_le_bytes()),
            }
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((4 + 8 + fmt.len() + 8 + data.len()) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&declared.to_le_bytes());
    out.extend_from_slice(&fmt);
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// `WAVE_FORMAT_EXTENSIBLE`'s 24 bytes after the common 16: cbSize 22,
/// valid bits, a stereo channel mask and the PCM sub-format GUID.
fn extensible_pcm_extension() -> Vec<u8> {
    let mut extension = Vec::new();
    extension.extend_from_slice(&22u16.to_le_bytes());
    extension.extend_from_slice(&16u16.to_le_bytes());
    extension.extend_from_slice(&3u32.to_le_bytes());
    extension.extend_from_slice(&[
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
        0x71,
    ]);
    extension
}

/// The bundled pack's darbuka hits carry a 20-byte PCM fmt chunk: the 16
/// that matter, then four zero bytes. Players read them; so must import.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_pcm_fmt_chunk_with_trailing_bytes_decodes() {
    check(
        wav_with_fmt(Encoding::Pcm16, &[0, 0, 0, 0]),
        Some("wav"),
        expect![[r#"
            2 ch, 8 frames @44100 Hz
            ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
            ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000"#]],
    );
}

/// The same for IEEE float, with a cbSize that is not 0 either.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_float_fmt_chunk_with_trailing_bytes_decodes() {
    check(
        wav_with_fmt(Encoding::Float32, &[2, 0, 0xAB, 0xCD]),
        Some("wav"),
        expect![[r#"
            2 ch, 8 frames @44100 Hz
            ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
            ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000"#]],
    );
}

/// An odd length is followed by RIFF's pad byte, which goes too.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_pcm_fmt_chunk_of_odd_length_decodes() {
    check(
        wav_with_fmt(Encoding::Pcm16, &[0, 0, 0]),
        Some("wav"),
        expect![[r#"
            2 ch, 8 frames @44100 Hz
            ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
            ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000"#]],
    );
}

/// The lengths symphonia reads itself keep working: 18 bytes with cbSize 0,
/// and the 40-byte `WAVE_FORMAT_EXTENSIBLE`.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn standard_fmt_chunk_lengths_still_decode() {
    let decoded = |bytes: Vec<u8>| match decode(bytes, Some("wav")) {
        Ok(buffer) => describe(&buffer),
        Err(error) => format!("rejected: {error:?}"),
    };
    expect![[r#"
        pcm 18: 2 ch, 8 frames @44100 Hz
        ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
        ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000
        float 18: 2 ch, 8 frames @44100 Hz
        ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
        ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000
        extensible 40: 2 ch, 8 frames @44100 Hz
        ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
        ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000"#]]
    .assert_eq(&format!(
        "pcm 18: {}\nfloat 18: {}\nextensible 40: {}",
        decoded(wav_with_fmt(Encoding::Pcm16, &[0, 0])),
        decoded(wav_with_fmt(Encoding::Float32, &[0, 0])),
        decoded(wav_with_fmt(
            Encoding::Extensible16,
            &extensible_pcm_extension()
        )),
    ));
}

/// A fmt chunk too short to hold a format is still refused, and says so.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_fmt_chunk_shorter_than_16_bytes_is_refused() {
    check(
        wav_with_fmt_len(Encoding::Pcm16, &[], 12),
        Some("wav"),
        expect![[
            r#"rejected: Corrupt("wav: malformed fmt chunk") — this file is damaged (wav: malformed fmt chunk)"#
        ]],
    );
}

/// A fmt chunk whose length runs past the end of the file is refused, not
/// read out of bounds.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_fmt_chunk_longer_than_the_file_is_refused() {
    check(
        wav_with_fmt_len(Encoding::Pcm16, &[0, 0, 0, 0], 4096),
        Some("wav"),
        expect![[
            r#"rejected: Corrupt("riff: chunk length exceeds parent (list) chunk length") — this file is damaged (riff: chunk length exceeds parent (list) chunk length)"#
        ]],
    );
}

/// The darbuka files' header through the resumable decoder, which IMPORT
/// and the Browser use.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_resumable_decode_reads_a_20_byte_fmt_chunk() {
    let bytes = wav_with_fmt(Encoding::Pcm16, &[0, 0, 0, 0]);
    let result = SampleDecoder::new(bytes, Some("wav")).and_then(|mut decoder| loop {
        if let Some(buffer) = decoder.step(1)? {
            break Ok(buffer);
        }
    });
    expect![[r#"
        2 ch, 8 frames @44100 Hz
        ch0: -0.468750 -0.437500 -0.406250 -0.375000 -0.343750 -0.312500
        ch1: -0.437500 -0.375000 -0.312500 -0.250000 -0.187500 -0.125000"#]]
    .assert_eq(&match result {
        Ok(buffer) => describe(&buffer),
        Err(error) => format!("rejected: {error:?}"),
    });
}
