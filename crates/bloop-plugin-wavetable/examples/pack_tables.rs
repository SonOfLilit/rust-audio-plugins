//! Packs wavetables for `tools/import.py`, which runs it: each table in the
//! smaller of its own file and [`Format::FlacDelta`], the latter kept only
//! when it decodes to the very samples the original does.
//!
//! Reads `<source>\t<destination without extension>\t<frame size>` lines on
//! stdin, writes each table under the directory named on the command line,
//! and prints `<file written>\t<format>\t<bytes>` a line, in the same order.
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "a tool run by tools/import.py, which reads its stdout; a table that will not pack stops it"
)]

use bloop_plugin_wavetable::bank::{decode, Format};
use flacenc::bitsink::MemSink;
use flacenc::component::BitRepr;
use flacenc::error::Verify;
use flacenc::source::Fill as _;
use std::io::BufRead as _;
use std::path::Path;

/// FLAC block size. Whole blocks only: `flacenc` mis-signals a trailing
/// partial block, so the audio is padded to a multiple of this and the
/// header says how many samples are real.
const BLOCK: usize = 4096;

fn main() {
    let out = std::env::args().nth(1).expect("the directory to write under");
    let out = Path::new(&out);
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("read stdin");
        let fields: Vec<&str> = line.split('\t').collect();
        let [source, destination, frame] = fields[..] else {
            panic!("want source, destination and frame size: {line:?}");
        };
        let frame: usize = frame.parse().expect("a frame size");
        let (written, format, bytes) = pack(Path::new(source), destination, frame, out);
        let format = match format {
            Format::Flac => "flac",
            Format::FlacDelta => "flac-delta",
            Format::Wav => "wav",
        };
        println!("{written}\t{format}\t{bytes}");
    }
}

fn pack(source: &Path, destination: &str, frame: usize, out: &Path) -> (String, Format, usize) {
    let original = std::fs::read(source).unwrap_or_else(|e| panic!("read {source:?}: {e}"));
    let (format, extension) = match source.extension().and_then(|e| e.to_str()) {
        Some("wav") => (Format::Wav, "wav"),
        _ => (Format::Flac, "wt2048"),
    };
    let (samples, rate) =
        decode(format, frame, &original).unwrap_or_else(|e| panic!("decode {source:?}: {e:?}"));
    let whole = &samples[..samples.len() / frame * frame];

    let delta = sixteen_bit(whole).and_then(|ints| {
        let packed = encode_delta(&ints, frame, rate);
        let (back, back_rate) = decode(Format::FlacDelta, frame, &packed).ok()?;
        let same = back_rate.to_bits() == rate.to_bits()
            && back.len() == whole.len()
            && back.iter().zip(whole).all(|(a, b)| a.to_bits() == b.to_bits());
        same.then_some(packed)
    });

    let (bytes, format, extension) = match delta {
        Some(packed) if packed.len() < original.len() => (packed, Format::FlacDelta, "wtd"),
        _ => (original, format, extension),
    };
    let file = format!("{destination}.{extension}");
    let path = out.join(&file);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("make the bank's folder");
    }
    std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    (file, format, bytes.len())
}

/// The samples as 16-bit integers, if that is what they are.
fn sixteen_bit(samples: &[f32]) -> Option<Vec<i32>> {
    samples
        .iter()
        .map(|s| {
            let int = (f64::from(*s) * 32768.0).round();
            let exact = (-32768.0..=32767.0).contains(&int) && (int / 32768.0) as f32 == *s;
            exact.then_some(int as i32)
        })
        .collect()
}

fn encode_delta(samples: &[i32], frame: usize, rate: f64) -> Vec<u8> {
    let mut deltas: Vec<i32> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| if i < frame { *s } else { s - samples[i - frame] })
        .collect();
    let real = deltas.len();
    deltas.resize(real.div_ceil(BLOCK) * BLOCK, 0);
    let source = flacenc::source::MemSource::from_samples(&deltas, 1, 24, rate as usize);
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .expect("the default encoder config is valid");
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, BLOCK)
        .unwrap_or_else(|e| panic!("encode: {e:?}"));
    let mut context = flacenc::source::Context::new(24, 1);
    context
        .fill_interleaved(&deltas[..real])
        .expect("checksum the real samples");
    let info = stream.stream_info_mut();
    info.set_total_samples(real);
    info.set_md5_digest(&context.md5_digest());
    let mut sink = MemSink::<u8>::new();
    stream.write(&mut sink).expect("write the FLAC");
    sink.into_inner()
}
