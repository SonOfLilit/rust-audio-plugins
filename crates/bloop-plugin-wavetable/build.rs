//! Lays the bundled wavetables out for embedding: one blob of every file
//! `wavetables/catalogue.tsv` lists — on the web only those it marks `web`
//! — and an index into it, so the catalogue's order, and so every table's
//! number, is the same on every target whether or not its audio is there.
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a build script reports to no one but cargo: a missing or malformed table stops the build"
)]

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wavetables");

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let web = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32");
    let root = PathBuf::from("wavetables");
    let listing =
        std::fs::read_to_string(root.join("catalogue.tsv")).expect("read wavetables/catalogue.tsv");

    let mut blob = Vec::new();
    let mut index = String::from("pub static CATALOGUE: &[Entry] = &[\n");
    let mut names = Vec::new();
    for line in listing.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 6, "a catalogue line has six fields: {line}");
        let (bank, name, file, format, frame, carried) = (f[0], f[1], f[2], f[3], f[4], f[5]);
        let format = match format {
            "flac" => "Format::Flac",
            "flac-delta" => "Format::FlacDelta",
            "wav" => "Format::Wav",
            other => panic!("unknown wavetable format {other}"),
        };
        let on_web = carried == "web";
        let audio = if web && !on_web {
            "None".to_string()
        } else {
            let bytes = std::fs::read(root.join(file)).expect("read a bundled wavetable");
            let at = blob.len();
            blob.extend_from_slice(&bytes);
            format!("Some(({at}, {}))", bytes.len())
        };
        let _ = writeln!(
            index,
            "    Entry {{ bank: {bank:?}, name: {name:?}, format: {format}, frame: {frame}, on_web: {on_web}, audio: {audio} }},"
        );
        names.push(format!("{name:?}"));
    }
    index.push_str("];\n");
    let _ = writeln!(
        index,
        "/// Every wavetable's name, in catalogue order.\npub static NAMES: [&str; {}] = [{}];",
        names.len(),
        names.join(", ")
    );

    std::fs::File::create(out.join("wavetables.blob"))
        .and_then(|mut f| f.write_all(&blob))
        .expect("write the wavetable blob");
    std::fs::write(out.join("catalogue.rs"), index).expect("write the wavetable index");
}
