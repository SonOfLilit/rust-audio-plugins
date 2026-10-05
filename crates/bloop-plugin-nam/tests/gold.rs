//! Gold tests: the port against NeuralAmpModelerCore, the C++ it was
//! ported from.
//!
//! A case is a model, a `tanh`, a signal and a block size. It runs through
//! the port, and what comes out — its level, its peak, a hash of every
//! output sample — is the snapshot.
//!
//! With `--features gold` the same cases also run through the original —
//! `vendor/nam`'s sources compiled unmodified with Eigen (`tests/cpp_harness`)
//! — and `gold_errors` snapshots how far apart the two are. They cannot
//! agree to the bit: the port sums each matrix product in a different
//! order from Eigen (`src/wavenet.rs`), and float addition is not
//! associative. What is held is that the difference is float rounding:
//! no sample more than 1e-4 (−80 dB) from the original's, and most within
//! a few parts in a million of the output's peak.

#![expect(
    clippy::unwrap_used,
    reason = "a test fails loudly, and that is the point"
)]
#![cfg_attr(
    feature = "gold",
    expect(
        clippy::expect_used,
        reason = "the harness fails loudly, and that is the point"
    )
)]
#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use std::fmt::Write as _;

use bloop_plugin_nam::preset::Shape;
use bloop_plugin_nam::{Model, Tanh};
use expect_test::{expect, Expect};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

const RATE: f32 = 48_000.0;

const OBSIDIAN: &str = include_str!("../models/obsidian.nam");
const EXAMPLE: &str = include_str!("../models/example.nam");
const TINY_LSTM: &str = include_str!("models/lstm.nam");

/// What a guitar DI might be: a plucked note every quarter second, each a
/// stack of decaying harmonics, at a peak of `level`, with a little noise.
fn plucks(level: f32, seconds: f32) -> Vec<f32> {
    let frames = (seconds * RATE) as usize;
    let notes = [82.41f32, 110.0, 146.83, 196.0, 246.94, 329.63];
    let mut noise: u64 = 0x2545_f491_4f6c_dd1d;
    (0..frames)
        .map(|i| {
            let note = i / 12_000;
            let t = (i % 12_000) as f32 / RATE;
            let f = notes[note % notes.len()];
            let mut v = 0.0;
            for h in 1..=8 {
                let amp = 1.0 / h as f32 * libm::expf(-t * (3.0 + h as f32));
                v += amp * libm::sinf(2.0 * core::f32::consts::PI * f * h as f32 * t);
            }
            noise ^= noise << 13;
            noise ^= noise >> 7;
            noise ^= noise << 17;
            let n = (noise >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
            level * (0.45 * v + 0.002 * n)
        })
        .collect()
}

struct Case {
    name: String,
    nam: String,
    tanh: Tanh,
    input: Vec<f32>,
    block: usize,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut add = |name: &str, nam: String, tanh: Tanh, input: Vec<f32>, block: usize| {
        cases.push(Case {
            name: name.to_string(),
            nam,
            tanh,
            input,
            block,
        });
    };
    for tanh in [Tanh::Exact, Tanh::Fast] {
        add(
            "obsidian, DI level",
            OBSIDIAN.into(),
            tanh,
            plucks(0.1, 1.0),
            128,
        );
        add(
            "obsidian, hot",
            OBSIDIAN.into(),
            tanh,
            plucks(1.0, 1.0),
            128,
        );
        add(
            "obsidian, silence",
            OBSIDIAN.into(),
            tanh,
            vec![0.0; 4800],
            128,
        );
        add(
            "example, DI level",
            EXAMPLE.into(),
            tanh,
            plucks(0.1, 1.0),
            128,
        );
        for shape in Shape::WAVENETS
            .into_iter()
            .chain([Shape::GatedNano, Shape::Lstm(16)])
        {
            add(&shape.name(), shape.nam(7), tanh, plucks(0.3, 0.5), 128);
        }
        add("tiny lstm", TINY_LSTM.into(), tanh, plucks(0.3, 0.5), 128);
    }
    for block in [1, 37, 100] {
        add(
            &format!("obsidian, blocks of {block}"),
            OBSIDIAN.into(),
            Tanh::Fast,
            plucks(0.1, 0.25),
            block,
        );
    }
    cases
}

fn run(case: &Case) -> Vec<f32> {
    let mut model = Model::parse(&case.nam).unwrap();
    model.reset(case.tanh);
    let mut out = vec![0.0; case.input.len()];
    for (i, o) in case
        .input
        .chunks(case.block)
        .zip(out.chunks_mut(case.block))
    {
        model.process(i, o, case.tanh);
    }
    out
}

fn db(x: f32) -> f32 {
    20.0 * libm::log10f(x.max(1e-12))
}

fn hash(samples: &[f32]) -> u64 {
    samples.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, s| {
        (h ^ u64::from(s.to_bits())).wrapping_mul(0x0100_0000_01b3)
    })
}

fn check(expect: &Expect) {
    let mut out = String::new();
    for case in cases() {
        let y = run(&case);
        let rms = libm::sqrtf(y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32);
        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let _ = writeln!(
            out,
            "{:<28} {:?}  rms {:6.1} dB  peak {:6.1} dB  {:016x}",
            case.name,
            case.tanh,
            db(rms),
            db(peak),
            hash(&y)
        );
    }
    expect.assert_eq(&out);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn every_case_through_the_port() {
    check(&expect![[r#"
        obsidian, DI level           Exact  rms  -26.1 dB  peak  -18.0 dB  27f0a67773907bd2
        obsidian, hot                Exact  rms  -22.1 dB  peak  -14.3 dB  c5740a2a68e77875
        obsidian, silence            Exact  rms  -85.4 dB  peak  -85.4 dB  21a04a0b722ea525
        example, DI level            Exact  rms  -16.9 dB  peak   -7.1 dB  3966584c78e1c2c2
        standard                     Exact  rms  -48.2 dB  peak  -46.4 dB  cc1117cb21756765
        lite                         Exact  rms  -58.1 dB  peak  -53.4 dB  931a7a301d205fa7
        feather                      Exact  rms  -71.9 dB  peak  -60.6 dB  64ec4bd0fd512544
        nano                         Exact  rms  -46.7 dB  peak  -46.6 dB  bb773e555bb2a30f
        gated-nano                   Exact  rms  -49.8 dB  peak  -47.8 dB  61938c87f6f55910
        lstm-16                      Exact  rms  -12.7 dB  peak  -12.6 dB  8845ec2139c756ab
        tiny lstm                    Exact  rms  -45.0 dB  peak  -35.6 dB  651046e958ad22c5
        obsidian, DI level           Fast  rms  -26.2 dB  peak  -18.1 dB  78b6f7623375129d
        obsidian, hot                Fast  rms  -22.1 dB  peak  -14.3 dB  4be1fc60aab841ff
        obsidian, silence            Fast  rms  -80.6 dB  peak  -80.6 dB  9b5243b7d199e7e5
        example, DI level            Fast  rms  -16.9 dB  peak   -7.1 dB  80eefa0fc6328405
        standard                     Fast  rms  -48.2 dB  peak  -46.4 dB  76cd6b47c4a46c0a
        lite                         Fast  rms  -58.2 dB  peak  -53.4 dB  4cc481f8406b00a2
        feather                      Fast  rms  -71.8 dB  peak  -60.5 dB  d4c48698649d1253
        nano                         Fast  rms  -46.7 dB  peak  -46.6 dB  73cf25fbd921f968
        gated-nano                   Fast  rms  -49.8 dB  peak  -47.8 dB  d4a1d0f11e264afb
        lstm-16                      Fast  rms  -12.7 dB  peak  -12.6 dB  b9eb585fe83172ec
        tiny lstm                    Fast  rms  -45.1 dB  peak  -35.8 dB  40463df69a6cc105
        obsidian, blocks of 1        Fast  rms  -25.9 dB  peak  -18.2 dB  d02738a9c0d0a69e
        obsidian, blocks of 37       Fast  rms  -25.9 dB  peak  -18.2 dB  d02738a9c0d0a69e
        obsidian, blocks of 100      Fast  rms  -25.9 dB  peak  -18.2 dB  d02738a9c0d0a69e
    "#]]);
}

#[cfg(feature = "gold")]
mod original {
    use super::Case;
    use bloop_plugin_nam::Tanh;
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::OnceLock;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn vendor() -> PathBuf {
        let path = root().join("../../../vendor/nam");
        assert!(
            path.join("NAM").exists() && path.join("Dependencies/eigen/Eigen").exists(),
            "vendor/nam and its Eigen are not checked out. The gold tests compile the original from them:\n  \
             git -c submodule.vendor/nam.update=checkout submodule update --init vendor/nam\n  \
             git -C vendor/nam submodule update --init --depth 1 Dependencies/eigen"
        );
        path
    }

    fn revision(path: &Path) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(path)
            .args(["rev-parse", "--short=12", "HEAD"])
            .output()
            .expect("run git to read the vendor revision");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Every NAM source and the harness, each its own object, built in
    /// parallel once per pin.
    fn build() -> PathBuf {
        let nam = vendor();
        let harness = root().join("tests/cpp_harness/main.cpp");
        let out =
            PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("nam-gold-{}", revision(&nam)));
        std::fs::create_dir_all(&out).expect("create the harness build directory");
        let mut sources = vec![harness.clone()];
        for dir in [nam.join("NAM"), nam.join("NAM/wavenet")] {
            for entry in std::fs::read_dir(dir).expect("read the NAM sources") {
                let path = entry.expect("a NAM source").path();
                if path.extension().is_some_and(|e| e == "cpp") {
                    sources.push(path);
                }
            }
        }
        let modified = |p: &Path| p.metadata().and_then(|m| m.modified()).ok();
        let mut objects = Vec::new();
        let mut children = Vec::new();
        for source in &sources {
            let stem = source
                .strip_prefix(&nam)
                .unwrap_or(source)
                .to_string_lossy()
                .replace('/', "_");
            let object = out.join(format!("{stem}.o"));
            if !object.exists() || modified(source) > modified(&object) {
                let child = Command::new("g++")
                    .args(["-std=c++20", "-O2", "-DNDEBUG", "-ffp-contract=off", "-w"])
                    .arg("-I")
                    .arg(&nam)
                    .arg("-I")
                    .arg(nam.join("Dependencies/eigen"))
                    .arg("-I")
                    .arg(nam.join("Dependencies/nlohmann"))
                    .arg("-c")
                    .arg(source)
                    .arg("-o")
                    .arg(&object)
                    .spawn()
                    .expect("run g++ (the gold tests need it installed)");
                children.push((source.clone(), child));
            }
            objects.push(object);
        }
        let relink = !children.is_empty();
        for (source, mut child) in children {
            let status = child.wait().expect("wait for g++");
            assert!(status.success(), "compiling {} failed", source.display());
        }
        let binary = out.join("harness");
        if relink || !binary.exists() {
            let status = Command::new("g++")
                .args(&objects)
                .arg("-o")
                .arg(&binary)
                .status()
                .expect("run the linker");
            assert!(status.success(), "linking the gold harness failed");
        }
        binary
    }

    fn binary() -> &'static Path {
        static BINARY: OnceLock<PathBuf> = OnceLock::new();
        BINARY.get_or_init(build)
    }

    /// The case through the original.
    pub fn run(case: &Case, dir: &Path, index: usize) -> Vec<f32> {
        let path = dir.join(format!("case-{index}.nam"));
        std::fs::write(&path, &case.nam).expect("write the case's model");
        let mut script = Vec::new();
        // Before the load: a WaveNet looks its activation up as it is
        // built, so NAM's plug-in turns fast tanh on before loading any.
        let _ = writeln!(script, "fast {}", u8::from(case.tanh == Tanh::Fast));
        let _ = writeln!(script, "load {}", path.display());
        let _ = writeln!(script, "reset 48000 {}", case.block.max(128));
        for block in case.input.chunks(case.block) {
            let _ = writeln!(script, "process {}", block.len());
            for s in block {
                script.extend_from_slice(&s.to_le_bytes());
            }
        }
        let mut child = Command::new(binary())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("start the gold harness");
        let mut stdin = child.stdin.take().expect("harness stdin");
        let writer = std::thread::spawn(move || stdin.write_all(&script));
        let output = child.wait_with_output().expect("run the gold harness");
        writer
            .join()
            .expect("join the writer")
            .expect("write the script");
        assert!(output.status.success(), "the gold harness failed");
        output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }
}

/// How far the port is from the original, case by case: the largest
/// difference in any sample, and that difference against the output's
/// peak.
#[cfg(feature = "gold")]
#[test]
fn gold_errors() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("nam-gold-cases");
    std::fs::create_dir_all(&dir).unwrap();
    let mut out = String::new();
    let mut worst = 0.0f32;
    for (i, case) in cases().iter().enumerate() {
        let ours = run(case);
        let theirs = original::run(case, &dir, i);
        assert_eq!(ours.len(), theirs.len(), "{}: lengths differ", case.name);
        let max = ours
            .iter()
            .zip(&theirs)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let peak = theirs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let relative = max / peak.max(1e-6);
        worst = worst.max(max);
        let _ = writeln!(
            out,
            "{:<28} {:?}  max |diff| {max:.1e}  of peak {relative:.1e}",
            case.name, case.tanh
        );
    }
    let _ = writeln!(out, "worst: {worst:.1e} ({:.0} dB)", db(worst));
    expect![[r#"
        obsidian, DI level           Exact  max |diff| 1.8e-6  of peak 1.4e-5
        obsidian, hot                Exact  max |diff| 2.3e-6  of peak 1.2e-5
        obsidian, silence            Exact  max |diff| 4.8e-9  of peak 8.9e-5
        example, DI level            Exact  max |diff| 2.7e-5  of peak 6.1e-5
        standard                     Exact  max |diff| 5.6e-9  of peak 1.2e-6
        lite                         Exact  max |diff| 7.1e-9  of peak 3.3e-6
        feather                      Exact  max |diff| 8.4e-9  of peak 8.9e-6
        nano                         Exact  max |diff| 9.3e-10  of peak 2.0e-7
        gated-nano                   Exact  max |diff| 2.1e-9  of peak 5.1e-7
        lstm-16                      Exact  max |diff| 3.0e-8  of peak 1.3e-7
        tiny lstm                    Exact  max |diff| 8.9e-8  of peak 5.4e-6
        obsidian, DI level           Fast  max |diff| 1.7e-6  of peak 1.4e-5
        obsidian, hot                Fast  max |diff| 2.2e-6  of peak 1.1e-5
        obsidian, silence            Fast  max |diff| 1.2e-9  of peak 1.3e-5
        example, DI level            Fast  max |diff| 2.8e-5  of peak 6.3e-5
        standard                     Fast  max |diff| 4.9e-9  of peak 1.0e-6
        lite                         Fast  max |diff| 6.9e-9  of peak 3.2e-6
        feather                      Fast  max |diff| 7.3e-9  of peak 7.7e-6
        nano                         Fast  max |diff| 9.3e-10  of peak 2.0e-7
        gated-nano                   Fast  max |diff| 2.6e-9  of peak 6.3e-7
        lstm-16                      Fast  max |diff| 3.0e-8  of peak 1.3e-7
        tiny lstm                    Fast  max |diff| 1.5e-7  of peak 9.2e-6
        obsidian, blocks of 1        Fast  max |diff| 1.4e-6  of peak 1.1e-5
        obsidian, blocks of 37       Fast  max |diff| 1.4e-6  of peak 1.1e-5
        obsidian, blocks of 100      Fast  max |diff| 1.4e-6  of peak 1.1e-5
        worst: 2.8e-5 (-91 dB)
    "#]]
    .assert_eq(&out);
    assert!(
        worst < 1e-4,
        "the port is further from the original than float rounding: {worst:e}"
    );
}
