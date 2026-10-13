//! Gold tests: each port against the Airwindows C++ it was ported from.
//!
//! A scenario is a [`Program`] — which plug-in, at what sample rate, which
//! knobs set to what, what audio it hears in what size of blocks, and when
//! the knobs move, the instance is cleared or the sample rate changes. It
//! runs through the Rust port, and a summary of what came out is the
//! snapshot: for each passage, each channel's level in and out and a strip
//! of its envelope over time, and a hash of every output sample.
//!
//! With `--features gold` the same program also runs through the original
//! — the plug-in's own `<Name>.cpp` and `<Name>Proc.cpp`, unmodified,
//! compiled with g++ against a stand-in for the VST SDK header
//! (`tests/cpp_harness`) — and the two outputs must agree bit for bit. The
//! snapshot's hashes are not checked then, since gold switches the Rust
//! side to the platform's maths library (src/math.rs) and the hash that
//! ships is libm's; everything else in it is.
//!
//! The sources come from the `vendor/airwindows` submodule:
//!
//! ```sh
//! git submodule update --init vendor/airwindows
//! ```
//!
//! or, to use a checkout elsewhere, from `AIRWINDOWS_SRC=<path to the
//! airwindows repository>`.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a scenario's length is its snapshot's: five lines a passage"
)]

use bloop_plugin_airwindows::{Airwindows, Kind};
use expect_test::{expect, Expect};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

enum Step {
    Set(usize, f32),
    Rate(f32),
    Clear,
    Play {
        label: &'static str,
        signal: Signal,
        block: usize,
    },
}

struct Program {
    kind: Kind,
    sample_rate: f32,
    steps: Vec<Step>,
}

impl Program {
    fn new(kind: Kind, sample_rate: f32) -> Self {
        Self {
            kind,
            sample_rate,
            steps: Vec::new(),
        }
    }

    /// The rate the program is running at from here on.
    fn rate(&self) -> f32 {
        self.steps
            .iter()
            .rev()
            .find_map(|s| match s {
                Step::Rate(rate) => Some(*rate),
                _ => None,
            })
            .unwrap_or(self.sample_rate)
    }

    /// The knob called `name` to `value`. Knobs are found by the names the
    /// original gives them, so a misnamed or renumbered knob fails here.
    fn set(mut self, name: &str, value: f32) -> Self {
        let matches: Vec<usize> = (0..self.kind.params().len())
            .filter(|&i| self.kind.params()[i].name == name)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "{:?} has {} knobs called {name:?}",
            self.kind,
            matches.len()
        );
        self.steps.push(Step::Set(matches[0], value));
        self
    }

    fn set_all(mut self, settings: &[(&str, f32)]) -> Self {
        for (name, value) in settings {
            self = self.set(name, *value);
        }
        self
    }

    /// Every knob to `value`.
    fn every_knob(mut self, value: f32) -> Self {
        for i in 0..self.kind.params().len() {
            self.steps.push(Step::Set(i, value));
        }
        self
    }

    fn sample_rate(mut self, rate: f32) -> Self {
        self.steps.push(Step::Rate(rate));
        self
    }

    fn clear(mut self) -> Self {
        self.steps.push(Step::Clear);
        self
    }

    /// `signal`, made at the program's current rate, through the plug-in
    /// in blocks of `block` samples.
    fn play(mut self, label: &'static str, signal: &Source, block: usize) -> Self {
        let signal = (signal.0)(f64::from(self.rate()));
        self.steps.push(Step::Play {
            label,
            signal,
            block,
        });
        self
    }
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Signal {
    left: Vec<f32>,
    right: Vec<f32>,
}

/// A signal, made at whatever rate the program is running at.
struct Source(Box<dyn Fn(f64) -> Signal>);

/// A repeatable noise source (xorshift32), so no platform's RNG is in it.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        f64::from(self.0) / f64::from(u32::MAX) * 2.0 - 1.0
    }
}

fn gain(db: f64) -> f64 {
    libm::pow(10.0, db / 20.0)
}

fn count(seconds: f64, rate: f64) -> usize {
    (seconds * rate) as usize
}

/// A sine at `db` plus noise at `noise_db`, the noise different in each
/// channel and the right's sine a fifth up.
fn tone(hz: f64, db: f64, noise_db: f64, seconds: f64) -> Source {
    Source(Box::new(move |rate| {
        let n = count(seconds, rate);
        let mut nl = Noise(0x1234_5678);
        let mut nr = Noise(0x9abc_def0);
        let sine = |i: usize, hz: f64| {
            gain(db) * libm::sin(2.0 * core::f64::consts::PI * hz * i as f64 / rate)
        };
        Signal {
            left: (0..n)
                .map(|i| (sine(i, hz) + gain(noise_db) * nl.next()) as f32)
                .collect(),
            right: (0..n)
                .map(|i| (sine(i, hz * 1.5) + gain(noise_db) * nr.next()) as f32)
                .collect(),
        }
    }))
}

/// A sine at -30 dBFS under noise at -50.
fn quiet() -> Source {
    tone(440.0, -30.0, -50.0, 0.5)
}

/// A sine at +9 dBFS and noise at -3: peaks near +12 dBFS.
fn hot() -> Source {
    tone(110.0, 9.0, -3.0, 0.5)
}

fn silence(seconds: f64) -> Source {
    Source(Box::new(move |rate| Signal {
        left: vec![0.0; count(seconds, rate)],
        right: vec![0.0; count(seconds, rate)],
    }))
}

/// Drum-like hits: a burst of noise decaying over 60 ms every quarter
/// second, alternately at -3 and -21 dBFS, the right a different noise.
fn drums() -> Source {
    Source(Box::new(|rate| {
        let period = count(0.25, rate);
        let mut l = Noise(0x0bad_cafe);
        let mut r = Noise(0x5eed_1234);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for hit in 0..4 {
            let level = if hit % 2 == 0 { -3.0 } else { -21.0 };
            for i in 0..period {
                let env = gain(level) * libm::exp(-(i as f64) / (0.06 * rate));
                left.push((env * l.next()) as f32);
                right.push((env * r.next()) as f32);
            }
        }
        Signal { left, right }
    }))
}

// ---------------------------------------------------------------------------
// The standard programs every plug-in runs
// ---------------------------------------------------------------------------

/// The constructor's knobs at 44.1 kHz: quiet, hot, then a tail.
fn defaults(kind: Kind) -> Program {
    Program::new(kind, 44100.0)
        .play("quiet", &quiet(), 64)
        .play("hot", &hot(), 128)
        .play("after", &silence(1.5), 333)
}

/// Every knob at 0, at 48 kHz, one sample at a time first.
fn all_zero(kind: Kind) -> Program {
    Program::new(kind, 48000.0)
        .every_knob(0.0)
        .play("quiet", &quiet(), 1)
        .play("hot", &hot(), 333)
        .play("after", &silence(0.5), 64)
}

/// Every knob at 1, at 96 kHz.
fn all_one(kind: Kind) -> Program {
    Program::new(kind, 96000.0)
        .every_knob(1.0)
        .play("quiet", &quiet(), 128)
        .play("hot", &hot(), 64)
        .play("after", &silence(1.0), 1)
}

/// Some knobs set; drums; one knob moved mid-run; drums again; cleared;
/// hot. At 96 kHz the rate also drops to 48 kHz after the clear.
fn mixed(
    kind: Kind,
    rate: f32,
    block: usize,
    settings: &[(&str, f32)],
    change: (&str, f32),
) -> Program {
    let program = Program::new(kind, rate)
        .set_all(settings)
        .play("drums", &drums(), block)
        .set(change.0, change.1)
        .play("knob moved", &drums(), block)
        .clear();
    let program = if rate > 48000.0 {
        program.sample_rate(48000.0)
    } else {
        program
    };
    program
        .play("cleared", &hot(), block)
        .play("after", &silence(0.5), block)
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// What one play step produced.
struct Played {
    label: &'static str,
    block: usize,
    rate: f32,
    input: Signal,
    output: Signal,
}

fn run_rust(program: &Program) -> Vec<Played> {
    let mut plugin = Airwindows::new(program.kind, program.sample_rate);
    let mut played = Vec::new();
    for step in &program.steps {
        match step {
            Step::Set(index, value) => plugin.set_param(*index, *value),
            Step::Rate(rate) => plugin.set_sample_rate(*rate),
            Step::Clear => plugin.clear(),
            Step::Play {
                label,
                signal,
                block,
            } => {
                let mut out = signal.clone();
                for (l, r) in out
                    .left
                    .chunks_mut(*block)
                    .zip(out.right.chunks_mut(*block))
                {
                    plugin.process(l, r);
                }
                played.push(Played {
                    label,
                    block: *block,
                    rate: plugin.sample_rate(),
                    input: signal.clone(),
                    output: out,
                });
            }
        }
    }
    played
}

fn db(x: f64) -> f64 {
    if x > 0.0 {
        20.0 * libm::log10(x)
    } else {
        f64::NEG_INFINITY
    }
}

fn levels(samples: &[f32]) -> String {
    let peak = samples
        .iter()
        .fold(0.0f64, |m, &s| m.max(f64::from(s.abs())));
    let rms = (samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len().max(1) as f64)
        .sqrt();
    let show = |x: f64| {
        let d = db(x);
        if x == 0.0 {
            "  zero".to_string()
        } else if d < -120.0 {
            "  -inf".to_string()
        } else {
            format!("{d:>6.1}")
        }
    };
    format!("{} pk {} rms", show(peak), show(rms))
}

/// The envelope over time: forty columns, each the peak of its stretch in
/// 6 dB steps, from blank (below -54 dB) to `@` (above -6 dB).
fn strip(samples: &[f32]) -> String {
    const RAMP: &[u8] = b" .:-=+*#%@";
    const COLUMNS: usize = 40;
    let per = samples.len().div_ceil(COLUMNS).max(1);
    samples
        .chunks(per)
        .map(|chunk| {
            let peak = chunk.iter().fold(0.0f64, |m, &s| m.max(f64::from(s.abs())));
            let level = ((db(peak) + 54.0) / 6.0).floor() + 1.0;
            let index = if level.is_finite() {
                level.clamp(0.0, 9.0) as usize
            } else {
                0
            };
            RAMP[index] as char
        })
        .collect()
}

fn fnv(signal: &Signal) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for s in signal.left.iter().chain(&signal.right) {
        for byte in s.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

fn summary(played: &[Played]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for p in played {
        let seconds = p.input.left.len() as f64 / f64::from(p.rate);
        let _ = writeln!(
            out,
            "{} — {seconds:.2}s at {}Hz in blocks of {}",
            p.label, p.rate, p.block
        );
        for (name, input, output) in [
            ("L", &p.input.left, &p.output.left),
            ("R", &p.input.right, &p.output.right),
        ] {
            let _ = writeln!(out, "  {name} in  {}  |{}|", levels(input), strip(input));
            let _ = writeln!(out, "  {name} out {}  |{}|", levels(output), strip(output));
        }
        let _ = writeln!(out, "  hash {:016x}", fnv(&p.output));
    }
    out
}

/// A snapshot as `expect_test` compares it: its common indent removed.
#[cfg(feature = "gold")]
fn dedent(text: &str) -> String {
    let text = text.strip_prefix('\n').unwrap_or(text);
    let indent = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    text.lines()
        .map(|l| l.get(indent..).unwrap_or("").trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(feature = "gold")]
fn without_hashes(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim_start().starts_with("hash "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn check(program: &Program, expect: &Expect) {
    let played = run_rust(program);
    #[cfg(feature = "gold")]
    {
        let cpp = cpp::run(program);
        compare(program, &played, &cpp);
        assert_eq!(
            without_hashes(&dedent(expect.data())),
            without_hashes(&dedent(&summary(&played))),
            "{:?}: the summary under gold differs from the snapshot beyond its hashes",
            program.kind
        );
    }
    #[cfg(not(feature = "gold"))]
    expect.assert_eq(&summary(&played));
}

/// The first sample where the two sides disagree. Two NaNs agree whatever
/// their payloads.
#[cfg(feature = "gold")]
fn compare(program: &Program, rust: &[Played], cpp: &[Signal]) {
    use std::sync::atomic::Ordering;
    assert_eq!(rust.len(), cpp.len(), "{:?}: passage count", program.kind);
    for (p, theirs) in rust.iter().zip(cpp) {
        for (channel, ours, theirs, input) in [
            ("left", &p.output.left, &theirs.left, &p.input.left),
            ("right", &p.output.right, &theirs.right, &p.input.right),
        ] {
            assert_eq!(
                ours.len(),
                theirs.len(),
                "{:?}, '{}': length",
                program.kind,
                p.label
            );
            let first = ours
                .iter()
                .zip(theirs.iter())
                .position(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()));
            assert!(
                first.is_none(),
                "{}",
                first.map_or_else(String::new, |i| format!(
                    "{:?}, '{}', {channel}: sample {i} of {} differs — rust {:e} ({:#010x}), C++ {:e} ({:#010x}); input {:e}\n{}",
                    program.kind,
                    p.label,
                    ours.len(),
                    ours[i],
                    ours[i].to_bits(),
                    theirs[i],
                    theirs[i].to_bits(),
                    input[i],
                    (i.saturating_sub(4)..(i + 4).min(ours.len()))
                        .map(|j| format!("  [{j}] in {:e}  rust {:e}  C++ {:e}", input[j], ours[j], theirs[j]))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ))
            );
            cpp::COMPARED.fetch_add(ours.len() as u64, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// The C++ side
// ---------------------------------------------------------------------------

#[cfg(feature = "gold")]
#[expect(
    clippy::expect_used,
    reason = "a harness that cannot build or run the originals has no one to hand the error to but the test"
)]
mod cpp {
    use super::{Program, Signal, Step};
    use bloop_plugin_airwindows::Kind;
    use std::collections::HashMap;
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::atomic::AtomicU64;
    use std::sync::{Mutex, OnceLock};
    use std::time::SystemTime;

    /// Samples compared bit for bit, per channel, across every test in
    /// this binary; `gold_tally` prints it.
    pub static COMPARED: AtomicU64 = AtomicU64::new(0);

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// The airwindows checkout: `AIRWINDOWS_SRC` if set, else the
    /// vendor/airwindows submodule.
    fn airwindows() -> PathBuf {
        if let Some(path) = std::env::var_os("AIRWINDOWS_SRC") {
            return PathBuf::from(path);
        }
        let path = root().join("../../vendor/airwindows");
        assert!(
            path.join("plugins/LinuxVST/src").exists(),
            "vendor/airwindows is not checked out. The gold tests compile the originals from it:\n  \
             git submodule update --init vendor/airwindows\n\
             (or point AIRWINDOWS_SRC at an airwindows checkout)"
        );
        path
    }

    fn mtime(path: &Path) -> SystemTime {
        path.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    }

    /// One plug-in's harness binary, built unless it is newer than every
    /// file it is built from.
    fn build(kind: Kind) -> PathBuf {
        let name = kind.name();
        let harness = root().join("tests/cpp_harness");
        let src = airwindows().join("plugins/LinuxVST/src").join(name);
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("airwindows-gold");
        std::fs::create_dir_all(&out).expect("create the harness build directory");
        let binary = out.join(name);
        let sources = [
            harness.join("main.cpp"),
            harness.join("audioeffectx.h"),
            src.join(format!("{name}.h")),
            src.join(format!("{name}.cpp")),
            src.join(format!("{name}Proc.cpp")),
        ];
        let built = mtime(&binary);
        if binary.exists() && sources.iter().all(|s| mtime(s) <= built) {
            return binary;
        }
        // -mlong-double-64: the dither's `5.5e-36l` at the width the port
        // computes it in (src/math.rs). -ffp-contract=off: the port never
        // fuses a multiply and an add, so neither may the compiler.
        let status = Command::new("g++")
            .args([
                "-std=c++17",
                "-O2",
                "-w",
                "-mlong-double-64",
                "-ffp-contract=off",
            ])
            .arg("-I")
            .arg(&harness)
            .arg("-I")
            .arg(&src)
            .arg(format!("-DPLUG={name}"))
            .arg(format!("-DPLUG_H=\"{name}.h\""))
            .arg(&sources[0])
            .arg(&sources[3])
            .arg(&sources[4])
            .arg("-o")
            .arg(&binary)
            .status()
            .expect("run g++ (the gold tests need it installed)");
        assert!(status.success(), "compiling the {name} gold harness failed");
        binary
    }

    fn binary(kind: Kind) -> PathBuf {
        static BINARIES: OnceLock<Mutex<HashMap<Kind, PathBuf>>> = OnceLock::new();
        let mut binaries = BINARIES
            .get_or_init(Mutex::default)
            .lock()
            .expect("the harness table");
        binaries.entry(kind).or_insert_with(|| build(kind)).clone()
    }

    /// The program through the original: one signal per passage.
    pub fn run(program: &Program) -> Vec<Signal> {
        let mut input = Vec::new();
        let _ = writeln!(input, "new {:?}", f64::from(program.sample_rate));
        let mut blocks_per_step = Vec::new();
        for step in &program.steps {
            match step {
                Step::Set(index, value) => {
                    let _ = writeln!(input, "set {index} {:08x}", value.to_bits());
                }
                Step::Rate(rate) => {
                    let _ = writeln!(input, "rate {:?}", f64::from(*rate));
                }
                Step::Clear => {
                    let _ = writeln!(input, "clear");
                }
                Step::Play { signal, block, .. } => {
                    let mut sizes = Vec::new();
                    for (l, r) in signal.left.chunks(*block).zip(signal.right.chunks(*block)) {
                        let _ = writeln!(input, "process {}", l.len());
                        for s in l.iter().chain(r) {
                            input.extend_from_slice(&s.to_le_bytes());
                        }
                        input.push(b'\n');
                        sizes.push(l.len());
                    }
                    blocks_per_step.push(sizes);
                }
            }
        }

        let mut child = Command::new(binary(program.kind))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the gold harness");
        let mut stdin = child.stdin.take().expect("harness stdin");
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let output = child.wait_with_output().expect("run the gold harness");
        writer
            .join()
            .expect("join the harness writer")
            .expect("write the program to the harness");
        assert!(
            output.status.success(),
            "the gold harness failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let floats: Vec<f32> = output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut at = 0;
        blocks_per_step
            .into_iter()
            .map(|sizes| {
                let (mut left, mut right) = (Vec::new(), Vec::new());
                for n in sizes {
                    left.extend_from_slice(&floats[at..at + n]);
                    right.extend_from_slice(&floats[at + n..at + 2 * n]);
                    at += 2 * n;
                }
                Signal { left, right }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn console7_channel() {
    let k = Kind::Console7Channel;
    check(
        &defaults(k),
        &expect![[r#"
        quiet — 0.50s at 44100Hz in blocks of 64
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -29.0 pk  -33.3 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -29.0 pk  -33.3 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash 0f0de1df26e77797
        hot — 0.50s at 44100Hz in blocks of 128
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -1.2 pk   -1.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -1.2 pk   -1.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash dfc7484c93cedd34
        after — 1.50s at 44100Hz in blocks of 333
          L in    zero pk   zero rms  |                                        |
          L out  -12.1 pk  -54.0 rms  |#                                       |
          R in    zero pk   zero rms  |                                        |
          R out  -30.9 pk  -71.2 rms  |=                                       |
          hash 5f5fd7ec3ca50933
    "#]],
    );
    check(
        &all_zero(k),
        &expect![[r#"
        quiet — 0.50s at 48000Hz in blocks of 1
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out   -inf pk   -inf rms  |                                        |
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out   -inf pk   -inf rms  |                                        |
          hash 6b2242325ecf94a0
        hot — 0.50s at 48000Hz in blocks of 333
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -inf pk   -inf rms  |                                        |
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -inf pk   -inf rms  |                                        |
          hash 256c9631bc80a742
        after — 0.50s at 48000Hz in blocks of 64
          L in    zero pk   zero rms  |                                        |
          L out   -inf pk   -inf rms  |                                        |
          R in    zero pk   zero rms  |                                        |
          R out   -inf pk   -inf rms  |                                        |
          hash 18e0603501ac6322
    "#]],
    );
    check(
        &all_one(k),
        &expect![[r#"
        quiet — 0.50s at 96000Hz in blocks of 128
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -24.5 pk  -28.8 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -24.6 pk  -28.8 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash 32ecdd3b9abfd519
        hot — 0.50s at 96000Hz in blocks of 64
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -3.4 pk   -3.7 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -3.4 pk   -3.7 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 8bd4409e7b4c05f7
        after — 1.00s at 96000Hz in blocks of 1
          L in    zero pk   zero rms  |                                        |
          L out   -7.8 pk  -54.7 rms  |%                                       |
          R in    zero pk   zero rms  |                                        |
          R out  -11.2 pk  -57.7 rms  |%                                       |
          hash 3cedd84a3625096d
    "#]],
    );
    check(
        &mixed(k, 44100.0, 333, &[("Fader", 0.5)], ("Fader", 0.9)),
        &expect![[r#"
            drums — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -6.7 pk  -27.3 rms  |%%##*++=-+++==-::. %%%#**+==-+++==-::.  |
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -5.9 pk  -27.2 rms  |%%#**++=-=++=--::. %@%#**+==-+++=--::.  |
              hash 6c86037a0b74bd1c
            knob moved — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -2.0 pk  -17.9 rms  |@@@%###*+*#**+==--:@@@@%%#**+###*++=--:.|
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -1.2 pk  -17.9 rms  |@@@%##**+*#**++=--:@@@@%%#**+##**++=--::|
              hash b69abe4b2c3dcc99
            cleared — 0.50s at 44100Hz in blocks of 333
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -2.5 pk   -2.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -2.5 pk   -2.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash cbc8f73cc606ec10
            after — 0.50s at 44100Hz in blocks of 333
              L in    zero pk   zero rms  |                                        |
              L out   -9.5 pk  -46.6 rms  |%                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -28.3 pk  -63.7 rms  |+                                       |
              hash 9017d22cc92fe575
        "#]],
    );
    check(
        &mixed(k, 96000.0, 1, &[("Fader", 0.95)], ("Fader", 0.2)),
        &expect![[r#"
            drums — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -3.0 pk  -17.8 rms  |@@@%%##*++#**++=--::@@@%%#**++#**++=--::|
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -3.0 pk  -17.8 rms  |@@@%%##*++#**++=--::@@@@%#**++#**++=--::|
              hash d5d8c99d2258b4a1
            knob moved — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -2.7 pk  -40.6 rms  |@+==--:.. -:..      *+==--:.. -:..      |
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -2.4 pk  -39.4 rms  |@++=--:.. -:..      *++=--:.. -:..      |
              hash 9fc8fdce20df3b9d
            cleared — 0.50s at 48000Hz in blocks of 1
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out  -11.4 pk  -17.6 rms  |#%#%#%%####%%#%#%%##%##%#####%#%#%#####%|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out  -11.2 pk  -17.6 rms  |###%%##%#%%######%####%#%%####%#%%######|
              hash de85a4258480aaf7
            after — 0.50s at 48000Hz in blocks of 1
              L in    zero pk   zero rms  |                                        |
              L out  -42.3 pk  -80.3 rms  |:                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -36.0 pk  -74.4 rms  |-                                       |
              hash 6332109521b0b1da
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn console7_buss() {
    let k = Kind::Console7Buss;
    check(
        &defaults(k),
        &expect![[r#"
        quiet — 0.50s at 44100Hz in blocks of 64
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -28.6 pk  -32.7 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -28.7 pk  -32.7 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash d938aed0ae7172e0
        hot — 0.50s at 44100Hz in blocks of 128
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out    6.0 pk    3.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out    6.5 pk    3.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 1e604390035374c0
        after — 1.50s at 44100Hz in blocks of 333
          L in    zero pk   zero rms  |                                        |
          L out  -20.3 pk  -65.1 rms  |*                                       |
          R in    zero pk   zero rms  |                                        |
          R out  -12.9 pk  -60.3 rms  |#                                       |
          hash 29d25528f705221b
    "#]],
    );
    check(
        &all_zero(k),
        &expect![[r#"
        quiet — 0.50s at 48000Hz in blocks of 1
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out   -inf pk   -inf rms  |                                        |
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out   -inf pk   -inf rms  |                                        |
          hash 6b2242325ecf94a0
        hot — 0.50s at 48000Hz in blocks of 333
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -inf pk   -inf rms  |                                        |
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -inf pk   -inf rms  |                                        |
          hash 256c9631bc80a742
        after — 0.50s at 48000Hz in blocks of 64
          L in    zero pk   zero rms  |                                        |
          L out   -inf pk   -inf rms  |                                        |
          R in    zero pk   zero rms  |                                        |
          R out   -inf pk   -inf rms  |                                        |
          hash 18e0603501ac6322
    "#]],
    );
    check(
        &all_one(k),
        &expect![[r#"
        quiet — 0.50s at 96000Hz in blocks of 128
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -29.0 pk  -32.7 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -29.1 pk  -32.8 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash f9fe564d8d88eb4a
        hot — 0.50s at 96000Hz in blocks of 64
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out    4.1 pk    3.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out    4.1 pk    3.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 89422ffd213ee5a8
        after — 1.00s at 96000Hz in blocks of 1
          L in    zero pk   zero rms  |                                        |
          L out  -14.3 pk  -59.7 rms  |#                                       |
          R in    zero pk   zero rms  |                                        |
          R out  -11.2 pk  -60.4 rms  |%                                       |
          hash 9b41f3fb2f0486f0
    "#]],
    );
    check(
        &mixed(k, 44100.0, 333, &[("Master", 0.6)], ("Master", 0.85)),
        &expect![[r#"
            drums — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -4.7 pk  -25.0 rms  |@%##**+==+*+==--:..%@%%#**+==+*+==--:.. |
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -5.0 pk  -25.0 rms  |@%##**+===*++=--:..%@%##**+==+++==--:.. |
              hash a6b79eb170c8cce4
            knob moved — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -1.4 pk  -22.2 rms  |@@%##**+=***+==-::.@@%%##**+=***+==--:..|
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -1.8 pk  -22.1 rms  |@@%##*++=+**+==--:.@@@%##**+=***+==-::..|
              hash c28dca123e86c437
            cleared — 0.50s at 44100Hz in blocks of 333
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out    5.1 pk    2.3 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out    5.3 pk    2.3 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 6c8f142e889d512e
            after — 0.50s at 44100Hz in blocks of 333
              L in    zero pk   zero rms  |                                        |
              L out  -21.6 pk  -61.8 rms  |*                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -14.4 pk  -57.0 rms  |#                                       |
              hash 32aece4d7cf88b8a
        "#]],
    );
    check(
        &mixed(k, 96000.0, 1, &[("Master", 0.3)], ("Master", 1.0)),
        &expect![[r#"
            drums — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out  -15.4 pk  -36.1 rms  |#**++=--::=--::.    #**++=--:.=--:..    |
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out  -15.1 pk  -36.0 rms  |#**++=--:.=--:..    #**+==--::=--:..    |
              hash 491f48a31710779b
            knob moved — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -4.7 pk  -25.7 rms  |@%%#**+==-*++=--:.. @%##**++=-*+==--:.. |
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -4.7 pk  -25.6 rms  |@%##**+==-*+==--:.. @%##**+==-*+==--:.. |
              hash 5b9028d5f4c2a0eb
            cleared — 0.50s at 48000Hz in blocks of 1
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out    5.5 pk    3.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out    5.6 pk    3.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 8c016f147c73d750
            after — 0.50s at 48000Hz in blocks of 1
              L in    zero pk   zero rms  |                                        |
              L out  -12.8 pk  -56.5 rms  |#                                       |
              R in    zero pk   zero rms  |                                        |
              R out   -6.4 pk  -49.9 rms  |%                                       |
              hash a825808a22f9877d
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn to_tape8() {
    let k = Kind::ToTape8;
    check(
        &defaults(k),
        &expect![[r#"
        quiet — 0.50s at 44100Hz in blocks of 64
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -29.2 pk  -33.1 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash 25f6a3f17fc953e2
        hot — 0.50s at 44100Hz in blocks of 128
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -0.4 pk   -1.6 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -0.4 pk   -1.8 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 2d7622fbfab6e54c
        after — 1.50s at 44100Hz in blocks of 333
          L in    zero pk   zero rms  |                                        |
          L out   -2.3 pk  -29.2 rms  |@                                       |
          R in    zero pk   zero rms  |                                        |
          R out   -2.8 pk  -31.6 rms  |@                                       |
          hash c74ab801d999b34c
    "#]],
    );
    check(
        &all_zero(k),
        &expect![[r#"
        quiet — 0.50s at 48000Hz in blocks of 1
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out   -inf pk   -inf rms  |                                        |
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out   -inf pk   -inf rms  |                                        |
          hash 6b2242325ecf94a0
        hot — 0.50s at 48000Hz in blocks of 333
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -inf pk   -inf rms  |                                        |
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -inf pk   -inf rms  |                                        |
          hash 256c9631bc80a742
        after — 0.50s at 48000Hz in blocks of 64
          L in    zero pk   zero rms  |                                        |
          L out   -inf pk   -inf rms  |                                        |
          R in    zero pk   zero rms  |                                        |
          R out   -inf pk   -inf rms  |                                        |
          hash 18e0603501ac6322
    "#]],
    );
    check(
        &all_one(k),
        &expect![[r#"
        quiet — 0.50s at 96000Hz in blocks of 128
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out   -inf pk   -inf rms  |                                        |
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out   -inf pk   -inf rms  |                                        |
          hash 02e69cbcc481295b
        hot — 0.50s at 96000Hz in blocks of 64
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -inf pk   -inf rms  |                                        |
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -inf pk   -inf rms  |                                        |
          hash 4c5b16942ae298a2
        after — 1.00s at 96000Hz in blocks of 1
          L in    zero pk   zero rms  |                                        |
          L out   -inf pk   -inf rms  |                                        |
          R in    zero pk   zero rms  |                                        |
          R out   -inf pk   -inf rms  |                                        |
          hash 40dc561ddf2f23f8
    "#]],
    );
    check(
        &mixed(
            k,
            44100.0,
            333,
            &[
                ("Input", 0.7),
                ("Tilt", 0.3),
                ("Shape", 0.8),
                ("Flutter", 0.6),
                ("FlutSpd", 0.9),
                ("Bias", 0.2),
                ("HeadBmp", 0.7),
                ("HeadFrq", 0.3),
                ("Output", 0.6),
            ],
            ("Bias", 0.8),
        ),
        &expect![[r#"
            drums — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -0.4 pk  -14.2 rms  |@@@@%%#*++%#*++:   @@@@@%%#*+###*++     |
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -0.4 pk  -14.1 rms  |@@@@%##*+*##*++.   @@@@@%##*+###*++:.   |
              hash 0bfe4e079bf64dc3
            knob moved — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -1.5 pk  -20.4 rms  |@@@%%##*+*##**+==--*@@%%%##*+###**+==-::|
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -1.9 pk  -20.4 rms  |@@@%%##*++##*++==-:#@@@%%##**###**++=-::|
              hash e167b9d1a7e8a2df
            cleared — 0.50s at 44100Hz in blocks of 333
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -0.4 pk   -0.8 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -0.4 pk   -0.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 75de7778ce49dfed
            after — 0.50s at 44100Hz in blocks of 333
              L in    zero pk   zero rms  |                                        |
              L out   -0.8 pk  -23.3 rms  |@*+=:                                   |
              R in    zero pk   zero rms  |                                        |
              R out   -0.4 pk  -25.2 rms  |@*+=:                                   |
              hash 1ddbed1545f58541
        "#]],
    );
    check(
        &mixed(
            k,
            96000.0,
            1,
            &[
                ("Input", 0.9),
                ("Tilt", 0.65),
                ("Shape", 0.25),
                ("Flutter", 0.85),
                ("FlutSpd", 0.4),
                ("Bias", 0.65),
                ("HeadBmp", 0.3),
                ("HeadFrq", 0.9),
                ("Output", 0.45),
            ],
            ("Flutter", 0.0),
        ),
        &expect![[r#"
            drums — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -2.3 pk  -21.3 rms  |@@%%###**+##**+==--:@@%%###**+##**++=--:|
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -1.7 pk  -21.2 rms  |@@@%###**+##**++=--:@@%%%##**+##**+==--:|
              hash 90807d9cdbb018a5
            knob moved — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -2.1 pk  -20.3 rms  |@@%%%##**+##**++=--:@@%%%##**+##**++=--:|
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -2.0 pk  -20.2 rms  |@@@%###**+##**++=--:@@@%%##**+##**++=--:|
              hash 459988f8bf4a912d
            cleared — 0.50s at 48000Hz in blocks of 1
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -0.4 pk   -1.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -0.4 pk   -1.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash e96342a24f4a83db
            after — 0.50s at 48000Hz in blocks of 1
              L in    zero pk   zero rms  |                                        |
              L out   -0.4 pk  -20.6 rms  |@*.                                     |
              R in    zero pk   zero rms  |                                        |
              R out   -0.4 pk  -21.6 rms  |@+                                      |
              hash 1e04f90286456288
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn density2() {
    let k = Kind::Density2;
    check(
        &defaults(k),
        &expect![[r#"
        quiet — 0.50s at 44100Hz in blocks of 64
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash c6d888018992c12f
        hot — 0.50s at 44100Hz in blocks of 128
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash c71c2fa1f82f80c9
        after — 1.50s at 44100Hz in blocks of 333
          L in    zero pk   zero rms  |                                        |
          L out   -inf pk   -inf rms  |                                        |
          R in    zero pk   zero rms  |                                        |
          R out   -inf pk   -inf rms  |                                        |
          hash 2a4673d9a5efda1e
    "#]],
    );
    check(
        &all_zero(k),
        &expect![[r#"
        quiet — 0.50s at 48000Hz in blocks of 1
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash 54fba3c11ce19546
        hot — 0.50s at 48000Hz in blocks of 333
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 733831dc6c10ab4a
        after — 0.50s at 48000Hz in blocks of 64
          L in    zero pk   zero rms  |                                        |
          L out   -inf pk   -inf rms  |                                        |
          R in    zero pk   zero rms  |                                        |
          R out   -inf pk   -inf rms  |                                        |
          hash 2d5e77d350486365
    "#]],
    );
    check(
        &all_one(k),
        &expect![[r#"
        quiet — 0.50s at 96000Hz in blocks of 128
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out    4.9 pk   -4.5 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out    4.6 pk   -4.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 152c7965857b1748
        hot — 0.50s at 96000Hz in blocks of 64
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out    8.2 pk    0.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out    8.1 pk    0.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 7c808c4df7fbfb7b
        after — 1.00s at 96000Hz in blocks of 1
          L in    zero pk   zero rms  |                                        |
          L out   -0.2 pk  -40.6 rms  |@                                       |
          R in    zero pk   zero rms  |                                        |
          R out   -2.5 pk  -44.9 rms  |@                                       |
          hash 2850a35e5dd8df7f
    "#]],
    );
    check(
        &mixed(
            k,
            44100.0,
            333,
            &[
                ("Density", 0.7),
                ("Highpass", 0.4),
                ("Output", 0.8),
                ("Dry/Wet", 0.75),
            ],
            ("Density", 0.1),
        ),
        &expect![[r#"
            drums — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out    2.4 pk  -10.7 rms  |@@@@@@%%%@@%%%#**++@@@@@@@%%%@@%%%#**++=|
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out    2.7 pk  -10.7 rms  |@@@@@@%%%%@%%%#**++@@@@@@@%%%@@%%%#**++=|
              hash 825ba56121f8f981
            knob moved — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -4.9 pk  -23.1 rms  |@%##*++=-+++==-::..@@%##*++==+++=--::.. |
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -4.6 pk  -23.1 rms  |@%##*++==+++==-::..@@%##*++==+++==-::.. |
              hash e76dd50b0ab97474
            cleared — 0.50s at 44100Hz in blocks of 333
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out    3.7 pk   -3.3 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out    3.9 pk   -2.4 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 7b71e961a7bdad1a
            after — 0.50s at 44100Hz in blocks of 333
              L in    zero pk   zero rms  |                                        |
              L out   -9.6 pk  -45.1 rms  |%                                       |
              R in    zero pk   zero rms  |                                        |
              R out   -4.4 pk  -39.5 rms  |@                                       |
              hash 635186692e5a0964
        "#]],
    );
    check(
        &mixed(
            k,
            96000.0,
            1,
            &[
                ("Density", 0.45),
                ("Highpass", 0.9),
                ("Output", 0.5),
                ("Dry/Wet", 1.0),
            ],
            ("Highpass", 0.0),
        ),
        &expect![[r#"
            drums — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -6.4 pk  -24.4 rms  |%%##**+==-*++=--::. %%##**+==-*+==--:.. |
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -6.1 pk  -24.4 rms  |%%%#**+==-*++=--::. %%##**+==-*+==--::. |
              hash a7aab020b3acba7f
            knob moved — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -6.4 pk  -22.3 rms  |%%##**++=-*++=--::. %%##**++=-*++=--::..|
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -6.5 pk  -22.3 rms  |%%%#**++=-*++=--::..%%##**++=-*++=--::..|
              hash c1f6e5d9f11fd0fe
            cleared — 0.50s at 48000Hz in blocks of 1
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -3.9 pk   -6.7 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -3.9 pk   -6.7 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 07aab9b2b8298f1c
            after — 0.50s at 48000Hz in blocks of 1
              L in    zero pk   zero rms  |                                        |
              L out  -31.0 pk  -69.2 rms  |=                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -25.7 pk  -63.9 rms  |+                                       |
              hash ee60b21538fb9741
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn galactic3() {
    let k = Kind::Galactic3;
    check(
        &defaults(k),
        &expect![[r#"
        quiet — 0.50s at 44100Hz in blocks of 64
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -36.3 pk  -48.7 rms  |                ........:::....:::------|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -36.8 pk  -46.0 rms  |                ...::::::---------------|
          hash d3a8552ef1c79613
        hot — 0.50s at 44100Hz in blocks of 128
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -1.8 pk  -13.7 rms  |--=-------------#%#######%%%%%%@@%%%@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -1.4 pk  -12.6 rms  |-------------===##%#%@@@@@@%%%%%##%%%@@@|
          hash 6ea1086f5fcd6785
        after — 1.50s at 44100Hz in blocks of 333
          L in    zero pk   zero rms  |                                        |
          L out   12.9 pk    1.3 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    zero pk   zero rms  |                                        |
          R out    9.5 pk    1.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 21270f97454732d0
    "#]],
    );
    check(
        &all_zero(k),
        &expect![[r#"
        quiet — 0.50s at 48000Hz in blocks of 1
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash 54fba3c11ce19546
        hot — 0.50s at 48000Hz in blocks of 333
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 733831dc6c10ab4a
        after — 0.50s at 48000Hz in blocks of 64
          L in    zero pk   zero rms  |                                        |
          L out   -inf pk   -inf rms  |                                        |
          R in    zero pk   zero rms  |                                        |
          R out   -inf pk   -inf rms  |                                        |
          hash 2d5e77d350486365
    "#]],
    );
    check(
        &all_one(k),
        &expect![[r#"
        quiet — 0.50s at 96000Hz in blocks of 128
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -29.5 pk  -40.3 rms  |           :::---===-=-=+---====-=+=====|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -26.4 pk  -39.5 rms  |           ::::---=====-===++=======++++|
          hash 2fe2afb9916d518f
        hot — 0.50s at 96000Hz in blocks of 64
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   14.4 pk    3.1 rms  |=+++=++==+=%%@@@%@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out    9.3 pk   -1.7 rms  |=+++==++++=%%%@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash e0d6f983e4c04c51
        after — 1.00s at 96000Hz in blocks of 1
          L in    zero pk   zero rms  |                                        |
          L out   17.6 pk    5.4 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    zero pk   zero rms  |                                        |
          R out   15.8 pk    5.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 4d4dbf7596416cd2
    "#]],
    );
    check(
        &mixed(
            k,
            44100.0,
            333,
            &[
                ("Replace", 0.3),
                ("Brightns", 0.7),
                ("Detune", 0.8),
                ("Derez", 0.6),
                ("Bigness", 0.4),
                ("Dry/Wet", 0.6),
            ],
            ("Bigness", 0.9),
        ),
        &expect![[r#"
            drums — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out  -21.2 pk  -34.5 rms  |+===+++++++++*++++++**++**+****+********|
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out  -20.5 pk  -34.4 rms  |+===++++++++++++*+****+**++**********+**|
              hash da03776ab746c03f
            knob moved — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out  -20.2 pk  -34.9 rms  |***++=+++++++++++++****+++++++++*++*+***|
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out  -20.2 pk  -35.0 rms  |**+++==+++++++++++*****+*++++++++++*+**+|
              hash 311aa75685c86ae6
            cleared — 0.50s at 44100Hz in blocks of 333
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -0.7 pk  -10.8 rms  |###############*###%%%%@@@@@@@@@@@@%%%@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out    0.9 pk  -10.2 rms  |##############%###%%##%#####%%@@@@@@@@@@|
              hash e85d6301f02f04d2
            after — 0.50s at 44100Hz in blocks of 333
              L in    zero pk   zero rms  |                                        |
              L out    6.5 pk   -2.5 rms  |@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    zero pk   zero rms  |                                        |
              R out    6.5 pk   -4.0 rms  |@@%#%@@@@%%%%@@@@@%@@@@@@@@@@@@@@@@@@@@@|
              hash 54f3727fa4ce8e0e
        "#]],
    );
    check(
        &mixed(
            k,
            96000.0,
            1,
            &[
                ("Replace", 0.8),
                ("Brightns", 0.2),
                ("Detune", 0.3),
                ("Derez", 1.0),
                ("Bigness", 0.7),
                ("Dry/Wet", 0.9),
            ],
            ("Replace", 0.0),
        ),
        &expect![[r#"
            drums — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out  -30.0 pk  -42.9 rms  |    -----========-====--==-=====+=======|
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out  -28.5 pk  -42.7 rms  |    :---==-==-==-===--======-=++===+====|
              hash 3ee5b02214aefe4a
            knob moved — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out  -31.0 pk  -42.6 rms  |====--===-========--====--==-=========-=|
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out  -31.1 pk  -42.9 rms  |======----=-=====-========-==-=====-=-==|
              hash d77b5a318deb7b66
            cleared — 0.50s at 48000Hz in blocks of 1
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out  -49.1 pk  -53.8 rms  |........................................|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out  -49.1 pk  -53.8 rms  |........................................|
              hash 0bce648e8c32f3c2
            after — 0.50s at 48000Hz in blocks of 1
              L in    zero pk   zero rms  |                                        |
              L out   -inf pk   -inf rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out   -inf pk   -inf rms  |                                        |
              hash eae3c693c8cd672d
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn baxandall2() {
    let k = Kind::Baxandall2;
    check(
        &defaults(k),
        &expect![[r#"
        quiet — 0.50s at 44100Hz in blocks of 64
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -28.9 pk  -32.8 rms  |++++++++++++++++++++++++++++++++++++++++|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -28.8 pk  -32.6 rms  |++++++++++++++++++++++++++++++++++++++++|
          hash 9de8b0130119500e
        hot — 0.50s at 44100Hz in blocks of 128
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   11.1 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   11.2 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash d940217681770539
        after — 1.50s at 44100Hz in blocks of 333
          L in    zero pk   zero rms  |                                        |
          L out  -26.3 pk  -70.2 rms  |+                                       |
          R in    zero pk   zero rms  |                                        |
          R out  -21.7 pk  -64.0 rms  |*                                       |
          hash 5f19d9ca8cefc2a9
    "#]],
    );
    check(
        &all_zero(k),
        &expect![[r#"
        quiet — 0.50s at 48000Hz in blocks of 1
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -46.9 pk  -50.7 rms  |::::::::::::::::::::::::::::::::::::::::|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -47.0 pk  -50.8 rms  |::::::::::::::::::::::::::::::::::::::::|
          hash d0b66b784c945195
        hot — 0.50s at 48000Hz in blocks of 333
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   -7.0 pk  -12.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   -6.8 pk  -11.6 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
          hash 7cff800ca078799c
        after — 0.50s at 48000Hz in blocks of 64
          L in    zero pk   zero rms  |                                        |
          L out  -20.6 pk  -46.2 rms  |*                                       |
          R in    zero pk   zero rms  |                                        |
          R out  -24.1 pk  -48.6 rms  |+                                       |
          hash 65601f870667f9f8
    "#]],
    );
    check(
        &all_one(k),
        &expect![[r#"
        quiet — 0.50s at 96000Hz in blocks of 128
          L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          L out  -17.2 pk  -27.2 rms  |#***************************************|
          R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
          R out  -21.1 pk  -31.0 rms  |****************************************|
          hash 55848e5228c9d458
        hot — 0.50s at 96000Hz in blocks of 64
          L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          L out   30.4 pk   23.6 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          R out   28.7 pk   20.8 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
          hash 8456f2d2c730f0e2
        after — 1.00s at 96000Hz in blocks of 1
          L in    zero pk   zero rms  |                                        |
          L out   26.4 pk   -3.0 rms  |@.                                      |
          R in    zero pk   zero rms  |                                        |
          R out   23.7 pk   -5.5 rms  |@                                       |
          hash e15627c6ea12d9c2
    "#]],
    );
    check(
        &mixed(
            k,
            44100.0,
            333,
            &[("Treble", 0.8), ("Bass", 0.3)],
            ("Treble", 0.2),
        ),
        &expect![[r#"
            drums — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out    9.9 pk  -10.3 rms  |@@@@@%##*%%%#**++=-@@@@@%%##*#%%#**++=--|
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out    9.6 pk  -10.2 rms  |@@@@@%##*#%##*+++=-@@@@@@%%#*%%%#**+==--|
              hash e3f274705fc31aab
            knob moved — 1.00s at 44100Hz in blocks of 333
              L in    -3.2 pk  -19.9 rms  |@%%##*++=**++==-::.@@%%##*++=**++=--::. |
              L out   -8.5 pk  -27.6 rms  |%##**+==-=+==--:.. %%##**+==-++==--:..  |
              R in    -3.1 pk  -19.9 rms  |@%%##*++=+*++=--::.@@%%##*++=**++==-::..|
              R out   -8.9 pk  -27.6 rms  |%##**+==-=+==-::.. #%##**+==-++==--:..  |
              hash bb7d0c1b0eb34a0d
            cleared — 0.50s at 44100Hz in blocks of 333
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out    3.4 pk   -1.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out    4.1 pk   -1.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 4d2044e8fea07e6d
            after — 0.50s at 44100Hz in blocks of 333
              L in    zero pk   zero rms  |                                        |
              L out  -11.6 pk  -42.2 rms  |%                                       |
              R in    zero pk   zero rms  |                                        |
              R out   -9.3 pk  -40.0 rms  |%                                       |
              hash 8f16f312a867fb2d
        "#]],
    );
    check(
        &mixed(
            k,
            96000.0,
            1,
            &[("Treble", 0.35), ("Bass", 0.9)],
            ("Bass", 0.5),
        ),
        &expect![[r#"
            drums — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out    0.5 pk  -20.9 rms  |@@%##**+==*++==--:..@@%##**+==**+==-:::.|
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -2.0 pk  -21.8 rms  |@@%##**++=#*+===:::.@@%##**+==**+==--:..|
              hash 20c9b031d4c369b1
            knob moved — 1.00s at 96000Hz in blocks of 1
              L in    -3.0 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              L out   -6.7 pk  -25.8 rms  |%%##**+==-++==-::.  %%#**++=--++==-::.. |
              R in    -3.1 pk  -19.9 rms  |@%%##*++==*++==-::..@%%##*++==*++==-::..|
              R out   -6.6 pk  -25.8 rms  |%%#**++==-++==-::.. %%#**++==-++==-::.  |
              hash 081b10982ab9a4a0
            cleared — 0.50s at 48000Hz in blocks of 1
              L in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   10.8 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    10.9 pk    6.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   10.9 pk    6.3 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash a9546fb69f95503d
            after — 0.50s at 48000Hz in blocks of 1
              L in    zero pk   zero rms  |                                        |
              L out  -17.2 pk  -51.0 rms  |#                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -16.3 pk  -47.5 rms  |#                                       |
              hash 844bfcad9c335e7a
        "#]],
    );
}

/// Knob values outside 0..1 reach the original pinned, as its
/// `setChunk` pins them, and the port holds them the same way.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn out_of_range_knobs_are_pinned() {
    check(
        &Program::new(Kind::Density2, 44100.0)
            .set("Density", -0.3)
            .set("Output", 1.4)
            .play("pinned", &quiet(), 128),
        &expect![[r#"
            pinned — 0.50s at 44100Hz in blocks of 128
              L in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
              L out  -52.5 pk  -61.6 rms  |........................................|
              R in   -29.2 pk  -33.0 rms  |++++++++++++++++++++++++++++++++++++++++|
              R out  -52.4 pk  -61.5 rms  |........................................|
              hash 255a95d19b46c5f4
        "#]],
    );
}

/// How many samples the gold run compared, once every scenario has run.
/// Run alone, after the rest: `cargo test --features gold --test it --
/// --ignored gold::gold_tally --nocapture` after the full run reports nothing,
/// so it reruns the scenarios itself.
#[cfg(feature = "gold")]
#[test]
#[ignore = "a report, not a check: run it on its own"]
#[expect(clippy::print_stdout, reason = "the report is the point")]
fn gold_tally() {
    use std::sync::atomic::Ordering;
    let scenarios: [(&str, fn()); 7] = [
        ("Console7Channel", console7_channel),
        ("Console7Buss", console7_buss),
        ("ToTape8", to_tape8),
        ("Density2", density2),
        ("Galactic3", galactic3),
        ("Baxandall2", baxandall2),
        ("Density2, pinned knobs", out_of_range_knobs_are_pinned),
    ];
    for (name, run) in scenarios {
        let before = cpp::COMPARED.load(Ordering::Relaxed);
        run();
        println!(
            "{name}: {} samples bit for bit, both channels counted",
            cpp::COMPARED.load(Ordering::Relaxed) - before
        );
    }
}
