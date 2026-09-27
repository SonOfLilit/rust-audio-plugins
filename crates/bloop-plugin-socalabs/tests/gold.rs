//! Gold tests: each ported processor against the C++ it was ported from.
//!
//! A scenario is a [`Program`] — which effect, at what sample rate, which
//! knobs set to what, and what audio it hears in what size of blocks. It
//! runs through the Rust port, and the summary of what came out is the
//! snapshot: how loud each passage went in and came out, and a hash of
//! every output sample.
//!
//! With `--features gold` the same program also runs through the originals
//! (`tests/cpp_harness`, built from the pinned `vendor/` submodules against the
//! real JUCE), and the two outputs must agree bit for bit. The snapshot is
//! not checked then: gold switches the Rust side to the platform's maths
//! library (src/math.rs), so its hash is not the one that ships.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_socalabs::{Effect, Kind};
use expect_test::{expect, Expect};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

enum Step {
    Set(usize, f32),
    Tempo(f64),
    Play {
        label: &'static str,
        left: Vec<f32>,
        right: Vec<f32>,
        block: usize,
    },
}

struct Program {
    kind: Kind,
    sample_rate: f64,
    steps: Vec<Step>,
}

impl Program {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            sample_rate: 44100.0,
            steps: Vec::new(),
        }
    }

    fn at(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Knob `index`, in the order the original's constructor adds them, to
    /// `value` in the knob's own units.
    fn set(mut self, index: usize, value: f32) -> Self {
        self.steps.push(Step::Set(index, value));
        self
    }

    /// What the host's transport says the tempo is.
    fn tempo(mut self, bpm: f64) -> Self {
        self.steps.push(Step::Tempo(bpm));
        self
    }

    /// `signal` through the effect in blocks of `block` samples.
    fn play(mut self, label: &'static str, signal: &Signal, block: usize) -> Self {
        self.steps.push(Step::Play {
            label,
            left: signal.left.clone(),
            right: signal.right.clone(),
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

/// A repeatable noise source (xorshift32), so no platform's RNG is in it.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (f64::from(self.0) / f64::from(u32::MAX) * 2.0 - 1.0) as f32
    }
}

fn samples(seconds: f64) -> usize {
    (seconds * 44100.0) as usize
}

/// A sine, the same in both channels.
fn tone(hz: f64, db: f64, seconds: f64) -> Signal {
    let amp = libm::pow(10.0, db / 20.0);
    let left: Vec<f32> = (0..samples(seconds))
        .map(|i| (amp * libm::sin(2.0 * core::f64::consts::PI * hz * i as f64 / 44100.0)) as f32)
        .collect();
    Signal {
        right: left.clone(),
        left,
    }
}

/// Drum-like hits: a burst of noise decaying over `decay` seconds, every
/// `every` seconds, alternating loud and quiet by `spread` dB around `db`,
/// with the right channel a different noise from the left.
fn hits(db: f64, spread: f64, every: f64, decay: f64, count: usize) -> Signal {
    let period = samples(every);
    let mut l = Noise(0x1234_5678);
    let mut r = Noise(0x9abc_def0);
    let mut left = Vec::with_capacity(period * count);
    let mut right = Vec::with_capacity(period * count);
    for hit in 0..count {
        let level = if hit % 2 == 0 { db } else { db - spread };
        let amp = libm::pow(10.0, level / 20.0);
        for i in 0..period {
            let env = amp * libm::exp(-(i as f64) / (decay * 44100.0));
            left.push((env * f64::from(l.next())) as f32);
            right.push((env * f64::from(r.next())) as f32);
        }
    }
    Signal { left, right }
}

fn silence(seconds: f64) -> Signal {
    Signal {
        left: vec![0.0; samples(seconds)],
        right: vec![0.0; samples(seconds)],
    }
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// What one play step produced.
struct Played {
    label: &'static str,
    block: usize,
    input: (Vec<f32>, Vec<f32>),
    output: (Vec<f32>, Vec<f32>),
}

fn run_rust(program: &Program) -> Vec<Played> {
    let mut effect = Effect::new(program.kind);
    effect.reset(program.sample_rate as f32, 512);
    let events = bloop_events::EventBuffer::new();
    let mut played = Vec::new();
    let mut tempo = 120.0;
    for step in &program.steps {
        match step {
            Step::Set(index, value) => effect.set_param(*index, *value),
            Step::Tempo(bpm) => tempo = *bpm,
            Step::Play {
                label,
                left,
                right,
                block,
            } => {
                let mut out_l = left.clone();
                let mut out_r = right.clone();
                for (l, r) in out_l.chunks_mut(*block).zip(out_r.chunks_mut(*block)) {
                    let mut outputs: [&mut [f32]; 2] = [l, r];
                    let mut context = ProcessContext {
                        outputs: &mut outputs,
                        input_events: &events,
                        sample_rate: program.sample_rate as f32,
                        tempo,
                        is_playing: true,
                    };
                    effect.process(&mut context);
                }
                played.push(Played {
                    label,
                    block: *block,
                    input: (left.clone(), right.clone()),
                    output: (out_l, out_r),
                });
            }
        }
    }
    played
}

fn level_db(samples: &[f32]) -> (f64, f64) {
    let peak = samples
        .iter()
        .fold(0.0f64, |m, &s| m.max(f64::from(s.abs())));
    let rms = (samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len().max(1) as f64)
        .sqrt();
    let db = |x: f64| {
        if x > 0.0 {
            20.0 * libm::log10(x)
        } else {
            -999.0
        }
    };
    (db(peak), db(rms))
}

fn fnv(played: &[Played]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for p in played {
        for s in p.output.0.iter().chain(&p.output.1) {
            for byte in s.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    hash
}

#[cfg_attr(
    feature = "gold",
    expect(
        dead_code,
        reason = "gold compares against C++ instead of the snapshot"
    )
)]
fn summary(played: &[Played]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for p in played {
        let both = |(l, r): &(Vec<f32>, Vec<f32>)| {
            let mut all = l.clone();
            all.extend_from_slice(r);
            level_db(&all)
        };
        let (in_peak, in_rms) = both(&p.input);
        let (out_peak, out_rms) = both(&p.output);
        let _ = writeln!(
            out,
            "{:<14} blocks of {:<4}  in peak {in_peak:>6.1} rms {in_rms:>6.1}  →  out peak {out_peak:>6.1} rms {out_rms:>6.1} dB",
            p.label, p.block
        );
    }
    let _ = write!(out, "hash {:016x}", fnv(played));
    out
}

fn check(program: &Program, expect: &Expect) {
    let played = run_rust(program);
    #[cfg(feature = "gold")]
    {
        let cpp = cpp::run(program);
        compare(program, &played, &cpp);
        let _ = expect;
    }
    #[cfg(not(feature = "gold"))]
    expect.assert_eq(&summary(&played));
}

/// The first sample where the two sides disagree, with enough around it
/// to see which part of the processor is at fault.
#[cfg(feature = "gold")]
fn compare(program: &Program, rust: &[Played], cpp: &[(Vec<f32>, Vec<f32>)]) {
    assert_eq!(rust.len(), cpp.len(), "{:?}: step count", program.kind);
    for (p, (cl, cr)) in rust.iter().zip(cpp) {
        for (channel, (ours, theirs)) in [(&p.output.0, cl), (&p.output.1, cr)]
            .into_iter()
            .enumerate()
        {
            let first = ours
                .iter()
                .zip(theirs.iter())
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                first.is_none(),
                "{}",
                first.map_or_else(String::new, |i| format!(
                    "{:?}, '{}', channel {channel}: sample {i} of {} differs — rust {:e} ({:#010x}), C++ {:e} ({:#010x}); input {:e}",
                    program.kind,
                    p.label,
                    ours.len(),
                    ours[i],
                    ours[i].to_bits(),
                    theirs[i],
                    theirs[i].to_bits(),
                    if channel == 0 { p.input.0[i] } else { p.input.1[i] },
                ))
            );
            assert_eq!(
                ours.len(),
                theirs.len(),
                "{:?}, '{}': length",
                program.kind,
                p.label
            );
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
    use super::{Program, Step};
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::OnceLock;
    use std::time::SystemTime;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn vendor(name: &str) -> PathBuf {
        let path = root().join("../../../vendor").join(name);
        assert!(
            path.join(".git").exists(),
            "vendor/{name} is not checked out. The gold tests compile the originals from it:\n  \
             git -c submodule.vendor/{name}.update=checkout submodule update --init vendor/{name}"
        );
        path
    }

    fn mtime(path: &Path) -> SystemTime {
        path.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    }

    /// The JUCE modules: built once, since the pin moves only by hand.
    const JUCE_UNITS: [&str; 8] = [
        "juce_core.cpp",
        "juce_core_CompilationTime.cpp",
        "juce_core_zlib.c",
        "juce_events.cpp",
        "juce_data_structures.cpp",
        "juce_audio_basics.cpp",
        "juce_audio_formats.cpp",
        "juce_dsp.cpp",
    ];
    /// The harness itself: rebuilt whenever anything in `cpp_harness` moves.
    const HARNESS_UNITS: [&str; 2] = ["gin_dsp.cpp", "main.cpp"];

    fn build() -> PathBuf {
        let harness = root().join("tests/cpp_harness");
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("socalabs-gold");
        std::fs::create_dir_all(&out).expect("create the harness build directory");
        let juce = vendor("juce").join("modules");
        let gin = vendor("gin").join("modules");
        vendor("slplugins");

        let newest_harness_file = std::fs::read_dir(&harness)
            .expect("read tests/cpp_harness")
            .filter_map(Result::ok)
            .map(|e| mtime(&e.path()))
            .max()
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let compile = |unit: &str| -> Option<std::process::Child> {
            let object = out.join(format!("{unit}.o"));
            let stale = !object.exists()
                || (HARNESS_UNITS.contains(&unit) && mtime(&object) < newest_harness_file);
            if !stale {
                return None;
            }
            let c = Path::new(unit).extension().is_some_and(|e| e == "c");
            let mut cmd = Command::new(if c { "gcc" } else { "g++" });
            if !c {
                cmd.arg("-std=c++20");
            }
            cmd.args(["-O2", "-DNDEBUG", "-include"])
                .arg(harness.join("config.h"))
                .arg("-I")
                .arg(&juce)
                .arg("-I")
                .arg(&gin)
                .arg("-c")
                .arg(harness.join(unit))
                .arg("-o")
                .arg(&object);
            Some(
                cmd.spawn()
                    .expect("run g++ (the gold tests need it installed)"),
            )
        };

        let children: Vec<(&str, std::process::Child)> = JUCE_UNITS
            .iter()
            .chain(&HARNESS_UNITS)
            .filter_map(|unit| compile(unit).map(|child| (*unit, child)))
            .collect();
        let relink = !children.is_empty();
        for (unit, mut child) in children {
            let status = child.wait().expect("wait for g++");
            assert!(
                status.success(),
                "compiling {unit} for the gold harness failed"
            );
        }

        let binary = out.join("harness");
        if relink || !binary.exists() {
            let status = Command::new("g++")
                .args(
                    JUCE_UNITS
                        .iter()
                        .chain(&HARNESS_UNITS)
                        .map(|u| out.join(format!("{u}.o"))),
                )
                .arg("-o")
                .arg(&binary)
                .args(["-lpthread", "-ldl"])
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

    /// The program through the originals: one (left, right) per play step.
    pub fn run(program: &Program) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut input = Vec::new();
        let _ = writeln!(input, "new {} {}", program.kind.name(), program.sample_rate);
        let mut blocks_per_step = Vec::new();
        for step in &program.steps {
            match step {
                Step::Set(index, value) => {
                    let _ = writeln!(input, "set {index} {value:?}");
                }
                Step::Tempo(bpm) => {
                    let _ = writeln!(input, "tempo {bpm:?}");
                }
                Step::Play {
                    left, right, block, ..
                } => {
                    let mut sizes = Vec::new();
                    for (l, r) in left.chunks(*block).zip(right.chunks(*block)) {
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

        let mut child = Command::new(binary())
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
                (left, right)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Loud and quiet hits, which every dynamics processor has something to say
/// about: the loud ones cross a -30 dB threshold, the quiet ones do not.
fn drums() -> Signal {
    hits(-6.0, 30.0, 0.25, 0.05, 8)
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn compressor() {
    check(
        &Program::new(Kind::Compressor).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults       blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -7.1 rms  -34.8 dB
            hash 2d1ac18111447c91"#]],
    );
    check(
        &Program::new(Kind::Compressor)
            .set(0, 30.0)
            .set(1, 400.0)
            .set(2, 30.0)
            .set(3, -40.0)
            .set(4, 0.0)
            .set(6, 12.0)
            .play("hard knee", &drums(), 256),
        &expect![[r#"
            hard knee      blocks of 256   in peak   -6.1 rms  -23.8  →  out peak   -2.2 rms  -31.7 dB
            hash 325a7521b05adc5f"#]],
    );
    check(
        &Program::new(Kind::Compressor)
            .play("settled", &drums(), 512)
            .set(3, -10.0)
            .set(4, 60.0)
            .set(5, 20.0)
            .play("knobs moving", &drums(), 333)
            .play("one at a time", &tone(220.0, -3.0, 0.1), 1),
        &expect![[r#"
            settled        blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -7.1 rms  -34.8 dB
            knobs moving   blocks of 333   in peak   -6.1 rms  -23.8  →  out peak    9.2 rms  -17.4 dB
            one at a time  blocks of 1     in peak   -3.0 rms   -6.0  →  out peak   -0.3 rms   -6.3 dB
            hash 43b8acdae6907147"#]],
    );
    check(
        &Program::new(Kind::Compressor)
            .at(48000.0)
            .set(2, 1.0)
            .play("ratio 1", &drums(), 512)
            .play("silence", &silence(0.1), 64),
        &expect![[r#"
            ratio 1        blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -24.7 dB
            silence        blocks of 64    in peak -999.0 rms -999.0  →  out peak -999.0 rms -999.0 dB
            hash 504d90f36228a032"#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn limiter() {
    check(
        &Program::new(Kind::Limiter).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults       blocks of 512   in peak   -6.1 rms  -23.8  →  out peak  -24.7 rms  -39.0 dB
            hash b2c6868505ec3d61"#]],
    );
    check(
        &Program::new(Kind::Limiter)
            .set(0, 5.0)
            .set(1, 500.0)
            .set(2, -3.0)
            .set(3, 30.0)
            .set(4, -6.0)
            .play("slammed", &drums(), 128)
            .set(2, -60.0)
            .play("threshold falling", &tone(100.0, 0.0, 0.2), 1000),
        &expect![[r#"
            slammed        blocks of 128   in peak   -6.1 rms  -23.8  →  out peak   13.6 rms  -20.0 dB
            threshold falling blocks of 1000  in peak   -0.0 rms   -3.0  →  out peak    3.4 rms  -17.5 dB
            hash d80caf2ee9ad6ed8"#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn gate() {
    check(
        &Program::new(Kind::Gate).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults       blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -23.9 dB
            hash 4940cf60191d9d98"#]],
    );
    check(
        &Program::new(Kind::Gate)
            .set(0, 2.0)
            .set(1, 100.0)
            .set(2, 250.0)
            .set(3, -20.0)
            .set(4, 12.0)
            .play("held open", &drums(), 441)
            .set(3, -100.0)
            .play("wide open", &drums(), 512),
        &expect![[r#"
            held open      blocks of 441   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -24.0 dB
            wide open      blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -23.9 dB
            hash b93fb9de405f2e75"#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn expander() {
    check(
        &Program::new(Kind::Expander).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults       blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -23.8 dB
            hash acab33a637a495dd"#]],
    );
    check(
        &Program::new(Kind::Expander)
            .set(2, 4.0)
            .set(3, -20.0)
            .set(4, 0.0)
            .play("ratio 4", &drums(), 512)
            .set(4, 30.0)
            .set(0, 200.0)
            .set(1, 2000.0)
            .play("slow, soft", &drums(), 100),
        &expect![[r#"
            ratio 4        blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -24.1 dB
            slow, soft     blocks of 100   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -28.0 dB
            hash b2e95013f99e554c"#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn delay() {
    // Mix defaults to 0: fully dry, so the output is the input.
    check(
        &Program::new(Kind::Delay).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults       blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -6.1 rms  -23.8 dB
            hash acab33a637a495dd"#]],
    );
    check(
        &Program::new(Kind::Delay)
            .set(1, 0.25)
            .set(3, -6.0)
            .set(5, 50.0)
            .play("echoes", &drums(), 512)
            .play("tail", &silence(1.0), 512),
        &expect![[r#"
            echoes         blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -5.3 rms  -23.0 dB
            tail           blocks of 512   in peak -999.0 rms -999.0  →  out peak  -13.3 rms  -34.3 dB
            hash 33c762976e00938d"#]],
    );
    check(
        &Program::new(Kind::Delay)
            .set(1, 0.3)
            .set(3, -100.0)
            .set(4, -3.0)
            .set(5, 100.0)
            .play("ping-pong", &hits(-6.0, 0.0, 1.0, 0.02, 1), 256)
            .play("tail", &silence(1.0), 256),
        &expect![[r#"
            ping-pong      blocks of 256   in peak   -6.3 rms  -30.8  →  out peak   -6.3 rms  -26.4 dB
            tail           blocks of 256   in peak -999.0 rms -999.0  →  out peak  -15.3 rms  -37.3 dB
            hash 844e840a701366ef"#]],
    );
    check(
        &Program::new(Kind::Delay)
            .tempo(90.0)
            .set(0, 1.0)
            .set(2, 10.0)
            .set(5, 40.0)
            .play("1/8 at 90", &drums(), 256)
            .tempo(140.0)
            .play("1/8 at 140", &drums(), 256)
            .set(2, 32.0)
            .play("32 bars", &drums(), 256),
        &expect![[r#"
            1/8 at 90      blocks of 256   in peak   -6.1 rms  -23.8  →  out peak   -6.2 rms  -23.5 dB
            1/8 at 140     blocks of 256   in peak   -6.1 rms  -23.8  →  out peak   -7.8 rms  -23.7 dB
            32 bars        blocks of 256   in peak   -6.1 rms  -23.8  →  out peak   -3.3 rms  -24.5 dB
            hash 1031df9d346bcbad"#]],
    );
    check(
        &Program::new(Kind::Delay)
            .at(48000.0)
            .set(1, 0.05)
            .set(5, 30.0)
            .play("short", &tone(440.0, -6.0, 0.3), 333)
            .set(1, 0.5)
            .play("time sliding", &tone(440.0, -6.0, 0.3), 1)
            .set(1, 7.0)
            .play("past the line", &drums(), 512),
        &expect![[r#"
            short          blocks of 333   in peak   -6.0 rms   -9.0  →  out peak   -2.5 rms   -8.4 dB
            time sliding   blocks of 1     in peak   -6.0 rms   -9.0  →  out peak   -2.4 rms   -9.8 dB
            past the line  blocks of 512   in peak   -6.1 rms  -23.8  →  out peak   -3.7 rms  -24.6 dB
            hash d64f51e03d32153b"#]],
    );
}
