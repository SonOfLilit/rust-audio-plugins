//! Gold tests: each ported processor against the C++ it was ported from.
//!
//! A scenario is a [`Program`] — which effect, at what sample rate, which
//! knobs set to what, and what audio it hears in what size of blocks. It
//! runs through the Rust port, and a summary of what came out is the
//! snapshot: for each passage, each channel's level in and out and a strip
//! of its envelope over time, and a hash of every output sample.
//!
//! With `--features gold` the same program also runs through the originals
//! — slPlugins' own `PluginProcessor.cpp` files and Gin's DSP, compiled
//! unmodified against the real JUCE from the pinned `vendor/` submodules
//! (`tests/cpp_harness`) — and the two outputs must agree bit for bit. The
//! snapshot's hashes are not checked then, since gold switches the Rust
//! side to the platform's maths library (src/math.rs) and the hash that
//! ships is libm's; everything else in it is.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a scenario's length is its snapshot's: five lines a passage"
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
    /// `Plugin::reset`: prepared again, at this rate, mid-program.
    Reset(f64),
    Play {
        label: &'static str,
        signal: Signal,
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

    /// The rate the program is running at from here on.
    fn rate(&self) -> f64 {
        self.steps
            .iter()
            .rev()
            .find_map(|s| match s {
                Step::Reset(rate) => Some(*rate),
                _ => None,
            })
            .unwrap_or(self.sample_rate)
    }

    /// The knob called `name` to `value`, in the knob's own units. Knobs
    /// are found by the names the Plug-in reports, so a misnamed or
    /// renumbered knob fails here.
    fn set(mut self, name: &str, value: f32) -> Self {
        let effect = Effect::new(self.kind);
        let matches: Vec<usize> = (0..effect.param_count())
            .filter(|&i| effect.param_info(i).name == name)
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

    /// What the host's transport says the tempo is.
    fn tempo(mut self, bpm: f64) -> Self {
        self.steps.push(Step::Tempo(bpm));
        self
    }

    fn reset(mut self, sample_rate: f64) -> Self {
        self.steps.push(Step::Reset(sample_rate));
        self
    }

    /// `signal`, made at the program's current rate, through the effect in
    /// blocks of `block` samples.
    fn play(mut self, label: &'static str, signal: &Source, block: usize) -> Self {
        let signal = (signal.0)(self.rate());
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

impl Source {
    fn new(f: impl Fn(f64) -> Signal + 'static) -> Self {
        Self(Box::new(f))
    }
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

fn gain(db: f64) -> f64 {
    libm::pow(10.0, db / 20.0)
}

fn count(seconds: f64, rate: f64) -> usize {
    (seconds * rate) as usize
}

fn both(left: Vec<f32>) -> Signal {
    Signal {
        right: left.clone(),
        left,
    }
}

/// A sine, the same in both channels.
fn tone(hz: f64, db: f64, seconds: f64) -> Source {
    Source::new(move |rate| {
        both(
            (0..count(seconds, rate))
                .map(|i| {
                    (gain(db) * libm::sin(2.0 * core::f64::consts::PI * hz * i as f64 / rate))
                        as f32
                })
                .collect(),
        )
    })
}

/// `source` with its right channel silenced.
fn left_only(source: Source) -> Source {
    Source::new(move |rate| {
        let s = (source.0)(rate);
        Signal {
            right: vec![0.0; s.left.len()],
            left: s.left,
        }
    })
}

/// `source` with its left channel silenced.
fn right_only(source: Source) -> Source {
    Source::new(move |rate| {
        let s = (source.0)(rate);
        Signal {
            left: vec![0.0; s.right.len()],
            right: s.right,
        }
    })
}

/// A sine whose level climbs steadily from `from_db` to `to_db`.
fn ramp(hz: f64, from_db: f64, to_db: f64, seconds: f64) -> Source {
    Source::new(move |rate| {
        let n = count(seconds, rate);
        both(
            (0..n)
                .map(|i| {
                    let level = from_db + (to_db - from_db) * i as f64 / n as f64;
                    (gain(level) * libm::sin(2.0 * core::f64::consts::PI * hz * i as f64 / rate))
                        as f32
                })
                .collect(),
        )
    })
}

/// A sweep from `from` to `to` Hz, exponential, in both channels.
fn sweep(from: f64, to: f64, db: f64, seconds: f64) -> Source {
    Source::new(move |rate| {
        let n = count(seconds, rate);
        let k = libm::log(to / from);
        let mut phase = 0.0f64;
        both(
            (0..n)
                .map(|i| {
                    let hz = from * libm::exp(k * i as f64 / n as f64);
                    phase += 2.0 * core::f64::consts::PI * hz / rate;
                    (gain(db) * libm::sin(phase)) as f32
                })
                .collect(),
        )
    })
}

/// Drum-like hits: a burst of noise decaying over `decay` seconds, every
/// `every` seconds, alternating loud and quiet by `spread` dB around `db`,
/// with the right channel a different noise from the left.
fn hits(db: f64, spread: f64, every: f64, decay: f64, n: usize) -> Source {
    Source::new(move |rate| {
        let period = count(every, rate);
        let mut l = Noise(0x1234_5678);
        let mut r = Noise(0x9abc_def0);
        let mut left = Vec::with_capacity(period * n);
        let mut right = Vec::with_capacity(period * n);
        for hit in 0..n {
            let level = if hit % 2 == 0 { db } else { db - spread };
            for i in 0..period {
                let env = gain(level) * libm::exp(-(i as f64) / (decay * rate));
                left.push((env * f64::from(l.next())) as f32);
                right.push((env * f64::from(r.next())) as f32);
            }
        }
        Signal { left, right }
    })
}

/// One sample at `db`, then silence.
fn impulse(db: f64, seconds: f64) -> Source {
    Source::new(move |rate| {
        let mut left = vec![0.0; count(seconds, rate).max(1)];
        left[0] = gain(db) as f32;
        both(left)
    })
}

/// A constant.
fn dc(value: f32, seconds: f64) -> Source {
    Source::new(move |rate| both(vec![value; count(seconds, rate)]))
}

fn silence(seconds: f64) -> Source {
    dc(0.0, seconds)
}

/// Loud and quiet hits, which every dynamics processor has something to say
/// about: the loud ones cross a -30 dB threshold, the quiet ones do not.
fn drums() -> Source {
    hits(-6.0, 30.0, 0.25, 0.05, 8)
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// What one play step produced.
struct Played {
    label: &'static str,
    block: usize,
    rate: f64,
    input: Signal,
    output: Signal,
}

fn run_rust(program: &Program) -> Vec<Played> {
    let mut effect = Effect::new(program.kind);
    let mut rate = program.sample_rate;
    effect.reset(rate as f32, 512);
    let events = bloop_events::EventBuffer::new();
    let mut played = Vec::new();
    let mut tempo = 120.0;
    for step in &program.steps {
        match step {
            Step::Set(index, value) => effect.set_param(*index, *value),
            Step::Tempo(bpm) => tempo = *bpm,
            Step::Reset(new_rate) => {
                rate = *new_rate;
                effect.reset(rate as f32, 512);
            }
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
                    let mut outputs: [&mut [f32]; 2] = [l, r];
                    let mut context = ProcessContext {
                        outputs: &mut outputs,
                        input_events: &events,
                        sample_rate: rate as f32,
                        tempo,
                        is_playing: true,
                    };
                    effect.process(&mut context);
                }
                played.push(Played {
                    label,
                    block: *block,
                    rate,
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
        if d < -120.0 {
            "  -inf".to_string()
        } else {
            format!("{d:>6.1}")
        }
    };
    format!("{} pk {} rms", show(peak), show(rms))
}

/// The envelope over time: forty columns, each the peak of its stretch in
/// 6 dB steps, from blank (below -60 dB) to `@` (above -6 dB).
fn strip(samples: &[f32]) -> String {
    const RAMP: &[u8] = b" .:-=+*#%@";
    const COLUMNS: usize = 40;
    let per = samples.len().div_ceil(COLUMNS).max(1);
    samples
        .chunks(per)
        .map(|chunk| {
            let peak = chunk.iter().fold(0.0f64, |m, &s| m.max(f64::from(s.abs())));
            let level = ((db(peak) + 60.0) / 6.0).floor() + 1.0;
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
        let seconds = p.input.left.len() as f64 / p.rate;
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

/// The first sample where the two sides disagree.
#[cfg(feature = "gold")]
fn compare(program: &Program, rust: &[Played], cpp: &[Signal]) {
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
                .position(|(a, b)| a.to_bits() != b.to_bits());
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
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
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

    /// The revision a vendor checkout is at, so a moved pin builds afresh.
    fn revision(path: &Path) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(path)
            .args(["rev-parse", "--short=12", "HEAD"])
            .output()
            .expect("run git to read a vendor revision");
        assert!(
            out.status.success(),
            "git rev-parse failed in {}",
            path.display()
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn mtime(path: &Path) -> SystemTime {
        path.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    }

    fn newest(dir: &Path) -> SystemTime {
        std::fs::read_dir(dir)
            .expect("read a harness directory")
            .filter_map(Result::ok)
            .map(|e| {
                if e.path().is_dir() {
                    newest(&e.path())
                } else {
                    mtime(&e.path())
                }
            })
            .max()
            .unwrap_or(SystemTime::UNIX_EPOCH)
    }

    /// JUCE's modules, as JUCE ships them: built once per pin.
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

    /// The slPlugins processors the harness links, from `slplugins.inc`.
    fn slplugins(harness: &Path) -> Vec<String> {
        std::fs::read_to_string(harness.join("slplugins.inc"))
            .expect("read slplugins.inc")
            .lines()
            .filter_map(|l| l.trim().strip_prefix("SLPLUGIN(")?.strip_suffix(')'))
            .map(str::to_string)
            .collect()
    }

    /// A copy of one plugin's `Source` directory, laid out as slPlugins
    /// lays it out so its relative includes still resolve, with the editor
    /// swapped for a stub (the real one is all GUI), and a unit that
    /// compiles it inside its own namespace — three of the processors are
    /// called plain `AudioProcessor`.
    fn slplugins_unit(out: &Path, sl: &Path, name: &str) -> PathBuf {
        let from = sl.join("plugins").join(name).join("Source");
        let to = out.join("slplugins/plugins").join(name).join("Source");
        std::fs::create_dir_all(&to).expect("create a plugin source copy");
        for file in ["PluginProcessor.cpp", "PluginProcessor.h"] {
            std::fs::copy(from.join(file), to.join(file)).expect("copy a plugin source");
        }
        let cpp = std::fs::read_to_string(from.join("PluginProcessor.cpp"))
            .expect("read a plugin source");
        let editor = cpp
            .split("return new ")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .expect("find the editor class in createEditor");
        write(
            &to.join("PluginEditor.h"),
            &format!(
                "#pragma once\n#include \"PluginProcessor.h\"\n\
                 class {editor} : public juce::AudioProcessorEditor\n{{\npublic:\n    \
                 template <class P> explicit {editor} (P&) {{}}\n}};\n"
            ),
        );
        let unit = to.join("unit.cpp");
        write(
            &unit,
            &format!(
                "#include <JuceHeader.h>\n\
                 #include \"../../../3rdparty/crossfeed/crossfeed.h\"\n\
                 namespace slplugins_{name}\n{{\n#include \"PluginProcessor.cpp\"\n}}\n\
                 juce::AudioProcessor* create_{name}() {{ return slplugins_{name}::createPluginFilter(); }}\n"
            ),
        );
        unit
    }

    /// Write `contents` unless the file already holds them, so an unchanged
    /// file keeps its mtime and its object stays built.
    fn write(path: &Path, contents: &str) {
        if std::fs::read_to_string(path).ok().as_deref() != Some(contents) {
            std::fs::write(path, contents).expect("write a harness file");
        }
    }

    fn build() -> PathBuf {
        let harness = root().join("tests/cpp_harness");
        let (juce, gin, sl) = (vendor("juce"), vendor("gin"), vendor("slplugins"));
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "socalabs-gold-{}-{}-{}",
            revision(&juce),
            revision(&gin),
            revision(&sl)
        ));
        std::fs::create_dir_all(out.join("slplugins/3rdparty/crossfeed"))
            .expect("create the harness build directory");
        for file in ["crossfeed.h", "crossfeed.c"] {
            std::fs::copy(
                sl.join("3rdparty/crossfeed").join(file),
                out.join("slplugins/3rdparty/crossfeed").join(file),
            )
            .expect("copy the crossfeed library");
        }
        let harness_changed = newest(&harness);

        let mut units: Vec<(PathBuf, bool)> = JUCE_UNITS
            .iter()
            .map(|u| (harness.join(u), false))
            .collect();
        units.push((harness.join("gin_dsp.cpp"), true));
        units.push((harness.join("main.cpp"), true));
        units.push((out.join("slplugins/3rdparty/crossfeed/crossfeed.c"), true));
        for name in slplugins(&harness) {
            units.push((slplugins_unit(&out, &sl, &name), true));
        }

        let compile = |source: &Path, tracks_harness: bool| -> (PathBuf, Option<Child>) {
            let stem = source
                .strip_prefix(&out)
                .or_else(|_| source.strip_prefix(&harness))
                .expect("a unit under the harness or the build directory")
                .to_string_lossy()
                .replace('/', "_");
            let object = out.join(format!("{stem}.o"));
            let stale = !object.exists()
                || mtime(source) > mtime(&object)
                || (tracks_harness && harness_changed > mtime(&object));
            if !stale {
                return (object, None);
            }
            let c = source.extension().is_some_and(|e| e == "c");
            let mut cmd = Command::new(if c { "gcc" } else { "g++" });
            if !c {
                cmd.arg("-std=c++20");
            }
            // No fused multiply-adds: the port never fuses, and a compiler
            // that did (clang on arm64 does by default) would differ from it
            // in the last bit for reasons that are not the port's.
            cmd.args(["-O2", "-DNDEBUG", "-ffp-contract=off", "-include"])
                .arg(harness.join("config.h"))
                .arg("-I")
                .arg(harness.join("shim"))
                .arg("-I")
                .arg(juce.join("modules"))
                .arg("-I")
                .arg(gin.join("modules"))
                .arg("-c")
                .arg(source)
                .arg("-o")
                .arg(&object);
            let child = cmd
                .spawn()
                .expect("run g++ (the gold tests need it installed)");
            (object, Some(child))
        };

        let built: Vec<(PathBuf, PathBuf, Option<Child>)> = units
            .iter()
            .map(|(source, tracks)| {
                let (object, child) = compile(source, *tracks);
                (source.clone(), object, child)
            })
            .collect();
        let relink = built.iter().any(|(_, _, child)| child.is_some());
        let mut objects = Vec::new();
        for (source, object, child) in built {
            if let Some(mut child) = child {
                let status = child.wait().expect("wait for g++");
                assert!(
                    status.success(),
                    "compiling {} for the gold harness failed",
                    source.display()
                );
            }
            objects.push(object);
        }

        let binary = out.join("harness");
        if relink || !binary.exists() {
            let status = Command::new("g++")
                .args(&objects)
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

    /// The program through the originals: one signal per passage.
    pub fn run(program: &Program) -> Vec<Signal> {
        let mut input = Vec::new();
        let _ = writeln!(
            input,
            "new {} {:?}",
            program.kind.name(),
            program.sample_rate
        );
        let mut blocks_per_step = Vec::new();
        for step in &program.steps {
            match step {
                Step::Set(index, value) => {
                    let _ = writeln!(input, "set {index} {:08x}", value.to_bits());
                }
                Step::Tempo(bpm) => {
                    let _ = writeln!(input, "tempo {bpm:?}");
                }
                Step::Reset(rate) => {
                    let _ = writeln!(input, "reset {rate:?}");
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
fn compressor() {
    check(
        &Program::new(Kind::Compressor).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -7.7 pk  -34.8 rms  |@**+==-:  @**+==-:  @**+==-:  @**+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -7.1 pk  -34.8 rms  |@**+==-:  @**+==-:  @**+==-:  @**+==-:  |
              hash 2d1ac18111447c91
        "#]],
    );
    check(
        &Program::new(Kind::Compressor)
            .set("Attack", 30.0)
            .set("Release", 400.0)
            .set("Ratio", 30.0)
            .set("Threshold", -40.0)
            .set("Knee", 0.0)
            .set("Input", -6.0)
            .set("Output", 12.0)
            .play("every knob", &drums(), 256),
        &expect![[r#"
            every knob — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -2.4 pk  -32.0 rms  |@+=-:-:.  @*+=:=:.  @*+=:=:.  @*+=:=:.  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -3.3 pk  -32.1 rms  |@+=-:-:.  @*+=:=:.  @*+=:=:.  @*+=:=:.  |
              hash 75c1276cf27a6e47
        "#]],
    );
    // A level climbing through the knee: below it the tone passes, across
    // it the gain eases in, above it the ratio holds.
    check(
        &Program::new(Kind::Compressor)
            .set("Threshold", -20.0)
            .set("Knee", 20.0)
            .set("Ratio", 10.0)
            .play("through the knee", &ramp(220.0, -50.0, 0.0, 2.0), 512),
        &expect![[r#"
            through the knee — 2.00s at 44100Hz in blocks of 512
              L in    -0.0 pk  -13.6 rms  |:-----=====+++++****#####%%%%%@@@@@@@@@@|
              L out  -16.5 pk  -24.4 rms  |:-----=====+++++*****#########%%%%%%%%%%|
              R in    -0.0 pk  -13.6 rms  |:-----=====+++++****#####%%%%%@@@@@@@@@@|
              R out  -16.5 pk  -24.4 rms  |:-----=====+++++*****#########%%%%%%%%%%|
              hash 9a3e941f2528d625
        "#]],
    );
    // Knobs set again before the last move has finished smoothing, then a
    // block at a time.
    check(
        &Program::new(Kind::Compressor)
            .play("settled", &drums(), 512)
            .set("Threshold", -10.0)
            .set("Input", 20.0)
            .play("halfway", &tone(220.0, -3.0, 0.03), 333)
            .set("Threshold", -50.0)
            .set("Input", -10.0)
            .play("turned back", &drums(), 333)
            .set("Knee", 60.0)
            .play("one at a time", &tone(220.0, -3.0, 0.1), 1),
        &expect![[r#"
            settled — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -7.7 pk  -34.8 rms  |@**+==-:  @**+==-:  @**+==-:  @**+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -7.1 pk  -34.8 rms  |@**+==-:  @**+==-:  @**+==-:  @**+==-:  |
              hash 2d1ac18111447c91
            halfway — 0.03s at 44100Hz in blocks of 333
              L in    -3.0 pk   -6.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -6.0 pk  -14.3 rms  |%###########%%#%%#%%#%%%@@%@@%@@@@@@@@@|
              R in    -3.0 pk   -6.1 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -6.0 pk  -14.3 rms  |%###########%%#%%#%%#%%%@@%@@%@@@@@@@@@|
              hash d99d7f72654e5ff5
            turned back — 2.00s at 44100Hz in blocks of 333
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.7 pk  -35.0 rms  |@=--:-.   #---:-.   *---:-.   *---:-.   |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.7 pk  -34.9 rms  |@=--:-.   #---:-.   #---:-.   #---:-.   |
              hash 711e39e144d44931
            one at a time — 0.10s at 44100Hz in blocks of 1
              L in    -3.0 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out  -31.4 pk  -44.0 rms  |+=======================================|
              R in    -3.0 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out  -31.4 pk  -44.0 rms  |+=======================================|
              hash b8c8c5bb5dd69c81
        "#]],
    );
    // Values off the ends of each range are held to them.
    check(
        &Program::new(Kind::Compressor)
            .set("Threshold", 10.0)
            .set("Release", 0.0)
            .set("Ratio", 1000.0)
            .set("Attack", -5.0)
            .play("out of range", &drums(), 512),
        &expect![[r#"
            out of range — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -24.4 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.2 pk  -24.4 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash 1f471378bcc2b7c7
        "#]],
    );
    check(
        &Program::new(Kind::Compressor)
            .at(48000.0)
            .set("Ratio", 1.0)
            .play("ratio 1", &drums(), 512)
            .play("silence", &silence(0.1), 64)
            .reset(44100.0)
            .set("Ratio", 30.0)
            .play("prepared again", &drums(), 512),
        &expect![[r#"
            ratio 1 — 2.00s at 48000Hz in blocks of 512
              L in    -6.2 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.2 pk  -24.5 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -24.5 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash 11bc34dade2df3f9
            silence — 0.10s at 48000Hz in blocks of 64
              L in    -inf pk   -inf rms  |                                        |
              L out   -inf pk   -inf rms  |                                        |
              R in    -inf pk   -inf rms  |                                        |
              R out   -inf pk   -inf rms  |                                        |
              hash 0a6fb8af536edb25
            prepared again — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.3 pk  -31.7 rms  |@**+==-:  @**+==-:  @**+==-:  @**+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.2 pk  -31.7 rms  |@**+==-:  @**+==-:  @**+==-:  @**+==-:  |
              hash ec8b53ecc6832b94
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn limiter() {
    check(
        &Program::new(Kind::Limiter).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out  -28.9 pk  -39.0 rms  |***+==-:  ***+==-:  ***+==-:  ***+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out  -24.7 pk  -39.0 rms  |***+==-:  ***+==-:  ***+==-:  ***+==-:  |
              hash b2c6868505ec3d61
        "#]],
    );
    check(
        &Program::new(Kind::Limiter)
            .set("Attack", 5.0)
            .set("Release", 500.0)
            .set("Threshold", -3.0)
            .set("Input", 30.0)
            .set("Output", -6.0)
            .play("slammed", &drums(), 128)
            .set("Threshold", -60.0)
            .play("threshold falling", &tone(100.0, 0.0, 0.2), 1000),
        &expect![[r#"
            slammed — 2.00s at 44100Hz in blocks of 128
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   13.1 pk  -20.1 rms  |@@%#*#*=-:@%#*+*+-:.@%#*+*+-:.@%#*+*+-:.|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   13.6 pk  -19.9 rms  |@@%#*#*=-:@%#*+*+-:.@%#*+*+-:.@%#*+*+-:.|
              hash 7df5c3d0daaa4a94
            threshold falling — 0.20s at 44100Hz in blocks of 1000
              L in    -0.0 pk   -3.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out    3.4 pk  -17.5 rms  |@@%##**++==--::..                       |
              R in    -0.0 pk   -3.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out    3.4 pk  -17.5 rms  |@@%##**++==--::..                       |
              hash 57943b450a4d7311
        "#]],
    );
    check(
        &Program::new(Kind::Limiter)
            .at(96000.0)
            .set("Threshold", -12.0)
            .play("at 96k", &ramp(1000.0, -40.0, 0.0, 1.0), 512)
            .reset(48000.0)
            .play("prepared at 48k", &ramp(1000.0, -40.0, 0.0, 1.0), 512),
        &expect![[r#"
            at 96k — 1.00s at 96000Hz in blocks of 512
              L in    -0.0 pk  -12.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              L out  -12.0 pk  -18.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              R in    -0.0 pk  -12.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              R out  -12.0 pk  -18.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              hash ad56a7cf5c4337c5
            prepared at 48k — 1.00s at 48000Hz in blocks of 512
              L in    -0.0 pk  -12.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              L out  -12.0 pk  -18.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              R in    -0.0 pk  -12.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              R out  -12.0 pk  -18.7 rms  |====++++++******######%%%%%%@@@@@@@@@@@@|
              hash 7fa5a7140aa75dbd
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn gate() {
    check(
        &Program::new(Kind::Gate).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -23.8 rms  |@%#       @%#       @%#       @%#       |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -23.9 rms  |@%#       @%#       @%#       @%#       |
              hash 4940cf60191d9d98
        "#]],
    );
    check(
        &Program::new(Kind::Gate)
            .set("Attack", 2.0)
            .set("Hold", 100.0)
            .set("Release", 250.0)
            .set("Threshold", -20.0)
            .set("Knee", 12.0)
            .play("held open", &drums(), 441)
            .set("Threshold", -100.0)
            .play("wide open", &drums(), 512),
        &expect![[r#"
            held open — 2.00s at 44100Hz in blocks of 441
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.3 pk  -24.0 rms  |@%+       @%#+=     @%#+=     @%#+=     |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -24.0 rms  |@%+       @%#+=     @%#+=     @%#+=     |
              hash 55e68ec8e605f7bb
            wide open — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.2 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash d7935496b124711b
        "#]],
    );
    check(
        &Program::new(Kind::Gate)
            .set("Input", 20.0)
            .set("Output", -20.0)
            .set("Threshold", -10.0)
            .play("input and output", &drums(), 512)
            .set("Hold", 500.0)
            .set("Release", 1.0)
            .play("long hold, sharp release", &drums(), 64),
        &expect![[r#"
            input and output — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -23.8 rms  |@%#       @%#       @%#       @%#       |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -23.9 rms  |@%#       @%#       @%#       @%#       |
              hash 50eeccac11d06d1a
            long hold, sharp release — 2.00s at 44100Hz in blocks of 64
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash 56aeab4c88efec86
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn expander() {
    check(
        &Program::new(Kind::Expander).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash acab33a637a495dd
        "#]],
    );
    check(
        &Program::new(Kind::Expander)
            .set("Ratio", 4.0)
            .set("Threshold", -20.0)
            .set("Knee", 0.0)
            .play("ratio 4", &drums(), 512)
            .set("Knee", 30.0)
            .set("Attack", 200.0)
            .set("Release", 2000.0)
            .play("slow, soft", &drums(), 100),
        &expect![[r#"
            ratio 4 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.3 pk  -24.1 rms  |@%-       @%-       @%=       @%-       |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -24.2 rms  |@%=       @%-       @%=       @%-       |
              hash b0a8be82a54daadd
            slow, soft — 2.00s at 44100Hz in blocks of 100
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.3 pk  -28.0 rms  |@%*+-=:.  @#*=--:   %#+=:-.   %#+=:-.   |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -28.1 rms  |@%*+-=:.  @#*=--:   %#+=:-.   %#+=:-.   |
              hash eea44e8b8971a584
        "#]],
    );
    check(
        &Program::new(Kind::Expander)
            .set("Ratio", 4.0)
            .set("Input", 12.0)
            .set("Output", -12.0)
            .play("input and output", &ramp(220.0, -60.0, 0.0, 2.0), 512),
        &expect![[r#"
            input and output — 2.00s at 44100Hz in blocks of 512
              L in    -0.0 pk  -14.4 rms  |....::::----====++++****####%%%%@@@@@@@@|
              L out   -0.0 pk  -14.4 rms  |         .:-====++++****####%%%%@@@@@@@@|
              R in    -0.0 pk  -14.4 rms  |....::::----====++++****####%%%%@@@@@@@@|
              R out   -0.0 pk  -14.4 rms  |         .:-====++++****####%%%%@@@@@@@@|
              hash 863a1485d0393a51
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn delay() {
    // Mix defaults to 0: fully dry, so the output is the input.
    check(
        &Program::new(Kind::Delay).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash acab33a637a495dd
        "#]],
    );
    check(
        &Program::new(Kind::Delay)
            .set("Time", 0.25)
            .set("Feedback", -6.0)
            .set("Mix", 50.0)
            .play("echoes", &drums(), 512)
            .play("tail", &silence(1.0), 512),
        &expect![[r#"
            echoes — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -5.3 pk  -23.0 rms  |@#*+-@#*+-@%#+=@%*+-@%#+=@%*+=@%#+=@%*+=|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -5.9 pk  -23.0 rms  |@%*+-@%*+-@%#+=@%*+-@%#+=@%*+=@%#+=@%*+-|
              hash ff0c39db21135c73
            tail — 1.00s at 44100Hz in blocks of 512
              L in    -inf pk   -inf rms  |                                        |
              L out  -13.9 pk  -34.3 rms  |%##*++=-:*#**+==-:.+*++=--:. =+==-::.   |
              R in    -inf pk   -inf rms  |                                        |
              R out  -13.3 pk  -34.3 rms  |%##*++=--*#**+==-::**++=--:..++==-::.   |
              hash 968f759a103582cb
        "#]],
    );
    // A click on the left, crossfed and not fed back: each repeat changes
    // side.
    check(
        &Program::new(Kind::Delay)
            .set("Time", 0.2)
            .set("Feedback", -100.0)
            .set("Crossfeed", -3.0)
            .set("Mix", 100.0)
            .play("ping-pong", &left_only(impulse(-6.0, 2.0)), 256),
        &expect![[r#"
            ping-pong — 2.00s at 44100Hz in blocks of 256
              L in    -6.0 pk  -55.5 rms  |@                                       |
              L out   -6.0 pk  -51.8 rms  |@   @       %       #       *       +   |
              R in    -inf pk   -inf rms  |                                        |
              R out   -9.0 pk  -57.2 rms  |        @       %       #       *       |
              hash a4a63544859e1ad9
        "#]],
    );
    // Feedback at 0 dB is held to 0.99, so the repeats still fade. With
    // Crossfeed at 0 dB as well the two add to 1.98 round the loop, and it
    // runs away — in the original as much as here.
    check(
        &Program::new(Kind::Delay)
            .set("Time", 0.1)
            .set("Feedback", 0.0)
            .set("Mix", 50.0)
            .play("feedback held below 1", &impulse(-6.0, 3.0), 512)
            .set("Crossfeed", 0.0)
            .play("both at 0 dB, running away", &impulse(-6.0, 1.0), 512),
        &expect![[r#"
            feedback held below 1 — 3.00s at 44100Hz in blocks of 512
              L in    -6.0 pk  -57.2 rms  |@                                       |
              L out   -6.0 pk  -54.0 rms  |@ ### ### ### ### ### ### ### ### ### ##|
              R in    -6.0 pk  -57.2 rms  |@                                       |
              R out   -6.0 pk  -54.0 rms  |@ ### ### ### ### ### ### ### ### ### ##|
              hash ec27b5f4f94726d5
            both at 0 dB, running away — 1.00s at 44100Hz in blocks of 512
              L in    -6.0 pk  -52.4 rms  |@                                       |
              L out   38.5 pk   -6.3 rms  |@ #@  #@  %@  @@  @@  @@  @@  @@  @@  @ |
              R in    -6.0 pk  -52.4 rms  |@                                       |
              R out   38.5 pk   -6.3 rms  |@ #@  #@  %@  @@  @@  @@  @@  @@  @@  @ |
              hash 2a0915238cd2711d
        "#]],
    );
    check(
        &Program::new(Kind::Delay)
            .tempo(90.0)
            .set("Sync", 1.0)
            .set("Beat", 10.0)
            .set("Mix", 40.0)
            .play("1/8 at 90", &drums(), 256)
            .tempo(140.0)
            .play("1/8 at 140", &drums(), 256)
            .set("Beat", 10.4)
            .play("10.4 snaps to 1/8", &drums(), 256)
            .set("Beat", 10.6)
            .play("10.6 snaps to 1/8d", &drums(), 256)
            .set("Sync", 0.4)
            .play("sync 0.4 is off", &drums(), 256)
            .set("Sync", 0.6)
            .set("Beat", 0.0)
            .play("sync 0.6 is on, 1/64t", &drums(), 256),
        &expect![[r#"
            1/8 at 90 — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.3 pk  -23.5 rms  |@%*+-=@%#+@%*#*+@%#+@%*#*+@%#+@%*#*+@%#+|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.2 pk  -23.5 rms  |@%*+-=@%#+@%*#*+@%#+@%*#*+@%#+@%*#*+@%#+|
              hash a5db8f17f0773e28
            1/8 at 140 — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -7.9 pk  -23.6 rms  |@%%*@%*+#*@%**@%*+#*@%**@%*+#*@%**@%*+#*|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -7.8 pk  -23.7 rms  |@%%*@%#+##@%**@%*+#*@%**@%*+#*@%**@%*+#*|
              hash 7030859a53353225
            10.4 snaps to 1/8 — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -7.8 pk  -23.6 rms  |@%**@%*+#*@%**@%*+#*@%**@%*+#*@%**@%*+#*|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -7.6 pk  -23.7 rms  |@%**@%*+#*@%**@%*+#*@%**@%*+#*@%**@%*+#*|
              hash 601e3ad3b260c4a7
            10.6 snaps to 1/8d — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -7.6 pk  -23.7 rms  |@%*+=*@%*+@%##*+@%**@%##*+@%**@%##*+@%**|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -7.6 pk  -23.7 rms  |@%*+=*@%*+@%##*+@%**@%##*+@%**@%##*+@%**|
              hash 79a6feda9d05af01
            sync 0.4 is off — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -3.1 pk  -23.9 rms  |@@#+++#*+=@%#*++#*+=@%#*=+*+--@@#*=+++--|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -3.5 pk  -24.1 rms  |@@%+++#*+=@%#*++#*+=@%#+=+++--@@#*=+*+--|
              hash 4ec2eb82de7533c4
            sync 0.6 is on, 1/64t — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -3.5 pk  -23.6 rms  |@@@@*+=:. @@%*++=:. @@%*=+=-. @@#*++=-. |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -2.3 pk  -23.6 rms  |@@@%*+=-. @@#*++=:. @@%*++=-. @@%*++=-. |
              hash a2993fc30fe54847
        "#]],
    );
    // No tempo is 120; tempos past 500 are 500. Thirty-two bars at 120 is
    // 64 s, well inside the line; at 60 it is 128 s, past the end of it,
    // and the read holds at the oldest sample.
    check(
        &Program::new(Kind::Delay)
            .set("Sync", 1.0)
            .set("Beat", 13.0)
            .set("Mix", 40.0)
            .tempo(0.0)
            .play("tempo 0", &drums(), 512)
            .tempo(900.0)
            .play("tempo 900", &drums(), 512)
            .set("Beat", 32.0)
            .tempo(120.0)
            .play("32 bars at 120", &drums(), 512)
            .tempo(60.0)
            .play("32 bars at 60", &drums(), 512),
        &expect![[r#"
            tempo 0 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -2.9 pk  -23.9 rms  |@%*+-=-.  @%#+=+-:. @%#*=+=:. @%#*=+=:. |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -3.2 pk  -23.9 rms  |@%*+-=-.  @%#+=+-:. @%#*=+-:. @%#+=+-:. |
              hash ed3966b4b7391f17
            tempo 900 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -3.0 pk  -23.6 rms  |@%*@*#***-@%@%##**+=@%@%##**+=@%@%##**+=|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -2.4 pk  -23.6 rms  |@%*@#%***-@%@%##**+=@%@%##**+=@%@%##**+=|
              hash 65c3f89877f5e9f3
            32 bars at 120 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.6 pk  -25.6 rms  |@%*+-=-.  @%*+-=-.  @%*+-=-.  @%*+-=-.  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.7 pk  -25.7 rms  |@%*+-=-.  @%*+-=-.  @%*+-=-.  @%*+-=-.  |
              hash 8090a40b26c0cfa2
            32 bars at 60 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -7.9 pk  -25.6 rms  |@%*+-=-.  @%*+-=-.  @%*+-=-.  @%*+-=-.  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -8.0 pk  -25.7 rms  |@%*+-=-.  @%*+-=-.  @%*+-=-.  @%*+-=-.  |
              hash 7c36536ffb024c2b
        "#]],
    );
    // The line holds 120.1 s. Past that the ring wraps, and a delay read
    // across the seam has to find the samples on the far side of it.
    check(
        &Program::new(Kind::Delay)
            .set("Time", 1.0)
            .set("Mix", 50.0)
            .set("Feedback", -3.0)
            .play("two minutes", &silence(120.0), 4096)
            .play("across the seam", &hits(-6.0, 0.0, 0.5, 0.05, 8), 512),
        &expect![[r#"
            two minutes — 120.00s at 44100Hz in blocks of 4096
              L in    -inf pk   -inf rms  |                                        |
              L out   -inf pk   -inf rms  |                                        |
              R in    -inf pk   -inf rms  |                                        |
              R out   -inf pk   -inf rms  |                                        |
              hash 1c36c9bf4aa08f25
            across the seam — 4.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@#=. @#=. @#=. @#=. @#=. @#=. @#=. @#=. |
              L out   -0.6 pk  -23.6 rms  |@*-  @*-  @#=. @#=. @#=: @#=. @#=: @%=: |
              R in    -6.0 pk  -23.8 rms  |@#=. @#=. @#=. @#=. @#=. @#=. @#=. @#=. |
              R out   -1.0 pk  -23.6 rms  |@*-  @*-  @#=. @#=. @#=: @#=: @#+: @%+: |
              hash 094fc6787cf2cda1
        "#]],
    );
    check(
        &Program::new(Kind::Delay)
            .at(48000.0)
            .set("Time", 0.0)
            .set("Mix", 50.0)
            .play("no delay", &tone(440.0, -6.0, 0.1), 512)
            .set("Time", 0.05)
            .play("short", &tone(440.0, -6.0, 0.3), 333)
            .set("Time", 0.5)
            .play("sliding", &tone(440.0, -6.0, 0.1), 1)
            .set("Time", 0.02)
            .play("sliding back", &tone(440.0, -6.0, 0.3), 1)
            .reset(44100.0)
            .play("prepared again", &drums(), 512),
        &expect![[r#"
            no delay — 0.10s at 48000Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -6.0 pk  -10.8 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -6.0 pk  -10.8 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash a9c5778bf2d63a85
            short — 0.30s at 48000Hz in blocks of 333
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -1.3 pk   -7.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -1.3 pk   -7.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 63f4fc5b06de2165
            sliding — 0.10s at 48000Hz in blocks of 1
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -1.3 pk   -7.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -1.3 pk   -7.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 8a7710abb1fbbc75
            sliding back — 0.30s at 48000Hz in blocks of 1
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -1.2 pk   -8.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -1.2 pk   -8.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 25c9ee48127502cd
            prepared again — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -5.2 pk  -23.6 rms  |@@%*=+=-. @@#*++=-. @@%*++=-. @@%*++=-. |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -5.2 pk  -23.6 rms  |@@%*++=:. @@%*++=-. @@%*++=:. @@#*++=:. |
              hash ef3af34af5eb6133
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn simpleverb() {
    check(
        &Program::new(Kind::SimpleVerb)
            .play("defaults", &drums(), 512)
            .play("tail", &silence(1.0), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out  -10.0 pk  -26.7 rms  |@@%#*==-: @@%#++=-: @@%#++=-: @@%#*+=-:.|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out  -10.0 pk  -26.7 rms  |@@%#++=-: @@%#+==-: @@%#*+=-. @@%#++=-:.|
              hash b3fce4de7dffa939
            tail — 1.00s at 44100Hz in blocks of 512
              L in    -inf pk   -inf rms  |                                        |
              L out  -66.7 pk  -94.0 rms  |                                        |
              R in    -inf pk   -inf rms  |                                        |
              R out  -68.0 pk  -93.8 rms  |                                        |
              hash 9baf8066164d3983
        "#]],
    );
    check(
        &Program::new(Kind::SimpleVerb)
            .set("Room", 1.0)
            .set("Damp", 0.0)
            .set("PreDelay", 0.0)
            .set("LP", 1.0)
            .set("HP", 0.0)
            .set("Wet", 1.0)
            .set("Dry", 0.0)
            .play("big, bright, wet", &impulse(-6.0, 3.0), 256),
        &expect![[r#"
            big, bright, wet — 3.00s at 44100Hz in blocks of 256
              L in    -6.0 pk  -57.2 rms  |@                                       |
              L out  -10.4 pk  -46.0 rms  |@########**#*#***********+***+**++++++++|
              R in    -6.0 pk  -57.2 rms  |@                                       |
              R out  -10.4 pk  -46.0 rms  |@#############****************+**+*+++++|
              hash 0150a8ed3c227d1d
        "#]],
    );
    check(
        &Program::new(Kind::SimpleVerb)
            .at(48000.0)
            .set("Room", 0.05)
            .set("Damp", 1.0)
            .set("PreDelay", 1.0)
            .set("HP", 0.8)
            .play("small, dark, late", &drums(), 441)
            .set("Room", 0.6)
            .set("PreDelay", 0.5)
            .set("Wet", 0.1)
            .set("Dry", 1.0)
            .play("moved mid-flight", &drums(), 1),
        &expect![[r#"
            small, dark, late — 2.00s at 48000Hz in blocks of 441
              L in    -6.2 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out  -10.6 pk  -28.0 rms  |@#*=-**+-:@#*=-#*+-:@#*=-#*+-:@#*=-#*+-:|
              R in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out  -10.5 pk  -28.0 rms  |@#*=-#*+=:@#*=-#*+=:@#*=-#*+-:@#*=-#*+=:|
              hash 93824472ecb567cb
            moved mid-flight — 2.00s at 48000Hz in blocks of 1
              L in    -6.2 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -0.1 pk  -18.5 rms  |@@%*+*=-. @@%*++=-. @@%*+*=-. @@%*++=-. |
              R in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -0.1 pk  -18.5 rms  |@@%*+*=-. @@%*+*=-. @@%*++=-. @@%*++=-. |
              hash 002a94ed41158ca1
        "#]],
    );
    // Only raising the rate: the original leaves its comb lengths where the
    // old rate put them, and lowering it would have it read past its
    // buffers.
    check(
        &Program::new(Kind::SimpleVerb)
            .set("LP", 0.5)
            .play("44.1k", &drums(), 512)
            .reset(96000.0)
            .play("96k, same room", &drums(), 512)
            .set("Room", 0.9)
            .play("96k, new room", &drums(), 512),
        &expect![[r#"
            44.1k — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out  -10.5 pk  -27.8 rms  |@%#*+=--. @%#*+=-:. @%#*==-:. @%#*+=-:. |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out  -10.6 pk  -27.9 rms  |@%#*+=-:. @%#*+=-:. @%#*+=-:. @%#*+=-:. |
              hash fe22c3495729efc7
            96k, same room — 2.00s at 96000Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out  -10.5 pk  -27.8 rms  |@%#+==-.  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out  -10.5 pk  -27.8 rms  |@%*+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash 60d46e82a4c48e53
            96k, new room — 2.00s at 96000Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out  -10.4 pk  -27.8 rms  |@####**+++@####****+@###***+++@###****++|
              R in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out  -10.4 pk  -27.8 rms  |@###***+++@#%##***++@####**+++@####**+++|
              hash c7ac39f173b5e1e0
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn stereo_enhancer() {
    check(
        &Program::new(Kind::StereoEnhancer).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -5.7 pk  -25.9 rms  |@%#+==-:  @%#+==-:  @%#+=+-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -5.3 pk  -26.1 rms  |@%#+=+-:  @%#+=+-:  @%#+=+-:  @%#+=+-:  |
              hash 85e3481f90ced56b
        "#]],
    );
    check(
        &Program::new(Kind::StereoEnhancer)
            .set("Width", 1.0)
            .set("Freq HP", 0.2)
            .set("Width HP", 1.0)
            .play("wide", &drums(), 512)
            .play("mono stays mono", &tone(220.0, -6.0, 0.5), 512),
        &expect![[r#"
            wide — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -0.4 pk  -20.6 rms  |@%%*++=:. @@#*++=:. @@#*++=-. @@#*++=:. |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -0.5 pk  -20.7 rms  |@%%*++=:. @@#*=+=:. @@%*++=-. @@#*++=:. |
              hash 160b688f70ef0f66
            mono stays mono — 0.50s at 44100Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 8adc6a2b16a048d4
        "#]],
    );
    check(
        &Program::new(Kind::StereoEnhancer)
            .at(48000.0)
            .set("Width", 0.0)
            .set("Width HP", 0.0)
            .play("narrowed", &left_only(sweep(20.0, 20000.0, -6.0, 2.0)), 1),
        &expect![[r#"
            narrowed — 2.00s at 48000Hz in blocks of 1
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -7.1 pk  -14.9 rms  |@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -inf pk   -inf rms  |                                        |
              R out  -12.0 pk  -15.1 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash ee6927cebc0eac4c
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn stereo_processor() {
    check(
        &Program::new(Kind::StereoProcessor).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash 77611d1fabde2709
        "#]],
    );
    // A left-only signal lands the rotation on its right-is-zero branch, a
    // right-only one on its left-is-zero branch.
    check(
        &Program::new(Kind::StereoProcessor)
            .set("Rotation", 0.75)
            .play("left, rotated", &left_only(tone(220.0, -6.0, 0.5)), 512)
            .play("right, rotated", &right_only(tone(220.0, -6.0, 0.5)), 512)
            .set("Rotation", 0.0)
            .play("both, rotated back", &drums(), 512),
        &expect![[r#"
            left, rotated — 0.50s at 44100Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -6.0 pk  -11.9 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -inf pk   -inf rms  |                                        |
              R out   -9.0 pk  -12.2 rms  |%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash c8e26f1a25363e68
            right, rotated — 0.50s at 44100Hz in blocks of 512
              L in    -inf pk   -inf rms  |                                        |
              L out   -9.0 pk  -12.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -9.0 pk  -12.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 906f8c4356b324a1
            both, rotated back — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -4.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -3.9 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash 5e82e2579f661977
        "#]],
    );
    check(
        &Program::new(Kind::StereoProcessor)
            .set("Width 1", 1.0)
            .set("Center 1", 0.0)
            .set("Pan 1", -1.0)
            .set("Pan 2", 0.5)
            .set("Center 2", 1.0)
            .set("Width 2", 0.0)
            .set("Output", 1.0)
            .play("every knob", &drums(), 333)
            .set("Output", 0.0)
            .play("output down", &drums(), 1),
        &expect![[r#"
            every knob — 2.00s at 44100Hz in blocks of 333
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   11.6 pk   -9.8 rms  |@@@%##*=-:@@@%*#*=-:@@@%##*+-:@@@%##*=-:|
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   11.6 pk   -9.8 rms  |@@@%##*=-:@@@%*#*=-:@@@%##*+-:@@@%##*=-:|
              hash 4dcb55cc9ceff9eb
            output down — 2.00s at 44100Hz in blocks of 1
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   10.7 pk  -21.6 rms  |@%=:..    *+-: .    *+-:..    *+-:..    |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   10.7 pk  -21.6 rms  |@%=:..    *+-: .    *+-:..    *+-:..    |
              hash cb3aaac79452ca4d
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn crossfeed() {
    // Switched on from the first block: the crossfed signal fades in over
    // 50 ms.
    check(
        &Program::new(Kind::Crossfeed).play("defaults", &left_only(tone(440.0, -6.0, 0.5)), 512),
        &expect![[r#"
            defaults — 0.50s at 44100Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -9.1 pk  -12.4 rms  |#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -inf pk   -inf rms  |                                        |
              R out  -15.0 pk  -18.3 rms  |*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash 406502282f4b4ed6
        "#]],
    );
    check(
        &Program::new(Kind::Crossfeed)
            .set("Enable", 0.0)
            .play("off", &drums(), 512)
            .set("Enable", 1.0)
            .play("switched on", &drums(), 256)
            .set("Enable", 0.4)
            .play("0.4 is off", &drums(), 1),
        &expect![[r#"
            off — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -6.1 pk  -24.6 rms  |%%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -6.2 pk  -24.7 rms  |%%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash c65c4c8cf96283ce
            switched on — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -4.7 pk  -23.6 rms  |@%#*=+-:  @%#+=+-:  @%#*=+-:  @%#+=+-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -4.6 pk  -23.7 rms  |@%#+=+-:  @%#+=+-:  @%#+=+-:. @%#+=+-:  |
              hash 654da77dc36fb1de
            0.4 is off — 2.00s at 44100Hz in blocks of 1
              L in    -6.1 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              L out   -5.2 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              R out   -4.5 pk  -23.8 rms  |@%#+==-:  @%#+==-:  @%#+==-:  @%#+==-:  |
              hash 3408f4af1a6a07c6
        "#]],
    );
    // Only 44.1, 48 and 96 kHz have kernels; any other rate takes the
    // nearest.
    check(
        &Program::new(Kind::Crossfeed)
            .at(48000.0)
            .play("48k", &left_only(impulse(-6.0, 0.1)), 512)
            .reset(96000.0)
            .play("96k", &left_only(impulse(-6.0, 0.1)), 512)
            .reset(88200.0)
            .play("88.2k takes 96k", &left_only(impulse(-6.0, 0.1)), 512)
            .reset(22050.0)
            .play("22.05k takes 44.1k", &left_only(impulse(-6.0, 0.1)), 512),
        &expect![[r#"
            48k — 0.10s at 48000Hz in blocks of 512
              L in    -6.0 pk  -42.8 rms  |@                                       |
              L out  -69.9 pk -104.0 rms  |                                        |
              R in    -inf pk   -inf rms  |                                        |
              R out  -69.9 pk -105.1 rms  |                                        |
              hash 00da4b54e99bdeef
            96k — 0.10s at 96000Hz in blocks of 512
              L in    -6.0 pk  -45.8 rms  |@                                       |
              L out   -6.0 pk  -45.8 rms  |@                                       |
              R in    -inf pk   -inf rms  |                                        |
              R out  -27.7 pk  -63.7 rms  |*                                       |
              hash a513992e8e53a7e7
            88.2k takes 96k — 0.10s at 88200Hz in blocks of 512
              L in    -6.0 pk  -45.5 rms  |@                                       |
              L out   -6.0 pk  -45.4 rms  |@                                       |
              R in    -inf pk   -inf rms  |                                        |
              R out  -27.7 pk  -63.4 rms  |*                                       |
              hash 57febb4f5dad2327
            22.05k takes 44.1k — 0.10s at 22050Hz in blocks of 512
              L in    -6.0 pk  -39.4 rms  |@                                       |
              L out   -6.1 pk  -39.4 rms  |@                                       |
              R in    -inf pk   -inf rms  |                                        |
              R out  -22.5 pk  -54.2 rms  |#                                       |
              hash a5ccc952a00e266d
        "#]],
    );
}
