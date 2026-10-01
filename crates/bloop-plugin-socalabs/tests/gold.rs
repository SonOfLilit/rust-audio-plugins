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
        let effect = Effect::new(self.kind, 44100.0);
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
    let mut rate = program.sample_rate;
    let mut effect = Effect::new(program.kind, rate as f32);
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

    /// The Gin effects wrapped by the harness itself, from `extras.inc`.
    fn extras(harness: &Path) -> Vec<String> {
        std::fs::read_to_string(harness.join("extras.inc"))
            .expect("read extras.inc")
            .lines()
            .filter_map(|l| l.trim().strip_prefix("GINEXTRA(")?.strip_suffix(')'))
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
        for name in extras(&harness) {
            units.push((harness.join("extras").join(format!("{name}.cpp")), true));
        }
        let config_changed = mtime(&harness.join("config.h"));

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
                || config_changed > mtime(&object)
                || (tracks_harness && harness_changed > mtime(&object));
            if !stale {
                return (object, None);
            }
            let c = source.extension().is_some_and(|e| e == "c");
            let mut cmd = Command::new(if c { "gcc" } else { "g++" });
            if !c {
                cmd.arg("-std=c++20");
            }
            // The one class that computes in long double, at the width the
            // port computes in (extras/Distortion.cpp).
            if source.ends_with("extras/Distortion.cpp") {
                cmd.arg("-mlong-double-64");
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -7.7 pk  -34.8 rms  |%++=--:.  %++=--:.  %++=--:.  %++=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -7.1 pk  -34.8 rms  |%++=--:.  %++=--:.  %++=--:.  %++=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -2.4 pk  -32.0 rms  |%=-:.:.   %+=-.-.   @+=-.-.   @+=-.-.   |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -3.3 pk  -32.1 rms  |%=-:.:.   @+=-.-.   @+=-.-.   @+=-.-.   |
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
              L in    -0.0 pk  -13.6 rms  |.:::::-----=====++++*****#####%%%%%@@@@@|
              L out  -16.5 pk  -24.4 rms  |.:::::-----=====+++++*********##########|
              R in    -0.0 pk  -13.6 rms  |.:::::-----=====++++*****#####%%%%%@@@@@|
              R out  -16.5 pk  -24.4 rms  |.:::::-----=====+++++*********##########|
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -7.7 pk  -34.8 rms  |%++=--:.  %++=--:.  %++=--:.  %++=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -7.1 pk  -34.8 rms  |%++=--:.  %++=--:.  %++=--:.  %++=--:.  |
              hash 2d1ac18111447c91
            halfway — 0.03s at 44100Hz in blocks of 333
              L in    -3.0 pk   -6.1 rms  |@@@@@@@@@@@@@@@@@%@@%@@%@@%@@%@@%@@%@@%|
              L out   -6.0 pk  -14.3 rms  |#***********##*##*##*###%%#%%#%%%%%%%%%|
              R in    -3.0 pk   -6.1 rms  |@@@@@@@@@@@@@@@@@%@@%@@%@@%@@%@@%@@%@@%|
              R out   -6.0 pk  -14.3 rms  |#***********##*##*##*###%%#%%#%%%%%%%%%|
              hash d99d7f72654e5ff5
            turned back — 2.00s at 44100Hz in blocks of 333
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.7 pk  -35.0 rms  |%-::.:    *:::.:    +:::.:    +:::.:    |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.7 pk  -34.9 rms  |%-::.:    *:::.:    *:::.:    *:::.:    |
              hash 711e39e144d44931
            one at a time — 0.10s at 44100Hz in blocks of 1
              L in    -3.0 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out  -31.4 pk  -44.0 rms  |=---------------------------------------|
              R in    -3.0 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out  -31.4 pk  -44.0 rms  |=---------------------------------------|
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -24.4 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.2 pk  -24.4 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.2 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.2 pk  -24.5 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -24.5 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              hash 11bc34dade2df3f9
            silence — 0.10s at 48000Hz in blocks of 64
              L in    zero pk   zero rms  |                                        |
              L out   zero pk   zero rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out   zero pk   zero rms  |                                        |
              hash 0a6fb8af536edb25
            prepared again — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.3 pk  -31.7 rms  |%++=--:.  %++=--:.  %++=--:.  %++=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.2 pk  -31.7 rms  |%++=--:.  %++=--:.  %++=--:.  %++=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out  -28.9 pk  -39.0 rms  |+++=--:.  +++=--:.  +++=--:.  +++=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out  -24.7 pk  -39.0 rms  |+++=--:.  +++=--:.  +++=--:.  +++=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   13.1 pk  -20.1 rms  |@%#*+*+-:.@#*+=+=:. @#*+=+=:. @#*+=+=:. |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   13.6 pk  -19.9 rms  |%%#*+*+-:.@#*+=+=:. @#*+=+=:. @#*+=+=:. |
              hash 7df5c3d0daaa4a94
            threshold falling — 0.20s at 44100Hz in blocks of 1000
              L in    -0.0 pk   -3.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out    3.4 pk  -17.5 rms  |@%#**++==--::..                         |
              R in    -0.0 pk   -3.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out    3.4 pk  -17.5 rms  |@%#**++==--::..                         |
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
              L in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              L out  -12.0 pk  -18.7 rms  |----======++++++******######%%%%%%%%%%%%|
              R in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              R out  -12.0 pk  -18.7 rms  |----======++++++******######%%%%%%%%%%%%|
              hash ad56a7cf5c4337c5
            prepared at 48k — 1.00s at 48000Hz in blocks of 512
              L in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              L out  -12.0 pk  -18.7 rms  |----======++++++******######%%%%%%%%%%%%|
              R in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              R out  -12.0 pk  -18.7 rms  |----======++++++******######%%%%%%%%%%%%|
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -23.8 rms  |%#*       %#*       %#*       %#*       |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -23.9 rms  |%#*       %#*       %#*       %#*       |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.3 pk  -24.0 rms  |%#=       %#*=-     %#*=-     %#*=-     |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -24.0 rms  |%#=       %#*=-     %#*=-     %#*=-     |
              hash 55e68ec8e605f7bb
            wide open — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.2 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -23.8 rms  |%#*       %#*       %#*       %#*       |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -23.9 rms  |%#*       %#*       %#*       %#*       |
              hash 50eeccac11d06d1a
            long hold, sharp release — 2.00s at 44100Hz in blocks of 64
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.3 pk  -24.1 rms  |%#:       %#:       %#-       %#:       |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -24.2 rms  |%#-       %#:       %#-       %#:       |
              hash b0a8be82a54daadd
            slow, soft — 2.00s at 44100Hz in blocks of 100
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.3 pk  -28.0 rms  |%#+=:-.   %*+-::.   #*=-.:    #*=-.:    |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -28.1 rms  |%#+=:-.   %*+-::.   #*=-.:    #*=-.:    |
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
              L in    -0.0 pk  -14.4 rms  |    ....::::----====++++****####%%%%@@@@|
              L out   -0.0 pk  -14.4 rms  |          .:----====++++****####%%%%@@@@|
              R in    -0.0 pk  -14.4 rms  |    ....::::----====++++****####%%%%@@@@|
              R out   -0.0 pk  -14.4 rms  |          .:----====++++****####%%%%@@@@|
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -5.3 pk  -23.0 rms  |%*+=:%*+=:%#*=-%#+=:@#*=-%#+=-%#*=-%#+=-|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -5.9 pk  -23.0 rms  |%#+=:%#+=:@#*=-%#+=:%#*=-%#+=-@#*=-%#+=:|
              hash ff0c39db21135c73
            tail — 1.00s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out  -13.9 pk  -34.3 rms  |#**+==-:.+*++=--:. =+==-::.  -=--:..    |
              R in    zero pk   zero rms  |                                        |
              R out  -13.3 pk  -34.3 rms  |#**+==-::+*++=--:..++==-::.  ==--:..    |
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
              L in    -6.0 pk  -55.5 rms  |%                                       |
              L out   -6.0 pk  -51.8 rms  |%   %       #       *       +       =   |
              R in    zero pk   zero rms  |                                        |
              R out   -9.0 pk  -57.2 rms  |        %       #       *       +       |
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
              L in    -6.0 pk  -57.2 rms  |%                                       |
              L out   -6.0 pk  -54.0 rms  |% *** *** *** *** *** *** *** *** *** **|
              R in    -6.0 pk  -57.2 rms  |%                                       |
              R out   -6.0 pk  -54.0 rms  |% *** *** *** *** *** *** *** *** *** **|
              hash ec27b5f4f94726d5
            both at 0 dB, running away — 1.00s at 44100Hz in blocks of 512
              L in    -6.0 pk  -52.4 rms  |%                                       |
              L out   38.5 pk   -6.3 rms  |% *%  *@  #@  %@  @@  @@  @@  @@  @@  @ |
              R in    -6.0 pk  -52.4 rms  |%                                       |
              R out   38.5 pk   -6.3 rms  |% *%  *@  #@  %@  @@  @@  @@  @@  @@  @ |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.3 pk  -23.5 rms  |%#+=:-%#*=%#+*+=%#*=%#+*+=%#*=%#+*+=%#*=|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.2 pk  -23.5 rms  |%#+=:-%#*=%#+*+=%#*=%#+*+=%#*=%#+*+=%#*=|
              hash a5db8f17f0773e28
            1/8 at 140 — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -7.9 pk  -23.6 rms  |%##+%#+=*+%#++%#+=*+%#++%#+=*+%#++%#+=*+|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -7.8 pk  -23.7 rms  |%##+%#*=**%#++%#+=*+%#++%#+=*+%#++%#+=*+|
              hash 7030859a53353225
            10.4 snaps to 1/8 — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -7.8 pk  -23.6 rms  |%#++%#+=*+%#++%#+=*+%#++%#+=*+%#++%#+=*+|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -7.6 pk  -23.7 rms  |%#++%#+=*+%#++%#+=*+%#++%#+=*+%#++%#+=*+|
              hash 601e3ad3b260c4a7
            10.6 snaps to 1/8d — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -7.6 pk  -23.7 rms  |%#+=-+%#+=%#**+=%#++%#**+=%#++%#**+=%#++|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -7.6 pk  -23.7 rms  |%#+=-+%#+=%#**+=%#++%#**+=%#++%#**+=%#++|
              hash 79a6feda9d05af01
            sync 0.4 is off — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -3.1 pk  -23.9 rms  |%%*===*+=-@#*+==*+=-@#*+-=+=::@%*+-===::|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -3.5 pk  -24.1 rms  |%%#===*+=-@#*+==*+=-@#*=-===::@%*+-=+=::|
              hash 4ec2eb82de7533c4
            sync 0.6 is on, 1/64t — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -3.5 pk  -23.6 rms  |@%%%+=-.  @%#+==-.  @%#+-=-:  @%*+==-:  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -2.3 pk  -23.6 rms  |@%%#+=-:  @%*+==-.  @%#+==-:  @%#+==-:  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -2.9 pk  -23.9 rms  |%#+=:-:   @#*=-=:.  @#*+-=-.  @#*+-=-.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -3.2 pk  -23.9 rms  |%#+=:-:   @#*=-=:.  @#*+-=:.  @#*=-=:.  |
              hash ed3966b4b7391f17
            tempo 900 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -3.0 pk  -23.6 rms  |@#+%+*+++:%#%#**++=-%#%#**++=-%#%#**++=-|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -2.4 pk  -23.6 rms  |@#+%*#+++:%#%#**++=-%#%#**++=-%#%#**++=-|
              hash 65c3f89877f5e9f3
            32 bars at 120 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.6 pk  -25.6 rms  |%#+=:-:   %#+=:-:   %#+=:-:   %#+=:-:   |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.7 pk  -25.7 rms  |%#+=:-:   %#+=:-:   %#+=:-:   %#+=:-:   |
              hash 8090a40b26c0cfa2
            32 bars at 60 — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -7.9 pk  -25.6 rms  |%#+=:-:   %#+=:-:   %#+=:-:   %#+=:-:   |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -8.0 pk  -25.7 rms  |%#+=:-:   %#+=:-:   %#+=:-:   %#+=:-:   |
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
              L in    zero pk   zero rms  |                                        |
              L out   zero pk   zero rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out   zero pk   zero rms  |                                        |
              hash 1c36c9bf4aa08f25
            across the seam — 4.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%*-  %*-  %*-  %*-  %*-  %*-  %*-  %*-  |
              L out   -0.6 pk  -23.6 rms  |%+:  %+:  @*-  @*-  @*-. @*-  @*-. @#-. |
              R in    -6.0 pk  -23.8 rms  |%*-  %*-  %*-  %*-  %*-  %*-  %*-  %*-  |
              R out   -1.0 pk  -23.6 rms  |%+:  %+:  @*-  @*-  @*-. @*-. @*=. @#=. |
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
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk  -10.8 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.0 pk  -10.8 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash a9c5778bf2d63a85
            short — 0.30s at 48000Hz in blocks of 333
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -1.3 pk   -7.9 rms  |%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -1.3 pk   -7.9 rms  |%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 63f4fc5b06de2165
            sliding — 0.10s at 48000Hz in blocks of 1
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -1.3 pk   -7.2 rms  |@@%@@@@@@@@@@@@@@@@@%@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -1.3 pk   -7.2 rms  |@@%@@@@@@@@@@@@@@@@@%@@@@@@@@@@@@@@@@@@@|
              hash 8a7710abb1fbbc75
            sliding back — 0.30s at 48000Hz in blocks of 1
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -1.2 pk   -8.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -1.2 pk   -8.2 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 25c9ee48127502cd
            prepared again — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -5.2 pk  -23.6 rms  |@%#+-=-:  @%*+==-:  @%#+==-:  @%#+==-:  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -5.2 pk  -23.6 rms  |@%#+==-.  @%#+==-:  @%#+==-.  %%*+==-.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out  -10.0 pk  -26.7 rms  |%%#*+--:. %%#*==-:. %%#*==-:. %%#*+=-:. |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out  -10.0 pk  -26.7 rms  |%%#*==-:. %%#*=--:. %%#*+=-:  %%#*==-:. |
              hash b3fce4de7dffa939
            tail — 1.00s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out  -66.7 pk  -94.0 rms  |                                        |
              R in    zero pk   zero rms  |                                        |
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
              L in    -6.0 pk  -57.2 rms  |%                                       |
              L out  -10.4 pk  -46.0 rms  |%********++*+*+++++++++++=+++=++========|
              R in    -6.0 pk  -57.2 rms  |%                                       |
              R out  -10.4 pk  -46.0 rms  |%*************++++++++++++++++=++=+=====|
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
              L in    -6.2 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out  -10.6 pk  -28.0 rms  |%*+-:++=:.%*+-:*+=:.%*+-:*+=:.%*+-:*+=:.|
              R in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out  -10.5 pk  -28.0 rms  |%*+-:*+=-.%*+-:*+=-.%*+-:*+=:.%*+-:*+=-.|
              hash 93824472ecb567cb
            moved mid-flight — 2.00s at 48000Hz in blocks of 1
              L in    -6.2 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -0.1 pk  -18.5 rms  |%%#+=+-:  @%#+==-:  @%#+=+-:  @%#+==-:  |
              R in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -0.1 pk  -18.5 rms  |%%#+=+-:  @%#+=+-:  @%#+==-:  @%#+==-:  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out  -10.5 pk  -27.8 rms  |%#*+=-::  %#*+=-:.  %#*+--:.  %#*+=-:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out  -10.6 pk  -27.9 rms  |%#*+=-:.  %#*+=-:.  %#*+=-:.  %#*+=-:.  |
              hash fe22c3495729efc7
            96k, same room — 2.00s at 96000Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out  -10.5 pk  -27.8 rms  |%#*=--:   %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out  -10.5 pk  -27.8 rms  |%#+=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              hash 60d46e82a4c48e53
            96k, new room — 2.00s at 96000Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out  -10.4 pk  -27.8 rms  |%****++===%****++++=%***+++===%***++++==|
              R in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out  -10.4 pk  -27.8 rms  |%***+++===%*#**+++==%****++===%****++===|
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -5.7 pk  -25.9 rms  |@#*=--:.  %#*=--:.  %#*=-=:.  @#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -5.3 pk  -26.1 rms  |@#*=-=:.  @#*=-=:.  @#*=-=:.  %#*=-=:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -0.4 pk  -20.6 rms  |@##+==-.  @%*+==-.  @%*+==-:  @%*+==-.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -0.5 pk  -20.7 rms  |@##+==-.  @%*+-=-.  @%#+==-:  @%*+==-.  |
              hash 160b688f70ef0f66
            mono stays mono — 0.50s at 44100Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
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
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -7.1 pk  -14.9 rms  |%#######################################|
              R in    zero pk   zero rms  |                                        |
              R out  -12.0 pk  -15.1 rms  |########################################|
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk  -11.9 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    zero pk   zero rms  |                                        |
              R out   -9.0 pk  -12.2 rms  |#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash c8e26f1a25363e68
            right, rotated — 0.50s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out   -9.0 pk  -12.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -9.0 pk  -12.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash 906f8c4356b324a1
            both, rotated back — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -4.1 pk  -23.9 rms  |@#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -3.9 pk  -23.8 rms  |@#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   11.6 pk   -9.8 rms  |@@@#**+-:.@@%#+*+-:.@@%#**+=:.@@%#**+-:.|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   11.6 pk   -9.8 rms  |@@@#**+-:.@@%#+*+-:.@@%#**+=:.@@%#**+-:.|
              hash 4dcb55cc9ceff9eb
            output down — 2.00s at 44100Hz in blocks of 1
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   10.7 pk  -21.6 rms  |@#-.      +=:.      +=:.      +=:.      |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   10.7 pk  -21.6 rms  |@#-.      +=:.      +=:.      +=:.      |
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
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -9.1 pk  -12.4 rms  |*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    zero pk   zero rms  |                                        |
              R out  -15.0 pk  -18.3 rms  |+*######################################|
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
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -24.6 rms  |##*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.2 pk  -24.7 rms  |##*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              hash c65c4c8cf96283ce
            switched on — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -4.7 pk  -23.6 rms  |%#*+-=:.  @#*=-=:.  @#*+-=:.  @#*=-=:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -4.6 pk  -23.7 rms  |%#*=-=:.  @#*=-=:.  @#*=-=:.  @#*=-=:.  |
              hash 654da77dc36fb1de
            0.4 is off — 2.00s at 44100Hz in blocks of 1
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -5.2 pk  -23.8 rms  |@#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -4.5 pk  -23.8 rms  |@#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
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
              L in    -6.0 pk  -42.8 rms  |%                                       |
              L out  -69.9 pk -104.0 rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out  -69.9 pk -105.1 rms  |                                        |
              hash 00da4b54e99bdeef
            96k — 0.10s at 96000Hz in blocks of 512
              L in    -6.0 pk  -45.8 rms  |%                                       |
              L out   -6.0 pk  -45.8 rms  |%                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -27.7 pk  -63.7 rms  |+                                       |
              hash a513992e8e53a7e7
            88.2k takes 96k — 0.10s at 88200Hz in blocks of 512
              L in    -6.0 pk  -45.5 rms  |%                                       |
              L out   -6.0 pk  -45.4 rms  |%                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -27.7 pk  -63.4 rms  |+                                       |
              hash 57febb4f5dad2327
            22.05k takes 44.1k — 0.10s at 22050Hz in blocks of 512
              L in    -6.0 pk  -39.4 rms  |%                                       |
              L out   -6.1 pk  -39.4 rms  |%                                       |
              R in    zero pk   zero rms  |                                        |
              R out  -22.5 pk  -54.2 rms  |*                                       |
              hash a5ccc952a00e266d
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn graphic_eq() {
    // Flat: every band at 0 dB adds nothing back.
    check(
        &Program::new(Kind::GraphicEq).play("flat", &sweep(20.0, 20000.0, -12.0, 2.0), 512),
        &expect![[r#"
            flat — 2.00s at 44100Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out  -12.0 pk  -15.0 rms  |########################################|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out  -12.0 pk  -15.0 rms  |########################################|
              hash d4fa0162bb3d0161
        "#]],
    );
    // A smile, then a frown, over a sweep: the strip shows where each band
    // sits.
    check(
        &Program::new(Kind::GraphicEq)
            .set("31 Hz", 12.0)
            .set("63 Hz", 9.0)
            .set("125 Hz", 3.0)
            .set("250 Hz", -3.0)
            .set("500 Hz", -9.0)
            .set("1k Hz", -12.0)
            .set("2k Hz", -9.0)
            .set("4k Hz", -3.0)
            .set("8k Hz", 9.0)
            .set("16k Hz", 12.0)
            .play("smile", &sweep(20.0, 20000.0, -12.0, 2.0), 512)
            .set("31 Hz", -12.0)
            .set("1k Hz", 12.0)
            .set("16k Hz", -12.0)
            .play("frown, turning", &sweep(20.0, 20000.0, -12.0, 2.0), 256),
        &expect![[r#"
            smile — 2.00s at 44100Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out    0.3 pk   -9.3 rms  |@@@@@@@@@%%%%%%#############%%%%%@@@@@@@|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out    0.3 pk   -9.3 rms  |@@@@@@@@@%%%%%%#############%%%%%@@@@@@@|
              hash d77416eb06715891
            frown, turning — 2.00s at 44100Hz in blocks of 256
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out   -0.1 pk  -10.6 rms  |%%#%%@@@%%%%%%#####%@@@@@%%####%%@@@@%%#|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out   -0.1 pk  -10.6 rms  |%%#%%@@@%%%%%%#####%@@@@@%%####%%@@@@%%#|
              hash 5f0dbcab4b552065
        "#]],
    );
    check(
        &Program::new(Kind::GraphicEq)
            .at(96000.0)
            .set("LP", 1.0)
            .set("LP Freq", 120.0)
            .set("LP Slope", 1.0)
            .set("HP", 1.0)
            .set("HP Freq", 40.0)
            .play(
                "at 96kHz, cut both ends",
                &sweep(20.0, 40000.0, -12.0, 2.0),
                512,
            ),
        &expect![[r#"
            at 96kHz, cut both ends — 2.00s at 96000Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out  -12.0 pk  -17.4 rms  |*=++**############%#############**+=-:. |
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out  -12.0 pk  -17.4 rms  |*=++**############%#############**+=-:. |
              hash 654b5417c42e7f19
        "#]],
    );
    // The highpass and lowpass at both slopes, then switched off with their
    // frequencies moved: an off filter's knobs do not smooth, so the EQ
    // stays on its per-sample path.
    check(
        &Program::new(Kind::GraphicEq)
            .at(48000.0)
            .set("HP", 1.0)
            .set("HP Freq", 60.0)
            .set("LP", 1.0)
            .set("LP Freq", 100.0)
            .play("12 dB each", &sweep(20.0, 20000.0, -6.0, 2.0), 512)
            .set("HP Slope", 1.0)
            .set("LP Slope", 1.0)
            .play("24 dB each", &sweep(20.0, 20000.0, -6.0, 2.0), 512)
            .set("HP", 0.0)
            .set("LP", 0.0)
            .set("HP Freq", 80.0)
            .set("LP Freq", 90.0)
            .play("off, knobs moved", &sweep(20.0, 20000.0, -6.0, 1.0), 512)
            .set("Output", -12.0)
            .play("output down", &drums(), 333),
        &expect![[r#"
            12 dB each — 2.00s at 48000Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.1 pk  -13.8 rms  |#:--===+***##%%%%%%%%%%%%%%%%%###**++==-|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.1 pk  -13.8 rms  |#:--===+***##%%%%%%%%%%%%%%%%%###**++==-|
              hash 52e3054b73a8803d
            24 dB each — 2.00s at 48000Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.2 pk  -14.8 rms  |-     ..--=+*#%%%%%%%%%%%%%%%#**+=-:.   |
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.2 pk  -14.8 rms  |-     ..--=+*#%%%%%%%%%%%%%%%#**+=-:.   |
              hash e05d49052749fcb1
            off, knobs moved — 1.00s at 48000Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash 6851964dcb1a9831
            output down — 2.00s at 48000Hz in blocks of 333
              L in    -6.2 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.3 pk  -31.9 rms  |%+=:..    *+=:..    *+=:..    *+=:..    |
              R in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.3 pk  -31.9 rms  |%+=:..    *+=:..    *+=:..    *+=:..    |
              hash 60a56bd7a85059e7
        "#]],
    );
    // Prepared again at another rate: the bands keep the coefficients
    // they were designed with, as the original's do.
    check(
        &Program::new(Kind::GraphicEq)
            .set("1k Hz", 12.0)
            .play("44.1k", &sweep(20.0, 20000.0, -12.0, 1.0), 512)
            .reset(96000.0)
            .play(
                "96k, bands as they were",
                &sweep(20.0, 20000.0, -12.0, 1.0),
                512,
            ),
        &expect![[r#"
            44.1k — 1.00s at 44100Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out    0.0 pk  -11.8 rms  |%%%%%%%%%%%%%%%%%%%%@@@@@%%%%%%%%%%%%%%%|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out    0.0 pk  -11.8 rms  |%%%%%%%%%%%%%%%%%%%%@@@@@%%%%%%%%%%%%%%%|
              hash 22617ecd4b3ac8dd
            96k, bands as they were — 1.00s at 96000Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out    0.0 pk  -11.8 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@%%%%%%%%%%|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out    0.0 pk  -11.8 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@%%%%%%%%%%|
              hash 4d14cb6ce32ed0ed
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn bitcrusher_dirty() {
    // Dirty mode never touches the oversampler, so it is held to the
    // original bit for bit. At its defaults — 16 bits, 48 kHz — the hold
    // steps every sample and the grid is fine enough to leave a tone be.
    check(
        &Program::new(Kind::BitCrusher).set("Dirty", 1.0).play(
            "16 bits, 48 kHz",
            &tone(1000.0, -6.0, 0.5),
            512,
        ),
        &expect![[r#"
            16 bits, 48 kHz — 0.50s at 44100Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.0 pk   -9.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 7753929cc8677585
        "#]],
    );
    check(
        &Program::new(Kind::BitCrusher)
            .set("Dirty", 1.0)
            .set("Bits", 4.0)
            .set("Rate", 8000.0)
            .play("4 bits, 8 kHz", &drums(), 512)
            .set("Bits", 1.0)
            .set("Rate", 500.0)
            .play("1 bit, 500 Hz", &drums(), 333)
            .set("Bits", 8.5)
            .set("Rate", 2500.0)
            .play("moving", &sweep(50.0, 5000.0, -3.0, 1.0), 1),
        &expect![[r#"
            4 bits, 8 kHz — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.0 pk  -23.6 rms  |%#*       %**       %**       %*        |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.0 pk  -23.7 rms  |%#*       %**       %**       %**       |
              hash 298bc257234eba3e
            1 bit, 500 Hz — 2.00s at 44100Hz in blocks of 333
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -4.4 pk  -31.6 rms  |@                                       |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -4.4 pk  -31.5 rms  |@                                       |
              hash 470cefd4be201320
            moving — 1.00s at 44100Hz in blocks of 1
              L in    -3.0 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              L out   -2.6 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -3.0 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R out   -2.6 pk   -6.0 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 22a8249149d94725
        "#]],
    );
    check(
        &Program::new(Kind::BitCrusher)
            .at(96000.0)
            .set("Dirty", 1.0)
            .set("Bits", 6.0)
            .set("Rate", 11025.0)
            .play("at 96k", &ramp(440.0, -40.0, 0.0, 1.0), 4096)
            .play(
                "longer than a crusher block",
                &tone(440.0, -6.0, 0.2),
                10000,
            )
            .reset(48000.0)
            .play("prepared at 48k", &tone(440.0, -6.0, 0.2), 512),
        &expect![[r#"
            at 96k — 1.00s at 96000Hz in blocks of 4096
              L in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              L out    0.0 pk  -12.7 rms  |---==========++++******#####%%%%%%@@@@@@|
              R in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              R out    0.0 pk  -12.7 rms  |---==========++++******#####%%%%%%@@@@@@|
              hash ade559c32660aea5
            longer than a crusher block — 0.20s at 96000Hz in blocks of 10000
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash fa7757a31a048985
            prepared at 48k — 0.20s at 48000Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash f3440faa61d08265
        "#]],
    );
}

/// The level of `hz` in `samples`, by Goertzel, as a sine's peak in dB.
fn tone_level(samples: &[f32], hz: f64, rate: f64) -> f64 {
    let w = 2.0 * core::f64::consts::PI * hz / rate;
    let coeff = 2.0 * libm::cos(w);
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &x in samples {
        let s0 = f64::from(x) + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    db(2.0 * power.sqrt() / samples.len() as f64)
}

/// The anti-aliased mode, which runs through bloop's own oversampler and so
/// has no original to match: what it lets through, how late, and how much
/// aliasing it keeps out next to dirty mode.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn bitcrusher_anti_aliased() {
    use std::fmt::Write as _;
    let rate = 44100.0;
    let crush = |settings: &[(&str, f32)], signal: &Source| {
        let mut p = Program::new(Kind::BitCrusher);
        for &(name, value) in settings {
            p = p.set(name, value);
        }
        let p = p.play("", signal, 512);
        run_rust(&p).remove(0).output.left
    };
    let mut out = String::new();

    let clean = crush(&[], &tone(1000.0, -6.0, 1.0));
    let _ = writeln!(
        out,
        "16 bits, 48 kHz, 1 kHz at -6.0 dB comes out at {:.1} dB",
        tone_level(&clean[4410..], 1000.0, rate)
    );
    let click = crush(&[], &impulse(0.0, 0.1));
    let late = click
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(i, _)| i);
    let _ = writeln!(out, "a click comes out {late} samples late");

    // A 7 kHz tone crushed to 4 bits makes odd harmonics at 21 and 35 kHz;
    // at 44.1 kHz those fold back to 23.1 and 9.1 kHz.
    let settings = [("Bits", 4.0), ("Rate", 48000.0)];
    let naive = crush(
        &[("Bits", 4.0), ("Rate", 48000.0), ("Dirty", 1.0)],
        &tone(7000.0, -1.0, 1.0),
    );
    let smooth = crush(&settings, &tone(7000.0, -1.0, 1.0));
    for (label, hz) in [("the tone", 7000.0), ("9.1 kHz alias", 9100.0)] {
        let _ = writeln!(
            out,
            "4 bits, {label}: dirty {:.1} dB, anti-aliased {:.1} dB",
            tone_level(&naive[4410..], hz, rate),
            tone_level(&smooth[4410..], hz, rate)
        );
    }
    expect![[r#"
        16 bits, 48 kHz, 1 kHz at -6.0 dB comes out at -6.0 dB
        a click comes out 6 samples late
        4 bits, the tone: dirty -1.0 dB, anti-aliased -1.0 dB
        4 bits, 9.1 kHz alias: dirty -43.8 dB, anti-aliased -151.6 dB
    "#]]
    .assert_eq(&out);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn plate_reverb() {
    check(
        &Program::new(Kind::PlateReverb)
            .play("defaults", &drums(), 512)
            .play("tail", &silence(2.0), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -8.1 pk  -25.0 rms  |%##****+++%%#****+++%##***++++%###***+++|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -8.6 pk  -25.1 rms  |%###****++%##****+++%##****++=%##****+++|
              hash 8983dc98a806a2e1
            tail — 2.00s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out  -31.5 pk  -53.1 rms  |===----::::...                          |
              R in    zero pk   zero rms  |                                        |
              R out  -31.9 pk  -53.2 rms  |===----:::.....                         |
              hash bd2f2fdc426808a9
        "#]],
    );
    check(
        &Program::new(Kind::PlateReverb)
            .at(48000.0)
            .set("Mix", 100.0)
            .set("Predelay", 50.0)
            .set("Decay", 90.0)
            .set("Size", 200.0)
            .set("Lowpass", 20000.0)
            .set("Damping", 16.0)
            .play("huge, dark, wet, late", &impulse(-6.0, 4.0), 256)
            .reset(44100.0)
            .play("after a reset to 44.1kHz", &impulse(-6.0, 1.0), 256),
        &expect![[r#"
            huge, dark, wet, late — 4.00s at 48000Hz in blocks of 256
              L in    -6.0 pk  -58.8 rms  |%                                       |
              L out   -8.5 pk  -41.7 rms  |%#*+=--:...:.......   .     . .         |
              R in    -6.0 pk  -58.8 rms  |%                                       |
              R out   -8.5 pk  -41.7 rms  |%#*+==-...:......    ...      ..        |
              hash c91fa353e0567dec
            after a reset to 44.1kHz — 1.00s at 44100Hz in blocks of 256
              L in    -6.0 pk  -52.4 rms  |%                                       |
              L out  -16.7 pk  -44.3 rms  |  ***##***++=+=-=--:::.:... .           |
              R in    -6.0 pk  -52.4 rms  |%                                       |
              R out  -17.5 pk  -44.3 rms  |  ####*****=+===--::::......            |
              hash c5ab8329c04c2997
        "#]],
    );
    check(
        &Program::new(Kind::PlateReverb)
            .set("Mix", 100.0)
            .set("Size", 10.0)
            .set("Decay", 95.0)
            .play("the smallest plate, the longest decay", &drums(), 128)
            .play("its tail", &silence(2.0), 128),
        &expect![[r#"
            the smallest plate, the longest decay — 2.00s at 44100Hz in blocks of 128
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out    3.4 pk  -13.2 rms  |@@@@@@%%%%@@@@@@@%%%@@@@@@%%%%@@@@@@@%%%|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out    3.7 pk  -13.2 rms  |@@@@@@%%%#@@@@@@%@%%@@@@@@%%%%@@@@@@%%%%|
              hash 27fb01acd271ac7c
            its tail — 2.00s at 44100Hz in blocks of 128
              L in    zero pk   zero rms  |                                        |
              L out  -12.5 pk  -33.6 rms  |####****++++++=====-----:::::.:.. .     |
              R in    zero pk   zero rms  |                                        |
              R out  -12.7 pk  -33.5 rms  |#####****+++++=====----:::::::......    |
              hash 70eb8bcc9625c1b0
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn distortion() {
    check(
        &Program::new(Kind::Distortion)
            .play("defaults", &tone(220.0, -6.0, 1.0), 512)
            .play(
                "digital black comes out as noise far under -120dB",
                &silence(0.5),
                512,
            ),
        &expect![[r#"
            defaults — 1.00s at 44100Hz in blocks of 512
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R out   -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash 12563c17754bb027
            digital black comes out as noise far under -120dB — 0.50s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out   -inf pk   -inf rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out   -inf pk   -inf rms  |                                        |
              hash 3c0396e0570fed62
        "#]],
    );
    check(
        &Program::new(Kind::Distortion)
            .set("Density", 4.0)
            .set("Highpass", 30.0)
            .set("Output", 60.0)
            .set("Mix", 70.0)
            .play("four passes, highpassed, part wet", &drums(), 256),
        &expect![[r#"
            four passes, highpassed, part wet — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -4.9 pk   -8.9 rms  |%%%%%%%%%#@%%%%%%%%#@%%%%%%%%#@%%%%%%%%#|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -4.9 pk   -8.9 rms  |%%%%%%%%%#@%%%%%%%%#@%%%%%%%%#@%%%%%%%%#|
              hash f87d9d8a719b95e4
        "#]],
    );
    check(
        &Program::new(Kind::Distortion)
            .at(96000.0)
            .set("Density", -1.0)
            .play("starved at 96kHz", &ramp(110.0, -40.0, 0.0, 1.0), 512)
            .set("Highpass", 100.0)
            .play(
                "and highpassed as hard as it goes",
                &ramp(110.0, -40.0, 0.0, 1.0),
                512,
            ),
        &expect![[r#"
            starved at 96kHz — 1.00s at 96000Hz in blocks of 512
              L in    -0.1 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              L out   -0.1 pk  -16.1 rms  |-           ...:::---===+++***###%%%@@@@|
              R in    -0.1 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              R out   -0.1 pk  -16.1 rms  |-           ...:::---===+++***###%%%@@@@|
              hash 3e1f3366cb797880
            and highpassed as hard as it goes — 1.00s at 96000Hz in blocks of 512
              L in    -0.1 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              L out  -69.0 pk  -85.6 rms  |                                        |
              R in    -0.1 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              R out  -69.0 pk  -85.6 rms  |                                        |
              hash 99818790b97ff0d9
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn transient_shaper() {
    check(
        &Program::new(Kind::TransientShaper).play("defaults", &drums(), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -6.1 pk  -24.1 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -6.2 pk  -24.2 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              hash f8efc7ddb5610eed
        "#]],
    );
    check(
        &Program::new(Kind::TransientShaper)
            .set("Attack", 4.0)
            .set("Smoothing", 0.0)
            .play("sharpened, snappy", &drums(), 256)
            .set("Attack", 0.0)
            .set("Sustain", -8.0)
            .set("Smoothing", 100.0)
            .play("tails cut, smooth", &drums(), 256),
        &expect![[r#"
            sharpened, snappy — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out    3.6 pk  -21.0 rms  |%#*=-=:.  @#*=-=:.  @#*=-=:.  @#*=-=:.  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out    3.3 pk  -21.1 rms  |%#*=-=:.  @#*=-=:.  @#*=-=:.  @#*=-=:.  |
              hash 2435b61eafb4a670
            tails cut, smooth — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out    3.3 pk  -24.5 rms  |@#*:      %#+:      ##+:      ##+:      |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out    3.0 pk  -24.5 rms  |@#*:      %#+:      ##+:      ##+:      |
              hash 846767d55a69ceed
        "#]],
    );
    check(
        &Program::new(Kind::TransientShaper)
            .set("Sustain", 8.0)
            .set("Low Cut", 40.0)
            .set("High Cut", 200.0)
            .set("Detector", 100.0)
            .set("Process", 0.0)
            .set("Output", 6.0)
            .play("40-200Hz band, side heard, mid shaped", &drums(), 512),
        &expect![[r#"
            40-200Hz band, side heard, mid shaped — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out    0.2 pk  -18.2 rms  |@%#+=+-:. @%#+=+-:. @%#+=+-:  @%#+=+-:  |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out    0.0 pk  -18.3 rms  |@%#+=+-:. @%#+=+-:. @%#+=+-:  @%#+=+-:  |
              hash 6ad8abe614c0ddd3
        "#]],
    );
    check(
        &Program::new(Kind::TransientShaper)
            .at(32000.0)
            .set("Attack", -4.0)
            .set("High Cut", 16000.0)
            .play("a high cut past the band's top is off", &drums(), 512),
        &expect![[r#"
            a high cut past the band's top is off — 2.00s at 32000Hz in blocks of 512
              L in    -6.2 pk  -23.7 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -9.4 pk  -26.1 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R in    -6.1 pk  -23.7 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -9.5 pk  -26.2 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              hash 6b44481827eb2df6
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn maximizer() {
    check(
        &Program::new(Kind::Maximizer)
            .play("a sample at a time", &hits(-6.0, 30.0, 0.1, 0.02, 3), 1)
            .play(
                "in blocks that are not a power of two",
                &hits(-6.0, 30.0, 0.1, 0.02, 3),
                333,
            ),
        &expect![[r#"
            a sample at a time — 0.30s at 44100Hz in blocks of 1
              L in    -6.3 pk  -22.5 rms  |%%##**++==--:---:..       %%%##**+==--::|
              L out   -0.6 pk  -16.8 rms  |      @@@%%##*++==-===--::..     @@%%##*|
              R in    -6.4 pk  -22.5 rms  |%%##**++==-::---:..       %%%##**+==--::|
              R out   -0.7 pk  -16.8 rms  |      @@@%%#**++==-===--::..     @@%%##*|
              hash 9925f79c722f40ba
            in blocks that are not a power of two — 0.30s at 44100Hz in blocks of 333
              L in    -6.3 pk  -22.5 rms  |%%##**++==--:---:..       %%%##**+==--::|
              L out   -0.6 pk  -16.8 rms  |*++==-@@@%%##*++==-===--::..     @@%%##*|
              R in    -6.4 pk  -22.5 rms  |%%##**++==-::---:..       %%%##**+==--::|
              R out   -0.7 pk  -16.8 rms  |*++==-@@@%%#**++==-===--::..     @@%%##*|
              hash dfbd252078d18f1b
        "#]],
    );
    check(
        &Program::new(Kind::Maximizer)
            .play("defaults", &drums(), 512)
            .play("the last 50ms come out", &silence(0.1), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -0.4 pk  -18.1 rms  |+@%#+==-: *@%*+==-. #@%#+==-: %@%#+==-. |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -0.4 pk  -18.2 rms  |@@%#+==-: @@%*+==-: @@%#+==-. @@%*+==-: |
              hash 38ec27cea73ca089
            the last 50ms come out — 0.10s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out  -65.2 pk  -76.5 rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out  -65.1 pk  -76.5 rms  |                                        |
              hash c961cda2b11a3255
        "#]],
    );
    check(
        &Program::new(Kind::Maximizer)
            .set("Threshold", -24.0)
            .set("Ceiling", 0.0)
            .set("Release", 0.0)
            .play(
                "24dB up, every half cycle alone",
                &ramp(440.0, -40.0, 0.0, 1.0),
                256,
            ),
        &expect![[r#"
            24dB up, every half cycle alone — 1.00s at 44100Hz in blocks of 256
              L in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              L out    0.0 pk   -4.8 rms  | :####%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              R out    0.0 pk   -4.8 rms  | :####%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash f761e2f571316945
        "#]],
    );
    check(
        &Program::new(Kind::Maximizer)
            .set("Threshold", -12.0)
            .set("Release", 1000.0)
            .play(
                "half cycles longer than it holds, held to the ceiling",
                &tone(5.0, -3.0, 1.0),
                512,
            )
            .reset(96000.0)
            .play("after a reset to 96kHz", &drums(), 512),
        &expect![[r#"
            half cycles longer than it holds, held to the ceiling — 1.00s at 44100Hz in blocks of 512
              L in    -3.0 pk   -6.0 rms  |%@@%%@@%@@@%@@@%@@@%@@@%@@@%@@@%@@@%@@@%|
              L out   -0.3 pk   -2.2 rms  |  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -3.0 pk   -6.0 rms  |%@@%%@@%@@@%@@@%@@@%@@@%@@@%@@@%@@@%@@@%|
              R out   -0.3 pk   -2.2 rms  |  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 4b7990080750e30d
            after a reset to 96kHz — 2.00s at 96000Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -0.3 pk  -17.8 rms  |*@%#+=+-:.@@%#+=+-:.@@%#+=+-: @@%#+=+-:.|
              R in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -0.3 pk  -17.8 rms  |@@%#+=+-: @@%#+=+-: @@%#+=+-:.%@%#+=+-: |
              hash db0ee915a34fe288
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn exciter() {
    check(
        &Program::new(Kind::Exciter).play("defaults", &sweep(100.0, 16000.0, -12.0, 2.0), 512),
        &expect![[r#"
            defaults — 2.00s at 44100Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out  -11.0 pk  -14.9 rms  |################################%%%%%%%%|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out  -11.0 pk  -14.9 rms  |################################%%%%%%%%|
              hash a993c8ba70f58849
        "#]],
    );
    check(
        &Program::new(Kind::Exciter)
            .set("Frequency", 1000.0)
            .set("Drive", 8.0)
            .set("Harmonics", 100.0)
            .set("Mix", 100.0)
            .play("everything up", &drums(), 256),
        &expect![[r#"
            everything up — 2.00s at 44100Hz in blocks of 256
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out    5.4 pk   -2.7 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out    5.2 pk   -2.7 rms  |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash 6607659c0a890644
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn soft_clipper() {
    check(
        &Program::new(Kind::SoftClipper).play(
            "defaults: a hard ceiling at 0dB",
            &ramp(220.0, -30.0, 12.0, 1.0),
            512,
        ),
        &expect![[r#"
            defaults: a hard ceiling at 0dB — 1.00s at 44100Hz in blocks of 512
              L in    12.0 pk   -0.9 rms  |+++++******######%%%%%@@@@@@@@@@@@@@@@@@|
              L out    0.0 pk   -5.7 rms  |+++++******######%%%%%@@@@@@@@@@@@@@@@@@|
              R in    12.0 pk   -0.9 rms  |+++++******######%%%%%@@@@@@@@@@@@@@@@@@|
              R out    0.0 pk   -5.7 rms  |+++++******######%%%%%@@@@@@@@@@@@@@@@@@|
              hash 9f3cce467c299131
        "#]],
    );
    check(
        &Program::new(Kind::SoftClipper)
            .set("Drive", 24.0)
            .set("Knee", 0.0)
            .set("Ceiling", -6.0)
            .play("hard clip at -6dB", &tone(220.0, -12.0, 0.5), 512),
        &expect![[r#"
            hard clip at -6dB — 0.50s at 44100Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out   -6.0 pk   -7.0 rms  |%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out   -6.0 pk   -7.0 rms  |%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
              hash b836da5c0f91026d
        "#]],
    );
    check(
        &Program::new(Kind::SoftClipper)
            .set("Drive", 12.0)
            .set("Knee", 100.0)
            .play(
                "a knee as wide as the drive",
                &ramp(220.0, -40.0, 0.0, 1.0),
                512,
            )
            .play("below 1e-10 is silence", &dc(1.0e-11, 0.1), 512),
        &expect![[r#"
            a knee as wide as the drive — 1.00s at 44100Hz in blocks of 512
              L in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              L out   -0.0 pk   -6.7 rms  |=+++******######%%%%%%%@@@@@@@@@@@@@@@@@|
              R in    -0.0 pk  -12.7 rms  |----======++++++******######%%%%%%@@@@@@|
              R out   -0.0 pk   -6.7 rms  |=+++******######%%%%%%%@@@@@@@@@@@@@@@@@|
              hash e5dcb9a876fc3171
            below 1e-10 is silence — 0.10s at 44100Hz in blocks of 512
              L in    -inf pk   -inf rms  |                                        |
              L out   zero pk   zero rms  |                                        |
              R in    -inf pk   -inf rms  |                                        |
              R out   zero pk   zero rms  |                                        |
              hash 55a0fca3f8fc0765
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn chorus() {
    check(
        &Program::new(Kind::Chorus).play(
            "defaults: 1 ms back, swept 1 ms deep three times a second, half wet",
            &tone(440.0, -12.0, 1.0),
            512,
        ),
        &expect![[r#"
            defaults: 1 ms back, swept 1 ms deep three times a second, half wet — 1.00s at 44100Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out   -9.6 pk  -16.6 rms  |%%%#*+*+*#%%%%%%##+**+##%%%%%%#+**+*#%%%|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out   -9.6 pk  -16.6 rms  |#+##%%%%%%#***+*#%%%%%%#*+*+*#%%%%%%##+*|
              hash b5435c6cdca2693b
        "#]],
    );
    check(
        &Program::new(Kind::Chorus)
            .set("Delay", 20.0)
            .set("Depth", 10.0)
            .set("Speed", 0.5)
            .reset(44100.0)
            .play(
                "a slow chorus at the widest, Width 50%: the two ears sweep opposite ways",
                &sweep(100.0, 8000.0, -12.0, 2.0),
                256,
            )
            .set("Width", 100.0)
            .play(
                "Width 100%, once it has settled: the sweeps a whole cycle apart, in step again",
                &sweep(100.0, 8000.0, -12.0, 2.0),
                256,
            )
            .play(
                "one ear in: the other lets out only the last passage's tail",
                &left_only(tone(220.0, -6.0, 0.5)),
                100,
            ),
        &expect![[r#"
            a slow chorus at the widest, Width 50%: the two ears sweep opposite ways — 2.00s at 44100Hz in blocks of 256
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out   -8.9 pk  -15.1 rms  |%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out   -8.9 pk  -15.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash 69a51d08689d269b
            Width 100%, once it has settled: the sweeps a whole cycle apart, in step again — 2.00s at 44100Hz in blocks of 256
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out   -8.9 pk  -14.9 rms  |%%%%#%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out   -8.9 pk  -15.0 rms  |%%%%#%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              hash 81529d035afc6cdb
            one ear in: the other lets out only the last passage's tail — 0.50s at 44100Hz in blocks of 100
              L in    -6.0 pk   -9.0 rms  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|
              L out   -3.0 pk  -10.4 rms  |@@#%%%@@@@@@@@@@@@@@%%%%###*+=+**#######|
              R in    zero pk   zero rms  |                                        |
              R out  -14.5 pk  -30.9 rms  |##                                      |
              hash 43500ed277d2f34e
        "#]],
    );
    check(
        &Program::new(Kind::Chorus)
            .set("Delay", 30.0)
            .set("Depth", 20.0)
            .set("Speed", 10.0)
            .set("Width", 100.0)
            .set("Mix", 100.0)
            .reset(44100.0)
            .play(
                "every knob at its top: all wet, as far as 50 ms back",
                &drums(),
                512,
            )
            .play("and what it still holds after", &silence(0.1), 512)
            .reset(44100.0)
            .play(
                "prepared again at the same rate: nothing left",
                &silence(0.1),
                512,
            ),
        &expect![[r#"
            every knob at its top: all wet, as far as 50 ms back — 2.00s at 44100Hz in blocks of 512
              L in    -6.1 pk  -23.8 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              L out   -3.2 pk  -22.2 rms  |@@#*==-:  @@#*==-:  @@#*==-:. @@#*==-:. |
              R in    -6.1 pk  -23.9 rms  |%#*=--:.  %#*=--:.  %#*=--:.  %#*=--:.  |
              R out   -3.3 pk  -22.3 rms  |@@#*===:. @@#*===:  @@#*==-:. @%#*==-:. |
              hash d425d2557fefd912
            and what it still holds after — 0.10s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out  -70.8 pk  -85.8 rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out  -71.3 pk  -85.9 rms  |                                        |
              hash 31b27063013ee182
            prepared again at the same rate: nothing left — 0.10s at 44100Hz in blocks of 512
              L in    zero pk   zero rms  |                                        |
              L out   zero pk   zero rms  |                                        |
              R in    zero pk   zero rms  |                                        |
              R out   zero pk   zero rms  |                                        |
              hash 55a0fca3f8fc0765
        "#]],
    );
    check(
        &Program::new(Kind::Chorus)
            .set("Delay", 0.1)
            .set("Depth", 0.1)
            .set("Speed", 0.1)
            .set("Width", 0.0)
            .set("Mix", 100.0)
            .reset(44100.0)
            .play(
                "every knob at its bottom but Mix: all wet, 0.1 to 0.2 ms back",
                &tone(440.0, -12.0, 0.5),
                512,
            ),
        &expect![[r#"
            every knob at its bottom but Mix: all wet, 0.1 to 0.2 ms back — 0.50s at 44100Hz in blocks of 512
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out  -12.0 pk  -15.0 rms  |##%##%#%#%##%#%############%#%#%#######%|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out  -12.0 pk  -15.0 rms  |##%##%#%#%##%#%############%#%#%#######%|
              hash ba34aa97a7e0de35
        "#]],
    );
    check(
        &Program::new(Kind::Chorus)
            .at(48000.0)
            .set("Depth", 20.0)
            .set("Mix", 100.0)
            .play(
                "at 48 kHz, a sample at a time",
                &tone(330.0, -12.0, 0.25),
                1,
            )
            .reset(96000.0)
            .play(
                "prepared again at 96 kHz with the depth moved",
                &tone(330.0, -12.0, 0.25),
                333,
            ),
        &expect![[r#"
            at 48 kHz, a sample at a time — 0.25s at 48000Hz in blocks of 1
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out   -9.0 pk  -15.0 rms  |%%%%%#%%###################%########%###|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out  -11.5 pk  -15.4 rms  |######%%################################|
              hash ef4c2376eaa039e0
            prepared again at 96 kHz with the depth moved — 0.25s at 96000Hz in blocks of 333
              L in   -12.0 pk  -15.0 rms  |########################################|
              L out  -12.0 pk  -15.4 rms  |   #############%#######################|
              R in   -12.0 pk  -15.0 rms  |########################################|
              R out  -12.0 pk  -15.0 rms  |########################################|
              hash 3baa14d5f2505f7f
        "#]],
    );
}

/// Mix at 0 is the input, bit for bit: the dry gain is exactly 1 and the
/// wet exactly 0, so nothing of the line gets in.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_dry_chorus_is_the_input() {
    let program = Program::new(Kind::Chorus)
        .set("Mix", 0.0)
        .set("Depth", 20.0)
        .reset(44100.0)
        .play("dry", &drums(), 512);
    let played = run_rust(&program);
    let same = played
        .iter()
        .all(|p| p.input.left == p.output.left && p.input.right == p.output.right);
    assert!(same, "a chorus at Mix 0 changed its input");
}
