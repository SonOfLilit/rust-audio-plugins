//! The port against the original.
//!
//! Each scenario plays a program — parameters, modulation, presets, MIDI —
//! through [`WavetablePlugin`] and snapshots what came out: per passage,
//! the level of each channel and a hash of the samples. With `--features
//! gold` the same program also runs through Wavetable's own
//! `WavetableAudioProcessor`, compiled from `vendor/wavetable` with JUCE
//! and Gin (`tests/cpp_harness`), and the two must agree bit for bit.
//!
//! Both read the same band-limited frames: the port builds them, and the
//! harness is handed them. Building them is compared on its own
//! (`table_builds_match_the_original`), to within a millionth of full scale, because the
//! original runs JUCE's FFT and the port its own.

#![expect(
    clippy::panic,
    reason = "a scenario that names a parameter, table or preset that does not exist is a broken test"
)]
#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

use bloop_events::{EventBuffer, Expression, NoteEvent, NoteId};
use bloop_note::Note;
use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_wavetable::{bank, modmatrix, params, presets, WavetablePlugin, PARAM_PRESET};
use expect_test::{expect, Expect};
use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Step {
    /// Parameter `uid` to a user value.
    Set(&'static str, f32),
    /// Modulation source `src` (by id) to destination `uid`.
    Mod(&'static str, &'static str, f32),
    /// Factory preset `index`, wavetables and all.
    Preset(usize),
    /// A preset file, read as the user's own are (`preset_file`) — the
    /// original reads it as `gin::Program` does (`presetxml`).
    PresetFile(&'static str),
    /// The two oscillators' tables, by catalogue name.
    Tables(&'static str, &'static str),
    Tempo(f64),
    /// MIDI at `offset` samples into the next passage.
    Midi(u32, Midi),
    /// Render `seconds` in blocks of `block`.
    Play {
        label: &'static str,
        seconds: f64,
        block: usize,
    },
}

#[derive(Clone, Copy, Debug)]
enum Midi {
    On(u8, u8),
    Off(u8),
    Bend(u16),
    Cc(u8, u8),
    /// A note with an id of its own, which expression can name.
    OnAs(u8, u8, u32),
    OffAs(u8, u32),
    /// One note's expression, by its id.
    Express(u32, Expression),
}

impl Midi {
    /// As bloop hands it to the Plug-in, at `time`.
    fn event(self, time: u32) -> NoteEvent {
        let on = |note: u8, velocity: u8, id: NoteId| NoteEvent::NoteOn {
            time,
            channel: 0,
            note: Note::new(note),
            velocity: f32::from(velocity) / 127.0,
            id,
        };
        let off = |note: u8, id: NoteId| NoteEvent::NoteOff {
            time,
            channel: 0,
            note: Note::new(note),
            id,
        };
        match self {
            Midi::On(note, velocity) => on(note, velocity, NoteId::NONE),
            Midi::OnAs(note, velocity, id) => on(note, velocity, NoteId(id)),
            Midi::Off(note) => off(note, NoteId::NONE),
            Midi::OffAs(note, id) => off(note, NoteId(id)),
            Midi::Bend(value) => NoteEvent::PitchBend {
                time,
                channel: 0,
                value: f32::from(value) / 16383.0,
            },
            Midi::Cc(cc, value) => NoteEvent::CC {
                time,
                channel: 0,
                cc,
                value: f32::from(value) / 127.0,
            },
            Midi::Express(id, expression) => NoteEvent::NoteExpression {
                time,
                id: NoteId(id),
                expression,
            },
        }
    }
}

#[derive(Clone, Debug)]
struct Program {
    sample_rate: f64,
    steps: Vec<Step>,
}

impl Program {
    fn new() -> Self {
        Self {
            sample_rate: 48000.0,
            steps: Vec::new(),
        }
    }

    fn at(mut self, rate: f64) -> Self {
        self.sample_rate = rate;
        self
    }

    fn set(mut self, uid: &'static str, value: f32) -> Self {
        self.steps.push(Step::Set(uid, value));
        self
    }

    fn modulate(mut self, src: &'static str, uid: &'static str, depth: f32) -> Self {
        self.steps.push(Step::Mod(src, uid, depth));
        self
    }

    fn preset(mut self, name: &str) -> Self {
        let index = presets::PRESETS
            .iter()
            .position(|p| p.name == name)
            .unwrap_or_else(|| panic!("no preset {name}"));
        self.steps.push(Step::Preset(index));
        self
    }

    fn preset_file(mut self, xml: &'static str) -> Self {
        self.steps.push(Step::PresetFile(xml));
        self
    }

    fn tables(mut self, osc1: &'static str, osc2: &'static str) -> Self {
        self.steps.push(Step::Tables(osc1, osc2));
        self
    }

    fn tempo(mut self, bpm: f64) -> Self {
        self.steps.push(Step::Tempo(bpm));
        self
    }

    fn on(mut self, at: u32, note: u8, velocity: u8) -> Self {
        self.steps.push(Step::Midi(at, Midi::On(note, velocity)));
        self
    }

    fn off(mut self, at: u32, note: u8) -> Self {
        self.steps.push(Step::Midi(at, Midi::Off(note)));
        self
    }

    fn bend(mut self, at: u32, value: u16) -> Self {
        self.steps.push(Step::Midi(at, Midi::Bend(value)));
        self
    }

    fn cc(mut self, at: u32, cc: u8, value: u8) -> Self {
        self.steps.push(Step::Midi(at, Midi::Cc(cc, value)));
        self
    }

    /// A note struck under id `id`.
    fn on_as(mut self, at: u32, note: u8, velocity: u8, id: u32) -> Self {
        self.steps
            .push(Step::Midi(at, Midi::OnAs(note, velocity, id)));
        self
    }

    fn off_as(mut self, at: u32, note: u8, id: u32) -> Self {
        self.steps.push(Step::Midi(at, Midi::OffAs(note, id)));
        self
    }

    /// Note `id` bent by `semitones`.
    fn note_bend(mut self, at: u32, id: u32, semitones: f32) -> Self {
        self.steps.push(Step::Midi(
            at,
            Midi::Express(id, Expression::Bend(semitones)),
        ));
        self
    }

    fn note_timbre(mut self, at: u32, id: u32, value: f32) -> Self {
        self.steps
            .push(Step::Midi(at, Midi::Express(id, Expression::Timbre(value))));
        self
    }

    fn note_pressure(mut self, at: u32, id: u32, value: f32) -> Self {
        self.steps.push(Step::Midi(
            at,
            Midi::Express(id, Expression::Pressure(value)),
        ));
        self
    }

    fn play(self, label: &'static str, seconds: f64) -> Self {
        self.play_in(label, seconds, 512)
    }

    fn play_in(mut self, label: &'static str, seconds: f64, block: usize) -> Self {
        self.steps.push(Step::Play {
            label,
            seconds,
            block,
        });
        self
    }

    /// A note held for `hold` seconds and left to ring for `tail`.
    fn note(self, note: u8, velocity: u8, hold: f64, tail: f64) -> Self {
        self.on(0, note, velocity)
            .play("held", hold)
            .off(0, note)
            .play("released", tail)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Signal {
    left: Vec<f32>,
    right: Vec<f32>,
}

fn samples(seconds: f64, rate: f64) -> usize {
    (seconds * rate).round() as usize
}

fn table_index(name: &str) -> u16 {
    bank::index_of(name).unwrap_or_else(|| panic!("no wavetable {name}")) as u16
}

/// Where the port's mod slot numbers a destination: 1 on, 0 for none.
fn destination_slot(uid: &str) -> f32 {
    let param = params::index_of(uid).unwrap_or_else(|| panic!("no parameter {uid}"));
    let dst = modmatrix::DESTINATION_OF[param];
    assert!(dst >= 0, "{uid} is not a modulation destination");
    (dst + 1) as f32
}

fn source_slot(id: &str) -> f32 {
    (modmatrix::source_named(id).unwrap_or_else(|| panic!("no mod source {id}")) + 1) as f32
}

fn set(plugin: &mut WavetablePlugin, uid: &str, value: f32) {
    let index = params::index_of(uid).unwrap_or_else(|| panic!("no parameter {uid}"));
    plugin.set_param(index, value);
}

/// The tables the plugin's oscillators are set to, built, and handed over.
fn hand_tables(plugin: &mut WavetablePlugin, rate: f64) {
    let tables = bank::tables_for(plugin.table_indices(), rate, bank::Quality::Render);
    plugin.set_tables(Some(tables));
}

fn retimed(event: NoteEvent, at: u32) -> NoteEvent {
    match event {
        NoteEvent::NoteOn {
            channel,
            note,
            velocity,
            id,
            ..
        } => NoteEvent::NoteOn {
            time: at,
            channel,
            note,
            velocity,
            id,
        },
        NoteEvent::NoteOff {
            channel, note, id, ..
        } => NoteEvent::NoteOff {
            time: at,
            channel,
            note,
            id,
        },
        NoteEvent::NoteExpression { id, expression, .. } => NoteEvent::NoteExpression {
            time: at,
            id,
            expression,
        },
        NoteEvent::Aftertouch {
            channel,
            note,
            pressure,
            ..
        } => NoteEvent::Aftertouch {
            time: at,
            channel,
            note,
            pressure,
        },
        NoteEvent::PitchBend { channel, value, .. } => NoteEvent::PitchBend {
            time: at,
            channel,
            value,
        },
        NoteEvent::CC {
            channel, cc, value, ..
        } => NoteEvent::CC {
            time: at,
            channel,
            cc,
            value,
        },
        NoteEvent::Choke { channel, note, .. } => NoteEvent::Choke {
            time: at,
            channel,
            note,
        },
    }
}

fn run_rust(program: &Program) -> Vec<(&'static str, Signal)> {
    let rate = program.sample_rate;
    let mut plugin = WavetablePlugin::at(rate);
    hand_tables(&mut plugin, rate);
    let mut tempo = 120.0;
    let mut slots = 0;
    let mut pending: Vec<NoteEvent> = Vec::new();
    let mut played = Vec::new();
    for step in &program.steps {
        match *step {
            Step::Set(uid, value) => set(&mut plugin, uid, value),
            Step::Mod(src, uid, depth) => {
                let at = params::id::MOD1_SRC + slots * params::ROUTE_PARAMS;
                plugin.set_param(at, source_slot(src));
                plugin.set_param(at + 1, destination_slot(uid));
                plugin.set_param(at + 2, depth);
                slots += 1;
            }
            Step::Preset(index) => {
                plugin.set_param(PARAM_PRESET, index as f32);
                hand_tables(&mut plugin, rate);
                slots = 0;
            }
            Step::PresetFile(xml) => {
                let (index, _) = bloop_plugin_wavetable::preset_file::import(xml.as_bytes())
                    .unwrap_or_else(|e| panic!("the preset reads: {e}"));
                plugin.set_param(PARAM_PRESET, index as f32);
                hand_tables(&mut plugin, rate);
                slots = 0;
            }
            Step::Tables(a, b) => {
                plugin.set_param(params::id::WT1, f32::from(table_index(a)));
                plugin.set_param(params::id::WT2, f32::from(table_index(b)));
                hand_tables(&mut plugin, rate);
            }
            Step::Tempo(bpm) => tempo = bpm,
            Step::Midi(time, midi) => pending.push(midi.event(time)),
            Step::Play {
                label,
                seconds,
                block,
            } => {
                // The engine hands a Plug-in its events in time order, and
                // the original's MidiBuffer keeps them so.
                assert!(
                    pending.windows(2).all(|w| w[0].time() <= w[1].time()),
                    "{label}: a passage's MIDI is written in time order"
                );
                let n = samples(seconds, rate);
                let mut out = Signal {
                    left: vec![0.0; n],
                    right: vec![0.0; n],
                };
                let mut start = 0;
                for (l, r) in out.left.chunks_mut(block).zip(out.right.chunks_mut(block)) {
                    let len = l.len();
                    let mut events = EventBuffer::new();
                    for e in &pending {
                        let t = e.time() as usize;
                        if (start..start + len).contains(&t) {
                            events.push(retimed(*e, (t - start) as u32));
                        }
                    }
                    let mut outputs: [&mut [f32]; 2] = [l, r];
                    let mut context = ProcessContext {
                        outputs: &mut outputs,
                        input_events: &events,
                        sample_rate: rate as f32,
                        tempo,
                        is_playing: true,
                    };
                    plugin.process(&mut context);
                    start += len;
                }
                pending.clear();
                played.push((label, out));
            }
        }
    }
    played
}

// ---------------------------------------------------------------------------
// What a passage sounded like
// ---------------------------------------------------------------------------

fn db(x: f64) -> String {
    if x <= 0.0 {
        "-inf".to_string()
    } else {
        format!("{:.1}", 20.0 * x.log10())
    }
}

fn level(samples: &[f32]) -> String {
    let peak = samples
        .iter()
        .fold(0.0f64, |m, s| m.max(f64::from(s.abs())));
    let rms = if samples.is_empty() {
        0.0
    } else {
        (samples.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
    };
    format!("peak {} rms {}", db(peak), db(rms))
}

fn fnv(signal: &Signal) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in signal.left.iter().chain(&signal.right) {
        for b in s.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn summary(played: &[(&'static str, Signal)]) -> String {
    let mut out = String::new();
    for (label, signal) in played {
        let nan = signal
            .left
            .iter()
            .chain(&signal.right)
            .any(|s| !s.is_finite());
        let _ = writeln!(
            out,
            "{label}: L {} | R {}{} #{:016x}",
            level(&signal.left),
            level(&signal.right),
            if nan { " NOT FINITE" } else { "" },
            fnv(signal)
        );
    }
    out
}

fn check(program: &Program, expect: &Expect) {
    let rust = run_rust(program);
    #[cfg(feature = "gold")]
    compare(program, &rust, &cpp::run(program));
    assert_summary(expect, &summary(&rust));
}

/// The snapshot is the default build's, whose maths is `libm`'s. Under
/// `gold` the shared Gin ports compute with glibc's, as the C++ does, so
/// the samples — and their hashes — differ in the last bits; the levels
/// must still agree.
fn assert_summary(expect: &Expect, actual: &str) {
    #[cfg(feature = "gold")]
    {
        let without_hashes = |text: &str| -> Vec<String> {
            text.lines()
                .map(|l| l.trim().split(" #").next().unwrap_or("").to_string())
                .filter(|l| !l.is_empty())
                .collect()
        };
        assert_eq!(
            without_hashes(expect.data()),
            without_hashes(actual),
            "the summary under gold differs from the snapshot beyond its hashes"
        );
    }
    #[cfg(not(feature = "gold"))]
    expect.assert_eq(actual);
}

#[cfg(feature = "gold")]
fn compare(program: &Program, rust: &[(&'static str, Signal)], cpp: &[Signal]) {
    assert_eq!(rust.len(), cpp.len(), "passage count");
    for ((label, ours), theirs) in rust.iter().zip(cpp) {
        for (channel, a, b) in [
            ("L", &ours.left, &theirs.left),
            ("R", &ours.right, &theirs.right),
        ] {
            assert_eq!(a.len(), b.len(), "{label} {channel}: length");
            if let Some(i) = (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits()) {
                let differing = (0..a.len())
                    .filter(|&i| a[i].to_bits() != b[i].to_bits())
                    .count();
                let worst = (0..a.len())
                    .map(|i| (f64::from(a[i]) - f64::from(b[i])).abs())
                    .fold(0.0, f64::max);
                panic!(
                    "{label} {channel}: first differs at sample {i} of {} \
                     (port {:e}, original {:e}); {differing} differ, worst by {worst:e}\n{program:?}",
                    a.len(),
                    a[i],
                    b[i]
                );
            }
        }
    }
}

/// [`compare`], allowing each sample to stray by up to `within`.
#[cfg(feature = "gold")]
fn compare_within(program: &Program, rust: &[(&'static str, Signal)], cpp: &[Signal], within: f64) {
    assert_eq!(rust.len(), cpp.len(), "passage count");
    for ((label, ours), theirs) in rust.iter().zip(cpp) {
        for (channel, a, b) in [
            ("L", &ours.left, &theirs.left),
            ("R", &ours.right, &theirs.right),
        ] {
            assert_eq!(a.len(), b.len(), "{label} {channel}: length");
            let worst = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                .fold(0.0, f64::max);
            assert!(
                worst <= within,
                "{label} {channel}: strays by {worst:e}\n{program:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The original
// ---------------------------------------------------------------------------

#[cfg(feature = "gold")]
#[expect(
    clippy::expect_used,
    reason = "a harness that cannot build or run the original has no one to hand the error to but the test"
)]
mod cpp {
    use super::{hand_tables, samples, set, source_slot, table_index, Midi, Program, Signal, Step};
    use bloop_plugin_api::Plugin as _;
    use bloop_plugin_wavetable::dsp::tables::Wavetable;
    use bloop_plugin_wavetable::mpe;
    use bloop_plugin_wavetable::{bank, params, presets, WavetablePlugin, PARAM_PRESET};
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::OnceLock;
    use std::time::SystemTime;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// A vendored checkout: the workspace's own, or, from a git worktree
    /// whose submodules are not checked out, the one `BLOOP_GOLD_VENDOR`
    /// names.
    fn vendor(name: &str) -> PathBuf {
        let own = root().join("../../vendor").join(name);
        let path = match std::env::var_os("BLOOP_GOLD_VENDOR") {
            Some(dir) if !own.join("modules").exists() && !own.join("plugin").exists() => {
                PathBuf::from(dir).join(name)
            }
            _ => own,
        };
        assert!(
            path.join(".git").exists(),
            "vendor/{name} is not checked out. The gold tests compile the original from it:\n  \
             git -c submodule.vendor/{name}.update=checkout submodule update --init vendor/{name}"
        );
        path
    }

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

    const JUCE_UNITS: [&str; 10] = [
        "juce_core.cpp",
        "juce_core_CompilationTime.cpp",
        "juce_core_zlib.c",
        "juce_events.cpp",
        "juce_data_structures.cpp",
        "juce_audio_basics.cpp",
        "juce_audio_formats.cpp",
        "juce_audio_formats_flac_1.c",
        "juce_audio_formats_flac_2.c",
        "juce_dsp.cpp",
    ];

    /// Wavetable's own units, compiled unmodified.
    const WAVETABLE_UNITS: [&str; 8] = [
        "PluginProcessor.cpp",
        "WavetableVoice.cpp",
        "FX/DeRez2.cpp",
        "FX/DeRez2Proc.cpp",
        "FX/FireAmp.cpp",
        "FX/FireAmpProc.cpp",
        "FX/GrindAmp.cpp",
        "FX/GrindAmpProc.cpp",
    ];

    fn write(path: &Path, contents: &str) {
        if std::fs::read_to_string(path).ok().as_deref() != Some(contents) {
            std::fs::write(path, contents).expect("write a harness file");
        }
    }

    /// A copy of Wavetable's sources with the editor swapped for a stub:
    /// the real one is all GUI.
    fn copy_sources(from: &Path, to: &Path) {
        std::fs::create_dir_all(to.join("FX")).expect("create the source copy");
        for entry in std::fs::read_dir(from)
            .expect("read Wavetable's sources")
            .chain(std::fs::read_dir(from.join("FX")).expect("read Wavetable's effects"))
        {
            let path = entry.expect("a source entry").path();
            if path.extension().is_some_and(|e| e == "cpp" || e == "h") {
                let rel = path.strip_prefix(from).expect("a source under the tree");
                let dst = to.join(rel);
                if mtime(&path) > mtime(&dst) {
                    std::fs::copy(&path, &dst).expect("copy a source");
                }
            }
        }
        write(
            &to.join("PluginEditor.h"),
            "#pragma once\n#include \"PluginProcessor.h\"\n\
             class WavetableAudioProcessorEditor : public juce::AudioProcessorEditor\n{\npublic:\n    \
             explicit WavetableAudioProcessorEditor (WavetableAudioProcessor&) {}\n};\n",
        );
    }

    fn build() -> PathBuf {
        let harness = root().join("tests/cpp_harness");
        let (juce, gin, wavetable) = (vendor("juce"), vendor("gin"), vendor("wavetable"));
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "wavetable-gold-{}-{}-{}",
            revision(&juce),
            revision(&gin),
            revision(&wavetable)
        ));
        let sources = out.join("wt");
        copy_sources(
            &wavetable.join("plugin").join(["Sou", "rce"].concat()),
            &sources,
        );
        let harness_changed = newest(&harness);
        let config_changed = mtime(&harness.join("config.h"));

        let mut units: Vec<(PathBuf, bool)> = JUCE_UNITS
            .iter()
            .map(|u| (harness.join(u), false))
            .collect();
        units.push((harness.join("gin.cpp"), true));
        units.push((harness.join("main.cpp"), true));
        for unit in WAVETABLE_UNITS {
            units.push((sources.join(unit), true));
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
            // No fused multiply-adds: the port never fuses.
            cmd.args(["-O2", "-DNDEBUG", "-ffp-contract=off", "-include"])
                .arg(harness.join("config.h"))
                .arg("-I")
                .arg(harness.join("shim"))
                .arg("-I")
                .arg(&sources)
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

    fn harness(input: Vec<u8>) -> Vec<u8> {
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
        output.stdout
    }

    fn bits(value: f32) -> String {
        format!("{:08x}", value.to_bits())
    }

    /// The frames the port plays, handed to the original's oscillator.
    fn send_table(input: &mut Vec<u8>, osc: usize, table: &Wavetable) {
        let first = table.frame(0);
        let _ = writeln!(
            input,
            "table {osc} {} {} {} {}",
            table.len(),
            first.copies(),
            first.table_size(),
            first.notes_per_table()
        );
        for f in 0..table.len() {
            let frame = table.frame(f);
            for c in 0..frame.copies() {
                for s in frame.copy(c) {
                    input.extend_from_slice(&s.to_le_bytes());
                }
            }
        }
        input.push(b'\n');
    }

    /// The tables a plugin set as `plugin` is would play, to the original.
    fn send_tables(input: &mut Vec<u8>, plugin: &mut WavetablePlugin, rate: f64) {
        hand_tables(plugin, rate);
        let tables = plugin
            .tables()
            .expect("tables were just handed over")
            .clone();
        let indices = plugin.table_indices();
        for (osc, index) in indices.into_iter().enumerate() {
            let table = tables
                .oscillator(osc, f32::from(index))
                .and_then(bloop_plugin_wavetable::dsp::tables::Table::exact)
                .expect("the set holds the oscillator's table, exact");
            send_table(input, osc, table);
        }
    }

    pub fn run(program: &Program) -> Vec<Signal> {
        let rate = program.sample_rate;
        // A plugin alongside, only to resolve presets and tables the way
        // the port does.
        let mut plugin = WavetablePlugin::at(rate);
        let mut input = Vec::new();
        let _ = writeln!(input, "new {rate:?}");
        send_tables(&mut input, &mut plugin, rate);
        let mut lengths = Vec::new();
        // The original hears the MIDI the Plug-in's own MPE zone makes of
        // bloop's notes (`mpe::Channels`), each note on its channel.
        let mut channels = mpe::Channels::default();
        let mpe_switch = params::index_of("mpe").expect("Wavetable has an MPE switch");
        for step in &program.steps {
            match *step {
                Step::Set(uid, value) => {
                    set(&mut plugin, uid, value);
                    let _ = writeln!(input, "set {uid} {}", bits(value));
                }
                Step::Mod(src, uid, depth) => {
                    let src = source_slot(src) as usize - 1;
                    let _ = writeln!(input, "mod {src} {uid} {}", bits(depth));
                }
                Step::Preset(index) => {
                    plugin.set_param(PARAM_PRESET, index as f32);
                    let preset = &presets::PRESETS[index];
                    let _ = writeln!(input, "defaults");
                    for &(uid, value) in preset.params {
                        let _ = writeln!(input, "set {uid} {}", bits(value));
                    }
                    for &(src, uid, depth) in preset.mods.iter().take(params::ROUTES) {
                        let src = source_slot(src) as usize - 1;
                        let _ = writeln!(input, "mod {src} {uid} {}", bits(depth));
                    }
                    send_tables(&mut input, &mut plugin, rate);
                    let _ = writeln!(input, "loaded");
                }
                Step::PresetFile(xml) => {
                    let (index, _) = bloop_plugin_wavetable::preset_file::import(xml.as_bytes())
                        .unwrap_or_else(|e| panic!("the preset reads: {e}"));
                    plugin.set_param(PARAM_PRESET, index as f32);
                    let _ = writeln!(input, "presetxml {}", xml.len());
                    input.extend_from_slice(xml.as_bytes());
                    input.push(b'\n');
                    send_tables(&mut input, &mut plugin, rate);
                    let _ = writeln!(input, "loaded");
                }
                Step::Tables(a, b) => {
                    plugin.set_param(params::id::WT1, f32::from(table_index(a)));
                    plugin.set_param(params::id::WT2, f32::from(table_index(b)));
                    send_tables(&mut input, &mut plugin, rate);
                }
                Step::Tempo(bpm) => {
                    let _ = writeln!(input, "tempo {bpm:?}");
                }
                Step::Midi(..) => {}
                Step::Play { seconds, block, .. } => {
                    let n = samples(seconds, rate);
                    let midi: Vec<(u32, Midi)> = program_midi_before(program, step);
                    let mut start = 0;
                    let mut sizes = Vec::new();
                    let mpe = plugin.get_param(mpe_switch) >= 0.5;
                    while start < n {
                        let len = block.min(n - start);
                        for &(t, m) in &midi {
                            let t = t as usize;
                            if (start..start + len).contains(&t) {
                                channels.translate(mpe, &m.event(0), |event| {
                                    if let Some([b0, b1, b2]) = event.midi() {
                                        let _ =
                                            writeln!(input, "midi {} {b0} {b1} {b2}", t - start);
                                    }
                                });
                            }
                        }
                        let _ = writeln!(input, "process {len}");
                        sizes.push(len);
                        start += len;
                    }
                    lengths.push(sizes);
                }
            }
        }

        let output = harness(input);
        let floats: Vec<f32> = output
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut at = 0;
        lengths
            .into_iter()
            .map(|sizes| {
                let mut signal = Signal::default();
                for n in sizes {
                    signal.left.extend_from_slice(&floats[at..at + n]);
                    signal.right.extend_from_slice(&floats[at + n..at + 2 * n]);
                    at += 2 * n;
                }
                signal
            })
            .collect()
    }

    /// The MIDI queued since the passage before `play`.
    fn program_midi_before(program: &Program, play: &Step) -> Vec<(u32, Midi)> {
        let end = program
            .steps
            .iter()
            .position(|s| std::ptr::eq(s, play))
            .expect("the passage is in the program");
        let start = program.steps[..end]
            .iter()
            .rposition(|s| matches!(s, Step::Play { .. }))
            .map_or(0, |p| p + 1);
        program.steps[start..end]
            .iter()
            .filter_map(|s| match *s {
                Step::Midi(t, m) => Some((t, m)),
                _ => None,
            })
            .collect()
    }

    /// Mono 16-bit samples as a WAV file.
    fn wav16(samples: &[f32], rate: f64) -> Vec<u8> {
        let data = u32::try_from(samples.len() * 2).expect("a table under 4 GB");
        let rate = rate as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data.to_le_bytes());
        for s in samples {
            wav.extend_from_slice(&((f64::from(*s) * 32768.0).round() as i16).to_le_bytes());
        }
        wav
    }

    /// What the original's `loadWaveTable` builds from catalogue entry
    /// `index` at `rate`: per frame, its band-limited copies.
    pub fn load(index: usize, rate: f64) -> Vec<Vec<Vec<f32>>> {
        let entry = &bank::catalogue()[index];
        let bytes = entry.bytes().expect("this build carries the table");
        // The original cannot read bloop's packing of a table: hand it
        // Wavetable's own file, or, for a table a preset carried, the
        // samples as a 16-bit WAV.
        let (format, bytes) = match entry.format {
            bank::Format::Flac => ("flac", bytes.to_vec()),
            bank::Format::Wav => ("wav", bytes.to_vec()),
            bank::Format::FlacDelta if entry.bank != "Presets" => {
                let file = vendor("wavetable")
                    .join("plugin/Resources/WavetablesFLAC")
                    .join(entry.bank)
                    .join(format!("{}.wt2048", entry.name));
                let original =
                    std::fs::read(&file).unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
                ("flac", original)
            }
            bank::Format::FlacDelta => {
                let (samples, rate) =
                    bank::decode(entry.format, entry.frame, bytes).expect("the table decodes");
                ("wav", wav16(&samples, rate))
            }
        };
        let mut input = Vec::new();
        let _ = writeln!(
            input,
            "wtload {format} {} {rate:?} {}",
            entry.frame,
            bytes.len()
        );
        input.extend_from_slice(&bytes);
        input.push(b'\n');
        let output = harness(input);
        let mut words = output.chunks_exact(4).map(|b| [b[0], b[1], b[2], b[3]]);
        let mut next_u32 = || u32::from_le_bytes(words.next().expect("more output")) as usize;
        let frames = next_u32();
        let mut out = Vec::new();
        for _ in 0..frames {
            let copies = next_u32();
            let len = next_u32();
            let mut frame = Vec::new();
            for _ in 0..copies {
                frame.push(
                    (0..len)
                        .map(|_| f32::from_bits(next_u32() as u32))
                        .collect(),
                );
            }
            out.push(frame);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Building the tables
// ---------------------------------------------------------------------------

#[cfg(feature = "gold")]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn table_builds_match_the_original() {
    let mut report = String::new();
    for (name, rate) in [
        ("AKWP bw_saw", 48000.0),
        ("AKWP 0001", 44100.0),
        ("NK - INERT", 48000.0),
        ("Untitled project - 2020-May-16_6", 48000.0),
        // BVKER's, packed as frame deltas.
        ("Growl 03", 48000.0),
        // Carried by a preset as a 16-bit WAV, packed as frame deltas.
        ("L_string ", 48000.0),
    ] {
        let Some(index) = bank::index_of(name) else {
            let _ = writeln!(report, "{name}: not in the catalogue");
            continue;
        };
        let ours = bank::load(index, rate).expect("the port builds the table");
        let theirs = cpp::load(index, rate);
        assert_eq!(ours.len(), theirs.len(), "{name}: frame count");
        let (mut differing, mut total, mut peak) = (0usize, 0usize, 0.0f32);
        for (f, frame) in theirs.iter().enumerate() {
            let mine = ours.frame(f);
            assert_eq!(mine.copies(), frame.len(), "{name}: copies of frame {f}");
            for (c, copy) in frame.iter().enumerate() {
                let m = mine.copy(c);
                assert_eq!(m.len(), copy.len(), "{name}: length of frame {f} copy {c}");
                for (a, b) in m.iter().zip(copy) {
                    peak = peak.max((a - b).abs());
                    differing += usize::from(a.to_bits() != b.to_bits());
                    total += 1;
                }
            }
        }
        let _ = writeln!(
            report,
            "{name} @ {rate}: {} frames, {differing} of {total} samples differ, by at most {peak:e} ({:.0} dB)",
            ours.len(),
            20.0 * f64::from(peak).log10()
        );
        assert!(peak < 2e-6, "{name}: the builds differ by {peak:e}");
    }
    expect![[r#"
        AKWP bw_saw @ 48000: 50 frames, 1385990 of 2151450 samples differ, by at most 4.7683716e-7 (-126 dB)
        AKWP 0001 @ 44100: 100 frames, 2844402 of 4302900 samples differ, by at most 7.748604e-7 (-122 dB)
        NK - INERT @ 48000: 256 frames, 7021876 of 11015424 samples differ, by at most 5.9604645e-7 (-124 dB)
        Untitled project - 2020-May-16_6 @ 48000: 106 frames, 252622 of 572082 samples differ, by at most 5.9604645e-8 (-144 dB)
        Growl 03 @ 48000: 256 frames, 7295381 of 11015424 samples differ, by at most 3.5762787e-7 (-129 dB)
        L_string  @ 48000: 256 frames, 6081155 of 11015424 samples differ, by at most 4.7683716e-7 (-126 dB)
    "#]].assert_eq(&report);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn init_plays_a_note() {
    check(
        &Program::new().note(60, 100, 0.5, 0.5),
        &expect![[r#"
        held: L peak -19.5 rms -25.9 | R peak -19.5 rms -25.9 #b16fb171aed34269
        released: L peak -21.7 rms -42.5 | R peak -21.7 rms -42.5 #e5fa50bd6ef941a5
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn a_chord_bent_and_released() {
    check(
        &Program::new()
            .on(0, 48, 90)
            .on(100, 55, 70)
            .on(200, 64, 127)
            .play("chord", 0.3)
            .bend(0, 16383)
            .play("bent up", 0.2)
            .bend(0, 0)
            .play("bent down", 0.2)
            .off(0, 48)
            .off(0, 55)
            .off(0, 64)
            .play("released", 0.4),
        &expect![[r#"
            chord: L peak -9.9 rms -18.8 | R peak -9.9 rms -18.8 #3734954238a4aaed
            bent up: L peak -12.0 rms -19.4 | R peak -12.0 rms -19.4 #4cc8e5bd9a9cd159
            bent down: L peak -11.6 rms -17.6 | R peak -11.6 rms -17.6 #51a7762847931939
            released: L peak -13.0 rms -33.3 | R peak -13.0 rms -33.3 #1115b49a46cfe3d9
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn position_swept_by_an_lfo() {
    check(
        &Program::new()
            .tables("NK - INERT", "AKWP bw_squ")
            .set("osc1pos", 25.0)
            .set("lfo1enable", 1.0)
            .set("lfo1rate", 2.0)
            .modulate("lfo1", "osc1pos", 0.5)
            .note(57, 100, 1.0, 0.3),
        &expect![[r#"
            held: L peak -17.3 rms -23.1 | R peak -17.3 rms -23.1 #419489959fd678f9
            released: L peak -18.2 rms -37.1 | R peak -18.2 rms -37.1 #db0a6e102430639d
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn both_oscillators_with_unison_formant_and_bend() {
    check(
        &Program::new()
            .set("osc1unison", 8.0)
            .set("osc1detune", 0.3)
            .set("osc1spread", 80.0)
            .set("osc1formant", 0.4)
            .set("osc2enable", 1.0)
            .set("osc2unison", 3.0)
            .set("osc2tune", 7.0)
            .set("osc2bend", -0.5)
            .set("osc2pan", 0.6)
            .note(45, 110, 0.6, 0.3),
        &expect![[r#"
            held: L peak -10.7 rms -20.4 | R peak -5.1 rms -14.6 #536aa08f35f0301b
            released: L peak -18.1 rms -35.6 | R peak -11.8 rms -29.4 #3e088d91c036475f
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn every_filter_type_under_its_envelope() {
    let mut program = Program::new()
        .set("fltenable", 1.0)
        .set("fltres", 60.0)
        .set("fltfreq", 70.0)
        .set("fltamount", 0.6)
        .set("fltdecay", 0.2);
    for kind in 0..8 {
        program = program.set("flttype", kind as f32).note(52, 100, 0.25, 0.1);
    }
    check(
        &program,
        &expect![[r#"
        held: L peak -11.5 rms -22.6 | R peak -11.5 rms -22.6 #d768c6dc8f9e6b2d
        released: L peak -14.2 rms -32.8 | R peak -14.2 rms -32.8 #f75b56507f08f1d9
        held: L peak -12.2 rms -22.7 | R peak -12.2 rms -22.7 #3e53d4c207ded6dd
        released: L peak -14.7 rms -32.1 | R peak -14.7 rms -32.1 #2a058bc189e49775
        held: L peak -17.0 rms -41.7 | R peak -17.0 rms -41.7 #75fcad16ee3c62d5
        released: L peak -22.5 rms -49.6 | R peak -22.5 rms -49.6 #3cf8f7b8c568476d
        held: L peak -20.5 rms -44.6 | R peak -20.5 rms -44.6 #344e6e46885e4b49
        released: L peak -25.9 rms -52.2 | R peak -25.9 rms -52.2 #507bc48ae386b779
        held: L peak -21.5 rms -45.1 | R peak -21.5 rms -45.1 #3ffe8638039f01f9
        released: L peak -24.5 rms -53.3 | R peak -24.5 rms -53.3 #8a3e867931e5a0ed
        held: L peak -24.1 rms -46.6 | R peak -24.1 rms -46.6 #8135a64237323d45
        released: L peak -26.2 rms -54.6 | R peak -26.2 rms -54.6 #f16afbbe5c2fcdd5
        held: L peak -13.2 rms -22.7 | R peak -13.2 rms -22.7 #1df2b56fdea4611d
        released: L peak -16.7 rms -32.5 | R peak -16.7 rms -32.5 #060a58b79536d49d
        held: L peak -13.8 rms -22.8 | R peak -13.8 rms -22.8 #217f6a5048bfa561
        released: L peak -18.3 rms -32.9 | R peak -18.3 rms -32.9 #89fcd9f68bd8afd9
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn sub_and_both_noises() {
    check(
        &Program::new()
            .set("osc1enable", 0.0)
            .set("subenable", 1.0)
            .set("subwave", 3.0)
            .set("noiseenable", 1.0)
            .note(40, 100, 0.3, 0.1)
            .set("noisetype", 1.0)
            .set("subwave", 5.0)
            .note(40, 100, 0.3, 0.1),
        &expect![[r#"
            held: L peak -7.4 rms -10.9 | R peak -7.4 rms -10.9 #1cc3b33875ba7005
            released: L peak -10.0 rms -20.7 | R peak -10.0 rms -20.7 #368f1e8aeca752d9
            held: L peak -4.6 rms -10.2 | R peak -4.6 rms -10.2 #7c184d23a4fa2229
            released: L peak -10.0 rms -20.4 | R peak -10.0 rms -20.4 #8812a8d5c19e2c35
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn mono_legato_glide() {
    check(
        &Program::new()
            .set("mono", 1.0)
            .set("legato", 1.0)
            .set("gMode", 2.0)
            .set("gRate", 200.0)
            .on(0, 48, 100)
            .play("first", 0.3)
            .on(0, 60, 100)
            .play("glide up", 0.3)
            .off(0, 60)
            .play("back down", 0.3)
            .off(0, 48)
            .play("released", 0.3),
        &expect![[r#"
            first: L peak -17.3 rms -24.0 | R peak -17.3 rms -24.0 #20afee39cba66dd5
            glide up: L peak -17.3 rms -23.4 | R peak -17.3 rms -23.4 #2506e1e80b47a975
            back down: L peak -17.3 rms -23.4 | R peak -17.3 rms -23.4 #ec355bb79b4b1c85
            released: L peak -20.7 rms -38.1 | R peak -20.7 rms -38.1 #3573b49be0b77fd9
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn mono_note_stack_without_glide_and_glissando_retriggering() {
    check(
        &Program::new()
            .set("mono", 1.0)
            .on(0, 50, 100)
            .on(1000, 53, 90)
            .on(2000, 57, 80)
            .play("three held", 0.2)
            .off(0, 53)
            .play("middle let go", 0.1)
            .off(0, 57)
            .play("back to the first", 0.2)
            .set("gMode", 1.0)
            .set("gRate", 150.0)
            .on(0, 62, 100)
            .play("glissando up", 0.3)
            .off(0, 62)
            .off(500, 50)
            .play("all released", 0.4),
        &expect![[r#"
            three held: L peak -25.5 rms -31.6 | R peak -25.5 rms -31.6 #ebffc944d8dde825
            middle let go: L peak -27.3 rms -31.3 | R peak -27.3 rms -31.3 #09dace2dbe956785
            back to the first: L peak -17.9 rms -23.5 | R peak -17.9 rms -23.5 #4a5f9d1ba8fa2a6d
            glissando up: L peak -17.7 rms -23.6 | R peak -17.7 rms -23.6 #d5dc3f4a4103d8ad
            all released: L peak -18.9 rms -36.1 | R peak -18.9 rms -36.1 #608549ce16bf7651
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn modulation_from_envelope_velocity_cc_and_step_lfo() {
    check(
        &Program::new()
            .set("env1enable", 1.0)
            .set("env1attack", 0.2)
            .set("slfoenable", 1.0)
            .set("slfostep1", 1.0)
            .set("slfostep3", -1.0)
            .modulate("env1", "osc1pos", 0.8)
            .modulate("vel", "osc1level", -0.4)
            .modulate("cc1", "osc1finetune", 0.7)
            .modulate("step", "osc1pan", 1.0)
            .cc(0, 1, 100)
            .on(0, 64, 40)
            .play("held", 0.6)
            .cc(0, 1, 0)
            .off(300, 64)
            .play("released", 0.3),
        &expect![[r#"
            held: L peak -45.4 rms -57.6 | R peak -36.0 rms -45.2 #0991e9bb03c22083
            released: L peak -45.3 rms -62.3 | R peak -45.3 rms -62.3 #28b3536b7f8681e7
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn tempo_synced_lfo_and_gate() {
    check(
        &Program::new()
            .tempo(140.0)
            .set("lfo2enable", 1.0)
            .set("lfo2sync", 1.0)
            .set("lfo2wave", 4.0)
            .modulate("mlfo2", "fltfreq", 0.3)
            .set("fltenable", 1.0)
            .set("gateenable", 1.0)
            .note(55, 100, 0.8, 0.2),
        &expect![[r#"
            held: L peak -15.5 rms -27.4 | R peak -15.5 rms -27.3 #dca20be64097f00e
            released: L peak -65.7 rms -85.0 | R peak -68.0 rms -87.4 #d9e541fc9e65c8d4
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn each_effect_alone() {
    let mut program = Program::new();
    for (enable, extra) in [
        ("chEnable", None),
        ("dsEnable", Some(("dsAmount", 0.6))),
        ("dlEnable", Some(("dlFb", 0.5))),
        ("rvEnable", Some(("rvbMix", 0.5))),
    ] {
        program = program.set(enable, 1.0);
        if let Some((uid, value)) = extra {
            program = program.set(uid, value);
        }
        program = program.note(60, 100, 0.2, 0.3).set(enable, 0.0);
    }
    for mode in 1..=3 {
        program = program
            .set("dsEnable", 1.0)
            .set("distMode", mode as f32)
            .note(48, 120, 0.2, 0.1);
    }
    check(
        &program,
        &expect![[r#"
        held: L peak -22.2 rms -29.0 | R peak -24.7 rms -32.9 #dfbae7ddc6ccfda6
        released: L peak -23.7 rms -41.8 | R peak -31.1 rms -49.9 #77f4f921c91134fe
        held: L peak -12.5 rms -19.0 | R peak -12.5 rms -19.0 #8fdb0eb688a4e921
        released: L peak -14.7 rms -33.1 | R peak -14.7 rms -33.1 #51f23fc407b4d4e5
        held: L peak -19.5 rms -26.2 | R peak -19.5 rms -26.2 #578efa885fda68b5
        released: L peak -22.0 rms -40.6 | R peak -22.0 rms -40.6 #84edf76ed544d86d
        held: L peak -18.0 rms -24.0 | R peak -18.0 rms -24.0 #fc674a546e09a621
        released: L peak -19.7 rms -34.4 | R peak -19.7 rms -34.4 #649368b3aedbf619
        held: L peak -10.6 rms -15.1 | R peak -10.6 rms -15.1 #27f01503712f7275
        released: L peak -12.9 rms -21.7 | R peak -12.9 rms -21.7 #94dd49c11d820d25
        held: L peak -4.3 rms -13.6 | R peak -4.3 rms -13.6 #06effdb0d569fbf4
        released: L peak -4.8 rms -18.9 | R peak -4.8 rms -18.9 #02de42d9b66f5114
        held: L peak -1.4 rms -11.0 | R peak -1.4 rms -11.0 #799760475d5cc56c
        released: L peak -1.4 rms -12.2 | R peak -1.4 rms -12.2 #f587b0df9f342712
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn a_held_note_through_a_moved_position_a_new_table_and_a_preset() {
    check(
        &Program::new()
            .on(0, 60, 100)
            .play("held", 0.3)
            .set("osc1pos", 60.0)
            .play("position moved", 0.2)
            .tables("NK - INERT", "AKWP bw_squ")
            .play("table switched", 0.2)
            .preset("DT Siren 1")
            .play("preset loaded", 0.3)
            .off(0, 60)
            .play("released", 0.3),
        &expect![[r#"
            held: L peak -19.5 rms -26.0 | R peak -19.5 rms -26.0 #9522326ad58d7afd
            position moved: L peak -19.7 rms -23.4 | R peak -19.7 rms -23.4 #fbac01db02c24ef1
            table switched: L peak -20.2 rms -25.8 | R peak -20.2 rms -25.8 #c2213c572e5d3f09
            preset loaded: L peak -inf rms -inf | R peak -inf rms -inf #2d9ab45bcfc84b25
            released: L peak -inf rms -inf | R peak -inf rms -inf #2d9ab45bcfc84b25
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn more_notes_than_voices_steal() {
    let mut program = Program::new().set("voices", 4.0).set("release", 1.0);
    for (i, note) in [48, 52, 55, 59, 62, 65].into_iter().enumerate() {
        program = program.on(i as u32 * 50, note, 100);
    }
    check(
        &program.play("six notes on four voices", 0.4),
        &expect![[r#"
        six notes on four voices: L peak -8.7 rms -19.8 | R peak -8.7 rms -19.8 #60b39995c02dafc5
    "#]],
    );
}

/// The MPE zone (bloop#pad-expression): two notes, each bent on its own
/// channel — one up an octave in steps, one down a fifth — then the
/// track's wheel bending both from the master channel, a bend back to
/// rest, and the release. A third note struck after the first let go
/// starts at rest on the channel the first had.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn mpe_each_note_bent_on_its_own() {
    check(
        &Program::new()
            .set("mpe", 1.0)
            .on_as(0, 60, 100, 1)
            .on_as(40, 67, 90, 2)
            .play("struck", 0.2)
            .note_bend(0, 1, 4.0)
            .note_bend(100, 2, -7.0)
            .note_bend(300, 1, 8.0)
            .note_bend(600, 1, 12.0)
            .play("bent apart", 0.3)
            .bend(0, 16383)
            .play("and the wheel", 0.2)
            .note_bend(0, 1, 0.0)
            .bend(0, 8192)
            .play("back to rest", 0.2)
            .off_as(0, 60, 1)
            .on_as(200, 64, 100, 3)
            .play("a new note on a freed channel", 0.3)
            .off_as(0, 67, 2)
            .off_as(0, 64, 3)
            .play("released", 0.3),
        &expect![[r#"
            struck: L peak -16.1 rms -25.3 | R peak -16.1 rms -25.3 #daf128e2bbd56255
            bent apart: L peak -18.2 rms -26.5 | R peak -18.2 rms -26.5 #54104b7f6a8fbfb5
            and the wheel: L peak -21.3 rms -27.8 | R peak -21.3 rms -27.8 #b08d04f7b486094d
            back to rest: L peak -16.8 rms -21.1 | R peak -16.8 rms -21.1 #b816773e4b32510d
            a new note on a freed channel: L peak -17.4 rms -24.8 | R peak -17.4 rms -24.8 #0cfc598a1e0f689d
            released: L peak -19.2 rms -39.7 | R peak -19.2 rms -39.7 #4b29338d459eda35
        "#]],
    );
}

/// Pressure and timbre into the matrix (`mpep`, `mpet`): the filter
/// opens with one note's pressure and the table moves with the other's
/// timbre, each note alone; a note let go of and struck again starts
/// with no pressure and timbre at rest.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn mpe_pressure_and_timbre_through_the_matrix() {
    check(
        &Program::new()
            .tables("NK - INERT", "AKWP bw_squ")
            .set("mpe", 1.0)
            .set("fltenable", 1.0)
            .set("fltfreq", 30.0)
            .modulate("mpep", "fltfreq", 0.6)
            .modulate("mpet", "osc1pos", 0.8)
            .on_as(0, 48, 100, 1)
            .on_as(0, 60, 100, 2)
            .play("struck", 0.2)
            .note_pressure(0, 1, 0.3)
            .note_timbre(200, 2, 0.1)
            .note_pressure(500, 1, 0.9)
            .note_timbre(800, 2, 1.0)
            .play("leaned and slid", 0.4)
            .off_as(0, 48, 1)
            .on_as(100, 48, 100, 3)
            .play("struck again", 0.3)
            .off_as(0, 60, 2)
            .off_as(0, 48, 3)
            .play("released", 0.3),
        &expect![[r#"
            struck: L peak -35.2 rms -41.2 | R peak -35.2 rms -41.2 #9fd2579f657b41b5
            leaned and slid: L peak -15.1 rms -20.7 | R peak -15.1 rms -20.7 #c212f620d2999e69
            struck again: L peak -18.3 rms -34.4 | R peak -18.3 rms -34.4 #dc889cd946e54761
            released: L peak -39.3 rms -55.9 | R peak -39.3 rms -55.9 #d658107307461715
        "#]],
    );
}

/// The MPE switch turned off with notes held: every note lets go, and
/// what follows is legacy MIDI on one channel, where controller 74 is the
/// timbre of the last note struck and the wheel bends by the Pitch Bend
/// knob's range. Turned on again, it lets go once more.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn mpe_switched_with_notes_held() {
    check(
        &Program::new()
            .set("mpe", 1.0)
            .set("pitchbend", 12.0)
            .modulate("mpet", "osc1pos", 0.8)
            .on_as(0, 55, 100, 1)
            .note_bend(100, 1, 2.0)
            .play("in the zone", 0.2)
            .set("mpe", 0.0)
            .play("switched off", 0.3)
            .on(0, 57, 100)
            .on(50, 61, 100)
            .cc(300, 74, 127)
            .bend(600, 12000)
            .play("legacy, with timbre and the wheel", 0.3)
            .set("mpe", 1.0)
            .play("switched on", 0.3),
        &expect![[r#"
            in the zone: L peak -21.8 rms -30.1 | R peak -21.8 rms -30.1 #39a3f0990968b375
            switched off: L peak -24.9 rms -44.6 | R peak -24.9 rms -44.6 #13072eec513cea15
            legacy, with timbre and the wheel: L peak -18.0 rms -28.2 | R peak -18.0 rms -28.2 #e53c93fa86ba0235
            switched on: L peak -21.4 rms -42.4 | R peak -21.4 rms -42.4 #13db304439f79b05
        "#]],
    );
}

/// More notes than member channels and than voices: seventeen held on
/// eight voices, the sixteenth and seventeenth on the master channel,
/// each struck with its pressure at once (as bloop starts a note at its
/// strike's velocity), then every note bent and the master's wheel
/// moved, and all let go.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn mpe_more_notes_than_channels_and_voices() {
    let mut program = Program::new()
        .set("mpe", 1.0)
        .set("voices", 8.0)
        .set("release", 0.3)
        .modulate("mpep", "osc1level", -0.5);
    for n in 1..=17u32 {
        let note = 40 + n as u8 * 2;
        program = program
            .on_as(n * 30, note, 100, n)
            .note_pressure(n * 30, n, n as f32 / 17.0);
    }
    program = program.play("seventeen struck", 0.4);
    for n in 1..=17u32 {
        program = program.note_bend(n * 10, n, n as f32 - 9.0);
    }
    program = program.bend(300, 4000).play("bent", 0.3);
    for n in 1..=17u32 {
        program = program.off_as(n * 5, 40 + n as u8 * 2, n);
    }
    check(
        &program.play("released", 0.4),
        &expect![[r#"
        seventeen struck: L peak -12.8 rms -22.8 | R peak -12.8 rms -22.8 #3320258c8ca2dbc9
        bent: L peak -13.6 rms -22.2 | R peak -13.6 rms -22.2 #4d16401d13c00445
        released: L peak -14.9 rms -33.5 | R peak -14.9 rms -33.5 #4eb1c06e15492745
    "#]],
    );
}

/// A factory preset that plays its filter by pressure, with each of two
/// notes leaned on in turn.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn mpe_a_factory_preset_played_by_pressure() {
    check(
        &Program::new()
            .preset("Simbad")
            .on_as(0, 45, 110, 1)
            .on_as(0, 52, 110, 2)
            .play("struck", 0.3)
            .note_pressure(0, 1, 1.0)
            .note_pressure(0, 2, 0.0)
            .play("the low one leaned on", 0.3)
            .note_pressure(0, 1, 0.0)
            .note_pressure(0, 2, 1.0)
            .play("the high one leaned on", 0.3)
            .off_as(0, 45, 1)
            .off_as(0, 52, 2)
            .play("released", 0.4),
        &expect![[r#"
            struck: L peak -9.9 rms -21.9 | R peak -8.3 rms -20.5 #9367eb39357837a0
            the low one leaned on: L peak -3.3 rms -12.6 | R peak -3.6 rms -12.6 #87f74bf77e83665a
            the high one leaned on: L peak -4.6 rms -14.5 | R peak -3.4 rms -14.5 #ba02010706c8c405
            released: L peak -5.8 rms -19.2 | R peak -8.1 rms -19.4 #9b0d3fddc0d0e7fd
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn other_rates_and_odd_blocks() {
    for (rate, block, expect) in [
        (
            44100.0,
            37,
            expect![[r#"
            held: L peak -9.6 rms -15.6 | R peak -9.6 rms -15.6 #038f368ed6e9278d
            released: L peak -11.5 rms -30.0 | R peak -11.5 rms -30.0 #7231834ac13f5205
        "#]],
        ),
        (
            96000.0,
            1000,
            expect![[r#"
            held: L peak -9.6 rms -15.5 | R peak -9.6 rms -15.5 #35d2c2beeece9739
            released: L peak -11.7 rms -30.0 | R peak -11.7 rms -30.0 #82f0b4112fa58a19
        "#]],
        ),
    ] {
        check(
            &Program::new()
                .at(rate)
                .set("subenable", 1.0)
                .set("rvEnable", 1.0)
                .on(5, 62, 100)
                .play_in("held", 0.3, block)
                .off(3, 62)
                .play_in("released", 0.3, block),
            &expect,
        );
    }
}

/// The presets the port plays within a few ulp of the original rather
/// than bit for bit: each has an LFO whose rate an envelope modulates,
/// and that LFO turning a level. Neither alone diverges (the smallest
/// case: an envelope on LFO 1's rate, LFO 1 on OSC 1's level). ROADMAP.
#[cfg(feature = "gold")]
const NOT_BIT_EXACT: [&str; 5] = [
    "DT Noise Bass 1",
    "DT Noise Bass 2",
    "DT Noise Bass 3",
    "DT Noise Bass 5",
    "DT Noise Bass 6",
];

/// Every factory preset by `author`, a chord held and let go. Not on
/// wasm, whose build carries only some presets' tables
/// (`Preset::on_web`); the plugin tests play those there. Each author's
/// in two tests, the first half and the second, so they run side by side:
/// building a preset's tables is most of what each costs.
#[cfg(not(target_arch = "wasm32"))]
fn factory_presets(author: &str, half: usize, expect: &Expect) {
    let mut report = String::new();
    let all: Vec<_> = presets::PRESETS
        .iter()
        .filter(|p| p.author == author)
        .collect();
    let (first, second) = all.split_at(all.len() / 2);
    for preset in [first, second][half] {
        let program = Program::new()
            .preset(preset.name)
            .on(0, 48, 100)
            .on(0, 55, 100)
            .on(0, 60, 100)
            .play("chord", 0.5)
            .off(0, 48)
            .off(0, 55)
            .off(0, 60)
            .play("released", 0.5);
        let rust = run_rust(&program);
        #[cfg(feature = "gold")]
        if NOT_BIT_EXACT.contains(&preset.name) {
            compare_within(&program, &rust, &cpp::run(&program), 2e-7);
        } else {
            compare(&program, &rust, &cpp::run(&program));
        }
        let _ = writeln!(report, "{}:\n{}", preset.name, summary(&rust));
    }
    assert!(!report.is_empty(), "no presets by {author}");
    assert_summary(expect, &report);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_doubletap_first_half() {
    factory_presets(
        "DoubleTap",
        0,
        &expect![[r#"
        DT Bwaing:
        chord: L peak -10.0 rms -19.7 | R peak -8.3 rms -18.4 #77947951b25caad0
        released: L peak -33.6 rms -55.4 | R peak -24.1 rms -46.4 #2265ca47d2958760

        DT Cymbal intro:
        chord: L peak -4.2 rms -22.6 | R peak -4.5 rms -22.6 #5246ba93c8ea72ff
        released: L peak -28.6 rms -42.8 | R peak -29.2 rms -43.1 #5ae5868f46e1ca7d

        DT Dive 2:
        chord: L peak -34.5 rms -50.1 | R peak -30.5 rms -50.3 #a448068b6f68dc09
        released: L peak -37.6 rms -63.9 | R peak -38.2 rms -66.0 #b8c03b5bfb53234a

        DT Dive 3:
        chord: L peak -13.4 rms -28.2 | R peak -13.8 rms -28.3 #a0efa3d5158243ec
        released: L peak -18.4 rms -45.8 | R peak -13.8 rms -43.9 #2a372742c7054c3e

        DT Dive 4:
        chord: L peak -9.8 rms -22.9 | R peak -8.1 rms -22.5 #1a90dfa105c3fe31
        released: L peak -16.8 rms -44.7 | R peak -16.6 rms -44.9 #53e1f8339d611add

        DT Dive 5 panning:
        chord: L peak -15.9 rms -29.2 | R peak -5.0 rms -19.6 #1512088fff195435
        released: L peak -22.5 rms -49.1 | R peak -13.1 rms -41.6 #b0314b813ccdeb2f

        DT Dive:
        chord: L peak -33.9 rms -49.9 | R peak -30.4 rms -50.4 #a8d21e1e5e46ef66
        released: L peak -36.7 rms -63.8 | R peak -40.0 rms -66.6 #c4553f1988db1b75

        DT Glitch 1:
        chord: L peak -8.2 rms -21.6 | R peak -8.2 rms -21.6 #97556a13b8760be5
        released: L peak -19.7 rms -33.8 | R peak -19.7 rms -33.8 #af2b24c0050a4121

        DT Glitch 2:
        chord: L peak -9.6 rms -19.6 | R peak -10.2 rms -19.9 #ab3dbc5d5e8a96a1
        released: L peak -12.1 rms -25.9 | R peak -12.5 rms -26.1 #c13011b8b26e41ce

        DT Glitch 3:
        chord: L peak -12.7 rms -22.1 | R peak -15.5 rms -24.2 #64783bedf3f6291e
        released: L peak -16.0 rms -30.6 | R peak -18.5 rms -32.9 #66391755953d72e5

        DT Kick 1:
        chord: L peak -7.5 rms -25.2 | R peak -7.5 rms -25.2 #f82657a86a9c9525
        released: L peak -28.6 rms -54.0 | R peak -28.6 rms -54.0 #f744d4f81b5b9985

        DT Kick 2:
        chord: L peak -8.6 rms -26.1 | R peak -8.6 rms -26.1 #a461b1b83ddb6a25
        released: L peak -105.1 rms -134.5 | R peak -105.1 rms -134.5 #432b9e7b204d45d5

        DT Noise Bass 1:
        chord: L peak -0.6 rms -3.8 | R peak -0.6 rms -4.0 #ea7381f5facb85b5
        released: L peak -1.1 rms -19.1 | R peak -1.7 rms -19.6 #fef882feaa9691c7

        DT Noise Bass 2:
        chord: L peak -1.5 rms -5.1 | R peak 0.5 rms -3.0 #4eaf4332d062ad73
        released: L peak -2.7 rms -20.5 | R peak -0.0 rms -16.1 #3c1d2996ba72e149

        DT Noise Bass 3:
        chord: L peak 1.2 rms -4.0 | R peak 1.2 rms -3.6 #7b2239ec0b11cb20
        released: L peak -0.5 rms -18.6 | R peak 0.5 rms -17.1 #4b1e19cbc2d007dc

        DT Noise Bass 5:
        chord: L peak -0.4 rms -4.4 | R peak 1.6 rms -2.4 #2ba2e44a02820070
        released: L peak -2.0 rms -19.8 | R peak 0.7 rms -15.4 #238740742dde06cb

        DT Noise Bass 6:
        chord: L peak -0.4 rms -4.2 | R peak 1.7 rms -2.2 #62049d6705124592
        released: L peak -2.1 rms -19.8 | R peak 0.8 rms -15.3 #099c7a74d1799c88

        DT Noisy Pluck:
        chord: L peak -21.8 rms -43.0 | R peak -30.1 rms -45.2 #fe9841b0729a808a
        released: L peak -30.2 rms -51.6 | R peak -34.5 rms -53.3 #d6e8b660408e0dac

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_doubletap_second_half() {
    factory_presets(
        "DoubleTap",
        1,
        &expect![[r#"
        DT Pad 2:
        chord: L peak -8.7 rms -20.4 | R peak -8.6 rms -19.1 #0a75539ff800ac30
        released: L peak -11.0 rms -22.3 | R peak -10.5 rms -22.2 #47dd0fd133b9c85c

        DT Pad:
        chord: L peak -30.0 rms -44.4 | R peak -33.4 rms -47.5 #66aee9f135612877
        released: L peak -27.6 rms -37.5 | R peak -30.9 rms -40.8 #b07153ee6912875c

        DT Perc snare:
        chord: L peak -15.0 rms -41.7 | R peak -15.0 rms -41.7 #1bc8bde4fc161c9d
        released: L peak -104.7 rms -137.0 | R peak -104.7 rms -137.0 #e0a73463bf58cbad

        DT Pluck 1:
        chord: L peak -8.1 rms -30.5 | R peak -8.9 rms -30.9 #acfc8af1af8a8270
        released: L peak -17.6 rms -41.6 | R peak -17.6 rms -41.7 #383d3451bf91e89f

        DT Pluck 2:
        chord: L peak -16.4 rms -34.7 | R peak -17.7 rms -35.4 #9103ef0fe6ea1e9f
        released: L peak -24.8 rms -45.4 | R peak -26.1 rms -46.3 #4920364611f64db6

        DT Riser 1:
        chord: L peak -33.7 rms -50.2 | R peak -33.7 rms -50.7 #abfbb525ac77d717
        released: L peak -38.1 rms -64.8 | R peak -40.9 rms -66.0 #54ccc05a889a2a41

        DT Seq 1:
        chord: L peak 1.0 rms -9.0 | R peak 1.0 rms -8.7 #779af447a4650560
        released: L peak 0.4 rms -20.1 | R peak 0.3 rms -20.1 #ccbd08e89c0f3a66

        DT Seq 2:
        chord: L peak -5.8 rms -17.9 | R peak -5.3 rms -14.2 #b26473c1f72d1de5
        released: L peak -10.6 rms -37.2 | R peak -11.8 rms -38.7 #e1681f1590b8dd1c

        DT Siren 1:
        chord: L peak -19.0 rms -30.4 | R peak -19.0 rms -30.4 #1e0f3e52696ca8b9
        released: L peak -18.0 rms -29.3 | R peak -18.0 rms -29.3 #8f00f7deb4869b8d

        DT Siren Complex 2:
        chord: L peak -9.6 rms -17.3 | R peak -9.5 rms -16.7 #d8756cf63f338f56
        released: L peak -9.7 rms -18.2 | R peak -9.5 rms -18.3 #3747de038d0f65c0

        DT Siren complex:
        chord: L peak -9.1 rms -16.9 | R peak -9.3 rms -16.4 #992f8ab30c7b4f26
        released: L peak -8.6 rms -18.5 | R peak -8.6 rms -18.6 #4aabc033bce94a18

        DT Snare 1:
        chord: L peak -18.2 rms -40.5 | R peak -18.2 rms -40.5 #9606b3f9ec455b89
        released: L peak -119.4 rms -155.2 | R peak -119.4 rms -155.2 #d5bad1da8b485df1

        DT Snare 2:
        chord: L peak -23.7 rms -47.8 | R peak -23.7 rms -47.8 #54e5e1dfe57a8331
        released: L peak -117.3 rms -154.1 | R peak -117.3 rms -154.1 #00c4447ff7ad930d

        DT Sweeper 1:
        chord: L peak -7.6 rms -22.6 | R peak -7.8 rms -22.2 #0802c2b381ecf208
        released: L peak -14.9 rms -44.8 | R peak -11.5 rms -40.3 #decd7956bda8f095

        DT Sweeper 2:
        chord: L peak -7.2 rms -21.6 | R peak -5.4 rms -21.2 #d6b4959cd0939c44
        released: L peak -16.0 rms -43.8 | R peak -12.8 rms -41.3 #457f8c54841def35

        DT Sweeper 3:
        chord: L peak -7.8 rms -22.9 | R peak -7.4 rms -23.3 #03a07308622d104b
        released: L peak -13.8 rms -44.2 | R peak -8.9 rms -38.3 #f24246e546677894

        DT Sweeper 4:
        chord: L peak -11.9 rms -25.1 | R peak -10.6 rms -24.5 #6e1878d86e3c1916
        released: L peak -16.7 rms -27.2 | R peak -16.1 rms -27.4 #4d297b909733070d

        DT Sweeper 5:
        chord: L peak -47.5 rms -59.2 | R peak -49.1 rms -58.5 #5da832fa0bf31008
        released: L peak -85.1 rms -116.8 | R peak -84.0 rms -112.3 #e1a89ddee007b009

        DT Sweeper 6:
        chord: L peak -9.1 rms -19.1 | R peak -7.0 rms -16.9 #b2950199f2dcb3b3
        released: L peak -8.1 rms -19.2 | R peak -8.6 rms -19.8 #12a4b0f8464c301e

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_flamedragonz_first_half() {
    factory_presets(
        "Flamedragonz",
        0,
        &expect![[r#"
        ARP Bass Pulse I:
        chord: L peak -0.6 rms -9.3 | R peak -0.1 rms -8.5 #acde3bb026ad56ee
        released: L peak -15.6 rms -42.0 | R peak -9.8 rms -35.2 #e1745948b92140d6

        ARP GrowlSaw Pulse I:
        chord: L peak 0.2 rms -10.9 | R peak 1.6 rms -10.0 #a323982ef7ca1822
        released: L peak -inf rms -inf | R peak -inf rms -inf #9ebf9a6ec921bb25

        ARP Hype Pulse I:
        chord: L peak 0.9 rms -6.8 | R peak 1.7 rms -6.9 #b95a33cdd370d334
        released: L peak -17.1 rms -37.8 | R peak -17.1 rms -38.8 #c80ad1b8a08b2381

        ARP Hype Pulse II:
        chord: L peak -0.5 rms -8.1 | R peak -0.5 rms -8.9 #be0226e0742d9ab0
        released: L peak -5.2 rms -30.7 | R peak -5.0 rms -27.5 #88433e848d4623f6

        ARP LSawSq I:
        chord: L peak -3.2 rms -10.5 | R peak -3.1 rms -9.7 #2bb9bd24af5f1f56
        released: L peak -3.4 rms -11.8 | R peak -3.3 rms -11.2 #f4a7c7b21249b7c0

        ARP Psy I:
        chord: L peak 1.6 rms -11.9 | R peak 2.4 rms -12.1 #340e5c92f8300c8e
        released: L peak -1.3 rms -17.6 | R peak -1.1 rms -17.6 #384ee0c8c3e59ed0

        ARP Voltage Pulse I:
        chord: L peak 1.0 rms -9.6 | R peak 1.0 rms -8.0 #2e1920d161119cf4
        released: L peak -inf rms -inf | R peak -17.8 rms -58.9 #630e6c804d6d9eab

        BR Brass I:
        chord: L peak 7.5 rms -2.6 | R peak 6.3 rms -2.8 #9325cc59592b093c
        released: L peak 6.2 rms -5.0 | R peak 3.6 rms -7.3 #fd061e6fa29348c1

        BS Filter Bass I:
        chord: L peak 0.4 rms -13.4 | R peak 0.5 rms -13.3 #652954606df88f9b
        released: L peak -19.4 rms -28.3 | R peak -20.2 rms -28.3 #1bbf0d64276ae97a

        BS Filter Bass II:
        chord: L peak 7.7 rms -2.3 | R peak 7.1 rms -2.5 #c22f2ce0d0425f15
        released: L peak -5.3 rms -23.7 | R peak -9.4 rms -24.5 #a1fd9843aac25643

        BS FM Bass I:
        chord: L peak 6.6 rms -4.8 | R peak 5.7 rms -5.4 #f45b82fb76bb51d3
        released: L peak 1.1 rms -17.0 | R peak 1.6 rms -19.5 #d22b86d1442735bc

        BS Growl Bass I:
        chord: L peak 0.5 rms -6.0 | R peak 1.2 rms -6.0 #a25b6b195e5d5189
        released: L peak -6.1 rms -25.5 | R peak -4.4 rms -25.6 #e1c5a385d65ecc97

        BS Growl Bass II:
        chord: L peak 2.9 rms -9.1 | R peak 0.7 rms -10.3 #6b5e52d3a5b0d36e
        released: L peak -2.6 rms -19.7 | R peak -2.5 rms -20.9 #019b9a369bbd5ac4

        BS Jazz Bass Guitar I:
        chord: L peak 0.1 rms -6.9 | R peak 0.1 rms -6.9 #6a60b1e09be57b4e
        released: L peak -8.1 rms -30.9 | R peak -8.7 rms -30.8 #c0445946cd47da50

        BS Pluck Bass I:
        chord: L peak 0.3 rms -7.5 | R peak -1.0 rms -10.7 #e87f3953bb8ae474
        released: L peak -4.0 rms -22.5 | R peak -9.7 rms -27.2 #9de40a63535f047b

        BS Pluck Bass II:
        chord: L peak -0.4 rms -10.7 | R peak -0.3 rms -11.3 #c8fdf35341ea4f4e
        released: L peak -55.8 rms -66.5 | R peak -55.7 rms -66.5 #34803e97e73c4416

        BS Punch Bass I:
        chord: L peak 0.0 rms -4.3 | R peak 0.0 rms -3.6 #4d0b0c10e7b0d2ea
        released: L peak -2.2 rms -25.2 | R peak -0.9 rms -25.5 #3b1d9ae647423fc6

        BS Punch Bass II:
        chord: L peak -1.0 rms -5.0 | R peak 0.5 rms -3.3 #6ac76dea7fec1618
        released: L peak -2.8 rms -25.3 | R peak -1.2 rms -24.0 #7c723da6e36a097e

        CH Min Chord I:
        chord: L peak 4.3 rms -4.7 | R peak 4.1 rms -4.4 #7da672c1bdfe44e9
        released: L peak 1.1 rms -18.6 | R peak 2.0 rms -18.3 #3d33ef31b52a0ac1

        CH Min Chord II:
        chord: L peak -2.8 rms -19.4 | R peak -5.6 rms -19.7 #ab737e74d4c94960
        released: L peak -25.9 rms -42.3 | R peak -26.4 rms -42.9 #cfa8e66563a215fb

        CH Min Chord III:
        chord: L peak -0.6 rms -12.2 | R peak -1.2 rms -13.1 #c4f63d662b917714
        released: L peak -18.6 rms -34.6 | R peak -18.9 rms -33.9 #e8ef99180ad17d12

        DR Basic Hat I:
        chord: L peak -4.3 rms -25.4 | R peak -4.9 rms -25.5 #2dd9f6cae78eccba
        released: L peak -118.6 rms -135.7 | R peak -117.8 rms -135.4 #5a539dee1266cf98

        DR Basic Kick:
        chord: L peak -0.1 rms -13.1 | R peak -0.2 rms -13.2 #ce8ce65b5ce757f3
        released: L peak -104.6 rms -133.3 | R peak -105.2 rms -133.7 #2c73c646bd8913d2

        DR Basic Snare I:
        chord: L peak -4.6 rms -24.1 | R peak -4.6 rms -23.9 #c8c2717e735e5797
        released: L peak -65.8 rms -96.5 | R peak -65.3 rms -91.7 #7ecd88416b90d0e2

        DR Hardcore Kick I:
        chord: L peak 0.0 rms -2.1 | R peak 0.0 rms -2.1 #70dc621287d1fd98
        released: L peak -0.6 rms -33.7 | R peak -0.4 rms -33.4 #59c7eb8e15a4942f

        DR Noise Tom I:
        chord: L peak -0.5 rms -11.0 | R peak -1.5 rms -11.6 #9a0c025c537d6a21
        released: L peak -10.1 rms -24.4 | R peak -12.8 rms -24.5 #41964657e8da3046

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_flamedragonz_second_half() {
    factory_presets(
        "Flamedragonz",
        1,
        &expect![[r#"
        FX Filter Growl I:
        chord: L peak 1.5 rms -3.4 | R peak 2.3 rms -3.4 #30be25deaa6fd595
        released: L peak -6.0 rms -22.5 | R peak -4.7 rms -21.2 #0fae654290164152

        FX Glitch:
        chord: L peak 3.3 rms -6.5 | R peak 3.2 rms -6.9 #598a3a0c19a54e46
        released: L peak 0.4 rms -22.8 | R peak 0.8 rms -23.6 #1eb74f9aac7e2d81

        FX Noise Explosion:
        chord: L peak 3.5 rms -6.2 | R peak 3.8 rms -6.1 #d07d4e6baaa79b01
        released: L peak 2.3 rms -7.5 | R peak 3.2 rms -6.9 #a6c0af3e893df6a4

        FX Red Alert I:
        chord: L peak -5.1 rms -16.8 | R peak -5.7 rms -17.1 #589b2bf1721ae9c0
        released: L peak -12.1 rms -32.6 | R peak -11.6 rms -33.1 #0aeb93d0a743605c

        FX Scratch I:
        chord: L peak -0.1 rms -9.9 | R peak -0.1 rms -9.9 #92af9ea8a2fc342d
        released: L peak -6.8 rms -30.7 | R peak -6.8 rms -30.7 #c978e7f5bbda5431

        LD Accordion I:
        chord: L peak 2.5 rms -7.1 | R peak 2.0 rms -7.1 #7dfd3c78eedc842c
        released: L peak -5.2 rms -26.4 | R peak -4.5 rms -26.8 #6bb431b626feed8e

        LD ACID KEY I:
        chord: L peak -6.3 rms -14.8 | R peak -7.1 rms -14.5 #8c0d51ab7d71c814
        released: L peak -12.4 rms -25.0 | R peak -13.8 rms -25.4 #6684bd924a832e6e

        LD EGuitar Dist I:
        chord: L peak -0.4 rms -11.5 | R peak 0.9 rms -11.4 #19ec71b8b8e10117
        released: L peak -1.6 rms -16.4 | R peak -2.2 rms -16.7 #5818886437c3e322

        LD NoiseSaw Lead I:
        chord: L peak -6.4 rms -16.2 | R peak -6.7 rms -16.6 #eac309d3a8150146
        released: L peak -19.6 rms -35.0 | R peak -16.9 rms -33.8 #4713b5eec4990953

        LD SawSquare Lead I:
        chord: L peak 2.4 rms -5.6 | R peak 0.4 rms -7.1 #875e8c12b6646f09
        released: L peak -3.5 rms -14.4 | R peak -7.2 rms -17.7 #833a20b151a1cdc9

        LD Slide Saw I:
        chord: L peak -2.6 rms -9.7 | R peak -1.7 rms -9.1 #9e1af0798e4bdefc
        released: L peak -6.8 rms -27.6 | R peak -2.4 rms -24.4 #f3d2cd4cebea29a1

        PD Abstract Choir I:
        chord: L peak -1.9 rms -14.0 | R peak -2.4 rms -13.7 #61515ff532f8171f
        released: L peak -3.7 rms -23.1 | R peak -4.3 rms -23.5 #45be493aaa651ed8

        PD Analog Pad I:
        chord: L peak -7.7 rms -19.3 | R peak -2.8 rms -17.7 #696b2e97f6a4f9f6
        released: L peak -6.4 rms -19.4 | R peak -4.5 rms -16.3 #af6356e449d5a4be

        PD Fantasy Pad I:
        chord: L peak -7.3 rms -21.1 | R peak -7.0 rms -22.1 #60b4ab502332d6e8
        released: L peak -4.8 rms -15.6 | R peak -6.3 rms -17.6 #0e77e6677f09bf33

        PD Fantasy Pad II:
        chord: L peak -9.9 rms -22.5 | R peak -9.3 rms -24.0 #0f171cbff4d6a30a
        released: L peak -8.2 rms -20.6 | R peak -9.0 rms -21.0 #8335b587d1fb3d5c

        PD Neu Atmopad I:
        chord: L peak -14.6 rms -26.2 | R peak -20.4 rms -32.7 #dd6feeedac822213
        released: L peak -14.9 rms -25.1 | R peak -17.9 rms -28.5 #990225dc6091a278

        PD Step Pad I:
        chord: L peak 2.7 rms -6.3 | R peak 3.8 rms -5.7 #7d0c99e7e54c37e6
        released: L peak -1.2 rms -20.2 | R peak -0.7 rms -19.7 #365e2db54da0f86a

        PL Cow Bell II:
        chord: L peak 3.7 rms -10.3 | R peak 5.1 rms -10.1 #3b5cc328b0ba2201
        released: L peak -4.9 rms -18.4 | R peak -3.4 rms -18.5 #c4e5a439def7498d

        PL Marimba I:
        chord: L peak -0.4 rms -13.0 | R peak -0.9 rms -13.2 #4c47ebb86e7fd41f
        released: L peak -9.8 rms -21.4 | R peak -10.1 rms -22.6 #5242eb2bef5c3ccd

        PL Noise String I:
        chord: L peak -0.9 rms -10.1 | R peak -2.1 rms -10.8 #4ae62a2e0c6528fa
        released: L peak -11.3 rms -20.0 | R peak -8.9 rms -20.2 #bc9b578af2d5f60d

        PL SawSquare I:
        chord: L peak 0.2 rms -16.5 | R peak -2.5 rms -18.9 #155f551ab92e7f1b
        released: L peak -16.3 rms -30.8 | R peak -20.2 rms -32.3 #c0c72c7c0b53c40c

        PL SawSquare II:
        chord: L peak 3.3 rms -11.6 | R peak 3.3 rms -11.0 #ed1bd08b3fc17bb7
        released: L peak -7.6 rms -21.9 | R peak -7.2 rms -21.4 #de8b95e43787d895

        PL Trance Pluck I:
        chord: L peak -0.8 rms -13.9 | R peak -0.7 rms -12.4 #fbf15495585a12c4
        released: L peak -6.4 rms -20.5 | R peak -7.0 rms -18.9 #64ab1ce0bc7aefe8

        PR 808 Cow Bell I:
        chord: L peak -8.5 rms -28.4 | R peak -8.5 rms -28.4 #1ad020a64c7e0769
        released: L peak -142.7 rms -158.3 | R peak -142.7 rms -158.3 #63df5bd1ca9c2e75

        ST Hype I:
        chord: L peak 1.3 rms -7.0 | R peak 1.3 rms -6.6 #0ce4cc3682922efa
        released: L peak -7.2 rms -19.5 | R peak -5.3 rms -18.6 #61d5ed8eca4ef2df

        VOX Dist Vocal I:
        chord: L peak -5.8 rms -18.0 | R peak -7.3 rms -18.2 #3cd06e48a823599c
        released: L peak -10.4 rms -23.5 | R peak -8.2 rms -23.2 #be8c2bff54254b19

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_havensteps_first_half() {
    factory_presets(
        "HavenSteps",
        0,
        &expect![[r#"
        8bit dream:
        chord: L peak -10.5 rms -22.2 | R peak -10.5 rms -22.2 #542c3ca247f3cec9
        released: L peak -10.1 rms -22.6 | R peak -10.1 rms -22.6 #11d9ebdc87693795

        16bit Dream:
        chord: L peak -13.9 rms -28.7 | R peak -10.3 rms -22.9 #ca2a500b00dc42dd
        released: L peak -inf rms -inf | R peak -57.7 rms -101.5 #0a984e7a615bf513

        80's 4th:
        chord: L peak -1.7 rms -10.8 | R peak -2.6 rms -11.0 #b8982c8c67588257
        released: L peak -2.7 rms -13.5 | R peak -3.2 rms -14.2 #02590e8423125bc6

        80s Dirty Lead:
        chord: L peak 0.6 rms -10.8 | R peak 1.5 rms -9.1 #0cd11202bf230d6d
        released: L peak -3.0 rms -22.0 | R peak -3.5 rms -20.1 #9539e96c836b894f

        90sEP:
        chord: L peak -2.7 rms -12.0 | R peak -2.6 rms -11.4 #f57abc033dce5f16
        released: L peak -2.4 rms -15.3 | R peak -5.4 rms -16.9 #1ecf161c0c353310

        Alejandr0:
        chord: L peak 5.4 rms -5.1 | R peak 5.1 rms -4.1 #1abbadda9646a89f
        released: L peak 4.2 rms -9.3 | R peak 4.0 rms -8.2 #27d871a7542b3ca3

        Angles:
        chord: L peak -1.9 rms -14.1 | R peak -2.1 rms -15.2 #9983f1c35cdda737
        released: L peak -2.2 rms -12.8 | R peak -2.2 rms -14.2 #2fdc83e905c29163

        Bassison:
        chord: L peak -4.8 rms -12.7 | R peak -4.2 rms -12.5 #1ae0273b29df2c03
        released: L peak -4.9 rms -24.4 | R peak -5.0 rms -22.7 #d77f1f68a83cf153

        Bassity:
        chord: L peak 1.0 rms -7.3 | R peak 1.2 rms -7.1 #a136e7ca169cfb49
        released: L peak -3.8 rms -16.9 | R peak -1.9 rms -15.5 #578443ee5f165cad

        BeauKeys:
        chord: L peak -1.8 rms -9.2 | R peak 1.3 rms -6.2 #d56cc433b8d8bb84
        released: L peak -5.5 rms -21.3 | R peak 0.3 rms -18.1 #ed5cd50ff2a21bce

        BossB:
        chord: L peak 0.1 rms -6.5 | R peak 0.1 rms -6.5 #b0c2b6a01315386d
        released: L peak -5.3 rms -25.4 | R peak -5.3 rms -25.4 #584cc7e39fc43f05

        BrokenCom:
        chord: L peak 6.5 rms -3.2 | R peak 5.7 rms -4.5 #19e03340d418e3ae
        released: L peak -inf rms -inf | R peak -13.5 rms -41.5 #0fa17bfb4fe13ebe

        Cimbalom:
        chord: L peak 0.0 rms -11.0 | R peak 0.0 rms -11.0 #5621c1584feb7ea9
        released: L peak -7.4 rms -22.8 | R peak -7.4 rms -22.8 #d980ecfada783225

        Concuas:
        chord: L peak -0.8 rms -9.2 | R peak 1.9 rms -8.4 #29cb4239559d4c28
        released: L peak -1.3 rms -14.4 | R peak -0.5 rms -14.0 #20775417eeb85b50

        CrazyBrass:
        chord: L peak 9.7 rms -2.7 | R peak 9.7 rms -2.7 #993e3e97ec67b659
        released: L peak 8.9 rms -6.2 | R peak 8.9 rms -6.2 #2d8dce3f4461010d

        CrazyOrgan:
        chord: L peak -2.0 rms -12.1 | R peak -1.0 rms -11.4 #06ad136d37adc8aa
        released: L peak -8.5 rms -36.0 | R peak -9.2 rms -36.5 #2e9e80d65b484ace

        DayDreaming:
        chord: L peak -22.9 rms -34.8 | R peak -22.8 rms -35.4 #74ec38294793807f
        released: L peak -23.7 rms -35.9 | R peak -20.3 rms -29.1 #b9d3e3bbc8bedd54

        Decisive:
        chord: L peak -10.4 rms -21.2 | R peak -9.4 rms -21.2 #63299a246276f5ba
        released: L peak -10.6 rms -25.4 | R peak -9.6 rms -25.0 #87d33225be253bd9

        Dramahit:
        chord: L peak 3.9 rms -9.3 | R peak 4.5 rms -8.7 #c9c57a31afdb4e18
        released: L peak -16.3 rms -30.5 | R peak -15.7 rms -30.1 #7fe1f900a3432fa7

        DramaLead:
        chord: L peak 8.5 rms -7.8 | R peak 8.6 rms -7.3 #334fda50f523301a
        released: L peak -11.0 rms -25.3 | R peak -11.0 rms -25.1 #9d95d5ddcc40d932

        Duke Bass:
        chord: L peak 0.3 rms -6.6 | R peak -0.5 rms -7.3 #c5bf3e6f6e09fef8
        released: L peak -3.8 rms -26.5 | R peak -6.8 rms -28.1 #2f390d0ffc314884

        E-Bass:
        chord: L peak 7.1 rms -2.5 | R peak 7.1 rms -2.5 #e6fab0e7aad53221
        released: L peak -7.8 rms -34.8 | R peak -7.8 rms -34.8 #af7f079eb54b0a8d

        Flutish:
        chord: L peak 3.1 rms -6.3 | R peak 4.4 rms -6.3 #0f8cb49f9c4ad7be
        released: L peak -0.9 rms -22.0 | R peak -1.5 rms -22.7 #70fd94ace06df14f

        Game Bass:
        chord: L peak -2.1 rms -11.7 | R peak -1.9 rms -11.9 #d4d875be6c43ee46
        released: L peak -25.6 rms -49.6 | R peak -26.3 rms -50.0 #5de9e1fc2ad543f5

        HavenSteps:
        chord: L peak -12.1 rms -27.9 | R peak -12.1 rms -27.9 #e8641f6b5edfa64f
        released: L peak -5.1 rms -20.5 | R peak -5.1 rms -20.5 #1696a5aa14e677f3

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_havensteps_second_half() {
    factory_presets(
        "HavenSteps",
        1,
        &expect![[r#"
        Lazzzer:
        chord: L peak -0.2 rms -17.8 | R peak -0.2 rms -17.8 #3ff4f93d4b41d0c9
        released: L peak -135.5 rms -166.1 | R peak -135.5 rms -166.1 #e28a25f8ef4e13c5

        LostTechno:
        chord: L peak -6.0 rms -15.5 | R peak -8.1 rms -18.4 #9b12024dcc3ef038
        released: L peak -13.0 rms -38.2 | R peak -10.8 rms -38.2 #22895fc242201ce4

        Medas4:
        chord: L peak -4.3 rms -22.2 | R peak -4.3 rms -22.2 #9912d41e187a0d6d
        released: L peak -5.7 rms -27.8 | R peak -5.7 rms -27.8 #8badf2665cb0e845

        Monoglis Madness:
        chord: L peak 5.4 rms -4.6 | R peak 5.0 rms -4.9 #50a68eaeaf3bfbf5
        released: L peak 1.9 rms -21.9 | R peak 2.7 rms -21.4 #cf0378d72c5e0db3

        MovieTrailerS:
        chord: L peak -9.1 rms -21.9 | R peak -8.7 rms -20.2 #c1d46679580af058
        released: L peak -6.9 rms -18.3 | R peak -7.6 rms -17.4 #a06f7ae1a74777ab

        Navi:
        chord: L peak -50.4 rms -61.7 | R peak -50.4 rms -61.7 #e8fcf9e2971122dd
        released: L peak -50.9 rms -60.5 | R peak -50.9 rms -60.5 #1576116a7fa83d8d

        Nostromo:
        chord: L peak 2.8 rms -9.5 | R peak 2.8 rms -9.5 #a9956b50887af0ed
        released: L peak -0.4 rms -25.4 | R peak -0.4 rms -25.4 #dea3f7a458863c15

        NylonSteel:
        chord: L peak -7.6 rms -22.3 | R peak -7.6 rms -22.3 #e9b43b0e5ef54305
        released: L peak -31.8 rms -45.5 | R peak -31.8 rms -45.5 #bfea8ce22d19e08d

        PadThai:
        chord: L peak -8.2 rms -23.5 | R peak -11.9 rms -24.0 #219211829db43276
        released: L peak -7.8 rms -18.8 | R peak -9.7 rms -20.9 #e6e3b4db4c95fd99

        PalmBlass:
        chord: L peak -8.4 rms -11.5 | R peak -8.4 rms -10.8 #0654c1513bf572da
        released: L peak -8.4 rms -23.4 | R peak -8.4 rms -23.0 #aba23dec79ea436e

        Play That Funky Music:
        chord: L peak -3.5 rms -16.3 | R peak -2.2 rms -15.5 #fecce4b33ea4a00e
        released: L peak -38.2 rms -60.6 | R peak -37.0 rms -59.4 #da3b91cb70dcd5f5

        PLead:
        chord: L peak 6.1 rms -5.3 | R peak 7.0 rms -5.9 #57c3b7dd7c3223cc
        released: L peak 2.1 rms -11.3 | R peak 2.8 rms -11.6 #65609dea68fe7ccb

        Pulse Droner:
        chord: L peak -6.6 rms -15.0 | R peak -6.5 rms -14.8 #6c1043f9ef98f381
        released: L peak -7.0 rms -19.2 | R peak -6.9 rms -19.5 #5ee3b3566350c9ae

        Raiser:
        chord: L peak -24.5 rms -37.2 | R peak -21.1 rms -31.4 #452760bd47a7929a
        released: L peak -27.8 rms -50.5 | R peak -25.7 rms -45.0 #d0a58f3da1c6812e

        Rhodeys:
        chord: L peak -10.2 rms -18.0 | R peak -10.9 rms -18.0 #8ede7def858ec5ab
        released: L peak -14.2 rms -28.4 | R peak -14.5 rms -28.0 #b59a941817f4dca3

        Shamisen:
        chord: L peak -4.3 rms -19.9 | R peak -4.3 rms -19.9 #87968c72c2a5df81
        released: L peak -21.3 rms -37.8 | R peak -21.3 rms -37.8 #7194f9d427c14f19

        Simbad:
        chord: L peak -3.8 rms -14.0 | R peak -1.8 rms -13.3 #91462e389612f3d3
        released: L peak -4.4 rms -17.1 | R peak -4.3 rms -17.2 #c231628fd37c40ae

        Snare:
        chord: L peak -3.0 rms -28.5 | R peak -3.0 rms -29.0 #91f19cb73f87a26c
        released: L peak -55.8 rms -73.6 | R peak -54.5 rms -73.0 #a84993cf583d6252

        SomeChurch:
        chord: L peak -4.9 rms -15.2 | R peak -4.0 rms -14.8 #59fcce4ef28c25be
        released: L peak -9.4 rms -36.7 | R peak -10.1 rms -37.6 #3c1ac1de8906f6bc

        Stars:
        chord: L peak -0.8 rms -15.9 | R peak -0.8 rms -15.9 #079c029f3e66e32b
        released: L peak 0.7 rms -12.5 | R peak 0.7 rms -12.4 #84070450d19b10ab

        SteelS:
        chord: L peak -6.8 rms -21.0 | R peak -6.8 rms -21.0 #fbb36952bf899f91
        released: L peak -22.6 rms -36.9 | R peak -22.6 rms -36.9 #7df10e52bfbfc425

        Sweeper:
        chord: L peak -17.6 rms -31.5 | R peak -16.8 rms -30.8 #b11810a0c6fa612e
        released: L peak -16.0 rms -28.2 | R peak -16.6 rms -29.1 #ad1a645c1d4bcd2b

        The Classic:
        chord: L peak 0.0 rms -3.7 | R peak 0.0 rms -3.8 #3a82d1da1bfe64ed
        released: L peak 0.0 rms -19.5 | R peak 0.0 rms -19.5 #4a4501fe73b33b87

        Unistep:
        chord: L peak 1.9 rms -10.1 | R peak 1.2 rms -8.7 #8950827528f4415f
        released: L peak -inf rms -inf | R peak -inf rms -inf #9ebf9a6ec921bb25

        WoW:
        chord: L peak 2.3 rms -7.1 | R peak 5.2 rms -4.3 #9e2d1811848f7f81
        released: L peak -0.7 rms -14.8 | R peak 3.4 rms -12.1 #1dc86c1698a6111a

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_iv_first_half() {
    factory_presets(
        "IV",
        0,
        &expect![[r#"
        IV's Cyber Lead (MW):
        chord: L peak -5.4 rms -15.7 | R peak -6.1 rms -15.9 #22f5d7ce622c8b94
        released: L peak -7.9 rms -25.1 | R peak -9.5 rms -26.5 #1b976e47e225851c

        IV's Heart Bells:
        chord: L peak -7.8 rms -16.3 | R peak -6.2 rms -17.1 #54c858cad9e9bf1b
        released: L peak -8.3 rms -17.3 | R peak -5.9 rms -16.9 #a5ef87b61b392e56

        IV's Init:
        chord: L peak -5.6 rms -14.6 | R peak -5.6 rms -14.6 #16bf759133c71809
        released: L peak -10.2 rms -23.7 | R peak -10.2 rms -23.7 #0a31fb6f8e0e807d

        IV's Organ Advanced:
        chord: L peak -16.1 rms -27.5 | R peak -17.9 rms -29.4 #bd149e2cd2b4311f
        released: L peak -19.3 rms -34.8 | R peak -22.2 rms -34.0 #2151ee77ba990a4f

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_iv_second_half() {
    factory_presets(
        "IV",
        1,
        &expect![[r#"
        IV's Organ Basic:
        chord: L peak -8.3 rms -19.8 | R peak -9.0 rms -20.7 #2eef7ad8bb7ee6d8
        released: L peak -13.4 rms -37.7 | R peak -13.9 rms -38.6 #11a425cc4c3441ac

        IV's Organ Superchorus:
        chord: L peak -4.2 rms -16.9 | R peak -4.5 rms -17.0 #01eba18c870908eb
        released: L peak -8.4 rms -24.0 | R peak -6.9 rms -24.1 #4a3361af3dcd9246

        IV's Pseudo Pizzicato:
        chord: L peak -5.2 rms -17.3 | R peak -3.9 rms -16.0 #5824d86ba8c2af94
        released: L peak -8.1 rms -24.0 | R peak -11.2 rms -25.0 #c5cc62dbb0479b9a

        IV's Rhytmic Metall Bass:
        chord: L peak -6.0 rms -16.5 | R peak -6.7 rms -17.5 #6f55544ee78471b6
        released: L peak -10.0 rms -26.7 | R peak -10.6 rms -27.5 #76f1e4864d742858

        IV's Simple Plucks:
        chord: L peak -3.0 rms -11.3 | R peak -3.2 rms -11.3 #d9ae202c310fa4a3
        released: L peak -6.3 rms -16.9 | R peak -6.3 rms -16.9 #e3ef68f3a890ef9c

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_madmcman_first_half() {
    factory_presets(
        "MadMcMan",
        0,
        &expect![[r#"
        Angel Voice:
        chord: L peak -2.3 rms -15.6 | R peak 2.0 rms -7.4 #41e08213d44554c6
        released: L peak -7.4 rms -22.2 | R peak 0.2 rms -19.7 #c8cf43d0161cf3a7

        Bowed Pad:
        chord: L peak -8.8 rms -20.7 | R peak -6.7 rms -19.2 #26f16825cf312637
        released: L peak -7.1 rms -18.8 | R peak -7.1 rms -17.4 #b9af43a08c0b1119

        Chorus Lead:
        chord: L peak -11.0 rms -22.8 | R peak -12.8 rms -25.5 #8295f6e919140d62
        released: L peak -12.1 rms -23.4 | R peak -12.9 rms -24.5 #fcdc64345dbd79e0

        Guitar Pad:
        chord: L peak -18.6 rms -29.6 | R peak -19.5 rms -29.7 #ffb8e54062c2aadd
        released: L peak -17.9 rms -29.0 | R peak -19.1 rms -29.1 #0d95b63f7ee10331

        Hat:
        chord: L peak -21.9 rms -39.9 | R peak -21.9 rms -39.9 #9da20b242ea60651
        released: L peak -32.9 rms -57.9 | R peak -32.9 rms -57.9 #93be59278f71b081

        hh:
        chord: L peak -26.2 rms -46.4 | R peak -25.1 rms -46.5 #2420a49b7d16eef7
        released: L peak -33.6 rms -66.1 | R peak -33.5 rms -66.0 #7cd12c82804d3a39

        High Pad:
        chord: L peak -7.1 rms -16.8 | R peak -5.5 rms -15.0 #a3ebe9d0feebf58d
        released: L peak -5.1 rms -15.3 | R peak -4.3 rms -14.3 #bdf6b245677e53d4

        Horny:
        chord: L peak -0.8 rms -11.4 | R peak -1.0 rms -11.3 #92fd0ca4720dc602
        released: L peak -0.8 rms -17.7 | R peak -1.1 rms -17.7 #26cc6b4d575b5146

        Kick:
        chord: L peak -4.9 rms -9.0 | R peak -4.9 rms -9.0 #6c8f53878455376d
        released: L peak -6.2 rms -19.0 | R peak -6.2 rms -19.0 #ad3f452af5897909

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_madmcman_second_half() {
    factory_presets(
        "MadMcMan",
        1,
        &expect![[r#"
        Mice:
        chord: L peak -1.5 rms -29.4 | R peak -1.9 rms -29.0 #97853045ee407e5e
        released: L peak -22.0 rms -37.5 | R peak -25.2 rms -41.8 #61063abbe9b87478

        Moving Pad:
        chord: L peak -4.3 rms -13.9 | R peak -2.8 rms -13.5 #133e00eb18115f65
        released: L peak -2.7 rms -14.3 | R peak -2.9 rms -14.2 #eedae86eea73778d

        Noise Riserish:
        chord: L peak -27.9 rms -40.6 | R peak -28.4 rms -40.7 #f8c9fd73073cb7c7
        released: L peak -25.0 rms -37.9 | R peak -25.9 rms -37.8 #81660ae294205001

        PLOP:
        chord: L peak -17.2 rms -33.4 | R peak -18.1 rms -34.3 #7138ce2f5da5db27
        released: L peak -68.5 rms -83.1 | R peak -69.4 rms -85.0 #057781b03e28963a

        Reverby Keys:
        chord: L peak 0.7 rms -10.1 | R peak 4.2 rms -3.8 #db66a1f23798e340
        released: L peak -5.3 rms -14.7 | R peak 1.7 rms -9.0 #c779166002ef37ad

        talky distortion:
        chord: L peak 6.8 rms -6.6 | R peak 5.9 rms -6.6 #1ae707ffa2d43351
        released: L peak 3.6 rms -13.0 | R peak 4.9 rms -13.1 #c55cd1b5530d16fa

        Twinkle Mallet:
        chord: L peak -5.1 rms -12.3 | R peak -2.7 rms -12.3 #c7a293caea02af72
        released: L peak -5.7 rms -15.8 | R peak -7.3 rms -16.0 #c8018d3661d2bb6f

        Twinkle:
        chord: L peak -10.6 rms -22.0 | R peak -9.6 rms -22.2 #fb93c16659b09838
        released: L peak -10.6 rms -22.8 | R peak -8.1 rms -22.7 #ec8016c4357e146a

        Whip:
        chord: L peak 1.0 rms -14.8 | R peak 1.4 rms -14.9 #7b0119f526fd1a3b
        released: L peak -5.4 rms -23.0 | R peak -4.3 rms -23.0 #ffd989de203013f6

        Wobly Pad:
        chord: L peak -2.1 rms -14.9 | R peak -2.6 rms -14.7 #7e01a7454a0258b1
        released: L peak -8.1 rms -25.6 | R peak -9.6 rms -29.0 #60255855292dc5ab

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_murti_first_half() {
    factory_presets(
        "murti",
        0,
        &expect![[r#"
        5th Wheel:
        chord: L peak -3.1 rms -14.4 | R peak -2.8 rms -13.8 #0633e9178308fe52
        released: L peak -33.7 rms -50.9 | R peak -33.6 rms -52.5 #5aa8695bd5306c51

        7th Heaven:
        chord: L peak -0.2 rms -9.5 | R peak -0.1 rms -9.6 #c1948b100029f110
        released: L peak -19.3 rms -36.8 | R peak -18.3 rms -37.7 #9813039588759045

        A1:
        chord: L peak -0.2 rms -9.1 | R peak -0.8 rms -9.1 #75a2786f123ed163
        released: L peak -2.7 rms -15.3 | R peak -4.5 rms -15.9 #360821c5e5df9680

        Augmented Reality:
        chord: L peak -0.2 rms -9.5 | R peak -1.8 rms -13.1 #c50b710a0386d30e
        released: L peak -7.1 rms -19.5 | R peak -8.2 rms -28.9 #b2d718c6d242ea75

        Bag Drone:
        chord: L peak -5.5 rms -15.1 | R peak -7.6 rms -18.2 #8f1668387e763fa7
        released: L peak -9.5 rms -19.7 | R peak -11.1 rms -22.2 #e1234c2b79465745

        Big Boy Reese:
        chord: L peak -5.0 rms -6.7 | R peak -5.0 rms -6.7 #4a88789c361a100d
        released: L peak -5.0 rms -20.0 | R peak -5.0 rms -20.1 #b25c662f3d44b581

        Big Boy Sub:
        chord: L peak -4.1 rms -8.7 | R peak -3.7 rms -8.4 #8f4e66f9edbd4577
        released: L peak -7.0 rms -25.8 | R peak -6.8 rms -25.5 #434e9c5a467c9e71

        Biter:
        chord: L peak -5.0 rms -7.1 | R peak -5.0 rms -7.1 #c6731dad948bcabc
        released: L peak -5.0 rms -19.7 | R peak -5.0 rms -19.9 #e206840cf4ea68a7

        Breathe:
        chord: L peak -0.1 rms -10.7 | R peak 0.0 rms -8.8 #2a7f641e8b4c81fa
        released: L peak -2.5 rms -16.5 | R peak -1.1 rms -12.3 #2726a5800313ca0a

        Chill Son:
        chord: L peak -12.1 rms -20.7 | R peak -8.6 rms -16.6 #9c093de13916f14f
        released: L peak -15.5 rms -23.2 | R peak -10.6 rms -20.5 #e31e34a7d6dde645

        Chop Saw:
        chord: L peak -0.9 rms -14.3 | R peak -1.4 rms -15.4 #29f0e30d28bba9a4
        released: L peak -8.9 rms -29.7 | R peak -11.1 rms -30.8 #9b15622a8fd3cf5b

        Clean Dirt:
        chord: L peak -0.9 rms -10.1 | R peak -0.8 rms -9.7 #cdcbd324d2f665db
        released: L peak -6.4 rms -18.4 | R peak -8.6 rms -17.1 #be08fd7d36870d4a

        Collide:
        chord: L peak 0.0 rms -11.1 | R peak -0.2 rms -11.1 #ed5af384e929ab2e
        released: L peak -0.9 rms -15.4 | R peak -0.8 rms -15.4 #37b6458cf33643fd

        Delay Compass:
        chord: L peak -1.3 rms -14.6 | R peak -1.3 rms -14.8 #3f681e7f8c8a3f35
        released: L peak -22.6 rms -36.8 | R peak -9.2 rms -25.7 #b7e804d6cda46c9a

        Dirty Dog:
        chord: L peak -5.0 rms -6.9 | R peak -5.0 rms -6.9 #896185648fed0cef
        released: L peak -5.0 rms -20.3 | R peak -5.2 rms -20.4 #92a7fbf750f898ee

        Distant Relative:
        chord: L peak -2.2 rms -13.2 | R peak -1.9 rms -12.5 #b2fd4263162fbb47
        released: L peak -9.2 rms -29.9 | R peak -10.2 rms -28.9 #b0a6bfcc25a59ef5

        Dragonball:
        chord: L peak 0.2 rms -6.7 | R peak -0.5 rms -8.0 #0ae4587289dc425d
        released: L peak -2.6 rms -19.0 | R peak -3.0 rms -20.8 #429fb62906b3f443

        Flower:
        chord: L peak -1.4 rms -15.5 | R peak -0.3 rms -14.8 #8c5c7c38fed4eea6
        released: L peak -11.7 rms -24.9 | R peak -13.2 rms -26.5 #78e3c1311b7e5896

        Foul Snap:
        chord: L peak -1.4 rms -11.6 | R peak -1.4 rms -11.3 #e8981f4e13370238
        released: L peak -17.0 rms -38.4 | R peak -16.5 rms -37.5 #90f639dfabbd06f0

        Heaven Ascent:
        chord: L peak -0.3 rms -13.0 | R peak 3.1 rms -5.2 #f0e4634ff2651074
        released: L peak -2.5 rms -16.7 | R peak 1.3 rms -11.0 #4e1c89b4a65ac545

        Hollow Bells:
        chord: L peak -2.4 rms -16.0 | R peak -1.3 rms -15.7 #edfd8de654506d9c
        released: L peak -21.3 rms -33.9 | R peak -19.5 rms -32.9 #47f5b55cce2913b6

        Lurkin:
        chord: L peak -1.4 rms -10.6 | R peak -3.1 rms -12.4 #5267c576cfcaf18a
        released: L peak -2.0 rms -24.3 | R peak -5.4 rms -28.2 #8bf7390c482a423d

        Madness:
        chord: L peak -5.0 rms -6.8 | R peak -5.0 rms -6.8 #1930450a01ba3ced
        released: L peak -5.6 rms -20.3 | R peak -5.6 rms -20.4 #0239d0f8420eb97c

        Magnet Bass:
        chord: L peak -0.4 rms -10.9 | R peak -1.6 rms -12.1 #9db79189ee1c567e
        released: L peak -2.2 rms -29.7 | R peak -5.4 rms -30.7 #5e02b212c16d5674

        Modular Waves:
        chord: L peak 0.0 rms -8.4 | R peak 0.0 rms -8.4 #78c43b06e8262705
        released: L peak -2.7 rms -25.4 | R peak -2.7 rms -25.4 #665e182c632286a5

    "#]],
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_by_murti_second_half() {
    factory_presets(
        "murti",
        1,
        &expect![[r#"
        Munch Growl:
        chord: L peak 1.1 rms -10.3 | R peak 0.8 rms -9.8 #8e71c720c555751e
        released: L peak -4.0 rms -26.5 | R peak -2.5 rms -24.3 #abd4f7013d00f3d1

        Omni Box:
        chord: L peak 1.4 rms -10.1 | R peak -1.0 rms -11.5 #5c0d1f6fef3432c9
        released: L peak 2.7 rms -15.6 | R peak -2.9 rms -18.3 #36458d709e4b3179

        Petal Swell:
        chord: L peak -1.2 rms -12.3 | R peak -2.7 rms -13.1 #8897e0e1bea44c05
        released: L peak -4.7 rms -17.1 | R peak -3.3 rms -18.5 #b2f41d007b70c785

        Pluckd:
        chord: L peak -2.6 rms -13.8 | R peak -1.8 rms -14.2 #1eefd3d2d0cd66be
        released: L peak -23.6 rms -38.0 | R peak -23.4 rms -39.7 #2559ff0383a8c52e

        Plucky Juno:
        chord: L peak -2.1 rms -12.4 | R peak -2.1 rms -12.4 #ddbb09e62db17a65
        released: L peak -2.7 rms -20.1 | R peak -2.7 rms -20.1 #581c5c345a0116dd

        Polyrhythmz:
        chord: L peak -6.1 rms -21.4 | R peak -4.6 rms -18.1 #4be668d70610bc7a
        released: L peak -21.0 rms -42.2 | R peak -21.5 rms -43.0 #7c31d19bb9cf7a5f

        Psy Opp:
        chord: L peak 0.0 rms -5.0 | R peak 0.0 rms -4.9 #43dcff8fe3b26857
        released: L peak -2.5 rms -24.3 | R peak -2.8 rms -25.0 #a5eff7c22e9cd374

        Pulse:
        chord: L peak -2.1 rms -12.1 | R peak -2.2 rms -13.9 #14bee444d69fe7b6
        released: L peak -9.4 rms -21.9 | R peak -8.8 rms -21.4 #4c0e523d479496fb

        Self-Made:
        chord: L peak -11.0 rms -26.3 | R peak -7.1 rms -22.6 #2d8e72ddff6630d8
        released: L peak -5.9 rms -17.2 | R peak -2.2 rms -16.0 #0a544d18d10d8097

        Shimmy Time:
        chord: L peak 1.0 rms -6.6 | R peak 2.4 rms -4.3 #8167b6ee987b454a
        released: L peak -0.8 rms -17.4 | R peak -0.3 rms -16.6 #1f7d877af4389c35

        Sign Pad:
        chord: L peak -0.9 rms -8.2 | R peak 0.2 rms -7.9 #fcf84efcc2f056e1
        released: L peak -0.0 rms -11.6 | R peak -1.3 rms -11.2 #b38e5bc142df0cec

        Sine Wub:
        chord: L peak -0.1 rms -7.7 | R peak -0.2 rms -7.9 #43f22e3b44ebd9cf
        released: L peak -2.2 rms -23.6 | R peak -3.5 rms -25.8 #e1bcbe36f51fc847

        Space Organ:
        chord: L peak 0.2 rms -11.6 | R peak -0.9 rms -11.5 #a46f330f4e217fae
        released: L peak -3.6 rms -26.0 | R peak -4.2 rms -26.1 #70fe0b8218fb2c2b

        Square-1:
        chord: L peak -8.2 rms -14.8 | R peak -8.4 rms -15.6 #75047872d186b39a
        released: L peak -12.7 rms -33.1 | R peak -12.5 rms -33.3 #c59d1b8022d2ede3

        Strings:
        chord: L peak 1.1 rms -7.2 | R peak 1.5 rms -8.3 #83df73bbff43533e
        released: L peak -1.9 rms -16.4 | R peak 1.2 rms -15.4 #d09c9040ed8a3d3d

        Super Saw:
        chord: L peak -2.9 rms -13.1 | R peak -2.6 rms -13.1 #b3680646d70cd55b
        released: L peak -6.6 rms -23.9 | R peak -7.1 rms -23.9 #6177413ade58d2dc

        Swelling:
        chord: L peak 1.8 rms -7.8 | R peak 3.3 rms -6.4 #353e374a45d0b87d
        released: L peak -2.3 rms -15.4 | R peak 1.6 rms -12.6 #f2dd8c162a1a5b0a

        Swelter:
        chord: L peak -3.0 rms -10.3 | R peak -3.6 rms -10.4 #79a925d055465ee6
        released: L peak -0.1 rms -19.1 | R peak -0.2 rms -20.2 #dad8a42ff0533060

        Talky:
        chord: L peak -1.8 rms -11.6 | R peak -1.6 rms -11.4 #67b161896b297b2a
        released: L peak -4.5 rms -25.8 | R peak -7.7 rms -29.6 #8abe06207785ca89

        The Deep:
        chord: L peak -2.0 rms -3.8 | R peak -2.0 rms -3.8 #ce0f2832fe2a4e96
        released: L peak -2.6 rms -17.4 | R peak -2.5 rms -17.5 #7cc4bab3c7ac92fa

        Tree of Eden:
        chord: L peak -6.9 rms -12.7 | R peak -12.2 rms -22.2 #08b91c14fb4598f8
        released: L peak -8.0 rms -21.0 | R peak -13.7 rms -26.6 #9d5e648c77f766aa

        Triplet Pluck:
        chord: L peak 0.2 rms -13.7 | R peak -0.8 rms -13.6 #b4b1ef8871beace4
        released: L peak -5.2 rms -26.8 | R peak -5.1 rms -28.1 #5218c055ee666d37

        Vernacular:
        chord: L peak 2.7 rms -7.1 | R peak 2.5 rms -7.2 #3dca06c9e5d83616
        released: L peak 1.0 rms -14.8 | R peak 0.5 rms -15.2 #d679cb535be41b80

        Vide Bells:
        chord: L peak -4.8 rms -18.0 | R peak -4.6 rms -16.8 #c7bbb9263a7bb291
        released: L peak -13.4 rms -24.2 | R peak -14.5 rms -24.6 #e5d1db2199ecfbbe

        Vindicate:
        chord: L peak -3.2 rms -14.4 | R peak -2.4 rms -14.3 #1c8d6a9f6dbfec13
        released: L peak -5.9 rms -19.6 | R peak -3.8 rms -18.2 #a8b4737a1f560202

    "#]],
    );
}

// ---------------------------------------------------------------------------
// Preset files
// ---------------------------------------------------------------------------

/// Three factory presets, read from the files SocaLabs ships them as,
/// the way a user's own are (`preset_file`): one plain, one with 21
/// routes, and one that carries its table inside itself. Each plays as
/// the built-in preset does, bit for bit, and — under `gold` — as the
/// original plays the file.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn factory_presets_read_from_their_files() {
    let mut out = String::new();
    for (name, xml) in [
        ("DT Bwaing", include_str!("../fixtures/DT Bwaing.xml")),
        ("Angel Voice", include_str!("../fixtures/Angel Voice.xml")),
        (
            "IV's Organ Advanced",
            include_str!("../fixtures/IV Organ Advanced.xml"),
        ),
    ] {
        let read = bloop_plugin_wavetable::preset_file::read(xml.as_bytes())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let notes = |p: Program| p.note(60, 100, 0.5, 0.25).on(0, 67, 80).play("again", 0.25);
        let built_in = run_rust(&notes(Program::new().preset(name)));
        let from_file = notes(Program::new().preset_file(xml));
        let played = run_rust(&from_file);
        #[cfg(feature = "gold")]
        compare(&from_file, &played, &cpp::run(&from_file));
        let _ = writeln!(
            out,
            "{name}: {} parameters, {} routes, tables {:?}, left out {:?}, ignored {:?}; as the built-in preset: {}",
            read.params.len(),
            read.routes.len(),
            read.tables.map(|t| t.and_then(|t| bank::name_of(usize::from(t)))),
            read.left_out,
            read.ignored,
            played == built_in
        );
        assert_eq!(played, built_in, "{name} from its file");
    }
    expect![[r#"
        DT Bwaing: 55 parameters, 5 routes, tables [Some("FM 07"), Some("FM 05")], left out [], ignored []; as the built-in preset: true
        Angel Voice: 58 parameters, 21 routes, tables [Some("Analog PWM Square 01"), Some("NK - CONVENT")], left out [], ignored []; as the built-in preset: true
        IV's Organ Advanced: 73 parameters, 0 routes, tables [Some("15-Flenders III"), Some("15-Flenders III")], left out [], ignored []; as the built-in preset: true
    "#]].assert_eq(&out);
}

/// A preset of bloop's making, past what a factory preset asks of the
/// matrix: thirty routes, every curve, both mappings said and unsaid, a
/// route switched off, the same source on one destination twice, a
/// curve bloop does not know (linear, as the original takes it), and a
/// parameter there is not. Under `gold` the original reads the file
/// itself.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn a_preset_with_thirty_routes_and_curves() {
    let xml = include_str!("../fixtures/Thirty Routes.xml");
    let read =
        bloop_plugin_wavetable::preset_file::read(xml.as_bytes()).unwrap_or_else(|e| panic!("{e}"));
    let curves: std::collections::BTreeSet<usize> = read.routes.iter().map(|r| r.curve).collect();
    let header = format!(
        "{} routes ({} on), {} curves, mappings {:?}; left out {:?}; ignored {:?}\n",
        read.routes.len(),
        read.routes.iter().filter(|r| r.enabled).count(),
        curves.len(),
        read.routes.iter().map(|r| r.mapping).collect::<Vec<_>>(),
        read.left_out,
        read.ignored,
    );
    let program = Program::new()
        .preset_file(xml)
        .on(0, 48, 100)
        .on(0, 60, 70)
        .play("held", 1.5)
        .off(0, 48)
        .off(0, 60)
        .play("released", 0.5);
    let rust = run_rust(&program);
    #[cfg(feature = "gold")]
    compare(&program, &rust, &cpp::run(&program));
    assert_summary(
        &expect![[r#"
        30 routes (29 on), 20 curves, mappings [0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]; left out []; ignored ["the parameter notaparameter, which bloop does not have"]
        held: L peak -6.3 rms -16.9 | R peak -4.7 rms -16.4 #9a72afe9c14059f8
        released: L peak -11.3 rms -32.2 | R peak -13.7 rms -35.9 #ebdc89997726410e
    "#]],
        &(header + &summary(&rust)),
    );
}

/// More routes than the matrix holds, and more on one destination than
/// it runs: the preset loads, the rest play, and what was left out is
/// said, each in words.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_preset_past_the_matrix_s_room() {
    let mut items = String::new();
    // 40 on OSC1 Pos, then 230 spread over the LFO rates and the rest.
    for i in 0..40 {
        let _ = writeln!(
            items,
            r#"<MODITEM srcId="cc{i}" depth="0.01" dstId="osc1pos"/>"#
        );
    }
    let destinations = [
        "lfo1rate",
        "lfo2rate",
        "lfo3rate",
        "osc2pos",
        "fltfreq",
        "fltres",
        "osc1formant",
        "osc2formant",
    ];
    for i in 0..230 {
        let _ = writeln!(
            items,
            r#"<MODITEM srcId="cc{}" depth="0.01" dstId="{}"/>"#,
            i % 100,
            destinations[i % destinations.len()]
        );
    }
    let xml = format!(
        r#"<state name="Crowded"><state wt1="AKWP bw_saw"><MODMATRIX>{items}</MODMATRIX></state></state>"#
    );
    let read =
        bloop_plugin_wavetable::preset_file::read(xml.as_bytes()).unwrap_or_else(|e| panic!("{e}"));
    let first_and_last = |v: &[String]| {
        format!(
            "{}: {:?} … {:?}",
            v.len(),
            v.first().map(String::as_str).unwrap_or_default(),
            v.last().map(String::as_str).unwrap_or_default()
        )
    };
    expect![[r#"
        256 routes kept; left out 14: "cc32 to osc1pos: past 32 routes on one destination" … "cc29 to fltres: past the matrix's 256 routes"
    "#]].assert_eq(&format!(
        "{} routes kept; left out {}\n",
        read.routes.len(),
        first_and_last(&read.left_out)
    ));
}

/// What is not a preset is said to be, in words.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn files_that_are_not_presets() {
    let mut out = String::new();
    for (what, bytes) in [
        ("binary", &[0xFFu8, 0xFE, 0x00][..]),
        ("not XML", b"hello".as_slice()),
        ("another root", b"<preset name=\"x\"/>".as_slice()),
        (
            "a table no bank has",
            br#"<state name="x"><state wt1="No Such Table"/></state>"#.as_slice(),
        ),
        (
            "a route to nowhere",
            br#"<state name="x"><state><MODMATRIX><MODITEM srcId="lfo9" dstId="osc1pos" depth="1"/><MODITEM srcId="lfo1" dstId="nowhere" depth="1"/></MODMATRIX></state></state>"#.as_slice(),
        ),
    ] {
        let said = match bloop_plugin_wavetable::preset_file::read(bytes) {
            Ok(p) => format!("reads, ignoring {:?}", p.ignored),
            Err(e) => format!("refused: {e}"),
        };
        let _ = writeln!(out, "{what}: {said}");
    }
    expect![[r#"
        binary: refused: it is not text
        not XML: refused: it is not XML: unknown token at 1:1
        another root: refused: it is not a Wavetable preset: its root is not <state>
        a table no bank has: reads, ignoring ["the table No Such Table for OSC 1, which bloop does not have"]
        a route to nowhere: reads, ignoring ["the route lfo9 to osc1pos: bloop has no source lfo9", "the route lfo1 to nowhere: bloop has no destination nowhere"]
    "#]].assert_eq(&out);
}
