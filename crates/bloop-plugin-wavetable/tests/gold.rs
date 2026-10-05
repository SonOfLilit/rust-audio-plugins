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

use bloop_core::Note;
use bloop_events::{EventBuffer, NoteEvent};
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
    let tables = bank::tables_for(plugin.table_indices(), rate);
    plugin.set_tables(Some(tables));
}

fn retimed(event: NoteEvent, at: u32) -> NoteEvent {
    match event {
        NoteEvent::NoteOn { channel, note, velocity, .. } => NoteEvent::NoteOn { time: at, channel, note, velocity },
        NoteEvent::NoteOff { channel, note, .. } => NoteEvent::NoteOff { time: at, channel, note },
        NoteEvent::Aftertouch { channel, note, pressure, .. } => NoteEvent::Aftertouch { time: at, channel, note, pressure },
        NoteEvent::PitchBend { channel, value, .. } => NoteEvent::PitchBend { time: at, channel, value },
        NoteEvent::CC { channel, cc, value, .. } => NoteEvent::CC { time: at, channel, cc, value },
        NoteEvent::Choke { channel, note, .. } => NoteEvent::Choke { time: at, channel, note },
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
                let at = params::id::MOD1_SRC + slots * 3;
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
            Step::Tables(a, b) => {
                plugin.set_param(params::id::WT1, f32::from(table_index(a)));
                plugin.set_param(params::id::WT2, f32::from(table_index(b)));
                hand_tables(&mut plugin, rate);
            }
            Step::Tempo(bpm) => tempo = bpm,
            Step::Midi(time, midi) => pending.push(match midi {
                Midi::On(note, velocity) => NoteEvent::NoteOn {
                    time,
                    channel: 0,
                    note: Note::new(note),
                    velocity: f32::from(velocity) / 127.0,
                },
                Midi::Off(note) => NoteEvent::NoteOff {
                    time,
                    channel: 0,
                    note: Note::new(note),
                },
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
            }),
            Step::Play {
                label,
                seconds,
                block,
            } => {
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
    let peak = samples.iter().fold(0.0f64, |m, s| m.max(f64::from(s.abs())));
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
        let nan = signal.left.iter().chain(&signal.right).any(|s| !s.is_finite());
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
    expect.assert_eq(&summary(&rust));
}

#[cfg(feature = "gold")]
fn compare(program: &Program, rust: &[(&'static str, Signal)], cpp: &[Signal]) {
    assert_eq!(rust.len(), cpp.len(), "passage count");
    for ((label, ours), theirs) in rust.iter().zip(cpp) {
        for (channel, a, b) in [("L", &ours.left, &theirs.left), ("R", &ours.right, &theirs.right)] {
            assert_eq!(a.len(), b.len(), "{label} {channel}: length");
            if let Some(i) = (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits()) {
                let differing = (0..a.len()).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
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
        let own = root().join("../../../vendor").join(name);
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
        assert!(out.status.success(), "git rev-parse failed in {}", path.display());
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
            .map(|e| if e.path().is_dir() { newest(&e.path()) } else { mtime(&e.path()) })
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
        for entry in std::fs::read_dir(from).expect("read Wavetable's sources").chain(
            std::fs::read_dir(from.join("FX")).expect("read Wavetable's effects"),
        ) {
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
        copy_sources(&wavetable.join("plugin").join(["Sou", "rce"].concat()), &sources);
        let harness_changed = newest(&harness);
        let config_changed = mtime(&harness.join("config.h"));

        let mut units: Vec<(PathBuf, bool)> =
            JUCE_UNITS.iter().map(|u| (harness.join(u), false)).collect();
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
            let child = cmd.spawn().expect("run g++ (the gold tests need it installed)");
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
                assert!(status.success(), "compiling {} for the gold harness failed", source.display());
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
        let tables = plugin.tables().expect("tables were just handed over").clone();
        let indices = plugin.table_indices();
        for (osc, index) in indices.into_iter().enumerate() {
            let table = tables
                .oscillator(osc, f32::from(index))
                .expect("the set holds the oscillator's table");
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
                    for &(src, uid, depth) in preset.mods.iter().take(params::MOD_SLOTS) {
                        let src = source_slot(src) as usize - 1;
                        let _ = writeln!(input, "mod {src} {uid} {}", bits(depth));
                    }
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
                    while start < n {
                        let len = block.min(n - start);
                        for &(t, m) in &midi {
                            let t = t as usize;
                            if (start..start + len).contains(&t) {
                                let (b0, b1, b2) = match m {
                                    Midi::On(note, velocity) => (0x90, note, velocity),
                                    Midi::Off(note) => (0x80, note, 0),
                                    Midi::Bend(v) => (0xE0, (v & 0x7F) as u8, (v >> 7) as u8),
                                    Midi::Cc(cc, v) => (0xB0, cc, v),
                                };
                                let _ = writeln!(input, "midi {} {b0} {b1} {b2}", t - start);
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

    /// What the original's `loadWaveTable` builds from catalogue entry
    /// `index` at `rate`: per frame, its band-limited copies.
    pub fn load(index: usize, rate: f64) -> Vec<Vec<Vec<f32>>> {
        let entry = &bank::catalogue()[index];
        let bytes = entry.bytes().expect("this build carries the table");
        let format = match entry.format {
            bank::Format::Flac => "flac",
            bank::Format::Wav => "wav",
        };
        let mut input = Vec::new();
        let _ = writeln!(input, "wtload {format} {} {rate:?} {}", entry.frame, bytes.len());
        input.extend_from_slice(bytes);
        input.push(b'\n');
        let output = harness(input);
        let mut words = output
            .chunks_exact(4)
            .map(|b| [b[0], b[1], b[2], b[3]]);
        let mut next_u32 = || u32::from_le_bytes(words.next().expect("more output")) as usize;
        let frames = next_u32();
        let mut out = Vec::new();
        for _ in 0..frames {
            let copies = next_u32();
            let len = next_u32();
            let mut frame = Vec::new();
            for _ in 0..copies {
                frame.push((0..len).map(|_| f32::from_bits(next_u32() as u32)).collect());
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
fn table_builds_match_the_original() {
    let mut report = String::new();
    for (name, rate) in [
        ("AKWP bw_saw", 48000.0),
        ("AKWP 0001", 44100.0),
        ("NK - INERT", 48000.0),
        ("Untitled project - 2020-May-16_6", 48000.0),
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
    "#]].assert_eq(&report);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn init_plays_a_note() {
    check(&Program::new().note(60, 100, 0.5, 0.5), &expect![[r#"
        held: L peak -19.5 rms -25.9 | R peak -19.5 rms -25.9 #a468c83be7ae57ad
        released: L peak -21.7 rms -42.5 | R peak -21.7 rms -42.5 #e5fa50bd6ef941a5
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            chord: L peak -9.9 rms -18.8 | R peak -9.9 rms -18.8 #0c76a316ef5fff4d
            bent up: L peak -12.0 rms -19.4 | R peak -12.0 rms -19.4 #3a4705aaaaaf3769
            bent down: L peak -11.6 rms -17.6 | R peak -11.6 rms -17.6 #aacc22acfd479c35
            released: L peak -13.0 rms -33.3 | R peak -13.0 rms -33.3 #e22ce05656c64625
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            held: L peak -17.3 rms -23.1 | R peak -17.3 rms -23.1 #6b477b39339a1c95
            released: L peak -18.2 rms -37.1 | R peak -18.2 rms -37.1 #db0a6e102430639d
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            held: L peak -10.7 rms -20.4 | R peak -5.1 rms -14.6 #9274b38b0da137f9
            released: L peak -18.1 rms -35.6 | R peak -11.8 rms -29.4 #3e088d91c036475f
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
    check(&program, &expect![[r#"
        held: L peak -11.5 rms -22.6 | R peak -11.5 rms -22.6 #ccf40cc0678ae9dd
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
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            held: L peak -7.4 rms -10.9 | R peak -7.4 rms -10.9 #0b1e6bda287e09a5
            released: L peak -10.0 rms -20.7 | R peak -10.0 rms -20.7 #d0bb9c1de212e861
            held: L peak -4.6 rms -10.2 | R peak -4.6 rms -10.2 #fc49040e4501af31
            released: L peak -10.0 rms -20.4 | R peak -10.0 rms -20.4 #9babf18f3d64b97d
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            first: L peak -17.3 rms -24.0 | R peak -17.3 rms -24.0 #353721d2aac700f9
            glide up: L peak -17.3 rms -23.4 | R peak -17.3 rms -23.4 #2506e1e80b47a975
            back down: L peak -17.3 rms -23.4 | R peak -17.3 rms -23.4 #ec355bb79b4b1c85
            released: L peak -20.7 rms -38.1 | R peak -20.7 rms -38.1 #3573b49be0b77fd9
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            three held: L peak -25.5 rms -31.6 | R peak -25.5 rms -31.6 #8edc1ee364674815
            middle let go: L peak -27.3 rms -31.3 | R peak -27.3 rms -31.3 #09dace2dbe956785
            back to the first: L peak -17.9 rms -23.5 | R peak -17.9 rms -23.5 #4a5f9d1ba8fa2a6d
            glissando up: L peak -17.7 rms -23.6 | R peak -17.7 rms -23.6 #d5dc3f4a4103d8ad
            all released: L peak -18.9 rms -36.1 | R peak -18.9 rms -36.1 #608549ce16bf7651
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            held: L peak -45.4 rms -57.6 | R peak -36.0 rms -45.2 #a7e81da37695436b
            released: L peak -45.3 rms -62.3 | R peak -45.3 rms -62.3 #aab2531d538b3fe3
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
            held: L peak -15.5 rms -27.4 | R peak -15.5 rms -27.3 #f344982297c478da
            released: L peak -65.7 rms -85.0 | R peak -68.0 rms -87.4 #d9e541fc9e65c8d4
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
    check(&program, &expect![[r#"
        held: L peak -22.2 rms -29.0 | R peak -24.7 rms -32.9 #ec5c930ed5c3dca5
        released: L peak -23.7 rms -41.8 | R peak -31.1 rms -49.9 #77f4f921c91134fe
        held: L peak -12.5 rms -19.0 | R peak -12.5 rms -19.0 #8fdb0eb688a4e921
        released: L peak -14.7 rms -33.1 | R peak -14.7 rms -33.1 #51f23fc407b4d4e5
        held: L peak -19.5 rms -26.2 | R peak -19.5 rms -26.2 #578efa885fda68b5
        released: L peak -22.0 rms -40.6 | R peak -22.0 rms -40.6 #84edf76ed544d86d
        held: L peak -18.0 rms -24.0 | R peak -18.0 rms -24.0 #c1ad1641baeb5705
        released: L peak -19.7 rms -34.4 | R peak -19.7 rms -34.4 #48a5360dcd3435e1
        held: L peak -10.6 rms -15.1 | R peak -10.6 rms -15.1 #27f01503712f7275
        released: L peak -12.9 rms -21.7 | R peak -12.9 rms -21.7 #94dd49c11d820d25
        held: L peak -4.3 rms -13.6 | R peak -4.3 rms -13.6 #06effdb0d569fbf4
        released: L peak -4.8 rms -18.9 | R peak -4.8 rms -18.9 #02de42d9b66f5114
        held: L peak -1.4 rms -11.0 | R peak -1.4 rms -11.0 #799760475d5cc56c
        released: L peak -1.4 rms -12.2 | R peak -1.4 rms -12.2 #f587b0df9f342712
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn more_notes_than_voices_steal() {
    let mut program = Program::new().set("voices", 4.0).set("release", 1.0);
    for (i, note) in [48, 52, 55, 59, 62, 65].into_iter().enumerate() {
        program = program.on(i as u32 * 50, note, 100);
    }
    check(&program.play("six notes on four voices", 0.4), &expect![[r#"
        six notes on four voices: L peak -8.7 rms -19.8 | R peak -8.7 rms -19.8 #4ba9b01c06cdc26d
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn other_rates_and_odd_blocks() {
    for (rate, block, expect) in [
        (44100.0, 37, expect![[r#"
            held: L peak -9.6 rms -15.6 | R peak -9.6 rms -15.6 #5afd8b390bd62199
            released: L peak -11.5 rms -30.0 | R peak -11.5 rms -30.0 #6f21d370e01c3d4d
        "#]]),
        (96000.0, 1000, expect![[r#"
            held: L peak -9.6 rms -15.5 | R peak -9.6 rms -15.5 #44064bd68a304945
            released: L peak -11.7 rms -30.0 | R peak -11.7 rms -30.0 #77a38fd9130533f9
        "#]]),
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

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn every_factory_preset() {
    let mut report = String::new();
    for preset in presets::PRESETS {
        let program = Program::new()
            .preset(preset.name)
            .on(0, 48, 100)
            .on(0, 55, 100)
            .on(0, 60, 100)
            .play("chord", 1.0)
            .off(0, 48)
            .off(0, 55)
            .off(0, 60)
            .play("released", 1.0);
        let rust = run_rust(&program);
        #[cfg(feature = "gold")]
        compare(&program, &rust, &cpp::run(&program));
        let _ = writeln!(report, "{}:\n{}", preset.name, summary(&rust));
    }
    expect![[r#"
        DT Siren 1:
        chord: L peak -16.1 rms -28.3 | R peak -16.1 rms -28.3 #9e66f81d46f5c8d1
        released: L peak -18.9 rms -33.2 | R peak -18.9 rms -33.2 #8af5eb12e54394d5

        DT Siren Complex 2:
        chord: L peak -9.5 rms -16.6 | R peak -9.4 rms -16.5 #402a95df24ae7c55
        released: L peak -11.3 rms -21.9 | R peak -11.9 rms -22.9 #c7f7e22797e66384

        DT Siren complex:
        chord: L peak -7.9 rms -16.6 | R peak -7.7 rms -16.5 #954e49beb22d6f78
        released: L peak -10.1 rms -20.8 | R peak -9.8 rms -21.6 #b677c5f575403d20

        DT Sweeper 6:
        chord: L peak -8.5 rms -19.3 | R peak -5.7 rms -16.8 #4ada1faf77aa0987
        released: L peak -10.8 rms -21.9 | R peak -11.2 rms -21.0 #d91e058d85d82a9d

        VOX Dist Vocal I:
        chord: L peak -5.7 rms -17.5 | R peak -5.4 rms -17.6 #0ee02a7a6ba1822d
        released: L peak -13.0 rms -26.3 | R peak -10.9 rms -26.4 #97d857124dd810f1

        Guitar Pad:
        chord: L peak -17.9 rms -29.2 | R peak -18.8 rms -29.3 #bb162510f4693a49
        released: L peak -18.7 rms -29.4 | R peak -17.7 rms -29.6 #f2dc10c2f2932f8c

        Noise Riserish:
        chord: L peak -25.5 rms -39.0 | R peak -24.4 rms -38.9 #10d3e9c6a54f7ef6
        released: L peak -25.1 rms -37.2 | R peak -23.6 rms -37.0 #9cc4905161a8248e

        Whip:
        chord: L peak 1.0 rms -17.2 | R peak 1.4 rms -17.3 #30aace4e8603afb4
        released: L peak -30.3 rms -53.6 | R peak -29.6 rms -53.5 #e0440cc7206585ed

    "#]].assert_eq(&report);
}
