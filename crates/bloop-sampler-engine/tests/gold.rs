//! Gold tests: every scenario runs on both the Rust port and the C++ reference
//! and the two renders must match bit for bit.
//!
//! The reference is the pinned `vendor/just-a-sample` submodule. The harness in
//! `tests/cpp_harness` compiles `CustomSamplerVoice.cpp` unmodified against
//! stand-ins for the JUCE, parameter and FX plumbing around it (see the file
//! comments there for exactly what is stubbed and why it cannot affect output).
//!
//! The snapshot records what the render *is* — per-block level and voice state —
//! so it can be reviewed for physical sense; the bit-exact comparison against
//! the reference is asserted separately and fails loudly.
//!
//! These tests are native-only: they shell out to a C++ compiler.

#![cfg(not(target_arch = "wasm32"))]

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use bloop_sampler_engine::{SampleBuffer, SamplerParams, SamplerVoice};
use expect_test::{expect, Expect};

// ---------------------------------------------------------------------------
// Harness compilation
// ---------------------------------------------------------------------------

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    manifest_dir()
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

/// The two reference translation units the harness compiles verbatim.
const REFERENCE_SOURCES: [&str; 2] = ["CustomSamplerVoice.h", "CustomSamplerVoice.cpp"];

fn mtime(path: &Path) -> std::time::SystemTime {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create build dir");
    for entry in std::fs::read_dir(from).expect("read harness dir") {
        let entry = entry.expect("harness dir entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("file type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy harness file");
        }
    }
}

/// Builds (if stale) and returns the path to the reference harness binary.
///
/// The reference sources are copied next to the stand-in headers so their
/// quoted includes resolve to the stand-ins rather than to Just-a-Sample's own
/// JUCE-dependent headers. The copies are byte-for-byte.
fn build_harness() -> PathBuf {
    static BUILD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let harness_src = manifest_dir().join("tests/cpp_harness");
    let reference_dir = workspace_root().join("vendor/just-a-sample/Source/Sampler");

    assert!(
        reference_dir.join("CustomSamplerVoice.cpp").exists(),
        "vendor/just-a-sample is empty — run `git submodule update --init vendor/just-a-sample`"
    );

    let build_dir = workspace_root().join("target/jas-harness");
    let binary = build_dir.join("harness");

    let newest_input = REFERENCE_SOURCES
        .iter()
        .map(|name| mtime(&reference_dir.join(name)))
        .chain(walk(&harness_src).iter().map(|p| mtime(p)))
        .max()
        .expect("at least one input");

    if binary.exists() && mtime(&binary) > newest_input {
        return binary;
    }

    copy_tree(&harness_src, &build_dir);
    for name in REFERENCE_SOURCES {
        std::fs::copy(
            reference_dir.join(name),
            build_dir.join("Sampler").join(name),
        )
        .expect("copy reference source");
    }

    let status = Command::new("g++")
        .arg("-std=c++20")
        .arg("-O2")
        .arg("-Wall")
        .arg("-I")
        .arg(&build_dir)
        .arg(build_dir.join("Sampler/CustomSamplerVoice.cpp"))
        .arg(build_dir.join("main.cpp"))
        .arg("-o")
        .arg(&binary)
        .status()
        .expect("g++ must be installed to run the sampler gold tests");

    assert!(status.success(), "reference harness failed to compile");
    binary
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read dir") {
        let entry = entry.expect("dir entry");
        if entry.file_type().expect("file type").is_dir() {
            out.extend(walk(&entry.path()));
        } else {
            out.push(entry.path());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scenario eDSL
// ---------------------------------------------------------------------------

/// Parameter ids, shared with `main.cpp`'s `ParamId`. Order is the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum P {
    SemitoneTuning,
    CentTuning,
    PitchWheelRange,
    WideTuning,
    A4Hz,
    WaveformSemitoneTuning,
    WaveformCentTuning,
    SpeedFactor,
    OctaveSpeedFactor,
    SkipAntialiasing,
    DisableWavetableMode,
    PlayUntilEnd,
    IsLooping,
    LoopingHasStart,
    LoopingHasEnd,
    SampleStart,
    SampleEnd,
    LoopStart,
    LoopEnd,
    AttackMs,
    ReleaseMs,
    AttackShape,
    ReleaseShape,
    CrossfadeSamples,
    GainDb,
    MonoOutput,
    DisableVelocity,
    MidiStart,
    MidiEnd,
    MidiRoot,
    FollowMidiPitch,
}

/// The deterministic waveforms a scenario can load. Both sides read the same
/// rendered floats — the buffer travels in the program file — so a generator
/// change can never desynchronise them.
#[derive(Clone, Copy, Debug)]
enum Wave {
    /// A full-scale sine, `hz` at the sample's own rate.
    Sine { hz: f64, frames: usize },
    /// A ramp from -1 to +1, which makes interpolation errors obvious.
    Ramp { frames: usize },
    /// Deterministic broadband noise, which exercises the antialiasing filter.
    Noise { frames: usize },
}

impl Wave {
    fn render(self, sample_rate: u32, channels: usize) -> SampleBuffer {
        let frames = match self {
            Wave::Sine { frames, .. } | Wave::Ramp { frames } | Wave::Noise { frames } => frames,
        };
        let mut data = vec![vec![0.0f32; frames]; channels];
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        for i in 0..frames {
            let value = match self {
                Wave::Sine { hz, .. } => {
                    let phase = std::f64::consts::TAU * hz * i as f64 / f64::from(sample_rate);
                    phase.sin() as f32
                }
                Wave::Ramp { .. } => {
                    (i as f32 / frames.max(2).saturating_sub(1) as f32) * 2.0 - 1.0
                }
                Wave::Noise { .. } => {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / 8388608.0) - 1.0
                }
            };
            for (ch, channel) in data.iter_mut().enumerate() {
                // Give the right channel a distinct signal so channel mix-ups
                // cannot hide behind a symmetric buffer.
                channel[i] = if ch == 0 { value } else { -value * 0.5 };
            }
        }
        SampleBuffer::new(data, sample_rate)
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Render(usize),
    NoteOn { note: u8, velocity: f32, wheel: i32 },
    NoteOff,
    Steal,
    PitchWheel(i32),
    SetParam(P, f64),
}

/// One gold scenario.
struct Program {
    wave: Wave,
    channels: usize,
    src_rate: u32,
    app_rate: f64,
    block: usize,
    params: Vec<(P, f64)>,
    ops: Vec<Op>,
}

impl Program {
    /// A mono 48 kHz scenario whose bounds span the whole sample.
    fn new(wave: Wave) -> Self {
        Self {
            wave,
            channels: 1,
            src_rate: 48_000,
            app_rate: 48_000.0,
            block: 256,
            params: Vec::new(),
            ops: Vec::new(),
        }
    }

    fn channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    fn src_rate(mut self, rate: u32) -> Self {
        self.src_rate = rate;
        self
    }

    fn app_rate(mut self, rate: f64) -> Self {
        self.app_rate = rate;
        self
    }

    fn param(mut self, id: P, value: f64) -> Self {
        self.params.push((id, value));
        self
    }

    fn note_on(mut self, note: u8, velocity: f32) -> Self {
        self.ops.push(Op::NoteOn {
            note,
            velocity,
            wheel: 8192,
        });
        self
    }

    fn note_off(mut self) -> Self {
        self.ops.push(Op::NoteOff);
        self
    }

    fn steal(mut self) -> Self {
        self.ops.push(Op::Steal);
        self
    }

    fn pitch_wheel(mut self, value: i32) -> Self {
        self.ops.push(Op::PitchWheel(value));
        self
    }

    fn set_param(mut self, id: P, value: f64) -> Self {
        self.ops.push(Op::SetParam(id, value));
        self
    }

    /// Render `count` blocks of the program's block size.
    fn render(mut self, count: usize) -> Self {
        let block = self.block;
        for _ in 0..count {
            self.ops.push(Op::Render(block));
        }
        self
    }

    /// Render one block of an unusual size, to catch block-size dependence.
    fn render_frames(mut self, frames: usize) -> Self {
        self.ops.push(Op::Render(frames));
        self
    }

    fn total_frames(&self) -> usize {
        self.ops
            .iter()
            .map(|op| if let Op::Render(n) = op { *n } else { 0 })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Running a program
// ---------------------------------------------------------------------------

/// The Rust render, plus the per-block trace the snapshot reports.
struct RustRun {
    output: Vec<Vec<f32>>,
    trace: Vec<String>,
}

fn apply_param(params: &mut SamplerParams, id: P, value: f64) {
    let as_i32 = value as i32;
    let as_f32 = value as f32;
    let as_bool = value != 0.0;
    match id {
        P::SemitoneTuning => params.semitone_tuning = as_i32,
        P::CentTuning => params.cent_tuning = as_i32,
        P::PitchWheelRange => params.pitch_wheel_range = as_f32,
        P::WideTuning => params.wide_tuning = as_f32,
        P::A4Hz => params.a4_hz = as_f32,
        P::WaveformSemitoneTuning => params.waveform_semitone_tuning = as_i32,
        P::WaveformCentTuning => params.waveform_cent_tuning = as_i32,
        P::SpeedFactor => params.speed_factor = as_f32,
        P::OctaveSpeedFactor => params.octave_speed_factor = as_f32,
        P::SkipAntialiasing => params.skip_antialiasing = as_bool,
        P::DisableWavetableMode => params.disable_wavetable_mode = as_bool,
        P::PlayUntilEnd => params.play_until_end = as_bool,
        P::IsLooping => params.is_looping = as_bool,
        P::LoopingHasStart => params.looping_has_start = as_bool,
        P::LoopingHasEnd => params.looping_has_end = as_bool,
        P::SampleStart => params.sample_start = as_i32,
        P::SampleEnd => params.sample_end = as_i32,
        P::LoopStart => params.loop_start = as_i32,
        P::LoopEnd => params.loop_end = as_i32,
        P::AttackMs => params.attack_ms = as_f32,
        P::ReleaseMs => params.release_ms = as_f32,
        P::AttackShape => params.attack_shape = as_f32,
        P::ReleaseShape => params.release_shape = as_f32,
        P::CrossfadeSamples => params.crossfade_samples = as_i32,
        P::GainDb => params.gain_db = as_f32,
        P::MonoOutput => params.mono_output = as_bool,
        P::DisableVelocity => params.disable_velocity = as_bool,
        P::MidiStart => params.midi_start = as_i32 as u8,
        P::MidiEnd => params.midi_end = as_i32 as u8,
        P::MidiRoot => params.midi_root = as_i32 as u8,
        P::FollowMidiPitch => params.follow_midi_pitch = as_bool,
    }
}

fn run_rust(program: &Program, sample: &SampleBuffer) -> RustRun {
    let mut params = SamplerParams::spanning(sample.num_samples());
    for &(id, value) in &program.params {
        apply_param(&mut params, id, value);
    }

    let max_block = program
        .ops
        .iter()
        .filter_map(|op| {
            if let Op::Render(n) = op {
                Some(*n)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(program.block);

    let mut voice = SamplerVoice::new(
        program.app_rate,
        max_block.max(program.block),
        program.channels,
    );
    let mut output = vec![vec![0.0f32; program.total_frames()]; 2];
    let (left, right) = output.split_at_mut(1);
    let mut out: [&mut [f32]; 2] = [&mut left[0], &mut right[0]];

    let mut trace = Vec::new();
    let mut position = 0usize;
    let mut block_index = 0usize;

    for op in &program.ops {
        match *op {
            Op::Render(frames) => {
                voice.render(sample, &params, &mut out, position, frames);
                let peak = |channel: &[f32]| {
                    channel[position..position + frames]
                        .iter()
                        .fold(0.0f32, |acc, v| acc.max(v.abs()))
                };
                trace.push(format!(
                    "block {block_index:>2} @{position:<6} peak L {:>7.4} R {:>7.4}  pos {:>9.3}  {:?}{}",
                    peak(out[0]),
                    peak(out[1]),
                    voice.position(),
                    voice.state(),
                    if voice.is_antialiasing() { " [lpf]" } else { "" },
                ));
                position += frames;
                block_index += 1;
            }
            Op::NoteOn {
                note,
                velocity,
                wheel,
            } => {
                let started = voice.note_on(sample, &params, note, velocity, wheel);
                trace.push(format!(
                    "note on  {note} vel {velocity:.2}{}{}",
                    if voice.is_wavetable_mode() {
                        " [wavetable]"
                    } else {
                        ""
                    },
                    if started {
                        ""
                    } else {
                        " — out of MIDI range, ignored"
                    }
                ));
            }
            Op::NoteOff => {
                voice.note_off();
                trace.push("note off".to_string());
            }
            Op::Steal => {
                voice.steal(sample, &params);
                trace.push("stolen (tail-off rendered)".to_string());
            }
            Op::PitchWheel(value) => {
                voice.pitch_wheel_moved(sample, &params, value);
                trace.push(format!("pitch wheel {value}"));
            }
            Op::SetParam(id, value) => {
                apply_param(&mut params, id, value);
                trace.push(format!("set {id:?} = {value}"));
            }
        }
    }

    RustRun { output, trace }
}

fn run_reference(program: &Program, sample: &SampleBuffer) -> Vec<Vec<f32>> {
    let binary = build_harness();
    let dir = workspace_root().join("target/jas-harness");
    // Tests run in parallel in one process, so each render needs its own files.
    static NEXT_RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let run = NEXT_RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let program_path = dir.join(format!("program-{}-{run}.bin", std::process::id()));
    let output_path = dir.join(format!("output-{}-{run}.bin", std::process::id()));

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x4a41_5331u32.to_le_bytes());
    bytes.extend_from_slice(&program.app_rate.to_le_bytes());
    bytes.extend_from_slice(&(program.block as u32).to_le_bytes());
    bytes.extend_from_slice(&program.src_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample.num_channels() as u32).to_le_bytes());
    bytes.extend_from_slice(&(sample.num_samples() as u32).to_le_bytes());
    for ch in 0..sample.num_channels() {
        for value in sample.channel(ch) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    // The reference's bounds default to zero, so send the resolved parameter
    // set rather than only the scenario's overrides.
    let mut resolved = SamplerParams::spanning(sample.num_samples());
    for &(id, value) in &program.params {
        apply_param(&mut resolved, id, value);
    }
    let all = full_param_list(&resolved);
    bytes.extend_from_slice(&(all.len() as u32).to_le_bytes());
    for (id, value) in all {
        bytes.extend_from_slice(&(id as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    bytes.extend_from_slice(&(program.total_frames() as u32).to_le_bytes());
    bytes.extend_from_slice(&(program.ops.len() as u32).to_le_bytes());
    for op in &program.ops {
        match *op {
            Op::Render(frames) => {
                bytes.push(0);
                bytes.extend_from_slice(&(frames as u32).to_le_bytes());
            }
            Op::NoteOn {
                note,
                velocity,
                wheel,
            } => {
                bytes.push(1);
                bytes.extend_from_slice(&u32::from(note).to_le_bytes());
                bytes.extend_from_slice(&velocity.to_le_bytes());
                bytes.extend_from_slice(&wheel.to_le_bytes());
            }
            Op::NoteOff => bytes.push(2),
            Op::Steal => bytes.push(3),
            Op::PitchWheel(value) => {
                bytes.push(4);
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Op::SetParam(id, value) => {
                bytes.push(5);
                bytes.extend_from_slice(&(id as u32).to_le_bytes());
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    std::fs::File::create(&program_path)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("write program");

    let status = Command::new(&binary)
        .arg(&program_path)
        .arg(&output_path)
        .status()
        .expect("run reference harness");
    assert!(status.success(), "reference harness exited with {status}");

    let raw = std::fs::read(&output_path).expect("read reference output");
    std::fs::remove_file(&program_path).ok();
    std::fs::remove_file(&output_path).ok();
    let frames = program.total_frames();
    assert_eq!(
        raw.len(),
        frames * 2 * 4,
        "reference wrote an unexpected number of frames"
    );

    let mut output = vec![vec![0.0f32; frames]; 2];
    for (ch, channel) in output.iter_mut().enumerate() {
        for (i, value) in channel.iter_mut().enumerate() {
            let at = (ch * frames + i) * 4;
            *value = f32::from_le_bytes(raw[at..at + 4].try_into().expect("four bytes"));
        }
    }
    output
}

/// Every parameter, so the reference starts from the same state as the port.
fn full_param_list(p: &SamplerParams) -> Vec<(P, f64)> {
    vec![
        (P::SemitoneTuning, f64::from(p.semitone_tuning)),
        (P::CentTuning, f64::from(p.cent_tuning)),
        (P::PitchWheelRange, f64::from(p.pitch_wheel_range)),
        (P::WideTuning, f64::from(p.wide_tuning)),
        (P::A4Hz, f64::from(p.a4_hz)),
        (
            P::WaveformSemitoneTuning,
            f64::from(p.waveform_semitone_tuning),
        ),
        (P::WaveformCentTuning, f64::from(p.waveform_cent_tuning)),
        (P::SpeedFactor, f64::from(p.speed_factor)),
        (P::OctaveSpeedFactor, f64::from(p.octave_speed_factor)),
        (
            P::SkipAntialiasing,
            f64::from(u8::from(p.skip_antialiasing)),
        ),
        (
            P::DisableWavetableMode,
            f64::from(u8::from(p.disable_wavetable_mode)),
        ),
        (P::PlayUntilEnd, f64::from(u8::from(p.play_until_end))),
        (P::IsLooping, f64::from(u8::from(p.is_looping))),
        (P::LoopingHasStart, f64::from(u8::from(p.looping_has_start))),
        (P::LoopingHasEnd, f64::from(u8::from(p.looping_has_end))),
        (P::SampleStart, f64::from(p.sample_start)),
        (P::SampleEnd, f64::from(p.sample_end)),
        (P::LoopStart, f64::from(p.loop_start)),
        (P::LoopEnd, f64::from(p.loop_end)),
        (P::AttackMs, f64::from(p.attack_ms)),
        (P::ReleaseMs, f64::from(p.release_ms)),
        (P::AttackShape, f64::from(p.attack_shape)),
        (P::ReleaseShape, f64::from(p.release_shape)),
        (P::CrossfadeSamples, f64::from(p.crossfade_samples)),
        (P::GainDb, f64::from(p.gain_db)),
        (P::MonoOutput, f64::from(u8::from(p.mono_output))),
        (P::DisableVelocity, f64::from(u8::from(p.disable_velocity))),
        (P::MidiStart, f64::from(p.midi_start)),
        (P::MidiEnd, f64::from(p.midi_end)),
        (P::MidiRoot, f64::from(p.midi_root)),
        (P::FollowMidiPitch, f64::from(u8::from(p.follow_midi_pitch))),
    ]
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

fn check(program: Program, expect: Expect) {
    let sample = program.wave.render(program.src_rate, program.channels);
    let rust = run_rust(&program, &sample);
    let reference = run_reference(&program, &sample);

    let mismatches: Vec<String> = (0..2)
        .flat_map(|ch| {
            let (ours, theirs) = (&rust.output[ch], &reference[ch]);
            (0..ours.len())
                .filter(move |&i| ours[i].to_bits() != theirs[i].to_bits())
                .map(move |i| {
                    format!(
                        "  ch{ch} frame {i}: port {:?} vs reference {:?}",
                        ours[i], theirs[i]
                    )
                })
        })
        .take(8)
        .collect();

    assert!(
        mismatches.is_empty(),
        "port diverged from the Just-a-Sample reference:\n{}",
        mismatches.join("\n")
    );

    let mut out = String::new();
    writeln!(
        out,
        "{} ch, {} frames @{} Hz -> app {} Hz, blocks of {}",
        sample.num_channels(),
        sample.num_samples(),
        program.src_rate,
        program.app_rate,
        program.block
    )
    .expect("format");
    writeln!(out, "matches reference bit-exactly").expect("format");
    for line in &rust.trace {
        writeln!(out, "{line}").expect("format");
    }
    expect.assert_eq(&out);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

// A slice must span more than 1/WAVETABLE_CUTOFF_HZ seconds (2400 frames at
// 48 kHz) or the reference plays it as a single waveform cycle instead. The
// scenarios below use 2560- and 4800-frame samples for ordinary playback, and
// deliberately short ones for the wavetable-mode tests.

/// Playing at the root note with no tuning walks the sample at exactly one
/// frame per output frame, then stops at the end.
#[test]
fn plays_a_sample_at_its_root_note() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .note_on(69, 1.0)
        .render(12),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.002  Playing
            block  2 @512    peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
            block  4 @1024   peak L  0.9999 R  0.9999  pos  1280.005  Playing
            block  5 @1280   peak L  1.0000 R  1.0000  pos  1536.005  Playing
            block  6 @1536   peak L  1.0000 R  1.0000  pos  1792.006  Playing
            block  7 @1792   peak L  1.0000 R  1.0000  pos  2048.007  Playing
            block  8 @2048   peak L  1.0000 R  1.0000  pos  2304.008  Playing
            block  9 @2304   peak L  0.9999 R  0.9999  pos  2559.009  Stopped
            block 10 @2560   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
            block 11 @2816   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
        "#]],
    );
}

/// An octave up consumes the sample twice as fast, and engages the
/// antialiasing lowpass (speed > 1).
#[test]
fn an_octave_up_doubles_the_playback_speed() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .note_on(81, 1.0)
        .render(7),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  81 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   512.002  Playing [lpf]
            block  1 @256    peak L  1.0000 R  1.0000  pos  1024.004  Playing [lpf]
            block  2 @512    peak L  1.0000 R  1.0000  pos  1536.005  Playing [lpf]
            block  3 @768    peak L  1.0000 R  1.0000  pos  2048.007  Playing [lpf]
            block  4 @1024   peak L  1.0000 R  1.0000  pos  2560.009  Stopped [lpf]
            block  5 @1280   peak L  0.0000 R  0.0000  pos  2560.009  Stopped [lpf]
            block  6 @1536   peak L  0.0000 R  0.0000  pos  2560.009  Stopped [lpf]
        "#]],
    );
}

/// An octave down halves the speed and leaves the lowpass off.
#[test]
fn an_octave_down_halves_the_playback_speed() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .note_on(57, 1.0)
        .render(22),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  57 vel 1.00
            block  0 @0      peak L  0.9999 R  0.9999  pos   128.000  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  2 @512    peak L  1.0000 R  1.0000  pos   384.001  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos   512.002  Playing
            block  4 @1024   peak L  0.9999 R  0.9999  pos   640.002  Playing
            block  5 @1280   peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  6 @1536   peak L  1.0000 R  1.0000  pos   896.003  Playing
            block  7 @1792   peak L  1.0000 R  1.0000  pos  1024.004  Playing
            block  8 @2048   peak L  0.9999 R  0.9999  pos  1152.004  Playing
            block  9 @2304   peak L  0.9999 R  0.9999  pos  1280.005  Playing
            block 10 @2560   peak L  1.0000 R  1.0000  pos  1408.005  Playing
            block 11 @2816   peak L  1.0000 R  1.0000  pos  1536.005  Playing
            block 12 @3072   peak L  1.0000 R  1.0000  pos  1664.006  Playing
            block 13 @3328   peak L  0.9999 R  0.9999  pos  1792.006  Playing
            block 14 @3584   peak L  0.9999 R  0.9999  pos  1920.007  Playing
            block 15 @3840   peak L  1.0000 R  1.0000  pos  2048.007  Playing
            block 16 @4096   peak L  1.0000 R  1.0000  pos  2176.008  Playing
            block 17 @4352   peak L  1.0000 R  1.0000  pos  2304.008  Playing
            block 18 @4608   peak L  0.9999 R  0.9999  pos  2432.009  Playing
            block 19 @4864   peak L  0.9999 R  0.9999  pos  2559.009  Stopped
            block 20 @5120   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
            block 21 @5376   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
        "#]],
    );
}

/// Lo-fi mode drops the Lanczos interpolation and the lowpass entirely.
#[test]
fn skipping_antialiasing_takes_the_nearest_frame() {
    check(
        Program::new(Wave::Ramp { frames: 2560 })
            .param(P::SkipAntialiasing, 1.0)
            .note_on(81, 1.0)
            .render(6),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  81 vel 1.00
            block  0 @0      peak L  0.9250 R  0.9250  pos   512.002  Playing [lpf]
            block  1 @256    peak L  0.5998 R  0.5998  pos  1024.004  Playing [lpf]
            block  2 @512    peak L  0.1997 R  0.1997  pos  1536.005  Playing [lpf]
            block  3 @768    peak L  0.5991 R  0.5991  pos  2048.007  Playing [lpf]
            block  4 @1024   peak L  0.9258 R  0.9258  pos  2560.009  Stopped [lpf]
            block  5 @1280   peak L  0.0000 R  0.0000  pos  2560.009  Stopped [lpf]
        "#]],
    );
}

/// Broadband content an octave up is where the eighth-order Butterworth
/// actually earns its keep.
#[test]
fn noise_an_octave_up_runs_through_the_antialiasing_filter() {
    check(
        Program::new(Wave::Noise { frames: 2560 })
            .note_on(81, 1.0)
            .render(6),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  81 vel 1.00
            block  0 @0      peak L  1.0152 R  1.0152  pos   512.002  Playing [lpf]
            block  1 @256    peak L  1.1646 R  1.1646  pos  1024.004  Playing [lpf]
            block  2 @512    peak L  1.1733 R  1.1733  pos  1536.005  Playing [lpf]
            block  3 @768    peak L  1.1702 R  1.1702  pos  2048.007  Playing [lpf]
            block  4 @1024   peak L  1.1324 R  1.1324  pos  2560.009  Stopped [lpf]
            block  5 @1280   peak L  0.0000 R  0.0000  pos  2560.009  Stopped [lpf]
        "#]],
    );
}

/// A sub-range of the sample plays only that slice — the basis of slicing one
/// import across several pads.
#[test]
fn playback_bounds_restrict_the_voice_to_a_slice() {
    check(
        Program::new(Wave::Ramp { frames: 4800 })
            .param(P::SampleStart, 400.0)
            .param(P::SampleEnd, 3000.0)
            .note_on(69, 1.0)
            .render(12),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.8133 R  0.8133  pos   656.001  Playing
            block  1 @256    peak L  0.7266 R  0.7266  pos   912.002  Playing
            block  2 @512    peak L  0.6199 R  0.6199  pos  1168.003  Playing
            block  3 @768    peak L  0.5132 R  0.5132  pos  1424.004  Playing
            block  4 @1024   peak L  0.4065 R  0.4065  pos  1680.005  Playing
            block  5 @1280   peak L  0.2999 R  0.2999  pos  1936.005  Playing
            block  6 @1536   peak L  0.1932 R  0.1932  pos  2192.006  Playing
            block  7 @1792   peak L  0.0865 R  0.0865  pos  2448.007  Playing
            block  8 @2048   peak L  0.1265 R  0.1265  pos  2704.008  Playing
            block  9 @2304   peak L  0.2303 R  0.2303  pos  2960.009  Playing
            block 10 @2560   peak L  0.1570 R  0.1570  pos  3000.009  Stopped
            block 11 @2816   peak L  0.0000 R  0.0000  pos  3000.009  Stopped
        "#]],
    );
}

/// Looping wraps back to the start with a power-preserving crossfade, so the
/// voice never stops on its own.
#[test]
fn a_looping_slice_crossfades_back_to_its_start() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 4800,
        })
        .param(P::IsLooping, 1.0)
        .param(P::SampleStart, 480.0)
        .param(P::SampleEnd, 3000.0)
        .param(P::CrossfadeSamples, 256.0)
        .note_on(69, 1.0)
        .render(14),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.9999 R  0.9999  pos   736.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   992.002  Playing
            block  2 @512    peak L  1.0000 R  1.0000  pos  1248.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1504.004  Playing
            block  4 @1024   peak L  1.0000 R  1.0000  pos  1760.005  Playing
            block  5 @1280   peak L  1.0000 R  1.0000  pos  2016.005  Playing
            block  6 @1536   peak L  1.0000 R  1.0000  pos  2272.006  Playing
            block  7 @1792   peak L  0.9999 R  0.9999  pos  2528.007  Playing
            block  8 @2048   peak L  1.0008 R  1.0008  pos   519.008  Playing
            block  9 @2304   peak L  1.0100 R  1.0100  pos   775.009  Playing
            block 10 @2560   peak L  1.0000 R  1.0000  pos  1031.010  Playing
            block 11 @2816   peak L  0.9999 R  0.9999  pos  1287.011  Playing
            block 12 @3072   peak L  1.0000 R  1.0000  pos  1543.012  Playing
            block 13 @3328   peak L  1.0000 R  1.0000  pos  1799.013  Playing
        "#]],
    );
}

/// Releasing a looping voice that has an end portion crossfades into the tail
/// past `sampleEnd` instead of fading out.
#[test]
fn releasing_a_loop_with_an_end_crossfades_into_the_tail() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 4800,
        })
        .param(P::IsLooping, 1.0)
        .param(P::LoopingHasEnd, 1.0)
        .param(P::SampleStart, 480.0)
        .param(P::SampleEnd, 3000.0)
        .param(P::LoopEnd, 4700.0)
        .param(P::CrossfadeSamples, 256.0)
        .note_on(69, 1.0)
        .render(4)
        .note_off()
        .render(10),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.9999 R  0.9999  pos   736.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   992.002  Playing
            block  2 @512    peak L  1.0000 R  1.0000  pos  1248.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1504.004  Playing
            note off
            block  4 @1024   peak L  0.9837 R  0.9837  pos  3256.001  PlayingEnd
            block  5 @1280   peak L  1.0000 R  1.0000  pos  3512.002  PlayingEnd
            block  6 @1536   peak L  0.9999 R  0.9999  pos  3768.003  PlayingEnd
            block  7 @1792   peak L  1.0000 R  1.0000  pos  4024.004  PlayingEnd
            block  8 @2048   peak L  0.9999 R  0.9999  pos  4280.005  PlayingEnd
            block  9 @2304   peak L  1.0000 R  1.0000  pos  4536.005  PlayingEnd
            block 10 @2560   peak L  1.0000 R  1.0000  pos  4700.006  Stopped
            block 11 @2816   peak L  0.0000 R  0.0000  pos  4700.006  Stopped
            block 12 @3072   peak L  0.0000 R  0.0000  pos  4700.006  Stopped
            block 13 @3328   peak L  0.0000 R  0.0000  pos  4700.006  Stopped
        "#]],
    );
}

/// A loop with a start portion begins before `sampleStart` and only then
/// enters the loop.
#[test]
fn a_loop_with_a_start_plays_the_lead_in_first() {
    check(
        Program::new(Wave::Ramp { frames: 4800 })
            .param(P::IsLooping, 1.0)
            .param(P::LoopingHasStart, 1.0)
            .param(P::LoopStart, 200.0)
            .param(P::SampleStart, 2000.0)
            .param(P::SampleEnd, 4600.0)
            .param(P::CrossfadeSamples, 256.0)
            .note_on(69, 1.0)
            .render(14),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.8966 R  0.8966  pos   456.001  Playing
            block  1 @256    peak L  0.8100 R  0.8100  pos   712.002  Playing
            block  2 @512    peak L  0.7033 R  0.7033  pos   968.003  Playing
            block  3 @768    peak L  0.5966 R  0.5966  pos  1224.004  Playing
            block  4 @1024   peak L  0.4899 R  0.4899  pos  1480.005  Playing
            block  5 @1280   peak L  0.3832 R  0.3832  pos  1736.005  Playing
            block  6 @1536   peak L  0.2765 R  0.2765  pos  1992.006  Playing
            block  7 @1792   peak L  0.1698 R  0.1698  pos  2248.007  Playing
            block  8 @2048   peak L  0.0631 R  0.0631  pos  2504.008  Playing
            block  9 @2304   peak L  0.1498 R  0.1498  pos  2760.009  Playing
            block 10 @2560   peak L  0.2565 R  0.2565  pos  3016.010  Playing
            block 11 @2816   peak L  0.3632 R  0.3632  pos  3272.011  Playing
            block 12 @3072   peak L  0.4699 R  0.4699  pos  3528.012  Playing
            block 13 @3328   peak L  0.5766 R  0.5766  pos  3784.013  Playing
        "#]],
    );
}

/// A long attack ramps the envelope in from silence; the shape parameter bends
/// the ramp.
#[test]
fn attack_shapes_the_start_of_the_note() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 4800,
        })
        .param(P::AttackMs, 20.0)
        .param(P::AttackShape, -6.0)
        .note_on(69, 1.0)
        .render(6),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.7866 R  0.7866  pos   256.001  Playing
            block  1 @256    peak L  0.9471 R  0.9471  pos   512.002  Playing
            block  2 @512    peak L  0.9922 R  0.9922  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
            block  4 @1024   peak L  0.9999 R  0.9999  pos  1280.005  Playing
            block  5 @1280   peak L  1.0000 R  1.0000  pos  1536.005  Playing
        "#]],
    );
}

/// A linear attack (shape 0) is the reference's degenerate case, and must not
/// divide by zero.
#[test]
fn a_zero_shape_attack_is_a_straight_ramp() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 4800,
        })
        .param(P::AttackMs, 20.0)
        .param(P::AttackShape, 0.0)
        .note_on(69, 1.0)
        .render(6),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.2563 R  0.2563  pos   256.001  Playing
            block  1 @256    peak L  0.4874 R  0.4874  pos   512.002  Playing
            block  2 @512    peak L  0.7672 R  0.7672  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
            block  4 @1024   peak L  0.9999 R  0.9999  pos  1280.005  Playing
            block  5 @1280   peak L  1.0000 R  1.0000  pos  1536.005  Playing
        "#]],
    );
}

/// Release fades the voice out over the configured time and then stops it.
#[test]
fn release_fades_the_note_out_after_note_off() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 4800,
        })
        .param(P::ReleaseMs, 10.0)
        .note_on(69, 1.0)
        .render(4)
        .note_off()
        .render(6),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.002  Playing
            block  2 @512    peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
            note off
            block  4 @1024   peak L  0.8311 R  0.8311  pos  1280.005  Playing
            block  5 @1280   peak L  0.2418 R  0.2418  pos  1505.005  Stopped
            block  6 @1536   peak L  0.0000 R  0.0000  pos  1505.005  Stopped
            block  7 @1792   peak L  0.0000 R  0.0000  pos  1505.005  Stopped
            block  8 @2048   peak L  0.0000 R  0.0000  pos  1505.005  Stopped
            block  9 @2304   peak L  0.0000 R  0.0000  pos  1505.005  Stopped
        "#]],
    );
}

/// A zero-length envelope is the pad-drum default: full level immediately, and
/// a hard stop at the end of the slice.
#[test]
fn a_zero_length_envelope_plays_the_slice_flat() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .param(P::AttackMs, 0.0)
        .param(P::ReleaseMs, 0.0)
        .note_on(69, 1.0)
        .render(12),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.002  Playing
            block  2 @512    peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
            block  4 @1024   peak L  0.9999 R  0.9999  pos  1280.005  Playing
            block  5 @1280   peak L  1.0000 R  1.0000  pos  1536.005  Playing
            block  6 @1536   peak L  1.0000 R  1.0000  pos  1792.006  Playing
            block  7 @1792   peak L  1.0000 R  1.0000  pos  2048.007  Playing
            block  8 @2048   peak L  1.0000 R  1.0000  pos  2304.008  Playing
            block  9 @2304   peak L  0.9999 R  0.9999  pos  2559.009  Stopped
            block 10 @2560   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
            block 11 @2816   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
        "#]],
    );
}

/// With `playUntilEnd` set and no loop, note-off is ignored and the sample runs
/// to its end — one-shot drum behaviour.
#[test]
fn play_until_end_ignores_note_off() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .param(P::PlayUntilEnd, 1.0)
        .note_on(69, 1.0)
        .render(2)
        .note_off()
        .render(10),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.002  Playing
            note off
            block  2 @512    peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
            block  4 @1024   peak L  0.9999 R  0.9999  pos  1280.005  Playing
            block  5 @1280   peak L  1.0000 R  1.0000  pos  1536.005  Playing
            block  6 @1536   peak L  1.0000 R  1.0000  pos  1792.006  Playing
            block  7 @1792   peak L  1.0000 R  1.0000  pos  2048.007  Playing
            block  8 @2048   peak L  1.0000 R  1.0000  pos  2304.008  Playing
            block  9 @2304   peak L  0.9999 R  0.9999  pos  2559.009  Stopped
            block 10 @2560   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
            block 11 @2816   peak L  0.0000 R  0.0000  pos  2559.009  Stopped
        "#]],
    );
}

/// Velocity applies the reference's 40·log10 curve.
#[test]
fn velocity_scales_the_output() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .note_on(69, 0.5)
        .render(4),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 0.50
            block  0 @0      peak L  0.2500 R  0.2500  pos   256.001  Playing
            block  1 @256    peak L  0.2500 R  0.2500  pos   512.002  Playing
            block  2 @512    peak L  0.2500 R  0.2500  pos   768.003  Playing
            block  3 @768    peak L  0.2500 R  0.2500  pos  1024.004  Playing
        "#]],
    );
}

/// Zero velocity lands on the reference's -inf decibel case and is silent.
#[test]
fn zero_velocity_is_silent() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .note_on(69, 0.0)
        .render(4),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 0.00
            block  0 @0      peak L  0.0000 R  0.0000  pos   256.001  Playing
            block  1 @256    peak L  0.0000 R  0.0000  pos   512.002  Playing
            block  2 @512    peak L  0.0000 R  0.0000  pos   768.003  Playing
            block  3 @768    peak L  0.0000 R  0.0000  pos  1024.004  Playing
        "#]],
    );
}

/// Disabling velocity pins the voice at full scale regardless of how hard the
/// pad was hit.
#[test]
fn disabling_velocity_ignores_how_hard_the_pad_was_hit() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .param(P::DisableVelocity, 1.0)
        .note_on(69, 0.1)
        .render(4),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 0.10
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.002  Playing
            block  2 @512    peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
        "#]],
    );
}

/// Sample gain is applied on top of velocity, in decibels.
#[test]
fn sample_gain_is_applied_in_decibels() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .param(P::GainDb, -6.0)
        .note_on(69, 1.0)
        .render(4),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.5012 R  0.5012  pos   256.001  Playing
            block  1 @256    peak L  0.5012 R  0.5012  pos   512.002  Playing
            block  2 @512    peak L  0.5012 R  0.5012  pos   768.003  Playing
            block  3 @768    peak L  0.5012 R  0.5012  pos  1024.004  Playing
        "#]],
    );
}

/// A stereo sample keeps its channels apart; the harness gives the right
/// channel a different signal so a swap would show.
#[test]
fn a_stereo_sample_keeps_its_channels() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .channels(2)
        .note_on(69, 1.0)
        .render(6),
        expect![[r#"
            2 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  1.0000 R  0.5000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  0.5000  pos   512.002  Playing
            block  2 @512    peak L  0.9999 R  0.5000  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  0.5000  pos  1024.004  Playing
            block  4 @1024   peak L  0.9999 R  0.5000  pos  1280.005  Playing
            block  5 @1280   peak L  1.0000 R  0.5000  pos  1536.005  Playing
        "#]],
    );
}

/// Mono output folds a stereo sample down to an identical pair of channels.
#[test]
fn mono_output_folds_a_stereo_sample_down() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .channels(2)
        .param(P::MonoOutput, 1.0)
        .note_on(69, 1.0)
        .render(6),
        expect![[r#"
            2 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.2500 R  0.2500  pos   256.001  Playing
            block  1 @256    peak L  0.2500 R  0.2500  pos   512.002  Playing
            block  2 @512    peak L  0.2500 R  0.2500  pos   768.003  Playing
            block  3 @768    peak L  0.2500 R  0.2500  pos  1024.004  Playing
            block  4 @1024   peak L  0.2500 R  0.2500  pos  1280.005  Playing
            block  5 @1280   peak L  0.2500 R  0.2500  pos  1536.005  Playing
        "#]],
    );
}

/// A sample recorded at a different rate than the host is resampled.
#[test]
fn a_sample_at_a_different_rate_is_resampled() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .src_rate(44_100)
        .app_rate(48_000.0)
        .note_on(69, 1.0)
        .render(10),
        expect![[r#"
            1 ch, 2560 frames @44100 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.9998 R  0.9998  pos   235.201  Playing
            block  1 @256    peak L  0.9994 R  0.9994  pos   470.402  Playing
            block  2 @512    peak L  0.9998 R  0.9998  pos   705.603  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos   940.803  Playing
            block  4 @1024   peak L  0.9996 R  0.9996  pos  1176.004  Playing
            block  5 @1280   peak L  1.0000 R  1.0000  pos  1411.205  Playing
            block  6 @1536   peak L  0.9998 R  0.9998  pos  1646.406  Playing
            block  7 @1792   peak L  0.9998 R  0.9998  pos  1881.607  Playing
            block  8 @2048   peak L  0.9997 R  0.9997  pos  2116.808  Playing
            block  9 @2304   peak L  0.9998 R  0.9998  pos  2352.008  Playing
        "#]],
    );
}

/// Semitone and cent tuning stack on top of the MIDI note.
#[test]
fn semitone_and_cent_tuning_stack_with_the_note() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .param(P::SemitoneTuning, 7.0)
        .param(P::CentTuning, -25.0)
        .note_on(69, 1.0)
        .render(8),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.9999 R  0.9999  pos   378.069  Playing [lpf]
            block  1 @256    peak L  0.9996 R  0.9996  pos   756.138  Playing [lpf]
            block  2 @512    peak L  0.9995 R  0.9995  pos  1134.207  Playing [lpf]
            block  3 @768    peak L  0.9999 R  0.9999  pos  1512.275  Playing [lpf]
            block  4 @1024   peak L  0.9998 R  0.9998  pos  1890.344  Playing [lpf]
            block  5 @1280   peak L  0.9998 R  0.9998  pos  2268.413  Playing [lpf]
            block  6 @1536   peak L  0.9995 R  0.9995  pos  2559.349  Stopped [lpf]
            block  7 @1792   peak L  0.0000 R  0.0000  pos  2559.349  Stopped [lpf]
        "#]],
    );
}

/// With MIDI tracking off, every pad plays the slice at the same pitch.
#[test]
fn not_following_midi_pitch_plays_every_note_the_same() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .param(P::FollowMidiPitch, 0.0)
        .note_on(45, 1.0)
        .render(4),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  45 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.002  Playing
            block  2 @512    peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.004  Playing
        "#]],
    );
}

/// The pitch wheel bends by the configured range, and moving it mid-note takes
/// effect on the next block.
#[test]
fn the_pitch_wheel_bends_within_its_range() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 4800,
        })
        .param(P::PitchWheelRange, 12.0)
        .note_on(69, 1.0)
        .render(4)
        .pitch_wheel(16383)
        .render(4),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.011  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.022  Playing
            block  2 @512    peak L  0.9999 R  0.9999  pos   768.033  Playing
            block  3 @768    peak L  1.0000 R  1.0000  pos  1024.043  Playing
            pitch wheel 16383
            block  4 @1024   peak L  1.0000 R  1.0000  pos  1536.043  Playing [lpf]
            block  5 @1280   peak L  0.9999 R  0.9999  pos  2048.043  Playing [lpf]
            block  6 @1536   peak L  1.0000 R  1.0000  pos  2560.043  Playing [lpf]
            block  7 @1792   peak L  1.0000 R  1.0000  pos  3072.043  Playing [lpf]
        "#]],
    );
}

/// A note outside the sound's MIDI range never starts.
#[test]
fn a_note_outside_the_midi_range_is_ignored() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .param(P::MidiStart, 60.0)
        .param(P::MidiEnd, 72.0)
        .note_on(48, 1.0)
        .render(4),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  48 vel 1.00 — out of MIDI range, ignored
            block  0 @0      peak L  0.0000 R  0.0000  pos     0.000  Stopped
            block  1 @256    peak L  0.0000 R  0.0000  pos     0.000  Stopped
            block  2 @512    peak L  0.0000 R  0.0000  pos     0.000  Stopped
            block  3 @768    peak L  0.0000 R  0.0000  pos     0.000  Stopped
        "#]],
    );
}

/// Very short bounds turn the slice into a single wavetable cycle, pitched from
/// the A4 reference rather than the sample rate, and always looping.
#[test]
fn very_short_bounds_switch_to_wavetable_mode() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 480,
        })
        .param(P::SampleEnd, 109.0)
        .param(P::CrossfadeSamples, 8.0)
        .note_on(69, 1.0)
        .render(8),
        expect![[r#"
            1 ch, 480 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00 [wavetable]
            block  0 @0      peak L  0.9998 R  0.9998  pos    17.680  Playing
            block  1 @256    peak L  0.9999 R  0.9999  pos    35.361  Playing
            block  2 @512    peak L  0.9997 R  0.9997  pos    53.041  Playing
            block  3 @768    peak L  0.9999 R  0.9999  pos    70.722  Playing
            block  4 @1024   peak L  0.9999 R  0.9999  pos    88.402  Playing
            block  5 @1280   peak L  0.9998 R  0.9998  pos     4.083  Playing
            block  6 @1536   peak L  0.9999 R  0.9999  pos    21.763  Playing
            block  7 @1792   peak L  0.9998 R  0.9998  pos    39.443  Playing
        "#]],
    );
}

/// Wavetable mode has its own tuning parameters, separate from the sample's.
#[test]
fn wavetable_mode_uses_its_own_tuning() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 480,
        })
        .param(P::SampleEnd, 109.0)
        .param(P::CrossfadeSamples, 8.0)
        .param(P::SemitoneTuning, 12.0)
        .param(P::WaveformSemitoneTuning, -12.0)
        .note_on(69, 1.0)
        .render(8),
        expect![[r#"
            1 ch, 480 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00 [wavetable]
            block  0 @0      peak L  0.9998 R  0.9998  pos    59.840  Playing
            block  1 @256    peak L  0.9999 R  0.9999  pos    17.680  Playing
            block  2 @512    peak L  0.9999 R  0.9999  pos    77.521  Playing
            block  3 @768    peak L  0.9999 R  0.9999  pos    35.361  Playing
            block  4 @1024   peak L  0.9998 R  0.9998  pos    95.201  Playing
            block  5 @1280   peak L  0.9998 R  0.9998  pos    53.041  Playing
            block  6 @1536   peak L  0.9999 R  0.9999  pos    10.881  Playing
            block  7 @1792   peak L  0.9999 R  0.9999  pos    70.722  Playing
        "#]],
    );
}

/// Wavetable mode can be turned off, which makes a short slice play as an
/// ordinary one-shot that ends.
#[test]
fn wavetable_mode_can_be_disabled() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 480,
        })
        .param(P::SampleEnd, 109.0)
        .param(P::DisableWavetableMode, 1.0)
        .note_on(69, 1.0)
        .render(4),
        expect![[r#"
            1 ch, 480 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.6581 R  0.6581  pos   109.000  Stopped
            block  1 @256    peak L  0.0000 R  0.0000  pos   109.000  Stopped
            block  2 @512    peak L  0.0000 R  0.0000  pos   109.000  Stopped
            block  3 @768    peak L  0.0000 R  0.0000  pos   109.000  Stopped
        "#]],
    );
}

/// Stealing a voice renders a 50-frame fade — visible in the trace as the
/// position jumping 50 frames past where the voice stopped. The reference then
/// holds that fade until a note starts again (a stopped voice returns from
/// `renderNextBlock` before reaching the tail-off mix), so it lands at the
/// start of the retrigger rather than in the silent block between.
#[test]
fn stealing_a_voice_fades_instead_of_clicking() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 4800,
        })
        .note_on(69, 1.0)
        .render(2)
        .steal()
        .render(1)
        .note_on(69, 1.0)
        .render(4),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  1 @256    peak L  1.0000 R  1.0000  pos   512.002  Playing
            stolen (tail-off rendered)
            block  2 @512    peak L  0.0000 R  0.0000  pos   562.002  Stopped
            note on  69 vel 1.00
            block  3 @768    peak L  1.0000 R  1.0000  pos   256.001  Playing
            block  4 @1024   peak L  1.0000 R  1.0000  pos   512.002  Playing
            block  5 @1280   peak L  0.9999 R  0.9999  pos   768.003  Playing
            block  6 @1536   peak L  1.0000 R  1.0000  pos  1024.004  Playing
        "#]],
    );
}

/// Changing bounds between notes takes effect on the next note, not the
/// sounding one — the pads share one buffer and edit their own slice.
#[test]
fn bounds_changes_apply_from_the_next_note() {
    check(
        Program::new(Wave::Ramp { frames: 4800 })
            .param(P::SampleEnd, 2400.0)
            .note_on(69, 1.0)
            .render(2)
            .set_param(P::SampleStart, 2400.0)
            .set_param(P::SampleEnd, 4799.0)
            .render(2)
            .note_on(69, 1.0)
            .render(4),
        expect![[r#"
            1 ch, 4800 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.9800 R  0.9800  pos   256.001  Playing
            block  1 @256    peak L  0.8933 R  0.8933  pos   512.002  Playing
            set SampleStart = 2400
            set SampleEnd = 4799
            block  2 @512    peak L  0.7866 R  0.7866  pos   768.003  Playing
            block  3 @768    peak L  0.6799 R  0.6799  pos  1024.004  Playing
            note on  69 vel 1.00
            block  4 @1024   peak L  0.1065 R  0.1065  pos  2656.001  Playing
            block  5 @1280   peak L  0.2132 R  0.2132  pos  2912.002  Playing
            block  6 @1536   peak L  0.3199 R  0.3199  pos  3168.003  Playing
            block  7 @1792   peak L  0.4265 R  0.4265  pos  3424.004  Playing
        "#]],
    );
}

/// Odd block sizes must not change the render — the position advances per
/// frame, not per block.
#[test]
fn output_does_not_depend_on_the_block_size() {
    check(
        Program::new(Wave::Sine {
            hz: 440.0,
            frames: 2560,
        })
        .note_on(69, 1.0)
        .render_frames(7)
        .render_frames(256)
        .render_frames(1)
        .render_frames(500),
        expect![[r#"
            1 ch, 2560 frames @48000 Hz -> app 48000 Hz, blocks of 256
            matches reference bit-exactly
            note on  69 vel 1.00
            block  0 @0      peak L  0.0423 R  0.0423  pos     7.000  Playing
            block  1 @7      peak L  1.0000 R  1.0000  pos   263.001  Playing
            block  2 @263    peak L  0.5314 R  0.5314  pos   264.001  Playing
            block  3 @264    peak L  1.0000 R  1.0000  pos   764.003  Playing
        "#]],
    );
}
