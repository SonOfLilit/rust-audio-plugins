//! Wavetable as a host sees it: what it says about itself, how its
//! tables, presets and voices behave, and what `process` costs. What it
//! sounds like against the original is tests/it/gold.rs.

#![expect(
    clippy::expect_used,
    reason = "a test that cannot find the table or knob it is about has nothing to say"
)]
#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;
use std::sync::Arc;

use bloop_core::Note;
use bloop_events::{EventBuffer, NoteEvent};
use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_wavetable::{
    bank, param_info, params, presets, WavetablePlugin, PARAM_PRESET, VOICE_SLOTS,
};
use expect_test::{expect, Expect};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

/// Counts the allocations and frees made on the thread that is asking, so tests
/// running beside each other do not count each other's.
struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // Safety: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // A free counts too: dropping the last handle on a table here is
        // what the audio thread must never do.
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // Safety: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // Safety: forwarded unchanged to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn allocations_during(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

const RATE: f64 = 48000.0;

fn on(time: u32, note: u8) -> NoteEvent {
    NoteEvent::NoteOn {
        time,
        channel: 0,
        note: Note::new(note),
        velocity: 0.8,
    }
}

fn off(time: u32, note: u8) -> NoteEvent {
    NoteEvent::NoteOff {
        time,
        channel: 0,
        note: Note::new(note),
    }
}

/// `samples` of output in 512-sample blocks, `events` landing in the first.
fn render(plugin: &mut WavetablePlugin, events: &[NoteEvent], samples: usize) -> Vec<f32> {
    let mut left = vec![0.0; samples];
    let mut right = vec![0.0; samples];
    render_into(plugin, events, &mut left, &mut right);
    left
}

/// As `render`, into buffers the caller allocated: allocates nothing.
fn render_into(
    plugin: &mut WavetablePlugin,
    events: &[NoteEvent],
    left: &mut [f32],
    right: &mut [f32],
) {
    let rate = plugin.sample_rate();
    let mut first = true;
    for (l, r) in left.chunks_mut(512).zip(right.chunks_mut(512)) {
        let mut buffer = EventBuffer::new();
        if first {
            for e in events {
                buffer.push(*e);
            }
            first = false;
        }
        let mut outputs: [&mut [f32]; 2] = [l, r];
        let mut context = ProcessContext {
            outputs: &mut outputs,
            input_events: &buffer,
            sample_rate: rate as f32,
            tempo: 120.0,
            is_playing: true,
        };
        plugin.process(&mut context);
    }
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0, |m, s| m.max(s.abs()))
}

fn db(x: f32) -> String {
    if x <= 0.0 {
        "-inf".to_string()
    } else {
        format!("{:.1} dB", 20.0 * x.log10())
    }
}

fn set(plugin: &mut WavetablePlugin, uid: &str, value: f32) {
    plugin.set_param(params::index_of(uid).expect("a Wavetable parameter"), value);
}

fn with_tables(mut plugin: WavetablePlugin) -> WavetablePlugin {
    let tables = bank::tables_for(
        plugin.table_indices(),
        plugin.sample_rate(),
        bank::Quality::Render,
    );
    plugin.set_tables(Some(tables));
    plugin
}

/// Upward zero crossings a second, after the first tenth of a second.
fn frequency(samples: &[f32], rate: f64) -> f64 {
    let skip = (rate / 10.0) as usize;
    let tail = &samples[skip..];
    let crossings = tail
        .windows(2)
        .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
        .count();
    crossings as f64 * rate / tail.len() as f64
}

fn check(text: &str, expect: &Expect) {
    expect.assert_eq(text);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn what_it_says_about_itself() {
    let plugin = WavetablePlugin::new();
    let info = plugin.info();
    let mut text = format!(
        "{} by {}, {:?}, {} in, {} out, {} parameters\n",
        info.name,
        info.vendor,
        info.category,
        info.num_audio_inputs,
        info.num_audio_outputs,
        plugin.param_count()
    );
    for uid in [
        "osc1pos",
        "osc1level",
        "osc1unison",
        "fltfreq",
        "flttype",
        "lfo1rate",
        "lfo1fade",
        "env1sustain",
        "voices",
        "dlTime",
        "wt1",
        "wt2",
        "mod1src",
        "mod1dst",
        "mod1depth",
    ] {
        let index = params::index_of(uid).expect("a listed parameter");
        let p = param_info(index);
        let _ = writeln!(
            text,
            "{index:>3} {uid:<12} {:<16} {} to {}, default {}",
            p.name, p.min, p.max, p.default
        );
    }
    let p = param_info(PARAM_PRESET);
    let _ = writeln!(
        text,
        "{PARAM_PRESET:>3} preset       {:<16} {} to {}, default {} ({} factory presets, then Init)",
        p.name,
        p.min,
        p.max,
        p.default,
        presets::PRESETS.len()
    );
    let _ = writeln!(
        text,
        "past the end: {:?}",
        param_info(PARAM_PRESET + 1).name
    );
    check(
        &text,
        &expect![[r#"
            Wavetable by SocaLabs, Instrument, 0 in, 2 out, 1768 parameters
              6 osc1pos      OSC1 Pos         0 to 100, default 0
              5 osc1level    OSC1 Level       -100 to 0, default 0
              2 osc1unison   OSC1 Unison      1 to 8, default 1
             39 fltfreq      FLT Freq         0 to 135.07623, default 64
             36 flttype      FLT Type         0 to 7, default 0
             72 lfo1rate     LFO1 Rate        0 to 50, default 10
             77 lfo1fade     LFO1 Fade        -60 to 60, default 0
             54 env1sustain  ENV1 Sustain     0 to 100, default 80
            148 voices       Voices           2 to 40, default 40
            198 dlTime       Delay Time       0 to 120, default 1
            229 wt1          OSC1 Wavetable   0 to 4095, default 26
            230 wt2          OSC2 Wavetable   0 to 4095, default 31
            231 mod1src      Mod 1 Source     0 to 255, default 0
            232 mod1dst      Mod 1 Destination 0 to 255, default 0
            233 mod1depth    Mod 1 Depth      -1 to 1, default 0
            1767 preset       Preset           0 to 217, default 217 (217 factory presets, then Init)
            past the end: ""
        "#]],
    );
}

/// Every bank with its table count, how each is stored, and how much of
/// it the web build carries; the presets by author, and how many of each
/// play on the web. The same on every target: what a build carries is a
/// flag of the catalogue, not whether this build has the audio.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_catalogue_and_the_presets() {
    use std::collections::BTreeMap;
    let mut text = String::new();
    let mut banks: Vec<(&str, [usize; 4])> = Vec::new();
    for entry in bank::catalogue() {
        if banks.last().is_none_or(|(b, _)| *b != entry.bank) {
            banks.push((entry.bank, [0; 4]));
        }
        let counts = &mut banks.last_mut().expect("just pushed").1;
        counts[0] += 1;
        counts[1] += usize::from(entry.format == bank::Format::FlacDelta);
        counts[2] += usize::from(entry.format == bank::Format::Wav);
        counts[3] += usize::from(entry.on_web);
    }
    for (name, [tables, delta, wav, web]) in &banks {
        let _ = writeln!(
            text,
            "{name:<14} {tables:>3} tables, {delta:>2} as frame deltas, {wav:>2} WAV, {web:>3} on the web"
        );
    }
    let mut authors: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for preset in presets::PRESETS {
        let counts = authors.entry(preset.author).or_default();
        counts.0 += 1;
        counts.1 += usize::from(preset.on_web());
    }
    for (author, (count, web)) in &authors {
        let _ = writeln!(text, "{author:<14} {count:>3} presets, {web:>3} on the web");
    }
    let untyped: Vec<&str> = presets::PRESETS
        .iter()
        .filter(|p| p.kind.is_empty())
        .map(|p| p.name)
        .collect();
    let _ = writeln!(text, "without a TYPE: {untyped:?}");
    let _ = writeln!(
        text,
        "Init plays {:?} and {:?}",
        bank::names()[usize::from(presets::INIT_TABLES[0])],
        bank::names()[usize::from(presets::INIT_TABLES[1])]
    );
    check(
        &text,
        &expect![[r#"
        Adventure Kid   65 tables,  3 as frame deltas,  0 WAV,  65 on the web
        Analog          42 tables, 13 as frame deltas,  0 WAV,  11 on the web
        Distorted       16 tables, 10 as frame deltas,  0 WAV,   8 on the web
        FM              17 tables, 13 as frame deltas,  0 WAV,   8 on the web
        Growl           18 tables, 12 as frame deltas,  0 WAV,   0 on the web
        Hyper            7 tables,  5 as frame deltas,  0 WAV,   1 on the web
        Neu KatalYst    50 tables, 50 as frame deltas,  0 WAV,  10 on the web
        Presets         11 tables,  3 as frame deltas,  8 WAV,  11 on the web
        DoubleTap       37 presets,  23 on the web
        Flamedragonz    52 presets,  23 on the web
        HavenSteps      50 presets,  23 on the web
        IV               9 presets,   6 on the web
        MadMcMan        19 presets,   8 on the web
        murti           50 presets,  19 on the web
        without a TYPE: []
        Init plays "AKWP bw_saw" and "AKWP bw_squ"
    "#]],
    );
}

/// Every table this build carries decodes, to whole frames of samples a
/// 16-bit file can hold; the frames per bank. Native only: the web build
/// carries a part.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn every_table_decodes() {
    let mut frames: Vec<(&str, usize, usize)> = Vec::new();
    for entry in bank::catalogue() {
        let bytes = entry.bytes().expect("a native build carries every table");
        let (samples, _rate) = bank::decode(entry.format, entry.frame, bytes)
            .unwrap_or_else(|e| panic!("{}: {e:?}", entry.name));
        assert!(
            samples.len() >= entry.frame,
            "{}: not one whole frame",
            entry.name
        );
        // A float WAV a preset carried may go past full scale; 16-bit
        // samples cannot.
        let limit = if entry.format == bank::Format::Wav {
            f32::MAX
        } else {
            1.0
        };
        assert!(
            samples.iter().all(|s| s.is_finite() && s.abs() <= limit),
            "{}: a sample out of range",
            entry.name
        );
        if frames.last().is_none_or(|(b, _, _)| *b != entry.bank) {
            frames.push((entry.bank, 0, 0));
        }
        let last = frames.last_mut().expect("just pushed");
        last.1 += samples.len() / entry.frame;
        last.2 += 1;
    }
    let mut text = String::new();
    for (bank, f, t) in &frames {
        let _ = writeln!(text, "{bank}: {f} frames in {t} tables");
    }
    check(
        &text,
        &expect![[r#"
        Adventure Kid: 4358 frames in 65 tables
        Analog: 10395 frames in 42 tables
        Distorted: 2865 frames in 16 tables
        FM: 3765 frames in 17 tables
        Growl: 4299 frames in 18 tables
        Hyper: 1387 frames in 7 tables
        Neu KatalYst: 12800 frames in 50 tables
        Presets: 1187 frames in 11 tables
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn silent_until_its_tables_arrive() {
    let mut plugin = WavetablePlugin::at(RATE);
    let without = peak(&render(&mut plugin, &[on(0, 60)], 4800));
    let mut plugin = with_tables(WavetablePlugin::at(RATE));
    let with = peak(&render(&mut plugin, &[on(0, 60)], 4800));
    check(
        &format!("without tables: {}\nwith: {}\n", db(without), db(with)),
        &expect![[r#"
            without tables: -inf
            with: -18.9 dB
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn an_oscillator_plays_only_the_table_its_knob_names() {
    let mut plugin = with_tables(WavetablePlugin::at(RATE));
    let before = peak(&render(&mut plugin, &[on(0, 60)], 4800));
    render(&mut plugin, &[off(0, 60)], 48000);
    let other = bank::index_of("NK - INERT").expect("a shipped table");
    plugin.set_param(params::id::WT1, other as f32);
    let stale = peak(&render(&mut plugin, &[on(0, 60)], 4800));
    render(&mut plugin, &[off(0, 60)], 48000);
    let tables = bank::tables_for(plugin.table_indices(), RATE, bank::Quality::Render);
    let held = plugin.set_tables(Some(tables));
    let after = peak(&render(&mut plugin, &[on(0, 60)], 4800));
    check(
        &format!(
            "Init tables: {}\nknob moved, old set still held: {} (no oscillator: the last note's DC blocker settling)\nnew set handed over: {}\n\
             the old set came back to the caller: {}\nnow holds {:?}\n",
            db(before),
            db(stale),
            db(after),
            held.is_some(),
            plugin.tables().map(|t| t.indices()),
        ),
        &expect![[r#"
            Init tables: -18.9 dB
            knob moved, old set still held: -72.3 dB (no oscillator: the last note's DC blocker settling)
            new set handed over: -15.7 dB
            the old set came back to the caller: true
            now holds Some([Some(183), Some(31)])
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn a_sub_sine_plays_in_tune_at_every_rate() {
    let mut text = String::new();
    for rate in [44100.0, 48000.0, 96000.0] {
        let mut plugin = with_tables(WavetablePlugin::at(rate));
        set(&mut plugin, "osc1enable", 0.0);
        set(&mut plugin, "subenable", 1.0);
        set(&mut plugin, "subwave", 0.0);
        let out = render(&mut plugin, &[on(0, 69)], rate as usize);
        let _ = writeln!(text, "{rate} Hz: A4 at {:.1} Hz", frequency(&out, rate));
    }
    let mut plugin = with_tables(WavetablePlugin::at(48000.0));
    plugin.reset(44100.0, 512);
    let tables = bank::tables_for(plugin.table_indices(), 44100.0, bank::Quality::Render);
    plugin.set_tables(Some(tables));
    set(&mut plugin, "osc1enable", 0.0);
    set(&mut plugin, "subenable", 1.0);
    let out = render(&mut plugin, &[on(0, 69)], 44100);
    let _ = writeln!(
        text,
        "reset from 48000 to {} Hz: A4 at {:.1} Hz",
        plugin.sample_rate(),
        frequency(&out, 44100.0)
    );
    check(
        &text,
        &expect![[r#"
        44100 Hz: A4 at 440.0 Hz
        48000 Hz: A4 at 440.0 Hz
        96000 Hz: A4 at 440.0 Hz
        reset from 48000 to 44100 Hz: A4 at 440.0 Hz
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn voices_are_bounded() {
    let mut text = String::new();
    for knob in [40.0, 4.0] {
        let mut plugin = with_tables(WavetablePlugin::at(RATE));
        set(&mut plugin, "voices", knob);
        let notes: Vec<NoteEvent> = (30..94).map(|n| on(n as u32, n)).collect();
        render(&mut plugin, &notes, 1024);
        let _ = writeln!(
            text,
            "Voices {knob}: 64 notes held, {} sounding (slots: {VOICE_SLOTS})",
            plugin.active_voices()
        );
    }
    check(
        &text,
        &expect![[r#"
        Voices 40: 64 notes held, 40 sounding (slots: 50)
        Voices 4: 64 notes held, 4 sounding (slots: 50)
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn choke_cuts_and_aftertouch_is_ignored() {
    let held = |events: &[NoteEvent]| {
        let mut plugin = with_tables(WavetablePlugin::at(RATE));
        set(&mut plugin, "release", 2.0);
        render(&mut plugin, &[on(0, 60)], 4800);
        let out = render(&mut plugin, events, 4800);
        (peak(&out[4000..]), plugin.active_voices())
    };
    let note = Note::new(60);
    let (untouched, _) = held(&[]);
    let (touched, _) = held(&[NoteEvent::Aftertouch {
        time: 0,
        channel: 0,
        note,
        pressure: 1.0,
    }]);
    let (released, released_voices) = held(&[off(0, 60)]);
    let (choked, choked_voices) = held(&[NoteEvent::Choke {
        time: 0,
        channel: 0,
        note,
    }]);
    check(
        &format!(
            "100 ms on, held: {}\n\
             held through aftertouch: {}\n\
             released (2 s release): {} ({released_voices} sounding)\n\
             choked: {} ({choked_voices} sounding)\n",
            db(untouched),
            db(touched),
            db(released),
            db(choked),
        ),
        &expect![[r#"
            100 ms on, held: -20.8 dB
            held through aftertouch: -20.8 dB
            released (2 s release): -20.7 dB (1 sounding)
            choked: -inf (0 sounding)
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn presets_load_through_the_preset_knob() {
    let mut plugin = WavetablePlugin::at(RATE);
    let mut text = String::new();
    let siren = presets::PRESETS
        .iter()
        .position(|p| p.name == "DT Siren 1")
        .expect("a shipped preset");
    plugin.set_param(PARAM_PRESET, siren as f32);
    let show = |plugin: &WavetablePlugin, text: &mut String| {
        let mods = (0..params::ROUTES)
            .map(|s| plugin.get_param(params::id::MOD1_SRC + s * params::ROUTE_PARAMS))
            .filter(|&src| src > 0.0)
            .count();
        let _ = writeln!(
            text,
            "preset {} ({}): tables {:?}, osc1pos {}, fltenable {}, {mods} mod slots",
            plugin.get_param(PARAM_PRESET),
            presets::name(plugin.get_param(PARAM_PRESET) as usize),
            plugin.table_indices(),
            plugin.get_param(params::index_of("osc1pos").expect("osc1pos")),
            plugin.get_param(params::index_of("fltenable").expect("fltenable")),
        );
    };
    show(&plugin, &mut text);
    plugin.set_param(PARAM_PRESET, presets::PRESETS.len() as f32);
    show(&plugin, &mut text);
    plugin.set_param(PARAM_PRESET, 999.0);
    show(&plugin, &mut text);
    check(
        &text,
        &expect![[r#"
            preset 26 (DT Siren 1): tables [213, 86], osc1pos 51.153572, fltenable 1, 2 mod slots
            preset 217 (Init): tables [26, 31], osc1pos 0, fltenable 1, 0 mod slots
            preset 217 (Init): tables [26, 31], osc1pos 0, fltenable 1, 0 mod slots
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn the_audio_thread_never_allocates() {
    let mut plugin = with_tables(WavetablePlugin::at(RATE));
    let spare = bank::tables_for(plugin.table_indices(), RATE, bank::Quality::Render);
    for uid in [
        "chEnable",
        "dsEnable",
        "dlEnable",
        "rvEnable",
        "gateenable",
        "lfo1enable",
    ] {
        set(&mut plugin, uid, 1.0);
    }
    render(&mut plugin, &[on(0, 40)], 512);
    let notes: Vec<NoteEvent> = (30..94).map(|n| on(n as u32, n)).collect();
    let (mut left, mut right) = (vec![0.0; 2048], vec![0.0; 2048]);
    let mut text = String::new();
    let count = allocations_during(|| {
        render_into(&mut plugin, &notes, &mut left, &mut right);
    });
    let _ = writeln!(text, "64 notes into every voice, every effect on: {count}");
    let count = allocations_during(|| {
        for preset in 0..=presets::PRESETS.len() {
            plugin.set_param(PARAM_PRESET, preset as f32);
            render_into(
                &mut plugin,
                &[on(0, 60), off(300, 60)],
                &mut left,
                &mut right,
            );
        }
    });
    let _ = writeln!(text, "every preset loaded and played: {count}");
    let mut returned = None;
    let count = allocations_during(|| {
        returned = plugin.set_tables(Some(Arc::clone(&spare)));
        plugin.reset(RATE as f32, 512);
        plugin.silence();
        render_into(&mut plugin, &[on(0, 60)], &mut left, &mut right);
    });
    let _ = writeln!(
        text,
        "tables swapped, reset at the same rate, silenced, played: {count} (old set returned: {})",
        returned.is_some()
    );
    check(
        &text,
        &expect![[r#"
        64 notes into every voice, every effect on: 0
        every preset loaded and played: 0
        tables swapped, reset at the same rate, silenced, played: 0 (old set returned: true)
    "#]],
    );
}

/// How far `a` is from `b`, and from which sample.
fn apart(a: &[f32], b: &[f32]) -> String {
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .map_or("bit for bit".to_string(), |at| {
            format!("{} from sample {at}", db(worst))
        })
}

/// Each preset the web carries played a C4 for a quarter second as it is loaded,
/// against the same preset left a second of silence first: unsettled, its
/// smoothed knobs and its matrix ease in from their defaults and the two
/// differ from the first sample; settled, they start where the preset
/// has them. What moves on its own is switched off, the same both ways —
/// the LFOs, the step LFO, the chorus's sweep, the reverb's modulation,
/// the distortion's dither, the gate and the noise — so what is left is
/// the Delay of DT Siren Complex 2 and DT Siren complex, whose time
/// rounds to no delay at all and reads wherever its line has got to.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn every_preset_starts_where_its_knobs_are() {
    let mut text = String::new();
    // The ones the web carries, so the table is the same on every target.
    for preset in (0..presets::PRESETS.len()).filter(|&p| presets::PRESETS[p].on_web()) {
        let loaded = || {
            let mut plugin = WavetablePlugin::at(RATE);
            plugin.set_param(PARAM_PRESET, preset as f32);
            for uid in [
                "lfo1enable",
                "lfo2enable",
                "lfo3enable",
                "slfoenable",
                "chEnable",
                "rvEnable",
                "dsEnable",
                "gateenable",
                "noiseenable",
            ] {
                set(&mut plugin, uid, 0.0);
            }
            with_tables(plugin)
        };
        let mut waited = loaded();
        let _ = render(&mut waited, &[], 48_128);
        let steady = render(&mut waited, &[on(0, 60)], 12_000);
        let eased = render(&mut loaded(), &[on(0, 60)], 12_000);
        let mut settled = loaded();
        settled.settle();
        let settled = render(&mut settled, &[on(0, 60)], 12_000);
        let _ = writeln!(
            text,
            "{:<18} unsettled {}; settled {}",
            presets::name(preset),
            apart(&eased, &steady),
            apart(&settled, &steady),
        );
    }
    check(
        &text,
        &expect![[r#"
            DT Dive 2          unsettled -93.3 dB from sample 1; settled bit for bit
            DT Dive 3          unsettled -73.8 dB from sample 1; settled bit for bit
            DT Dive            unsettled -93.2 dB from sample 1; settled bit for bit
            DT Kick 1          unsettled -1.0 dB from sample 1; settled bit for bit
            DT Kick 2          unsettled -6.2 dB from sample 0; settled bit for bit
            DT Noise Bass 1    unsettled -9.2 dB from sample 1; settled bit for bit
            DT Noise Bass 2    unsettled -9.8 dB from sample 1; settled bit for bit
            DT Noise Bass 3    unsettled -4.7 dB from sample 1; settled bit for bit
            DT Noise Bass 5    unsettled -7.9 dB from sample 1; settled bit for bit
            DT Noisy Pluck     unsettled -17.2 dB from sample 0; settled bit for bit
            DT Pad 2           unsettled -36.9 dB from sample 0; settled bit for bit
            DT Pad             unsettled -83.6 dB from sample 0; settled bit for bit
            DT Pluck 1         unsettled -10.7 dB from sample 0; settled bit for bit
            DT Pluck 2         unsettled -20.6 dB from sample 0; settled bit for bit
            DT Riser 1         unsettled -92.8 dB from sample 1; settled bit for bit
            DT Seq 1           unsettled -26.2 dB from sample 1; settled bit for bit
            DT Seq 2           unsettled -18.5 dB from sample 1; settled bit for bit
            DT Siren 1         unsettled -35.6 dB from sample 1; settled bit for bit
            DT Siren Complex 2 unsettled -10.1 dB from sample 0; settled -15.2 dB from sample 1
            DT Siren complex   unsettled -22.6 dB from sample 0; settled -87.4 dB from sample 1
            DT Sweeper 4       unsettled -70.0 dB from sample 1; settled bit for bit
            DT Sweeper 5       unsettled -81.2 dB from sample 1; settled bit for bit
            DT Sweeper 6       unsettled -25.9 dB from sample 0; settled bit for bit
            ARP Bass Pulse I   unsettled -7.3 dB from sample 1; settled bit for bit
            ARP Psy I          unsettled -2.8 dB from sample 0; settled bit for bit
            BS FM Bass I       unsettled -20.4 dB from sample 0; settled bit for bit
            BS Jazz Bass Guitar I unsettled -7.8 dB from sample 1; settled bit for bit
            BS Pluck Bass I    unsettled -2.2 dB from sample 1; settled bit for bit
            BS Punch Bass I    unsettled -5.8 dB from sample 1; settled bit for bit
            CH Min Chord III   unsettled -3.6 dB from sample 0; settled -36.2 dB from sample 1
            DR Basic Kick      unsettled -7.7 dB from sample 1; settled bit for bit
            DR Basic Snare I   unsettled -10.7 dB from sample 0; settled bit for bit
            DR Noise Tom I     unsettled -21.2 dB from sample 0; settled bit for bit
            FX Glitch          unsettled -0.5 dB from sample 0; settled -22.9 dB from sample 1
            FX Noise Explosion unsettled -16.0 dB from sample 0; settled bit for bit
            FX Red Alert I     unsettled -48.7 dB from sample 0; settled -86.6 dB from sample 1
            LD Accordion I     unsettled -26.4 dB from sample 0; settled bit for bit
            LD EGuitar Dist I  unsettled -20.2 dB from sample 0; settled bit for bit
            LD NoiseSaw Lead I unsettled -15.3 dB from sample 0; settled -34.0 dB from sample 1
            LD Slide Saw I     unsettled -25.2 dB from sample 0; settled -88.1 dB from sample 1
            PD Analog Pad I    unsettled -31.8 dB from sample 0; settled bit for bit
            PD Fantasy Pad I   unsettled -44.6 dB from sample 0; settled -110.9 dB from sample 1
            PD Neu Atmopad I   unsettled -67.3 dB from sample 0; settled bit for bit
            PL Noise String I  unsettled -10.0 dB from sample 0; settled bit for bit
            PL SawSquare II    unsettled -1.6 dB from sample 0; settled -15.9 dB from sample 1
            VOX Dist Vocal I   unsettled -34.8 dB from sample 0; settled bit for bit
            8bit dream         unsettled -56.9 dB from sample 1; settled bit for bit
            16bit Dream        unsettled -59.3 dB from sample 1; settled bit for bit
            80s Dirty Lead     unsettled -3.2 dB from sample 0; settled bit for bit
            90sEP              unsettled -8.3 dB from sample 1; settled bit for bit
            Angles             unsettled -61.8 dB from sample 1; settled bit for bit
            Bassity            unsettled -10.4 dB from sample 1; settled bit for bit
            BeauKeys           unsettled -0.7 dB from sample 1; settled bit for bit
            CrazyBrass         unsettled -52.2 dB from sample 1; settled bit for bit
            CrazyOrgan         unsettled -18.4 dB from sample 96; settled bit for bit
            Duke Bass          unsettled -10.7 dB from sample 1; settled bit for bit
            E-Bass             unsettled -0.3 dB from sample 1; settled bit for bit
            Flutish            unsettled -3.2 dB from sample 1; settled bit for bit
            Lazzzer            unsettled 1.8 dB from sample 1; settled bit for bit
            LostTechno         unsettled -17.5 dB from sample 1; settled bit for bit
            Medas4             unsettled -8.9 dB from sample 1; settled bit for bit
            MovieTrailerS      unsettled -55.9 dB from sample 0; settled bit for bit
            PadThai            unsettled -97.8 dB from sample 1; settled bit for bit
            Play That Funky Music unsettled -4.1 dB from sample 1; settled bit for bit
            Pulse Droner       unsettled -13.3 dB from sample 0; settled bit for bit
            Raiser             unsettled -74.1 dB from sample 1; settled bit for bit
            SomeChurch         unsettled -19.2 dB from sample 96; settled bit for bit
            SteelS             unsettled -11.2 dB from sample 1; settled bit for bit
            The Classic        unsettled -19.7 dB from sample 1; settled bit for bit
            IV's Cyber Lead (MW) unsettled -11.4 dB from sample 0; settled bit for bit
            IV's Heart Bells   unsettled -14.5 dB from sample 0; settled -18.1 dB from sample 1
            IV's Init          unsettled -13.3 dB from sample 1; settled bit for bit
            IV's Organ Advanced unsettled -25.6 dB from sample 0; settled bit for bit
            IV's Organ Basic   unsettled -17.6 dB from sample 0; settled bit for bit
            IV's Organ Superchorus unsettled -13.2 dB from sample 0; settled bit for bit
            Guitar Pad         unsettled -39.2 dB from sample 0; settled bit for bit
            Hat                unsettled -58.6 dB from sample 1; settled bit for bit
            hh                 unsettled -58.6 dB from sample 1; settled bit for bit
            Mice               unsettled -21.9 dB from sample 0; settled bit for bit
            Noise Riserish     unsettled bit for bit; settled bit for bit
            talky distortion   unsettled -9.4 dB from sample 0; settled -24.3 dB from sample 1
            Whip               unsettled bit for bit; settled bit for bit
            Wobly Pad          unsettled -18.5 dB from sample 1; settled bit for bit
            5th Wheel          unsettled -9.6 dB from sample 1; settled bit for bit
            7th Heaven         unsettled -7.3 dB from sample 1; settled bit for bit
            Breathe            unsettled -32.6 dB from sample 0; settled bit for bit
            Clean Dirt         unsettled -9.9 dB from sample 0; settled bit for bit
            Distant Relative   unsettled -7.9 dB from sample 0; settled bit for bit
            Dragonball         unsettled -23.2 dB from sample 1; settled bit for bit
            Flower             unsettled -9.3 dB from sample 0; settled bit for bit
            Heaven Ascent      unsettled -28.1 dB from sample 0; settled bit for bit
            Lurkin             unsettled -30.2 dB from sample 0; settled bit for bit
            Magnet Bass        unsettled -8.1 dB from sample 0; settled bit for bit
            Modular Waves      unsettled -27.7 dB from sample 1; settled bit for bit
            Pulse              unsettled -33.4 dB from sample 0; settled bit for bit
            Self-Made          unsettled -76.2 dB from sample 0; settled bit for bit
            Sign Pad           unsettled -27.4 dB from sample 0; settled bit for bit
            Sine Wub           unsettled -30.5 dB from sample 1; settled bit for bit
            Square-1           unsettled -24.3 dB from sample 1; settled bit for bit
            Swelter            unsettled -47.3 dB from sample 0; settled bit for bit
            Talky              unsettled -35.1 dB from sample 1; settled bit for bit
            Vide Bells         unsettled -8.9 dB from sample 0; settled bit for bit
        "#]],
    );
}
