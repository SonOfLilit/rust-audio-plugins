//! Wavetable as a host sees it: what it says about itself, how its
//! tables, presets and voices behave, and what `process` costs. What it
//! sounds like against the original is tests/gold.rs.

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
fn render_into(plugin: &mut WavetablePlugin, events: &[NoteEvent], left: &mut [f32], right: &mut [f32]) {
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
    let tables = bank::tables_for(plugin.table_indices(), plugin.sample_rate());
    plugin.set_tables(Some(tables));
    plugin
}

/// Upward zero crossings a second, after the first tenth of a second.
fn frequency(samples: &[f32], rate: f64) -> f64 {
    let skip = (rate / 10.0) as usize;
    let tail = &samples[skip..];
    let crossings = tail.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
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
        "osc1pos", "osc1level", "osc1unison", "fltfreq", "flttype", "lfo1rate", "lfo1fade",
        "env1sustain", "voices", "dlTime", "wt1", "wt2", "mod1src", "mod1dst", "mod1depth",
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
    let _ = writeln!(text, "past the end: {:?}", param_info(PARAM_PRESET + 1).name);
    check(&text, &expect![[r#"
        Wavetable by SocaLabs, Instrument, 0 in, 2 out, 304 parameters
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
        303 preset       Preset           0 to 8, default 8 (8 factory presets, then Init)
        past the end: ""
    "#]]);
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
    let tables = bank::tables_for(plugin.table_indices(), RATE);
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
            now holds Some([Some(83), Some(31)])
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
    let tables = bank::tables_for(plugin.table_indices(), 44100.0);
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
    check(&text, &expect![[r#"
        44100 Hz: A4 at 440.0 Hz
        48000 Hz: A4 at 440.0 Hz
        96000 Hz: A4 at 440.0 Hz
        reset from 48000 to 44100 Hz: A4 at 440.0 Hz
    "#]]);
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
    check(&text, &expect![[r#"
        Voices 40: 64 notes held, 40 sounding (slots: 50)
        Voices 4: 64 notes held, 4 sounding (slots: 50)
    "#]]);
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
    let (touched, _) = held(&[NoteEvent::Aftertouch { time: 0, channel: 0, note, pressure: 1.0 }]);
    let (released, released_voices) = held(&[off(0, 60)]);
    let (choked, choked_voices) = held(&[NoteEvent::Choke { time: 0, channel: 0, note }]);
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
        let mods = (0..params::MOD_SLOTS)
            .map(|s| plugin.get_param(params::id::MOD1_SRC + s * 3))
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
    check(&text, &expect![[r#"
        preset 0 (DT Siren 1): tables [113, 31], osc1pos 51.153572, fltenable 1, 2 mod slots
        preset 8 (Init): tables [26, 31], osc1pos 0, fltenable 1, 0 mod slots
        preset 8 (Init): tables [26, 31], osc1pos 0, fltenable 1, 0 mod slots
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn the_audio_thread_never_allocates() {
    let mut plugin = with_tables(WavetablePlugin::at(RATE));
    let spare = bank::tables_for(plugin.table_indices(), RATE);
    for uid in ["chEnable", "dsEnable", "dlEnable", "rvEnable", "gateenable", "lfo1enable"] {
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
            render_into(&mut plugin, &[on(0, 60), off(300, 60)], &mut left, &mut right);
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
    check(&text, &expect![[r#"
        64 notes into every voice, every effect on: 0
        every preset loaded and played: 0
        tables swapped, reset at the same rate, silenced, played: 0 (old set returned: true)
    "#]]);
}

