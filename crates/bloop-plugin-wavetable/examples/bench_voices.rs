//! What a Wavetable costs at full polyphony, second by second.
//!
//! Forty notes held on one preset, rendered in 192-sample blocks (`--block B`) at
//! 48 kHz for ten seconds (`--seconds S`); each line is one second of audio and what it
//! cost, as a share of one core. `--ftz` sets the x86 flush-to-zero and
//! denormals-are-zero bits first, so a run with and without it says
//! whether denormals are what the time goes on.
//!
//! Run: `cargo run -p bloop-plugin-wavetable --example bench_voices --release -- [--preset N] [--block B] [--seconds S] [--ftz] [--exact] [--set uid=value]...`
//!
//! The tables are the compact form a live engine plays; `--exact` plays the
//! exact form a render does instead (`bank::Quality`).

#![expect(clippy::print_stdout, reason = "a bench says what it measured")]
#![expect(clippy::expect_used, reason = "a bench with a bad argument has no one to tell but its user")]

use bloop_events::{EventBuffer, NoteEvent};
use bloop_note::Note;
use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_wavetable::{bank, presets, WavetablePlugin, PARAM_PRESET};
use std::time::Instant;

const RATE: f64 = 48_000.0;

#[cfg(target_arch = "x86_64")]
fn flush_to_zero() {
    // FTZ (bit 15) and DAZ (bit 6) of MXCSR, as JUCE's ScopedNoDenormals.
    #[expect(deprecated, reason = "the intrinsic is what sets MXCSR")]
    // Safety: setting MXCSR's FTZ and DAZ bits changes only how
    // denormals round.
    unsafe {
        core::arch::x86_64::_mm_setcsr(core::arch::x86_64::_mm_getcsr() | 0x8040);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn flush_to_zero() {}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let preset = args
        .iter()
        .position(|a| a == "--preset")
        .and_then(|at| args.get(at + 1))
        .map_or(presets::PRESETS.len(), |n| n.parse().expect("a preset number"));
    if args.iter().any(|a| a == "--ftz") {
        flush_to_zero();
    }
    let mut plugin = WavetablePlugin::at(RATE);
    plugin.set_param(PARAM_PRESET, preset as f32);
    // `--set uid=value`, any number of times: a knob moved after the
    // preset, to see what it costs.
    for (at, arg) in args.iter().enumerate() {
        if arg == "--set" {
            let (uid, value) = args[at + 1].split_once('=').expect("--set uid=value");
            let index = bloop_plugin_wavetable::params::index_of(uid).expect("a parameter");
            plugin.set_param(index, value.parse().expect("a number"));
        }
    }
    let quality = if args.iter().any(|a| a == "--exact") {
        bank::Quality::Render
    } else {
        bank::Quality::Live
    };
    let tables = bank::tables_for(plugin.table_indices(), RATE, quality);
    println!("tables: {quality:?}, {:.1} MB", tables.bytes() as f64 / 1e6);
    plugin.set_tables(Some(tables));
    println!("preset {preset}: {}", presets::name(preset));

    let block: usize = args
        .iter()
        .position(|a| a == "--block")
        .and_then(|at| args.get(at + 1))
        .map_or(192, |n| n.parse().expect("a block size"));
    let (mut left, mut right) = (vec![0.0f32; block], vec![0.0f32; block]);
    let blocks_per_second = RATE as usize / block;
    let seconds = args
        .iter()
        .position(|a| a == "--seconds")
        .and_then(|at| args.get(at + 1))
        .map_or(10, |n| n.parse().expect("a number of seconds"));
    for second in 0..seconds {
        let start = Instant::now();
        let mut peak = 0.0f32;
        for block in 0..blocks_per_second {
            let mut events = EventBuffer::new();
            if second == 0 && block == 0 {
                for n in 0..40u8 {
                    events.push(NoteEvent::NoteOn {
                        time: 0,
                        channel: 0,
                        note: Note::new(36 + n),
                        velocity: 1.0,
                        id: bloop_events::NoteId::NONE,
                    });
                }
            }
            let mut outputs: [&mut [f32]; 2] = [&mut left, &mut right];
            let mut context = ProcessContext {
                outputs: &mut outputs,
                input_events: &events,
                sample_rate: RATE as f32,
                tempo: 120.0,
                is_playing: true,
            };
            plugin.process(&mut context);
            peak = left.iter().fold(peak, |m, s| m.max(s.abs()));
        }
        let cost = start.elapsed().as_secs_f64();
        println!(
            "second {second}: cpu_load {:5.1}%  peak {peak:.3}  voices {}",
            cost * 100.0,
            plugin.active_voices()
        );
    }
}
