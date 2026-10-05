//! What a wavetable's band-limited copies cost in memory, and what
//! keeping fewer samples of the high ones would cost in sound.
//!
//! A frame is band-limited 21 times, once per six notes, each copy 2049
//! samples. The copy for the top notes holds only a few harmonics, so it
//! could be kept at a fraction of the samples. For each copy and each
//! decimation (every 2nd, 4th … sample), this plays the copy at the lowest
//! note that reads it, for a second, from the full copy and from the
//! decimated one through the same linear interpolation, and prints the
//! difference relative to the signal, the worst over every frame.
//!
//! Run: `cargo run -p bloop-plugin-wavetable --example table_memory --release -- [table name]`

#![expect(clippy::print_stdout, reason = "a measurement says what it measured")]
#![expect(clippy::expect_used, reason = "a measurement with a bad argument has no one to tell but its user")]

use bloop_plugin_wavetable::bank;
use bloop_plugin_wavetable::dsp::tables::linear_in;

const RATE: f64 = 48_000.0;
const FACTORS: [usize; 6] = [2, 4, 8, 16, 32, 64];

/// `copy` with every `factor`th sample kept, and the wrap sample after.
fn decimate(copy: &[f32], factor: usize) -> Vec<f32> {
    let size = copy.len() - 1;
    let mut out: Vec<f32> = copy[..size].iter().step_by(factor).copied().collect();
    out.push(out[0]);
    out
}

/// A second of `copy` played at `note`, as the oscillator steps through it.
fn play(copy: &[f32], note: f32) -> Vec<f32> {
    let freq = 440.0 * 2f64.powf((f64::from(note) - 69.0) / 12.0);
    let delta = (freq / RATE) as f32;
    let mut phase = 0.0f32;
    (0..RATE as usize)
        .map(|_| {
            let s = linear_in(copy, phase.min(1.0 - f32::EPSILON));
            phase += delta;
            if phase >= 1.0 {
                phase -= 1.0;
            }
            s
        })
        .collect()
}

fn db(ratio: f64) -> f64 {
    if ratio <= 0.0 { -200.0 } else { 20.0 * ratio.log10() }
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "NK - INERT".to_string());
    let index = bank::index_of(&name).expect("a table in the catalogue");
    let table = bank::load(index, RATE).expect("the table builds");
    let first = table.frame(0);
    let copies = first.copies();
    let notes = first.notes_per_table() as f32;
    println!(
        "{name}: {} frames x {copies} copies x {} samples = {:.1} MB",
        table.len(),
        first.table_size() + 1,
        table.bytes() as f64 / 1e6
    );
    print!("copy  notes      ");
    for f in FACTORS {
        print!("  1/{f:<5}");
    }
    println!("  (worst difference, dB)");
    // worst[copy][factor]
    let mut worst = vec![[-200.0f64; FACTORS.len()]; copies];
    for f in 0..table.len() {
        let frame = table.frame(f);
        for (c, row) in worst.iter_mut().enumerate() {
            let copy = frame.copy(c);
            let note = c as f32 * notes + 0.5;
            let full = play(copy, note);
            let power = full.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>();
            if power == 0.0 {
                continue;
            }
            for (i, factor) in FACTORS.into_iter().enumerate() {
                let small = play(&decimate(copy, factor), note);
                let diff = full
                    .iter()
                    .zip(&small)
                    .map(|(a, b)| f64::from(a - b).powi(2))
                    .sum::<f64>();
                row[i] = row[i].max(db((diff / power).sqrt()));
            }
        }
    }
    for (c, row) in worst.iter().enumerate() {
        let low = c as f32 * notes + 0.5;
        print!("{c:4}  {:5.1}-{:5.1}", low, low + notes);
        for v in row {
            print!("  {v:7.1}");
        }
        println!();
    }
    for threshold in [-120.0, -96.0, -80.0] {
        let kept: f64 = worst
            .iter()
            .map(|row| {
                let factor = FACTORS
                    .iter()
                    .zip(row)
                    .filter(|(_, v)| **v <= threshold)
                    .map(|(f, _)| *f)
                    .max()
                    .unwrap_or(1);
                1.0 / factor as f64
            })
            .sum();
        println!(
            "keeping every copy within {threshold} dB of the full one: {:.1}% of the samples, {:.1} MB",
            kept / copies as f64 * 100.0,
            table.bytes() as f64 * kept / copies as f64 / 1e6
        );
    }
}
