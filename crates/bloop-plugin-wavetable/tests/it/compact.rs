//! The compact form of a wavetable, which a live engine plays, against
//! the exact form a render plays and the gold tests hold to the original.
//!
//! Each case plays one oscillator, from each form, at a note, and
//! measures, in decibels against the exact form's level:
//!
//! - `diff`: the whole difference between the two;
//! - `low`: the difference below a quarter of the sample rate, where both
//!   keep every harmonic the note has: what the 16-bit samples and the
//!   shorter copies cost;
//! - `alias exact`, `alias compact`: what lies off the note's harmonics,
//!   which is aliasing and interpolation images.
//!
//! What the compact form gives up is the top octave of harmonics for
//! some notes (`dsp::tables::CompactFrame`), which `diff` shows and `low`
//! does not.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]
#![expect(clippy::panic, reason = "a case that names a table there is not is a broken test")]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

use bloop_events::{EventBuffer, NoteEvent};
use bloop_note::Note;
use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_wavetable::bank::{self, Quality};
use bloop_plugin_wavetable::dsp::fft::{Complex, Fft};
use bloop_plugin_wavetable::dsp::osc::{VoicedParams, VoicedWtOscillator};
use bloop_plugin_wavetable::dsp::tables::Table;
use bloop_plugin_wavetable::{presets, WavetablePlugin, PARAM_PRESET};
use expect_test::{expect, Expect};
use std::fmt::Write as _;


const RATE: f64 = 48_000.0;
/// Samples analysed per case: a power of two, for the FFT, long enough
/// that the lowest note's harmonics lie forty bins apart.
const LEN: usize = 1 << 16;
/// How far from a harmonic, in bins, the window's main lobe reaches.
const LOBE: f64 = 4.0;

fn table(name: &str, quality: Quality) -> Table {
    let index = bank::index_of(name).unwrap_or_else(|| panic!("no table {name}"));
    bank::build(index, RATE, quality).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

/// `LEN` samples of one oscillator over `table` at `note`, a second in so
/// the DC blocker has settled, in 32-sample blocks as a voice renders.
fn play(table: &Table, note: f32, position: f32) -> Vec<f32> {
    let mut osc = VoicedWtOscillator::default();
    osc.set_sample_rate(RATE);
    osc.note_on_all(0.0);
    let params = VoicedParams {
        position,
        ..VoicedParams::default()
    };
    let settle = RATE as usize;
    let mut out = Vec::with_capacity(settle + LEN);
    while out.len() < settle + LEN {
        let (mut l, mut r) = ([0.0f32; 32], [0.0f32; 32]);
        osc.process_adding(Some(table), note, &params, &mut l, &mut r);
        out.extend_from_slice(&l);
    }
    out[settle..settle + LEN].to_vec()
}

/// The spectrum under a four-term Blackman-Harris window, whose side
/// lobes lie 92 dB down, bins 0 to `LEN / 2`.
fn spectrum(signal: &[f32]) -> Vec<Complex> {
    let fft = Fft::new(LEN.trailing_zeros());
    let tau = 2.0 * std::f64::consts::PI / LEN as f64;
    let windowed: Vec<Complex> = signal
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let x = tau * i as f64;
            let w = 0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos()
                - 0.01168 * (3.0 * x).cos();
            Complex {
                re: (f64::from(s) * w) as f32,
                im: 0.0,
            }
        })
        .collect();
    let mut out = vec![Complex::default(); LEN];
    fft.perform(&windowed, &mut out, false);
    out.truncate(LEN / 2 + 1);
    out
}

fn power(c: Complex) -> f64 {
    f64::from(c.re).powi(2) + f64::from(c.im).powi(2)
}

fn db(ratio: f64) -> f64 {
    if ratio <= 0.0 {
        -200.0
    } else {
        10.0 * ratio.log10()
    }
}

/// The frequency the oscillator plays `note` at: its phase step, as
/// `phase_delta` rounds it, times the rate.
fn played(note: f32) -> f64 {
    let freq = (RATE / 2.0).min(440.0 * 2f64.powf((f64::from(note) - 69.0) / 12.0)) as f32;
    let delta = 1.0 / (((1.0 / freq) as f64 * RATE) as f32);
    f64::from(delta) * RATE
}

/// Whether bin `bin` lies on one of the note's harmonics, within the
/// window's main lobe.
fn on_a_harmonic(bin: usize, freq: f64) -> bool {
    let bin_hz = RATE / LEN as f64;
    let h = (bin as f64 * bin_hz / freq).round();
    h >= 1.0 && (bin as f64 - h * freq / bin_hz).abs() <= LOBE
}

struct Measured {
    diff: f64,
    low: f64,
    alias_exact: f64,
    alias_compact: f64,
}

fn measure(exact: &Table, compact: &Table, note: f32, position: f32) -> Measured {
    let (e, c) = (
        spectrum(&play(exact, note, position)),
        spectrum(&play(compact, note, position)),
    );
    let freq = played(note);
    let total: f64 = e.iter().map(|x| power(*x)).sum();
    let quarter = LEN / 4;
    let mut diff = 0.0;
    let mut low = 0.0;
    let (mut alias_e, mut alias_c) = (0.0, 0.0);
    for (bin, (x, y)) in e.iter().zip(&c).enumerate() {
        let d = power(Complex {
            re: x.re - y.re,
            im: x.im - y.im,
        });
        diff += d;
        if bin < quarter && on_a_harmonic(bin, freq) {
            low += d;
        }
        if bin as f64 > LOBE && !on_a_harmonic(bin, freq) {
            alias_e += power(*x);
            alias_c += power(*y);
        }
    }
    Measured {
        diff: db(diff / total),
        low: db(low / total),
        alias_exact: db(alias_e / total),
        alias_compact: db(alias_c / total),
    }
}

/// The compact form's aliasing may be this far under the signal whatever
/// the exact form's is.
const ALIAS_FLOOR: f64 = -70.0;
/// The most the two may differ on the harmonics under a quarter of the
/// rate.
const LOW_MOST: f64 = -60.0;

/// Every note of `notes` on table `name`, a line each; and it holds the
/// compact form's aliasing to within `margin` dB of the exact form's, or
/// under [`ALIAS_FLOOR`], and their harmonics under a quarter of the rate
/// to within [`LOW_MOST`] of each other.
fn check(name: &str, position: f32, notes: &[f32], margin: f64, expect: &Expect) {
    let exact = table(name, Quality::Render);
    let compact = table(name, Quality::Live);
    let mut out = String::new();
    let _ = writeln!(out, "{name} at {:.0}%", position * 100.0);
    let mut worst_low = -200.0f64;
    let mut worst_alias = -200.0f64;
    let mut failed = Vec::new();
    for &note in notes {
        let m = measure(&exact, &compact, note, position);
        let _ = writeln!(
            out,
            "  note {note:5.1}: diff {:6.1}  low {:6.1}  alias exact {:6.1}  compact {:6.1}",
            m.diff, m.low, m.alias_exact, m.alias_compact
        );
        if m.alias_compact > (m.alias_exact + margin).max(ALIAS_FLOOR) {
            failed.push(format!(
                "note {note}: the compact form aliases at {:.1} dB, the exact one at {:.1} dB",
                m.alias_compact, m.alias_exact
            ));
        }
        if m.low > LOW_MOST {
            failed.push(format!("note {note}: {:.1} dB apart below a quarter of the rate", m.low));
        }
        worst_low = worst_low.max(m.low);
        worst_alias = worst_alias.max(m.alias_compact - m.alias_exact);
    }
    let _ = writeln!(
        out,
        "  worst: low {worst_low:.1}, compact aliasing {worst_alias:+.1} dB on the exact form's"
    );
    expect.assert_eq(&out);
    assert!(failed.is_empty(), "{name}: {failed:#?}");
}

const NOTES: [f32; 9] = [24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0];

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn a_saw_in_both_forms() {
    check("AKWP bw_saw", 0.0, &NOTES, 3.0, &expect![[r#"
        AKWP bw_saw at 0%
          note  24.0: diff  -40.5  low  -80.2  alias exact  -53.8  compact  -58.8
          note  36.0: diff  -37.6  low  -90.1  alias exact  -62.5  compact  -64.0
          note  48.0: diff  -29.1  low -105.3  alias exact  -68.5  compact  -72.6
          note  60.0: diff  -25.8  low -108.9  alias exact  -77.3  compact  -80.9
          note  72.0: diff  -22.9  low -110.9  alias exact  -83.9  compact  -85.4
          note  84.0: diff  -20.0  low  -87.7  alias exact  -88.9  compact  -85.1
          note  96.0: diff  -17.1  low  -83.0  alias exact  -87.7  compact  -82.2
          note 108.0: diff  -15.6  low  -80.8  alias exact  -88.2  compact  -79.2
          note 120.0: diff  -79.3  low  -87.4  alias exact  -87.1  compact  -84.0
          worst: low -80.2, compact aliasing +8.9 dB on the exact form's
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn a_morphing_table_in_both_forms() {
    check("NK - INERT", 0.6, &NOTES, 3.0, &expect![[r#"
        NK - INERT at 60%
          note  24.0: diff  -63.0  low -100.1  alias exact  -63.2  compact  -63.3
          note  36.0: diff  -39.3  low  -90.7  alias exact  -64.4  compact  -66.4
          note  48.0: diff  -30.1  low -108.0  alias exact  -69.5  compact  -73.5
          note  60.0: diff  -27.8  low -109.6  alias exact  -78.8  compact  -82.2
          note  72.0: diff  -22.4  low -113.6  alias exact  -84.0  compact  -86.2
          note  84.0: diff  -21.0  low  -88.7  alias exact  -88.7  compact  -86.8
          note  96.0: diff  -21.5  low  -82.9  alias exact  -87.4  compact  -81.5
          note 108.0: diff  -36.5  low  -78.6  alias exact  -87.9  compact  -76.8
          note 120.0: diff  -77.3  low  -88.2  alias exact  -87.7  compact  -83.2
          worst: low -78.6, compact aliasing +11.1 dB on the exact form's
    "#]]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn an_analog_table_in_both_forms() {
    check("Analog Classic", 0.3, &NOTES, 3.0, &expect![[r#"
        Analog Classic at 30%
          note  24.0: diff  -35.6  low  -75.4  alias exact  -48.9  compact  -53.8
          note  36.0: diff  -32.0  low  -83.1  alias exact  -58.9  compact  -63.4
          note  48.0: diff  -28.8  low -105.6  alias exact  -68.2  compact  -72.5
          note  60.0: diff  -25.8  low -108.3  alias exact  -77.2  compact  -80.9
          note  72.0: diff  -22.9  low -110.7  alias exact  -83.9  compact  -85.5
          note  84.0: diff  -20.0  low  -87.4  alias exact  -88.9  compact  -85.2
          note  96.0: diff  -17.1  low  -83.0  alias exact  -87.7  compact  -82.3
          note 108.0: diff  -15.6  low  -80.7  alias exact  -88.2  compact  -79.2
          note 120.0: diff  -79.3  low  -87.1  alias exact  -87.1  compact  -84.0
          worst: low -75.4, compact aliasing +8.9 dB on the exact form's
    "#]]);
}

/// Detuned off the six-note grid the exact form's copies sit on, and in
/// between the octaves the compact form's do.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn notes_between_the_copies() {
    check(
        "AKWP bw_saw",
        0.0,
        &[29.3, 41.9, 53.5, 66.7, 77.1, 89.9, 101.2, 113.6, 126.9],
        3.0,
        &expect![[r#"
            AKWP bw_saw at 0%
              note  29.3: diff  -52.7  low  -96.3  alias exact  -58.6  compact  -58.3
              note  41.9: diff  -45.4  low -101.6  alias exact  -64.3  compact  -64.1
              note  53.5: diff  -41.5  low -100.0  alias exact  -72.9  compact  -72.7
              note  66.7: diff  -22.9  low -109.0  alias exact  -86.4  compact  -89.5
              note  77.1: diff  -32.2  low -111.4  alias exact  -87.7  compact  -87.0
              note  89.9: diff  -26.1  low  -91.6  alias exact  -96.2  compact  -86.5
              note 101.2: diff  -19.9  low  -85.5  alias exact  -90.0  compact  -82.6
              note 113.6: diff  -13.4  low  -80.4  alias exact  -98.8  compact  -79.4
              note 126.9: diff  -85.1  low -200.0  alias exact  -95.8  compact  -90.6
              worst: low -80.4, compact aliasing +19.4 dB on the exact form's
        "#]],
    );
}

/// What each form keeps of a table, and its shape: copies a frame and how
/// long each is.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn memory_per_table() {
    let mut out = String::new();
    for name in ["AKWP bw_saw", "NK - INERT", "Analog Classic"] {
        let exact = table(name, Quality::Render);
        let compact = table(name, Quality::Live);
        let frames = exact.len();
        let _ = writeln!(
            out,
            "{name}: {frames} frames; exact {:.2} MB, compact {:.2} MB ({:.1}x smaller)",
            exact.bytes() as f64 / 1e6,
            compact.bytes() as f64 / 1e6,
            exact.bytes() as f64 / compact.bytes() as f64
        );
    }
    let compact = table("AKWP bw_saw", Quality::Live);
    let frame = compact.compact().map(|t| t.frame(0)).expect("compact");
    let lens: Vec<String> = (0..frame.copies())
        .map(|c| frame.copy_len(c).to_string())
        .collect();
    let _ = writeln!(out, "compact copy lengths: {}", lens.join(" "));
    let exact = table("AKWP bw_saw", Quality::Render);
    let first = exact.exact().map(|t| t.frame(0)).expect("exact");
    let _ = writeln!(
        out,
        "exact: {} copies of {}",
        first.copies(),
        first.table_size() + 1
    );
    expect![[r#"
        AKWP bw_saw: 50 frames; exact 8.61 MB, compact 1.47 MB (5.9x smaller)
        NK - INERT: 256 frames; exact 44.06 MB, compact 7.51 MB (5.9x smaller)
        Analog Classic: 8 frames; exact 1.38 MB, compact 0.23 MB (5.9x smaller)
        compact copy lengths: 2048 2048 2048 2048 2048 2048 1024 512 256 256 256
        exact: 21 copies of 2049
    "#]].assert_eq(&out);
}

/// One note of `preset` with tables of `quality`, half a second held and
/// a quarter released: its peak level, in dB, and whether every sample
/// was finite.
#[cfg(not(target_arch = "wasm32"))]
fn preset_level(index: usize, quality: Quality) -> (f64, bool) {
    let mut plugin = WavetablePlugin::at(RATE);
    plugin.set_param(PARAM_PRESET, index as f32);
    plugin.set_tables(Some(bank::tables_for(plugin.table_indices(), RATE, quality)));
    let mut peak = 0.0f32;
    let mut finite = true;
    let blocks = (0.75 * RATE) as usize / 256;
    for block in 0..blocks {
        let mut events = EventBuffer::new();
        if block == 0 {
            events.push(NoteEvent::NoteOn {
                time: 0,
                channel: 0,
                note: Note::new(60),
                velocity: 0.8,
                id: bloop_events::NoteId::NONE,
            });
        }
        if block == blocks * 2 / 3 {
            events.push(NoteEvent::NoteOff {
                time: 0,
                channel: 0,
                note: Note::new(60),
                id: bloop_events::NoteId::NONE,
            });
        }
        let (mut l, mut r) = (vec![0.0f32; 256], vec![0.0f32; 256]);
        let mut outputs: [&mut [f32]; 2] = [&mut l, &mut r];
        let mut context = ProcessContext {
            outputs: &mut outputs,
            input_events: &events,
            sample_rate: RATE as f32,
            tempo: 120.0,
            is_playing: true,
        };
        plugin.process(&mut context);
        for s in l.iter().chain(&r) {
            finite &= s.is_finite();
            peak = peak.max(s.abs());
        }
    }
    let level = if peak > 0.0 {
        20.0 * f64::from(peak).log10()
    } else {
        -200.0
    };
    (level, finite)
}

/// Every factory preset of the `part`th quarter plays in both forms, as
/// loud as each other to within a decibel and a half. Every preset's
/// tables are in the native build only.
#[cfg(not(target_arch = "wasm32"))]
fn presets_play_in_both_forms(part: usize, expect: &Expect) {
    let quarter = presets::PRESETS.len().div_ceil(4);
    let mut played = 0;
    let mut silent = Vec::new();
    let mut worst = (0.0f64, "");
    for (index, preset) in presets::PRESETS
        .iter()
        .enumerate()
        .skip(part * quarter)
        .take(quarter)
    {
        let (exact, finite_exact) = preset_level(index, Quality::Render);
        let (compact, finite_compact) = preset_level(index, Quality::Live);
        assert!(finite_exact && finite_compact, "{}: not finite", preset.name);
        if exact <= -120.0 {
            silent.push(preset.name);
            assert!(compact <= -120.0, "{}: silent exact, not compact", preset.name);
            continue;
        }
        let gap = (exact - compact).abs();
        assert!(
            gap <= 1.5,
            "{}: exact peaks at {exact:.1} dB, compact at {compact:.1} dB",
            preset.name
        );
        if gap > worst.0 {
            worst = (gap, preset.name);
        }
        played += 1;
    }
    expect.assert_eq(&format!(
        "{played} presets played in both; silent on a held C4 in both: {silent:?}\nlargest peak gap {:.2} dB ({})\n",
        worst.0, worst.1
    ));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn every_preset_plays_in_both_forms_first_quarter() {
    presets_play_in_both_forms(0, &expect![[r#"
        55 presets played in both; silent on a held C4 in both: []
        largest peak gap 0.56 dB (DT Noisy Pluck)
    "#]]);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn every_preset_plays_in_both_forms_second_quarter() {
    presets_play_in_both_forms(1, &expect![[r#"
        55 presets played in both; silent on a held C4 in both: []
        largest peak gap 0.19 dB (Dramahit)
    "#]]);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn every_preset_plays_in_both_forms_third_quarter() {
    presets_play_in_both_forms(2, &expect![[r#"
        55 presets played in both; silent on a held C4 in both: []
        largest peak gap 0.56 dB (Medas4)
    "#]]);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "heavy: cargo test -p bloop-plugin-wavetable -- --include-ignored"]
fn every_preset_plays_in_both_forms_last_quarter() {
    presets_play_in_both_forms(3, &expect![[r#"
        52 presets played in both; silent on a held C4 in both: []
        largest peak gap 0.52 dB (Petal Swell)
    "#]]);
}
