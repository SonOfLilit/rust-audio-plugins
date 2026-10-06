//! What each plug-in does to real signals, in numbers that can be checked
//! against what it is meant to do: an EQ's boost at the frequencies it
//! boosts, a saturator's harmonics, a tape's compression, a reverb's tail.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use bloop_plugin_airwindows::{Airwindows, Kind};
use expect_test::{expect, Expect};
use std::fmt::Write as _;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

const PI: f64 = core::f64::consts::PI;

/// The provenance comment in Cargo.toml names the revision the License
/// page cites, so the two cannot drift apart.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn cargo_toml_cites_the_ported_revision() {
    let manifest = include_str!("../../Cargo.toml");
    let rev = bloop_plugin_airwindows::PORTED_FROM_AIRWINDOWS_REV;
    assert!(manifest.contains(rev), "Cargo.toml must cite {rev}");
}

fn gain(db: f64) -> f64 {
    libm::pow(10.0, db / 20.0)
}

fn db(x: f64) -> f64 {
    20.0 * libm::log10(x)
}

fn sine(hz: f64, level_db: f64, rate: f64, seconds: f64) -> Vec<f32> {
    (0..(seconds * rate) as usize)
        .map(|i| (gain(level_db) * libm::sin(2.0 * PI * hz * i as f64 / rate)) as f32)
        .collect()
}

/// A repeatable noise (xorshift32) at `level_db` peak.
fn noise(level_db: f64, frames: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..frames)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((f64::from(s) / f64::from(u32::MAX) * 2.0 - 1.0) * gain(level_db)) as f32
        })
        .collect()
}

/// A plug-in at `rate` with `settings` (by knob name).
fn plugin(kind: Kind, rate: f32, settings: &[(&str, f32)]) -> Airwindows {
    let mut p = Airwindows::new(kind, rate);
    for (name, value) in settings {
        let index = kind
            .params()
            .iter()
            .position(|param| param.name == *name)
            .unwrap_or_else(|| unreachable!("{kind:?} has no knob {name:?}"));
        p.set_param(index, *value);
    }
    p
}

/// `input` in both channels through `p` in blocks of 128; the left out.
fn run(p: &mut Airwindows, input: &[f32]) -> Vec<f32> {
    let mut left = input.to_vec();
    let mut right = input.to_vec();
    for (l, r) in left.chunks_mut(128).zip(right.chunks_mut(128)) {
        p.process(l, r);
    }
    left
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / x.len() as f64).sqrt()
}

fn peak(x: &[f32]) -> f64 {
    x.iter().fold(0.0, |m: f64, &s| m.max(f64::from(s.abs())))
}

/// The amplitude of `hz` in `x`, Hann-windowed (Goertzel).
fn tone_level(x: &[f32], hz: f64, rate: f64) -> f64 {
    let n = x.len();
    let w = 2.0 * PI * hz / rate;
    let coeff = 2.0 * libm::cos(w);
    let (mut s1, mut s2) = (0.0, 0.0);
    let mut window_sum = 0.0;
    for (i, &v) in x.iter().enumerate() {
        let window = 0.5 - 0.5 * libm::cos(2.0 * PI * i as f64 / n as f64);
        window_sum += window;
        let s0 = f64::from(v) * window + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    2.0 * power.sqrt() / window_sum
}

/// The gain at `hz`, in dB: a -20 dBFS sine in, its level out over the
/// second half of half a second.
fn gain_at(p: &mut Airwindows, hz: f64) -> f64 {
    let rate = f64::from(p.sample_rate());
    p.clear();
    let input = sine(hz, -20.0, rate, 0.5);
    let out = run(p, &input);
    let half = input.len() / 2;
    db(rms(&out[half..]) / rms(&input[half..]))
}

/// How long after `burst` ends the output stays above `threshold_db`, in
/// seconds: the end of the last 10 ms window whose peak crosses it.
fn ring(p: &mut Airwindows, burst: &[f32], seconds: f64, threshold_db: f64) -> f64 {
    let rate = f64::from(p.sample_rate());
    run(p, burst);
    let after = run(p, &vec![0.0; (seconds * rate) as usize]);
    let window = (rate * 0.01) as usize;
    let last = after
        .chunks(window)
        .rposition(|w| db(peak(w)) > threshold_db)
        .map_or(0, |i| i + 1);
    (last * window) as f64 / rate
}

fn check(text: &str, expect: &Expect) {
    expect.assert_eq(text);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn every_kind_names_its_knobs_and_type() {
    let mut out = String::new();
    for kind in Kind::ALL {
        let p = Airwindows::new(kind, 44100.0);
        let _ = writeln!(
            out,
            "{} ({}), tail {}s at its defaults",
            kind.name(),
            kind.category(),
            p.tail_seconds()
        );
        for (i, param) in kind.params().iter().enumerate() {
            assert_eq!(p.get_param(i), param.default, "{kind:?} {}", param.name);
            let _ = writeln!(out, "  {} = {}", param.name, param.default);
        }
    }
    check(
        &out,
        &expect![[r#"
        Console7Channel (Distortion), tail 0s at its defaults
          Fader = 0.772
        Console7Buss (Distortion), tail 0s at its defaults
          Master = 1
        ToTape8 (Distortion), tail 0.5s at its defaults
          Input = 0.5
          Tilt = 0.5
          Shape = 0.5
          Flutter = 0.5
          FlutSpd = 0.5
          Bias = 0.5
          HeadBmp = 0.5
          HeadFrq = 0.5
          Output = 0.5
        Density2 (Distortion), tail 0s at its defaults
          Density = 0.2
          Highpass = 0
          Output = 1
          Dry/Wet = 1
        Galactic3 (Reverb), tail 27.519913s at its defaults
          Replace = 0.5
          Brightns = 0.5
          Detune = 0.5
          Derez = 1
          Bigness = 1
          Dry/Wet = 1
        Baxandall2 (EQ), tail 0.05s at its defaults
          Treble = 0.5
          Bass = 0.5
    "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn knobs_are_pinned_to_their_range_and_unknown_ones_ignored() {
    let mut p = Airwindows::new(Kind::Baxandall2, 48000.0);
    let mut out = String::new();
    for value in [-0.5, 0.0, 0.25, 1.0, 3.0, f32::NAN] {
        p.set_param(0, value);
        let _ = writeln!(out, "Treble set to {value} reads {}", p.get_param(0));
    }
    p.set_param(2, 0.7);
    let _ = writeln!(
        out,
        "knob 2 (there is none) set to 0.7 reads {}; Bass still {}",
        p.get_param(2),
        p.get_param(1)
    );
    check(
        &out,
        &expect![[r#"
        Treble set to -0.5 reads 0
        Treble set to 0 reads 0
        Treble set to 0.25 reads 0.25
        Treble set to 1 reads 1
        Treble set to 3 reads 1
        Treble set to NaN reads NaN
        knob 2 (there is none) set to 0.7 reads 0; Bass still 0.5
    "#]],
    );
}

/// A cleared plug-in is a new one with the same knobs and rate: the same
/// output, bit for bit, dither included.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn clear_is_a_fresh_instance() {
    let input = noise(-6.0, 20000, 7);
    for kind in Kind::ALL {
        let mut used = Airwindows::new(kind, 44100.0);
        used.set_param(0, 0.8);
        used.set_sample_rate(96000.0);
        run(&mut used, &input);
        used.clear();
        let mut fresh = Airwindows::new(kind, 96000.0);
        fresh.set_param(0, 0.8);
        let ours = run(&mut used, &input);
        let theirs = run(&mut fresh, &input);
        assert!(
            ours.iter()
                .zip(&theirs)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{kind:?}"
        );
        assert_eq!(used.sample_rate(), 96000.0, "{kind:?}");
        assert_eq!(used.get_param(0), 0.8, "{kind:?}");
    }
}

/// Silence in: what comes out is the denormal guard's and the dither's
/// noise, far below anything audible — except that Density2 at full
/// Density saturates the guard's noise (about -150 dBFS) fifteen times
/// over, each pass a gain of pi/2 on something that small.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn silence_stays_silent() {
    let mut out = String::new();
    for kind in Kind::ALL {
        let mut p = Airwindows::new(kind, 48000.0);
        let output = run(&mut p, &vec![0.0; 48000]);
        let _ = writeln!(
            out,
            "{:<16} peak {:.0} dBFS",
            kind.name(),
            db(peak(&output))
        );
    }
    let mut p = plugin(Kind::Density2, 48000.0, &[("Density", 1.0)]);
    let output = run(&mut p, &vec![0.0; 48000]);
    let _ = writeln!(
        out,
        "Density2 at Density 1: peak {:.0} dBFS",
        db(peak(&output))
    );
    check(
        &out,
        &expect![[r#"
        Console7Channel  peak -143 dBFS
        Console7Buss     peak -145 dBFS
        ToTape8          peak -147 dBFS
        Density2         peak -146 dBFS
        Galactic3        peak -145 dBFS
        Baxandall2       peak -145 dBFS
        Density2 at Density 1: peak -83 dBFS
    "#]],
    );
}

/// Baxandall2's shelves: Treble and Bass each ±24 dB across their knob,
/// the other band left alone, and 0.5 (0 dB) about flat.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn baxandall2_shelves() {
    let mut out = String::from("knob    |   50 Hz  1 kHz  10 kHz\n");
    for (name, other) in [("Treble", "Bass"), ("Bass", "Treble")] {
        for value in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut p = plugin(Kind::Baxandall2, 48000.0, &[(name, value), (other, 0.5)]);
            let _ = writeln!(
                out,
                "{name:<6} {value:<4} | {:>+6.1} {:>+6.1} {:>+6.1}",
                gain_at(&mut p, 50.0),
                gain_at(&mut p, 1000.0),
                gain_at(&mut p, 10000.0)
            );
        }
    }
    check(
        &out,
        &expect![[r#"
        knob    |   50 Hz  1 kHz  10 kHz
        Treble 0    |   +0.2   -2.3  -26.3
        Treble 0.25 |   +0.1   -0.9  -12.6
        Treble 0.5  |   +0.0   +0.6   -0.1
        Treble 0.75 |   -0.0   -4.9  +12.8
        Treble 1    |   -0.0   -4.3  +23.5
        Bass   0    |  -21.4   -1.3   +0.1
        Bass   0.25 |  -11.7   -0.0   +0.2
        Bass   0.5  |   +0.0   +0.6   -0.1
        Bass   0.75 |  +11.8   -3.4   +0.0
        Bass   1    |  +21.6  -15.5   +0.0
    "#]],
    );
}

/// Density2 on a 1 kHz sine at -6 dBFS: the harmonics it adds grow with
/// Density (0.2 is its zero: no saturation at all).
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn density2_harmonics_grow_with_density() {
    let rate = 48000.0;
    let input = sine(1000.0, -6.0, rate, 0.5);
    let mut out = String::from("Density | out rms | H2 dBc  H3 dBc  H5 dBc\n");
    for density in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let mut p = plugin(Kind::Density2, rate as f32, &[("Density", density)]);
        let output = run(&mut p, &input);
        let tail = &output[output.len() / 2..];
        let fundamental = tone_level(tail, 1000.0, rate);
        let h = |n: f64| db(tone_level(tail, 1000.0 * n, rate) / fundamental).max(-150.0);
        let _ = writeln!(
            out,
            "{density:<7} | {:>+7.1} | {:>6.1}  {:>6.1}  {:>6.1}",
            db(rms(tail)),
            h(2.0),
            h(3.0),
            h(5.0)
        );
    }
    check(
        &out,
        &expect![[r#"
        Density | out rms | H2 dBc  H3 dBc  H5 dBc
        0       |   -14.9 | -150.0   -14.3   -29.9
        0.2     |    -9.0 | -150.0  -150.0  -150.0
        0.4     |    -5.7 | -150.0   -31.4   -73.6
        0.6     |    -0.9 | -150.0   -12.1   -21.4
        0.8     |    -0.2 | -150.0    -9.6   -14.3
        1       |    -0.2 | -150.0    -9.6   -14.3
    "#]],
    );
}

/// ToTape8 at its defaults on a 1 kHz sine: unity-ish when quiet, the gain
/// falling as the level climbs — tape compresses — and the peak held
/// under full scale by ClipOnly2.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn to_tape8_compresses_hot_input() {
    let rate = 48000.0;
    let mut out = String::from("in dBFS | out rms  gain | out peak  H3 dBc\n");
    for level in [-24.0, -12.0, -6.0, 0.0, 6.0, 12.0] {
        let mut p = Airwindows::new(Kind::ToTape8, rate as f32);
        let input = sine(1000.0, level, rate, 0.5);
        let output = run(&mut p, &input);
        let half = input.len() / 2;
        let (i, o) = (&input[half..], &output[half..]);
        let _ = writeln!(
            out,
            "{level:>+7} | {:>+7.1} {:>+5.1} | {:>+8.1} {:>7.1}",
            db(rms(o)),
            db(rms(o) / rms(i)),
            db(peak(o)),
            db(tone_level(o, 3000.0, rate) / tone_level(o, 1000.0, rate))
        );
    }
    check(
        &out,
        &expect![[r#"
        in dBFS | out rms  gain | out peak  H3 dBc
            -24 |   -27.0  -0.0 |    -24.0   -63.0
            -12 |   -15.1  -0.1 |    -12.1   -47.1
             -6 |    -9.3  -0.3 |     -6.3   -37.5
             +0 |    -4.1  -1.0 |     -1.4   -27.1
             +6 |    -1.9  -4.9 |     -0.4   -16.1
            +12 |    -1.5 -10.5 |     -0.4   -12.7
    "#]],
    );
}

/// Console7Channel's fader on a full-scale 1 kHz sine. Below 0.772 the
/// fader is a gain of `(Fader * 1.272)^2` and clean; 0.772 is what the
/// original calls neutral for a full-scale sine, peaks rounded off by a
/// dB and a half; above it the sine squashes harder — more third
/// harmonic, no louder.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn console7_channel_fader() {
    let rate = 48000.0;
    let input = sine(1000.0, 0.0, rate, 0.5);
    let mut out = String::from("Fader | out peak  out rms | H3 dBc\n");
    for fader in [0.25, 0.5, 0.772, 1.0] {
        let mut p = plugin(Kind::Console7Channel, rate as f32, &[("Fader", fader)]);
        let output = run(&mut p, &input);
        let tail = &output[output.len() / 2..];
        let _ = writeln!(
            out,
            "{fader:<5} | {:>+8.2} {:>+8.2} | {:>6.1}",
            db(peak(tail)),
            db(rms(tail)),
            db(tone_level(tail, 3000.0, rate) / tone_level(tail, 1000.0, rate))
        );
    }
    check(
        &out,
        &expect![[r#"
        Fader | out peak  out rms | H3 dBc
        0.25  |   -19.90   -22.91 | -101.2
        0.5   |    -7.88   -10.89 |  -62.7
        0.772 |    -1.54    -4.10 |  -27.3
        1     |    -3.41    -4.35 |  -12.1
    "#]],
    );
}

/// Console7Buss on a 1 kHz sine at rising levels: the arcsine expands
/// what the channels' sine curve squashed, so the gain rises with level.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn console7_buss_expands() {
    let rate = 48000.0;
    let mut out = String::from("in dBFS | gain | H3 dBc\n");
    for level in [-24.0, -12.0, -6.0, -3.0, 0.0] {
        let mut p = Airwindows::new(Kind::Console7Buss, rate as f32);
        let input = sine(1000.0, level, rate, 0.5);
        let output = run(&mut p, &input);
        let half = input.len() / 2;
        let (i, o) = (&input[half..], &output[half..]);
        let _ = writeln!(
            out,
            "{level:>+7} | {:>+4.1} | {:>6.1}",
            db(rms(o) / rms(i)),
            db(tone_level(o, 3000.0, rate) / tone_level(o, 1000.0, rate))
        );
    }
    check(
        &out,
        &expect![[r#"
        in dBFS | gain | H3 dBc
            -24 | +0.3 |  -83.6
            -12 | +0.3 |  -58.4
             -6 | +0.4 |  -43.2
             -3 | +0.7 |  -33.7
             +0 | +2.5 |  -17.6
    "#]],
    );
}

/// Galactic3's tail after a 50 ms noise burst: how long it stays above
/// -60 and -100 dBFS at each Replace, against what `tail_seconds` claims.
/// Replace 0 holds what it has: loaded at 0.5 and then set to 0, it is as
/// loud ten seconds on.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-airwindows -- --include-ignored"]
fn galactic3_tail() {
    let rate = 44100.0f32;
    let burst = noise(-6.0, 2205, 3);
    let mut out = String::from("Replace | above -60 | above -100 | tail_seconds\n");
    for replace in [1.0, 0.75, 0.5, 0.25] {
        let mut p = plugin(Kind::Galactic3, rate, &[("Replace", replace)]);
        let claimed = p.tail_seconds();
        let sixty = ring(&mut p, &burst, 60.0, -60.0);
        p.clear();
        let hundred = ring(&mut p, &burst, 60.0, -100.0);
        assert!(
            hundred <= f64::from(claimed),
            "Replace {replace}: rings {hundred}s, claims {claimed}s"
        );
        let _ = writeln!(
            out,
            "{replace:<7} | {sixty:>8.2}s | {hundred:>9.2}s | {claimed:.2}s"
        );
    }
    let mut p = plugin(Kind::Galactic3, rate, &[("Replace", 0.5)]);
    run(&mut p, &burst);
    run(&mut p, &vec![0.0; 22050]);
    p.set_param(0, 0.0);
    let early = run(&mut p, &vec![0.0; 44100]);
    run(&mut p, &vec![0.0; 44100 * 8]);
    let late = run(&mut p, &vec![0.0; 44100]);
    let _ = writeln!(
        out,
        "frozen  | rms {:.1} dBFS a second in, {:.1} ten seconds in | {}",
        db(rms(&early)),
        db(rms(&late)),
        p.tail_seconds()
    );
    check(
        &out,
        &expect![[r#"
        Replace | above -60 | above -100 | tail_seconds
        1       |     5.64s |     11.20s | 11.82s
        0.75    |     7.38s |     15.36s | 17.11s
        0.5     |    10.61s |     23.51s | 27.52s
        0.25    |    18.49s |     45.86s | 58.50s
        frozen  | rms -37.7 dBFS a second in, -36.2 ten seconds in | inf
    "#]],
    );
}

/// How long the rest ring above -90 dBFS after a hot burst stops: no time
/// at all for the filters and saturators, longer for ToTape8's head bump.
/// Each within its `tail_seconds`, the ones that claim 0 within 20 ms.
/// (Not -100, and Density2 not at full Density: there it lifts the
/// denormal guard's noise to -83 dBFS, input or no input — see
/// `silence_stays_silent`.)
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn tails_of_the_rest() {
    let rate = 48000.0f32;
    let burst = noise(6.0, 4800, 11);
    let mut out = String::new();
    let cases: [(Kind, &[(&str, f32)]); 9] = [
        (Kind::Console7Channel, &[]),
        (Kind::Console7Buss, &[]),
        (Kind::Density2, &[("Density", 0.8)]),
        (Kind::Baxandall2, &[("Bass", 1.0)]),
        (Kind::ToTape8, &[]),
        (Kind::ToTape8, &[("HeadBmp", 1.0), ("HeadFrq", 0.0)]),
        (
            Kind::ToTape8,
            &[
                ("HeadBmp", 1.0),
                ("Input", 1.0),
                ("HeadFrq", 0.0),
                ("Shape", 0.0),
            ],
        ),
        (Kind::ToTape8, &[("Flutter", 1.0), ("FlutSpd", 0.0)]),
        (Kind::ToTape8, &[("Bias", 0.0), ("Output", 1.0)]),
    ];
    for (kind, settings) in cases {
        let mut p = plugin(kind, rate, settings);
        let claimed = f64::from(p.tail_seconds()).max(0.02);
        let rings = ring(&mut p, &burst, 5.0, -90.0);
        assert!(rings <= claimed, "{kind:?} {settings:?}: rings {rings}s");
        let _ = writeln!(
            out,
            "{:<16} {settings:?}: {rings:.2}s, claims {}s",
            kind.name(),
            p.tail_seconds()
        );
    }
    check(
        &out,
        &expect![[r#"
        Console7Channel  []: 0.01s, claims 0s
        Console7Buss     []: 0.01s, claims 0s
        Density2         [("Density", 0.8)]: 0.01s, claims 0s
        Baxandall2       [("Bass", 1.0)]: 0.03s, claims 0.05s
        ToTape8          []: 0.09s, claims 0.5s
        ToTape8          [("HeadBmp", 1.0), ("HeadFrq", 0.0)]: 0.22s, claims 0.5s
        ToTape8          [("HeadBmp", 1.0), ("Input", 1.0), ("HeadFrq", 0.0), ("Shape", 0.0)]: 0.33s, claims 0.5s
        ToTape8          [("Flutter", 1.0), ("FlutSpd", 0.0)]: 0.09s, claims 0.5s
        ToTape8          [("Bias", 0.0), ("Output", 1.0)]: 0.19s, claims 0.5s
    "#]],
    );
}
