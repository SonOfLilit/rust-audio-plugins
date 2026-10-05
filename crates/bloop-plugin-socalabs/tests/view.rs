//! What the effects' screens draw, against what the effects do: the
//! GraphicEQ's curve against a tone measured through the EQ, a dynamics
//! processor's transfer curve against a tone measured through it, and the
//! meters against a tone of known level.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use std::fmt::Write as _;

use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_socalabs::view::{EqCurve, Transfer};
use bloop_plugin_socalabs::{Effect, Kind};
use expect_test::expect;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

const RATE: f32 = 48_000.0;
const BLOCK: usize = 128;

fn process(effect: &mut Effect, left: &mut [f32], right: &mut [f32]) {
    let events = bloop_events::EventBuffer::new();
    let mut outputs: [&mut [f32]; 2] = [left, right];
    let mut context = ProcessContext {
        outputs: &mut outputs,
        input_events: &events,
        sample_rate: RATE,
        tempo: 120.0,
        is_playing: true,
    };
    effect.process(&mut context);
}

fn db(gain: f32) -> f32 {
    20.0 * libm::log10f(gain)
}

/// A sine of `freq` at `amplitude` through `effect` for `seconds`, block by
/// block; the peak of its last `tail` seconds in and out.
fn tone(effect: &mut Effect, freq: f32, amplitude: f32, seconds: f32, tail: f32) -> (f32, f32) {
    let (peaks, _) = tone_levels(effect, freq, amplitude, seconds, tail);
    peaks
}

/// What a stretch read in and out.
type Levels = (f32, f32);

/// [`tone`], with the RMS of the same stretch in and out besides: a peak
/// read off samples misses a high tone's crests by as much as its phase
/// says, and the EQ turns the phase.
fn tone_levels(
    effect: &mut Effect,
    freq: f32,
    amplitude: f32,
    seconds: f32,
    tail: f32,
) -> (Levels, Levels) {
    let total = (seconds * RATE) as usize;
    let from = total - (tail * RATE) as usize;
    let (mut peak_in, mut peak_out) = (0.0f32, 0.0f32);
    let (mut sum_in, mut sum_out, mut count) = (0.0f64, 0.0f64, 0usize);
    let mut at = 0;
    while at < total {
        let n = BLOCK.min(total - at);
        let mut left = [0.0f32; BLOCK];
        for (i, x) in left[..n].iter_mut().enumerate() {
            let t = (at + i) as f64 / f64::from(RATE);
            *x = amplitude * libm::sin(2.0 * std::f64::consts::PI * f64::from(freq) * t) as f32;
        }
        let input = left;
        let mut right = left;
        process(effect, &mut left[..n], &mut right[..n]);
        for i in 0..n {
            if at + i >= from {
                peak_in = peak_in.max(input[i].abs());
                peak_out = peak_out.max(left[i].abs());
                sum_in += f64::from(input[i]).powi(2);
                sum_out += f64::from(left[i]).powi(2);
                count += 1;
            }
        }
        at += n;
    }
    let rms = |sum: f64| (sum / count.max(1) as f64).sqrt() as f32;
    ((peak_in, peak_out), (rms(sum_in), rms(sum_out)))
}

fn with_knobs(kind: Kind, knobs: &[(usize, f32)]) -> (Effect, Vec<f32>) {
    let mut effect = Effect::new(kind, RATE);
    for &(index, value) in knobs {
        effect.set_param(index, value);
    }
    let values = (0..effect.param_count())
        .map(|i| effect.get_param(i))
        .collect();
    (effect, values)
}

/// The GraphicEQ's curve is what the EQ does to a tone: for each setting,
/// at each band's centre and between them, the curve's gain and a sine's
/// measured through a fresh EQ (its RMS over the last half second of
/// two), and the largest gap between the two. Settings: flat; the bands
/// alternating ±12 dB; 1 kHz alone at +12; the 24 dB highpass at its
/// lowest note above 200 Hz with the 12 dB lowpass near 2 kHz; and Output
/// at −6 dB.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[expect(
    clippy::too_many_lines,
    reason = "the settings are the scenario, written out"
)]
fn the_eq_curve_is_what_the_eq_does() {
    // HP Freq and LP Freq are MIDI notes: 55 is 196 Hz, 95 is 1976 Hz.
    let settings: [(&str, Vec<(usize, f32)>); 5] = [
        ("flat", vec![]),
        (
            "alternating ±12",
            (0..10)
                .map(|b| (b, if b % 2 == 0 { 12.0 } else { -12.0 }))
                .collect(),
        ),
        ("1 kHz +12", vec![(5, 12.0)]),
        (
            "HP 24 dB at 196 Hz, LP 12 dB at 1976 Hz",
            vec![(13, 1.0), (14, 55.0), (15, 1.0), (10, 1.0), (11, 95.0), (12, 0.0)],
        ),
        ("Output -6", vec![(16, -6.0)]),
    ];
    let freqs = [
        31.5, 45.0, 63.0, 125.0, 200.0, 250.0, 500.0, 700.0, 1000.0, 2000.0, 3000.0, 4000.0,
        8000.0, 16000.0,
    ];
    let mut out = String::new();
    for (name, knobs) in &settings {
        let (_, values) = with_knobs(Kind::GraphicEq, knobs);
        let curve = EqCurve::new(&values, RATE);
        let mut worst = 0.0f32;
        let _ = writeln!(out, "{name}:");
        for freq in freqs {
            let (mut effect, _) = with_knobs(Kind::GraphicEq, knobs);
            let (_, (rms_in, rms_out)) = tone_levels(&mut effect, freq, 0.25, 2.0, 0.5);
            let measured = db(rms_out / rms_in);
            let drawn = curve.db(freq);
            worst = worst.max((measured - drawn).abs());
            let _ = writeln!(
                out,
                "  {freq:>7} Hz  curve {drawn:>7.2} dB  measured {measured:>7.2} dB"
            );
        }
        let _ = writeln!(out, "  largest gap {worst:.3} dB");
    }
    expect![[r#"
        flat:
             31.5 Hz  curve    0.00 dB  measured    0.00 dB
               45 Hz  curve    0.00 dB  measured    0.00 dB
               63 Hz  curve    0.00 dB  measured    0.00 dB
              125 Hz  curve    0.00 dB  measured    0.00 dB
              200 Hz  curve    0.00 dB  measured    0.00 dB
              250 Hz  curve    0.00 dB  measured    0.00 dB
              500 Hz  curve    0.00 dB  measured    0.00 dB
              700 Hz  curve    0.00 dB  measured    0.00 dB
             1000 Hz  curve    0.00 dB  measured    0.00 dB
             2000 Hz  curve    0.00 dB  measured    0.00 dB
             3000 Hz  curve    0.00 dB  measured    0.00 dB
             4000 Hz  curve    0.00 dB  measured    0.00 dB
             8000 Hz  curve    0.00 dB  measured    0.00 dB
            16000 Hz  curve    0.00 dB  measured    0.00 dB
          largest gap 0.000 dB
        alternating ±12:
             31.5 Hz  curve   11.98 dB  measured   11.98 dB
               45 Hz  curve    4.88 dB  measured    4.88 dB
               63 Hz  curve   -5.03 dB  measured   -5.03 dB
              125 Hz  curve   11.94 dB  measured   11.94 dB
              200 Hz  curve    3.01 dB  measured    3.01 dB
              250 Hz  curve   -5.20 dB  measured   -5.20 dB
              500 Hz  curve   11.94 dB  measured   11.94 dB
              700 Hz  curve    5.71 dB  measured    5.71 dB
             1000 Hz  curve   -5.12 dB  measured   -5.12 dB
             2000 Hz  curve   11.94 dB  measured   11.94 dB
             3000 Hz  curve    5.10 dB  measured    5.10 dB
             4000 Hz  curve   -3.94 dB  measured   -3.94 dB
             8000 Hz  curve   11.85 dB  measured   11.85 dB
            16000 Hz  curve    2.36 dB  measured    2.36 dB
          largest gap 0.003 dB
        1 kHz +12:
             31.5 Hz  curve    0.01 dB  measured    0.00 dB
               45 Hz  curve    0.01 dB  measured    0.01 dB
               63 Hz  curve    0.03 dB  measured    0.03 dB
              125 Hz  curve    0.11 dB  measured    0.11 dB
              200 Hz  curve    0.30 dB  measured    0.30 dB
              250 Hz  curve    0.48 dB  measured    0.48 dB
              500 Hz  curve    2.34 dB  measured    2.34 dB
              700 Hz  curve    5.61 dB  measured    5.61 dB
             1000 Hz  curve   12.00 dB  measured   12.00 dB
             2000 Hz  curve    2.73 dB  measured    2.73 dB
             3000 Hz  curve    1.33 dB  measured    1.33 dB
             4000 Hz  curve    0.91 dB  measured    0.91 dB
             8000 Hz  curve    0.54 dB  measured    0.54 dB
            16000 Hz  curve    0.44 dB  measured    0.44 dB
          largest gap 0.003 dB
        HP 24 dB at 196 Hz, LP 12 dB at 1976 Hz:
             31.5 Hz  curve  -63.52 dB  measured  -63.49 dB
               45 Hz  curve  -51.15 dB  measured  -51.15 dB
               63 Hz  curve  -39.53 dB  measured  -39.53 dB
              125 Hz  curve  -16.96 dB  measured  -16.96 dB
              200 Hz  curve   -5.68 dB  measured   -5.68 dB
              250 Hz  curve   -2.79 dB  measured   -2.79 dB
              500 Hz  curve   -0.22 dB  measured   -0.22 dB
              700 Hz  curve   -0.12 dB  measured   -0.12 dB
             1000 Hz  curve   -0.29 dB  measured   -0.29 dB
             2000 Hz  curve   -3.12 dB  measured   -3.12 dB
             3000 Hz  curve   -8.01 dB  measured   -8.01 dB
             4000 Hz  curve  -12.50 dB  measured  -12.50 dB
             8000 Hz  curve  -24.31 dB  measured  -24.31 dB
            16000 Hz  curve  -36.34 dB  measured  -36.34 dB
          largest gap 0.036 dB
        Output -6:
             31.5 Hz  curve   -6.00 dB  measured   -6.00 dB
               45 Hz  curve   -6.00 dB  measured   -6.00 dB
               63 Hz  curve   -6.00 dB  measured   -6.00 dB
              125 Hz  curve   -6.00 dB  measured   -6.00 dB
              200 Hz  curve   -6.00 dB  measured   -6.00 dB
              250 Hz  curve   -6.00 dB  measured   -6.00 dB
              500 Hz  curve   -6.00 dB  measured   -6.00 dB
              700 Hz  curve   -6.00 dB  measured   -6.00 dB
             1000 Hz  curve   -6.00 dB  measured   -6.00 dB
             2000 Hz  curve   -6.00 dB  measured   -6.00 dB
             3000 Hz  curve   -6.00 dB  measured   -6.00 dB
             4000 Hz  curve   -6.00 dB  measured   -6.00 dB
             8000 Hz  curve   -6.00 dB  measured   -6.00 dB
            16000 Hz  curve   -6.00 dB  measured   -6.00 dB
          largest gap 0.000 dB
    "#]].assert_eq(&out);
}

/// A processor, named, and its knobs set.
type Case = (&'static str, Kind, Vec<(usize, f32)>);

/// Each dynamics processor's transfer curve, at levels in either side of
/// its threshold: the curve's level out, and a sine's measured through the
/// processor (its peak over the last 100 ms of half a second, Release at
/// its longest so the detector holds the sine's peak rather than riding
/// its cycles). Then where, scanning up from −60 dB in tenths, the curve
/// leaves the straight line (a compressor or limiter: the threshold, or
/// half the knee under it) or joins it (a gate or expander: the
/// threshold).
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_transfer_curve_is_what_the_processor_does() {
    // (name, kind, knobs): Compressor 4:1 at -30 dB, no knee and a 10 dB
    // knee; the Limiter at -12; the Gate at -40; the Expander 2:1 at -30.
    let cases: [Case; 5] = [
        (
            "Compressor 4:1 at -30, no knee",
            Kind::Compressor,
            vec![(1, 2000.0), (2, 4.0), (3, -30.0), (4, 0.0)],
        ),
        (
            "Compressor 4:1 at -30, 10 dB knee",
            Kind::Compressor,
            vec![(1, 2000.0), (2, 4.0), (3, -30.0), (4, 10.0)],
        ),
        (
            "Limiter at -12",
            Kind::Limiter,
            vec![(1, 500.0), (2, -12.0)],
        ),
        (
            "Gate at -40",
            Kind::Gate,
            vec![(2, 500.0), (3, -40.0)],
        ),
        (
            "Expander 2:1 at -30, no knee",
            Kind::Expander,
            vec![(1, 2000.0), (2, 2.0), (3, -30.0), (4, 0.0)],
        ),
    ];
    let mut out = String::new();
    for (name, kind, knobs) in &cases {
        let (_, values) = with_knobs(*kind, knobs);
        let transfer = Transfer::of(*kind, &values).expect("a dynamics processor");
        let _ = writeln!(out, "{name}: threshold {} dB", transfer.threshold_db);
        for level in [-50.0f32, -35.0, -30.0, -25.0, -12.0, -6.0] {
            let (mut effect, _) = with_knobs(*kind, knobs);
            let amplitude = libm::powf(10.0, level / 20.0);
            let (peak_in, peak_out) = tone(&mut effect, 1000.0, amplitude, 0.5, 0.1);
            let _ = writeln!(
                out,
                "  in {level:>5.1}  curve {:>8.2}  measured {:>8.2}",
                transfer.out_db(level),
                db(peak_out) - db(peak_in) + level
            );
        }
        let straight = |x: f32| (transfer.out_db(x) - x).abs() <= 0.01;
        let bends = (-600..=0)
            .map(|tenths| tenths as f32 / 10.0)
            .find(|&x| straight(x) != straight(-60.0));
        let _ = writeln!(
            out,
            "  {} the straight line at {bends:?}",
            if straight(-60.0) { "leaves" } else { "joins" }
        );
    }
    expect![[r#"
        Compressor 4:1 at -30, no knee: threshold -30 dB
          in -50.0  curve   -50.00  measured   -50.00
          in -35.0  curve   -35.00  measured   -35.00
          in -30.0  curve   -30.00  measured   -30.00
          in -25.0  curve   -28.75  measured   -28.72
          in -12.0  curve   -25.50  measured   -25.47
          in  -6.0  curve   -24.00  measured   -23.97
          leaves the straight line at Some(-29.9)
        Compressor 4:1 at -30, 10 dB knee: threshold -30 dB
          in -50.0  curve   -50.00  measured   -50.00
          in -35.0  curve   -35.00  measured   -35.00
          in -30.0  curve   -30.94  measured   -30.92
          in -25.0  curve   -28.75  measured   -28.72
          in -12.0  curve   -25.50  measured   -25.47
          in  -6.0  curve   -24.00  measured   -23.97
          leaves the straight line at Some(-34.4)
        Limiter at -12: threshold -12 dB
          in -50.0  curve   -50.00  measured   -50.00
          in -35.0  curve   -35.00  measured   -35.00
          in -30.0  curve   -30.00  measured   -30.00
          in -25.0  curve   -25.00  measured   -25.00
          in -12.0  curve   -12.00  measured   -12.00
          in  -6.0  curve   -12.00  measured   -12.00
          leaves the straight line at Some(-11.9)
        Gate at -40: threshold -40 dB
          in -50.0  curve -1000.00  measured     -inf
          in -35.0  curve   -35.00  measured   -35.00
          in -30.0  curve   -30.00  measured   -30.00
          in -25.0  curve   -25.00  measured   -25.00
          in -12.0  curve   -12.00  measured   -12.00
          in  -6.0  curve    -6.00  measured    -6.00
          joins the straight line at Some(-40.0)
        Expander 2:1 at -30, no knee: threshold -30 dB
          in -50.0  curve   -70.00  measured   -70.04
          in -35.0  curve   -40.00  measured   -40.04
          in -30.0  curve   -30.00  measured   -30.04
          in -25.0  curve   -25.00  measured   -25.00
          in -12.0  curve   -12.00  measured   -12.00
          in  -6.0  curve    -6.00  measured    -6.00
          joins the straight line at Some(-30.0)
    "#]].assert_eq(&out);
}

/// A sine at −12 dBFS through a Compressor at −30 dB, 4:1, no knee: the
/// meters read −12 in, −13.5 of reduction and −25.5 out once it has
/// settled, and after the tone stops each holds 50 ms and then moves 30 dB a
/// second — the levels falling, the reduction rising back to none. An
/// Input of +6 dB shows on the reduction and the output, not on the input
/// meter, which reads the signal before it. A Delay has no meter.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_meters_read_a_tone_of_known_level() {
    let amplitude = libm::powf(10.0, -12.0 / 20.0);
    let knobs = [(1, 2000.0), (2, 4.0), (3, -30.0), (4, 0.0)];
    let say = |effect: &Effect, what: &str| {
        let m = effect.meter().expect("a Compressor has meters");
        format!(
            "{what:<28} in {:>7.2}  reduction {:>7.2}  out {:>7.2}  bands {}",
            m.input_db, m.reduction_db[0], m.output_db, m.bands
        )
    };
    let (mut effect, _) = with_knobs(Kind::Compressor, &knobs);
    let mut out = vec![say(&effect, "new")];
    tone(&mut effect, 1000.0, amplitude, 0.5, 0.1);
    out.push(say(&effect, "-12 dBFS for 0.5 s"));
    for (what, seconds) in [("then silence, 40 ms", 0.04), ("then 100 ms more", 0.1)] {
        tone(&mut effect, 1000.0, 0.0, seconds, 0.0);
        out.push(say(&effect, what));
    }
    let (mut louder, _) = with_knobs(Kind::Compressor, &[knobs.as_slice(), &[(5, 6.0)]].concat());
    tone(&mut louder, 1000.0, amplitude, 0.5, 0.1);
    out.push(say(&louder, "Input +6, -12 dBFS, 0.5 s"));
    out.push(format!(
        "a Delay: {:?}",
        Effect::new(Kind::Delay, RATE).meter()
    ));
    expect![[r#"
        new                          in -100.00  reduction    0.00  out -100.00  bands 1
        -12 dBFS for 0.5 s           in  -12.00  reduction  -13.47  out  -25.47  bands 1
        then silence, 40 ms          in  -13.02  reduction  -13.33  out  -26.17  bands 1
        then 100 ms more             in  -16.02  reduction  -12.85  out  -29.17  bands 1
        Input +6, -12 dBFS, 0.5 s    in  -12.00  reduction  -17.97  out  -23.97  bands 1
        a Delay: None"#]].assert_eq(&out.join("\n"));
}
