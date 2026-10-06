//! What the Maximizer and the TransientShaper trace for their screens, and
//! the Maximizer's meters, against tones of known level: two numbers a
//! 64-sample chunk, folded across blocks of any size, which change nothing
//! that comes out.

#![expect(
    clippy::cast_precision_loss,
    reason = "sample counts of a second or two, far inside f32"
)]
#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_socalabs::{Effect, Kind};
use expect_test::expect;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

const RATE: f32 = 48_000.0;
/// The points after a knob's 0.1 s glide is over, which each block reads
/// once and so ends on a block's edge: 0.2 s.
const SETTLED: usize = 150;

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

fn effect_release_zero(effect: &mut Effect) {
    effect.set_param(2, 0.0);
}

fn db(gain: f32) -> f32 {
    20.0 * libm::log10f(gain.max(1e-6))
}

/// `samples` of a 1 kHz sine at `level_db` dBFS from sample `from`, or
/// silence where `level_db` is `None`.
fn tone(level_db: Option<f32>, from: usize, samples: usize) -> Vec<f32> {
    (from..from + samples)
        .map(|n| {
            level_db.map_or(0.0, |level| {
                libm::powf(10.0, level / 20.0)
                    * libm::sinf(2.0 * core::f32::consts::PI * 1000.0 * n as f32 / RATE)
            })
        })
        .collect()
}

/// `input` through `effect` in blocks of `block`, and every point it
/// traced, in order, with how many each block made.
fn run(effect: &mut Effect, input: &[f32], block: usize) -> (Vec<[f32; 2]>, Vec<usize>) {
    let mut points = Vec::new();
    let mut counts = Vec::new();
    for chunk in input.chunks(block) {
        let (mut left, mut right) = (chunk.to_vec(), chunk.to_vec());
        process(effect, &mut left, &mut right);
        points.extend_from_slice(effect.trace());
        counts.push(effect.trace().len());
    }
    (points, counts)
}

/// The Maximizer at Threshold −12 dB, Ceiling −1 dB, no Release: a 1 kHz
/// sine at −6 dBFS has every half cycle scaled back to the threshold, so
/// each point reads the input's peak (−6 dB) and a gain of −6 dB, and the
/// meters read −6 in, the ceiling out and 6 dB of reduction. A sine at
/// −20 dBFS, under the threshold, is let through at unity: the gain reads
/// 0 dB at once, the reduction meter rises back 30 dB a second after its
/// hold. In blocks of 100 a point falls every 64 samples whatever the
/// block, the chunk across two blocks folded across them, the same points
/// as in blocks of 128 (at a Threshold that turns it down) once the
/// Threshold's glide to it is over — a knob glides a block at a time. What comes out
/// does not depend on the block either; that tracing changes nothing that
/// comes out is the gold harness's to prove (`--features gold`).
///
/// A 100 Hz sine, a half cycle 240 samples long, at Release 0 and in
/// blocks of 32: most chunks hold no zero crossing, and each still reads
/// the 6 dB its half cycle was scaled by, and the −6 dB it went in at.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_maximizer_traces_its_input_and_gain() {
    let mut effect = Effect::new(Kind::Maximizer, RATE);
    for (param, value) in [(0, -12.0), (1, -1.0), (2, 0.0)] {
        effect.set_param(param, value);
    }
    let mut out = Vec::new();
    let mut say = |what: &str, effect: &Effect, points: &[[f32; 2]]| {
        let last = &points[points.len().saturating_sub(16)..];
        let peak = last.iter().fold(0.0f32, |m, p| m.max(p[0]));
        let gain = last.iter().fold(1.0f32, |m, p| m.min(p[1]));
        let meter = effect.meter().map_or("no meter".to_string(), |m| {
            format!(
                "meters in {:.2}, out {:.2}, GR {:.2} dB",
                m.input_db, m.output_db, m.reduction_db[0]
            )
        });
        out.push(format!(
            "{what:<34} last 16 points: input peak {:.2} dB, least gain {:.2} dB; {meter}",
            db(peak),
            db(gain)
        ));
    };
    let loud = tone(Some(-6.0), 0, 24_000);
    let (points, _) = run(&mut effect, &loud, 128);
    say("-6 dBFS, half a second", &effect, &points);
    let quiet = tone(Some(-20.0), 24_000, 4800);
    let (points, _) = run(&mut effect, &quiet, 128);
    say("then -20 dBFS, 100 ms", &effect, &points);
    let quiet = tone(Some(-20.0), 28_800, 9600);
    let (points, _) = run(&mut effect, &quiet, 128);
    say("-20 dBFS, 300 ms in", &effect, &points);
    out.push(format!(
        "{} points from {} samples of 128-sample blocks",
        points.len(),
        quiet.len()
    ));

    let input = tone(Some(-6.0), 0, 24_000);
    let at_12 = || {
        let mut effect = Effect::new(Kind::Maximizer, RATE);
        effect.set_param(0, -12.0);
        effect
    };
    let mut by_100 = at_12();
    let mut by_128 = at_12();
    let (points_100, counts) = run(&mut by_100, &input, 100);
    let (points_128, _) = run(&mut by_128, &input, 128);
    out.push(format!(
        "blocks of 100: points a block {:?}…, {} in all; blocks of 128: {} in all; the same points once the Threshold has settled: {}",
        &counts[..6],
        points_100.len(),
        points_128.len(),
        points_100[SETTLED..] == points_128[SETTLED..]
    ));
    let (mut a, mut b) = (
        Effect::new(Kind::Maximizer, RATE),
        Effect::new(Kind::Maximizer, RATE),
    );
    let mut out_a = Vec::new();
    for chunk in input.chunks(100) {
        let (mut l, mut r) = (chunk.to_vec(), chunk.to_vec());
        process(&mut a, &mut l, &mut r);
        out_a.extend(l);
    }
    let mut out_b = Vec::new();
    for chunk in input.chunks(128) {
        let (mut l, mut r) = (chunk.to_vec(), chunk.to_vec());
        process(&mut b, &mut l, &mut r);
        out_b.extend(l);
    }
    out.push(format!(
        "the audio out the same in either: {}",
        out_a == out_b
    ));
    let mut low = at_12();
    effect_release_zero(&mut low);
    let bass: Vec<f32> = (0..24_000)
        .map(|n| 0.501 * libm::sinf(2.0 * core::f32::consts::PI * 100.0 * n as f32 / RATE))
        .collect();
    let (points, _) = run(&mut low, &bass, 32);
    let settled = &points[points.len() / 2..];
    let (lo_gain, hi_gain) = settled.iter().fold((f32::INFINITY, 0.0f32), |(lo, hi), p| {
        (lo.min(p[1]), hi.max(p[1]))
    });
    let (lo_peak, hi_peak) = settled.iter().fold((f32::INFINITY, 0.0f32), |(lo, hi), p| {
        (lo.min(p[0]), hi.max(p[0]))
    });
    out.push(format!(
        "100 Hz in blocks of 32, the settled half's {} points: gain {:.2} to {:.2} dB, peak {:.2} to {:.2} dB",
        settled.len(),
        db(lo_gain),
        db(hi_gain),
        db(lo_peak),
        db(hi_peak)
    ));
    expect![[r#"
        -6 dBFS, half a second             last 16 points: input peak -6.00 dB, least gain -6.00 dB; meters in -6.00, out -1.00, GR -6.00 dB
        then -20 dBFS, 100 ms              last 16 points: input peak -20.00 dB, least gain 0.00 dB; meters in -8.74, out -2.26, GR -5.54 dB
        -20 dBFS, 300 ms in                last 16 points: input peak -20.00 dB, least gain 0.00 dB; meters in -14.74, out -8.26, GR 0.00 dB
        150 points from 9600 samples of 128-sample blocks
        blocks of 100: points a block [1, 2, 1, 2, 1, 2]…, 375 in all; blocks of 128: 375 in all; the same points once the Threshold has settled: true
        the audio out the same in either: true
        100 Hz in blocks of 32, the settled half's 188 points: gain -6.00 to -6.00 dB, peak -13.82 to -6.00 dB"#]].assert_eq(&out.join("\n"));
}

/// The TransientShaper with Attack at +2 and Sustain at −2, Smoothing 10%:
/// 200 ms of silence, 300 ms of a 1 kHz sine at −6 dBFS, 500 ms of silence,
/// each 50 ms summed up as the screen draws it — the envelope's peak, the
/// gain at its least and most. Silence reads the detector's floor and unity
/// gain, after the first few milliseconds' swell up from nothing that
/// Gin's reset leaves; the attack turns the gain up as the tone starts,
/// the steady tone brings it back, and the tail after it is turned down
/// far. The envelope, the follower of the rectified mid at half weight,
/// settles at half the sine's mean, 0.5 × 0.501 × 2/π: about −16 dB. A Compressor traces nothing, and the
/// TransientShaper keeps no meter.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_transient_shaper_traces_its_envelope_and_gain() {
    let mut effect = Effect::new(Kind::TransientShaper, RATE);
    for (param, value) in [(0, 2.0), (1, -2.0), (2, 10.0)] {
        effect.set_param(param, value);
    }
    let mut input = tone(None, 0, 9600);
    input.extend(tone(Some(-6.0), 9600, 14_400));
    input.extend(tone(None, 24_000, 24_000));
    let (points, _) = run(&mut effect, &input, 128);
    // 50 ms is 37.5 points: windows of 37 and 38 in turn would be honest,
    // but 2400 samples are 37.5 chunks — sum each window's points by the
    // sample it ends on instead.
    let mut out = Vec::new();
    for window in 0..20usize {
        let (from, to) = (window * 2400, (window + 1) * 2400);
        let inside: Vec<&[f32; 2]> = points
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let end = (i + 1) * 64;
                end > from && end <= to
            })
            .map(|(_, p)| p)
            .collect();
        let env = inside.iter().fold(0.0f32, |m, p| m.max(p[0]));
        let least = inside.iter().fold(f32::INFINITY, |m, p| m.min(p[1]));
        let most = inside.iter().fold(0.0f32, |m, p| m.max(p[1]));
        out.push(format!(
            "{:>4}–{:>4} ms: envelope {:>7.2} dB, gain {:>+6.2} to {:>+6.2} dB",
            from / 48,
            to / 48,
            db(env),
            db(least),
            db(most)
        ));
    }
    let mut compressor = Effect::new(Kind::Compressor, RATE);
    let (traced, _) = run(&mut compressor, &input[..4800], 128);
    out.push(format!(
        "a Compressor traces {} points; the TransientShaper's meter: {:?}",
        traced.len(),
        effect.meter()
    ));
    expect![[r#"
           0–  50 ms: envelope -100.40 dB, gain -14.40 to  +0.53 dB
          50– 100 ms: envelope -100.02 dB, gain  +0.00 to  +0.10 dB
         100– 150 ms: envelope -100.00 dB, gain  +0.00 to  +0.00 dB
         150– 200 ms: envelope -100.00 dB, gain  +0.00 to  +0.00 dB
         200– 250 ms: envelope  -16.33 dB, gain  +0.15 to  +4.81 dB
         250– 300 ms: envelope  -15.94 dB, gain  +0.05 to  +0.14 dB
         300– 350 ms: envelope  -15.93 dB, gain  +0.05 to  +0.05 dB
         350– 400 ms: envelope  -15.93 dB, gain  +0.05 to  +0.05 dB
         400– 450 ms: envelope  -15.93 dB, gain  +0.05 to  +0.05 dB
         450– 500 ms: envelope  -15.93 dB, gain  +0.05 to  +0.05 dB
         500– 550 ms: envelope  -15.96 dB, gain -11.39 to  +0.00 dB
         550– 600 ms: envelope  -42.87 dB, gain -23.70 to -11.72 dB
         600– 650 ms: envelope  -70.25 dB, gain -33.54 to -24.01 dB
         650– 700 ms: envelope  -92.62 dB, gain -35.69 to -33.71 dB
         700– 750 ms: envelope  -99.53 dB, gain -35.61 to -34.53 dB
         750– 800 ms: envelope  -99.98 dB, gain -34.49 to -33.16 dB
         800– 850 ms: envelope -100.00 dB, gain -33.12 to -31.81 dB
         850– 900 ms: envelope -100.00 dB, gain -31.78 to -30.43 dB
         900– 950 ms: envelope -100.00 dB, gain -30.39 to -29.09 dB
         950–1000 ms: envelope -100.00 dB, gain -29.05 to -27.71 dB
        a Compressor traces 0 points; the TransientShaper's meter: None"#]]
    .assert_eq(&out.join("\n"));
}
