//! The effects as a host sees them: what each says about itself, what its
//! knobs do with the values they are given, and what `process` and `reset`
//! cost. What the audio sounds like is tests/gold.rs.

#![expect(
    clippy::expect_used,
    reason = "a test that cannot find the knob it is about has nothing to say"
)]
#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;

use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_socalabs::{Effect, Kind, Recipe};
use expect_test::{expect, Expect};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

/// Counts the allocations made on the thread that is asking, so tests
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

fn process(effect: &mut Effect, left: &mut [f32], right: &mut [f32]) {
    let events = bloop_events::EventBuffer::new();
    let mut outputs: [&mut [f32]; 2] = [left, right];
    let mut context = ProcessContext {
        outputs: &mut outputs,
        input_events: &events,
        sample_rate: 48000.0,
        tempo: 120.0,
        is_playing: true,
    };
    effect.process(&mut context);
}

fn check(text: &str, expect: &Expect) {
    expect.assert_eq(text);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[expect(
    clippy::too_many_lines,
    reason = "the table is the snapshot, one line a knob"
)]
fn every_effect_describes_itself() {
    let mut out = String::new();
    for kind in Kind::ALL {
        let effect = Effect::new(kind, 48000.0);
        let info = effect.info();
        let _ = writeln!(
            out,
            "{} by {}, {:?}, {} in {} out",
            info.name, info.vendor, info.category, info.num_audio_inputs, info.num_audio_outputs
        );
        for i in 0..effect.param_count() {
            let p = effect.param_info(i);
            let _ = writeln!(
                out,
                "  {i:>2}. {:<10} {:>9} to {:<9} {:<3} default {:<9} skew {}",
                p.name,
                p.min,
                p.max,
                effect.param_unit(i),
                p.default,
                effect.param_skew(i)
            );
        }
    }
    check(
        &out,
        &expect![[r#"
            BitCrusher by SocaLabs, Effect, 2 in 2 out
               0. Bits               1 to 16            default 16        skew 1
               1. Rate             500 to 48000     Hz  default 48000     skew 0.3
               2. Dirty              0 to 1             default 0         skew 1
            Chorus by SocaLabs, Effect, 2 in 2 out
               0. Delay            0.1 to 30        ms  default 1         skew 1
               1. Depth            0.1 to 20        ms  default 1         skew 1
               2. Speed            0.1 to 10        Hz  default 3         skew 1
               3. Width              0 to 100       %   default 50        skew 1
               4. Mix                0 to 100       %   default 50        skew 1
            Compressor by SocaLabs, Effect, 2 in 2 out
               0. Attack             1 to 200       ms  default 1         skew 0.4
               1. Release            1 to 2000      ms  default 5         skew 0.4
               2. Ratio              1 to 30            default 5         skew 0.4
               3. Threshold        -60 to 0         dB  default -30       skew 1
               4. Knee               0 to 60        dB  default 5         skew 1
               5. Input            -30 to 30        dB  default 0         skew 1
               6. Output           -30 to 30        dB  default 0         skew 1
            Crossfeed by SocaLabs, Effect, 2 in 2 out
               0. Enable             0 to 1             default 1         skew 1
            Delay by SocaLabs, Effect, 2 in 2 out
               0. Sync               0 to 1             default 0         skew 1
               1. Time               0 to 120       s   default 1         skew 0.3
               2. Beat               0 to 32            default 13        skew 1
               3. Feedback        -100 to 0         dB  default -10       skew 5
               4. Crossfeed       -100 to 0         dB  default -100      skew 5
               5. Mix                0 to 100       %   default 0         skew 1
            Distortion by SocaLabs, Effect, 2 in 2 out
               0. Density           -1 to 4             default 0         skew 1
               1. Highpass           0 to 100       %   default 0         skew 1
               2. Output             0 to 100       %   default 100       skew 1
               3. Mix                0 to 100       %   default 100       skew 1
            Exciter by SocaLabs, Effect, 2 in 2 out
               0. Frequency       1000 to 16000     Hz  default 7000      skew 0.5
               1. Drive              1 to 8             default 1         skew 1
               2. Harmonics          0 to 100       %   default 0         skew 0.5
               3. Mix                0 to 100       %   default 50        skew 1
            Expander by SocaLabs, Effect, 2 in 2 out
               0. Attack             1 to 200       ms  default 1         skew 0.4
               1. Release            1 to 2000      ms  default 5         skew 0.4
               2. Ratio              1 to 4             default 1         skew 1
               3. Threshold        -60 to 0         dB  default -30       skew 1
               4. Knee               0 to 60        dB  default 5         skew 1
               5. Input            -30 to 30        dB  default 0         skew 1
               6. Output           -30 to 30        dB  default 0         skew 1
            Gate by SocaLabs, Effect, 2 in 2 out
               0. Attack             0 to 5         ms  default 0         skew 1
               1. Hold               0 to 500       ms  default 0         skew 0.3
               2. Release            1 to 500       ms  default 5         skew 0.3
               3. Threshold       -100 to 0         dB  default -30       skew 1
               4. Knee               0 to 60        dB  default 0         skew 1
               5. Input            -30 to 30        dB  default 0         skew 1
               6. Output           -30 to 30        dB  default 0         skew 1
            GraphicEQ by SocaLabs, Effect, 2 in 2 out
               0. 31 Hz            -12 to 12        dB  default 0         skew 1
               1. 63 Hz            -12 to 12        dB  default 0         skew 1
               2. 125 Hz           -12 to 12        dB  default 0         skew 1
               3. 250 Hz           -12 to 12        dB  default 0         skew 1
               4. 500 Hz           -12 to 12        dB  default 0         skew 1
               5. 1k Hz            -12 to 12        dB  default 0         skew 1
               6. 2k Hz            -12 to 12        dB  default 0         skew 1
               7. 4k Hz            -12 to 12        dB  default 0         skew 1
               8. 8k Hz            -12 to 12        dB  default 0         skew 1
               9. 16k Hz           -12 to 12        dB  default 0         skew 1
              10. LP                 0 to 1             default 0         skew 1
              11. LP Freq      83.2131 to 135.07623     default 135.07623 skew 1
              12. LP Slope           0 to 1             default 0         skew 1
              13. HP                 0 to 1             default 0         skew 1
              14. HP Freq     15.48682 to 95.2131       default 15.48682  skew 1
              15. HP Slope           0 to 1             default 0         skew 1
              16. Output           -12 to 12        dB  default 0         skew 1
            Limiter by SocaLabs, Effect, 2 in 2 out
               0. Attack             0 to 5         ms  default 0         skew 1
               1. Release            1 to 500       ms  default 5         skew 0.3
               2. Threshold        -60 to 0         dB  default -30       skew 1
               3. Input            -30 to 30        dB  default 0         skew 1
               4. Output           -30 to 30        dB  default 0         skew 1
            Maximizer by SocaLabs, Effect, 2 in 2 out
               0. Threshold        -24 to 0         dB  default -6        skew 1
               1. Ceiling          -24 to 0         dB  default -0.3      skew 1
               2. Release            0 to 1000      ms  default 100       skew 0.5
            Oscilloscope by SocaLabs, Effect, 2 in 2 out
               0. Samp/px       0.0625 to 48            default 1         skew 0.3
               1. Zoom             0.1 to 100           default 1         skew 0.3
               2. Offset L          -2 to 2             default 0         skew 1
               3. Offset R          -2 to 2             default 0         skew 1
               4. Trigger Chan        -1 to 1             default 0         skew 1
               5. Trigger Mode         0 to 3             default 3         skew 1
               6. Trigger Level        -1 to 1             default 0         skew 1
               7. Trigger Pos         0 to 1             default 0         skew 1
               8. Trigger Run         0 to 1             default 0         skew 1
               9. Sync               0 to 1             default 0         skew 1
              10. Beats              1 to 32            default 4         skew 1
            PlateReverb by SocaLabs, Effect, 2 in 2 out
               0. Mix                0 to 100       %   default 25        skew 1
               1. Predelay           0 to 100       ms  default 0         skew 1
               2. Lowpass           16 to 20000     Hz  default 10000     skew 0.3
               3. Decay              0 to 95        %   default 50        skew 1
               4. Size              10 to 200       %   default 100       skew 1
               5. Damping           16 to 20000     Hz  default 10000     skew 0.3
            SimpleVerb by SocaLabs, Effect, 2 in 2 out
               0. Room               0 to 1         %   default 0.3       skew 1
               1. Damp               0 to 1         %   default 0.6       skew 1
               2. PreDelay           0 to 1         %   default 0.3       skew 1
               3. LP                 0 to 1         %   default 0.9       skew 1
               4. HP                 0 to 1         %   default 0.1       skew 1
               5. Wet                0 to 1         %   default 0.4       skew 1
               6. Dry                0 to 1         %   default 0.3       skew 1
            SoftClipper by SocaLabs, Effect, 2 in 2 out
               0. Drive              0 to 24        dB  default 0         skew 1
               1. Knee               0 to 100       %   default 50        skew 1
               2. Ceiling          -24 to 0         dB  default 0         skew 1
            SpectrumAnalyzer by SocaLabs, Effect, 2 in 2 out
               0. Mode               0 to 1             default 0         skew 1
               1. Log                0 to 1             default 0         skew 1
            StereoEnhancer by SocaLabs, Effect, 2 in 2 out
               0. Width              0 to 1         %   default 0.5       skew 1
               1. Freq HP            0 to 1         %   default 0.5       skew 1
               2. Width HP           0 to 1         %   default 0         skew 1
            StereoProcessor by SocaLabs, Effect, 2 in 2 out
               0. Width 1            0 to 1         %   default 0.5       skew 1
               1. Center 1           0 to 1         %   default 0.5       skew 1
               2. Pan 1             -1 to 1         %   default 0         skew 1
               3. Rotation           0 to 1         %   default 0.5       skew 1
               4. Pan 2             -1 to 1         %   default 0         skew 1
               5. Center 2           0 to 1         %   default 0.5       skew 1
               6. Width 2            0 to 1         %   default 0.5       skew 1
               7. Output             0 to 1         %   default 0.5       skew 1
            TransientShaper by SocaLabs, Effect, 2 in 2 out
               0. Attack            -4 to 4             default 0         skew 1
               1. Sustain           -8 to 8             default 0         skew 1
               2. Smoothing          0 to 100       %   default 50        skew 1
               3. Low Cut            0 to 2000      Hz  default 0         skew 0.5
               4. High Cut          20 to 20000     Hz  default 20000     skew 0.3
               5. Detector           0 to 100       %   default 50        skew 1
               6. Process            0 to 100       %   default 50        skew 1
               7. Output           -24 to 12        dB  default 0         skew 1
            XYScope by SocaLabs, Effect, 2 in 2 out
               0. Zoom            0.25 to 4             default 1         skew 0.5
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn every_knob_starts_at_its_default() {
    for kind in Kind::ALL {
        let effect = Effect::new(kind, 48000.0);
        for i in 0..effect.param_count() {
            assert_eq!(
                effect.get_param(i).to_bits(),
                effect.param_info(i).default.to_bits(),
                "{kind:?} knob {i}"
            );
        }
    }
}

/// What reading a knob back gives after setting it: snapped to its step
/// and held to its range, or left alone.
fn after_setting(kind: Kind, index: usize, values: &[f32]) -> String {
    let mut effect = Effect::new(kind, 48000.0);
    let name = effect.param_info(index).name;
    values
        .iter()
        .map(|&v| {
            effect.set_param(index, v);
            format!("{name} {v} → {}", effect.get_param(index))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn knobs_snap_and_hold_to_their_range() {
    check(
        &after_setting(
            Kind::Compressor,
            3,
            &[
                -40.0,
                -1000.0,
                10.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
        ),
        &expect![[r#"
            Threshold -40 → -40
            Threshold -1000 → -60
            Threshold 10 → 0
            Threshold NaN → 0
            Threshold inf → 0
            Threshold -inf → 0"#]],
    );
    check(
        &after_setting(Kind::Delay, 2, &[10.4, 10.6, 10.5, -3.0, 99.0]),
        &expect![[r#"
            Beat 10.4 → 10
            Beat 10.6 → 11
            Beat 10.5 → 11
            Beat -3 → 0
            Beat 99 → 32"#]],
    );
    check(
        &after_setting(Kind::Delay, 0, &[0.4, 0.6, 1.0, 2.0]),
        &expect![[r#"
        Sync 0.4 → 0
        Sync 0.6 → 1
        Sync 1 → 1
        Sync 2 → 1"#]],
    );
    check(
        &after_setting(Kind::Crossfeed, 0, &[0.49, 0.51, 0.0]),
        &expect![[r#"
        Enable 0.49 → 0
        Enable 0.51 → 1
        Enable 0 → 0"#]],
    );
    check(
        &after_setting(Kind::Chorus, 0, &[0.0, 0.1, 15.0, 30.0, 45.0]),
        &expect![[r#"
            Delay 0 → 0.1
            Delay 0.1 → 0.1
            Delay 15 → 15
            Delay 30 → 30
            Delay 45 → 30"#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_knob_past_the_last_is_nothing() {
    let mut out = String::new();
    for kind in Kind::ALL {
        let mut effect = Effect::new(kind, 48000.0);
        let past = effect.param_count();
        effect.set_param(past, 1.0);
        let info = effect.param_info(past);
        let _ = writeln!(
            out,
            "{kind:?}: {past} knobs; knob {past} is {:?} {}..{} reading {}",
            info.name,
            info.min,
            info.max,
            effect.get_param(past)
        );
    }
    check(
        &out,
        &expect![[r#"
            BitCrusher: 3 knobs; knob 3 is "" 0..0 reading 0
            Chorus: 5 knobs; knob 5 is "" 0..0 reading 0
            Compressor: 7 knobs; knob 7 is "" 0..0 reading 0
            Crossfeed: 1 knobs; knob 1 is "" 0..0 reading 0
            Delay: 6 knobs; knob 6 is "" 0..0 reading 0
            Distortion: 4 knobs; knob 4 is "" 0..0 reading 0
            Exciter: 4 knobs; knob 4 is "" 0..0 reading 0
            Expander: 7 knobs; knob 7 is "" 0..0 reading 0
            Gate: 7 knobs; knob 7 is "" 0..0 reading 0
            GraphicEq: 17 knobs; knob 17 is "" 0..0 reading 0
            Limiter: 5 knobs; knob 5 is "" 0..0 reading 0
            Maximizer: 3 knobs; knob 3 is "" 0..0 reading 0
            Oscilloscope: 11 knobs; knob 11 is "" 0..0 reading 0
            PlateReverb: 6 knobs; knob 6 is "" 0..0 reading 0
            SimpleVerb: 7 knobs; knob 7 is "" 0..0 reading 0
            SoftClipper: 3 knobs; knob 3 is "" 0..0 reading 0
            SpectrumAnalyzer: 2 knobs; knob 2 is "" 0..0 reading 0
            StereoEnhancer: 3 knobs; knob 3 is "" 0..0 reading 0
            StereoProcessor: 8 knobs; knob 8 is "" 0..0 reading 0
            TransientShaper: 8 knobs; knob 8 is "" 0..0 reading 0
            XyScope: 1 knobs; knob 1 is "" 0..0 reading 0
        "#]],
    );
}

/// The Delay's time is an internal parameter behind Time, Sync and Beat;
/// it is not a knob, and setting the index after the last knob must not
/// reach it.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_delays_internal_time_is_not_a_knob() {
    let run = |set_seventh: bool| {
        let mut effect = Effect::new(Kind::Delay, 48000.0);
        effect.reset(48000.0, 512);
        effect.set_param(5, 100.0);
        if set_seventh {
            effect.set_param(6, 0.01);
        }
        let mut left = vec![0.0f32; 4800];
        let mut right = vec![0.0f32; 4800];
        left[0] = 1.0;
        process(&mut effect, &mut left, &mut right);
        left
    };
    assert_eq!(run(false), run(true));
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn an_effect_given_no_stereo_pair_does_nothing() {
    for kind in Kind::ALL {
        let mut effect = Effect::new(kind, 48000.0);
        let events = bloop_events::EventBuffer::new();
        let mut mono = vec![0.5f32; 64];
        let mut outputs: [&mut [f32]; 1] = [&mut mono];
        let mut context = ProcessContext {
            outputs: &mut outputs,
            input_events: &events,
            sample_rate: 48000.0,
            tempo: 120.0,
            is_playing: true,
        };
        effect.process(&mut context);
        assert!(
            mono.iter().all(|&s| s == 0.5),
            "{kind:?} touched a mono buffer"
        );
        let (mut left, mut right) = (Vec::new(), Vec::new());
        process(&mut effect, &mut left, &mut right);
    }
}

/// The audio thread may call `process`, turn knobs, and `reset` at the rate
/// the effect was prepared for; none of it may allocate.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_audio_thread_never_allocates() {
    for kind in Kind::ALL {
        let mut effect = Effect::new(kind, 48000.0);
        effect.reset(48000.0, 512);
        let mut left = vec![0.25f32; 512];
        let mut right = vec![-0.25f32; 512];
        let count = allocations_during(|| {
            for block in 0..20 {
                for i in 0..effect.param_count() {
                    let info = effect.param_info(i);
                    let at = if block % 2 == 0 { info.max } else { info.min };
                    effect.set_param(i, at);
                }
                process(&mut effect, &mut left, &mut right);
                process(&mut effect, &mut left[..1], &mut right[..1]);
            }
            effect.reset(48000.0, 512);
            process(&mut effect, &mut left, &mut right);
        });
        assert_eq!(
            count, 0,
            "{kind:?} allocated {count} times on the audio thread"
        );
    }
}

/// Two short stereo hits, loud then quiet, the channels different noise:
/// something for every knob to act on — a threshold to cross, a side
/// channel to widen, a tail to shape.
fn hits() -> (Vec<f32>, Vec<f32>) {
    let mut state = 0x1234_5678u32;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0) as f32
    };
    let mut left = Vec::with_capacity(16384);
    let mut right = Vec::with_capacity(16384);
    for i in 0..16384 {
        let (level, t) = if i < 8192 { (0.5, i) } else { (0.02, i - 8192) };
        let env = level * libm::expf(-(t as f32) / 1500.0);
        left.push(env * noise());
        right.push(env * noise());
    }
    (left, right)
}

/// A fingerprint of what `kind` makes of the hits with `knobs` set, the
/// smoothers settled by a reset before the first block.
fn render(kind: Kind, knobs: &[(usize, f32)]) -> u64 {
    let mut effect = Effect::new(kind, 48000.0);
    for &(index, value) in knobs {
        effect.set_param(index, value);
    }
    effect.reset(48000.0, 512);
    let (mut left, mut right) = hits();
    for (l, r) in left.chunks_mut(512).zip(right.chunks_mut(512)) {
        process(&mut effect, l, r);
    }
    left.iter()
        .chain(&right)
        .fold(0xcbf2_9ce4_8422_2325u64, |h, s| {
            (h ^ u64::from(s.to_bits())).wrapping_mul(0x0100_0000_01b3)
        })
}

/// What a knob needs set before what it does can be heard: a delay's
/// time says nothing while the mix is dry.
fn needs(kind: Kind, knob: &str) -> &'static [(&'static str, f32)] {
    match (kind, knob) {
        // The knobs the others wait on.
        (Kind::Delay, "Mix")
        | (Kind::Expander, "Ratio")
        | (Kind::Distortion, "Density")
        | (Kind::TransientShaper, "Attack") => &[],
        (Kind::Delay, "Beat") => &[("Sync", 1.0), ("Mix", 100.0)],
        (Kind::Delay, "Sync") => &[("Beat", 1.0), ("Mix", 100.0)],
        (Kind::Delay, "Time") => &[("Mix", 100.0)],
        (Kind::Delay, _) => &[("Time", 0.05), ("Mix", 100.0)],
        (Kind::Maximizer, "Release") => &[("Threshold", -24.0)],
        (Kind::Expander, _) => &[("Ratio", 4.0)],
        (Kind::GraphicEq, "LP Freq" | "LP Slope") => &[("LP", 1.0)],
        (Kind::GraphicEq, "HP Freq" | "HP Slope") => &[("HP", 1.0)],
        (Kind::Distortion, _) => &[("Density", 2.0)],
        (Kind::TransientShaper, _) => &[("Attack", 2.0)],
        (Kind::SoftClipper, "Knee") => &[("Drive", 12.0)],
        _ => &[],
    }
}

/// Every knob of every effect changes what comes out: from its default,
/// turned to one end of its range or the other, the output is not the
/// same bits. A knob that reached nothing — read into the wrong slot,
/// never passed on — would be listed as still.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[ignore = "heavy: cargo test -p bloop-plugin-socalabs -- --include-ignored"]
fn every_knob_changes_what_comes_out() {
    let mut out = String::new();
    for kind in Kind::ALL {
        let probe = Effect::new(kind, 48000.0);
        let index_of = |name: &str| {
            (0..probe.param_count())
                .find(|&i| probe.param_info(i).name == name)
                .expect("a prerequisite names a knob the effect has")
        };
        let mut moved = Vec::new();
        let mut still = Vec::new();
        for i in 0..probe.param_count() {
            let info = probe.param_info(i);
            let before: Vec<(usize, f32)> = needs(kind, info.name)
                .iter()
                .map(|&(name, value)| (index_of(name), value))
                .collect();
            let base = render(kind, &before);
            let turned = [info.min, info.max].into_iter().any(|end| {
                let mut knobs = before.clone();
                knobs.push((i, end));
                render(kind, &knobs) != base
            });
            let label = match needs(kind, info.name) {
                [] => info.name.to_string(),
                prerequisites => format!(
                    "{} (with {})",
                    info.name,
                    prerequisites
                        .iter()
                        .map(|(n, v)| format!("{n} {v}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            };
            if turned {
                moved.push(label);
            } else {
                still.push(label);
            }
        }
        let _ = writeln!(out, "{}: {}", kind.name(), moved.join(", "));
        if !still.is_empty() {
            let _ = writeln!(out, "  STILL: {}", still.join(", "));
        }
    }
    check(
        &out,
        &expect![[r#"
            BitCrusher: Bits, Rate, Dirty
            Chorus: Delay, Depth, Speed, Width, Mix
            Compressor: Attack, Release, Ratio, Threshold, Knee, Input, Output
            Crossfeed: Enable
            Delay: Sync (with Beat 1, Mix 100), Time (with Mix 100), Beat (with Sync 1, Mix 100), Feedback (with Time 0.05, Mix 100), Crossfeed (with Time 0.05, Mix 100), Mix
            Distortion: Density, Highpass (with Density 2), Output (with Density 2), Mix (with Density 2)
            Exciter: Frequency, Drive, Harmonics, Mix
            Expander: Attack (with Ratio 4), Release (with Ratio 4), Ratio, Threshold (with Ratio 4), Knee (with Ratio 4), Input (with Ratio 4), Output (with Ratio 4)
            Gate: Attack, Hold, Release, Threshold, Knee, Input, Output
            GraphicEQ: 31 Hz, 63 Hz, 125 Hz, 250 Hz, 500 Hz, 1k Hz, 2k Hz, 4k Hz, 8k Hz, 16k Hz, LP, LP Freq (with LP 1), LP Slope (with LP 1), HP, HP Freq (with HP 1), HP Slope (with HP 1), Output
            Limiter: Attack, Release, Threshold, Input, Output
            Maximizer: Threshold, Ceiling, Release (with Threshold -24)
            Oscilloscope: 
              STILL: Samp/px, Zoom, Offset L, Offset R, Trigger Chan, Trigger Mode, Trigger Level, Trigger Pos, Trigger Run, Sync, Beats
            PlateReverb: Mix, Predelay, Lowpass, Decay, Size, Damping
            SimpleVerb: Room, Damp, PreDelay, LP, HP, Wet, Dry
            SoftClipper: Drive, Knee (with Drive 12), Ceiling
            SpectrumAnalyzer: 
              STILL: Mode, Log
            StereoEnhancer: Width, Freq HP, Width HP
            StereoProcessor: Width 1, Center 1, Pan 1, Rotation, Pan 2, Center 2, Width 2, Output
            TransientShaper: Attack, Sustain (with Attack 2), Smoothing (with Attack 2), Low Cut (with Attack 2), High Cut (with Attack 2), Detector (with Attack 2), Process (with Attack 2), Output (with Attack 2)
            XYScope: 
              STILL: Zoom
        "#]],
    );
}

/// The room a Delay is built with, for the time it is asked to reach:
/// doubled from the least, never more than the original's.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_delays_room_doubles_until_it_reaches() {
    let out: Vec<String> = [
        0.0, 1.0, 1.9, 1.95, 2.0, 3.9, 10.0, 60.0, 119.0, 120.0, 500.0,
    ]
    .iter()
    .map(|&s| format!("{s} s → room {}", Recipe::room_for(s)))
    .collect();
    check(
        &out.join("\n"),
        &expect![[r#"
        0 s → room 2
        1 s → room 2
        1.9 s → room 2
        1.95 s → room 4
        2 s → room 4
        3.9 s → room 4
        10 s → room 16
        60 s → room 64
        119 s → room 120.1
        120 s → room 120.1
        500 s → room 120.1"#]],
    );
}

/// Where an impulse comes back, through a Delay with `room` seconds of
/// line set to `time`, fully wet: the first sample that is not silent.
fn echo_at(room: f32, time: f32) -> String {
    let mut effect = Effect::build(
        Recipe {
            kind: Kind::Delay,
            delay_room: room,
        },
        48000.0,
    );
    let names: Vec<&str> = (0..effect.param_count())
        .map(|i| effect.param_info(i).name)
        .collect();
    let knob = |name: &str| names.iter().position(|n| *n == name).expect("a Delay knob");
    effect.set_param(knob("Time"), time);
    effect.set_param(knob("Mix"), 100.0);
    effect.set_param(knob("Feedback"), -100.0);
    effect.reset(48000.0, 512);
    let seconds = 8usize;
    let (mut left, mut right) = (vec![0.0f32; 48000 * seconds], vec![0.0f32; 48000 * seconds]);
    left[0] = 1.0;
    right[0] = 1.0;
    for (l, r) in left.chunks_mut(512).zip(right.chunks_mut(512)) {
        process(&mut effect, l, r);
    }
    left.iter().position(|s| s.abs() > 1e-3).map_or_else(
        || "no echo in 8 s".to_string(),
        |i| format!("echo at {:.3} s", i as f64 / 48000.0),
    )
}

/// A Delay reaches as far back as its room: a time past it reads from the
/// far end of the line, which is why the host builds it again with more.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_delay_reaches_as_far_as_its_room() {
    let out: Vec<String> = [
        (2.0, 1.0),
        (2.0, 5.0),
        (8.0, 5.0),
        (Recipe::MAX_DELAY_ROOM, 5.0),
    ]
    .iter()
    .map(|&(room, time)| format!("room {room:>5} s, Time {time} s: {}", echo_at(room, time)))
    .collect();
    check(
        &out.join("\n"),
        &expect![[r#"
        room     2 s, Time 1 s: echo at 1.000 s
        room     2 s, Time 5 s: echo at 2.000 s
        room     8 s, Time 5 s: echo at 5.000 s
        room 120.1 s, Time 5 s: echo at 5.000 s"#]],
    );
}

/// A knob's travel, 0 to 1, as the parameter under it: skewed where the
/// range is, stepped for a switch or a list, and back again.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_knobs_travel_is_its_parameters_range() {
    let mut out = String::new();
    for (kind, index) in [
        (Kind::Delay, 1),
        (Kind::Delay, 2),
        (Kind::Delay, 0),
        (Kind::Compressor, 3),
        (Kind::GraphicEq, 11),
        (Kind::Chorus, 2),
    ] {
        let effect = Effect::new(kind, 48000.0);
        let name = effect.param_info(index).name;
        let values: Vec<String> = [0.0, 0.25, 0.49, 0.51, 0.75, 1.0]
            .iter()
            .map(|&f| {
                let v = effect.knob_value(index, f);
                format!("{f}→{v}→{:.3}", effect.knob_fraction(index, v))
            })
            .collect();
        let _ = writeln!(out, "{} {name}: {}", kind.name(), values.join("  "));
    }
    // A skewed knob's travel is a power, and `gold` swaps the port's math
    // from `libm` to glibc's, as the C++ it is compared with links: the two
    // disagree in the last places of a float, so each has its snapshot.
    #[cfg(not(feature = "gold"))]
    let expected = expect![[r#"
        Delay Time: 0→0→0.000  0.25→1.181176→0.250  0.49→11.130163→0.490  0.51→12.717894→0.510  0.75→45.99587→0.750  1→120→1.000
        Delay Beat: 0→0→0.000  0.25→8→0.250  0.49→16→0.500  0.51→16→0.500  0.75→24→0.750  1→32→1.000
        Delay Sync: 0→0→0.000  0.25→0→0.000  0.49→0→0.000  0.51→1→1.000  0.75→1→1.000  1→1→1.000
        Compressor Threshold: 0→-60→0.000  0.25→-45→0.250  0.49→-30.599998→0.490  0.51→-29.400002→0.510  0.75→-15→0.750  1→0→1.000
        GraphicEQ LP Freq: 0→83.2131→0.000  0.25→96.17888→0.250  0.49→108.62604→0.490  0.51→109.6633→0.510  0.75→122.11045→0.750  1→135.07623→1.000
        Chorus Speed: 0→0.1→0.000  0.25→2.5749998→0.250  0.49→4.9509997→0.490  0.51→5.1489997→0.510  0.75→7.5249996→0.750  1→10→1.000
    "#]];
    #[cfg(feature = "gold")]
    let expected = expect![[r#"
        Delay Time: 0→0→0.000  0.25→1.181176→0.250  0.49→11.130163→0.490  0.51→12.717894→0.510  0.75→45.995865→0.750  1→120→1.000
        Delay Beat: 0→0→0.000  0.25→8→0.250  0.49→16→0.500  0.51→16→0.500  0.75→24→0.750  1→32→1.000
        Delay Sync: 0→0→0.000  0.25→0→0.000  0.49→0→0.000  0.51→1→1.000  0.75→1→1.000  1→1→1.000
        Compressor Threshold: 0→-60→0.000  0.25→-45→0.250  0.49→-30.599998→0.490  0.51→-29.400002→0.510  0.75→-15→0.750  1→0→1.000
        GraphicEQ LP Freq: 0→83.2131→0.000  0.25→96.17888→0.250  0.49→108.62604→0.490  0.51→109.6633→0.510  0.75→122.11045→0.750  1→135.07623→1.000
        Chorus Speed: 0→0.1→0.000  0.25→2.5749998→0.250  0.49→4.9509997→0.490  0.51→5.1489997→0.510  0.75→7.5249996→0.750  1→10→1.000
    "#]];
    check(&out, &expected);
}

/// `clear` is Choke All Notes' part in an effect: whatever it is still
/// holding — echoes, a reverb's tank, an envelope — goes, and the effect
/// is otherwise as it was. Each effect, wet all the way where it has a Mix
/// or a Wet, is fed a tenth of a second of noise and cleared. Then: the
/// loudest sample of a second of silence after it; whether a second of
/// fresh noise then comes out as it does from the same effect that only
/// ever heard silence (the largest difference); and what `clear`
/// allocated.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn clear_leaves_nothing_ringing() {
    let wet = |effect: &mut Effect| {
        for i in 0..effect.param_count() {
            let p = effect.param_info(i);
            if matches!(p.name, "Mix" | "Wet") {
                effect.set_param(i, p.max);
            }
        }
    };
    let noise = |seed: &mut u32| {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        #[expect(clippy::cast_precision_loss, reason = "noise, not a value")]
        let x = *seed as f32 / u32::MAX as f32;
        x - 0.5
    };
    // A second of blocks: silence, or noise from `seed`.
    let run = |effect: &mut Effect, seed: Option<u32>| -> Vec<f32> {
        let mut seed = seed;
        let mut out = Vec::new();
        for _ in 0..400 {
            let mut left: Vec<f32> = (0..120)
                .map(|_| seed.as_mut().map_or(0.0, &noise))
                .collect();
            let mut right = left.clone();
            process(effect, &mut left, &mut right);
            out.extend(left);
            out.extend(right);
        }
        out
    };
    let mut out = String::new();
    for kind in Kind::ALL {
        let mut effect = Effect::new(kind, 48000.0);
        wet(&mut effect);
        let mut seed = 0x2545_f491_u32;
        for _ in 0..40 {
            let mut left: Vec<f32> = (0..120).map(|_| noise(&mut seed)).collect();
            let mut right = left.clone();
            process(&mut effect, &mut left, &mut right);
        }
        let allocated = allocations_during(|| effect.clear());
        let loudest = run(&mut effect, None)
            .iter()
            .fold(0.0f32, |most, x| most.max(x.abs()));
        effect.clear();
        // The same effect that never heard the noise and was never
        // cleared: the same knobs, and the same tenth of a second, of
        // silence. A clear that stopped an effect working shows here.
        let mut fresh = Effect::new(kind, 48000.0);
        wet(&mut fresh);
        for _ in 0..40 {
            let (mut left, mut right) = (vec![0.0f32; 120], vec![0.0f32; 120]);
            process(&mut fresh, &mut left, &mut right);
        }
        // As long as the cleared one has run: the silence it was asked
        // for, and its own tenth of a second.
        let _ = run(&mut fresh, None);
        let heard = run(&mut effect, Some(7));
        let expected = run(&mut fresh, Some(7));
        let apart = heard
            .iter()
            .zip(&expected)
            .fold(0.0f32, |most, (a, b)| most.max((a - b).abs()));
        let _ = writeln!(
            out,
            "{:<16} silence after clear: {loudest:.6}, then noise off by {apart:.6}, {allocated} allocations",
            kind.name()
        );
    }
    check(
        &out,
        &expect![[r#"
            BitCrusher       silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Chorus           silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Compressor       silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Crossfeed        silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Delay            silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Distortion       silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Exciter          silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Expander         silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Gate             silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            GraphicEQ        silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Limiter          silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Maximizer        silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            Oscilloscope     silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            PlateReverb      silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            SimpleVerb       silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            SoftClipper      silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            SpectrumAnalyzer silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            StereoEnhancer   silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            StereoProcessor  silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
            TransientShaper  silence after clear: 0.000000, then noise off by 0.000003, 0 allocations
            XYScope          silence after clear: 0.000000, then noise off by 0.000000, 0 allocations
        "#]],
    );
}

/// The provenance comment in Cargo.toml names the revisions the License
/// pages cite, so the two cannot drift apart.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn cargo_toml_cites_the_ported_revisions() {
    let manifest = include_str!("../Cargo.toml");
    for rev in [
        bloop_plugin_socalabs::PORTED_FROM_SLPLUGINS_REV,
        bloop_plugin_socalabs::PORTED_FROM_GIN_REV,
        bloop_plugin_socalabs::PORTED_FROM_WAVETABLE_REV,
    ] {
        assert!(manifest.contains(rev), "Cargo.toml must cite {rev}");
    }
}
