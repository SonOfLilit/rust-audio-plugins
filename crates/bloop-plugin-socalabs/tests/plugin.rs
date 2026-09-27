//! The effects as a host sees them: what each says about itself, what its
//! knobs do with the values they are given, and what `process` and `reset`
//! cost. What the audio sounds like is tests/gold.rs.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;

use bloop_plugin_api::{Plugin, ProcessContext};
use bloop_plugin_socalabs::{Effect, Kind};
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
               2. Harmonics          0 to 100       %   default 0         skew 1
               3. Mix                0 to 200       %   default 50        skew 1
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
            PlateReverb by SocaLabs, Effect, 2 in 2 out
               0. Mix                0 to 100       %   default 25        skew 1
               1. Predelay           0 to 100       ms  default 0         skew 1
               2. Lowpass           16 to 20000     Hz  default 10000     skew 0.3
               3. Decay              0 to 100       %   default 50        skew 1
               4. Size               0 to 200       %   default 100       skew 1
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
               4. High Cut           0 to 20000     Hz  default 0         skew 0.3
               5. Detector           0 to 100       %   default 50        skew 1
               6. Process            0 to 100       %   default 50        skew 1
               7. Output           -24 to 24        dB  default 0         skew 1
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
            &[-40.0, 10.0, -1000.0, f32::NAN, f32::INFINITY],
        ),
        &expect![[r#"
            Threshold -40 → -40
            Threshold 10 → 0
            Threshold -1000 → -60
            Threshold NaN → -60
            Threshold inf → -60"#]],
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
            PlateReverb: 6 knobs; knob 6 is "" 0..0 reading 0
            SimpleVerb: 7 knobs; knob 7 is "" 0..0 reading 0
            SoftClipper: 3 knobs; knob 3 is "" 0..0 reading 0
            StereoEnhancer: 3 knobs; knob 3 is "" 0..0 reading 0
            StereoProcessor: 8 knobs; knob 8 is "" 0..0 reading 0
            TransientShaper: 8 knobs; knob 8 is "" 0..0 reading 0
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
