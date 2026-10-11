//! A few plug-ins whose behaviour is known, as bloop drives them: their
//! knobs as they describe them, and what they do to a signal.

use bloop_events::EventBuffer;
use bloop_plugin_airwindows_cpp::{Effect, Id};
use bloop_plugin_api::Plugin;
use expect_test::{expect, Expect};

use crate::{noise, peak, rms_db, run, sine, RATE};

#[expect(
    clippy::panic,
    reason = "a test naming a plug-in that was not compiled"
)]
fn id(name: &str) -> Id {
    Id::named(name).unwrap_or_else(|| panic!("{name} was not compiled"))
}

/// A plug-in's knobs: name, unit, default and its reading at 0, the
/// default and 1, as `param_text` gives it.
fn check_knobs(name: &str, expect: &Expect) {
    let effect = Effect::new(id(name), RATE);
    let rows: Vec<String> = id(name)
        .params()
        .iter()
        .enumerate()
        .map(|(index, p)| {
            format!(
                "{:<10} label {:<4} default {:.3}  reads {:?} / {:?} / {:?}",
                p.name,
                format!("{:?}", p.label),
                p.default,
                effect.param_text(index, 0.0),
                effect.param_text(index, p.default),
                effect.param_text(index, 1.0),
            )
        })
        .collect();
    expect.assert_eq(&format!(
        "{name}: {}\n{}",
        id(name).listing().about,
        rows.join("\n")
    ));
}

#[test]
fn each_plugin_describes_its_own_knobs() {
    check_knobs(
        "PurestGain",
        &expect![[r#"
        PurestGain: PurestGain is a high-res noise shaped gain, with smoothed fader.
        Gain       label "dB" default 0.500  reads "-40.0 dB" / "0.00 dB" / "40.0 dB"
        Slow Fade  label ""   default 1.000  reads "0.00" / "1.00" / "1.00""#]],
    );
    check_knobs(
        "Flipity",
        &expect![[r#"
        Flipity: Flipity is a channel phase flipper/swapper utility.
        Flipity    label ""   default 0.000  reads "Dry" / "Dry" / "Swip LR""#]],
    );
    check_knobs(
        "EveryTrim",
        &expect![[r#"
        EveryTrim: EveryTrim is Left/Right, Mid/Side, and Master in one plugin.
        Left       label "dB" default 0.500  reads "-12.0 dB" / "0.00 dB" / "12.0 dB"
        Right      label "dB" default 0.500  reads "-12.0 dB" / "0.00 dB" / "12.0 dB"
        Mid        label "dB" default 0.500  reads "-12.0 dB" / "0.00 dB" / "12.0 dB"
        Side       label "dB" default 0.500  reads "-12.0 dB" / "0.00 dB" / "12.0 dB"
        Master     label "dB" default 0.500  reads "-12.0 dB" / "0.00 dB" / "12.0 dB""#]],
    );
    check_knobs(
        "kCathedral5",
        &expect![[r#"
        kCathedral5: kCathedral5 lets you place yourself anywhere you like in the space.
        Regen      label ""   default 0.500  reads "0.00" / "0.50" / "1.00"
        Derez      label ""   default 0.500  reads "0.00" / "0.50" / "1.00"
        Filter     label ""   default 0.240  reads "0.00" / "0.24" / "1.00"
        EarlyRF    label ""   default 0.500  reads "0.00" / "0.50" / "1.00"
        Positin    label ""   default 0.500  reads "0.00" / "0.50" / "1.00"
        Dry/Wet    label ""   default 0.500  reads "0.00" / "0.50" / "1.00""#]],
    );
}

/// A plug-in with `knobs` set, over a 100 Hz and a 10 kHz sine: each
/// one's level in and out, in dB, and how far the output is from the
/// input at its worst.
fn check_signal(name: &str, knobs: &[(usize, f32)], expect: &Expect) {
    let events = EventBuffer::new();
    let mut rows = Vec::new();
    for hz in [100.0, 10_000.0] {
        let mut effect = Effect::new(id(name), RATE);
        for &(index, value) in knobs {
            effect.set_param(index, value);
        }
        let input = sine(hz, 24_000, 0.5);
        let mut left = input.clone();
        let mut right = input.clone();
        run(&mut effect, &events, &mut left, &mut right);
        // The last half: past any knob easing in from its default.
        let tail = 12_000..;
        let apart = |out: &[f32]| {
            out[tail.clone()]
                .iter()
                .zip(&input[tail.clone()])
                .map(|(out, inp)| (out - inp).abs())
                .fold(0.0f32, f32::max)
        };
        rows.push(format!(
            "{hz:>6} Hz in {:+.2} dB  out L {:+.2} R {:+.2} dB  worst apart L {:.1e} R {:.1e}",
            rms_db(&input[tail.clone()]),
            rms_db(&left[tail.clone()]),
            rms_db(&right[tail.clone()]),
            apart(&left),
            apart(&right),
        ));
    }
    expect.assert_eq(&format!("{name} {knobs:?}\n{}", rows.join("\n")));
}

#[test]
fn known_plugins_do_what_they_say() {
    // 0 dB: the signal through, but for Airwindows' dither in the last bit.
    check_signal(
        "PurestGain",
        &[(0, 0.5)],
        &expect![[r#"
        PurestGain [(0, 0.5)]
           100 Hz in -9.03 dB  out L -9.03 R -9.03 dB  worst apart L 0.0e0 R 0.0e0
         10000 Hz in -9.03 dB  out L -9.03 R -9.03 dB  worst apart L 0.0e0 R 0.0e0"#]],
    );
    // -6 dB on both sides, by its knob (A * 80 - 40 dB).
    check_signal(
        "PurestGain",
        &[(0, 0.425)],
        &expect![[r#"
        PurestGain [(0, 0.425)]
           100 Hz in -9.03 dB  out L -15.03 R -15.03 dB  worst apart L 2.5e-1 R 2.5e-1
         10000 Hz in -9.03 dB  out L -15.03 R -15.03 dB  worst apart L 2.5e-1 R 2.5e-1"#]],
    );
    // Dry: through untouched; Flip L: the left inverted, the right as it was.
    check_signal(
        "Flipity",
        &[(0, 0.0)],
        &expect![[r#"
        Flipity [(0, 0.0)]
           100 Hz in -9.03 dB  out L -9.03 R -9.03 dB  worst apart L 0.0e0 R 0.0e0
         10000 Hz in -9.03 dB  out L -9.03 R -9.03 dB  worst apart L 0.0e0 R 0.0e0"#]],
    );
    check_signal(
        "Flipity",
        &[(0, 0.2)],
        &expect![[r#"
        Flipity [(0, 0.2)]
           100 Hz in -9.03 dB  out L -9.03 R -9.03 dB  worst apart L 1.0e0 R 0.0e0
         10000 Hz in -9.03 dB  out L -9.03 R -9.03 dB  worst apart L 1.0e0 R 0.0e0"#]],
    );
    // A lowpass turned well down keeps the 100 Hz and takes the 10 kHz.
    check_signal(
        "Lowpass",
        &[(0, 0.3)],
        &expect![[r#"
        Lowpass [(0, 0.3)]
           100 Hz in -9.03 dB  out L -9.11 R -9.11 dB  worst apart L 5.9e-2 R 5.9e-2
         10000 Hz in -9.03 dB  out L -28.87 R -28.87 dB  worst apart L 4.5e-1 R 4.5e-1"#]],
    );
}

/// A reverb fed noise and then cleared: what it held is gone, and silence
/// in is silence out (to the dither's level); its knobs are kept. A clone
/// keeps the knobs too, and starts empty.
#[test]
fn clearing_silences_what_a_plugin_holds_and_keeps_its_knobs() {
    let events = EventBuffer::new();
    let mut effect = Effect::new(id("kCathedral5"), RATE);
    effect.set_param(0, 0.9);
    let mut left = noise(12_000, 3);
    let mut right = noise(12_000, 5);
    run(&mut effect, &events, &mut left, &mut right);
    let mut tail = (vec![0.0; 12_000], vec![0.0; 12_000]);
    run(&mut effect, &events, &mut tail.0, &mut tail.1);
    let ringing = peak(&tail.0).max(peak(&tail.1));

    let mut copy = effect.clone();
    let mut copied = (vec![0.0; 12_000], vec![0.0; 12_000]);
    run(&mut copy, &events, &mut copied.0, &mut copied.1);

    effect.clear();
    let mut cleared = (vec![0.0; 12_000], vec![0.0; 12_000]);
    run(&mut effect, &events, &mut cleared.0, &mut cleared.1);
    expect![[r#"
        after noise, silence in: peak 6.3e-2
        cleared, silence in: peak 3.4e-8
        a clone, silence in: peak 3.4e-8
        knob 1 kept: 0.9 / clone 0.9"#]]
    .assert_eq(&format!(
        "after noise, silence in: peak {ringing:.1e}\n\
         cleared, silence in: peak {:.1e}\n\
         a clone, silence in: peak {:.1e}\n\
         knob 1 kept: {} / clone {}",
        peak(&cleared.0).max(peak(&cleared.1)),
        peak(&copied.0).max(peak(&copied.1)),
        effect.get_param(0),
        copy.get_param(0),
    ));
}

/// What a host may hand an effect that it should not: a knob that is not
/// a number, one past either end, one that is not there, a rate that is
/// not one.
#[test]
fn edges_are_held() {
    let mut effect = Effect::new(id("PurestGain"), f32::NAN);
    let rate = effect.sample_rate();
    effect.set_param(0, 2.0);
    let high = effect.get_param(0);
    effect.set_param(0, -1.0);
    let low = effect.get_param(0);
    effect.set_param(0, f32::NAN);
    let nan = effect.get_param(0);
    effect.set_param(99, 0.3);
    let info = effect.param_info(99);
    effect.reset(0.0, 128);
    expect![[r#"
        rate 48000, knob at 2.0 → 1, at -1.0 → 0, at NaN → 0.5
        knob 99: 0.0 "" reads "", count 2
        reset at 0 Hz → 48000 Hz"#]]
    .assert_eq(&format!(
        "rate {rate}, knob at 2.0 → {high}, at -1.0 → {low}, at NaN → {nan}\n\
         knob 99: {:?} {:?} reads {:?}, count {}\n\
         reset at 0 Hz → {} Hz",
        effect.get_param(99),
        info.name,
        effect.param_text(99, 0.5),
        effect.param_count(),
        effect.sample_rate(),
    ));
}
