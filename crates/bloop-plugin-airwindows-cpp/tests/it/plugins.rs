//! The collection as a whole: what was compiled, what was left out, and
//! that every plug-in compiled runs clean.

use std::collections::BTreeMap;

use bloop_events::EventBuffer;
use bloop_plugin_airwindows_cpp::{Effect, Id, COUNT, SKIPPED};
use bloop_plugin_api::Plugin;
use expect_test::expect;

use crate::{allocations, noise, peak, run, RATE};

/// How many plug-ins were compiled, which were left out and why, and how
/// many Airwindopedia files under each category — the Browser's TYPE.
#[test]
fn the_catalog_lists_every_plugin_compiled() {
    let mut by_category: BTreeMap<&str, usize> = BTreeMap::new();
    for id in Id::all() {
        *by_category.entry(id.listing().category).or_default() += 1;
    }
    let categories: Vec<String> = by_category
        .iter()
        .map(|(category, n)| {
            let category = if category.is_empty() {
                "(none)"
            } else {
                category
            };
            format!("  {category}: {n}")
        })
        .collect();
    let skipped: Vec<String> = SKIPPED
        .iter()
        .map(|(name, why)| format!("  {name}: {why}"))
        .collect();
    let named: Vec<String> = [
        "Acceleration",
        "kCathedral5",
        "ToTape8",
        "PunchyGuitar",
        "Flipity",
    ]
    .into_iter()
    .map(|name| format!("  {name}: {:?}", Id::named(name).map(Id::name)))
    .collect();
    let sorted = Id::all()
        .zip(Id::all().skip(1))
        .all(|(a, b)| a.name().to_lowercase() < b.name().to_lowercase());
    expect![[r#"
        compiled: 517
        by name: true
        skipped:
          Baxandall2: ported to Rust (bloop-plugin-airwindows); the Browser lists the port
          Console7Buss: ported to Rust (bloop-plugin-airwindows); the Browser lists the port
          Console7Channel: ported to Rust (bloop-plugin-airwindows); the Browser lists the port
          Density2: ported to Rust (bloop-plugin-airwindows); the Browser lists the port
          Galactic3: ported to Rust (bloop-plugin-airwindows); the Browser lists the port
          ToTape8: ported to Rust (bloop-plugin-airwindows); the Browser lists the port
          PunchyGuitar: its output grows without bound at its defaults: inf within a second of noise
        categories:
          (none): 20
          Ambience: 25
          Amp Sims: 15
          Bass: 10
          Biquads: 9
          Brightness: 26
          Clipping: 10
          Consoles: 43
          Distortion: 12
          Dithers: 23
          Dynamics: 35
          Effects: 28
          Filter: 52
          Lo-Fi: 20
          Noise: 9
          Reverb: 40
          Saturation: 18
          Stereo: 12
          Subtlety: 20
          Tape: 11
          Tone Color: 17
          Utility: 38
          XYZ Filters: 24
        named:
          Acceleration: Some("Acceleration")
          kCathedral5: Some("kCathedral5")
          ToTape8: None
          PunchyGuitar: None
          Flipity: Some("Flipity")
        first: Some(Listing { name: "Acceleration", category: "Brightness", about: "Acceleration is an acceleration limiter that tames edge, leaves brightness." })
        last: Some(Listing { name: "ZRegion2", category: "XYZ Filters", about: "ZRegion2 is an Emu e6400 style Airwindows Region filter, with control smoothing." })"#]].assert_eq(&format!(
        "compiled: {COUNT}\nby name: {sorted}\nskipped:\n{}\ncategories:\n{}\nnamed:\n{}\nfirst: {:?}\nlast: {:?}",
        skipped.join("\n"),
        categories.join("\n"),
        named.join("\n"),
        Id::all().next().map(Id::listing),
        Id::all().last().map(Id::listing),
    ));
}

/// Every plug-in compiled: built at 48 kHz, it has the knobs it says; a
/// second of noise through it, three times, comes out finite; its processing allocates
/// nothing (Linux, where the binary counts the C++'s allocations too);
/// cleared, it runs again; reset to 44.1 kHz, it runs again. Lists the
/// ones that do not.
#[test]
#[ignore = "heavy: every plug-in of the collection over three seconds of noise, ~15 s"]
fn every_plugin_runs_clean_and_allocates_nothing() {
    let input_left = noise(48_000, 7);
    let input_right = noise(48_000, 11);
    let events = EventBuffer::new();
    let mut wrong = Vec::new();
    let mut silent = Vec::new();
    for id in Id::all() {
        let mut effect = Effect::new(id, RATE);
        if effect.param_count() != id.params().len() {
            wrong.push(format!(
                "{}: {} knobs, says {}",
                id.name(),
                effect.param_count(),
                id.params().len()
            ));
        }
        let mut loudest = 0.0f32;
        let mut finite = true;
        let mut allocated = 0;
        for round in 0..3 {
            let mut left = input_left.clone();
            let mut right = input_right.clone();
            let before = allocations();
            run(&mut effect, &events, &mut left, &mut right);
            if let (Some(before), Some(after)) = (before, allocations()) {
                allocated += after - before;
            }
            finite &= left.iter().chain(&right).all(|s| s.is_finite());
            loudest = loudest.max(peak(&left)).max(peak(&right));
            let before = allocations();
            match round {
                0 => effect.clear(),
                _ => effect.reset(44_100.0, 256),
            }
            if let (Some(before), Some(after)) = (before, allocations()) {
                allocated += after - before;
            }
        }
        if !finite {
            wrong.push(format!("{}: not finite", id.name()));
        }
        if allocated > 0 {
            wrong.push(format!("{}: {allocated} allocations", id.name()));
        }
        if loudest < 1e-6 {
            silent.push(id.name());
        }
    }
    expect![[r#"
        plug-ins: 517
        wrong: []
        silent at their defaults: ["Air2", "ContentHideD", "DubCenter", "DubSub", "HipCrush"]"#]]
    .assert_eq(&format!(
        "plug-ins: {COUNT}\nwrong: {wrong:#?}\nsilent at their defaults: {silent:?}"
    ));
}

/// The count the test above holds to none sees the C++'s allocations as
/// well as Rust's: building a plug-in is one of Rust's (where its knobs are
/// kept) and two of the C++'s (the instance and the plug-in's memory).
#[test]
fn the_allocation_count_sees_the_cpp() {
    let Some(id) = Id::named("PurestGain") else {
        return;
    };
    let _ = id.params();
    let before = allocations();
    let effect = Effect::new(id, RATE);
    let after = allocations();
    drop(effect);
    expect!["Some(3)"].assert_eq(&format!("{:?}", after.zip(before).map(|(a, b)| a - b)));
}
