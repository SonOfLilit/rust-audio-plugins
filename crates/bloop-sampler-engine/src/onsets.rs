//! Finding where the hits are.
//!
//! Chopping a break evenly only lands on the beat if the break was played to a
//! grid. Detecting onsets puts the cuts where the drums actually are, which is
//! what makes a chopped break usable without hand-nudging every marker.
//!
//! The method is deliberately plain: short-time energy, positive difference,
//! peak-picking against a local median. It runs on the UI thread over the
//! whole sample, so it is linear and allocation-tolerant, and it is easy to
//! reason about when a slice lands somewhere surprising.

use crate::SampleBuffer;

/// Frames per analysis step. About 2.7 ms at 48 kHz — fine enough to place a
/// cut inside a drum's attack, coarse enough that a whole break is a few
/// thousand steps.
const HOP: usize = 128;

/// Frames each step measures over. Two hops, so windows overlap and a
/// transient between steps still registers.
const WINDOW: usize = 256;

/// How far either side the local median looks, in steps. Wide enough to span
/// a beat at most tempos, so a quiet passage does not raise the bar for the
/// hits inside it.
const MEDIAN_RADIUS: usize = 24;

/// How far apart onsets must be, in seconds, at the lowest and highest
/// sensitivity.
///
/// This is what actually decides how many cuts a dense break yields: on real
/// material almost every allowed slot contains something that clears an
/// energy threshold, so separation, not loudness, is the useful control. Half
/// a second finds the beats; thirty milliseconds finds the ghost notes.
///
/// The knob interpolates between them *geometrically*, because the number of
/// cuts goes as one over the separation: stepping the separation linearly
/// spends half the travel between twelve cuts and eighteen and then leaps to
/// a hundred. Geometrically, equal turns multiply the density by equal
/// factors, so the knob is about twice as fine every quarter turn all the way
/// along.
const MIN_SEPARATION_RANGE: (f32, f32) = (0.5, 0.03);

/// How far before the strongest rise of a hit its attack may be, in
/// seconds. A drum's stick comes a few tens of milliseconds before its
/// shell rings up, and the shell is often the bigger rise.
const ATTACK_LOOKBACK: f32 = 0.06;

/// How far the energy may sag between an attack and the rise it belongs
/// to, as a share of where the attack took it. More than that, and what
/// came before was a hit of its own that died away — a ghost note, which in
/// the funky break falls to a tenth before the hit after it. A flam's first
/// stroke, which in Drums_170BPM falls to a third, is the hit's attack.
const ATTACK_HOLDS: f32 = 0.25;

/// How quiet it must be just before an earlier attack, as a share of the
/// hit's loudest, for the attack to be the hit's: a flam's first stroke
/// comes out of silence, a ripple in a busy break does not.
const ATTACK_FROM_QUIET: f32 = 0.05;

/// How big a rise before a hit must be, as a share of the hit's own, to be
/// its attack rather than noise ahead of it. Rises are in energy, so this is
/// an attack at about 40% of the body's amplitude.
const ATTACK_SHARE: f32 = 0.15;

/// Where the hits are in `range`, as absolute frame positions, ascending.
///
/// These are *interior* boundaries: a hit already sounding at the start of the
/// range has no rise to detect, so the caller bounds the first slice with the
/// range's own start.
///
/// A returned position sits at the start of the analysis window the energy
/// rose into, which puts it up to a window early — deliberately, since a cut
/// just before a transient keeps the attack and a cut just after clips it.
///
/// `sensitivity` runs 0.0 (only the most obvious hits) to 1.0 (nearly every
/// rise in energy).
///
/// Returns an empty list when there is nothing to find — silence, or a range
/// too short to analyse.
pub fn detect_onsets(sample: &SampleBuffer, range: (usize, usize), sensitivity: f32) -> Vec<usize> {
    let (from, to) = (range.0.min(range.1), range.1.max(range.0));
    let span = to.saturating_sub(from);
    if span < WINDOW * 2 || sample.num_channels() == 0 {
        return Vec::new();
    }

    // Short-time energy, all channels together: a hit on one side of a stereo
    // break is still a hit.
    let steps = (span - WINDOW) / HOP + 1;
    let mut energy = Vec::with_capacity(steps);
    for step in 0..steps {
        let start = from + step * HOP;
        let mut sum = 0.0f64;
        for channel in 0..sample.num_channels() {
            for value in &sample.channel(channel)[start..(start + WINDOW).min(sample.num_samples())]
            {
                sum += f64::from(*value) * f64::from(*value);
            }
        }
        energy.push((sum / WINDOW as f64) as f32);
    }

    // Rises only: energy falling away is a decay, not a new hit.
    let flux: Vec<f32> = energy
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).max(0.0))
        .collect();
    let peak = flux.iter().copied().fold(0.0f32, f32::max);
    if peak <= f32::EPSILON {
        return Vec::new();
    }

    let sensitivity = sensitivity.clamp(0.0, 1.0);
    // How far above the local median a rise must sit to count at all.
    let factor = 3.0 - 2.0 * sensitivity;
    let floor = peak * 0.02 * (1.0 - sensitivity);
    let (coarse, fine) = MIN_SEPARATION_RANGE;
    let separation = coarse * libm::powf(fine / coarse, sensitivity);
    let min_gap_steps = ((separation * sample.sample_rate() as f32) / HOP as f32)
        .ceil()
        .max(1.0) as usize;

    // Every local maximum that clears the threshold, strongest first.
    let mut scratch: Vec<f32> = Vec::with_capacity(MEDIAN_RADIUS * 2 + 1);
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (step, &value) in flux.iter().enumerate() {
        // A local maximum, so one hit yields one cut rather than a smear.
        let rising = step == 0 || flux[step - 1] <= value;
        let falling = step + 1 >= flux.len() || flux[step + 1] < value;
        if !(rising && falling) {
            continue;
        }

        let low = step.saturating_sub(MEDIAN_RADIUS);
        let high = (step + MEDIAN_RADIUS + 1).min(flux.len());
        scratch.clear();
        scratch.extend_from_slice(&flux[low..high]);
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = scratch[scratch.len() / 2];

        if value >= median * factor + floor {
            candidates.push((step, value));
        }
    }

    // Thin them out strongest-first, so a hit wins over the weaker rise just
    // before it. Taking them in time order instead would keep the precursor
    // and drop the hit.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    let mut kept: Vec<usize> = Vec::new();
    for (step, _) in candidates {
        if kept
            .iter()
            .any(|&other| step.abs_diff(other) < min_gap_steps)
        {
            continue;
        }
        kept.push(step);
    }

    // Each hit is cut at its attack: an earlier stroke of the same hit — a
    // stick before the shell, a flam's first — if one rose out of quiet
    // shortly before the rise strongest-first picked, and the energy held
    // up from it into that rise. The local median does not judge that
    // stroke: the body right after it is what raises the median there.
    let lookback = ((ATTACK_LOOKBACK * sample.sample_rate() as f32) / HOP as f32).ceil() as usize;
    let peak_at = |step: usize| {
        (step == 0 || flux[step - 1] <= flux[step])
            && (step + 1 >= flux.len() || flux[step + 1] < flux[step])
    };
    // Where the rise through `step` began.
    let foot = |mut step: usize| {
        while step > 0 && flux[step - 1] > 0.0 {
            step -= 1;
        }
        step
    };
    for step in &mut kept {
        let strength = flux[*step];
        let loudest = energy[*step + 1..(*step + 1 + lookback).min(energy.len())]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let held = |at: usize| {
            let reached = energy[at + 1];
            energy[at + 1..=*step]
                .iter()
                .all(|&level| level >= reached * ATTACK_HOLDS)
        };
        if let Some(attack) = (step.saturating_sub(lookback)..*step).find(|&at| {
            peak_at(at)
                && flux[at] >= strength * ATTACK_SHARE
                && energy[foot(at)] <= loudest * ATTACK_FROM_QUIET
                && held(at)
        }) {
            *step = attack;
        }
    }

    kept.sort_unstable();
    kept.dedup();
    // The range's own edges are slice boundaries too, so a hit within the
    // separation of one is not a new slice — it is the start (or the end) of
    // the range arriving slightly late. Without this a break whose first
    // transient sits a few milliseconds in gets a runt first slice: a sliver
    // of pre-roll on its own pad, and the actual downbeat on the next.
    let last_step = flux.len().saturating_sub(1);
    kept.retain(|&step| step >= min_gap_steps && last_step.saturating_sub(step) >= min_gap_steps);

    // `flux[step]` is the rise *into* the next window, so that is where the
    // hit begins.
    kept.into_iter()
        .map(|step| from + (step + 1) * HOP)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Silence with decaying hits at known frames — a stand-in for a break.
    fn hits(frames: usize, at: &[usize]) -> SampleBuffer {
        let mut data = vec![0.0f32; frames];
        for &start in at {
            for i in 0..2000usize {
                let Some(slot) = data.get_mut(start + i) else {
                    break;
                };
                let decay = (-(i as f32) / 300.0).exp();
                // A tone rather than a click, so the energy is realistic.
                *slot += (i as f32 * 0.35).sin() * decay;
            }
        }
        SampleBuffer::new(vec![data], 48_000)
    }

    fn report(onsets: &[usize], expected: &[usize]) -> String {
        let mut out = format!("found {} onsets\n", onsets.len());
        for &onset in onsets {
            let nearest = expected
                .iter()
                .map(|&e| (e as i64 - onset as i64).abs())
                .min()
                .unwrap_or(-1);
            out.push_str(&format!(
                "  {onset:>6}  (nearest hit {nearest:>5} frames away)\n"
            ));
        }
        out
    }

    /// Four evenly spaced hits are found at their frames, within a hop.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn evenly_spaced_hits_are_found() {
        let at = [0, 12_000, 24_000, 36_000];
        let sample = hits(48_000, &at);
        let onsets = detect_onsets(&sample, (0, 48_000), 0.5);
        expect![[r#"
            found 3 onsets
               11904  (nearest hit    96 frames away)
               23936  (nearest hit    64 frames away)
               35840  (nearest hit   160 frames away)
        "#]]
        .assert_eq(&report(&onsets, &at));
    }

    /// A break is rarely on a grid — the point of detecting rather than
    /// dividing is that the cuts follow the playing.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn hits_off_the_grid_are_still_found() {
        let at = [0, 9_500, 23_000, 31_700];
        let sample = hits(48_000, &at);
        let onsets = detect_onsets(&sample, (0, 48_000), 0.5);
        expect![[r#"
            found 3 onsets
                9344  (nearest hit   156 frames away)
               22912  (nearest hit    88 frames away)
               31616  (nearest hit    84 frames away)
        "#]]
        .assert_eq(&report(&onsets, &at));
    }

    /// Two hits closer together than the minimum separation are one onset,
    /// not two — a snare's attack and its body are the same hit.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn hits_inside_the_minimum_separation_merge() {
        let at = [0, 600, 24_000];
        let sample = hits(48_000, &at);
        let onsets = detect_onsets(&sample, (0, 48_000), 0.5);
        expect![[r#"
            found 1 onsets
               23936  (nearest hit    64 frames away)
        "#]]
        .assert_eq(&report(&onsets, &at));
    }

    /// A drum's attack is often a smaller rise than its body arriving a
    /// few tens of milliseconds later: the stick, then the shell. The cut
    /// goes before the attack, not before the body — cut at the body and
    /// the attack is left on the end of the Slice before. Hits at 12 000
    /// and 30 000 whose bodies, twice as loud, follow 2 000 frames (42 ms)
    /// later. A quieter hit that dies away before the next is its own
    /// hit, and does not pull the next one's cut back: that is
    /// `a_ghost_note_is_not_the_next_hits_attack`.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_cut_goes_before_the_attack_not_the_body() {
        let bodies = hits(48_000, &[14_000, 32_000]);
        // The stick: half the body's level, still ringing when the shell
        // comes in.
        let stick = |i: usize| {
            [12_000usize, 30_000]
                .iter()
                .filter(|&&at| (at..at + 2_000).contains(&i))
                .map(|&at| ((i - at) as f32 * 0.35).sin() * 0.5)
                .sum::<f32>()
        };
        let drum: Vec<f32> = (0..48_000)
            .map(|i| stick(i) + bodies.sample(0, i))
            .collect();
        let sample = SampleBuffer::new(vec![drum], 48_000);
        let at = [12_000, 30_000];
        let report = |sensitivity: f32| {
            format!(
                "sensitivity {sensitivity}: {}",
                report(&detect_onsets(&sample, (0, 48_000), sensitivity), &at)
            )
        };
        expect![[r#"
            sensitivity 0.1: found 0 onsets
            sensitivity 0.5: found 2 onsets
               11904  (nearest hit    96 frames away)
               29952  (nearest hit    48 frames away)
        "#]]
        .assert_eq(&format!("{}{}", report(0.1), report(0.5)));
    }

    /// A ghost note 50 ms before a hit, dying away before it: at a
    /// sensitivity that keeps only the hit, the cut stays at the hit.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_ghost_note_is_not_the_next_hits_attack() {
        let loud = hits(48_000, &[14_000, 32_000]);
        let ghosts = hits(48_000, &[11_600, 29_600]);
        let beat: Vec<f32> = (0..48_000)
            .map(|i| loud.sample(0, i) + ghosts.sample(0, i) * 0.45)
            .collect();
        let sample = SampleBuffer::new(vec![beat], 48_000);
        expect![[r#"
            found 2 onsets
               13824  (nearest hit   176 frames away)
               31872  (nearest hit   128 frames away)
        "#]]
        .assert_eq(&report(
            &detect_onsets(&sample, (0, 48_000), 0.3),
            &[14_000, 32_000],
        ));
    }

    /// Sensitivity trades hits found against noise found.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn sensitivity_trades_hits_against_noise() {
        // Three loud hits and two much quieter ones between them.
        let mut sample = hits(48_000, &[0, 16_000, 32_000]);
        let quiet = hits(48_000, &[8_000, 24_000]);
        let mixed: Vec<f32> = (0..48_000)
            .map(|i| sample.sample(0, i) + quiet.sample(0, i) * 0.15)
            .collect();
        sample = SampleBuffer::new(vec![mixed], 48_000);

        let counts: Vec<String> = [0.0f32, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|&s| format!("{s:.2} → {}", detect_onsets(&sample, (0, 48_000), s).len()))
            .collect();
        expect![[r#"
            0.00 → 0
            0.25 → 2
            0.50 → 4
            0.75 → 4
            1.00 → 4"#]]
        .assert_eq(&counts.join("\n"));
    }

    /// Silence has no hits, and neither does a range too short to analyse.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn silence_and_tiny_ranges_find_nothing() {
        let silent = SampleBuffer::new(vec![vec![0.0; 48_000]], 48_000);
        let loud = hits(48_000, &[0, 24_000]);
        expect![[r#"
            silence: 0
            tiny range: 0"#]]
        .assert_eq(&format!(
            "silence: {}\ntiny range: {}",
            detect_onsets(&silent, (0, 48_000), 0.5).len(),
            detect_onsets(&loud, (0, 100), 0.5).len(),
        ));
    }

    /// Detection is scoped to the range asked for, so zooming in and
    /// re-detecting chops only what is on screen.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn detection_is_scoped_to_the_range() {
        let at = [0, 12_000, 24_000, 36_000];
        let sample = hits(48_000, &at);
        let onsets = detect_onsets(&sample, (20_000, 48_000), 0.5);
        expect![[r#"
            found 1 onsets
               35872  (nearest hit   128 frames away)
        "#]]
        .assert_eq(&report(&onsets, &at));
    }
}
