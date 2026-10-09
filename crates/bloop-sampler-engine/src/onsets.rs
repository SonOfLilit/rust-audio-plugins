//! Finding where the hits are.
//!
//! Chopping a break evenly only lands on the beat if the break was played to a
//! grid. Detecting onsets puts the cuts where the drums actually are, which is
//! what makes a chopped break usable without hand-nudging every marker.
//!
//! The method is deliberately plain: short-time energy, how far it rises,
//! peak-picking against a local median. It runs on the UI thread over the
//! whole sample, so it is linear and allocation-tolerant, and it is easy to
//! reason about when a slice lands somewhere surprising.
//!
//! **The energy is measured in decibels**, which is how loud a sound is to
//! the ear rather than how much power it carries. In power, a rise is
//! scored by how much louder the sound got, so loud material wins: in the
//! vocal "Check This Out" the vowel swelling inside "Check" rose five times
//! as much as the start of "This", and no setting of the knob cut the three
//! words apart. In decibels a rise is scored by how many *times* louder the
//! sound got, so a word or a drum starting out of quiet stands out however
//! loud it ends up, and a swell or a ripple on top of something already
//! sounding does not. Three rules keep that from finding everything:
//!
//! - quiet is quiet: the level has a floor [`FLOOR_DB`] below the loudest
//!   moment, so a rise out of hiss is measured from the floor, not from
//!   the hiss's own (meaningless) depth;
//! - a rise counts only if it lands within [`LANDS_WITHIN_DB`] of the
//!   loudest moment, so something that rises out of silence to stay
//!   quiet — a breath, a hat's bleed — is not a hit;
//! - a rise is measured from the loudest the sound was over the last
//!   [`RECENT`], so a dip inside a sound and its recovery — a snare's
//!   rattle, a vowel's wobble — is not a new sound.
//!
//! That also makes the attack of a hit win over its body: a stick out of
//! silence is a bigger rise, in decibels, than the shell ringing up on top
//! of it, so the cut goes before the stick without looking back for it.
//!
//! Loudness still has a say in one place: which of two rises too close
//! together to both be cuts is kept. A ghost note out of silence can be as
//! big a rise as the hit after it; the hit lands louder, and wins
//! ([`TENFOLD_DB`]).
//!
//! `cut_sweep` in bloop-samples runs this over the pack's funky break,
//! Drums_170BPM and the vocal at every tenth of the knob, against where
//! their hits and words are; that table is how these numbers were chosen.

use crate::SampleBuffer;

/// Frames per analysis step. About 2.7 ms at 48 kHz — fine enough to place a
/// cut inside a drum's attack, coarse enough that a whole break is a few
/// thousand steps.
const HOP: usize = 128;

/// Frames each step measures over. Two hops, so windows overlap and a
/// transient between steps still registers.
const WINDOW: usize = 256;

/// How many steps a rise is measured across: about 11 ms at 48 kHz. A
/// word's consonant or a soft attack takes a few steps to reach its level,
/// and a rise measured one step at a time splits it into small ones that
/// a single loud step outscores.
const RISE_STEPS: usize = 4;

/// How far below the loudest moment the level stops going down, in dB.
/// Silence and the hiss of a recording are both "quiet"; without a floor,
/// digital silence would make any rise out of it infinite.
const FLOOR_DB: f32 = 50.0;

/// Energy at or below which a whole range is silence: -90 dB, below the
/// dither of a 16-bit recording. Decibels make a rise as big in a faint
/// recording as in a loud one, which is right down to here and wrong below
/// it, where the only rises are rounding noise.
const SILENCE: f32 = 1e-9;

/// How far below the loudest moment a rise may land and still be a hit, in
/// dB. A rise out of silence that tops out 30 dB down is a breath or bleed,
/// not something to put on a pad.
const LANDS_WITHIN_DB: f32 = 30.0;

/// How far back, in seconds, a rise looks for the level it rises from: it
/// is measured from the loudest of these steps, not just the one before.
/// A sound that dips and comes back inside this — a snare's rattle in
/// Drums_170BPM, ~15 ms deep — is one sound. Two hits closer than this
/// are one hit anyway.
const RECENT: f32 = 0.02;

/// How much more a rise must be than the rises around it (their median),
/// in dB, at the lowest and the highest sensitivity. Additive, because
/// the level is: 8 dB is a sound getting six times louder than the
/// material's usual flicker; 2 dB is any real step up.
const MARGIN_DB_RANGE: (f32, f32) = (8.0, 2.0);

/// How far either side the local median looks, in steps. Wide enough to span
/// a beat at most tempos, so a quiet passage does not raise the bar for the
/// hits inside it.
const MEDIAN_RADIUS: usize = 24;

/// How far apart onsets must be, in seconds, at the lowest and highest
/// sensitivity.
///
/// This is what actually decides how many cuts a dense break yields: on real
/// material almost every allowed slot contains something that clears an
/// energy threshold, so separation, not loudness, is the useful control. A
/// third of a second finds the beats; thirty milliseconds finds the ghost
/// notes. (It was half a second when rises were measured in power, which
/// let more through: with fewer, weaker candidates to thin, a half second
/// kept only every other beat of Drums_170BPM well up the knob.)
///
/// The knob interpolates between them *geometrically*, because the number of
/// cuts goes as one over the separation: stepping the separation linearly
/// spends half the travel between twelve cuts and eighteen and then leaps to
/// a hundred. Geometrically, equal turns multiply the density by equal
/// factors, so the knob is about twice as fine every quarter turn all the way
/// along.
const MIN_SEPARATION_RANGE: (f32, f32) = (0.35, 0.03);

/// How many steps back the cut may go from the steepest step of a rise, to
/// a gap before it, and how much quieter the gap must be, in dB. A word
/// can begin from a gap a few milliseconds before its steepest step ("Out",
/// in "Check This Out", climbs steepest ~6 ms after its gap, from 7 dB
/// down), and the cut belongs in the gap. A drum in a busy break creeps up
/// a dB or two on the hats before it, and that is not a gap.
const VALLEY_STEPS: usize = 2;
const VALLEY_DB: f32 = 3.0;

/// How far below the loudest moment a rise counts a tenth as much, in dB,
/// when the strongest rises are kept first. Only the order: what is a hit
/// at all is the rise alone.
const TENFOLD_DB: f32 = 40.0;

/// Where the hits are in `range`, as absolute frame positions, ascending.
///
/// These are *interior* boundaries: a hit already sounding at the start of the
/// range has no rise to detect, so the caller bounds the first slice with the
/// range's own start.
///
/// A returned position sits where the last quiet analysis window before the
/// rise ends, which puts it up to a step early — deliberately, since a cut
/// just before a transient keeps the attack and a cut just after clips it.
///
/// `sensitivity` runs 0.0 (only the most obvious hits) to 1.0 (every sound
/// that starts).
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
    let loudest = energy.iter().copied().fold(0.0f32, f32::max);
    if loudest <= SILENCE || steps <= RISE_STEPS {
        return Vec::new();
    }

    // The level in dB below the loudest moment: 0 at the loudest, never
    // below -FLOOR_DB.
    let peak_db = 10.0 * libm::log10f(loudest);
    let level: Vec<f32> = energy
        .iter()
        .map(|&e| (10.0 * libm::log10f(e.max(f32::MIN_POSITIVE)) - peak_db).max(-FLOOR_DB))
        .collect();

    // `rise[step]` is how far the level climbs from around `step` to
    // `RISE_STEPS` later, measured from the loudest of the recent steps,
    // and only where it lands loud enough to be a hit.
    let recent = ((RECENT * sample.sample_rate() as f32) / HOP as f32).round() as usize;
    let rise: Vec<f32> = (0..steps - RISE_STEPS)
        .map(|step| {
            let lands = level[step + RISE_STEPS];
            if lands < -LANDS_WITHIN_DB {
                return 0.0;
            }
            let was = level[step.saturating_sub(recent)..=step]
                .iter()
                .copied()
                .fold(-FLOOR_DB, f32::max);
            (lands - was).max(0.0)
        })
        .collect();
    if rise.iter().all(|&r| r <= 0.0) {
        return Vec::new();
    }

    let sensitivity = sensitivity.clamp(0.0, 1.0);
    let (strict, lenient) = MARGIN_DB_RANGE;
    let margin = strict + (lenient - strict) * sensitivity;
    let (coarse, fine) = MIN_SEPARATION_RANGE;
    let separation = coarse * libm::powf(fine / coarse, sensitivity);
    let min_gap_steps = ((separation * sample.sample_rate() as f32) / HOP as f32)
        .ceil()
        .max(1.0) as usize;

    // Every local maximum that clears the threshold, strongest first.
    let mut scratch: Vec<f32> = Vec::with_capacity(MEDIAN_RADIUS * 2 + 1);
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (step, &value) in rise.iter().enumerate() {
        // A local maximum, so one hit yields one cut rather than a smear.
        let rising = step == 0 || rise[step - 1] <= value;
        let falling = step + 1 >= rise.len() || rise[step + 1] < value;
        if !(rising && falling) || value <= 0.0 {
            continue;
        }

        let low = step.saturating_sub(MEDIAN_RADIUS);
        let high = (step + MEDIAN_RADIUS + 1).min(rise.len());
        scratch.clear();
        scratch.extend_from_slice(&rise[low..high]);
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = scratch[scratch.len() / 2];

        if value >= median + margin {
            // How loud it gets, as a weight: 1 at the loudest, a tenth
            // TENFOLD_DB below.
            let reaches = level[step + 1..=step + RISE_STEPS]
                .iter()
                .copied()
                .fold(-FLOOR_DB, f32::max);
            candidates.push((step, value * libm::powf(10.0, reaches / TENFOLD_DB)));
        }
    }

    // Thin them out strongest-first, so a hit wins over the weaker rise just
    // before it. Taking them in time order instead would keep the precursor
    // and drop the hit. Strongest is the rise weighted by how loud it gets:
    // between two rises out of quiet, the hit that lands loud is the more
    // obvious than the ghost note that lands soft.
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

    kept.sort_unstable();
    // The range's own edges are slice boundaries too, so a hit within the
    // separation of one is not a new slice — it is the start (or the end) of
    // the range arriving slightly late. Without this a break whose first
    // transient sits a few milliseconds in gets a runt first slice: a sliver
    // of pre-roll on its own pad, and the actual downbeat on the next.
    let last_step = rise.len().saturating_sub(1);
    kept.retain(|&step| step >= min_gap_steps && last_step.saturating_sub(step) >= min_gap_steps);

    // The cut goes before the steepest single step of the rise: the window
    // there is the last before the sound, and it ends where the sound
    // begins, give or take a step. If the level climbs into that step out
    // of a gap — a word's consonant out of the pause before it — the cut
    // goes back into the gap.
    kept.into_iter()
        .map(|step| {
            let around =
                step.saturating_sub(VALLEY_STEPS).max(1)..(step + RISE_STEPS).min(steps - 1);
            let steepest = around
                .rev()
                .max_by(|&a, &b| {
                    (level[a + 1] - level[a])
                        .partial_cmp(&(level[b + 1] - level[b]))
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .unwrap_or(step);
            let quiet = (steepest.saturating_sub(VALLEY_STEPS)..steepest)
                .find(|&before| level[before] <= level[steepest] - VALLEY_DB)
                .unwrap_or(steepest);
            from + (quiet + 2) * HOP
        })
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
               35968  (nearest hit    32 frames away)
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
                9472  (nearest hit    28 frames away)
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

    /// A drum's attack is often a smaller rise in power than its body
    /// arriving a few tens of milliseconds later: the stick, then the
    /// shell. In dB it is the bigger one, out of silence. The cut goes
    /// before the attack, not before the body — cut at the body and the
    /// attack is left on the end of the Slice before. Hits at 12 000
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
            sensitivity 0.1: found 1 onsets
               29952  (nearest hit    48 frames away)
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
               13952  (nearest hit    48 frames away)
               32000  (nearest hit     0 frames away)
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

    /// Every synthetic case across the knob, 0.1 to 0.9: the cuts, in
    /// frames from the nearest hit the case puts there (`!` for a cut on
    /// none, within 1 000 frames), and the hits no cut is near.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_synthetic_cases_across_the_knob() {
        let mix = |parts: &[(&[usize], f32)]| {
            let layers: Vec<(SampleBuffer, f32)> = parts
                .iter()
                .map(|(at, gain)| (hits(48_000, at), *gain))
                .collect();
            let data = (0..48_000)
                .map(|i| layers.iter().map(|(s, g)| s.sample(0, i) * g).sum())
                .collect();
            SampleBuffer::new(vec![data], 48_000)
        };
        let stick = {
            let bodies = hits(48_000, &[14_000, 32_000]);
            let data = (0..48_000)
                .map(|i| {
                    let stick: f32 = [12_000usize, 30_000]
                        .iter()
                        .filter(|&&at| (at..at + 2_000).contains(&i))
                        .map(|&at| ((i - at) as f32 * 0.35).sin() * 0.5)
                        .sum();
                    stick + bodies.sample(0, i)
                })
                .collect();
            SampleBuffer::new(vec![data], 48_000)
        };
        let cases: [(&str, SampleBuffer, &[usize]); 6] = [
            (
                "even",
                hits(48_000, &[0, 12_000, 24_000, 36_000]),
                &[12_000, 24_000, 36_000],
            ),
            (
                "off grid",
                hits(48_000, &[0, 9_500, 23_000, 31_700]),
                &[9_500, 23_000, 31_700],
            ),
            ("merge", hits(48_000, &[0, 600, 24_000]), &[24_000]),
            ("stick", stick, &[12_000, 30_000]),
            (
                "ghost",
                mix(&[(&[14_000, 32_000], 1.0), (&[11_600, 29_600], 0.45)]),
                &[14_000, 32_000],
            ),
            (
                "quiet",
                mix(&[(&[0, 16_000, 32_000], 1.0), (&[8_000, 24_000], 0.15)]),
                &[16_000, 32_000, 8_000, 24_000],
            ),
        ];
        let mut out = String::new();
        for (name, sample, at) in &cases {
            out.push_str(&format!("{name}\n"));
            for tenth in 1..=9u8 {
                let sensitivity = f32::from(tenth) / 10.0;
                let onsets = detect_onsets(sample, (0, 48_000), sensitivity);
                let cuts: Vec<String> = onsets
                    .iter()
                    .map(|&onset| {
                        let d = at
                            .iter()
                            .map(|&e| onset as i64 - e as i64)
                            .min_by_key(|d| d.abs())
                            .filter(|d| d.abs() <= 1_000);
                        d.map_or(format!("{onset}!"), |d| format!("{onset}{d:+}"))
                    })
                    .collect();
                let missed: Vec<String> = at
                    .iter()
                    .filter(|&&e| !onsets.iter().any(|&o| o.abs_diff(e) <= 1_000))
                    .map(ToString::to_string)
                    .collect();
                out.push_str(&format!("  {sensitivity:.1} {}", cuts.join(" ")));
                if !missed.is_empty() {
                    out.push_str(&format!("  missed {}", missed.join(" ")));
                }
                out.push('\n');
            }
        }
        expect![[r#"
            even
              0.1   missed 12000 24000 36000
              0.2 11904-96 23936-64 35968-32
              0.3 11904-96 23936-64 35968-32
              0.4 11904-96 23936-64 35968-32
              0.5 11904-96 23936-64 35968-32
              0.6 11904-96 23936-64 35968-32
              0.7 11904-96 23936-64 35968-32
              0.8 11904-96 23936-64 35968-32
              0.9 11904-96 23936-64 35968-32
            off grid
              0.1 31616-84  missed 9500 23000
              0.2 31616-84  missed 9500 23000
              0.3 9472-28 22912-88 31616-84
              0.4 9472-28 22912-88 31616-84
              0.5 9472-28 22912-88 31616-84
              0.6 9472-28 22912-88 31616-84
              0.7 9472-28 22912-88 31616-84
              0.8 9472-28 22912-88 31616-84
              0.9 9472-28 22912-88 31616-84
            merge
              0.1 23936-64
              0.2 23936-64
              0.3 23936-64
              0.4 23936-64
              0.5 23936-64
              0.6 23936-64
              0.7 23936-64
              0.8 23936-64
              0.9 23936-64
            stick
              0.1 29952-48  missed 12000
              0.2 11904-96 29952-48
              0.3 11904-96 29952-48
              0.4 11904-96 29952-48
              0.5 11904-96 29952-48
              0.6 11904-96 29952-48
              0.7 11904-96 29952-48
              0.8 11904-96 29952-48
              0.9 11904-96 13952! 29952-48
            ghost
              0.1 13952-48 32000+0
              0.2 13952-48 32000+0
              0.3 13952-48 32000+0
              0.4 13952-48 32000+0
              0.5 13952-48 32000+0
              0.6 13952-48 32000+0
              0.7 13952-48 32000+0
              0.8 11520! 13952-48 29568! 32000+0
              0.9 11520! 13952-48 29568! 32000+0
            quiet
              0.1 16000+0 32000+0  missed 8000 24000
              0.2 16000+0 32000+0  missed 8000 24000
              0.3 16000+0 32000+0  missed 8000 24000
              0.4 7936-64 16000+0 23936-64 32000+0
              0.5 7936-64 16000+0 23936-64 32000+0
              0.6 7936-64 16000+0 23936-64 32000+0
              0.7 7936-64 16000+0 23936-64 32000+0
              0.8 7936-64 16000+0 23936-64 32000+0
              0.9 7936-64 16000+0 23936-64 32000+0
        "#]]
        .assert_eq(&out);
    }

    /// Decibels make a faint recording cut where a loud one does: the
    /// same hits 40 dB down land on the same frames. Below a 16-bit
    /// recording's dither there is nothing to find, however it wobbles.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_faint_recording_cuts_where_a_loud_one_does() {
        let at = [0, 9_500, 23_000, 31_700];
        let loud = hits(48_000, &at);
        let scaled = |gain: f32| {
            let data = (0..48_000).map(|i| loud.sample(0, i) * gain).collect();
            SampleBuffer::new(vec![data], 48_000)
        };
        let noise = {
            let mut seed = 1u32;
            let data = (0..48_000)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 * 2e-5 - 1e-5
                })
                .collect();
            SampleBuffer::new(vec![data], 48_000)
        };
        expect![[r#"
            loud: [9472, 22912, 31616]
            40 dB down: [9472, 22912, 31616]
            dither: []"#]]
        .assert_eq(&format!(
            "loud: {:?}\n40 dB down: {:?}\ndither: {:?}",
            detect_onsets(&loud, (0, 48_000), 0.5),
            detect_onsets(&scaled(0.01), (0, 48_000), 0.5),
            detect_onsets(&noise, (0, 48_000), 1.0),
        ));
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
               36000  (nearest hit     0 frames away)
        "#]]
        .assert_eq(&report(&onsets, &at));
    }
}
