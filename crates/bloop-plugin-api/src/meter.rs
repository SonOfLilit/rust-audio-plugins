//! What a dynamics effect's screen draws, as the effect measures it on the
//! audio thread: the level in, the level out and how far it is turning the
//! signal down.
//!
//! [`LevelTracker`] is Gin's (`gin_dsp/dsp/gin_leveltracker.{h,cpp}`): a
//! peak in decibels, held for 50 ms and then let fall at a fixed rate — or,
//! with a negative rate, a trough held and let rise, which is how Gin's
//! dynamics track gain reduction. Gin clocks the hold and the fall by the
//! wall clock; here they run on the samples the effect has processed, so a
//! meter reads the same in a test, offline and live.

/// How long a peak is held before it starts to fall, in seconds: Gin's 50 ms.
pub const HOLD_SECONDS: f32 = 0.05;

/// The quietest a meter reads, in dB: JUCE's `gainToDecibels` floor.
pub const FLOOR_DB: f32 = -100.0;

/// `juce::Decibels::gainToDecibels` with its default floor.
#[must_use]
pub fn gain_to_db(gain: f32) -> f32 {
    if gain > 0.0 {
        (20.0 * libm::log10f(gain)).max(FLOOR_DB)
    } else {
        FLOOR_DB
    }
}

/// `gin::LevelTracker`: a level in dB that jumps to a louder peak — or,
/// with a negative fall, a quieter trough — holds it [`HOLD_SECONDS`], and
/// then moves back at `fall_db_per_second`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelTracker {
    /// The last peak taken, in dB.
    peak_db: f32,
    /// Seconds of audio since it was taken.
    since: f32,
    /// How fast the level falls after the hold, in dB a second: negative
    /// for a tracker of troughs, which rises.
    fall_db_per_second: f32,
    /// Whether anything over 0 dBFS has been seen since the last
    /// [`LevelTracker::clear_clip`].
    clip: bool,
}

impl LevelTracker {
    /// A tracker of peaks falling at `fall_db_per_second` — Gin's default is
    /// 30 — or of troughs rising at its negation (Gin's reduction meter is
    /// built with −30). A tracker of troughs starts at 0 dB, no reduction;
    /// Gin's starts at its −100 and takes three seconds to rise from it.
    #[must_use]
    pub const fn new(fall_db_per_second: f32) -> Self {
        Self {
            peak_db: if fall_db_per_second < 0.0 { 0.0 } else { FLOOR_DB },
            since: 0.0,
            fall_db_per_second,
            clip: false,
        }
    }

    /// A block of `seconds` whose loudest sample was `peak` (linear,
    /// either sign): `trackBuffer`. A tracker of troughs is handed a gain
    /// instead, the smallest of the block: `trackSample`.
    pub fn track(&mut self, peak: f32, seconds: f32) {
        self.since += seconds;
        let db = gain_to_db(peak.abs());
        if db > 0.0 {
            self.clip = true;
        }
        let level = self.level();
        let replaces = if self.fall_db_per_second < 0.0 {
            db < level
        } else {
            db > level
        };
        if replaces {
            self.peak_db = db;
            self.since = 0.0;
        }
    }

    /// `getLevel`: the peak while it is held, then falling (or rising).
    #[must_use]
    pub fn level(&self) -> f32 {
        if self.since < HOLD_SECONDS {
            self.peak_db
        } else {
            self.peak_db - self.fall_db_per_second * (self.since - HOLD_SECONDS)
        }
    }

    /// Whether a sample over 0 dBFS has gone through since the last clear.
    #[must_use]
    pub fn clipped(&self) -> bool {
        self.clip
    }

    pub fn clear_clip(&mut self) {
        self.clip = false;
    }

    /// Back to silence, as a new one.
    pub fn reset(&mut self) {
        *self = Self::new(self.fall_db_per_second);
    }
}

/// The largest magnitude in `samples`: what `trackBuffer` finds with
/// `findMinAndMax`.
#[must_use]
pub fn peak_of(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |peak, x| peak.max(x.abs()))
}

/// The most bands a [`DynamicsMeter`] carries: the Multiband's three.
pub const METER_BANDS: usize = 3;

/// What a dynamics effect's screen draws, after the block the engine last
/// ran: Gin's three level meters, and a gain reduction per band for an
/// effect that splits the signal. Fixed-size and `Copy`, so the engine can
/// hand it over in its meters without allocating.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicsMeter {
    /// The level going in, in dBFS, held and falling as Gin's input meter.
    pub input_db: f32,
    /// The level coming out, in dBFS, the same way.
    pub output_db: f32,
    /// How far each band is being turned down, in dB, 0 or below; only the
    /// first `bands` mean anything.
    pub reduction_db: [f32; METER_BANDS],
    /// How many bands: 1 for a compressor, gate, limiter or expander, 3 for
    /// the Multiband.
    pub bands: usize,
}

impl Default for DynamicsMeter {
    fn default() -> Self {
        Self {
            input_db: FLOOR_DB,
            output_db: FLOOR_DB,
            reduction_db: [0.0; METER_BANDS],
            bands: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    const RATE: f32 = 48_000.0;
    const BLOCK: usize = 128;

    /// `blocks` blocks of a sine at `db` dBFS through `tracker`.
    fn sine(tracker: &mut LevelTracker, db: f32, blocks: usize) {
        let amplitude = libm::powf(10.0, db / 20.0);
        let mut block = [0.0f32; BLOCK];
        for b in 0..blocks {
            for (i, x) in block.iter_mut().enumerate() {
                let n = (b * BLOCK + i) as f32;
                *x = amplitude * libm::sinf(2.0 * core::f32::consts::PI * 1000.0 * n / RATE);
            }
            tracker.track(peak_of(&block), BLOCK as f32 / RATE);
        }
    }

    /// A 1 kHz sine at −12 dBFS reads −12 on a peak tracker, and holds
    /// there while it plays; silence after it holds the peak 50 ms, then
    /// falls 30 dB a second to the floor. A sample over full scale lights
    /// the clip until it is cleared.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_peak_tracker_holds_then_falls() {
        let mut tracker = LevelTracker::new(30.0);
        let mut out = vec![format!("new: {:.1} dB", tracker.level())];
        sine(&mut tracker, -12.0, 40);
        out.push(format!("a -12 dBFS sine, 107 ms: {:.2} dB", tracker.level()));
        let silence = BLOCK as f32 / RATE;
        let mut t = 0.0;
        for until in [0.04, 0.1, 0.55, 3.0, 6.0] {
            while t < until {
                tracker.track(0.0, silence);
                t += silence;
            }
            out.push(format!("silence, {t:.3} s: {:.2} dB", tracker.level()));
        }
        tracker.track(1.5, silence);
        out.push(format!("1.5 in: {:.2} dB, clip {}", tracker.level(), tracker.clipped()));
        tracker.clear_clip();
        out.push(format!("cleared: clip {}", tracker.clipped()));
        expect![[r#"
            new: -100.0 dB
            a -12 dBFS sine, 107 ms: -12.00 dB
            silence, 0.040 s: -12.00 dB
            silence, 0.101 s: -13.62 dB
            silence, 0.552 s: -27.14 dB
            silence, 3.000 s: -100.00 dB
            silence, 6.003 s: -100.00 dB
            1.5 in: 3.52 dB, clip true
            cleared: clip false"#]].assert_eq(&out.join("\n"));
    }

    /// Gin's reduction meter: a tracker of troughs, handed the smallest
    /// gain of each block. It drops at once to the deepest reduction, holds
    /// 50 ms, then rises 30 dB a second back to none.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_trough_tracker_holds_then_rises() {
        let mut tracker = LevelTracker::new(-30.0);
        let block = BLOCK as f32 / RATE;
        let mut out = vec![];
        tracker.track(1.0, block);
        out.push(format!("no reduction: {:.2} dB", tracker.level()));
        tracker.track(libm::powf(10.0, -12.0 / 20.0), block);
        out.push(format!("-12 dB of reduction: {:.2} dB", tracker.level()));
        tracker.track(libm::powf(10.0, -6.0 / 20.0), block);
        out.push(format!("then -6: {:.2} dB", tracker.level()));
        let mut t = block;
        for until in [0.04, 0.1, 0.3, 0.5] {
            while t < until {
                tracker.track(1.0, block);
                t += block;
            }
            out.push(format!("none again, {t:.3} s: {:.2} dB", tracker.level()));
        }
        expect![[r#"
            no reduction: 0.00 dB
            -12 dB of reduction: -12.00 dB
            then -6: -12.00 dB
            none again, 0.040 s: -12.00 dB
            none again, 0.101 s: -10.46 dB
            none again, 0.301 s: -4.46 dB
            none again, 0.501 s: 0.00 dB"#]].assert_eq(&out.join("\n"));
    }
}
