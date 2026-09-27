//! `gin::NoteDuration`: note lengths a tempo-synced knob steps through.
//!
//! Ported from Gin's `gin_plugin/plugin/gin_noteduration.cpp`.

use crate::juce::jlimit;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteDuration {
    pub name: &'static str,
    bars: f32,
    note: f32,
}

const DOT: f32 = 1.5;
const TRIPLET: f32 = 2.0 / 3.0;

const fn note(name: &'static str, note: f32) -> NoteDuration {
    NoteDuration {
        name,
        bars: 0.0,
        note,
    }
}

const fn bars(name: &'static str, bars: f32) -> NoteDuration {
    NoteDuration {
        name,
        bars,
        note: 0.0,
    }
}

/// `getNoteDurations()`, in order.
pub const DURATIONS: [NoteDuration; 33] = [
    note("1/64t", 0.015_625 * TRIPLET),
    note("1/64", 0.015_625),
    note("1/64d", 0.015_625 * DOT),
    note("1/32t", 0.031_25 * TRIPLET),
    note("1/32", 0.031_25),
    note("1/32d", 0.031_25 * DOT),
    note("1/16t", 0.0625 * TRIPLET),
    note("1/16", 0.0625),
    note("1/16d", 0.0625 * DOT),
    note("1/8t", 0.125 * TRIPLET),
    note("1/8", 0.125),
    note("1/8d", 0.125 * DOT),
    note("1/4t", 0.25 * TRIPLET),
    note("1/4", 0.25),
    note("1/4d", 0.25 * DOT),
    note("1/2t", 0.5 * TRIPLET),
    note("1/2", 0.5),
    note("1/2d", 0.5 * DOT),
    note("1/1t", 1.0 * TRIPLET),
    note("1/1", 1.0),
    note("1/1d", 1.0 * DOT),
    bars("1 bar", 1.0),
    bars("2 bars", 2.0),
    bars("4 bars", 4.0),
    bars("6 bars", 6.0),
    bars("8 bars", 8.0),
    bars("10 bars", 10.0),
    bars("12 bars", 12.0),
    bars("16 bars", 16.0),
    bars("20 bars", 20.0),
    bars("24 bars", 24.0),
    bars("28 bars", 28.0),
    bars("32 bars", 32.0),
];

impl NoteDuration {
    /// `toSeconds (playhead)`, for a host at `bpm` in 4/4.
    #[must_use]
    pub fn to_seconds(self, bpm: f32) -> f32 {
        let numerator = 4.0f32;
        let denominator = 4.0f32;
        let bpm = if bpm == 0.0 { 120.0 } else { bpm };
        let bpm = jlimit(1.0, 500.0, bpm);
        let seconds_per_beat = 1.0 / (bpm / 60.0);
        let seconds_per_whole = seconds_per_beat * denominator;
        if self.note > 0.0 {
            seconds_per_whole * self.note
        } else if self.bars > 0.0 {
            let beat = seconds_per_whole * 1.0 / denominator;
            self.bars * numerator * beat
        } else {
            0.0
        }
    }
}
