//! What a Wavetable's voices are doing, as its screen reads them from a
//! [`Watch`](bloop_plugin_api::Watch): which lane of a voice, and which
//! of the instrument's globals, holds what.
//!
//! A voice's five envelopes — the amplifier's, the filter's and the three
//! modulation envelopes — are each two lanes: the stage it is in
//! ([`Stage`], as a number) and its level, 0 to 1. Its three LFOs and its
//! step LFO are each a lane, where it is in its cycle, 0 to 1; and each
//! oscillator's position in its table, modulation and all, is a lane.
//! The mono LFOs and the mono step LFO, which the whole instrument runs
//! once, are globals: where each is in its cycle.

use bloop_plugin_api::{VoiceWatch, WATCH_GLOBALS, WATCH_LANES};

/// The amplifier's envelope: its stage, then its level.
pub const AMP: usize = 0;
/// The filter's envelope: its stage, then its level.
pub const FILTER: usize = 2;
/// The three modulation envelopes: each its stage, then its level.
pub const ENV: [usize; 3] = [4, 6, 8];
/// The three LFOs: each where it is in its cycle.
pub const LFO: [usize; 3] = [10, 11, 12];
/// The step LFO: where it is in its cycle of steps.
pub const STEP: usize = 13;
/// Each oscillator's position in its table, 0 to 1.
pub const POS: [usize; 2] = [14, 15];

/// The mono LFOs: each where it is in its cycle.
pub const MONO_LFO: [usize; 3] = [0, 1, 2];
/// The mono step LFO: where it is in its cycle of steps.
pub const MONO_STEP: usize = 3;

const _: () = assert!(POS[1] < WATCH_LANES && MONO_STEP < WATCH_GLOBALS);

/// Which stage an envelope is in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    /// Not started, or done.
    #[default]
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl Stage {
    /// The stage a lane holds: the number [`Stage::lane`] wrote, idle for
    /// one it did not.
    #[must_use]
    pub fn of_lane(value: f32) -> Stage {
        match value.round() as i32 {
            1 => Stage::Attack,
            2 => Stage::Decay,
            3 => Stage::Sustain,
            4 => Stage::Release,
            _ => Stage::Idle,
        }
    }

    /// The stage as a lane holds it.
    #[must_use]
    pub fn lane(self) -> f32 {
        match self {
            Stage::Idle => 0.0,
            Stage::Attack => 1.0,
            Stage::Decay => 2.0,
            Stage::Sustain => 3.0,
            Stage::Release => 4.0,
        }
    }
}

/// An envelope as a voice's watch holds it: its stage and its level.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Envelope {
    pub stage: Stage,
    pub level: f32,
}

impl Envelope {
    /// The envelope whose lanes start at `lane`.
    #[must_use]
    pub fn of(voice: &VoiceWatch, lane: usize) -> Envelope {
        Envelope {
            stage: Stage::of_lane(voice.lanes[lane]),
            level: voice.lanes[lane + 1],
        }
    }

    pub(crate) fn write(self, voice: &mut VoiceWatch, lane: usize) {
        voice.lanes[lane] = self.stage.lane();
        voice.lanes[lane + 1] = self.level;
    }
}
