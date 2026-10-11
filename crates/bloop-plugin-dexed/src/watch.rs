//! What the FM synth's voices say to a screen, and where in a voice's
//! lanes ([`bloop_plugin_api::VoiceWatch::lanes`]) each of it goes: the
//! API carries numbers, and this is what the FM synth's numbers mean.
//!
//! Lanes 0–5 are operators 1–6's envelope levels, 6–11 their gains, and
//! 12–17 the stages their envelopes are in.

use bloop_plugin_api::{VoiceWatch, WATCH_LANES};

/// How many operators a voice has.
pub const OPS: usize = 6;
/// The first lane of each of an operator's three numbers.
pub const LEVEL_LANES: usize = 0;
pub const AMP_LANES: usize = OPS;
pub const STAGE_LANES: usize = 2 * OPS;

const _: () = assert!(
    STAGE_LANES + OPS <= WATCH_LANES,
    "a voice has room for every operator"
);

/// Where one operator's envelope is, as a four-rate, four-level envelope
/// walks it: rate 1 toward level 1, rate 2 toward level 2, rate 3 toward
/// level 3, held at level 3 while the key is down, then rate 4 toward
/// level 4 once it is let go — from wherever it had got to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnvStage {
    /// Key down, moving toward level 1 at rate 1: the attack.
    #[default]
    Rate1,
    /// Key down, moving toward level 2 at rate 2.
    Rate2,
    /// Key down, moving toward level 3 at rate 3.
    Rate3,
    /// Key down, at level 3, and staying there until the key comes up.
    Sustain,
    /// Key up, moving toward level 4 at rate 4: the release.
    Rate4,
    /// At level 4, where it stays: silence when level 4 is 0, a note that
    /// rings on for ever when it is not.
    End,
}

impl EnvStage {
    const ALL: [EnvStage; 6] = [
        EnvStage::Rate1,
        EnvStage::Rate2,
        EnvStage::Rate3,
        EnvStage::Sustain,
        EnvStage::Rate4,
        EnvStage::End,
    ];

    /// The stage as a lane's number: 0 for Rate 1 to 5 for the end.
    #[must_use]
    pub fn lane(self) -> f32 {
        match self {
            EnvStage::Rate1 => 0.0,
            EnvStage::Rate2 => 1.0,
            EnvStage::Rate3 => 2.0,
            EnvStage::Sustain => 3.0,
            EnvStage::Rate4 => 4.0,
            EnvStage::End => 5.0,
        }
    }

    /// The stage a lane's number says: the nearest, Rate 1 below 0 and
    /// the end past 5.
    #[must_use]
    pub fn of_lane(lane: f32) -> EnvStage {
        let at = lane.clamp(0.0, 5.0).round();
        EnvStage::ALL
            .into_iter()
            .find(|stage| stage.lane() == at)
            .unwrap_or_default()
    }
}

/// One operator of a sounding voice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpWatch {
    pub stage: EnvStage,
    /// Where the envelope is, in the patch's own level units — the scale
    /// its levels are written in — so a dot drawn at it sits on the
    /// envelope the patch draws. Exactly the patch's
    /// level while the envelope holds one (`Sustain`, `End`, and a level
    /// the envelope is waiting at); between levels, the level the engine's
    /// own envelope is at, read back to that scale.
    ///
    /// The envelope is the one the voice took at its note-on: an edit made
    /// while it sounds is heard from the next note, and so is drawn there.
    /// The operator's output level, velocity and keyboard scaling are not
    /// in it — those are in `amp` — except that an envelope's attack
    /// starts from a fixed loudness, and so from higher on its own scale
    /// the quieter the operator is.
    pub level: f32,
    /// How loud the operator is: its envelope with every scaling applied
    /// (output level, velocity, keyboard level scaling, amplitude
    /// modulation), as a linear gain where 1.0 is an operator at full
    /// output level and full envelope level struck at a velocity its
    /// sensitivity leaves alone. A loud hit on a velocity-sensitive
    /// operator can go past 1.0.
    ///
    /// The gain at the end of the last chunk the engine rendered. Exactly
    /// 0.0 when, at that chunk's end, the engine would not render the
    /// operator — switched off, at output level 0, or below the level it
    /// skips — and for a voice struck too late in the block to have
    /// rendered yet.
    ///
    /// This is the gain the operator's sine is multiplied by. In an
    /// algorithm where it modulates another operator, that is its
    /// modulation depth rather than anything heard directly.
    pub amp: f32,
}

impl OpWatch {
    /// Operator `op` (1–6) of `voice`, read out of its lanes. An operator
    /// outside 1–6 reads as operator 1 or 6.
    #[must_use]
    pub fn of(voice: &VoiceWatch, op: u8) -> OpWatch {
        let n = usize::from(op.clamp(1, 6) - 1);
        OpWatch {
            stage: EnvStage::of_lane(voice.lanes[STAGE_LANES + n]),
            level: voice.lanes[LEVEL_LANES + n],
            amp: voice.lanes[AMP_LANES + n],
        }
    }

    /// Write this as operator `op` (1–6) of `voice`'s lanes.
    pub fn write(self, voice: &mut VoiceWatch, op: u8) {
        let n = usize::from(op.clamp(1, 6) - 1);
        voice.lanes[STAGE_LANES + n] = self.stage.lane();
        voice.lanes[LEVEL_LANES + n] = self.level;
        voice.lanes[AMP_LANES + n] = self.amp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Each operator's three numbers go into lanes of their own and come
    /// back out as they went in; a stage between two reads as the nearer,
    /// and one out of range as the nearer end.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_operator_goes_into_its_lanes_and_back() {
        let mut voice = VoiceWatch::default();
        for (op, stage) in (1..=6u8).zip(EnvStage::ALL) {
            OpWatch {
                stage,
                level: f32::from(op) * 10.0,
                amp: f32::from(op) / 10.0,
            }
            .write(&mut voice, op);
        }
        let back: Vec<String> = (1..=6u8)
            .map(|op| format!("{op}: {:?}", OpWatch::of(&voice, op)))
            .collect();
        let stages: Vec<String> = [-3.0, 0.4, 2.6, 4.5, 9.0]
            .iter()
            .map(|lane| format!("{lane} → {:?}", EnvStage::of_lane(*lane)))
            .collect();
        expect![[r#"
            lanes [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            1: OpWatch { stage: Rate1, level: 10.0, amp: 0.1 }
            2: OpWatch { stage: Rate2, level: 20.0, amp: 0.2 }
            3: OpWatch { stage: Rate3, level: 30.0, amp: 0.3 }
            4: OpWatch { stage: Sustain, level: 40.0, amp: 0.4 }
            5: OpWatch { stage: Rate4, level: 50.0, amp: 0.5 }
            6: OpWatch { stage: End, level: 60.0, amp: 0.6 }
            -3 → Rate1
            0.4 → Rate1
            2.6 → Sustain
            4.5 → End
            9 → End"#]].assert_eq(&format!(
            "lanes {:?}\n{}\n{}",
            voice.lanes,
            back.join("\n"),
            stages.join("\n")
        ));
    }
}
