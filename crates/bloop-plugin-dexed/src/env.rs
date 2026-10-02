//! DX7 operator envelope generation.
//!
//! Port of `env.cc`/`env.h` from Dexed's msfa fork (Copyright 2017 Pascal
//! Gauthier, Copyright 2012 Google Inc., Apache-2.0) — the "accurate
//! envelope" variant with static-segment timing measured from real DX7
//! hardware (the C++ `ACCURATE_ENVELOPE` mode).

use crate::tables::Tables;
use bloop_plugin_api::EnvStage;

const LEVEL_LUT: [i32; 20] = [
    0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46,
];

/// Number of samples at 44.1 kHz a "static" (level-holding) envelope segment
/// lasts, indexed by qRate. Measured empirically on DX7 hardware (TF1s) by
/// the Dexed project.
const STATICS: [i32; 77] = [
    1764000, 1764000, 1411200, 1411200, 1190700, 1014300, 992250, 882000, 705600, 705600, 584325,
    507150, 502740, 441000, 418950, 352800, 308700, 286650, 253575, 220500, 220500, 176400, 145530,
    145530, 125685, 110250, 110250, 88200, 88200, 74970, 61740, 61740, 55125, 48510, 44100, 37485,
    31311, 30870, 27562, 27562, 22050, 18522, 17640, 15435, 14112, 13230, 11025, 9261, 9261, 7717,
    6615, 6615, 5512, 5512, 4410, 3969, 3969, 3439, 2866, 2690, 2249, 1984, 1896, 1808, 1411, 1367,
    1234, 1146, 926, 837, 837, 705, 573, 573, 529, 441, 441,
];

/// One operator's 4-stage rate/level envelope. Levels are Q24-doubling log
/// gain (2^24 = one doubling); `getsample` is subsampled once per `N`-sample
/// render chunk.
#[derive(Clone, Copy, Default)]
pub struct Env {
    initialised: bool,
    rates: [i32; 4],
    levels: [i32; 4],
    outlevel: i32,
    rate_scaling: i32,
    level: i32,
    targetlevel: i32,
    rising: bool,
    ix: usize,
    inc: i32,
    staticcount: i32,
    down: bool,
    /// Where the current segment set out from: the internal level, and
    /// what `level_units` read there. Only for reading the level back.
    from_level: i32,
    from_units: f32,
}

/// The internal level every target is clamped up to (`advance`).
const FLOOR: i32 = 16 << 16;

impl Env {
    /// `rates`/`levels` are DX7 parameters (0..99). `outlevel` is in
    /// microsteps (~0.023 dB, 99*32 nominal full scale); `rate_scaling` is
    /// in qRate units (0..63).
    pub fn init(
        &mut self,
        t: &Tables,
        rates: [i32; 4],
        levels: [i32; 4],
        outlevel: i32,
        rate_scaling: i32,
    ) {
        self.initialised = true;
        self.rates = rates;
        self.levels = levels;
        self.outlevel = outlevel;
        self.rate_scaling = rate_scaling;
        self.level = 0;
        self.down = true;
        self.advance(t, 0);
        // The attack sets out from silence, which reads as the floor.
        self.from_level = FLOOR;
        self.from_units = self.units_of(FLOOR);
    }

    /// Next envelope value in Q24-doubling log format.
    pub fn getsample(&mut self, t: &Tables) -> i32 {
        if self.staticcount != 0 {
            self.staticcount -= crate::fm_core::N as i32;
            if self.staticcount <= 0 {
                self.staticcount = 0;
                self.advance(t, self.ix + 1);
            }
        }

        if self.ix < 3 || (self.ix < 4 && !self.down) {
            if self.staticcount != 0 {
                // Holding a measured static segment: level stays put.
            } else if self.rising {
                const JUMP_TARGET: i32 = 1716;
                if self.level < (JUMP_TARGET << 16) {
                    self.level = JUMP_TARGET << 16;
                }
                self.level += (((17 << 24) - self.level) >> 24) * self.inc;
                if self.level >= self.targetlevel {
                    self.level = self.targetlevel;
                    self.advance(t, self.ix + 1);
                }
            } else {
                self.level -= self.inc;
                if self.level <= self.targetlevel {
                    self.level = self.targetlevel;
                    self.advance(t, self.ix + 1);
                }
            }
        }
        self.level
    }

    pub fn keydown(&mut self, t: &Tables, down: bool) {
        if self.down != down {
            self.down = down;
            self.advance(t, if down { 0 } else { 3 });
        }
    }

    pub fn scaleoutlevel(outlevel: i32) -> i32 {
        if outlevel >= 20 {
            28 + outlevel
        } else {
            LEVEL_LUT[outlevel as usize]
        }
    }

    fn advance(&mut self, t: &Tables, newix: usize) {
        // Once per segment, not per sample: so the read starts where the
        // last one left off.
        self.from_units = self.level_units();
        self.from_level = self.level;
        self.ix = newix;
        if self.ix < 4 {
            let newlevel = self.levels[self.ix];
            self.targetlevel = self.internal_level(newlevel);
            self.rising = self.targetlevel > self.level;

            let mut qrate = (self.rates[self.ix] * 41) >> 6;
            qrate += self.rate_scaling;
            qrate = qrate.min(63);

            if self.targetlevel == self.level || (self.ix == 0 && newlevel == 0) {
                let mut staticrate = self.rates[self.ix];
                staticrate += self.rate_scaling;
                staticrate = staticrate.min(99);
                self.staticcount = if staticrate < 77 {
                    STATICS[staticrate as usize]
                } else {
                    20 * (99 - staticrate)
                };
                if staticrate < 77 && self.ix == 0 && newlevel == 0 {
                    self.staticcount /= 20; // attack is scaled faster
                }
                self.staticcount =
                    ((i64::from(self.staticcount) * i64::from(t.env_sr_multiplier)) >> 24) as i32;
            } else {
                self.staticcount = 0;
            }
            self.inc = (4 + (qrate & 3)) << (2 + crate::fm_core::LG_N + (qrate >> 2));
            self.inc = ((i64::from(self.inc) * i64::from(t.env_sr_multiplier)) >> 24) as i32;
        }
    }

    /// The internal level (Q16) a patch level (0..99) is at for this
    /// envelope's output level: what a segment heading for it targets.
    fn internal_level(&self, level: i32) -> i32 {
        let mut actuallevel = Self::scaleoutlevel(level) >> 1;
        actuallevel = (actuallevel << 6) + self.outlevel - 4256;
        actuallevel = actuallevel.max(16);
        actuallevel << 16
    }

    /// An envelope is active if it has been initialised and either has not
    /// reached L4 yet or L4 > 0 (a DX7 envelope can sustain forever).
    pub fn is_active(&self) -> bool {
        self.initialised && (self.ix < 4 || self.levels[3] > 0)
    }

    /// Which segment the envelope is in, for a screen.
    pub fn stage(&self) -> EnvStage {
        match self.ix {
            0 => EnvStage::Rate1,
            1 => EnvStage::Rate2,
            2 => EnvStage::Rate3,
            // Index 3 heads for L4, but only moves once the key is up:
            // while it is down the envelope sits where stage 2 left it.
            3 if self.down => EnvStage::Sustain,
            3 => EnvStage::Rate4,
            // L4 the same as L3 leaves rate 4 nothing to do, and the
            // engine runs on to the end with the key still down: that is
            // still the sustain.
            _ if self.down => EnvStage::Sustain,
            _ => EnvStage::End,
        }
    }

    /// Where the envelope is, in the patch's 0..99 level units.
    ///
    /// Exact wherever the envelope holds a level of the patch: at L3 in
    /// sustain, at L4 at the end, and at a level it waits at (a static
    /// segment). Between levels, `level` read back through `advance`'s
    /// mapping.
    ///
    /// The mapping drops a bit (`scaleoutlevel(L) >> 1`): L63 and L62 are
    /// one internal level. So the read is corrected at both ends of the
    /// segment — to start where the last read left off and to arrive
    /// exactly on the level it heads for — in proportion to how far along
    /// it is. A dot glides from level to level without a step.
    ///
    /// The floor `advance` clamps every target to (16) is where any level
    /// too quiet for this operator's output level ends up. A segment
    /// heading there is corrected the same way, so the dot carries on
    /// down past the floor's own reading and arrives on the level the
    /// patch asks for. Waiting at the floor to rise, the level is only
    /// known to be at most the floor's: the read is the lowest of that,
    /// where the envelope came from, and where it is going.
    pub fn level_units(&self) -> f32 {
        let held = match self.ix {
            ix if ix >= 3 && self.down => Some(2),
            ix if ix >= 4 => Some(3),
            ix if self.staticcount != 0 => Some(ix),
            _ => None,
        };
        let read = match held {
            Some(ix) => self.levels[ix] as f32,
            None => {
                let heading = self.levels[self.ix] as f32;
                let read = self.units_of(self.level.max(FLOOR));
                if self.level <= FLOOR {
                    // About to rise from the floor: no higher than where
                    // it came from, nor than where it is going.
                    read.min(self.from_units).min(heading)
                } else {
                    let from = self.from_level.max(FLOOR);
                    let at_start = self.from_units - self.units_of(from);
                    let at_end = heading - self.units_of(self.targetlevel);
                    let span = self.targetlevel - from;
                    let along = if span == 0 {
                        1.0
                    } else {
                        ((self.level - from) as f32 / span as f32).clamp(0.0, 1.0)
                    };
                    read + at_start + (at_end - at_start) * along
                }
            }
        };
        read.clamp(0.0, 99.0)
    }

    /// `advance`'s level mapping inverted: an internal Q16 level back to
    /// 0..99 level units, given this envelope's output level.
    fn units_of(&self, level: i32) -> f32 {
        // advance: level = (scaleoutlevel(L) >> 1) * 64 + outlevel - 4256.
        let scaled = (level as f32 / 65536.0 - self.outlevel as f32 + 4256.0) / 32.0;
        unscale_level(scaled)
    }

    /// Current stage index (0-3 = stages, 4 = finished) — for tests.
    #[cfg(test)]
    pub fn position(&self) -> usize {
        self.ix
    }
}

/// `Env::scaleoutlevel` inverted, continuously: 0..=127 back to 0..=99,
/// linear between the table's points below 20.
fn unscale_level(scaled: f32) -> f32 {
    if scaled >= 48.0 {
        return scaled - 28.0;
    }
    // Below 20 the curve is LEVEL_LUT, and 19 -> 46 joins 20 -> 48.
    let mut below = (0.0, 0.0);
    for (level, at) in LEVEL_LUT.iter().copied().chain([48]).enumerate() {
        let (level, at) = (level as f32, at as f32);
        if scaled <= at {
            let (from_level, from_at) = below;
            let span = at - from_at;
            let t = if span > 0.0 {
                (scaled - from_at) / span
            } else {
                0.0
            };
            return from_level + t * (level - from_level);
        }
        below = (level, at);
    }
    scaled - 28.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Drive one envelope: key down at chunk 0, key up at `keyup_at`.
    /// Snapshot stage transitions (exact chunk) plus the level every
    /// `stride` chunks. One chunk = 64 samples ≈ 1.45ms at 44.1kHz.
    /// Levels print in the engine's Q16 "actuallevel" domain (0..~4400,
    /// higher = louder; 0 = silent).
    #[allow(clippy::too_many_arguments)] // scenario eDSL, mirrors the DX7 param set
    fn check_env(
        rates: [i32; 4],
        levels: [i32; 4],
        outlevel99: i32,
        rate_scaling: i32,
        keyup_at: usize,
        chunks: usize,
        stride: usize,
        expect: Expect,
    ) {
        let t = Tables::new(44100.0);
        let mut env = Env::default();
        // Same outlevel staging as dx7note::init (no velocity, no scaling).
        let outlevel = Env::scaleoutlevel(outlevel99).min(127) << 5;
        env.init(&t, rates, levels, outlevel, rate_scaling);
        let mut lines = Vec::new();
        let mut stage = env.position();
        for chunk in 0..chunks {
            if chunk == keyup_at {
                env.keydown(&t, false);
                lines.push(format!("chunk {chunk:>4}: keyup"));
            }
            let level = env.getsample(&t);
            if env.position() != stage {
                stage = env.position();
                lines.push(format!(
                    "chunk {chunk:>4}: -> stage {stage} level={}",
                    level >> 16
                ));
            }
            if chunk % stride == 0 {
                lines.push(format!("chunk {chunk:>4}: level={}", level >> 16));
            }
        }
        lines.push(format!("active={}", env.is_active()));
        expect.assert_eq(&lines.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn fast_attack_full_sustain_release() {
        // R99 attack to L99, sustain at L99, release R50 on keyup.
        check_env(
            [99, 99, 99, 50],
            [99, 99, 99, 0],
            99,
            0,
            120,
            240,
            30,
            expect![[r#"
                chunk    0: -> stage 1 level=3840
                chunk    0: level=3840
                chunk    1: -> stage 2 level=3840
                chunk    2: -> stage 3 level=3840
                chunk   30: level=3840
                chunk   60: level=3840
                chunk   90: level=3840
                chunk  120: keyup
                chunk  120: level=3836
                chunk  150: level=3716
                chunk  180: level=3596
                chunk  210: level=3476
                active=true"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn slow_attack_rises_smoothly() {
        // R40 attack: the level must rise monotonically over many chunks.
        check_env(
            [40, 99, 99, 99],
            [99, 99, 99, 0],
            99,
            0,
            400,
            420,
            40,
            expect![[r#"
                chunk    0: level=1728
                chunk   40: level=2171
                chunk   80: level=2538
                chunk  120: level=2838
                chunk  160: level=3086
                chunk  200: level=3286
                chunk  240: level=3447
                chunk  280: level=3593
                chunk  320: level=3693
                chunk  360: level=3793
                chunk  379: -> stage 1 level=3840
                chunk  380: -> stage 2 level=3840
                chunk  381: -> stage 3 level=3840
                chunk  400: keyup
                chunk  400: level=2944
                chunk  404: -> stage 4 level=16
                active=false"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn percussive_decays_while_held() {
        // L2=L3=0: decays to silence with the key still down (marimba
        // style), and the envelope reports inactive at the end.
        check_env(
            [99, 60, 99, 99],
            [99, 0, 0, 0],
            99,
            0,
            2000,
            2100,
            250,
            expect![[r#"
                chunk    0: -> stage 1 level=3840
                chunk    0: level=3840
                chunk  250: level=840
                chunk  319: -> stage 2 level=16
                chunk  320: -> stage 3 level=16
                chunk  500: level=16
                chunk  750: level=16
                chunk 1000: level=16
                chunk 1250: level=16
                chunk 1500: level=16
                chunk 1750: level=16
                chunk 2000: keyup
                chunk 2000: -> stage 4 level=16
                chunk 2000: level=16
                active=false"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn rate_scaling_shortens_decay() {
        // Same envelope, rate_scaling 40 qRate: everything runs faster.
        check_env(
            [99, 60, 99, 99],
            [99, 0, 0, 0],
            99,
            40,
            2000,
            2100,
            250,
            expect![[r#"
                chunk    0: -> stage 1 level=3840
                chunk    0: level=3840
                chunk    5: -> stage 2 level=16
                chunk    6: -> stage 3 level=16
                chunk  250: level=16
                chunk  500: level=16
                chunk  750: level=16
                chunk 1000: level=16
                chunk 1250: level=16
                chunk 1500: level=16
                chunk 1750: level=16
                chunk 2000: keyup
                chunk 2000: -> stage 4 level=16
                chunk 2000: level=16
                active=false"#]],
        );
    }

    /// Every level a segment can head for, read back to patch units from
    /// the internal level `advance` targets, at three output levels. The
    /// read is what a dot between two levels is drawn at, so it should be
    /// the level itself: exact for an even level of 20 and up, one below
    /// an odd one (the mapping drops that bit), near it below 20. A level
    /// too quiet for the operator's output level sits on the floor (_),
    /// where all that is known is "at most this".
    ///
    /// After the slash, what `level_units` says the instant the key goes
    /// down, before the first chunk: the envelope is below its floor, so
    /// the read is the lower of the floor's and L1 — the target where that
    /// is the quieter, 0 where L1 is 0 (a static segment, held exactly).
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn levels_read_back_through_the_mapping() {
        let t = Tables::new(44100.0);
        let mut lines = Vec::new();
        for output in [99, 70, 30] {
            let outlevel = Env::scaleoutlevel(output).min(127) << 5;
            let reads: Vec<String> = [0, 1, 5, 10, 15, 19, 20, 21, 40, 62, 63, 80, 98, 99]
                .iter()
                .map(|&level| {
                    let mut env = Env::default();
                    env.init(&t, [50, 50, 50, 50], [level, 0, 0, 0], outlevel, 0);
                    let floor = if env.targetlevel <= 16 << 16 { "_" } else { "" };
                    format!(
                        "{level}>{:.1}{floor}/{:.1}",
                        env.units_of(env.targetlevel),
                        env.level_units()
                    )
                })
                .collect();
            lines.push(format!("OL{output}: {}", reads.join(" ")));
        }
        expect![[r#"
            OL99: 0>1.4_/0.0 1>1.4_/1.0 5>5.0/1.4 10>9.5/1.4 15>14.5/1.4 19>19.0/1.4 20>20.0/1.4 21>20.0/1.4 40>40.0/1.4 62>62.0/1.4 63>62.0/1.4 80>80.0/1.4 98>98.0/1.4 99>98.0/1.4
            OL70: 0>12.2_/0.0 1>12.2_/1.0 5>12.2_/5.0 10>12.2_/10.0 15>14.5/12.2 19>19.0/12.2 20>20.0/12.2 21>20.0/12.2 40>40.0/12.2 62>62.0/12.2 63>62.0/12.2 80>80.0/12.2 98>98.0/12.2 99>98.0/12.2
            OL30: 0>47.5_/0.0 1>47.5_/1.0 5>47.5_/5.0 10>47.5_/10.0 15>47.5_/15.0 19>47.5_/19.0 20>47.5_/20.0 21>47.5_/21.0 40>47.5_/40.0 62>62.0/47.5 63>62.0/47.5 80>80.0/47.5 98>98.0/47.5 99>98.0/47.5"#]].assert_eq(&lines.join("\n"));
    }

    /// Run one envelope at output level `output` for `chunks`, letting go
    /// at `keyup_at`, and print
    /// its stage and level whenever the stage changes, for the three
    /// chunks after that, and whenever the level crosses a multiple of 10.
    fn check_stages(output: i32, levels: [i32; 4], keyup_at: usize, chunks: usize, expect: Expect) {
        let t = Tables::new(44100.0);
        let mut env = Env::default();
        let outlevel = Env::scaleoutlevel(output) << 5;
        env.init(&t, [70, 50, 50, 50], levels, outlevel, 0);
        let mut lines = vec![format!(
            "struck   : {:?} {:.1}",
            env.stage(),
            env.level_units()
        )];
        let (mut stage, mut tenth, mut since) = (env.stage(), -1, 0);
        for chunk in 0..chunks {
            if chunk == keyup_at {
                env.keydown(&t, false);
                lines.push(format!("chunk {chunk:>4}: keyup"));
            }
            env.getsample(&t);
            let level = env.level_units();
            since = if env.stage() == stage { since + 1 } else { 0 };
            let now = (level / 10.0) as i32;
            if since < 3 || now != tenth {
                lines.push(format!("chunk {chunk:>4}: {:?} {level:.1}", env.stage()));
            }
            (stage, tenth) = (env.stage(), now);
        }
        expect.assert_eq(&lines.join("\n"));
    }

    /// The stage follows the segment; the level is exact while it holds
    /// (sustain, end), moves in between, and arrives on each level
    /// without a step — even leaving the odd L1=99, which is one internal
    /// level with 98.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn stage_and_level_follow_the_envelope() {
        check_stages(
            99,
            [99, 70, 41, 10],
            1000,
            1350,
            expect![[r#"
            struck   : Rate1 1.4
            chunk    0: Rate1 42.2
            chunk    1: Rate1 51.2
            chunk    3: Rate1 65.3
            chunk    4: Rate1 71.4
            chunk    6: Rate1 80.5
            chunk    9: Rate1 90.6
            chunk   13: Rate2 99.0
            chunk   14: Rate2 98.9
            chunk   15: Rate2 98.7
            chunk   83: Rate2 89.9
            chunk  160: Rate2 80.0
            chunk  237: Rate3 70.0
            chunk  238: Rate3 69.9
            chunk  239: Rate3 69.8
            chunk  320: Rate3 60.0
            chunk  403: Rate3 49.9
            chunk  477: Sustain 41.0
            chunk  478: Sustain 41.0
            chunk  479: Sustain 41.0
            chunk 1000: keyup
            chunk 1000: Rate4 40.9
            chunk 1001: Rate4 40.7
            chunk 1002: Rate4 40.6
            chunk 1007: Rate4 40.0
            chunk 1086: Rate4 30.0
            chunk 1171: Rate4 20.0
            chunk 1303: End 10.0
            chunk 1304: End 10.0
            chunk 1305: End 10.0"#]],
        );
    }

    /// A quieter operator, whose floor is higher up its scale: at output
    /// level 50 the floor reads about 27, so the release from L3=80 to
    /// L4=0 heads for a level the engine clamps to the floor. The dot
    /// still glides all the way down to 0 and arrives there, rather than
    /// stopping at the floor's reading and jumping.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_release_below_the_floor_glides_to_its_level() {
        check_stages(
            50,
            [99, 70, 80, 0],
            700,
            1400,
            expect![[r#"
            struck   : Rate1 27.5
            chunk    0: Rate1 91.5
            chunk    1: Rate2 99.0
            chunk    2: Rate2 98.9
            chunk    3: Rate2 98.7
            chunk   71: Rate2 89.9
            chunk  148: Rate2 80.0
            chunk  225: Rate3 70.0
            chunk  226: Sustain 80.0
            chunk  227: Sustain 80.0
            chunk  228: Sustain 80.0
            chunk  700: keyup
            chunk  700: Rate4 79.8
            chunk  701: Rate4 79.6
            chunk  702: Rate4 79.4
            chunk  752: Rate4 69.9
            chunk  805: Rate4 59.8
            chunk  857: Rate4 49.9
            chunk  910: Rate4 39.8
            chunk  962: Rate4 29.9
            chunk 1015: Rate4 19.8
            chunk 1067: Rate4 9.9
            chunk 1119: End 0.0
            chunk 1120: End 0.0
            chunk 1121: End 0.0"#]],
        );
    }

    /// At output level 30 the floor reads 47.5: the decay to L2=40 heads
    /// below it and must arrive on 40, and the climb to L3=70 sets out
    /// from there.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_decay_below_the_floor_glides_to_its_level() {
        check_stages(
            30,
            [99, 40, 70, 0],
            700,
            1400,
            expect![[r#"
            struck   : Rate1 47.5
            chunk    0: Rate2 99.0
            chunk    1: Rate2 98.9
            chunk    2: Rate2 98.7
            chunk   62: Rate2 89.9
            chunk  131: Rate2 79.9
            chunk  199: Rate2 69.9
            chunk  268: Rate2 59.9
            chunk  336: Rate2 49.9
            chunk  404: Rate3 40.0
            chunk  405: Sustain 70.0
            chunk  406: Sustain 70.0
            chunk  407: Sustain 70.0
            chunk  700: keyup
            chunk  700: Rate4 69.6
            chunk  701: Rate4 69.2
            chunk  702: Rate4 68.8
            chunk  725: Rate4 59.9
            chunk  751: Rate4 49.8
            chunk  777: Rate4 39.7
            chunk  802: Rate4 29.9
            chunk  828: Rate4 19.8
            chunk  854: Rate4 9.7
            chunk  879: End 0.0
            chunk  880: End 0.0
            chunk  881: End 0.0"#]],
        );
    }

    /// With L3 equal to L4 the envelope has nowhere to go after its third
    /// rate, and the engine moves on to its end with the key still down.
    /// That is still the sustain, and reads so — until the key comes up.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_held_key_sustains_when_l3_is_l4() {
        check_stages(
            99,
            [99, 99, 60, 60],
            900,
            1000,
            expect![[r#"
            struck   : Rate1 1.4
            chunk    0: Rate1 42.2
            chunk    1: Rate1 51.2
            chunk    3: Rate1 65.3
            chunk    4: Rate1 71.4
            chunk    6: Rate1 80.5
            chunk    9: Rate1 90.6
            chunk   13: Rate2 99.0
            chunk   14: Rate2 99.0
            chunk   15: Rate2 99.0
            chunk  117: Rate3 98.9
            chunk  118: Rate3 98.7
            chunk  119: Rate3 98.6
            chunk  187: Rate3 89.9
            chunk  265: Rate3 79.9
            chunk  343: Rate3 69.9
            chunk  420: Sustain 60.0
            chunk  421: Sustain 60.0
            chunk  422: Sustain 60.0
            chunk  900: keyup
            chunk  900: Rate4 60.0
            chunk  901: Rate4 60.0
            chunk  902: Rate4 60.0"#]],
        );
    }

    /// A key let go mid-attack releases from where the attack got to, not
    /// from L3.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_key_let_go_mid_attack_releases_from_there() {
        check_stages(
            99,
            [99, 70, 40, 10],
            5,
            320,
            expect![[r#"
            struck   : Rate1 1.4
            chunk    0: Rate1 42.2
            chunk    1: Rate1 51.2
            chunk    3: Rate1 65.3
            chunk    4: Rate1 71.4
            chunk    5: keyup
            chunk    5: Rate4 71.3
            chunk    6: Rate4 71.1
            chunk    7: Rate4 71.0
            chunk   16: Rate4 69.9
            chunk   95: Rate4 60.0
            chunk  175: Rate4 49.9
            chunk  255: Rate4 39.9"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn scaleoutlevel_curve() {
        let levels: Vec<String> = (0..100)
            .step_by(11)
            .map(|l| format!("{l} -> {}", Env::scaleoutlevel(l)))
            .collect();
        expect![[r#"
            0 -> 0
            11 -> 33
            22 -> 50
            33 -> 61
            44 -> 72
            55 -> 83
            66 -> 94
            77 -> 105
            88 -> 116
            99 -> 127"#]]
        .assert_eq(&levels.join("\n"));
    }
}
