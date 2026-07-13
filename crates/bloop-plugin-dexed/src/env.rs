//! DX7 operator envelope generation.
//!
//! Port of `env.cc`/`env.h` from Dexed's msfa fork (Copyright 2017 Pascal
//! Gauthier, Copyright 2012 Google Inc., Apache-2.0) — the "accurate
//! envelope" variant with static-segment timing measured from real DX7
//! hardware (the C++ `ACCURATE_ENVELOPE` mode).

use crate::tables::Tables;

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
}

impl Env {
    /// `rates`/`levels` are DX7 parameters (0..99). `outlevel` is in
    /// microsteps (~0.023 dB, 99*32 nominal full scale); `rate_scaling` is
    /// in qRate units (0..63).
    pub fn init(&mut self, t: &Tables, rates: [i32; 4], levels: [i32; 4], outlevel: i32, rate_scaling: i32) {
        self.initialised = true;
        self.rates = rates;
        self.levels = levels;
        self.outlevel = outlevel;
        self.rate_scaling = rate_scaling;
        self.level = 0;
        self.down = true;
        self.advance(t, 0);
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
        self.ix = newix;
        if self.ix < 4 {
            let newlevel = self.levels[self.ix];
            let mut actuallevel = Self::scaleoutlevel(newlevel) >> 1;
            actuallevel = (actuallevel << 6) + self.outlevel - 4256;
            actuallevel = actuallevel.max(16);
            self.targetlevel = actuallevel << 16;
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

    /// An envelope is active if it has been initialised and either has not
    /// reached L4 yet or L4 > 0 (a DX7 envelope can sustain forever).
    pub fn is_active(&self) -> bool {
        self.initialised && (self.ix < 4 || self.levels[3] > 0)
    }

    /// Current stage index (0-3 = stages, 4 = finished) — for tests.
    #[cfg(test)]
    pub fn position(&self) -> usize {
        self.ix
    }
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
            expect![[""]],
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
            expect![[""]],
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
            expect![[""]],
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
            expect![[""]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn scaleoutlevel_curve() {
        let levels: Vec<String> = (0..100)
            .step_by(11)
            .map(|l| format!("{l} -> {}", Env::scaleoutlevel(l)))
            .collect();
        expect![[""]].assert_eq(&levels.join("\n"));
    }
}
