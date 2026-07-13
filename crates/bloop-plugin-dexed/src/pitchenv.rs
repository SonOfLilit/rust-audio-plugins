//! DX7 pitch envelope.
//!
//! Port of `pitchenv.cc`/`pitchenv.h` from MSFA (Copyright 2012-2013 Google
//! Inc., Apache-2.0).

use crate::tables::Tables;

pub const PITCHENV_RATE: [u8; 100] = [
    1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15,
    16, 16, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 36, 37, 38, 39,
    41, 42, 44, 46, 47, 49, 51, 53, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 79, 82, 85,
    88, 91, 94, 98, 102, 106, 110, 115, 120, 125, 130, 135, 141, 147, 153, 159, 165, 171, 178,
    185, 193, 202, 211, 232, 243, 254, 255,
];

pub const PITCHENV_TAB: [i8; 100] = [
    -128, -116, -104, -95, -85, -76, -68, -61, -56, -52, -49, -46, -43, -41, -39, -37, -35, -33,
    -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14,
    -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 38, 40, 43, 46, 49, 53, 58, 65, 73, 82, 92, 103, 115, 127,
];

/// 4-stage pitch envelope; output is Q24 per octave.
#[derive(Clone, Copy, Default)]
pub struct PitchEnv {
    rates: [i32; 4],
    levels: [i32; 4],
    level: i32,
    targetlevel: i32,
    rising: bool,
    ix: usize,
    inc: i32,
    down: bool,
}

impl PitchEnv {
    /// `rates`/`levels` are DX7 parameters (0..99).
    pub fn set(&mut self, t: &Tables, rates: [i32; 4], levels: [i32; 4]) {
        self.rates = rates;
        self.levels = levels;
        self.level = i32::from(PITCHENV_TAB[levels[3] as usize]) << 19;
        self.down = true;
        self.advance(t, 0);
    }

    pub fn getsample(&mut self, t: &Tables) -> i32 {
        if self.ix < 3 || (self.ix < 4 && !self.down) {
            if self.rising {
                self.level += self.inc;
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

    fn advance(&mut self, t: &Tables, newix: usize) {
        self.ix = newix;
        if self.ix < 4 {
            let newlevel = self.levels[self.ix];
            self.targetlevel = i32::from(PITCHENV_TAB[newlevel as usize]) << 19;
            self.rising = self.targetlevel > self.level;
            self.inc = i32::from(PITCHENV_RATE[self.rates[self.ix] as usize]) * t.pitchenv_unit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Snapshot the pitch offset in semitones every `stride` chunks.
    /// Level 50 is neutral (0 semitones); the Q24-per-octave output maps
    /// to semitones as level / 2^24 * 12.
    fn check_pitchenv(
        rates: [i32; 4],
        levels: [i32; 4],
        keyup_at: usize,
        chunks: usize,
        stride: usize,
        expect: Expect,
    ) {
        let t = Tables::new(44100.0);
        let mut env = PitchEnv::default();
        env.set(&t, rates, levels);
        let mut lines = Vec::new();
        for chunk in 0..chunks {
            if chunk == keyup_at {
                env.keydown(&t, false);
                lines.push(format!("chunk {chunk:>4}: keyup"));
            }
            let level = env.getsample(&t);
            if chunk % stride == 0 {
                let semis = f64::from(level) / f64::from(1 << 24) * 12.0;
                lines.push(format!("chunk {chunk:>4}: {semis:+.2} semitones"));
            }
        }
        expect.assert_eq(&lines.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn neutral_stays_at_zero() {
        // All levels 50 = no pitch modulation, ever.
        check_pitchenv([99; 4], [50; 4], 40, 80, 20, expect![[r#"
            chunk    0: +0.00 semitones
            chunk   20: +0.00 semitones
            chunk   40: keyup
            chunk   40: +0.00 semitones
            chunk   60: +0.00 semitones"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn scoop_up_then_settle() {
        // Start an octave down (L4=38 wait, L1 target 50): classic scoop —
        // rises from below to neutral, then returns to the L4 start point
        // on release.
        check_pitchenv([70, 99, 99, 70], [50, 50, 50, 26], 60, 120, 10, expect![[r#"
            chunk    0: -8.94 semitones
            chunk   10: -8.29 semitones
            chunk   20: -7.64 semitones
            chunk   30: -7.00 semitones
            chunk   40: -6.35 semitones
            chunk   50: -5.71 semitones
            chunk   60: keyup
            chunk   60: -5.19 semitones
            chunk   70: -5.84 semitones
            chunk   80: -6.48 semitones
            chunk   90: -7.13 semitones
            chunk  100: -7.77 semitones
            chunk  110: -8.42 semitones"#]]);
    }
}
