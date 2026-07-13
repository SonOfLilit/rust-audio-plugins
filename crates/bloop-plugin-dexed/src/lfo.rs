//! Low-frequency oscillator, compatible with the DX7.
//!
//! Port of `lfo.cc`/`lfo.h` from Dexed's msfa fork (Copyright 2013 Google
//! Inc., Apache-2.0), including Dexed's hardware-measured LFO rate table.

use crate::tables::Tables;

/// LFO frequency in Hz for each DX7 rate value 0..99, measured from
/// hardware by the Dexed project.
const LFO_SOURCE: [f64; 100] = [
    0.062541, 0.125031, 0.312393, 0.437120, 0.624610, 0.750694, 0.936330, 1.125302, 1.249609,
    1.436782, 1.560915, 1.752081, 1.875117, 2.062494, 2.247191, 2.374451, 2.560492, 2.686728,
    2.873976, 2.998950, 3.188013, 3.369840, 3.500175, 3.682224, 3.812065, 4.000800, 4.186202,
    4.310716, 4.501260, 4.623209, 4.814636, 4.930480, 5.121901, 5.315191, 5.434783, 5.617346,
    5.750431, 5.946717, 6.062811, 6.248438, 6.431695, 6.564264, 6.749460, 6.868132, 7.052186,
    7.250580, 7.375719, 7.556294, 7.687577, 7.877738, 7.993605, 8.181967, 8.372405, 8.504848,
    8.685079, 8.810573, 8.986341, 9.122423, 9.300595, 9.500285, 9.607994, 9.798158, 9.950249,
    10.117361, 11.251125, 11.384335, 12.562814, 13.676149, 13.904338, 15.092062, 16.366612,
    16.638935, 17.869907, 19.193858, 19.425019, 20.833333, 21.034918, 22.502250, 24.003841,
    24.260068, 25.746653, 27.173913, 27.578599, 29.052876, 30.693677, 31.191516, 32.658393,
    34.317090, 34.674064, 36.416606, 38.197097, 38.550501, 40.387722, 40.749796, 42.625746,
    44.326241, 44.883303, 46.772685, 48.590865, 49.261084,
];

#[derive(Clone, Copy, Default)]
pub struct Lfo {
    phase: u32, // Q32
    delta: u32,
    waveform: u8,
    randstate: u8,
    sync: bool,
    delaystate: u32,
    delayinc: u32,
    delayinc2: u32,
}

impl Lfo {
    /// `params` are the six LFO bytes of an unpacked patch
    /// (`patch[137..143]`): speed, delay, PMD, AMD, sync, waveform.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn reset(&mut self, t: &Tables, params: &[u8; 6]) {
        let rate = params[0] as usize; // 0..99
        self.delta = (LFO_SOURCE[rate.min(99)] * f64::from(t.lfo_ratio)) as u32;
        let mut a = 99 - i32::from(params[1]); // LFO delay
        if a == 99 {
            self.delayinc = !0u32;
            self.delayinc2 = !0u32;
        } else {
            a = (16 + (a & 15)) << (1 + (a >> 4));
            self.delayinc = t.lfo_unit.wrapping_mul(a as u32);
            a &= 0xff80;
            a = a.max(0x80);
            self.delayinc2 = t.lfo_unit.wrapping_mul(a as u32);
        }
        self.waveform = params[5];
        self.sync = params[4] != 0;
    }

    /// Next LFO value, 0..1 in Q24. (The C++ reads the sine table through a
    /// global; here `Tables` is passed in.)
    pub fn getsample(&mut self, t: &Tables) -> i32 {
        self.phase = self.phase.wrapping_add(self.delta);
        match self.waveform {
            0 => {
                // triangle
                let mut x = (self.phase >> 7) as i32;
                x ^= -((self.phase >> 31) as i32);
                x &= (1 << 24) - 1;
                x
            }
            1 => ((!self.phase ^ (1u32 << 31)) >> 8) as i32, // saw down
            2 => ((self.phase ^ (1u32 << 31)) >> 8) as i32,  // saw up
            3 => ((!self.phase >> 7) & (1 << 24)) as i32,    // square
            4 => (1 << 23) + (t.sin_lookup((self.phase >> 8) as i32) >> 1), // sine
            5 => {
                // sample & hold
                if self.phase < self.delta {
                    self.randstate = self.randstate.wrapping_mul(179).wrapping_add(17);
                }
                let x = i32::from(self.randstate ^ 0x80);
                (x + 1) << 16
            }
            _ => 1 << 23,
        }
    }

    /// Next delay-ramp value, 0..1 in Q24.
    pub fn getdelay(&mut self) -> i32 {
        let delta = if self.delaystate < (1u32 << 31) {
            self.delayinc
        } else {
            self.delayinc2
        };
        let d = u64::from(self.delaystate) + u64::from(delta);
        if d > u64::from(u32::MAX) {
            return 1 << 24;
        }
        self.delaystate = d as u32;
        if d < (1u64 << 31) {
            0
        } else {
            ((d >> 7) & ((1 << 24) - 1)) as i32
        }
    }

    pub fn keydown(&mut self) {
        if self.sync {
            self.phase = (1u32 << 31) - 1;
        }
        self.delaystate = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Drive the LFO at 44.1kHz and snapshot its normalized value (0..1)
    /// every `stride` chunks, plus the measured cycle length in chunks
    /// (peak-to-peak) so the rate can be checked against the Hz table.
    fn check_lfo(params: [u8; 6], chunks: usize, stride: usize, expect: Expect) {
        let t = Tables::new(44100.0);
        let mut lfo = Lfo::default();
        lfo.reset(&t, &params);
        lfo.keydown();
        let mut lines = Vec::new();
        let mut values = Vec::new();
        for chunk in 0..chunks {
            let v = f64::from(lfo.getsample(&t)) / f64::from(1 << 24);
            values.push(v);
            if chunk % stride == 0 {
                lines.push(format!("chunk {chunk:>4}: {v:.3}"));
            }
        }
        // Cycle length: distance between the first two rising crossings
        // of 0.5.
        let crossings: Vec<usize> = values
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] < 0.5 && w[1] >= 0.5)
            .map(|(i, _)| i)
            .collect();
        if crossings.len() >= 2 {
            let period_chunks = crossings[1] - crossings[0];
            let hz = 44100.0 / 64.0 / period_chunks as f64;
            lines.push(format!("period={period_chunks} chunks (~{hz:.2}Hz)"));
        }
        expect.assert_eq(&lines.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn triangle_at_rate_70() {
        // Rate 70 ≈ 16.37Hz per the hardware-measured table.
        check_lfo(
            [70, 0, 0, 0, 1, 0],
            128,
            8,
            expect![[r#"
            chunk    0: 0.951
            chunk    8: 0.558
            chunk   16: 0.166
            chunk   24: 0.227
            chunk   32: 0.620
            chunk   40: 0.988
            chunk   48: 0.595
            chunk   56: 0.202
            chunk   64: 0.190
            chunk   72: 0.583
            chunk   80: 0.976
            chunk   88: 0.632
            chunk   96: 0.239
            chunk  104: 0.153
            chunk  112: 0.546
            chunk  120: 0.939
            period=41 chunks (~16.81Hz)"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn square_at_rate_70() {
        check_lfo(
            [70, 0, 0, 0, 1, 3],
            128,
            8,
            expect![[r#"
            chunk    0: 0.000
            chunk    8: 0.000
            chunk   16: 0.000
            chunk   24: 1.000
            chunk   32: 1.000
            chunk   40: 0.000
            chunk   48: 0.000
            chunk   56: 0.000
            chunk   64: 1.000
            chunk   72: 1.000
            chunk   80: 1.000
            chunk   88: 0.000
            chunk   96: 0.000
            chunk  104: 1.000
            chunk  112: 1.000
            chunk  120: 1.000
            period=41 chunks (~16.81Hz)"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn sine_at_rate_70() {
        check_lfo(
            [70, 0, 0, 0, 1, 4],
            128,
            8,
            expect![[r#"
            chunk    0: 0.423
            chunk    8: 0.008
            chunk   16: 0.251
            chunk   24: 0.827
            chunk   32: 0.965
            chunk   40: 0.481
            chunk   48: 0.022
            chunk   56: 0.203
            chunk   64: 0.781
            chunk   72: 0.983
            chunk   80: 0.538
            chunk   88: 0.042
            chunk   96: 0.159
            chunk  104: 0.732
            chunk  112: 0.995
            chunk  120: 0.596
            period=41 chunks (~16.81Hz)"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn delay_ramps_in() {
        let t = Tables::new(44100.0);
        let mut lfo = Lfo::default();
        lfo.reset(&t, &[70, 60, 0, 0, 1, 0]);
        lfo.keydown();
        let mut lines = Vec::new();
        for chunk in 0..400 {
            let d = f64::from(lfo.getdelay()) / f64::from(1 << 24);
            if chunk % 40 == 0 {
                lines.push(format!("chunk {chunk:>4}: delay={d:.3}"));
            }
        }
        expect![[r#"
            chunk    0: delay=0.000
            chunk   40: delay=0.000
            chunk   80: delay=0.000
            chunk  120: delay=0.000
            chunk  160: delay=0.000
            chunk  200: delay=0.000
            chunk  240: delay=0.000
            chunk  280: delay=0.000
            chunk  320: delay=0.005
            chunk  360: delay=0.092"#]]
        .assert_eq(&lines.join("\n"));
    }
}
