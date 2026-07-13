//! Fixed-point lookup tables: sine, exp2, and log-frequency → phase delta.
//!
//! Port of `sin.cc`/`sin.h`, `exp2.cc`/`exp2.h` and `freqlut.cc` from MSFA
//! (music-synthesizer-for-android), Copyright 2012-2013 Google Inc.,
//! Apache-2.0. The C++ originals use global tables initialised once; here
//! everything lives in one `Tables` value owned by the plugin so there is
//! no global state and the sample-rate-dependent parts rebuild on `reset`.

pub const SIN_LG_N_SAMPLES: usize = 10;
pub const SIN_N_SAMPLES: usize = 1 << SIN_LG_N_SAMPLES;

pub const EXP2_LG_N_SAMPLES: usize = 10;
pub const EXP2_N_SAMPLES: usize = 1 << EXP2_LG_N_SAMPLES;

const FREQ_LG_N_SAMPLES: usize = 10;
const FREQ_N_SAMPLES: usize = 1 << FREQ_LG_N_SAMPLES;
const FREQ_SAMPLE_SHIFT: i32 = 24 - FREQ_LG_N_SAMPLES as i32;
const MAX_LOGFREQ_INT: i32 = 20;

/// All engine lookup tables plus the sample-rate-derived scalars the
/// envelope/LFO/pitch-envelope units need. Built on the UI thread
/// (construction allocates nothing itself, but callers typically `Box` it);
/// the audio thread only reads it.
pub struct Tables {
    /// Interleaved (delta, value) sine table (the C++ `SIN_DELTA` layout).
    sintab: [i32; SIN_N_SAMPLES * 2],
    /// Interleaved (delta, value) 2^x table.
    exp2tab: [i32; EXP2_N_SAMPLES * 2],
    /// Log-frequency (Q24, 1.0 = one octave) → phase delta per sample.
    freqlut: [i32; FREQ_N_SAMPLES + 1],
    /// Envelope rate correction: `(44100 / sample_rate) * 2^24` (env.cc is
    /// calibrated to 44.1 kHz).
    pub env_sr_multiplier: u32,
    /// Pitch-envelope increment unit for this sample rate.
    pub pitchenv_unit: i32,
    /// LFO delay unit for this sample rate.
    pub lfo_unit: u32,
    /// LFO rate → phase delta ratio for this sample rate.
    pub lfo_ratio: u32,
}

impl Tables {
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(sample_rate: f64) -> Self {
        let mut sintab = [0i32; SIN_N_SAMPLES * 2];
        // Generate the sine table with a complex rotation, exactly like the
        // C++ (values match bit-for-bit; floats only seed the rotation).
        let dphase = 2.0 * std::f64::consts::PI / SIN_N_SAMPLES as f64;
        let c = (dphase.cos() * f64::from(1 << 30) + 0.5).floor() as i64;
        let s = (dphase.sin() * f64::from(1 << 30) + 0.5).floor() as i64;
        let r = 1i64 << 29;
        let mut u = 1i64 << 30;
        let mut v = 0i64;
        for i in 0..SIN_N_SAMPLES / 2 {
            sintab[(i << 1) + 1] = ((v + 32) >> 6) as i32;
            sintab[((i + SIN_N_SAMPLES / 2) << 1) + 1] = -(((v + 32) >> 6) as i32);
            let t = (u * s + v * c + r) >> 30;
            u = (u * c - v * s + r) >> 30;
            v = t;
        }
        for i in 0..SIN_N_SAMPLES - 1 {
            sintab[i << 1] = sintab[(i << 1) + 3] - sintab[(i << 1) + 1];
        }
        sintab[(SIN_N_SAMPLES << 1) - 2] = -sintab[(SIN_N_SAMPLES << 1) - 1];

        let mut exp2tab = [0i32; EXP2_N_SAMPLES * 2];
        let inc = (1.0 / EXP2_N_SAMPLES as f64).exp2();
        let mut y = f64::from(1 << 30);
        for i in 0..EXP2_N_SAMPLES {
            exp2tab[(i << 1) + 1] = (y + 0.5).floor() as i32;
            y *= inc;
        }
        for i in 0..EXP2_N_SAMPLES - 1 {
            exp2tab[i << 1] = exp2tab[(i << 1) + 3] - exp2tab[(i << 1) + 1];
        }
        exp2tab[(EXP2_N_SAMPLES << 1) - 2] =
            ((1u32 << 31) as i64 - i64::from(exp2tab[(EXP2_N_SAMPLES << 1) - 1])) as i32;

        let mut freqlut = [0i32; FREQ_N_SAMPLES + 1];
        let mut y = (1i64 << (24 + MAX_LOGFREQ_INT)) as f64 / sample_rate;
        let inc = 2f64.powf(1.0 / FREQ_N_SAMPLES as f64);
        for slot in freqlut.iter_mut() {
            *slot = (y + 0.5).floor() as i32;
            y *= inc;
        }

        Self {
            sintab,
            exp2tab,
            freqlut,
            env_sr_multiplier: ((44100.0 / sample_rate) * f64::from(1 << 24)) as u32,
            pitchenv_unit: (crate::fm_core::N as f64 * f64::from(1 << 24) / (21.3 * sample_rate)
                + 0.5) as i32,
            lfo_unit: (crate::fm_core::N as f64 * 25190424.0 / sample_rate + 0.5) as u32,
            lfo_ratio: (4437500000.0 * crate::fm_core::N as f64 / sample_rate) as u32,
        }
    }

    /// Sine of `phase` (Q24 per cycle in, Q24 out), linearly interpolated.
    #[inline]
    pub fn sin_lookup(&self, phase: i32) -> i32 {
        const SHIFT: i32 = 24 - SIN_LG_N_SAMPLES as i32;
        let lowbits = phase & ((1 << SHIFT) - 1);
        let phase_int = ((phase >> (SHIFT - 1)) & (((SIN_N_SAMPLES - 1) << 1) as i32)) as usize;
        let dy = i64::from(self.sintab[phase_int]);
        let y0 = self.sintab[phase_int + 1];
        y0.wrapping_add(((dy * i64::from(lowbits)) >> SHIFT) as i32)
    }

    /// 2^x (Q24 in, Q24 out).
    #[inline]
    pub fn exp2_lookup(&self, x: i32) -> i32 {
        const SHIFT: i32 = 24 - EXP2_LG_N_SAMPLES as i32;
        let lowbits = x & ((1 << SHIFT) - 1);
        let x_int = ((x >> (SHIFT - 1)) & (((EXP2_N_SAMPLES - 1) << 1) as i32)) as usize;
        let dy = i64::from(self.exp2tab[x_int]);
        let y0 = self.exp2tab[x_int + 1];
        let y = y0.wrapping_add(((dy * i64::from(lowbits)) >> SHIFT) as i32);
        y >> (6 - (x >> 24))
    }

    /// Log-frequency (Q24, 1.0 = one octave, 0 = 1 Hz at `MAX_LOGFREQ_INT`
    /// = 20 semantics — see freqlut.cc) → oscillator phase delta per sample.
    #[inline]
    pub fn freq_lookup(&self, logfreq: i32) -> i32 {
        let ix = ((logfreq & 0xff_ffff) >> FREQ_SAMPLE_SHIFT) as usize;
        let y0 = i64::from(self.freqlut[ix]);
        let y1 = i64::from(self.freqlut[ix + 1]);
        let lowbits = i64::from(logfreq & ((1 << FREQ_SAMPLE_SHIFT) - 1));
        let y = (y0 + (((y1 - y0) * lowbits) >> FREQ_SAMPLE_SHIFT)) as i32;
        let hibits = logfreq >> 24;
        // The C++ shifts by up to `MAX_LOGFREQ_INT - hibits` unchecked;
        // deep negative log-frequencies (sub-1Hz after extreme pitch
        // bends) would shift ≥ 32, which is UB there and a panic here.
        y >> (MAX_LOGFREQ_INT - hibits).clamp(0, 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Snapshot table landmarks in physical units: the sine at cardinal
    /// phases, exp2 at integer inputs, and the frequencies implied by the
    /// freqlut phase deltas for a few MIDI notes (delta/2^24 cycles per
    /// sample × sample rate = Hz).
    fn check_tables(sample_rate: f64, expect: Expect) {
        let t = Tables::new(sample_rate);
        let mut lines = Vec::new();

        for (label, phase) in [
            ("0", 0i32),
            ("1/4", 1 << 22),
            ("1/2", 1 << 23),
            ("3/4", 3 << 22),
            ("1/8", 1 << 21),
        ] {
            lines.push(format!(
                "sin({label}) = {:+.6}",
                f64::from(t.sin_lookup(phase)) / f64::from(1 << 24)
            ));
        }
        for x in [-2i32, -1, 0, 1, 2] {
            lines.push(format!(
                "exp2({x}) = {:.6}",
                f64::from(t.exp2_lookup(x << 24)) / f64::from(1 << 24)
            ));
        }
        // midinote → logfreq → phase delta → Hz. A4=440, A3=220, A1=55.
        for (name, midinote) in [("A4", 69), ("A3", 57), ("A1", 33), ("C4", 60)] {
            let logfreq = 50857777 + ((1 << 24) / 12) * midinote;
            let delta = t.freq_lookup(logfreq);
            let hz = f64::from(delta) / f64::from(1 << 24) * sample_rate;
            lines.push(format!("{name} = {hz:.2}Hz"));
        }
        expect.assert_eq(&lines.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn landmarks_48k() {
        check_tables(48000.0, expect![[r#"
            sin(0) = +0.000000
            sin(1/4) = +1.000000
            sin(1/2) = +0.000000
            sin(3/4) = -1.000000
            sin(1/8) = +0.707107
            exp2(-2) = 0.250000
            exp2(-1) = 0.500000
            exp2(0) = 1.000000
            exp2(1) = 2.000000
            exp2(2) = 4.000000
            A4 = 440.00Hz
            A3 = 220.00Hz
            A1 = 55.00Hz
            C4 = 261.62Hz"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn landmarks_44k1() {
        // The frequency table depends on the sample rate; pitches must
        // come out identical in Hz.
        check_tables(44100.0, expect![[r#"
            sin(0) = +0.000000
            sin(1/4) = +1.000000
            sin(1/2) = +0.000000
            sin(3/4) = -1.000000
            sin(1/8) = +0.707107
            exp2(-2) = 0.250000
            exp2(-1) = 0.500000
            exp2(0) = 1.000000
            exp2(1) = 2.000000
            exp2(2) = 4.000000
            A4 = 440.00Hz
            A3 = 220.00Hz
            A1 = 55.00Hz
            C4 = 261.62Hz"#]]);
    }
}
