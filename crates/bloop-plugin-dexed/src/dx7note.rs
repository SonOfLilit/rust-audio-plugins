//! One playing DX7 voice: six operators + pitch envelope, driven by an
//! unpacked 156-byte patch.
//!
//! Port of `dx7note.cc`/`dx7note.h` from Dexed's msfa fork (Copyright
//! 2016-2025 Pascal Gauthier, Copyright 2012 Google Inc., Apache-2.0).
//! Simplifications relative to the original: no microtuning (MTS/Surge
//! tuning-library glue was not Apache-licensed), no portamento, no MPE —
//! see the roadmap. Pitch bend and the LFO/controller modulation paths are
//! kept intact.

use crate::env::Env;
use crate::fm_core::{FmCore, FmOpParams, N};
use crate::pitchenv::PitchEnv;
use crate::tables::Tables;

const FEEDBACK_BITDEPTH: i32 = 8;

const COARSEMUL: [i32; 32] = [
    -16777216, 0, 16777216, 26591258, 33554432, 38955489, 43368474, 47099600, 50331648, 53182516,
    55732705, 58039632, 60145690, 62083076, 63876816, 65546747, 67108864, 68576247, 69959732,
    71268397, 72509921, 73690858, 74816848, 75892776, 76922906, 77910978, 78860292, 79773775,
    80654032, 81503396, 82323963, 83117622,
];

const VELOCITY_DATA: [u8; 64] = [
    0, 70, 86, 97, 106, 114, 121, 126, 132, 138, 142, 148, 152, 156, 160, 163, 166, 170, 173, 174,
    178, 181, 184, 186, 189, 190, 194, 196, 198, 200, 202, 205, 206, 209, 211, 214, 216, 218, 220,
    222, 224, 225, 227, 229, 230, 232, 233, 235, 237, 238, 240, 241, 242, 243, 244, 246, 246, 248,
    249, 250, 251, 252, 253, 254,
];

const EXP_SCALE_DATA: [u8; 33] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 16, 19, 23, 27, 33, 39, 47, 56, 66, 80, 94, 110, 126,
    142, 158, 174, 190, 206, 222, 238, 250,
];

const PITCHMODSENSTAB: [u8; 8] = [0, 10, 20, 33, 55, 92, 153, 255];

// 0, 66, 109, 255 scaled to Q24.
const AMPMODSENSTAB: [u32; 4] = [0, 4342338, 7171437, 16777216];

/// Velocity delta in microsteps (velocity 0..127, sensitivity 0..7).
fn scale_velocity(velocity: i32, sensitivity: i32) -> i32 {
    let clamped_vel = velocity.clamp(0, 127);
    let vel_value = i32::from(VELOCITY_DATA[(clamped_vel >> 1) as usize]) - 239;
    ((sensitivity * vel_value + 7) >> 3) << 4
}

/// Keyboard rate scaling: qRate delta for this note.
fn scale_rate(midinote: i32, sensitivity: i32) -> i32 {
    let x = (midinote / 3 - 7).clamp(0, 31);
    (sensitivity * x) >> 3
}

fn scale_curve(group: i32, depth: i32, curve: i32) -> i32 {
    let mut scale = if curve == 0 || curve == 3 {
        // linear
        (group * depth * 329) >> 12
    } else {
        // exponential
        let raw_exp = i32::from(EXP_SCALE_DATA[(group as usize).min(EXP_SCALE_DATA.len() - 1)]);
        (raw_exp * depth * 329) >> 15
    };
    if curve < 2 {
        scale = -scale;
    }
    scale
}

/// Keyboard level scaling around the breakpoint.
fn scale_level(
    midinote: i32,
    break_pt: i32,
    left_depth: i32,
    right_depth: i32,
    left_curve: i32,
    right_curve: i32,
) -> i32 {
    let offset = midinote - break_pt - 17;
    if offset >= 0 {
        scale_curve((offset + 1) / 3, right_depth, right_curve)
    } else {
        scale_curve(-(offset - 1) / 3, left_depth, left_curve)
    }
}

/// Standard-tuning MIDI note → log-frequency (Q24, 1.0 = one octave).
/// (The original routes this through the tuning library; this is its
/// standard-tuning formula, also present in Google's dx7note.cc.)
fn midinote_to_logfreq(midinote: i32) -> i32 {
    const BASE: i32 = 50857777; // (1 << 24) * (log(440) / log(2) - 69/12)
    const STEP: i32 = (1 << 24) / 12;
    BASE + STEP * midinote
}

/// Oscillator log-frequency for a note, including the hardware-measured
/// detune curve.
#[allow(clippy::cast_possible_truncation)]
fn osc_freq(midinote: i32, mode: i32, coarse: i32, fine: i32, detune: i32) -> i32 {
    let mut logfreq: i32;
    if mode == 0 {
        logfreq = midinote_to_logfreq(midinote);

        // Measured from a real DX7 by the Dexed project.
        let detune_ratio = 0.0209 * (-0.396 * (f64::from(logfreq) / f64::from(1 << 24))).exp() / 7.0;
        logfreq += (detune_ratio * f64::from(logfreq) * f64::from(detune - 7)) as i32;

        logfreq += COARSEMUL[(coarse & 31) as usize];
        if fine != 0 {
            // (1 << 24) / log(2)
            logfreq += (24204406.323123 * (1.0 + 0.01 * f64::from(fine)).ln() + 0.5).floor() as i32;
        }
    } else {
        // Fixed frequency: ((1 << 24) * log(10) / log(2) * .01) << 3
        logfreq = (4458616 * ((coarse & 3) * 100 + fine)) >> 3;
        logfreq += if detune > 7 { 13457 * (detune - 7) } else { 0 };
    }
    logfreq
}

/// The controller state a voice reads while rendering. A trimmed-down port
/// of `controllers.h`: pitch bend plus the modulation totals the CC mapping
/// layer would produce (we have no CC sources wired yet, so `amp_mod` and
/// `pitch_mod` stay 0 and `eg_mod` stays 127 = no EG attenuation).
#[derive(Clone, Copy)]
pub struct Controllers {
    /// Raw 14-bit pitch bend, 0..16383, center 8192.
    pub pitch_bend: i32,
    /// Bend range in semitones, up / down.
    pub pitch_range_up: i32,
    pub pitch_range_dn: i32,
    /// If nonzero, bend quantizes to `12 / step` semitone steps.
    pub pitch_step: i32,
    pub amp_mod: i32,
    pub pitch_mod: i32,
    pub eg_mod: i32,
    /// Master tune offset in Q24 log-frequency.
    pub master_tune: i32,
}

impl Default for Controllers {
    fn default() -> Self {
        Self {
            pitch_bend: 8192,
            pitch_range_up: 2,
            pitch_range_dn: 2,
            pitch_step: 0,
            amp_mod: 0,
            pitch_mod: 0,
            eg_mod: 127,
            master_tune: 0,
        }
    }
}

/// One voice.
#[derive(Clone, Copy, Default)]
pub struct Dx7Note {
    initialised: bool,
    env: [Env; 6],
    params: [FmOpParams; 6],
    pitchenv: PitchEnv,
    basepitch: [i32; 6],
    fb_buf: [i32; 2],
    fb_shift: i32,
    ampmodsens: [i32; 6],
    opmode: [u8; 6],
    ampmoddepth: i32,
    algorithm: usize,
    pitchmoddepth: i32,
    pitchmodsens: i32,
}

impl Dx7Note {
    /// Start the voice from an unpacked 156-byte patch.
    pub fn init(&mut self, t: &Tables, patch: &[u8; 156], midinote: i32, velocity: i32) {
        self.initialised = true;
        for op in 0..6 {
            let off = op * 21;
            let mut rates = [0i32; 4];
            let mut levels = [0i32; 4];
            for i in 0..4 {
                rates[i] = i32::from(patch[off + i]);
                levels[i] = i32::from(patch[off + 4 + i]);
            }
            let mut outlevel = Env::scaleoutlevel(i32::from(patch[off + 16]));
            let level_scaling = scale_level(
                midinote,
                i32::from(patch[off + 8]),
                i32::from(patch[off + 9]),
                i32::from(patch[off + 10]),
                i32::from(patch[off + 11]),
                i32::from(patch[off + 12]),
            );
            outlevel += level_scaling;
            outlevel = outlevel.min(127);
            outlevel <<= 5;
            outlevel += scale_velocity(velocity, i32::from(patch[off + 15]));
            outlevel = outlevel.max(0);
            let rate_scaling = scale_rate(midinote, i32::from(patch[off + 13]));
            self.env[op].init(t, rates, levels, outlevel, rate_scaling);

            let mode = i32::from(patch[off + 17]);
            let coarse = i32::from(patch[off + 18]);
            let fine = i32::from(patch[off + 19]);
            let detune = i32::from(patch[off + 20]);
            self.basepitch[op] = osc_freq(midinote, mode, coarse, fine, detune);
            self.opmode[op] = mode as u8;
            self.ampmodsens[op] = AMPMODSENSTAB[(patch[off + 14] & 3) as usize] as i32;
        }
        let mut rates = [0i32; 4];
        let mut levels = [0i32; 4];
        for i in 0..4 {
            rates[i] = i32::from(patch[126 + i]);
            levels[i] = i32::from(patch[130 + i]);
        }
        self.pitchenv.set(t, rates, levels);
        self.algorithm = usize::from(patch[134]);
        let feedback = i32::from(patch[135]);
        self.fb_shift = if feedback == 0 { 16 } else { FEEDBACK_BITDEPTH - feedback };
        self.pitchmoddepth = (i32::from(patch[139]) * 165) >> 6;
        self.pitchmodsens = i32::from(PITCHMODSENSTAB[(patch[143] & 7) as usize]);
        self.ampmoddepth = (i32::from(patch[140]) * 165) >> 6;
    }

    /// Render one `N`-sample chunk, *adding* into `buf`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn compute(
        &mut self,
        t: &Tables,
        core: &mut FmCore,
        buf: &mut [i32; N],
        lfo_val: i32,
        lfo_delay: i32,
        ctrls: &Controllers,
    ) {
        // ==== PITCH ====
        let pmd = u32::wrapping_mul(self.pitchmoddepth as u32, lfo_delay as u32); // Q32
        let senslfo = self.pitchmodsens * (lfo_val - (1 << 23));
        let pmod_1 = (((i64::from(pmd)) * i64::from(senslfo)) >> 39) as i32;
        let pmod_1 = pmod_1.abs();
        let pmod_2 = ((i64::from(ctrls.pitch_mod) * i64::from(senslfo)) >> 14) as i32;
        let pmod_2 = pmod_2.abs();
        let mut pitch_mod = pmod_1.max(pmod_2);
        pitch_mod = self.pitchenv.getsample(t) + pitch_mod * if senslfo < 0 { -1 } else { 1 };

        // ---- PITCH BEND ----
        let mut pb = ctrls.pitch_bend - 0x2000;
        if pb != 0 {
            if ctrls.pitch_step == 0 {
                let range = if pb >= 0 { ctrls.pitch_range_up } else { ctrls.pitch_range_dn };
                pb = ((f64::from(pb << 11)) * f64::from(range) / 12.0) as i32;
            } else {
                let stp = 12 / ctrls.pitch_step;
                pb = pb * stp / 8191;
                pb = (pb * (8191 / stp)) << 11;
            }
        }
        let pitch_base = pb + ctrls.master_tune;
        pitch_mod += pitch_base;

        // ==== AMP MOD ====
        let lfo_val = (1 << 24) - lfo_val;
        let amod_1 = ((i64::from(self.ampmoddepth) * i64::from(lfo_delay)) >> 8) as u32; // Q24
        let amod_1 = ((u64::from(amod_1) * (lfo_val as u64)) >> 24) as u32;
        let amod_2 = ((i64::from(ctrls.amp_mod) * i64::from(lfo_val)) >> 7) as u32;
        let mut amd_mod = amod_1.max(amod_2);

        // ==== EG AMP MOD ====
        let amod_3 = ((ctrls.eg_mod + 1) << 17) as u32;
        amd_mod = amd_mod.max((1 << 24) - amod_3);

        // ==== OP RENDER ====
        for op in 0..6 {
            let basepitch = self.basepitch[op];
            if self.opmode[op] != 0 {
                // Fixed-frequency operators ignore pitch env and LFO.
                self.params[op].freq = t.freq_lookup(basepitch.wrapping_add(pitch_base));
            } else {
                self.params[op].freq = t.freq_lookup(basepitch.wrapping_add(pitch_mod));
            }

            let mut level = self.env[op].getsample(t);
            if self.ampmodsens[op] != 0 {
                let sensamp =
                    ((u64::from(amd_mod) * (self.ampmodsens[op] as u64)) >> 24) as u32;

                // The original notes this needs real tuning ("TODO: mehhh").
                let pt = ((f64::from(sensamp) / 262144.0) * 0.07 + 12.2).exp() as u64;
                let ldiff = (((level as u64) * (pt << 4)) >> 28) as i32;
                level -= ldiff;
            }
            self.params[op].level_in = level;
        }
        core.render(t, buf, &mut self.params, self.algorithm, &mut self.fb_buf, self.fb_shift);
    }

    pub fn keyup(&mut self, t: &Tables) {
        for op in 0..6 {
            self.env[op].keydown(t, false);
        }
        self.pitchenv.keydown(t, false);
    }

    /// A note is playing if it has been initialised and any carrier's amp
    /// envelope is still active.
    pub fn is_playing(&self) -> bool {
        self.initialised
            && (0..6).any(|op| FmCore::is_carrier(self.algorithm, op) && self.env[op].is_active())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::harmonic_profile;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// A synthetic unpacked patch: instant full envelopes, no LFO, no
    /// scaling — everything neutral except algorithm, feedback, and the
    /// given operators. `ops` lists (operator number 1-6, coarse, level).
    /// Note the patch layout stores operator 6 first.
    fn synth_patch(algorithm: u8, feedback: u8, ops: &[(usize, u8, u8)]) -> [u8; 156] {
        let mut patch = [0u8; 156];
        for op in 0..6 {
            let off = op * 21;
            patch[off..off + 4].copy_from_slice(&[99, 99, 99, 99]); // rates
            patch[off + 4..off + 8].copy_from_slice(&[99, 99, 99, 99]); // levels
            patch[off + 16] = 0; // output level: off unless listed
            patch[off + 18] = 1; // coarse = 1x
        }
        for &(opno, coarse, level) in ops {
            let off = (6 - opno) * 21;
            patch[off + 18] = coarse;
            patch[off + 16] = level;
        }
        for i in 0..4 {
            patch[126 + i] = 99; // pitch env rates
            patch[130 + i] = 50; // pitch env levels: neutral
        }
        patch[134] = algorithm - 1;
        patch[135] = feedback;
        patch[144] = 24; // transpose: none
        patch[155] = 0x3f;
        patch
    }

    /// Render a note through the full voice path and snapshot which
    /// harmonics of the note's fundamental carry energy. This is the
    /// algorithm-routing proof: carriers-only patches give a bare h1,
    /// modulator stacks spread energy across the series.
    fn check_spectrum(patch: &[u8; 156], midinote: i32, expect: Expect) {
        let t = Tables::new(48000.0);
        let mut core = FmCore::default();
        let mut note = Dx7Note::default();
        note.init(&t, patch, midinote, 100);
        let ctrls = Controllers::default();
        let mut samples = Vec::new();
        let mut buf = [0i32; N];
        for _ in 0..750 {
            buf = [0; N];
            // Neutral LFO (its zero point is 1<<23), delay fully open.
            note.compute(&t, &mut core, &mut buf, 1 << 23, 1 << 24, &ctrls);
            samples.extend(buf.iter().map(|&v| v as f32 / 16_777_216.0));
        }
        let _ = buf;
        // Skip the attack, analyze the sustained tail.
        let sustained = &samples[16000..];
        let f0 = 440.0 * 2f64.powf((f64::from(midinote) - 69.0) / 12.0);
        let profile = harmonic_profile(sustained, 48000.0, f0, 8);
        expect.assert_eq(&format!("f0={f0:.1}Hz\n{}", profile.join("\n")));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn algorithm_32_single_carrier_is_pure_sine() {
        // Alg 32: six parallel carriers; only op1 on → one spectral line.
        check_spectrum(&synth_patch(32, 0, &[(1, 1, 99)]), 69, expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn algorithm_1_modulator_creates_sidebands() {
        // Alg 1: op2 (ratio 2) modulates carrier op1 (ratio 1) → energy
        // appears at h1±2k: odd harmonics 1, 3, 5…
        check_spectrum(&synth_patch(1, 0, &[(1, 1, 99), (2, 2, 75)]), 69, expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn feedback_widens_the_spectrum() {
        // Alg 32 with feedback 7 on operator 6 (the feedback op of alg
        // 32): self-modulation approaches a sawtooth — all harmonics.
        check_spectrum(&synth_patch(32, 7, &[(6, 1, 99)]), 69, expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn coarse_2_doubles_pitch() {
        // Single carrier at coarse 2: all energy at h2 of the note's f0.
        check_spectrum(&synth_patch(32, 0, &[(1, 2, 99)]), 69, expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn release_ends_the_note() {
        let t = Tables::new(48000.0);
        let mut core = FmCore::default();
        let mut note = Dx7Note::default();
        let patch = synth_patch(32, 0, &[(1, 1, 99)]);
        note.init(&t, &patch, 69, 100);
        let ctrls = Controllers::default();
        let mut buf = [0i32; N];
        let mut lines = Vec::new();
        let mut probe = |label: &str, note: &Dx7Note, buf: &[i32; N]| {
            let peak = buf.iter().map(|v| v.abs()).max().unwrap_or(0);
            lines.push(format!(
                "{label}: playing={} peak={:.4}",
                note.is_playing(),
                f64::from(peak) / f64::from(1 << 24)
            ));
        };
        for _ in 0..100 {
            buf = [0; N];
            note.compute(&t, &mut core, &mut buf, 1 << 23, 1 << 24, &ctrls);
        }
        probe("held", &note, &buf);
        note.keyup(&t);
        for _ in 0..4000 {
            buf = [0; N];
            note.compute(&t, &mut core, &mut buf, 1 << 23, 1 << 24, &ctrls);
        }
        probe("after release", &note, &buf);
        expect![[""]].assert_eq(&lines.join("\n"));
    }
}
