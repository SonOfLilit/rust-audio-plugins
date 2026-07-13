//! The FM operator core: the 32 DX7 algorithms and the per-operator
//! render kernels.
//!
//! Port of `fm_core.cc`/`fm_core.h` and `fm_op_kernel.cc`/`fm_op_kernel.h`
//! from MSFA (Copyright 2012 Google Inc., Apache-2.0).

use crate::tables::Tables;

/// Render chunk size in samples (the engine computes envelopes and LFO once
/// per chunk).
pub const N: usize = 64;
pub const LG_N: i32 = 6;

// Flag bits: 0-1 output bus number, 2 add-to-bus, 4-5 input bus number,
// 6 feedback in, 7 feedback out.
const OUT_BUS_ADD: u8 = 1 << 2;
const FB_IN: u8 = 1 << 6;
const FB_OUT: u8 = 1 << 7;

/// Per-operator routing for one algorithm: 6 flag bytes, one per operator
/// (operator 6 first). Bits: 0-1 output bus, 2 add-to-bus, 4-5 input bus,
/// 6 feedback in, 7 feedback out.
type FmAlgorithm = [u8; 6];

/// The DX7's 32 algorithms.
pub const ALGORITHMS: [FmAlgorithm; 32] = [
    [0xc1, 0x11, 0x11, 0x14, 0x01, 0x14], // 1
    [0x01, 0x11, 0x11, 0x14, 0xc1, 0x14], // 2
    [0xc1, 0x11, 0x14, 0x01, 0x11, 0x14], // 3
    [0xc1, 0x11, 0x94, 0x01, 0x11, 0x14], // 4
    [0xc1, 0x14, 0x01, 0x14, 0x01, 0x14], // 5
    [0xc1, 0x94, 0x01, 0x14, 0x01, 0x14], // 6
    [0xc1, 0x11, 0x05, 0x14, 0x01, 0x14], // 7
    [0x01, 0x11, 0xc5, 0x14, 0x01, 0x14], // 8
    [0x01, 0x11, 0x05, 0x14, 0xc1, 0x14], // 9
    [0x01, 0x05, 0x14, 0xc1, 0x11, 0x14], // 10
    [0xc1, 0x05, 0x14, 0x01, 0x11, 0x14], // 11
    [0x01, 0x05, 0x05, 0x14, 0xc1, 0x14], // 12
    [0xc1, 0x05, 0x05, 0x14, 0x01, 0x14], // 13
    [0xc1, 0x05, 0x11, 0x14, 0x01, 0x14], // 14
    [0x01, 0x05, 0x11, 0x14, 0xc1, 0x14], // 15
    [0xc1, 0x11, 0x02, 0x25, 0x05, 0x14], // 16
    [0x01, 0x11, 0x02, 0x25, 0xc5, 0x14], // 17
    [0x01, 0x11, 0x11, 0xc5, 0x05, 0x14], // 18
    [0xc1, 0x14, 0x14, 0x01, 0x11, 0x14], // 19
    [0x01, 0x05, 0x14, 0xc1, 0x14, 0x14], // 20
    [0x01, 0x14, 0x14, 0xc1, 0x14, 0x14], // 21
    [0xc1, 0x14, 0x14, 0x14, 0x01, 0x14], // 22
    [0xc1, 0x14, 0x14, 0x01, 0x14, 0x04], // 23
    [0xc1, 0x14, 0x14, 0x14, 0x04, 0x04], // 24
    [0xc1, 0x14, 0x14, 0x04, 0x04, 0x04], // 25
    [0xc1, 0x05, 0x14, 0x01, 0x14, 0x04], // 26
    [0x01, 0x05, 0x14, 0xc1, 0x14, 0x04], // 27
    [0x04, 0xc1, 0x11, 0x14, 0x01, 0x14], // 28
    [0xc1, 0x14, 0x01, 0x14, 0x04, 0x04], // 29
    [0x04, 0xc1, 0x11, 0x14, 0x04, 0x04], // 30
    [0xc1, 0x14, 0x04, 0x04, 0x04, 0x04], // 31
    [0xc4, 0x04, 0x04, 0x04, 0x04, 0x04], // 32
];

/// Per-operator render state: envelope level in (log), computed gain out,
/// oscillator frequency (phase delta) and phase.
#[derive(Clone, Copy, Default)]
pub struct FmOpParams {
    pub level_in: i32,
    pub gain_out: i32,
    pub freq: i32,
    pub phase: i32,
}

/// Basic FM operator: sine of (phase + modulator input), with a linear
/// gain ramp across the chunk.
#[allow(clippy::too_many_arguments)] // mirrors the C++ kernel signature
fn compute(
    t: &Tables,
    output: &mut [i32; N],
    input: &[i32; N],
    phase0: i32,
    freq: i32,
    gain1: i32,
    gain2: i32,
    add: bool,
) {
    let dgain = (gain2 - gain1 + (N as i32 >> 1)) >> LG_N;
    let mut gain = gain1;
    let mut phase = phase0;
    for i in 0..N {
        gain += dgain;
        let y = t.sin_lookup(phase.wrapping_add(input[i]));
        let y1 = ((i64::from(y) * i64::from(gain)) >> 24) as i32;
        if add {
            output[i] = output[i].wrapping_add(y1);
        } else {
            output[i] = y1;
        }
        phase = phase.wrapping_add(freq);
    }
}

/// Sine generator, no modulation input.
fn compute_pure(
    t: &Tables,
    output: &mut [i32; N],
    phase0: i32,
    freq: i32,
    gain1: i32,
    gain2: i32,
    add: bool,
) {
    let dgain = (gain2 - gain1 + (N as i32 >> 1)) >> LG_N;
    let mut gain = gain1;
    let mut phase = phase0;
    for slot in output.iter_mut() {
        gain += dgain;
        let y = t.sin_lookup(phase);
        let y1 = ((i64::from(y) * i64::from(gain)) >> 24) as i32;
        if add {
            *slot = slot.wrapping_add(y1);
        } else {
            *slot = y1;
        }
        phase = phase.wrapping_add(freq);
    }
}

/// One operator modulating itself through the two-sample feedback buffer.
#[allow(clippy::too_many_arguments)] // mirrors the C++ kernel signature
fn compute_fb(
    t: &Tables,
    output: &mut [i32; N],
    phase0: i32,
    freq: i32,
    gain1: i32,
    gain2: i32,
    fb_buf: &mut [i32; 2],
    fb_shift: i32,
    add: bool,
) {
    let dgain = (gain2 - gain1 + (N as i32 >> 1)) >> LG_N;
    let mut gain = gain1;
    let mut phase = phase0;
    let mut y0 = fb_buf[0];
    let mut y = fb_buf[1];
    for slot in output.iter_mut() {
        gain += dgain;
        let scaled_fb = (y0.wrapping_add(y)) >> (fb_shift + 1);
        y0 = y;
        y = t.sin_lookup(phase.wrapping_add(scaled_fb));
        y = ((i64::from(y) * i64::from(gain)) >> 24) as i32;
        if add {
            *slot = slot.wrapping_add(y);
        } else {
            *slot = y;
        }
        phase = phase.wrapping_add(freq);
    }
    fb_buf[0] = y0;
    fb_buf[1] = y;
}

/// The operator graph evaluator: two intermediate buses plus the output.
pub struct FmCore {
    buf: [[i32; N]; 2],
}

impl Default for FmCore {
    fn default() -> Self {
        Self { buf: [[0; N]; 2] }
    }
}

impl FmCore {
    /// True if `op` (0 = operator 6 … 5 = operator 1) sums into the output
    /// bus for `algorithm` (0-31).
    pub fn is_carrier(algorithm: usize, op: usize) -> bool {
        ALGORITHMS[algorithm & 31][op] & OUT_BUS_ADD != 0
    }

    /// Render one chunk: runs all six operators through the algorithm's
    /// routing, summing carriers into `output`.
    pub fn render(
        &mut self,
        t: &Tables,
        output: &mut [i32; N],
        params: &mut [FmOpParams; 6],
        algorithm: usize,
        fb_buf: &mut [i32; 2],
        feedback_shift: i32,
    ) {
        const LEVEL_THRESH: i32 = 1120;
        let alg = ALGORITHMS[algorithm & 31];
        let mut has_contents = [true, false, false];
        for op in 0..6 {
            let flags = alg[op];
            let mut add = flags & OUT_BUS_ADD != 0;
            let inbus = ((flags >> 4) & 3) as usize;
            let outbus = (flags & 3) as usize;
            let gain1 = params[op].gain_out;
            let gain2 = t.exp2_lookup(params[op].level_in - (14 << 24));
            params[op].gain_out = gain2;

            if gain1 >= LEVEL_THRESH || gain2 >= LEVEL_THRESH {
                if !has_contents[outbus] {
                    add = false;
                }
                let (phase, freq) = (params[op].phase, params[op].freq);
                if inbus == 0 || !has_contents[inbus] {
                    if (flags & (FB_IN | FB_OUT)) == (FB_IN | FB_OUT) && feedback_shift < 16 {
                        // The self-modulating operator of the algorithm.
                        if outbus == 0 {
                            compute_fb(
                                t,
                                output,
                                phase,
                                freq,
                                gain1,
                                gain2,
                                fb_buf,
                                feedback_shift,
                                add,
                            );
                        } else {
                            compute_fb(
                                t,
                                &mut self.buf[outbus - 1],
                                phase,
                                freq,
                                gain1,
                                gain2,
                                fb_buf,
                                feedback_shift,
                                add,
                            );
                        }
                    } else if outbus == 0 {
                        compute_pure(t, output, phase, freq, gain1, gain2, add);
                    } else {
                        compute_pure(t, &mut self.buf[outbus - 1], phase, freq, gain1, gain2, add);
                    }
                } else {
                    // Reading one bus while possibly writing the other (or
                    // the output). The C++ aliases pointers here; the two
                    // buses are distinct (inbus != outbus always holds in
                    // the algorithm table when both are internal).
                    let input = self.buf[inbus - 1];
                    if outbus == 0 {
                        compute(t, output, &input, phase, freq, gain1, gain2, add);
                    } else {
                        compute(
                            t,
                            &mut self.buf[outbus - 1],
                            &input,
                            phase,
                            freq,
                            gain1,
                            gain2,
                            add,
                        );
                    }
                }
                has_contents[outbus] = true;
            } else if !add {
                has_contents[outbus] = false;
            }
            params[op].phase = params[op]
                .phase
                .wrapping_add(params[op].freq.wrapping_shl(LG_N as u32));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn carrier_map_matches_the_dx7_chart() {
        // One line per algorithm; `C` marks carriers, `.` modulators, in
        // operator order 6..1 (the patch layout order). Checkable against
        // any DX7 algorithm chart: e.g. alg 1 has carriers op1+op3,
        // alg 32 has all six.
        let lines: Vec<String> = (0..32)
            .map(|alg| {
                let ops: String = (0..6)
                    .map(|op| {
                        if FmCore::is_carrier(alg, op) {
                            'C'
                        } else {
                            '.'
                        }
                    })
                    .collect();
                format!("alg {:>2}: {ops}", alg + 1)
            })
            .collect();
        expect![[r#"
            alg  1: ...C.C
            alg  2: ...C.C
            alg  3: ..C..C
            alg  4: ..C..C
            alg  5: .C.C.C
            alg  6: .C.C.C
            alg  7: ..CC.C
            alg  8: ..CC.C
            alg  9: ..CC.C
            alg 10: .CC..C
            alg 11: .CC..C
            alg 12: .CCC.C
            alg 13: .CCC.C
            alg 14: .C.C.C
            alg 15: .C.C.C
            alg 16: ...CCC
            alg 17: ...CCC
            alg 18: ...CCC
            alg 19: .CC..C
            alg 20: .CC.CC
            alg 21: .CC.CC
            alg 22: .CCC.C
            alg 23: .CC.CC
            alg 24: .CCCCC
            alg 25: .CCCCC
            alg 26: .CC.CC
            alg 27: .CC.CC
            alg 28: C..C.C
            alg 29: .C.CCC
            alg 30: C..CCC
            alg 31: .CCCCC
            alg 32: CCCCCC"#]]
        .assert_eq(&lines.join("\n"));
    }
}
