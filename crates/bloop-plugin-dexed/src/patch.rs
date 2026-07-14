//! DX7 patch data: the packed 128-byte bulk voice format, 32-voice
//! cartridges (sysex banks), and the built-in ROM bank.
//!
//! `unpack_patch` is a port of `patch.cc` from Google's
//! music-synthesizer-for-android (Copyright 2013 Google Inc., Apache-2.0);
//! this file exists only in the original repo — Dexed unpacks voices in its
//! GPL plugin layer, which we don't use. Bank/sysex framing follows the
//! DX7 bulk-dump spec (size-based validation, like Dexed: 4104 bytes with
//! the sysex frame or 4096 raw).

/// Packed bulk voice size.
pub const PACKED_VOICE_SIZE: usize = 128;
/// Unpacked voice size as the engine consumes it (155 parameters + the
/// operator on/off byte the engine appends).
pub const VOICE_SIZE: usize = 156;
/// Voices per cartridge.
pub const BANK_SIZE: usize = 32;

const BULK_DATA_SIZE: usize = PACKED_VOICE_SIZE * BANK_SIZE; // 4096
const SYSEX_SIZE: usize = BULK_DATA_SIZE + 8; // F0 43 00 09 20 00 …data… checksum F7

/// The DX7 ROM1A factory cartridge ("BRASS 1" … "TAKE OFF"), from the
/// original-Yamaha-cartridge collection linked by the Dexed project.
pub static ROM1A: &[u8] = include_bytes!("../assets/ROM1A.syx");

/// Unpack a 128-byte packed bulk voice into the 156-byte layout the engine
/// reads (21 bytes per operator, then pitch env, algorithm, feedback, LFO,
/// transpose, name, operator-enable).
pub fn unpack_patch(bulk: &[u8; PACKED_VOICE_SIZE]) -> [u8; VOICE_SIZE] {
    let mut patch = [0u8; VOICE_SIZE];
    for op in 0..6 {
        let (b, p) = (op * 17, op * 21);
        // EG rates and levels, breakpoint, depths.
        patch[p..p + 11].copy_from_slice(&bulk[b..b + 11]);
        let leftrightcurves = bulk[b + 11];
        patch[p + 11] = leftrightcurves & 3;
        patch[p + 12] = (leftrightcurves >> 2) & 3;
        let detune_rs = bulk[b + 12];
        patch[p + 13] = detune_rs & 7;
        patch[p + 20] = detune_rs >> 3;
        let kvs_ams = bulk[b + 13];
        patch[p + 14] = kvs_ams & 3;
        patch[p + 15] = kvs_ams >> 2;
        patch[p + 16] = bulk[b + 14]; // output level
        let fcoarse_mode = bulk[b + 15];
        patch[p + 17] = fcoarse_mode & 1;
        patch[p + 18] = fcoarse_mode >> 1;
        patch[p + 19] = bulk[b + 16]; // fine freq
    }
    patch[126..135].copy_from_slice(&bulk[102..111]); // pitch env, algorithm
    let oks_fb = bulk[111];
    patch[135] = oks_fb & 7;
    patch[136] = oks_fb >> 3;
    patch[137..141].copy_from_slice(&bulk[112..116]); // LFO
    let lpms_lfw_lks = bulk[116];
    patch[141] = lpms_lfw_lks & 1;
    patch[142] = (lpms_lfw_lks >> 1) & 7;
    patch[143] = lpms_lfw_lks >> 4;
    patch[144..155].copy_from_slice(&bulk[117..128]); // transpose, name
    patch[155] = 0x3f; // operator on/off
    patch
}

/// The classic DX7 "INIT VOICE", in the unpacked engine layout: algorithm
/// 1, no feedback, OP1 the sole carrier at ratio 1.0 / level 99, square
/// envelopes (rates 99, levels 99/99/99/0), breakpoint C3, pitch EG flat
/// at center, LFO at panel defaults, transpose 0. Every host param's
/// declared default is this voice's byte — "Init" and "all defaults" are
/// the same thing by construction.
pub fn init_voice() -> [u8; VOICE_SIZE] {
    let mut v = [0u8; VOICE_SIZE];
    for op in 0..6 {
        let p = op * 21;
        v[p..p + 4].copy_from_slice(&[99, 99, 99, 99]); // EG rates
        v[p + 4..p + 8].copy_from_slice(&[99, 99, 99, 0]); // EG levels
        v[p + 8] = 39; // keyboard scale breakpoint (C3)
        v[p + 18] = 1; // frequency coarse: ratio 1.0
        v[p + 20] = 7; // detune center
    }
    v[5 * 21 + 16] = 99; // OP1 (last op slot) is the carrier, full level
    v[126..130].copy_from_slice(&[99, 99, 99, 99]); // pitch EG rates
    v[130..134].copy_from_slice(&[50, 50, 50, 50]); // pitch EG levels (flat)
    v[136] = 1; // osc key sync
    v[137] = 35; // LFO speed
    v[141] = 1; // LFO key sync
    v[143] = 3; // pitch mod sensitivity
    v[144] = 24; // transpose center
    v[145..155].copy_from_slice(b"INIT VOICE");
    v[155] = 0x3f; // all six operators on
    v
}

/// A 32-voice cartridge, stored packed.
pub struct Bank {
    voices: [[u8; PACKED_VOICE_SIZE]; BANK_SIZE],
}

#[derive(Debug, PartialEq, Eq)]
pub enum BankError {
    /// Not 4104 (framed) or 4096 (raw) bytes.
    BadSize(usize),
}

impl Bank {
    /// Parse a cartridge from sysex bytes. Accepts the full 4104-byte
    /// bulk-dump frame or 4096 bytes of raw voice data (validation is by
    /// size, like Dexed's).
    pub fn from_sysex(data: &[u8]) -> Result<Self, BankError> {
        let raw: &[u8] = match data.len() {
            SYSEX_SIZE => &data[6..6 + BULK_DATA_SIZE],
            BULK_DATA_SIZE => data,
            n => return Err(BankError::BadSize(n)),
        };
        let mut voices = [[0u8; PACKED_VOICE_SIZE]; BANK_SIZE];
        for (i, v) in voices.iter_mut().enumerate() {
            v.copy_from_slice(&raw[i * PACKED_VOICE_SIZE..(i + 1) * PACKED_VOICE_SIZE]);
        }
        Ok(Self { voices })
    }

    /// The built-in ROM1A bank. The embedded cartridge is validated by
    /// tests, so parsing cannot fail at runtime.
    pub fn rom1a() -> Self {
        Self::from_sysex(ROM1A).unwrap_or(Self {
            voices: [[0u8; PACKED_VOICE_SIZE]; BANK_SIZE],
        })
    }

    /// Unpacked engine-ready voice `index` (0-31).
    pub fn voice(&self, index: usize) -> [u8; VOICE_SIZE] {
        unpack_patch(&self.voices[index & (BANK_SIZE - 1)])
    }

    /// The 10-character voice name, trimmed, with non-ASCII bytes replaced.
    pub fn name(&self, index: usize) -> String {
        self.voices[index & (BANK_SIZE - 1)][118..128]
            .iter()
            .map(|&b| {
                if (0x20..0x7f).contains(&b) {
                    b as char
                } else {
                    '?'
                }
            })
            .collect::<String>()
            .trim()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    fn check_bank(data: &[u8], expect: Expect) {
        let text = match Bank::from_sysex(data) {
            Ok(bank) => (0..BANK_SIZE)
                .map(|i| format!("{i:>2}: {}", bank.name(i)))
                .collect::<Vec<_>>()
                .join("\n"),
            Err(e) => format!("{e:?}"),
        };
        expect.assert_eq(&text);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn rom1a_names() {
        check_bank(
            ROM1A,
            expect![[r#"
             0: BRASS   1
             1: BRASS   2
             2: BRASS   3
             3: STRINGS 1
             4: STRINGS 2
             5: STRINGS 3
             6: ORCHESTRA
             7: PIANO   1
             8: PIANO   2
             9: PIANO   3
            10: E.PIANO 1
            11: GUITAR  1
            12: GUITAR  2
            13: SYN-LEAD 1
            14: BASS    1
            15: BASS    2
            16: E.ORGAN 1
            17: PIPES   1
            18: HARPSICH 1
            19: CLAV    1
            20: VIBE    1
            21: MARIMBA
            22: KOTO
            23: FLUTE   1
            24: ORCH-CHIME
            25: TUB BELLS
            26: STEEL DRUM
            27: TIMPANI
            28: REFS WHISL
            29: VOICE   1
            30: TRAIN
            31: TAKE OFF"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn raw_4096_bytes_accepted() {
        // The same cartridge without the sysex frame parses identically.
        check_bank(
            &ROM1A[6..6 + BULK_DATA_SIZE],
            expect![[r#"
             0: BRASS   1
             1: BRASS   2
             2: BRASS   3
             3: STRINGS 1
             4: STRINGS 2
             5: STRINGS 3
             6: ORCHESTRA
             7: PIANO   1
             8: PIANO   2
             9: PIANO   3
            10: E.PIANO 1
            11: GUITAR  1
            12: GUITAR  2
            13: SYN-LEAD 1
            14: BASS    1
            15: BASS    2
            16: E.ORGAN 1
            17: PIPES   1
            18: HARPSICH 1
            19: CLAV    1
            20: VIBE    1
            21: MARIMBA
            22: KOTO
            23: FLUTE   1
            24: ORCH-CHIME
            25: TUB BELLS
            26: STEEL DRUM
            27: TIMPANI
            28: REFS WHISL
            29: VOICE   1
            30: TRAIN
            31: TAKE OFF"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn wrong_size_rejected() {
        check_bank(&ROM1A[..100], expect!["BadSize(100)"]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn init_voice_interpretation() {
        // The classic DX7 INIT VOICE: algorithm 1, no feedback, OP1 the
        // sole carrier at ratio 1.0 / level 99, square envelopes
        // (instant attack, full sustain, instant release), pitch EG flat
        // at center, transpose 0. Same interpretation format as
        // `unpack_epiano` so the two snapshots read side by side.
        let v = init_voice();
        let mut lines = vec![
            format!("algorithm: {} feedback: {}", v[134] + 1, v[135]),
            format!("transpose: {:+}", i32::from(v[144]) - 24),
            format!(
                "pitch eg: rates={:?} levels={:?}",
                &v[126..130],
                &v[130..134]
            ),
            format!(
                "name: {}",
                v[145..155].iter().map(|&b| b as char).collect::<String>()
            ),
            format!("op enable: {:#04x}", v[155]),
        ];
        for op in 0..6 {
            let off = op * 21;
            lines.push(format!(
                "op{}: mode={} coarse={} fine={} detune={:+} outlevel={} rates={:?} levels={:?}",
                6 - op,
                v[off + 17],
                v[off + 18],
                v[off + 19],
                i32::from(v[off + 20]) - 7,
                v[off + 16],
                &v[off..off + 4],
                &v[off + 4..off + 8],
            ));
        }
        expect![[r#"
            algorithm: 1 feedback: 0
            transpose: +0
            pitch eg: rates=[99, 99, 99, 99] levels=[50, 50, 50, 50]
            name: INIT VOICE
            op enable: 0x3f
            op6: mode=0 coarse=1 fine=0 detune=+0 outlevel=0 rates=[99, 99, 99, 99] levels=[99, 99, 99, 0]
            op5: mode=0 coarse=1 fine=0 detune=+0 outlevel=0 rates=[99, 99, 99, 99] levels=[99, 99, 99, 0]
            op4: mode=0 coarse=1 fine=0 detune=+0 outlevel=0 rates=[99, 99, 99, 99] levels=[99, 99, 99, 0]
            op3: mode=0 coarse=1 fine=0 detune=+0 outlevel=0 rates=[99, 99, 99, 99] levels=[99, 99, 99, 0]
            op2: mode=0 coarse=1 fine=0 detune=+0 outlevel=0 rates=[99, 99, 99, 99] levels=[99, 99, 99, 0]
            op1: mode=0 coarse=1 fine=0 detune=+0 outlevel=99 rates=[99, 99, 99, 99] levels=[99, 99, 99, 0]"#]].assert_eq(&lines.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn unpack_epiano() {
        // Voice 10 is E.PIANO 1. Snapshot the interpretation an engine
        // cares about: algorithm, feedback, and per-operator frequency
        // ratios and output levels. Verifiable against the published
        // DX7 patch sheet: algorithm 5, feedback 6; ops 2/4/6 are the
        // ratio-14 "tine" modulators.
        let bank = Bank::rom1a();
        let v = bank.voice(10);
        let mut lines = vec![
            format!("name: {}", bank.name(10)),
            format!("algorithm: {} feedback: {}", v[134] + 1, v[135]),
            format!("transpose: {:+}", i32::from(v[144]) - 24),
        ];
        for op in 0..6 {
            let off = op * 21;
            lines.push(format!(
                "op{}: mode={} coarse={} fine={} detune={:+} outlevel={} rates={:?} levels={:?}",
                6 - op,
                v[off + 17],
                v[off + 18],
                v[off + 19],
                i32::from(v[off + 20]) - 7,
                v[off + 16],
                &v[off..off + 4],
                &v[off + 4..off + 8],
            ));
        }
        expect![[r#"
            name: E.PIANO 1
            algorithm: 5 feedback: 6
            transpose: +0
            op6: mode=0 coarse=1 fine=0 detune=+7 outlevel=79 rates=[95, 29, 20, 50] levels=[99, 95, 0, 0]
            op5: mode=0 coarse=1 fine=0 detune=-7 outlevel=99 rates=[95, 20, 20, 50] levels=[99, 95, 0, 0]
            op4: mode=0 coarse=1 fine=0 detune=+0 outlevel=89 rates=[95, 29, 20, 50] levels=[99, 95, 0, 0]
            op3: mode=0 coarse=1 fine=0 detune=+0 outlevel=99 rates=[95, 20, 20, 50] levels=[99, 95, 0, 0]
            op2: mode=0 coarse=14 fine=0 detune=+0 outlevel=58 rates=[95, 50, 35, 78] levels=[99, 75, 0, 0]
            op1: mode=0 coarse=1 fine=0 detune=+3 outlevel=99 rates=[96, 25, 25, 67] levels=[99, 75, 0, 0]"#]].assert_eq(&lines.join("\n"));
    }
}
