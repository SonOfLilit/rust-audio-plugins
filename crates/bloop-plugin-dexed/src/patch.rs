//! DX7 patch data: the packed 128-byte bulk voice format, 32-voice
//! cartridges (sysex banks), and the built-in ROM banks.
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

/// Number of built-in cartridges.
pub const NUM_BANKS: usize = 12;

/// Voices in the drum kit — the only partial bank (its remaining slots
/// are INIT VOICE padding, hidden by `voices`).
const DRUMS_VOICES: usize = 7;

/// Total real voices across all built-in cartridges. Only the LAST bank
/// may be partial — preset index math (bank = preset / [`BANK_SIZE`])
/// relies on every earlier bank being full (test-guarded).
pub const NUM_LIBRARY_VOICES: usize = (NUM_BANKS - 1) * BANK_SIZE + DRUMS_VOICES;

/// One built-in cartridge: display name, short tag (used in browser
/// filter labels and composite voice names), how many of its 32 slots
/// hold real voices, and the raw sysex dump.
pub struct BankInfo {
    pub name: &'static str,
    pub short: &'static str,
    pub voices: usize,
    pub data: &'static [u8],
}

/// The built-in cartridges, in preset order (preset index = bank ×
/// [`BANK_SIZE`] + voice): the eight original Yamaha DX7 factory ROMs
/// (ROM1A "BRASS 1" … ROM4B), from the original-cartridge collection at
/// yamahablackboxes.com, followed by the three Finetales curated banks
/// (EPs & bells / plucks, basses & brass / strings, pads etc.) and a
/// curated drum kit.
pub static BANKS: [BankInfo; NUM_BANKS] = [
    BankInfo {
        name: "ROM1A",
        short: "1A",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM1A.syx"),
    },
    BankInfo {
        name: "ROM1B",
        short: "1B",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM1B.syx"),
    },
    BankInfo {
        name: "ROM2A",
        short: "2A",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM2A.syx"),
    },
    BankInfo {
        name: "ROM2B",
        short: "2B",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM2B.syx"),
    },
    BankInfo {
        name: "ROM3A",
        short: "3A",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM3A.syx"),
    },
    BankInfo {
        name: "ROM3B",
        short: "3B",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM3B.syx"),
    },
    BankInfo {
        name: "ROM4A",
        short: "4A",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM4A.syx"),
    },
    BankInfo {
        name: "ROM4B",
        short: "4B",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/ROM4B.syx"),
    },
    BankInfo {
        name: "Finetales 1",
        short: "FT1",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/Finetales1.syx"),
    },
    BankInfo {
        name: "Finetales 2",
        short: "FT2",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/Finetales2.syx"),
    },
    BankInfo {
        name: "Finetales 3",
        short: "FT3",
        voices: BANK_SIZE,
        data: include_bytes!("../assets/Finetales3.syx"),
    },
    // A curated drum kit, voices extracted from the Dexed cartridge
    // collection (hsjp.eu/downloads/Dexed/Dexed_cart_1.0.zip): KICK DRUM
    // (Yamaha DX5A2), Swissnare (Yamaha DX7IIFD 32B), SYNT TOM 1 (Yamaha
    // YAMAHA05), DMX TOMS 1 + HANDCLAPS5 (CASTLE1), **tom 2** (drums CW),
    // Hi-Hat 1 (DRUMS04).
    BankInfo {
        name: "Drums",
        short: "DRM",
        voices: DRUMS_VOICES,
        data: include_bytes!("../assets/Drums.syx"),
    },
];

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

    /// The built-in cartridge `index` (0..[`NUM_BANKS`], clamped). The
    /// embedded data is validated by tests, so parsing cannot fail at
    /// runtime. Allocation-free: the cartridge is copied onto the stack.
    pub fn builtin(index: usize) -> Self {
        Self::from_sysex(BANKS[index.min(NUM_BANKS - 1)].data).unwrap_or(Self {
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
    fn builtin_bank_names() {
        // Every embedded cartridge, all 32 voice names (8 per row). A bank
        // that failed to parse would show as rows of "?" garbage here, so
        // this snapshot is also the parse validation for the embedded data.
        let text = BANKS
            .iter()
            .enumerate()
            .map(|(b, info)| {
                let bank = Bank::builtin(b);
                let names: Vec<String> = (0..info.voices).map(|i| bank.name(i)).collect();
                let rows: Vec<String> = names
                    .chunks(8)
                    .map(|row| {
                        row.iter()
                            .map(|n| format!("{n:<11}"))
                            .collect::<String>()
                            .trim_end()
                            .to_string()
                    })
                    .collect();
                format!("{} ({})\n  {}", info.name, info.short, rows.join("\n  "))
            })
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
            ROM1A (1A)
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF
            ROM1B (1B)
              PIANO   4  PIANO   5  E.PIANO 2  E.PIANO 3  E.PIANO 4  PIANO 5THS CELESTE    TOY PIANO
              HARPSICH 2 HARPSICH 3 CLAV    2  CLAV    3  E.ORGAN 2  E.ORGAN 3  E.ORGAN 4  E.ORGAN 5
              PIPES   2  PIPES   3  PIPES   4  CALIOPE    ACCORDION  SITAR      GUITAR  3  GUITAR  4
              GUITAR  5  GUITAR  6  LUTE       BANJO      HARP    1  HARP    2  BASS    3  BASS    4
            ROM2A (2A)
              PICCOLO    FLUTE   2  OBOE       CLARINET   SAX BC     BASSOON    STRINGS 4  STRINGS 5
              STRINGS 6  STRINGS 7  STRINGS 8  BRASS   4  BRASS   5  BRASS 6 BC BRASS   7  BRASS   8
              RECORDER   HARMONICA1 HRMNCA2 BC VOICE   2  VOICE   3  GLOKENSPL  VIBE    2  XYLOPHONE
              CHIMES     GONG    1  GONG    2  BELLS      COW BELL   BLOCK      FLEXATONE  LOG DRUM
            ROM2B (2B)
              SYN-LEAD 2 SYN-LEAD 3 SYN-LEAD 4 SYN-LEAD 5 SYN-CLAV 1 SYN-CLAV 2 SYN-CLAV 3 SYN-PIANO
              SYNBRASS 1 SYNBRASS 2 SYNORGAN 1 SYNORGAN 2 SYN-VOX    SYN-ORCH   SYN-BASS 1 SYN-BASS 2
              HARP-FLUTE BELL-FLUTE E.P-BRS BC T.BL-EXPA  CHIME-STRG B.DRM-SNAR SHIMMER    EVOLUTION
              WATER GDN  WASP STING LASER GUN  DESCENT    OCTAVE WAR GRAND PRIX ST.HELENS  EXPLOSION
            ROM3A (3A)
              FLUTE   1  HARPSICH 1 STRG ENS 1 BRIGHT BOW BRASSHORNS BR TRUMPET MARIMBA    E.PIANO 1
              PIANO   1  PIPES   1  E.ORGAN 1  E.BASS  1  CLAV    1  HARMONICA1 JAZZ GUIT1 PRC SYNTH1
              SAX BC     FRETLESS 1 HARP    1  TIMPANI    HEAVYMETAL STEEL DRUM SYN-LEAD 1 VOICES BC
              CLAV ENS   LASERSWEEP TUB ERUPT  GRAND PRIX REFS WHISL TRAIN      BRASS S H  TAKE OFF
            ROM3B (3B)
              PIANO   2  E.GRAND 1  E.GRAND 2  HONKY TONK E.PIANO 2  E.PIANO 3  E.PIANO 4  CELESTE
              FUNK CLAV  CLAV ENS 2 PERC CLAV  HARPSICH 2 E.ORGAN 2  E.ORGAN 3  60-S ORGAN PIPES   2
              PIPES   3  CALIOPE    ACCORDION  TOY PIANO  SITAR      KOTO       JAZZ GUIT2 SPANISHGTR
              FOLK GUIT  LUTE       BANJO      CLAS.GUIT  HARP    2  E.BASS  2  FRETLESS 2 PLUCK BASS
            ROM4A (4A)
              PICCOLO    FLUTE   2  OBOE       CLARINET   BASSOON    PAN FLUTE  LEAD BRASS HORNS
              SOLO TBONE BRASS BC   BRASS 5THS SYNTHBRASS STRG QRT 1 STRG ENS 2 VIOLA SECN STRGS LOW
              HIGH STRGS PIZZ STGS  STG CRSNDO STGS 5THS  BELLS      TUB BELLS  RECORDERS  CHIMES
              VOICES     XYLOPHONE  COWBELL    WOOD BLOCK FLEXATONE  LOG DRUM   GLOKENSPL  VIBE
            ROM4B (4B)
              CLAV-E.PNO PERC BRASS PRC SYNTH2 HARPSI-STG CHIME-STRG HARP-FLUTE BELL-FLUTE STRG-CHIME
              STRG-MARIM STRG-PIZZT ORCHESTRA  LEAD GUITR PIANO-BRS  BRS-CHIME  B.DRM-SNAR E.P-BRS BC
              ORG-BRS BC CLV-BRS BC WHISTLES   FILTER SWP FUNKY RISE WILD BOAR  SHIMMER    EVOLUTION
              WATER GDN  WASP STING MULTI NOTE DESCENT    OCTAVE WAR ..GOTCHA.. ST.HELENS  EXPLOSION
            Finetales 1 (FT1)
              EP 1       EP 2       EP 3       EP 4       EP 5       EP 6       EP 7       EP 8
              EP 9       EP 10      EP 11      EP 12      EP Strings EP MkIII   EP Tremolo Tub Bells
              Tub Erupt  Bass/Bell  Rode Bell  BELLashun  PrinkBell  Bell 1     Bell 2     76 Bells
              Plick Bell GlitterBel Handbell   HllwnBell  KitaroPLK  MinelvaPLK PilgrimPLK Melo Pizz
            Finetales 2 (FT2)
              Celesta    CelestBox  Music Box  Kalimba 1  Kalimba 2  Dulcimer   Dulcibell  Glock 1
              Glock 2    ApolloHarp Sarod      Guitarimba Toy Box    Zither     SanukaitBL Koto
              Hard Xylo  Bass 1     Bass 4     Nice Bass  Superbass2 Superbass3 E Bass 2   Synth Bass
              Pick Bass  FirenzeBSS Sting Bass AirhedBass 3J Brs/Bas PntrBrass  Hiep Brass Moog Brass
            Finetales 3 (FT3)
              Airblower  Sukiyaki   HarvestFLT Low Strgs  High Strgs SloSwell   Clouds Pad Ice Pad
              Cascade    Sensitar   Curved Air Soft Pond  Hi Lyz     AnaAtkStr  Jarre Pad  Ondes Pad
              Vollenweid PPG PadFX  Ana Pad    Thin Atmos Soft Space Ens Pad    Warm Pad   Phaze Pad
              Chip Pizz  12 String  Muted Gtr  PipeOrgan1 PipeOrgan2 Hrpsichord Rigel EP/B BakketDrum
            Drums (DRM)
              KICK DRUM  Swissnare  SYNT TOM 1 DMX TOMS 1 HANDCLAPS5 **tom 2**  Hi-Hat 1"#]].assert_eq(&text);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn raw_4096_bytes_accepted() {
        // The same cartridge without the sysex frame parses identically.
        check_bank(
            &BANKS[0].data[6..6 + BULK_DATA_SIZE],
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
        check_bank(&BANKS[0].data[..100], expect!["BadSize(100)"]);
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
        let bank = Bank::builtin(0);
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
