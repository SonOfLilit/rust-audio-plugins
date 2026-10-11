//! Reading a `.syx` file of the user's: the voices in it, and what was
//! wrong with it that did not stop it loading.
//!
//! Two kinds of message carry voices, both "bulk dumps" (sub-status 0, any
//! device number):
//!
//! - the 32-voice cartridge, `F0 43 0n 09 20 00`, 4096 bytes of packed
//!   voices (128 each, [`crate::patch::unpack_patch`] reads them — the
//!   decoder the built-in cartridges go through), a checksum, `F7`;
//! - the single voice, `F0 43 0n 00 01 1B`, 155 bytes in the engine's own
//!   order (VCED), a checksum, `F7`.
//!
//! A file can hold any number of them, one after another, and other
//! messages between: those are skipped and said. 4096 bytes with no frame
//! at all are a cartridge too, as Dexed reads them. A checksum that is
//! wrong or missing, an `F7` that never comes, bytes outside the messages
//! — each is said, and the voices load as they are. Every voice, of
//! either kind, keeps each field to the bits the format gives it
//! (`patch::into_bounds`), so that a damaged file cannot send the engine
//! past the end of a table.

use crate::patch::{unpack_patch, unpack_single, PACKED_VOICE_SIZE, VOICE_SIZE};

/// The bytes a 32-voice cartridge's packed voices take.
const BULK_BYTES: usize = PACKED_VOICE_SIZE * 32;
/// The bytes a single voice dump's voice takes: every parameter of the
/// engine's layout but the operator switches.
const VCED_BYTES: usize = VOICE_SIZE - 1;
/// The bytes before a dump's data: `F0`, maker, sub-status and device,
/// format, two bytes of count.
const HEADER: usize = 6;
/// Yamaha's maker byte.
const YAMAHA: u8 = 0x43;
/// The format bytes, and the counts each says (high and low seven bits).
const FORMAT_BANK: (u8, [u8; 2]) = (0x09, [0x20, 0x00]);
const FORMAT_VOICE: (u8, [u8; 2]) = (0x00, [0x01, 0x1B]);

/// Something a file had that its voices loaded in spite of, or that was
/// left out of it. `message` counts the file's SysEx messages from 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Remark {
    /// The checksum after a dump's data is not the one its data sums to.
    BadChecksum { message: usize },
    /// The dump ends before its checksum.
    NoChecksum { message: usize },
    /// The dump's `F7` never came: the next message, or the file's end,
    /// ended it.
    Unterminated { message: usize },
    /// The dump ends partway: `voices` whole voices of it were there.
    Truncated { message: usize, voices: usize },
    /// Bytes between a dump's checksum and its `F7`.
    ExtraBytes { message: usize, bytes: usize },
    /// A message that holds no voices bloop reads: another maker's, or
    /// another kind of Yamaha message. Its first bytes, as they were.
    Skipped { message: usize, starts: Vec<u8> },
    /// Bytes outside any message.
    Stray { bytes: usize },
}

impl std::fmt::Display for Remark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Remark::BadChecksum { message } => {
                write!(f, "message {message}'s checksum is wrong")
            }
            Remark::NoChecksum { message } => write!(f, "message {message} has no checksum"),
            Remark::Unterminated { message } => write!(f, "message {message} never ends"),
            Remark::Truncated { message, voices } => {
                write!(f, "message {message} is cut short after {voices} voices")
            }
            Remark::ExtraBytes { message, bytes } => {
                write!(f, "message {message} has {bytes} bytes more than a dump")
            }
            Remark::Skipped { message, .. } => {
                write!(f, "message {message} holds no voices and was left out")
            }
            Remark::Stray { bytes } => write!(f, "{bytes} bytes outside any message were left out"),
        }
    }
}

/// The voices a file holds, in the engine's layout, in the file's order,
/// and what was wrong with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Voices {
    pub voices: Vec<[u8; VOICE_SIZE]>,
    pub remarks: Vec<Remark>,
}

/// Why a file gave no voices at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SysexError {
    /// Nothing in it.
    Empty,
    /// Not one voice in it; what it held instead.
    NoVoices(Vec<Remark>),
}

impl std::fmt::Display for SysexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SysexError::Empty => write!(f, "the file is empty"),
            SysexError::NoVoices(_) => write!(f, "it holds no FM voices"),
        }
    }
}

/// The voice's name, as the Browser shows it: its ten characters trimmed,
/// a byte that is no printable character shown as `?`.
#[must_use]
pub fn voice_name(voice: &[u8; VOICE_SIZE]) -> String {
    voice[145..155]
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

/// The checksum a dump's data carries: what brings the data's sum to a
/// multiple of 128.
#[must_use]
pub fn checksum(data: &[u8]) -> u8 {
    let sum = data.iter().fold(0u8, |sum, &b| sum.wrapping_add(b));
    sum.wrapping_neg() & 0x7f
}

/// Every voice in a `.syx` file's bytes.
///
/// # Errors
/// The file is empty, or holds no voice.
pub fn read_sysex(bytes: &[u8]) -> Result<Voices, SysexError> {
    if bytes.is_empty() {
        return Err(SysexError::Empty);
    }
    let mut out = Voices {
        voices: Vec::new(),
        remarks: Vec::new(),
    };
    if bytes.len() == BULK_BYTES && bytes[0] != 0xf0 {
        bank_voices(bytes, &mut out.voices);
    } else {
        read_messages(bytes, &mut out);
    }
    if out.voices.is_empty() {
        return Err(SysexError::NoVoices(out.remarks));
    }
    Ok(out)
}

/// Read the messages `bytes` hold, one after another.
fn read_messages(bytes: &[u8], out: &mut Voices) {
    let mut stray = 0;
    let mut at = 0;
    let mut message = 0;
    while at < bytes.len() {
        if bytes[at] != 0xf0 {
            stray += 1;
            at += 1;
            continue;
        }
        message += 1;
        // A message runs to its F7, or to the next status byte — the
        // next F0 above all — when its F7 was lost. A real-time byte
        // (F8 and up: a clock, active sensing) may come in the middle of
        // one, and is no part of it.
        let body_end = bytes[at + 1..]
            .iter()
            .position(|&b| b & 0x80 != 0 && b < 0xf8)
            .map_or(bytes.len(), |end| at + 1 + end);
        let terminated = bytes.get(body_end) == Some(&0xf7);
        let body: Vec<u8> = bytes[at..body_end]
            .iter()
            .copied()
            .filter(|&b| b < 0xf8)
            .collect();
        dump(&body, message, terminated, out);
        at = if terminated { body_end + 1 } else { body_end };
    }
    if stray > 0 {
        out.remarks.push(Remark::Stray { bytes: stray });
    }
}

/// One message, from its `F0` up to (not including) its `F7`.
fn dump(body: &[u8], message: usize, terminated: bool, out: &mut Voices) {
    let header = body.get(..HEADER);
    let format = match header {
        Some(&[_, YAMAHA, status, format, high, low]) if status & 0xf0 == 0 => {
            [FORMAT_BANK, FORMAT_VOICE]
                .into_iter()
                .find(|(byte, count)| *byte == format && *count == [high, low])
        }
        _ => None,
    };
    let Some((format, _)) = format else {
        out.remarks.push(Remark::Skipped {
            message,
            starts: body.iter().take(HEADER).copied().collect(),
        });
        return;
    };
    let size = if format == FORMAT_BANK.0 {
        BULK_BYTES
    } else {
        VCED_BYTES
    };
    let data = &body[HEADER..];
    if !terminated {
        out.remarks.push(Remark::Unterminated { message });
    }
    if data.len() < size {
        if format == FORMAT_BANK.0 {
            let whole = data.len() / PACKED_VOICE_SIZE;
            bank_voices(&data[..whole * PACKED_VOICE_SIZE], &mut out.voices);
            out.remarks.push(Remark::Truncated {
                message,
                voices: whole,
            });
        } else {
            out.remarks.push(Remark::Truncated { message, voices: 0 });
        }
        return;
    }
    let (data, rest) = data.split_at(size);
    match rest.first() {
        None => out.remarks.push(Remark::NoChecksum { message }),
        Some(&sum) if sum != checksum(data) => out.remarks.push(Remark::BadChecksum { message }),
        Some(_) => {}
    }
    if rest.len() > 1 {
        out.remarks.push(Remark::ExtraBytes {
            message,
            bytes: rest.len() - 1,
        });
    }
    if format == FORMAT_BANK.0 {
        bank_voices(data, &mut out.voices);
    } else {
        let mut vced = [0u8; VCED_BYTES];
        vced.copy_from_slice(data);
        out.voices.push(unpack_single(&vced));
    }
}

/// The packed voices `data` holds, whole ones only, unpacked.
fn bank_voices(data: &[u8], voices: &mut Vec<[u8; VOICE_SIZE]>) {
    for packed in data.chunks_exact(PACKED_VOICE_SIZE) {
        let mut bulk = [0u8; PACKED_VOICE_SIZE];
        bulk.copy_from_slice(packed);
        voices.push(unpack_patch(&bulk));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::{Bank, BANKS, BANK_SIZE};
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// The voices of `bytes` by name, and every remark, or the error.
    fn check(bytes: &[u8], expect: Expect) {
        let text = match read_sysex(bytes) {
            Ok(read) => {
                let names: Vec<String> = read.voices.iter().map(voice_name).collect();
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
                let remarks: Vec<String> = read
                    .remarks
                    .iter()
                    .map(|r| format!("remark: {r} ({r:?})"))
                    .collect();
                format!(
                    "{} voices\n  {}\n{}",
                    read.voices.len(),
                    rows.join("\n  "),
                    remarks.join("\n")
                )
                .trim_end()
                .to_string()
            }
            Err(error) => format!("error: {error} ({error:?})"),
        };
        expect.assert_eq(&text);
    }

    /// ROM1A as it is shipped: a 4104-byte cartridge.
    fn rom1a() -> Vec<u8> {
        BANKS[0].data.to_vec()
    }

    /// A single voice dump of `voice`, on device `device`.
    fn single(voice: &[u8; VOICE_SIZE], device: u8) -> Vec<u8> {
        let data = &voice[..VCED_BYTES];
        let mut out = vec![0xf0, YAMAHA, device, 0x00, 0x01, 0x1b];
        out.extend_from_slice(data);
        out.push(checksum(data));
        out.push(0xf7);
        out
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_cartridge() {
        check(&rom1a(), expect![[r#"
            32 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_shipped_cartridges_checksums() {
        // Every cartridge bloop ships is a well-formed dump: no remark.
        let text: Vec<String> = BANKS
            .iter()
            .map(|bank| {
                let read = read_sysex(bank.data).map(|r| (r.voices.len(), r.remarks));
                format!("{}: {read:?}", bank.name)
            })
            .collect();
        expect![[r#"
            ROM1A: Ok((32, []))
            ROM1B: Ok((32, []))
            ROM2A: Ok((32, []))
            ROM2B: Ok((32, []))
            ROM3A: Ok((32, []))
            ROM3B: Ok((32, []))
            ROM4A: Ok((32, []))
            ROM4B: Ok((32, []))
            Finetales 1: Ok((32, []))
            Finetales 2: Ok((32, []))
            Finetales 3: Ok((32, []))
            Drums: Ok((32, []))"#]].assert_eq(&text.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn raw_voice_data_without_a_frame() {
        check(&rom1a()[HEADER..HEADER + BULK_BYTES], expect![[r#"
            32 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_single_voice() {
        // E.PIANO 1 sent alone, on device 3 (sub-status byte 0x02 = 0000
        // 0010: device numbers are the low nibble, any of them).
        let voice = Bank::builtin(0).voice(10);
        check(&single(&voice, 0x02), expect![[r#"
            1 voices
              E.PIANO 1"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn several_dumps_one_after_another() {
        // A cartridge, a voice, another cartridge: 65 voices in file order.
        let mut bytes = rom1a();
        bytes.extend(single(&Bank::builtin(2).voice(0), 0));
        bytes.extend_from_slice(BANKS[8].data);
        check(&bytes, expect![[r#"
            65 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF
              PICCOLO    EP 1       EP 2       EP 3       EP 4       EP 5       EP 6       EP 7
              EP 8       EP 9       EP 10      EP 11      EP 12      EP Strings EP MkIII   EP Tremolo
              Tub Bells  Tub Erupt  Bass/Bell  Rode Bell  BELLashun  PrinkBell  Bell 1     Bell 2
              76 Bells   Plick Bell GlitterBel Handbell   HllwnBell  KitaroPLK  MinelvaPLK PilgrimPLK
              Melo Pizz"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_wrong_checksum_loads_and_is_said() {
        let mut bytes = rom1a();
        let at = bytes.len() - 2;
        bytes[at] ^= 0x01;
        check(&bytes, expect![[r#"
            32 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF
            remark: message 1's checksum is wrong (BadChecksum { message: 1 })"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_missing_checksum_and_a_missing_f7() {
        // The checksum gone (F7 straight after the data), then a dump whose
        // F7 is lost: the next F0 ends it.
        let mut bytes = rom1a();
        bytes.remove(bytes.len() - 2);
        let mut lost = single(&Bank::builtin(0).voice(0), 0);
        lost.pop();
        bytes.extend(lost);
        bytes.extend(single(&Bank::builtin(0).voice(1), 0));
        check(&bytes, expect![[r#"
            34 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF
              BRASS   1  BRASS   2
            remark: message 1 has no checksum (NoChecksum { message: 1 })
            remark: message 2 never ends (Unterminated { message: 2 })"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn truncated() {
        // Cut in the middle of the 13th voice: twelve whole ones load.
        let bytes = &rom1a()[..HEADER + 12 * PACKED_VOICE_SIZE + 50];
        check(bytes, expect![[r#"
            12 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1
            remark: message 1 never ends (Unterminated { message: 1 })
            remark: message 1 is cut short after 12 voices (Truncated { message: 1, voices: 12 })"#]]);
        // A single voice cut short loads nothing, and nothing else is
        // there.
        let voice = single(&Bank::builtin(0).voice(0), 0);
        check(&voice[..100], expect!["error: it holds no FM voices (NoVoices([Unterminated { message: 1 }, Truncated { message: 1, voices: 0 }]))"]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn other_messages_are_skipped() {
        // Another maker's SysEx, a Yamaha parameter change (sub-status 1),
        // a Yamaha dump of another format (a performance, format 0x01)
        // and a cartridge whose count is not 4096: all left out, the
        // cartridge after them read.
        let mut bytes = vec![0xf0, 0x41, 0x10, 0x42, 0x12, 0x00, 0x7f, 0xf7];
        bytes.extend([0xf0, YAMAHA, 0x10, 0x01, 0x22, 0x40, 0xf7]);
        bytes.extend([0xf0, YAMAHA, 0x00, 0x01, 0x00, 0x5e, 0x00, 0xf7]);
        bytes.extend([0xf0, YAMAHA, 0x00, 0x09, 0x10, 0x00, 0x00, 0xf7]);
        bytes.extend(rom1a());
        check(&bytes, expect![[r#"
            32 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF
            remark: message 1 holds no voices and was left out (Skipped { message: 1, starts: [240, 65, 16, 66, 18, 0] })
            remark: message 2 holds no voices and was left out (Skipped { message: 2, starts: [240, 67, 16, 1, 34, 64] })
            remark: message 3 holds no voices and was left out (Skipped { message: 3, starts: [240, 67, 0, 1, 0, 94] })
            remark: message 4 holds no voices and was left out (Skipped { message: 4, starts: [240, 67, 0, 9, 16, 0] })"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn real_time_bytes_inside_a_dump() {
        // A capture of a MIDI cable: clock (F8) and active sensing (FE)
        // between the data bytes. The dump reads whole, checksum and all.
        let mut bytes = rom1a();
        bytes.insert(1000, 0xf8);
        bytes.insert(3000, 0xfe);
        bytes.insert(10, 0xf8);
        let read = read_sysex(&bytes).map(|r| (r.voices.len(), r.remarks));
        expect!["Ok((32, []))"].assert_eq(&format!("{read:?}"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn junk_around_the_dump() {
        // A text header in front, a newline and zero padding behind, and
        // a few bytes between the checksum and F7.
        let mut bytes = b"DX cart\n".to_vec();
        let mut cart = rom1a();
        let f7 = cart.pop();
        cart.extend([0x00, 0x00]);
        cart.extend(f7);
        bytes.extend(cart);
        bytes.extend([b'\n', 0, 0, 0]);
        check(&bytes, expect![[r#"
            32 voices
              BRASS   1  BRASS   2  BRASS   3  STRINGS 1  STRINGS 2  STRINGS 3  ORCHESTRA  PIANO   1
              PIANO   2  PIANO   3  E.PIANO 1  GUITAR  1  GUITAR  2  SYN-LEAD 1 BASS    1  BASS    2
              E.ORGAN 1  PIPES   1  HARPSICH 1 CLAV    1  VIBE    1  MARIMBA    KOTO       FLUTE   1
              ORCH-CHIME TUB BELLS  STEEL DRUM TIMPANI    REFS WHISL VOICE   1  TRAIN      TAKE OFF
            remark: message 1 has 2 bytes more than a dump (ExtraBytes { message: 1, bytes: 2 })
            remark: 12 bytes outside any message were left out (Stray { bytes: 12 })"#]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn nothing_to_read() {
        check(&[], expect!["error: the file is empty (Empty)"]);
        check(b"not a sysex file at all", expect!["error: it holds no FM voices (NoVoices([Stray { bytes: 23 }]))"]);
        check(&[0xf0, 0x41, 0x10, 0x42, 0xf7], expect!["error: it holds no FM voices (NoVoices([Skipped { message: 1, starts: [240, 65, 16, 66] }]))"]);
        // A Sample file picked by mistake: RIFF bytes, no F0 anywhere.
        check(b"RIFF\0\0\0\0WAVEfmt ", expect!["error: it holds no FM voices (NoVoices([Stray { bytes: 16 }]))"]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn damaged_values_are_kept_to_their_bits() {
        // A single voice with its algorithm at 40 (five bits: 8), OP6's
        // detune at 20 (four bits: 4), its pitch EG's first level at 120
        // (the top is 99), OP6's EG rate at 127 (kept: the engine reads
        // it, and a shipped cartridge has one) and its name untouched.
        let mut voice = Bank::builtin(0).voice(10);
        voice[134] = 40;
        voice[20] = 20;
        voice[130] = 120;
        voice[0] = 127;
        let read = read_sysex(&single(&voice, 0)).map(|r| {
            let v = r.voices[0];
            format!(
                "alg byte {} detune byte {} pitch EG L1 {} OP6 R1 {} name {:?} {:?}",
                v[134],
                v[20],
                v[130],
                v[0],
                voice_name(&v),
                r.remarks
            )
        });
        expect![[r#"Ok("alg byte 8 detune byte 4 pitch EG L1 99 OP6 R1 127 name \"E.PIANO 1\" []")"#]].assert_eq(&format!("{read:?}"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn every_shipped_voice_reads_as_the_built_in_one() {
        // The round trip: each cartridge bloop ships, read as a file of the
        // user's, gives the very bytes the built-in library plays.
        let mut differ = Vec::new();
        for (b, bank) in BANKS.iter().enumerate() {
            let read = read_sysex(bank.data).map(|r| r.voices).unwrap_or_default();
            let built_in = Bank::builtin(b);
            for v in 0..BANK_SIZE {
                if read.get(v) != Some(&built_in.voice(v)) {
                    differ.push(format!("{} {v}", bank.name));
                }
            }
        }
        expect!["differ: []"].assert_eq(&format!("differ: {differ:?}"));
    }
}
