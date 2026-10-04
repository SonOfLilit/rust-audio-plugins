//! MIDI as it arrives on a MIDI IN: bytes, and the messages they spell.
//!
//! A port hands over bytes, not messages. Most hand over whole messages,
//! but Android's `MidiReceiver` may split one across two calls or put
//! several in one, and every port may use *running status* — a run of
//! messages of one kind sending its status byte once. So each port keeps
//! one [`MidiParser`], fed every byte it sends, in order.
//!
//! What a keyboard or controller plays is kept: notes, polyphonic and
//! channel pressure, controllers, pitch bend, program changes. System
//! exclusive dumps, the system common messages and the real-time ones
//! (clock, active sensing) are passed over — clock sync is not MIDI
//! input in the sense this is for.

use core::fmt;

/// One channel message. Channels count from 1, as the manual's
/// Channel menus do; numbers are the wire's, 0–127.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MidiMessage {
    /// A key went down. `velocity` is 1–127: a note-on at velocity 0 is a
    /// note-off, and arrives as [`MidiMessage::NoteOff`].
    NoteOn { channel: u8, note: u8, velocity: u8 },
    /// A key came up, as fast as `velocity` says.
    NoteOff { channel: u8, note: u8, velocity: u8 },
    /// A held key pressed harder or softer (polyphonic aftertouch).
    PolyPressure { channel: u8, note: u8, pressure: u8 },
    /// The whole keyboard pressed harder or softer (channel aftertouch):
    /// what most keyboards that have aftertouch send.
    ChannelPressure { channel: u8, pressure: u8 },
    /// A knob, a fader, a pedal or the mod wheel (controller 1).
    ControlChange { channel: u8, controller: u8, value: u8 },
    /// The pitch wheel: 0–16383, at rest at [`PITCH_BEND_CENTRE`].
    PitchBend { channel: u8, value: u16 },
    ProgramChange { channel: u8, program: u8 },
}

/// Where a pitch wheel rests: no bend.
pub const PITCH_BEND_CENTRE: u16 = 8192;

/// The controller a mod wheel sends.
pub const MOD_WHEEL: u8 = 1;

/// The velocity a note-off is taken to have when it came as a note-on at
/// velocity 0, which says none: the MIDI 1.0 specification's default.
pub const DEFAULT_RELEASE_VELOCITY: u8 = 64;

impl MidiMessage {
    /// The channel it is on, 1–16.
    #[must_use]
    pub fn channel(self) -> u8 {
        match self {
            Self::NoteOn { channel, .. }
            | Self::NoteOff { channel, .. }
            | Self::PolyPressure { channel, .. }
            | Self::ChannelPressure { channel, .. }
            | Self::ControlChange { channel, .. }
            | Self::PitchBend { channel, .. }
            | Self::ProgramChange { channel, .. } => channel,
        }
    }

    /// The message as bytes on the wire, without running status: what a
    /// demo or a test that wants to go through [`MidiParser`] writes.
    /// Out-of-range numbers are masked to seven bits, the channel to 1–16.
    #[must_use]
    pub fn bytes(self) -> ([u8; 3], usize) {
        let ch = self.channel().wrapping_sub(1) & 0x0F;
        let d = |b: u8| b & 0x7F;
        match self {
            Self::NoteOff { note, velocity, .. } => ([0x80 | ch, d(note), d(velocity)], 3),
            Self::NoteOn { note, velocity, .. } => ([0x90 | ch, d(note), d(velocity)], 3),
            Self::PolyPressure { note, pressure, .. } => ([0xA0 | ch, d(note), d(pressure)], 3),
            Self::ControlChange {
                controller, value, ..
            } => ([0xB0 | ch, d(controller), d(value)], 3),
            Self::ProgramChange { program, .. } => ([0xC0 | ch, d(program), 0], 2),
            Self::ChannelPressure { pressure, .. } => ([0xD0 | ch, d(pressure), 0], 2),
            Self::PitchBend { value, .. } => {
                let value = value.min(0x3FFF);
                let (lsb, msb) = ((value & 0x7F) as u8, (value >> 7) as u8);
                ([0xE0 | ch, lsb, msb], 3)
            }
        }
    }
}

/// How a message reads in a log or a gold table: `ch1 on 60 @100`.
impl fmt::Display for MidiMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NoteOn {
                channel,
                note,
                velocity,
            } => write!(f, "ch{channel} on {note} @{velocity}"),
            Self::NoteOff {
                channel,
                note,
                velocity,
            } => write!(f, "ch{channel} off {note} @{velocity}"),
            Self::PolyPressure {
                channel,
                note,
                pressure,
            } => write!(f, "ch{channel} pressure {note} @{pressure}"),
            Self::ChannelPressure { channel, pressure } => {
                write!(f, "ch{channel} pressure @{pressure}")
            }
            Self::ControlChange {
                channel,
                controller,
                value,
            } => write!(f, "ch{channel} cc{controller} {value}"),
            Self::PitchBend { channel, value } => write!(f, "ch{channel} bend {value}"),
            Self::ProgramChange { channel, program } => {
                write!(f, "ch{channel} program {program}")
            }
        }
    }
}

/// Bytes to messages, for one port. See the module documentation.
#[derive(Clone, Copy, Debug, Default)]
pub struct MidiParser {
    /// The channel status byte data bytes belong to, kept for running
    /// status. `None` before the first one, inside a system exclusive
    /// message and after a system common one, which all cancel it.
    status: Option<u8>,
    data: [u8; 2],
    have: usize,
    /// Data bytes still to pass over that belong to no channel message: a
    /// system exclusive dump's (until its end, `usize::MAX`) or a system
    /// common message's.
    skipping: usize,
}

impl MidiParser {
    #[must_use]
    pub const fn new() -> MidiParser {
        MidiParser {
            status: None,
            data: [0; 2],
            have: 0,
            skipping: 0,
        }
    }

    /// One byte: the message it completes, if any.
    pub fn feed(&mut self, byte: u8) -> Option<MidiMessage> {
        match byte {
            // Real-time bytes may come between any two others, even
            // inside another message, and leave it as it was.
            0xF8..=0xFF => None,
            0xF0 => {
                self.status = None;
                self.have = 0;
                self.skipping = usize::MAX;
                None
            }
            0xF7 => {
                self.skipping = 0;
                None
            }
            // System common: song position (2 data bytes), MTC quarter
            // frame and song select (1), tune request and the undefined
            // ones (0). Each cancels running status.
            0xF1..=0xF6 => {
                self.status = None;
                self.have = 0;
                self.skipping = match byte {
                    0xF2 => 2,
                    0xF1 | 0xF3 => 1,
                    _ => 0,
                };
                None
            }
            // A channel status byte; it also ends a system exclusive dump
            // whose end byte was lost.
            0x80..=0xEF => {
                self.status = Some(byte);
                self.have = 0;
                self.skipping = 0;
                None
            }
            data => {
                if self.skipping > 0 {
                    if self.skipping != usize::MAX {
                        self.skipping -= 1;
                    }
                    return None;
                }
                let status = self.status?;
                self.data[self.have] = data;
                self.have += 1;
                if self.have < data_bytes(status) {
                    return None;
                }
                self.have = 0;
                Some(message(status, self.data))
            }
        }
    }

    /// Every message `bytes` completes, in order, to `f`.
    pub fn feed_all(&mut self, bytes: &[u8], mut f: impl FnMut(MidiMessage)) {
        for &byte in bytes {
            if let Some(message) = self.feed(byte) {
                f(message);
            }
        }
    }
}

fn data_bytes(status: u8) -> usize {
    match status & 0xF0 {
        0xC0 | 0xD0 => 1,
        _ => 2,
    }
}

fn message(status: u8, [a, b]: [u8; 2]) -> MidiMessage {
    let channel = (status & 0x0F) + 1;
    match status & 0xF0 {
        0x80 => MidiMessage::NoteOff {
            channel,
            note: a,
            velocity: b,
        },
        0x90 if b == 0 => MidiMessage::NoteOff {
            channel,
            note: a,
            velocity: DEFAULT_RELEASE_VELOCITY,
        },
        0x90 => MidiMessage::NoteOn {
            channel,
            note: a,
            velocity: b,
        },
        0xA0 => MidiMessage::PolyPressure {
            channel,
            note: a,
            pressure: b,
        },
        0xB0 => MidiMessage::ControlChange {
            channel,
            controller: a,
            value: b,
        },
        0xC0 => MidiMessage::ProgramChange {
            channel,
            program: a,
        },
        0xD0 => MidiMessage::ChannelPressure {
            channel,
            pressure: a,
        },
        _ => MidiMessage::PitchBend {
            channel,
            value: u16::from(a) | (u16::from(b) << 7),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::String, vec::Vec};
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Feed the bytes, in the chunks given, to one parser; print what came
    /// out, one message a line.
    fn check(chunks: &[&[u8]], expect: Expect) {
        let mut parser = MidiParser::new();
        let mut out: Vec<String> = Vec::new();
        for chunk in chunks {
            parser.feed_all(chunk, |message| out.push(format!("{message}")));
        }
        expect.assert_eq(&out.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn one_of_each_message() {
        check(
            &[&[
                0x90, 60, 100, // note on
                0x80, 60, 30, // note off with a release velocity
                0xA1, 62, 90, // poly pressure, channel 2
                0xB0, 1, 64, // mod wheel
                0xB0, 7, 127, // volume
                0xC9, 5, // program change, channel 10
                0xDF, 33, // channel pressure, channel 16
                0xE0, 0x00, 0x40, // pitch bend, at rest
                0xE0, 0x00, 0x00, // all the way down
                0xE0, 0x7F, 0x7F, // all the way up
            ]],
            expect![[r#"
                ch1 on 60 @100
                ch1 off 60 @30
                ch2 pressure 62 @90
                ch1 cc1 64
                ch1 cc7 127
                ch10 program 5
                ch16 pressure @33
                ch1 bend 8192
                ch1 bend 0
                ch1 bend 16383"#]],
        );
    }

    /// A run of note-ons sends its status once; a note-on at velocity 0 in
    /// the run is a note-off, which is how most keyboards send them.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn running_status_and_velocity_zero() {
        check(
            &[&[0x90, 60, 100, 64, 90, 60, 0, 67, 80, 64, 0, 67, 0]],
            expect![[r#"
                ch1 on 60 @100
                ch1 on 64 @90
                ch1 off 60 @64
                ch1 on 67 @80
                ch1 off 64 @64
                ch1 off 67 @64"#]],
        );
    }

    /// A message split between two deliveries, and two in one, come out
    /// the same as whole ones. Running status carries over a delivery.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn split_and_joined_deliveries() {
        check(
            &[&[0x90], &[60], &[100, 0x80, 60], &[0, 62], &[0], &[0xE0, 0x00], &[0x50]],
            expect![[r#"
                ch1 on 60 @100
                ch1 off 60 @0
                ch1 off 62 @0
                ch1 bend 10240"#]],
        );
    }

    /// A system exclusive dump is passed over, whatever its length, and it
    /// cancels running status: data bytes after it belong to nothing until
    /// a status byte comes.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn system_exclusive_is_skipped() {
        check(
            &[
                &[0x90, 60, 100],
                &[0xF0, 0x43, 0x10, 0x01, 0x10, 0x02],
                &[0x60, 0x7F, 0xF7],
                &[62, 100],
                &[0x90, 64, 100],
            ],
            expect![[r#"
                ch1 on 60 @100
                ch1 on 64 @100"#]],
        );
    }

    /// A real-time byte inside a dump does not end it.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn real_time_inside_a_dump_does_not_end_it() {
        check(&[&[0xF0, 1, 0xF8, 2, 0xFE, 3, 0xF7, 0x90, 60, 100]], expect!["ch1 on 60 @100"]);
    }

    /// An end-of-dump byte with no dump is nothing: the message it lands
    /// in completes as if it were not there.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_stray_end_of_dump_is_ignored() {
        check(&[&[0x90, 60, 0xF7, 100, 62, 100]], expect![[r#"
            ch1 on 60 @100
            ch1 on 62 @100"#]]);
    }

    /// Running status for the messages of one data byte: three channel
    /// pressures, then two program changes.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn running_status_with_one_data_byte() {
        check(&[&[0xD0, 0x10, 0x20, 0x30, 0xC2, 4, 5]], expect![[r#"
            ch1 pressure @16
            ch1 pressure @32
            ch1 pressure @48
            ch3 program 4
            ch3 program 5"#]]);
    }

    /// A dump whose end byte is lost ends at the next status byte.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_status_byte_ends_an_unterminated_dump() {
        check(&[&[0xF0, 1, 2, 3, 0x91, 60, 100]], expect!["ch2 on 60 @100"]);
    }

    /// Clock and active sensing may land inside a message, and leave it
    /// and running status as they were.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn real_time_bytes_interleave() {
        check(
            &[&[0xF8, 0x90, 0xFE, 60, 0xF8, 100, 0xFA, 62, 0xFC, 90]],
            expect![[r#"
                ch1 on 60 @100
                ch1 on 62 @90"#]],
        );
    }

    /// System common messages are passed over with their data bytes, and
    /// cancel running status: song position's two data bytes are not a
    /// note, and neither is what follows a tune request.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn system_common_is_skipped_and_cancels_running_status() {
        check(
            &[&[0x90, 60, 100, 0xF2, 10, 20, 61, 100, 0xF6, 62, 100, 0x90, 63, 100]],
            expect![[r#"
                ch1 on 60 @100
                ch1 on 63 @100"#]],
        );
    }

    /// Data bytes before any status byte — a port opened mid-message —
    /// belong to nothing.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn stray_data_before_a_status_is_dropped() {
        check(&[&[60, 100, 0x90, 60, 100]], expect!["ch1 on 60 @100"]);
    }

    /// `bytes` is the wire form the parser reads back.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn bytes_round_trip_through_the_parser() {
        let messages = [
            MidiMessage::NoteOn {
                channel: 10,
                note: 36,
                velocity: 127,
            },
            MidiMessage::NoteOff {
                channel: 1,
                note: 36,
                velocity: 0,
            },
            MidiMessage::PolyPressure {
                channel: 3,
                note: 40,
                pressure: 5,
            },
            MidiMessage::ChannelPressure {
                channel: 16,
                pressure: 99,
            },
            MidiMessage::ControlChange {
                channel: 1,
                controller: MOD_WHEEL,
                value: 64,
            },
            MidiMessage::PitchBend {
                channel: 1,
                value: 12_345,
            },
            MidiMessage::ProgramChange {
                channel: 2,
                program: 7,
            },
        ];
        let mut parser = MidiParser::new();
        let mut out: Vec<String> = Vec::new();
        for message in messages {
            let (bytes, len) = message.bytes();
            parser.feed_all(&bytes[..len], |back| {
                out.push(format!("{message} -> {back}"));
            });
        }
        expect![[r#"
            ch10 on 36 @127 -> ch10 on 36 @127
            ch1 off 36 @0 -> ch1 off 36 @0
            ch3 pressure 40 @5 -> ch3 pressure 40 @5
            ch16 pressure @99 -> ch16 pressure @99
            ch1 cc1 64 -> ch1 cc1 64
            ch1 bend 12345 -> ch1 bend 12345
            ch2 program 7 -> ch2 program 7"#]].assert_eq(&out.join("\n"));
    }
}
