//! MIDI as it arrives on a MIDI IN: bytes, and the messages they spell.
//!
//! A port hands over bytes, not messages. Most hand over whole messages,
//! but Android's `MidiReceiver` may split one across two calls or put
//! several in one, and every port may use *running status* — a run of
//! messages of one kind sending its status byte once. So each port keeps
//! one [`MidiParser`], fed every byte it sends, in order.
//!
//! What a keyboard or controller plays is kept: notes, polyphonic and
//! channel pressure, controllers, pitch bend, program changes, as
//! [`MidiMessage`]s. System exclusive messages, the system common ones
//! and the real-time ones (clock, active sensing) are not played, but
//! [`MidiParser::feed_any`] hands them on as their bytes
//! ([`SystemMessage`]), so the touch log can record what a device sends:
//! the M-VAVE SMC-PAD's DAW preset sends its pads as system exclusive.
//! A system exclusive message is kept to [`SYSEX_CAPACITY`] bytes in the
//! parser itself; nothing allocates.

use core::fmt;

/// One channel message. Channels count from 1, as the manual's
/// Channel menus do; numbers are the wire's, 0–127.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MidiMessage {
    /// A key went down. `velocity` is 1–127: a note-on at velocity 0 is a
    /// note-off, and arrives as [`MidiMessage::NoteOff`].
    NoteOn {
        channel: u8,
        note: u8,
        velocity: u8,
    },
    /// A key came up, as fast as `velocity` says.
    NoteOff {
        channel: u8,
        note: u8,
        velocity: u8,
    },
    /// A held key pressed harder or softer (polyphonic aftertouch).
    PolyPressure {
        channel: u8,
        note: u8,
        pressure: u8,
    },
    /// The whole keyboard pressed harder or softer (channel aftertouch):
    /// what most keyboards that have aftertouch send.
    ChannelPressure {
        channel: u8,
        pressure: u8,
    },
    /// A knob, a fader, a pedal or the mod wheel (controller 1).
    ControlChange {
        channel: u8,
        controller: u8,
        value: u8,
    },
    /// The pitch wheel: 0–16383, at rest at [`PITCH_BEND_CENTRE`].
    PitchBend {
        channel: u8,
        value: u16,
    },
    ProgramChange {
        channel: u8,
        program: u8,
    },
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

/// The most of one system exclusive message a [`MidiParser`] keeps,
/// its `F0` and `F7` included. A longer one is handed on cut to this
/// length, saying how long it was ([`SystemMessage::sent`]).
pub const SYSEX_CAPACITY: usize = 256;

/// Something a port sent that belongs to no channel: a system exclusive
/// message, a system common one (song position, MTC quarter frame, song
/// select, tune request) or a real-time byte (clock, start, stop, active
/// sensing). It is its bytes as they came, status byte first, borrowed
/// from the parser that read them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMessage<'a> {
    /// The bytes kept: all of them, unless a system exclusive message was
    /// longer than [`SYSEX_CAPACITY`].
    pub bytes: &'a [u8],
    /// How many bytes the port sent for it, real-time bytes inside it
    /// not counted. More than `bytes.len()` when it was cut.
    pub sent: usize,
    /// Whether it ended as it should: a system exclusive message whose
    /// `F7` never came, ended by the next status byte, did not.
    pub terminated: bool,
}

impl SystemMessage<'_> {
    /// Whether it was longer than was kept.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.sent > self.bytes.len()
    }

    /// Whether it is a system exclusive message.
    #[must_use]
    pub fn is_exclusive(&self) -> bool {
        self.bytes.first() == Some(&0xF0)
    }

    /// Whether it is a real-time byte.
    #[must_use]
    pub fn is_real_time(&self) -> bool {
        self.bytes.first().is_some_and(|&b| b >= 0xF8)
    }
}

/// Hex bytes, `F0 35 59 10 05 7F F7`, then what was cut or lost:
/// ` … (300 bytes sent)`, ` (no F7)`.
impl fmt::Display for SystemMessage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (at, byte) in self.bytes.iter().enumerate() {
            if at > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{byte:02X}")?;
        }
        if self.truncated() {
            write!(f, " … ({} bytes sent)", self.sent)?;
        }
        if !self.terminated {
            f.write_str(" (no F7)")?;
        }
        Ok(())
    }
}

/// What a byte completed: a channel message, or a system one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Received<'a> {
    Channel(MidiMessage),
    System(SystemMessage<'a>),
}

/// How a message reads in a log or a gold table: a channel message as
/// [`MidiMessage`] spells it, a system one as `sys` and its bytes.
impl fmt::Display for Received<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Channel(message) => write!(f, "{message}"),
            Self::System(message) => write!(f, "sys {message}"),
        }
    }
}

/// Where the parser is among the bytes that belong to no channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum System {
    /// Not in a system message.
    Idle,
    /// Inside a system exclusive message, `sent` bytes in.
    Exclusive { sent: usize },
    /// Inside a system common message, waiting for `need` more data bytes.
    Common { need: usize },
}

/// Bytes to messages, for one port. See the module documentation.
#[derive(Clone, Copy)]
pub struct MidiParser {
    /// The channel status byte data bytes belong to, kept for running
    /// status. `None` before the first one, inside a system exclusive
    /// message and after a system common one, which all cancel it.
    status: Option<u8>,
    data: [u8; 2],
    have: usize,
    system: System,
    /// The system message being read, as far as it is kept.
    kept: [u8; SYSEX_CAPACITY],
    len: usize,
}

impl Default for MidiParser {
    fn default() -> MidiParser {
        MidiParser::new()
    }
}

/// Without the buffer: 256 bytes of it would bury the rest.
impl fmt::Debug for MidiParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MidiParser")
            .field("status", &self.status)
            .field("have", &self.have)
            .field("system", &self.system)
            .field("kept", &self.len)
            .finish_non_exhaustive()
    }
}

impl MidiParser {
    #[must_use]
    pub const fn new() -> MidiParser {
        MidiParser {
            status: None,
            data: [0; 2],
            have: 0,
            system: System::Idle,
            kept: [0; SYSEX_CAPACITY],
            len: 0,
        }
    }

    /// One byte: the channel message it completes, if any. System
    /// messages are passed over; [`MidiParser::feed_any`] has them.
    pub fn feed(&mut self, byte: u8) -> Option<MidiMessage> {
        let mut out = None;
        self.feed_any(byte, |received| {
            if let Received::Channel(message) = received {
                out = Some(message);
            }
        });
        out
    }

    /// Every channel message `bytes` completes, in order, to `f`.
    pub fn feed_all(&mut self, bytes: &[u8], mut f: impl FnMut(MidiMessage)) {
        self.feed_all_any(bytes, |received| {
            if let Received::Channel(message) = received {
                f(message);
            }
        });
    }

    /// Every message `bytes` completes, channel and system, in order, to
    /// `f`.
    pub fn feed_all_any(&mut self, bytes: &[u8], mut f: impl FnMut(Received<'_>)) {
        for &byte in bytes {
            self.feed_any(byte, &mut f);
        }
    }

    /// One byte, and what it completes to `f`: nothing, one message, or
    /// two — a status byte that ends a system exclusive message whose `F7`
    /// was lost hands that on first, then what it completes itself.
    pub fn feed_any(&mut self, byte: u8, mut f: impl FnMut(Received<'_>)) {
        match byte {
            // Real-time bytes may come between any two others, even
            // inside another message, and leave it as it was.
            0xF8..=0xFF => f(Received::System(SystemMessage {
                bytes: &[byte],
                sent: 1,
                terminated: true,
            })),
            // An end with no system exclusive message is nothing: the
            // message it lands in completes as if it were not there.
            0xF7 => {
                if let System::Exclusive { sent } = self.system {
                    self.keep(byte);
                    self.system = System::Idle;
                    f(Received::System(SystemMessage {
                        bytes: &self.kept[..self.len],
                        sent: sent + 1,
                        terminated: true,
                    }));
                }
            }
            // A status byte, channel or system: it ends a system exclusive
            // message whose end byte was lost.
            0x80..=0xF6 => {
                self.interrupted(&mut f);
                self.have = 0;
                if byte < 0xF0 {
                    self.status = Some(byte);
                    return;
                }
                // System exclusive and system common: song position (2
                // data bytes), MTC quarter frame and song select (1), tune
                // request and the undefined ones (0). Each cancels running
                // status.
                self.status = None;
                self.len = 0;
                self.keep(byte);
                self.system = match byte {
                    0xF0 => System::Exclusive { sent: 1 },
                    0xF2 => System::Common { need: 2 },
                    0xF1 | 0xF3 => System::Common { need: 1 },
                    _ => System::Common { need: 0 },
                };
                self.common_complete(&mut f);
            }
            data => match self.system {
                System::Exclusive { sent } => {
                    self.keep(data);
                    self.system = System::Exclusive { sent: sent + 1 };
                }
                System::Common { need } => {
                    self.keep(data);
                    self.system = System::Common {
                        need: need.saturating_sub(1),
                    };
                    self.common_complete(&mut f);
                }
                System::Idle => {
                    let Some(status) = self.status else {
                        return;
                    };
                    self.data[self.have] = data;
                    self.have += 1;
                    if self.have < data_bytes(status) {
                        return;
                    }
                    self.have = 0;
                    f(Received::Channel(message(status, self.data)));
                }
            },
        }
    }

    /// A byte of the system message being read, if there is room for it.
    fn keep(&mut self, byte: u8) {
        if let Some(slot) = self.kept.get_mut(self.len) {
            *slot = byte;
            self.len += 1;
        }
    }

    /// A system common message with all its data bytes is handed on.
    fn common_complete(&mut self, f: &mut impl FnMut(Received<'_>)) {
        if self.system == (System::Common { need: 0 }) {
            self.system = System::Idle;
            f(Received::System(SystemMessage {
                bytes: &self.kept[..self.len],
                sent: self.len,
                terminated: true,
            }));
        }
    }

    /// A status byte came: a system exclusive message it interrupts is
    /// handed on as it stood, and a system common message short of its
    /// data bytes is dropped.
    fn interrupted(&mut self, f: &mut impl FnMut(Received<'_>)) {
        if let System::Exclusive { sent } = self.system {
            f(Received::System(SystemMessage {
                bytes: &self.kept[..self.len],
                sent,
                terminated: false,
            }));
        }
        self.system = System::Idle;
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
            &[
                &[0x90],
                &[60],
                &[100, 0x80, 60],
                &[0, 62],
                &[0],
                &[0xE0, 0x00],
                &[0x50],
            ],
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
        check(
            &[&[0xF0, 1, 0xF8, 2, 0xFE, 3, 0xF7, 0x90, 60, 100]],
            expect!["ch1 on 60 @100"],
        );
    }

    /// An end-of-dump byte with no dump is nothing: the message it lands
    /// in completes as if it were not there.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_stray_end_of_dump_is_ignored() {
        check(
            &[&[0x90, 60, 0xF7, 100, 62, 100]],
            expect![[r#"
            ch1 on 60 @100
            ch1 on 62 @100"#]],
        );
    }

    /// Running status for the messages of one data byte: three channel
    /// pressures, then two program changes.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn running_status_with_one_data_byte() {
        check(
            &[&[0xD0, 0x10, 0x20, 0x30, 0xC2, 4, 5]],
            expect![[r#"
            ch1 pressure @16
            ch1 pressure @32
            ch1 pressure @48
            ch3 program 4
            ch3 program 5"#]],
        );
    }

    /// A dump whose end byte is lost ends at the next status byte.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_status_byte_ends_an_unterminated_dump() {
        check(
            &[&[0xF0, 1, 2, 3, 0x91, 60, 100]],
            expect!["ch2 on 60 @100"],
        );
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
            &[&[
                0x90, 60, 100, 0xF2, 10, 20, 61, 100, 0xF6, 62, 100, 0x90, 63, 100,
            ]],
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

    /// Feed the bytes, in the chunks given; print every message, channel
    /// and system, one a line.
    fn check_any(chunks: &[&[u8]], expect: Expect) {
        let mut parser = MidiParser::new();
        let mut out: Vec<String> = Vec::new();
        for chunk in chunks {
            parser.feed_all_any(chunk, |received| out.push(format!("{received}")));
        }
        expect.assert_eq(&out.join("\n"));
    }

    /// The SMC-PAD's DAW-preset pad, split across three deliveries, comes
    /// out whole, once; running status is cancelled by it as before.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_split_sysex_comes_out_whole() {
        check_any(
            &[
                &[0x90, 60, 100, 0xF0, 0x35],
                &[0x59, 0x10],
                &[0x05, 0x7F, 0xF7, 62, 100],
                &[0x90, 64, 100],
            ],
            expect![[r#"
                ch1 on 60 @100
                sys F0 35 59 10 05 7F F7
                ch1 on 64 @100"#]],
        );
    }

    /// Clock, start and active sensing inside a dump come out where they
    /// fell, and the dump around them is unbroken.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn real_time_inside_a_sysex_comes_out_beside_it() {
        check_any(
            &[&[0xF0, 0x35, 0xF8, 0x59, 0xFA, 0x10, 0x01, 0xFE, 0x00, 0xF7]],
            expect![[r#"
                sys F8
                sys FA
                sys FE
                sys F0 35 59 10 01 00 F7"#]],
        );
    }

    /// A dump whose F7 is lost is handed on, marked, when the next status
    /// byte ends it — and the note that status byte starts still plays.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_unterminated_sysex_then_a_note() {
        check_any(
            &[&[0xF0, 0x35, 0x59, 0x10, 0x05, 0x91, 60, 100]],
            expect![[r#"
                sys F0 35 59 10 05 (no F7)
                ch2 on 60 @100"#]],
        );
        // A tune request ends one too: two messages from one byte.
        check_any(
            &[&[0xF0, 1, 2, 0xF6, 0x90, 60, 1]],
            expect![[r#"
            sys F0 01 02 (no F7)
            sys F6
            ch1 on 60 @1"#]],
        );
    }

    /// A dump longer than the parser keeps comes out cut, saying how long
    /// it was; the next one is whole again.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_over_long_sysex_is_truncated() {
        let mut long = alloc::vec![0xF0];
        long.extend((0..300u32).map(|n| (n % 128) as u8));
        long.push(0xF7);
        let mut parser = MidiParser::new();
        let mut out: Vec<String> = Vec::new();
        parser.feed_all_any(&long, |received| {
            if let Received::System(message) = received {
                out.push(format!(
                    "kept {} of {}, truncated {}, last kept {:02X}, ends {}",
                    message.bytes.len(),
                    message.sent,
                    message.truncated(),
                    message.bytes[message.bytes.len() - 1],
                    message.terminated
                ));
            }
        });
        parser.feed_all_any(&[0xF0, 0x7E, 0xF7, 0x90, 60, 100], |received| {
            out.push(format!("{received}"));
        });
        expect![[r#"
            kept 256 of 302, truncated true, last kept 7E, ends true
            sys F0 7E F7
            ch1 on 60 @100"#]]
        .assert_eq(&out.join("\n"));
    }

    /// Song position, MTC, song select and tune request come out with
    /// their data bytes; one cut short by a status byte is dropped.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn system_common_comes_out_with_its_data() {
        check_any(
            &[&[
                0xF2, 0x10, 0x20, 0xF1, 0x35, 0xF3, 0x02, 0xF6, 0xF2, 0x10, 0x90, 60, 100, 0xF7,
            ]],
            expect![[r#"
                sys F2 10 20
                sys F1 35
                sys F3 02
                sys F6
                ch1 on 60 @100"#]],
        );
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
            ch2 program 7 -> ch2 program 7"#]]
        .assert_eq(&out.join("\n"));
    }
}
