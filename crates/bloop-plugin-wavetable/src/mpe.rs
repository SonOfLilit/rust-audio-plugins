//! bloop's notes as the MIDI an MPE controller would send them
//! (bloop#pad-expression).
//!
//! With its MPE switch on, Wavetable plays an MPE lower zone, as the
//! original does (`gin::Synthesiser::setMPE`): channel 1 is the zone's
//! master, whose pitch wheel bends every note by up to 2 semitones, and
//! channels 2 to 16 are its members, each holding one note, whose own
//! pitch wheel bends that note by up to 48, whose controller 74 is its
//! timbre and whose channel pressure is its pressure. bloop names a note
//! by its [`NoteId`] instead, so [`Channels`] hands each note a member
//! channel as it starts and turns its [`Expression`]s into that channel's
//! messages: what reaches the synth is what a controller would have sent,
//! and the gold tests send the original the same.
//!
//! - A note goes on the member channel let go of longest ago, so a
//!   release still ringing keeps its channel for as long as it can.
//! - With all fifteen holding a note, a sixteenth goes on the master
//!   channel: it sounds, and takes no expression of its own.
//! - The track's pitch wheel and controllers go to the master channel.
//! - With MPE off, everything is on channel 1, as before, and an
//!   expression is dropped: the original's legacy mode has no note of its
//!   own to give it to.
//!
//! Allocation-free.

use crate::synth::Event;
use bloop_events::{Expression, NoteEvent, NoteId};

/// The zone's master channel.
pub const MASTER: u8 = 1;
/// How many member channels the zone has: 2 to 16.
pub const MEMBERS: usize = 15;
/// How far a member channel's pitch wheel bends its note, either way:
/// MPE's default, and the original's (`MPEZoneLayout::setLowerZone`).
pub const NOTE_BEND_RANGE: f32 = 48.0;

/// Controller 74, MPE's timbre ("slide").
pub const TIMBRE_CC: u8 = 74;

#[derive(Clone, Copy, Debug, Default)]
struct Member {
    /// The note it holds: its id and pitch.
    holding: Option<(NoteId, u8)>,
    /// When it was last let go of (or never: 0), on [`Channels`]' clock.
    freed: u32,
}

/// Which member channel holds which note, and whether the zone is on.
#[derive(Clone, Debug, Default)]
pub struct Channels {
    mpe: bool,
    members: [Member; MEMBERS],
    clock: u32,
}

impl Channels {
    /// `event` as the synth's events, to `out`: none, one, or (a choke of
    /// a pitch held on several channels) several. `mpe` is whether the
    /// synth plays an MPE zone this block; a change lets go of every
    /// channel, as the synth lets go of every note.
    pub fn translate(&mut self, mpe: bool, event: &NoteEvent, mut out: impl FnMut(Event)) {
        if mpe != self.mpe {
            *self = Self {
                mpe,
                ..Self::default()
            };
        }
        match *event {
            NoteEvent::NoteOn {
                note, velocity, id, ..
            } => {
                let channel = if self.mpe {
                    self.take(id, note.as_u8())
                } else {
                    MASTER
                };
                out(Event::NoteOn {
                    channel,
                    note: note.as_u8(),
                    // A note-on at 0 is a note-off: a strike softer than a step is the softest.
                    velocity: seven_bit(velocity).max(1),
                });
            }
            NoteEvent::NoteOff { note, id, .. } => {
                let channel = self.free(id, note.as_u8());
                out(Event::NoteOff {
                    channel,
                    note: note.as_u8(),
                });
            }
            NoteEvent::PitchBend { value, .. } => out(Event::PitchWheel {
                channel: MASTER,
                value: (value * 16383.0).round().clamp(0.0, 16383.0) as i32,
            }),
            NoteEvent::CC { cc, value, .. } => out(Event::Controller {
                channel: MASTER,
                cc,
                value: seven_bit(value),
            }),
            NoteEvent::Choke { note, .. } => {
                let note = note.as_u8();
                let mut any = false;
                for at in 0..MEMBERS {
                    if self.members[at].holding.is_some_and(|(_, n)| n == note) {
                        any = true;
                        let channel = self.release(at);
                        out(Event::Choke { channel, note });
                    }
                }
                if !any {
                    out(Event::Choke {
                        channel: MASTER,
                        note,
                    });
                }
            }
            NoteEvent::NoteExpression { id, expression, .. } => {
                if !self.mpe || id.is_none() {
                    return;
                }
                let Some(at) = self
                    .members
                    .iter()
                    .position(|m| m.holding.is_some_and(|(held, _)| held == id))
                else {
                    return;
                };
                let channel = member_channel(at);
                out(match expression {
                    Expression::Bend(semitones) => Event::PitchWheel {
                        channel,
                        value: fourteen_bit_bend(semitones / NOTE_BEND_RANGE),
                    },
                    Expression::Timbre(value) => Event::Controller {
                        channel,
                        cc: TIMBRE_CC,
                        value: seven_bit(value),
                    },
                    Expression::Pressure(value) => Event::ChannelPressure {
                        channel,
                        value: seven_bit(value),
                    },
                });
            }
            // Poly pressure is a legacy keyboard's: the original takes it
            // only on an MPE zone's master channel, for nothing bloop
            // sends there.
            NoteEvent::Aftertouch { .. } => {}
        }
    }

    /// A member channel for a note starting: the free one let go of
    /// longest ago, or the master when none is free.
    fn take(&mut self, id: NoteId, note: u8) -> u8 {
        let free = (0..MEMBERS)
            .filter(|&at| self.members[at].holding.is_none())
            .min_by_key(|&at| self.members[at].freed);
        match free {
            Some(at) => {
                self.members[at].holding = Some((id, note));
                member_channel(at)
            }
            None => MASTER,
        }
    }

    /// The channel a note stopping is on, let go of: the member holding
    /// its id, or, for a note with none, the first holding its pitch with
    /// none; the master when no member holds it.
    fn free(&mut self, id: NoteId, note: u8) -> u8 {
        let held = |m: &Member| {
            m.holding
                .is_some_and(|(held, n)| n == note && (held == id || held.is_none()))
        };
        let exact = self
            .members
            .iter()
            .position(|m| m.holding == Some((id, note)));
        match exact.or_else(|| self.members.iter().position(held)) {
            Some(at) => self.release(at),
            None => MASTER,
        }
    }

    fn release(&mut self, at: usize) -> u8 {
        self.clock = self.clock.wrapping_add(1);
        self.members[at] = Member {
            holding: None,
            freed: self.clock,
        };
        member_channel(at)
    }
}

/// The channel of member `at`: 2 to 16.
fn member_channel(at: usize) -> u8 {
    at as u8 + 2
}

/// A 0-to-1 value as the 7-bit MIDI value it would arrive as.
#[must_use]
pub fn seven_bit(value: f32) -> u8 {
    (value * 127.0).round().clamp(0.0, 127.0) as u8
}

/// A bend of `-1` to `1` of a wheel's range as the 14-bit value that
/// `juce::MPEValue::asSignedFloat` reads back as it: 8192 at rest, its
/// lower half 8192 steps and its upper 8191.
#[must_use]
pub fn fourteen_bit_bend(signed: f32) -> i32 {
    let signed = if signed.is_finite() {
        signed.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let value = if signed < 0.0 {
        ((signed + 1.0) * 8192.0).round()
    } else {
        (signed * 8191.0).round() + 8192.0
    };
    (value as i32).clamp(0, 16383)
}
