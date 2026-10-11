//! bloop-events: Note/MIDI event types and allocation-free event buffer.

#![no_std]

#[cfg(test)]
extern crate alloc;

pub mod feel;
pub mod midi;

use bloop_core::Note;

/// The controller number of the modulation wheel, MIDI's CC 1: a
/// [`NoteEvent::CC`] with it carries the wheel, 0 at rest to 1 all the way
/// up.
pub const CC_MOD_WHEEL: u8 = 1;

/// One sounding note, for its life: what a [`NoteEvent::NoteExpression`]
/// names, so that two notes on one track — or two strikes of one pitch,
/// one after the other — are told apart.
///
/// The engine gives each note played live an id of its own as it starts
/// it, and its note-off carries the same one. A note nobody will address
/// on its own — a Pattern's, Note Repeat's, the Arp's — has
/// [`NoteId::NONE`]: a plug-in that follows ids tells those apart by
/// their pitch, as every plug-in did before ids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NoteId(pub u32);

impl NoteId {
    /// A note no expression will name.
    pub const NONE: NoteId = NoteId(0);

    /// Whether this is [`NoteId::NONE`].
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// How one held note is being played, beside its pitch and velocity:
/// a finger leaning, sliding or lifting on its pad, or an MPE
/// controller's own message for the note (bloop#pad-expression).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Expression {
    /// Semitones from the note's own pitch, either way: MPE's per-note
    /// pitch bend.
    Bend(f32),
    /// 0 to 1, resting at 0.5: MPE's "slide", controller 74.
    Timbre(f32),
    /// 0 to 1: how hard the note is held, MPE's per-note pressure.
    Pressure(f32),
}

/// Events that flow between sequencer, plugins, and audio graph.
/// Sorted by `time` within a buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoteEvent {
    NoteOn {
        time: u32,
        channel: u8,
        note: Note,
        velocity: f32,
        id: NoteId,
    },
    NoteOff {
        time: u32,
        channel: u8,
        note: Note,
        /// The id its note-on carried, or [`NoteId::NONE`].
        id: NoteId,
    },
    Aftertouch {
        time: u32,
        channel: u8,
        note: Note,
        pressure: f32,
    },
    PitchBend {
        time: u32,
        channel: u8,
        value: f32,
    },
    CC {
        time: u32,
        channel: u8,
        cc: u8,
        value: f32,
    },
    /// Cut everything this track is sounding, now, without a release.
    ///
    /// Distinct from `NoteOff` because the two mean different things: a
    /// note-off says the key came up, and a one-shot sampler pad ignores it
    /// on purpose so that lifting a finger cannot truncate a drum hit.
    /// Choking is the other thing — the closed hat stopping the open one —
    /// and it has to land whatever the pad thinks about note-offs.
    ///
    /// This is the same concept as CLAP's `CLAP_EVENT_NOTE_CHOKE`, whose
    /// documentation describes exactly this case, so a CLAP plugin hosted
    /// later maps straight onto it. VST3 has no equivalent: its adapter has
    /// to degrade this to a note-off, which is right for any instrument that
    /// honours release and wrong only for a third-party one-shot player,
    /// where MIDI CC 120 (All Sound Off) is the nearest thing.
    ///
    /// Per note, like CLAP's, even though Bloop's choke groups run between
    /// tracks and a track is one instrument today: the engine knows which
    /// notes it started, so it can say. Naming the note is what lets a
    /// drum-machine plugin fed by several pads choke one of them without
    /// silencing the others — the case a per-track event could not express.
    /// A choke for a note that has already finished is a no-op.
    Choke {
        time: u32,
        channel: u8,
        note: Note,
    },
    /// One note's own expression, beside the channel's pitch bend and
    /// controllers. A plug-in with no use for it ignores it; one with a
    /// use for it finds the note by `id`, and ignores an id it is not
    /// playing (a note already let go).
    NoteExpression {
        time: u32,
        id: NoteId,
        expression: Expression,
    },
}

impl NoteEvent {
    const ZERO: Self = Self::NoteOff {
        time: 0,
        channel: 0,
        note: Note::new(0),
        id: NoteId::NONE,
    };

    pub fn time(&self) -> u32 {
        match self {
            Self::NoteOn { time, .. }
            | Self::NoteOff { time, .. }
            | Self::Aftertouch { time, .. }
            | Self::PitchBend { time, .. }
            | Self::CC { time, .. }
            | Self::Choke { time, .. }
            | Self::NoteExpression { time, .. } => *time,
        }
    }

    fn time_mut(&mut self) -> &mut u32 {
        match self {
            Self::NoteOn { time, .. }
            | Self::NoteOff { time, .. }
            | Self::Aftertouch { time, .. }
            | Self::PitchBend { time, .. }
            | Self::CC { time, .. }
            | Self::Choke { time, .. }
            | Self::NoteExpression { time, .. } => time,
        }
    }
}

/// Fixed-capacity event buffer. No heap allocation during audio processing.
/// `time` fields are sample offsets within the current audio block.
pub struct EventBuffer {
    events: [NoteEvent; Self::CAPACITY],
    len: usize,
}

impl Default for EventBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBuffer {
    pub const CAPACITY: usize = 256;

    pub fn new() -> Self {
        Self {
            events: [NoteEvent::ZERO; Self::CAPACITY],
            len: 0,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn push(&mut self, event: NoteEvent) -> bool {
        if self.len >= Self::CAPACITY {
            return false;
        }
        self.events[self.len] = event;
        self.len += 1;
        true
    }

    pub fn iter(&self) -> impl Iterator<Item = &NoteEvent> {
        self.events[..self.len].iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut NoteEvent> {
        self.events[..self.len].iter_mut()
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sort events by sample offset.
    /// Order the block's events by when they happen, and — at the same
    /// sample — by what they are: a release before a strike.
    ///
    /// The tie-break is not decoration. The sort is unstable, so two
    /// events sharing a sample could come out either way round, and a
    /// retrigger is exactly that pair: the note off that ends the last
    /// step and the note on that starts the next. Sorted the wrong way
    /// the synth starts the note and then stops it, and the step is
    /// silent. It stayed hidden while the only events sharing a time were
    /// one voice's, and surfaced when the repeat engine moved onto the
    /// audio thread and began placing both at the sample they were due.
    pub fn sort_by_time(&mut self) {
        self.events[..self.len].sort_unstable_by_key(|e| (e.time(), Self::at_same_sample(e)));
    }

    /// What has to happen first when two events share a sample.
    fn at_same_sample(event: &NoteEvent) -> u8 {
        match event {
            NoteEvent::NoteOff { .. } => 0,
            // Everything that is not a strike is state the strike should
            // see: a bend or an aftertouch set before the note starts.
            NoteEvent::NoteOn { .. } => 2,
            // An expression names a note by its id, which the strike at
            // the same sample is what starts.
            NoteEvent::NoteExpression { .. } => 3,
            _ => 1,
        }
    }

    /// Add `offset` to the time of every event from index `from` on.
    /// Song mode generates a block in segments; each segment's events are
    /// produced segment-relative and shifted to their real block offset.
    pub fn shift_times(&mut self, from: usize, offset: u32) {
        for event in &mut self.events[from.min(self.len)..self.len] {
            *event.time_mut() += offset;
        }
    }
}

// --- Test harness ---

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::String, vec::Vec};
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// `on:60`, `off:60`, `on#3:60` (with id 3), `bend#3:1.5`,
    /// `timbre#3:0.8`, `pressure#3:0.4`, `cc:1`.
    fn parse_event(time: u32, desc: &str) -> NoteEvent {
        let (kind, value) = desc.split_once(':').unwrap();
        let (kind, id) = match kind.split_once('#') {
            Some((kind, id)) => (kind, NoteId(id.parse().unwrap())),
            None => (kind, NoteId::NONE),
        };
        let number = |v: &str| -> f32 { v.parse().unwrap() };
        match kind {
            "on" => NoteEvent::NoteOn {
                time,
                channel: 0,
                note: Note::new(value.parse().unwrap()),
                velocity: 1.0,
                id,
            },
            "off" => NoteEvent::NoteOff {
                time,
                channel: 0,
                note: Note::new(value.parse().unwrap()),
                id,
            },
            "bend" => NoteEvent::NoteExpression {
                time,
                id,
                expression: Expression::Bend(number(value)),
            },
            "timbre" => NoteEvent::NoteExpression {
                time,
                id,
                expression: Expression::Timbre(number(value)),
            },
            "pressure" => NoteEvent::NoteExpression {
                time,
                id,
                expression: Expression::Pressure(number(value)),
            },
            "cc" => NoteEvent::CC {
                time,
                channel: 0,
                cc: value.parse().unwrap(),
                value: 1.0,
            },
            _ => panic!("unknown event: {desc}"),
        }
    }

    fn check_buffer(events: &[(u32, &str)], expect: Expect) {
        let mut buf = EventBuffer::new();
        for &(time, desc) in events {
            buf.push(parse_event(time, desc));
        }
        buf.sort_by_time();
        let id_of = |id: &NoteId| {
            if id.is_none() {
                String::new()
            } else {
                format!("#{}", id.0)
            }
        };
        let lines: Vec<String> = buf
            .iter()
            .map(|e| match e {
                NoteEvent::NoteOn { time, note, id, .. } => {
                    format!("t={time} on{}:{note}", id_of(id))
                }
                NoteEvent::NoteOff { time, note, id, .. } => {
                    format!("t={time} off{}:{note}", id_of(id))
                }
                NoteEvent::NoteExpression {
                    time,
                    id,
                    expression,
                } => format!("t={time} {}: {expression:?}", id_of(id)),
                other => format!("{other:?}"),
            })
            .collect();
        expect.assert_eq(&lines.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn event_buffer_sorting() {
        check_buffer(
            &[(64, "on:60"), (0, "on:48"), (32, "on:36")],
            expect![[r#"
            t=0 on:48
            t=32 on:36
            t=64 on:60"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn empty_buffer() {
        check_buffer(&[], expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn overflow_returns_false() {
        let mut buf = EventBuffer::new();
        for i in 0..EventBuffer::CAPACITY {
            assert!(buf.push(NoteEvent::NoteOff {
                time: i as u32,
                channel: 0,
                note: Note::new(60),
                id: NoteId::NONE,
            }));
        }
        assert!(!buf.push(NoteEvent::NoteOff {
            time: 999,
            channel: 0,
            note: Note::new(60),
            id: NoteId::NONE,
        }));
    }

    /// At one sample: the release first, then what the strike should
    /// see (a controller), then the strike, then the expressions that
    /// name it — an expression sorted ahead of its own note-on would
    /// name a note not yet playing.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_expression_follows_the_strike_it_names() {
        check_buffer(
            &[
                (5, "timbre#2:0.8"),
                (5, "on#2:60"),
                (5, "bend#2:-1.5"),
                (5, "cc:1"),
                (5, "off#1:60"),
                (0, "pressure#1:0.25"),
            ],
            expect![[r#"
                t=0 #1: Pressure(0.25)
                t=5 off#1:60
                CC { time: 5, channel: 0, cc: 1, value: 1.0 }
                t=5 on#2:60
                t=5 #2: Timbre(0.8)
                t=5 #2: Bend(-1.5)"#]],
        );
    }

    /// Two strikes of one pitch, one after the other, are two notes:
    /// each note-off names its own strike, and each expression the strike
    /// it was meant for, in the order they were played.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn two_strikes_of_one_pitch_keep_their_own_ids() {
        check_buffer(
            &[
                (0, "on#1:60"),
                (10, "bend#1:0.5"),
                (20, "off#1:60"),
                (20, "on#2:60"),
                (30, "bend#2:-0.5"),
                (30, "bend#1:3"),
                (40, "off#2:60"),
            ],
            expect![[r#"
                t=0 on#1:60
                t=10 #1: Bend(0.5)
                t=20 off#1:60
                t=20 on#2:60
                t=30 #2: Bend(-0.5)
                t=30 #1: Bend(3.0)
                t=40 off#2:60"#]],
        );
    }

    /// An expression's time moves with the segment it is in, as every
    /// other event's does.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_expression_shifts_with_its_segment() {
        let mut buf = EventBuffer::new();
        buf.push(parse_event(0, "on#4:60"));
        buf.push(parse_event(3, "timbre#4:0.5"));
        buf.shift_times(1, 100);
        let times: Vec<u32> = buf.iter().map(NoteEvent::time).collect();
        expect![[r#"
            [
                0,
                103,
            ]
        "#]]
        .assert_debug_eq(&times);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn none_is_the_only_id_that_is_none() {
        expect![[r#"
            [
                true,
                true,
                false,
            ]
        "#]]
        .assert_debug_eq(&[
            NoteId::NONE.is_none(),
            NoteId::default().is_none(),
            NoteId(1).is_none(),
        ]);
    }
}
