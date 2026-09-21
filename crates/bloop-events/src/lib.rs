//! bloop-events: Note/MIDI event types and allocation-free event buffer.

#![no_std]

#[cfg(test)]
extern crate alloc;

use bloop_core::Note;

/// Events that flow between sequencer, plugins, and audio graph.
/// Sorted by `time` within a buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoteEvent {
    NoteOn {
        time: u32,
        channel: u8,
        note: Note,
        velocity: f32,
    },
    NoteOff {
        time: u32,
        channel: u8,
        note: Note,
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
}

impl NoteEvent {
    const ZERO: Self = Self::NoteOff {
        time: 0,
        channel: 0,
        note: Note::new(0),
    };

    pub fn time(&self) -> u32 {
        match self {
            Self::NoteOn { time, .. }
            | Self::NoteOff { time, .. }
            | Self::Aftertouch { time, .. }
            | Self::PitchBend { time, .. }
            | Self::CC { time, .. }
            | Self::Choke { time, .. } => *time,
        }
    }

    fn time_mut(&mut self) -> &mut u32 {
        match self {
            Self::NoteOn { time, .. }
            | Self::NoteOff { time, .. }
            | Self::Aftertouch { time, .. }
            | Self::PitchBend { time, .. }
            | Self::CC { time, .. }
            | Self::Choke { time, .. } => time,
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

    fn parse_event(time: u32, desc: &str) -> NoteEvent {
        if let Some(note) = desc.strip_prefix("on:") {
            NoteEvent::NoteOn {
                time,
                channel: 0,
                note: Note::new(note.parse().unwrap()),
                velocity: 1.0,
            }
        } else if let Some(note) = desc.strip_prefix("off:") {
            NoteEvent::NoteOff {
                time,
                channel: 0,
                note: Note::new(note.parse().unwrap()),
            }
        } else {
            panic!("unknown event: {desc}")
        }
    }

    fn check_buffer(events: &[(u32, &str)], expect: Expect) {
        let mut buf = EventBuffer::new();
        for &(time, desc) in events {
            buf.push(parse_event(time, desc));
        }
        buf.sort_by_time();
        let lines: Vec<String> = buf
            .iter()
            .map(|e| match e {
                NoteEvent::NoteOn { time, note, .. } => format!("t={time} on:{note}"),
                NoteEvent::NoteOff { time, note, .. } => format!("t={time} off:{note}"),
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
                note: Note::new(60)
            }));
        }
        assert!(!buf.push(NoteEvent::NoteOff {
            time: 999,
            channel: 0,
            note: Note::new(60)
        }));
    }
}
