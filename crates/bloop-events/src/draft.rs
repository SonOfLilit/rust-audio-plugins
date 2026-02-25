//! bloop-events: Note/MIDI event types and allocation-free event buffer.
//!
//! Not part of the build yet — this is a design draft.

// use bloop_core::MusicalTime;

/// Events that flow between sequencer, plugins, and audio graph.
/// Sorted by `time` within a buffer.
#[derive(Clone, Debug, PartialEq)]
pub enum NoteEvent {
    NoteOn { time: u32, channel: u8, note: u8, velocity: f32 },
    NoteOff { time: u32, channel: u8, note: u8 },
    Aftertouch { time: u32, channel: u8, note: u8, pressure: f32 },
    PitchBend { time: u32, channel: u8, value: f32 },
    CC { time: u32, channel: u8, cc: u8, value: f32 },
}

impl NoteEvent {
    pub fn time(&self) -> u32 {
        match self {
            Self::NoteOn { time, .. }
            | Self::NoteOff { time, .. }
            | Self::Aftertouch { time, .. }
            | Self::PitchBend { time, .. }
            | Self::CC { time, .. } => *time,
        }
    }
}

/// Fixed-capacity event buffer. No heap allocation during audio processing.
/// `time` fields are sample offsets within the current audio block.
pub struct EventBuffer {
    events: [Option<NoteEvent>; Self::CAPACITY],
    len: usize,
}

impl EventBuffer {
    pub const CAPACITY: usize = 256;

    pub fn new() -> Self {
        Self { events: [const { None }; Self::CAPACITY], len: 0 }
    }

    pub fn clear(&mut self) { self.len = 0; }

    pub fn push(&mut self, event: NoteEvent) -> bool {
        if self.len >= Self::CAPACITY { return false; }
        self.events[self.len] = Some(event);
        self.len += 1;
        true
    }

    pub fn iter(&self) -> impl Iterator<Item = &NoteEvent> {
        self.events[..self.len].iter().filter_map(|e| e.as_ref())
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Insertion sort — buffer is small and nearly sorted.
    pub fn sort_by_time(&mut self) {
        for i in 1..self.len {
            let mut j = i;
            while j > 0 {
                let a_time = self.events[j - 1].as_ref().map(|e| e.time()).unwrap_or(0);
                let b_time = self.events[j].as_ref().map(|e| e.time()).unwrap_or(0);
                if a_time <= b_time { break; }
                self.events.swap(j - 1, j);
                j -= 1;
            }
        }
    }
}

// --- Test harness ---

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};

    fn parse_event(time: u32, desc: &str) -> NoteEvent {
        if let Some(note) = desc.strip_prefix("on:") {
            NoteEvent::NoteOn { time, channel: 0, note: note.parse().unwrap(), velocity: 1.0 }
        } else if let Some(note) = desc.strip_prefix("off:") {
            NoteEvent::NoteOff { time, channel: 0, note: note.parse().unwrap() }
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
        let lines: Vec<String> = buf.iter().map(|e| match e {
            NoteEvent::NoteOn { time, note, .. } => format!("t={time} on:{note}"),
            NoteEvent::NoteOff { time, note, .. } => format!("t={time} off:{note}"),
            other => format!("{other:?}"),
        }).collect();
        expect.assert_eq(&lines.join("\n"));
    }

    #[test]
    fn event_buffer_sorting() {
        check_buffer(&[
            (64, "on:60"),
            (0, "on:48"),
            (32, "on:36"),
            (64, "off:48"),
        ], expect![[r#"
            t=0 on:48
            t=32 on:36
            t=64 on:60
            t=64 off:48
        "#]]);
    }
}
