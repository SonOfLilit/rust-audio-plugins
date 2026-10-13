//! bloop-note: a strongly typed MIDI note number. `no_std`.
//!
//! The one piece of the host's vocabulary the event and plug-in crates need,
//! kept on its own so they do not depend on the host's.

#![no_std]

#[cfg(test)]
extern crate alloc;

/// Strongly typed MIDI note number (0–127).
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Note(u8);

impl Note {
    pub const MIDDLE_C: Self = Self(60);

    /// Pitch-class names indexed by `note % 12`.
    pub const PITCH_CLASS_NAMES: [&'static str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];

    pub const fn new(n: u8) -> Self {
        Self(if n > 127 { 127 } else { n })
    }

    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// Human-readable name in scientific pitch notation (60 = "C4").
    /// Returns a `Display`able wrapper so `no_std` callers avoid allocation.
    pub const fn name(self) -> NoteName {
        NoteName(self)
    }
}

/// `Display`s a [`Note`] as its scientific-pitch name, e.g. "C4", "F#-1".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteName(Note);

impl core::fmt::Display for NoteName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let n = self.0 .0;
        let pitch_class = Note::PITCH_CLASS_NAMES[(n % 12) as usize];
        let octave = (n / 12) as i8 - 1;
        write!(f, "{pitch_class}{octave}")
    }
}

impl core::fmt::Debug for Note {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl core::fmt::Display for Note {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u8> for Note {
    fn from(n: u8) -> Self {
        Self::new(n)
    }
}

impl From<Note> for u8 {
    fn from(n: Note) -> Self {
        n.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    fn check_note(note: Note, expect: Expect) {
        expect.assert_eq(&format!(
            "debug={:?} u8={} display={} name={}",
            note,
            u8::from(note),
            note,
            note.name()
        ));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn note_newtype() {
        check_note(Note::new(60), expect!["debug=60 u8=60 display=60 name=C4"]);
        check_note(Note::MIDDLE_C, expect!["debug=60 u8=60 display=60 name=C4"]);
        check_note(Note::new(0), expect!["debug=0 u8=0 display=0 name=C-1"]);
        check_note(
            Note::new(127),
            expect!["debug=127 u8=127 display=127 name=G9"],
        );
        check_note(
            Note::new(255),
            expect!["debug=127 u8=127 display=127 name=G9"],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn note_names() {
        // Scientific pitch notation: octave increments at C, MIDI 60 = C4.
        check_note(Note::new(21), expect!["debug=21 u8=21 display=21 name=A0"]); // lowest piano key: A0
        check_note(Note::new(59), expect!["debug=59 u8=59 display=59 name=B3"]); // B3 — just below middle C
        check_note(Note::new(61), expect!["debug=61 u8=61 display=61 name=C#4"]); // C#4
        check_note(Note::new(69), expect!["debug=69 u8=69 display=69 name=A4"]); // A440
        check_note(Note::new(11), expect!["debug=11 u8=11 display=11 name=B-1"]);
        // B-1 — top of lowest octave
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn note_conversions() {
        let n: Note = 60u8.into();
        check_note(n, expect!["debug=60 u8=60 display=60 name=C4"]);
        let raw: u8 = n.into();
        assert_eq!(raw, 60);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn note_ordering() {
        assert!(Note::new(60) < Note::new(72));
        assert!(Note::new(0) < Note::new(127));
        assert_eq!(Note::default(), Note::new(0));
    }
}
