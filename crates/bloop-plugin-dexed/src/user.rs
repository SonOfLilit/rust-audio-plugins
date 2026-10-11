//! Voices the user imported, given preset numbers past the built-in
//! library's, so that every place a preset is a number — the synth's
//! Preset parameter, the host's `(kind, preset)` — takes them as it takes
//! a built-in one.
//!
//! The table is the process's: one Project can open on the live engine,
//! on an export's private one and on a manual page's, and all of them
//! must hear the same voice behind the same number. It only grows — a
//! number once given keeps its voice until the process ends — and the
//! same voice added again has the number it had, so importing a file
//! twice uses no more of it.
//!
//! The audio thread reads it when a preset loads: each slot is a
//! [`OnceLock`], read with one atomic load and never locked. Adding takes
//! a lock, on the UI thread only.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock, PoisonError};

use crate::patch::VOICE_SIZE;

/// How many voices can be imported in one run: 128 cartridges.
pub const USER_VOICES: usize = 4096;

/// The preset number of the first imported voice: one past Init.
pub const PRESET_USER: usize = crate::PRESET_COUNT;

/// The last preset number there is.
pub const PRESET_LAST: usize = PRESET_USER + USER_VOICES - 1;

static SLOTS: [OnceLock<[u8; VOICE_SIZE]>; USER_VOICES] = [const { OnceLock::new() }; USER_VOICES];

/// Which slot holds each voice added, for adding it again.
static HELD: Mutex<BTreeMap<[u8; VOICE_SIZE], usize>> = Mutex::new(BTreeMap::new());

/// The preset number `voice` plays under: the one it was given before,
/// or the next free one. `None` when every number is taken.
///
/// UI thread only: it takes a lock.
#[must_use]
pub fn add_voice(voice: &[u8; VOICE_SIZE]) -> Option<usize> {
    let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&slot) = held.get(voice) {
        return Some(PRESET_USER + slot);
    }
    let slot = held.len();
    SLOTS.get(slot)?.set(*voice).ok()?;
    held.insert(*voice, slot);
    Some(PRESET_USER + slot)
}

/// The imported voice preset number `preset` plays; `None` for a number
/// that is not an imported voice's. Allocation-free and lock-free.
#[must_use]
pub fn user_voice(preset: usize) -> Option<[u8; VOICE_SIZE]> {
    SLOTS.get(preset.checked_sub(PRESET_USER)?)?.get().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_same_voice_twice_is_one_number() {
        // Other tests of this process add voices too, so the numbers are
        // read relative to each other rather than from PRESET_USER.
        let mut a = crate::init_voice();
        a[145..155].copy_from_slice(b"USER TEST1");
        let mut b = a;
        b[134] = 4;
        let first = add_voice(&a);
        let again = add_voice(&a);
        let other = add_voice(&b);
        let text = format!(
            "first == again: {}, other differs: {}, both past Init: {}, \
             reads back: {}, Init is none: {:?}, past the end is none: {:?}",
            first == again,
            other != first,
            first.zip(other).is_some_and(|(f, o)| f >= PRESET_USER && o >= PRESET_USER),
            first.and_then(user_voice) == Some(a) && other.and_then(user_voice) == Some(b),
            user_voice(crate::PRESET_INIT),
            user_voice(PRESET_LAST + 1),
        );
        expect_test::expect!["first == again: true, other differs: true, both past Init: true, reads back: true, Init is none: None, past the end is none: None"].assert_eq(&text);
    }
}
