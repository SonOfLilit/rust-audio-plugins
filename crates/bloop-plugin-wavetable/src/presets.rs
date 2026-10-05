//! Wavetable's factory presets, as far as bloop may ship them
//! (`tools/import.py` says which, and why), and loading one the way
//! `gin::Program::loadProcessor` does: every parameter to its default,
//! then the preset's own, its wavetables and its modulation.

pub use crate::presets_data::{INIT_TABLES, PRESETS};

/// One factory preset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    pub name: &'static str,
    /// The folder it comes in: who made it.
    pub author: &'static str,
    /// Its own tags, as the author wrote them.
    pub tags: &'static str,
    /// The Browser's TYPE for it.
    pub kind: &'static str,
    /// The catalogue entries its oscillators play, `u16::MAX` for one it
    /// leaves at its default.
    pub tables: [u16; 2],
    /// Parameter id and value, for those it sets.
    pub params: &'static [(&'static str, f32)],
    /// Source id, destination id and depth, in the order the original sums
    /// them.
    pub mods: &'static [(&'static str, &'static str, f32)],
}

/// How many presets the preset knob reaches: the factory ones, then Init.
#[must_use]
pub fn count() -> usize {
    PRESETS.len() + 1
}

/// The name of preset `index`; past the factory ones, Init.
#[must_use]
pub fn name(index: usize) -> &'static str {
    PRESETS.get(index).map_or("Init", |p| p.name)
}
