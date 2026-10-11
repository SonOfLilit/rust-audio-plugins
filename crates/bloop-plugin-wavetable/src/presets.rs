//! Wavetable's factory presets, every one SocaLabs ships
//! (`tools/import.py` brings them in), and loading one the way
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

impl Preset {
    /// Whether the web build carries every table it plays: the tables of
    /// the oscillators it switches on. One it lacks plays silent there.
    #[must_use]
    pub fn on_web(&self) -> bool {
        let enabled = |uid: &str, default: f32| {
            self.params
                .iter()
                .find(|(id, _)| *id == uid)
                .map_or(default, |(_, value)| *value)
                != 0.0
        };
        let on = [enabled("osc1enable", 1.0), enabled("osc2enable", 0.0)];
        self.tables.iter().zip(on).all(|(&table, on)| {
            !on || table == u16::MAX
                || crate::bank::catalogue()
                    .get(usize::from(table))
                    .is_some_and(|entry| entry.on_web)
        })
    }
}

/// How many factory presets there are, Init the last.
#[must_use]
pub fn count() -> usize {
    PRESETS.len() + 1
}

/// The name of preset `index`: a factory one, Init one past them, then
/// the user's own ([`user`]); past those, Init.
#[must_use]
pub fn name(index: usize) -> &'static str {
    PRESETS
        .get(index)
        .map(|p| p.name)
        .or_else(|| user(index).map(|p| p.name.as_str()))
        .unwrap_or("Init")
}

// ——— the user's own ———

/// How many presets of the user's one run of the app can hold.
pub const USER_PRESETS: usize = 1024;

/// A route as a preset holds it, resolved: the slot the matrix runs.
pub use crate::modmatrix::RouteSlot;

/// A preset that came from outside bloop: a Wavetable `.xml` the user
/// imported ([`crate::preset_file::read`]), or a WAV imported as a table,
/// which plays as Init with that table on OSC 1.
#[derive(Clone, Debug, PartialEq)]
pub struct UserPreset {
    pub name: String,
    pub author: String,
    pub tags: String,
    /// The Browser's TYPE for it.
    pub kind: String,
    /// Parameter index and value, for those it sets.
    pub params: Vec<(usize, f32)>,
    /// The tables its oscillators play, `None` for one it leaves at Init's.
    pub tables: [Option<u16>; 2],
    /// Its routes, in the order the original sums them: at most
    /// [`crate::params::ROUTES`].
    pub routes: Vec<RouteSlot>,
    /// The routes it holds that do not play, each said in words: past
    /// [`crate::params::ROUTES`], or past
    /// [`crate::modmatrix::MAX_PER_DESTINATION`] on one destination.
    pub left_out: Vec<String>,
    /// What it names that bloop does not have, in words: a parameter, a
    /// source, a destination, a table.
    pub ignored: Vec<String>,
}

fn user_presets() -> &'static [std::sync::OnceLock<UserPreset>; USER_PRESETS] {
    static PRESETS_OF_THE_USER: [std::sync::OnceLock<UserPreset>; USER_PRESETS] =
        [const { std::sync::OnceLock::new() }; USER_PRESETS];
    &PRESETS_OF_THE_USER
}

/// Preset `index`, when it is one of the user's: those come after Init.
/// Lock-free, so the audio thread loads one as it loads a factory one.
#[must_use]
pub fn user(index: usize) -> Option<&'static UserPreset> {
    user_presets().get(index.checked_sub(count())?)?.get()
}

/// Keep `preset` for the rest of the run, and say its number: the one
/// it already had, if one the same was kept before; `None` when there is
/// no room for another. UI thread.
#[must_use]
pub fn keep(preset: UserPreset) -> Option<usize> {
    let mut preset = preset;
    for (at, slot) in user_presets().iter().enumerate() {
        if slot.get().is_none() {
            match slot.set(preset) {
                Ok(()) => return Some(count() + at),
                Err(ours) => preset = ours,
            }
        }
        if slot.get() == Some(&preset) {
            return Some(count() + at);
        }
    }
    None
}

/// Every preset of the user's kept this run, by number.
pub fn users() -> impl Iterator<Item = (usize, &'static UserPreset)> {
    user_presets()
        .iter()
        .map_while(std::sync::OnceLock::get)
        .enumerate()
        .map(|(at, p)| (count() + at, p))
}

/// The preset called `name`: the last of the user's that is, or else
/// the factory one.
#[must_use]
pub fn index_named(name: &str) -> Option<usize> {
    let users: Vec<usize> = users()
        .filter(|(_, p)| p.name == name)
        .map(|(i, _)| i)
        .collect();
    users.last().copied().or_else(|| {
        PRESETS
            .iter()
            .position(|p| p.name == name)
            .or_else(|| (name == "Init").then_some(PRESETS.len()))
    })
}
