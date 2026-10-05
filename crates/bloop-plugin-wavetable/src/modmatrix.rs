//! `gin::ModMatrix` as Wavetable sets it up (`setupModMatrix`): its
//! sources, the parameters it can reach, and how a parameter's value is
//! worked out from its knob and whatever modulates it
//! (`gin_plugin/plugin/gin_modmatrix.{h,cpp}`).
//!
//! A route is a slot: a source, a destination and a depth. The original
//! keeps one list of sources per destination, in the order they were
//! added — the order of a preset's `MODITEM`s — and sums them in that
//! order; so does this, from the slots in slot order. Every route here is
//! enabled, linear, and maps as its source does (`sameAsSource`), which is
//! all a factory preset asks for: none sets a curve, a mapping or
//! `enabled="0"`.

use crate::params::{Kind, PARAM_COUNT, SPECS};
use bloop_plugin_socalabs::shared::{jlimit, ValueSmoother};

/// One modulation source: its id as a preset names it, its name, and
/// whether it is per voice and whether it swings both ways.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    pub id: &'static str,
    pub name: &'static str,
    pub poly: bool,
    pub bipolar: bool,
}

const fn poly(id: &'static str, name: &'static str, bipolar: bool) -> Source {
    Source {
        id,
        name,
        poly: true,
        bipolar,
    }
}

const fn mono(id: &'static str, name: &'static str, bipolar: bool) -> Source {
    Source {
        id,
        name,
        poly: false,
        bipolar,
    }
}

macro_rules! ccs {
    ($($n:literal)*) => {
        [$(mono(concat!("cc", $n), concat!("CC ", $n), false),)*]
    };
}

const NAMED: [Source; 18] = [
    poly("mpep", "MPE Pressure", false),
    poly("mpet", "MPE Timbre", false),
    poly("mpebp", "MPE Pitch Bend", true),
    mono("pb", "Pitch Bend", true),
    poly("note", "MIDI Note Number", false),
    poly("vel", "MIDI Velocity", false),
    mono("mlfo1", "LFO 1 (Mono)", true),
    mono("mlfo2", "LFO 2 (Mono)", true),
    mono("mlfo3", "LFO 3 (Mono)", true),
    poly("lfo1", "LFO 1", true),
    poly("lfo2", "LFO 2", true),
    poly("lfo3", "LFO 3", true),
    mono("mstep", "Step LFO (Mono)", true),
    poly("step", "Step LFO", true),
    poly("fenv", "Filter Envelope", false),
    poly("env1", "Envelope 1", false),
    poly("env2", "Envelope 2", false),
    poly("env3", "Envelope 3", false),
];

const CCS: [Source; 120] = ccs!(
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
    60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89
    90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
);

/// Every source, in the order Wavetable adds them; a source's index here
/// is its id in the matrix.
pub const SOURCES: [Source; NAMED.len() + CCS.len()] = {
    let mut all = [NAMED[0]; NAMED.len() + CCS.len()];
    let mut i = 0;
    while i < NAMED.len() {
        all[i] = NAMED[i];
        i += 1;
    }
    let mut j = 0;
    while j < CCS.len() {
        all[NAMED.len() + j] = CCS[j];
        j += 1;
    }
    all
};

/// Source ids, as the DSP sets them.
pub mod src {
    pub const PRESSURE: usize = 0;
    pub const TIMBRE: usize = 1;
    pub const MPE_BEND: usize = 2;
    pub const PITCH_BEND: usize = 3;
    pub const NOTE: usize = 4;
    pub const VELOCITY: usize = 5;
    pub const MONO_LFO: [usize; 3] = [6, 7, 8];
    pub const LFO: [usize; 3] = [9, 10, 11];
    pub const MONO_STEP: usize = 12;
    pub const STEP: usize = 13;
    pub const FILTER_ENV: usize = 14;
    pub const ENV: [usize; 3] = [15, 16, 17];
    pub const CC0: usize = 18;
}

pub const SOURCE_COUNT: usize = SOURCES.len();

/// Whether parameter `index` is one the matrix can reach.
const fn reachable(index: usize) -> bool {
    matches!(SPECS[index].kind, Kind::Ext | Kind::IntModulated)
}

const fn count_destinations() -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < PARAM_COUNT {
        if reachable(i) {
            n += 1;
        }
        i += 1;
    }
    n
}

/// How many parameters the matrix reaches.
pub const DESTINATION_COUNT: usize = count_destinations();

/// The parameters the matrix reaches, in its order: a destination's index
/// here is what a slot's Destination knob holds, less one.
pub const DESTINATIONS: [usize; DESTINATION_COUNT] = {
    let mut all = [0; DESTINATION_COUNT];
    let mut n = 0;
    let mut i = 0;
    while i < PARAM_COUNT {
        if reachable(i) {
            all[n] = i;
            n += 1;
        }
        i += 1;
    }
    all
};

/// Each parameter's place among the destinations, if it has one.
pub const DESTINATION_OF: [i16; PARAM_COUNT] = {
    let mut of = [-1i16; PARAM_COUNT];
    let mut n = 0;
    while n < DESTINATION_COUNT {
        of[DESTINATIONS[n]] = n as i16;
        n += 1;
    }
    of
};

/// The source with preset id `id`.
#[must_use]
pub fn source_named(id: &str) -> Option<usize> {
    SOURCES.iter().position(|s| s.id == id)
}

/// How many routes reach one destination at most. A factory preset's
/// most is 4; a fifth slot on the same destination is ignored.
pub const MAX_PER_DESTINATION: usize = 8;

/// A route the matrix runs: which source, how deep.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Route {
    pub source: u8,
    pub depth: f32,
}

/// `ModMatrix::shape` for a linear route mapped as its source maps: a
/// bipolar source is moved to 0..1 and back, through the original's
/// arithmetic, which is not quite the identity.
#[inline]
fn shape(value: f32, bipolar: bool) -> f32 {
    if bipolar {
        let unipolar = (1.0 * (value - -1.0)) / (1.0 - -1.0);
        -1.0 + (2.0 * (unipolar - 0.0)) / (1.0 - 0.0)
    } else {
        value
    }
}

/// The routes, by destination, and the mono sources' values.
#[derive(Clone, Debug)]
pub struct Matrix {
    routes: [[Route; MAX_PER_DESTINATION]; DESTINATION_COUNT],
    counts: [u8; DESTINATION_COUNT],
    mono: [f32; SOURCE_COUNT],
    smoothers: [ValueSmoother; DESTINATION_COUNT],
}

impl Default for Matrix {
    fn default() -> Self {
        Self {
            routes: [[Route::default(); MAX_PER_DESTINATION]; DESTINATION_COUNT],
            counts: [0; DESTINATION_COUNT],
            mono: [0.0; SOURCE_COUNT],
            smoothers: core::array::from_fn(|_| ValueSmoother::default()),
        }
    }
}

/// A voice's side of the matrix: its poly sources' values and its own
/// smoothers.
#[derive(Clone, Debug)]
pub struct VoiceMod {
    pub values: [f32; SOURCE_COUNT],
    smoothers: [ValueSmoother; DESTINATION_COUNT],
    /// `ModVoice::disableSmoothing`: values taken as they stand, while a
    /// note starts.
    pub disable_smoothing: bool,
    pub age: i32,
}

impl Default for VoiceMod {
    fn default() -> Self {
        Self {
            values: [0.0; SOURCE_COUNT],
            smoothers: core::array::from_fn(|_| ValueSmoother::default()),
            disable_smoothing: false,
            age: 0,
        }
    }
}

impl VoiceMod {
    /// `ModVoice::finishBlock`: every smoother `samples` on.
    pub fn finish_block(&mut self, samples: usize) {
        for s in &mut self.smoothers {
            s.process(samples);
        }
    }

    /// `snapParams`: every smoother at its target.
    pub fn snap(&mut self) {
        for s in &mut self.smoothers {
            s.snap_to_value();
        }
    }
}

impl Matrix {
    /// `setSampleRate`, for the matrix and every voice: 20 ms smoothing.
    pub fn set_sample_rate<'a>(
        &mut self,
        sample_rate: f64,
        voices: impl Iterator<Item = &'a mut VoiceMod>,
    ) {
        for s in &mut self.smoothers {
            s.set_sample_rate(sample_rate);
            s.set_time(0.02);
        }
        for v in voices {
            for s in &mut v.smoothers {
                s.set_sample_rate(sample_rate);
                s.set_time(0.02);
            }
        }
    }

    /// The routes, from the slots `(source, destination, depth)`, each
    /// numbered from 1 with 0 for none.
    pub fn set_routes(&mut self, slots: impl Iterator<Item = (usize, usize, f32)>) {
        self.counts = [0; DESTINATION_COUNT];
        for (source, destination, depth) in slots {
            let (Some(source), Some(destination)) = (source.checked_sub(1), destination.checked_sub(1))
            else {
                continue;
            };
            if source >= SOURCE_COUNT || destination >= DESTINATION_COUNT {
                continue;
            }
            let count = usize::from(self.counts[destination]);
            if count < MAX_PER_DESTINATION {
                self.routes[destination][count] = Route {
                    source: source as u8,
                    depth,
                };
                self.counts[destination] += 1;
            }
        }
    }

    /// The routes reaching `destination`, in order.
    #[must_use]
    pub fn routes(&self, destination: usize) -> &[Route] {
        &self.routes[destination][..usize::from(self.counts[destination])]
    }

    pub fn set_mono(&mut self, source: usize, value: f32) {
        self.mono[source] = value;
    }

    #[must_use]
    pub fn mono(&self, source: usize) -> f32 {
        self.mono[source]
    }

    /// `ModMatrix::finishBlock`: the matrix's own smoothers `samples` on.
    pub fn finish_block(&mut self, samples: usize) {
        for s in &mut self.smoothers {
            s.process(samples);
        }
    }

    /// The modulated, smoothed position (0 to 1) of destination `dst`
    /// whose knob sits at `base`, with poly sources read from `voice`.
    fn sum(&self, dst: usize, base: f32, voice: Option<&[f32; SOURCE_COUNT]>) -> f32 {
        let mut base = base;
        for route in self.routes(dst) {
            let source = usize::from(route.source);
            let info = SOURCES[source];
            let value = if info.poly {
                match voice {
                    Some(values) => values[source],
                    None => continue,
                }
            } else {
                self.mono[source]
            };
            base += shape(value, info.bipolar) * route.depth;
        }
        jlimit(0.0, 1.0, base)
    }

    /// `ModMatrix::getValue (p)`: a parameter as the whole synth sees it,
    /// poly sources read from `active` — the voice started last, if one
    /// is playing. `base` is the knob's position, 0 to 1.
    pub fn value_global(
        &mut self,
        dst: usize,
        base: f32,
        active: Option<&[f32; SOURCE_COUNT]>,
        smoothed: bool,
    ) -> f32 {
        let v = self.sum(dst, base, active);
        let smoother = &mut self.smoothers[dst];
        smoother.set_value(v);
        if smoothed {
            smoother.current()
        } else {
            v
        }
    }

    /// `ModMatrix::getValue (voice, p)`: a parameter as one voice sees it.
    pub fn value_voice(&self, voice: &mut VoiceMod, dst: usize, base: f32) -> f32 {
        let v = self.sum(dst, base, Some(&voice.values));
        let smoother = &mut voice.smoothers[dst];
        smoother.set_value(v);
        if voice.disable_smoothing {
            v
        } else {
            smoother.current()
        }
    }
}
