//! Every parameter of SocaLabs' Wavetable, in the order its constructor
//! registers them (`PluginProcessor.cpp`, `*Params::setup`), then the three
//! kinds bloop adds: which wavetable each oscillator plays, and the
//! modulation matrix as slots.
//!
//! A parameter's index here is its bloop parameter index. The original's
//! are its `uid`s, which is what a preset names, and the order matters to
//! the modulation matrix only for which parameters it lists — every
//! registered parameter that is not internal (`addExtParam`), plus the
//! delay's time. What is left out: the editor's own state (`uiLFO`,
//! `uiENV`, `uiMOD`, and each LFO's `xgrid` and `ygrid`, the grid its
//! editor draws).

use crate::range::Range;
use bloop_plugin_socalabs::shared::{decibels_to_gain, DURATIONS};

/// What a parameter's value becomes before the DSP sees it: the
/// original's `conversionFunction`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    None,
    /// `juce::Decibels::decibelsToGain (in)`.
    DecibelsToGain,
    /// `in / 100.0f`.
    Percent,
    /// `in / 1000.0f`.
    Millis,
}

impl Conversion {
    #[must_use]
    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::None => value,
            Self::DecibelsToGain => decibels_to_gain(value),
            Self::Percent => value / 100.0,
            Self::Millis => value / 1000.0,
        }
    }
}

/// How the parameter is registered: `addExtParam`, which the modulation
/// matrix can reach, or `addIntParam`, which it cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Ext,
    Int,
    /// `dlDelay`: registered internal, and added to the matrix anyway.
    IntModulated,
    /// One of bloop's own: a wavetable or a modulation slot.
    Host,
}

/// What a value means, for a knob to read it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Named positions.
    Choice(&'static [&'static str]),
    /// Off or On.
    Switch,
    Semitones,
    Cents,
    Decibels,
    Percent,
    Seconds,
    Millis,
    Hertz,
    /// A filter cutoff held as a MIDI note, read as the hertz it is.
    NoteHertz,
    /// One of Gin's note durations, by index.
    Beat,
    /// A plain number.
    Number,
    /// Whole numbers: voices, steps.
    Count,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    /// The original's parameter id, which a preset names.
    pub uid: &'static str,
    /// The original's full name, which bloop's knobs are labelled with.
    pub name: &'static str,
    pub range: Range,
    pub default: f32,
    pub kind: Kind,
    pub conversion: Conversion,
    pub unit: Unit,
}

const fn ext(
    uid: &'static str,
    name: &'static str,
    range: Range,
    default: f32,
    unit: Unit,
) -> Spec {
    Spec {
        uid,
        name,
        range,
        default,
        kind: Kind::Ext,
        conversion: Conversion::None,
        unit,
    }
}

const fn int(
    uid: &'static str,
    name: &'static str,
    range: Range,
    default: f32,
    unit: Unit,
) -> Spec {
    Spec {
        uid,
        name,
        range,
        default,
        kind: Kind::Int,
        conversion: Conversion::None,
        unit,
    }
}

const fn conv(spec: Spec, conversion: Conversion) -> Spec {
    Spec { conversion, ..spec }
}

const fn switch(uid: &'static str, name: &'static str, default: f32) -> Spec {
    int(uid, name, Range::new(0.0, 1.0, 1.0, 1.0), default, Unit::Switch)
}

const SWITCH: Range = Range::new(0.0, 1.0, 1.0, 1.0);

/// Gin's note durations, by the names its Beat knobs show.
pub const BEATS: [&str; 33] = {
    let mut names = [""; 33];
    let mut i = 0;
    while i < 33 {
        names[i] = DURATIONS[i].name;
        i += 1;
    }
    names
};

pub const SUB_WAVES: [&str; 6] = [
    "Sine",
    "Triangle",
    "Saw",
    "Pulse 50%",
    "Pulse 25%",
    "Pulse 12%",
];
pub const NOISE_TYPES: [&str; 2] = ["White", "Pink"];
pub const FILTER_TYPES: [&str; 8] = [
    "LP 12", "LP 24", "HP 12", "HP 24", "BP 12", "BP 24", "NT 12", "NT 24",
];
/// Indexed by `gin::LFO::WaveShape`; the knob runs from Sine.
pub const LFO_WAVES: [&str; 18] = [
    "None",
    "Sine",
    "Triangle",
    "Saw Up",
    "Saw Down",
    "Square",
    "Square+",
    "S&H",
    "Noise",
    "Step Up 3",
    "Step Up 4",
    "Step Up 8",
    "Step Down 3",
    "Step Down 4",
    "Step Down 8",
    "Pyramid 3",
    "Pyramid 5",
    "Pyramid 9",
];
pub const GLIDE_MODES: [&str; 3] = ["Off", "Glissando", "Portamento"];
pub const FX_NAMES: [&str; 5] = ["Gate", "Chorus", "Distortion", "Delay", "Reverb"];
pub const DIST_MODES: [&str; 4] = ["Simple", "Bitcrusher", "Fire Amp", "Grind Amp"];

const LEVEL: Range = Range::new(-100.0, 0.0, 1.0, 4.0);
const TUNE: Range = Range::new(-36.0, 36.0, 1.0, 1.0);
const BIPOLAR: Range = Range::new(-1.0, 1.0, 0.0, 1.0);
const PERCENT: Range = Range::new(0.0, 100.0, 0.0, 1.0);
const TIME: Range = Range::new(0.0, 60.0, 0.0, 0.2);
const BEAT: Range = Range::new(0.0, 32.0, 1.0, 1.0);
/// `getMidiNoteFromHertz (20000.0)`, the filter's top.
pub const FILTER_MAX_NOTE: f32 = 135.076_23;

macro_rules! osc {
    ($n:literal) => {
        [
            int(
                concat!("osc", $n, "enable"),
                concat!("OSC", $n, " Enable"),
                SWITCH,
                if $n == 1 { 1.0 } else { 0.0 },
                Unit::Switch,
            ),
            switch(concat!("osc", $n, "retrig"), concat!("OSC", $n, " Retrig"), 0.0),
            int(
                concat!("osc", $n, "unison"),
                concat!("OSC", $n, " Unison"),
                Range::new(1.0, 8.0, 1.0, 1.0),
                1.0,
                Unit::Count,
            ),
            ext(concat!("osc", $n, "tune"), concat!("OSC", $n, " Tune"), TUNE, 0.0, Unit::Semitones),
            ext(
                concat!("osc", $n, "finetune"),
                concat!("OSC", $n, " Fine Tune"),
                Range::new(-100.0, 100.0, 0.0, 1.0),
                0.0,
                Unit::Cents,
            ),
            conv(
                ext(concat!("osc", $n, "level"), concat!("OSC", $n, " Level"), LEVEL, 0.0, Unit::Decibels),
                Conversion::DecibelsToGain,
            ),
            ext(concat!("osc", $n, "pos"), concat!("OSC", $n, " Pos"), PERCENT, 0.0, Unit::Percent),
            ext(
                concat!("osc", $n, "detune"),
                concat!("OSC", $n, " Detune"),
                Range::new(0.0, 0.5, 0.0, 1.0),
                0.0,
                Unit::Number,
            ),
            ext(
                concat!("osc", $n, "spread"),
                concat!("OSC", $n, " Spread"),
                Range::new(-100.0, 100.0, 0.0, 1.0),
                0.0,
                Unit::Percent,
            ),
            ext(concat!("osc", $n, "pan"), concat!("OSC", $n, " Pan"), BIPOLAR, 0.0, Unit::Number),
            ext(concat!("osc", $n, "formant"), concat!("OSC", $n, " Formant"), BIPOLAR, 0.0, Unit::Number),
            ext(concat!("osc", $n, "bend"), concat!("OSC", $n, " Bend"), BIPOLAR, 0.0, Unit::Number),
        ]
    };
}

macro_rules! env {
    ($n:literal) => {
        [
            switch(concat!("env", $n, "enable"), concat!("ENV", $n, " Enable"), 0.0),
            switch(concat!("env", $n, "retrig"), concat!("ENV", $n, " Retrig"), 1.0),
            ext(concat!("env", $n, "attack"), concat!("ENV", $n, " Attack"), TIME, 0.1, Unit::Seconds),
            ext(concat!("env", $n, "decay"), concat!("ENV", $n, " Decay"), TIME, 0.1, Unit::Seconds),
            conv(
                ext(concat!("env", $n, "sustain"), concat!("ENV", $n, " Sustain"), PERCENT, 80.0, Unit::Percent),
                Conversion::Percent,
            ),
            ext(concat!("env", $n, "release"), concat!("ENV", $n, " Release"), TIME, 0.1, Unit::Seconds),
        ]
    };
}

macro_rules! lfo {
    ($n:literal) => {
        [
            switch(concat!("lfo", $n, "enable"), concat!("LFO", $n, " Enable"), 0.0),
            switch(concat!("lfo", $n, "sync"), concat!("LFO", $n, " Sync"), 0.0),
            switch(concat!("lfo", $n, "retrig"), concat!("LFO", $n, " Retrig"), 1.0),
            int(
                concat!("lfo", $n, "wave"),
                concat!("LFO", $n, " Wave"),
                Range::new(1.0, 17.0, 1.0, 1.0),
                1.0,
                Unit::Choice(&LFO_WAVES),
            ),
            ext(
                concat!("lfo", $n, "rate"),
                concat!("LFO", $n, " Rate"),
                Range::new(0.0, 50.0, 0.0, 0.3),
                10.0,
                Unit::Hertz,
            ),
            int(concat!("lfo", $n, "beat"), concat!("LFO", $n, " Beat"), BEAT, 13.0, Unit::Beat),
            ext(concat!("lfo", $n, "depth"), concat!("LFO", $n, " Depth"), BIPOLAR, 1.0, Unit::Number),
            ext(concat!("lfo", $n, "phase"), concat!("LFO", $n, " Phase"), BIPOLAR, 0.0, Unit::Number),
            ext(concat!("lfo", $n, "offset"), concat!("LFO", $n, " Offset"), BIPOLAR, 0.0, Unit::Number),
            ext(
                concat!("lfo", $n, "fade"),
                concat!("LFO", $n, " Fade"),
                Range::symmetric(-60.0, 60.0, 0.0, 0.2),
                0.0,
                Unit::Seconds,
            ),
            ext(concat!("lfo", $n, "delay"), concat!("LFO", $n, " Delay"), TIME, 0.0, Unit::Seconds),
        ]
    };
}

const fn step(uid: &'static str, name: &'static str) -> Spec {
    int(uid, name, BIPOLAR, 0.0, Unit::Number)
}

const fn gate_step(uid: &'static str, name: &'static str, n: usize) -> Spec {
    // `(i % 2 == 0 || i % 5 == 0)`, for step `n` counted from 1.
    let i = n - 1;
    switch(uid, name, if i % 2 == 0 || i % 5 == 0 { 1.0 } else { 0.0 })
}

macro_rules! steps {
    ($($n:literal)*) => {
        [$(step(concat!("slfostep", $n), concat!("Step LFOStep ", $n)),)*]
    };
}

macro_rules! gate_steps {
    ($($n:literal)*) => {
        [$(
            gate_step(concat!("gatel", $n), concat!("GateL ", $n), $n),
            gate_step(concat!("gater", $n), concat!("GateR ", $n), $n),
        )*]
    };
}

const OSC1: [Spec; 12] = osc!(1);
const OSC2: [Spec; 12] = osc!(2);

const SUB: [Spec; 6] = [
    switch("subenable", "SUB Enable", 0.0),
    switch("subretrig", "SUB Retrig", 0.0),
    int(
        "subwave",
        "SUB Wave",
        Range::new(0.0, 5.0, 1.0, 1.0),
        1.0,
        Unit::Choice(&SUB_WAVES),
    ),
    ext("subtune", "SUB Tune", TUNE, 0.0, Unit::Semitones),
    conv(
        ext("sublevel", "SUB Level", LEVEL, 0.0, Unit::Decibels),
        Conversion::DecibelsToGain,
    ),
    ext("subpan", "SUB Pan", BIPOLAR, 0.0, Unit::Number),
];

const NOISE: [Spec; 4] = [
    switch("noiseenable", "Noise Enable", 0.0),
    int(
        "noisetype",
        "Noise Type",
        SWITCH,
        0.0,
        Unit::Choice(&NOISE_TYPES),
    ),
    conv(
        ext("noiselevel", "Noise Level", LEVEL, 0.0, Unit::Decibels),
        Conversion::DecibelsToGain,
    ),
    ext("noisepan", "Noise Pan", BIPOLAR, 0.0, Unit::Number),
];

const FILTER: [Spec; 16] = [
    switch("fltenable", "FLT Enable", 1.0),
    switch("fltretrig", "FLT Retrig", 1.0),
    int(
        "flttype",
        "FLT Type",
        Range::new(0.0, 7.0, 1.0, 1.0),
        0.0,
        Unit::Choice(&FILTER_TYPES),
    ),
    conv(ext("fltkey", "FLT Key", PERCENT, 0.0, Unit::Percent), Conversion::Percent),
    conv(ext("fltvel", "FLT Vel", PERCENT, 0.0, Unit::Percent), Conversion::Percent),
    ext(
        "fltfreq",
        "FLT Freq",
        Range::new(0.0, FILTER_MAX_NOTE, 0.0, 1.0),
        64.0,
        Unit::NoteHertz,
    ),
    ext("fltres", "FLT Res", PERCENT, 0.0, Unit::Percent),
    ext("fltamount", "FLT Amount", BIPOLAR, 0.0, Unit::Number),
    ext("fltattack", "FLT Attack", TIME, 0.1, Unit::Seconds),
    ext("fltdecay", "FLT Decay", TIME, 0.1, Unit::Seconds),
    conv(
        ext("fltsustain", "FLT Sustain", PERCENT, 80.0, Unit::Percent),
        Conversion::Percent,
    ),
    ext("fltrelease", "FLT Release", TIME, 0.1, Unit::Seconds),
    switch("fltwt1", "FLT WT1", 1.0),
    switch("fltwt2", "FLT WT2", 1.0),
    switch("fltsub", "FLT sub", 1.0),
    switch("fltnoise", "FLT noise", 1.0),
];

const ENV1: [Spec; 6] = env!(1);
const ENV2: [Spec; 6] = env!(2);
const ENV3: [Spec; 6] = env!(3);
const LFO1: [Spec; 11] = lfo!(1);
const LFO2: [Spec; 11] = lfo!(2);
const LFO3: [Spec; 11] = lfo!(3);

const STEP_LFO: [Spec; 4] = [
    switch("slfoenable", "Step LFOEnable", 0.0),
    int("slfobeat", "Step LFOBeat", BEAT, 13.0, Unit::Beat),
    int(
        "slfolength",
        "Step LFOLength",
        Range::new(2.0, 32.0, 1.0, 1.0),
        8.0,
        Unit::Count,
    ),
    switch("slforetrig", "Step LFORetrig", 1.0),
];

const STEPS: [Spec; 32] = steps!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32);

const AMP: [Spec; 6] = [
    switch("retrig", "Retrig", 1.0),
    conv(ext("vel", "Vel", PERCENT, 100.0, Unit::Percent), Conversion::Percent),
    ext("attack", "Attack", TIME, 0.1, Unit::Seconds),
    ext("decay", "Decay", TIME, 0.1, Unit::Seconds),
    conv(ext("sustain", "Sustain", PERCENT, 80.0, Unit::Percent), Conversion::Percent),
    ext("release", "Release", TIME, 0.1, Unit::Seconds),
];

const GLOBAL: [Spec; 8] = [
    switch("mono", "Mono", 0.0),
    int(
        "gMode",
        "Glide Mode",
        Range::new(0.0, 2.0, 0.0, 1.0),
        0.0,
        Unit::Choice(&GLIDE_MODES),
    ),
    ext(
        "gRate",
        "Glide Time",
        Range::new(0.001, 20.0, 0.0, 0.2),
        0.3,
        Unit::Seconds,
    ),
    switch("legato", "Legato", 0.0),
    conv(ext("level", "Level", LEVEL, 0.0, Unit::Decibels), Conversion::DecibelsToGain),
    int(
        "voices",
        "Voices",
        Range::new(2.0, 40.0, 1.0, 1.0),
        40.0,
        Unit::Count,
    ),
    switch("mpe", "MPE", 0.0),
    int(
        "pitchbend",
        "Pitch Bend",
        Range::new(0.0, 48.0, 1.0, 1.0),
        2.0,
        Unit::Semitones,
    ),
];

const GATE: [Spec; 5] = [
    switch("gateenable", "GateEnable", 0.0),
    int("gatebeat", "GateBeat", BEAT, 7.0, Unit::Beat),
    int(
        "gatelength",
        "GateLength",
        Range::new(2.0, 16.0, 1.0, 1.0),
        8.0,
        Unit::Count,
    ),
    ext("gateattack", "GateAttack", Range::new(0.0, 1.0, 0.0, 0.2), 0.1, Unit::Seconds),
    ext("gaterelease", "GateRelease", Range::new(0.0, 1.0, 0.0, 0.2), 0.1, Unit::Seconds),
];

const GATE_STEPS: [Spec; 32] = gate_steps!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);

const CHORUS: [Spec; 6] = [
    switch("chEnable", "Chorus Enable", 0.0),
    conv(
        ext("chDelay", "Chorus Delay", Range::new(0.1, 30.0, 0.0, 1.0), 1.0, Unit::Millis),
        Conversion::Millis,
    ),
    conv(
        ext("chDepth", "Chorus Depth", Range::new(0.1, 20.0, 0.0, 1.0), 1.0, Unit::Millis),
        Conversion::Millis,
    ),
    ext("chSpeed", "Chorus Speed", Range::new(0.1, 10.0, 0.0, 1.0), 3.0, Unit::Hertz),
    ext("chWidth", "Chorus Width", Range::new(0.0, 1.0, 0.0, 1.0), 0.5, Unit::Number),
    ext("chMix", "Chorus Mix", Range::new(0.0, 1.0, 0.0, 1.0), 0.5, Unit::Number),
];

const DISTORTION: [Spec; 2] = [
    switch("dsEnable", "Distortion Enable", 0.0),
    ext("dsAmount", "Distortion Amount", Range::new(0.0, 1.0, 0.0, 1.0), 0.2, Unit::Number),
];

const DELAY: [Spec; 8] = [
    switch("dlEnable", "Delay Enable", 0.0),
    ext("dlSync", "Delay Sync", Range::new(0.0, 1.0, 1.0, 1.0), 0.0, Unit::Switch),
    ext("dlTime", "Delay Time", Range::new(0.0, 120.0, 0.0, 0.3), 1.0, Unit::Seconds),
    ext("dlBeat", "Delay Beat", BEAT, 13.0, Unit::Beat),
    conv(
        ext("dlFb", "Delay FB", Range::new(-100.0, 0.0, 0.0, 5.0), -10.0, Unit::Decibels),
        Conversion::DecibelsToGain,
    ),
    conv(
        ext("dlCf", "Delay CF", Range::new(-100.0, 0.0, 0.0, 5.0), -100.0, Unit::Decibels),
        Conversion::DecibelsToGain,
    ),
    conv(
        ext("dlMix", "Delay Mix", Range::new(0.0, 100.0, 0.0, 1.0), 0.5, Unit::Percent),
        Conversion::Percent,
    ),
    Spec {
        kind: Kind::IntModulated,
        ..int("dlDelay", "Delay Delay", Range::new(0.0, 120.0, 0.0, 1.0), 1.0, Unit::Seconds)
    },
];

const REVERB: [Spec; 7] = [
    switch("rvEnable", "Reverb Enable", 0.0),
    ext("rvbSize", "Reverb Size", Range::new(0.0, 1.0, 0.0, 2.0), 0.0, Unit::Number),
    ext("rvbDecay", "Reverb Decay", Range::new(0.0, 1.0, 0.0, 1.0), 0.0, Unit::Number),
    ext(
        "rvbLowpass",
        "Reverb Lowpass",
        Range::new(16.0, 20000.0, 0.0, 0.3),
        10000.0,
        Unit::Hertz,
    ),
    ext(
        "rvbDamping",
        "Reverb Damping",
        Range::new(16.0, 20000.0, 0.0, 0.3),
        10000.0,
        Unit::Hertz,
    ),
    ext("rvbPredelay", "Reverb Predelay", Range::new(0.0, 0.1, 0.0, 1.0), 0.0, Unit::Seconds),
    ext("rvbMix", "Reverb Mix", Range::new(0.0, 1.0, 0.0, 1.0), 0.0, Unit::Number),
];

const FX: [Spec; 6] = [
    int("fxOrder1", "FX1", Range::new(0.0, 4.0, 1.0, 1.0), 0.0, Unit::Choice(&FX_NAMES)),
    int("fxOrder2", "FX2", Range::new(0.0, 4.0, 1.0, 1.0), 1.0, Unit::Choice(&FX_NAMES)),
    int("fxOrder3", "FX3", Range::new(0.0, 4.0, 1.0, 1.0), 2.0, Unit::Choice(&FX_NAMES)),
    int("fxOrder4", "FX4", Range::new(0.0, 4.0, 1.0, 1.0), 3.0, Unit::Choice(&FX_NAMES)),
    int("fxOrder5", "FX5", Range::new(0.0, 4.0, 1.0, 1.0), 4.0, Unit::Choice(&FX_NAMES)),
    int(
        "distMode",
        "Dist Mode",
        Range::new(0.0, 3.0, 1.0, 1.0),
        0.0,
        Unit::Choice(&DIST_MODES),
    ),
];

const UNIT: Range = Range::new(0.0, 1.0, 0.0, 1.0);

const CRUSH: [Spec; 4] = [
    ext("crushrate", "Crush Rate", UNIT, 0.5, Unit::Number),
    ext("crushrez", "Crush Rez", UNIT, 0.5, Unit::Number),
    ext("crushhard", "Crush Hard", UNIT, 0.8, Unit::Number),
    ext("crushmix", "Crush Mix", UNIT, 1.0, Unit::Number),
];

const FIRE: [Spec; 4] = [
    ext("firegain", "Fire Gain", UNIT, 0.5, Unit::Number),
    ext("firetone", "Fire Tone", UNIT, 0.5, Unit::Number),
    ext("fireoutput", "Fire Output", UNIT, 0.8, Unit::Number),
    ext("firemix", "Fire Mix", UNIT, 1.0, Unit::Number),
];

const GRIND: [Spec; 4] = [
    ext("grindgain", "Grind Gain", UNIT, 0.5, Unit::Number),
    ext("grindtone", "Grind Tone", UNIT, 0.5, Unit::Number),
    ext("grindoutput", "Grind Output", UNIT, 0.8, Unit::Number),
    ext("grindmix", "Grind Mix", UNIT, 1.0, Unit::Number),
];

/// How many modulation slots bloop gives the matrix: room for the most a
/// factory preset uses (21), with a few to spare.
pub const MOD_SLOTS: usize = 24;

/// bloop's own parameters, after the original's.
const fn host(uid: &'static str, name: &'static str, end: f32, unit: Unit) -> Spec {
    Spec {
        uid,
        name,
        range: Range::new(0.0, end, 1.0, 1.0),
        default: 0.0,
        kind: Kind::Host,
        conversion: Conversion::None,
        unit,
    }
}

macro_rules! concat_all {
    ($($part:expr),* $(,)?) => {{
        const LEN: usize = 0 $(+ $part.len())*;
        let mut out = [EMPTY; LEN];
        let mut at = 0;
        $(
            let part = $part;
            let mut i = 0;
            while i < part.len() {
                out[at] = part[i];
                at += 1;
                i += 1;
            }
        )*
        let _ = at;
        out
    }};
}

const EMPTY: Spec = host("", "", 0.0, Unit::Number);

const ORIGINAL_LEN: usize = OSC1.len()
    + OSC2.len()
    + SUB.len()
    + NOISE.len()
    + FILTER.len()
    + ENV1.len() * 3
    + LFO1.len() * 3
    + STEP_LFO.len()
    + STEPS.len()
    + AMP.len()
    + GLOBAL.len()
    + GATE.len()
    + GATE_STEPS.len()
    + CHORUS.len()
    + DISTORTION.len()
    + DELAY.len()
    + REVERB.len()
    + FX.len()
    + CRUSH.len()
    + FIRE.len()
    + GRIND.len();

/// Every parameter the original registers, in its order.
const ORIGINAL: [Spec; ORIGINAL_LEN] = concat_all!(
    OSC1, OSC2, SUB, NOISE, FILTER, ENV1, ENV2, ENV3, LFO1, LFO2, LFO3, STEP_LFO, STEPS, AMP,
    GLOBAL, GATE, GATE_STEPS, CHORUS, DISTORTION, DELAY, REVERB, FX, CRUSH, FIRE, GRIND,
);

/// How many wavetables there are to choose from — the knob's top, in the
/// table the host builds parameters from; `catalogue` says which they are.
pub const TABLE_KNOB_TOP: f32 = 4095.0;

const TABLES: [Spec; 2] = [
    Spec {
        default: crate::presets_data::INIT_TABLES[0] as f32,
        ..host("wt1", "OSC1 Wavetable", TABLE_KNOB_TOP, Unit::Count)
    },
    Spec {
        default: crate::presets_data::INIT_TABLES[1] as f32,
        ..host("wt2", "OSC2 Wavetable", TABLE_KNOB_TOP, Unit::Count)
    },
];

/// The highest source and destination numbers a slot can hold; 0 is
/// none. Which is which is `crate::modmatrix`'s to say.
pub const SOURCE_KNOB_TOP: f32 = 255.0;
pub const DESTINATION_KNOB_TOP: f32 = 255.0;

macro_rules! slots {
    ($($n:literal)*) => {
        [$(
            host(concat!("mod", $n, "src"), concat!("Mod ", $n, " Source"), SOURCE_KNOB_TOP, Unit::Count),
            host(concat!("mod", $n, "dst"), concat!("Mod ", $n, " Destination"), DESTINATION_KNOB_TOP, Unit::Count),
            Spec {
                range: Range::new(-1.0, 1.0, 0.0, 1.0),
                ..host(concat!("mod", $n, "depth"), concat!("Mod ", $n, " Depth"), 0.0, Unit::Number)
            },
        )*]
    };
}

const SLOTS: [Spec; MOD_SLOTS * 3] =
    slots!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24);

/// Every parameter, by bloop index.
pub const PARAMS: [Spec; ORIGINAL.len() + TABLES.len() + SLOTS.len()] =
    concat_all!(ORIGINAL, TABLES, SLOTS);

/// The number of parameters the original registers (bloop's own follow).
pub const ORIGINAL_COUNT: usize = ORIGINAL.len();

/// The index of the parameter with `uid`.
#[must_use]
pub fn index_of(uid: &str) -> Option<usize> {
    PARAMS.iter().position(|p| p.uid == uid)
}

/// Named indices, so the DSP reads `P::OSC1_POS` rather than a number.
pub mod id {
    use super::{index_of_const, PARAMS};

    macro_rules! ids {
        ($($name:ident = $uid:literal;)*) => {
            $(pub const $name: usize = index_of_const(&PARAMS, $uid);)*
        };
    }

    ids! {
        OSC1_ENABLE = "osc1enable";
        OSC2_ENABLE = "osc2enable";
        SUB_ENABLE = "subenable";
        SUB_RETRIG = "subretrig";
        SUB_WAVE = "subwave";
        SUB_TUNE = "subtune";
        SUB_LEVEL = "sublevel";
        SUB_PAN = "subpan";
        NOISE_ENABLE = "noiseenable";
        NOISE_TYPE = "noisetype";
        NOISE_LEVEL = "noiselevel";
        NOISE_PAN = "noisepan";
        FLT_ENABLE = "fltenable";
        FLT_RETRIG = "fltretrig";
        FLT_TYPE = "flttype";
        FLT_KEY = "fltkey";
        FLT_VEL = "fltvel";
        FLT_FREQ = "fltfreq";
        FLT_RES = "fltres";
        FLT_AMOUNT = "fltamount";
        FLT_ATTACK = "fltattack";
        FLT_DECAY = "fltdecay";
        FLT_SUSTAIN = "fltsustain";
        FLT_RELEASE = "fltrelease";
        FLT_WT1 = "fltwt1";
        FLT_WT2 = "fltwt2";
        FLT_SUB = "fltsub";
        FLT_NOISE = "fltnoise";
        ENV1_ENABLE = "env1enable";
        LFO1_ENABLE = "lfo1enable";
        SLFO_ENABLE = "slfoenable";
        SLFO_BEAT = "slfobeat";
        SLFO_LENGTH = "slfolength";
        SLFO_RETRIG = "slforetrig";
        SLFO_STEP1 = "slfostep1";
        AMP_RETRIG = "retrig";
        AMP_VEL = "vel";
        AMP_ATTACK = "attack";
        AMP_DECAY = "decay";
        AMP_SUSTAIN = "sustain";
        AMP_RELEASE = "release";
        MONO = "mono";
        GLIDE_MODE = "gMode";
        GLIDE_RATE = "gRate";
        LEGATO = "legato";
        LEVEL = "level";
        VOICES = "voices";
        MPE = "mpe";
        PITCH_BEND = "pitchbend";
        GATE_ENABLE = "gateenable";
        GATE_BEAT = "gatebeat";
        GATE_LENGTH = "gatelength";
        GATE_ATTACK = "gateattack";
        GATE_RELEASE = "gaterelease";
        GATE_L1 = "gatel1";
        CH_ENABLE = "chEnable";
        CH_DELAY = "chDelay";
        CH_DEPTH = "chDepth";
        CH_SPEED = "chSpeed";
        CH_WIDTH = "chWidth";
        CH_MIX = "chMix";
        DS_ENABLE = "dsEnable";
        DS_AMOUNT = "dsAmount";
        DL_ENABLE = "dlEnable";
        DL_SYNC = "dlSync";
        DL_TIME = "dlTime";
        DL_BEAT = "dlBeat";
        DL_FB = "dlFb";
        DL_CF = "dlCf";
        DL_MIX = "dlMix";
        DL_DELAY = "dlDelay";
        RV_ENABLE = "rvEnable";
        RV_SIZE = "rvbSize";
        RV_DECAY = "rvbDecay";
        RV_LOWPASS = "rvbLowpass";
        RV_DAMPING = "rvbDamping";
        RV_PREDELAY = "rvbPredelay";
        RV_MIX = "rvbMix";
        FX_ORDER1 = "fxOrder1";
        DIST_MODE = "distMode";
        CRUSH_RATE = "crushrate";
        CRUSH_REZ = "crushrez";
        CRUSH_HARD = "crushhard";
        CRUSH_MIX = "crushmix";
        FIRE_GAIN = "firegain";
        FIRE_TONE = "firetone";
        FIRE_OUTPUT = "fireoutput";
        FIRE_MIX = "firemix";
        GRIND_GAIN = "grindgain";
        GRIND_TONE = "grindtone";
        GRIND_OUTPUT = "grindoutput";
        GRIND_MIX = "grindmix";
        WT1 = "wt1";
        WT2 = "wt2";
        MOD1_SRC = "mod1src";
    }

    /// The first parameter of oscillator `n` (0 or 1): its Enable.
    pub const OSC: [usize; 2] = [OSC1_ENABLE, OSC2_ENABLE];
    /// Offsets within an oscillator's block, from its Enable.
    pub mod osc {
        pub const ENABLE: usize = 0;
        pub const RETRIG: usize = 1;
        pub const UNISON: usize = 2;
        pub const TUNE: usize = 3;
        pub const FINETUNE: usize = 4;
        pub const LEVEL: usize = 5;
        pub const POS: usize = 6;
        pub const DETUNE: usize = 7;
        pub const SPREAD: usize = 8;
        pub const PAN: usize = 9;
        pub const FORMANT: usize = 10;
        pub const BEND: usize = 11;
    }
    /// The first parameter of mod envelope `n` (0 to 2): its Enable.
    pub const ENV: [usize; 3] = [ENV1_ENABLE, ENV1_ENABLE + 6, ENV1_ENABLE + 12];
    pub mod env {
        pub const ENABLE: usize = 0;
        pub const RETRIG: usize = 1;
        pub const ATTACK: usize = 2;
        pub const DECAY: usize = 3;
        pub const SUSTAIN: usize = 4;
        pub const RELEASE: usize = 5;
    }
    /// The first parameter of LFO `n` (0 to 2): its Enable.
    pub const LFO: [usize; 3] = [LFO1_ENABLE, LFO1_ENABLE + 11, LFO1_ENABLE + 22];
    pub mod lfo {
        pub const ENABLE: usize = 0;
        pub const SYNC: usize = 1;
        pub const RETRIG: usize = 2;
        pub const WAVE: usize = 3;
        pub const RATE: usize = 4;
        pub const BEAT: usize = 5;
        pub const DEPTH: usize = 6;
        pub const PHASE: usize = 7;
        pub const OFFSET: usize = 8;
        pub const FADE: usize = 9;
        pub const DELAY: usize = 10;
    }
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// [`index_of`] at compile time, for [`id`]: a uid that is not there
/// fails the build.
#[expect(
    clippy::panic,
    reason = "evaluated only in constants, where a panic is a compile error"
)]
const fn index_of_const(params: &[Spec], uid: &str) -> usize {
    let mut i = 0;
    while i < params.len() {
        if str_eq(params[i].uid, uid) {
            return i;
        }
        i += 1;
    }
    panic!("no such parameter")
}

/// What a knob of this parameter reads at user value `value`.
#[must_use]
pub fn reading(spec: &Spec, value: f32) -> String {
    match spec.unit {
        Unit::Choice(names) => names
            .get(value.round().max(0.0) as usize)
            .map_or_else(|| format!("{value:.0}"), |n| (*n).to_string()),
        Unit::Switch => if value >= 0.5 { "On" } else { "Off" }.to_string(),
        Unit::Semitones => format!("{value:+.0} st"),
        Unit::Cents => format!("{value:+.0} ct"),
        Unit::Decibels => {
            if value <= -100.0 {
                "-inf dB".to_string()
            } else {
                format!("{value:.1} dB")
            }
        }
        Unit::Percent => format!("{value:.0}%"),
        Unit::Seconds => {
            if value < 1.0 {
                format!("{:.0} ms", value * 1000.0)
            } else {
                format!("{value:.2} s")
            }
        }
        Unit::Millis => format!("{value:.1} ms"),
        Unit::Hertz => {
            if value >= 1000.0 {
                format!("{:.1} kHz", value / 1000.0)
            } else {
                format!("{value:.1} Hz")
            }
        }
        Unit::NoteHertz => {
            let hz = bloop_plugin_socalabs::shared::midi_note_in_hertz(value);
            if hz >= 1000.0 {
                format!("{:.1} kHz", hz / 1000.0)
            } else {
                format!("{hz:.0} Hz")
            }
        }
        Unit::Beat => BEATS
            .get(value.round().max(0.0) as usize)
            .map_or_else(|| format!("{value:.0}"), |n| (*n).to_string()),
        Unit::Number => format!("{value:.2}"),
        Unit::Count => format!("{value:.0}"),
    }
}

/// Every parameter's user value, as `gin::Parameter` stores it: snapped to
/// its step and held in its range.
#[derive(Clone, Debug, PartialEq)]
pub struct Values {
    values: [f32; PARAMS.len()],
}

impl Default for Values {
    fn default() -> Self {
        Self {
            values: PARAMS.map(|p| p.default),
        }
    }
}

impl Values {
    /// `setUserValue`: snapped and clamped. Returns whether it moved
    /// (more than float rounding).
    pub fn set(&mut self, index: usize, value: f32) -> bool {
        let Some(spec) = PARAMS.get(index) else {
            return false;
        };
        let v = spec.range.clamp(spec.range.snap(value));
        let slot = &mut self.values[index];
        if bloop_plugin_socalabs::shared::approximately_equal(*slot, v) {
            false
        } else {
            *slot = v;
            true
        }
    }

    /// Every parameter back to its default.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// `getUserValue`.
    #[must_use]
    pub fn user(&self, index: usize) -> f32 {
        self.values[index]
    }

    /// `isOn`: anywhere but the bottom of its range.
    #[must_use]
    pub fn is_on(&self, index: usize) -> bool {
        self.values[index] != PARAMS[index].range.start
    }

    /// `getValue`: where the knob sits, 0 to 1, after the skew.
    #[must_use]
    pub fn normalised(&self, index: usize) -> f32 {
        let range = &PARAMS[index].range;
        bloop_plugin_socalabs::shared::jlimit(0.0, 1.0, range.to_0to1(self.values[index]))
    }

    /// `getProcValue`: the user value through its conversion.
    #[must_use]
    pub fn proc(&self, index: usize) -> f32 {
        PARAMS[index].conversion.apply(self.values[index])
    }
}
