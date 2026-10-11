//! bloop-plugin-api: The trait every synth/effect implements.

use bloop_events::EventBuffer;

pub mod meter;
pub mod trace;
pub use meter::{DynamicsMeter, LevelTracker};
pub use trace::Trace;

#[derive(Clone, Debug)]
pub struct PluginInfo {
    pub name: &'static str,
    pub vendor: &'static str,
    pub category: PluginCategory,
    pub num_audio_inputs: usize,
    pub num_audio_outputs: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginCategory {
    Instrument,
    Effect,
}

#[derive(Clone, Debug)]
pub struct ParamInfo {
    pub name: &'static str,
    pub default: f32,
    pub min: f32,
    pub max: f32,
}

/// How much a render may spend on a sample: what an effect with a cheap
/// and a costly way to run (NAM's fast or exact `tanh`) is told to use.
///
/// Belongs to the render, not the Project: the same Project plays [`Live`]
/// through the speakers and renders at [`Render`] when it is exported or
/// bounced, which nobody waits on in real time.
///
/// [`Live`]: Quality::Live
/// [`Render`]: Quality::Render
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quality {
    /// Played as it happens: has to keep up on a phone.
    #[default]
    Live,
    /// Rendered offline (EXPORT, a bounce): may take longer than real time.
    Render,
}

/// Context passed to Plugin::process each audio block.
pub struct ProcessContext<'a> {
    pub outputs: &'a mut [&'a mut [f32]],
    pub input_events: &'a EventBuffer,
    pub sample_rate: f32,
    pub tempo: f64,
    pub is_playing: bool,
}

/// Where the Pattern a plug-in plays in is, for one that plays in time
/// with it (the Audio plug-in's loops). The engine fills one per block
/// before `process`: whether the transport runs, and the beat — quarter
/// notes since the Pattern's start — at the first sample of each stretch
/// of the block it played straight through. Between anchors and after the
/// last the beat moves at the tempo. A new anchor is a jump: the Pattern
/// wrapped, the song moved on to another, or the playhead was moved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timeline {
    pub playing: bool,
    anchors: [TimelineAnchor; Timeline::MAX_ANCHORS],
    len: usize,
}

/// The beat at one sample of a block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TimelineAnchor {
    pub offset: u32,
    pub beat: f64,
}

impl Timeline {
    /// More jumps than this in one block are not kept: a block holds a
    /// few hundred samples, and a Pattern shorter than that is not one
    /// anybody loops audio in.
    pub const MAX_ANCHORS: usize = 8;

    /// The transport stopped: nothing to be in time with.
    #[must_use]
    pub const fn stopped() -> Self {
        Self {
            playing: false,
            anchors: [TimelineAnchor {
                offset: 0,
                beat: 0.0,
            }; Self::MAX_ANCHORS],
            len: 0,
        }
    }

    /// Running, with the Pattern at `beat` at sample `offset`. Anchors are
    /// pushed in offset order; one past [`Self::MAX_ANCHORS`] is dropped
    /// and `false` returned.
    pub fn push(&mut self, offset: u32, beat: f64) -> bool {
        self.playing = true;
        if self.len >= Self::MAX_ANCHORS {
            return false;
        }
        self.anchors[self.len] = TimelineAnchor { offset, beat };
        self.len += 1;
        true
    }

    #[must_use]
    pub fn anchors(&self) -> &[TimelineAnchor] {
        &self.anchors[..self.len]
    }

    /// The beat at `offset` (which may lie past the block), moving at
    /// `beats_per_sample` from the last anchor at or before it; `None`
    /// while stopped.
    #[must_use]
    pub fn beat_at(&self, offset: f64, beats_per_sample: f64) -> Option<f64> {
        if !self.playing {
            return None;
        }
        let anchor = self
            .anchors()
            .iter()
            .rev()
            .find(|a| f64::from(a.offset) <= offset)
            .or_else(|| self.anchors().first())?;
        Some(anchor.beat + (offset - f64::from(anchor.offset)) * beats_per_sample)
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::stopped()
    }
}

/// The trait every plugin implements.
pub trait Plugin: Send {
    fn info(&self) -> PluginInfo;
    fn param_count(&self) -> usize;
    fn param_info(&self, index: usize) -> ParamInfo;
    fn set_param(&mut self, index: usize, value: f32);
    fn get_param(&self, index: usize) -> f32;

    /// Process one block of audio. Must be allocation-free.
    fn process(&mut self, context: &mut ProcessContext);

    fn reset(&mut self, sample_rate: f32, max_buffer_size: usize);

    /// Where every value set so far is headed, there at once: a knob the
    /// plug-in smooths jumps to where it was set rather than easing there
    /// from where it stood. Whatever the plug-in holds — a voice, a
    /// delay's line, an LFO's phase — stays as it is.
    ///
    /// The engine calls it on a plug-in's first block, once everything it
    /// was told before then has arrived: a Project opened, an export or a
    /// bounce starts each knob where the Project has it, not at the
    /// plug-in's default for the first tenth of a second. A knob turned
    /// while it plays still eases. Called on the audio thread, so it must
    /// be allocation-free.
    ///
    /// The default is for a plug-in that smooths nothing.
    fn settle(&mut self) {}

    /// What every voice is doing right now, for a screen to draw while it
    /// plays: overwrites all of `out`. Called on the audio thread after
    /// `process`, so it must be allocation-free and cheap.
    ///
    /// The default has no voices to show and writes every one inactive.
    fn watch(&self, out: &mut Watch) {
        *out = Watch::default();
    }

    /// Parameters `params` — by index, [`NO_PARAM`] for an empty knob — as
    /// each voice hears them now, into `out`, voice for voice as
    /// [`Plugin::watch`] numbers them: what a knob's ring draws while a
    /// modulation route moves it. Called on the audio thread after
    /// `watch`, so it must be allocation-free and cheap.
    ///
    /// The default says nothing and leaves `out` as it is: the engine
    /// clears it first.
    fn watch_params(
        &self,
        _params: &[u16; WATCH_KNOBS],
        _out: &mut [[f32; WATCH_KNOBS]; WATCH_VOICES],
    ) {
    }
}

/// A knob [`Plugin::watch_params`] is asked about that has no parameter.
pub const NO_PARAM: u16 = u16::MAX;

/// How many voices a [`Watch`] can show: room for the most any plug-in
/// here has (a plug-in that fills one asserts its voices fit).
pub const WATCH_VOICES: usize = 8;
/// How many numbers a voice carries in a [`Watch`]: what each means is
/// the plug-in's own to say, and its screen's to read.
pub const WATCH_LANES: usize = 18;
/// How many numbers the whole instrument carries beside its voices.
pub const WATCH_GLOBALS: usize = 16;
/// How many knobs a page has, and so how many parameters a [`Watch`]
/// follows through the voices ([`Plugin::watch_params`]).
pub const WATCH_KNOBS: usize = 8;

/// One voice of the instrument.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoiceWatch {
    /// Sounding: its key is down, or it is still ringing out. Every other
    /// field of an inactive voice is its default.
    pub active: bool,
    /// Which note-on of this plug-in instance started the voice: 1 for its
    /// first since it was built, and on, wrapping from `u32::MAX` through
    /// 0 (a screen colouring by it modulo 3 sees one seam). Stable
    /// for the voice's life, so a screen can colour it by this and have
    /// the colour stay put while voices start and stop around it. Nothing
    /// more: another plug-in on the track keeps its own count.
    pub note_on: u32,
    /// Whether its key is still down (as opposed to ringing out its
    /// release).
    pub key_down: bool,
    /// The key it sounds, as a MIDI note number: the key struck, moved by
    /// the patch's own transpose; pitch bend and modulation are not in it.
    pub pitch: u8,
    /// What the voice is doing, in the plug-in's own numbering: the FM
    /// synth's operator levels and gains, a subtractive synth's envelope
    /// stages and LFO phases. The API says nothing about what a lane
    /// means, so it names no plug-in; a lane a plug-in does not use is 0.
    pub lanes: [f32; WATCH_LANES],
}

/// What an instrument's voices are doing, for a screen that draws them as
/// they play. Fixed-size and `Copy`, so the audio thread can hand it over
/// in its meters without allocating. A voice keeps its index for as long as
/// it sounds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Watch {
    pub voices: [VoiceWatch; WATCH_VOICES],
    /// What the instrument is doing beside its voices, in its own
    /// numbering: a mono LFO's phase. 0 where unused.
    pub globals: [f32; WATCH_GLOBALS],
    /// Per voice, the eight parameters the engine was asked to follow
    /// (`AudioCommand::WatchParams`: the knob page showing) as each voice
    /// hears them, modulation and all, in the parameters' own units. All 0
    /// for a plug-in that does not say ([`Plugin::watch_params`]).
    pub knobs: [[f32; WATCH_KNOBS]; WATCH_VOICES],
}

impl Watch {
    /// The voices that are sounding, with their indices.
    pub fn active(&self) -> impl Iterator<Item = (usize, &VoiceWatch)> {
        self.voices.iter().enumerate().filter(|(_, v)| v.active)
    }
}
