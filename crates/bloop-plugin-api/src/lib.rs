//! bloop-plugin-api: The trait every synth/effect implements.

use bloop_events::EventBuffer;

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

/// Context passed to Plugin::process each audio block.
pub struct ProcessContext<'a> {
    pub outputs: &'a mut [&'a mut [f32]],
    pub input_events: &'a EventBuffer,
    pub sample_rate: f32,
    pub tempo: f64,
    pub is_playing: bool,
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

    /// What every voice is doing right now, for a screen to draw while it
    /// plays: overwrites all of `out`. Called on the audio thread after
    /// `process`, so it must be allocation-free and cheap.
    ///
    /// The default has no voices to show and writes every one inactive.
    fn watch(&self, out: &mut Watch) {
        *out = Watch::default();
    }
}

/// How many voices a [`Watch`] can show: room for the most any plug-in
/// here has (a plug-in that fills one asserts its voices fit).
pub const WATCH_VOICES: usize = 8;
/// How many operators a voice has in a [`Watch`].
pub const WATCH_OPS: usize = 6;

/// Where one operator's envelope is, as a four-rate, four-level envelope
/// walks it: rate 1 toward level 1, rate 2 toward level 2, rate 3 toward
/// level 3, held at level 3 while the key is down, then rate 4 toward
/// level 4 once it is let go — from wherever it had got to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnvStage {
    /// Key down, moving toward level 1 at rate 1: the attack.
    #[default]
    Rate1,
    /// Key down, moving toward level 2 at rate 2.
    Rate2,
    /// Key down, moving toward level 3 at rate 3.
    Rate3,
    /// Key down, at level 3, and staying there until the key comes up.
    Sustain,
    /// Key up, moving toward level 4 at rate 4: the release.
    Rate4,
    /// At level 4, where it stays: silence when level 4 is 0, a note that
    /// rings on for ever when it is not.
    End,
}

/// One operator of a sounding voice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpWatch {
    pub stage: EnvStage,
    /// Where the envelope is, in the patch's own level units — the scale
    /// its levels are written in — so a dot drawn at it sits on the
    /// envelope the patch draws. Exactly the patch's
    /// level while the envelope holds one (`Sustain`, `End`, and a level
    /// the envelope is waiting at); between levels, the level the engine's
    /// own envelope is at, read back to that scale.
    ///
    /// The envelope is the one the voice took at its note-on: an edit made
    /// while it sounds is heard from the next note, and so is drawn there.
    /// The operator's output level, velocity and keyboard scaling are not
    /// in it — those are in `amp` — except that an envelope's attack
    /// starts from a fixed loudness, and so from higher on its own scale
    /// the quieter the operator is.
    pub level: f32,
    /// How loud the operator is: its envelope with every scaling applied
    /// (output level, velocity, keyboard level scaling, amplitude
    /// modulation), as a linear gain where 1.0 is an operator at full
    /// output level and full envelope level struck at a velocity its
    /// sensitivity leaves alone. A loud hit on a velocity-sensitive
    /// operator can go past 1.0.
    ///
    /// The gain at the end of the last chunk the engine rendered. Exactly
    /// 0.0 when, at that chunk's end, the engine would not render the
    /// operator — switched off, at output level 0, or below the level it
    /// skips — and for a voice struck too late in the block to have
    /// rendered yet.
    ///
    /// This is the gain the operator's sine is multiplied by. In an
    /// algorithm where it modulates another operator, that is its
    /// modulation depth rather than anything heard directly.
    pub amp: f32,
}

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
    /// Its operators, in the order the patch's own pages number them: index
    /// 0 is operator 1.
    pub ops: [OpWatch; WATCH_OPS],
}

/// What an instrument's voices are doing, for a screen that draws them as
/// they play. Fixed-size and `Copy`, so the audio thread can hand it over
/// in its meters without allocating. A voice keeps its index for as long as
/// it sounds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Watch {
    pub voices: [VoiceWatch; WATCH_VOICES],
}

impl Watch {
    /// The voices that are sounding, with their indices.
    pub fn active(&self) -> impl Iterator<Item = (usize, &VoiceWatch)> {
        self.voices.iter().enumerate().filter(|(_, v)| v.active)
    }
}
