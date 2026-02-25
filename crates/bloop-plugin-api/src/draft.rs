//! bloop-plugin-api: The trait every synth/effect implements.
//!
//! Not part of the build yet — this is a design draft.

// use bloop_core::*;
// use bloop_events::{EventBuffer, NoteEvent};

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
    pub inputs: &'a [&'a [f32]],
    pub outputs: &'a mut [&'a mut [f32]],
    // pub input_events: &'a EventBuffer,
    // pub output_events: &'a mut EventBuffer,
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

    /// Serialize plugin state for project save. Called off audio thread.
    fn save_state(&self) -> Vec<u8>;
    fn restore_state(&mut self, data: &[u8]);
}

// No separate harness — tested through plugin crates and integration tests.
