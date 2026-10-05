//! Ports of slPlugins' effect processors, one module per plug-in family.

pub mod bitcrusher;
pub mod crossfeed;
pub mod delay;
pub mod dynamics;
pub mod extras;
pub mod graphiceq;
pub mod scopes;
pub mod simpleverb;
pub mod stereo;

use crate::gin::param::Param;

/// What a processor may ask of its host: `getPlayHead()`, as far as the
/// ported processors read it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Host {
    pub bpm: f32,
}

/// What each ported processor does: its parameters, and the three calls a
/// host makes of a JUCE `AudioProcessor` that reach audio.
pub trait Processor: Send {
    fn params(&self) -> &[Param];
    fn params_mut(&mut self) -> &mut [Param];
    /// `prepareToPlay`, then `reset`, as a host does before the first
    /// block.
    fn prepare(&mut self, sample_rate: f64);
    fn reset(&mut self);
    /// Silence whatever the effect still holds and keep its knobs: bloop's
    /// Choke All Notes. The original's `reset` does that for most effects,
    /// so it is the default; one whose `reset` leaves its state alone says
    /// what clearing it takes.
    fn clear(&mut self) {
        self.reset();
    }
    /// `processBlock`, on a stereo pair in place.
    fn process(&mut self, left: &mut [f32], right: &mut [f32], host: Host);
}

/// `gin::Processor::prepareToPlay` and `reset` over a parameter list.
pub fn prepare_params(params: &mut [Param], sample_rate: f64) {
    for p in params {
        p.prepare(sample_rate);
    }
}

pub fn reset_params(params: &mut [Param]) {
    for p in params {
        p.reset();
    }
}
