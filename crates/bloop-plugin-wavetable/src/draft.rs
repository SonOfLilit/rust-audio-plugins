//! bloop-plugin-wavetable: Wavetable synth.
//! Mipmap anti-aliasing, morphing, filter, ADSR.
//!
//! Not part of the build yet — this is a design draft.

pub struct WavetablePlugin {
    // Wavetable data: Vec of single-cycle waveforms (2048 samples each)
    // Mipmap levels per octave for bandlimiting
    // Current table position (morphable via modulation)
    // Oscillator: phase accumulator, cubic interpolation between mipmap levels
    // Filter (LP/HP/BP with envelope/LFO)
    // ADSR amplitude envelope
    // Voice management
}

// impl Plugin for WavetablePlugin { ... }
