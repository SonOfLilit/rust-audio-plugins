//! What a screen draws a Wavetable from, worked out by the synth's own
//! pieces so the picture is what plays: each frame of a table as the
//! oscillator reads it (Gin's `WavetableComponent`), an LFO's curve, and
//! the filter's response. UI thread; each allocates what it returns.

use crate::dsp::lfo::{Lfo, LfoParams, Shape};
use crate::dsp::osc::{WtOscillator, WtParams};
use crate::dsp::random::JuceRandom;
use crate::dsp::tables::Table;
use crate::params::PARAMS;
use bloop_plugin_socalabs::shared::{midi_note_in_hertz, Filter, FilterType, Slope};

/// How many samples a frame's cycle is drawn in: Gin's.
pub const SAMPLES: usize = 64;

/// The rate a cycle is drawn at: Gin's, so a cycle of [`SAMPLES`] is a
/// note of `44100 / SAMPLES` Hz, and reads the copy of each frame that
/// note would.
const RATE: f64 = 44100.0;

/// One cycle of every frame of `table`, in order, as the oscillator reads
/// it with `bend` and `formant` (`createWavetablePath`): what the stack is
/// drawn from. Empty for an empty table.
#[must_use]
pub fn cycles(table: &Table, bend: f32, formant: f32) -> Vec<[f32; SAMPLES]> {
    let frames = table.len();
    let hz = RATE / SAMPLES as f64;
    let note = (69.0 + 12.0 * libm::log2(hz / 440.0)) as f32;
    (0..frames)
        .map(|frame| {
            let mut osc = WtOscillator::default();
            osc.set_sample_rate(RATE);
            osc.note_on(0.0);
            let params = WtParams {
                left_gain: 1.0,
                right_gain: 1.0,
                // The middle of the frame, so it is the frame chosen.
                position: (frame as f32 + 0.5) / frames as f32,
                bend,
                formant,
            };
            let mut left = [0.0f32; SAMPLES];
            let mut right = [0.0f32; SAMPLES];
            osc.process_adding(Some(table), note, &params, &mut left, &mut right);
            left
        })
        .collect()
}

/// Parameter `index` at its knob's value `user`, as the DSP sees it: a
/// percentage as a fraction, decibels as a gain.
#[must_use]
pub fn converted(index: usize, user: f32) -> f32 {
    PARAMS
        .get(index)
        .map_or(user, |spec| spec.conversion.apply(user))
}

/// What an LFO draws as: its shape (the Wave knob's value), how many
/// cycles fit the picture, and its Phase, Offset and Depth as the DSP sees
/// them ([`converted`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LfoShape {
    pub wave: f32,
    pub cycles: f32,
    pub phase: f32,
    pub offset: f32,
    pub depth: f32,
}

/// The LFO's output at `points` points across a picture `shape.cycles`
/// cycles wide, run through the port's own LFO a point a sample.
#[must_use]
pub fn lfo(shape: LfoShape, points: usize) -> Vec<f32> {
    let mut lfo = Lfo::default();
    let mut system = JuceRandom::new(1);
    lfo.set_sample_rate(points.max(1) as f64);
    lfo.set_parameters(
        LfoParams {
            shape: Shape::from_index(shape.wave as i32),
            frequency: shape.cycles,
            phase: shape.phase,
            offset: shape.offset,
            depth: shape.depth,
            delay: 0.0,
            fade: 0.0,
        },
        &mut system,
    );
    lfo.note_on(0.0, &mut system);
    let mut out = Vec::with_capacity(points);
    out.push(lfo.output());
    for _ in 1..points {
        lfo.process(1);
        out.push(lfo.output());
    }
    out
}

/// The filter's gain, as a factor, at each of `freqs` (Hz): its Type knob
/// at `kind`, its Cutoff at `note` (the knob's value, a MIDI note) and its
/// Resonance at `resonance` percent, as the voice sets it with no
/// envelope, key or velocity moving it.
#[must_use]
pub fn filter_response(kind: f32, note: f32, resonance: f32, freqs: &[f32]) -> Vec<f32> {
    const Q: f32 = 0.707_106_77;
    let (kind, slope) = match kind.round() as i32 {
        0 => (FilterType::LoPass, Slope::Db12),
        1 => (FilterType::LoPass, Slope::Db24),
        2 => (FilterType::HiPass, Slope::Db12),
        3 => (FilterType::HiPass, Slope::Db24),
        4 => (FilterType::BandPass, Slope::Db12),
        5 => (FilterType::BandPass, Slope::Db24),
        6 => (FilterType::Notch, Slope::Db12),
        _ => (FilterType::Notch, Slope::Db24),
    };
    let rate = 48000.0;
    let mut filter = Filter::new(kind);
    filter.set_sample_rate(rate);
    filter.set_slope(slope);
    let f = midi_note_in_hertz(note).clamp(4.0, 20000.0);
    let q = Q / (1.0 - (resonance / 100.0) * 0.99);
    filter.set_params(f, q, 0.0);
    freqs
        .iter()
        .map(|&freq| filter.response_magnitude(freq))
        .collect()
}
