//! SocaLabs Wavetable as a bloop Plug-in.
//!
//! A Rust port of Roland Rabien's Wavetable (`vendor/wavetable`, BSD-3):
//! two wavetable oscillators with unison, formant and bend, a sub and
//! noise, a multimode filter with its own envelope, three modulation
//! envelopes, three LFOs and a step LFO, each per voice and (the LFOs and
//! step LFO) once more for the whole synth, a modulation matrix, glide and
//! mono, and gate, chorus, distortion, delay and reverb over the output.
//!
//! [`WavetablePlugin`] is the Plug-in; [`params`] lists its parameters,
//! [`presets`] its factory presets, [`bank`] its wavetables. Every piece is
//! held to the C++ it was ported from by `tests/gold.rs` (`--features
//! gold`): bit for bit from a wavetable's frames on, and to within a
//! millionth of full scale (about -120 dB) for the frames themselves,
//! which the original band-limits with JUCE's FFT and bloop with its own
//! (`dsp::fft`).
//!
//! Its wavetables are large and slow to build, so the synth never builds
//! them: the host builds them on the UI thread ([`bank::tables_for`]) and
//! hands them over with [`WavetablePlugin::set_tables`]. Until it does,
//! the oscillators, the sub and the noise are silent.

pub mod bank;
pub mod dsp;
pub mod modmatrix;
pub mod params;
pub mod presets;
mod presets_data;
mod range;
mod synth;
mod voice;

pub use bank::TableSet;
pub use synth::{Event, DELAY_ROOM, VOICE_SLOTS};

use bloop_events::NoteEvent;
use bloop_plugin_api::{ParamInfo, Plugin, PluginCategory, PluginInfo, ProcessContext};
use params::{id, PARAMS};
use std::sync::Arc;
use synth::Synth;

/// The Plug-in's name.
pub const NAME: &str = "Wavetable";
/// Who makes it.
pub const VENDOR: &str = "SocaLabs";

/// The Wavetable revision this is ported from: the one `vendor/wavetable`
/// pins, and the one the License page cites.
pub const PORTED_FROM_WAVETABLE_REV: &str = "5f932ff87c453d32cebee233af1fde946f53ac89";

/// The preset parameter: which factory preset is loaded, the last being
/// Init. Setting it loads that preset over everything.
pub const PARAM_PRESET: usize = params::PARAM_COUNT;

/// The most events one block hands the synth; more are dropped.
const MAX_EVENTS: usize = 256;

/// The Plug-in.
pub struct WavetablePlugin {
    synth: Synth,
    preset: usize,
    events: Box<[(u32, Event); MAX_EVENTS]>,
}

impl Default for WavetablePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl WavetablePlugin {
    /// A Wavetable at Init, prepared for 44.1 kHz. Allocates: UI thread.
    #[must_use]
    pub fn new() -> Self {
        Self::at(44100.0)
    }

    /// A Wavetable at Init, prepared for `sample_rate`, so a host that
    /// runs at that rate resets it without allocating. UI thread.
    #[must_use]
    pub fn at(sample_rate: f64) -> Self {
        let mut synth = Synth::new(1);
        synth.prepare(sample_rate);
        Self {
            synth,
            preset: presets::PRESETS.len(),
            events: Box::new([(0, Event::PitchWheel(8192)); MAX_EVENTS]),
        }
    }

    /// The rate it is prepared for.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.synth.sample_rate()
    }

    /// The wavetables its oscillators and sub read; the set it held comes
    /// back, for the caller to drop on the UI thread. Allocation-free.
    pub fn set_tables(&mut self, tables: Option<Arc<TableSet>>) -> Option<Arc<TableSet>> {
        self.synth.set_tables(tables)
    }

    /// The set it holds.
    #[must_use]
    pub fn tables(&self) -> Option<&Arc<TableSet>> {
        self.synth.tables()
    }

    /// Which catalogue entries its oscillators are set to play.
    #[must_use]
    pub fn table_indices(&self) -> [u16; 2] {
        [
            self.synth.values.user(id::WT1) as u16,
            self.synth.values.user(id::WT2) as u16,
        ]
    }

    /// Whether a voice is sounding.
    #[must_use]
    pub fn is_sounding(&self) -> bool {
        self.synth.is_sounding()
    }

    /// Every voice stopped and every effect's tail emptied, now.
    /// Allocation-free.
    pub fn silence(&mut self) {
        self.synth.silence();
    }

    /// Back to Init, keeping the wavetables it holds.
    pub fn restore_defaults(&mut self) {
        self.load_preset(presets::PRESETS.len());
    }

    /// Preset `index` over everything, as `gin::Program::loadProcessor`
    /// does: every parameter to its default, then the preset's.
    /// Allocation-free.
    pub fn load_preset(&mut self, index: usize) {
        let values = &mut self.synth.values;
        values.reset();
        if let Some(preset) = presets::PRESETS.get(index) {
            for &(uid, value) in preset.params {
                if let Some(at) = params::index_of(uid) {
                    values.set(at, value);
                }
            }
            for (i, table) in preset.tables.into_iter().enumerate() {
                if table != u16::MAX {
                    values.set(id::WT1 + i, f32::from(table));
                }
            }
            for (slot, &(source, destination, depth)) in
                preset.mods.iter().enumerate().take(params::MOD_SLOTS)
            {
                let source = modmatrix::source_named(source).map_or(0, |s| s + 1);
                let destination = params::index_of(destination)
                    .map(|p| modmatrix::DESTINATION_OF[p])
                    .filter(|&d| d >= 0)
                    .map_or(0, |d| d as usize + 1);
                let at = id::MOD1_SRC + slot * 3;
                values.set(at, source as f32);
                values.set(at + 1, destination as f32);
                values.set(at + 2, depth);
            }
        }
        self.preset = index.min(presets::PRESETS.len());
        self.synth.preset_loaded();
    }

    fn take_events(&mut self, context: &ProcessContext) -> usize {
        let mut n = 0;
        for event in context.input_events.iter() {
            if n == MAX_EVENTS {
                break;
            }
            let converted = match *event {
                NoteEvent::NoteOn { note, velocity, .. } => Event::NoteOn {
                    note: note.as_u8(),
                    velocity: seven_bit(velocity),
                },
                NoteEvent::NoteOff { note, .. } => Event::NoteOff { note: note.as_u8() },
                NoteEvent::PitchBend { value, .. } => {
                    Event::PitchWheel((value * 16383.0).round().clamp(0.0, 16383.0) as i32)
                }
                NoteEvent::CC { cc, value, .. } => Event::Controller {
                    cc,
                    value: seven_bit(value),
                },
                NoteEvent::Choke { note, .. } => Event::Choke { note: note.as_u8() },
                // Legacy MIDI mode ignores poly aftertouch, as the original's
                // `MPEInstrument` does.
                NoteEvent::Aftertouch { .. } => continue,
            };
            self.events[n] = (event.time(), converted);
            n += 1;
        }
        n
    }
}

/// A 0-to-1 value as the 7-bit MIDI value it would arrive as.
fn seven_bit(value: f32) -> u8 {
    (value * 127.0).round().clamp(0.0, 127.0) as u8
}

impl Plugin for WavetablePlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: NAME,
            vendor: VENDOR,
            category: PluginCategory::Instrument,
            num_audio_inputs: 0,
            num_audio_outputs: 2,
        }
    }

    fn param_count(&self) -> usize {
        PARAMS.len() + 1
    }

    fn param_info(&self, index: usize) -> ParamInfo {
        param_info(index)
    }

    fn set_param(&mut self, index: usize, value: f32) {
        if index == PARAM_PRESET {
            self.load_preset(value.round().max(0.0) as usize);
        } else {
            self.synth.set_param(index, value);
        }
    }

    fn get_param(&self, index: usize) -> f32 {
        if index == PARAM_PRESET {
            self.preset as f32
        } else if index < PARAMS.len() {
            self.synth.values.user(index)
        } else {
            0.0
        }
    }

    fn process(&mut self, context: &mut ProcessContext) {
        let n = self.take_events(context);
        self.synth.set_bpm(context.tempo as f32);
        let events = &self.events[..n];
        // Stereo out only: the synth's pan, spread and effects are
        // stereo, and every bloop track is.
        if let [left, right, ..] = context.outputs {
            self.synth.process(left, right, events);
        }
    }

    fn reset(&mut self, sample_rate: f32, _max_buffer_size: usize) {
        self.synth.prepare(f64::from(sample_rate));
        self.synth.reset();
    }
}

/// What parameter `index` is called and where it runs.
#[must_use]
pub fn param_info(index: usize) -> ParamInfo {
    if index == PARAM_PRESET {
        return ParamInfo {
            name: "Preset",
            default: presets::PRESETS.len() as f32,
            min: 0.0,
            max: presets::PRESETS.len() as f32,
        };
    }
    PARAMS.get(index).map_or(
        ParamInfo {
            name: "",
            default: 0.0,
            min: 0.0,
            max: 0.0,
        },
        |p| ParamInfo {
            name: p.name,
            default: p.default,
            min: p.range.start,
            max: p.range.end,
        },
    )
}
