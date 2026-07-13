//! bloop-plugin-dexed: 6-operator FM synth (DX7-compatible).
//!
//! A Rust port of the MSFA engine (music-synthesizer-for-android,
//! Apache-2.0) as maintained in Dexed's `Source/msfa` fork. Statically
//! linked — Apache-2.0 needs no dynamic-linking boundary (unlike PAPU's
//! LGPL). Ported files carry their original attribution in each module
//! header. Left for the roadmap: portamento, microtuning, MPE, CC
//! modulation routing (mod wheel etc.), operator on/off switches, custom
//! cartridge loading.

mod dx7note;
mod env;
mod fm_core;
mod lfo;
mod patch;
mod pitchenv;
mod tables;

pub use patch::{Bank, BankError, BANK_SIZE, PACKED_VOICE_SIZE, VOICE_SIZE};

use bloop_events::NoteEvent;
use bloop_plugin_api::{ParamInfo, Plugin, PluginCategory, PluginInfo, ProcessContext};
use dx7note::{Controllers, Dx7Note};
use fm_core::{FmCore, N};
use lfo::Lfo;
use tables::Tables;

/// Polyphony. 8 voices covers pads comfortably; bump when a use case needs
/// more (roadmap).
pub const NUM_VOICES: usize = 8;

pub const PARAM_VOLUME: usize = 0;
pub const PARAM_PRESET: usize = 1;
const PARAM_COUNT: usize = 2;

struct ParamDef {
    name: &'static str,
    min: f32,
    max: f32,
    default: f32,
}

static PARAM_DEFS: [ParamDef; PARAM_COUNT] = [
    ParamDef { name: "Volume", min: 0.0, max: 1.0, default: 1.0 },
    ParamDef { name: "Preset", min: 0.0, max: 31.0, default: 0.0 },
];

#[derive(Clone, Copy, Default)]
struct Voice {
    note: Dx7Note,
    midinote: u8,
    keydown: bool,
    /// Allocation order, for oldest-first stealing.
    serial: u32,
}

pub struct DexedPlugin {
    tables: Box<Tables>,
    sample_rate: f64,
    core: FmCore,
    lfo: Lfo,
    ctrls: Controllers,
    voices: [Voice; NUM_VOICES],
    serial: u32,
    /// The active unpacked patch.
    patch: [u8; VOICE_SIZE],
    bank: Bank,
    params: [f32; PARAM_COUNT],
    /// Rendered samples carried over when the host block isn't a multiple
    /// of the engine chunk size `N`.
    extra: [f32; N],
    extra_len: usize,
    extra_pos: usize,
}

impl Default for DexedPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DexedPlugin {
    pub fn new() -> Self {
        let bank = Bank::rom1a();
        let mut plugin = Self {
            tables: Box::new(Tables::new(48000.0)),
            sample_rate: 48000.0,
            core: FmCore::default(),
            lfo: Lfo::default(),
            ctrls: Controllers::default(),
            voices: [Voice::default(); NUM_VOICES],
            serial: 0,
            patch: bank.voice(0),
            bank,
            params: [0.0; PARAM_COUNT],
            extra: [0.0; N],
            extra_len: 0,
            extra_pos: 0,
        };
        for (i, def) in PARAM_DEFS.iter().enumerate() {
            plugin.params[i] = def.default;
        }
        plugin.load_preset(0);
        plugin
    }

    /// Names of the built-in bank's 32 presets, in order.
    pub fn preset_names(&self) -> Vec<String> {
        (0..patch::BANK_SIZE).map(|i| self.bank.name(i)).collect()
    }

    fn load_preset(&mut self, index: usize) {
        self.patch = self.bank.voice(index);
        let mut lfo_params = [0u8; 6];
        lfo_params.copy_from_slice(&self.patch[137..143]);
        self.lfo.reset(&self.tables, &lfo_params);
    }

    fn note_on(&mut self, midinote: u8, velocity: f32) {
        // DX7 transpose: patch[144] is 0..48 with 24 = no transpose.
        let transpose = i32::from(self.patch[144]) - 24;
        let note = (i32::from(midinote) + transpose).clamp(0, 127);
        let vel = (velocity * 127.0).round().clamp(0.0, 127.0) as i32;

        let slot = (0..NUM_VOICES)
            .find(|&v| !self.voices[v].keydown && !self.voices[v].note.is_playing())
            .unwrap_or_else(|| {
                // All voices busy: steal the oldest.
                (0..NUM_VOICES)
                    .min_by_key(|&v| self.voices[v].serial)
                    .unwrap_or(0)
            });
        self.serial = self.serial.wrapping_add(1);
        let voice = &mut self.voices[slot];
        voice.note.init(&self.tables, &self.patch, note, vel);
        voice.midinote = midinote;
        voice.keydown = true;
        voice.serial = self.serial;
        self.lfo.keydown();
    }

    fn note_off(&mut self, midinote: u8) {
        for voice in &mut self.voices {
            if voice.keydown && voice.midinote == midinote {
                voice.note.keyup(&self.tables);
                voice.keydown = false;
            }
        }
    }

    fn apply_event(&mut self, event: &NoteEvent) {
        match *event {
            NoteEvent::NoteOn { note, velocity, .. } => self.note_on(note.as_u8(), velocity),
            NoteEvent::NoteOff { note, .. } => self.note_off(note.as_u8()),
            NoteEvent::PitchBend { value, .. } => {
                self.ctrls.pitch_bend = (value * 16383.0).round().clamp(0.0, 16383.0) as i32;
            }
            _ => {}
        }
    }

    /// Render one engine chunk into `self.extra` (post-volume, mono).
    fn render_chunk(&mut self) {
        let mut buf = [0i32; N];
        let lfo_value = self.lfo.getsample(&self.tables);
        let lfo_delay = self.lfo.getdelay();
        for voice in &mut self.voices {
            if voice.keydown || voice.note.is_playing() {
                voice.note.compute(
                    &self.tables,
                    &mut self.core,
                    &mut buf,
                    lfo_value,
                    lfo_delay,
                    &self.ctrls,
                );
            }
        }
        let volume = self.params[PARAM_VOLUME];
        for (out, &v) in self.extra.iter_mut().zip(buf.iter()) {
            // Same headroom/clip staging as MSFA's synth_unit: >>4 then
            // clamp to ±2^24, mapped here to ±1.0 floats.
            let val = (v >> 4).clamp(-(1 << 24), 1 << 24);
            *out = val as f32 / 16_777_216.0 * volume; // ±2^24 → ±1.0
        }
        self.extra_len = N;
        self.extra_pos = 0;
    }
}

impl Plugin for DexedPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "FM (DX7)",
            vendor: "Bloop",
            category: PluginCategory::Instrument,
            num_audio_inputs: 0,
            num_audio_outputs: 2,
        }
    }

    fn param_count(&self) -> usize {
        PARAM_COUNT
    }

    fn param_info(&self, index: usize) -> ParamInfo {
        if index < PARAM_COUNT {
            let d = &PARAM_DEFS[index];
            ParamInfo { name: d.name, default: d.default, min: d.min, max: d.max }
        } else {
            ParamInfo { name: "?", default: 0.0, min: 0.0, max: 1.0 }
        }
    }

    fn set_param(&mut self, index: usize, value: f32) {
        if index < PARAM_COUNT {
            self.params[index] = value;
            if index == PARAM_PRESET {
                let preset = (value.round() as usize).min(patch::BANK_SIZE - 1);
                self.load_preset(preset);
            }
        }
    }

    fn get_param(&self, index: usize) -> f32 {
        if index < PARAM_COUNT {
            self.params[index]
        } else {
            0.0
        }
    }

    fn process(&mut self, context: &mut ProcessContext) {
        let block_size = context.outputs[0].len();
        let mut written = 0usize;
        let mut next_event = 0usize;
        let events = context.input_events;

        while written < block_size {
            // Apply events that are due (engine-chunk granularity, like
            // the original: envelopes/LFO advance per 64-sample chunk).
            while let Some(e) = events.iter().nth(next_event).copied() {
                if event_time(&e) as usize > written {
                    break;
                }
                self.apply_event(&e);
                next_event += 1;
            }

            if self.extra_pos >= self.extra_len {
                self.render_chunk();
            }
            let take = (self.extra_len - self.extra_pos).min(block_size - written);
            for i in 0..take {
                let s = self.extra[self.extra_pos + i];
                for out in context.outputs.iter_mut() {
                    out[written + i] = s;
                }
            }
            self.extra_pos += take;
            written += take;
        }

        // Events past the last chunk boundary (or arriving while carry
        // filled the whole block) still take effect for the next block.
        while let Some(e) = events.iter().nth(next_event).copied() {
            self.apply_event(&e);
            next_event += 1;
        }
    }

    fn reset(&mut self, sample_rate: f32, _max_buffer_size: usize) {
        let sample_rate = f64::from(sample_rate);
        if sample_rate != self.sample_rate {
            self.sample_rate = sample_rate;
            *self.tables = Tables::new(sample_rate);
        }
        self.voices = [Voice::default(); NUM_VOICES];
        self.extra_len = 0;
        self.extra_pos = 0;
        let preset = (self.params[PARAM_PRESET].round() as usize).min(patch::BANK_SIZE - 1);
        self.load_preset(preset);
    }
}

fn event_time(event: &NoteEvent) -> u32 {
    match *event {
        NoteEvent::NoteOn { time, .. }
        | NoteEvent::NoteOff { time, .. }
        | NoteEvent::Aftertouch { time, .. }
        | NoteEvent::PitchBend { time, .. }
        | NoteEvent::CC { time, .. } => time,
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Shared analysis helpers for the snapshot harnesses.

    /// Magnitude of `samples` at `freq_hz` (Goertzel), normalized so a
    /// full-scale sine reports ~1.0.
    pub fn goertzel(samples: &[f32], sample_rate: f64, freq_hz: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * freq_hz / sample_rate;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &x in samples {
            let s0 = f64::from(x) + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
        (power.max(0.0)).sqrt() / (samples.len() as f64 / 2.0)
    }

    /// The frequency (whole Hz, 30..=2000) with the strongest response.
    pub fn dominant_freq(samples: &[f32], sample_rate: f64) -> u32 {
        let mut best = (30u32, f64::MIN);
        let mut hz = 30u32;
        while hz <= 2000 {
            let mag = goertzel(samples, sample_rate, f64::from(hz));
            if mag > best.1 {
                best = (hz, mag);
            }
            hz += 1;
        }
        best.0
    }

    /// Magnitudes at multiples of `f0`, in dB relative to the strongest,
    /// formatted one line per harmonic ("h1 0.0dB" …). Harmonics under
    /// -40 dB print as "-".
    pub fn harmonic_profile(
        samples: &[f32],
        sample_rate: f64,
        f0: f64,
        harmonics: usize,
    ) -> Vec<String> {
        let mags: Vec<f64> = (1..=harmonics)
            .map(|h| goertzel(samples, sample_rate, f0 * h as f64))
            .collect();
        let max = mags.iter().cloned().fold(1e-12, f64::max);
        mags.iter()
            .enumerate()
            .map(|(i, &m)| {
                let db = 20.0 * (m / max).log10();
                if db < -40.0 {
                    format!("h{} -", i + 1)
                } else {
                    format!("h{} {db:.1}dB", i + 1)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{dominant_freq, goertzel};
    use bloop_events::EventBuffer;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    const BLOCK: usize = 128;
    const SR: f32 = 48000.0;

    /// Scheduled event at a block index: a note event or a param change.
    #[derive(Clone)]
    enum Ev {
        Note(NoteEvent),
        Param(usize, f32),
    }

    fn note_on(note: u8, velocity: f32) -> Ev {
        Ev::Note(NoteEvent::NoteOn {
            time: 0,
            channel: 0,
            note: bloop_core::Note::new(note),
            velocity,
        })
    }

    fn note_off(note: u8) -> Ev {
        Ev::Note(NoteEvent::NoteOff {
            time: 0,
            channel: 0,
            note: bloop_core::Note::new(note),
        })
    }

    fn bend(value: f32) -> Ev {
        Ev::Note(NoteEvent::PitchBend { time: 0, channel: 0, value })
    }

    /// Render `num_blocks` 128-sample blocks with scheduled events;
    /// returns the left channel.
    fn render_blocks(
        synth: &mut DexedPlugin,
        num_blocks: usize,
        events: &[(usize, Ev)],
    ) -> Vec<f32> {
        let mut all = Vec::with_capacity(num_blocks * BLOCK);
        for block in 0..num_blocks {
            let mut buf = EventBuffer::new();
            for (b, ev) in events {
                if *b == block {
                    match ev {
                        Ev::Note(e) => {
                            buf.push(*e);
                        }
                        Ev::Param(idx, val) => synth.set_param(*idx, *val),
                    }
                }
            }
            let mut left = vec![0.0f32; BLOCK];
            let mut right = vec![0.0f32; BLOCK];
            let mut outputs: Vec<&mut [f32]> = vec![&mut left[..], &mut right[..]];
            let mut ctx = ProcessContext {
                outputs: &mut outputs,
                input_events: &buf,
                sample_rate: SR,
                tempo: 120.0,
                is_playing: false,
            };
            synth.process(&mut ctx);
            all.extend_from_slice(&left);
        }
        all
    }

    /// Peak |sample| per window of `window_blocks` blocks: a direct trace
    /// of the amplitude envelope (attack / sustain / release).
    fn format_peaks(samples: &[f32], window_blocks: usize) -> Vec<String> {
        samples
            .chunks(window_blocks * BLOCK)
            .enumerate()
            .map(|(w, chunk)| {
                let peak = chunk.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
                format!("w{w:>2} peak={peak:.4}")
            })
            .collect()
    }

    /// The main harness: select a preset, run the event script, snapshot
    /// the amplitude-envelope trace (peaks per 8-block ≈ 21ms window).
    fn check(preset: f32, events: &[(usize, Ev)], num_blocks: usize, expect: Expect) {
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, preset);
        let samples = render_blocks(&mut synth, num_blocks, events);
        expect.assert_eq(&format_peaks(&samples, 8).join("\n"));
    }

    /// Pitch harness: play a note on a preset, measure the dominant
    /// frequency over the sustained portion.
    fn check_pitch(preset: f32, events: &[(usize, Ev)], expect: Expect) {
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, preset);
        let samples = render_blocks(&mut synth, 64, events);
        // Skip the attack (first 16 blocks), analyze the rest.
        let sustained = &samples[16 * BLOCK..];
        expect.assert_eq(&format!("dominant={}Hz", dominant_freq(sustained, f64::from(SR))));
    }

    const EPIANO: f32 = 10.0; // ROM1A "E.PIANO 1"
    const EORGAN: f32 = 16.0; // ROM1A "E.ORGAN 1" (sustained, near-organ tone)
    const MARIMBA: f32 = 21.0; // ROM1A "MARIMBA" (fast natural decay)
    const BRASS: f32 = 0.0; // ROM1A "BRASS 1" (slow attack, sustains)

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn epiano_gate() {
        // E.PIANO 1, one A4: fast attack, decaying sustain while held,
        // release to silence after note-off at block 48.
        check(
            EPIANO,
            &[(0, note_on(69, 1.0)), (48, note_off(69))],
            96,
            expect![[""]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn brass_attack_sustains() {
        // BRASS 1 ramps up slowly and holds while the key is down.
        check(
            BRASS,
            &[(0, note_on(60, 1.0)), (56, note_off(60))],
            80,
            expect![[""]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn marimba_decays_naturally() {
        // MARIMBA decays to silence on its own even while the key is held
        // (L4 = 0 percussive envelope).
        check(MARIMBA, &[(0, note_on(60, 1.0))], 96, expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn organ_pitch_tracks_notes() {
        // A4 (440 Hz) and A3 (220 Hz): the dominant frequency must track
        // the played note.
        check_pitch(EORGAN, &[(0, note_on(69, 1.0))], expect![[""]]);
        check_pitch(EORGAN, &[(0, note_on(57, 1.0))], expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn pitch_bend_full_up_is_two_semitones() {
        // Bend fully up (default range 2 semitones): A4 → B4 ≈ 494 Hz.
        check_pitch(
            EORGAN,
            &[(0, note_on(69, 1.0)), (0, bend(1.0))],
            expect![[""]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn velocity_scales_level() {
        // Same note at full and low velocity: quieter, not just scaled —
        // DX7 velocity also softens the modulator (timbre), so we compare
        // peak amplitude windows.
        check(EPIANO, &[(0, note_on(69, 1.0))], 32, expect![[""]]);
        check(EPIANO, &[(0, note_on(69, 0.3))], 32, expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn chord_is_polyphonic() {
        // C major triad: all three notes sound (energy at each pitch).
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, EORGAN);
        let samples = render_blocks(
            &mut synth,
            64,
            &[(0, note_on(60, 1.0)), (0, note_on(64, 1.0)), (0, note_on(67, 1.0))],
        );
        let sustained = &samples[16 * BLOCK..];
        let mags: Vec<String> = [261.63, 329.63, 392.0, 466.16]
            .iter()
            .map(|&hz| {
                let m = goertzel(sustained, f64::from(SR), hz);
                format!("{hz}Hz {}", if m > 0.01 { "sounding" } else { "-" })
            })
            .collect();
        expect![[""]].assert_eq(&mags.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn voice_stealing_keeps_playing() {
        // 10 note-ons exceed the 8 voices: the two oldest are stolen, the
        // synth keeps rendering (no panic), and releasing everything
        // still fades to silence (no stuck voices).
        let events: Vec<(usize, Ev)> = (0..10)
            .map(|i| (i, note_on(48 + i as u8 * 2, 1.0)))
            .chain((0..10).map(|i| (40 + i, note_off(48 + i as u8 * 2))))
            .collect();
        check(EORGAN, &events, 96, expect![[""]]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn preset_switch_changes_sound() {
        // Switch from sustained BRASS to percussive MARIMBA mid-run: the
        // note played after the switch decays on its own.
        check(
            BRASS,
            &[
                (0, note_on(60, 1.0)),
                (24, note_off(60)),
                (32, Ev::Param(PARAM_PRESET, MARIMBA)),
                (36, note_on(60, 1.0)),
            ],
            112,
            expect![[""]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn volume_param_scales_output() {
        check(
            EORGAN,
            &[(0, Ev::Param(PARAM_VOLUME, 0.25)), (0, note_on(69, 1.0))],
            32,
            expect![[""]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn odd_block_sizes_render_identically() {
        // The engine renders in 64-sample chunks; the carry buffer must
        // make output independent of the host block size.
        let render = |sizes: &[usize]| -> Vec<f32> {
            let mut synth = DexedPlugin::new();
            synth.reset(SR, 1024);
            synth.set_param(PARAM_PRESET, EORGAN);
            let mut ev = EventBuffer::new();
            ev.push(NoteEvent::NoteOn {
                time: 0,
                channel: 0,
                note: bloop_core::Note::new(69),
                velocity: 1.0,
            });
            let empty = EventBuffer::new();
            let mut all = Vec::new();
            let mut first = true;
            for &size in sizes {
                let mut left = vec![0.0f32; size];
                let mut right = vec![0.0f32; size];
                let mut outputs: Vec<&mut [f32]> = vec![&mut left[..], &mut right[..]];
                let mut ctx = ProcessContext {
                    outputs: &mut outputs,
                    input_events: if first { &ev } else { &empty },
                    sample_rate: SR,
                    tempo: 120.0,
                    is_playing: false,
                };
                synth.process(&mut ctx);
                all.extend_from_slice(&left);
                first = false;
            }
            all
        };
        let a = render(&[128; 6]);
        let b = render(&[96, 32, 128, 1, 63, 64, 128, 128, 128, 96, 32]);
        assert_eq!(a.len(), b.len());
        let identical = a == b;
        expect![[""]].assert_eq(&format!("identical={identical}"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn preset_names_are_rom1a() {
        let synth = DexedPlugin::new();
        expect![[""]].assert_eq(&synth.preset_names().join("\n"));
    }
}
