//! bloop-plugin-dexed: 6-operator FM synth (DX7-compatible).
//!
//! A Rust port of the MSFA engine (music-synthesizer-for-android,
//! Apache-2.0) as maintained in Dexed's `Source/msfa` fork. Statically
//! linked — Apache-2.0 needs no dynamic-linking boundary (unlike PAPU's
//! LGPL). Ported files carry their original attribution in each module
//! header. Left for the roadmap: portamento, microtuning, MPE, CC
//! modulation routing (mod wheel etc.). A user's own cartridges are read by
//! [`read_sysex`] and played under the preset numbers [`add_voice`] gives
//! them, past the built-in library's.

mod dx7note;
mod env;
mod fm_core;
mod lfo;
mod patch;
mod pitchenv;
mod sysex;
mod tables;
mod user;
pub mod watch;

/// The 32 algorithms' routing bytes, the ones the engine renders with —
/// for a picture of an algorithm that cannot disagree with its sound.
pub use dx7note::{tremolo_share, vibrato_semitones};
pub use fm_core::ALGORITHMS;

pub use patch::{
    init_voice, Bank, BankError, BankInfo, BANKS, BANK_SIZE, NUM_BANKS, NUM_LIBRARY_VOICES,
    PACKED_VOICE_SIZE, VOICE_SIZE,
};
pub use sysex::{checksum, read_sysex, voice_name, Remark, SysexError, Voices};
pub use user::{add_voice, user_voice, PRESET_LAST, PRESET_USER, USER_VOICES};

/// The Dexed revision whose `Source/msfa` engine this crate is a port of.
/// The manual's License page cites it (test-guarded, like the papurs pin).
pub const PORTED_FROM_DEXED_REV: &str = "2e182b3db85c09083ab13c8b9b00565ce7d9ff85";
/// The music-synthesizer-for-android revision the packed-voice unpacker
/// (`patch.rs`) is ported from.
pub const PORTED_FROM_MSFA_REV: &str = "f67d41d313b7dc85f6fb99e79e515cc9d208cfff";

use bloop_events::NoteEvent;
use bloop_plugin_api::{
    ParamInfo, Plugin, PluginCategory, PluginInfo, ProcessContext, VoiceWatch, Watch, WATCH_VOICES,
};
use dx7note::{Controllers, Dx7Note};
use fm_core::{FmCore, N};
use lfo::Lfo;
use tables::Tables;

/// Polyphony. 8 voices covers pads comfortably; bump when a use case needs
/// more (roadmap).
pub const NUM_VOICES: usize = 8;

/// The DX7 function's MOD WHEEL RANGE, as Dexed's wheel `FmMod` holds it
/// (0–99 on the panel, a percentage of the wheel), with the wheel assigned
/// to PITCH only — not AMP, not EG BIAS. Fixed, as the function settings
/// are the instrument's and not the patch's, and bloop has no page for
/// them. Half: the patch's pitch sensitivity then sets the depth, from
/// none (PMS 0) to about ±1.3 semitones at PMS 4 with the wheel all the
/// way up.
pub const MOD_WHEEL_RANGE: i32 = 50;

pub const PARAM_VOLUME: usize = 0;
pub const PARAM_PRESET: usize = 1;
/// Params `PARAM_VOICE_BASE + o` map 1:1 onto unpacked-voice bytes
/// `o` in `0..NUM_VOICE_PARAMS` — six 21-byte operator blocks (OP6 first,
/// per the DX7 layout), then pitch EG, algorithm, feedback, key sync, LFO,
/// pitch-mod sensitivity, transpose. The name bytes (145-154) are not
/// params.
pub const PARAM_VOICE_BASE: usize = 2;
pub const NUM_VOICE_PARAMS: usize = 145;
/// Six operator on/off switches: `PARAM_OP_ENABLE_BASE + slot` toggles bit
/// `slot` of patch byte 155. Slot order follows the patch (0 = OP6 … 5 =
/// OP1). Applied at note-on, like every other voice edit.
pub const PARAM_OP_ENABLE_BASE: usize = PARAM_VOICE_BASE + NUM_VOICE_PARAMS;
pub const PARAM_COUNT: usize = PARAM_OP_ENABLE_BASE + 6;

/// Library size: every voice of the built-in cartridges plus the Init
/// voice. Presets are bank-major: preset = bank × 32 + voice, in
/// [`BANKS`] order (the last bank is partial), with Init last.
pub const PRESET_COUNT: usize = NUM_LIBRARY_VOICES + 1;
/// The Init preset (all params at their defaults — the DX7 INIT VOICE).
pub const PRESET_INIT: usize = NUM_LIBRARY_VOICES;

/// The unpacked voice a preset index selects: a built-in one, Init, or
/// past Init a voice the user imported ([`add_voice`]); a number that is
/// none of them is Init. The single library lookup behind both preset
/// loads and the UI knob-mirror seed. Allocation-free and lock-free.
pub fn preset_voice(preset: usize) -> [u8; VOICE_SIZE] {
    if preset < PRESET_INIT {
        Bank::builtin(preset / BANK_SIZE).voice(preset % BANK_SIZE)
    } else {
        user_voice(preset).unwrap_or_else(patch::init_voice)
    }
}

/// Voice names in the built-in cartridges that name another maker's
/// instrument, and the name the library gives each instead. The cartridge
/// bytes are left as they were: a voice's name plays no part in its sound.
pub const RENAMED_VOICES: [(&str, &str); 3] = [
    ("Moog Brass", "Ana Brass"),
    ("PPG PadFX", "Wave PadFX"),
    ("DMX TOMS 1", "DRUM TOMS 1"),
];

/// A preset's library name: bank short tag + voice name ("1A BRASS   1"),
/// an imported voice's own name, or "Init" for the blank-slate preset (and
/// anything out of range). A voice in [`RENAMED_VOICES`] goes by its new
/// name.
pub fn preset_name(preset: usize) -> String {
    if preset >= PRESET_INIT {
        user_voice(preset).map_or_else(|| "Init".to_string(), |voice| voice_name(&voice))
    } else {
        let bank = preset / BANK_SIZE;
        let voice = Bank::builtin(bank).name(preset % BANK_SIZE);
        let voice = RENAMED_VOICES
            .iter()
            .find(|(old, _)| *old == voice)
            .map_or(voice.as_str(), |(_, new)| new);
        format!("{} {voice}", BANKS[bank].short)
    }
}

/// Per-operator param names, `OP_PARAM_NAMES[slot][field]` with slot 0 =
/// OP6 (patch order) and field = byte offset within the op block.
macro_rules! op_names {
    ($n:literal) => {
        [
            concat!("OP", $n, " EG R1"),
            concat!("OP", $n, " EG R2"),
            concat!("OP", $n, " EG R3"),
            concat!("OP", $n, " EG R4"),
            concat!("OP", $n, " EG L1"),
            concat!("OP", $n, " EG L2"),
            concat!("OP", $n, " EG L3"),
            concat!("OP", $n, " EG L4"),
            concat!("OP", $n, " Brk Pt"),
            concat!("OP", $n, " L Depth"),
            concat!("OP", $n, " R Depth"),
            concat!("OP", $n, " L Curve"),
            concat!("OP", $n, " R Curve"),
            concat!("OP", $n, " Rate Scl"),
            concat!("OP", $n, " AM Sens"),
            concat!("OP", $n, " Velocity"),
            concat!("OP", $n, " Level"),
            concat!("OP", $n, " Mode"),
            concat!("OP", $n, " Coarse"),
            concat!("OP", $n, " Fine"),
            concat!("OP", $n, " Detune"),
        ]
    };
}
static OP_PARAM_NAMES: [[&str; 21]; 6] = [
    op_names!(6),
    op_names!(5),
    op_names!(4),
    op_names!(3),
    op_names!(2),
    op_names!(1),
];

/// Maximum raw value of each per-operator byte (field = offset % 21).
/// Everything is 0-based from 0; see the DX7 voice parameter chart.
const OP_FIELD_MAX: [u8; 21] = [
    99, 99, 99, 99, // EG rates
    99, 99, 99, 99, // EG levels
    99, // breakpoint (A-1..C8)
    99, 99, // left/right scale depth
    3, 3,  // left/right curve (-Lin/-Exp/+Exp/+Lin)
    7,  // rate scaling
    3,  // amp mod sensitivity
    7,  // key velocity sensitivity
    99, // output level
    1,  // osc mode (ratio/fixed)
    31, // frequency coarse
    99, // frequency fine
    14, // detune (7 = center)
];

/// Names and maxima of the global voice bytes (offsets 126-144).
static GLOBAL_PARAMS: [(&str, u8); 19] = [
    ("Pitch EG R1", 99),
    ("Pitch EG R2", 99),
    ("Pitch EG R3", 99),
    ("Pitch EG R4", 99),
    ("Pitch EG L1", 99),
    ("Pitch EG L2", 99),
    ("Pitch EG L3", 99),
    ("Pitch EG L4", 99),
    ("Algorithm", 31),
    ("Feedback", 7),
    ("Osc Sync", 1),
    ("LFO Speed", 99),
    ("LFO Delay", 99),
    ("LFO PMD", 99),
    ("LFO AMD", 99),
    ("LFO Sync", 1),
    ("LFO Wave", 5),
    ("P Mod Sens", 7),
    ("Transpose", 48),
];

/// `(name, max)` of the voice byte at `offset` (0..NUM_VOICE_PARAMS).
fn voice_param_meta(offset: usize) -> (&'static str, u8) {
    if offset < 126 {
        let (slot, field) = (offset / 21, offset % 21);
        (OP_PARAM_NAMES[slot][field], OP_FIELD_MAX[field])
    } else {
        GLOBAL_PARAMS[offset - 126]
    }
}

/// Host-visible metadata of every param, as a free function so UI tables
/// can share the single source of truth (`DexedPlugin::param_info`
/// delegates here). Unknown indices return the "?" placeholder.
pub fn param_info(index: usize) -> ParamInfo {
    match index {
        PARAM_VOLUME => ParamInfo {
            name: "Volume",
            default: 1.0,
            min: 0.0,
            max: 1.0,
        },
        PARAM_PRESET => ParamInfo {
            name: "Preset",
            default: 0.0,
            min: 0.0,
            max: (PRESET_COUNT - 1) as f32,
        },
        PARAM_VOICE_BASE..PARAM_OP_ENABLE_BASE => {
            let offset = index - PARAM_VOICE_BASE;
            let (name, max) = voice_param_meta(offset);
            ParamInfo {
                name,
                default: f32::from(patch::init_voice()[offset]),
                min: 0.0,
                max: f32::from(max),
            }
        }
        PARAM_OP_ENABLE_BASE..PARAM_COUNT => {
            let slot = index - PARAM_OP_ENABLE_BASE;
            static OP_ENABLE_NAMES: [&str; 6] =
                ["OP6 On", "OP5 On", "OP4 On", "OP3 On", "OP2 On", "OP1 On"];
            ParamInfo {
                name: OP_ENABLE_NAMES[slot],
                default: 1.0,
                min: 0.0,
                max: 1.0,
            }
        }
        _ => ParamInfo {
            name: "?",
            default: 0.0,
            min: 0.0,
            max: 1.0,
        },
    }
}

/// Normalized (0.0-1.0) knob positions a preset puts every param in:
/// the voice bytes over their ranges, volume at its default, all six
/// operators on. Seeds/mirrors the UI knob display when a preset loads
/// (the FM counterpart of `bloop_plugin_papurs::preset_positions`).
pub fn preset_positions(preset: usize) -> [f32; PARAM_COUNT] {
    let preset = preset.min(PRESET_LAST);
    let voice = preset_voice(preset);
    let mut positions = [0.0f32; PARAM_COUNT];
    positions[PARAM_VOLUME] = 1.0;
    positions[PARAM_PRESET] = (preset as f32 / (PRESET_COUNT - 1) as f32).min(1.0);
    for offset in 0..NUM_VOICE_PARAMS {
        let (_, max) = voice_param_meta(offset);
        positions[PARAM_VOICE_BASE + offset] = f32::from(voice[offset]) / f32::from(max);
    }
    for slot in 0..6 {
        positions[PARAM_OP_ENABLE_BASE + slot] = f32::from(voice[155] >> slot & 1);
    }
    positions
}

#[derive(Clone, Copy, Default)]
struct Voice {
    note: Dx7Note,
    midinote: u8,
    /// The key it sounds: `midinote` transposed.
    pitch: u8,
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
    /// The active unpacked patch — the live value store for every voice
    /// param (edits land here; presets overwrite it wholesale).
    patch: [u8; VOICE_SIZE],
    /// Host-side params that live outside the patch: volume and the
    /// last-loaded preset index.
    volume: f32,
    preset: usize,
    /// Rendered samples carried over when the host block isn't a multiple
    /// of the engine chunk size `N`.
    extra: [f32; N],
    extra_len: usize,
    extra_pos: usize,
    /// Dexed's output DC filter.
    dc: DcBlock,
}

/// The "very basic DC filter" Dexed runs its output through
/// (`PluginFx::process`): a one-pole high-pass with its pole at
/// `1 - 126 / rate`, a corner of about 20 Hz. The DX7's output stage is
/// AC-coupled; MSFA's voices are not, and a patch whose carrier is phase
/// modulated by an operator at its own ratio comes out well off zero.
/// Its arithmetic is Dexed's, in `f32`.
#[derive(Clone, Copy)]
struct DcBlock {
    pole: f32,
    last_in: f32,
    last_out: f32,
}

impl DcBlock {
    fn new(sample_rate: f64) -> DcBlock {
        DcBlock {
            pole: (1.0 - 126.0 / sample_rate) as f32,
            last_in: 0.0,
            last_out: 0.0,
        }
    }

    fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let input = *sample;
            *sample = input - self.last_in + self.pole * self.last_out;
            self.last_in = input;
            self.last_out = *sample;
        }
        // Silence decays the feedback towards zero through the subnormals,
        // which x86 does slowly; Dexed lets it. Far below anything a
        // 24-bit output resolves.
        if self.last_out.abs() < 1e-20 {
            self.last_out = 0.0;
        }
    }
}

impl Default for DexedPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DexedPlugin {
    pub fn new() -> Self {
        let mut plugin = Self {
            tables: Box::new(Tables::new(48000.0)),
            sample_rate: 48000.0,
            core: FmCore::default(),
            lfo: Lfo::default(),
            ctrls: Controllers::default(),
            voices: [Voice::default(); NUM_VOICES],
            serial: 0,
            patch: preset_voice(0),
            volume: 1.0,
            preset: 0,
            extra: [0.0; N],
            extra_len: 0,
            extra_pos: 0,
            dc: DcBlock::new(48000.0),
        };
        plugin.load_preset(0);
        plugin
    }

    fn load_preset(&mut self, index: usize) {
        self.patch = preset_voice(index);
        self.reset_lfo();
    }

    /// A program change. A note still held keeps sounding — each voice
    /// took its patch at note-on (`preset_switch_spares_playing_voices`) —
    /// but what was let go stops. A released voice of a patch whose
    /// carriers end above nought (L4 > 0: TRAIN) never stops by itself,
    /// and left running it sounds under every preset loaded after it.
    fn let_go_of_released(&mut self) {
        for voice in &mut self.voices {
            if !voice.keydown {
                *voice = Voice::default();
            }
        }
    }

    /// Re-seed the LFO from the patch's LFO bytes (speed, delay, PMD,
    /// AMD, sync, wave).
    fn reset_lfo(&mut self) {
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
        voice.pitch = u8::try_from(note).unwrap_or(midinote);
        voice.keydown = true;
        voice.serial = self.serial;
        self.lfo.keydown();
    }

    /// Both wheels back at rest: the pitch wheel in the middle, the mod
    /// wheel down. For a synth handed out again from a pool, which the
    /// host takes to be at rest.
    pub fn wheels_at_rest(&mut self) {
        self.ctrls.pitch_bend = Controllers::default().pitch_bend;
        self.ctrls.pitch_mod = 0;
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
            // A choke asks for no release, but this synth honours note-off
            // and so was already stopping when the engine injected one.
            // Releasing keeps that behaviour rather than introducing a click
            // for a case nothing exercises; it has no hard-stop to reach for.
            NoteEvent::Choke { note, .. } => self.note_off(note.as_u8()),
            NoteEvent::PitchBend { value, .. } => {
                self.ctrls.pitch_bend = (value * 16383.0).round().clamp(0.0, 16383.0) as i32;
            }
            // The DX7's mod wheel, as Dexed's `Controllers::refresh` turns
            // it into a modulation total: the wheel's 0–127 times the
            // function's MOD WHEEL RANGE, assigned to pitch. The voice
            // multiplies it by the patch's own pitch sensitivity and the
            // LFO, so a patch with none does not wobble.
            NoteEvent::CC { cc, value, .. } if cc == bloop_events::CC_MOD_WHEEL => {
                let wheel = (value * 127.0).round().clamp(0.0, 127.0) as i32;
                self.ctrls.pitch_mod = wheel * MOD_WHEEL_RANGE / 100;
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
        let volume = self.volume;
        for (out, &v) in self.extra.iter_mut().zip(buf.iter()) {
            // Same headroom/clip staging as MSFA's synth_unit: >>4 then
            // clamp to ±2^24, mapped here to ±1.0 floats.
            let val = (v >> 4).clamp(-(1 << 24), 1 << 24);
            *out = val as f32 / 16_777_216.0; // ±2^24 → ±1.0
        }
        self.dc.process(&mut self.extra);
        for out in &mut self.extra {
            *out *= volume;
        }
        self.extra_len = N;
        self.extra_pos = 0;
    }
}

impl Plugin for DexedPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: "FM",
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
        match index {
            PARAM_VOLUME => ParamInfo {
                name: "Volume",
                default: 1.0,
                min: 0.0,
                max: 1.0,
            },
            PARAM_PRESET => ParamInfo {
                name: "Preset",
                default: 0.0,
                min: 0.0,
                max: (PRESET_COUNT - 1) as f32,
            },
            PARAM_VOICE_BASE..PARAM_OP_ENABLE_BASE => {
                let offset = index - PARAM_VOICE_BASE;
                let (name, max) = voice_param_meta(offset);
                ParamInfo {
                    name,
                    default: f32::from(patch::init_voice()[offset]),
                    min: 0.0,
                    max: f32::from(max),
                }
            }
            PARAM_OP_ENABLE_BASE..PARAM_COUNT => {
                let slot = index - PARAM_OP_ENABLE_BASE;
                static OP_ENABLE_NAMES: [&str; 6] =
                    ["OP6 On", "OP5 On", "OP4 On", "OP3 On", "OP2 On", "OP1 On"];
                ParamInfo {
                    name: OP_ENABLE_NAMES[slot],
                    default: 1.0,
                    min: 0.0,
                    max: 1.0,
                }
            }
            _ => ParamInfo {
                name: "?",
                default: 0.0,
                min: 0.0,
                max: 1.0,
            },
        }
    }

    fn set_param(&mut self, index: usize, value: f32) {
        match index {
            PARAM_VOLUME => self.volume = value.clamp(0.0, 1.0),
            PARAM_PRESET => {
                let preset = (value.round().max(0.0) as usize).min(PRESET_LAST);
                // The same preset said again reloads its patch, and is no
                // program change: what is sounding goes on.
                if preset != self.preset {
                    self.let_go_of_released();
                }
                self.preset = preset;
                self.load_preset(self.preset);
            }
            PARAM_VOICE_BASE..PARAM_OP_ENABLE_BASE => {
                let offset = index - PARAM_VOICE_BASE;
                let (_, max) = voice_param_meta(offset);
                self.patch[offset] = value.round().clamp(0.0, f32::from(max)) as u8;
                // The LFO runs from its own registers, not the patch:
                // re-seed it when an LFO byte changes. (Everything else is
                // read per note-on.)
                if (137..143).contains(&offset) {
                    self.reset_lfo();
                }
            }
            PARAM_OP_ENABLE_BASE..PARAM_COUNT => {
                let bit = 1u8 << (index - PARAM_OP_ENABLE_BASE);
                if value >= 0.5 {
                    self.patch[155] |= bit;
                } else {
                    self.patch[155] &= !bit;
                }
            }
            _ => {}
        }
    }

    fn get_param(&self, index: usize) -> f32 {
        match index {
            PARAM_VOLUME => self.volume,
            PARAM_PRESET => self.preset as f32,
            PARAM_VOICE_BASE..PARAM_OP_ENABLE_BASE => {
                f32::from(self.patch[index - PARAM_VOICE_BASE])
            }
            PARAM_OP_ENABLE_BASE..PARAM_COUNT => {
                f32::from(self.patch[155] >> (index - PARAM_OP_ENABLE_BASE) & 1)
            }
            _ => 0.0,
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
        self.dc = DcBlock::new(sample_rate);
        // The patch is the live state — do NOT reload the preset here, or
        // voice edits would vanish on every sample-rate change and synth
        // round-trip. Only the LFO re-seeds (its tables may have changed).
        self.reset_lfo();
    }

    /// Each voice as of the last chunk rendered. A voice is active on the
    /// same terms `render_chunk` renders it; its `note_on` is the serial
    /// that voice stealing already orders voices by.
    ///
    /// Levels are the DX7's 0..99, the scale of the patch's L1..L4. An
    /// amp of 1.0 is the gain of an operator at output level 99 and
    /// envelope level 99 at a velocity its sensitivity leaves alone
    /// (about 100); velocity 127 at sensitivity 7 adds 0.875 doublings, so
    /// the most an amp reads is about 1.83.
    fn watch(&self, out: &mut Watch) {
        for (slot, seen) in out.voices.iter_mut().enumerate() {
            *seen = match self.voices.get(slot) {
                Some(voice) if voice.keydown || voice.note.is_playing() => {
                    let mut seen = VoiceWatch {
                        active: true,
                        note_on: voice.serial,
                        key_down: voice.keydown,
                        pitch: voice.pitch,
                        ..VoiceWatch::default()
                    };
                    for op in 1..=6u8 {
                        // The patch stores operator 6 first.
                        voice
                            .note
                            .watch_op(usize::from(6 - op))
                            .write(&mut seen, op);
                    }
                    seen
                }
                _ => VoiceWatch::default(),
            };
        }
        out.globals = Default::default();
        out.knobs = Default::default();
    }
}

const _: () = assert!(
    NUM_VOICES <= WATCH_VOICES,
    "a Watch must have room for every voice"
);

fn event_time(event: &NoteEvent) -> u32 {
    match *event {
        NoteEvent::NoteOn { time, .. }
        | NoteEvent::NoteOff { time, .. }
        | NoteEvent::Aftertouch { time, .. }
        | NoteEvent::PitchBend { time, .. }
        | NoteEvent::CC { time, .. }
        | NoteEvent::Choke { time, .. } => time,
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
    /// -35 dB print as "-" (the margin keeps borderline leakage from
    /// flipping the snapshot across platforms).
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
                if db < -35.0 {
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
        Ev::Note(NoteEvent::PitchBend {
            time: 0,
            channel: 0,
            value,
        })
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
        expect.assert_eq(&format!(
            "dominant={}Hz",
            dominant_freq(sustained, f64::from(SR))
        ));
    }

    const EPIANO: f32 = 10.0; // ROM1A "E.PIANO 1"
    const EORGAN: f32 = 16.0; // ROM1A "E.ORGAN 1" (sustained, near-organ tone)
    const MARIMBA: f32 = 21.0; // ROM1A "MARIMBA" (fast natural decay)
    const BRASS: f32 = 0.0; // ROM1A "BRASS 1" (slow attack, sustains)

    /// TRAIN is one of the DX7 patches whose carriers end on a level above
    /// nought (L4 > 0), so a note of it sustains after its key comes up —
    /// the hardware's own behaviour. Loading another preset then left that
    /// voice running on the new patch's numbers, and the two were heard
    /// together. A program change silences what was let go.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_preset_change_silences_what_was_sounding() {
        const TRAIN: f32 = 30.0;
        check(
            TRAIN,
            &[
                (0, note_on(60, 0.8)),
                (8, note_off(60)),
                (40, Ev::Param(PARAM_PRESET, MARIMBA)),
            ],
            64,
            expect![[r#"
                w 0 peak=0.0695
                w 1 peak=0.0683
                w 2 peak=0.0364
                w 3 peak=0.0323
                w 4 peak=0.0342
                w 5 peak=0.0000
                w 6 peak=0.0000
                w 7 peak=0.0000"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn renamed_voices_are_in_the_cartridges() {
        // Every voice the library renames is found under its cartridge
        // name, so a rename cannot silently miss: the preset, the name the
        // cartridge gives it and the name the library shows.
        let text = (0..PRESET_INIT)
            .filter_map(|p| {
                let raw = Bank::builtin(p / BANK_SIZE).name(p % BANK_SIZE);
                RENAMED_VOICES
                    .iter()
                    .any(|(old, _)| *old == raw)
                    .then(|| format!("{p:>3}: {raw:<10} -> {}", preset_name(p)))
            })
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
            319: Moog Brass -> FT2 Ana Brass
            337: PPG PadFX  -> FT3 Wave PadFX
            355: DMX TOMS 1 -> DRM DRUM TOMS 1"#]]
        .assert_eq(&text);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn preset_library_boundaries() {
        // The preset index is bank * 32 + voice with Init last: 31→32
        // crosses ROM1A→ROM1B, 256 starts Finetales 1, 352 starts the
        // partial drum bank (7 voices, 352-358), 359 is Init, and
        // out-of-range clamps to Init. Names and algorithms come through
        // `preset_voice`, the same lookup `load_preset` plays.
        let text = [
            0usize, 10, 31, 32, 63, 64, 255, 256, 320, 351, 352, 358, 359, PRESET_LAST + 1,
        ]
        .iter()
        .map(|&p| {
            let v = preset_voice(p);
            format!("{p:>3}: {:<14} alg={:>2}", preset_name(p), v[134] + 1)
        })
        .collect::<Vec<_>>()
        .join("\n");
        expect![[r#"
            presets=360 init=359
              0: 1A BRASS   1   alg=22
             10: 1A E.PIANO 1   alg= 5
             31: 1A TAKE OFF    alg=10
             32: 1B PIANO   4   alg=18
             63: 1B BASS    4   alg=17
             64: 2A PICCOLO     alg= 5
            255: 4B EXPLOSION   alg=16
            256: FT1 EP 1       alg= 5
            320: FT3 Airblower  alg=16
            351: FT3 BakketDrum alg= 9
            352: DRM KICK DRUM  alg=16
            358: DRM Hi-Hat 1   alg=11
            359: Init           alg= 1
            4456: Init           alg= 1"#]]
        .assert_eq(&format!(
            "presets={PRESET_COUNT} init={PRESET_INIT}\n{text}"
        ));
    }

    /// The whole library imported as the user's own: every voice of every
    /// cartridge bloop ships, read from its file at run time
    /// ([`read_sysex`]) and played under the number [`add_voice`] gives
    /// it, sounds as the built-in preset does, sample for sample, and
    /// leaves every parameter where the built-in one does.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_imported_voice_plays_as_the_built_in_one() {
        let mut differ = Vec::new();
        let mut checked = 0;
        for (b, bank) in BANKS.iter().enumerate() {
            let read = read_sysex(bank.data).map(|r| r.voices).unwrap_or_default();
            for (v, voice) in read.iter().enumerate().take(bank.voices) {
                let built_in = b * BANK_SIZE + v;
                let Some(imported) = add_voice(voice) else {
                    differ.push(format!("{} {v}: no number", bank.name));
                    continue;
                };
                let play = |preset: usize| {
                    let mut synth = DexedPlugin::new();
                    synth.reset(SR, 1024);
                    synth.set_param(PARAM_PRESET, preset as f32);
                    let params: Vec<f32> = (PARAM_VOICE_BASE..PARAM_COUNT)
                        .map(|p| synth.get_param(p))
                        .collect();
                    let audio = render_blocks(
                        &mut synth,
                        12,
                        &[(0, note_on(60, 0.8)), (6, note_off(60))],
                    );
                    (params, audio)
                };
                checked += 1;
                if play(built_in) != play(imported) {
                    differ.push(format!("{} {v}", bank.name));
                }
            }
        }
        expect!["359 voices, differ: []"].assert_eq(&format!("{checked} voices, differ: {differ:?}"));
    }

    /// A damaged file's voices play without taking the audio thread
    /// down: every data byte at its highest (0x7F) and every byte at
    /// nought, as a cartridge and as a single voice, each played on every
    /// algorithm the masked byte can name.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_damaged_voice_plays() {
        let mut out = Vec::new();
        for fill in [0x7fu8, 0x00] {
            let mut single = vec![0xf0, 0x43, 0x00, 0x00, 0x01, 0x1b];
            single.extend(std::iter::repeat_n(fill, VOICE_SIZE - 1));
            single.extend([0x00, 0xf7]);
            let cartridge = vec![fill; PACKED_VOICE_SIZE * BANK_SIZE];
            for (what, bytes) in [("single", single), ("cartridge", cartridge)] {
                let voices = read_sysex(&bytes).map(|r| r.voices).unwrap_or_default();
                let Some(voice) = voices.first() else {
                    out.push(format!("{fill:#04x} {what}: no voice"));
                    continue;
                };
                let mut peak = 0.0f32;
                for algorithm in 0..32u8 {
                    let mut voice = *voice;
                    voice[134] = algorithm;
                    let Some(preset) = add_voice(&voice) else {
                        continue;
                    };
                    let mut synth = DexedPlugin::new();
                    synth.reset(SR, 1024);
                    synth.set_param(PARAM_PRESET, preset as f32);
                    let audio = render_blocks(
                        &mut synth,
                        6,
                        &[(0, note_on(127, 1.0)), (1, note_on(0, 1.0)), (3, note_off(127))],
                    );
                    peak = audio.iter().fold(peak, |p, s| p.max(s.abs()));
                }
                out.push(format!("{fill:#04x} {what}: plays, finite {}", peak.is_finite()));
            }
        }
        expect![[r#"
            0x7f single: plays, finite true
            0x7f cartridge: plays, finite true
            0x00 single: plays, finite true
            0x00 cartridge: plays, finite true"#]].assert_eq(&out.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn library_stride_invariant() {
        // Preset math (bank = preset / 32) requires every bank except the
        // last to be full, and the totals to agree with the registry.
        assert!(BANKS[..NUM_BANKS - 1].iter().all(|b| b.voices == BANK_SIZE));
        assert_eq!(BANKS.iter().map(|b| b.voices).sum::<usize>(), PRESET_INIT);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn drum_bank_preset_renders() {
        // The partial drum bank's first voice (KICK DRUM, preset 352)
        // must sound: a fast percussive attack decaying on its own, gone
        // after the note-off at block 48.
        check(
            352.0,
            &[(0, note_on(60, 1.0)), (48, note_off(60))],
            64,
            expect![[r#"
                w 0 peak=0.1616
                w 1 peak=0.0797
                w 2 peak=0.0615
                w 3 peak=0.0493
                w 4 peak=0.0302
                w 5 peak=0.0220
                w 6 peak=0.0140
                w 7 peak=0.0046"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn cross_bank_preset_renders() {
        // A voice from a non-ROM1A cartridge (Finetales 1 "EP 1",
        // preset 256) must actually sound: the engine's preset load
        // indexes past the first bank. Envelope shape: keyed attack,
        // decay while held, release to silence after note-off.
        check(
            256.0,
            &[(0, note_on(69, 1.0)), (48, note_off(69))],
            64,
            expect![[r#"
                w 0 peak=0.3394
                w 1 peak=0.3204
                w 2 peak=0.3483
                w 3 peak=0.3680
                w 4 peak=0.3696
                w 5 peak=0.3603
                w 6 peak=0.3102
                w 7 peak=0.1395"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn epiano_gate() {
        // E.PIANO 1, one A4: fast attack, decaying sustain while held,
        // release to silence after note-off at block 48.
        check(
            EPIANO,
            &[(0, note_on(69, 1.0)), (48, note_off(69))],
            96,
            expect![[r#"
                w 0 peak=0.3394
                w 1 peak=0.3204
                w 2 peak=0.3483
                w 3 peak=0.3680
                w 4 peak=0.3696
                w 5 peak=0.3603
                w 6 peak=0.3102
                w 7 peak=0.1395
                w 8 peak=0.0937
                w 9 peak=0.0471
                w10 peak=0.0235
                w11 peak=0.0100"#]],
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
            expect![[r#"
                w 0 peak=0.5393
                w 1 peak=0.5473
                w 2 peak=0.4980
                w 3 peak=0.4912
                w 4 peak=0.4992
                w 5 peak=0.5176
                w 6 peak=0.5181
                w 7 peak=0.4690
                w 8 peak=0.0873
                w 9 peak=0.0214"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn marimba_decays_naturally() {
        // MARIMBA decays to silence on its own even while the key is held
        // (L4 = 0 percussive envelope).
        check(
            MARIMBA,
            &[(0, note_on(60, 1.0))],
            96,
            expect![[r#"
                w 0 peak=0.2246
                w 1 peak=0.2063
                w 2 peak=0.1966
                w 3 peak=0.1849
                w 4 peak=0.1702
                w 5 peak=0.1622
                w 6 peak=0.1437
                w 7 peak=0.1282
                w 8 peak=0.1170
                w 9 peak=0.1043
                w10 peak=0.0981
                w11 peak=0.0838"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn organ_pitch_tracks_notes() {
        // The dominant line must track the played note. E.ORGAN 1's
        // strongest component is its 16' foundation — a coarse-0.5
        // operator one octave BELOW the key (the pure-sine algorithm test
        // pins absolute tuning; this pins tracking): A4 → 220, A3 → 110.
        check_pitch(EORGAN, &[(0, note_on(69, 1.0))], expect!["dominant=220Hz"]);
        check_pitch(EORGAN, &[(0, note_on(57, 1.0))], expect!["dominant=110Hz"]);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn pitch_bend_full_up_is_two_semitones() {
        // Bend fully up (default range 2 semitones): A4 → B4 ≈ 494 Hz.
        check_pitch(
            EORGAN,
            &[(0, note_on(69, 1.0)), (0, bend(1.0))],
            expect!["dominant=247Hz"],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn velocity_scales_level() {
        // Same note at descending velocities, one snapshot: the overall
        // peak must fall monotonically (DX7 velocity also softens the
        // modulators, so it's a timbre change, not a linear gain).
        let lines: Vec<String> = [1.0f32, 0.7, 0.4, 0.1]
            .iter()
            .map(|&vel| {
                let mut synth = DexedPlugin::new();
                synth.reset(SR, 1024);
                synth.set_param(PARAM_PRESET, EPIANO);
                let samples = render_blocks(&mut synth, 32, &[(0, note_on(69, vel))]);
                let peak = samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
                format!("vel {vel:.1}: peak={peak:.4}")
            })
            .collect();
        expect![[r#"
            vel 1.0: peak=0.3680
            vel 0.7: peak=0.2596
            vel 0.4: peak=0.2579
            vel 0.1: peak=0.1972"#]]
        .assert_eq(&lines.join("\n"));
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
            &[
                (0, note_on(60, 1.0)),
                (0, note_on(64, 1.0)),
                (0, note_on(67, 1.0)),
            ],
        );
        let sustained = &samples[16 * BLOCK..];
        let mags: Vec<String> = [261.63, 329.63, 392.0, 466.16]
            .iter()
            .map(|&hz| {
                let m = goertzel(sustained, f64::from(SR), hz);
                format!("{hz}Hz {}", if m > 0.01 { "sounding" } else { "-" })
            })
            .collect();
        expect![[r#"
            261.63Hz sounding
            329.63Hz sounding
            392Hz sounding
            466.16Hz -"#]]
        .assert_eq(&mags.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn voice_stealing_takes_the_oldest() {
        // 10 sustained notes on 8 voices: the two OLDEST (the first two
        // played) must be the ones stolen. Check per-pitch energy in the
        // sustained window, then release everything — it must fade out
        // (no stuck voices).
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, EORGAN);
        // A chromatic cluster: no note's fundamental coincides with any
        // harmonic (or the organ's sub-octave) of another, so "stolen"
        // can't be masked by a surviving note's overtones.
        let notes: Vec<u8> = (0..10).map(|i| 60 + i).collect();
        let mut events: Vec<(usize, Ev)> = notes
            .iter()
            .enumerate()
            .map(|(i, &n)| (i * 2, note_on(n, 1.0)))
            .collect();
        events.extend(notes.iter().map(|&n| (60usize, note_off(n))));
        let samples = render_blocks(&mut synth, 120, &events);
        let sustained = &samples[32 * BLOCK..56 * BLOCK];
        let mut lines: Vec<String> = notes
            .iter()
            .map(|&n| {
                let hz = 440.0 * 2f64.powf((f64::from(n) - 69.0) / 12.0);
                let mag = goertzel(sustained, f64::from(SR), hz);
                format!(
                    "note {n}: {}",
                    if mag > 0.005 {
                        "sounding"
                    } else {
                        "stolen/quiet"
                    }
                )
            })
            .collect();
        let tail = &samples[112 * BLOCK..];
        let tail_peak = tail.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        lines.push(format!("tail peak after release: {tail_peak:.4}"));
        expect![[r#"
            note 60: sounding
            note 61: sounding
            note 62: sounding
            note 63: sounding
            note 64: sounding
            note 65: sounding
            note 66: sounding
            note 67: sounding
            note 68: sounding
            note 69: sounding
            tail peak after release: 0.0000"#]]
        .assert_eq(&lines.join("\n"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn preset_switch_spares_playing_voices() {
        // A held BRASS note must keep sustaining across a preset switch to
        // percussive MARIMBA (each voice copies its patch at note-on); the
        // note played AFTER the switch decays on its own. The peak trace
        // shows: sustained plateau throughout the hold, then the second
        // note's fast marimba decay.
        check(
            BRASS,
            &[
                (0, note_on(60, 1.0)),
                (32, Ev::Param(PARAM_PRESET, MARIMBA)),
                (56, note_off(60)),
                (64, note_on(60, 1.0)),
            ],
            136,
            expect![[r#"
                w 0 peak=0.5393
                w 1 peak=0.5473
                w 2 peak=0.4980
                w 3 peak=0.4912
                w 4 peak=0.5191
                w 5 peak=0.5200
                w 6 peak=0.5184
                w 7 peak=0.3596
                w 8 peak=0.2592
                w 9 peak=0.2121
                w10 peak=0.1978
                w11 peak=0.1843
                w12 peak=0.1702
                w13 peak=0.1622
                w14 peak=0.1437
                w15 peak=0.1282
                w16 peak=0.1170"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn events_mid_block_start_on_the_next_chunk() {
        // The engine advances in 64-sample chunks: an event at sample 70
        // of a 128-sample block takes effect at the next chunk boundary —
        // sample 128 (worst-case ~1.3ms quantization at 48kHz).
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, EORGAN);
        let mut buf = EventBuffer::new();
        buf.push(NoteEvent::NoteOn {
            time: 70,
            channel: 0,
            note: bloop_core::Note::new(69),
            velocity: 1.0,
        });
        let mut left = vec![0.0f32; 2 * BLOCK];
        let mut right = vec![0.0f32; 2 * BLOCK];
        let (l2, r2) = (left.split_at_mut(BLOCK), right.split_at_mut(BLOCK));
        let empty = EventBuffer::new();
        for (i, (l, r)) in [(l2.0, r2.0), (l2.1, r2.1)].into_iter().enumerate() {
            let mut outputs: Vec<&mut [f32]> = vec![l, r];
            let mut ctx = ProcessContext {
                outputs: &mut outputs,
                input_events: if i == 0 { &buf } else { &empty },
                sample_rate: SR,
                tempo: 120.0,
                is_playing: false,
            };
            synth.process(&mut ctx);
        }
        let onset = left.iter().position(|&s| s.abs() > 1e-4);
        expect!["first_nonzero=Some(129)"].assert_eq(&format!("first_nonzero={onset:?}"));
    }

    /// A note held for a second and let go, in quarter-second windows:
    /// its offset from zero (`dc`) beside what it plays (`ac`, the RMS
    /// around that offset).
    fn check_dc(preset: usize, expect: Expect) {
        let mut synth = DexedPlugin::new();
        synth.set_param(PARAM_PRESET, preset as f32);
        let quarter = (SR as usize / 4) / BLOCK;
        let out = render_blocks(
            &mut synth,
            quarter * 8,
            &[(0, note_on(60, 1.0)), (quarter * 4, note_off(60))],
        );
        let lines: Vec<String> = out
            .chunks(quarter * BLOCK)
            .enumerate()
            .map(|(i, window)| {
                let n = window.len() as f64;
                let dc = window.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
                let ac = (window
                    .iter()
                    .map(|&x| (f64::from(x) - dc).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt();
                format!(
                    "{:.2}-{:.2}s  dc {dc:+.4}  ac {ac:.4}",
                    i as f64 / 4.0,
                    (i + 1) as f64 / 4.0
                )
            })
            .collect();
        expect.assert_eq(&format!("{}\n{}", preset_name(preset), lines.join("\n")));
    }

    /// Dexed runs its output through a DC filter (`PluginFx::process`,
    /// a one-pole high-pass at about 20 Hz), as the DX7's own output stage
    /// is AC-coupled. Some patches need it: SYN-LEAD 1, the kit's Lead,
    /// has OP2 and OP3 at OP1's ratio, so OP1's phase modulation is
    /// lopsided and its output sits well off zero for as long as the
    /// note is held — a step at every note-on and note-off.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_held_note_sits_on_zero() {
        check_dc(
            13,
            expect![[r#"
            1A SYN-LEAD 1
            0.00-0.25s  dc -0.0013  ac 0.0665
            0.25-0.50s  dc -0.0006  ac 0.0793
            0.50-0.75s  dc +0.0005  ac 0.0814
            0.75-1.00s  dc +0.0011  ac 0.0731
            1.00-1.25s  dc +0.0003  ac 0.0108
            1.25-1.50s  dc -0.0000  ac 0.0000
            1.50-1.75s  dc +0.0000  ac 0.0000
            1.75-2.00s  dc +0.0000  ac 0.0000"#]],
        );
        check_dc(
            1,
            expect![[r#"
            1A BRASS   2
            0.00-0.25s  dc +0.0009  ac 0.1399
            0.25-0.50s  dc +0.0029  ac 0.1571
            0.50-0.75s  dc -0.0001  ac 0.1585
            0.75-1.00s  dc -0.0010  ac 0.1151
            1.00-1.25s  dc -0.0027  ac 0.0168
            1.25-1.50s  dc +0.0000  ac 0.0000
            1.50-1.75s  dc +0.0000  ac 0.0000
            1.75-2.00s  dc +0.0000  ac 0.0000"#]],
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn volume_param_scales_output() {
        check(
            EORGAN,
            &[(0, Ev::Param(PARAM_VOLUME, 0.25)), (0, note_on(69, 1.0))],
            32,
            expect![[r#"
                w 0 peak=0.0727
                w 1 peak=0.0730
                w 2 peak=0.0726
                w 3 peak=0.0707"#]],
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
        let b = render(&[96, 32, 128, 1, 63, 64, 128, 128, 128]);
        assert_eq!(a.len(), b.len());
        let identical = a == b;
        expect!["identical=true"].assert_eq(&format!("identical={identical}"));
    }

    /// The full parameter surface, one line per param: index, name, range,
    /// default. This snapshot IS the spec of the host-visible DX7 param
    /// map (voice bytes 0-144 behind PARAM_VOICE_BASE, then the six
    /// operator switches) — review any change against the DX7 voice layout.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn param_surface() {
        let synth = DexedPlugin::new();
        let lines: Vec<String> = (0..synth.param_count())
            .map(|p| {
                let i = synth.param_info(p);
                format!(
                    "{p:>3} {:<12} {:>3}..{:<3} default {}",
                    i.name, i.min, i.max, i.default
                )
            })
            .collect();
        expect![[r#"
              0 Volume         0..1   default 1
              1 Preset         0..359 default 0
              2 OP6 EG R1      0..99  default 99
              3 OP6 EG R2      0..99  default 99
              4 OP6 EG R3      0..99  default 99
              5 OP6 EG R4      0..99  default 99
              6 OP6 EG L1      0..99  default 99
              7 OP6 EG L2      0..99  default 99
              8 OP6 EG L3      0..99  default 99
              9 OP6 EG L4      0..99  default 0
             10 OP6 Brk Pt     0..99  default 39
             11 OP6 L Depth    0..99  default 0
             12 OP6 R Depth    0..99  default 0
             13 OP6 L Curve    0..3   default 0
             14 OP6 R Curve    0..3   default 0
             15 OP6 Rate Scl   0..7   default 0
             16 OP6 AM Sens    0..3   default 0
             17 OP6 Velocity   0..7   default 0
             18 OP6 Level      0..99  default 0
             19 OP6 Mode       0..1   default 0
             20 OP6 Coarse     0..31  default 1
             21 OP6 Fine       0..99  default 0
             22 OP6 Detune     0..14  default 7
             23 OP5 EG R1      0..99  default 99
             24 OP5 EG R2      0..99  default 99
             25 OP5 EG R3      0..99  default 99
             26 OP5 EG R4      0..99  default 99
             27 OP5 EG L1      0..99  default 99
             28 OP5 EG L2      0..99  default 99
             29 OP5 EG L3      0..99  default 99
             30 OP5 EG L4      0..99  default 0
             31 OP5 Brk Pt     0..99  default 39
             32 OP5 L Depth    0..99  default 0
             33 OP5 R Depth    0..99  default 0
             34 OP5 L Curve    0..3   default 0
             35 OP5 R Curve    0..3   default 0
             36 OP5 Rate Scl   0..7   default 0
             37 OP5 AM Sens    0..3   default 0
             38 OP5 Velocity   0..7   default 0
             39 OP5 Level      0..99  default 0
             40 OP5 Mode       0..1   default 0
             41 OP5 Coarse     0..31  default 1
             42 OP5 Fine       0..99  default 0
             43 OP5 Detune     0..14  default 7
             44 OP4 EG R1      0..99  default 99
             45 OP4 EG R2      0..99  default 99
             46 OP4 EG R3      0..99  default 99
             47 OP4 EG R4      0..99  default 99
             48 OP4 EG L1      0..99  default 99
             49 OP4 EG L2      0..99  default 99
             50 OP4 EG L3      0..99  default 99
             51 OP4 EG L4      0..99  default 0
             52 OP4 Brk Pt     0..99  default 39
             53 OP4 L Depth    0..99  default 0
             54 OP4 R Depth    0..99  default 0
             55 OP4 L Curve    0..3   default 0
             56 OP4 R Curve    0..3   default 0
             57 OP4 Rate Scl   0..7   default 0
             58 OP4 AM Sens    0..3   default 0
             59 OP4 Velocity   0..7   default 0
             60 OP4 Level      0..99  default 0
             61 OP4 Mode       0..1   default 0
             62 OP4 Coarse     0..31  default 1
             63 OP4 Fine       0..99  default 0
             64 OP4 Detune     0..14  default 7
             65 OP3 EG R1      0..99  default 99
             66 OP3 EG R2      0..99  default 99
             67 OP3 EG R3      0..99  default 99
             68 OP3 EG R4      0..99  default 99
             69 OP3 EG L1      0..99  default 99
             70 OP3 EG L2      0..99  default 99
             71 OP3 EG L3      0..99  default 99
             72 OP3 EG L4      0..99  default 0
             73 OP3 Brk Pt     0..99  default 39
             74 OP3 L Depth    0..99  default 0
             75 OP3 R Depth    0..99  default 0
             76 OP3 L Curve    0..3   default 0
             77 OP3 R Curve    0..3   default 0
             78 OP3 Rate Scl   0..7   default 0
             79 OP3 AM Sens    0..3   default 0
             80 OP3 Velocity   0..7   default 0
             81 OP3 Level      0..99  default 0
             82 OP3 Mode       0..1   default 0
             83 OP3 Coarse     0..31  default 1
             84 OP3 Fine       0..99  default 0
             85 OP3 Detune     0..14  default 7
             86 OP2 EG R1      0..99  default 99
             87 OP2 EG R2      0..99  default 99
             88 OP2 EG R3      0..99  default 99
             89 OP2 EG R4      0..99  default 99
             90 OP2 EG L1      0..99  default 99
             91 OP2 EG L2      0..99  default 99
             92 OP2 EG L3      0..99  default 99
             93 OP2 EG L4      0..99  default 0
             94 OP2 Brk Pt     0..99  default 39
             95 OP2 L Depth    0..99  default 0
             96 OP2 R Depth    0..99  default 0
             97 OP2 L Curve    0..3   default 0
             98 OP2 R Curve    0..3   default 0
             99 OP2 Rate Scl   0..7   default 0
            100 OP2 AM Sens    0..3   default 0
            101 OP2 Velocity   0..7   default 0
            102 OP2 Level      0..99  default 0
            103 OP2 Mode       0..1   default 0
            104 OP2 Coarse     0..31  default 1
            105 OP2 Fine       0..99  default 0
            106 OP2 Detune     0..14  default 7
            107 OP1 EG R1      0..99  default 99
            108 OP1 EG R2      0..99  default 99
            109 OP1 EG R3      0..99  default 99
            110 OP1 EG R4      0..99  default 99
            111 OP1 EG L1      0..99  default 99
            112 OP1 EG L2      0..99  default 99
            113 OP1 EG L3      0..99  default 99
            114 OP1 EG L4      0..99  default 0
            115 OP1 Brk Pt     0..99  default 39
            116 OP1 L Depth    0..99  default 0
            117 OP1 R Depth    0..99  default 0
            118 OP1 L Curve    0..3   default 0
            119 OP1 R Curve    0..3   default 0
            120 OP1 Rate Scl   0..7   default 0
            121 OP1 AM Sens    0..3   default 0
            122 OP1 Velocity   0..7   default 0
            123 OP1 Level      0..99  default 99
            124 OP1 Mode       0..1   default 0
            125 OP1 Coarse     0..31  default 1
            126 OP1 Fine       0..99  default 0
            127 OP1 Detune     0..14  default 7
            128 Pitch EG R1    0..99  default 99
            129 Pitch EG R2    0..99  default 99
            130 Pitch EG R3    0..99  default 99
            131 Pitch EG R4    0..99  default 99
            132 Pitch EG L1    0..99  default 50
            133 Pitch EG L2    0..99  default 50
            134 Pitch EG L3    0..99  default 50
            135 Pitch EG L4    0..99  default 50
            136 Algorithm      0..31  default 0
            137 Feedback       0..7   default 0
            138 Osc Sync       0..1   default 1
            139 LFO Speed      0..99  default 35
            140 LFO Delay      0..99  default 0
            141 LFO PMD        0..99  default 0
            142 LFO AMD        0..99  default 0
            143 LFO Sync       0..1   default 1
            144 LFO Wave       0..5   default 0
            145 P Mod Sens     0..7   default 3
            146 Transpose      0..48  default 24
            147 OP6 On         0..1   default 1
            148 OP5 On         0..1   default 1
            149 OP4 On         0..1   default 1
            150 OP3 On         0..1   default 1
            151 OP2 On         0..1   default 1
            152 OP1 On         0..1   default 1"#]]
        .assert_eq(&lines.join("\n"));
    }

    /// Loading the Init preset must land every parameter exactly on its
    /// declared default — "Init" and "all defaults" are the same thing.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn init_preset_is_all_defaults() {
        let mut synth = DexedPlugin::new();
        synth.set_param(PARAM_PRESET, PRESET_INIT as f32);
        let mismatches: Vec<String> = (0..synth.param_count())
            .filter(|&p| p != PARAM_PRESET)
            .filter_map(|p| {
                let info = synth.param_info(p);
                let got = synth.get_param(p);
                (got != info.default)
                    .then(|| format!("{p} {}: {got} != default {}", info.name, info.default))
            })
            .collect();
        expect![[r#"
            params checked: 153
            mismatches: []"#]]
        .assert_eq(&format!(
            "params checked: {}\nmismatches: {mismatches:?}",
            synth.param_count()
        ));
    }

    /// The Init voice is a plain sine: dominant at the played pitch, no
    /// overtones above the -35dB floor, and a square gate (full level
    /// while held, instant silence on release).
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn init_preset_is_a_pure_sine_gate() {
        use crate::testutil::harmonic_profile;
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, PRESET_INIT as f32);
        let samples = render_blocks(&mut synth, 96, &[(0, note_on(69, 1.0)), (64, note_off(69))]);
        let sustained = &samples[16 * BLOCK..56 * BLOCK];
        let mut lines = vec![format!(
            "dominant={}Hz",
            dominant_freq(sustained, f64::from(SR))
        )];
        lines.extend(harmonic_profile(sustained, f64::from(SR), 440.0, 5));
        lines.push("---".to_string());
        lines.extend(format_peaks(&samples, 16));
        expect![[r#"
            dominant=440Hz
            h1 0.0dB
            h2 -
            h3 -
            h4 -
            h5 -
            ---
            w 0 peak=0.1257
            w 1 peak=0.1250
            w 2 peak=0.1250
            w 3 peak=0.1250
            w 4 peak=0.0998
            w 5 peak=0.0000"#]]
        .assert_eq(&lines.join("\n"));
    }

    /// Voice params read back the loaded preset's patch bytes: spot-check
    /// E.PIANO 1 values a DX7 patch sheet pins (algorithm 5 → raw 4,
    /// feedback 6, OP2 the ratio-14 tine modulator, OP1 carrier level 99).
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn voice_params_read_back_the_preset() {
        let mut synth = DexedPlugin::new();
        synth.set_param(PARAM_PRESET, EPIANO);
        let read = |off: usize| synth.get_param(PARAM_VOICE_BASE + off);
        expect!["algorithm=4 feedback=6 op2_coarse=14 op1_level=99 op1_enabled=1"].assert_eq(
            &format!(
                "algorithm={} feedback={} op2_coarse={} op1_level={} op1_enabled={}",
                read(134),
                read(135),
                read(4 * 21 + 18),
                read(5 * 21 + 16),
                synth.get_param(PARAM_OP_ENABLE_BASE + 5),
            ),
        );
    }

    /// Setting OP1's Coarse to 2 on the Init voice doubles the pitch —
    /// the ratio param drives the oscillator (A4 → 880Hz).
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn coarse_param_doubles_pitch() {
        let op1_coarse = PARAM_VOICE_BASE + 5 * 21 + 18;
        check_pitch(
            PRESET_INIT as f32,
            &[(0, Ev::Param(op1_coarse, 2.0)), (0, note_on(69, 1.0))],
            expect!["dominant=880Hz"],
        );
    }

    /// Transpose is stored 0..48 with 24 = center; 36 = +12 semitones.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn transpose_param_shifts_pitch() {
        let transpose = PARAM_VOICE_BASE + 144;
        check_pitch(
            PRESET_INIT as f32,
            &[(0, Ev::Param(transpose, 36.0)), (0, note_on(69, 1.0))],
            expect!["dominant=880Hz"],
        );
    }

    /// Lowering OP1's EG R4 (release rate) turns the Init voice's instant
    /// cutoff into an audible tail after note-off at block 32.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn eg_release_rate_param_lengthens_the_tail() {
        let op1_r4 = PARAM_VOICE_BASE + 5 * 21 + 3;
        check(
            PRESET_INIT as f32,
            &[
                (0, Ev::Param(op1_r4, 30.0)),
                (0, note_on(69, 1.0)),
                (32, note_off(69)),
            ],
            96,
            expect![[r#"
                w 0 peak=0.1257
                w 1 peak=0.1251
                w 2 peak=0.1250
                w 3 peak=0.1250
                w 4 peak=0.1250
                w 5 peak=0.1228
                w 6 peak=0.1207
                w 7 peak=0.1187
                w 8 peak=0.1166
                w 9 peak=0.1145
                w10 peak=0.1126
                w11 peak=0.1106"#]],
        );
    }

    /// Operator switches: disabling OP1 (the Init voice's only carrier)
    /// mid-hold leaves the sounding note untouched (op state applies at
    /// note-on) but silences the NEXT note; re-enabling brings it back.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn op_enable_silences_the_carrier() {
        let op1_enable = PARAM_OP_ENABLE_BASE + 5;
        check(
            PRESET_INIT as f32,
            &[
                (0, note_on(69, 1.0)),
                // Toggled off while the note is held: no drop in the trace.
                (8, Ev::Param(op1_enable, 0.0)),
                (16, note_off(69)),
                // A new note with the carrier off is silent.
                (24, note_on(69, 1.0)),
                (40, note_off(69)),
                (48, Ev::Param(op1_enable, 1.0)),
                (48, note_on(69, 1.0)),
                (64, note_off(69)),
            ],
            72,
            expect![[r#"
                w 0 peak=0.1257
                w 1 peak=0.1251
                w 2 peak=0.1210
                w 3 peak=0.0002
                w 4 peak=0.0000
                w 5 peak=0.0000
                w 6 peak=0.1266
                w 7 peak=0.1251
                w 8 peak=0.1010"#]],
        );
    }

    /// The op switches take a normalized-ish value like every param: the
    /// 0.5 threshold decides the bit, and readback is exactly 0 or 1.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn op_enable_rounds_at_half() {
        let op1_enable = PARAM_OP_ENABLE_BASE + 5;
        let mut synth = DexedPlugin::new();
        let mut lines = Vec::new();
        for v in [0.4f32, 0.5, 0.6, 0.0, 1.0] {
            synth.set_param(op1_enable, v);
            lines.push(format!("set {v} → {}", synth.get_param(op1_enable)));
        }
        expect![[r#"
            set 0.4 → 0
            set 0.5 → 1
            set 0.6 → 1
            set 0 → 0
            set 1 → 1"#]]
        .assert_eq(&lines.join("\n"));
    }

    /// Out-of-range indices: param_info falls back to the "?" placeholder,
    /// set_param is a no-op, get_param reads 0.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn params_past_the_end_are_inert() {
        let mut synth = DexedPlugin::new();
        synth.set_param(PARAM_COUNT, 42.0);
        let info = synth.param_info(PARAM_COUNT);
        expect!["count=153 info=? 0..1 get=0"].assert_eq(&format!(
            "count={} info={} {}..{} get={}",
            synth.param_count(),
            info.name,
            info.min,
            info.max,
            synth.get_param(PARAM_COUNT)
        ));
    }

    /// LFO params are live, not just stored bytes: deep pitch modulation
    /// (speed + PMD + pitch-mod sensitivity) smears the Init voice's pure
    /// 440Hz line, so the energy measured exactly at 440Hz drops hard.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn lfo_vibrato_spreads_the_line() {
        let mag_at_440 = |events: &[(usize, Ev)]| {
            let mut synth = DexedPlugin::new();
            synth.reset(SR, 1024);
            synth.set_param(PARAM_PRESET, PRESET_INIT as f32);
            let samples = render_blocks(&mut synth, 64, events);
            goertzel(&samples[16 * BLOCK..], f64::from(SR), 440.0)
        };
        let dry = mag_at_440(&[(0, note_on(69, 1.0))]);
        let lfo_speed = PARAM_VOICE_BASE + 137;
        let lfo_pmd = PARAM_VOICE_BASE + 139;
        let pms = PARAM_VOICE_BASE + 143;
        let wobbly = mag_at_440(&[
            (0, Ev::Param(lfo_speed, 70.0)),
            (0, Ev::Param(lfo_pmd, 99.0)),
            (0, Ev::Param(pms, 7.0)),
            (0, note_on(69, 1.0)),
        ]);
        expect![[r#"
            dry mag@440: 0.125
            vibrato mag@440: 0.009
            spread (dry/vibrato > 4): true"#]]
        .assert_eq(&format!(
            "dry mag@440: {dry:.3}\nvibrato mag@440: {:.3}\nspread (dry/vibrato > 4): {}",
            wobbly,
            dry / wobbly > 4.0
        ));
    }

    /// Disabling a modulator (E.PIANO's OP2, the ratio-14 "tine") keeps
    /// the note sounding but strips the high sidebands it modulated in.
    /// Phase modulation at ratio 14 puts energy at 440 ± 14·440 Hz — the
    /// 5720/6600Hz "tine" band — which must collapse with OP2 off while
    /// the fundamental stays put.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn op_enable_off_modulator_strips_the_tine_band() {
        let op2_enable = PARAM_OP_ENABLE_BASE + 4;
        let mags = |events: &[(usize, Ev)]| {
            let mut synth = DexedPlugin::new();
            synth.reset(SR, 1024);
            synth.set_param(PARAM_PRESET, EPIANO);
            let samples = render_blocks(&mut synth, 32, events);
            let sustained = &samples[8 * BLOCK..];
            let m = |hz: f64| goertzel(sustained, f64::from(SR), hz);
            (m(440.0), m(5720.0) + m(6600.0))
        };
        let (stock_f0, stock_tine) = mags(&[(0, note_on(69, 1.0))]);
        let (muted_f0, muted_tine) =
            mags(&[(0, Ev::Param(op2_enable, 0.0)), (0, note_on(69, 1.0))]);
        expect![[r#"
            stock:   f0 0.189  tine band 0.0327
            op2 off: f0 0.191  tine band 0.0017
            tine collapses (>10x): true  fundamental survives: true"#]]
        .assert_eq(&format!(
            "stock:   f0 {stock_f0:.3}  tine band {stock_tine:.4}\n\
             op2 off: f0 {muted_f0:.3}  tine band {muted_tine:.4}\n\
             tine collapses (>10x): {}  fundamental survives: {}",
            stock_tine / muted_tine > 10.0,
            muted_f0 > stock_f0 * 0.5
        ));
    }

    /// Edits survive `reset` (sample-rate change / synth switch): the
    /// patch is the live state, not a preset index to reload. The edited
    /// coarse ratio must still sound after a reset.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn edits_survive_reset() {
        let op1_coarse = PARAM_VOICE_BASE + 5 * 21 + 18;
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, PRESET_INIT as f32);
        synth.set_param(op1_coarse, 2.0);
        synth.reset(SR, 1024);
        let readback = synth.get_param(op1_coarse);
        let samples = render_blocks(&mut synth, 64, &[(0, note_on(69, 1.0))]);
        let sustained = &samples[16 * BLOCK..];
        expect![[r#"
            coarse after reset: 2
            dominant=880Hz"#]]
        .assert_eq(&format!(
            "coarse after reset: {readback}\ndominant={}Hz",
            dominant_freq(sustained, f64::from(SR))
        ));
    }

    /// `preset_positions` (the UI knob-mirror seed) must agree with what
    /// `get_param` reports after actually loading the preset.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn preset_positions_match_loaded_params() {
        let mut mismatches = Vec::new();
        // 200 = ROM4A voice 8, 256 = Finetales 1 voice 0: the knob-mirror
        // seed must agree with the engine for cross-bank presets too.
        for preset in [0usize, 10, 200, 256, PRESET_INIT] {
            let positions = preset_positions(preset);
            let mut synth = DexedPlugin::new();
            synth.set_param(PARAM_PRESET, preset as f32);
            for p in (0..synth.param_count()).filter(|&p| p != PARAM_PRESET) {
                let info = synth.param_info(p);
                let expected = if info.max > info.min {
                    (synth.get_param(p) - info.min) / (info.max - info.min)
                } else {
                    0.0
                };
                if (positions[p] - expected).abs() > 1e-6 {
                    mismatches.push(format!(
                        "preset {preset} p{p} {}: {} != {expected}",
                        info.name, positions[p]
                    ));
                }
            }
        }
        expect!["mismatches: []"].assert_eq(&format!("mismatches: {mismatches:?}"));
    }

    /// The Cargo.toml provenance comment must cite the same pinned
    /// revisions as the public constants the License page displays.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn cargo_toml_cites_the_pinned_revisions() {
        let manifest = include_str!("../Cargo.toml");
        assert!(
            manifest.contains(&PORTED_FROM_DEXED_REV[..8]),
            "Cargo.toml must cite the Dexed msfa revision {PORTED_FROM_DEXED_REV}"
        );
        assert!(
            manifest.contains(&PORTED_FROM_MSFA_REV[..8]),
            "Cargo.toml must cite the msfa revision {PORTED_FROM_MSFA_REV}"
        );
    }
}

#[cfg(test)]
mod watch_tests {
    //! What the voices report to a screen while they play (`Plugin::watch`).

    use super::*;
    use crate::watch::{EnvStage, OpWatch};
    use bloop_events::EventBuffer;
    use expect_test::{expect, Expect};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    const BLOCK: usize = 128;
    const SR: f32 = 48000.0;

    /// Where one byte of operator `op` (1-6) lives among the params. The
    /// patch stores operator 6 first; `field` is the offset in its 21.
    fn op_param(op: usize, field: usize) -> usize {
        PARAM_VOICE_BASE + (6 - op) * 21 + field
    }

    /// An operator's envelope rates and levels, and its output level.
    fn eg(op: usize, rates: [u8; 4], levels: [u8; 4], output: u8) -> Vec<(usize, f32)> {
        let mut edits: Vec<(usize, f32)> = (0..4)
            .flat_map(|i| {
                [
                    (op_param(op, i), f32::from(rates[i])),
                    (op_param(op, 4 + i), f32::from(levels[i])),
                ]
            })
            .collect();
        edits.push((op_param(op, 16), f32::from(output)));
        edits
    }

    /// The switch of operator `op` (1-6).
    fn op_switch(op: usize) -> usize {
        PARAM_OP_ENABLE_BASE + 6 - op
    }

    #[derive(Clone, Copy)]
    enum Key {
        On(u8, f32),
        /// Struck at sample 100 of its block: after the block's last
        /// 64-sample chunk has started, so it sounds from the next block.
        Late(u8),
        Off(u8),
        /// A parameter set before the block renders.
        Param(usize, f32),
    }

    fn short(stage: EnvStage) -> &'static str {
        match stage {
            EnvStage::Rate1 => "R1",
            EnvStage::Rate2 => "R2",
            EnvStage::Rate3 => "R3",
            EnvStage::Sustain => "S",
            EnvStage::Rate4 => "R4",
            EnvStage::End => "E",
        }
    }

    /// The INIT voice with `patch` edits on it, played by `keys` (each at
    /// the start of its 128-sample block), and the Watch after each block
    /// in `look_at`: per sounding voice its slot, note-on count and key,
    /// and for each operator in `ops` (1-6) its stage, level in patch
    /// units and amp. Every voice not shown must be all default.
    fn check_watch(
        patch: &[(usize, f32)],
        keys: &[(usize, Key)],
        look_at: &[usize],
        ops: &[usize],
        expect: Expect,
    ) {
        let mut synth = DexedPlugin::new();
        synth.reset(SR, 1024);
        synth.set_param(PARAM_PRESET, PRESET_INIT as f32);
        for &(param, value) in patch {
            synth.set_param(param, value);
        }
        let last = look_at.iter().copied().max().unwrap_or(0);
        let mut lines = Vec::new();
        for block in 0..=last {
            let mut events = EventBuffer::new();
            for &(_, key) in keys.iter().filter(|(at, _)| *at == block) {
                let note = bloop_core::Note::new;
                let (time, n, velocity) = match key {
                    Key::On(n, velocity) => (0, n, velocity),
                    Key::Late(n) => (100, n, 1.0),
                    Key::Off(n) => {
                        events.push(NoteEvent::NoteOff {
                            time: 0,
                            channel: 0,
                            note: note(n),
                        });
                        continue;
                    }
                    Key::Param(param, value) => {
                        synth.set_param(param, value);
                        continue;
                    }
                };
                events.push(NoteEvent::NoteOn {
                    time,
                    channel: 0,
                    note: note(n),
                    velocity,
                });
            }
            let (mut left, mut right) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            let mut outputs: [&mut [f32]; 2] = [&mut left, &mut right];
            synth.process(&mut ProcessContext {
                outputs: &mut outputs,
                input_events: &events,
                sample_rate: SR,
                tempo: 120.0,
                is_playing: false,
            });
            if !look_at.contains(&block) {
                continue;
            }
            // Filled with junk first: `watch` must overwrite all of it.
            let junk = VoiceWatch {
                active: true,
                note_on: 999,
                key_down: true,
                pitch: 1,
                lanes: [-1.0; bloop_plugin_api::WATCH_LANES],
            };
            let mut watch = Watch {
                voices: [junk; WATCH_VOICES],
                globals: [-1.0; bloop_plugin_api::WATCH_GLOBALS],
                knobs: [[-1.0; bloop_plugin_api::WATCH_KNOBS]; WATCH_VOICES],
                params: [7; bloop_plugin_api::WATCH_KNOBS],
            };
            synth.watch(&mut watch);
            assert!(
                watch
                    .globals
                    .iter()
                    .chain(watch.knobs.iter().flatten())
                    .all(|v| *v == 0.0),
                "the FM synth has no globals and follows no knobs"
            );
            let ms = (block + 1) * BLOCK * 1000 / SR as usize;
            let mut voices = Vec::new();
            for (slot, voice) in watch.active() {
                let key = if voice.key_down { "down" } else { "up  " };
                let shown: Vec<String> = ops
                    .iter()
                    .map(|&op| {
                        let o = OpWatch::of(voice, u8::try_from(op).unwrap_or(1));
                        // An exact nought, which the engine is not
                        // rendering, apart from one that rounds to it.
                        let amp = if o.amp == 0.0 {
                            "0".to_string()
                        } else {
                            format!("{:.4}", o.amp)
                        };
                        format!("{op}:{:<2} {:>4.1} {amp:<6}", short(o.stage), o.level)
                    })
                    .collect();
                voices.push(format!(
                    "v{slot} #{} key {} {key} | {}",
                    voice.note_on,
                    voice.pitch,
                    shown.join(" | ")
                ));
            }
            assert!(
                watch
                    .voices
                    .iter()
                    .all(|v| v.active || *v == VoiceWatch::default()),
                "an inactive voice carried state"
            );
            if voices.is_empty() {
                voices.push("(silent)".to_string());
            }
            lines.push(format!(
                "@{block:>4} {ms:>5}ms {}",
                voices.join("\n              ")
            ));
        }
        expect.assert_eq(&lines.join("\n"));
    }

    /// One voice from note-on to silence. OP1 (a carrier in algorithm 1)
    /// climbs at R1=60 to L1=99, falls at R2=50 to L2=80, at R3=50 to
    /// L3=62, sustains there, and releases at R4=50 to L4=0 once the key
    /// comes up a second in. OP2 (OP1's modulator) is at output level 70
    /// on an instant envelope that sustains at 99. OP3 is at full output
    /// level but switched off; OP4 is at output level 0.
    ///
    /// What to check: OP1's level walks 99 → 80 → 62 and holds at exactly
    /// 62 in sustain, with amp 2^-4.5 ≈ 0.044 there (L62 is 18 steps of
    /// 1/4 doubling under L99 in the engine's mapping); OP2 reads 99 with
    /// amp 2^-3.625 ≈ 0.081 (output level 70 is 29 steps of 1/8 doubling
    /// under 99); OP3 and OP4 keep time but never have any amp; at the end
    /// the voice is gone.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn one_voice_from_strike_to_silence() {
        let mut patch = eg(1, [60, 50, 50, 50], [99, 80, 62, 0], 99);
        patch.extend(eg(2, [99, 99, 99, 99], [99, 99, 99, 0], 70));
        patch.extend(eg(3, [99, 99, 99, 99], [99, 99, 99, 0], 99));
        patch.push((op_switch(3), 0.0));
        check_watch(
            &patch,
            &[(0, Key::On(60, 1.0)), (375, Key::Off(60))],
            &[
                0, 1, 3, 10, 30, 60, 100, 150, 250, 374, 376, 400, 450, 550, 700, 800, 900,
            ],
            &[1, 2, 3, 4],
            expect![[r#"
                @   0     2ms v0 #1 key 60 down | 1:R1 38.7 0.0056 | 2:R3 99.0 0.0811 | 3:R3 99.0 0      | 4:R3 99.0 0     
                @   1     5ms v0 #1 key 60 down | 1:R1 44.9 0.0096 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @   3    10ms v0 #1 key 60 down | 1:R1 55.7 0.0242 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @  10    29ms v0 #1 key 60 down | 1:R1 81.1 0.2140 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @  30    82ms v0 #1 key 60 down | 1:R2 96.7 0.8277 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @  60   162ms v0 #1 key 60 down | 1:R2 89.4 0.4556 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @ 100   269ms v0 #1 key 60 down | 1:R3 79.8 0.2061 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @ 150   402ms v0 #1 key 60 down | 1:R3 68.3 0.0762 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @ 250   669ms v0 #1 key 60 down | 1:S  62.0 0.0442 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @ 374  1000ms v0 #1 key 60 down | 1:S  62.0 0.0442 | 2:S  99.0 0.0811 | 3:S  99.0 0      | 4:S  99.0 0     
                @ 376  1005ms v0 #1 key 60 up   | 1:R4 61.5 0.0425 | 2:E   0.0 0      | 3:E   0.0 0      | 4:E   0.0 0     
                @ 400  1069ms v0 #1 key 60 up   | 1:R4 55.9 0.0263 | 2:E   0.0 0      | 3:E   0.0 0      | 4:E   0.0 0     
                @ 450  1202ms v0 #1 key 60 up   | 1:R4 44.3 0.0097 | 2:E   0.0 0      | 3:E   0.0 0      | 4:E   0.0 0     
                @ 550  1469ms v0 #1 key 60 up   | 1:R4 20.9 0.0013 | 2:E   0.0 0      | 3:E   0.0 0      | 4:E   0.0 0     
                @ 700  1869ms v0 #1 key 60 up   | 1:R4  2.3 0.0001 | 2:E   0.0 0      | 3:E   0.0 0      | 4:E   0.0 0     
                @ 800  2136ms (silent)
                @ 900  2402ms (silent)"#]],
        );
    }

    /// Voices come and go. Three notes struck in turn, the first let go;
    /// once it has finished it is inactive, and the next note takes its
    /// slot with the next note-on count. With all eight voices held, a
    /// ninth steals the oldest: its slot, a new count. OP1 releases at
    /// R4=70 so that a released voice is over within the run.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn voices_count_their_note_ons_and_free_their_slots() {
        let patch = eg(1, [99, 99, 99, 70], [99, 99, 99, 0], 99);
        let mut keys = vec![
            (0, Key::On(60, 1.0)),
            (10, Key::On(64, 1.0)),
            (20, Key::On(67, 1.0)),
            (30, Key::Off(60)),
            (200, Key::On(72, 1.0)),
        ];
        // Five more fill all eight voices; then a ninth.
        for (i, note) in (74..79).enumerate() {
            keys.push((300 + i, Key::On(note, 1.0)));
        }
        keys.push((320, Key::On(84, 1.0)));
        check_watch(
            &patch,
            &keys,
            &[0, 10, 20, 31, 199, 200, 310, 320],
            &[1],
            expect![[r#"
                @   0     2ms v0 #1 key 60 down | 1:R3 99.0 1.0000
                @  10    29ms v0 #1 key 60 down | 1:S  99.0 1.0000
                              v1 #2 key 64 down | 1:R3 99.0 1.0000
                @  20    56ms v0 #1 key 60 down | 1:S  99.0 1.0000
                              v1 #2 key 64 down | 1:S  99.0 1.0000
                              v2 #3 key 67 down | 1:R3 99.0 1.0000
                @  31    85ms v0 #1 key 60 up   | 1:R4 95.3 0.7273
                              v1 #2 key 64 down | 1:S  99.0 1.0000
                              v2 #3 key 67 down | 1:S  99.0 1.0000
                @ 199   533ms v1 #2 key 64 down | 1:S  99.0 1.0000
                              v2 #3 key 67 down | 1:S  99.0 1.0000
                @ 200   536ms v0 #4 key 72 down | 1:R3 99.0 1.0000
                              v1 #2 key 64 down | 1:S  99.0 1.0000
                              v2 #3 key 67 down | 1:S  99.0 1.0000
                @ 310   829ms v0 #4 key 72 down | 1:S  99.0 1.0000
                              v1 #2 key 64 down | 1:S  99.0 1.0000
                              v2 #3 key 67 down | 1:S  99.0 1.0000
                              v3 #5 key 74 down | 1:S  99.0 1.0000
                              v4 #6 key 75 down | 1:S  99.0 1.0000
                              v5 #7 key 76 down | 1:S  99.0 1.0000
                              v6 #8 key 77 down | 1:S  99.0 1.0000
                              v7 #9 key 78 down | 1:S  99.0 1.0000
                @ 320   856ms v0 #4 key 72 down | 1:S  99.0 1.0000
                              v1 #10 key 84 down | 1:R3 99.0 1.0000
                              v2 #3 key 67 down | 1:S  99.0 1.0000
                              v3 #5 key 74 down | 1:S  99.0 1.0000
                              v4 #6 key 75 down | 1:S  99.0 1.0000
                              v5 #7 key 76 down | 1:S  99.0 1.0000
                              v6 #8 key 77 down | 1:S  99.0 1.0000
                              v7 #9 key 78 down | 1:S  99.0 1.0000"#]],
        );
    }

    /// The amp is how loud the operator is, and so it moves with
    /// velocity; the level is where the envelope is. OP1 at full output
    /// level with velocity sensitivity 7 and a slow attack (R1=40), struck
    /// at velocity 127 and at 38. Held, both sit at exactly 99; full
    /// velocity is louder than the neutral 1.0 (2^0.875 ≈ 1.83), the soft
    /// one 3.5 doublings quieter (≈ 0.088). On the way up the two are
    /// equally loud — an attack climbs through loudness, the same for
    /// both, until each reaches its own top — and so the soft note, whose
    /// top is lower, is further up its own scale: its dot runs ahead.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn velocity_moves_the_amp_and_not_the_level() {
        let mut patch = eg(1, [40, 99, 99, 99], [99, 99, 99, 0], 99);
        patch.push((op_param(1, 15), 7.0));
        check_watch(
            &patch,
            &[(0, Key::On(60, 1.0)), (0, Key::On(64, 0.3))],
            &[0, 20, 60, 400],
            &[1],
            expect![[r#"
                @   0     2ms v0 #1 key 60 down | 1:R1 25.8 0.0034
                              v1 #2 key 64 down | 1:R1 60.9 0.0034
                @  20    56ms v0 #1 key 60 down | 1:R1 38.6 0.0102
                              v1 #2 key 64 down | 1:R1 73.8 0.0102
                @  60   162ms v0 #1 key 60 down | 1:R1 58.4 0.0559
                              v1 #2 key 64 down | 1:R1 93.7 0.0559
                @ 400  1069ms v0 #1 key 60 down | 1:S  99.0 1.8340
                              v1 #2 key 64 down | 1:S  99.0 0.0884"#]],
        );
    }

    /// The key a voice says is the one it sounds: the struck key moved by
    /// the patch's transpose, here an octave up (36; 24 is none), and no
    /// further than the top of the keyboard.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_voice_says_the_key_it_sounds() {
        let mut patch = eg(1, [99, 99, 99, 99], [99, 99, 99, 0], 99);
        patch.push((PARAM_VOICE_BASE + 144, 36.0));
        check_watch(
            &patch,
            &[(0, Key::On(60, 1.0)), (0, Key::On(120, 1.0))],
            &[0],
            &[1],
            expect![[r#"
                @   0     2ms v0 #1 key 72 down | 1:R3 99.0 1.0000
                              v1 #2 key 127 down | 1:R3 99.0 1.0000"#]],
        );
    }

    /// Keyboard level scaling is in the amp too, not in the level: a
    /// right depth of 99 on a falling linear curve above a breakpoint of
    /// C3 makes a high note quieter than a low one at the same level. And
    /// a carrier whose L4 is above nought rings on after its key is up —
    /// at End, at exactly L4, active for ever.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn keyboard_scaling_is_in_the_amp_and_a_nonzero_l4_rings_on() {
        let mut patch = eg(1, [99, 99, 99, 60], [99, 99, 90, 50], 99);
        patch.extend([
            (op_param(1, 8), 39.0),  // breakpoint C3
            (op_param(1, 10), 99.0), // right depth
            (op_param(1, 12), 0.0),  // right curve: -LIN
        ]);
        check_watch(
            &patch,
            &[
                (0, Key::On(48, 1.0)),
                (0, Key::On(96, 1.0)),
                (40, Key::Off(48)),
                (40, Key::Off(96)),
            ],
            &[20, 41, 300, 1500],
            &[1],
            expect![[r#"
                @  20    56ms v0 #1 key 48 down | 1:S  90.0 0.5000
                              v1 #2 key 96 down | 1:S  90.0 0.0001
                @  41   112ms v0 #1 key 48 up   | 1:R4 88.6 0.4437
                              v1 #2 key 96 up   | 1:R4 83.5 0.0001
                @ 300   802ms v0 #1 key 48 up   | 1:E  50.0 0.0156
                              v1 #2 key 96 up   | 1:E  50.0 0     
                @1500  4002ms v0 #1 key 48 up   | 1:E  50.0 0.0156
                              v1 #2 key 96 up   | 1:E  50.0 0     "#]],
        );
    }

    /// Each voice plays the patch it was struck with, and shows it. With a
    /// note held in sustain at L3=99, L3 is turned to 50: the held note
    /// stays at 99, the next note sustains at 50. Then a program change:
    /// what was let go stops at once, what is held plays on.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_edit_is_shown_on_the_next_note_and_a_program_change_ends_released_ones() {
        let patch = eg(1, [99, 99, 99, 30], [99, 99, 99, 0], 99);
        check_watch(
            &patch,
            &[
                (0, Key::On(60, 1.0)),
                (30, Key::Param(op_param(1, 6), 50.0)),
                (40, Key::On(64, 1.0)),
                (70, Key::Off(60)),
                (80, Key::Param(PARAM_PRESET, 0.0)),
            ],
            &[29, 30, 60, 79, 80],
            &[1],
            expect![[r#"
                @  29    80ms v0 #1 key 60 down | 1:S  99.0 1.0000
                @  30    82ms v0 #1 key 60 down | 1:S  99.0 1.0000
                @  60   162ms v0 #1 key 60 down | 1:S  99.0 1.0000
                              v1 #2 key 64 down | 1:S  50.0 0.0156
                @  79   213ms v0 #1 key 60 up   | 1:R4 98.7 0.9785
                              v1 #2 key 64 down | 1:S  50.0 0.0156
                @  80   216ms v1 #2 key 64 down | 1:S  50.0 0.0156"#]],
        );
    }

    /// Amplitude modulation is in the amp and not in the level: the LFO
    /// at full depth (AMD 99) on an operator with AMS 3 swings the amp
    /// while the envelope sits at exactly 99.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn amplitude_modulation_moves_the_amp_and_not_the_level() {
        let mut patch = eg(1, [99, 99, 99, 99], [99, 99, 99, 0], 99);
        patch.extend([
            (op_param(1, 14), 3.0),         // amp mod sensitivity
            (PARAM_VOICE_BASE + 137, 70.0), // LFO speed
            (PARAM_VOICE_BASE + 138, 0.0),  // LFO delay
            (PARAM_VOICE_BASE + 140, 99.0), // amp mod depth
        ]);
        check_watch(
            &patch,
            &[(0, Key::On(60, 1.0))],
            &[20, 30, 40, 50, 60, 70, 80],
            &[1],
            expect![[r#"
                @  20    56ms v0 #1 key 60 down | 1:S  99.0 0.8205
                @  30    82ms v0 #1 key 60 down | 1:S  99.0 0.0137
                @  40   109ms v0 #1 key 60 down | 1:S  99.0 0.6219
                @  50   136ms v0 #1 key 60 down | 1:S  99.0 0.1673
                @  60   162ms v0 #1 key 60 down | 1:S  99.0 0.3197
                @  70   189ms v0 #1 key 60 down | 1:S  99.0 0.4749
                @  80   216ms v0 #1 key 60 down | 1:S  99.0 0.0647"#]],
        );
    }

    /// A note struck after its block's last chunk has started has not
    /// sounded by the end of that block. Its slot was a loud voice's — the
    /// ninth of nine sustained notes steals the oldest — but it shows no
    /// amp until it renders, a block later.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_voice_shows_no_amp_before_it_has_sounded() {
        let patch = eg(1, [99, 99, 99, 99], [99, 99, 99, 0], 99);
        let mut keys: Vec<(usize, Key)> = (0..8).map(|i| (i, Key::On(60 + i as u8, 1.0))).collect();
        keys.push((20, Key::Late(80)));
        check_watch(
            &patch,
            &keys,
            &[19, 20, 21],
            &[1],
            expect![[r#"
                @  19    53ms v0 #1 key 60 down | 1:S  99.0 1.0000
                              v1 #2 key 61 down | 1:S  99.0 1.0000
                              v2 #3 key 62 down | 1:S  99.0 1.0000
                              v3 #4 key 63 down | 1:S  99.0 1.0000
                              v4 #5 key 64 down | 1:S  99.0 1.0000
                              v5 #6 key 65 down | 1:S  99.0 1.0000
                              v6 #7 key 66 down | 1:S  99.0 1.0000
                              v7 #8 key 67 down | 1:S  99.0 1.0000
                @  20    56ms v0 #9 key 80 down | 1:R1  1.4 0     
                              v1 #2 key 61 down | 1:S  99.0 1.0000
                              v2 #3 key 62 down | 1:S  99.0 1.0000
                              v3 #4 key 63 down | 1:S  99.0 1.0000
                              v4 #5 key 64 down | 1:S  99.0 1.0000
                              v5 #6 key 65 down | 1:S  99.0 1.0000
                              v6 #7 key 66 down | 1:S  99.0 1.0000
                              v7 #8 key 67 down | 1:S  99.0 1.0000
                @  21    58ms v0 #9 key 80 down | 1:R3 99.0 1.0000
                              v1 #2 key 61 down | 1:S  99.0 1.0000
                              v2 #3 key 62 down | 1:S  99.0 1.0000
                              v3 #4 key 63 down | 1:S  99.0 1.0000
                              v4 #5 key 64 down | 1:S  99.0 1.0000
                              v5 #6 key 65 down | 1:S  99.0 1.0000
                              v6 #7 key 66 down | 1:S  99.0 1.0000
                              v7 #8 key 67 down | 1:S  99.0 1.0000"#]],
        );
    }
}
