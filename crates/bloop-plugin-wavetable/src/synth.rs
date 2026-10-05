//! The synth around the voices: `WavetableAudioProcessor`'s
//! `processBlock`, `updateParams` and effects (`PluginProcessor.cpp`), and
//! the note handling it inherits — `gin::Synthesiser`
//! (`gin_synthesiser.h`: mono and legato, glide, voice stealing) over
//! JUCE's `MPESynthesiser` and `MPEInstrument` in legacy mode (one MIDI
//! channel, notes rendered sample-accurately with sub-blocks of at least
//! 32 samples after the first).
//!
//! JUCE's classes are written here from their documented behaviour; the
//! gold tests run the real ones.

use crate::dsp::derez::DeRez2;
use crate::dsp::fireamp::FireAmp;
use crate::dsp::fx::{distort, Gain, Gate, StereoDelay};
use crate::dsp::grindamp::GrindAmp;
use crate::dsp::lfo::{Lfo, LfoParams, Shape, StepLfo};
use crate::dsp::random::JuceRandom;
use crate::dsp::tables::NoiseSource;
use crate::modmatrix::{src, Matrix, DESTINATION_OF};
use crate::params::{id, Values, MOD_SLOTS, PARAMS};
use crate::bank::TableSet;
use crate::voice::{Context, GlideInfo, KeyState, MpeValue, Note, Voice, MAX_BLOCK};
use bloop_plugin_socalabs::shared::{GlibcRand, Modulation, PlateReverb, DURATIONS};

/// How many voices the synth keeps: Wavetable's fifty, ten more than its
/// Voices knob reaches, so a voice stolen can ring out its last 10 ms
/// while the new note takes another.
pub const VOICE_SLOTS: usize = 50;

/// How far back the delay reaches. The original's line holds 120 s (46 MB
/// at 48 kHz) whatever the delay is set to; this one holds what a factory
/// preset asks of it, and a longer time is held at this.
pub const DELAY_ROOM: f64 = 4.0;

/// A MIDI-like event, at a sample of the block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    /// 14-bit, 8192 at rest.
    PitchWheel(i32),
    /// Controller `cc` to a 7-bit value.
    Controller { cc: u8, value: u8 },
    /// The note cut without a release.
    Choke { note: u8 },
}

impl Event {
    fn is_note_on(self) -> bool {
        matches!(self, Self::NoteOn { velocity, .. } if velocity > 0)
    }

    fn is_note_off(self) -> bool {
        matches!(
            self,
            Self::NoteOff { .. } | Self::NoteOn { velocity: 0, .. } | Self::Choke { .. }
        )
    }
}

/// The notes held, as `MPEInstrument` lists them.
#[derive(Clone, Copy, Debug)]
struct Notes {
    notes: [Note; 128],
    len: usize,
}

impl Default for Notes {
    fn default() -> Self {
        Self {
            notes: [Note::default(); 128],
            len: 0,
        }
    }
}

impl Notes {
    fn as_slice(&self) -> &[Note] {
        &self.notes[..self.len]
    }

    fn position(&self, id: u16) -> Option<usize> {
        self.as_slice().iter().position(|n| n.id == id)
    }

    fn remove(&mut self, at: usize) -> Note {
        let note = self.notes[at];
        self.notes.copy_within(at + 1..self.len, at);
        self.len -= 1;
        note
    }

    fn push(&mut self, note: Note) {
        if self.len < self.notes.len() {
            self.notes[self.len] = note;
            self.len += 1;
        }
    }

    fn last(&self) -> Option<Note> {
        self.as_slice().last().copied()
    }
}

/// The synth: Wavetable's processor, voices and effects.
#[expect(
    clippy::struct_excessive_bools,
    reason = "gin::Synthesiser's mono, legato, glissando and portamento flags, and the processor's own"
)]
pub struct Synth {
    pub values: Values,
    matrix: Matrix,
    routes_dirty: bool,
    voices: Vec<Voice>,
    tables: Option<std::sync::Arc<TableSet>>,
    noise: NoiseSource,
    rng: JuceRandom,
    system: JuceRandom,

    notes: Notes,
    note_stack: Notes,
    last_pitchbend: MpeValue,
    pitchbend_range: i32,
    mono: bool,
    legato: bool,
    glissando: bool,
    portamento: bool,
    glide_rate: f32,
    num_voices: i32,
    last_note: f32,
    note_on_counter: u32,
    note_on_index: i32,
    note_off_index: i32,
    active_voice: Option<usize>,
    next_age: i32,

    mono_lfos: [Lfo; 3],
    mono_step: StepLfo,
    gate: Gate,
    chorus: Modulation,
    distortion: f32,
    bitcrusher: DeRez2,
    fire_amp: FireAmp,
    grind_amp: GrindAmp,
    delay: StereoDelay,
    reverb: PlateReverb,
    output_gain: Gain,

    sample_rate: f64,
    bpm: f32,
    preset_loaded: bool,
    last_mono: bool,
}

impl Synth {
    /// A synth as Wavetable's constructor leaves it: every parameter at
    /// its default, prepared for 44.1 kHz. Allocates.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Lfo::warm();
        // Airwindows seeds its dither from glibc's `rand()`, first as each
        // effect is built and again as the processor resets them.
        let mut glibc = GlibcRand::new(1);
        let mut rand = || glibc.rand();
        let mut bitcrusher = DeRez2::new(&mut rand);
        let mut fire_amp = FireAmp::new(&mut rand);
        let mut grind_amp = GrindAmp::new(&mut rand);
        bitcrusher.reset(&mut rand);
        fire_amp.reset(&mut rand);
        grind_amp.reset(&mut rand);

        let mut synth = Self {
            values: Values::default(),
            matrix: Matrix::default(),
            routes_dirty: true,
            voices: vec![Voice::default(); VOICE_SLOTS],
            tables: None,
            noise: NoiseSource::new(seed),
            rng: JuceRandom::new(i64::from(seed)),
            system: JuceRandom::new(i64::from(seed) ^ 0x5DEE_CE66),
            notes: Notes::default(),
            note_stack: Notes::default(),
            last_pitchbend: MpeValue::CENTRE,
            pitchbend_range: 2,
            mono: false,
            legato: false,
            glissando: false,
            portamento: false,
            glide_rate: 500.0,
            num_voices: 32,
            last_note: -1.0,
            note_on_counter: 0,
            note_on_index: -1,
            note_off_index: -1,
            active_voice: None,
            next_age: 0,
            mono_lfos: [Lfo::default(); 3],
            mono_step: StepLfo::default(),
            gate: Gate::default(),
            chorus: Modulation::new(0.5),
            distortion: 0.0,
            bitcrusher,
            fire_amp,
            grind_amp,
            delay: StereoDelay::new(DELAY_ROOM, 48000.0),
            reverb: PlateReverb::new(44100.0),
            output_gain: Gain::default(),
            sample_rate: 0.0,
            bpm: 120.0,
            preset_loaded: false,
            last_mono: false,
        };
        synth.prepare(44100.0);
        synth
    }

    /// `prepareToPlay`: everything at `sample_rate`. Allocates nothing at
    /// a rate it has run at before... except the delay line, sized once
    /// per rate.
    pub fn prepare(&mut self, sample_rate: f64) {
        if self.sample_rate != sample_rate {
            self.release_all_notes();
            self.sample_rate = sample_rate;
        }
        self.turn_off_all_voices(false);
        for v in &mut self.voices {
            v.set_sample_rate(sample_rate);
        }
        self.matrix
            .set_sample_rate(sample_rate, self.voices.iter_mut().map(|v| &mut v.modulation));
        self.gate.set_sample_rate(sample_rate);
        self.chorus.prepare(sample_rate);
        self.delay.set_sample_rate(sample_rate);
        self.reverb.set_sample_rate(sample_rate as f32);
        for l in &mut self.mono_lfos {
            l.set_sample_rate(sample_rate);
        }
        self.mono_step.set_sample_rate(sample_rate);
    }

    /// `reset`: the effects' tails and the mono modulators cleared.
    pub fn reset(&mut self) {
        self.gate.reset();
        self.chorus.reset();
        self.delay.reset();
        self.reverb.reset();
        for l in &mut self.mono_lfos {
            l.reset();
        }
        self.mono_step.reset();
    }

    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// The tables the oscillators, the sub and the noise read; the old set
    /// comes back, for the caller to drop off the audio thread.
    pub fn set_tables(
        &mut self,
        tables: Option<std::sync::Arc<TableSet>>,
    ) -> Option<std::sync::Arc<TableSet>> {
        core::mem::replace(&mut self.tables, tables)
    }

    #[must_use]
    pub fn tables(&self) -> Option<&std::sync::Arc<TableSet>> {
        self.tables.as_ref()
    }

    /// A parameter set, as the host sets it.
    pub fn set_param(&mut self, index: usize, value: f32) {
        self.values.set(index, value);
        if (id::MOD1_SRC..id::MOD1_SRC + MOD_SLOTS * 3).contains(&index) {
            self.routes_dirty = true;
        }
    }

    /// Note that a preset has been put in place: on the next block every
    /// voice stops and the delay and reverb empty (`presetLoaded`).
    pub fn preset_loaded(&mut self) {
        self.routes_dirty = true;
        self.preset_loaded = true;
    }

    /// Whether any voice is sounding.
    #[must_use]
    pub fn is_sounding(&self) -> bool {
        self.voices.iter().any(Voice::is_active)
    }

    /// How many voices are sounding, those being cut short excepted.
    #[must_use]
    pub fn active_voices(&self) -> usize {
        self.active_count() as usize
    }

    fn rebuild_routes(&mut self) {
        let values = &self.values;
        self.matrix.set_routes((0..MOD_SLOTS).map(|slot| {
            let at = id::MOD1_SRC + slot * 3;
            (
                values.user(at) as usize,
                values.user(at + 1) as usize,
                values.user(at + 2),
            )
        }));
        self.routes_dirty = false;
    }

    // ——— the matrix, from the synth's side ———

    /// `modMatrix.getValue (p)`: a parameter as the whole synth sees it.
    fn value(&mut self, param: usize) -> f32 {
        let spec = &PARAMS[param];
        let base = self.values.normalised(param);
        let dst = DESTINATION_OF[param];
        let v = if dst >= 0 {
            let active = self.active_voice.map(|v| &self.voices[v].modulation.values);
            self.matrix.value_global(dst as usize, base, active, true)
        } else {
            base
        };
        spec.conversion.apply(spec.range.convert_from_0to1(v))
    }

    fn voice_started(&mut self, voice: usize) {
        self.next_age += 1;
        self.voices[voice].modulation.age = self.next_age;
        self.active_voice = Some(voice);
    }

    fn voice_stopped(&mut self, voice: usize) {
        if self.active_voice == Some(voice) {
            let mut youngest = 0;
            self.active_voice = None;
            for (i, v) in self.voices.iter().enumerate() {
                if i == voice {
                    continue;
                }
                if v.is_active() && v.modulation.age > youngest {
                    youngest = v.modulation.age;
                    self.active_voice = Some(i);
                }
            }
        }
    }

    // ——— voices ———

    fn with_context<R>(&mut self, voice: usize, f: impl FnOnce(&mut Voice, &mut Context) -> R) -> R {
        let tables = self.tables.as_deref();
        let wt = |i: usize| {
            tables.and_then(|t| {
                t.oscillator(i, self.values.user(if i == 0 { id::WT1 } else { id::WT2 }))
            })
        };
        let mut ctx = Context {
            values: &self.values,
            matrix: &self.matrix,
            tables: [wt(0), wt(1)],
            analog: tables.map(TableSet::analog),
            noise: &mut self.noise,
            rng: &mut self.rng,
            system: &mut self.system,
            bpm: self.bpm,
        };
        f(&mut self.voices[voice], &mut ctx)
    }

    fn start_voice(&mut self, voice: usize, note: Note) {
        self.update_glide(voice);
        self.voices[voice].note = note;
        self.voices[voice].note_on_time = self.note_on_counter;
        self.note_on_counter = self.note_on_counter.wrapping_add(1);
        self.voice_started(voice);
        self.with_context(voice, Voice::note_started);
    }

    fn stop_voice(&mut self, voice: usize, note: Note, tail: bool) {
        self.voices[voice].note = note;
        if self.voices[voice].note_stopped(tail) {
            self.voice_stopped(voice);
        }
    }

    fn retrigger_voice(&mut self, voice: usize, note: Note) {
        self.update_glide(voice);
        self.voices[voice].note = note;
        self.with_context(voice, Voice::note_retriggered);
    }

    fn update_glide(&mut self, voice: usize) {
        let glide = GlideInfo {
            from_note: self.last_note,
            glissando: self.glissando,
            portamento: self.portamento,
            rate: self.glide_rate,
        };
        self.voices[voice].glide = if self.glissando || self.portamento {
            if self.last_note < 0.0 || (self.legato && self.notes.len <= 1) {
                GlideInfo::default()
            } else {
                glide
            }
        } else {
            GlideInfo::default()
        };
    }

    fn active_count(&self) -> i32 {
        self.voices
            .iter()
            .filter(|v| v.is_active() && !v.fast_kill)
            .count() as i32
    }

    /// `findVoiceToSteal`: the oldest note on the same key, else the
    /// oldest released, else the oldest not held, else the oldest — never
    /// the lowest or highest held note while another will do.
    fn voice_to_steal(&self, note: &Note) -> Option<usize> {
        let mut usable: [usize; VOICE_SLOTS] = [0; VOICE_SLOTS];
        let mut count = 0;
        let mut low: Option<usize> = None;
        let mut top: Option<usize> = None;
        for (i, v) in self.voices.iter().enumerate() {
            if !v.is_active() || v.fast_kill {
                continue;
            }
            usable[count] = i;
            count += 1;
            if !v.is_playing_but_released() {
                let n = v.note.initial_note;
                if low.is_none_or(|l| n < self.voices[l].note.initial_note) {
                    low = Some(i);
                }
                if top.is_none_or(|t| n > self.voices[t].note.initial_note) {
                    top = Some(i);
                }
            }
        }
        let usable = &mut usable[..count];
        usable.sort_unstable_by_key(|&i| self.voices[i].note_on_time);
        if top == low {
            top = None;
        }
        let protected = |i: usize| Some(i) == low || Some(i) == top;
        if note.is_valid() {
            if let Some(&i) = usable
                .iter()
                .find(|&&i| self.voices[i].note.initial_note == note.initial_note)
            {
                return Some(i);
            }
        }
        if let Some(&i) = usable
            .iter()
            .find(|&&i| !protected(i) && self.voices[i].is_playing_but_released())
        {
            return Some(i);
        }
        if let Some(&i) = usable
            .iter()
            .find(|&&i| !protected(i) && self.voices[i].note.key != KeyState::Down)
        {
            return Some(i);
        }
        if let Some(&i) = usable.iter().find(|&&i| !protected(i)) {
            return Some(i);
        }
        top.or(low)
    }

    /// `noteAdded`.
    fn note_added(&mut self, note: Note) {
        if self.mono {
            self.note_stack.push(note);
            if self.voices[0].is_active() {
                self.retrigger_voice(0, note);
            } else {
                self.start_voice(0, note);
            }
            return;
        }
        let Some(free) = self.voices.iter().position(|v| !v.is_active()) else {
            return;
        };
        if self.active_count() >= self.num_voices {
            if let Some(steal) = self.voice_to_steal(&note) {
                self.voices[steal].fast_kill = true;
                let playing = self.voices[steal].note;
                self.stop_voice(steal, playing, true);
            }
        }
        self.start_voice(free, note);
    }

    /// `noteReleased`.
    fn note_released(&mut self, note: Note) {
        if self.mono {
            self.note_released_mono(note, self.glissando || self.portamento);
        }
        for i in 0..self.voices.len() {
            if self.voices[i].is_playing(&note) {
                self.stop_voice(i, note, true);
            }
        }
    }

    fn note_released_mono(&mut self, note: Note, glide: bool) {
        let Some(at) = self.note_stack.position(note.id) else {
            return;
        };
        let current = at == self.note_stack.len - 1;
        self.note_stack.remove(at);
        if self.note_stack.len == 0 {
            if glide {
                if self.voices[0].is_active() {
                    self.stop_voice(0, note, true);
                }
            } else {
                for i in 0..self.voices.len() {
                    if self.voices[i].is_playing(&note) {
                        self.stop_voice(i, note, true);
                    }
                }
            }
        } else if current {
            if let Some(last) = self.note_stack.last() {
                if self.voices[0].is_active() {
                    self.retrigger_voice(0, last);
                } else {
                    self.start_voice(0, last);
                }
            }
        }
    }

    /// `MPEInstrument::noteOn`.
    fn instrument_note_on(&mut self, number: u8, velocity: MpeValue) {
        let mut note = Note::new(1, number, velocity);
        note.pitchbend = self.last_pitchbend;
        note.total_pitchbend =
            f64::from(note.pitchbend.signed() * self.pitchbend_range as f32);
        self.release_note(note.id);
        self.notes.push(note);
        self.note_added(note);
    }

    /// `MPEInstrument::noteOff`.
    fn instrument_note_off(&mut self, number: u8) {
        if self.notes.len == 0 {
            return;
        }
        self.release_note((1 << 7) + u16::from(number));
    }

    /// A held note let go: the listeners hear of it while it is still
    /// among the playing notes, and it leaves them after, as
    /// `MPEInstrument` does — a mono glide back to the note below counts
    /// the one being released.
    fn release_note(&mut self, id: u16) {
        if let Some(at) = self.notes.position(id) {
            let mut note = self.notes.as_slice()[at];
            note.key = KeyState::Off;
            self.note_released(note);
            if let Some(at) = self.notes.position(id) {
                self.notes.remove(at);
            }
        }
    }

    /// `releaseAllNotes`.
    fn release_all_notes(&mut self) {
        // Every note stays among the playing ones until all have been
        // released, as in `MPEInstrument::releaseAllNotes`.
        for at in (0..self.notes.len).rev() {
            let mut note = self.notes.as_slice()[at];
            note.key = KeyState::Off;
            self.note_released(note);
        }
        self.notes.len = 0;
    }

    /// `turnOffAllVoices`.
    pub fn turn_off_all_voices(&mut self, tail: bool) {
        for i in 0..self.voices.len() {
            self.voices[i].note.key = KeyState::Off;
            if self.voices[i].note_stopped(tail) {
                self.voice_stopped(i);
            }
        }
        self.release_all_notes();
        self.note_stack.len = 0;
        self.last_note = -1.0;
    }

    fn pitch_wheel(&mut self, value: i32) {
        let value = MpeValue(value.clamp(0, 16383));
        self.last_pitchbend = value;
        for at in (0..self.notes.len).rev() {
            let note = &mut self.notes.notes[at];
            if note.pitchbend != value {
                note.pitchbend = value;
                note.total_pitchbend = f64::from(value.signed() * self.pitchbend_range as f32);
                let changed = *note;
                for v in &mut self.voices {
                    if v.is_playing(&changed) {
                        v.note = changed;
                        v.pitchbend_changed();
                    }
                }
            }
        }
        let range = self.pitchbend_range as f32;
        for v in &mut self.voices {
            if v.is_playing_but_released() {
                v.note.total_pitchbend = f64::from(value.signed() * range);
            }
        }
        self.matrix
            .set_mono(src::PITCH_BEND, value.0 as f32 / 8192.0 - 1.0);
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::NoteOn { note, velocity } if velocity > 0 => {
                self.instrument_note_on(note, MpeValue::from_7bit(i32::from(velocity)));
            }
            Event::NoteOn { note, .. } | Event::NoteOff { note } => self.instrument_note_off(note),
            Event::PitchWheel(value) => self.pitch_wheel(value),
            Event::Controller { cc, value } => {
                if cc <= 119 {
                    self.matrix
                        .set_mono(src::CC0 + usize::from(cc), f32::from(value) / 127.0);
                }
            }
            Event::Choke { note } => {
                let id = (1 << 7) + u16::from(note);
                self.instrument_note_off(note);
                for i in 0..self.voices.len() {
                    if self.voices[i].is_active() && self.voices[i].note.id == id {
                        let playing = self.voices[i].note;
                        self.stop_voice(i, playing, false);
                    }
                }
            }
        }
    }

    fn render_voices(&mut self, left: &mut [f32], right: &mut [f32]) {
        for i in 0..self.voices.len() {
            if self.voices[i].is_active() {
                let freed = self.with_context(i, |v, ctx| v.render(ctx, left, right));
                if freed {
                    self.voice_stopped(i);
                }
            }
        }
    }

    /// `gin::Synthesiser::renderNextBlock` over `[start, start + n)`.
    fn render_next_block(
        &mut self,
        left: &mut [f32],
        right: &mut [f32],
        events: &[(u32, Event)],
        start: usize,
        n: usize,
    ) {
        self.note_on_index = -1;
        self.note_off_index = -1;
        let before = self.notes.len;
        let end = start + n;
        let mut prev = start;
        for &(time, event) in events {
            let time = time as usize;
            if time < start {
                continue;
            }
            if time >= end {
                break;
            }
            let min = if prev == start { 1 } else { MAX_BLOCK };
            if time >= prev + min {
                self.render_voices(&mut left[prev..time], &mut right[prev..time]);
                prev = time;
            }
            self.handle_event(event);
        }
        if prev < end {
            self.render_voices(&mut left[prev..end], &mut right[prev..end]);
        }

        let in_range = |&&(time, _): &&(u32, Event)| (start..end).contains(&(time as usize));
        if before == 0 && self.notes.len > 0 {
            if let Some((time, _)) = events
                .iter()
                .filter(in_range)
                .find(|(_, e)| e.is_note_on())
            {
                self.note_on_index = *time as i32 - start as i32;
            }
        }
        if before > 0 && self.notes.len == 0 {
            self.note_off_index = 0;
            for (time, _) in events
                .iter()
                .filter(in_range)
                .filter(|(_, e)| e.is_note_off())
            {
                // The original sets the note-on index here, not the
                // note-off: the gate reopens where the last key came up.
                self.note_on_index = *time as i32 - start as i32;
            }
        }
        if let Some(newest) = self
            .voices
            .iter()
            .filter(|v| v.is_active())
            .max_by_key(|v| v.note_on_time)
        {
            self.last_note = newest.current_note();
        }
    }

    /// The tempo a synced LFO, step LFO, gate or delay follows.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm;
    }

    fn beat_seconds(&self, index: usize) -> f32 {
        DURATIONS[index.min(DURATIONS.len() - 1)].to_seconds(self.bpm)
    }

    /// `WavetableAudioProcessor::updateParams`: the synth-wide modulators
    /// and the effects, once a slice.
    fn update_params(&mut self, block: usize) {
        for i in 0..3 {
            let base = id::LFO[i];
            if self.values.is_on(base + id::lfo::ENABLE) {
                let frequency = if self.values.user(base + id::lfo::SYNC) > 0.0 {
                    1.0 / self.beat_seconds(self.values.user(base + id::lfo::BEAT) as usize)
                } else {
                    self.value(base + id::lfo::RATE)
                };
                let params = LfoParams {
                    shape: Shape::from_index(self.values.user(base + id::lfo::WAVE) as i32),
                    frequency,
                    phase: self.value(base + id::lfo::PHASE),
                    offset: self.value(base + id::lfo::OFFSET),
                    depth: self.value(base + id::lfo::DEPTH),
                    delay: 0.0,
                    fade: 0.0,
                };
                self.mono_lfos[i].set_parameters(params, &mut self.system);
                self.mono_lfos[i].process(block);
                let out = self.mono_lfos[i].output();
                self.matrix.set_mono(src::MONO_LFO[i], out);
            } else {
                self.matrix.set_mono(src::MONO_LFO[i], 0.0);
            }
        }

        if self.values.is_on(id::SLFO_ENABLE) {
            let frequency = 1.0 / self.beat_seconds(self.values.user(id::SLFO_BEAT) as usize);
            self.mono_step.set_freq(frequency);
            let n = self.values.user(id::SLFO_LENGTH) as usize;
            self.mono_step.set_num_points(n);
            for i in (0..n).rev() {
                self.mono_step.set_point(i, self.values.user(id::SLFO_STEP1 + i));
            }
            self.mono_step.process(block);
            let out = self.mono_step.output();
            self.matrix.set_mono(src::MONO_STEP, out);
        } else {
            self.matrix.set_mono(src::MONO_STEP, 0.0);
        }

        if self.values.is_on(id::GATE_ENABLE) {
            let frequency = 1.0 / self.beat_seconds(self.values.user(id::GATE_BEAT) as usize);
            let n = self.values.user(id::GATE_LENGTH) as usize;
            self.gate.set_length(n);
            for i in 0..n {
                let l = self.values.is_on(id::GATE_L1 + i * 2);
                let r = self.values.is_on(id::GATE_L1 + i * 2 + 1);
                self.gate.set_step(i, l, r);
            }
            self.gate.set_frequency(frequency);
            let attack = self.value(id::GATE_ATTACK);
            self.gate.set_attack(attack);
            let release = self.value(id::GATE_RELEASE);
            self.gate.set_release(release);
        }

        if self.values.is_on(id::CH_ENABLE) {
            let delay = self.value(id::CH_DELAY);
            let rate = self.value(id::CH_SPEED);
            let depth = self.value(id::CH_DEPTH);
            let width = self.value(id::CH_WIDTH);
            let mix = self.value(id::CH_MIX);
            self.chorus.set_params(delay, rate, depth, width, mix);
        }

        if self.values.is_on(id::DS_ENABLE) {
            match self.values.user(id::DIST_MODE) as i32 {
                0 => self.distortion = self.value(id::DS_AMOUNT),
                1 => {
                    let p = [
                        self.value(id::CRUSH_REZ),
                        self.value(id::CRUSH_RATE),
                        self.value(id::CRUSH_HARD),
                        self.value(id::CRUSH_MIX),
                    ];
                    for (i, v) in p.into_iter().enumerate() {
                        self.bitcrusher.set_parameter(i, v);
                    }
                }
                2 => {
                    let p = [
                        self.value(id::FIRE_GAIN),
                        self.value(id::FIRE_TONE),
                        self.value(id::FIRE_OUTPUT),
                        self.value(id::FIRE_MIX),
                    ];
                    for (i, v) in p.into_iter().enumerate() {
                        self.fire_amp.set_parameter(i, v);
                    }
                }
                _ => {
                    let p = [
                        self.value(id::GRIND_GAIN),
                        self.value(id::GRIND_TONE),
                        self.value(id::GRIND_OUTPUT),
                        self.value(id::GRIND_MIX),
                    ];
                    for (i, v) in p.into_iter().enumerate() {
                        self.grind_amp.set_parameter(i, v);
                    }
                }
            }
        }

        if self.values.is_on(id::DL_ENABLE) {
            if self.values.is_on(id::DL_SYNC) {
                let beat = self.value(id::DL_BEAT) as usize;
                let seconds = self.beat_seconds(beat);
                self.values.set(id::DL_DELAY, seconds);
                let delay = self.values.user(id::DL_DELAY);
                let mix = self.value(id::DL_MIX);
                let fb = self.value(id::DL_FB);
                let cf = self.value(id::DL_CF);
                self.delay.set_params(delay, mix, fb, cf);
            } else {
                let time = self.value(id::DL_TIME);
                self.values.set(id::DL_DELAY, time);
                let delay = self.value(id::DL_DELAY);
                let mix = self.value(id::DL_MIX);
                let fb = self.value(id::DL_FB);
                let cf = self.value(id::DL_CF);
                self.delay.set_params(delay, mix, fb, cf);
            }
        }

        if self.values.is_on(id::RV_ENABLE) {
            let size = self.value(id::RV_SIZE);
            self.reverb.set_size(size);
            let decay = self.value(id::RV_DECAY);
            self.reverb.set_decay(decay);
            let lowpass = self.value(id::RV_LOWPASS);
            self.reverb.set_lowpass(lowpass);
            let damping = self.value(id::RV_DAMPING);
            self.reverb.set_damping(damping);
            let predelay = self.value(id::RV_PREDELAY);
            self.reverb.set_predelay(predelay);
            let mix = self.value(id::RV_MIX);
            self.reverb.set_mix(mix);
        }

        let level = self.value(id::LEVEL);
        self.output_gain.set_gain(level);
    }

    fn apply_effect(&mut self, effect: i32, left: &mut [f32], right: &mut [f32]) {
        match effect {
            0 if self.values.is_on(id::GATE_ENABLE) => {
                self.gate
                    .process(left, right, self.note_on_index, self.note_off_index);
            }
            1 if self.values.is_on(id::CH_ENABLE) => self.chorus.process(left, right),
            2 if self.values.is_on(id::DS_ENABLE) => match self.values.user(id::DIST_MODE) as i32 {
                0 => distort(left, right, self.distortion),
                1 => self.bitcrusher.process(left, right, self.sample_rate),
                2 => self.fire_amp.process(left, right, self.sample_rate),
                _ => self.grind_amp.process(left, right, self.sample_rate),
            },
            3 if self.values.is_on(id::DL_ENABLE) => {
                if self.values.is_on(id::DL_SYNC) {
                    self.delay.process(left, right);
                } else {
                    self.delay.process_smoothed(left, right);
                }
            }
            4 if self.values.is_on(id::RV_ENABLE) => self.reverb.process(left, right),
            _ => {}
        }
    }

    /// `processBlock`: `left`/`right` overwritten with what the synth
    /// plays over them, `events` (sorted by time, times within the block)
    /// handled as they come.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32], events: &[(u32, Event)]) {
        if self.routes_dirty {
            self.rebuild_routes();
        }
        let mono = self.values.is_on(id::MONO);
        if self.preset_loaded || self.last_mono != mono {
            self.preset_loaded = false;
            self.last_mono = mono;
            self.delay.reset();
            self.reverb.reset();
            self.turn_off_all_voices(false);
        }

        let range = self.values.user(id::PITCH_BEND) as i32;
        if range != self.pitchbend_range {
            self.release_all_notes();
            self.pitchbend_range = range;
        }

        left.fill(0.0);
        right.fill(0.0);

        self.mono = self.last_mono;
        self.legato = self.values.is_on(id::LEGATO);
        self.glissando = self.values.proc(id::GLIDE_MODE) == 1.0;
        self.portamento = self.values.proc(id::GLIDE_MODE) == 2.0;
        self.glide_rate = self.values.proc(id::GLIDE_RATE);
        self.num_voices = self.values.proc(id::VOICES) as i32;

        let total = left.len().min(right.len());
        let mut pos = 0;
        while pos < total {
            let n = (total - pos).min(MAX_BLOCK);
            self.update_params(n);
            self.render_next_block(left, right, events, pos, n);
            let (l, r) = (&mut left[pos..pos + n], &mut right[pos..pos + n]);
            for slot in 0..5 {
                let effect = self.values.user(id::FX_ORDER1 + slot) as i32;
                self.apply_effect(effect, l, r);
            }
            self.output_gain.process(l, r);
            self.matrix.finish_block(n);
            pos += n;
        }
    }

    /// Every voice stopped and every effect's tail emptied, now: Choke
    /// All. Allocates nothing.
    pub fn silence(&mut self) {
        self.turn_off_all_voices(false);
        self.reset();
    }
}
