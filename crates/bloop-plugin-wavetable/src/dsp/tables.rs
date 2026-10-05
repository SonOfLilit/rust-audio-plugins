//! Band-limited lookup tables: Gin's `BandLimitedLookupTable`,
//! `Wavetable` and `BandLimitedLookupTables`
//! (`gin_dsp/dsp/gin_bandlimitedlookuptable.{h,cpp}`, `gin_wavetable.h`,
//! `gin_wtoscillators.cpp`'s `loadWavetables`).
//!
//! A table holds one waveform several times over, each copy with the
//! harmonics removed that would alias for the notes it serves — one copy
//! per `notes_per_table` notes. Building them allocates and takes a
//! while, so it happens on the UI thread; the audio thread only reads.

use super::fft::{Complex, Fft};
use super::random::{PinkNoise, WhiteNoise};
use bloop_plugin_socalabs::shared::{math, midi_note_in_hertz};
use core::f32::consts::PI;

/// `gin::BandLimitedLookupTable`: one waveform, band-limited once per
/// `notes_per_table` notes, each copy `table_size + 1` long so that
/// interpolation can read one past the end without wrapping.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookupTable {
    data: Vec<f32>,
    table_size: usize,
    notes_per_table: i32,
    count: usize,
}

/// The notes a table is made for: `notes_per_table + 0.5` and on by
/// `notes_per_table` below 127, the first clamped to 126.5.
fn table_notes(notes_per_table: i32) -> impl Iterator<Item = f32> {
    let first = (notes_per_table as f32 + 0.5).min(126.5);
    let step = notes_per_table as f32;
    core::iter::successors(Some(first), move |note| Some(note + step)).take_while(|n| *n < 127.0)
}

impl LookupTable {
    /// `reset (function, sampleRate, notesPerTable, tableSize)`: each copy
    /// worked out from `function (phase, freq, sampleRate)`.
    #[must_use]
    pub fn from_function(
        function: fn(f32, f32, f32) -> f32,
        sample_rate: f32,
        notes_per_table: i32,
        table_size: usize,
    ) -> Self {
        let mut table = Self {
            data: Vec::new(),
            table_size,
            notes_per_table,
            count: 0,
        };
        for note in table_notes(notes_per_table) {
            let freq = midi_note_in_hertz(note);
            let start = table.data.len();
            table
                .data
                .extend((0..table_size).map(|i| function(i as f32 / table_size as f32, freq, sample_rate)));
            table.data.push(table.data[start]);
            table.count += 1;
        }
        table
    }

    /// `loadFromBuffer (fft, playbackSampleRate, buffer, fileSampleRate,
    /// notesPerTable)`: one frame of a wavetable, each copy low-passed in
    /// the frequency domain to what `playback_rate` can hold at its notes.
    #[must_use]
    pub fn from_frame(
        fft: &Fft,
        playback_rate: f32,
        frame: &[f32],
        file_rate: f32,
        notes_per_table: i32,
    ) -> Self {
        let sz = frame.len();
        let duration = sz as f32 / file_rate;
        let base_freq = 1.0 / duration;
        let mut table = Self {
            data: Vec::with_capacity(21 * (sz + 1)),
            table_size: sz,
            notes_per_table,
            count: 0,
        };

        let mut time: Vec<Complex> = frame.iter().map(|&re| Complex { re, im: 0.0 }).collect();
        let mut freq = vec![Complex::default(); sz];
        fft.perform(&time, &mut freq, false);

        let index_to_freq = |i: usize| {
            let bin = i.min(sz - i);
            bin as f32 * (file_rate / sz as f32)
        };

        for note in table_notes(notes_per_table) {
            let note_freq = midi_note_in_hertz(note);
            if note_freq < base_freq {
                table.data.extend_from_slice(frame);
                table.data.push(frame[0]);
            } else {
                let ratio = note_freq / base_freq;
                for (i, bin) in freq.iter_mut().enumerate() {
                    if index_to_freq(i) * ratio > playback_rate / 2.0 {
                        *bin = Complex::default();
                    }
                }
                fft.perform(&freq, &mut time, true);
                table.data.extend(time.iter().map(|c| c.re));
                table.data.push(time[0].re);
            }
            table.count += 1;
        }
        table
    }

    /// How many band-limited copies it holds.
    #[must_use]
    pub fn copies(&self) -> usize {
        self.count
    }

    #[must_use]
    pub fn table_size(&self) -> usize {
        self.table_size
    }

    /// Copy `index`, all `table_size + 1` samples of it.
    #[must_use]
    pub fn copy(&self, index: usize) -> &[f32] {
        let len = self.table_size + 1;
        &self.data[index * len..(index + 1) * len]
    }

    /// `tableIndexForNote`.
    #[inline]
    #[must_use]
    pub fn index_for_note(&self, note: f32) -> usize {
        let index = ((f64::from(note) - 0.5) / f64::from(self.notes_per_table)) as i32;
        index.clamp(0, self.count as i32 - 1) as usize
    }

    /// `processLinear (note, phase)`.
    #[inline]
    #[must_use]
    pub fn process_linear(&self, note: f32, phase: f32) -> f32 {
        self.linear(self.index_for_note(note), phase)
    }

    /// `getLinear (tableIndex, phase)`.
    #[inline]
    #[must_use]
    pub fn linear(&self, index: usize, phase: f32) -> f32 {
        let size = self.table_size as i32;
        let scaled = phase * size as f32;
        let pos = (scaled as i32).min(size - 1);
        let frac = scaled - pos as f32;
        let base = index * (self.table_size + 1) + pos as usize;
        (self.data[base] * (1.0 - frac)) + (self.data[base + 1] * frac)
    }

    /// How many bytes its samples take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len() * core::mem::size_of::<f32>()
    }
}

/// `gin::Wavetable`: a wavetable's frames, each a [`LookupTable`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Wavetable {
    frames: Vec<LookupTable>,
}

impl Wavetable {
    /// `loadWavetables (bllt, playbackSampleRate, buffer, fileSampleRate,
    /// tableSize)`: `samples` cut into frames of `table_size`, each
    /// band-limited for `playback_rate`, six notes to a copy.
    #[must_use]
    pub fn load(playback_rate: f64, samples: &[f32], file_rate: f64, table_size: usize) -> Self {
        let order = table_size.trailing_zeros();
        let fft = Fft::new(order);
        let frames = samples
            .chunks_exact(table_size)
            .map(|frame| {
                LookupTable::from_frame(&fft, playback_rate as f32, frame, file_rate as f32, 6)
            })
            .collect();
        Self { frames }
    }

    /// Frames built elsewhere, as they are.
    #[must_use]
    pub fn from_frames(frames: Vec<LookupTable>) -> Self {
        Self { frames }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    #[inline]
    #[must_use]
    pub fn frame(&self, index: usize) -> &LookupTable {
        &self.frames[index]
    }

    /// How many bytes its samples take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.frames.iter().map(LookupTable::bytes).sum()
    }
}

// ——— the analog waveforms ———

/// `gin::sine`.
fn sine(phase: f32, _: f32, _: f32) -> f32 {
    math::sinf(phase * 2.0 * PI)
}

/// `gin::triangle`: the odd harmonics, each `±1 / k²`, up to Nyquist.
fn triangle(phase: f32, freq: f32, sample_rate: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut k = 1.0f32;
    while freq * k < sample_rate / 2.0 {
        // `std::pow (-1.0f, (k - 1) / 2.0f)`: an exact ±1 for every odd k.
        let sign = if ((k - 1.0) / 2.0) % 2.0 == 0.0 {
            1.0
        } else {
            -1.0
        };
        sum += sign / (k * k) * math::sinf(k * (phase * 2.0 * PI));
        k += 2.0;
    }
    8.0 / (PI * PI) * sum
}

/// `gin::sawUp`: every harmonic, alternating, `1 / k`.
fn saw_up(phase: f32, freq: f32, sample_rate: f32) -> f32 {
    -2.0 / PI * saw_sum(phase, freq, sample_rate)
}

/// `gin::sawDown`.
fn saw_down(phase: f32, freq: f32, sample_rate: f32) -> f32 {
    2.0 / PI * saw_sum(phase, freq, sample_rate)
}

fn saw_sum(phase: f32, freq: f32, sample_rate: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut k = 1i32;
    let mut fk = 1.0f32;
    while freq * (k as f32) < sample_rate / 2.0 {
        let odd_even = if k % 2 == 0 { 1.0f32 } else { -1.0 };
        sum += odd_even * math::sinf(fk * (phase * 2.0 * PI)) / fk;
        k += 1;
        fk += 1.0;
    }
    sum
}

/// `gin::Wave`, as the sub oscillator and the noise use it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    Sine,
    Triangle,
    SawUp,
    Pulse,
    WhiteNoise,
    PinkNoise,
}

/// `gin::BandLimitedLookupTables` at its defaults (three notes to a copy,
/// 2048 samples): sine, both saws and triangle at one sample rate. The
/// noise generators it also owns are each voice's own here
/// ([`NoiseSource`]); see there for why.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalogTables {
    sample_rate: f64,
    sine: LookupTable,
    saw_up: LookupTable,
    saw_down: LookupTable,
    triangle: LookupTable,
}

impl AnalogTables {
    /// The tables for `sample_rate`. Slow — a second or so, summing
    /// thousands of sines per sample for the low notes — so build once
    /// per rate and share.
    #[must_use]
    pub fn new(sample_rate: f64) -> Self {
        let rate = sample_rate as f32;
        Self {
            sample_rate,
            sine: LookupTable::from_function(sine, rate, 64, 2048),
            saw_up: LookupTable::from_function(saw_up, rate, 3, 2048),
            saw_down: LookupTable::from_function(saw_down, rate, 3, 2048),
            triangle: LookupTable::from_function(triangle, rate, 3, 2048),
        }
    }

    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn index(table: &LookupTable, count: usize, note: f32) -> usize {
        let index = ((f64::from(note) - 0.5) / f64::from(table.notes_per_table)) as i32;
        index.clamp(0, count as i32 - 1) as usize
    }

    fn index_f32(table: &LookupTable, count: usize, note: f32) -> usize {
        let index = ((note - 0.5) / table.notes_per_table as f32) as i32;
        index.clamp(0, count as i32 - 1) as usize
    }

    /// `processPulse (note, phase, pw)`: two saws, half a width apart.
    fn pulse(&self, note: f32, phase: f32, pw: f32) -> f32 {
        let pw = pw.clamp(0.01, 0.99);
        let mut up = phase + 0.5 * pw;
        let mut down = phase - 0.5 * pw;
        if up >= 1.0 {
            up -= 1.0;
        }
        if down < 0.0 {
            down += 1.0;
        }
        if down >= 1.0 {
            down -= 1.0;
        }
        let count = self.saw_down.count.min(self.saw_up.count);
        let index = Self::index_f32(&self.saw_up, count, note);
        let dc = 2.0 * pw - 1.0;
        let s1 = self.saw_down.linear(index, down);
        let s2 = self.saw_up.linear(index, up);
        -dc + s1 + s2
    }

    /// `process (wave, note, phase, pw)` for the shapes with a table.
    #[inline]
    #[must_use]
    pub fn process(&self, wave: Wave, note: f32, phase: f32, pw: f32) -> f32 {
        match wave {
            Wave::Sine => self.sine.linear(0, phase),
            Wave::Triangle => self
                .triangle
                .linear(Self::index(&self.triangle, self.triangle.count, note), phase),
            Wave::SawUp => self
                .saw_up
                .linear(Self::index(&self.saw_up, self.saw_up.count, note), phase),
            Wave::Pulse => self.pulse(note, phase, pw),
            Wave::WhiteNoise | Wave::PinkNoise => 0.0,
        }
    }

    /// How many bytes its samples take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.sine.bytes() + self.saw_up.bytes() + self.saw_down.bytes() + self.triangle.bytes()
    }
}

/// The noise generators `gin::BandLimitedLookupTables` owns. The original
/// shares one pair between every voice of the synth, through its single
/// table set, so each voice's noise draws the next samples of the same
/// stream in the order the voices render; bloop keeps that, one pair per
/// synth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseSource {
    pub white: WhiteNoise,
    pub pink: PinkNoise,
}

impl NoiseSource {
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self {
            white: WhiteNoise::new(seed),
            pink: PinkNoise::default(),
        }
    }
}
