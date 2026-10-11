//! The wavetables bloop ships, and turning one into the band-limited
//! frames an oscillator reads.
//!
//! A table is built in one of two forms ([`Table`]), by the quality a
//! render asks for:
//!
//! - **exact** ([`Quality::Render`]): what Wavetable's `loadWaveTable`
//!   builds, bit for bit — an FFT and twenty-one inverse FFTs a frame, a
//!   band-limited copy every six notes, each the whole frame long: about
//!   170 kB a frame at 2048 samples, 44 MB for a 256-frame table;
//! - **compact** ([`Quality::Live`]): a copy an octave, shorter as it
//!   holds fewer harmonics, in 16 bits (`dsp::tables::CompactFrame`):
//!   about a sixth of that.
//!
//! Either is slow to build, so it happens on the UI thread, once per
//! table, sample rate and form, and the result is shared: [`tables_for`]
//! keeps what is in use and hands every synth that asks the same `Arc`.
//! The synth holds a [`TableSet`] — the two oscillators' tables and the
//! analog shapes the sub reads — handed to it whole.

use crate::dsp::tables::{AnalogTables, CompactWavetable, Table, Wavetable};
pub use bloop_plugin_api::Quality;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// How a bundled table's audio is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// 16-bit FLAC, 2048-sample frames: Wavetable's `.wt2048`, as it ships.
    Flac,
    /// The same 16-bit samples with each frame stored as its difference
    /// from the frame before (the first as itself), in a 24-bit FLAC
    /// (`.wtd`). A table that morphs a little from frame to frame packs a
    /// good deal smaller this way; [`decode`] sums the frames back up.
    /// `tools/import.py` keeps this form only for a table it makes
    /// smaller, and only once it decodes to the original bit for bit.
    FlacDelta,
    /// A WAV a preset carried inside itself.
    Wav,
}

/// One wavetable of the catalogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The folder Wavetable files it under.
    pub bank: &'static str,
    pub name: &'static str,
    pub format: Format,
    /// Samples a frame.
    pub frame: usize,
    /// Whether the web build carries its audio (every other build does).
    pub on_web: bool,
    /// Where its file sits in the blob; `None` where this build does not
    /// carry it (the web build carries a subset).
    audio: Option<(usize, usize)>,
}

include!(concat!(env!("OUT_DIR"), "/catalogue.rs"));

static BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wavetables.blob"));

impl Entry {
    /// Its file, if this build carries it.
    #[must_use]
    pub fn bytes(&self) -> Option<&'static [u8]> {
        self.audio.map(|(at, len)| &BLOB[at..at + len])
    }
}

/// Every wavetable, in the order the Wavetable knob steps through them.
#[must_use]
pub fn catalogue() -> &'static [Entry] {
    CATALOGUE
}

/// Every wavetable's name, in catalogue order: what the Wavetable knob
/// steps through.
#[must_use]
pub fn names() -> &'static [&'static str] {
    &NAMES
}

/// The catalogue entry called `name`.
#[must_use]
pub fn index_of(name: &str) -> Option<usize> {
    CATALOGUE.iter().position(|e| e.name == name)
}

/// Why a table could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// No catalogue entry at that number.
    NoSuchTable,
    /// The entry is listed but this build does not carry its audio.
    NotCarried,
    /// The file would not decode.
    Decode(String),
}


/// Table `index`'s samples, as the original reads them from its file:
/// the samples, their rate and how many make a frame.
fn source(index: usize) -> Result<(Vec<f32>, f64, usize), LoadError> {
    let entry = CATALOGUE.get(index).ok_or(LoadError::NoSuchTable)?;
    let bytes = entry.bytes().ok_or(LoadError::NotCarried)?;
    let (samples, file_rate) = decode(entry.format, entry.frame, bytes)?;
    Ok((samples, file_rate, entry.frame))
}

/// Table `index` decoded and band-limited for `sample_rate`: what
/// Wavetable's `loadWaveTable` builds, the exact form. Slow; UI thread.
///
/// # Errors
/// When there is no such table, this build lacks its audio, or it does not
/// decode.
pub fn load(index: usize, sample_rate: f64) -> Result<Wavetable, LoadError> {
    let (samples, file_rate, frame) = source(index)?;
    // A WAV reads only whole frames; a FLAC reads all, and the frames
    // come out the same.
    Ok(Wavetable::load(sample_rate, &samples, file_rate, frame))
}

/// Table `index` in the compact form, for `sample_rate`. Slow; UI thread.
///
/// # Errors
/// As [`load`].
pub fn load_compact(index: usize, sample_rate: f64) -> Result<CompactWavetable, LoadError> {
    let (samples, _, frame) = source(index)?;
    Ok(CompactWavetable::load(sample_rate, &samples, frame))
}

/// Table `index` in the form `quality` plays: exact for a render, compact
/// live.
///
/// # Errors
/// As [`load`].
pub fn build(index: usize, sample_rate: f64, quality: Quality) -> Result<Table, LoadError> {
    Ok(match quality {
        Quality::Render => Table::Exact(load(index, sample_rate)?),
        Quality::Live => Table::Compact(load_compact(index, sample_rate)?),
    })
}

/// A table's file, `frame` samples a frame, as the samples the original
/// reads from it, and their rate.
///
/// # Errors
/// When it does not decode, or a [`Format::FlacDelta`] holds a part frame
/// or sums to a sample past 16 bits.
pub fn decode(format: Format, frame: usize, bytes: &[u8]) -> Result<(Vec<f32>, f64), LoadError> {
    let extension = match format {
        Format::Flac | Format::FlacDelta => "flac",
        Format::Wav => "wav",
    };
    let decoded = bloop_sampler_engine::decode(bytes.to_vec(), Some(extension))
        .map_err(|e| LoadError::Decode(e.to_string()))?;
    let rate = f64::from(decoded.sample_rate());
    let samples = decoded.channel(0);
    if format != Format::FlacDelta {
        return Ok((samples.to_vec(), rate));
    }
    if frame == 0 || samples.len() % frame != 0 {
        return Err(LoadError::Decode(format!(
            "{} samples are not whole frames of {frame}",
            samples.len()
        )));
    }
    let mut out = Vec::with_capacity(samples.len());
    let mut sums = vec![0i32; frame];
    for deltas in samples.chunks_exact(frame) {
        for (sum, delta) in sums.iter_mut().zip(deltas) {
            // A 24-bit sample: a whole number of 2^-23.
            *sum += (f64::from(*delta) * 8_388_608.0).round() as i32;
            if !(-32768..=32767).contains(sum) {
                return Err(LoadError::Decode(format!("{sum} is past 16 bits")));
            }
            out.push(*sum as f32 / 32768.0);
        }
    }
    Ok((out, rate))
}

/// What a synth reads its oscillators' tables and its sub's shapes from.
#[derive(Debug)]
pub struct TableSet {
    sample_rate: f64,
    quality: Quality,
    oscillators: [Option<(u16, Arc<Table>)>; 2],
    analog: Arc<AnalogTables>,
}

impl TableSet {
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// The quality it was built for: [`Quality::Render`] holds the exact
    /// form, [`Quality::Live`] the compact.
    #[must_use]
    pub fn quality(&self) -> Quality {
        self.quality
    }

    /// Oscillator `osc`'s table, if it is the one `wanted` names.
    #[must_use]
    pub fn oscillator(&self, osc: usize, wanted: f32) -> Option<&Table> {
        self.oscillators[osc]
            .as_ref()
            .filter(|(index, _)| f32::from(*index) == wanted)
            .map(|(_, table)| &**table)
    }

    /// Which catalogue entries it holds for the two oscillators.
    #[must_use]
    pub fn indices(&self) -> [Option<u16>; 2] {
        [
            self.oscillators[0].as_ref().map(|(i, _)| *i),
            self.oscillators[1].as_ref().map(|(i, _)| *i),
        ]
    }

    /// Whether it holds `indices` at `sample_rate` in the form `quality`
    /// plays: what a synth that has it needs no new set for.
    #[must_use]
    pub fn holds(&self, indices: [u16; 2], sample_rate: f64, quality: Quality) -> bool {
        self.indices() == indices.map(Some)
            && (self.sample_rate - sample_rate).abs() <= f64::EPSILON
            && self.quality == quality
    }

    #[must_use]
    pub fn analog(&self) -> &AnalogTables {
        &self.analog
    }

    /// The samples it holds, in bytes, counting a table both oscillators
    /// share once.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let [a, b] = &self.oscillators;
        let one = a.as_ref().map_or(0, |(_, t)| t.bytes());
        let two = match (a, b) {
            (Some((_, x)), Some((_, y))) if Arc::ptr_eq(x, y) => 0,
            (_, Some((_, t))) => t.bytes(),
            _ => 0,
        };
        one + two + self.analog.bytes()
    }
}

/// Which table, at which rate, in which form.
type Key = (usize, u64, bool);

fn key(index: usize, rate: f64, quality: Quality) -> Key {
    (index, rate.to_bits(), quality == Quality::Render)
}

/// Built tables, shared while anything holds them, and the analog shapes
/// per rate, kept.
#[derive(Default)]
struct Cache {
    tables: HashMap<Key, Weak<Table>>,
    analog: HashMap<u64, Arc<AnalogTables>>,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: std::sync::OnceLock<Mutex<Cache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

fn lock() -> std::sync::MutexGuard<'static, Cache> {
    cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Table `index` at `rate` in `quality`'s form, shared if anything holds
/// it, built if not. Built with the cache unlocked, so that two threads
/// building different tables do not wait on each other; two building the
/// same one keep the first.
fn shared_table(index: usize, rate: f64, quality: Quality) -> Option<Arc<Table>> {
    let key = key(index, rate, quality);
    if let Some(table) = lock().tables.get(&key).and_then(Weak::upgrade) {
        return Some(table);
    }
    let built = Arc::new(build(index, rate, quality).ok()?);
    let mut cache = lock();
    if let Some(table) = cache.tables.get(&key).and_then(Weak::upgrade) {
        return Some(table);
    }
    cache.tables.insert(key, Arc::downgrade(&built));
    cache.tables.retain(|_, t| t.strong_count() > 0);
    Some(built)
}

/// The tables for wavetables `indices` at `sample_rate`, in the form
/// `quality` plays, built or shared. A table this build lacks is left
/// out, and its oscillator is silent. UI thread: slow the first time a
/// table, rate or form is asked for.
#[must_use]
pub fn tables_for(indices: [u16; 2], sample_rate: f64, quality: Quality) -> Arc<TableSet> {
    let analog = Arc::clone(
        lock()
            .analog
            .entry(sample_rate.to_bits())
            .or_insert_with(|| Arc::new(AnalogTables::new(sample_rate))),
    );
    let oscillators = indices.map(|index| {
        shared_table(usize::from(index), sample_rate, quality).map(|t| (index, t))
    });
    Arc::new(TableSet {
        sample_rate,
        quality,
        oscillators,
        analog,
    })
}
