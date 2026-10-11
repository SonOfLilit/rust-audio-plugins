//! Every Airwindows plug-in, compiled from Chris Johnson's C++.
//!
//! `bloop-plugin-airwindows` ports six of them to Rust, bit for bit. This
//! crate takes the rest of the LinuxVST collection as published: build.rs
//! compiles each plug-in's two .cpp files against a stand-in for the VST2
//! base class (cpp/audioeffectx.h), and [`Effect`] drives one through a C
//! ABI (cpp/api.cpp). Nothing of a plug-in is written here: its knobs, their
//! names, their defaults and how each reads are the plug-in's own, asked of
//! it.
//!
//! What a plug-in may do on the audio thread, it does without allocating
//! or locking: none of their `processReplacing` allocates, and the two
//! things that would — glibc's `rand()`, which takes a lock, and the
//! `std::set` every plug-in keeps of host capabilities — are replaced by
//! the stand-in (an LCG of each instance's own, and a set that holds
//! nothing). So [`Effect::clear`] can destroy a plug-in and construct it
//! again in the same memory on the audio thread.
//!
//! Not on wasm32: there is no C++ toolchain for it here, and the crate
//! builds there with no plug-ins ([`COUNT`] is 0).

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock, PoisonError};

use bloop_plugin_api::{
    DynamicsMeter, ParamInfo, Plugin, PluginCategory, PluginInfo, ProcessContext,
};

/// Who makes them, as the Browser's VENDOR filter shows it.
pub const VENDOR: &str = "Airwindows";

/// The Airwindows revision compiled: the one `vendor/airwindows` pins.
pub const AIRWINDOWS_REV: &str = "e718c9bcfcdd736deeddb08bffe6bce2aa8e0eea";

/// One compiled plug-in, as Airwindows describes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Listing {
    /// Its name, as its folder and its class are called.
    pub name: &'static str,
    /// The first category Airwindopedia lists it under ("Reverb",
    /// "Lo-Fi"…); empty for one it lists under none.
    pub category: &'static str,
    /// Its line in Airwindows' what.txt; empty for one that has none.
    pub about: &'static str,
}

mod catalog {
    use super::Listing;
    include!(concat!(env!("OUT_DIR"), "/catalog.rs"));
}

pub use catalog::{COUNT, SKIPPED};

const _: () = assert!(COUNT <= u16::MAX as usize, "an Id is a u16");

/// The longest an Airwindows delay (the Ambience category: TapeDelay2,
/// PitchDelay, Doublelay…) can be silent before its echo comes back, in
/// seconds: longer than any of their lines hold at 44.1 kHz and up.
const AMBIENCE_GAP_SECONDS: f32 = 3.0;

/// Which plug-in: an index into the catalog, so only ever one that was
/// compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(u16);

impl Id {
    /// Every compiled plug-in, by name.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "COUNT fits a u16 (asserted)"
    )]
    pub fn all() -> impl Iterator<Item = Id> + Clone {
        (0..COUNT).map(|at| Id(at as u16))
    }

    /// The plug-in called `name`, if one was compiled.
    #[must_use]
    pub fn named(name: &str) -> Option<Id> {
        Id::all().find(|id| id.name() == name)
    }

    /// Where it is among [`Id::all`], from 0: below [`COUNT`].
    #[must_use]
    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    #[must_use]
    pub fn listing(self) -> &'static Listing {
        &catalog::PLUGINS[usize::from(self.0)]
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        self.listing().name
    }

    /// Its knobs, as a fresh instance of it says: names, units and
    /// defaults. Asked once a process, of an instance built for it and
    /// dropped after.
    #[must_use]
    pub fn params(self) -> &'static [Param] {
        static PARAMS: [OnceLock<Box<[Param]>>; COUNT] = [const { OnceLock::new() }; COUNT];
        PARAMS[usize::from(self.0)].get_or_init(|| {
            let handle = Handle::new(self, 44_100.0);
            (0..handle.param_count())
                .map(|index| Param {
                    name: handle.text(index, ffi::param_name).trim().to_string(),
                    label: handle.text(index, ffi::param_label).trim().to_string(),
                    default: handle.get(index),
                })
                .collect()
        })
    }

    fn entry(self) -> i32 {
        i32::from(self.0)
    }
}

/// One knob of a plug-in, as it describes it. Every knob runs 0 to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// `getParameterName`.
    pub name: String,
    /// `getParameterLabel`: the unit its reading is followed by ("dB",
    /// "ms"), or empty.
    pub label: String,
    /// Where its constructor sets it.
    pub default: f32,
}

/// The C ABI of cpp/api.cpp, as safe calls on a live handle. On wasm32,
/// where nothing is compiled and so no handle is ever made, stand-ins.
mod ffi {
    use std::ffi::{c_char, c_void};

    /// How many bytes every text the ABI writes may take.
    pub const TEXT: usize = 64;

    #[cfg(not(target_arch = "wasm32"))]
    mod raw {
        use std::ffi::{c_char, c_void};
        extern "C" {
            pub fn bloop_aw_entry_size(entry: i32) -> std::ffi::c_ulong;
            pub fn bloop_aw_new(entry: i32, sample_rate: f64) -> *mut c_void;
            pub fn bloop_aw_free(handle: *mut c_void);
            pub fn bloop_aw_param_count(handle: *const c_void) -> i32;
            pub fn bloop_aw_get(handle: *mut c_void, index: i32) -> f32;
            pub fn bloop_aw_set(handle: *mut c_void, index: i32, value: f32);
            pub fn bloop_aw_param_name(handle: *mut c_void, index: i32, out: *mut c_char);
            pub fn bloop_aw_param_label(handle: *mut c_void, index: i32, out: *mut c_char);
            pub fn bloop_aw_param_display_at(
                handle: *mut c_void,
                index: i32,
                value: f32,
                out: *mut c_char,
            );
            pub fn bloop_aw_set_rate(handle: *mut c_void, sample_rate: f64);
            pub fn bloop_aw_clear(handle: *mut c_void);
            pub fn bloop_aw_process(
                handle: *mut c_void,
                left: *mut f32,
                right: *mut f32,
                frames: i32,
            );
        }
    }

    /// Stand-ins with the native ABI's shape, never called: no plug-in is
    /// compiled for wasm32.
    #[cfg(target_arch = "wasm32")]
    mod raw {
        use std::ffi::{c_char, c_void};
        pub unsafe fn bloop_aw_entry_size(_: i32) -> std::ffi::c_ulong {
            0
        }
        pub unsafe fn bloop_aw_new(_: i32, _: f64) -> *mut c_void {
            std::ptr::null_mut()
        }
        pub unsafe fn bloop_aw_free(_: *mut c_void) {}
        pub unsafe fn bloop_aw_param_count(_: *const c_void) -> i32 {
            0
        }
        pub unsafe fn bloop_aw_get(_: *mut c_void, _: i32) -> f32 {
            0.0
        }
        pub unsafe fn bloop_aw_set(_: *mut c_void, _: i32, _: f32) {}
        pub unsafe fn bloop_aw_param_name(_: *mut c_void, _: i32, _: *mut c_char) {}
        pub unsafe fn bloop_aw_param_label(_: *mut c_void, _: i32, _: *mut c_char) {}
        pub unsafe fn bloop_aw_param_display_at(_: *mut c_void, _: i32, _: f32, _: *mut c_char) {}
        pub unsafe fn bloop_aw_set_rate(_: *mut c_void, _: f64) {}
        pub unsafe fn bloop_aw_clear(_: *mut c_void) {}
        pub unsafe fn bloop_aw_process(_: *mut c_void, _: *mut f32, _: *mut f32, _: i32) {}
    }

    pub fn entry_size(entry: i32) -> usize {
        // Safety: any entry is valid to ask; one out of range answers 0.
        let size = unsafe { raw::bloop_aw_entry_size(entry) };
        usize::try_from(size).unwrap_or(usize::MAX)
    }

    /// Null when the memory could not be had.
    pub fn new(entry: i32, sample_rate: f64) -> *mut c_void {
        // Safety: any entry is valid to ask; one out of range answers null.
        unsafe { raw::bloop_aw_new(entry, sample_rate) }
    }

    // Safety, for everything below: `handle` came from `new`, is not null,
    // and has not been freed — [`super::Handle`] holds it so. `index` may
    // be anything: the C side ignores one out of range. Each text buffer is
    // TEXT bytes, and the C side writes at most 17 of them.

    pub fn free(handle: *mut c_void) {
        unsafe { raw::bloop_aw_free(handle) }
    }

    pub fn param_count(handle: *mut c_void) -> usize {
        usize::try_from(unsafe { raw::bloop_aw_param_count(handle) }).unwrap_or(0)
    }

    pub fn get(handle: *mut c_void, index: i32) -> f32 {
        unsafe { raw::bloop_aw_get(handle, index) }
    }

    pub fn set(handle: *mut c_void, index: i32, value: f32) {
        unsafe { raw::bloop_aw_set(handle, index, value) }
    }

    pub fn param_name(handle: *mut c_void, index: i32, out: &mut [c_char; TEXT]) {
        unsafe { raw::bloop_aw_param_name(handle, index, out.as_mut_ptr()) }
    }

    pub fn param_label(handle: *mut c_void, index: i32, out: &mut [c_char; TEXT]) {
        unsafe { raw::bloop_aw_param_label(handle, index, out.as_mut_ptr()) }
    }

    pub fn display_at(handle: *mut c_void, index: i32, value: f32, out: &mut [c_char; TEXT]) {
        unsafe { raw::bloop_aw_param_display_at(handle, index, value, out.as_mut_ptr()) }
    }

    pub fn set_rate(handle: *mut c_void, sample_rate: f64) {
        unsafe { raw::bloop_aw_set_rate(handle, sample_rate) }
    }

    pub fn clear(handle: *mut c_void) {
        unsafe { raw::bloop_aw_clear(handle) }
    }

    /// `left` and `right` in place, as many frames as the shorter holds.
    pub fn process(handle: *mut c_void, left: &mut [f32], right: &mut [f32]) {
        let frames = i32::try_from(left.len().min(right.len())).unwrap_or(i32::MAX);
        unsafe { raw::bloop_aw_process(handle, left.as_mut_ptr(), right.as_mut_ptr(), frames) }
    }
}

/// A constructed plug-in, owned: freed when dropped.
struct Handle(NonNull<c_void>);

// Safety: a plug-in is plain memory with no tie to the thread that built
// it; `Handle` is never shared, only moved (an `Effect` lends it through
// `&mut` or a `Mutex`).
unsafe impl Send for Handle {}

impl Handle {
    /// UI thread: allocates the plug-in's memory, which is aborted on,
    /// as Rust's own allocations are, when it cannot be had.
    fn new(id: Id, sample_rate: f32) -> Self {
        match NonNull::new(ffi::new(id.entry(), f64::from(sample_rate))) {
            Some(handle) => Handle(handle),
            None => std::alloc::handle_alloc_error(
                std::alloc::Layout::from_size_align(ffi::entry_size(id.entry()).max(1), 16)
                    .unwrap_or(std::alloc::Layout::new::<u8>()),
            ),
        }
    }

    fn ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    fn param_count(&self) -> usize {
        ffi::param_count(self.ptr())
    }

    fn get(&self, index: usize) -> f32 {
        ffi::get(self.ptr(), index_of(index))
    }

    fn text(
        &self,
        index: usize,
        ask: fn(*mut c_void, i32, &mut [std::ffi::c_char; ffi::TEXT]),
    ) -> String {
        let mut out = [0; ffi::TEXT];
        ask(self.ptr(), index_of(index), &mut out);
        text_of(&out)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        ffi::free(self.ptr());
    }
}

fn index_of(index: usize) -> i32 {
    i32::try_from(index).unwrap_or(-1)
}

fn text_of(out: &[std::ffi::c_char; ffi::TEXT]) -> String {
    let bytes: Vec<u8> = out
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c.to_ne_bytes()[0])
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// One Airwindows plug-in as a bloop effect: its knobs 0 to 1, as it has
/// them, and a stereo block in place.
pub struct Effect {
    id: Id,
    sample_rate: f32,
    /// Where each knob is, as last set: read back without asking the
    /// plug-in, so reading a knob needs no lock.
    values: Box<[f32]>,
    /// Locked only to read a knob's text (`&self`, the UI thread); the
    /// audio thread reaches it through `&mut`, which takes no lock.
    plugin: Mutex<Handle>,
}

impl Effect {
    /// The plug-in at its defaults, at `sample_rate`. UI thread: it
    /// allocates the plug-in's memory, some megabytes for a reverb.
    #[must_use]
    pub fn new(id: Id, sample_rate: f32) -> Self {
        let sample_rate = valid_rate(sample_rate);
        let values = id.params().iter().map(|p| p.default).collect();
        Effect {
            id,
            sample_rate,
            values,
            plugin: Mutex::new(Handle::new(id, sample_rate)),
        }
    }

    #[must_use]
    pub fn id(&self) -> Id {
        self.id
    }

    /// The rate it is prepared for.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn handle(&mut self) -> &mut Handle {
        self.plugin
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Silence what the plug-in holds and keep its knobs: the plug-in is
    /// constructed again in its own memory. Allocates nothing.
    pub fn clear(&mut self) {
        ffi::clear(self.handle().ptr());
    }

    /// Knob `index` at `value`, as the plug-in reads it, with its unit:
    /// "0.50", "-6.02 dB", "Dry".
    #[must_use]
    pub fn param_text(&self, index: usize, value: f32) -> String {
        let Some(param) = self.id.params().get(index) else {
            return String::new();
        };
        let value = clamp(value, param.default);
        let plugin = self.plugin.lock().unwrap_or_else(PoisonError::into_inner);
        let mut out = [0; ffi::TEXT];
        ffi::display_at(plugin.ptr(), index_of(index), value, &mut out);
        let reading = text_of(&out);
        let reading = reading.trim();
        if param.label.is_empty() {
            reading.to_string()
        } else {
            format!("{reading} {}", param.label)
        }
    }

    /// Airwindows' knobs read in no unit bloop's knob feel knows.
    #[must_use]
    pub fn param_reads_in(&self, _index: usize) -> Option<(&'static str, f32)> {
        None
    }

    /// Every knob turns continuously, as far as the host is told.
    #[must_use]
    pub fn param_steps(&self, _index: usize) -> Option<usize> {
        None
    }

    /// A knob's travel is its value.
    #[must_use]
    pub fn knob_value(&self, index: usize, fraction: f32) -> f32 {
        clamp(fraction, self.default(index))
    }

    #[must_use]
    pub fn knob_fraction(&self, index: usize, value: f32) -> f32 {
        clamp(value, self.default(index))
    }

    fn default(&self, index: usize) -> f32 {
        self.id.params().get(index).map_or(0.0, |p| p.default)
    }

    /// No side-chain input: the block as it always is.
    pub fn process_keyed(&mut self, context: &mut ProcessContext, _key: [&[f32]; 2]) {
        self.process(context);
    }

    /// No dynamics screen.
    #[must_use]
    pub fn meter(&self) -> Option<DynamicsMeter> {
        None
    }

    /// The longest the plug-in can be silent while still holding something
    /// it will play: a delay's line, for the Ambience ones.
    #[must_use]
    pub fn quiet_gap_seconds(&self, _bpm: f32) -> f32 {
        if self.id.listing().category == "Ambience" {
            AMBIENCE_GAP_SECONDS
        } else {
            0.0
        }
    }
}

impl Clone for Effect {
    /// A fresh instance with the same knobs. UI thread: it allocates.
    fn clone(&self) -> Self {
        let mut copy = Effect::new(self.id, self.sample_rate);
        for (index, &value) in self.values.iter().enumerate() {
            copy.set_param(index, value);
        }
        copy
    }
}

impl std::fmt::Debug for Effect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Effect")
            .field("name", &self.id.name())
            .field("sample_rate", &self.sample_rate)
            .field("values", &self.values)
            .finish_non_exhaustive()
    }
}

impl Plugin for Effect {
    fn info(&self) -> PluginInfo {
        PluginInfo {
            name: self.id.name(),
            vendor: VENDOR,
            category: PluginCategory::Effect,
            num_audio_inputs: 2,
            num_audio_outputs: 2,
        }
    }

    fn param_count(&self) -> usize {
        self.values.len()
    }

    fn param_info(&self, index: usize) -> ParamInfo {
        self.id.params().get(index).map_or(
            ParamInfo {
                name: "",
                default: 0.0,
                min: 0.0,
                max: 0.0,
            },
            |p| ParamInfo {
                name: p.name.as_str(),
                default: p.default,
                min: 0.0,
                max: 1.0,
            },
        )
    }

    /// Held to 0..1; not a number sets the default.
    fn set_param(&mut self, index: usize, value: f32) {
        if index >= self.values.len() {
            return;
        }
        let value = clamp(value, self.default(index));
        self.values[index] = value;
        ffi::set(self.handle().ptr(), index_of(index), value);
    }

    fn get_param(&self, index: usize) -> f32 {
        self.values.get(index).copied().unwrap_or(0.0)
    }

    /// In place: the outputs arrive holding the input.
    fn process(&mut self, context: &mut ProcessContext) {
        let handle = self.handle().ptr();
        if let [left, right, ..] = &mut context.outputs[..] {
            ffi::process(handle, left, right);
        }
    }

    /// At `sample_rate`, cleared. Allocates nothing.
    fn reset(&mut self, sample_rate: f32, _max_buffer_size: usize) {
        self.sample_rate = valid_rate(sample_rate);
        let rate = f64::from(self.sample_rate);
        let handle = self.handle().ptr();
        ffi::set_rate(handle, rate);
        ffi::clear(handle);
    }
}

fn clamp(value: f32, default: f32) -> f32 {
    if value.is_nan() {
        default
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn valid_rate(sample_rate: f32) -> f32 {
    if sample_rate.is_finite() && sample_rate > 0.0 {
        sample_rate
    } else {
        48_000.0
    }
}
