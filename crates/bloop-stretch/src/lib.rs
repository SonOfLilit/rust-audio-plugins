// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Ported from Bungee's src/Stretcher.{h,cpp}, Grain.{h,cpp}, Grains.cpp,
// Timing.cpp, Input.cpp, Output.cpp, Window.cpp, Synthesis.cpp and
// Resample.h (Copyright (C) 2020-2026 Parabola Research Limited,
// SPDX-License-Identifier: MPL-2.0).

//! bloop-stretch: time-stretching and pitch-shifting, a port of Bungee
//! Basic (v2.4.10), the stretcher behind Just-a-Sample's ADVANCED mode.
//!
//! Bungee is a phase vocoder that works a grain at a time. The caller says
//! where in the input each grain is centred ([`Request::position`]) and at
//! what pitch; the stretcher asks for the input around it
//! ([`Stretcher::specify_grain`]), takes it ([`Stretcher::analyse_grain`])
//! and hands back the next stretch of output ([`Stretcher::synthesise_grain`]).
//! How far apart consecutive positions are is the speed: the output always
//! advances by one hop (512 frames at 44.1 or 48 kHz) per grain, so a
//! caller that moves the position by 256 frames a grain plays at half
//! speed. Positions may jump; a [`Request::reset`] grain starts afresh.
//!
//! Each grain is a 4096-point transform (8 hops), whose spectrum is split
//! into partials around its peaks. A peak's phase is advanced by what the
//! input's phase did over the analysis hop, scaled to the synthesis hop,
//! and the bins around it turn with it (identity phase locking). A peak
//! whose energy jumps by half again since the previous grain is a
//! transient, and its bins lose their lock. The grain's central four hops
//! are windowed and overlap-added, four deep. A pitch other than 1
//! stretches time by the pitch ratio and resamples the output by it,
//! linearly, as Bungee does.
//!
//! What is not ported: input resampling (Bungee's `resampleMode` other
//! than its default, `autoOut`), the instrumentation, and the mute counts
//! of `analyseGrain` — a caller supplies zeros where there is no input.
//! PFFFT is replaced by rustfft ([`fourier`]).
//!
//! Construction allocates; nothing after it does.

mod fourier;
mod partials;
mod phase;

use fourier::RealFft;
use partials::Partial;
use phase::Phase;
use rustfft::num_complex::Complex32;

/// Bungee's limit on how far a pitch can move, in octaves either way:
/// Bungee ships 2 and Just-a-Sample builds it with 4. Three is the most
/// the Audio plug-in's Tune reaches (36 semitones). Only buffer sizes
/// depend on it.
pub const MAX_PITCH_OCTAVES: u32 = 3;

/// Frames kept either side of the output's overlap-add buffer, for the
/// resampler to read past its ends.
const PADDING: usize = 32;

/// The input and output sample rates, in hertz.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleRates {
    pub input: u32,
    pub output: u32,
}

/// What a grain is: where it is centred in the input, how fast the input
/// is being read, and at what pitch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    /// The input frame the grain is centred on. NaN is a grain that plays
    /// nothing, for flushing.
    pub position: f64,
    /// Input frames per output frame. Used only where the step from the
    /// previous grain's position cannot say it: the first grain, and a
    /// reset.
    pub speed: f64,
    /// A frequency ratio: 1 leaves the pitch alone, 2 is an octave up.
    pub pitch: f64,
    /// Forget every earlier grain and start again from this one.
    pub reset: bool,
}

impl Request {
    /// A grain at `position`, at `speed` and `pitch`.
    #[must_use]
    pub fn new(position: f64, speed: f64, pitch: f64) -> Self {
        Self {
            position,
            speed,
            pitch,
            reset: false,
        }
    }

    const NONE: Self = Self {
        position: f64::NAN,
        speed: f64::NAN,
        pitch: 1.0,
        reset: false,
    };
}

/// The input frames a grain needs: `begin..end`, which can run off either
/// end of the input (the caller supplies zeros there).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputChunk {
    pub begin: i32,
    pub end: i32,
}

impl InputChunk {
    /// How many frames per channel `analyse_grain` wants.
    #[must_use]
    pub fn len(self) -> usize {
        (self.end - self.begin).max(0) as usize
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// What [`Stretcher::synthesise_grain`] produced: `frames` frames, read
/// with [`Stretcher::output`], playing the input from `begin.position`
/// (at the first frame) toward `end.position` (just after the last). Both
/// positions are NaN while the stretcher is still filling up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputChunk {
    pub frames: usize,
    pub begin: Request,
    pub end: Request,
}

/// One direction of resampling, and its ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Operation {
    active: bool,
    ratio: f64,
}

impl Operation {
    const NONE: Self = Self {
        active: false,
        ratio: 1.0,
    };
}

/// Bungee's `Resample::Operations::setup` in `autoOut` mode: a pitch or a
/// rate change is made by resampling the output. Returns how many input
/// frames one synthesis frame is worth before speed.
fn setup_output(rates: SampleRates, pitch: f64, output: &mut Operation) -> f64 {
    let rate_ratio = f64::from(rates.input) / f64::from(rates.output);
    let resample_ratio = pitch * f64::from(rates.input) / f64::from(rates.output);
    if resample_ratio == 1.0 {
        *output = Operation::NONE;
        rate_ratio
    } else {
        *output = Operation {
            active: true,
            ratio: resample_ratio,
        };
        rate_ratio / resample_ratio
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Analysis {
    position_error: f64,
    hop_ideal: f64,
    speed: f64,
    hop: i32,
}

/// One grain, without its spectra (those live in [`Spectra`], of which only
/// the current and previous grain need one).
#[derive(Clone, Copy, Debug)]
struct Grain {
    request: Request,
    continuous: bool,
    valid_bin_count: usize,
    output: Operation,
    analysis: Analysis,
}

impl Grain {
    const NONE: Self = Self {
        request: Request::NONE,
        continuous: false,
        valid_bin_count: 0,
        output: Operation::NONE,
        analysis: Analysis {
            position_error: 0.0,
            hop_ideal: 0.0,
            speed: 0.0,
            hop: 0,
        },
    };

    fn valid(&self) -> bool {
        !self.request.position.is_nan()
    }

    fn reverse(&self) -> bool {
        self.analysis.hop < 0
    }
}

/// A grain's per-bin phase, energy and rotation, and its partials.
struct Spectra {
    phase: Vec<Phase>,
    energy: Vec<f32>,
    rotation: Vec<Phase>,
    partials: Vec<Partial>,
    partial_count: usize,
}

impl Spectra {
    fn new(bins: usize, transform_length: usize) -> Self {
        // Bungee sizes these a few bins past the last, for the sentinels
        // `partials::enumerate` borrows.
        Self {
            phase: vec![0; bins + 3],
            energy: vec![0.0; bins + 3],
            rotation: vec![0; bins + 3],
            partials: vec![Partial::default(); transform_length],
            partial_count: 0,
        }
    }
}

/// The stretcher. See the crate documentation for how to drive it.
pub struct Stretcher {
    rates: SampleRates,
    channels: usize,
    log2_hop: u32,
    hop: usize,
    transform_length: usize,
    bins: usize,

    fft: RealFft,
    analysis_window: Vec<f32>,
    synthesis_window: Vec<f32>,
    windowed: Vec<f32>,
    /// `bins` per channel.
    transformed: Vec<Vec<Complex32>>,
    temporary: Vec<Complex32>,
    delta: Vec<Phase>,
    inverse: Vec<Vec<f32>>,

    /// `grains[(head + i) % 4]` is Bungee's `grains[i]`: 0 the grain being
    /// made, 1 the one before it, and so on.
    grains: [Grain; 4],
    head: usize,
    /// `spectra[current]` is grain 0's, the other grain 1's.
    spectra: [Spectra; 2],
    current: usize,

    /// The overlap-add buffer, per channel: `PADDING` frames of history,
    /// then the transform length, then `PADDING` more.
    lapped: Vec<Vec<f32>>,
    lapped_offset: f64,
    resampled: Vec<Vec<f32>>,
    output_frames: usize,
    output_resampled: bool,
}

/// A raised cosine `gain * (1 + cos(2 pi t / length))`, peaking at `t = 0`
/// and laid out round the circle: what Bungee's `Window::
/// fromFrequencyDomainCoefficients` makes from the coefficients `{1, 0.5}`.
fn raised_cosine(length: usize, gain: f32) -> Vec<f32> {
    (0..length)
        .map(|t| {
            let c = libm::cos(2.0 * core::f64::consts::PI * t as f64 / length as f64);
            (f64::from(gain) * (1.0 + c)) as f32
        })
        .collect()
}

impl Stretcher {
    /// A stretcher for `channels` channels. Allocates.
    #[must_use]
    pub fn new(rates: SampleRates, channels: usize) -> Self {
        let log2_hop = rates.input.max(1).ilog2() - 6;
        let hop = 1_usize << log2_hop;
        let transform_length = hop << 3;
        let bins = transform_length / 2 + 1;

        let pi = core::f32::consts::PI;
        let analysis_gain = (3.0 * pi) / (3.0 * pi + 8.0) / transform_length as f32;

        let max_output = ((u64::from(rates.output) << (MAX_PITCH_OCTAVES + log2_hop))
            / u64::from(rates.input.max(1))) as usize
            + 1;

        Self {
            rates,
            channels,
            log2_hop,
            hop,
            transform_length,
            bins,
            fft: RealFft::new(transform_length),
            analysis_window: raised_cosine(transform_length, analysis_gain),
            synthesis_window: raised_cosine(4 * hop, 0.25),
            windowed: vec![0.0; transform_length],
            transformed: vec![vec![Complex32::default(); bins]; channels],
            temporary: vec![Complex32::default(); bins],
            delta: vec![0; bins + 3],
            inverse: vec![vec![0.0; transform_length]; channels],
            grains: [Grain::NONE; 4],
            head: 0,
            spectra: [
                Spectra::new(bins, transform_length),
                Spectra::new(bins, transform_length),
            ],
            current: 0,
            lapped: vec![vec![0.0; 2 * PADDING + transform_length]; channels],
            lapped_offset: 0.0,
            resampled: vec![vec![0.0; max_output]; channels],
            output_frames: 0,
            output_resampled: false,
        }
    }

    /// Frames of output each grain makes at a pitch of 1 and equal rates.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// The most input frames a grain can ask for.
    #[must_use]
    pub fn max_input_frame_count(&self) -> usize {
        let max = (u64::from(self.rates.input) << (MAX_PITCH_OCTAVES + self.log2_hop + 3))
            / u64::from(self.rates.output.max(1));
        max as usize + 1
    }

    fn grain(&self, i: usize) -> &Grain {
        &self.grains[(self.head + i) % 4]
    }

    fn input_hop(&self, request: &Request) -> f64 {
        let mut output = Operation::NONE;
        let unit = self.hop as f64 * setup_output(self.rates, request.pitch, &mut output);
        unit * request.speed
    }

    /// Step `request` back by one grain and mark it a reset: the grain to
    /// specify first so that the next one, at the original position, is
    /// heard from its first frame.
    pub fn preroll(&self, request: &mut Request) {
        request.position -= self.input_hop(request);
        request.reset = true;
    }

    /// Step `request` on by one grain at its speed.
    pub fn next(&self, request: &mut Request) {
        if !request.speed.is_nan() && !request.position.is_nan() {
            request.position += self.input_hop(request);
            request.reset = false;
        }
    }

    /// Begin a grain: say what it is, and get back the input it needs,
    /// counted from `buffer_start` (0 for a caller that counts input
    /// frames from the start).
    pub fn specify_grain(&mut self, request: &Request, buffer_start: f64) -> InputChunk {
        self.head = (self.head + 3) % 4;
        self.current ^= 1;

        let previous = *self.grain(1);
        let rates = self.rates;
        let log2_hop = self.log2_hop;
        let hop = self.hop;
        let half = (self.transform_length / 2) as i32;
        let grain = &mut self.grains[self.head];

        grain.request = *request;
        let unit_hop = hop as f64 * setup_output(rates, request.pitch, &mut grain.output);

        let mut request_hop = request.position - previous.request.position;
        if request_hop.is_nan() || request.reset {
            request_hop = request.speed * unit_hop;
        }

        grain.analysis.hop_ideal = request_hop;
        grain.continuous = !request.reset && !previous.request.position.is_nan();
        if grain.continuous {
            grain.analysis.position_error =
                previous.analysis.position_error - grain.analysis.hop_ideal;
            grain.analysis.hop = (-grain.analysis.position_error).round() as i32;
            grain.analysis.position_error += f64::from(grain.analysis.hop);
        } else {
            grain.analysis.hop = grain.analysis.hop_ideal.round() as i32;
            grain.analysis.position_error = request.position.round() - request.position;
        }
        grain.analysis.speed = grain.analysis.hop_ideal / (1_u32 << log2_hop) as f64;

        if request.position.is_nan() {
            return InputChunk::default();
        }
        let offset = (request.position - buffer_start).round() as i32;
        InputChunk {
            begin: offset - half,
            end: offset + half,
        }
    }

    /// Analyse the grain just specified. `input` holds the frames its
    /// [`InputChunk`] asked for, channel after channel: `chunk.len()` of
    /// the first channel, then of the second, and so on.
    pub fn analyse_grain(&mut self, input: &[f32]) {
        let current = self.current;
        let previous_spectra = current ^ 1;
        let head = self.head;
        self.grains[head].valid_bin_count = 0;
        if !self.grains[head].valid() {
            return;
        }

        let n = self.transform_length;
        let half = n / 2;
        let frames = input.len() / self.channels.max(1);
        for channel in 0..self.channels {
            let x = &input[channel * frames..(channel + 1) * frames];
            let middle = frames / 2;
            // Bungee's `Input::applyAnalysisWindow`: the window's peak at
            // the grain's centre, rotated to sample 0.
            for j in 0..half {
                self.windowed[j] = x[middle + j] * self.analysis_window[j];
                self.windowed[half + j] = x[middle - half + j] * self.analysis_window[half + j];
            }
            self.fft
                .forward(&self.windowed, &mut self.transformed[channel]);
        }

        let last = self.bins - 1;
        let grain = &mut self.grains[head];
        let valid = ((last as f64 / grain.output.ratio).ceil() as usize).min(last) + 1;
        grain.valid_bin_count = valid;
        for channel in &mut self.transformed {
            for bin in &mut channel[valid..] {
                *bin = Complex32::default();
            }
        }

        let (a, b) = self.spectra.split_at_mut(1);
        let (spectra, previous) = if current == 0 {
            (&mut a[0], &b[0])
        } else {
            (&mut b[0], &a[0])
        };
        debug_assert_eq!(previous_spectra, current ^ 1);
        for i in 0..valid {
            let x = self
                .transformed
                .iter()
                .fold(Complex32::default(), |sum, channel| sum + channel[i]);
            spectra.energy[i] = x.re * x.re + x.im * x.im;
            spectra.phase[i] = phase::from_radians(libm::atan2f(x.im, x.re));
        }

        partials::enumerate(
            &mut spectra.partials,
            &mut spectra.partial_count,
            valid,
            &mut spectra.energy,
        );
        if grain.continuous {
            partials::suppress_transient_partials(
                &mut spectra.partials[..spectra.partial_count],
                &spectra.energy,
                &previous.energy,
            );
        }
    }

    /// Turn the analysed grain's phases and add it to the output, then
    /// hand back the next stretch of output.
    pub fn synthesise_grain(&mut self) -> OutputChunk {
        const K: f32 = core::f32::consts::PI / 32768.0;
        let head = self.head;
        if self.grains[head].valid() {
            self.synthesise();
            let grain = self.grains[head];
            let valid = grain.valid_bin_count;
            let spectra = &self.spectra[self.current];
            for (t, &rotation) in self.temporary[..valid].iter_mut().zip(&spectra.rotation) {
                let angle = f32::from(rotation) * K;
                *t = Complex32::new(libm::cosf(angle), libm::sinf(angle));
            }
            for (channel, inverse) in self.transformed.iter_mut().zip(&mut self.inverse) {
                for (bin, t) in channel[..valid].iter_mut().zip(&self.temporary) {
                    let x = if grain.reverse() { bin.conj() } else { *bin };
                    *bin = x * t;
                }
                self.fft.inverse(channel, inverse);
            }
        }

        self.apply_synthesis_window();
        let begin = *self.grain(2);
        let end = *self.grain(1);
        self.resample(begin.output, end.output);

        OutputChunk {
            frames: self.output_frames,
            begin: begin.request,
            end: end.request,
        }
    }

    /// The last output chunk's frames on `channel`.
    #[must_use]
    pub fn output(&self, channel: usize) -> &[f32] {
        if self.output_resampled {
            &self.resampled[channel][..self.output_frames]
        } else {
            &self.lapped[channel][PADDING..PADDING + self.output_frames]
        }
    }

    /// Whether every grain the stretcher holds is a NaN one: whatever it
    /// was playing has been played out.
    #[must_use]
    pub fn is_flushed(&self) -> bool {
        self.grains.iter().all(|g| !g.valid())
    }

    /// Bungee's `Synthesis::synthesise`: each bin's rotation.
    fn synthesise(&mut self) {
        let grain = self.grains[self.head];
        let previous_grain = *self.grain(1);
        let (a, b) = self.spectra.split_at_mut(1);
        let (spectra, previous) = if self.current == 0 {
            (&mut a[0], &b[0])
        } else {
            (&mut b[0], &a[0])
        };
        let valid = grain.valid_bin_count;
        let partials = &spectra.partials[..spectra.partial_count];

        phase::Frequency::new(grain.analysis.speed as f32).apply(
            valid,
            &mut spectra.rotation,
            &spectra.phase,
        );

        if grain.continuous {
            let time = phase::Time::new(
                self.log2_hop,
                grain.analysis.hop,
                previous_grain.analysis.hop,
            );
            for (i, partial) in partials.iter().enumerate() {
                let peak = partial.peak as usize;
                let offset = time.offset(spectra.phase[peak], previous.phase[peak]);
                let stretched = time.delta(spectra.phase[peak], previous.phase[peak], peak as i32);
                self.delta[i] = previous.rotation[peak]
                    .wrapping_sub(offset)
                    .wrapping_add(stretched)
                    .wrapping_sub(spectra.rotation[peak]);
            }
        } else {
            for (i, partial) in partials.iter().enumerate() {
                self.delta[i] = spectra.rotation[partial.peak as usize].wrapping_neg();
            }
        }

        let mut n = 0;
        for (i, partial) in partials.iter().enumerate() {
            loop {
                spectra.rotation[n] = spectra.rotation[n].wrapping_add(self.delta[i]);
                n += 1;
                if n >= partial.end as usize {
                    break;
                }
            }
        }

        let nyquist = self.bins - 1;
        spectra.rotation[nyquist] = spectra.rotation[nyquist - 1];
    }

    /// Bungee's `Output::applySynthesisWindow`: window the grain's central
    /// four hops and add them in, one hop on.
    fn apply_synthesis_window(&mut self) {
        let hop = self.hop;
        let valid = self.grains[self.head].valid();
        for (lapped, inverse) in self.lapped.iter_mut().zip(&self.inverse) {
            lapped.copy_within(hop..hop + PADDING, 0);
            let body = &mut lapped[PADDING..PADDING + self.transform_length];
            if valid {
                for i in 0..4 {
                    let window = &self.synthesis_window[hop * (i ^ 2)..hop * (i ^ 2) + hop];
                    let j = (i + 6) % 8;
                    let grain = &inverse[hop * j..hop * j + hop];
                    for k in 0..hop {
                        let windowed = grain[k] * window[k];
                        body[i * hop + k] = if i < 3 {
                            windowed + body[(i + 1) * hop + k]
                        } else {
                            windowed
                        };
                    }
                }
            } else {
                body.copy_within(hop..4 * hop, 0);
                body[3 * hop..4 * hop].fill(0.0);
            }
        }
    }

    /// Bungee's `Output::resample` with its bilinear interpolation: the
    /// hop just completed, resampled from `begin`'s ratio to `end`'s.
    fn resample(&mut self, begin: Operation, end: Operation) {
        let active = begin.active || end.active;
        self.output_resampled = active;
        if !active {
            self.output_frames = self.hop;
            return;
        }
        let (ratio_begin, mut ratio_end) = (begin.ratio, end.ratio);
        let align_end = !end.active;
        let frame_count = self.hop as f64;
        let capacity = self.resampled[0].len();

        let ideal = (2.0 * (frame_count - self.lapped_offset) / (ratio_begin + ratio_end)).round();
        let mut frames = capacity;
        if ideal <= capacity as f64 {
            frames = ideal.max(0.0) as usize;
        }
        self.output_frames = frames;
        if frames == 0 {
            return;
        }
        if align_end {
            let mean = (frame_count - self.lapped_offset) / frames as f64;
            ratio_end = 2.0 * mean - ratio_begin;
        }
        let gradient = (ratio_end - ratio_begin) / frames as f64;
        let constant = ratio_begin == ratio_end;

        let mut x_end = 0.0;
        for (lapped, out) in self.lapped.iter().zip(&mut self.resampled) {
            let mut x = PADDING as f64 + self.lapped_offset;
            let mut ratio = if constant {
                ratio_begin
            } else {
                ratio_begin + 0.5 * gradient
            };
            for sample in &mut out[..frames] {
                let xf = x as f32;
                let integer = xf as usize;
                let fraction = xf - integer as f32;
                let mut value = lapped[integer + 1] * fraction;
                value += lapped[integer] * (1.0 - fraction);
                *sample = value;
                x += ratio;
                if !constant {
                    ratio += gradient;
                }
            }
            x_end = x;
        }
        self.lapped_offset = x_end - PADDING as f64 - frame_count;
        let tolerance = if align_end { 1e-2 } else { ratio_end * 1.1 };
        if self.lapped_offset.abs() > tolerance {
            self.lapped_offset = 0.0;
        }
    }
}
