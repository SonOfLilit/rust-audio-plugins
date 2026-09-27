//! `gin::GraphicEQ` with ten bands: a bandpass per octave, run in parallel,
//! each adding its share back at its own gain.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_graphiceq.h`. The original runs each
//! band over the whole block into a scratch buffer and accumulates; here
//! each sample goes through every band in turn, which adds the same terms
//! in the same order without the scratch.

use super::filter::{Filter, FilterType, Slope};
use crate::juce::decibels_to_gain;

pub const BANDS: usize = 10;
const FREQUENCIES: [f32; BANDS] = [
    31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];
/// One octave apart.
const BAND_Q: f32 = 2.9;

#[derive(Clone, Debug)]
pub struct GraphicEq {
    sample_rate: f64,
    channels: usize,
    /// `[channel][band]`.
    gains: [[f32; BANDS]; 2],
    filters: [[Filter; BANDS]; 2],
}

impl Default for GraphicEq {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            channels: 0,
            gains: [[0.0; BANDS]; 2],
            filters: [[Filter::new(FilterType::BandPass); BANDS]; 2],
        }
    }
}

impl GraphicEq {
    /// `setSampleRate`. The bands' coefficients are not worked out again:
    /// the original only designs them when the channel count changes.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for channel in &mut self.filters[..self.channels] {
            for filter in channel {
                filter.set_sample_rate(sample_rate);
            }
        }
    }

    pub fn set_num_channels(&mut self, channels: usize) {
        let channels = channels.min(2);
        if self.channels == channels {
            return;
        }
        for c in self.channels.min(channels)..channels {
            self.gains[c] = [0.0; BANDS];
            for (filter, freq) in self.filters[c].iter_mut().zip(FREQUENCIES) {
                *filter = Filter::new(FilterType::BandPass);
                filter.set_sample_rate(self.sample_rate);
                filter.set_slope(Slope::Db12);
                filter.set_params(freq, BAND_Q, 1.0);
            }
        }
        self.channels = channels;
    }

    pub fn reset(&mut self) {
        for channel in &mut self.filters[..self.channels] {
            for filter in channel {
                filter.reset();
            }
        }
    }

    pub fn set_band_gain(&mut self, channel: usize, band: usize, gain_db: f32) {
        self.gains[channel][band] = gain_db;
    }

    pub fn process(&mut self, channels: [&mut [f32]; 2]) {
        for (c, data) in channels.into_iter().enumerate().take(self.channels) {
            for x in data.iter_mut() {
                let mut accumulator = 0.0f32;
                for (filter, &gain) in self.filters[c].iter_mut().zip(&self.gains[c]) {
                    let filtered = filter.process(0, *x);
                    accumulator += filtered * (decibels_to_gain(gain) - 1.0);
                }
                *x += accumulator;
            }
        }
    }
}
