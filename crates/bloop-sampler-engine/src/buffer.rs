//! The shared, immutable audio a voice reads from.

use std::sync::Arc;

/// A decoded, deinterleaved, immutable piece of audio.
///
/// Voices only ever borrow this, so one import can back any number of pads
/// playing different slices of it: the pads share a single [`SharedSample`]
/// and differ only in their start/end bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleBuffer {
    channels: Vec<Vec<f32>>,
    sample_rate: u32,
}

/// A [`SampleBuffer`] shared between pads. Cheap to clone.
pub type SharedSample = Arc<SampleBuffer>;

impl SampleBuffer {
    /// Builds a buffer from one slice per channel. All channels must be the
    /// same length.
    pub fn new(channels: Vec<Vec<f32>>, sample_rate: u32) -> Self {
        debug_assert!(
            channels.windows(2).all(|w| w[0].len() == w[1].len()),
            "every channel must hold the same number of frames"
        );
        Self {
            channels,
            sample_rate,
        }
    }

    /// An empty buffer, which every voice treats as "no sample loaded".
    pub fn empty() -> Self {
        Self {
            channels: Vec::new(),
            sample_rate: 0,
        }
    }

    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Frames per channel. Zero when no channels are loaded.
    pub fn num_samples(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channel(&self, channel: usize) -> &[f32] {
        &self.channels[channel]
    }

    /// Apply `f` to every sample of every channel.
    pub(crate) fn map_samples(&mut self, mut f: impl FnMut(f32) -> f32) {
        for channel in &mut self.channels {
            for value in channel.iter_mut() {
                *value = f(*value);
            }
        }
    }

    /// Apply `f` to each channel's samples as a slice, for edits that need
    /// to see a whole channel at once (reversing, fading).
    pub(crate) fn map_channel_slices(&mut self, mut f: impl FnMut(&mut [f32])) {
        for channel in &mut self.channels {
            f(channel);
        }
    }

    /// `juce::AudioBuffer::getSample`. Out-of-range indices are a programming
    /// error, matching the reference's `jassert`-guarded accessor.
    pub fn sample(&self, channel: usize, index: usize) -> f32 {
        self.channels[channel][index]
    }
}
