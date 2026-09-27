//! `gin::Maximizer`: each half cycle scaled back to the threshold as a
//! whole, after it is over, and the lot made up to the ceiling.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_maximizer.h`.

use crate::juce::{decibels_to_gain, jlimit};
use crate::math::{std_max, std_min};

const SAFETY_LIMIT: f32 = 10.0;

#[derive(Clone, Debug)]
struct Channel {
    buffer: Vec<f32>,
    length: i32,
    max_sample: f32,
    gain: f32,
    last_positive: bool,
}

#[derive(Clone, Debug)]
pub struct Maximizer {
    sample_rate: f64,
    buffer_size: i32,
    buffer_pos: i32,
    max_cycle_length: f32,
    threshold: f32,
    ceiling: f32,
    makeup: f32,
    release_seconds: f32,
    release: f32,
    release_active: bool,
    chans: [Channel; 2],
    channels: usize,
}

impl Default for Maximizer {
    fn default() -> Self {
        let channel = Channel {
            buffer: Vec::new(),
            length: 0,
            max_sample: 0.0,
            gain: 1.0,
            last_positive: false,
        };
        Self {
            sample_rate: 44100.0,
            buffer_size: 0,
            buffer_pos: 0,
            max_cycle_length: 0.05,
            threshold: 1.0,
            ceiling: 1.0,
            makeup: 1.0,
            release_seconds: 0.0,
            release: 1.0,
            release_active: false,
            chans: [channel.clone(), channel],
            channels: 0,
        }
    }
}

impl Maximizer {
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.allocate();
        self.set_release(self.release_seconds);
    }

    pub fn set_num_channels(&mut self, channels: usize) {
        self.channels = channels.min(2);
        self.allocate();
    }

    pub fn set_threshold(&mut self, db: f32) {
        self.threshold = decibels_to_gain(db);
        self.recalc_makeup();
    }

    pub fn set_ceiling(&mut self, db: f32) {
        self.ceiling = decibels_to_gain(db);
        self.recalc_makeup();
    }

    pub fn set_release(&mut self, seconds: f32) {
        self.release_seconds = seconds;
        self.release_active = seconds > 0.0;
        self.release = if self.release_active {
            decibels_to_gain((-6.0 / (f64::from(seconds) * self.sample_rate)) as f32)
        } else {
            1.0
        };
    }

    /// `allocate`: every channel's buffer sized for the longest half cycle.
    /// A buffer already that size is cleared rather than made again.
    fn allocate(&mut self) {
        self.buffer_size = ((f64::from(self.max_cycle_length) * self.sample_rate) as i32).max(1);
        for c in &mut self.chans[..self.channels] {
            if c.buffer.len() == self.buffer_size as usize {
                c.buffer.fill(0.0);
            } else {
                c.buffer = vec![0.0; self.buffer_size as usize];
            }
        }
        self.reset();
    }

    fn recalc_makeup(&mut self) {
        self.makeup = if self.threshold > 0.0 {
            self.ceiling / self.threshold
        } else {
            1.0
        };
    }

    pub fn reset(&mut self) {
        for c in &mut self.chans[..self.channels] {
            c.buffer.fill(0.0);
            c.length = 0;
            c.max_sample = 0.0;
            c.gain = 1.0;
            c.last_positive = false;
        }
        self.buffer_pos = 0;
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        if self.buffer_size <= 0 {
            return;
        }
        let size = self.buffer_size;
        let samples = left.len().min(right.len());
        let n = self.channels;
        for i in 0..samples {
            let pos = self.buffer_pos;
            for (ch, c) in self.chans[..n].iter_mut().enumerate() {
                let input = if ch == 0 { left[i] } else { right[i] };
                c.buffer[pos as usize] = input;
                let positive = input > 0.0;
                c.max_sample = std_max(c.max_sample, input.abs());
                if positive != c.last_positive {
                    let needed = if c.max_sample > self.threshold {
                        self.threshold / c.max_sample
                    } else {
                        1.0
                    };
                    c.gain = std_min(needed, c.gain);
                    let mut back = c.length;
                    while back >= 0 {
                        let mut idx = pos - back;
                        if idx < 0 {
                            idx += size;
                        }
                        c.buffer[idx as usize] *= c.gain;
                        back -= 1;
                    }
                    c.length = -1;
                    c.max_sample = 0.0;
                }
                c.length += 1;
                if c.length >= size {
                    c.length = 0;
                    c.max_sample = 0.0;
                }
                c.last_positive = positive;
            }

            self.buffer_pos += 1;
            if self.buffer_pos >= size {
                self.buffer_pos = 0;
            }

            let read = self.buffer_pos as usize;
            for (ch, c) in self.chans[..n].iter_mut().enumerate() {
                c.gain = if self.release_active {
                    std_min(c.gain / self.release, 1.0)
                } else {
                    1.0
                };
                let out = jlimit(-SAFETY_LIMIT, SAFETY_LIMIT, c.buffer[read] * self.makeup);
                if ch == 0 {
                    left[i] = out;
                } else {
                    right[i] = out;
                }
            }
        }
    }
}
