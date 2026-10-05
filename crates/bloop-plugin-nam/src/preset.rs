//! The model shapes NAM's trainer offers, as `.nam` text with made-up
//! weights: what the benchmarks time and the gold tests compare, since a
//! shape costs the same whatever was learned into it.
//!
//! The four WaveNet sizes are the trainer's `standard`, `lite`, `feather`
//! and `nano` (neural-amp-modeler, `nam/train/core.py`); the LSTMs are the
//! sizes people train at.

use std::fmt::Write as _;

/// One of the shapes the trainer offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Standard,
    Lite,
    Feather,
    Nano,
    /// The nano size with gated layers (`"gated": true`), which older
    /// trainer versions offered: twice the convolution's channels, the
    /// first half through `tanh` times the second through a sigmoid.
    GatedNano,
    /// The nano size with a kernel of one: no look back at all, every
    /// layer a 1x1.
    KernelOne,
    /// The nano size with another activation, by the name a `.nam` file
    /// gives it.
    NanoWith(&'static str),
    /// One LSTM layer of this many hidden units.
    Lstm(usize),
    /// Two LSTM layers of this many hidden units each.
    Lstm2(usize),
}

impl Shape {
    pub const WAVENETS: [Self; 4] = [Self::Standard, Self::Lite, Self::Feather, Self::Nano];

    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Standard => "standard".into(),
            Self::Lite => "lite".into(),
            Self::Feather => "feather".into(),
            Self::Nano => "nano".into(),
            Self::GatedNano => "gated-nano".into(),
            Self::KernelOne => "kernel-1".into(),
            Self::NanoWith(activation) => format!("nano-{activation}"),
            Self::Lstm(h) => format!("lstm-{h}"),
            Self::Lstm2(h) => format!("lstm-2x{h}"),
        }
    }

    /// The `.nam` file of this shape, its weights drawn from `seed`.
    #[must_use]
    pub fn nam(self, seed: u64) -> String {
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut weight = move |scale: f32| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let unit = (rng >> 40) as f32 / (1u64 << 24) as f32;
            (unit * 2.0 - 1.0) * scale
        };
        let mut weights = Vec::new();
        let lstm = match self {
            Self::Lstm(h) => Some((h, 1)),
            Self::Lstm2(h) => Some((h, 2)),
            _ => None,
        };
        let config = if let Some((hidden, layers)) = lstm {
            let scale = 1.0 / (hidden as f32).sqrt();
            for layer in 0..layers {
                let input = if layer == 0 { 1 } else { hidden };
                let count = 4 * hidden * (input + hidden) + 4 * hidden + 2 * hidden;
                weights.extend((0..count).map(|_| weight(scale)));
            }
            weights.extend((0..=hidden).map(|_| weight(scale)));
            format!(
                r#""architecture": "LSTM", "config": {{"input_size": 1, "hidden_size": {hidden}, "num_layers": {layers}}}"#
            )
        } else {
            let (c1, c2, d1, d2): (usize, usize, &[usize], &[usize]) = match self {
                Self::Standard => (16, 8, &STANDARD, &STANDARD),
                Self::Lite => (12, 6, &LITE_1, &LITE_2),
                Self::Feather => (8, 4, &LITE_1, &LITE_2),
                _ => (4, 2, &LITE_1, &LITE_2),
            };
            let gated = self == Self::GatedNano;
            let g = if gated { 2 } else { 1 };
            let kernel = if self == Self::KernelOne { 1 } else { 3 };
            let activation = if let Self::NanoWith(a) = self { a } else { "Tanh" };
            let arrays = [(1, c1, c2, d1, false), (c1, c2, 1, d2, true)];
            let mut layers = String::new();
            for (i, &(input, channels, head, dilations, head_bias)) in arrays.iter().enumerate() {
                let scale = 1.0 / (channels as f32 * 3.0).sqrt();
                let count = input * channels
                    + dilations.len()
                        * (kernel * channels * g * channels
                            + 2 * g * channels
                            + channels * channels
                            + channels)
                    + channels * head
                    + usize::from(head_bias) * head;
                weights.extend((0..count).map(|_| weight(scale)));
                if i > 0 {
                    layers.push_str(", ");
                }
                let _ = write!(
                    layers,
                    r#"{{"input_size": {input}, "condition_size": 1, "head_size": {head}, "channels": {channels}, "kernel_size": {kernel}, "dilations": {dilations:?}, "activation": "{activation}", "gated": {gated}, "head_bias": {head_bias}}}"#
                );
            }
            weights.push(0.02);
            format!(
                r#""architecture": "WaveNet", "config": {{"layers": [{layers}], "head": null, "head_scale": 0.02}}"#
            )
        };
        let mut text = String::new();
        let _ = write!(
            text,
            r#"{{"version": "0.5.4", {config}, "sample_rate": 48000, "weights": ["#
        );
        for (i, w) in weights.iter().enumerate() {
            if i > 0 {
                text.push_str(", ");
            }
            let _ = write!(text, "{w:?}");
        }
        text.push_str("]}");
        text
    }
}

const STANDARD: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
const LITE_1: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const LITE_2: [usize; 13] = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
