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
    /// The trainer's A2-Full: one array of 23 layers, 8 channels, kernels
    /// of 6 and 15, `LeakyReLU`, a head convolution of 16 taps.
    A2Full,
    /// A2-Lite: the same with 3 channels.
    A2Lite,
    /// A2-Lite and A2-Full in a `SlimmableContainer`, as A2 captures ship.
    A2Container,
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
            Self::A2Full => "a2-full".into(),
            Self::A2Lite => "a2-lite".into(),
            Self::A2Container => "a2-container".into(),
        }
    }

    /// The `.nam` file of this shape, its weights drawn from `seed`.
    #[must_use]
    pub fn nam(self, seed: u64) -> String {
        match self {
            Self::A2Full => return a2(8, seed),
            Self::A2Lite => return a2(3, seed),
            Self::A2Container => {
                return format!(
                    r#"{{"version": "0.7.0", "architecture": "SlimmableContainer", "config": {{"submodels": [{{"max_value": 0.5, "model": {}}}, {{"max_value": 1.0, "model": {}}}]}}, "weights": [], "sample_rate": 48000}}"#,
                    a2(3, seed),
                    a2(8, seed + 1)
                )
            }
            _ => {}
        }
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

/// A2's layers: the kernel sizes and dilations the trainer gives both
/// A2 sizes (neural-amp-modeler v0.13; `NAM/wavenet/a2_fast.h`).
const A2_KERNELS: [usize; 23] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
];
const A2_DILATIONS: [usize; 23] = [
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
];

/// An A2 WaveNet of `c` channels, as the trainer exports one, its weights
/// drawn from `seed`.
fn a2(c: usize, seed: u64) -> String {
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut weight = move |scale: f32| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let unit = (rng >> 40) as f32 / (1u64 << 24) as f32;
        (unit * 2.0 - 1.0) * scale
    };
    let mut weights: Vec<f32> = (0..c).map(|_| weight(1.0)).collect();
    for &k in &A2_KERNELS {
        let scale = 1.0 / ((k * c) as f32).sqrt();
        weights.extend((0..k * c * c + c).map(|_| weight(scale)));
        weights.extend((0..c).map(|_| weight(0.5)));
        let scale = 1.0 / (c as f32).sqrt();
        weights.extend((0..c * c + c).map(|_| weight(scale)));
    }
    let scale = 1.0 / ((16 * c) as f32).sqrt();
    weights.extend((0..=16 * c).map(|_| weight(scale)));
    weights.push(0.01);
    let film = r#"{"active": false, "shift": true, "groups": 1}"#;
    let films: Vec<String> = [
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    ]
    .iter()
    .map(|k| format!(r#""{k}": {film}"#))
    .collect();
    let leaky = vec![r#"{"type": "LeakyReLU", "negative_slope": 0.01}"#; 23].join(", ");
    let none = vec![r#""none""#; 23].join(", ");
    let nulls = vec!["null"; 23].join(", ");
    let mut text = format!(
        r#"{{"version": "0.7.0", "metadata": {{"loudness": -20.0}}, "architecture": "WaveNet", "config": {{"layers": [{{"input_size": 1, "condition_size": 1, "head": {{"out_channels": 1, "kernel_size": 16, "bias": true}}, "channels": {c}, "kernel_sizes": {A2_KERNELS:?}, "dilations": {A2_DILATIONS:?}, "activation": [{leaky}], "bottleneck": {c}, "head1x1": {{"active": false, "out_channels": 1, "groups": 1}}, "layer1x1": {{"active": true, "groups": 1}}, "groups_input": 1, "groups_input_mixin": 1, {}, "gating_mode": [{none}], "secondary_activation": [{nulls}], "slimmable": null}}], "head": null, "head_scale": 0.01}}, "sample_rate": 48000, "weights": ["#,
        films.join(", ")
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
