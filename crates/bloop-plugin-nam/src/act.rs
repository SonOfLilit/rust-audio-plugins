//! The activation functions a `.nam` model names, as NAM's core computes
//! them (`NAM/activations.h`, `NAM/activations.cpp`).
//!
//! `Tanh` has two forms. [`Tanh::Exact`] is the C library's `tanhf` —
//! here `libm`'s, so a render is the same bits on wasm as on native — and
//! [`Tanh::Fast`] is the rational approximation NAM's own plug-in switches
//! on (`Activation::enable_fast_tanh`), which is a few multiplies and one
//! divide and vectorizes. Which one a model runs is the render's choice,
//! not the model's.

use serde_json::Value;

/// Which `tanh` a model's `Tanh` layers run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tanh {
    /// `tanhf`, to the last bit the maths library gives.
    Exact,
    /// NAM's `fast_tanh`: within 4.4e-4 of `tanhf` everywhere.
    Fast,
}

/// One activation, as a `.nam` file names it.
#[derive(Clone, Debug, PartialEq)]
pub enum Activation {
    /// What a layer without a gate has in its gate's place: nothing.
    Identity,
    /// `tanh`, the render's choice of which ([`Tanh`]).
    Tanh,
    /// `"Fasttanh"`: a model that names it runs the fast one always, as
    /// in NAM.
    FastTanh,
    Hardtanh,
    /// The identity between `min` and `max`, and lines of slope
    /// `min_slope` and `max_slope` beyond them.
    LeakyHardtanh {
        min: f32,
        max: f32,
        min_slope: f32,
        max_slope: f32,
    },
    Relu,
    LeakyRelu(f32),
    /// A leaky `ReLU` with a slope per channel, or one for all.
    PRelu(Box<[f32]>),
    Sigmoid,
    Silu,
    Hardswish,
    Softsign,
}

impl Activation {
    /// The activation a `.nam` file calls `name`, with NAM's defaults for
    /// the ones that take parameters.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Some(match name {
            "Tanh" => Self::Tanh,
            "Fasttanh" => Self::FastTanh,
            "Hardtanh" => Self::Hardtanh,
            "LeakyHardtanh" | "LeakyHardTanh" => Self::LeakyHardtanh {
                min: -1.0,
                max: 1.0,
                min_slope: 0.01,
                max_slope: 0.01,
            },
            "ReLU" => Self::Relu,
            "LeakyReLU" => Self::LeakyRelu(0.01),
            "PReLU" => Self::PRelu(Box::new([0.01])),
            "Sigmoid" => Self::Sigmoid,
            "SiLU" => Self::Silu,
            "Hardswish" => Self::Hardswish,
            "Softsign" => Self::Softsign,
            _ => return None,
        })
    }

    /// An activation as a `.nam` file gives it: a name, or an object with
    /// a `type` and whichever parameters that type takes
    /// (`ActivationConfig::from_json`).
    ///
    /// # Errors
    /// Neither, or a type NAM does not have.
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let unknown = |name: &str| format!("bloop does not run the {name} activation yet");
        if let Some(name) = value.as_str() {
            return Self::named(name).ok_or_else(|| unknown(name));
        }
        let Some(name) = value.get("type").and_then(Value::as_str) else {
            return Err("an activation is neither a name nor an object with a type".into());
        };
        let mut activation = Self::named(name).ok_or_else(|| unknown(name))?;
        let number = |key: &str| value.get(key).and_then(Value::as_f64).map(|v| v as f32);
        match &mut activation {
            Self::LeakyRelu(slope) => {
                if let Some(s) = number("negative_slope") {
                    *slope = s;
                }
            }
            Self::PRelu(slopes) => {
                if let Some(s) = number("negative_slope") {
                    *slopes = Box::new([s]);
                } else if let Some(list) = value.get("negative_slopes") {
                    let list = list
                        .as_array()
                        .ok_or("a PReLU's negative_slopes is not a list")?
                        .iter()
                        .map(|v| v.as_f64().map(|v| v as f32))
                        .collect::<Option<Vec<f32>>>()
                        .ok_or("a PReLU's negative slope is not a number")?;
                    if list.is_empty() {
                        return Err("a PReLU has no negative slopes".into());
                    }
                    *slopes = list.into_boxed_slice();
                }
            }
            Self::LeakyHardtanh {
                min,
                max,
                min_slope,
                max_slope,
            } => {
                for (key, slot) in [
                    ("min_val", min),
                    ("max_val", max),
                    ("min_slope", min_slope),
                    ("max_slope", max_slope),
                ] {
                    if let Some(v) = number(key) {
                        *slot = v;
                    }
                }
            }
            _ => {}
        }
        Ok(activation)
    }

    /// Whether it fits `rows` channels: a `PReLU` has one slope for all of
    /// them or one for each. `per_channel` is NAM's gate, which reads a
    /// slope per channel whatever it was given.
    pub(crate) fn check_channels(&self, rows: usize, per_channel: bool) -> Result<(), String> {
        match self {
            Self::PRelu(slopes) if slopes.len() != rows && (per_channel || slopes.len() != 1) => {
                Err(format!(
                    "a PReLU has {} negative slopes for {rows} channels",
                    slopes.len()
                ))
            }
            _ => Ok(()),
        }
    }

    /// Applies it in place to `data`, which is channel `channel`'s row.
    pub fn apply(&self, data: &mut [f32], channel: usize, tanh: Tanh) {
        match self {
            Self::Identity => {}
            Self::Tanh => match tanh {
                Tanh::Exact => data.iter_mut().for_each(|x| *x = libm::tanhf(*x)),
                Tanh::Fast => data.iter_mut().for_each(|x| *x = fast_tanh(*x)),
            },
            Self::FastTanh => data.iter_mut().for_each(|x| *x = fast_tanh(*x)),
            Self::Hardtanh => data.iter_mut().for_each(|x| *x = hard_tanh(*x)),
            &Self::LeakyHardtanh {
                min,
                max,
                min_slope,
                max_slope,
            } => data.iter_mut().for_each(|x| {
                *x = if *x < min {
                    (*x - min) * min_slope + min
                } else if *x > max {
                    (*x - max) * max_slope + max
                } else {
                    *x
                };
            }),
            Self::Relu => data
                .iter_mut()
                .for_each(|x| *x = if *x > 0.0 { *x } else { 0.0 }),
            &Self::LeakyRelu(slope) => leaky(data, slope),
            Self::PRelu(slopes) => leaky(data, slopes[channel % slopes.len()]),
            Self::Sigmoid => data.iter_mut().for_each(|x| *x = sigmoid(*x)),
            Self::Silu => data.iter_mut().for_each(|x| *x *= sigmoid(*x)),
            Self::Hardswish => data.iter_mut().for_each(|x| *x = hardswish(*x)),
            Self::Softsign => data.iter_mut().for_each(|x| *x /= 1.0 + x.abs()),
        }
    }
}

#[inline]
fn leaky(data: &mut [f32], slope: f32) {
    data.iter_mut()
        .for_each(|x| *x = if *x > 0.0 { *x } else { slope * *x });
}

/// NAM's `fast_tanh`, operation for operation.
#[must_use]
#[inline]
#[expect(
    clippy::excessive_precision,
    reason = "NAM's digits, which round to the same floats as fewer would"
)]
pub fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;
    (x * (2.455_507_507_029_56
        + 2.455_507_507_029_56 * ax
        + (0.893_229_853_513_558 + 0.821_226_666_969_744 * ax) * x2))
        / (2.445_066_346_522_99
            + (2.445_066_346_522_99 + x2) * (x + 0.814_642_734_961_073 * x * ax).abs())
}

/// NAM's `fast_sigmoid`: what the LSTM's gates run with fast `tanh` on.
#[must_use]
#[inline]
pub fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (fast_tanh(x * 0.5) + 1.0)
}

#[must_use]
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[inline]
fn hard_tanh(x: f32) -> f32 {
    let t = if x < -1.0 { -1.0 } else { x };
    if t > 1.0 {
        1.0
    } else {
        t
    }
}

#[inline]
fn hardswish(x: f32) -> f32 {
    x * (x + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)
}
