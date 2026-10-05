//! The activation functions a `.nam` model names, as NAM's core computes
//! them (`NAM/activations.h`).
//!
//! `Tanh` has two forms. [`Tanh::Exact`] is the C library's `tanhf` —
//! here `libm`'s, so a render is the same bits on wasm as on native — and
//! [`Tanh::Fast`] is the rational approximation NAM's own plug-in switches
//! on (`Activation::enable_fast_tanh`), which is a few multiplies and one
//! divide and vectorizes. Which one a model runs is the render's choice,
//! not the model's: see [`crate::Quality`].

/// Which `tanh` a model's `Tanh` layers run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tanh {
    /// `tanhf`, to the last bit the maths library gives.
    Exact,
    /// NAM's `fast_tanh`: within 2e-4 of `tanhf` everywhere.
    Fast,
}

/// One activation, as a `.nam` file names it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    Tanh,
    Hardtanh,
    Relu,
    LeakyRelu(f32),
    Sigmoid,
    Silu,
    Hardswish,
    Softsign,
}

impl Activation {
    /// The activation a `.nam` file calls `name`.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Some(match name {
            "Tanh" | "Fasttanh" => Self::Tanh,
            "Hardtanh" => Self::Hardtanh,
            "ReLU" => Self::Relu,
            "LeakyReLU" => Self::LeakyRelu(0.01),
            "Sigmoid" => Self::Sigmoid,
            "SiLU" => Self::Silu,
            "Hardswish" => Self::Hardswish,
            "Softsign" => Self::Softsign,
            _ => return None,
        })
    }

    /// Applies it to every value of `data` in place.
    pub fn apply(self, data: &mut [f32], tanh: Tanh) {
        match self {
            Self::Tanh => match tanh {
                Tanh::Exact => data.iter_mut().for_each(|x| *x = libm::tanhf(*x)),
                Tanh::Fast => data.iter_mut().for_each(|x| *x = fast_tanh(*x)),
            },
            Self::Hardtanh => data.iter_mut().for_each(|x| *x = hard_tanh(*x)),
            Self::Relu => data.iter_mut().for_each(|x| *x = if *x > 0.0 { *x } else { 0.0 }),
            Self::LeakyRelu(slope) => data
                .iter_mut()
                .for_each(|x| *x = if *x > 0.0 { *x } else { slope * *x }),
            Self::Sigmoid => data.iter_mut().for_each(|x| *x = sigmoid(*x)),
            Self::Silu => data.iter_mut().for_each(|x| *x *= sigmoid(*x)),
            Self::Hardswish => data.iter_mut().for_each(|x| *x = hardswish(*x)),
            Self::Softsign => data.iter_mut().for_each(|x| *x /= 1.0 + x.abs()),
        }
    }
}

/// NAM's `fast_tanh`, operation for operation.
#[must_use]
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;
    (x * (2.455_507_507_029_56 + 2.455_507_507_029_56 * ax + (0.893_229_853_513_558 + 0.821_226_666_969_744 * ax) * x2))
        / (2.445_066_346_522_99 + (2.445_066_346_522_99 + x2) * (x + 0.814_642_734_961_073 * x * ax).abs())
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
    let t = x + 3.0;
    let clamped = if t < 0.0 {
        0.0
    } else if t > 6.0 {
        6.0
    } else {
        t
    };
    x * clamped * (1.0 / 6.0)
}
