//! The transcendentals every port calls, and which library computes them.
//!
//! `std`'s `exp` and friends are the platform C library's on native targets
//! and a Rust reimplementation on wasm32, and the two disagree in the last
//! ulp — enough to move a rendered hash. So by default these are `libm`'s,
//! which is the same code on every target.
//!
//! The C++ originals call the C library, though, and a bit-exact comparison
//! against them is only a comparison of *algorithms* if both sides use the
//! same one. With the `gold` feature these forward to `std`, which on the
//! machine running the gold tests is that same C library. The algorithm is
//! then proven against the original, and what ships differs from it only in
//! which library rounds the last bit.
//!
//! Functions IEEE 754 defines exactly (`sqrt`, `abs`, `floor`) need neither
//! and are called directly.

macro_rules! pick {
    ($(#[$doc:meta])* $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty, $std:expr, $libm:path) => {
        $(#[$doc])*
        #[inline]
        #[must_use]
        pub fn $name($($arg: $ty),*) -> $ret {
            #[cfg(feature = "gold")]
            {
                $std
            }
            #[cfg(not(feature = "gold"))]
            {
                $libm($($arg),*)
            }
        }
    };
}

pick!(expf(x: f32) -> f32, x.exp(), libm::expf);
pick!(exp(x: f64) -> f64, x.exp(), libm::exp);
pick!(logf(x: f32) -> f32, x.ln(), libm::logf);
pick!(log10f(x: f32) -> f32, x.log10(), libm::log10f);
pick!(powf(x: f32, y: f32) -> f32, x.powf(y), libm::powf);
pick!(sinf(x: f32) -> f32, x.sin(), libm::sinf);
pick!(cosf(x: f32) -> f32, x.cos(), libm::cosf);
pick!(atanf(x: f32) -> f32, x.atan(), libm::atanf);

/// `std::max (a, b)` as C++ defines it, `(a < b) ? b : a` — which differs
/// from `f32::max` on a NaN and on the sign of a zero.
#[inline]
#[must_use]
pub fn std_max(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

/// `std::min (a, b)`: `(b < a) ? b : a`.
#[inline]
#[must_use]
pub fn std_min(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}
