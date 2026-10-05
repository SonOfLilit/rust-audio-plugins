//! The transcendentals the ports call, and which library computes them;
//! and the pieces every Airwindows plug-in shares: its seeds, its denormal
//! guard and its 32-bit floating-point dither.
//!
//! `std`'s `sin` and friends are the platform C library's on native targets
//! and a Rust reimplementation on wasm32, and the two disagree in the last
//! ulp — enough to move a rendered hash. So by default these are `libm`'s,
//! which is the same code on every target.
//!
//! The C++ originals call the C library, though, and a bit-exact comparison
//! against them is only a comparison of *algorithms* if both sides use the
//! same one. With the `gold` feature these forward to `std`, which on the
//! machine running the gold tests is that same C library.
//!
//! Functions IEEE 754 defines exactly (`sqrt`, `abs`, `floor`) need neither
//! and are called directly.

macro_rules! pick {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty, $std:expr, $libm:path) => {
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

pick!(sin(x: f64) -> f64, x.sin(), libm::sin);
pick!(cos(x: f64) -> f64, x.cos(), libm::cos);
pick!(tan(x: f64) -> f64, x.tan(), libm::tan);
pick!(asin(x: f64) -> f64, x.asin(), libm::asin);
pick!(log(x: f64) -> f64, x.ln(), libm::log);
pick!(pow(x: f64, y: f64) -> f64, x.powf(y), libm::pow);

/// `fpd = 1.0; while (fpd < 16386) fpd = rand() * UINT32_MAX;` — the way
/// every Airwindows constructor seeds its dither, `rand()`'s `int` taken as
/// unsigned and multiplied modulo 2^32.
const fn seed(rand: u32) -> u32 {
    rand.wrapping_mul(u32::MAX)
}

/// The left and right seeds of a plug-in constructed first thing in a
/// fresh process after `srand(1)`: glibc's `rand()` then returns 1804289383
/// and 846930886, both of which pass the `< 16386` test on the first draw.
/// The originals take whatever `rand()` gives; a bloop instance must sound
/// the same every time, so every instance starts where that one does.
pub const SEEDS: (u32, u32) = (seed(1_804_289_383), seed(846_930_886));

/// The step every Airwindows plug-in takes its `fpd` through.
#[inline]
pub fn xorshift(fpd: &mut u32) {
    *fpd ^= *fpd << 13;
    *fpd ^= *fpd >> 17;
    *fpd ^= *fpd << 5;
}

/// `if (fabs(x) < 1.18e-23) x = fpd * 1.18e-17;` — a sample too quiet to
/// matter is swapped for a tiny noise that keeps the filters out of
/// denormals.
#[inline]
#[must_use]
pub fn guard(x: f32, fpd: u32) -> f64 {
    let x = f64::from(x);
    if x.abs() < 1.18e-23 {
        f64::from(fpd) * 1.18e-17
    } else {
        x
    }
}

/// The exponent glibc's `frexpf` stores: `x == m * 2^e` with `0.5 <= |m| <
/// 1`, and 0 for zero, infinities and NaN.
#[inline]
#[must_use]
pub fn frexpf_exponent(x: f32) -> i32 {
    let bits = x.to_bits() & 0x7fff_ffff;
    let biased = (bits >> 23) as i32;
    if bits == 0 || biased == 0xff {
        0
    } else if biased == 0 {
        (32 - bits.leading_zeros() as i32) - 149
    } else {
        biased - 126
    }
}

/// `pow(2, n)`, exactly, for an `n` whose power of two is a normal double
/// (the dither's `expon + 62` always is). Both C libraries return the exact
/// power for these; building it from bits says so without a library call.
#[inline]
#[must_use]
fn exp2i(n: i32) -> f64 {
    f64::from_bits(((n + 1023) as u64) << 52)
}

/// The 32-bit stereo floating-point dither every plug-in ends on, one
/// channel of it:
///
/// ```c
/// int expon; frexpf((float)x, &expon);
/// fpd ^= fpd << 13; fpd ^= fpd >> 17; fpd ^= fpd << 5;
/// x += ((double(fpd)-uint32_t(0x7fffffff)) * 5.5e-36l * pow(2,expon+62));
/// ```
///
/// `5.5e-36l` is a `long double`, whose width is the platform's: 80-bit on
/// x86 Linux, 64-bit on Apple's arm64 and on MSVC. The port takes it as
/// 64-bit, a `double`, and the gold harness is built with
/// `-mlong-double-64` to match.
#[inline]
#[must_use]
pub fn dither(x: f64, fpd: &mut u32) -> f64 {
    let expon = frexpf_exponent(x as f32);
    xorshift(fpd);
    x + ((f64::from(*fpd) - 2_147_483_647.0) * 5.5e-36 * exp2i(expon + 62))
}

#[cfg(test)]
#[expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]
mod tests {
    use super::*;
    use expect_test::expect;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn seeds_are_what_srand_1_gives() {
        expect![[r#"
            (
                2490677913,
                3448036410,
            )
        "#]]
        .assert_debug_eq(&SEEDS);
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn frexpf_exponent_matches_frexp_on_every_kind_of_float() {
        let cases = [
            0.0f32,
            -0.0,
            1.0,
            0.5,
            0.75,
            -3.0,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x0040_0000),
            f32::INFINITY,
            f32::NAN,
        ];
        let shown: Vec<String> = cases
            .iter()
            .map(|&x| format!("{x:e} -> {}", frexpf_exponent(x)))
            .collect();
        expect![[r#"
            0e0 -> 0
            -0e0 -> 0
            1e0 -> 1
            5e-1 -> 0
            7.5e-1 -> 0
            -3e0 -> 2
            3.4028235e38 -> 128
            1.1754944e-38 -> -125
            1e-45 -> -148
            5.877472e-39 -> -126
            inf -> 0
            NaN -> 0"#]]
        .assert_eq(&shown.join("\n"));
        for x in cases.into_iter().filter(|x| x.is_finite() && *x != 0.0) {
            let (m, e) = libm::frexpf(x);
            assert_eq!(e, frexpf_exponent(x), "{x}");
            assert!((0.5..1.0).contains(&m.abs()), "{x}");
        }
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn exp2i_is_the_power_of_two() {
        for n in -86..=190 {
            assert_eq!(exp2i(n), libm::pow(2.0, f64::from(n)), "{n}");
        }
    }
}
