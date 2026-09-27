//! Jeremy Pepper's crossfeed filter (BSD-3; see LICENSE), which slPlugins'
//! Crossfeed plug-in wraps.
//!
//! Ported from slPlugins' `3rdparty/crossfeed/crossfeed.{h,c}`.

/// The kernels are C double literals initialising a float array, so each
/// is rounded to double first and then to float — which is not always the
/// float nearest the decimal. (A bare `0.1 as f32` would not do it: Rust
/// types a literal cast straight to f32 as an f32.)
macro_rules! kernel {
    ($($x:expr),* $(,)?) => { [$((($x) as f64) as f32),*] };
}

const KERNEL_96K: [f32; 22] = kernel![
    1.0 - 0.007_385_683_2,
    -0.007_519_417_4,
    -0.007_722_332_6,
    -0.007_890_662_2,
    -0.008_164_638_7,
    -0.008_391_402_7,
    -0.008_781_943_5,
    -0.009_115_370_9,
    -0.009_704_460_4,
    -0.010_244_164,
    -0.011_201_29,
    -0.012_166_876,
    -0.013_881_951,
    -0.015_828_054,
    -0.019_321_838,
    -0.023_897_322,
    -0.032_408_956,
    -0.045_482_289,
    -0.070_983_656,
    -0.112_067_52,
    -0.163_623_41,
    -0.121_029_93,
];

const KERNEL_48K: [f32; 11] = kernel![
    1.0 - 0.013_584_239,
    -0.014_855_062,
    -0.015_109_27,
    -0.017_407_341,
    -0.018_212_127,
    -0.023_438_375,
    -0.027_174_231,
    -0.044_839_676,
    -0.071_772_039,
    -0.192_037_54,
    -0.278_241_46,
];

const KERNEL_44K: [f32; 10] = kernel![
    1.0 - 0.015_422_851,
    -0.015_586_1,
    -0.017_845_599,
    -0.018_381_938,
    -0.023_416_32,
    -0.026_318_349,
    -0.043_148_093,
    -0.066_815_346,
    -0.189_797_33,
    -0.297_861_13,
];

/// `crossfeed_t`: mid passes, side is filtered, at one of three rates.
#[derive(Clone, Debug)]
pub struct Crossfeed {
    mid: [f32; 25],
    side: [f32; 25],
    filter: &'static [f32],
    pos: usize,
}

impl Crossfeed {
    /// `crossfeed_init`, for the rates it has kernels for.
    #[must_use]
    pub fn new(sample_rate: i32) -> Option<Self> {
        let filter: &'static [f32] = match sample_rate {
            44100 => &KERNEL_44K,
            48000 => &KERNEL_48K,
            96000 => &KERNEL_96K,
            _ => return None,
        };
        Some(Self {
            mid: [0.0; 25],
            side: [0.0; 25],
            filter,
            pos: 0,
        })
    }

    /// `crossfeed_filter_inplace_noninterleaved`.
    pub fn filter_in_place(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = self.filter.len();
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mid = (*l + *r) / 2.0;
            let side = (*l - *r) / 2.0;
            let mut oside = 0.0f32;
            self.mid[self.pos % len] = mid;
            self.side[self.pos] = side;
            for (i, tap) in self.filter.iter().enumerate() {
                oside += self.side[(self.pos + len - i) % len] * tap;
            }
            *l = self.mid[self.pos] + oside;
            *r = self.mid[self.pos] - oside;
            self.pos = (self.pos + 1) % len;
        }
    }
}
