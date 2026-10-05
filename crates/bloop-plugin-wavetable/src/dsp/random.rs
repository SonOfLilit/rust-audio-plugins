//! The random numbers the original draws: `juce::Random`, and the
//! `std::default_random_engine` and `std::normal_distribution<float>` of
//! libstdc++ that Gin's white noise uses.
//!
//! `juce::Random` is the 48-bit linear congruential generator JUCE
//! documents; the other two are what the C++ standard specifies
//! (`minstd_rand0`) and the polar method libstdc++ draws a normal deviate
//! with, which is what the gold tests' C++ links.

use bloop_plugin_socalabs::shared::math;

/// `juce::Random`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JuceRandom {
    seed: i64,
}

impl JuceRandom {
    #[must_use]
    pub const fn new(seed: i64) -> Self {
        Self { seed }
    }

    pub fn set_seed(&mut self, seed: i64) {
        self.seed = seed;
    }

    /// `nextInt()`.
    pub fn next_int(&mut self) -> i32 {
        let seed = (self.seed as u64)
            .wrapping_mul(0x5_DEEC_E66D)
            .wrapping_add(11)
            & 0xFFFF_FFFF_FFFF;
        self.seed = seed as i64;
        (seed >> 16) as u32 as i32
    }

    /// `nextInt (maxValue)`: 0 to `max - 1`.
    pub fn next_int_below(&mut self, max: i32) -> i32 {
        ((u64::from(self.next_int() as u32) * max as u64) >> 32) as i32
    }

    /// `nextFloat()`: 0 up to, not including, 1.
    pub fn next_float(&mut self) -> f32 {
        let result = (self.next_int() as u32) as f32 / (u32::MAX as f32 + 1.0);
        result.min(1.0 - f32::EPSILON)
    }
}

/// `std::minstd_rand0`, which libstdc++'s `default_random_engine` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinStdRand0 {
    state: u32,
}

impl MinStdRand0 {
    const MODULUS: u64 = 2_147_483_647;

    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut state = u64::from(seed) % Self::MODULUS;
        if state == 0 {
            state = 1;
        }
        Self {
            state: state as u32,
        }
    }

    pub fn generate(&mut self) -> u32 {
        self.state = ((u64::from(self.state) * 16807) % Self::MODULUS) as u32;
        self.state
    }

    /// `std::generate_canonical<float, 24>`: one draw scaled to [0, 1).
    fn canonical(&mut self) -> f32 {
        // One draw covers 24 bits; the range is 2^31 - 2, which in float
        // is 2^31.
        let sum = (self.generate() - 1) as f32;
        let ret = sum / 2_147_483_648.0f32;
        if ret >= 1.0 {
            1.0f32.next_down_compat()
        } else {
            ret
        }
    }
}

trait NextDown {
    fn next_down_compat(self) -> Self;
}

impl NextDown for f32 {
    fn next_down_compat(self) -> Self {
        f32::from_bits(self.to_bits() - 1)
    }
}

/// `gin::WhiteNoise`: `std::normal_distribution<float> (0, 0.1)` over
/// `std::default_random_engine`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhiteNoise {
    engine: MinStdRand0,
    saved: f32,
    saved_available: bool,
}

impl WhiteNoise {
    const MEAN: f32 = 0.0;
    const STDDEV: f32 = 0.1;

    /// The original seeds from `std::random_device`; bloop seeds it, so a
    /// note renders the same every time.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self {
            engine: MinStdRand0::new(seed),
            saved: 0.0,
            saved_available: false,
        }
    }

    pub fn next_sample(&mut self) -> f32 {
        let ret = if self.saved_available {
            self.saved_available = false;
            self.saved
        } else {
            let (mut x, mut y, mut r2);
            loop {
                x = (f64::from(2.0f32 * self.engine.canonical()) - 1.0) as f32;
                y = (f64::from(2.0f32 * self.engine.canonical()) - 1.0) as f32;
                r2 = x * x + y * y;
                if !(f64::from(r2) > 1.0 || r2 == 0.0) {
                    break;
                }
            }
            let mult = (-2.0 * math::logf(r2) / r2).sqrt();
            self.saved = x * mult;
            self.saved_available = true;
            y * mult
        };
        ret * Self::STDDEV + Self::MEAN
    }
}

/// `gin::PinkNoise`: Thomas Merchant's Voss-McCartney generator (MIT),
/// nine octaves over a middle-square Weyl sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinkNoise {
    s: u64,
    x: u64,
    w: u64,
    out: f32,
    octaves: [f32; 9],
    counter: u8,
}

impl Default for PinkNoise {
    fn default() -> Self {
        Self {
            s: 0x2252_e7cd_9884_6fdd,
            x: 0,
            w: 0,
            out: 0.0,
            octaves: [0.0; 9],
            counter: 0,
        }
    }
}

impl PinkNoise {
    fn next_int(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(self.x);
        self.w = self.w.wrapping_add(self.s);
        self.x = self.x.wrapping_add(self.w);
        self.x = self.x.rotate_left(32);
        self.x as u32 as i32
    }

    pub fn next_sample(&mut self) -> f32 {
        let octave = usize::from(trailing_zeros(self.counter));
        self.out -= self.octaves[octave];
        let value = self.next_int() as f32 / i32::MAX as f32;
        self.octaves[octave] = value / (10 - octave) as f32;
        self.out += self.octaves[octave];
        self.counter = self.counter.wrapping_add(1);
        self.out
    }
}

/// The original's table: the trailing zeros of a byte, with 8 for 0.
fn trailing_zeros(counter: u8) -> u8 {
    if counter == 0 {
        8
    } else {
        counter.trailing_zeros() as u8
    }
}
