//! Faithful transcriptions of the handful of JUCE facilities the ported voice
//! relies on for its exact output.
//!
//! Everything here mirrors JUCE 8 (`modules/juce_audio_basics`) line for line.
//! The C++ gold harness in `tests/cpp_harness` reimplements the same pieces, so
//! any drift shows up as a gold comparison failure rather than silently
//! detuning the port.

/// `juce::Decibels::decibelsToGain`, with JUCE's default -100 dB floor.
pub fn decibels_to_gain(decibels: f32) -> f32 {
    const DEFAULT_MINUS_INFINITY_DB: f32 = -100.0;
    if decibels > DEFAULT_MINUS_INFINITY_DB {
        libm::powf(10.0, decibels * 0.05)
    } else {
        0.0
    }
}

/// `juce::jmap` — remap a value from one range onto another.
pub fn jmap(
    source_value: f32,
    source_min: f32,
    source_max: f32,
    target_min: f32,
    target_max: f32,
) -> f32 {
    target_min
        + ((target_max - target_min) * (source_value - source_min)) / (source_max - source_min)
}

/// `juce::approximatelyEqual` with `Tolerance<float>().withAbsolute(t)`.
///
/// JUCE's absolute-tolerance comparison is `std::abs(a - b) <= tolerance`, with
/// exact equality short-circuiting so that infinities compare equal.
pub fn approximately_equal_absolute(a: f32, b: f32, tolerance: f32) -> bool {
    a == b || (a - b).abs() <= tolerance
}

/// `juce::IIRCoefficients` — five normalised biquad coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IirCoefficients {
    pub c: [f32; 5],
}

impl IirCoefficients {
    /// JUCE's six-argument constructor, which normalises by `c4`.
    fn new(c1: f64, c2: f64, c3: f64, c4: f64, c5: f64, c6: f64) -> Self {
        let a = 1.0 / c4;
        Self {
            c: [
                (c1 * a) as f32,
                (c2 * a) as f32,
                (c3 * a) as f32,
                (c5 * a) as f32,
                (c6 * a) as f32,
            ],
        }
    }

    /// `juce::IIRCoefficients::makeLowPass(sampleRate, frequency, Q)`.
    pub fn make_low_pass(sample_rate: f64, frequency: f64, q: f64) -> Self {
        let n = 1.0 / libm::tan(core::f64::consts::PI * frequency / sample_rate);
        let n_squared = n * n;
        let c1 = 1.0 / (1.0 + 1.0 / q * n + n_squared);

        Self::new(
            c1,
            c1 * 2.0,
            c1,
            1.0,
            c1 * 2.0 * (1.0 - n_squared),
            c1 * (1.0 - 1.0 / q * n + n_squared),
        )
    }
}

/// `juce::SingleThreadedIIRFilter` — a direct-form-II biquad.
///
/// `JUCE_SNAP_TO_ZERO` is applied unconditionally. In JUCE it is compiled in
/// only on Intel targets, which would make the plugin's own output differ
/// between desktop and phone; flushing denormals everywhere keeps Bloop
/// bit-identical across platforms and matches the reference build we compare
/// against.
#[derive(Clone, Copy, Debug, Default)]
pub struct IirFilter {
    coefficients: IirCoefficients,
    v1: f32,
    v2: f32,
}

impl IirFilter {
    pub fn set_coefficients(&mut self, coefficients: IirCoefficients) {
        self.coefficients = coefficients;
    }

    pub fn reset(&mut self) {
        self.v1 = 0.0;
        self.v2 = 0.0;
    }

    pub fn process_single_sample_raw(&mut self, input: f32) -> f32 {
        let c = &self.coefficients.c;
        let mut out = c[0] * input + self.v1;

        // Spelled exactly as JUCE_SNAP_TO_ZERO. A range `contains` check reads
        // better but flushes NaN differently: both comparisons are false for
        // NaN, so this form zeroes it and `contains` would let it through.
        #[allow(
            clippy::manual_range_contains,
            reason = "NaN handling differs from the range form"
        )]
        if !(out < -1.0e-8 || out > 1.0e-8) {
            out = 0.0;
        }

        self.v1 = c[1] * input - c[3] * out + self.v2;
        self.v2 = c[2] * input - c[4] * out;

        out
    }
}
