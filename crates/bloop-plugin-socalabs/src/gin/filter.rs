//! `gin::Filter`, and the part of the `AudioFilter` library (MIT; see
//! LICENSE) under it that the Graphic EQ reaches.
//!
//! Ported from Gin's `gin_dsp/dsp/gin_filter.h` and `AudioFilter`'s
//! `AudioFilterTypes.h`, `FilterInstance.h` and `ParametricCreator.cpp`.
//! A stage is designed by the matched z-transform with the `MZTi` FIR
//! correction (Guiness and Chauhan) and run as a state-variable filter.
//! The three types the Graphic EQ uses are ported — bandpass, lowpass and
//! highpass — and the notch SocaLabs' Wavetable adds, and of
//! `getMztTransform` the branches those reach with a Q above 0.5: complex
//! poles, and zeros either absent, at DC, a double zero at DC, or the
//! notch's complex pair on the unit circle.

use crate::juce::gain_to_decibels;
use crate::math;
use core::f64::consts::PI;

/// `gin::Q`, the Butterworth Q the second stage of a 24 dB filter uses:
/// the original's `0.70710678f`, which is this as an `f32`.
const BUTTERWORTH_Q: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// `AudioFilter::BiquadParam`: s- or z-domain biquad coefficients, with the
/// FIR that corrects a matched-z design.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BiquadParam {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    use_compensation: bool,
}

impl Default for BiquadParam {
    fn default() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            c0: 1.0,
            c1: 0.0,
            c2: 0.0,
            use_compensation: false,
        }
    }
}

/// The filter types ported, as `AudioFilter::FilterType` names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterType {
    LoPass,
    HiPass,
    BandPass,
    Notch,
}

/// `getAnalog`: the s-domain prototype.
fn analog(q: f64, gain: f64, kind: FilterType) -> BiquadParam {
    // `A` is worked out for every type; these three do not use it.
    let _ = math::pow(10.0, 0.025 * gain);
    let mut b = BiquadParam::default();
    match kind {
        FilterType::LoPass => {
            b.b0 = 1.0;
            b.b1 = 0.0;
            b.b2 = 0.0;
            b.a1 = 1.0 / q;
            b.a2 = 1.0;
        }
        FilterType::HiPass => {
            b.b0 = 0.0;
            b.b1 = 0.0;
            b.b2 = 1.0;
            b.a1 = 1.0 / q;
            b.a2 = 1.0;
        }
        FilterType::BandPass => {
            b.b0 = 0.0;
            b.b1 = 1.0 / q;
            b.b2 = 0.0;
            b.a1 = 1.0 / q;
            b.a2 = 1.0;
        }
        FilterType::Notch => {
            b.b0 = 1.0;
            b.b1 = 0.0;
            b.b2 = 1.0;
            b.a1 = 1.0 / q;
            b.a2 = 1.0;
        }
    }
    b
}

/// `-x * 2 * pi * freq / sampleRate`, evaluated left to right as the
/// original writes it.
fn angle(x: f64, freq: f64, sample_rate: f64) -> f64 {
    -x * 2.0 * PI * freq / sample_rate
}

/// `getMztTransform`, for the branches the ported types reach.
fn mzt_transform(analog: &BiquadParam, digital: &mut BiquadParam, freq: f64, sample_rate: f64) {
    if analog.b2 == 0.0 {
        if analog.b1 == 0.0 {
            digital.b0 = 1.0;
            digital.b1 = 0.0;
            digital.b2 = 0.0;
        } else {
            let za0 = analog.b0 / analog.b1;
            digital.b0 = 1.0;
            digital.b1 = -math::exp(angle(za0, freq, sample_rate));
            digital.b2 = 0.0;
        }
    } else {
        let b11 = analog.b1 * analog.b1;
        let b02 = 4.0 * analog.b0 * analog.b2;
        if b11 >= b02 {
            // Real zeros: the only zeros a highpass has, both at DC.
            let za0 = (analog.b1 + (b11 - b02).sqrt()) / (2.0 * analog.b2);
            let za1 = (analog.b1 - (b11 - b02).sqrt()) / (2.0 * analog.b2);
            let zd0 = math::exp(angle(za0, freq, sample_rate));
            let zd1 = math::exp(angle(za1, freq, sample_rate));
            digital.b0 = 1.0;
            digital.b1 = -zd0 - zd1;
            digital.b2 = zd0 * zd1;
        } else {
            // Complex zeros: a notch's, on the unit circle at its centre.
            let za_re = analog.b1 / (2.0 * analog.b2);
            let za_im = (b02 - b11).sqrt() / (2.0 * analog.b2);
            let d_rad = math::exp(angle(za_re, freq, sample_rate));
            let zd_re = d_rad * math::cos(angle(za_im, freq, sample_rate));
            let zd_im = d_rad * math::sin(angle(za_im, freq, sample_rate));
            digital.b0 = 1.0;
            digital.b1 = -2.0 * zd_re;
            digital.b2 = zd_re * zd_re + zd_im * zd_im;
        }
    }

    // Complex poles: every Q above 0.5.
    let a02 = 4.0 * analog.a2;
    let a11 = analog.a1 * analog.a1;
    let za_re = analog.a1 / (2.0 * analog.a2);
    let za_im = (a02 - a11).sqrt() / (2.0 * analog.a2);
    let d_rad = math::exp(angle(za_re, freq, sample_rate));
    let zd_re = d_rad * math::cos(angle(za_im, freq, sample_rate));
    let zd_im = d_rad * math::sin(angle(za_im, freq, sample_rate));
    digital.a1 = -2.0 * zd_re;
    digital.a2 = zd_re * zd_re + zd_im * zd_im;
}

/// `getCorrection`: the FIR that pulls the digital response back onto the
/// analog one at three frequencies.
fn correction(digital: &mut BiquadParam, analog: &BiquadParam, freq: f64, sample_rate: f64) {
    let freqs = [
        sample_rate / 44100.0,
        sample_rate * 1.0 / 6.0,
        sample_rate * 2.0 / 6.0,
    ];
    let mut h = [0.0f64; 3];
    for (h, f) in h.iter_mut().zip(freqs) {
        let phi0 = math::sin(PI * f / sample_rate);
        let phi = phi0 * phi0;
        let s = f / freq;
        let a0 = 1.0;

        let (b0, b1, b2, a1, a2) = (digital.b0, digital.b1, digital.b2, digital.a1, digital.a2);
        let hdz2 = (b0 + b1 + b2) * (b0 + b1 + b2)
            - 4.0 * (b0 * b1 + 4.0 * b0 * b2 + b1 * b2) * phi
            + 16.0 * b0 * b2 * phi * phi;
        let hdp2 = (a0 + a1 + a2) * (a0 + a1 + a2)
            - 4.0 * (a0 * a1 + 4.0 * a0 * a2 + a1 * a2) * phi
            + 16.0 * a0 * a2 * phi * phi;
        let hd = hdz2 / hdp2;

        let real_zero = analog.b0 - analog.b2 * s * s;
        let imag_zero = s * analog.b1;
        let real_pole = a0 - analog.a2 * s * s;
        let imag_pole = s * analog.a1;
        let haz2 = real_zero * real_zero + imag_zero * imag_zero;
        let hap2 = real_pole * real_pole + imag_pole * imag_pole;
        let ha = haz2 / hap2;

        *h = (ha / hd).sqrt();
    }
    let c1 = 0.5 * (h[0] - (h[0] * h[0] - 2.0 * h[1] * h[1] + 2.0 * h[2] * h[2]).sqrt());
    let c2 = (3.0 * (h[0] - c1)
        - (-3.0 * h[0] * h[0] + 12.0 * h[1] * h[1] - 6.0 * h[0] * c1 - 3.0 * c1 * c1).sqrt())
        / 6.0;
    let c0 = h[0] - c1 - c2;
    digital.c0 = c0;
    digital.c1 = c1;
    digital.c2 = c2;
}

/// `ParametricCreator::createMZTiStage`.
fn mzti_stage(freq: f64, gain: f64, q: f64, kind: FilterType, sample_rate: f64) -> BiquadParam {
    let analog = analog(q, gain, kind);
    let mut param = BiquadParam::default();
    mzt_transform(&analog, &mut param, freq, sample_rate);
    correction(&mut param, &analog, freq, sample_rate);
    param.use_compensation = true;
    param
}

/// `AudioFilter::Response::getResponsePoint`: the magnitude of one stage's
/// response at `freq`, the correction FIR's with it.
fn response_point(p: &BiquadParam, freq: f64, sample_rate: f64) -> f64 {
    let sin2w = math::sin(PI * freq / sample_rate);
    let phi = sin2w * sin2w;
    let (b0, b1, b2, a1, a2) = (p.b0, p.b1, p.b2, p.a1, p.a2);
    let nom = (b0 + b1 + b2) * (b0 + b1 + b2) - 4.0 * (b0 * b1 + 4.0 * b0 * b2 + b1 * b2) * phi
        + 16.0 * b0 * b2 * phi * phi;
    let den = (1.0 + a1 + a2) * (1.0 + a1 + a2)
        - 4.0 * (1.0 * a1 + 4.0 * 1.0 * a2 + a1 * a2) * phi
        + 16.0 * 1.0 * a2 * phi * phi;
    let mut gain = nom / den;
    if p.use_compensation {
        let (c0, c1, c2) = (p.c0, p.c1, p.c2);
        let cor = (c0 + c1 + c2) * (c0 + c1 + c2) - 4.0 * (c0 * c1 + 4.0 * c0 * c2 + c1 * c2) * phi
            + 16.0 * c0 * c2 * phi * phi;
        gain *= cor;
    }
    gain.sqrt()
}

/// One stage's response at `freq`, as a complex gain `(re, im)`: the
/// biquad and its correction FIR evaluated on the unit circle.
fn response_complex(p: &BiquadParam, freq: f64, sample_rate: f64) -> (f64, f64) {
    let w = 2.0 * PI * freq / sample_rate;
    // z^-1 and z^-2.
    let (c1, s1) = (math::cos(w), -math::sin(w));
    let (c2, s2) = (math::cos(2.0 * w), -math::sin(2.0 * w));
    let poly = |a: f64, b: f64, c: f64| (a + b * c1 + c * c2, b * s1 + c * s2);
    let mul = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| (ar * br - ai * bi, ar * bi + ai * br);
    let num = poly(p.b0, p.b1, p.b2);
    let (dr, di) = poly(1.0, p.a1, p.a2);
    let den2 = dr * dr + di * di;
    let mut h = mul(num, (dr / den2, -di / den2));
    if p.use_compensation {
        h = mul(h, poly(p.c0, p.c1, p.c2));
    }
    h
}

/// `AudioFilter::FilterParams<float>`: a biquad as state-variable filter
/// coefficients, and its correction FIR.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FilterParams {
    f: f32,
    r: f32,
    g: f32,
    h: f32,
    b: f32,
    l: f32,
    c0: f32,
    c1: f32,
    c2: f32,
    use_compensation: bool,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            f: 1.0,
            r: 2.0,
            g: 0.25,
            h: 1.0,
            b: 2.0,
            l: 1.0,
            c0: 1.0,
            c1: 0.0,
            c2: 0.0,
            use_compensation: false,
        }
    }
}

impl FilterParams {
    /// `setFromBiquadParams`. `f` and `r` are rounded to float before `a0`
    /// is worked out from them, in float, as the original's members are.
    fn from_biquad(p: &BiquadParam) -> Self {
        let f = ((1.0 + p.a1 + p.a2) / (1.0 - p.a1 + p.a2)).sqrt() as f32;
        let r =
            (-2.0 * (p.a2 - 1.0) / (p.a2 * p.a2 + 2.0 * p.a2 - p.a1 * p.a1 + 1.0).sqrt()) as f32;
        let a0 = f64::from(f * (f + r) + 1.0);
        let r2f = 0.5 / f64::from(f);
        Self {
            f,
            r,
            g: (1.0 / a0) as f32,
            h: ((p.b0 - p.b1 + p.b2) * 0.25 * a0) as f32,
            b: ((p.b0 - p.b2) * r2f * a0) as f32,
            l: ((p.b0 + p.b1 + p.b2) * (r2f * r2f) * a0) as f32,
            c0: p.c0 as f32,
            c1: p.c1 as f32,
            c2: p.c2 as f32,
            use_compensation: p.use_compensation,
        }
    }
}

/// `AudioFilter::FilterState<float>`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FilterState {
    z1: f32,
    z2: f32,
    x1: f32,
    x2: f32,
}

impl FilterState {
    /// One sample through `processFilter`: the state-variable stage, then
    /// the correction FIR. The original runs the FIR as a second pass over
    /// the block; per sample the arithmetic is the same.
    fn process(&mut self, p: &FilterParams, x0: f32) -> f32 {
        let hp = (x0 - (p.f + p.r) * self.z1 - self.z2) * p.g;
        let bp = self.z1 + p.f * hp;
        let lp = self.z2 + p.f * bp;
        self.z1 = 2.0 * bp - self.z1;
        self.z2 = 2.0 * lp - self.z2;
        let out = p.h * hp + p.b * bp + p.l * lp;
        if !p.use_compensation {
            return out;
        }
        let y = out * p.c0 + self.x1 * p.c1 + self.x2 * p.c2;
        self.x2 = self.x1;
        self.x1 = out;
        y
    }
}

/// `AudioFilter::FilterInstance<float>` of one stage over up to two
/// channels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FilterInstance {
    params: FilterParams,
    states: [FilterState; 2],
    /// Whether `setParams` has sized the states yet; the first call clears
    /// them, later ones leave them running.
    sized: bool,
}

impl FilterInstance {
    fn set_params(&mut self, biquad: &BiquadParam) {
        self.params = FilterParams::from_biquad(biquad);
        if !self.sized {
            self.sized = true;
            self.clear();
        }
    }

    fn clear(&mut self) {
        if self.sized {
            self.states = [FilterState::default(); 2];
        }
    }

    fn process(&mut self, channel: usize, x: f32) -> f32 {
        if !self.sized {
            return x;
        }
        self.states[channel].process(&self.params, x)
    }
}

/// How steep `gin::Filter` is: one stage or two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slope {
    Db12,
    Db24,
}

/// `gin::Filter`: a lowpass, highpass, bandpass or notch on up to two
/// channels, at 12 or 24 dB.
#[derive(Clone, Copy, Debug)]
pub struct Filter {
    kind: FilterType,
    slope: Slope,
    sample_rate: f64,
    stages: [FilterInstance; 2],
    /// What each stage was last designed as, for its response: the
    /// original designs them again in `getResponseMagnitude` from the same
    /// settings, which comes to the same coefficients.
    designs: [BiquadParam; 2],
}

impl Filter {
    #[must_use]
    pub fn new(kind: FilterType) -> Self {
        Self {
            kind,
            slope: Slope::Db12,
            sample_rate: 44100.0,
            stages: [FilterInstance::default(); 2],
            designs: [BiquadParam::default(); 2],
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    pub fn set_slope(&mut self, slope: Slope) {
        self.slope = slope;
    }

    /// `setType`: what the next `setParams` designs.
    pub fn set_type(&mut self, kind: FilterType) {
        self.kind = kind;
    }

    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.clear();
        }
    }

    /// `setParams (freq, q, g)`, `g` a linear gain.
    pub fn set_params(&mut self, freq: f32, q: f32, g: f32) {
        let gain = f64::from(gain_to_decibels(g));
        let first = mzti_stage(
            f64::from(freq),
            gain,
            f64::from(q),
            self.kind,
            self.sample_rate,
        );
        self.stages[0].set_params(&first);
        let second = mzti_stage(
            f64::from(freq),
            gain,
            f64::from(BUTTERWORTH_Q),
            self.kind,
            self.sample_rate,
        );
        self.stages[1].set_params(&second);
        self.designs = [first, second];
    }

    /// The filter's response at `freq` as a complex gain `(re, im)`: one
    /// stage at 12 dB, both at 24.
    #[must_use]
    pub fn response(&self, freq: f32) -> (f64, f64) {
        let stages = match self.slope {
            Slope::Db12 => 1,
            Slope::Db24 => 2,
        };
        self.designs[..stages].iter().fold((1.0, 0.0), |(ar, ai), design| {
            let (br, bi) = response_complex(design, f64::from(freq), self.sample_rate);
            (ar * br - ai * bi, ar * bi + ai * br)
        })
    }

    /// `getResponseMagnitude`: the filter's gain at `freq`, as a factor —
    /// one stage at 12 dB, both at 24. A filter never given its settings
    /// passes everything.
    #[must_use]
    pub fn response_magnitude(&self, freq: f32) -> f32 {
        let stages = match self.slope {
            Slope::Db12 => 1,
            Slope::Db24 => 2,
        };
        self.designs[..stages]
            .iter()
            .map(|design| response_point(design, f64::from(freq), self.sample_rate))
            .product::<f64>() as f32
    }

    /// One sample of `channel` through the filter.
    pub fn process(&mut self, channel: usize, x: f32) -> f32 {
        let y = self.stages[0].process(channel, x);
        match self.slope {
            Slope::Db12 => y,
            Slope::Db24 => self.stages[1].process(channel, y),
        }
    }
}
