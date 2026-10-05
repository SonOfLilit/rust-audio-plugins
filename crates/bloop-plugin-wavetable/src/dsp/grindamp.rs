//! `GrindAmp`, the Airwindows high-gain amp simulator: two highpass and
//! overdrive stages, seven stages that saturate a separately-tracked bass
//! signal, three-sample averaging throughout, and an 83-tap nonlinear
//! cabinet run at (about) 44.1 kHz.
//!
//! Ported from the copy SocaLabs Wavetable carries
//! (`plugin/Source/FX/GrindAmp.cpp`, `GrindAmpProc.cpp`), single-precision
//! path only: `processReplacing`, which computes in `double` and stores
//! `float`. `GrindAmp` is by Chris Johnson (Airwindows), MIT licence.
//!
//! The arithmetic is the original's, operation for operation, so that the
//! output is bit-identical to the C++ (with the same maths library). The
//! quirks that carries over:
//!
//! - The ultrasonic biquads take `tan(juce::MathConstants<float>::pi * f)`:
//!   a single-precision pi, promoted to double.
//! - In the bass stages, the averaging's weights come from the *bass*
//!   signal's sine, not from the stage's own input.
//! - The cabinet's dry/wet blend uses 0.044 on the left and 0.04 on the
//!   right.
//! - The original's dither is commented out, so `fpd` is drawn once per
//!   channel, by the constructor and `reset`, and never changes. It still
//!   sets how much of the previous sample is blended in (`randy`).
//! - `reset` puts the parameters back to their defaults, as the original's
//!   does.
//! - The cabinet runs once every `floor(rate / 44100)` samples (clamped to
//!   1..=4), interpolating between.

use bloop_plugin_socalabs::shared::math;

/// The cabinet's taps on `b[1..=83]`: each adds `b * (c + d * |b|)`.
///
/// The original writes each as `s ±= b * (x ± y * fabs(b))`; the signs are
/// folded into `c` and `d` here. That is exact: IEEE subtraction is
/// addition of the negation, and negation commutes with round-to-nearest,
/// so `s - b * (x - p)` and `s + b * (-x + p)` round identically. The
/// digits are the original's.
const TAPS: [(f64, f64); 83] = [
    (1.29550481610475132, 0.19713872057074355),
    (1.42302569895462616, 0.30599505521284787),
    (1.28728195804197565, 0.23168333460446133),
    (0.88553784290822690, 0.14263256172918892),
    (0.37129054918432319, 0.00150040944205920),
    (-0.12150959412556320, -0.32776273620569107),
    (-0.44900065463203775, -0.74101214925298819),
    (-0.54058781908186482, -1.07821707459008387),
    (-0.49361966401791391, -1.23540109014850508),
    (-0.39819495093078133, -1.11247213730917749),
    (-0.31379279985435521, -0.80330360359638298),
    (-0.30744359242808555, -0.42132528876858205),
    (-0.33943170284673974, -0.09183418349389982),
    (-0.33838775119286391, 0.06453051658561271),
    (-0.30682305697961665, 0.09549380253249232),
    (-0.23408741339295336, 0.08083404732361277),
    (-0.10411746814025019, -0.00253651281245780),
    (0.00133623776084696, -0.04447267870865820),
    (0.02461903992114161, 0.07530671732655550),
    (0.02086715842475373, 0.22795860236804899),
    (0.02761433637100917, 0.26108320417844094),
    (0.04475285369162533, 0.19160705011061663),
    (0.09447338372862381, 0.03681550508743799),
    (0.13445890343722280, -0.13713036462146147),
    (0.13872868945088121, -0.22401242373298191),
    (0.14915650097434549, -0.26718804981526367),
    (0.12766643217091783, -0.27745664795660430),
    (0.03675849788393101, -0.18338278173550679),
    (-0.06307306864232835, -0.06089480869040766),
    (-0.14947389348962944, -0.04642103054798480),
    (-0.25235266566401526, -0.08423062596460507),
    (-0.33496344048679683, -0.09808328256677995),
    (-0.36590030482175445, -0.10622650888958179),
    (-0.35015197011464372, -0.08982043516016047),
    (-0.26808437585665090, -0.00735561860229533),
    (-0.11624318543291220, 0.07142484314510467),
    (0.05617084165377551, 0.11785854050350089),
    (0.20540028692589385, 0.20479174663329586),
    (0.30455415003043818, 0.29074864580096849),
    (0.33810750937829476, 0.29182307921316802),
    (0.31936133365277430, 0.26535537727394987),
    (0.27388548321981876, 0.19735049990538350),
    (0.21454597517994098, 0.06415909270247236),
    (0.15001045817707717, -0.03831118543404573),
    (0.07283437284653138, -0.09281952429543777),
    (-0.03917872184241358, -0.14306291461398810),
    (-0.16695932032148642, -0.19138995946950504),
    (-0.27055854466909462, -0.22531296466343192),
    (-0.33256357307578271, -0.23305840475692102),
    (-0.33459770116834442, -0.24091822618917569),
    (-0.27156687236338090, -0.24062938573512443),
    (-0.17197093288412094, -0.19083085091993421),
    (-0.06738628195910543, -0.10268609751019808),
    (0.00222429218204290, 0.01439664435720548),
    (0.01346992803494091, 0.15947137113534526),
    (-0.02038911881377448, 0.26763170752416160),
    (-0.08233579178189687, 0.29415931086406055),
    (-0.15447855089824883, 0.26489186990840807),
    (-0.20518281113362655, 0.16135382257522859),
    (-0.22244686050232007, -0.00847180390247432),
    (-0.21849243134998034, -0.14460595245753741),
    (-0.20256105734574054, -0.18932793221831667),
    (-0.18604070054295399, -0.17250665610927965),
    (-0.17222844322058231, -0.12992472027850357),
    (-0.14447856616566443, -0.09089219002147308),
    (-0.10385520794251019, -0.08600465834570559),
    (-0.07124435678265063, -0.09071532210549428),
    (-0.05216857461197572, -0.06794061706070262),
    (-0.05235381920184123, -0.02818101717909346),
    (-0.07569701245553526, 0.00634228544764946),
    (-0.10320125382718826, 0.02751486906644141),
    (-0.12122120969079088, 0.05434007312178933),
    (-0.13438969117200902, 0.09135218559713874),
    (-0.13534390437529981, 0.10437672041458675),
    (-0.11424128854188388, 0.08693450726462598),
    (-0.08166894518596159, 0.06949989431475120),
    (-0.04293976378555305, 0.05718625137421843),
    (0.00933076320644409, 0.01728285211520138),
    (0.06450430362918153, -0.02492994833691022),
    (0.10187400687649277, -0.03578455940532403),
    (0.11039763294094571, -0.03995523517573508),
    (0.08557960776024547, -0.03482514309492527),
    (0.02730881850805332, -0.00514750108411127),
];

/// The Q of each of the six ultrasonic lowpass biquads, in order.
#[expect(clippy::approx_constant, reason = "the original's digits")]
const RESO: [f64; 6] = [
    4.46570214, 1.51387132, 0.93979296, 0.70710678, 0.52972649, 0.50316379,
];

/// Airwindows' own half pi, which is not quite `FRAC_PI_2`.
#[expect(clippy::approx_constant, reason = "the original's digits")]
const HALF_PI: f64 = 1.57079633;

/// One fixed-frequency lowpass biquad's coefficients.
#[derive(Clone, Copy, Debug)]
struct Biquad {
    a0: f64,
    a1: f64,
    a2: f64,
    b1: f64,
    b2: f64,
}

impl Biquad {
    fn lowpass(freq: f64, reso: f64) -> Self {
        let k = math::tan(f64::from(std::f32::consts::PI) * freq);
        let norm = 1.0 / (1.0 + k / reso + k * k);
        let a0 = k * k * norm;
        Self {
            a0,
            a1: 2.0 * a0,
            a2: a0,
            b1: 2.0 * (k * k - 1.0) * norm,
            b2: (1.0 - k / reso + k * k) * norm,
        }
    }

    /// Transposed direct form II, `s` being this channel's two states.
    fn run(&self, s: &mut [f64; 2], x: f64) -> f64 {
        let out = (x * self.a0) + s[0];
        s[0] = (x * self.a1) - (out * self.b1) + s[1];
        s[1] = (x * self.a2) - (out * self.b2);
        out
    }
}

/// What one call to `process` derives from the parameters and the rate.
struct Block {
    inputlevel: f64,
    eq: f64,
    beq: f64,
    tone_eq: f64,
    outputlevel: f64,
    wet: f64,
    bassdrive: f64,
    fix: [Biquad; 6],
}

/// Draws a seed the way the original does:
/// `fpd = 1.0; while (fpd < 16386) fpd = rand() * UINT32_MAX;` — an `int`
/// times an `unsigned`, so the product wraps modulo 2^32.
fn seed(rand: &mut dyn FnMut() -> i32) -> u32 {
    let mut fpd: u32 = 1;
    while fpd < 16386 {
        fpd = (rand() as u32).wrapping_mul(u32::MAX);
    }
    fpd
}

/// `double(fpd) / UINT32_MAX`.
fn unit(fpd: u32) -> f64 {
    f64::from(fpd) / f64::from(u32::MAX)
}

/// Waveshapes `x * gain` through a sine clamped at its peak, returning the
/// shaped magnitude and the result with the sign of `x` (zero and NaN take
/// the negative branch, as in the original).
fn sine_clip(x: f64, gain: f64) -> (f64, f64) {
    let mut bridgerectifier = (x * gain).abs();
    if bridgerectifier > HALF_PI {
        bridgerectifier = HALF_PI;
    }
    bridgerectifier = math::sin(bridgerectifier);
    let signed = if x > 0.0 {
        bridgerectifier
    } else {
        -bridgerectifier
    };
    (bridgerectifier, signed)
}

/// One channel's state. The original keeps these as `…L`/`…R` pairs; the
/// two channels never read each other's.
#[derive(Clone, Debug)]
struct Channel {
    fix: [[f64; 2]; 6],
    smooth: [f64; 11],
    second: [f64; 11],
    third: [f64; 11],
    iir_sample: [f64; 9],
    store_sample: f64,
    b: [f64; 90],
    last_cab_sample: f64,
    smooth_cab_a: f64,
    smooth_cab_b: f64,
    last_ref: [f64; 10],
    fpd: u32,
    /// How much of the previous cabinet output the cabinet's blend takes,
    /// per unit of `fpd`: 0.044 left, 0.04 right.
    cab_randy: f64,
}

impl Channel {
    fn new(fpd: u32, cab_randy: f64) -> Self {
        Self {
            fix: [[0.0; 2]; 6],
            smooth: [0.0; 11],
            second: [0.0; 11],
            third: [0.0; 11],
            iir_sample: [0.0; 9],
            store_sample: 0.0,
            b: [0.0; 90],
            last_cab_sample: 0.0,
            smooth_cab_a: 0.0,
            smooth_cab_b: 0.0,
            last_ref: [0.0; 10],
            fpd,
            cab_randy,
        }
    }

    fn biquad(&mut self, k: &Block, i: usize, x: f64) -> f64 {
        k.fix[i].run(&mut self.fix[i], x)
    }

    /// Three-sample averaging lowpass, weighted by `bridgerectifier`.
    fn average(&mut self, stage: usize, x: f64, bridgerectifier: f64) -> f64 {
        let inverse = (bridgerectifier + 1.0) / 2.0;
        let out = self.smooth[stage]
            + (self.second[stage] * inverse)
            + (self.third[stage] * bridgerectifier)
            + x;
        self.third[stage] = self.second[stage];
        self.second[stage] = self.smooth[stage];
        self.smooth[stage] = x;
        out
    }

    /// Gain, highpass (by `basscut`), clamp and average: stages A and B.
    fn highpass(&mut self, stage: usize, mut x: f64, k: &Block, basscut: f64) -> f64 {
        x *= k.inputlevel;
        self.iir_sample[stage] = (self.iir_sample[stage] * (1.0 - k.eq)) + (x * k.eq);
        x -= self.iir_sample[stage] * basscut;
        x = x.clamp(-1.0, 1.0);
        self.average(stage, x, x.abs())
    }

    /// Lowpasses and saturates `basscatch`, then clamps and averages `x`
    /// weighted by the bass's sine: stages C to I.
    fn bass(&mut self, stage: usize, x: f64, basscatch: &mut f64, k: &Block) -> f64 {
        self.iir_sample[stage] = (self.iir_sample[stage] * (1.0 - k.beq)) + (*basscatch * k.beq);
        let (bridgerectifier, shaped) = sine_clip(self.iir_sample[stage] * k.bassdrive, 1.0);
        *basscatch = shaped;
        self.average(stage, x.clamp(-1.0, 1.0), bridgerectifier)
    }

    /// The amp, from the input up to the dry/wet mix.
    fn amp(&mut self, input: f64, k: &Block) -> f64 {
        let mut x = self.biquad(k, 0, input);
        x = self.highpass(0, x, k, 0.92);
        let mut basscatch = x;
        x = self.biquad(k, 1, x);
        x = self.highpass(1, x, k, 0.79);
        x = self.bass(2, x, &mut basscatch, k);
        x = self.biquad(k, 2, x);
        x = self.bass(3, x, &mut basscatch, k);
        x = self.biquad(k, 3, x);
        x = self.bass(4, x, &mut basscatch, k);
        x = self.bass(5, x, &mut basscatch, k);
        x = self.biquad(k, 4, x);
        x = self.bass(6, x, &mut basscatch, k);
        x = self.bass(7, x, &mut basscatch, k);
        x = self.biquad(k, 5, x);
        x = self.bass(8, x, &mut basscatch, k);
        x = self.average(9, x, x.abs());
        x = self.average(10, x, x.abs());

        basscatch /= 2.0;
        x = (x * k.tone_eq) + basscatch;
        x = sine_clip(x, k.outputlevel).1;
        x += basscatch;
        x /= 1.0 + k.tone_eq;

        let randy = unit(self.fpd) * 0.061;
        x = ((x * (1.0 - randy)) + (self.store_sample * randy)) * k.outputlevel;
        self.store_sample = x;
        x
    }

    /// The cabinet, mixed with `dry` and scaled; runs once per cycle.
    fn cab(&mut self, input: f64, dry: f64, k: &Block) -> f64 {
        let mut x = (input + self.smooth_cab_a) / 3.0;
        self.smooth_cab_a = input;
        self.b.copy_within(0..TAPS.len(), 1);
        self.b[0] = x;
        for (&b, &(c, d)) in self.b[1..].iter().zip(&TAPS) {
            x += b * (c + (d * b.abs()));
        }
        let temp = (x + self.smooth_cab_b) / 3.0;
        self.smooth_cab_b = x;
        x = temp / 4.0;

        let randy = unit(self.fpd) * self.cab_randy;
        let out = ((((x * (1.0 - randy)) + (self.last_cab_sample * randy)) * k.wet)
            + (dry * (1.0 - k.wet)))
            * k.outputlevel;
        self.last_cab_sample = x;
        out
    }

    /// Lays out the references the samples up to the next cabinet run
    /// read: a straight line from the previous run's output to this one's.
    fn interpolate(&mut self, x: f64, cycle_end: i32) {
        let r = &mut self.last_ref;
        match cycle_end {
            4 => {
                r[0] = r[4];
                r[2] = (r[0] + x) / 2.0;
                r[1] = (r[0] + r[2]) / 2.0;
                r[3] = (r[2] + x) / 2.0;
                r[4] = x;
            }
            3 => {
                r[0] = r[3];
                r[2] = (r[0] + r[0] + x) / 3.0;
                r[1] = (r[0] + x + x) / 3.0;
                r[3] = x;
            }
            2 => {
                r[0] = r[2];
                r[1] = (r[0] + x) / 2.0;
                r[2] = x;
            }
            _ => r[0] = x,
        }
    }

    /// The multi-pole average over the undersampled output: one
    /// two-sample average per step of `cycle_end` above 1, on `last_ref[7]`,
    /// `[6]` and `[5]` in that order, as the original's falling-through
    /// `switch` does.
    fn undersample_average(&mut self, mut x: f64, cycle_end: i32) -> f64 {
        let r = &mut self.last_ref;
        for (pole, from) in [(7, 4), (6, 3), (5, 2)] {
            if cycle_end >= from {
                r[8] = x;
                x = (x + r[pole]) * 0.5;
                r[pole] = r[8];
            }
        }
        x
    }
}

/// `GrindAmp`: Gain (A), Tone (B), Output (C), Dry/Wet (D), each 0..1.
#[derive(Clone, Debug)]
pub struct GrindAmp {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    left: Channel,
    right: Channel,
    cycle: i32,
}

impl GrindAmp {
    /// A `GrindAmp` at its defaults (A 0.5, B 0.5, C 0.8, D 1.0). Draws its
    /// two seeds from `rand`, left first, as the original's constructor
    /// calls `rand()`.
    pub fn new(rand: &mut dyn FnMut() -> i32) -> Self {
        let fpd_l = seed(rand);
        let fpd_r = seed(rand);
        Self {
            a: 0.5,
            b: 0.5,
            c: 0.8,
            d: 1.0,
            left: Channel::new(fpd_l, 0.044),
            right: Channel::new(fpd_r, 0.04),
            cycle: 0,
        }
    }

    /// Clears every state and puts the parameters back to their defaults,
    /// drawing new seeds from `rand` as `new` does.
    pub fn reset(&mut self, rand: &mut dyn FnMut() -> i32) {
        *self = Self::new(rand);
    }

    /// Sets parameter `index` (0..4 = A..D). The original throws on any
    /// other index; here it is ignored.
    pub fn set_parameter(&mut self, index: usize, value: f32) {
        match index {
            0 => self.a = value,
            1 => self.b = value,
            2 => self.c = value,
            3 => self.d = value,
            _ => {}
        }
    }

    fn block(&self, sample_rate: f64) -> Block {
        let gain = f64::from(self.a);
        let tone = f64::from(self.b);
        let mut trim_eq = 1.1 - tone;
        let tone_eq = trim_eq / 1.2;
        trim_eq /= 50.0;
        trim_eq += 0.165;

        let cutoff = ((18000.0 + (tone * 1000.0)) / sample_rate).clamp(0.001, 0.49);

        Block {
            // `pow(A, 2)`: the square of a float's 24-bit significand fits
            // a double's 53, so the product is exact, as is `pow`'s.
            inputlevel: gain * gain,
            eq: ((trim_eq - (tone_eq / 6.1)) / sample_rate) * 22050.0,
            beq: ((trim_eq + (tone_eq / 2.1)) / sample_rate) * 22050.0,
            tone_eq,
            outputlevel: f64::from(self.c),
            wet: f64::from(self.d),
            bassdrive: HALF_PI * (2.5 - tone_eq),
            fix: RESO.map(|reso| Biquad::lowpass(cutoff, reso)),
        }
    }

    /// Processes `left` and `right` in place at `sample_rate`, over as many
    /// frames as the shorter of the two holds.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32], sample_rate: f64) {
        let overallscale = (1.0 / 44100.0) * sample_rate;
        let cycle_end = (overallscale.floor() as i32).clamp(1, 4);
        if self.cycle > cycle_end - 1 {
            self.cycle = cycle_end - 1;
        }
        let k = self.block(sample_rate);

        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let dry_l = f64::from(*l);
            let dry_r = f64::from(*r);

            let mut out_l = self.left.amp(dry_l, &k);
            let mut out_r = self.right.amp(dry_r, &k);

            if k.wet != 1.0 {
                out_l = (out_l * k.wet) + (dry_l * (1.0 - k.wet));
                out_r = (out_r * k.wet) + (dry_r * (1.0 - k.wet));
            }

            self.cycle += 1;
            if self.cycle == cycle_end {
                out_l = self.left.cab(out_l, dry_l, &k);
                out_r = self.right.cab(out_r, dry_r, &k);
                self.left.interpolate(out_l, cycle_end);
                self.right.interpolate(out_r, cycle_end);
                self.cycle = 0;
            }
            let at = self.cycle as usize;
            out_l = self
                .left
                .undersample_average(self.left.last_ref[at], cycle_end);
            out_r = self
                .right
                .undersample_average(self.right.last_ref[at], cycle_end);

            *l = out_l as f32;
            *r = out_r as f32;
        }
    }
}
