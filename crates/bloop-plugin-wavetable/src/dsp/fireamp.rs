//! `FireAmp`, the Airwindows amp simulator: twelve stages of highpass,
//! overdrive and two-sample averaging, a speaker-cone comb, a waveshaper,
//! and an 84-tap nonlinear cabinet run at (about) 44.1 kHz.
//!
//! Ported from the copy SocaLabs Wavetable carries
//! (`plugin/Source/FX/FireAmp.cpp`, `FireAmpProc.cpp`), single-precision
//! path only: `processReplacing`, which computes in `double` and stores
//! `float`. `FireAmp` is by Chris Johnson (Airwindows), MIT licence.
//!
//! The arithmetic is the original's, operation for operation, so that the
//! output is bit-identical to the C++ (with the same maths library). The
//! quirks that carries over:
//!
//! - The ultrasonic biquads take `tan(juce::MathConstants<float>::pi * f)`:
//!   a single-precision pi, promoted to double.
//! - The original's dither is commented out, so `fpd` is drawn once per
//!   channel, by the constructor and `reset`, and never changes. It still
//!   sets how much of the previous sample is blended in (`randy`).
//! - `reset` puts the parameters back to their defaults, as the original's
//!   does.
//! - The comb's taps are integer truncations of the sample rate
//!   (`diagonal`, `side`, `down`), and the cabinet runs once every
//!   `floor(rate / 44100)` samples (clamped to 1..=4), interpolating
//!   between.

use bloop_plugin_socalabs::shared::math;

/// The cabinet's taps on `b[1..=84]`: each adds `b * (c + d * |b|)`.
///
/// The original writes each as `s ±= b * (x ± y * fabs(b))`; the signs are
/// folded into `c` and `d` here. That is exact: IEEE subtraction is
/// addition of the negation, and negation commutes with round-to-nearest,
/// so `s - b * (x - p)` and `s + b * (-x + p)` round identically. The
/// digits are the original's.
const TAPS: [(f64, f64); 84] = [
    (1.31698250313308396, -0.08140616497621633),
    (1.47229016949915326, -0.27680278993637253),
    (1.30410109086044956, -0.35629113432046489),
    (0.81766210474551260, -0.26808782337659753),
    (0.19868872545506663, -0.11105517193919669),
    (-0.39115909132567039, 0.12630622002682679),
    (-0.76881891559343574, 0.40879849500403143),
    (-0.87146861782680340, 0.59529560488000599),
    (-0.79504575932563670, 0.60877047551611796),
    (-0.61653017622406314, 0.47662851438557335),
    (-0.40718195794382067, 0.24955839378539713),
    (-0.31794900040616203, 0.04169792259600613),
    (-0.41075032540217843, -0.00368483996076280),
    (-0.56901352922170667, 0.11027360805893105),
    (-0.62443222391889264, 0.22198075154245228),
    (-0.53462856723129204, 0.22933544545324852),
    (-0.34441703361995046, 0.12956809502269492),
    (-0.13947052337867882, -0.00339775055962799),
    (0.03771252648928484, -0.10863931549251718),
    (0.18280210770271693, -0.17413646599296417),
    (0.24621986701761467, -0.14547053270435095),
    (0.22347075142737360, -0.02493869490104031),
    (0.14346348482123716, 0.11284054747963246),
    (0.00834364862916028, 0.24284684053733926),
    (-0.11559740296078347, 0.32623054435304538),
    (-0.18067604561283060, 0.32311481551122478),
    (-0.22927997789035612, 0.26991539052832925),
    (-0.28487666578669446, 0.22437227250279349),
    (-0.31992973037153838, 0.15289876100963865),
    (-0.35174606303520733, 0.05656293023086628),
    (-0.36894898011375254, -0.04333925421463558),
    (-0.32567576055307507, -0.14594589410921388),
    (-0.27440135050585784, -0.15529667398122521),
    (-0.21998973785078091, -0.05083553737157104),
    (-0.10323624876862457, 0.04651829594199963),
    (0.02091603687851074, 0.12000046818439322),
    (0.11344930914138468, 0.17697142512225839),
    (0.22766779627643968, 0.13645102964003858),
    (0.38378309953638229, -0.01997653307333791),
    (0.52789400804568076, -0.21409137428422448),
    (0.55444630296938280, -0.32331980931576626),
    (0.42333237669264601, -0.26855847463044280),
    (0.21942831522035078, -0.12051365248820624),
    (-0.00584169427830633, 0.03706970171280329),
    (-0.24279799124660351, 0.17296440491477982),
    (-0.40173760787507085, 0.21717989835163351),
    (-0.43930035724188155, 0.16425928481378199),
    (-0.41067765934041811, 0.10390115786636855),
    (-0.34409235547165967, 0.07268159377411920),
    (-0.26542883122568151, 0.05483457497365785),
    (-0.22024754776138800, 0.06484897950087598),
    (-0.20394367993632415, 0.08746309731952180),
    (-0.17565242431124092, 0.07611309538078760),
    (-0.10116623231246825, 0.00642818706295112),
    (-0.00782648272053632, -0.08004141267685004),
    (0.05059046006747323, -0.12436676387548490),
    (0.06241531553254467, -0.11530779547021434),
    (0.04952694587101836, -0.08340945324333944),
    (0.00843873294401687, -0.03279659052562903),
    (-0.05161338949440241, 0.03428181149163798),
    (-0.08165520146902012, 0.08196746092283110),
    (-0.06639532849935320, 0.09797462781896329),
    (-0.02953430910661621, 0.09175612938515763),
    (0.00741058547442938, 0.05442091048731967),
    (0.01832866125391727, 0.00306243693643687),
    (0.00526964230373573, -0.04364102661136410),
    (-0.00300984373848200, -0.09742737841278880),
    (-0.00413616769576694, -0.14380661694523073),
    (-0.00588769034931419, -0.16012843578892538),
    (-0.00688588239450581, -0.14074464279305798),
    (-0.02277307992926315, -0.07914752191801366),
    (-0.04627166091180877, 0.00192787268067208),
    (-0.05562045897455786, 0.05932868727665747),
    (-0.05134243784922165, 0.08245334798868090),
    (-0.04719409472239919, 0.07498680629253825),
    (-0.05889738914266415, 0.06116127018043697),
    (-0.09428363535111127, 0.06535868867863834),
    (-0.15181756953225126, 0.08982979655234427),
    (-0.20878969456036670, 0.10761070891499538),
    (-0.22647885581813790, 0.08462542510349125),
    (-0.19723482443646323, 0.02665160920736287),
    (-0.16441643451155163, -0.02314691954338197),
    (-0.15201914054931515, -0.04424903493886839),
    (-0.15454370641307855, -0.04223203797913008),
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
    bassfill: f64,
    outputlevel: f64,
    wet: f64,
    eq: f64,
    tone_eq: f64,
    bleed: f64,
    bassfactor: f64,
    beq: f64,
    diagonal: usize,
    side: usize,
    down: usize,
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

/// Waveshapes `x * gain` through a sine clamped at its peak, with the sign
/// of `x` (zero and NaN take the negative branch, as in the original).
fn sine_clip(x: f64, gain: f64) -> f64 {
    let mut bridgerectifier = (x * gain).abs();
    if bridgerectifier > HALF_PI {
        bridgerectifier = HALF_PI;
    }
    bridgerectifier = math::sin(bridgerectifier);
    if x > 0.0 {
        bridgerectifier
    } else {
        -bridgerectifier
    }
}

/// One channel's state. The original keeps these as `…L`/`…R` pairs; the
/// two channels never read each other's.
#[derive(Clone, Debug)]
struct Channel {
    fix: [[f64; 2]; 6],
    iir_sample: [f64; 12],
    smooth: [f64; 12],
    iir_lowpass: f64,
    iir_spk_a: f64,
    iir_spk_b: f64,
    iir_sub: f64,
    store_sample: f64,
    odd: [f64; 257],
    even: [f64; 257],
    b: [f64; 90],
    last_cab_sample: f64,
    smooth_cab_a: f64,
    smooth_cab_b: f64,
    last_ref: [f64; 10],
    fpd: u32,
}

impl Channel {
    fn new(fpd: u32) -> Self {
        Self {
            fix: [[0.0; 2]; 6],
            iir_sample: [0.0; 12],
            smooth: [0.0; 12],
            iir_lowpass: 0.0,
            iir_spk_a: 0.0,
            iir_spk_b: 0.0,
            iir_sub: 0.0,
            store_sample: 0.0,
            odd: [0.0; 257],
            even: [0.0; 257],
            b: [0.0; 90],
            last_cab_sample: 0.0,
            smooth_cab_a: 0.0,
            smooth_cab_b: 0.0,
            last_ref: [0.0; 10],
            fpd,
        }
    }

    /// The amp, from the input up to the dry/wet mix. `count` is the comb's
    /// write position, already brought into 0..=128.
    fn amp(&mut self, input: f64, k: &Block, flip: bool, count: usize) -> f64 {
        let mut x = k.fix[0].run(&mut self.fix[0], input).clamp(-1.0, 1.0);
        let mut basscut = 0.98;
        let mut inputlevel = k.bassfill;
        for stage in 0..12 {
            if stage > 0 && stage % 2 == 0 {
                let f = stage / 2;
                x = k.fix[f].run(&mut self.fix[f], x);
            }
            x *= inputlevel;
            inputlevel = ((inputlevel * 7.0) + 1.0) / 8.0;
            self.iir_sample[stage] = (self.iir_sample[stage] * (1.0 - k.eq)) + (x * k.eq);
            basscut *= k.bassfactor;
            x -= self.iir_sample[stage] * basscut;
            x -= x * (x.abs() * 0.654) * (x.abs() * 0.654);
            let bridgerectifier = self.smooth[stage] + x;
            self.smooth[stage] = x;
            x = bridgerectifier;
        }

        self.iir_lowpass = (self.iir_lowpass * (1.0 - k.tone_eq)) + (x * k.tone_eq);
        x = self.iir_lowpass;
        self.iir_spk_a = (self.iir_spk_a * (1.0 - k.beq)) + (x * k.beq);

        let comb = if flip { &mut self.odd } else { &mut self.even };
        comb[count + 128] = self.iir_spk_a;
        comb[count] = self.iir_spk_a;
        let result_b = comb[count + k.down] + comb[count + k.side] + comb[count + k.diagonal];
        self.iir_spk_b = (self.iir_spk_b * (1.0 - k.beq)) + (result_b * k.beq);
        x += self.iir_spk_b * k.bleed;

        x = sine_clip(x, k.outputlevel);

        self.iir_sub = (self.iir_sub * (1.0 - k.beq)) + (x * k.beq);
        x += self.iir_sub * k.bassfill * k.outputlevel;

        let randy = unit(self.fpd) * 0.053;
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

        let randy = unit(self.fpd) * 0.057;
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
    fn average(&mut self, mut x: f64, cycle_end: i32) -> f64 {
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

/// `FireAmp`: Gain (A), Tone (B), Output (C), Dry/Wet (D), each 0..1.
#[derive(Clone, Debug)]
pub struct FireAmp {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    left: Channel,
    right: Channel,
    flip: bool,
    count: i32,
    cycle: i32,
}

impl FireAmp {
    /// A `FireAmp` at its defaults (A 0.5, B 0.5, C 0.8, D 1.0). Draws its
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
            left: Channel::new(fpd_l),
            right: Channel::new(fpd_r),
            flip: false,
            count: 0,
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
        let bassfill = f64::from(self.a);
        let outputlevel = f64::from(self.c);
        let tone = f64::from(self.b);
        let basstrim = bassfill / 16.0;
        let bleed = outputlevel / 16.0;
        // `(int)` truncation; the saturating cast agrees with it at any
        // positive rate, and stays an index at a nonsensical one.
        let diagonal = ((0.000861678 * sample_rate) as usize).min(127);
        let side = (diagonal as f64 / std::f64::consts::SQRT_2) as usize;
        let down = (side + diagonal) / 2;

        let cutoff = ((15000.0 + (tone * 10000.0)) / sample_rate).clamp(0.001, 0.49);

        Block {
            bassfill,
            outputlevel,
            wet: f64::from(self.d),
            eq: (basstrim / sample_rate) * 22050.0,
            tone_eq: (tone / sample_rate) * 22050.0,
            bleed,
            bassfactor: 1.0 - (basstrim * basstrim),
            beq: (bleed / sample_rate) * 22050.0,
            diagonal,
            side,
            down,
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

            if self.count < 0 || self.count > 128 {
                self.count = 128;
            }
            let count = self.count as usize;
            let mut out_l = self.left.amp(dry_l, &k, self.flip, count);
            let mut out_r = self.right.amp(dry_r, &k, self.flip, count);
            self.count -= 1;
            self.flip = !self.flip;

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
            out_l = self.left.average(self.left.last_ref[at], cycle_end);
            out_r = self.right.average(self.right.last_ref[at], cycle_end);

            *l = out_l as f32;
            *r = out_r as f32;
        }
    }
}
