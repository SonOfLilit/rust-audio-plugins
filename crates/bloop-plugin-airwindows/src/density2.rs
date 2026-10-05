//! Density2: Density's sine saturation (or, below its zero, a starved
//! "1 - cos" curve), run twice — once on the input and once on a
//! half-sample-delayed interpolation of it — and their differences from
//! the dry signal blended back in through a pair of all-pass-ish filters.

use crate::math::{cos, dither, guard, pow, sin, SEEDS};

/// One channel's state.
#[derive(Clone, Copy, Debug, Default)]
struct Channel {
    last3: f64,
    last2: f64,
    last1: f64,
    ata_a: f64,
    ata_b: f64,
    ata_c: f64,
    last_diff: f64,
    iir_a: f64,
    iir_b: f64,
}

/// The per-block values every sample reads.
struct Block {
    density: f64,
    out: f64,
    iir_amount: f64,
    output: f64,
    wet: f64,
}

/// Density's curve, applied `density` times over (the whole part of it
/// as plain sines, the rest as a blend by `out`).
fn densify(mut x: f64, k: &Block) -> f64 {
    let mut count = k.density;
    while count > 1.0 {
        let mut b = x.abs() * 1.57079633;
        if b > 1.57079633 {
            b = 1.57079633;
        }
        b = sin(b);
        x = if x > 0.0 { b } else { -b };
        count -= 1.0;
    }
    let mut b = x.abs() * 1.57079633;
    if b > 1.57079633 {
        b = 1.57079633;
    }
    b = if k.density > 0.0 {
        sin(b)
    } else {
        1.0 - cos(b)
    };
    if x > 0.0 {
        (x * (1.0 - k.out)) + (b * k.out)
    } else {
        (x * (1.0 - k.out)) - (b * k.out)
    }
}

impl Channel {
    fn run(&mut self, input: f64, k: &Block) -> f64 {
        let dry = input;
        let mut halfway =
            (input + self.last1 + ((-self.last2 + self.last3) * 0.0414213562373095048801688)) / 2.0;
        let half_dry = halfway;
        self.last3 = self.last2;
        self.last2 = self.last1;
        self.last1 = input;

        self.iir_b = (self.iir_b * (1.0 - k.iir_amount)) + (halfway * k.iir_amount);
        halfway -= self.iir_b;
        halfway = densify(halfway, k);

        self.ata_c = halfway - half_dry;
        self.ata_a *= 0.915965594177219015;
        self.ata_b *= 0.915965594177219015;
        self.ata_b += self.ata_c;
        self.ata_a -= self.ata_c;
        self.ata_c = self.ata_b;
        let half_diff = self.ata_c * 0.915965594177219015;

        let mut x = input;
        self.iir_a = (self.iir_a * (1.0 - k.iir_amount)) + (x * k.iir_amount);
        x -= self.iir_a;
        x = densify(x, k);

        self.ata_c = x - dry;
        self.ata_a *= 0.915965594177219015;
        self.ata_b *= 0.915965594177219015;
        self.ata_a += self.ata_c;
        self.ata_b -= self.ata_c;
        self.ata_c = self.ata_a;
        let diff = self.ata_c * 0.915965594177219015;

        x = dry + ((diff + half_diff + self.last_diff) / 1.187);
        self.last_diff = diff / 2.0;
        x *= k.output;
        (dry * (1.0 - k.wet)) + (x * k.wet)
    }
}

#[derive(Clone, Debug)]
pub struct Density2 {
    pub params: [f32; 4],
    pub sample_rate: f32,
    left: Channel,
    right: Channel,
    fpd_l: u32,
    fpd_r: u32,
}

impl Density2 {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            params: [0.2_f64 as f32, 0.0, 1.0, 1.0],
            sample_rate,
            left: Channel::default(),
            right: Channel::default(),
            fpd_l: SEEDS.0,
            fpd_r: SEEDS.1,
        }
    }

    pub fn clear(&mut self) {
        *self = Self {
            params: self.params,
            ..Self::new(self.sample_rate)
        };
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let [a, b, c, d] = self.params.map(f64::from);
        let mut overallscale = 1.0;
        overallscale /= 44100.0;
        overallscale *= f64::from(self.sample_rate);
        let density = (a * 5.0) - 1.0;
        let mut out = density.abs();
        while out > 1.0 {
            out -= 1.0;
        }
        let k = Block {
            density: density * density.abs(),
            out,
            iir_amount: pow(b, 3.0) / overallscale,
            output: c,
            wet: d,
        };
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let sl = self.left.run(guard(*l, self.fpd_l), &k);
            let sr = self.right.run(guard(*r, self.fpd_r), &k);
            *l = dither(sl, &mut self.fpd_l) as f32;
            *r = dither(sr, &mut self.fpd_r) as f32;
        }
    }
}
