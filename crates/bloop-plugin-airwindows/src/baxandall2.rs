//! Baxandall2: a two-knob shelving EQ, ±24 dB. The treble is the input
//! less a lowpass, the bass a lowpass, each scaled and summed; each
//! filter's corner moves against its gain, and each is two biquads that
//! take alternate samples.

use crate::math::{dither, guard, pow, tan, SEEDS};
use core::f64::consts::PI;

/// How long Baxandall2 rings after its input stops, at most: its bass
/// lowpass, at full boost a +24 dB resonance near 500 Hz, takes some tens
/// of milliseconds to fall below -90 dBFS after a hot burst
/// (tests/behaviour.rs).
pub const TAIL_SECONDS: f32 = 0.05;

/// `trebleAL[9]` and its kin: frequency, resonance, five coefficients and
/// a transposed-direct-form-II state.
type Biquad = [f64; 9];

/// The coefficients, written into all four of a filter's biquads.
fn coefficients(biquads: [&mut Biquad; 4], freq: f64, reso: f64) {
    let k = tan(PI * freq);
    let norm = 1.0 / (1.0 + k / reso + k * k);
    let a0 = k * k * norm;
    let b1 = 2.0 * (k * k - 1.0) * norm;
    let b2 = (1.0 - k / reso + k * k) * norm;
    for b in biquads {
        b[0] = freq;
        b[1] = reso;
        b[2] = a0;
        b[3] = 2.0 * a0;
        b[4] = a0;
        b[5] = b1;
        b[6] = b2;
    }
}

fn run(b: &mut Biquad, x: f64) -> f64 {
    let y = (x * b[2]) + b[7];
    b[7] = (x * b[3]) - (y * b[5]) + b[8];
    b[8] = (x * b[4]) - (y * b[6]);
    y
}

#[derive(Clone, Debug)]
pub struct Baxandall2 {
    pub params: [f32; 2],
    pub sample_rate: f32,
    treble_al: Biquad,
    treble_bl: Biquad,
    bass_al: Biquad,
    bass_bl: Biquad,
    treble_ar: Biquad,
    treble_br: Biquad,
    bass_ar: Biquad,
    bass_br: Biquad,
    flip: bool,
    fpd_l: u32,
    fpd_r: u32,
}

impl Baxandall2 {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            params: [0.5, 0.5],
            sample_rate,
            treble_al: [0.0; 9],
            treble_bl: [0.0; 9],
            bass_al: [0.0; 9],
            bass_bl: [0.0; 9],
            treble_ar: [0.0; 9],
            treble_br: [0.0; 9],
            bass_ar: [0.0; 9],
            bass_br: [0.0; 9],
            flip: false,
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
        let [a, b] = self.params.map(f64::from);
        let rate = f64::from(self.sample_rate);
        let treble_gain = pow(10.0, ((a * 48.0) - 24.0) / 20.0);
        let mut treble_freq = (4410.0 * treble_gain) / rate;
        if treble_freq > 0.45 {
            treble_freq = 0.45;
        }
        let bass_gain = pow(10.0, ((b * 48.0) - 24.0) / 20.0);
        let mut bass_freq = pow(10.0, -((b * 48.0) - 24.0) / 20.0);
        bass_freq = (8820.0 * bass_freq) / rate;
        if bass_freq > 0.45 {
            bass_freq = 0.45;
        }
        coefficients(
            [
                &mut self.treble_al,
                &mut self.treble_bl,
                &mut self.treble_ar,
                &mut self.treble_br,
            ],
            treble_freq,
            0.4,
        );
        coefficients(
            [
                &mut self.bass_al,
                &mut self.bass_bl,
                &mut self.bass_ar,
                &mut self.bass_br,
            ],
            bass_freq,
            0.2,
        );

        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let il = guard(*l, self.fpd_l);
            let ir = guard(*r, self.fpd_r);
            let (tl, bl, tr, br) = if self.flip {
                (
                    &mut self.treble_al,
                    &mut self.bass_al,
                    &mut self.treble_ar,
                    &mut self.bass_ar,
                )
            } else {
                (
                    &mut self.treble_bl,
                    &mut self.bass_bl,
                    &mut self.treble_br,
                    &mut self.bass_br,
                )
            };
            let treble_l = il - run(tl, il);
            let bass_l = run(bl, il);
            let treble_r = ir - run(tr, ir);
            let bass_r = run(br, ir);
            self.flip = !self.flip;

            let sl = (bass_l * bass_gain) + (treble_l * treble_gain);
            let sr = (bass_r * bass_gain) + (treble_r * treble_gain);
            *l = dither(sl, &mut self.fpd_l) as f32;
            *r = dither(sr, &mut self.fpd_r) as f32;
        }
    }
}
