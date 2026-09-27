//! `gin::StereoEnhancer` and `gin::StereoProcessor`, both after Michael
//! "LOSER" Gruhn's stereo tools (ISC; see LICENSE).
//!
//! Ported from Gin's `gin_dsp/dsp/gin_stereoenhancer.{h,cpp}` and
//! `gin_stereoprocessor.{h,cpp}`.

use crate::math;
use core::f32::consts::PI;

/// `gin::sign`: +1 above zero, otherwise -1.
fn sign(x: f32) -> i32 {
    if x > 0.0 {
        1
    } else {
        -1
    }
}

/// `gin::StereoEnhancer`: the signal split at a crossover, and each half
/// widened on its own. The one processor that uses it never sets its
/// sample rate, so the crossover is always where 44.1kHz puts it.
#[derive(Clone, Debug)]
pub struct StereoEnhancer {
    sample_rate: f32,
    width_lp: f32,
    width_coeff_lp: f32,
    width_hp: f32,
    width_coeff_hp: f32,
    a0_hp: f32,
    b1_hp: f32,
    tmp1_hp: f32,
    tmp2_hp: f32,
}

impl Default for StereoEnhancer {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            width_lp: 0.0,
            width_coeff_lp: 0.0,
            width_hp: 0.0,
            width_coeff_hp: 0.0,
            a0_hp: 0.0,
            b1_hp: 0.0,
            tmp1_hp: 0.0,
            tmp2_hp: 0.0,
        }
    }
}

impl StereoEnhancer {
    pub fn set_parameters(&mut self, width_lp: f32, freq_hp_fader: f32, width_hp: f32) {
        self.width_lp = width_lp;
        self.width_coeff_lp = math::std_max(width_lp, 1.0);
        let freq_hp = freq_hp_fader * freq_hp_fader * freq_hp_fader * 24000.0;
        let x_hp = math::expf(-2.0 * PI * freq_hp / self.sample_rate);
        self.a0_hp = 1.0 - x_hp;
        self.b1_hp = -x_hp;
        self.width_hp = width_hp;
        self.width_coeff_hp = math::std_max(width_hp, 1.0);
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        const DC_ADD: f32 = 1e-30;
        for (out1, out2) in left.iter_mut().zip(right.iter_mut()) {
            let (in1, in2) = (*out1, *out2);
            self.tmp1_hp = self.a0_hp * in1 - self.b1_hp * self.tmp1_hp + DC_ADD;
            self.tmp2_hp = self.a0_hp * in2 - self.b1_hp * self.tmp2_hp + DC_ADD;
            let (lp1, lp2) = (self.tmp1_hp, self.tmp2_hp);
            let hp1 = in1 - lp1;
            let hp2 = in2 - lp2;

            let mono_lp = (lp1 + lp2) / 2.0;
            let stereo_lp = lp1 - lp2;
            let o1 = (mono_lp + stereo_lp * self.width_lp) / self.width_coeff_lp;
            let o2 = (mono_lp - stereo_lp * self.width_lp) / self.width_coeff_lp;

            let mono_hp = (hp1 + hp2) / 2.0;
            let stereo_hp = hp1 - hp2;
            let h1 = (mono_hp + stereo_hp * self.width_hp) / self.width_coeff_hp;
            let h2 = (mono_hp - stereo_hp * self.width_hp) / self.width_coeff_hp;

            *out1 = o1 + h1;
            *out2 = o2 + h2;
        }
    }
}

/// `gin::StereoProcessor`: width, centre and pan, a rotation of the stereo
/// field, pan, centre and width again, and a level.
#[derive(Clone, Debug, Default)]
pub struct StereoProcessor {
    rotation: f32,
    width1: f32,
    width_coeff1: f32,
    center1: f32,
    side1: f32,
    width2: f32,
    width_coeff2: f32,
    center2: f32,
    side2: f32,
    pan1_left: f32,
    pan1_right: f32,
    pan2_left: f32,
    pan2_right: f32,
    out_vol: f32,
}

fn pan_gains(pan: f32) -> (f32, f32) {
    (
        if pan > 0.0 { 1.0 - pan } else { 1.0 },
        if pan < 0.0 { 1.0 + pan } else { 1.0 },
    )
}

impl StereoProcessor {
    #[expect(
        clippy::too_many_arguments,
        reason = "the original's signature: eight faders, each its own knob"
    )]
    pub fn set_parameters(
        &mut self,
        width1: f32,
        center_fader1: f32,
        pan1: f32,
        rotation_fader: f32,
        pan2: f32,
        center_fader2: f32,
        width2: f32,
        out_vol_fader: f32,
    ) {
        self.width1 = width1;
        self.width_coeff1 = math::std_max(width1, 1.0);
        self.center1 = math::std_min(center_fader1 * 2.0, 1.0);
        self.side1 = 2.0 - center_fader1 * 2.0;
        (self.pan1_left, self.pan1_right) = pan_gains(pan1);
        self.rotation = (rotation_fader - 0.5) * 180.0 * 0.017_453_292;
        (self.pan2_left, self.pan2_right) = pan_gains(pan2);
        self.center2 = math::std_min(center_fader2 * 2.0, 1.0);
        self.side2 = 2.0 - center_fader2 * 2.0;
        self.width2 = width2;
        self.width_coeff2 = math::std_max(width2, 1.0);
        let db2amp = |db: f32| math::expf(db * 0.115_524_53);
        let v = out_vol_fader - 0.5;
        let doubled = v * 2.0;
        self.out_vol = db2amp(sign(v) as f32 * (doubled * doubled) * 18.0);
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (out1, out2) in left.iter_mut().zip(right.iter_mut()) {
            let (in1, in2) = (*out1, *out2);
            let mono = ((in1 + in2) / 2.0) * self.center1;
            let stereo = (in1 - in2) * self.side1;
            let mut o1 = (mono + stereo * self.width1) / self.width_coeff1;
            let mut o2 = (mono - stereo * self.width1) / self.width_coeff1;

            o1 *= self.pan1_left;
            o2 *= self.pan1_right;

            if self.rotation != 0.0 {
                let mut angle;
                if o2 == 0.0 {
                    angle = if o1 > 0.0 { PI / 2.0 } else { PI * 1.5 };
                } else if o1 == 0.0 {
                    angle = if o2 > 0.0 { 0.0 } else { PI };
                } else {
                    let s1 = sign(o1);
                    let s2 = sign(o2);
                    angle = math::atanf(o1 / o2);
                    // `(s1 == 1 && s2 == -1) || (s1 == -1 && s2 == -1)` in
                    // the original: whichever sign the left has.
                    if s2 == -1 {
                        angle += PI;
                    }
                    if s1 == -1 && s2 == 1 {
                        angle += PI;
                    }
                }
                // `std::pow (x, 2.0f)`, which GCC compiles as `x * x`.
                let radius = (o1 * o1 + o2 * o2).sqrt();
                angle -= self.rotation;
                o1 = math::sinf(angle) * radius;
                o2 = math::cosf(angle) * radius;
            }

            o1 *= self.pan2_left;
            o2 *= self.pan2_right;

            let mono = ((o1 + o2) / 2.0) * self.center2;
            let stereo = (o1 - o2) * self.side2;
            o1 = (mono + stereo * self.width2) / self.width_coeff2;
            o2 = (mono - stereo * self.width2) / self.width_coeff2;

            *out1 = o1 * self.out_vol;
            *out2 = o2 * self.out_vol;
        }
    }
}
