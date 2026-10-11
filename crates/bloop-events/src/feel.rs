//! How hard a pad plays for where it is hit, and how a controller's pad
//! velocity is bent: SETTINGS, HARDWARE, Knob 1 (Pads SENSITIVITY) and
//! Knob 2 (Pads SCALING), bloop#pad-feel.
//!
//! A screen has no pressure sensor, so a screen pad plays by where the
//! finger lands: its *distance* out from the pad's centre, 0 at the
//! centre and 1 in a corner. SENSITIVITY sets the range that distance
//! covers, from the velocity a corner plays ([`PadFeel::edge`]) to full in
//! the middle, which past 50 is a disc that grows ([`PadFeel::middle`]).
//! SCALING is the curve between them: Linear is a straight line, Soft n
//! plays a hit in between louder, Hard n softer, as the hardness raised to
//! [`Scaling::exponent`].
//!
//! At SENSITIVITY 50 and Linear a pad plays what it always has: full at
//! the centre, falling in a straight line to [`EDGE_VELOCITY`] in the
//! corner. The math is `libm`'s, so it is the same on every target.

/// What a corner plays at SENSITIVITY 50: a fifth of full.
pub const EDGE_VELOCITY: f32 = 0.2;

/// What a corner plays at SENSITIVITY 0: velocity 1 of 127, the softest
/// a note-on carries.
pub const SOFTEST_EDGE: f32 = 1.0 / 127.0;

/// What a corner plays at SENSITIVITY 100.
pub const HARDEST_EDGE: f32 = 0.8;

/// How far out the middle that plays full reaches at SENSITIVITY 100, in
/// the distance's measure: halfway to the corner.
pub const WIDEST_MIDDLE: f32 = 0.5;

/// SENSITIVITY's default, and the setting a pad has always played at.
pub const DEFAULT_SENSITIVITY: f32 = 50.0;

/// Pads SCALING: the curve from a soft hit to a hard one.
///
/// Its steps run from Soft 3 (-3) through Linear (0) to Hard 3 (3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Scaling(i8);

impl Scaling {
    /// A straight line: what a pad has always played.
    pub const LINEAR: Scaling = Scaling(0);

    /// The step `steps` from Linear, Soft below and Hard above; `None`
    /// past 3 either way.
    #[must_use]
    pub fn new(steps: i8) -> Option<Scaling> {
        (-3..=3).contains(&steps).then_some(Scaling(steps))
    }

    /// The option the spec names it by: `Soft 3` … `Linear` … `Hard 3`.
    /// `None` for anything else.
    #[must_use]
    pub fn parse(text: &str) -> Option<Scaling> {
        if text == "Linear" {
            return Some(Scaling::LINEAR);
        }
        let (sign, n) = if let Some(n) = text.strip_prefix("Soft ") {
            (-1, n)
        } else {
            (1, text.strip_prefix("Hard ")?)
        };
        let n: i8 = n.parse().ok().filter(|n| (1..=3).contains(n))?;
        Scaling::new(sign * n)
    }

    /// Steps from Linear: negative Soft, positive Hard.
    #[must_use]
    pub fn steps(self) -> i8 {
        self.0
    }

    /// The power a hit's hardness (0 softest, 1 full) is raised to:
    /// 2 to the half of the steps. Hard 2 squares it, so a hit halfway
    /// plays a quarter of the way; Soft 2 takes its square root, so it
    /// plays seven tenths of the way.
    #[must_use]
    pub fn exponent(self) -> f32 {
        libm::exp2f(f32::from(self.0) / 2.0)
    }

    /// A hardness, 0 to 1, bent.
    #[must_use]
    pub fn bend(self, hardness: f32) -> f32 {
        let hardness = hardness.clamp(0.0, 1.0);
        if self == Scaling::LINEAR {
            return hardness;
        }
        libm::powf(hardness, self.exponent())
    }

    /// [`Scaling::bend`]'s inverse.
    #[must_use]
    pub fn unbend(self, bent: f32) -> f32 {
        let bent = bent.clamp(0.0, 1.0);
        if self == Scaling::LINEAR {
            return bent;
        }
        libm::powf(bent, 1.0 / self.exponent())
    }

    /// A controller pad's velocity or pressure, 0–127, bent: the device
    /// measured how hard it was hit, so only the curve applies. 0 stays 0
    /// and anything above stays at least 1, so a note-on stays a note-on.
    #[must_use]
    pub fn midi(self, value: u8) -> u8 {
        if value == 0 || self == Scaling::LINEAR {
            return value;
        }
        let bent = self.bend(f32::from(value.min(127)) / 127.0) * 127.0;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 1..=127 first"
        )]
        let bent = libm::roundf(bent).clamp(1.0, 127.0) as u8;
        bent
    }
}

/// Pads SENSITIVITY and SCALING together: what a screen pad plays for
/// where it is hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadFeel {
    sensitivity: f32,
    scaling: Scaling,
}

impl Default for PadFeel {
    fn default() -> PadFeel {
        PadFeel {
            sensitivity: DEFAULT_SENSITIVITY,
            scaling: Scaling::LINEAR,
        }
    }
}

impl PadFeel {
    /// SENSITIVITY 0–100 (clamped) and SCALING.
    #[must_use]
    pub fn new(sensitivity: f32, scaling: Scaling) -> PadFeel {
        PadFeel {
            sensitivity: if sensitivity.is_nan() {
                DEFAULT_SENSITIVITY
            } else {
                sensitivity.clamp(0.0, 100.0)
            },
            scaling,
        }
    }

    #[must_use]
    pub fn scaling(self) -> Scaling {
        self.scaling
    }

    /// SENSITIVITY over its default: 0 to 2, 1 at 50.
    fn above(self) -> f32 {
        self.sensitivity / DEFAULT_SENSITIVITY
    }

    /// What a hit in the corner plays, 0 to 1: [`SOFTEST_EDGE`] at
    /// SENSITIVITY 0, [`EDGE_VELOCITY`] at 50 and [`HARDEST_EDGE`] at 100,
    /// in a straight line between each.
    #[must_use]
    pub fn edge(self) -> f32 {
        let k = self.above();
        if k <= 1.0 {
            EDGE_VELOCITY - (EDGE_VELOCITY - SOFTEST_EDGE) * (1.0 - k)
        } else {
            EDGE_VELOCITY + (HARDEST_EDGE - EDGE_VELOCITY) * (k - 1.0)
        }
    }

    /// How far out a hit still plays full: the centre alone up to
    /// SENSITIVITY 50, then a disc growing to [`WIDEST_MIDDLE`] at 100.
    #[must_use]
    pub fn middle(self) -> f32 {
        (WIDEST_MIDDLE * (self.above() - 1.0)).max(0.0)
    }

    /// What a hit `distance` out from a pad's centre (0 the centre, 1 the
    /// corner; clamped) plays, 0 to 1.
    #[must_use]
    pub fn velocity(self, distance: f32) -> f32 {
        let distance = distance.clamp(0.0, 1.0);
        let middle = self.middle();
        // 0 at the middle's rim, 1 in the corner.
        let out = if middle == 0.0 {
            distance
        } else {
            ((distance - middle) / (1.0 - middle)).clamp(0.0, 1.0)
        };
        let softness = if self.scaling == Scaling::LINEAR {
            out
        } else {
            1.0 - self.scaling.bend(1.0 - out)
        };
        1.0 - (1.0 - self.edge()) * softness
    }

    /// How far out from a pad's centre a hit still plays `velocity`, from
    /// 0 to 1, or harder, in [`PadFeel::velocity`]'s measure: its inverse,
    /// so a ring or a finger drawn there cannot drift from what a hit
    /// there plays. Full or more is the middle's rim; the corner's
    /// velocity or less, the corner.
    #[must_use]
    pub fn reach(self, velocity: f32) -> f32 {
        let edge = self.edge();
        let out = if self.scaling == Scaling::LINEAR {
            ((1.0 - velocity) / (1.0 - edge)).clamp(0.0, 1.0)
        } else {
            1.0 - self.scaling.unbend((velocity - edge) / (1.0 - edge))
        };
        let middle = self.middle();
        middle + out * (1.0 - middle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;
    use expect_test::expect;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    const STEPS: [&str; 7] = [
        "Soft 3", "Soft 2", "Soft 1", "Linear", "Hard 1", "Hard 2", "Hard 3",
    ];

    fn scaling(text: &str) -> Scaling {
        Scaling::parse(text).unwrap_or_else(|| panic!("{text} is a step"))
    }

    /// What a hit plays, in MIDI velocity, out from the centre to the
    /// corner, at each SENSITIVITY and SCALING: the edge is the least a
    /// pad plays, the centre the most.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_hit_plays_by_how_far_out_it_lands() {
        let mut out = String::from(
            "sens  scaling  | centre  0.25  half  0.75  corner | middle\n",
        );
        for sensitivity in [0.0, 25.0, 50.0, 75.0, 100.0] {
            for step in STEPS {
                let feel = PadFeel::new(sensitivity, scaling(step));
                let row: Vec<String> = [0.0, 0.25, 0.5, 0.75, 1.0]
                    .into_iter()
                    .map(|d| format!("{:>5.1}", feel.velocity(d) * 127.0))
                    .collect();
                out += &format!(
                    "{sensitivity:>4}  {step:<7}  | {} | {:.3}\n",
                    row.join(" "),
                    feel.middle()
                );
            }
        }
        expect![[r#"
            sens  scaling  | centre  0.25  half  0.75  corner | middle
               0  Soft 3   | 127.0 114.8  99.6  78.2   1.0 | 0.000
               0  Soft 2   | 127.0 110.1  90.1  64.0   1.0 | 0.000
               0  Soft 1   | 127.0 103.8  78.2  48.3   1.0 | 0.000
               0  Linear   | 127.0  95.5  64.0  32.5   1.0 | 0.000
               0  Hard 1   | 127.0  84.9  48.3  18.7   1.0 | 0.000
               0  Hard 2   | 127.0  71.9  32.5   8.9   1.0 | 0.000
               0  Hard 3   | 127.0  56.8  18.7   3.5   1.0 | 0.000
              25  Soft 3   | 127.0 116.0 102.3  82.9  13.2 | 0.000
              25  Soft 2   | 127.0 111.8  93.7  70.1  13.2 | 0.000
              25  Soft 1   | 127.0 106.1  82.9  55.9  13.2 | 0.000
              25  Linear   | 127.0  98.6  70.1  41.6  13.2 | 0.000
              25  Hard 1   | 127.0  89.0  55.9  29.2  13.2 | 0.000
              25  Hard 2   | 127.0  77.2  41.6  20.3  13.2 | 0.000
              25  Hard 3   | 127.0  63.6  29.2  15.5  13.2 | 0.000
              50  Soft 3   | 127.0 117.2 104.9  87.6  25.4 | 0.000
              50  Soft 2   | 127.0 113.4  97.2  76.2  25.4 | 0.000
              50  Soft 1   | 127.0 108.3  87.6  63.5  25.4 | 0.000
              50  Linear   | 127.0 101.6  76.2  50.8  25.4 | 0.000
              50  Hard 1   | 127.0  93.0  63.5  39.7  25.4 | 0.000
              50  Hard 2   | 127.0  82.5  50.8  31.8  25.4 | 0.000
              50  Hard 3   | 127.0  70.4  39.7  27.4  25.4 | 0.000
              75  Soft 3   | 127.0 127.0 118.5 106.6  63.5 | 0.250
              75  Soft 2   | 127.0 127.0 115.3 100.2  63.5 | 0.250
              75  Soft 1   | 127.0 127.0 111.2  92.7  63.5 | 0.250
              75  Linear   | 127.0 127.0 105.8  84.7  63.5 | 0.250
              75  Hard 1   | 127.0 127.0  99.3  76.9  63.5 | 0.250
              75  Hard 2   | 127.0 127.0  91.7  70.6  63.5 | 0.250
              75  Hard 3   | 127.0 127.0  83.7  66.3  63.5 | 0.250
             100  Soft 3   | 127.0 127.0 127.0 121.5 101.6 | 0.500
             100  Soft 2   | 127.0 127.0 127.0 119.6 101.6 | 0.500
             100  Soft 1   | 127.0 127.0 127.0 117.2 101.6 | 0.500
             100  Linear   | 127.0 127.0 127.0 114.3 101.6 | 0.500
             100  Hard 1   | 127.0 127.0 127.0 111.1 101.6 | 0.500
             100  Hard 2   | 127.0 127.0 127.0 108.0 101.6 | 0.500
             100  Hard 3   | 127.0 127.0 127.0 105.2 101.6 | 0.500
        "#]].assert_eq(&out);
    }

    /// At the defaults a pad plays exactly what it did before the knobs
    /// did anything: full at the centre, falling in a straight line to a
    /// fifth in the corner, to the last bit.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_defaults_play_as_a_pad_always_has() {
        let feel = PadFeel::default();
        let differ: Vec<f32> = (0..=1000)
            .map(|i| i as f32 / 1000.0)
            .filter(|&d| feel.velocity(d) != 1.0 - (1.0 - 0.2) * d)
            .collect();
        expect!["[]"].assert_eq(&format!("{differ:?}"));
    }

    /// The reach of a velocity is where a hit plays it, at every setting:
    /// read back through [`PadFeel::velocity`], each comes back within a
    /// quarter of a MIDI step, from the corner's velocity up to full. The
    /// quarter is f32's last bit of a distance near the corner, where Soft
    /// 3's curve is steepest; a pixel there is many times that.
    /// Under the corner's it is the corner; full or more, the middle's
    /// rim.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_velocity_reaches_out_to_where_a_hit_plays_it() {
        let mut worst = 0.0f32;
        let mut ends = String::new();
        for sensitivity in (0..=100).step_by(5) {
            for step in STEPS {
                #[expect(clippy::cast_precision_loss, reason = "0 to 100")]
                let feel = PadFeel::new(sensitivity as f32, scaling(step));
                for v in 1..=127u8 {
                    let velocity = f32::from(v) / 127.0;
                    if velocity < feel.edge() {
                        continue;
                    }
                    let back = feel.velocity(feel.reach(velocity));
                    worst = worst.max((back - velocity).abs() * 127.0);
                }
                if sensitivity % 50 == 0 && (step == "Linear" || step == "Hard 3") {
                    ends += &format!(
                        "{sensitivity:>3} {step:<6}: reach of 0 {:.3}, of the edge {:.3}, of full {:.3}, of 1.5 {:.3}\n",
                        feel.reach(0.0),
                        feel.reach(feel.edge()),
                        feel.reach(1.0),
                        feel.reach(1.5),
                    );
                }
            }
        }
        expect![[r#"
            worst 0.2400 of a MIDI step
              0 Linear: reach of 0 1.000, of the edge 1.000, of full 0.000, of 1.5 0.000
              0 Hard 3: reach of 0 1.000, of the edge 1.000, of full 0.000, of 1.5 0.000
             50 Linear: reach of 0 1.000, of the edge 1.000, of full 0.000, of 1.5 0.000
             50 Hard 3: reach of 0 1.000, of the edge 1.000, of full 0.000, of 1.5 0.000
            100 Linear: reach of 0 1.000, of the edge 1.000, of full 0.500, of 1.5 0.500
            100 Hard 3: reach of 0 1.000, of the edge 1.000, of full 0.500, of 1.5 0.500
        "#]].assert_eq(&format!("worst {worst:.4} of a MIDI step\n{ends}"));
    }

    /// The options parse, and nothing else does; out-of-range
    /// sensitivities clamp.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_settings_read() {
        let steps: Vec<String> = STEPS
            .into_iter()
            .chain(["linear", "Soft 4", "Hard 0", "Hard", "", "Soft  1"])
            .map(|text| match Scaling::parse(text) {
                Some(s) => format!("{text:?} {} x^{:.3}", s.steps(), s.exponent()),
                None => format!("{text:?} none"),
            })
            .collect();
        let clamped: Vec<String> = [-10.0, 0.0, 150.0, f32::NAN]
            .into_iter()
            .map(|s| format!("{s}: edge {:.3}", PadFeel::new(s, Scaling::LINEAR).edge()))
            .collect();
        expect![[r#"
            "Soft 3" -3 x^0.354
            "Soft 2" -2 x^0.500
            "Soft 1" -1 x^0.707
            "Linear" 0 x^1.000
            "Hard 1" 1 x^1.414
            "Hard 2" 2 x^2.000
            "Hard 3" 3 x^2.828
            "linear" none
            "Soft 4" none
            "Hard 0" none
            "Hard" none
            "" none
            "Soft  1" none
            -10: edge 0.008
            0: edge 0.008
            150: edge 0.800
            NaN: edge 0.200"#]].assert_eq(&format!("{}\n{}", steps.join("\n"), clamped.join("\n")));
    }

    /// A controller's pad velocity, bent by SCALING alone: 0 stays 0 (a
    /// pressure let go), 1 stays a note, 127 stays full.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_controller_s_velocity_bends() {
        let mut out = String::from("        ");
        let values = [0u8, 1, 16, 32, 64, 96, 127];
        for v in values {
            out += &format!("{v:>5}");
        }
        for step in STEPS {
            out += &format!("\n{step:<7} ");
            for v in values {
                out += &format!("{:>5}", scaling(step).midi(v));
            }
        }
        expect![[r#"
                        0    1   16   32   64   96  127
            Soft 3      0   23   61   78  100  115  127
            Soft 2      0   11   45   64   90  110  127
            Soft 1      0    4   29   48   78  104  127
            Linear      0    1   16   32   64   96  127
            Hard 1      0    1    7   18   48   85  127
            Hard 2      0    1    2    8   32   73  127
            Hard 3      0    1    1    3   18   58  127"#]].assert_eq(&out);
    }
}
