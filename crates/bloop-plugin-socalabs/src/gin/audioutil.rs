//! Helpers from Gin's `gin_dsp/dsp/gin_audioutil.h`.

use crate::math;

/// `gin::WetDryMix`: an equal-power crossfade, `mix` 0 dry to 1 wet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WetDryMix {
    pub wet_gain: f32,
    pub dry_gain: f32,
}

impl WetDryMix {
    #[must_use]
    pub fn new(mix: f32) -> Self {
        Self {
            wet_gain: mix_to_gain(mix),
            dry_gain: mix_to_gain(1.0 - mix),
        }
    }
}

fn mix_to_gain(mix: f32) -> f32 {
    math::sinf(mix * (0.5 * core::f32::consts::PI))
}
