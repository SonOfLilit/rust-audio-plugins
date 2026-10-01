//! slPlugins' Crossfeed: headphone crossfeed, faded in and out over 50ms
//! as it is switched.
//!
//! Ported from `plugins/Crossfeed/Source/PluginProcessor.cpp`. The
//! original filters a copy of the block and then mixes the two; here each
//! sample is filtered and mixed in one pass, which is the same arithmetic
//! in the same order without the copy.

use super::{reset_params, Host, Processor};
use crate::gin::crossfeed::Crossfeed;
use crate::gin::param::{Conversion, Param, ParamSpec, Smoothing};
use crate::gin::text::Text;
use crate::juce::{LinearSmoothedValue, NormalisableRange as Range};

pub const CROSSFEED: [ParamSpec; 1] = [ParamSpec::new(
    "Enable",
    "",
    Range::new(0.0, 1.0, 1.0, 1.0),
    1.0,
    Smoothing::None,
    Conversion::Identity,
)
.with_text(Text::OnOff)];

/// The rate with a kernel nearest `sample_rate`.
fn nearest_kernel_rate(sample_rate: f64) -> i32 {
    let rate = sample_rate as i32;
    let mut best = 44100;
    for candidate in [48000, 96000] {
        if (rate - candidate).abs() < (rate - best).abs() {
            best = candidate;
        }
    }
    best
}

#[derive(Clone, Debug)]
pub struct CrossfeedProcessor {
    params: Vec<Param>,
    crossfeed: Crossfeed,
    enable: LinearSmoothedValue,
    disable: LinearSmoothedValue,
}

impl Default for CrossfeedProcessor {
    fn default() -> Self {
        Self {
            params: CROSSFEED.iter().copied().map(Param::new).collect(),
            crossfeed: Crossfeed::new(44100)
                .unwrap_or_else(|| unreachable!("44.1kHz has a kernel")),
            enable: LinearSmoothedValue::default(),
            disable: LinearSmoothedValue::default(),
        }
    }
}

/// How one of the two gains applies across a block: `gin::applyGain`
/// ramps sample by sample if the value is moving when the block starts,
/// and otherwise multiplies by where it is headed.
fn gains(value: &mut LinearSmoothedValue) -> impl FnMut() -> f32 + '_ {
    let smoothing = value.is_smoothing();
    move || {
        if smoothing {
            value.next_value()
        } else {
            value.target()
        }
    }
}

impl Processor for CrossfeedProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    /// The original's `prepareToPlay` does not pass the rate on to its
    /// parameters, as the others do.
    fn prepare(&mut self, sample_rate: f64) {
        if let Some(crossfeed) = Crossfeed::new(nearest_kernel_rate(sample_rate)) {
            self.crossfeed = crossfeed;
        }
        self.enable.reset(sample_rate, 0.05);
        self.disable.reset(sample_rate, 0.05);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let on = self.params[0].user_value() > 0.5;
        self.enable.set_target(if on { 1.0 } else { 0.0 });
        self.disable.set_target(if on { 0.0 } else { 1.0 });
        let mut dry_gain = gains(&mut self.disable);
        let mut wet_gain = gains(&mut self.enable);
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (dry_l, dry_r) = (*l, *r);
            let (mut wet_l, mut wet_r) = (dry_l, dry_r);
            self.crossfeed.filter_in_place(
                core::slice::from_mut(&mut wet_l),
                core::slice::from_mut(&mut wet_r),
            );
            let d = dry_gain();
            let w = wet_gain();
            *l = dry_l * d + wet_l * w;
            *r = dry_r * d + wet_r * w;
        }
    }
}
