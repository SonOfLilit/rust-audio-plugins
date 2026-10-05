//! slPlugins' Oscilloscope, SpectrumAnalyzer and XYScope: processors that
//! let their audio through untouched and hand a copy to their editor,
//! which draws it.
//!
//! Ported from `plugins/{Oscilloscope,SpectrumAnalyzer,XYScope}/Source/
//! PluginProcessor.cpp`: their knobs, in the order the constructors add
//! them, with their text functions. What the processors do to audio is
//! nothing; the copy goes out through the host (bloop's engine sends the
//! audio of the effect a screen watches), and the drawing is
//! `bloop-plugin-screens`. Left behind: the Oscilloscope's pitch
//! detection (Cycfi Q, not vendored) and its recorder, which write files.
//!
//! The XYScope has no knobs in the original. Gin's `XYScope` has a zoom
//! its plug-in never turns; here it is a knob, **Zoom**, of bloop's
//! choosing, so the scope has a page to show.

#![expect(clippy::doc_markdown, reason = "the plug-ins' names, as slPlugins spells them")]

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::param::{Conversion, Param, ParamSpec, Smoothing};
use crate::gin::text::Text;
use crate::juce::NormalisableRange as Range;

const fn knob(name: &'static str, range: Range, default: f32, text: Text) -> ParamSpec {
    ParamSpec::new(name, "", range, default, Smoothing::None, Conversion::Identity).with_text(text)
}

/// The Oscilloscope's knobs, as its constructor adds them.
pub const OSCILLOSCOPE: [ParamSpec; 11] = [
    knob("Samp/px", Range::new(0.0625, 48.0, 0.0, 0.3), 1.0, Text::SamplesPerPixel),
    knob("Zoom", Range::new(0.1, 100.0, 0.0, 0.3), 1.0, Text::Number),
    knob("Offset L", Range::new(-2.0, 2.0, 0.0, 1.0), 0.0, Text::Number),
    knob("Offset R", Range::new(-2.0, 2.0, 0.0, 1.0), 0.0, Text::Number),
    knob("Trigger Chan", Range::new(-1.0, 1.0, 1.0, 1.0), 0.0, Text::Channel),
    knob("Trigger Mode", Range::new(0.0, 3.0, 1.0, 1.0), 3.0, Text::TriggerMode),
    knob("Trigger Level", Range::new(-1.0, 1.0, 0.0, 1.0), 0.0, Text::TriggerLevel),
    knob("Trigger Pos", Range::new(0.0, 1.0, 0.0, 1.0), 0.0, Text::TwoDecimals),
    knob("Trigger Run", Range::new(0.0, 1.0, 1.0, 1.0), 0.0, Text::Run),
    knob("Sync", Range::new(0.0, 1.0, 1.0, 1.0), 0.0, Text::OnOff),
    knob("Beats", Range::new(1.0, 32.0, 1.0, 1.0), 4.0, Text::Beats),
];

/// Where each of the Oscilloscope's knobs is.
pub mod oscilloscope {
    pub const SAMPLES_PER_PIXEL: usize = 0;
    pub const ZOOM: usize = 1;
    pub const OFFSET_L: usize = 2;
    pub const OFFSET_R: usize = 3;
    pub const TRIGGER_CHANNEL: usize = 4;
    pub const TRIGGER_MODE: usize = 5;
    pub const TRIGGER_LEVEL: usize = 6;
    pub const TRIGGER_POS: usize = 7;
    pub const TRIGGER_RUN: usize = 8;
    pub const SYNC: usize = 9;
    pub const BEATS: usize = 10;
}

/// The SpectrumAnalyzer's: which of drow's two displays, and whether its
/// frequencies run on a log scale.
pub const SPECTRUM_ANALYZER: [ParamSpec; 2] = [
    knob("Mode", Range::new(0.0, 1.0, 1.0, 1.0), 0.0, Text::AnalyzerMode),
    knob("Log", Range::new(0.0, 1.0, 1.0, 1.0), 0.0, Text::OnOff),
];

/// The XYScope's one knob, bloop's: Gin's `setZoomFactor`.
pub const XY_SCOPE: [ParamSpec; 1] = [knob(
    "Zoom",
    Range::new(0.25, 4.0, 0.0, 0.5),
    1.0,
    Text::Number,
)];

/// Which of the three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Analyser {
    Oscilloscope,
    SpectrumAnalyzer,
    XyScope,
}

impl Analyser {
    #[must_use]
    pub fn specs(self) -> &'static [ParamSpec] {
        match self {
            Self::Oscilloscope => &OSCILLOSCOPE,
            Self::SpectrumAnalyzer => &SPECTRUM_ANALYZER,
            Self::XyScope => &XY_SCOPE,
        }
    }
}

/// One of the three: knobs, and audio let through.
#[derive(Clone, Debug)]
pub struct ScopeProcessor {
    params: Vec<Param>,
}

impl ScopeProcessor {
    #[must_use]
    pub fn new(scope: Analyser) -> Self {
        Self {
            params: scope.specs().iter().copied().map(Param::new).collect(),
        }
    }
}

impl Processor for ScopeProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
    }

    /// Nothing: the audio goes through as it came.
    fn process(&mut self, _left: &mut [f32], _right: &mut [f32], _host: Host) {}
}
