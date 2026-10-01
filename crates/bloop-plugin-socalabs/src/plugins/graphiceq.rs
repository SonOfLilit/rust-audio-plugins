//! slPlugins' GraphicEQ: ten octave bands, a switchable highpass and
//! lowpass after them, and an output level.
//!
//! Ported from `plugins/GraphicEQ/Source/PluginProcessor.cpp`.

use super::{prepare_params, reset_params, Host, Processor};
use crate::gin::filter::{Filter, FilterType, Slope};
use crate::gin::graphiceq::{GraphicEq, BANDS};
use crate::gin::param::{
    any_smoothing, midi_note_from_hertz, Conversion, Param, ParamSpec, Smoothing,
};
use crate::gin::text::Text;
use crate::juce::NormalisableRange as Range;

const SMOOTH: Smoothing = Smoothing::Linear(0.1);
const SWITCH: Range = Range::new(0.0, 1.0, 1.0, 1.0);

const BAND_NAMES: [&str; BANDS] = [
    "31 Hz", "63 Hz", "125 Hz", "250 Hz", "500 Hz", "1k Hz", "2k Hz", "4k Hz", "8k Hz", "16k Hz",
];

const LP_ENABLE: usize = BANDS;
const LP_FREQ: usize = BANDS + 1;
const LP_SLOPE: usize = BANDS + 2;
const HP_ENABLE: usize = BANDS + 3;
const HP_FREQ: usize = BANDS + 4;
const HP_SLOPE: usize = BANDS + 5;
const OUTPUT: usize = BANDS + 6;

/// The GraphicEQ's knobs, in the order its constructor adds them. The
/// filter frequencies run in MIDI notes, whose ends the original works out
/// from their frequencies, so this is a function and not a table.
#[must_use]
pub fn specs() -> [ParamSpec; BANDS + 7] {
    let band = |name| {
        ParamSpec::new(
            name,
            "dB",
            Range::new(-12.0, 12.0, 0.0, 1.0),
            0.0,
            SMOOTH,
            Conversion::Identity,
        )
    };
    let switch = |name, text| {
        ParamSpec::new(name, "", SWITCH, 0.0, Smoothing::None, Conversion::Identity).with_text(text)
    };
    let freq = |name, low: f32, high: f32, default: f32| {
        let (low, high) = (midi_note_from_hertz(low), midi_note_from_hertz(high));
        ParamSpec::new(
            name,
            "",
            Range::new(low, high, 0.0, 1.0),
            midi_note_from_hertz(default),
            SMOOTH,
            Conversion::MidiNoteToHertz,
        )
        .with_text(Text::NoteFrequency)
    };
    let bands = BAND_NAMES.map(band);
    [
        bands[0],
        bands[1],
        bands[2],
        bands[3],
        bands[4],
        bands[5],
        bands[6],
        bands[7],
        bands[8],
        bands[9],
        switch("LP", Text::OnOff),
        freq("LP Freq", 1000.0, 20000.0, 20000.0),
        switch("LP Slope", Text::Slope),
        switch("HP", Text::OnOff),
        freq("HP Freq", 20.0, 2000.0, 20.0),
        switch("HP Slope", Text::Slope),
        ParamSpec::new(
            "Output",
            "dB",
            Range::new(-12.0, 12.0, 0.0, 1.0),
            0.0,
            SMOOTH,
            Conversion::DecibelsToGain,
        ),
    ]
}

#[derive(Clone, Debug)]
pub struct GraphicEqProcessor {
    params: Vec<Param>,
    eq: GraphicEq,
    lp: Filter,
    hp: Filter,
}

impl Default for GraphicEqProcessor {
    fn default() -> Self {
        Self {
            params: specs().into_iter().map(Param::new).collect(),
            eq: GraphicEq::default(),
            lp: Filter::new(FilterType::LoPass),
            hp: Filter::new(FilterType::HiPass),
        }
    }
}

fn slope(on: bool) -> Slope {
    if on {
        Slope::Db24
    } else {
        Slope::Db12
    }
}

impl GraphicEqProcessor {
    /// One stretch of `processBlock`: the band gains, the EQ, each filter
    /// if it is on, the level. `steps` is the stretch's length, and each
    /// knob is read only where the original reads it — a filter that is
    /// off does not move its knobs' smoothing on.
    fn run(&mut self, left: &mut [f32], right: &mut [f32], steps: usize) {
        for b in 0..BANDS {
            let gain = self.params[b].proc_value(steps);
            self.eq.set_band_gain(0, b, gain);
            self.eq.set_band_gain(1, b, gain);
        }
        self.eq.process([&mut *left, &mut *right]);
        for (enable, freq, slope_param, filter) in [
            (HP_ENABLE, HP_FREQ, HP_SLOPE, &mut self.hp),
            (LP_ENABLE, LP_FREQ, LP_SLOPE, &mut self.lp),
        ] {
            if self.params[enable].proc_value(steps) > 0.5 {
                filter.set_slope(slope(self.params[slope_param].proc_value(steps) > 0.5));
                filter.set_params(self.params[freq].proc_value(steps), 0.707, 0.0);
                for (channel, data) in [&mut *left, &mut *right].into_iter().enumerate() {
                    for x in data.iter_mut() {
                        *x = filter.process(channel, *x);
                    }
                }
            }
        }
        let gain = self.params[OUTPUT].proc_value(steps);
        for x in left.iter_mut().chain(right.iter_mut()) {
            *x *= gain;
        }
    }
}

impl Processor for GraphicEqProcessor {
    fn params(&self) -> &[Param] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut [Param] {
        &mut self.params
    }

    fn prepare(&mut self, sample_rate: f64) {
        prepare_params(&mut self.params, sample_rate);
        self.eq.set_sample_rate(sample_rate);
        self.eq.set_num_channels(2);
        self.eq.reset();
        self.lp.set_sample_rate(sample_rate);
        self.lp.reset();
        self.hp.set_sample_rate(sample_rate);
        self.hp.reset();
        self.reset();
    }

    fn reset(&mut self) {
        reset_params(&mut self.params);
        self.eq.reset();
        self.lp.reset();
        self.hp.reset();
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32], _host: Host) {
        let samples = left.len().min(right.len());
        if any_smoothing(&self.params) {
            for i in 0..samples {
                self.run(&mut left[i..=i], &mut right[i..=i], 1);
            }
        } else {
            self.run(&mut left[..samples], &mut right[..samples], samples);
        }
    }
}
