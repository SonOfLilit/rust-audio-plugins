//! What an effect's screen draws that the effect itself works out: the
//! GraphicEQ's response curve and a dynamics processor's transfer curve,
//! from where the knobs stand — the editors' `FilterResponseCurve` and
//! `gin::DynamicsMeter`, which ask the processor's own DSP.
//!
//! A screen is drawn on the UI thread, which does not hold the effect the
//! audio thread runs; it holds the knobs' values. So each curve is built
//! here from those values with the same DSP the effect is built from, and
//! answers what the effect would.

use crate::gin::dynamics::{Curve, DynamicsType};
use crate::gin::filter::{Filter, FilterType, Slope};
use crate::gin::graphiceq::{GraphicEq, BANDS};
use crate::gin::param::midi_note_in_hertz;
use crate::juce::decibels_to_gain;
use crate::plugins::graphiceq::{HP_ENABLE, HP_FREQ, HP_SLOPE, LP_ENABLE, LP_FREQ, LP_SLOPE, OUTPUT};
use crate::Kind;

/// The GraphicEQ's gain at any frequency, for knobs at `values` (each
/// knob's own value — dB, a note, a switch — by its index).
///
/// The original's curve is the ten bands and the two filters; this one
/// adds the Output knob, so the curve is what the whole effect does to a
/// tone and a turn of Output moves it, and sums the bands as complex gains
/// ([`GraphicEq::response`]).
#[derive(Clone, Debug)]
pub struct EqCurve {
    eq: GraphicEq,
    filters: [Option<Filter>; 2],
    output: f32,
}

impl EqCurve {
    #[must_use]
    pub fn new(values: &[f32], sample_rate: f32) -> Self {
        let value = |index: usize| values.get(index).copied().unwrap_or_default();
        let sample_rate = f64::from(sample_rate);
        let mut eq = GraphicEq::default();
        eq.set_sample_rate(sample_rate);
        eq.set_num_channels(1);
        for band in 0..BANDS {
            eq.set_band_gain(0, band, value(band));
        }
        let filter = |kind, enable: usize, freq: usize, slope: usize| {
            (value(enable) > 0.5).then(|| {
                let mut filter = Filter::new(kind);
                filter.set_sample_rate(sample_rate);
                filter.set_slope(if value(slope) > 0.5 {
                    Slope::Db24
                } else {
                    Slope::Db12
                });
                filter.set_params(midi_note_in_hertz(value(freq)), 0.707, 0.0);
                filter
            })
        };
        Self {
            eq,
            filters: [
                filter(FilterType::HiPass, HP_ENABLE, HP_FREQ, HP_SLOPE),
                filter(FilterType::LoPass, LP_ENABLE, LP_FREQ, LP_SLOPE),
            ],
            output: decibels_to_gain(value(OUTPUT)),
        }
    }

    /// The gain at `freq`, as a factor.
    #[must_use]
    pub fn magnitude(&self, freq: f32) -> f32 {
        let filters: f32 = self
            .filters
            .iter()
            .flatten()
            .map(|f| f.response_magnitude(freq))
            .product();
        self.eq.response(0, freq) * filters * self.output
    }

    /// The gain at `freq`, in dB.
    #[must_use]
    pub fn db(&self, freq: f32) -> f32 {
        bloop_plugin_api::meter::gain_to_db(self.magnitude(freq))
    }
}

/// A dynamics processor's transfer curve, for knobs at `values`: what its
/// gain computer makes of a level, and the Input and Output gains either
/// side of it.
#[derive(Clone, Copy, Debug)]
pub struct Transfer {
    curve: Curve,
    /// The Input knob, in dB: what the detector hears is the signal this
    /// much louder.
    pub input_db: f32,
    /// The Output knob, in dB.
    pub output_db: f32,
    /// Where the curve bends, in dB of what the detector hears.
    pub threshold_db: f32,
}

impl Transfer {
    /// The curve of a Compressor, Limiter, Gate or Expander; `None` for any
    /// other effect. The knobs are read as each processor hands them to its
    /// `gin::Dynamics`: the Limiter and the Gate with a ratio of 1000, the
    /// Limiter with no knee.
    #[must_use]
    pub fn of(kind: Kind, values: &[f32]) -> Option<Self> {
        let value = |index: usize| values.get(index).copied().unwrap_or_default();
        // (curve, threshold, ratio, knee, input, output), by knob index.
        let (shape, threshold, ratio, knee, input, output) = match kind {
            Kind::Compressor => (DynamicsType::Compressor, 3, Some(2), Some(4), 5, 6),
            Kind::Expander => (DynamicsType::Expander, 3, Some(2), Some(4), 5, 6),
            Kind::Limiter => (DynamicsType::Limiter, 2, None, None, 3, 4),
            Kind::Gate => (DynamicsType::Gate, 3, None, Some(4), 5, 6),
            _ => return None,
        };
        let threshold_db = value(threshold);
        Some(Self {
            curve: Curve::new(
                shape,
                threshold_db,
                ratio.map_or(1000.0, value),
                knee.map_or(0.0, value),
            ),
            input_db: value(input),
            output_db: value(output),
            threshold_db,
        })
    }

    /// `calcCurve`: the level the gain computer makes of `db_in`, in dB —
    /// the curve the original's `DynamicsMeter` draws.
    #[must_use]
    pub fn out_db(&self, db_in: f32) -> f32 {
        self.curve.calc(db_in)
    }
}
