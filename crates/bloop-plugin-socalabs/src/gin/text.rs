//! How a parameter prints its value: `gin::Parameter::getUserValueText`,
//! and the text functions slPlugins hands `addExtParam`.
//!
//! Ported from Gin's `gin_plugin/plugin/gin_parameter.cpp` (`getText`),
//! `gin/utilities/gin_util.h` (`formatNumber`), and the `*TextFunction`s
//! in slPlugins' `PluginProcessor.cpp`s. None of it reaches audio; it is
//! what a host shows over the knob.

use crate::gin::noteduration::DURATIONS;
use crate::gin::param::midi_note_in_hertz;
use crate::juce::approximately_equal;

/// Which of the originals' text functions a parameter prints with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Text {
    /// Gin's own, for a parameter given none: a whole number where the
    /// range steps by one, `formatNumber` otherwise, then the unit.
    Number,
    /// `onOffTextFunction` / `enableTextFunction`: "On" above zero.
    OnOff,
    /// BitCrusher's `freqTextFunction`: "500 Hz", "48.0 kHz".
    Frequency,
    /// GraphicEQ's `freqTextFunction`: a MIDI note, printed as the
    /// frequency it stands for.
    NoteFrequency,
    /// GraphicEQ's `slopeTextFunction`: "12 dB" or "24 dB" an octave.
    Slope,
    /// Delay's `durationTextFunction`: the note length the Beat knob picks.
    Duration,
    /// SimpleVerb's and the stereo plug-ins' `percentTextFunction`: a
    /// fader's 0 to 1 as a whole percentage.
    Percent,
}

impl Text {
    /// `value` as the original prints it, with `unit` after it, separated
    /// the way bloop's screens separate one ("120 BPM" but "50%"). The
    /// original glues its label on with no space and also appends it after
    /// a text function that already said Hz; neither reads well on a
    /// screen six characters wide.
    #[must_use]
    pub fn show(self, value: f32, unit: &str, interval: f32) -> String {
        match self {
            Text::Number => {
                // `getText`: "if (juce::exactlyEqual (range.interval, 1.0f))
                // return juce::String (int (uv))", truncating.
                let number = if interval == 1.0 {
                    format!("{}", value as i32)
                } else {
                    format_number(value)
                };
                with_unit(&number, unit)
            }
            Text::OnOff => if value > 0.0 { "On" } else { "Off" }.to_string(),
            Text::Frequency => frequency(value),
            Text::NoteFrequency => frequency(midi_note_in_hertz(value)),
            Text::Slope => if value > 0.5 { "24 dB" } else { "12 dB" }.to_string(),
            Text::Duration => DURATIONS
                .get(value.max(0.0) as usize)
                .map_or_else(String::new, |d| d.name.to_string()),
            Text::Percent => format!("{}%", round_to_int(value * 100.0)),
        }
    }
}

/// `gin::formatNumber`: three significant-ish figures — no decimals from
/// ten up, one below ten, two below one, three below a tenth — and a
/// plain "0" for zero.
#[must_use]
pub fn format_number(value: f32) -> String {
    if approximately_equal(value, 0.0) {
        return "0".to_string();
    }
    // A decimal for each of ten, one and a tenth the value is below.
    let magnitude = value.abs();
    let decimals = [10.0, 1.0, 0.1]
        .iter()
        .filter(|&&below| magnitude < below)
        .count();
    if decimals == 0 {
        format!("{}", round_to_int(value))
    } else {
        format!("{value:.decimals$}")
    }
}

/// The frequency text both `freqTextFunction`s share: kHz to one decimal
/// from a thousand up, whole Hz below.
fn frequency(hertz: f32) -> String {
    if hertz >= 1000.0 {
        format!("{:.1} kHz", hertz / 1000.0)
    } else {
        format!("{} Hz", round_to_int(hertz))
    }
}

/// `juce::roundToInt`: to nearest, ties to even, as the FPU does it.
fn round_to_int(value: f32) -> i32 {
    f64::from(value).round_ties_even() as i32
}

fn with_unit(number: &str, unit: &str) -> String {
    if unit.is_empty() {
        number.to_string()
    } else if unit.starts_with(char::is_alphanumeric) {
        format!("{number} {unit}")
    } else {
        format!("{number}{unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gin::param::midi_note_from_hertz;
    use std::fmt::Write as _;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// Gin's number format at each of its four precisions and their
    /// edges, and each text function at the values its knob reaches.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn values_print_as_the_originals_print_them() {
        let mut out = String::new();
        for value in [
            0.0, 0.000_001, 0.05, -0.05, 0.1, 0.5, 0.999, 1.0, 9.99, 9.96, 10.0, 10.5, 11.5, -30.0,
            120.0, 48_000.0,
        ] {
            let _ = writeln!(out, "formatNumber({value}) = {}", format_number(value));
        }
        let cases = [
            (Text::Number, 16.0, "", 0.0),
            (Text::Number, 2.5, "ms", 0.0),
            (Text::Number, -10.0, "dB", 0.0),
            (Text::Number, 50.0, "%", 0.0),
            (Text::Number, 1.7, "", 1.0),
            (Text::Number, -1.7, "", 1.0),
            (Text::OnOff, 0.0, "", 1.0),
            (Text::OnOff, 1.0, "", 1.0),
            (Text::Frequency, 500.0, "Hz", 0.0),
            (Text::Frequency, 999.4, "Hz", 0.0),
            (Text::Frequency, 999.6, "Hz", 0.0),
            (Text::Frequency, 48_000.0, "Hz", 0.0),
            (Text::NoteFrequency, 69.0, "", 0.0),
            (Text::NoteFrequency, midi_note_from_hertz(20.0), "", 0.0),
            (Text::NoteFrequency, midi_note_from_hertz(20_000.0), "", 0.0),
            (Text::Slope, 0.0, "", 1.0),
            (Text::Slope, 1.0, "", 1.0),
            (Text::Duration, 0.0, "", 1.0),
            (Text::Duration, 13.0, "", 1.0),
            (Text::Duration, 32.0, "", 1.0),
            (Text::Duration, 33.0, "", 1.0),
            (Text::Duration, -1.0, "", 1.0),
            (Text::Percent, 0.305, "%", 0.0),
            (Text::Percent, 0.306, "%", 0.0),
            (Text::Percent, -1.0, "%", 0.0),
        ];
        for (text, value, unit, interval) in cases {
            let _ = writeln!(
                out,
                "{text:?}({value} {unit:?}) = {}",
                text.show(value, unit, interval)
            );
        }
        expect_test::expect![[r#"
            formatNumber(0) = 0
            formatNumber(0.000001) = 0.000
            formatNumber(0.05) = 0.050
            formatNumber(-0.05) = -0.050
            formatNumber(0.1) = 0.10
            formatNumber(0.5) = 0.50
            formatNumber(0.999) = 1.00
            formatNumber(1) = 1.0
            formatNumber(9.99) = 10.0
            formatNumber(9.96) = 10.0
            formatNumber(10) = 10
            formatNumber(10.5) = 10
            formatNumber(11.5) = 12
            formatNumber(-30) = -30
            formatNumber(120) = 120
            formatNumber(48000) = 48000
            Number(16 "") = 16
            Number(2.5 "ms") = 2.5 ms
            Number(-10 "dB") = -10 dB
            Number(50 "%") = 50%
            Number(1.7 "") = 1
            Number(-1.7 "") = -1
            OnOff(0 "") = Off
            OnOff(1 "") = On
            Frequency(500 "Hz") = 500 Hz
            Frequency(999.4 "Hz") = 999 Hz
            Frequency(999.6 "Hz") = 1000 Hz
            Frequency(48000 "Hz") = 48.0 kHz
            NoteFrequency(69 "") = 440 Hz
            NoteFrequency(15.48682 "") = 20 Hz
            NoteFrequency(135.07623 "") = 20.0 kHz
            Slope(0 "") = 12 dB
            Slope(1 "") = 24 dB
            Duration(0 "") = 1/64t
            Duration(13 "") = 1/4
            Duration(32 "") = 32 bars
            Duration(33 "") = 
            Duration(-1 "") = 1/64t
            Percent(0.305 "%") = 30%
            Percent(0.306 "%") = 31%
            Percent(-1 "%") = -100%
        "#]]
        .assert_eq(&out);
    }
}
