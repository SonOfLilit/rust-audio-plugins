//! Destructive edits: operations that make a new buffer.
//!
//! Every one returns a fresh [`SampleBuffer`] rather than mutating in place,
//! because a buffer is shared. Sixteen pads chopped from one break all point
//! at the same audio, so normalising "this pad" must not quietly change the
//! other fifteen — the edited pad gets its own copy and the rest are left
//! alone.
//!
//! They all take a frame range, and all of them return only that range. An
//! edit is therefore also a trim: the pad's bounds reset to the whole of its
//! new, shorter buffer, which is what makes a chopped-then-edited slice cheap
//! rather than sixteen copies of a whole break.

use crate::SampleBuffer;

/// The range of `sample` as its own buffer.
///
/// The plain trim, and the shared first step of every other edit here.
pub fn trim(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    let frames = sample.num_samples();
    let from = range.0.min(range.1).min(frames);
    let to = range.1.max(range.0).min(frames);
    let channels = (0..sample.num_channels())
        .map(|channel| sample.channel(channel)[from..to].to_vec())
        .collect();
    SampleBuffer::new(channels, sample.sample_rate())
}

/// The range, scaled so its loudest sample sits at full scale.
///
/// One gain for every channel, so a sample panned to one side keeps its
/// position instead of being centred by the edit. Silence is returned
/// unchanged — there is nothing to normalise, and scaling it would only
/// amplify whatever noise floor it has.
pub fn normalize(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    let mut out = trim(sample, range);
    let peak = (0..out.num_channels())
        .flat_map(|channel| out.channel(channel).iter())
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    if peak <= f32::EPSILON || peak >= 1.0 {
        return out;
    }
    let gain = 1.0 / peak;
    out.map_samples(|value| value * gain);
    out
}

/// The range, played backwards.
pub fn reverse(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    let mut out = trim(sample, range);
    out.map_channel_slices(<[f32]>::reverse);
    out
}

/// The range with a linear fade in over its first `frames` and a fade out
/// over its last.
///
/// Both fades at once, because that is nearly always what a slice wants: a
/// chop lands mid-waveform at each end, and a couple of milliseconds either
/// side is the difference between a clean hit and a click. Passing zero
/// leaves that end alone.
pub fn fade(
    sample: &SampleBuffer,
    range: (usize, usize),
    fade_in: usize,
    fade_out: usize,
) -> SampleBuffer {
    let mut out = trim(sample, range);
    let length = out.num_samples();
    if length == 0 {
        return out;
    }
    // Overlapping fades would fight each other; share the room instead.
    let (fade_in, fade_out) = if fade_in + fade_out > length {
        let half = length / 2;
        (fade_in.min(half), fade_out.min(length - half))
    } else {
        (fade_in, fade_out)
    };

    out.map_channel_slices(|samples| {
        for (i, value) in samples.iter_mut().enumerate().take(fade_in) {
            *value *= i as f32 / fade_in as f32;
        }
        for i in 0..fade_out {
            let index = length - 1 - i;
            samples[index] *= i as f32 / fade_out as f32;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    /// A ramp whose value reads back as its position, so an edit's effect is
    /// visible in the numbers.
    fn ramp(frames: usize, scale: f32) -> SampleBuffer {
        let data = (0..frames)
            .map(|i| (i as f32 / (frames - 1) as f32) * scale)
            .collect();
        SampleBuffer::new(vec![data], 48_000)
    }

    fn show(sample: &SampleBuffer) -> String {
        let mut out = format!("{} frames:", sample.num_samples());
        for i in 0..sample.num_samples().min(10) {
            out.push_str(&format!(" {:.3}", sample.sample(0, i)));
        }
        out
    }

    /// A trim keeps only its range, and the result starts at the range's
    /// first frame.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_trim_keeps_only_its_range() {
        let sample = ramp(11, 1.0);
        expect!["4 frames: 0.400 0.500 0.600 0.700"].assert_eq(&show(&trim(&sample, (4, 8))));
    }

    /// Normalising lifts the loudest sample to full scale and everything
    /// else with it.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn normalising_lifts_the_peak_to_full_scale() {
        let sample = ramp(6, 0.25);
        expect!["6 frames: 0.000 0.200 0.400 0.600 0.800 1.000"]
            .assert_eq(&show(&normalize(&sample, (0, 6))));
    }

    /// A stereo sample keeps its balance: one gain for both channels, so a
    /// quiet side stays quiet relative to the loud one.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn normalising_keeps_the_stereo_balance() {
        let left = vec![0.4f32, 0.2, 0.4];
        let right = vec![0.1f32, 0.05, 0.1];
        let sample = SampleBuffer::new(vec![left, right], 48_000);
        let out = normalize(&sample, (0, 3));
        expect![[r#"
            L [1.0, 0.5, 1.0]
            R [0.25, 0.125, 0.25]"#]]
        .assert_eq(&format!(
            "L {:?}\nR {:?}",
            out.channel(0)
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>(),
            out.channel(1)
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>(),
        ));
    }

    /// Silence has no peak to lift, and already-full-scale audio has nowhere
    /// to go; both come back untouched.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn normalising_leaves_silence_and_full_scale_alone() {
        let silent = SampleBuffer::new(vec![vec![0.0; 4]], 48_000);
        let loud = SampleBuffer::new(vec![vec![0.0, 1.0, -1.0, 0.5]], 48_000);
        expect![[r#"
            silence: 4 frames: 0.000 0.000 0.000 0.000
            full scale: 4 frames: 0.000 1.000 -1.000 0.500"#]]
        .assert_eq(&format!(
            "silence: {}\nfull scale: {}",
            show(&normalize(&silent, (0, 4))),
            show(&normalize(&loud, (0, 4))),
        ));
    }

    /// Reversing plays the range backwards.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn reversing_plays_the_range_backwards() {
        let sample = ramp(6, 1.0);
        expect!["6 frames: 1.000 0.800 0.600 0.400 0.200 0.000"]
            .assert_eq(&show(&reverse(&sample, (0, 6))));
    }

    /// A fade ramps in from silence and back out to it, leaving the middle
    /// alone.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_fade_ramps_both_ends() {
        let sample = SampleBuffer::new(vec![vec![1.0f32; 10]], 48_000);
        expect!["10 frames: 0.000 0.333 0.667 1.000 1.000 1.000 1.000 0.667 0.333 0.000"]
            .assert_eq(&show(&fade(&sample, (0, 10), 3, 3)));
    }

    /// Fades longer than the sample share the room rather than fighting over
    /// it, so the result still starts and ends at silence.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn overlapping_fades_share_the_room() {
        let sample = SampleBuffer::new(vec![vec![1.0f32; 6]], 48_000);
        expect!["6 frames: 0.000 0.333 0.667 0.667 0.333 0.000"].assert_eq(&show(&fade(
            &sample,
            (0, 6),
            10,
            10,
        )));
    }

    /// Editing a pad must not touch the audio other pads share with it.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_edit_leaves_the_original_alone() {
        let sample = ramp(6, 1.0);
        let before = show(&sample);
        let _ = reverse(&sample, (0, 6));
        let _ = normalize(&sample, (0, 6));
        expect![[r#"
            before: 6 frames: 0.000 0.200 0.400 0.600 0.800 1.000
            after:  6 frames: 0.000 0.200 0.400 0.600 0.800 1.000"#]]
        .assert_eq(&format!("before: {before}\nafter:  {}", show(&sample)));
    }
}
