//! Operations that make a new buffer.
//!
//! Deliberately few. Almost everything a pad can do to its sound — level,
//! direction, the shape of its ends — is a playback parameter, which costs no
//! memory, applies to one pad without disturbing the fifteen others sharing
//! the same audio, and is taken back by the ordinary undo journal.
//!
//! What is left are the two things a parameter cannot express: throwing audio
//! away, and the reversed copy the sampler plays while its Reverse parameter
//! is on (the ported voice only reads forwards).
//!
//! Both return a fresh [`SampleBuffer`] rather than mutating in place,
//! because a buffer is shared.

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

/// The range, played backwards.
pub fn reverse(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    let mut out = trim(sample, range);
    out.map_channel_slices(<[f32]>::reverse);
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

    /// Reversing plays the range backwards.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn reversing_plays_the_range_backwards() {
        let sample = ramp(6, 1.0);
        expect!["6 frames: 1.000 0.800 0.600 0.400 0.200 0.000"]
            .assert_eq(&show(&reverse(&sample, (0, 6))));
    }

    /// Editing a pad must not touch the audio other pads share with it.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_edit_leaves_the_original_alone() {
        let sample = ramp(6, 1.0);
        let before = show(&sample);
        let _ = reverse(&sample, (0, 6));
        let _ = trim(&sample, (2, 4));
        expect![[r#"
            before: 6 frames: 0.000 0.200 0.400 0.600 0.800 1.000
            after:  6 frames: 0.000 0.200 0.400 0.600 0.800 1.000"#]]
        .assert_eq(&format!("before: {before}\nafter:  {}", show(&sample)));
    }
}
