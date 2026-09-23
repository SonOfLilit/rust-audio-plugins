//! Operations that make a new buffer.
//!
//! Deliberately few. Almost everything a pad can do to its sound — level,
//! direction, the shape of its ends — is a playback parameter, which costs no
//! memory, applies to one pad without disturbing the fifteen others sharing
//! the same audio, and is taken back by the ordinary undo journal.
//!
//! What is left are the things a parameter cannot express: throwing audio
//! away, the reversed copy the sampler plays while its Reverse parameter is
//! on (the ported voice only reads forwards), and the Sample Editor's audio
//! functions that change part of a Sample — a fade over a selection, a
//! silence, a cut, a paste.
//!
//! Every one returns a fresh [`SampleBuffer`] rather than mutating in place,
//! because a buffer is shared. None of them keeps what it replaced: an edit
//! is undone by rebuilding from where the audio came from and applying the
//! edits that remain, not by holding a history of buffers.
//!
//! Ranges are frames, `from..to`, clamped to the buffer.

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

/// A range clamped into `0..=frames`, in order.
fn clamp(sample: &SampleBuffer, range: (usize, usize)) -> (usize, usize) {
    let frames = sample.num_samples();
    (
        range.0.min(range.1).min(frames),
        range.1.max(range.0).min(frames),
    )
}

/// The whole buffer, with `edit` applied to the range of every channel.
fn within(
    sample: &SampleBuffer,
    range: (usize, usize),
    mut edit: impl FnMut(&mut [f32]),
) -> SampleBuffer {
    let (from, to) = clamp(sample, range);
    let channels = (0..sample.num_channels())
        .map(|channel| {
            let mut data = sample.channel(channel).to_vec();
            edit(&mut data[from..to]);
            data
        })
        .collect();
    SampleBuffer::new(channels, sample.sample_rate())
}

/// The gain at each of `len` frames of a straight ramp from silence up.
#[expect(
    clippy::cast_precision_loss,
    reason = "a frame index in a ramp of a few million frames is exact enough in f32 for a gain"
)]
fn rising(at: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    at as f32 / (len - 1) as f32
}

/// FADE IN: the range rises from silence to full, straight.
pub fn fade_in(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    within(sample, range, |data| {
        let len = data.len();
        for (at, value) in data.iter_mut().enumerate() {
            *value *= rising(at, len);
        }
    })
}

/// FADE OUT: the range falls from full to silence, straight.
pub fn fade_out(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    within(sample, range, |data| {
        let len = data.len();
        for (at, value) in data.iter_mut().enumerate() {
            *value *= 1.0 - rising(at, len);
        }
    })
}

/// FIX DC: the range's average is taken out of it, so it centres on zero.
#[expect(
    clippy::cast_precision_loss,
    reason = "a mean over a few million frames is exact enough in f64"
)]
pub fn fix_dc(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    within(sample, range, |data| {
        if data.is_empty() {
            return;
        }
        let mean = data.iter().map(|v| f64::from(*v)).sum::<f64>() / data.len() as f64;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "an average of f32 samples is an f32 sample"
        )]
        let mean = mean as f32;
        for value in data.iter_mut() {
            *value -= mean;
        }
    })
}

/// SILENCE: the range becomes silence, and keeps its length.
pub fn silence(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    within(sample, range, |data| data.fill(0.0))
}

/// CUT: the range is taken out, and what follows it closes the gap.
pub fn cut(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    let (from, to) = clamp(sample, range);
    let channels = (0..sample.num_channels())
        .map(|channel| {
            let data = sample.channel(channel);
            [&data[..from], &data[to..]].concat()
        })
        .collect();
    SampleBuffer::new(channels, sample.sample_rate())
}

/// PASTE: `clip` goes in at frame `at`, and what was there moves along.
///
/// A clip with fewer channels than the Sample repeats its last one — a mono
/// clip goes into both sides of a stereo Sample — and one with more has the
/// extra dropped.
pub fn insert(sample: &SampleBuffer, at: usize, clip: &SampleBuffer) -> SampleBuffer {
    let at = at.min(sample.num_samples());
    let clip_channels = clip.num_channels().max(1);
    let channels = (0..sample.num_channels())
        .map(|channel| {
            let data = sample.channel(channel);
            let from = if clip.num_channels() == 0 {
                &[][..]
            } else {
                clip.channel(channel.min(clip_channels - 1))
            };
            [&data[..at], from, &data[at..]].concat()
        })
        .collect();
    SampleBuffer::new(channels, sample.sample_rate())
}

/// DUPLICATE: the range is copied in straight after itself.
pub fn duplicate(sample: &SampleBuffer, range: (usize, usize)) -> SampleBuffer {
    let (from, to) = clamp(sample, range);
    insert(sample, to, &trim(sample, (from, to)))
}

/// What NORMALIZE sets, in dB: the gain that brings the loudest frame of
/// the range to full scale. Silence has no peak to lift, and is 0.
pub fn normalize_db(sample: &SampleBuffer, range: (usize, usize)) -> f32 {
    let (from, to) = clamp(sample, range);
    let peak = (0..sample.num_channels())
        .flat_map(|channel| sample.channel(channel)[from..to].iter())
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    if peak <= f32::EPSILON {
        0.0
    } else {
        -20.0 * libm::log10f(peak)
    }
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

    /// The Sample Editor's functions over a selection, on a ramp whose
    /// every frame reads as its position.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_audio_functions_edit_only_their_range() {
        let sample = ramp(10, 0.9);
        let with_dc = SampleBuffer::new(vec![vec![0.5, 0.7, 0.3, 0.5, 0.5, 0.1]], 48_000);
        let clip = SampleBuffer::new(vec![vec![9.0, 9.0]], 48_000);
        let out = [
            format!("the ramp     {}", show(&sample)),
            format!("fade in 2-6  {}", show(&fade_in(&sample, (2, 6)))),
            format!("fade out 6-10 {}", show(&fade_out(&sample, (6, 10)))),
            format!("fix dc 1-4   {}", show(&fix_dc(&with_dc, (1, 4)))),
            format!("silence 3-5  {}", show(&silence(&sample, (3, 5)))),
            format!("cut 3-5      {}", show(&cut(&sample, (3, 5)))),
            format!("paste at 2   {}", show(&insert(&sample, 2, &clip))),
            format!("paste at end {}", show(&insert(&ramp(3, 1.0), 99, &clip))),
            format!("duplicate 1-3 {}", show(&duplicate(&sample, (1, 3)))),
            format!("backwards range {}", show(&silence(&sample, (5, 3)))),
            format!(
                "normalize all {:.2} dB, 0-5 {:.2} dB, silence {:.2} dB",
                normalize_db(&sample, (0, 10)),
                normalize_db(&sample, (0, 5)),
                normalize_db(&silence(&sample, (0, 10)), (0, 10))
            ),
        ];
        expect![[r#"
            the ramp     10 frames: 0.000 0.100 0.200 0.300 0.400 0.500 0.600 0.700 0.800 0.900
            fade in 2-6  10 frames: 0.000 0.100 0.000 0.100 0.267 0.500 0.600 0.700 0.800 0.900
            fade out 6-10 10 frames: 0.000 0.100 0.200 0.300 0.400 0.500 0.600 0.467 0.267 0.000
            fix dc 1-4   6 frames: 0.500 0.200 -0.200 0.000 0.500 0.100
            silence 3-5  10 frames: 0.000 0.100 0.200 0.000 0.000 0.500 0.600 0.700 0.800 0.900
            cut 3-5      8 frames: 0.000 0.100 0.200 0.500 0.600 0.700 0.800 0.900
            paste at 2   12 frames: 0.000 0.100 9.000 9.000 0.200 0.300 0.400 0.500 0.600 0.700
            paste at end 5 frames: 0.000 0.500 1.000 9.000 9.000
            duplicate 1-3 12 frames: 0.000 0.100 0.200 0.100 0.200 0.300 0.400 0.500 0.600 0.700
            backwards range 10 frames: 0.000 0.100 0.200 0.000 0.000 0.500 0.600 0.700 0.800 0.900
            normalize all 0.92 dB, 0-5 7.96 dB, silence 0.00 dB"#]]
        .assert_eq(&out.join("\n"));
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
