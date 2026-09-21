//! A waveform reduced for drawing.
//!
//! The reduction happens where the audio lives — here, next to
//! [`SampleBuffer`](crate::SampleBuffer) — so nothing downstream ever sees
//! the sample itself and a ten-minute import costs the same to draw as a
//! kick.
//!
//! It is described rather than drawn. A description can be printed, which
//! is what lets a demo snapshot show exactly what was on screen, and it
//! keeps both apps' painters dumb: they turn min/max pairs into rectangles
//! and know nothing about audio.
//!
//! Lifted out of `bloop-menu-framework`, where it was written for the
//! legacy sampler page, when the spec-driven app needed the same picture.
//! It has no UI dependency of any kind, so the two apps share the
//! arithmetic and a waveform cannot mean two different things in them.

/// Horizontal resolution a waveform is reduced to before it reaches the
/// screen. Columns rather than pixels, so the reduction is deterministic and
/// snapshot-able regardless of the device's screen width.
pub const WAVE_COLUMNS: usize = 96;

/// A waveform reduced for display: one min/max pair per column, plus the
/// marks drawn over it.
///
/// The reduction happens where the audio lives, so the screen never sees the
/// sample itself and a ten-minute import costs the same to draw as a kick.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct WaveView {
    /// Per column, the lowest and highest sample in it, each in -1.0..=1.0.
    /// Empty means "no audio loaded".
    pub peaks: Vec<(f32, f32)>,
    /// The portion of the whole sample this view spans, as fractions in
    /// 0.0..=1.0. Zooming in narrows it.
    pub view: (f32, f32),
    /// Edit points drawn over the wave, positioned in view space.
    pub marks: Vec<WaveMark>,
}

/// A labelled position on the waveform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveMark {
    /// Where it sits, as a fraction of the visible view (0.0..=1.0). Marks
    /// outside that range are clipped by the renderer.
    pub at: f32,
    pub kind: MarkKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkKind {
    /// Where playback begins.
    Start,
    /// Where playback ends.
    End,
    /// A slice boundary — the divisions a chopped sample is spread across.
    Slice,
    /// Where a loop returns to.
    LoopStart,
    /// Where a loop returns from.
    LoopEnd,
    /// A voice's live position.
    Playhead,
}

impl MarkKind {
    /// The single character that stands for this mark in a text rendering.
    pub fn glyph(self) -> char {
        match self {
            Self::Start => '[',
            Self::End => ']',
            Self::Slice => '|',
            Self::LoopStart => '{',
            Self::LoopEnd => '}',
            Self::Playhead => '>',
        }
    }
}

impl WaveView {
    /// Reduce `samples` to `columns` min/max pairs over the frame range
    /// `range`, mixing every channel together.
    ///
    /// Min/max rather than averaging: a kick's transient survives being drawn
    /// 96 columns wide, where an average would flatten it into a smudge.
    pub fn reduce<'a>(
        channels: impl Iterator<Item = &'a [f32]> + Clone,
        range: (usize, usize),
        total_frames: usize,
        columns: usize,
    ) -> Self {
        let (from, to) = (range.0.min(range.1), range.1.max(range.0));
        let span = to.saturating_sub(from);
        if span == 0 || columns == 0 || total_frames == 0 {
            return Self::default();
        }

        let mut peaks = Vec::with_capacity(columns);
        for column in 0..columns {
            let start = from + span * column / columns;
            let end = (from + span * (column + 1) / columns).max(start + 1);

            let mut low = f32::MAX;
            let mut high = f32::MIN;
            for channel in channels.clone() {
                for value in channel.get(start..end.min(channel.len())).unwrap_or(&[]) {
                    low = low.min(*value);
                    high = high.max(*value);
                }
            }
            // A column past the end of every channel reads as silence.
            if low > high {
                low = 0.0;
                high = 0.0;
            }
            peaks.push((low, high));
        }

        Self {
            peaks,
            view: (
                from as f32 / total_frames as f32,
                to as f32 / total_frames as f32,
            ),
            marks: Vec::new(),
        }
    }

    /// Place a mark at an absolute frame, dropping it if it falls outside the
    /// visible view.
    pub fn mark_frame(&mut self, frame: usize, total_frames: usize, kind: MarkKind) {
        if total_frames == 0 {
            return;
        }
        let position = frame as f32 / total_frames as f32;
        let (from, to) = self.view;
        if position < from || position > to || to <= from {
            return;
        }
        self.marks.push(WaveMark {
            at: (position - from) / (to - from),
            kind,
        });
    }

    /// Render as text: `rows` lines of `peaks.len()` characters, with the
    /// marks overlaid.
    ///
    /// This is what demo snapshots capture. It is a faithful reduction of what
    /// the screen draws — same columns, same marks — so a waveform regression
    /// shows up as a diff in the manual rather than only in a pixel hash.
    ///
    /// Rows are trimmed on the right: trailing spaces in a snapshot are the
    /// first thing an editor or a commit hook eats, and a snapshot that breaks
    /// when someone strips whitespace is worse than a ragged one.
    pub fn to_text(&self, rows: usize) -> Vec<String> {
        if self.peaks.is_empty() || rows == 0 {
            return vec![String::new(); rows];
        }

        let mut grid = vec![vec![' '; self.peaks.len()]; rows];
        let half = rows as f32 / 2.0;

        for (column, &(low, high)) in self.peaks.iter().enumerate() {
            // Row 0 is the top (+1.0), the last row is the bottom (-1.0).
            let to_row = |value: f32| {
                let scaled = half - value.clamp(-1.0, 1.0) * half;
                (scaled as usize).min(rows - 1)
            };
            let (top, bottom) = (to_row(high), to_row(low));
            for row in grid.iter_mut().take(bottom + 1).skip(top) {
                row[column] = '#';
            }
        }

        for mark in &self.marks {
            let column = ((mark.at.clamp(0.0, 1.0) * (self.peaks.len() - 1) as f32).round()
                as usize)
                .min(self.peaks.len() - 1);
            for row in grid.iter_mut() {
                row[column] = mark.kind.glyph();
            }
        }

        grid.into_iter()
            .map(|row| row.into_iter().collect::<String>().trim_end().to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    fn sine(frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (i as f32 / frames as f32 * core::f32::consts::TAU).sin())
            .collect()
    }

    /// A full-scale sine reduces to a symmetric band that fills the height at
    /// its peaks and pinches to the centre line at its zero crossings.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_sine_reduces_to_a_symmetric_band() {
        let audio = sine(960);
        let view = WaveView::reduce([audio.as_slice()].into_iter(), (0, 960), 960, 48);
        expect![[r#"
                  ############
                ###          ###
              ###              ###
            ###                  ###
            #                      ##                      #
                                    ###                  ###
                                      ###              ###
                                        ###          ###
                                          ############"#]]
        .assert_eq(&view.to_text(9).join("\n"));
    }

    /// Min/max keeps a transient that averaging would flatten: one loud frame
    /// in a quiet second still reaches the top of the screen.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_single_loud_frame_survives_the_reduction() {
        let mut audio = vec![0.02f32; 480];
        audio[200] = 1.0;
        audio[201] = -1.0;
        let view = WaveView::reduce([audio.as_slice()].into_iter(), (0, 480), 480, 24);
        expect![[r#"
                      #
                      #
                      #
            ########################
                      #
                      #
                      #"#]]
        .assert_eq(&view.to_text(7).join("\n"));
    }

    /// Zooming in reduces a sub-range, and reports which part of the whole it
    /// is showing so the UI can draw a scrollbar.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn zooming_in_reduces_only_the_visible_range() {
        let audio = sine(960);
        let view = WaveView::reduce([audio.as_slice()].into_iter(), (240, 480), 960, 24);
        expect![[r#"
            showing 0.25..0.50 of the sample
            ############
                       #######
                             #####
                                 ###


        "#]]
        .assert_eq(&format!(
            "showing {:.2}..{:.2} of the sample\n{}",
            view.view.0,
            view.view.1,
            view.to_text(7).join("\n")
        ));
    }

    /// Marks land on the column their frame falls in, and ones scrolled out of
    /// view are dropped rather than clamped to the edge.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn marks_land_on_their_frame_and_clip_when_scrolled_away() {
        let audio = vec![0.5f32; 960];
        let mut view = WaveView::reduce([audio.as_slice()].into_iter(), (240, 720), 960, 24);
        view.mark_frame(240, 960, MarkKind::Start);
        view.mark_frame(480, 960, MarkKind::Slice);
        view.mark_frame(720, 960, MarkKind::End);
        // Outside the visible range.
        view.mark_frame(0, 960, MarkKind::LoopStart);
        view.mark_frame(900, 960, MarkKind::LoopEnd);
        expect![[r#"
            3 marks visible
            [           |          ]
            [###########|##########]
            [           |          ]
            [           |          ]
            [           |          ]"#]]
        .assert_eq(&format!(
            "{} marks visible\n{}",
            view.marks.len(),
            view.to_text(5).join("\n")
        ));
    }

    /// Stereo mixes both channels into one band, so a slice that is loud on
    /// only one side still shows.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn stereo_channels_are_drawn_together() {
        let left = vec![0.0f32; 240];
        let right: Vec<f32> = (0..240).map(|i| if i > 120 { 0.9 } else { 0.0 }).collect();
        let view = WaveView::reduce(
            [left.as_slice(), right.as_slice()].into_iter(),
            (0, 240),
            240,
            16,
        );
        expect![[r#"
                    ########
                    ########
            ################

        "#]]
        .assert_eq(&view.to_text(5).join("\n"));
    }

    /// An empty range draws nothing rather than dividing by zero.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_empty_range_draws_nothing() {
        let view = WaveView::reduce([[].as_slice()].into_iter(), (0, 0), 0, 24);
        expect![[r#"peaks 0 rows ["", "", ""]"#]].assert_eq(&format!(
            "peaks {} rows {:?}",
            view.peaks.len(),
            view.to_text(3)
        ));
    }
}
