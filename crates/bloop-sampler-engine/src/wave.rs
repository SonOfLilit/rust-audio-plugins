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
//! It has no UI dependency of any kind, so the app and the manual's gold
//! tables share the arithmetic and a waveform cannot mean two different
//! things in them.

/// Horizontal resolution a waveform is reduced to before it reaches the
/// screen. Columns rather than pixels, so the reduction is deterministic and
/// snapshot-able regardless of the device's screen width.
pub const WAVE_COLUMNS: usize = 96;

/// The most frames a view carries its samples for: three a point across
/// the widest display there is. A view spanning more is drawn as columns
/// whatever the screen; one spanning this many or fewer carries its
/// samples, and the screen decides by its own width whether it is close
/// enough to draw as a line through them.
pub const MOST_SAMPLES: usize = 2048;

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
    /// Per column, its loudness: the root of the mean of its squares.
    pub rms: Vec<f32>,
    /// The view's own samples, one a frame, when it spans at most
    /// [`MOST_SAMPLES`] frames — zoomed in far enough to draw a line
    /// through them. Empty otherwise. Each is the frame's loudest channel,
    /// as the peaks count every channel: a stereo take with one side quiet
    /// is as tall zoomed in as it is zoomed out.
    pub samples: Vec<f32>,
    /// The portion of the whole sample this view spans, as fractions in
    /// 0.0..=1.0. Zooming in narrows it.
    pub view: (f32, f32),
    /// Edit points drawn over the wave, positioned in view space.
    pub marks: Vec<WaveMark>,
    /// On an overview, the part another display shows zoomed in, in view
    /// space: drawn as a frame. `None` on the zoomed view itself, and
    /// where it shows the whole sample.
    pub window: Option<(f32, f32)>,
    /// A span picked out, in view space: on an overview of a sliced
    /// sample, the focused Slice.
    pub focus: Option<(f32, f32)>,
    /// The play range, in view space and not clamped to it — a start
    /// left of the view is below 0. What lies outside it is drawn greyed
    /// out. `None` where there is no play range to show.
    pub play: Option<(f32, f32)>,
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
    /// Where the Sample Editor's selection — what an audio function acts
    /// on — begins.
    SelectionStart,
    /// Where it ends.
    SelectionEnd,
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
            Self::SelectionStart => '(',
            Self::SelectionEnd => ')',
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
        let mut rms = Vec::with_capacity(columns);
        for column in 0..columns {
            let start = from + span * column / columns;
            let end = (from + span * (column + 1) / columns).max(start + 1);

            let mut low = f32::MAX;
            let mut high = f32::MIN;
            let (mut squares, mut count) = (0.0f64, 0usize);
            for channel in channels.clone() {
                for value in channel.get(start..end.min(channel.len())).unwrap_or(&[]) {
                    low = low.min(*value);
                    high = high.max(*value);
                    squares += f64::from(*value) * f64::from(*value);
                    count += 1;
                }
            }
            // A column past the end of every channel reads as silence.
            if low > high {
                low = 0.0;
                high = 0.0;
            }
            peaks.push((low, high));
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a level is a few digits; f32 is the screen's precision"
            )]
            rms.push(if count == 0 {
                0.0
            } else {
                libm::sqrt(squares / count as f64) as f32
            });
        }

        // Zoomed in, the samples themselves: each frame, its loudest channel.
        let samples = if span <= MOST_SAMPLES {
            (from..to)
                .map(|frame| {
                    channels
                        .clone()
                        .map(|channel| channel.get(frame).copied().unwrap_or(0.0))
                        .fold(
                            0.0f32,
                            |loudest, v| {
                                if v.abs() > loudest.abs() {
                                    v
                                } else {
                                    loudest
                                }
                            },
                        )
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            peaks,
            rms,
            samples,
            view: (
                from as f32 / total_frames as f32,
                to as f32 / total_frames as f32,
            ),
            marks: Vec::new(),
            window: None,
            focus: None,
            play: None,
        }
    }

    /// Set the play range from where it starts and ends in the whole
    /// sample, as fractions: [`WaveView::play`], in this view's space.
    pub fn set_play(&mut self, from: f32, to: f32) {
        let (view_from, view_to) = self.view;
        let width = view_to - view_from;
        if width <= 0.0 {
            return;
        }
        let at = |position: f32| (position - view_from) / width;
        self.play = Some((at(from.min(to)), at(from.max(to))));
    }

    /// The parts of this view outside the play range, as fractions of
    /// the view: what is greyed out. None without a play range, or when
    /// it covers the whole view; all of it when it lies off to a side.
    pub fn outside_play(&self) -> Vec<(f32, f32)> {
        let Some((from, to)) = self.play else {
            return Vec::new();
        };
        let (from, to) = (from.clamp(0.0, 1.0), to.clamp(0.0, 1.0));
        let mut spans = Vec::new();
        if from > 0.0 {
            spans.push((0.0, from));
        }
        if to < 1.0 && to >= from {
            spans.push((to, 1.0));
        }
        spans
    }

    /// A span of the whole sample, `from..to` in fractions, as fractions
    /// of this view — what [`WaveView::window`] and [`WaveView::focus`]
    /// hold. `None` when none of it is in view: a span that only touches
    /// the view's edge is not in it, a span of no width inside it is.
    pub fn span_in_view(&self, from: f32, to: f32) -> Option<(f32, f32)> {
        let (view_from, view_to) = self.view;
        let width = view_to - view_from;
        let (from, to) = (from.min(to), from.max(to));
        if width <= 0.0 || to < view_from || from > view_to {
            return None;
        }
        let at = |position: f32| ((position - view_from) / width).clamp(0.0, 1.0);
        let (start, end) = (at(from), at(to));
        (end > start || to == from).then_some((start, end))
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
        // Audio outside the play range is greyed out: `.` for `#`.
        let outside = self.outside_play();
        let columns = self.peaks.len() as f32;

        for (column, &(low, high)) in self.peaks.iter().enumerate() {
            let centre = (column as f32 + 0.5) / columns;
            let ink = if outside
                .iter()
                .any(|(from, to)| centre >= *from && centre < *to)
            {
                '.'
            } else {
                '#'
            };
            // Row 0 is the top (+1.0), the last row is the bottom (-1.0).
            let to_row = |value: f32| {
                let scaled = half - value.clamp(-1.0, 1.0) * half;
                (scaled as usize).min(rows - 1)
            };
            let (top, bottom) = (to_row(high), to_row(low));
            for row in grid.iter_mut().take(bottom + 1).skip(top) {
                row[column] = ink;
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

        let mut lines: Vec<String> = grid
            .into_iter()
            .map(|row| row.into_iter().collect::<String>().trim_end().to_string())
            .collect();
        // Under the wave, the spans it picks out: `=` under the window
        // another display shows, `^` under the focus.
        if self.window.is_some() || self.focus.is_some() {
            let columns = self.peaks.len();
            let column_of = |at: f32| {
                ((at.clamp(0.0, 1.0) * columns as f32) as usize).min(columns.saturating_sub(1))
            };
            let mut under = vec![' '; columns];
            for (span, glyph) in [(self.window, '='), (self.focus, '^')] {
                if let Some((from, to)) = span {
                    let first = column_of(from);
                    let last = column_of(to - f32::EPSILON).max(first);
                    for cell in &mut under[first..=last] {
                        *cell = glyph;
                    }
                }
            }
            lines.push(under.into_iter().collect::<String>().trim_end().to_string());
        }
        lines
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

    /// A span of the whole sample as fractions of a view of its middle
    /// half: one inside it scales, one across an edge is cut at the edge,
    /// one backwards comes back low first, one outside it or only
    /// touching its edge is not in view, and one of no width inside it is.
    /// A view of no width has nothing in it.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_span_is_placed_in_the_view() {
        let view = WaveView {
            view: (0.25, 0.75),
            ..WaveView::default()
        };
        let empty = WaveView {
            view: (0.5, 0.5),
            ..WaveView::default()
        };
        let out: Vec<String> = [
            ("inside", view.span_in_view(0.375, 0.5)),
            ("whole", view.span_in_view(0.0, 1.0)),
            ("across the start", view.span_in_view(0.0, 0.5)),
            ("across the end", view.span_in_view(0.5, 1.0)),
            ("backwards", view.span_in_view(0.5, 0.375)),
            ("before", view.span_in_view(0.0, 0.125)),
            ("touching the start", view.span_in_view(0.0, 0.25)),
            ("touching the end", view.span_in_view(0.75, 1.0)),
            ("no width inside", view.span_in_view(0.5, 0.5)),
            ("in a view of no width", empty.span_in_view(0.0, 1.0)),
        ]
        .iter()
        .map(|(name, span)| format!("{name}: {span:?}"))
        .collect();
        expect![[r#"
            inside: Some((0.25, 0.5))
            whole: Some((0.0, 1.0))
            across the start: Some((0.0, 0.5))
            across the end: Some((0.5, 1.0))
            backwards: Some((0.25, 0.5))
            before: None
            touching the start: None
            touching the end: None
            no width inside: Some((0.5, 0.5))
            in a view of no width: None"#]]
        .assert_eq(&out.join("\n"));
    }

    /// The play range greys out what is outside it, `.` for `#`: both
    /// ends trimmed, a view zoomed inside it (nothing greyed), a view
    /// wholly before its start (all of it), a range given backwards, and
    /// no play range at all.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn outside_the_play_range_is_greyed_out() {
        let audio = vec![0.5f32; 160];
        let whole = WaveView::reduce([audio.as_slice()].into_iter(), (0, 160), 160, 16);
        let zoomed = WaveView::reduce([audio.as_slice()].into_iter(), (40, 80), 160, 16);
        let show = |name: &str, wave: &WaveView, from: f32, to: f32| {
            let mut wave = wave.clone();
            wave.set_play(from, to);
            format!(
                "{name}: play {:?}, outside {:?}\n{}",
                wave.play,
                wave.outside_play(),
                wave.to_text(2).join("\n")
            )
        };
        let none = format!(
            "no play range: outside {:?}\n{}",
            whole.outside_play(),
            whole.to_text(2).join("\n")
        );
        expect![[r#"
            trimmed: play Some((0.25, 0.75)), outside [(0.0, 0.25), (0.75, 1.0)]
            ....########....

            zoomed inside: play Some((-1.0, 3.0)), outside []
            ################

            zoomed before the start: play Some((2.0, 3.0)), outside [(0.0, 1.0)]
            ................

            zoomed across the end: play Some((-1.0, 0.75)), outside [(0.75, 1.0)]
            ############....

            backwards: play Some((0.25, 0.75)), outside [(0.0, 0.25), (0.75, 1.0)]
            ....########....

            no play range: outside []
            ################
        "#]]
        .assert_eq(
            &[
                show("trimmed", &whole, 0.25, 0.75),
                show("zoomed inside", &zoomed, 0.0, 1.0),
                show("zoomed before the start", &zoomed, 0.75, 1.0),
                show("zoomed across the end", &zoomed, 0.0, 0.4375),
                show("backwards", &whole, 0.75, 0.25),
                none,
            ]
            .join("\n"),
        );
    }

    /// Under a wave with a window or a focus, one more line: `=` under the
    /// window, `^` under the focus over it. A window to the very end
    /// reaches the last column, a focus of no width is one column, and a
    /// wave with neither has no line more.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_window_and_the_focus_are_underlined() {
        let audio = vec![0.5f32; 160];
        let wave = WaveView::reduce([audio.as_slice()].into_iter(), (0, 160), 160, 16);
        let with = |window: Option<(f32, f32)>, focus: Option<(f32, f32)>| {
            WaveView {
                window,
                focus,
                ..wave.clone()
            }
            .to_text(2)
            .join("\n")
        };
        expect![[r#"
            neither:
            ################

            window to the end, a focus in it:
            ################

                    ==^^====
            a focus of no width:
            ################

                ^"#]]
        .assert_eq(
            &[
                format!("neither:\n{}", with(None, None)),
                format!(
                    "window to the end, a focus in it:\n{}",
                    with(Some((0.5, 1.0)), Some((0.625, 0.75)))
                ),
                format!("a focus of no width:\n{}", with(None, Some((0.25, 0.25)))),
            ]
            .join("\n"),
        );
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

    /// Each column carries its loudness beside its peaks, every channel
    /// counted: a full-scale square wave is as loud as it is high, a sine
    /// about 0.71 of it, and silence nothing. A view of at most
    /// `MOST_SAMPLES` frames carries the frames themselves, each its
    /// loudest channel's; a wider one carries none.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_view_carries_its_loudness_and_zoomed_in_its_samples() {
        let square: Vec<f32> = (0..960)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let silence = vec![0.0f32; 960];
        let audio = sine(960);
        let rms = |view: &WaveView| {
            view.rms
                .iter()
                .map(|r| format!("{r:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let whole = WaveView::reduce([audio.as_slice()].into_iter(), (0, 960), 960, 4);
        let squared = WaveView::reduce([square.as_slice()].into_iter(), (0, 960), 960, 2);
        let mixed = WaveView::reduce(
            [square.as_slice(), silence.as_slice()].into_iter(),
            (0, 960),
            960,
            2,
        );
        let near = WaveView::reduce(
            [square.as_slice(), silence.as_slice()].into_iter(),
            (100, 106),
            960,
            3,
        );
        let long = sine(MOST_SAMPLES + 1);
        let edge = WaveView::reduce(
            [long.as_slice()].into_iter(),
            (0, MOST_SAMPLES),
            long.len(),
            4,
        );
        let past = WaveView::reduce(
            [long.as_slice()].into_iter(),
            (0, MOST_SAMPLES + 1),
            long.len(),
            4,
        );
        expect![[r#"
            sine rms 0.71 0.71 0.71 0.71, 960 samples
            square rms 1.00 1.00
            square and silence rms 0.71 0.71
            six frames, the louder channel's: [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
            2048 frames: 2048 samples; one more: 0"#]]
        .assert_eq(&format!(
            "sine rms {}, {} samples\nsquare rms {}\nsquare and silence rms {}\n\
             six frames, the louder channel's: {:?}\n{MOST_SAMPLES} frames: {} samples; one more: {}",
            rms(&whole),
            whole.samples.len(),
            rms(&squared),
            rms(&mixed),
            near.samples,
            edge.samples.len(),
            past.samples.len(),
        ));
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
