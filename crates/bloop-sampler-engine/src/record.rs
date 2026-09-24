//! Capturing audio into a buffer.
//!
//! The recorder is deliberately platform-free: it takes frames and gives back
//! a [`SampleBuffer`]. Where the frames come from — a microphone, a line in,
//! a test — is the platform's problem, which is what lets the whole state
//! machine be tested without a sound card.
//!
//! It runs on the UI thread. The input callback hands frames across a ring
//! buffer and the UI drains them, so nothing here needs to be allocation-free
//! — but the capacity is fixed up front anyway, because a recording that
//! reallocates mid-take is a recording that glitches.

use crate::SampleBuffer;

/// Where a recording is up to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecordState {
    /// Not listening.
    #[default]
    Idle,
    /// Listening, waiting for something loud enough to start on.
    Armed,
    /// Capturing.
    Recording,
    /// Capturing stopped because the buffer filled.
    Full,
}

/// Accumulates captured audio.
pub struct Recorder {
    channels: Vec<Vec<f32>>,
    sample_rate: u32,
    state: RecordState,
    /// Level that starts an armed take. Zero starts on the first frame.
    threshold: f32,
    /// Loudest sample seen since the last poll, for the level meter.
    peak: f32,
    max_frames: usize,
}

impl Recorder {
    /// A recorder that can hold `max_seconds` of `channels`-channel audio.
    pub fn new(sample_rate: u32, channels: usize, max_seconds: f32) -> Self {
        let max_frames = (sample_rate as f32 * max_seconds.max(0.1)) as usize;
        Self {
            channels: (0..channels.max(1))
                .map(|_| Vec::with_capacity(max_frames))
                .collect(),
            sample_rate,
            state: RecordState::Idle,
            threshold: 0.0,
            peak: 0.0,
            max_frames,
        }
    }

    pub fn state(&self) -> RecordState {
        self.state
    }

    /// Frames captured so far.
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// The rate the input is captured at.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Change the level an armed take waits for, keeping what it has.
    ///
    /// An infinite threshold listens without keeping anything: the input
    /// is open and the level meter moves, but the take has not begun —
    /// which is how a take kept to a bar waits for it, so the device's
    /// start-up is over by the time the bar comes.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.max(0.0);
    }

    /// What has been captured so far, a buffer per channel.
    pub fn captured(&self) -> &[Vec<f32>] {
        &self.channels
    }

    /// How full the buffer is, 0.0..=1.0.
    pub fn fullness(&self) -> f32 {
        if self.max_frames == 0 {
            return 1.0;
        }
        self.frames() as f32 / self.max_frames as f32
    }

    /// The loudest sample since the last call, and reset. Drives the meter,
    /// which is the only way to tell whether the input is live before
    /// committing to a take.
    pub fn take_peak(&mut self) -> f32 {
        core::mem::replace(&mut self.peak, 0.0)
    }

    /// Listen, starting the take once the input passes `threshold`.
    ///
    /// A threshold of zero starts immediately, which is what you want when
    /// recording something already playing; a threshold is what you want when
    /// recording yourself, so the take does not begin with you reaching for
    /// the pad.
    pub fn arm(&mut self, threshold: f32) {
        for channel in &mut self.channels {
            channel.clear();
        }
        self.threshold = threshold.max(0.0);
        self.peak = 0.0;
        self.state = RecordState::Armed;
    }

    /// Stop listening and throw away anything captured.
    pub fn cancel(&mut self) {
        for channel in &mut self.channels {
            channel.clear();
        }
        self.state = RecordState::Idle;
    }

    /// Feed captured frames, interleaved by channel.
    ///
    /// Ignored unless armed or recording, so a stray callback after a take
    /// cannot append to it.
    pub fn push(&mut self, interleaved: &[f32]) {
        if matches!(self.state, RecordState::Idle | RecordState::Full) {
            return;
        }
        let channels = self.channels.len();

        for (index, value) in interleaved.iter().enumerate() {
            let level = value.abs();
            self.peak = self.peak.max(level);

            if self.state == RecordState::Armed {
                if level < self.threshold {
                    continue;
                }
                // Start on the frame boundary, not mid-frame, or the
                // channels would be swapped for the rest of the take.
                if index % channels != 0 {
                    continue;
                }
                self.state = RecordState::Recording;
            }

            if self.frames() >= self.max_frames {
                self.state = RecordState::Full;
                break;
            }
            self.channels[index % channels].push(*value);
        }

        // A callback can end mid-frame; keep every channel the same length so
        // the buffer is always well formed.
        let shortest = self.channels.iter().map(Vec::len).min().unwrap_or(0);
        for channel in &mut self.channels {
            channel.truncate(shortest);
        }
    }

    /// End the take at exactly `frames` long: cut where it ran on, and
    /// padded with silence where the input fell short of it. `None`, as
    /// [`Recorder::stop`], when nothing was captured.
    ///
    /// For a take kept to the bar: the input is read once a frame, so it
    /// stops within a frame of the bar, and a loop a few milliseconds off
    /// its length drifts against the Pattern it was recorded to.
    pub fn stop_at_length(&mut self, frames: usize) -> Option<SampleBuffer> {
        if self.frames() == 0 {
            self.state = RecordState::Idle;
            return None;
        }
        for channel in &mut self.channels {
            channel.resize(frames, 0.0);
        }
        self.stop()
    }

    /// End the take and hand back what was captured.
    ///
    /// `None` when nothing was: an armed take that never passed its
    /// threshold has nothing to give, and a pad should not be handed silence.
    pub fn stop(&mut self) -> Option<SampleBuffer> {
        self.state = RecordState::Idle;
        if self.frames() == 0 {
            return None;
        }
        let channels = self
            .channels
            .iter_mut()
            .map(core::mem::take)
            .collect::<Vec<_>>();
        for channel in &mut self.channels {
            channel.reserve(self.max_frames);
        }
        Some(SampleBuffer::new(channels, self.sample_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    fn report(recorder: &Recorder, captured: Option<&SampleBuffer>) -> String {
        let mut out = format!("{:?}, {} frames", recorder.state(), recorder.frames());
        if let Some(buffer) = captured {
            out.push_str(&format!(
                "\ncaptured {} ch x {} frames @{} Hz:",
                buffer.num_channels(),
                buffer.num_samples(),
                buffer.sample_rate()
            ));
            for channel in 0..buffer.num_channels() {
                let values: Vec<String> = buffer
                    .channel(channel)
                    .iter()
                    .take(8)
                    .map(|v| format!("{v:.2}"))
                    .collect();
                out.push_str(&format!("\n  ch{channel}: {}", values.join(" ")));
            }
        }
        out
    }

    /// An idle recorder ignores whatever the input is doing.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn an_idle_recorder_ignores_input() {
        let mut recorder = Recorder::new(48_000, 1, 1.0);
        recorder.push(&[0.5, 0.5, 0.5]);
        expect!["Idle, 0 frames"].assert_eq(&report(&recorder, None));
    }

    /// Armed with no threshold, a take starts on the first frame.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_take_with_no_threshold_starts_immediately() {
        let mut recorder = Recorder::new(48_000, 1, 1.0);
        recorder.arm(0.0);
        recorder.push(&[0.1, 0.2, 0.3]);
        let captured = recorder.stop();
        expect![[r#"
            Idle, 0 frames
            captured 1 ch x 3 frames @48000 Hz:
              ch0: 0.10 0.20 0.30"#]]
        .assert_eq(&report(&recorder, captured.as_ref()));
    }

    /// A threshold skips the quiet run-in, so a take does not begin with the
    /// sound of reaching for the pad.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_threshold_skips_the_quiet_run_in() {
        let mut recorder = Recorder::new(48_000, 1, 1.0);
        recorder.arm(0.25);
        recorder.push(&[0.01, 0.02, 0.4, 0.5, 0.05]);
        let captured = recorder.stop();
        expect![[r#"
            Idle, 0 frames
            captured 1 ch x 3 frames @48000 Hz:
              ch0: 0.40 0.50 0.05"#]]
        .assert_eq(&report(&recorder, captured.as_ref()));
    }

    /// Stereo input keeps its channels apart, and a take that starts
    /// mid-buffer still starts on a frame boundary.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn stereo_input_starts_on_a_frame_boundary() {
        let mut recorder = Recorder::new(48_000, 2, 1.0);
        recorder.arm(0.25);
        // L R L R L R — the first loud sample is a right channel, so the
        // take must begin at the frame after it, not part-way through one.
        recorder.push(&[0.0, 0.0, 0.1, 0.9, 0.7, 0.3]);
        let captured = recorder.stop();
        expect![[r#"
            Idle, 0 frames
            captured 2 ch x 1 frames @48000 Hz:
              ch0: 0.70
              ch1: 0.30"#]]
        .assert_eq(&report(&recorder, captured.as_ref()));
    }

    /// A take that never gets loud enough hands back nothing rather than
    /// putting silence on a pad.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_take_that_never_triggers_captures_nothing() {
        let mut recorder = Recorder::new(48_000, 1, 1.0);
        recorder.arm(0.5);
        recorder.push(&[0.01; 64]);
        let captured = recorder.stop();
        expect![[r#"
            Idle, 0 frames
            captured: None"#]]
        .assert_eq(&format!(
            "{}\ncaptured: {:?}",
            report(&recorder, None),
            captured.map(|b| b.num_samples())
        ));
    }

    /// Filling the buffer stops the take rather than growing it, and what
    /// was captured is still good.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn filling_the_buffer_ends_the_take() {
        // A tenth of a second at 100 Hz is ten frames.
        let mut recorder = Recorder::new(100, 1, 0.1);
        recorder.arm(0.0);
        recorder.push(&[0.5; 40]);
        let state = recorder.state();
        let captured = recorder.stop();
        expect![[r#"
            Full while full
            captured Some(10) frames"#]]
        .assert_eq(&format!(
            "{state:?} while full\ncaptured {:?} frames",
            captured.map(|b| b.num_samples())
        ));
    }

    /// The meter reports the loudest sample since it was last read, so a
    /// quiet passage does not keep showing an old peak.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn the_meter_reports_and_resets() {
        let mut recorder = Recorder::new(48_000, 1, 1.0);
        recorder.arm(0.0);
        recorder.push(&[0.2, -0.8, 0.1]);
        let loud = recorder.take_peak();
        recorder.push(&[0.05, 0.05]);
        let quiet = recorder.take_peak();
        expect!["0.80 then 0.05"].assert_eq(&format!("{loud:.2} then {quiet:.2}"));
    }

    /// Re-arming throws away the previous take rather than appending to it.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn re_arming_starts_a_new_take() {
        let mut recorder = Recorder::new(48_000, 1, 1.0);
        recorder.arm(0.0);
        recorder.push(&[0.1, 0.2]);
        recorder.arm(0.0);
        recorder.push(&[0.7, 0.8]);
        let captured = recorder.stop();
        expect![[r#"
            Idle, 0 frames
            captured 1 ch x 2 frames @48000 Hz:
              ch0: 0.70 0.80"#]]
        .assert_eq(&report(&recorder, captured.as_ref()));
    }
}
