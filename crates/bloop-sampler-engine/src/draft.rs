//! bloop-sampler-engine: Sample loading, memory management, pitch shifting.
//!
//! Not part of the build yet — this is a design draft.

// use bloop_core::*;

pub struct SampleData {
    pub channels: u16,
    pub sample_rate: u32,
    pub samples: Vec<f32>,   // interleaved if stereo
}

/// Manages loaded samples. Lives on the UI thread.
pub struct SampleBank {
    // HashMap<SampleId, Arc<SampleData>>
}

/// Reads samples with pitch shifting via variable-rate playback.
pub struct SamplePlayer {
    sample_rate: f64,
    playback_rate: f64,  // 1.0 = original pitch, 2.0 = octave up
    position: f64,       // fractional sample position
    playing: bool,
}

impl SamplePlayer {
    pub fn new(sample_rate: f64) -> Self { todo!() }
    pub fn set_playback_rate(&mut self, rate: f64) { todo!() }
    pub fn trigger(&mut self) { todo!() }
    pub fn release(&mut self) { todo!() }

    /// Render into output buffer. Allocation-free.
    pub fn process(&mut self, sample: &SampleData, output: &mut [f32]) { todo!() }
}

// --- Test harness ---

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};

    fn check_playback(
        sample: &[f32],
        sample_rate: u32,
        playback_rate: f64,
        output_len: usize,
        expect: Expect,
    ) {
        let data = SampleData { channels: 1, sample_rate, samples: sample.to_vec() };
        let mut player = SamplePlayer::new(sample_rate as f64);
        player.set_playback_rate(playback_rate);
        player.trigger();
        let mut output = vec![0.0; output_len];
        player.process(&data, &mut output);
        let snap: Vec<String> = output.iter().map(|s| format!("{s:.3}")).collect();
        expect.assert_eq(&snap.join(", "));
    }

    #[test]
    fn original_speed() {
        check_playback(
            &[0.0, 0.25, 0.5, 0.75, 1.0],
            48000, 1.0, 5,
            expect![[r#"0.000, 0.250, 0.500, 0.750, 1.000"#]],
        );
    }

    #[test]
    fn double_speed() {
        check_playback(
            &[0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25],
            48000, 2.0, 4,
            expect![[r#"0.000, 0.500, 1.000, 0.500"#]],
        );
    }
}
