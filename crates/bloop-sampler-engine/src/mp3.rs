//! Stereo MP3s for EXPORT and SHARE: the format every app a song is sent
//! through takes.
//!
//! The encoder is `rusty_mp3` (Apache-2.0, pure Rust): MPEG-1 Layer III,
//! constant bitrate, its psychoacoustic model and bit reservoir. It links
//! statically on every target bloop builds for; LAME, the usual encoder,
//! is LGPL and would have to be a library of its own.
//!
//! [`Mp3Encoding`] encodes a slice at a time, so a long song does not
//! stall the frame it finishes in.

use rusty_mp3::{Mp3Encoder, Mp3EncoderConfig};

/// The version of `rusty_mp3` the build pins (Cargo.toml), which the
/// manual's License page cites.
pub const RUSTY_MP3_VERSION: &str = "1.0.0";

/// The bitrate every MP3 is written at, constant: transparent for most
/// music, about 1.9 MB a minute. The highest at which the encoder keeps
/// its bit reservoir on.
pub const MP3_KBPS: u32 = 256;

/// Silence after the audio, in frames: the encoder and a decoder delay
/// what they are given by about 1057 frames between them, and what is
/// still in the encoder at its end would be lost — a song's last 22 ms.
/// One MP3 frame's worth carries all of it out.
const FLUSH: usize = 1152;

/// One stereo MP3 being encoded, a slice at a time.
pub struct Mp3Encoding {
    encoder: Mp3Encoder,
    left: Vec<f32>,
    right: Vec<f32>,
    rate: u32,
    gain: f32,
    /// How many frames of `left` and `right` have been handed over.
    at: usize,
    /// One slice, interleaved: kept between calls.
    scratch: Vec<f32>,
    out: Vec<u8>,
}

impl Mp3Encoding {
    /// Start encoding `left` and `right` (the shorter's length) at
    /// `rate`, each sample times `gain`.
    ///
    /// # Errors
    /// MP3 has no such sample rate: it has 32, 44.1 and 48 kHz and their
    /// halves and quarters.
    pub fn new(left: Vec<f32>, right: Vec<f32>, rate: u32, gain: f32) -> Result<Self, String> {
        rusty_mp3::encoder_header(rate, 2, MP3_KBPS)
            .map_err(|_| format!("MP3 has no sample rate of {rate} Hz"))?;
        let frames = left.len().min(right.len());
        Ok(Mp3Encoding {
            encoder: Mp3Encoder::new(Mp3EncoderConfig {
                bitrate_kbps: MP3_KBPS,
                vbr_quality: None,
            }),
            left,
            right,
            rate,
            gain,
            at: 0,
            scratch: Vec::new(),
            // 256 kbps is 32 bytes a millisecond.
            out: Vec::with_capacity(frames / 48 * 32 + 1024),
        })
    }

    /// Encode up to `budget` more frames; the whole file once the last
    /// is in.
    ///
    /// # Errors
    /// The encoder refused the audio: not expected of a rate [`new`]
    /// took.
    ///
    /// [`new`]: Mp3Encoding::new
    pub fn step(&mut self, budget: usize) -> Result<Option<Vec<u8>>, String> {
        let frames = self.frames();
        let end = self.at.saturating_add(budget.max(1)).min(frames);
        self.scratch.clear();
        for (l, r) in self.left[self.at..end]
            .iter()
            .zip(&self.right[self.at..end])
        {
            self.scratch.push(l * self.gain);
            self.scratch.push(r * self.gain);
        }
        if !self.scratch.is_empty() {
            self.encoder
                .push_pcm_f32(&self.scratch, 2, self.rate)
                .map_err(|error| format!("MP3: {error}"))?;
        }
        self.at = end;
        if end == frames {
            self.encoder
                .push_pcm_f32(&[0.0; 2 * FLUSH], 2, self.rate)
                .map_err(|error| format!("MP3: {error}"))?;
            self.encoder.finish();
        }
        while let Ok(packet) = self.encoder.next_packet() {
            self.out.extend_from_slice(&packet);
        }
        Ok((end == frames).then(|| std::mem::take(&mut self.out)))
    }

    /// How much of it has been encoded, 0 to 1.
    #[must_use]
    pub fn progress(&self) -> f32 {
        let frames = self.frames();
        if frames == 0 {
            return 1.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "a fraction")]
        let done = self.at as f32 / frames as f32;
        done
    }

    fn frames(&self) -> usize {
        self.left.len().min(self.right.len())
    }
}

/// Encode `left` and `right` as one stereo MP3 at `rate`, at once.
///
/// # Errors
/// As [`Mp3Encoding::new`].
pub fn encode_mp3(left: &[f32], right: &[f32], rate: u32) -> Result<Vec<u8>, String> {
    let mut encoding = Mp3Encoding::new(left.to_vec(), right.to_vec(), rate, 1.0)?;
    loop {
        if let Some(bytes) = encoding.step(usize::MAX)? {
            return Ok(bytes);
        }
    }
}
