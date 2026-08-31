//! Turning an audio file into the [`SampleBuffer`] pads share.
//!
//! Decoding runs on the UI thread — it allocates, and a long file takes a
//! while — and produces one immutable buffer that any number of pads can slice.
//!
//! Everything is decoded to `f32` at the file's own sample rate: the voice
//! resamples on playback (`sampleRateConversion`), so converting here would
//! only lose quality twice.

use symphonia::core::audio::{Channels, GenericAudioBufferRef};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::voice::MAX_CHANNELS;
use crate::SampleBuffer;

/// Why a file could not become a [`SampleBuffer`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportError {
    /// The bytes are not a container format Bloop was built to read.
    UnrecognisedFormat,
    /// The container has no audio track, or the track is unplayable.
    NoAudioTrack,
    /// The audio codec inside the container is not one Bloop was built to
    /// decode.
    UnsupportedCodec,
    /// The file decoded to nothing.
    Empty,
    /// The file is damaged, or truncated part-way through.
    Corrupt(String),
}

impl core::fmt::Display for ImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnrecognisedFormat => f.write_str("not an audio file Bloop can read"),
            Self::NoAudioTrack => f.write_str("no audio in this file"),
            Self::UnsupportedCodec => f.write_str("this file's audio codec is not supported"),
            Self::Empty => f.write_str("this file contains no audio"),
            Self::Corrupt(why) => write!(f, "this file is damaged ({why})"),
        }
    }
}

impl std::error::Error for ImportError {}

/// The formats an import can be, for the browser's file filter and for the
/// decode hint. Bloop reads WAV, AIFF, FLAC, Ogg Vorbis and MP3.
pub const IMPORTABLE_EXTENSIONS: [&str; 8] =
    ["wav", "wave", "aiff", "aif", "flac", "ogg", "oga", "mp3"];

/// Decode `bytes` into a shareable buffer, all at once.
///
/// `extension` (without the dot) only hints at the container; the bytes decide.
/// Files with more than [`MAX_CHANNELS`] channels are folded down, because
/// Bloop is stereo end to end.
///
/// A long file takes a while — a 76-second loop is around 100 ms — so a UI
/// that must stay responsive should drive [`SampleDecoder`] a slice at a time
/// instead.
pub fn decode(bytes: Vec<u8>, extension: Option<&str>) -> Result<SampleBuffer, ImportError> {
    let mut decoder = SampleDecoder::new(bytes, extension)?;
    loop {
        if let Some(buffer) = decoder.step(usize::MAX)? {
            return Ok(buffer);
        }
    }
}

/// A decode in progress.
///
/// Decoding a long sample costs more than a frame, and it runs on the UI
/// thread — so it is resumable: the caller spends a slice of it per frame and
/// the interface keeps moving. Short samples finish in the first slice.
pub struct SampleDecoder {
    reader: Box<dyn symphonia::core::formats::FormatReader + 'static>,
    decoder: Box<dyn symphonia::core::codecs::audio::AudioDecoder>,
    track_id: u32,
    channels: Vec<Vec<f32>>,
    rate: u32,
    interleaved: Vec<f32>,
    /// Total frames the container claims, for progress. Zero when unknown.
    expected_frames: u64,
    finished: bool,
}

impl SampleDecoder {
    /// Open `bytes` and get as far as the first packet.
    pub fn new(bytes: Vec<u8>, extension: Option<&str>) -> Result<Self, ImportError> {
        let mut hint = Hint::new();
        if let Some(extension) = extension {
            hint.with_extension(extension);
        }

        let stream = MediaSourceStream::new(
            Box::new(std::io::Cursor::new(bytes)),
            MediaSourceStreamOptions::default(),
        );

        let reader = symphonia::default::get_probe()
            .probe(
                &hint,
                stream,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|error| match error {
                // The container was recognised but ran out part-way through
                // its header — "damaged" is a truer thing to tell the user
                // than "not an audio file".
                SymphoniaError::IoError(_) => ImportError::Corrupt("it ends early".to_string()),
                _ => ImportError::UnrecognisedFormat,
            })?;

        let track = reader
            .first_track(TrackType::Audio)
            .ok_or(ImportError::NoAudioTrack)?;
        let track_id = track.id;
        let expected_frames = track.num_frames.unwrap_or(0);
        let Some(CodecParameters::Audio(params)) = track.codec_params.clone() else {
            return Err(ImportError::NoAudioTrack);
        };

        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &Default::default())
            .map_err(|_| ImportError::UnsupportedCodec)?;

        Ok(Self {
            reader,
            decoder,
            track_id,
            channels: Vec::new(),
            rate: 0,
            interleaved: Vec::new(),
            expected_frames,
            finished: false,
        })
    }

    /// Decode at most `max_packets` more packets.
    ///
    /// `Ok(None)` means there is more to do; `Ok(Some(buffer))` means the
    /// file is finished. Calling again after it finishes returns the error
    /// for an empty file rather than looping.
    pub fn step(&mut self, max_packets: usize) -> Result<Option<SampleBuffer>, ImportError> {
        for _ in 0..max_packets {
            if self.finished {
                break;
            }
            let packet = match self.reader.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    self.finished = true;
                    break;
                }
                // A file that stops mid-stream still yields what was decoded:
                // a truncated import is more useful than no import.
                Err(SymphoniaError::IoError(_)) => {
                    self.finished = true;
                    break;
                }
                Err(error) => return Err(ImportError::Corrupt(error.to_string())),
            };
            if packet.track_id != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                // Symphonia asks callers to skip a packet it could not decode
                // and keep going, rather than abandon the file.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(error) => return Err(ImportError::Corrupt(error.to_string())),
            };

            append(
                &decoded,
                &mut self.channels,
                &mut self.rate,
                &mut self.interleaved,
            );
        }

        if !self.finished {
            return Ok(None);
        }

        if self.channels.is_empty() || self.channels[0].is_empty() {
            return Err(ImportError::Empty);
        }
        let channels = core::mem::take(&mut self.channels);
        Ok(Some(SampleBuffer::new(fold_channels(channels), self.rate)))
    }

    /// How far along, in 0.0..=1.0. Best effort: containers that do not state
    /// their length report 0 until the decode finishes.
    pub fn progress(&self) -> f32 {
        if self.finished {
            return 1.0;
        }
        if self.expected_frames == 0 {
            return 0.0;
        }
        let done = self.channels.first().map_or(0, Vec::len) as f64;
        (done / self.expected_frames as f64).clamp(0.0, 1.0) as f32
    }
}

/// Append one decoded packet, growing the channel layout to fit.
fn append(
    decoded: &GenericAudioBufferRef<'_>,
    channels: &mut Vec<Vec<f32>>,
    rate: &mut u32,
    interleaved: &mut Vec<f32>,
) {
    let spec = decoded.spec();
    *rate = spec.rate();
    let count = channel_count(spec.channels()).max(1);

    // The first packet decides the layout; a later packet with more channels
    // (rare, but formats allow it) widens the buffer, back-filling silence.
    let frames_so_far = channels.first().map_or(0, Vec::len);
    while channels.len() < count {
        channels.push(vec![0.0; frames_so_far]);
    }

    decoded.copy_to_vec_interleaved(interleaved);
    for (index, sample) in interleaved.iter().enumerate() {
        channels[index % count].push(*sample);
    }

    // Keep every channel the same length even if a packet was short.
    let frames = channels.iter().map(Vec::len).max().unwrap_or(0);
    for channel in channels.iter_mut() {
        channel.resize(frames, 0.0);
    }
}

fn channel_count(channels: &Channels) -> usize {
    channels.count()
}

/// Fold to at most [`MAX_CHANNELS`].
///
/// Anything wider than stereo is averaged into a stereo pair — left half to
/// the left, right half to the right — which keeps a surround import roughly
/// where it was rather than silently dropping its rear channels.
fn fold_channels(channels: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if channels.len() <= MAX_CHANNELS {
        return channels;
    }

    let frames = channels[0].len();
    let split = channels.len().div_ceil(2);
    let mut folded = vec![vec![0.0f32; frames]; MAX_CHANNELS];
    for (index, channel) in channels.iter().enumerate() {
        let side = usize::from(index >= split);
        let width = if side == 0 {
            split
        } else {
            channels.len() - split
        };
        let gain = 1.0 / width as f32;
        for (frame, sample) in channel.iter().enumerate() {
            folded[side][frame] += sample * gain;
        }
    }
    folded
}
