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
/// decode hint. Bloop reads WAV, FLAC, Ogg Vorbis and MP3.
pub const IMPORTABLE_EXTENSIONS: [&str; 6] = ["wav", "wave", "flac", "ogg", "oga", "mp3"];

/// Decode `bytes` into a shareable buffer.
///
/// `extension` (without the dot) only hints at the container; the bytes decide.
/// Files with more than [`MAX_CHANNELS`] channels are folded down, because
/// Bloop is stereo end to end.
pub fn decode(bytes: Vec<u8>, extension: Option<&str>) -> Result<SampleBuffer, ImportError> {
    let mut hint = Hint::new();
    if let Some(extension) = extension {
        hint.with_extension(extension);
    }

    let stream = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );

    let mut reader = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| match error {
            // The container was recognised but ran out part-way through its
            // header — "damaged" is a truer thing to tell the user than "not
            // an audio file".
            SymphoniaError::IoError(_) => ImportError::Corrupt("it ends early".to_string()),
            _ => ImportError::UnrecognisedFormat,
        })?;

    let track = reader
        .first_track(TrackType::Audio)
        .ok_or(ImportError::NoAudioTrack)?;
    let track_id = track.id;
    let Some(CodecParameters::Audio(params)) = track.codec_params.clone() else {
        return Err(ImportError::NoAudioTrack);
    };

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &Default::default())
        .map_err(|_| ImportError::UnsupportedCodec)?;

    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut rate = 0u32;
    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match reader.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            // A file that stops mid-stream still yields what was decoded: a
            // truncated import is more useful than no import.
            Err(SymphoniaError::IoError(_)) => break,
            Err(error) => return Err(ImportError::Corrupt(error.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // Symphonia asks callers to skip a packet it could not decode and
            // keep going, rather than abandon the file.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(ImportError::Corrupt(error.to_string())),
        };

        append(&decoded, &mut channels, &mut rate, &mut interleaved);
    }

    if channels.is_empty() || channels[0].is_empty() {
        return Err(ImportError::Empty);
    }

    Ok(SampleBuffer::new(fold_channels(channels), rate))
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
