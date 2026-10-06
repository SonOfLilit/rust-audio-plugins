//! Six Airwindows plug-ins by Chris Johnson, ported to Rust: Console7Channel,
//! Console7Buss, ToTape8, Density2, Galactic3 and Baxandall2.
//!
//! Each is the LinuxVST source's `processReplacing` — float in and out,
//! double inside — operation for operation: the same promotions between
//! float and double, the same literals, the denormal guard and the 32-bit
//! floating-point dither every Airwindows plug-in ends on (src/math.rs).
//! `cargo test -p bloop-plugin-airwindows --features gold` compiles the
//! unmodified originals from `vendor/airwindows` and holds every port to
//! them bit for bit (tests/it/gold.rs).
//!
//! Two things the originals leave to the platform, the port fixes:
//!
//! - **The dither seeds.** The originals seed from `rand()`. A bloop
//!   instance starts with the seeds a plug-in constructed first in a fresh
//!   process after `srand(1)` gets (`math::SEEDS`), and [`Airwindows::clear`]
//!   goes back to them.
//! - **`long double`.** The dither scales by `5.5e-36l`, a `long double`:
//!   80-bit on x86 Linux, 64-bit on Apple's arm64 and on MSVC. The port
//!   computes it as a 64-bit `double`; the gold harness is built with
//!   `-mlong-double-64` to match.
//!
//! Every transcendental is `libm`'s by default, so output is the same bits
//! on every target, wasm included; `gold` switches them to the C library
//! the C++ links (src/math.rs).

/// The Airwindows revision the ports follow, which `vendor/airwindows`
/// pins and the gold tests compile.
pub const PORTED_FROM_AIRWINDOWS_REV: &str = "e718c9bcfcdd736deeddb08bffe6bce2aa8e0eea";

mod baxandall2;
mod console7;
mod density2;
mod galactic3;
pub mod math;
mod to_tape8;

use baxandall2::Baxandall2;
pub use console7::{Console7Buss, Console7Channel};
use density2::Density2;
use galactic3::Galactic3;
use to_tape8::ToTape8;

/// Which plug-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Console7Channel,
    Console7Buss,
    ToTape8,
    Density2,
    Galactic3,
    Baxandall2,
}

/// One knob: its name as the original's `getParameterName` gives it, and
/// the value its constructor sets. Values are the original's, 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub default: f32,
}

/// A knob whose constructor sets it from a `double` literal, rounded to
/// `float` as the C++ assignment rounds it.
const fn param(name: &'static str, default: f64) -> Param {
    Param {
        name,
        default: default as f32,
    }
}

const CONSOLE7_CHANNEL: &[Param] = &[param("Fader", 0.772)];
const CONSOLE7_BUSS: &[Param] = &[param("Master", 1.0)];
const TO_TAPE8: &[Param] = &[
    param("Input", 0.5),
    param("Tilt", 0.5),
    param("Shape", 0.5),
    param("Flutter", 0.5),
    param("FlutSpd", 0.5),
    param("Bias", 0.5),
    param("HeadBmp", 0.5),
    param("HeadFrq", 0.5),
    param("Output", 0.5),
];
const DENSITY2: &[Param] = &[
    param("Density", 0.2),
    param("Highpass", 0.0),
    param("Output", 1.0),
    param("Dry/Wet", 1.0),
];
const GALACTIC3: &[Param] = &[
    param("Replace", 0.5),
    param("Brightns", 0.5),
    param("Detune", 0.5),
    param("Derez", 1.0),
    param("Bigness", 1.0),
    param("Dry/Wet", 1.0),
];
const BAXANDALL2: &[Param] = &[param("Treble", 0.5), param("Bass", 0.5)];

impl Kind {
    pub const ALL: [Kind; 6] = [
        Kind::Console7Channel,
        Kind::Console7Buss,
        Kind::ToTape8,
        Kind::Density2,
        Kind::Galactic3,
        Kind::Baxandall2,
    ];

    /// The original's plug-in name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Kind::Console7Channel => "Console7Channel",
            Kind::Console7Buss => "Console7Buss",
            Kind::ToTape8 => "ToTape8",
            Kind::Density2 => "Density2",
            Kind::Galactic3 => "Galactic3",
            Kind::Baxandall2 => "Baxandall2",
        }
    }

    /// The knobs, in the original's A, B, C… order.
    #[must_use]
    pub fn params(self) -> &'static [Param] {
        match self {
            Kind::Console7Channel => CONSOLE7_CHANNEL,
            Kind::Console7Buss => CONSOLE7_BUSS,
            Kind::ToTape8 => TO_TAPE8,
            Kind::Density2 => DENSITY2,
            Kind::Galactic3 => GALACTIC3,
            Kind::Baxandall2 => BAXANDALL2,
        }
    }

    /// Which of bloop's Browser types it lists under.
    ///
    /// The Console7 pair are a console's saturation — a sine curve on the
    /// channel, an arcsine on the buss — with a fader in front, and
    /// `ToTape8` is tape: a saturating record and playback path with flutter,
    /// bias and head bump around it. What each does to a signal is mostly
    /// colour and soft clipping, so all three are "Distortion", beside
    /// `Density2`. None of them follows the level the way a compressor does.
    #[must_use]
    pub fn category(self) -> &'static str {
        match self {
            Kind::Console7Channel | Kind::Console7Buss | Kind::ToTape8 | Kind::Density2 => {
                "Distortion"
            }
            Kind::Galactic3 => "Reverb",
            Kind::Baxandall2 => "EQ",
        }
    }
}

/// One Airwindows plug-in, running.
#[derive(Clone, Debug)]
pub enum Airwindows {
    Console7Channel(Box<Console7Channel>),
    Console7Buss(Box<Console7Buss>),
    ToTape8(Box<ToTape8>),
    Density2(Box<Density2>),
    Galactic3(Box<Galactic3>),
    Baxandall2(Box<Baxandall2>),
}

/// `pinParameter`: 0 to 1, a NaN let through as the original lets it.
fn pin(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

macro_rules! each {
    ($self:expr, $p:ident => $body:expr) => {
        match $self {
            Airwindows::Console7Channel($p) => $body,
            Airwindows::Console7Buss($p) => $body,
            Airwindows::ToTape8($p) => $body,
            Airwindows::Density2($p) => $body,
            Airwindows::Galactic3($p) => $body,
            Airwindows::Baxandall2($p) => $body,
        }
    };
}

impl Airwindows {
    /// A plug-in as its constructor leaves it. Allocates (Galactic3's
    /// delay lines are a megabyte); build it off the audio thread.
    #[must_use]
    pub fn new(kind: Kind, sample_rate: f32) -> Self {
        match kind {
            Kind::Console7Channel => {
                Self::Console7Channel(Box::new(Console7Channel::new(sample_rate)))
            }
            Kind::Console7Buss => Self::Console7Buss(Box::new(Console7Buss::new(sample_rate))),
            Kind::ToTape8 => Self::ToTape8(Box::new(ToTape8::new(sample_rate))),
            Kind::Density2 => Self::Density2(Box::new(Density2::new(sample_rate))),
            Kind::Galactic3 => Self::Galactic3(Galactic3::boxed(sample_rate)),
            Kind::Baxandall2 => Self::Baxandall2(Box::new(Baxandall2::new(sample_rate))),
        }
    }

    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Self::Console7Channel(_) => Kind::Console7Channel,
            Self::Console7Buss(_) => Kind::Console7Buss,
            Self::ToTape8(_) => Kind::ToTape8,
            Self::Density2(_) => Kind::Density2,
            Self::Galactic3(_) => Kind::Galactic3,
            Self::Baxandall2(_) => Kind::Baxandall2,
        }
    }

    fn params_mut(&mut self) -> &mut [f32] {
        each!(self, p => &mut p.params[..])
    }

    fn params(&self) -> &[f32] {
        each!(self, p => &p.params[..])
    }

    /// Knob `index` to `value`, held to 0..1. An index the plug-in does not
    /// have is ignored.
    pub fn set_param(&mut self, index: usize, value: f32) {
        if let Some(p) = self.params_mut().get_mut(index) {
            *p = pin(value);
        }
    }

    /// Knob `index`'s value; 0 for an index the plug-in does not have, as
    /// the original's `getParameter` returns.
    #[must_use]
    pub fn get_param(&self, index: usize) -> f32 {
        self.params().get(index).copied().unwrap_or(0.0)
    }

    /// `processReplacing` with the input and output the same buffers. Runs
    /// over as many frames as the shorter channel has. Allocation-free.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let n = left.len().min(right.len());
        let (left, right) = (&mut left[..n], &mut right[..n]);
        each!(self, p => p.process(left, right));
    }

    /// Back to the constructor's state, the knobs and the sample rate kept,
    /// the dither seeds the ones a new instance gets. Allocation-free.
    pub fn clear(&mut self) {
        each!(self, p => p.clear());
    }

    /// What the original's `getSampleRate()` returns from now on.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        each!(self, p => p.sample_rate = sample_rate);
    }

    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        each!(self, p => p.sample_rate)
    }

    /// How long the plug-in can go on sounding after its input stops, at
    /// its knobs as they are now. Infinite for a Galactic3 at Replace 0,
    /// which holds what it has for ever.
    #[must_use]
    pub fn tail_seconds(&self) -> f32 {
        match self {
            Self::Galactic3(p) => p.tail_seconds(),
            Self::ToTape8(_) => to_tape8::TAIL_SECONDS,
            Self::Baxandall2(_) => baxandall2::TAIL_SECONDS,
            Self::Console7Channel(_) | Self::Console7Buss(_) | Self::Density2(_) => 0.0,
        }
    }
}
