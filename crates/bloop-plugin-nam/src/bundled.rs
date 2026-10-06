//! The captures bloop ships.
//!
//! NAM captures are made by the community and licensed however their
//! makers chose; bloop ships only ones whose licence plainly allows it.
//!
//! Two are Steven Atkinson's, from his MIT-licensed repositories, both the
//! trainer's A1 standard WaveNet:
//!
//! - **Obsidian**: an Omega Ampworks Obsidian, high gain, modelled by him —
//!   the capture NeuralAmpModelerPlugin ships as its REAPER example
//!   (`REAPER/model.nam`, NeuralAmpModelerPlugin rev
//!   16be869746b8915885c7a35bafdcc0061faeb50e).
//! - **Example**: NeuralAmpModelerCore's own A1 "standard" example
//!   (`example_models/wavenet_a1_standard.nam`, the rev `vendor/nam` pins),
//!   which says nothing of what it captured. It plays like a high-gain amp
//!   through a cabinet: heavily compressed, its harmonics falling away above
//!   4 kHz.
//!
//! Four are VoLum's (github.com/guitarlum/VoLum, `rigs/`, rev
//! ddb7822ce06df9eaa74156027cf175a9ffe8305a): amp and cabinet captured
//! together, retrained by VoLum as A2 (Lite and Full in one container).
//! The repository is MIT — "Copyright (c) 2024-2026 Steffen Dangmann and
//! VoLum contributors" — with no exception for the rigs, which its own
//! history shows its maintainer training; the files' metadata names no
//! other author. `models/volum/LICENSE` is that licence. Each is one gain
//! channel and one cabinet of a VoLum amp, named here for how it plays:
//!
//! - **Clean**: `V30-ORS1-1` (channel 1, V30 cabinet).
//! - **Blues**: `G12-Sbgo-1`, a boutique pedal-platform amp near breakup.
//! - **Crunch**: `G12-2203-2`, a 1970s British master-volume head.
//! - **High Gain**: `V30-SLO1-3`, a high-gain lead channel.
//!
//! All six were trained at 48 kHz.

use std::sync::OnceLock;

use crate::Model;

/// A capture bloop ships: its name in the Model list, and its file.
pub struct Capture {
    pub name: &'static str,
    pub text: &'static str,
}

pub const CAPTURES: [Capture; 6] = [
    Capture {
        name: "Obsidian",
        text: include_str!("../models/obsidian.nam"),
    },
    Capture {
        name: "Example",
        text: include_str!("../models/example.nam"),
    },
    Capture {
        name: "Clean",
        text: include_str!("../models/volum/V30-ORS1-1.nam"),
    },
    Capture {
        name: "Blues",
        text: include_str!("../models/volum/G12-Sbgo-1.nam"),
    },
    Capture {
        name: "Crunch",
        text: include_str!("../models/volum/G12-2203-2.nam"),
    },
    Capture {
        name: "High Gain",
        text: include_str!("../models/volum/V30-SLO1-3.nam"),
    },
];

/// The VoLum revision the four VoLum captures come from.
pub const VOLUM_REV: &str = "ddb7822ce06df9eaa74156027cf175a9ffe8305a";

/// Every shipped capture, read and prewarmed once, in [`CAPTURES`] order:
/// a plug-in takes copies, and puts a copy back to how it was here when it
/// switches to it. (A capture that did not read would be left out; a test
/// reads every one.)
#[must_use]
pub fn models() -> &'static [Model] {
    static MODELS: OnceLock<Vec<Model>> = OnceLock::new();
    MODELS.get_or_init(|| {
        CAPTURES
            .iter()
            .filter_map(|c| Model::parse(c.text).ok())
            .collect()
    })
}
