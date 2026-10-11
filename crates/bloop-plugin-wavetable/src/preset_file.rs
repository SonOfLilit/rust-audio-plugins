//! A Wavetable preset file, as the plug-in saves one: what `tools/import.py`
//! reads for the factory presets, read here at runtime, so a user's own
//! preset loads as theirs does in the plug-in.
//!
//! ```xml
//! <state name="…" author="…" tags="…">
//!   <state wt1="FM 07" wt2="…" wt1Data="<size>.<juce base64>" wt1Size="2048" …>
//!     <MODMATRIX>
//!       <MODITEM srcId="env1" dstId="osc1pos" depth="0.5" enabled="1"
//!                function="sineIn" biPolarMapping="1"/>
//!     </MODMATRIX>
//!   </state>
//!   <param uid="osc1pos" val="37.6"/>
//! </state>
//! ```
//!
//! What it reads, it reads as the original does: a value is a double
//! rounded to a float (`float (var)`), a table it carries (`wt1Data`) is a
//! WAV of frames `wt1Size` long or as its `clm ` chunk says, one it names
//! is the bank table of that name, a route without `biPolarMapping` maps as
//! its source swings (Wavetable's `sameAsSource`), and an unknown
//! `function` is linear (`strToFunc`). What bloop does not have — a
//! parameter, a source, a destination, a table — is left out and said
//! ([`UserPreset::ignored`]); so is a route past the matrix's room
//! ([`UserPreset::left_out`]).

use crate::modmatrix::{self, RouteSlot, MAX_PER_DESTINATION};
use crate::params::{self, CURVE_IDS, ROUTES};
use crate::presets::UserPreset;
use crate::{bank, presets};

/// The editor's own state, which a preset carries and bloop has no use
/// for (`tools/import.py`'s `EDITOR_ONLY`).
fn editor_only(uid: &str) -> bool {
    matches!(uid, "uiLFO" | "uiENV" | "uiMOD")
        || (uid.starts_with("lfo") && (uid.ends_with("xgrid") || uid.ends_with("ygrid")))
}

/// `juce::MemoryBlock::fromBase64Encoding`: `<size>.<chars>`, six bits a
/// character from JUCE's own alphabet, least significant bit first.
#[must_use]
pub fn juce_base64(text: &str) -> Vec<u8> {
    const ALPHABET: &[u8; 64] = b".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+";
    let (size, chars) = text.split_once('.').unwrap_or((text, ""));
    let size: usize = size.trim().parse().unwrap_or(0);
    let mut out = vec![0u8; size];
    let mut bit = 0usize;
    for c in chars.bytes() {
        let Some(v) = ALPHABET.iter().position(|&a| a == c) else {
            continue;
        };
        for k in 0..6 {
            let at = bit + k;
            if at / 8 >= size {
                break;
            }
            if v >> k & 1 == 1 {
                out[at / 8] |= 1 << (at % 8);
            }
        }
        bit += 6;
    }
    out
}

/// A preset's number, as the original reads one: a double, rounded to a
/// float. `None` for text that is no number.
fn float_of(text: &str) -> Option<f32> {
    text.trim().parse::<f64>().ok().map(|v| v as f32)
}

/// `bool (var (text))`: a number other than zero, or "true".
fn truth_of(text: &str) -> bool {
    let t = text.trim();
    t.eq_ignore_ascii_case("true") || t.parse::<f64>().is_ok_and(|v| v != 0.0)
}

/// The Browser's TYPE for a preset with these tags: the first tag that
/// names one, as `tools/import.py` files the factory presets; Synth Misc
/// when none does.
#[must_use]
pub fn type_of_tags(tags: &str) -> &'static str {
    for word in tags.split_whitespace() {
        let kind = match word.to_lowercase().as_str() {
            "kick" | "snare" | "drum" | "drums" | "hat" => "Drums",
            "percussion" => "Percussion",
            "bass" | "basses" => "Bass",
            "fx" | "sfx" | "effect" | "riser" | "sweeper" | "siren" | "dive" | "glitch"
            | "glitchy" => "Sound Effects",
            "lead" | "leads" => "Synth Lead",
            "pad" => "Synth Pad",
            "key" | "keys" => "Piano / Keys",
            "brass" => "Brass",
            "string" => "Bowed Strings",
            "mallet" | "bells" => "Mallet Instruments",
            "vocalise" | "vocals" => "Vocal",
            "pluck" | "plucks" | "plucked" | "chord" | "arp" | "arpeggio" | "sequence" | "stab"
            | "synth" | "gate" | "initial" => "Synth Misc",
            _ => continue,
        };
        return kind;
    }
    "Synth Misc"
}

/// The table oscillator `n` (1 or 2) plays, from the inner `state`: one
/// it carries, kept as a table of the user's; or one it names. `None`,
/// said in `ignored`, for a name no bank has.
fn table_of(state: roxmltree::Node, n: usize, ignored: &mut Vec<String>) -> Option<u16> {
    let name = state
        .attribute(format!("wt{n}").as_str())
        .unwrap_or_default();
    let data = juce_base64(
        state
            .attribute(format!("wt{n}Data").as_str())
            .unwrap_or_default(),
    );
    if !data.is_empty() {
        let size = state
            .attribute(format!("wt{n}Size").as_str())
            .and_then(|s| s.trim().parse::<i64>().ok())
            .and_then(|s| usize::try_from(s).ok())
            .filter(|&s| s > 0);
        return match bank::keep_wav(name, &data, size) {
            Ok(index) => Some(index),
            Err(error) => {
                ignored.push(format!(
                    "the table {name} it carries for OSC {n}: {error:?}"
                ));
                None
            }
        };
    }
    if name.is_empty() {
        return None;
    }
    let found = bank::index_of(name).and_then(|i| u16::try_from(i).ok());
    if found.is_none() {
        ignored.push(format!(
            "the table {name} for OSC {n}, which bloop does not have"
        ));
    }
    found
}

/// A `MODITEM` as a route, or why it is none.
fn route_of(item: roxmltree::Node) -> Result<(RouteSlot, String), String> {
    let source_id = item.attribute("srcId").unwrap_or_default();
    let destination_id = item.attribute("dstId").unwrap_or_default();
    let said = format!("{source_id} to {destination_id}");
    let source = modmatrix::source_named(source_id)
        .ok_or_else(|| format!("the route {said}: bloop has no source {source_id}"))?;
    let destination = modmatrix::destination_named(destination_id)
        .ok_or_else(|| format!("the route {said}: bloop has no destination {destination_id}"))?;
    let curve = item
        .attribute("function")
        .and_then(|f| CURVE_IDS.iter().position(|id| *id == f))
        .unwrap_or(0);
    let mapping = match item.attribute("biPolarMapping") {
        None => 0,
        Some(b) if truth_of(b) => 2,
        Some(_) => 1,
    };
    Ok((
        RouteSlot {
            source: source + 1,
            destination: destination + 1,
            depth: item.attribute("depth").and_then(float_of).unwrap_or(0.0),
            enabled: item.attribute("enabled").is_none_or(truth_of),
            curve,
            mapping,
        },
        said,
    ))
}

/// A Wavetable preset file, read. The tables it carries are kept as the
/// user's own on the way ([`bank::keep_wav`]), so the preset plays them.
///
/// # Errors
/// It is not text, not XML, or not a Wavetable preset.
pub fn read(bytes: &[u8]) -> Result<UserPreset, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "it is not text".to_string())?;
    let document = roxmltree::Document::parse(text).map_err(|e| format!("it is not XML: {e}"))?;
    let root = document.root_element();
    if root.tag_name().name() != "state" {
        return Err("it is not a Wavetable preset: its root is not <state>".to_string());
    }
    let mut ignored = Vec::new();
    let mut params = Vec::new();
    for param in root.children().filter(|n| n.has_tag_name("param")) {
        let uid = param.attribute("uid").unwrap_or_default();
        if editor_only(uid) {
            continue;
        }
        match (
            params::index_of(uid),
            param.attribute("val").and_then(float_of),
        ) {
            (Some(at), Some(value)) if params::PARAMS[at].kind != params::Kind::Host => {
                params.push((at, value));
            }
            _ => ignored.push(format!("the parameter {uid}, which bloop does not have")),
        }
    }
    let inner = root.children().find(|n| n.has_tag_name("state"));
    let tables = match inner {
        Some(state) => [1, 2].map(|n| table_of(state, n, &mut ignored)),
        None => [None, None],
    };
    let mut routes = Vec::new();
    let mut left_out = Vec::new();
    let mut per_destination = [0usize; modmatrix::DESTINATION_COUNT];
    let items = inner
        .and_then(|s| s.children().find(|n| n.has_tag_name("MODMATRIX")))
        .into_iter()
        .flat_map(|m| m.children().filter(|n| n.has_tag_name("MODITEM")));
    for item in items {
        let (route, said) = match route_of(item) {
            Ok(found) => found,
            Err(why) => {
                ignored.push(why);
                continue;
            }
        };
        if routes.len() == ROUTES {
            left_out.push(format!("{said}: past the matrix's {ROUTES} routes"));
            continue;
        }
        if route.enabled {
            let on = &mut per_destination[route.destination - 1];
            if *on == MAX_PER_DESTINATION {
                left_out.push(format!(
                    "{said}: past {MAX_PER_DESTINATION} routes on one destination"
                ));
                continue;
            }
            *on += 1;
        }
        routes.push(route);
    }
    let tags = root.attribute("tags").unwrap_or_default().to_string();
    Ok(UserPreset {
        name: root.attribute("name").unwrap_or("Untitled").to_string(),
        author: root.attribute("author").unwrap_or_default().to_string(),
        kind: type_of_tags(&tags).to_string(),
        tags,
        params,
        tables,
        routes,
        left_out,
        ignored,
    })
}

/// A WAV of the user's as a table (Wavetable's "+"): kept, and a preset
/// that is Init with it on OSC 1, named after it.
///
/// # Errors
/// What [`bank::keep_wav`] says.
pub fn table_preset(name: &str, wav: &[u8]) -> Result<UserPreset, bank::LoadError> {
    let index = bank::keep_wav(name, wav, None)?;
    Ok(UserPreset {
        name: name.to_string(),
        author: String::new(),
        tags: String::new(),
        kind: "Synth Misc".to_string(),
        params: Vec::new(),
        tables: [Some(index), None],
        routes: Vec::new(),
        left_out: Vec::new(),
        ignored: Vec::new(),
    })
}

/// [`read`], and the preset kept for the run ([`presets::keep`]): its
/// number, for the preset parameter.
///
/// # Errors
/// What [`read`] says, or that there is no room for another.
pub fn import(bytes: &[u8]) -> Result<(usize, &'static UserPreset), String> {
    let preset = read(bytes)?;
    let index =
        presets::keep(preset).ok_or_else(|| "there is no room for another preset".to_string())?;
    let kept = presets::user(index).ok_or_else(|| "the preset was not kept".to_string())?;
    Ok((index, kept))
}
