//! A `.nam` file: its JSON read into a network ready to run.
//!
//! Read on the UI thread (it allocates); [`Model::process`] allocates
//! nothing.
//!
//! A capture can come in more than one size. NAM's A2 files are
//! `SlimmableContainer`s holding the same capture trained twice, A2-Lite
//! and A2-Full; a slimmable WaveNet runs on fewer channels than it has.
//! Both pick what runs from a value from 0 to 1 (`SetSlimmableSize`),
//! which NAM's core defaults to the full size. A [`Model`] keeps two: the
//! full one, and the one a step below it ([`Size::Light`]), which is what
//! a phone can afford live.

use serde_json::Value;

use crate::act::Tanh;
use crate::config::{self, WaveNetConfig};
use crate::lstm::Lstm;
use crate::slim;
use crate::wavenet::{WaveNet, MAX_BLOCK};

/// What a `.nam` file says about the capture beyond its weights.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metadata {
    pub name: Option<String>,
    pub modeled_by: Option<String>,
    pub gear_make: Option<String>,
    pub gear_model: Option<String>,
    /// How loud the model plays a standard input, in dB: what NAM's
    /// plug-in levels its output by.
    pub loudness: Option<f64>,
}

impl Metadata {
    fn read(meta: &Value) -> Self {
        let text = |k: &str| meta[k].as_str().map(str::to_owned);
        Self {
            name: text("name"),
            modeled_by: text("modeled_by"),
            gear_make: text("gear_make"),
            gear_model: text("gear_model"),
            loudness: meta["loudness"].as_f64(),
        }
    }
}

/// Which of a capture's sizes runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Size {
    /// The whole capture: what NAM's core runs unless told otherwise.
    #[default]
    Full,
    /// The size a step below it, where the capture has one (A2-Lite, or a
    /// slimmable WaveNet slimmed); the full one where it has not.
    Light,
}

#[derive(Clone, Debug)]
enum Net {
    WaveNet(WaveNet),
    Lstm(Lstm),
}

/// One size of a capture.
#[derive(Clone, Debug)]
struct Variant {
    net: Net,
    /// The rate as the file gives it, if it does: NAM prewarms an LSTM
    /// that gives none for one sample only.
    given_rate: Option<f64>,
    loudness: Option<f64>,
    architecture: &'static str,
    /// The slim value it is NAM's choice for (`SetSlimmableSize`).
    value: f64,
}

impl Variant {
    fn prewarm_samples(&self) -> usize {
        match &self.net {
            Net::WaveNet(w) => w.prewarm_samples(),
            Net::Lstm(_) => Lstm::prewarm_samples(self.given_rate),
        }
    }

    fn restore(&mut self, from: &Self) -> bool {
        match (&mut self.net, &from.net) {
            (Net::WaveNet(a), Net::WaveNet(b)) => a.restore(b),
            (Net::Lstm(a), Net::Lstm(b)) => a.restore(b),
            _ => false,
        }
    }

    fn process_chunk(&mut self, input: &[f32], output: &mut [f32], tanh: Tanh) {
        match &mut self.net {
            Net::WaveNet(w) => w.process_block(input, output, tanh),
            Net::Lstm(l) => l.process_block(input, output, tanh),
        }
    }

    /// Forgets what it has heard and runs NAM's prewarm, in blocks of
    /// [`MAX_BLOCK`].
    fn reset(&mut self, tanh: Tanh) {
        match &mut self.net {
            Net::WaveNet(w) => w.clear(),
            Net::Lstm(l) => l.clear(),
        }
        let zeros = [0.0; MAX_BLOCK];
        let mut out = [0.0; MAX_BLOCK];
        let mut done = 0;
        let total = self.prewarm_samples();
        while done < total {
            self.process_chunk(&zeros, &mut out, tanh);
            done += MAX_BLOCK;
        }
    }

    fn macs_per_sample(&self) -> usize {
        match &self.net {
            Net::WaveNet(w) => w.macs_per_sample(),
            Net::Lstm(l) => l.macs_per_sample(),
        }
    }
}

/// A capture, ready to run.
#[derive(Clone)]
pub struct Model {
    full: Variant,
    light: Option<Variant>,
    size: Size,
    /// The rate it was trained at: it sounds right only at this rate.
    pub sample_rate: f64,
    pub metadata: Metadata,
    /// The full size's architecture.
    pub architecture: &'static str,
}

/// What it is, not its weights and histories: a capture holds tens of
/// thousands of numbers, and a command or an effect that carries one is
/// formatted by harnesses.
impl core::fmt::Debug for Model {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Model")
            .field("architecture", &self.architecture)
            .field("name", &self.metadata.name)
            .field("size", &self.size())
            .field("macs_per_sample", &self.macs_per_sample())
            .finish_non_exhaustive()
    }
}

impl Model {
    /// Reads a `.nam` file's text.
    ///
    /// # Errors
    /// Text that is not a `.nam` file, or a model this port does not run
    /// (the error says which part).
    pub fn parse(text: &str) -> Result<Self, String> {
        let json: Value =
            serde_json::from_str(text).map_err(|e| format!("not a .nam file: {e}"))?;
        if !json.is_object() {
            return Err("not a .nam file: it is JSON, but not a model".into());
        }
        let (full, light) = sizes(&json)?;
        let sample_rate = json["sample_rate"].as_f64().unwrap_or(48_000.0);
        let mut metadata = Metadata::read(&json["metadata"]);
        if metadata.loudness.is_none() {
            metadata.loudness = full.loudness;
        }
        let mut model = Self {
            architecture: full.architecture,
            full,
            light,
            size: Size::Full,
            sample_rate,
            metadata,
        };
        model.reset(Tanh::Fast);
        Ok(model)
    }

    /// Runs the size `size` from the next block. Allocates nothing; the
    /// size switched to plays on from wherever it was left (a model is
    /// prewarmed in both), so a switch while playing wants a
    /// [`Self::restore_size`] after it.
    pub fn set_size(&mut self, size: Size) {
        self.size = size;
    }

    /// Takes on `from`'s state in the size that runs only: what a switch
    /// to it puts back, as the core resets and prewarms the size it
    /// switches to. `false`, and nothing changed, for another network.
    pub fn restore_size(&mut self, from: &Self) -> bool {
        let size = self.size;
        match (self.light.as_mut(), from.light.as_ref(), size) {
            (Some(a), Some(b), Size::Light) => a.restore(b),
            _ => self.full.restore(&from.full),
        }
    }

    /// The size that runs.
    #[must_use]
    pub fn size(&self) -> Size {
        if self.light.is_some() {
            self.size
        } else {
            Size::Full
        }
    }

    /// The value NAM's `SetSlimmableSize` would be given to run the size
    /// that runs: 1 for the full size.
    #[must_use]
    pub fn slim_value(&self) -> f64 {
        self.active().value
    }

    /// Whether the capture has a lighter size than its full one.
    #[must_use]
    pub fn has_light(&self) -> bool {
        self.light.is_some()
    }

    fn active(&self) -> &Variant {
        self.variant(self.size)
    }

    fn variant(&self, size: Size) -> &Variant {
        match (&self.light, size) {
            (Some(l), Size::Light) => l,
            _ => &self.full,
        }
    }

    fn variant_mut(&mut self, size: Size) -> &mut Variant {
        match (&mut self.light, size) {
            (Some(l), Size::Light) => l,
            _ => &mut self.full,
        }
    }

    fn variants_mut(&mut self) -> impl Iterator<Item = &mut Variant> {
        std::iter::once(&mut self.full).chain(self.light.as_mut())
    }

    /// How loud the size that runs plays NAM's test signal, in dB, if the
    /// file says: each size of an A2 capture is measured on its own.
    #[must_use]
    pub fn loudness(&self) -> Option<f64> {
        self.loudness_in(self.size)
    }

    /// How loud `size` plays NAM's test signal, as [`Self::loudness`]
    /// says for the size that runs.
    #[must_use]
    pub fn loudness_in(&self, size: Size) -> Option<f64> {
        self.variant(size).loudness.or(self.metadata.loudness)
    }

    /// How many samples of silence NAM runs through a fresh model before
    /// it plays, so it starts where silence leaves it rather than where
    /// zeroed buffers do.
    #[must_use]
    pub fn prewarm_samples(&self) -> usize {
        self.active().prewarm_samples()
    }

    /// Takes on `from`'s state — a copy of the same capture, kept as it
    /// was just after its prewarm, puts this one back there — in every
    /// size, without allocating. `false`, and nothing changed, if `from`
    /// is another network.
    pub fn restore(&mut self, from: &Self) -> bool {
        let same = self.light.is_some() == from.light.is_some();
        if !same {
            return false;
        }
        if !self.full.restore(&from.full) {
            return false;
        }
        match (&mut self.light, &from.light) {
            (Some(a), Some(b)) => a.restore(b),
            _ => true,
        }
    }

    /// Forgets what it has heard and runs NAM's prewarm in every size, in
    /// blocks of [`MAX_BLOCK`], with `tanh`. Allocates nothing.
    /// [`Self::parse`] leaves a model prewarmed with [`Tanh::Fast`], as
    /// NAM's plug-in does.
    pub fn reset(&mut self, tanh: Tanh) {
        for v in self.variants_mut() {
            v.reset(tanh);
        }
    }

    /// Multiplies and adds per sample, in the size that runs.
    #[must_use]
    pub fn macs_per_sample(&self) -> usize {
        self.active().macs_per_sample()
    }

    /// Runs `input` into `output`, mono, as far as the shorter goes.
    pub fn process(&mut self, input: &[f32], output: &mut [f32], tanh: Tanh) {
        self.process_in(self.size, input, output, tanh);
    }

    /// [`Self::process`] in `size` rather than the size that runs: each
    /// size keeps its own state, so the size switched away from can play
    /// on beside the one switched to while one fades into the other. A
    /// capture with one size runs it whatever `size` says.
    pub fn process_in(&mut self, size: Size, input: &[f32], output: &mut [f32], tanh: Tanh) {
        let v = self.variant_mut(size);
        let n = input.len().min(output.len());
        for (i, o) in input[..n]
            .chunks(MAX_BLOCK)
            .zip(output[..n].chunks_mut(MAX_BLOCK))
        {
            v.process_chunk(i, o, tanh);
        }
    }
}

/// The largest LSTM size a config may give: past any capture people
/// train (32 hidden units is large).
const MAX_LSTM: u64 = 1024;

fn usize_of(config: &Value, key: &str) -> Result<usize, String> {
    let n = config[key]
        .as_u64()
        .ok_or_else(|| format!("the config has no {key}"))?;
    if n > MAX_LSTM {
        return Err(format!("the config's {key} is {n}, more than bloop runs"));
    }
    usize::try_from(n).map_err(|_| format!("the config's {key} is {n}, more than bloop runs"))
}

fn weights_of(json: &Value) -> Result<Vec<f32>, String> {
    json["weights"]
        .as_array()
        .ok_or("the file has no weights")?
        .iter()
        .map(|v| {
            v.as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| "a weight is not a number".to_string())
        })
        .collect()
}

/// The slim value in the middle of the stretch just below the top one:
/// what picks [`Size::Light`]. `None` when there is one size.
fn light_value(breakpoints: &[f64]) -> Option<f64> {
    let (&top, rest) = breakpoints.split_last()?;
    let below = rest.last().copied().unwrap_or(0.0);
    Some(f64::midpoint(below, top))
}

/// The full and the light size of a model's JSON.
fn sizes(json: &Value) -> Result<(Variant, Option<Variant>), String> {
    let architecture = json["architecture"]
        .as_str()
        .ok_or("the file names no architecture")?;
    if architecture != "SlimmableContainer" {
        return single(json);
    }
    let submodels = json["config"]["submodels"]
        .as_array()
        .filter(|s| !s.is_empty())
        .ok_or("the container holds no models")?;
    let mut held = Vec::with_capacity(submodels.len());
    for s in submodels {
        let max = s["max_value"]
            .as_f64()
            .ok_or("a contained model has no max_value")?;
        if held.last().is_some_and(|(m, _)| max <= *m) {
            return Err("the container's models are not in ascending max_value".into());
        }
        let model = &s["model"];
        if model["architecture"].as_str() == Some("SlimmableContainer") {
            return Err("bloop does not run a container inside a container".into());
        }
        held.push((max, model));
    }
    if held.last().is_some_and(|(m, _)| *m < 1.0) {
        return Err("the container's last model does not reach a max_value of 1".into());
    }
    let index_for = |value: f64| {
        held.iter()
            .position(|(max, _)| value < *max)
            .unwrap_or(held.len() - 1)
    };
    let rate = json["sample_rate"].as_f64();
    let read = |i: usize| -> Result<Variant, String> {
        let model = held[i].1;
        if let (Some(r), Some(m)) = (rate, model["sample_rate"].as_f64()) {
            if (r - m).abs() > f64::EPSILON {
                return Err(format!(
                    "the container plays at {r} Hz, but a model in it at {m} Hz"
                ));
            }
        }
        Ok(single(model)?.0)
    };
    // The core plays the last unless told otherwise.
    let full_index = held.len() - 1;
    let breakpoints: Vec<f64> = held[..held.len() - 1].iter().map(|(m, _)| *m).collect();
    let light_index = light_value(&breakpoints).map(index_for).filter(|&i| i != full_index);
    for i in 0..held.len() {
        if Some(i) != light_index && i != full_index {
            read(i)?;
        }
    }
    let light = match (light_index, light_value(&breakpoints)) {
        (Some(i), Some(value)) => Some(Variant {
            value,
            ..read(i)?
        }),
        _ => None,
    };
    Ok((read(full_index)?, light))
}

/// A model that is not a container: its full size, and its light one if
/// it is a slimmable WaveNet.
fn single(json: &Value) -> Result<(Variant, Option<Variant>), String> {
    let architecture = json["architecture"]
        .as_str()
        .ok_or("the file names no architecture")?;
    let config = &json["config"];
    let weights = weights_of(json)?;
    let given_rate = json["sample_rate"].as_f64();
    let loudness = json["metadata"]["loudness"].as_f64();
    let variant = |net: Net, architecture: &'static str| Variant {
        net,
        given_rate,
        loudness,
        architecture,
        value: 1.0,
    };
    match architecture {
        "WaveNet" => {
            let cfg = config::read(config)?;
            let full = variant(Net::WaveNet(wavenet(&cfg, &weights, given_rate)?), "WaveNet");
            if !cfg.slimmable() {
                return Ok((full, None));
            }
            // As the core does, a model that cannot be slimmed plays at
            // its full size: only slimming checks what slimming needs.
            let light = slim::check(&cfg).ok().and_then(|()| {
                let value = light_value(&slim::breakpoints(&cfg))?;
                let (slimmed, w) =
                    slim::slim(&cfg, &weights, &slim::channels_for(&cfg, value)).ok()?;
                let net = wavenet(&slimmed, &w, given_rate).ok()?;
                Some(Variant {
                    value,
                    ..variant(Net::WaveNet(net), "WaveNet")
                })
            });
            Ok((full, light))
        }
        "LSTM" => Ok((
            variant(
                Net::Lstm(Lstm::build(
                    usize_of(config, "num_layers")?,
                    usize_of(config, "input_size")?,
                    usize_of(config, "hidden_size")?,
                    &weights,
                )?),
                "LSTM",
            ),
            None,
        )),
        other => Err(format!("bloop does not run {other} models yet")),
    }
}

/// A WaveNet from its config and weights, its condition DSP built first.
fn wavenet(cfg: &WaveNetConfig, weights: &[f32], rate: Option<f64>) -> Result<WaveNet, String> {
    if cfg.in_channels != 1 {
        return Err(format!(
            "the model takes {} input channels; bloop runs mono models",
            cfg.in_channels
        ));
    }
    let out = cfg
        .head
        .as_ref()
        .map_or_else(|| cfg.arrays.last().map_or(0, |a| a.head_size), |h| h.out_channels);
    if out != 1 {
        return Err(format!(
            "the model outputs {out} channels; bloop runs mono models"
        ));
    }
    let condition = match &cfg.condition_dsp {
        None => None,
        Some(dsp) => Some(condition_dsp(dsp, rate)?),
    };
    WaveNet::build(&cfg.arrays, cfg.head.as_ref(), weights, condition)
}

/// A condition DSP: a WaveNet of its own, of any number of outputs, at
/// the rate of the model it conditions.
fn condition_dsp(json: &Value, rate: Option<f64>) -> Result<WaveNet, String> {
    if json["architecture"].as_str() != Some("WaveNet") {
        return Err("bloop runs only WaveNets as a condition DSP".into());
    }
    let own = json["sample_rate"].as_f64();
    if let (Some(a), Some(b)) = (rate, own) {
        if (a - b).abs() > f64::EPSILON {
            return Err(format!(
                "the condition DSP plays at {b} Hz, but its model at {a} Hz"
            ));
        }
    }
    let cfg = config::read(&json["config"])?;
    if cfg.slimmable() || cfg.in_channels != 1 {
        return Err("bloop runs only plain mono WaveNets as a condition DSP".into());
    }
    let condition = match &cfg.condition_dsp {
        None => None,
        Some(dsp) => Some(condition_dsp(dsp, own)?),
    };
    WaveNet::build(&cfg.arrays, cfg.head.as_ref(), &weights_of(json)?, condition)
}
