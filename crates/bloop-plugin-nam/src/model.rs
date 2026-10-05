//! A `.nam` file: its JSON read into a network ready to run.
//!
//! Read on the UI thread (it allocates); [`Model::process`] allocates
//! nothing.

use serde_json::Value;

use crate::act::{Activation, Tanh};
use crate::lstm::Lstm;
use crate::wavenet::{ArrayConfig, Gating, WaveNet, MAX_BLOCK};

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

#[derive(Clone, Debug)]
enum Net {
    WaveNet(WaveNet),
    Lstm(Lstm),
}

/// A capture, ready to run.
#[derive(Clone, Debug)]
pub struct Model {
    net: Net,
    /// The rate it was trained at: it sounds right only at this rate.
    pub sample_rate: f64,
    pub metadata: Metadata,
    pub architecture: &'static str,
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
        let architecture = json["architecture"]
            .as_str()
            .ok_or("the file names no architecture")?;
        let config = &json["config"];
        let weights: Vec<f32> = json["weights"]
            .as_array()
            .ok_or("the file has no weights")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|x| x as f32)
                    .ok_or("a weight is not a number")
            })
            .collect::<Result<_, _>>()?;
        let sample_rate = json["sample_rate"].as_f64().unwrap_or(48_000.0);
        let (net, architecture) = match architecture {
            "WaveNet" => (Net::WaveNet(wavenet(config, &weights)?), "WaveNet"),
            "LSTM" => (
                Net::Lstm(Lstm::build(
                    usize_of(config, "num_layers")?,
                    usize_of(config, "input_size")?,
                    usize_of(config, "hidden_size")?,
                    &weights,
                )?),
                "LSTM",
            ),
            other => return Err(format!("bloop does not run {other} models yet")),
        };
        let meta = &json["metadata"];
        let text = |k: &str| meta[k].as_str().map(str::to_owned);
        let metadata = Metadata {
            name: text("name"),
            modeled_by: text("modeled_by"),
            gear_make: text("gear_make"),
            gear_model: text("gear_model"),
            loudness: meta["loudness"].as_f64(),
        };
        let mut model = Self {
            net,
            sample_rate,
            metadata,
            architecture,
        };
        model.reset(Tanh::Fast);
        Ok(model)
    }

    /// How many samples of silence NAM runs through a fresh model before
    /// it plays, so it starts where silence leaves it rather than where
    /// zeroed buffers do.
    #[must_use]
    pub fn prewarm_samples(&self) -> usize {
        match &self.net {
            Net::WaveNet(w) => w.prewarm_samples(),
            Net::Lstm(_) => Lstm::prewarm_samples(self.sample_rate),
        }
    }

    /// Takes on `from`'s state — a copy of the same capture, kept as it
    /// was just after its prewarm, puts this one back there — without
    /// allocating. `false`, and nothing changed, if `from` is another
    /// network.
    pub fn restore(&mut self, from: &Self) -> bool {
        match (&mut self.net, &from.net) {
            (Net::WaveNet(a), Net::WaveNet(b)) => a.restore(b),
            (Net::Lstm(a), Net::Lstm(b)) => a.restore(b),
            _ => false,
        }
    }

    /// Forgets what it has heard and runs NAM's prewarm, in blocks of
    /// [`MAX_BLOCK`], with `tanh`. Allocates nothing; costs
    /// [`Self::prewarm_samples`] samples of processing. [`Self::parse`]
    /// leaves a model prewarmed with [`Tanh::Fast`], as NAM's plug-in does.
    pub fn reset(&mut self, tanh: Tanh) {
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

    /// Multiplies and adds per sample.
    #[must_use]
    pub fn macs_per_sample(&self) -> usize {
        match &self.net {
            Net::WaveNet(w) => w.macs_per_sample(),
            Net::Lstm(l) => l.macs_per_sample(),
        }
    }

    fn process_chunk(&mut self, input: &[f32], output: &mut [f32], tanh: Tanh) {
        match &mut self.net {
            Net::WaveNet(w) => w.process_block(input, output, tanh),
            Net::Lstm(l) => l.process_block(input, output, tanh),
        }
    }

    /// Runs `input` into `output` (the same length), mono.
    pub fn process(&mut self, input: &[f32], output: &mut [f32], tanh: Tanh) {
        for (i, o) in input.chunks(MAX_BLOCK).zip(output.chunks_mut(MAX_BLOCK)) {
            self.process_chunk(i, o, tanh);
        }
    }
}

fn usize_of(config: &Value, key: &str) -> Result<usize, String> {
    config[key]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("the config has no {key}"))
}

/// The keys of a layer array bloop's WaveNet reads. A file with any other
/// is a newer architecture (NAM's A2: `FiLM`, bottlenecks, grouped
/// convolutions, per-layer kernels) this port does not run yet, and is
/// refused rather than run wrong.
const ARRAY_KEYS: [&str; 9] = [
    "input_size",
    "condition_size",
    "head_size",
    "channels",
    "kernel_size",
    "dilations",
    "activation",
    "gated",
    "head_bias",
];

fn wavenet(config: &Value, weights: &[f32]) -> Result<WaveNet, String> {
    if !config["head"].is_null() {
        return Err("bloop does not run WaveNets with a post-stack head yet".into());
    }
    if !config["condition_dsp"].is_null() {
        return Err("bloop does not run WaveNets with a condition DSP yet".into());
    }
    let arrays = config["layers"]
        .as_array()
        .ok_or("the WaveNet has no layers")?;
    let mut configs = Vec::with_capacity(arrays.len());
    for (i, a) in arrays.iter().enumerate() {
        let obj = a.as_object().ok_or("a layer array is not an object")?;
        if let Some(key) = obj.keys().find(|k| !ARRAY_KEYS.contains(&k.as_str())) {
            return Err(format!(
                "layer array {i} uses \"{key}\", which bloop does not run yet"
            ));
        }
        let activation_name = a["activation"]
            .as_str()
            .ok_or_else(|| format!("layer array {i}'s activation is not a name"))?;
        let activation = Activation::named(activation_name)
            .ok_or_else(|| format!("bloop does not run the {activation_name} activation yet"))?;
        let dilations = a["dilations"]
            .as_array()
            .ok_or_else(|| format!("layer array {i} has no dilations"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|d| d as usize)
                    .ok_or("a dilation is not a number")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dilations.is_empty() || dilations.contains(&0) {
            return Err(format!("layer array {i} has no layers, or a dilation of 0"));
        }
        let kernel_size = usize_of(a, "kernel_size")?;
        if kernel_size == 0 {
            return Err(format!("layer array {i} has a kernel of 0"));
        }
        configs.push(ArrayConfig {
            input_size: usize_of(a, "input_size")?,
            condition_size: usize_of(a, "condition_size")?,
            head_size: usize_of(a, "head_size")?,
            channels: usize_of(a, "channels")?,
            kernel_size,
            dilations,
            activation,
            gating: if a["gated"].as_bool().unwrap_or(false) {
                Gating::Gated(Activation::Sigmoid)
            } else {
                Gating::None
            },
            head_bias: a["head_bias"].as_bool().unwrap_or(false),
        });
    }
    WaveNet::build(&configs, weights)
}
