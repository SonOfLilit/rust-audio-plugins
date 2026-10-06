//! A WaveNet's `config`, read the way NeuralAmpModelerCore reads it
//! (`wavenet::parse_config_json`, `NAM/wavenet/model.cpp`): the A1 keys
//! (`kernel_size`, `gated`, `head_size`, `head_bias`) and the A2 ones that
//! replace or extend them (`kernel_sizes`, `bottleneck`, `gating_mode`,
//! `secondary_activation`, `head`, `layer1x1`, `head1x1`, the groups, the
//! eight `FiLM`s, a post-stack `head`, a `condition_dsp`, `slimmable`),
//! with the same defaults.

use serde_json::Value;

use crate::act::Activation;

/// How a layer's activation is applied.
#[derive(Clone, Debug, PartialEq)]
pub enum Gating {
    /// The activation over the convolution's channels.
    None,
    /// The convolution makes twice the channels: the activation of the
    /// first half times this second activation of the second.
    Gated(Activation),
    /// Twice the channels: the activation of the first half blended with
    /// the first half as it was, by this second activation of the second
    /// (`alpha · act(a) + (1 − alpha) · a`).
    Blended(Activation),
}

impl Gating {
    /// How many channels the layer's convolution makes for a bottleneck
    /// of `bottleneck`.
    #[must_use]
    pub fn z_channels(&self, bottleneck: usize) -> usize {
        match self {
            Self::None => bottleneck,
            Self::Gated(_) | Self::Blended(_) => 2 * bottleneck,
        }
    }
}

/// One of the `FiLM`s a layer may have: a 1x1 of the condition into a
/// scale per channel (and a shift, with `shift`), in `groups`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilmConfig {
    pub shift: bool,
    pub groups: usize,
}

/// Where in a layer a `FiLM` sits, in the order NAM reads their weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Films {
    pub conv_pre: Option<FilmConfig>,
    pub conv_post: Option<FilmConfig>,
    pub input_mixin_pre: Option<FilmConfig>,
    pub input_mixin_post: Option<FilmConfig>,
    pub activation_pre: Option<FilmConfig>,
    pub activation_post: Option<FilmConfig>,
    pub layer1x1_post: Option<FilmConfig>,
    pub head1x1_post: Option<FilmConfig>,
}

/// One layer array, as the file describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayConfig {
    pub input_size: usize,
    pub condition_size: usize,
    /// What the array hands on as its head, after its head convolution.
    pub head_size: usize,
    pub head_kernel_size: usize,
    pub head_dilation: usize,
    pub head_bias: bool,
    pub channels: usize,
    pub bottleneck: usize,
    pub kernel_sizes: Vec<usize>,
    pub dilations: Vec<usize>,
    pub activations: Vec<Activation>,
    pub gatings: Vec<Gating>,
    pub groups_input: usize,
    pub groups_input_mixin: usize,
    /// The 1x1 from the bottleneck back to the channels, added to the
    /// layer's input for the next layer: its groups, or `None` when the
    /// layer hands its input on unchanged.
    pub layer1x1: Option<usize>,
    /// A 1x1 from the bottleneck into what the layer adds to the head:
    /// its outputs and groups. `None`: the activation itself is added.
    pub head1x1: Option<(usize, usize)>,
    pub films: Films,
    /// The channel counts a slimmable array may run at, ascending, the
    /// last its full count (`slice_channels_uniform`); empty if it is
    /// not slimmable.
    pub slimmable: Vec<usize>,
}

impl ArrayConfig {
    /// How many channels each layer adds to the head.
    #[must_use]
    pub fn head_inputs(&self) -> usize {
        self.head1x1.map_or(self.bottleneck, |(out, _)| out)
    }
}

/// The post-stack head: an activation then a convolution, per kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadConfig {
    pub in_channels: usize,
    pub channels: usize,
    pub out_channels: usize,
    pub kernel_sizes: Vec<usize>,
    pub activation: Activation,
}

/// A WaveNet's whole configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveNetConfig {
    pub in_channels: usize,
    pub arrays: Vec<ArrayConfig>,
    pub head: Option<HeadConfig>,
    /// The model a WaveNet with a condition DSP runs its input through
    /// to make its condition: the `.nam` JSON, read by the caller.
    pub condition_dsp: Option<Value>,
}

impl WaveNetConfig {
    /// Whether any array is slimmable.
    #[must_use]
    pub fn slimmable(&self) -> bool {
        self.arrays.iter().any(|a| !a.slimmable.is_empty())
    }
}

fn count(value: &Value, what: &str) -> Result<usize, String> {
    value
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("the WaveNet's {what} is not a count"))
}

fn key(obj: &Value, k: &str, i: usize) -> Result<usize, String> {
    count(&obj[k], &format!("layer array {i}'s {k}"))
}

fn flag(obj: &Value, k: &str, i: usize) -> Result<bool, String> {
    obj[k]
        .as_bool()
        .ok_or_else(|| format!("layer array {i}'s {k} is not true or false"))
}

fn counts(value: &Value, what: &str) -> Result<Vec<usize>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|v| count(v, what))
        .collect()
}

/// `parse_film_params`: absent or `false` is off; an object is on unless
/// it says `"active": false`, shifting unless `"shift": false`.
fn film(layer: &Value, k: &str) -> Option<FilmConfig> {
    let v = layer.get(k)?;
    if v.is_null() || *v == Value::Bool(false) {
        return None;
    }
    if !v.get("active").and_then(Value::as_bool).unwrap_or(true) {
        return None;
    }
    Some(FilmConfig {
        shift: v.get("shift").and_then(Value::as_bool).unwrap_or(true),
        groups: v
            .get("groups")
            .and_then(Value::as_u64)
            .map_or(1, |g| g as usize),
    })
}

/// One value per layer: a list of exactly `n`, or one value for all.
fn per_layer<T: Clone>(
    value: &Value,
    n: usize,
    what: &str,
    i: usize,
    read: impl Fn(&Value) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    if let Some(list) = value.as_array() {
        if list.len() != n {
            return Err(format!(
                "layer array {i} has {} {what}s for {n} layers",
                list.len()
            ));
        }
        list.iter().map(read).collect()
    } else {
        Ok(vec![read(value)?; n])
    }
}

fn gating_mode(name: &Value, i: usize) -> Result<&str, String> {
    match name.as_str() {
        Some(m @ ("none" | "gated" | "blended")) => Ok(m),
        _ => Err(format!("layer array {i} has a gating mode NAM does not have")),
    }
}

fn array(layer: &Value, i: usize) -> Result<ArrayConfig, String> {
    if !layer.is_object() {
        return Err(format!("layer array {i} is not an object"));
    }
    let channels = key(layer, "channels", i)?;
    let bottleneck = match layer.get("bottleneck") {
        Some(b) => count(b, &format!("layer array {i}'s bottleneck"))?,
        None => channels,
    };
    let group = |k: &str| match layer.get(k) {
        Some(g) => count(g, &format!("layer array {i}'s {k}")),
        None => Ok(1),
    };
    let groups_input = group("groups_input")?;
    let groups_input_mixin = group("groups_input_mixin")?;
    let layer1x1 = match layer.get("layer1x1") {
        Some(l) => {
            if flag(l, "active", i)? {
                Some(key(l, "groups", i)?)
            } else {
                None
            }
        }
        None => Some(1),
    };
    let (head_size, head_kernel_size, head_dilation, head_bias) =
        match layer.get("head").filter(|h| !h.is_null()) {
            Some(head) => {
                if !head.is_object() {
                    return Err(format!("layer array {i}'s head is not an object"));
                }
                let dilation = match head.get("head_dilation") {
                    Some(d) => count(d, &format!("layer array {i}'s head dilation"))?,
                    None => 1,
                };
                (
                    key(head, "out_channels", i)?,
                    key(head, "kernel_size", i)?,
                    dilation,
                    flag(head, "bias", i)?,
                )
            }
            None => (
                key(layer, "head_size", i).map_err(|_| {
                    format!("layer array {i} has neither a head nor a head_size")
                })?,
                1,
                1,
                flag(layer, "head_bias", i)?,
            ),
        };
    let dilations = counts(&layer["dilations"], &format!("layer array {i}'s dilations"))?;
    let n = dilations.len();
    let kernel_sizes = match (layer.get("kernel_size"), layer.get("kernel_sizes")) {
        (Some(_), Some(_)) => {
            return Err(format!(
                "layer array {i} gives both a kernel_size and kernel_sizes"
            ))
        }
        (Some(k), None) => vec![count(k, &format!("layer array {i}'s kernel_size"))?; n],
        (None, Some(ks)) => {
            let ks = counts(ks, &format!("layer array {i}'s kernel_sizes"))?;
            if ks.len() != n {
                return Err(format!(
                    "layer array {i} has {} kernel sizes for {n} layers",
                    ks.len()
                ));
            }
            ks
        }
        (None, None) => return Err(format!("layer array {i} has no kernel size")),
    };
    let activations = per_layer(&layer["activation"], n, "activation", i, Activation::from_json)?;
    let sigmoid = || Activation::Sigmoid;
    let secondary = layer.get("secondary_activation");
    let gatings = if let Some(modes) = layer.get("gating_mode") {
        let modes = per_layer(modes, n, "gating mode", i, |m| {
            gating_mode(m, i).map(str::to_owned)
        })?;
        if let Some(list) = secondary.and_then(Value::as_array) {
            if list.len() != n {
                return Err(format!(
                    "layer array {i} has {} secondary activations for {n} layers",
                    list.len()
                ));
            }
        }
        modes
            .iter()
            .enumerate()
            .map(|(l, mode)| {
                let second = || -> Result<Activation, String> {
                    match secondary {
                        None => Ok(sigmoid()),
                        Some(Value::Array(list)) => Activation::from_json(&list[l]),
                        Some(one) => Activation::from_json(one),
                    }
                };
                Ok(match mode.as_str() {
                    "gated" => Gating::Gated(second()?),
                    "blended" => Gating::Blended(second()?),
                    _ => Gating::None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    } else if layer.get("gated").and_then(Value::as_bool).unwrap_or(false) {
        vec![Gating::Gated(sigmoid()); n]
    } else {
        vec![Gating::None; n]
    };
    let head1x1 = match layer.get("head1x1") {
        Some(h) if flag(h, "active", i)? => {
            Some((key(h, "out_channels", i)?, key(h, "groups", i)?))
        }
        _ => None,
    };
    let films = Films {
        conv_pre: film(layer, "conv_pre_film"),
        conv_post: film(layer, "conv_post_film"),
        input_mixin_pre: film(layer, "input_mixin_pre_film"),
        input_mixin_post: film(layer, "input_mixin_post_film"),
        activation_pre: film(layer, "activation_pre_film"),
        activation_post: film(layer, "activation_post_film"),
        layer1x1_post: film(layer, "layer1x1_post_film"),
        head1x1_post: film(layer, "head1x1_post_film"),
    };
    if films.layer1x1_post.is_some() && layer1x1.is_none() {
        return Err(format!(
            "layer array {i} has a FiLM after a layer 1x1 it does not have"
        ));
    }
    if films.head1x1_post.is_some() && head1x1.is_none() {
        return Err(format!(
            "layer array {i} has a FiLM after a head 1x1 it does not have"
        ));
    }
    let slimmable = match layer.get("slimmable").filter(|s| s.is_object()) {
        None => Vec::new(),
        Some(s) => {
            let method = s.get("method").and_then(Value::as_str).unwrap_or("");
            if method != "slice_channels_uniform" {
                return Err(format!(
                    "layer array {i} slims by \"{method}\", which bloop does not run"
                ));
            }
            match s.get("kwargs").and_then(|k| k.get("allowed_channels")) {
                Some(list) => counts(list, &format!("layer array {i}'s allowed channels"))?,
                None => (1..=channels).collect(),
            }
        }
    };
    Ok(ArrayConfig {
        input_size: key(layer, "input_size", i)?,
        condition_size: key(layer, "condition_size", i)?,
        head_size,
        head_kernel_size,
        head_dilation,
        head_bias,
        channels,
        bottleneck,
        kernel_sizes,
        dilations,
        activations,
        gatings,
        groups_input,
        groups_input_mixin,
        layer1x1,
        head1x1,
        films,
        slimmable,
    })
}

/// Reads a WaveNet's `config`.
///
/// # Errors
/// A key missing or of the wrong kind, or lists that disagree on how many
/// layers there are.
pub fn read(config: &Value) -> Result<WaveNetConfig, String> {
    let layers = config["layers"]
        .as_array()
        .ok_or("the WaveNet has no layers")?;
    if layers.is_empty() {
        return Err("the WaveNet has no layers".into());
    }
    let arrays = layers
        .iter()
        .enumerate()
        .map(|(i, l)| array(l, i))
        .collect::<Result<Vec<_>, _>>()?;
    let in_channels = match config.get("in_channels") {
        Some(c) => count(c, "in_channels")?,
        None => 1,
    };
    let head = match config.get("head").filter(|h| !h.is_null()) {
        None => None,
        Some(h) => {
            let implied = arrays.last().map_or(0, |a| a.head_size);
            if let Some(given) = h.get("in_channels").filter(|v| !v.is_null()) {
                if count(given, "head's in_channels")? != implied {
                    return Err(format!(
                        "the WaveNet's head takes {given} channels, but its last layer array hands on {implied}"
                    ));
                }
            }
            let kernel_sizes = counts(&h["kernel_sizes"], "the head's kernel_sizes")?;
            if kernel_sizes.is_empty() {
                return Err("the WaveNet's head has no kernels".into());
            }
            Some(HeadConfig {
                in_channels: implied,
                channels: count(&h["channels"], "head's channels")?,
                out_channels: count(&h["out_channels"], "head's out_channels")?,
                kernel_sizes,
                activation: Activation::from_json(&h["activation"])?,
            })
        }
    };
    Ok(WaveNetConfig {
        in_channels,
        arrays,
        head,
        condition_dsp: config
            .get("condition_dsp")
            .filter(|c| !c.is_null())
            .cloned(),
    })
}
