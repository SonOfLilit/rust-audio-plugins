//! NAM's WaveNet, the architecture of nearly every `.nam` capture: the
//! standard, lite, feather and nano sizes the trainer offers are all this,
//! with more or fewer channels and layers (`NAM/wavenet/model.cpp`).
//!
//! A model is a stack of layer arrays. Each array re-channels its input,
//! then runs a layer per dilation: a dilated causal convolution of the
//! layer's input plus a 1x1 mix of the dry input (the "condition"), an
//! activation (gated or not), a 1x1 back to the layer's channels added to
//! its input for the next layer, and the activation itself summed into the
//! array's head. The head is re-channelled and handed on as the next
//! array's starting sum; the last array's head, scaled, is the output.
//!
//! Frames are processed a block at a time and stored channel by channel:
//! each channel's block of frames is one contiguous row. Every matrix
//! product is then a run of `row += weight × row` over a whole block, the
//! loop the compiler turns into NEON on a phone and SSE here, which is
//! several times faster than multiplying a frame's handful of channels at
//! a time. The C++ multiplies frame by frame through Eigen, so its sums
//! are added in another order: the two agree to float rounding, not bit
//! for bit (`tests/gold.rs`).

use crate::act::{Activation, Tanh};

/// The most frames processed in one pass; a longer block is split.
pub const MAX_BLOCK: usize = 128;

/// The most channels a layer may have (a gated layer's convolution makes
/// twice this). The trainer's largest preset has 16.
pub const MAX_CHANNELS: usize = 64;

/// `channels` rows of [`MAX_BLOCK`] frames.
#[derive(Clone, Debug)]
struct Rows {
    data: Vec<f32>,
}

impl Rows {
    fn new(channels: usize) -> Self {
        Self {
            data: vec![0.0; channels * MAX_BLOCK],
        }
    }

    #[inline]
    fn row(&self, ch: usize, frames: usize) -> &[f32] {
        &self.data[ch * MAX_BLOCK..ch * MAX_BLOCK + frames]
    }

    #[inline]
    fn row_mut(&mut self, ch: usize, frames: usize) -> &mut [f32] {
        &mut self.data[ch * MAX_BLOCK..ch * MAX_BLOCK + frames]
    }
}

/// `a + w · x`. On a phone (aarch64) one fused multiply-add, which every
/// `ARMv8` core has: half the instructions of a multiply then an add, in the
/// loop all of a model's time goes to. Fused, the product is not rounded
/// before the add, so a phone's output differs from this machine's in the
/// last bits — well inside what `tests/gold.rs` holds the port to. Elsewhere
/// (x86 without a guaranteed FMA unit, and wasm, whose renders the tests
/// compare against native ones) the multiply and the add stay separate.
#[inline]
fn madd(a: f32, w: f32, x: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        w.mul_add(x, a)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a + w * x
    }
}

/// How many frames of one output row are summed in registers at once.
const LANES: usize = 32;

/// One output row of a matrix product, `frames` long:
///
/// `row[f] = bias + scale · init[f] + Σₜ w[t] · data[start(t) + f]`
///
/// `init` is the condition a layer mixes in or the residual it adds to.
/// The row is summed [`LANES`] frames at a time in an array the compiler
/// keeps in registers, so each weight costs one load and one
/// multiply-add per vector of frames rather than a load and a store of
/// the row as well.
#[inline]
fn dense_row(
    row: &mut [f32],
    bias: f32,
    init: Option<(f32, &[f32])>,
    w: &[f32],
    data: &[f32],
    start: impl Fn(usize) -> usize,
) {
    let frames = row.len();
    let mut s = 0;
    while s + LANES <= frames {
        let mut acc = [bias; LANES];
        if let Some((scale, init)) = init {
            if let Some(x) = init[s..].first_chunk::<LANES>() {
                for (a, x) in acc.iter_mut().zip(x) {
                    *a = madd(*a, scale, *x);
                }
            }
        }
        for (t, &w) in w.iter().enumerate() {
            if let Some(x) = data[start(t) + s..].first_chunk::<LANES>() {
                for (a, x) in acc.iter_mut().zip(x) {
                    *a = madd(*a, w, *x);
                }
            }
        }
        row[s..s + LANES].copy_from_slice(&acc);
        s += LANES;
    }
    for f in s..frames {
        let mut acc = bias;
        if let Some((scale, init)) = init {
            acc = madd(acc, scale, init[f]);
        }
        for (t, &w) in w.iter().enumerate() {
            acc = madd(acc, w, data[start(t) + f]);
        }
        row[f] = acc;
    }
}

/// A 1x1 convolution: a matrix, `outputs` by `inputs`, and maybe a bias.
#[derive(Clone, Debug)]
pub(crate) struct Conv1x1 {
    inputs: usize,
    outputs: usize,
    /// Row-major `[output][input]`, as NAM lays it out.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Conv1x1 {
    /// Reads its weights as NAM lays them out: `[output][input]`, then the
    /// bias.
    pub(crate) fn read(
        inputs: usize,
        outputs: usize,
        bias: bool,
        weights: &mut Weights,
    ) -> Result<Self, String> {
        let weight = weights.take(inputs * outputs)?;
        let bias = if bias {
            Some(weights.take(outputs)?)
        } else {
            None
        };
        Ok(Self {
            inputs,
            outputs,
            weight,
            bias,
        })
    }

    /// `out = W x (+ b)` over `frames` frames.
    fn apply(&self, x: &Rows, out: &mut Rows, frames: usize) {
        let n = self.inputs;
        for o in 0..self.outputs {
            let bias = self.bias.as_ref().map_or(0.0, |b| b[o]);
            dense_row(
                out.row_mut(o, frames),
                bias,
                None,
                &self.weight[o * n..(o + 1) * n],
                &x.data,
                |i| i * MAX_BLOCK,
            );
        }
    }
}

/// The weights of a `.nam` file, read in order.
pub(crate) struct Weights<'a> {
    values: &'a [f32],
    at: usize,
}

impl<'a> Weights<'a> {
    pub(crate) fn new(values: &'a [f32]) -> Self {
        Self { values, at: 0 }
    }

    pub(crate) fn next(&mut self) -> Result<f32, String> {
        let v = self.values.get(self.at).copied().ok_or_else(|| {
            format!(
                "the model has {} weights, fewer than its layers need",
                self.values.len()
            )
        })?;
        self.at += 1;
        Ok(v)
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<Vec<f32>, String> {
        (0..n).map(|_| self.next()).collect()
    }

    pub(crate) fn finish(&self) -> Result<(), String> {
        if self.at == self.values.len() {
            Ok(())
        } else {
            Err(format!(
                "the model has {} weights, but its layers use {}",
                self.values.len(),
                self.at
            ))
        }
    }
}

/// What a layer has heard, for its dilated convolution to look back over:
/// per channel, `lookback` frames before the block being processed, then
/// the block. When a row fills, its last `lookback` frames move to the
/// front (NAM's `RingBuffer` does the same).
#[derive(Clone, Debug)]
struct History {
    channels: usize,
    lookback: usize,
    capacity: usize,
    data: Vec<f32>,
    /// The frame the next block is written at.
    at: usize,
}

impl History {
    fn new(channels: usize, lookback: usize) -> Self {
        let capacity = 2 * lookback + MAX_BLOCK;
        Self {
            channels,
            lookback,
            capacity,
            data: vec![0.0; capacity * channels],
            at: lookback,
        }
    }

    fn clear(&mut self) {
        self.data.fill(0.0);
        self.at = self.lookback;
    }

    /// Appends `frames` frames of `block`, making room first if needed.
    fn write(&mut self, block: &Rows, frames: usize) {
        let cap = self.capacity;
        if self.at + frames > cap {
            for ch in 0..self.channels {
                let row = &mut self.data[ch * cap..(ch + 1) * cap];
                row.copy_within(self.at - self.lookback..self.at, 0);
            }
            self.at = self.lookback;
        }
        for ch in 0..self.channels {
            let start = ch * cap + self.at;
            self.data[start..start + frames].copy_from_slice(block.row(ch, frames));
        }
    }

    /// Done with the block just written.
    fn advance(&mut self, frames: usize) {
        self.at += frames;
    }
}

/// How a layer's activation is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Gating {
    /// The activation over all the convolution's channels.
    None,
    /// The convolution makes twice the channels: the activation of the
    /// first half times this second activation (a sigmoid) of the second.
    Gated(Activation),
}

#[derive(Clone, Debug)]
struct Layer {
    channels: usize,
    kernel: usize,
    dilation: usize,
    activation: Activation,
    gating: Gating,
    /// `[output][tap][input]`, tap 0 the furthest back.
    conv: Vec<f32>,
    conv_bias: Vec<f32>,
    /// The condition's 1x1, no bias: one weight per output.
    mixin: Vec<f32>,
    /// Back to the layer's channels, with a bias.
    out: Conv1x1,
    history: History,
    /// Where in `history` each weight of an output's row of `conv` reads,
    /// past the start of the lookback: channel `i`'s row, `back` frames
    /// before the block, for weight `tap · channels + i`.
    offsets: Vec<usize>,
    /// The pre-activation sum, then the activation.
    z: Rows,
}

impl Layer {
    fn z_channels(&self) -> usize {
        match self.gating {
            Gating::None => self.channels,
            Gating::Gated(_) => 2 * self.channels,
        }
    }

    /// One block: reads `input`, adds the activation to `head`, writes
    /// `input + out(activation)` to `next`.
    fn process(
        &mut self,
        input: &Rows,
        condition: &[f32],
        head: &mut Rows,
        next: &mut Rows,
        frames: usize,
        tanh: Tanh,
    ) {
        let c = self.channels;
        let zc = self.z_channels();
        self.history.write(input, frames);
        let kc = self.kernel * c;
        let base = self.history.at - self.history.lookback;
        let offsets = &self.offsets;
        for o in 0..zc {
            dense_row(
                self.z.row_mut(o, frames),
                self.conv_bias[o],
                Some((self.mixin[o], condition)),
                &self.conv[o * kc..(o + 1) * kc],
                &self.history.data,
                |t| base + offsets[t],
            );
        }
        self.history.advance(frames);
        match self.gating {
            Gating::None => {
                for o in 0..c {
                    self.activation.apply(self.z.row_mut(o, frames), tanh);
                }
            }
            Gating::Gated(secondary) => {
                let (top, bottom) = self.z.data.split_at_mut(c * MAX_BLOCK);
                for o in 0..c {
                    let t = &mut top[o * MAX_BLOCK..o * MAX_BLOCK + frames];
                    let b = &mut bottom[o * MAX_BLOCK..o * MAX_BLOCK + frames];
                    self.activation.apply(t, tanh);
                    secondary.apply(b, tanh);
                    for (t, b) in t.iter_mut().zip(b.iter()) {
                        *t *= *b;
                    }
                }
            }
        }
        let z = &self.z;
        for o in 0..c {
            for (h, a) in head.row_mut(o, frames).iter_mut().zip(z.row(o, frames)) {
                *h += *a;
            }
            dense_row(
                next.row_mut(o, frames),
                self.out.bias.as_ref().map_or(0.0, |b| b[o]),
                Some((1.0, input.row(o, frames))),
                &self.out.weight[o * c..(o + 1) * c],
                &z.data,
                |i| i * MAX_BLOCK,
            );
        }
    }
}

#[derive(Clone, Debug)]
struct LayerArray {
    channels: usize,
    rechannel: Conv1x1,
    layers: Vec<Layer>,
    head_rechannel: Conv1x1,
    /// The layers' inputs and outputs, ping-pong.
    a: Rows,
    b: Rows,
    /// The head's running sum.
    head: Rows,
    /// The head re-channelled.
    head_out: Rows,
}

impl LayerArray {
    fn receptive_field(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.kernel - 1) * l.dilation)
            .sum()
    }

    /// Runs the block. `input` has `rechannel.inputs` rows; `head_in` is
    /// the previous array's head (`None` for the first). Leaves the
    /// array's output in `a` and its head in `head_out`.
    fn process(
        &mut self,
        input: &Rows,
        condition: &[f32],
        head_in: Option<&Rows>,
        frames: usize,
        tanh: Tanh,
    ) {
        let c = self.channels;
        match head_in {
            Some(h) => self.head.data[..c * MAX_BLOCK].copy_from_slice(&h.data[..c * MAX_BLOCK]),
            None => self.head.data.fill(0.0),
        }
        self.rechannel.apply(input, &mut self.a, frames);
        for layer in &mut self.layers {
            layer.process(
                &self.a,
                condition,
                &mut self.head,
                &mut self.b,
                frames,
                tanh,
            );
            std::mem::swap(&mut self.a, &mut self.b);
        }
        self.head_rechannel
            .apply(&self.head, &mut self.head_out, frames);
    }
}

/// One layer array's shape, as the `.nam` file gives it.
#[derive(Clone, Debug)]
pub(crate) struct ArrayConfig {
    pub input_size: usize,
    pub condition_size: usize,
    pub head_size: usize,
    pub channels: usize,
    pub kernel_size: usize,
    pub dilations: Vec<usize>,
    pub activation: Activation,
    pub gating: Gating,
    pub head_bias: bool,
}

/// A WaveNet model, ready to run.
#[derive(Clone, Debug)]
pub struct WaveNet {
    arrays: Vec<LayerArray>,
    head_scale: f32,
    /// The input block as one row.
    input: Rows,
}

impl WaveNet {
    pub(crate) fn build(configs: &[ArrayConfig], weights: &[f32]) -> Result<Self, String> {
        let mut w = Weights::new(weights);
        let mut arrays = Vec::with_capacity(configs.len());
        for (i, cfg) in configs.iter().enumerate() {
            check(configs, i)?;
            let c = cfg.channels;
            let rechannel = Conv1x1::read(cfg.input_size, c, false, &mut w)?;
            let zc = match cfg.gating {
                Gating::None => c,
                Gating::Gated(_) => 2 * c,
            };
            let k = cfg.kernel_size;
            let mut layers = Vec::with_capacity(cfg.dilations.len());
            for &dilation in &cfg.dilations {
                let mut conv = vec![0.0; k * zc * c];
                for o in 0..zc {
                    for i in 0..c {
                        for tap in 0..k {
                            conv[o * k * c + tap * c + i] = w.next()?;
                        }
                    }
                }
                let conv_bias = w.take(zc)?;
                let mixin = w.take(zc)?;
                let out = Conv1x1::read(c, c, true, &mut w)?;
                let history = History::new(c, (k - 1) * dilation);
                let offsets = (0..k * c)
                    .map(|t| {
                        let back = (k - 1 - t / c) * dilation;
                        (t % c) * history.capacity + history.lookback - back
                    })
                    .collect();
                layers.push(Layer {
                    channels: c,
                    kernel: k,
                    dilation,
                    activation: cfg.activation,
                    gating: cfg.gating,
                    conv,
                    conv_bias,
                    mixin,
                    out,
                    history,
                    offsets,
                    z: Rows::new(zc),
                });
            }
            let head_rechannel = Conv1x1::read(c, cfg.head_size, cfg.head_bias, &mut w)?;
            arrays.push(LayerArray {
                channels: c,
                rechannel,
                layers,
                head_rechannel,
                a: Rows::new(c),
                b: Rows::new(c),
                head: Rows::new(c),
                head_out: Rows::new(cfg.head_size),
            });
        }
        let head_scale = w.next()?;
        w.finish()?;
        match configs.last() {
            Some(last) if last.head_size == 1 => {}
            Some(last) => {
                return Err(format!(
                    "the model outputs {} channels; bloop runs mono models",
                    last.head_size
                ))
            }
            None => return Err("the model has no layers".into()),
        }
        Ok(Self {
            arrays,
            head_scale,
            input: Rows::new(1),
        })
    }

    /// How many samples of silence settle it, as NAM counts them: one,
    /// plus every array's receptive field.
    #[must_use]
    pub fn prewarm_samples(&self) -> usize {
        1 + self
            .arrays
            .iter()
            .map(LayerArray::receptive_field)
            .sum::<usize>()
    }

    /// Forgets everything it has heard.
    pub fn clear(&mut self) {
        for array in &mut self.arrays {
            for layer in &mut array.layers {
                layer.history.clear();
            }
        }
    }

    /// Takes on `from`'s state — what it has heard — without allocating.
    /// `false`, and nothing changed, if the two are not the same network.
    pub fn restore(&mut self, from: &Self) -> bool {
        let same = self.arrays.len() == from.arrays.len()
            && self.arrays.iter().zip(&from.arrays).all(|(a, b)| {
                a.layers.len() == b.layers.len()
                    && a.layers.iter().zip(&b.layers).all(|(x, y)| {
                        x.history.data.len() == y.history.data.len()
                            && x.history.lookback == y.history.lookback
                    })
            });
        if !same {
            return false;
        }
        for (a, b) in self.arrays.iter_mut().zip(&from.arrays) {
            for (x, y) in a.layers.iter_mut().zip(&b.layers) {
                x.history.data.copy_from_slice(&y.history.data);
                x.history.at = y.history.at;
            }
        }
        true
    }

    /// Multiplies and adds per sample: what one sample costs.
    #[must_use]
    pub fn macs_per_sample(&self) -> usize {
        self.arrays
            .iter()
            .map(|a| {
                let c = a.channels;
                a.rechannel.inputs * c
                    + c * a.head_rechannel.outputs
                    + a.layers
                        .iter()
                        .map(|l| l.kernel * c * l.z_channels() + l.z_channels() + c * c)
                        .sum::<usize>()
            })
            .sum()
    }

    /// Runs `input` into `output`, at most [`MAX_BLOCK`] frames.
    pub(crate) fn process_block(&mut self, input: &[f32], output: &mut [f32], tanh: Tanh) {
        let frames = input.len();
        self.input.row_mut(0, frames).copy_from_slice(input);
        let condition = self.input.row(0, frames);
        let (first, rest) = self.arrays.split_at_mut(1);
        let mut prev = &mut first[0];
        prev.process(&self.input, condition, None, frames, tanh);
        for array in rest {
            array.process(&prev.a, condition, Some(&prev.head_out), frames, tanh);
            prev = array;
        }
        for (o, h) in output.iter_mut().zip(prev.head_out.row(0, frames)) {
            *o = self.head_scale * *h;
        }
    }
}

/// Whether array `i` is one this port runs and fits the array before it.
fn check(configs: &[ArrayConfig], i: usize) -> Result<(), String> {
    let cfg = &configs[i];
    if cfg.channels == 0 || cfg.channels > MAX_CHANNELS {
        return Err(format!(
            "layer array {i} has {} channels; bloop runs 1 to {MAX_CHANNELS}",
            cfg.channels
        ));
    }
    if cfg.condition_size != 1 {
        return Err(format!(
            "layer array {i} is conditioned on {} channels; bloop runs models conditioned on the input alone",
            cfg.condition_size
        ));
    }
    if i == 0 && cfg.input_size != 1 {
        return Err("the model takes more than one input channel; bloop runs mono models".into());
    }
    if i > 0 {
        let prev = &configs[i - 1];
        if prev.channels != cfg.input_size {
            return Err(format!(
                "layer array {i} takes {} channels, but the array before it outputs {}",
                cfg.input_size, prev.channels
            ));
        }
        if prev.head_size != cfg.channels {
            return Err(format!(
                "layer array {i} has {} channels, but the array before it hands on a head of {}",
                cfg.channels, prev.head_size
            ));
        }
    }
    Ok(())
}
