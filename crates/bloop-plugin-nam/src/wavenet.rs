//! NAM's WaveNet, the architecture of nearly every `.nam` capture
//! (`NAM/wavenet/model.cpp`, `NAM/wavenet/detail.h`, `NAM/film.h`).
//!
//! A model is a stack of layer arrays. Each array re-channels its input,
//! then runs a layer per dilation: a dilated causal convolution of the
//! layer's input plus a 1x1 mix of the condition (the dry input, or what a
//! condition DSP makes of it) into the layer's bottleneck, an activation
//! (plain, gated or blended), a 1x1 back to the layer's channels added to
//! its input for the next layer, and the activation — or a 1x1 of it —
//! summed into the array's head. The head, convolved, is handed on as the
//! next array's starting sum; the last array's head, scaled (and run
//! through a post-stack head, if the model has one), is the output.
//!
//! The trainer's A1 sizes (standard, lite, feather, nano) are one shape of
//! this: a kernel of 3, the bottleneck the channels, a 1x1 head. A2 is
//! another: one array of 23 layers with kernels of 6 and 15, `LeakyReLU`,
//! and a head convolution 16 taps long; A2-Full has 8 channels, A2-Lite 3.
//! Everything else the core reads — grouped convolutions, gating and
//! blending with any second activation, a head 1x1, the eight `FiLM`s
//! that scale (and shift) a layer's signals by a 1x1 of the condition, a
//! condition DSP, a post-stack head — runs here too.
//!
//! Frames are processed a block at a time and stored channel by channel:
//! each channel's block of frames is one contiguous row. Every matrix
//! product is then a run of `row += weight × row` over a whole block, the
//! loop the compiler turns into NEON on a phone and SSE here, which is
//! several times faster than multiplying a frame's handful of channels at
//! a time. The C++ multiplies frame by frame through Eigen, so its sums
//! are added in another order: the two agree to float rounding, not bit
//! for bit (`tests/it/gold.rs`).

use crate::act::{Activation, Tanh};
use crate::config::{ArrayConfig, FilmConfig, Gating, HeadConfig};

/// The most frames processed in one pass; a longer block is split.
pub const MAX_BLOCK: usize = 128;

/// The most channels any convolution of a model may have. The trainer's
/// largest preset has 16 (a gated layer's convolution makes twice that).
pub const MAX_CHANNELS: usize = 64;

/// The furthest back any convolution may look, in frames: A2's longest
/// is 1,195, A1's 1,024.
pub const MAX_LOOKBACK: usize = 1 << 14;

/// The most frames of history all of a model's layers may hold together,
/// channels times frames: A2-Full holds 126,000, A1 standard 115,000.
pub const MAX_HISTORY: usize = 1 << 22;

/// The longest kernel any convolution may have: A2's is 16.
pub const MAX_KERNEL: usize = 256;

/// `channels` rows of [`MAX_BLOCK`] frames.
#[derive(Clone, Debug)]
pub(crate) struct Rows {
    data: Vec<f32>,
}

impl Rows {
    pub(crate) fn new(channels: usize) -> Self {
        Self {
            data: vec![0.0; channels * MAX_BLOCK],
        }
    }

    #[inline]
    pub(crate) fn row(&self, ch: usize, frames: usize) -> &[f32] {
        &self.data[ch * MAX_BLOCK..ch * MAX_BLOCK + frames]
    }

    #[inline]
    pub(crate) fn row_mut(&mut self, ch: usize, frames: usize) -> &mut [f32] {
        &mut self.data[ch * MAX_BLOCK..ch * MAX_BLOCK + frames]
    }

    /// Rows `0..channels` set to `from`'s.
    fn copy_from(&mut self, from: &Self, channels: usize) {
        self.data[..channels * MAX_BLOCK].copy_from_slice(&from.data[..channels * MAX_BLOCK]);
    }

    /// Rows `0..channels`, plus `from`'s.
    fn add(&mut self, from: &Self, channels: usize, frames: usize) {
        for ch in 0..channels {
            for (a, b) in self.row_mut(ch, frames).iter_mut().zip(from.row(ch, frames)) {
                *a += *b;
            }
        }
    }
}

/// `a + w · x`. On a phone (aarch64) one fused multiply-add, which every
/// `ARMv8` core has: half the instructions of a multiply then an add, in the
/// loop all of a model's time goes to. Fused, the product is not rounded
/// before the add, so a phone's output differs from this machine's in the
/// last bits — well inside what `tests/it/gold.rs` holds the port to. Elsewhere
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
    debug_assert!(
        w.iter().enumerate().all(|(t, _)| start(t) + frames <= data.len()),
        "a weight reads past its row"
    );
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

    /// How many are left to read.
    pub(crate) fn remaining(&self) -> usize {
        self.values.len() - self.at
    }

    /// How many there are.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<Vec<f32>, String> {
        if n > self.remaining() {
            return Err(format!(
                "the model has {} weights, fewer than its layers need",
                self.values.len()
            ));
        }
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

/// What a convolution has heard, for it to look back over: per channel,
/// `lookback` frames before the block being processed, then the block.
/// When a row fills, its last `lookback` frames move to the front (NAM's
/// `RingBuffer` does the same).
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

    fn restore(&mut self, from: &Self) -> bool {
        if self.data.len() != from.data.len() || self.lookback != from.lookback {
            return false;
        }
        self.data.copy_from_slice(&from.data);
        self.at = from.at;
        true
    }
}

/// A causal convolution, `inputs` channels to `outputs`, over `kernel`
/// taps `dilation` frames apart, its channels split into `groups` that
/// see only their own (`Conv1D`; with a kernel of 1, `Conv1x1`). A 1x1
/// reads its input directly; a longer kernel keeps a [`History`].
#[derive(Clone, Debug)]
pub(crate) struct Conv {
    inputs: usize,
    outputs: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    /// Per output, `[tap][input of its group]`, tap 0 the furthest back.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    history: Option<History>,
    /// Per group, where weight `tap · in_per_group + i` of its outputs
    /// reads: past the start of the lookback in `history`, or into the
    /// input's rows.
    offsets: Vec<usize>,
}

impl Conv {
    /// Reads its weights as NAM lays them out: per group, per output of
    /// it, per input of it, per tap; then the bias.
    pub(crate) fn read(
        inputs: usize,
        outputs: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        weights: &mut Weights,
    ) -> Result<Self, String> {
        if inputs == 0 || outputs == 0 || inputs > MAX_CHANNELS || outputs > 2 * MAX_CHANNELS {
            return Err(format!(
                "a convolution of {inputs} channels into {outputs}; bloop runs 1 to {MAX_CHANNELS}"
            ));
        }
        if groups == 0 || !inputs.is_multiple_of(groups) || !outputs.is_multiple_of(groups) {
            return Err(format!(
                "a convolution of {inputs} channels into {outputs} cannot be split into {groups} groups"
            ));
        }
        if kernel == 0 || dilation == 0 {
            return Err("a convolution has a kernel or a dilation of 0".into());
        }
        if kernel > MAX_KERNEL {
            return Err(format!(
                "a convolution has a kernel of {kernel}; bloop runs up to {MAX_KERNEL}"
            ));
        }
        let lookback = (kernel - 1)
            .checked_mul(dilation)
            .filter(|&l| l <= MAX_LOOKBACK)
            .ok_or_else(|| {
                format!("a convolution looks further back than bloop runs ({MAX_LOOKBACK} frames)")
            })?;
        let ipg = inputs / groups;
        let per_output = kernel * ipg;
        // Before anything is made of it: a file that names a convolution
        // its weights cannot fill is refused, not allocated for.
        if weights.remaining() < outputs * per_output {
            return Err(format!(
                "the model has {} weights, fewer than its layers need",
                weights.len()
            ));
        }
        let mut weight = vec![0.0; outputs * per_output];
        for o in 0..outputs {
            for i in 0..ipg {
                for tap in 0..kernel {
                    weight[o * per_output + tap * ipg + i] = weights.next()?;
                }
            }
        }
        let bias = if bias {
            Some(weights.take(outputs)?)
        } else {
            None
        };
        let history = (kernel > 1).then(|| History::new(inputs, lookback));
        let mut offsets = Vec::with_capacity(groups * per_output);
        for g in 0..groups {
            for t in 0..per_output {
                let ch = g * ipg + t % ipg;
                offsets.push(match &history {
                    Some(h) => ch * h.capacity + h.lookback - (kernel - 1 - t / ipg) * dilation,
                    None => ch * MAX_BLOCK,
                });
            }
        }
        Ok(Self {
            inputs,
            outputs,
            kernel,
            dilation,
            groups,
            weight,
            bias,
            history,
            offsets,
        })
    }

    fn lookback(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }

    fn macs(&self) -> usize {
        self.weight.len()
    }

    /// `out = conv(input) (+ bias)`, each output row plus `init(o)`'s
    /// scaled row if it gives one, over `frames` frames.
    fn run<'a>(
        &mut self,
        input: &Rows,
        out: &mut Rows,
        frames: usize,
        init: impl Fn(usize) -> Option<(f32, &'a [f32])>,
    ) {
        let per_output = self.weight.len() / self.outputs;
        let opg = self.outputs / self.groups;
        let (data, base) = match &mut self.history {
            Some(h) => {
                h.write(input, frames);
                (&h.data[..], h.at - h.lookback)
            }
            None => (&input.data[..], 0),
        };
        for o in 0..self.outputs {
            let offsets = &self.offsets[(o / opg) * per_output..(o / opg + 1) * per_output];
            dense_row(
                out.row_mut(o, frames),
                self.bias.as_ref().map_or(0.0, |b| b[o]),
                init(o),
                &self.weight[o * per_output..(o + 1) * per_output],
                data,
                |t| base + offsets[t],
            );
        }
        if let Some(h) = &mut self.history {
            h.at += frames;
        }
    }

    fn clear(&mut self) {
        if let Some(h) = &mut self.history {
            h.clear();
        }
    }

    fn restore(&mut self, from: &Self) -> bool {
        match (&mut self.history, &from.history) {
            (Some(a), Some(b)) => a.restore(b),
            (None, None) => true,
            _ => false,
        }
    }
}

/// Feature-wise linear modulation (`NAM/film.h`): a 1x1 of the condition
/// into a scale per channel, and a shift, by which a layer's signal is
/// multiplied (and to which it is added).
#[derive(Clone, Debug)]
struct Film {
    dim: usize,
    shift: bool,
    conv: Conv,
    scale_shift: Rows,
}

impl Film {
    fn read(
        condition: usize,
        dim: usize,
        config: Option<FilmConfig>,
        weights: &mut Weights,
    ) -> Result<Option<Self>, String> {
        let Some(c) = config else { return Ok(None) };
        let outputs = if c.shift { 2 * dim } else { dim };
        Ok(Some(Self {
            dim,
            shift: c.shift,
            conv: Conv::read(condition, outputs, 1, 1, c.groups, true, weights)?,
            scale_shift: Rows::new(outputs),
        }))
    }

    /// Rows `0..dim` of `x`, modulated in place.
    fn apply(&mut self, x: &mut Rows, condition: &Rows, frames: usize) {
        self.conv
            .run(condition, &mut self.scale_shift, frames, |_| None);
        for c in 0..self.dim {
            let scale = self.scale_shift.row(c, frames);
            let row = &mut x.data[c * MAX_BLOCK..c * MAX_BLOCK + frames];
            if self.shift {
                let shift = self.scale_shift.row(c + self.dim, frames);
                for ((v, s), b) in row.iter_mut().zip(scale).zip(shift) {
                    *v = *v * *s + *b;
                }
            } else {
                for (v, s) in row.iter_mut().zip(scale) {
                    *v *= *s;
                }
            }
        }
    }
}

/// A layer's `FiLM`s, where it has them.
#[derive(Clone, Debug, Default)]
struct LayerFilms {
    conv_pre: Option<Film>,
    conv_post: Option<Film>,
    input_mixin_pre: Option<Film>,
    input_mixin_post: Option<Film>,
    activation_pre: Option<Film>,
    activation_post: Option<Film>,
    layer1x1_post: Option<Film>,
    head1x1_post: Option<Film>,
}

impl LayerFilms {
    fn all(&self) -> [&Option<Film>; 8] {
        [
            &self.conv_pre,
            &self.conv_post,
            &self.input_mixin_pre,
            &self.input_mixin_post,
            &self.activation_pre,
            &self.activation_post,
            &self.layer1x1_post,
            &self.head1x1_post,
        ]
    }
}

/// How the condition is mixed into a layer.
#[derive(Clone, Debug)]
enum Mixin {
    /// A condition of one channel and nothing between: a weight per
    /// output, summed in with the convolution in one pass.
    Fused(Vec<f32>),
    /// Anything else: a 1x1 into its own rows, added after.
    Separate(Conv, Rows),
}

#[derive(Clone, Debug)]
struct Layer {
    channels: usize,
    bottleneck: usize,
    activation: Activation,
    gating: Gating,
    conv: Conv,
    mixin: Mixin,
    layer1x1: Option<Conv>,
    head1x1: Option<Conv>,
    films: LayerFilms,
    /// The convolution's sum, then the activation.
    z: Rows,
    /// The first half of `z` before its activation, for a blend.
    pre: Rows,
    /// The input after `conv_pre_film`; the condition after
    /// `input_mixin_pre_film`; the 1x1s' outputs, where something comes
    /// between them and their sums.
    conv_in: Rows,
    condition_in: Rows,
    residual: Rows,
    head_out: Rows,
}

impl Layer {
    fn read(cfg: &ArrayConfig, l: usize, weights: &mut Weights) -> Result<Self, String> {
        let (c, b, cond) = (cfg.channels, cfg.bottleneck, cfg.condition_size);
        let gating = cfg.gatings[l].clone();
        let activation = cfg.activations[l].clone();
        let zc = gating.z_channels(b);
        let per_channel = !matches!(gating, Gating::None);
        activation.check_channels(b, per_channel)?;
        if let Gating::Gated(second) | Gating::Blended(second) = &gating {
            second.check_channels(b, true)?;
        }
        let conv = Conv::read(
            c,
            zc,
            cfg.kernel_sizes[l],
            cfg.dilations[l],
            cfg.groups_input,
            true,
            weights,
        )?;
        let mixin = Conv::read(cond, zc, 1, 1, cfg.groups_input_mixin, false, weights)?;
        let layer1x1 = match cfg.layer1x1 {
            Some(groups) => Some(Conv::read(b, c, 1, 1, groups, true, weights)?),
            None => None,
        };
        let head1x1 = match cfg.head1x1 {
            Some((out, groups)) => Some(Conv::read(b, out, 1, 1, groups, true, weights)?),
            None => None,
        };
        let f = &cfg.films;
        let films = LayerFilms {
            conv_pre: Film::read(cond, c, f.conv_pre, weights)?,
            conv_post: Film::read(cond, zc, f.conv_post, weights)?,
            input_mixin_pre: Film::read(cond, cond, f.input_mixin_pre, weights)?,
            input_mixin_post: Film::read(cond, zc, f.input_mixin_post, weights)?,
            activation_pre: Film::read(cond, zc, f.activation_pre, weights)?,
            activation_post: Film::read(cond, b, f.activation_post, weights)?,
            layer1x1_post: Film::read(cond, c, f.layer1x1_post, weights)?,
            head1x1_post: Film::read(cond, cfg.head_inputs(), f.head1x1_post, weights)?,
        };
        let fused = cond == 1
            && films.conv_post.is_none()
            && films.input_mixin_pre.is_none()
            && films.input_mixin_post.is_none();
        let mixin = if fused {
            Mixin::Fused(mixin.weight)
        } else {
            Mixin::Separate(mixin, Rows::new(zc))
        };
        Ok(Self {
            channels: c,
            bottleneck: b,
            activation,
            gating,
            conv,
            mixin,
            layer1x1,
            head1x1,
            films,
            z: Rows::new(zc),
            pre: Rows::new(b),
            conv_in: Rows::new(c),
            condition_in: Rows::new(cond),
            residual: Rows::new(c),
            head_out: Rows::new(cfg.head_inputs()),
        })
    }

    fn macs(&self) -> usize {
        let mixin = match &self.mixin {
            Mixin::Fused(w) => w.len(),
            Mixin::Separate(conv, _) => conv.macs(),
        };
        self.conv.macs()
            + mixin
            + self.layer1x1.as_ref().map_or(0, Conv::macs)
            + self.head1x1.as_ref().map_or(0, Conv::macs)
            + self
                .films
                .all()
                .iter()
                .filter_map(|f| f.as_ref())
                .map(|f| f.conv.macs() + f.dim)
                .sum::<usize>()
    }

    /// One block (`Layer::Process`): reads `input`, adds to `head`, writes
    /// `input + layer1x1(activation)` to `next`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the layer's inputs and outputs, all distinct buffers"
    )]
    fn process(
        &mut self,
        input: &Rows,
        condition: &Rows,
        head: &mut Rows,
        head_rows: usize,
        next: &mut Rows,
        frames: usize,
        tanh: Tanh,
    ) {
        let (c, b) = (self.channels, self.bottleneck);
        let zc = self.gating.z_channels(b);
        let films = &mut self.films;
        let conv_in = match &mut films.conv_pre {
            Some(film) => {
                self.conv_in.copy_from(input, c);
                film.apply(&mut self.conv_in, condition, frames);
                &self.conv_in
            }
            None => input,
        };
        match &mut self.mixin {
            Mixin::Fused(w) => {
                let cond = condition.row(0, frames);
                self.conv
                    .run(conv_in, &mut self.z, frames, |o| Some((w[o], cond)));
            }
            Mixin::Separate(mixin, mixed) => {
                self.conv.run(conv_in, &mut self.z, frames, |_| None);
                if let Some(film) = &mut films.conv_post {
                    film.apply(&mut self.z, condition, frames);
                }
                let cond_in = match &mut films.input_mixin_pre {
                    Some(film) => {
                        self.condition_in.copy_from(condition, mixin.inputs);
                        film.apply(&mut self.condition_in, condition, frames);
                        &self.condition_in
                    }
                    None => condition,
                };
                mixin.run(cond_in, mixed, frames, |_| None);
                if let Some(film) = &mut films.input_mixin_post {
                    film.apply(mixed, condition, frames);
                }
                self.z.add(mixed, zc, frames);
            }
        }
        if let Some(film) = &mut films.activation_pre {
            film.apply(&mut self.z, condition, frames);
        }
        match &self.gating {
            Gating::None => {
                for o in 0..b {
                    self.activation.apply(self.z.row_mut(o, frames), o, tanh);
                }
            }
            Gating::Gated(second) => {
                let (top, bottom) = self.z.data.split_at_mut(b * MAX_BLOCK);
                for o in 0..b {
                    let t = &mut top[o * MAX_BLOCK..o * MAX_BLOCK + frames];
                    let g = &mut bottom[o * MAX_BLOCK..o * MAX_BLOCK + frames];
                    self.activation.apply(t, o, tanh);
                    second.apply(g, o, tanh);
                    for (t, g) in t.iter_mut().zip(g.iter()) {
                        *t *= *g;
                    }
                }
            }
            Gating::Blended(second) => {
                self.pre.copy_from(&self.z, b);
                let (top, bottom) = self.z.data.split_at_mut(b * MAX_BLOCK);
                for o in 0..b {
                    let t = &mut top[o * MAX_BLOCK..o * MAX_BLOCK + frames];
                    let alpha = &mut bottom[o * MAX_BLOCK..o * MAX_BLOCK + frames];
                    self.activation.apply(t, o, tanh);
                    second.apply(alpha, o, tanh);
                    for ((t, a), p) in t.iter_mut().zip(alpha.iter()).zip(self.pre.row(o, frames))
                    {
                        *t = *a * *t + (1.0 - *a) * *p;
                    }
                }
            }
        }
        if let Some(film) = &mut films.activation_post {
            film.apply(&mut self.z, condition, frames);
        }
        match (&mut self.layer1x1, &mut films.layer1x1_post) {
            (Some(l1), None) => {
                l1.run(&self.z, next, frames, |o| Some((1.0, input.row(o, frames))));
            }
            (Some(l1), Some(film)) => {
                l1.run(&self.z, &mut self.residual, frames, |_| None);
                film.apply(&mut self.residual, condition, frames);
                next.copy_from(input, c);
                next.add(&self.residual, c, frames);
            }
            (None, _) => next.copy_from(input, c),
        }
        match &mut self.head1x1 {
            Some(h1) => {
                h1.run(&self.z, &mut self.head_out, frames, |_| None);
                if let Some(film) = &mut films.head1x1_post {
                    film.apply(&mut self.head_out, condition, frames);
                }
                head.add(&self.head_out, head_rows, frames);
            }
            None => head.add(&self.z, head_rows, frames),
        }
    }

    fn convs_mut(&mut self) -> impl Iterator<Item = &mut Conv> {
        std::iter::once(&mut self.conv)
    }

    fn convs(&self) -> impl Iterator<Item = &Conv> {
        std::iter::once(&self.conv)
    }
}

#[derive(Clone, Debug)]
struct LayerArray {
    /// How many channels each layer adds to the head.
    head_inputs: usize,
    rechannel: Conv,
    layers: Vec<Layer>,
    head_rechannel: Conv,
    /// The layers' inputs and outputs, ping-pong.
    a: Rows,
    b: Rows,
    /// The head's running sum.
    head: Rows,
    /// The head convolved.
    head_out: Rows,
}

impl LayerArray {
    fn read(cfg: &ArrayConfig, weights: &mut Weights) -> Result<Self, String> {
        let rechannel = Conv::read(cfg.input_size, cfg.channels, 1, 1, 1, false, weights)?;
        let layers = (0..cfg.dilations.len())
            .map(|l| Layer::read(cfg, l, weights))
            .collect::<Result<Vec<_>, _>>()?;
        let head_rechannel = Conv::read(
            cfg.head_inputs(),
            cfg.head_size,
            cfg.head_kernel_size,
            cfg.head_dilation,
            1,
            cfg.head_bias,
            weights,
        )?;
        Ok(Self {
            head_inputs: cfg.head_inputs(),
            rechannel,
            layers,
            head_rechannel,
            a: Rows::new(cfg.channels),
            b: Rows::new(cfg.channels),
            head: Rows::new(cfg.head_inputs()),
            head_out: Rows::new(cfg.head_size),
        })
    }

    fn receptive_field(&self) -> usize {
        self.layers.iter().map(|l| l.conv.lookback()).sum::<usize>()
            + self.head_rechannel.lookback()
    }

    fn macs(&self) -> usize {
        self.rechannel.macs()
            + self.head_rechannel.macs()
            + self.layers.iter().map(Layer::macs).sum::<usize>()
    }

    /// Runs the block. `input` has `rechannel.inputs` rows; `head_in` is
    /// the previous array's head (`None` for the first). Leaves the
    /// array's output in `a` and its head in `head_out`.
    fn process(
        &mut self,
        input: &Rows,
        condition: &Rows,
        head_in: Option<&Rows>,
        frames: usize,
        tanh: Tanh,
    ) {
        let h = self.head_inputs;
        match head_in {
            Some(prev) => self.head.copy_from(prev, h),
            None => self.head.data.fill(0.0),
        }
        self.rechannel.run(input, &mut self.a, frames, |_| None);
        for layer in &mut self.layers {
            layer.process(
                &self.a,
                condition,
                &mut self.head,
                h,
                &mut self.b,
                frames,
                tanh,
            );
            std::mem::swap(&mut self.a, &mut self.b);
        }
        self.head_rechannel
            .run(&self.head, &mut self.head_out, frames, |_| None);
    }

    fn convs_mut(&mut self) -> impl Iterator<Item = &mut Conv> {
        self.layers
            .iter_mut()
            .flat_map(Layer::convs_mut)
            .chain(std::iter::once(&mut self.head_rechannel))
    }

    fn convs(&self) -> impl Iterator<Item = &Conv> {
        self.layers
            .iter()
            .flat_map(Layer::convs)
            .chain(std::iter::once(&self.head_rechannel))
    }
}

/// The post-stack head (`detail::Head`): per kernel, the activation, then
/// a convolution with a bias.
#[derive(Clone, Debug)]
struct PostHead {
    activation: Activation,
    convs: Vec<Conv>,
    outs: Vec<Rows>,
    in_channels: usize,
}

impl PostHead {
    fn read(cfg: &HeadConfig, weights: &mut Weights) -> Result<Self, String> {
        let n = cfg.kernel_sizes.len();
        let mut convs = Vec::with_capacity(n);
        let mut outs = Vec::with_capacity(n);
        let mut cin = cfg.in_channels;
        for (i, &k) in cfg.kernel_sizes.iter().enumerate() {
            let cout = if i + 1 == n {
                cfg.out_channels
            } else {
                cfg.channels
            };
            cfg.activation.check_channels(cin, false)?;
            convs.push(Conv::read(cin, cout, k, 1, 1, true, weights)?);
            outs.push(Rows::new(cout));
            cin = cout;
        }
        Ok(Self {
            activation: cfg.activation.clone(),
            convs,
            outs,
            in_channels: cfg.in_channels,
        })
    }

    /// Runs `work` (scaled head, modified in place) through; the output
    /// is the last of `outs`.
    fn process(&mut self, work: &mut Rows, frames: usize, tanh: Tanh) {
        let mut rows = self.in_channels;
        for i in 0..self.convs.len() {
            let (done, rest) = self.outs.split_at_mut(i);
            let input = match done.last_mut() {
                Some(prev) => prev,
                None => &mut *work,
            };
            for ch in 0..rows {
                self.activation.apply(input.row_mut(ch, frames), ch, tanh);
            }
            self.convs[i].run(input, &mut rest[0], frames, |_| None);
            rows = self.convs[i].outputs;
        }
    }

    fn output(&self) -> &Rows {
        &self.outs[self.outs.len() - 1]
    }
}

/// A WaveNet model, ready to run.
#[derive(Clone, Debug)]
pub struct WaveNet {
    /// What makes the condition from the input, if not the input itself.
    condition: Option<Box<WaveNet>>,
    arrays: Vec<LayerArray>,
    post_head: Option<PostHead>,
    head_scale: f32,
    out_channels: usize,
    /// The input block as one row.
    input: Rows,
    /// The scaled head, which a post-stack head works on.
    scaled: Rows,
    /// The output, `out_channels` rows.
    output: Rows,
}

impl WaveNet {
    /// Builds a WaveNet from its arrays, weights, and the condition DSP
    /// `condition` (already built) if it has one.
    pub(crate) fn build(
        arrays: &[ArrayConfig],
        head: Option<&HeadConfig>,
        weights: &[f32],
        condition: Option<WaveNet>,
    ) -> Result<Self, String> {
        check(arrays, condition.as_ref())?;
        let mut w = Weights::new(weights);
        let arrays_built = arrays
            .iter()
            .map(|cfg| LayerArray::read(cfg, &mut w))
            .collect::<Result<Vec<_>, _>>()?;
        let post_head = match head {
            Some(h) => Some(PostHead::read(h, &mut w)?),
            None => None,
        };
        let head_scale = w.next()?;
        w.finish()?;
        let last = arrays.last().map_or(0, |a| a.head_size);
        let out_channels = head.map_or(last, |h| h.out_channels);
        Ok(Self {
            condition: condition.map(Box::new),
            arrays: arrays_built,
            post_head,
            head_scale,
            out_channels,
            input: Rows::new(1),
            scaled: Rows::new(last),
            output: Rows::new(out_channels),
        })
    }

    /// How many channels it outputs.
    #[must_use]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// How many samples of silence settle it, as NAM counts them: what
    /// its condition DSP needs (or one), plus every array's receptive
    /// field, plus the post-stack head's.
    #[must_use]
    pub fn prewarm_samples(&self) -> usize {
        self.condition.as_ref().map_or(1, |c| c.prewarm_samples())
            + self
                .arrays
                .iter()
                .map(LayerArray::receptive_field)
                .sum::<usize>()
            + self
                .post_head
                .as_ref()
                .map_or(0, |h| h.convs.iter().map(Conv::lookback).sum())
    }

    fn convs_mut(&mut self) -> impl Iterator<Item = &mut Conv> {
        self.arrays
            .iter_mut()
            .flat_map(LayerArray::convs_mut)
            .chain(self.post_head.iter_mut().flat_map(|h| h.convs.iter_mut()))
    }

    fn convs(&self) -> impl Iterator<Item = &Conv> {
        self.arrays
            .iter()
            .flat_map(LayerArray::convs)
            .chain(self.post_head.iter().flat_map(|h| h.convs.iter()))
    }

    /// Forgets everything it has heard.
    pub fn clear(&mut self) {
        if let Some(c) = &mut self.condition {
            c.clear();
        }
        self.convs_mut().for_each(Conv::clear);
    }

    /// Takes on `from`'s state — what it has heard — without allocating.
    /// `false`, and nothing changed, if the two are not the same network.
    pub fn restore(&mut self, from: &Self) -> bool {
        let same_condition = match (&self.condition, &from.condition) {
            (Some(a), Some(b)) => a.same_shape(b),
            (None, None) => true,
            _ => false,
        };
        if !same_condition || !self.same_shape(from) {
            return false;
        }
        if let (Some(a), Some(b)) = (&mut self.condition, &from.condition) {
            a.restore(b);
        }
        for (a, b) in self.convs_mut().zip(from.convs()) {
            a.restore(b);
        }
        true
    }

    fn same_shape(&self, other: &Self) -> bool {
        self.convs().count() == other.convs().count()
            && self.convs().zip(other.convs()).all(|(a, b)| {
                a.inputs == b.inputs && a.kernel == b.kernel && a.dilation == b.dilation
            })
    }

    /// Multiplies and adds per sample: what one sample costs.
    #[must_use]
    pub fn macs_per_sample(&self) -> usize {
        self.condition.as_ref().map_or(0, |c| c.macs_per_sample())
            + self.arrays.iter().map(LayerArray::macs).sum::<usize>()
            + self
                .post_head
                .as_ref()
                .map_or(0, |h| h.convs.iter().map(Conv::macs).sum())
    }

    /// Runs `input`, at most [`MAX_BLOCK`] frames, leaving the output in
    /// `self.output`.
    fn run(&mut self, input: &[f32], tanh: Tanh) {
        let frames = input.len();
        self.input.row_mut(0, frames).copy_from_slice(input);
        let condition = match &mut self.condition {
            Some(c) => {
                c.run(input, tanh);
                &c.output
            }
            None => &self.input,
        };
        let (first, rest) = self.arrays.split_at_mut(1);
        let mut prev = &mut first[0];
        prev.process(&self.input, condition, None, frames, tanh);
        for array in rest {
            array.process(&prev.a, condition, Some(&prev.head_out), frames, tanh);
            prev = array;
        }
        let head = &prev.head_out;
        match &mut self.post_head {
            Some(post) => {
                for ch in 0..post.in_channels {
                    for (s, h) in self
                        .scaled
                        .row_mut(ch, frames)
                        .iter_mut()
                        .zip(head.row(ch, frames))
                    {
                        *s = self.head_scale * *h;
                    }
                }
                post.process(&mut self.scaled, frames, tanh);
                self.output.copy_from(post.output(), self.out_channels);
            }
            None => {
                for ch in 0..self.out_channels {
                    for (o, h) in self
                        .output
                        .row_mut(ch, frames)
                        .iter_mut()
                        .zip(head.row(ch, frames))
                    {
                        *o = self.head_scale * *h;
                    }
                }
            }
        }
    }

    /// Runs `input` into `output`, at most [`MAX_BLOCK`] frames, mono.
    pub(crate) fn process_block(&mut self, input: &[f32], output: &mut [f32], tanh: Tanh) {
        self.run(input, tanh);
        output.copy_from_slice(self.output.row(0, input.len()));
    }
}

/// Whether the arrays fit each other and the condition (what
/// `WaveNet::WaveNet` and `Layer::Layer` check, and the shapes that would
/// read out of bounds in the C++).
fn check(arrays: &[ArrayConfig], condition: Option<&WaveNet>) -> Result<(), String> {
    let condition_size = condition.map_or(1, WaveNet::out_channels);
    // What its histories would hold, before any is made: a file of few
    // weights can still name thousands of layers that look far back.
    let mut history = 0usize;
    for cfg in arrays {
        for (&k, &d) in cfg.kernel_sizes.iter().zip(&cfg.dilations) {
            let lookback = k.saturating_sub(1).saturating_mul(d);
            history = history.saturating_add(
                cfg.channels
                    .saturating_mul(lookback.saturating_mul(2).saturating_add(MAX_BLOCK)),
            );
        }
    }
    if history > MAX_HISTORY {
        return Err("the model would hold more history than bloop runs".into());
    }
    for (i, cfg) in arrays.iter().enumerate() {
        let n = cfg.dilations.len();
        if n == 0 {
            return Err(format!("layer array {i} has no layers"));
        }
        if cfg.dilations.contains(&0) {
            return Err(format!("layer array {i} has a dilation of 0"));
        }
        if cfg.kernel_sizes.contains(&0) || cfg.head_kernel_size == 0 {
            return Err(format!("layer array {i} has a kernel of 0"));
        }
        if cfg.channels == 0 || cfg.channels > MAX_CHANNELS {
            return Err(format!(
                "layer array {i} has {} channels; bloop runs 1 to {MAX_CHANNELS}",
                cfg.channels
            ));
        }
        if cfg.bottleneck == 0 || cfg.bottleneck > MAX_CHANNELS {
            return Err(format!(
                "layer array {i} has a bottleneck of {}; bloop runs 1 to {MAX_CHANNELS}",
                cfg.bottleneck
            ));
        }
        if cfg.layer1x1.is_none() && cfg.bottleneck != cfg.channels {
            return Err(format!(
                "layer array {i} has no layer 1x1, so its bottleneck must be its channels"
            ));
        }
        if cfg.condition_size != condition_size {
            return Err(format!(
                "layer array {i} is conditioned on {} channels, but its condition has {condition_size}",
                cfg.condition_size
            ));
        }
        if i == 0 && cfg.input_size != 1 {
            return Err(
                "the model takes more than one input channel; bloop runs mono models".into(),
            );
        }
        if i > 0 {
            let prev = &arrays[i - 1];
            if prev.channels != cfg.input_size {
                return Err(format!(
                    "layer array {i} takes {} channels, but the array before it outputs {}",
                    cfg.input_size, prev.channels
                ));
            }
            if prev.head_size != cfg.channels || prev.head_size != cfg.head_inputs() {
                return Err(format!(
                    "layer array {i} sums a head of {}, but the array before it hands on {}",
                    cfg.head_inputs(),
                    prev.head_size
                ));
            }
        }
    }
    Ok(())
}
