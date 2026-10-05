//! NAM's LSTM (`NAM/lstm.cpp`): a stack of LSTM cells, sample by sample,
//! and a linear head on the last cell's hidden state.

use crate::act::{fast_sigmoid, fast_tanh, sigmoid, Tanh};
use crate::wavenet::Weights;

#[derive(Clone, Debug)]
struct Cell {
    input_size: usize,
    hidden: usize,
    /// `[input + hidden][4 × hidden]`: `PyTorch`'s matrix transposed, so the
    /// product runs along the gates, which vectorizes. The gates are in
    /// `PyTorch`'s order: input, forget, cell, output.
    w: Vec<f32>,
    b: Vec<f32>,
    /// The input, then the hidden state.
    xh: Vec<f32>,
    c: Vec<f32>,
    ifgo: Vec<f32>,
    /// The hidden and cell state the file starts it from.
    h0: Vec<f32>,
    c0: Vec<f32>,
}

impl Cell {
    fn read(input_size: usize, hidden: usize, weights: &mut Weights) -> Result<Self, String> {
        let n = input_size + hidden;
        let rows = weights.take(4 * hidden * n)?;
        let mut w = vec![0.0; rows.len()];
        for r in 0..4 * hidden {
            for j in 0..n {
                w[j * 4 * hidden + r] = rows[r * n + j];
            }
        }
        let b = weights.take(4 * hidden)?;
        let h0 = weights.take(hidden)?;
        let c0 = weights.take(hidden)?;
        let mut xh = vec![0.0; input_size + hidden];
        xh[input_size..].copy_from_slice(&h0);
        Ok(Self {
            input_size,
            hidden,
            w,
            b,
            xh,
            c: c0.clone(),
            ifgo: vec![0.0; 4 * hidden],
            h0,
            c0,
        })
    }

    fn clear(&mut self) {
        self.xh[..self.input_size].fill(0.0);
        self.xh[self.input_size..].copy_from_slice(&self.h0);
        self.c.copy_from_slice(&self.c0);
    }

    fn hidden_state(&self) -> &[f32] {
        &self.xh[self.input_size..]
    }

    fn process(&mut self, x: &[f32], tanh: Tanh) {
        self.xh[..self.input_size].copy_from_slice(x);
        self.ifgo.copy_from_slice(&self.b);
        for (column, &x) in self.w.chunks_exact(4 * self.hidden).zip(&self.xh) {
            for (o, w) in self.ifgo.iter_mut().zip(column) {
                *o += w * x;
            }
        }
        match tanh {
            Tanh::Fast => self.gates(fast_sigmoid, fast_tanh),
            Tanh::Exact => self.gates(sigmoid, libm::tanhf),
        }
    }

    /// The cell and hidden state from the gates' sums, with the given
    /// sigmoid and tanh (function items, so each pair inlines).
    #[inline]
    fn gates(&mut self, sig: impl Fn(f32) -> f32, th: impl Fn(f32) -> f32) {
        let h = self.hidden;
        let (i, rest) = self.ifgo.split_at(h);
        let (f, rest) = rest.split_at(h);
        let (g, o) = rest.split_at(h);
        for (((c, i), f), g) in self.c.iter_mut().zip(i).zip(f).zip(g) {
            *c = sig(*f) * *c + sig(*i) * th(*g);
        }
        for ((x, o), c) in self.xh[self.input_size..].iter_mut().zip(o).zip(&self.c) {
            *x = sig(*o) * th(*c);
        }
    }
}

/// An LSTM model, ready to run.
#[derive(Clone, Debug)]
pub struct Lstm {
    cells: Vec<Cell>,
    head_weight: Vec<f32>,
    head_bias: f32,
    hidden: Vec<f32>,
}

impl Lstm {
    pub(crate) fn build(
        num_layers: usize,
        input_size: usize,
        hidden: usize,
        weights: &[f32],
    ) -> Result<Self, String> {
        if input_size != 1 {
            return Err(format!(
                "the LSTM takes {input_size} inputs; bloop runs mono models"
            ));
        }
        if num_layers == 0 || hidden == 0 {
            return Err("the LSTM has no layers".into());
        }
        let mut w = Weights::new(weights);
        let mut cells = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            cells.push(Cell::read(
                if i == 0 { input_size } else { hidden },
                hidden,
                &mut w,
            )?);
        }
        let head_weight = w.take(hidden)?;
        let head_bias = w.next()?;
        w.finish()?;
        Ok(Self {
            cells,
            head_weight,
            head_bias,
            hidden: vec![0.0; hidden],
        })
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    /// Takes on `from`'s state without allocating; `false`, and nothing
    /// changed, if the two are not the same network.
    pub fn restore(&mut self, from: &Self) -> bool {
        let same = self.cells.len() == from.cells.len()
            && self
                .cells
                .iter()
                .zip(&from.cells)
                .all(|(a, b)| a.xh.len() == b.xh.len() && a.c.len() == b.c.len());
        if !same {
            return false;
        }
        for (a, b) in self.cells.iter_mut().zip(&from.cells) {
            a.xh.copy_from_slice(&b.xh);
            a.c.copy_from_slice(&b.c);
        }
        true
    }

    /// NAM's guess at what settles an LSTM: half a second.
    #[must_use]
    pub fn prewarm_samples(sample_rate: f64) -> usize {
        let n = (0.5 * sample_rate) as i64;
        n.max(1) as usize
    }

    #[must_use]
    pub fn macs_per_sample(&self) -> usize {
        self.cells
            .iter()
            .map(|c| 4 * c.hidden * (c.input_size + c.hidden))
            .sum::<usize>()
            + self.head_weight.len()
    }

    pub(crate) fn process_block(&mut self, input: &[f32], output: &mut [f32], tanh: Tanh) {
        for (x, y) in input.iter().zip(output.iter_mut()) {
            let (first, rest) = self.cells.split_at_mut(1);
            first[0].process(std::slice::from_ref(x), tanh);
            let mut prev = &first[0];
            for cell in rest {
                self.hidden.copy_from_slice(prev.hidden_state());
                cell.process(&self.hidden, tanh);
                prev = cell;
            }
            *y = self
                .head_weight
                .iter()
                .zip(prev.hidden_state())
                .map(|(w, h)| w * h)
                .sum::<f32>()
                + self.head_bias;
        }
    }
}
