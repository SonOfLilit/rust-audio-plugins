//! What an effect's screen draws over time, as the effect measures it on
//! the audio thread: two numbers for every [`CHUNK`] samples it processes.
//!
//! A meter ([`crate::DynamicsMeter`]) is read once a block and says where a
//! level is now; a trace keeps every chunk of the block, so a screen can
//! draw an envelope or a gain as it moved, at the same resolution whatever
//! size of block the platform runs. The Maximizer traces its input's peak
//! and the gain it applies, the TransientShaper its detected envelope and
//! its gain.
//!
//! Fixed-size and allocation-free: the effect calls [`Trace::begin`] at the
//! start of a block and [`Trace::sample`] for every sample; the engine reads
//! [`Trace::points`] after it.

/// Samples a point: 1.3 ms at 48 kHz.
pub const CHUNK: usize = 64;
/// The most points a block keeps: a block of 8192 samples. A longer
/// block's later points are dropped.
pub const POINTS: usize = 128;

/// How the samples of a chunk are folded into its point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    /// The largest: a peak.
    Max,
    /// The smallest: a gain turned down.
    Min,
    /// The one farthest from 1 either way, as a ratio: a gain that may turn
    /// up or down.
    FarthestFromOne,
}

impl Fold {
    fn start(self) -> f32 {
        match self {
            Fold::Max => f32::NEG_INFINITY,
            Fold::Min => f32::INFINITY,
            Fold::FarthestFromOne => 1.0,
        }
    }

    /// `x` folded into what is `kept`: the larger, the smaller, or the
    /// one farther from one.
    #[must_use]
    pub fn fold(self, kept: f32, x: f32) -> f32 {
        match self {
            Fold::Max => kept.max(x),
            Fold::Min => kept.min(x),
            Fold::FarthestFromOne => {
                // |log x| against |log kept|, without the log: a ratio's
                // distance from one is its own or its inverse's, above one.
                let distance = |g: f32| if g >= 1.0 { g } else { 1.0 / g.max(1e-12) };
                if distance(x) > distance(kept) {
                    x
                } else {
                    kept
                }
            }
        }
    }
}

/// Two numbers a [`CHUNK`] of samples, the block's points kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trace {
    folds: [Fold; 2],
    points: [[f32; 2]; POINTS],
    len: usize,
    /// The chunk being folded, which carries over into the next block.
    kept: [f32; 2],
    count: usize,
}

impl Trace {
    /// A trace folding its two numbers each by its own rule.
    #[must_use]
    pub const fn new(folds: [Fold; 2]) -> Self {
        Self {
            folds,
            points: [[0.0; 2]; POINTS],
            len: 0,
            kept: [0.0; 2],
            count: 0,
        }
    }

    /// A block begins: the last block's points are let go. A chunk the
    /// last block left part-way goes on filling.
    pub fn begin(&mut self) {
        self.len = 0;
    }

    /// One sample's two numbers.
    pub fn sample(&mut self, a: f32, b: f32) {
        if self.count == 0 {
            self.kept = [self.folds[0].start(), self.folds[1].start()];
        }
        self.kept = [
            self.folds[0].fold(self.kept[0], a),
            self.folds[1].fold(self.kept[1], b),
        ];
        self.count += 1;
        if self.count == CHUNK {
            self.count = 0;
            if let Some(point) = self.points.get_mut(self.len) {
                *point = self.kept;
                self.len += 1;
            }
        }
    }

    /// The points the block made, oldest first.
    #[must_use]
    pub fn points(&self) -> &[[f32; 2]] {
        &self.points[..self.len]
    }

    /// Back to nothing, as a new one: no points, no chunk part-way.
    pub fn reset(&mut self) {
        self.len = 0;
        self.count = 0;
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    reason = "sample numbers of a few hundred"
)]
mod tests {
    use super::*;
    use expect_test::expect;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    fn show(trace: &Trace) -> String {
        trace
            .points()
            .iter()
            .map(|[a, b]| format!("[{a}, {b}]"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Blocks of 100 samples, sample n being (n, a gain that dips to 0.5 at
    /// sample 70 and swells to 3 at sample 150): a point every 64 samples,
    /// the chunk that straddles two blocks folded across them; the peak of
    /// each chunk is its last sample; the dip and the swell each land in
    /// the chunk they fall in, the swell winning its chunk over a smaller
    /// dip. `begin` lets the last block's points go; `reset` the chunk
    /// part-way too. A dip and a swell as far from one either way (0.5 and
    /// 2) leave the one that came first.
    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn a_trace_folds_each_chunk_into_a_point() {
        let mut trace = Trace::new([Fold::Max, Fold::FarthestFromOne]);
        let gain = |n: usize| match n {
            70 => 0.5,
            140 => 0.8,
            150 => 3.0,
            _ => 1.0,
        };
        let mut out = Vec::new();
        let mut n = 0;
        for block in 0..3 {
            trace.begin();
            for _ in 0..100 {
                trace.sample(n as f32, gain(n));
                n += 1;
            }
            out.push(format!("block {block}: {}", show(&trace)));
        }
        trace.begin();
        out.push(format!("begun: {}", show(&trace)));
        trace.reset();
        for i in 0..64 {
            trace.sample(i as f32, 1.0);
        }
        out.push(format!("reset, then 64: {}", show(&trace)));
        let mut tie = Trace::new([Fold::Max, Fold::FarthestFromOne]);
        for i in 0..64 {
            tie.sample(
                0.0,
                match i {
                    10 => 0.5,
                    20 => 2.0,
                    _ => 1.0,
                },
            );
        }
        out.push(format!("0.5 then 2: {}", show(&tie)));
        let mut low = Trace::new([Fold::Min, Fold::Min]);
        for _ in 0..CHUNK * (POINTS + 3) {
            low.sample(-1.0, 2.0);
        }
        out.push(format!(
            "{} chunks in one block: {} points kept, first {:?}",
            POINTS + 3,
            low.points().len(),
            low.points()[0]
        ));
        expect![[r#"
            block 0: [63, 1]
            block 1: [127, 0.5] [191, 3]
            block 2: [255, 1]
            begun: 
            reset, then 64: [63, 1]
            0.5 then 2: [0, 0.5]
            131 chunks in one block: 128 points kept, first [-1.0, 2.0]"#]]
        .assert_eq(&out.join("\n"));
    }
}
