// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Ported from Bungee's src/Fourier.{h,cpp} (Copyright (C) 2020-2026
// Parabola Research Limited, SPDX-License-Identifier: MPL-2.0).

//! The real transforms the stretcher runs, unnormalised both ways as
//! Bungee's PFFFT ones are: a forward then an inverse multiplies by the
//! length.
//!
//! PFFFT is not ported. A real transform of length `n` is a complex one of
//! `n / 2` (rustfft's scalar planner) and a twiddle pass, with the twiddles
//! from libm. The scalar planner picks the same algorithm on every target
//! and nothing here calls the platform's maths, so a stretch renders the
//! same samples on wasm as on native — the SIMD planners would not.

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlannerScalar};
use std::sync::Arc;

pub(crate) struct RealFft {
    half: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    /// `e^(-2 pi i k / n)` for `k` in `0..=n/2`.
    twiddles: Vec<Complex32>,
    buffer: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl RealFft {
    /// Plan both directions at `length` (a power of two, at least 4).
    /// Allocates; build it on the UI thread.
    pub(crate) fn new(length: usize) -> Self {
        let half = length / 2;
        let mut planner = FftPlannerScalar::<f32>::new();
        let forward = planner.plan_fft_forward(half);
        let inverse = planner.plan_fft_inverse(half);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let twiddles = (0..=half)
            .map(|k| {
                let angle = -2.0 * core::f64::consts::PI * k as f64 / length as f64;
                Complex32::new(libm::cos(angle) as f32, libm::sin(angle) as f32)
            })
            .collect();
        Self {
            half,
            forward,
            inverse,
            twiddles,
            buffer: vec![Complex32::default(); half],
            scratch: vec![Complex32::default(); scratch_len],
        }
    }

    /// `out[k]` for `k` in `0..=n/2` is bin `k` of `input`'s transform.
    pub(crate) fn forward(&mut self, input: &[f32], out: &mut [Complex32]) {
        let m = self.half;
        for (k, z) in self.buffer.iter_mut().enumerate() {
            *z = Complex32::new(input[2 * k], input[2 * k + 1]);
        }
        self.forward
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        for (k, out) in out[..=m].iter_mut().enumerate() {
            let zk = self.buffer[k % m];
            let zc = self.buffer[(m - k) % m].conj();
            let even = (zk + zc) * 0.5;
            let diff = (zk - zc) * 0.5;
            let odd = Complex32::new(diff.im, -diff.re);
            *out = even + self.twiddles[k] * odd;
        }
    }

    /// `out` is the length-`n` signal whose bins are `bins[0..=n/2]`, times
    /// `n`. As PFFFT does, the imaginary parts of bin 0 and bin `n/2` are
    /// not read.
    pub(crate) fn inverse(&mut self, bins: &[Complex32], out: &mut [f32]) {
        let m = self.half;
        let bin = |k: usize| {
            let x = bins[k];
            if k == 0 || k == m {
                Complex32::new(x.re, 0.0)
            } else {
                x
            }
        };
        for k in 0..m {
            let xk = bin(k);
            let xc = bin(m - k).conj();
            let even = xk + xc;
            let odd = (xk - xc) * self.twiddles[k].conj();
            self.buffer[k] = even + Complex32::new(-odd.im, odd.re);
        }
        self.inverse
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        for (k, z) in self.buffer.iter().enumerate() {
            out[2 * k] = z.re;
            out[2 * k + 1] = z.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    fn dft(x: &[f32]) -> Vec<(f64, f64)> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                x.iter().enumerate().fold((0.0, 0.0), |(re, im), (t, &v)| {
                    let a = -2.0 * core::f64::consts::PI * (k * t) as f64 / n as f64;
                    (re + f64::from(v) * a.cos(), im + f64::from(v) * a.sin())
                })
            })
            .collect()
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn forward_is_the_dft_and_inverse_undoes_it_times_the_length() {
        let n = 64;
        let x: Vec<f32> = (0..n)
            .map(|t| ((t * 7919 % 113) as f32 / 56.0) - 1.0)
            .collect();
        let mut fft = RealFft::new(n);
        let mut bins = vec![Complex32::default(); n / 2 + 1];
        fft.forward(&x, &mut bins);
        let worst_bin = dft(&x)
            .iter()
            .zip(&bins)
            .map(|(&(re, im), b)| (re - f64::from(b.re)).hypot(im - f64::from(b.im)))
            .fold(0.0, f64::max);
        let mut back = vec![0.0; n];
        fft.inverse(&bins, &mut back);
        let worst_sample = x
            .iter()
            .zip(&back)
            .map(|(a, b)| (f64::from(*a) * n as f64 - f64::from(*b)).abs())
            .fold(0.0, f64::max);
        assert!(worst_bin < 1e-4, "forward is off the DFT by {worst_bin}");
        assert!(worst_sample < 1e-3, "inverse is off by {worst_sample}");
    }
}
