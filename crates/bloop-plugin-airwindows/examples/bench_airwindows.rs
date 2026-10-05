//! What each Airwindows port costs on this machine: ten seconds of stereo
//! noise through it in blocks of 128 at 48 kHz, best of three, reported as
//! nanoseconds a frame and as a share of real time (one core). The noise is
//! the one the C++ timing driver feeds the originals, so the two compare.
//!
//! `cargo run -q --release -p bloop-plugin-airwindows --example bench_airwindows`

#![expect(
    clippy::print_stdout,
    reason = "a benchmark reports on standard output"
)]

use std::time::Instant;

use bloop_plugin_airwindows::{Airwindows, Kind};

const RATE: f32 = 48_000.0;
const BLOCK: usize = 128;
const SECONDS: usize = 10;

/// Noise at 0.3 of full scale, the left an xorshift64 and the right the
/// same samples reordered.
fn noise(frames: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let left: Vec<f32> = (0..frames)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.3
        })
        .collect();
    let right = (0..frames).map(|i| left[(i * 7) % frames]).collect();
    (left, right)
}

fn main() {
    let frames = SECONDS * 48_000;
    let (in_left, in_right) = noise(frames);
    for kind in Kind::ALL {
        let mut plugin = Airwindows::new(kind, RATE);
        let mut left = vec![0.0; BLOCK];
        let mut right = vec![0.0; BLOCK];
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = Instant::now();
            for (l, r) in in_left.chunks(BLOCK).zip(in_right.chunks(BLOCK)) {
                left[..l.len()].copy_from_slice(l);
                right[..r.len()].copy_from_slice(r);
                plugin.process(&mut left[..l.len()], &mut right[..r.len()]);
            }
            best = best.min(start.elapsed().as_secs_f64());
            std::hint::black_box(&left);
        }
        println!(
            "{:<16} {:>6.1} ns/frame  {:>5.2}% of real time",
            kind.name(),
            best * 1e9 / frames as f64,
            100.0 * best / SECONDS as f64,
        );
    }
}
