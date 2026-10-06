//! What each NAM model shape costs on this machine, one core: ten seconds
//! of a guitar-level signal through it in blocks of 128 at 48 kHz, with
//! the fast and the exact `tanh`, reported as nanoseconds a sample and as
//! a share of real time.
//!
//! `cargo run -q --release -p bloop-plugin-nam --example bench_nam`
//!
//! A `.nam` file given as an argument is timed too; `--write <dir>`
//! writes the shapes timed here as `.nam` files instead, for NAM's own
//! `benchmodel` to time.

#![expect(
    clippy::print_stdout,
    reason = "a benchmark reports on standard output"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "sample counts and nanoseconds as rates"
)]

use std::time::Instant;

use bloop_plugin_nam::preset::Shape;
use bloop_plugin_nam::{Model, Size, Tanh};

const RATE: f64 = 48_000.0;
const BLOCK: usize = 128;
const SECONDS: usize = 2;
/// Runs of each; the fastest is reported, which a busy machine slows least.
const RUNS: usize = 15;

fn signal(frames: usize) -> Vec<f32> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    (0..frames)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
            let t = i as f32 / RATE as f32;
            0.3 * libm::sinf(2.0 * core::f32::consts::PI * 110.0 * t) + 0.05 * noise
        })
        .collect()
}

fn shapes() -> impl Iterator<Item = Shape> {
    Shape::WAVENETS
        .into_iter()
        .chain([Shape::A2Full, Shape::A2Lite])
        .chain([8, 16, 24, 32].map(Shape::Lstm))
}

fn time(model: &mut Model, input: &[f32], tanh: Tanh) -> f64 {
    let mut out = vec![0.0; BLOCK];
    let mut best = f64::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        for block in input.chunks(BLOCK) {
            model.process(block, &mut out[..block.len()], tanh);
        }
        std::hint::black_box(&out);
        best = best.min(start.elapsed().as_secs_f64());
    }
    best * 1e9 / input.len() as f64
}

fn report(name: &str, model: &mut Model, input: &[f32]) {
    let fast = time(model, input, Tanh::Fast);
    let exact = time(model, input, Tanh::Exact);
    let share = |ns: f64| ns * RATE / 1e9 * 100.0;
    println!(
        "{name:<14} {:>6} MAC/sample  fast {fast:>7.1} ns/sample {:>5.1}%   exact {exact:>7.1} ns/sample {:>5.1}%",
        model.macs_per_sample(),
        share(fast),
        share(exact),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [flag, dir] = args.as_slice() {
        if flag == "--write" {
            for shape in shapes() {
                let path = format!("{dir}/{}.nam", shape.name());
                if let Err(e) = std::fs::write(&path, shape.nam(1)) {
                    println!("{path}: {e}");
                }
            }
            return;
        }
    }
    let input = signal(SECONDS * RATE as usize);
    println!("one core, 48 kHz, best of {RUNS} runs of {SECONDS} s; % is of real time");
    for shape in shapes() {
        let Ok(mut model) = Model::parse(&shape.nam(1)) else {
            println!("{}: does not load", shape.name());
            continue;
        };
        report(&shape.name(), &mut model, &input);
    }
    for path in args {
        match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| Model::parse(&t))
        {
            Ok(mut model) => {
                report(&path, &mut model, &input);
                if model.has_light() {
                    model.set_size(Size::Light);
                    report(&format!("{path} (light)"), &mut model, &input);
                }
            }
            Err(e) => println!("{path}: {e}"),
        }
    }
}
