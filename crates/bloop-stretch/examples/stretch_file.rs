// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Stretch a raw mono f32 file: `stretch_file in.f32 out.f32 rate speed
//! pitch`. Prints the frames made and how many times faster than real time
//! the stretch ran, for measuring the stretcher outside the app.

use bloop_stretch::{Request, SampleRates, Stretcher};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output, rate, speed, pitch] = args.as_slice() else {
        return Err("usage: stretch_file in.f32 out.f32 rate speed pitch".into());
    };
    let samples: Vec<f32> = std::fs::read(input)?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let rate: u32 = rate.parse()?;
    let speed: f64 = speed.parse()?;
    let pitch: f64 = pitch.parse()?;
    let n = samples.len() as i64;
    let want = (samples.len() as f64 / speed) as usize;

    let mut stretcher = Stretcher::new(
        SampleRates {
            input: rate,
            output: rate,
        },
        1,
    );
    let mut chunk = vec![0.0; stretcher.max_input_frame_count() * 4 + 16];
    let mut out = Vec::with_capacity(want + 10000);
    let mut request = Request::new(0.0, speed, pitch);
    request.reset = true;
    stretcher.preroll(&mut request);

    let start = Instant::now();
    let mut started = false;
    while out.len() < want {
        let needed = stretcher.specify_grain(&request, 0.0);
        let len = needed.len();
        if chunk.len() < len {
            chunk.resize(len, 0.0);
        }
        for (i, s) in chunk[..len].iter_mut().enumerate() {
            let p = i64::from(needed.begin) + i as i64;
            *s = if (0..n).contains(&p) {
                samples[p as usize]
            } else {
                0.0
            };
        }
        stretcher.analyse_grain(&chunk[..len]);
        let made = stretcher.synthesise_grain();
        stretcher.next(&mut request);
        let (b, e) = (made.begin.position, made.end.position);
        for (i, &s) in stretcher.output(0).iter().enumerate() {
            let pos = b + (e - b) * i as f64 / made.frames as f64;
            if !started && (pos.is_nan() || pos < 0.0) {
                continue;
            }
            started = true;
            out.push(s);
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let made = out.len();
    out.truncate(want);
    let bytes: Vec<u8> = out.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(output, bytes)?;
    #[expect(clippy::print_stderr, reason = "a measuring tool's one line of output")]
    {
        eprintln!(
            "frames={made} cpu={seconds:.4}s realtime_x={:.1}",
            made as f64 / f64::from(rate) / seconds
        );
    }
    Ok(())
}
