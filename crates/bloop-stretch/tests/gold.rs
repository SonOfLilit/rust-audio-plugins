// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Gold tests: the port against Bungee, grain by grain.
//!
//! A scenario is an input signal and the list of grains asked of it. It
//! runs through the port, and the snapshot is what came out: how many
//! frames, where the output says it is in the input, each channel's level,
//! and a hash of every sample.
//!
//! With `--features gold` the same grains also run through Bungee Basic,
//! compiled unmodified from `vendor/bungee` (`tests/cpp_harness`), and the
//! two must agree. Not bit for bit: Bungee splits each spectrum into
//! partials by comparing neighbouring bins' energies, so the last bit of
//! an FFT decides where a partial ends whenever two bins are nearly equal,
//! which on noise is all the time — and the port's FFT is rustfft, not
//! PFFFT. The yardstick is Bungee against itself: the same grains run on
//! the input nudged by 1e-7 (−140 dB). The port must sit no further from
//! Bungee than three times Bungee's distance from itself under that nudge. On a passthrough,
//! where no partial is split, that is within a few ulps.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]

use bloop_stretch::{Request, SampleRates, Stretcher};
use expect_test::{expect, Expect};
use std::fmt::Write as _;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

struct Scenario {
    rates: SampleRates,
    /// Channel after channel.
    input: Vec<Vec<f32>>,
    grains: Vec<Request>,
}

struct Chunk {
    begin: f64,
    end: f64,
    channels: Vec<Vec<f32>>,
}

const RATE: u32 = 48000;

fn rates(input: u32, output: u32) -> SampleRates {
    SampleRates { input, output }
}

/// A deterministic noise in [-1, 1): the same on every target.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1 << 23) as f32 - 1.0
    }
}

fn seconds(rate: u32, s: f64) -> usize {
    (f64::from(rate) * s) as usize
}

/// A 220 Hz tone with eight harmonics falling as 1/k.
fn tone(rate: u32, s: f64) -> Vec<f32> {
    (0..seconds(rate, s))
        .map(|t| {
            let t = t as f64 / f64::from(rate);
            (1..=8)
                .map(|k| {
                    libm::sin(2.0 * core::f64::consts::PI * 220.0 * f64::from(k) * t + f64::from(k))
                        / f64::from(k)
                })
                .sum::<f64>() as f32
                * 0.2
        })
        .collect()
}

/// Two bars at 120 BPM: a kick on beats 1 and 3, a snare on 2 and 4, a
/// hat on every off-beat.
fn drums(rate: u32) -> Vec<f32> {
    let beat = seconds(rate, 0.5);
    let mut x = vec![0.0f32; beat * 8];
    let mut noise = Noise(0x1234_5678);
    for b in 0..8 {
        let start = b * beat;
        for i in 0..seconds(rate, 0.25) {
            let t = i as f64 / f64::from(rate);
            let v = if b % 2 == 0 {
                let phase = 2.0
                    * core::f64::consts::PI
                    * (50.0 * t + 100.0 / 30.0 * (1.0 - libm::exp(-30.0 * t)));
                libm::sin(phase) * libm::exp(-12.0 * t)
            } else {
                (f64::from(noise.next()) * 0.6
                    + libm::sin(2.0 * core::f64::consts::PI * 190.0 * t) * 0.5)
                    * libm::exp(-25.0 * t)
            };
            x[start + i] += 0.5 * v as f32;
        }
        let hat = start + beat / 2;
        for i in 0..seconds(rate, 0.05) {
            let t = i as f64 / f64::from(rate);
            x[hat + i] += 0.15 * noise.next() * libm::exp(-80.0 * t) as f32;
        }
    }
    x
}

/// `count` grains from `position` at a steady speed and pitch, preceded by
/// Bungee's preroll — the way a caller starts a stretch.
fn steady(
    stretcher: &Stretcher,
    position: f64,
    speed: f64,
    pitch: f64,
    count: usize,
) -> Vec<Request> {
    let mut request = Request::new(position, speed, pitch);
    request.reset = true;
    stretcher.preroll(&mut request);
    let mut grains = Vec::new();
    for _ in 0..count {
        grains.push(request);
        stretcher.next(&mut request);
    }
    grains
}

/// `count` NaN grains: the stretcher plays out what it holds.
fn flush(count: usize) -> Vec<Request> {
    vec![Request::new(f64::NAN, 1.0, 1.0); count]
}

fn run_port(scenario: &Scenario) -> Vec<Chunk> {
    let channels = scenario.input.len();
    let frames = scenario.input[0].len() as i64;
    let mut stretcher = Stretcher::new(scenario.rates, channels);
    let mut chunk = Vec::new();
    scenario
        .grains
        .iter()
        .map(|request| {
            let needed = stretcher.specify_grain(request, 0.0);
            let len = needed.len();
            chunk.clear();
            for channel in &scenario.input {
                chunk.extend((0..len).map(|i| {
                    let p = i64::from(needed.begin) + i as i64;
                    if (0..frames).contains(&p) {
                        channel[p as usize]
                    } else {
                        0.0
                    }
                }));
            }
            stretcher.analyse_grain(&chunk);
            let made = stretcher.synthesise_grain();
            Chunk {
                begin: made.begin.position,
                end: made.end.position,
                channels: (0..channels)
                    .map(|c| stretcher.output(c).to_vec())
                    .collect(),
            }
        })
        .collect()
}

fn db(x: f64) -> f64 {
    if x <= 0.0 {
        -999.0
    } else {
        20.0 * libm::log10(x)
    }
}

fn summary(chunks: &[Chunk]) -> String {
    let mut out = String::new();
    let frames: usize = chunks.iter().map(|c| c.channels[0].len()).sum();
    let first = chunks.iter().find(|c| !c.begin.is_nan());
    let last = chunks.iter().rev().find(|c| !c.end.is_nan());
    let _ = writeln!(out, "{} grains, {frames} frames out", chunks.len());
    if let (Some(first), Some(last)) = (first, last) {
        let _ = writeln!(out, "plays input {:.1} .. {:.1}", first.begin, last.end);
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for c in 0..chunks[0].channels.len() {
        let samples: Vec<f32> = chunks
            .iter()
            .flat_map(|k| k.channels[c].iter().copied())
            .collect();
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let rms = (samples.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>()
            / samples.len().max(1) as f64)
            .sqrt();
        let _ = writeln!(
            out,
            "ch{c}: peak {:.1} dB, rms {:.1} dB",
            db(f64::from(peak)),
            db(rms)
        );
        for s in &samples {
            for byte in s.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    let _ = writeln!(out, "hash {hash:016x}");
    out
}

fn check(scenario: &Scenario, expect: &Expect) {
    let port = run_port(scenario);
    expect.assert_eq(&summary(&port));
    #[cfg(feature = "gold")]
    gold::compare(scenario, &port);
}

fn mono(input: Vec<f32>) -> Vec<Vec<f32>> {
    vec![input]
}

fn stretcher(rates: SampleRates) -> Stretcher {
    Stretcher::new(rates, 1)
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn speed_one_passes_the_input_through() {
    let r = rates(RATE, RATE);
    let input = tone(RATE, 1.0);
    let right: Vec<f32> = input.iter().map(|s| -0.5 * s).collect();
    check(
        &Scenario {
            rates: r,
            grains: steady(&Stretcher::new(r, 2), 0.0, 1.0, 1.0, 96),
            input: vec![input, right],
        },
        &expect![[r#"
            96 grains, 49152 frames out
            plays input -512.0 .. 47616.0
            ch0: peak -9.5 dB, rms -15.3 dB
            ch1: peak -15.5 dB, rms -21.3 dB
            hash 52fa742e5f8125f3
        "#]],
    );
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_tone_stretched() {
    let r = rates(RATE, RATE);
    for (speed, expect) in [
        (
            0.5,
            expect![[r#"
            120 grains, 61440 frames out
            plays input -256.0 .. 29952.0
            ch0: peak -6.8 dB, rms -15.2 dB
            hash 8451e04a71093d86
        "#]],
        ),
        (
            0.8,
            expect![[r#"
            120 grains, 61440 frames out
            plays input -409.6 .. 47923.2
            ch0: peak -8.0 dB, rms -15.3 dB
            hash 01667317394fa7de
        "#]],
        ),
        (
            1.25,
            expect![[r#"
            120 grains, 61440 frames out
            plays input -640.0 .. 74880.0
            ch0: peak -7.2 dB, rms -15.3 dB
            hash 7b449444697c0d5d
        "#]],
        ),
        (
            2.0,
            expect![[r#"
            120 grains, 61440 frames out
            plays input -1024.0 .. 119808.0
            ch0: peak -5.5 dB, rms -16.2 dB
            hash 0e6656acdd499488
        "#]],
        ),
    ] {
        check(
            &Scenario {
                rates: r,
                grains: steady(&stretcher(r), 0.0, speed, 1.0, 120),
                input: mono(tone(RATE, 2.0)),
            },
            &expect,
        );
    }
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn drums_stretched() {
    let r = rates(RATE, RATE);
    for (speed, expect) in [
        (
            0.5,
            expect![[r#"
        150 grains, 76800 frames out
        plays input -256.0 .. 37632.0
        ch0: peak -5.0 dB, rms -22.7 dB
        hash 6ef2187a14f0e4f3
    "#]],
        ),
        (
            2.0,
            expect![[r#"
        150 grains, 76800 frames out
        plays input -1024.0 .. 150528.0
        ch0: peak -4.9 dB, rms -20.5 dB
        hash 5758b20baddea0bb
    "#]],
        ),
    ] {
        check(
            &Scenario {
                rates: r,
                grains: steady(&stretcher(r), 0.0, speed, 1.0, 150),
                input: mono(drums(RATE)),
            },
            &expect,
        );
    }
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_tone_pitched() {
    let r = rates(RATE, RATE);
    for (pitch, expect) in [
        (
            0.7,
            expect![[r#"
        120 grains, 87423 frames out
        plays input -731.4 .. 85577.1
        ch0: peak -6.6 dB, rms -15.2 dB
        hash be93b2e1f813f5be
    "#]],
        ),
        (
            1.4983,
            expect![[r#"
        120 grains, 41245 frames out
        plays input -341.7 .. 39981.3
        ch0: peak -6.5 dB, rms -15.3 dB
        hash 77339d56acd2e528
    "#]],
        ),
    ] {
        check(
            &Scenario {
                rates: r,
                grains: steady(&stretcher(r), 0.0, 1.0, pitch, 120),
                input: mono(tone(RATE, 2.0)),
            },
            &expect,
        );
    }
}

/// The pitch glides up a fifth and back to 1 while the speed changes: the
/// resampler's ratio ramps across each chunk, and lands exactly on the hop
/// when the pitch returns to 1.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn pitch_and_speed_change_every_grain() {
    let r = rates(RATE, RATE);
    let s = stretcher(r);
    let mut request = Request::new(0.0, 1.0, 1.0);
    request.reset = true;
    s.preroll(&mut request);
    let mut grains = Vec::new();
    for g in 0..140 {
        let x = f64::from(g) / 140.0;
        request.pitch = if g < 100 {
            1.0 + 0.5 * libm::sin(core::f64::consts::PI * x / 0.714)
        } else {
            1.0
        };
        request.speed = 0.75 + 0.5 * x;
        grains.push(request);
        s.next(&mut request);
    }
    check(
        &Scenario {
            rates: r,
            grains,
            input: mono(tone(RATE, 2.5)),
        },
        &expect![[r#"
            140 grains, 59899 frames out
            plays input -512.0 .. 58827.4
            ch0: peak -8.7 dB, rms -15.3 dB
            hash c9425ca3eef39fc4
        "#]],
    );
}

/// A jump back to the start, as a loop wrapping: a reset grain, then on.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_jump_with_a_reset() {
    let r = rates(RATE, RATE);
    let s = stretcher(r);
    let mut grains = steady(&s, 0.0, 0.8, 1.0, 60);
    grains.extend(steady(&s, 0.0, 0.8, 1.0, 60));
    check(
        &Scenario {
            rates: r,
            grains,
            input: mono(drums(RATE)),
        },
        &expect![[r#"
            120 grains, 61440 frames out
            plays input -409.6 .. 23347.2
            ch0: peak -6.2 dB, rms -20.4 dB
            hash 490695f85ab272c1
        "#]],
    );
}

/// A 44.1 kHz input played at 48 kHz: the output is resampled even at
/// pitch 1.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_different_input_rate() {
    let r = rates(44100, RATE);
    check(
        &Scenario {
            rates: r,
            grains: steady(&stretcher(r), 0.0, 1.0, 1.0, 100),
            input: mono(tone(44100, 1.5)),
        },
        &expect![[r#"
            100 grains, 55659 frames out
            plays input -512.0 .. 49664.0
            ch0: peak -9.5 dB, rms -15.3 dB
            hash 0160b0115ea42779
        "#]],
    );
}

/// Backwards from the end at speed -1.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn played_backwards() {
    let r = rates(RATE, RATE);
    let input = tone(RATE, 1.0);
    let end = input.len() as f64;
    check(
        &Scenario {
            rates: r,
            grains: steady(&stretcher(r), end, -1.0, 1.0, 90),
            input: mono(input),
        },
        &expect![[r#"
            90 grains, 46080 frames out
            plays input 48512.0 .. 3456.0
            ch0: peak -9.5 dB, rms -15.3 dB
            hash 7797a2d54e96fb14
        "#]],
    );
}

/// NaN grains after the input: the last of it plays out, then silence,
/// and the stretcher says it is flushed.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn flushing() {
    let r = rates(RATE, RATE);
    let mut grains = steady(&stretcher(r), 0.0, 1.0, 1.0, 20);
    grains.extend(flush(6));
    let scenario = Scenario {
        rates: r,
        grains,
        input: mono(tone(RATE, 0.5)),
    };
    let port = run_port(&scenario);
    let tail: Vec<String> = port[20..]
        .iter()
        .map(|c| {
            let peak = c.channels[0].iter().fold(0.0f32, |m, s| m.max(s.abs()));
            format!(
                "{} frames peak {peak:.3} begins {}",
                c.channels[0].len(),
                c.begin
            )
        })
        .collect();
    let mut s = Stretcher::new(r, 1);
    for request in &scenario.grains {
        let needed = s.specify_grain(request, 0.0);
        s.analyse_grain(&vec![0.0; needed.len()]);
        s.synthesise_grain();
    }
    expect![[r#"
        512 frames peak 0.335 begins 8704
        512 frames peak 0.219 begins 9216
        512 frames peak 0.065 begins NaN
        512 frames peak 0.000 begins NaN
        512 frames peak 0.000 begins NaN
        512 frames peak 0.000 begins NaN
        flushed true"#]]
    .assert_eq(&format!("{}\nflushed {}", tail.join("\n"), s.is_flushed()));
    #[cfg(feature = "gold")]
    gold::compare(&scenario, &port);
}

#[cfg(feature = "gold")]
#[expect(
    clippy::expect_used,
    reason = "a harness that cannot build or run the original has nothing to compare"
)]
mod gold {
    use super::{Chunk, Scenario};
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::OnceLock;

    fn bungee() -> PathBuf {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/bungee");
        assert!(
            path.join("src/Stretcher.cpp").exists() && path.join("submodules/eigen/Eigen").exists(),
            "vendor/bungee and its eigen and pffft submodules are not checked out. The gold tests compile Bungee from them:\n  \
             git -c submodule.vendor/bungee.update=checkout submodule update --init vendor/bungee\n  \
             git -C vendor/bungee submodule update --init --depth 1 submodules/eigen submodules/pffft"
        );
        path
    }

    fn build() -> PathBuf {
        let b = bungee();
        let head = Command::new("git")
            .arg("-C")
            .arg(&b)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("run git to read vendor/bungee's revision");
        assert_eq!(
            String::from_utf8_lossy(&head.stdout).trim(),
            bloop_stretch::BUNGEE_REV,
            "vendor/bungee is not at the revision the port (and its License notice) cites"
        );
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bungee-gold");
        std::fs::create_dir_all(&out).expect("create the harness build directory");
        let binary = out.join("harness");
        let harness = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cpp_harness/main.cpp");
        let newest = |p: &Path| p.metadata().and_then(|m| m.modified()).ok();
        if binary.exists() && newest(&binary) >= newest(&harness) {
            return binary;
        }
        let pffft = out.join("pffft.o");
        let status = Command::new("gcc")
            .args(["-O2", "-c", "-ffast-math", "-fno-finite-math-only"])
            .arg(b.join("submodules/pffft/pffft.c"))
            .arg("-o")
            .arg(&pffft)
            .status()
            .expect("run gcc (the gold tests need it installed)");
        assert!(status.success(), "compiling pffft failed");
        let sources: Vec<PathBuf> = std::fs::read_dir(b.join("src"))
            .expect("read vendor/bungee/src")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "cpp"))
            .collect();
        let status = Command::new("g++")
            .args([
                "-std=c++20",
                "-O2",
                "-fwrapv",
                "-DEIGEN_DONT_PARALLELIZE=1",
                "-DBUNGEE_SELF_TEST=0",
                "-DBUNGEE_VERSION=\"2.4.10\"",
                "-Deigen_assert=BUNGEE_ASSERT1",
            ])
            .arg("-I")
            .arg(b.join("submodules/eigen"))
            .arg("-I")
            .arg(b.join("submodules"))
            .arg("-I")
            .arg(&b)
            .arg(&harness)
            .args(&sources)
            .arg(&pffft)
            .arg("-o")
            .arg(&binary)
            .status()
            .expect("run g++");
        assert!(status.success(), "compiling the Bungee gold harness failed");
        binary
    }

    fn binary() -> &'static Path {
        static BINARY: OnceLock<PathBuf> = OnceLock::new();
        BINARY.get_or_init(build)
    }

    fn run(scenario: &Scenario, input: &[Vec<f32>]) -> Vec<Chunk> {
        let channels = input.len();
        let mut program = Vec::new();
        for v in [
            scenario.rates.input,
            scenario.rates.output,
            channels as u32,
            input[0].len() as u32,
        ] {
            program.extend_from_slice(&v.to_le_bytes());
        }
        for channel in input {
            for s in channel {
                program.extend_from_slice(&s.to_le_bytes());
            }
        }
        program.extend_from_slice(&(scenario.grains.len() as u32).to_le_bytes());
        for g in &scenario.grains {
            for v in [g.position, g.speed, g.pitch] {
                program.extend_from_slice(&v.to_le_bytes());
            }
            program.push(u8::from(g.reset));
        }
        let mut child = Command::new(binary())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the gold harness");
        let mut stdin = child.stdin.take().expect("harness stdin");
        let writer = std::thread::spawn(move || stdin.write_all(&program));
        let output = child.wait_with_output().expect("run the gold harness");
        writer
            .join()
            .expect("join the writer")
            .expect("write the program");
        assert!(
            output.status.success(),
            "the harness failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let bytes = output.stdout;
        let mut at = 0;
        let mut take = |n: usize| {
            let s = &bytes[at..at + n];
            at += n;
            s.to_vec()
        };
        (0..scenario.grains.len())
            .map(|_| {
                let frames = u32::from_le_bytes(take(4).try_into().expect("4 bytes")) as usize;
                let begin = f64::from_le_bytes(take(8).try_into().expect("8 bytes"));
                let end = f64::from_le_bytes(take(8).try_into().expect("8 bytes"));
                let channels = (0..channels)
                    .map(|_| {
                        take(4 * frames)
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect()
                    })
                    .collect();
                Chunk {
                    begin,
                    end,
                    channels,
                }
            })
            .collect()
    }

    fn flat(chunks: &[Chunk]) -> Vec<f32> {
        chunks
            .iter()
            .flat_map(|c| c.channels.iter().flatten().copied())
            .collect()
    }

    fn rms_difference(a: &[f32], b: &[f32]) -> f64 {
        (a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(x - y).powi(2))
            .sum::<f64>()
            / a.len().max(1) as f64)
            .sqrt()
    }

    pub(super) fn compare(scenario: &Scenario, port: &[Chunk]) {
        let original = run(scenario, &scenario.input);
        let mut noise = super::Noise(0x9e37_79b9);
        let nudged_input: Vec<Vec<f32>> = scenario
            .input
            .iter()
            .map(|c| c.iter().map(|s| s + 1e-7 * noise.next()).collect())
            .collect();
        let nudged = run(scenario, &nudged_input);

        for (g, (p, o)) in port.iter().zip(&original).enumerate() {
            assert_eq!(
                p.channels[0].len(),
                o.channels[0].len(),
                "grain {g}: frames differ"
            );
            assert!(
                p.begin.to_bits() == o.begin.to_bits() || (p.begin.is_nan() && o.begin.is_nan()),
                "grain {g}: begins at {} where Bungee begins at {}",
                p.begin,
                o.begin
            );
        }
        let (p, o, n) = (flat(port), flat(&original), flat(&nudged));
        let port_off = rms_difference(&p, &o);
        let bungee_off = rms_difference(&n, &o);
        // Three times, because where a nudge happens to tip a partial is
        // luck: a scenario's yardstick moves by half from one nudge to the
        // next.
        assert!(
            port_off <= 3.0 * bungee_off + 1e-7,
            "the port is {port_off:.3e} rms from Bungee, where Bungee is {bungee_off:.3e} from itself under a 1e-7 nudge"
        );
    }
}
