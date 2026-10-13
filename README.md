# rust-audio-plugins

Rust ports of open source audio plug-ins and DSP, written for bloop, a
mobile groovebox, and published here with their history so that their
sources are available under their own licences.

**Each crate has its own licence**, given in its `Cargo.toml` and its
`LICENSE` file (and `NOTICE`, where the licence asks for one). There is no
licence for the repository as a whole.

| crate | what it is | licence |
|-------|------------|---------|
| `bloop-note` | `Note`, a MIDI note number: the one type of the host's the crates below need | MIT |
| `bloop-events` | `NoteEvent`, `EventBuffer` (allocation-free), MIDI parsing, the pad velocity curve | MIT |
| `bloop-plugin-api` | the `Plugin` trait every plug-in here implements, its meters and traces | MIT |
| `bloop-plugin-dexed` | FM synth: a port of MSFA as [Dexed](https://github.com/asb2m10/dexed) maintains it, with the DX7 ROM cartridges and a SysEx reader | Apache-2.0 |
| `bloop-sampler-engine` | sample playback: a port of [Just-a-Sample](https://github.com/BOBONA/Just-a-Sample)'s `CustomSamplerVoice` (BASIC mode), and decoding, editing, onsets, recording | MIT |
| `bloop-stretch` | time-stretch and pitch-shift: a port of [Bungee](https://github.com/bungee-audio-stretch/bungee) Basic | MPL-2.0, every file |
| `bloop-plugin-socalabs` | 18 effects from SocaLabs' [slPlugins](https://github.com/FigBug/slPlugins) and [Gin](https://github.com/FigBug/Gin) | BSD-3-Clause AND MIT AND ISC AND WTFPL |
| `bloop-plugin-wavetable` | SocaLabs [Wavetable](https://github.com/FigBug/Wavetable), its 217 presets and its wavetables (`wavetables/`, packed by `tools/import.py`) | BSD-3-Clause AND MIT AND CC0-1.0 |
| `bloop-plugin-nam` | [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore): WaveNet (A1, A2) and LSTM `.nam` players, the plug-in's tone stack, bundled captures | MIT |
| `bloop-plugin-airwindows` | six [Airwindows](https://github.com/airwindows/airwindows) plug-ins, ported bit-exact | MIT |
| `bloop-plugin-airwindows-cpp` | every Airwindows plug-in of the LinuxVST collection, compiled from the C++ as it is | MIT |

The `bloop-` names are kept so the crates keep their names in bloop.

## Building

```sh
git clone https://github.com/SonOfLilit/rust-audio-plugins.git
cd rust-audio-plugins
git submodule update --init --recursive        # vendor/just-a-sample and vendor/airwindows
cargo test --workspace
```

`bloop-plugin-airwindows-cpp` compiles the Airwindows C++ in its build
script, so it needs `vendor/airwindows` (checked out with the rest), or
`BLOOP_AIRWINDOWS_SRC` pointing at a checkout of it. On wasm32 it builds with no plug-ins in it.

Tests over about 1.5 s are `#[ignore = "heavy: …"]`; run them with
`cargo test -p <crate> -- --include-ignored`.

## Gold tests

Each port is checked against the original it ports, compiled from the
submodule under `vendor/`, at the revision the crate cites. Just-a-Sample
and Airwindows come with a plain `git submodule update --init`, since a
build needs them; the rest are `update = none`: fetch only those you need.

| crate | run | needs |
|-------|-----|-------|
| `bloop-sampler-engine` | `cargo test -p bloop-sampler-engine` (always) | `vendor/just-a-sample` |
| `bloop-stretch` | `cargo test -p bloop-stretch --features gold` | `vendor/bungee`, then `git -C vendor/bungee submodule update --init --depth 1 submodules/eigen submodules/pffft` |
| `bloop-plugin-socalabs` | `cargo test -p bloop-plugin-socalabs --features gold -- --include-ignored` | `vendor/slplugins`, `vendor/gin`, `vendor/juce` |
| `bloop-plugin-wavetable` | `cargo test -p bloop-plugin-wavetable --features gold -- --include-ignored` | `vendor/wavetable`, `vendor/gin`, `vendor/juce`, here or under `BLOOP_GOLD_VENDOR` |
| `bloop-plugin-nam` | `cargo test -p bloop-plugin-nam --features gold -- --include-ignored` | `vendor/nam`, then `git -C vendor/nam submodule update --init --depth 1 Dependencies/eigen` |
| `bloop-plugin-airwindows` | `cargo test -p bloop-plugin-airwindows --features gold -- --include-ignored` | `vendor/airwindows` |

Fetch one with

```sh
git -c submodule.vendor/gin.update=checkout submodule update --init vendor/gin
```

What "matches" means differs by port:

- **SocaLabs, Wavetable, Airwindows**: bit for bit. `gold` switches the
  port's math to `std` (the C library the C++ links); the default is `libm`,
  the same on every target. Wavetable's harness is handed the port's own
  band-limited tables, since JUCE's FFT and the port's differ by up to 8e-7.
- **Just-a-Sample**: bit for bit until the interpolation window was
  tabulated; each scenario pins its divergence, worst 3.0e-7.
- **NAM**: within 1e-4, as it sums matrix products in another order than
  Eigen.
- **Bungee**: its partial split is chaotic on noise, so the port must stay
  within 3x of Bungee's own distance from itself under a 1e-7 nudge.

A worktree whose `vendor/` is empty can point `BLOOP_GOLD_VENDOR` (for
Wavetable) and `BLOOP_AIRWINDOWS_SRC` at a checkout that has them.
