//! Slimmable WaveNets (`NAM/wavenet/slimmable.cpp`): a WaveNet trained so
//! that the first few channels of each array, alone, still make a model.
//! Running it slimmed is taking those channels' weights out of the full
//! set and building the smaller WaveNet from them.
//!
//! How much to slim is a value from 0 to 1 (`SetSlimmableSize`). An array
//! that may run at `n` channel counts runs at the one `⌊value · n⌋` picks
//! (`ratio_to_channels`); a `SlimmableContainer` holds whole models
//! instead, each covering the values up to its `max_value`.

use crate::config::{ArrayConfig, WaveNetConfig};
use crate::wavenet::Weights;

/// The channels each array runs at for a slim value of `value`.
#[must_use]
pub fn channels_for(config: &WaveNetConfig, value: f64) -> Vec<usize> {
    config
        .arrays
        .iter()
        .map(|a| {
            if a.slimmable.is_empty() {
                a.channels
            } else {
                let n = a.slimmable.len();
                let idx = ((value * n as f64).floor() as usize).min(n - 1);
                a.slimmable[idx]
            }
        })
        .collect()
}

/// Where the slim value changes what runs, ascending, inside (0, 1)
/// (`get_ratio_breakpoints`).
#[must_use]
pub fn breakpoints(config: &WaveNetConfig) -> Vec<f64> {
    let mut b: Vec<f64> = config
        .arrays
        .iter()
        .flat_map(|a| {
            let n = a.slimmable.len();
            (1..n).map(move |i| i as f64 / n as f64)
        })
        .collect();
    b.sort_by(f64::total_cmp);
    b.dedup();
    b
}

/// `compute_slim_bottleneck`.
fn slim_bottleneck(a: &ArrayConfig, channels: usize) -> usize {
    if a.layer1x1.is_none() {
        channels
    } else {
        (a.bottleneck * channels / a.channels).max(1)
    }
}

/// Checks a slimmable config the way `SlimmableWavenet` does.
///
/// # Errors
/// What the core refuses: groups, a head convolution longer than 1, a
/// post-stack head, an allowed list not ascending to the full count.
pub fn check(config: &WaveNetConfig) -> Result<(), String> {
    if config.head.is_some() {
        return Err("bloop does not slim WaveNets with a post-stack head".into());
    }
    for (i, a) in config.arrays.iter().enumerate() {
        if a.slimmable.is_empty() {
            continue;
        }
        if a.slimmable.windows(2).any(|w| w[1] <= w[0]) || a.slimmable.contains(&0) {
            return Err(format!(
                "layer array {i}'s allowed channels do not ascend"
            ));
        }
        if a.slimmable.last() != Some(&a.channels) {
            return Err(format!(
                "layer array {i}'s allowed channels do not end at its {} channels",
                a.channels
            ));
        }
    }
    for (i, a) in config.arrays.iter().enumerate() {
        if a.head_kernel_size != 1 {
            return Err(format!(
                "layer array {i} has a head kernel of {}; bloop slims only 1x1 heads",
                a.head_kernel_size
            ));
        }
        let grouped = a.groups_input != 1
            || a.groups_input_mixin != 1
            || a.layer1x1.is_some_and(|g| g != 1)
            || a.head1x1.is_some_and(|(_, g)| g != 1);
        if grouped {
            return Err(format!(
                "layer array {i} has grouped convolutions, which bloop does not slim"
            ));
        }
    }
    Ok(())
}

/// A 1x1's weights for the first `slim_out` outputs of the first
/// `slim_in` inputs.
#[expect(
    clippy::too_many_arguments,
    reason = "the core's extract_conv1x1, argument for argument"
)]
fn conv1x1(
    src: &mut Weights,
    full_in: usize,
    full_out: usize,
    slim_in: usize,
    slim_out: usize,
    bias: bool,
    dst: &mut Vec<f32>,
) -> Result<(), String> {
    for i in 0..full_out {
        for j in 0..full_in {
            let w = src.next()?;
            if i < slim_out && j < slim_in {
                dst.push(w);
            }
        }
    }
    if bias {
        for i in 0..full_out {
            let b = src.next()?;
            if i < slim_out {
                dst.push(b);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the core's extract_conv1d, argument for argument"
)]
fn conv1d(
    src: &mut Weights,
    full_in: usize,
    full_out: usize,
    slim_in: usize,
    slim_out: usize,
    kernel: usize,
    dst: &mut Vec<f32>,
) -> Result<(), String> {
    for i in 0..full_out {
        for j in 0..full_in {
            for _ in 0..kernel {
                let w = src.next()?;
                if i < slim_out && j < slim_in {
                    dst.push(w);
                }
            }
        }
    }
    for i in 0..full_out {
        let b = src.next()?;
        if i < slim_out {
            dst.push(b);
        }
    }
    Ok(())
}

fn copy(src: &mut Weights, n: usize, dst: &mut Vec<f32>) -> Result<(), String> {
    for _ in 0..n {
        dst.push(src.next()?);
    }
    Ok(())
}

/// The config and weights of `config` slimmed to `channels` per array
/// (`extract_slimmed_weights`, `modify_params_for_channels`).
///
/// # Errors
/// Fewer weights than the full model needs.
pub fn slim(
    config: &WaveNetConfig,
    weights: &[f32],
    channels: &[usize],
) -> Result<(WaveNetConfig, Vec<f32>), String> {
    let mut src = Weights::new(weights);
    let mut dst = Vec::new();
    let arrays = &config.arrays;
    let n = arrays.len();
    let mut out = config.clone();
    for (arr, p) in arrays.iter().enumerate() {
        let full_ch = p.channels;
        let full_bn = p.bottleneck;
        let slim_ch = channels[arr];
        let slim_bn = slim_bottleneck(p, slim_ch);
        let slim_input = if arr == 0 {
            p.input_size
        } else {
            channels[arr - 1]
        };
        let slim_head = if arr + 1 < n {
            channels[arr + 1]
        } else {
            p.head_size
        };
        let full_head_out = p.head1x1.map_or(full_bn, |(o, _)| o);
        let slim_head_out = p.head1x1.map_or(slim_bn, |(o, _)| o);
        conv1x1(&mut src, p.input_size, full_ch, slim_input, slim_ch, false, &mut dst)?;
        let cond = p.condition_size;
        for (l, &kernel) in p.kernel_sizes.iter().enumerate() {
            let zg = p.gatings[l].z_channels(1);
            let (full_bg, slim_bg) = (zg * full_bn, zg * slim_bn);
            conv1d(&mut src, full_ch, full_bg, slim_ch, slim_bg, kernel, &mut dst)?;
            conv1x1(&mut src, cond, full_bg, cond, slim_bg, false, &mut dst)?;
            if p.layer1x1.is_some() {
                conv1x1(&mut src, full_bn, full_ch, slim_bn, slim_ch, true, &mut dst)?;
            }
            if let Some((out_ch, _)) = p.head1x1 {
                conv1x1(&mut src, full_bn, out_ch, slim_bn, out_ch, true, &mut dst)?;
            }
            let f = &p.films;
            let film = |c: Option<crate::config::FilmConfig>, full: usize, slim: usize| {
                c.map(|c| {
                    let k = if c.shift { 2 } else { 1 };
                    (k * full, k * slim)
                })
            };
            for (full, slim) in [
                film(f.conv_pre, full_ch, slim_ch),
                film(f.conv_post, full_bg, slim_bg),
            ]
            .into_iter()
            .flatten()
            {
                conv1x1(&mut src, cond, full, cond, slim, true, &mut dst)?;
            }
            if let Some(c) = f.input_mixin_pre {
                let dim = if c.shift { 2 * cond } else { cond };
                copy(&mut src, cond * dim + dim, &mut dst)?;
            }
            for (full, slim) in [
                film(f.input_mixin_post, full_bg, slim_bg),
                film(f.activation_pre, full_bg, slim_bg),
                film(f.activation_post, full_bn, slim_bn),
                film(f.layer1x1_post.filter(|_| p.layer1x1.is_some()), full_ch, slim_ch),
            ]
            .into_iter()
            .flatten()
            {
                conv1x1(&mut src, cond, full, cond, slim, true, &mut dst)?;
            }
            if let (Some(c), Some((out_ch, _))) = (f.head1x1_post, p.head1x1) {
                let dim = if c.shift { 2 * out_ch } else { out_ch };
                copy(&mut src, cond * dim + dim, &mut dst)?;
            }
        }
        conv1x1(
            &mut src,
            full_head_out,
            p.head_size,
            slim_head_out,
            slim_head,
            p.head_bias,
            &mut dst,
        )?;
        let o = &mut out.arrays[arr];
        o.input_size = slim_input;
        o.head_size = slim_head;
        o.channels = slim_ch;
        o.bottleneck = slim_bn;
    }
    dst.push(src.next()?);
    Ok((out, dst))
}
