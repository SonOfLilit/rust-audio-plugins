// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Ported from Bungee's src/Phase.h and src/Stretch.{h,cpp} (Copyright (C)
// 2020-2026 Parabola Research Limited, SPDX-License-Identifier: MPL-2.0).

//! Phases as 16-bit integers, a whole turn being 65536, so that arithmetic
//! on them wraps round the circle by itself. Bungee relies on signed
//! overflow wrapping (it builds with `-fwrapv`); here every operation that
//! can wrap says so.

/// An angle in [-pi, pi), a turn being 65536.
pub(crate) type Phase = i16;

/// `radians` as a [`Phase`], truncated toward zero. Pi itself wraps to
/// -pi, as the C++ conversion does through a 32-bit integer.
pub(crate) fn from_radians(radians: f32) -> Phase {
    const K: f32 = (65536.0 / (2.0 * core::f64::consts::PI)) as f32;
    (K * radians) as i32 as i16
}

/// How far each bin's phase turns relative to its neighbour when the grain
/// is stretched in frequency by `speed` (`Stretch::Frequency`).
pub(crate) struct Frequency {
    multiplier: i32,
}

impl Frequency {
    const SHIFT: u32 = 8;

    pub(crate) fn new(speed: f32) -> Self {
        let speed = speed.abs() + 1e-20;
        let multiplier = (((1 << Self::SHIFT) as f32 / -speed).round()).max(f32::from(i16::MIN));
        Self {
            multiplier: multiplier as i32,
        }
    }

    pub(crate) fn apply(&self, n: usize, rotation: &mut [Phase], phase: &[Phase]) {
        rotation[0] = 0;
        for m in 1..n {
            let delta = phase[m - 1].wrapping_sub(phase[m]);
            let x = (i32::from(delta).wrapping_mul(self.multiplier)) >> Self::SHIFT;
            rotation[m] = rotation[m - 1].wrapping_add(x as i16).wrapping_add(delta);
        }
    }
}

/// How far a peak's phase has to turn for the grain to follow on from the
/// previous one, given the hop the input moved by (`Stretch::Time`).
pub(crate) struct Time {
    reverse: bool,
    reverse_previous: bool,
    a: i32,
    multiplier: i32,
}

impl Time {
    const LOG2_SYNTHESIS_HOP_REVOLUTION: i32 = -3;

    pub(crate) fn new(
        log2_synthesis_hop: u32,
        analysis_hop: i32,
        analysis_hop_previous: i32,
    ) -> Self {
        let log2_transform_length = log2_synthesis_hop as i32 - Self::LOG2_SYNTHESIS_HOP_REVOLUTION;
        let a = analysis_hop.wrapping_shl((32 - log2_transform_length) as u32);
        let dividend = (1_i32 << log2_synthesis_hop) << 16;
        let divisor = analysis_hop.wrapping_shl(1);
        let multiplier = if divisor == 0 {
            0
        } else {
            (dividend + divisor.abs() / 2) / divisor
        };
        Self {
            reverse: analysis_hop < 0,
            reverse_previous: analysis_hop_previous < 0,
            a,
            multiplier,
        }
    }

    pub(crate) fn offset(&self, phase: Phase, previous: Phase) -> Phase {
        let phase = if self.reverse {
            phase.wrapping_neg()
        } else {
            phase
        };
        let previous = if self.reverse_previous {
            previous.wrapping_neg()
        } else {
            previous
        };
        phase.wrapping_sub(previous)
    }

    pub(crate) fn delta(&self, phase: Phase, previous: Phase, m: i32) -> Phase {
        const LOG_S: u32 = (32 + Time::LOG2_SYNTHESIS_HOP_REVOLUTION) as u32;
        let (phase, previous) = (i32::from(phase) << 16, i32::from(previous) << 16);
        let da = phase
            .wrapping_sub(previous)
            .wrapping_sub(m.wrapping_mul(self.a));
        let wide = m
            .wrapping_shl(LOG_S)
            .wrapping_add((da >> 15).wrapping_mul(self.multiplier));
        (wide >> 16) as i16
    }
}
