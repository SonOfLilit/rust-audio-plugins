// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Ported from Bungee's src/Partials.{h,cpp} (Copyright (C) 2020-2026
// Parabola Research Limited, SPDX-License-Identifier: MPL-2.0).

//! A grain's spectrum split into partials: each a peak bin and the run of
//! bins around it that falls away from it, which then turn together so the
//! peak's phase carries its neighbours (identity phase locking).

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Partial {
    pub peak: i16,
    pub end: i16,
}

/// Split bins `0..n` of `energy` into partials. `energy` has room for two
/// more bins past `n`, which this borrows as sentinels and puts back.
pub(crate) fn enumerate(partials: &mut [Partial], count: &mut usize, n: usize, energy: &mut [f32]) {
    let undo = [energy[n], energy[n + 1]];
    energy[n] = -1.0;
    energy[n + 1] = 0.0;

    *count = 0;
    let mut m = 1;
    loop {
        while energy[m] < energy[m + 1] {
            m += 1;
        }
        partials[*count].peak = m as i16;
        m += 1;

        #[expect(
            clippy::neg_cmp_op_on_partial_ord,
            reason = "Bungee's own test: a run ends where the energy rises"
        )]
        while !(energy[m] < energy[m + 1]) {
            m += 1;
        }
        partials[*count].end = m as i16;
        *count += 1;
        m += 1;

        if m > n {
            break;
        }
    }

    energy[n] = undo[0];
    energy[n + 1] = undo[1];
}

fn suppress_partial(partials: &mut [Partial], i: usize, energy: &[f32]) {
    if energy[partials[i - 1].end as usize] > energy[partials[i].end as usize] {
        partials[i - 1].end = partials[i].end;
    } else {
        partials[i].end = partials[i - 1].end;
    }
}

/// Merge every partial but the strongest whose peak jumped by half again
/// since the previous grain into a neighbour: a transient's bins then stop
/// being locked to peaks of their own.
pub(crate) fn suppress_transient_partials(
    partials: &mut [Partial],
    energy: &[f32],
    previous_energy: &[f32],
) {
    // Bungee: "fudge: lower constant helps transients, higher helps tones".
    const K: f32 = 1.5;

    let mut strongest = 0;
    for i in 1..partials.len() {
        if energy[partials[i].peak as usize] > energy[partials[strongest].peak as usize] {
            strongest = i;
        }
    }

    for i in 1..partials.len().saturating_sub(1) {
        if i != strongest {
            let peak = partials[i].peak as usize;
            if energy[peak] > K * previous_energy[peak] {
                suppress_partial(partials, i, energy);
            }
        }
    }
}
