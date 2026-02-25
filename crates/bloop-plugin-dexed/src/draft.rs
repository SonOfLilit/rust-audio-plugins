//! bloop-plugin-dexed: 6-operator FM synth.
//! Rust port of MSFA (Apache 2.0). Lives in a separate public repo.
//!
//! Not part of the build yet — this is a design draft.

pub struct DexedPlugin {
    // 6 operators, each with:
    //   - frequency ratio
    //   - output level
    //   - 8-stage envelope (Rate1-4, Level1-4 — NOT standard ADSR)
    // 32 algorithms (operator routing from DX7 ROM)
    // Feedback on self-modulating operator
    // Voice allocation (8-16 voices)
    // DX7 SysEx patch loading
}

// impl Plugin for DexedPlugin { ... }

// --- Test harness ---

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::{expect, Expect};

    fn check_fm(algorithm: u8, operators: &str, note: u8, frames: usize, expect: Expect) {
        // Set algorithm + operator params, render, compute spectrum
        // Snapshot strongest harmonics
        let _ = (algorithm, operators, note, frames);
        expect.assert_eq("TODO");
    }

    #[test]
    fn simple_sine() {
        // Algorithm 32 (single carrier, no modulation) → pure sine
        check_fm(32, "op1:ratio=1.0,level=1.0", 69, 4096, expect![[r#"TODO"#]]);
    }

    #[test]
    fn classic_epiano() {
        check_fm(5, "op1:ratio=1.0,level=0.8 op2:ratio=14.0,level=0.3", 60, 4096, expect![[r#"TODO"#]]);
    }
}
