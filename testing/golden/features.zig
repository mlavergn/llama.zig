//! Which goldens this CPU compares against.
//!
//! **Not a port.**

const std = @import("std");
const builtin = @import("builtin");

/// Whether the unit tests compare against the goldens captured at `generic`
/// rather than at the baseline: `cpu_quants_arm_golden_generic.zig` and
/// `cpu_vec_golden_generic.zig`, written by `GOLDEN_CPU=generic
/// testing/scripts/vecdot-golden` and `testing/scripts/vec-golden`. The reference computes
/// different bits there -- the `F16` arm, `ggml_vdotq_s32`'s grouping, the
/// vectorizer's width -- so it is a different oracle, not a looser one.
/// What decides is those three, not the model's name: `apple_a7` to
/// `apple_a11` have neither `fullfp16` nor `dotprod` and the vectorizer's
/// narrower interleave (`simd.strict_loop`), so they compute `generic`'s
/// bits; the A14 and later share the baseline's. The A12 and A13, with
/// `fullfp16` and without the rest, match neither set.
pub const golden_generic = builtin.cpu.arch == .aarch64 and
    !std.Target.aarch64.featureSetHas(builtin.cpu.features, .fullfp16) and
    !std.Target.aarch64.featureSetHas(builtin.cpu.features, .dotprod) and
    !std.Target.aarch64.featureSetHas(builtin.cpu.features, .max_interleave_factor_4);
