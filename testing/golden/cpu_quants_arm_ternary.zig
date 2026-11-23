//! Golden checks for `src/ggml/cpu/quants/arm/ternary.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

const golden = if (mod.features.golden_generic) mod.cpu_quants_arm_golden_generic else mod.cpu_quants_arm_golden;

test "tq1_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.ternary.ggml_vec_dot_tq1_0_q8_K, ggml.impl.ggml_c.GGML_TYPE_TQ1_0, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.tq1_0);
}

test "tq2_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.ternary.ggml_vec_dot_tq2_0_q8_K, ggml.impl.ggml_c.GGML_TYPE_TQ2_0, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.tq2_0);
}
