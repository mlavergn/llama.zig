//! Golden checks for `src/ggml/cpu/quants/arm/k.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

const golden = if (mod.features.golden_generic) mod.cpu_quants_arm_golden_generic else mod.cpu_quants_arm_golden;

test "q2_K dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.k.ggml_vec_dot_q2_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q2_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.q2_K);
}

test "q3_K dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.k.ggml_vec_dot_q3_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q3_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.q3_K);
}

test "q4_K dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.k.ggml_vec_dot_q4_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q4_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.q4_K);
}

test "q5_K dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.k.ggml_vec_dot_q5_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q5_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.q5_K);
}

test "q6_K dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.k.ggml_vec_dot_q6_K_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q6_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.q6_K);
}
