//! Golden checks for `src/ggml/cpu/quants/arm/legacy.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

const golden = if (mod.features.golden_generic) mod.cpu_quants_arm_golden_generic else mod.cpu_quants_arm_golden;

test "q1_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_q1_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q1_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.q1_0);
}

test "q2_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_q2_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q2_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.q2_0);
}

test "q4_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_q4_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q4_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.q4_0);
}

test "q4_1 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_q4_1_q8_1, ggml.impl.ggml_c.GGML_TYPE_Q4_1, ggml.impl.ggml_c.GGML_TYPE_Q8_1, golden.q4_1);
}

test "mxfp4 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_mxfp4_q8_0, ggml.impl.ggml_c.GGML_TYPE_MXFP4, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.mxfp4);
}

test "nvfp4 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_nvfp4_q8_0, ggml.impl.ggml_c.GGML_TYPE_NVFP4, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.nvfp4);
}

test "q5_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_q5_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q5_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.q5_0);
}

test "q5_1 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_q5_1_q8_1, ggml.impl.ggml_c.GGML_TYPE_Q5_1, ggml.impl.ggml_c.GGML_TYPE_Q8_1, golden.q5_1);
}

test "q8_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_q8_0_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.q8_0);
}

test "iq4_nl dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.legacy.ggml_vec_dot_iq4_nl_q8_0, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.iq4_nl);
}
