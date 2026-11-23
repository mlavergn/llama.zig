//! Golden checks for `src/ggml/cpu/quants/legacy.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

test "q1_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_q1_0_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_Q1_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.q1_0);
}

test "q2_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_q2_0_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_Q2_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.q2_0);
}

test "q4_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_q4_0_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_Q4_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.q4_0);
}

test "q4_1 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_q4_1_q8_1_generic, ggml.impl.ggml_c.GGML_TYPE_Q4_1, ggml.impl.ggml_c.GGML_TYPE_Q8_1, mod.cpu_quants_golden.q4_1);
}

test "q5_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_q5_0_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_Q5_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.q5_0);
}

test "q5_1 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_q5_1_q8_1_generic, ggml.impl.ggml_c.GGML_TYPE_Q5_1, ggml.impl.ggml_c.GGML_TYPE_Q8_1, mod.cpu_quants_golden.q5_1);
}

test "q8_0 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_q8_0_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_Q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.q8_0);
}

test "mxfp4 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_mxfp4_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_MXFP4, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.mxfp4);
}

test "nvfp4 dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_nvfp4_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_NVFP4, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.nvfp4);
}

test "iq4_nl dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.legacy.ggml_vec_dot_iq4_nl_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_IQ4_NL, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.iq4_nl);
}
