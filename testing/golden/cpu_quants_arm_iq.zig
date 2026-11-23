//! Golden checks for `src/ggml/cpu/quants/arm/iq.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

const golden = if (mod.features.golden_generic) mod.cpu_quants_arm_golden_generic else mod.cpu_quants_arm_golden;

test "iq2_xxs dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq2_xxs_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq2_xxs);
}

test "iq2_xs dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq2_xs_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ2_XS, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq2_xs);
}

test "iq2_s dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq2_s_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ2_S, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq2_s);
}

test "iq3_xxs dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq3_xxs_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq3_xxs);
}

test "iq3_s dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq3_s_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ3_S, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq3_s);
}

test "iq1_s dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq1_s_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ1_S, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq1_s);
}

test "iq1_m dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq1_m_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ1_M, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq1_m);
}

test "iq4_xs dot matches the C" {
    try mod.cpu_quants_fixture.check(ggml.cpu.quants.arch.iq.ggml_vec_dot_iq4_xs_q8_K, ggml.impl.ggml_c.GGML_TYPE_IQ4_XS, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.iq4_xs);
}
