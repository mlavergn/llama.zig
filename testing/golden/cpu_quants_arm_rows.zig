//! Golden checks for `src/ggml/cpu/quants/arm/rows.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

const golden = if (mod.features.golden_generic) mod.cpu_quants_arm_golden_generic else mod.cpu_quants_arm_golden;

test "q8_0 row quantizer matches the C" {
    try mod.cpu_quants_fixture.checkRow(ggml.cpu.quants.arch.rows.quantize_row_q8_0, ggml.impl.ggml_c.GGML_TYPE_Q8_0, golden.row_q8_0);
}

test "q8_1 row quantizer matches the C" {
    try mod.cpu_quants_fixture.checkRow(ggml.cpu.quants.arch.rows.quantize_row_q8_1, ggml.impl.ggml_c.GGML_TYPE_Q8_1, golden.row_q8_1);
}

test "q8_K row quantizer matches the C" {
    try mod.cpu_quants_fixture.checkRow(ggml.cpu.quants.arch.rows.quantize_row_q8_K, ggml.impl.ggml_c.GGML_TYPE_Q8_K, golden.row_q8_K);
}
