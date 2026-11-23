//! Golden checks for `src/ggml/cpu/quants/rows.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

test "the generic q8 row quantizers match the C" {
    // These three are the `_generic` fallbacks, unreachable on this target
    // like the generic dot products, so the goldens are their only gate.
    try mod.cpu_quants_fixture.checkRow(ggml.cpu.quants.rows.quantize_row_q8_0_generic, ggml.impl.ggml_c.GGML_TYPE_Q8_0, mod.cpu_quants_golden.row_q8_0);
    try mod.cpu_quants_fixture.checkRow(ggml.cpu.quants.rows.quantize_row_q8_1_generic, ggml.impl.ggml_c.GGML_TYPE_Q8_1, mod.cpu_quants_golden.row_q8_1);
    try mod.cpu_quants_fixture.checkRow(ggml.cpu.quants.rows.quantize_row_q8_K_generic, ggml.impl.ggml_c.GGML_TYPE_Q8_K, mod.cpu_quants_golden.row_q8_K);
}
