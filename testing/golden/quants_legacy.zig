//! Golden checks for `src/ggml/quants/legacy.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

/// Runs one format's reference quantizer and dequantizer against its goldens,
/// for every input pattern.
///
/// The patterns matter more than the count of them: `zeros` is the only case
/// that reaches the `id = d ? 1/d : 0` guard every symmetric quantizer opens
/// with, and `constant` the only one where an asymmetric format sees
/// `min == max`. Random input alone never takes either branch, and deleting
/// the guard from `quantize_row_q4_0_ref` is caught by `zeros` alone.
///
/// **`constant` cannot prove the same for the asymmetric formats**, and that
/// is worth knowing rather than assuming. With `min == max` the scale is zero,
/// so a missing guard computes `(x - min) * inf`, which is `0 * inf` = NaN;
/// `impl.truncTo` maps NaN to 0, and the correct `id = 0` path also yields 0.
/// The two coincide for every input, so no golden can separate them. The guard
/// is still right to have -- it is what keeps the value defined rather than
/// relying on that coincidence -- but this test does not defend it.
///
/// Parameters:
/// - `name`: the type's name in `golden.zig`.
/// - `Block`: the C block struct.
/// - `quant`: the ported `quantize_row_*_ref`.
/// - `dequant`: the ported `dequantize_row_*`, or null when the C has none.
fn checkFormat(
    comptime name: []const u8,
    comptime Block: type,
    quant: *const fn ([*c]const f32, [*c]Block, i64) callconv(.c) void,
    comptime dequant: ?*const fn ([*c]const Block, [*c]f32, i64) callconv(.c) void,
) !void {
    for (mod.quants_fixture.all_patterns) |pattern| {
        const g = mod.quants_fixture.find(name, pattern);

        var src: [mod.quants_fixture.n_elem]f32 = undefined;
        mod.quants_fixture.fillSrc(pattern, &src);

        // Generous: the widest format here is under 2 bytes per weight.
        var buf: [mod.quants_fixture.n_elem * 4]u8 align(16) = undefined;
        @memset(&buf, 0);

        quant(&src, @ptrCast(@alignCast(&buf)), @intCast(mod.quants_fixture.n_elem));

        // Row size comes from the golden record rather than the traits table,
        // so this test links without `ggml-quants.c` and can therefore run
        // before the whole translation unit is ported.
        const used = g.row_size * mod.quants_golden.n_rows;
        std.testing.expectEqual(g.ref.?, std.hash.Fnv1a_64.hash(buf[0..used])) catch |e| {
            std.debug.print("{s}: quantize differs on pattern '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        if (dequant) |dq| {
            var out: [mod.quants_fixture.n_elem]f32 = undefined;
            @memset(&out, 0);
            dq(@ptrCast(@alignCast(&buf)), &out, @intCast(mod.quants_fixture.n_elem));
            std.testing.expectEqual(g.deq.?, std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(out[0..]))) catch |e| {
                std.debug.print("{s}: dequantize differs on pattern '{s}'\n", .{ name, pattern.name() });
                return e;
            };
        }
    }
}

test "q1_0 matches the C, quantize and dequantize" {
    try checkFormat("Q1_0", ggml.quants.blocks.Q1_0, ggml.quants.legacy.quantize_row_q1_0_ref, ggml.quants.legacy.dequantize_row_q1_0);
}

test "q2_0 matches the C, quantize and dequantize" {
    try checkFormat("Q2_0", ggml.quants.blocks.Q2_0, ggml.quants.legacy.quantize_row_q2_0_ref, ggml.quants.legacy.dequantize_row_q2_0);
}

test "q4_0 matches the C, quantize and dequantize" {
    try checkFormat("Q4_0", ggml.quants.blocks.Q4_0, ggml.quants.legacy.quantize_row_q4_0_ref, ggml.quants.legacy.dequantize_row_q4_0);
}

test "q4_1 matches the C, quantize and dequantize" {
    try checkFormat("Q4_1", ggml.quants.blocks.Q4_1, ggml.quants.legacy.quantize_row_q4_1_ref, ggml.quants.legacy.dequantize_row_q4_1);
}

test "q5_0 matches the C, quantize and dequantize" {
    try checkFormat("Q5_0", ggml.quants.blocks.Q5_0, ggml.quants.legacy.quantize_row_q5_0_ref, ggml.quants.legacy.dequantize_row_q5_0);
}

test "q5_1 matches the C, quantize and dequantize" {
    try checkFormat("Q5_1", ggml.quants.blocks.Q5_1, ggml.quants.legacy.quantize_row_q5_1_ref, ggml.quants.legacy.dequantize_row_q5_1);
}

test "q8_0 matches the C, quantize and dequantize" {
    try checkFormat("Q8_0", ggml.quants.blocks.Q8_0, ggml.quants.legacy.quantize_row_q8_0_ref, ggml.quants.legacy.dequantize_row_q8_0);
}

test "q8_1 matches the C -- quantize only, the format has no dequantizer" {
    try checkFormat("Q8_1", ggml.quants.blocks.Q8_1, ggml.quants.legacy.quantize_row_q8_1_ref, null);
}

test "mxfp4 matches the C, quantize and dequantize" {
    try checkFormat("MXFP4", ggml.quants.blocks.MXFP4, ggml.quants.legacy.quantize_row_mxfp4_ref, ggml.quants.legacy.dequantize_row_mxfp4);
}

test "nvfp4 matches the C, quantize and dequantize" {
    try checkFormat("NVFP4", ggml.quants.blocks.NVFP4, ggml.quants.legacy.quantize_row_nvfp4_ref, ggml.quants.legacy.dequantize_row_nvfp4);
}
