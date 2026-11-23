//! Golden checks for `src/ggml/quants/k.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

/// Same shape as `legacy.zig`'s, checking every format against every pattern.
fn checkK(
    comptime name: []const u8,
    comptime Block: type,
    quant: *const fn ([*c]const f32, [*c]Block, i64) callconv(.c) void,
    dequant: *const fn ([*c]const Block, [*c]f32, i64) callconv(.c) void,
) !void {
    for (mod.quants_fixture.all_patterns) |pattern| {
        const g = mod.quants_fixture.find(name, pattern);

        var src: [mod.quants_fixture.n_elem]f32 = undefined;
        mod.quants_fixture.fillSrc(pattern, &src);

        var buf: [mod.quants_fixture.n_elem * 4]u8 align(16) = undefined;
        @memset(&buf, 0);

        quant(&src, @ptrCast(@alignCast(&buf)), @intCast(mod.quants_fixture.n_elem));

        const used = g.row_size * mod.quants_golden.n_rows;
        std.testing.expectEqual(g.ref.?, std.hash.Fnv1a_64.hash(buf[0..used])) catch |e| {
            std.debug.print("{s}: quantize differs on pattern '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        var out: [mod.quants_fixture.n_elem]f32 = undefined;
        @memset(&out, 0);
        dequant(@ptrCast(@alignCast(&buf)), &out, @intCast(mod.quants_fixture.n_elem));
        std.testing.expectEqual(g.deq.?, std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(out[0..]))) catch |e| {
            std.debug.print("{s}: dequantize differs on pattern '{s}'\n", .{ name, pattern.name() });
            return e;
        };
    }
}

test "q2_K matches the C on every pattern" {
    try checkK("Q2_K", ggml.quants.blocks.Q2_K, ggml.quants.k.quantize_row_q2_K_ref, ggml.quants.k.dequantize_row_q2_K);
}

test "q3_K matches the C on every pattern" {
    try checkK("Q3_K", ggml.quants.blocks.Q3_K, ggml.quants.k.quantize_row_q3_K_ref, ggml.quants.k.dequantize_row_q3_K);
}

test "q4_K matches the C on every pattern" {
    try checkK("Q4_K", ggml.quants.blocks.Q4_K, ggml.quants.k.quantize_row_q4_K_ref, ggml.quants.k.dequantize_row_q4_K);
}

test "q5_K matches the C on every pattern" {
    try checkK("Q5_K", ggml.quants.blocks.Q5_K, ggml.quants.k.quantize_row_q5_K_ref, ggml.quants.k.dequantize_row_q5_K);
}

test "q6_K matches the C on every pattern" {
    try checkK("Q6_K", ggml.quants.blocks.Q6_K, ggml.quants.k.quantize_row_q6_K_ref, ggml.quants.k.dequantize_row_q6_K);
}

test "q8_K matches the C on every pattern" {
    try checkK("Q8_K", ggml.quants.blocks.Q8_K, ggml.quants.k.quantize_row_q8_K_ref, ggml.quants.k.dequantize_row_q8_K);
}
