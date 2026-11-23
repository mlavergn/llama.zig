//! Golden checks for `src/ggml/quants/ternary.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

fn checkTernary(
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

test "tq1_0 matches the C on every pattern" {
    try checkTernary("TQ1_0", ggml.quants.blocks.TQ1_0, ggml.quants.ternary.quantize_row_tq1_0_ref, ggml.quants.ternary.dequantize_row_tq1_0);
}

test "tq2_0 matches the C on every pattern" {
    try checkTernary("TQ2_0", ggml.quants.blocks.TQ2_0, ggml.quants.ternary.quantize_row_tq2_0_ref, ggml.quants.ternary.dequantize_row_tq2_0);
}
