//! Golden checks for `src/ggml/quants/iq4.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

/// Checks a format that has both a reference quantizer and a chunk entry
/// point, against every input pattern.
///
/// The chunk path is checked with *and* without an importance matrix, because
/// for these formats the two take different branches -- `quant_weights`
/// changes the error weighting, and `ntry` differs between the ref and chunk
/// entry points besides.
fn checkIq4(
    comptime name: []const u8,
    comptime Block: type,
    ref: *const fn ([*c]const f32, [*c]Block, i64) callconv(.c) void,
    chunk: *const fn ([*c]const f32, ?*anyopaque, i64, i64, [*c]const f32) callconv(.c) usize,
    dequant: *const fn ([*c]const Block, [*c]f32, i64) callconv(.c) void,
) !void {
    for (mod.quants_fixture.all_patterns) |pattern| {
        // `quantize_row_iq4_nl_ref` reads uninitialized memory for an all-zero
        // block -- see the note in the function. The golden recorded one
        // sample of it, and the C does not even agree with itself across two
        // calls, so there is nothing here to match.
        if (std.mem.eql(u8, name, "IQ4_NL") and pattern == .zeros) continue;

        const g = mod.quants_fixture.find(name, pattern);

        var src: [mod.quants_fixture.n_elem]f32 = undefined;
        var imatrix: [mod.quants_golden.n_per_row]f32 = undefined;
        mod.quants_fixture.fillSrc(pattern, &src);
        mod.quants_fixture.fillImatrix(&imatrix);

        var buf: [mod.quants_fixture.n_elem * 4]u8 align(16) = undefined;

        @memset(&buf, 0);
        ref(&src, @ptrCast(@alignCast(&buf)), @intCast(mod.quants_fixture.n_elem));
        const used = g.row_size * mod.quants_golden.n_rows;
        std.testing.expectEqual(g.ref.?, std.hash.Fnv1a_64.hash(buf[0..used])) catch |e| {
            std.debug.print("{s}: quantize_row_*_ref differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        var out: [mod.quants_fixture.n_elem]f32 = undefined;
        @memset(&out, 0);
        dequant(@ptrCast(@alignCast(&buf)), &out, @intCast(mod.quants_fixture.n_elem));
        std.testing.expectEqual(g.deq.?, std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(out[0..]))) catch |e| {
            std.debug.print("{s}: dequantize differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        @memset(&buf, 0);
        const n0 = chunk(&src, &buf, mod.quants_golden.n_rows, mod.quants_golden.n_per_row, null);
        std.testing.expectEqual(g.chunk.?, std.hash.Fnv1a_64.hash(buf[0..n0])) catch |e| {
            std.debug.print("{s}: quantize_* differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };

        @memset(&buf, 0);
        const n1 = chunk(&src, &buf, mod.quants_golden.n_rows, mod.quants_golden.n_per_row, &imatrix);
        std.testing.expectEqual(g.chunk_imatrix.?, std.hash.Fnv1a_64.hash(buf[0..n1])) catch |e| {
            std.debug.print("{s}: quantize_* with imatrix differs on '{s}'\n", .{ name, pattern.name() });
            return e;
        };
    }
}

test "iq4_nl matches the C on every pattern, ref and chunk" {
    try checkIq4("IQ4_NL", ggml.quants.blocks.IQ4_NL, ggml.quants.iq4.quantize_row_iq4_nl_ref, ggml.quants.iq4.quantize_iq4_nl, ggml.quants.iq_dequant.dequantize_row_iq4_nl);
}

test "iq4_xs matches the C on every pattern, ref and chunk" {
    try checkIq4("IQ4_XS", ggml.quants.blocks.IQ4_XS, ggml.quants.iq4.quantize_row_iq4_xs_ref, ggml.quants.iq4.quantize_iq4_xs, ggml.quants.iq_dequant.dequantize_row_iq4_xs);
}
