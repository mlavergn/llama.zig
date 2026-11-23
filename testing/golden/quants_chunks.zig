//! Golden checks for `src/ggml/quants/chunks.zig`: its kernels against the bits the C
//! reference computes, captured by the golden generators.
//!
//! **Not a port.** Lifted out of the ported file so the release tree
//! carries no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

/// The exported chunk symbols are generated in `comptime` blocks, so they have
/// no Zig-visible names. Reaching them as the C does is also the stronger
/// test: a mis-wired `@export` name shows up here rather than at the swap.
const ChunkFn = *const fn ([*c]const f32, ?*anyopaque, i64, i64, [*c]const f32) callconv(.c) usize;

/// The chunk entry point exported as `name`, found by that name.
fn exported(comptime name: []const u8) ChunkFn {
    return @extern(ChunkFn, .{ .name = name });
}

test "a chunk entry point reports the byte count its type implies" {
    // Independent of the checksums: a wrong return value would silently
    // truncate or overrun whatever the caller writes next.
    var src: [mod.quants_fixture.n_elem]f32 = undefined;
    mod.quants_fixture.fillSrc(.random, &src);
    var buf: [mod.quants_fixture.n_elem * 4]u8 align(16) = undefined;

    inline for (.{
        .{ "Q4_0", exported("quantize_q4_0") },
        .{ "Q6_K", exported("quantize_q6_K") },
        .{ "TQ2_0", exported("quantize_tq2_0") },
    }) |case| {
        const g = mod.quants_fixture.find(case[0], .random);
        const n = case[1](&src, &buf, mod.quants_golden.n_rows, mod.quants_golden.n_per_row, null);
        try std.testing.expectEqual(g.row_size * mod.quants_golden.n_rows, n);
    }
}

/// Checks one format's chunk entry point, with and without an importance
/// matrix, on every input pattern.
fn checkChunk(comptime name: []const u8, chunk: ChunkFn) !void {
    for (mod.quants_fixture.all_patterns) |pattern| {
        const g = mod.quants_fixture.find(name, pattern);

        var src: [mod.quants_fixture.n_elem]f32 = undefined;
        var imatrix: [mod.quants_golden.n_per_row]f32 = undefined;
        mod.quants_fixture.fillSrc(pattern, &src);
        mod.quants_fixture.fillImatrix(&imatrix);

        var buf: [mod.quants_fixture.n_elem * 4]u8 align(16) = undefined;

        if (g.chunk) |want| {
            @memset(&buf, 0);
            const n = chunk(&src, &buf, mod.quants_golden.n_rows, mod.quants_golden.n_per_row, null);
            std.testing.expectEqual(want, std.hash.Fnv1a_64.hash(buf[0..n])) catch |e| {
                std.debug.print("{s}: quantize_* differs on '{s}'\n", .{ name, pattern.name() });
                return e;
            };
        }

        if (g.chunk_imatrix) |want| {
            @memset(&buf, 0);
            const n = chunk(&src, &buf, mod.quants_golden.n_rows, mod.quants_golden.n_per_row, &imatrix);
            std.testing.expectEqual(want, std.hash.Fnv1a_64.hash(buf[0..n])) catch |e| {
                std.debug.print("{s}: quantize_* with imatrix differs on '{s}'\n", .{ name, pattern.name() });
                return e;
            };
        }
    }
}

test "the legacy chunk entry points match the C" {
    try checkChunk("Q4_0", exported("quantize_q4_0"));
    try checkChunk("Q4_1", exported("quantize_q4_1"));
    try checkChunk("Q5_0", exported("quantize_q5_0"));
    try checkChunk("Q5_1", exported("quantize_q5_1"));
    try checkChunk("Q8_0", exported("quantize_q8_0"));
    try checkChunk("MXFP4", exported("quantize_mxfp4"));
    try checkChunk("NVFP4", exported("quantize_nvfp4"));
}

test "the ternary chunk entry points match the C" {
    try checkChunk("TQ1_0", exported("quantize_tq1_0"));
    try checkChunk("TQ2_0", exported("quantize_tq2_0"));
}

test "the K-quant chunk entry points match the C" {
    try checkChunk("Q2_K", exported("quantize_q2_K"));
    try checkChunk("Q3_K", exported("quantize_q3_K"));
    try checkChunk("Q4_K", exported("quantize_q4_K"));
    try checkChunk("Q5_K", exported("quantize_q5_K"));
    try checkChunk("Q6_K", exported("quantize_q6_K"));
}
