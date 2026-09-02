//! Quantizing a row of floats into one of ggml's block formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml.c` (v0.3.0, `c1d0e7a00`), lines
//! 7911-8030. Each function names the C function it replaces and the line it
//! began at.
//!
//! # This file dispatches; it does not quantize
//!
//! The arithmetic lives in `ggml-quants.c`, which is still C and is Stage 3
//! step 3's problem. What is ported here is the entry point: validate, make
//! sure any lookup tables the format needs are built, pick the routine, and
//! check the result came out the expected size.
//!
//! # The i-quants need a table built first
//!
//! `IQ2_XXS` and its relatives index into a shared codebook that is computed
//! once and cached in a process-global. `ggml_quantize_init` builds it and
//! `ggml_quantize_free` releases it, both under ggml's critical section
//! because a caller may quantize from several threads. Every path into a
//! quantize call goes through `ggml_quantize_init` first -- it is idempotent,
//! so the cost of being wrong about whether it has run is nothing.

const std = @import("std");
const impl = @import("impl.zig");
const types = @import("types.zig");
const c = impl.c;

/// Every `quantize_*` routine in `ggml-quants.c` has this shape.
const QuantizeFn = *const fn (
    src: [*c]const f32,
    dst: ?*anyopaque,
    nrows: i64,
    n_per_row: i64,
    imatrix: [*c]const f32,
) callconv(.c) usize;

extern fn iq2xs_init_impl(t: c.enum_ggml_type) void;
extern fn iq2xs_free_impl(t: c.enum_ggml_type) void;
extern fn iq3xs_init_impl(grid_size: c_int) void;
extern fn iq3xs_free_impl(grid_size: c_int) void;

/// The `switch` in `ggml_quantize_chunk` (ggml.c:7957 @c1d0e7a00), as a table.
///
/// The C spells out one `case` per type, each an identical call differing only
/// in the routine's name. Listing the pairs instead keeps the mapping readable
/// and lets the lookup below be generated, which is the same treatment
/// `types.zig` gives the traits table.
///
/// `F16`, `BF16`, and `F32` are absent: they are not block formats and the C
/// handles them with three special cases, reproduced in `ggml_quantize_chunk`.
const quantizers = .{
    .{ c.GGML_TYPE_Q1_0, "quantize_q1_0" },
    .{ c.GGML_TYPE_Q2_0, "quantize_q2_0" },
    .{ c.GGML_TYPE_Q4_0, "quantize_q4_0" },
    .{ c.GGML_TYPE_Q4_1, "quantize_q4_1" },
    .{ c.GGML_TYPE_Q5_0, "quantize_q5_0" },
    .{ c.GGML_TYPE_Q5_1, "quantize_q5_1" },
    .{ c.GGML_TYPE_Q8_0, "quantize_q8_0" },
    .{ c.GGML_TYPE_MXFP4, "quantize_mxfp4" },
    .{ c.GGML_TYPE_NVFP4, "quantize_nvfp4" },
    .{ c.GGML_TYPE_Q2_K, "quantize_q2_K" },
    .{ c.GGML_TYPE_Q3_K, "quantize_q3_K" },
    .{ c.GGML_TYPE_Q4_K, "quantize_q4_K" },
    .{ c.GGML_TYPE_Q5_K, "quantize_q5_K" },
    .{ c.GGML_TYPE_Q6_K, "quantize_q6_K" },
    .{ c.GGML_TYPE_TQ1_0, "quantize_tq1_0" },
    .{ c.GGML_TYPE_TQ2_0, "quantize_tq2_0" },
    .{ c.GGML_TYPE_IQ2_XXS, "quantize_iq2_xxs" },
    .{ c.GGML_TYPE_IQ2_XS, "quantize_iq2_xs" },
    .{ c.GGML_TYPE_IQ3_XXS, "quantize_iq3_xxs" },
    .{ c.GGML_TYPE_IQ3_S, "quantize_iq3_s" },
    .{ c.GGML_TYPE_IQ2_S, "quantize_iq2_s" },
    .{ c.GGML_TYPE_IQ1_S, "quantize_iq1_s" },
    .{ c.GGML_TYPE_IQ1_M, "quantize_iq1_m" },
    .{ c.GGML_TYPE_IQ4_NL, "quantize_iq4_nl" },
    .{ c.GGML_TYPE_IQ4_XS, "quantize_iq4_xs" },
};

/// Return: the routine for `t`, or null when `t` is not a block format.
fn quantizerFor(t: c.enum_ggml_type) ?QuantizeFn {
    inline for (quantizers) |entry| {
        if (t == entry[0]) return @extern(QuantizeFn, .{ .name = entry[1] });
    }
    return null;
}

/// Ports `ggml_quantize_init` (ggml.c:7917 @c1d0e7a00).
///
/// Builds the shared codebook the i-quants index into. Idempotent, and a
/// no-op for every other format.
pub export fn ggml_quantize_init(t: c.enum_ggml_type) void {
    c.ggml_critical_section_start();

    switch (t) {
        c.GGML_TYPE_IQ2_XXS,
        c.GGML_TYPE_IQ2_XS,
        c.GGML_TYPE_IQ2_S,
        c.GGML_TYPE_IQ1_S,
        c.GGML_TYPE_IQ1_M,
        => iq2xs_init_impl(t),
        c.GGML_TYPE_IQ3_XXS => iq3xs_init_impl(256),
        c.GGML_TYPE_IQ3_S => iq3xs_init_impl(512),
        else => {}, // nothing
    }

    c.ggml_critical_section_end();
}

/// Ports `ggml_quantize_free` (ggml.c:7935 @c1d0e7a00).
///
/// Releases every codebook unconditionally rather than tracking which were
/// built; the free routines tolerate a table that was never allocated.
pub export fn ggml_quantize_free() void {
    c.ggml_critical_section_start();

    iq2xs_free_impl(c.GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(c.GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(c.GGML_TYPE_IQ2_S);
    iq2xs_free_impl(c.GGML_TYPE_IQ1_S);
    iq2xs_free_impl(c.GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);

    c.ggml_critical_section_end();
}

/// Ports `ggml_quantize_requires_imatrix` (ggml.c:7949 @c1d0e7a00).
///
/// Note `IQ1_M` is absent: the C lists it commented out, and leaving it out is
/// deliberate rather than an oversight, so it is left out here too.
pub export fn ggml_quantize_requires_imatrix(t: c.enum_ggml_type) bool {
    return t == c.GGML_TYPE_IQ2_XXS or
        t == c.GGML_TYPE_IQ2_XS or
        t == c.GGML_TYPE_IQ1_S;
}

/// Ports `ggml_quantize_chunk` (ggml.c:7957 @c1d0e7a00).
///
/// Quantizes `nrows` rows starting at element `start`, which must fall on both
/// a block boundary and a row boundary.
///
/// Parameters:
/// - `t`: destination format.
/// - `src`: source floats, indexed from `start`.
/// - `dst`: destination, written from row `start / n_per_row` onward.
/// - `start`: first element to convert.
/// - `nrows`, `n_per_row`: how many rows and how wide.
/// - `imatrix`: importance matrix, required by the formats
///   `ggml_quantize_requires_imatrix` names and ignored by the rest.
///
/// Return: bytes written, which the C asserts equals `nrows * row_size`.
pub export fn ggml_quantize_chunk(
    t: c.enum_ggml_type,
    src: [*c]const f32,
    dst: ?*anyopaque,
    start: i64,
    nrows: i64,
    n_per_row: i64,
    imatrix: [*c]const f32,
) usize {
    const n = nrows * n_per_row;

    if (ggml_quantize_requires_imatrix(t)) {
        impl.assert(imatrix != null, "imatrix != NULL");
    }

    impl.assert(@rem(start, types.ggml_blck_size(t)) == 0, "start % type_traits[type].blck_size == 0");
    impl.assert(@rem(start, n_per_row) == 0, "start % n_per_row == 0");

    ggml_quantize_init(t); // this is noop if already initialized

    const start_row: usize = @intCast(@divTrunc(start, n_per_row));
    const row_size = types.ggml_row_size(t, n_per_row);

    var result: usize = 0;

    if (quantizerFor(t)) |quantize| {
        const out = @as([*]u8, @ptrCast(dst.?)) + start_row * row_size;
        result = quantize(src + @as(usize, @intCast(start)), out, nrows, n_per_row, imatrix);
    } else switch (t) {
        // Not block formats: a straight elementwise narrowing, and note these
        // three index `dst` by element rather than by row.
        c.GGML_TYPE_F16 => {
            const elemsize = @sizeOf(c.ggml_fp16_t);
            c.ggml_fp32_to_fp16_row(src + @as(usize, @intCast(start)), @as([*c]c.ggml_fp16_t, @ptrCast(@alignCast(dst))) + @as(usize, @intCast(start)), n);
            result = @as(usize, @intCast(n)) * elemsize;
        },
        c.GGML_TYPE_BF16 => {
            const elemsize = @sizeOf(c.ggml_bf16_t);
            c.ggml_fp32_to_bf16_row_ref(src + @as(usize, @intCast(start)), @as([*c]c.ggml_bf16_t, @ptrCast(@alignCast(dst))) + @as(usize, @intCast(start)), n);
            result = @as(usize, @intCast(n)) * elemsize;
        },
        c.GGML_TYPE_F32 => {
            const elemsize = @sizeOf(f32);
            result = @as(usize, @intCast(n)) * elemsize;
            const out = @as([*]u8, @ptrCast(dst.?)) + @as(usize, @intCast(start)) * elemsize;
            @memcpy(out[0..result], @as([*]const u8, @ptrCast(src + @as(usize, @intCast(start))))[0..result]);
        },
        else => std.debug.assert(false),
    }

    impl.assert(result == @as(usize, @intCast(nrows)) * row_size, "result == nrows * row_size");

    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "only the three i-quant formats the C lists require an importance matrix" {
    try std.testing.expect(ggml_quantize_requires_imatrix(c.GGML_TYPE_IQ2_XXS));
    try std.testing.expect(ggml_quantize_requires_imatrix(c.GGML_TYPE_IQ2_XS));
    try std.testing.expect(ggml_quantize_requires_imatrix(c.GGML_TYPE_IQ1_S));

    // IQ1_M is commented out in the C. If a future pin uncomments it this
    // test fails, which is the point.
    try std.testing.expect(!ggml_quantize_requires_imatrix(c.GGML_TYPE_IQ1_M));
    try std.testing.expect(!ggml_quantize_requires_imatrix(c.GGML_TYPE_Q4_K));
    try std.testing.expect(!ggml_quantize_requires_imatrix(c.GGML_TYPE_F32));
}

test "the dispatch table covers every quantized format the C switches on" {
    // Q8_1 and Q8_K are quantized formats with no entry in the C's switch:
    // they exist as the intermediate the dot-product kernels quantize an
    // activation into, never as a destination a caller asks for. Reaching
    // `ggml_quantize_chunk` with either is a caller error, and the C answers
    // it with `default: assert(false)`.
    const not_destinations = [_]c.enum_ggml_type{ c.GGML_TYPE_Q8_1, c.GGML_TYPE_Q8_K };

    for (0..c.GGML_TYPE_COUNT) |i| {
        const t: c.enum_ggml_type = @intCast(i);
        // Retired slots are zeroed in the traits table and report block size
        // zero; they are not formats and belong in neither branch.
        if (types.ggml_blck_size(t) == 0) continue;
        if (!types.ggml_is_quantized(t)) continue;
        if (std.mem.indexOfScalar(c.enum_ggml_type, &not_destinations, t) != null) {
            try std.testing.expect(quantizerFor(t) == null);
            continue;
        }
        try std.testing.expect(quantizerFor(t) != null);
    }

    // The three non-block formats are handled by their own cases, not here.
    try std.testing.expect(quantizerFor(c.GGML_TYPE_F32) == null);
    try std.testing.expect(quantizerFor(c.GGML_TYPE_F16) == null);
    try std.testing.expect(quantizerFor(c.GGML_TYPE_BF16) == null);
    try std.testing.expect(quantizerFor(c.GGML_TYPE_I32) == null);
}
