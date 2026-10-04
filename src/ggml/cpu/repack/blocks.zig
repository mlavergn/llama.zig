//! The interleaved block layouts `repack` produces.
//!
//! # Provenance
//!
//! Mirrors `llama.cpp/ggml/src/ggml-cpu/repack.h` at v0.3.0 (`c1d0e7a00`).
//!
//! These are `extern struct` because the layout is the on-disk-equivalent
//! contract the gemm kernels read; each carries a `comptime` size assertion
//! against the C's own `static_assert`, which is what catches a mistyped
//! array bound.

const std = @import("std");
const impl = @import("../../impl.zig");

const c = impl.c;
const half = c.ggml_half;

const QK4_0 = 32;
const QK8_0 = 32;
const QK_K = 256;
const K_SCALE_SIZE = 12;
const QK4_NL = 32;
const QK_MXFP4 = 32;

/// Mirrors `block` (ggml-cpu/repack.h:23 @c1d0e7a00), the C's `template <int K, int N>`:
/// `N` interleaved `qK_0` blocks.
pub fn Block(comptime K: comptime_int, comptime N: comptime_int) type {
    const qk = switch (K) {
        4 => QK4_0,
        8 => QK8_0,
        else => @compileError("QK_0<K>() is -1 for this K"),
    };
    return extern struct {
        /// deltas for N qK_0 blocks
        d: [N]half,
        /// quants for N qK_0 blocks
        qs: [(qk * N * K) / 8]i8,
    };
}

pub const block_q4_0x4 = Block(4, 4);
pub const block_q4_0x8 = Block(4, 8);
pub const block_q4_0x16 = Block(4, 16);
pub const block_q8_0x4 = Block(8, 4);
pub const block_q8_0x8 = Block(8, 8);
pub const block_q8_0x16 = Block(8, 16);

/// Mirrors `block_q4_Kx8` (ggml-cpu/repack.h:43 @c1d0e7a00).
pub const block_q4_Kx8 = extern struct {
    d: [8]half,
    dmin: [8]half,
    scales: [96]u8,
    qs: [1024]u8,
};

/// Mirrors `block_q4_Kx16` (ggml-cpu/repack.h:51 @c1d0e7a00).
pub const block_q4_Kx16 = extern struct {
    d: [16]half,
    dmin: [16]half,
    scales: [192]u8,
    qs: [2048]u8,
};

/// Mirrors `block_q2_Kx8` (ggml-cpu/repack.h:59 @c1d0e7a00).
pub const block_q2_Kx8 = extern struct {
    d: [8]half,
    dmin: [8]half,
    scales: [128]u8,
    qs: [512]u8,
};

/// Mirrors `block_q2_Kx16` (ggml-cpu/repack.h:67 @c1d0e7a00).
pub const block_q2_Kx16 = extern struct {
    d: [16]half,
    dmin: [16]half,
    scales: [256]u8,
    qs: [1024]u8,
};

/// Mirrors `block_q5_Kx8` (ggml-cpu/repack.h:75 @c1d0e7a00).
pub const block_q5_Kx8 = extern struct {
    d: [8]half,
    dmin: [8]half,
    scales: [96]u8,
    qh: [QK_K * 8 / 8]u8,
    qs: [QK_K * 8 / 2]u8,
};

/// Mirrors `block_q6_Kx8` (ggml-cpu/repack.h:86 @c1d0e7a00).
pub const block_q6_Kx8 = extern struct {
    d: [8]half,
    scales: [QK_K / 16 * 8]i8,
    ql: [QK_K / 2 * 8]u8,
    qh: [QK_K / 4 * 8]u8,
};

/// Mirrors `block_q8_Kx4` (ggml-cpu/repack.h:96 @c1d0e7a00).
pub const block_q8_Kx4 = extern struct {
    d: [4]f32,
    qs: [QK_K * 4]i8,
    bsums: [QK_K / 4]i16,
};

/// Mirrors `block_iq4_nlx4` (ggml-cpu/repack.h:104 @c1d0e7a00).
pub const block_iq4_nlx4 = extern struct {
    d: [4]half,
    qs: [QK4_NL * 2]u8,
};

/// Mirrors `block_iq4_nlx8` (ggml-cpu/repack.h:111 @c1d0e7a00).
pub const block_iq4_nlx8 = extern struct {
    d: [8]half,
    qs: [QK4_NL * 4]u8,
};

/// Mirrors `block_iq4_nlx16` (ggml-cpu/repack.h:118 @c1d0e7a00).
pub const block_iq4_nlx16 = extern struct {
    d: [16]half,
    qs: [QK4_NL * 8]u8,
};

/// Mirrors `block_mxfp4x4` (ggml-cpu/repack.h:124 @c1d0e7a00).
pub const block_mxfp4x4 = extern struct {
    e: [4]u8,
    qs: [QK_MXFP4 * 2]u8,
};

/// Mirrors `block_mxfp4x8` (ggml-cpu/repack.h:130 @c1d0e7a00).
pub const block_mxfp4x8 = extern struct {
    e: [8]u8,
    qs: [QK_MXFP4 * 4]u8,
};

// The C's own `static_assert`s, which are what catch a mistyped bound.
comptime {
    const h = @sizeOf(half);
    std.debug.assert(@sizeOf(block_q4_0x4) == 4 * h + QK8_0 * 2);
    std.debug.assert(@sizeOf(block_q4_0x8) == 8 * h + QK8_0 * 4);
    std.debug.assert(@sizeOf(block_q4_0x16) == 16 * h + QK8_0 * 8);
    std.debug.assert(@sizeOf(block_q8_0x4) == 4 * h + QK8_0 * 4);
    std.debug.assert(@sizeOf(block_q8_0x8) == 8 * h + QK8_0 * 8);
    std.debug.assert(@sizeOf(block_q8_0x16) == 16 * h + QK8_0 * 16);
    std.debug.assert(@sizeOf(block_q4_Kx8) == h * 16 + K_SCALE_SIZE * 8 + QK_K * 4);
    std.debug.assert(@sizeOf(block_q4_Kx16) == h * 32 + K_SCALE_SIZE * 16 + QK_K * 8);
    std.debug.assert(@sizeOf(block_q2_Kx8) == h * 16 + QK_K / 2 + QK_K * 2);
    std.debug.assert(@sizeOf(block_q2_Kx16) == h * 32 + QK_K + QK_K * 4);
    std.debug.assert(@sizeOf(block_q5_Kx8) == h * 16 + K_SCALE_SIZE * 8 + QK_K * 5);
    std.debug.assert(@sizeOf(block_q6_Kx8) == h * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8);
    std.debug.assert(@sizeOf(block_q8_Kx4) == @sizeOf(f32) * 4 + QK_K * 4 + (QK_K / 4) * @sizeOf(i16));
    std.debug.assert(@sizeOf(block_iq4_nlx4) == 4 * h + QK4_NL * 2);
    std.debug.assert(@sizeOf(block_iq4_nlx8) == 8 * h + QK4_NL * 4);
    std.debug.assert(@sizeOf(block_iq4_nlx16) == 16 * h + QK4_NL * 8);
    std.debug.assert(@sizeOf(block_mxfp4x4) == 4 + QK_MXFP4 * 2);
    std.debug.assert(@sizeOf(block_mxfp4x8) == 8 + QK_MXFP4 * 4);
}
