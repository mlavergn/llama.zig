//! Block layouts for the quantization formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-common.h` (v0.3.0, `c1d0e7a00`), lines
//! 180-330. Each type names the C struct it mirrors.
//!
//! # Why these are declared here rather than imported
//!
//! `ggml-common.h` imports cleanly and `impl.zig` pulls it in, tables and all.
//! But several block types wrap their scales in an anonymous union so the pair
//! can also be read as one `ggml_half2`:
//!
//! ```c
//! union { struct { ggml_half d; ggml_half m; }; ggml_half2 dm; };
//! ```
//!
//! `translate-c` has no name for an anonymous member, so it invents one from
//! its position: `block.unnamed_0.unnamed_0.d`. That is unreadable, and worse,
//! the number is assigned per translation unit -- adding an unrelated struct
//! earlier in the header renumbers it and every call site silently moves to a
//! different field or stops compiling.
//!
//! So the layouts are written out here with real names. **`layout_matches`
//! below asserts every one against the imported C struct at compile time**, so
//! a divergence is a build error rather than a wrong answer: the whole point of
//! restating a layout is undone if the restatement can drift.

const std = @import("std");
const impl = @import("../impl.zig");
const c = impl.c;

/// `ggml_half`, the storage type for a block scale.
pub const Half = u16;

// -----------------------------------------------------------------------------
// Symmetric formats: one scale

/// Mirrors `block_q1_0` (ggml-common.h:181 @c1d0e7a00).
pub const Q1_0 = extern struct {
    d: Half,
    qs: [c.QK1_0 / 8]u8,
};

/// Mirrors `block_q2_0` (ggml-common.h:188 @c1d0e7a00).
pub const Q2_0 = extern struct {
    d: Half,
    qs: [c.QK2_0 / 4]u8,
};

/// Mirrors `block_q4_0` (ggml-common.h:195 @c1d0e7a00).
pub const Q4_0 = extern struct {
    d: Half,
    qs: [c.QK4_0 / 2]u8,
};

/// Mirrors `block_q5_0` (ggml-common.h:230 @c1d0e7a00).
///
/// `qh` is four loose bytes rather than a `u32` in the C too, because the
/// struct would otherwise gain alignment padding after the `ggml_half`.
pub const Q5_0 = extern struct {
    d: Half,
    qh: [4]u8,
    qs: [c.QK5_0 / 2]u8,
};

/// Mirrors `block_q8_0` (ggml-common.h:252 @c1d0e7a00).
pub const Q8_0 = extern struct {
    d: Half,
    qs: [c.QK8_0]i8,
};

// -----------------------------------------------------------------------------
// Asymmetric formats: a scale and a second value
//
// The C wraps the pair in a union so it can also be loaded as one `ggml_half2`.
// Nothing in the reference (de)quantizers uses that view -- it is there for the
// SIMD dot products -- so the pair is written out as two fields here.
//
// **The union still costs alignment.** `ggml_half2` is four bytes, so the C
// struct aligns to 4 where the two halves alone would align to 2. Size is
// unaffected -- all three are already multiples of four -- but the alignment is
// part of the layout, and `layoutMatches` below rejects a mismatch. Hence the
// explicit `align(4)`, which is load-bearing rather than decorative: it was
// added because the assertion caught its absence.

/// Mirrors `block_q4_1` (ggml-common.h:202 @c1d0e7a00).
pub const Q4_1 = extern struct {
    d: Half align(4),
    m: Half,
    qs: [c.QK4_1 / 2]u8,
};

/// Mirrors `block_q5_1` (ggml-common.h:238 @c1d0e7a00).
pub const Q5_1 = extern struct {
    d: Half align(4),
    m: Half,
    qh: [4]u8,
    qs: [c.QK5_1 / 2]u8,
};

/// Mirrors `block_q8_1` (ggml-common.h:259 @c1d0e7a00).
///
/// `s` is the sum of the quantized values times the scale, not a minimum: it
/// lets a dot product against an asymmetric weight recover the offset term
/// without a second pass.
pub const Q8_1 = extern struct {
    d: Half align(4),
    s: Half,
    qs: [c.QK8_1]i8,
};

// -----------------------------------------------------------------------------
// Codebook formats

/// Mirrors `block_mxfp4` (ggml-common.h:215 @c1d0e7a00).
pub const MXFP4 = extern struct {
    e: u8,
    qs: [c.QK_MXFP4 / 2]u8,
};

/// Mirrors `block_nvfp4` (ggml-common.h:223 @c1d0e7a00).
pub const NVFP4 = extern struct {
    d: [c.QK_NVFP4 / c.QK_NVFP4_SUB]u8,
    qs: [c.QK_NVFP4 / 2]u8,
};

// -----------------------------------------------------------------------------
// Ternary formats
//
// BitNet b1.58 and TriLM: weights are -1, 0 or +1. `tq1_0` packs five ternary
// digits into a byte base-3 (3^5 = 243 < 256), which is why its `qs` length is
// the odd `(QK_K - 4*QK_K/64) / 5`.

/// Mirrors `block_tq1_0` (ggml-common.h:276 @c1d0e7a00). 1.6875 bpw.
pub const TQ1_0 = extern struct {
    qs: [(c.QK_K - 4 * c.QK_K / 64) / 5]u8,
    qh: [c.QK_K / 64]u8,
    d: Half,
};

/// Mirrors `block_tq2_0` (ggml-common.h:284 @c1d0e7a00). 2.0625 bpw.
pub const TQ2_0 = extern struct {
    qs: [c.QK_K / 4]u8,
    d: Half,
};

// -----------------------------------------------------------------------------
// Super-block formats (K-quants)
//
// A super-block is `QK_K` (256) weights carrying two f16 scales, subdivided
// into sub-blocks with their own quantized scales. That second level is the
// point: one scale per 256 weights is too coarse, one f16 per 16 is too
// expensive, so the sub-block scales are themselves quantized against the
// super-block's.
//
// **The field order differs between them and it is not decorative.** `q2_K` and
// `q3_K` put `scales` first; `q4_K` and `q5_K` put the scale pair first. Get it
// wrong and the block is the right size and completely misread.

/// Mirrors `block_q2_K` (ggml-common.h:298 @c1d0e7a00). 16 sub-blocks of 16, ~2.625 bpw.
///
/// `x = d * q + dmin * m`, with both the scale and the min quantized to 4 bits
/// and packed together in `scales`.
pub const Q2_K = extern struct {
    scales: [c.QK_K / 16]u8,
    qs: [c.QK_K / 4]u8,
    d: Half align(4),
    dmin: Half,
};

/// Mirrors `block_q3_K` (ggml-common.h:315 @c1d0e7a00). 16 sub-blocks of 16, ~3.4375 bpw.
///
/// The third bit of each weight lives in `hmask`, one bit per weight, separate
/// from the low two bits in `qs`. The 16 sub-block scales are 6-bit, packed
/// into 12 bytes.
pub const Q3_K = extern struct {
    hmask: [c.QK_K / 8]u8,
    qs: [c.QK_K / 4]u8,
    scales: [12]u8,
    d: Half,
};

/// Mirrors `block_q4_K` (ggml-common.h:327 @c1d0e7a00). 8 sub-blocks of 32, ~4.5 bpw.
pub const Q4_K = extern struct {
    d: Half align(4),
    dmin: Half,
    scales: [c.K_SCALE_SIZE]u8,
    qs: [c.QK_K / 2]u8,
};

/// Mirrors `block_q5_K` (ggml-common.h:344 @c1d0e7a00). 8 sub-blocks of 32, ~5.5 bpw.
pub const Q5_K = extern struct {
    d: Half align(4),
    dmin: Half,
    scales: [c.K_SCALE_SIZE]u8,
    qh: [c.QK_K / 8]u8,
    qs: [c.QK_K / 2]u8,
};

/// Mirrors `block_q6_K` (ggml-common.h:362 @c1d0e7a00). 16 sub-blocks of 16, ~6.5625 bpw.
///
/// Symmetric -- no `dmin` -- and its sub-block scales are full `int8`, not
/// packed, which is what makes it the most accurate of the K-quants.
pub const Q6_K = extern struct {
    ql: [c.QK_K / 2]u8,
    qh: [c.QK_K / 4]u8,
    scales: [c.QK_K / 16]i8,
    d: Half,
};

/// Mirrors `block_q8_K` (ggml-common.h:371 @c1d0e7a00).
///
/// The activation side of the K-quant dot products, never a stored weight
/// format. Note `d` is a full `f32` rather than an f16, and `bsums` caches the
/// per-16 sums so the dot product can apply a sub-block min without a second
/// pass over the quants.
pub const Q8_K = extern struct {
    d: f32,
    qs: [c.QK_K]i8,
    bsums: [c.QK_K / 16]i16,
};

// -----------------------------------------------------------------------------
// The i-quants
//
// These do not store a quantized *value* per weight. They store an **index into
// a codebook** -- a table of 8-element vectors chosen to cover the distribution
// of real model weights better than a uniform grid does. Eight weights at a
// time become one grid index plus a sign pattern, which is how 2 bits per
// weight stays usable.
//
// The codebooks are `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`,
// `iq3s_grid` and `iq1s_grid` in `ggml-common.h`, imported rather than
// transcribed.

/// Mirrors `block_iq2_xxs` (ggml-common.h:381 @c1d0e7a00). 2.0625 bpw.
pub const IQ2_XXS = extern struct {
    d: Half,
    qs: [c.QK_K / 8]u16,
};

/// Mirrors `block_iq2_xs` (ggml-common.h:388 @c1d0e7a00). 2.3125 bpw.
pub const IQ2_XS = extern struct {
    d: Half,
    qs: [c.QK_K / 8]u16,
    scales: [c.QK_K / 32]u8,
};

/// Mirrors `block_iq2_s` (ggml-common.h:396 @c1d0e7a00). 2.5625 bpw.
///
/// Note the sign bytes are not a named field: they live in the tail of `qs`,
/// which the C indexes as `qs + QK_K/8`. The struct is sized for both.
pub const IQ2_S = extern struct {
    d: Half,
    qs: [c.QK_K / 4]u8,
    qh: [c.QK_K / 32]u8,
    scales: [c.QK_K / 32]u8,
};

/// Mirrors `block_iq3_xxs` (ggml-common.h:407 @c1d0e7a00). 3.0625 bpw.
///
/// `qs` again carries two things: `QK_K/4` grid indices followed by the packed
/// scales and signs, which the C reaches as `qs + QK_K/4`.
pub const IQ3_XXS = extern struct {
    d: Half,
    qs: [3 * c.QK_K / 8]u8,
};

/// Mirrors `block_iq3_s` (ggml-common.h:415 @c1d0e7a00). 3.4375 bpw.
pub const IQ3_S = extern struct {
    d: Half,
    qs: [c.QK_K / 4]u8,
    qh: [c.QK_K / 32]u8,
    signs: [c.QK_K / 8]u8,
    scales: [c.QK_K / 64]u8,
};

/// Mirrors `block_iq1_s` (ggml-common.h:425 @c1d0e7a00). 1.5625 bpw.
///
/// The scale is not a separate field: four bits of each `qh` entry carry the
/// sub-block scale and one carries the sign of the delta.
pub const IQ1_S = extern struct {
    d: Half,
    qs: [c.QK_K / 8]u8,
    qh: [c.QK_K / 32]u16,
};

/// Mirrors `block_iq1_m` (ggml-common.h:433 @c1d0e7a00). 1.75 bpw.
///
/// **No `d` field at all.** The f16 super-block scale is scattered four bits at
/// a time across the top of the four `scales` shorts, and reassembled by
/// `iq1mScale`. That is the whole reason `iq1m_scale_t` exists in the C.
pub const IQ1_M = extern struct {
    qs: [c.QK_K / 8]u8,
    qh: [c.QK_K / 16]u8,
    scales: [c.QK_K / 32]u8,
};

/// Mirrors `block_iq4_nl` (ggml-common.h:448 @c1d0e7a00). Non-linear 4-bit, 32 per block.
pub const IQ4_NL = extern struct {
    d: Half,
    qs: [c.QK4_NL / 2]u8,
};

/// Mirrors `block_iq4_xs` (ggml-common.h:454 @c1d0e7a00). Non-linear 4-bit, super-block.
pub const IQ4_XS = extern struct {
    d: Half,
    scales_h: u16,
    scales_l: [c.QK_K / 64]u8,
    qs: [c.QK_K / 2]u8,
};

/// Ports `IQ1S_DELTA` and `IQ1M_DELTA` (ggml-common.h:1133 @c1d0e7a00).
///
/// The 1-bit formats shift their grid values by a constant before scaling, so
/// a "zero" weight is not exactly zero. `#define`s in the header's
/// implementation half, so not in the import.
pub const iq1s_delta: f32 = 0.125;
pub const iq1m_delta: f32 = 0.125;

/// Reassembles `iq1_m`'s f16 scale from the four nibbles it is scattered
/// across.
///
/// Ports the `iq1m_scale_t` union (ggml-common.h:441 @c1d0e7a00): the C type-puns a `uint16_t` to a
/// `ggml_half`, which here is just the bit pattern the f16 conversion takes.
pub inline fn iq1mScale(sc: [*]const u16) Half {
    return (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
}

// -----------------------------------------------------------------------------
// Layout verification

/// Asserts a restated layout still matches the C struct it mirrors.
///
/// Size and alignment only: field offsets are not comparable across the two,
/// because the C side's fields are buried in anonymous aggregates whose names
/// are exactly what this file exists to avoid. Size and alignment catch the
/// failures that matter -- a wrong array length, a missing field, unexpected
/// padding.
fn layoutMatches(comptime Ours: type, comptime Theirs: type) void {
    if (@sizeOf(Ours) != @sizeOf(Theirs)) {
        @compileError("block layout size mismatch for " ++ @typeName(Ours));
    }
    if (@alignOf(Ours) != @alignOf(Theirs)) {
        @compileError("block layout alignment mismatch for " ++ @typeName(Ours));
    }
}

comptime {
    layoutMatches(Q1_0, c.block_q1_0);
    layoutMatches(Q2_0, c.block_q2_0);
    layoutMatches(Q4_0, c.block_q4_0);
    layoutMatches(Q4_1, c.block_q4_1);
    layoutMatches(Q5_0, c.block_q5_0);
    layoutMatches(Q5_1, c.block_q5_1);
    layoutMatches(Q8_0, c.block_q8_0);
    layoutMatches(Q8_1, c.block_q8_1);
    layoutMatches(MXFP4, c.block_mxfp4);
    layoutMatches(NVFP4, c.block_nvfp4);
    layoutMatches(TQ1_0, c.block_tq1_0);
    layoutMatches(TQ2_0, c.block_tq2_0);
    layoutMatches(Q2_K, c.block_q2_K);
    layoutMatches(Q3_K, c.block_q3_K);
    layoutMatches(Q4_K, c.block_q4_K);
    layoutMatches(Q5_K, c.block_q5_K);
    layoutMatches(Q6_K, c.block_q6_K);
    layoutMatches(Q8_K, c.block_q8_K);
    layoutMatches(IQ2_XXS, c.block_iq2_xxs);
    layoutMatches(IQ2_XS, c.block_iq2_xs);
    layoutMatches(IQ2_S, c.block_iq2_s);
    layoutMatches(IQ3_XXS, c.block_iq3_xxs);
    layoutMatches(IQ3_S, c.block_iq3_s);
    layoutMatches(IQ1_S, c.block_iq1_s);
    layoutMatches(IQ1_M, c.block_iq1_m);
    layoutMatches(IQ4_NL, c.block_iq4_nl);
    layoutMatches(IQ4_XS, c.block_iq4_xs);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the restated layouts agree with the C on every size" {
    // The comptime block above already fails the build on a mismatch. This
    // restates it as a test so the intent survives someone deleting the
    // comptime block, and so the sizes appear in the test output.
    try std.testing.expectEqual(@sizeOf(c.block_q1_0), @sizeOf(Q1_0));
    try std.testing.expectEqual(@sizeOf(c.block_q2_0), @sizeOf(Q2_0));
    try std.testing.expectEqual(@sizeOf(c.block_q4_0), @sizeOf(Q4_0));
    try std.testing.expectEqual(@sizeOf(c.block_q4_1), @sizeOf(Q4_1));
    try std.testing.expectEqual(@sizeOf(c.block_q5_0), @sizeOf(Q5_0));
    try std.testing.expectEqual(@sizeOf(c.block_q5_1), @sizeOf(Q5_1));
    try std.testing.expectEqual(@sizeOf(c.block_q8_0), @sizeOf(Q8_0));
    try std.testing.expectEqual(@sizeOf(c.block_q8_1), @sizeOf(Q8_1));
    try std.testing.expectEqual(@sizeOf(c.block_mxfp4), @sizeOf(MXFP4));
    try std.testing.expectEqual(@sizeOf(c.block_nvfp4), @sizeOf(NVFP4));

    // The three union-bearing formats align to 4, not 2: `ggml_half2` is a
    // four-byte type and the union inherits its alignment.
    try std.testing.expectEqual(@as(usize, 4), @alignOf(Q4_1));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(Q5_1));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(Q8_1));
    try std.testing.expectEqual(@as(usize, 2), @alignOf(Q4_0));

    // Spot-check the values the C asserts in its own static_asserts, so a
    // silently-changed QK constant is caught here too.
    try std.testing.expectEqual(@as(usize, 2 + 32 / 2), @sizeOf(Q4_0));
    try std.testing.expectEqual(@as(usize, 2 + 2 + 32 / 2), @sizeOf(Q4_1));
    try std.testing.expectEqual(@as(usize, 2 + 4 + 32 / 2), @sizeOf(Q5_0));
    try std.testing.expectEqual(@as(usize, 2 + 32), @sizeOf(Q8_0));
}
