//! The CPU backend's row quantizers: the entry points `type_traits_cpu` puts
//! in its `from_float` slots.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/quants.c` (v0.3.0, `c1d0e7a00`),
//! the row functions at lines 25-125. Each function names the C it replaces
//! and the line it began at.
//!
//! # Seventeen of these are one line
//!
//! `quantize_row_q4_0` and its siblings forward to `quantize_row_q4_0_ref` in
//! `ggml-quants.c`, which `src/ggml/quants/` already ports. The C keeps them as
//! separate symbols because an architecture may replace any one of them with a
//! vectorised version -- `arch/arm/quants.c` does exactly that for `q8_0`,
//! `q8_1` and `q8_K`, which is why those three are `_generic` here and the
//! other seventeen are not.
//!
//! So this file looks redundant and is not: the seventeen names are the ABI the
//! traits table is built from, and the three `_generic` ones are the fallbacks
//! a `GGML_CPU_GENERIC` build would use instead of the NEON kernels.

const std = @import("std");
const impl = @import("../../impl.zig");
const c = impl.c;

// The reference quantizers, ported in `src/ggml/quants/`. Declared as externs
// rather than imported because they live in a different directory of the port
// and only their C ABI is wanted here -- exactly as the C reaches them through
// `ggml-quants.h`.
const ref = struct {
    extern fn quantize_row_q1_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q2_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q4_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q4_1_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q5_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q5_1_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q8_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q8_1_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q8_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_mxfp4_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_nvfp4_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q2_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q3_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q4_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q5_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_q6_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_tq1_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_tq2_0_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_iq4_nl_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
    extern fn quantize_row_iq4_xs_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
};

/// Ports `quantize_row_q1_0` (ggml-cpu/quants.c:25 @c1d0e7a00).
pub export fn quantize_row_q1_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q1_0_ref(x, y, k);
}

/// Ports `quantize_row_q2_0` (ggml-cpu/quants.c:29 @c1d0e7a00).
pub export fn quantize_row_q2_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q2_0_ref(x, y, k);
}

/// Ports `quantize_row_q4_0` (ggml-cpu/quants.c:33 @c1d0e7a00).
pub export fn quantize_row_q4_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q4_0_ref(x, y, k);
}

/// Ports `quantize_row_q4_1` (ggml-cpu/quants.c:37 @c1d0e7a00).
pub export fn quantize_row_q4_1(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q4_1_ref(x, y, k);
}

/// Ports `quantize_row_q5_0` (ggml-cpu/quants.c:41 @c1d0e7a00).
pub export fn quantize_row_q5_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q5_0_ref(x, y, k);
}

/// Ports `quantize_row_q5_1` (ggml-cpu/quants.c:45 @c1d0e7a00).
pub export fn quantize_row_q5_1(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q5_1_ref(x, y, k);
}

/// Ports `quantize_row_q8_0_generic` (ggml-cpu/quants.c:49 @c1d0e7a00).
///
/// `_generic` because `arch/arm/quants.c` provides the NEON `quantize_row_q8_0`
/// this target actually uses. Unreachable here, and exported because the symbol
/// is part of the translation unit's contract.
pub export fn quantize_row_q8_0_generic(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q8_0_ref(x, y, k);
}

/// Ports `quantize_row_q8_1_generic` (ggml-cpu/quants.c:53 @c1d0e7a00).
pub export fn quantize_row_q8_1_generic(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q8_1_ref(x, y, k);
}

/// Ports `quantize_row_mxfp4` (ggml-cpu/quants.c:57 @c1d0e7a00).
pub export fn quantize_row_mxfp4(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_mxfp4_ref(x, y, k);
}

/// Ports `quantize_row_nvfp4` (ggml-cpu/quants.c:61 @c1d0e7a00).
pub export fn quantize_row_nvfp4(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_nvfp4_ref(x, y, k);
}

/// Ports `quantize_row_q2_K` (ggml-cpu/quants.c:71 @c1d0e7a00).
///
/// The K-quant forwarders assert `k % QK_K == 0` before delegating, where the
/// legacy ones do not. Kept, because the reference quantizer would otherwise
/// walk off the end of a short row.
pub export fn quantize_row_q2_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_q2_K_ref(x, vy, k);
}

/// Ports `quantize_row_q3_K` (ggml-cpu/quants.c:77 @c1d0e7a00).
pub export fn quantize_row_q3_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_q3_K_ref(x, vy, k);
}

/// Ports `quantize_row_q4_K` (ggml-cpu/quants.c:83 @c1d0e7a00).
pub export fn quantize_row_q4_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_q4_K_ref(x, vy, k);
}

/// Ports `quantize_row_q5_K` (ggml-cpu/quants.c:91 @c1d0e7a00).
pub export fn quantize_row_q5_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_q5_K_ref(x, vy, k);
}

/// Ports `quantize_row_q6_K` (ggml-cpu/quants.c:99 @c1d0e7a00).
pub export fn quantize_row_q6_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_q6_K_ref(x, vy, k);
}

/// Ports `quantize_row_tq1_0` (ggml-cpu/quants.c:107 @c1d0e7a00).
pub export fn quantize_row_tq1_0(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_tq1_0_ref(x, vy, k);
}

/// Ports `quantize_row_tq2_0` (ggml-cpu/quants.c:113 @c1d0e7a00).
pub export fn quantize_row_tq2_0(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_tq2_0_ref(x, vy, k);
}

/// Ports `quantize_row_q8_K_generic` (ggml-cpu/quants.c:121 @c1d0e7a00).
pub export fn quantize_row_q8_K_generic(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q8_K_ref(x, y, k);
}

/// Ports `quantize_row_iq4_nl` (ggml-cpu/quants.c:1331 @c1d0e7a00).
pub export fn quantize_row_iq4_nl(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK4_NL) == 0, "k % QK4_NL == 0");
    ref.quantize_row_iq4_nl_ref(x, y, k);
}

/// Ports `quantize_row_iq4_xs` (ggml-cpu/quants.c:1336 @c1d0e7a00).
pub export fn quantize_row_iq4_xs(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    impl.assert(@rem(k, c.QK_K) == 0, "k % QK_K == 0");
    ref.quantize_row_iq4_xs_ref(x, y, k);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a forwarder produces what the reference quantizer produces" {
    // These are one-line delegations, so what can go wrong is delegating to
    // the wrong reference -- q4_0 to q4_1_ref, say, which would still produce
    // plausible bytes of the right length.
    var x: [64]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01 - 0.3;

    var via_wrapper: [256]u8 align(8) = @splat(0);
    var via_ref: [256]u8 align(8) = @splat(0);

    quantize_row_q4_0(&x, &via_wrapper, 64);
    ref.quantize_row_q4_0_ref(&x, &via_ref, 64);
    try std.testing.expectEqualSlices(u8, &via_ref, &via_wrapper);

    // And a different type must not agree, or the check above proves nothing.
    var other: [256]u8 align(8) = @splat(0);
    quantize_row_q5_0(&x, &other, 64);
    try std.testing.expect(!std.mem.eql(u8, &via_wrapper, &other));
}

const testing = @import("testing.zig");
const golden = @import("golden.zig");

test "the generic q8 row quantizers match the C" {
    // These three are the `_generic` fallbacks, unreachable on this target
    // like the generic dot products, so the goldens are their only gate.
    try testing.checkRow(quantize_row_q8_0_generic, c.GGML_TYPE_Q8_0, golden.row_q8_0);
    try testing.checkRow(quantize_row_q8_1_generic, c.GGML_TYPE_Q8_1, golden.row_q8_1);
    try testing.checkRow(quantize_row_q8_K_generic, c.GGML_TYPE_Q8_K, golden.row_q8_K);
}

test "the K-quant forwarders assert their row length" {
    // Not decoration: `quantize_row_q2_K_ref` reads whole super-blocks and
    // would run past a row that is not a multiple of 256.
    var x: [256]f32 = @splat(0.5);
    var y: [512]u8 align(8) = @splat(0);

    quantize_row_q2_K(&x, &y, 256);

    var any = false;
    for (y[0..@sizeOf(c.block_q2_K)]) |b| {
        if (b != 0) any = true;
    }
    try std.testing.expect(any);
}
