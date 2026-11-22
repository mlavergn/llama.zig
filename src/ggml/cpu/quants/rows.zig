//! The CPU backend's row quantizers: the entry points `type_traits_cpu` puts
//! in its `from_float` slots.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/quants.c` (v1.0.0, `b1b770d90`),
//! the row functions at lines 26-130. Each function names the C it replaces
//! and the line it began at.
//!
//! # Eighteen of these are one line
//!
//! `quantize_row_q4_0` and its siblings forward to `quantize_row_q4_0_ref` in
//! `ggml-quants.c`, which `src/ggml/quants/` already ports. The C keeps them as
//! separate symbols because an architecture may replace any one of them with a
//! vectorised version -- `arch/arm/quants.c` does exactly that for `q8_0`,
//! `q8_1` and `q8_K`, which is why those three are `_generic` here and the
//! other eighteen are not.
//!
//! So this file looks redundant and is not: the eighteen names are the ABI the
//! traits table is built from, and the three `_generic` ones are the fallbacks
//! a `GGML_CPU_GENERIC` build would use instead of the NEON kernels.

const std = @import("std");
const ggml = @import("../../module.zig");

// The reference quantizers, ported in `src/ggml/quants/`. Declared as externs
// rather than imported because they live in a different directory of the port
// and only their C ABI is wanted here -- exactly as the C reaches them through
// `ggml-quants.h`.

/// Ports `quantize_row_q1_0` (ggml-cpu/quants.c:26 @b1b770d90).
pub export fn quantize_row_q1_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q1_0_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q2_0` (ggml-cpu/quants.c:30 @b1b770d90).
pub export fn quantize_row_q2_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q2_0_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q4_0` (ggml-cpu/quants.c:34 @b1b770d90).
pub export fn quantize_row_q4_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q4_0_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q4_1` (ggml-cpu/quants.c:38 @b1b770d90).
pub export fn quantize_row_q4_1(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q4_1_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q5_0` (ggml-cpu/quants.c:42 @b1b770d90).
pub export fn quantize_row_q5_0(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q5_0_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q5_1` (ggml-cpu/quants.c:46 @b1b770d90).
pub export fn quantize_row_q5_1(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q5_1_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q8_0_generic` (ggml-cpu/quants.c:50 @b1b770d90).
///
/// `_generic` because `arch/arm/quants.c` provides the NEON `quantize_row_q8_0`
/// this target actually uses. Unreachable here, and exported because the symbol
/// is part of the translation unit's contract.
pub export fn quantize_row_q8_0_generic(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q8_0_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q8_1_generic` (ggml-cpu/quants.c:54 @b1b770d90).
pub export fn quantize_row_q8_1_generic(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_q8_1_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_mxfp4` (ggml-cpu/quants.c:58 @b1b770d90).
pub export fn quantize_row_mxfp4(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_mxfp4_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_nvfp4` (ggml-cpu/quants.c:62 @b1b770d90).
pub export fn quantize_row_nvfp4(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_nvfp4_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_f8_e4m3` (ggml-cpu/quants.c:66 @b1b770d90).
pub export fn quantize_row_f8_e4m3(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.legacy.quantize_row_f8_e4m3_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_q2_K` (ggml-cpu/quants.c:76 @b1b770d90).
///
/// The K-quant forwarders assert `k % QK_K == 0` before delegating, where the
/// legacy ones do not. Kept, because the reference quantizer would otherwise
/// walk off the end of a short row.
pub export fn quantize_row_q2_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.k.quantize_row_q2_K_ref(x, @ptrCast(@alignCast(vy)), k);
}

/// Ports `quantize_row_q3_K` (ggml-cpu/quants.c:82 @b1b770d90).
pub export fn quantize_row_q3_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.k.quantize_row_q3_K_ref(x, @ptrCast(@alignCast(vy)), k);
}

/// Ports `quantize_row_q4_K` (ggml-cpu/quants.c:88 @b1b770d90).
pub export fn quantize_row_q4_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.k.quantize_row_q4_K_ref(x, @ptrCast(@alignCast(vy)), k);
}

/// Ports `quantize_row_q5_K` (ggml-cpu/quants.c:96 @b1b770d90).
pub export fn quantize_row_q5_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.k.quantize_row_q5_K_ref(x, @ptrCast(@alignCast(vy)), k);
}

/// Ports `quantize_row_q6_K` (ggml-cpu/quants.c:104 @b1b770d90).
pub export fn quantize_row_q6_K(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.k.quantize_row_q6_K_ref(x, @ptrCast(@alignCast(vy)), k);
}

/// Ports `quantize_row_tq1_0` (ggml-cpu/quants.c:112 @b1b770d90).
pub export fn quantize_row_tq1_0(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.ternary.quantize_row_tq1_0_ref(x, @ptrCast(@alignCast(vy)), k);
}

/// Ports `quantize_row_tq2_0` (ggml-cpu/quants.c:118 @b1b770d90).
pub export fn quantize_row_tq2_0(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.ternary.quantize_row_tq2_0_ref(x, @ptrCast(@alignCast(vy)), k);
}

/// Ports `quantize_row_q8_K_generic` (ggml-cpu/quants.c:126 @b1b770d90).
pub export fn quantize_row_q8_K_generic(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.quants.k.quantize_row_q8_K_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_iq4_nl` (ggml-cpu/quants.c:1379 @b1b770d90).
pub export fn quantize_row_iq4_nl(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK4_NL) == 0, "k % QK4_NL == 0");
    ggml.quants.iq4.quantize_row_iq4_nl_ref(x, @ptrCast(@alignCast(y)), k);
}

/// Ports `quantize_row_iq4_xs` (ggml-cpu/quants.c:1384 @b1b770d90).
pub export fn quantize_row_iq4_xs(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ggml.impl.assert(@rem(k, ggml.impl.c.QK_K) == 0, "k % QK_K == 0");
    ggml.quants.iq4.quantize_row_iq4_xs_ref(x, @ptrCast(@alignCast(y)), k);
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
    ggml.quants.legacy.quantize_row_q4_0_ref(&x, @ptrCast(&via_ref), 64);
    try std.testing.expectEqualSlices(u8, &via_ref, &via_wrapper);

    // And a different type must not agree, or the check above proves nothing.
    var other: [256]u8 align(8) = @splat(0);
    quantize_row_q5_0(&x, &other, 64);
    try std.testing.expect(!std.mem.eql(u8, &via_wrapper, &other));
}

test "the generic q8 row quantizers match the C" {
    // These three are the `_generic` fallbacks, unreachable on this target
    // like the generic dot products, so the goldens are their only gate.
    try ggml.cpu.quants.testing.checkRow(quantize_row_q8_0_generic, ggml.impl.c.GGML_TYPE_Q8_0, ggml.cpu.quants.golden.row_q8_0);
    try ggml.cpu.quants.testing.checkRow(quantize_row_q8_1_generic, ggml.impl.c.GGML_TYPE_Q8_1, ggml.cpu.quants.golden.row_q8_1);
    try ggml.cpu.quants.testing.checkRow(quantize_row_q8_K_generic, ggml.impl.c.GGML_TYPE_Q8_K, ggml.cpu.quants.golden.row_q8_K);
}

test "the K-quant forwarders assert their row length" {
    // Not decoration: `quantize_row_q2_K_ref` reads whole super-blocks and
    // would run past a row that is not a multiple of 256.
    var x: [256]f32 = @splat(0.5);
    var y: [512]u8 align(8) = @splat(0);

    quantize_row_q2_K(&x, &y, 256);

    var any = false;
    for (y[0..@sizeOf(ggml.impl.c.block_q2_K)]) |b| {
        if (b != 0) any = true;
    }
    try std.testing.expect(any);
}
