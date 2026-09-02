//! The NEON row quantizers: `q8_0`, `q8_1` and `q8_K`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (v0.3.0,
//! `c1d0e7a00`), the row functions at lines 41, 85 and 134. Each function names
//! the C it replaces and the line it began at.
//!
//! # These are live, and they feed everything
//!
//! `type_traits_cpu` puts them in the `from_float` slots for `q8_0`, `q8_1` and
//! `q8_K`, which are the `vec_dot_type` of nearly every quantized format. So a
//! quantized `mul_mat` with an `f32` right-hand operand stages it through one
//! of these first: a wrong byte here is a wrong answer for every type.
//!
//! # They agree with the scalar quantizers, and that is measured
//!
//! The NEON path rounds with `fcvtns` and the scalar path with `nearest_int`,
//! two different routines for round-half-to-even. They produce byte-identical
//! output for all three types on all five golden patterns -- checked, because
//! ggml relies on it: a `GGML_CPU_GENERIC` build uses the scalar ones and has
//! to load the same models.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const blocks = @import("../../../quants/blocks.zig");
const neon = @import("neon.zig");
const c = impl.c;

const f32x4 = neon.f32x4;

/// The reference quantizer `q8_K` delegates to, ported in `src/ggml/quants/`.
extern fn quantize_row_q8_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;

/// The absolute maximum of 32 floats, reduced as the C reduces it.
///
/// The C builds an eight-entry array of vectors and folds it in three rounds:
/// pairs, then pairs of pairs, then the two halves. `@reduce(.Max, ...)` over a
/// flat 32 would give the same answer -- maximum is associative -- but the
/// shape is kept because it is what the two callers share, and because a
/// reduction that *looks* like the C's is easier to check against it.
///
/// Parameters:
/// - `srcv`: the eight loaded vectors.
///
/// Return: the largest absolute value across all 32 elements.
fn absMax(srcv: *const [8]f32x4) f32 {
    var asrcv: [8]f32x4 = undefined;
    for (0..8) |j| asrcv[j] = neon.abs_f32(srcv[j]);

    var amaxv: [8]f32x4 = undefined;
    for (0..4) |j| amaxv[2 * j] = neon.max_f32(asrcv[2 * j], asrcv[2 * j + 1]);
    for (0..2) |j| amaxv[4 * j] = neon.max_f32(amaxv[4 * j], amaxv[4 * j + 2]);
    amaxv[0] = neon.max_f32(amaxv[0], amaxv[4]);

    return neon.maxvq_f32(amaxv[0]);
}

/// Ports `quantize_row_q8_0` (arch/arm/quants.c:41 @c1d0e7a00).
///
/// Parameters:
/// - `x`: the source row.
/// - `vy`: destination, `k / 32` blocks.
/// - `k`: element count; must be a multiple of `QK8_0`.
pub export fn quantize_row_q8_0(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    comptime std.debug.assert(c.QK8_0 == 32);
    std.debug.assert(@rem(k, c.QK8_0) == 0);
    const nb: usize = @intCast(@divTrunc(k, c.QK8_0));

    const y: [*]blocks.Q8_0 = @ptrCast(@alignCast(vy.?));

    for (0..nb) |i| {
        var srcv: [8]f32x4 = undefined;
        for (0..8) |j| srcv[j] = neon.loadFrom(f32x4, x + i * 32 + 4 * j);

        const amax = absMax(&srcv);

        const d = amax / ((1 << 7) - 1);
        const id = if (d != 0) 1.0 / d else 0.0;

        y[i].d = convert.cpuFp32ToFp16(d);

        for (0..8) |j| {
            const v = neon.mul_n_f32(srcv[j], id);
            const vi = neon.cvtnq_s32_f32(v);

            // The C reads the four lanes out one at a time rather than
            // narrowing and storing, and the narrowing is a truncation to
            // `int8_t` either way.
            inline for (0..4) |l| {
                y[i].qs[4 * j + l] = @truncate(neon.lane(vi, l));
            }
        }
    }
}

/// Ports `quantize_row_q8_1` (arch/arm/quants.c:85 @c1d0e7a00).
///
/// Identical to `q8_0` but for the per-block sum in `s`, which the `q4_1` and
/// `q5_1` dot products need to apply their minimum without a second pass.
///
/// Parameters:
/// - `x`: the source row.
/// - `vy`: destination, `k / 32` blocks.
/// - `k`: element count; must be a multiple of `QK8_1`.
pub export fn quantize_row_q8_1(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    std.debug.assert(@rem(k, c.QK8_1) == 0);
    const nb: usize = @intCast(@divTrunc(k, c.QK8_1));

    const y: [*]blocks.Q8_1 = @ptrCast(@alignCast(vy.?));

    for (0..nb) |i| {
        var srcv: [8]f32x4 = undefined;
        for (0..8) |j| srcv[j] = neon.loadFrom(f32x4, x + i * 32 + 4 * j);

        const amax = absMax(&srcv);

        const d = amax / ((1 << 7) - 1);
        const id = if (d != 0) 1.0 / d else 0.0;

        y[i].d = convert.cpuFp32ToFp16(d);

        var accv = neon.dup(neon.i32x4, 0);

        for (0..8) |j| {
            const v = neon.mul_n_f32(srcv[j], id);
            const vi = neon.cvtnq_s32_f32(v);

            inline for (0..4) |l| {
                y[i].qs[4 * j + l] = @truncate(neon.lane(vi, l));
            }

            // Accumulated as `i32` lanes and reduced once at the end, so the
            // sum is exact rather than eight rounded partial sums.
            accv = neon.add(accv, vi);
        }

        y[i].s = convert.cpuFp32ToFp16(d * @as(f32, @floatFromInt(neon.addvq_s32(accv))));
    }
}

/// Ports `quantize_row_q8_K` (arch/arm/quants.c:134 @c1d0e7a00).
///
/// The C's comment calls this a "placeholder implementation for Apple
/// targets": there is no NEON version, so the symbol exists only to occupy the
/// name that `arch-fallback.h` would otherwise alias to the generic one.
pub export fn quantize_row_q8_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    quantize_row_q8_K_ref(x, y, k);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = @import("../testing.zig");
const golden = @import("golden.zig");

test "q8_0 row quantizer matches the C" {
    try testing.checkRow(quantize_row_q8_0, c.GGML_TYPE_Q8_0, golden.row_q8_0);
}

test "q8_1 row quantizer matches the C" {
    try testing.checkRow(quantize_row_q8_1, c.GGML_TYPE_Q8_1, golden.row_q8_1);
}

test "q8_K row quantizer matches the C" {
    try testing.checkRow(quantize_row_q8_K, c.GGML_TYPE_Q8_K, golden.row_q8_K);
}

test "the NEON and scalar q8 quantizers agree byte for byte" {
    // ggml relies on this: a GGML_CPU_GENERIC build uses the scalar ones and
    // must load the same models. The two round by different routines --
    // `fcvtns` here, `nearest_int` there -- so it is worth asserting rather
    // than assuming.
    const generic = @import("../rows.zig");

    var x: [512]f32 = undefined;
    var seed: u32 = 7;
    for (&x) |*v| {
        seed = 1103515245 *% seed +% 12345;
        v.* = (@as(f32, @floatFromInt(seed >> 16)) / 32768.0) - 1.0;
    }

    var a: [2048]u8 align(8) = @splat(0);
    var b: [2048]u8 align(8) = @splat(0);

    quantize_row_q8_0(&x, &a, 512);
    generic.quantize_row_q8_0_generic(&x, &b, 512);
    try std.testing.expectEqualSlices(u8, &b, &a);

    @memset(&a, 0);
    @memset(&b, 0);
    quantize_row_q8_1(&x, &a, 512);
    generic.quantize_row_q8_1_generic(&x, &b, 512);
    try std.testing.expectEqualSlices(u8, &b, &a);
}

test "an all-zero row takes the divide-by-zero guard" {
    // `amax` is 0, so `id` must be 0 rather than an infinity; the C spells
    // this as `d ? 1.0f/d : 0.0f`.
    var x: [32]f32 = @splat(0.0);
    var out: [64]u8 align(8) = @splat(0xAA);

    quantize_row_q8_0(&x, &out, 32);

    const blk: *const blocks.Q8_0 = @ptrCast(&out);
    try std.testing.expectEqual(@as(u16, 0), blk.d);
    for (blk.qs) |q| try std.testing.expectEqual(@as(i8, 0), q);
}
