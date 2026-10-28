//! The AVX2 row quantizers: `q8_0`, `q8_1` and `q8_K`, the activation
//! formats the dot products take.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v0.4.0,
//! `5266f24da`). Each function names the C it replaces and the line it
//! began at.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const blocks = @import("../../../quants/blocks.zig");
const avx = @import("avx2.zig");
const common = @import("common.zig");
const c = impl.c;

/// The shared body of `quantize_row_q8_0` and `quantize_row_q8_1`
/// (arch/x86/quants.c:302, 400 @5266f24da), parameterised on the block
/// type: the second also stores the sum.
///
/// The amax is reduced with `maxps`, whose answer on equal or NaN lanes is
/// its second operand; `_mm256_round_ps(.., _MM_ROUND_NEAREST)` rounds ties
/// to even, where the reference quantizer rounds them away; and the two
/// packs saturate and interleave 128-bit halves, which `permutevar8x32`
/// undoes.
inline fn quantizeQ8(comptime Block: type, x_in: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    std.debug.assert(@rem(k, 32) == 0);
    const nb: usize = @intCast(@divTrunc(k, 32));

    const y: [*]Block = @ptrCast(@alignCast(vy.?));
    var x: [*]const f32 = x_in;

    for (0..nb) |i| {
        // Load elements into 4 AVX vectors
        var v0 = avx.mm256_loadu_ps(x);
        var v1 = avx.mm256_loadu_ps(x + 8);
        var v2 = avx.mm256_loadu_ps(x + 16);
        var v3 = avx.mm256_loadu_ps(x + 24);
        x += 32;

        // Compute max(abs(e)) for the block
        const signBit = avx.mm256_set1_ps(-0.0);
        var maxAbs = avx.mm256_andnot_ps(signBit, v0);
        maxAbs = avx.mm256_max_ps(maxAbs, avx.mm256_andnot_ps(signBit, v1));
        maxAbs = avx.mm256_max_ps(maxAbs, avx.mm256_andnot_ps(signBit, v2));
        maxAbs = avx.mm256_max_ps(maxAbs, avx.mm256_andnot_ps(signBit, v3));

        var max4 = avx.mm_max_ps(avx.mm256_extractf128_ps(maxAbs, 1), avx.mm256_castps256_ps128(maxAbs));
        max4 = avx.mm_max_ps(max4, avx.mm_movehl_ps(max4, max4));
        max4 = avx.mm_max_ss(max4, avx.mm_movehdup_ps(max4));
        const max_scalar = avx.mm_cvtss_f32(max4);

        // Quantize these floats
        const d = max_scalar / 127.0;
        y[i].d = convert.cpuFp32ToFp16(d);
        const id: f32 = if (max_scalar != 0.0) 127.0 / max_scalar else 0.0;
        const mul = avx.mm256_set1_ps(id);

        // Apply the multiplier
        v0 = avx.mm256_mul_ps(v0, mul);
        v1 = avx.mm256_mul_ps(v1, mul);
        v2 = avx.mm256_mul_ps(v2, mul);
        v3 = avx.mm256_mul_ps(v3, mul);

        // Round to nearest integer
        v0 = avx.mm256_round_ps(v0, avx.MM_ROUND_NEAREST);
        v1 = avx.mm256_round_ps(v1, avx.MM_ROUND_NEAREST);
        v2 = avx.mm256_round_ps(v2, avx.MM_ROUND_NEAREST);
        v3 = avx.mm256_round_ps(v3, avx.MM_ROUND_NEAREST);

        // Convert floats to integers. The C's `i0`..`i3`, renamed digit for
        // digit: Zig reserves them as integer types.
        var j0 = avx.mm256_cvtps_epi32(v0);
        const j1 = avx.mm256_cvtps_epi32(v1);
        var j2 = avx.mm256_cvtps_epi32(v2);
        const j3 = avx.mm256_cvtps_epi32(v3);

        if (Block == blocks.Q8_1) {
            // Compute the sum of the quants and set y[i].s
            y[i].s = convert.cpuFp32ToFp16(d * @as(f32, @floatFromInt(common.hsum_i32_8(
                avx.mm256_add_epi32(avx.mm256_add_epi32(j0, j1), avx.mm256_add_epi32(j2, j3)),
            ))));
        }

        // Convert int32 to int16
        j0 = avx.mm256_packs_epi32(j0, j1);
        j2 = avx.mm256_packs_epi32(j2, j3);
        // Convert int16 to int8
        j0 = avx.mm256_packs_epi16(j0, j2);

        // The packs work on 128-bit halves independently; this restores the
        // order.
        const perm = avx.mm256_setr_epi32(.{ 0, 4, 1, 5, 2, 6, 3, 7 });
        j0 = avx.mm256_permutevar8x32_epi32(j0, perm);

        avx.mm256_storeu_si256(&y[i].qs, j0);
    }
}

/// Ports `quantize_row_q8_0` (arch/x86/quants.c:302 @5266f24da), the
/// `__AVX2__ || __AVX__` arm with its `__AVX2__` pack.
pub export fn quantize_row_q8_0(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    quantizeQ8(blocks.Q8_0, x, vy, k);
}

/// Ports `quantize_row_q8_1` (arch/x86/quants.c:400 @5266f24da), the
/// `__AVX2__ || __AVX__` arm with its `__AVX2__` pack and sum.
pub export fn quantize_row_q8_1(x: [*c]const f32, vy: ?*anyopaque, k: i64) void {
    quantizeQ8(blocks.Q8_1, x, vy, k);
}

const ref = struct {
    extern fn quantize_row_q8_K_ref(x: [*c]const f32, y: ?*anyopaque, k: i64) void;
};

/// Ports `quantize_row_q8_K` (arch/x86/quants.c:505 @5266f24da): on x86 it
/// has no vector arm, and calls `ggml-quants.c`'s reference quantizer as it
/// stands.
pub export fn quantize_row_q8_K(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
    ref.quantize_row_q8_K_ref(x, y, k);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
