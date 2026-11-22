//! The static helpers `arch/x86/repack.cpp` shares between its kernels.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` (v1.0.0,
//! `b1b770d90`): the `__AVX2__` arms of the helpers at the top of the file.
//! Each names the C++ it replaces and the line it began at. x86-64-v3 has
//! F16C, so the `GGML_F32Cx8` loads are the `_mm256_cvtph_ps` macros, not the
//! table-driven `__avx_f32cx8_load` family below them, which it does not
//! compile; and it has no VNNI, which `tigerlake` adds.

const std = @import("std");
const ggml = @import("../../../module.zig");

/// `GGML_CPU_FP16_TO_FP32`: the table lookup, as the reference compiles it.
pub inline fn f(h: ggml.impl.c.ggml_half) f32 {
    return ggml.cpu.convert.cpuFp16ToFp32(h);
}

/// `_mm256_cvtph_ps`: eight halves widened by F16C's `vcvtph2ps`, exact for
/// every value but a signalling NaN, which it quiets.
inline fn cvtph(h: [8]ggml.impl.c.ggml_half) ggml.cpu.quants.arch.avx2.m256 {
    const v: @Vector(8, f16) = @bitCast(h);
    return @floatCast(v);
}

/// Ports `GGML_F32Cx8_LOAD` (arch/x86/repack.cpp:34 @b1b770d90), the
/// `__F16C__` arm x86-64-v3 compiles.
pub inline fn f32cx8Load(x: *const [8]ggml.impl.c.ggml_half) ggml.cpu.quants.arch.avx2.m256 {
    return cvtph(x.*);
}

/// Ports `GGML_F32Cx8_REPEAT_LOAD` (arch/x86/repack.cpp:35 @b1b770d90), the
/// `__F16C__` arm: a masked load of four halves, repeated by
/// `_mm_shuffle_epi32(.., 68)` into both 64-bit halves, then widened.
pub inline fn f32cx8RepeatLoad(x: *const [4]ggml.impl.c.ggml_half) ggml.cpu.quants.arch.avx2.m256 {
    return cvtph(x.* ++ x.*);
}

/// Ports `GGML_F32Cx8_REARRANGE_LOAD` (arch/x86/repack.cpp:36 @b1b770d90),
/// the `__F16C__` arm: the eight halves reordered by a byte shuffle, then
/// widened.
pub inline fn f32cx8RearrangeLoad(x: *const [8]ggml.impl.c.ggml_half, arrangeMask: ggml.cpu.quants.arch.avx2.m128i) ggml.cpu.quants.arch.avx2.m256 {
    return cvtph(ggml.cpu.quants.arch.avx2.as([8]u16, ggml.cpu.quants.arch.avx2.mm_shuffle_epi8(ggml.cpu.quants.arch.avx2.mm_loadu_si128(x), arrangeMask)));
}

/// Ports `sum_i16_pairs_acc_int32x8` (arch/x86/repack.cpp:146 @b1b770d90).
pub inline fn sum_i16_pairs_acc_int32x8(acc: ggml.cpu.quants.arch.avx2.m256i, x: ggml.cpu.quants.arch.avx2.m256i) ggml.cpu.quants.arch.avx2.m256i {
    const ones = ggml.cpu.quants.arch.avx2.mm256_set1_epi16(1);
    return ggml.cpu.quants.arch.avx2.mm256_add_epi32(acc, ggml.cpu.quants.arch.avx2.mm256_madd_epi16(ones, x));
}

/// Ports `mul_sum_us8_pairs_acc_int32x8` (arch/x86/repack.cpp:151 @b1b770d90):
/// the `__AVX512VNNI__ && __AVX512VL__` arm where the target has it, the
/// plain arm otherwise -- no VNNI at x86-64-v3 or v4.
pub inline fn mul_sum_us8_pairs_acc_int32x8(acc: ggml.cpu.quants.arch.avx2.m256i, ax: ggml.cpu.quants.arch.avx2.m256i, sy: ggml.cpu.quants.arch.avx2.m256i) ggml.cpu.quants.arch.avx2.m256i {
    if (comptime ggml.cpu.quants.arch.avx2.has_avx512vnni_vl) return ggml.cpu.quants.arch.avx2.mm256_dpbusd_epi32(acc, ax, sy);
    if (comptime ggml.cpu.quants.arch.avx2.has_avxvnni_only) @compileError("the __AVXVNNI__ arm of mul_sum_us8_pairs_acc_int32x8 is not ported");
    const dot = ggml.cpu.quants.arch.avx2.mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x8(acc, dot);
}

/// Ports `mul_sum_i8_pairs_acc_int32x8` (arch/x86/repack.cpp:165 @b1b770d90),
/// the arm without `__AVXVNNIINT8__`.
pub inline fn mul_sum_i8_pairs_acc_int32x8(acc: ggml.cpu.quants.arch.avx2.m256i, x: ggml.cpu.quants.arch.avx2.m256i, y: ggml.cpu.quants.arch.avx2.m256i) ggml.cpu.quants.arch.avx2.m256i {
    if (comptime ggml.cpu.quants.arch.avx2.has_avxvnniint8) @compileError("the __AVXVNNIINT8__ arm of mul_sum_i8_pairs_acc_int32x8 is not ported");
    const ax = ggml.cpu.quants.arch.avx2.mm256_sign_epi8(x, x);
    const sy = ggml.cpu.quants.arch.avx2.mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
}

/// A 128-bit load repeated into both halves: the C's `_mm256_castsi128_si256`
/// then `_mm256_permute2f128_si256(v, v, 0)`.
pub inline fn repeat128(v: ggml.cpu.quants.arch.avx2.m128i) ggml.cpu.quants.arch.avx2.m256i {
    const w = ggml.cpu.quants.arch.avx2.mm256_castsi128_si256(v);
    return ggml.cpu.quants.arch.avx2.mm256_permute2f128_si256(w, w, 0);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
