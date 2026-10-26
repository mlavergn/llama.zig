//! The static helpers `arch/x86/repack.cpp` shares between its kernels.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` (v0.3.0,
//! `c1d0e7a00`): the `__AVX2__` arms of the helpers at the top of the file.
//! Each names the C++ it replaces and the line it began at. x86-64-v3 has
//! F16C, so the `GGML_F32Cx8` loads are the `_mm256_cvtph_ps` macros, not the
//! table-driven `__avx_f32cx8_load` family below them, which it does not
//! compile; and it has no VNNI, which `tigerlake` adds.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const avx = @import("../../quants/x86/avx2.zig");

const c = impl.c;
const m256i = avx.m256i;
const m128i = avx.m128i;
const m256 = avx.m256;

/// `GGML_CPU_FP16_TO_FP32`: the table lookup, as the reference compiles it.
pub inline fn f(h: c.ggml_half) f32 {
    return convert.cpuFp16ToFp32(h);
}

/// `_mm256_cvtph_ps`: eight halves widened by F16C's `vcvtph2ps`, exact for
/// every value but a signalling NaN, which it quiets.
inline fn cvtph(h: [8]c.ggml_half) m256 {
    const v: @Vector(8, f16) = @bitCast(h);
    return @floatCast(v);
}

/// Ports `GGML_F32Cx8_LOAD` (arch/x86/repack.cpp:34 @c1d0e7a00), the
/// `__F16C__` arm x86-64-v3 compiles.
pub inline fn f32cx8Load(x: *const [8]c.ggml_half) m256 {
    return cvtph(x.*);
}

/// Ports `GGML_F32Cx8_REPEAT_LOAD` (arch/x86/repack.cpp:35 @c1d0e7a00), the
/// `__F16C__` arm: a masked load of four halves, repeated by
/// `_mm_shuffle_epi32(.., 68)` into both 64-bit halves, then widened.
pub inline fn f32cx8RepeatLoad(x: *const [4]c.ggml_half) m256 {
    return cvtph(x.* ++ x.*);
}

/// Ports `GGML_F32Cx8_REARRANGE_LOAD` (arch/x86/repack.cpp:36 @c1d0e7a00),
/// the `__F16C__` arm: the eight halves reordered by a byte shuffle, then
/// widened.
pub inline fn f32cx8RearrangeLoad(x: *const [8]c.ggml_half, arrangeMask: m128i) m256 {
    return cvtph(avx.as([8]u16, avx.mm_shuffle_epi8(avx.mm_loadu_si128(x), arrangeMask)));
}

/// Ports `sum_i16_pairs_acc_int32x8` (arch/x86/repack.cpp:146 @c1d0e7a00).
pub inline fn sum_i16_pairs_acc_int32x8(acc: m256i, x: m256i) m256i {
    const ones = avx.mm256_set1_epi16(1);
    return avx.mm256_add_epi32(acc, avx.mm256_madd_epi16(ones, x));
}

/// Ports `mul_sum_us8_pairs_acc_int32x8` (arch/x86/repack.cpp:151 @c1d0e7a00):
/// the `__AVX512VNNI__ && __AVX512VL__` arm where the target has it, the
/// plain arm otherwise -- no VNNI at x86-64-v3 or v4.
pub inline fn mul_sum_us8_pairs_acc_int32x8(acc: m256i, ax: m256i, sy: m256i) m256i {
    if (comptime avx.has_avx512vnni_vl) return avx.mm256_dpbusd_epi32(acc, ax, sy);
    if (comptime avx.has_avxvnni_only) @compileError("the __AVXVNNI__ arm of mul_sum_us8_pairs_acc_int32x8 is not ported");
    const dot = avx.mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x8(acc, dot);
}

/// Ports `mul_sum_i8_pairs_acc_int32x8` (arch/x86/repack.cpp:165 @c1d0e7a00),
/// the arm without `__AVXVNNIINT8__`.
pub inline fn mul_sum_i8_pairs_acc_int32x8(acc: m256i, x: m256i, y: m256i) m256i {
    if (comptime avx.has_avxvnniint8) @compileError("the __AVXVNNIINT8__ arm of mul_sum_i8_pairs_acc_int32x8 is not ported");
    const ax = avx.mm256_sign_epi8(x, x);
    const sy = avx.mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
}

/// A 128-bit load repeated into both halves: the C's `_mm256_castsi128_si256`
/// then `_mm256_permute2f128_si256(v, v, 0)`.
pub inline fn repeat128(v: m128i) m256i {
    const w = avx.mm256_castsi128_si256(v);
    return avx.mm256_permute2f128_si256(w, w, 0);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
