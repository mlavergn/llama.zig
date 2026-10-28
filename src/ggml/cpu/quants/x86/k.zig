//! AVX2 dot products for the K-quant super-block formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v0.4.0,
//! `5266f24da`), the `__AVX2__` arms of the K-quant dot products. Each
//! function names the C it replaces and the line it began at; the `__AVX__`
//! and scalar arms below each are not compiled at x86-64-v3.
//!
//! # The integer work is exact; the float epilogue is not
//!
//! Every product and sum before `_mm256_cvtepi32_ps` is integer, wrapping
//! or saturating exactly as the instruction does -- `avx2.zig` reaches those
//! instructions. What can move a bit is the float tail, so each kernel's
//! fusions were read from the reference's disassembly, not from the C:
//! `_mm256_fmadd_ps` is explicit, and clang contracts the scalar
//! `summs += dmin * x` into one `vfnmadd231ss`.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../../../quants/blocks.zig");
const avx = @import("avx2.zig");
const common = @import("common.zig");
const c = impl.c;

const f = common.f;
const as = common.as;

/// Ports `ggml_vec_dot_q2_K_q8_K` (arch/x86/quants.c:1574 @5266f24da), the
/// `__AVX2__` arm.
///
/// The mins go straight into the float accumulator, eight lanes fused with
/// `dmin` before any weight is read; the weights follow, fused with `d`. Two
/// `fmadd`s into one `acc` per super-block, in that order. The C asserts
/// only `nrc == 1` here, not `n % QK_K == 0`.
pub export fn ggml_vec_dot_q2_K_q8_K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q2_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m3 = avx.mm256_set1_epi8(3);
    const m4 = avx.mm_set1_epi8(0xF);

    var acc = avx.mm256_setzero_ps();

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);
        const dmin = -y[i].d * f(x[i].dmin);

        var q2: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        const mins_and_scales = avx.mm_loadu_si128(&x[i].scales);
        const scales8 = avx.mm_and_si128(mins_and_scales, m4);
        const mins8 = avx.mm_and_si128(avx.mm_srli_epi16(mins_and_scales, 4), m4);
        const mins = avx.mm256_cvtepi8_epi16(mins8);
        const prod = avx.mm256_madd_epi16(mins, avx.mm256_loadu_si256(&y[i].bsums));

        acc = avx.mm256_fmadd_ps(avx.mm256_broadcast_ss(&dmin), avx.mm256_cvtepi32_ps(prod), acc);

        const all_scales = avx.mm256_cvtepi8_epi16(scales8);
        const l_scales = avx.mm256_extracti128_si256(all_scales, 0);
        const h_scales = avx.mm256_extracti128_si256(all_scales, 1);
        const scales = [2]avx.m256i{ avx.mm256_set_m128i(l_scales, l_scales), avx.mm256_set_m128i(h_scales, h_scales) };

        var sumi = avx.mm256_setzero_si256();

        for (0..c.QK_K / 128) |j| {
            const q2bits = avx.mm256_loadu_si256(q2);
            q2 += 32;

            const q8_0 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_3 = avx.mm256_loadu_si256(q8);
            q8 += 32;

            const q2_0 = avx.mm256_and_si256(q2bits, m3);
            const q2_1 = avx.mm256_and_si256(avx.mm256_srli_epi16(q2bits, 2), m3);
            const q2_2 = avx.mm256_and_si256(avx.mm256_srli_epi16(q2bits, 4), m3);
            const q2_3 = avx.mm256_and_si256(avx.mm256_srli_epi16(q2bits, 6), m3);

            var p0 = avx.mm256_maddubs_epi16(q2_0, q8_0);
            var p1 = avx.mm256_maddubs_epi16(q2_1, q8_1);
            var p2 = avx.mm256_maddubs_epi16(q2_2, q8_2);
            var p3 = avx.mm256_maddubs_epi16(q2_3, q8_3);

            p0 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(0)), p0);
            p1 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(1)), p1);
            p2 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(2)), p2);
            p3 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(3)), p3);

            p0 = avx.mm256_add_epi32(p0, p1);
            p2 = avx.mm256_add_epi32(p2, p3);

            sumi = avx.mm256_add_epi32(sumi, avx.mm256_add_epi32(p0, p2));
        }

        acc = avx.mm256_fmadd_ps(avx.mm256_broadcast_ss(&d), avx.mm256_cvtepi32_ps(sumi), acc);
    }

    s[0] = common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_q3_K_q8_K` (arch/x86/quants.c:1766 @5266f24da), the
/// `__AVX2__` arm.
///
/// Two low bits from `qs` and the third from `hmask`, inverted: the high
/// part is multiplied separately and subtracted, so it is 4 where the bit is
/// clear and 0 where it is set. **`is` is never advanced** in this arm --
/// every 128-weight half takes shuffles 0 to 3 of its own `scales[j]` -- and
/// the port keeps it that way, since it is what the C computes.
pub export fn ggml_vec_dot_q3_K_q8_K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;

    const x = as(blocks.Q3_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m3 = avx.mm256_set1_epi8(3);
    const mone = avx.mm256_set1_epi8(1);
    const m32 = avx.mm_set1_epi8(32);

    var acc = avx.mm256_setzero_ps();

    var aux: [3]u32 = undefined;

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);

        var q3: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        // Set up scales
        @memcpy(std.mem.asBytes(&aux), &x[i].scales);
        var scales128 = avx.mm_set_epi32(
            @bitCast(((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)),
            @bitCast(((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)),
            @bitCast((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)),
            @bitCast((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)),
        );
        scales128 = avx.mm_sub_epi8(scales128, m32);
        const all_scales = avx.mm256_cvtepi8_epi16(scales128);
        const l_scales = avx.mm256_extracti128_si256(all_scales, 0);
        const h_scales = avx.mm256_extracti128_si256(all_scales, 1);
        const scales = [2]avx.m256i{ avx.mm256_set_m128i(l_scales, l_scales), avx.mm256_set_m128i(h_scales, h_scales) };

        // high bit
        const hbits = avx.mm256_loadu_si256(&x[i].hmask);

        // integer accumulator
        var sumi = avx.mm256_setzero_si256();

        var bit: u32 = 0;
        const is: usize = 0;

        for (0..c.QK_K / 128) |j| {
            // load low 2 bits
            const q3bits = avx.mm256_loadu_si256(q3);
            q3 += 32;

            // prepare low and high bits
            const q3l_0 = avx.mm256_and_si256(q3bits, m3);
            const q3h_0 = avx.mm256_slli_epi16(avx.mm256_srli_epi16(avx.mm256_andnot_si256(hbits, avx.mm256_slli_epi16(mone, bit)), bit), 2);
            bit += 1;

            const q3l_1 = avx.mm256_and_si256(avx.mm256_srli_epi16(q3bits, 2), m3);
            const q3h_1 = avx.mm256_slli_epi16(avx.mm256_srli_epi16(avx.mm256_andnot_si256(hbits, avx.mm256_slli_epi16(mone, bit)), bit), 2);
            bit += 1;

            const q3l_2 = avx.mm256_and_si256(avx.mm256_srli_epi16(q3bits, 4), m3);
            const q3h_2 = avx.mm256_slli_epi16(avx.mm256_srli_epi16(avx.mm256_andnot_si256(hbits, avx.mm256_slli_epi16(mone, bit)), bit), 2);
            bit += 1;

            const q3l_3 = avx.mm256_and_si256(avx.mm256_srli_epi16(q3bits, 6), m3);
            const q3h_3 = avx.mm256_slli_epi16(avx.mm256_srli_epi16(avx.mm256_andnot_si256(hbits, avx.mm256_slli_epi16(mone, bit)), bit), 2);
            bit += 1;

            // load Q8 quants
            const q8_0 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_3 = avx.mm256_loadu_si256(q8);
            q8 += 32;

            // The low two bits and the high one are multiplied separately so
            // `maddubs` can take both, and the high part subtracted.
            const q8s_0 = avx.mm256_maddubs_epi16(q3h_0, q8_0);
            const q8s_1 = avx.mm256_maddubs_epi16(q3h_1, q8_1);
            const q8s_2 = avx.mm256_maddubs_epi16(q3h_2, q8_2);
            const q8s_3 = avx.mm256_maddubs_epi16(q3h_3, q8_3);

            var p16_0 = avx.mm256_maddubs_epi16(q3l_0, q8_0);
            var p16_1 = avx.mm256_maddubs_epi16(q3l_1, q8_1);
            var p16_2 = avx.mm256_maddubs_epi16(q3l_2, q8_2);
            var p16_3 = avx.mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = avx.mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = avx.mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = avx.mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = avx.mm256_sub_epi16(p16_3, q8s_3);

            // multiply with scales
            p16_0 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = avx.mm256_madd_epi16(avx.mm256_shuffle_epi8(scales[j], common.get_scale_shuffle_q3k(is + 3)), p16_3);

            // accumulate
            p16_0 = avx.mm256_add_epi32(p16_0, p16_1);
            p16_2 = avx.mm256_add_epi32(p16_2, p16_3);
            sumi = avx.mm256_add_epi32(sumi, avx.mm256_add_epi32(p16_0, p16_2));
        }

        // multiply with block scale and accumulate
        acc = avx.mm256_fmadd_ps(avx.mm256_broadcast_ss(&d), avx.mm256_cvtepi32_ps(sumi), acc);
    }

    s[0] = common.hsum_float_8(acc);
}

/// Ports `ggml_vec_dot_q4_K_q8_K` (arch/x86/quants.c:2038 @5266f24da), the
/// `__AVX2__` arm.
///
/// `q5_K` without the fifth bit, and with the mins kept as four lanes,
/// `acc_m`, folded once at the end -- `(m0 + m2) + (m1 + m3)` -- where
/// `q5_K` folds each super-block's to a scalar first.
pub export fn ggml_vec_dot_q4_K_q8_K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m4 = avx.mm256_set1_epi8(0xF);

    var acc = avx.mm256_setzero_ps();
    var acc_m = avx.mm_setzero_ps();

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);
        const dmin = -y[i].d * f(x[i].dmin);

        const utmp = common.unpackScales(&x[i].scales);

        var q4: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        const mins_and_scales = avx.mm256_cvtepu8_epi16(avx.mm_set_epi32(
            @bitCast(utmp[3]),
            @bitCast(utmp[2]),
            @bitCast(utmp[1]),
            @bitCast(utmp[0]),
        ));

        const q8sums = avx.mm256_loadu_si256(&y[i].bsums);
        const q8s = avx.mm_hadd_epi16(avx.mm256_extracti128_si256(q8sums, 0), avx.mm256_extracti128_si256(q8sums, 1));
        const prod = avx.mm_madd_epi16(avx.mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = avx.mm_fmadd_ps(avx.mm_set1_ps(dmin), avx.mm_cvtepi32_ps(prod), acc_m);

        const sc128 = avx.mm256_extracti128_si256(mins_and_scales, 0);
        const scales = avx.mm256_set_m128i(sc128, sc128);

        var sumi = avx.mm256_setzero_si256();

        for (0..c.QK_K / 64) |j| {
            const scale_l = avx.mm256_shuffle_epi8(scales, common.get_scale_shuffle_k4(2 * j + 0));
            const scale_h = avx.mm256_shuffle_epi8(scales, common.get_scale_shuffle_k4(2 * j + 1));

            const q4bits = avx.mm256_loadu_si256(q4);
            q4 += 32;
            const q4l = avx.mm256_and_si256(q4bits, m4);
            const q4h = avx.mm256_and_si256(avx.mm256_srli_epi16(q4bits, 4), m4);

            const q8l = avx.mm256_loadu_si256(q8);
            q8 += 32;
            var p16l = avx.mm256_maddubs_epi16(q4l, q8l);
            p16l = avx.mm256_madd_epi16(scale_l, p16l);

            const q8h = avx.mm256_loadu_si256(q8);
            q8 += 32;
            var p16h = avx.mm256_maddubs_epi16(q4h, q8h);
            p16h = avx.mm256_madd_epi16(scale_h, p16h);
            const sumj = avx.mm256_add_epi32(p16l, p16h);

            sumi = avx.mm256_add_epi32(sumi, sumj);
        }

        const vd = avx.mm256_set1_ps(d);
        acc = avx.mm256_fmadd_ps(vd, avx.mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = avx.mm_add_ps(acc_m, avx.mm_movehl_ps(acc_m, acc_m));
    acc_m = avx.mm_add_ss(acc_m, avx.mm_movehdup_ps(acc_m));

    s[0] = common.hsum_float_8(acc) + avx.mm_cvtss_f32(acc_m);
}

/// Ports `ggml_vec_dot_q5_K_q8_K` (arch/x86/quants.c:2216 @5266f24da), the
/// `__AVX2__` arm.
///
/// The mins are applied first, against `q8_K`'s `bsums`, through two
/// `phadd`s into one scalar per super-block; then four 32-weight groups per
/// half, the fifth bit taken from `qh` one bit plane at a time.
///
/// `summs += dmin * hsum` is one `vfnmadd231ss` in the reference: `dmin` is
/// negated, so `-(y.d * x.dmin) * hsum + summs`, which `@mulAdd` with the
/// negated factor computes to the same bits.
pub export fn ggml_vec_dot_q5_K_q8_K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m4 = avx.mm256_set1_epi8(0xF);
    const mzero = avx.mm_setzero_si128();
    const mone = avx.mm256_set1_epi8(1);

    var acc = avx.mm256_setzero_ps();

    var summs: f32 = 0;

    for (0..nb) |i| {
        var q5: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        const d = y[i].d * f(x[i].d);
        const dmin = -y[i].d * f(x[i].dmin);

        const utmp = common.unpackScales(&x[i].scales);

        const mins_and_scales = avx.mm256_cvtepu8_epi16(avx.mm_set_epi32(
            @bitCast(utmp[3]),
            @bitCast(utmp[2]),
            @bitCast(utmp[1]),
            @bitCast(utmp[0]),
        ));

        const q8sums = avx.mm256_loadu_si256(&y[i].bsums);
        const q8s = avx.mm_hadd_epi16(avx.mm256_extracti128_si256(q8sums, 0), avx.mm256_extracti128_si256(q8sums, 1));
        const prod = avx.mm_madd_epi16(avx.mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const hsum = avx.mm_hadd_epi32(avx.mm_hadd_epi32(prod, mzero), mzero);
        summs = @mulAdd(f32, dmin, @floatFromInt(avx.mm_extract_epi32(hsum, 0)), summs);

        const sc128 = avx.mm256_extracti128_si256(mins_and_scales, 0);
        const scales = avx.mm256_set_m128i(sc128, sc128);

        const hbits = avx.mm256_loadu_si256(&x[i].qh);
        var hmask = mone;

        var sumi = avx.mm256_setzero_si256();

        var bit: u32 = 0;

        for (0..c.QK_K / 64) |j| {
            const scale_0 = avx.mm256_shuffle_epi8(scales, common.get_scale_shuffle_k4(2 * j + 0));
            const scale_1 = avx.mm256_shuffle_epi8(scales, common.get_scale_shuffle_k4(2 * j + 1));

            const q5bits = avx.mm256_loadu_si256(q5);
            q5 += 32;

            const q5l_0 = avx.mm256_and_si256(q5bits, m4);
            const q5h_0 = avx.mm256_slli_epi16(avx.mm256_srli_epi16(avx.mm256_and_si256(hbits, hmask), bit), 4);
            bit += 1;
            const q5_0 = avx.mm256_add_epi8(q5l_0, q5h_0);
            hmask = avx.mm256_slli_epi16(hmask, 1);

            const q5l_1 = avx.mm256_and_si256(avx.mm256_srli_epi16(q5bits, 4), m4);
            const q5h_1 = avx.mm256_slli_epi16(avx.mm256_srli_epi16(avx.mm256_and_si256(hbits, hmask), bit), 4);
            bit += 1;
            const q5_1 = avx.mm256_add_epi8(q5l_1, q5h_1);
            hmask = avx.mm256_slli_epi16(hmask, 1);

            const q8_0 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;

            var p16_0 = avx.mm256_maddubs_epi16(q5_0, q8_0);
            var p16_1 = avx.mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = avx.mm256_madd_epi16(scale_0, p16_0);
            p16_1 = avx.mm256_madd_epi16(scale_1, p16_1);

            sumi = avx.mm256_add_epi32(sumi, avx.mm256_add_epi32(p16_0, p16_1));
        }

        const vd = avx.mm256_set1_ps(d);
        acc = avx.mm256_fmadd_ps(vd, avx.mm256_cvtepi32_ps(sumi), acc);
    }

    s[0] = common.hsum_float_8(acc) + summs;
}

/// Ports `ggml_vec_dot_q6_K_q8_K` (arch/x86/quants.c:2426 @5266f24da), the
/// `__AVX2__` arm.
///
/// Six-bit weights, four in `ql` and two in `qh`, with signed `int8`
/// scales. The bias of 32 comes off once per super-block, as `bsums`
/// against the scales shifted left by five, subtracted in integers before
/// the one fused multiply-add.
pub export fn ggml_vec_dot_q6_K_q8_K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q6_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m3 = avx.mm256_set1_epi8(3);
    const m15 = avx.mm256_set1_epi8(15);

    var acc = avx.mm256_setzero_ps();

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);

        var q4: [*]const u8 = &x[i].ql;
        var qh: [*]const u8 = &x[i].qh;
        var q8: [*]const i8 = &y[i].qs;

        const q8sums = avx.mm256_loadu_si256(&y[i].bsums);
        const scales = avx.mm_loadu_si128(&x[i].scales);
        const scales_16 = avx.mm256_cvtepi8_epi16(scales);
        const q8sclsub = avx.mm256_slli_epi32(avx.mm256_madd_epi16(q8sums, scales_16), 5);

        var sumi = avx.mm256_setzero_si256();

        var is: usize = 0;

        for (0..c.QK_K / 128) |_| {
            const q4bits1 = avx.mm256_loadu_si256(q4);
            q4 += 32;
            const q4bits2 = avx.mm256_loadu_si256(q4);
            q4 += 32;
            const q4bitsH = avx.mm256_loadu_si256(qh);
            qh += 32;

            const q4h_0 = avx.mm256_slli_epi16(avx.mm256_and_si256(q4bitsH, m3), 4);
            const q4h_1 = avx.mm256_slli_epi16(avx.mm256_and_si256(q4bitsH, avx.mm256_set1_epi8(12)), 2);
            const q4h_2 = avx.mm256_and_si256(q4bitsH, avx.mm256_set1_epi8(48));
            const q4h_3 = avx.mm256_srli_epi16(avx.mm256_and_si256(q4bitsH, avx.mm256_set1_epi8(-64)), 2);

            const q4_0 = avx.mm256_or_si256(avx.mm256_and_si256(q4bits1, m15), q4h_0);
            const q4_1 = avx.mm256_or_si256(avx.mm256_and_si256(q4bits2, m15), q4h_1);
            const q4_2 = avx.mm256_or_si256(avx.mm256_and_si256(avx.mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const q4_3 = avx.mm256_or_si256(avx.mm256_and_si256(avx.mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const q8_0 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_3 = avx.mm256_loadu_si256(q8);
            q8 += 32;

            var p16_0 = avx.mm256_maddubs_epi16(q4_0, q8_0);
            var p16_1 = avx.mm256_maddubs_epi16(q4_1, q8_1);
            var p16_2 = avx.mm256_maddubs_epi16(q4_2, q8_2);
            var p16_3 = avx.mm256_maddubs_epi16(q4_3, q8_3);

            const scale_0 = avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(is + 0));
            const scale_1 = avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(is + 1));
            const scale_2 = avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(is + 2));
            const scale_3 = avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = avx.mm256_madd_epi16(avx.mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = avx.mm256_madd_epi16(avx.mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = avx.mm256_madd_epi16(avx.mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = avx.mm256_madd_epi16(avx.mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = avx.mm256_add_epi32(sumi, avx.mm256_add_epi32(p16_0, p16_1));
            sumi = avx.mm256_add_epi32(sumi, avx.mm256_add_epi32(p16_2, p16_3));
        }

        sumi = avx.mm256_sub_epi32(sumi, q8sclsub);
        acc = avx.mm256_fmadd_ps(avx.mm256_broadcast_ss(&d), avx.mm256_cvtepi32_ps(sumi), acc);
    }

    s[0] = common.hsum_float_8(acc);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
