//! AVX2 dot products for the codebook formats: the `iq1`, `iq2`, `iq3`
//! families and `iq4_xs`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v0.3.0,
//! `c1d0e7a00`), the `__AVX2__` arms. Each function names the C it replaces
//! and the line it began at.
//!
//! # The grids are gathered with `set`, not loaded
//!
//! Each weight group is an index into a codebook -- `iq2xxs_grid` and its
//! kin, in `ggml-common.h` -- and the C builds each register from four or
//! eight table entries with `_mm256_set_epi64x` or `_mm256_set_epi32`,
//! highest lane first. The entries are bytes in `[0, 255]` used as the
//! unsigned operand of `maddubs`; the signs travel separately and are applied
//! to `q8_K` instead.
//!
//! # x86-64-v3 has BMI2
//!
//! So `iq1_s` and `iq1_m` take their `__BMI2__` arms, which assemble the
//! grid indices with `pdep`. The other arms compute the same indices; these
//! are the ones the reference compiles.
//!
//! # The scalar fusions
//!
//! Read from the reference's disassembly: `iq1_s`'s `accum1 += d * sumi1`
//! and both `iq1` kernels' final `+ DELTA * x` are one `vfmadd` each; the
//! `0.125f *` and `0.25f *` finishing scales are plain multiplies.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../../../quants/blocks.zig");
const avx = @import("avx2.zig");
const common = @import("common.zig");
const c = impl.c;

const f = common.f;
const as = common.as;

/// Ports `keven_signs_q2xs` (arch/x86/quants.c:2624 @c1d0e7a00), read as
/// `uint64_t *`.
///
/// Entry `i` has `-1` in byte `j` where bit `j` of `i` is set, for `j` in
/// 0..6, and byte 7 set so the eight bytes multiply to `+1` -- the parity
/// that lets seven bits carry eight signs.
const keven_signs: [128]u64 = blk: {
    @setEvalBranchQuota(20000);
    var t: [128]u64 = undefined;
    for (0..128) |i| {
        var bytes: [8]i8 = undefined;
        var prod: i32 = 1;
        for (0..7) |j| {
            bytes[j] = if ((i >> j) & 1 != 0) -1 else 1;
            prod *= bytes[j];
        }
        bytes[7] = if (prod == 1) 1 else -1;
        t[i] = @bitCast(bytes);
    }
    break :blk t;
};

inline fn grid64(comptime name: []const u8) [*]const u64 {
    return @ptrCast(&@field(c, name));
}

inline fn grid32(comptime name: []const u8) [*]const u32 {
    return @ptrCast(&@field(c, name));
}

/// Four 64-bit grid entries into one register, `e3` highest.
inline fn set64(e3: u64, e2: u64, e1: u64, e0: u64) avx.m256i {
    return avx.mm256_set_epi64x(@bitCast(e3), @bitCast(e2), @bitCast(e1), @bitCast(e0));
}

/// Eight 32-bit grid entries into one register, `v[0]` highest.
inline fn set32(v: [8]u32) avx.m256i {
    return avx.mm256_set_epi32(@bitCast(v));
}

/// The `(2*ls+1)` broadcast the `iq2`/`iq3` kernels multiply by.
inline fn oddScale(ls: u32) avx.m256i {
    return avx.mm256_set1_epi16(@intCast(2 * ls + 1));
}

/// Ports `ggml_vec_dot_iq2_xxs_q8_K` (arch/x86/quants.c:2660 @c1d0e7a00),
/// the `__AVX2__` arm.
pub export fn ggml_vec_dot_iq2_xxs_q8_K(
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

    const x = as(blocks.IQ2_XXS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const grid = grid64("iq2xxs_grid");
    const signs64 = &keven_signs;

    var aux32: [4]u32 = undefined;
    const aux8: *const [16]u8 = @ptrCast(&aux32);

    var accumf = avx.mm256_setzero_ps();
    for (0..nb) |i| {
        const d = f(x[i].d) * y[i].d;
        var q2: [*]const u16 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;
        var sumi1 = avx.mm256_setzero_si256();
        var sumi2 = avx.mm256_setzero_si256();
        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            @memcpy(std.mem.asBytes(&aux32), std.mem.sliceAsBytes(q2[0..8]));
            q2 += 8;
            const q2_1 = set64(grid[aux8[3]], grid[aux8[2]], grid[aux8[1]], grid[aux8[0]]);
            const q2_2 = set64(grid[aux8[11]], grid[aux8[10]], grid[aux8[9]], grid[aux8[8]]);
            const s2_1 = set64(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127], signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const s2_2 = set64(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127], signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const q8s_1 = avx.mm256_sign_epi8(q8_1, s2_1);
            const q8s_2 = avx.mm256_sign_epi8(q8_2, s2_2);
            const dot1 = avx.mm256_maddubs_epi16(q2_1, q8s_1);
            const dot2 = avx.mm256_maddubs_epi16(q2_2, q8s_2);
            const ls1 = aux32[1] >> 28;
            const ls2 = aux32[3] >> 28;
            const p1 = avx.mm256_madd_epi16(dot1, oddScale(ls1));
            const p2 = avx.mm256_madd_epi16(dot2, oddScale(ls2));
            sumi1 = avx.mm256_add_epi32(sumi1, p1);
            sumi2 = avx.mm256_add_epi32(sumi2, p2);
        }

        accumf = avx.mm256_fmadd_ps(avx.mm256_set1_ps(d), avx.mm256_cvtepi32_ps(avx.mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    s[0] = 0.125 * common.hsum_float_8(accumf);
}

/// Ports `ggml_vec_dot_iq2_xs_q8_K` (arch/x86/quants.c:2778 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// Nine-bit grid indices and seven sign bits per `u16`; the eighth sign is
/// the parity of the seven, recovered with a 16-entry `pshufb` over their
/// XOR-folded value, and the signs are spread to bytes with a shuffle and a
/// compare against a bit selector.
pub export fn ggml_vec_dot_iq2_xs_q8_K(
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

    const x = as(blocks.IQ2_XS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const grid = grid64("iq2xs_grid");

    const mone = avx.mm256_set1_epi8(1);
    const block_sign_shuffle_mask_1 = [32]u8{
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    const block_sign_shuffle_mask_2 = [32]u8{
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    const bit_selector_mask_bytes = [32]u8{
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const bit_selector_mask = comptime avx.as(avx.m256i, bit_selector_mask_bytes);
    const block_sign_shuffle_1 = comptime avx.as(avx.m256i, block_sign_shuffle_mask_1);
    const block_sign_shuffle_2 = comptime avx.as(avx.m256i, block_sign_shuffle_mask_2);

    const k_bit_helper = [32]u8{
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const bit_helper = comptime avx.as(avx.m256i, k_bit_helper);
    const m511 = avx.mm256_set1_epi16(511);
    const m4 = avx.mm_set1_epi8(0xf);
    const m1 = avx.mm_set1_epi8(1);

    var accumf = avx.mm256_setzero_ps();
    for (0..nb) |i| {
        const d = f(x[i].d) * y[i].d;
        var q2: [*]const u16 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        const aux64 = std.mem.readInt(u64, &x[i].scales, .little);
        var stmp = avx.mm_set1_epi64x(@bitCast(aux64));
        stmp = avx.mm_unpacklo_epi8(avx.mm_and_si128(stmp, m4), avx.mm_and_si128(avx.mm_srli_epi16(stmp, 4), m4));
        const scales = avx.mm_add_epi8(avx.mm_slli_epi16(stmp, 1), m1);

        var sumi1 = avx.mm256_setzero_si256();
        var sumi2 = avx.mm256_setzero_si256();
        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 4) {
            const q2_data = avx.mm256_loadu_si256(q2);
            q2 += 16;
            // The C reads these sixteen indices back through a `uint16_t *`
            // aliasing the register.
            const gindex = avx.as([16]u16, avx.mm256_and_si256(q2_data, m511));

            const partial_sign_bits = avx.mm256_srli_epi16(q2_data, 9);
            const partial_sign_bits_upper = avx.mm256_srli_epi16(q2_data, 13);
            const partial_sign_bits_for_counting = avx.mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);

            const odd_bits = avx.mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const full_sign_bits = avx.mm256_or_si256(partial_sign_bits, odd_bits);

            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_3 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_4 = avx.mm256_loadu_si256(q8);
            q8 += 32;

            const q2_1 = set64(grid[gindex[3]], grid[gindex[2]], grid[gindex[1]], grid[gindex[0]]);
            const q2_2 = set64(grid[gindex[7]], grid[gindex[6]], grid[gindex[5]], grid[gindex[4]]);
            const q2_3 = set64(grid[gindex[11]], grid[gindex[10]], grid[gindex[9]], grid[gindex[8]]);
            const q2_4 = set64(grid[gindex[15]], grid[gindex[14]], grid[gindex[13]], grid[gindex[12]]);

            const full_signs_l = avx.mm256_castsi256_si128(full_sign_bits);
            const full_signs_h = avx.mm256_extractf128_si256(full_sign_bits, 1);
            const full_signs_1 = avx.mm256_set_m128i(full_signs_l, full_signs_l);
            const full_signs_2 = avx.mm256_set_m128i(full_signs_h, full_signs_h);

            var signs = avx.mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = avx.mm256_cmpeq_epi8(avx.mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const q8s_1 = avx.mm256_sign_epi8(q8_1, avx.mm256_or_si256(signs, mone));

            signs = avx.mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = avx.mm256_cmpeq_epi8(avx.mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const q8s_2 = avx.mm256_sign_epi8(q8_2, avx.mm256_or_si256(signs, mone));

            signs = avx.mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = avx.mm256_cmpeq_epi8(avx.mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const q8s_3 = avx.mm256_sign_epi8(q8_3, avx.mm256_or_si256(signs, mone));

            signs = avx.mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = avx.mm256_cmpeq_epi8(avx.mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const q8s_4 = avx.mm256_sign_epi8(q8_4, avx.mm256_or_si256(signs, mone));

            const dot1 = avx.mm256_maddubs_epi16(q2_1, q8s_1);
            const dot2 = avx.mm256_maddubs_epi16(q2_2, q8s_2);
            const dot3 = avx.mm256_maddubs_epi16(q2_3, q8s_3);
            const dot4 = avx.mm256_maddubs_epi16(q2_4, q8s_4);

            const sc1 = avx.mm256_cvtepi8_epi16(avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(ib32 + 0)));
            const sc2 = avx.mm256_cvtepi8_epi16(avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(ib32 + 1)));
            const sc3 = avx.mm256_cvtepi8_epi16(avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(ib32 + 2)));
            const sc4 = avx.mm256_cvtepi8_epi16(avx.mm_shuffle_epi8(scales, common.get_scale_shuffle(ib32 + 3)));

            sumi1 = avx.mm256_add_epi32(sumi1, avx.mm256_madd_epi16(dot1, sc1));
            sumi2 = avx.mm256_add_epi32(sumi2, avx.mm256_madd_epi16(dot2, sc2));
            sumi1 = avx.mm256_add_epi32(sumi1, avx.mm256_madd_epi16(dot3, sc3));
            sumi2 = avx.mm256_add_epi32(sumi2, avx.mm256_madd_epi16(dot4, sc4));
        }

        accumf = avx.mm256_fmadd_ps(avx.mm256_set1_ps(d), avx.mm256_cvtepi32_ps(avx.mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    s[0] = 0.125 * common.hsum_float_8(accumf);
}

/// The two masks `ggml_vec_dot_iq2_s_q8_K` and `ggml_vec_dot_iq3_s_q8_K`
/// spread sixteen sign bits to bytes with (arch/x86/quants.c:3075, 3384
/// @c1d0e7a00): the first picks the byte holding each weight's bit, the
/// second the bit within it.
const k_mask1 = [32]u8{
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
};
const k_mask2 = [32]u8{
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
};

/// `q8` with the sign of each weight applied, from 32 sign bits:
/// `(q8 ^ s) - s` where `s` is all ones on a set bit.
inline fn applySigns(q8: avx.m256i, bits: u32, mask1: avx.m256i, mask2: avx.m256i) avx.m256i {
    var aux256 = avx.mm256_set1_epi32(@bitCast(bits));
    aux256 = avx.mm256_and_si256(avx.mm256_shuffle_epi8(aux256, mask1), mask2);
    const s2 = avx.mm256_cmpeq_epi8(aux256, mask2);
    return avx.mm256_sub_epi8(avx.mm256_xor_si256(s2, q8), s2);
}

/// Ports `ggml_vec_dot_iq2_s_q8_K` (arch/x86/quants.c:3075 @c1d0e7a00), the
/// `__AVX2__` arm.
pub export fn ggml_vec_dot_iq2_s_q8_K(
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

    const x = as(blocks.IQ2_S, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const grid = grid64("iq2s_grid");

    const m4 = avx.mm_set1_epi8(0xf);
    const m1 = avx.mm_set1_epi8(1);

    const mask1 = comptime avx.as(avx.m256i, k_mask1);
    const mask2 = comptime avx.as(avx.m256i, k_mask2);

    var accumf = avx.mm256_setzero_ps();
    for (0..nb) |i| {
        const d = f(x[i].d) * y[i].d;
        var qs: [*]const u8 = &x[i].qs;
        const qh: [*]const u8 = &x[i].qh;
        var signs: [*]const u8 = x[i].qs[c.QK_K / 8 ..].ptr;
        var q8: [*]const i8 = &y[i].qs;

        const aux64 = std.mem.readInt(u64, &x[i].scales, .little);
        const scales8 = avx.mm_add_epi8(avx.mm_slli_epi16(avx.mm_and_si128(avx.mm_set_epi64x(@bitCast(aux64 >> 4), @bitCast(aux64)), m4), 1), m1);
        const scales16 = avx.mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        var sumi1 = avx.mm256_setzero_si256();
        var sumi2 = avx.mm256_setzero_si256();
        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const h0: u32 = qh[ib32 + 0];
            const h1: u32 = qh[ib32 + 1];
            const q2_1 = set64(
                grid[qs[3] | ((h0 << 2) & 0x300)],
                grid[qs[2] | ((h0 << 4) & 0x300)],
                grid[qs[1] | ((h0 << 6) & 0x300)],
                grid[qs[0] | ((h0 << 8) & 0x300)],
            );
            const q2_2 = set64(
                grid[qs[7] | ((h1 << 2) & 0x300)],
                grid[qs[6] | ((h1 << 4) & 0x300)],
                grid[qs[5] | ((h1 << 6) & 0x300)],
                grid[qs[4] | ((h1 << 8) & 0x300)],
            );
            qs += 8;

            const q8s_1 = applySigns(q8_1, std.mem.readInt(u32, signs[0..4], .little), mask1, mask2);
            const q8s_2 = applySigns(q8_2, std.mem.readInt(u32, signs[4..8], .little), mask1, mask2);

            signs += 8;

            const dot1 = avx.mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const dot2 = avx.mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const p1 = avx.mm256_madd_epi16(dot1, avx.mm256_shuffle_epi8(scales16, common.get_scale_shuffle_k4(ib32 + 0)));
            const p2 = avx.mm256_madd_epi16(dot2, avx.mm256_shuffle_epi8(scales16, common.get_scale_shuffle_k4(ib32 + 1)));
            sumi1 = avx.mm256_add_epi32(sumi1, p1);
            sumi2 = avx.mm256_add_epi32(sumi2, p2);
        }

        accumf = avx.mm256_fmadd_ps(avx.mm256_set1_ps(d), avx.mm256_cvtepi32_ps(avx.mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    s[0] = 0.125 * common.hsum_float_8(accumf);
}

/// Ports `ggml_vec_dot_iq3_xxs_q8_K` (arch/x86/quants.c:3260 @c1d0e7a00),
/// the `__AVX2__` arm.
pub export fn ggml_vec_dot_iq3_xxs_q8_K(
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

    const x = as(blocks.IQ3_XXS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const grid = grid32("iq3xxs_grid");
    const signs64 = &keven_signs;

    var aux32: [2]u32 = undefined;

    var accumf = avx.mm256_setzero_ps();
    for (0..nb) |i| {
        const d = f(x[i].d) * y[i].d;
        var q3: [*]const u8 = &x[i].qs;
        var gas: [*]const u8 = x[i].qs[c.QK_K / 4 ..].ptr;
        var q8: [*]const i8 = &y[i].qs;
        var sumi1 = avx.mm256_setzero_si256();
        var sumi2 = avx.mm256_setzero_si256();
        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q2_1 = set32(.{ grid[q3[7]], grid[q3[6]], grid[q3[5]], grid[q3[4]], grid[q3[3]], grid[q3[2]], grid[q3[1]], grid[q3[0]] });
            q3 += 8;
            const q2_2 = set32(.{ grid[q3[7]], grid[q3[6]], grid[q3[5]], grid[q3[4]], grid[q3[3]], grid[q3[2]], grid[q3[1]], grid[q3[0]] });
            q3 += 8;
            @memcpy(std.mem.asBytes(&aux32), gas[0..8]);
            gas += 8;
            const s2_1 = set64(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127], signs64[(aux32[0] >> 7) & 127], signs64[(aux32[0] >> 0) & 127]);
            const s2_2 = set64(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127], signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const q8s_1 = avx.mm256_sign_epi8(q8_1, s2_1);
            const q8s_2 = avx.mm256_sign_epi8(q8_2, s2_2);
            const dot1 = avx.mm256_maddubs_epi16(q2_1, q8s_1);
            const dot2 = avx.mm256_maddubs_epi16(q2_2, q8s_2);
            const ls1 = aux32[0] >> 28;
            const ls2 = aux32[1] >> 28;
            const p1 = avx.mm256_madd_epi16(dot1, oddScale(ls1));
            const p2 = avx.mm256_madd_epi16(dot2, oddScale(ls2));
            sumi1 = avx.mm256_add_epi32(sumi1, p1);
            sumi2 = avx.mm256_add_epi32(sumi2, p2);
        }

        accumf = avx.mm256_fmadd_ps(avx.mm256_set1_ps(d), avx.mm256_cvtepi32_ps(avx.mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    s[0] = 0.25 * common.hsum_float_8(accumf);
}

/// Ports `ggml_vec_dot_iq3_s_q8_K` (arch/x86/quants.c:3384 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// Nine-bit indices: eight from `qs` and the ninth from `qh`, moved into
/// place with a per-lane variable shift. The C keeps them in a union with
/// the two registers and reads the lanes back as `uint32_t`.
pub export fn ggml_vec_dot_iq3_s_q8_K(
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

    const x = as(blocks.IQ3_S, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const grid = grid32("iq3s_grid");

    const mask1 = comptime avx.as(avx.m256i, k_mask1);
    const mask2 = comptime avx.as(avx.m256i, k_mask2);

    const idx_shift = avx.mm256_set_epi32(.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    const idx_mask = avx.mm256_set1_epi32(256);

    var accumf = avx.mm256_setzero_ps();
    for (0..nb) |i| {
        const d = f(x[i].d) * y[i].d;
        var qs: [*]const u8 = &x[i].qs;
        const qh: [*]const u8 = &x[i].qh;
        var signs: [*]const u8 = &x[i].signs;
        var q8: [*]const i8 = &y[i].qs;
        var sumi1 = avx.mm256_setzero_si256();
        var sumi2 = avx.mm256_setzero_si256();
        var ib32: usize = 0;
        while (ib32 < c.QK_K / 32) : (ib32 += 2) {
            const q8_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const idx_l = avx.mm256_cvtepu8_epi16(avx.mm_loadu_si128(qs));
            qs += 16;
            var vec0 = avx.mm256_set1_epi32(qh[ib32 + 0]);
            var vec1 = avx.mm256_set1_epi32(qh[ib32 + 1]);
            vec0 = avx.mm256_and_si256(avx.mm256_sllv_epi32(vec0, idx_shift), idx_mask);
            vec1 = avx.mm256_and_si256(avx.mm256_sllv_epi32(vec1, idx_shift), idx_mask);
            vec0 = avx.mm256_or_si256(vec0, avx.mm256_cvtepi16_epi32(avx.mm256_castsi256_si128(idx_l)));
            vec1 = avx.mm256_or_si256(vec1, avx.mm256_cvtepi16_epi32(avx.mm256_extractf128_si256(idx_l, 1)));
            const index0 = avx.as([8]u32, vec0);
            const index1 = avx.as([8]u32, vec1);

            const q2_1 = set32(.{ grid[index0[7]], grid[index0[6]], grid[index0[5]], grid[index0[4]], grid[index0[3]], grid[index0[2]], grid[index0[1]], grid[index0[0]] });
            const q2_2 = set32(.{ grid[index1[7]], grid[index1[6]], grid[index1[5]], grid[index1[4]], grid[index1[3]], grid[index1[2]], grid[index1[1]], grid[index1[0]] });

            const q8s_1 = applySigns(q8_1, std.mem.readInt(u32, signs[0..4], .little), mask1, mask2);
            const q8s_2 = applySigns(q8_2, std.mem.readInt(u32, signs[4..8], .little), mask1, mask2);

            signs += 8;

            const dot1 = avx.mm256_maddubs_epi16(q2_1, q8s_1);
            const dot2 = avx.mm256_maddubs_epi16(q2_2, q8s_2);
            const ls1: u32 = x[i].scales[ib32 / 2] & 0xf;
            const ls2: u32 = x[i].scales[ib32 / 2] >> 4;
            const p1 = avx.mm256_madd_epi16(dot1, oddScale(ls1));
            const p2 = avx.mm256_madd_epi16(dot2, oddScale(ls2));
            sumi1 = avx.mm256_add_epi32(sumi1, p1);
            sumi2 = avx.mm256_add_epi32(sumi2, p2);
        }

        accumf = avx.mm256_fmadd_ps(avx.mm256_set1_ps(d), avx.mm256_cvtepi32_ps(avx.mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    s[0] = common.hsum_float_8(accumf);
}

/// Ports `ggml_vec_dot_iq1_s_q8_K` (arch/x86/quants.c:3594 @c1d0e7a00), the
/// `__AVX2__` arm with its `__BMI2__` index assembly.
///
/// The grid holds signed bytes, so `mul_add_epi8`'s sign trick takes them.
/// The per-group delta, `±IQ1S_DELTA` by `qh`'s top bit, is applied once
/// per super-block through `bsums` in a scalar accumulator.
pub export fn ggml_vec_dot_iq1_s_q8_K(
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

    const x = as(blocks.IQ1_S, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const grid = grid64("iq1s_grid");

    var accum = avx.mm256_setzero_ps();
    var accum1: f32 = 0;
    for (0..nb) |i| {
        var q8: [*]const i8 = &y[i].qs;
        var qs: [*]const u8 = &x[i].qs;
        const qh: [*]const u16 = &x[i].qh;

        var sumi = avx.mm256_setzero_si256();
        var sumi1: i32 = 0;
        var ib: usize = 0;
        while (ib < c.QK_K / 32) : (ib += 2) {
            const packed_idx1 = avx.pdep_u64(std.mem.readInt(u32, qs[0..4], .little), 0x00ff00ff00ff00ff) | avx.pdep_u64(qh[ib], 0x700070007000700);
            const packed_idx2 = avx.pdep_u64(std.mem.readInt(u32, qs[4..8], .little), 0x00ff00ff00ff00ff) | avx.pdep_u64(qh[ib + 1], 0x700070007000700);
            const idx1: [4]u16 = @bitCast(packed_idx1);
            const idx2: [4]u16 = @bitCast(packed_idx2);
            const q1b_1 = set64(grid[idx1[3]], grid[idx1[2]], grid[idx1[1]], grid[idx1[0]]);
            const q1b_2 = set64(grid[idx2[3]], grid[idx2[2]], grid[idx2[1]], grid[idx2[0]]);
            qs += 8;
            const q8b_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8b_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;

            const dot1 = common.mul_add_epi8(q1b_1, q8b_1);
            const dot2 = common.mul_add_epi8(q1b_2, q8b_2);
            const ls1: i16 = @intCast(2 * ((qh[ib + 0] >> 12) & 7) + 1);
            const ls2: i16 = @intCast(2 * ((qh[ib + 1] >> 12) & 7) + 1);
            const p1 = avx.mm256_madd_epi16(dot1, avx.mm256_set1_epi16(ls1));
            const p2 = avx.mm256_madd_epi16(dot2, avx.mm256_set1_epi16(ls2));

            sumi = avx.mm256_add_epi32(sumi, avx.mm256_add_epi32(p1, p2));
            const sign1: i32 = if (qh[ib + 0] & 0x8000 != 0) -1 else 1;
            const sign2: i32 = if (qh[ib + 1] & 0x8000 != 0) -1 else 1;
            sumi1 += (@as(i32, y[i].bsums[2 * ib + 0]) + y[i].bsums[2 * ib + 1]) * sign1 * ls1 +
                (@as(i32, y[i].bsums[2 * ib + 2]) + y[i].bsums[2 * ib + 3]) * sign2 * ls2;
        }

        const d = y[i].d * f(x[i].d);
        accum = avx.mm256_fmadd_ps(avx.mm256_set1_ps(d), avx.mm256_cvtepi32_ps(sumi), accum);
        accum1 = @mulAdd(f32, d, @floatFromInt(sumi1), accum1);
    }

    s[0] = @mulAdd(f32, c.IQ1S_DELTA, accum1, common.hsum_float_8(accum));
}

/// Ports `ggml_vec_dot_iq1_m_q8_K` (arch/x86/quants.c:3713 @c1d0e7a00), the
/// `__AVX2__` arm with its `__BMI2__` index and delta assembly.
///
/// The super-block's fp16 scale is spread across the top nibbles of the four
/// 16-bit scale words; the 3-bit group scales are the rest. The delta's
/// sign per eight weights becomes a byte that is negative or `+1`, applied
/// with `psign` into a second accumulator, scaled by `IQ1M_DELTA` at the end.
pub export fn ggml_vec_dot_iq1_m_q8_K(
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

    const x = as(blocks.IQ1_M, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const grid = grid64("iq1s_grid");

    const mask = avx.mm256_set1_epi16(0x7);
    const mone = avx.mm256_set1_epi16(1);
    const mone8 = avx.mm256_set1_epi8(1);
    const mtwo8 = avx.mm256_set1_epi8(2);
    // VPSHUFB cannot cross 128-bit lanes so odd shifts go to upper half.
    const scales_shift = avx.mm256_set_epi64x(9, 3, 6, 0);

    var accum1 = avx.mm256_setzero_ps();
    var accum2 = avx.mm256_setzero_ps();
    for (0..nb) |i| {
        var q8: [*]const i8 = &y[i].qs;
        var qs: [*]const u8 = &x[i].qs;
        var qh: [*]const u8 = &x[i].qh;
        const sc: [4]u16 = @bitCast(x[i].scales);

        const scale: u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        // Extract 3-bit scales (16 values)
        var scales = avx.mm256_set1_epi64x(@bitCast(x[i].scales));
        scales = avx.mm256_srlv_epi64(scales, scales_shift);
        scales = avx.mm256_add_epi16(avx.mm256_slli_epi16(avx.mm256_and_si256(scales, mask), 1), mone);

        // Indices to repeat each scale 8 times.
        var scales_idx1 = avx.mm256_set1_epi16(0x0100);
        var scales_idx2 = avx.mm256_add_epi8(scales_idx1, avx.mm256_set1_epi8(8));

        var sumi1 = avx.mm256_setzero_si256();
        var sumi2 = avx.mm256_setzero_si256();
        var ib: usize = 0;
        while (ib < c.QK_K / 32) : (ib += 2) {
            const packed_idx1 = avx.pdep_u64(std.mem.readInt(u32, qs[0..4], .little), 0x00ff00ff00ff00ff) |
                avx.pdep_u64(std.mem.readInt(u16, qh[0..2], .little) & 0x7777, 0xf000f000f000f00);
            const packed_idx2 = avx.pdep_u64(std.mem.readInt(u32, qs[4..8], .little), 0x00ff00ff00ff00ff) |
                avx.pdep_u64(std.mem.readInt(u16, qh[2..4], .little) & 0x7777, 0xf000f000f000f00);
            const idx1: [4]u16 = @bitCast(packed_idx1);
            const idx2: [4]u16 = @bitCast(packed_idx2);
            const q1b_1 = set64(grid[idx1[3]], grid[idx1[2]], grid[idx1[1]], grid[idx1[0]]);
            const q1b_2 = set64(grid[idx2[3]], grid[idx2[2]], grid[idx2[1]], grid[idx2[0]]);

            // Convert signs to bytes 0x81 (negative) or 0x01 (positive)
            const delta_sign = avx.pdep_u64(std.mem.readInt(u32, qh[0..4], .little) & 0x88888888, 0xf0f0f0f0f0f0f0f0);
            const delta1 = avx.mm256_or_si256(mone8, avx.mm256_cvtepi8_epi64(avx.mm_set1_epi32(@bitCast(@as(u32, @truncate(delta_sign))))));
            const delta2 = avx.mm256_or_si256(mone8, avx.mm256_cvtepi8_epi64(avx.mm_set1_epi32(@bitCast(@as(u32, @truncate(delta_sign >> 32))))));

            const q8b_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8b_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;

            const dot1 = common.mul_add_epi8(q1b_1, q8b_1);
            const dot2 = common.mul_add_epi8(q1b_2, q8b_2);
            const dot3 = avx.mm256_maddubs_epi16(mone8, avx.mm256_sign_epi8(q8b_1, delta1));
            const dot4 = avx.mm256_maddubs_epi16(mone8, avx.mm256_sign_epi8(q8b_2, delta2));

            const scale1 = avx.mm256_shuffle_epi8(scales, scales_idx1);
            const scale2 = avx.mm256_shuffle_epi8(scales, scales_idx2);

            scales_idx1 = avx.mm256_add_epi8(scales_idx1, mtwo8);
            scales_idx2 = avx.mm256_add_epi8(scales_idx2, mtwo8);

            const p1 = avx.mm256_madd_epi16(dot1, scale1);
            const p2 = avx.mm256_madd_epi16(dot2, scale2);
            const p3 = avx.mm256_madd_epi16(dot3, scale1);
            const p4 = avx.mm256_madd_epi16(dot4, scale2);

            sumi1 = avx.mm256_add_epi32(sumi1, avx.mm256_add_epi32(p1, p2));
            sumi2 = avx.mm256_add_epi32(sumi2, avx.mm256_add_epi32(p3, p4));

            qs += 8;
            qh += 4;
        }

        const d = avx.mm256_set1_ps(y[i].d * f(scale));

        accum1 = avx.mm256_fmadd_ps(d, avx.mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = avx.mm256_fmadd_ps(d, avx.mm256_cvtepi32_ps(sumi2), accum2);
    }

    s[0] = @mulAdd(f32, c.IQ1M_DELTA, common.hsum_float_8(accum2), common.hsum_float_8(accum1));
}

/// Ports `ggml_vec_dot_iq4_xs_q8_K` (arch/x86/quants.c:4004 @c1d0e7a00), the
/// `__AVX2__` arm.
///
/// `iq4_nl`'s codebook in super-blocks, with 6-bit signed group scales split
/// between `scales_l` and `scales_h`.
pub export fn ggml_vec_dot_iq4_xs_q8_K(
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
    std.debug.assert(@rem(n, c.QK_K) == 0);

    const x = as(blocks.IQ4_XS, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const values128 = avx.mm_loadu_si128(&c.kvalues_iq4nl);
    const m4b = avx.mm_set1_epi8(0x0f);

    var accum = avx.mm256_setzero_ps();
    for (0..nb) |ibl| {
        var qs: [*]const u8 = &x[ibl].qs;
        var q8: [*]const i8 = &y[ibl].qs;
        var sh: u16 = x[ibl].scales_h;
        var sumi1 = avx.mm256_setzero_si256();
        var sumi2 = avx.mm256_setzero_si256();
        var ib: usize = 0;
        while (ib < c.QK_K / 32) : (ib += 2) {
            const q4bits_1 = avx.mm_loadu_si128(qs);
            qs += 16;
            const q4bits_2 = avx.mm_loadu_si128(qs);
            qs += 16;
            const q8b_1 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q8b_2 = avx.mm256_loadu_si256(q8);
            q8 += 32;
            const q4b_1 = avx.mm256_set_m128i(
                avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_1, 4), m4b)),
                avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_1, m4b)),
            );
            const q4b_2 = avx.mm256_set_m128i(
                avx.mm_shuffle_epi8(values128, avx.mm_and_si128(avx.mm_srli_epi16(q4bits_2, 4), m4b)),
                avx.mm_shuffle_epi8(values128, avx.mm_and_si128(q4bits_2, m4b)),
            );
            const p16_1 = common.mul_add_epi8(q4b_1, q8b_1);
            const p16_2 = common.mul_add_epi8(q4b_2, q8b_2);
            const sl: i32 = x[ibl].scales_l[ib / 2];
            const ls1: i16 = @intCast(((sl & 0xf) | ((@as(i32, sh) << 4) & 0x30)) - 32);
            const ls2: i16 = @intCast(((sl >> 4) | ((@as(i32, sh) << 2) & 0x30)) - 32);
            sh >>= 4;
            const p_1 = avx.mm256_madd_epi16(p16_1, avx.mm256_set1_epi16(ls1));
            const p_2 = avx.mm256_madd_epi16(p16_2, avx.mm256_set1_epi16(ls2));
            sumi1 = avx.mm256_add_epi32(p_1, sumi1);
            sumi2 = avx.mm256_add_epi32(p_2, sumi2);
        }
        accum = avx.mm256_fmadd_ps(avx.mm256_set1_ps(f(x[ibl].d) * y[ibl].d), avx.mm256_cvtepi32_ps(avx.mm256_add_epi32(sumi1, sumi2)), accum);
    }

    s[0] = common.hsum_float_8(accum);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
