//! The `__AVX2__` arms of the IQ panel path.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/iqp.cpp` (v0.4.1, `b29c606e2`):
//! every `#if defined(__AVX2__)` block and arm, over the intrinsics in
//! `quants/x86/avx2.zig`. Each names the C it replaces and the line it began
//! at; the portable half and the `#else` arms are `iqp.zig`.
//!
//! # The inner arms, chosen at comptime as the C's `#if` chooses
//!
//! - `iqp_sign_mask`: `vgf2p8affineqb` under `__GFNI__` (`tigerlake`), the
//!   `and` and `cmpeq` pair otherwise (x86-64-v3, v4).
//! - `mul_sum_us8_pairs_acc_int32x8`: `vpdpbusd` under `__AVX512VNNI__ &&
//!   __AVX512VL__` (`tigerlake`), its AVX-VNNI encoding under `__AVXVNNI__`,
//!   `maddubs` then `madd` otherwise.
//! - `mul_sum_i8_pairs_acc_int32x8`: `vpdpbssd` under `__AVXVNNIINT8__`, the
//!   `sign` trick otherwise.
//! - `GGML_IQP_USE_BIAS` (`iqp.use_bias`, VNNI): activations offset by 128
//!   and fed to `vpdpbusd` as unsigned, with `bias[]` subtracted; otherwise
//!   the signed dot product.
//!
//! No machine this project gates on has AVX-VNNI or AVX-VNNI-INT8; those two
//! arms are checked by compiling and disassembling only.
//!
//! # The float epilogue
//!
//! The only float arithmetic is the C's explicit `_mm256_mul_ps` of the two
//! scales and `_mm256_fmadd_ps` into the accumulator: named in the C, so
//! nothing here depends on contraction.
//!
//! # Only compiled for AVX2
//!
//! `iqp.zig` imports this file only when the target has AVX2.

const std = @import("std");
const qblocks = @import("../quants/blocks.zig");
const avx = @import("quants/x86/avx2.zig");
const iqp = @import("iqp.zig");
const impl = @import("../impl.zig");
const c = impl.c;

const m256i = avx.m256i;
const m256 = avx.m256;
const Q8K = qblocks.Q8_K;
const BlockX8 = iqp.BlockX8;
const QK_K = c.QK_K;
const nsb = iqp.nsb;

/// Ports `iqp_sign_mask` (iqp.cpp:71 @b29c606e2): `0xFF` in every byte
/// whose sign bit is set; `sv` holds each sign byte broadcast over the eight
/// bytes it governs.
///
/// Under `__GFNI__` one affine transform does the `and` and the compare: the
/// matrix is the sign byte in all eight rows, so output bit `k` of byte `j`
/// is the parity of `s & (1 << j)` -- bit `j` of `s`, in every bit.
inline fn signMask(sv: m256i) m256i {
    const sel = avx.mm256_set1_epi64x(@bitCast(@as(u64, 0x8040201008040201)));

    if (comptime avx.has_gfni) return avx.mm256_gf2p8affine_epi64_epi8(sel, sv, 0);

    return avx.mm256_cmpeq_epi8(avx.mm256_and_si256(sv, sel), sel);
}

/// Ports `iqp_sign_bytes` (iqp.cpp:83 @b29c606e2): `signs` holds four sign
/// bytes, byte `l` governing values `8l .. 8l+7`; each is spread over its
/// eight lanes.
inline fn signBytes(signs: u32) m256i {
    const bcast = avx.mm256_setr_epi8(.{
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    });

    return avx.mm256_shuffle_epi8(avx.mm256_set1_epi32(@bitCast(signs)), bcast);
}

/// Ports `iqp_apply_signs` (iqp.cpp:91 @b29c606e2): `x ^ m - m` negates the
/// lanes where `m` is `0xFF`.
inline fn applySigns(x: m256i, m: m256i) m256i {
    return avx.mm256_sub_epi8(avx.mm256_xor_si256(x, m), m);
}

/// Ports the `__AVX2__` arm of `iqp_store_signed_x8` (iqp.cpp:98 @b29c606e2).
pub inline fn storeSignedX8(dst: [*]i8, g0: u64, g1: u64, g2: u64, g3: u64, signs: u32) void {
    const g = avx.mm256_set_epi64x(@bitCast(g3), @bitCast(g2), @bitCast(g1), @bitCast(g0));
    const m = signMask(signBytes(signs));

    avx.mm256_storeu_si256(dst, applySigns(g, m));
}

/// Ports the `__AVX2__` arm of `iqp_store_signed_x4` (iqp.cpp:124 @b29c606e2).
pub inline fn storeSignedX4(
    dst: [*]i8,
    g0a: u32,
    g0b: u32,
    g1a: u32,
    g1b: u32,
    g2a: u32,
    g2b: u32,
    g3a: u32,
    g3b: u32,
    signs: u32,
) void {
    const g = avx.mm256_setr_epi32(.{
        @bitCast(g0a), @bitCast(g0b), @bitCast(g1a), @bitCast(g1b),
        @bitCast(g2a), @bitCast(g2b), @bitCast(g3a), @bitCast(g3b),
    });
    const m = signMask(signBytes(signs));

    avx.mm256_storeu_si256(dst, applySigns(g, m));
}

/// Ports the `__AVX2__` arm of `iqp_store_iq1_x8` (iqp.cpp:158 @b29c606e2).
/// AVX2 has no byte shift, so `8 * g` is three doublings.
pub inline fn storeIq1X8(dst: [*]i8, g0: u64, g1: u64, g2: u64, g3: u64, deltas: u32) void {
    var g = avx.mm256_set_epi64x(@bitCast(g3), @bitCast(g2), @bitCast(g1), @bitCast(g0));

    g = avx.mm256_add_epi8(g, g);
    g = avx.mm256_add_epi8(g, g);
    g = avx.mm256_add_epi8(g, g);

    avx.mm256_storeu_si256(dst, avx.mm256_add_epi8(g, signBytes(deltas)));
}

/// Ports the `__AVX2__` arm of `iqp_store_iq4_x32` (iqp.cpp:188 @b29c606e2):
/// `kvalues_iq4nl` as a `pshufb` table.
pub inline fn storeIq4X32(dst: [*]i8, qs: [*]const u8) void {
    const q = avx.mm_loadu_si128(qs);
    const lut = avx.mm_loadu_si128(&c.kvalues_iq4nl);
    const m4 = avx.mm_set1_epi8(0xf);

    avx.mm_storeu_si128(dst + 0, avx.mm_shuffle_epi8(lut, avx.mm_and_si128(q, m4)));
    avx.mm_storeu_si128(dst + 16, avx.mm_shuffle_epi8(lut, avx.mm_and_si128(avx.mm_srli_epi16(q, 4), m4)));
}

/// Ports the `__AVX2__` arm of `iqp_weighted_sum` (iqp.cpp:207 @b29c606e2).
///
/// Groups of four bytes are summed into `i32`; per 32-byte load the low four
/// lanes cover sub-block `2i` and the high four `2i + 1`, so each half takes
/// its own scale. Integer throughout.
pub inline fn weightedSum(vals: *const [QK_K]i8, iscales: *const [nsb]i8) i32 {
    comptime std.debug.assert(iqp.sb_size == 16);

    const ones8 = avx.mm256_set1_epi8(1);
    const ones16 = avx.mm256_set1_epi16(1);

    var acc = avx.mm256_setzero_si256();

    for (0..QK_K / 32) |i| {
        const v = avx.mm256_loadu_si256(vals[32 * i ..].ptr);
        const p = avx.mm256_madd_epi16(avx.mm256_maddubs_epi16(ones8, v), ones16);

        const s = avx.mm256_set_m128i(avx.mm_set1_epi32(iscales[2 * i + 1]), avx.mm_set1_epi32(iscales[2 * i + 0]));

        acc = avx.mm256_add_epi32(acc, avx.mm256_mullo_epi32(p, s));
    }

    var sum = avx.mm_add_epi32(avx.mm256_castsi256_si128(acc), avx.mm256_extracti128_si256(acc, 1));

    sum = avx.mm_add_epi32(sum, avx.mm_shuffle_epi32(sum, avx.MM_SHUFFLE(1, 0, 3, 2)));
    sum = avx.mm_add_epi32(sum, avx.mm_shuffle_epi32(sum, avx.MM_SHUFFLE(2, 3, 0, 1)));

    return avx.mm_cvtsi128_si32(sum);
}

/// Ports `iqp_interleave_x8` (iqp.cpp:524 @b29c606e2): the 8x8 `int32`
/// transpose of the 32-column group starting at column `off` -- rows paired
/// into dword couples, then qword quadruples, then the 128-bit halves
/// swapped.
pub inline fn interleaveX8(dst: [*]i8, vals: *const [iqp.nb_rows][QK_K]i8, off: usize) void {
    comptime std.debug.assert(iqp.nb_rows == 8);

    var v: [iqp.nb_rows]m256i = undefined;

    for (0..iqp.nb_rows) |r| {
        v[r] = avx.mm256_loadu_si256(vals[r][off..].ptr);
    }

    const a0 = avx.mm256_unpacklo_epi32(v[0], v[1]);
    const a1 = avx.mm256_unpackhi_epi32(v[0], v[1]);
    const a2 = avx.mm256_unpacklo_epi32(v[2], v[3]);
    const a3 = avx.mm256_unpackhi_epi32(v[2], v[3]);
    const a4 = avx.mm256_unpacklo_epi32(v[4], v[5]);
    const a5 = avx.mm256_unpackhi_epi32(v[4], v[5]);
    const a6 = avx.mm256_unpacklo_epi32(v[6], v[7]);
    const a7 = avx.mm256_unpackhi_epi32(v[6], v[7]);

    const b0 = avx.mm256_unpacklo_epi64(a0, a2);
    const b1 = avx.mm256_unpackhi_epi64(a0, a2);
    const b2 = avx.mm256_unpacklo_epi64(a1, a3);
    const b3 = avx.mm256_unpackhi_epi64(a1, a3);
    const b4 = avx.mm256_unpacklo_epi64(a4, a6);
    const b5 = avx.mm256_unpackhi_epi64(a4, a6);
    const b6 = avx.mm256_unpacklo_epi64(a5, a7);
    const b7 = avx.mm256_unpackhi_epi64(a5, a7);

    avx.mm256_storeu_si256(dst + 0 * 32, avx.mm256_permute2x128_si256(b0, b4, 0x20));
    avx.mm256_storeu_si256(dst + 1 * 32, avx.mm256_permute2x128_si256(b1, b5, 0x20));
    avx.mm256_storeu_si256(dst + 2 * 32, avx.mm256_permute2x128_si256(b2, b6, 0x20));
    avx.mm256_storeu_si256(dst + 3 * 32, avx.mm256_permute2x128_si256(b3, b7, 0x20));
    avx.mm256_storeu_si256(dst + 4 * 32, avx.mm256_permute2x128_si256(b0, b4, 0x31));
    avx.mm256_storeu_si256(dst + 5 * 32, avx.mm256_permute2x128_si256(b1, b5, 0x31));
    avx.mm256_storeu_si256(dst + 6 * 32, avx.mm256_permute2x128_si256(b2, b6, 0x31));
    avx.mm256_storeu_si256(dst + 7 * 32, avx.mm256_permute2x128_si256(b3, b7, 0x31));
}

// -----------------------------------------------------------------------------
// The integer dot products

/// Ports `sum_i16_pairs_acc_int32x8` (iqp.cpp:782 @b29c606e2): adjacent
/// `i16` pairs summed into `i32`, then the accumulator added.
inline fn sum_i16_pairs_acc_int32x8(acc: m256i, x: m256i) m256i {
    const ones = avx.mm256_set1_epi16(1);
    return avx.mm256_add_epi32(acc, avx.mm256_madd_epi16(ones, x));
}

/// Ports `mul_sum_us8_pairs_acc_int32x8` (iqp.cpp:787 @b29c606e2): unsigned
/// bytes of `ax` times signed bytes of `sy`, four to a lane, into `acc`.
inline fn mul_sum_us8_pairs_acc_int32x8(acc: m256i, ax: m256i, sy: m256i) m256i {
    if (comptime avx.has_avx512vnni_vl) return avx.mm256_dpbusd_epi32(acc, ax, sy);
    if (comptime avx.has_avxvnni) return avx.mm256_dpbusd_avx_epi32(acc, ax, sy);

    // Multiply and create 16-bit values.
    const dot = avx.mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x8(acc, dot);
}

/// Ports `mul_sum_i8_pairs_acc_int32x8` (iqp.cpp:801 @b29c606e2): signed
/// bytes times signed bytes, four to a lane, into `acc`.
inline fn mul_sum_i8_pairs_acc_int32x8(acc: m256i, x: m256i, y: m256i) m256i {
    if (comptime avx.has_avxvnniint8) return avx.mm256_dpbssd_epi32(acc, x, y);

    // The absolute values of x, and y with x's signs.
    const ax = avx.mm256_sign_epi8(x, x);
    const sy = avx.mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
}

/// Ports `iqp_load_y` (iqp.cpp:814 @b29c606e2): the sixteen activations of
/// one sub-block in both halves, offset by 128 when they go to `vpdpbusd`
/// as unsigned bytes.
inline fn loadY(qs: [*]const i8) m256i {
    var y = avx.mm_loadu_si128(qs);
    if (comptime iqp.use_bias) {
        y = avx.mm_xor_si128(y, avx.mm_set1_epi8(-128));
    }
    return avx.mm256_broadcastsi128_si256(y);
}

/// Ports `iqp_dot4` (iqp.cpp:823 @b29c606e2): `xv` holds eight rows of four
/// signed weights, `yb` the matching four activation bytes in all eight
/// lanes.
inline fn dot4(acc: m256i, xv: m256i, yb: m256i) m256i {
    if (comptime iqp.use_bias) return mul_sum_us8_pairs_acc_int32x8(acc, yb, xv);
    return mul_sum_i8_pairs_acc_int32x8(acc, xv, yb);
}

/// Ports `iqp_load_iscales` (iqp.cpp:831 @b29c606e2): eight sub-block scales
/// sign-extended to `i32`.
inline fn loadIscales(iscales: [*]const i8) m256i {
    return avx.mm256_cvtepi8_epi32(avx.mm_loadl_epi64(iscales));
}

/// Ports `iqp_acc_block` (iqp.cpp:836 @b29c606e2): one super-block of eight
/// interleaved rows against one `q8_K` row, in `i32`. The worst case,
/// `16 * 32 * 16 * 255 * 127 = 2.65e8` plus a bias of at most `1.33e8`, does
/// not overflow.
inline fn accBlock(b: *const BlockX8, a: *const Q8K) m256i {
    var sumi = avx.mm256_setzero_si256();

    for (0..nsb) |sb| {
        const qs: [*]const i8 = b.qs[sb * 128 ..].ptr;

        const yv = loadY(a.qs[sb * 16 ..].ptr);

        var isum = avx.mm256_setzero_si256();

        isum = dot4(isum, avx.mm256_loadu_si256(qs + 0), avx.mm256_shuffle_epi32(yv, 0x00));
        isum = dot4(isum, avx.mm256_loadu_si256(qs + 32), avx.mm256_shuffle_epi32(yv, 0x55));
        isum = dot4(isum, avx.mm256_loadu_si256(qs + 64), avx.mm256_shuffle_epi32(yv, 0xAA));
        isum = dot4(isum, avx.mm256_loadu_si256(qs + 96), avx.mm256_shuffle_epi32(yv, 0xFF));

        sumi = avx.mm256_add_epi32(sumi, avx.mm256_mullo_epi32(isum, loadIscales(b.iscales[sb * 8 ..].ptr)));
    }

    if (comptime iqp.use_bias) {
        sumi = avx.mm256_sub_epi32(sumi, avx.mm256_loadu_si256(&b.bias));
    }

    return sumi;
}

/// Ports `iqp_gemm_tile_4` (iqp.cpp:862 @b29c606e2): one 4-row by
/// `nc`-column tile; `s` points at the first of the four output rows, `bs`
/// floats apart.
pub fn gemmTile4(nb: usize, s: [*]f32, bs: usize, b_ptr_start: [*]const BlockX8, a_ptr: [4][*]const Q8K, nc: c_int) void {
    const ncols_interleaved = 8;

    for (0..@intCast(@divTrunc(nc, ncols_interleaved))) |x| {
        const b_ptr = b_ptr_start + x * nb;

        var sumf: [4]m256 = undefined;
        for (0..4) |m| {
            sumf[m] = avx.mm256_setzero_ps();
        }

        for (0..nb) |l| {
            var sumi: [4]m256i = undefined;
            for (0..4) |m| {
                sumi[m] = avx.mm256_setzero_si256();
            }

            for (0..nsb) |sb| {
                const qs: [*]const i8 = b_ptr[l].qs[sb * 128 ..].ptr;

                var yv: [4]m256i = undefined;
                var isum: [4]m256i = undefined;
                for (0..4) |m| {
                    yv[m] = loadY(a_ptr[m][l].qs[sb * 16 ..].ptr);
                    isum[m] = avx.mm256_setzero_si256();
                }

                const xv0 = avx.mm256_loadu_si256(qs + 0);
                const xv1 = avx.mm256_loadu_si256(qs + 32);
                const xv2 = avx.mm256_loadu_si256(qs + 64);
                const xv3 = avx.mm256_loadu_si256(qs + 96);

                for (0..4) |m| {
                    isum[m] = dot4(isum[m], xv0, avx.mm256_shuffle_epi32(yv[m], 0x00));
                    isum[m] = dot4(isum[m], xv1, avx.mm256_shuffle_epi32(yv[m], 0x55));
                    isum[m] = dot4(isum[m], xv2, avx.mm256_shuffle_epi32(yv[m], 0xAA));
                    isum[m] = dot4(isum[m], xv3, avx.mm256_shuffle_epi32(yv[m], 0xFF));
                }

                const isc = loadIscales(b_ptr[l].iscales[sb * 8 ..].ptr);
                for (0..4) |m| {
                    sumi[m] = avx.mm256_add_epi32(sumi[m], avx.mm256_mullo_epi32(isum[m], isc));
                }
            }

            if (comptime iqp.use_bias) {
                const bias = avx.mm256_loadu_si256(&b_ptr[l].bias);
                for (0..4) |m| {
                    sumi[m] = avx.mm256_sub_epi32(sumi[m], bias);
                }
            }

            const dfac = avx.mm256_loadu_ps(&b_ptr[l].dfac);
            for (0..4) |m| {
                sumf[m] = avx.mm256_fmadd_ps(
                    avx.mm256_cvtepi32_ps(sumi[m]),
                    avx.mm256_mul_ps(dfac, avx.mm256_set1_ps(a_ptr[m][l].d)),
                    sumf[m],
                );
            }
        }

        for (0..4) |m| {
            avx.mm256_storeu_ps(s + m * bs + x * ncols_interleaved, sumf[m]);
        }
    }
}

/// Ports the `__AVX2__` arm of `iqp_gemv_8x8_q8_K` (iqp.cpp:934 @b29c606e2).
pub fn gemv(n: c_int, s: [*]f32, b_ptr_start: [*]const BlockX8, a_ptr: [*]const Q8K, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, QK_K));
    const ncols_interleaved = 8;

    for (0..@intCast(@divTrunc(nc, ncols_interleaved))) |x| {
        const b_ptr = b_ptr_start + x * nb;

        var sumf = avx.mm256_setzero_ps();

        for (0..nb) |l| {
            const dv = avx.mm256_mul_ps(avx.mm256_loadu_ps(&b_ptr[l].dfac), avx.mm256_set1_ps(a_ptr[l].d));

            sumf = avx.mm256_fmadd_ps(avx.mm256_cvtepi32_ps(accBlock(&b_ptr[l], &a_ptr[l])), dv, sumf);
        }

        avx.mm256_storeu_ps(s + x * ncols_interleaved, sumf);
    }
}

/// Ports the `__AVX2__` arm of `iqp_gemm_8x8_q8_K` (iqp.cpp:976 @b29c606e2).
pub fn gemm(n: c_int, s: [*]f32, bs: usize, b_ptr_start: [*]const BlockX8, a_ptr_start: [*]const Q8K, nr: c_int, nc: c_int) void {
    const nb: usize = @intCast(@divTrunc(n, QK_K));

    for (0..@intCast(@divTrunc(nr, 4))) |y| {
        var a_ptr: [4][*]const Q8K = undefined;
        for (0..4) |m| {
            a_ptr[m] = a_ptr_start + (y * 4 + m) * nb;
        }

        gemmTile4(nb, s + y * 4 * bs, bs, b_ptr_start, a_ptr, nc);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
