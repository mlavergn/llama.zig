//! `flash_attn_ext`, `flash_attn_back` and `lightning_indexer`.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-cpu/ops.cpp`    — the kernels
//! - `llama.cpp/ggml/src/ggml-cpu/simd-gemm.h` — `simd_gemm`, which the tiled
//!                                               path calls; `static` there,
//!                                               so it has no symbol
//! - `llama.cpp/ggml/src/ggml-cpu/common.h`    — the tile sizes
//! - `llama.cpp/ggml/src/ggml-impl.h`          — `ggml_up`
//!
//! all at v0.3.0 (`c1d0e7a00`). Each declaration below names the C++ it
//! replaces and the line it began at.
//!
//! # Three paths through `flash_attn_ext`
//!
//! `ggml_compute_forward_flash_attn_ext_f16` picks one per call, and each is
//! ported as the C has it:
//!
//! - **split-KV**, for a single query row against at least 512 keys: each
//!   thread runs `one_chunk` over its slice of the keys and writes partial
//!   `M`, `S`, `VKQ`, and `reduce_partials` combines them after a barrier.
//! - **tiled**, for at least 64 query rows of `f32`/`f16` K and V: 64×64
//!   tiles through `simd_gemm`, with the online softmax per row.
//! - **one_chunk**, everything else: one query row at a time through the K
//!   type's own `vec_dot`.
//!
//! The order of summation differs between them, so which path a call takes
//! is visible in the last bit; the selection logic is reproduced exactly.
//!
//! # Float contraction
//!
//! `S = S*ms + vs` and its kin are single expressions and clang fuses them.
//! Each site is written as `@mulAdd` and says so. Two that look like sites
//! and are not: `S[tq] += ggml_vec_soft_max_f32(...)` adds a `double`, so the
//! sum is formed in `double` and narrowed once — reproduced with explicit
//! widening — and `s = s*scale` is its own statement.
//!
//! # Loop index names
//!
//! `i1`, `i2`, `i3` are Zig integer type names. Renamed `j1`, `j2`, `j3`,
//! digit for digit.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const threading = @import("../threading.zig");
const convert = @import("../convert.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

extern fn expf(x: f32) f32;
extern fn tanhf(x: f32) f32;
extern fn powf(x: f32, y: f32) f32;
extern fn sqrtf(x: f32) f32;
extern fn fmaxf(x: f32, y: f32) f32;
extern fn log2(x: f64) f64;
extern fn floor(x: f64) f64;

const inf = std.math.inf(f32);

inline fn off(i: i64, nb: usize) usize {
    return @as(usize, @intCast(i)) * nb;
}

inline fn at(comptime T: type, base: ?*anyopaque, byte_off: usize) [*]T {
    const b: [*]u8 = @ptrCast(base.?);
    return @ptrCast(@alignCast(b + byte_off));
}

/// Ports `GGML_FA_TILE_Q` (ggml-cpu/common.h:9 @c1d0e7a00), the query rows
/// per tile of the tiled path.
const q_tile_sz: usize = 64;

/// Ports `GGML_FA_TILE_KV` (ggml-cpu/common.h:10 @c1d0e7a00), the keys per
/// tile.
const kv_tile_sz: usize = 64;

/// `GGML_F32_EPR` (simd-mappings.h:338 @c1d0e7a00), the NEON arm.
const f32_epr: i64 = 4;

/// `GGML_SOFT_MAX_UNROLL` (vec.h:49 @c1d0e7a00).
const soft_max_unroll: i64 = 4;

/// Ports `ggml_up` (ggml-impl.h:68 @c1d0e7a00): round `n` up to a multiple of
/// the power of two `m`.
inline fn up(n: i64, m: i64) i64 {
    impl.assert((m & (m - 1)) == 0, "(m & (m - 1)) == 0");
    return (n + m - 1) & ~(m - 1);
}

/// The ALiBi slope for head `h`, the expression `one_chunk` and `tiled` both
/// open with. Kept inline in the C; factored out here because it is the same
/// sixteen tokens twice.
inline fn alibiSlope(max_bias: f32, h: u32, n_head_log2: u32, m0: f32, m1: f32) f32 {
    if (!(max_bias > 0.0)) return 1.0;
    return if (h < n_head_log2)
        powf(m0, @floatFromInt(h + 1))
    else
        powf(m1, @floatFromInt(2 * (h - n_head_log2) + 1));
}

/// The scale, ALiBi bias and softcap every forward path reads out of
/// `op_params`, with the C's `scale /= logit_softcap` applied.
const AttnParams = struct {
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    n_head_log2: u32,
    m0: f32,
    m1: f32,

    fn of(dst: *const Tensor, n_head: u32) AttnParams {
        var scale = impl.getOpParamsF32(dst, 0);
        const max_bias = impl.getOpParamsF32(dst, 1);
        const logit_softcap = impl.getOpParamsF32(dst, 2);

        if (logit_softcap != 0) {
            scale /= logit_softcap;
        }

        const n_head_log2: u32 = @as(u32, 1) << @intFromFloat(floor(log2(@floatFromInt(n_head))));

        const nhl: f32 = @floatFromInt(n_head_log2);
        return .{
            .scale = scale,
            .max_bias = max_bias,
            .logit_softcap = logit_softcap,
            .n_head_log2 = n_head_log2,
            .m0 = powf(2.0, -(max_bias) / nhl),
            .m1 = powf(2.0, -(max_bias / 2.0) / nhl),
        };
    }
};

/// The shape checks all three `flash_attn_ext` functions open with.
fn checkShapes(dst: *const Tensor) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const v = impl.one(Tensor, dst.src[2]);

    const DK = k.ne[0];
    const DV = v.ne[0];
    const N = q.ne[1];

    impl.assert(dst.ne[0] == DV, "ne0 == DV");
    impl.assert(dst.ne[2] == N, "ne2 == N");

    // input tensor rows must be contiguous
    impl.assert(q.nb[0] == c.ggml_type_size(q.type), "nbq0 == ggml_type_size(q->type)");
    impl.assert(k.nb[0] == c.ggml_type_size(k.type), "nbk0 == ggml_type_size(k->type)");
    impl.assert(v.nb[0] == c.ggml_type_size(v.type), "nbv0 == ggml_type_size(v->type)");

    impl.assert(q.ne[0] == DK, "neq0 == DK");
    impl.assert(k.ne[0] == DK, "nek0 == DK");
    impl.assert(v.ne[0] == DV, "nev0 == DV");

    impl.assert(q.ne[1] == N, "neq1 == N");

    // dst cannot be transposed or permuted
    impl.assert(dst.nb[0] == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(dst.nb[0] <= dst.nb[1], "nb0 <= nb1");
    impl.assert(dst.nb[1] <= dst.nb[2], "nb1 <= nb2");
    impl.assert(dst.nb[2] <= dst.nb[3], "nb2 <= nb3");
}

// -----------------------------------------------------------------------------
// simd_gemm

/// Ports `simd_gemm` (simd-gemm.h:60 @c1d0e7a00), the `GGML_SIMD` arm without
/// SVE: `C[M x N] += A[M x K] * B[K x N]`.
///
/// The C blocks this into 4×16 micro-kernels plus three kinds of remainder,
/// but every one of them computes each `C[i][j]` the same way: start from the
/// stored value and fold in `A[i][kk] * B[kk][j]` for `kk` ascending, one
/// fused multiply-add per step — `vfmaq_f32` in the kernels, and `a += A*B`
/// contracted by clang in the scalar remainders. Elements never meet. So the
/// blocking is a schedule, not a result, and one loop with `kk` outside `j`
/// gives the same bits.
fn simdGemm(C: [*]f32, A: [*]const f32, B: [*]const f32, M: usize, K: usize, N: usize) void {
    for (0..M) |i| {
        const crow = C + i * N;
        for (0..K) |kk| {
            const a = A[i * K + kk];
            const brow = B + kk * N;
            for (0..N) |j| crow[j] = @mulAdd(f32, a, brow[j], crow[j]);
        }
    }
}

// -----------------------------------------------------------------------------
// flash_attn_ext

/// Ports `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
/// (ops.cpp:8475 @c1d0e7a00).
///
/// Parameters:
/// - `ir0`, `ir1`: the query rows to compute.
/// - `ic_start`, `ic_end`: the keys to attend over.
/// - `partials`, `partial_stride`: when non-null, write `[M, S, VKQ]` per row
///   for `reduce_partials` instead of normalising into `dst`.
fn oneChunk(
    params: *const ComputeParams,
    dst: *Tensor,
    ir0: i64,
    ir1: i64,
    ic_start: i64,
    ic_end: i64,
    partials: ?[*]f32,
    partial_stride: i64,
) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const v = impl.one(Tensor, dst.src[2]);
    const mask: ?*Tensor = dst.src[3];
    const sinks: ?*Tensor = dst.src[4];

    const neq1 = q.ne[1];
    const neq2 = q.ne[2];
    const neq3 = q.ne[3];
    const nbq1 = q.nb[1];
    const nbq2 = q.nb[2];
    const nbq3 = q.nb[3];
    const nbk1 = k.nb[1];
    const nbk2 = k.nb[2];
    const nbk3 = k.nb[3];
    const nbv1 = v.nb[1];
    const nbv2 = v.nb[2];
    const nbv3 = v.nb[3];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const nb1 = dst.nb[1];

    const DK = k.ne[0];
    const DV = v.ne[0];

    checkShapes(dst);

    // broadcast factors
    const rk2 = @divTrunc(neq2, k.ne[2]);
    const rk3 = @divTrunc(neq3, k.ne[3]);

    const rv2 = @divTrunc(neq2, v.ne[2]);
    const rv3 = @divTrunc(neq3, v.ne[3]);

    const ap = AttnParams.of(dst, @intCast(neq2));

    const k_vec_dot_type = c.ggml_get_type_traits_cpu(k.type).*.vec_dot_type;
    const q_to_vec_dot = c.ggml_get_type_traits_cpu(k_vec_dot_type).*.from_float;
    const kq_vec_dot = c.ggml_get_type_traits_cpu(k.type).*.vec_dot;
    const v_to_float = c.ggml_get_type_traits(v.type).*.to_float;

    impl.assert(q_to_vec_dot != null, "q_to_vec_dot && \"fattn: unsupported K-type\"");
    impl.assert(v.type == c.GGML_TYPE_F32 or v_to_float != null, "(v->type == GGML_TYPE_F32 || v_to_float) && \"fattn: unsupported V-type\"");

    const ith: usize = @intCast(params.ith);
    const dk: usize = @intCast(DK);
    const dv: usize = @intCast(DV);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // q indices
        const iq3 = @divTrunc(ir, neq2 * neq1);
        const iq2 = @divTrunc(ir - iq3 * neq2 * neq1, neq1);
        const iq1 = ir - iq3 * neq2 * neq1 - iq2 * neq1;

        const h: u32 = @intCast(iq2); // head index
        const slope = alibiSlope(ap.max_bias, h, ap.n_head_log2, ap.m0, ap.m1);

        var S: f32 = 0.0; // sum
        var M: f32 = -inf; // maximum KQ value

        const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
        const VKQ32 = wbase + ith * (1 * dk + 2 * dv + common.cache_line_size_f32); // FP32 VKQ accumulator
        const V32 = VKQ32 + 1 * dv; // (temporary) FP32 V buffer
        const VKQ16: [*]c.ggml_fp16_t = @ptrCast(@alignCast(VKQ32 + 1 * dv)); // (temporary) FP16 VKQ accumulator
        const Q_q: [*]c.ggml_fp16_t = @ptrCast(@alignCast(VKQ32 + 2 * dv)); // (temporary) buffer for Q converted to quantized/FP16

        if (v.type == c.GGML_TYPE_F16) {
            @memset(VKQ16[0..dv], 0);
        } else {
            @memset(VKQ32[0..dv], 0);
        }

        const mp: ?[*]const c.ggml_fp16_t = if (mask) |m|
            at(c.ggml_fp16_t, m.data, off(iq1, m.nb[1]) + off(@rem(iq2, m.ne[2]), m.nb[2]) + off(@rem(iq3, m.ne[3]), m.nb[3]))
        else
            null;

        // k indices
        const ik3 = @divTrunc(iq3, rk3);
        const ik2 = @divTrunc(iq2, rk2);

        // v indices
        const iv3 = @divTrunc(iq3, rv3);
        const iv2 = @divTrunc(iq2, rv2);

        const pq = at(f32, q.data, off(iq1, nbq1) + off(iq2, nbq2) + off(iq3, nbq3));
        q_to_vec_dot.?(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        var ic: i64 = ic_start;
        while (ic < ic_end) : (ic += 1) {
            const mv: f32 = if (mp) |p| slope * impl.fp16ToFp32(p[@intCast(ic)]) else 0.0;
            if (mv == -inf) {
                continue;
            }

            var s: f32 = undefined; // KQ value

            const k_data = at(u8, k.data, off(ic, nbk1) + off(ik2, nbk2) + off(ik3, nbk3));
            kq_vec_dot.?(@intCast(DK), &s, 0, k_data, 0, Q_q, 0, 1);

            s = s * ap.scale; // scale KQ value

            if (ap.logit_softcap != 0.0) {
                s = ap.logit_softcap * tanhf(s);
            }

            s += mv; // apply mask

            const Mold = M;

            var ms: f32 = 1.0; // upon new higher max val, scale VKQ and KQ sum with this value
            var vs: f32 = 1.0; // post-softmax KQ value, expf(s - M)

            const v_data = at(u8, v.data, off(ic, nbv1) + off(iv2, nbv2) + off(iv3, nbv3));

            if (v.type == c.GGML_TYPE_F16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    vec.scale_f16(DV, VKQ16, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                vec.mad_f16(DV, VKQ16, @ptrCast(@alignCast(v_data)), vs);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    vec.scale_f32(DV, VKQ32, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                if (v_to_float) |to_float| {
                    to_float(v_data, V32, DV);
                    vec.mad_f32(DV, VKQ32, V32, vs);
                } else {
                    // V is F32
                    vec.mad_f32(DV, VKQ32, @ptrCast(@alignCast(v_data)), vs);
                }
            }

            // scale and increment sum with partial sum: one expression, fused
            S = @mulAdd(f32, S, ms, vs);
        }

        if (v.type == c.GGML_TYPE_F16) {
            for (0..dv) |d| {
                VKQ32[d] = impl.fp16ToFp32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        if (sinks != null and ic_start == 0) {
            const s = at(f32, sinks.?.data, 0)[h];

            var ms: f32 = 1.0;
            var vs: f32 = 1.0;

            if (s > M) {
                ms = expf(M - s);
                M = s;
                vec.scale_f32(DV, VKQ32, ms);
            } else {
                vs = expf(s - M);
            }

            S = @mulAdd(f32, S, ms, vs);
        }

        if (partials) |p| {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            const partial = p + @as(usize, @intCast(ir * partial_stride));
            partial[0] = M;
            partial[1] = S;
            @memcpy((partial + 2)[0..dv], VKQ32[0..dv]);
        } else {
            // V /= S
            const S_inv: f32 = if (S == 0.0) 0.0 else 1.0 / S;
            vec.scale_f32(DV, VKQ32, S_inv);

            // dst indices
            const j1 = iq1;
            const j2 = iq2;
            const j3 = iq3;

            // permute(0, 2, 1, 3)
            const out = at(u8, dst.data, off(j3 * ne2 * ne1 + j2 + j1 * ne1, nb1));
            @memcpy(out[0..nb1], @as([*]const u8, @ptrCast(VKQ32))[0..nb1]);
        }
    }
}

/// Ports `ggml_compute_forward_flash_attn_ext_tiled` (ops.cpp:8713 @c1d0e7a00).
fn tiled(params: *const ComputeParams, dst: *Tensor, ir0: i64, ir1: i64) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const v = impl.one(Tensor, dst.src[2]);
    const mask: ?*Tensor = dst.src[3];
    const sinks: ?*Tensor = dst.src[4];

    const neq1 = q.ne[1];
    const neq2 = q.ne[2];
    const neq3 = q.ne[3];
    const nbq1 = q.nb[1];
    const nbq2 = q.nb[2];
    const nbq3 = q.nb[3];
    const nek1 = k.ne[1];
    const nbk1 = k.nb[1];
    const nbk2 = k.nb[2];
    const nbk3 = k.nb[3];
    const nbv1 = v.nb[1];
    const nbv2 = v.nb[2];
    const nbv3 = v.nb[3];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const nb1 = dst.nb[1];

    const DK = k.ne[0];
    const DV = v.ne[0];

    checkShapes(dst);

    impl.assert(k.type == v.type, "k->type == v->type");
    const kv_type = k.type;

    // broadcast factors
    const rk2 = @divTrunc(neq2, k.ne[2]);
    const rk3 = @divTrunc(neq3, k.ne[3]);

    const rv2 = @divTrunc(neq2, v.ne[2]);
    const rv3 = @divTrunc(neq3, v.ne[3]);

    const ap = AttnParams.of(dst, @intCast(neq2));

    const ith: usize = @intCast(params.ith);
    const dk: usize = @intCast(DK);
    const dv: usize = @intCast(DV);

    const Q_TILE_SZ = q_tile_sz;
    const KV_TILE_SZ = kv_tile_sz;

    var ir: i64 = ir0;
    while (ir < ir1) {
        // q indices for the start of this tile
        const iq3 = @divTrunc(ir, neq2 * neq1);
        const iq2 = @divTrunc(ir - iq3 * neq2 * neq1, neq1);
        const iq1 = ir - iq3 * neq2 * neq1 - iq2 * neq1;

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        const tile_rows_i: i64 = @min(@as(i64, Q_TILE_SZ), @min(ir1 - ir, neq1 - iq1));
        impl.assert(tile_rows_i > 0, "tile_rows > 0");
        const tile_rows: usize = @intCast(tile_rows_i);

        const h: u32 = @intCast(iq2); // head index
        const slope = alibiSlope(ap.max_bias, h, ap.n_head_log2, ap.m0, ap.m1);

        var S: [Q_TILE_SZ]f32 = undefined;
        var M: [Q_TILE_SZ]f32 = undefined;

        for (0..Q_TILE_SZ) |i| {
            S[i] = 0.0;
            M[i] = -inf;
        }

        // Per-thread scratch layout:
        // Q_q:    Q_TILE_SZ * DK (converted Q tile — F32 for GEMM, KV type for scalar)
        // KQ:     Q_TILE_SZ * KV_TILE_SZ (attention scores in float)
        // mask:   Q_TILE_SZ * KV_TILE_SZ (mask in float)
        // VKQ32:  Q_TILE_SZ * DV (FP32 output accumulator)
        // V32:    KV_TILE_SZ * DV (F32 buffer for V tile)
        // K_f32:  KV_TILE_SZ * DK (F32 buffer for K tile — GEMM path)
        const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
        const base = wbase + ith * (Q_TILE_SZ * dk + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * dv + KV_TILE_SZ * dv + KV_TILE_SZ * dk + common.cache_line_size_f32);

        const Q_q = base;
        const KQ = base + Q_TILE_SZ * dk;
        const mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        const VKQ32 = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        const V32 = VKQ32 + Q_TILE_SZ * dv;
        const K_f32 = V32 + KV_TILE_SZ * dv;

        @memset(VKQ32[0 .. Q_TILE_SZ * dv], 0);
        @memset(mask32[0 .. Q_TILE_SZ * KV_TILE_SZ], 0);

        // k indices
        const ik3 = @divTrunc(iq3, rk3);
        const ik2 = @divTrunc(iq2, rk2);

        // v indices
        const iv3 = @divTrunc(iq3, rv3);
        const iv2 = @divTrunc(iq2, rv2);

        {
            const Q_f32 = Q_q;
            for (0..tile_rows) |tq| {
                const pq = at(f32, q.data, off(iq1 + @as(i64, @intCast(tq)), nbq1) + off(iq2, nbq2) + off(iq3, nbq3));
                @memcpy((Q_f32 + tq * dk)[0..dk], pq[0..dk]);
            }
            for (tile_rows..Q_TILE_SZ) |tq| {
                @memset((Q_f32 + tq * dk)[0..dk], 0);
            }
        }

        @memset(K_f32[0 .. dk * KV_TILE_SZ], 0);
        @memset(V32[0 .. KV_TILE_SZ * dv], 0);

        var ic: i64 = 0;
        while (ic < nek1) : (ic += @intCast(KV_TILE_SZ)) {
            const kv_tile: usize = @intCast(@min(@as(i64, KV_TILE_SZ), nek1 - ic));

            // skip the tile entirely if all the masks are -inf
            if (mask) |m| {
                var can_skip = true;
                for (0..tile_rows) |tq| {
                    const mp_row = at(c.ggml_fp16_t, m.data, off(iq1 + @as(i64, @intCast(tq)), m.nb[1]) + off(@rem(iq2, m.ne[2]), m.nb[2]) + off(@rem(iq3, m.ne[3]), m.nb[3]));
                    for (0..kv_tile) |tk| {
                        mask32[tq * KV_TILE_SZ + tk] = slope * impl.fp16ToFp32(mp_row[@as(usize, @intCast(ic)) + tk]);
                        if (mask32[tq * KV_TILE_SZ + tk] != -inf) {
                            can_skip = false;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    for (kv_tile..KV_TILE_SZ) |tk| {
                        mask32[tq * KV_TILE_SZ + tk] = -inf;
                    }
                }

                if (can_skip) {
                    continue;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            for (0..kv_tile) |tk| {
                const k_off = off(ic + @as(i64, @intCast(tk)), nbk1) + off(ik2, nbk2) + off(ik3, nbk3);
                if (kv_type == c.GGML_TYPE_F16) {
                    const k_f16 = at(c.ggml_fp16_t, k.data, k_off);
                    for (0..dk) |d| {
                        K_f32[d * KV_TILE_SZ + tk] = impl.fp16ToFp32(k_f16[d]);
                    }
                } else {
                    const k_f32_src = at(f32, k.data, k_off);
                    for (0..dk) |d| {
                        K_f32[d * KV_TILE_SZ + tk] = k_f32_src[d];
                    }
                }
            }
            @memset(KQ[0 .. Q_TILE_SZ * KV_TILE_SZ], 0);
            simdGemm(KQ, Q_q, K_f32, Q_TILE_SZ, dk, KV_TILE_SZ);
            vec.scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, ap.scale);

            // Set padded KQ entries to -inf so softmax gives them zero weight
            if (kv_tile < KV_TILE_SZ) {
                for (0..Q_TILE_SZ) |tq| {
                    for (kv_tile..KV_TILE_SZ) |tk| {
                        KQ[tq * KV_TILE_SZ + tk] = -inf;
                    }
                }
            }

            if (ap.logit_softcap != 0.0) {
                vec.tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                vec.scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, ap.logit_softcap);
            }

            if (mask != null) {
                vec.add_f32(@intCast(tile_rows * KV_TILE_SZ), KQ, KQ, mask32);
            }

            var skip = [_]bool{false} ** Q_TILE_SZ;

            for (0..Q_TILE_SZ) |tq| {
                const kq_row = KQ + tq * KV_TILE_SZ;

                var tile_max: f32 = undefined;
                vec.max_f32(KV_TILE_SZ, &tile_max, kq_row);

                if (tile_max == -inf) {
                    skip[tq] = true;
                    continue;
                }

                const Mold = M[tq];
                const Mnew = fmaxf(Mold, tile_max);

                if (Mnew > Mold) {
                    const ms = expf(Mold - Mnew);
                    vec.scale_f32(DV, VKQ32 + tq * dv, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                // `ggml_vec_soft_max_f32` returns `ggml_float`, so `+=`
                // widens `S[tq]`, adds in `double`, and narrows once.
                S[tq] = @floatCast(@as(f64, S[tq]) + vec.soft_max_f32(@intCast(KV_TILE_SZ), kq_row, kq_row, Mnew));
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            for (0..kv_tile) |tk| {
                const v_off = off(ic + @as(i64, @intCast(tk)), nbv1) + off(iv2, nbv2) + off(iv3, nbv3);
                if (kv_type == c.GGML_TYPE_F16) {
                    convert.ggml_cpu_fp16_to_fp32(at(c.ggml_fp16_t, v.data, v_off), V32 + tk * dv, DV);
                } else {
                    @memcpy((V32 + tk * dv)[0..dv], at(f32, v.data, v_off)[0..dv]);
                }
            }
            for (0..Q_TILE_SZ) |tq| {
                if (skip[tq]) {
                    @memset((KQ + tq * KV_TILE_SZ)[0..KV_TILE_SZ], 0);
                }
            }
            simdGemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, dv);
        }

        // sinks (apply only to valid rows in the tile)
        if (sinks) |sk| {
            const s = at(f32, sk.data, 0)[h];

            for (0..tile_rows) |tq| {
                var ms: f32 = 1.0;
                var vs: f32 = 1.0;

                if (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    vec.scale_f32(DV, VKQ32 + tq * dv, ms);
                } else {
                    vs = expf(s - M[tq]);
                }

                // one expression, fused
                S[tq] = @mulAdd(f32, S[tq], ms, vs);
            }
        }

        for (0..tile_rows) |tq| {
            // V /= S
            const S_inv: f32 = if (S[tq] == 0.0) 0.0 else 1.0 / S[tq];
            vec.scale_f32(DV, VKQ32 + tq * dv, S_inv);

            // dst indices
            const j1 = iq1 + @as(i64, @intCast(tq));
            const j2 = iq2;
            const j3 = iq3;

            // permute(0, 2, 1, 3)
            const out = at(u8, dst.data, off(j3 * ne2 * ne1 + j2 + j1 * ne1, nb1));
            @memcpy(out[0..nb1], @as([*]const u8, @ptrCast(VKQ32 + tq * dv))[0..nb1]);
        }

        ir += tile_rows_i;
    }
}

/// Ports `ggml_flash_attn_ext_reduce_partials` (ops.cpp:9003 @c1d0e7a00).
///
/// Combines the per-chunk `[M, S, VKQ]` the split-KV path wrote. Partials
/// layout in `wdata`: `[n_q_heads][n_chunks][2 + DV]`.
fn reducePartials(params: *const ComputeParams, dst: *Tensor, n_chunks: i64, chunk_size: i64) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const v = impl.one(Tensor, dst.src[2]);

    const DK = k.ne[0];
    const DV = v.ne[0];
    const nek1 = k.ne[1];
    const n_q_heads = q.ne[2];

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));

    const wdata_per_thread = DK + 2 * DV + @as(i64, common.cache_line_size_f32);
    const thread_wdata = wbase + @as(usize, @intCast(ith * wdata_per_thread));

    const partials_offset = nth * (DK + 2 * DV + @as(i64, common.cache_line_size_f32));
    const partial_size = 2 + DV;
    const partials_base = wbase + @as(usize, @intCast(partials_offset));

    // Output layout
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const nb1 = dst.nb[1];

    const dv: usize = @intCast(DV);

    // Each thread reduces a subset of query heads
    var q_head: i64 = ith;
    while (q_head < n_q_heads) : (q_head += nth) {
        var M_final: f32 = -inf;
        var S_final: f32 = 0.0;
        const VKQ_final = thread_wdata;
        @memset(VKQ_final[0..dv], 0);

        // Combine partials from all chunks
        var chunk_idx: i64 = 0;
        while (chunk_idx < n_chunks) : (chunk_idx += 1) {
            const ic_start = chunk_idx * chunk_size;
            if (ic_start >= nek1) continue;

            const partial = partials_base + @as(usize, @intCast((q_head * n_chunks + chunk_idx) * partial_size));
            const M_chunk = partial[0];
            const S_chunk = partial[1];
            const VKQ_chunk = partial + 2;

            if (S_chunk == 0.0) continue;

            const M_new = fmaxf(M_final, M_chunk);
            const scale_old = expf(M_final - M_new);
            const scale_new = expf(M_chunk - M_new);

            // `a*b + c*d`: clang fuses the left product.
            for (0..dv) |d| {
                VKQ_final[d] = @mulAdd(f32, VKQ_final[d], scale_old, VKQ_chunk[d] * scale_new);
            }
            S_final = @mulAdd(f32, S_final, scale_old, S_chunk * scale_new);
            M_final = M_new;
        }

        // Normalize and write to output
        if (S_final != 0.0) {
            const S_inv = 1.0 / S_final;
            vec.scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        const out = at(u8, dst.data, off(0 * ne2 * ne1 + q_head + 0 * ne1, nb1));
        @memcpy(out[0..nb1], @as([*]const u8, @ptrCast(VKQ_final))[0..nb1]);
    }
}

/// Ports `ggml_compute_forward_flash_attn_ext_f16` (ops.cpp:9073 @c1d0e7a00),
/// the `GGML_SIMD` arm without SVE for the tiled-path guard.
fn flashAttnExtF16(params: *const ComputeParams, dst: *Tensor) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const v = impl.one(Tensor, dst.src[2]);

    const neq1 = q.ne[1];
    const neq2 = q.ne[2];
    const neq3 = q.ne[3];
    const nek1 = k.ne[1];

    const DK = k.ne[0];
    const DV = v.ne[0];

    checkShapes(dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const tp = params.threadpool.?;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    const use_ref = params.use_ref;

    const kv_is_f32_or_f16 = (k.type == c.GGML_TYPE_F32 or k.type == c.GGML_TYPE_F16);
    const use_split_kv_path = !use_ref and (neq1 == 1 and neq3 == 1) and kv_is_f32_or_f16 and (k.type == v.type) and q.type == c.GGML_TYPE_F32 and nek1 >= 512;

    if (use_split_kv_path) {
        const chunk_size = @divTrunc(nek1 + nth - 1, nth);

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        const partial_size = 2 + DV;
        const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
        const partials_base = wbase + @as(usize, @intCast(nth * (DK + 2 * DV + @as(i64, common.cache_line_size_f32))));

        const ic_start = ith * chunk_size;
        const ic_end = @min(ic_start + chunk_size, nek1);

        const partial_stride = nth * partial_size;
        const chunk_partials = partials_base + @as(usize, @intCast(ith * partial_size));

        if (ic_start < nek1) {
            var q_head: i64 = 0;
            while (q_head < neq2) : (q_head += 1) {
                oneChunk(params, dst, q_head, q_head + 1, ic_start, ic_end, chunk_partials, partial_stride);
            }
        } else {
            var q_head: i64 = 0;
            while (q_head < neq2) : (q_head += 1) {
                const q_partials = chunk_partials + @as(usize, @intCast(q_head * partial_stride));
                q_partials[0] = -inf; // M
                q_partials[1] = 0.0; // S
            }
        }

        threading.ggml_barrier(tp);
        reducePartials(params, dst, nth, chunk_size);
    } else {
        // total rows in q
        const nr = neq1 * neq2 * neq3;

        // disable for NUMA
        const disable_chunking = threading.ggml_is_numa();

        // 4x chunks per thread
        const nth_scaled = nth * 4;
        const chunk_size = @divTrunc(nr + nth_scaled - 1, nth_scaled);
        var nchunk = @divTrunc(nr + chunk_size - 1, chunk_size);

        if (nth == 1 or nchunk < nth or disable_chunking) {
            nchunk = nth;
        }

        if (ith == 0) {
            threading.ggml_threadpool_chunk_set(tp, @intCast(nth));
        }

        threading.ggml_barrier(tp);

        const dr = @divTrunc(nr + nchunk - 1, nchunk);

        var use_tiled = !use_ref and
            (q.type == c.GGML_TYPE_F32 and
                kv_is_f32_or_f16 and
                k.type == v.type and
                neq1 >= @as(i64, q_tile_sz));
        use_tiled = use_tiled and (@rem(DV, f32_epr) == 0);

        var current_chunk: i64 = ith;

        while (current_chunk < nchunk) {
            const ir0 = dr * current_chunk;
            const ir1 = @min(ir0 + dr, nr);

            if (use_tiled) {
                tiled(params, dst, ir0, ir1);
            } else {
                oneChunk(params, dst, ir0, ir1, 0, nek1, null, 0);
            }

            current_chunk = threading.ggml_threadpool_chunk_add(tp, 1);
        }
    }
}

/// Ports `ggml_compute_forward_flash_attn_ext` (ops.cpp:9209 @c1d0e7a00).
pub export fn ggml_compute_forward_flash_attn_ext(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    switch (impl.getOpParamsI32(dst, 3)) {
        c.GGML_PREC_DEFAULT, c.GGML_PREC_F32 => {
            // uses F32 accumulators
            flashAttnExtF16(params, dst);
        },
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// flash_attn_back

/// Ports `ggml_compute_forward_flash_attn_back_f32` (ops.cpp:9228 @c1d0e7a00),
/// the arm without `GGML_SOFT_MAX_ACCELERATE`, which nothing defines (it is
/// commented out at ggml.c:327).
fn flashAttnBackF32(params: *const ComputeParams, masked: bool, dst: *Tensor) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const v = impl.one(Tensor, dst.src[2]);
    const d = impl.one(Tensor, dst.src[3]);

    const neq0 = q.ne[0];
    const neq1 = q.ne[1];
    const neq2 = q.ne[2];
    const nbq1 = q.nb[1];
    const nbq2 = q.nb[2];
    const nbq3 = q.nb[3];
    const nek0 = k.ne[0];
    const nek1 = k.ne[1];
    const nek2 = k.ne[2];
    const nek3 = k.ne[3];
    const nbk1 = k.nb[1];
    const nbk2 = k.nb[2];
    const nbk3 = k.nb[3];
    const nev0 = v.ne[0];
    const nev1 = v.ne[1];
    const nbv1 = v.nb[1];
    const nbv2 = v.nb[2];
    const nbv3 = v.nb[3];
    const ned0 = d.ne[0];
    const ned1 = d.ne[1];
    const nbd0 = d.nb[0];
    const nbd1 = d.nb[1];
    const nbd2 = d.nb[2];
    const nbd3 = d.nb[3];
    const nb0 = dst.nb[0];

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const D = neq0;
    const N = neq1;
    const P = nek1 - N;
    const M = P + N;

    const Mup = up(M, soft_max_unroll);
    const mxDM = @max(D, Mup);

    impl.assert(P >= 0, "P >= 0");

    impl.assert(q.nb[0] == @sizeOf(f32), "nbq0 == sizeof(float)");
    impl.assert(k.nb[0] == @sizeOf(f32), "nbk0 == sizeof(float)");
    impl.assert(v.nb[0] == @sizeOf(f32), "nbv0 == sizeof(float)");

    impl.assert(neq0 == D, "neq0 == D");
    impl.assert(nek0 == D, "nek0 == D");
    impl.assert(nev1 == D, "nev1 == D");
    impl.assert(ned0 == D, "ned0 == D");

    impl.assert(neq1 == N, "neq1 == N");
    impl.assert(nek1 == N + P, "nek1 == N + P");
    impl.assert(nev1 == D, "nev1 == D");
    impl.assert(ned1 == N, "ned1 == N");

    // dst cannot be transposed or permuted
    impl.assert(nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(nb0 <= dst.nb[1], "nb0 <= nb1");
    impl.assert(dst.nb[1] <= dst.nb[2], "nb1 <= nb2");
    impl.assert(dst.nb[2] <= dst.nb[3], "nb2 <= nb3");

    if (ith == 0) {
        const n = nb0 * @as(usize, @intCast(dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]));
        @memset(at(u8, dst.data, 0)[0..n], 0);
    }
    threading.ggml_barrier(params.threadpool.?);

    const elem_q: usize = @intCast(c.ggml_nelements(q));
    const elem_k: usize = @intCast(c.ggml_nelements(k));

    const result_type = dst.type;
    impl.assert(c.ggml_blck_size(result_type) == 1, "ggml_blck_size(result_type) == 1");
    const tsize = c.ggml_type_size(result_type);

    const offs_q: usize = 0;
    const offs_k = offs_q + impl.pad(elem_q * tsize, c.GGML_MEM_ALIGN);
    const offs_v = offs_k + impl.pad(elem_k * tsize, c.GGML_MEM_ALIGN);

    const grad_q = at(u8, dst.data, 0);
    const grad_k = at(u8, dst.data, offs_k);
    const grad_v = at(u8, dst.data, offs_v);

    const unq0: usize = @intCast(neq0);
    const unq1: usize = @intCast(neq1);
    const unq2: usize = @intCast(neq2);
    const unk0: usize = @intCast(nek0);
    const unk1: usize = @intCast(nek1);
    const unv0: usize = @intCast(nev0);
    const unv1: usize = @intCast(nev1);

    const nbgq1 = nb0 * unq0;
    const nbgq2 = nb0 * unq0 * unq1;
    const nbgq3 = nb0 * unq0 * unq1 * unq2;

    const nbgk1 = nb0 * unk0;
    const nbgk2 = nb0 * unk0 * unk1;
    const nbgk3 = nb0 * unk0 * unk1 * unq2;

    const nbgv1 = nb0 * unv0;
    const nbgv2 = nb0 * unv0 * unv1;
    const nbgv3 = nb0 * unv0 * unv1 * unq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    const nr = nek2 * nek3;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const scale: f32 = 1.0 / sqrtf(@floatFromInt(D));

    // how often k2 (and v2) is repeated in q2
    const nrep = @divTrunc(neq2, nek2);

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const slot: usize = @intCast(mxDM + @as(i64, common.cache_line_size_f32));

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // q indices
        const ik3 = @divTrunc(ir, nek2);
        const ik2 = ir - ik3 * nek2;

        const iq3 = ik3;
        const id3 = ik3;
        const iv3 = ik3;
        const iv2 = ik2;

        var irep: i64 = 0;
        while (irep < nrep) : (irep += 1) {
            const iq2 = ik2 + irep * nek2;
            const id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            var iq1: i64 = 0;
            while (iq1 < neq1) : (iq1 += 1) {
                const id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                const S = wbase + @as(usize, @intCast(ith)) * 2 * slot + 0 * slot;
                const SM = wbase + @as(usize, @intCast(ith)) * 2 * slot + 1 * slot;

                var i: i64 = M;
                while (i < Mup) : (i += 1) {
                    S[@intCast(i)] = -inf;
                }

                const masked_begin: i64 = if (masked) (P + iq1 + 1) else M;
                var ic: i64 = 0;
                while (ic < masked_begin) : (ic += 1) {
                    // k indices
                    const ik1 = ic;

                    // S indices
                    const j1 = ik1;

                    vec.dot_f32(
                        @intCast(neq0),
                        &S[@intCast(j1)],
                        0,
                        at(f32, k.data, off(ik1, nbk1) + off(ik2, nbk2) + off(ik3, nbk3)),
                        0,
                        at(f32, q.data, off(iq1, nbq1) + off(iq2, nbq2) + off(iq3, nbq3)),
                        0,
                        1,
                    );
                }

                // scale
                vec.scale_f32(masked_begin, S, scale);

                i = masked_begin;
                while (i < M) : (i += 1) {
                    S[@intCast(i)] = -inf;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    var max: f32 = -inf;
                    vec.max_f32(masked_begin, &max, S);

                    var sum: f64 = vec.soft_max_f32(@intCast(Mup), SM, S, max);

                    sum = 1.0 / sum;
                    vec.scale_f32(masked_begin, SM, @floatCast(sum));
                }

                // S = gradSM = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // exclude known future zero S[..] values from operation
                vec.set_f32(masked_begin, S, 0);
                ic = 0;
                while (ic < D) : (ic += 1) {
                    vec.mad_f32(
                        masked_begin,
                        S,
                        at(f32, v.data, off(ic, nbv1) + off(iv2, nbv2) + off(iv3, nbv3)),
                        at(f32, d.data, off(ic, nbd0) + off(id1, nbd1) + off(id2, nbd2) + off(id3, nbd3))[0],
                    );
                }

                // S = SM * (S - dot(SM, S))
                var dot_SM_gradSM: f32 = 0;
                vec.dot_f32(@intCast(masked_begin), &dot_SM_gradSM, 0, SM, 0, S, 0, 1);
                vec.acc1_f32(M, S, -dot_SM_gradSM);
                vec.mul_f32(masked_begin, S, S, SM);

                // S = diag_mask_zero(S, P) * scale
                // already done by above ggml_vec_set_f32

                // exclude known zero S[..] values from operation
                vec.scale_f32(masked_begin, S, scale);

                // grad[q][:D,iq1,iq2,iq3] += S @ kcur
                // exclude known zero S[..] values from loop
                ic = 0;
                while (ic < masked_begin) : (ic += 1) {
                    vec.mad_f32(
                        D,
                        @ptrCast(@alignCast(grad_q + off(iq1, nbgq1) + off(iq2, nbgq2) + off(iq3, nbgq3))),
                        at(f32, k.data, off(ic, nbk1) + off(ik2, nbk2) + off(ik3, nbk3)),
                        S[@intCast(ic)],
                    );
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // exclude known zero S[..] values from loop
                ic = 0;
                while (ic < masked_begin) : (ic += 1) {
                    vec.mad_f32(
                        D,
                        @ptrCast(@alignCast(grad_k + off(ic, nbgk1) + off(ik2, nbgk2) + off(ik3, nbgk3))),
                        at(f32, q.data, off(iq1, nbq1) + off(iq2, nbq2) + off(iq3, nbq3)),
                        S[@intCast(ic)],
                    );
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T @ SM
                // exclude known zero SM[..] values from mad
                ic = 0;
                while (ic < D) : (ic += 1) {
                    vec.mad_f32(
                        masked_begin,
                        @ptrCast(@alignCast(grad_v + off(ic, nbgv1) + off(iv2, nbgv2) + off(iv3, nbgv3))),
                        SM,
                        at(f32, d.data, off(ic, nbd0) + off(id1, nbd1) + off(id2, nbd2) + off(id3, nbd3))[0],
                    );
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_flash_attn_back` (ops.cpp:9543 @c1d0e7a00).
pub export fn ggml_compute_forward_flash_attn_back(params: *const ComputeParams, masked: bool, dst: *Tensor) callconv(.c) void {
    const q = impl.one(Tensor, dst.src[0]);

    switch (q.type) {
        c.GGML_TYPE_F32 => flashAttnBackF32(params, masked, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// lightning_indexer

/// Ports `ggml_compute_forward_lightning_indexer` (ops.cpp:11941 @c1d0e7a00).
///
/// `score += MAX(qk, 0.0f) * w_row[h]` is one expression — the ternary the
/// macro expands to is the product's left operand, not a barrier — so each
/// head's contribution is fused.
pub export fn ggml_compute_forward_lightning_indexer(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const w = impl.one(Tensor, dst.src[2]); // weights
    const m = impl.one(Tensor, dst.src[3]); // mask

    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");
    impl.assert(q.type == c.GGML_TYPE_F32, "q->type == GGML_TYPE_F32");
    impl.assert(w.type == c.GGML_TYPE_F32, "w->type == GGML_TYPE_F32");
    impl.assert(m.type == c.GGML_TYPE_F16, "m->type == GGML_TYPE_F16");

    impl.assert(dst.nb[0] == c.ggml_type_size(dst.type), "nb0 == ggml_type_size(dst->type)");
    impl.assert(q.nb[0] == c.ggml_type_size(q.type), "nbq0 == ggml_type_size(q->type)");
    impl.assert(k.nb[0] == c.ggml_type_size(k.type), "nbk0 == ggml_type_size(k->type)");
    impl.assert(w.nb[0] == c.ggml_type_size(w.type), "nbw0 == ggml_type_size(w->type)");
    impl.assert(m.nb[0] == c.ggml_type_size(m.type), "nbm0 == ggml_type_size(m->type)");

    const n_embd = q.ne[0];
    const n_head = q.ne[1];
    const n_tokens = q.ne[2];
    const n_stream = q.ne[3];
    const n_kv = k.ne[2];

    const k_to_float = c.ggml_get_type_traits(k.type).*.to_float;
    impl.assert(k.type == c.GGML_TYPE_F32 or k_to_float != null, "(k->type == GGML_TYPE_F32 || k_to_float) && \"lightning indexer: unsupported K-type\"");

    const nr = n_kv;
    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // (temporary) buffer for K converted to float
    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    var k_row_f32 = wbase + @as(usize, @intCast(ith * (1 * n_embd + @as(i64, common.cache_line_size_f32))));

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    var s: i64 = 0;
    while (s < n_stream) : (s += 1) {
        var t: i64 = 0;
        while (t < n_tokens) : (t += 1) {
            const w_row = at(f32, w.data, off(t, w.nb[1]) + off(s, w.nb[3]));
            const m_row = at(c.ggml_fp16_t, m.data, off(t, m.nb[1]) + off(@rem(s, m.ne[3]), m.nb[3]));
            const dst_row = at(f32, dst.data, off(t, dst.nb[1]) + off(s, dst.nb[3]));
            var ik: i64 = ir0;
            while (ik < ir1) : (ik += 1) {
                const k_row = at(u8, k.data, off(ik, k.nb[2]) + off(s, k.nb[3]));
                if (k_to_float) |to_float| {
                    to_float(k_row, k_row_f32, n_embd);
                } else {
                    k_row_f32 = @ptrCast(@alignCast(k_row));
                }
                var score: f32 = 0.0;
                var h: i64 = 0;
                while (h < n_head) : (h += 1) {
                    // dot product of q and k for head h
                    var qk: f32 = 0.0;
                    const q_row = at(f32, q.data, off(h, q.nb[1]) + off(t, q.nb[2]) + off(s, q.nb[3]));
                    vec.dot_f32(@intCast(n_embd), &qk, 0, q_row, 0, k_row_f32, 0, 1);
                    // ReLU and weights (prescaled), fused into the sum
                    score = @mulAdd(f32, if (qk > 0.0) qk else 0.0, w_row[@intCast(h)], score);
                }
                // apply mask
                dst_row[@intCast(ik)] = score + impl.fp16ToFp32(m_row[@intCast(ik)]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "simd_gemm accumulates into C with one fused step per k" {
    // 1 + 2^-12 squared needs 25 significand bits; only a fused step keeps
    // the 2^-24 term once the addend has cancelled the leading 1.
    const e: f32 = 1.0 + 0x1p-12;
    var C = [_]f32{-(1.0 + 0x1p-11)};
    simdGemm(&C, &[_]f32{e}, &[_]f32{e}, 1, 1, 1);
    try std.testing.expectEqual(@as(f32, 0x1p-24), C[0]);
}

test "simd_gemm is a plain matrix product" {
    // [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50], added to C = 1.
    var C = [_]f32{1} ** 4;
    simdGemm(&C, &[_]f32{ 1, 2, 3, 4 }, &[_]f32{ 5, 6, 7, 8 }, 2, 2, 2);
    try std.testing.expectEqual([4]f32{ 20, 23, 44, 51 }, C);
}

test "ggml_up rounds to the next multiple" {
    try std.testing.expectEqual(@as(i64, 8), up(5, 4));
    try std.testing.expectEqual(@as(i64, 8), up(8, 4));
}
