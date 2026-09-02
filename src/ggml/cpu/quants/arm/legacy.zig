//! NEON dot products for the block-per-32 formats and the two float4
//! codebooks.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (v0.3.0,
//! `c1d0e7a00`), the `__ARM_NEON` arms of the dot products at lines 140, 222,
//! 297, 590, 749, 810, 920, 1032, 1150 and 4196. Each function names the C it
//! replaces and the line it began at.
//!
//! Only the `__ARM_NEON` arm of each is here. Every one of these functions has
//! an `__ARM_FEATURE_SVE` or `__ARM_FEATURE_MATMUL_INT8` variant above it that
//! this target does not compile -- `q4_0` alone has 240 such lines against 50
//! live ones.
//!
//! # These are live, and byte-exact against the C
//!
//! `type_traits_cpu` points at them, so every quantized `mul_mat` on the CPU
//! runs one. `arm/golden.zig` holds the exact `f32` bits the C produces, and
//! `testing.check` compares on bits: `test-backend-ops` and token parity reach
//! these kernels but would let a last-bit difference through.
//!
//! # The shape they share
//!
//! Two blocks per iteration into two accumulators, then `vaddvq_f32` on each
//! and a scalar tail for a final odd block. The two accumulators are not an
//! optimisation detail to be tidied away: `sumv0` and `sumv1` are summed
//! separately and added at the end, and folding them into one changes the
//! order of the float additions and therefore the answer.

const std = @import("std");
const impl = @import("../../../impl.zig");
const convert = @import("../../convert.zig");
const blocks = @import("../../../quants/blocks.zig");
const neon = @import("neon.zig");
const c = impl.c;

const i8x16 = neon.i8x16;
const u8x16 = neon.u8x16;
const i32x4 = neon.i32x4;
const f32x4 = neon.f32x4;

inline fn f(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

inline fn as(comptime Block: type, p: ?*const anyopaque) [*]const Block {
    return @ptrCast(@alignCast(p.?));
}

/// The four-bit mask and the bias of eight that `q4_0` and its relatives use.
const m4b: u8x16 = @splat(0x0F);
const s8b: i8x16 = @splat(8);

/// Ports `ggml_vdotq_s32(vdupq_n_s32(0), a, b)` chained twice, the idiom that
/// opens almost every kernel here.
inline fn dot2(a0: i8x16, b0: i8x16, a1: i8x16, b1: i8x16) i32x4 {
    const zero: i32x4 = @splat(0);
    return neon.dotq_s32(neon.dotq_s32(zero, a0, b0), a1, b1);
}

// -----------------------------------------------------------------------------
// q1_0

/// Ports `table_b2b_0` (arch/arm/quants.c:37 @c1d0e7a00): a byte's eight bits expanded to
/// eight bytes, each `0x10` where the bit is set.
///
/// The C builds this with an eight-deep macro (`B8`), which expands to 256
/// literals. Generated here instead, and checked: byte `j` of entry `i` is
/// `((i >> j) & 1) ? 0x10 : 0x00`, verified against the macro's output for all
/// 256 entries.
const table_b2b_0 = blk: {
    @setEvalBranchQuota(8000);
    var t: [256]u64 = undefined;
    for (0..256) |i| {
        var bytes: [8]u8 = undefined;
        for (0..8) |j| bytes[j] = if ((i >> j) & 1 != 0) 0x10 else 0x00;
        t[i] = @bitCast(bytes);
    }
    break :blk t;
};

/// Ports `table_b2b_1` (arch/arm/quants.c:38 @c1d0e7a00): the complement of
/// `table_b2b_0`, `0x10` where the bit is **clear**.
///
/// `q5_0` uses it to fold the fifth bit and the bias of 16 into one subtract.
/// The weight is `(nibble | bit << 4) - 16`, and `nibble - (!bit) * 16` is the
/// same number: 16 comes off when the bit is clear and nothing when it is set.
/// `q5_1`, which has no bias, uses `table_b2b_0` and an `or` instead.
const table_b2b_1 = blk: {
    @setEvalBranchQuota(8000);
    var t: [256]u64 = undefined;
    for (0..256) |i| {
        var bytes: [8]u8 = undefined;
        for (0..8) |j| bytes[j] = if ((i >> j) & 1 != 0) 0x00 else 0x10;
        t[i] = @bitCast(bytes);
    }
    break :blk t;
};

/// Expands the 32 bits of a `q5_0`/`q5_1` block's `qh` into two `i8x16`
/// vectors through one of the two tables.
///
/// Parameters:
/// - `table`: `table_b2b_0` or `table_b2b_1`.
/// - `qh`: the four packed bytes of high bits.
///
/// Return: `{ low 16 lanes, high 16 lanes }`.
inline fn expandQh(comptime table: *const [256]u64, qh: u32) [2]i8x16 {
    var tmp: [4]u64 = undefined;
    tmp[0] = table[(qh >> 0) & 0xFF];
    tmp[1] = table[(qh >> 8) & 0xFF];
    tmp[2] = table[(qh >> 16) & 0xFF];
    tmp[3] = table[qh >> 24];
    return .{
        neon.load(i8x16, @ptrCast(&tmp[0])),
        neon.load(i8x16, @ptrCast(&tmp[2])),
    };
}

/// Ports `ggml_vec_dot_q1_0_q8_0` (arch/arm/quants.c:140 @c1d0e7a00).
///
/// One-bit weights: each bit selects `+y` or `-y`. The sign vector is built by
/// expanding four bytes of bits through `table_b2b_0`, shifting the `0x10`
/// down to 1, then mapping `{0,1}` to `{-1,+1}` with `2*v - 1`.
pub export fn ggml_vec_dot_q1_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK1_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q1_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumv: f32x4 = @splat(0.0);

    for (0..@intCast(nb)) |i| {
        const d0 = f(x[i].d);

        // A q1_0 block covers 128 elements, so four q8_0 blocks, each with its
        // own delta.
        for (0..4) |k| {
            const yb = &y[i * 4 + k];
            const d1 = f(yb.d);

            const bits = x[i].qs[k * 4 ..];

            const y0 = neon.loadFrom(i8x16, &yb.qs);
            const y1 = neon.loadFrom(i8x16, yb.qs[16..]);

            var sgn: [4]neon.i8x8 = undefined;
            const one: neon.i8x8 = @splat(1);
            inline for (0..4) |b| {
                const e = neon.create_u8(table_b2b_0[bits[b]]);
                const v: neon.i8x8 = @bitCast(neon.shrN(e, 4));
                // `2*v - 1` maps {0,1} to {-1,+1}.
                sgn[b] = neon.sub(neon.add(v, v), one);
            }

            const signs0 = neon.combine(sgn[0], sgn[1]);
            const signs1 = neon.combine(sgn[2], sgn[3]);

            const p1 = dot2(signs0, y0, signs1, y1);

            sumv = neon.mla_n_f32(sumv, neon.cvt_f32_s32(p1), d0 * d1);
        }
    }

    s[0] = neon.addvq_f32(sumv);
}

// -----------------------------------------------------------------------------
// q2_0

/// Ports `tbl_idx_lo`, `tbl_idx_hi` and `shift_vals` (arch/arm/quants.c:243 @c1d0e7a00).
///
/// Each byte of the packed row holds four 2-bit weights. Replicating the byte
/// four times and shifting by `{0,-2,-4,-6}` puts a different field in the low
/// bits of each lane -- which is why `shlq` has to honour negative amounts.
const tbl_idx_lo: u8x16 = .{ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
const tbl_idx_hi: u8x16 = .{ 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7 };
const shift_vals: i8x16 = .{ 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6 };

/// Ports `ggml_vec_dot_q2_0_q8_0` (arch/arm/quants.c:222 @c1d0e7a00).
pub export fn ggml_vec_dot_q2_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK2_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q2_0, vx);
    const y = as(blocks.Q8_0, vy);

    const mask2: u8x16 = @splat(0x03);
    const one: i8x16 = @splat(1);

    var sumv: f32x4 = @splat(0.0);

    for (0..@intCast(nb)) |i| {
        const d0 = f(x[i].d);

        // One q2_0 block is 64 weights, so two q8_0 blocks.
        for (0..2) |k| {
            const yb = &y[i * 2 + k];
            const d1 = f(yb.d);

            const raw = neon.loadFrom(neon.u8x8, x[i].qs[k * 8 ..].ptr);
            const raw16 = neon.combine(raw, raw);

            const bytes0 = neon.qtbl1q_u8(raw16, tbl_idx_lo);
            const qv0 = neon.sub(
                @as(i8x16, @bitCast(neon.@"and"(neon.shlq(bytes0, shift_vals), mask2))),
                one,
            );

            const bytes1 = neon.qtbl1q_u8(raw16, tbl_idx_hi);
            const qv1 = neon.sub(
                @as(i8x16, @bitCast(neon.@"and"(neon.shlq(bytes1, shift_vals), mask2))),
                one,
            );

            const y0 = neon.loadFrom(i8x16, &yb.qs);
            const y1 = neon.loadFrom(i8x16, yb.qs[16..]);

            const p1 = dot2(qv0, y0, qv1, y1);

            sumv = neon.mla_n_f32(sumv, neon.cvt_f32_s32(p1), d0 * d1);
        }
    }

    s[0] = neon.addvq_f32(sumv);
}

// -----------------------------------------------------------------------------
// q4_0 and q4_1

/// Ports `ggml_vec_dot_q4_0_q8_0` (arch/arm/quants.c:297 @c1d0e7a00), the `__ARM_NEON`
/// arm at line 527.
pub export fn ggml_vec_dot_q4_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumv0: f32x4 = @splat(0.0);
    var sumv1: f32x4 = @splat(0.0);

    var ib: usize = 0;
    while (ib + 1 < nb) : (ib += 2) {
        const x0 = &x[ib + 0];
        const x1 = &x[ib + 1];
        const y0 = &y[ib + 0];
        const y1 = &y[ib + 1];

        const v0_0 = neon.loadFrom(u8x16, &x0.qs);
        const v0_1 = neon.loadFrom(u8x16, &x1.qs);

        const v0_0l: i8x16 = @bitCast(neon.@"and"(v0_0, m4b));
        const v0_0h: i8x16 = @bitCast(neon.shrN(v0_0, 4));
        const v0_1l: i8x16 = @bitCast(neon.@"and"(v0_1, m4b));
        const v0_1h: i8x16 = @bitCast(neon.shrN(v0_1, 4));

        const v0_0ls = neon.sub(v0_0l, s8b);
        const v0_0hs = neon.sub(v0_0h, s8b);
        const v0_1ls = neon.sub(v0_1l, s8b);
        const v0_1hs = neon.sub(v0_1h, s8b);

        const v1_0l = neon.loadFrom(i8x16, &y0.qs);
        const v1_0h = neon.loadFrom(i8x16, y0.qs[16..]);
        const v1_1l = neon.loadFrom(i8x16, &y1.qs);
        const v1_1h = neon.loadFrom(i8x16, y1.qs[16..]);

        const p_0 = dot2(v0_0ls, v1_0l, v0_0hs, v1_0h);
        const p_1 = dot2(v0_1ls, v1_1l, v0_1hs, v1_1h);

        sumv0 = neon.mla_n_f32(sumv0, neon.cvt_f32_s32(p_0), f(x0.d) * f(y0.d));
        sumv1 = neon.mla_n_f32(sumv1, neon.cvt_f32_s32(p_1), f(x1.d) * f(y1.d));
    }

    // The two accumulators are reduced separately and then added, which is not
    // the same as reducing one combined accumulator.
    var sumf = neon.addvq_f32(sumv0) + neon.addvq_f32(sumv1);

    // The odd final block, scalar. Identical to the body of
    // `ggml_vec_dot_q4_0_q8_0_generic`, but it accumulates into a `sumf` that
    // already holds the vector part, so it cannot delegate.
    while (ib < nb) : (ib += 1) {
        var sumi0: i32 = 0;
        var sumi1: i32 = 0;
        for (0..qk / 2) |j| {
            const v0: i32 = @as(i32, x[ib].qs[j] & 0x0F) - 8;
            const v1: i32 = @as(i32, x[ib].qs[j] >> 4) - 8;
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + qk / 2];
        }
        const sumi = sumi0 + sumi1;
        sumf += @as(f32, @floatFromInt(sumi)) * f(x[ib].d) * f(y[ib].d);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q4_1_q8_1` (arch/arm/quants.c:590 @c1d0e7a00).
///
/// Like `q4_0` without the bias of eight, plus the per-block minimum times the
/// right operand's sum. `summs` accumulates as a scalar across the whole
/// vector loop and is added last, after both accumulators.
pub export fn ggml_vec_dot_q4_1_q8_1(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_1;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_1, vx);
    const y = as(blocks.Q8_1, vy);

    var sumv0: f32x4 = @splat(0.0);
    var sumv1: f32x4 = @splat(0.0);
    var summs: f32 = 0;

    var ib: usize = 0;
    while (ib + 1 < nb) : (ib += 2) {
        const x0 = &x[ib + 0];
        const x1 = &x[ib + 1];
        const y0 = &y[ib + 0];
        const y1 = &y[ib + 1];

        summs += f(x0.m) * f(y0.s) + f(x1.m) * f(y1.s);

        const v0_0 = neon.loadFrom(u8x16, &x0.qs);
        const v0_1 = neon.loadFrom(u8x16, &x1.qs);

        const v0_0l: i8x16 = @bitCast(neon.@"and"(v0_0, m4b));
        const v0_0h: i8x16 = @bitCast(neon.shrN(v0_0, 4));
        const v0_1l: i8x16 = @bitCast(neon.@"and"(v0_1, m4b));
        const v0_1h: i8x16 = @bitCast(neon.shrN(v0_1, 4));

        const v1_0l = neon.loadFrom(i8x16, &y0.qs);
        const v1_0h = neon.loadFrom(i8x16, y0.qs[16..]);
        const v1_1l = neon.loadFrom(i8x16, &y1.qs);
        const v1_1h = neon.loadFrom(i8x16, y1.qs[16..]);

        const p_0 = dot2(v0_0l, v1_0l, v0_0h, v1_0h);
        const p_1 = dot2(v0_1l, v1_1l, v0_1h, v1_1h);

        sumv0 = neon.mla_n_f32(sumv0, neon.cvt_f32_s32(p_0), f(x0.d) * f(y0.d));
        sumv1 = neon.mla_n_f32(sumv1, neon.cvt_f32_s32(p_1), f(x1.d) * f(y1.d));
    }

    var sumf = neon.addvq_f32(sumv0) + neon.addvq_f32(sumv1) + summs;

    while (ib < nb) : (ib += 1) {
        var sumi0: i32 = 0;
        var sumi1: i32 = 0;
        for (0..qk / 2) |j| {
            const v0: i32 = x[ib].qs[j] & 0x0F;
            const v1: i32 = x[ib].qs[j] >> 4;
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + qk / 2];
        }
        const sumi = sumi0 + sumi1;
        sumf += (f(x[ib].d) * f(y[ib].d)) * @as(f32, @floatFromInt(sumi)) +
            f(x[ib].m) * f(y[ib].s);
    }

    s[0] = sumf;
}

// -----------------------------------------------------------------------------
// mxfp4 and nvfp4

/// Ports `ggml_vec_dot_mxfp4_q8_0` (arch/arm/quants.c:749 @c1d0e7a00).
///
/// Four-bit indices into `kvalues_mxfp4`, looked up with a table permute
/// rather than arithmetic. Note the scale: the C reaches for
/// `GGML_E8M0_TO_FP32_HALF`, the arithmetic form, where `nvfp4` below uses the
/// *table* form for its own scale.
pub export fn ggml_vec_dot_mxfp4_q8_0(
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
    std.debug.assert(@rem(n, c.QK_MXFP4) == 0);
    comptime std.debug.assert(c.QK_MXFP4 == c.QK8_0);
    _ = .{ bs, bx, by };

    const x = as(blocks.MXFP4, vx);
    const y = as(blocks.Q8_0, vy);

    const nb = @divTrunc(n, c.QK_MXFP4);

    const values = neon.loadFrom(i8x16, @ptrCast(&c.kvalues_mxfp4));

    // `sumf` accumulates one fused term per iteration -- see the note on
    // `iq4_nl`, which shares this shape and where the ordering was pinned
    // against the C prefix by prefix.
    var sumf: f32 = 0;

    var ib: usize = 0;
    while (ib + 1 < nb) : (ib += 2) {
        const q4bits0 = neon.loadFrom(u8x16, &x[ib + 0].qs);
        const q4bits1 = neon.loadFrom(u8x16, &x[ib + 1].qs);

        const q8b0 = neon.loadFrom(i8x16, &y[ib + 0].qs);
        const q8b1 = neon.loadFrom(i8x16, y[ib + 0].qs[16..]);
        const q8b2 = neon.loadFrom(i8x16, &y[ib + 1].qs);
        const q8b3 = neon.loadFrom(i8x16, y[ib + 1].qs[16..]);

        const q4b0 = neon.qtbl1q_s8(values, neon.@"and"(q4bits0, m4b));
        const q4b1 = neon.qtbl1q_s8(values, neon.shrN(q4bits0, 4));
        const q4b2 = neon.qtbl1q_s8(values, neon.@"and"(q4bits1, m4b));
        const q4b3 = neon.qtbl1q_s8(values, neon.shrN(q4bits1, 4));

        const prod_1 = dot2(q4b0, q8b0, q4b1, q8b1);
        const prod_2 = dot2(q4b2, q8b2, q4b3, q8b3);

        const s0 = impl.e8m0ToFp32Half(x[ib + 0].e) * f(y[ib + 0].d);
        const s1 = impl.e8m0ToFp32Half(x[ib + 1].e) * f(y[ib + 1].d);
        const p0: f32 = @floatFromInt(neon.addvq_s32(prod_1));
        const p1: f32 = @floatFromInt(neon.addvq_s32(prod_2));

        sumf += @mulAdd(f32, s0, p0, s1 * p1);
    }

    while (ib < nb) : (ib += 1) {
        const d = f(y[ib].d) * impl.e8m0ToFp32Half(x[ib].e);
        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..c.QK_MXFP4 / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j]) * c.kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + c.QK_MXFP4 / 2]) * c.kvalues_mxfp4[x[ib].qs[j] >> 4];
        }
        sumf = @mulAdd(f32, d, @as(f32, @floatFromInt(sumi1 + sumi2)), sumf);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_nvfp4_q8_0` (arch/arm/quants.c:810 @c1d0e7a00).
///
/// A 64-element super-block of four 16-element sub-blocks over two q8_0
/// blocks. `vpaddq_s32` folds the two halves' four-lane sums into one vector
/// whose lanes line up with the four sub-block scales.
///
/// **The scale comes from `ggml_table_f32_f32_ue4m3`, not arithmetic.**
/// `GGML_CPU_UE4M3_TO_FP32` is a table lookup on NEON, where the generic
/// kernel calls `ggml_ue4m3_to_fp32`. The table is filled by `ggml_cpu_init`,
/// so a caller that has never run a graph gets zeros -- which is exactly what
/// the first capture of this kernel's goldens recorded.
pub export fn ggml_vec_dot_nvfp4_q8_0(
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
    std.debug.assert(@rem(n, c.QK_NVFP4) == 0);
    _ = .{ bs, bx, by };

    const x = as(blocks.NVFP4, vx);
    const y = as(blocks.Q8_0, vy);

    const nb = @divTrunc(n, c.QK_NVFP4);

    const values = neon.loadFrom(i8x16, @ptrCast(&c.kvalues_mxfp4));

    var acc: f32x4 = @splat(0.0);

    for (0..@intCast(nb)) |ib| {
        const q4bits_0 = neon.loadFrom(u8x16, &x[ib].qs);
        const q4bits_1 = neon.loadFrom(u8x16, x[ib].qs[16..]);

        const q4_lo_0 = neon.qtbl1q_s8(values, neon.@"and"(q4bits_0, m4b));
        const q4_hi_0 = neon.qtbl1q_s8(values, neon.shrN(q4bits_0, 4));
        const q4_lo_1 = neon.qtbl1q_s8(values, neon.@"and"(q4bits_1, m4b));
        const q4_hi_1 = neon.qtbl1q_s8(values, neon.shrN(q4bits_1, 4));

        const q8_0a = neon.loadFrom(i8x16, &y[2 * ib].qs);
        const q8_0b = neon.loadFrom(i8x16, y[2 * ib].qs[16..]);
        const q8_lo_0 = neon.combine(neon.low(q8_0a), neon.low(q8_0b));
        const q8_hi_0 = neon.combine(neon.high(q8_0a), neon.high(q8_0b));

        const q8_1a = neon.loadFrom(i8x16, &y[2 * ib + 1].qs);
        const q8_1b = neon.loadFrom(i8x16, y[2 * ib + 1].qs[16..]);
        const q8_lo_1 = neon.combine(neon.low(q8_1a), neon.low(q8_1b));
        const q8_hi_1 = neon.combine(neon.high(q8_1a), neon.high(q8_1b));

        const zero: i32x4 = @splat(0);
        const p0 = neon.add(
            neon.dotq_s32(zero, q4_lo_0, q8_lo_0),
            neon.dotq_s32(zero, q4_hi_0, q8_hi_0),
        );
        const p1 = neon.add(
            neon.dotq_s32(zero, q4_lo_1, q8_lo_1),
            neon.dotq_s32(zero, q4_hi_1, q8_hi_1),
        );
        const sumi = neon.paddq_s32(p0, p1);

        const dy0 = f(y[2 * ib].d);
        const dy1 = f(y[2 * ib + 1].d);

        const nvsc: f32x4 = .{
            cpuUe4m3ToFp32(x[ib].d[0]),
            cpuUe4m3ToFp32(x[ib].d[1]),
            cpuUe4m3ToFp32(x[ib].d[2]),
            cpuUe4m3ToFp32(x[ib].d[3]),
        };
        const dys: f32x4 = .{ dy0, dy0, dy1, dy1 };
        const scales = neon.mul_f32(nvsc, dys);

        acc = neon.fma_f32(acc, neon.cvt_f32_s32(sumi), scales);
    }

    s[0] = neon.addvq_f32(acc);
}

/// Ports `GGML_CPU_UE4M3_TO_FP32` (simd-mappings.h:138 @c1d0e7a00), the NEON arm.
///
/// A lookup, not the arithmetic `ggml_ue4m3_to_fp32` the generic kernel uses.
/// The values agree -- `ggml_cpu_init` fills the table from that same function
/// -- but reading the table is what the C does, and it means the table has to
/// have been filled.
inline fn cpuUe4m3ToFp32(x: u8) f32 {
    return convert.ggml_table_f32_ue4m3[x];
}

// -----------------------------------------------------------------------------
// q5_0, q5_1 and q8_0

/// Ports `ggml_vec_dot_q5_0_q8_0` (arch/arm/quants.c:920 @c1d0e7a00).
///
/// The fifth bit and the bias of 16 are folded into one subtract through
/// `table_b2b_1` -- see the note there.
pub export fn ggml_vec_dot_q5_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(c.QK8_0 == c.QK5_0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumv0: f32x4 = @splat(0.0);
    var sumv1: f32x4 = @splat(0.0);

    var ib: usize = 0;
    while (ib + 1 < nb) : (ib += 2) {
        const x0 = &x[ib];
        const x1 = &x[ib + 1];
        const y0 = &y[ib];
        const y1 = &y[ib + 1];

        const qh0 = std.mem.readInt(u32, &x0.qh, .little);
        const qh1 = std.mem.readInt(u32, &x1.qh, .little);

        const e0 = expandQh(&table_b2b_1, qh0);
        const e1 = expandQh(&table_b2b_1, qh1);

        const v0_0 = neon.loadFrom(u8x16, &x0.qs);
        const v0_1 = neon.loadFrom(u8x16, &x1.qs);

        const v0_0l: i8x16 = @bitCast(neon.@"and"(v0_0, m4b));
        const v0_0h: i8x16 = @bitCast(neon.shrN(v0_0, 4));
        const v0_1l: i8x16 = @bitCast(neon.@"and"(v0_1, m4b));
        const v0_1h: i8x16 = @bitCast(neon.shrN(v0_1, 4));

        const v0_0lf = neon.sub(v0_0l, e0[0]);
        const v0_0hf = neon.sub(v0_0h, e0[1]);
        const v0_1lf = neon.sub(v0_1l, e1[0]);
        const v0_1hf = neon.sub(v0_1h, e1[1]);

        const v1_0l = neon.loadFrom(i8x16, &y0.qs);
        const v1_0h = neon.loadFrom(i8x16, y0.qs[16..]);
        const v1_1l = neon.loadFrom(i8x16, &y1.qs);
        const v1_1h = neon.loadFrom(i8x16, y1.qs[16..]);

        // Two independent dot products added, not chained: the accumulator
        // starts at zero for each half.
        const zero: i32x4 = @splat(0);
        const p_0 = neon.add(
            neon.dotq_s32(zero, v0_0lf, v1_0l),
            neon.dotq_s32(zero, v0_0hf, v1_0h),
        );
        const p_1 = neon.add(
            neon.dotq_s32(zero, v0_1lf, v1_1l),
            neon.dotq_s32(zero, v0_1hf, v1_1h),
        );

        sumv0 = neon.mla_n_f32(sumv0, neon.cvt_f32_s32(p_0), f(x0.d) * f(y0.d));
        sumv1 = neon.mla_n_f32(sumv1, neon.cvt_f32_s32(p_1), f(x1.d) * f(y1.d));
    }

    var sumf = neon.addvq_f32(sumv0) + neon.addvq_f32(sumv1);

    while (ib < nb) : (ib += 1) {
        const qh = std.mem.readInt(u32, &x[ib].qh, .little);
        var sumi0: i32 = 0;
        var sumi1: i32 = 0;
        for (0..qk / 2) |j| {
            const sh: u5 = @intCast(j);
            const xh_0: u8 = @truncate(((qh & (@as(u32, 1) << sh)) >> sh) << 4);
            const xh_1: u8 = @truncate((qh & (@as(u32, 1) << @intCast(j + 16))) >> @intCast(j + 12));
            const x0v: i32 = @as(i8, @bitCast(((x[ib].qs[j] & 0x0F) | xh_0) -% 16));
            const x1v: i32 = @as(i8, @bitCast(((x[ib].qs[j] >> 4) | xh_1) -% 16));
            sumi0 += x0v * y[ib].qs[j];
            sumi1 += x1v * y[ib].qs[j + qk / 2];
        }
        const sumi = sumi0 + sumi1;
        sumf += (f(x[ib].d) * f(y[ib].d)) * @as(f32, @floatFromInt(sumi));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q5_1_q8_1` (arch/arm/quants.c:1032 @c1d0e7a00).
///
/// Unlike `q4_1`, the two blocks' minimum terms go into **separate**
/// accumulators, `summs0` and `summs1`, and are added in that order at the
/// end. Merging them changes the result.
pub export fn ggml_vec_dot_q5_1_q8_1(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_1;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    comptime std.debug.assert(c.QK8_1 == c.QK5_1);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_1, vx);
    const y = as(blocks.Q8_1, vy);

    var sumv0: f32x4 = @splat(0.0);
    var sumv1: f32x4 = @splat(0.0);
    var summs0: f32 = 0.0;
    var summs1: f32 = 0.0;

    var ib: usize = 0;
    while (ib + 1 < nb) : (ib += 2) {
        const x0 = &x[ib];
        const x1 = &x[ib + 1];
        const y0 = &y[ib];
        const y1 = &y[ib + 1];

        summs0 += f(x0.m) * f(y0.s);
        summs1 += f(x1.m) * f(y1.s);

        const qh0 = std.mem.readInt(u32, &x0.qh, .little);
        const qh1 = std.mem.readInt(u32, &x1.qh, .little);

        const e0 = expandQh(&table_b2b_0, qh0);
        const e1 = expandQh(&table_b2b_0, qh1);

        const v0_0 = neon.loadFrom(u8x16, &x0.qs);
        const v0_1 = neon.loadFrom(u8x16, &x1.qs);

        const v0_0l: i8x16 = @bitCast(neon.@"and"(v0_0, m4b));
        const v0_0h: i8x16 = @bitCast(neon.shrN(v0_0, 4));
        const v0_1l: i8x16 = @bitCast(neon.@"and"(v0_1, m4b));
        const v0_1h: i8x16 = @bitCast(neon.shrN(v0_1, 4));

        const v0_0lf = neon.orr(v0_0l, e0[0]);
        const v0_0hf = neon.orr(v0_0h, e0[1]);
        const v0_1lf = neon.orr(v0_1l, e1[0]);
        const v0_1hf = neon.orr(v0_1h, e1[1]);

        const v1_0l = neon.loadFrom(i8x16, &y0.qs);
        const v1_0h = neon.loadFrom(i8x16, y0.qs[16..]);
        const v1_1l = neon.loadFrom(i8x16, &y1.qs);
        const v1_1h = neon.loadFrom(i8x16, y1.qs[16..]);

        const zero: i32x4 = @splat(0);
        const p_0 = neon.add(
            neon.dotq_s32(zero, v0_0lf, v1_0l),
            neon.dotq_s32(zero, v0_0hf, v1_0h),
        );
        const p_1 = neon.add(
            neon.dotq_s32(zero, v0_1lf, v1_1l),
            neon.dotq_s32(zero, v0_1hf, v1_1h),
        );

        sumv0 = neon.mla_n_f32(sumv0, neon.cvt_f32_s32(p_0), f(x0.d) * f(y0.d));
        sumv1 = neon.mla_n_f32(sumv1, neon.cvt_f32_s32(p_1), f(x1.d) * f(y1.d));
    }

    var sumf = neon.addvq_f32(sumv0) + neon.addvq_f32(sumv1) + summs0 + summs1;

    while (ib < nb) : (ib += 1) {
        const qh = std.mem.readInt(u32, &x[ib].qh, .little);
        var sumi0: i32 = 0;
        var sumi1: i32 = 0;
        for (0..qk / 2) |j| {
            const xh_0: u8 = @truncate(((qh >> @intCast(j)) << 4) & 0x10);
            const xh_1: u8 = @truncate((qh >> @intCast(j + 12)) & 0x10);
            const x0v: i32 = (x[ib].qs[j] & 0xF) | xh_0;
            const x1v: i32 = (x[ib].qs[j] >> 4) | xh_1;
            sumi0 += x0v * y[ib].qs[j];
            sumi1 += x1v * y[ib].qs[j + qk / 2];
        }
        const sumi = sumi0 + sumi1;
        sumf += (f(x[ib].d) * f(y[ib].d)) * @as(f32, @floatFromInt(sumi)) +
            f(x[ib].m) * f(y[ib].s);
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_q8_0_q8_0` (arch/arm/quants.c:1150 @c1d0e7a00).
pub export fn ggml_vec_dot_q8_0_q8_0(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    const qk = c.QK8_0;
    const nb = @divTrunc(n, qk);

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q8_0, vx);
    const y = as(blocks.Q8_0, vy);

    var sumv0: f32x4 = @splat(0.0);
    var sumv1: f32x4 = @splat(0.0);

    var ib: usize = 0;
    while (ib + 1 < nb) : (ib += 2) {
        const x0 = &x[ib + 0];
        const x1 = &x[ib + 1];
        const y0 = &y[ib + 0];
        const y1 = &y[ib + 1];

        const x0_0 = neon.loadFrom(i8x16, &x0.qs);
        const x0_1 = neon.loadFrom(i8x16, x0.qs[16..]);
        const x1_0 = neon.loadFrom(i8x16, &x1.qs);
        const x1_1 = neon.loadFrom(i8x16, x1.qs[16..]);

        const y0_0 = neon.loadFrom(i8x16, &y0.qs);
        const y0_1 = neon.loadFrom(i8x16, y0.qs[16..]);
        const y1_0 = neon.loadFrom(i8x16, &y1.qs);
        const y1_1 = neon.loadFrom(i8x16, y1.qs[16..]);

        const zero: i32x4 = @splat(0);
        const p_0 = neon.add(
            neon.dotq_s32(zero, x0_0, y0_0),
            neon.dotq_s32(zero, x0_1, y0_1),
        );
        const p_1 = neon.add(
            neon.dotq_s32(zero, x1_0, y1_0),
            neon.dotq_s32(zero, x1_1, y1_1),
        );

        sumv0 = neon.mla_n_f32(sumv0, neon.cvt_f32_s32(p_0), f(x0.d) * f(y0.d));
        sumv1 = neon.mla_n_f32(sumv1, neon.cvt_f32_s32(p_1), f(x1.d) * f(y1.d));
    }

    var sumf = neon.addvq_f32(sumv0) + neon.addvq_f32(sumv1);

    while (ib < nb) : (ib += 1) {
        var sumi: i32 = 0;
        for (0..qk) |j| {
            sumi += @as(i32, x[ib].qs[j]) * y[ib].qs[j];
        }
        sumf += @as(f32, @floatFromInt(sumi)) * (f(x[ib].d) * f(y[ib].d));
    }

    s[0] = sumf;
}

/// Ports `ggml_vec_dot_iq4_nl_q8_0` (arch/arm/quants.c:4196 @c1d0e7a00).
///
/// The same shape as `mxfp4`, over `kvalues_iq4nl` and with an fp16 delta in
/// place of the e8m0 exponent.
pub export fn ggml_vec_dot_iq4_nl_q8_0(
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
    std.debug.assert(@rem(n, c.QK4_NL) == 0);
    comptime std.debug.assert(c.QK4_NL == c.QK8_0);
    _ = .{ bs, bx, by };

    const x = as(blocks.IQ4_NL, vx);
    const y = as(blocks.Q8_0, vy);

    const nb = @divTrunc(n, c.QK4_NL);

    const values = neon.loadFrom(i8x16, @ptrCast(&c.kvalues_iq4nl));

    // Two fused multiply-adds per iteration, not `sumf += A + B`.
    //
    // The C writes one statement, `sumf += SA*PA + SB*PB`. Under
    // `-ffp-contract=on` -- which is how this file ships -- clang fuses the
    // *left* multiply into the inner add, giving `sumf + fma(SA, PA, SB*PB)`,
    // and fuses the tail's multiply into its accumulate as well. Three
    // roundings become one in each place, and the answers differ in the last
    // bit.
    //
    // **Determined against an oracle, not guessed.** `scripts/vecdot-prefix`
    // asks the shipped kernel for every row prefix from one block to sixteen;
    // eight candidate orderings were tried and exactly one reproduces all
    // sixteen values. The plain reading, `sumf += SA*PA + SB*PB`, misses eleven
    // of them.
    //
    // This is the one place where naming fusion sites is sound. The quantizers
    // in `src/ggml/quants/` deliberately do *not* -- see the note at the top of
    // `helpers.zig` -- because there the sites are implicit in 5,000 lines of
    // plain arithmetic and the guess broke. Here they are pinned per kernel by
    // sixteen equations.
    var sumf: f32 = 0;

    var ib: usize = 0;
    while (ib + 1 < nb) : (ib += 2) {
        const q4bits0 = neon.loadFrom(u8x16, &x[ib + 0].qs);
        const q4bits1 = neon.loadFrom(u8x16, &x[ib + 1].qs);

        const q8b0 = neon.loadFrom(i8x16, &y[ib + 0].qs);
        const q8b1 = neon.loadFrom(i8x16, y[ib + 0].qs[16..]);
        const q8b2 = neon.loadFrom(i8x16, &y[ib + 1].qs);
        const q8b3 = neon.loadFrom(i8x16, y[ib + 1].qs[16..]);

        const q4b0 = neon.qtbl1q_s8(values, neon.@"and"(q4bits0, m4b));
        const q4b1 = neon.qtbl1q_s8(values, neon.shrN(q4bits0, 4));
        const q4b2 = neon.qtbl1q_s8(values, neon.@"and"(q4bits1, m4b));
        const q4b3 = neon.qtbl1q_s8(values, neon.shrN(q4bits1, 4));

        const prod_1 = dot2(q4b0, q8b0, q4b1, q8b1);
        const prod_2 = dot2(q4b2, q8b2, q4b3, q8b3);

        const s0 = f(x[ib + 0].d) * f(y[ib + 0].d);
        const s1 = f(x[ib + 1].d) * f(y[ib + 1].d);
        const p0: f32 = @floatFromInt(neon.addvq_s32(prod_1));
        const p1: f32 = @floatFromInt(neon.addvq_s32(prod_2));

        sumf += @mulAdd(f32, s0, p0, s1 * p1);
    }

    while (ib < nb) : (ib += 1) {
        const d = f(y[ib].d) * f(x[ib].d);
        var sumi1: i32 = 0;
        var sumi2: i32 = 0;
        for (0..c.QK4_NL / 2) |j| {
            sumi1 += @as(i32, y[ib].qs[j]) * c.kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += @as(i32, y[ib].qs[j + c.QK4_NL / 2]) * c.kvalues_iq4nl[x[ib].qs[j] >> 4];
        }
        sumf = @mulAdd(f32, d, @as(f32, @floatFromInt(sumi1 + sumi2)), sumf);
    }

    s[0] = sumf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

const testing = @import("../testing.zig");
const golden = @import("golden.zig");

test "q1_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q1_0_q8_0, c.GGML_TYPE_Q1_0, c.GGML_TYPE_Q8_0, golden.q1_0);
}

test "q2_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q2_0_q8_0, c.GGML_TYPE_Q2_0, c.GGML_TYPE_Q8_0, golden.q2_0);
}

test "q4_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q4_0_q8_0, c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q8_0, golden.q4_0);
}

test "q4_1 dot matches the C" {
    try testing.check(ggml_vec_dot_q4_1_q8_1, c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q8_1, golden.q4_1);
}

test "mxfp4 dot matches the C" {
    try testing.check(ggml_vec_dot_mxfp4_q8_0, c.GGML_TYPE_MXFP4, c.GGML_TYPE_Q8_0, golden.mxfp4);
}

test "nvfp4 dot matches the C" {
    try testing.check(ggml_vec_dot_nvfp4_q8_0, c.GGML_TYPE_NVFP4, c.GGML_TYPE_Q8_0, golden.nvfp4);
}

test "q5_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q5_0_q8_0, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q8_0, golden.q5_0);
}

test "q5_1 dot matches the C" {
    try testing.check(ggml_vec_dot_q5_1_q8_1, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q8_1, golden.q5_1);
}

test "q8_0 dot matches the C" {
    try testing.check(ggml_vec_dot_q8_0_q8_0, c.GGML_TYPE_Q8_0, c.GGML_TYPE_Q8_0, golden.q8_0);
}

test "iq4_nl dot matches the C" {
    try testing.check(ggml_vec_dot_iq4_nl_q8_0, c.GGML_TYPE_IQ4_NL, c.GGML_TYPE_Q8_0, golden.iq4_nl);
}

test "the two bit-expansion tables are complements" {
    // `q5_0` folds the bias of 16 into a subtract through `table_b2b_1` and
    // `q5_1` ORs `table_b2b_0`. Swapping them inverts every fifth bit.
    for (0..256) |i| {
        const a: [8]u8 = @bitCast(table_b2b_0[i]);
        const b: [8]u8 = @bitCast(table_b2b_1[i]);
        for (0..8) |j| {
            try std.testing.expectEqual(@as(u8, 0x10), a[j] | b[j]);
            try std.testing.expectEqual(@as(u8, 0x00), a[j] & b[j]);
        }
    }
}

test "the bit-expansion table matches the C's macro" {
    // The C builds this with an eight-deep token-pasting macro. Byte `j` of
    // entry `i` must be `0x10` exactly when bit `j` of `i` is set, and getting
    // the bit order backwards would be a plausible-looking wrong answer.
    try std.testing.expectEqual(@as(u64, 0x0000000000000000), table_b2b_0[0]);
    try std.testing.expectEqual(@as(u64, 0x0000000000000010), table_b2b_0[1]);
    try std.testing.expectEqual(@as(u64, 0x0000000000001010), table_b2b_0[3]);
    try std.testing.expectEqual(@as(u64, 0x1000000000000000), table_b2b_0[128]);
    try std.testing.expectEqual(@as(u64, 0x1010101010101010), table_b2b_0[255]);
}

test "the q2_0 field extraction lines up with the packing" {
    // `tbl_idx_lo` replicates each of the first four bytes four times, and
    // `shift_vals` then selects a different 2-bit field per lane. A transposed
    // pair here reads the fields out of order.
    var raw: [8]u8 = @splat(0);
    raw[0] = 0b11100100; // fields 0,1,2,3 from low to high
    const raw16 = neon.combine(@as(neon.u8x8, raw), @as(neon.u8x8, raw));

    const bytes0 = neon.qtbl1q_u8(raw16, tbl_idx_lo);
    const mask2: u8x16 = @splat(0x03);
    const fields: [16]u8 = neon.@"and"(neon.shlq(bytes0, shift_vals), mask2);

    try std.testing.expectEqual(@as(u8, 0), fields[0]);
    try std.testing.expectEqual(@as(u8, 1), fields[1]);
    try std.testing.expectEqual(@as(u8, 2), fields[2]);
    try std.testing.expectEqual(@as(u8, 3), fields[3]);
}
