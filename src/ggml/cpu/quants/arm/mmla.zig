//! The two-row i8mm arms of the NEON dot products.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` (v1.0.0,
//! `b1b770d90`), the `__ARM_FEATURE_MATMUL_INT8` arms of the dot products at
//! lines 297, 590, 1150, 2334 and 2964. Each function names the C it replaces
//! and the line its arm began at.
//!
//! # Where these sit
//!
//! In the C each is an `if (nrc == 2) { ... return; }` at the top of the
//! function whose one-row NEON body `legacy.zig` and `k.zig` already port.
//! They live in a file of their own because they exist only on a CPU with
//! i8mm: `type_traits_cpu` asks for two rows per call exactly when
//! `__ARM_FEATURE_MATMUL_INT8` is defined (`cpu/traits.zig`), and each
//! exported kernel calls in here behind `comptime enabled`. A build for the
//! baseline never reaches this file.
//!
//! # The shape they share
//!
//! Two rows of the weight and two columns of the activation at once. One
//! block from each of the four is split into 8-byte halves, and the halves
//! are zipped so one `smmla` gives all four products of the pair:
//! `[x0.y0, x0.y1, x1.y0, x1.y1]`, row-major. The per-pair scales are laid
//! out in the same order, so a single fused `vmlaq_f32` accumulates all four.
//!
//! The epilogue reorders that to `[x0.y0, x1.y0, x0.y1, x1.y1]` --
//! `vzip1q_f32(v, vextq_f32(v, v, 2))` -- and stores the first two to `s`
//! and the second two to `s + bs`: one output column per activation column.
//!
//! # The float arithmetic is the C's
//!
//! Each four-lane scale is four plain products in an initializer -- not
//! fused, they are separate expressions -- and each accumulate is one
//! `vmlaq_f32` or `vmlsq_f32`, which `arm_neon.h` writes as one expression
//! and `-ffp-contract=on` makes one `fmla`/`fmls`. There is no horizontal
//! reduction: every lane is a separate output, so none of the pairwise
//! traps in `neon.zig` arise.

const std = @import("std");
const ggml = @import("../../../module.zig");

inline fn f(h: u16) f32 {
    return ggml.cpu.convert.cpuFp16ToFp32(h);
}

/// Row `off` bytes past `p`, as the C's `(const block *) ((const uint8_t *) vx + bx)`.
inline fn rowAt(comptime Block: type, p: ?*const anyopaque, off: usize) [*]const Block {
    return @ptrCast(@alignCast(@as([*]const u8, @ptrCast(p.?)) + off));
}

/// `vzip1q_f32(v, vextq_f32(v, v, 2))` and the two `vst1_f32`: lanes
/// `ABCD` become `AC` at `s` and `BD` at `s + bs`.
inline fn storeTwoColumns(s: [*c]f32, bs: usize, v: ggml.cpu.quants.arch.neon.f32x4) void {
    storeReordered(s, bs, reorder(v));
}

/// The two `vst1_f32` alone, for a vector already in `ACBD` order.
inline fn storeReordered(s: [*c]f32, bs: usize, r: ggml.cpu.quants.arch.neon.f32x4) void {
    s[0] = r[0];
    s[1] = r[1];
    s[bs + 0] = r[2];
    s[bs + 1] = r[3];
}

/// `vzip1q_f32(v, vextq_f32(v, v, 2))`: `ABCD` to `ACBD`.
inline fn reorder(v: ggml.cpu.quants.arch.neon.f32x4) ggml.cpu.quants.arch.neon.f32x4 {
    return ggml.cpu.quants.arch.neon.zip1q_f32(v, ggml.cpu.quants.arch.neon.extq_f32(v, v, 2));
}

/// The chained four `vmmlaq_s32` the `q4_0`, `q4_1` and `q8_0` arms close
/// on: both halves of both 16-byte groups of a 32-weight block.
///
/// Parameters:
/// - `x0l`, `x0h`, `x1l`, `x1h`: the two weight rows' low and high 16 bytes.
/// - `y0l`, `y0h`, `y1l`, `y1h`: the same for the two activation columns.
///
/// Return: `[x0.y0, x0.y1, x1.y0, x1.y1]`.
inline fn mmlaBlock32(x0l: ggml.cpu.quants.arch.neon.i8x16, x0h: ggml.cpu.quants.arch.neon.i8x16, x1l: ggml.cpu.quants.arch.neon.i8x16, x1h: ggml.cpu.quants.arch.neon.i8x16, y0l: ggml.cpu.quants.arch.neon.i8x16, y0h: ggml.cpu.quants.arch.neon.i8x16, y1l: ggml.cpu.quants.arch.neon.i8x16, y1h: ggml.cpu.quants.arch.neon.i8x16) ggml.cpu.quants.arch.neon.i32x4 {
    const l0 = ggml.cpu.quants.arch.neon.zip1q_s64(x0l, x1l);
    const l1 = ggml.cpu.quants.arch.neon.zip2q_s64(x0l, x1l);
    const l2 = ggml.cpu.quants.arch.neon.zip1q_s64(x0h, x1h);
    const l3 = ggml.cpu.quants.arch.neon.zip2q_s64(x0h, x1h);

    const r0 = ggml.cpu.quants.arch.neon.zip1q_s64(y0l, y1l);
    const r1 = ggml.cpu.quants.arch.neon.zip2q_s64(y0l, y1l);
    const r2 = ggml.cpu.quants.arch.neon.zip1q_s64(y0h, y1h);
    const r3 = ggml.cpu.quants.arch.neon.zip2q_s64(y0h, y1h);

    const zero: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
    return ggml.cpu.quants.arch.neon.mmlaq_s32(ggml.cpu.quants.arch.neon.mmlaq_s32(ggml.cpu.quants.arch.neon.mmlaq_s32(ggml.cpu.quants.arch.neon.mmlaq_s32(zero, l0, r0), l1, r1), l2, r2), l3, r3);
}

/// The four products of two scales by two scales, in `mmlaBlock32`'s
/// lane order. Plain multiplies: the C writes them into a `float32_t[4]`.
inline fn outer(a0: f32, a1: f32, b0: f32, b1: f32) ggml.cpu.quants.arch.neon.f32x4 {
    return .{ a0 * b0, a0 * b1, a1 * b0, a1 * b1 };
}

/// `vaddq_s32(vmull_s16(low a, low b), vmull_s16(high a, high b))`.
inline fn mullSum(a: ggml.cpu.quants.arch.neon.i16x8, b: ggml.cpu.quants.arch.neon.i16x8) ggml.cpu.quants.arch.neon.i32x4 {
    return ggml.cpu.quants.arch.neon.add(
        ggml.cpu.quants.arch.neon.mull_s16(ggml.cpu.quants.arch.neon.low(a), ggml.cpu.quants.arch.neon.low(b)),
        ggml.cpu.quants.arch.neon.mull_s16(ggml.cpu.quants.arch.neon.high(a), ggml.cpu.quants.arch.neon.high(b)),
    );
}

// -----------------------------------------------------------------------------
// q4_0, q4_1 and q8_0

const m4b: ggml.cpu.quants.arch.neon.u8x16 = @splat(0x0F);
const s8b: ggml.cpu.quants.arch.neon.i8x16 = @splat(0x8);

/// Ports `ggml_vec_dot_q4_0_q8_0` (arch/arm/quants.c:300 @b1b770d90), the
/// `__ARM_FEATURE_MATMUL_INT8` arm at line 315: two rows.
pub fn q4_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize) void {
    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK8_0));

    const vx0 = rowAt(ggml.quants.blocks.Q4_0, vx, 0);
    const vx1 = rowAt(ggml.quants.blocks.Q4_0, vx, bx);
    const vy0 = rowAt(ggml.quants.blocks.Q8_0, vy, 0);
    const vy1 = rowAt(ggml.quants.blocks.Q8_0, vy, by);

    var sumv0: ggml.cpu.quants.arch.neon.f32x4 = @splat(0.0);

    for (0..nb) |i| {
        const b_x0 = &vx0[i];
        const b_x1 = &vx1[i];
        const b_y0 = &vy0[i];
        const b_y1 = &vy1[i];

        const v0_0 = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, &b_x0.qs);
        const v0_1 = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, &b_x1.qs);

        // 4-bit -> 8-bit, then the bias of eight.
        const x0_l = ggml.cpu.quants.arch.neon.sub(@as(ggml.cpu.quants.arch.neon.i8x16, @bitCast(ggml.cpu.quants.arch.neon.@"and"(v0_0, m4b))), s8b);
        const x0_h = ggml.cpu.quants.arch.neon.sub(@as(ggml.cpu.quants.arch.neon.i8x16, @bitCast(ggml.cpu.quants.arch.neon.shrN(v0_0, 4))), s8b);
        const x1_l = ggml.cpu.quants.arch.neon.sub(@as(ggml.cpu.quants.arch.neon.i8x16, @bitCast(ggml.cpu.quants.arch.neon.@"and"(v0_1, m4b))), s8b);
        const x1_h = ggml.cpu.quants.arch.neon.sub(@as(ggml.cpu.quants.arch.neon.i8x16, @bitCast(ggml.cpu.quants.arch.neon.shrN(v0_1, 4))), s8b);

        const y0_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_y0.qs);
        const y0_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_y0.qs[16..]);
        const y1_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_y1.qs);
        const y1_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_y1.qs[16..]);

        const scale = outer(f(b_x0.d), f(b_x1.d), f(b_y0.d), f(b_y1.d));

        const p = mmlaBlock32(x0_l, x0_h, x1_l, x1_h, y0_l, y0_h, y1_l, y1_h);
        sumv0 = ggml.cpu.quants.arch.neon.mlaq_f32(sumv0, ggml.cpu.quants.arch.neon.cvt_f32_s32(p), scale);
    }

    storeTwoColumns(s, bs, sumv0);
}

/// Ports `ggml_vec_dot_q4_1_q8_1` (arch/arm/quants.c:593 @b1b770d90), the
/// `__ARM_FEATURE_MATMUL_INT8` arm at line 608: two rows.
///
/// `summs0` is built already in the reordered `ACBD` layout -- the C's
/// `summs_t` lists `x0.y0, x1.y0, x0.y1, x1.y1` -- and is added after the
/// reorder, not before.
pub fn q4_1(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize) void {
    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK8_1));

    const vx0 = rowAt(ggml.quants.blocks.Q4_1, vx, 0);
    const vx1 = rowAt(ggml.quants.blocks.Q4_1, vx, bx);
    const vy0 = rowAt(ggml.quants.blocks.Q8_1, vy, 0);
    const vy1 = rowAt(ggml.quants.blocks.Q8_1, vy, by);

    var sumv0: ggml.cpu.quants.arch.neon.f32x4 = @splat(0.0);
    var summs0: ggml.cpu.quants.arch.neon.f32x4 = @splat(0.0);

    for (0..nb) |i| {
        const b_x0 = &vx0[i];
        const b_x1 = &vx1[i];
        const b_y0 = &vy0[i];
        const b_y1 = &vy1[i];

        const summs_t: ggml.cpu.quants.arch.neon.f32x4 = .{
            f(b_x0.m) * f(b_y0.s),
            f(b_x1.m) * f(b_y0.s),
            f(b_x0.m) * f(b_y1.s),
            f(b_x1.m) * f(b_y1.s),
        };
        summs0 = summs0 + summs_t;

        const v0_0 = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, &b_x0.qs);
        const v0_1 = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, &b_x1.qs);

        // 4-bit -> 8-bit; no bias, the minimum is in `summs0`.
        const x0_l: ggml.cpu.quants.arch.neon.i8x16 = @bitCast(ggml.cpu.quants.arch.neon.@"and"(v0_0, m4b));
        const x0_h: ggml.cpu.quants.arch.neon.i8x16 = @bitCast(ggml.cpu.quants.arch.neon.shrN(v0_0, 4));
        const x1_l: ggml.cpu.quants.arch.neon.i8x16 = @bitCast(ggml.cpu.quants.arch.neon.@"and"(v0_1, m4b));
        const x1_h: ggml.cpu.quants.arch.neon.i8x16 = @bitCast(ggml.cpu.quants.arch.neon.shrN(v0_1, 4));

        const y0_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_y0.qs);
        const y0_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_y0.qs[16..]);
        const y1_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_y1.qs);
        const y1_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_y1.qs[16..]);

        const scale = outer(f(b_x0.d), f(b_x1.d), f(b_y0.d), f(b_y1.d));

        const p = mmlaBlock32(x0_l, x0_h, x1_l, x1_h, y0_l, y0_h, y1_l, y1_h);
        sumv0 = ggml.cpu.quants.arch.neon.mlaq_f32(sumv0, ggml.cpu.quants.arch.neon.cvt_f32_s32(p), scale);
    }

    storeReordered(s, bs, reorder(sumv0) + summs0);
}

/// Ports `ggml_vec_dot_q8_0_q8_0` (arch/arm/quants.c:1153 @b1b770d90), the
/// `__ARM_FEATURE_MATMUL_INT8` arm at line 1168: two rows.
pub fn q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize) void {
    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK8_0));

    const vx0 = rowAt(ggml.quants.blocks.Q8_0, vx, 0);
    const vx1 = rowAt(ggml.quants.blocks.Q8_0, vx, bx);
    const vy0 = rowAt(ggml.quants.blocks.Q8_0, vy, 0);
    const vy1 = rowAt(ggml.quants.blocks.Q8_0, vy, by);

    var sumv0: ggml.cpu.quants.arch.neon.f32x4 = @splat(0.0);

    for (0..nb) |i| {
        const b_x0 = &vx0[i];
        const b_y0 = &vy0[i];
        const b_x1 = &vx1[i];
        const b_y1 = &vy1[i];

        const x0_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_x0.qs);
        const x0_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_x0.qs[16..]);
        const x1_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_x1.qs);
        const x1_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_x1.qs[16..]);

        const y0_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_y0.qs);
        const y0_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_y0.qs[16..]);
        const y1_l = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, &b_y1.qs);
        const y1_h = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, b_y1.qs[16..]);

        const scale = outer(f(b_x0.d), f(b_x1.d), f(b_y0.d), f(b_y1.d));

        const p = mmlaBlock32(x0_l, x0_h, x1_l, x1_h, y0_l, y0_h, y1_l, y1_h);
        sumv0 = ggml.cpu.quants.arch.neon.mlaq_f32(sumv0, ggml.cpu.quants.arch.neon.cvt_f32_s32(p), scale);
    }

    storeTwoColumns(s, bs, sumv0);
}

// -----------------------------------------------------------------------------
// q4_K and q6_K

/// One `q4_K` super-block's scales and mins, decoded as the i8mm arm does:
/// eight `i8` scales and eight mins widened to `i16`.
const Q4KScales = struct {
    scales: [8]i8,
    mins: ggml.cpu.quants.arch.neon.i16x8,
};

/// The two braced blocks at the top of the `q4_K` arm's loop.
///
/// Parameters:
/// - `packed_scales`: a `block_q4_K`'s twelve `scales` bytes.
///
/// Return: the decoded scales and mins.
inline fn decodeQ4K(packed_scales: *const [ggml.impl.ggml_c.K_SCALE_SIZE]u8) Q4KScales {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;

    var scales_mins: [3]u32 = undefined;
    @memcpy(std.mem.asBytes(&scales_mins), packed_scales);

    const mins_0_3 = scales_mins[1] & kmask1;
    const mins_4_7 = ((scales_mins[2] >> 4) & kmask2) | (((scales_mins[1] >> 6) & kmask3) << 4);
    const mins: ggml.cpu.quants.arch.neon.u32x2 = .{ mins_0_3, mins_4_7 };

    var scales: [2]u32 = undefined;
    scales[0] = scales_mins[0] & kmask1; // scales 0~3
    scales[1] = (scales_mins[2] & kmask2) | (((scales_mins[0] >> 6) & kmask3) << 4); // scales 4~7

    return .{
        .scales = @bitCast(scales),
        .mins = @bitCast(@as(ggml.cpu.quants.arch.neon.u16x8, ggml.cpu.quants.arch.neon.movl_u8(@bitCast(mins)))),
    };
}

/// Ports `ggml_vec_dot_q4_K_q8_K` (arch/arm/quants.c:2337 @b1b770d90), the
/// `__ARM_FEATURE_MATMUL_INT8` arm at line 2569: two rows.
///
/// The block scales multiply in integers (`vmlaq_s32` into `visum`), so only
/// the super-block epilogue is float: the mins first, fused-subtracted, then
/// the scaled sum, fused-added -- two statements, in that order.
pub fn q4_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize) void {
    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK_K));

    const x0 = rowAt(ggml.quants.blocks.Q4_K, vx, 0);
    const x1 = rowAt(ggml.quants.blocks.Q4_K, vx, bx);
    const y0 = rowAt(ggml.quants.blocks.Q8_K, vy, 0);
    const y1 = rowAt(ggml.quants.blocks.Q8_K, vy, by);

    var vfsum: ggml.cpu.quants.arch.neon.f32x4 = @splat(0.0);

    for (0..nb) |i| {
        const qx0: [*]const u8 = &x0[i].qs;
        const qx1: [*]const u8 = &x1[i].qs;
        const qy0: [*]const i8 = &y0[i].qs;
        const qy1: [*]const i8 = &y1[i].qs;

        const sm0 = decodeQ4K(&x0[i].scales);
        const sm1 = decodeQ4K(&x1[i].scales);

        var visum: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);

        // 64 weights per iteration, 256 in all.
        for (0..ggml.impl.ggml_c.QK_K / 64) |j| {
            const yo = j * 64;
            const vy0 = [4]ggml.cpu.quants.arch.neon.i8x16{
                ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy0 + yo),      ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy0 + yo + 16),
                ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy0 + yo + 32), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy0 + yo + 48),
            };
            const vy1 = [4]ggml.cpu.quants.arch.neon.i8x16{
                ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy1 + yo),      ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy1 + yo + 16),
                ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy1 + yo + 32), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy1 + yo + 48),
            };

            const xo = j * 32;
            const vx0 = nibbles(ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, qx0 + xo), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, qx0 + xo + 16));
            const vx1 = nibbles(ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, qx1 + xo), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, qx1 + xo + 16));

            // 32 weights, one block scale, per iteration.
            for (0..2) |k| {
                const blk = j * 2 + k;
                const block_scale: ggml.cpu.quants.arch.neon.i32x4 = .{
                    sm0.scales[blk],
                    sm0.scales[blk],
                    sm1.scales[blk],
                    sm1.scales[blk],
                };

                var vr: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
                for (0..2) |l| {
                    const idx = k * 2 + l;
                    const vx_l = ggml.cpu.quants.arch.neon.zip1q_s64(vx0[idx], vx1[idx]);
                    const vx_h = ggml.cpu.quants.arch.neon.zip2q_s64(vx0[idx], vx1[idx]);
                    const vy_l = ggml.cpu.quants.arch.neon.zip1q_s64(vy0[idx], vy1[idx]);
                    const vy_h = ggml.cpu.quants.arch.neon.zip2q_s64(vy0[idx], vy1[idx]);
                    vr = ggml.cpu.quants.arch.neon.mmlaq_s32(vr, vx_l, vy_l);
                    vr = ggml.cpu.quants.arch.neon.mmlaq_s32(vr, vx_h, vy_h);
                }
                // Will not overflow: block_scale * sum_256(int4*int8) is
                // under 2^28.
                visum = ggml.cpu.quants.arch.neon.mlaq_s32(visum, vr, block_scale);
            }
        }

        // The bias, then the super-block scale.
        const y0_sums = ggml.cpu.quants.arch.neon.paddq_s16(ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, &y0[i].bsums), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, y0[i].bsums[8..]));
        const y1_sums = ggml.cpu.quants.arch.neon.paddq_s16(ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, &y1[i].bsums), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, y1[i].bsums[8..]));
        const bias: ggml.cpu.quants.arch.neon.i32x4 = .{
            ggml.cpu.quants.arch.neon.addvq_s32(mullSum(y0_sums, sm0.mins)),
            ggml.cpu.quants.arch.neon.addvq_s32(mullSum(y1_sums, sm0.mins)),
            ggml.cpu.quants.arch.neon.addvq_s32(mullSum(y0_sums, sm1.mins)),
            ggml.cpu.quants.arch.neon.addvq_s32(mullSum(y1_sums, sm1.mins)),
        };
        const dmins: ggml.cpu.quants.arch.neon.f32x4 = .{
            f(x0[i].dmin) * y0[i].d,
            f(x0[i].dmin) * y1[i].d,
            f(x1[i].dmin) * y0[i].d,
            f(x1[i].dmin) * y1[i].d,
        };
        vfsum = ggml.cpu.quants.arch.neon.mlsq_f32(vfsum, ggml.cpu.quants.arch.neon.cvt_f32_s32(bias), dmins);

        const superblock_scale: ggml.cpu.quants.arch.neon.f32x4 = .{
            f(x0[i].d) * y0[i].d,
            f(x0[i].d) * y1[i].d,
            f(x1[i].d) * y0[i].d,
            f(x1[i].d) * y1[i].d,
        };
        vfsum = ggml.cpu.quants.arch.neon.mlaq_f32(vfsum, ggml.cpu.quants.arch.neon.cvt_f32_s32(visum), superblock_scale);
    }

    storeTwoColumns(s, bs, vfsum);
}

/// The `q4_K` arm's 4-bit to 8-bit split of 32 packed bytes: low nibbles of
/// both halves, then high nibbles of both. Not the `q4_0` order.
inline fn nibbles(v0: ggml.cpu.quants.arch.neon.u8x16, v1: ggml.cpu.quants.arch.neon.u8x16) [4]ggml.cpu.quants.arch.neon.i8x16 {
    const mask: ggml.cpu.quants.arch.neon.u8x16 = @splat(0x0f);
    return .{
        @bitCast(ggml.cpu.quants.arch.neon.@"and"(v0, mask)),
        @bitCast(ggml.cpu.quants.arch.neon.@"and"(v1, mask)),
        @bitCast(ggml.cpu.quants.arch.neon.shrN(v0, 4)),
        @bitCast(ggml.cpu.quants.arch.neon.shrN(v1, 4)),
    };
}

/// The `q6_K` arm's de-quantization of one half super-block -- 128 weights,
/// 64 `ql` bytes and 32 `qh` bytes -- into eight vectors of six-bit values,
/// **without** the bias of 32, which the arm takes out of the sum instead.
///
/// Parameters:
/// - `ql`: 64 bytes of low nibbles.
/// - `qh`: 32 bytes of high bit pairs.
///
/// Return: the eight vectors, in the C's `vx[0..8]` order.
inline fn dequantQ6K(ql: [*]const u8, qh: [*]const u8) [8]ggml.cpu.quants.arch.neon.i8x16 {
    const mone: ggml.cpu.quants.arch.neon.u8x16 = @splat(0x30);
    const mask: ggml.cpu.quants.arch.neon.u8x16 = @splat(0x0f);

    const qh_bits = [2]ggml.cpu.quants.arch.neon.u8x16{ ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, qh), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, qh + 16) };
    const ql_bits = [4]ggml.cpu.quants.arch.neon.u8x16{
        ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, ql),      ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, ql + 16),
        ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, ql + 32), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.u8x16, ql + 48),
    };

    var vx: [8]ggml.cpu.quants.arch.neon.i8x16 = undefined;

    var q6h_0 = ggml.cpu.quants.arch.neon.@"and"(mone, ggml.cpu.quants.arch.neon.shlN(qh_bits[0], 4));
    var q6h_1 = ggml.cpu.quants.arch.neon.@"and"(mone, ggml.cpu.quants.arch.neon.shlN(qh_bits[1], 4));
    var q6h_2 = ggml.cpu.quants.arch.neon.@"and"(mone, ggml.cpu.quants.arch.neon.shlN(qh_bits[0], 2));
    var q6h_3 = ggml.cpu.quants.arch.neon.@"and"(mone, ggml.cpu.quants.arch.neon.shlN(qh_bits[1], 2));

    vx[0] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.@"and"(ql_bits[0], mask), q6h_0));
    vx[1] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.@"and"(ql_bits[1], mask), q6h_1));
    vx[2] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.@"and"(ql_bits[2], mask), q6h_2));
    vx[3] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.@"and"(ql_bits[3], mask), q6h_3));

    q6h_0 = ggml.cpu.quants.arch.neon.@"and"(mone, qh_bits[0]);
    q6h_1 = ggml.cpu.quants.arch.neon.@"and"(mone, qh_bits[1]);
    q6h_2 = ggml.cpu.quants.arch.neon.@"and"(mone, ggml.cpu.quants.arch.neon.shrN(qh_bits[0], 2));
    q6h_3 = ggml.cpu.quants.arch.neon.@"and"(mone, ggml.cpu.quants.arch.neon.shrN(qh_bits[1], 2));

    vx[4] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.shrN(ql_bits[0], 4), q6h_0));
    vx[5] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.shrN(ql_bits[1], 4), q6h_1));
    vx[6] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.shrN(ql_bits[2], 4), q6h_2));
    vx[7] = @bitCast(ggml.cpu.quants.arch.neon.orr(ggml.cpu.quants.arch.neon.shrN(ql_bits[3], 4), q6h_3));

    return vx;
}

/// `vaddvq_s32` of the `q6_K` arm's bias product: sixteen `bsums` against
/// sixteen sign-extended scales, as four widening multiplies.
inline fn q6Bias(q8sums: [2]ggml.cpu.quants.arch.neon.i16x8, q6scales: [2]ggml.cpu.quants.arch.neon.i16x8) i32 {
    return ggml.cpu.quants.arch.neon.addvq_s32(ggml.cpu.quants.arch.neon.add(mullSum(q8sums[0], q6scales[0]), mullSum(q8sums[1], q6scales[1])));
}

/// The sixteen `i8` scales of a `q6_K` block widened to two `i16x8`.
inline fn q6Scales(scales: *const [ggml.impl.ggml_c.QK_K / 16]i8) [2]ggml.cpu.quants.arch.neon.i16x8 {
    const v = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, scales);
    return .{ ggml.cpu.quants.arch.neon.movl_s8(ggml.cpu.quants.arch.neon.low(v)), ggml.cpu.quants.arch.neon.movl_s8(ggml.cpu.quants.arch.neon.high(v)) };
}

/// Ports `ggml_vec_dot_q6_K_q8_K` (arch/arm/quants.c:2969 @b1b770d90), the
/// `__ARM_FEATURE_MATMUL_INT8` arm at line 3175: two rows.
///
/// The bias of 32 is not subtracted from the weights. It comes off the
/// integer sum at the end of each super-block as `32 * sum(bsums * scales)`
/// -- the same arithmetic, exact, done once.
pub fn q6_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, bx: usize, vy: ?*const anyopaque, by: usize) void {
    const nb: usize = @intCast(@divTrunc(n, ggml.impl.ggml_c.QK_K));

    const x0 = rowAt(ggml.quants.blocks.Q6_K, vx, 0);
    const x1 = rowAt(ggml.quants.blocks.Q6_K, vx, bx);
    const y0 = rowAt(ggml.quants.blocks.Q8_K, vy, 0);
    const y1 = rowAt(ggml.quants.blocks.Q8_K, vy, by);

    var vfsum: ggml.cpu.quants.arch.neon.f32x4 = @splat(0.0);

    for (0..nb) |i| {
        const qy0: [*]const i8 = &y0[i].qs;
        const qy1: [*]const i8 = &y1[i].qs;

        var visum: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);

        // Eight blocks of sixteen per iteration, sixteen in all.
        for (0..2) |j| {
            const vx0 = dequantQ6K(x0[i].ql[j * 64 ..].ptr, x0[i].qh[j * 32 ..].ptr);
            const vx1 = dequantQ6K(x1[i].ql[j * 64 ..].ptr, x1[i].qh[j * 32 ..].ptr);

            // Sixteen weights, one block scale, per iteration.
            for (0..8) |k| {
                const blk = j * 8 + k;

                const vy0 = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy0 + blk * 16);
                const vy1 = ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i8x16, qy1 + blk * 16);

                const block_scale: ggml.cpu.quants.arch.neon.i32x4 = .{
                    x0[i].scales[blk],
                    x0[i].scales[blk],
                    x1[i].scales[blk],
                    x1[i].scales[blk],
                };

                // Four results at once, as an outer product.
                const vx_l = ggml.cpu.quants.arch.neon.zip1q_s64(vx0[k], vx1[k]);
                const vx_h = ggml.cpu.quants.arch.neon.zip2q_s64(vx0[k], vx1[k]);
                const vy_l = ggml.cpu.quants.arch.neon.zip1q_s64(vy0, vy1);
                const vy_h = ggml.cpu.quants.arch.neon.zip2q_s64(vy0, vy1);
                var vr: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
                vr = ggml.cpu.quants.arch.neon.mmlaq_s32(vr, vx_l, vy_l);
                vr = ggml.cpu.quants.arch.neon.mmlaq_s32(vr, vx_h, vy_h);

                // Will not overflow: block_scale * sum_256(int6*int8) is
                // under 2^30.
                visum = ggml.cpu.quants.arch.neon.mlaq_s32(visum, vr, block_scale);
            }
        }

        // The bias, then the super-block scale.
        const q8sums0 = [2]ggml.cpu.quants.arch.neon.i16x8{ ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, &y0[i].bsums), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, y0[i].bsums[8..]) };
        const q8sums1 = [2]ggml.cpu.quants.arch.neon.i16x8{ ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, &y1[i].bsums), ggml.cpu.quants.arch.neon.loadFrom(ggml.cpu.quants.arch.neon.i16x8, y1[i].bsums[8..]) };
        const q6scales0 = q6Scales(&x0[i].scales);
        const q6scales1 = q6Scales(&x1[i].scales);

        const bias: ggml.cpu.quants.arch.neon.i32x4 = .{
            q6Bias(q8sums0, q6scales0),
            q6Bias(q8sums1, q6scales0),
            q6Bias(q8sums0, q6scales1),
            q6Bias(q8sums1, q6scales1),
        };
        const vibias = ggml.cpu.quants.arch.neon.mulq_n_s32(bias, 32);

        const superblock_scale: ggml.cpu.quants.arch.neon.f32x4 = .{
            f(x0[i].d) * y0[i].d,
            f(x0[i].d) * y1[i].d,
            f(x1[i].d) * y0[i].d,
            f(x1[i].d) * y1[i].d,
        };

        visum = ggml.cpu.quants.arch.neon.sub(visum, vibias);
        vfsum = ggml.cpu.quants.arch.neon.mlaq_f32(vfsum, ggml.cpu.quants.arch.neon.cvt_f32_s32(visum), superblock_scale);
    }

    storeTwoColumns(s, bs, vfsum);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
