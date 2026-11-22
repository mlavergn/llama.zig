//! The NEON `q6_K` × `q8_K` interleaved gemv.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Not shaped like its K-quant siblings
//!
//! Three differences from `q4_k.zig` / `q5_k.zig`, each load-bearing:
//!
//! - **The accumulator is integer.** `acc[g]` sums `sb_acc * scale` with
//!   `vmlaq_s32` and only converts to `f32` once, at the end of the block.
//!   The others convert per sub-block.
//! - **The bias is the `- 32` zero point, not a `dmin`.** `q6_K` has no
//!   `dmin`; the bias is `sum(scale * bsum) << 5`, subtracted from the
//!   integer accumulator. The `<< 5` is the 32 of the format's zero point
//!   folded in.
//! - **The final combine is a plain multiply then add**, not an FMA:
//!   `acc_f32 += f32(acc) * sb_scale`. Two roundings, and `@mulAdd` here
//!   would be one. The siblings use `vfmaq_f32`; this one does not, and
//!   the difference is visible in the last bit.
//!
//! # `qh` is shifted only for the upper sub-blocks
//!
//! `if (sb > 1)` shifts `qh` right by two. Sub-blocks 0 and 1 read the low
//! pair of bits, 2 and 3 the high pair — a conditional shift, not the
//! running one `q5_K` uses.

const std = @import("std");
const ggml = @import("../../../module.zig");

// The `_generic` bodies, in `cpu/repack/`, that each export below falls to
// without `__ARM_FEATURE_DOTPROD` -- the C's trailing call after its `#if`.
// Zig's `aarch64-linux` baseline, `generic`, takes them; no Apple core does.

inline fn loadqu(p: [*]const u8) ggml.cpu.quants.arch.neon.u8x16 {
    return p[0..16].*;
}
inline fn load4qu(p: [*]const u8) [4]ggml.cpu.quants.arch.neon.u8x16 {
    return .{ p[0..16].*, p[16..32].*, p[32..48].*, p[48..64].* };
}
inline fn load4s16(p: [*]const i16) ggml.cpu.quants.arch.neon.i16x4 {
    return p[0..4].*;
}

const m4b: ggml.cpu.quants.arch.neon.u8x16 = @splat(0x0f);
const mask_lo: ggml.cpu.quants.arch.neon.u8x16 = @splat(0x03);
const mask_hi: ggml.cpu.quants.arch.neon.u8x16 = @splat(0x30);
const sh2: @Vector(16, u3) = @splat(2);
const sh4: @Vector(16, u3) = @splat(4);

/// Ports `ggml_gemv_q6_K_8x4_q8_K` (arch/arm/repack.cpp:1381 @b1b770d90).
pub export fn ggml_gemv_q6_K_8x4_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !ggml.cpu.quants.arch.neon.has_dotprod) return ggml.cpu.repack.q6_k.ggml_gemv_q6_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const col_groups: usize = 2;

    var acc_f32: [2]ggml.cpu.quants.arch.neon.f32x4 = undefined;
    const q8_ptr: [*]const ggml.impl.c.block_q8_K = @ptrCast(@alignCast(vy));

    var x: c_int = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const q6_ptr: [*]const ggml.cpu.repack.blocks.block_q6_Kx8 =
            @as([*]const ggml.cpu.repack.blocks.block_q6_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..2) |i| acc_f32[i] = @splat(0);

        var b: usize = 0;
        while (b < @as(usize, @intCast(nb))) : (b += 1) {
            const q6_d_0 = ggml.cpu.quants.arch.neon.cvt_f32_f16(ggml.cpu.quants.arch.neon.load_f16x4(@ptrCast(&q6_ptr[b].d)));
            const q6_d_1 = ggml.cpu.quants.arch.neon.cvt_f32_f16(ggml.cpu.quants.arch.neon.load_f16x4(@as([*]const u16, @ptrCast(&q6_ptr[b].d)) + 4));
            const q8_d: ggml.cpu.quants.arch.neon.f32x4 = @splat(q8_ptr[b].d);
            const sb_scale_0 = q6_d_0 * q8_d;
            const sb_scale_1 = q6_d_1 * q8_d;

            var acc: [col_groups]ggml.cpu.quants.arch.neon.i32x4 = .{ @splat(0), @splat(0) };

            // The sixteen sub-block scales, widened to i16 and spilled so
            // the lane loads below can index them freely.
            var q6_scales: [16 * 8]i16 = undefined;
            for (0..16) |i| {
                const sc: @Vector(8, i8) = @as([*]const i8, &q6_ptr[b].scales)[i * 8 ..][0..8].*;
                const widened: ggml.cpu.quants.arch.neon.i16x8 = ggml.cpu.quants.arch.neon.movl_s8(sc);
                q6_scales[i * 8 ..][0..8].* = widened;
            }

            var bias_lo: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
            var bias_hi: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
            {
                var i: usize = 0;
                while (i < 16) : (i += 4) {
                    const bsums_vec = load4s16(@as([*]const i16, @ptrCast(&q8_ptr[b].bsums)) + i);
                    inline for (0..4) |k| {
                        const lo = load4s16(@as([*]const i16, &q6_scales) + (i + k) * 8);
                        const hi = load4s16(@as([*]const i16, &q6_scales) + (i + k) * 8 + 4);
                        bias_lo = ggml.cpu.quants.arch.neon.mlal_lane_s16(bias_lo, lo, bsums_vec, k);
                        bias_hi = ggml.cpu.quants.arch.neon.mlal_lane_s16(bias_hi, hi, bsums_vec, k);
                    }
                }
            }
            // `<< 5` is the format's 32 zero point folded into the bias.
            // Shifted on the unsigned reinterpretation, as elsewhere:
            // Zig has no `<<%`, and the bits leaving a signed lane are
            // discarded.
            bias_lo = @bitCast(@as(@Vector(4, u32), @bitCast(bias_lo)) << @as(@Vector(4, u5), @splat(5)));
            bias_hi = @bitCast(@as(@Vector(4, u32), @bitCast(bias_hi)) << @as(@Vector(4, u5), @splat(5)));

            for (0..2) |half| {
                const ql_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].ql) + half * 512;
                const qh_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].qh) + half * 256;
                for (0..4) |sb| {
                    const q8_base_l: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + half * 128 + sb * 16;
                    const q8_base_h = q8_base_l + 64;
                    var q8_l: [4]ggml.cpu.quants.arch.neon.i8x16 = undefined;
                    var q8_h: [4]ggml.cpu.quants.arch.neon.i8x16 = undefined;
                    for (0..4) |i| {
                        // 32-bit dup, not 64: four bytes into every lane group.
                        const wl: u32 = @bitCast(@as([4]u8, @as([*]const u8, @ptrCast(q8_base_l + i * 4))[0..4].*));
                        const wh: u32 = @bitCast(@as([4]u8, @as([*]const u8, @ptrCast(q8_base_h + i * 4))[0..4].*));
                        q8_l[i] = @bitCast(@as(@Vector(4, u32), @splat(wl)));
                        q8_h[i] = @bitCast(@as(@Vector(4, u32), @splat(wh)));
                    }

                    const ql_off_base = sb * 256 / 2;
                    const qh_off_base = ql_off_base & 255;
                    var q6_ql: [8]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                    var q6_qh: [8]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                    q6_ql[0..4].* = load4qu(ql_base + ql_off_base);
                    q6_ql[4..8].* = load4qu(ql_base + ql_off_base + 64);
                    q6_qh[0..4].* = load4qu(qh_base + qh_off_base);
                    q6_qh[4..8].* = load4qu(qh_base + qh_off_base + 64);
                    if (sb > 1) {
                        for (0..8) |i| q6_qh[i] = q6_qh[i] >> sh2;
                    }

                    for (0..col_groups) |g| {
                        var sb_acc_l: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
                        var sb_acc_h: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
                        inline for (0..4) |chunk| {
                            const idx = chunk * 2 + g;
                            const qs_l = q6_ql[idx];
                            const qs_h = q6_qh[idx];
                            const qs_hh = qs_h & mask_hi;
                            const q6_l: ggml.cpu.quants.arch.neon.i8x16 = @bitCast(ggml.cpu.quants.arch.neon.sli_n_u8(qs_l & m4b, qs_h & mask_lo, 4));
                            const q6_h: ggml.cpu.quants.arch.neon.i8x16 = @bitCast((qs_l >> sh4) | qs_hh);
                            sb_acc_l = ggml.cpu.quants.arch.neon.dotq_s32(sb_acc_l, q6_l, q8_l[chunk]);
                            sb_acc_h = ggml.cpu.quants.arch.neon.dotq_s32(sb_acc_h, q6_h, q8_h[chunk]);
                        }
                        const scale_idx_l = half * 8 + sb;
                        const scale_idx_h = half * 8 + sb + 4;
                        const scale_vec_l = ggml.cpu.quants.arch.neon.movl_s16(load4s16(@as([*]const i16, &q6_scales) + scale_idx_l * 8 + g * 4));
                        const scale_vec_h = ggml.cpu.quants.arch.neon.movl_s16(load4s16(@as([*]const i16, &q6_scales) + scale_idx_h * 8 + g * 4));
                        acc[g] = ggml.cpu.quants.arch.neon.mlaq_s32(acc[g], sb_acc_l, scale_vec_l);
                        acc[g] = ggml.cpu.quants.arch.neon.mlaq_s32(acc[g], sb_acc_h, scale_vec_h);
                    }
                }
            }

            acc[0] = acc[0] -% bias_lo;
            acc[1] = acc[1] -% bias_hi;
            // Plain multiply then add -- two roundings. Not an FMA; the
            // other K-quant kernels use one here and this does not.
            const w_0123 = ggml.cpu.quants.arch.neon.cvt_f32_s32(acc[0]) * sb_scale_0;
            const w_4567 = ggml.cpu.quants.arch.neon.cvt_f32_s32(acc[1]) * sb_scale_1;
            acc_f32[0] = acc_f32[0] + w_0123;
            acc_f32[1] = acc_f32[1] + w_4567;
        }

        const base: usize = @intCast(x * ncols_interleaved);
        s[base..][0..4].* = acc_f32[0];
        s[base + 4 ..][0..4].* = acc_f32[1];
    }
}

/// Ports `ggml_gemv_q6_K_8x8_q8_K` (arch/arm/repack.cpp:1570 @b1b770d90).
///
/// The 8x4 kernel's arithmetic with a **third** accumulator shape: four
/// two-lane accumulators, one per column pair, where 8x4 uses two
/// four-lane ones. Each sub-block's four-lane dot result is folded to two
/// lanes by the **pairwise** `vpadd_s32` before the scale is applied, and
/// the four pairs are recombined into two `f32x4` only at the end.
///
/// The bias is computed four-lane exactly as in 8x4 and then split with
/// `vget_low/high_s32` to match.
pub export fn ggml_gemv_q6_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !ggml.cpu.quants.arch.neon.has_dotprod) return ggml.cpu.repack.q6_k.ggml_gemv_q6_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const col_pairs: usize = 4;

    var acc_f32: [2]ggml.cpu.quants.arch.neon.f32x4 = undefined;
    const q8_ptr: [*]const ggml.impl.c.block_q8_K = @ptrCast(@alignCast(vy));

    var x: c_int = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const q6_ptr: [*]const ggml.cpu.repack.blocks.block_q6_Kx8 =
            @as([*]const ggml.cpu.repack.blocks.block_q6_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        acc_f32[0] = @splat(0);
        acc_f32[1] = @splat(0);

        var b: usize = 0;
        while (b < @as(usize, @intCast(nb))) : (b += 1) {
            const q6_d_0 = ggml.cpu.quants.arch.neon.cvt_f32_f16(ggml.cpu.quants.arch.neon.load_f16x4(@ptrCast(&q6_ptr[b].d)));
            const q6_d_1 = ggml.cpu.quants.arch.neon.cvt_f32_f16(ggml.cpu.quants.arch.neon.load_f16x4(@as([*]const u16, @ptrCast(&q6_ptr[b].d)) + 4));
            const q8_d: ggml.cpu.quants.arch.neon.f32x4 = @splat(q8_ptr[b].d);
            const sb_scale_0 = q6_d_0 * q8_d;
            const sb_scale_1 = q6_d_1 * q8_d;

            var acc: [col_pairs]ggml.cpu.quants.arch.neon.i32x2 = @splat(@as(ggml.cpu.quants.arch.neon.i32x2, @splat(0)));

            var q6_scales: [16 * 8]i16 = undefined;
            for (0..16) |i| {
                const sc: @Vector(8, i8) = @as([*]const i8, &q6_ptr[b].scales)[i * 8 ..][0..8].*;
                const widened: ggml.cpu.quants.arch.neon.i16x8 = ggml.cpu.quants.arch.neon.movl_s8(sc);
                q6_scales[i * 8 ..][0..8].* = widened;
            }

            var bias_lo: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
            var bias_hi: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
            {
                var i: usize = 0;
                while (i < 16) : (i += 4) {
                    const bsums_vec = load4s16(@as([*]const i16, @ptrCast(&q8_ptr[b].bsums)) + i);
                    inline for (0..4) |k| {
                        const lo = load4s16(@as([*]const i16, &q6_scales) + (i + k) * 8);
                        const hi = load4s16(@as([*]const i16, &q6_scales) + (i + k) * 8 + 4);
                        bias_lo = ggml.cpu.quants.arch.neon.mlal_lane_s16(bias_lo, lo, bsums_vec, k);
                        bias_hi = ggml.cpu.quants.arch.neon.mlal_lane_s16(bias_hi, hi, bsums_vec, k);
                    }
                }
            }
            bias_lo = @bitCast(@as(@Vector(4, u32), @bitCast(bias_lo)) << @as(@Vector(4, u5), @splat(5)));
            bias_hi = @bitCast(@as(@Vector(4, u32), @bitCast(bias_hi)) << @as(@Vector(4, u5), @splat(5)));

            for (0..2) |half| {
                const ql_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].ql) + half * 512;
                const qh_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].qh) + half * 256;
                for (0..4) |sb| {
                    const q8_base_l: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + half * 128 + sb * 16;
                    const q8_base_h = q8_base_l + 64;
                    // Two 64-bit dups here, where 8x4 uses four 32-bit ones.
                    var q8_l: [2]ggml.cpu.quants.arch.neon.i8x16 = undefined;
                    var q8_h: [2]ggml.cpu.quants.arch.neon.i8x16 = undefined;
                    for (0..2) |i| {
                        q8_l[i] = ggml.cpu.quants.arch.neon.dupq_i8x16_from8(q8_base_l + i * 8);
                        q8_h[i] = ggml.cpu.quants.arch.neon.dupq_i8x16_from8(q8_base_h + i * 8);
                    }

                    const ql_off_base = sb * 256 / 2;
                    const qh_off_base = ql_off_base & 255;
                    var q6_ql: [8]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                    var q6_qh: [8]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                    q6_ql[0..4].* = load4qu(ql_base + ql_off_base);
                    q6_ql[4..8].* = load4qu(ql_base + ql_off_base + 64);
                    q6_qh[0..4].* = load4qu(qh_base + qh_off_base);
                    q6_qh[4..8].* = load4qu(qh_base + qh_off_base + 64);
                    if (sb > 1) {
                        for (0..8) |i| q6_qh[i] = q6_qh[i] >> sh2;
                    }

                    for (0..col_pairs) |cp| {
                        var sb_acc_l: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
                        var sb_acc_h: ggml.cpu.quants.arch.neon.i32x4 = @splat(0);
                        inline for (0..2) |chunk| {
                            const idx = chunk * 4 + cp;
                            const qs_l = q6_ql[idx];
                            const qs_h = q6_qh[idx];
                            const qs_hh = qs_h & mask_hi;
                            const q6_l: ggml.cpu.quants.arch.neon.i8x16 = @bitCast(ggml.cpu.quants.arch.neon.sli_n_u8(qs_l & m4b, qs_h & mask_lo, 4));
                            const q6_h: ggml.cpu.quants.arch.neon.i8x16 = @bitCast((qs_l >> sh4) | qs_hh);
                            sb_acc_l = ggml.cpu.quants.arch.neon.dotq_s32(sb_acc_l, q6_l, q8_l[chunk]);
                            sb_acc_h = ggml.cpu.quants.arch.neon.dotq_s32(sb_acc_h, q6_h, q8_h[chunk]);
                        }
                        // Four lanes folded to two, pairwise.
                        const sum_l = ggml.cpu.quants.arch.neon.padd_s32(ggml.cpu.quants.arch.neon.low(sb_acc_l), ggml.cpu.quants.arch.neon.high(sb_acc_l));
                        const sum_h = ggml.cpu.quants.arch.neon.padd_s32(ggml.cpu.quants.arch.neon.low(sb_acc_h), ggml.cpu.quants.arch.neon.high(sb_acc_h));

                        const scale_idx_l = half * 8 + sb;
                        const scale_idx_h = half * 8 + sb + 4;
                        const scale_vec_l: ggml.cpu.quants.arch.neon.i32x2 = .{
                            q6_scales[scale_idx_l * 8 + cp * 2],
                            q6_scales[scale_idx_l * 8 + cp * 2 + 1],
                        };
                        const scale_vec_h: ggml.cpu.quants.arch.neon.i32x2 = .{
                            q6_scales[scale_idx_h * 8 + cp * 2],
                            q6_scales[scale_idx_h * 8 + cp * 2 + 1],
                        };
                        acc[cp] = ggml.cpu.quants.arch.neon.mla_s32x2(acc[cp], sum_l, scale_vec_l);
                        acc[cp] = ggml.cpu.quants.arch.neon.mla_s32x2(acc[cp], sum_h, scale_vec_h);
                    }
                }
            }

            acc[0] = acc[0] -% ggml.cpu.quants.arch.neon.low(bias_lo);
            acc[1] = acc[1] -% ggml.cpu.quants.arch.neon.high(bias_lo);
            acc[2] = acc[2] -% ggml.cpu.quants.arch.neon.low(bias_hi);
            acc[3] = acc[3] -% ggml.cpu.quants.arch.neon.high(bias_hi);

            // Plain multiply then add, as in the 8x4 kernel.
            const w_01 = ggml.cpu.quants.arch.neon.cvt_f32_s32x2(acc[0]) * ggml.cpu.quants.arch.neon.low(sb_scale_0);
            const w_23 = ggml.cpu.quants.arch.neon.cvt_f32_s32x2(acc[1]) * ggml.cpu.quants.arch.neon.high(sb_scale_0);
            const w_45 = ggml.cpu.quants.arch.neon.cvt_f32_s32x2(acc[2]) * ggml.cpu.quants.arch.neon.low(sb_scale_1);
            const w_67 = ggml.cpu.quants.arch.neon.cvt_f32_s32x2(acc[3]) * ggml.cpu.quants.arch.neon.high(sb_scale_1);
            acc_f32[0] = acc_f32[0] + ggml.cpu.quants.arch.neon.combine(w_01, w_23);
            acc_f32[1] = acc_f32[1] + ggml.cpu.quants.arch.neon.combine(w_45, w_67);
        }

        const base: usize = @intCast(x * ncols_interleaved);
        s[base..][0..4].* = acc_f32[0];
        s[base + 4 ..][0..4].* = acc_f32[1];
    }
}

/// Ports `ggml_gemm_q6_K_8x4_q8_K` (arch/arm/repack.cpp:4782 @b1b770d90).
///
/// **The zero point moves.** The gemv kernels fold the `-32` into an
/// integer bias and subtract it from the accumulator at the end; this one
/// subtracts `m32s` from the assembled `i8` quants directly, before the
/// dot product. Same arithmetic, different place, and there is no bias
/// term here at all.
///
/// The combine is `vmlaq_f32` — fused — unlike the gemv kernels in this
/// file, which multiply and add separately. The two halves of `q6_K` do
/// not agree with each other on this, and both are reproduced as written.
pub export fn ggml_gemm_q6_K_8x4_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    if (comptime !ggml.cpu.quants.arch.neon.has_dotprod) return ggml.cpu.repack.q6_k.ggml_gemm_q6_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
    const nb = @divTrunc(n, 256);
    const ncols_interleaved: c_int = 8;
    const q8_k_blocklen: usize = 4;
    const col_groups: usize = 2;
    const acc_size: usize = q8_k_blocklen * col_groups;
    const m32s: @Vector(16, i8) = @splat(32);

    var acc_f32: [acc_size]ggml.cpu.quants.arch.neon.f32x4 = undefined;

    var y: c_int = 0;
    while (y < @divTrunc(nr, @as(c_int, q8_k_blocklen))) : (y += 1) {
        const q8_ptr: [*]const ggml.cpu.repack.blocks.block_q8_Kx4 =
            @as([*]const ggml.cpu.repack.blocks.block_q8_Kx4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));

        var x: c_int = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const q6_ptr: [*]const ggml.cpu.repack.blocks.block_q6_Kx8 =
                @as([*]const ggml.cpu.repack.blocks.block_q6_Kx8, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

            for (0..acc_size) |i| acc_f32[i] = @splat(0);

            var b: usize = 0;
            while (b < @as(usize, @intCast(nb))) : (b += 1) {
                const q6_d_0123 = ggml.cpu.quants.arch.neon.cvt_f32_f16(ggml.cpu.quants.arch.neon.load_f16x4(@ptrCast(&q6_ptr[b].d)));
                const q6_d_4567 = ggml.cpu.quants.arch.neon.cvt_f32_f16(ggml.cpu.quants.arch.neon.load_f16x4(@as([*]const u16, @ptrCast(&q6_ptr[b].d)) + 4));
                const q8_d_0123: ggml.cpu.quants.arch.neon.f32x4 = @as([*]const f32, &q8_ptr[b].d)[0..4].*;

                var sbd_scale_0123: [q8_k_blocklen]ggml.cpu.quants.arch.neon.f32x4 = undefined;
                var sbd_scale_4567: [q8_k_blocklen]ggml.cpu.quants.arch.neon.f32x4 = undefined;
                inline for (0..4) |row| {
                    const lane: ggml.cpu.quants.arch.neon.f32x4 = @splat(q8_d_0123[row]);
                    sbd_scale_0123[row] = q6_d_0123 * lane;
                    sbd_scale_4567[row] = q6_d_4567 * lane;
                }

                var acc_s32: [acc_size]ggml.cpu.quants.arch.neon.i32x4 = @splat(@as(ggml.cpu.quants.arch.neon.i32x4, @splat(0)));

                var q6_scales: [16 * 8]i16 = undefined;
                for (0..16) |i| {
                    const sc: @Vector(8, i8) = @as([*]const i8, &q6_ptr[b].scales)[i * 8 ..][0..8].*;
                    const widened: ggml.cpu.quants.arch.neon.i16x8 = ggml.cpu.quants.arch.neon.movl_s8(sc);
                    q6_scales[i * 8 ..][0..8].* = widened;
                }

                for (0..2) |half| {
                    const ql_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].ql) + half * 512;
                    const qh_base: [*]const u8 = @as([*]const u8, &q6_ptr[b].qh) + half * 256;
                    for (0..4) |sb| {
                        var acc_lo: [acc_size]ggml.cpu.quants.arch.neon.i32x4 = @splat(@as(ggml.cpu.quants.arch.neon.i32x4, @splat(0)));
                        var acc_hi: [acc_size]ggml.cpu.quants.arch.neon.i32x4 = @splat(@as(ggml.cpu.quants.arch.neon.i32x4, @splat(0)));

                        const q8_base_l: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + half * 512 + sb * 64;
                        const q8_base_h: [*]const i8 = @as([*]const i8, @ptrCast(&q8_ptr[b].qs)) + half * 512 + 256 + sb * 64;
                        var q8_l: [4]ggml.cpu.quants.arch.neon.i8x16 = undefined;
                        var q8_h: [4]ggml.cpu.quants.arch.neon.i8x16 = undefined;
                        for (0..4) |k| {
                            q8_l[k] = @bitCast(@as(@Vector(16, i8), (q8_base_l + 16 * k)[0..16].*));
                            q8_h[k] = @bitCast(@as(@Vector(16, i8), (q8_base_h + 16 * k)[0..16].*));
                        }

                        const ql_off_base = sb * 256 / 2;
                        const qh_off_base = ql_off_base & 255;
                        var ql_0123: [4]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                        var ql_4567: [4]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                        var qh_0123: [4]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                        var qh_4567: [4]ggml.cpu.quants.arch.neon.u8x16 = undefined;
                        for (0..4) |k| {
                            ql_0123[k] = loadqu(ql_base + ql_off_base + k * 32);
                            ql_4567[k] = loadqu(ql_base + ql_off_base + k * 32 + 16);
                            qh_0123[k] = loadqu(qh_base + qh_off_base + k * 32);
                            qh_4567[k] = loadqu(qh_base + qh_off_base + k * 32 + 16);
                        }
                        if (sb > 1) {
                            for (0..4) |k| {
                                qh_0123[k] = qh_0123[k] >> sh2;
                                qh_4567[k] = qh_4567[k] >> sh2;
                            }
                        }

                        for (0..4) |k| {
                            const hbit_lo_0123 = qh_0123[k] & mask_lo;
                            const hbit_hi_0123 = qh_0123[k] & mask_hi;
                            const hbit_lo_4567 = qh_4567[k] & mask_lo;
                            const hbit_hi_4567 = qh_4567[k] & mask_hi;

                            // The `- 32` is applied here, to the quants.
                            const q6_0123_lo: ggml.cpu.quants.arch.neon.i8x16 =
                                @as(@Vector(16, i8), @bitCast(ggml.cpu.quants.arch.neon.sli_n_u8(ql_0123[k] & m4b, hbit_lo_0123, 4))) -% m32s;
                            const q6_0123_hi: ggml.cpu.quants.arch.neon.i8x16 =
                                @as(@Vector(16, i8), @bitCast((ql_0123[k] >> sh4) | hbit_hi_0123)) -% m32s;
                            inline for (0..4) |r| {
                                acc_lo[r] = ggml.cpu.quants.arch.neon.dotq_laneq_s32(acc_lo[r], q6_0123_lo, q8_l[k], r);
                                acc_hi[r] = ggml.cpu.quants.arch.neon.dotq_laneq_s32(acc_hi[r], q6_0123_hi, q8_h[k], r);
                            }

                            const q6_4567_lo: ggml.cpu.quants.arch.neon.i8x16 =
                                @as(@Vector(16, i8), @bitCast(ggml.cpu.quants.arch.neon.sli_n_u8(ql_4567[k] & m4b, hbit_lo_4567, 4))) -% m32s;
                            const q6_4567_hi: ggml.cpu.quants.arch.neon.i8x16 =
                                @as(@Vector(16, i8), @bitCast((ql_4567[k] >> sh4) | hbit_hi_4567)) -% m32s;
                            inline for (0..4) |r| {
                                acc_lo[4 + r] = ggml.cpu.quants.arch.neon.dotq_laneq_s32(acc_lo[4 + r], q6_4567_lo, q8_l[k], r);
                                acc_hi[4 + r] = ggml.cpu.quants.arch.neon.dotq_laneq_s32(acc_hi[4 + r], q6_4567_hi, q8_h[k], r);
                            }
                        }

                        const scale_idx_l = half * 8 + sb;
                        const scale_idx_h = half * 8 + sb + 4;
                        for (0..col_groups) |g| {
                            const scale_vec_l = ggml.cpu.quants.arch.neon.movl_s16(load4s16(@as([*]const i16, &q6_scales) + scale_idx_l * 8 + g * 4));
                            const scale_vec_h = ggml.cpu.quants.arch.neon.movl_s16(load4s16(@as([*]const i16, &q6_scales) + scale_idx_h * 8 + g * 4));
                            const acc_offset = g * q8_k_blocklen;
                            for (0..q8_k_blocklen) |row| {
                                const idx = row * 2 + g;
                                acc_s32[idx] = ggml.cpu.quants.arch.neon.mlaq_s32(acc_s32[idx], acc_lo[acc_offset + row], scale_vec_l);
                                acc_s32[idx] = ggml.cpu.quants.arch.neon.mlaq_s32(acc_s32[idx], acc_hi[acc_offset + row], scale_vec_h);
                            }
                        }
                    }
                }

                inline for (0..4) |row| {
                    const idx0 = 2 * row;
                    const idx1 = 2 * row + 1;
                    acc_f32[idx0] = ggml.cpu.quants.arch.neon.fma_f32(acc_f32[idx0], ggml.cpu.quants.arch.neon.cvt_f32_s32(acc_s32[idx0]), sbd_scale_0123[row]);
                    acc_f32[idx1] = ggml.cpu.quants.arch.neon.fma_f32(acc_f32[idx1], ggml.cpu.quants.arch.neon.cvt_f32_s32(acc_s32[idx1]), sbd_scale_4567[row]);
                }
            }

            for (0..q8_k_blocklen) |i| {
                const row: usize = @intCast(y * @as(c_int, q8_k_blocklen) + @as(c_int, @intCast(i)));
                for (0..2) |j| {
                    const col: usize = @intCast(x * ncols_interleaved + @as(c_int, @intCast(j)) * 4);
                    s[row * bs + col ..][0..4].* = acc_f32[2 * i + j];
                }
            }
        }
    }
}

comptime {
    _ = std;
}
