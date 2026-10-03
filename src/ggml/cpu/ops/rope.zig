//! `rope` and `rope_back`, with the YaRN and multi-section caches they build.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Four fused multiply-adds
//!
//! `ops.cpp` compiles at `-ffp-contract=on`, and clang fuses `a*b + c` when it
//! is one expression. This file has four such expressions, and each is named
//! with `@mulAdd` where it occurs:
//!
//! - the YaRN blend `theta_interp*(1 - ramp_mix) + theta_extrap*ramp_mix`,
//! - the magnitude correction `1.0f + 0.1f*logf(...)`,
//! - the rotation's two outputs, `x0*cos - x1*sin` and `x0*sin + x1*cos`.
//!
//! Where both operands of the `+` or `-` are products, clang fuses the
//! **left** one and rounds the right one on its own, so the rotation is
//! `fma(x0, cos, -(x1*sin))` and not the mirror image. The rotation is the
//! hot loop of every transformer layer on this backend; its last bit is what
//! `make ops-diff`'s `rope` and `rope_ext` cases compare.
//!
//! # `rope_back` is `rope` with the sine negated
//!
//! The C has one templated kernel taking a `forward` flag, and both exported
//! entry points call it. Here too.
//!
//! # Loop index names
//!
//! `i0`, `i1`, `i2`, `i3` are Zig integer type names. Renamed `j0`, `j1`,
//! `j2`, `j3`, digit for digit.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

extern fn powf(x: f32, y: f32) f32;
extern fn logf(x: f32) f32;

/// Typed pointer at a byte offset; see `common.ptr`.
const at = common.ptr;

/// Stride as `i64`; see `common.sz`.
const s = common.sz;

/// Ports `rope_yarn_ramp` (ops.cpp:5818 @c1d0e7a00).
///
/// `MIN` and `MAX` are the C's ternary macros, mixing `int` literals with
/// `float`; the comparisons happen in `float`.
fn ropeYarnRamp(low: f32, high: f32, j0: c_int) f32 {
    const span = high - low;
    const den: f32 = if (0.001 > span) 0.001 else span;
    const y = (@as(f32, @floatFromInt(@divTrunc(j0, 2))) - low) / den;
    const lo: f32 = if (0 > y) 0 else y;
    const hi: f32 = if (1 < lo) 1 else lo;
    return 1 - hi;
}

/// The cosine and sine `rope_yarn` writes through its two out-pointers.
const CosSin = struct { cos: f32, sin: f32 };

/// Ports `rope_yarn` (ops.cpp:5825 @c1d0e7a00).
///
/// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from
/// https://github.com/jquesnelle/yarn — MIT licensed. Copyright (c) 2023
/// Jeffrey Quesnelle and Bowen Peng.
fn ropeYarn(
    theta_extrap: f32,
    freq_scale: f32,
    corr_dims: [2]f32,
    j0: i64,
    ext_factor: f32,
    mscale_in: f32,
) CosSin {
    var mscale = mscale_in;
    // Get n-d rotational scaling corrected for extrapolation
    const theta_interp = freq_scale * theta_extrap;
    var theta = theta_interp;
    if (ext_factor != 0.0) {
        const ramp_mix = ropeYarnRamp(corr_dims[0], corr_dims[1], @truncate(j0)) * ext_factor;
        // One expression, two products: clang fuses the left one.
        theta = @mulAdd(f32, theta_interp, 1 - ramp_mix, theta_extrap * ramp_mix);

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= @mulAdd(f32, 0.1, logf(1.0 / freq_scale), 1.0);
    }
    // `cosf(theta)` and `sinf(theta)`: clang folds the pair into one
    // `__sincosf_stret` call, which rounds differently -- see `common.sinCos`.
    const sc = common.sinCos(theta);
    return .{ .cos = sc.cos * mscale, .sin = sc.sin * mscale };
}

/// Ports `ggml_rope_cache_init` (ops.cpp:5842 @c1d0e7a00).
fn ropeCacheInit(
    theta_base: f32,
    freq_scale: f32,
    freq_factors: ?[*]const f32,
    corr_dims: [2]f32,
    ne0: i64,
    ext_factor: f32,
    mscale: f32,
    cache: [*]f32,
    sin_sign: f32,
    theta_scale: f32,
) void {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    var theta = theta_base;
    var j0: i64 = 0;
    while (j0 < ne0) : (j0 += 2) {
        const k: usize = @intCast(j0);
        const ff: f32 = if (freq_factors) |f| f[k / 2] else 1.0;
        const cs = ropeYarn(theta / ff, freq_scale, corr_dims, j0, ext_factor, mscale);
        cache[k + 0] = cs.cos;
        cache[k + 1] = cs.sin;
        cache[k + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

/// Ports `ggml_mrope_cache_init` (ops.cpp:5858 @c1d0e7a00).
fn mropeCacheInit(
    theta_base_t: f32,
    theta_base_h: f32,
    theta_base_w: f32,
    theta_base_e: f32,
    sections: [4]c_int,
    is_imrope: bool,
    indep_sects: bool,
    freq_scale: f32,
    freq_factors: ?[*]const f32,
    corr_dims: [2]f32,
    ne0: i64,
    ext_factor: f32,
    mscale: f32,
    cache: [*]f32,
    sin_sign: f32,
    theta_scale: f32,
) void {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    var theta_t = theta_base_t;
    var theta_h = theta_base_h;
    var theta_w = theta_base_w;
    var theta_e = theta_base_e; // extra position id for vision encoder
    const sect_dims: c_int = sections[0] + sections[1] + sections[2] + sections[3];
    const sec_w: c_int = sections[1] + sections[0];
    const sec_e: c_int = sections[2] + sec_w;
    impl.assert(sect_dims <= ne0, "sect_dims <= ne0");

    var j0: i64 = 0;
    while (j0 < ne0) : (j0 += 2) {
        const k: usize = @intCast(j0);
        const ff: f32 = if (freq_factors) |f| f[k / 2] else 1.0;

        const sector: c_int = @intCast(@rem(@divTrunc(j0, 2), sect_dims));
        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            } else if (sector == sections[0]) {
                theta_h = theta_base_h;
            } else if (sector == sec_w) {
                theta_w = theta_base_w;
            } else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        var theta = theta_t;
        if (is_imrope) { // qwen3vl apply interleaved mrope
            if (@rem(sector, 3) == 1 and sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (@rem(sector, 3) == 2 and sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (@rem(sector, 3) == 0 and sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
        } else {
            if (sector >= sections[0] and sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w and sector < sec_w + sections[2]) {
                theta = theta_w;
            } else if (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        const cs = ropeYarn(theta / ff, freq_scale, corr_dims, j0, ext_factor, mscale);
        cache[k + 0] = cs.cos;
        cache[k + 1] = cs.sin;
        cache[k + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

/// Ports `rotate_pairs` (ops.cpp:5930 @c1d0e7a00).
///
/// Parameters:
/// - `n`: elements to rotate, stepped two at a time.
/// - `n_offset`: distance between the two halves of a pair.
/// - `scale`: 2 everywhere except `GGML_ROPE_TYPE_NORMAL`, where it is 1 so
///   that `ic = i0` and the pair is adjacent.
inline fn rotatePairs(comptime T: type, n: i64, n_offset: i64, cache: [*]const f32, src_data: [*]const T, dst_data: [*]T, scale: i64) void {
    const off: usize = @intCast(n_offset);
    var j0: i64 = 0;
    while (j0 < n) : (j0 += 2) {
        const ic: usize = @intCast(@divTrunc(j0, scale)); // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

        const k: usize = @intCast(j0);
        const cos_theta = cache[k + 0];
        const sin_theta = cache[k + 1];

        const src = src_data + ic;
        const dst = dst_data + ic;

        const x0 = common.toF32(T, src[0]);
        const x1 = common.toF32(T, src[off]);

        // Both are one expression over two products; clang fuses the left.
        dst[0] = common.fromF32(T, @mulAdd(f32, x0, cos_theta, -(x1 * sin_theta)));
        dst[off] = common.fromF32(T, @mulAdd(f32, x0, sin_theta, x1 * cos_theta));
    }
}

/// Ports `ggml_compute_forward_rope_flt` (ops.cpp:5949 @c1d0e7a00).
///
/// Parameters:
/// - `T`: `f32` or `ggml_fp16_t`.
/// - `forward`: false for `rope_back`, which rotates by the transpose — the
///   same rotation with the sine negated.
fn ropeFlt(comptime T: type, params: *const ComputeParams, dst: *Tensor, forward: bool) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const src2: ?*Tensor = dst.src[2];

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(src1.type == c.GGML_TYPE_I32, "src1->type == GGML_TYPE_I32");

    //const n_past     = ((int32_t *) dst->op_params)[0];
    const n_dims: c_int = impl.getOpParamsI32(dst, 1);
    const mode: c_int = impl.getOpParamsI32(dst, 2);
    //const n_ctx      = ((int32_t *) dst->op_params)[3];
    const n_ctx_orig: c_int = impl.getOpParamsI32(dst, 4);

    const freq_base = impl.getOpParamsF32(dst, 5);
    const freq_scale = impl.getOpParamsF32(dst, 6);
    const ext_factor = impl.getOpParamsF32(dst, 7);
    const attn_factor = impl.getOpParamsF32(dst, 8);
    const beta_fast = impl.getOpParamsF32(dst, 9);
    const beta_slow = impl.getOpParamsF32(dst, 10);
    const sections = [4]c_int{
        impl.getOpParamsI32(dst, 11),
        impl.getOpParamsI32(dst, 12),
        impl.getOpParamsI32(dst, 13),
        impl.getOpParamsI32(dst, 14),
    };

    const n_offs: c_int = impl.getOpParamsI32(dst, 15);

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.nb0 == l.nb00, "nb0 == nb00");
    impl.assert(l.nb0 == @sizeOf(T), "nb0 == sizeof(T)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = c.ggml_nrows(dst);

    impl.assert(n_dims <= l.ne0, "n_dims <= ne0");
    impl.assert(@rem(n_dims, 2) == 0, "n_dims % 2 == 0");

    impl.assert(n_offs >= 0, "n_offs >= 0");
    impl.assert(@rem(n_offs, 2) == 0, "n_offs % 2 == 0");
    impl.assert(n_offs + n_dims <= l.ne0, "n_offs + n_dims <= ne0");

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    // row index used to determine which thread to use
    var ir: i64 = 0;

    const theta_scale = powf(freq_base, -2.0 / @as(f32, @floatFromInt(n_dims)));

    var corr_dims: [2]f32 = undefined;
    c.ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr_dims);

    const is_imrope = mode == c.GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    const mrope_used = (mode & c.GGML_ROPE_TYPE_MROPE) != 0; // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    const is_vision = mode == c.GGML_ROPE_TYPE_VISION;

    if (mrope_used) {
        impl.assert(sections[0] > 0 or sections[1] > 0 or sections[2] > 0, "sections[0] > 0 || sections[1] > 0 || sections[2] > 0");
    }

    if (is_vision) {
        impl.assert(n_dims == @divTrunc(l.ne0, 2), "n_dims == ne0/2");
        impl.assert(n_offs == 0, "n_offs == 0");
    }

    var freq_factors: ?[*]const f32 = null;
    if (src2) |f| {
        impl.assert(f.type == c.GGML_TYPE_F32, "src2->type == GGML_TYPE_F32");
        impl.assert(f.ne[0] >= @divTrunc(n_dims, 2), "src2->ne[0] >= n_dims / 2");
        freq_factors = at(f32, f.data, 0);
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const sin_sign: f32 = if (forward) 1.0 else -1.0;

    const pos = at(i32, src1.data, 0);

    var last_i2: i64 = -1;

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));

    var j3: i64 = 0;
    while (j3 < l.ne3) : (j3 += 1) { // batch
        var j2: i64 = 0;
        while (j2 < l.ne2) : (j2 += 1) { // seq-len
            var j1: i64 = 0;
            while (j1 < l.ne1) : (j1 += 1) { // attn-heads
                // `if (ir++ < ir0) continue; if (ir > ir1) break;` -- and the
                // `break` leaves only this innermost loop.
                const before = ir;
                ir += 1;
                if (before < ir0) continue; // skip rows mapped to other threads
                if (ir > ir1) break;

                const cache = wbase + (@as(usize, @intCast(l.ne0)) + common.cache_line_size_f32) * @as(usize, @intCast(ith));
                if (last_i2 != j2) {
                    const k: usize = @intCast(j2);
                    const ne2: usize = @intCast(l.ne2);
                    if (!mrope_used) {
                        const p: i64 = pos[k];
                        ropeCacheInit(@floatFromInt(p), freq_scale, freq_factors, corr_dims, l.ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    } else {
                        const p_t: i64 = pos[k];
                        const p_h: i64 = pos[k + ne2];
                        const p_w: i64 = pos[k + ne2 * 2];
                        const p_e: i64 = pos[k + ne2 * 3];
                        mropeCacheInit(
                            @floatFromInt(p_t),
                            @floatFromInt(p_h),
                            @floatFromInt(p_w),
                            @floatFromInt(p_e),
                            sections,
                            is_imrope,
                            is_vision,
                            freq_scale,
                            freq_factors,
                            corr_dims,
                            l.ne0,
                            ext_factor,
                            attn_factor,
                            cache,
                            sin_sign,
                            theta_scale,
                        );
                    }

                    last_i2 = j2;
                }

                const src = at(T, src0.data, j3 * s(l.nb03) + j2 * s(l.nb02) + j1 * s(l.nb01));
                const dst_data = at(T, dst.data, j3 * s(l.nb3) + j2 * s(l.nb2) + j1 * s(l.nb1));

                const offs: usize = @intCast(n_offs);
                switch (mode) {
                    c.GGML_ROPE_TYPE_NORMAL => rotatePairs(T, n_dims, 1, cache, src + offs, dst_data + offs, 1),
                    c.GGML_ROPE_TYPE_NEOX,
                    c.GGML_ROPE_TYPE_MROPE,
                    c.GGML_ROPE_TYPE_IMROPE,
                    => rotatePairs(T, n_dims, @divTrunc(n_dims, 2), cache, src + offs, dst_data + offs, 2),
                    c.GGML_ROPE_TYPE_VISION => rotatePairs(T, l.ne0, n_dims, cache, src, dst_data, 2),
                    else => impl.abort("rope type not supported"),
                }

                if (!is_vision) {
                    // fill the remain channels with data from src tensor
                    var j0: i64 = 0;
                    while (j0 < l.ne0) : (j0 += 2) {
                        if (j0 == n_offs) {
                            j0 += n_dims - 2; // skip the rotated channels
                            continue;
                        }
                        const sp = at(T, src0.data, j3 * s(l.nb03) + j2 * s(l.nb02) + j1 * s(l.nb01) + j0 * s(l.nb00));
                        const dp = at(T, dst.data, j3 * s(l.nb3) + j2 * s(l.nb2) + j1 * s(l.nb1) + j0 * s(l.nb0));

                        dp[0] = sp[0];
                        dp[1] = sp[1];
                    }
                }
            } //attn-heads
        }
    }
}

/// Ports `ggml_compute_forward_rope` (ops.cpp:6107 @c1d0e7a00).
pub export fn ggml_compute_forward_rope(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F16 => ropeFlt(c.ggml_fp16_t, params, dst, true),
        c.GGML_TYPE_F32 => ropeFlt(f32, params, dst, true),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_rope_back` (ops.cpp:6131 @c1d0e7a00).
pub export fn ggml_compute_forward_rope_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F16 => ropeFlt(c.ggml_fp16_t, params, dst, false),
        c.GGML_TYPE_F32 => ropeFlt(f32, params, dst, false),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the YaRN ramp clamps to [0, 1] and halves i0 as an integer" {
    // low 2, high 6: i0 = 4 gives y = 0, ramp 1; i0 = 13 gives i0/2 = 6
    // (not 6.5), y = 1, ramp 0.
    try std.testing.expectEqual(@as(f32, 1), ropeYarnRamp(2, 6, 4));
    try std.testing.expectEqual(@as(f32, 0), ropeYarnRamp(2, 6, 13));
    try std.testing.expectEqual(@as(f32, 0.5), ropeYarnRamp(2, 6, 8));
}

test "the rotation fuses the left product" {
    // x0*cos - x1*sin with x0 = cos = 1 + 2^-12, x1 = 1, sin = 1 + 2^-11.
    // x0*cos is 1 + 2^-11 + 2^-24, which f32 cannot hold; only the fused
    // form keeps the 2^-24 once x1*sin cancels the rest.
    const e: f32 = 1.0 + 0x1p-12;
    const cache = [_]f32{ e, 1 + 0x1p-11 };
    const src = [_]f32{ e, 1 };
    var dst: [2]f32 = undefined;
    rotatePairs(f32, 2, 1, &cache, &src, &dst, 1);
    try std.testing.expectEqual(@as(f32, 0x1p-24), dst[0]);
}
