//! `ROPE`, `ROPE_BACK`, the fused `ROPE + VIEW + SET_ROWS` and the fused
//! `RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)` that `ggml_cuda_try_fuse`
//! calls.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/rope.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/rope.cuh` at v1.0.0 (`b1b770d90`). The
//! kernels -- `rope_norm`, `rope_neox`, `rope_multi`, `rope_vision`,
//! `rms_norm_mul_rope_f32` -- stay CUDA.
//!
//! # Launch styles
//!
//! Not uniform, and kept as the C has them: `rope_norm` and `rope_vision`
//! launch with `<<<>>>`; `rope_neox`, `rope_multi` and
//! `rms_norm_mul_rope_f32` through `ggml_cuda_kernel_launch` (PDL).

const std = @import("std");
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `CUDA_ROPE_BLOCK_SIZE` (rope.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `rope_corr_dims` (rope.cu:6 @b1b770d90).
const CorrDims = extern struct {
    v: [2]f32,
};

/// Ports `mrope_sections` (rope.cu:11 @b1b770d90).
const MropeSections = extern struct {
    v: [4]c_int,
};

const CPtr = ?*const anyopaque;
const Ptr = ?*anyopaque;

/// The arguments every `rope_*_cuda` launcher shares, in the C's
/// parameter order.
const Args = struct {
    x: CPtr,
    dst: Ptr,
    ne00: c_int,
    ne01: c_int,
    ne02: c_int,
    s01: c_int,
    s02: c_int,
    s03: c_int,
    s1: c_int,
    s2: c_int,
    s3: c_int,
    n_dims: c_int,
    n_offs: c_int,
    nr: c_int,
    pos: CPtr,
    freq_scale: f32,
    freq_base: f32,
    ext_factor: f32,
    attn_factor: f32,
    corr_dims: CorrDims,
    freq_factors: CPtr,
};

/// `block_nums` and `block_dims` for a `rope_*_cuda` launcher, after its
/// `GGML_ASSERT(ne00 % 2 == 0)`.
fn grid(a: *const Args) [2]ggml.cuda.kernels.Dim3 {
    ggml.impl.assert(@rem(a.ne00, 2) == 0, "ne00 % 2 == 0");
    const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = 1, .y = block_size, .z = 1 };
    const n_blocks_x: c_int = @divTrunc(a.ne00 + 2 * block_size - 1, 2 * block_size);
    const block_nums: ggml.cuda.kernels.Dim3 = .{ .x = @bitCast(a.nr), .y = @bitCast(n_blocks_x), .z = 1 };
    return .{ block_nums, block_dims };
}

fn thetaScale(a: *const Args) f32 {
    return stdlib.libm.powf(a.freq_base, -2.0 / @as(f32, @floatFromInt(a.n_dims)));
}

fn key4(comptime name: []const u8, comptime forward: bool, comptime has_ff: bool, comptime T: type, comptime D: type) []const u8 {
    return ggml.cuda.kernels.tmpl(name, &.{ ggml.cuda.kernels.boolean(forward), ggml.cuda.kernels.boolean(has_ff), ggml.cuda.kernels.typeName(T), ggml.cuda.kernels.typeName(D) });
}

fn key3(comptime name: []const u8, comptime forward: bool, comptime has_ff: bool, comptime T: type) []const u8 {
    return ggml.cuda.kernels.tmpl(name, &.{ ggml.cuda.kernels.boolean(forward), ggml.cuda.kernels.boolean(has_ff), ggml.cuda.kernels.typeName(T) });
}

/// Ports `rope_norm_cuda` (rope.cu:368 @b1b770d90).
fn ropeNormCuda(comptime forward: bool, comptime T: type, comptime D: type, a: Args, row_indices: CPtr, set_rows_stride: c_int, inplace: bool, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const g = grid(&a);
    const theta_scale = thetaScale(&a);
    const p: ggml.cuda.kernels.LaunchParams = .{ .grid = g[0], .block = g[1], .shmem = 0, .stream = stream };
    const args = .{
        a.x,         a.dst,          a.ne00,      a.ne01,          a.ne02,  a.s01,        a.s02,        a.s03,         a.s1,
        a.s2,        a.s3,           a.n_dims,    a.n_offs,        a.pos,   a.freq_scale, a.ext_factor, a.attn_factor, a.corr_dims,
        theta_scale, a.freq_factors, row_indices, set_rows_stride, inplace,
    };

    if (a.freq_factors == null) {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("rope", key4("rope_norm", forward, false, T, D)), p, args);
    } else {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("rope", key4("rope_norm", forward, true, T, D)), p, args);
    }
}

/// Ports `rope_neox_cuda` (rope.cu:412 @b1b770d90).
fn ropeNeoxCuda(comptime forward: bool, comptime T: type, comptime D: type, a: Args, row_indices: CPtr, set_rows_stride: c_int, inplace: bool, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const g = grid(&a);
    const theta_scale = thetaScale(&a);
    const launch_params: ggml.cuda.kernels.LaunchParams = .{ .grid = g[0], .block = g[1], .shmem = 0, .stream = stream };
    const args = .{
        a.x,         a.dst,          a.ne00,      a.ne01,          a.ne02,  a.s01,        a.s02,        a.s03,         a.s1,
        a.s2,        a.s3,           a.n_dims,    a.n_offs,        a.pos,   a.freq_scale, a.ext_factor, a.attn_factor, a.corr_dims,
        theta_scale, a.freq_factors, row_indices, set_rows_stride, inplace,
    };

    if (a.freq_factors == null) {
        ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key4("rope_neox", forward, false, T, D)), launch_params, args);
    } else {
        ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key4("rope_neox", forward, true, T, D)), launch_params, args);
    }
}

/// Ports `rope_multi_cuda` (rope.cu:457 @b1b770d90). Since v0.8.0 it has
/// `rope_norm_cuda`'s and `rope_neox_cuda`'s destination type and
/// `ROPE + VIEW + SET_ROWS` parameters, for qwen's MRoPE.
fn ropeMultiCuda(comptime forward: bool, comptime T: type, comptime D: type, a: Args, sections: MropeSections, is_imrope: bool, row_indices: CPtr, set_rows_stride: c_int, inplace: bool, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const g = grid(&a);
    const theta_scale = thetaScale(&a);
    const launch_params: ggml.cuda.kernels.LaunchParams = .{ .grid = g[0], .block = g[1], .shmem = 0, .stream = stream };
    const args = .{
        a.x,         a.dst,          a.ne00,   a.ne01,    a.ne02,      a.s01,           a.s02,        a.s03,         a.s1,
        a.s2,        a.s3,           a.n_dims, a.n_offs,  a.pos,       a.freq_scale,    a.ext_factor, a.attn_factor, a.corr_dims,
        theta_scale, a.freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace,
    };

    if (a.freq_factors == null) {
        ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key4("rope_multi", forward, false, T, D)), launch_params, args);
    } else {
        ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key4("rope_multi", forward, true, T, D)), launch_params, args);
    }
}

/// Ports `rope_vision_cuda` (rope.cu:505 @b1b770d90). Takes no `n_offs`.
fn ropeVisionCuda(comptime forward: bool, comptime T: type, a: Args, sections: MropeSections, stream: ggml.cuda.common.cu.cudaStream_t) void {
    const g = grid(&a);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);
    const theta_scale = thetaScale(&a);
    const p: ggml.cuda.kernels.LaunchParams = .{ .grid = g[0], .block = g[1], .shmem = 0, .stream = stream };
    const args = .{
        a.x,            a.dst,    a.ne00,   a.ne01, a.ne02,       a.s01,        a.s02,         a.s03,       a.s1,
        a.s2,           a.s3,     a.n_dims, a.pos,  a.freq_scale, a.ext_factor, a.attn_factor, a.corr_dims, theta_scale,
        a.freq_factors, sections,
    };

    if (a.freq_factors == null) {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("rope", key3("rope_vision", forward, false, T)), p, args);
    } else {
        ggml.cuda.kernels.launch(ggml.cuda.kernels.get("rope", key3("rope_vision", forward, true, T)), p, args);
    }
}

/// `size_t` or `int64_t` passed to an `int` parameter.
fn int(v: anytype) c_int {
    const w: u64 = @bitCast(v);
    return @bitCast(@as(u32, @truncate(w)));
}

/// Ports `ggml_cuda_op_rope_impl` (rope.cu:548 @b1b770d90).
fn opRopeImpl(comptime forward: bool, ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor, set_rows: ?*const ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);
    const src2: ?*const ggml.impl.ggml_c.ggml_tensor = dst.src[2];

    var dst_d: ?*anyopaque = dst.data;
    var row_indices: ?*const anyopaque = null;
    var dst_type = dst.type;
    var set_rows_stride: c_int = 0;

    if (set_rows) |sr| {
        ggml.impl.assert(forward, "forward");
        dst_d = sr.data;
        row_indices = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, sr.src[1]).data;
        dst_type = sr.type;
        set_rows_stride = @truncate(@as(i64, @intCast(sr.nb[1] / ggml.impl.ggml_c.ggml_type_size(sr.type))));
    }
    const stream = ctx.stream();

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32 or dst.type == ggml.impl.ggml_c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    ggml.impl.assert(src0.type == dst.type or (src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 and dst.type == ggml.impl.ggml_c.GGML_TYPE_F16), "src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16)");

    const ne00 = src0.ne[0]; // head dims
    const ne01 = src0.ne[1]; // num heads
    const ne02 = src0.ne[2]; // num heads
    const nr = ggml.impl.ggml_c.ggml_nrows(src0);

    const ts0 = ggml.impl.ggml_c.ggml_type_size(src0.type);
    const s01 = src0.nb[1] / ts0;
    const s02 = src0.nb[2] / ts0;
    const s03 = src0.nb[3] / ts0;

    const tsd = ggml.impl.ggml_c.ggml_type_size(dst.type);
    const s1 = dst.nb[1] / tsd;
    const s2 = dst.nb[2] / tsd;
    const s3 = dst.nb[3] / tsd;

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const n_dims: c_int = ggml.impl.getOpParamsI32(dst, 1);
    const mode: c_int = ggml.impl.getOpParamsI32(dst, 2);
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const n_ctx_orig: c_int = ggml.impl.getOpParamsI32(dst, 4);
    const n_offs: c_int = ggml.impl.getOpParamsI32(dst, 15);
    var sections: MropeSections = undefined;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const inplace = dst_d == src0.data;

    // RoPE alteration for extended context
    const freq_base = ggml.impl.getOpParamsF32(dst, 5);
    const freq_scale = ggml.impl.getOpParamsF32(dst, 6);
    const ext_factor = ggml.impl.getOpParamsF32(dst, 7);
    const attn_factor = ggml.impl.getOpParamsF32(dst, 8);
    const beta_fast = ggml.impl.getOpParamsF32(dst, 9);
    const beta_slow = ggml.impl.getOpParamsF32(dst, 10);
    for (0..4) |i| sections.v[i] = ggml.impl.getOpParamsI32(dst, 11 + i);

    const is_neox = (mode & ggml.impl.ggml_c.GGML_ROPE_TYPE_NEOX) != 0;
    const is_mrope = (mode & ggml.impl.ggml_c.GGML_ROPE_TYPE_MROPE) != 0;
    const is_imrope = mode == ggml.impl.ggml_c.GGML_ROPE_TYPE_IMROPE;
    const is_vision = mode == ggml.impl.ggml_c.GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        ggml.impl.assert(sections.v[0] > 0 or sections.v[1] > 0 or sections.v[2] > 0, "sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0");
    }

    if (is_vision) {
        ggml.impl.assert(n_dims == @divTrunc(ne00, 2), "n_dims == ne00/2");
        ggml.impl.assert(n_offs == 0, "n_offs == 0"); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const pos: CPtr = src1.data;

    var freq_factors: CPtr = null;
    if (src2) |t| {
        freq_factors = t.data;
    }

    var corr_dims: CorrDims = undefined;
    ggml.impl.ggml_c.ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml.impl.ggml_c.ggml_rope_get_truncate(dst), &corr_dims.v);

    const a: Args = .{
        .x = src0.data,
        .dst = dst_d,
        .ne00 = int(ne00),
        .ne01 = int(ne01),
        .ne02 = int(ne02),
        .s01 = int(s01),
        .s02 = int(s02),
        .s03 = int(s03),
        .s1 = int(s1),
        .s2 = int(s2),
        .s3 = int(s3),
        .n_dims = n_dims,
        .n_offs = n_offs,
        .nr = int(nr),
        .pos = pos,
        .freq_scale = freq_scale,
        .freq_base = freq_base,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .corr_dims = corr_dims,
        .freq_factors = freq_factors,
    };

    const F32 = ggml.impl.ggml_c.GGML_TYPE_F32;
    const F16 = ggml.impl.ggml_c.GGML_TYPE_F16;

    // compute
    if (is_neox) {
        if (src0.type == F32 and dst_type == F32) {
            ropeNeoxCuda(forward, f32, f32, a, row_indices, set_rows_stride, inplace, stream);
        } else if (src0.type == F32 and dst_type == F16) {
            ropeNeoxCuda(forward, f32, ggml.cuda.kernels.half, a, row_indices, set_rows_stride, inplace, stream);
        } else if (src0.type == F16 and dst_type == F16) {
            ropeNeoxCuda(forward, ggml.cuda.kernels.half, ggml.cuda.kernels.half, a, row_indices, set_rows_stride, inplace, stream);
        } else {
            ggml.impl.abort("fatal error");
        }
    } else if (is_mrope and !is_vision) {
        if (src0.type == F32 and dst_type == F32) {
            ropeMultiCuda(forward, f32, f32, a, sections, is_imrope, row_indices, set_rows_stride, inplace, stream);
        } else if (src0.type == F32 and dst_type == F16) {
            ropeMultiCuda(forward, f32, ggml.cuda.kernels.half, a, sections, is_imrope, row_indices, set_rows_stride, inplace, stream);
        } else if (src0.type == F16 and dst_type == F16) {
            ropeMultiCuda(forward, ggml.cuda.kernels.half, ggml.cuda.kernels.half, a, sections, is_imrope, row_indices, set_rows_stride, inplace, stream);
        } else {
            ggml.impl.abort("fatal error");
        }
    } else if (is_vision) {
        if (src0.type == F32) {
            ropeVisionCuda(forward, f32, a, sections, stream);
        } else if (src0.type == F16) {
            ropeVisionCuda(forward, ggml.cuda.kernels.half, a, sections, stream);
        } else {
            ggml.impl.abort("fatal error");
        }
    } else {
        if (src0.type == F32 and dst_type == F32) {
            ropeNormCuda(forward, f32, f32, a, row_indices, set_rows_stride, inplace, stream);
        } else if (src0.type == F32 and dst_type == F16) {
            ropeNormCuda(forward, f32, ggml.cuda.kernels.half, a, row_indices, set_rows_stride, inplace, stream);
        } else if (src0.type == F16 and dst_type == F16) {
            ropeNormCuda(forward, ggml.cuda.kernels.half, ggml.cuda.kernels.half, a, row_indices, set_rows_stride, inplace, stream);
        } else {
            ggml.impl.abort("fatal error");
        }
    }
}

/// Ports `ggml_cuda_op_rope` (rope.cu:714 @b1b770d90).
pub fn opRope(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    opRopeImpl(true, ctx, dst, null);
}

/// Ports `ggml_cuda_op_rope_back` (rope.cu:718 @b1b770d90).
pub fn opRopeBack(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    opRopeImpl(false, ctx, dst, null);
}

/// Ports `ggml_cuda_op_rope_fused` (rope.cu:722 @b1b770d90): `ROPE` written
/// straight into the rows a following `SET_ROWS` selects.
pub fn opRopeFused(ctx: *ggml.cuda.common.Context, rope: *ggml.impl.ggml_c.ggml_tensor, set_rows: *ggml.impl.ggml_c.ggml_tensor) void {
    opRopeImpl(true, ctx, rope, set_rows);
}

/// Ports `rms_norm_mul_rope_cuda` (rope.cu:819 @b1b770d90). Since v0.7.0 the
/// grid's y and z are clamped to `UINT16_MAX` and the kernel loops over the
/// channels and samples past them, so both counts are arguments.
fn rmsNormMulRopeCuda(
    comptime D: type,
    x: CPtr,
    dst: Ptr,
    ncols: c_int,
    nrows: c_int,
    nchannels: c_int,
    nsamples: c_int,
    s01: i64,
    s02: i64,
    s03: i64,
    s1: i64,
    s2: i64,
    s3: i64,
    eps: f32,
    mul: CPtr,
    mul_s01: i64,
    mul_s02: i64,
    mul_s03: i64,
    mul_ncols: u32,
    mul_nrows: u32,
    mul_nchannels: u32,
    mul_nsamples: u32,
    n_dims: c_int,
    pos: CPtr,
    freq_scale: f32,
    freq_base: f32,
    ext_factor: f32,
    attn_factor: f32,
    corr_dims: CorrDims,
    freq_factors: CPtr,
    row_indices: CPtr,
    set_rows_stride: c_int,
    is_neox: bool,
    stream: ggml.cuda.common.cu.cudaStream_t,
) void {
    ggml.impl.assert(@rem(ncols, 2) == 0, "ncols % 2 == 0");

    const blocks_num: ggml.cuda.kernels.Dim3 = .{ .x = @bitCast(nrows), .y = @bitCast(@min(nchannels, std.math.maxInt(u16))), .z = @bitCast(@min(nsamples, std.math.maxInt(u16))) };

    const theta_scale = stdlib.libm.powf(freq_base, -2.0 / @as(f32, @floatFromInt(n_dims)));

    const mul_ncols_packed = ggml.cuda.common.initFastdivValues(mul_ncols);
    const mul_nrows_packed = ggml.cuda.common.initFastdivValues(mul_nrows);
    const mul_nchannels_packed = ggml.cuda.common.initFastdivValues(mul_nchannels);
    const mul_nsamples_packed = ggml.cuda.common.initFastdivValues(mul_nsamples);

    const args = .{
        x,                dst,              ncols,                nchannels,           nsamples, s01, s02,        s03,        s1,          s2,        s3,          eps,          mul,         mul_s01,         mul_s02, mul_s03,
        mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, n_dims,   pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, is_neox,
    };

    const key = struct {
        fn f(comptime bs: comptime_int, comptime has_ff: bool) []const u8 {
            return ggml.cuda.kernels.tmpl("rms_norm_mul_rope_f32", &.{ ggml.cuda.kernels.int(bs), ggml.cuda.kernels.boolean(has_ff), ggml.cuda.kernels.typeName(D) });
        }
    }.f;

    if (ncols < 1024) {
        const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = 256 };
        const launch_params: ggml.cuda.kernels.LaunchParams = .{ .grid = blocks_num, .block = block_dims, .shmem = 32 * @sizeOf(f32), .stream = stream };
        if (freq_factors == null) {
            ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key(256, false)), launch_params, args);
        } else {
            ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key(256, true)), launch_params, args);
        }
    } else {
        const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = 1024 };
        const launch_params: ggml.cuda.kernels.LaunchParams = .{ .grid = blocks_num, .block = block_dims, .shmem = 32 * @sizeOf(f32), .stream = stream };
        if (freq_factors == null) {
            ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key(1024, false)), launch_params, args);
        } else {
            ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("rope", key(1024, true)), launch_params, args);
        }
    }
}

/// Ports `ggml_cuda_op_rms_norm_mul_rope_fused` (rope.cu:881 @b1b770d90).
/// `set_rows` is null when the chain ends at the `ROPE`.
pub fn opRmsNormMulRopeFused(ctx: *ggml.cuda.common.Context, rms_norm: *ggml.impl.ggml_c.ggml_tensor, mul: *ggml.impl.ggml_c.ggml_tensor, rope: *ggml.impl.ggml_c.ggml_tensor, set_rows: ?*ggml.impl.ggml_c.ggml_tensor) void {
    const x = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, rms_norm.src[0]);
    const mul_src = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, if (mul.src[0] == rms_norm) mul.src[1] else mul.src[0]);

    const eps = ggml.impl.getOpParamsF32(rms_norm, 0);
    ggml.impl.assert(eps >= 0.0, "eps >= 0.0f");

    ggml.impl.assert(x.type == ggml.impl.ggml_c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    ggml.impl.assert(mul_src.type == ggml.impl.ggml_c.GGML_TYPE_F32, "mul_src->type == GGML_TYPE_F32");
    ggml.impl.assert(rope.type == ggml.impl.ggml_c.GGML_TYPE_F32, "rope->type == GGML_TYPE_F32");

    var dst_d: ?*anyopaque = rope.data;
    var dst_type = rope.type;
    var row_indices: CPtr = null;
    var set_rows_stride: c_int = 0;

    if (set_rows) |sr| {
        dst_d = sr.data;
        dst_type = sr.type;
        row_indices = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, sr.src[1]).data;
        set_rows_stride = @truncate(@as(i64, @intCast(sr.nb[1] / ggml.impl.ggml_c.ggml_type_size(sr.type))));
    }

    const n_dims: c_int = ggml.impl.getOpParamsI32(rope, 1);
    const mode: c_int = ggml.impl.getOpParamsI32(rope, 2);
    const n_ctx_orig: c_int = ggml.impl.getOpParamsI32(rope, 4);

    const freq_base = ggml.impl.getOpParamsF32(rope, 5);
    const freq_scale = ggml.impl.getOpParamsF32(rope, 6);
    const ext_factor = ggml.impl.getOpParamsF32(rope, 7);
    const attn_factor = ggml.impl.getOpParamsF32(rope, 8);
    const beta_fast = ggml.impl.getOpParamsF32(rope, 9);
    const beta_slow = ggml.impl.getOpParamsF32(rope, 10);

    const is_neox = (mode & ggml.impl.ggml_c.GGML_ROPE_TYPE_NEOX) != 0;

    const pos: CPtr = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, rope.src[1]).data;

    const freq_factors: CPtr = if (rope.src[2] != null) ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, rope.src[2]).data else null;

    var corr_dims: CorrDims = undefined;
    ggml.impl.ggml_c.ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml.impl.ggml_c.ggml_rope_get_truncate(rope), &corr_dims.v);

    const ts0 = ggml.impl.ggml_c.ggml_type_size(x.type);
    ggml.impl.assert(x.nb[0] == ts0, "x->nb[0] == ts0");
    const s01: i64 = @intCast(x.nb[1] / ts0);
    const s02: i64 = @intCast(x.nb[2] / ts0);
    const s03: i64 = @intCast(x.nb[3] / ts0);

    const ts_mul = ggml.impl.ggml_c.ggml_type_size(mul_src.type);
    ggml.impl.assert(mul_src.nb[0] == ts_mul, "mul_src->nb[0] == ts_mul");
    const mul_s01: i64 = @intCast(mul_src.nb[1] / ts_mul);
    const mul_s02: i64 = @intCast(mul_src.nb[2] / ts_mul);
    const mul_s03: i64 = @intCast(mul_src.nb[3] / ts_mul);

    const ts_dst = ggml.impl.ggml_c.ggml_type_size(rope.type);
    const s1: i64 = @intCast(rope.nb[1] / ts_dst);
    const s2: i64 = @intCast(rope.nb[2] / ts_dst);
    const s3: i64 = @intCast(rope.nb[3] / ts_dst);

    const stream = ctx.stream();

    const u = struct {
        fn f(v: i64) u32 {
            return @truncate(@as(u64, @bitCast(v)));
        }
    }.f;

    if (dst_type == ggml.impl.ggml_c.GGML_TYPE_F32) {
        rmsNormMulRopeCuda(f32, x.data, dst_d, int(x.ne[0]), int(x.ne[1]), int(x.ne[2]), int(x.ne[3]), s01, s02, s03, s1, s2, s3, eps, mul_src.data, mul_s01, mul_s02, mul_s03, u(mul_src.ne[0]), u(mul_src.ne[1]), u(mul_src.ne[2]), u(mul_src.ne[3]), n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, row_indices, set_rows_stride, is_neox, stream);
    } else if (dst_type == ggml.impl.ggml_c.GGML_TYPE_F16) {
        rmsNormMulRopeCuda(ggml.cuda.kernels.half, x.data, dst_d, int(x.ne[0]), int(x.ne[1]), int(x.ne[2]), int(x.ne[3]), s01, s02, s03, s1, s2, s3, eps, mul_src.data, mul_s01, mul_s02, mul_s03, u(mul_src.ne[0]), u(mul_src.ne[1]), u(mul_src.ne[2]), u(mul_src.ne[3]), n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, row_indices, set_rows_stride, is_neox, stream);
    } else {
        ggml.impl.abort("fatal error");
    }
}
