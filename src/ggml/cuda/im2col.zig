//! `GGML_OP_IM2COL` and `GGML_OP_IM2COL_3D`: unfold convolution windows
//! into rows, so a convolution becomes a matrix multiplication.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/im2col.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/im2col.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels, `im2col_kernel` and `im2col_3d_kernel`, stay CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `CUDA_IM2COL_BLOCK_SIZE` (im2col.cuh:3 @c1d0e7a00).
const block_size = 256;
/// Ports `MAX_GRIDDIM_Y` (im2col.cu:3 @c1d0e7a00).
const max_griddim_y = 65535;
/// Ports `MAX_GRIDDIM_Z` (im2col.cu:4 @c1d0e7a00).
const max_griddim_z = 65535;

/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `im2col_cuda` (im2col.cu:48 @c1d0e7a00), reached through
/// `im2col_cuda_f16` and `im2col_cuda_f32` (im2col.cu:62, 70 @c1d0e7a00).
fn im2colCuda(
    comptime T: type,
    x: ?*const anyopaque,
    dst: ?*anyopaque,
    IW: i64,
    IH: i64,
    OW: i64,
    OH: i64,
    KW: i64,
    KH: i64,
    IC: i64,
    N: i64,
    IC_IH_IW: i64,
    IH_IW: i64,
    s0: c_int,
    s1: c_int,
    p0: c_int,
    p1: c_int,
    d0: c_int,
    d1: c_int,
    stream: cu.cudaStream_t,
) void {
    const IC_KH_KW = IC * KH * KW;
    const num_blocks = @divTrunc(IC_KH_KW + block_size - 1, block_size);
    const N_OH = N * OH;
    const KH_KW = KW * KH;
    const block_nums: kernels.Dim3 = .{ .x = dim(num_blocks), .y = dim(@min(OW, max_griddim_y)), .z = dim(@min(N_OH, max_griddim_z)) };
    kernels.launch(kernels.get("im2col", kernels.tmpl("im2col_kernel", &.{kernels.typeName(T)})), .{
        .grid = block_nums,
        .block = .{ .x = dim(@min(IC_KH_KW, block_size)) },
        .shmem = 0,
        .stream = stream,
    }, .{ x, dst, IC, IW, IH, OH, OW, KW, KH, IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW, s0, s1, p0, p1, d0, d1 });
}

/// Ports `ggml_cuda_op_im2col` (im2col.cu:78 @c1d0e7a00).
pub fn opIm2col(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const src1_d: ?*const anyopaque = src1.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F16 or dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32");

    const s0 = impl.getOpParamsI32(dst, 0);
    const s1 = impl.getOpParamsI32(dst, 1);
    const p0 = impl.getOpParamsI32(dst, 2);
    const p1 = impl.getOpParamsI32(dst, 3);
    const d0 = impl.getOpParamsI32(dst, 4);
    const d1 = impl.getOpParamsI32(dst, 5);

    const is_2D = impl.getOpParamsI32(dst, 6) == 1;

    const IC = src1.ne[if (is_2D) 2 else 1];
    const IH = if (is_2D) src1.ne[1] else 1;
    const IW = src1.ne[0];

    const KH = if (is_2D) src0.ne[1] else 1;
    const KW = src0.ne[0];

    const OH = if (is_2D) dst.ne[2] else 1;
    const OW = dst.ne[1];

    const IC_IH_IW: i64 = @bitCast(src1.nb[if (is_2D) 2 else 1] / 4); // nb is byte offset, src is type float32
    const N = src1.ne[if (is_2D) 3 else 2];
    const IH_IW: i64 = @bitCast(src1.nb[if (is_2D) 3 else 2] / 4); // nb is byte offset, src is type float32

    if (dst.type == c.GGML_TYPE_F16) {
        im2colCuda(half, src1_d, dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2colCuda(f32, src1_d, dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

/// Ports `im2col_3d_cuda` (im2col.cu:165 @c1d0e7a00), reached through
/// `im2col_3d_cuda_f16` and `im2col_3d_cuda_f32` (im2col.cu:193, 204
/// @c1d0e7a00).
fn im2col3dCuda(
    comptime T: type,
    src: ?*const anyopaque,
    dst: ?*anyopaque,
    N: i64,
    IC: i64,
    ID: i64,
    IH: i64,
    IW: i64,
    OC: i64,
    KD: i64,
    KH: i64,
    KW: i64,
    OD: i64,
    OH: i64,
    OW: i64,
    stride_q: i64,
    stride_z: i64,
    stride_y: i64,
    stride_x: i64,
    s0: c_int,
    s1: c_int,
    s2: c_int,
    p0: c_int,
    p1: c_int,
    p2: c_int,
    d0: c_int,
    d1: c_int,
    d2: c_int,
    stream: cu.cudaStream_t,
) void {
    const OH_OW = OH * OW;
    const KD_KH_KW = KD * KH * KW;
    const ID_IH_IW = ID * IH * IW;
    const KH_KW = KH * KW;
    const IH_IW = IH * IW;
    const IC_KD_KH_KW = IC * KD * KH * KW;
    const OW_KD_KH_KW = OW * KD * KH * KW;
    const N_OD_OH = N * OD * OH;
    const OD_OH = OD * OH;
    const IC_ID_IH_IW = IC * ID * IH * IW;
    const OD_OH_OW_IC_KD_KH_KW = OD * OH * OW * IC * KD * KH * KW;
    const OH_OW_IC_KD_KH_KW = OH * OW * IC * KD * KH * KW;
    const OW_IC_KD_KH_KW = OW * IC * KD * KH * KW;
    const num_blocks = @divTrunc(IC_KD_KH_KW + block_size - 1, block_size);
    const block_nums: kernels.Dim3 = .{ .x = dim(num_blocks), .y = dim(@min(OW, max_griddim_y)), .z = dim(@min(N_OD_OH, max_griddim_z)) };
    kernels.launch(kernels.get("im2col", kernels.tmpl("im2col_3d_kernel", &.{kernels.typeName(T)})), .{
        .grid = block_nums,
        .block = .{ .x = dim(@min(IC_KD_KH_KW, block_size)) },
        .shmem = 0,
        .stream = stream,
    }, .{
        src,            dst,     N,     IC,          ID,          IH,          IW,                   OC,
        KD,             KH,      KW,    OD,          OH,          OW,          OH_OW,                KD_KH_KW,
        ID_IH_IW,       KH_KW,   IH_IW, IC_ID_IH_IW, IC_KD_KH_KW, OW_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW,
        OW_IC_KD_KH_KW, N_OD_OH, OD_OH, stride_q,    stride_z,    stride_y,    stride_x,             s0,
        s1,             s2,      p0,    p1,          p2,          d0,          d1,                   d2,
    });
}

/// Ports `ggml_cuda_op_im2col_3d` (im2col.cu:215 @c1d0e7a00).
pub fn opIm2col3d(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const src1_d: ?*const anyopaque = src1.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F16 or dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32");

    const s0 = impl.getOpParamsI32(dst, 0);
    const s1 = impl.getOpParamsI32(dst, 1);
    const s2 = impl.getOpParamsI32(dst, 2);
    const p0 = impl.getOpParamsI32(dst, 3);
    const p1 = impl.getOpParamsI32(dst, 4);
    const p2 = impl.getOpParamsI32(dst, 5);
    const d0 = impl.getOpParamsI32(dst, 6);
    const d1 = impl.getOpParamsI32(dst, 7);
    const d2 = impl.getOpParamsI32(dst, 8);
    const IC = impl.getOpParamsI32(dst, 9);

    const N = @divTrunc(src1.ne[3], IC);
    const ID = src1.ne[2];
    const IH = src1.ne[1];
    const IW = src1.ne[0];

    const OC = @divTrunc(src0.ne[3], IC);
    const KD = src0.ne[2];
    const KH = src0.ne[1];
    const KW = src0.ne[0];

    const OD = @divTrunc(dst.ne[3], N);
    const OH = dst.ne[2];
    const OW = dst.ne[1];

    const es = c.ggml_element_size(src1);
    const stride_x: i64 = @bitCast(src1.nb[0] / es);
    const stride_y: i64 = @bitCast(src1.nb[1] / es);
    const stride_z: i64 = @bitCast(src1.nb[2] / es);
    const stride_q: i64 = @bitCast(src1.nb[3] / es);

    if (dst.type == c.GGML_TYPE_F16) {
        im2col3dCuda(half, src1_d, dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } else {
        im2col3dCuda(f32, src1_d, dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
