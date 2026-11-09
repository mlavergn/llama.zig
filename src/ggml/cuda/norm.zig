//! Normalisation: `NORM`, `RMS_NORM`, `RMS_NORM_BACK`, `L2_NORM`,
//! `GROUP_NORM`, and the fused `RMS_NORM + MUL (+ ADD)` and (since v0.6.0)
//! `RMS_NORM + SCALE` entry points `ggml_cuda_try_fuse` calls.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/norm.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/norm.cuh` at v0.8.0 (`8642541b2`). The
//! kernels -- `norm_f32`, `group_norm_f32`, `rms_norm_f32`,
//! `rms_norm_back_f32`, `l2_norm_f32` -- stay CUDA.
//!
//! # Two launch styles
//!
//! `norm_f32`, `group_norm_f32` and `rms_norm_back_f32` launch with
//! `<<<>>>`; `rms_norm_f32` and `l2_norm_f32` launch through
//! `ggml_cuda_kernel_launch` (PDL). `rms_norm_f32` has seventeen trailing
//! parameters with C++ defaults (v0.6.0 added `scale_out`), and `cudaLaunchKernelEx` does not apply
//! defaults, so the C spells the zeros out at every launch; since v0.8.0
//! one dispatcher, `rmsNormF32Dispatch`, makes every launch.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

const warp_size = common.warp_size;

/// `const float *`, `float *` as kernel arguments.
const CPtr = ?*const anyopaque;
const Ptr = ?*anyopaque;

/// A runtime copy of `v`. A comptime-known value in a tuple literal
/// becomes a comptime field, which `kernels.launch` cannot take the
/// address of; the defaulted arguments below go through this.
fn rt(comptime T: type, v: T) T {
    return v;
}

/// `make_uint3(0, 0, 0)`, at runtime.
fn uint3Zero() cu.uint3 {
    return rt(cu.uint3, .{ .x = 0, .y = 0, .z = 0 });
}

/// An `int` converted to a `dim3` member (`unsigned int`).
fn uint(v: c_int) u32 {
    return @bitCast(v);
}

/// `dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX))`:
/// since v0.7.0 the kernels loop over the channels and samples past the
/// grid's limit, so the grid is clamped and the counts travel as arguments.
fn rowGrid(nrows: c_int, nchannels: c_int, nsamples: c_int) kernels.Dim3 {
    return .{ .x = uint(nrows), .y = uint(@min(nchannels, std.math.maxInt(u16))), .z = uint(@min(nsamples, std.math.maxInt(u16))) };
}

/// An `int64_t` passed to an `int` parameter.
fn int(v: i64) c_int {
    return @truncate(v);
}

/// Ports `norm_f32_cuda` (norm.cu:362 @cd959bae9).
fn normF32Cuda(x: CPtr, dst: Ptr, ncols: c_int, nrows: c_int, nchannels: c_int, nsamples: c_int, stride_row: i64, stride_channel: i64, stride_sample: i64, eps: f32, stream: cu.cudaStream_t) void {
    const blocks_num = rowGrid(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const block_dims: kernels.Dim3 = .{ .x = warp_size };
        kernels.launch(kernels.get("norm", "norm_f32<(int)32>"), .{
            .grid = blocks_num,
            .block = block_dims,
            .shmem = 0,
            .stream = stream,
        }, .{ x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps });
    } else {
        const block_dims: kernels.Dim3 = .{ .x = 1024 };
        kernels.launch(kernels.get("norm", "norm_f32<(int)1024>"), .{
            .grid = blocks_num,
            .block = block_dims,
            .shmem = if (block_dims.x > warp_size) 32 * 2 * @sizeOf(f32) else 0, // sizeof(float2)
            .stream = stream,
        }, .{ x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps });
    }
}

/// Ports `group_norm_f32_cuda` (norm.cu:375 @cd959bae9).
fn groupNormF32Cuda(x: CPtr, dst: Ptr, num_groups: c_int, eps: f32, group_size: c_int, ne_elements: c_int, stream: cu.cudaStream_t) void {
    if (group_size < 1024) {
        const block_dims: kernels.Dim3 = .{ .x = warp_size };
        kernels.launch(kernels.get("norm", "group_norm_f32<(int)32>"), .{
            .grid = .{ .x = uint(num_groups) },
            .block = block_dims,
            .shmem = 0,
            .stream = stream,
        }, .{ x, dst, group_size, ne_elements, eps });
    } else {
        const block_dims: kernels.Dim3 = .{ .x = 1024 };
        kernels.launch(kernels.get("norm", "group_norm_f32<(int)1024>"), .{
            .grid = .{ .x = uint(num_groups) },
            .block = block_dims,
            .shmem = if (block_dims.x > warp_size) 2 * 32 * @sizeOf(f32) else 0,
            .stream = stream,
        }, .{ x, dst, group_size, ne_elements, eps });
    }
}

/// `rms_norm_f32<block_size, vpt, do_vec, do_multiply, do_add, do_scale>`;
/// `do_scale` is v0.6.0's, `vpt` and `do_vec` v0.8.0's, and the demangled
/// key spells every template argument, defaulted or not.
fn rmsNormKernel(comptime block_size: comptime_int, comptime vpt: comptime_int, comptime vec: bool, comptime do_multiply: bool, comptime do_add: bool, comptime do_scale: bool) kernels.Kernel {
    return kernels.get("norm", kernels.tmpl("rms_norm_f32", &.{ kernels.int(block_size), kernels.int(vpt), kernels.boolean(vec), kernels.boolean(do_multiply), kernels.boolean(do_add), kernels.boolean(do_scale) }));
}

/// A `mul` or `add` operand of `rms_norm_f32`: its tensor, its strides in
/// elements and its four packed extents, as the C passes them positionally.
const RmsOperand = struct {
    t: ?*const Tensor = null,
    stride_row: i64 = 0,
    stride_channel: i64 = 0,
    stride_sample: i64 = 0,
    ncols_packed: cu.uint3 = .{ .x = 0, .y = 0, .z = 0 },
    nrows_packed: cu.uint3 = .{ .x = 0, .y = 0, .z = 0 },
    nchannels_packed: cu.uint3 = .{ .x = 0, .y = 0, .z = 0 },
    nsamples_packed: cu.uint3 = .{ .x = 0, .y = 0, .z = 0 },

    /// `mul_t ? mul_t->data : nullptr`.
    fn data(self: RmsOperand) CPtr {
        return if (self.t) |t| t.data else null;
    }
};

/// Ports `RMS_NORM_LAUNCH` (norm.cu:386 @cd959bae9).
fn rmsNormLaunch(comptime block_size: comptime_int, comptime vpt: comptime_int, comptime vec: bool, comptime do_multiply: bool, comptime do_add: bool, comptime do_scale: bool, blocks_num: kernels.Dim3, stream: cu.cudaStream_t, args: anytype) void {
    const block_dims: kernels.Dim3 = .{ .x = block_size };
    kernels.launchPdl(rmsNormKernel(block_size, vpt, vec, do_multiply, do_add, do_scale), .{
        .grid = blocks_num,
        .block = block_dims,
        .shmem = if (block_dims.x > warp_size) 32 * @sizeOf(f32) else 0,
        .stream = stream,
    }, args);
}

/// Ports `rms_norm_f32_dispatch` (norm.cu:399 @cd959bae9), new at v0.8.0:
/// one row per block, in `float4` loads held in registers where the row and
/// its operands are 16-byte aligned, and otherwise the scalar kernel the two
/// launchers below had each spelled out.
fn rmsNormF32Dispatch(
    comptime do_multiply: bool,
    comptime do_add: bool,
    comptime do_scale: bool,
    blocks_num: kernels.Dim3,
    x_t: *const Tensor,
    dst: Ptr,
    ncols: c_int,
    nchannels: c_int,
    nsamples: c_int,
    stride_row: i64,
    stride_channel: i64,
    stride_sample: i64,
    eps: f32,
    mul: RmsOperand,
    add: RmsOperand,
    stream: cu.cudaStream_t,
    scale_out: f32,
) void {
    comptime std.debug.assert(!do_add or do_multiply); // fusing add is not supported without multiplying
    comptime std.debug.assert(!do_scale or !do_multiply); // fusing scale is not supported with multiplying

    // underlying cudaLaunchKernelEx does not support default params
    const args = .{
        @as(CPtr, x_t.data),
        dst,
        ncols,
        nchannels,
        nsamples,
        stride_row,
        stride_channel,
        stride_sample,
        eps,
        mul.data(),
        mul.stride_row,
        mul.stride_channel,
        mul.stride_sample,
        mul.ncols_packed,
        mul.nrows_packed,
        mul.nchannels_packed,
        mul.nsamples_packed,
        add.data(),
        add.stride_row,
        add.stride_channel,
        add.stride_sample,
        add.ncols_packed,
        add.nrows_packed,
        add.nchannels_packed,
        add.nsamples_packed,
        scale_out,
    };

    // dst is written at row * ncols, i.e. always at a multiple of ncols, so only its base needs checking here
    var vec = @rem(ncols, 4) == 0 and ncols > 0 and (@intFromPtr(dst) & 15) == 0;
    vec = vec and common.isAligned(x_t, 16);
    if (do_multiply) {
        vec = vec and mul.ncols_packed.z == @as(u32, @bitCast(ncols)) and common.isAligned(mul.t.?, 16);
    }
    if (do_add) {
        vec = vec and add.ncols_packed.z == @as(u32, @bitCast(ncols)) and common.isAligned(add.t.?, 16);
    }

    const n4 = @divTrunc(ncols, 4);
    if (vec) {
        if (n4 <= 32) {
            rmsNormLaunch(32, 1, true, do_multiply, do_add, do_scale, blocks_num, stream, args);
        } else if (n4 <= 128) {
            rmsNormLaunch(128, 1, true, do_multiply, do_add, do_scale, blocks_num, stream, args);
        } else if (n4 <= 256) {
            rmsNormLaunch(256, 1, true, do_multiply, do_add, do_scale, blocks_num, stream, args);
        } else if (n4 <= 512) {
            rmsNormLaunch(512, 1, true, do_multiply, do_add, do_scale, blocks_num, stream, args);
        } else if (n4 <= 1024) {
            rmsNormLaunch(1024, 1, true, do_multiply, do_add, do_scale, blocks_num, stream, args);
        } else if (n4 <= 4096) {
            rmsNormLaunch(1024, 4, true, do_multiply, do_add, do_scale, blocks_num, stream, args);
        } else {
            rmsNormLaunch(1024, 1, false, do_multiply, do_add, do_scale, blocks_num, stream, args);
        }
    } else if (ncols < 1024) {
        rmsNormLaunch(256, 1, false, do_multiply, do_add, do_scale, blocks_num, stream, args);
    } else {
        rmsNormLaunch(1024, 1, false, do_multiply, do_add, do_scale, blocks_num, stream, args);
    }
}

/// Ports `rms_norm_f32_cuda` (norm.cu:453 @cd959bae9), its `do_scale`
/// template argument comptime; `scale_out` multiplies the normalised row
/// when it is set (the `RMS_NORM + SCALE` fusion) and is ignored otherwise.
fn rmsNormF32Cuda(comptime do_scale: bool, x_t: *const Tensor, dst: Ptr, ncols: c_int, nrows: c_int, nchannels: c_int, nsamples: c_int, stride_row: i64, stride_channel: i64, stride_sample: i64, eps: f32, stream: cu.cudaStream_t, scale_out: f32) void {
    const blocks_num = rowGrid(nrows, nchannels, nsamples);
    rmsNormF32Dispatch(false, false, do_scale, blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, .{}, .{}, stream, scale_out);
}

/// Ports `rms_norm_mul_f32_cuda` (norm.cu:465 @cd959bae9).
fn rmsNormMulF32Cuda(
    x_t: *const Tensor,
    mul_t: ?*const Tensor,
    add_t: ?*const Tensor,
    dst: Ptr,
    ncols: c_int,
    nrows: c_int,
    nchannels: c_int,
    nsamples: c_int,
    stride_row: i64,
    stride_channel: i64,
    stride_sample: i64,
    mul_stride_row: i64,
    mul_stride_channel: i64,
    mul_stride_sample: i64,
    mul_ncols: u32,
    mul_nrows: u32,
    mul_nchannels: u32,
    mul_nsamples: u32,
    add_stride_row: i64,
    add_stride_channel: i64,
    add_stride_sample: i64,
    add_ncols: u32,
    add_nrows: u32,
    add_nchannels: u32,
    add_nsamples: u32,
    eps: f32,
    stream: cu.cudaStream_t,
) void {
    const blocks_num = rowGrid(nrows, nchannels, nsamples);
    if (mul_t == null) {
        rmsNormF32Cuda(false, x_t, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream, 1.0);
        return;
    }
    const mul: RmsOperand = .{
        .t = mul_t,
        .stride_row = mul_stride_row,
        .stride_channel = mul_stride_channel,
        .stride_sample = mul_stride_sample,
        .ncols_packed = common.initFastdivValues(mul_ncols),
        .nrows_packed = common.initFastdivValues(mul_nrows),
        .nchannels_packed = common.initFastdivValues(mul_nchannels),
        .nsamples_packed = common.initFastdivValues(mul_nsamples),
    };

    if (add_t == null) {
        rmsNormF32Dispatch(true, false, false, blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, .{}, stream, 1.0);
    } else {
        const add: RmsOperand = .{
            .t = add_t,
            .stride_row = add_stride_row,
            .stride_channel = add_stride_channel,
            .stride_sample = add_stride_sample,
            .ncols_packed = common.initFastdivValues(add_ncols),
            .nrows_packed = common.initFastdivValues(add_nrows),
            .nchannels_packed = common.initFastdivValues(add_nchannels),
            .nsamples_packed = common.initFastdivValues(add_nsamples),
        };
        rmsNormF32Dispatch(true, true, false, blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, add, stream, 1.0);
    }
}

/// Ports `rms_norm_back_f32_cuda` (norm.cu:523 @cd959bae9).
fn rmsNormBackF32Cuda(grad: CPtr, xf: CPtr, dst: Ptr, ncols: c_int, nrows: c_int, eps: f32, stream: cu.cudaStream_t) void {
    if (ncols < 1024) {
        const block_dims: kernels.Dim3 = .{ .x = warp_size };
        kernels.launch(kernels.get("norm", "rms_norm_back_f32<(int)32>"), .{
            .grid = .{ .x = uint(nrows) },
            .block = block_dims,
            .shmem = 0,
            .stream = stream,
        }, .{ grad, xf, dst, ncols, eps });
    } else {
        const block_dims: kernels.Dim3 = .{ .x = 1024 };
        kernels.launch(kernels.get("norm", "rms_norm_back_f32<(int)1024>"), .{
            .grid = .{ .x = uint(nrows) },
            .block = block_dims,
            .shmem = 0,
            .stream = stream,
        }, .{ grad, xf, dst, ncols, eps });
    }
}

/// Ports `l2_norm_f32_cuda` (norm.cu:533 @cd959bae9).
fn l2NormF32Cuda(x: CPtr, dst: Ptr, ncols: c_int, nrows: c_int, nchannels: c_int, nsamples: c_int, stride_row: i64, stride_channel: i64, stride_sample: i64, eps: f32, stream: cu.cudaStream_t) void {
    const blocks_num = rowGrid(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const block_dims: kernels.Dim3 = .{ .x = warp_size };
        kernels.launchPdl(kernels.get("norm", "l2_norm_f32<(int)32>"), .{
            .grid = blocks_num,
            .block = block_dims,
            .shmem = 0,
            .stream = stream,
        }, .{ x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps });
    } else {
        const block_dims: kernels.Dim3 = .{ .x = 1024 };
        kernels.launchPdl(kernels.get("norm", "l2_norm_f32<(int)1024>"), .{
            .grid = blocks_num,
            .block = block_dims,
            .shmem = if (block_dims.x > warp_size) 32 * @sizeOf(f32) else 0,
            .stream = stream,
        }, .{ x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps });
    }
}

fn srcOf(t: *const Tensor, i: usize) *Tensor {
    return impl.one(Tensor, t.src[i]);
}

/// The shared head of `ggml_cuda_op_norm`, `_rms_norm` and `_l2_norm`:
/// the type asserts, `eps`, and the strides in elements.
const RowArgs = struct {
    src0: *const Tensor,
    eps: f32,
    s01: i64,
    s02: i64,
    s03: i64,
};

fn rowArgs(dst: *const Tensor) RowArgs {
    const src0 = srcOf(dst, 0);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const eps = impl.getOpParamsF32(dst, 0);
    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const ts0 = c.ggml_type_size(src0.type);
    impl.assert(src0.nb[0] == ts0, "nb00 == ts0");
    return .{
        .src0 = src0,
        .eps = eps,
        .s01 = @intCast(src0.nb[1] / ts0),
        .s02 = @intCast(src0.nb[2] / ts0),
        .s03 = @intCast(src0.nb[3] / ts0),
    };
}

/// Ports `ggml_cuda_op_norm` (norm.cu:548 @cd959bae9).
pub fn opNorm(ctx: *common.Context, dst: *Tensor) void {
    const stream = ctx.stream();
    const a = rowArgs(dst);
    const ne = a.src0.ne;
    normF32Cuda(a.src0.data, dst.data, int(ne[0]), int(ne[1]), int(ne[2]), int(ne[3]), a.s01, a.s02, a.s03, a.eps, stream);
}

/// Ports `ggml_cuda_op_group_norm` (norm.cu:572 @cd959bae9).
pub fn opGroupNorm(ctx: *common.Context, dst: *Tensor) void {
    const src0 = srcOf(dst, 0);
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");

    const num_groups: c_int = impl.getOpParamsI32(dst, 0);

    const eps = impl.getOpParamsF32(dst, 1);
    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const group_size: c_int = int(src0.ne[0] * src0.ne[1] * @divTrunc(src0.ne[2] + num_groups - 1, num_groups));
    groupNormF32Cuda(src0.data, dst.data, int(num_groups * src0.ne[3]), eps, group_size, int(c.ggml_nelements(src0)), stream);
}

/// Ports `ggml_cuda_op_rms_norm` (norm.cu:591 @cd959bae9).
pub fn opRmsNorm(ctx: *common.Context, dst: *Tensor) void {
    const stream = ctx.stream();
    const a = rowArgs(dst);
    const ne = a.src0.ne;
    rmsNormF32Cuda(false, a.src0, dst.data, int(ne[0]), int(ne[1]), int(ne[2]), int(ne[3]), a.s01, a.s02, a.s03, a.eps, stream, 1.0);
}

/// Ports `ggml_cuda_op_rms_norm_scale_fused` (norm.cu:614 @cd959bae9), new
/// at v0.6.0: `RMS_NORM` followed by a bias-free `SCALE` of its result,
/// written straight to the `SCALE` node.
pub fn opRmsNormScaleFused(ctx: *common.Context, dst: *Tensor, scale_tensor: *Tensor) void {
    const src0 = srcOf(dst, 0);
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(scale_tensor.type == c.GGML_TYPE_F32, "scale_tensor->type == GGML_TYPE_F32");

    const eps = impl.getOpParamsF32(dst, 0);
    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const scale = impl.getOpParamsF32(scale_tensor, 0);

    const ts0 = c.ggml_type_size(src0.type);
    impl.assert(src0.nb[0] == ts0, "nb00 == ts0");
    const s01: i64 = @intCast(src0.nb[1] / ts0);
    const s02: i64 = @intCast(src0.nb[2] / ts0);
    const s03: i64 = @intCast(src0.nb[3] / ts0);

    const ne = src0.ne;
    rmsNormF32Cuda(true, src0, scale_tensor.data, int(ne[0]), int(ne[1]), int(ne[2]), int(ne[3]), s01, s02, s03, eps, stream, scale);
}

/// The other operand of a binary node one of whose sources is `t`; the
/// C's `if (op->src[0] == t) ... else if (op->src[1] == t) ... else
/// GGML_ASSERT(false)`.
fn otherSrc(op: *const Tensor, t: *const Tensor) *const Tensor {
    if (op.src[0] == t) return srcOf(op, 1);
    if (op.src[1] == t) return srcOf(op, 0);
    impl.assert(false, "false");
    unreachable;
}

/// Strides in elements and extents of a broadcast operand, as
/// `ggml_cuda_op_rms_norm_fused` computes them for `mul_src`.
const Bcast = struct {
    s01: i64,
    s02: i64,
    s03: i64,
    ncols: c_int,
    nrows: c_int,
    nchannels: c_int,
    nsamples: c_int,
};

fn bcastOf(t: *const Tensor, comptime which: []const u8) Bcast {
    const ts = c.ggml_type_size(t.type);
    impl.assert(t.nb[0] == ts, which ++ "_src->nb[0] == ts_" ++ which);
    return .{
        .s01 = @intCast(t.nb[1] / ts),
        .s02 = @intCast(t.nb[2] / ts),
        .s03 = @intCast(t.nb[3] / ts),
        .ncols = int(t.ne[0]),
        .nrows = int(t.ne[1]),
        .nchannels = int(t.ne[2]),
        .nsamples = int(t.ne[3]),
    };
}

/// Ports `ggml_cuda_op_rms_norm_fused` (norm.cu:640 @cd959bae9): `RMS_NORM`
/// followed by a `MUL` with `dst` as one operand.
pub fn opRmsNormFused(ctx: *common.Context, dst: *Tensor, mul_tensor: *Tensor) void {
    const rms_norm_src = srcOf(dst, 0);
    const eps = impl.getOpParamsF32(dst, 0);

    const mul_src = otherSrc(mul_tensor, dst);
    const stream = ctx.stream();

    impl.assert(rms_norm_src.type == c.GGML_TYPE_F32, "rms_norm_src->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(mul_tensor.type == c.GGML_TYPE_F32, "mul_tensor->type == GGML_TYPE_F32");
    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const ne00 = rms_norm_src.ne[0];
    const ne01 = rms_norm_src.ne[1];
    const ne02 = rms_norm_src.ne[2];
    const ne03 = rms_norm_src.ne[3];

    const ts0 = c.ggml_type_size(rms_norm_src.type);
    impl.assert(rms_norm_src.nb[0] == ts0, "rms_norm_src->nb[0] == ts0");
    const s01: i64 = @intCast(rms_norm_src.nb[1] / ts0);
    const s02: i64 = @intCast(rms_norm_src.nb[2] / ts0);
    const s03: i64 = @intCast(rms_norm_src.nb[3] / ts0);

    const m = bcastOf(mul_src, "mul");

    rmsNormMulF32Cuda(
        rms_norm_src,
        mul_src,
        null,
        mul_tensor.data,
        int(ne00),
        int(ne01),
        int(ne02),
        int(ne03),
        s01,
        s02,
        s03,
        m.s01,
        m.s02,
        m.s03,
        @bitCast(m.ncols),
        @bitCast(m.nrows),
        @bitCast(m.nchannels),
        @bitCast(m.nsamples),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        eps,
        stream,
    );
}

/// Ports `ggml_cuda_op_rms_norm_fused_add` (norm.cu:696 @cd959bae9):
/// `RMS_NORM`, then `MUL`, then `ADD`.
pub fn opRmsNormFusedAdd(ctx: *common.Context, dst: *Tensor, mul_tensor: *Tensor, add_tensor: *Tensor) void {
    const rms_norm_src = srcOf(dst, 0);
    const eps = impl.getOpParamsF32(dst, 0);

    const mul_src = otherSrc(mul_tensor, dst);
    const add_src = otherSrc(add_tensor, mul_tensor);

    const stream = ctx.stream();

    impl.assert(rms_norm_src.type == c.GGML_TYPE_F32, "rms_norm_src->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(mul_tensor.type == c.GGML_TYPE_F32, "mul_tensor->type == GGML_TYPE_F32");
    impl.assert(add_tensor.type == c.GGML_TYPE_F32, "add_tensor->type == GGML_TYPE_F32");
    impl.assert(eps >= 0.0, "eps >= 0.0f");

    const ne00 = rms_norm_src.ne[0];
    const ne01 = rms_norm_src.ne[1];
    const ne02 = rms_norm_src.ne[2];
    const ne03 = rms_norm_src.ne[3];

    const ts0 = c.ggml_type_size(rms_norm_src.type);
    impl.assert(rms_norm_src.nb[0] == ts0, "rms_norm_src->nb[0] == ts0");
    const s01: i64 = @intCast(rms_norm_src.nb[1] / ts0);
    const s02: i64 = @intCast(rms_norm_src.nb[2] / ts0);
    const s03: i64 = @intCast(rms_norm_src.nb[3] / ts0);

    const m = bcastOf(mul_src, "mul");
    const a = bcastOf(add_src, "add");

    rmsNormMulF32Cuda(
        rms_norm_src,
        mul_src,
        add_src,
        add_tensor.data,
        int(ne00),
        int(ne01),
        int(ne02),
        int(ne03),
        s01,
        s02,
        s03,
        m.s01,
        m.s02,
        m.s03,
        @bitCast(m.ncols),
        @bitCast(m.nrows),
        @bitCast(m.nchannels),
        @bitCast(m.nsamples),
        a.s01,
        a.s02,
        a.s03,
        @bitCast(a.ncols),
        @bitCast(a.nrows),
        @bitCast(a.nchannels),
        @bitCast(a.nsamples),
        eps,
        stream,
    );
}

/// Ports `ggml_cuda_op_rms_norm_back` (norm.cu:777 @cd959bae9).
pub fn opRmsNormBack(ctx: *common.Context, dst: *Tensor) void {
    const grad = srcOf(dst, 0); // gradients
    const src0f = srcOf(dst, 1); // src0 from forward pass

    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(grad), "ggml_is_contiguous(grad)");

    impl.assert(grad.type == c.GGML_TYPE_F32, " grad->type == GGML_TYPE_F32");
    impl.assert(src0f.type == c.GGML_TYPE_F32, "src0f->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "  dst->type == GGML_TYPE_F32");

    const ne00 = src0f.ne[0];
    const nrows = c.ggml_nrows(src0f);

    const eps = impl.getOpParamsF32(dst, 0);
    impl.assert(eps >= 0.0, "eps >= 0.0f");

    rmsNormBackF32Cuda(grad.data, src0f.data, dst.data, int(ne00), int(nrows), eps, stream);
}

/// Ports `ggml_cuda_op_l2_norm` (norm.cu:803 @cd959bae9).
pub fn opL2Norm(ctx: *common.Context, dst: *Tensor) void {
    const stream = ctx.stream();
    const a = rowArgs(dst);
    const ne = a.src0.ne;
    l2NormF32Cuda(a.src0.data, dst.data, int(ne[0]), int(ne[1]), int(ne[2]), int(ne[3]), a.s01, a.s02, a.s03, a.eps, stream);
}
