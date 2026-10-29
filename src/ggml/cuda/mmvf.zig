//! MMVF: matrix-vector products for `F32`, `F16` and `BF16` weights.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/mmvf.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/mmvf.cuh` at v0.4.1 (`b29c606e2`), the host
//! half. The kernel, `mul_mat_vec_f`, stays CUDA.
//!
//! `ggml_cuda_op_mul_mat_vec_f`, the row-split entry point, is not ported:
//! nothing in this version of upstream calls it.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;
const bf16 = kernels.bf16;

/// Ports `MMVF_MAX_BATCH_SIZE` (mmvf.cuh:3 @b29c606e2).
pub const max_batch_size = 8;

fn i32Of(x: i64) c_int {
    return @truncate(x);
}

fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// The arguments of one `mul_mat_vec_f` launch, at the kernel's parameter
/// types: `switch_fusion` takes the column and row strides as `int64_t` and
/// the kernel narrows them to `int`.
const Args = struct {
    x: ?*const anyopaque,
    y: ?*const f32,
    ids: ?*const i32,
    fusion: common.MmFusionArgsDevice,
    dst: ?*f32,
    ncols2: c_int,
    nchannels_y: cu.uint3,
    stride_row: c_int,
    stride_col_y2: c_int,
    stride_col_dst: c_int,
    channel_ratio: cu.uint3,
    stride_channel_x: c_int,
    stride_channel_y: c_int,
    stride_channel_dst: c_int,
    sample_ratio: cu.uint3,
    stride_sample_x: c_int,
    stride_sample_y: c_int,
    stride_sample_dst: c_int,
    ids_stride: c_int,
};

fn mmvfKernel(comptime T: type, comptime A: type, comptime ncols_dst: comptime_int, comptime block_size: comptime_int, comptime has_fusion: bool, comptime is_multi_token_id: bool) kernels.Kernel {
    return kernels.get("mmvf", kernels.tmpl("mul_mat_vec_f", &.{ kernels.typeName(T), kernels.typeName(A), kernels.int(ncols_dst), kernels.int(block_size), kernels.boolean(has_fusion), kernels.boolean(is_multi_token_id) }));
}

/// Ports `mul_mat_vec_f_switch_fusion` (mmvf.cu:387 @b29c606e2).
fn switchFusion(comptime T: type, comptime A: type, comptime ncols_dst: comptime_int, comptime block_size: comptime_int, comptime is_multi_token_id: bool, a: *const Args, block_dims: kernels.Dim3, block_nums: kernels.Dim3, nbytes_shared: c_int, stream: cu.cudaStream_t) void {
    const p: kernels.LaunchParams = .{ .grid = block_nums, .block = block_dims, .shmem = @intCast(nbytes_shared), .stream = stream };

    const f = a.fusion;
    const has_fusion = f.gate != null or f.x_bias != null or f.gate_bias != null;
    const args = .{
        a.x,               a.y,                a.ids,               a.fusion,             a.dst,
        a.ncols2,          a.nchannels_y,      a.stride_row,        a.stride_col_y2,      a.stride_col_dst,
        a.channel_ratio,   a.stride_channel_x, a.stride_channel_y,  a.stride_channel_dst, a.sample_ratio,
        a.stride_sample_x, a.stride_sample_y,  a.stride_sample_dst, a.ids_stride,
    };
    if (ncols_dst == 1) {
        if (has_fusion) {
            kernels.launchPdl(comptime mmvfKernel(T, A, ncols_dst, block_size, true, is_multi_token_id), p, args);
            return;
        }
    }

    impl.assert(!has_fusion, "!has_fusion && \"fusion only supported for ncols_dst=1\"");

    kernels.launchPdl(comptime mmvfKernel(T, A, ncols_dst, block_size, false, is_multi_token_id), p, args);
}

/// Ports `launch_mul_mat_vec_f_cuda` (mmvf.cu:418 @b29c606e2).
fn launchMulMatVecF(
    comptime T: type,
    comptime A: type,
    comptime ncols_dst: comptime_int,
    comptime is_multi_token_id: bool,
    x: ?*const anyopaque,
    y: ?*const f32,
    ids: ?*const i32,
    fusion: common.MmFusionArgsDevice,
    dst: ?*f32,
    ncols: i64,
    nrows: i64,
    stride_row: i64,
    stride_col_y: i64,
    stride_col_dst: i64,
    nchannels_x: i64,
    nchannels_y: i64,
    nchannels_dst: i64,
    stride_channel_x: i64,
    stride_channel_y: i64,
    stride_channel_dst: i64,
    nsamples_x: i64,
    nsamples_dst: i64,
    stride_sample_x: i64,
    stride_sample_y: i64,
    stride_sample_dst: i64,
    nsamples_or_ntokens: i64,
    ids_stride: i64,
    stream: cu.cudaStream_t,
) void {
    impl.assert(@rem(ncols, 2) == 0, "ncols        % 2 == 0");
    impl.assert(@rem(stride_row, 2) == 0, "stride_row   % 2 == 0");
    impl.assert(@rem(stride_col_y, 2) == 0, "stride_col_y % 2 == 0");
    impl.assert(ids != null or @rem(nchannels_dst, nchannels_x) == 0, "ids || nchannels_dst % nchannels_x == 0");
    impl.assert(@rem(nsamples_dst, nsamples_x) == 0, "       nsamples_dst  % nsamples_x  == 0");
    const zero3: cu.uint3 = .{ .x = 0, .y = 0, .z = 0 };
    const nchannels_y_fd = if (ids != null) common.initFastdivValues(@bitCast(nchannels_y)) else zero3;
    const channel_ratio_fd = if (ids != null) zero3 else common.initFastdivValues(@bitCast(@divTrunc(nchannels_dst, nchannels_x)));
    const sample_ratio_fd = common.initFastdivValues(@bitCast(@divTrunc(nsamples_dst, nsamples_x)));

    const device = common.getDevice();
    const warp_size: i64 = common.info().devices[@intCast(device)].warp_size;

    var block_size_best: i64 = warp_size;
    var niter_best: i64 = @divTrunc(ncols + 2 * warp_size - 1, 2 * warp_size);
    var max_block_size: i64 = 256;
    const dcc = common.info().devices[@intCast(device)].cc;
    if (dcc > common.cc.offset_amd and dcc < common.cc.offset_amd + 0x1010) max_block_size = 128;
    var block_size: i64 = 2 * warp_size;
    while (block_size <= max_block_size) : (block_size += warp_size) {
        const niter = @divTrunc(ncols + 2 * block_size - 1, 2 * block_size);
        if (niter < niter_best) {
            niter_best = niter;
            block_size_best = block_size;
        }
    }

    const has_fusion = fusion.gate != null or fusion.x_bias != null or fusion.gate_bias != null;

    const ws: c_int = @intCast(warp_size);
    const nbytes_shared: c_int = ws * @sizeOf(f32) + (if (has_fusion) ws * @sizeOf(f32) else 0);
    const block_nums: kernels.Dim3 = .{ .x = u32Of(nrows), .y = u32Of(nchannels_dst), .z = u32Of(nsamples_or_ntokens) };
    const block_dims: kernels.Dim3 = .{ .x = u32Of(block_size_best), .y = 1, .z = 1 };

    // `ncols/2` and `stride_col_y/2` are `int64_t` until the kernel narrows them.
    const a: Args = .{
        .x = x,
        .y = y,
        .ids = ids,
        .fusion = fusion,
        .dst = dst,
        .ncols2 = i32Of(@divTrunc(ncols, 2)),
        .nchannels_y = nchannels_y_fd,
        .stride_row = i32Of(stride_row),
        .stride_col_y2 = i32Of(@divTrunc(stride_col_y, 2)),
        .stride_col_dst = i32Of(stride_col_dst),
        .channel_ratio = channel_ratio_fd,
        .stride_channel_x = i32Of(stride_channel_x),
        .stride_channel_y = i32Of(stride_channel_y),
        .stride_channel_dst = i32Of(stride_channel_dst),
        .sample_ratio = sample_ratio_fd,
        .stride_sample_x = i32Of(stride_sample_x),
        .stride_sample_y = i32Of(stride_sample_y),
        .stride_sample_dst = i32Of(stride_sample_dst),
        .ids_stride = i32Of(ids_stride),
    };

    switch (block_size_best) {
        inline 32, 64, 96, 128, 160, 192, 224, 256 => |bs| switchFusion(T, A, ncols_dst, bs, is_multi_token_id, &a, block_dims, block_nums, nbytes_shared, stream),
        else => impl.abort("fatal error"),
    }
}

/// Ports `mul_mat_vec_f_cuda_switch_ncols_dst` (mmvf.cu:513 @b29c606e2).
fn switchNcolsDst(comptime T: type, comptime A: type, x: ?*const anyopaque, y: ?*const f32, ids: ?*const i32, fusion: common.MmFusionArgsDevice, dst: ?*f32, ncols: i64, nrows: i64, ncols_dst: i64, stride_row: i64, stride_col_y: i64, stride_col_dst: i64, nchannels_x: i64, nchannels_y: i64, nchannels_dst: i64, stride_channel_x: i64, stride_channel_y: i64, stride_channel_dst: i64, nsamples_x: i64, nsamples_dst: i64, stride_sample_x: i64, stride_sample_y: i64, stride_sample_dst: i64, ids_stride: i64, stream: cu.cudaStream_t) void {
    const has_ids = ids != null;

    if (has_ids and ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only
        launchMulMatVecF(T, A, 1, true, x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        launchMulMatVecF(T, A, 1, false, x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        inline 1, 2, 3, 4, 5, 6, 7, 8 => |n| launchMulMatVecF(T, A, n, false, x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, nsamples_dst, ids_stride, stream),
        else => impl.abort("fatal error"),
    }
}

/// Ports `mul_mat_vec_f_cuda` (mmvf.cu:610 @b29c606e2). Note its
/// `stride_col_dst` parameter is `int`, unlike its neighbours.
fn mulMatVecFCuda(comptime T: type, x: ?*const anyopaque, y: ?*const f32, ids: ?*const i32, fusion: common.MmFusionArgsDevice, dst: ?*f32, ncols: i64, nrows: i64, ncols_dst: i64, stride_row: i64, stride_col_y: i64, stride_col_dst: c_int, nchannels_x: i64, nchannels_y: i64, nchannels_dst: i64, stride_channel_x: i64, stride_channel_y: i64, stride_channel_dst: i64, nsamples_x: i64, nsamples_dst: i64, stride_sample_x: i64, stride_sample_y: i64, stride_sample_dst: i64, ids_stride: i64, prec: c.enum_ggml_prec, stream: cu.cudaStream_t) void {
    if (T == half) {
        if (prec == c.GGML_PREC_DEFAULT) {
            switchNcolsDst(T, half, x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    switchNcolsDst(T, f32, x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

/// Ports `ggml_cuda_mul_mat_vec_f` (mmvf.cu:634 @b29c606e2).
pub fn mulMatVecF(ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, ids: ?*const Tensor, dst: *Tensor, fusion: ?*const common.MmFusionArgsHost) void {
    impl.assert(src1.type == c.GGML_TYPE_F32, "        src1->type == GGML_TYPE_F32");
    impl.assert(ids == null or ids.?.type == c.GGML_TYPE_I32, "!ids ||  ids->type == GGML_TYPE_I32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "         dst->type == GGML_TYPE_F32");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];

    const ts_src0 = c.ggml_type_size(src0.type);
    const ts_src1 = c.ggml_type_size(src1.type);
    const ts_dst = c.ggml_type_size(dst.type);

    impl.assert(ids == null or ne12 <= max_batch_size, "!ids || ne12 <= MMVF_MAX_BATCH_SIZE");
    impl.assert(ne13 == ne3, "ne13 == ne3");

    impl.assert(src0.nb[0] == ts_src0, "        nb00       == ts_src0");
    impl.assert(src1.nb[0] == ts_src1, "        nb10       == ts_src1");
    impl.assert(ids == null or ids.?.nb[0] == c.ggml_type_size(ids.?.type), "!ids || ids->nb[0] == ggml_type_size(ids->type)");
    impl.assert(dst.nb[0] == ts_dst, "        nb0        == ts_dst");

    const x = common.info().devices[@intCast(common.getDevice())].cc;
    const prec: c.enum_ggml_prec = if (common.fastFp16Available(x)) @intCast(dst.op_params[0]) else c.GGML_PREC_F32;

    const src1_d: ?*const f32 = @ptrCast(@alignCast(src1.data));
    const ids_d: ?*const i32 = if (ids) |i| @ptrCast(@alignCast(i.data)) else null;
    const dst_d: ?*f32 = @ptrCast(@alignCast(dst.data));

    var fusion_local: common.MmFusionArgsDevice = .{};

    if (fusion) |f| {
        impl.assert(ids == null or dst.ne[2] == 1, " !ids || dst->ne[2] == 1");
        impl.assert(ids != null or dst.ne[1] == 1, "  ids || dst->ne[1] == 1");
        if (f.x_bias) |b| {
            impl.assert(b.type == c.GGML_TYPE_F32, "fusion->x_bias->type == GGML_TYPE_F32");
            impl.assert(b.ne[0] == dst.ne[0], "fusion->x_bias->ne[0] == dst->ne[0]");
            impl.assert(ids == null or b.ne[1] == src0.ne[2], "!ids || fusion->x_bias->ne[1] == src0->ne[2]");
            fusion_local.x_bias = b.data;
        }
        if (f.gate) |g| {
            impl.assert(g.type == src0.type and c.ggml_are_same_stride(g, src0), "fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0)");
            fusion_local.gate = g.data;
        }
        if (f.gate_bias) |b| {
            impl.assert(b.type == c.GGML_TYPE_F32, "fusion->gate_bias->type == GGML_TYPE_F32");
            impl.assert(b.ne[0] == dst.ne[0], "fusion->gate_bias->ne[0] == dst->ne[0]");
            impl.assert(ids == null or b.ne[1] == src0.ne[2], "!ids || fusion->gate_bias->ne[1] == src0->ne[2]");
            fusion_local.gate_bias = b.data;
        }
        fusion_local.glu_op = f.glu_op;
        fusion_local.glu_limit = f.glu_limit;
    }

    const s01: i64 = @intCast(src0.nb[1] / ts_src0);
    const s11: i64 = @intCast(src1.nb[1] / ts_src1);
    const s1: i64 = @intCast(dst.nb[1] / ts_dst);
    const s02: i64 = @intCast(src0.nb[2] / ts_src0);
    const s12: i64 = @intCast(src1.nb[2] / ts_src1);
    const s2: i64 = @intCast(dst.nb[2] / ts_dst);
    const s03: i64 = @intCast(src0.nb[3] / ts_src0);
    const s13: i64 = @intCast(src1.nb[3] / ts_src1);
    const s3: i64 = @intCast(dst.nb[3] / ts_dst);

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const has_ids = ids != null;
    const ncols_dst = if (has_ids) ne2 else ne1;
    const nchannels_y = if (has_ids) ne11 else ne12;
    const nchannels_dst = if (has_ids) ne1 else ne2;
    const stride_col_dst = if (has_ids) s2 else s1;
    const stride_col_y = if (has_ids) s12 else s11;
    const stride_channel_dst = if (has_ids) s1 else s2;
    const stride_channel_y = if (has_ids) s11 else s12;

    const ids_stride: i64 = if (ids) |i| @intCast(i.nb[1] / c.ggml_type_size(i.type)) else 0;

    const stream = ctx.stream();
    switch (src0.type) {
        c.GGML_TYPE_F32 => mulMatVecFCuda(f32, src0.data, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, i32Of(stride_col_dst), ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst, ne03, ne3, s03, s13, s3, ids_stride, prec, stream),
        c.GGML_TYPE_F16 => mulMatVecFCuda(half, src0.data, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, i32Of(stride_col_dst), ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst, ne03, ne3, s03, s13, s3, ids_stride, prec, stream),
        c.GGML_TYPE_BF16 => mulMatVecFCuda(bf16, src0.data, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, i32Of(stride_col_dst), ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst, ne03, ne3, s03, s13, s3, ids_stride, prec, stream),
        else => {
            impl.logError("unsupported type: %s\n", .{c.ggml_type_name(src0.type)});
            impl.abort("unsupported type");
        },
    }
}

/// Ports `ggml_cuda_should_use_mmvf` (mmvf.cu:792 @b29c606e2), NVIDIA arms.
pub fn shouldUseMmvf(ty: c.enum_ggml_type, x: c_int, src0_ne: *const [4]i64, src0_nb: *const [4]usize, ne11: i64) bool {
    if (@rem(src0_ne[0], 2) != 0) return false;

    const ts = c.ggml_type_size(ty);
    if (src0_nb[0] != ts) return false;

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would
    // result in a crash:
    for (1..c.GGML_MAX_DIMS) |i| {
        if (src0_nb[i] % (2 * ts) != 0) return false;
    }

    switch (ty) {
        c.GGML_TYPE_F32 => {
            if (common.cc.isNvidia(x)) {
                if (common.ampereMmaAvailable(x)) return ne11 <= 3;
                if (x >= common.cc.turing) return ne11 <= 4;
                return ne11 <= 3;
            }
            return ne11 <= 8;
        },
        c.GGML_TYPE_F16 => {
            if (common.cc.isNvidia(x)) {
                const src0_small = (src0_ne[1] <= 512 or src0_ne[2] * src0_ne[3] == 1);
                if (common.ampereMmaAvailable(x)) return src0_small and ne11 == 1;
                if (x >= common.cc.ada_lovelace) return src0_small and ne11 <= 4;
                if (common.fp16MmaHardwareAvailable(x)) return src0_small and ne11 <= 3;
                return ne11 <= 8;
            }
            return ne11 <= 8;
        },
        c.GGML_TYPE_BF16 => {
            if (common.cc.isNvidia(x)) {
                const src0_small = (src0_ne[1] <= 512 or src0_ne[2] * src0_ne[3] == 1);
                if (common.ampereMmaAvailable(x)) return src0_small and ne11 == 1;
                if (x >= common.cc.ada_lovelace) return src0_small and ne11 <= 4;
                if (common.bf16MmaHardwareAvailable(x)) return src0_small and ne11 <= 3;
                return ne11 <= 8;
            }
            return ne11 <= 8;
        },
        else => return false,
    }
}
