//! `GGML_OP_CONV_3D`: 3-D convolution, new at v0.6.0.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/conv3d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/conv3d.cuh` at v0.7.0 (`620d2225c`). The
//! kernels -- `conv3d_kernel`, `conv3d_pad_f16`, `conv3d_implicit_gemm_f16`
//! and `conv3d_reduce_split_k` -- stay CUDA.
//!
//! # Paths
//!
//! `conv2d.zig`'s shape one dimension up. An F16 kernel whose sizes fit in
//! `int` takes the implicit GEMM: the input is converted to F16 (padded
//! first if the convolution pads), then a pointwise convolution is one
//! cuBLAS GEMM per batch on hardware with fast F16, and anything else a
//! tiled kernel -- tensor-core MMA from Turing on, SIMT before -- whose long
//! reductions are split `split_k` ways and summed by
//! `conv3d_reduce_split_k`. Everything else is the direct kernel. Unlike
//! `conv2d.cu`, there is no small-convolution or MUSA gate on the GEMM.
//!
//! The channel and batch counts are not in the shapes: `ggml_conv_3d_direct`
//! folds them into `ne[3]` and writes them to op_params slots 9, 10 and 11.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const convert = @import("convert.zig");
const fattn_common = @import("fattn_common.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `conv3d_params` (conv3d.cu:5 @620d2225c), passed to the kernels by
/// value: 22 `int64_t`.
const Conv3dParams = extern struct {
    IW: i64,
    IH: i64,
    ID: i64,
    OW: i64,
    OH: i64,
    OD: i64,
    KW: i64,
    KH: i64,
    KD: i64,
    ST_X: i64,
    ST_Y: i64,
    ST_Z: i64,
    PD_X: i64,
    PD_Y: i64,
    PD_Z: i64,
    DL_X: i64,
    DL_Y: i64,
    DL_Z: i64,
    IC: i64,
    OC: i64,
    B: i64,
    TOTAL: i64,

    comptime {
        if (@sizeOf(Conv3dParams) != 22 * 8) @compileError("conv3d_params is 22 int64_t");
    }
};

/// Ports `conv3d_launch_implicit_gemm` (conv3d.cu:253 @620d2225c): the
/// 3x3x3, 1x1x3 and 1x1x1 kernels have their sizes as template arguments,
/// the rest read them from `params`. Vector loads of the weights need the
/// base pointer and every weight row 16-byte aligned; the kernel is told
/// which.
fn conv3dLaunchImplicitGemm(
    comptime use_mma: bool,
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    output: ?*anyopaque,
    params: Conv3dParams,
    split_k: c_int,
    grid: kernels.Dim3,
    block: kernels.Dim3,
    stream: cu.cudaStream_t,
) void {
    // Vector loads require both the base pointer and each weight row to be 16-byte aligned.
    const sizeof_int4 = 16;
    const aligned_weights = @intFromPtr(weight) % sizeof_int4 == 0 and
        @rem(params.IC * params.KW * params.KH * params.KD, sizeof_int4 / @sizeOf(half)) == 0;
    const lp: kernels.LaunchParams = .{ .grid = grid, .block = block, .shmem = 0, .stream = stream };
    const args = .{ input, weight, output, params, split_k, aligned_weights };
    const mma = comptime kernels.boolean(use_mma);
    if (params.KW == 3 and params.KH == 3 and params.KD == 3) {
        kernels.launch(kernels.get("conv3d", kernels.tmpl("conv3d_implicit_gemm_f16", &.{ kernels.int(3), kernels.int(3), kernels.int(3), mma })), lp, args);
    } else if (params.KW == 1 and params.KH == 1 and params.KD == 3) {
        kernels.launch(kernels.get("conv3d", kernels.tmpl("conv3d_implicit_gemm_f16", &.{ kernels.int(1), kernels.int(1), kernels.int(3), mma })), lp, args);
    } else if (params.KW == 1 and params.KH == 1 and params.KD == 1) {
        kernels.launch(kernels.get("conv3d", kernels.tmpl("conv3d_implicit_gemm_f16", &.{ kernels.int(1), kernels.int(1), kernels.int(1), mma })), lp, args);
    } else {
        kernels.launch(kernels.get("conv3d", kernels.tmpl("conv3d_implicit_gemm_f16", &.{ kernels.int(0), kernels.int(0), kernels.int(0), mma })), lp, args);
    }
}

/// Ports `ggml_cuda_op_conv3d` (conv3d.cu:279 @620d2225c).
///
/// Parameters:
/// - `ctx`: the backend context whose stream, pool and cuBLAS handle are used.
/// - `dst`: the `CONV_3D` node; `src[0]` the F16 or F32 kernel, `src[1]` the
///   F32 input.
pub fn opConv3d(ctx: *common.Context, dst: *Tensor) void {
    const kernel = impl.one(Tensor, dst.src[0]);
    const input = impl.one(Tensor, dst.src[1]);
    impl.assert(input.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32, "input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32");
    impl.assert(kernel.type == c.GGML_TYPE_F16 or kernel.type == c.GGML_TYPE_F32, "kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(input) and c.ggml_is_contiguous(kernel) and c.ggml_is_contiguous(dst), "ggml_is_contiguous(input) && ggml_is_contiguous(kernel) && ggml_is_contiguous(dst)");

    var p: [12]i32 = undefined;
    for (&p, 0..) |*v, i| v.* = impl.getOpParamsI32(dst, i);
    const IW: i64 = input.ne[0];
    const IH: i64 = input.ne[1];
    const ID: i64 = input.ne[2];
    const OW: i64 = dst.ne[0];
    const OH: i64 = dst.ne[1];
    const OD: i64 = dst.ne[2];
    const KW: i64 = kernel.ne[0];
    const KH: i64 = kernel.ne[1];
    const KD: i64 = kernel.ne[2];
    const IC: i64 = p[9];
    const B: i64 = p[10];
    const OC: i64 = p[11];
    impl.assert(IC > 0 and B > 0 and OC > 0 and input.ne[3] == IC * B and kernel.ne[3] == IC * OC, "IC > 0 && B > 0 && OC > 0 && input->ne[3] == IC * B && kernel->ne[3] == IC * OC");
    impl.assert(dst.ne[3] == OC * B and p[0] > 0 and p[1] > 0 and p[2] > 0 and p[6] > 0 and p[7] > 0 and p[8] > 0, "dst->ne[3] == OC * B && p[0] > 0 && p[1] > 0 && p[2] > 0 && p[6] > 0 && p[7] > 0 && p[8] > 0");
    const total: i64 = c.ggml_nelements(dst);
    const params: Conv3dParams = .{
        .IW = IW,
        .IH = IH,
        .ID = ID,
        .OW = OW,
        .OH = OH,
        .OD = OD,
        .KW = KW,
        .KH = KH,
        .KD = KD,
        .ST_X = p[0],
        .ST_Y = p[1],
        .ST_Z = p[2],
        .PD_X = p[3],
        .PD_Y = p[4],
        .PD_Z = p[5],
        .DL_X = p[6],
        .DL_Y = p[7],
        .DL_Z = p[8],
        .IC = IC,
        .OC = OC,
        .B = B,
        .TOTAL = total,
    };
    const x: ?*const anyopaque = input.data;
    const w: ?*const anyopaque = kernel.data;
    const y: ?*anyopaque = dst.data;
    const stream = ctx.stream();
    const device = &common.info().devices[@intCast(ctx.device)];
    const use_mma = common.turingMmaAvailable(device.cc) or fattn_common.amdWmmaAvailable(device.cc) or fattn_common.amdMfmaAvailable(device.cc);
    const pointwise = KW == 1 and KH == 1 and KD == 1 and p[0] == 1 and p[1] == 1 and p[2] == 1 and p[3] == 0 and p[4] == 0 and p[5] == 0;
    const use_blas = pointwise and common.fastFp16HardwareAvailable(device.cc);
    const limit: i64 = std.math.maxInt(c_int) - 256;
    const pw: i64 = IW + 2 * @as(i64, p[3]);
    const ph: i64 = IH + 2 * @as(i64, p[4]);
    const pd: i64 = ID + 2 * @as(i64, p[5]);
    const padded_fits = pw > 0 and pw <= limit and ph > 0 and ph <= limit and pd > 0 and pd <= limit and
        pw * ph <= @divTrunc(limit, pd) and IC * B <= @divTrunc(limit, pw * ph * pd);
    if (kernel.type == c.GGML_TYPE_F16 and KW > 0 and KH > 0 and KD > 0 and c.ggml_nelements(input) <= limit and
        c.ggml_nelements(kernel) <= limit and total <= limit and padded_fits and p[3] >= 0 and p[4] >= 0 and p[5] >= 0 and
        (OW - 1) * p[0] + (KW - 1) * p[6] < pw and (OH - 1) * p[1] + (KH - 1) * p[7] < ph and
        (OD - 1) * p[2] + (KD - 1) * p[8] < pd and @divTrunc(OC + 63, 64) <= 65535 and B <= 65535)
    {
        const padded_total: c_int = @truncate(pw * ph * pd * IC * B);
        var x_half = common.PoolAlloc(half).initAlloc(ctx.pool(), @intCast(padded_total));
        defer x_half.deinit();
        // Match im2col's F16 input precision without materializing all patches in global memory.
        if (p[3] == 0 and p[4] == 0 and p[5] == 0) {
            convert.getToFp16(input.type).?(x, x_half.get(), padded_total, stream);
        } else {
            kernels.launch(kernels.get("conv3d", "conv3d_pad_f16"), .{
                .grid = .{ .x = @intCast(@divTrunc(padded_total + 255, 256)) },
                .block = .{ .x = 256 },
                .shmem = 0,
                .stream = stream,
            }, .{
                x,                         @as(?*anyopaque, x_half.get()), @as(c_int, @truncate(IW)),
                @as(c_int, @truncate(IH)), @as(c_int, @truncate(ID)),      @as(c_int, @truncate(pw)),
                @as(c_int, @truncate(ph)), @as(c_int, @truncate(pd)),      p[3],
                p[4],                      p[5],                           padded_total,
            });
        }
        const padded_params: Conv3dParams = .{
            .IW = pw,
            .IH = ph,
            .ID = pd,
            .OW = OW,
            .OH = OH,
            .OD = OD,
            .KW = KW,
            .KH = KH,
            .KD = KD,
            .ST_X = p[0],
            .ST_Y = p[1],
            .ST_Z = p[2],
            .PD_X = 0,
            .PD_Y = 0,
            .PD_Z = 0,
            .DL_X = p[6],
            .DL_Y = p[7],
            .DL_Z = p[8],
            .IC = IC,
            .OC = OC,
            .B = B,
            .TOTAL = total,
        };
        const positions: c_int = @truncate(OW * OH * OD);
        if (use_blas) {
            const alpha: f32 = 1.0;
            const beta: f32 = 0.0;
            const cublas_h = ctx.cublasHandle();
            const xh: [*]const half = x_half.get().?;
            const yd: [*]f32 = @ptrCast(@alignCast(y.?));
            var n: c_int = 0;
            while (n < B) : (n += 1) {
                common.cublasCheck(cu.cublasGemmEx(
                    cublas_h,
                    cu.CUBLAS_OP_N,
                    cu.CUBLAS_OP_N,
                    positions,
                    @as(c_int, @truncate(OC)),
                    @as(c_int, @truncate(IC)),
                    &alpha,
                    xh + @as(usize, @intCast(@as(i64, n) * IC * positions)),
                    cu.CUDA_R_16F,
                    positions,
                    w,
                    cu.CUDA_R_16F,
                    @as(c_int, @truncate(IC)),
                    &beta,
                    yd + @as(usize, @intCast(@as(i64, n) * OC * positions)),
                    cu.CUDA_R_32F,
                    positions,
                    cu.CUBLAS_COMPUTE_32F,
                    cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                ), "cublasGemmEx(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, positions, int(OC), int(IC), &alpha, x_half.get() + int64_t(n) * IC * positions, CUDA_R_16F, positions, w, CUDA_R_16F, int(IC), &beta, y + int64_t(n) * OC * positions, CUDA_R_32F, positions, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)", @src());
            }
            return;
        }
        const blocks: i64 = @divTrunc(@as(i64, positions) + 63, 64) * @divTrunc(OC + 63, 64) * B;
        const target: c_int = 8 * device.nsm;
        const split_k: c_int = @intCast(@min(
            @min(@as(i64, 32), @divTrunc(@as(i64, 65535), B)),
            @min(@divTrunc(IC * KW * KH * KD + 63, 64), @max(@as(i64, 1), @divTrunc(target + blocks - 1, blocks))),
        ));
        var partial = common.PoolAlloc(f32).init(ctx.pool());
        defer partial.deinit();
        const result: ?*anyopaque = if (split_k == 1) y else @ptrCast(partial.alloc(@intCast(total * split_k)));
        const block: kernels.Dim3 = .{ .x = @bitCast(device.warp_size), .y = 4 };
        const grid: kernels.Dim3 = .{
            .x = @truncate(@as(u64, @bitCast(@divTrunc(@as(i64, positions) + 63, 64)))),
            .y = @truncate(@as(u64, @bitCast(@divTrunc(OC + 63, 64)))),
            .z = @truncate(@as(u64, @bitCast(B * split_k))),
        };
        if (use_mma) {
            conv3dLaunchImplicitGemm(true, x_half.get(), w, result, padded_params, split_k, grid, block, stream);
        } else {
            conv3dLaunchImplicitGemm(false, x_half.get(), w, result, padded_params, split_k, grid, block, stream);
        }
        if (split_k > 1) {
            kernels.launch(kernels.get("conv3d", "conv3d_reduce_split_k"), .{
                .grid = .{ .x = @truncate(@as(u64, @bitCast(@divTrunc(total + 255, 256)))) },
                .block = .{ .x = 256 },
                .shmem = 0,
                .stream = stream,
            }, .{ @as(?*const anyopaque, result), y, @as(c_int, @truncate(total)), @as(c_int, @truncate(OC * positions)), split_k });
        }
        return;
    }
    const blocks: c_int = @intCast(@min(@as(i64, 65535), @divTrunc(total + 255, 256)));
    const lp: kernels.LaunchParams = .{ .grid = .{ .x = @bitCast(blocks) }, .block = .{ .x = 256 }, .shmem = 0, .stream = stream };
    if (kernel.type == c.GGML_TYPE_F16) {
        kernels.launch(kernels.get("conv3d", kernels.tmpl("conv3d_kernel", &.{kernels.typeName(half)})), lp, .{ x, w, y, params });
    } else {
        kernels.launch(kernels.get("conv3d", kernels.tmpl("conv3d_kernel", &.{kernels.typeName(f32)})), lp, .{ x, w, y, params });
    }
}
