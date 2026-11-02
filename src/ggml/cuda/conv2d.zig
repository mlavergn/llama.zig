//! `GGML_OP_CONV_2D`: 2-D convolution, WHCN layout only.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/conv2d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/conv2d.cuh` at v0.5.0 (`7fe450e19`). The
//! kernels -- `conv2d_kernel`, `conv2d_pad_f16`, `conv2d_implicit_gemm_f16`
//! and `conv2d_reduce_split_k` -- stay CUDA.
//!
//! # Paths
//!
//! An F16 kernel that fits in `int` takes the implicit GEMM (v0.5.0): the
//! input is converted to F16 (padded first if the convolution pads), then a
//! pointwise convolution is one cuBLAS GEMM per batch on hardware with fast
//! F16, and anything else a tiled kernel -- tensor-core MMA from Turing on,
//! SIMT on Moore Threads -- whose long reductions are split `split_k` ways
//! and summed by `conv2d_reduce_split_k`. Everything else is the direct
//! kernel.

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

/// Ports `CUDA_CONV2D_BLOCK_SIZE` (conv2d.cuh:4 @7fe450e19).
const block_size = 256;

/// Ports `conv_params` (conv2d.cu:5 @7fe450e19), passed to the kernel by
/// value.
const ConvParams = extern struct {
    IW: i64,
    IH: i64,
    OW: i64,
    OH: i64,
    KW: i64,
    KH: i64,
    ST_X: i64,
    ST_Y: i64,
    PD_X: i64,
    PD_Y: i64,
    DL_X: i64,
    DL_Y: i64,
    IC: i64,
    OC: i64,
    B: i64,
    TOTAL: i64,
};

/// Ports `conv2d_cuda` (conv2d.cu:110 @7fe450e19), reached through
/// `conv2d_cuda_f16` and `conv2d_cuda_f32` (conv2d.cu:329, 333 @7fe450e19).
fn conv2dCuda(comptime T: type, X_D: ?*const anyopaque, K_D: ?*const anyopaque, Y_D: ?*anyopaque, P: ConvParams, st: cu.cudaStream_t) void {
    const blocks: c_int = @truncate(@divTrunc(P.TOTAL + block_size - 1, block_size));
    kernels.launch(kernels.get("conv2d", kernels.tmpl("conv2d_kernel", &.{ kernels.typeName(T), "whcn_layout" })), .{
        .grid = .{ .x = @bitCast(blocks) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = st,
    }, .{ X_D, K_D, Y_D, P });
}

/// Ports `conv2d_launch_implicit_gemm` (conv2d.cu:312 @7fe450e19): the 3x3
/// and 1x1 kernels have their sizes as template arguments, the rest read
/// them from `params`.
fn conv2dLaunchImplicitGemm(
    comptime use_mma: bool,
    input: ?*const anyopaque,
    weight: ?*const anyopaque,
    output: ?*anyopaque,
    params: ConvParams,
    split_k: c_int,
    grid: kernels.Dim3,
    block: kernels.Dim3,
    stream: cu.cudaStream_t,
) void {
    const lp: kernels.LaunchParams = .{ .grid = grid, .block = block, .shmem = 0, .stream = stream };
    const args = .{ input, weight, output, params, split_k };
    const mma = comptime kernels.boolean(use_mma);
    if (params.KW == 3 and params.KH == 3) {
        kernels.launch(kernels.get("conv2d", kernels.tmpl("conv2d_implicit_gemm_f16", &.{ kernels.int(3), kernels.int(3), mma })), lp, args);
    } else if (params.KW == 1 and params.KH == 1) {
        kernels.launch(kernels.get("conv2d", kernels.tmpl("conv2d_implicit_gemm_f16", &.{ kernels.int(1), kernels.int(1), mma })), lp, args);
    } else {
        kernels.launch(kernels.get("conv2d", kernels.tmpl("conv2d_implicit_gemm_f16", &.{ kernels.int(0), kernels.int(0), mma })), lp, args);
    }
}

/// Ports `ggml_cuda_op_conv2d` (conv2d.cu:337 @7fe450e19).
pub fn opConv2d(ctx: *common.Context, dst: *Tensor) void {
    const kernel = impl.one(Tensor, dst.src[0]);
    const input = impl.one(Tensor, dst.src[1]);
    const K_D: ?*const anyopaque = kernel.data;
    const X_D: ?*const anyopaque = input.data;
    const Y_D: ?*anyopaque = dst.data;

    impl.assert(input.type == c.GGML_TYPE_F32 and dst.type == c.GGML_TYPE_F32, "input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(input), "ggml_is_contiguous(input)");
    impl.assert(c.ggml_is_contiguous(kernel), "ggml_is_contiguous(kernel)");
    impl.assert(kernel.type == c.GGML_TYPE_F16 or kernel.type == c.GGML_TYPE_F32, "kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32");

    // same number of input channels
    impl.assert(input.ne[2] == kernel.ne[2], "input->ne[2] == kernel->ne[2]");

    const st = ctx.stream();

    const ST_X = impl.getOpParamsI32(dst, 0); // stride_x
    const ST_Y = impl.getOpParamsI32(dst, 1); // stride_y
    const PD_X = impl.getOpParamsI32(dst, 2); // padding_x
    const PD_Y = impl.getOpParamsI32(dst, 3); // padding_y
    const DL_X = impl.getOpParamsI32(dst, 4); // dilation_x
    const DL_Y = impl.getOpParamsI32(dst, 5); // dilation_y

    // No cwhn
    impl.assert(impl.getOpParamsI32(dst, 6) == 0, "p[6] == false");

    const IW: i64 = input.ne[0]; // input_w
    const IH: i64 = input.ne[1]; // input_h
    const OW: i64 = dst.ne[0]; // output_w
    const OH: i64 = dst.ne[1]; // output_h
    const KW: i64 = kernel.ne[0]; // kernel_w
    const KH: i64 = kernel.ne[1]; // kernel_h
    const IC: i64 = input.ne[2]; // input_channels
    const OC: i64 = kernel.ne[3]; // ouptut_chanles
    const B: i64 = input.ne[3]; // n_batches

    const total: i64 = B * OC * OH * OW;
    const params: ConvParams = .{
        .IW = IW,
        .IH = IH,
        .OW = OW,
        .OH = OH,
        .KW = KW,
        .KH = KH,
        .ST_X = ST_X,
        .ST_Y = ST_Y,
        .PD_X = PD_X,
        .PD_Y = PD_Y,
        .DL_X = DL_X,
        .DL_Y = DL_Y,
        .IC = IC,
        .OC = OC,
        .B = B,
        .TOTAL = total,
    };

    const device = &common.info().devices[@intCast(ctx.device)];
    const use_mma = common.turingMmaAvailable(device.cc) or fattn_common.amdWmmaAvailable(device.cc) or fattn_common.amdMfmaAvailable(device.cc);
    // MUSA can share the tiling without a native fragment implementation in mma.cuh.
    const use_simt = common.cc.isMthreads(device.cc);
    const pointwise = KW == 1 and KH == 1 and ST_X == 1 and ST_Y == 1 and PD_X == 0 and PD_Y == 0;
    const use_blas = pointwise and common.fastFp16HardwareAvailable(device.cc);
    // Short reductions on small maps do not amortize conversion and launch costs.
    const small_conv = IC * KW * KH < 64 and OW * OH < 512;

    const limit: i64 = std.math.maxInt(c_int) - 256;
    const padded_w: i64 = IW + 2 * @as(i64, PD_X);
    const padded_h: i64 = IH + 2 * @as(i64, PD_Y);
    const padded_fits = padded_w > 0 and padded_w <= limit and padded_h > 0 and padded_h <= limit and
        padded_w * padded_h <= limit and IC * B <= @divTrunc(limit, padded_w * padded_h);
    if (kernel.type == c.GGML_TYPE_F16 and (use_mma or use_blas or use_simt) and (use_blas or !small_conv) and
        c.ggml_nelements(input) <= limit and c.ggml_nelements(kernel) <= limit and total <= limit and padded_fits and
        PD_X >= 0 and PD_Y >= 0 and ST_X > 0 and ST_Y > 0 and DL_X > 0 and DL_Y > 0 and
        (OW - 1) * ST_X + (KW - 1) * DL_X < padded_w and (OH - 1) * ST_Y + (KH - 1) * DL_Y < padded_h and
        @divTrunc(OC + 63, 64) <= 65535 and B <= 65535)
    {
        const pw: c_int = @intCast(padded_w);
        const ph: c_int = @intCast(padded_h);
        const padded_total: c_int = @truncate(padded_w * padded_h * IC * B);

        var x_half = common.PoolAlloc(half).initAlloc(ctx.pool(), @intCast(padded_total));
        defer x_half.deinit();
        // Match im2col's F16 input precision, but expand patches only in shared memory and accumulate in F32.
        if (PD_X == 0 and PD_Y == 0) {
            convert.getToFp16(input.type).?(X_D, x_half.get(), padded_total, st);
        } else {
            kernels.launch(kernels.get("conv2d", "conv2d_pad_f16"), .{
                .grid = .{ .x = @intCast(@divTrunc(padded_total + 255, 256)) },
                .block = .{ .x = 256 },
                .shmem = 0,
                .stream = st,
            }, .{ X_D, @as(?*anyopaque, x_half.get()), @as(c_int, @truncate(IW)), @as(c_int, @truncate(IH)), pw, ph, PD_X, PD_Y, padded_total });
        }
        const padded_params: ConvParams = .{
            .IW = pw,
            .IH = ph,
            .OW = OW,
            .OH = OH,
            .KW = KW,
            .KH = KH,
            .ST_X = ST_X,
            .ST_Y = ST_Y,
            .PD_X = 0,
            .PD_Y = 0,
            .DL_X = DL_X,
            .DL_Y = DL_Y,
            .IC = IC,
            .OC = OC,
            .B = B,
            .TOTAL = total,
        };
        if (use_blas) {
            const alpha: f32 = 1.0;
            const beta: f32 = 0.0;
            const positions: c_int = @truncate(OW * OH);
            const cublas_h = ctx.cublasHandle();
            const xh: [*]const half = x_half.get().?;
            const yd: [*]f32 = @ptrCast(@alignCast(Y_D.?));
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
                    K_D,
                    cu.CUDA_R_16F,
                    @as(c_int, @truncate(IC)),
                    &beta,
                    yd + @as(usize, @intCast(@as(i64, n) * OC * positions)),
                    cu.CUDA_R_32F,
                    positions,
                    cu.CUBLAS_COMPUTE_32F,
                    cu.CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                ), "cublasGemmEx(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, positions, int(OC), int(IC), &alpha, x_half.get() + int64_t(n) * IC * positions, CUDA_R_16F, positions, K_D, CUDA_R_16F, int(IC), &beta, Y_D + int64_t(n) * OC * positions, CUDA_R_32F, positions, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)", @src());
            }
            return;
        }
        const blocks: i64 = @divTrunc(OW * OH + 63, 64) * @divTrunc(OC + 63, 64) * B;
        const target: c_int = 8 * common.info().devices[@intCast(ctx.device)].nsm;
        // Split long reductions so small spatial maps still occupy the GPU.
        const split_k: c_int = @intCast(@min(
            @min(@as(i64, 32), @divTrunc(@as(i64, 65535), B)),
            @min(@divTrunc(IC * KW * KH + 63, 64), @max(@as(i64, 1), @divTrunc(target + blocks - 1, blocks))),
        ));

        var partial = common.PoolAlloc(f32).init(ctx.pool());
        defer partial.deinit();
        const result: ?*anyopaque = if (split_k == 1) Y_D else @ptrCast(partial.alloc(@intCast(total * split_k)));
        const block: kernels.Dim3 = .{ .x = @bitCast(device.warp_size), .y = 4 };
        const grid: kernels.Dim3 = .{
            .x = @truncate(@as(u64, @bitCast(@divTrunc(OW * OH + 63, 64)))),
            .y = @truncate(@as(u64, @bitCast(@divTrunc(OC + 63, 64)))),
            .z = @truncate(@as(u64, @bitCast(B * split_k))),
        };
        if (use_mma) {
            conv2dLaunchImplicitGemm(true, x_half.get(), K_D, result, padded_params, split_k, grid, block, st);
        } else {
            conv2dLaunchImplicitGemm(false, x_half.get(), K_D, result, padded_params, split_k, grid, block, st);
        }
        if (split_k > 1) {
            kernels.launch(kernels.get("conv2d", "conv2d_reduce_split_k"), .{
                .grid = .{ .x = @truncate(@as(u64, @bitCast(@divTrunc(total + 255, 256)))) },
                .block = .{ .x = 256 },
                .shmem = 0,
                .stream = st,
            }, .{ @as(?*const anyopaque, result), Y_D, @as(c_int, @truncate(total)), @as(c_int, @truncate(OC * OW * OH)), split_k });
        }
        return;
    }

    if (kernel.type == c.GGML_TYPE_F16) {
        conv2dCuda(half, X_D, K_D, Y_D, params, st);
    } else {
        conv2dCuda(f32, X_D, K_D, Y_D, params, st);
    }
}
