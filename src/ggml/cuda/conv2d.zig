//! `GGML_OP_CONV_2D`: direct 2-D convolution, WHCN layout only.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/conv2d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/conv2d.cuh` at v0.4.0 (`5266f24da`). The
//! kernel, `conv2d_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `CUDA_CONV2D_BLOCK_SIZE` (conv2d.cuh:4 @5266f24da).
const block_size = 256;

/// Ports `conv_params` (conv2d.cu:4 @5266f24da), passed to the kernel by
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

/// Ports `conv2d_cuda` (conv2d.cu:109 @5266f24da), reached through
/// `conv2d_cuda_f16` and `conv2d_cuda_f32` (conv2d.cu:114, 118 @5266f24da).
fn conv2dCuda(comptime T: type, X_D: ?*const anyopaque, K_D: ?*const anyopaque, Y_D: ?*anyopaque, P: ConvParams, st: cu.cudaStream_t) void {
    const blocks: c_int = @truncate(@divTrunc(P.TOTAL + block_size - 1, block_size));
    kernels.launch(kernels.get("conv2d", kernels.tmpl("conv2d_kernel", &.{ kernels.typeName(T), "whcn_layout" })), .{
        .grid = .{ .x = @bitCast(blocks) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = st,
    }, .{ X_D, K_D, Y_D, P });
}

/// Ports `ggml_cuda_op_conv2d` (conv2d.cu:122 @5266f24da).
pub fn opConv2d(ctx: *common.Context, dst: *Tensor) void {
    const kernel = impl.one(Tensor, dst.src[0]);
    const input = impl.one(Tensor, dst.src[1]);
    const K_D: ?*const anyopaque = kernel.data;
    const X_D: ?*const anyopaque = input.data;
    const Y_D: ?*anyopaque = dst.data;

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

    const IW: c_int = @truncate(input.ne[0]); // input_w
    const IH: c_int = @truncate(input.ne[1]); // input_h
    const OW: c_int = @truncate(dst.ne[0]); // output_w
    const OH: c_int = @truncate(dst.ne[1]); // output_h
    const KW: c_int = @truncate(kernel.ne[0]); // kernel_w
    const KH: c_int = @truncate(kernel.ne[1]); // kernel_h
    const IC: c_int = @truncate(input.ne[2]); // input_channels
    const OC: c_int = @truncate(kernel.ne[3]); // ouptut_chanles
    const B: c_int = @truncate(input.ne[3]); // n_batches

    // `int` arithmetic in the C.
    const total: i64 = B *% OC *% OH *% OW;
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

    if (kernel.type == c.GGML_TYPE_F16) {
        conv2dCuda(half, X_D, K_D, Y_D, params, st);
    } else {
        conv2dCuda(f32, X_D, K_D, Y_D, params, st);
    }
}
