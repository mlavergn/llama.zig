//! `GGML_OP_SSM_CONV`: the depthwise causal convolution of Mamba and
//! Qwen3.5's delta-net layers, with the fused `+ bias` and `silu` the
//! compute loop's fusion asks for.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/ssm-conv.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels, `ssm_conv_f32` and `ssm_conv_long_token_f32`, stay CUDA.
//!
//! The short-sequence kernel goes through `ggml_cuda_kernel_launch` (PDL);
//! the long-token one through `<<<>>>`. That asymmetry is upstream's.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// A `size_t` passed into an `int` parameter: C++ converts modulo 2^32.
fn int(x: usize) c_int {
    return @bitCast(@as(u32, @truncate(x)));
}

/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `ssm_conv_f32_cuda` (ssm-conv.cu:127 @c1d0e7a00).
fn ssmConvF32Cuda(
    comptime apply_silu: bool,
    src0: ?*const anyopaque,
    src1: ?*const anyopaque,
    bias: ?*const anyopaque,
    src0_nb0: c_int,
    src0_nb1: c_int,
    src0_nb2: c_int,
    src1_nb1: c_int,
    dst: ?*anyopaque,
    dst_nb0: c_int,
    dst_nb1: c_int,
    dst_nb2: c_int,
    nc: i64,
    nr: i64,
    n_t: i64,
    n_s: i64,
    stream: cu.cudaStream_t,
) void {
    const threads = 128;
    impl.assert(@rem(nr, threads) == 0, "nr % threads == 0");

    const launchKernel = struct {
        fn f(comptime kNC: comptime_int, a: anytype) void {
            if (a.n_t <= 32) {
                const blocks: kernels.Dim3 = .{ .x = dim(a.n_s), .y = dim(@divTrunc(a.nr + threads - 1, threads)), .z = 1 };
                kernels.launchPdl(kernels.get("ssm-conv", kernels.tmpl("ssm_conv_f32", &.{ kernels.boolean(apply_silu), kernels.ulong(threads), kernels.ulong(kNC) })), .{
                    .grid = blocks,
                    .block = .{ .x = threads },
                    .shmem = 0,
                    .stream = a.stream,
                }, .{ a.src0, a.src1, a.bias, a.src0_nb0, a.src0_nb1, a.src0_nb2, a.src1_nb1, a.dst, a.dst_nb0, a.dst_nb1, a.dst_nb2, a.n_t });
            } else {
                const split_n_t: i64 = 32;
                const blocks: kernels.Dim3 = .{ .x = dim(a.n_s), .y = dim(@divTrunc(a.nr + threads - 1, threads)), .z = dim(@divTrunc(a.n_t + split_n_t - 1, split_n_t)) };
                const smem_size: usize = threads * (kNC - 1 + split_n_t) * @sizeOf(f32);
                kernels.launch(kernels.get("ssm-conv", kernels.tmpl("ssm_conv_long_token_f32", &.{ kernels.boolean(apply_silu), kernels.ulong(threads), kernels.ulong(kNC), kernels.long(split_n_t) })), .{
                    .grid = blocks,
                    .block = .{ .x = threads },
                    .shmem = smem_size,
                    .stream = a.stream,
                }, .{ a.src0, a.src1, a.bias, a.src0_nb0, a.src0_nb1, a.src0_nb2, a.src1_nb1, a.dst, a.dst_nb0, a.dst_nb1, a.dst_nb2, a.n_t });
            }
        }
    }.f;

    const a = .{
        .src0 = src0,
        .src1 = src1,
        .bias = bias,
        .src0_nb0 = src0_nb0,
        .src0_nb1 = src0_nb1,
        .src0_nb2 = src0_nb2,
        .src1_nb1 = src1_nb1,
        .dst = dst,
        .dst_nb0 = dst_nb0,
        .dst_nb1 = dst_nb1,
        .dst_nb2 = dst_nb2,
        .nr = nr,
        .n_t = n_t,
        .n_s = n_s,
        .stream = stream,
    };

    switch (nc) {
        3 => launchKernel(3, a),
        4 => launchKernel(4, a),
        5 => launchKernel(5, a),
        9 => launchKernel(9, a),
        15 => launchKernel(15, a),
        else => impl.abort("Only support kernel sizes 3, 4, 5, 9, 15 right now."),
    }
}

/// Ports `ggml_cuda_op_ssm_conv` (ssm-conv.cu:160 @c1d0e7a00).
///
/// Parameters:
/// - `bias_add_node`: the fused `ADD` whose other operand is the bias, or
///   null; never without `silu_dst`.
/// - `silu_dst`: the fused `SILU`, which the result is written to, or null.
pub fn opSsmConv(ctx: *common.Context, dst: *Tensor, bias_add_node: ?*Tensor, silu_dst: ?*Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // conv_x
    const src1 = impl.one(Tensor, dst.src[1]); // conv1d.weight
    const fuse_bias = bias_add_node != null;
    const fuse_silu = silu_dst != null;

    // bias always comes with silu.
    impl.assert(!fuse_bias or fuse_silu, "!fuse_bias || fuse_silu");

    // The bias (when fused) is the non-conv operand of the ADD node.
    const bias: ?*const Tensor = if (bias_add_node) |b|
        (if (b.src[0] == @as([*c]Tensor, dst)) impl.one(Tensor, b.src[1]) else impl.one(Tensor, b.src[0]))
    else
        null;

    // When fusing, write to silu_dst (the node downstream references).
    const out: *const Tensor = if (silu_dst) |s| s else dst;

    const nc = src1.ne[0]; // d_conv
    const nr = src0.ne[1]; // d_inner
    const n_t = out.ne[1]; // tokens per sequence
    const n_s = out.ne[2]; // number of sequences in the batch

    impl.assert(out.ne[0] == nr, "out->ne[0] == nr");
    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "src1->nb[0] == sizeof(float)");
    impl.assert(src0.nb[1] == @as(usize, @bitCast(src0.ne[0])) * @sizeOf(f32), "src0->nb[1] == src0->ne[0] * sizeof(float)");

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;
    const bias_d: ?*const anyopaque = if (fuse_bias) bias.?.data else null;
    const dst_d: ?*anyopaque = out.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(out.type == c.GGML_TYPE_F32, "out->type == GGML_TYPE_F32");
    if (fuse_bias) {
        impl.assert(bias.?.type == c.GGML_TYPE_F32, "bias->type == GGML_TYPE_F32");
        impl.assert(c.ggml_is_contiguous(bias.?), "ggml_is_contiguous(bias)");
        impl.assert(c.ggml_nelements(bias.?) == nr, "ggml_nelements(bias) == nr");
    }

    if (fuse_silu) {
        ssmConvF32Cuda(true, src0_d, src1_d, bias_d, int(src0.nb[0]), int(src0.nb[1]), int(src0.nb[2]), int(src1.nb[1]), dst_d, int(out.nb[0]), int(out.nb[1]), int(out.nb[2]), nc, nr, n_t, n_s, stream);
    } else {
        ssmConvF32Cuda(false, src0_d, src1_d, bias_d, int(src0.nb[0]), int(src0.nb[1]), int(src0.nb[2]), int(src1.nb[1]), dst_d, int(out.nb[0]), int(out.nb[1]), int(out.nb[2]), nc, nr, n_t, n_s, stream);
    }
}
