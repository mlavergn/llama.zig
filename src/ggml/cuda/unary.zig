//! Elementwise unary ops (`GGML_OP_UNARY`), the gated ones (`GGML_OP_GLU`),
//! `SQR`, `SQRT`, `SIN`, `COS`, `LOG`, `LEAKY_RELU`, `SILU_BACK`, and the
//! two fused entry points `graph.zig`'s fusion calls: unary + `MUL`, and
//! `RELU` + `SQR`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/unary.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/unary.cuh` at v0.4.1 (`b29c606e2`). The
//! kernels -- `unary_op_kernel`, `unary_gated_op_kernel`,
//! `swiglu_oai_kernel`, `swiglu_clamp_kernel`, `xielu_kernel`,
//! `silu_back_kernel`, `leaky_relu_kernel` -- stay CUDA.
//!
//! # Two launch styles in one file
//!
//! `unary_cuda` and `unary_gated_cuda` go through `ggml_cuda_kernel_launch`
//! (PDL); `swiglu_oai_cuda`, `swiglu_clamp_cuda`, `xielu_cuda`,
//! `silu_back_cuda` and `leaky_relu_cuda` use `<<<>>>`. Kept exactly so.
//!
//! # Grid arithmetic that is not a plain ceiling
//!
//! `xielu_cuda` computes `(k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE`,
//! one block more than needed whenever `k` is a multiple of 256; the kernel's
//! bounds check makes it harmless, and it is reproduced.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `CUDA_NEG_BLOCK_SIZE`, `CUDA_SILU_BLOCK_SIZE`,
/// `CUDA_SILU_BACK_BLOCK_SIZE`, `CUDA_RELU_BLOCK_SIZE`,
/// `CUDA_GLU_BLOCK_SIZE` and `CUDA_XIELU_BLOCK_SIZE`
/// (unary.cuh:4, 7, 8, 10, 19, 20 @b29c606e2). All 256.
const neg_block_size = 256;
const silu_block_size = 256;
const silu_back_block_size = 256;
const relu_block_size = 256;
const glu_block_size = 256;
const xielu_block_size = 256;

/// The `__device__` functions a `unary_op_kernel` or `unary_gated_op_kernel`
/// is instantiated with; the tag is the C name after `op_`.
pub const Op = enum {
    abs,
    sgn,
    neg,
    step,
    gelu,
    gelu_erf,
    gelu_quick,
    silu,
    tanh,
    relu,
    sigmoid,
    hardsigmoid,
    hardswish,
    exp,
    sqr,
    relu_sqr,
    sqrt,
    sin,
    cos,
    log,
    expm1,
    softplus,
    elu,
    floor,
    ceil,
    round,
    trunc,
};

fn unaryKernel(comptime op: Op, comptime T: type) kernels.Kernel {
    return comptime kernels.get("unary", kernels.tmpl("unary_op_kernel", &.{ "&op_" ++ @tagName(op), kernels.typeName(T) }));
}

fn gatedKernel(comptime op: Op, comptime T: type) kernels.Kernel {
    return comptime kernels.get("unary", kernels.tmpl("unary_gated_op_kernel", &.{ "&op_" ++ @tagName(op), kernels.typeName(T) }));
}

/// C's conversion of a signed integer to `unsigned int` (a `dim3` member).
fn uint(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// C's conversion of `int64_t` to an `int` parameter.
fn int(x: i64) c_int {
    return @truncate(x);
}

/// `p + n` on a `T *` held as an opaque pointer.
fn offset(comptime T: type, p: ?*anyopaque, n: i64) ?*anyopaque {
    const base: [*]u8 = @ptrCast(p.?);
    return @ptrCast(base + @as(usize, @intCast(n)) * @sizeOf(T));
}

/// Ports `unary_cuda` (unary.cu:131 @b29c606e2).
fn unaryCuda(comptime op: Op, comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, k: c_int, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(k + neg_block_size - 1, neg_block_size);
    kernels.launchPdl(unaryKernel(op, T), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = neg_block_size },
        .stream = stream,
    }, .{ x, dst, k });
}

/// Ports `ggml_cuda_op_unary` (unary.cu:138 @b29c606e2).
fn opUnaryImpl(comptime op: Op, ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    if (src0.type == c.GGML_TYPE_F16) {
        unaryCuda(op, half, src0_d, dst_d, int(c.ggml_nelements(src0)), stream);
    } else {
        unaryCuda(op, f32, src0_d, dst_d, int(c.ggml_nelements(src0)), stream);
    }
}

/// Ports `ggml_cuda_op_abs`, `ggml_cuda_op_sgn`, `ggml_cuda_op_neg`,
/// `ggml_cuda_op_step`, `ggml_cuda_op_gelu`, `ggml_cuda_op_gelu_erf`,
/// `ggml_cuda_op_gelu_quick`, `ggml_cuda_op_silu`, `ggml_cuda_op_tanh`,
/// `ggml_cuda_op_relu`, `ggml_cuda_op_sigmoid`, `ggml_cuda_op_hardsigmoid`,
/// `ggml_cuda_op_hardswish`, `ggml_cuda_op_exp`, `ggml_cuda_op_sqr`,
/// `ggml_cuda_op_sqrt`, `ggml_cuda_op_sin`, `ggml_cuda_op_cos`,
/// `ggml_cuda_op_log`, `ggml_cuda_op_elu`, `ggml_cuda_op_floor`,
/// `ggml_cuda_op_ceil`, `ggml_cuda_op_round`, `ggml_cuda_op_trunc`,
/// `ggml_cuda_op_expm1` and `ggml_cuda_op_softplus`
/// (unary.cu:157, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197, 201,
/// 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253, 257
/// @b29c606e2): each is `ggml_cuda_op_unary<op_NAME>`.
pub fn opUnary(comptime op: Op, ctx: *common.Context, dst: *Tensor) void {
    comptime std.debug.assert(op != .relu_sqr); // only the fused path uses it
    opUnaryImpl(op, ctx, dst);
}

/// Ports `unary_gated_cuda` (unary.cu:280 @b29c606e2).
fn unaryGatedCuda(comptime op: Op, comptime T: type, x: ?*const anyopaque, g: ?*const anyopaque, dst: ?*anyopaque, k: i64, n: i64, o0: i64, o1: i64, stream: cu.cudaStream_t) void {
    const num_blocks: i64 = @divTrunc(k + glu_block_size - 1, glu_block_size);
    kernels.launchPdl(gatedKernel(op, T), .{
        .grid = .{ .x = uint(num_blocks) },
        .block = .{ .x = glu_block_size },
        .stream = stream,
    }, .{ x, g, dst, k, n, o0, o1 });
}

/// `int64_t / sizeof(T)`: the division is unsigned, the result converted
/// back to `int64_t`.
fn strideOf(comptime T: type, o: i64) i64 {
    return @bitCast(@as(u64, @bitCast(o)) / @sizeOf(T));
}

/// Ports `ggml_cuda_op_unary_gated` (unary.cu:287 @b29c606e2).
fn opUnaryGated(comptime op: Op, ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1_opt: ?*const Tensor = dst.src[1];
    const src0_d: ?*anyopaque = src0.data;
    const src1_d: ?*anyopaque = if (src1_opt) |s1| s1.data else src0.data;
    const src0_o: i64 = @intCast(src0.nb[1]);
    const src1_o: i64 = if (src1_opt) |s1| @intCast(s1.nb[1]) else @intCast(src0.nb[1]);
    const dst_d: ?*anyopaque = dst.data;
    const nc: i64 = if (src1_opt != null) src0.ne[0] else @divTrunc(src0.ne[0], 2);
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous_1(src0), "ggml_is_contiguous_1(src0)");
    impl.assert(src0.nb[0] == c.ggml_element_size(src0), "src0->nb[0] == ggml_element_size(src0)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");
    impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
    impl.assert(c.ggml_nrows(dst) == c.ggml_nrows(src0), "ggml_nrows(dst) == ggml_nrows(src0)");

    if (src1_opt) |src1| {
        impl.assert(c.ggml_is_contiguous_1(src1), "ggml_is_contiguous_1(src1)");
        impl.assert(src1.nb[0] == c.ggml_element_size(src1), "src1->nb[0] == ggml_element_size(src1)");
        impl.assert(src1.ne[0] == nc, "src1->ne[0] == nc");
        impl.assert(src0.type == src1.type, "src0->type == src1->type");
    }

    const swapped: i32 = dst.op_params[1];

    if (src0.type == c.GGML_TYPE_F16) {
        var src0_p = src0_d;
        var src1_p = src1_d;

        if (src1_opt == null) {
            src0_p = offset(half, src0_p, if (swapped != 0) nc else 0);
            src1_p = offset(half, src1_p, if (swapped != 0) 0 else nc);
        }

        unaryGatedCuda(op, half, src0_p, src1_p, dst_d, c.ggml_nelements(dst), nc, strideOf(half, src0_o), strideOf(half, src1_o), stream);
    } else {
        var src0_p = src0_d;
        var src1_p = src1_d;

        if (src1_opt == null) {
            src0_p = offset(f32, src0_p, if (swapped != 0) nc else 0);
            src1_p = offset(f32, src1_p, if (swapped != 0) 0 else nc);
        }

        unaryGatedCuda(op, f32, src0_p, src1_p, dst_d, c.ggml_nelements(dst), nc, strideOf(f32, src0_o), strideOf(f32, src1_o), stream);
    }
}

/// Ports `ggml_cuda_op_reglu`, `ggml_cuda_op_geglu`, `ggml_cuda_op_swiglu`,
/// `ggml_cuda_op_geglu_erf` and `ggml_cuda_op_geglu_quick`
/// (unary.cu:340, 344, 348, 352, 356 @b29c606e2): `ggml_cuda_op_unary_gated`
/// with `op_relu`, `op_gelu`, `op_silu`, `op_gelu_erf` and `op_gelu_quick`.
pub fn opGated(comptime op: Op, ctx: *common.Context, dst: *Tensor) void {
    comptime std.debug.assert(op == .relu or op == .gelu or op == .silu or op == .gelu_erf or op == .gelu_quick);
    opUnaryGated(op, ctx, dst);
}

/// Ports `swiglu_oai_cuda` (unary.cu:381 @b29c606e2), which launches with
/// `<<<>>>`, not PDL.
fn swigluOaiCuda(x: ?*const anyopaque, g: ?*const anyopaque, dst: ?*anyopaque, k: i64, n: i64, o0: i64, o1: i64, alpha: f32, limit: f32, stream: cu.cudaStream_t) void {
    const num_blocks: i64 = @divTrunc(k + glu_block_size - 1, glu_block_size);
    kernels.launch(kernels.get("unary", "swiglu_oai_kernel<float>"), .{
        .grid = .{ .x = uint(num_blocks) },
        .block = .{ .x = glu_block_size },
        .stream = stream,
    }, .{ x, g, dst, k, n, o0, o1, alpha, limit });
}

/// Ports `ggml_cuda_op_swiglu_oai` (unary.cu:386 @b29c606e2).
pub fn opSwigluOai(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1_opt: ?*const Tensor = dst.src[1];
    const src0_d: ?*anyopaque = src0.data;
    const src1_d: ?*anyopaque = if (src1_opt) |s1| s1.data else src0.data;
    const src0_o: i64 = @intCast(src0.nb[1]);
    const src1_o: i64 = if (src1_opt) |s1| @intCast(s1.nb[1]) else @intCast(src0.nb[1]);
    const dst_d: ?*anyopaque = dst.data;
    const nc: i64 = if (src1_opt != null) src0.ne[0] else @divTrunc(src0.ne[0], 2);
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous_1(src0), "ggml_is_contiguous_1(src0)");
    impl.assert(src0.nb[0] == c.ggml_element_size(src0), "src0->nb[0] == ggml_element_size(src0)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, " dst->type == GGML_TYPE_F32");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");
    impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
    impl.assert(c.ggml_nrows(dst) == c.ggml_nrows(src0), "ggml_nrows(dst) == ggml_nrows(src0)");

    if (src1_opt) |src1| {
        impl.assert(c.ggml_is_contiguous_1(src1), "ggml_is_contiguous_1(src1)");
        impl.assert(src1.nb[0] == c.ggml_element_size(src1), "src1->nb[0] == ggml_element_size(src1)");
        impl.assert(src1.ne[0] == nc, "src1->ne[0] == nc");
        impl.assert(src0.type == src1.type, "src0->type == src1->type");
    }

    const swapped = impl.getOpParamsI32(dst, 1);
    const alpha = impl.getOpParamsF32(dst, 2);
    const limit = impl.getOpParamsF32(dst, 3);

    var src0_p = src0_d;
    var src1_p = src1_d;

    if (src1_opt == null) {
        src0_p = offset(f32, src0_p, if (swapped != 0) nc else 0);
        src1_p = offset(f32, src1_p, if (swapped != 0) 0 else nc);
    }

    swigluOaiCuda(src0_p, src1_p, dst_d, c.ggml_nelements(dst), nc, strideOf(f32, src0_o), strideOf(f32, src1_o), alpha, limit, stream);
}

/// Ports `swiglu_clamp_cuda` (unary.cu:447 @b29c606e2), which launches
/// `swiglu_clamp_kernel` (unary.cu:433 @b29c606e2) with `<<<>>>`, not PDL.
fn swigluClampCuda(comptime T: type, gate: ?*const anyopaque, up: ?*const anyopaque, dst: ?*anyopaque, k: i64, n: i64, o0: i64, o1: i64, limit: f32, stream: cu.cudaStream_t) void {
    const num_blocks: i64 = @divTrunc(k + glu_block_size - 1, glu_block_size);
    kernels.launch(comptime kernels.get("unary", kernels.tmpl("swiglu_clamp_kernel", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = uint(num_blocks) },
        .block = .{ .x = glu_block_size },
        .stream = stream,
    }, .{ gate, up, dst, k, n, o0, o1, limit });
}

/// Ports `ggml_cuda_op_swiglu_clamp` (unary.cu:452 @b29c606e2).
pub fn opSwigluClamp(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1_opt: ?*const Tensor = dst.src[1];
    const src0_d: ?*anyopaque = src0.data;
    const src1_d: ?*anyopaque = if (src1_opt) |s1| s1.data else src0.data;
    const src0_o: i64 = @intCast(src0.nb[1]);
    const src1_o: i64 = if (src1_opt) |s1| @intCast(s1.nb[1]) else @intCast(src0.nb[1]);
    const dst_d: ?*anyopaque = dst.data;
    const nc: i64 = if (src1_opt != null) src0.ne[0] else @divTrunc(src0.ne[0], 2);
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous_1(src0), "ggml_is_contiguous_1(src0)");
    impl.assert(src0.nb[0] == c.ggml_element_size(src0), "src0->nb[0] == ggml_element_size(src0)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");
    impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
    impl.assert(c.ggml_nrows(dst) == c.ggml_nrows(src0), "ggml_nrows(dst) == ggml_nrows(src0)");

    if (src1_opt) |src1| {
        impl.assert(c.ggml_is_contiguous_1(src1), "ggml_is_contiguous_1(src1)");
        impl.assert(src1.nb[0] == c.ggml_element_size(src1), "src1->nb[0] == ggml_element_size(src1)");
        impl.assert(src1.ne[0] == nc, "src1->ne[0] == nc");
        impl.assert(src0.type == src1.type, "src0->type == src1->type");
    }

    const swapped = impl.getOpParamsI32(dst, 1);
    const limit = impl.getOpParamsF32(dst, 3);

    if (src0.type == c.GGML_TYPE_F16) {
        var src0_p = src0_d;
        var src1_p = src1_d;

        if (src1_opt == null) {
            src0_p = offset(half, src0_p, if (swapped != 0) nc else 0);
            src1_p = offset(half, src1_p, if (swapped != 0) 0 else nc);
        }

        swigluClampCuda(half, src0_p, src1_p, dst_d, c.ggml_nelements(dst), nc, strideOf(half, src0_o), strideOf(half, src1_o), limit, stream);
    } else {
        var src0_p = src0_d;
        var src1_p = src1_d;

        if (src1_opt == null) {
            src0_p = offset(f32, src0_p, if (swapped != 0) nc else 0);
            src1_p = offset(f32, src1_p, if (swapped != 0) 0 else nc);
        }

        swigluClampCuda(f32, src0_p, src1_p, dst_d, c.ggml_nelements(dst), nc, strideOf(f32, src0_o), strideOf(f32, src1_o), limit, stream);
    }
}

/// Ports `xielu_cuda` (unary.cu:527 @b29c606e2).
fn xieluCuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, k: c_int, alpha_n: f32, alpha_p: f32, beta: f32, eps: f32, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(k + xielu_block_size, xielu_block_size);
    kernels.launch(kernels.get("unary", kernels.tmpl("xielu_kernel", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = xielu_block_size },
        .stream = stream,
    }, .{ x, dst, k, alpha_n, alpha_p, beta, eps });
}

/// Ports `ggml_cuda_op_xielu` (unary.cu:532 @b29c606e2).
pub fn opXielu(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    const alpha_n = impl.getOpParamsF32(dst, 1);
    const alpha_p = impl.getOpParamsF32(dst, 2);
    const beta = impl.getOpParamsF32(dst, 3);
    const eps = impl.getOpParamsF32(dst, 4);

    if (src0.type == c.GGML_TYPE_F16) {
        xieluCuda(half, src0_d, dst_d, int(c.ggml_nelements(src0)), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xieluCuda(f32, src0_d, dst_d, int(c.ggml_nelements(src0)), alpha_n, alpha_p, beta, eps, stream);
    }
}

/// Ports `silu_back_cuda` (unary.cu:577 @b29c606e2). The C divides by
/// `CUDA_SILU_BLOCK_SIZE` after adding `CUDA_SILU_BACK_BLOCK_SIZE - 1`; both
/// are 256.
fn siluBackCuda(comptime T: type, grad: ?*const anyopaque, x: ?*const anyopaque, dst: ?*anyopaque, k: c_int, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(k + silu_back_block_size - 1, silu_block_size);
    kernels.launch(kernels.get("unary", kernels.tmpl("silu_back_kernel", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = silu_back_block_size },
        .stream = stream,
    }, .{ grad, x, dst, k });
}

/// Ports `ggml_cuda_op_silu_back` (unary.cu:582 @b29c606e2).
pub fn opSiluBack(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // input from forward pass
    const src1 = impl.one(Tensor, dst.src[1]); // grads of forward pass output

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;
    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    if (src0.type == c.GGML_TYPE_F16) {
        siluBackCuda(half, src0_d, src1_d, dst_d, int(c.ggml_nelements(src0)), stream);
    } else {
        siluBackCuda(f32, src0_d, src1_d, dst_d, int(c.ggml_nelements(src0)), stream);
    }
}

/// Ports `leaky_relu_cuda` (unary.cu:623 @b29c606e2).
fn leakyReluCuda(comptime T: type, x: ?*const anyopaque, dst: ?*anyopaque, k: c_int, negative_slope: f32, stream: cu.cudaStream_t) void {
    const num_blocks: c_int = @divTrunc(k + relu_block_size - 1, relu_block_size);
    kernels.launch(kernels.get("unary", kernels.tmpl("leaky_relu_kernel", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @bitCast(num_blocks) },
        .block = .{ .x = relu_block_size },
        .stream = stream,
    }, .{ x, dst, k, negative_slope });
}

/// Ports `ggml_cuda_op_leaky_relu` (unary.cu:628 @b29c606e2).
pub fn opLeakyRelu(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, " dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    const negative_slope: f32 = @bitCast(dst.op_params[0]);

    if (src0.type == c.GGML_TYPE_F16) {
        leakyReluCuda(half, src0_d, dst_d, int(c.ggml_nelements(src0)), negative_slope, stream);
    } else {
        leakyReluCuda(f32, src0_d, dst_d, int(c.ggml_nelements(src0)), negative_slope, stream);
    }
}

/// Ports `ggml_cuda_op_unary_mul_impl` (unary.cu:653 @b29c606e2).
fn opUnaryMulImpl(comptime op: Op, ctx: *common.Context, unary_node: *Tensor, mul_node: *Tensor) void {
    // unary_node: UNARY op applied to unary_node->src[0]
    // mul_node:   MUL(a, b) where one of a/b is unary_node
    // Output goes to mul_node->data

    const unary_src = impl.one(Tensor, unary_node.src[0]); // input to the unary op
    const other_src = impl.one(Tensor, if (mul_node.src[0] == unary_node) mul_node.src[1] else mul_node.src[0]);

    impl.assert(c.ggml_is_contiguous_1(unary_src), "ggml_is_contiguous_1(unary_src)");
    impl.assert(unary_src.nb[0] == c.ggml_element_size(unary_src), "unary_src->nb[0] == ggml_element_size(unary_src)");
    impl.assert(c.ggml_is_contiguous_1(other_src), "ggml_is_contiguous_1(other_src)");
    impl.assert(other_src.nb[0] == c.ggml_element_size(other_src), "other_src->nb[0] == ggml_element_size(other_src)");
    impl.assert(c.ggml_are_same_shape(unary_src, other_src), "ggml_are_same_shape(unary_src, other_src)");

    impl.assert(unary_src.type == c.GGML_TYPE_F32 or unary_src.type == c.GGML_TYPE_F16, "unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16");
    impl.assert(unary_src.type == other_src.type, "unary_src->type == other_src->type");
    impl.assert(unary_src.type == mul_node.type, "unary_src->type == mul_node->type");

    const stream = ctx.stream();

    const k = c.ggml_nelements(mul_node);
    const nc = unary_src.ne[0];
    const unary_stride: i64 = @intCast(unary_src.nb[1]);
    const other_stride: i64 = @intCast(other_src.nb[1]);

    if (unary_src.type == c.GGML_TYPE_F16) {
        unaryGatedCuda(op, half, unary_src.data, other_src.data, mul_node.data, k, nc, strideOf(half, unary_stride), strideOf(half, other_stride), stream);
    } else {
        unaryGatedCuda(op, f32, unary_src.data, other_src.data, mul_node.data, k, nc, strideOf(f32, unary_stride), strideOf(f32, other_stride), stream);
    }
}

/// Ports `ggml_cuda_op_unary_mul` (unary.cu:689 @b29c606e2), the fused
/// unary + `MUL`.
pub fn opUnaryMul(ctx: *common.Context, unary_node: *Tensor, mul_node: *Tensor) void {
    switch (c.ggml_get_unary_op(unary_node)) {
        c.GGML_UNARY_OP_SILU => opUnaryMulImpl(.silu, ctx, unary_node, mul_node),
        c.GGML_UNARY_OP_SIGMOID => opUnaryMulImpl(.sigmoid, ctx, unary_node, mul_node),
        c.GGML_UNARY_OP_SOFTPLUS => opUnaryMulImpl(.softplus, ctx, unary_node, mul_node),
        else => impl.abort("Unsupported unary op for fused unary+mul"),
    }
}

/// Ports `ggml_cuda_op_relu_sqr` (unary.cu:707 @b29c606e2), the fused
/// `RELU` + `SQR`.
pub fn opReluSqr(ctx: *common.Context, relu_node: *Tensor, sqr_node: *Tensor) void {
    const src = impl.one(Tensor, relu_node.src[0]);
    const stream = ctx.stream();

    impl.assert(c.ggml_is_contiguous(src), "ggml_is_contiguous(src)");
    impl.assert(src.type == c.GGML_TYPE_F32 or src.type == c.GGML_TYPE_F16, "src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16");
    impl.assert(src.type == sqr_node.type, "src->type == sqr_node->type");

    const k: c_int = int(c.ggml_nelements(src));
    if (src.type == c.GGML_TYPE_F16) {
        unaryCuda(.relu_sqr, half, src.data, sqr_node.data, k, stream);
    } else {
        unaryCuda(.relu_sqr, f32, src.data, sqr_node.data, k, stream);
    }
}
