//! `GGML_OP_GET_ROWS` and `GGML_OP_GET_ROWS_BACK`, and `get_rows_cuda`,
//! which `ggml-cuda.cu`'s `MUL_MAT_ID` path calls too.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/getrows.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/getrows.cuh` at v0.4.1 (`b29c606e2`). The
//! kernels -- `k_get_rows`, `k_get_rows_kq`, `k_get_rows_float`,
//! `k_get_rows_float_vec`, `k_get_rows_back_float` -- stay CUDA.
//!
//! # Launches
//!
//! The quantized kernels and the backward kernel launch with `<<<>>>`; the
//! two float kernels through `ggml_cuda_kernel_launch`, so with PDL. That
//! split is upstream's and is kept.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;
const bf16 = kernels.bf16;

/// Ports `CUDA_GET_ROWS_BLOCK_SIZE` (getrows.cuh:3 @b29c606e2).
const block_size = 256;

/// Ports `CUDA_GET_ROWS_BACK_BLOCK_SIZE` (getrows.cuh:4 @b29c606e2).
const back_block_size = 256;

/// `QK_K` (ggml-common.h:89 @b29c606e2).
const qk_k = 256;

const u16_max = std.math.maxInt(u16);
const u32_max = std.math.maxInt(u32);

/// A `dim3` member from an `int64_t`: C++ converts modulo 2^32.
fn dimOf(v: i64) u32 {
    return @truncate(@as(u64, @bitCast(v)));
}

/// The strides and sizes every `get_rows_cuda_*` takes, in the C's order.
const Args = struct {
    ne00: i64,
    nb01: usize,
    nb02: usize,
    nb03: usize,
    ne10: i64,
    ne11: i64,
    ne12: i64,
    nb10: usize,
    nb11: usize,
    nb12: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,
};

/// The `GGML_ASSERT`s and `init_fastdiv_values(ne12)` every launcher shares.
fn ne12Fastdiv(a: *const Args) cu.uint3 {
    impl.assert(a.ne12 > 0, "ne12 > 0");
    impl.assert(a.ne11 <= @divTrunc(@as(i64, u32_max), a.ne12), "ne11 <= std::numeric_limits<uint32_t>::max() / ne12");
    return common.initFastdivValues(@bitCast(a.ne12));
}

/// Ports `get_rows_cuda_q` (getrows.cu:160 @b29c606e2).
fn getRowsCudaQ(comptime qk: comptime_int, comptime qr: comptime_int, comptime dq: []const u8, comptime D: type, src0_d: ?*const anyopaque, src1_d: ?*const anyopaque, dst_d: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    const block_dims: kernels.Dim3 = .{ .x = block_size };
    const block_num_y: c_int = @truncate(@divTrunc(a.ne00 + 2 * block_size - 1, 2 * block_size));
    const block_nums: kernels.Dim3 = .{
        .x = dimOf(a.ne10),
        .y = @bitCast(@min(block_num_y, u16_max)),
        .z = dimOf(@min(a.ne11 * a.ne12, u16_max)),
    };

    // strides in elements
    const s1 = a.nb1 / @sizeOf(D);
    const s2 = a.nb2 / @sizeOf(D);
    const s3 = a.nb3 / @sizeOf(D);

    const s10 = a.nb10 / @sizeOf(i32);
    const s11 = a.nb11 / @sizeOf(i32);
    const s12 = a.nb12 / @sizeOf(i32);

    impl.assert(@rem(a.ne00, 2) == 0, "ne00 % 2 == 0");

    const ne12_fdv = ne12Fastdiv(&a);

    const key = comptime kernels.tmpl("k_get_rows", &.{ kernels.int(qk), kernels.int(qr), "&" ++ dq, kernels.typeName(D) });
    kernels.launch(kernels.get("getrows", key), .{
        .grid = block_nums,
        .block = block_dims,
        .stream = stream,
    }, .{
        src0_d, src1_d, dst_d,
        a.ne00, a.ne11, ne12_fdv,
        s1,     s2,     s3,
        a.nb01, a.nb02, a.nb03,
        s10,    s11,    s12,
    });
}

/// Ports `get_rows_cuda_kq` (getrows.cu:197 @b29c606e2).
fn getRowsCudaKq(comptime block_dim: comptime_int, comptime D: type, comptime dq: []const u8, src0_d: ?*const anyopaque, src1_d: ?*const anyopaque, dst_d: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    impl.assert(@rem(a.ne00, qk_k) == 0, "ne00 % QK_K == 0");
    const nsb: i64 = @divTrunc(a.ne00, qk_k);

    const block_dims: kernels.Dim3 = .{ .x = block_dim };
    const block_nums: kernels.Dim3 = .{
        .x = dimOf(a.ne10),
        .y = dimOf(@min(nsb, u16_max)),
        .z = dimOf(@min(a.ne11 * a.ne12, u16_max)),
    };

    // strides in elements
    const s1 = a.nb1 / @sizeOf(D);
    const s2 = a.nb2 / @sizeOf(D);
    const s3 = a.nb3 / @sizeOf(D);

    const s10 = a.nb10 / @sizeOf(i32);
    const s11 = a.nb11 / @sizeOf(i32);
    const s12 = a.nb12 / @sizeOf(i32);

    const ne12_fdv = ne12Fastdiv(&a);

    const t = comptime kernels.typeName(D);
    const key = comptime kernels.tmpl("k_get_rows_kq", &.{ t, "&" ++ dq ++ "<" ++ t ++ ">" });
    kernels.launch(kernels.get("getrows", key), .{
        .grid = block_nums,
        .block = block_dims,
        .stream = stream,
    }, .{
        src0_d, src1_d, dst_d,
        a.ne00, a.ne11, ne12_fdv,
        s1,     s2,     s3,
        a.nb01, a.nb02, a.nb03,
        s10,    s11,    s12,
    });
}

/// Ports `get_rows_cuda_float` (getrows.cu:234 @b29c606e2).
fn getRowsCudaFloat(comptime S: type, comptime D: type, src0_d: ?*const anyopaque, src1_d: ?*const anyopaque, dst_d: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    const block_dims: kernels.Dim3 = .{ .x = block_size };

    // strides in elements
    const s1 = a.nb1 / @sizeOf(D);
    const s2 = a.nb2 / @sizeOf(D);
    const s3 = a.nb3 / @sizeOf(D);

    const s10 = a.nb10 / @sizeOf(i32);
    const s11 = a.nb11 / @sizeOf(i32);
    const s12 = a.nb12 / @sizeOf(i32);

    const ne12_fdv = ne12Fastdiv(&a);

    if (S == D) {
        const vec = 16 / @sizeOf(D);
        const ne00v: i64 = @divTrunc(a.ne00, vec);
        const vec_block_num_y: i64 = @divTrunc(ne00v + block_size - 1, block_size);
        const enough_blocks = vec_block_num_y * a.ne10 * a.ne11 * a.ne12 >= 128;
        const can_vec = vec > 1 and enough_blocks and
            (@rem(a.ne00, vec) == 0) and
            (a.nb01 % 16 == 0) and (a.nb02 % 16 == 0) and (a.nb03 % 16 == 0) and
            (a.nb1 % 16 == 0) and (a.nb2 % 16 == 0) and (a.nb3 % 16 == 0) and
            (@intFromPtr(src0_d) % 16 == 0) and (@intFromPtr(dst_d) % 16 == 0);

        if (can_vec) {
            const block_num_y: c_int = @truncate(vec_block_num_y);
            const block_nums: kernels.Dim3 = .{
                .x = dimOf(a.ne10),
                .y = @bitCast(@min(block_num_y, u16_max)),
                .z = dimOf(@min(a.ne11 * a.ne12, u16_max)),
            };
            const key = comptime kernels.tmpl("k_get_rows_float_vec", &.{kernels.typeName(D)});
            kernels.launchPdl(kernels.get("getrows", key), .{
                .grid = block_nums,
                .block = block_dims,
                .stream = stream,
            }, .{
                src0_d, src1_d, dst_d,
                ne00v,  a.ne11, ne12_fdv,
                s1,     s2,     s3,
                a.nb01, a.nb02, a.nb03,
                s10,    s11,    s12,
            });
            return;
        }
    }

    const block_num_y: c_int = @truncate(@divTrunc(a.ne00 + block_size - 1, block_size));
    const block_nums: kernels.Dim3 = .{
        .x = dimOf(a.ne10),
        .y = @bitCast(@min(block_num_y, u16_max)),
        .z = dimOf(@min(a.ne11 * a.ne12, u16_max)),
    };

    const key = comptime kernels.tmpl("k_get_rows_float", &.{ kernels.typeName(S), kernels.typeName(D) });
    kernels.launchPdl(kernels.get("getrows", key), .{
        .grid = block_nums,
        .block = block_dims,
        .stream = stream,
    }, .{
        src0_d, src1_d, dst_d,
        a.ne00, a.ne11, ne12_fdv,
        s1,     s2,     s3,
        a.nb01, a.nb02, a.nb03,
        s10,    s11,    s12,
    });
}

/// Ports `ggml_cuda_get_rows_switch_src0_type` (getrows.cu:296 @b29c606e2).
fn switchSrc0Type(comptime D: type, src0_d: ?*const anyopaque, src0_type: c.ggml_type, src1_d: ?*const anyopaque, dst_d: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    switch (src0_type) {
        c.GGML_TYPE_F16 => getRowsCudaFloat(half, D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_F32 => getRowsCudaFloat(f32, D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_I32 => getRowsCudaFloat(i32, D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_BF16 => getRowsCudaFloat(bf16, D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q1_0 => getRowsCudaQ(128, 1, "dequantize_q1_0", D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q2_0 => getRowsCudaQ(64, 1, "dequantize_q2_0", D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q4_0 => getRowsCudaQ(32, 2, "dequantize_q4_0", D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q4_1 => getRowsCudaQ(32, 2, "dequantize_q4_1", D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q5_0 => getRowsCudaQ(32, 2, "dequantize_q5_0", D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q5_1 => getRowsCudaQ(32, 2, "dequantize_q5_1", D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q8_0 => getRowsCudaQ(32, 1, "dequantize_q8_0", D, src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q2_K => getRowsCudaKq(64, D, "dequantize_q2_K", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q3_K => getRowsCudaKq(64, D, "dequantize_q3_K", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q4_K => getRowsCudaKq(32, D, "dequantize_q4_K", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q5_K => getRowsCudaKq(64, D, "dequantize_q5_K", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_Q6_K => getRowsCudaKq(64, D, "dequantize_q6_K", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ2_XXS => getRowsCudaKq(32, D, "dequantize_iq2_xxs", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ2_XS => getRowsCudaKq(32, D, "dequantize_iq2_xs", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ2_S => getRowsCudaKq(32, D, "dequantize_iq2_s", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ3_XXS => getRowsCudaKq(32, D, "dequantize_iq3_xxs", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ3_S => getRowsCudaKq(32, D, "dequantize_iq3_s", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ1_S => getRowsCudaKq(32, D, "dequantize_iq1_s", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ1_M => getRowsCudaKq(32, D, "dequantize_iq1_m", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ4_NL => getRowsCudaKq(32, D, "dequantize_iq4_nl", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_IQ4_XS => getRowsCudaKq(32, D, "dequantize_iq4_xs", src0_d, src1_d, dst_d, a, stream),
        c.GGML_TYPE_MXFP4 => getRowsCudaKq(32, D, "dequantize_mxfp4", src0_d, src1_d, dst_d, a, stream),
        else => c.ggml_abort(@src().file.ptr, @intCast(@src().line), "%s: unsupported src0 type: %s\n", "ggml_cuda_get_rows_switch_src0_type", c.ggml_type_name(src0_type)),
    }
}

/// Ports `get_rows_cuda` (getrows.cu:413 @b29c606e2).
///
/// Parameters:
/// - `src0_d`, `src0_type`: the rows to gather from.
/// - `src1_d`: the `int32_t` row indices.
/// - `dst_d`, `dst_type`: the output.
/// - the rest: the C's sizes and byte strides, in its order.
/// - `stream`: the stream to launch on.
pub fn getRowsCuda(
    src0_d: ?*const anyopaque,
    src0_type: c.ggml_type,
    src1_d: ?*const anyopaque,
    dst_d: ?*anyopaque,
    dst_type: c.ggml_type,
    ne00: i64,
    nb01: usize,
    nb02: usize,
    nb03: usize,
    ne10: i64,
    ne11: i64,
    ne12: i64,
    nb10: usize,
    nb11: usize,
    nb12: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    stream: cu.cudaStream_t,
) void {
    const a: Args = .{
        .ne00 = ne00,
        .nb01 = nb01,
        .nb02 = nb02,
        .nb03 = nb03,
        .ne10 = ne10,
        .ne11 = ne11,
        .ne12 = ne12,
        .nb10 = nb10,
        .nb11 = nb11,
        .nb12 = nb12,
        .nb1 = nb1,
        .nb2 = nb2,
        .nb3 = nb3,
    };
    switch (dst_type) {
        c.GGML_TYPE_F32 => switchSrc0Type(f32, src0_d, src0_type, src1_d, dst_d, a, stream),
        c.GGML_TYPE_I32 => switchSrc0Type(i32, src0_d, src0_type, src1_d, dst_d, a, stream),
        c.GGML_TYPE_F16 => switchSrc0Type(half, src0_d, src0_type, src1_d, dst_d, a, stream),
        c.GGML_TYPE_BF16 => switchSrc0Type(bf16, src0_d, src0_type, src1_d, dst_d, a, stream),
        else => c.ggml_abort(@src().file.ptr, @intCast(@src().line), "%s: unsupported dst type: %s\n", "get_rows_cuda", c.ggml_type_name(dst_type)),
    }
}

fn srcOf(dst: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, dst.src[i]);
}

/// Ports `ggml_cuda_op_get_rows` (getrows.cu:442 @b29c606e2).
pub fn opGetRows(ctx: *common.Context, dst: *Tensor) void {
    const src0 = srcOf(dst, 0);
    const src1 = srcOf(dst, 1);

    const stream = ctx.stream();

    impl.assert(src1.type == c.GGML_TYPE_I32, "src1->type == GGML_TYPE_I32");
    impl.assert(src1.ne[3] == 1, "ne13 == 1");

    impl.assert(src0.nb[0] == c.ggml_type_size(src0.type), "src0->nb[0] == ggml_type_size(src0->type)");
    impl.assert(src1.nb[0] == c.ggml_type_size(src1.type), "src1->nb[0] == ggml_type_size(src1->type)");
    impl.assert(dst.nb[0] == c.ggml_type_size(dst.type), "dst->nb[0]  == ggml_type_size(dst->type)");

    getRowsCuda(
        src0.data,
        src0.type,
        src1.data,
        dst.data,
        dst.type,
        src0.ne[0],
        src0.nb[1],
        src0.nb[2],
        src0.nb[3],
        src1.ne[0],
        src1.ne[1],
        src1.ne[2],
        src1.nb[0],
        src1.nb[1],
        src1.nb[2],
        dst.nb[1],
        dst.nb[2],
        dst.nb[3],
        stream,
    );
}

/// Ports `ggml_cuda_op_get_rows_back` (getrows.cu:461 @b29c606e2).
pub fn opGetRowsBack(ctx: *common.Context, dst: *Tensor) void {
    const src0 = srcOf(dst, 0); // gradients of forward pass output
    const src1 = srcOf(dst, 1); // src1 in forward pass

    const ne00 = src0.ne[0];
    const ne10 = src1.ne[0];
    const ne1 = dst.ne[1];

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;
    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_I32, "src1->type == GGML_TYPE_I32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    impl.assert(src0.ne[2] * src0.ne[3] == 1, "ne02*ne03 == 1");
    impl.assert(src1.ne[2] * src1.ne[3] == 1, "ne12*ne13 == 1");
    impl.assert(dst.ne[2] * dst.ne[3] == 1, "ne2*ne3 == 1");

    const block_dims: kernels.Dim3 = .{ .x = back_block_size };
    const block_num_x: c_int = @truncate(@divTrunc(ne00 + back_block_size - 1, back_block_size));
    const block_nums: kernels.Dim3 = .{ .x = @bitCast(block_num_x), .y = dimOf(@min(ne1, @as(i64, u16_max))), .z = 1 };

    kernels.launch(kernels.get("getrows", "k_get_rows_back_float<float, float>"), .{
        .grid = block_nums,
        .block = block_dims,
        .stream = stream,
    }, .{ src0_d, src1_d, dst_d, ne00, ne10, ne1 });
}
