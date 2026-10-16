//! `GGML_OP_SET_ROWS`: scatter rows of `src0` into `dst` at the indices in
//! `src1`, converting or quantizing on the way.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/set-rows.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/set-rows.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels, `k_set_rows` and `k_set_rows_quant`, stay CUDA.
//!
//! The fused `ROPE -> VIEW -> SET_ROWS` path is not here: upstream writes
//! the rows from inside `rope.cu`'s kernel, and `set-rows.cu` has no entry
//! point fusion calls.
//!
//! # Launches
//!
//! `k_set_rows` goes through `ggml_cuda_kernel_launch` (PDL);
//! `k_set_rows_quant` through `<<<>>>`. Both are upstream's choice.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;
const bf16 = kernels.bf16;

/// Ports `CUDA_SET_ROWS_BLOCK_SIZE` (set-rows.cuh:5 @c1d0e7a00).
const block_size = 256;

/// The sizes and byte strides both launchers take, in the C's order.
const Args = struct {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    ne10: i64,
    ne11: i64,
    ne12: i64,
    ne13: i64,
    nb01: usize,
    nb02: usize,
    nb03: usize,
    nb10: usize,
    nb11: usize,
    nb12: usize,
    nb1: usize,
    nb2: usize,
    nb3: usize,
};

/// `(uint32_t) x`.
fn u32Of(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// `int64_t s = nb / sizeof(T)`: a `size_t` quotient stored in `int64_t`.
fn strideOf(nb: usize, comptime size: usize) i64 {
    return @bitCast(@as(u64, nb / size));
}

/// Ports the quantized `set_rows_cuda_quant` (set-rows.cu:75 @c1d0e7a00).
/// `block` and `quantize` name the template's `block_type` and
/// `quantize_func`.
fn setRowsCudaQuant(comptime I: type, comptime block: []const u8, comptime qk: comptime_int, comptime quantize: []const u8, src0_d: ?*const anyopaque, src1_d: ?*const anyopaque, dst_d: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    impl.assert(@rem(a.ne00, qk) == 0, "ne00 % qk == 0");
    const ne_total: i64 = @divTrunc(a.ne00 * a.ne01 * a.ne02 * a.ne03, qk);
    const num_blocks: c_int = @truncate(@divTrunc(ne_total + block_size - 1, block_size));
    const block_dims: kernels.Dim3 = .{ .x = block_size };
    const grid_size: kernels.Dim3 = .{ .x = @bitCast(num_blocks) };

    const s01 = strideOf(a.nb01, @sizeOf(f32));
    const s02 = strideOf(a.nb02, @sizeOf(f32));
    const s03 = strideOf(a.nb03, @sizeOf(f32));
    const s10 = strideOf(a.nb10, @sizeOf(I));
    const s11 = strideOf(a.nb11, @sizeOf(I));
    const s12 = strideOf(a.nb12, @sizeOf(I));
    const s1: i64 = @bitCast(@as(u64, a.nb1));
    const s2: i64 = @bitCast(@as(u64, a.nb2));
    const s3: i64 = @bitCast(@as(u64, a.nb3));

    if (ne_total > 0 and a.ne00 > 0 and a.ne01 > 0 and a.ne02 > 0 and a.ne11 > 0 and a.ne12 > 0) {
        const ne00_fd = common.initFastdivValues(u32Of(a.ne00));
        const ne01_fd = common.initFastdivValues(u32Of(a.ne01));
        const ne02_fd = common.initFastdivValues(u32Of(a.ne02));
        const ne11_fd = common.initFastdivValues(u32Of(a.ne11));
        const ne12_fd = common.initFastdivValues(u32Of(a.ne12));

        const key = comptime kernels.tmpl("k_set_rows_quant", &.{ kernels.typeName(I), block, kernels.int(qk), "&" ++ quantize });
        kernels.launch(kernels.get("set-rows", key), .{
            .grid = grid_size,
            .block = block_dims,
            .stream = stream,
        }, .{
            src0_d, src1_d,  dst_d,   ne_total, a.ne10,  a.ne11,  a.ne12, a.ne13,
            s01,    s02,     s03,     s10,      s11,     s12,     s1,     s2,
            s3,     ne00_fd, ne01_fd, ne02_fd,  ne11_fd, ne12_fd,
        });
    }
}

/// Ports the element-wise `set_rows_cuda` (set-rows.cu:180 @c1d0e7a00).
fn setRowsCudaLaunch(comptime S: type, comptime I: type, comptime D: type, src0_d: ?*const anyopaque, src1_d: ?*const anyopaque, dst_d: ?*anyopaque, a: Args, stream: cu.cudaStream_t) void {
    const ne_total: i64 = a.ne00 * a.ne01 * a.ne02 * a.ne03;
    const num_blocks: c_int = @truncate(@divTrunc(ne_total + block_size - 1, block_size));
    const block_dims: kernels.Dim3 = .{ .x = block_size };
    const grid_size: kernels.Dim3 = .{ .x = @bitCast(num_blocks) };

    const s01 = strideOf(a.nb01, @sizeOf(S));
    const s02 = strideOf(a.nb02, @sizeOf(S));
    const s03 = strideOf(a.nb03, @sizeOf(S));
    const s10 = strideOf(a.nb10, @sizeOf(I));
    const s11 = strideOf(a.nb11, @sizeOf(I));
    const s12 = strideOf(a.nb12, @sizeOf(I));
    const s1 = strideOf(a.nb1, @sizeOf(D));
    const s2 = strideOf(a.nb2, @sizeOf(D));
    const s3 = strideOf(a.nb3, @sizeOf(D));

    if (ne_total > 0 and a.ne00 > 0 and a.ne01 > 0 and a.ne02 > 0 and a.ne11 > 0 and a.ne12 > 0) {
        const ne00_fd = common.initFastdivValues(u32Of(a.ne00));
        const ne01_fd = common.initFastdivValues(u32Of(a.ne01));
        const ne02_fd = common.initFastdivValues(u32Of(a.ne02));
        const ne11_fd = common.initFastdivValues(u32Of(a.ne11));
        const ne12_fd = common.initFastdivValues(u32Of(a.ne12));

        const key = comptime kernels.tmpl("k_set_rows", &.{ kernels.typeName(S), kernels.typeName(I), kernels.typeName(D) });
        kernels.launchPdl(kernels.get("set-rows", key), .{
            .grid = grid_size,
            .block = block_dims,
            .stream = stream,
        }, .{
            src0_d, src1_d,  dst_d,   ne_total, a.ne10,  a.ne11,  a.ne12, a.ne13,
            s01,    s02,     s03,     s10,      s11,     s12,     s1,     s2,
            s3,     ne00_fd, ne01_fd, ne02_fd,  ne11_fd, ne12_fd,
        });
    }
}

/// Ports the per-tensor `set_rows_cuda<src_t, idx_t>` (set-rows.cu:221
/// @c1d0e7a00) and its two `half` specialisations (set-rows.cu:326, 351
/// @c1d0e7a00), which accept only an `F16` destination.
fn setRowsCuda(comptime S: type, comptime I: type, ctx: *common.Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;

    const a: Args = .{
        .ne00 = src0.ne[0],
        .ne01 = src0.ne[1],
        .ne02 = src0.ne[2],
        .ne03 = src0.ne[3],
        .ne10 = src1.ne[0],
        .ne11 = src1.ne[1],
        .ne12 = src1.ne[2],
        .ne13 = src1.ne[3],
        .nb01 = src0.nb[1],
        .nb02 = src0.nb[2],
        .nb03 = src0.nb[3],
        .nb10 = src1.nb[0],
        .nb11 = src1.nb[1],
        .nb12 = src1.nb[2],
        .nb1 = dst.nb[1],
        .nb2 = dst.nb[2],
        .nb3 = dst.nb[3],
    };

    const stream = ctx.stream();
    const d = dst.data;

    if (S == half) {
        if (dst.type == c.GGML_TYPE_F16) {
            setRowsCudaLaunch(half, I, half, src0_d, src1_d, d, a, stream);
        } else {
            c.ggml_abort(@src().file.ptr, @intCast(@src().line), "unsupported type %s", c.ggml_type_name(dst.type));
        }
        return;
    }

    switch (dst.type) {
        c.GGML_TYPE_F32 => setRowsCudaLaunch(S, I, f32, src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_F16 => setRowsCudaLaunch(S, I, half, src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_BF16 => setRowsCudaLaunch(S, I, bf16, src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_Q4_0 => setRowsCudaQuant(I, "block_q4_0", 32, "quantize_f32_q4_0_block", src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_Q4_1 => setRowsCudaQuant(I, "block_q4_1", 32, "quantize_f32_q4_1_block", src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_Q5_0 => setRowsCudaQuant(I, "block_q5_0", 32, "quantize_f32_q5_0_block", src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_Q5_1 => setRowsCudaQuant(I, "block_q5_1", 32, "quantize_f32_q5_1_block", src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_Q8_0 => setRowsCudaQuant(I, "block_q8_0", 32, "quantize_f32_q8_0_block", src0_d, src1_d, d, a, stream),
        c.GGML_TYPE_IQ4_NL => setRowsCudaQuant(I, "block_iq4_nl", 32, "quantize_f32_iq4_nl_block", src0_d, src1_d, d, a, stream),
        else => c.ggml_abort(@src().file.ptr, @intCast(@src().line), "unsupported type %s", c.ggml_type_name(dst.type)),
    }
}

/// Ports `ggml_cuda_op_set_rows` (set-rows.cu:376 @c1d0e7a00).
pub fn opSetRows(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(src0.type == c.GGML_TYPE_F32 or (src0.type == c.GGML_TYPE_F16 and dst.type == c.GGML_TYPE_F16), "src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16)");
    impl.assert(src1.type == c.GGML_TYPE_I64 or src1.type == c.GGML_TYPE_I32, "src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32");

    if (src0.type == c.GGML_TYPE_F32) {
        if (src1.type == c.GGML_TYPE_I64) {
            setRowsCuda(f32, i64, ctx, src0, src1, dst);
        } else {
            setRowsCuda(f32, i32, ctx, src0, src1, dst);
        }
    } else if (src0.type == c.GGML_TYPE_F16) {
        if (src1.type == c.GGML_TYPE_I64) {
            setRowsCuda(half, i64, ctx, src0, src1, dst);
        } else {
            setRowsCuda(half, i32, ctx, src0, src1, dst);
        }
    } else {
        c.ggml_abort(@src().file.ptr, @intCast(@src().line), "unsupported type %s", c.ggml_type_name(src0.type));
    }
}
