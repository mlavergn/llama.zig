//! The convolution family: im2col (2D and 3D), col2im, the transposed 1D
//! convolution, pooling, the direct 2D/3D and depthwise convolutions, plus
//! the timestep embedding and leaky_relu. Each is one `ggml_vk_op_f32`
//! call; what differs is the push constants.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.3.0
//! (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const m = @import("module.zig");

const c = impl.c;
const BackendCtx = m.context.BackendCtx;
const Context = m.submit.Context;
const Tensor = c.ggml_tensor;
const push = m.push;
const ops = m.ops;

/// Stores `v` into `pc.name` the way a C++ implicit conversion does: an
/// integer narrowed to the field's width, two's complement kept.
fn set(pc: anytype, comptime name: []const u8, v: i64) void {
    const F = @FieldType(@typeInfo(@TypeOf(pc)).pointer.child, name);
    const bits: u64 = @bitCast(v);
    @field(pc, name) = switch (F) {
        u32 => @truncate(bits),
        i32 => @bitCast(@as(u32, @truncate(bits))),
        u64 => bits,
        else => @compileError("unhandled push field type " ++ @typeName(F)),
    };
}

fn param(t: *const Tensor, i: usize) i64 {
    return t.op_params[i];
}

/// `dst`'s device address, as the im2col shaders write through it.
fn dstAddr(dst: *const Tensor) u64 {
    const buf = m.buffer.tensorBuffer(dst);
    return buf.bda_addr + m.buffer.tensorOffset(dst) + dst.view_offs;
}

/// Ports `ggml_vk_im2col` (ggml-vulkan.cpp:13968 @c1d0e7a00).
pub fn im2col(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const is_2D = dst.op_params[6] == 1;
    const IC = src1.ne[if (is_2D) 2 else 1];
    const IH: i64 = if (is_2D) src1.ne[1] else 1;
    const IW = src1.ne[0];
    const KH: i64 = if (is_2D) src0.ne[1] else 1;
    const KW = src0.ne[0];
    const OH: i64 = if (is_2D) dst.ne[2] else 1;
    const OW = dst.ne[1];
    // nb is a byte offset; src is float32
    const offset_delta: i64 = @intCast(src1.nb[if (is_2D) 2 else 1] / 4);
    const batch_offset: i64 = @intCast(src1.nb[if (is_2D) 3 else 2] / 4);
    const batch = src1.ne[if (is_2D) 3 else 2];

    // The C++ multiplies these as uint32_t, so they wrap at 32 bits.
    const u = ops.u32Of;
    var pc = std.mem.zeroes(push.Im2col);
    pc.dst_addr = dstAddr(dst);
    set(&pc, "batch_offset", batch_offset);
    set(&pc, "offset_delta", offset_delta);
    set(&pc, "IC", IC);
    set(&pc, "IW", IW);
    set(&pc, "IH", IH);
    set(&pc, "OW", OW);
    set(&pc, "OH", OH);
    set(&pc, "KW", KW);
    set(&pc, "KH", KH);
    pc.OH_batch = u(OH) *% u(batch);
    pc.CHW = u(IC) *% u(KH) *% u(KW);
    set(&pc, "s0", param(dst, 0));
    set(&pc, "s1", param(dst, 1));
    set(&pc, "p0", param(dst, 2));
    set(&pc, "p1", param(dst, 3));
    set(&pc, "d0", param(dst, 4));
    set(&pc, "d1", param(dst, 5));
    pc.batch_IC = u(batch) *% u(IC);
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_IM2COL, pc);
}

/// Ports `ggml_vk_im2col_3d` (ggml-vulkan.cpp:14008 @c1d0e7a00).
pub fn im2col3d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    const IC = param(dst, 9);
    const N = @divTrunc(src1.ne[3], IC);
    const ID = src1.ne[2];
    const IH = src1.ne[1];
    const IW = src1.ne[0];
    const KD = src0.ne[2];
    const KH = src0.ne[1];
    const KW = src0.ne[0];
    const OD = @divTrunc(dst.ne[3], N);
    const OH = dst.ne[2];
    const OW = dst.ne[1];
    const ts: u64 = c.ggml_type_size(src1.type);

    var pc: push.Im2col3d = std.mem.zeroes(push.Im2col3d);
    pc.dst_addr = dstAddr(dst);
    set(&pc, "nb10", @intCast(src1.nb[0] / ts));
    set(&pc, "nb11", @intCast(src1.nb[1] / ts));
    set(&pc, "nb12", @intCast(src1.nb[2] / ts));
    set(&pc, "nb13", @intCast(src1.nb[3] / ts));
    set(&pc, "s0", param(dst, 0));
    set(&pc, "s1", param(dst, 1));
    set(&pc, "s2", param(dst, 2));
    set(&pc, "p0", param(dst, 3));
    set(&pc, "p1", param(dst, 4));
    set(&pc, "p2", param(dst, 5));
    set(&pc, "d0", param(dst, 6));
    set(&pc, "d1", param(dst, 7));
    set(&pc, "d2", param(dst, 8));
    set(&pc, "IW", IW);
    set(&pc, "IH", IH);
    set(&pc, "ID", ID);
    set(&pc, "IC", IC);
    set(&pc, "KW", KW);
    set(&pc, "OH", OH);
    // int64 products in the C++, narrowed on store.
    set(&pc, "KD_KH_KW", KD *% KH *% KW);
    set(&pc, "KH_KW", KH *% KW);
    set(&pc, "IC_KD_KH_KW", IC *% KD *% KH *% KW);
    set(&pc, "N_OD_OH", N *% OD *% OH);
    set(&pc, "OD_OH", OD *% OH);
    set(&pc, "OD_OH_OW_IC_KD_KH_KW", OD *% OH *% OW *% IC *% KD *% KH *% KW);
    set(&pc, "OH_OW_IC_KD_KH_KW", OH *% OW *% IC *% KD *% KH *% KW);
    set(&pc, "OW_IC_KD_KH_KW", OW *% IC *% KD *% KH *% KW);
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_IM2COL_3D, pc);
}

/// Ports `ggml_vk_timestep_embedding` (ggml-vulkan.cpp:14074 @c1d0e7a00).
pub fn timestepEmbedding(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    var pc = std.mem.zeroes(push.TimestepEmbedding);
    set(&pc, "nb1", @intCast(dst.nb[1] / c.ggml_type_size(dst.type)));
    set(&pc, "dim", param(dst, 0));
    set(&pc, "max_period", param(dst, 1));
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_TIMESTEP_EMBEDDING, pc);
}

/// Ports `ggml_vk_conv_transpose_1d` (ggml-vulkan.cpp:14084 @c1d0e7a00).
/// src0 is the kernel (K, Cout, Cin), src1 the input (L, Cin).
pub fn convTranspose1d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(src0.nb[0] == @sizeOf(f32), "nb00 == sizeof(float)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "nb10 == sizeof(float)");

    var p = std.mem.zeroes(push.ConvTranspose1d);
    set(&p, "Cout", src0.ne[1]);
    set(&p, "Cin", src0.ne[2]);
    set(&p, "K", src0.ne[0]);
    set(&p, "L", src1.ne[0]);
    set(&p, "KL", dst.ne[0]);
    set(&p, "nb01", @intCast(src0.nb[1] / src0.nb[0]));
    set(&p, "nb02", @intCast(src0.nb[2] / src0.nb[0]));
    set(&p, "nb11", @intCast(src1.nb[1] / src1.nb[0]));
    set(&p, "nb1", @intCast(dst.nb[1] / dst.nb[0]));
    set(&p, "s0", param(dst, 0));
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_CONV_TRANSPOSE_1D, p);
}

/// Ports `ggml_vk_col2im_1d` (ggml-vulkan.cpp:14115 @c1d0e7a00): src0 is
/// [K_OC, T_in] columns from a matmul, dst [T_out, OC].
pub fn col2im1d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const K_OC = ops.u32Of(src0.ne[0]);
    const OC: u32 = @bitCast(dst.op_params[1]);
    var p = std.mem.zeroes(push.Col2im1d);
    p.T_out = ops.u32Of(dst.ne[0]);
    p.OC = OC;
    p.K_OC = K_OC;
    p.T_in = ops.u32Of(src0.ne[1]);
    p.K = K_OC / OC;
    set(&p, "stride", param(dst, 0));
    set(&p, "p0", param(dst, 2));
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_COL2IM_1D, p);
}

/// Ports `ggml_vk_pool_1d` (ggml-vulkan.cpp:14180 @c1d0e7a00).
pub fn pool1d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const u = ops.u32Of;
    const N = u(dst.ne[3] *% dst.ne[2]);
    const OC = u(dst.ne[1]);
    const OL = u(dst.ne[0]);
    var p = std.mem.zeroes(push.Pool1d);
    p.IL = u(src0.ne[0]);
    p.OL = OL;
    p.OC = OC;
    p.pelements = N *% OC *% OL;
    set(&p, "op", param(dst, 0));
    set(&p, "k0", param(dst, 1));
    set(&p, "s0", param(dst, 2));
    set(&p, "p0", param(dst, 3));
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_POOL_1D, p);
}

/// Ports `ggml_vk_pool_2d` (ggml-vulkan.cpp:14203 @c1d0e7a00).
pub fn pool2d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    const u = ops.u32Of;
    const N = u(dst.ne[3]);
    const OC = u(dst.ne[2]);
    const OH = u(dst.ne[1]);
    const OW = u(dst.ne[0]);
    var p = std.mem.zeroes(push.Pool2d);
    p.IW = u(src0.ne[0]);
    p.IH = u(src0.ne[1]);
    p.OW = OW;
    p.OH = OH;
    p.OC = OC;
    p.pelements = N *% OC *% OH *% OW;
    set(&p, "op", param(dst, 0));
    set(&p, "k0", param(dst, 2));
    set(&p, "k1", param(dst, 1));
    set(&p, "s0", param(dst, 4));
    set(&p, "s1", param(dst, 3));
    set(&p, "p0", param(dst, 6));
    set(&p, "p1", param(dst, 5));
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_POOL_2D, p);
}

fn convAsserts(src0: *const Tensor, src1: *const Tensor, dst: *const Tensor) void {
    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(src0.nb[0] == @sizeOf(f32) or src0.nb[0] == @sizeOf(c.ggml_fp16_t), "nb00 == sizeof(float) || nb00 == sizeof(ggml_fp16_t)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "nb10 == sizeof(float)");
    impl.assert(dst.nb[0] == @sizeOf(f32), "nb0 == sizeof(float)");
}

/// The nine element strides both direct convolutions pass.
fn convStrides(p: anytype, src0: *const Tensor, src1: *const Tensor, dst: *const Tensor) void {
    set(p, "nb01", @intCast(src0.nb[1] / src0.nb[0]));
    set(p, "nb02", @intCast(src0.nb[2] / src0.nb[0]));
    set(p, "nb03", @intCast(src0.nb[3] / src0.nb[0]));
    set(p, "nb11", @intCast(src1.nb[1] / src1.nb[0]));
    set(p, "nb12", @intCast(src1.nb[2] / src1.nb[0]));
    set(p, "nb13", @intCast(src1.nb[3] / src1.nb[0]));
    set(p, "nb1", @intCast(dst.nb[1] / dst.nb[0]));
    set(p, "nb2", @intCast(dst.nb[2] / dst.nb[0]));
    set(p, "nb3", @intCast(dst.nb[3] / dst.nb[0]));
}

/// Ports `ggml_vk_conv_2d` (ggml-vulkan.cpp:14231 @c1d0e7a00), forward and
/// transposed.
pub fn conv2d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    convAsserts(src0, src1, dst);
    const transpose = dst.op == c.GGML_OP_CONV_TRANSPOSE_2D;

    var p = std.mem.zeroes(push.Conv2d);
    set(&p, "Cout", if (!transpose) src0.ne[3] else src0.ne[2]);
    set(&p, "Cin", if (!transpose) src0.ne[2] else src0.ne[3]);
    set(&p, "N", src1.ne[3]);
    impl.assert(p.Cout == dst.ne[2], "p.Cout == ne2");
    impl.assert(p.Cin == src1.ne[2], "p.Cin == ne12");

    set(&p, "W", src1.ne[0]);
    set(&p, "H", src1.ne[1]);
    set(&p, "OW", dst.ne[0]);
    set(&p, "OH", dst.ne[1]);
    convStrides(&p, src0, src1, dst);
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, dst.op, p);
}

/// Ports `ggml_vk_conv_3d` (ggml-vulkan.cpp:14271 @c1d0e7a00).
pub fn conv3d(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    convAsserts(src0, src1, dst);

    var p = std.mem.zeroes(push.Conv3d);
    set(&p, "IC", param(dst, 9));
    set(&p, "N", param(dst, 10));
    set(&p, "OC", param(dst, 11));
    impl.assert(src0.ne[3] == @as(i64, p.IC) * p.OC, "src0->ne[3] == (int64_t)p.IC * p.OC");
    impl.assert(src1.ne[3] == @as(i64, p.IC) * p.N, "src1->ne[3] == (int64_t)p.IC * p.N");
    impl.assert(dst.ne[3] == @as(i64, p.OC) * p.N, "dst->ne[3] == (int64_t)p.OC * p.N");

    set(&p, "IW", src1.ne[0]);
    set(&p, "IH", src1.ne[1]);
    set(&p, "ID", src1.ne[2]);
    set(&p, "OW", dst.ne[0]);
    set(&p, "OH", dst.ne[1]);
    set(&p, "OD", dst.ne[2]);

    // The shader clamps source addresses in uint32, so the input element
    // count must fit in one.
    impl.assert(@as(u64, p.IC) *% p.N *% p.IW *% p.IH *% p.ID <= 0xFFFFFFFF, "(uint64_t)p.IC * p.N * p.IW * p.IH * p.ID <= 0xFFFFFFFFull");

    convStrides(&p, src0, src1, dst);
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_CONV_3D, p);
}

/// Ports `ggml_vk_conv_2d_dw` (ggml-vulkan.cpp:14316 @c1d0e7a00).
pub fn conv2dDw(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, src1: *const Tensor, dst: *Tensor) void {
    var p = std.mem.zeroes(push.Conv2dDw);
    set(&p, "ne", c.ggml_nelements(dst));
    set(&p, "channels", dst.ne[2]);
    set(&p, "batches", dst.ne[3]);
    set(&p, "dst_w", dst.ne[0]);
    set(&p, "dst_h", dst.ne[1]);
    set(&p, "src_w", src1.ne[0]);
    set(&p, "src_h", src1.ne[1]);
    set(&p, "knl_w", src0.ne[0]);
    set(&p, "knl_h", src0.ne[1]);
    set(&p, "stride_x", param(dst, 0));
    set(&p, "stride_y", param(dst, 1));
    set(&p, "pad_x", param(dst, 2));
    set(&p, "pad_y", param(dst, 3));
    set(&p, "dilation_x", param(dst, 4));
    set(&p, "dilation_y", param(dst, 5));
    impl.assert(src0.ne[3] == p.channels, "src0->ne[3] == p.channels");
    impl.assert(src1.ne[3] == p.batches, "src1->ne[3] == p.batches");
    ops.opF32(ctx, subctx, src0, src1, null, null, dst, c.GGML_OP_CONV_2D_DW, p);
}

/// Ports `ggml_vk_leaky_relu` (ggml-vulkan.cpp:14340 @c1d0e7a00).
pub fn leakyRelu(ctx: *BackendCtx, subctx: *Context, src0: *const Tensor, dst: *Tensor) void {
    var p = ops.unaryPushConstantsInit(src0, dst, 0);
    p.param1 = @bitCast(dst.op_params[0]);
    ops.opF32(ctx, subctx, src0, null, null, null, dst, c.GGML_OP_LEAKY_RELU, p);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
