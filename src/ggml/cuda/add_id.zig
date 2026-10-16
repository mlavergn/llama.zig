//! `GGML_OP_ADD_ID`: each row of `src0` plus the `src1` row `src2` selects
//! for it (the MoE expert bias).
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/add-id.cu` at v0.3.0
//! (`c1d0e7a00`). The kernel, `add_id_kernel`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_op_add_id` (add-id.cu:28 @c1d0e7a00), which launches with
/// `<<<>>>`.
pub fn opAddId(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const src2 = impl.one(Tensor, dst.src[2]);

    // GGML_TENSOR_TERNARY_OP_LOCALS
    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const nb00 = src0.nb[0];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb10 = src1.nb[0];
    const nb11 = src1.nb[1];
    const nb20 = src2.nb[0];
    const nb21 = src2.nb[1];
    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];

    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");
    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(src2.type == c.GGML_TYPE_I32, "src2->type == GGML_TYPE_I32");

    impl.assert(nb00 == @sizeOf(f32), "nb00 == sizeof(float)");
    impl.assert(nb10 == @sizeOf(f32), "nb10 == sizeof(float)");
    impl.assert(nb20 == @sizeOf(i32), "nb20 == sizeof(int32_t)");

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;
    const src2_d: ?*const anyopaque = src2.data;
    const dst_d: ?*anyopaque = dst.data;

    const threads: c_int = @min(@as(c_int, @truncate(ne00)), 768); // cols
    const blocks: kernels.Dim3 = .{ // n_experts_used, n_tokens
        .x = @truncate(@as(u64, @bitCast(ne01))),
        .y = @truncate(@as(u64, @bitCast(ne02))),
    };
    kernels.launch(kernels.get("add-id", "add_id_kernel"), .{ .grid = blocks, .block = .{ .x = @bitCast(threads) }, .stream = ctx.stream() }, .{
        src0_d, src1_d, src2_d, dst_d,
        ne0,    ne1,    nb01,   nb02,
        nb11,   nb21,
    });
}
