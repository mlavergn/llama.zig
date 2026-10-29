//! `GGML_OP_SET`: `dst = src0` with `src1` written over a strided view of
//! it, as two `ggml_cuda_cpy` calls.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/set.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/set.cuh` at v0.4.1 (`b29c606e2`). It has no
//! kernel of its own.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const cpy = @import("cpy.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// `(size_t) ((int32_t *) dst->op_params)[i]`: sign-extended, then
/// reinterpreted.
fn opParamSize(dst: *const Tensor, i: usize) usize {
    return @bitCast(@as(i64, impl.getOpParamsI32(dst, i)));
}

/// Ports `ggml_cuda_op_set` (set.cu:4 @b29c606e2).
pub fn opSet(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert((src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_I32), "(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)");
    impl.assert(src1.type == src0.type, "src1->type == src0->type");
    impl.assert(dst.type == src0.type, "dst ->type == src0->type");

    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");

    const nb1 = opParamSize(dst, 0);
    const nb2 = opParamSize(dst, 1);
    const nb3 = opParamSize(dst, 2);
    const offset = opParamSize(dst, 3);
    const inplace = impl.getOpParamsI32(dst, 4) != 0;

    if (!inplace) {
        cpy.cpy(ctx, src0, dst);
    }

    var dst_view: Tensor = dst.*;
    dst_view.data = @ptrFromInt(@intFromPtr(dst.data) +% offset);
    dst_view.ne[0] = src1.ne[0];
    dst_view.ne[1] = src1.ne[1];
    dst_view.ne[2] = src1.ne[2];
    dst_view.ne[3] = src1.ne[3];

    dst_view.nb[0] = c.ggml_element_size(dst);
    dst_view.nb[1] = nb1;
    dst_view.nb[2] = nb2;
    dst_view.nb[3] = nb3;

    cpy.cpy(ctx, src1, &dst_view);
}
