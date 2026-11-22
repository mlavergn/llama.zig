//! `GGML_OP_SET`: `dst = src0` with `src1` written over a strided view of
//! it, as two `ggml_cuda_cpy` calls.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/set.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/set.cuh` at v1.0.0 (`b1b770d90`). It has no
//! kernel of its own.

const ggml = @import("../module.zig");
/// `(size_t) ((int32_t *) dst->op_params)[i]`: sign-extended, then
/// reinterpreted.
fn opParamSize(dst: *const ggml.impl.ggml_c.ggml_tensor, i: usize) usize {
    return @bitCast(@as(i64, ggml.impl.getOpParamsI32(dst, i)));
}

/// Ports `ggml_cuda_op_set` (set.cu:4 @b1b770d90).
pub fn opSet(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert((src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_I32), "(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)");
    ggml.impl.assert(src1.type == src0.type, "src1->type == src0->type");
    ggml.impl.assert(dst.type == src0.type, "dst ->type == src0->type");

    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");

    const nb1 = opParamSize(dst, 0);
    const nb2 = opParamSize(dst, 1);
    const nb3 = opParamSize(dst, 2);
    const offset = opParamSize(dst, 3);
    const inplace = ggml.impl.getOpParamsI32(dst, 4) != 0;

    if (!inplace) {
        ggml.cuda.cpy.cpy(ctx, src0, dst);
    }

    var dst_view: ggml.impl.ggml_c.ggml_tensor = dst.*;
    dst_view.data = @ptrFromInt(@intFromPtr(dst.data) +% offset);
    dst_view.ne[0] = src1.ne[0];
    dst_view.ne[1] = src1.ne[1];
    dst_view.ne[2] = src1.ne[2];
    dst_view.ne[3] = src1.ne[3];

    dst_view.nb[0] = ggml.impl.ggml_c.ggml_element_size(dst);
    dst_view.nb[1] = nb1;
    dst_view.nb[2] = nb2;
    dst_view.nb[3] = nb3;

    ggml.cuda.cpy.cpy(ctx, src1, &dst_view);
}
