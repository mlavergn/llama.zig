//! `GGML_OP_COUNT_EQUAL`: the number of equal elements of two tensors.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/count-equal.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/count-equal.cuh` at v1.0.0 (`b1b770d90`).
//! The kernel, `count_equal<int>`, stays CUDA.

const ggml = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `CUDA_COUNT_EQUAL_CHUNK_SIZE` (count-equal.cuh:3 @b1b770d90).
const chunk_size = 128;

/// Ports `ggml_cuda_count_equal` (count-equal.cu:28 @b1b770d90), which
/// launches with `<<<>>>`.
pub fn opCountEqual(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(src0.type == src1.type, "src0->type == src1->type");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_I64, " dst->type == GGML_TYPE_I64");

    ggml.impl.assert(ggml.types.ggml_are_same_shape(src0, src1), "ggml_are_same_shape(src0, src1)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(src1), "ggml_is_contiguous(src1)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const dst_d: ?*anyopaque = dst.data;

    const stream = ctx.stream();
    const nsm: c_int = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].nsm;

    const ne: i64 = ggml.types.ggml_nelements(src0);
    ggml.impl.assert(ne < (1 << 30), "ne < (1 << 30) && \"atomicAdd implementation only supports int\"");
    const dne: i64 = stdlib.num.pad(i64, @divTrunc(ne + 4 * nsm - 1, 4 * nsm), chunk_size);

    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemsetAsync(dst_d, 0, ggml.types.ggml_nbytes(dst), stream), "cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream)", @src());

    const blocks_dim: ggml.cuda.kernels.Dim3 = .{ .x = ggml.cuda.common.warp_size };
    const blocks_num: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(@min(@as(i64, 4) * nsm, @divTrunc(ne + chunk_size - 1, chunk_size))))) };

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_I32 => {
            const src0_d: ?*const anyopaque = src0.data;
            const src1_d: ?*const anyopaque = src1.data;
            ggml.cuda.kernels.launch(ggml.cuda.kernels.get("count-equal", "count_equal<int>"), .{ .grid = blocks_num, .block = blocks_dim, .stream = stream }, .{ src0_d, src1_d, dst_d, dne, ne });
        },
        else => ggml.impl.assert(false, "false"),
    }
}
