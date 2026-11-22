//! `GGML_OP_DIAG`: a vector spread onto the diagonal of a square matrix.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/diag.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/diag.cuh` at v1.0.0 (`b1b770d90`). The
//! kernel, `diag_kernel`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_DIAG_BLOCK_SIZE` (diag.cuh:3 @b1b770d90).
const block_size = 256;

/// Ports `ggml_cuda_op_diag` (diag.cu:36 @b1b770d90), which launches with
/// `<<<>>>`.
pub fn opDiag(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);

    const dst_d: ?*anyopaque = dst.data;
    const src0_d: ?*const anyopaque = src0.data;

    const stream = ctx.stream();

    ggml.impl.assert(ggml.impl.c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    ggml.impl.assert(ggml.impl.c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];

    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];

    ggml.impl.assert(ne00 == ne0, "ne00 == ne0");
    ggml.impl.assert(ne01 == 1, "ne01 == 1");
    ggml.impl.assert(ne02 == ne2, "ne02 == ne2");
    ggml.impl.assert(ne03 == ne3, "ne03 == ne3");

    const n_elems: i64 = ggml.impl.c.ggml_nelements(dst);
    const num_blocks: i64 = @divTrunc(n_elems + block_size - 1, block_size);
    const p: ggml.cuda.kernels.LaunchParams = .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))) },
        .block = .{ .x = block_size },
        .stream = stream,
    };

    switch (dst.type) {
        ggml.impl.c.GGML_TYPE_F32 => ggml.cuda.kernels.launch(ggml.cuda.kernels.get("diag", "diag_kernel<float>"), p, .{ dst_d, src0_d, ne0, ne1, ne2, ne3, n_elems }),
        ggml.impl.c.GGML_TYPE_F16 => ggml.cuda.kernels.launch(ggml.cuda.kernels.get("diag", "diag_kernel<__half>"), p, .{ dst_d, src0_d, ne0, ne1, ne2, ne3, n_elems }),
        else => ggml.impl.abort("unsupported type"),
    }
}
