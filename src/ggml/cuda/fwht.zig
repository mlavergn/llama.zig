//! The Fast Walsh-Hadamard transform. There is no `GGML_OP_FWHT`:
//! `ggml_cuda_mul_mat` calls `opFwht` when `src0` carries the
//! `GGML_HINT_SRC0_IS_HADAMARD` hint, and falls back to a matmul when it
//! returns false.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fwht.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/fwht.cuh` at v1.0.0 (`b1b770d90`). The
//! kernels, `fwht_cuda` and `fwht_cuda_block`, stay CUDA.

const ggml = @import("../module.zig");
/// Ports `GGML_CUDA_FWHT_BLOCK_NT` (fwht.cu:6 @b1b770d90): wide FWHT blocks
/// use one row per thread block with this many threads.
const block_nt = 256;

/// Ports `ggml_cuda_op_fwht_impl` (fwht.cu:147 @b1b770d90): the launch, for
/// an F32 or F16 source `T`; the output is F32. Rows of 1024 to 8192 are taken
/// too (the `!GGML_USE_MUSA` arm), one row per thread
/// block of `GGML_CUDA_FWHT_BLOCK_NT` threads.
fn opFwhtImpl(comptime T: type, ctx: *ggml.cuda.common.Context, src: *const ggml.impl.c.ggml_tensor, dst: *ggml.impl.c.ggml_tensor) bool {
    const n: c_int = @truncate(src.ne[0]);
    const rows = ggml.impl.c.ggml_nrows(src);

    const src_d: ?*const anyopaque = src.data;
    const dst_d: ?*anyopaque = dst.data;

    const warp_size = ggml.cuda.common.info().devices[@intCast(ggml.cuda.common.getDevice())].warp_size;
    const rows_per_block = 4;

    const num_blocks = @divTrunc(rows + rows_per_block - 1, rows_per_block);

    const stream = ctx.stream();
    const grid_dims: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))), .y = 1, .z = 1 };
    const block_dims: ggml.cuda.kernels.Dim3 = .{ .x = @bitCast(warp_size), .y = rows_per_block, .z = 1 };
    const p: ggml.cuda.kernels.LaunchParams = .{ .grid = grid_dims, .block = block_dims, .shmem = 0, .stream = stream };

    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(n)));

    switch (n) {
        inline 64, 128, 256, 512 => |N| {
            ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("fwht", ggml.cuda.kernels.tmpl("fwht_cuda", &.{ ggml.cuda.kernels.int(N), ggml.cuda.kernels.typeName(T) })), p, .{ src_d, dst_d, rows, scale });
            return true;
        },
        else => {},
    }

    // wide blocks: one row per thread block
    {
        const nt = block_nt;

        const grid_dims_w: ggml.cuda.kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(rows))), .y = 1, .z = 1 };
        const block_dims_w: ggml.cuda.kernels.Dim3 = .{ .x = nt, .y = 1, .z = 1 };
        const launch_params_w: ggml.cuda.kernels.LaunchParams = .{ .grid = grid_dims_w, .block = block_dims_w, .shmem = 0, .stream = stream };

        switch (n) {
            inline 1024, 2048, 4096, 8192 => |N| {
                ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("fwht", ggml.cuda.kernels.tmpl("fwht_cuda_block", &.{ ggml.cuda.kernels.int(N), ggml.cuda.kernels.int(nt), ggml.cuda.kernels.typeName(T) })), launch_params_w, .{ src_d, dst_d, rows, scale });
                return true;
            },
            else => return false,
        }
    }
}

/// Ports `ggml_cuda_op_mul_mat_use_fwht` (fwht.cu:215 @b1b770d90): whether a `MUL_MAT` is the Hadamard transform `opFwht` can take
/// -- the hint, F32 `src0` and output, and an F32 or F16 contiguous `src1`
/// of the output's shape. `mulmat.zig` asks it before `opFwht`, and
/// `supports.zig` to admit an F16 `src1` against a non-F16 `src0`.
pub fn opMulMatUseFwht(op: *const ggml.impl.c.ggml_tensor) bool {
    const a = ggml.impl.one(ggml.impl.c.ggml_tensor, op.src[0]);
    const b = ggml.impl.one(ggml.impl.c.ggml_tensor, op.src[1]);

    return op.op == ggml.impl.c.GGML_OP_MUL_MAT and ggml.impl.getOpParamsI32(op, 1) == ggml.impl.c.GGML_HINT_SRC0_IS_HADAMARD and
        a.type == ggml.impl.c.GGML_TYPE_F32 and op.type == ggml.impl.c.GGML_TYPE_F32 and
        (b.type == ggml.impl.c.GGML_TYPE_F32 or b.type == ggml.impl.c.GGML_TYPE_F16) and ggml.impl.c.ggml_is_contiguous(b) and ggml.impl.c.ggml_is_contiguous(op) and
        ggml.impl.c.ggml_are_same_shape(b, op);
}

/// Ports `ggml_cuda_op_fwht` (fwht.cu:225 @b1b770d90).
///
/// Return: whether the transform could be used -- false for a
/// non-contiguous operand, a non-F32 output, a source other than F32 or
/// F16, or a row length other than a power of two from 64 to 8192, with
/// nothing launched.
pub fn opFwht(ctx: *ggml.cuda.common.Context, src: *const ggml.impl.c.ggml_tensor, dst: *ggml.impl.c.ggml_tensor) bool {
    ggml.impl.assert(ggml.impl.c.ggml_are_same_shape(src, dst), "ggml_are_same_shape(src, dst)");
    if (!ggml.impl.c.ggml_is_contiguous(src) or !ggml.impl.c.ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst.type != ggml.impl.c.GGML_TYPE_F32) {
        return false;
    }

    return switch (src.type) {
        ggml.impl.c.GGML_TYPE_F32 => opFwhtImpl(f32, ctx, src, dst),
        ggml.impl.c.GGML_TYPE_F16 => opFwhtImpl(ggml.cuda.kernels.half, ctx, src, dst),
        else => false,
    };
}
