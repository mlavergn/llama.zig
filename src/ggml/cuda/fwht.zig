//! The Fast Walsh-Hadamard transform. There is no `GGML_OP_FWHT`:
//! `ggml_cuda_mul_mat` calls `opFwht` when `src0` carries the
//! `GGML_HINT_SRC0_IS_HADAMARD` hint, and falls back to a matmul when it
//! returns false.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fwht.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/fwht.cuh` at v0.7.0 (`620d2225c`). The
//! kernels, `fwht_cuda` and (since v0.7.0) `fwht_cuda_block`, stay CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `GGML_CUDA_FWHT_BLOCK_NT` (fwht.cu:6 @620d2225c): wide FWHT blocks
/// use one row per thread block with this many threads.
const block_nt = 256;

/// Ports `ggml_cuda_op_fwht_impl` (fwht.cu:147 @620d2225c): the launch, for
/// an F32 or (since v0.6.0) F16 source `T`; the output is F32. v0.7.0 added
/// rows of 1024 to 8192 (the `!GGML_USE_MUSA` arm), one row per thread
/// block of `GGML_CUDA_FWHT_BLOCK_NT` threads.
fn opFwhtImpl(comptime T: type, ctx: *common.Context, src: *const Tensor, dst: *Tensor) bool {
    const n: c_int = @truncate(src.ne[0]);
    const rows = c.ggml_nrows(src);

    const src_d: ?*const anyopaque = src.data;
    const dst_d: ?*anyopaque = dst.data;

    const warp_size = common.info().devices[@intCast(common.getDevice())].warp_size;
    const rows_per_block = 4;

    const num_blocks = @divTrunc(rows + rows_per_block - 1, rows_per_block);

    const stream = ctx.stream();
    const grid_dims: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(num_blocks))), .y = 1, .z = 1 };
    const block_dims: kernels.Dim3 = .{ .x = @bitCast(warp_size), .y = rows_per_block, .z = 1 };
    const p: kernels.LaunchParams = .{ .grid = grid_dims, .block = block_dims, .shmem = 0, .stream = stream };

    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(n)));

    switch (n) {
        inline 64, 128, 256, 512 => |N| {
            kernels.launchPdl(kernels.get("fwht", kernels.tmpl("fwht_cuda", &.{ kernels.int(N), kernels.typeName(T) })), p, .{ src_d, dst_d, rows, scale });
            return true;
        },
        else => {},
    }

    // wide blocks: one row per thread block
    {
        const nt = block_nt;

        const grid_dims_w: kernels.Dim3 = .{ .x = @truncate(@as(u64, @bitCast(rows))), .y = 1, .z = 1 };
        const block_dims_w: kernels.Dim3 = .{ .x = nt, .y = 1, .z = 1 };
        const launch_params_w: kernels.LaunchParams = .{ .grid = grid_dims_w, .block = block_dims_w, .shmem = 0, .stream = stream };

        switch (n) {
            inline 1024, 2048, 4096, 8192 => |N| {
                kernels.launchPdl(kernels.get("fwht", kernels.tmpl("fwht_cuda_block", &.{ kernels.int(N), kernels.int(nt), kernels.typeName(T) })), launch_params_w, .{ src_d, dst_d, rows, scale });
                return true;
            },
            else => return false,
        }
    }
}

/// Ports `ggml_cuda_op_mul_mat_use_fwht` (fwht.cu:215 @620d2225c), new at
/// v0.6.0: whether a `MUL_MAT` is the Hadamard transform `opFwht` can take
/// -- the hint, F32 `src0` and output, and an F32 or F16 contiguous `src1`
/// of the output's shape. `mulmat.zig` asks it before `opFwht`, and
/// `supports.zig` to admit an F16 `src1` against a non-F16 `src0`.
pub fn opMulMatUseFwht(op: *const Tensor) bool {
    const a = impl.one(Tensor, op.src[0]);
    const b = impl.one(Tensor, op.src[1]);

    return op.op == c.GGML_OP_MUL_MAT and impl.getOpParamsI32(op, 1) == c.GGML_HINT_SRC0_IS_HADAMARD and
        a.type == c.GGML_TYPE_F32 and op.type == c.GGML_TYPE_F32 and
        (b.type == c.GGML_TYPE_F32 or b.type == c.GGML_TYPE_F16) and c.ggml_is_contiguous(b) and c.ggml_is_contiguous(op) and
        c.ggml_are_same_shape(b, op);
}

/// Ports `ggml_cuda_op_fwht` (fwht.cu:225 @620d2225c).
///
/// Return: whether the transform could be used -- false for a
/// non-contiguous operand, a non-F32 output, a source other than F32 or
/// F16, or a row length other than a power of two from 64 to 8192, with
/// nothing launched.
pub fn opFwht(ctx: *common.Context, src: *const Tensor, dst: *Tensor) bool {
    impl.assert(c.ggml_are_same_shape(src, dst), "ggml_are_same_shape(src, dst)");
    if (!c.ggml_is_contiguous(src) or !c.ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst.type != c.GGML_TYPE_F32) {
        return false;
    }

    return switch (src.type) {
        c.GGML_TYPE_F32 => opFwhtImpl(f32, ctx, src, dst),
        c.GGML_TYPE_F16 => opFwhtImpl(kernels.half, ctx, src, dst),
        else => false,
    };
}
