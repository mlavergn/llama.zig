//! The Fast Walsh-Hadamard transform. There is no `GGML_OP_FWHT`:
//! `ggml_cuda_mul_mat` calls `opFwht` when `src0` carries the
//! `GGML_HINT_SRC0_IS_HADAMARD` hint, and falls back to a matmul when it
//! returns false.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/fwht.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/fwht.cuh` at v0.4.0 (`5266f24da`). The
//! kernel, `fwht_cuda`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_op_fwht` (fwht.cu:61 @5266f24da).
///
/// Return: whether the transform could be used -- false for a
/// non-contiguous operand or a row length other than 64, 128, 256 or 512,
/// with nothing launched.
pub fn opFwht(ctx: *common.Context, src: *const Tensor, dst: *Tensor) bool {
    impl.assert(c.ggml_are_same_shape(src, dst), "ggml_are_same_shape(src, dst)");
    if (!c.ggml_is_contiguous(src) or !c.ggml_is_contiguous(dst)) {
        return false;
    }
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
            kernels.launchPdl(kernels.get("fwht", kernels.tmpl("fwht_cuda", &.{kernels.int(N)})), p, .{ src_d, dst_d, rows, scale });
            return true;
        },
        else => return false,
    }
}
