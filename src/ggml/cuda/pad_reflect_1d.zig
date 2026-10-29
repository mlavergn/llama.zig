//! `GGML_OP_PAD_REFLECT_1D`: reflection padding along dimension 0.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/pad_reflect_1d.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/pad_reflect_1d.cuh` at v0.4.1
//! (`b29c606e2`). The kernel, `pad_reflect_1d_kernel_f32`, stays CUDA.
//!
//! The grid's x covers rows and column tiles together, `ne01 * tiles0`; the
//! kernel splits it back with `fast_div_modulo` over `ne01`'s fastdiv
//! values.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_PAD_REFLECT_1D_BLOCK_SIZE` (pad_reflect_1d.cuh:3 @b29c606e2).
const block_size = 256;

fn uint(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `ggml_cuda_op_pad_reflect_1d` (pad_reflect_1d.cu:58 @b29c606e2),
/// which launches with `<<<>>>`.
pub fn opPadReflect1d(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const p0: c_int = dst.op_params[0];
    const p1: c_int = dst.op_params[1];

    const ne00: i64 = src0.ne[0];
    const ne01: i64 = src0.ne[1];
    const ne01_packed = common.initFastdivValues(@bitCast(ne01));
    const ne02: i64 = src0.ne[2];
    const ne03: i64 = src0.ne[3];

    const ne0: i64 = dst.ne[0];

    // sanity: padded length matches
    impl.assert(ne0 == ne00 + p0 + p1, "ne0 == ne00 + p0 + p1");

    const bx: i64 = block_size; // threads per block (x)
    const tiles0: i64 = @divTrunc(ne0 + bx - 1, bx); // number of tiles along i0
    // grid.x covers i1 and all tiles of i0: [ne01 * tiles0]
    // grid.y covers i2: [ne02]
    // grid.z covers i3: [ne03]
    const grid_dims: kernels.Dim3 = .{ .x = uint(ne01 * tiles0), .y = uint(ne02), .z = uint(ne03) };
    const block_dims: kernels.Dim3 = .{ .x = uint(bx), .y = 1, .z = 1 };

    const nb = [8]i64{
        @bitCast(src0.nb[0]), @bitCast(src0.nb[1]), @bitCast(src0.nb[2]), @bitCast(src0.nb[3]),
        @bitCast(dst.nb[0]),  @bitCast(dst.nb[1]),  @bitCast(dst.nb[2]),  @bitCast(dst.nb[3]),
    };

    const src0_d: ?*const anyopaque = src0.data;
    const dst_d: ?*anyopaque = dst.data;
    kernels.launch(kernels.get("pad_reflect_1d", "pad_reflect_1d_kernel_f32"), .{ .grid = grid_dims, .block = block_dims, .stream = stream }, .{
        src0_d, dst_d, ne0,   ne00,  ne01_packed, ne02,  ne03,
        nb[0],  nb[1], nb[2], nb[3], nb[4],       nb[5], nb[6],
        nb[7],  p0,    p1,
    });
}
