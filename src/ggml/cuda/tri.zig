//! `GGML_OP_TRI`: keep the lower or upper triangle of each matrix, with or
//! without the diagonal, and zero the rest.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/tri.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/tri.cuh` at v0.4.1 (`b29c606e2`). The
//! kernel, `tri_kernel<T, prefix_keep, add_to_split>`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_TRI_BLOCK_SIZE` (tri.cuh:3 @b29c606e2).
const block_size = 256;

fn triKernel(comptime T: type, comptime prefix_keep: bool, comptime add_to_split: comptime_int) kernels.Kernel {
    return comptime kernels.get("tri", kernels.tmpl("tri_kernel", &.{ kernels.typeName(T), kernels.boolean(prefix_keep), kernels.int(add_to_split) }));
}

fn uint(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// `int64_t / size_t`: the division is unsigned, converted back to
/// `int64_t` at the parameter.
fn div(x: i64, d: usize) i64 {
    return @bitCast(@as(u64, @bitCast(x)) / d);
}

/// Ports `tri_cuda` (tri.cu:44 @b29c606e2), which launches with `<<<>>>`.
fn triCuda(
    comptime T: type,
    src: ?*const anyopaque,
    dst: ?*anyopaque,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: i64,
    nb01: i64,
    nb02: i64,
    nb03: i64,
    nb0: i64,
    nb1: i64,
    nb2: i64,
    nb3: i64,
    ttype: c.enum_ggml_tri_type,
    stream: cu.cudaStream_t,
) void {
    const block_dims: kernels.Dim3 = .{ .x = block_size };
    const grid_dims: kernels.Dim3 = .{ .x = uint(ne01), .y = uint(ne02), .z = uint(ne03) };
    const type_size: usize = @sizeOf(T);

    const add_to_split: c_int = if (ttype == c.GGML_TRI_TYPE_LOWER_DIAG or ttype == c.GGML_TRI_TYPE_UPPER) 1 else 0;
    const prefix_keep = (ttype == c.GGML_TRI_TYPE_LOWER or ttype == c.GGML_TRI_TYPE_LOWER_DIAG);

    const p: kernels.LaunchParams = .{ .grid = grid_dims, .block = block_dims, .stream = stream };
    const args = .{
        src,                  dst,
        ne00,                 ne01,
        ne02,                 ne03,
        div(nb00, type_size), div(nb01, type_size),
        div(nb02, type_size), div(nb03, type_size),
        div(nb0, type_size),  div(nb1, type_size),
        div(nb2, type_size),  div(nb3, type_size),
    };

    if (prefix_keep) {
        if (add_to_split == 0) {
            kernels.launch(triKernel(T, true, 0), p, args);
        } else { // only 0 and 1 supported
            kernels.launch(triKernel(T, true, 1), p, args);
        }
    } else {
        if (add_to_split == 0) {
            kernels.launch(triKernel(T, false, 0), p, args);
        } else {
            kernels.launch(triKernel(T, false, 1), p, args);
        }
    }
}

/// Ports `ggml_cuda_op_tri` (tri.cu:94 @b29c606e2).
pub fn opTri(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const stream = ctx.stream();

    const ttype: c.enum_ggml_tri_type = @intCast(impl.getOpParamsI32(dst, 0));

    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    const nb = [8]i64{
        @intCast(src0.nb[0]), @intCast(src0.nb[1]), @intCast(src0.nb[2]), @intCast(src0.nb[3]),
        @intCast(dst.nb[0]),  @intCast(dst.nb[1]),  @intCast(dst.nb[2]),  @intCast(dst.nb[3]),
    };

    switch (src0.type) {
        inline c.GGML_TYPE_F32, c.GGML_TYPE_F16, c.GGML_TYPE_BF16 => |t| {
            const T = switch (t) {
                c.GGML_TYPE_F32 => f32,
                c.GGML_TYPE_F16 => kernels.half,
                else => kernels.bf16,
            };
            triCuda(T, src0.data, dst.data, src0.ne[0], src0.ne[1], src0.ne[2], src0.ne[3], nb[0], nb[1], nb[2], nb[3], nb[4], nb[5], nb[6], nb[7], ttype, stream);
        },
        else => impl.abort("fatal error"),
    }
}
