//! `GGML_OP_TOP_K_MASK`: a row of `-inf` with the mask cells
//! of the live top-k picks, and of any extra cells, copied through. The
//! sparse-attention (QSA/DSA) mask.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/top-k-mask.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/top-k-mask.cuh` at v1.0.0 (`b1b770d90`).
//! The kernel, `top_k_mask_kernel`, stays CUDA.

const ggml = @import("../module.zig");
/// Ports `CUDA_TOP_K_MASK_BLOCK_SIZE` (top-k-mask.cu:4 @b1b770d90).
const block_size = 512;

/// `sizeof(int4)`: the kernel's vectorized store.
const sizeof_int4 = 16;

/// A byte stride in elements, into an `int64_t` parameter; zero for a
/// source the op does not have, as the C's `score ? score->nb[0]/sf : 0`.
fn s(t: ?*const ggml.impl.c.ggml_tensor, comptime i: usize, comptime size: usize) i64 {
    return if (t) |x| @bitCast(x.nb[i] / size) else 0;
}

/// `t ? (const T *) t->data : nullptr`.
fn data(t: ?*const ggml.impl.c.ggml_tensor) ?*const anyopaque {
    return if (t) |x| x.data else null;
}

/// Ports `top_k_mask_cuda` (top-k-mask.cu:86 @b1b770d90), its element type
/// `T` comptime: `f32` or `kernels.half`.
fn topKMaskCuda(comptime T: type, ctx: *ggml.cuda.common.Context, dst: *ggml.impl.c.ggml_tensor) void {
    const mask = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const top_k = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);
    const score: ?*const ggml.impl.c.ggml_tensor = dst.src[2];
    const blocks: ?*const ggml.impl.c.ggml_tensor = dst.src[3];
    const extra: ?*const ggml.impl.c.ggml_tensor = dst.src[4];

    const n_kv = mask.ne[0];
    const ts = @sizeOf(T);

    ggml.impl.assert(dst.nb[0] == ts and mask.nb[0] == ts, "dst->nb[0] == sizeof(T) && mask->nb[0] == sizeof(T)");
    ggml.impl.assert(dst.nb[1] % ts == 0 and mask.nb[1] % ts == 0, "dst->nb[1] % sizeof(T) == 0 && mask->nb[1] % sizeof(T) == 0");

    // 16-byte stores need every dst row aligned
    const vec = (@as(u64, @bitCast(n_kv)) *% ts) % sizeof_int4 == 0 and
        dst.nb[1] % sizeof_int4 == 0 and dst.nb[2] % sizeof_int4 == 0 and dst.nb[3] % sizeof_int4 == 0 and
        @intFromPtr(dst.data) % sizeof_int4 == 0;

    const n_pick: i64 = if (blocks) |b| b.ne[1] else if (score) |sc| sc.ne[0] else n_kv;

    const si = @sizeOf(i32);
    const sf = @sizeOf(f32);

    ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("top-k-mask", ggml.cuda.kernels.tmpl("top_k_mask_kernel", &.{ggml.cuda.kernels.typeName(T)})), .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(ggml.impl.c.ggml_nrows(dst)))) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = ctx.stream(),
    }, .{
        @as(?*const anyopaque, mask.data),
        @as(?*const anyopaque, top_k.data),
        data(score),
        data(blocks),
        data(extra),
        @as(?*anyopaque, dst.data),
        n_kv,
        dst.ne[1],
        dst.ne[2],
        mask.ne[1],
        mask.ne[2],
        mask.ne[3],
        top_k.ne[0],
        n_pick,
        if (blocks) |b| b.ne[0] else @as(i64, 1),
        if (extra) |e| e.ne[0] else @as(i64, 0),
        s(mask, 1, ts),
        s(mask, 2, ts),
        s(mask, 3, ts),
        s(dst, 1, ts),
        s(dst, 2, ts),
        s(dst, 3, ts),
        s(top_k, 0, si),
        s(top_k, 1, si),
        s(top_k, 2, si),
        s(top_k, 3, si),
        s(score, 0, sf),
        s(score, 1, sf),
        s(score, 2, sf),
        s(score, 3, sf),
        s(blocks, 0, si),
        s(blocks, 1, si),
        s(extra, 0, si),
        s(extra, 1, si),
        s(extra, 2, si),
        s(extra, 3, si),
        vec,
    });
}

/// Ports `ggml_cuda_op_top_k_mask` (top-k-mask.cu:130 @b1b770d90).
pub fn opTopKMask(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.c.ggml_tensor) void {
    switch (dst.type) {
        ggml.impl.c.GGML_TYPE_F16 => topKMaskCuda(ggml.cuda.kernels.half, ctx, dst),
        ggml.impl.c.GGML_TYPE_F32 => topKMaskCuda(f32, ctx, dst),
        else => ggml.impl.abort("unsupported type"),
    }
}
