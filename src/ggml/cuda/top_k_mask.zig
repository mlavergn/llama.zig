//! `GGML_OP_TOP_K_MASK`, new at v0.8.0: a row of `-inf` with the mask cells
//! of the live top-k picks, and of any extra cells, copied through. The
//! sparse-attention (QSA/DSA) mask.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/top-k-mask.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/top-k-mask.cuh` at v0.8.0 (`8642541b2`).
//! The kernel, `top_k_mask_kernel`, stays CUDA.

const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `CUDA_TOP_K_MASK_BLOCK_SIZE` (top-k-mask.cu:4 @8642541b2).
const block_size = 512;

/// `sizeof(int4)`: the kernel's vectorized store.
const sizeof_int4 = 16;

/// A byte stride in elements, into an `int64_t` parameter; zero for a
/// source the op does not have, as the C's `score ? score->nb[0]/sf : 0`.
fn s(t: ?*const Tensor, comptime i: usize, comptime size: usize) i64 {
    return if (t) |x| @bitCast(x.nb[i] / size) else 0;
}

/// `t ? (const T *) t->data : nullptr`.
fn data(t: ?*const Tensor) ?*const anyopaque {
    return if (t) |x| x.data else null;
}

/// Ports `top_k_mask_cuda` (top-k-mask.cu:86 @8642541b2), its element type
/// `T` comptime: `f32` or `kernels.half`.
fn topKMaskCuda(comptime T: type, ctx: *common.Context, dst: *Tensor) void {
    const mask = impl.one(Tensor, dst.src[0]);
    const top_k = impl.one(Tensor, dst.src[1]);
    const score: ?*const Tensor = dst.src[2];
    const blocks: ?*const Tensor = dst.src[3];
    const extra: ?*const Tensor = dst.src[4];

    const n_kv = mask.ne[0];
    const ts = @sizeOf(T);

    impl.assert(dst.nb[0] == ts and mask.nb[0] == ts, "dst->nb[0] == sizeof(T) && mask->nb[0] == sizeof(T)");
    impl.assert(dst.nb[1] % ts == 0 and mask.nb[1] % ts == 0, "dst->nb[1] % sizeof(T) == 0 && mask->nb[1] % sizeof(T) == 0");

    // 16-byte stores need every dst row aligned
    const vec = (@as(u64, @bitCast(n_kv)) *% ts) % sizeof_int4 == 0 and
        dst.nb[1] % sizeof_int4 == 0 and dst.nb[2] % sizeof_int4 == 0 and dst.nb[3] % sizeof_int4 == 0 and
        @intFromPtr(dst.data) % sizeof_int4 == 0;

    const n_pick: i64 = if (blocks) |b| b.ne[1] else if (score) |sc| sc.ne[0] else n_kv;

    const si = @sizeOf(i32);
    const sf = @sizeOf(f32);

    kernels.launchPdl(kernels.get("top-k-mask", kernels.tmpl("top_k_mask_kernel", &.{kernels.typeName(T)})), .{
        .grid = .{ .x = @truncate(@as(u64, @bitCast(c.ggml_nrows(dst)))) },
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

/// Ports `ggml_cuda_op_top_k_mask` (top-k-mask.cu:130 @8642541b2).
pub fn opTopKMask(ctx: *common.Context, dst: *Tensor) void {
    switch (dst.type) {
        c.GGML_TYPE_F16 => topKMaskCuda(kernels.half, ctx, dst),
        c.GGML_TYPE_F32 => topKMaskCuda(f32, ctx, dst),
        else => impl.abort("unsupported type"),
    }
}
