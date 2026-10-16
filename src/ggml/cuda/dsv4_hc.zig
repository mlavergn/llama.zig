//! `GGML_OP_DSV4_HC_COMB`, `GGML_OP_DSV4_HC_PRE` and `GGML_OP_DSV4_HC_POST`:
//! DeepSeek-V4's hyper-connection mixing.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/dsv4-hc.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/dsv4-hc.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels -- `dsv4_hc_comb_f32`, `dsv4_hc_pre_f32`, `dsv4_hc_post_f32` --
//! stay CUDA. All three launch through `ggml_cuda_kernel_launch` (PDL).

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// Ports `DSV4_HC` (dsv4-hc.cu:5 @c1d0e7a00).
const hc_n = 4;

/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// A byte stride in floats, `nb / sizeof(float)` into an `int64_t`
/// parameter.
fn fs(nb: usize) i64 {
    return @bitCast(nb / @sizeOf(f32));
}

fn srcOf(dst: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, dst.src[i]);
}

/// Ports `ggml_cuda_op_dsv4_hc_comb` (dsv4-hc.cu:184 @c1d0e7a00).
pub fn opDsv4HcComb(ctx: *common.Context, dst: *Tensor) void {
    const mixes = srcOf(dst, 0);
    const scale = srcOf(dst, 1);
    const base = srcOf(dst, 2);

    impl.assert(mixes.type == c.GGML_TYPE_F32, "mixes->type == GGML_TYPE_F32");
    impl.assert(scale.type == c.GGML_TYPE_F32, "scale->type == GGML_TYPE_F32");
    impl.assert(base.type == c.GGML_TYPE_F32, "base->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const hc_mix_dim: i64 = (2 + hc_n) * hc_n;

    impl.assert(mixes.ne[0] == hc_mix_dim, "mixes->ne[0] == hc_mix_dim");
    impl.assert(dst.ne[0] == hc_n, "dst->ne[0] == DSV4_HC");
    impl.assert(dst.ne[1] == hc_n, "dst->ne[1] == DSV4_HC");
    impl.assert(dst.ne[2] == mixes.ne[1], "dst->ne[2] == mixes->ne[1]");
    impl.assert(scale.ne[0] >= 3, "scale->ne[0] >= 3");
    impl.assert(base.ne[0] == hc_mix_dim, "base->ne[0] == hc_mix_dim");

    const n_tokens = mixes.ne[1];
    const eps = impl.getOpParamsF32(dst, 0);
    const n_iter = impl.getOpParamsI32(dst, 1);

    const block_size = 256;
    kernels.launchPdl(kernels.get("dsv4-hc", "dsv4_hc_comb_f32"), .{
        .grid = .{ .x = dim(@divTrunc(n_tokens + block_size - 1, block_size)) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = ctx.stream(),
    }, .{
        @as(?*const anyopaque, mixes.data), @as(?*const anyopaque, scale.data), @as(?*const anyopaque, base.data), @as(?*anyopaque, dst.data),
        n_tokens,                           fs(mixes.nb[0]),                    fs(mixes.nb[1]),                   fs(scale.nb[0]),
        fs(base.nb[0]),                     fs(dst.nb[0]),                      fs(dst.nb[1]),                     fs(dst.nb[2]),
        eps,                                n_iter,
    });
}

/// Ports `ggml_cuda_op_dsv4_hc_pre` (dsv4-hc.cu:227 @c1d0e7a00).
pub fn opDsv4HcPre(ctx: *common.Context, dst: *Tensor) void {
    const x = srcOf(dst, 0);
    const weights = srcOf(dst, 1);

    impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const n_embd = x.ne[0];
    const hc = x.ne[1];
    const n_tokens = x.ne[2];

    const block_size = 256;
    const nr = n_embd * n_tokens;
    kernels.launchPdl(kernels.get("dsv4-hc", "dsv4_hc_pre_f32"), .{
        .grid = .{ .x = dim(@divTrunc(nr + block_size - 1, block_size)) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = ctx.stream(),
    }, .{
        @as(?*const anyopaque, x.data), @as(?*const anyopaque, weights.data), @as(?*anyopaque, dst.data),
        n_embd,                         hc,                                   n_tokens,
        fs(x.nb[0]),                    fs(x.nb[1]),                          fs(x.nb[2]),
        fs(weights.nb[0]),              fs(weights.nb[1]),                    fs(dst.nb[0]),
        fs(dst.nb[1]),
    });
}

/// Ports `ggml_cuda_op_dsv4_hc_post` (dsv4-hc.cu:257 @c1d0e7a00).
pub fn opDsv4HcPost(ctx: *common.Context, dst: *Tensor) void {
    const x = srcOf(dst, 0);
    const residual = srcOf(dst, 1);
    const post = srcOf(dst, 2);
    const comb = srcOf(dst, 3);

    impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    impl.assert(residual.type == c.GGML_TYPE_F32, "residual->type == GGML_TYPE_F32");
    impl.assert(post.type == c.GGML_TYPE_F32, "post->type == GGML_TYPE_F32");
    impl.assert(comb.type == c.GGML_TYPE_F32, "comb->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const n_embd = x.ne[0];
    const n_tokens = x.ne[1];
    const hc = residual.ne[1];

    const block_size = 256;
    const nr = n_embd * hc * n_tokens;
    kernels.launchPdl(kernels.get("dsv4-hc", "dsv4_hc_post_f32"), .{
        .grid = .{ .x = dim(@divTrunc(nr + block_size - 1, block_size)) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = ctx.stream(),
    }, .{
        @as(?*const anyopaque, x.data),    @as(?*const anyopaque, residual.data), @as(?*const anyopaque, post.data),
        @as(?*const anyopaque, comb.data), @as(?*anyopaque, dst.data),            n_embd,
        hc,                                n_tokens,                              fs(x.nb[0]),
        fs(x.nb[1]),                       fs(residual.nb[0]),                    fs(residual.nb[1]),
        fs(residual.nb[2]),                fs(post.nb[0]),                        fs(post.nb[1]),
        fs(comb.nb[0]),                    fs(comb.nb[1]),                        fs(comb.nb[2]),
        fs(dst.nb[0]),                     fs(dst.nb[1]),                         fs(dst.nb[2]),
    });
}
