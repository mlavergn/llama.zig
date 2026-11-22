//! `GGML_OP_DSV4_HC_COMB`, `GGML_OP_DSV4_HC_PRE` and `GGML_OP_DSV4_HC_POST`:
//! DeepSeek-V4's hyper-connection mixing.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/dsv4-hc.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/dsv4-hc.cuh` at v1.0.0 (`b1b770d90`). The
//! kernels -- `dsv4_hc_comb_f32`, `dsv4_hc_pre_f32<gated>`,
//! `dsv4_hc_post_f32<has_comb, gated>` -- stay CUDA. All three launch through
//! `ggml_cuda_kernel_launch` (PDL).

const ggml = @import("../module.zig");
/// Ports `DSV4_HC` (dsv4-hc.cu:5 @b1b770d90).
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

fn srcOf(dst: *const ggml.impl.ggml_c.ggml_tensor, i: usize) *const ggml.impl.ggml_c.ggml_tensor {
    return ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[i]);
}

/// Ports `ggml_cuda_op_dsv4_hc_comb` (dsv4-hc.cu:205 @b1b770d90).
pub fn opDsv4HcComb(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const mixes = srcOf(dst, 0);
    const scale = srcOf(dst, 1);
    const base = srcOf(dst, 2);

    ggml.impl.assert(mixes.type == ggml.impl.ggml_c.GGML_TYPE_F32, "mixes->type == GGML_TYPE_F32");
    ggml.impl.assert(scale.type == ggml.impl.ggml_c.GGML_TYPE_F32, "scale->type == GGML_TYPE_F32");
    ggml.impl.assert(base.type == ggml.impl.ggml_c.GGML_TYPE_F32, "base->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const hc_mix_dim: i64 = (2 + hc_n) * hc_n;

    ggml.impl.assert(mixes.ne[0] == hc_mix_dim, "mixes->ne[0] == hc_mix_dim");
    ggml.impl.assert(dst.ne[0] == hc_n, "dst->ne[0] == DSV4_HC");
    ggml.impl.assert(dst.ne[1] == hc_n, "dst->ne[1] == DSV4_HC");
    ggml.impl.assert(dst.ne[2] == mixes.ne[1], "dst->ne[2] == mixes->ne[1]");
    ggml.impl.assert(scale.ne[0] >= 3, "scale->ne[0] >= 3");
    ggml.impl.assert(base.ne[0] == hc_mix_dim, "base->ne[0] == hc_mix_dim");

    const n_tokens = mixes.ne[1];
    const eps = ggml.impl.getOpParamsF32(dst, 0);
    const n_iter = ggml.impl.getOpParamsI32(dst, 1);

    const block_size = 256;
    ggml.cuda.kernels.launchPdl(ggml.cuda.kernels.get("dsv4-hc", "dsv4_hc_comb_f32"), .{
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

/// Ports `ggml_cuda_op_dsv4_hc_pre` (dsv4-hc.cu:248 @b1b770d90). Op params
/// slot 0 is a scale the sum is multiplied by and slot 1 says whether the
/// weights are gated: `ggml_dsv4_hc_pre_gated` gives one weight per
/// element, put through a sigmoid in the kernel, where the plain op gives
/// one per stream.
pub fn opDsv4HcPre(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const x = srcOf(dst, 0);
    const weights = srcOf(dst, 1);

    ggml.impl.assert(x.type == ggml.impl.ggml_c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    ggml.impl.assert(weights.type == ggml.impl.ggml_c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const n_embd = x.ne[0];
    const hc = x.ne[1];
    const n_tokens = x.ne[2];

    const scale = ggml.impl.getOpParamsF32(dst, 0);
    const gated = ggml.impl.getOpParamsI32(dst, 1) != 0;

    const block_size = 256;
    const nr = n_embd * n_tokens;
    const kernel = if (gated) ggml.cuda.kernels.get("dsv4-hc", "dsv4_hc_pre_f32<(bool)1>") else ggml.cuda.kernels.get("dsv4-hc", "dsv4_hc_pre_f32<(bool)0>");
    ggml.cuda.kernels.launchPdl(kernel, .{
        .grid = .{ .x = dim(@divTrunc(nr + block_size - 1, block_size)) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = ctx.stream(),
    }, .{
        @as(?*const anyopaque, x.data), @as(?*const anyopaque, weights.data), @as(?*anyopaque, dst.data),
        n_embd,                         hc,                                   n_tokens,
        fs(x.nb[0]),                    fs(x.nb[1]),                          fs(x.nb[2]),
        fs(weights.nb[0]),              fs(weights.nb[1]),                    fs(weights.nb[2]),
        fs(dst.nb[0]),                  fs(dst.nb[1]),                        scale,
    });
}

/// Ports `ggml_cuda_op_dsv4_hc_post` (dsv4-hc.cu:283 @b1b770d90). A null
/// `comb` (`src[3]`) is the identity: each stream adds its own residual,
/// through `dsv4_hc_post_f32<false, gated>`, which never reads `comb` or its
/// strides, both passed as zero. `gate_scale_in` and
/// `gate_scale_out` are the `SCALE` nodes of a fused `SCALE -> SIGMOID ->
/// SCALE` chain feeding `src[2]` (qwen4exp's `hc_combine`): the kernel reads
/// the raw weights from before the chain and applies it.
pub fn opDsv4HcPost(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.ggml_c.ggml_tensor, gate_scale_in: ?*const ggml.impl.ggml_c.ggml_tensor, gate_scale_out: ?*const ggml.impl.ggml_c.ggml_tensor) void {
    const x = srcOf(dst, 0);
    const residual = srcOf(dst, 1);
    const comb: ?*const ggml.impl.ggml_c.ggml_tensor = dst.src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const gated = gate_scale_in != null;
    const post = if (gate_scale_in) |g| srcOf(g, 0) else srcOf(dst, 2);

    const gate_in: f32 = if (gate_scale_in) |g| ggml.impl.getOpParamsF32(g, 0) else 1.0;
    const gate_out: f32 = if (gated) ggml.impl.getOpParamsF32(gate_scale_out.?, 0) else 1.0;

    ggml.impl.assert(x.type == ggml.impl.ggml_c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    ggml.impl.assert(residual.type == ggml.impl.ggml_c.GGML_TYPE_F32, "residual->type == GGML_TYPE_F32");
    ggml.impl.assert(post.type == ggml.impl.ggml_c.GGML_TYPE_F32, "post->type == GGML_TYPE_F32");
    ggml.impl.assert(comb == null or comb.?.type == ggml.impl.ggml_c.GGML_TYPE_F32, "comb == nullptr || comb->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const nbc0: usize = if (comb) |t| t.nb[0] else 0;
    const nbc1: usize = if (comb) |t| t.nb[1] else 0;
    const nbc2: usize = if (comb) |t| t.nb[2] else 0;

    const n_embd = x.ne[0];
    const n_tokens = x.ne[1];
    const hc = residual.ne[1];

    const block_size = 256;
    const nr = n_embd * hc * n_tokens;
    const kernel = if (comb != null)
        (if (gated) ggml.cuda.kernels.get("dsv4-hc", "dsv4_hc_post_f32<(bool)1, (bool)1>") else ggml.cuda.kernels.get("dsv4-hc", "dsv4_hc_post_f32<(bool)1, (bool)0>"))
    else
        (if (gated) ggml.cuda.kernels.get("dsv4-hc", "dsv4_hc_post_f32<(bool)0, (bool)1>") else ggml.cuda.kernels.get("dsv4-hc", "dsv4_hc_post_f32<(bool)0, (bool)0>"));
    const comb_d: ?*const anyopaque = if (comb) |t| t.data else null;
    ggml.cuda.kernels.launchPdl(kernel, .{
        .grid = .{ .x = dim(@divTrunc(nr + block_size - 1, block_size)) },
        .block = .{ .x = block_size },
        .shmem = 0,
        .stream = ctx.stream(),
    }, .{
        @as(?*const anyopaque, x.data), @as(?*const anyopaque, residual.data), @as(?*const anyopaque, post.data),
        comb_d,                         @as(?*anyopaque, dst.data),            n_embd,
        hc,                             n_tokens,                              fs(x.nb[0]),
        fs(x.nb[1]),                    fs(residual.nb[0]),                    fs(residual.nb[1]),
        fs(residual.nb[2]),             fs(post.nb[0]),                        fs(post.nb[1]),
        fs(nbc0),                       fs(nbc1),                              fs(nbc2),
        fs(dst.nb[0]),                  fs(dst.nb[1]),                         fs(dst.nb[2]),
        gate_in,                        gate_out,
    });
}
