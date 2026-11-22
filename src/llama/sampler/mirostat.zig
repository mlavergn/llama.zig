//! The samplers that steer toward a target: both mirostats and adaptive-p.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Fused multiply-adds
//!
//! Read off the reference `llama-sampler.o`'s disassembly, as in `basic.zig`:
//!
//! - mirostat, `sum_ti_bi += t_i * b_i` and `sum_ti_sq += t_i * t_i`:
//!   `fmadd s9, s8, s0, s9` and `fmadd s10, s8, s8, s10`
//! - both mirostats, `mu - eta * e`: `fmsub s0, s2, s0, s1`
//! - adaptive-p, `2.0f * target - (weighted_sum / total_weight)`:
//!   `fmadd s1, s1, s2, s0` over the negated quotient -- the product is
//!   exact, but the fused form is what the C computes, so it is written that
//!   way
//! - adaptive-p's accept, both EMA updates: one `fmla.2s v2, v1, v0[0]`,
//!   so `orig + decay * weighted_sum` and `1 + decay * total_weight` are
//!   each fused.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

// -----------------------------------------------------------------------------
// mirostat

/// Ports `llama_sampler_mirostat` (llama-sampler.cpp:2447 @b1b770d90).
pub const Mirostat = struct {
    n_vocab: i32,

    seed: u32,
    seed_cur: u32,

    tau: f32,
    eta: f32,

    m: i32,

    mu: f32,

    rng: stdlib.libcxx.Mt19937,
};

/// Ports `llama_sampler_mirostat_name` (llama-sampler.cpp:2463 @b1b770d90).
fn mirostatName(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "mirostat";
}

/// Ports `llama_sampler_mirostat_apply` (llama-sampler.cpp:2467 @b1b770d90).
///
/// Estimates the Zipf exponent from the top `m` probabilities, derives the
/// top-k that should give surprise `mu`, draws, and moves `mu` toward `tau`.
fn mirostatApply(smpl: [*c]llama.impl.c.struct_llama_sampler, cur_p: [*c]llama.impl.c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Mirostat, smpl);

    llama.sampler.core.softmaxImpl(cur_p, true);

    var s_hat: f32 = 0.0;
    var sum_ti_bi: f32 = 0.0;
    var sum_ti_sq: f32 = 0.0;
    var i: usize = 0;
    // `size_t(ctx->m - 1)` and `cur_p->size - 1`, both unsigned.
    const m_lim: usize = @bitCast(@as(isize, ctx.m - 1));
    while (i < m_lim and i < cur_p.*.size -% 1) : (i += 1) {
        const t_i = stdlib.libm.logf(@as(f32, @floatFromInt(i + 2)) / @as(f32, @floatFromInt(i + 1)));
        const b_i = stdlib.libm.logf(cur_p.*.data[i].p / cur_p.*.data[i + 1].p);
        sum_ti_bi = @mulAdd(f32, t_i, b_i, sum_ti_bi);
        sum_ti_sq = @mulAdd(f32, t_i, t_i, sum_ti_sq);
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    const epsilon_hat = s_hat - 1;
    const k = stdlib.libm.powf((epsilon_hat * stdlib.libm.powf(2, ctx.mu)) / (1 - stdlib.libm.powf(@floatFromInt(ctx.n_vocab), -epsilon_hat)), 1 / s_hat);

    llama.sampler.core.topKImpl(cur_p, stdlib.libcxx.max(c_int, llama.sampler.core.cvtzs(k), 1));

    llama.sampler.core.softmaxImpl(cur_p, true);

    const idx = llama.sampler.core.sampleDist(cur_p, &ctx.rng);

    cur_p.*.selected = idx;

    const observed_surprise = -stdlib.libm.log2f(cur_p.*.data[@intCast(idx)].p);
    const e = observed_surprise - ctx.tau;

    ctx.mu = @mulAdd(f32, -ctx.eta, e, ctx.mu);
}

/// Ports `llama_sampler_mirostat_clone` (llama-sampler.cpp:2503 @b1b770d90).
fn mirostatClone(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(Mirostat, smpl);
    const result = llama_sampler_init_mirostat(ctx.n_vocab, ctx.seed, ctx.tau, ctx.eta, ctx.m);

    const result_ctx = llama.sampler.core.ctxOf(Mirostat, result);
    result_ctx.mu = ctx.mu;
    result_ctx.rng = ctx.rng;

    return result;
}

/// Ports `llama_sampler_mirostat_reset` (llama-sampler.cpp:2518 @b1b770d90).
fn mirostatReset(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Mirostat, smpl);
    ctx.mu = 2.0 * ctx.tau;
    ctx.seed_cur = llama.sampler.core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
}

/// Ports `llama_sampler_mirostat_free` (llama-sampler.cpp:2525 @b1b770d90).
fn mirostatFree(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(Mirostat, llama.sampler.core.ctxOf(Mirostat, smpl));
}

/// Ports `llama_sampler_mirostat_i` (llama-sampler.cpp:2529 @b1b770d90).
pub var mirostat_iface: llama.impl.c.struct_llama_sampler_i = .{
    .name = mirostatName,
    .accept = null,
    .apply = mirostatApply,
    .reset = mirostatReset,
    .clone = mirostatClone,
    .free = mirostatFree,
};

/// Ports `llama_sampler_init_mirostat` (llama-sampler.cpp:2544 @b1b770d90).
///
/// Parameters:
/// - `n_vocab`: the vocabulary size.
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
/// - `tau`: the target surprise.
/// - `eta`: the learning rate.
/// - `m`: how many top tokens estimate the exponent.
///
/// Return: the sampler.
pub export fn llama_sampler_init_mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const seed_cur = llama.sampler.core.getRngSeed(seed);

    return llama.sampler.core.llama_sampler_init(&mirostat_iface, llama.sampler.core.create(Mirostat, .{
        .n_vocab = n_vocab,
        .seed = seed,
        .seed_cur = seed_cur,
        .tau = tau,
        .eta = eta,
        .m = m,
        .mu = 2.0 * tau,
        .rng = stdlib.libcxx.Mt19937.init(seed_cur),
    }));
}

// -----------------------------------------------------------------------------
// mirostat v2

/// Ports `llama_sampler_mirostat_v2` (llama-sampler.cpp:2564 @b1b770d90).
pub const MirostatV2 = struct {
    seed: u32,
    seed_cur: u32,

    tau: f32,
    eta: f32,

    mu: f32,

    rng: stdlib.libcxx.Mt19937,
};

/// Ports `llama_sampler_mirostat_v2_name` (llama-sampler.cpp:2576 @b1b770d90).
fn mirostatV2Name(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "mirostat-v2";
}

/// Ports `llama_sampler_mirostat_v2_apply` (llama-sampler.cpp:2580 @b1b770d90).
///
/// Cuts at the first candidate whose surprise exceeds `mu`, keeping at least
/// one, draws, and moves `mu` toward `tau`.
fn mirostatV2Apply(smpl: [*c]llama.impl.c.struct_llama_sampler, cur_p: [*c]llama.impl.c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(MirostatV2, smpl);

    llama.sampler.core.softmaxImpl(cur_p, true);

    var cut: usize = 0;
    while (cut < cur_p.*.size) : (cut += 1) {
        if (-stdlib.libm.log2f(cur_p.*.data[cut].p) > ctx.mu) break;
    }
    cur_p.*.size = cut;

    if (cur_p.*.size == 0) {
        cur_p.*.size = 1;
    }

    llama.sampler.core.softmaxImpl(cur_p, true);

    const idx = llama.sampler.core.sampleDist(cur_p, &ctx.rng);

    cur_p.*.selected = idx;

    const observed_surprise = -stdlib.libm.log2f(cur_p.*.data[@intCast(idx)].p);
    const e = observed_surprise - ctx.tau;

    ctx.mu = @mulAdd(f32, -ctx.eta, e, ctx.mu);
}

/// Ports `llama_sampler_mirostat_v2_reset` (llama-sampler.cpp:2608 @b1b770d90).
fn mirostatV2Reset(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(MirostatV2, smpl);
    ctx.mu = 2.0 * ctx.tau;
    ctx.seed_cur = llama.sampler.core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
}

/// Ports `llama_sampler_mirostat_v2_clone` (llama-sampler.cpp:2615 @b1b770d90).
fn mirostatV2Clone(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(MirostatV2, smpl);

    const result = llama_sampler_init_mirostat_v2(ctx.seed, ctx.tau, ctx.eta);

    const result_ctx = llama.sampler.core.ctxOf(MirostatV2, result);
    result_ctx.mu = ctx.mu;
    result_ctx.rng = ctx.rng;

    return result;
}

/// Ports `llama_sampler_mirostat_v2_free` (llama-sampler.cpp:2631 @b1b770d90).
fn mirostatV2Free(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(MirostatV2, llama.sampler.core.ctxOf(MirostatV2, smpl));
}

/// Ports `llama_sampler_mirostat_v2_i` (llama-sampler.cpp:2635 @b1b770d90).
pub var mirostat_v2_iface: llama.impl.c.struct_llama_sampler_i = .{
    .name = mirostatV2Name,
    .accept = null,
    .apply = mirostatV2Apply,
    .reset = mirostatV2Reset,
    .clone = mirostatV2Clone,
    .free = mirostatV2Free,
};

/// Ports `llama_sampler_init_mirostat_v2` (llama-sampler.cpp:2650 @b1b770d90).
///
/// Parameters:
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
/// - `tau`: the target surprise.
/// - `eta`: the learning rate.
///
/// Return: the sampler.
pub export fn llama_sampler_init_mirostat_v2(seed: u32, tau: f32, eta: f32) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const seed_cur = llama.sampler.core.getRngSeed(seed);
    return llama.sampler.core.llama_sampler_init(&mirostat_v2_iface, llama.sampler.core.create(MirostatV2, .{
        .seed = seed,
        .seed_cur = seed_cur,
        .tau = tau,
        .eta = eta,
        .mu = 2.0 * tau,
        .rng = stdlib.libcxx.Mt19937.init(seed_cur),
    }));
}

// -----------------------------------------------------------------------------
// adaptive-p

/// Ports `llama_sampler_adaptive_p` (llama-sampler.cpp:3723 @b1b770d90).
const AdaptiveP = struct {
    target: f32,
    decay: f32,
    seed: u32,
    seed_cur: u32,
    rng: stdlib.libcxx.Mt19937,
    weighted_sum: f32,
    total_weight: f32,
    original_probs: std.ArrayList(f32),
    pending_token_id: llama.impl.c.llama_token,
    pending_token_idx: i32,
};

// Ports `DISTRIBUTION_WIDTH`, `PEAK_LOGIT_VALUE`, `SHARPNESS`, `INV_WIDTH`
// (llama-sampler.cpp:3737, 3738, 3739, 3740 @b1b770d90).
const DISTRIBUTION_WIDTH: f32 = 0.3;
const PEAK_LOGIT_VALUE: f32 = 5.0;
const SHARPNESS: f32 = 10.0;
const INV_WIDTH: f32 = 1.0 / DISTRIBUTION_WIDTH;

/// Ports `llama_sampler_adaptive_p_name` (llama-sampler.cpp:3742 @b1b770d90).
fn adaptivePName(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "adaptive-p";
}

/// Ports `llama_sampler_adaptive_p_apply` (llama-sampler.cpp:3746 @b1b770d90).
///
/// Re-weights candidates toward a target probability that adapts to the
/// running average of what was picked, then draws.
fn adaptivePApply(smpl: [*c]llama.impl.c.struct_llama_sampler, cur_p: [*c]llama.impl.c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(AdaptiveP, smpl);

    llama.sampler.core.softmaxImpl(cur_p, false);

    if (ctx.target < 0.0) {
        cur_p.*.selected = llama.sampler.core.sampleDist(cur_p, &ctx.rng);
        return;
    }

    ctx.original_probs.resize(std.heap.c_allocator, cur_p.*.size) catch llama.impl.abort(llama.sampler.core.src, 3763, "out of memory");
    for (llama.sampler.core.items(cur_p), 0..) |d, i| ctx.original_probs.items[i] = d.p;

    const target = stdlib.libcxx.clamp(f32, ctx.target, 0.0, 1.0);
    const adapted_target = stdlib.libcxx.clamp(
        f32,
        if (ctx.total_weight == 0.0) target else @mulAdd(f32, target, 2.0, -ctx.weighted_sum / ctx.total_weight),
        0.0,
        1.0,
    );

    for (llama.sampler.core.items(cur_p)) |*d| {
        if (d.logit == -std.math.inf(f32)) continue;
        const dist = @abs((d.p - adapted_target) * INV_WIDTH);
        d.logit = PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0 + dist);
    }

    llama.sampler.core.softmaxImpl(cur_p, false);
    const idx = llama.sampler.core.sampleDist(cur_p, &ctx.rng);
    cur_p.*.selected = idx;

    ctx.pending_token_id = cur_p.*.data[@intCast(idx)].id;
    ctx.pending_token_idx = idx;
}

/// Ports `llama_sampler_adaptive_p_accept` (llama-sampler.cpp:3797 @b1b770d90).
///
/// Folds the selected token's original probability into the average, when
/// the accepted token is the one this sampler picked.
fn adaptivePAccept(smpl: [*c]llama.impl.c.struct_llama_sampler, token: llama.impl.c.llama_token) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(AdaptiveP, smpl);
    if (ctx.pending_token_id == token) {
        llama.impl.assert(ctx.pending_token_id != llama.impl.c.LLAMA_TOKEN_NULL, llama.sampler.core.src, 3834, "ctx->pending_token_id != LLAMA_TOKEN_NULL");
        llama.impl.assert(ctx.pending_token_idx != -1, llama.sampler.core.src, 3835, "ctx->pending_token_idx != -1");
        ctx.weighted_sum = @mulAdd(f32, ctx.decay, ctx.weighted_sum, ctx.original_probs.items[@intCast(ctx.pending_token_idx)]);
        ctx.total_weight = @mulAdd(f32, ctx.decay, ctx.total_weight, 1.0);
    }
    ctx.pending_token_id = llama.impl.c.LLAMA_TOKEN_NULL;
    ctx.pending_token_idx = -1;
}

/// Ports `llama_sampler_adaptive_p_reset` (llama-sampler.cpp:3810 @b1b770d90).
fn adaptivePReset(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(AdaptiveP, smpl);
    ctx.weighted_sum = ctx.target / (1.0 - ctx.decay);
    ctx.total_weight = 1.0 / (1.0 - ctx.decay);
    ctx.pending_token_id = llama.impl.c.LLAMA_TOKEN_NULL;
    ctx.pending_token_idx = -1;
    ctx.seed_cur = llama.sampler.core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
}

/// Ports `llama_sampler_adaptive_p_clone` (llama-sampler.cpp:3823 @b1b770d90).
///
/// Copies the average and the pending token but not the engine: the clone
/// starts from a freshly seeded one, as the C++'s does.
fn adaptivePClone(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(AdaptiveP, smpl);
    const result = llama_sampler_init_adaptive_p(ctx.target, ctx.decay, ctx.seed);
    const result_ctx = llama.sampler.core.ctxOf(AdaptiveP, result);

    result_ctx.weighted_sum = ctx.weighted_sum;
    result_ctx.total_weight = ctx.total_weight;
    result_ctx.pending_token_id = ctx.pending_token_id;
    result_ctx.pending_token_idx = ctx.pending_token_idx;

    return result;
}

/// Ports `llama_sampler_adaptive_p_free` (llama-sampler.cpp:3837 @b1b770d90).
fn adaptivePFree(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(AdaptiveP, smpl);
    ctx.original_probs.deinit(std.heap.c_allocator);
    llama.sampler.core.destroy(AdaptiveP, ctx);
}

/// Ports `llama_sampler_adaptive_p_i` (llama-sampler.cpp:3841 @b1b770d90).
var adaptive_p_iface: llama.impl.c.struct_llama_sampler_i = .{
    .name = adaptivePName,
    .accept = adaptivePAccept,
    .apply = adaptivePApply,
    .reset = adaptivePReset,
    .clone = adaptivePClone,
    .free = adaptivePFree,
};

/// Ports `llama_sampler_init_adaptive_p` (llama-sampler.cpp:3856 @b1b770d90).
///
/// Parameters:
/// - `target`: the target probability; negative makes it a plain draw.
/// - `decay`: the average's decay, clamped to [0, 0.99].
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
///
/// Return: the sampler.
pub export fn llama_sampler_init_adaptive_p(target: f32, decay: f32, seed: u32) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const seed_cur = llama.sampler.core.getRngSeed(seed);
    const clamped_decay = stdlib.libcxx.clamp(f32, decay, 0.0, 0.99);
    return llama.sampler.core.llama_sampler_init(&adaptive_p_iface, llama.sampler.core.create(AdaptiveP, .{
        .target = target,
        .decay = clamped_decay,
        .seed = seed,
        .seed_cur = seed_cur,
        .rng = stdlib.libcxx.Mt19937.init(seed_cur),
        .weighted_sum = target / (1.0 - clamped_decay),
        .total_weight = 1.0 / (1.0 - clamped_decay),
        .original_probs = .empty,
        .pending_token_id = llama.impl.c.LLAMA_TOKEN_NULL,
        .pending_token_idx = -1,
    }));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
