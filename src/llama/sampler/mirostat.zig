//! The samplers that steer toward a target: both mirostats and adaptive-p.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v0.4.1, `b29c606e2`).
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
const impl = @import("../impl.zig");
const libcxx = @import("../libcxx.zig");
const core = @import("core.zig");

const c = impl.c;
const Sampler = core.Sampler;
const SamplerI = core.SamplerI;
const TokenDataArray = core.TokenDataArray;
const Token = core.Token;
const src = core.src;

// -----------------------------------------------------------------------------
// mirostat

/// Ports `llama_sampler_mirostat` (llama-sampler.cpp:2440 @b29c606e2).
pub const Mirostat = struct {
    n_vocab: i32,

    seed: u32,
    seed_cur: u32,

    tau: f32,
    eta: f32,

    m: i32,

    mu: f32,

    rng: libcxx.Mt19937,
};

/// Ports `llama_sampler_mirostat_name` (llama-sampler.cpp:2456 @b29c606e2).
fn mirostatName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "mirostat";
}

/// Ports `llama_sampler_mirostat_apply` (llama-sampler.cpp:2460 @b29c606e2).
///
/// Estimates the Zipf exponent from the top `m` probabilities, derives the
/// top-k that should give surprise `mu`, draws, and moves `mu` toward `tau`.
fn mirostatApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(Mirostat, smpl);

    core.softmaxImpl(cur_p, true);

    var s_hat: f32 = 0.0;
    var sum_ti_bi: f32 = 0.0;
    var sum_ti_sq: f32 = 0.0;
    var i: usize = 0;
    // `size_t(ctx->m - 1)` and `cur_p->size - 1`, both unsigned.
    const m_lim: usize = @bitCast(@as(isize, ctx.m - 1));
    while (i < m_lim and i < cur_p.*.size -% 1) : (i += 1) {
        const t_i = impl.logf(@as(f32, @floatFromInt(i + 2)) / @as(f32, @floatFromInt(i + 1)));
        const b_i = impl.logf(cur_p.*.data[i].p / cur_p.*.data[i + 1].p);
        sum_ti_bi = @mulAdd(f32, t_i, b_i, sum_ti_bi);
        sum_ti_sq = @mulAdd(f32, t_i, t_i, sum_ti_sq);
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    const epsilon_hat = s_hat - 1;
    const k = impl.powf((epsilon_hat * impl.powf(2, ctx.mu)) / (1 - impl.powf(@floatFromInt(ctx.n_vocab), -epsilon_hat)), 1 / s_hat);

    core.topKImpl(cur_p, libcxx.max(c_int, core.cvtzs(k), 1));

    core.softmaxImpl(cur_p, true);

    const idx = core.sampleDist(cur_p, &ctx.rng);

    cur_p.*.selected = idx;

    const observed_surprise = -impl.log2f(cur_p.*.data[@intCast(idx)].p);
    const e = observed_surprise - ctx.tau;

    ctx.mu = @mulAdd(f32, -ctx.eta, e, ctx.mu);
}

/// Ports `llama_sampler_mirostat_clone` (llama-sampler.cpp:2496 @b29c606e2).
fn mirostatClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(Mirostat, smpl);
    const result = llama_sampler_init_mirostat(ctx.n_vocab, ctx.seed, ctx.tau, ctx.eta, ctx.m);

    const result_ctx = core.ctxOf(Mirostat, result);
    result_ctx.mu = ctx.mu;
    result_ctx.rng = ctx.rng;

    return result;
}

/// Ports `llama_sampler_mirostat_reset` (llama-sampler.cpp:2511 @b29c606e2).
fn mirostatReset(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(Mirostat, smpl);
    ctx.mu = 2.0 * ctx.tau;
    ctx.seed_cur = core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
}

/// Ports `llama_sampler_mirostat_free` (llama-sampler.cpp:2518 @b29c606e2).
fn mirostatFree(smpl: [*c]Sampler) callconv(.c) void {
    core.destroy(Mirostat, core.ctxOf(Mirostat, smpl));
}

/// Ports `llama_sampler_mirostat_i` (llama-sampler.cpp:2522 @b29c606e2).
pub var mirostat_iface: SamplerI = .{
    .name = mirostatName,
    .accept = null,
    .apply = mirostatApply,
    .reset = mirostatReset,
    .clone = mirostatClone,
    .free = mirostatFree,
};

/// Ports `llama_sampler_init_mirostat` (llama-sampler.cpp:2537 @b29c606e2).
///
/// Parameters:
/// - `n_vocab`: the vocabulary size.
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
/// - `tau`: the target surprise.
/// - `eta`: the learning rate.
/// - `m`: how many top tokens estimate the exponent.
///
/// Return: the sampler.
pub export fn llama_sampler_init_mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) callconv(.c) [*c]Sampler {
    const seed_cur = core.getRngSeed(seed);

    return core.llama_sampler_init(&mirostat_iface, core.create(Mirostat, .{
        .n_vocab = n_vocab,
        .seed = seed,
        .seed_cur = seed_cur,
        .tau = tau,
        .eta = eta,
        .m = m,
        .mu = 2.0 * tau,
        .rng = libcxx.Mt19937.init(seed_cur),
    }));
}

// -----------------------------------------------------------------------------
// mirostat v2

/// Ports `llama_sampler_mirostat_v2` (llama-sampler.cpp:2557 @b29c606e2).
pub const MirostatV2 = struct {
    seed: u32,
    seed_cur: u32,

    tau: f32,
    eta: f32,

    mu: f32,

    rng: libcxx.Mt19937,
};

/// Ports `llama_sampler_mirostat_v2_name` (llama-sampler.cpp:2569 @b29c606e2).
fn mirostatV2Name(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "mirostat-v2";
}

/// Ports `llama_sampler_mirostat_v2_apply` (llama-sampler.cpp:2573 @b29c606e2).
///
/// Cuts at the first candidate whose surprise exceeds `mu`, keeping at least
/// one, draws, and moves `mu` toward `tau`.
fn mirostatV2Apply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(MirostatV2, smpl);

    core.softmaxImpl(cur_p, true);

    var cut: usize = 0;
    while (cut < cur_p.*.size) : (cut += 1) {
        if (-impl.log2f(cur_p.*.data[cut].p) > ctx.mu) break;
    }
    cur_p.*.size = cut;

    if (cur_p.*.size == 0) {
        cur_p.*.size = 1;
    }

    core.softmaxImpl(cur_p, true);

    const idx = core.sampleDist(cur_p, &ctx.rng);

    cur_p.*.selected = idx;

    const observed_surprise = -impl.log2f(cur_p.*.data[@intCast(idx)].p);
    const e = observed_surprise - ctx.tau;

    ctx.mu = @mulAdd(f32, -ctx.eta, e, ctx.mu);
}

/// Ports `llama_sampler_mirostat_v2_reset` (llama-sampler.cpp:2601 @b29c606e2).
fn mirostatV2Reset(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(MirostatV2, smpl);
    ctx.mu = 2.0 * ctx.tau;
    ctx.seed_cur = core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
}

/// Ports `llama_sampler_mirostat_v2_clone` (llama-sampler.cpp:2608 @b29c606e2).
fn mirostatV2Clone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(MirostatV2, smpl);

    const result = llama_sampler_init_mirostat_v2(ctx.seed, ctx.tau, ctx.eta);

    const result_ctx = core.ctxOf(MirostatV2, result);
    result_ctx.mu = ctx.mu;
    result_ctx.rng = ctx.rng;

    return result;
}

/// Ports `llama_sampler_mirostat_v2_free` (llama-sampler.cpp:2624 @b29c606e2).
fn mirostatV2Free(smpl: [*c]Sampler) callconv(.c) void {
    core.destroy(MirostatV2, core.ctxOf(MirostatV2, smpl));
}

/// Ports `llama_sampler_mirostat_v2_i` (llama-sampler.cpp:2628 @b29c606e2).
pub var mirostat_v2_iface: SamplerI = .{
    .name = mirostatV2Name,
    .accept = null,
    .apply = mirostatV2Apply,
    .reset = mirostatV2Reset,
    .clone = mirostatV2Clone,
    .free = mirostatV2Free,
};

/// Ports `llama_sampler_init_mirostat_v2` (llama-sampler.cpp:2643 @b29c606e2).
///
/// Parameters:
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
/// - `tau`: the target surprise.
/// - `eta`: the learning rate.
///
/// Return: the sampler.
pub export fn llama_sampler_init_mirostat_v2(seed: u32, tau: f32, eta: f32) callconv(.c) [*c]Sampler {
    const seed_cur = core.getRngSeed(seed);
    return core.llama_sampler_init(&mirostat_v2_iface, core.create(MirostatV2, .{
        .seed = seed,
        .seed_cur = seed_cur,
        .tau = tau,
        .eta = eta,
        .mu = 2.0 * tau,
        .rng = libcxx.Mt19937.init(seed_cur),
    }));
}

// -----------------------------------------------------------------------------
// adaptive-p

/// Ports `llama_sampler_adaptive_p` (llama-sampler.cpp:3727 @b29c606e2).
const AdaptiveP = struct {
    target: f32,
    decay: f32,
    seed: u32,
    seed_cur: u32,
    rng: libcxx.Mt19937,
    weighted_sum: f32,
    total_weight: f32,
    original_probs: std.ArrayList(f32),
    pending_token_id: Token,
    pending_token_idx: i32,
};

// Ports `DISTRIBUTION_WIDTH`, `PEAK_LOGIT_VALUE`, `SHARPNESS`, `INV_WIDTH`
// (llama-sampler.cpp:3741, 3742, 3743, 3744 @b29c606e2).
const DISTRIBUTION_WIDTH: f32 = 0.3;
const PEAK_LOGIT_VALUE: f32 = 5.0;
const SHARPNESS: f32 = 10.0;
const INV_WIDTH: f32 = 1.0 / DISTRIBUTION_WIDTH;

/// Ports `llama_sampler_adaptive_p_name` (llama-sampler.cpp:3746 @b29c606e2).
fn adaptivePName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "adaptive-p";
}

/// Ports `std::clamp(v, lo, hi)` for floats: `v < lo ? lo : hi < v ? hi : v`.
fn clamp(v: f32, lo: f32, hi: f32) f32 {
    return if (v < lo) lo else if (hi < v) hi else v;
}

/// Ports `llama_sampler_adaptive_p_apply` (llama-sampler.cpp:3750 @b29c606e2).
///
/// Re-weights candidates toward a target probability that adapts to the
/// running average of what was picked, then draws.
fn adaptivePApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(AdaptiveP, smpl);

    core.softmaxImpl(cur_p, false);

    if (ctx.target < 0.0) {
        cur_p.*.selected = core.sampleDist(cur_p, &ctx.rng);
        return;
    }

    ctx.original_probs.resize(impl.allocator, cur_p.*.size) catch impl.abort(src, 3763, "out of memory");
    for (core.items(cur_p), 0..) |d, i| ctx.original_probs.items[i] = d.p;

    const target = clamp(ctx.target, 0.0, 1.0);
    const adapted_target = clamp(
        if (ctx.total_weight == 0.0) target else @mulAdd(f32, target, 2.0, -ctx.weighted_sum / ctx.total_weight),
        0.0,
        1.0,
    );

    for (core.items(cur_p)) |*d| {
        if (d.logit == -std.math.inf(f32)) continue;
        const dist = @abs((d.p - adapted_target) * INV_WIDTH);
        d.logit = PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0 + dist);
    }

    core.softmaxImpl(cur_p, false);
    const idx = core.sampleDist(cur_p, &ctx.rng);
    cur_p.*.selected = idx;

    ctx.pending_token_id = cur_p.*.data[@intCast(idx)].id;
    ctx.pending_token_idx = idx;
}

/// Ports `llama_sampler_adaptive_p_accept` (llama-sampler.cpp:3801 @b29c606e2).
///
/// Folds the selected token's original probability into the average, when
/// the accepted token is the one this sampler picked.
fn adaptivePAccept(smpl: [*c]Sampler, token: Token) callconv(.c) void {
    const ctx = core.ctxOf(AdaptiveP, smpl);
    if (ctx.pending_token_id == token) {
        impl.assert(ctx.pending_token_id != c.LLAMA_TOKEN_NULL, src, 3804, "ctx->pending_token_id != LLAMA_TOKEN_NULL");
        impl.assert(ctx.pending_token_idx != -1, src, 3805, "ctx->pending_token_idx != -1");
        ctx.weighted_sum = @mulAdd(f32, ctx.decay, ctx.weighted_sum, ctx.original_probs.items[@intCast(ctx.pending_token_idx)]);
        ctx.total_weight = @mulAdd(f32, ctx.decay, ctx.total_weight, 1.0);
    }
    ctx.pending_token_id = c.LLAMA_TOKEN_NULL;
    ctx.pending_token_idx = -1;
}

/// Ports `llama_sampler_adaptive_p_reset` (llama-sampler.cpp:3814 @b29c606e2).
fn adaptivePReset(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(AdaptiveP, smpl);
    ctx.weighted_sum = ctx.target / (1.0 - ctx.decay);
    ctx.total_weight = 1.0 / (1.0 - ctx.decay);
    ctx.pending_token_id = c.LLAMA_TOKEN_NULL;
    ctx.pending_token_idx = -1;
    ctx.seed_cur = core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
}

/// Ports `llama_sampler_adaptive_p_clone` (llama-sampler.cpp:3827 @b29c606e2).
///
/// Copies the average and the pending token but not the engine: the clone
/// starts from a freshly seeded one, as the C++'s does.
fn adaptivePClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(AdaptiveP, smpl);
    const result = llama_sampler_init_adaptive_p(ctx.target, ctx.decay, ctx.seed);
    const result_ctx = core.ctxOf(AdaptiveP, result);

    result_ctx.weighted_sum = ctx.weighted_sum;
    result_ctx.total_weight = ctx.total_weight;
    result_ctx.pending_token_id = ctx.pending_token_id;
    result_ctx.pending_token_idx = ctx.pending_token_idx;

    return result;
}

/// Ports `llama_sampler_adaptive_p_free` (llama-sampler.cpp:3841 @b29c606e2).
fn adaptivePFree(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(AdaptiveP, smpl);
    ctx.original_probs.deinit(impl.allocator);
    core.destroy(AdaptiveP, ctx);
}

/// Ports `llama_sampler_adaptive_p_i` (llama-sampler.cpp:3845 @b29c606e2).
var adaptive_p_iface: SamplerI = .{
    .name = adaptivePName,
    .accept = adaptivePAccept,
    .apply = adaptivePApply,
    .reset = adaptivePReset,
    .clone = adaptivePClone,
    .free = adaptivePFree,
};

/// Ports `llama_sampler_init_adaptive_p` (llama-sampler.cpp:3860 @b29c606e2).
///
/// Parameters:
/// - `target`: the target probability; negative makes it a plain draw.
/// - `decay`: the average's decay, clamped to [0, 0.99].
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
///
/// Return: the sampler.
pub export fn llama_sampler_init_adaptive_p(target: f32, decay: f32, seed: u32) callconv(.c) [*c]Sampler {
    const seed_cur = core.getRngSeed(seed);
    const clamped_decay = clamp(decay, 0.0, 0.99);
    return core.llama_sampler_init(&adaptive_p_iface, core.create(AdaptiveP, .{
        .target = target,
        .decay = clamped_decay,
        .seed = seed,
        .seed_cur = seed_cur,
        .rng = libcxx.Mt19937.init(seed_cur),
        .weighted_sum = target / (1.0 - clamped_decay),
        .total_weight = 1.0 / (1.0 - clamped_decay),
        .original_probs = .empty,
        .pending_token_id = c.LLAMA_TOKEN_NULL,
        .pending_token_idx = -1,
    }));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
