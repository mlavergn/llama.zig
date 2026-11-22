//! The truncating and temperature samplers, and the two final pickers.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`):
//! greedy, dist, top-k, top-p, min-p, typical, temp, temp-ext, XTC and
//! top-n-sigma, with the backend graph builders of those that have one. Each
//! declaration below names the C++ it replaces and the line it began at.
//!
//! # Fused multiply-adds
//!
//! clang compiles the C++ with `-ffp-contract=on`, and Zig is strict, so each
//! site clang fused is named here with `@mulAdd`. **Which** sites was read off
//! the reference `llama-sampler.o`'s disassembly, not reasoned about:
//!
//! - typical, `entropy += -p * logf(p)`: `fmsub s15, s8, s0, s15` -- but
//!   only in the loop's tail; clang unrolls it by four and the unrolled
//!   body rounds each product first. See `typicalApply`.
//! - temp-ext, `entropy -= prob * logf(prob)`: `fmsub s13, s9, s0, s13`
//! - temp-ext, `min_temp + (max_temp - min_temp) * powf(...)`:
//!   `fmadd s0, s9, s0, s10`
//! - top-n-sigma, `max - (n * std)`: `fmsub s0, s0, s2, s1`
//!
//! The `fmadd`s in `dist` are libc++'s `generate_canonical`, whose product is
//! exact. `make sampler-diff` is the oracle that confirms each.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

const inf = std.math.inf(f32);

/// Ports `llama_sampler_backend_copy_state` (llama-sampler.cpp:575 @b1b770d90)
/// for the samplers whose only state is immutable configuration:
/// `llama_sampler_backend::copy_state` does nothing.
fn copyStateNone(src_smpl: [*c]const llama.impl.llama_c.struct_llama_sampler, dst_smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    _ = src_smpl;
    _ = dst_smpl;
}

// -----------------------------------------------------------------------------
// greedy

/// Ports `llama_sampler_greedy` (llama-sampler.cpp:1022 @b1b770d90).
const Greedy = struct {
    base: llama.sampler.core.Backend,
};

/// Ports `llama_sampler_greedy_name` (llama-sampler.cpp:1025 @b1b770d90).
fn greedyName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(Greedy, smpl).base.getName();
}

/// Ports `llama_sampler_greedy_reset` (llama-sampler.cpp:1030 @b1b770d90).
fn greedyReset(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    _ = smpl;
}

/// Ports `llama_sampler_greedy_clone` (llama-sampler.cpp:1035 @b1b770d90).
fn greedyClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    _ = smpl;
    return llama_sampler_init_greedy();
}

/// Ports `llama_sampler_greedy_free` (llama-sampler.cpp:1050 @b1b770d90).
fn greedyFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(Greedy, llama.sampler.core.ctxOf(Greedy, smpl));
}

/// Ports `llama_sampler_greedy_apply` (llama-sampler.cpp:1054 @b1b770d90).
///
/// Selects the first maximum.
fn greedyApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    _ = smpl;
    cur_p.*.selected = 0;
    for (1..cur_p.*.size) |i| {
        if (cur_p.*.data[i].logit > cur_p.*.data[@intCast(cur_p.*.selected)].logit) {
            cur_p.*.selected = @intCast(i);
        }
    }
}

/// Ports `llama_sampler_greedy_backend_init` (llama-sampler.cpp:1063 @b1b770d90).
fn greedyBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = n_outputs_max_per_seq;
    const sctx = llama.sampler.core.ctxOf(Greedy, smpl);
    const res = llama.sampler.core.backendSupport(smpl, buft);
    sctx.base.initBackend(res);
    return res;
}

/// Ports `llama_sampler_greedy_backend_apply` (llama-sampler.cpp:1077 @b1b770d90).
///
/// A chain whose earlier samplers left `candidates` -- a
/// filtered list, so the argmax is a position in it -- maps the argmax back
/// to the token id through it.
fn greedyBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    _ = gf;
    _ = smpl;

    const logits = llama.impl.llama_c.ggml_reshape_1d(ctx, data.*.logits, llama.impl.llama_c.ggml_nelements(data.*.logits));

    var curl = llama.impl.llama_c.ggml_argmax(ctx, logits);
    _ = llama.impl.llama_c.ggml_set_name(curl, "greedy_argmax");

    if (data.*.candidates != null) {
        const candidates = llama.impl.llama_c.ggml_reshape_2d(ctx, data.*.candidates, 1, llama.impl.llama_c.ggml_nelements(data.*.candidates));
        curl = llama.impl.llama_c.ggml_get_rows(ctx, candidates, curl);
        _ = llama.impl.llama_c.ggml_set_name(curl, "greedy_sampled_token");
    }

    data.*.sampled = curl;
}

/// Ports `llama_sampler_greedy_i` (llama-sampler.cpp:1099 @b1b770d90).
var greedy_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = greedyName,
    .accept = null,
    .apply = greedyApply,
    .reset = greedyReset,
    .clone = greedyClone,
    .free = greedyFree,
    .backend_init = greedyBackendInit,
    .backend_accept = null,
    .backend_apply = greedyBackendApply,
    .backend_set_input = null,
    .backend_reset = null,
    .copy_state = copyStateNone,
};

/// Ports `llama_sampler_init_greedy` (llama-sampler.cpp:1114 @b1b770d90).
///
/// Return: the sampler.
pub fn llama_sampler_init_greedy() callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    return llama.sampler.core.llama_sampler_init(&greedy_iface, llama.sampler.core.create(Greedy, .{ .base = llama.sampler.core.Backend.init("greedy") }));
}

// -----------------------------------------------------------------------------
// dist

/// Ports `llama_sampler_dist` (llama-sampler.cpp:1125 @b1b770d90).
pub const Dist = struct {
    base: llama.sampler.core.Backend,
    seed: u32,
    seed_cur: u32,

    rng: stdlib.libcxx.Mt19937,

    /// Draws for multi-output backend sampling come from this scratch engine,
    /// so a rejected token does not advance `rng`.
    backend_transactional: bool,
    rng_backend: stdlib.libcxx.Mt19937,
    n_backend_draws_generated: usize,
    n_backend_draws_committed: usize,

    /// The current sampling graph's inputs.
    inp_uniforms: std.ArrayList(*llama.impl.llama_c.struct_ggml_tensor),

    /// Ports `llama_sampler_dist`'s `copy_state` (llama-sampler.cpp:1142 @b1b770d90).
    fn copyState(self: *Dist, other: *const Dist) void {
        // inp_uniforms and backend_transactional belong to the current graph.
        self.seed_cur = other.seed_cur;
        self.rng = other.rng;
        self.rng_backend = other.rng_backend;
        self.n_backend_draws_generated = other.n_backend_draws_generated;
        self.n_backend_draws_committed = other.n_backend_draws_committed;
    }
};

/// Ports `llama_sampler_dist_name` (llama-sampler.cpp:1152 @b1b770d90).
fn distName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(Dist, smpl).base.getName();
}

/// Ports `llama_sampler_dist_apply` (llama-sampler.cpp:1157 @b1b770d90).
///
/// Softmax and the draw in one pass: the probabilities are summed in
/// `double`, the target is that sum times one `uniform<double>` draw, and the
/// first candidate whose running `double` sum reaches it is selected. Each
/// `p` is then divided by the `double` sum and rounded back to `float`.
fn distApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Dist, smpl);

    if (cur_p.*.size == 0) {
        cur_p.*.selected = -1;
        return;
    }

    cur_p.*.selected = 0;

    if (cur_p.*.size == 1) {
        // Keeps the engine aligned with backend sampling, which draws once
        // per output.
        _ = stdlib.libcxx.uniformF64(&ctx.rng);
        cur_p.*.data[0].p = 1.0;
        return;
    }

    var max_l = cur_p.*.data[0].logit;
    if (!cur_p.*.sorted) {
        for (1..cur_p.*.size) |i| max_l = stdlib.libcxx.max(f32, max_l, cur_p.*.data[i].logit);
    }

    var sum_cum: f64 = 0.0;
    for (llama.sampler.core.items(cur_p)) |*d| {
        const p = stdlib.libm.expf(d.logit - max_l);
        d.p = p;
        sum_cum += p;
    }

    const rnd = stdlib.libcxx.uniformF64(&ctx.rng);

    var sum_run: f64 = 0.0;
    const sum_tgt = sum_cum * rnd;

    var found = false;
    for (llama.sampler.core.items(cur_p), 0..) |*d, i| {
        if (!found) {
            sum_run += d.p;
            if (sum_run >= sum_tgt) {
                cur_p.*.selected = @intCast(i);
                found = true;
            }
        }
        d.p = @floatCast(@as(f64, d.p) / sum_cum);
    }

    if (!found) {
        cur_p.*.selected = @intCast(cur_p.*.size - 1);
    }
}

/// Ports `llama_sampler_dist_reset` (llama-sampler.cpp:1232 @b1b770d90).
fn distReset(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Dist, smpl);
    ctx.seed_cur = llama.sampler.core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
    ctx.rng_backend = ctx.rng;
    ctx.n_backend_draws_generated = 0;
    ctx.n_backend_draws_committed = 0;
}

/// Ports `llama_sampler_dist_clone` (llama-sampler.cpp:1241 @b1b770d90).
fn distClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(Dist, smpl);
    const result = llama_sampler_init_dist(ctx.seed);

    const result_ctx = llama.sampler.core.ctxOf(Dist, result);
    result_ctx.seed_cur = ctx.seed_cur;
    result_ctx.rng = ctx.rng;
    result_ctx.backend_transactional = ctx.backend_transactional;
    result_ctx.rng_backend = ctx.rng_backend;
    result_ctx.n_backend_draws_generated = ctx.n_backend_draws_generated;
    result_ctx.n_backend_draws_committed = ctx.n_backend_draws_committed;

    return result;
}

/// Ports `llama_sampler_dist_free` (llama-sampler.cpp:1260 @b1b770d90).
fn distFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Dist, smpl);
    ctx.inp_uniforms.deinit(std.heap.c_allocator);
    llama.sampler.core.destroy(Dist, ctx);
}

/// Ports `llama_sampler_dist_backend_init` (llama-sampler.cpp:1264 @b1b770d90).
fn distBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    const sctx = llama.sampler.core.ctxOf(Dist, smpl);

    const res = llama.sampler.core.backendSupport(smpl, buft);

    sctx.base.initBackend(res);
    sctx.backend_transactional = n_outputs_max_per_seq > 1;
    sctx.rng_backend = sctx.rng;
    sctx.n_backend_draws_generated = 0;
    sctx.n_backend_draws_committed = 0;

    return res;
}

/// Ports `llama_sampler_dist_backend_apply` (llama-sampler.cpp:1281 @b1b770d90).
///
/// Softmax, cumulative sum, and the index where the sum first exceeds a
/// uniform input: counted as the number of steps that are past it.
fn distBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    _ = gf;

    const sctx = llama.sampler.core.ctxOf(Dist, smpl);

    const inp_uniform = llama.impl.llama_c.ggml_new_tensor_1d(ctx, llama.impl.llama_c.GGML_TYPE_F32, 1);
    _ = llama.impl.llama_c.ggml_format_name(inp_uniform, "uniform_%zu", sctx.inp_uniforms.items.len);
    llama.impl.llama_c.ggml_set_input(inp_uniform);
    sctx.inp_uniforms.append(std.heap.c_allocator, inp_uniform) catch llama.impl.abort(llama.sampler.core.src, 1286, "out of memory");

    const logits = llama.impl.llama_c.ggml_reshape_1d(ctx, data.*.logits, llama.impl.llama_c.ggml_nelements(data.*.logits));

    const probs = llama.impl.llama_c.ggml_soft_max(ctx, logits);
    _ = llama.impl.llama_c.ggml_set_name(probs, "dist_probs");

    const cumsum = llama.impl.llama_c.ggml_cumsum(ctx, probs);
    _ = llama.impl.llama_c.ggml_set_name(cumsum, "dist_cumsum");

    const diff = llama.impl.llama_c.ggml_sub(ctx, cumsum, inp_uniform);
    _ = llama.impl.llama_c.ggml_set_name(diff, "dist_cumsum");

    const mask = llama.impl.llama_c.ggml_step(ctx, diff);
    _ = llama.impl.llama_c.ggml_set_name(mask, "dist_mask");

    var idxf = llama.impl.llama_c.ggml_sum(ctx, mask);
    _ = llama.impl.llama_c.ggml_set_name(idxf, "dist_index_f32");

    idxf = llama.impl.llama_c.ggml_clamp(ctx, idxf, 1.0, @floatFromInt(mask[0].ne[0]));

    const idx = llama.impl.llama_c.ggml_cast(ctx, llama.impl.llama_c.ggml_scale_bias(ctx, idxf, -1.0, @floatFromInt(mask[0].ne[0])), llama.impl.llama_c.GGML_TYPE_I32);
    _ = llama.impl.llama_c.ggml_set_name(idx, "dist_index_i32");

    var sampled_token = idx;
    if (data.*.candidates != null) {
        const candidates = llama.impl.llama_c.ggml_reshape_2d(ctx, data.*.candidates, 1, llama.impl.llama_c.ggml_nelements(data.*.candidates));

        sampled_token = llama.impl.llama_c.ggml_get_rows(ctx, candidates, idx);
        _ = llama.impl.llama_c.ggml_set_name(sampled_token, "dist_sampled_token");
    }

    data.*.sampled = sampled_token;
    data.*.probs = probs;
}

/// Ports `llama_sampler_dist_backend_set_input` (llama-sampler.cpp:1345 @b1b770d90).
///
/// One `uniform<double>` draw per graph input, rounded to `float`, so the
/// backend sees the same numbers the CPU path's draws would be.
fn distBackendSetInput(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const sctx = llama.sampler.core.ctxOf(Dist, smpl);

    llama.impl.assert(sctx.inp_uniforms.items.len != 0, llama.sampler.core.src, 1348, "!sctx->inp_uniforms.empty()");

    const rng = if (sctx.backend_transactional) &sctx.rng_backend else &sctx.rng;

    for (sctx.inp_uniforms.items) |inp_uniform| {
        const rnd: f32 = @floatCast(stdlib.libcxx.uniformF64(rng));
        llama.impl.llama_c.ggml_backend_tensor_set(inp_uniform, &rnd, 0, @sizeOf(f32));

        if (sctx.backend_transactional) {
            sctx.n_backend_draws_generated += 1;
        }
    }
}

/// Ports `llama_sampler_dist_backend_reset` (llama-sampler.cpp:1371 @b1b770d90).
fn distBackendReset(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.ctxOf(Dist, smpl).inp_uniforms.clearRetainingCapacity();
}

/// Ports `llama_sampler_dist_accept` (llama-sampler.cpp:1376 @b1b770d90).
///
/// Commits one transactional backend draw to the real engine.
fn distAccept(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, token: llama.impl.llama_c.llama_token) callconv(.c) void {
    _ = token;

    const sctx = llama.sampler.core.ctxOf(Dist, smpl);

    if (!sctx.backend_transactional or sctx.n_backend_draws_committed >= sctx.n_backend_draws_generated) {
        return;
    }

    _ = stdlib.libcxx.uniformF64(&sctx.rng);
    sctx.n_backend_draws_committed += 1;
}

fn distCopyState(src_smpl: [*c]const llama.impl.llama_c.struct_llama_sampler, dst_smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.ctxOf(Dist, dst_smpl).copyState(llama.sampler.core.ctxOf(Dist, src_smpl));
}

/// Ports `llama_sampler_dist_i` (llama-sampler.cpp:1391 @b1b770d90).
pub var dist_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = distName,
    .accept = distAccept,
    .apply = distApply,
    .reset = distReset,
    .clone = distClone,
    .free = distFree,
    .backend_init = distBackendInit,
    .backend_accept = null,
    .backend_apply = distBackendApply,
    .backend_set_input = distBackendSetInput,
    .backend_reset = distBackendReset,
    .copy_state = distCopyState,
};

/// Ports `llama_sampler_init_dist` (llama-sampler.cpp:1406 @b1b770d90).
///
/// Parameters:
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
///
/// Return: the sampler.
pub fn llama_sampler_init_dist(seed: u32) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const seed_cur = llama.sampler.core.getRngSeed(seed);
    return llama.sampler.core.llama_sampler_init(&dist_iface, llama.sampler.core.create(Dist, .{
        .base = llama.sampler.core.Backend.init("dist"),
        .seed = seed,
        .seed_cur = seed_cur,
        .rng = stdlib.libcxx.Mt19937.init(seed_cur),
        .backend_transactional = false,
        .rng_backend = stdlib.libcxx.Mt19937.init(seed_cur),
        .n_backend_draws_generated = 0,
        .n_backend_draws_committed = 0,
        .inp_uniforms = .empty,
    }));
}

// -----------------------------------------------------------------------------
// top-k

/// Ports `llama_sampler_top_k` (llama-sampler.cpp:1447 @b1b770d90).
const TopK = struct {
    base: llama.sampler.core.Backend,
    k: i32,
};

/// Ports `llama_sampler_top_k_name` (llama-sampler.cpp:1451 @b1b770d90).
fn topKName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(TopK, smpl).base.getName();
}

/// Ports `llama_sampler_top_k_apply` (llama-sampler.cpp:1456 @b1b770d90).
fn topKApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    llama.sampler.core.topKImpl(cur_p, llama.sampler.core.ctxOf(TopK, smpl).k);
}

/// Ports `llama_sampler_top_k_clone` (llama-sampler.cpp:1461 @b1b770d90).
fn topKClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    return llama_sampler_init_top_k(llama.sampler.core.ctxOf(TopK, smpl).k);
}

/// Ports `llama_sampler_top_k_free` (llama-sampler.cpp:1466 @b1b770d90).
fn topKFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(TopK, llama.sampler.core.ctxOf(TopK, smpl));
}

/// Ports `llama_sampler_top_k_backend_init` (llama-sampler.cpp:1470 @b1b770d90).
fn topKBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = n_outputs_max_per_seq;
    const sctx = llama.sampler.core.ctxOf(TopK, smpl);
    const res = llama.sampler.core.backendSupport(smpl, buft);
    sctx.base.initBackend(res);
    return res;
}

/// Ports `llama_sampler_top_k_backend_apply` (llama-sampler.cpp:1484 @b1b770d90).
fn topKBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    const sctx = llama.sampler.core.ctxOf(TopK, smpl);

    const logits = llama.impl.llama_c.ggml_reshape_1d(ctx, data.*.logits, llama.impl.llama_c.ggml_nelements(data.*.logits));

    const top_k = llama.impl.llama_c.ggml_top_k(ctx, logits, sctx.k);
    _ = llama.impl.llama_c.ggml_set_name(top_k, "top_k");

    if (data.*.candidates != null) {
        const candidates_rows = llama.impl.llama_c.ggml_reshape_2d(ctx, data.*.candidates, 1, data.*.candidates[0].ne[0]);
        data.*.candidates = llama.impl.llama_c.ggml_get_rows(ctx, candidates_rows, top_k);
        _ = llama.impl.llama_c.ggml_set_name(data.*.candidates, "top_k_candidates");
    } else {
        data.*.candidates = top_k;
    }

    const logits_rows = llama.impl.llama_c.ggml_reshape_2d(ctx, logits, 1, logits[0].ne[0]);
    data.*.logits = llama.impl.llama_c.ggml_get_rows(ctx, logits_rows, top_k);
    _ = llama.impl.llama_c.ggml_set_name(data.*.logits, "top_k_rows");

    _ = gf;
}

/// Ports `llama_sampler_top_k_i` (llama-sampler.cpp:1511 @b1b770d90).
var top_k_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = topKName,
    .accept = null,
    .apply = topKApply,
    .reset = null,
    .clone = topKClone,
    .free = topKFree,
    .backend_init = topKBackendInit,
    .backend_accept = null,
    .backend_apply = topKBackendApply,
    .backend_set_input = null,
    .backend_reset = null,
    .copy_state = copyStateNone,
};

/// Ports `llama_sampler_init_top_k` (llama-sampler.cpp:1526 @b1b770d90).
///
/// Parameters:
/// - `k`: how many candidates to keep; `<= 0` disables it.
///
/// Return: the sampler, or the `?top-k` placeholder when disabled.
pub fn llama_sampler_init_top_k(k: i32) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = k <= 0;

    if (is_empty) return llama.sampler.core.initEmpty("?top-k");

    return llama.sampler.core.llama_sampler_init(&top_k_iface, llama.sampler.core.create(TopK, .{ .base = llama.sampler.core.Backend.init("top-k"), .k = k }));
}

// -----------------------------------------------------------------------------
// top-p

/// Ports `llama_sampler_top_p` (llama-sampler.cpp:1544 @b1b770d90).
const TopP = struct {
    base: llama.sampler.core.Backend,
    p: f32,
    min_keep: usize,
    buf_sort: std.ArrayList(llama.impl.llama_c.llama_token_data),
};

/// Ports `llama_sampler_top_p_name` (llama-sampler.cpp:1551 @b1b770d90).
fn topPName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(TopP, smpl).base.getName();
}

/// Ports `llama_sampler_top_p_apply` (llama-sampler.cpp:1556 @b1b770d90).
///
/// Over 1024 unsorted candidates it first tries a 256-wide partial sort, and
/// widens to a full one only when the cumulative probability has not reached
/// `p` by then.
fn topPApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(TopP, smpl);

    if (ctx.p >= 1.0) return;

    llama.sampler.core.softmaxImpl(cur_p, false);

    var k: usize = cur_p.*.size;
    var pdata: [*c]llama.impl.llama_c.llama_token_data = cur_p.*.data;

    const buf_sort = &ctx.buf_sort;

    if (!cur_p.*.sorted and cur_p.*.size > 1024) {
        k = stdlib.libcxx.min(usize, 256, cur_p.*.size);
        llama.sampler.core.partialSortInto(cur_p, @intCast(k), buf_sort);
        pdata = buf_sort.items.ptr;
    } else if (!cur_p.*.sorted) {
        llama.sampler.core.partialSortInplace(cur_p, @intCast(k));
    }

    var cum_sum: f32 = 0.0;
    var last_idx: usize = cur_p.*.size;

    for (0..cur_p.*.size) |i| {
        cum_sum += pdata[i].p;

        if (cum_sum >= ctx.p and i + 1 >= ctx.min_keep) {
            last_idx = i + 1;
            break;
        }

        if (!cur_p.*.sorted and i == k - 1) {
            k = cur_p.*.size;
            llama.sampler.core.partialSortInto(cur_p, @intCast(k), buf_sort);
            pdata = buf_sort.items.ptr;
        }
    }

    if (!cur_p.*.sorted) {
        @memcpy(cur_p.*.data[0..last_idx], buf_sort.items[0..last_idx]);
        cur_p.*.sorted = true;
    }

    cur_p.*.size = last_idx;
}

/// Ports `llama_sampler_top_p_clone` (llama-sampler.cpp:1611 @b1b770d90).
fn topPClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(TopP, smpl);
    return llama_sampler_init_top_p(ctx.p, ctx.min_keep);
}

/// Ports `llama_sampler_top_p_free` (llama-sampler.cpp:1616 @b1b770d90).
fn topPFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(TopP, smpl);
    ctx.buf_sort.deinit(std.heap.c_allocator);
    llama.sampler.core.destroy(TopP, ctx);
}

/// Ports `llama_sampler_top_p_backend_init` (llama-sampler.cpp:1620 @b1b770d90).
fn topPBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = n_outputs_max_per_seq;
    const sctx = llama.sampler.core.ctxOf(TopP, smpl);
    const res = llama.sampler.core.backendSupport(smpl, buft);
    sctx.base.initBackend(res);
    return res;
}

/// The `ggml_sort` lambda (llama-sampler.cpp:1644 @b1b770d90) in
/// `llama_sampler_top_p_backend_apply`: a 1-row tensor reordered by indices.
fn ggmlSort(ctx: ?*llama.impl.llama_c.struct_ggml_context, a: [*c]llama.impl.llama_c.struct_ggml_tensor, b: [*c]llama.impl.llama_c.struct_ggml_tensor) [*c]llama.impl.llama_c.struct_ggml_tensor {
    llama.impl.assert(llama.impl.llama_c.ggml_nrows(a) == 1, llama.sampler.core.src, 1645, "ggml_nrows(a) == 1");
    const a_reshaped = llama.impl.llama_c.ggml_reshape_2d(ctx, a, 1, a[0].ne[0]);
    return llama.impl.llama_c.ggml_get_rows(ctx, a_reshaped, b);
}

/// Ports `llama_sampler_top_p_backend_apply` (llama-sampler.cpp:1634 @b1b770d90).
fn topPBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    const sctx = llama.sampler.core.ctxOf(TopP, smpl);

    const logits = llama.impl.llama_c.ggml_reshape_1d(ctx, data.*.logits, llama.impl.llama_c.ggml_nelements(data.*.logits));

    const sorted_idx = llama.impl.llama_c.ggml_argsort(ctx, logits, llama.impl.llama_c.GGML_SORT_ORDER_DESC);
    _ = llama.impl.llama_c.ggml_set_name(sorted_idx, "top_p_sorted_idx");

    var sorted_logits = ggmlSort(ctx, logits, sorted_idx);
    _ = llama.impl.llama_c.ggml_set_name(sorted_logits, "top_p_sorted_logits");

    sorted_logits = llama.impl.llama_c.ggml_reshape_1d(ctx, sorted_logits, llama.impl.llama_c.ggml_nelements(sorted_logits));
    const softmax = llama.impl.llama_c.ggml_soft_max(ctx, sorted_logits);
    _ = llama.impl.llama_c.ggml_set_name(softmax, "top_p_softmax");

    if (data.*.candidates != null) {
        data.*.candidates = ggmlSort(ctx, data.*.candidates, sorted_idx);
    } else {
        data.*.candidates = sorted_idx;
    }
    _ = llama.impl.llama_c.ggml_set_name(data.*.candidates, "top_p_candidates");

    const cdf = llama.impl.llama_c.ggml_cumsum(ctx, softmax);
    _ = llama.impl.llama_c.ggml_set_name(cdf, "top_p_cdf");

    const cdf_scaled = llama.impl.llama_c.ggml_scale_bias(ctx, cdf, -1.0, sctx.p);
    _ = llama.impl.llama_c.ggml_set_name(cdf_scaled, "top_p_cdf_scaled");

    var mask = llama.impl.llama_c.ggml_step(ctx, cdf_scaled);
    _ = llama.impl.llama_c.ggml_set_name(mask, "top_p_mask");

    var idxf = llama.impl.llama_c.ggml_sum(ctx, mask);
    _ = llama.impl.llama_c.ggml_set_name(idxf, "top_p_index_f32");

    idxf = llama.impl.llama_c.ggml_clamp(ctx, idxf, 0.0, @floatFromInt(mask[0].ne[0] - 1));

    const ones = llama.impl.llama_c.ggml_scale_bias(ctx, idxf, 0.0, 1.0);
    _ = llama.impl.llama_c.ggml_set_name(ones, "top_p_ones");

    var mask_reshaped = llama.impl.llama_c.ggml_reshape_2d(ctx, mask, 1, mask[0].ne[0]);

    mask_reshaped = llama.impl.llama_c.ggml_set_rows(ctx, mask_reshaped, ones, llama.impl.llama_c.ggml_cast(ctx, idxf, llama.impl.llama_c.GGML_TYPE_I32));
    mask = llama.impl.llama_c.ggml_reshape_1d(ctx, mask_reshaped, mask[0].ne[0]);

    const top_p_bias = llama.impl.llama_c.ggml_log(ctx, mask);
    _ = llama.impl.llama_c.ggml_set_name(top_p_bias, "top_p_bias");

    data.*.logits = llama.impl.llama_c.ggml_add(ctx, sorted_logits, top_p_bias);
    _ = llama.impl.llama_c.ggml_set_name(data.*.logits, "top_p_logits");

    _ = gf;
}

/// Ports `llama_sampler_top_p_i` (llama-sampler.cpp:1711 @b1b770d90).
var top_p_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = topPName,
    .accept = null,
    .apply = topPApply,
    .reset = null,
    .clone = topPClone,
    .free = topPFree,
    .backend_init = topPBackendInit,
    .backend_accept = null,
    .backend_apply = topPBackendApply,
    .backend_set_input = null,
    .backend_reset = null,
    .copy_state = copyStateNone,
};

/// Ports `llama_sampler_init_top_p` (llama-sampler.cpp:1726 @b1b770d90).
///
/// Parameters:
/// - `p`: the cumulative probability to keep; `>= 1` disables it.
/// - `min_keep`: the fewest candidates to leave.
///
/// Return: the sampler, or the `?top-p` placeholder when disabled.
pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = p >= 1.0;

    if (is_empty) return llama.sampler.core.initEmpty("?top-p");

    return llama.sampler.core.llama_sampler_init(&top_p_iface, llama.sampler.core.create(TopP, .{
        .base = llama.sampler.core.Backend.init("top-p"),
        .p = p,
        .min_keep = min_keep,
        .buf_sort = .empty,
    }));
}

// -----------------------------------------------------------------------------
// min-p

/// Ports `llama_sampler_min_p` (llama-sampler.cpp:1746 @b1b770d90).
const MinP = struct {
    base: llama.sampler.core.Backend,
    p: f32,
    min_keep: usize,
};

/// Ports `llama_sampler_min_p_name` (llama-sampler.cpp:1751 @b1b770d90).
fn minPName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(MinP, smpl).base.getName();
}

/// Ports `llama_sampler_min_p_apply` (llama-sampler.cpp:1756 @b1b770d90).
///
/// Keeps the candidates whose logit is within `log(p)` of the best. Unsorted
/// input is filtered in place order first; only when that leaves fewer than
/// `min_keep` does it sort and cut.
fn minPApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(MinP, smpl);

    if (ctx.p <= 0.0 or cur_p.*.size == 0) return;

    var min_p_applied = false;

    if (!cur_p.*.sorted) {
        var filtered_tokens: std.ArrayList(llama.impl.llama_c.llama_token_data) = .empty;
        defer filtered_tokens.deinit(std.heap.c_allocator);

        var max_logit: f32 = -std.math.floatMax(f32);
        for (llama.sampler.core.items(cur_p)) |d| max_logit = stdlib.libcxx.max(f32, max_logit, d.logit);
        const min_logit = max_logit + stdlib.libm.logf(ctx.p);

        for (llama.sampler.core.items(cur_p)) |d| {
            if (d.logit >= min_logit) {
                filtered_tokens.append(std.heap.c_allocator, d) catch llama.impl.abort(llama.sampler.core.src, 1770, "out of memory");
            }
        }

        if (filtered_tokens.items.len != 0 and filtered_tokens.items.len >= ctx.min_keep) {
            @memcpy(cur_p.*.data[0..filtered_tokens.items.len], filtered_tokens.items);
            cur_p.*.size = filtered_tokens.items.len;
            min_p_applied = true;
        }
    }

    if (!min_p_applied) {
        if (!cur_p.*.sorted) {
            llama.sampler.core.partialSortInplace(cur_p, @intCast(cur_p.*.size));
        }

        const min_logit = cur_p.*.data[0].logit + stdlib.libm.logf(ctx.p);
        var i: usize = 1;

        while (i < cur_p.*.size) : (i += 1) {
            if (cur_p.*.data[i].logit < min_logit and i >= ctx.min_keep) break;
        }

        cur_p.*.size = i;
    }
}

/// Ports `llama_sampler_min_p_clone` (llama-sampler.cpp:1810 @b1b770d90).
fn minPClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(MinP, smpl);
    return llama_sampler_init_min_p(ctx.p, ctx.min_keep);
}

/// Ports `llama_sampler_min_p_free` (llama-sampler.cpp:1815 @b1b770d90).
fn minPFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(MinP, llama.sampler.core.ctxOf(MinP, smpl));
}

/// Ports `llama_sampler_min_p_backend_init` (llama-sampler.cpp:1819 @b1b770d90).
fn minPBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = n_outputs_max_per_seq;
    const sctx = llama.sampler.core.ctxOf(MinP, smpl);
    const res = llama.sampler.core.backendSupport(smpl, buft);
    sctx.base.initBackend(res);
    return res;
}

/// Ports `llama_sampler_min_p_backend_apply` (llama-sampler.cpp:1833 @b1b770d90).
fn minPBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    const sctx = llama.sampler.core.ctxOf(MinP, smpl);

    const logits = llama.impl.llama_c.ggml_reshape_1d(ctx, data.*.logits, llama.impl.llama_c.ggml_nelements(data.*.logits));

    const max_idx = llama.impl.llama_c.ggml_argmax(ctx, logits);
    _ = llama.impl.llama_c.ggml_set_name(max_idx, "max_idx");

    const logits_rows = llama.impl.llama_c.ggml_reshape_2d(ctx, logits, 1, logits[0].ne[0]);
    _ = llama.impl.llama_c.ggml_set_name(logits_rows, "logits_rows");

    const max_logit = llama.impl.llama_c.ggml_get_rows(ctx, logits_rows, max_idx);
    _ = llama.impl.llama_c.ggml_set_name(max_logit, "max_logit");

    const threshold = llama.impl.llama_c.ggml_scale_bias(ctx, max_logit, 1.0, stdlib.libm.logf(sctx.p));
    _ = llama.impl.llama_c.ggml_set_name(threshold, "min_p_threshold");

    const sub = llama.impl.llama_c.ggml_sub(ctx, logits, threshold);

    const mask = llama.impl.llama_c.ggml_step(ctx, sub);
    _ = llama.impl.llama_c.ggml_set_name(mask, "min_p_mask");

    const min_p_bias = llama.impl.llama_c.ggml_log(ctx, mask);
    _ = llama.impl.llama_c.ggml_set_name(min_p_bias, "min_p_bias");

    data.*.logits = llama.impl.llama_c.ggml_add(ctx, logits, min_p_bias);
    _ = llama.impl.llama_c.ggml_set_name(data.*.logits, "min_p_logits");

    _ = gf;
}

/// Ports `llama_sampler_min_p_i` (llama-sampler.cpp:1874 @b1b770d90).
var min_p_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = minPName,
    .accept = null,
    .apply = minPApply,
    .reset = null,
    .clone = minPClone,
    .free = minPFree,
    .backend_init = minPBackendInit,
    .backend_accept = null,
    .backend_apply = minPBackendApply,
    .backend_set_input = null,
    .backend_reset = null,
    .copy_state = copyStateNone,
};

/// Ports `llama_sampler_init_min_p` (llama-sampler.cpp:1889 @b1b770d90).
///
/// Parameters:
/// - `p`: the probability floor, relative to the best; `<= 0` disables it.
/// - `min_keep`: the fewest candidates to leave.
///
/// Return: the sampler, or the `?min-p` placeholder when disabled.
pub fn llama_sampler_init_min_p(p: f32, min_keep: usize) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = p <= 0.0;

    if (is_empty) return llama.sampler.core.initEmpty("?min-p");

    return llama.sampler.core.llama_sampler_init(&min_p_iface, llama.sampler.core.create(MinP, .{
        .base = llama.sampler.core.Backend.init("min-p"),
        .p = p,
        .min_keep = min_keep,
    }));
}

// -----------------------------------------------------------------------------
// typical

/// Ports `llama_sampler_typical` (llama-sampler.cpp:1908 @b1b770d90).
const Typical = struct {
    p: f32,
    min_keep: usize,
};

/// Ports `llama_sampler_typical_name` (llama-sampler.cpp:1913 @b1b770d90).
fn typicalName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "typical";
}

const ScoreCtx = struct { scores: []const f32 };

/// The comparator of the `sort` (llama-sampler.cpp:1945 @b1b770d90) in
/// `llama_sampler_typical_apply`.
fn lessShiftedScore(ctx: ScoreCtx, a: usize, b: usize) bool {
    return ctx.scores[a] < ctx.scores[b];
}

/// Ports `llama_sampler_typical_apply` (llama-sampler.cpp:1917 @b1b770d90).
///
/// Keeps the candidates whose surprisal is closest to the distribution's
/// entropy. The entropy sum is fused -- see the file header.
fn typicalApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Typical, smpl);

    if (ctx.p >= 1.0) return;

    llama.sampler.core.softmaxImpl(cur_p, true);

    // clang unrolls the loop summing `entropy` (llama-sampler.cpp:1931 @b1b770d90)
    // by four: in the
    // unrolled body each `p * logf(p)` is rounded and then subtracted, and
    // only the `size % 4` left over run through the fused `fmsub`. Below
    // four candidates every term is fused. Read off the reference object --
    // `and x23, x20, #~3`, four `fmul`/`fsub` pairs, then the `fmsub` tail
    // -- and confirmed by `make sampler-diff`, which saw tied shifted scores
    // swap order at 4096 candidates with the loop fused throughout.
    var entropy: f32 = 0.0;
    const n_unrolled = if (cur_p.*.size >= 4) cur_p.*.size & ~@as(usize, 3) else 0;
    for (llama.sampler.core.items(cur_p), 0..) |d, i| {
        if (i < n_unrolled) {
            entropy = entropy - d.p * stdlib.libm.logf(d.p);
        } else {
            entropy = @mulAdd(f32, -d.p, stdlib.libm.logf(d.p), entropy);
        }
    }

    const n = cur_p.*.size;
    const shifted_scores = std.heap.c_allocator.alloc(f32, n) catch llama.impl.abort(llama.sampler.core.src, 1928, "out of memory");
    defer std.heap.c_allocator.free(shifted_scores);
    for (llama.sampler.core.items(cur_p), 0..) |d, i| {
        shifted_scores[i] = @abs(-stdlib.libm.logf(d.p) - entropy);
    }

    const indices = std.heap.c_allocator.alloc(usize, n) catch llama.impl.abort(llama.sampler.core.src, 1935, "out of memory");
    defer std.heap.c_allocator.free(indices);
    for (indices, 0..) |*v, i| v.* = i;

    stdlib.libcxx.sort(usize, indices, ScoreCtx{ .scores = shifted_scores }, lessShiftedScore);

    var cum_sum: f32 = 0.0;
    var last_idx: usize = indices.len;

    for (indices, 0..) |idx, i| {
        cum_sum += cur_p.*.data[idx].p;

        if (cum_sum > ctx.p and (ctx.min_keep == 0 or i >= ctx.min_keep - 1)) {
            last_idx = i + 1;
            break;
        }
    }

    const cur_p_new = std.heap.c_allocator.alloc(llama.impl.llama_c.llama_token_data, last_idx) catch llama.impl.abort(llama.sampler.core.src, 1958, "out of memory");
    defer std.heap.c_allocator.free(cur_p_new);
    for (0..last_idx) |i| cur_p_new[i] = cur_p.*.data[indices[i]];

    @memcpy(cur_p.*.data[0..last_idx], cur_p_new);
    cur_p.*.size = last_idx;
    cur_p.*.sorted = false;
}

/// Ports `llama_sampler_typical_clone` (llama-sampler.cpp:1977 @b1b770d90).
fn typicalClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(Typical, smpl);
    return llama_sampler_init_typical(ctx.p, ctx.min_keep);
}

/// Ports `llama_sampler_typical_free` (llama-sampler.cpp:1982 @b1b770d90).
fn typicalFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(Typical, llama.sampler.core.ctxOf(Typical, smpl));
}

/// Ports `llama_sampler_typical_i` (llama-sampler.cpp:1986 @b1b770d90).
var typical_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = typicalName,
    .accept = null,
    .apply = typicalApply,
    .reset = null,
    .clone = typicalClone,
    .free = typicalFree,
};

/// Ports `llama_sampler_init_typical` (llama-sampler.cpp:2001 @b1b770d90).
///
/// Parameters:
/// - `p`: the cumulative probability to keep; `>= 1` disables it.
/// - `min_keep`: the fewest candidates to leave.
///
/// Return: the sampler, or the `?typical` placeholder when disabled.
pub fn llama_sampler_init_typical(p: f32, min_keep: usize) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = p >= 1.0;

    if (is_empty) return llama.sampler.core.initEmpty("?typical");

    return llama.sampler.core.llama_sampler_init(&typical_iface, llama.sampler.core.create(Typical, .{ .p = p, .min_keep = min_keep }));
}

// -----------------------------------------------------------------------------
// temp

/// Ports `llama_sampler_temp` (llama-sampler.cpp:2019 @b1b770d90).
const Temp = struct {
    base: llama.sampler.core.Backend,
    temp: f32,
};

/// Ports `llama_sampler_temp_name` (llama-sampler.cpp:2023 @b1b770d90).
fn tempName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(Temp, smpl).base.getName();
}

/// Ports `llama_sampler_temp_apply` (llama-sampler.cpp:2028 @b1b770d90).
fn tempApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    llama.sampler.core.tempImpl(cur_p, llama.sampler.core.ctxOf(Temp, smpl).temp);
}

/// Ports `llama_sampler_temp_clone` (llama-sampler.cpp:2034 @b1b770d90).
fn tempClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    return llama_sampler_init_temp(llama.sampler.core.ctxOf(Temp, smpl).temp);
}

/// Ports `llama_sampler_temp_free` (llama-sampler.cpp:2039 @b1b770d90).
fn tempFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(Temp, llama.sampler.core.ctxOf(Temp, smpl));
}

/// Ports `llama_sampler_backend_temp_sampling` (llama-sampler.cpp:2043 @b1b770d90).
///
/// At `temp <= 0` the backend keeps the argmax alone; otherwise it scales.
fn backendTempSampling(ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data, temp: f32) void {
    if (temp <= 0.0) {
        const logits = llama.impl.llama_c.ggml_reshape_1d(ctx, data.*.logits, llama.impl.llama_c.ggml_nelements(data.*.logits));

        const max_idx = llama.impl.llama_c.ggml_argmax(ctx, logits);
        _ = llama.impl.llama_c.ggml_set_name(max_idx, "temp_max_idx");

        if (data.*.candidates != null) {
            const candidates_rows = llama.impl.llama_c.ggml_reshape_2d(ctx, data.*.candidates, 1, llama.impl.llama_c.ggml_nelements(data.*.candidates));
            data.*.candidates = llama.impl.llama_c.ggml_get_rows(ctx, candidates_rows, max_idx);
        } else {
            data.*.candidates = max_idx;
        }

        const logits_rows = llama.impl.llama_c.ggml_reshape_2d(ctx, logits, 1, llama.impl.llama_c.ggml_nelements(logits));
        data.*.logits = llama.impl.llama_c.ggml_get_rows(ctx, logits_rows, max_idx);

        return;
    }

    data.*.logits = llama.impl.llama_c.ggml_scale(ctx, data.*.logits, 1.0 / temp);

    _ = gf;
}

/// Ports `llama_sampler_temp_backend_init` (llama-sampler.cpp:2073 @b1b770d90).
fn tempBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = n_outputs_max_per_seq;
    const sctx = llama.sampler.core.ctxOf(Temp, smpl);
    const res = llama.sampler.core.backendSupport(smpl, buft);
    sctx.base.initBackend(res);
    return res;
}

/// Ports `llama_sampler_temp_backend_apply` (llama-sampler.cpp:2087 @b1b770d90).
fn tempBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    backendTempSampling(ctx, gf, data, llama.sampler.core.ctxOf(Temp, smpl).temp);
}

/// Ports `llama_sampler_temp_i` (llama-sampler.cpp:2096 @b1b770d90).
var temp_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = tempName,
    .accept = null,
    .apply = tempApply,
    .reset = null,
    .clone = tempClone,
    .free = tempFree,
    .backend_init = tempBackendInit,
    .backend_accept = null,
    .backend_apply = tempBackendApply,
    .backend_set_input = null,
    .backend_reset = null,
    .copy_state = copyStateNone,
};

/// Ports `llama_sampler_init_temp` (llama-sampler.cpp:2111 @b1b770d90).
///
/// Parameters:
/// - `temp`: the temperature; exactly 1 disables it.
///
/// Return: the sampler, or the `?temp` placeholder when disabled.
pub fn llama_sampler_init_temp(temp: f32) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = temp == 1.0;

    if (is_empty) return llama.sampler.core.initEmpty("?temp");

    return llama.sampler.core.llama_sampler_init(&temp_iface, llama.sampler.core.create(Temp, .{ .base = llama.sampler.core.Backend.init("temp"), .temp = temp }));
}

// -----------------------------------------------------------------------------
// temp-ext

/// Ports `llama_sampler_temp_ext` (llama-sampler.cpp:2129 @b1b770d90).
const TempExt = struct {
    base: llama.sampler.core.Backend,
    temp: f32,
    delta: f32,
    exponent: f32,
};

/// Ports `llama_sampler_temp_ext_name` (llama-sampler.cpp:2135 @b1b770d90).
fn tempExtName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(TempExt, smpl).base.getName();
}

/// Ports `llama_sampler_temp_ext_apply` (llama-sampler.cpp:2140 @b1b770d90).
///
/// Dynamic temperature: the softmax entropy, normalised by its maximum,
/// picks a temperature within `temp ± delta`. The entropy sum and the
/// temperature mapping are fused -- see the file header -- and the
/// re-normalisation after scaling runs in `double`.
fn tempExtApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(TempExt, smpl);
    if (ctx.delta > 0) {
        const min_temp = stdlib.libcxx.max(f32, 0.0, ctx.temp - ctx.delta);
        const max_temp = ctx.temp + ctx.delta;

        const exponent_val = ctx.exponent;

        if (cur_p.*.size <= 1) return;

        const max_entropy = -stdlib.libm.logf(1.0 / @as(f32, @floatFromInt(cur_p.*.size)));

        llama.sampler.core.softmaxImpl(cur_p, true);

        var entropy: f32 = 0.0;
        for (llama.sampler.core.items(cur_p)) |d| {
            const prob = d.p;
            if (prob > 0.0) {
                entropy = @mulAdd(f32, -prob, stdlib.libm.logf(prob), entropy);
            }
        }

        const normalized_entropy = entropy / max_entropy;

        const dyn_temp = @mulAdd(f32, max_temp - min_temp, stdlib.libm.powf(normalized_entropy, exponent_val), min_temp);

        llama.sampler.core.tempImpl(cur_p, dyn_temp);

        const max_l_double: f64 = cur_p.*.data[0].logit;

        var cum_sum_double: f64 = 0.0;
        for (llama.sampler.core.items(cur_p)) |*d| {
            const p = stdlib.libm.exp(@as(f64, d.logit) - max_l_double);
            d.p = @floatCast(p);
            cum_sum_double += p;
        }

        for (llama.sampler.core.items(cur_p)) |*d| {
            d.p = @floatCast(@as(f64, d.p) / cum_sum_double);
        }
    } else {
        llama.sampler.core.tempImpl(cur_p, ctx.temp);
    }
}

/// Ports `llama_sampler_temp_ext_clone` (llama-sampler.cpp:2211 @b1b770d90).
fn tempExtClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(TempExt, smpl);
    return llama_sampler_init_temp_ext(ctx.temp, ctx.delta, ctx.exponent);
}

/// Ports `llama_sampler_temp_ext_free` (llama-sampler.cpp:2216 @b1b770d90).
fn tempExtFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(TempExt, llama.sampler.core.ctxOf(TempExt, smpl));
}

/// Ports `llama_sampler_temp_ext_backend_init` (llama-sampler.cpp:2220 @b1b770d90).
fn tempExtBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = n_outputs_max_per_seq;
    const sctx = llama.sampler.core.ctxOf(TempExt, smpl);
    const res = llama.sampler.core.backendSupport(smpl, buft);
    sctx.base.initBackend(res);
    return res;
}

/// Ports `llama_sampler_temp_ext_backend_apply` (llama-sampler.cpp:2234 @b1b770d90).
fn tempExtBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    const sctx = llama.sampler.core.ctxOf(TempExt, smpl);

    if (sctx.delta <= 0.0 or sctx.temp <= 0.0) {
        backendTempSampling(ctx, gf, data, sctx.temp);
        return;
    }

    const logits = llama.impl.llama_c.ggml_reshape_1d(ctx, data.*.logits, llama.impl.llama_c.ggml_nelements(data.*.logits));

    const min_temp = stdlib.libcxx.max(f32, 0.0, sctx.temp - sctx.delta);
    const max_temp = sctx.temp + sctx.delta;
    const max_entropy = stdlib.libm.logf(@floatFromInt(logits[0].ne[0]));

    const probs = llama.impl.llama_c.ggml_soft_max(ctx, logits);
    _ = llama.impl.llama_c.ggml_set_name(probs, "temp_ext_softmax_probs");

    const probs_clamped = llama.impl.llama_c.ggml_clamp(ctx, probs, 1e-10, 1.0);
    _ = llama.impl.llama_c.ggml_set_name(probs_clamped, "temp_ext_probs_clamped");

    const log_probs = llama.impl.llama_c.ggml_log(ctx, probs_clamped);
    const p_log_p = llama.impl.llama_c.ggml_mul(ctx, probs_clamped, log_probs);
    const sum_p_log_p = llama.impl.llama_c.ggml_sum(ctx, p_log_p);
    const entropy = llama.impl.llama_c.ggml_scale(ctx, sum_p_log_p, -1.0);
    _ = llama.impl.llama_c.ggml_set_name(log_probs, "temp_ext_log_probs");
    _ = llama.impl.llama_c.ggml_set_name(p_log_p, "temp_ext_p_log_p");
    _ = llama.impl.llama_c.ggml_set_name(sum_p_log_p, "temp_ext_sum_p_log_p");
    _ = llama.impl.llama_c.ggml_set_name(entropy, "temp_ext_entropy");

    const norm_entropy = llama.impl.llama_c.ggml_scale(ctx, entropy, 1.0 / max_entropy);
    _ = llama.impl.llama_c.ggml_set_name(norm_entropy, "temp_ext_norm_entropy");

    const log_norm_entropy = llama.impl.llama_c.ggml_log(ctx, norm_entropy);
    const scaled_log = llama.impl.llama_c.ggml_scale(ctx, log_norm_entropy, sctx.exponent);
    const pow_entropy = llama.impl.llama_c.ggml_exp(ctx, scaled_log);
    const dyn_temp = llama.impl.llama_c.ggml_scale_bias(ctx, pow_entropy, max_temp - min_temp, min_temp);
    _ = llama.impl.llama_c.ggml_set_name(log_norm_entropy, "temp_ext_log_norm_entropy");
    _ = llama.impl.llama_c.ggml_set_name(scaled_log, "temp_ext_scaled_log");
    _ = llama.impl.llama_c.ggml_set_name(pow_entropy, "temp_ext_pow_entropy");
    _ = llama.impl.llama_c.ggml_set_name(dyn_temp, "temp_ext_dyn_temp");

    const scaled_logits = llama.impl.llama_c.ggml_div(ctx, logits, dyn_temp);
    _ = llama.impl.llama_c.ggml_set_name(scaled_logits, "temp_ext_scaled_logits");

    data.*.logits = scaled_logits;
}

/// Ports `llama_sampler_temp_ext_i` (llama-sampler.cpp:2299 @b1b770d90).
var temp_ext_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = tempExtName,
    .accept = null,
    .apply = tempExtApply,
    .reset = null,
    .clone = tempExtClone,
    .free = tempExtFree,
    .backend_init = tempExtBackendInit,
    .backend_accept = null,
    .backend_apply = tempExtBackendApply,
    .backend_set_input = null,
    .backend_reset = null,
    .copy_state = copyStateNone,
};

/// Ports `llama_sampler_init_temp_ext` (llama-sampler.cpp:2314 @b1b770d90).
///
/// Parameters:
/// - `temp`: the base temperature.
/// - `delta`: the dynamic range; `<= 0` makes this plain temperature.
/// - `exponent`: shapes the entropy-to-temperature mapping.
///
/// Return: the sampler, or the `?temp-ext` placeholder at `temp == 1` with no
/// range.
pub fn llama_sampler_init_temp_ext(temp: f32, delta: f32, exponent: f32) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = temp == 1.0 and delta <= 0.0;

    if (is_empty) return llama.sampler.core.initEmpty("?temp-ext");

    return llama.sampler.core.llama_sampler_init(&temp_ext_iface, llama.sampler.core.create(TempExt, .{
        .base = llama.sampler.core.Backend.init("temp-ext"),
        .temp = temp,
        .delta = delta,
        .exponent = exponent,
    }));
}

// -----------------------------------------------------------------------------
// xtc

/// Ports `llama_sampler_xtc` (llama-sampler.cpp:2336 @b1b770d90).
const Xtc = struct {
    probability: f32,
    threshold: f32,
    min_keep: usize,

    seed: u32,
    seed_cur: u32,

    rng: stdlib.libcxx.Mt19937,
};

/// Ports `llama_sampler_xtc_name` (llama-sampler.cpp:2347 @b1b770d90).
fn xtcName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "xtc";
}

/// Ports `llama_sample_xtc_apply` (llama-sampler.cpp:2351 @b1b770d90).
///
/// With probability `probability`, drops every top candidate above
/// `threshold` except the last of them. The draw is `uniform<float>`: one
/// engine step, which can return exactly 1.0 -- see `libcxx.canonicalF32`.
fn xtcApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Xtc, smpl);

    if (ctx.probability <= 0.0 or ctx.threshold > 0.5 or cur_p.*.size < 2) return;

    const chance = stdlib.libcxx.uniformF32(&ctx.rng);
    if (chance > ctx.probability) return;

    llama.sampler.core.softmaxImpl(cur_p, true);

    var pos_last: c_int = 0;

    for (llama.sampler.core.items(cur_p), 0..) |d, i| {
        if (d.p >= ctx.threshold) {
            pos_last = @intCast(i);
        } else break;
    }

    if (cur_p.*.size - @as(usize, @intCast(pos_last)) >= ctx.min_keep and pos_last > 0) {
        cur_p.*.data += @as(usize, @intCast(pos_last));
        cur_p.*.size -= @intCast(pos_last);
    }
}

/// Ports `llama_sampler_xtc_clone` (llama-sampler.cpp:2384 @b1b770d90).
fn xtcClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(Xtc, smpl);
    const result = llama_sampler_init_xtc(ctx.probability, ctx.threshold, ctx.min_keep, ctx.seed);

    llama.sampler.core.ctxOf(Xtc, result).rng = ctx.rng;

    return result;
}

/// Ports `llama_sampler_xtc_free` (llama-sampler.cpp:2398 @b1b770d90).
fn xtcFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(Xtc, llama.sampler.core.ctxOf(Xtc, smpl));
}

/// Ports `llama_sampler_xtc_reset` (llama-sampler.cpp:2402 @b1b770d90).
fn xtcReset(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(Xtc, smpl);
    ctx.seed_cur = llama.sampler.core.getRngSeed(ctx.seed);
    ctx.rng.seed(ctx.seed_cur);
}

/// Ports `llama_sampler_xtc_i` (llama-sampler.cpp:2408 @b1b770d90).
var xtc_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = xtcName,
    .accept = null,
    .apply = xtcApply,
    .reset = xtcReset,
    .clone = xtcClone,
    .free = xtcFree,
};

/// Ports `llama_sampler_init_xtc` (llama-sampler.cpp:2423 @b1b770d90).
///
/// Parameters:
/// - `p`: the probability of applying it; `<= 0` disables it.
/// - `t`: the probability threshold; `> 0.5` disables it.
/// - `min_keep`: the fewest candidates to leave.
/// - `seed`: the seed; `LLAMA_DEFAULT_SEED` picks a random one.
///
/// Return: the sampler, or the `?xtc` placeholder when disabled.
pub fn llama_sampler_init_xtc(p: f32, t: f32, min_keep: usize, seed: u32) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = p <= 0.0 or t > 0.5;

    if (is_empty) return llama.sampler.core.initEmpty("?xtc");

    const seed_cur = llama.sampler.core.getRngSeed(seed);

    return llama.sampler.core.llama_sampler_init(&xtc_iface, llama.sampler.core.create(Xtc, .{
        .probability = p,
        .threshold = t,
        .min_keep = min_keep,
        .seed = seed,
        .seed_cur = seed_cur,
        .rng = stdlib.libcxx.Mt19937.init(seed_cur),
    }));
}

// -----------------------------------------------------------------------------
// top-n-sigma

/// Ports `llama_sampler_top_n_sigma` (llama-sampler.cpp:3202 @b1b770d90).
const TopNSigma = struct {
    n: f32,
};

/// Ports `llama_sampler_top_n_sigma_name` (llama-sampler.cpp:3206 @b1b770d90).
fn topNSigmaName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "top-n-sigma";
}

/// Ports `llama_sampler_top_n_sigma_apply` (llama-sampler.cpp:3210 @b1b770d90).
///
/// Masks every logit more than `n` standard deviations below the best,
/// ignoring -inf. `pow(x, 2)` is `pow(double, double)` in the C++ -- the
/// `float` argument is promoted -- and the running sum rounds back to
/// `float` each step. The cut is fused -- see the file header.
fn topNSigmaApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(TopNSigma, smpl);

    if (ctx.n <= 0.0 or cur_p.*.size <= 1) return;

    var max = cur_p.*.data[0].logit;
    var logits_sum: f32 = 0;
    var valid_count: usize = 0;
    for (llama.sampler.core.items(cur_p)) |d| {
        if (d.logit != -inf) {
            max = stdlib.libcxx.max(f32, max, d.logit);
            logits_sum += d.logit;
            valid_count += 1;
        }
    }
    const mean: f32 = if (valid_count > 0) logits_sum / @as(f32, @floatFromInt(valid_count)) else 0;

    var acc: f32 = 0;
    for (llama.sampler.core.items(cur_p)) |d| {
        if (d.logit != -inf) {
            acc = @floatCast(@as(f64, acc) + stdlib.libm.pow(d.logit - mean, 2));
        }
    }
    const std_dev: f32 = if (valid_count > 0) @sqrt(@as(f32, acc / @as(f32, @floatFromInt(valid_count)))) else 0;

    for (llama.sampler.core.items(cur_p)) |*d| {
        if (d.logit < @mulAdd(f32, -ctx.n, std_dev, max)) {
            d.logit = -inf;
        }
    }
}

/// Ports `llama_sampler_top_n_sigma_clone` (llama-sampler.cpp:3249 @b1b770d90).
fn topNSigmaClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    return llama_sampler_init_top_n_sigma(llama.sampler.core.ctxOf(TopNSigma, smpl).n);
}

/// Ports `llama_sampler_top_n_sigma_free` (llama-sampler.cpp:3254 @b1b770d90).
fn topNSigmaFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    llama.sampler.core.destroy(TopNSigma, llama.sampler.core.ctxOf(TopNSigma, smpl));
}

/// Ports `llama_sampler_top_n_sigma_i` (llama-sampler.cpp:3258 @b1b770d90).
var top_n_sigma_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = topNSigmaName,
    .accept = null,
    .apply = topNSigmaApply,
    .reset = null,
    .clone = topNSigmaClone,
    .free = topNSigmaFree,
};

/// Ports `llama_sampler_init_top_n_sigma` (llama-sampler.cpp:3273 @b1b770d90).
///
/// Parameters:
/// - `n`: how many standard deviations to keep; `<= 0` disables it.
///
/// Return: the sampler, or the `?top-n-sigma` placeholder when disabled.
pub fn llama_sampler_init_top_n_sigma(n: f32) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = n <= 0.0;

    if (is_empty) return llama.sampler.core.initEmpty("?top-n-sigma");

    return llama.sampler.core.llama_sampler_init(&top_n_sigma_iface, llama.sampler.core.create(TopNSigma, .{ .n = n }));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
