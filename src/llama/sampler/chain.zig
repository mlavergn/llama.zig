//! The sampler chain, `llama_sampler_sample`, and the API calls that look
//! across sampler types: copy, seed and the perf counters.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # The two C++-linkage entry points
//!
//! `llama_sampler_backend_begin` and `llama_sampler_backend_n_nodes` are
//! declared in `llama-sampler.h` without `extern "C"`, and `llama-context.cpp`
//! calls them under their mangled names. Both take a plain pointer, so this
//! file exports them under those names -- taken from `nm` on the reference
//! object, not constructed -- and the C++ caller links unchanged. It is the
//! move `src/ggml/metal/tuning.zig` made.
//!
//! # A by-value struct received across the C ABI
//!
//! `llama_sampler_chain_init` takes `struct llama_sampler_chain_params` by
//! value, and that struct is one `bool`: one byte. Zig 0.16 miscompiles a
//! small `extern struct` received by value -- the callee sees zeros -- unless
//! its size is a multiple of 4 with no padding. So the
//! parameter is declared as the byte it is and bit-cast, and the entry point
//! is in `harness/abi_structs.c`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_sampler_chain`'s `info` (llama-sampler.h:20 @b1b770d90).
const Info = struct {
    is_backend: bool,
    ptr: [*c]llama.impl.c.struct_llama_sampler,
};

/// Ports `llama_sampler_chain` (llama-sampler.h:12 @b1b770d90).
///
/// Its layout never crosses a translation unit: `llama-context.cpp` reaches a
/// chain only through `llama_sampler_chain_get` and `_n`. That is what lets
/// it be a Zig struct.
pub const Chain = struct {
    params: llama.impl.c.llama_sampler_chain_params,
    is_init: bool,
    n_nodes: u32,
    samplers: std.ArrayList(Info),
    /// The candidate buffer `llama_sampler_sample` reuses across calls.
    cur: std.ArrayList(llama.impl.c.llama_token_data),
    t_sample_us: i64,
    n_sample: i32,
};

/// Ports `llama_sampler_chain_name` (llama-sampler.cpp:666 @b1b770d90).
fn chainName(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "chain";
}

/// Ports `llama_sampler_chain_accept` (llama-sampler.cpp:670 @b1b770d90).
fn chainAccept(smpl: [*c]llama.impl.c.struct_llama_sampler, token: llama.impl.c.llama_token) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    const tm = llama.impl.TimeMeas.init(&chain.t_sample_us, chain.params.no_perf);
    defer tm.deinit();

    for (chain.samplers.items) |s| llama.sampler.core.llama_sampler_accept(s.ptr, token);

    chain.n_sample += 1;
}

/// Ports `llama_sampler_chain_apply` (llama-sampler.cpp:682 @b1b770d90).
///
/// After `backend_init`, the leading run of samplers that ran on the backend
/// is skipped: the candidates arriving here are already their output.
fn chainApply(smpl: [*c]llama.impl.c.struct_llama_sampler, cur_p: [*c]llama.impl.c.llama_token_data_array) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    const tm = llama.impl.TimeMeas.init(&chain.t_sample_us, chain.params.no_perf);
    defer tm.deinit();

    var is_backend = chain.is_init;

    for (chain.samplers.items) |s| {
        if (is_backend and s.is_backend) continue;

        is_backend = false;

        if (s.ptr.*.iface.*.apply == null) continue;

        llama.sampler.core.llama_sampler_apply(s.ptr, cur_p);
    }
}

/// Ports `llama_sampler_chain_reset` (llama-sampler.cpp:704 @b1b770d90).
fn chainReset(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);
    for (chain.samplers.items) |s| llama.sampler.core.llama_sampler_reset(s.ptr);
}

/// Ports `llama_sampler_chain_clone` (llama-sampler.cpp:712 @b1b770d90).
fn chainClone(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const chain_src = llama.sampler.core.ctxOf(Chain, smpl);

    const result = chainInit(chain_src.params);

    for (chain_src.samplers.items) |s| {
        llama_sampler_chain_add(result, llama.sampler.core.llama_sampler_clone(s.ptr));
    }

    return result;
}

/// Ports `llama_sampler_chain_free` (llama-sampler.cpp:724 @b1b770d90).
fn chainFree(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    for (chain.samplers.items) |s| llama.sampler.core.llama_sampler_free(s.ptr);

    chain.samplers.deinit(std.heap.c_allocator);
    chain.cur.deinit(std.heap.c_allocator);
    llama.sampler.core.destroy(Chain, chain);
}

/// Ports `llama_sampler_chain_backend_init` (llama-sampler.cpp:734 @b1b770d90).
///
/// Marks the longest prefix of samplers that can run on the backend, then
/// counts the nodes its graph needs.
fn chainBackendInit(smpl: [*c]llama.impl.c.struct_llama_sampler, buft: llama.impl.c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    llama.impl.assert(chain.is_init == false, llama.sampler.core.src, 740, "chain->is_init == false && \"llama_sampler_chain_backend_init() called twice\"");

    chain.is_init = true;

    var res = true;
    var backend_prefix = true;

    for (chain.samplers.items) |*s| {
        var cur_prefix = backend_prefix;

        if (cur_prefix and s.ptr.*.iface.*.backend_init != null) {
            if (!s.ptr.*.iface.*.backend_init.?(s.ptr, buft, n_outputs_max_per_seq)) {
                cur_prefix = false;
            }
        } else {
            cur_prefix = false;
        }

        s.is_backend = cur_prefix;
        backend_prefix = cur_prefix;

        res = res and cur_prefix;
    }

    const probe = llama.sampler.core.probeGraph(smpl, 128 * 1024, llama.impl.c.GGML_DEFAULT_GRAPH_SIZE, false);
    defer probe.deinit();
    chain.n_nodes = llama.sampler.core.probeNNodes(probe);

    return res;
}

/// Ports `llama_sampler_chain_backend_accept` (llama-sampler.cpp:774 @b1b770d90).
fn chainBackendAccept(smpl: [*c]llama.impl.c.struct_llama_sampler, ctx: ?*llama.impl.c.struct_ggml_context, gf: ?*llama.impl.c.struct_ggml_cgraph, selected_token: [*c]llama.impl.c.struct_ggml_tensor) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    for (chain.samplers.items) |s| {
        if (!s.is_backend) break;

        if (s.ptr.*.iface.*.backend_accept) |f| f(s.ptr, ctx, gf, selected_token);
    }
}

/// Ports `llama_sampler_chain_backend_apply` (llama-sampler.cpp:792 @b1b770d90).
fn chainBackendApply(smpl: [*c]llama.impl.c.struct_llama_sampler, ctx: ?*llama.impl.c.struct_ggml_context, gf: ?*llama.impl.c.struct_ggml_cgraph, data: [*c]llama.impl.c.struct_llama_sampler_data) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    llama.impl.assert(chain.is_init, llama.sampler.core.src, 799, "chain->is_init && \"llama_sampler_chain_backend_init() not called\"");

    for (chain.samplers.items) |s| {
        if (!s.is_backend) break;

        if (s.ptr.*.iface.*.backend_apply) |f| f(s.ptr, ctx, gf, data);
    }
}

/// Ports `llama_sampler_chain_backend_set_input` (llama-sampler.cpp:812 @b1b770d90).
fn chainBackendSetInput(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    for (chain.samplers.items) |s| {
        if (!s.is_backend) break;

        if (s.ptr.*.iface.*.backend_set_input) |f| f(s.ptr);
    }
}

/// Ports `llama_sampler_chain_backend_reset` (llama-sampler.cpp:826 @b1b770d90).
fn chainBackendReset(smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const chain = llama.sampler.core.ctxOf(Chain, smpl);

    for (chain.samplers.items) |entry| {
        if (!entry.is_backend) break;
        if (entry.ptr.*.iface.*.backend_reset) |f| f(entry.ptr);
    }
}

/// Ports `llama_sampler_chain_copy_state` (llama-sampler.cpp:839 @b1b770d90).
fn chainCopyState(src_smpl: [*c]const llama.impl.c.struct_llama_sampler, dst_smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const src_chain = llama.sampler.core.ctxOf(Chain, src_smpl);
    const dst_chain = llama.sampler.core.ctxOf(Chain, dst_smpl);

    llama.impl.assert(src_chain.samplers.items.len == dst_chain.samplers.items.len, llama.sampler.core.src, 843, "src_chain->samplers.size() == dst_chain->samplers.size()");

    for (src_chain.samplers.items, dst_chain.samplers.items) |s, d| {
        llama_sampler_copy(s.ptr, d.ptr);
    }

    // is_init, n_nodes and is_backend belong to the current sampling graph.
    dst_chain.params = src_chain.params;
    dst_chain.cur.clearRetainingCapacity();
    dst_chain.cur.appendSlice(std.heap.c_allocator, src_chain.cur.items) catch llama.impl.abort(llama.sampler.core.src, 850, "out of memory");
    dst_chain.t_sample_us = src_chain.t_sample_us;
    dst_chain.n_sample = src_chain.n_sample;
}

/// Ports `llama_sampler_chain_i` (llama-sampler.cpp:856 @b1b770d90).
///
/// Compared by address in `llama_sampler_sample`, `llama_sampler_chain_get`
/// and elsewhere, so it is a single `var`, never copied.
pub var chain_iface: llama.impl.c.struct_llama_sampler_i = .{
    .name = chainName,
    .accept = chainAccept,
    .apply = chainApply,
    .reset = chainReset,
    .clone = chainClone,
    .free = chainFree,
    .backend_init = chainBackendInit,
    .backend_accept = chainBackendAccept,
    .backend_apply = chainBackendApply,
    .backend_set_input = chainBackendSetInput,
    .backend_reset = chainBackendReset,
    .copy_state = chainCopyState,
};

/// `llama_sampler_chain_init` for Zig callers, which pass the params struct
/// itself; the export below receives it as a byte.
///
/// Parameters:
/// - `params`: the chain's parameters.
///
/// Return: an empty chain, owned by the caller.
pub fn chainInit(params: llama.impl.c.llama_sampler_chain_params) [*c]llama.impl.c.struct_llama_sampler {
    return llama.sampler.core.llama_sampler_init(&chain_iface, llama.sampler.core.create(Chain, .{
        .params = params,
        .is_init = false,
        .n_nodes = 0,
        .samplers = .empty,
        .cur = .empty,
        .t_sample_us = 0,
        .n_sample = 0,
    }));
}

/// Ports `llama_sampler_chain_init` (llama-sampler.cpp:871 @b1b770d90).
///
/// `params` is a one-byte struct received by value; see the file header.
///
/// Parameters:
/// - `params_bits`: `struct llama_sampler_chain_params`, as its byte.
///
/// Return: an empty chain.
pub export fn llama_sampler_chain_init(params_bits: u8) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const params = std.mem.bytesToValue(llama.impl.c.llama_sampler_chain_params, std.mem.asBytes(&params_bits));
    return chainInit(params);
}

/// Ports `llama_sampler_backend_n_nodes` (llama-sampler.cpp:886 @b1b770d90).
///
/// C++ linkage; exported under its mangled name. See the file header.
///
/// Parameters:
/// - `sampler`: a chain that has been through `backend_init`.
///
/// Return: the node count its backend graph needs.
fn backendNNodes(sampler: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) u32 {
    llama.impl.assert(sampler != null, llama.sampler.core.src, 887, "sampler != nullptr");
    llama.impl.assert(sampler.*.iface == &chain_iface, llama.sampler.core.src, 888, "sampler->iface == &llama_sampler_chain_i");

    const chain = llama.sampler.core.ctxOf(Chain, sampler);
    llama.impl.assert(chain.is_init, llama.sampler.core.src, 891, "chain->is_init");

    return chain.n_nodes;
}

/// Ports `llama_sampler_sample` (llama-sampler.cpp:896 @b1b770d90).
///
/// Returns the token a backend sampler already picked, when there is one;
/// otherwise builds the candidates from whatever the backend left -- sampled
/// probabilities, sampled logits, or the raw logits -- applies the sampler
/// and accepts its choice.
///
/// Parameters:
/// - `smpl`: the sampler.
/// - `ctx`: the context the last decode ran in.
/// - `idx`: the output's position in the batch; negative counts from the end.
///
/// Return: the sampled token.
pub export fn llama_sampler_sample(smpl: [*c]llama.impl.c.struct_llama_sampler, ctx: ?*llama.impl.c.struct_llama_context, idx: i32) callconv(.c) llama.impl.c.llama_token {
    const sampled_token = llama.api.context.llama_get_sampled_token_ith(ctx, idx);
    const sampled_probs = llama.api.context.llama_get_sampled_probs_ith(ctx, idx);
    const sampled_logits = llama.api.context.llama_get_sampled_logits_ith(ctx, idx);
    const sampled_ids = llama.api.context.llama_get_sampled_candidates_ith(ctx, idx);

    if (sampled_token != llama.impl.c.LLAMA_TOKEN_NULL) {
        llama.impl.logDebug("%s: Backend sampler selected token for idx %d. Skipping CPU samplers\n", .{ "llama_sampler_sample", idx });
        llama.sampler.core.llama_sampler_accept(smpl, sampled_token);
        return sampled_token;
    }

    const model = llama.api.context.llama_get_model(ctx);
    const vocab = llama.api.model.llama_model_get_vocab(model);

    const n_vocab = llama.api.vocab.llama_vocab_n_tokens(vocab);

    // The chain's own buffer when there is one, otherwise a local.
    var cur_local: std.ArrayList(llama.impl.c.llama_token_data) = .empty;
    defer cur_local.deinit(std.heap.c_allocator);

    const cur: *std.ArrayList(llama.impl.c.llama_token_data) = if (smpl.*.iface == &chain_iface)
        &llama.sampler.core.ctxOf(Chain, smpl).cur
    else
        &cur_local;

    if (sampled_probs != null) {
        const count = llama.api.context.llama_get_sampled_probs_count_ith(ctx, idx);
        cur.resize(std.heap.c_allocator, count) catch llama.impl.abort(llama.sampler.core.src, 928, "out of memory");
        for (0..count) |i| {
            cur.items[i] = .{ .id = sampled_ids[i], .logit = sampled_logits.?[i], .p = sampled_probs.?[i] };
        }
    } else if (sampled_logits != null) {
        const count = llama.api.context.llama_get_sampled_logits_count_ith(ctx, idx);
        cur.resize(std.heap.c_allocator, count) catch llama.impl.abort(llama.sampler.core.src, 934, "out of memory");
        for (0..count) |i| {
            cur.items[i] = .{ .id = sampled_ids[i], .logit = sampled_logits.?[i], .p = 0.0 };
        }
    } else {
        const logits = llama.api.context.llama_get_logits_ith(ctx, idx);
        llama.impl.assert(logits != null, llama.sampler.core.src, 941, "logits != nullptr");
        cur.resize(std.heap.c_allocator, @intCast(n_vocab)) catch llama.impl.abort(llama.sampler.core.src, 941, "out of memory");
        for (0..@intCast(n_vocab)) |token_id| {
            cur.items[token_id] = .{ .id = @intCast(token_id), .logit = logits.?[token_id], .p = 0.0 };
        }
    }

    var cur_p: llama.impl.c.llama_token_data_array = .{
        .data = cur.items.ptr,
        .size = cur.items.len,
        .selected = -1,
        .sorted = false,
    };

    llama.sampler.core.llama_sampler_apply(smpl, &cur_p);

    llama.impl.assert(cur_p.selected >= 0 and cur_p.selected < @as(i32, @intCast(cur_p.size)), llama.sampler.core.src, 957, "cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size");

    const token = cur_p.data[@intCast(cur_p.selected)].id;

    llama.sampler.core.llama_sampler_accept(smpl, token);

    return token;
}

/// Ports `llama_sampler_chain_add` (llama-sampler.cpp:967 @b1b770d90).
///
/// Parameters:
/// - `chain`: the chain; takes ownership of `smpl`.
/// - `smpl`: the sampler to append.
///
/// Return: nothing.
pub export fn llama_sampler_chain_add(chain: [*c]llama.impl.c.struct_llama_sampler, smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const p = llama.sampler.core.ctxOf(Chain, chain);
    p.samplers.append(std.heap.c_allocator, .{ .is_backend = false, .ptr = smpl }) catch llama.impl.abort(llama.sampler.core.src, 968, "out of memory");
}

/// Ports `llama_sampler_chain_get` (llama-sampler.cpp:975 @b1b770d90).
///
/// Parameters:
/// - `chain`: the chain.
/// - `i`: the index; -1 returns the chain itself.
///
/// Return: the sampler, or null when `chain` is null or not a chain, or `i`
/// is out of range.
pub export fn llama_sampler_chain_get(chain: [*c]llama.impl.c.struct_llama_sampler, i: i32) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    if (chain == null) return null;

    if (chain.*.iface != &chain_iface) return null;

    if (i == -1) return chain;

    const p = llama.sampler.core.ctxOf(Chain, chain);

    if (i < 0 or @as(usize, @intCast(i)) >= p.samplers.items.len) return null;

    return p.samplers.items[@intCast(i)].ptr;
}

/// Ports `llama_sampler_chain_remove` (llama-sampler.cpp:997 @b1b770d90).
///
/// Parameters:
/// - `chain`: the chain.
/// - `i`: the index.
///
/// Return: the removed sampler, now owned by the caller; null when out of
/// range.
pub export fn llama_sampler_chain_remove(chain: [*c]llama.impl.c.struct_llama_sampler, i: i32) callconv(.c) [*c]llama.impl.c.struct_llama_sampler {
    const p = llama.sampler.core.ctxOf(Chain, chain);

    if (i < 0 or @as(usize, @intCast(i)) >= p.samplers.items.len) return null;

    const result = p.samplers.items[@intCast(i)].ptr;
    _ = p.samplers.orderedRemove(@intCast(i));

    return result;
}

/// Ports `llama_sampler_chain_n` (llama-sampler.cpp:1010 @b1b770d90).
///
/// Return: the number of samplers in the chain.
pub export fn llama_sampler_chain_n(chain: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) i32 {
    const p = llama.sampler.core.ctxOf(Chain, chain);
    return @intCast(p.samplers.items.len);
}

/// Ports `llama_sampler_backend_begin` (llama-sampler.cpp:1424 @b1b770d90).
///
/// C++ linkage; exported under its mangled name. Starts a transaction for
/// every transactional `dist` in the backend prefix: its scratch engine
/// restarts from the committed one.
///
/// Parameters:
/// - `sampler`: a chain, or a sampler within one.
///
/// Return: nothing.
fn backendBegin(sampler: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    llama.impl.assert(sampler != null, llama.sampler.core.src, 1425, "sampler != nullptr");

    if (sampler.*.iface == &chain_iface) {
        const chain = llama.sampler.core.ctxOf(Chain, sampler);
        for (chain.samplers.items) |entry| {
            if (!entry.is_backend) break;
            backendBegin(entry.ptr);
        }
    } else if (sampler.*.iface == &llama.sampler.basic.dist_iface) {
        const ctx = llama.sampler.core.ctxOf(llama.sampler.basic.Dist, sampler);
        if (ctx.backend_transactional) {
            ctx.rng_backend = ctx.rng;
            ctx.n_backend_draws_generated = 0;
            ctx.n_backend_draws_committed = 0;
        }
    }
}

comptime {
    // Mangled names from `nm` on the reference `llama-sampler.o`.
    @export(&backendBegin, .{ .name = "_Z27llama_sampler_backend_beginP13llama_sampler" });
    @export(&backendNNodes, .{ .name = "_Z29llama_sampler_backend_n_nodesPK13llama_sampler" });
}

/// Ports `llama_sampler_copy` (llama-sampler.cpp:4295 @b1b770d90).
///
/// Carries `src_smpl`'s state into `dst_smpl`: through `copy_state` where the
/// type has one, otherwise by cloning and transplanting the clone's context.
///
/// Parameters:
/// - `src_smpl`: the sampler to copy from.
/// - `dst_smpl`: a sampler of the same type.
///
/// Return: nothing.
pub export fn llama_sampler_copy(src_smpl: [*c]const llama.impl.c.struct_llama_sampler, dst_smpl: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    if (src_smpl == null or dst_smpl == null or src_smpl == @as([*c]const llama.impl.c.struct_llama_sampler, dst_smpl)) return;

    llama.impl.assert(src_smpl.*.iface == dst_smpl.*.iface, llama.sampler.core.src, 4334, "src->iface == dst->iface && \"llama_sampler_copy: cannot copy between different sampler types\"");

    if (dst_smpl.*.iface.*.copy_state) |f| {
        f(src_smpl, dst_smpl);
        return;
    }

    const tmp = llama.sampler.core.llama_sampler_clone(src_smpl);

    if (dst_smpl.*.iface.*.free) |f| f(dst_smpl);

    dst_smpl.*.ctx = tmp.*.ctx;
    tmp.*.ctx = null;
    llama.sampler.core.destroy(llama.impl.c.struct_llama_sampler, tmp);
}

/// Ports `llama_sampler_get_seed` (llama-sampler.cpp:4323 @b1b770d90).
///
/// Parameters:
/// - `smpl`: the sampler.
///
/// Return: the seed `dist` or a mirostat is using, the last such in a chain,
/// or `LLAMA_DEFAULT_SEED`.
pub export fn llama_sampler_get_seed(smpl: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) u32 {
    if (smpl.*.iface == &llama.sampler.basic.dist_iface) {
        return llama.sampler.core.ctxOf(llama.sampler.basic.Dist, smpl).seed_cur;
    }

    if (smpl.*.iface == &llama.sampler.mirostat.mirostat_iface) {
        return llama.sampler.core.ctxOf(llama.sampler.mirostat.Mirostat, smpl).seed_cur;
    }

    if (smpl.*.iface == &llama.sampler.mirostat.mirostat_v2_iface) {
        return llama.sampler.core.ctxOf(llama.sampler.mirostat.MirostatV2, smpl).seed_cur;
    }

    if (smpl.*.iface == &chain_iface) {
        const ctx = llama.sampler.core.ctxOf(Chain, smpl);
        var it = ctx.samplers.items.len;
        while (it > 0) {
            it -= 1;
            const seed = llama_sampler_get_seed(ctx.samplers.items[it].ptr);
            if (seed != llama.impl.c.LLAMA_DEFAULT_SEED) return seed;
        }
    }

    return llama.impl.c.LLAMA_DEFAULT_SEED;
}

/// Ports `llama_perf_sampler` (llama-sampler.cpp:4351 @b1b770d90).
///
/// Parameters:
/// - `chain`: a chain; anything else aborts.
///
/// Return: its accumulated time and sample count.
pub export fn llama_perf_sampler(chain: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) llama.impl.c.struct_llama_perf_sampler_data {
    var data: llama.impl.c.struct_llama_perf_sampler_data = .{};

    if (chain == null or chain.*.iface != &chain_iface) {
        llama.impl.c.ggml_abort(llama.sampler.core.src, 4359, "%s: invalid sampler passed - requires a sampler created with llama_sampler_chain_init()\n", "llama_perf_sampler");
    }

    const ctx = llama.sampler.core.ctxOf(Chain, chain);

    data.t_sample_ms = 1e-3 * @as(f64, @floatFromInt(ctx.t_sample_us));
    data.n_sample = stdlib.libcxx.max(i32, 0, ctx.n_sample);

    return data;
}

/// Ports `llama_perf_sampler_print` (llama-sampler.cpp:4366 @b1b770d90).
///
/// Return: nothing.
pub export fn llama_perf_sampler_print(chain: [*c]const llama.impl.c.struct_llama_sampler) callconv(.c) void {
    const data = llama_perf_sampler(chain);

    llama.impl.logInfo("%s:    samplers time = %10.2f ms / %5d runs\n", .{ "llama_perf_sampler_print", data.t_sample_ms, data.n_sample });
}

/// Ports `llama_perf_sampler_reset` (llama-sampler.cpp:4372 @b1b770d90).
///
/// Return: nothing.
pub export fn llama_perf_sampler_reset(chain: [*c]llama.impl.c.struct_llama_sampler) callconv(.c) void {
    if (chain == null or chain.*.iface != &chain_iface) {
        llama.impl.c.ggml_abort(llama.sampler.core.src, 4378, "%s: invalid sampler passed - requires a sampler created with llama_sampler_chain_init()\n", "llama_perf_sampler_reset");
    }

    const ctx = llama.sampler.core.ctxOf(Chain, chain);

    ctx.t_sample_us = 0;
    ctx.n_sample = 0;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
