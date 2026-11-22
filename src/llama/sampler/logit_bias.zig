//! The logit-bias sampler.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v1.0.0, `b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_sampler_logit_bias` (llama-sampler.cpp:3882 @b1b770d90).
const LogitBias = struct {
    base: llama.sampler.core.Backend,

    n_vocab: i32,

    logit_bias: []llama.impl.llama_c.llama_logit_bias,

    to_search: std.ArrayList(llama.impl.llama_c.llama_logit_bias),

    inp_logit_bias: ?*llama.impl.llama_c.struct_ggml_tensor,
    inp_logit_idxs: ?*llama.impl.llama_c.struct_ggml_tensor,
};

/// Ports `llama_sampler_logit_bias_name` (llama-sampler.cpp:3893 @b1b770d90).
fn logitBiasName(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]const u8 {
    return llama.sampler.core.ctxOf(LogitBias, smpl).base.getName();
}

/// Ports `llama_sampler_logit_bias_apply` (llama-sampler.cpp:3898 @b1b770d90).
///
/// A bias whose token sits at its own index is applied directly; the rest
/// are searched for, each candidate taking the first bias that names it.
fn logitBiasApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, cur_p: [*c]llama.impl.llama_c.llama_token_data_array) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(LogitBias, smpl);

    if (ctx.logit_bias.len == 0) return;

    ctx.to_search.clearRetainingCapacity();

    for (ctx.logit_bias) |lb| {
        if (lb.token >= 0 and cur_p.*.size > @as(usize, @intCast(lb.token)) and cur_p.*.data[@intCast(lb.token)].id == lb.token) {
            cur_p.*.data[@intCast(lb.token)].logit += lb.bias;
        } else {
            ctx.to_search.append(std.heap.c_allocator, lb) catch llama.impl.abort(llama.sampler.core.src, 3916, "out of memory");
        }
    }

    if (ctx.to_search.items.len == 0) return;

    for (llama.sampler.core.items(cur_p)) |*d| {
        for (ctx.to_search.items) |lb| {
            if (d.id == lb.token) {
                d.logit += lb.bias;
                break;
            }
        }
    }
}

/// Ports `llama_sampler_logit_bias_clone` (llama-sampler.cpp:3931 @b1b770d90).
fn logitBiasClone(smpl: [*c]const llama.impl.llama_c.struct_llama_sampler) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const ctx = llama.sampler.core.ctxOf(LogitBias, smpl);
    return llama_sampler_init_logit_bias(ctx.n_vocab, @intCast(ctx.logit_bias.len), ctx.logit_bias.ptr);
}

/// Ports `llama_sampler_logit_bias_free` (llama-sampler.cpp:3936 @b1b770d90).
fn logitBiasFree(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const ctx = llama.sampler.core.ctxOf(LogitBias, smpl);
    std.heap.c_allocator.free(ctx.logit_bias);
    ctx.to_search.deinit(std.heap.c_allocator);
    llama.sampler.core.destroy(LogitBias, ctx);
}

/// Ports `llama_sampler_logit_bias_backend_apply` (llama-sampler.cpp:3940 @b1b770d90).
///
/// Scatters the biases into a zero tensor shaped like the logits and adds it.
fn logitBiasBackendApply(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, ctx: ?*llama.impl.llama_c.struct_ggml_context, gf: ?*llama.impl.llama_c.struct_ggml_cgraph, data: [*c]llama.impl.llama_c.struct_llama_sampler_data) callconv(.c) void {
    _ = gf;

    const sctx = llama.sampler.core.ctxOf(LogitBias, smpl);
    if (sctx.logit_bias.len == 0) return;

    const n: i64 = @intCast(sctx.logit_bias.len);

    if (sctx.inp_logit_bias == null) {
        llama.impl.assert(sctx.inp_logit_idxs == null, llama.sampler.core.src, 3990, "sctx->inp_logit_idxs == nullptr");

        sctx.inp_logit_bias = llama.impl.llama_c.ggml_new_tensor_2d(ctx, llama.impl.llama_c.GGML_TYPE_F32, 1, n);
        _ = llama.impl.llama_c.ggml_set_name(sctx.inp_logit_bias, "logit_bias");
        llama.impl.llama_c.ggml_set_input(sctx.inp_logit_bias);

        sctx.inp_logit_idxs = llama.impl.llama_c.ggml_new_tensor_1d(ctx, llama.impl.llama_c.GGML_TYPE_I32, n);
        _ = llama.impl.llama_c.ggml_set_name(sctx.inp_logit_idxs, "logit_idxs");
        llama.impl.llama_c.ggml_set_input(sctx.inp_logit_idxs);
    }

    var cur = llama.impl.llama_c.ggml_fill(ctx, data.*.logits, 0.0);

    cur = llama.impl.llama_c.ggml_reshape_2d(ctx, cur, 1, llama.impl.llama_c.ggml_nelements(cur));
    cur = llama.impl.llama_c.ggml_set_rows(ctx, cur, sctx.inp_logit_bias, sctx.inp_logit_idxs);
    cur = llama.impl.llama_c.ggml_reshape_1d(ctx, cur, llama.impl.llama_c.ggml_nelements(cur));

    data.*.logits = llama.impl.llama_c.ggml_add(ctx, data.*.logits, cur);
}

/// Ports `llama_sampler_logit_bias_backend_set_input` (llama-sampler.cpp:3976 @b1b770d90).
fn logitBiasBackendSetInput(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const sctx = llama.sampler.core.ctxOf(LogitBias, smpl);
    if (sctx.logit_bias.len == 0) return;

    llama.impl.assert(sctx.inp_logit_bias != null, llama.sampler.core.src, 4016, "sctx->inp_logit_bias != nullptr");
    llama.impl.assert(sctx.inp_logit_idxs != null, llama.sampler.core.src, 4017, "sctx->inp_logit_idxs != nullptr");

    const n = sctx.logit_bias.len;

    const data_logit_bias = std.heap.c_allocator.alloc(f32, n) catch llama.impl.abort(llama.sampler.core.src, 3991, "out of memory");
    defer std.heap.c_allocator.free(data_logit_bias);
    const data_logit_idxs = std.heap.c_allocator.alloc(i32, n) catch llama.impl.abort(llama.sampler.core.src, 3992, "out of memory");
    defer std.heap.c_allocator.free(data_logit_idxs);
    for (sctx.logit_bias, 0..) |lb, i| {
        llama.impl.assert(lb.token >= 0 and lb.token < sctx.n_vocab, llama.sampler.core.src, 4025, "lb.token >= 0 && lb.token < (int32_t) sctx->n_vocab");
        data_logit_bias[i] = lb.bias;
        data_logit_idxs[i] = lb.token;
    }

    llama.impl.llama_c.ggml_backend_tensor_set(sctx.inp_logit_bias, data_logit_bias.ptr, 0, llama.impl.llama_c.ggml_nbytes(sctx.inp_logit_bias));
    llama.impl.llama_c.ggml_backend_tensor_set(sctx.inp_logit_idxs, data_logit_idxs.ptr, 0, llama.impl.llama_c.ggml_nbytes(sctx.inp_logit_idxs));
}

/// Ports `llama_sampler_logit_bias_backend_reset` (llama-sampler.cpp:4000 @b1b770d90).
fn logitBiasBackendReset(smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    const sctx = llama.sampler.core.ctxOf(LogitBias, smpl);
    sctx.inp_logit_bias = null;
    sctx.inp_logit_idxs = null;
}

/// Ports `llama_sampler_logit_bias_backend_init` (llama-sampler.cpp:4006 @b1b770d90).
///
/// Always claims the backend: its ops are the fill, scatter and add every
/// device has.
fn logitBiasBackendInit(smpl: [*c]llama.impl.llama_c.struct_llama_sampler, buft: llama.impl.llama_c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = buft;
    _ = n_outputs_max_per_seq;

    const sctx = llama.sampler.core.ctxOf(LogitBias, smpl);

    sctx.base.initBackend(true);

    return true;
}

fn logitBiasCopyState(src_smpl: [*c]const llama.impl.llama_c.struct_llama_sampler, dst_smpl: [*c]llama.impl.llama_c.struct_llama_sampler) callconv(.c) void {
    _ = src_smpl;
    _ = dst_smpl;
}

/// Ports `llama_sampler_logit_bias_i` (llama-sampler.cpp:4024 @b1b770d90).
var logit_bias_iface: llama.impl.llama_c.struct_llama_sampler_i = .{
    .name = logitBiasName,
    .accept = null,
    .apply = logitBiasApply,
    .reset = null,
    .clone = logitBiasClone,
    .free = logitBiasFree,
    .backend_init = logitBiasBackendInit,
    .backend_accept = null,
    .backend_apply = logitBiasBackendApply,
    .backend_set_input = logitBiasBackendSetInput,
    .backend_reset = logitBiasBackendReset,
    .copy_state = logitBiasCopyState,
};

/// Ports `llama_sampler_init_logit_bias` (llama-sampler.cpp:4039 @b1b770d90).
///
/// Parameters:
/// - `n_vocab`: the vocabulary size; the backend input needs it.
/// - `n_logit_bias`: how many biases; `<= 0` disables it.
/// - `logit_bias`: the biases; copied.
///
/// Return: the sampler, or the `?logit-bias` placeholder when disabled.
pub fn llama_sampler_init_logit_bias(n_vocab: i32, n_logit_bias: i32, logit_bias: [*c]const llama.impl.llama_c.llama_logit_bias) callconv(.c) [*c]llama.impl.llama_c.struct_llama_sampler {
    const is_empty = n_logit_bias <= 0;

    if (is_empty) return llama.sampler.core.initEmpty("?logit-bias");

    const copy = std.heap.c_allocator.dupe(llama.impl.llama_c.llama_logit_bias, logit_bias[0..@intCast(n_logit_bias)]) catch llama.impl.abort(llama.sampler.core.src, 4058, "out of memory");

    return llama.sampler.core.llama_sampler_init(&logit_bias_iface, llama.sampler.core.create(LogitBias, .{
        .base = llama.sampler.core.Backend.init("logit-bias"),
        .n_vocab = n_vocab,
        .logit_bias = copy,
        .to_search = .empty,
        .inp_logit_bias = null,
        .inp_logit_idxs = null,
    }));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
