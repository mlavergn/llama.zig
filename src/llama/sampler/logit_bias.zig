//! The logit-bias sampler.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v0.3.0, `c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const core = @import("core.zig");

const c = impl.c;
const Sampler = core.Sampler;
const SamplerI = core.SamplerI;
const TokenDataArray = core.TokenDataArray;
const Backend = core.Backend;
const src = core.src;

/// Ports `llama_sampler_logit_bias` (llama-sampler.cpp:3886 @c1d0e7a00).
const LogitBias = struct {
    base: Backend,

    n_vocab: i32,

    logit_bias: []c.llama_logit_bias,

    to_search: std.ArrayList(c.llama_logit_bias),

    inp_logit_bias: ?*c.struct_ggml_tensor,
    inp_logit_idxs: ?*c.struct_ggml_tensor,
};

/// Ports `llama_sampler_logit_bias_name` (llama-sampler.cpp:3897 @c1d0e7a00).
fn logitBiasName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    return core.ctxOf(LogitBias, smpl).base.getName();
}

/// Ports `llama_sampler_logit_bias_apply` (llama-sampler.cpp:3902 @c1d0e7a00).
///
/// A bias whose token sits at its own index is applied directly; the rest
/// are searched for, each candidate taking the first bias that names it.
fn logitBiasApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(LogitBias, smpl);

    if (ctx.logit_bias.len == 0) return;

    ctx.to_search.clearRetainingCapacity();

    for (ctx.logit_bias) |lb| {
        if (lb.token >= 0 and cur_p.*.size > @as(usize, @intCast(lb.token)) and cur_p.*.data[@intCast(lb.token)].id == lb.token) {
            cur_p.*.data[@intCast(lb.token)].logit += lb.bias;
        } else {
            ctx.to_search.append(impl.allocator, lb) catch impl.abort(src, 3916, "out of memory");
        }
    }

    if (ctx.to_search.items.len == 0) return;

    for (core.items(cur_p)) |*d| {
        for (ctx.to_search.items) |lb| {
            if (d.id == lb.token) {
                d.logit += lb.bias;
                break;
            }
        }
    }
}

/// Ports `llama_sampler_logit_bias_clone` (llama-sampler.cpp:3935 @c1d0e7a00).
fn logitBiasClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    const ctx = core.ctxOf(LogitBias, smpl);
    return llama_sampler_init_logit_bias(ctx.n_vocab, @intCast(ctx.logit_bias.len), ctx.logit_bias.ptr);
}

/// Ports `llama_sampler_logit_bias_free` (llama-sampler.cpp:3940 @c1d0e7a00).
fn logitBiasFree(smpl: [*c]Sampler) callconv(.c) void {
    const ctx = core.ctxOf(LogitBias, smpl);
    impl.allocator.free(ctx.logit_bias);
    ctx.to_search.deinit(impl.allocator);
    core.destroy(LogitBias, ctx);
}

/// Ports `llama_sampler_logit_bias_backend_apply` (llama-sampler.cpp:3944 @c1d0e7a00).
///
/// Scatters the biases into a zero tensor shaped like the logits and adds it.
fn logitBiasBackendApply(smpl: [*c]Sampler, ctx: ?*c.struct_ggml_context, gf: ?*c.struct_ggml_cgraph, data: [*c]c.struct_llama_sampler_data) callconv(.c) void {
    _ = gf;

    const sctx = core.ctxOf(LogitBias, smpl);
    if (sctx.logit_bias.len == 0) return;

    const n: i64 = @intCast(sctx.logit_bias.len);

    if (sctx.inp_logit_bias == null) {
        impl.assert(sctx.inp_logit_idxs == null, src, 3960, "sctx->inp_logit_idxs == nullptr");

        sctx.inp_logit_bias = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 1, n);
        _ = c.ggml_set_name(sctx.inp_logit_bias, "logit_bias");
        c.ggml_set_input(sctx.inp_logit_bias);

        sctx.inp_logit_idxs = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n);
        _ = c.ggml_set_name(sctx.inp_logit_idxs, "logit_idxs");
        c.ggml_set_input(sctx.inp_logit_idxs);
    }

    var cur = c.ggml_fill(ctx, data.*.logits, 0.0);

    cur = c.ggml_reshape_2d(ctx, cur, 1, c.ggml_nelements(cur));
    cur = c.ggml_set_rows(ctx, cur, sctx.inp_logit_bias, sctx.inp_logit_idxs);
    cur = c.ggml_reshape_1d(ctx, cur, c.ggml_nelements(cur));

    data.*.logits = c.ggml_add(ctx, data.*.logits, cur);
}

/// Ports `llama_sampler_logit_bias_backend_set_input` (llama-sampler.cpp:3980 @c1d0e7a00).
fn logitBiasBackendSetInput(smpl: [*c]Sampler) callconv(.c) void {
    const sctx = core.ctxOf(LogitBias, smpl);
    if (sctx.logit_bias.len == 0) return;

    impl.assert(sctx.inp_logit_bias != null, src, 3986, "sctx->inp_logit_bias != nullptr");
    impl.assert(sctx.inp_logit_idxs != null, src, 3987, "sctx->inp_logit_idxs != nullptr");

    const n = sctx.logit_bias.len;

    const data_logit_bias = impl.allocator.alloc(f32, n) catch impl.abort(src, 3991, "out of memory");
    defer impl.allocator.free(data_logit_bias);
    const data_logit_idxs = impl.allocator.alloc(i32, n) catch impl.abort(src, 3992, "out of memory");
    defer impl.allocator.free(data_logit_idxs);
    for (sctx.logit_bias, 0..) |lb, i| {
        impl.assert(lb.token >= 0 and lb.token < sctx.n_vocab, src, 3995, "lb.token >= 0 && lb.token < (int32_t) sctx->n_vocab");
        data_logit_bias[i] = lb.bias;
        data_logit_idxs[i] = lb.token;
    }

    c.ggml_backend_tensor_set(sctx.inp_logit_bias, data_logit_bias.ptr, 0, c.ggml_nbytes(sctx.inp_logit_bias));
    c.ggml_backend_tensor_set(sctx.inp_logit_idxs, data_logit_idxs.ptr, 0, c.ggml_nbytes(sctx.inp_logit_idxs));
}

/// Ports `llama_sampler_logit_bias_backend_reset` (llama-sampler.cpp:4004 @c1d0e7a00).
fn logitBiasBackendReset(smpl: [*c]Sampler) callconv(.c) void {
    const sctx = core.ctxOf(LogitBias, smpl);
    sctx.inp_logit_bias = null;
    sctx.inp_logit_idxs = null;
}

/// Ports `llama_sampler_logit_bias_backend_init` (llama-sampler.cpp:4010 @c1d0e7a00).
///
/// Always claims the backend: its ops are the fill, scatter and add every
/// device has.
fn logitBiasBackendInit(smpl: [*c]Sampler, buft: c.ggml_backend_buffer_type_t, n_outputs_max_per_seq: u32) callconv(.c) bool {
    _ = buft;
    _ = n_outputs_max_per_seq;

    const sctx = core.ctxOf(LogitBias, smpl);

    sctx.base.initBackend(true);

    return true;
}

fn logitBiasCopyState(src_smpl: [*c]const Sampler, dst_smpl: [*c]Sampler) callconv(.c) void {
    _ = src_smpl;
    _ = dst_smpl;
}

/// Ports `llama_sampler_logit_bias_i` (llama-sampler.cpp:4028 @c1d0e7a00).
var logit_bias_iface: SamplerI = .{
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

/// Ports `llama_sampler_init_logit_bias` (llama-sampler.cpp:4043 @c1d0e7a00).
///
/// Parameters:
/// - `n_vocab`: the vocabulary size; the backend input needs it.
/// - `n_logit_bias`: how many biases; `<= 0` disables it.
/// - `logit_bias`: the biases; copied.
///
/// Return: the sampler, or the `?logit-bias` placeholder when disabled.
pub export fn llama_sampler_init_logit_bias(n_vocab: i32, n_logit_bias: i32, logit_bias: [*c]const c.llama_logit_bias) callconv(.c) [*c]Sampler {
    const is_empty = n_logit_bias <= 0;

    if (is_empty) return core.initEmpty("?logit-bias");

    const copy = impl.allocator.dupe(c.llama_logit_bias, logit_bias[0..@intCast(n_logit_bias)]) catch impl.abort(src, 4058, "out of memory");

    return core.llama_sampler_init(&logit_bias_iface, core.create(LogitBias, .{
        .base = Backend.init("logit-bias"),
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
