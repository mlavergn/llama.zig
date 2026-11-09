//! The blocks LFM2 and LFM3 share: the gated short convolution, with a
//! recurrent state or from a zero one, and the two feed-forwards.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/lfm-base.cpp` at v0.8.0 (`8642541b2`):
//! every member of `llm_build_lfm_base`. The C++ makes them members of a
//! subclass of `llm_graph_context`; here they are functions over the
//! builder.

const std = @import("std");
const impl = @import("../impl.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const memory_recurrent = @import("../memory_recurrent.zig");

const c = impl.c;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llm_build_lfm_base`'s `build_lfm_shortconv` (lfm-base.cpp:9
/// @8642541b2), the overload over a recurrent state: a gated depthwise
/// convolution over the last `n_shortconv_l_cache` tokens, whose state is
/// the recurrent memory.
///
/// Parameters:
/// - `model`: the model.
/// - `b`: the graph being built.
/// - `mctx_recr`: the recurrent memory's context.
/// - `inp_recr`: the graph's recurrent input.
/// - `cur_`: `[n_embd, n_tokens]`.
/// - `il`: the layer.
///
/// Return: `[n_embd, n_tokens]`.
pub fn buildLfmShortconv(model: *const Model, b: *const graph.Builder, mctx_recr: *const memory_recurrent.RecurrentContext, inp_recr: *const graph.InpRs, cur_: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    const kv_head: u32 = mctx_recr.getHead();
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;
    const n_seqs: i64 = b.ubatch.n_seqs;
    impl.assert(n_seqs != 0, "models/lfm-base.cpp", 17, "n_seqs != 0");
    impl.assert(b.ubatch.equalSeqs(), "models/lfm-base.cpp", 18, "ubatch.equal_seqs()");
    impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/lfm-base.cpp", 19, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    impl.assert(hp.n_shortconv_l_cache > 1, "models/lfm-base.cpp", 21, "hparams.n_shortconv_l_cache > 1");
    const d_conv: u32 = hp.n_shortconv_l_cache - 1;
    const n_embd: i64 = hp.n_embd;

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    const cur: *Tensor = c.ggml_reshape_3d(ctx, cur_, cur_.ne[0], n_seq_tokens, n_seqs);

    const bcx = b.buildLoraMm(layer.shortconv.in_proj.?, cur, null);
    b.cb(bcx, "model.layers.{}.conv.in_proj", il);

    const n_chunks = 3;
    impl.assert(@rem(bcx.ne[0], n_chunks) == 0, "models/lfm-base.cpp", 31, "bcx->ne[0] % n_chunks == 0");
    const chunk_size = @divTrunc(bcx.ne[0], n_chunks);
    const es = c.ggml_element_size(bcx);
    const chunk_bytes: usize = @intCast(chunk_size);
    const bv: *Tensor = c.ggml_view_3d(ctx, bcx, chunk_size, bcx.ne[1], bcx.ne[2], bcx.nb[1], bcx.nb[2], 0 * chunk_bytes * es);
    const cv: *Tensor = c.ggml_view_3d(ctx, bcx, chunk_size, bcx.ne[1], bcx.ne[2], bcx.nb[1], bcx.nb[2], 1 * chunk_bytes * es);
    const xv: *Tensor = c.ggml_view_3d(ctx, bcx, chunk_size, bcx.ne[1], bcx.ne[2], bcx.nb[1], bcx.nb[2], 2 * chunk_bytes * es);

    var bx: *Tensor = c.ggml_transpose(ctx, c.ggml_mul(ctx, bv, xv));

    // Read the conv state.
    const conv_state = mctx_recr.getRL(il);
    const conv_rs = b.buildRs(inp_recr, conv_state, @intCast(hp.nEmbdR()), @intCast(n_seqs), null);
    const conv: *Tensor = c.ggml_reshape_3d(ctx, conv_rs, d_conv, n_embd, n_seqs);

    // Causal prepends the state; non-causal pads symmetrically for a
    // centered window.
    if (hp.causal_attn) {
        bx = c.ggml_concat(ctx, conv, bx, 0);
    } else {
        const pad: i64 = @divTrunc(@as(i64, hp.n_shortconv_l_cache) - 1, 2);
        const left: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, conv, pad, n_embd, n_seqs, conv.nb[1], conv.nb[2], @as(usize, @intCast(@as(i64, d_conv) - pad)) * conv.nb[0]));
        bx = c.ggml_pad_ext(ctx, c.ggml_concat(ctx, left, bx, 0), 0, @intCast(pad), 0, 0, 0, 0, 0, 0);
    }
    impl.assert(bx.ne[0] > conv.ne[0], "models/lfm-base.cpp", 56, "bx->ne[0] > conv->ne[0]");

    // Write the conv states: slot 0 is the final state, slot s the state s
    // tokens back (for a partial rollback).
    const K: i64 = if (hp.causal_attn and b.cparams.n_rs_seq > 0) @as(i64, b.cparams.n_rs_seq) + 1 else 1;
    const n_written: i64 = @min(n_seq_tokens, K);
    const mem_size: usize = mctx_recr.getSize();
    const row_size: usize = c.ggml_row_size(conv_state.type, @as(i64, d_conv) * n_embd);

    var slot: i64 = 0;
    while (slot < n_written) : (slot += 1) {
        const conv_snap: *Tensor = c.ggml_view_3d(ctx, bx, d_conv, bx.ne[1], bx.ne[2], bx.nb[1], bx.nb[2], @as(usize, @intCast(bx.ne[0] - d_conv - slot)) * c.ggml_element_size(bx));
        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, conv_snap, c.ggml_view_2d(ctx, conv_state, @as(i64, d_conv) * n_embd, n_seqs, conv_state.nb[1], (@as(usize, @intCast(slot)) * mem_size + kv_head) * row_size)));
    }

    const conv_kernel = layer.shortconv.conv;
    const conv_out: *Tensor = c.ggml_ssm_conv(ctx, bx, conv_kernel);
    b.cb(conv_out, "model.layers.{}.conv.conv", il);

    var y: *Tensor = c.ggml_mul(ctx, cv, conv_out);
    y = b.buildLoraMm(layer.shortconv.out_proj.?, y, null);
    b.cb(y, "model.layers.{}.conv.out_proj", il);
    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    y = c.ggml_reshape_2d(ctx, y, y.ne[0], n_seq_tokens * n_seqs);

    return y;
}

/// Ports `llm_build_lfm_base`'s `build_lfm_shortconv` (lfm-base.cpp:86
/// @cd959bae9), the overload with no recurrent state, for the decision
/// graphs: causal over the ubatch from a zero state.
///
/// Parameters:
/// - `model`: the model.
/// - `b`: the graph being built.
/// - `cur`: `[n_embd, n_tokens]`.
/// - `il`: the layer.
///
/// Return: `[n_embd, n_tokens]`.
pub fn buildLfmShortconvNoState(model: *const Model, b: *const graph.Builder, cur: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const d_conv: i64 = @as(i64, b.hparams.n_shortconv_l_cache) - 1;

    const bcx = b.buildLoraMm(layer.shortconv.in_proj.?, cur, null);
    b.cb(bcx, "model.layers.{}.conv.in_proj", il);

    const chunk_size = @divTrunc(bcx.ne[0], 3);
    const es = c.ggml_element_size(bcx);
    const chunk_bytes: usize = @intCast(chunk_size);
    const bv: *Tensor = c.ggml_view_2d(ctx, bcx, chunk_size, bcx.ne[1], bcx.nb[1], 0 * chunk_bytes * es);
    const cv: *Tensor = c.ggml_view_2d(ctx, bcx, chunk_size, bcx.ne[1], bcx.nb[1], 1 * chunk_bytes * es);
    const xv: *Tensor = c.ggml_view_2d(ctx, bcx, chunk_size, bcx.ne[1], bcx.nb[1], 2 * chunk_bytes * es);

    // Causal from a zero state.
    var bx: *Tensor = c.ggml_cont(ctx, c.ggml_transpose(ctx, c.ggml_mul(ctx, bv, xv)));
    bx = c.ggml_pad_ext(ctx, bx, @intCast(d_conv), 0, 0, 0, 0, 0, 0, 0);

    const conv_out: *Tensor = c.ggml_ssm_conv(ctx, bx, layer.shortconv.conv);
    b.cb(conv_out, "model.layers.{}.conv.conv", il);

    var y: *Tensor = c.ggml_mul(ctx, cv, c.ggml_reshape_2d(ctx, conv_out, chunk_size, b.n_tokens));
    y = b.buildLoraMm(layer.shortconv.out_proj.?, y, null);
    b.cb(y, "model.layers.{}.conv.out_proj", il);

    return y;
}

/// Ports `llm_build_lfm_base`'s `build_lfm_dense_ffn` (lfm-base.cpp:114
/// @8642541b2).
///
/// Parameters:
/// - `model`: the model.
/// - `b`: the graph being built.
/// - `cur`: the normalized input.
/// - `il`: the layer.
///
/// Return: the feed-forward's output.
pub fn buildLfmDenseFfn(model: *const Model, b: *const graph.Builder, cur: *Tensor, il: c_int) *Tensor {
    const layer = &model.layers[@intCast(il)];
    impl.assert(layer.ffn_up_b == null, "models/lfm-base.cpp", 116, "!model.layers[il].ffn_up_b");
    impl.assert(layer.ffn_gate_b == null, "models/lfm-base.cpp", 117, "!model.layers[il].ffn_gate_b");
    impl.assert(layer.ffn_down_b == null, "models/lfm-base.cpp", 118, "!model.layers[il].ffn_down_b");
    return b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
}

/// Ports `llm_build_lfm_base`'s `build_lfm_moe_ffn` (lfm-base.cpp:127
/// @8642541b2).
///
/// Parameters:
/// - `model`: the model.
/// - `b`: the graph being built.
/// - `cur`: the normalized input.
/// - `il`: the layer.
///
/// Return: the experts' output.
pub fn buildLfmMoeFfn(model: *const Model, b: *const graph.Builder, cur: *Tensor, il: c_int) *Tensor {
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    return b.buildMoeFfn(cur, .{
        .gate_inp = layer.ffn_gate_inp,
        .up_exps = layer.ffn_up_exps,
        .gate_exps = layer.ffn_gate_exps,
        .down_exps = layer.ffn_down_exps.?,
        .exp_probs_b = layer.ffn_exp_probs_b,
        .n_expert = b.n_expert,
        .n_expert_used = b.n_expert_used,
        .type_op = .silu,
        .norm_w = true,
        .w_scale = hp.expert_weights_scale,
        .gating_op = @fromBackingInt(@intCast(hp.expert_gating_func)),
    }, il);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
