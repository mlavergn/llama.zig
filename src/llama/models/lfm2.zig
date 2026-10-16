//! LFM2: Liquid's hybrid of gated short convolutions and grouped-query
//! attention with Q/K norms, each followed by a SwiGLU feed-forward or,
//! past the leading dense blocks, a mixture of experts; the attention
//! layers may use a sliding window.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/lfm2.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_lfm2`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph<iswa>` with the lambdas inside it.
//! LFM2-MoE (`lfm2moe.zig`) builds the same graph.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const hparams_mod = @import("../hparams.zig");
const memory_hybrid = @import("../memory_hybrid.zig");
const memory_hybrid_iswa = @import("../memory_hybrid_iswa.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_lfm2`'s `load_arch_hparams` (models/lfm2.cpp:7 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    hp.n_shortconv_l_cache = (try meta.getKey(u32, .shortconv_l_cache, true, exc)).?;
    hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;

    var il: u32 = 0;
    while (il < hp.nLayer()) : (il += 1) {
        hp.is_recr_impl[il] = @intFromBool(hp.nHeadKv(il) == 0);
    }

    hp.n_layer_dense_lead = hp.nLayer();

    self.type = switch (hp.nFf(0)) {
        2560 => .@"230m",
        4608 => .@"350m",
        6912 => .@"700m",
        8192 => .@"1_2b",
        10752 => .@"2_6b",
        else => .unknown,
    };

    if (try meta.getKey(u32, .attention_sliding_window, false, exc)) |v| {
        hp.n_swa = v;
        if (hp.n_swa > 0) {
            hp.swa_type = @backingInt(hparams_mod.SwaType.standard);
            il = 0;
            while (il < hp.nLayer()) : (il += 1) {
                hp.is_swa_impl[il] = @intFromBool(hp.is_recr_impl[il] == 0);
            }
        }
    }
}

/// Ports `llama_model_lfm2`'s `load_arch_tensors` (models/lfm2.cpp:34
/// @c1d0e7a00); LFM2-MoE's (models/lfm2moe.cpp:23 @c1d0e7a00) is the same
/// body.
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    try loadTensors(self, "models/lfm2.cpp", 54, 72, exc);
}

/// The body both `load_arch_tensors` share, with the lines its two
/// assertions are at in each file.
pub fn loadTensors(self: *Model, comptime file: [:0]const u8, comptime line_moe: c_int, comptime line_gqa: c_int, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm_lfm2, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, nr, exc);

    if (self.output == null) {
        self.output = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, loader.tensor_duplicated, exc);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        const is_moe_layer = i >= @as(c_int, @bitCast(hp.n_layer_dense_lead));

        // The FFN or MoE is the same for the attention and conv layers.
        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        if (is_moe_layer) {
            impl.assert(l.n_expert != 0 and l.n_expert_used != 0, file, line_moe, "n_expert && n_expert_used");
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_exps = try self.createTensor(.{ .tensor = .ffn_gate_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ hp.n_ff_exp, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, hp.n_ff_exp, l.n_expert }, 0, exc);
            layer.ffn_exp_probs_b = try self.createTensor(.{ .tensor = .ffn_exp_probs_b, .suffix = "bias", .bid = i }, &.{l.n_expert}, 0, exc);
        } else {
            // Dense.
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        }

        // For the operator norm.
        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        if (!hp.isRecr(@intCast(i))) {
            layer.attn_q_norm = try self.createTensor(.{ .tensor = .attn_q_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
            layer.attn_k_norm = try self.createTensor(.{ .tensor = .attn_k_norm, .suffix = "weight", .bid = i }, &.{l.n_embd_head_k}, 0, exc);
            impl.assert(l.n_embd_v_gqa == l.n_embd_k_gqa, file, line_gqa, "n_embd_v_gqa == n_embd_k_gqa");

            try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, hp.nEmbdKGqa(@intCast(i)), hp.nEmbdVGqa(@intCast(i)), 0, exc);

            layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        } else {
            layer.shortconv.conv = try self.createTensor(.{ .tensor = .shortconv_conv, .suffix = "weight", .bid = i }, &.{ hp.n_shortconv_l_cache, l.n_embd }, 0, exc);
            layer.shortconv.in_proj = try self.createTensor(.{ .tensor = .shortconv_inproj, .suffix = "weight", .bid = i }, &.{ l.n_embd, 3 * l.n_embd }, 0, exc);
            layer.shortconv.out_proj = try self.createTensor(.{ .tensor = .shortconv_outproj, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        }
    }

    // For LFM2-ColBert-350M.
    self.dense_2_out_layers = try self.createTensor(.{ .tensor = .dense_2_out, .suffix = "weight" }, &.{ l.n_embd, hp.nEmbdOut() }, nr, exc);
    self.dense_2_out_layers_b = try self.createTensor(.{ .tensor = .dense_2_out, .suffix = "bias" }, &.{hp.nEmbdOut()}, nr, exc);
}

/// Ports `llama_model_lfm2`'s `build_arch_graph` (models/lfm2.cpp:89
/// @c1d0e7a00); LFM2-MoE's (models/lfm2moe.cpp:78 @c1d0e7a00) is the same.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    const swa_type: hparams_mod.SwaType = @fromBackingInt(@intCast(model.hparams.swa_type));
    if (swa_type == .standard) {
        buildGraph(true, model, b);
    } else {
        buildGraph(false, model, b);
    }
}

/// The constructor's `build_dense_feed_forward` lambda (models/lfm2.cpp:105
/// @c1d0e7a00).
fn buildDenseFeedForward(model: *const Model, b: *const graph.Builder, cur: *Tensor, il: c_int) *Tensor {
    const layer = &model.layers[@intCast(il)];
    impl.assert(layer.ffn_up_b == null, "models/lfm2.cpp", 106, "!model.layers[il].ffn_up_b");
    impl.assert(layer.ffn_gate_b == null, "models/lfm2.cpp", 107, "!model.layers[il].ffn_gate_b");
    impl.assert(layer.ffn_down_b == null, "models/lfm2.cpp", 108, "!model.layers[il].ffn_down_b");
    return b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
}

/// The constructor's `build_moe_feed_forward` lambda (models/lfm2.cpp:115
/// @c1d0e7a00).
fn buildMoeFeedForward(model: *const Model, b: *const graph.Builder, cur: *Tensor, il: c_int) *Tensor {
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

/// The constructor's `build_attn_block` lambda (models/lfm2.cpp:128
/// @c1d0e7a00): Q/K norms, RoPE and attention.
fn buildAttnBlock(model: *const Model, b: *const graph.Builder, cur_: *Tensor, inp_pos: *Tensor, inp_attn: anytype, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    impl.assert(hp.nEmbdVGqa(@intCast(il)) == hp.nEmbdKGqa(@intCast(il)), "models/lfm2.cpp", 132, "hparams.n_embd_v_gqa(il) == hparams.n_embd_k_gqa(il)");
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_head_kv: i64 = hp.nHeadKv(@intCast(il));

    const qkv = b.buildQkv(layer, cur_, n_embd_head, b.n_head, n_head_kv, il);
    var q = qkv.q;
    var k = qkv.k;
    const v = qkv.v;

    // The QK norms.
    q = b.buildNorm(q, layer.attn_q_norm, null, .rms, il);
    b.cb(q, "model.layers.{}.self_attn.q_layernorm", il);
    k = b.buildNorm(k, layer.attn_k_norm, null, .rms, il);
    b.cb(k, "model.layers.{}.self_attn.k_layernorm", il);

    // RoPE.
    q = c.ggml_rope_ext(ctx, q, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    k = c.ggml_rope_ext(ctx, k, inp_pos, null, @intCast(b.n_rot), b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    const cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, q, k, v, null, null, null, 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)), il);

    b.cb(cur, "model.layers.{}.self_attn.out_proj", il);

    return cur;
}

/// The constructor's `build_shortconv_block` lambda (models/lfm2.cpp:159
/// @c1d0e7a00): a gated depthwise convolution over the last
/// `n_shortconv_l_cache` tokens, whose state is the recurrent memory.
fn buildShortconvBlock(comptime iswa: bool, model: *const Model, b: *const graph.Builder, cur_: *Tensor, inp_recr: *const graph.InpRs, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const layer = &model.layers[@intCast(il)];
    const MemHybridCtx = if (iswa) memory_hybrid_iswa.HybridIswaContext else memory_hybrid.HybridContext;
    const mctx_cur = b.mctx.?.cast(MemHybridCtx).getRecr();
    const kv_head: u32 = mctx_cur.getHead();
    const n_seq_tokens: i64 = b.ubatch.n_seq_tokens;
    const n_seqs: i64 = b.ubatch.n_seqs;
    impl.assert(n_seqs != 0, "models/lfm2.cpp", 166, "n_seqs != 0");
    impl.assert(b.ubatch.equalSeqs(), "models/lfm2.cpp", 167, "ubatch.equal_seqs()");
    impl.assert(b.ubatch.n_tokens == n_seq_tokens * n_seqs, "models/lfm2.cpp", 168, "ubatch.n_tokens == n_seq_tokens * n_seqs");

    impl.assert(hp.n_shortconv_l_cache > 1, "models/lfm2.cpp", 170, "hparams.n_shortconv_l_cache > 1");
    const d_conv: u32 = hp.n_shortconv_l_cache - 1;

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    var cur: *Tensor = c.ggml_reshape_3d(ctx, cur_, cur_.ne[0], n_seq_tokens, n_seqs);

    const bcx = b.buildLoraMm(layer.shortconv.in_proj.?, cur, null);
    b.cb(bcx, "model.layers.{}.conv.in_proj", il);

    const n_chunks = 3;
    impl.assert(@rem(bcx.ne[0], n_chunks) == 0, "models/lfm2.cpp", 180, "bcx->ne[0] % n_chunks == 0");
    const chunk_size = @divTrunc(bcx.ne[0], n_chunks);
    const es = c.ggml_element_size(bcx);
    const chunk_bytes: usize = @intCast(chunk_size);
    const bv: *Tensor = c.ggml_view_3d(ctx, bcx, chunk_size, bcx.ne[1], bcx.ne[2], bcx.nb[1], bcx.nb[2], 0 * chunk_bytes * es);
    const cv: *Tensor = c.ggml_view_3d(ctx, bcx, chunk_size, bcx.ne[1], bcx.ne[2], bcx.nb[1], bcx.nb[2], 1 * chunk_bytes * es);
    const xv: *Tensor = c.ggml_view_3d(ctx, bcx, chunk_size, bcx.ne[1], bcx.ne[2], bcx.nb[1], bcx.nb[2], 2 * chunk_bytes * es);

    var bx: *Tensor = c.ggml_transpose(ctx, c.ggml_mul(ctx, bv, xv));

    // Read the conv state.
    const conv_state = mctx_cur.getRL(il);
    const conv_rs = b.buildRs(inp_recr, conv_state, @intCast(hp.nEmbdR()), @intCast(n_seqs), null);
    const conv: *Tensor = c.ggml_reshape_3d(ctx, conv_rs, d_conv, hp.n_embd, n_seqs);

    // Causal prepends the state; non-causal pads symmetrically for a
    // centered window.
    if (hp.causal_attn) {
        bx = c.ggml_concat(ctx, conv, bx, 0);
    } else {
        const pad: i64 = @divTrunc(@as(i64, hp.n_shortconv_l_cache) - 1, 2);
        const left: *Tensor = c.ggml_cont(ctx, c.ggml_view_3d(ctx, conv, pad, hp.n_embd, n_seqs, conv.nb[1], conv.nb[2], @as(usize, @intCast(@as(i64, d_conv) - pad)) * conv.nb[0]));
        bx = c.ggml_pad_ext(ctx, c.ggml_concat(ctx, left, bx, 0), 0, @intCast(pad), 0, 0, 0, 0, 0, 0);
    }
    impl.assert(bx.ne[0] > conv.ne[0], "models/lfm2.cpp", 205, "bx->ne[0] > conv->ne[0]");

    // Write the conv states: slot 0 is the final state, slot s the state s
    // tokens back (for a partial rollback).
    const K: i64 = if (hp.causal_attn and b.cparams.n_rs_seq > 0) @as(i64, b.cparams.n_rs_seq) + 1 else 1;
    const n_written: i64 = @min(n_seq_tokens, K);
    const mem_size: usize = mctx_cur.getSize();
    const row_size: usize = c.ggml_row_size(conv_state.type, @as(i64, d_conv) * b.n_embd);

    var slot: i64 = 0;
    while (slot < n_written) : (slot += 1) {
        const conv_snap: *Tensor = c.ggml_view_3d(ctx, bx, d_conv, bx.ne[1], bx.ne[2], bx.nb[1], bx.nb[2], @as(usize, @intCast(bx.ne[0] - d_conv - slot)) * c.ggml_element_size(bx));
        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, conv_snap, c.ggml_view_2d(ctx, conv_state, @as(i64, d_conv) * b.n_embd, n_seqs, conv_state.nb[1], (@as(usize, @intCast(slot)) * mem_size + kv_head) * row_size)));
    }

    const conv_kernel = layer.shortconv.conv;
    const conv_out: *Tensor = c.ggml_ssm_conv(ctx, bx, conv_kernel);
    b.cb(conv_out, "model.layers.{}.conv.conv", il);

    var y: *Tensor = c.ggml_mul(ctx, cv, conv_out);
    y = b.buildLoraMm(layer.shortconv.out_proj.?, y, null);
    b.cb(y, "model.layers.{}.conv.out_proj", il);
    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    y = c.ggml_reshape_2d(ctx, y, y.ne[0], n_seq_tokens * n_seqs);

    cur = y;
    return cur;
}

/// Ports the constructor of `llama_model_lfm2`'s `graph<iswa>`
/// (models/lfm2.cpp:98 @c1d0e7a00).
fn buildGraph(comptime iswa: bool, model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;

    var cur = b.buildInpEmbd(model.tok_embd.?);
    b.cb(cur, "model.embed_tokens", -1);

    c.ggml_build_forward_expand(b.gf, cur);

    const inp_hybrid = if (iswa) b.buildInpMemHybridIswa() else b.buildInpMemHybrid();

    const inp_pos = b.buildInpPos();
    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        b.res.t_layer_inp[@intCast(il)] = cur;

        const is_moe_layer = il >= @as(c_int, @bitCast(hp.n_layer_dense_lead));

        var prev_cur = cur;
        cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
        b.cb(cur, "model.layers.{}.operator_norm", il);

        cur = if (hp.isRecr(@intCast(il)))
            buildShortconvBlock(iswa, model, b, cur, inp_hybrid.getRecr(), il)
        else
            buildAttnBlock(model, b, cur, inp_pos, inp_hybrid.getAttn(), il);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            prev_cur = c.ggml_get_rows(ctx, prev_cur, inp_out_ids);
        }

        cur = c.ggml_add(ctx, prev_cur, cur);

        const ffn_norm_out = b.buildNorm(cur, layer.ffn_norm, null, .rms, il);
        b.cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        const ffn_out = if (is_moe_layer) buildMoeFeedForward(model, b, ffn_norm_out, il) else buildDenseFeedForward(model, b, ffn_norm_out, il);
        // The C++ names the norm's output here, not the FFN's.
        b.cb(ffn_norm_out, "model.layers.{}.ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_out);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);
    }

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    if (!b.cparams.embeddings) {
        cur = b.buildLoraMm(model.output.?, cur, model.output_s);
        b.cb(cur, "result_output", -1);

        b.res.t_logits = cur;
    }

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
