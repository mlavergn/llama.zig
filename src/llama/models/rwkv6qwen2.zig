//! RWKV6-Qwen2: a Qwen2 model whose attention is replaced by RWKV-6's time
//! mix, keeping Qwen2's RMS norms and SwiGLU feed-forward; without
//! `time_mix_first` it is QRWKV, running gated linear attention instead of
//! WKV-6.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/rwkv6qwen2.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_rwkv6qwen2`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, and its `graph`. The time
//! mix is in `rwkv6_base.zig`.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_rwkv6qwen2`'s `load_arch_hparams` (models/rwkv6qwen2.cpp:3
/// @b1b770d90): the same keys and size table as RWKV-6.
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    if (try meta.getKey(f32, .attention_layernorm_eps, false, exc)) |v| hp.f_norm_eps = v;
    if (try meta.getKey(f32, .attention_layernorm_rms_eps, false, exc)) |v| hp.f_norm_rms_eps = v;
    hp.wkv_head_size = (try meta.getKey(u32, .wkv_head_size, true, exc)).?;
    hp.time_mix_extra_dim = (try meta.getKey(u32, .time_mix_extra_dim, true, exc)).?;
    hp.time_decay_extra_dim = (try meta.getKey(u32, .time_decay_extra_dim, true, exc)).?;
    if (try meta.getKey(u32, .rescale_every_n_layers, false, exc)) |v| hp.rescale_every_n_layers = v;
    if (try meta.getKey(u32, .token_shift_count, false, exc)) |v| hp.token_shift_count = v;

    self.type = llama.models.rwkv6.typeOf(hp.nLayer(), hp.n_embd);
}

/// Ports `llama_model_rwkv6qwen2`'s `load_arch_tensors` (models/rwkv6qwen2.cpp:26 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{l.n_embd}, nr, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    const time_mix_extra_dim: i64 = @as(c_int, @bitCast(hp.time_mix_extra_dim));
    const time_decay_extra_dim: i64 = @as(c_int, @bitCast(hp.time_decay_extra_dim));
    const head_size: c_int = @bitCast(hp.wkv_head_size);
    const attn_hidden_size: c_int = @intCast(l.n_embd);
    var attn_key_value_size: c_int = undefined;
    if (l.n_head_kv == 0 or @divTrunc(attn_hidden_size, head_size) == l.n_head_kv) {
        attn_key_value_size = attn_hidden_size;
    } else {
        attn_key_value_size = @intCast(l.n_head_kv * head_size);
    }

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.time_mix_w1 = try self.createTensor(.{ .tensor = .time_mix_w1, .suffix = "weight", .bid = i }, &.{ l.n_embd, time_mix_extra_dim * 5 }, 0, exc);
        layer.time_mix_w2 = try self.createTensor(.{ .tensor = .time_mix_w2, .suffix = "weight", .bid = i }, &.{ time_mix_extra_dim, l.n_embd, 5 }, 0, exc);

        layer.time_mix_lerp_x = try self.createTensor(.{ .tensor = .time_mix_lerp_x, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, 0, exc);
        layer.time_mix_lerp_fused = try self.createTensor(.{ .tensor = .time_mix_lerp_fused, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1, 5 }, 0, exc);

        layer.time_mix_first = try self.createTensor(.{ .tensor = .time_mix_first, .suffix = "weight", .bid = i }, &.{ head_size, @divTrunc(l.n_embd, head_size) }, nr, exc);
        layer.time_mix_decay = try self.createTensor(.{ .tensor = .time_mix_decay, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.time_mix_decay_w1 = try self.createTensor(.{ .tensor = .time_mix_decay_w1, .suffix = "weight", .bid = i }, &.{ l.n_embd, time_decay_extra_dim }, 0, exc);
        layer.time_mix_decay_w2 = try self.createTensor(.{ .tensor = .time_mix_decay_w2, .suffix = "weight", .bid = i }, &.{ time_decay_extra_dim, attn_hidden_size }, 0, exc);
        layer.time_mix_key = try self.createTensor(.{ .tensor = .time_mix_key, .suffix = "weight", .bid = i }, &.{ l.n_embd, attn_key_value_size }, 0, exc);
        layer.time_mix_value = try self.createTensor(.{ .tensor = .time_mix_value, .suffix = "weight", .bid = i }, &.{ l.n_embd, attn_key_value_size }, 0, exc);
        layer.time_mix_receptance = try self.createTensor(.{ .tensor = .time_mix_receptance, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, l.n_embd }, 0, exc);
        layer.time_mix_gate = try self.createTensor(.{ .tensor = .time_mix_gate, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, l.n_embd }, 0, exc);
        // Optional bias tensors.
        layer.time_mix_key_b = try self.createTensor(.{ .tensor = .time_mix_key, .suffix = "bias", .bid = i }, &.{attn_key_value_size}, nr, exc);
        layer.time_mix_value_b = try self.createTensor(.{ .tensor = .time_mix_value, .suffix = "bias", .bid = i }, &.{attn_key_value_size}, nr, exc);
        layer.time_mix_receptance_b = try self.createTensor(.{ .tensor = .time_mix_receptance, .suffix = "bias", .bid = i }, &.{attn_hidden_size}, nr, exc);

        layer.time_mix_output = try self.createTensor(.{ .tensor = .time_mix_output, .suffix = "weight", .bid = i }, &.{ l.n_embd, attn_hidden_size }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_rwkv6qwen2`'s `build_arch_graph` (models/rwkv6qwen2.cpp:80 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_rwkv6qwen2`'s `graph`
/// (models/rwkv6qwen2.cpp:84 @b1b770d90): one token shift, before the time
/// mix only.
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    llama.impl.assert(b.n_embd == hp.nEmbdR(), "models/rwkv6qwen2.cpp", 85, "n_embd == hparams.n_embd_r()");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const rs_inp = b.buildRsInp();

    const n_embd: i64 = hp.n_embd;
    const n_seq_tokens: i64 = ubatch.n_seq_tokens;
    const n_seqs: i64 = ubatch.n_seqs;
    const n_tokens = b.n_tokens;

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        inpL = llama.impl.llama_c.ggml_reshape_3d(ctx, inpL, n_embd, n_seq_tokens, n_seqs);

        var token_shift = b.buildRwkvTokenShiftLoad(rs_inp, ubatch, il);

        const att_norm = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .rms, il);
        b.cb(att_norm, "attn_norm", il);

        const x_prev: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_concat(ctx, token_shift, llama.impl.llama_c.ggml_view_3d(ctx, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm.nb[1], att_norm.nb[2], 0), 1);

        var cur = llama.models.rwkv6_base.buildRwkv6TimeMix(model, b, rs_inp, att_norm, x_prev, ubatch, il);

        const last: usize = @intCast((n_seq_tokens - 1) * n_embd);
        token_shift = llama.impl.llama_c.ggml_view_3d(ctx, att_norm, n_embd, 1, n_seqs, att_norm.nb[1], att_norm.nb[2], last * llama.impl.llama_c.ggml_element_size(att_norm));
        llama.impl.llama_c.ggml_build_forward_expand(b.gf, b.buildRwkvTokenShiftStore(token_shift, ubatch, il));

        var ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpL);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = llama.impl.llama_c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);
        ffn_inp = llama.impl.llama_c.ggml_reshape_2d(ctx, ffn_inp, n_embd, n_tokens);

        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            ffn_inp = llama.impl.llama_c.ggml_get_rows(ctx, ffn_inp, inp_out_ids);
        }

        // The feed-forward network.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }

    var cur = inpL;
    cur = b.buildNorm(cur, model.output_norm, model.output_norm_b, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
