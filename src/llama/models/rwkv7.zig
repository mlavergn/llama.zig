//! RWKV-7 ("Goose"): an attention-free recurrent model whose blocks are a
//! WKV-7 time mix and a squared-ReLU channel mix, each fed the previous
//! token through a stored token shift.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/rwkv7.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_model_rwkv7`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`. The shared blocks are in
//! `rwkv7_base.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const rwkv7_base = @import("rwkv7_base.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_rwkv7`'s `load_arch_hparams` (models/rwkv7.cpp:3 @b29c606e2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    try loadHparams(self, ml, exc);
}

/// The body `llama_model_rwkv7`'s and `llama_model_arwkv7`'s
/// `load_arch_hparams` share, keys and size table alike.
pub fn loadHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    if (try meta.getKey(f32, .attention_layernorm_eps, false, exc)) |v| hp.f_norm_eps = v;
    if (try meta.getKey(f32, .attention_layernorm_rms_eps, false, exc)) |v| hp.f_norm_rms_eps = v;
    hp.wkv_head_size = (try meta.getKey(u32, .wkv_head_size, true, exc)).?;
    hp.n_lora_decay = (try meta.getKey(u32, .attention_decay_lora_rank, true, exc)).?;
    hp.n_lora_iclr = (try meta.getKey(u32, .attention_iclr_lora_rank, true, exc)).?;
    hp.n_lora_value_res_mix = (try meta.getKey(u32, .attention_value_residual_mix_lora_rank, true, exc)).?;
    if (try meta.getKey(u32, .attention_gate_lora_rank, false, exc)) |v| hp.n_lora_gate = v;
    if (try meta.getKey(u32, .token_shift_count, false, exc)) |v| hp.token_shift_count = v;

    self.type = switch (hp.nLayer()) {
        12 => switch (hp.n_embd) {
            768 => .@"190m",
            else => .unknown,
        },
        24 => switch (hp.n_embd) {
            1024 => .@"450m",
            2048 => .@"1_5b",
            else => .unknown,
        },
        28 => switch (hp.n_embd) {
            1536 => .@"1_5b",
            3584 => .@"7b",
            else => .unknown,
        },
        32 => switch (hp.n_embd) {
            2560 => .@"2_9b",
            4096 => .@"7b",
            else => .unknown,
        },
        61 => switch (hp.n_embd) {
            4096 => .@"14b",
            else => .unknown,
        },
        else => .unknown,
    };
}

/// Ports `llama_model_rwkv7`'s `load_arch_tensors` (models/rwkv7.cpp:46 @b29c606e2).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Block 0, LN0.
    self.tok_norm = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "weight", .bid = 0 }, &.{l.n_embd}, 0, exc);
    self.tok_norm_b = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "bias", .bid = 0 }, &.{l.n_embd}, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    const n_lora_decay: i64 = @as(c_int, @bitCast(hp.n_lora_decay));
    const n_lora_iclr: i64 = @as(c_int, @bitCast(hp.n_lora_iclr));
    const n_lora_value_res_mix: i64 = @as(c_int, @bitCast(hp.n_lora_value_res_mix));
    const n_lora_gate: i64 = @as(c_int, @bitCast(hp.n_lora_gate));
    const attn_hidden_size: i64 = l.n_embd;
    const ffn_size: i64 = @as(c_int, @bitCast(hp.n_ff_arr[0]));

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.attn_norm_2 = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_2_b = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        try createTimeMixLora(self, layer, i, l.n_embd, n_lora_decay, n_lora_iclr, n_lora_value_res_mix, exc);

        layer.time_mix_g1 = try self.createTensor(.{ .tensor = .time_mix_g1, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_lora_gate }, 0, exc);
        layer.time_mix_g2 = try self.createTensor(.{ .tensor = .time_mix_g2, .suffix = "weight", .bid = i }, &.{ n_lora_gate, l.n_embd }, 0, exc);

        layer.time_mix_lerp_fused = try self.createTensor(.{ .tensor = .time_mix_lerp_fused, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1, 6 }, 0, exc);

        try createTimeMixKr(self, layer, i, l.n_embd, attn_hidden_size, exc);

        layer.time_mix_ln = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.time_mix_ln_b = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.time_mix_output = try self.createTensor(.{ .tensor = .time_mix_output, .suffix = "weight", .bid = i }, &.{ l.n_embd, attn_hidden_size }, 0, exc);

        layer.channel_mix_lerp_k = try self.createTensor(.{ .tensor = .channel_mix_lerp_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, 0, exc);

        layer.channel_mix_key = try self.createTensor(.{ .tensor = .channel_mix_key, .suffix = "weight", .bid = i }, &.{ l.n_embd, ffn_size }, 0, exc);
        layer.channel_mix_value = try self.createTensor(.{ .tensor = .channel_mix_value, .suffix = "weight", .bid = i }, &.{ ffn_size, l.n_embd }, 0, exc);
    }
}

/// The decay, in-context learning rate and value-residual tensors, from
/// `time_mix_w0` on, which RWKV-7 and ARWKV-7 load alike
/// (models/rwkv7.cpp:76 @b29c606e2, and `time_mix_w0` in
/// models/arwkv7.cpp:66 @b29c606e2). The first layer's value-residual
/// tensors are loaded at the ICLR rank and never used.
pub fn createTimeMixLora(self: *Model, layer: *model_mod.Layer, i: i32, n_embd: i64, n_lora_decay: i64, n_lora_iclr: i64, n_lora_value_res_mix: i64, exc: *Exception) Thrown!void {
    layer.time_mix_w0 = try self.createTensor(.{ .tensor = .time_mix_w0, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);
    layer.time_mix_w1 = try self.createTensor(.{ .tensor = .time_mix_w1, .suffix = "weight", .bid = i }, &.{ n_embd, n_lora_decay }, 0, exc);
    layer.time_mix_w2 = try self.createTensor(.{ .tensor = .time_mix_w2, .suffix = "weight", .bid = i }, &.{ n_lora_decay, n_embd }, 0, exc);

    layer.time_mix_a0 = try self.createTensor(.{ .tensor = .time_mix_a0, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);
    layer.time_mix_a1 = try self.createTensor(.{ .tensor = .time_mix_a1, .suffix = "weight", .bid = i }, &.{ n_embd, n_lora_iclr }, 0, exc);
    layer.time_mix_a2 = try self.createTensor(.{ .tensor = .time_mix_a2, .suffix = "weight", .bid = i }, &.{ n_lora_iclr, n_embd }, 0, exc);

    if (i == 0) {
        // Actually not used.
        layer.time_mix_v0 = try self.createTensor(.{ .tensor = .time_mix_v0, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);
        layer.time_mix_v1 = try self.createTensor(.{ .tensor = .time_mix_v1, .suffix = "weight", .bid = i }, &.{ n_embd, n_lora_iclr }, 0, exc);
        layer.time_mix_v2 = try self.createTensor(.{ .tensor = .time_mix_v2, .suffix = "weight", .bid = i }, &.{ n_lora_iclr, n_embd }, 0, exc);
    } else {
        layer.time_mix_v0 = try self.createTensor(.{ .tensor = .time_mix_v0, .suffix = "weight", .bid = i }, &.{n_embd}, 0, exc);
        layer.time_mix_v1 = try self.createTensor(.{ .tensor = .time_mix_v1, .suffix = "weight", .bid = i }, &.{ n_embd, n_lora_value_res_mix }, 0, exc);
        layer.time_mix_v2 = try self.createTensor(.{ .tensor = .time_mix_v2, .suffix = "weight", .bid = i }, &.{ n_lora_value_res_mix, n_embd }, 0, exc);
    }
}

/// The key/receptance mixing vectors and the key, value and receptance
/// projections, from `time_mix_k_k` on, which RWKV-7 and ARWKV-7 load
/// alike (models/rwkv7.cpp:100 @b29c606e2, and `time_mix_k_k` in
/// models/arwkv7.cpp:95 @b29c606e2).
pub fn createTimeMixKr(self: *Model, layer: *model_mod.Layer, i: i32, n_embd: i64, attn_hidden_size: i64, exc: *Exception) Thrown!void {
    layer.time_mix_k_k = try self.createTensor(.{ .tensor = .time_mix_k_k, .suffix = "weight", .bid = i }, &.{attn_hidden_size}, 0, exc);
    layer.time_mix_k_a = try self.createTensor(.{ .tensor = .time_mix_k_a, .suffix = "weight", .bid = i }, &.{attn_hidden_size}, 0, exc);
    layer.time_mix_r_k = try self.createTensor(.{ .tensor = .time_mix_r_k, .suffix = "weight", .bid = i }, &.{attn_hidden_size}, 0, exc);

    layer.time_mix_key = try self.createTensor(.{ .tensor = .time_mix_key, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, n_embd }, 0, exc);
    layer.time_mix_value = try self.createTensor(.{ .tensor = .time_mix_value, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, n_embd }, 0, exc);
    layer.time_mix_receptance = try self.createTensor(.{ .tensor = .time_mix_receptance, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, n_embd }, 0, exc);
}

/// Ports `llama_model_rwkv7`'s `build_arch_graph` (models/rwkv7.cpp:120 @b29c606e2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_rwkv7`'s `graph` (models/rwkv7.cpp:124 @b29c606e2).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    impl.assert(hp.token_shift_count == 2, "models/rwkv7.cpp", 126, "hparams.token_shift_count == 2");

    var v_first: ?*Tensor = null;

    var inpL = b.buildInpEmbd(model.tok_embd.?);
    inpL = b.buildNorm(inpL, model.tok_norm, model.tok_norm_b, .norm, 0);

    const rs_inp = b.buildRsInp();

    const n_embd: i64 = hp.n_embd;
    const n_seq_tokens: i64 = ubatch.n_seq_tokens;
    const n_seqs: i64 = ubatch.n_seqs;
    const n_tokens = b.n_tokens;

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        inpL = c.ggml_reshape_3d(ctx, inpL, n_embd, n_seq_tokens, n_seqs);

        var token_shift = b.buildRwkvTokenShiftLoad(rs_inp, ubatch, il);

        const att_shift: *Tensor = c.ggml_view_3d(ctx, token_shift, n_embd, 1, n_seqs, token_shift.nb[1], token_shift.nb[2], 0);
        const ffn_shift: *Tensor = c.ggml_view_3d(ctx, token_shift, n_embd, 1, n_seqs, token_shift.nb[1], token_shift.nb[2], @as(usize, @intCast(n_embd)) * c.ggml_element_size(token_shift));

        const att_norm = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .norm, il);
        b.cb(att_norm, "attn_norm", il);

        var x_prev: *Tensor = c.ggml_concat(ctx, att_shift, c.ggml_view_3d(ctx, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm.nb[1], att_norm.nb[2], 0), 1);

        const cur_tm = rwkv7_base.buildRwkv7TimeMix(model, b, rs_inp, att_norm, x_prev, &v_first, ubatch, il);

        var ffn_inp: *Tensor = c.ggml_add(ctx, cur_tm, inpL);
        b.cb(ffn_inp, "ffn_inp", il);

        var ffn_norm = b.buildNorm(ffn_inp, layer.attn_norm_2, layer.attn_norm_2_b, .norm, il);
        b.cb(ffn_norm, "ffn_norm", il);

        x_prev = c.ggml_concat(ctx, ffn_shift, c.ggml_view_3d(ctx, ffn_norm, n_embd, n_seq_tokens - 1, n_seqs, ffn_norm.nb[1], ffn_norm.nb[2], 0), 1);

        const last: usize = @intCast((n_seq_tokens - 1) * n_embd);
        token_shift = c.ggml_concat(
            ctx,
            c.ggml_view_3d(ctx, att_norm, n_embd, 1, n_seqs, att_norm.nb[1], att_norm.nb[2], last * c.ggml_element_size(att_norm)),
            c.ggml_view_3d(ctx, ffn_norm, n_embd, 1, n_seqs, ffn_norm.nb[1], ffn_norm.nb[2], last * c.ggml_element_size(ffn_norm)),
            1,
        );
        c.ggml_build_forward_expand(b.gf, b.buildRwkvTokenShiftStore(token_shift, ubatch, il));

        ffn_inp = c.ggml_reshape_2d(ctx, ffn_inp, n_embd, n_tokens);
        ffn_norm = c.ggml_reshape_2d(ctx, ffn_norm, n_embd, n_tokens);
        x_prev = c.ggml_reshape_2d(ctx, x_prev, n_embd, n_tokens);

        if (il == b.n_layer - 1) {
            ffn_inp = c.ggml_get_rows(ctx, ffn_inp, inp_out_ids);
            ffn_norm = c.ggml_get_rows(ctx, ffn_norm, inp_out_ids);
            x_prev = c.ggml_get_rows(ctx, x_prev, inp_out_ids);
        }
        var cur = rwkv7_base.buildRwkv7ChannelMix(b, layer, ffn_norm, x_prev, .rwkv7);
        cur = c.ggml_add(ctx, cur, ffn_inp);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;
    cur = b.buildNorm(cur, model.output_norm, model.output_norm_b, .norm, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
