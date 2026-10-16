//! RWKV-6 ("Finch"): an attention-free recurrent model whose blocks are a
//! WKV-6 time mix and a squared-ReLU channel mix, each fed the previous
//! token through a stored token shift.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/rwkv6.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_rwkv6`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`. The shared blocks are in
//! `rwkv6_base.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const rwkv6_base = @import("rwkv6_base.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_rwkv6`'s `load_arch_hparams` (models/rwkv6.cpp:3 @c1d0e7a00).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    const meta = &ml.meta;
    const hp = &self.hparams;
    if (try meta.getKey(f32, .attention_layernorm_eps, false, exc)) |v| hp.f_norm_eps = v;
    if (try meta.getKey(f32, .attention_layernorm_rms_eps, false, exc)) |v| hp.f_norm_rms_eps = v;
    hp.wkv_head_size = (try meta.getKey(u32, .wkv_head_size, true, exc)).?;
    hp.time_mix_extra_dim = (try meta.getKey(u32, .time_mix_extra_dim, true, exc)).?;
    hp.time_decay_extra_dim = (try meta.getKey(u32, .time_decay_extra_dim, true, exc)).?;
    if (try meta.getKey(u32, .rescale_every_n_layers, false, exc)) |v| hp.rescale_every_n_layers = v;
    if (try meta.getKey(u32, .token_shift_count, false, exc)) |v| hp.token_shift_count = v;

    self.type = typeOf(hp.nLayer(), hp.n_embd);
}

/// The model size `load_arch_hparams` reads off the layer count and the
/// width; RWKV6-Qwen2 uses the same table.
pub fn typeOf(n_layer: u32, n_embd: u32) @FieldType(Model, "type") {
    return switch (n_layer) {
        24 => .@"1_6b",
        32 => switch (n_embd) {
            2560 => .@"3b",
            4096 => .@"7b",
            else => .unknown,
        },
        61 => .@"14b",
        64 => .@"32b",
        else => .unknown,
    };
}

/// Ports `llama_model_rwkv6`'s `load_arch_tensors` (models/rwkv6.cpp:26 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Block 0, LN0.
    self.tok_norm = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "weight", .bid = 0 }, &.{l.n_embd}, 0, exc);
    self.tok_norm_b = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "bias", .bid = 0 }, &.{l.n_embd}, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output_norm_b = try self.createTensor(.{ .tensor = .output_norm, .suffix = "bias" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    const time_mix_extra_dim: i64 = @as(c_int, @bitCast(hp.time_mix_extra_dim));
    const time_decay_extra_dim: i64 = @as(c_int, @bitCast(hp.time_decay_extra_dim));
    const head_size: i64 = @as(c_int, @bitCast(hp.wkv_head_size));
    const attn_hidden_size: i64 = l.n_embd;
    const ffn_size: i64 = @as(c_int, @bitCast(hp.n_ff_arr[0]));

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_b = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.attn_norm_2 = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_norm_2_b = try self.createTensor(.{ .tensor = .attn_norm_2, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.time_mix_w1 = try self.createTensor(.{ .tensor = .time_mix_w1, .suffix = "weight", .bid = i }, &.{ l.n_embd, time_mix_extra_dim * 5 }, 0, exc);
        layer.time_mix_w2 = try self.createTensor(.{ .tensor = .time_mix_w2, .suffix = "weight", .bid = i }, &.{ time_mix_extra_dim, l.n_embd, 5 }, 0, exc);

        layer.time_mix_lerp_x = try self.createTensor(.{ .tensor = .time_mix_lerp_x, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, 0, exc);
        layer.time_mix_lerp_w = try self.createTensor(.{ .tensor = .time_mix_lerp_w, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, nr, exc);
        layer.time_mix_lerp_k = try self.createTensor(.{ .tensor = .time_mix_lerp_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, nr, exc);
        layer.time_mix_lerp_v = try self.createTensor(.{ .tensor = .time_mix_lerp_v, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, nr, exc);
        layer.time_mix_lerp_r = try self.createTensor(.{ .tensor = .time_mix_lerp_r, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, nr, exc);
        layer.time_mix_lerp_g = try self.createTensor(.{ .tensor = .time_mix_lerp_g, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, nr, exc);
        layer.time_mix_lerp_fused = try self.createTensor(.{ .tensor = .time_mix_lerp_fused, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1, 5 }, nr, exc);
        impl.assert(!(layer.time_mix_lerp_fused == null and layer.time_mix_lerp_w == null), "models/rwkv6.cpp", 65, "!(layer.time_mix_lerp_fused == NULL && layer.time_mix_lerp_w == NULL)");

        layer.time_mix_first = try self.createTensor(.{ .tensor = .time_mix_first, .suffix = "weight", .bid = i }, &.{ head_size, @divTrunc(l.n_embd, head_size) }, 0, exc);
        layer.time_mix_decay = try self.createTensor(.{ .tensor = .time_mix_decay, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.time_mix_decay_w1 = try self.createTensor(.{ .tensor = .time_mix_decay_w1, .suffix = "weight", .bid = i }, &.{ l.n_embd, time_decay_extra_dim }, 0, exc);
        layer.time_mix_decay_w2 = try self.createTensor(.{ .tensor = .time_mix_decay_w2, .suffix = "weight", .bid = i }, &.{ time_decay_extra_dim, attn_hidden_size }, 0, exc);
        layer.time_mix_key = try self.createTensor(.{ .tensor = .time_mix_key, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, l.n_embd }, 0, exc);
        layer.time_mix_value = try self.createTensor(.{ .tensor = .time_mix_value, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, l.n_embd }, 0, exc);
        layer.time_mix_receptance = try self.createTensor(.{ .tensor = .time_mix_receptance, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, l.n_embd }, 0, exc);
        layer.time_mix_gate = try self.createTensor(.{ .tensor = .time_mix_gate, .suffix = "weight", .bid = i }, &.{ attn_hidden_size, l.n_embd }, 0, exc);

        layer.time_mix_ln = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.time_mix_ln_b = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.time_mix_output = try self.createTensor(.{ .tensor = .time_mix_output, .suffix = "weight", .bid = i }, &.{ l.n_embd, attn_hidden_size }, 0, exc);

        layer.channel_mix_lerp_k = try self.createTensor(.{ .tensor = .channel_mix_lerp_k, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, 0, exc);
        layer.channel_mix_lerp_r = try self.createTensor(.{ .tensor = .channel_mix_lerp_r, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1 }, 0, exc);

        layer.channel_mix_key = try self.createTensor(.{ .tensor = .channel_mix_key, .suffix = "weight", .bid = i }, &.{ l.n_embd, ffn_size }, 0, exc);
        layer.channel_mix_value = try self.createTensor(.{ .tensor = .channel_mix_value, .suffix = "weight", .bid = i }, &.{ ffn_size, l.n_embd }, 0, exc);
        layer.channel_mix_receptance = try self.createTensor(.{ .tensor = .channel_mix_receptance, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
    }
}

/// Ports `llama_model_rwkv6`'s `build_arch_graph` (models/rwkv6.cpp:90 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_rwkv6`'s `graph` (models/rwkv6.cpp:94 @c1d0e7a00).
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    impl.assert(hp.token_shift_count == 2, "models/rwkv6.cpp", 96, "hparams.token_shift_count == 2");

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

        var cur = rwkv6_base.buildRwkv6TimeMix(model, b, rs_inp, att_norm, x_prev, ubatch, il);

        var ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
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
        cur = c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);

        if (il == b.n_layer - 1) {
            ffn_inp = c.ggml_get_rows(ctx, ffn_inp, inp_out_ids);
            ffn_norm = c.ggml_get_rows(ctx, ffn_norm, inp_out_ids);
            x_prev = c.ggml_get_rows(ctx, x_prev, inp_out_ids);
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
        }
        cur = rwkv6_base.buildRwkv6ChannelMix(b, layer, ffn_norm, x_prev, .rwkv6);
        cur = c.ggml_add(ctx, cur, ffn_inp);

        if (hp.rescale_every_n_layers != 0 and @as(u32, @intCast(il + 1)) % hp.rescale_every_n_layers == 0) {
            cur = c.ggml_scale(ctx, cur, 0.5);
        }
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
