//! ARWKV-7: a transformer distilled into RWKV-7's time mix, keeping its
//! RMS norms and SwiGLU feed-forward; the gate and the group norm are
//! optional.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/arwkv7.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_arwkv7`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`. The time mix is in
//! `rwkv7_base.zig`.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_arwkv7`'s `load_arch_hparams` (models/arwkv7.cpp:3
/// @b1b770d90): the same keys and size table as RWKV-7.
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    try llama.models.rwkv7.loadHparams(self, ml, exc);
}

/// Ports `llama_model_arwkv7`'s `load_arch_tensors` (models/arwkv7.cpp:46 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;

    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    // Output.
    self.output_norm = try self.createTensor(.{ .tensor = .output_norm, .suffix = "weight" }, &.{l.n_embd}, 0, exc);
    self.output = try self.createTensor(.{ .tensor = .output, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);

    const n_lora_decay: i64 = @as(c_int, @bitCast(hp.n_lora_decay));
    const n_lora_iclr: i64 = @as(c_int, @bitCast(hp.n_lora_iclr));
    const n_lora_value_res_mix: i64 = @as(c_int, @bitCast(hp.n_lora_value_res_mix));
    const n_lora_gate: i64 = @as(c_int, @bitCast(hp.n_lora_gate));
    const attn_hidden_size: i64 = l.n_embd;

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        layer.attn_norm = try self.createTensor(.{ .tensor = .attn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        try llama.models.rwkv7.createTimeMixLora(self, layer, i, l.n_embd, n_lora_decay, n_lora_iclr, n_lora_value_res_mix, exc);

        layer.time_mix_g1 = try self.createTensor(.{ .tensor = .time_mix_g1, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_lora_gate }, nr, exc);
        layer.time_mix_g2 = try self.createTensor(.{ .tensor = .time_mix_g2, .suffix = "weight", .bid = i }, &.{ n_lora_gate, l.n_embd }, nr, exc);

        // The C++ catches the `std::runtime_error` of the six-way shape:
        // ARWKV models may not have gate tensors.
        layer.time_mix_lerp_fused = self.createTensor(.{ .tensor = .time_mix_lerp_fused, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1, 6 }, 0, exc) catch |e| blk: {
            if (exc.kind != .runtime_error) return e;
            break :blk try self.createTensor(.{ .tensor = .time_mix_lerp_fused, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1, 5 }, 0, exc);
        };

        try llama.models.rwkv7.createTimeMixKr(self, layer, i, l.n_embd, attn_hidden_size, exc);

        layer.time_mix_ln = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.time_mix_ln_b = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.time_mix_output = try self.createTensor(.{ .tensor = .time_mix_output, .suffix = "weight", .bid = i }, &.{ l.n_embd, attn_hidden_size }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_arwkv7`'s `build_arch_graph` (models/arwkv7.cpp:116 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_arwkv7`'s `graph`
/// (models/arwkv7.cpp:120 @b1b770d90): one token shift, before the time
/// mix only.
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    llama.impl.assert(b.n_embd == hp.nEmbdR(), "models/arwkv7.cpp", 121, "n_embd == hparams.n_embd_r()");

    var v_first: ?*llama.impl.llama_c.ggml_tensor = null;

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

        var cur = llama.models.rwkv7_base.buildRwkv7TimeMix(model, b, rs_inp, att_norm, x_prev, &v_first, ubatch, il);

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
