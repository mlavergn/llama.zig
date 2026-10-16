//! ARWKV-7: a transformer distilled into RWKV-7's time mix, keeping its
//! RMS norms and SwiGLU feed-forward; the gate and the group norm are
//! optional.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/arwkv7.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_arwkv7`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`. The time mix is in
//! `rwkv7_base.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const rwkv7 = @import("rwkv7.zig");
const rwkv7_base = @import("rwkv7_base.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_arwkv7`'s `load_arch_hparams` (models/arwkv7.cpp:3
/// @c1d0e7a00): the same keys and size table as RWKV-7.
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    try rwkv7.loadHparams(self, ml, exc);
}

/// Ports `llama_model_arwkv7`'s `load_arch_tensors` (models/arwkv7.cpp:46 @c1d0e7a00).
pub fn loadArchTensors(self: *Model, _: *loader.Loader, exc: *Exception) Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = loader.tensor_not_required;

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

        try rwkv7.createTimeMixLora(self, layer, i, l.n_embd, n_lora_decay, n_lora_iclr, n_lora_value_res_mix, exc);

        layer.time_mix_g1 = try self.createTensor(.{ .tensor = .time_mix_g1, .suffix = "weight", .bid = i }, &.{ l.n_embd, n_lora_gate }, nr, exc);
        layer.time_mix_g2 = try self.createTensor(.{ .tensor = .time_mix_g2, .suffix = "weight", .bid = i }, &.{ n_lora_gate, l.n_embd }, nr, exc);

        // The C++ catches the `std::runtime_error` of the six-way shape:
        // ARWKV models may not have gate tensors.
        layer.time_mix_lerp_fused = self.createTensor(.{ .tensor = .time_mix_lerp_fused, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1, 6 }, 0, exc) catch |e| blk: {
            if (exc.kind != .runtime_error) return e;
            break :blk try self.createTensor(.{ .tensor = .time_mix_lerp_fused, .suffix = "weight", .bid = i }, &.{ l.n_embd, 1, 1, 5 }, 0, exc);
        };

        try rwkv7.createTimeMixKr(self, layer, i, l.n_embd, attn_hidden_size, exc);

        layer.time_mix_ln = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "weight", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.time_mix_ln_b = try self.createTensor(.{ .tensor = .time_mix_ln, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
        layer.time_mix_output = try self.createTensor(.{ .tensor = .time_mix_output, .suffix = "weight", .bid = i }, &.{ l.n_embd, attn_hidden_size }, 0, exc);

        layer.ffn_norm = try self.createTensor(.{ .tensor = .ffn_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);

        layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
    }
}

/// Ports `llama_model_arwkv7`'s `build_arch_graph` (models/arwkv7.cpp:116 @c1d0e7a00).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_arwkv7`'s `graph`
/// (models/arwkv7.cpp:120 @c1d0e7a00): one token shift, before the time
/// mix only.
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    impl.assert(b.n_embd == hp.nEmbdR(), "models/arwkv7.cpp", 121, "n_embd == hparams.n_embd_r()");

    var v_first: ?*Tensor = null;

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
        inpL = c.ggml_reshape_3d(ctx, inpL, n_embd, n_seq_tokens, n_seqs);

        var token_shift = b.buildRwkvTokenShiftLoad(rs_inp, ubatch, il);

        const att_norm = b.buildNorm(inpL, layer.attn_norm, layer.attn_norm_b, .rms, il);
        b.cb(att_norm, "attn_norm", il);

        const x_prev: *Tensor = c.ggml_concat(ctx, token_shift, c.ggml_view_3d(ctx, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm.nb[1], att_norm.nb[2], 0), 1);

        var cur = rwkv7_base.buildRwkv7TimeMix(model, b, rs_inp, att_norm, x_prev, &v_first, ubatch, il);

        const last: usize = @intCast((n_seq_tokens - 1) * n_embd);
        token_shift = c.ggml_view_3d(ctx, att_norm, n_embd, 1, n_seqs, att_norm.nb[1], att_norm.nb[2], last * c.ggml_element_size(att_norm));
        c.ggml_build_forward_expand(b.gf, b.buildRwkvTokenShiftStore(token_shift, ubatch, il));

        var ffn_inp: *Tensor = c.ggml_add(ctx, cur, inpL);
        b.cb(ffn_inp, "ffn_inp", il);

        cur = c.ggml_reshape_2d(ctx, cur, n_embd, n_tokens);
        ffn_inp = c.ggml_reshape_2d(ctx, ffn_inp, n_embd, n_tokens);

        if (il == b.n_layer - 1) {
            cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
            ffn_inp = c.ggml_get_rows(ctx, ffn_inp, inp_out_ids);
        }
        // The feed-forward network.
        cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
        b.cb(cur, "ffn_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_inp);

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

    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
