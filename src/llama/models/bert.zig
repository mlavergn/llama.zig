//! BERT: the bidirectional encoder -- learned absolute positions, token-type
//! embeddings, LayerNorms after the attention and the feed-forward rather
//! than before -- producing embeddings rather than logits.
//!
//! Its graph is shared: Jina-BERT v2 and v3 and Nomic-BERT (dense and MoE)
//! build with it, and it switches on the architecture for ALiBi, RoPE, the
//! MoE layers and the feed-forward's shape.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/bert.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_bert`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, and its `graph`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_bert`'s `load_arch_hparams` (models/bert.cpp:3 @b1b770d90).
pub fn loadArchHparams(self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const hp = &self.hparams;
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    self.type = switch (hp.nLayer()) {
        3 => .@"17m", // bge-micro
        6 => .@"22m", // MiniLM-L6
        12 => switch (hp.n_embd) {
            384 => .@"33m", // MiniLM-L12, bge-small
            768 => .@"109m", // bge-base
            else => .unknown,
        },
        24 => .@"335m", // bge-large
        else => .unknown,
    };
}

/// Ports `llama_model_bert`'s `load_arch_tensors` (models/bert.cpp:23 @b1b770d90).
pub fn loadArchTensors(self: *llama.model.Model, _: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    return loadBertTensors(self, .{ .pos_cls = true, .moe = false, .gate = false }, exc);
}

/// What tells the BERT family's `load_arch_tensors` apart: BERT's learned
/// positions and classifier heads, Nomic-BERT MoE's expert layers, and
/// Nomic-BERT's SwiGLU gate.
pub const BertTensors = struct { pos_cls: bool, moe: bool, gate: bool };

/// The body BERT, Jina-BERT v3, Nomic-BERT and Nomic-BERT MoE share in
/// their `load_arch_tensors`, each citing its own; `v` picks the lines that
/// differ.
pub fn loadBertTensors(self: *llama.model.Model, comptime v: BertTensors, exc: *llama.impl.Exception) llama.impl.Thrown!void {
    const l = self.loadLocals();
    const hp = &self.hparams;
    const nr = llama.loader.tensor_not_required;

    if (l.n_token_types == 0) {
        return exc.throw(.runtime_error, "{s} model needs to define token type count", .{self.archName()});
    }
    self.tok_embd = try self.createTensor(.{ .tensor = .token_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_vocab }, 0, exc);
    self.type_embd = try self.createTensor(.{ .tensor = .token_types, .suffix = "weight" }, &.{ l.n_embd, l.n_token_types }, nr, exc);

    if (v.pos_cls) {
        self.pos_embd = try self.createTensor(.{ .tensor = .pos_embd, .suffix = "weight" }, &.{ l.n_embd, l.n_ctx_train }, 0, exc);

        self.cls = try self.createTensor(.{ .tensor = .cls, .suffix = "weight" }, &.{ l.n_embd, l.n_embd }, nr, exc);
        self.cls_b = try self.createTensor(.{ .tensor = .cls, .suffix = "bias" }, &.{l.n_embd}, nr, exc);

        self.cls_out = try self.createTensor(.{ .tensor = .cls_out, .suffix = "weight" }, &.{ l.n_embd, hp.n_cls_out }, nr, exc);
        self.cls_out_b = try self.createTensor(.{ .tensor = .cls_out, .suffix = "bias" }, &.{hp.n_cls_out}, nr, exc);
    }

    self.tok_norm = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "weight", .bid = 0 }, &.{l.n_embd}, 0, exc);
    self.tok_norm_b = try self.createTensor(.{ .tensor = .token_embd_norm, .suffix = "bias", .bid = 0 }, &.{l.n_embd}, 0, exc);

    var i: i32 = 0;
    while (i < l.n_layer) : (i += 1) {
        const layer = &self.layers[@intCast(i)];

        try self.createTensorQkv(layer, i, l.n_embd, l.n_embd, l.n_embd_gqa, l.n_embd_gqa, 0, exc);

        layer.wo = try self.createTensor(.{ .tensor = .attn_out, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_embd }, 0, exc);
        layer.wo_b = try self.createTensor(.{ .tensor = .attn_out, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);

        layer.attn_out_norm = try self.createTensor(.{ .tensor = .attn_out_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.attn_out_norm_b = try self.createTensor(.{ .tensor = .attn_out_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);

        if (v.moe and hp.moe_every_n_layers > 0 and @as(u32, @intCast(i)) % hp.moe_every_n_layers == 1) {
            layer.ffn_up_exps = try self.createTensor(.{ .tensor = .ffn_up_exps, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff, l.n_expert }, 0, exc);
            layer.ffn_down_exps = try self.createTensor(.{ .tensor = .ffn_down_exps, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd, l.n_expert }, 0, exc);
            layer.ffn_gate_inp = try self.createTensor(.{ .tensor = .ffn_gate_inp, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_expert }, 0, exc);
        } else {
            layer.ffn_up = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
            layer.ffn_up_b = try self.createTensor(.{ .tensor = .ffn_up, .suffix = "bias", .bid = i }, &.{l.n_ff}, nr, exc);
            layer.ffn_down = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "weight", .bid = i }, &.{ l.n_ff, l.n_embd }, 0, exc);
            layer.ffn_down_b = try self.createTensor(.{ .tensor = .ffn_down, .suffix = "bias", .bid = i }, &.{l.n_embd}, nr, exc);
        }

        if (v.gate) {
            layer.ffn_gate = try self.createTensor(.{ .tensor = .ffn_gate, .suffix = "weight", .bid = i }, &.{ l.n_embd, l.n_ff }, 0, exc);
        }

        layer.layer_out_norm = try self.createTensor(.{ .tensor = .layer_out_norm, .suffix = "weight", .bid = i }, &.{l.n_embd}, 0, exc);
        layer.layer_out_norm_b = try self.createTensor(.{ .tensor = .layer_out_norm, .suffix = "bias", .bid = i }, &.{l.n_embd}, 0, exc);
    }
}

/// Ports `llama_model_bert`'s `build_arch_graph` (models/bert.cpp:64 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_bert`'s `graph`
/// (models/bert.cpp:68 @b1b770d90), which Jina-BERT v2 and v3 and
/// Nomic-BERT build with too.
///
/// Parameters:
/// - `model`: the model; its architecture picks the variant.
/// - `b`: the graph being built.
pub fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/bert.cpp", 71, "n_embd_head == hparams.n_embd_head_k()");

    var inp_pos: ?*llama.impl.llama_c.ggml_tensor = null;
    var embd_pos: ?*llama.impl.llama_c.ggml_tensor = null;

    if (model.arch != .jina_bert_v2) {
        inp_pos = b.buildInpPos();
    }

    // The position rows are ordered ahead of the input embeddings, without
    // marking them for compute.
    if (model.arch == .bert) {
        embd_pos = llama.impl.llama_c.ggml_get_rows(ctx, model.pos_embd, inp_pos);
        llama.impl.llama_c.ggml_build_forward_order(b.gf, embd_pos);
    }

    // The input embeddings: token, type and position.
    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The token types are hardcoded to zero ("Sentence A").
    if (model.type_embd) |type_embd| {
        const type_row0: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_view_1d(ctx, type_embd, b.n_embd, 0);
        inpL = llama.impl.llama_c.ggml_add(ctx, inpL, type_row0);
    }
    if (model.arch == .bert) {
        inpL = llama.impl.llama_c.ggml_add(ctx, embd_pos, inpL);
    }
    b.cb(inpL, "inp_embd", -1);

    // The embeddings' layer norm.
    inpL = b.buildNorm(inpL, model.tok_norm, model.tok_norm_b, .norm, 0);
    b.cb(inpL, "inp_norm", 0);

    const inp_attn = b.buildAttnInpNoCache();

    const inp_out_ids = b.buildInpOutIds();

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var cur = inpL;

        {
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            if (layer.attn_q_norm != null) {
                Qcur = llama.impl.llama_c.ggml_reshape_2d(ctx, Qcur, n_embd_head * b.n_head, b.n_tokens);

                Qcur = b.buildNorm(Qcur, layer.attn_q_norm, layer.attn_q_norm_b, .norm, il);

                Qcur = llama.impl.llama_c.ggml_reshape_3d(ctx, Qcur, n_embd_head, b.n_head, b.n_tokens);
            }

            if (layer.attn_k_norm != null) {
                Kcur = llama.impl.llama_c.ggml_reshape_2d(ctx, Kcur, n_embd_head * b.n_head_kv, b.n_tokens);

                Kcur = b.buildNorm(Kcur, layer.attn_k_norm, layer.attn_k_norm_b, .norm, il);

                Kcur = llama.impl.llama_c.ggml_reshape_3d(ctx, Kcur, n_embd_head, b.n_head_kv, b.n_tokens);
            }

            // RoPE.
            if (model.arch == .nomic_bert or model.arch == .nomic_bert_moe or model.arch == .jina_bert_v3) {
                Qcur = b.buildRope(Qcur, inp_pos, il);

                Kcur = b.buildRope(Kcur, inp_pos, il);
            }

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
            b.cb(cur, "kqv_out", il);
        }

        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpL = llama.impl.llama_c.ggml_get_rows(ctx, inpL, inp_out_ids);
        }

        // Re-add the layer input.
        cur = llama.impl.llama_c.ggml_add(ctx, cur, inpL);

        // The attention's layer norm.
        cur = b.buildNorm(cur, layer.attn_out_norm, layer.attn_out_norm_b, .norm, il);

        if (layer.attn_norm_2 != null) {
            cur = llama.impl.llama_c.ggml_add(ctx, cur, inpL); // Re-add the layer input.
            cur = b.buildNorm(cur, layer.attn_norm_2, layer.attn_norm_2_b, .norm, il);
        }

        const ffn_inp = cur;
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        if (hp.moe_every_n_layers > 0 and @as(u32, @intCast(il)) % hp.moe_every_n_layers == 1) {
            // The mixture of experts.
            cur = b.buildMoeFfn(cur, .{
                .gate_inp = layer.ffn_gate_inp,
                .up_exps = layer.ffn_up_exps,
                .gate_exps = null,
                .down_exps = layer.ffn_down_exps.?,
                .n_expert = hp.n_expert,
                .n_expert_used = hp.nExpertUsed(0),
                .type_op = .gelu,
                .norm_w = false,
                .w_scale = hp.expert_weights_scale,
                .gating_op = .softmax,
            }, il);
            b.cb(cur, "ffn_moe_out", il);
        } else if (model.arch == .bert or model.arch == .nomic_bert_moe or model.arch == .jina_bert_v3) {
            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, null, null, null, layer.ffn_down, layer.ffn_down_b, null, null, .gelu, .seq, il);
            b.cb(cur, "ffn_out", il);
        } else if (model.arch == .jina_bert_v2) {
            const up_contains_gate = layer.ffn_gate == null and layer.ffn_up.?.ne[1] != hp.nFf(0);
            const type_op: llama.graph.FfnOp = if (up_contains_gate) .geglu else .gelu;
            cur = b.buildFfn(cur, layer.ffn_up, layer.ffn_up_b, null, layer.ffn_gate, null, null, layer.ffn_down, layer.ffn_down_b, null, null, type_op, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        }

        // The attention output bypasses the feed-forward.
        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);

        // The output's layer norm.
        cur = b.buildNorm(cur, layer.layer_out_norm, layer.layer_out_norm_b, .norm, il);

        // The next layer's input.
        inpL = cur;
    }

    const cur = inpL;

    b.cb(cur, "result_embd", -1);
    b.res.t_embd = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
