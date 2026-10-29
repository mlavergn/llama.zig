//! Nemotron-H MoE: Nemotron-H with mixture-of-experts feed-forward layers,
//! and an optional multi-token-prediction draft head that folds a NoPE
//! attention sub-layer and a MoE sub-layer into one trailing block.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/nemotron-h-moe.cpp` at v0.4.1
//! (`b29c606e2`): `llama_model_nemotron_h_moe`'s `build_arch_graph` and its
//! `graph_mtp`. It reuses `llama_model_nemotron_h`'s loading and `graph`,
//! in `nemotron_h.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const nemotron_h = @import("nemotron_h.zig");

const c = impl.c;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Ports `llama_model_nemotron_h_moe`'s `build_arch_graph`
/// (models/nemotron-h-moe.cpp:3 @b29c606e2): the MTP draft head for that
/// graph type, Nemotron-H's graph otherwise.
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, params: *const graph.Params) void {
    if (params.gtype == .decoder_mtp) return buildGraphMtp(model, b);
    return nemotron_h.buildGraph(model, b);
}

/// Ports the constructor of `llama_model_nemotron_h_moe`'s `graph_mtp`
/// (models/nemotron-h-moe.cpp:11 @b29c606e2): the MTP draft head.
fn buildGraphMtp(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const cp = b.cparams;
    impl.assert(hp.n_layer_nextn == 1, "models/nemotron-h-moe.cpp", 13, "hparams.n_layer_nextn == 1 && \"NEMOTRON_H_MOE MTP currently supports a single MTP block\"");

    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/nemotron-h-moe.cpp", 16, "n_embd_head == hparams.n_embd_head_k()");

    const il: c_int = @intCast(hp.nLayer());
    const layer = &model.layers[@intCast(il)];

    impl.assert(layer.nextn.eh_proj != null and layer.nextn.enorm != null and layer.nextn.hnorm != null, "models/nemotron-h-moe.cpp", 21, "layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm");
    impl.assert(layer.ffn_gate_inp != null, "models/nemotron-h-moe.cpp", 22, "layer.ffn_gate_inp");

    // The token embedding weights.
    const tok_embd_w = layer.nextn.embed_tokens orelse model.tok_embd;
    impl.assert(tok_embd_w != null, "models/nemotron-h-moe.cpp", 26, "tok_embd_w != nullptr && \"NEMOTRON_H_MOE MTP requires token embeddings\"");

    const inp = impl.allocator.create(graph.InpEmbdH) catch impl.abort("models/nemotron-h-moe.cpp", 28, "out of memory");
    inp.* = .{ .n_embd = hp.n_embd };

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, b.n_tokens);
    c.ggml_set_input(inp.tokens);

    inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.nEmbdInp(), b.n_tokens);
    c.ggml_set_input(inp.embd);

    var tok_embd: *Tensor = undefined;
    if (b.ubatch.token != null) {
        tok_embd = c.ggml_get_rows(ctx, tok_embd_w, inp.tokens);
    } else {
        tok_embd = inp.embd.?;
    }
    b.cb(tok_embd, "mtp_tok_embd", il);

    inp.h = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, hp.n_embd, b.n_tokens);
    c.ggml_set_input(inp.h);
    _ = c.ggml_set_name(inp.h, "mtp_h_input");

    const h_embd = inp.h.?;

    _ = b.res.addInput(graph.InpEmbdH, inp);

    const inp_out_ids = b.buildInpOutIds();

    // Attention fills the KV cache over every token, but the MoE is
    // position-wise: the output rows are gathered before it to save FFN
    // compute, unless unmasked embeddings_nextn needs the full-length
    // hidden state.
    const emit_h_nextn = cp.embeddings_nextn;
    const crop_before_ffn = !emit_h_nextn or cp.embeddings_nextn_masked;

    const inp_attn = b.buildAttnInpKv();

    const h_norm = b.buildNorm(h_embd, layer.nextn.hnorm, null, .rms, il);
    b.cb(h_norm, "mtp_hnorm", il);
    const e_norm = b.buildNorm(tok_embd, layer.nextn.enorm, null, .rms, il);
    b.cb(e_norm, "mtp_enorm", il);

    const concat: *Tensor = c.ggml_concat(ctx, e_norm, h_norm, 0);
    b.cb(concat, "mtp_concat", il);

    var cur = b.buildLoraMm(layer.nextn.eh_proj.?, concat, layer.nextn.eh_proj_s);
    b.cb(cur, "mtp_eh_proj", il);

    // The dense NoPE attention sub-layer (mtp.layers.0).
    const inpSA = cur;
    cur = b.buildNorm(cur, layer.attn_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_norm", il);

    {
        const qkv = b.buildQkv(layer, cur, n_embd_head, hp.nHead(@intCast(il)), hp.nHeadKv(@intCast(il)), il);
        const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;
        cur = b.buildAttn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, qkv.q, qkv.k, qkv.v, null, null, null, kq_scale, il);
        b.cb(cur, "mtp_attn_out", il);
    }

    cur = c.ggml_add(ctx, cur, inpSA);
    b.cb(cur, "mtp_attn_residual", il);

    // The output rows are gathered here so the MoE FFN below runs only on
    // the positions kept.
    if (crop_before_ffn) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    // The MoE FFN sub-layer (mtp.layers.1).
    const ffn_residual = cur;
    cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
    b.cb(cur, "mtp_attn_post_norm", il);

    {
        const router_logits = b.buildLoraMm(layer.ffn_gate_inp.?, cur, null);
        b.cb(router_logits, "mtp_ffn_moe_logits", il);

        const moe_out = b.buildMoeFfn(cur, .{
            .gate_inp = layer.ffn_gate_inp,
            .up_exps = layer.ffn_up_exps,
            .gate_exps = null, // no gate
            .down_exps = layer.ffn_down_exps.?,
            .exp_probs_b = layer.ffn_exp_probs_b,
            .n_expert = b.n_expert,
            .n_expert_used = b.n_expert_used,
            .type_op = .relu_sqr,
            .norm_w = hp.expert_weights_norm,
            .w_scale = hp.expert_weights_scale,
            .gating_op = .sigmoid,
            .probs_in = router_logits,
            .gate_up_exps = null,
            .up_exps_s = layer.ffn_up_exps_s,
            .gate_exps_s = null, // no gate
            .down_exps_s = layer.ffn_down_exps_s,
        }, il);
        b.cb(moe_out, "mtp_ffn_moe_out", il);

        const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, layer.ffn_up_shexp_s, null, null, null, layer.ffn_down_shexp, null, layer.ffn_down_shexp_s, null, .relu_sqr, .par, il);
        b.cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = c.ggml_add(ctx, moe_out, ffn_shexp);
        b.cb(cur, "mtp_ffn_out", il);
    }

    cur = c.ggml_add(ctx, cur, ffn_residual);
    b.cb(cur, "mtp_post_ffn", il);

    // The final head norm: the MTP head has its own LayerNorm.
    impl.assert(layer.nextn.shared_head_norm != null, "models/nemotron-h-moe.cpp", 137, "layer.nextn.shared_head_norm && \"NEMOTRON_H_MOE MTP: missing final head norm\"");
    cur = b.buildNorm(cur, layer.nextn.shared_head_norm, null, .norm, -1);

    b.cb(cur, "h_nextn", -1);
    b.res.t_h_nextn = cur;

    if (!crop_before_ffn) {
        cur = c.ggml_get_rows(ctx, cur, inp_out_ids);
    }

    // The LM head.
    const head_w = layer.nextn.shared_head_head orelse model.output;
    const head_s = if (layer.nextn.shared_head_head != null) layer.nextn.shared_head_head_s else model.output_s;
    impl.assert(head_w != null, "models/nemotron-h-moe.cpp", 150, "head_w != nullptr && \"NEMOTRON_H_MOE MTP requires an output projection\"");
    cur = b.buildLoraMm(head_w.?, cur, head_s);
    b.cb(cur, "result_output", -1);

    b.res.t_logits = cur;
    c.ggml_build_forward_expand(b.gf, cur);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
