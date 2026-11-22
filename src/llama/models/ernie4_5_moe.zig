//! ERNIE 4.5 MoE: ERNIE 4.5's decoder, with a mixture of experts -- routed by
//! softmax, selected with a bias, plus an optional shared expert -- on every
//! `n_moe_layer_step`-th block past the leading dense ones.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/ernie4-5-moe.cpp` at v1.0.0
//! (`b1b770d90`): `llama_model_ernie4_5_moe`'s `build_arch_graph` and its
//! `graph`. Its loading is `llama_model_ernie4_5`'s, in `ernie4_5.zig`.

const std = @import("std");
const llama = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `llama_model_ernie4_5_moe`'s `build_arch_graph` (models/ernie4-5-moe.cpp:3 @b1b770d90).
pub fn buildArchGraph(model: *const llama.model.Model, b: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    buildGraph(model, b);
}

/// Ports the constructor of `llama_model_ernie4_5_moe`'s `graph`
/// (models/ernie4-5-moe.cpp:7 @b1b770d90).
fn buildGraph(model: *const llama.model.Model, b: *const llama.graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    llama.impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/ernie4-5-moe.cpp", 11, "n_embd_head == hparams.n_embd_head_k()");
    llama.impl.assert(n_embd_head == b.n_rot, "models/ernie4-5-moe.cpp", 12, "n_embd_head == n_rot");

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    // The positions.
    const inp_pos = b.buildInpPos();

    const inp_attn = b.buildAttnInpKv();

    const inp_out_ids = b.buildInpOutIds();

    llama.impl.assert(hp.n_moe_layer_step > 0, "models/ernie4-5-moe.cpp", 26, "hparams.n_moe_layer_step > 0 && \"Ernie 4.5 MoE requires n_moe_layer_step > 0\"");
    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        var inpSA = inpL;
        var cur: *llama.impl.llama_c.ggml_tensor = undefined;
        // Norm.
        {
            cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
            b.cb(cur, "attn_norm", il);
        }
        // Self-attention.
        {
            // Q, K and V, and RoPE on Q and K.
            const qkv = b.buildQkv(layer, cur, n_embd_head, b.n_head, b.n_head_kv, il);
            var Qcur = qkv.q;
            var Kcur = qkv.k;
            const Vcur = qkv.v;

            Qcur = b.buildRope(Qcur, inp_pos, il);
            Kcur = b.buildRope(Kcur, inp_pos, il);

            b.cb(Qcur, "Qcur", il);
            b.cb(Kcur, "Kcur", il);
            b.cb(Vcur, "Vcur", il);

            cur = b.buildAttn(inp_attn, layer.wo, null, layer.wo_s, Qcur, Kcur, Vcur, null, null, null, 1.0 / stdlib.libm.sqrtfInt(n_embd_head), il);
            b.cb(cur, "attn_out", il);
        }
        if (il == b.n_layer - 1) {
            cur = llama.impl.llama_c.ggml_get_rows(ctx, cur, inp_out_ids);
            inpSA = llama.impl.llama_c.ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const ffn_inp: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_add(ctx, cur, inpSA);
        b.cb(ffn_inp, "ffn_inp", il);

        // The feed-forward network.
        const is_moe_layer = @as(u32, @bitCast(il)) >= hp.n_layer_dense_lead and @as(u32, @bitCast(il + 1)) % hp.n_moe_layer_step == 0;

        if (!is_moe_layer) {
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            cur = b.buildFfn(cur, layer.ffn_up, null, null, layer.ffn_gate, null, null, layer.ffn_down, null, null, null, .silu, .par, il);
            b.cb(cur, "ffn_out", il);
        } else {
            // The mixture of experts.
            cur = b.buildNorm(ffn_inp, layer.ffn_norm, null, .rms, il);
            b.cb(cur, "ffn_norm", il);

            const moe_out = b.buildMoeFfn(cur, .{
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
                .gating_op = .softmax,
            }, il);
            b.cb(moe_out, "ffn_moe_out", il);

            // The shared expert, if present.
            if (hp.n_ff_shexp > 0) {
                const ffn_shexp = b.buildFfn(cur, layer.ffn_up_shexp, null, null, layer.ffn_gate_shexp, null, null, layer.ffn_down_shexp, null, null, null, .silu, .par, il);
                b.cb(ffn_shexp, "ffn_shexp", il);

                cur = llama.impl.llama_c.ggml_add(ctx, moe_out, ffn_shexp);
            } else {
                cur = moe_out;
            }
            b.cb(cur, "ffn_out", il);
        }
        cur = llama.impl.llama_c.ggml_add(ctx, cur, ffn_inp);
        b.cb(cur, "ffn_out", il);

        cur = b.buildCvec(cur, il);
        b.cb(cur, "l_out", il);

        // The next layer's input.
        inpL = cur;
    }
    var cur = inpL;

    cur = b.buildNorm(cur, model.output_norm, null, .rms, -1);

    b.cb(cur, "result_norm", -1);
    b.res.t_embd = cur;

    // The LM head.
    cur = b.buildLoraMm(model.output.?, cur, model.output_s);

    b.cb(cur, "result_output", -1);
    b.res.t_logits = cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
