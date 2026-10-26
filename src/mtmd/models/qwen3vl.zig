//! The Qwen3-VL vision encoder: Qwen2-VL's temporally merged input with
//! interpolated learned position embeddings, a fused-QKV M-RoPE ViT and
//! deepstack features.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/qwen3vl.cpp` at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const qwen2vl = @import("qwen2vl.zig");
const c = impl.c;

/// Ports `build` (mtmd/models/qwen3vl.cpp:3 @c1d0e7a00), a member of `clip_graph_qwen3vl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const d_head = g.d_head;
    const n_head = g.n_head;
    const n_embd = g.n_embd;
    impl.ggmlAssert(model.patch_bias != null, "qwen3vl.cpp", 4, "model.patch_bias != nullptr");
    impl.ggmlAssert(model.position_embeddings != null, "qwen3vl.cpp", 5, "model.position_embeddings != nullptr");
    impl.ggmlAssert(model.class_embedding == null, "qwen3vl.cpp", 6, "model.class_embedding == nullptr");

    const batch_size = 1;
    const n_pos = g.n_patches;
    const num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    const norm_t: model_mod.NormType = .normal;

    var mrope_sections = [4]c_int{ @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4) };

    var inp = qwen2vl.buildInpWithTemporalMerge(g);

    // spatial merge
    inp = qwen2vl.spatialMerge(g, inp, batch_size);

    // add patch bias
    if (model.patch_bias != null) {
        inp = c.ggml_add(ctx0, inp, model.patch_bias);
        g.cb(inp, "patch_bias", -1);
    }

    // calculate absolute position embedding and apply
    var learned_pos_embd = g.resizePositionEmbeddings(c.GGML_SCALE_MODE_BILINEAR | c.GGML_SCALE_FLAG_ALIGN_CORNERS);
    learned_pos_embd = c.ggml_cont_4d(ctx0, learned_pos_embd, n_embd * 2, @divTrunc(g.n_patches_x, 2), g.n_patches_y, batch_size);
    learned_pos_embd = c.ggml_reshape_4d(ctx0, learned_pos_embd, n_embd * 2, @divTrunc(g.n_patches_x, 2), 2, batch_size * @divTrunc(g.n_patches_y, 2));
    learned_pos_embd = c.ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = c.ggml_cont_3d(ctx0, learned_pos_embd, n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    inp = c.ggml_add(ctx0, inp, learned_pos_embd);
    g.cb(inp, "inp_pos_emb", -1);

    var inpL = inp;

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, num_position_ids);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w != null) inpL = g.buildNorm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);

    // deepstack features (stack along the feature dimension)
    var deepstack_features: graph.T = null;
    const merge_factor: c_int = if (hparams.n_merge > 0) hparams.n_merge * hparams.n_merge else 4; // default 2x2=4 for qwen3vl

    // loop over layers
    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];

        var cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        g.cb(cur, "ln1", il);

        // self-attention
        {
            cur = g.buildMm(layer.qkv_w, cur);
            cur = c.ggml_add(ctx0, cur, layer.qkv_b);

            const rs = c.ggml_row_size(cur.?.type, d_head);
            var Qcur = c.ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, rs, graph.nb(cur, 1), 0);
            var Kcur = c.ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, rs, graph.nb(cur, 1), c.ggml_row_size(cur.?.type, n_embd));
            const Vcur = c.ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, rs, graph.nb(cur, 1), c.ggml_row_size(cur.?.type, 2 * n_embd));

            g.cb(Qcur, "Qcur", il);
            g.cb(Kcur, "Kcur", il);
            g.cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = c.ggml_rope_multi(ctx0, Qcur, positions, null, @divTrunc(d_head, 2), &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = c.ggml_rope_multi(ctx0, Kcur, positions, null, @divTrunc(d_head, 2), &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            g.cb(Qcur, "Qcur_rope", il);
            g.cb(Kcur, "Kcur_rope", il);

            cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, null, g.kq_scale, il, null);
            g.cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = c.ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        g.cb(cur, "ffn_inp", il);

        // layernorm2
        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        g.cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);

        g.cb(cur, "ffn_out", il);

        // residual 2
        cur = c.ggml_add(ctx0, inpL, cur);
        g.cb(cur, "layer_out", il);

        if (layer.hasDeepstack()) {
            var feat = c.ggml_reshape_3d(ctx0, cur, n_embd * merge_factor, @divTrunc(n_pos, merge_factor), batch_size);
            feat = g.buildNorm(feat, layer.deepstack_norm_w, layer.deepstack_norm_b, norm_t, eps, il);
            feat = g.buildFfn(feat, layer.deepstack_fc1_w, layer.deepstack_fc1_b, null, null, layer.deepstack_fc2_w, layer.deepstack_fc2_b, .gelu, il);

            if (deepstack_features == null) {
                deepstack_features = feat;
            } else {
                // concat along the feature dimension
                deepstack_features = c.ggml_concat(ctx0, deepstack_features, feat, 0);
            }
        }

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w != null) inpL = g.buildNorm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, g.n_layer);

    // multimodal projection
    var embeddings = inpL;
    embeddings = c.ggml_reshape_3d(ctx0, embeddings, n_embd * 4, @divTrunc(n_pos, 4), batch_size);

    embeddings = g.buildFfn(embeddings, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, .gelu, -1);

    // concat along the feature dimension
    if (deepstack_features != null) embeddings = c.ggml_concat(ctx0, embeddings, deepstack_features, 0);

    // build the graph
    c.ggml_build_forward_expand(g.gf, embeddings);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
