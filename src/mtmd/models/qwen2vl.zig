//! The Qwen2-VL and Qwen2.5-VL vision encoders: a temporally merged patch
//! embedding, an M-RoPE ViT with optional window attention, and the merger.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/qwen2vl.cpp` at v0.8.0 (`8642541b2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

/// The "second conv dimension" / "spatial merge" reshape shared by the
/// qwen-vl encoders (the block at qwen2vl.cpp line 51): 2x2 patch groups become
/// consecutive rows.
pub fn spatialMerge(g: *graph.Graph, inp_in: graph.T, batch_size: c_int) graph.T {
    const ctx0 = g.ctx0;
    var inp = c.ggml_permute(ctx0, inp_in, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
    inp = c.ggml_cont_4d(ctx0, inp, g.n_embd * 2, @divTrunc(g.n_patches_x, 2), g.n_patches_y, batch_size);
    inp = c.ggml_reshape_4d(ctx0, inp, g.n_embd * 2, @divTrunc(g.n_patches_x, 2), 2, batch_size * @divTrunc(g.n_patches_y, 2));
    inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
    inp = c.ggml_cont_3d(ctx0, inp, g.n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    return inp;
}

/// Ports `build` (mtmd/models/qwen2vl.cpp:3 @8642541b2), a member of `clip_graph_qwen2vl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const d_head = g.d_head;
    const n_head = g.n_head;
    impl.ggmlAssert(model.patch_bias == null, "qwen2vl.cpp", 4, "model.patch_bias == nullptr");
    impl.ggmlAssert(model.class_embedding == null, "qwen2vl.cpp", 5, "model.class_embedding == nullptr");

    const batch_size = 1;
    const use_window_attn = hparams.n_wa_pattern > 0;
    const n_wa_pattern = hparams.n_wa_pattern;
    const n_pos = g.n_patches;
    const num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    const norm_t: model_mod.NormType = if (g.proj_type == .qwen25vl)
        .rms // qwen 2.5 vl
    else
        .normal; // qwen 2 vl

    var mrope_sections = [4]c_int{ @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4) };

    var inp = g.buildInpTemporalMerge();

    // second conv dimension
    inp = spatialMerge(g, inp, batch_size);

    var inpL = inp;
    var window_mask: graph.T = null;
    var window_idx: graph.T = null;
    var inv_window_idx: graph.T = null;

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, num_position_ids);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w != null) inpL = g.buildNorm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);

    if (use_window_attn) {
        // handle window attention inputs
        inv_window_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, @divTrunc(n_pos, 4));
        _ = c.ggml_set_name(inv_window_idx, "inv_window_idx");
        c.ggml_set_input(inv_window_idx);
        // mask for window attention
        window_mask = g.buildInpAttnMask(n_pos, n_pos, 0);

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        inpL = c.ggml_reshape_2d(ctx0, inpL, g.n_embd * 4, @divTrunc(g.n_patches_x * g.n_patches_y * batch_size, 4));
        inpL = c.ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = c.ggml_reshape_3d(ctx0, inpL, g.n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    }

    // loop over layers
    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        const full_attn = if (use_window_attn) @rem(il + 1, n_wa_pattern) == 0 else true;

        var cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        g.cb(cur, "ln1", il);

        // self-attention
        {
            var Qcur = c.ggml_add(ctx0, g.buildMm(layer.q_w, cur), layer.q_b);
            var Kcur = c.ggml_add(ctx0, g.buildMm(layer.k_w, cur), layer.k_b);
            var Vcur = c.ggml_add(ctx0, g.buildMm(layer.v_w, cur), layer.v_b);

            Qcur = c.ggml_reshape_3d(ctx0, Qcur, d_head, n_head, g.n_patches);
            Kcur = c.ggml_reshape_3d(ctx0, Kcur, d_head, n_head, g.n_patches);
            Vcur = c.ggml_reshape_3d(ctx0, Vcur, d_head, n_head, g.n_patches);

            g.cb(Qcur, "Qcur", il);
            g.cb(Kcur, "Kcur", il);
            g.cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = c.ggml_rope_multi(ctx0, Qcur, positions, null, @divTrunc(d_head, 2), &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = c.ggml_rope_multi(ctx0, Kcur, positions, null, @divTrunc(d_head, 2), &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            g.cb(Qcur, "Qcur_rope", il);
            g.cb(Kcur, "Kcur_rope", il);

            const attn_mask = if (full_attn) null else window_mask;

            cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, g.kq_scale, il, null);
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

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w != null) inpL = g.buildNorm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, g.n_layer);

    // multimodal projection
    var embeddings = inpL;
    embeddings = c.ggml_reshape_3d(ctx0, embeddings, g.n_embd * 4, @divTrunc(n_pos, 4), batch_size);
    embeddings = g.buildFfn(embeddings, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, .gelu, -1);

    if (use_window_attn) {
        window_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, @divTrunc(n_pos, 4));
        _ = c.ggml_set_name(window_idx, "window_idx");
        c.ggml_set_input(window_idx);

        // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        embeddings = c.ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, @divTrunc(g.n_patches_x * g.n_patches_y, 4));
        embeddings = c.ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = c.ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, @divTrunc(g.n_patches_x * g.n_patches_y, 4), batch_size);
    }

    // build the graph
    c.ggml_build_forward_expand(g.gf, embeddings);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
