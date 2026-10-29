//! The Youtu-VL encoder graph: a SigLIP2 ViT with 2-D M-RoPE and windowed
//! attention on all but the listed full-attention layers, then a 2x2
//! patch merger.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/youtuvl.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;
const T = graph.T;

/// Ports `build` (youtuvl.cpp:3 @b29c606e2), a member of `clip_graph_youtuvl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const n_embd = g.n_embd;
    const n_patches = g.n_patches;
    const patch_size = g.patch_size;
    const d_head = g.d_head;

    impl.ggmlAssert(model.class_embedding == null, "youtuvl.cpp", 4, "model.class_embedding == nullptr");
    const batch_size: c_int = 1;
    const use_window_attn = hparams.wa_layer_indexes.count() != 0;
    const n_pos: c_int = n_patches;
    const num_position_ids: c_int = n_pos * 4;
    const m: c_int = 2;
    const Wp: c_int = g.n_patches_x;
    const Hp: c_int = g.n_patches_y;
    const Hm: c_int = @divTrunc(Hp, m);
    const Wm: c_int = @divTrunc(Wp, m);
    const norm_t: model_mod.NormType = .normal;

    var mrope_sections = [4]c_int{ @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4) };

    var inp = g.buildInpRaw(3);

    // change conv3d to linear
    // reshape and permute to get patches, permute from (patch_size, m, Wm, patch_size, m, Hm, C) to (C, patch_size, patch_size, m, m, Wm, Hm)
    {
        inp = c.ggml_reshape_4d(ctx0, inp, Wm * m * patch_size, m * patch_size, Hm, 3);
        inp = c.ggml_permute(ctx0, inp, 1, 2, 3, 0);
        inp = c.ggml_cont_4d(ctx0, inp, m * patch_size * 3, Wm, m * patch_size, Hm);

        inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = c.ggml_cont_4d(ctx0, inp, m * patch_size * 3, patch_size, m, Hm * Wm);

        inp = c.ggml_permute(ctx0, inp, 1, 0, 2, 3);
        inp = c.ggml_cont_4d(ctx0, inp, patch_size, 3, patch_size, Hm * Wm * m * m);

        inp = c.ggml_permute(ctx0, inp, 2, 0, 1, 3);
        inp = c.ggml_cont_3d(ctx0, inp, 3 * patch_size * patch_size, Hm * Wm * m * m, 1);
    }
    inp = g.buildMm(model.patch_embeddings_0, inp);

    if (model.patch_bias != null) inp = c.ggml_add(ctx0, inp, model.patch_bias);

    inp = c.ggml_reshape_2d(ctx0, inp, n_embd, n_patches);

    var inpL = inp;
    var window_mask: T = null;
    var window_idx: T = null;
    var inv_window_idx: T = null;

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, num_position_ids);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w != null) inpL = g.buildNorm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    if (use_window_attn) {
        inv_window_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, @divTrunc(n_pos, 4));
        _ = c.ggml_set_name(inv_window_idx, "inv_window_idx");
        c.ggml_set_input(inv_window_idx);
        // mask for window attention
        window_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_pos, n_pos);
        _ = c.ggml_set_name(window_mask, "window_mask");
        c.ggml_set_input(window_mask);

        // if flash attn is used, we need to pad the mask and cast to f16
        if (g.flash_attn_type == .enabled) window_mask = c.ggml_cast(ctx0, window_mask, c.GGML_TYPE_F16);

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        impl.ggmlAssert(batch_size == 1, "youtuvl.cpp", 82, "batch_size == 1");
        inpL = c.ggml_reshape_2d(ctx0, inpL, n_embd * 4, @divTrunc(g.n_patches_x * g.n_patches_y * batch_size, 4));
        inpL = c.ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = c.ggml_reshape_3d(ctx0, inpL, n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    }

    // loop over layers
    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        const full_attn = if (use_window_attn) hparams.wa_layer_indexes.contains(il) else true;

        var cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        // self-attention
        {
            var Qcur = c.ggml_add(ctx0, g.buildMm(layer.q_w, cur), layer.q_b);
            var Kcur = c.ggml_add(ctx0, g.buildMm(layer.k_w, cur), layer.k_b);
            var Vcur = c.ggml_add(ctx0, g.buildMm(layer.v_w, cur), layer.v_b);

            Qcur = c.ggml_reshape_3d(ctx0, Qcur, d_head, g.n_head, n_patches);
            Kcur = c.ggml_reshape_3d(ctx0, Kcur, d_head, g.n_head, n_patches);
            Vcur = c.ggml_reshape_3d(ctx0, Vcur, d_head, g.n_head, n_patches);

            Qcur = c.ggml_rope_multi(ctx0, Qcur, positions, null, @divTrunc(d_head, 2), &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = c.ggml_rope_multi(ctx0, Kcur, positions, null, @divTrunc(d_head, 2), &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            const attn_mask: T = if (full_attn) null else window_mask;

            cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, g.kq_scale, il, null);
        }
        // re-add the layer input, e.g., residual
        cur = c.ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        // layernorm2
        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);

        // ffn
        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, null, null, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);

        // residual 2
        cur = c.ggml_add(ctx0, inpL, cur);

        inpL = cur;
    }

    var embeddings = inpL;
    if (use_window_attn) {
        const spatial_merge_unit: c_int = 4;
        window_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, @divTrunc(n_pos, spatial_merge_unit));
        _ = c.ggml_set_name(window_idx, "window_idx");
        c.ggml_set_input(window_idx);
        impl.ggmlAssert(batch_size == 1, "youtuvl.cpp", 149, "batch_size == 1");
        embeddings = c.ggml_reshape_2d(ctx0, embeddings, n_embd * spatial_merge_unit, @divTrunc(n_patches, spatial_merge_unit));
        embeddings = c.ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = c.ggml_reshape_3d(ctx0, embeddings, n_embd, n_patches, batch_size);
        g.cb(embeddings, "window_order_restored", -1);
    }

    // post-layernorm (part of Siglip2VisionTransformer, applied after encoder)
    if (model.post_ln_w != null) embeddings = g.buildNorm(embeddings, model.post_ln_w, model.post_ln_b, norm_t, eps, g.n_layer);

    // Now apply merger (VLPatchMerger):
    // 1. Apply RMS norm (ln_q in VLPatchMerger)
    embeddings = g.buildNorm(embeddings, model.mm_input_norm_w, null, .rms, 1e-6, -1);
    g.cb(embeddings, "merger_normed", -1);

    // 2. First reshape for spatial merge (merge 2x2 patches)
    embeddings = c.ggml_reshape_3d(ctx0, embeddings, n_embd * 4, @divTrunc(n_pos, 4), batch_size);
    g.cb(embeddings, "merger_reshaped", -1);

    embeddings = g.buildFfn(embeddings, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, .gelu, -1);
    c.ggml_build_forward_expand(g.gf, embeddings);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
