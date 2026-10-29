//! The EXAONE 4.5 vision encoder: Qwen2-VL's layout with GQA attention.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/exaone4_5.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

/// Ports `build` (exaone4_5.cpp:4 @b29c606e2), a member of `clip_graph_exaone4_5`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const d_head = g.d_head;
    const n_head = g.n_head;
    const n_embd = g.n_embd;
    const patch_size = g.patch_size;
    impl.ggmlAssert(model.patch_bias == null, "exaone4_5.cpp", 5, "model.patch_bias == nullptr");
    impl.ggmlAssert(model.class_embedding == null, "exaone4_5.cpp", 6, "model.class_embedding == nullptr");

    const batch_size = 1;
    const use_window_attn = hparams.n_wa_pattern > 0;
    const n_wa_pattern = hparams.n_wa_pattern;
    const n_pos = g.n_patches;
    const num_position_ids = n_pos * 4;

    const norm_t: model_mod.NormType = .rms;

    const n_kv_head: i64 = if (hparams.n_head_kv > 0) hparams.n_head_kv else n_head;
    impl.ggmlAssert(@rem(n_head, n_kv_head) == 0, "exaone4_5.cpp", 17, "n_head % n_kv_head == 0");

    var rope_sections = [4]c_int{ @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4) };
    const rope_freq_base: f32 = if (hparams.rope_theta > 0.0) hparams.rope_theta else 10000.0;

    const inp_raw = g.buildInpRaw(3);
    var inp: graph.T = c.ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    impl.ggmlAssert(@rem(g.img.nx(), patch_size * 2) == 0, "exaone4_5.cpp", 25, "img.nx() % (patch_size * 2) == 0");
    impl.ggmlAssert(@rem(g.img.ny(), patch_size * 2) == 0, "exaone4_5.cpp", 26, "img.ny() % (patch_size * 2) == 0");

    {
        const inp_1 = c.ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = c.ggml_add(ctx0, inp, inp_1);
        inp = c.ggml_permute(ctx0, inp, 1, 2, 0, 3);
        inp = c.ggml_cont_4d(ctx0, inp, n_embd * 2, @divTrunc(g.n_patches_x, 2), g.n_patches_y, batch_size);
        inp = c.ggml_reshape_4d(ctx0, inp, n_embd * 2, @divTrunc(g.n_patches_x, 2), 2, batch_size * @divTrunc(g.n_patches_y, 2));
        inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = c.ggml_cont_3d(ctx0, inp, n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    }

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, num_position_ids);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);

    var window_mask: graph.T = null;
    var window_idx: graph.T = null;
    var inv_window_idx: graph.T = null;

    if (use_window_attn) {
        window_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, @divTrunc(n_pos, 4));
        _ = c.ggml_set_name(window_idx, "window_idx");
        c.ggml_set_input(window_idx);

        inv_window_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, @divTrunc(n_pos, 4));
        _ = c.ggml_set_name(inv_window_idx, "inv_window_idx");
        c.ggml_set_input(inv_window_idx);

        window_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_pos, n_pos);
        _ = c.ggml_set_name(window_mask, "window_mask");
        c.ggml_set_input(window_mask);

        if (g.flash_attn_type == .enabled) window_mask = c.ggml_cast(ctx0, window_mask, c.GGML_TYPE_F16);
    }

    var inpL = inp;

    if (use_window_attn) {
        inpL = c.ggml_reshape_2d(ctx0, inpL, n_embd * 4, @divTrunc(g.n_patches_x * g.n_patches_y * batch_size, 4));
        inpL = c.ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = c.ggml_reshape_3d(ctx0, inpL, n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    }

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        const full_attn = if (use_window_attn) @rem(il + 1, n_wa_pattern) == 0 else true;
        var cur = inpL;

        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        g.cb(cur, "ln1", il);

        {
            impl.ggmlAssert(layer.qkv_w != null, "exaone4_5.cpp", 88, "layer.qkv_w != nullptr");
            cur = g.buildMm(layer.qkv_w, cur);
            if (layer.qkv_b != null) cur = c.ggml_add(ctx0, cur, layer.qkv_b);

            const n_embd_kv: i64 = d_head * n_kv_head;
            const rs = c.ggml_row_size(cur.?.type, d_head);
            var Qcur = c.ggml_view_3d(ctx0, cur, d_head, n_head, g.n_patches, rs, graph.nb(cur, 1), 0);
            var Kcur = c.ggml_view_3d(ctx0, cur, d_head, n_kv_head, g.n_patches, rs, graph.nb(cur, 1), c.ggml_row_size(cur.?.type, n_embd));
            const Vcur = c.ggml_view_3d(ctx0, cur, d_head, n_kv_head, g.n_patches, rs, graph.nb(cur, 1), c.ggml_row_size(cur.?.type, n_embd + n_embd_kv));

            g.cb(Qcur, "Qcur", il);
            g.cb(Kcur, "Kcur", il);
            g.cb(Vcur, "Vcur", il);

            Qcur = c.ggml_rope_multi(ctx0, Qcur, positions, null, @divTrunc(d_head, 2), &rope_sections, c.GGML_ROPE_TYPE_VISION, 32768, rope_freq_base, 1, 0, 1, 32, 1);
            Kcur = c.ggml_rope_multi(ctx0, Kcur, positions, null, @divTrunc(d_head, 2), &rope_sections, c.GGML_ROPE_TYPE_VISION, 32768, rope_freq_base, 1, 0, 1, 32, 1);

            g.cb(Qcur, "Qcur_rope", il);
            g.cb(Kcur, "Kcur_rope", il);
            g.cb(Vcur, "Vcur", il);

            const attn_mask = if (full_attn) null else window_mask;
            cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, g.kq_scale, il, null);
            g.cb(cur, "attn_out", il);
        }

        cur = c.ggml_add(ctx0, cur, inpL);
        inpL = cur;

        g.cb(cur, "ffn_inp", il);

        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        g.cb(cur, "ffn_inp_normed", il);

        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);

        g.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx0, inpL, cur);
        g.cb(cur, "layer_out", il);

        inpL = cur;
    }

    var embeddings = inpL;
    embeddings = g.buildNorm(embeddings, model.post_ln_w, model.post_ln_b, norm_t, eps, g.n_layer);
    embeddings = c.ggml_reshape_3d(ctx0, embeddings, n_embd * 4, @divTrunc(n_pos, 4), batch_size);
    embeddings = g.buildFfn(embeddings, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, .gelu, -1);

    if (use_window_attn) {
        embeddings = c.ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, @divTrunc(g.n_patches_x * g.n_patches_y, 4));
        embeddings = c.ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = c.ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, @divTrunc(g.n_patches_x * g.n_patches_y, 4), batch_size);
    }

    c.ggml_build_forward_expand(g.gf, embeddings);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
