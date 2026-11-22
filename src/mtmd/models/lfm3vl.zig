//! The LFM3-VL vision encoder: a temporally merged patchifier, a hybrid stack
//! of 2D conv and attention layers in raster-scan order, then a 2x2 merge
//! into an MLP projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/lfm3vl.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration names what it ports and the line it began at.

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `build` (lfm3vl.cpp:3 @b1b770d90), a member of `mtmd_graph_lfm3vl`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const d_head = g.d_head;
    const n_head = g.n_head;
    const n_head_kv = g.n_head_kv;
    const eps = g.eps;
    const patch_size = g.patch_size;
    const n_patches_x = g.n_patches_x;
    const n_patches_y = g.n_patches_y;

    mtmd.impl.ggmlAssert(model.patch_embeddings_0 != null, "lfm3vl.cpp", 4, "model.patch_embeddings_0 != nullptr");
    mtmd.impl.ggmlAssert(model.lfm3vl_embedding_norm_w != null, "lfm3vl.cpp", 5, "model.lfm3vl_embedding_norm_w != nullptr");
    mtmd.impl.ggmlAssert(model.class_embedding == null, "lfm3vl.cpp", 6, "model.class_embedding == nullptr");
    mtmd.impl.ggmlAssert(g.n_batch == 1 or g.n_batch == 2, "lfm3vl.cpp", 7, "n_batch == 1 || n_batch == 2");

    const batch_size: c_int = 1;
    const n_pos = g.n_patches;
    const num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    const norm_t: mtmd.model.NormType = .rms;

    // 2D rope: pairs (j, j+n_dims); first d_head/4 pairs rotate by y, next d_head/4 by x
    var mrope_sections = [4]c_int{ @divTrunc(d_head, 4), @divTrunc(d_head, 4), 0, 0 };

    // one patchifier kernel per temporal slice: a still image fills both slices, video frames are merged in pairs
    var inp: mtmd.graph.T = undefined;
    if (model.patch_embeddings_1 == null) {
        mtmd.impl.ggmlAssert(g.n_batch == 1, "lfm3vl.cpp", 21, "n_batch == 1");
        inp = mtmd.impl.c.ggml_conv_2d(ctx0, model.patch_embeddings_0, g.buildInpRaw(3), patch_size, patch_size, 0, 0, 1, 1);
    } else {
        inp = g.buildInpTemporalMerge();
    }
    inp = mtmd.impl.c.ggml_reshape_2d(ctx0, inp, n_pos, n_embd);
    inp = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, inp));
    g.cb(inp, "patchifier_out", -1);

    // the 2x2 interleave waits for the projector, the conv layers need raster-scan order
    inp = g.buildNorm(inp, model.lfm3vl_embedding_norm_w, null, norm_t, eps, -1);
    g.cb(inp, "embedding_norm", -1);

    var inp_l = inp;

    // position IDs for 2D RoPE (same format as Qwen3VL)
    const positions = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, num_position_ids);
    _ = mtmd.impl.c.ggml_set_name(positions, "positions");
    mtmd.impl.c.ggml_set_input(positions);

    const rope_theta = hparams.rope_theta;

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];

        const is_conv_layer = layer.conv2d_in_proj_w != null;

        var cur = inp_l;

        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        g.cb(cur, "ln1", il);

        if (is_conv_layer) {
            const proj = g.buildMm(layer.conv2d_in_proj_w, cur);
            g.cb(proj, "conv_in_proj", il);

            const first = mtmd.impl.c.ggml_view_2d(ctx0, proj, n_embd, n_pos, mtmd.graph.nb(proj, 1), 0);
            const middle = mtmd.impl.c.ggml_view_2d(ctx0, proj, n_embd, n_pos, mtmd.graph.nb(proj, 1), mtmd.impl.c.ggml_row_size(proj.?.type, n_embd));
            const third = mtmd.impl.c.ggml_view_2d(ctx0, proj, n_embd, n_pos, mtmd.graph.nb(proj, 1), mtmd.impl.c.ggml_row_size(proj.?.type, 2 * n_embd));

            const gated = mtmd.impl.c.ggml_mul(ctx0, first, third);
            g.cb(gated, "conv_gated", il);

            var conv_out: mtmd.graph.T = undefined;
            if (layer.conv2d_spatial_w != null) {
                const gated_3d = mtmd.impl.c.ggml_reshape_4d(ctx0, gated, n_embd, n_patches_x, n_patches_y, batch_size);
                const gated_2d = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, gated_3d, 2, 0, 1, 3));
                g.cb(gated_2d, "conv_gated_2d", il);

                conv_out = mtmd.impl.c.ggml_conv_2d_dw_direct(ctx0, layer.conv2d_spatial_w, gated_2d, 1, 1, 1, 1, 1, 1);
                g.cb(conv_out, "conv_spatial", il);

                conv_out = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, conv_out, 1, 2, 0, 3));
                conv_out = mtmd.impl.c.ggml_reshape_2d(ctx0, conv_out, n_embd, n_pos);
            } else {
                conv_out = gated;
            }

            cur = mtmd.impl.c.ggml_mul(ctx0, middle, conv_out);
            g.cb(cur, "conv_mul", il);

            cur = g.buildMm(layer.conv2d_out_proj_w, cur);
            g.cb(cur, "conv_out_proj", il);
        } else {
            var q_cur = g.buildMm(layer.q_w, cur);
            var k_cur = g.buildMm(layer.k_w, cur);
            var v_cur = g.buildMm(layer.v_w, cur);

            g.cb(q_cur, "Qcur", il);
            g.cb(k_cur, "Kcur", il);
            g.cb(v_cur, "Vcur", il);

            q_cur = mtmd.impl.c.ggml_reshape_3d(ctx0, q_cur, d_head, n_head, n_pos);
            k_cur = mtmd.impl.c.ggml_reshape_3d(ctx0, k_cur, d_head, n_head_kv, n_pos);
            v_cur = mtmd.impl.c.ggml_reshape_3d(ctx0, v_cur, d_head, n_head_kv, n_pos);

            if (layer.q_norm != null) q_cur = g.buildNorm(q_cur, layer.q_norm, null, norm_t, eps, il);
            if (layer.k_norm != null) k_cur = g.buildNorm(k_cur, layer.k_norm, null, norm_t, eps, il);

            q_cur = mtmd.impl.c.ggml_rope_multi(ctx0, q_cur, positions, null, @divTrunc(d_head, 2), &mrope_sections, mtmd.impl.c.GGML_ROPE_TYPE_VISION, 32768, rope_theta, 1, 0, 1, 32, 1);
            k_cur = mtmd.impl.c.ggml_rope_multi(ctx0, k_cur, positions, null, @divTrunc(d_head, 2), &mrope_sections, mtmd.impl.c.GGML_ROPE_TYPE_VISION, 32768, rope_theta, 1, 0, 1, 32, 1);

            g.cb(q_cur, "Qcur_rope", il);
            g.cb(k_cur, "Kcur_rope", il);

            cur = g.buildAttn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, null, g.kq_scale, il, null);
            g.cb(cur, "attn_out", il);
        }

        cur = mtmd.impl.c.ggml_add(ctx0, cur, inp_l);
        inp_l = cur;
        g.cb(cur, "ffn_inp", il);

        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        g.cb(cur, "ffn_inp_normed", il);

        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
        g.cb(cur, "ffn_out", il);

        cur = mtmd.impl.c.ggml_add(ctx0, inp_l, cur);
        g.cb(cur, "layer_out", il);

        inp_l = cur;
    }

    // no post-layernorm for LFM3-VL

    var embeddings = inp_l;

    // projector input LayerNorm, eps is projector_layer_norm_eps (not the encoder eps)
    embeddings = g.buildNorm(embeddings, model.mm_input_norm_w, model.mm_input_norm_b, .normal, 1e-6, -1);
    g.cb(embeddings, "proj_input_norm", -1);

    // 2x2 patch interleave (same as Qwen3VL) so the reshape below merges spatial 2x2 blocks
    {
        embeddings = mtmd.impl.c.ggml_cont_4d(ctx0, embeddings, n_embd * 2, @divTrunc(n_patches_x, 2), n_patches_y, batch_size);
        embeddings = mtmd.impl.c.ggml_reshape_4d(ctx0, embeddings, n_embd * 2, @divTrunc(n_patches_x, 2), 2, batch_size * @divTrunc(n_patches_y, 2));
        embeddings = mtmd.impl.c.ggml_permute(ctx0, embeddings, 0, 2, 1, 3);
        embeddings = mtmd.impl.c.ggml_cont_3d(ctx0, embeddings, n_embd, n_patches_x * n_patches_y, batch_size);
    }
    g.cb(embeddings, "proj_interleaved", -1);

    embeddings = mtmd.impl.c.ggml_reshape_3d(ctx0, embeddings, n_embd * 4, @divTrunc(n_pos, 4), batch_size);

    // projector MLP: linear_1 -> GELU (erf, as HF "gelu") -> linear_2
    embeddings = g.buildFfn(embeddings, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, .gelu_erf, -1);

    mtmd.impl.c.ggml_build_forward_expand(g.gf, embeddings);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
