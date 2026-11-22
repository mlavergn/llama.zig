//! The MiniCPM-V vision encoders: the resampler-projected ViT of MiniCPM-V
//! up to 4.5, and MiniCPM-V 4.6's ViT with an inserted window-attention
//! merger and a 2x2 downsample MLP.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/minicpmv.cpp` at v1.0.0 (`b1b770d90`).

const std = @import("std");
const mtmd = @import("../module.zig");
const stdlib = @import("stdlib");

/// Ports `build` (minicpmv.cpp:3 @b1b770d90), a member of `mtmd_graph_minicpmv`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    mtmd.impl.ggmlAssert(model.class_embedding == null, "minicpmv.cpp", 4, "model.class_embedding == nullptr");
    const n_pos = g.n_patches;
    const n_embd_proj = g.n_mmproj_embd;

    // position embeddings for the projector (not for ViT): base frequency omega
    const omega = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_F32, @divTrunc(n_embd_proj, 4));
    _ = mtmd.impl.c.ggml_set_name(omega, "omega");
    mtmd.impl.c.ggml_set_input(omega);

    // 2D input positions (using float for sinusoidal embeddings)
    const pos_h = mtmd.impl.c.ggml_new_tensor_2d(ctx0, mtmd.impl.c.GGML_TYPE_F32, 1, n_pos);
    _ = mtmd.impl.c.ggml_set_name(pos_h, "pos_h");
    mtmd.impl.c.ggml_set_input(pos_h);
    const pos_w = mtmd.impl.c.ggml_new_tensor_2d(ctx0, mtmd.impl.c.GGML_TYPE_F32, 1, n_pos);
    _ = mtmd.impl.c.ggml_set_name(pos_w, "pos_w");
    mtmd.impl.c.ggml_set_input(pos_w);

    // for selecting learned pos embd, used by ViT
    const positions = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, n_pos);
    _ = mtmd.impl.c.ggml_set_name(positions, "positions");
    mtmd.impl.c.ggml_set_input(positions);

    const learned_pos_embd = mtmd.impl.c.ggml_get_rows(ctx0, model.position_embeddings, positions);

    const inp = g.buildInp();
    var embeddings = g.buildVit(inp, n_pos, .normal, hparams.ffn_op, learned_pos_embd, null, .{});

    // resampler projector (it is just another transformer)

    var q = model.mm_model_query;
    var v = g.buildMm(model.mm_model_kv_proj, embeddings);

    // norm
    q = g.buildNorm(q, model.mm_model_ln_q_w, model.mm_model_ln_q_b, .normal, eps, -1);
    v = g.buildNorm(v, model.mm_model_ln_kv_w, model.mm_model_ln_kv_b, .normal, eps, -1);

    // calculate sinusoidal pos embd
    var pos_embed: mtmd.graph.T = null;
    {
        // outer product
        const omega_b = mtmd.impl.c.ggml_repeat_4d(ctx0, omega, mtmd.graph.ne(omega, 0), n_pos, 1, 1); // n_pos rows
        const theta_x = mtmd.impl.c.ggml_mul(ctx0, omega_b, pos_w);
        const theta_y = mtmd.impl.c.ggml_mul(ctx0, omega_b, pos_h);
        // sin and cos, concatenated on the first dim
        const pos_embd_x = mtmd.impl.c.ggml_concat(ctx0, mtmd.impl.c.ggml_sin(ctx0, theta_x), mtmd.impl.c.ggml_cos(ctx0, theta_x), 0);
        const pos_embd_y = mtmd.impl.c.ggml_concat(ctx0, mtmd.impl.c.ggml_sin(ctx0, theta_y), mtmd.impl.c.ggml_cos(ctx0, theta_y), 0);
        pos_embed = mtmd.impl.c.ggml_concat(ctx0, pos_embd_x, pos_embd_y, 0);
    }

    // k = v + pos_embed
    const k = mtmd.impl.c.ggml_add(ctx0, v, pos_embed);

    // attention
    {
        const d_head = 128;
        const n_head = @divTrunc(n_embd_proj, d_head);
        // Use actual config value if available, otherwise fall back to hardcoded values
        const num_query = hparams.minicpmv_query_num;
        var Q = mtmd.impl.c.ggml_add(ctx0, g.buildMm(model.mm_model_attn_q_w, q), model.mm_model_attn_q_b);
        var K = mtmd.impl.c.ggml_add(ctx0, g.buildMm(model.mm_model_attn_k_w, k), model.mm_model_attn_k_b);
        var V = mtmd.impl.c.ggml_add(ctx0, g.buildMm(model.mm_model_attn_v_w, v), model.mm_model_attn_v_b);

        Q = mtmd.impl.c.ggml_reshape_3d(ctx0, Q, d_head, n_head, num_query);
        K = mtmd.impl.c.ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
        V = mtmd.impl.c.ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);

        g.cb(Q, "resampler_Q", -1);
        g.cb(K, "resampler_K", -1);
        g.cb(V, "resampler_V", -1);

        const resampler_kq_scale: f32 = 1.0 / stdlib.libm.sqrtfInt(d_head);
        embeddings = g.buildAttn(model.mm_model_attn_o_w, model.mm_model_attn_o_b, Q, K, V, null, resampler_kq_scale, -1, null);
        g.cb(embeddings, "resampler_attn_out", -1);
    }
    // layernorm
    embeddings = g.buildNorm(embeddings, model.mm_model_ln_post_w, model.mm_model_ln_post_b, .normal, eps, -1);

    // projection
    embeddings = g.buildMm(model.mm_model_proj, embeddings);

    // build the graph
    mtmd.impl.c.ggml_build_forward_expand(g.gf, embeddings);
    return g.gf;
}

/// The `add_i32_input` lambda (minicpmv.cpp:124 @b1b770d90): a named I32 input.
fn addI32Input(g: *mtmd.graph.Graph, name: [*:0]const u8, n: c_int) mtmd.graph.T {
    const t = mtmd.impl.c.ggml_new_tensor_1d(g.ctx0, mtmd.impl.c.GGML_TYPE_I32, n);
    _ = mtmd.impl.c.ggml_set_name(t, name);
    mtmd.impl.c.ggml_set_input(t);
    return t;
}

/// The `build_vit_layers` lambda (minicpmv.cpp:175 @b1b770d90): layers
/// `[il_begin, il_end)` of the ViT over `n_pos_layer` tokens, with optional
/// layer scales.
fn buildVitLayers(g: *mtmd.graph.Graph, input_in: mtmd.graph.T, il_begin: c_int, il_end: c_int, n_pos_layer: i64) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const model = g.model;
    const eps = g.eps;
    var input = input_in;
    var il = il_begin;
    while (il < il_end) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        var cur = input;

        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .normal, eps, il);
        g.cb(cur, "layer_inp_normed", il);

        {
            var Qcur = g.buildMm(layer.q_w, cur);
            if (layer.q_b != null) Qcur = mtmd.impl.c.ggml_add(ctx0, Qcur, layer.q_b);
            var Kcur = g.buildMm(layer.k_w, cur);
            if (layer.k_b != null) Kcur = mtmd.impl.c.ggml_add(ctx0, Kcur, layer.k_b);
            var Vcur = g.buildMm(layer.v_w, cur);
            if (layer.v_b != null) Vcur = mtmd.impl.c.ggml_add(ctx0, Vcur, layer.v_b);

            Qcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Qcur, g.d_head, g.n_head, n_pos_layer);
            Kcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Kcur, g.d_head, g.n_head, n_pos_layer);
            Vcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Vcur, g.d_head, g.n_head, n_pos_layer);
            g.cb(Qcur, "Qcur", il);
            g.cb(Kcur, "Kcur", il);
            g.cb(Vcur, "Vcur", il);

            cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, null, g.kq_scale, il, null);
            g.cb(cur, "attn_out", il);
        }

        if (layer.ls_1_w != null) {
            cur = mtmd.impl.c.ggml_mul(ctx0, cur, layer.ls_1_w);
            g.cb(cur, "attn_out_scaled", il);
        }
        cur = mtmd.impl.c.ggml_add(ctx0, cur, input);
        input = cur;
        g.cb(cur, "ffn_inp", il);

        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, eps, il);
        g.cb(cur, "ffn_inp_normed", il);

        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, g.hparams.ffn_op, il);
        g.cb(cur, "ffn_out", il);

        if (layer.ls_2_w != null) {
            cur = mtmd.impl.c.ggml_mul(ctx0, cur, layer.ls_2_w);
            g.cb(cur, "ffn_out_scaled", il);
        }
        input = mtmd.impl.c.ggml_add(ctx0, input, cur);
        g.cb(input, "layer_out", il);
    }
    return input;
}

/// Ports `build` (minicpmv.cpp:116 @b1b770d90), a member of `mtmd_graph_minicpmv4_6`.
pub fn build4_6(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const d_head = g.d_head;
    const n_head = g.n_head;
    const is_4x = hparams.n_merge == 2;
    const n_pos = g.n_patches;
    const half_h = @divTrunc(g.n_patches_y, 2);
    const half_w = @divTrunc(g.n_patches_x, 2);
    const n_ds = half_h * half_w;
    const n_out = if (is_4x) n_ds else @divTrunc(half_h, 2) * @divTrunc(half_w, 2);

    // position indices for ViT learned positional embeddings
    const positions = addI32Input(g, "positions", n_pos);
    const learned_pos_embd = mtmd.impl.c.ggml_get_rows(ctx0, model.position_embeddings, positions);

    var vit_merger_window_idx: mtmd.graph.T = null;
    var vit_merger_inv_window_idx: mtmd.graph.T = null;
    var vit_merger_window_mask: mtmd.graph.T = null;
    var vit_merger_ds_idx_0: mtmd.graph.T = null;
    var vit_merger_ds_idx_1: mtmd.graph.T = null;
    var vit_merger_ds_idx_2: mtmd.graph.T = null;
    var vit_merger_ds_idx_3: mtmd.graph.T = null;

    if (!is_4x) {
        // ViT merger window reorder indices + block-diagonal mask
        // (mask layout follows qwen2vl: -inf except for 4x4 blocks on the diagonal,
        // so each window-major group of 4 tokens only attends to itself)
        vit_merger_window_idx = addI32Input(g, "vit_merger_window_idx", n_pos);
        vit_merger_inv_window_idx = addI32Input(g, "vit_merger_inv_window_idx", n_pos);
        vit_merger_window_mask = g.buildInpAttnMask(n_pos, n_pos, 0);

        // ViT merger 2x2 downsample gather indices
        vit_merger_ds_idx_0 = addI32Input(g, "vit_merger_ds_idx_0", n_ds);
        vit_merger_ds_idx_1 = addI32Input(g, "vit_merger_ds_idx_1", n_ds);
        vit_merger_ds_idx_2 = addI32Input(g, "vit_merger_ds_idx_2", n_ds);
        vit_merger_ds_idx_3 = addI32Input(g, "vit_merger_ds_idx_3", n_ds);
    }

    // final merger 2x2 downsample gather indices
    const merger_ds_idx_0 = addI32Input(g, "merger_ds_idx_0", n_out);
    const merger_ds_idx_1 = addI32Input(g, "merger_ds_idx_1", n_out);
    const merger_ds_idx_2 = addI32Input(g, "merger_ds_idx_2", n_out);
    const merger_ds_idx_3 = addI32Input(g, "merger_ds_idx_3", n_out);

    // patch embedding + positional embedding
    var inp = g.buildInp();
    inp = mtmd.impl.c.ggml_add(ctx0, inp, learned_pos_embd);
    g.cb(inp, "pos_embed", -1);

    var inpL = inp;
    if (model.pre_ln_w != null) {
        inpL = g.buildNorm(inpL, model.pre_ln_w, model.pre_ln_b, .normal, eps, -1);
        g.cb(inpL, "pre_ln", -1);
    }

    if (!is_4x) {
        const insert_lid = hparams.insert_layer_id;

        inpL = buildVitLayers(g, inpL, 0, insert_lid + 1, n_pos);

        // ViT merger: window self-attention. Tokens are reordered to
        // window-major (4 tokens per window are contiguous), and a
        // block-diagonal mask restricts attention to within each window.
        {
            const residual = inpL;
            var cur = g.buildNorm(inpL, model.vit_merger_ln1_w, model.vit_merger_ln1_b, .normal, eps, -1);
            g.cb(cur, "vit_merger_attn_inp_normed", -1);

            cur = mtmd.impl.c.ggml_get_rows(ctx0, cur, vit_merger_window_idx);
            g.cb(cur, "vit_merger_window_reorder", -1);

            var Qcur = g.buildMm(model.vit_merger_attn_q_w, cur);
            if (model.vit_merger_attn_q_b != null) Qcur = mtmd.impl.c.ggml_add(ctx0, Qcur, model.vit_merger_attn_q_b);
            var Kcur = g.buildMm(model.vit_merger_attn_k_w, cur);
            if (model.vit_merger_attn_k_b != null) Kcur = mtmd.impl.c.ggml_add(ctx0, Kcur, model.vit_merger_attn_k_b);
            var Vcur = g.buildMm(model.vit_merger_attn_v_w, cur);
            if (model.vit_merger_attn_v_b != null) Vcur = mtmd.impl.c.ggml_add(ctx0, Vcur, model.vit_merger_attn_v_b);

            Qcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
            g.cb(Qcur, "vit_merger_Qcur", -1);
            g.cb(Kcur, "vit_merger_Kcur", -1);
            g.cb(Vcur, "vit_merger_Vcur", -1);

            cur = g.buildAttn(model.vit_merger_attn_o_w, model.vit_merger_attn_o_b, Qcur, Kcur, Vcur, vit_merger_window_mask, g.kq_scale, -1, null);
            g.cb(cur, "vit_merger_attn_out", -1);

            cur = mtmd.impl.c.ggml_get_rows(ctx0, cur, vit_merger_inv_window_idx);
            inpL = mtmd.impl.c.ggml_add(ctx0, cur, residual);
            g.cb(inpL, "vit_merger_attn_residual", -1);
        }

        // ViT merger: 2x2 spatial downsample + MLP (4 tokens -> 1)
        {
            const p0 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_0);
            const p1 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_1);
            const p2 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_2);
            const p3 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, vit_merger_ds_idx_3);

            var mean_res = mtmd.impl.c.ggml_add(ctx0, p0, p1);
            mean_res = mtmd.impl.c.ggml_add(ctx0, mean_res, p2);
            mean_res = mtmd.impl.c.ggml_add(ctx0, mean_res, p3);
            mean_res = mtmd.impl.c.ggml_scale(ctx0, mean_res, 0.25);
            g.cb(mean_res, "vit_merger_ds_mean_res", -1);

            var cat = mtmd.impl.c.ggml_concat(ctx0, p0, p1, 0);
            cat = mtmd.impl.c.ggml_concat(ctx0, cat, p2, 0);
            cat = mtmd.impl.c.ggml_concat(ctx0, cat, p3, 0);

            var cur = g.buildNorm(cat, model.vit_merger_ds_ln_w, model.vit_merger_ds_ln_b, .normal, eps, -1);
            g.cb(cur, "vit_merger_ds_normed", -1);

            // ViTWindowAttentionMerger downsample MLP uses gelu_pytorch_tanh (FFN_GELU)
            cur = g.buildFfn(cur, model.vit_merger_ds_up_w, model.vit_merger_ds_up_b, null, null, model.vit_merger_ds_down_w, model.vit_merger_ds_down_b, .gelu, -1);
            g.cb(cur, "vit_merger_ds_mlp_out", -1);

            inpL = mtmd.impl.c.ggml_add(ctx0, cur, mean_res);
            g.cb(inpL, "vit_merger_ds_out", -1);
        }

        inpL = buildVitLayers(g, inpL, insert_lid + 1, g.n_layer, n_ds);
    } else {
        inpL = buildVitLayers(g, inpL, 0, g.n_layer, n_pos);
    }

    if (model.post_ln_w != null) {
        inpL = g.buildNorm(inpL, model.post_ln_w, model.post_ln_b, .normal, eps, -1);
        g.cb(inpL, "post_ln", -1);
    }

    // Final Merger (DownsampleMLP): another 2x2 spatial merge -> projector embedding
    {
        const p0 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, merger_ds_idx_0);
        const p1 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, merger_ds_idx_1);
        const p2 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, merger_ds_idx_2);
        const p3 = mtmd.impl.c.ggml_get_rows(ctx0, inpL, merger_ds_idx_3);

        var cat = mtmd.impl.c.ggml_concat(ctx0, p0, p1, 0);
        cat = mtmd.impl.c.ggml_concat(ctx0, cat, p2, 0);
        cat = mtmd.impl.c.ggml_concat(ctx0, cat, p3, 0);

        var cur = g.buildNorm(cat, model.mm_input_norm_w, model.mm_input_norm_b, .normal, eps, -1);
        g.cb(cur, "merger_normed", -1);

        // MiniCPMV4_6DownsampleMLP uses nn.GELU() (erf-based, FFN_GELU_ERF)
        cur = g.buildFfn(cur, model.mm_ffn_up_w, model.mm_ffn_up_b, null, null, model.mm_ffn_down_w, model.mm_ffn_down_b, .gelu_erf, -1);
        g.cb(cur, "merger_out", -1);

        inpL = cur;
    }

    inpL = g.buildSuffix(inpL);

    mtmd.impl.c.ggml_build_forward_expand(g.gf, inpL);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
