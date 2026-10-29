//! The graph llava, granite and glm-edge share: a ViT written out rather
//! than through `buildVit`, since granite stacks feature layers, then one
//! of the llava projectors (MLP, MLP+norm, MobileVLM's LDP and LDPv2) or
//! glm-edge's adapter.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/llava.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;

/// Ports `build` (llava.cpp:5 @b29c606e2), a member of `clip_graph_llava`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const batch_size: c_int = 1;
    const n_pos: c_int = g.n_patches + @as(c_int, if (model.class_embedding != null) 1 else 0);

    impl.ggmlAssert(g.n_patches_x == g.n_patches_y, "llava.cpp", 9, "n_patches_x == n_patches_y && \"only square images supported\"");

    // Calculate the deepest feature layer based on hparams and projector type
    var max_feature_layer: c_int = g.n_layer;
    {
        // Get the index of the second to last layer; this is the default for models that have a llava projector
        var il_last: c_int = hparams.n_layer - 1;
        var deepest_feature_layer: c_int = -1;

        if (g.proj_type == .minicpmv or g.proj_type == .glm_edge) il_last += 1;

        // If we set explicit vision feature layers, only go up to the deepest one
        // NOTE: only used by granite-vision models for now
        for (hparams.feature_layers.items) |feature_layer| {
            if (feature_layer > deepest_feature_layer) deepest_feature_layer = feature_layer;
        }
        max_feature_layer = if (deepest_feature_layer < 0) il_last else deepest_feature_layer;
    }

    var inp = g.buildInp();

    // concat class_embeddings and patch_embeddings
    if (model.class_embedding != null) inp = c.ggml_concat(ctx0, inp, model.class_embedding, 1);

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);

    inp = c.ggml_add(ctx0, inp, c.ggml_get_rows(ctx0, model.position_embeddings, positions));

    var inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w != null) {
        inpL = g.buildNorm(inpL, model.pre_ln_w, model.pre_ln_b, .normal, eps, -1);
        g.cb(inpL, "pre_ln", -1);
    }

    var embedding_stack: std.ArrayList(T) = .empty;
    defer embedding_stack.deinit(impl.gpa);

    // loop over layers
    var il: c_int = 0;
    while (il < max_feature_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        var cur = inpL; // inpL = residual, cur = hidden_states

        // If this is an embedding feature layer, save the output.
        // NOTE: 0 index here refers to the input to the encoder.
        if (hparams.isFeatureLayer(il)) try embedding_stack.append(impl.gpa, cur);

        // layernorm1
        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .normal, eps, il);
        g.cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            var Qcur = g.buildMm(layer.q_w, cur);
            if (layer.q_b != null) Qcur = c.ggml_add(ctx0, Qcur, layer.q_b);

            var Kcur = g.buildMm(layer.k_w, cur);
            if (layer.k_b != null) Kcur = c.ggml_add(ctx0, Kcur, layer.k_b);

            var Vcur = g.buildMm(layer.v_w, cur);
            if (layer.v_b != null) Vcur = c.ggml_add(ctx0, Vcur, layer.v_b);

            Qcur = c.ggml_reshape_3d(ctx0, Qcur, g.d_head, g.n_head, n_pos);
            Kcur = c.ggml_reshape_3d(ctx0, Kcur, g.d_head, g.n_head, n_pos);
            Vcur = c.ggml_reshape_3d(ctx0, Vcur, g.d_head, g.n_head, n_pos);

            g.cb(Qcur, "Qcur", il);
            g.cb(Kcur, "Kcur", il);
            g.cb(Vcur, "Vcur", il);

            cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, null, g.kq_scale, il, null);
            g.cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = c.ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        g.cb(cur, "ffn_inp", il);

        // layernorm2
        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, eps, il);
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
    if (model.post_ln_w != null) inpL = g.buildNorm(inpL, model.post_ln_w, model.post_ln_b, .normal, eps, -1);

    var embeddings = inpL;

    // process vision feature layers (used by granite)
    {
        // final layer is a vision feature layer
        if (hparams.isFeatureLayer(max_feature_layer)) try embedding_stack.append(impl.gpa, inpL);

        // If feature layers are explicitly set, stack them (if we have multiple)
        if (embedding_stack.items.len != 0) {
            embeddings = embedding_stack.items[0];
            for (embedding_stack.items[1..]) |e| embeddings = c.ggml_concat(ctx0, embeddings, e, 0);
        }
    }

    // llava projector (also used by granite)
    if (hparams.has_llava_projector) {
        embeddings = c.ggml_reshape_2d(ctx0, embeddings, ne(embeddings, 0), ne(embeddings, 1));

        const patches = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
        _ = c.ggml_set_name(patches, "patches");
        c.ggml_set_input(patches);

        // shape [1, 576, 1024]
        // ne is whcn, ne = [1024, 576, 1, 1]
        embeddings = c.ggml_get_rows(ctx0, embeddings, patches);

        // llava projector
        if (g.proj_type == .mlp) {
            embeddings = g.buildMm(model.mm_0_w, embeddings);
            embeddings = c.ggml_add(ctx0, embeddings, model.mm_0_b);

            embeddings = c.ggml_gelu(ctx0, embeddings);
            if (model.mm_2_w != null) {
                embeddings = g.buildMm(model.mm_2_w, embeddings);
                embeddings = c.ggml_add(ctx0, embeddings, model.mm_2_b);
            }
        } else if (g.proj_type == .mlp_norm) {
            embeddings = g.buildMm(model.mm_0_w, embeddings);
            embeddings = c.ggml_add(ctx0, embeddings, model.mm_0_b);
            // First LayerNorm
            embeddings = c.ggml_norm(ctx0, embeddings, eps);
            embeddings = c.ggml_add(ctx0, c.ggml_mul(ctx0, embeddings, model.mm_1_w), model.mm_1_b);

            // GELU activation
            embeddings = c.ggml_gelu(ctx0, embeddings);

            // Second linear layer
            embeddings = g.buildMm(model.mm_3_w, embeddings);
            embeddings = c.ggml_add(ctx0, embeddings, model.mm_3_b);

            // Second LayerNorm
            embeddings = c.ggml_norm(ctx0, embeddings, eps);
            embeddings = c.ggml_add(ctx0, c.ggml_mul(ctx0, embeddings, model.mm_4_w), model.mm_4_b);
        } else if (g.proj_type == .ldp) {
            // MobileVLM projector
            const n_patch: c_int = 24;
            var mlp_1 = g.buildMm(model.mm_model_mlp_1_w, embeddings);
            mlp_1 = c.ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
            mlp_1 = c.ggml_gelu(ctx0, mlp_1);
            var mlp_3 = g.buildMm(model.mm_model_mlp_3_w, mlp_1);
            mlp_3 = c.ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
            // mlp_3 shape = [1, 576, 2048], ne = [2048, 576, 1, 1]

            // block 1
            var block_1: T = null;
            {
                // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
                mlp_3 = c.ggml_permute(ctx0, mlp_3, 1, 0, 2, 3);
                mlp_3 = c.ggml_cont_4d(ctx0, mlp_3, n_patch, n_patch, ne(mlp_3, 1), ne(mlp_3, 2));
                // stride = 1, padding = 1, bias is nullptr
                block_1 = c.ggml_conv_2d_dw(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

                // layer norm
                block_1 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                block_1 = c.ggml_norm(ctx0, block_1, eps);
                block_1 = c.ggml_add(ctx0, c.ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w), model.mm_model_block_1_block_0_1_b);
                block_1 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, block_1, 2, 0, 1, 3));

                // hardswish
                const block_1_hw = c.ggml_hardswish(ctx0, block_1);

                block_1 = c.ggml_pool_2d(ctx0, block_1_hw, c.GGML_OP_POOL_AVG, @intCast(ne(block_1_hw, 0)), @intCast(ne(block_1_hw, 1)), @intCast(ne(block_1_hw, 0)), @intCast(ne(block_1_hw, 1)), 0, 0);
                // pointwise conv
                block_1 = c.ggml_reshape_2d(ctx0, block_1, ne(block_1, 0) * ne(block_1, 1) * ne(block_1, 2), ne(block_1, 3));
                block_1 = g.buildMm(model.mm_model_block_1_block_1_fc1_w, block_1);
                block_1 = c.ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
                block_1 = c.ggml_relu(ctx0, block_1);
                block_1 = g.buildMm(model.mm_model_block_1_block_1_fc2_w, block_1);
                block_1 = c.ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
                block_1 = c.ggml_hardsigmoid(ctx0, block_1);
                block_1 = c.ggml_reshape_4d(ctx0, block_1, 1, 1, ne(block_1, 0), ne(block_1, 1));
                block_1 = c.ggml_mul(ctx0, block_1_hw, block_1);

                const w: c_int = @truncate(ne(block_1, 0));
                const h: c_int = @truncate(ne(block_1, 1));
                block_1 = c.ggml_reshape_3d(ctx0, block_1, w * h, ne(block_1, 2), ne(block_1, 3));
                block_1 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, block_1, 1, 0, 2, 3));

                block_1 = g.buildMm(model.mm_model_block_1_block_2_0_w, block_1);
                block_1 = c.ggml_reshape_4d(ctx0, block_1, ne(block_1, 0), w, h, ne(block_1, 3));

                block_1 = c.ggml_norm(ctx0, block_1, eps);
                block_1 = c.ggml_add(ctx0, c.ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w), model.mm_model_block_1_block_2_1_b);
                block_1 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // residual
                block_1 = c.ggml_add(ctx0, mlp_3, block_1);
            }

            // block_2
            {
                // stride = 2
                block_1 = c.ggml_conv_2d_dw(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

                // layer norm
                block_1 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                block_1 = c.ggml_norm(ctx0, block_1, eps);
                block_1 = c.ggml_add(ctx0, c.ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w), model.mm_model_block_2_block_0_1_b);
                block_1 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // hardswish
                const block_1_hw = c.ggml_hardswish(ctx0, block_1);

                // not sure the parameters is right for globalAvgPooling
                block_1 = c.ggml_pool_2d(ctx0, block_1_hw, c.GGML_OP_POOL_AVG, @intCast(ne(block_1_hw, 0)), @intCast(ne(block_1_hw, 1)), @intCast(ne(block_1_hw, 0)), @intCast(ne(block_1_hw, 1)), 0, 0);
                // pointwise conv
                block_1 = c.ggml_reshape_2d(ctx0, block_1, ne(block_1, 0) * ne(block_1, 1) * ne(block_1, 2), ne(block_1, 3));
                block_1 = g.buildMm(model.mm_model_block_2_block_1_fc1_w, block_1);
                block_1 = c.ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
                block_1 = c.ggml_relu(ctx0, block_1);
                block_1 = g.buildMm(model.mm_model_block_2_block_1_fc2_w, block_1);
                block_1 = c.ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
                block_1 = c.ggml_hardsigmoid(ctx0, block_1);

                block_1 = c.ggml_reshape_4d(ctx0, block_1, 1, 1, ne(block_1, 0), ne(block_1, 1));
                block_1 = c.ggml_mul(ctx0, block_1_hw, block_1);

                const w: c_int = @truncate(ne(block_1, 0));
                const h: c_int = @truncate(ne(block_1, 1));
                block_1 = c.ggml_reshape_3d(ctx0, block_1, w * h, ne(block_1, 2), ne(block_1, 3));
                block_1 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, block_1, 1, 0, 2, 3));
                block_1 = g.buildMm(model.mm_model_block_2_block_2_0_w, block_1);
                block_1 = c.ggml_reshape_4d(ctx0, block_1, ne(block_1, 0), w, h, ne(block_1, 3));

                block_1 = c.ggml_norm(ctx0, block_1, eps);
                block_1 = c.ggml_add(ctx0, c.ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w), model.mm_model_block_2_block_2_1_b);
                block_1 = c.ggml_reshape_3d(ctx0, block_1, ne(block_1, 0), ne(block_1, 1) * ne(block_1, 2), ne(block_1, 3));
            }
            embeddings = block_1;
        } else if (g.proj_type == .ldpv2) {
            const n_patch: c_int = 24;
            var mlp_0 = g.buildMm(model.mm_model_mlp_0_w, embeddings);
            mlp_0 = c.ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
            mlp_0 = c.ggml_gelu(ctx0, mlp_0);
            var mlp_2 = g.buildMm(model.mm_model_mlp_2_w, mlp_0);
            mlp_2 = c.ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
            // AVG Pool Layer 2*2, strides = 2
            mlp_2 = c.ggml_permute(ctx0, mlp_2, 1, 0, 2, 3);
            mlp_2 = c.ggml_cont_4d(ctx0, mlp_2, n_patch, n_patch, ne(mlp_2, 1), ne(mlp_2, 2));
            mlp_2 = c.ggml_pool_2d(ctx0, mlp_2, c.GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
            var peg_0 = c.ggml_conv_2d_dw(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
            peg_0 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
            peg_0 = c.ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
            mlp_2 = c.ggml_cont(ctx0, c.ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
            peg_0 = c.ggml_add(ctx0, peg_0, mlp_2);
            peg_0 = c.ggml_reshape_3d(ctx0, peg_0, ne(peg_0, 0), ne(peg_0, 1) * ne(peg_0, 2), ne(peg_0, 3));
            embeddings = peg_0;
        } else {
            impl.abort("llava.cpp", 331, "fatal error");
        }
    }

    // glm projector
    else if (g.proj_type == .glm_edge) {
        const gridsz: usize = @intFromFloat(@sqrt(@as(f64, @floatFromInt(ne(embeddings, 1)))));
        embeddings = c.ggml_permute(ctx0, embeddings, 1, 0, 2, 3);
        embeddings = c.ggml_cont_3d(ctx0, embeddings, @intCast(gridsz), @intCast(gridsz), ne(embeddings, 1));
        embeddings = c.ggml_conv_2d(ctx0, model.mm_model_adapter_conv_w, embeddings, 2, 2, 0, 0, 1, 1);
        embeddings = c.ggml_reshape_3d(ctx0, embeddings, ne(embeddings, 0) * ne(embeddings, 1), ne(embeddings, 2), batch_size);
        embeddings = c.ggml_cont(ctx0, c.ggml_permute(ctx0, embeddings, 1, 0, 2, 3));
        embeddings = c.ggml_add(ctx0, embeddings, model.mm_model_adapter_conv_b);
        // GLU
        {
            embeddings = g.buildMm(model.mm_model_mlp_0_w, embeddings);
            embeddings = c.ggml_norm(ctx0, embeddings, eps);
            embeddings = c.ggml_add(ctx0, c.ggml_mul(ctx0, embeddings, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
            embeddings = c.ggml_gelu_inplace(ctx0, embeddings);
            var x = embeddings;
            embeddings = g.buildMm(model.mm_model_mlp_2_w, embeddings);
            x = g.buildMm(model.mm_model_mlp_1_w, x);
            embeddings = c.ggml_swiglu_split(ctx0, embeddings, x);
            embeddings = g.buildMm(model.mm_model_mlp_3_w, embeddings);
        }
        // arrangement of BOI/EOI token embeddings
        // note: these embeddings are not present in text model, hence we cannot process them as text tokens
        {
            embeddings = c.ggml_concat(ctx0, model.mm_boi, embeddings, 1); // BOI
            embeddings = c.ggml_concat(ctx0, embeddings, model.mm_eoi, 1); // EOI
        }
    } else {
        impl.abort("llava.cpp", 366, "llava: unknown projector type");
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
