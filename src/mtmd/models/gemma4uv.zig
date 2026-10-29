//! The Gemma 4 unified vision embedder: normed im2col patches, lookup-table
//! position embeddings and the multimodal embedder, with no transformer.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/gemma4uv.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const gemma4v = @import("gemma4v.zig");
const c = impl.c;

/// Ports `build` (gemma4uv.cpp:4 @b29c606e2), a member of `clip_graph_gemma4uv`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const patch_size = g.patch_size;

    const inp_raw = g.buildInpRaw(3);

    // Gemma4UnifiedVisionEmbedder uses default pytorch LayerNorm, not RMSNorm
    const eps: f32 = 1e-5; // default eps for pytorch LayerNorm

    var inp: graph.T = null;
    {
        // note: we cannot use ggml_conv_2d here because we need to apply norm after im2col
        const ch = graph.ne(inp_raw, 2);
        const kernel = c.ggml_new_tensor_3d(ctx0, c.GGML_TYPE_F32, patch_size, patch_size, ch);
        inp = c.ggml_im2col(ctx0, kernel, inp_raw, patch_size, patch_size, 0, 0, 1, 1, true, inp_raw.?.type);
        // inp shape: [patch_size * patch_size * c, n_patches_w, n_patches_h]

        inp = c.ggml_reshape_2d(ctx0, inp, graph.ne(inp, 0), graph.ne(inp, 1) * graph.ne(inp, 2) * graph.ne(inp, 3));
        inp = g.buildNorm(inp, model.patch_norm_1_w, model.patch_norm_1_b, .normal, eps, -1);
        // inp shape: [patch_size * patch_size * c, n_patches]

        inp = c.ggml_mul_mat(ctx0, model.patch_embeddings_0, inp);
        inp = c.ggml_add(ctx0, inp, model.patch_bias);
        // inp shape: [n_embd, n_patches]

        inp = g.buildNorm(inp, model.patch_norm_2_w, model.patch_norm_2_b, .normal, eps, -1);
    }

    const pos_x = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_x, "pos_x");
    c.ggml_set_input(pos_x);

    const pos_y = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_y, "pos_y");
    c.ggml_set_input(pos_y);

    {
        inp = gemma4v.addPosEmbd(g, inp, pos_x, pos_y);

        // pos_norm
        inp = g.buildNorm(inp, model.patch_norm_3_w, model.patch_norm_3_b, .normal, eps, -1);
    }

    var cur = inp;

    // Gemma4UnifiedMultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = c.ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = g.buildMm(model.mm_input_proj_w, cur);
        g.cb(cur, "projected", -1);
    }

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
