//! The SigLIP vision encoder, with the gemma3, idefics3, lfm2, janus-pro and
//! phi4 projectors.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/siglip.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;

/// Ports `build` (siglip.cpp:3 @b29c606e2), a member of `clip_graph_siglip`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const pt = g.proj_type;

    const inp = g.buildInp();

    var learned_pos_embd = model.position_embeddings;
    if (pt == .lfm2 or pt == .phi4) learned_pos_embd = g.resizePositionEmbeddings(graph.default_interpolation_mode);

    var cur = g.buildVit(inp, g.n_patches, .normal, hparams.ffn_op, learned_pos_embd, null, .{});

    if (pt == .gemma3) {
        const batch_size = 1;
        impl.ggmlAssert(g.n_patches_x == g.n_patches_y, "siglip.cpp", 20, "n_patches_x == n_patches_y");
        const patches_per_image = g.n_patches_x;
        const kernel_size = hparams.n_merge;

        cur = c.ggml_transpose(ctx0, cur);
        cur = c.ggml_cont_4d(ctx0, cur, patches_per_image, patches_per_image, g.n_embd, batch_size);

        // doing a pool2d to reduce the number of output tokens
        cur = c.ggml_pool_2d(ctx0, cur, c.GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        cur = c.ggml_reshape_3d(ctx0, cur, graph.ne(cur, 0) * graph.ne(cur, 0), g.n_embd, batch_size);
        cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));

        // apply norm before projection
        cur = c.ggml_rms_norm(ctx0, cur, g.eps);
        cur = c.ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

        // apply projection
        cur = c.ggml_mul_mat(ctx0, c.ggml_cont(ctx0, c.ggml_transpose(ctx0, model.mm_input_proj_w)), cur);
    } else if (pt == .idefics3) {
        // pixel_shuffle
        const scale_factor = hparams.n_merge;
        cur = g.buildPatchMergePermute(cur, scale_factor);
        cur = g.buildMm(model.mm_fc_w, cur);
    } else if (pt == .lfm2) {
        // pixel unshuffle block
        const scale_factor = hparams.n_merge;
        cur = g.buildPatchMergePermute(cur, scale_factor);

        // projection, in LFM2-VL input norm is optional
        if (model.mm_input_norm_w != null) {
            cur = c.ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
            cur = c.ggml_mul(ctx0, cur, model.mm_input_norm_w);
        }
        if (model.mm_input_norm_b != null) cur = c.ggml_add(ctx0, cur, model.mm_input_norm_b);

        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu, -1);
    } else if (pt == .janus_pro) {
        cur = g.buildFfn(cur, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, hparams.ffn_op, -1);
    } else if (pt == .phi4) {
        cur = g.buildFfn(cur, model.mm_0_w, model.mm_0_b, null, null, model.mm_2_w, model.mm_2_b, .gelu, -1);
    } else {
        impl.abort("siglip.cpp", 87, "SigLIP: Unsupported projector type");
    }

    // build the graph
    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
