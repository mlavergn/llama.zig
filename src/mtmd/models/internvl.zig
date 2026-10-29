//! The InternVL vision encoder: a ViT with a CLS token, a pixel shuffle and
//! a GELU projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/internvl.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

/// Ports `build` (internvl.cpp:3 @b29c606e2), a member of `clip_graph_internvl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    impl.ggmlAssert(model.class_embedding != null, "internvl.cpp", 4, "model.class_embedding != nullptr");
    impl.ggmlAssert(model.position_embeddings != null, "internvl.cpp", 5, "model.position_embeddings != nullptr");

    const n_pos = g.n_patches + 1;
    var inp = g.buildInp();

    // add CLS token
    const cls_repeated = c.ggml_repeat_4d(ctx0, model.class_embedding, graph.ne(model.class_embedding, 0), 1, g.n_batch, 1);
    inp = c.ggml_concat(ctx0, inp, cls_repeated, 1);

    // The larger models use a different ViT, which uses RMS norm instead of layer norm
    const norm_t: model_mod.NormType = if (hparams.n_embd == 3200 and hparams.n_layer == 45)
        .rms // 6B ViT (Used by InternVL 2.5/3 - 26B, 38B, 78B)
    else
        .normal; // 300M ViT (Used by all smaller InternVL models)

    var cur = g.buildVit(inp, n_pos, norm_t, hparams.ffn_op, model.position_embeddings, null, .{});

    // remove CLS token
    cur = c.ggml_view_3d(ctx0, cur, g.n_embd, g.n_patches, g.n_batch, graph.nb(cur, 1), graph.nb(cur, 2), 0);
    cur = c.ggml_cont(ctx0, cur);

    // pixel shuffle
    {
        const scale_factor = hparams.n_merge;
        const bsz = g.n_batch;
        const height = g.n_patches_y;
        const width = g.n_patches_x;
        impl.ggmlAssert(scale_factor > 0, "internvl.cpp", 40, "scale_factor > 0");
        cur = c.ggml_reshape_4d(ctx0, cur, g.n_embd * scale_factor, @divTrunc(height, scale_factor), width, bsz);
        cur = c.ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = c.ggml_cont_4d(ctx0, cur, g.n_embd * scale_factor * scale_factor, @divTrunc(height, scale_factor), @divTrunc(width, scale_factor), bsz);
        cur = c.ggml_permute(ctx0, cur, 0, 2, 1, 3);
        // flatten to 2D
        cur = c.ggml_cont_3d(ctx0, cur, g.n_embd * scale_factor * scale_factor, graph.ne(cur, 1) * graph.ne(cur, 2), graph.ne(cur, 3));
    }

    // projector (always using GELU activation)
    {
        // projector LayerNorm uses pytorch's default eps = 1e-5
        cur = g.buildNorm(cur, model.mm_0_w, model.mm_0_b, .normal, 1e-5, -1);
        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_3_w, model.mm_3_b, .gelu, -1);
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
