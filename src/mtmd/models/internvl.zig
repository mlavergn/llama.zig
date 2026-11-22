//! The InternVL vision encoder: a ViT with a CLS token, a pixel shuffle and
//! a GELU projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/internvl.cpp` at v1.0.0 (`b1b770d90`).

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `build` (internvl.cpp:3 @b1b770d90), a member of `mtmd_graph_internvl`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    mtmd.impl.ggmlAssert(model.class_embedding != null, "internvl.cpp", 4, "model.class_embedding != nullptr");
    mtmd.impl.ggmlAssert(model.position_embeddings != null, "internvl.cpp", 5, "model.position_embeddings != nullptr");

    const n_pos = g.n_patches + 1;
    var inp = g.buildInp();

    // add CLS token
    const cls_repeated = mtmd.impl.c.ggml_repeat_4d(ctx0, model.class_embedding, mtmd.graph.ne(model.class_embedding, 0), 1, g.n_batch, 1);
    inp = mtmd.impl.c.ggml_concat(ctx0, cls_repeated, inp, 1);

    // The larger models use a different ViT, which uses RMS norm instead of layer norm
    const norm_t: mtmd.model.NormType = if (hparams.n_embd == 3200 and hparams.n_layer == 45)
        .rms // 6B ViT (Used by InternVL 2.5/3 - 26B, 38B, 78B)
    else
        .normal; // 300M ViT (Used by all smaller InternVL models)

    var cur = g.buildVit(inp, n_pos, norm_t, hparams.ffn_op, model.position_embeddings, null, .{});

    // remove CLS token
    cur = mtmd.impl.c.ggml_view_3d(ctx0, cur, g.n_embd, g.n_patches, g.n_batch, mtmd.graph.nb(cur, 1), mtmd.graph.nb(cur, 2), mtmd.graph.nb(cur, 1));
    cur = mtmd.impl.c.ggml_cont(ctx0, cur);

    // pixel shuffle
    {
        const scale_factor = hparams.n_merge;
        const bsz = g.n_batch;
        const height = g.n_patches_y;
        const width = g.n_patches_x;
        mtmd.impl.ggmlAssert(scale_factor > 0, "internvl.cpp", 40, "scale_factor > 0");
        cur = mtmd.impl.c.ggml_reshape_4d(ctx0, cur, g.n_embd * scale_factor, @divTrunc(height, scale_factor), width, bsz);
        cur = mtmd.impl.c.ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = mtmd.impl.c.ggml_cont_4d(ctx0, cur, g.n_embd * scale_factor * scale_factor, @divTrunc(height, scale_factor), @divTrunc(width, scale_factor), bsz);
        cur = mtmd.impl.c.ggml_permute(ctx0, cur, 0, 2, 1, 3);
        // flatten to 2D
        cur = mtmd.impl.c.ggml_cont_3d(ctx0, cur, g.n_embd * scale_factor * scale_factor, mtmd.graph.ne(cur, 1) * mtmd.graph.ne(cur, 2), mtmd.graph.ne(cur, 3));
    }

    // projector (always using GELU activation)
    {
        // projector LayerNorm uses pytorch's default eps = 1e-5
        cur = g.buildNorm(cur, model.mm_0_w, model.mm_0_b, .normal, 1e-5, -1);
        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_3_w, model.mm_3_b, .gelu, -1);
    }

    // build the graph
    mtmd.impl.c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
