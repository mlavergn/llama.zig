//! The Gemma 4 unified audio embedder: the precomputed audio features,
//! RMS-normalised and projected into the text model's space.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/gemma4ua.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `build` (gemma4ua.cpp:4 @b1b770d90), a member of
/// `mtmd_graph_gemma4ua`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const inp = g.buildInpRaw(1);

    var cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // Gemma4UnifiedMultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = mtmd.impl.c.ggml_rms_norm(ctx0, cur, g.hparams.eps);
        cur = g.buildMm(g.model.mm_input_proj_w, cur);
        g.cb(cur, "projected", -1);
    }

    mtmd.impl.c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
