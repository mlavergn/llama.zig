//! The Gemma 4 unified audio embedder: the precomputed audio features,
//! RMS-normalised and projected into the text model's space.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/gemma4ua.cpp` at v0.4.0
//! (`5266f24da`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;

/// Ports `build` (gemma4ua.cpp:4 @5266f24da), a member of
/// `clip_graph_gemma4ua`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const inp = g.buildInpRaw(1);

    var cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // Gemma4UnifiedMultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = c.ggml_rms_norm(ctx0, cur, g.hparams.eps);
        cur = g.buildMm(g.model.mm_input_proj_w, cur);
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
