//! The PaddleOCR-VL vision encoder: an M-RoPE ViT with resized learned
//! position embeddings and the mlp_AR projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/paddleocr.cpp` at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const dotsocr = @import("dotsocr.zig");
const c = impl.c;

/// Ports `build` (mtmd/models/paddleocr.cpp:3 @c1d0e7a00), a member of `clip_graph_paddleocr`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_pos = g.n_patches;
    const num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    const q = @divTrunc(g.d_head, 4);
    var rope: dotsocr.MropeVision = .{ .positions = null, .mrope_sections = .{ q, q, q, q } };

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, num_position_ids);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);
    rope.positions = positions;

    const learned_pos_embd = g.resizePositionEmbeddings(graph.default_interpolation_mode);
    const inp = g.buildInp();
    var cur = g.buildVit(inp, g.n_patches, .normal, hparams.ffn_op, learned_pos_embd, .{ .ctx = &rope, .f = dotsocr.MropeVision.addPos }, .{});

    g.cb(cur, "vit_out", -1);

    {
        // mlp_AR paddleocr projector
        const proj_norm_eps: f32 = 1e-5;
        cur = g.buildNorm(cur, model.mm_input_norm_w, model.mm_input_norm_b, .normal, proj_norm_eps, -1);

        const scale_factor = hparams.n_merge;
        cur = g.buildPatchMergePermute(cur, scale_factor);
        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, hparams.ffn_op, -1);
        g.cb(cur, "mlp_out", -1);
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
