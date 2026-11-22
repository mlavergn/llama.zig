//! The Nemotron v2 VL vision encoder: a ViT with register tokens, a pixel
//! shuffle and a squared-ReLU projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/nemotron-v2-vl.cpp` at v1.0.0 (`b1b770d90`).

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `build` (nemotron-v2-vl.cpp:3 @b1b770d90), a member of `mtmd_graph_nemotron_v2_vl`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    mtmd.impl.ggmlAssert(model.class_embedding != null, "nemotron-v2-vl.cpp", 4, "model.class_embedding != nullptr");
    mtmd.impl.ggmlAssert(model.position_embeddings != null, "nemotron-v2-vl.cpp", 5, "model.position_embeddings != nullptr");

    const n_registers: c_int = @intCast(mtmd.graph.ne(model.class_embedding, 1));
    const n_pos = g.n_patches + n_registers;

    var inp = g.buildInp();

    // add position embeddings (pre-downsampled during GGUF conversion for fixed 512x512 input)
    inp = mtmd.impl.c.ggml_add(ctx0, inp, model.position_embeddings);
    g.cb(inp, "inp_pos", -1);

    inp = mtmd.impl.c.ggml_concat(ctx0, model.class_embedding, inp, 1);

    var cur = g.buildVit(inp, n_pos, .normal, g.hparams.ffn_op, null, null, .{});

    cur = mtmd.impl.c.ggml_view_2d(ctx0, cur, g.n_embd, g.n_patches, mtmd.impl.c.ggml_row_size(cur.?.type, g.n_embd), @as(usize, @intCast(n_registers)) * mtmd.impl.c.ggml_row_size(cur.?.type, g.n_embd));

    cur = g.buildPatchMergePermute(cur, g.hparams.n_merge);

    {
        cur = g.buildNorm(cur, model.mm_0_w, null, .rms, 1e-6, -1);
        cur = g.buildFfn(cur, model.mm_1_w, null, null, null, model.mm_3_w, null, .relu_sqr, -1);
    }

    mtmd.impl.c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
