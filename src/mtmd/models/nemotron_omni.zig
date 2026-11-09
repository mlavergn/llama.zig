//! The Nemotron Omni vision encoder (Nemotron Super 3.5 VL, Nemotron Bolt):
//! Nemotron v2 VL's ViT with register tokens, a temporally merged patch
//! embedding for video, bilinear-resized position embeddings, a pixel shuffle
//! and a squared-ReLU projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/nemotron-omni.cpp` at v0.8.0 (`8642541b2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;

/// Ports `build_inp_with_temporal_merge` (nemotron-omni.cpp:3 @cd959bae9), a
/// member of `clip_graph_nemotron_omni`: one Conv2D over two frames stacked
/// on the channel axis, a still image going through the plain patch embedding.
pub fn buildInpWithTemporalMerge(g: *graph.Graph) graph.T {
    const ctx0 = g.ctx0;
    const model = g.model;
    if (g.n_batch == 1 and !g.img_batch.?.is_video) {
        return g.buildInp();
    }

    impl.ggmlAssert(g.n_batch == 1 or g.n_batch == 2, "nemotron-omni.cpp", 8, "n_batch == 1 || n_batch == 2");
    impl.ggmlAssert(model.patch_embeddings_1 != null, "nemotron-omni.cpp", 9, "model.patch_embeddings_1 != nullptr");
    impl.ggmlAssert(graph.ne(model.patch_embeddings_1, 2) == 6, "nemotron-omni.cpp", 10, "model.patch_embeddings_1->ne[2] == 6");

    var inp_raw = g.buildInpRaw(3);
    if (g.n_batch == 1) {
        // pad the last video frame inside the graph
        inp_raw = c.ggml_concat(ctx0, inp_raw, inp_raw, 2);
    } else {
        inp_raw = c.ggml_reshape_4d(ctx0, inp_raw, g.img.nx(), g.img.ny(), 6, 1);
    }

    var inp = c.ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, g.patch_size, g.patch_size, 0, 0, 1, 1);
    inp = c.ggml_reshape_3d(ctx0, inp, g.n_patches, g.n_embd, 1);
    return c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp));
}

/// Ports `build` (nemotron-omni.cpp:25 @cd959bae9), a member of `clip_graph_nemotron_omni`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    impl.ggmlAssert(model.class_embedding != null, "nemotron-omni.cpp", 26, "model.class_embedding != nullptr");
    impl.ggmlAssert(model.position_embeddings != null, "nemotron-omni.cpp", 27, "model.position_embeddings != nullptr");

    const n_registers: c_int = @intCast(graph.ne(model.class_embedding, 1));
    const n_pos = g.n_patches + n_registers;

    var inp = buildInpWithTemporalMerge(g);

    inp = c.ggml_add(ctx0, inp, g.resizePositionEmbeddings(c.GGML_SCALE_MODE_BILINEAR));
    g.cb(inp, "inp_pos", -1);

    inp = c.ggml_concat(ctx0, model.class_embedding, inp, 1);

    var cur = g.buildVit(inp, n_pos, .normal, g.hparams.ffn_op, null, null, .{});

    cur = c.ggml_view_2d(ctx0, cur, g.n_embd, g.n_patches, c.ggml_row_size(cur.?.type, g.n_embd), @as(usize, @intCast(n_registers)) * c.ggml_row_size(cur.?.type, g.n_embd));

    cur = g.buildPatchMergePermute(cur, g.hparams.n_merge);

    {
        cur = g.buildNorm(cur, model.mm_0_w, null, .rms, 1e-5, -1);
        cur = g.buildFfn(cur, model.mm_1_w, null, null, null, model.mm_3_w, null, .relu_sqr, -1);
    }

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
