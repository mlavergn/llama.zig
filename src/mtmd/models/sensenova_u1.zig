//! The SenseNova U1 vision encoder: a patch embedding with adjacent-pair 2D
//! RoPE, downsampled by a strided convolution.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/sensenova-u1.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `build` (mtmd/models/sensenova-u1.cpp:3 @b1b770d90), a member of
/// `mtmd_graph_sensenova_u1`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_patches = g.n_patches;
    const n_embd = g.n_embd;
    const n_batch = g.n_batch;

    mtmd.impl.ggmlAssert(model.patch_embeddings_0 != null, "sensenova-u1.cpp", 4, "model.patch_embeddings_0 != nullptr");
    mtmd.impl.ggmlAssert(model.patch_bias != null, "sensenova-u1.cpp", 5, "model.patch_bias != nullptr");
    mtmd.impl.ggmlAssert(model.mm_0_w != null, "sensenova-u1.cpp", 6, "model.mm_0_w != nullptr");
    mtmd.impl.ggmlAssert(model.mm_0_b != null, "sensenova-u1.cpp", 7, "model.mm_0_b != nullptr");
    mtmd.impl.ggmlAssert(hparams.n_merge == 2, "sensenova-u1.cpp", 8, "hparams.n_merge == 2");

    const pos_x = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_x, "pos_x");
    mtmd.impl.c.ggml_set_input(pos_x);

    const pos_y = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_y, "pos_y");
    mtmd.impl.c.ggml_set_input(pos_y);

    var inp = g.buildInp();
    inp = mtmd.impl.c.ggml_gelu(ctx0, inp);

    // Apply adjacent-pair 2D RoPE to patch embeddings before downsampling.
    inp = mtmd.impl.c.ggml_reshape_4d(ctx0, inp, n_embd, 1, n_patches, n_batch);
    inp = g.buildRope2d(ctx0, inp, pos_x, pos_y, hparams.rope_theta, false);
    inp = mtmd.impl.c.ggml_reshape_3d(ctx0, inp, n_embd, n_patches, n_batch);
    g.cb(inp, "patch_embd_rope", -1);

    inp = mtmd.impl.c.ggml_reshape_4d(ctx0, inp, n_embd, g.n_patches_x, g.n_patches_y, n_batch);
    inp = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = mtmd.impl.c.ggml_conv_2d(ctx0, model.mm_0_w, inp, hparams.n_merge, hparams.n_merge, 0, 0, 1, 1);

    const n_output_patches = @divTrunc(n_patches, hparams.n_merge * hparams.n_merge);
    inp = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, inp, 1, 2, 0, 3));
    inp = mtmd.impl.c.ggml_reshape_3d(ctx0, inp, g.n_mmproj_embd, n_output_patches, n_batch);
    inp = mtmd.impl.c.ggml_add(ctx0, inp, model.mm_0_b);
    g.cb(inp, "proj_out", -1);

    mtmd.impl.c.ggml_build_forward_expand(g.gf, inp);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
