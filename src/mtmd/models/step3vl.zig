//! The Step3-VL vision encoder: a 2D-RoPE ViT with resized learned position
//! embeddings, two strided-conv downsamplers and a linear projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/step3vl.cpp` at v1.0.0 (`b1b770d90`).

const std = @import("std");
const mtmd = @import("../module.zig");

/// The `add_spatial_bias` lambda (step3vl.cpp:25 @b1b770d90): adds a
/// per-channel bias to a [w, h, c] tensor.
fn addSpatialBias(g: *mtmd.graph.Graph, cur_in: mtmd.graph.T, bias: mtmd.graph.T) mtmd.graph.T {
    if (bias == null) return cur_in;
    const ctx0 = g.ctx0;

    const width = mtmd.graph.ne(cur_in, 0);
    const height = mtmd.graph.ne(cur_in, 1);
    const channels = mtmd.graph.ne(cur_in, 2);

    var cur = mtmd.impl.c.ggml_reshape_2d(ctx0, cur_in, width * height, channels);
    cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, cur));
    cur = mtmd.impl.c.ggml_add(ctx0, cur, bias);
    cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, cur));
    cur = mtmd.impl.c.ggml_reshape_3d(ctx0, cur, width, height, channels);

    return cur;
}

/// Ports `build` (step3vl.cpp:3 @b1b770d90), a member of `mtmd_graph_step3vl`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    mtmd.impl.ggmlAssert(model.class_embedding == null, "step3vl.cpp", 4, "model.class_embedding == nullptr");
    mtmd.impl.ggmlAssert(model.patch_embeddings_0 != null, "step3vl.cpp", 5, "model.patch_embeddings_0 != nullptr");
    mtmd.impl.ggmlAssert(model.position_embeddings != null, "step3vl.cpp", 6, "model.position_embeddings != nullptr");

    const norm_t: mtmd.model.NormType = .normal;

    const pos_h = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, g.n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_h, "pos_h");
    mtmd.impl.c.ggml_set_input(pos_h);

    const pos_w = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, g.n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_w, "pos_w");
    mtmd.impl.c.ggml_set_input(pos_w);

    const inp = g.buildInp();
    const learned_pos_embd = g.resizePositionEmbeddings(mtmd.graph.default_interpolation_mode);

    var rope: mtmd.models.kimivl.Rope2d = .{ .pos_a = pos_w, .pos_b = pos_h, .freq_base = hparams.rope_theta };

    var cur = g.buildVit(inp, g.n_patches, norm_t, hparams.ffn_op, learned_pos_embd, .{ .ctx = &rope, .f = mtmd.models.kimivl.Rope2d.addPos }, .{});
    g.cb(cur, "vit_out", -1);

    // [n_embd, n_patches] -> [w, h, n_embd] for spatial downsampling convolutions.
    cur = mtmd.impl.c.ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = mtmd.impl.c.ggml_cont_3d(ctx0, cur, g.n_patches_x, g.n_patches_y, g.n_embd);

    // First downsampler: Conv2d(1536 -> 3072, k=3, s=2, p=1)
    cur = mtmd.impl.c.ggml_conv_2d(ctx0, model.mm_0_w, cur, 2, 2, 1, 1, 1, 1);
    cur = addSpatialBias(g, cur, model.mm_0_b);
    g.cb(cur, "downsample_0", -1);

    // Second downsampler: Conv2d(3072 -> 6144, k=3, s=2, p=1)
    cur = mtmd.impl.c.ggml_conv_2d(ctx0, model.mm_1_w, cur, 2, 2, 1, 1, 1, 1);
    cur = addSpatialBias(g, cur, model.mm_1_b);
    g.cb(cur, "downsample_1", -1);

    // [w, h, c] -> [c, w*h]
    {
        const w = mtmd.graph.ne(cur, 0);
        const h = mtmd.graph.ne(cur, 1);
        cur = mtmd.impl.c.ggml_reshape_3d(ctx0, cur, w * h, mtmd.graph.ne(cur, 2), mtmd.graph.ne(cur, 3));
        cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }
    g.cb(cur, "downsample_flatten", -1);

    // Final projector: Linear(6144 -> projection_dim)
    cur = mtmd.impl.c.ggml_mul_mat(ctx0, model.mm_model_proj, cur);
    g.cb(cur, "projector_out", -1);

    mtmd.impl.c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
