//! The Step3-VL vision encoder: a 2D-RoPE ViT with resized learned position
//! embeddings, two strided-conv downsamplers and a linear projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/step3vl.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const kimivl = @import("kimivl.zig");
const c = impl.c;

/// The `add_spatial_bias` lambda (step3vl.cpp:25 @b29c606e2): adds a
/// per-channel bias to a [w, h, c] tensor.
fn addSpatialBias(g: *graph.Graph, cur_in: graph.T, bias: graph.T) graph.T {
    if (bias == null) return cur_in;
    const ctx0 = g.ctx0;

    const width = graph.ne(cur_in, 0);
    const height = graph.ne(cur_in, 1);
    const channels = graph.ne(cur_in, 2);

    var cur = c.ggml_reshape_2d(ctx0, cur_in, width * height, channels);
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = c.ggml_add(ctx0, cur, bias);
    cur = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, cur));
    cur = c.ggml_reshape_3d(ctx0, cur, width, height, channels);

    return cur;
}

/// Ports `build` (step3vl.cpp:3 @b29c606e2), a member of `clip_graph_step3vl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    impl.ggmlAssert(model.class_embedding == null, "step3vl.cpp", 4, "model.class_embedding == nullptr");
    impl.ggmlAssert(model.patch_embeddings_0 != null, "step3vl.cpp", 5, "model.patch_embeddings_0 != nullptr");
    impl.ggmlAssert(model.position_embeddings != null, "step3vl.cpp", 6, "model.position_embeddings != nullptr");

    const norm_t: model_mod.NormType = .normal;

    const pos_h = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_h, "pos_h");
    c.ggml_set_input(pos_h);

    const pos_w = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_w, "pos_w");
    c.ggml_set_input(pos_w);

    const inp = g.buildInp();
    const learned_pos_embd = g.resizePositionEmbeddings(graph.default_interpolation_mode);

    var rope: kimivl.Rope2d = .{ .pos_a = pos_w, .pos_b = pos_h, .freq_base = hparams.rope_theta };

    var cur = g.buildVit(inp, g.n_patches, norm_t, hparams.ffn_op, learned_pos_embd, .{ .ctx = &rope, .f = kimivl.Rope2d.addPos }, .{});
    g.cb(cur, "vit_out", -1);

    // [n_embd, n_patches] -> [w, h, n_embd] for spatial downsampling convolutions.
    cur = c.ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = c.ggml_cont_3d(ctx0, cur, g.n_patches_x, g.n_patches_y, g.n_embd);

    // First downsampler: Conv2d(1536 -> 3072, k=3, s=2, p=1)
    cur = c.ggml_conv_2d(ctx0, model.mm_0_w, cur, 2, 2, 1, 1, 1, 1);
    cur = addSpatialBias(g, cur, model.mm_0_b);
    g.cb(cur, "downsample_0", -1);

    // Second downsampler: Conv2d(3072 -> 6144, k=3, s=2, p=1)
    cur = c.ggml_conv_2d(ctx0, model.mm_1_w, cur, 2, 2, 1, 1, 1, 1);
    cur = addSpatialBias(g, cur, model.mm_1_b);
    g.cb(cur, "downsample_1", -1);

    // [w, h, c] -> [c, w*h]
    {
        const w = graph.ne(cur, 0);
        const h = graph.ne(cur, 1);
        cur = c.ggml_reshape_3d(ctx0, cur, w * h, graph.ne(cur, 2), graph.ne(cur, 3));
        cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }
    g.cb(cur, "downsample_flatten", -1);

    // Final projector: Linear(6144 -> projection_dim)
    cur = c.ggml_mul_mat(ctx0, model.mm_model_proj, cur);
    g.cb(cur, "projector_out", -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
