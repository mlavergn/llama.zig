//! The Llama 4 vision encoder: an unfold-convolution patch embedding, a
//! 2D-RoPE ViT with a CLS token, a pixel shuffle and the adapter MLP.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/llama4.cpp` at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const kimivl = @import("kimivl.zig");
const c = impl.c;

/// Ports `build` (mtmd/models/llama4.cpp:3 @c1d0e7a00), a member of `clip_graph_llama4`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const patch_size = g.patch_size;
    impl.ggmlAssert(model.class_embedding != null, "llama4.cpp", 4, "model.class_embedding != nullptr");
    impl.ggmlAssert(model.position_embeddings != null, "llama4.cpp", 5, "model.position_embeddings != nullptr");

    const n_pos = g.n_patches + 1; // +1 for [CLS]

    // 2D input positions
    const pos_h = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos);
    _ = c.ggml_set_name(pos_h, "pos_h");
    c.ggml_set_input(pos_h);

    const pos_w = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos);
    _ = c.ggml_set_name(pos_w, "pos_w");
    c.ggml_set_input(pos_w);

    var inp = g.buildInpRaw(3);

    // Llama4UnfoldConvolution
    {
        const kernel = c.ggml_reshape_4d(ctx0, model.patch_embeddings_0, patch_size, patch_size, 3, n_embd);
        inp = c.ggml_im2col(ctx0, kernel, inp, patch_size, patch_size, 0, 0, 1, 1, true, inp.?.type);
        inp = g.buildMm(model.patch_embeddings_0, inp);
        inp = c.ggml_reshape_2d(ctx0, inp, n_embd, g.n_patches);
        g.cb(inp, "patch_conv", -1);
    }

    // add CLS token
    inp = c.ggml_concat(ctx0, inp, model.class_embedding, 1);

    // build ViT with 2D position embeddings; the first half is X axis and the second half is Y axis
    var rope: kimivl.Rope2d = .{ .pos_a = pos_w, .pos_b = pos_h, .freq_base = hparams.rope_theta };
    var cur = g.buildVit(inp, n_pos, .normal, hparams.ffn_op, model.position_embeddings, .{ .ctx = &rope, .f = kimivl.Rope2d.addPos }, .{});

    // remove CLS token
    cur = c.ggml_view_2d(ctx0, cur, n_embd, g.n_patches, c.ggml_row_size(cur.?.type, n_embd), 0);

    // pixel shuffle, based on Llama4VisionPixelShuffleMLP
    {
        const scale_factor = model.hparams.n_merge;
        const bsz = 1; // batch size, always 1 for now since we don't support batching
        impl.ggmlAssert(scale_factor > 0, "llama4.cpp", 58, "scale_factor > 0");
        impl.ggmlAssert(g.n_patches_x == g.n_patches_y, "llama4.cpp", 59, "n_patches_x == n_patches_y"); // llama4 only supports square images
        cur = c.ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, @divTrunc(g.n_patches_x, scale_factor), g.n_patches_y, bsz);
        cur = c.ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = c.ggml_cont_4d(ctx0, cur, n_embd * scale_factor * scale_factor, @divTrunc(g.n_patches_x, scale_factor), @divTrunc(g.n_patches_y, scale_factor), bsz);
        // flatten to 2D
        cur = c.ggml_cont_2d(ctx0, cur, n_embd * scale_factor * scale_factor, @divTrunc(@divTrunc(g.n_patches, scale_factor), scale_factor));
        g.cb(cur, "pixel_shuffle", -1);
    }

    // based on Llama4VisionMLP2 (always uses GELU activation, no bias)
    {
        cur = g.buildMm(model.mm_model_mlp_1_w, cur);
        cur = c.ggml_gelu(ctx0, cur);
        cur = g.buildMm(model.mm_model_mlp_2_w, cur);
        cur = c.ggml_gelu(ctx0, cur);
        g.cb(cur, "adapter_mlp", -1);
    }

    // Llama4MultiModalProjector
    cur = g.buildMm(model.mm_model_proj, cur);
    g.cb(cur, "projected", -1);

    // build the graph
    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
