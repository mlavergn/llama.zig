//! The Kimi-K2.5 and Kimi-K3.1 vision encoders: Kimi-VL's ViT and projector
//! over a 3-D learned position-embedding grid, resized bicubically for K2.5
//! and bilinearly for K3.1, whose ViT is RMS-normed and whose projector
//! (patchmergerv2) has no pre-norm and an RMS norm after the MLP.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/kimik25.cpp` at v1.0.0 (`b1b770d90`).

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `resize_position_embeddings_3d` (kimik25.cpp:8 @b1b770d90), a
/// member of `mtmd_graph_kimik25`: `resize_position_embeddings` with w and
/// h stored in ne[1] and ne[2].
pub fn resizePositionEmbeddings3d(g: *mtmd.graph.Graph, interpolation_mode: u32) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const n_embd = g.n_embd;
    var pos_embd = g.model.position_embeddings;
    const height = @divTrunc(g.img.ny(), g.patch_size);
    const width = @divTrunc(g.img.nx(), g.patch_size);
    const mode = interpolation_mode;

    mtmd.impl.ggmlAssert(pos_embd != null, "kimik25.cpp", 14, "pos_embd");

    const stored_c = mtmd.graph.ne(pos_embd, 0); // C = 1152
    const orig_w = mtmd.graph.ne(pos_embd, 1); // W = 64
    const orig_h = mtmd.graph.ne(pos_embd, 2); // H = 64

    mtmd.impl.ggmlAssert(stored_c == n_embd, "kimik25.cpp", 20, "stored_c == n_embd");

    if (height == @as(c_int, @truncate(orig_h)) and width == @as(c_int, @truncate(orig_w))) {
        // No interpolation needed, just flatten to [C, H*W]
        return mtmd.impl.c.ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    }

    pos_embd = mtmd.impl.c.ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = mtmd.impl.c.ggml_interpolate(ctx0, pos_embd, height, width, n_embd, 1, mode);
    pos_embd = mtmd.impl.c.ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = mtmd.impl.c.ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    return pos_embd;
}

/// Ports `build` (kimik25.cpp:34 @b1b770d90), a member of `mtmd_graph_kimik25`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    const pos_h = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, g.n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_h, "pos_h");
    mtmd.impl.c.ggml_set_input(pos_h);

    const pos_w = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, g.n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_w, "pos_w");
    mtmd.impl.c.ggml_set_input(pos_w);

    const is_k31 = g.proj_type == .kimik31;

    const learned_pos_embd = resizePositionEmbeddings3d(g, if (is_k31) mtmd.impl.c.GGML_SCALE_MODE_BILINEAR else mtmd.impl.c.GGML_SCALE_MODE_BICUBIC);

    // Kimi-K2.5 uses interleaved 2D RoPE pattern natively, but
    // Q / K are permuted during conversion to use split format.
    var rope: mtmd.models.kimivl.Rope2d = .{ .pos_a = pos_w, .pos_b = pos_h, .freq_base = hparams.rope_theta };

    var inp = g.buildInp();

    // added here rather than through build_vit, as upstream does
    inp = mtmd.impl.c.ggml_add(ctx0, inp, learned_pos_embd);

    var cur = g.buildVit(inp, g.n_patches, if (is_k31) .rms else .normal, hparams.ffn_op, null, .{ .ctx = &rope, .f = mtmd.models.kimivl.Rope2d.addPos }, .{});

    g.cb(cur, "vit_out", -1);

    if (is_k31) {
        // patchmergerv2: no pre-norm, RMS norm after the MLP
        cur = g.buildPatchMergePermute(cur, hparams.n_merge);
        cur = g.buildFfn(cur, model.mm_1_w, null, null, null, model.mm_2_w, null, .gelu_erf, -1);
        cur = g.buildNorm(cur, model.mm_post_norm_w, null, .rms, 1e-5, -1);
        g.cb(cur, "proj_out", -1);
    } else {
        // patch_merger
        const scale_factor = model.hparams.n_merge;
        cur = g.buildPatchMergePermute(cur, scale_factor);

        // projection norm
        const proj_inp_dim: c_int = @truncate(mtmd.graph.ne(cur, 0));
        const n_merged_patches: c_int = @truncate(mtmd.graph.ne(cur, 1));
        cur = mtmd.impl.c.ggml_view_2d(ctx0, cur, g.n_embd, n_merged_patches * scale_factor * scale_factor, mtmd.impl.c.ggml_row_size(cur.?.type, g.n_embd), 0);
        cur = mtmd.impl.c.ggml_norm(ctx0, cur, hparams.eps);
        cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = mtmd.impl.c.ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = mtmd.impl.c.ggml_view_2d(ctx0, cur, proj_inp_dim, n_merged_patches, mtmd.impl.c.ggml_row_size(cur.?.type, proj_inp_dim), 0);
        g.cb(cur, "proj_inp_normed", -1);

        // projection mlp
        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);

        g.cb(cur, "proj_out", -1);
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
