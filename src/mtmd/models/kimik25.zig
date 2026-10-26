//! The Kimi-K2.5 vision encoder: Kimi-VL's ViT and projector over a 3-D
//! learned position-embedding grid resized bicubically.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/kimik25.cpp` at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const kimivl = @import("kimivl.zig");
const c = impl.c;

/// Ports `resize_position_embeddings_3d` (kimik25.cpp:8 @c1d0e7a00), a
/// member of `clip_graph_kimik25`: `resize_position_embeddings` with w and
/// h stored in ne[1] and ne[2].
pub fn resizePositionEmbeddings3d(g: *graph.Graph, interpolation_mode: u32) graph.T {
    const ctx0 = g.ctx0;
    const n_embd = g.n_embd;
    var pos_embd = g.model.position_embeddings;
    const height = @divTrunc(g.img.ny(), g.patch_size);
    const width = @divTrunc(g.img.nx(), g.patch_size);
    const mode = interpolation_mode;

    impl.ggmlAssert(pos_embd != null, "kimik25.cpp", 14, "pos_embd");

    const stored_c = graph.ne(pos_embd, 0); // C = 1152
    const orig_w = graph.ne(pos_embd, 1); // W = 64
    const orig_h = graph.ne(pos_embd, 2); // H = 64

    impl.ggmlAssert(stored_c == n_embd, "kimik25.cpp", 20, "stored_c == n_embd");

    if (height == @as(c_int, @truncate(orig_h)) and width == @as(c_int, @truncate(orig_w))) {
        // No interpolation needed, just flatten to [C, H*W]
        return c.ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    }

    pos_embd = c.ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = c.ggml_interpolate(ctx0, pos_embd, height, width, n_embd, 1, mode);
    pos_embd = c.ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = c.ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    return pos_embd;
}

/// Ports `build` (kimik25.cpp:34 @c1d0e7a00), a member of `clip_graph_kimik25`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    const pos_h = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_h, "pos_h");
    c.ggml_set_input(pos_h);

    const pos_w = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_w, "pos_w");
    c.ggml_set_input(pos_w);

    const learned_pos_embd = resizePositionEmbeddings3d(g, c.GGML_SCALE_MODE_BICUBIC);

    // Kimi-K2.5 uses interleaved 2D RoPE pattern natively, but
    // Q / K are permuted during conversion to use split format.
    var rope: kimivl.Rope2d = .{ .pos_a = pos_w, .pos_b = pos_h, .freq_base = hparams.rope_theta };

    var inp = g.buildInp();

    // added here rather than through build_vit, as upstream does
    inp = c.ggml_add(ctx0, inp, learned_pos_embd);

    var cur = g.buildVit(inp, g.n_patches, .normal, hparams.ffn_op, null, .{ .ctx = &rope, .f = kimivl.Rope2d.addPos }, .{});

    g.cb(cur, "vit_out", -1);

    {
        // patch_merger
        const scale_factor = model.hparams.n_merge;
        cur = g.buildPatchMergePermute(cur, scale_factor);

        // projection norm
        const proj_inp_dim: c_int = @truncate(graph.ne(cur, 0));
        const n_merged_patches: c_int = @truncate(graph.ne(cur, 1));
        cur = c.ggml_view_2d(ctx0, cur, g.n_embd, n_merged_patches * scale_factor * scale_factor, c.ggml_row_size(cur.?.type, g.n_embd), 0);
        cur = c.ggml_norm(ctx0, cur, hparams.eps);
        cur = c.ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = c.ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = c.ggml_view_2d(ctx0, cur, proj_inp_dim, n_merged_patches, c.ggml_row_size(cur.?.type, proj_inp_dim), 0);
        g.cb(cur, "proj_inp_normed", -1);

        // projection mlp
        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu, -1);

        g.cb(cur, "proj_out", -1);
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
