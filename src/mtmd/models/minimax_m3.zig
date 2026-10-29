//! The MiniMax-M3 vision encoder: a temporally merged patch embedding, a
//! ViT whose heads rotate h and w on two NEOX thirds, and a two-stage
//! projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/minimax-m3.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

/// Ports `apply_rope` (mtmd/models/minimax-m3.cpp:3 @b29c606e2), a member of
/// `clip_graph_minimax_m3`. The head is laid out [t, h, w, pad]: t and pad
/// stay unrotated.
pub fn applyRope(g: *graph.Graph, x_in: graph.T, pos_h: graph.T, pos_w: graph.T) graph.T {
    const ctx0 = g.ctx0;
    const dh: c_int = @intCast(graph.ne(x_in, 0));
    const axd = 2 * @divTrunc(@divTrunc(2 * @divTrunc(dh, 2), 3), 2);

    impl.ggmlAssert(3 * axd <= dh, "minimax-m3.cpp", 8, "3 * axd <= dh");

    const th = g.hparams.rope_theta;

    var x = c.ggml_rope_ext(ctx0, x_in, pos_h, null, axd, c.GGML_ROPE_TYPE_NEOX, 0, th, 1.0, 0.0, 1.0, 0.0, 0.0);
    x = c.ggml_rope_set_offset(x, axd);

    x = c.ggml_rope_ext(ctx0, x, pos_w, null, axd, c.GGML_ROPE_TYPE_NEOX, 0, th, 1.0, 0.0, 1.0, 0.0, 0.0);
    x = c.ggml_rope_set_offset(x, 2 * axd);

    return x;
}

/// The `add_pos` lambda passed to `build_vit` (mtmd/models/minimax-m3.cpp:55 @b29c606e2),
/// the call it is an argument of.
const AddPos = struct {
    pos_h: graph.T,
    pos_w: graph.T,

    fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: graph.T, layer: *const model_mod.Layer) graph.T {
        _ = layer;
        const self: *AddPos = @ptrCast(@alignCast(ctx.?));
        return applyRope(g, cur, self.pos_h, self.pos_w);
    }
};

/// Ports `build` (mtmd/models/minimax-m3.cpp:23 @b29c606e2), a member of `clip_graph_minimax_m3`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const patch_size = g.patch_size;
    impl.ggmlAssert(model.patch_bias == null, "minimax-m3.cpp", 24, "model.patch_bias     == nullptr");
    impl.ggmlAssert(model.class_embedding == null, "minimax-m3.cpp", 25, "model.class_embedding == nullptr");
    impl.ggmlAssert(model.patch_embeddings_0 != null and model.patch_embeddings_1 != null, "minimax-m3.cpp", 26, "model.patch_embeddings_0 && model.patch_embeddings_1");
    impl.ggmlAssert(model.mm_1_w != null and model.mm_2_w != null, "minimax-m3.cpp", 27, "model.mm_1_w && model.mm_2_w");
    impl.ggmlAssert(model.mm_merger_fc1_w != null and model.mm_merger_fc2_w != null, "minimax-m3.cpp", 28, "model.mm_merger_fc1_w && model.mm_merger_fc2_w");

    const batch_size = 1;
    const n_pos = g.n_patches;
    const merge = hparams.n_merge;

    // patch embedding
    const inp_raw = g.buildInpRaw(3);
    var inp = c.ggml_add(ctx0, c.ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1), c.ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));

    // spatial merge
    {
        inp = c.ggml_permute(ctx0, inp, 1, 2, 0, 3);
        inp = c.ggml_cont_4d(ctx0, inp, n_embd * merge, @divTrunc(g.n_patches_x, merge), g.n_patches_y, batch_size);
        inp = c.ggml_reshape_4d(ctx0, inp, n_embd * merge, @divTrunc(g.n_patches_x, merge), merge, batch_size * @divTrunc(g.n_patches_y, merge));
        inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = c.ggml_cont_3d(ctx0, inp, n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    }

    // t (time axis) is always 0 for now, so we leave it unrotated
    const pos_h = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos);
    _ = c.ggml_set_name(pos_h, "minimax_pos_h");
    c.ggml_set_input(pos_h);
    const pos_w = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos);
    _ = c.ggml_set_name(pos_w, "minimax_pos_w");
    c.ggml_set_input(pos_w);

    var rope: AddPos = .{ .pos_h = pos_h, .pos_w = pos_w };
    const inpL = g.buildVit(inp, n_pos, .normal, .gelu_erf, null, .{ .ctx = &rope, .f = AddPos.addPos }, .{});

    // projector
    var emb = inpL;
    emb = g.buildFfn(emb, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);

    const proj = graph.ne(emb, 0);
    emb = c.ggml_reshape_2d(ctx0, emb, proj * merge * merge, @divTrunc(n_pos, merge * merge));

    emb = g.buildFfn(emb, model.mm_merger_fc1_w, model.mm_merger_fc1_b, null, null, model.mm_merger_fc2_w, model.mm_merger_fc2_b, .gelu_erf, -1);

    c.ggml_build_forward_expand(g.gf, emb);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
