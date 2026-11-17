//! The MageVL-Cowork vision encoder: a ViT whose patches are embedded
//! through an F32 im2col and regrouped into 2x2 blocks, with M-RoPE over
//! pairs that take their two angles from two halves of the model's
//! `rope_freqs`, then a 4-patch merge and an FFN projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/magevl.cpp` at v1.0.0 (`b1b770d90`),
//! new in 9f0e5098a. Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;
const T = graph.T;
const nb = graph.nb;

/// The `add_pos` lambda's captures.
const PosCtx = struct {
    positions: T,
    sections: [4]c_int,
    n_pair: c_int,
};

/// The `add_pos` lambda (magevl.cpp:17 @b1b770d90) of `build`: a pair uses
/// 2 angles, even outputs come from one rope, odd outputs from the other.
fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: T, layer: *const model_mod.Layer) T {
    _ = layer;
    const p: *PosCtx = @ptrCast(@alignCast(ctx.?));
    const ctx0 = g.ctx0;
    const rope_freqs = g.model.rope_freqs;
    const n_pair = p.n_pair;

    const freqs_a = c.ggml_view_1d(ctx0, rope_freqs, n_pair, 0);
    const freqs_b = c.ggml_view_1d(ctx0, rope_freqs, n_pair, c.ggml_row_size(rope_freqs.?.type, n_pair));

    const rope_a = c.ggml_rope_multi(ctx0, cur, p.positions, freqs_a, g.d_head, &p.sections, c.GGML_ROPE_TYPE_MROPE, 32768, 1.0, 1.0, 0.0, 1.0, 32.0, 1.0);
    const rope_b = c.ggml_rope_multi(ctx0, cur, p.positions, freqs_b, g.d_head, &p.sections, c.GGML_ROPE_TYPE_MROPE, 32768, 1.0, 1.0, 0.0, 1.0, 32.0, 1.0);

    const even = c.ggml_view_3d(ctx0, rope_a, n_pair, g.n_head, g.n_patches, nb(rope_a, 1), nb(rope_a, 2), 0);
    const odd = c.ggml_view_3d(ctx0, rope_b, n_pair, g.n_head, g.n_patches, nb(rope_b, 1), nb(rope_b, 2), c.ggml_row_size(rope_b.*.type, n_pair));

    return c.ggml_concat(ctx0, even, odd, 0);
}

/// Ports `build` (magevl.cpp:3 @b1b770d90), a member of `mtmd_graph_magevl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const n_patches = g.n_patches;

    impl.ggmlAssert(hparams.n_merge == 2, "magevl.cpp", 4, "hparams.n_merge == 2");

    const n_pair = @divTrunc(g.d_head, 2);
    const unit = @divTrunc(n_pair, 16);

    // pairs are grouped as t, h, w at conversion
    var pc: PosCtx = .{ .positions = null, .sections = .{ 4 * unit, 6 * unit, 6 * unit, 0 }, .n_pair = n_pair };

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_patches * 4);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);
    pc.positions = positions;

    // ggml_conv_2d rounds patches to F16
    const inp_raw = g.buildInpRaw(3);
    const patches = c.ggml_im2col(ctx0, model.patch_embeddings_0, inp_raw, g.patch_size, g.patch_size, 0, 0, 1, 1, true, c.GGML_TYPE_F32);
    const n_patch_dim = graph.ne(patches, 0);
    var inp = c.ggml_mul_mat(ctx0, c.ggml_reshape_2d(ctx0, model.patch_embeddings_0, n_patch_dim, n_embd), c.ggml_reshape_2d(ctx0, patches, n_patch_dim, n_patches));

    // reorder into 2x2 blocks
    inp = c.ggml_reshape_4d(ctx0, inp, n_embd * 2, @divTrunc(g.n_patches_x, 2), 2, @divTrunc(g.n_patches_y, 2));
    inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
    inp = c.ggml_cont_3d(ctx0, inp, n_embd, n_patches, 1);

    var cur = g.buildVit(inp, n_patches, .normal, hparams.ffn_op, null, .{ .ctx = &pc, .f = addPos }, .{});

    // multimodal projection
    cur = c.ggml_reshape_3d(ctx0, cur, n_embd * 4, @divTrunc(n_patches, 4), 1);
    cur = g.buildFfn(cur, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, hparams.ffn_op, -1);

    c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
