//! The Ling 3.0 VL vision encoder: Qwen3-VL's tower with a norm-only merger
//! and the projector MLP at the top level.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/ling3vl.cpp`, new at v0.9.0
//! (`cd959bae9`), and the `clip_graph_ling3vl` declaration in
//! `llama.cpp/tools/mtmd/models/models.h`. Each declaration names what it
//! ports and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const qwen2vl = @import("qwen2vl.zig");
const c = impl.c;
const T = graph.T;

/// The `add_pos` lambda's captures.
const PosCtx = struct {
    positions: T,
    mrope_sections: [4]c_int,
};

/// The M-RoPE lambda `build` passes to `build_vit` (ling3vl.cpp:58 @cd959bae9).
fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: T, layer: *const model_mod.Layer) T {
    _ = layer;
    const p: *PosCtx = @ptrCast(@alignCast(ctx.?));
    return c.ggml_rope_multi(g.ctx0, cur, p.positions, null, @divTrunc(g.d_head, 2), &p.mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
}

/// Ports `build` (ling3vl.cpp:3 @cd959bae9), a member of `clip_graph_ling3vl`
/// (mtmd/models/models.h:79 @cd959bae9), which derives from `clip_graph_qwen3vl` for
/// `build_inp_temporal_merge` alone.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const d_head = g.d_head;
    // same vision tower as qwen3vl, but the merger is norm-only (no fc1/fc2) and
    // the projector MLP lives at the top level (mm.0 / mm.2)
    impl.ggmlAssert(model.class_embedding == null, "ling3vl.cpp", 6, "model.class_embedding == nullptr");
    impl.ggmlAssert(model.mm_input_norm_w != null, "ling3vl.cpp", 7, "model.mm_input_norm_w != nullptr"); // merger norm (pre spatial merge)

    const batch_size: c_int = 1;
    const n_pos = g.n_patches;

    const norm_t: model_mod.NormType = .normal;

    // vision M-RoPE, same layout as qwen3vl: [row, col, row, col] quarters
    var pc: PosCtx = .{ .positions = null, .mrope_sections = .{ @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4), @divTrunc(d_head, 4) } };

    var inp = g.buildInpTemporalMerge();

    // spatial merge
    {
        inp = c.ggml_permute(ctx0, inp, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
        inp = c.ggml_cont_4d(ctx0, inp, n_embd * 2, @divTrunc(g.n_patches_x, 2), g.n_patches_y, batch_size);
        inp = c.ggml_reshape_4d(ctx0, inp, n_embd * 2, @divTrunc(g.n_patches_x, 2), 2, batch_size * @divTrunc(g.n_patches_y, 2));
        inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = c.ggml_cont_3d(ctx0, inp, n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    }

    // add patch bias
    if (model.patch_bias != null) {
        inp = c.ggml_add(ctx0, inp, model.patch_bias);
        g.cb(inp, "patch_bias", -1);
    }

    // calculate absolute position embedding and apply
    var learned_pos_embd = g.resizePositionEmbeddings(c.GGML_SCALE_MODE_BILINEAR | c.GGML_SCALE_FLAG_ALIGN_CORNERS);
    learned_pos_embd = c.ggml_cont_4d(ctx0, learned_pos_embd, n_embd * 2, @divTrunc(g.n_patches_x, 2), g.n_patches_y, batch_size);
    learned_pos_embd = c.ggml_reshape_4d(ctx0, learned_pos_embd, n_embd * 2, @divTrunc(g.n_patches_x, 2), 2, batch_size * @divTrunc(g.n_patches_y, 2));
    learned_pos_embd = c.ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = c.ggml_cont_3d(ctx0, learned_pos_embd, n_embd, g.n_patches_x * g.n_patches_y, batch_size);

    const num_position_ids = n_pos * 4; // m-rope requires 4 dim per position
    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, num_position_ids);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);
    pc.positions = positions;

    const inpL = g.buildVit(inp, n_pos, norm_t, hparams.ffn_op, learned_pos_embd, .{ .ctx = &pc, .f = addPos }, .{});

    // multimodal projection (linear_proj MLP over the merged patches)
    var embeddings = inpL;

    // per-patch merger norm, applied post-blocks before the 2x2 merge
    // (merger.norm, LayerNorm over n_embd)
    embeddings = g.buildNorm(embeddings, model.mm_input_norm_w, model.mm_input_norm_b, norm_t, g.eps, -1);
    g.cb(embeddings, "merger_norm", -1);

    embeddings = c.ggml_reshape_3d(ctx0, embeddings, n_embd * 4, @divTrunc(n_pos, 4), batch_size);

    embeddings = g.buildFfn(embeddings, model.mm_0_w, model.mm_0_b, null, null, model.mm_1_w, model.mm_1_b, .gelu, -1);

    // build the graph
    c.ggml_build_forward_expand(g.gf, embeddings);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
