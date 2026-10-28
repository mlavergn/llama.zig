//! The HunyuanVL encoder graph: a ViT with a CPU-filled position embedding,
//! then a perceiver projector wrapped in begin/end tokens.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/hunyuanvl.cpp` at v0.4.0 (`5266f24da`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const ne = graph.ne;

/// Ports `build` (hunyuanvl.cpp:3 @5266f24da), a member of `clip_graph_hunyuanvl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const merge = g.hparams.n_merge;
    const pw = g.n_patches_x;
    const ph = g.n_patches_y;

    // position embedding: declared as a graph input, filled on CPU
    // by clip_image_batch_encode (see PROJECTOR_TYPE_HUNYUANVL branch there).
    const pos_embd = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, g.n_embd, ph * pw);
    _ = c.ggml_set_name(pos_embd, "hunyuanvl_pos_embd");
    c.ggml_set_input(pos_embd);

    const inp = g.buildInp();
    var cur = g.buildVit(inp, g.n_patches, .normal, g.hparams.ffn_op, pos_embd, null, .{});

    // perceiver projector
    cur = g.buildNorm(cur, model.mm_pre_norm_w, null, .rms, g.eps, -1);

    // [C, W*H] -> [W, H, C] for conv2d
    cur = c.ggml_reshape_3d(ctx0, cur, g.n_embd, pw, ph);
    cur = c.ggml_permute(ctx0, cur, 2, 0, 1, 3);
    cur = c.ggml_cont(ctx0, cur);

    // Conv2d(1152->2304, k=2, s=2) + GELU + Conv2d(2304->4608, k=1, s=1)
    cur = c.ggml_conv_2d(ctx0, model.mm_0_w, cur, merge, merge, 0, 0, 1, 1);
    if (model.mm_0_b != null) {
        cur = c.ggml_add(ctx0, cur, c.ggml_reshape_3d(ctx0, model.mm_0_b, 1, 1, ne(model.mm_0_b, 0)));
    }
    cur = c.ggml_gelu(ctx0, cur);
    cur = c.ggml_conv_2d(ctx0, model.mm_1_w, cur, 1, 1, 0, 0, 1, 1);
    if (model.mm_1_b != null) {
        cur = c.ggml_add(ctx0, cur, c.ggml_reshape_3d(ctx0, model.mm_1_b, 1, 1, ne(model.mm_1_b, 0)));
    }

    const ow = @divTrunc(pw, merge);
    const oh = @divTrunc(ph, merge);
    const idim: c_int = @intCast(ne(cur, 2)); // OC = 4608

    // append newline along W (dim 0)
    var nl = c.ggml_reshape_4d(ctx0, model.image_newline, 1, 1, idim, 1);
    nl = c.ggml_repeat_4d(ctx0, nl, 1, oh, idim, 1);
    cur = c.ggml_concat(ctx0, cur, nl, 0);

    // [OW+1, OH, OC] -> [OC, (OW+1)*OH]
    cur = c.ggml_permute(ctx0, cur, 1, 2, 0, 3);
    cur = c.ggml_cont_2d(ctx0, cur, idim, (ow + 1) * oh);

    // project to LLM hidden size
    cur = g.buildMm(model.mm_model_proj, cur);
    if (model.mm_model_proj_b != null) cur = c.ggml_add(ctx0, cur, model.mm_model_proj_b);

    // wrap with begin/end tokens
    cur = c.ggml_concat(ctx0, c.ggml_reshape_2d(ctx0, model.mm_img_begin, ne(model.mm_img_begin, 0), 1), cur, 1);
    cur = c.ggml_concat(ctx0, cur, c.ggml_reshape_2d(ctx0, model.mm_img_end, ne(model.mm_img_end, 0), 1), 1);

    cur = g.buildNorm(cur, model.mm_post_norm_w, null, .rms, g.eps, -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
