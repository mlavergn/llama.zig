//! The Pixtral vision encoder (also LightOnOCR): a 2D-RoPE ViT, Mistral
//! Small 3.1's patch merger, a GELU projector and the [IMG_BREAK] rows.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/pixtral.cpp` at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const kimivl = @import("kimivl.zig");
const c = impl.c;

/// Ports `build` (pixtral.cpp:3 @c1d0e7a00), a member of `clip_graph_pixtral`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_merge = hparams.n_merge;

    // 2D input positions
    const pos_h = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_h, "pos_h");
    c.ggml_set_input(pos_h);

    const pos_w = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_w, "pos_w");
    c.ggml_set_input(pos_w);

    var rope: kimivl.Rope2d = .{ .pos_a = pos_h, .pos_b = pos_w, .freq_base = hparams.rope_theta, .interleave_freq = true };

    const inp = g.buildInp();
    var cur = g.buildVit(inp, g.n_patches, .rms, hparams.ffn_op, null, .{ .ctx = &rope, .f = kimivl.Rope2d.addPos }, .{});

    // mistral small 3.1 patch merger
    if (model.mm_patch_merger_w != null) {
        impl.ggmlAssert(hparams.n_merge > 0, "pixtral.cpp", 30, "hparams.n_merge > 0");

        cur = c.ggml_mul(ctx0, c.ggml_rms_norm(ctx0, cur, g.eps), model.mm_input_norm_w);

        // reshape image tokens to 2D grid
        cur = c.ggml_reshape_3d(ctx0, cur, g.n_embd, g.n_patches_x, g.n_patches_y);
        cur = c.ggml_permute(ctx0, cur, 2, 0, 1, 3); // [x, y, n_embd]
        cur = c.ggml_cont(ctx0, cur);

        // torch.nn.functional.unfold is just an im2col under the hood
        // we just need a dummy kernel to make it work
        const kernel = c.ggml_view_3d(ctx0, cur, n_merge, n_merge, graph.ne(cur, 2), 0, 0, 0);
        cur = c.ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp.?.type);

        // project to n_embd
        cur = c.ggml_reshape_2d(ctx0, cur, graph.ne(cur, 0), graph.ne(cur, 1) * graph.ne(cur, 2));
        cur = g.buildMm(model.mm_patch_merger_w, cur);
    }

    // LlavaMultiModalProjector (always using GELU activation)
    cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu, -1);

    // arrangement of the [IMG_BREAK] token
    if (model.token_embd_img_break != null) {
        // view the embeddings as [n_embd, n_patches_per_row, n_rows] and append
        // the [IMG_BREAK] token to each row
        const p_y = @divTrunc(g.n_patches_y, n_merge);
        const p_x = @divTrunc(g.n_patches_x, n_merge);
        const p_total = p_x * p_y;
        const n_embd_text: c_int = @intCast(graph.ne(cur, 0));
        const n_tokens_output = p_total + p_y - 1; // one [IMG_BREAK] per row, except the last row

        var tmp = c.ggml_reshape_3d(ctx0, cur, n_embd_text, p_x, p_y);
        var tok = c.ggml_new_tensor_3d(ctx0, tmp.*.type, n_embd_text, 1, p_y);
        tok = c.ggml_scale(ctx0, tok, 0.0); // clear the tensor
        tok = c.ggml_add(ctx0, tok, model.token_embd_img_break);
        tmp = c.ggml_concat(ctx0, tmp, tok, 1);
        cur = c.ggml_view_2d(ctx0, tmp, n_embd_text, n_tokens_output, c.ggml_row_size(tmp.*.type, n_embd_text), 0);
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
