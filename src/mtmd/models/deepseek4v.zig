//! The DeepSeek-V4-Flash-Vision encoder (deepseek4v): a native-resolution
//! ViT (RMSNorm, SwiGLU, 2D RoPE, no CLS or learned position embedding),
//! then the "aligner" -- a 3x3 patch merge (`torch.nn.functional.unfold`)
//! and a two-layer GELU MLP -- and the LLM token block assembled from the
//! aligner output and four learned sentinel embeddings:
//!
//!     [PAD]*lead_pad [START] <interleaved rows> [PAD]*pad_last [END]
//!
//! Each aligner row ends with a NEWLINE, an odd row count is padded with a
//! full row of PADs, and pairs of adjacent rows are interleaved column-wise
//! ("N-layout"). The mapping is precomputed on the CPU as the `layout_idx`
//! input (`arms_inputs.zig`, the `deepseek4v` arm of `clip_encode`).
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/deepseek4v.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

/// The `add_pos` lambda (deepseek4v.cpp:27 @b29c606e2): a two-section
/// vision M-RoPE over the `positions` input, rows then columns.
const Rope = struct {
    positions: graph.T,
    sections: [4]c_int,
    rope_theta: f32,

    fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: graph.T, layer: *const model_mod.Layer) graph.T {
        _ = layer;
        const self: *Rope = @ptrCast(@alignCast(ctx.?));
        return c.ggml_rope_multi(g.ctx0, cur, self.positions, null, @divTrunc(g.d_head, 2), &self.sections, c.GGML_ROPE_TYPE_VISION, 0, self.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
    }
};

/// `CLIP_ALIGN` (clip-impl.h:438 @b29c606e2), over `int`.
fn clipAlign(x: c_int, n: c_int) c_int {
    return @divTrunc(x + n - 1, n) * n;
}

/// Ports `build` (deepseek4v.cpp:18 @b29c606e2), a member of
/// `clip_graph_deepseek4v`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_merge = hparams.n_merge;

    // 2D input positions
    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches * 4);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);

    const q = @divTrunc(g.d_head, 4);
    var rope: Rope = .{ .positions = positions, .sections = .{ q, q, 0, 0 }, .rope_theta = hparams.rope_theta };

    const inp = g.buildInp();
    var cur = g.buildVit(inp, g.n_patches, .rms, hparams.ffn_op, null, // no learned pos embd
        .{ .ctx = &rope, .f = Rope.addPos }, .{});
    g.cb(cur, "vit_out", -1);

    // aligner patch merge: zero-pad the patch grid to a multiple of n_merge
    // then F.unfold == im2col with a dummy kernel (same trick as pixtral)
    {
        cur = c.ggml_reshape_3d(ctx0, cur, g.n_embd, g.n_patches_x, g.n_patches_y);
        cur = c.ggml_permute(ctx0, cur, 2, 0, 1, 3); // [x, y, n_embd]
        cur = c.ggml_cont(ctx0, cur);

        const pad_x = @rem(n_merge - @rem(g.n_patches_x, n_merge), n_merge);
        const pad_y = @rem(n_merge - @rem(g.n_patches_y, n_merge), n_merge);
        if (pad_x != 0 or pad_y != 0) {
            cur = c.ggml_pad(ctx0, cur, pad_x, pad_y, 0, 0);
        }

        const kernel = c.ggml_view_3d(ctx0, cur, n_merge, n_merge, graph.ne(cur, 2), 0, 0, 0);
        cur = c.ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp.?.type);
        cur = c.ggml_reshape_2d(ctx0, cur, graph.ne(cur, 0), graph.ne(cur, 1) * graph.ne(cur, 2));

        // aligner MLP (F.gelu in the reference == erf-based gelu)
        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
        g.cb(cur, "aligner_out", -1);
    }

    // assemble the token block: append the sentinel embeddings as extra rows
    // then reorder everything with the precomputed layout index
    {
        const n_embd_out = graph.ne(cur, 0);
        const n_grid = graph.ne(cur, 1); // n_llm_w * n_llm_h

        // rows n_grid + 0..3, keep in sync with the index computation in set_input
        const sentinels = [_]graph.T{
            model.token_embd_img_start,
            model.token_embd_img_end,
            model.image_newline,
            model.token_embd_img_pad,
        };
        for (sentinels) |tok| {
            cur = c.ggml_concat(ctx0, cur, c.ggml_reshape_2d(ctx0, tok, n_embd_out, 1), 1);
        }

        const n_llm_w = @divTrunc(clipAlign(g.n_patches_x, n_merge), n_merge);
        const n_llm_h = @divTrunc(clipAlign(g.n_patches_y, n_merge), n_merge);
        const n_out = impl.dsv4GetBlockLayout(n_llm_w, n_llm_h, g.img.lead_pad).n_out;
        impl.ggmlAssert(n_grid == @as(i64, n_llm_w) * n_llm_h, "deepseek4v.cpp", 89, "n_grid == n_llm_w * n_llm_h");

        const layout_idx = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_out);
        _ = c.ggml_set_name(layout_idx, "layout_idx");
        c.ggml_set_input(layout_idx);

        cur = c.ggml_get_rows(ctx0, cur, layout_idx);
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
