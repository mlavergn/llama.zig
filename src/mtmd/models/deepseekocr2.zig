//! The DeepSeek-OCR 2 encoder graph: DeepSeek-OCR's SAM, then a Qwen2
//! encoder over SAM's tokens with learned queries appended, keeping only
//! the queries.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/deepseekocr2.cpp` at v0.3.0
//! (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const deepseekocr = @import("deepseekocr.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// The `add_rope` lambda (deepseekocr2.cpp:50 @c1d0e7a00) of `build`.
fn addRope(ctx: ?*anyopaque, g: *graph.Graph, x: T, layer: *const model_mod.Layer) T {
    _ = layer;
    const inp_pos: *T = @ptrCast(@alignCast(ctx.?));
    return c.ggml_rope_ext(g.ctx0, x, inp_pos.*, null, g.d_head, c.GGML_ROPE_TYPE_NEOX, 131072, 1000000, 1, 0, 1, 0, 0);
}

/// Ports `build` (deepseekocr2.cpp:3 @c1d0e7a00), a member of `clip_graph_deepseekocr2`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    impl.ggmlAssert(g.hparams.n_head_kv > 0, "deepseekocr2.cpp", 4, "hparams.n_head_kv > 0");
    impl.ggmlAssert(@rem(g.n_head, g.hparams.n_head_kv) == 0, "deepseekocr2.cpp", 5, "n_head % hparams.n_head_kv == 0");

    // patch embedding
    const inp_raw = g.buildInpRaw(3);

    const sam_out = deepseekocr.buildSam(g, inp_raw);

    var qwen2_out: T = undefined;
    // Building Qwen2 encoder
    {
        // H*W, C, B
        var inp: T = c.ggml_reshape_3d(ctx0, sam_out, ne(sam_out, 0) * ne(sam_out, 1), ne(sam_out, 2), ne(sam_out, 3));
        inp = c.ggml_cont(ctx0, c.ggml_permute(ctx0, inp, 1, 0, 2, 3)); // C, H*W, B

        const num_image_tokens = ne(inp, 1); // H*W
        impl.ggmlAssert(num_image_tokens == 144 or num_image_tokens == 256, "deepseekocr2.cpp", 22, "num_image_tokens == 144 || num_image_tokens == 256");

        // query based on numbers of image tokens (in SAM output)
        // 16x16 -> query_1024 (1024x1024 images)
        // 12x12 -> query_768 (768x768 images)

        var query_embed = model.resample_query_1024;
        var num_queries: c_int = 256;

        if (num_image_tokens == 144) {
            query_embed = model.resample_query_768;
            num_queries = 144;
        }

        // repeat the query embedding per batch item, then append: (C, num_image_tokens + num_queries, B)
        query_embed = c.ggml_cast(ctx0, query_embed, inp.?.type);
        query_embed = c.ggml_repeat_4d(ctx0, query_embed, ne(query_embed, 0), num_queries, ne(inp, 2), 1);
        inp = c.ggml_concat(ctx0, inp, query_embed, 1);

        const seq_len = ne(inp, 1);

        // qwen2 encoder attention mask
        const attn_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, seq_len, seq_len);
        _ = c.ggml_set_name(attn_mask, "qwen2_attn_mask");
        c.ggml_set_input(attn_mask);

        var inp_pos: T = c.ggml_cast(ctx0, c.ggml_arange(ctx0, 0, @floatFromInt(seq_len), 1), c.GGML_TYPE_I32);

        // build_vit applies model.post_ln_w internally; do not re-apply
        var cur = g.buildVit(inp, seq_len, .rms, .silu, null, .{ .ctx = @ptrCast(&inp_pos), .f = addRope }, .{ .attn_mask = attn_mask });

        cur = c.ggml_cont(ctx0, c.ggml_view_3d(ctx0, cur, ne(cur, 0), num_queries, ne(cur, 2), nb(cur, 1), nb(cur, 2), nb(cur, 1) * @as(usize, @intCast(ne(cur, 1) - num_queries)))); // only take query tokens for output

        c.ggml_build_forward_expand(g.gf, cur);
        qwen2_out = cur;
    }

    var cur = c.ggml_mul_mat(ctx0, model.mm_fc_w, qwen2_out);
    cur = c.ggml_add(ctx0, cur, model.mm_fc_b);

    // view_seperator only after the global view
    if (g.img.add_viewsep) {
        const vs = c.ggml_repeat_4d(ctx0, model.view_seperator, ne(model.view_seperator, 0), 1, ne(cur, 2), 1);
        cur = c.ggml_concat(ctx0, cur, vs, 1); // (n_dim, 257, n_batch)
    }

    g.cb(cur, "dsocr2_output", -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
