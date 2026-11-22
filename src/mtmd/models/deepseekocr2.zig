//! The DeepSeek-OCR 2 encoder graph: DeepSeek-OCR's SAM, then a Qwen2
//! encoder over SAM's tokens with learned queries appended, keeping only
//! the queries.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/deepseekocr2.cpp` at v1.0.0
//! (`b1b770d90`).

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// The `add_rope` lambda (deepseekocr2.cpp:48 @b1b770d90) of `build`.
fn addRope(ctx: ?*anyopaque, g: *mtmd.graph.Graph, x: mtmd.graph.T, layer: *const mtmd.model.Layer) mtmd.graph.T {
    _ = layer;
    const inp_pos: *mtmd.graph.T = @ptrCast(@alignCast(ctx.?));
    return llama.impl.llama_c.ggml_rope_ext(g.ctx0, x, inp_pos.*, null, g.d_head, llama.impl.llama_c.GGML_ROPE_TYPE_NEOX, 131072, 1000000, 1, 0, 1, 0, 0);
}

/// Ports `build` (deepseekocr2.cpp:3 @b1b770d90), a member of `mtmd_graph_deepseekocr2`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    mtmd.impl.ggmlAssert(g.hparams.n_head_kv > 0, "deepseekocr2.cpp", 4, "hparams.n_head_kv > 0");
    mtmd.impl.ggmlAssert(@rem(g.n_head, g.hparams.n_head_kv) == 0, "deepseekocr2.cpp", 5, "n_head % hparams.n_head_kv == 0");

    // patch embedding
    const inp_raw = g.buildInpRaw(3);

    const sam_out = mtmd.models.deepseekocr.buildSam(g, inp_raw);

    var qwen2_out: mtmd.graph.T = undefined;
    // Building Qwen2 encoder
    {
        // H*W, C, B
        var inp: mtmd.graph.T = llama.impl.llama_c.ggml_reshape_3d(ctx0, sam_out, mtmd.graph.ne(sam_out, 0) * mtmd.graph.ne(sam_out, 1), mtmd.graph.ne(sam_out, 2), mtmd.graph.ne(sam_out, 3));
        inp = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_permute(ctx0, inp, 1, 0, 2, 3)); // C, H*W, B

        const num_image_tokens = mtmd.graph.ne(inp, 1); // H*W
        mtmd.impl.ggmlAssert(num_image_tokens == 144 or num_image_tokens == 256, "deepseekocr2.cpp", 22, "num_image_tokens == 144 || num_image_tokens == 256");

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
        query_embed = llama.impl.llama_c.ggml_cast(ctx0, query_embed, inp.?.type);
        query_embed = llama.impl.llama_c.ggml_repeat_4d(ctx0, query_embed, mtmd.graph.ne(query_embed, 0), num_queries, mtmd.graph.ne(inp, 2), 1);
        inp = llama.impl.llama_c.ggml_concat(ctx0, inp, query_embed, 1);

        const seq_len = mtmd.graph.ne(inp, 1);

        // qwen2 encoder attention mask
        const attn_mask = g.buildInpAttnMask(seq_len, seq_len, 0);

        var inp_pos: mtmd.graph.T = llama.impl.llama_c.ggml_cast(ctx0, llama.impl.llama_c.ggml_arange(ctx0, 0, @floatFromInt(seq_len), 1), llama.impl.llama_c.GGML_TYPE_I32);

        // build_vit applies model.post_ln_w internally; do not re-apply
        var cur = g.buildVit(inp, seq_len, .rms, .silu, null, .{ .ctx = @ptrCast(&inp_pos), .f = addRope }, .{ .attn_mask = attn_mask });

        cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_3d(ctx0, cur, mtmd.graph.ne(cur, 0), num_queries, mtmd.graph.ne(cur, 2), mtmd.graph.nb(cur, 1), mtmd.graph.nb(cur, 2), mtmd.graph.nb(cur, 1) * @as(usize, @intCast(mtmd.graph.ne(cur, 1) - num_queries)))); // only take query tokens for output

        llama.impl.llama_c.ggml_build_forward_expand(g.gf, cur);
        qwen2_out = cur;
    }

    var cur = llama.impl.llama_c.ggml_mul_mat(ctx0, model.mm_fc_w, qwen2_out);
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, model.mm_fc_b);

    // view_seperator only after the global view
    if (g.img.add_viewsep) {
        const vs = llama.impl.llama_c.ggml_repeat_4d(ctx0, model.view_seperator, mtmd.graph.ne(model.view_seperator, 0), 1, mtmd.graph.ne(cur, 2), 1);
        cur = llama.impl.llama_c.ggml_concat(ctx0, cur, vs, 1); // (n_dim, 257, n_batch)
    }

    g.cb(cur, "dsocr2_output", -1);

    llama.impl.llama_c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
