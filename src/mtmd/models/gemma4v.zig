//! The Gemma 4 vision encoder: lookup-table position embeddings, a NEOX
//! 2D-RoPE ViT, the vision pooler and the multimodal embedder, with
//! Gemma4ClippableLinear's clamped matmuls.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/gemma4v.cpp` at v1.0.0 (`b1b770d90`).

const std = @import("std");
const mtmd = @import("../module.zig");
const stdlib = @import("stdlib");

/// The `add_pos` lambda (gemma4v.cpp:46 @b1b770d90): `build_rope_2d` with
/// NEOX ordering, `pos_x` on the first half of each head and `pos_y` on
/// the second.
pub const NeoxRope2d = struct {
    pos_x: mtmd.graph.T,
    pos_y: mtmd.graph.T,
    rope_theta: f32,

    pub fn addPos(ctx: ?*anyopaque, g: *mtmd.graph.Graph, cur_in: mtmd.graph.T, layer: *const mtmd.model.Layer) mtmd.graph.T {
        _ = layer;
        const self: *NeoxRope2d = @ptrCast(@alignCast(ctx.?));
        const n_dim = mtmd.graph.ne(cur_in, 0);
        const half: c_int = @intCast(@divTrunc(n_dim, 2));

        // first half, dims [0, n_dim/2)
        var cur = mtmd.impl.c.ggml_rope_ext(g.ctx0, cur_in, self.pos_x, null, half, mtmd.impl.c.GGML_ROPE_TYPE_NEOX, 0, self.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

        // second half, dims [n_dim/2, n_dim)
        cur = mtmd.impl.c.ggml_rope_ext(g.ctx0, cur, self.pos_y, null, half, mtmd.impl.c.GGML_ROPE_TYPE_NEOX, 0, self.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
        cur = mtmd.impl.c.ggml_rope_set_offset(cur, half);
        return cur;
    }
};

/// The x and y position-embedding lookups shared with gemma4uv
/// (the block at gemma4v.cpp line 26): two tables stacked in
/// `position_embeddings`.
pub fn addPosEmbd(g: *mtmd.graph.Graph, inp_in: mtmd.graph.T, pos_x: mtmd.graph.T, pos_y: mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const pe = g.model.position_embeddings;
    const pos_size = mtmd.graph.ne(pe, 1);
    const nb1 = mtmd.impl.c.ggml_row_size(pe.?.type, g.n_embd);

    // positional embeddings are stored as lookup tables (one for x, one for y)
    const tbl_x = mtmd.impl.c.ggml_view_2d(ctx0, pe, g.n_embd, pos_size, nb1, 0);
    const tbl_y = mtmd.impl.c.ggml_view_2d(ctx0, pe, g.n_embd, pos_size, nb1, @as(usize, @intCast(pos_size)) * nb1);

    // ggml_get_rows: [n_embd, n_patches]
    const emb_x = mtmd.impl.c.ggml_get_rows(ctx0, tbl_x, pos_x);
    const emb_y = mtmd.impl.c.ggml_get_rows(ctx0, tbl_y, pos_y);

    var inp = mtmd.impl.c.ggml_add(ctx0, inp_in, emb_x);
    inp = mtmd.impl.c.ggml_add(ctx0, inp, emb_y);
    g.cb(inp, "pos_embd", -1);
    return inp;
}

/// Ports `build` (gemma4v.cpp:4 @b1b770d90), a member of `mtmd_graph_gemma4v`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    g.mm_hook = buildMm;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const patch_size = g.patch_size;

    var inp_raw = g.buildInpRaw(3);

    // patches = 2 * (patches - 0.5), equivalent to: patches * 2 - 1
    inp_raw = mtmd.impl.c.ggml_scale_bias(ctx0, inp_raw, 2.0, -1.0);
    _ = mtmd.impl.c.ggml_set_name(inp_raw, "inp_raw_scaled");

    var inp: mtmd.graph.T = mtmd.impl.c.ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = mtmd.impl.c.ggml_reshape_3d(ctx0, inp, g.n_patches, g.n_embd, g.n_batch);
    inp = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, inp));
    _ = mtmd.impl.c.ggml_set_name(inp, "inp");
    // note: no patch bias

    const pos_x = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, g.n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_x, "pos_x");
    mtmd.impl.c.ggml_set_input(pos_x);

    const pos_y = mtmd.impl.c.ggml_new_tensor_1d(ctx0, mtmd.impl.c.GGML_TYPE_I32, g.n_patches);
    _ = mtmd.impl.c.ggml_set_name(pos_y, "pos_y");
    mtmd.impl.c.ggml_set_input(pos_y);

    inp = addPosEmbd(g, inp, pos_x, pos_y);

    var rope: NeoxRope2d = .{ .pos_x = pos_x, .pos_y = pos_y, .rope_theta = hparams.rope_theta };

    g.kq_scale = 1.0;
    // pos embd is already handled above
    var cur = g.buildVit(inp, g.n_patches, .rms, hparams.ffn_op, null, .{ .ctx = &rope, .f = NeoxRope2d.addPos }, .{});

    // Gemma4VisionPooler
    {
        const kernel_size = hparams.n_merge;
        mtmd.impl.ggmlAssert(kernel_size > 0, "gemma4v.cpp", 86, "kernel_size > 0");

        // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, n_batch]
        cur = mtmd.impl.c.ggml_cont_4d(ctx0, mtmd.impl.c.ggml_transpose(ctx0, cur), g.n_patches_x, g.n_patches_y, g.n_embd, g.n_batch);
        cur = mtmd.impl.c.ggml_pool_2d(ctx0, cur, mtmd.impl.c.GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        const out_x = @divTrunc(g.n_patches_x, kernel_size);
        const out_y = @divTrunc(g.n_patches_y, kernel_size);
        // [out_x, out_y, n_embd, n_batch] -> [n_embd, out_x * out_y, n_batch]
        cur = mtmd.impl.c.ggml_reshape_3d(ctx0, cur, out_x * out_y, g.n_embd, g.n_batch);
        cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, cur));
        cur = mtmd.impl.c.ggml_scale(ctx0, cur, stdlib.libm.sqrtfInt(g.n_embd));
        g.cb(cur, "pooled", -1);
    }

    // hidden_states = (hidden_states - self.std_bias) * self.std_scale
    if (model.std_bias != null and model.std_scale != null) {
        cur = mtmd.impl.c.ggml_sub(ctx0, cur, model.std_bias);
        cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.std_scale);
        g.cb(cur, "std_scaled", -1);
    }

    // Gemma4MultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = mtmd.impl.c.ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = g.buildMm(model.mm_input_proj_w, cur);
        g.cb(cur, "projected", -1);
    }

    mtmd.impl.c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

/// Ports `build_mm` (gemma4v.cpp:120 @b1b770d90), a member of
/// `mtmd_graph_gemma4v`: Gemma4ClippableLinear, clamping a weight's input
/// and output where the model lists bounds for it.
pub fn buildMm(g: *const mtmd.graph.Graph, w: mtmd.graph.T, x: mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const ci = g.model.clamp_info_map.get(std.mem.sliceTo(&w.?.name, 0)) orelse return mtmd.impl.c.ggml_mul_mat(ctx0, w, x);
    const clamped = mtmd.impl.c.ggml_clamp(ctx0, x, ci.inp_min, ci.inp_max);
    var out = mtmd.impl.c.ggml_mul_mat(ctx0, w, clamped);
    out = mtmd.impl.c.ggml_clamp(ctx0, out, ci.out_min, ci.out_max);
    return out;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
