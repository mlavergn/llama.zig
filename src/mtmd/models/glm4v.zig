//! The GLM4V encoder graph: a ViT over a decoupled Conv3D patch embedding
//! with M-RoPE, then GLM4V's patch merger, FC and FFN projectors.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/glm4v.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;

/// The `add_pos` lambda's captures.
const PosCtx = struct {
    positions: T,
    mrope_sections: [4]c_int,
};

/// The `add_pos` lambda (glm4v.cpp:64 @b29c606e2) of `build`.
fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: T, layer: *const model_mod.Layer) T {
    _ = layer;
    const p: *PosCtx = @ptrCast(@alignCast(ctx.?));
    return c.ggml_rope_multi(g.ctx0, cur, p.positions, null, @divTrunc(g.d_head, 2), &p.mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, g.hparams.rope_theta, 1, 0, 1, 32, 1);
}

/// Ports `build` (glm4v.cpp:3 @b29c606e2), a member of `clip_graph_glm4v`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    impl.ggmlAssert(model.patch_bias != null, "glm4v.cpp", 4, "model.patch_bias != nullptr");
    impl.ggmlAssert(model.class_embedding == null, "glm4v.cpp", 5, "model.class_embedding == nullptr");

    const batch_size: c_int = 1;
    const n_embd = g.n_embd;
    const npx = g.n_patches_x;
    const npy = g.n_patches_y;

    const norm_t: model_mod.NormType = .rms;

    const inp_raw = g.buildInpRaw(3);
    var inp = c.ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, g.patch_size, g.patch_size, 0, 0, 1, 1);

    var pc: PosCtx = .{ .positions = null, .mrope_sections = .{ @divTrunc(g.d_head, 4), @divTrunc(g.d_head, 4), @divTrunc(g.d_head, 4), @divTrunc(g.d_head, 4) } };
    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches * 4);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);
    pc.positions = positions;

    impl.ggmlAssert(@rem(g.img.nx(), g.patch_size * 2) == 0, "glm4v.cpp", 19, "img.nx() % (patch_size * 2) == 0");
    impl.ggmlAssert(@rem(g.img.ny(), g.patch_size * 2) == 0, "glm4v.cpp", 20, "img.ny() % (patch_size * 2) == 0");

    // second conv dimension
    {
        const inp_1 = c.ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, g.patch_size, g.patch_size, 0, 0, 1, 1);
        inp = c.ggml_add(ctx0, inp, inp_1);

        inp = c.ggml_permute(ctx0, inp, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
        inp = c.ggml_cont_4d(ctx0, inp, n_embd * 2, @divTrunc(npx, 2), npy, batch_size);
        inp = c.ggml_reshape_4d(ctx0, inp, n_embd * 2, @divTrunc(npx, 2), 2, batch_size * @divTrunc(npy, 2));
        inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = c.ggml_cont_3d(ctx0, inp, n_embd, npx * npy, batch_size);
    }

    // add patch bias
    inp = c.ggml_add(ctx0, inp, model.patch_bias);
    g.cb(inp, "patch_bias", -1);

    // pos-conv norm
    inp = g.buildNorm(inp, model.norm_embd_w, model.norm_embd_b, norm_t, g.eps, -1);

    var learned_pos_embd: T = null;
    // Note: GLM-OCR does not have learned position embeddings
    if (model.position_embeddings != null) {
        learned_pos_embd = g.resizePositionEmbeddings(c.GGML_SCALE_MODE_BICUBIC);
        learned_pos_embd = c.ggml_cont_4d(ctx0, learned_pos_embd, n_embd * 2, @divTrunc(npx, 2), npy, batch_size);
        learned_pos_embd = c.ggml_reshape_4d(ctx0, learned_pos_embd, n_embd * 2, @divTrunc(npx, 2), 2, batch_size * @divTrunc(npy, 2));
        learned_pos_embd = c.ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
        learned_pos_embd = c.ggml_cont_3d(ctx0, learned_pos_embd, n_embd, npx * npy, batch_size);
        g.cb(learned_pos_embd, "learned_pos_embd", -1);
    }

    var cur = g.buildVit(inp, g.n_patches, norm_t, g.hparams.ffn_op, learned_pos_embd, .{ .ctx = &pc, .f = addPos }, .{});

    g.cb(cur, "vit_out", -1);

    // GLM4V projector
    // ref: https://github.com/huggingface/transformers/blob/40dc11cd3eb4126652aa41ef8272525affd4a636/src/transformers/models/glm4v/modeling_glm4v.py#L116-L130

    // patch merger (downsample)
    {
        const n_merge = g.hparams.n_merge;
        impl.ggmlAssert(n_merge > 0, "glm4v.cpp", 87, "n_merge > 0");

        const n_token_out = @divTrunc(@divTrunc(g.n_patches, n_merge), n_merge);
        cur = c.ggml_reshape_4d(ctx0, cur, n_embd, n_merge, n_merge, n_token_out);
        cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 2, 0, 1, 3)); // [n_merge, n_merge, n_embd, n_token_out]
        cur = c.ggml_conv_2d(ctx0, model.mm_patch_merger_w, cur, n_merge, n_merge, 0, 0, 1, 1);
        cur = c.ggml_reshape_2d(ctx0, cur, ne(cur, 2), n_token_out); // [n_embd_out, n_token_out]

        cur = c.ggml_add(ctx0, cur, model.mm_patch_merger_b);
    }

    // FC projector
    {
        cur = g.buildMm(model.mm_fc_w, cur);
        // default LayerNorm (post_projection_norm)
        cur = g.buildNorm(cur, model.mm_post_norm_w, model.mm_post_norm_b, .normal, 1e-5, -1);
        cur = c.ggml_gelu_erf(ctx0, cur);
        g.cb(cur, "after_fc_proj", -1);
    }

    // FFN projector
    {
        cur = g.buildFfn(cur, model.mm_ffn_up_w, model.mm_ffn_up_b, model.mm_ffn_gate_w, model.mm_ffn_gate_b, model.mm_ffn_down_w, model.mm_ffn_down_b, g.hparams.ffn_op, -1);
        g.cb(cur, "after_ffn_proj", -1);
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
