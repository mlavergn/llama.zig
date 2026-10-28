//! The CogVLM vision encoder: a post-norm ViT with a CLS token, and the
//! SwiGLU projector framed by the boi and eoi embeddings.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/cogvlm.cpp` at v0.4.0 (`5266f24da`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;

/// Ports `build` (mtmd/models/cogvlm.cpp:3 @5266f24da), a member of `clip_graph_cogvlm`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const d_head = g.d_head;
    const n_embd = g.n_embd;
    impl.ggmlAssert(model.class_embedding != null, "cogvlm.cpp", 4, "model.class_embedding != nullptr");
    impl.ggmlAssert(model.position_embeddings != null, "cogvlm.cpp", 5, "model.position_embeddings != nullptr");

    const n_pos = g.n_patches + 1; // +1 for [CLS]

    // build input and concatenate class embedding
    var inp = g.buildInp();
    inp = c.ggml_concat(ctx0, inp, model.class_embedding, 1);

    inp = c.ggml_add(ctx0, inp, model.position_embeddings);
    g.cb(inp, "inp_pos", -1);

    var inpL = inp;

    const f32_size = @sizeOf(f32);
    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        var cur = inpL;

        cur = g.buildMm(layer.qkv_w, cur);

        cur = c.ggml_add(ctx0, cur, layer.qkv_b);

        const Qcur = c.ggml_view_3d(ctx0, cur, d_head, g.n_head, n_pos, @as(usize, @intCast(d_head)) * f32_size, graph.nb(cur, 1), 0);
        const Kcur = c.ggml_view_3d(ctx0, cur, d_head, g.n_head, n_pos, @as(usize, @intCast(d_head)) * f32_size, graph.nb(cur, 1), @as(usize, @intCast(n_embd)) * f32_size);
        const Vcur = c.ggml_view_3d(ctx0, cur, d_head, g.n_head, n_pos, @as(usize, @intCast(d_head)) * f32_size, graph.nb(cur, 1), 2 * @as(usize, @intCast(n_embd)) * f32_size);

        g.cb(Qcur, "Qcur", il);
        g.cb(Kcur, "Kcur", il);
        g.cb(Vcur, "Vcur", il);

        cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, null, g.kq_scale, il, null);
        g.cb(cur, "attn_out", il);

        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .normal, eps, il);
        g.cb(cur, "attn_post_norm", il);

        cur = c.ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);

        g.cb(cur, "ffn_out", il);

        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, eps, il);
        g.cb(cur, "ffn_post_norm", il);

        cur = c.ggml_add(ctx0, cur, inpL);
        g.cb(cur, "layer_out", il);
        inpL = cur;
    }

    // remove CLS token (like build_llama4 does)
    var cur = c.ggml_view_2d(ctx0, inpL, n_embd, g.n_patches, c.ggml_row_size(inpL.?.type, n_embd), 0);

    // Multiply with mm_model_proj
    cur = g.buildMm(model.mm_model_proj, cur);

    // Apply layernorm, weight, bias
    cur = g.buildNorm(cur, model.mm_post_fc_norm_w, model.mm_post_fc_norm_b, .normal, 1e-5, -1);

    // Apply GELU
    cur = c.ggml_gelu_inplace(ctx0, cur);

    // Branch 1: multiply with mm_h_to_4h_w
    const h_to_4h = g.buildMm(model.mm_h_to_4h_w, cur);

    // Branch 2: multiply with mm_gate_w
    var gate = g.buildMm(model.mm_gate_w, cur);

    // Apply silu
    gate = c.ggml_swiglu_split(ctx0, gate, h_to_4h);

    // Apply mm_4h_to_h_w
    cur = g.buildMm(model.mm_4h_to_h_w, gate);

    // Concatenate with boi and eoi
    cur = c.ggml_concat(ctx0, model.mm_boi, cur, 1);
    cur = c.ggml_concat(ctx0, cur, model.mm_eoi, 1);

    // build the graph
    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
