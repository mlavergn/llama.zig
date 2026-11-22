//! The dots.3note audio encoder (`mtmd_graph_dots3note_a`): one 60 s mel
//! chunk through three strided conv2d blocks, an RMS-norm ViT with partial
//! NEOX rotary, and a LayerNorm-MLP adapter.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/dots3note.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// The `conv_block` lambda (mtmd/models/dots3note.cpp:12 @b1b770d90) of `build`.
fn convBlock(ctx0: ?*llama.impl.llama_c.ggml_context, x_in: mtmd.graph.T, w: mtmd.graph.T, b: mtmd.graph.T) mtmd.graph.T {
    var x = llama.impl.llama_c.ggml_conv_2d(ctx0, w, x_in, 2, 2, 1, 1, 1, 1);
    x = llama.impl.llama_c.ggml_add(ctx0, x, llama.impl.llama_c.ggml_reshape_4d(ctx0, b, 1, 1, mtmd.graph.ne(x, 2), 1));
    return llama.impl.llama_c.ggml_gelu_erf(ctx0, x);
}

/// The `add_pos` lambda (mtmd/models/dots3note.cpp:40 @b1b770d90) of `build`: partial
/// rotary, the first half of each head, NEOX style. `ctx` is the positions
/// tensor.
fn addPos(ctx: ?*anyopaque, g: *mtmd.graph.Graph, cur: mtmd.graph.T, layer: *const mtmd.model.Layer) mtmd.graph.T {
    _ = layer;
    const positions: mtmd.graph.T = @ptrCast(@alignCast(ctx));
    return llama.impl.llama_c.ggml_rope_ext(g.ctx0, cur, positions, null, @divTrunc(g.d_head, 2), llama.impl.llama_c.GGML_ROPE_TYPE_NEOX, 0, g.hparams.rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
}

/// Ports `build` (mtmd/models/dots3note.cpp:3 @b1b770d90), a member of
/// `mtmd_graph_dots3note_a`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    // inp_raw: [n_frames, n_mel, 1], one 60s chunk, mel frames not padded
    // the reference impl zero-masks conv inputs beyond the valid length at each stage;
    // running on exactly the valid frames with the convs' zero padding is equivalent
    var inp = g.buildInpRaw(1);
    mtmd.impl.ggmlAssert(inp.?.type == llama.impl.llama_c.GGML_TYPE_F32, "dots3note.cpp", 8, "inp->type == GGML_TYPE_F32");

    // 3x conv2d (k=3, s=2, p=1) + gelu
    {
        inp = convBlock(ctx0, inp, model.conv2d_1_w, model.conv2d_1_b);
        inp = convBlock(ctx0, inp, model.conv2d_2_w, model.conv2d_2_b);
        inp = convBlock(ctx0, inp, model.conv2d_3_w, model.conv2d_3_b);
        // inp: [OW=n_frames/8, OH=n_mel/8, OC=480, 1]
        g.cb(inp, "after_conv_stem", -1);
    }

    // [OW, OH, OC, 1] -> [OH*OC, OW], feature index f + OH*c (matches the reference permute+reshape)
    inp = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = llama.impl.llama_c.ggml_reshape_2d(ctx0, inp, mtmd.graph.ne(inp, 0) * mtmd.graph.ne(inp, 1), mtmd.graph.ne(inp, 2));

    // project to d_model (no bias)
    inp = llama.impl.llama_c.ggml_mul_mat(ctx0, model.conv_out_w, inp);
    g.cb(inp, "after_conv_out", -1);

    const n_pos = mtmd.graph.ne(inp, 1);

    const positions = llama.impl.llama_c.ggml_new_tensor_1d(ctx0, llama.impl.llama_c.GGML_TYPE_I32, n_pos);
    _ = llama.impl.llama_c.ggml_set_name(positions, "positions");
    llama.impl.llama_c.ggml_set_input(positions);

    var cur = g.buildVit(inp, n_pos, .rms, hparams.ffn_op, null, .{ .ctx = positions, .f = addPos }, .{});
    g.cb(cur, "after_transformer", -1);

    // adapter: LayerNorm -> Linear -> GELU -> Linear
    cur = g.buildNorm(cur, model.mm_norm_pre_w, model.mm_norm_pre_b, .normal, 1e-5, -1);
    cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
    g.cb(cur, "projected", -1);

    llama.impl.llama_c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
