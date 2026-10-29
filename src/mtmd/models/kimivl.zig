//! The Kimi-VL vision encoder: a 2D-RoPE ViT, a patch merger and a GELU
//! projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/kimivl.cpp` at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

/// The `add_pos` lambda of a 2D RoPE (kimivl.cpp:16 @b29c606e2):
/// `build_rope_2d(ctx0, cur, pos_a, pos_b, freq_base, interleave_freq)`,
/// `pos_a` on the first half of each head and `pos_b` on the second.
pub const Rope2d = struct {
    pos_a: graph.T,
    pos_b: graph.T,
    freq_base: f32,
    interleave_freq: bool = false,

    pub fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: graph.T, layer: *const model_mod.Layer) graph.T {
        _ = layer;
        const self: *Rope2d = @ptrCast(@alignCast(ctx.?));
        return g.buildRope2d(g.ctx0, cur, self.pos_a, self.pos_b, self.freq_base, self.interleave_freq);
    }
};

/// Ports `build` (kimivl.cpp:3 @b29c606e2), a member of `clip_graph_kimivl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;

    // 2D input positions
    const pos_h = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_h, "pos_h");
    c.ggml_set_input(pos_h);

    const pos_w = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, g.n_patches);
    _ = c.ggml_set_name(pos_w, "pos_w");
    c.ggml_set_input(pos_w);

    const learned_pos_embd = g.resizePositionEmbeddings(graph.default_interpolation_mode);

    // build ViT with 2D position embeddings
    var rope: Rope2d = .{ .pos_a = pos_w, .pos_b = pos_h, .freq_base = hparams.rope_theta };

    const inp = g.buildInp();
    var cur = g.buildVit(inp, g.n_patches, .normal, hparams.ffn_op, learned_pos_embd, .{ .ctx = &rope, .f = Rope2d.addPos }, .{});

    g.cb(cur, "vit_out", -1);

    {
        // patch_merger
        const scale_factor = hparams.n_merge;
        cur = g.buildPatchMergePermute(cur, scale_factor);

        // projection norm
        const proj_inp_dim: c_int = @intCast(graph.ne(cur, 0));
        cur = c.ggml_view_2d(ctx0, cur, g.n_embd, graph.ne(cur, 1) * scale_factor * scale_factor, c.ggml_row_size(cur.?.type, g.n_embd), 0);
        cur = c.ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
        cur = c.ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = c.ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = c.ggml_view_2d(ctx0, cur, proj_inp_dim, @divTrunc(@divTrunc(graph.ne(cur, 1), scale_factor), scale_factor), c.ggml_row_size(cur.?.type, proj_inp_dim), 0);
        g.cb(cur, "proj_inp_normed", -1);

        // projection mlp
        cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_2_w, model.mm_2_b, .gelu, -1);
        g.cb(cur, "proj_out", -1);
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
