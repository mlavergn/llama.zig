//! The dots.ocr vision encoder (also dots3note's vision tower): an M-RoPE
//! ViT, its patch merger and projector.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/dotsocr.cpp` at v0.4.0 (`5266f24da`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;

/// The `add_pos` lambda of a 4-section vision M-RoPE (dotsocr.cpp:14,
/// paddleocr.cpp:13 @5266f24da): `ggml_rope_multi` over the `positions`
/// input, a quarter of the head per section.
pub const MropeVision = struct {
    positions: graph.T,
    mrope_sections: [4]c_int,

    pub fn addPos(ctx: ?*anyopaque, g: *graph.Graph, cur: graph.T, layer: *const model_mod.Layer) graph.T {
        _ = layer;
        const self: *MropeVision = @ptrCast(@alignCast(ctx.?));
        return c.ggml_rope_multi(g.ctx0, cur, self.positions, null, @divTrunc(g.d_head, 2), &self.mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
    }
};

/// Ports `build` (dotsocr.cpp:3 @5266f24da), a member of `clip_graph_dotsocr`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_pos = g.n_patches;
    const num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    // note: similar to PaddleOCR
    const q = @divTrunc(g.d_head, 4);
    var rope: MropeVision = .{ .positions = null, .mrope_sections = .{ q, q, q, q } };

    const positions = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, num_position_ids);
    _ = c.ggml_set_name(positions, "positions");
    c.ggml_set_input(positions);
    rope.positions = positions;

    const inp = g.buildInp();
    var cur = g.buildVit(inp, g.n_patches, .rms, hparams.ffn_op, null, .{ .ctx = &rope, .f = MropeVision.addPos }, .{});

    g.cb(cur, "vit_out", -1);

    // dots.ocr patch merger + projector
    {
        impl.ggmlAssert(hparams.n_merge > 0, "dotsocr.cpp", 33, "hparams.n_merge > 0");
        cur = g.buildNorm(cur, model.mm_input_norm_w, model.mm_input_norm_b, .normal, 1e-6, -1);
        cur = g.buildPatchMergePermute(cur, hparams.n_merge);
        g.cb(cur, "after_patch_merger", -1);
        // nn.GELU() defaults to exact erf-based GELU
        cur = g.buildFfn(cur, model.mm_0_w, model.mm_0_b, null, null, model.mm_2_w, model.mm_2_b, .gelu_erf, -1);
        g.cb(cur, "after_projector", -1);
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
