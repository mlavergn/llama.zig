//! The MuseGlimmer vision encoder: a 2D-RoPE ViT with block-diagonal window
//! attention (every `sparse_factor`-th and the last layer global), a pixel
//! shuffle, the adapter MLP and the LLM's vision projection.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/muse-glimmer.cpp` at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const kimivl = @import("kimivl.zig");
const c = impl.c;

/// The `inp_i32` lambda (mtmd/models/muse-glimmer.cpp:21 @c1d0e7a00): a named I32 input.
fn inpI32(g: *graph.Graph, name: [*:0]const u8, n: i64) graph.T {
    const t = c.ggml_new_tensor_1d(g.ctx0, c.GGML_TYPE_I32, n);
    _ = c.ggml_set_name(t, name);
    c.ggml_set_input(t);
    return t;
}

/// Ports `build` (mtmd/models/muse-glimmer.cpp:14 @c1d0e7a00), a member of `clip_graph_muse_glimmer`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const ds = hparams.n_merge; // downsample factor (2)
    const sf = hparams.muse_glimmer_sparse_factor; // 4
    const n_tok = g.n_patches;
    const n_out = @divTrunc(g.n_patches_x, ds) * @divTrunc(g.n_patches_y, ds);
    const rope_base = hparams.rope_theta; // 10000

    const pos_w = inpI32(g, "muse_glimmer_pos_w", n_tok);
    const pos_h = inpI32(g, "muse_glimmer_pos_h", n_tok);
    const sp_perm = inpI32(g, "muse_glimmer_sp_perm", n_tok);
    const inv_perm = inpI32(g, "muse_glimmer_inv_perm", n_tok);
    const ds_perm = inpI32(g, "muse_glimmer_ds_perm", n_tok);

    const sp_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_tok, n_tok);
    _ = c.ggml_set_name(sp_mask, "muse_glimmer_sp_mask");
    c.ggml_set_input(sp_mask);

    // patchify via build_inp (conv2d over raw pixels) + bilinear-resized learned pos-emb
    var x = g.buildInp(); // [n_embd, n_tok, 1]
    x = c.ggml_add(ctx0, x, g.resizePositionEmbeddings(c.GGML_SCALE_MODE_BILINEAR));
    g.cb(x, "after_posemb", -1);

    // group patches into pgrid x pgrid windows (sparse attention order)
    x = c.ggml_get_rows(ctx0, x, sp_perm);
    g.cb(x, "after_sp_perm", -1);

    // per-layer mask: sparse layers get sp_mask, global layers (every sf-th and last) get none
    const attn_mask_layers = try impl.gpa.alloc(graph.T, @intCast(g.n_layer));
    defer impl.gpa.free(attn_mask_layers);
    for (attn_mask_layers, 0..) |*m, i| {
        const il: c_int = @intCast(i);
        const is_global = (il == g.n_layer - 1) or (@rem(il + 1, sf) == 0);
        m.* = if (is_global) null else sp_mask;
    }

    // 2D RoPE: first half of head_dim uses width pos, second half uses height pos
    var rope: kimivl.Rope2d = .{ .pos_a = pos_w, .pos_b = pos_h, .freq_base = rope_base };

    // pre_ln, per-layer transformer, post_ln (all inside build_vit); reference uses exact (erf) GELU
    x = g.buildVit(x, n_tok, .normal, .gelu_erf, null, .{ .ctx = &rope, .f = kimivl.Rope2d.addPos }, .{ .attn_mask_layers = attn_mask_layers });

    // un-permute back to original grid order
    x = c.ggml_get_rows(ctx0, x, inv_perm);
    g.cb(x, "after_inv_perm", -1);

    // pixel-shuffle downsample: gather f*f spatial neighbors then concat channel-outer.
    // out[c*(ds*ds)+s, o] = x[ds_perm gathered][o*(ds*ds)+s, c]
    x = c.ggml_get_rows(ctx0, x, ds_perm); // [n_embd, n_tok], grouped
    x = c.ggml_reshape_3d(ctx0, x, n_embd, ds * ds, n_out); // [c, s, o]
    x = c.ggml_permute(ctx0, x, 1, 0, 2, 3); // [s, c, o]
    x = c.ggml_cont(ctx0, x);
    x = c.ggml_reshape_2d(ctx0, x, n_embd * ds * ds, n_out); // [6144, n_out]
    g.cb(x, "encoder_out", -1);

    // adapter (6144->4096->4096, exact GELU each) + LLM vision_projection (4096->6656)
    x = g.buildMm(model.mm_0_w, x);
    x = c.ggml_gelu_erf(ctx0, x);
    x = g.buildMm(model.mm_1_w, x);
    x = c.ggml_gelu_erf(ctx0, x);
    x = g.buildMm(model.mm_2_w, x); // [6656, n_out]
    g.cb(x, "projected", -1);

    c.ggml_build_forward_expand(g.gf, x);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
