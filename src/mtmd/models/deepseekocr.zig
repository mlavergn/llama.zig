//! The DeepSeek-OCR encoder graph: a SAM ViT with windowed and global
//! attention over decomposed relative positions, its neck, then a CLIP ViT
//! over SAM's output, the two concatenated and projected; tile rows and the
//! overview are woven with newline and view-separator embeddings.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/deepseekocr.cpp` at v0.4.0
//! (`5266f24da`). Each declaration names the C++ it replaces and the line
//! it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

extern fn sqrtf(x: f32) f32;

/// Ports `window_partition` (deepseekocr.cpp:5 @5266f24da).
fn windowPartition(ctx0: ?*c.ggml_context, x: T, window: c_int) T {
    const ch = ne(x, 0);
    const w = ne(x, 1);
    const h = ne(x, 2);
    const b = ne(x, 3);
    // same as
    // x = ggml_win_part(m, x, window);
    // x = ggml_reshape_3d(m, x, c, window * window, x->ne[3]);

    const px = @rem(window - @rem(w, window), window);
    const py = @rem(window - @rem(h, window), window);
    const npw = @divTrunc(w + px, window);
    const nph = @divTrunc(h + py, window);

    var cur = x;
    if (px > 0 or py > 0) cur = c.ggml_pad(ctx0, cur, 0, @intCast(px), @intCast(py), 0);
    cur = c.ggml_reshape_4d(ctx0, cur, ch * window, npw, window, nph * b);
    cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = c.ggml_reshape_4d(ctx0, cur, ch, window, window, npw * nph * b);
    return cur;
}

/// Ports `window_unpartition` (deepseekocr.cpp:28 @5266f24da).
fn windowUnpartition(ctx0: ?*c.ggml_context, x: T, w: c_int, h: c_int, window: c_int) T {
    const ch = ne(x, 0);
    // same as
    // x = ggml_reshape_4d(m, x, c, window, window, x->ne[2]);
    // x = ggml_win_unpart(m, x, w, h, window);

    const px: i64 = @rem(window - @rem(w, window), window);
    const py: i64 = @rem(window - @rem(h, window), window);
    const npw = @divTrunc(w + px, window);
    const nph = @divTrunc(h + py, window);

    const b = @divTrunc(ne(x, 3), npw * nph);
    var cur = x;
    cur = c.ggml_reshape_4d(ctx0, cur, ch * window, window, npw, nph * b);
    cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = c.ggml_reshape_4d(ctx0, cur, ch, w + px, h + py, b);
    cur = c.ggml_view_4d(ctx0, cur, ne(cur, 0), w, h, ne(cur, 3), nb(cur, 1), nb(cur, 2), nb(cur, 3), 0);
    cur = c.ggml_cont(ctx0, cur);
    return cur;
}

/// Ports `get_rel_pos` (deepseekocr.cpp:53 @5266f24da).
fn getRelPos(ctx0: ?*c.ggml_context, rel_pos: T, indices: T, q_size: c_int, k_size: c_int) T {
    const C = ne(rel_pos, 0); // channels
    const L = ne(rel_pos, 1); // length

    impl.ggmlAssert(indices != null, "deepseekocr.cpp", 61, "indices != nullptr");
    impl.ggmlAssert(indices.?.type == c.GGML_TYPE_I32, "deepseekocr.cpp", 62, "indices->type == GGML_TYPE_I32");
    impl.ggmlAssert(ne(indices, 0) == k_size, "deepseekocr.cpp", 63, "indices->ne[0] == k_size");
    impl.ggmlAssert(ne(indices, 1) == q_size, "deepseekocr.cpp", 64, "indices->ne[1] == q_size");

    const max_rel_dist: c_int = 2 * @max(q_size, k_size) - 1;
    var cur = rel_pos;

    if (max_rel_dist != L) {
        // Linear interpolation
        const ne0 = ne(cur, 0);
        const ne1 = ne(cur, 1);
        const ne2 = ne(cur, 2);
        const ne3 = ne(cur, 3);

        cur = c.ggml_reshape_3d(ctx0, c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 1, 0, 2, 3)), ne1, 1, ne0 * ne2 * ne3);
        cur = c.ggml_reshape_4d(ctx0, c.ggml_interpolate(ctx0, cur, max_rel_dist, 1, ne0 * ne2 * ne3, 1, c.GGML_SCALE_MODE_BILINEAR), max_rel_dist, ne0, ne2, ne3);
        cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }

    // Flatten indices to 1D for ggml_get_rows
    const qk = q_size * k_size;

    cur = c.ggml_reshape_3d(ctx0, c.ggml_get_rows(ctx0, cur, c.ggml_reshape_1d(ctx0, indices, qk)), C, k_size, q_size);

    return cur; // [C, k_size, q_size]
}

/// Ports `conv_2d_f32` (deepseekocr.cpp:92 @5266f24da): `ggml_conv_2d`
/// with the im2col kept in F32.
fn conv2dF32(ctx0: ?*c.ggml_context, a: T, b: T, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int) T {
    const im2col_type: c.ggml_type = if (a.?.type == c.GGML_TYPE_F16) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;
    const im2col = c.ggml_im2col(ctx0, a, b, s0, s1, p0, p1, d0, d1, true, im2col_type); // [N, OH, OW, IC * KH * KW]

    var result = c.ggml_mul_mat(ctx0, c.ggml_reshape_2d(ctx0, im2col, ne(im2col, 0), ne(im2col, 3) * ne(im2col, 2) * ne(im2col, 1)), c.ggml_reshape_2d(ctx0, a, ne(a, 0) * ne(a, 1) * ne(a, 2), ne(a, 3)));

    result = c.ggml_reshape_4d(ctx0, result, ne(im2col, 1), ne(im2col, 2), ne(im2col, 3), ne(a, 3)); // [OC, N, OH, OW]
    result = c.ggml_cont(ctx0, c.ggml_permute(ctx0, result, 0, 1, 3, 2)); // [N, OC, OH, OW]

    return result;
}

/// Ports `build_sam` (deepseekocr.cpp:108 @5266f24da), a member of
/// `clip_graph_deepseekocr`: the SAM encoder, shared with DeepSeek-OCR 2.
pub fn buildSam(g: *graph.Graph, inp_raw: T) T {
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    // Building SAM
    const n_embd = hparams.sam_n_embd;
    const n_layer = hparams.sam_n_layer;
    const n_heads = hparams.sam_n_head;
    const d_heads = @divTrunc(n_embd, n_heads);
    const window = hparams.attn_window_size;
    // SAM stage runs its layernorms at 1e-6
    const sam_eps: f32 = 1e-6;

    var inpL = conv2dF32(ctx0, model.patch_embed_proj_w, inp_raw, @intCast(ne(model.patch_embed_proj_w, 0)), @intCast(ne(model.patch_embed_proj_w, 1)), 0, 0, 1, 1);
    inpL = c.ggml_add(ctx0, inpL, c.ggml_reshape_3d(ctx0, model.patch_embed_proj_b, 1, 1, n_embd));
    inpL = c.ggml_cont(ctx0, c.ggml_permute(ctx0, inpL, 1, 2, 0, 3));

    const rel_pos_indices_local = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I32, window, window);
    const rel_pos_indices_global = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I32, ne(inpL, 1), ne(inpL, 2));
    _ = c.ggml_set_name(rel_pos_indices_local, "rel_pos_indices_local");
    _ = c.ggml_set_name(rel_pos_indices_global, "rel_pos_indices_global");
    c.ggml_set_input(rel_pos_indices_local);
    c.ggml_set_input(rel_pos_indices_global);

    var cur: T = undefined;
    const tgt_size = ne(inpL, 1);
    const str_size = ne(model.pos_embed, 1);

    if (str_size != tgt_size) {
        const old_pos_embed = c.ggml_cont(ctx0, c.ggml_permute(ctx0, model.pos_embed, 2, 0, 1, 3));
        var new_pos_embed = c.ggml_interpolate(ctx0, old_pos_embed, tgt_size, tgt_size, n_embd, 1, c.GGML_SCALE_MODE_BICUBIC);
        new_pos_embed = c.ggml_cont(ctx0, c.ggml_permute(ctx0, new_pos_embed, 1, 2, 0, 3));
        cur = c.ggml_add(ctx0, inpL, new_pos_embed);
    } else {
        cur = c.ggml_add(ctx0, inpL, model.pos_embed);
    }

    // loop over layers
    var il: c_int = 0;
    while (il < n_layer) : (il += 1) {
        const layer = &model.sam_layers.items[@intCast(il)];
        const shortcut = cur;

        // layernorm1
        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .normal, sam_eps, il);

        const w0 = ne(cur, 1);
        const h0 = ne(cur, 2);

        var indices: T = undefined;

        if (hparams.isGlobalAttn(il)) {
            indices = rel_pos_indices_global;
        } else {
            // local attention layer - apply window partition
            cur = windowPartition(ctx0, cur, window);
            indices = rel_pos_indices_local;
        }

        const W = ne(cur, 1);
        const H = ne(cur, 2);
        // self-attention
        {
            const B: c_int = @truncate(ne(cur, 3));

            cur = c.ggml_mul_mat(ctx0, layer.qkv_w, cur);
            cur = c.ggml_add(ctx0, cur, layer.qkv_b);
            cur = c.ggml_reshape_4d(ctx0, cur, n_embd, 3, W * H, B);

            var Q = c.ggml_view_3d(ctx0, cur, n_embd, W * H, B, nb(cur, 2), nb(cur, 3), 0 * nb(cur, 1));
            Q = c.ggml_reshape_4d(ctx0, c.ggml_cont(ctx0, Q), d_heads, n_heads, W * H, B);

            var K = c.ggml_view_3d(ctx0, cur, n_embd, W * H, B, nb(cur, 2), nb(cur, 3), 1 * nb(cur, 1));
            K = c.ggml_reshape_4d(ctx0, c.ggml_cont(ctx0, K), d_heads, n_heads, W * H, B);

            var V = c.ggml_view_3d(ctx0, cur, n_embd, W * H, B, nb(cur, 2), nb(cur, 3), 2 * nb(cur, 1));
            V = c.ggml_reshape_4d(ctx0, c.ggml_cont(ctx0, V), d_heads, n_heads, W * H, B);

            var rw = getRelPos(ctx0, layer.rel_pos_w, indices, @truncate(W), @truncate(W)); // [W, W, C]
            var rh = getRelPos(ctx0, layer.rel_pos_h, indices, @truncate(H), @truncate(H)); // [H, H, C]
            var qr = c.ggml_permute(ctx0, Q, 0, 2, 1, 3);
            qr = c.ggml_reshape_4d(ctx0, c.ggml_cont(ctx0, qr), d_heads, W, H, B * n_heads);

            rw = c.ggml_mul_mat(ctx0, rw, c.ggml_cont(ctx0, c.ggml_permute(ctx0, qr, 0, 2, 1, 3))); // [B*n_heads, W, H, W]
            rw = c.ggml_cont(ctx0, c.ggml_permute(ctx0, rw, 0, 2, 1, 3)); // [B*n_heads, H, W, W]
            rw = c.ggml_reshape_4d(ctx0, rw, W, 1, W * H, n_heads * B);
            rw = c.ggml_repeat_4d(ctx0, rw, W, H, W * H, n_heads * B);
            rh = c.ggml_mul_mat(ctx0, rh, qr); // [B*n_heads, H, W, H]
            rh = c.ggml_reshape_4d(ctx0, rh, 1, H, W * H, n_heads * B);
            var mask = c.ggml_add(ctx0, rw, rh); // [B*n_heads, H*W, H, W]
            mask = c.ggml_reshape_4d(ctx0, mask, W * H, W * H, n_heads, B);
            // casting mask to F16 only required when flash-attn is enabled
            if (g.flash_attn_type == .enabled) mask = c.ggml_cast(ctx0, mask, c.GGML_TYPE_F16);

            const scale: f32 = 1.0 / sqrtf(@floatFromInt(d_heads));

            cur = g.buildAttn(layer.o_w, layer.o_b, Q, K, V, mask, scale, il, null); // [B, H*W, n_embd]
            cur = c.ggml_reshape_4d(ctx0, c.ggml_cont(ctx0, cur), n_embd, W, H, B);
        }

        if (hparams.isGlobalAttn(il) == false) {
            // local attention layer - reverse window partition
            cur = windowUnpartition(ctx0, cur, @truncate(w0), @truncate(h0), window);
        }

        // re-add the layer input, e.g., residual
        cur = c.ggml_add(ctx0, cur, shortcut);

        const inpFF = cur;

        // layernorm2
        cur = g.buildNorm(inpFF, layer.ln_2_w, layer.ln_2_b, .normal, sam_eps, il);

        // ffn
        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, null, null, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);

        // residual 2
        cur = c.ggml_add(ctx0, cur, inpFF);
        g.cb(cur, "sam_layer_out", il);
    }

    cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 2, 0, 1, 3));

    cur = conv2dF32(ctx0, model.neck_0_w, cur, 1, 1, 0, 0, 1, 1);
    cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 1, 2, 0, 3));
    cur = g.buildNorm(cur, model.neck_1_w, model.neck_1_b, .normal, sam_eps, -1);
    cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 2, 0, 1, 3));

    cur = conv2dF32(ctx0, model.neck_2_w, cur, 1, 1, 1, 1, 1, 1);
    cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 1, 2, 0, 3));
    cur = g.buildNorm(cur, model.neck_3_w, model.neck_3_b, .normal, sam_eps, -1);
    cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 2, 0, 1, 3));

    cur = conv2dF32(ctx0, model.net_2, cur, 2, 2, 1, 1, 1, 1);
    cur = conv2dF32(ctx0, model.net_3, cur, 2, 2, 1, 1, 1, 1);
    g.cb(cur, "sam_output", -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return cur;
}

/// Ports `build` (deepseekocr.cpp:267 @5266f24da), a member of `clip_graph_deepseekocr`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const img = g.img;
    const n_embd = g.n_embd;

    // patch embedding
    var inp_raw = g.buildInpRaw(3);

    const is_overview = img.add_viewsep;
    var n_tiles_per_row: c_int = 0;
    // number of separate "row" images batched together in this graph call
    // (captured now, before n_batch below gets repurposed as the SAM/ViT batch size)
    const n_rows_batch = g.n_batch;

    // note: we expect either a batch of rows or a batch of overviews, but not a mix of both

    if (!is_overview) {
        // handle the case where we have a batch of rows
        // sanity check
        for (g.img_batch.?.entries.items) |*entry| {
            if (entry.add_viewsep) return clip.throwStr("DeepSeek-OCR: mixed overview and non-overview images in batch");
            if (entry.nx() != img.nx() or entry.ny() != img.ny()) return clip.throwStr("DeepSeek-OCR: mixed image sizes in batch");
        }

        impl.ggmlAssert(img.ny() >= img.nx(), "deepseekocr.cpp", 291, "img.ny() >= img.nx()");
        impl.ggmlAssert(@rem(img.ny(), img.nx()) == 0, "deepseekocr.cpp", 292, "img.ny() % img.nx() == 0");
        n_tiles_per_row = @divTrunc(img.ny(), img.nx());

        // each entry is one "row" image of shape [tile_size, tile_size * n_tiles_per_row, 3];
        // merge the tile axis into the batch axis (tile fast, row slow)
        inp_raw = c.ggml_reshape_4d(ctx0, inp_raw, img.nx() * img.nx(), n_tiles_per_row, 3, n_rows_batch);
        inp_raw = c.ggml_cont(ctx0, c.ggml_permute(ctx0, inp_raw, 0, 2, 1, 3));
        inp_raw = c.ggml_reshape_4d(ctx0, inp_raw, img.nx(), img.nx(), 3, n_tiles_per_row * n_rows_batch);
    }

    var sam_out = buildSam(g, inp_raw);

    if (!is_overview) g.n_batch = n_tiles_per_row * n_rows_batch;
    const n_batch = g.n_batch;

    const clip_n_patches: c_int = @truncate(ne(sam_out, 0) * ne(sam_out, 1));

    var clip_out: T = undefined;
    // Building DS-OCR CLIP
    {
        // sam_out: [patch_h, patch_w, n_embd, n_batch]
        // -> [n_embd, clip_n_patches, n_batch]
        var inp = c.ggml_reshape_3d(ctx0, sam_out, clip_n_patches, ne(sam_out, 2), ne(sam_out, 3));
        inp = c.ggml_cont(ctx0, c.ggml_permute(ctx0, inp, 1, 0, 2, 3));

        var new_pos_embd = model.position_embeddings;

        var n_pos: c_int = @truncate(ne(new_pos_embd, 1)); // +1 for [CLS]
        const tgt_size: c_int = @intFromFloat(@sqrt(@as(f64, @floatFromInt(ne(inp, 1)))));
        const src_size: c_int = @intFromFloat(@sqrt(@as(f64, @floatFromInt(n_pos - 1))));

        if (tgt_size != src_size) {
            const old_pos_embd = c.ggml_view_2d(ctx0, new_pos_embd, ne(new_pos_embd, 0), src_size * src_size, c.ggml_row_size(new_pos_embd.?.type, ne(new_pos_embd, 0)), 0);
            const cls_tok = c.ggml_view_2d(ctx0, new_pos_embd, ne(new_pos_embd, 0), 1, c.ggml_row_size(new_pos_embd.?.type, ne(new_pos_embd, 0)), @intCast(src_size * src_size));
            new_pos_embd = c.ggml_interpolate(ctx0, old_pos_embd, tgt_size, tgt_size, ne(new_pos_embd, 0), 1, c.GGML_SCALE_MODE_BICUBIC);
            new_pos_embd = c.ggml_reshape_3d(ctx0, new_pos_embd, n_embd, tgt_size * tgt_size, 1);
            new_pos_embd = c.ggml_concat(ctx0, new_pos_embd, cls_tok, 1);
            n_pos = tgt_size * tgt_size + 1;
        }

        // add CLS token per batch item
        const cls_embd = c.ggml_repeat_4d(ctx0, model.class_embedding, n_embd, 1, n_batch, 1);
        inp = c.ggml_concat(ctx0, cls_embd, inp, 1);

        // for selecting learned pos embd, used by ViT
        const positions = c.ggml_cast(ctx0, c.ggml_arange(ctx0, 0, @floatFromInt(n_pos), 1), c.GGML_TYPE_I32);
        const learned_pos_embd = c.ggml_get_rows(ctx0, new_pos_embd, positions);

        const cur = g.buildVit(inp, n_pos, .normal, .gelu_quick, learned_pos_embd, null, .{});

        c.ggml_build_forward_expand(g.gf, cur);
        clip_out = cur;
    }

    // sam_out: [patch_h, patch_w, n_embd, n_batch]
    // -> [n_embd, clip_n_patches, n_batch]
    sam_out = c.ggml_cont(ctx0, c.ggml_permute(ctx0, sam_out, 1, 2, 0, 3));
    sam_out = c.ggml_reshape_3d(ctx0, sam_out, ne(sam_out, 0), clip_n_patches, n_batch);

    // clip_out: [n_embd, n_pos, n_batch] where n_pos = clip_n_patches + 1 (CLS)
    // strip CLS token: skip first position, view only the patch tokens
    clip_out = c.ggml_view_3d(ctx0, clip_out, n_embd, clip_n_patches, n_batch, nb(clip_out, 1), nb(clip_out, 2), nb(clip_out, 1));

    var cur = c.ggml_concat(ctx0, clip_out, sam_out, 0);
    cur = c.ggml_mul_mat(ctx0, model.mm_fc_w, cur);
    cur = c.ggml_add(ctx0, cur, model.mm_fc_b);

    if (is_overview) {
        // global view: weave one newline per row + trailing view separator
        const h: c_int = @intFromFloat(sqrtf(@floatFromInt(ne(cur, 1))));
        const w = h;
        const n_dim = ne(cur, 0);

        const imgnl = c.ggml_repeat_4d(ctx0, model.image_newline, n_dim, 1, h, n_batch);
        cur = c.ggml_reshape_4d(ctx0, cur, n_dim, w, h, n_batch);
        cur = c.ggml_reshape_3d(ctx0, c.ggml_concat(ctx0, cur, imgnl, 1), n_dim, (w + 1) * h, n_batch);
        const vs = c.ggml_repeat_4d(ctx0, model.view_seperator, n_dim, 1, n_batch, 1);
        cur = c.ggml_concat(ctx0, cur, vs, 1); // (n_dim, h*(w+1) + 1, n_batch)
    } else {
        // tile row: interleave tiles within each row, add newline per row
        const grid_x: c_int = @intFromFloat(sqrtf(@floatFromInt(clip_n_patches)));
        const grid_y = grid_x;
        const n_dim = ne(cur, 0);

        // merge n_dim into the grid_x axis, freeing the 4th axis for n_rows_batch
        cur = c.ggml_reshape_4d(ctx0, cur, n_dim * grid_x, grid_y, n_tiles_per_row, n_rows_batch);

        // interleave tiles: -> (n_dim*grid_x, n_tiles_per_row, grid_y, n_rows_batch)
        cur = c.ggml_cont(ctx0, c.ggml_permute(ctx0, cur, 0, 2, 1, 3));

        // merge: -> (n_dim, grid_x*n_tiles_per_row, grid_y, n_rows_batch)
        cur = c.ggml_reshape_4d(ctx0, cur, n_dim, grid_x * n_tiles_per_row, grid_y, n_rows_batch);

        // append newline per row
        const imgnl = c.ggml_repeat_4d(ctx0, model.image_newline, n_dim, 1, grid_y, n_rows_batch);
        cur = c.ggml_concat(ctx0, cur, imgnl, 1);

        // flatten: (n_dim, (grid_x*n_tiles_per_row+1)*grid_y, n_rows_batch)
        cur = c.ggml_reshape_3d(ctx0, cur, n_dim, (grid_x * n_tiles_per_row + 1) * grid_y, n_rows_batch);
    }

    g.cb(cur, "dsocr_output", -1);

    c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
