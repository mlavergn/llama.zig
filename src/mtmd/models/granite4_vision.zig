//! The Granite Vision 4.1 encoder graph: a SigLIP tower over a stack of
//! anyres tiles, WindowQFormer projector blocks reading chosen layers
//! (deepstack and spatial), their outputs concatenated, then the tiles
//! assembled into one token sequence with a newline per row.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/granite4-vision.cpp` and the
//! `clip_graph_granite4_vision` declaration in
//! `llama.cpp/tools/mtmd/models/models.h`, at v0.4.0 (`5266f24da`). Each
//! declaration names the C++ it replaces and the line it began at.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;
const nb = graph.nb;

/// Ports `clip_graph_granite4_vision` (tools/mtmd/models/models.h:412 @5266f24da): the
/// members the derived graph adds to `clip_graph`.
const Granite4Vision = struct {
    g: *graph.Graph,
    // the input image is a stack of tiles on the Y axis: [overview, tile(0,0), tile(0,1), ...]
    anyres: impl.AnyresInfo,
    n_tiles: c_int,
    tile_side: c_int, // patches per tile side

    /// Ports `build_tile_inp` (granite4-vision.cpp:25 @5266f24da): split
    /// the stacked tiles into the batch axis, then run the usual patch
    /// embedding.
    fn buildTileInp(self: *const Granite4Vision) T {
        const g = self.g;
        const ctx0 = g.ctx0;
        var inp_raw = g.buildInpRaw(3);

        if (self.n_tiles > 1) {
            const px = g.img.nx();
            inp_raw = c.ggml_reshape_4d(ctx0, inp_raw, px * px, self.n_tiles, 3, 1);
            inp_raw = c.ggml_cont(ctx0, c.ggml_permute(ctx0, inp_raw, 0, 2, 1, 3));
            inp_raw = c.ggml_reshape_4d(ctx0, inp_raw, px, px, 3, self.n_tiles);
        }

        var inp = c.ggml_conv_2d(ctx0, g.model.patch_embeddings_0, inp_raw, g.patch_size, g.patch_size, 0, 0, 1, 1);
        inp = c.ggml_reshape_3d(ctx0, inp, self.tile_side * self.tile_side, g.n_embd, self.n_tiles);
        inp = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp));
        if (g.model.patch_bias != null) inp = c.ggml_add(ctx0, inp, g.model.patch_bias);
        return inp;
    }

    /// Ports `gather` (granite4-vision.cpp:44 @5266f24da).
    fn gather(self: *const Granite4Vision, src: T, name: [*:0]const u8, idx_len: c_int) T {
        const ctx0 = self.g.ctx0;
        // one index row per tile, all rows hold the same permutation
        const idx = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_I32, idx_len, self.n_tiles);
        _ = c.ggml_set_name(idx, name);
        c.ggml_set_input(idx);
        return c.ggml_get_rows(ctx0, src, idx);
    }

    /// Ports `interp_down` (granite4-vision.cpp:55 @5266f24da).
    fn interpDown(self: *const Granite4Vision, src: T, side: c_int, new_side: c_int) T {
        const ctx0 = self.g.ctx0;
        const n_embd: c_int = @truncate(ne(src, 0));
        var t = c.ggml_reshape_4d(ctx0, src, n_embd, side, side, self.n_tiles);
        t = c.ggml_cont(ctx0, c.ggml_permute(ctx0, t, 2, 0, 1, 3));
        // fold the tile axis into the channel axis, ggml_pool_2d only pools the first two axes
        t = c.ggml_reshape_3d(ctx0, t, side, side, n_embd * self.n_tiles);
        const kernel = @divTrunc(side, new_side);
        t = c.ggml_pool_2d(ctx0, t, c.GGML_OP_POOL_AVG, kernel, kernel, kernel, kernel, 0, 0);
        t = c.ggml_reshape_4d(ctx0, t, new_side, new_side, n_embd, self.n_tiles);
        t = c.ggml_cont(ctx0, c.ggml_permute(ctx0, t, 1, 2, 0, 3));
        return c.ggml_reshape_3d(ctx0, t, n_embd, new_side * new_side, self.n_tiles);
    }

    /// The `cbx` lambda (granite4-vision.cpp:94 @5266f24da) of `build_block`.
    fn cbx(t: T, bid: c_int, step: []const u8) void {
        var buf: [256]u8 = undefined;
        const name = std.fmt.bufPrintSentinel(&buf, "g4v_blk{d}_{s}", .{ bid, step }, 0) catch unreachable;
        _ = c.ggml_set_name(t, name.ptr);
    }

    /// The `linear` lambda (granite4-vision.cpp:145 @5266f24da) of
    /// `build_block`: a projection with the windows batched.
    fn linear(self: *const Granite4Vision, x: T, w: T, b: T) T {
        const ctx0 = self.g.ctx0;
        var t = c.ggml_reshape_2d(ctx0, x, ne(x, 0), ne(x, 1) * ne(x, 2));
        t = self.g.buildMm(w, t);
        if (b != null) t = c.ggml_add(ctx0, t, b);
        return t;
    }

    /// Ports `build_block` (granite4-vision.cpp:75 @5266f24da): one
    /// WindowQFormer block.
    fn buildBlock(self: *const Granite4Vision, blk: *const model_mod.QfBlock, h: T, bid: c_int, spatial_offset: c_int, image_side: c_int, window_side: c_int, query_side: c_int, qformer_eps: f32) T {
        const g = self.g;
        const ctx0 = g.ctx0;
        const n_tiles = self.n_tiles;
        var name_buf: [256]u8 = undefined;

        const n_embd: c_int = @truncate(ne(h, 0));
        impl.ggmlAssert(ne(h, 1) == image_side * image_side, "granite4-vision.cpp", 86, "h->ne[1] == image_side * image_side");
        const n = @divTrunc(image_side, window_side);
        const new_side = n * query_side;
        const n_windows = n * n;
        const n_win_all = n_windows * n_tiles; // windows of every tile, batched together
        const enc_len = window_side * window_side;
        const query_len = query_side * query_side;

        // 1. Top-level LN
        cbx(h, bid, "inp");
        const x = g.buildNorm(h, blk.qf_proj_norm_w, blk.qf_proj_norm_b, .normal, g.eps, bid);
        cbx(x, bid, "norm");

        // 2. enc = _win(x, image_side, window_side)
        var enc: T = undefined;
        {
            const enc_flat = self.gather(x, std.fmt.bufPrintSentinel(&name_buf, "g4v_blk{d}_win_idx", .{bid}, 0) catch unreachable, image_side * image_side);
            enc = c.ggml_reshape_3d(ctx0, enc_flat, n_embd, enc_len, n_win_all);
        }
        cbx(enc, bid, "enc");

        // 3. downsampled = downsampler(x)
        var d: T = undefined;
        if (spatial_offset >= 0) {
            d = self.gather(x, std.fmt.bufPrintSentinel(&name_buf, "g4v_blk{d}_spatial_idx", .{bid}, 0) catch unreachable, new_side * new_side);
        } else {
            d = self.interpDown(x, image_side, new_side);
        }
        cbx(d, bid, "downsampled");

        // 4. query_embeds = query + _win(d, new_side, query_side)
        var q_in: T = undefined;
        {
            const dw_flat = self.gather(d, std.fmt.bufPrintSentinel(&name_buf, "g4v_blk{d}_qwin_idx", .{bid}, 0) catch unreachable, new_side * new_side);
            const dw = c.ggml_reshape_3d(ctx0, dw_flat, n_embd, query_len, n_win_all);
            q_in = c.ggml_add(ctx0, dw, blk.qf_proj_query);
        }
        cbx(q_in, bid, "query_embeds");

        // 5. encoder_embeds = enc + image_positions → (C, enc_len, n_windows)
        const e_in = c.ggml_add(ctx0, enc, blk.qf_proj_img_pos);
        cbx(e_in, bid, "encoder_embeds");

        // 6. Qformer forward.
        const q = g.buildNorm(q_in, blk.qf_proj_post_norm_w, blk.qf_proj_post_norm_b, .normal, qformer_eps, bid);

        // Get the single QFormer layer
        impl.ggmlAssert(blk.qf_proj_layers.items.len == 1, "granite4-vision.cpp", 153, "blk.qf_proj_layers.size() == 1");
        const pl = &blk.qf_proj_layers.items[0];

        // 6a. Self-attention
        var sa_out: T = undefined;
        {
            const d_h: c_int = 64;
            const n_head = @divTrunc(n_embd, d_h);
            const nq: c_int = @truncate(ne(q, 1));
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_h)));

            var Q = self.linear(q, pl.q_w, pl.q_b);
            var K = self.linear(q, pl.k_w, pl.k_b);
            var V = self.linear(q, pl.v_w, pl.v_b);

            Q = c.ggml_reshape_4d(ctx0, Q, d_h, n_head, nq, n_win_all);
            K = c.ggml_reshape_4d(ctx0, K, d_h, n_head, nq, n_win_all);
            V = c.ggml_reshape_4d(ctx0, V, d_h, n_head, nq, n_win_all);

            sa_out = g.buildAttn(pl.o_w, pl.o_b, Q, K, V, null, scale, bid, null);
            sa_out = c.ggml_reshape_3d(ctx0, sa_out, n_embd, nq, n_win_all);

            sa_out = c.ggml_add(ctx0, sa_out, q);
            sa_out = g.buildNorm(sa_out, pl.ln_1_w, pl.ln_1_b, .normal, qformer_eps, bid);
        }
        cbx(sa_out, bid, "sa_out");

        // 6b. Cross-attention
        var ca_out: T = undefined;
        {
            const d_h: c_int = 64;
            const n_head = @divTrunc(n_embd, d_h);
            const nq: c_int = @truncate(ne(sa_out, 1));
            const nkv: c_int = @truncate(ne(e_in, 1));
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_h)));

            var Q = self.linear(sa_out, pl.cross_attn_q_w, pl.cross_attn_q_b);
            var K = self.linear(e_in, pl.cross_attn_k_w, pl.cross_attn_k_b);
            var V = self.linear(e_in, pl.cross_attn_v_w, pl.cross_attn_v_b);

            Q = c.ggml_reshape_4d(ctx0, Q, d_h, n_head, nq, n_win_all);
            K = c.ggml_reshape_4d(ctx0, K, d_h, n_head, nkv, n_win_all);
            V = c.ggml_reshape_4d(ctx0, V, d_h, n_head, nkv, n_win_all);

            ca_out = g.buildAttn(pl.cross_attn_o_w, pl.cross_attn_o_b, Q, K, V, null, scale, bid, null);
            ca_out = c.ggml_reshape_3d(ctx0, ca_out, n_embd, nq, n_win_all);

            ca_out = c.ggml_add(ctx0, ca_out, sa_out);
            ca_out = g.buildNorm(ca_out, pl.cross_attn_norm_w, pl.cross_attn_norm_b, .normal, qformer_eps, bid);
        }
        cbx(ca_out, bid, "ca_out");

        // 6c. FFN
        var ffn: T = undefined;
        {
            var t = c.ggml_reshape_2d(ctx0, ca_out, n_embd, query_len * n_win_all);
            t = g.buildMm(pl.ff_up_w, t);
            if (pl.ff_up_b != null) t = c.ggml_add(ctx0, t, pl.ff_up_b);
            t = c.ggml_gelu_erf(ctx0, t);
            t = g.buildMm(pl.ff_down_w, t);
            if (pl.ff_down_b != null) t = c.ggml_add(ctx0, t, pl.ff_down_b);
            t = c.ggml_reshape_3d(ctx0, t, n_embd, query_len, n_win_all);
            ffn = c.ggml_add(ctx0, t, ca_out);
            ffn = g.buildNorm(ffn, pl.ln_2_w, pl.ln_2_b, .normal, qformer_eps, bid);
        }
        cbx(ffn, bid, "qformer_out");

        // 7. _unwin back to raster
        var unwinned: T = undefined;
        {
            const flat = c.ggml_reshape_3d(ctx0, ffn, n_embd, query_len * n_windows, n_tiles);
            unwinned = self.gather(flat, std.fmt.bufPrintSentinel(&name_buf, "g4v_blk{d}_unwin_idx", .{bid}, 0) catch unreachable, new_side * new_side);
        }
        cbx(unwinned, bid, "unwin");

        // 8. out_linear
        var out = g.buildMm(blk.qf_proj_linear_w, unwinned);
        if (blk.qf_proj_linear_b != null) out = c.ggml_add(ctx0, out, blk.qf_proj_linear_b);
        cbx(out, bid, "out");

        return out;
    }

    /// Ports `build_newline_row` (granite4-vision.cpp:247 @5266f24da): the
    /// K-tiled newline row, shape (n_mmproj_embd, 1).
    fn buildNewlineRow(self: *const Granite4Vision, ctx0: ?*c.ggml_context) T {
        const g = self.g;
        const model = g.model;
        const K: c_int = @intCast(model.qf_proj_blocks.items.len);
        impl.ggmlAssert(K > 0, "granite4-vision.cpp", 249, "K > 0");
        impl.ggmlAssert(@rem(g.n_mmproj_embd, K) == 0, "granite4-vision.cpp", 250, "n_mmproj_embd % K == 0");
        const projection_dim = @divTrunc(g.n_mmproj_embd, K);
        impl.ggmlAssert(model.image_newline != null, "granite4-vision.cpp", 252, "model.image_newline != nullptr");
        impl.ggmlAssert(c.ggml_nelements(model.image_newline) == projection_dim, "granite4-vision.cpp", 253, "ggml_nelements(model.image_newline) == projection_dim");

        // Build newline_row[k*projection_dim + d] = nl[d] * (k == 0 ? base : 1.0)
        const nl = model.image_newline; // (projection_dim,)
        const nl_first_2d = c.ggml_reshape_2d(ctx0, nl, projection_dim, 1);
        var nl_row_2d: T = undefined;
        if (K == 1) {
            nl_row_2d = nl_first_2d;
        } else {
            const nl_2d = c.ggml_reshape_2d(ctx0, nl, projection_dim, 1);
            const rest_template = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, projection_dim, K - 1);
            const nl_rest = c.ggml_repeat(ctx0, nl_2d, rest_template);
            nl_row_2d = c.ggml_concat(ctx0, nl_first_2d, nl_rest, 1); // (projection_dim, K)
        }
        nl_row_2d = c.ggml_cont(ctx0, nl_row_2d);
        return c.ggml_reshape_2d(ctx0, nl_row_2d, g.n_mmproj_embd, 1);
    }

    /// Ports `build_anyres_assembly` (granite4-vision.cpp:275 @5266f24da):
    /// the overview tokens first, then the tile grid in raster order with
    /// one newline per row.
    fn buildAnyresAssembly(self: *const Granite4Vision, cur: T, out_side: c_int) T {
        const ctx0 = self.g.ctx0;
        const n_dim: c_int = @truncate(ne(cur, 0));
        const grid_x = self.anyres.grid_x;
        const grid_y = self.anyres.grid_y;
        const cur_w = grid_x * out_side;
        const cur_h = grid_y * out_side;
        impl.ggmlAssert(ne(cur, 1) == out_side * out_side, "granite4-vision.cpp", 281, "cur->ne[1] == out_side * out_side");
        impl.ggmlAssert(ne(cur, 2) == 1 + grid_x * grid_y, "granite4-vision.cpp", 282, "cur->ne[2] == 1 + grid_x * grid_y");

        const base = c.ggml_view_2d(ctx0, cur, n_dim, out_side * out_side, nb(cur, 1), 0);

        var tiles = c.ggml_view_3d(ctx0, cur, n_dim, out_side * out_side, grid_x * grid_y, nb(cur, 1), nb(cur, 2), nb(cur, 2));

        // (n_dim*out_side, out_side, grid_x, grid_y) -> interleave the tiles of a grid row
        tiles = c.ggml_reshape_4d(ctx0, tiles, n_dim * out_side, out_side, grid_x, grid_y);
        tiles = c.ggml_cont(ctx0, c.ggml_permute(ctx0, tiles, 0, 2, 1, 3));
        tiles = c.ggml_reshape_3d(ctx0, tiles, n_dim, cur_w, cur_h);

        // drop the tokens that only cover the padding added when resizing to the grid
        const up = impl.anyresUnpad(cur_w, cur_h, self.anyres.orig_nx, self.anyres.orig_ny);
        const off_x = up.off_x;
        const off_y = up.off_y;
        const w = up.out_w;
        const h = up.out_h;
        if (w != cur_w or h != cur_h) {
            tiles = c.ggml_cont(ctx0, c.ggml_view_3d(ctx0, tiles, n_dim, w, h, nb(tiles, 1), nb(tiles, 2), @as(usize, @intCast(off_x)) * nb(tiles, 1) + @as(usize, @intCast(off_y)) * nb(tiles, 2)));
        }

        const nl = c.ggml_repeat_4d(ctx0, self.buildNewlineRow(ctx0), n_dim, 1, h, 1);
        tiles = c.ggml_concat(ctx0, tiles, nl, 1);
        tiles = c.ggml_reshape_2d(ctx0, tiles, n_dim, (w + 1) * h);

        return c.ggml_concat(ctx0, base, tiles, 1);
    }
};

/// Ports `build` (granite4-vision.cpp:310 @5266f24da), a member of `clip_graph_granite4_vision`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    // the member initializers of `clip_graph_granite4_vision` (tools/mtmd/models/models.h:413 @5266f24da)
    const self: Granite4Vision = .{
        .g = g,
        .anyres = g.img.anyres,
        .n_tiles = @divTrunc(g.img.ny(), g.img.nx()),
        .tile_side = @divTrunc(g.img.nx(), g.patch_size),
    };
    const n_tiles = self.n_tiles;
    const tile_side = self.tile_side;

    impl.ggmlAssert(model.patch_embeddings_0 != null, "granite4-vision.cpp", 311, "model.patch_embeddings_0 != nullptr");
    impl.ggmlAssert(model.position_embeddings != null, "granite4-vision.cpp", 312, "model.position_embeddings != nullptr");
    impl.ggmlAssert(model.class_embedding == null, "granite4-vision.cpp", 313, "model.class_embedding == nullptr");
    impl.ggmlAssert(model.qf_proj_blocks.items.len != 0, "granite4-vision.cpp", 314, "!model.qf_proj_blocks.empty()");

    // --- Stage 1a: SigLIP encoder producing intermediate hidden states ---
    var inp = self.buildTileInp();
    inp = c.ggml_add(ctx0, inp, model.position_embeddings);
    g.cb(inp, "pos_embed", -1);

    const tile_n_patches = tile_side * tile_side;

    var inpL = inp;
    const layer_outs = impl.gpa.alloc(T, @intCast(g.n_layer)) catch return error.OutOfMemory;
    defer impl.gpa.free(layer_outs);
    @memset(layer_outs, null);

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        var cur = inpL;

        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .normal, g.eps, il);

        // Self-attention
        var Qcur = g.buildMm(layer.q_w, cur);
        if (layer.q_b != null) Qcur = c.ggml_add(ctx0, Qcur, layer.q_b);
        var Kcur = g.buildMm(layer.k_w, cur);
        if (layer.k_b != null) Kcur = c.ggml_add(ctx0, Kcur, layer.k_b);
        var Vcur = g.buildMm(layer.v_w, cur);
        if (layer.v_b != null) Vcur = c.ggml_add(ctx0, Vcur, layer.v_b);

        Qcur = c.ggml_reshape_4d(ctx0, Qcur, g.d_head, g.n_head, tile_n_patches, n_tiles);
        Kcur = c.ggml_reshape_4d(ctx0, Kcur, g.d_head, g.n_head, tile_n_patches, n_tiles);
        Vcur = c.ggml_reshape_4d(ctx0, Vcur, g.d_head, g.n_head, tile_n_patches, n_tiles);

        cur = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, null, g.kq_scale, il, null);
        cur = c.ggml_reshape_3d(ctx0, cur, g.n_embd, tile_n_patches, n_tiles);

        cur = c.ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .normal, g.eps, il);
        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
        cur = c.ggml_add(ctx0, inpL, cur);
        g.cb(cur, "layer_out", il);
        layer_outs[@intCast(il)] = cur;
        inpL = cur;
    }

    // --- Stage 1b/1c: WindowQFormer blocks ---
    const projector_count: c_int = @intCast(hparams.feature_layers.items.len);
    const qformer_eps: f32 = 1e-12;

    var mmproj: T = null;
    var bid: c_int = 0;
    while (bid < projector_count) : (bid += 1) {
        const blk = &model.qf_proj_blocks.items[@intCast(bid)];

        const vlayer: c_int = hparams.feature_layers.items[@intCast(bid)];
        impl.ggmlAssert(vlayer >= 0 and vlayer < g.n_layer, "granite4-vision.cpp", 372, "vlayer >= 0 && vlayer < n_layer");
        const h = layer_outs[@intCast(vlayer)];

        const stream = self.buildBlock(blk, h, bid, hparams.proj_spatial_offsets.items[@intCast(bid)], tile_side, hparams.downsample_window_side, hparams.downsample_query_side, qformer_eps);
        var name_buf: [256]u8 = undefined;
        g.cb(stream, std.fmt.bufPrintSentinel(&name_buf, "proj_{d}_v_out", .{bid}, 0) catch unreachable, vlayer);
        mmproj = if (mmproj != null) c.ggml_concat(ctx0, mmproj, stream, 0) else stream;
    }

    // --- Stage 1d: assemble the tiles and weave in the newline tokens ---
    if (self.anyres.isTiled()) {
        const out_side = @divTrunc(tile_side, hparams.downsample_window_side) * hparams.downsample_query_side;
        mmproj = self.buildAnyresAssembly(mmproj, out_side);
        _ = c.ggml_set_name(mmproj, "g4v_mmproj_out_anyres");
    } else {
        _ = c.ggml_set_name(mmproj, "g4v_mmproj_out");
    }
    c.ggml_build_forward_expand(g.gf, mmproj);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
