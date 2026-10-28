//! The MiMoVL vision tower of MiMo-V2.5: a Qwen2.5-VL-shaped ViT with GQA,
//! per-head attention sinks on windowed layers, and per-layer row or column
//! window attention.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/mimovl.cpp` at v0.4.0 (`5266f24da`).

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const c = impl.c;

extern fn sqrtf(x: f32) f32;

/// Ports `build_mm` (mimovl.cpp:3 @5266f24da), a member of
/// `clip_graph_mimovl`: every matmul at F32 precision.
pub fn buildMm(g: *const graph.Graph, w: graph.T, x: graph.T) graph.T {
    const cur = c.ggml_mul_mat(g.ctx0, w, x);
    c.ggml_mul_mat_set_prec(cur, c.GGML_PREC_F32);
    return cur;
}

/// The `reorder` lambda (mimovl.cpp:100 @5266f24da): permutes patches at
/// merge-unit granularity, keeping each group's (h, w) order.
fn reorder(g: *graph.Graph, x: graph.T, idx: graph.T, merge_unit: c_int, n_units: c_int, n_pos: c_int, batch_size: c_int) graph.T {
    var y = c.ggml_reshape_2d(g.ctx0, x, g.n_embd * merge_unit, n_units);
    y = c.ggml_get_rows(g.ctx0, y, idx);
    return c.ggml_reshape_3d(g.ctx0, y, g.n_embd, n_pos, batch_size);
}

/// Ports `build` (mimovl.cpp:22 @5266f24da), a member of `clip_graph_mimovl`.
pub fn build(g: *graph.Graph, params: ?*const clip.EncodeParams) clip.Error!?*c.ggml_cgraph {
    _ = params;
    g.mm_hook = buildMm;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const eps = g.eps;
    const n_head = g.n_head;
    const n_embd = g.n_embd;
    const patch_size = g.patch_size;
    impl.ggmlAssert(model.patch_embeddings_0 != null, "mimovl.cpp", 23, "model.patch_embeddings_0 != nullptr");
    impl.ggmlAssert(model.patch_embeddings_1 != null, "mimovl.cpp", 24, "model.patch_embeddings_1 != nullptr");
    impl.ggmlAssert(model.class_embedding == null, "mimovl.cpp", 25, "model.class_embedding == nullptr");
    impl.ggmlAssert(hparams.n_head_kv > 0, "mimovl.cpp", 26, "hparams.n_head_kv > 0");
    impl.ggmlAssert(@rem(n_head, hparams.n_head_kv) == 0, "mimovl.cpp", 27, "n_head % hparams.n_head_kv == 0");
    impl.ggmlAssert(@as(c_int, @intCast(hparams.wa_pattern_mode.items.len)) == g.n_layer, "mimovl.cpp", 28, "(int) hparams.wa_pattern_mode.size() == n_layer");

    const batch_size = 1;
    const n_pos = g.n_patches;
    const n_head_kv: c_int = hparams.n_head_kv;
    const merge: c_int = if (hparams.n_merge > 0) hparams.n_merge else 2;
    const merge_unit = merge * merge;
    const n_units = @divTrunc(n_pos, merge_unit);
    impl.ggmlAssert(n_units * merge_unit == n_pos, "mimovl.cpp", 36, "n_units * merge_unit == n_pos");

    // n_embd is not n_head*head_dim here: derive head_dim from the fused QKV
    // projection, rows = (n_head + 2*n_head_kv)*head_dim
    impl.ggmlAssert(model.layers.items[0].qkv_w != null, "mimovl.cpp", 41, "model.layers[0].qkv_w != nullptr");
    const qkv_rows: c_int = @intCast(graph.ne(model.layers.items[0].qkv_w, 1));
    const head_dim = @divTrunc(qkv_rows, n_head + 2 * n_head_kv);
    impl.ggmlAssert(head_dim * (n_head + 2 * n_head_kv) == qkv_rows, "mimovl.cpp", 44, "head_dim * (n_head + 2 * n_head_kv) == qkv_rows");
    const attn_scale: f32 = 1.0 / sqrtf(@floatFromInt(head_dim));
    const rope_n_dims = @divTrunc(head_dim, 2);
    var mrope_sections = [4]c_int{ @divTrunc(rope_n_dims, 2), @divTrunc(rope_n_dims, 2), 0, 0 };

    // Patch embed: Conv3D(kt=2) split into two Conv2D, then interleave-merge
    // along the height axis to match the merge-tile token order.
    const inp_raw = g.buildInpRaw(3);
    var inp: graph.T = c.ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    {
        const inp_1 = c.ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = c.ggml_add(ctx0, inp, inp_1);

        impl.ggmlAssert(@rem(g.img.nx(), patch_size * 2) == 0, "mimovl.cpp", 59, "img.nx() % (patch_size * 2) == 0");
        impl.ggmlAssert(@rem(g.img.ny(), patch_size * 2) == 0, "mimovl.cpp", 60, "img.ny() % (patch_size * 2) == 0");

        inp = c.ggml_permute(ctx0, inp, 1, 2, 0, 3); // [w,h,c,b] -> [c,w,h,b]
        inp = c.ggml_cont_4d(ctx0, inp, n_embd * 2, @divTrunc(g.n_patches_x, 2), g.n_patches_y, batch_size);
        inp = c.ggml_reshape_4d(ctx0, inp, n_embd * 2, @divTrunc(g.n_patches_x, 2), 2, batch_size * @divTrunc(g.n_patches_y, 2));
        inp = c.ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = c.ggml_cont_3d(ctx0, inp, n_embd, g.n_patches_x * g.n_patches_y, batch_size);
    }
    g.cb(inp, "patch_embed", -1);

    const positions_row = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos * 4);
    _ = c.ggml_set_name(positions_row, "mimovl_positions_row");
    c.ggml_set_input(positions_row);

    const positions_col = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_I32, n_pos * 4);
    _ = c.ggml_set_name(positions_col, "mimovl_positions_col");
    c.ggml_set_input(positions_col);

    // idx_col is the col-major merge-unit permutation, taken as F32 so its
    // inverse can be derived in-graph with ggml_argsort, and cast back to I32
    // for ggml_get_rows.
    const idx_col_f = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, n_units);
    _ = c.ggml_set_name(idx_col_f, "mimovl_idx_col");
    c.ggml_set_input(idx_col_f);
    const idx_col = c.ggml_cast(ctx0, idx_col_f, c.GGML_TYPE_I32);
    const idx_col_inv = c.ggml_argsort(ctx0, idx_col_f, c.GGML_SORT_ORDER_ASC);

    const window_mask = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_pos, n_pos);
    _ = c.ggml_set_name(window_mask, "mimovl_window_mask");
    c.ggml_set_input(window_mask);

    const window_mask_attn = if (g.flash_attn_type == .enabled)
        c.ggml_cast(ctx0, window_mask, c.GGML_TYPE_F16)
    else
        window_mask;

    var inpL = inp;
    var prev_mode: i32 = -1;

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        const mode = hparams.wa_pattern_mode.items[@intCast(il)];
        const is_full = (mode == -1);
        const is_col = (mode == 1);

        // Reorder transitions on entry/exit of a col-mode run.
        if (is_col and prev_mode != 1) {
            inpL = reorder(g, inpL, idx_col, merge_unit, n_units, n_pos, batch_size);
            g.cb(inpL, "reorder_to_col", il);
        } else if (!is_col and prev_mode == 1) {
            inpL = reorder(g, inpL, idx_col_inv, merge_unit, n_units, n_pos, batch_size);
            g.cb(inpL, "reorder_to_row", il);
        }

        var cur = inpL;

        // Pre-attention RMSNorm.
        cur = g.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, .rms, eps, il);
        g.cb(cur, "ln1", il);

        // Fused QKV with GQA.
        var qkv = g.buildMm(layer.qkv_w, cur);
        qkv = c.ggml_add(ctx0, qkv, layer.qkv_b);

        const row = c.ggml_row_size(qkv.?.type, head_dim);
        const off_k = c.ggml_row_size(qkv.?.type, n_head * head_dim);
        const off_v = c.ggml_row_size(qkv.?.type, (n_head + n_head_kv) * head_dim);

        var Qcur = c.ggml_view_3d(ctx0, qkv, head_dim, n_head, n_pos, row, graph.nb(qkv, 1), 0);
        var Kcur = c.ggml_view_3d(ctx0, qkv, head_dim, n_head_kv, n_pos, row, graph.nb(qkv, 1), off_k);
        const Vcur = c.ggml_view_3d(ctx0, qkv, head_dim, n_head_kv, n_pos, row, graph.nb(qkv, 1), off_v);

        g.cb(Qcur, "Qcur", il);
        g.cb(Kcur, "Kcur", il);
        g.cb(Vcur, "Vcur", il);

        // 2D RoPE
        const pos = if (is_col) positions_col else positions_row;
        Qcur = c.ggml_rope_multi(ctx0, Qcur, pos, null, rope_n_dims, &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000.0, 1.0, 0.0, 1.0, 32.0, 1.0);
        Kcur = c.ggml_rope_multi(ctx0, Kcur, pos, null, rope_n_dims, &mrope_sections, c.GGML_ROPE_TYPE_VISION, 32768, 10000.0, 1.0, 0.0, 1.0, 32.0, 1.0);
        g.cb(Qcur, "Qcur_rope", il);
        g.cb(Kcur, "Kcur_rope", il);

        // Full layers: plain attention. Windowed layers: banded mask and per-head sinks.
        const mask = if (is_full) null else window_mask_attn;
        const sinks = if (is_full) null else layer.attn_sinks;
        if (!is_full) impl.ggmlAssert(layer.attn_sinks != null, "mimovl.cpp", 157, "layer.attn_sinks != nullptr");
        const attn_out = g.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, mask, attn_scale, il, sinks);
        g.cb(attn_out, "attn_out", il);

        // Residual 1.
        cur = c.ggml_add(ctx0, attn_out, inpL);
        inpL = cur;
        g.cb(cur, "ffn_inp", il);

        // Pre-FFN RMSNorm.
        cur = g.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, .rms, eps, il);
        g.cb(cur, "ffn_inp_normed", il);

        // SwiGLU MLP with biases
        cur = g.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
        g.cb(cur, "ffn_out", il);

        // Residual 2.
        cur = c.ggml_add(ctx0, inpL, cur);
        g.cb(cur, "layer_out", il);

        inpL = cur;
        prev_mode = mode;
    }

    // If the last block was col-mode, undo the transpose so the merger sees patches in row order.
    if (prev_mode == 1) {
        inpL = reorder(g, inpL, idx_col_inv, merge_unit, n_units, n_pos, batch_size);
        g.cb(inpL, "reorder_to_row_final", -1);
    }

    // Merger: post-LayerNorm
    inpL = g.buildNorm(inpL, model.post_ln_w, model.post_ln_b, .normal, 1e-6, g.n_layer);
    g.cb(inpL, "post_ln", -1);

    // Spatial merge: pack each merge_unit of patches into a single
    // (n_embd*merge_unit)-wide row, then run the 2-layer MLP.
    var embeddings = c.ggml_reshape_3d(ctx0, inpL, n_embd * merge_unit, n_units, batch_size);
    embeddings = g.buildFfn(embeddings, model.mm_0_w, null, null, null, model.mm_1_w, null, .gelu, -1);
    g.cb(embeddings, "vit_out", -1);

    c.ggml_build_forward_expand(g.gf, embeddings);
    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
