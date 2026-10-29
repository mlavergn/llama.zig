//! The per-projector tensors: the model-specific switch of `load_tensors`,
//! and `clip_n_mmproj_embd`, which reads the projector's output size off
//! them.
//!
//! # Provenance
//!
//! Ports the model-specific switch of `clip_model_loader::load_tensors`
//! and `clip_n_mmproj_embd` from `llama.cpp/tools/mtmd/clip.cpp` at v0.4.1
//! (`b29c606e2`). Each declaration names what it ports and the line it
//! began at.
//!
//! # Order is part of the contract
//!
//! Every `get_tensor` call appends to `tensors_to_load` and charges the
//! device's weight memory, and the data is read in that order; gemma4's
//! clamp scalars are looked up by walking the same list. So the calls here
//! are in the C++'s order, arm by arm, optional ones included.

const std = @import("std");
const impl = @import("impl.zig");
const clip = @import("clip.zig");
const names = @import("names.zig");
const model_mod = @import("model.zig");
const graph = @import("graph.zig");
const c = impl.c;
const gpa = impl.gpa;
const T = graph.T;
const Error = clip.Error;
const TensorLoader = clip.TensorLoader;

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;

const W: [*:0]const u8 = "weight";
const B: [*:0]const u8 = "bias";

/// A string argument for `string_format`.
inline fn s(comptime x: [:0]const u8) [*:0]const u8 {
    return x.ptr;
}

/// An `int` argument for `string_format`.
inline fn ci(v: anytype) c_int {
    return @intCast(v);
}

/// `get_tensor(string_format(fmt, args...), required)`.
fn get(tl: *TensorLoader, comptime fmt: [*:0]const u8, args: anytype, required: bool) Error!T {
    var buf: [256]u8 = undefined;
    _ = @call(.auto, snprintf, .{ &buf, buf.len, fmt } ++ args);
    return tl.getTensor(@ptrCast(&buf), required);
}

/// `std::vector<E>::resize(n)`: existing elements kept, new ones
/// value-initialized.
fn resizeList(comptime E: type, list: *std.ArrayList(E), n: usize) Error!void {
    const old = list.items.len;
    try list.resize(gpa, n);
    if (n > old) {
        for (list.items[old..]) |*e| e.* = .{};
    }
}

/// The clamp-info loop the Gemma4 arms share, at lines 2570 and 3272 of
/// mtmd/clip.cpp: for every tensor loaded so far named `*.weight`, the four
/// Gemma4ClippableLinear bounds, read as scalar tensors where present.
fn loadClampInfo(tl: *TensorLoader) Error!void {
    const model = &tl.ctx_clip.model;
    var a: std.heap.ArenaAllocator = .init(gpa);
    defer a.deinit();
    const ar = a.allocator();
    for (tl.tensors_to_load.items) |tensor| {
        const name = std.mem.sliceTo(&tensor.?.name, 0);
        if (!std.mem.endsWith(u8, name, ".weight")) continue;
        const inp_max = try ar.dupeSentinel(u8, try impl.stringReplaceAll(ar, name, ".weight", ".input_max"), 0);
        const inp_min = try ar.dupeSentinel(u8, try impl.stringReplaceAll(ar, name, ".weight", ".input_min"), 0);
        const out_max = try ar.dupeSentinel(u8, try impl.stringReplaceAll(ar, name, ".weight", ".output_max"), 0);
        const out_min = try ar.dupeSentinel(u8, try impl.stringReplaceAll(ar, name, ".weight", ".output_min"), 0);
        const info: model_mod.ClampInfo = .{
            .inp_max = try tl.getScalar(inp_max.ptr, std.math.floatMax(f32)),
            .inp_min = try tl.getScalar(inp_min.ptr, -std.math.floatMax(f32)),
            .out_max = try tl.getScalar(out_max.ptr, std.math.floatMax(f32)),
            .out_min = try tl.getScalar(out_min.ptr, -std.math.floatMax(f32)),
        };
        if (model.clamp_info_map.getPtr(name)) |v| {
            v.* = info;
        } else {
            const key = try gpa.dupe(u8, name);
            errdefer gpa.free(key);
            try model.clamp_info_map.put(gpa, key, info);
        }
    }
}

/// Ports the model-specific switch of `load_tensors` (mtmd/clip.cpp:2099
/// @b29c606e2), the `switch (model.proj_type)` at its line 2335.
pub fn load(tl: *TensorLoader) Error!void {
    const func = "load_tensors";
    const model = &tl.ctx_clip.model;
    const hparams = &model.hparams;
    const prefix = tl.prefix;

    switch (model.proj_type) {
        .mlp, .mlp_norm => {
            // LLaVA projection
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, false);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, false);
            // Yi-type llava
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, false);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, false);
            // missing in Yi-type llava
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, false);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, false);
            // Yi-type llava
            model.mm_3_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(3), W }, false);
            model.mm_3_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(3), B }, false);
            model.mm_4_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(4), W }, false);
            model.mm_4_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(4), B }, false);
            if (model.mm_3_w != null) {
                // TODO: this is a hack to support Yi-type llava
                model.proj_type = .mlp_norm;
            }
            model.image_newline = try tl.getTensor(names.TN_IMAGE_NEWLINE, false);
        },
        .ldp => {
            // MobileVLM projection
            model.mm_model_mlp_1_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_model_mlp_1_b = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(1), B }, true);
            model.mm_model_mlp_3_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
            model.mm_model_mlp_3_b = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(3), B }, true);
            model.mm_model_block_1_block_0_0_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(0), s("0.weight") }, true);
            model.mm_model_block_1_block_0_1_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(0), s("1.weight") }, true);
            model.mm_model_block_1_block_0_1_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(0), s("1.bias") }, true);
            model.mm_model_block_1_block_1_fc1_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc1.weight") }, true);
            model.mm_model_block_1_block_1_fc1_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc1.bias") }, true);
            model.mm_model_block_1_block_1_fc2_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc2.weight") }, true);
            model.mm_model_block_1_block_1_fc2_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc2.bias") }, true);
            model.mm_model_block_1_block_2_0_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(2), s("0.weight") }, true);
            model.mm_model_block_1_block_2_1_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(2), s("1.weight") }, true);
            model.mm_model_block_1_block_2_1_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(2), s("1.bias") }, true);
            model.mm_model_block_2_block_0_0_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(0), s("0.weight") }, true);
            model.mm_model_block_2_block_0_1_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(0), s("1.weight") }, true);
            model.mm_model_block_2_block_0_1_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(0), s("1.bias") }, true);
            model.mm_model_block_2_block_1_fc1_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc1.weight") }, true);
            model.mm_model_block_2_block_1_fc1_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc1.bias") }, true);
            model.mm_model_block_2_block_1_fc2_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc2.weight") }, true);
            model.mm_model_block_2_block_1_fc2_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc2.bias") }, true);
            model.mm_model_block_2_block_2_0_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(2), s("0.weight") }, true);
            model.mm_model_block_2_block_2_1_w = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(2), s("1.weight") }, true);
            model.mm_model_block_2_block_2_1_b = try get(tl, names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(2), s("1.bias") }, true);
        },
        .ldpv2 => {
            // MobilVLM_V2 projection
            model.mm_model_mlp_0_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_model_mlp_0_b = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(0), B }, true);
            model.mm_model_mlp_2_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(2), W }, true);
            model.mm_model_mlp_2_b = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(2), B }, true);
            model.mm_model_peg_0_w = try get(tl, names.TN_MVLM_PROJ_PEG, .{ ci(0), W }, true);
            model.mm_model_peg_0_b = try get(tl, names.TN_MVLM_PROJ_PEG, .{ ci(0), B }, true);
        },
        .minicpmv => {
            model.mm_model_pos_embed_k = try tl.getTensor(names.TN_MINICPMV_POS_EMBD_K, true);
            model.mm_model_query = try tl.getTensor(names.TN_MINICPMV_QUERY, true);
            model.mm_model_proj = try tl.getTensor(names.TN_MINICPMV_PROJ, true);
            model.mm_model_kv_proj = try tl.getTensor(names.TN_MINICPMV_KV_PROJ, true);
            model.mm_model_attn_q_w = try get(tl, names.TN_MINICPMV_ATTN, .{ s("q"), W }, true);
            model.mm_model_attn_k_w = try get(tl, names.TN_MINICPMV_ATTN, .{ s("k"), W }, true);
            model.mm_model_attn_v_w = try get(tl, names.TN_MINICPMV_ATTN, .{ s("v"), W }, true);
            model.mm_model_attn_q_b = try get(tl, names.TN_MINICPMV_ATTN, .{ s("q"), B }, true);
            model.mm_model_attn_k_b = try get(tl, names.TN_MINICPMV_ATTN, .{ s("k"), B }, true);
            model.mm_model_attn_v_b = try get(tl, names.TN_MINICPMV_ATTN, .{ s("v"), B }, true);
            model.mm_model_attn_o_w = try get(tl, names.TN_MINICPMV_ATTN, .{ s("out"), W }, true);
            model.mm_model_attn_o_b = try get(tl, names.TN_MINICPMV_ATTN, .{ s("out"), B }, true);
            model.mm_model_ln_q_w = try get(tl, names.TN_MINICPMV_LN, .{ s("q"), W }, true);
            model.mm_model_ln_q_b = try get(tl, names.TN_MINICPMV_LN, .{ s("q"), B }, true);
            model.mm_model_ln_kv_w = try get(tl, names.TN_MINICPMV_LN, .{ s("kv"), W }, true);
            model.mm_model_ln_kv_b = try get(tl, names.TN_MINICPMV_LN, .{ s("kv"), B }, true);
            model.mm_model_ln_post_w = try get(tl, names.TN_MINICPMV_LN, .{ s("post"), W }, true);
            model.mm_model_ln_post_b = try get(tl, names.TN_MINICPMV_LN, .{ s("post"), B }, true);
        },
        .minicpmv4_6 => {
            // the `get_merger_tensor` lambda: required only with a 4x merge
            const mr = hparams.n_merge == 4;

            // ViT merger: window self-attention
            model.vit_merger_ln1_w = try get(tl, names.TN_VIT_MERGER_LN1, .{W}, mr);
            model.vit_merger_ln1_b = try get(tl, names.TN_VIT_MERGER_LN1, .{B}, mr);
            model.vit_merger_attn_q_w = try get(tl, names.TN_VIT_MERGER_ATTN_Q, .{W}, mr);
            model.vit_merger_attn_q_b = try get(tl, names.TN_VIT_MERGER_ATTN_Q, .{B}, false);
            model.vit_merger_attn_k_w = try get(tl, names.TN_VIT_MERGER_ATTN_K, .{W}, mr);
            model.vit_merger_attn_k_b = try get(tl, names.TN_VIT_MERGER_ATTN_K, .{B}, false);
            model.vit_merger_attn_v_w = try get(tl, names.TN_VIT_MERGER_ATTN_V, .{W}, mr);
            model.vit_merger_attn_v_b = try get(tl, names.TN_VIT_MERGER_ATTN_V, .{B}, false);
            model.vit_merger_attn_o_w = try get(tl, names.TN_VIT_MERGER_ATTN_O, .{W}, mr);
            model.vit_merger_attn_o_b = try get(tl, names.TN_VIT_MERGER_ATTN_O, .{B}, false);
            // ViT merger: MLP downsample
            model.vit_merger_ds_ln_w = try get(tl, names.TN_VIT_MERGER_DS_LN, .{W}, mr);
            model.vit_merger_ds_ln_b = try get(tl, names.TN_VIT_MERGER_DS_LN, .{B}, mr);
            model.vit_merger_ds_up_w = try get(tl, names.TN_VIT_MERGER_DS_UP, .{W}, mr);
            model.vit_merger_ds_up_b = try get(tl, names.TN_VIT_MERGER_DS_UP, .{B}, false);
            model.vit_merger_ds_down_w = try get(tl, names.TN_VIT_MERGER_DS_DOWN, .{W}, mr);
            model.vit_merger_ds_down_b = try get(tl, names.TN_VIT_MERGER_DS_DOWN, .{B}, false);
            // Final Merger (DownsampleMLP)
            model.mm_input_norm_w = try tl.getTensor(names.TN_MM_INP_NORM, true);
            model.mm_input_norm_b = try tl.getTensor(names.TN_MM_INP_NORM_B, false);
            model.mm_ffn_up_w = try get(tl, names.TN_MM_UP, .{W}, true);
            model.mm_ffn_up_b = try get(tl, names.TN_MM_UP, .{B}, false);
            model.mm_ffn_down_w = try get(tl, names.TN_MM_DOWN, .{W}, true);
            model.mm_ffn_down_b = try get(tl, names.TN_MM_DOWN, .{B}, false);
        },
        .glm_edge => {
            model.mm_model_adapter_conv_w = try get(tl, names.TN_GLM_ADAPER_CONV, .{W}, true);
            model.mm_model_adapter_conv_b = try get(tl, names.TN_GLM_ADAPER_CONV, .{B}, true);
            model.mm_model_mlp_0_w = try get(tl, names.TN_GLM_ADAPTER_LINEAR, .{W}, true);
            model.mm_model_ln_q_w = try get(tl, names.TN_GLM_ADAPTER_NORM_1, .{W}, true);
            model.mm_model_ln_q_b = try get(tl, names.TN_GLM_ADAPTER_NORM_1, .{B}, true);
            model.mm_model_mlp_1_w = try get(tl, names.TN_GLM_ADAPTER_D_H_2_4H, .{W}, true);
            model.mm_model_mlp_2_w = try get(tl, names.TN_GLM_ADAPTER_GATE, .{W}, true);
            model.mm_model_mlp_3_w = try get(tl, names.TN_GLM_ADAPTER_D_4H_2_H, .{W}, true);
            model.mm_boi = try get(tl, names.TN_TOK_GLM_BOI, .{}, true);
            model.mm_eoi = try get(tl, names.TN_TOK_GLM_EOI, .{}, true);
        },
        .qwen2vl, .qwen25vl, .exaone4_5 => {
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .qwen3vl => {
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .mimovl => {
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, false);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, false);
        },
        .minimax_m3 => {
            // per-patch MLP: mm.1 -> gelu -> mm.2
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            // 2x2 merge MLP: mm.merge.fc1 -> gelu -> mm.merge.fc2
            model.mm_merger_fc1_w = try get(tl, names.TN_MM_MERGER_FC1, .{W}, true);
            model.mm_merger_fc1_b = try get(tl, names.TN_MM_MERGER_FC1, .{B}, true);
            model.mm_merger_fc2_w = try get(tl, names.TN_MM_MERGER_FC2, .{W}, true);
            model.mm_merger_fc2_b = try get(tl, names.TN_MM_MERGER_FC2, .{B}, true);
        },
        .muse_glimmer => {
            // 3-linear MLP: fc -> erf-GELU -> proj -> erf-GELU -> vision_proj (into LLM residual dim)
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
        },
        .step3vl => {
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, false);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, false);
            model.mm_model_proj = try get(tl, names.TN_MM_PROJECTOR, .{W}, true);
        },
        .youtuvl => {
            model.mm_input_norm_w = try tl.getTensor(names.TN_MM_INP_NORM, true); // merger.ln_q (RMS norm)
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true); // merger.mlp.0
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true); // merger.mlp.2
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .yasa2 => {
            // reuse tensors already loaded by the common section
            // (TN_PATCH_EMBD and TN_PATCH_BIAS have the same tensor names)
            impl.ggmlAssert(model.patch_embeddings_0 != null, "clip.cpp", 2528, "model.patch_embeddings_0 && \"yasa2 requires v.patch_embd.weight\"");
            model.yasa_patch_w = model.patch_embeddings_0;
            model.yasa_patch_b = model.patch_bias;
            model.yasa_patch_ln_w = try tl.getTensor(names.TN_YASA_PATCH_LN_W, false);
            model.yasa_patch_ln_b = try tl.getTensor(names.TN_YASA_PATCH_LN_B, false);
            model.yasa_backbone_ln_w = try tl.getTensor(names.TN_YASA_BACKBONE_LN_W, false);
            model.yasa_backbone_ln_b = try tl.getTensor(names.TN_YASA_BACKBONE_LN_B, false);
            model.yasa_vision_pos_embed = try tl.getTensor(names.TN_YASA_POS_EMBD, false);
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, false);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, false);

            for (model.yasa_stages.items) |*st| st.blocks.deinit(gpa);
            model.yasa_stages.clearRetainingCapacity();
            var si: c_int = 0;
            while (true) : (si += 1) {
                var stage: model_mod.Yasa2Stage = .{};
                errdefer stage.blocks.deinit(gpa);
                stage.down_ln_w = try get(tl, names.TN_YASA_STAGE_DOWN_LN, .{ si, W }, false);
                stage.down_ln_b = try get(tl, names.TN_YASA_STAGE_DOWN_LN, .{ si, B }, false);
                stage.down_conv_w = try get(tl, names.TN_YASA_STAGE_DOWN_CONV, .{ si, W }, false);
                stage.down_conv_b = try get(tl, names.TN_YASA_STAGE_DOWN_CONV, .{ si, B }, false);

                var bi: c_int = 0;
                while (true) : (bi += 1) {
                    var blk: model_mod.Yasa2Block = .{};
                    blk.dw_w = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("dw"), W }, false);
                    if (blk.dw_w == null) break;
                    blk.dw_b = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("dw"), B }, false);
                    blk.ln_w = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("ln"), W }, false);
                    blk.ln_b = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("ln"), B }, false);
                    blk.pw1_w = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw1"), W }, false);
                    blk.pw1_b = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw1"), B }, false);
                    blk.grn_w = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("grn"), W }, false);
                    blk.grn_b = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("grn"), B }, false);
                    blk.pw2_w = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw2"), W }, false);
                    blk.pw2_b = try get(tl, names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw2"), B }, false);
                    try stage.blocks.append(gpa, blk);
                }

                if (stage.down_conv_w == null and stage.blocks.items.len == 0) {
                    stage.blocks.deinit(gpa);
                    break;
                }
                try model.yasa_stages.append(gpa, stage);
            }
        },
        .glm4v => {
            model.mm_fc_w = try get(tl, names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_ffn_up_w = try get(tl, names.TN_MM_UP, .{W}, true);
            model.mm_ffn_up_b = try get(tl, names.TN_MM_UP, .{B}, false);
            model.mm_ffn_gate_w = try get(tl, names.TN_MM_GATE, .{W}, true);
            model.mm_ffn_gate_b = try get(tl, names.TN_MM_GATE, .{B}, false);
            model.mm_ffn_down_w = try get(tl, names.TN_MM_DOWN, .{W}, true);
            model.mm_ffn_down_b = try get(tl, names.TN_MM_DOWN, .{B}, false);
            model.mm_post_norm_w = try get(tl, names.TN_MM_POST_NORM, .{W}, true);
            model.mm_post_norm_b = try get(tl, names.TN_MM_POST_NORM, .{B}, false);
            model.mm_patch_merger_w = try get(tl, names.TN_MM_PATCH_MERGER, .{W}, true);
            model.mm_patch_merger_b = try get(tl, names.TN_MM_PATCH_MERGER, .{B}, true);
        },
        .gemma3 => {
            model.mm_input_proj_w = try tl.getTensor(names.TN_MM_INP_PROJ, true);
            model.mm_soft_emb_norm_w = try tl.getTensor(names.TN_MM_SOFT_EMB_N, true);
        },
        .gemma4v => {
            model.mm_input_proj_w = try tl.getTensor(names.TN_MM_INP_PROJ, true);
            model.std_bias = try tl.getTensor(names.TN_STD_BIAS, false);
            model.std_scale = try tl.getTensor(names.TN_STD_SCALE, false);
            // load scalar for Gemma4ClippableLinear
            try loadClampInfo(tl);
        },
        .gemma4uv => {
            model.mm_input_proj_w = try tl.getTensor(names.TN_MM_INP_PROJ, true);
            model.patch_norm_1_w = try get(tl, names.TN_PATCH_NORM, .{ ci(1), W }, true);
            model.patch_norm_1_b = try get(tl, names.TN_PATCH_NORM, .{ ci(1), B }, true);
            model.patch_norm_2_w = try get(tl, names.TN_PATCH_NORM, .{ ci(2), W }, true);
            model.patch_norm_2_b = try get(tl, names.TN_PATCH_NORM, .{ ci(2), B }, true);
            model.patch_norm_3_w = try get(tl, names.TN_PATCH_NORM, .{ ci(3), W }, true); // pos_norm
            model.patch_norm_3_b = try get(tl, names.TN_PATCH_NORM, .{ ci(3), B }, true); // pos_norm
        },
        .gemma3nv => {
            model.mobilenet_stem_conv_w = try tl.getTensor(names.TN_MNV5_STEM_CONV, false);
            model.mobilenet_stem_conv_b = try tl.getTensor(names.TN_MNV5_STEM_BIAS, false);
            model.mobilenet_stem_norm_w = try tl.getTensor(names.TN_MNV5_STEM_BN, false);

            model.msfa_ffn_expand_w = try tl.getTensor(names.TN_MNV5_MSFA_FFN_EXP_W, false);
            model.msfa_ffn_expand_bn = try tl.getTensor(names.TN_MNV5_MSFA_FFN_EXP_BN, false); // Consume BN if present but likely folded
            model.msfa_ffn_project_w = try tl.getTensor(names.TN_MNV5_MSFA_FFN_PROJ_W, false);
            model.msfa_ffn_project_bn = try tl.getTensor(names.TN_MNV5_MSFA_FFN_PROJ_BN, false);

            model.msfa_concat_norm_w = try tl.getTensor(names.TN_MNV5_MSFA_NORM, false);

            // Dynamically load blocks stage by stage
            var stage: c_int = 0;
            while (stage < 4) : (stage += 1) {
                var blocks_found_in_stage: c_int = 0;

                var blk_idx: c_int = 0;
                while (true) : (blk_idx += 1) {
                    var found_block = false;
                    var block: model_mod.MobilenetV5Block = .{};

                    // 1. Check for Edge Residual (S0)
                    block.s0_conv_exp_w = try get(tl, names.TN_MNV5_BLK_S0_EXP_W, .{ stage, blk_idx }, false);
                    if (block.s0_conv_exp_w != null) {
                        found_block = true;
                        block.s0_bn1_w = try get(tl, names.TN_MNV5_BLK_S0_BN1_W, .{ stage, blk_idx }, false);
                        block.s0_conv_pwl_w = try get(tl, names.TN_MNV5_BLK_S0_PWL_W, .{ stage, blk_idx }, false);
                        block.s0_bn2_w = try get(tl, names.TN_MNV5_BLK_S0_BN2_W, .{ stage, blk_idx }, false);
                    }
                    // 2. Check for UIR (Universal Inverted Residual)
                    else {
                        // Check for dw_start OR pw_exp (some UIR blocks skip dw_start)
                        block.dw_start_w = try get(tl, names.TN_MNV5_BLK_DW_START_W, .{ stage, blk_idx }, false);
                        block.pw_exp_w = try get(tl, names.TN_MNV5_BLK_PW_EXP_W, .{ stage, blk_idx }, false);

                        if (block.dw_start_w != null or block.pw_exp_w != null) {
                            found_block = true;
                            if (block.dw_start_w != null) {
                                block.dw_start_bn_w = try get(tl, names.TN_MNV5_BLK_DW_START_BN, .{ stage, blk_idx }, false);
                            }
                            if (block.pw_exp_w != null) {
                                block.pw_exp_bn_w = try get(tl, names.TN_MNV5_BLK_PW_EXP_BN, .{ stage, blk_idx }, false);
                            }
                            block.dw_mid_w = try get(tl, names.TN_MNV5_BLK_DW_MID_W, .{ stage, blk_idx }, false);
                            if (block.dw_mid_w != null) {
                                block.dw_mid_bn_w = try get(tl, names.TN_MNV5_BLK_DW_MID_BN, .{ stage, blk_idx }, false);
                            }
                            block.pw_proj_w = try get(tl, names.TN_MNV5_BLK_PW_PROJ_W, .{ stage, blk_idx }, false);
                            if (block.pw_proj_w != null) {
                                block.pw_proj_bn_w = try get(tl, names.TN_MNV5_BLK_PW_PROJ_BN, .{ stage, blk_idx }, false);
                            }
                            block.layer_scale_w = try get(tl, names.TN_MNV5_BLK_LAYER_SCALE, .{ stage, blk_idx }, false);
                        }
                    }

                    // 3. Check for Attention (MQA)
                    // Even if UIR/Edge check failed, this might be a pure attention block
                    const attn_q_check = try get(tl, names.TN_MNV5_ATTN_Q_W, .{ stage, blk_idx }, false);
                    if (attn_q_check != null) {
                        found_block = true;
                        block.attn_q_w = attn_q_check;
                        block.attn_k_w = try get(tl, names.TN_MNV5_ATTN_K_W, .{ stage, blk_idx }, false);
                        block.attn_v_w = try get(tl, names.TN_MNV5_ATTN_V_W, .{ stage, blk_idx }, false);
                        block.attn_o_w = try get(tl, names.TN_MNV5_ATTN_O_W, .{ stage, blk_idx }, false);
                        block.attn_k_dw_w = try get(tl, names.TN_MNV5_ATTN_K_DW, .{ stage, blk_idx }, false);
                        block.attn_k_norm_w = try get(tl, names.TN_MNV5_ATTN_K_NORM, .{ stage, blk_idx }, false);
                        block.attn_v_dw_w = try get(tl, names.TN_MNV5_ATTN_V_DW, .{ stage, blk_idx }, false);
                        block.attn_v_norm_w = try get(tl, names.TN_MNV5_ATTN_V_NORM, .{ stage, blk_idx }, false);
                        block.attn_norm_w = try get(tl, names.TN_MNV5_ATTN_NORM, .{ stage, blk_idx }, false);
                        // Note: Attention blocks also have layer_scale, load it if not already loaded by UIR check
                        if (block.layer_scale_w == null) {
                            block.layer_scale_w = try get(tl, names.TN_MNV5_BLK_LAYER_SCALE, .{ stage, blk_idx }, false);
                        }
                    }

                    if (found_block) {
                        try model.mobilenet_blocks.append(gpa, block);
                        blocks_found_in_stage += 1;
                    } else {
                        // End of blocks for this stage
                        break;
                    }
                }

                // Track where this stage ends in the flat vector
                if (blocks_found_in_stage > 0) {
                    try model.mobilenet_stage_ends.append(gpa, @intCast(model.mobilenet_blocks.items.len - 1));
                    impl.logInf("%s: Stage %d ended at global block index %zu\n", .{ func, stage, model.mobilenet_blocks.items.len - 1 });
                }
            }
            model.mm_input_proj_w = try tl.getTensor(names.TN_MM_INP_PROJ, true);
            model.mm_soft_emb_norm_w = try tl.getTensor(names.TN_MM_SOFT_EMB_N, true);
        },
        .idefics3 => {
            model.mm_fc_w = try get(tl, names.TN_MM_PROJECTOR, .{W}, true);
        },
        .lfm2 => {
            model.mm_input_norm_w = try tl.getTensor(names.TN_MM_INP_NORM, false);
            model.mm_input_norm_b = try tl.getTensor(names.TN_MM_INP_NORM_B, false);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .kimivl, .paddleocr, .kimik25 => {
            model.mm_input_norm_w = try tl.getTensor(names.TN_MM_INP_NORM, true);
            model.mm_input_norm_b = try tl.getTensor(names.TN_MM_INP_NORM_B, true);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .deepseek4v => {
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            // sentinel token embeddings written into the output block
            model.image_newline = try tl.getTensor(names.TN_IMAGE_NEWLINE, true);
            model.token_embd_img_start = try tl.getTensor(names.TN_TOK_IMG_START, true);
            model.token_embd_img_end = try tl.getTensor(names.TN_TOK_IMG_END, true);
            model.token_embd_img_pad = try tl.getTensor(names.TN_TOK_IMG_PAD, true);
        },
        .pixtral => {
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, false);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, false);
            // [IMG_BREAK] token embedding
            model.token_embd_img_break = try tl.getTensor(names.TN_TOK_IMG_BREAK, true);
            // for mistral small 3.1
            model.mm_input_norm_w = try tl.getTensor(names.TN_MM_INP_NORM, false);
            model.mm_patch_merger_w = try get(tl, names.TN_MM_PATCH_MERGER, .{W}, false);
        },
        .lightonocr => {
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, false);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, false);
            model.mm_input_norm_w = try tl.getTensor(names.TN_MM_INP_NORM, false);
            model.mm_patch_merger_w = try get(tl, names.TN_MM_PATCH_MERGER, .{W}, false);
        },
        .dots_ocr, .dots3note_v => {
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            model.mm_input_norm_w = try tl.getTensor(names.TN_MM_INP_NORM, true);
            model.mm_input_norm_b = try tl.getTensor(names.TN_MM_INP_NORM_B, true);
            // post_trunk_norm: applied after all ViT blocks, before the merger
            model.post_ln_w = try get(tl, names.TN_MM_POST_NORM, .{W}, true);
        },
        .dots3note_a => {
            model.conv2d_1_w = try get(tl, names.TN_CONV2D, .{ ci(1), W }, true);
            model.conv2d_1_b = try get(tl, names.TN_CONV2D, .{ ci(1), B }, true);
            model.conv2d_2_w = try get(tl, names.TN_CONV2D, .{ ci(2), W }, true);
            model.conv2d_2_b = try get(tl, names.TN_CONV2D, .{ ci(2), B }, true);
            model.conv2d_3_w = try get(tl, names.TN_CONV2D, .{ ci(3), W }, true);
            model.conv2d_3_b = try get(tl, names.TN_CONV2D, .{ ci(3), B }, true);
            model.conv_out_w = try get(tl, names.TN_CONV_OUT, .{W}, true); // no bias
            // adapter: LayerNorm -> Linear -> GELU -> Linear
            model.mm_norm_pre_w = try get(tl, names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_pre_b = try get(tl, names.TN_MM_NORM_PRE, .{B}, true);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(3), W }, true);
            model.mm_2_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(3), B }, true);
        },
        .ultravox => {
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_norm_pre_w = try get(tl, names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_mid_w = try get(tl, names.TN_MM_NORM_MID, .{W}, true);
        },
        .meralion => {
            // Whisper encoder conv layers
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, names.TN_CONV1D, .{ ci(2), B }, true);
            // MERaLiON adaptor: 4 linear layers + ln_pre
            model.mm_0_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
            model.mm_3_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(3), W }, true);
            model.mm_3_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(3), B }, true);
            // ln_speech (LayerNorm before adaptor)
            model.mm_norm_pre_w = try get(tl, names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_pre_b = try get(tl, names.TN_MM_NORM_PRE, .{B}, true);
        },
        .qwen2a => {
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_fc_w = try get(tl, names.TN_MM_AUDIO_FC, .{W}, true);
            model.mm_fc_b = try get(tl, names.TN_MM_AUDIO_FC, .{B}, true);
        },
        .qwen3a => {
            model.conv2d_1_w = try get(tl, names.TN_CONV2D, .{ ci(1), W }, true);
            model.conv2d_1_b = try get(tl, names.TN_CONV2D, .{ ci(1), B }, true);
            model.conv2d_2_w = try get(tl, names.TN_CONV2D, .{ ci(2), W }, true);
            model.conv2d_2_b = try get(tl, names.TN_CONV2D, .{ ci(2), B }, true);
            model.conv2d_3_w = try get(tl, names.TN_CONV2D, .{ ci(3), W }, true);
            model.conv2d_3_b = try get(tl, names.TN_CONV2D, .{ ci(3), B }, true);
            model.conv_out_w = try get(tl, names.TN_CONV_OUT, .{W}, true); // no bias
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
        },
        .mimo_audio => {
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, names.TN_CONV1D, .{ ci(2), B }, true);
            model.downsample_conv_w = try get(tl, names.TN_A_DOWNSAMPLE_CONV, .{W}, true);
            model.downsample_norm_w = try get(tl, names.TN_A_DOWNSAMPLE_NORM, .{W}, true);
            model.downsample_norm_b = try get(tl, names.TN_A_DOWNSAMPLE_NORM, .{B}, true);
            model.rvq_codebook = try get(tl, names.TN_A_RVQ_CODEBOOK, .{W}, false);
            model.mm_a_code_embd = try get(tl, names.TN_MM_A_CODE_EMBD, .{W}, false);
            if (model.rvq_codebook == null or model.mm_a_code_embd == null) {
                return clip.throwErr("%s: mimo_audio: missing %s or %s\n", .{ func, s(names.TN_A_RVQ_CODEBOOK), s(names.TN_MM_A_CODE_EMBD) });
            }
            // hparams.rvq_codebook_size comes from GGUF metadata and is independent of the
            // tensors' actual shapes - bound it so codebook/code_embd views built from it
            // (mimo-audio.cpp) can never read past either tensor's allocated bins/vocab.
            for (hparams.rvq_codebook_size.items) |bins| {
                if (bins <= 0 or bins > graph.ne(model.rvq_codebook, 1) or bins > graph.ne(model.mm_a_code_embd, 1)) {
                    return clip.throwErr("%s: mimo_audio: %s entry (%d) out of range for codebook/code_embd tensors\n", .{ func, s(names.KEY_A_RVQ_CODEBOOK_SIZE), bins });
                }
            }

            // LLM-side connector: input_local_transformer + projection
            try resizeList(model_mod.Layer, &model.mm_a_local_layers, @intCast(hparams.audio_local_n_layer));
            for (model.mm_a_local_layers.items, 0..) |*layer, ilu| {
                const il: c_int = @intCast(ilu);
                layer.q_w = try get(tl, names.TN_MM_A_LOCAL_ATTN_Q, .{ il, W }, true);
                layer.q_b = try get(tl, names.TN_MM_A_LOCAL_ATTN_Q, .{ il, B }, true);
                layer.k_w = try get(tl, names.TN_MM_A_LOCAL_ATTN_K, .{ il, W }, true);
                layer.k_b = try get(tl, names.TN_MM_A_LOCAL_ATTN_K, .{ il, B }, true);
                layer.v_w = try get(tl, names.TN_MM_A_LOCAL_ATTN_V, .{ il, W }, true);
                layer.v_b = try get(tl, names.TN_MM_A_LOCAL_ATTN_V, .{ il, B }, true);
                layer.o_w = try get(tl, names.TN_MM_A_LOCAL_ATTN_OUT, .{ il, W }, true);
                layer.ff_gate_w = try get(tl, names.TN_MM_A_LOCAL_FFN_GATE, .{ il, W }, true);
                layer.ff_up_w = try get(tl, names.TN_MM_A_LOCAL_FFN_UP, .{ il, W }, true);
                layer.ff_down_w = try get(tl, names.TN_MM_A_LOCAL_FFN_DOWN, .{ il, W }, true);
                layer.ln_1_w = try get(tl, names.TN_MM_A_LOCAL_LN1, .{ il, W }, true);
                layer.ln_2_w = try get(tl, names.TN_MM_A_LOCAL_LN2, .{ il, W }, true);
            }
            model.mm_a_local_norm_w = try get(tl, names.TN_MM_A_LOCAL_NORM, .{W}, true);

            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
        },
        .qwen3tts_spkenc => {
            // stem TDNN (block 0)
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(0), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(0), B }, true);

            // SE-Res2Net blocks (GGUF bid 1..3, one per hparams.n_layer)
            try resizeList(model_mod.Layer, &model.layers, @intCast(hparams.n_layer));
            for (model.layers.items, 0..) |*layer, ilu| {
                const bid: c_int = @intCast(ilu + 1);
                layer.conv_pw1_w = try get(tl, names.TN_CONV_PW1, .{ prefix, bid, W }, true);
                layer.conv_pw1_b = try get(tl, names.TN_CONV_PW1, .{ prefix, bid, B }, true);
                layer.conv_pw2_w = try get(tl, names.TN_CONV_PW2, .{ prefix, bid, W }, true);
                layer.conv_pw2_b = try get(tl, names.TN_CONV_PW2, .{ prefix, bid, B }, true);
                layer.se_conv1_w = try get(tl, names.TN_A_SE_CONV1, .{ bid, W }, true);
                layer.se_conv1_b = try get(tl, names.TN_A_SE_CONV1, .{ bid, B }, true);
                layer.se_conv2_w = try get(tl, names.TN_A_SE_CONV2, .{ bid, W }, true);
                layer.se_conv2_b = try get(tl, names.TN_A_SE_CONV2, .{ bid, B }, true);
                try layer.res2_conv_w.resize(gpa, 7);
                try layer.res2_conv_b.resize(gpa, 7);
                @memset(layer.res2_conv_w.items, null);
                @memset(layer.res2_conv_b.items, null);
                for (0..7) |xid| {
                    layer.res2_conv_w.items[xid] = try get(tl, names.TN_A_CONV_RES2, .{ bid, ci(xid), W }, true);
                    layer.res2_conv_b.items[xid] = try get(tl, names.TN_A_CONV_RES2, .{ bid, ci(xid), B }, true);
                }
            }

            // multi-layer feature aggregation
            model.conv_out_w = try get(tl, names.TN_CONV_OUT, .{W}, true);
            model.conv_out_b = try get(tl, names.TN_CONV_OUT, .{B}, true);

            // attentive statistics pooling
            model.spk_asp_attn_w = try get(tl, names.TN_A_ASP_ATTN, .{W}, true);
            model.spk_asp_attn_b = try get(tl, names.TN_A_ASP_ATTN, .{B}, true);
            model.spk_asp_tdnn_w = try get(tl, names.TN_A_ASP_TDNN, .{W}, true);
            model.spk_asp_tdnn_b = try get(tl, names.TN_A_ASP_TDNN, .{B}, true);

            // final speaker embedding projection
            model.mm_fc_w = try get(tl, names.TN_MM_AUDIO_FC, .{W}, true);
            model.mm_fc_b = try get(tl, names.TN_MM_AUDIO_FC, .{B}, true);
        },
        .pockettts_spkenc => {
            try tl.loadSeanet(&model.seanet, false);
            model.downsample_w = try get(tl, names.TN_A_DOWNSAMPLE_CONV, .{W}, true);
            model.spk_proj_w = try get(tl, names.TN_A_SPEAKER_PROJ, .{W}, true);
        },
        .pockettts_gen => {
            const flow = &model.flow;
            flow.input_proj_w = try get(tl, names.TN_A_GEN_FLOW_INPUT_PROJ, .{W}, true);
            flow.input_proj_b = try get(tl, names.TN_A_GEN_FLOW_INPUT_PROJ, .{B}, true);
            flow.cond_embd_w = try get(tl, names.TN_A_GEN_FLOW_COND_EMBD, .{W}, true);
            flow.cond_embd_b = try get(tl, names.TN_A_GEN_FLOW_COND_EMBD, .{B}, true);
            flow.final_ada_w = try get(tl, names.TN_A_GEN_FLOW_FINAL_ADA, .{W}, true);
            flow.final_ada_b = try get(tl, names.TN_A_GEN_FLOW_FINAL_ADA, .{B}, true);
            flow.final_proj_w = try get(tl, names.TN_A_GEN_FLOW_FINAL_PROJ, .{W}, true);
            flow.final_proj_b = try get(tl, names.TN_A_GEN_FLOW_FINAL_PROJ, .{B}, true);

            try resizeList(model_mod.FlowNet.TimeEmbd, &flow.time, 2);
            for (flow.time.items, 0..) |*t, iu| {
                const i: c_int = @intCast(iu);
                t.freqs = try get(tl, names.TN_A_GEN_FLOW_TIME_FREQS, .{i}, true);
                t.up_w = try get(tl, names.TN_A_GEN_FLOW_TIME_UP, .{ i, W }, true);
                t.up_b = try get(tl, names.TN_A_GEN_FLOW_TIME_UP, .{ i, B }, true);
                t.down_w = try get(tl, names.TN_A_GEN_FLOW_TIME_DOWN, .{ i, W }, true);
                t.down_b = try get(tl, names.TN_A_GEN_FLOW_TIME_DOWN, .{ i, B }, true);
                t.norm = try get(tl, names.TN_A_GEN_FLOW_TIME_NORM, .{i}, true);
            }

            // one AdaLN block per flow depth, the count is only known from the tensors
            var il: c_int = 0;
            while (true) : (il += 1) {
                const probe = try get(tl, names.TN_A_GEN_FLOW_BLK_NORM, .{ il, W }, false);
                if (probe == null) break;
                var blk: model_mod.FlowNet.Block = .{};
                blk.norm_w = probe;
                blk.norm_b = try get(tl, names.TN_A_GEN_FLOW_BLK_NORM, .{ il, B }, true);
                blk.up_w = try get(tl, names.TN_A_GEN_FLOW_BLK_UP, .{ il, W }, true);
                blk.up_b = try get(tl, names.TN_A_GEN_FLOW_BLK_UP, .{ il, B }, true);
                blk.down_w = try get(tl, names.TN_A_GEN_FLOW_BLK_DOWN, .{ il, W }, true);
                blk.down_b = try get(tl, names.TN_A_GEN_FLOW_BLK_DOWN, .{ il, B }, true);
                blk.ada_w = try get(tl, names.TN_A_GEN_FLOW_BLK_ADA, .{ il, W }, true);
                blk.ada_b = try get(tl, names.TN_A_GEN_FLOW_BLK_ADA, .{ il, B }, true);
                try flow.blocks.append(gpa, blk);
            }

            model.gen_out_eos_w = try get(tl, names.TN_A_GEN_OUT_EOS, .{W}, true);
            model.gen_out_eos_b = try get(tl, names.TN_A_GEN_OUT_EOS, .{B}, true);
            model.gen_input_lin_w = try get(tl, names.TN_A_GEN_INPUT_LINEAR, .{W}, true);
            model.gen_emb_mean = try tl.getTensor(names.TN_A_GEN_EMB_MEAN, true);
            model.gen_emb_std = try tl.getTensor(names.TN_A_GEN_EMB_STD, true);

            // mimi decoder
            model.gen_quant_out_w = try get(tl, names.TN_A_GEN_WAV_QUANT_OUT, .{W}, true);
            model.gen_upsample_w = try get(tl, names.TN_A_GEN_WAV_UPSAMPLE, .{W}, true);
            try tl.loadSeanet(&model.seanet, true);
            try resizeList(model_mod.Layer, &model.gen_tfm_layers, @intCast(hparams.n_layer));
            const p = s("a.gen.wav.tfm");
            for (model.gen_tfm_layers.items, 0..) |*layer, ilu| {
                const l: c_int = @intCast(ilu);
                layer.ln_1_w = try get(tl, names.TN_LN_1, .{ p, l, W }, true);
                layer.ln_1_b = try get(tl, names.TN_LN_1, .{ p, l, B }, true);
                layer.q_w = try get(tl, names.TN_ATTN_Q, .{ p, l, W }, true);
                layer.k_w = try get(tl, names.TN_ATTN_K, .{ p, l, W }, true);
                layer.v_w = try get(tl, names.TN_ATTN_V, .{ p, l, W }, true);
                layer.o_w = try get(tl, names.TN_ATTN_OUTPUT, .{ p, l, W }, true);
                layer.ls_1_w = try get(tl, names.TN_LS_1, .{ p, l, W }, true);
                layer.ln_2_w = try get(tl, names.TN_LN_2, .{ p, l, W }, true);
                layer.ln_2_b = try get(tl, names.TN_LN_2, .{ p, l, B }, true);
                layer.ff_up_w = try get(tl, names.TN_FFN_UP, .{ p, l, W }, true);
                layer.ff_down_w = try get(tl, names.TN_FFN_DOWN, .{ p, l, W }, true);
                layer.ls_2_w = try get(tl, names.TN_LS_2, .{ p, l, W }, true);
            }
        },
        .qwen3tts_gen => {
            // code_predictor
            model.gen_code_proj_in_w = try get(tl, names.TN_A_GEN_CODE_PROJ_IN, .{W}, false);
            model.gen_code_proj_in_b = try get(tl, names.TN_A_GEN_CODE_PROJ_IN, .{B}, false);
            model.gen_code_embd_w = try get(tl, names.TN_A_GEN_CODE_EMBD, .{W}, true);
            model.gen_code_head_w = try get(tl, names.TN_A_GEN_CODE_HEAD, .{W}, true);
            model.gen_code_out_embd_w = try get(tl, names.TN_A_GEN_CODE_OUT_EMBD, .{W}, true);
            model.gen_code_norm_w = try get(tl, names.TN_A_GEN_CODE_NORM, .{W}, true);

            // code2wav: RVQ codes -> raw PCM, lives in the same ctx as code_predictor
            const c2w = &model.c2w;

            c2w.quant_first_in_w = try get(tl, names.TN_A_GEN_WAV_QUANT_FIRST_IN, .{W}, true);
            c2w.quant_first_out_w = try get(tl, names.TN_A_GEN_WAV_QUANT_FIRST_OUT, .{W}, true);
            c2w.quant_first_cb_w = try get(tl, names.TN_A_GEN_WAV_QUANT_FIRST_CB, .{W}, true);
            c2w.quant_rest_in_w = try get(tl, names.TN_A_GEN_WAV_QUANT_REST_IN, .{W}, true);
            c2w.quant_rest_out_w = try get(tl, names.TN_A_GEN_WAV_QUANT_REST_OUT, .{W}, true);
            c2w.quant_rest_cb_w = try get(tl, names.TN_A_GEN_WAV_QUANT_REST_CB, .{W}, true);

            c2w.pre_conv_w = try get(tl, names.TN_A_GEN_WAV_PRE_CONV, .{W}, true);
            c2w.pre_conv_b = try get(tl, names.TN_A_GEN_WAV_PRE_CONV, .{B}, true);

            c2w.tfm_in_proj_w = try get(tl, names.TN_A_GEN_WAV_TFM_IN_PROJ, .{W}, true);
            c2w.tfm_in_proj_b = try get(tl, names.TN_A_GEN_WAV_TFM_IN_PROJ, .{B}, true);
            c2w.tfm_out_proj_w = try get(tl, names.TN_A_GEN_WAV_TFM_OUT_PROJ, .{W}, true);
            c2w.tfm_out_proj_b = try get(tl, names.TN_A_GEN_WAV_TFM_OUT_PROJ, .{B}, true);
            c2w.tfm_output_norm_w = try get(tl, names.TN_A_GEN_WAV_TFM_OUT_NORM, .{W}, true);

            // loaded manually, the generic model.layers loop is taken by code_predictor
            try resizeList(model_mod.Layer, &c2w.tfm_layers, @intCast(hparams.wav_tfm_n_layer));
            const p = s("a.gen.wav.tfm");
            for (c2w.tfm_layers.items, 0..) |*layer, ilu| {
                const l: c_int = @intCast(ilu);
                layer.q_w = try get(tl, names.TN_ATTN_Q, .{ p, l, W }, true);
                layer.k_w = try get(tl, names.TN_ATTN_K, .{ p, l, W }, true);
                layer.v_w = try get(tl, names.TN_ATTN_V, .{ p, l, W }, true);
                layer.o_w = try get(tl, names.TN_ATTN_OUTPUT, .{ p, l, W }, true);
                layer.ln_1_w = try get(tl, names.TN_LN_1, .{ p, l, W }, true);
                layer.ln_2_w = try get(tl, names.TN_LN_2, .{ p, l, W }, true);
                layer.ls_1_w = try get(tl, names.TN_LS_1, .{ p, l, W }, true);
                layer.ls_2_w = try get(tl, names.TN_LS_2, .{ p, l, W }, true);
                layer.ff_gate_w = try get(tl, names.TN_FFN_GATE, .{ p, l, W }, true);
                layer.ff_up_w = try get(tl, names.TN_FFN_UP, .{ p, l, W }, true);
                layer.ff_down_w = try get(tl, names.TN_FFN_DOWN, .{ p, l, W }, true);
            }

            // upsample: 2x (causal ConvTranspose1d + ConvNeXt block)
            try resizeList(model_mod.Code2Wav.UpsampleBlock, &c2w.upsample, @intCast(hparams.wav_upsample_n_block));
            for (c2w.upsample.items, 0..) |*up, ilu| {
                const l: c_int = @intCast(ilu);
                up.conv_w = try get(tl, names.TN_A_GEN_WAV_UP_CONV, .{ l, W }, true);
                up.conv_b = try get(tl, names.TN_A_GEN_WAV_UP_CONV, .{ l, B }, true);
                up.dwconv_w = try get(tl, names.TN_A_GEN_WAV_UP_DWCONV, .{ l, W }, true);
                up.dwconv_b = try get(tl, names.TN_A_GEN_WAV_UP_DWCONV, .{ l, B }, true);
                up.norm_w = try get(tl, names.TN_A_GEN_WAV_UP_NORM, .{ l, W }, true);
                up.norm_b = try get(tl, names.TN_A_GEN_WAV_UP_NORM, .{ l, B }, true);
                up.pw1_w = try get(tl, names.TN_A_GEN_WAV_UP_PW1, .{ l, W }, true);
                up.pw1_b = try get(tl, names.TN_A_GEN_WAV_UP_PW1, .{ l, B }, true);
                up.pw2_w = try get(tl, names.TN_A_GEN_WAV_UP_PW2, .{ l, W }, true);
                up.pw2_b = try get(tl, names.TN_A_GEN_WAV_UP_PW2, .{ l, B }, true);
                up.gamma = try get(tl, names.TN_A_GEN_WAV_UP_GAMMA, .{l}, true);
            }

            // DAC decoder: conv_pre + n upsample blocks (each with n_res residual units) + conv_post
            c2w.dac_entry_w = try get(tl, names.TN_A_GEN_WAV_DAC_ENTRY, .{W}, true);
            c2w.dac_entry_b = try get(tl, names.TN_A_GEN_WAV_DAC_ENTRY, .{B}, true);

            try resizeList(model_mod.Code2Wav.DacBlock, &c2w.dac, @intCast(hparams.wav_dac_n_block));
            for (c2w.dac.items, 0..) |*blk, ilu| {
                const l: c_int = @intCast(ilu);
                blk.snake_alpha = try get(tl, names.TN_A_GEN_WAV_DAC_SNAKE, .{ l, s("alpha") }, true);
                blk.snake_beta = try get(tl, names.TN_A_GEN_WAV_DAC_SNAKE, .{ l, s("beta") }, true);
                blk.conv_w = try get(tl, names.TN_A_GEN_WAV_DAC_CONV, .{ l, W }, true);
                blk.conv_b = try get(tl, names.TN_A_GEN_WAV_DAC_CONV, .{ l, B }, true);

                try resizeList(model_mod.Code2Wav.DacRes, &blk.res, @intCast(hparams.wav_dac_n_res));
                for (blk.res.items, 0..) |*res, iru| {
                    const ir: c_int = @intCast(iru);
                    res.act1_alpha = try get(tl, names.TN_A_GEN_WAV_DAC_RES_ACT1, .{ l, ir, s("alpha") }, true);
                    res.act1_beta = try get(tl, names.TN_A_GEN_WAV_DAC_RES_ACT1, .{ l, ir, s("beta") }, true);
                    res.conv1_w = try get(tl, names.TN_A_GEN_WAV_DAC_RES_CONV1, .{ l, ir, W }, true);
                    res.conv1_b = try get(tl, names.TN_A_GEN_WAV_DAC_RES_CONV1, .{ l, ir, B }, true);
                    res.act2_alpha = try get(tl, names.TN_A_GEN_WAV_DAC_RES_ACT2, .{ l, ir, s("alpha") }, true);
                    res.act2_beta = try get(tl, names.TN_A_GEN_WAV_DAC_RES_ACT2, .{ l, ir, s("beta") }, true);
                    res.conv2_w = try get(tl, names.TN_A_GEN_WAV_DAC_RES_CONV2, .{ l, ir, W }, true);
                    res.conv2_b = try get(tl, names.TN_A_GEN_WAV_DAC_RES_CONV2, .{ l, ir, B }, true);
                }
            }

            c2w.dac_post_snake_alpha = try get(tl, names.TN_A_GEN_WAV_DAC_POST_SNAKE, .{s("alpha")}, true);
            c2w.dac_post_snake_beta = try get(tl, names.TN_A_GEN_WAV_DAC_POST_SNAKE, .{s("beta")}, true);
            c2w.dac_post_conv_w = try get(tl, names.TN_A_GEN_WAV_DAC_POST_CONV, .{W}, true);
            c2w.dac_post_conv_b = try get(tl, names.TN_A_GEN_WAV_DAC_POST_CONV, .{B}, true);
        },
        .voxtral => {
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
        },
        .music_flamingo => {
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
        },
        .internvl => {
            model.mm_0_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(1), B }, true);
            model.mm_3_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
            model.mm_3_b = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(3), B }, true);
        },
        .nemotron_v2_vl => {
            model.mm_0_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_1_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_3_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
        },
        .glma => {
            model.conv1d_1_w = try get(tl, names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
            model.mm_norm_pre_w = try get(tl, names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_pre_b = try get(tl, names.TN_MM_NORM_PRE, .{B}, true);
            model.mm_boi = try get(tl, names.TN_TOK_BOI, .{}, true);
            model.mm_eoi = try get(tl, names.TN_TOK_EOI, .{}, true);
        },
        .llama4 => {
            model.mm_model_proj = try get(tl, names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_model_mlp_1_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_model_mlp_2_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(2), W }, true);
        },
        .cogvlm => {
            model.mm_model_proj = try get(tl, names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_post_fc_norm_w = try get(tl, names.TN_MM_POST_FC_NORM, .{W}, true);
            model.mm_post_fc_norm_b = try get(tl, names.TN_MM_POST_FC_NORM, .{B}, true);
            model.mm_h_to_4h_w = try get(tl, names.TN_MM_H_TO_4H, .{W}, true);
            model.mm_gate_w = try get(tl, names.TN_MM_GATE, .{W}, true);
            model.mm_4h_to_h_w = try get(tl, names.TN_MM_4H_TO_H, .{W}, true);
            model.mm_boi = try tl.getTensor(names.TN_TOK_BOI, true);
            model.mm_eoi = try tl.getTensor(names.TN_TOK_EOI, true);
        },
        .hunyuanvl => {
            // proj.0 -> mm.0 (conv1), proj.2 -> mm.2 (conv2), mlp -> mm.model.fc (linear)
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            model.mm_model_proj = try get(tl, names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_model_proj_b = try get(tl, names.TN_MM_PROJECTOR, .{B}, true);
            model.mm_pre_norm_w = try get(tl, names.TN_MM_PRE_NORM, .{W}, true);
            model.mm_post_norm_w = try get(tl, names.TN_MM_POST_NORM, .{W}, true);
            model.mm_img_begin = try tl.getTensor(names.TN_MM_IMG_BEGIN, true);
            model.mm_img_end = try tl.getTensor(names.TN_MM_IMG_END, true);
            model.image_newline = try tl.getTensor(names.TN_IMAGE_NEWLINE, true);
            model.view_seperator = try tl.getTensor(names.TN_IMAGE_SEPERATOR, false);
        },
        .janus_pro => {
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
        },
        .phi4 => {
            model.mm_0_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_2_w = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .deepseekocr, .deepseekocr2 => {
            model.pos_embed = try get(tl, names.TN_SAM_POS_EMBD, .{W}, true);
            model.patch_embed_proj_w = try get(tl, names.TN_SAM_PATCH_EMBD, .{W}, true);
            model.patch_embed_proj_b = try get(tl, names.TN_SAM_PATCH_EMBD, .{B}, true);
            try resizeList(model_mod.Layer, &model.sam_layers, @intCast(model.n_sam_layers));
            for (model.sam_layers.items, 0..) |*layer, ilu| {
                const il: c_int = @intCast(ilu);
                layer.qkv_w = try get(tl, names.TN_SAM_ATTN_QKV, .{ il, W }, true);
                layer.qkv_b = try get(tl, names.TN_SAM_ATTN_QKV, .{ il, B }, true);
                layer.o_w = try get(tl, names.TN_SAM_ATTN_OUT, .{ il, W }, true);
                layer.o_b = try get(tl, names.TN_SAM_ATTN_OUT, .{ il, B }, true);
                layer.ln_1_w = try get(tl, names.TN_SAM_PRE_NORM, .{ il, W }, true);
                layer.ln_1_b = try get(tl, names.TN_SAM_PRE_NORM, .{ il, B }, true);
                layer.ln_2_w = try get(tl, names.TN_SAM_POST_NORM, .{ il, W }, true);
                layer.ln_2_b = try get(tl, names.TN_SAM_POST_NORM, .{ il, B }, true);
                layer.rel_pos_h = try get(tl, names.TN_SAM_ATTN_POS_H, .{ il, W }, true);
                layer.rel_pos_w = try get(tl, names.TN_SAM_ATTN_POS_W, .{ il, W }, true);
                layer.ff_up_w = try get(tl, names.TN_SAM_FFN_UP, .{ il, W }, true);
                layer.ff_up_b = try get(tl, names.TN_SAM_FFN_UP, .{ il, B }, true);
                layer.ff_down_w = try get(tl, names.TN_SAM_FFN_DOWN, .{ il, W }, true);
                layer.ff_down_b = try get(tl, names.TN_SAM_FFN_DOWN, .{ il, B }, true);
            }
            model.neck_0_w = try get(tl, names.TN_SAM_NECK, .{ ci(0), W }, true);
            model.neck_1_b = try get(tl, names.TN_SAM_NECK, .{ ci(1), B }, true);
            model.neck_1_w = try get(tl, names.TN_SAM_NECK, .{ ci(1), W }, true);
            model.neck_2_w = try get(tl, names.TN_SAM_NECK, .{ ci(2), W }, true);
            model.neck_3_b = try get(tl, names.TN_SAM_NECK, .{ ci(3), B }, true);
            model.neck_3_w = try get(tl, names.TN_SAM_NECK, .{ ci(3), W }, true);
            model.net_2 = try get(tl, names.TN_SAM_NET, .{ ci(2), W }, true);
            model.net_3 = try get(tl, names.TN_SAM_NET, .{ ci(3), W }, true);
            model.image_newline = try tl.getTensor(names.TN_IMAGE_NEWLINE, false);
            model.view_seperator = try tl.getTensor(names.TN_IMAGE_SEPERATOR, true);
            model.mm_fc_w = try get(tl, names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_fc_b = try get(tl, names.TN_MM_PROJECTOR, .{B}, true);
            model.resample_query_768 = try get(tl, names.TN_RESMPL_QUERY, .{ ci(768), W }, false);
            model.resample_query_1024 = try get(tl, names.TN_RESMPL_QUERY, .{ ci(1024), W }, false);
        },
        .gemma4a => {
            for (0..2) |iu| {
                const i: c_int = @intCast(iu);
                model.sscp_conv_w[iu] = try get(tl, names.TN_A_CONV1D, .{ i, W }, true);
                model.sscp_conv_b[iu] = try get(tl, names.TN_A_CONV1D, .{ i, B }, false);
                model.sscp_norm_w[iu] = try get(tl, names.TN_A_CONV1D_NORM, .{ i, W }, false);
            }
            model.sscp_inp_proj_w = try get(tl, names.TN_A_INP_PROJ, .{W}, true);
            model.sscp_inp_proj_b = try get(tl, names.TN_A_INP_PROJ, .{B}, false);
            model.audio_out_proj_w = try get(tl, names.TN_A_OUT_PROJ, .{W}, false);
            model.audio_out_proj_b = try get(tl, names.TN_A_OUT_PROJ, .{B}, false);
            // audio multimodal embedder (mm.a.* namespace, not mm.*)
            model.mm_soft_emb_norm_w = try get(tl, names.TN_A_MM_SOFT_EMB_N, .{W}, false);
            model.mm_input_proj_w = try get(tl, names.TN_A_MM_INP_PROJ, .{W}, false);

            // Per-layer tensors NOT loaded by the generic loop above
            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                // Gemma4 audio conformer-specific tensors
                layer.ff_norm_w = try get(tl, names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.attn_pre_norm_w = try get(tl, names.TN_A_ATTN_PRE_NORM, .{ prefix, il, W }, false);
                layer.per_dim_scale_w = try get(tl, names.TN_A_PER_DIM_SCALE, .{ prefix, il, W }, false);
                layer.per_dim_k_scale_w = try get(tl, names.TN_A_PER_DIM_K_SCALE, .{ prefix, il, W }, false);
                layer.attn_k_rel_w = try get(tl, names.TN_A_ATTN_K_REL, .{ prefix, il, W }, false);

                // Convolution module
                // Note: conv_norm / norm_conv are swapped in GGUF due to
                // upstream tensor_mapping.py, so we load them in reverse order
                layer.norm_conv_w = try get(tl, names.TN_CONV_NORM, .{ prefix, il, W }, false);
                layer.norm_conv_b = try get(tl, names.TN_CONV_NORM, .{ prefix, il, B }, false);
                layer.conv_pw1_w = try get(tl, names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, names.TN_CONV_PW1, .{ prefix, il, B }, false);
                layer.conv_dw_w = try get(tl, names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_dw_b = try get(tl, names.TN_CONV_DW, .{ prefix, il, B }, false);
                layer.conv_norm_w = try get(tl, names.TN_NORM_CONV, .{ prefix, il, W }, false);
                layer.conv_norm_b = try get(tl, names.TN_NORM_CONV, .{ prefix, il, B }, false);
                layer.conv_pw2_w = try get(tl, names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, names.TN_CONV_PW2, .{ prefix, il, B }, false);

                // FFN2 (second half-step)
                layer.ff_norm_1_w = try get(tl, names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_up_1_w = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, B }, false);
                layer.ff_down_1_w = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, B }, false);
                layer.ff_post_norm_1_w = try get(tl, names.TN_A_FFN_POST_NORM_1, .{ prefix, il, W }, false);
            }

            // Load clamp info for ClippableLinear AFTER all tensors are loaded
            try loadClampInfo(tl);
        },
        .gemma4ua => {
            model.mm_input_proj_w = try get(tl, names.TN_A_MM_INP_PROJ, .{W}, true);
        },
        .lfm2a => {
            for ([_]usize{ 0, 2, 3, 5, 6 }) |i| {
                model.pre_encode_conv_X_w[i] = try get(tl, names.TN_CONV1D, .{ ci(i), W }, true);
                model.pre_encode_conv_X_b[i] = try get(tl, names.TN_CONV1D, .{ ci(i), B }, true);
            }
            model.pre_encode_out_w = try get(tl, names.TN_PRE_ENCODE_OUT, .{W}, true);
            model.pre_encode_out_b = try get(tl, names.TN_PRE_ENCODE_OUT, .{B}, true);

            model.mm_0_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_3_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(3), W }, true);
            model.mm_3_b = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(3), B }, true);

            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                layer.ff_norm_w = try get(tl, names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.ff_norm_b = try get(tl, names.TN_FFN_NORM, .{ prefix, il, B }, true);
                layer.ff_norm_1_w = try get(tl, names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_norm_1_b = try get(tl, names.TN_FFN_NORM_1, .{ prefix, il, B }, true);
                layer.ff_up_1_w = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, B }, true);
                layer.ff_down_1_w = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, B }, true);

                layer.pos_bias_u = try get(tl, names.TN_POS_BIAS_U, .{ prefix, il }, true);
                layer.pos_bias_v = try get(tl, names.TN_POS_BIAS_V, .{ prefix, il }, true);

                layer.norm_conv_w = try get(tl, names.TN_NORM_CONV, .{ prefix, il, W }, true);
                layer.norm_conv_b = try get(tl, names.TN_NORM_CONV, .{ prefix, il, B }, true);

                layer.linear_pos_w = try get(tl, names.TN_LINEAR_POS, .{ prefix, il, W }, true);

                layer.conv_norm_w = try get(tl, names.TN_CONV_NORM, .{ prefix, il, W }, true);
                layer.conv_norm_b = try get(tl, names.TN_CONV_NORM, .{ prefix, il, B }, true);
                layer.conv_dw_w = try get(tl, names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_dw_b = try get(tl, names.TN_CONV_DW, .{ prefix, il, B }, true);
                layer.conv_pw1_w = try get(tl, names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, names.TN_CONV_PW1, .{ prefix, il, B }, true);
                layer.conv_pw2_w = try get(tl, names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, names.TN_CONV_PW2, .{ prefix, il, B }, true);
            }
        },
        .parakeet => {
            hparams.mel_filters.deinit(gpa);
            hparams.mel_filters = try tl.getVector(names.TN_MEL_FILTERS);
            hparams.window.deinit(gpa);
            hparams.window = try tl.getVector(names.TN_WINDOW);

            // Subsampling layers (conv1d)
            for ([_]usize{ 0, 2, 3, 5, 6 }) |i| {
                model.pre_encode_conv_X_w[i] = try get(tl, names.TN_CONV1D, .{ ci(i), W }, true);
                model.pre_encode_conv_X_b[i] = try get(tl, names.TN_CONV1D, .{ ci(i), B }, true);
            }
            model.pre_encode_out_w = try get(tl, names.TN_PRE_ENCODE_OUT, .{W}, true);
            model.pre_encode_out_b = try get(tl, names.TN_PRE_ENCODE_OUT, .{B}, true);

            // Projection layers
            model.mm_norm_pre_w = try get(tl, names.TN_MM_NORM_PRE, .{W}, false);
            model.mm_0_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(1), W }, false);
            model.mm_1_w = try get(tl, names.TN_MM_AUDIO_MLP, .{ ci(2), W }, false);

            // Encoder layers
            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                // Relative position encoding
                layer.linear_pos_w = try get(tl, names.TN_LINEAR_POS, .{ prefix, il, W }, true);
                layer.pos_bias_u = try get(tl, names.TN_POS_BIAS_U, .{ prefix, il }, true);
                layer.pos_bias_v = try get(tl, names.TN_POS_BIAS_V, .{ prefix, il }, true);

                // Convolution module
                layer.conv_pw1_w = try get(tl, names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, names.TN_CONV_PW1, .{ prefix, il, B }, false);
                layer.conv_dw_w = try get(tl, names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_dw_b = try get(tl, names.TN_CONV_DW, .{ prefix, il, B }, false);
                layer.conv_norm_w = try get(tl, names.TN_CONV_NORM, .{ prefix, il, W }, true);
                layer.conv_norm_b = try get(tl, names.TN_CONV_NORM, .{ prefix, il, B }, true);
                layer.conv_norm_mean = try get(tl, names.TN_CONV_NORM_MEAN, .{ prefix, il }, true);
                layer.conv_norm_var = try get(tl, names.TN_CONV_NORM_VAR, .{ prefix, il }, true);
                layer.conv_pw2_w = try get(tl, names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, names.TN_CONV_PW2, .{ prefix, il, B }, false);

                // Feed-forward networks
                layer.ff_norm_w = try get(tl, names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.ff_norm_b = try get(tl, names.TN_FFN_NORM, .{ prefix, il, B }, true);

                layer.ff_norm_1_w = try get(tl, names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_norm_1_b = try get(tl, names.TN_FFN_NORM_1, .{ prefix, il, B }, true);
                layer.ff_up_1_w = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, B }, false);
                layer.ff_down_1_w = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, B }, false);

                // Layer norms
                layer.norm_conv_w = try get(tl, names.TN_NORM_CONV, .{ prefix, il, W }, true);
                layer.norm_conv_b = try get(tl, names.TN_NORM_CONV, .{ prefix, il, B }, true);
            }

            model.mm_model_mlp_1_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_model_mlp_2_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_model_mlp_3_w = try get(tl, names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
        },
        .granite_speech => {
            model.inp_proj_w = try get(tl, names.TN_INP_PROJ, .{W}, true);
            model.inp_proj_b = try get(tl, names.TN_INP_PROJ, .{B}, true);
            model.ctc_out_w = try get(tl, names.TN_CTC_OUT, .{W}, true);
            model.ctc_out_b = try get(tl, names.TN_CTC_OUT, .{B}, true);
            model.ctc_out_mid_w = try get(tl, names.TN_CTC_OUT_MID, .{W}, true);
            model.ctc_out_mid_b = try get(tl, names.TN_CTC_OUT_MID, .{B}, true);

            // per-layer tensors not loaded by the generic loop above
            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                layer.attn_rel_pos_emb = try get(tl, names.TN_ATTN_REL_POS_EMB, .{ prefix, il }, true);

                layer.ff_norm_w = try get(tl, names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.ff_norm_b = try get(tl, names.TN_FFN_NORM, .{ prefix, il, B }, true);

                layer.ff_norm_1_w = try get(tl, names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_norm_1_b = try get(tl, names.TN_FFN_NORM_1, .{ prefix, il, B }, true);
                layer.ff_up_1_w = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, names.TN_FFN_UP_1, .{ prefix, il, B }, true);
                layer.ff_down_1_w = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, names.TN_FFN_DOWN_1, .{ prefix, il, B }, true);

                layer.norm_conv_w = try get(tl, names.TN_NORM_CONV, .{ prefix, il, W }, true);
                layer.norm_conv_b = try get(tl, names.TN_NORM_CONV, .{ prefix, il, B }, true);
                layer.conv_norm_w = try get(tl, names.TN_CONV_NORM, .{ prefix, il, W }, true);
                layer.conv_norm_b = try get(tl, names.TN_CONV_NORM, .{ prefix, il, B }, true);
                layer.conv_dw_w = try get(tl, names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_pw1_w = try get(tl, names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, names.TN_CONV_PW1, .{ prefix, il, B }, true);
                layer.conv_pw2_w = try get(tl, names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, names.TN_CONV_PW2, .{ prefix, il, B }, true);
            }

            try resizeList(model_mod.QfBlock, &model.qf_proj_blocks, 1);
            const qf = &model.qf_proj_blocks.items[0];
            qf.qf_proj_query = try get(tl, names.TN_QF_PROJ_QUERY, .{prefix}, true);
            qf.qf_proj_norm_w = try get(tl, names.TN_QF_PROJ_NORM, .{ prefix, W }, true);
            qf.qf_proj_norm_b = try get(tl, names.TN_QF_PROJ_NORM, .{ prefix, B }, true);
            qf.qf_proj_linear_w = try get(tl, names.TN_QF_PROJ_LINEAR, .{ prefix, W }, true);
            qf.qf_proj_linear_b = try get(tl, names.TN_QF_PROJ_LINEAR, .{ prefix, B }, true);

            const n_proj_layers = 2;
            try resizeList(model_mod.Layer, &qf.qf_proj_layers, n_proj_layers);
            for (qf.qf_proj_layers.items, 0..) |*pl, ilu| {
                try loadQfLayer(tl, pl, prefix, @intCast(ilu));
            }
        },
        .granite4_vision => {
            // image_newline lives at the top-level.
            model.image_newline = try tl.getTensor(names.TN_IMAGE_NEWLINE, true);

            // Load separate layerwise and spatial projector tensors
            const projector_count = hparams.feature_layers.items.len;
            try resizeList(model_mod.QfBlock, &model.qf_proj_blocks, projector_count);
            for (model.qf_proj_blocks.items, 0..) |*b, bidu| {
                // the C++ passes a `size_t` to `%d`; printf reads its low 32 bits
                const bid: c_int = @truncate(@as(i64, @intCast(bidu)));

                // non-layerwise tensors
                b.qf_proj_img_pos = try get(tl, names.TN_MULTI_PROJ_IMG_POS, .{bid}, true);
                b.qf_proj_query = try get(tl, names.TN_MULTI_PROJ_QUERY, .{ prefix, bid }, true);
                b.qf_proj_linear_w = try get(tl, names.TN_MULTI_PROJ_LINEAR, .{ prefix, bid, W }, true);
                b.qf_proj_linear_b = try get(tl, names.TN_MULTI_PROJ_LINEAR, .{ prefix, bid, B }, true);
                b.qf_proj_norm_w = try get(tl, names.TN_MULTI_PROJ_NORM, .{ prefix, bid, W }, true);
                b.qf_proj_norm_b = try get(tl, names.TN_MULTI_PROJ_NORM, .{ prefix, bid, B }, true);
                b.qf_proj_post_norm_w = try get(tl, names.TN_MULTI_PROJ_POST_NORM, .{ prefix, bid, W }, true);
                b.qf_proj_post_norm_b = try get(tl, names.TN_MULTI_PROJ_POST_NORM, .{ prefix, bid, B }, true);

                // laywerwise tensors
                // NOTE: If any model uses multi-layer qformers, this will need to change
                try resizeList(model_mod.Layer, &b.qf_proj_layers, 1);
                try loadQfLayer(tl, &b.qf_proj_layers.items[0], prefix, bid);
            }
        },
        else => impl.ggmlAssert(false, "clip.cpp", 3581, "false && \"unknown projector type\""),
    }
}

/// One QFormer layer's tensors, as the granite speech and granite4 vision
/// arms load them at lines 3453 and 3509 of clip.cpp, indexed by `il`.
fn loadQfLayer(tl: *TensorLoader, pl: *model_mod.Layer, prefix: [*:0]const u8, il: c_int) Error!void {
    pl.q_w = try get(tl, names.TN_QF_SELF_ATTN_Q, .{ prefix, il, W }, true);
    pl.q_b = try get(tl, names.TN_QF_SELF_ATTN_Q, .{ prefix, il, B }, true);
    pl.k_w = try get(tl, names.TN_QF_SELF_ATTN_K, .{ prefix, il, W }, true);
    pl.k_b = try get(tl, names.TN_QF_SELF_ATTN_K, .{ prefix, il, B }, true);
    pl.v_w = try get(tl, names.TN_QF_SELF_ATTN_V, .{ prefix, il, W }, true);
    pl.v_b = try get(tl, names.TN_QF_SELF_ATTN_V, .{ prefix, il, B }, true);
    pl.o_w = try get(tl, names.TN_QF_SELF_ATTN_O, .{ prefix, il, W }, true);
    pl.o_b = try get(tl, names.TN_QF_SELF_ATTN_O, .{ prefix, il, B }, true);
    pl.ln_1_w = try get(tl, names.TN_QF_SELF_ATTN_N, .{ prefix, il, W }, true);
    pl.ln_1_b = try get(tl, names.TN_QF_SELF_ATTN_N, .{ prefix, il, B }, true);

    pl.cross_attn_q_w = try get(tl, names.TN_QF_CROSS_ATTN_Q, .{ prefix, il, W }, true);
    pl.cross_attn_q_b = try get(tl, names.TN_QF_CROSS_ATTN_Q, .{ prefix, il, B }, true);
    pl.cross_attn_k_w = try get(tl, names.TN_QF_CROSS_ATTN_K, .{ prefix, il, W }, true);
    pl.cross_attn_k_b = try get(tl, names.TN_QF_CROSS_ATTN_K, .{ prefix, il, B }, true);
    pl.cross_attn_v_w = try get(tl, names.TN_QF_CROSS_ATTN_V, .{ prefix, il, W }, true);
    pl.cross_attn_v_b = try get(tl, names.TN_QF_CROSS_ATTN_V, .{ prefix, il, B }, true);
    pl.cross_attn_o_w = try get(tl, names.TN_QF_CROSS_ATTN_O, .{ prefix, il, W }, true);
    pl.cross_attn_o_b = try get(tl, names.TN_QF_CROSS_ATTN_O, .{ prefix, il, B }, true);
    pl.cross_attn_norm_w = try get(tl, names.TN_QF_CROSS_ATTN_N, .{ prefix, il, W }, true);
    pl.cross_attn_norm_b = try get(tl, names.TN_QF_CROSS_ATTN_N, .{ prefix, il, B }, true);

    pl.ff_up_w = try get(tl, names.TN_QF_FFN_UP, .{ prefix, il, W }, true);
    pl.ff_up_b = try get(tl, names.TN_QF_FFN_UP, .{ prefix, il, B }, true);
    pl.ff_down_w = try get(tl, names.TN_QF_FFN_DOWN, .{ prefix, il, W }, true);
    pl.ff_down_b = try get(tl, names.TN_QF_FFN_DOWN, .{ prefix, il, B }, true);
    pl.ln_2_w = try get(tl, names.TN_QF_FFN_NORM, .{ prefix, il, W }, true);
    pl.ln_2_b = try get(tl, names.TN_QF_FFN_NORM, .{ prefix, il, B }, true);
}

/// `t->ne[i]` as the C++'s `int` return reads it.
inline fn neI(t: T, i: usize) c_int {
    return @truncate(graph.ne(t, i));
}

/// Ports `clip_n_mmproj_embd` (mtmd/clip.cpp:5911 @b29c606e2): the embedding
/// size of the projector's output, which must be the text model's.
pub fn nMmprojEmbd(ctx: *const clip.Ctx) c_int {
    const m = &ctx.model;
    return switch (m.proj_type) {
        .ldp => neI(m.mm_model_block_1_block_2_1_b, 0),
        .ldpv2 => neI(m.mm_model_peg_0_b, 0),
        .mlp, .phi4, .pixtral, .lightonocr, .dots_ocr, .dots3note_v, .dots3note_a => neI(m.mm_2_w, 1),
        .mlp_norm => neI(m.mm_3_b, 0),
        .minicpmv => neI(m.mm_model_proj, 0),
        .minicpmv4_6 => neI(m.mm_ffn_down_w, 1),
        .glm_edge => neI(m.mm_model_mlp_3_w, 1),
        .minimax_m3 => neI(m.mm_merger_fc2_b, 0),
        .muse_glimmer => neI(m.mm_2_w, 1),
        .qwen2vl, .qwen25vl, .exaone4_5, .janus_pro, .youtuvl => neI(m.mm_1_b, 0),
        // main path + deepstack paths
        .qwen3vl => @truncate(graph.ne(m.mm_1_b, 0) * (1 + @as(i64, m.n_deepstack_layers))),
        .mimovl => neI(m.mm_1_w, 1),
        .step3vl => neI(m.mm_model_proj, 1),
        .gemma3, .gemma3nv => neI(m.mm_input_proj_w, 0),
        .gemma4v, .gemma4uv, .gemma4a, .gemma4ua => neI(m.mm_input_proj_w, 1),
        .idefics3 => neI(m.mm_fc_w, 1),
        .ultravox, .voxtral, .music_flamingo => neI(m.mm_2_w, 1),
        .meralion => neI(m.mm_3_w, 1), // out_proj output dim
        .internvl, .nemotron_v2_vl => neI(m.mm_3_w, 1),
        .llama4 => neI(m.mm_model_proj, 1),
        .qwen2a => neI(m.mm_fc_w, 1),
        .qwen3a => neI(m.mm_2_w, 1),
        .glma, .lfm2, .kimivl, .paddleocr, .kimik25, .yasa2, .deepseek4v => neI(m.mm_2_w, 1),
        .hunyuanvl => neI(m.mm_model_proj, 1),
        .cogvlm => neI(m.mm_4h_to_h_w, 1),
        .deepseekocr, .deepseekocr2 => neI(m.mm_fc_w, 1),
        .lfm2a => neI(m.position_embeddings, 0),
        .granite_speech => neI(m.qf_proj_blocks.items[0].qf_proj_linear_w, 1),
        // `size_t * int`, narrowed to the `int` return
        .granite4_vision => @truncate(@as(i64, @intCast(m.qf_proj_blocks.items.len *% @as(usize, @bitCast(@as(i64, m.hparams.projection_dim)))))),
        .glm4v => neI(m.mm_ffn_down_w, 1),
        .mimo_audio => neI(m.mm_2_w, 1),
        .qwen3tts_spkenc => neI(m.mm_fc_w, 2),
        .qwen3tts_gen => neI(m.gen_code_out_embd_w, 0),
        .pockettts_spkenc => neI(m.spk_proj_w, 1),
        .pockettts_gen => neI(m.gen_input_lin_w, 1),
        .parakeet => neI(m.mm_1_w, 1),
        else => impl.abort("clip.cpp", 6011, "Unknown projector type"),
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
