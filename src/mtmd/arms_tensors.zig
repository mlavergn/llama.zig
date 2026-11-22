//! The per-projector tensors: the model-specific switch of `load_tensors`,
//! and `mtmd_model::n_mmproj_embd`, which reads the projector's output size off
//! them.
//!
//! # Provenance
//!
//! Ports the model-specific switch of `mtmd_model_loader::load_tensors`
//! and `mtmd_model::n_mmproj_embd` from `llama.cpp/tools/mtmd/mtmd-model.cpp`
//! at v1.0.0 (`b1b770d90`) -- `clip.cpp` until v0.9.0. Each declaration names what it ports and the line it
//! began at.
//!
//! # Order is part of the contract
//!
//! Every `get_tensor` call appends to `tensors_to_load` and charges the
//! device's weight memory, and the data is read in that order; gemma4's
//! clamp scalars are looked up by walking the same list. So the calls here
//! are in the C++'s order, arm by arm, optional ones included.

const std = @import("std");
const llama = @import("llama");
const mod = @import("module.zig");
const stdlib = @import("stdlib");

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
fn get(tl: *mod.clip.TensorLoader, comptime fmt: [*:0]const u8, args: anytype, required: bool) mod.clip.Error!mod.graph.T {
    var buf: [256]u8 = undefined;
    _ = @call(.auto, stdlib.libc.snprintf, .{ &buf, buf.len, fmt } ++ args);
    return tl.getTensor(@ptrCast(&buf), required);
}

/// `std::vector<E>::resize(n)`: existing elements kept, new ones
/// value-initialized.
fn resizeList(comptime E: type, list: *std.ArrayList(E), n: usize) mod.clip.Error!void {
    const old = list.items.len;
    try list.resize(std.heap.c_allocator, n);
    if (n > old) {
        for (list.items[old..]) |*e| e.* = .{};
    }
}

/// The clamp-info loop the Gemma4 arms share, at lines 2004 and 3112 of
/// mtmd-model.cpp: for every tensor loaded so far named `*.weight`, the four
/// Gemma4ClippableLinear bounds, read as scalar tensors where present.
fn loadClampInfo(tl: *mod.clip.TensorLoader) mod.clip.Error!void {
    const model = tl.model;
    var a: std.heap.ArenaAllocator = .init(std.heap.c_allocator);
    defer a.deinit();
    const ar = a.allocator();
    for (tl.tensors_to_load.items) |tensor| {
        const name = std.mem.sliceTo(&tensor.?.name, 0);
        if (!std.mem.endsWith(u8, name, ".weight")) continue;
        const inp_max = try ar.dupeSentinel(u8, try mod.impl.stringReplaceAll(ar, name, ".weight", ".input_max"), 0);
        const inp_min = try ar.dupeSentinel(u8, try mod.impl.stringReplaceAll(ar, name, ".weight", ".input_min"), 0);
        const out_max = try ar.dupeSentinel(u8, try mod.impl.stringReplaceAll(ar, name, ".weight", ".output_max"), 0);
        const out_min = try ar.dupeSentinel(u8, try mod.impl.stringReplaceAll(ar, name, ".weight", ".output_min"), 0);
        const info: mod.model.ClampInfo = .{
            .inp_max = try tl.getScalar(inp_max.ptr, std.math.floatMax(f32)),
            .inp_min = try tl.getScalar(inp_min.ptr, -std.math.floatMax(f32)),
            .out_max = try tl.getScalar(out_max.ptr, std.math.floatMax(f32)),
            .out_min = try tl.getScalar(out_min.ptr, -std.math.floatMax(f32)),
        };
        if (model.clamp_info_map.getPtr(name)) |v| {
            v.* = info;
        } else {
            const key = try std.heap.c_allocator.dupe(u8, name);
            errdefer std.heap.c_allocator.free(key);
            try model.clamp_info_map.put(std.heap.c_allocator, key, info);
        }
    }
}

/// `get_tensor(a + b, required)` for the chatterbox speaker encoder's
/// string-concatenated names.
fn getCat(tl: *mod.clip.TensorLoader, a: []const u8, b: []const u8, required: bool) mod.clip.Error!mod.graph.T {
    var buf: [256]u8 = undefined;
    const name = std.fmt.bufPrintSentinel(&buf, "{s}{s}", .{ a, b }, 0) catch return mod.clip.throwStr("chatterbox: tensor name too long");
    return tl.getTensor(name.ptr, required);
}

/// The `load_bn` lambda of the CHATTERBOX_SPKENC arm (mtmd-model.cpp:2835
/// @b1b770d90).
fn cbxLoadBn(tl: *mod.clip.TensorLoader, p: []const u8, n: *mod.model.Chatterbox.Bn) mod.clip.Error!void {
    n.mean = try getCat(tl, p, ".running_mean", true);
    n.@"var" = try getCat(tl, p, ".running_var", true);
    n.w = try getCat(tl, p, ".weight", false);
    n.b = try getCat(tl, p, ".bias", false);
}

/// The `load_res2d` lambda of the CHATTERBOX_SPKENC arm (mtmd-model.cpp:2841
/// @b1b770d90).
fn cbxLoadRes2d(tl: *mod.clip.TensorLoader, p: []const u8, r: *mod.model.Chatterbox.SpkRes2d) mod.clip.Error!void {
    var buf: [256]u8 = undefined;
    r.conv1_w = try getCat(tl, p, ".conv1.weight", true);
    try cbxLoadBn(tl, std.fmt.bufPrint(&buf, "{s}.bn1", .{p}) catch unreachable, &r.bn1);
    r.conv2_w = try getCat(tl, p, ".conv2.weight", true);
    try cbxLoadBn(tl, std.fmt.bufPrint(&buf, "{s}.bn2", .{p}) catch unreachable, &r.bn2);
    r.shortcut_w = try getCat(tl, p, ".shortcut.0.weight", false);
    if (r.shortcut_w != null) {
        try cbxLoadBn(tl, std.fmt.bufPrint(&buf, "{s}.shortcut.1", .{p}) catch unreachable, &r.shortcut_bn);
    }
}

/// The `load_cam_block` lambda of the CHATTERBOX_SPKENC arm
/// (mtmd-model.cpp:2851 @b1b770d90).
fn cbxLoadCamBlock(tl: *mod.clip.TensorLoader, bp: []const u8, tp: []const u8, blk: *mod.model.Chatterbox.SpkCamBlock) mod.clip.Error!void {
    var pbuf: [200]u8 = undefined;
    var nbuf: [256]u8 = undefined;
    var li: c_int = 1;
    while (true) : (li += 1) {
        const probe = std.fmt.bufPrintSentinel(&nbuf, "{s}.tdnnd{d}.linear1.weight", .{ bp, li }, 0) catch unreachable;
        if (llama.impl.llama_c.gguf_find_tensor(tl.loader.ctx_gguf, probe.ptr) < 0) break;
        const p = std.fmt.bufPrint(&pbuf, "{s}.tdnnd{d}", .{ bp, li }) catch unreachable;
        var l: mod.model.Chatterbox.SpkCamLayer = .{};
        try cbxLoadBn(tl, std.fmt.bufPrint(&nbuf, "{s}.nonlinear1.batchnorm", .{p}) catch unreachable, &l.nl1_bn);
        l.linear1_w = try getCat(tl, p, ".linear1.weight", true);
        try cbxLoadBn(tl, std.fmt.bufPrint(&nbuf, "{s}.nonlinear2.batchnorm", .{p}) catch unreachable, &l.nl2_bn);
        l.local_w = try getCat(tl, p, ".cam_layer.linear_local.weight", true);
        l.ctx1_w = try getCat(tl, p, ".cam_layer.linear1.weight", true);
        l.ctx1_b = try getCat(tl, p, ".cam_layer.linear1.bias", true);
        l.ctx2_w = try getCat(tl, p, ".cam_layer.linear2.weight", true);
        l.ctx2_b = try getCat(tl, p, ".cam_layer.linear2.bias", true);
        try blk.layers.append(std.heap.c_allocator, l);
    }
    try cbxLoadBn(tl, std.fmt.bufPrint(&nbuf, "{s}.nonlinear.batchnorm", .{tp}) catch unreachable, &blk.transit_bn);
    blk.transit_w = try getCat(tl, tp, ".linear.weight", true);
}

/// The `has` lambda of the CHATTERBOX arm (mtmd-model.cpp:2577 @b1b770d90),
/// over a name `std::to_string` and `+` would build.
fn cbxHas(tl: *mod.clip.TensorLoader, comptime fmt: []const u8, args: anytype) bool {
    var buf: [256]u8 = undefined;
    const name = std.fmt.bufPrintSentinel(&buf, fmt, args, 0) catch return false;
    return llama.impl.llama_c.gguf_find_tensor(tl.loader.ctx_gguf, name.ptr) >= 0;
}

/// The `load_enc_layer` lambda of the CHATTERBOX arm (mtmd-model.cpp:2580
/// @b1b770d90).
fn cbxLoadEncLayer(tl: *mod.clip.TensorLoader, p: []const u8, l: *mod.model.Chatterbox.EncLayer) mod.clip.Error!void {
    l.norm_mha_w = try getCat(tl, p, ".norm_mha.weight", true);
    l.norm_mha_b = try getCat(tl, p, ".norm_mha.bias", true);
    l.attn_q_w = try getCat(tl, p, ".self_attn.linear_q.weight", true);
    l.attn_q_b = try getCat(tl, p, ".self_attn.linear_q.bias", true);
    l.attn_k_w = try getCat(tl, p, ".self_attn.linear_k.weight", true);
    l.attn_k_b = try getCat(tl, p, ".self_attn.linear_k.bias", true);
    l.attn_v_w = try getCat(tl, p, ".self_attn.linear_v.weight", true);
    l.attn_v_b = try getCat(tl, p, ".self_attn.linear_v.bias", true);
    l.attn_pos_w = try getCat(tl, p, ".self_attn.linear_pos.weight", true);
    l.attn_pos_bias_u = try getCat(tl, p, ".self_attn.pos_bias_u", true);
    l.attn_pos_bias_v = try getCat(tl, p, ".self_attn.pos_bias_v", true);
    l.attn_out_w = try getCat(tl, p, ".self_attn.linear_out.weight", true);
    l.attn_out_b = try getCat(tl, p, ".self_attn.linear_out.bias", true);
    l.norm_ff_w = try getCat(tl, p, ".norm_ff.weight", true);
    l.norm_ff_b = try getCat(tl, p, ".norm_ff.bias", true);
    l.ffn_1_w = try getCat(tl, p, ".feed_forward.w_1.weight", true);
    l.ffn_1_b = try getCat(tl, p, ".feed_forward.w_1.bias", true);
    l.ffn_2_w = try getCat(tl, p, ".feed_forward.w_2.weight", true);
    l.ffn_2_b = try getCat(tl, p, ".feed_forward.w_2.bias", true);
}

/// The `load_causal` lambda of the CHATTERBOX arm (mtmd-model.cpp:2601
/// @b1b770d90).
fn cbxLoadCausal(tl: *mod.clip.TensorLoader, p: []const u8, b: *mod.model.Chatterbox.CausalBlock) mod.clip.Error!void {
    b.conv_w = try getCat(tl, p, ".block.0.weight", true);
    b.conv_b = try getCat(tl, p, ".block.0.bias", true);
    b.norm_w = try getCat(tl, p, ".block.2.weight", true);
    b.norm_b = try getCat(tl, p, ".block.2.bias", true);
}

/// The `load_resnet` lambda of the CHATTERBOX arm (mtmd-model.cpp:2607
/// @b1b770d90).
fn cbxLoadResnet(tl: *mod.clip.TensorLoader, p: []const u8, r: *mod.model.Chatterbox.Resnet) mod.clip.Error!void {
    var buf: [200]u8 = undefined;
    try cbxLoadCausal(tl, std.fmt.bufPrint(&buf, "{s}.block1", .{p}) catch unreachable, &r.block1);
    try cbxLoadCausal(tl, std.fmt.bufPrint(&buf, "{s}.block2", .{p}) catch unreachable, &r.block2);
    r.mlp_w = try getCat(tl, p, ".mlp.1.weight", true);
    r.mlp_b = try getCat(tl, p, ".mlp.1.bias", true);
    r.res_conv_w = try getCat(tl, p, ".res_conv.weight", true);
    r.res_conv_b = try getCat(tl, p, ".res_conv.bias", true);
}

/// The `load_tfm` lambda of the CHATTERBOX arm (mtmd-model.cpp:2615
/// @b1b770d90).
fn cbxLoadTfm(tl: *mod.clip.TensorLoader, p: []const u8, b: *mod.model.Chatterbox.TfmBlock) mod.clip.Error!void {
    b.norm1_w = try getCat(tl, p, ".norm1.weight", true);
    b.norm1_b = try getCat(tl, p, ".norm1.bias", true);
    b.attn_q_w = try getCat(tl, p, ".attn1.to_q.weight", true);
    b.attn_k_w = try getCat(tl, p, ".attn1.to_k.weight", true);
    b.attn_v_w = try getCat(tl, p, ".attn1.to_v.weight", true);
    b.attn_out_w = try getCat(tl, p, ".attn1.to_out.0.weight", true);
    b.attn_out_b = try getCat(tl, p, ".attn1.to_out.0.bias", true);
    b.norm3_w = try getCat(tl, p, ".norm3.weight", true);
    b.norm3_b = try getCat(tl, p, ".norm3.bias", true);
    b.ff_in_w = try getCat(tl, p, ".ff.net.0.proj.weight", true);
    b.ff_in_b = try getCat(tl, p, ".ff.net.0.proj.bias", true);
    b.ff_out_w = try getCat(tl, p, ".ff.net.2.weight", true);
    b.ff_out_b = try getCat(tl, p, ".ff.net.2.bias", true);
}

/// The `load_stage` lambda of the CHATTERBOX arm (mtmd-model.cpp:2630
/// @b1b770d90).
fn cbxLoadStage(tl: *mod.clip.TensorLoader, p: []const u8, s_: *mod.model.Chatterbox.EstStage, boundary: bool) mod.clip.Error!void {
    var buf: [200]u8 = undefined;
    try cbxLoadResnet(tl, std.fmt.bufPrint(&buf, "{s}.0", .{p}) catch unreachable, &s_.res);
    var j: c_int = 0;
    while (cbxHas(tl, "{s}.1.{d}.norm1.weight", .{ p, j })) : (j += 1) {
        var b: mod.model.Chatterbox.TfmBlock = .{};
        try cbxLoadTfm(tl, std.fmt.bufPrint(&buf, "{s}.1.{d}", .{ p, j }) catch unreachable, &b);
        try s_.tfm.append(std.heap.c_allocator, b);
    }
    if (boundary) {
        s_.conv_w = try getCat(tl, p, ".2.weight", true);
        s_.conv_b = try getCat(tl, p, ".2.bias", true);
    }
}

/// The `load_hres` lambda of the CHATTERBOX arm (mtmd-model.cpp:2642
/// @b1b770d90).
fn cbxLoadHres(tl: *mod.clip.TensorLoader, p: []const u8, r: *mod.model.Chatterbox.HiftRes) mod.clip.Error!void {
    var buf: [256]u8 = undefined;
    var j: c_int = 0;
    while (cbxHas(tl, "{s}.convs1.{d}.weight", .{ p, j })) : (j += 1) {
        var u: mod.model.Chatterbox.HiftResUnit = .{};
        u.act1_alpha = try tl.getTensor((std.fmt.bufPrintSentinel(&buf, "{s}.activations1.{d}.alpha", .{ p, j }, 0) catch unreachable).ptr, true);
        u.act2_alpha = try tl.getTensor((std.fmt.bufPrintSentinel(&buf, "{s}.activations2.{d}.alpha", .{ p, j }, 0) catch unreachable).ptr, true);
        u.conv1_w = try tl.getTensor((std.fmt.bufPrintSentinel(&buf, "{s}.convs1.{d}.weight", .{ p, j }, 0) catch unreachable).ptr, true);
        u.conv1_b = try tl.getTensor((std.fmt.bufPrintSentinel(&buf, "{s}.convs1.{d}.bias", .{ p, j }, 0) catch unreachable).ptr, true);
        u.conv2_w = try tl.getTensor((std.fmt.bufPrintSentinel(&buf, "{s}.convs2.{d}.weight", .{ p, j }, 0) catch unreachable).ptr, true);
        u.conv2_b = try tl.getTensor((std.fmt.bufPrintSentinel(&buf, "{s}.convs2.{d}.bias", .{ p, j }, 0) catch unreachable).ptr, true);
        try r.units.append(std.heap.c_allocator, u);
    }
}

/// `get_tensor(string_format(fmt, args...))` with std::format-style
/// arguments, as the CHATTERBOX arm builds its names with `std::to_string`.
fn cbxGet(tl: *mod.clip.TensorLoader, comptime fmt: []const u8, args: anytype) mod.clip.Error!mod.graph.T {
    var buf: [256]u8 = undefined;
    const name = std.fmt.bufPrintSentinel(&buf, fmt, args, 0) catch return mod.clip.throwStr("chatterbox: tensor name too long");
    return tl.getTensor(name.ptr, true);
}

/// The `PROJECTOR_TYPE_CHATTERBOX` arm (mtmd-model.cpp:2574 @b1b770d90) of
/// load_tensors.
fn loadChatterbox(tl: *mod.clip.TensorLoader) mod.clip.Error!void {
    const func = "load_tensors";
    const model = tl.model;
    const cb = &model.cbx;
    var pbuf: [200]u8 = undefined;

    // flow encoder
    cb.input_embedding_w = try tl.getTensor("a.gen.flow.input_embedding.weight", true);
    cb.spk_affine_w = try tl.getTensor("a.gen.flow.spk_embed_affine_layer.weight", true);
    cb.spk_affine_b = try tl.getTensor("a.gen.flow.spk_embed_affine_layer.bias", true);
    cb.embed_linear_w = try tl.getTensor("a.gen.fenc.embed.out.0.weight", true);
    cb.embed_linear_b = try tl.getTensor("a.gen.fenc.embed.out.0.bias", true);
    cb.embed_norm_w = try tl.getTensor("a.gen.fenc.embed.out.1.weight", true);
    cb.embed_norm_b = try tl.getTensor("a.gen.fenc.embed.out.1.bias", true);
    cb.pre_conv1_w = try tl.getTensor("a.gen.fenc.pre_lookahead_layer.conv1.weight", true);
    cb.pre_conv1_b = try tl.getTensor("a.gen.fenc.pre_lookahead_layer.conv1.bias", true);
    cb.pre_conv2_w = try tl.getTensor("a.gen.fenc.pre_lookahead_layer.conv2.weight", true);
    cb.pre_conv2_b = try tl.getTensor("a.gen.fenc.pre_lookahead_layer.conv2.bias", true);
    var i: c_int = 0;
    while (cbxHas(tl, "a.gen.fenc.encoders.{d}.norm_mha.weight", .{i})) : (i += 1) {
        var l: mod.model.Chatterbox.EncLayer = .{};
        try cbxLoadEncLayer(tl, std.fmt.bufPrint(&pbuf, "a.gen.fenc.encoders.{d}", .{i}) catch unreachable, &l);
        try cb.enc.append(std.heap.c_allocator, l);
    }
    cb.up_conv_w = try tl.getTensor("a.gen.fenc.up_layer.conv.weight", true);
    cb.up_conv_b = try tl.getTensor("a.gen.fenc.up_layer.conv.bias", true);
    cb.up_embed_linear_w = try tl.getTensor("a.gen.fenc.up_embed.out.0.weight", true);
    cb.up_embed_linear_b = try tl.getTensor("a.gen.fenc.up_embed.out.0.bias", true);
    cb.up_embed_norm_w = try tl.getTensor("a.gen.fenc.up_embed.out.1.weight", true);
    cb.up_embed_norm_b = try tl.getTensor("a.gen.fenc.up_embed.out.1.bias", true);
    i = 0;
    while (cbxHas(tl, "a.gen.fenc.up_encoders.{d}.norm_mha.weight", .{i})) : (i += 1) {
        var l: mod.model.Chatterbox.EncLayer = .{};
        try cbxLoadEncLayer(tl, std.fmt.bufPrint(&pbuf, "a.gen.fenc.up_encoders.{d}", .{i}) catch unreachable, &l);
        try cb.up_enc.append(std.heap.c_allocator, l);
    }
    cb.after_norm_w = try tl.getTensor("a.gen.fenc.after_norm.weight", true);
    cb.after_norm_b = try tl.getTensor("a.gen.fenc.after_norm.bias", true);
    cb.encoder_proj_w = try tl.getTensor("a.gen.flow.encoder_proj.weight", true);
    cb.encoder_proj_b = try tl.getTensor("a.gen.flow.encoder_proj.bias", true);

    // cfm estimator
    cb.time_mlp_1_w = try tl.getTensor("a.gen.est.time_mlp.linear_1.weight", true);
    cb.time_mlp_1_b = try tl.getTensor("a.gen.est.time_mlp.linear_1.bias", true);
    cb.time_mlp_2_w = try tl.getTensor("a.gen.est.time_mlp.linear_2.weight", true);
    cb.time_mlp_2_b = try tl.getTensor("a.gen.est.time_mlp.linear_2.bias", true);
    cb.time_embed_mixer_w = try tl.getTensor("a.gen.est.time_embed_mixer.weight", false);
    try cbxLoadStage(tl, "a.gen.est.down_blocks.0", &cb.est_down, true);
    i = 0;
    while (cbxHas(tl, "a.gen.est.mid_blocks.{d}.0.block1.block.0.weight", .{i})) : (i += 1) {
        var st: mod.model.Chatterbox.EstStage = .{};
        try cbxLoadStage(tl, std.fmt.bufPrint(&pbuf, "a.gen.est.mid_blocks.{d}", .{i}) catch unreachable, &st, false);
        try cb.est_mid.append(std.heap.c_allocator, st);
    }
    try cbxLoadStage(tl, "a.gen.est.up_blocks.0", &cb.est_up, true);
    try cbxLoadCausal(tl, "a.gen.est.final_block", &cb.est_final_block);
    cb.est_final_proj_w = try tl.getTensor("a.gen.est.final_proj.weight", true);
    cb.est_final_proj_b = try tl.getTensor("a.gen.est.final_proj.bias", true);

    // hift vocoder
    cb.hift_pre_w = try tl.getTensor("a.gen.hift.conv_pre.weight", true);
    cb.hift_pre_b = try tl.getTensor("a.gen.hift.conv_pre.bias", true);
    cb.hift_post_w = try tl.getTensor("a.gen.hift.conv_post.weight", true);
    cb.hift_post_b = try tl.getTensor("a.gen.hift.conv_post.bias", true);
    i = 0;
    while (cbxHas(tl, "a.gen.hift.ups.{d}.weight", .{i})) : (i += 1) {
        var up: mod.model.Chatterbox.HiftUp = .{};
        up.up_w = try cbxGet(tl, "a.gen.hift.ups.{d}.weight", .{i});
        up.up_b = try cbxGet(tl, "a.gen.hift.ups.{d}.bias", .{i});
        up.source_down_w = try cbxGet(tl, "a.gen.hift.source_downs.{d}.weight", .{i});
        up.source_down_b = try cbxGet(tl, "a.gen.hift.source_downs.{d}.bias", .{i});
        try cbxLoadHres(tl, std.fmt.bufPrint(&pbuf, "a.gen.hift.source_resblocks.{d}", .{i}) catch unreachable, &up.source_res);
        try cbxLoadHres(tl, std.fmt.bufPrint(&pbuf, "a.gen.hift.resblocks.{d}", .{3 * i}) catch unreachable, &up.res_0);
        try cbxLoadHres(tl, std.fmt.bufPrint(&pbuf, "a.gen.hift.resblocks.{d}", .{3 * i + 1}) catch unreachable, &up.res_1);
        try cbxLoadHres(tl, std.fmt.bufPrint(&pbuf, "a.gen.hift.resblocks.{d}", .{3 * i + 2}) catch unreachable, &up.res_2);
        try cb.hift_ups.append(std.heap.c_allocator, up);
    }
    i = 0;
    while (cbxHas(tl, "a.gen.hift.f0_predictor.condnet.{d}.weight", .{i})) : (i += 2) {
        var fc: mod.model.Chatterbox.F0Conv = .{};
        fc.w = try cbxGet(tl, "a.gen.hift.f0_predictor.condnet.{d}.weight", .{i});
        fc.b = try cbxGet(tl, "a.gen.hift.f0_predictor.condnet.{d}.bias", .{i});
        try cb.f0_condnet.append(std.heap.c_allocator, fc);
    }
    cb.f0_classifier_w = try tl.getTensor("a.gen.hift.f0_predictor.classifier.weight", true);
    cb.f0_classifier_b = try tl.getTensor("a.gen.hift.f0_predictor.classifier.bias", true);

    // s3 tokenizer
    cb.s3tok_conv1_w = try tl.getTensor("a.s3tok.encoder.conv1.weight", true);
    cb.s3tok_conv1_b = try tl.getTensor("a.s3tok.encoder.conv1.bias", true);
    cb.s3tok_conv2_w = try tl.getTensor("a.s3tok.encoder.conv2.weight", true);
    cb.s3tok_conv2_b = try tl.getTensor("a.s3tok.encoder.conv2.bias", true);
    i = 0;
    while (cbxHas(tl, "a.s3tok.encoder.blocks.{d}.attn_ln.weight", .{i})) : (i += 1) {
        const p = std.fmt.bufPrint(&pbuf, "a.s3tok.encoder.blocks.{d}", .{i}) catch unreachable;
        var b: mod.model.Chatterbox.S3tokBlock = .{};
        b.attn_ln_w = try getCat(tl, p, ".attn_ln.weight", true);
        b.attn_ln_b = try getCat(tl, p, ".attn_ln.bias", true);
        b.attn_q_w = try getCat(tl, p, ".attn.query.weight", true);
        b.attn_q_b = try getCat(tl, p, ".attn.query.bias", true);
        b.attn_k_w = try getCat(tl, p, ".attn.key.weight", true);
        b.attn_v_w = try getCat(tl, p, ".attn.value.weight", true);
        b.attn_v_b = try getCat(tl, p, ".attn.value.bias", true);
        b.fsmn_w = try getCat(tl, p, ".attn.fsmn_block.weight", true);
        b.attn_out_w = try getCat(tl, p, ".attn.out.weight", true);
        b.attn_out_b = try getCat(tl, p, ".attn.out.bias", true);
        b.mlp_ln_w = try getCat(tl, p, ".mlp_ln.weight", true);
        b.mlp_ln_b = try getCat(tl, p, ".mlp_ln.bias", true);
        b.mlp_in_w = try getCat(tl, p, ".mlp.0.weight", true);
        b.mlp_in_b = try getCat(tl, p, ".mlp.0.bias", true);
        b.mlp_out_w = try getCat(tl, p, ".mlp.2.weight", true);
        b.mlp_out_b = try getCat(tl, p, ".mlp.2.bias", true);
        try cb.s3tok_blocks.append(std.heap.c_allocator, b);
    }
    cb.s3tok_down_w = try tl.getTensor("a.s3tok.quantizer._codebook.project_down.weight", true);
    cb.s3tok_down_b = try tl.getTensor("a.s3tok.quantizer._codebook.project_down.bias", true);

    // read on the host: embedding tables, source module
    var t: ?*llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_get_first_tensor(tl.loader.ctx_meta);
    while (t != null) : (t = llama.impl.llama_c.ggml_get_next_tensor(tl.loader.ctx_meta, t)) {
        const name = std.mem.sliceTo(&t.?.name, 0);
        if (std.mem.startsWith(u8, name, "a.gen.code.") or std.mem.startsWith(u8, name, "a.gen.t3.") or
            std.mem.startsWith(u8, name, "a.gen.hift.m_source."))
        {
            const loaded = try tl.getTensor(@ptrCast(&t.?.name), true);
            const gop = try model.cbx_tensors.getOrPut(std.heap.c_allocator, name);
            if (!gop.found_existing) gop.key_ptr.* = try std.heap.c_allocator.dupe(u8, name);
            gop.value_ptr.* = loaded;
        }
    }

    // voice.<name> holds the talker rows, voice.<name>.ref_* the flow decoder reference
    const out_embd = (model.cbx_tensors.get("a.gen.code.out_embd.weight") orelse return mod.clip.throwStr("map::at:  key not found")).?;
    const n_embd_talker = out_embd.ne[0];
    const ctx_gguf = tl.loader.ctx_gguf;
    var ids: std.ArrayList(f32) = .empty; // the converter stores 1D tensors as F32, ids are exact
    defer ids.deinit(std.heap.c_allocator);
    var it: i64 = 0;
    while (it < llama.impl.llama_c.gguf_get_n_tensors(ctx_gguf)) : (it += 1) {
        const name_z = llama.impl.llama_c.gguf_get_tensor_name(ctx_gguf, it);
        const name = std.mem.span(name_z);
        if (!std.mem.startsWith(u8, name, "voice.")) continue;
        const vt: *llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_get_tensor(tl.loader.ctx_meta, name_z);
        var voice: []const u8 = name[6..];
        ids.clearRetainingCapacity();
        var dst: []f32 = &.{};
        var ok = false;
        const n_el: usize = @intCast(llama.impl.llama_c.ggml_nelements(vt));
        if (cbxTake(&voice, ".ref_tokens")) {
            try ids.resize(std.heap.c_allocator, n_el);
            dst = ids.items;
            ok = vt.type == llama.impl.llama_c.GGML_TYPE_F32;
        } else if (cbxTake(&voice, ".ref_feat")) {
            const v = &(try model.gen_voices.getOrPut(voice)).ref.feat;
            try v.resize(std.heap.c_allocator, n_el);
            dst = v.items;
            ok = vt.type == llama.impl.llama_c.GGML_TYPE_F32 and vt.ne[0] == 80;
        } else if (cbxTake(&voice, ".ref_spk")) {
            const v = &(try model.gen_voices.getOrPut(voice)).ref.spk;
            try v.resize(std.heap.c_allocator, n_el);
            dst = v.items;
            ok = vt.type == llama.impl.llama_c.GGML_TYPE_F32 and vt.ne[0] == 192;
        } else {
            const v = &(try model.gen_voices.getOrPut(voice)).rows;
            try v.resize(std.heap.c_allocator, n_el);
            dst = v.items;
            ok = vt.type == llama.impl.llama_c.GGML_TYPE_F32 and llama.impl.llama_c.ggml_n_dims(vt) <= 2 and vt.ne[0] == n_embd_talker;
        }
        if (!ok) return mod.clip.throwErr("%s: invalid voice tensor %s\n", .{ func, name_z });
        if (!tl.readTensorBytes(name, std.mem.sliceAsBytes(dst)[0..llama.impl.llama_c.ggml_nbytes(vt)])) {
            return mod.clip.throwErr("%s: failed to read %s\n", .{ func, name_z });
        }
        if (ids.items.len != 0) {
            const tokens = &(try model.gen_voices.getOrPut(voice)).ref.tokens;
            try tokens.resize(std.heap.c_allocator, ids.items.len);
            for (tokens.items, ids.items) |*d, f| d.* = @intFromFloat(f);
        }
    }
    for (model.gen_voices.entries.items) |*e| {
        const v = &e.voice;
        if (v.rows.items.len == 0 or v.ref.tokens.items.len == 0 or v.ref.feat.items.len != 2 * 80 * v.ref.tokens.items.len or v.ref.spk.items.len == 0) {
            return mod.clip.throwErr("%s: incomplete voice %s\n", .{ func, e.name.ptr });
        }
    }
}

/// The `take` lambda of the CHATTERBOX arm's voice loop (mtmd-model.cpp:2784
/// @b1b770d90): strips `suffix` off `voice` if it ends with it.
fn cbxTake(voice: *[]const u8, suffix: []const u8) bool {
    const n = suffix.len;
    if (voice.len > n and std.mem.eql(u8, voice.*[voice.len - n ..], suffix)) {
        voice.* = voice.*[0 .. voice.len - n];
        return true;
    }
    return false;
}

/// Ports the `load_code2wav` lambda (mtmd-model.cpp:1580 @b1b770d90) of
/// `load_tensors`: code2wav, RVQ codes to raw PCM; drama3 has no quant,
/// pre_conv or tfm projections (`with_proj` false).
fn loadCode2wav(tl: *mod.clip.TensorLoader, with_proj: bool) mod.clip.Error!void {
    const model = tl.model;
    const hparams = &model.hparams;
    const c2w = &model.c2w;

    c2w.quant_first_in_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_FIRST_IN, .{W}, with_proj);
    c2w.quant_first_out_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_FIRST_OUT, .{W}, with_proj);
    c2w.quant_first_cb_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_FIRST_CB, .{W}, true);
    c2w.quant_rest_in_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_REST_IN, .{W}, with_proj);
    c2w.quant_rest_out_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_REST_OUT, .{W}, with_proj);
    c2w.quant_rest_cb_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_REST_CB, .{W}, true);

    c2w.pre_conv_w = try get(tl, mod.names.TN_A_GEN_WAV_PRE_CONV, .{W}, with_proj);
    c2w.pre_conv_b = try get(tl, mod.names.TN_A_GEN_WAV_PRE_CONV, .{B}, with_proj);

    c2w.tfm_in_proj_w = try get(tl, mod.names.TN_A_GEN_WAV_TFM_IN_PROJ, .{W}, with_proj);
    c2w.tfm_in_proj_b = try get(tl, mod.names.TN_A_GEN_WAV_TFM_IN_PROJ, .{B}, with_proj);
    c2w.tfm_out_proj_w = try get(tl, mod.names.TN_A_GEN_WAV_TFM_OUT_PROJ, .{W}, with_proj);
    c2w.tfm_out_proj_b = try get(tl, mod.names.TN_A_GEN_WAV_TFM_OUT_PROJ, .{B}, with_proj);
    c2w.tfm_output_norm_w = try get(tl, mod.names.TN_A_GEN_WAV_TFM_OUT_NORM, .{W}, true);

    // loaded manually, the generic model.layers loop is taken by code_predictor
    try resizeList(mod.model.Layer, &c2w.tfm_layers, @intCast(hparams.wav_tfm_n_layer));
    const p = s("a.gen.wav.tfm");
    for (c2w.tfm_layers.items, 0..) |*layer, ilu| {
        const l: c_int = @intCast(ilu);
        layer.q_w = try get(tl, mod.names.TN_ATTN_Q, .{ p, l, W }, true);
        layer.k_w = try get(tl, mod.names.TN_ATTN_K, .{ p, l, W }, true);
        layer.v_w = try get(tl, mod.names.TN_ATTN_V, .{ p, l, W }, true);
        layer.o_w = try get(tl, mod.names.TN_ATTN_OUTPUT, .{ p, l, W }, true);
        layer.ln_1_w = try get(tl, mod.names.TN_LN_1, .{ p, l, W }, true);
        layer.ln_2_w = try get(tl, mod.names.TN_LN_2, .{ p, l, W }, true);
        layer.ls_1_w = try get(tl, mod.names.TN_LS_1, .{ p, l, W }, true);
        layer.ls_2_w = try get(tl, mod.names.TN_LS_2, .{ p, l, W }, true);
        layer.ff_gate_w = try get(tl, mod.names.TN_FFN_GATE, .{ p, l, W }, true);
        layer.ff_up_w = try get(tl, mod.names.TN_FFN_UP, .{ p, l, W }, true);
        layer.ff_down_w = try get(tl, mod.names.TN_FFN_DOWN, .{ p, l, W }, true);
    }

    // upsample: 2x (causal ConvTranspose1d + ConvNeXt block)
    try resizeList(mod.model.Code2Wav.UpsampleBlock, &c2w.upsample, @intCast(hparams.wav_upsample_n_block));
    for (c2w.upsample.items, 0..) |*up, ilu| {
        const l: c_int = @intCast(ilu);
        up.conv_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_CONV, .{ l, W }, true);
        up.conv_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_CONV, .{ l, B }, true);
        up.dwconv_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_DWCONV, .{ l, W }, true);
        up.dwconv_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_DWCONV, .{ l, B }, true);
        up.norm_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_NORM, .{ l, W }, true);
        up.norm_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_NORM, .{ l, B }, true);
        up.pw1_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW1, .{ l, W }, true);
        up.pw1_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW1, .{ l, B }, true);
        up.pw2_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW2, .{ l, W }, true);
        up.pw2_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW2, .{ l, B }, true);
        up.gamma = try get(tl, mod.names.TN_A_GEN_WAV_UP_GAMMA, .{l}, true);
    }

    // DAC decoder: conv_pre + n upsample blocks (each with n_res residual units) + conv_post
    c2w.dac_entry_w = try get(tl, mod.names.TN_A_GEN_WAV_DAC_ENTRY, .{W}, true);
    c2w.dac_entry_b = try get(tl, mod.names.TN_A_GEN_WAV_DAC_ENTRY, .{B}, true);

    try resizeList(mod.model.Code2Wav.DacBlock, &c2w.dac, @intCast(hparams.wav_dac_n_block));
    for (c2w.dac.items, 0..) |*blk, ilu| {
        const l: c_int = @intCast(ilu);
        blk.snake_alpha = try get(tl, mod.names.TN_A_GEN_WAV_DAC_SNAKE, .{ l, s("alpha") }, true);
        blk.snake_beta = try get(tl, mod.names.TN_A_GEN_WAV_DAC_SNAKE, .{ l, s("beta") }, true);
        blk.conv_w = try get(tl, mod.names.TN_A_GEN_WAV_DAC_CONV, .{ l, W }, true);
        blk.conv_b = try get(tl, mod.names.TN_A_GEN_WAV_DAC_CONV, .{ l, B }, true);

        try resizeList(mod.model.Code2Wav.DacRes, &blk.res, @intCast(hparams.wav_dac_n_res));
        for (blk.res.items, 0..) |*res, iru| {
            const ir: c_int = @intCast(iru);
            res.act1_alpha = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_ACT1, .{ l, ir, s("alpha") }, true);
            res.act1_beta = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_ACT1, .{ l, ir, s("beta") }, true);
            res.conv1_w = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_CONV1, .{ l, ir, W }, true);
            res.conv1_b = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_CONV1, .{ l, ir, B }, true);
            res.act2_alpha = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_ACT2, .{ l, ir, s("alpha") }, true);
            res.act2_beta = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_ACT2, .{ l, ir, s("beta") }, true);
            res.conv2_w = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_CONV2, .{ l, ir, W }, true);
            res.conv2_b = try get(tl, mod.names.TN_A_GEN_WAV_DAC_RES_CONV2, .{ l, ir, B }, true);
        }
    }

    c2w.dac_post_snake_alpha = try get(tl, mod.names.TN_A_GEN_WAV_DAC_POST_SNAKE, .{s("alpha")}, true);
    c2w.dac_post_snake_beta = try get(tl, mod.names.TN_A_GEN_WAV_DAC_POST_SNAKE, .{s("beta")}, true);
    c2w.dac_post_conv_w = try get(tl, mod.names.TN_A_GEN_WAV_DAC_POST_CONV, .{W}, true);
    c2w.dac_post_conv_b = try get(tl, mod.names.TN_A_GEN_WAV_DAC_POST_CONV, .{B}, true);
}

/// Ports the model-specific switch of `load_tensors` (mtmd-model.cpp:1321
/// @b1b770d90), the `switch (model.proj_type)` after its generic layer loop.
pub fn load(tl: *mod.clip.TensorLoader) mod.clip.Error!void {
    const func = "load_tensors";
    const model = tl.model;
    const hparams = &model.hparams;
    const prefix = tl.prefix;

    switch (model.proj_type) {
        .mlp, .mlp_norm => {
            // LLaVA projection: mm.0 -> gelu -> mm.2
            // Yi-type llava (MLP_NORM): mm.0 -> norm mm.1 -> gelu -> mm.3 -> norm mm.4
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_3_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(3), W }, model.proj_type == .mlp_norm);
            if (model.mm_3_w != null) {
                // TODO: this is a hack to support Yi-type llava
                model.proj_type = .mlp_norm;
                model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
                model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
                model.mm_3_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(3), B }, true);
                model.mm_4_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(4), W }, true);
                model.mm_4_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(4), B }, true);
            } else {
                model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
                model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            }
            model.image_newline = try tl.getTensor(mod.names.TN_IMAGE_NEWLINE, false);
        },
        .ldp => {
            // MobileVLM projection
            model.mm_model_mlp_1_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_model_mlp_1_b = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(1), B }, true);
            model.mm_model_mlp_3_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
            model.mm_model_mlp_3_b = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(3), B }, true);
            model.mm_model_block_1_block_0_0_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(0), s("0.weight") }, true);
            model.mm_model_block_1_block_0_1_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(0), s("1.weight") }, true);
            model.mm_model_block_1_block_0_1_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(0), s("1.bias") }, true);
            model.mm_model_block_1_block_1_fc1_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc1.weight") }, true);
            model.mm_model_block_1_block_1_fc1_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc1.bias") }, true);
            model.mm_model_block_1_block_1_fc2_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc2.weight") }, true);
            model.mm_model_block_1_block_1_fc2_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(1), s("fc2.bias") }, true);
            model.mm_model_block_1_block_2_0_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(2), s("0.weight") }, true);
            model.mm_model_block_1_block_2_1_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(2), s("1.weight") }, true);
            model.mm_model_block_1_block_2_1_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(1), ci(2), s("1.bias") }, true);
            model.mm_model_block_2_block_0_0_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(0), s("0.weight") }, true);
            model.mm_model_block_2_block_0_1_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(0), s("1.weight") }, true);
            model.mm_model_block_2_block_0_1_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(0), s("1.bias") }, true);
            model.mm_model_block_2_block_1_fc1_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc1.weight") }, true);
            model.mm_model_block_2_block_1_fc1_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc1.bias") }, true);
            model.mm_model_block_2_block_1_fc2_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc2.weight") }, true);
            model.mm_model_block_2_block_1_fc2_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(1), s("fc2.bias") }, true);
            model.mm_model_block_2_block_2_0_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(2), s("0.weight") }, true);
            model.mm_model_block_2_block_2_1_w = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(2), s("1.weight") }, true);
            model.mm_model_block_2_block_2_1_b = try get(tl, mod.names.TN_MVLM_PROJ_BLOCK, .{ ci(2), ci(2), s("1.bias") }, true);
        },
        .ldpv2 => {
            // MobilVLM_V2 projection
            model.mm_model_mlp_0_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_model_mlp_0_b = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(0), B }, true);
            model.mm_model_mlp_2_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(2), W }, true);
            model.mm_model_mlp_2_b = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(2), B }, true);
            model.mm_model_peg_0_w = try get(tl, mod.names.TN_MVLM_PROJ_PEG, .{ ci(0), W }, true);
            model.mm_model_peg_0_b = try get(tl, mod.names.TN_MVLM_PROJ_PEG, .{ ci(0), B }, true);
        },
        .minicpmv => {
            model.mm_model_pos_embed_k = try tl.getTensor(mod.names.TN_MINICPMV_POS_EMBD_K, true);
            model.mm_model_query = try tl.getTensor(mod.names.TN_MINICPMV_QUERY, true);
            model.mm_model_proj = try tl.getTensor(mod.names.TN_MINICPMV_PROJ, true);
            model.mm_model_kv_proj = try tl.getTensor(mod.names.TN_MINICPMV_KV_PROJ, true);
            model.mm_model_attn_q_w = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("q"), W }, true);
            model.mm_model_attn_k_w = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("k"), W }, true);
            model.mm_model_attn_v_w = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("v"), W }, true);
            model.mm_model_attn_q_b = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("q"), B }, true);
            model.mm_model_attn_k_b = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("k"), B }, true);
            model.mm_model_attn_v_b = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("v"), B }, true);
            model.mm_model_attn_o_w = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("out"), W }, true);
            model.mm_model_attn_o_b = try get(tl, mod.names.TN_MINICPMV_ATTN, .{ s("out"), B }, true);
            model.mm_model_ln_q_w = try get(tl, mod.names.TN_MINICPMV_LN, .{ s("q"), W }, true);
            model.mm_model_ln_q_b = try get(tl, mod.names.TN_MINICPMV_LN, .{ s("q"), B }, true);
            model.mm_model_ln_kv_w = try get(tl, mod.names.TN_MINICPMV_LN, .{ s("kv"), W }, true);
            model.mm_model_ln_kv_b = try get(tl, mod.names.TN_MINICPMV_LN, .{ s("kv"), B }, true);
            model.mm_model_ln_post_w = try get(tl, mod.names.TN_MINICPMV_LN, .{ s("post"), W }, true);
            model.mm_model_ln_post_b = try get(tl, mod.names.TN_MINICPMV_LN, .{ s("post"), B }, true);
        },
        .minicpmv4_6, .minicpmv4_7, .minicpm_ocr_1_0 => {
            // the `get_merger_tensor` lambda: required only with a 4x merge
            const mr = hparams.n_merge == 4;

            // ViT merger: window self-attention
            model.vit_merger_ln1_w = try get(tl, mod.names.TN_VIT_MERGER_LN1, .{W}, mr);
            model.vit_merger_ln1_b = try get(tl, mod.names.TN_VIT_MERGER_LN1, .{B}, mr);
            model.vit_merger_attn_q_w = try get(tl, mod.names.TN_VIT_MERGER_ATTN_Q, .{W}, mr);
            model.vit_merger_attn_q_b = try get(tl, mod.names.TN_VIT_MERGER_ATTN_Q, .{B}, false);
            model.vit_merger_attn_k_w = try get(tl, mod.names.TN_VIT_MERGER_ATTN_K, .{W}, mr);
            model.vit_merger_attn_k_b = try get(tl, mod.names.TN_VIT_MERGER_ATTN_K, .{B}, false);
            model.vit_merger_attn_v_w = try get(tl, mod.names.TN_VIT_MERGER_ATTN_V, .{W}, mr);
            model.vit_merger_attn_v_b = try get(tl, mod.names.TN_VIT_MERGER_ATTN_V, .{B}, false);
            model.vit_merger_attn_o_w = try get(tl, mod.names.TN_VIT_MERGER_ATTN_O, .{W}, mr);
            model.vit_merger_attn_o_b = try get(tl, mod.names.TN_VIT_MERGER_ATTN_O, .{B}, false);
            // ViT merger: MLP downsample
            model.vit_merger_ds_ln_w = try get(tl, mod.names.TN_VIT_MERGER_DS_LN, .{W}, mr);
            model.vit_merger_ds_ln_b = try get(tl, mod.names.TN_VIT_MERGER_DS_LN, .{B}, mr);
            model.vit_merger_ds_up_w = try get(tl, mod.names.TN_VIT_MERGER_DS_UP, .{W}, mr);
            model.vit_merger_ds_up_b = try get(tl, mod.names.TN_VIT_MERGER_DS_UP, .{B}, false);
            model.vit_merger_ds_down_w = try get(tl, mod.names.TN_VIT_MERGER_DS_DOWN, .{W}, mr);
            model.vit_merger_ds_down_b = try get(tl, mod.names.TN_VIT_MERGER_DS_DOWN, .{B}, false);
            // Final Merger (DownsampleMLP)
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, true);
            model.mm_input_norm_b = try tl.getTensor(mod.names.TN_MM_INP_NORM_B, false);
            model.mm_ffn_up_w = try get(tl, mod.names.TN_MM_UP, .{W}, true);
            model.mm_ffn_up_b = try get(tl, mod.names.TN_MM_UP, .{B}, false);
            model.mm_ffn_down_w = try get(tl, mod.names.TN_MM_DOWN, .{W}, true);
            model.mm_ffn_down_b = try get(tl, mod.names.TN_MM_DOWN, .{B}, false);
            model.tok_embd_sep = try tl.getTensor(mod.names.TN_TOK_EMBD_SEP, model.proj_type == .minicpmv4_7);
        },
        .glm_edge => {
            model.mm_model_adapter_conv_w = try get(tl, mod.names.TN_GLM_ADAPER_CONV, .{W}, true);
            model.mm_model_adapter_conv_b = try get(tl, mod.names.TN_GLM_ADAPER_CONV, .{B}, true);
            model.mm_model_mlp_0_w = try get(tl, mod.names.TN_GLM_ADAPTER_LINEAR, .{W}, true);
            model.mm_model_ln_q_w = try get(tl, mod.names.TN_GLM_ADAPTER_NORM_1, .{W}, true);
            model.mm_model_ln_q_b = try get(tl, mod.names.TN_GLM_ADAPTER_NORM_1, .{B}, true);
            model.mm_model_mlp_1_w = try get(tl, mod.names.TN_GLM_ADAPTER_D_H_2_4H, .{W}, true);
            model.mm_model_mlp_2_w = try get(tl, mod.names.TN_GLM_ADAPTER_GATE, .{W}, true);
            model.mm_model_mlp_3_w = try get(tl, mod.names.TN_GLM_ADAPTER_D_4H_2_H, .{W}, true);
            model.mm_boi = try get(tl, mod.names.TN_TOK_GLM_BOI, .{}, true);
            model.mm_eoi = try get(tl, mod.names.TN_TOK_GLM_EOI, .{}, true);
        },
        .qwen2vl, .qwen25vl, .exaone4_5 => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .magevl => {
            // d_head rope factors: the first half drives the even outputs, the second half the odd ones
            model.rope_freqs = try tl.getTensor(mod.names.TN_ROPE_FREQS, true);
            const d_head: c_int = if (hparams.n_head > 0) @divTrunc(hparams.n_embd, hparams.n_head) else 0;
            if (d_head <= 0 or d_head * hparams.n_head != hparams.n_embd or @rem(d_head, 32) != 0 or
                model.rope_freqs.?.type != llama.impl.llama_c.GGML_TYPE_F32 or llama.impl.llama_c.ggml_nelements(model.rope_freqs) != d_head)
            {
                return mod.clip.throwErr("%s: hparams do not match the tensor shapes\n", .{func});
            }
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .qwen3vl => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .lfm3vl => {
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, true);
            model.mm_input_norm_b = try tl.getTensor(mod.names.TN_MM_INP_NORM_B, true);
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            // each layer must be either a conv block or a complete attention block
            for (model.layers.items, 0..) |*layer, il| {
                const is_conv = layer.conv2d_in_proj_w != null;
                const is_attn = layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null;
                if (is_conv == is_attn) {
                    return mod.clip.throwErr("%s: lfm3vl layer %zu is neither a complete conv nor a complete attention block\n", .{ func, il });
                }
            }
        },
        .ling3vl => {
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, true); // merger.norm
            model.mm_input_norm_b = try tl.getTensor(mod.names.TN_MM_INP_NORM_B, true); // merger.norm
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true); // linear_proj.0
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true); // linear_proj.2
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .mimovl => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, false);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, false);
        },
        .minimax_m3 => {
            // per-patch MLP: mm.1 -> gelu -> mm.2
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            // 2x2 merge MLP: mm.merge.fc1 -> gelu -> mm.merge.fc2
            model.mm_merger_fc1_w = try get(tl, mod.names.TN_MM_MERGER_FC1, .{W}, true);
            model.mm_merger_fc1_b = try get(tl, mod.names.TN_MM_MERGER_FC1, .{B}, true);
            model.mm_merger_fc2_w = try get(tl, mod.names.TN_MM_MERGER_FC2, .{W}, true);
            model.mm_merger_fc2_b = try get(tl, mod.names.TN_MM_MERGER_FC2, .{B}, true);
        },
        .muse_spark_audio => {
            try resizeList(mod.model.Layer, &model.layers, @intCast(hparams.n_layer));
            try resizeList(mod.model.AudioPatch, &model.a_patches, hparams.audio_patch_sizes.items.len);
            for (model.a_patches.items, 0..) |*pe, i| {
                pe.proj_w = try get(tl, mod.names.TN_A_PATCH, .{ ci(i), W }, true);
                pe.proj_b = try get(tl, mod.names.TN_A_PATCH, .{ ci(i), B }, true);
                pe.norm_w = try get(tl, mod.names.TN_A_PATCH_NORM, .{ ci(i), W }, true);
                if (mod.graph.ne(pe.proj_w, 1) != hparams.audio_hidden_sizes.items[i]) {
                    return mod.clip.throwErr("%s: audio stage %zu projection has width %lld, expected %d\n", .{ func, i, @as(c_longlong, mod.graph.ne(pe.proj_w, 1)), hparams.audio_hidden_sizes.items[i] });
                }
            }

            const pa: [*:0]const u8 = "a";
            for (model.layers.items, 0..) |*layer, ilu| {
                const il = ci(ilu);
                layer.conv_w = try get(tl, mod.names.TN_CONV1D, .{ il, W }, true);
                layer.conv_b = try get(tl, mod.names.TN_CONV1D, .{ il, B }, true);
                layer.conv_norm_w = try get(tl, mod.names.TN_A_CONV1D_NORM, .{ il, W }, true);
                layer.ff_up_w = try get(tl, mod.names.TN_FFN_UP, .{ pa, il, W }, true);
                layer.ff_up_b = try get(tl, mod.names.TN_FFN_UP, .{ pa, il, B }, true);
                layer.ff_down_w = try get(tl, mod.names.TN_FFN_DOWN, .{ pa, il, W }, true);
                layer.ff_down_b = try get(tl, mod.names.TN_FFN_DOWN, .{ pa, il, B }, true);
            }

            // pre_ln and post_ln come from the generic path above, where they are optional
            if (model.pre_ln_w == null or model.post_ln_w == null) {
                return mod.clip.throwErr("%s: missing pre_ln or post_ln weight\n", .{func});
            }
            model.mm_fc_w = try get(tl, mod.names.TN_MM_AUDIO_FC, .{W}, true);
            model.mm_fc_b = try get(tl, mod.names.TN_MM_AUDIO_FC, .{B}, true);
        },
        .muse_glimmer => {
            // 3-linear MLP: fc -> erf-GELU -> proj -> erf-GELU -> vision_proj (into LLM residual dim)
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
        },
        .step3vl => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, false);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, false);
            model.mm_model_proj = try get(tl, mod.names.TN_MM_PROJECTOR, .{W}, true);
        },
        .youtuvl => {
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, true); // merger.ln_q (RMS norm)
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true); // merger.mlp.0
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true); // merger.mlp.2
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .yasa2 => {
            // reuse tensors already loaded by the common section
            // (TN_PATCH_EMBD and TN_PATCH_BIAS have the same tensor names)
            model.yasa_patch_w = model.patch_embeddings_0;
            model.yasa_patch_b = model.patch_bias;
            model.yasa_patch_ln_w = try tl.getTensor(mod.names.TN_YASA_PATCH_LN_W, false);
            model.yasa_patch_ln_b = try tl.getTensor(mod.names.TN_YASA_PATCH_LN_B, false);
            model.yasa_backbone_ln_w = try tl.getTensor(mod.names.TN_YASA_BACKBONE_LN_W, false);
            model.yasa_backbone_ln_b = try tl.getTensor(mod.names.TN_YASA_BACKBONE_LN_B, false);
            model.yasa_vision_pos_embed = try tl.getTensor(mod.names.TN_YASA_POS_EMBD, false);
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, false);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, false);

            for (model.yasa_stages.items) |*st| st.blocks.deinit(std.heap.c_allocator);
            model.yasa_stages.clearRetainingCapacity();
            var si: c_int = 0;
            while (true) : (si += 1) {
                var stage: mod.model.Yasa2Stage = .{};
                errdefer stage.blocks.deinit(std.heap.c_allocator);
                stage.down_ln_w = try get(tl, mod.names.TN_YASA_STAGE_DOWN_LN, .{ si, W }, false);
                stage.down_ln_b = try get(tl, mod.names.TN_YASA_STAGE_DOWN_LN, .{ si, B }, false);
                stage.down_conv_w = try get(tl, mod.names.TN_YASA_STAGE_DOWN_CONV, .{ si, W }, false);
                stage.down_conv_b = try get(tl, mod.names.TN_YASA_STAGE_DOWN_CONV, .{ si, B }, false);

                var bi: c_int = 0;
                while (true) : (bi += 1) {
                    var blk: mod.model.Yasa2Block = .{};
                    blk.dw_w = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("dw"), W }, false);
                    if (blk.dw_w == null) break;
                    blk.dw_b = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("dw"), B }, false);
                    blk.ln_w = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("ln"), W }, false);
                    blk.ln_b = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("ln"), B }, false);
                    blk.pw1_w = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw1"), W }, false);
                    blk.pw1_b = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw1"), B }, false);
                    blk.grn_w = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("grn"), W }, false);
                    blk.grn_b = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("grn"), B }, false);
                    blk.pw2_w = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw2"), W }, false);
                    blk.pw2_b = try get(tl, mod.names.TN_YASA_STAGE_BLK, .{ si, bi, s("pw2"), B }, false);
                    try stage.blocks.append(std.heap.c_allocator, blk);
                }

                if (stage.down_conv_w == null and stage.blocks.items.len == 0) {
                    stage.blocks.deinit(std.heap.c_allocator);
                    break;
                }
                try model.yasa_stages.append(std.heap.c_allocator, stage);
            }
        },
        .glm4v, .glm5v => {
            model.mm_fc_w = try get(tl, mod.names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_ffn_up_w = try get(tl, mod.names.TN_MM_UP, .{W}, true);
            model.mm_ffn_up_b = try get(tl, mod.names.TN_MM_UP, .{B}, false);
            model.mm_ffn_gate_w = try get(tl, mod.names.TN_MM_GATE, .{W}, true);
            model.mm_ffn_gate_b = try get(tl, mod.names.TN_MM_GATE, .{B}, false);
            model.mm_ffn_down_w = try get(tl, mod.names.TN_MM_DOWN, .{W}, true);
            model.mm_ffn_down_b = try get(tl, mod.names.TN_MM_DOWN, .{B}, false);
            model.mm_post_norm_w = try get(tl, mod.names.TN_MM_POST_NORM, .{W}, true);
            model.mm_post_norm_b = try get(tl, mod.names.TN_MM_POST_NORM, .{B}, false);
            model.mm_patch_merger_w = try get(tl, mod.names.TN_MM_PATCH_MERGER, .{W}, true);
            model.mm_patch_merger_b = try get(tl, mod.names.TN_MM_PATCH_MERGER, .{B}, true);
        },
        .gemma3 => {
            model.mm_input_proj_w = try tl.getTensor(mod.names.TN_MM_INP_PROJ, true);
            model.mm_soft_emb_norm_w = try tl.getTensor(mod.names.TN_MM_SOFT_EMB_N, true);
        },
        .gemma4v => {
            model.mm_input_proj_w = try tl.getTensor(mod.names.TN_MM_INP_PROJ, true);
            model.std_bias = try tl.getTensor(mod.names.TN_STD_BIAS, false);
            model.std_scale = try tl.getTensor(mod.names.TN_STD_SCALE, false);
            // load scalar for Gemma4ClippableLinear
            try loadClampInfo(tl);
        },
        .gemma4uv => {
            model.mm_input_proj_w = try tl.getTensor(mod.names.TN_MM_INP_PROJ, true);
            model.patch_norm_1_w = try get(tl, mod.names.TN_PATCH_NORM, .{ ci(1), W }, true);
            model.patch_norm_1_b = try get(tl, mod.names.TN_PATCH_NORM, .{ ci(1), B }, true);
            model.patch_norm_2_w = try get(tl, mod.names.TN_PATCH_NORM, .{ ci(2), W }, true);
            model.patch_norm_2_b = try get(tl, mod.names.TN_PATCH_NORM, .{ ci(2), B }, true);
            model.patch_norm_3_w = try get(tl, mod.names.TN_PATCH_NORM, .{ ci(3), W }, true); // pos_norm
            model.patch_norm_3_b = try get(tl, mod.names.TN_PATCH_NORM, .{ ci(3), B }, true); // pos_norm
        },
        .gemma3nv => {
            model.mobilenet_stem_conv_w = try tl.getTensor(mod.names.TN_MNV5_STEM_CONV, false);
            model.mobilenet_stem_conv_b = try tl.getTensor(mod.names.TN_MNV5_STEM_BIAS, false);
            model.mobilenet_stem_norm_w = try tl.getTensor(mod.names.TN_MNV5_STEM_BN, false);

            model.msfa_ffn_expand_w = try tl.getTensor(mod.names.TN_MNV5_MSFA_FFN_EXP_W, false);
            model.msfa_ffn_expand_bn = try tl.getTensor(mod.names.TN_MNV5_MSFA_FFN_EXP_BN, false); // Consume BN if present but likely folded
            model.msfa_ffn_project_w = try tl.getTensor(mod.names.TN_MNV5_MSFA_FFN_PROJ_W, false);
            model.msfa_ffn_project_bn = try tl.getTensor(mod.names.TN_MNV5_MSFA_FFN_PROJ_BN, false);

            model.msfa_concat_norm_w = try tl.getTensor(mod.names.TN_MNV5_MSFA_NORM, false);

            // Dynamically load blocks stage by stage
            var stage: c_int = 0;
            while (stage < 4) : (stage += 1) {
                var blocks_found_in_stage: c_int = 0;

                var blk_idx: c_int = 0;
                while (true) : (blk_idx += 1) {
                    var found_block = false;
                    var block: mod.model.MobilenetV5Block = .{};

                    // 1. Check for Edge Residual (S0)
                    block.s0_conv_exp_w = try get(tl, mod.names.TN_MNV5_BLK_S0_EXP_W, .{ stage, blk_idx }, false);
                    if (block.s0_conv_exp_w != null) {
                        found_block = true;
                        block.s0_bn1_w = try get(tl, mod.names.TN_MNV5_BLK_S0_BN1_W, .{ stage, blk_idx }, false);
                        block.s0_conv_pwl_w = try get(tl, mod.names.TN_MNV5_BLK_S0_PWL_W, .{ stage, blk_idx }, false);
                        block.s0_bn2_w = try get(tl, mod.names.TN_MNV5_BLK_S0_BN2_W, .{ stage, blk_idx }, false);
                    }
                    // 2. Check for UIR (Universal Inverted Residual)
                    else {
                        // Check for dw_start OR pw_exp (some UIR blocks skip dw_start)
                        block.dw_start_w = try get(tl, mod.names.TN_MNV5_BLK_DW_START_W, .{ stage, blk_idx }, false);
                        block.pw_exp_w = try get(tl, mod.names.TN_MNV5_BLK_PW_EXP_W, .{ stage, blk_idx }, false);

                        if (block.dw_start_w != null or block.pw_exp_w != null) {
                            found_block = true;
                            if (block.dw_start_w != null) {
                                block.dw_start_bn_w = try get(tl, mod.names.TN_MNV5_BLK_DW_START_BN, .{ stage, blk_idx }, false);
                            }
                            if (block.pw_exp_w != null) {
                                block.pw_exp_bn_w = try get(tl, mod.names.TN_MNV5_BLK_PW_EXP_BN, .{ stage, blk_idx }, false);
                            }
                            block.dw_mid_w = try get(tl, mod.names.TN_MNV5_BLK_DW_MID_W, .{ stage, blk_idx }, false);
                            if (block.dw_mid_w != null) {
                                block.dw_mid_bn_w = try get(tl, mod.names.TN_MNV5_BLK_DW_MID_BN, .{ stage, blk_idx }, false);
                            }
                            block.pw_proj_w = try get(tl, mod.names.TN_MNV5_BLK_PW_PROJ_W, .{ stage, blk_idx }, false);
                            if (block.pw_proj_w != null) {
                                block.pw_proj_bn_w = try get(tl, mod.names.TN_MNV5_BLK_PW_PROJ_BN, .{ stage, blk_idx }, false);
                            }
                            block.layer_scale_w = try get(tl, mod.names.TN_MNV5_BLK_LAYER_SCALE, .{ stage, blk_idx }, false);
                        }
                    }

                    // 3. Check for Attention (MQA)
                    // Even if UIR/Edge check failed, this might be a pure attention block
                    const attn_q_check = try get(tl, mod.names.TN_MNV5_ATTN_Q_W, .{ stage, blk_idx }, false);
                    if (attn_q_check != null) {
                        found_block = true;
                        block.attn_q_w = attn_q_check;
                        block.attn_k_w = try get(tl, mod.names.TN_MNV5_ATTN_K_W, .{ stage, blk_idx }, false);
                        block.attn_v_w = try get(tl, mod.names.TN_MNV5_ATTN_V_W, .{ stage, blk_idx }, false);
                        block.attn_o_w = try get(tl, mod.names.TN_MNV5_ATTN_O_W, .{ stage, blk_idx }, false);
                        block.attn_k_dw_w = try get(tl, mod.names.TN_MNV5_ATTN_K_DW, .{ stage, blk_idx }, false);
                        block.attn_k_norm_w = try get(tl, mod.names.TN_MNV5_ATTN_K_NORM, .{ stage, blk_idx }, false);
                        block.attn_v_dw_w = try get(tl, mod.names.TN_MNV5_ATTN_V_DW, .{ stage, blk_idx }, false);
                        block.attn_v_norm_w = try get(tl, mod.names.TN_MNV5_ATTN_V_NORM, .{ stage, blk_idx }, false);
                        block.attn_norm_w = try get(tl, mod.names.TN_MNV5_ATTN_NORM, .{ stage, blk_idx }, false);
                        // Note: Attention blocks also have layer_scale, load it if not already loaded by UIR check
                        if (block.layer_scale_w == null) {
                            block.layer_scale_w = try get(tl, mod.names.TN_MNV5_BLK_LAYER_SCALE, .{ stage, blk_idx }, false);
                        }
                    }

                    if (found_block) {
                        try model.mobilenet_blocks.append(std.heap.c_allocator, block);
                        blocks_found_in_stage += 1;
                    } else {
                        // End of blocks for this stage
                        break;
                    }
                }

                // Track where this stage ends in the flat vector
                if (blocks_found_in_stage > 0) {
                    try model.mobilenet_stage_ends.append(std.heap.c_allocator, @intCast(model.mobilenet_blocks.items.len - 1));
                    mod.impl.logInf("%s: Stage %d ended at global block index %zu\n", .{ func, stage, model.mobilenet_blocks.items.len - 1 });
                }
            }
            model.mm_input_proj_w = try tl.getTensor(mod.names.TN_MM_INP_PROJ, true);
            model.mm_soft_emb_norm_w = try tl.getTensor(mod.names.TN_MM_SOFT_EMB_N, true);
        },
        .idefics3 => {
            model.mm_fc_w = try get(tl, mod.names.TN_MM_PROJECTOR, .{W}, true);
        },
        .cohere2v => {
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .lfm2, .d1omni_v => {
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, false);
            model.mm_input_norm_b = try tl.getTensor(mod.names.TN_MM_INP_NORM_B, false);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .kimivl, .paddleocr, .kimik25 => {
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, true);
            model.mm_input_norm_b = try tl.getTensor(mod.names.TN_MM_INP_NORM_B, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .kimik31 => {
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_post_norm_w = try get(tl, mod.names.TN_MM_POST_NORM, .{W}, true);
        },
        .deepseek4v => {
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            // sentinel token embeddings written into the output block
            model.image_newline = try tl.getTensor(mod.names.TN_IMAGE_NEWLINE, true);
            model.token_embd_img_start = try tl.getTensor(mod.names.TN_TOK_IMG_START, true);
            model.token_embd_img_end = try tl.getTensor(mod.names.TN_TOK_IMG_END, true);
            model.token_embd_img_pad = try tl.getTensor(mod.names.TN_TOK_IMG_PAD, true);
        },
        .deepseek41v => {
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            // image_start / image_end / image_newline, written into the output block
            model.image_newline = try tl.getTensor(mod.names.TN_IMAGE_NEWLINE, true);
            model.token_embd_img_start = try tl.getTensor(mod.names.TN_TOK_IMG_START, true);
            model.token_embd_img_end = try tl.getTensor(mod.names.TN_TOK_IMG_END, true);
        },
        .pixtral => {
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, false);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, false);
            // [IMG_BREAK] token embedding
            model.token_embd_img_break = try tl.getTensor(mod.names.TN_TOK_IMG_BREAK, true);
            // for mistral small 3.1
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, false);
            model.mm_patch_merger_w = try get(tl, mod.names.TN_MM_PATCH_MERGER, .{W}, false);
        },
        .lightonocr => {
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, false);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, false);
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, false);
            model.mm_patch_merger_w = try get(tl, mod.names.TN_MM_PATCH_MERGER, .{W}, false);
        },
        .dots_ocr, .dots3note_v => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            model.mm_input_norm_w = try tl.getTensor(mod.names.TN_MM_INP_NORM, true);
            model.mm_input_norm_b = try tl.getTensor(mod.names.TN_MM_INP_NORM_B, true);
            // post_trunk_norm: applied after all ViT blocks, before the merger
            model.post_ln_w = try get(tl, mod.names.TN_MM_POST_NORM, .{W}, true);
        },
        .dots3note_a => {
            model.conv2d_1_w = try get(tl, mod.names.TN_CONV2D, .{ ci(1), W }, true);
            model.conv2d_1_b = try get(tl, mod.names.TN_CONV2D, .{ ci(1), B }, true);
            model.conv2d_2_w = try get(tl, mod.names.TN_CONV2D, .{ ci(2), W }, true);
            model.conv2d_2_b = try get(tl, mod.names.TN_CONV2D, .{ ci(2), B }, true);
            model.conv2d_3_w = try get(tl, mod.names.TN_CONV2D, .{ ci(3), W }, true);
            model.conv2d_3_b = try get(tl, mod.names.TN_CONV2D, .{ ci(3), B }, true);
            model.conv_out_w = try get(tl, mod.names.TN_CONV_OUT, .{W}, true); // no bias
            // adapter: LayerNorm -> Linear -> GELU -> Linear
            model.mm_norm_pre_w = try get(tl, mod.names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_pre_b = try get(tl, mod.names.TN_MM_NORM_PRE, .{B}, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), B }, true);
        },
        .ultravox => {
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, mod.names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, mod.names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_norm_pre_w = try get(tl, mod.names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_mid_w = try get(tl, mod.names.TN_MM_NORM_MID, .{W}, true);
        },
        .meralion => {
            // Whisper encoder conv layers
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, mod.names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, mod.names.TN_CONV1D, .{ ci(2), B }, true);
            // MERaLiON adaptor: 4 linear layers + ln_pre
            model.mm_0_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
            model.mm_3_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), W }, true);
            model.mm_3_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), B }, true);
            // ln_speech (LayerNorm before adaptor)
            model.mm_norm_pre_w = try get(tl, mod.names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_pre_b = try get(tl, mod.names.TN_MM_NORM_PRE, .{B}, true);
        },
        .qwen2a => {
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, mod.names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, mod.names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_fc_w = try get(tl, mod.names.TN_MM_AUDIO_FC, .{W}, true);
            model.mm_fc_b = try get(tl, mod.names.TN_MM_AUDIO_FC, .{B}, true);
        },
        .qwen3a => {
            model.conv2d_1_w = try get(tl, mod.names.TN_CONV2D, .{ ci(1), W }, true);
            model.conv2d_1_b = try get(tl, mod.names.TN_CONV2D, .{ ci(1), B }, true);
            model.conv2d_2_w = try get(tl, mod.names.TN_CONV2D, .{ ci(2), W }, true);
            model.conv2d_2_b = try get(tl, mod.names.TN_CONV2D, .{ ci(2), B }, true);
            model.conv2d_3_w = try get(tl, mod.names.TN_CONV2D, .{ ci(3), W }, true);
            model.conv2d_3_b = try get(tl, mod.names.TN_CONV2D, .{ ci(3), B }, true);
            model.conv_out_w = try get(tl, mod.names.TN_CONV_OUT, .{W}, true); // no bias
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
        },
        .mimo_audio => {
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, mod.names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, mod.names.TN_CONV1D, .{ ci(2), B }, true);
            model.downsample_conv_w = try get(tl, mod.names.TN_A_DOWNSAMPLE_CONV, .{W}, true);
            model.downsample_norm_w = try get(tl, mod.names.TN_A_DOWNSAMPLE_NORM, .{W}, true);
            model.downsample_norm_b = try get(tl, mod.names.TN_A_DOWNSAMPLE_NORM, .{B}, true);
            model.rvq_codebook = try get(tl, mod.names.TN_A_RVQ_CODEBOOK, .{W}, false);
            model.mm_a_code_embd = try get(tl, mod.names.TN_MM_A_CODE_EMBD, .{W}, false);
            if (model.rvq_codebook == null or model.mm_a_code_embd == null) {
                return mod.clip.throwErr("%s: mimo_audio: missing %s or %s\n", .{ func, s(mod.names.TN_A_RVQ_CODEBOOK), s(mod.names.TN_MM_A_CODE_EMBD) });
            }
            // hparams.rvq_codebook_size comes from GGUF metadata and is independent of the
            // tensors' actual shapes - bound it so codebook/code_embd views built from it
            // (mimo-audio.cpp) can never read past either tensor's allocated bins/vocab.
            for (hparams.rvq_codebook_size.items) |bins| {
                if (bins <= 0 or bins > mod.graph.ne(model.rvq_codebook, 1) or bins > mod.graph.ne(model.mm_a_code_embd, 1)) {
                    return mod.clip.throwErr("%s: mimo_audio: %s entry (%d) out of range for codebook/code_embd tensors\n", .{ func, s(mod.names.KEY_A_RVQ_CODEBOOK_SIZE), bins });
                }
            }

            // LLM-side connector: input_local_transformer + projection
            try resizeList(mod.model.Layer, &model.mm_a_local_layers, @intCast(hparams.audio_local_n_layer));
            for (model.mm_a_local_layers.items, 0..) |*layer, ilu| {
                const il: c_int = @intCast(ilu);
                layer.q_w = try get(tl, mod.names.TN_MM_A_LOCAL_ATTN_Q, .{ il, W }, true);
                layer.q_b = try get(tl, mod.names.TN_MM_A_LOCAL_ATTN_Q, .{ il, B }, true);
                layer.k_w = try get(tl, mod.names.TN_MM_A_LOCAL_ATTN_K, .{ il, W }, true);
                layer.k_b = try get(tl, mod.names.TN_MM_A_LOCAL_ATTN_K, .{ il, B }, true);
                layer.v_w = try get(tl, mod.names.TN_MM_A_LOCAL_ATTN_V, .{ il, W }, true);
                layer.v_b = try get(tl, mod.names.TN_MM_A_LOCAL_ATTN_V, .{ il, B }, true);
                layer.o_w = try get(tl, mod.names.TN_MM_A_LOCAL_ATTN_OUT, .{ il, W }, true);
                layer.ff_gate_w = try get(tl, mod.names.TN_MM_A_LOCAL_FFN_GATE, .{ il, W }, true);
                layer.ff_up_w = try get(tl, mod.names.TN_MM_A_LOCAL_FFN_UP, .{ il, W }, true);
                layer.ff_down_w = try get(tl, mod.names.TN_MM_A_LOCAL_FFN_DOWN, .{ il, W }, true);
                layer.ln_1_w = try get(tl, mod.names.TN_MM_A_LOCAL_LN1, .{ il, W }, true);
                layer.ln_2_w = try get(tl, mod.names.TN_MM_A_LOCAL_LN2, .{ il, W }, true);
            }
            model.mm_a_local_norm_w = try get(tl, mod.names.TN_MM_A_LOCAL_NORM, .{W}, true);

            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
        },
        .qwen3tts_spkenc => {
            // stem TDNN (block 0)
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(0), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(0), B }, true);

            // SE-Res2Net blocks (GGUF bid 1..3, one per hparams.n_layer)
            try resizeList(mod.model.Layer, &model.layers, @intCast(hparams.n_layer));
            for (model.layers.items, 0..) |*layer, ilu| {
                const bid: c_int = @intCast(ilu + 1);
                layer.conv_pw1_w = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, bid, W }, true);
                layer.conv_pw1_b = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, bid, B }, true);
                layer.conv_pw2_w = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, bid, W }, true);
                layer.conv_pw2_b = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, bid, B }, true);
                layer.se_conv1_w = try get(tl, mod.names.TN_A_SE_CONV1, .{ bid, W }, true);
                layer.se_conv1_b = try get(tl, mod.names.TN_A_SE_CONV1, .{ bid, B }, true);
                layer.se_conv2_w = try get(tl, mod.names.TN_A_SE_CONV2, .{ bid, W }, true);
                layer.se_conv2_b = try get(tl, mod.names.TN_A_SE_CONV2, .{ bid, B }, true);
                try layer.res2_conv_w.resize(std.heap.c_allocator, 7);
                try layer.res2_conv_b.resize(std.heap.c_allocator, 7);
                @memset(layer.res2_conv_w.items, null);
                @memset(layer.res2_conv_b.items, null);
                for (0..7) |xid| {
                    layer.res2_conv_w.items[xid] = try get(tl, mod.names.TN_A_CONV_RES2, .{ bid, ci(xid), W }, true);
                    layer.res2_conv_b.items[xid] = try get(tl, mod.names.TN_A_CONV_RES2, .{ bid, ci(xid), B }, true);
                }
            }

            // multi-layer feature aggregation
            model.conv_out_w = try get(tl, mod.names.TN_CONV_OUT, .{W}, true);
            model.conv_out_b = try get(tl, mod.names.TN_CONV_OUT, .{B}, true);

            // attentive statistics pooling
            model.spk_asp_attn_w = try get(tl, mod.names.TN_A_ASP_ATTN, .{W}, true);
            model.spk_asp_attn_b = try get(tl, mod.names.TN_A_ASP_ATTN, .{B}, true);
            model.spk_asp_tdnn_w = try get(tl, mod.names.TN_A_ASP_TDNN, .{W}, true);
            model.spk_asp_tdnn_b = try get(tl, mod.names.TN_A_ASP_TDNN, .{B}, true);

            // final speaker embedding projection
            model.mm_fc_w = try get(tl, mod.names.TN_MM_AUDIO_FC, .{W}, true);
            model.mm_fc_b = try get(tl, mod.names.TN_MM_AUDIO_FC, .{B}, true);
        },
        .pockettts_spkenc => {
            try tl.loadSeanet(&model.seanet, false);
            model.downsample_w = try get(tl, mod.names.TN_A_DOWNSAMPLE_CONV, .{W}, true);
            model.spk_proj_w = try get(tl, mod.names.TN_A_SPEAKER_PROJ, .{W}, true);
        },
        // `PROJECTOR_TYPE_CHATTERBOX_SPKENC` (mtmd-model.cpp:2829 @b1b770d90)
        .chatterbox_spkenc => {
            const cb = &model.cbx;
            cb.spk_conv1_w = try tl.getTensor("a.spk.head.conv1.weight", true);
            try cbxLoadBn(tl, "a.spk.head.bn1", &cb.spk_bn1);
            try cbxLoadRes2d(tl, "a.spk.head.layer1.0", &cb.spk_layer1_0);
            try cbxLoadRes2d(tl, "a.spk.head.layer1.1", &cb.spk_layer1_1);
            try cbxLoadRes2d(tl, "a.spk.head.layer2.0", &cb.spk_layer2_0);
            try cbxLoadRes2d(tl, "a.spk.head.layer2.1", &cb.spk_layer2_1);
            cb.spk_conv2_w = try tl.getTensor("a.spk.head.conv2.weight", true);
            try cbxLoadBn(tl, "a.spk.head.bn2", &cb.spk_bn2);
            cb.spk_tdnn_w = try tl.getTensor("a.spk.xvector.tdnn.linear.weight", true);
            try cbxLoadBn(tl, "a.spk.xvector.tdnn.nonlinear.batchnorm", &cb.spk_tdnn_bn);
            try cbxLoadCamBlock(tl, "a.spk.xvector.block1", "a.spk.xvector.transit1", &cb.spk_block1);
            try cbxLoadCamBlock(tl, "a.spk.xvector.block2", "a.spk.xvector.transit2", &cb.spk_block2);
            try cbxLoadCamBlock(tl, "a.spk.xvector.block3", "a.spk.xvector.transit3", &cb.spk_block3);
            try cbxLoadBn(tl, "a.spk.xvector.out_nonlinear.batchnorm", &cb.spk_out_bn);
            cb.spk_dense_w = try tl.getTensor("a.spk.xvector.dense.linear.weight", true);
            try cbxLoadBn(tl, "a.spk.xvector.dense.nonlinear.batchnorm", &cb.spk_dense_bn);

            // read on the host: voice encoder lstm, conditioning encoder
            var t: ?*llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_get_first_tensor(tl.loader.ctx_meta);
            while (t != null) : (t = llama.impl.llama_c.ggml_get_next_tensor(tl.loader.ctx_meta, t)) {
                const name = std.mem.sliceTo(&t.?.name, 0);
                if (std.mem.startsWith(u8, name, "a.ve.") or std.mem.startsWith(u8, name, "a.cenc.")) {
                    const loaded = try tl.getTensor(@ptrCast(&t.?.name), true);
                    const gop = try model.cbx_tensors.getOrPut(std.heap.c_allocator, name);
                    if (!gop.found_existing) gop.key_ptr.* = try std.heap.c_allocator.dupe(u8, name);
                    gop.value_ptr.* = loaded;
                }
            }
        },
        // `PROJECTOR_TYPE_CHATTERBOX` (mtmd-model.cpp:2574 @b1b770d90)
        .chatterbox => try loadChatterbox(tl),
        .pockettts_gen => {
            const flow = &model.flow;
            flow.input_proj_w = try get(tl, mod.names.TN_A_GEN_FLOW_INPUT_PROJ, .{W}, true);
            flow.input_proj_b = try get(tl, mod.names.TN_A_GEN_FLOW_INPUT_PROJ, .{B}, true);
            flow.cond_embd_w = try get(tl, mod.names.TN_A_GEN_FLOW_COND_EMBD, .{W}, true);
            flow.cond_embd_b = try get(tl, mod.names.TN_A_GEN_FLOW_COND_EMBD, .{B}, true);
            flow.final_ada_w = try get(tl, mod.names.TN_A_GEN_FLOW_FINAL_ADA, .{W}, true);
            flow.final_ada_b = try get(tl, mod.names.TN_A_GEN_FLOW_FINAL_ADA, .{B}, true);
            flow.final_proj_w = try get(tl, mod.names.TN_A_GEN_FLOW_FINAL_PROJ, .{W}, true);
            flow.final_proj_b = try get(tl, mod.names.TN_A_GEN_FLOW_FINAL_PROJ, .{B}, true);

            try resizeList(mod.model.FlowNet.TimeEmbd, &flow.time, 2);
            for (flow.time.items, 0..) |*t, iu| {
                const i: c_int = @intCast(iu);
                t.freqs = try get(tl, mod.names.TN_A_GEN_FLOW_TIME_FREQS, .{i}, true);
                t.up_w = try get(tl, mod.names.TN_A_GEN_FLOW_TIME_UP, .{ i, W }, true);
                t.up_b = try get(tl, mod.names.TN_A_GEN_FLOW_TIME_UP, .{ i, B }, true);
                t.down_w = try get(tl, mod.names.TN_A_GEN_FLOW_TIME_DOWN, .{ i, W }, true);
                t.down_b = try get(tl, mod.names.TN_A_GEN_FLOW_TIME_DOWN, .{ i, B }, true);
                t.norm = try get(tl, mod.names.TN_A_GEN_FLOW_TIME_NORM, .{i}, true);
            }

            // one AdaLN block per flow depth, the count is only known from the tensors
            var il: c_int = 0;
            while (true) : (il += 1) {
                const probe = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_NORM, .{ il, W }, false);
                if (probe == null) break;
                var blk: mod.model.FlowNet.Block = .{};
                blk.norm_w = probe;
                blk.norm_b = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_NORM, .{ il, B }, true);
                blk.up_w = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_UP, .{ il, W }, true);
                blk.up_b = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_UP, .{ il, B }, true);
                blk.down_w = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_DOWN, .{ il, W }, true);
                blk.down_b = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_DOWN, .{ il, B }, true);
                blk.ada_w = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_ADA, .{ il, W }, true);
                blk.ada_b = try get(tl, mod.names.TN_A_GEN_FLOW_BLK_ADA, .{ il, B }, true);
                try flow.blocks.append(std.heap.c_allocator, blk);
            }

            model.gen_out_eos_w = try get(tl, mod.names.TN_A_GEN_OUT_EOS, .{W}, true);
            model.gen_out_eos_b = try get(tl, mod.names.TN_A_GEN_OUT_EOS, .{B}, true);
            model.gen_input_lin_w = try get(tl, mod.names.TN_A_GEN_INPUT_LINEAR, .{W}, true);
            model.gen_emb_mean = try tl.getTensor(mod.names.TN_A_GEN_EMB_MEAN, true);
            model.gen_emb_std = try tl.getTensor(mod.names.TN_A_GEN_EMB_STD, true);

            // mimi decoder
            model.gen_quant_out_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_OUT, .{W}, true);
            model.gen_upsample_w = try get(tl, mod.names.TN_A_GEN_WAV_UPSAMPLE, .{W}, true);
            try tl.loadSeanet(&model.seanet, true);
            try resizeList(mod.model.Layer, &model.gen_tfm_layers, @intCast(hparams.n_layer));
            const p = s("a.gen.wav.tfm");
            for (model.gen_tfm_layers.items, 0..) |*layer, ilu| {
                const l: c_int = @intCast(ilu);
                layer.ln_1_w = try get(tl, mod.names.TN_LN_1, .{ p, l, W }, true);
                layer.ln_1_b = try get(tl, mod.names.TN_LN_1, .{ p, l, B }, true);
                layer.q_w = try get(tl, mod.names.TN_ATTN_Q, .{ p, l, W }, true);
                layer.k_w = try get(tl, mod.names.TN_ATTN_K, .{ p, l, W }, true);
                layer.v_w = try get(tl, mod.names.TN_ATTN_V, .{ p, l, W }, true);
                layer.o_w = try get(tl, mod.names.TN_ATTN_OUTPUT, .{ p, l, W }, true);
                layer.ls_1_w = try get(tl, mod.names.TN_LS_1, .{ p, l, W }, true);
                layer.ln_2_w = try get(tl, mod.names.TN_LN_2, .{ p, l, W }, true);
                layer.ln_2_b = try get(tl, mod.names.TN_LN_2, .{ p, l, B }, true);
                layer.ff_up_w = try get(tl, mod.names.TN_FFN_UP, .{ p, l, W }, true);
                layer.ff_down_w = try get(tl, mod.names.TN_FFN_DOWN, .{ p, l, W }, true);
                layer.ls_2_w = try get(tl, mod.names.TN_LS_2, .{ p, l, W }, true);
            }

            const ctx_gguf = tl.loader.ctx_gguf;
            var i: i64 = 0;
            while (i < llama.impl.llama_c.gguf_get_n_tensors(ctx_gguf)) : (i += 1) {
                const name = llama.impl.llama_c.gguf_get_tensor_name(ctx_gguf, i);
                if (!std.mem.startsWith(u8, std.mem.span(name), "voice.")) continue;
                const t: *const llama.impl.llama_c.ggml_tensor = llama.impl.llama_c.ggml_get_tensor(tl.loader.ctx_meta, name);
                if (t.type != llama.impl.llama_c.GGML_TYPE_F32 or llama.impl.llama_c.ggml_n_dims(t) > 2 or t.ne[0] != model.gen_input_lin_w.?.ne[1]) {
                    return mod.clip.throwErr("%s: invalid voice tensor %s\n", .{ "load_tensors", name });
                }
                const embd = &(try model.gen_voices.getOrPut(std.mem.span(name + 6))).rows;
                const old_len = embd.items.len;
                try embd.resize(std.heap.c_allocator, @intCast(llama.impl.llama_c.ggml_nelements(t)));
                if (embd.items.len > old_len) @memset(embd.items[old_len..], 0);
                if (!tl.readTensorBytes(std.mem.span(name), std.mem.sliceAsBytes(embd.items)[0..llama.impl.llama_c.ggml_nbytes(t)])) {
                    return mod.clip.throwErr("%s: failed to read %s\n", .{ "load_tensors", name });
                }
            }
            if (model.gen_voices.count() != 0) {
                mod.impl.logInf("%s: %zu predefined voices\n", .{ "load_tensors", model.gen_voices.count() });
            }
        },
        .lfm3a_gen => {
            // must match the "tts" chat template, see conversion/lfm3.py
            // note for contributor: this is one-off fix, do not repeat this for future models
            _ = try model.gen_voices.getOrPut("female");
            _ = try model.gen_voices.getOrPut("male");

            // depthformer, the generic loop above loaded the a.gen.code.blk.* tensors
            for (model.layers.items, 0..) |*layer, il| {
                layer.attn_gate_w = try get(tl, mod.names.TN_ATTN_GATE, .{ prefix, ci(il), W }, true);
            }
            model.gen_code_proj_in_w = try get(tl, mod.names.TN_A_GEN_CODE_PROJ_IN, .{W}, true);
            model.gen_code_proj_in_b = try get(tl, mod.names.TN_A_GEN_CODE_PROJ_IN, .{B}, true);
            model.gen_code_embd_w = try get(tl, mod.names.TN_A_GEN_CODE_EMBD, .{W}, true); // tied head
            model.gen_code_out_embd_w = try get(tl, mod.names.TN_A_GEN_CODE_OUT_EMBD, .{W}, true);
            model.gen_code_norm_w = try get(tl, mod.names.TN_A_GEN_CODE_NORM, .{W}, true);
            model.gen_code_head_norm_w = try get(tl, mod.names.TN_A_GEN_CODE_HEAD_NORM, .{W}, true);
            model.gen_code_mask_embd = try tl.getTensor(mod.names.TN_A_GEN_CODE_MASK_EMBD, true);

            // codec decoder: FSQ level table, one projection over all codebooks, LFM3 decoder -> polar STFT frames
            const c2w = &model.c2w;
            c2w.quant_first_cb_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_FIRST_CB, .{W}, true);
            c2w.quant_first_out_w = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_FIRST_OUT, .{W}, true);
            c2w.quant_first_out_b = try get(tl, mod.names.TN_A_GEN_WAV_QUANT_FIRST_OUT, .{B}, true);
            c2w.tfm_in_proj_w = try get(tl, mod.names.TN_A_GEN_WAV_TFM_IN_PROJ, .{W}, true);
            c2w.tfm_in_proj_b = try get(tl, mod.names.TN_A_GEN_WAV_TFM_IN_PROJ, .{B}, true);
            c2w.tfm_out_proj_w = try get(tl, mod.names.TN_A_GEN_WAV_TFM_OUT_PROJ, .{W}, true);
            c2w.tfm_out_proj_b = try get(tl, mod.names.TN_A_GEN_WAV_TFM_OUT_PROJ, .{B}, true);
            c2w.tfm_output_norm_w = try get(tl, mod.names.TN_A_GEN_WAV_TFM_OUT_NORM, .{W}, true);

            // each layer is a short conv block or a gated attention block, both with sandwich norms
            try resizeList(mod.model.Layer, &c2w.tfm_layers, @intCast(hparams.wav_tfm_n_layer));
            const pw: [*:0]const u8 = "a.gen.wav.tfm";
            for (c2w.tfm_layers.items, 0..) |*layer, ilu| {
                const il = ci(ilu);
                layer.ln_1_w = try get(tl, mod.names.TN_LN_1, .{ pw, il, W }, true);
                layer.attn_post_norm_w = try get(tl, mod.names.TN_ATTN_POST_NORM, .{ pw, il, W }, true);
                layer.ln_2_w = try get(tl, mod.names.TN_LN_2, .{ pw, il, W }, true);
                layer.ff_post_norm_w = try get(tl, mod.names.TN_FFN_POST_NORM, .{ pw, il, W }, true);
                layer.ff_gate_w = try get(tl, mod.names.TN_FFN_GATE, .{ pw, il, W }, true);
                layer.ff_up_w = try get(tl, mod.names.TN_FFN_UP, .{ pw, il, W }, true);
                layer.ff_down_w = try get(tl, mod.names.TN_FFN_DOWN, .{ pw, il, W }, true);
                layer.conv_pw1_w = try get(tl, mod.names.TN_CONV_PW1, .{ pw, il, W }, false);
                layer.conv_dw_w = try get(tl, mod.names.TN_CONV_DW, .{ pw, il, W }, false);
                layer.conv_pw2_w = try get(tl, mod.names.TN_CONV_PW2, .{ pw, il, W }, false);
                layer.q_w = try get(tl, mod.names.TN_ATTN_Q, .{ pw, il, W }, false);
                layer.k_w = try get(tl, mod.names.TN_ATTN_K, .{ pw, il, W }, false);
                layer.v_w = try get(tl, mod.names.TN_ATTN_V, .{ pw, il, W }, false);
                layer.o_w = try get(tl, mod.names.TN_ATTN_OUTPUT, .{ pw, il, W }, false);
                layer.q_norm = try get(tl, mod.names.TN_ATTN_Q_NORM, .{ pw, il, W }, false);
                layer.k_norm = try get(tl, mod.names.TN_ATTN_K_NORM, .{ pw, il, W }, false);
                layer.attn_gate_w = try get(tl, mod.names.TN_ATTN_GATE, .{ pw, il, W }, false);
                const is_conv = layer.conv_pw1_w != null and layer.conv_dw_w != null and layer.conv_pw2_w != null;
                const is_attn = layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null and layer.q_norm != null and layer.k_norm != null and layer.attn_gate_w != null;
                if (is_conv == is_attn) {
                    return mod.clip.throwErr("%s: codec layer %d is neither a conv nor an attention block\n", .{ func, il });
                }
                if (is_attn) {
                    const d_head: c_int = @intCast(mod.graph.ne(layer.q_norm, 0));
                    hparams.wav_tfm_n_head = @intCast(@divTrunc(mod.graph.ne(layer.q_w, 1), d_head));
                    hparams.wav_tfm_n_head_kv = @intCast(@divTrunc(mod.graph.ne(layer.k_w, 1), d_head));
                }
            }
        },
        .qwen3tts_gen => {
            // code_predictor
            model.gen_code_proj_in_w = try get(tl, mod.names.TN_A_GEN_CODE_PROJ_IN, .{W}, false);
            model.gen_code_proj_in_b = try get(tl, mod.names.TN_A_GEN_CODE_PROJ_IN, .{B}, false);
            model.gen_code_embd_w = try get(tl, mod.names.TN_A_GEN_CODE_EMBD, .{W}, true);
            model.gen_code_head_w = try get(tl, mod.names.TN_A_GEN_CODE_HEAD, .{W}, true);
            model.gen_code_out_embd_w = try get(tl, mod.names.TN_A_GEN_CODE_OUT_EMBD, .{W}, true);
            model.gen_code_norm_w = try get(tl, mod.names.TN_A_GEN_CODE_NORM, .{W}, true);

            // code2wav: RVQ codes -> raw PCM, lives in the same ctx as code_predictor
            try loadCode2wav(tl, true);
        },
        .drama3_gen => {
            // generic a.gen.code.blk.* tensors are loaded by the loop above
            try resizeList(mod.model.Drama3Layer, &model.drama3_layers, @intCast(hparams.n_layer));
            for (model.drama3_layers.items, 0..) |*dl, ilu| {
                const il: c_int = @intCast(ilu);
                model.layers.items[ilu].attn_gate_w = try get(tl, mod.names.TN_ATTN_GATE, .{ prefix, il, W }, true);
                dl.enorm_w = try get(tl, mod.names.TN_A_GEN_CODE_ENORM, .{ il, W }, true);
                dl.hnorm_w = try get(tl, mod.names.TN_A_GEN_CODE_HNORM, .{ il, W }, true);
                dl.eh_proj_w = try get(tl, mod.names.TN_A_GEN_CODE_EH_PROJ, .{ il, W }, true);
                dl.embd_w = try get(tl, mod.names.TN_A_GEN_CODE_BLK_EMBD, .{ il, W }, true);
                dl.out_norm_w = try get(tl, mod.names.TN_A_GEN_CODE_BLK_OUT_NORM, .{ il, W }, true);
            }
            model.gen_code_embd_w = try get(tl, mod.names.TN_A_GEN_CODE_EMBD, .{W}, true); // merged 3D, one table per codebook
            model.gen_code_head_w = try get(tl, mod.names.TN_A_GEN_CODE_HEAD, .{W}, true); // merged 3D, codebooks 1..
            model.gen_code_out_embd_w = try get(tl, mod.names.TN_A_GEN_CODE_OUT_EMBD, .{W}, true); // backbone rows of the semantic band
            try loadCode2wav(tl, false);
        },
        .soprano => {
            const v = &model.vocos;
            v.input_w = try get(tl, mod.names.TN_A_GEN_WAV_INPUT, .{W}, true);
            v.input_b = try get(tl, mod.names.TN_A_GEN_WAV_INPUT, .{B}, true);
            v.norm_w = try get(tl, mod.names.TN_A_GEN_WAV_NORM, .{W}, true);
            v.norm_b = try get(tl, mod.names.TN_A_GEN_WAV_NORM, .{B}, true);
            v.output_norm_w = try get(tl, mod.names.TN_A_GEN_WAV_OUTPUT_NORM, .{W}, true);
            v.output_norm_b = try get(tl, mod.names.TN_A_GEN_WAV_OUTPUT_NORM, .{B}, true);
            v.output_w = try get(tl, mod.names.TN_A_GEN_WAV_OUTPUT, .{W}, true);
            v.output_b = try get(tl, mod.names.TN_A_GEN_WAV_OUTPUT, .{B}, true);
            try resizeList(mod.model.Code2Wav.UpsampleBlock, &v.blocks, @intCast(hparams.n_layer));
            for (v.blocks.items, 0..) |*b, ilu| {
                const il = ci(ilu);
                b.dwconv_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_DWCONV, .{ il, W }, true);
                b.dwconv_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_DWCONV, .{ il, B }, true);
                b.norm_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_NORM, .{ il, W }, true);
                b.norm_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_NORM, .{ il, B }, true);
                b.pw1_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW1, .{ il, W }, true);
                b.pw1_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW1, .{ il, B }, true);
                b.pw2_w = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW2, .{ il, W }, true);
                b.pw2_b = try get(tl, mod.names.TN_A_GEN_WAV_UP_PW2, .{ il, B }, true);
                b.gamma = try get(tl, mod.names.TN_A_GEN_WAV_UP_GAMMA, .{il}, true);
            }
        },
        .voxtral, .voxtral_rt => {
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, mod.names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, mod.names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
        },
        .music_flamingo => {
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, mod.names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, mod.names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
        },
        .internvl => {
            model.mm_0_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(1), B }, true);
            model.mm_3_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
            model.mm_3_b = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(3), B }, true);
        },
        .nemotron_v2_vl, .nemotron_omni => {
            model.mm_0_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_3_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
        },
        .glma => {
            model.conv1d_1_w = try get(tl, mod.names.TN_CONV1D, .{ ci(1), W }, true);
            model.conv1d_1_b = try get(tl, mod.names.TN_CONV1D, .{ ci(1), B }, true);
            model.conv1d_2_w = try get(tl, mod.names.TN_CONV1D, .{ ci(2), W }, true);
            model.conv1d_2_b = try get(tl, mod.names.TN_CONV1D, .{ ci(2), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), B }, true);
            model.mm_norm_pre_w = try get(tl, mod.names.TN_MM_NORM_PRE, .{W}, true);
            model.mm_norm_pre_b = try get(tl, mod.names.TN_MM_NORM_PRE, .{B}, true);
            model.mm_boi = try get(tl, mod.names.TN_TOK_BOI, .{}, true);
            model.mm_eoi = try get(tl, mod.names.TN_TOK_EOI, .{}, true);
        },
        .llama4 => {
            model.mm_model_proj = try get(tl, mod.names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_model_mlp_1_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_model_mlp_2_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(2), W }, true);
        },
        .cogvlm => {
            model.mm_model_proj = try get(tl, mod.names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_post_fc_norm_w = try get(tl, mod.names.TN_MM_POST_FC_NORM, .{W}, true);
            model.mm_post_fc_norm_b = try get(tl, mod.names.TN_MM_POST_FC_NORM, .{B}, true);
            model.mm_h_to_4h_w = try get(tl, mod.names.TN_MM_H_TO_4H, .{W}, true);
            model.mm_gate_w = try get(tl, mod.names.TN_MM_GATE, .{W}, true);
            model.mm_4h_to_h_w = try get(tl, mod.names.TN_MM_4H_TO_H, .{W}, true);
            model.mm_boi = try tl.getTensor(mod.names.TN_TOK_BOI, true);
            model.mm_eoi = try tl.getTensor(mod.names.TN_TOK_EOI, true);
        },
        .hunyuanvl => {
            // proj.0 -> mm.0 (conv1), proj.2 -> mm.2 (conv2), mlp -> mm.model.fc (linear)
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
            model.mm_model_proj = try get(tl, mod.names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_model_proj_b = try get(tl, mod.names.TN_MM_PROJECTOR, .{B}, true);
            model.mm_pre_norm_w = try get(tl, mod.names.TN_MM_PRE_NORM, .{W}, true);
            model.mm_post_norm_w = try get(tl, mod.names.TN_MM_POST_NORM, .{W}, true);
            model.mm_img_begin = try tl.getTensor(mod.names.TN_MM_IMG_BEGIN, true);
            model.mm_img_end = try tl.getTensor(mod.names.TN_MM_IMG_END, true);
            model.image_newline = try tl.getTensor(mod.names.TN_IMAGE_NEWLINE, true);
            model.view_seperator = try tl.getTensor(mod.names.TN_IMAGE_SEPERATOR, false);
            // the position embedding is resampled on the host as F32
            if (model.position_embeddings == null or model.position_embeddings.?.type != llama.impl.llama_c.GGML_TYPE_F32) {
                var pb: [256]u8 = undefined;
                return mod.clip.throwErr("%s: HunyuanVL requires an F32 %s\n", .{ func, mod.clip.fmtKey(&pb, mod.names.TN_POS_EMBD, prefix) });
            }
        },
        .janus_pro => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(1), B }, true);
        },
        .phi4 => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
            model.mm_2_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), W }, true);
            model.mm_2_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(2), B }, true);
        },
        .deepseekocr, .deepseekocr2 => {
            model.pos_embed = try get(tl, mod.names.TN_SAM_POS_EMBD, .{W}, true);
            model.patch_embed_proj_w = try get(tl, mod.names.TN_SAM_PATCH_EMBD, .{W}, true);
            model.patch_embed_proj_b = try get(tl, mod.names.TN_SAM_PATCH_EMBD, .{B}, true);
            model.n_sam_layers = hparams.sam_n_layer;
            try resizeList(mod.model.Layer, &model.sam_layers, @intCast(model.n_sam_layers));
            for (model.sam_layers.items, 0..) |*layer, ilu| {
                const il: c_int = @intCast(ilu);
                layer.qkv_w = try get(tl, mod.names.TN_SAM_ATTN_QKV, .{ il, W }, true);
                layer.qkv_b = try get(tl, mod.names.TN_SAM_ATTN_QKV, .{ il, B }, true);
                layer.o_w = try get(tl, mod.names.TN_SAM_ATTN_OUT, .{ il, W }, true);
                layer.o_b = try get(tl, mod.names.TN_SAM_ATTN_OUT, .{ il, B }, true);
                layer.ln_1_w = try get(tl, mod.names.TN_SAM_PRE_NORM, .{ il, W }, true);
                layer.ln_1_b = try get(tl, mod.names.TN_SAM_PRE_NORM, .{ il, B }, true);
                layer.ln_2_w = try get(tl, mod.names.TN_SAM_POST_NORM, .{ il, W }, true);
                layer.ln_2_b = try get(tl, mod.names.TN_SAM_POST_NORM, .{ il, B }, true);
                layer.rel_pos_h = try get(tl, mod.names.TN_SAM_ATTN_POS_H, .{ il, W }, true);
                layer.rel_pos_w = try get(tl, mod.names.TN_SAM_ATTN_POS_W, .{ il, W }, true);
                layer.ff_up_w = try get(tl, mod.names.TN_SAM_FFN_UP, .{ il, W }, true);
                layer.ff_up_b = try get(tl, mod.names.TN_SAM_FFN_UP, .{ il, B }, true);
                layer.ff_down_w = try get(tl, mod.names.TN_SAM_FFN_DOWN, .{ il, W }, true);
                layer.ff_down_b = try get(tl, mod.names.TN_SAM_FFN_DOWN, .{ il, B }, true);
            }
            model.neck_0_w = try get(tl, mod.names.TN_SAM_NECK, .{ ci(0), W }, true);
            model.neck_1_b = try get(tl, mod.names.TN_SAM_NECK, .{ ci(1), B }, true);
            model.neck_1_w = try get(tl, mod.names.TN_SAM_NECK, .{ ci(1), W }, true);
            model.neck_2_w = try get(tl, mod.names.TN_SAM_NECK, .{ ci(2), W }, true);
            model.neck_3_b = try get(tl, mod.names.TN_SAM_NECK, .{ ci(3), B }, true);
            model.neck_3_w = try get(tl, mod.names.TN_SAM_NECK, .{ ci(3), W }, true);
            model.net_2 = try get(tl, mod.names.TN_SAM_NET, .{ ci(2), W }, true);
            model.net_3 = try get(tl, mod.names.TN_SAM_NET, .{ ci(3), W }, true);
            model.image_newline = try tl.getTensor(mod.names.TN_IMAGE_NEWLINE, false);
            model.view_seperator = try tl.getTensor(mod.names.TN_IMAGE_SEPERATOR, true);
            model.mm_fc_w = try get(tl, mod.names.TN_MM_PROJECTOR, .{W}, true);
            model.mm_fc_b = try get(tl, mod.names.TN_MM_PROJECTOR, .{B}, true);
            model.resample_query_768 = try get(tl, mod.names.TN_RESMPL_QUERY, .{ ci(768), W }, false);
            model.resample_query_1024 = try get(tl, mod.names.TN_RESMPL_QUERY, .{ ci(1024), W }, false);
        },
        .gemma4a => {
            for (0..2) |iu| {
                const i: c_int = @intCast(iu);
                model.sscp_conv_w[iu] = try get(tl, mod.names.TN_A_CONV1D, .{ i, W }, true);
                model.sscp_conv_b[iu] = try get(tl, mod.names.TN_A_CONV1D, .{ i, B }, false);
                model.sscp_norm_w[iu] = try get(tl, mod.names.TN_A_CONV1D_NORM, .{ i, W }, false);
            }
            model.sscp_inp_proj_w = try get(tl, mod.names.TN_A_INP_PROJ, .{W}, true);
            model.sscp_inp_proj_b = try get(tl, mod.names.TN_A_INP_PROJ, .{B}, false);
            model.audio_out_proj_w = try get(tl, mod.names.TN_A_OUT_PROJ, .{W}, false);
            model.audio_out_proj_b = try get(tl, mod.names.TN_A_OUT_PROJ, .{B}, false);
            // audio multimodal embedder (mm.a.* namespace, not mm.*)
            model.mm_soft_emb_norm_w = try get(tl, mod.names.TN_A_MM_SOFT_EMB_N, .{W}, false);
            model.mm_input_proj_w = try get(tl, mod.names.TN_A_MM_INP_PROJ, .{W}, false);

            // Per-layer tensors NOT loaded by the generic loop above
            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                // Gemma4 audio conformer-specific tensors
                layer.ff_norm_w = try get(tl, mod.names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.attn_pre_norm_w = try get(tl, mod.names.TN_A_ATTN_PRE_NORM, .{ prefix, il, W }, false);
                layer.per_dim_scale_w = try get(tl, mod.names.TN_A_PER_DIM_SCALE, .{ prefix, il, W }, false);
                layer.per_dim_k_scale_w = try get(tl, mod.names.TN_A_PER_DIM_K_SCALE, .{ prefix, il, W }, false);
                layer.attn_k_rel_w = try get(tl, mod.names.TN_A_ATTN_K_REL, .{ prefix, il, W }, false);

                // Convolution module
                // Note: conv_norm / norm_conv are swapped in GGUF due to
                // upstream tensor_mapping.py, so we load them in reverse order
                layer.norm_conv_w = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, W }, false);
                layer.norm_conv_b = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, B }, false);
                layer.conv_pw1_w = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, B }, false);
                layer.conv_dw_w = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_dw_b = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, B }, false);
                layer.conv_norm_w = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, W }, false);
                layer.conv_norm_b = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, B }, false);
                layer.conv_pw2_w = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, B }, false);

                // FFN2 (second half-step)
                layer.ff_norm_1_w = try get(tl, mod.names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_up_1_w = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, B }, false);
                layer.ff_down_1_w = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, B }, false);
                layer.ff_post_norm_1_w = try get(tl, mod.names.TN_A_FFN_POST_NORM_1, .{ prefix, il, W }, false);
            }

            // Load clamp info for ClippableLinear AFTER all tensors are loaded
            try loadClampInfo(tl);
        },
        .gemma4ua => {
            model.mm_input_proj_w = try get(tl, mod.names.TN_A_MM_INP_PROJ, .{W}, true);
        },
        .lfm2a, .d1omni_a => {
            for ([_]usize{ 0, 2, 3, 5, 6 }) |i| {
                model.pre_encode_conv_X_w[i] = try get(tl, mod.names.TN_CONV1D, .{ ci(i), W }, true);
                model.pre_encode_conv_X_b[i] = try get(tl, mod.names.TN_CONV1D, .{ ci(i), B }, true);
            }
            model.pre_encode_out_w = try get(tl, mod.names.TN_PRE_ENCODE_OUT, .{W}, true);
            model.pre_encode_out_b = try get(tl, mod.names.TN_PRE_ENCODE_OUT, .{B}, true);

            model.mm_0_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_3_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), W }, true);
            model.mm_3_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), B }, true);

            // residual block after the projector: norm, down, up
            if (model.proj_type == .d1omni_a) {
                model.mm_4_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(4), W }, true);
                model.mm_4_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(4), B }, true);
                model.mm_5_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(5), W }, true);
                model.mm_5_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(5), B }, true);
                model.mm_6_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(6), W }, true);
                model.mm_6_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(6), B }, true);
            }

            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                layer.ff_norm_w = try get(tl, mod.names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.ff_norm_b = try get(tl, mod.names.TN_FFN_NORM, .{ prefix, il, B }, true);
                layer.ff_norm_1_w = try get(tl, mod.names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_norm_1_b = try get(tl, mod.names.TN_FFN_NORM_1, .{ prefix, il, B }, true);
                layer.ff_up_1_w = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, B }, true);
                layer.ff_down_1_w = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, B }, true);

                layer.pos_bias_u = try get(tl, mod.names.TN_POS_BIAS_U, .{ prefix, il }, true);
                layer.pos_bias_v = try get(tl, mod.names.TN_POS_BIAS_V, .{ prefix, il }, true);

                layer.norm_conv_w = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, W }, true);
                layer.norm_conv_b = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, B }, true);

                layer.linear_pos_w = try get(tl, mod.names.TN_LINEAR_POS, .{ prefix, il, W }, true);

                layer.conv_norm_w = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, W }, true);
                layer.conv_norm_b = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, B }, true);
                layer.conv_dw_w = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_dw_b = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, B }, true);
                layer.conv_pw1_w = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, B }, true);
                layer.conv_pw2_w = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, B }, true);
            }
        },
        .lfm3a => {
            for ([_]usize{ 0, 2, 3, 5, 6 }) |i| {
                model.pre_encode_conv_X_w[i] = try get(tl, mod.names.TN_CONV1D, .{ ci(i), W }, true);
                model.pre_encode_conv_X_b[i] = try get(tl, mod.names.TN_CONV1D, .{ ci(i), B }, true);
            }
            model.pre_encode_out_w = try get(tl, mod.names.TN_PRE_ENCODE_OUT, .{W}, true);
            model.pre_encode_out_b = try get(tl, mod.names.TN_PRE_ENCODE_OUT, .{B}, true);

            model.mm_0_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(0), B }, true);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, true);
            model.mm_1_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), B }, true);
            model.mm_3_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), W }, true);
            model.mm_3_b = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(3), B }, true);

            if (model.post_ln_w == null) {
                var nb: [256]u8 = undefined;
                _ = stdlib.libc.snprintf(&nb, nb.len, mod.names.TN_LN_POST, prefix, W);
                return mod.clip.throwErr("%s: missing tensor %s\n", .{ func, @as([*:0]const u8, @ptrCast(&nb)) });
            }

            // each layer is either a short conv block or a complete attention block
            for (model.layers.items, 0..) |*layer, ilu| {
                const il = ci(ilu);
                layer.conv_pw1_w = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, W }, false);
                layer.conv_dw_w = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, W }, false);
                layer.conv_pw2_w = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, W }, false);
                const is_conv = layer.conv_pw1_w != null and layer.conv_dw_w != null and layer.conv_pw2_w != null;
                const is_attn = layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null and layer.q_norm != null and layer.k_norm != null;
                if (is_conv == is_attn) {
                    return mod.clip.throwErr("%s: lfm3a layer %d is neither a complete conv nor a complete attention block\n", .{ func, il });
                }
            }
        },
        .parakeet => {
            hparams.mel_filters.deinit(std.heap.c_allocator);
            hparams.mel_filters = try tl.getVector(mod.names.TN_MEL_FILTERS);
            hparams.window.deinit(std.heap.c_allocator);
            hparams.window = try tl.getVector(mod.names.TN_WINDOW);

            // Subsampling layers (conv1d)
            for ([_]usize{ 0, 2, 3, 5, 6 }) |i| {
                model.pre_encode_conv_X_w[i] = try get(tl, mod.names.TN_CONV1D, .{ ci(i), W }, true);
                model.pre_encode_conv_X_b[i] = try get(tl, mod.names.TN_CONV1D, .{ ci(i), B }, true);
            }
            model.pre_encode_out_w = try get(tl, mod.names.TN_PRE_ENCODE_OUT, .{W}, true);
            model.pre_encode_out_b = try get(tl, mod.names.TN_PRE_ENCODE_OUT, .{B}, true);

            // Projection layers
            model.mm_norm_pre_w = try get(tl, mod.names.TN_MM_NORM_PRE, .{W}, false);
            model.mm_0_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(1), W }, false);
            model.mm_1_w = try get(tl, mod.names.TN_MM_AUDIO_MLP, .{ ci(2), W }, false);

            // Encoder layers
            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                // Relative position encoding
                layer.linear_pos_w = try get(tl, mod.names.TN_LINEAR_POS, .{ prefix, il, W }, true);
                layer.pos_bias_u = try get(tl, mod.names.TN_POS_BIAS_U, .{ prefix, il }, true);
                layer.pos_bias_v = try get(tl, mod.names.TN_POS_BIAS_V, .{ prefix, il }, true);

                // Convolution module
                layer.conv_pw1_w = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, B }, false);
                layer.conv_dw_w = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_dw_b = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, B }, false);
                layer.conv_norm_w = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, W }, true);
                layer.conv_norm_b = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, B }, true);
                layer.conv_norm_mean = try get(tl, mod.names.TN_CONV_NORM_MEAN, .{ prefix, il }, true);
                layer.conv_norm_var = try get(tl, mod.names.TN_CONV_NORM_VAR, .{ prefix, il }, true);
                layer.conv_pw2_w = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, B }, false);

                // Feed-forward networks
                layer.ff_norm_w = try get(tl, mod.names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.ff_norm_b = try get(tl, mod.names.TN_FFN_NORM, .{ prefix, il, B }, true);

                layer.ff_norm_1_w = try get(tl, mod.names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_norm_1_b = try get(tl, mod.names.TN_FFN_NORM_1, .{ prefix, il, B }, true);
                layer.ff_up_1_w = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, B }, false);
                layer.ff_down_1_w = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, B }, false);

                // Layer norms
                layer.norm_conv_w = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, W }, true);
                layer.norm_conv_b = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, B }, true);
            }

            model.mm_model_mlp_1_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(0), W }, true);
            model.mm_model_mlp_2_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(1), W }, true);
            model.mm_model_mlp_3_w = try get(tl, mod.names.TN_MVLM_PROJ_MLP, .{ ci(3), W }, true);
        },
        // PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE
        .nemotron_transcribe => {
            hparams.mel_filters.deinit(std.heap.c_allocator);
            hparams.mel_filters = try tl.getVector(mod.names.TN_MEL_FILTERS);
            hparams.window.deinit(std.heap.c_allocator);
            hparams.window = try tl.getVector(mod.names.TN_WINDOW);

            // ASR expert: blocks, pre_ln and post_ln come from the generic loader
            model.inp_proj_w = try get(tl, mod.names.TN_INP_PROJ, .{W}, true);
            for (model.layers.items, 0..) |*layer, ilu| {
                const il = ci(ilu);
                layer.q_norm_b = try get(tl, mod.names.TN_ATTN_Q_NORM, .{ prefix, il, B }, true);
                layer.k_norm_b = try get(tl, mod.names.TN_ATTN_K_NORM, .{ prefix, il, B }, true);
            }

            // speaker expert
            const sp: [*:0]const u8 = "a.spk";
            model.spk_inp_proj_w = try get(tl, mod.names.TN_A_SPK_INP_PROJ, .{W}, true);
            model.spk_pre_ln_w = try get(tl, mod.names.TN_LN_PRE, .{ sp, W }, true);
            model.spk_pre_ln_b = try get(tl, mod.names.TN_LN_PRE, .{ sp, B }, true);
            model.spk_post_ln_w = try get(tl, mod.names.TN_LN_POST, .{ sp, W }, true);
            model.spk_post_ln_b = try get(tl, mod.names.TN_LN_POST, .{ sp, B }, true);
            try resizeList(mod.model.Layer, &model.spk_layers, @intCast(hparams.spk_n_layer));
            for (model.spk_layers.items, 0..) |*layer, ilu| {
                const il = ci(ilu);
                layer.ln_1_w = try get(tl, mod.names.TN_LN_1, .{ sp, il, W }, true);
                layer.ln_1_b = try get(tl, mod.names.TN_LN_1, .{ sp, il, B }, true);
                layer.qkv_w = try get(tl, mod.names.TN_ATTN_QKV, .{ sp, il, W }, true);
                layer.o_w = try get(tl, mod.names.TN_ATTN_OUTPUT, .{ sp, il, W }, true);
                layer.o_b = try get(tl, mod.names.TN_ATTN_OUTPUT, .{ sp, il, B }, true);
                layer.ln_2_w = try get(tl, mod.names.TN_LN_2, .{ sp, il, W }, true);
                layer.ln_2_b = try get(tl, mod.names.TN_LN_2, .{ sp, il, B }, true);
                layer.ff_up_w = try get(tl, mod.names.TN_FFN_UP, .{ sp, il, W }, true);
                layer.ff_up_b = try get(tl, mod.names.TN_FFN_UP, .{ sp, il, B }, true);
                layer.ff_down_w = try get(tl, mod.names.TN_FFN_DOWN, .{ sp, il, W }, true);
                layer.ff_down_b = try get(tl, mod.names.TN_FFN_DOWN, .{ sp, il, B }, true);
            }

            // Sortformer head, fusion
            model.spk_head_proj_w = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("proj"), W }, true);
            model.spk_head_proj_b = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("proj"), B }, true);
            model.spk_head_up_w = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("upsample"), W }, true);
            model.spk_head_up_b = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("upsample"), B }, true);
            model.spk_head_fc1_w = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("fc1"), W }, true);
            model.spk_head_fc1_b = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("fc1"), B }, true);
            model.spk_head_fc2_w = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("fc2"), W }, true);
            model.spk_head_fc2_b = try get(tl, mod.names.TN_A_SPK_HEAD, .{ s("fc2"), B }, true);
            model.fusion_asr_norm_w = try get(tl, mod.names.TN_A_FUSION_NORM, .{ s("asr_norm"), W }, true);
            model.fusion_asr_norm_b = try get(tl, mod.names.TN_A_FUSION_NORM, .{ s("asr_norm"), B }, true);
            model.fusion_spk_norm_w = try get(tl, mod.names.TN_A_FUSION_NORM, .{ s("spk_norm"), W }, true);
            model.fusion_spk_norm_b = try get(tl, mod.names.TN_A_FUSION_NORM, .{ s("spk_norm"), B }, true);
            model.fusion_spk_kernel = try tl.getTensor(mod.names.TN_A_FUSION_KERNEL, true);

            model.spk_sil_emb.deinit(std.heap.c_allocator);
            model.spk_sil_emb = try tl.getVector(mod.names.TN_A_SPK_SIL_EMB);

            // the head layouts and the widths the graph and the speaker state use must match the tensors
            hparams.spk_n_embd = @truncate(mod.graph.ne(model.spk_inp_proj_w, 1));
            const n_embd: i64 = hparams.n_embd;
            const spk_n_embd: i64 = hparams.spk_n_embd;
            const d_head: i64 = if (hparams.n_head > 0) @divTrunc(hparams.n_embd, hparams.n_head) else 0;
            const d_head_spk: i64 = if (hparams.spk_n_head > 0) @divTrunc(hparams.spk_n_embd, hparams.spk_n_head) else 0;
            const n_spk: i64 = mod.graph.ne(model.spk_head_fc2_w, 1);
            // the graph splits each fused QKV into three views of the layer width
            var qkv_ok = true;
            for (model.layers.items) |*layer| {
                qkv_ok = qkv_ok and layer.qkv_w != null and mod.graph.ne(layer.qkv_w, 0) == n_embd and mod.graph.ne(layer.qkv_w, 1) == 3 * n_embd;
            }
            for (model.spk_layers.items) |*layer| {
                qkv_ok = qkv_ok and mod.graph.ne(layer.qkv_w, 0) == spk_n_embd and mod.graph.ne(layer.qkv_w, 1) == 3 * spk_n_embd;
            }
            if (!qkv_ok or mod.graph.ne(model.inp_proj_w, 1) != n_embd or d_head <= 0 or d_head * hparams.n_head != n_embd or
                (hparams.n_embd_head > 0 and hparams.n_embd_head != d_head) or
                hparams.audio_rope_dims <= 0 or @rem(hparams.audio_rope_dims, 2) != 0 or hparams.audio_rope_dims > d_head or
                d_head_spk <= 0 or @rem(d_head_spk, 2) != 0 or d_head_spk * hparams.spk_n_head != spk_n_embd or
                @as(i64, @intCast(model.spk_sil_emb.items.len)) != spk_n_embd or n_spk <= 0 or
                mod.graph.ne(model.fusion_spk_kernel, 0) != n_spk or mod.graph.ne(model.fusion_spk_kernel, 1) != n_embd or
                hparams.mel_filters.items.len != @as(usize, @bitCast(@as(isize, hparams.n_mel_bins))) *% @as(usize, @bitCast(@as(isize, @divTrunc(hparams.audio_n_fft, 2) + 1))) or
                hparams.window.items.len != @as(usize, @bitCast(@as(isize, hparams.audio_window_len))))
            {
                return mod.clip.throwErr("%s: hparams do not match the tensor shapes\n", .{func});
            }

            // projection to the text model
            model.mm_fc_w = try get(tl, mod.names.TN_MM_AUDIO_FC, .{W}, true);
            model.mm_fc_b = try get(tl, mod.names.TN_MM_AUDIO_FC, .{B}, true);
        },
        .granite_speech => {
            model.inp_proj_w = try get(tl, mod.names.TN_INP_PROJ, .{W}, true);
            model.inp_proj_b = try get(tl, mod.names.TN_INP_PROJ, .{B}, true);
            model.ctc_out_w = try get(tl, mod.names.TN_CTC_OUT, .{W}, true);
            model.ctc_out_b = try get(tl, mod.names.TN_CTC_OUT, .{B}, true);
            model.ctc_out_mid_w = try get(tl, mod.names.TN_CTC_OUT_MID, .{W}, true);
            model.ctc_out_mid_b = try get(tl, mod.names.TN_CTC_OUT_MID, .{B}, true);

            // per-layer tensors not loaded by the generic loop above
            var ilc: c_int = 0;
            while (ilc < hparams.n_layer) : (ilc += 1) {
                const layer = &model.layers.items[@intCast(ilc)];
                const il = ilc;

                layer.attn_rel_pos_emb = try get(tl, mod.names.TN_ATTN_REL_POS_EMB, .{ prefix, il }, true);

                layer.ff_norm_w = try get(tl, mod.names.TN_FFN_NORM, .{ prefix, il, W }, true);
                layer.ff_norm_b = try get(tl, mod.names.TN_FFN_NORM, .{ prefix, il, B }, true);

                layer.ff_norm_1_w = try get(tl, mod.names.TN_FFN_NORM_1, .{ prefix, il, W }, true);
                layer.ff_norm_1_b = try get(tl, mod.names.TN_FFN_NORM_1, .{ prefix, il, B }, true);
                layer.ff_up_1_w = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, W }, true);
                layer.ff_up_1_b = try get(tl, mod.names.TN_FFN_UP_1, .{ prefix, il, B }, true);
                layer.ff_down_1_w = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, W }, true);
                layer.ff_down_1_b = try get(tl, mod.names.TN_FFN_DOWN_1, .{ prefix, il, B }, true);

                layer.norm_conv_w = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, W }, true);
                layer.norm_conv_b = try get(tl, mod.names.TN_NORM_CONV, .{ prefix, il, B }, true);
                layer.conv_norm_w = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, W }, true);
                layer.conv_norm_b = try get(tl, mod.names.TN_CONV_NORM, .{ prefix, il, B }, true);
                layer.conv_dw_w = try get(tl, mod.names.TN_CONV_DW, .{ prefix, il, W }, true);
                layer.conv_pw1_w = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, W }, true);
                layer.conv_pw1_b = try get(tl, mod.names.TN_CONV_PW1, .{ prefix, il, B }, true);
                layer.conv_pw2_w = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, W }, true);
                layer.conv_pw2_b = try get(tl, mod.names.TN_CONV_PW2, .{ prefix, il, B }, true);
            }

            try resizeList(mod.model.QfBlock, &model.qf_proj_blocks, 1);
            const qf = &model.qf_proj_blocks.items[0];
            qf.qf_proj_query = try get(tl, mod.names.TN_QF_PROJ_QUERY, .{prefix}, true);
            qf.qf_proj_norm_w = try get(tl, mod.names.TN_QF_PROJ_NORM, .{ prefix, W }, true);
            qf.qf_proj_norm_b = try get(tl, mod.names.TN_QF_PROJ_NORM, .{ prefix, B }, true);
            qf.qf_proj_linear_w = try get(tl, mod.names.TN_QF_PROJ_LINEAR, .{ prefix, W }, true);
            qf.qf_proj_linear_b = try get(tl, mod.names.TN_QF_PROJ_LINEAR, .{ prefix, B }, true);

            const n_proj_layers = 2;
            try resizeList(mod.model.Layer, &qf.qf_proj_layers, n_proj_layers);
            for (qf.qf_proj_layers.items, 0..) |*pl, ilu| {
                try loadQfLayer(tl, pl, prefix, @intCast(ilu));
            }
        },
        .sensenova_u1 => {
            model.mm_0_w = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), W }, true);
            model.mm_0_b = try get(tl, mod.names.TN_LLAVA_PROJ, .{ ci(0), B }, true);
        },
        .granite4_vision => {
            // image_newline lives at the top-level.
            model.image_newline = try tl.getTensor(mod.names.TN_IMAGE_NEWLINE, true);

            // Load separate layerwise and spatial projector tensors
            const projector_count = hparams.feature_layers.items.len;
            try resizeList(mod.model.QfBlock, &model.qf_proj_blocks, projector_count);
            for (model.qf_proj_blocks.items, 0..) |*b, bidu| {
                // the C++ passes a `size_t` to `%d`; printf reads its low 32 bits
                const bid: c_int = @truncate(@as(i64, @intCast(bidu)));

                // non-layerwise tensors
                b.qf_proj_img_pos = try get(tl, mod.names.TN_MULTI_PROJ_IMG_POS, .{bid}, true);
                b.qf_proj_query = try get(tl, mod.names.TN_MULTI_PROJ_QUERY, .{ prefix, bid }, true);
                b.qf_proj_linear_w = try get(tl, mod.names.TN_MULTI_PROJ_LINEAR, .{ prefix, bid, W }, true);
                b.qf_proj_linear_b = try get(tl, mod.names.TN_MULTI_PROJ_LINEAR, .{ prefix, bid, B }, true);
                b.qf_proj_norm_w = try get(tl, mod.names.TN_MULTI_PROJ_NORM, .{ prefix, bid, W }, true);
                b.qf_proj_norm_b = try get(tl, mod.names.TN_MULTI_PROJ_NORM, .{ prefix, bid, B }, true);
                b.qf_proj_post_norm_w = try get(tl, mod.names.TN_MULTI_PROJ_POST_NORM, .{ prefix, bid, W }, true);
                b.qf_proj_post_norm_b = try get(tl, mod.names.TN_MULTI_PROJ_POST_NORM, .{ prefix, bid, B }, true);

                // laywerwise tensors
                // NOTE: If any model uses multi-layer qformers, this will need to change
                try resizeList(mod.model.Layer, &b.qf_proj_layers, 1);
                try loadQfLayer(tl, &b.qf_proj_layers.items[0], prefix, bid);
            }
        },
        else => mod.impl.ggmlAssert(false, "mtmd-model.cpp", 3511, "false && \"unknown projector type\""),
    }
}

/// One QFormer layer's tensors, as the granite speech and granite4 vision
/// arms load them at lines 3418 and 3474 of mtmd-model.cpp, indexed by `il`.
fn loadQfLayer(tl: *mod.clip.TensorLoader, pl: *mod.model.Layer, prefix: [*:0]const u8, il: c_int) mod.clip.Error!void {
    pl.q_w = try get(tl, mod.names.TN_QF_SELF_ATTN_Q, .{ prefix, il, W }, true);
    pl.q_b = try get(tl, mod.names.TN_QF_SELF_ATTN_Q, .{ prefix, il, B }, true);
    pl.k_w = try get(tl, mod.names.TN_QF_SELF_ATTN_K, .{ prefix, il, W }, true);
    pl.k_b = try get(tl, mod.names.TN_QF_SELF_ATTN_K, .{ prefix, il, B }, true);
    pl.v_w = try get(tl, mod.names.TN_QF_SELF_ATTN_V, .{ prefix, il, W }, true);
    pl.v_b = try get(tl, mod.names.TN_QF_SELF_ATTN_V, .{ prefix, il, B }, true);
    pl.o_w = try get(tl, mod.names.TN_QF_SELF_ATTN_O, .{ prefix, il, W }, true);
    pl.o_b = try get(tl, mod.names.TN_QF_SELF_ATTN_O, .{ prefix, il, B }, true);
    pl.ln_1_w = try get(tl, mod.names.TN_QF_SELF_ATTN_N, .{ prefix, il, W }, true);
    pl.ln_1_b = try get(tl, mod.names.TN_QF_SELF_ATTN_N, .{ prefix, il, B }, true);

    pl.cross_attn_q_w = try get(tl, mod.names.TN_QF_CROSS_ATTN_Q, .{ prefix, il, W }, true);
    pl.cross_attn_q_b = try get(tl, mod.names.TN_QF_CROSS_ATTN_Q, .{ prefix, il, B }, true);
    pl.cross_attn_k_w = try get(tl, mod.names.TN_QF_CROSS_ATTN_K, .{ prefix, il, W }, true);
    pl.cross_attn_k_b = try get(tl, mod.names.TN_QF_CROSS_ATTN_K, .{ prefix, il, B }, true);
    pl.cross_attn_v_w = try get(tl, mod.names.TN_QF_CROSS_ATTN_V, .{ prefix, il, W }, true);
    pl.cross_attn_v_b = try get(tl, mod.names.TN_QF_CROSS_ATTN_V, .{ prefix, il, B }, true);
    pl.cross_attn_o_w = try get(tl, mod.names.TN_QF_CROSS_ATTN_O, .{ prefix, il, W }, true);
    pl.cross_attn_o_b = try get(tl, mod.names.TN_QF_CROSS_ATTN_O, .{ prefix, il, B }, true);
    pl.cross_attn_norm_w = try get(tl, mod.names.TN_QF_CROSS_ATTN_N, .{ prefix, il, W }, true);
    pl.cross_attn_norm_b = try get(tl, mod.names.TN_QF_CROSS_ATTN_N, .{ prefix, il, B }, true);

    pl.ff_up_w = try get(tl, mod.names.TN_QF_FFN_UP, .{ prefix, il, W }, true);
    pl.ff_up_b = try get(tl, mod.names.TN_QF_FFN_UP, .{ prefix, il, B }, true);
    pl.ff_down_w = try get(tl, mod.names.TN_QF_FFN_DOWN, .{ prefix, il, W }, true);
    pl.ff_down_b = try get(tl, mod.names.TN_QF_FFN_DOWN, .{ prefix, il, B }, true);
    pl.ln_2_w = try get(tl, mod.names.TN_QF_FFN_NORM, .{ prefix, il, W }, true);
    pl.ln_2_b = try get(tl, mod.names.TN_QF_FFN_NORM, .{ prefix, il, B }, true);
}

/// `t->ne[i]` as the C++'s `int` return reads it.
inline fn neI(t: mod.graph.T, i: usize) c_int {
    return @truncate(mod.graph.ne(t, i));
}

/// Ports `n_mmproj_embd` (mtmd-model.cpp:4223 @b1b770d90), a member of
/// `mtmd_model`: the embedding size of the projector's output, which must
/// be the text model's.
pub fn nMmprojEmbd(m: *const mod.model.Model) c_int {
    return switch (m.proj_type) {
        .ldp => neI(m.mm_model_block_1_block_2_1_b, 0),
        .ldpv2 => neI(m.mm_model_peg_0_b, 0),
        .mlp, .phi4, .pixtral, .lightonocr, .dots_ocr, .dots3note_v, .dots3note_a => neI(m.mm_2_w, 1),
        .mlp_norm => neI(m.mm_3_b, 0),
        .minicpmv => neI(m.mm_model_proj, 1),
        .minicpmv4_6, .minicpmv4_7, .minicpm_ocr_1_0 => neI(m.mm_ffn_down_w, 1),
        .glm_edge => neI(m.mm_model_mlp_3_w, 1),
        .minimax_m3 => neI(m.mm_merger_fc2_b, 0),
        .muse_glimmer => neI(m.mm_2_w, 1),
        .qwen2vl, .qwen25vl, .magevl, .lfm3vl, .exaone4_5, .janus_pro, .youtuvl => neI(m.mm_1_b, 0),
        .sensenova_u1, .soprano => m.hparams.projection_dim,
        // main path + deepstack paths
        .qwen3vl => @truncate(mod.graph.ne(m.mm_1_b, 0) * (1 + @as(i64, m.n_deepstack_layers))),
        .ling3vl => neI(m.mm_1_b, 0),
        .mimovl => neI(m.mm_1_w, 1),
        .step3vl => neI(m.mm_model_proj, 1),
        .gemma3, .gemma3nv => neI(m.mm_input_proj_w, 0),
        .gemma4v, .gemma4uv, .gemma4a, .gemma4ua => neI(m.mm_input_proj_w, 1),
        .idefics3, .muse_spark_audio => neI(m.mm_fc_w, 1),
        .ultravox, .voxtral, .voxtral_rt, .music_flamingo => neI(m.mm_2_w, 1),
        .meralion => neI(m.mm_3_w, 1), // out_proj output dim
        .internvl, .nemotron_v2_vl, .nemotron_omni => neI(m.mm_3_w, 1),
        .llama4 => neI(m.mm_model_proj, 1),
        .qwen2a, .nemotron_transcribe => neI(m.mm_fc_w, 1),
        .qwen3a => neI(m.mm_2_w, 1),
        .glma, .lfm2, .d1omni_v, .kimivl, .paddleocr, .kimik25, .yasa2, .deepseek4v, .deepseek41v, .cohere2v => neI(m.mm_2_w, 1),
        .kimik31 => neI(m.mm_2_w, 1),
        .hunyuanvl => neI(m.mm_model_proj, 1),
        .cogvlm => neI(m.mm_4h_to_h_w, 1),
        .deepseekocr, .deepseekocr2 => neI(m.mm_fc_w, 1),
        .lfm2a => neI(m.position_embeddings, 0),
        .d1omni_a, .lfm3a => neI(m.mm_3_w, 1),
        .granite_speech => neI(m.qf_proj_blocks.items[0].qf_proj_linear_w, 1),
        // `size_t * int`, narrowed to the `int` return
        .granite4_vision => @truncate(@as(i64, @intCast(m.qf_proj_blocks.items.len *% @as(usize, @bitCast(@as(i64, m.hparams.projection_dim)))))),
        .glm4v, .glm5v => neI(m.mm_ffn_down_w, 1),
        .mimo_audio => neI(m.mm_2_w, 1),
        .qwen3tts_spkenc => neI(m.mm_fc_w, 2),
        .qwen3tts_gen, .drama3_gen, .lfm3a_gen => neI(m.gen_code_out_embd_w, 0),
        .pockettts_spkenc => neI(m.spk_proj_w, 1),
        .pockettts_gen => neI(m.gen_input_lin_w, 1),
        // talker conditioning rows, the speaker projection carries their width
        .chatterbox_spkenc => neI(m.cbx_tensors.get("a.cenc.spkr_enc.weight") orelse
            mod.impl.abortUncaught("std::out_of_range", "unordered_map::at: key not found"), 1),
        // talker conditioning rows, the speech table carries their width
        .chatterbox => neI(m.cbx_tensors.get("a.gen.code.out_embd.weight") orelse
            mod.impl.abortUncaught("std::out_of_range", "map::at:  key not found"), 0),
        .parakeet => neI(m.mm_1_w, 1),
        else => mod.impl.abort("mtmd-model.cpp", 4353, "Unknown projector type"),
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
