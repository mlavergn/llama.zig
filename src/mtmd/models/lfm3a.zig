//! The LFM3-Audio encoder: a conv downsampler (8x in time), a causal LFM2
//! stack, then an MLP projector. The graph always streams: a zero state
//! encodes a clip from its start.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/lfm3a.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration names what it ports and the line it began at.
//!
//! `state_in` is a `std::map` read only through `at()`, so a hash map stands
//! in; `state_out` is a vector, iterated in insertion order, and stays a
//! list.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// A named `state_out` entry.
const StateOut = struct { name: [:0]const u8, t: mtmd.graph.T };

/// Ports `list_lfm3a_state_slots` (lfm3a.cpp:5 @b1b770d90): the streaming
/// encoder state: base position, conv left context, short-conv taps and
/// attention window. A zero state gives the same result as encoding the whole
/// clip. The caller frees the slice with `module.freeSlots`.
pub fn listStateSlots(hparams: *const mtmd.model.Hparams, model: *const mtmd.model.Model) mtmd.clip.Error![]mtmd.models.dispatch.StateSlot {
    var slots: std.ArrayList(mtmd.models.dispatch.StateSlot) = .empty;
    errdefer {
        for (slots.items) |s| std.heap.c_allocator.free(s.name);
        slots.deinit(std.heap.c_allocator);
    }

    const n_mel: i64 = hparams.n_mel_bins;
    const n_ch = mtmd.graph.ne(model.pre_encode_conv_X_w[0], 3);
    const prefix: i64 = hparams.attn_window_size - 1;

    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "pos", 0), .ne0 = 1, .ne1 = 1 });
    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "ds0", 0), .ne0 = 1, .ne1 = n_mel });
    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "ds1", 0), .ne0 = 1, .ne1 = @divTrunc(n_mel, 2) * n_ch });
    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "ds2", 0), .ne0 = 1, .ne1 = @divTrunc(n_mel, 4) * n_ch });
    for (model.layers.items, 0..) |*layer, il| {
        if (layer.conv_dw_w != null) {
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "conv_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.conv_dw_w, 0) - 1, .ne1 = hparams.n_embd });
        } else {
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "k_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.k_w, 1), .ne1 = prefix });
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "v_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.v_w, 1), .ne1 = prefix });
        }
    }

    return slots.toOwnedSlice(std.heap.c_allocator);
}

/// The graph's state tensors, as the C++'s two lambdas capture them.
const Builder = struct {
    ctx0: ?*llama.impl.llama_c.ggml_context,
    arena: std.mem.Allocator,
    state_in: std.StringHashMapUnmanaged(mtmd.graph.T) = .empty,
    state_out: std.ArrayList(StateOut) = .empty,

    /// `std::map::at` on a missing key: an uncaught `std::out_of_range`.
    fn at(self: *const Builder, name: []const u8) mtmd.graph.T {
        return self.state_in.get(name) orelse mtmd.impl.abort("lfm3a.cpp", 0, "map::at: key not found");
    }

    fn out(self: *Builder, name: [:0]const u8, t: mtmd.graph.T) mtmd.clip.Error!void {
        try self.state_out.append(self.arena, .{ .name = name, .t = t });
    }

    /// The `conv_down` lambda (lfm3a.cpp:46 @b1b770d90): a stride-2 conv over
    /// [time, freq, ch]; the left column comes from the state and the right
    /// edge is zero-padded.
    fn convDown(self: *Builder, x_in: mtmd.graph.T, name: [:0]const u8, w: mtmd.graph.T, b: mtmd.graph.T, depthwise: bool) mtmd.clip.Error!mtmd.graph.T {
        const ctx0 = self.ctx0;
        const left = llama.impl.llama_c.ggml_reshape_4d(ctx0, self.at(name), 1, mtmd.graph.ne(x_in, 1), mtmd.graph.ne(x_in, 2), 1);
        var x = llama.impl.llama_c.ggml_concat(ctx0, left, x_in, 0);
        const last = llama.impl.llama_c.ggml_view_4d(ctx0, x, 1, mtmd.graph.ne(x, 1), mtmd.graph.ne(x, 2), 1, mtmd.graph.nb(x, 1), mtmd.graph.nb(x, 2), mtmd.graph.nb(x, 3), @as(usize, @intCast(mtmd.graph.ne(x, 0) - 1)) * mtmd.graph.nb(x, 0));
        try self.out(name, llama.impl.llama_c.ggml_reshape_2d(ctx0, llama.impl.llama_c.ggml_cont(ctx0, last), 1, mtmd.graph.ne(x, 1) * mtmd.graph.ne(x, 2)));
        x = llama.impl.llama_c.ggml_pad_ext(ctx0, x, 0, 1, 0, 0, 0, 0, 0, 0);
        x = if (depthwise) llama.impl.llama_c.ggml_conv_2d_dw_direct(ctx0, w, x, 2, 2, 0, 1, 1, 1) else llama.impl.llama_c.ggml_conv_2d(ctx0, w, x, 2, 2, 0, 1, 1, 1);
        return llama.impl.llama_c.ggml_add(ctx0, x, b);
    }
};

/// Ports `build` (lfm3a.cpp:30 @b1b770d90), a member of `mtmd_graph_lfm3a`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*llama.impl.llama_c.ggml_cgraph {
    _ = params;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const n_embd = g.n_embd;
    const d_head = g.d_head;
    const n_head = g.n_head;
    const n_head_kv = g.n_head_kv;
    const eps = g.eps;

    const norm_t: mtmd.model.NormType = .rms;
    const rope_theta = hparams.rope_theta;
    const prefix: i64 = hparams.attn_window_size - 1;

    var arena_state: std.heap.ArenaAllocator = .init(std.heap.c_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // the graph always streams, a zero state encodes a clip from its start
    var b: Builder = .{ .ctx0 = ctx0, .arena = arena };
    {
        const slots = try listStateSlots(hparams, model);
        defer mtmd.models.dispatch.freeSlots(slots);
        for (slots) |slot| {
            const t = llama.impl.llama_c.ggml_new_tensor_2d(ctx0, llama.impl.llama_c.GGML_TYPE_F32, slot.ne0, slot.ne1);
            const name = try mtmd.impl.stringFormat(arena, "state_in_%s", .{slot.name.ptr});
            _ = llama.impl.llama_c.ggml_set_name(t, name.ptr);
            llama.impl.llama_c.ggml_set_input(t);
            try b.state_in.put(arena, try arena.dupe(u8, slot.name), t);
        }
    }

    // inp: [n_frames, n_mel, 1]
    var cur = g.buildInpRaw(1);

    // downsampler, same layout as the lfm2a pre-encoder but with SiLU
    cur = try b.convDown(cur, "ds0", model.pre_encode_conv_X_w[0], model.pre_encode_conv_X_b[0], false);
    cur = llama.impl.llama_c.ggml_silu_inplace(ctx0, cur);

    cur = try b.convDown(cur, "ds1", model.pre_encode_conv_X_w[2], model.pre_encode_conv_X_b[2], true);
    cur = llama.impl.llama_c.ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
    cur = llama.impl.llama_c.ggml_silu_inplace(ctx0, cur);

    cur = try b.convDown(cur, "ds2", model.pre_encode_conv_X_w[5], model.pre_encode_conv_X_b[5], true);
    cur = llama.impl.llama_c.ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
    cur = llama.impl.llama_c.ggml_silu_inplace(ctx0, cur);
    g.cb(cur, "downsample.conv", -1);

    // [n_pos, n_mel/8, n_ch] -> [n_ch * n_mel/8, n_pos]
    const n_pos: c_int = @intCast(mtmd.graph.ne(cur, 0));
    cur = llama.impl.llama_c.ggml_reshape_2d(ctx0, cur, n_pos, mtmd.graph.ne(cur, 1) * mtmd.graph.ne(cur, 2));
    cur = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, cur));
    cur = g.buildMm(model.pre_encode_out_w, cur);
    cur = llama.impl.llama_c.ggml_add(ctx0, cur, model.pre_encode_out_b);
    g.cb(cur, "downsample.out", -1);

    const positions = llama.impl.llama_c.ggml_new_tensor_1d(ctx0, llama.impl.llama_c.GGML_TYPE_I32, n_pos);
    _ = llama.impl.llama_c.ggml_set_name(positions, "positions");
    llama.impl.llama_c.ggml_set_input(positions);
    try b.out("pos", llama.impl.llama_c.ggml_scale_bias(ctx0, b.at("pos"), 1.0, @floatFromInt(n_pos)));

    // causal mask with a sliding window, shared by all attention layers
    const kq_mask = g.buildInpAttnMask(prefix + n_pos, n_pos, 0);

    var inp_l = cur;

    var il: c_int = 0;
    while (il < g.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];

        cur = g.buildNorm(inp_l, layer.ln_1_w, null, norm_t, eps, il);
        g.cb(cur, "operator_norm", il);

        if (layer.conv_dw_w != null) {
            // short conv: in_proj -> b * x -> causal depthwise conv -> c * conv -> out_proj
            const bcx = g.buildMm(layer.conv_pw1_w, cur);
            g.cb(bcx, "conv_in_proj", il);

            const bb = llama.impl.llama_c.ggml_view_2d(ctx0, bcx, n_embd, n_pos, mtmd.graph.nb(bcx, 1), 0);
            const cc = llama.impl.llama_c.ggml_view_2d(ctx0, bcx, n_embd, n_pos, mtmd.graph.nb(bcx, 1), llama.impl.llama_c.ggml_row_size(bcx.?.type, n_embd));
            const x = llama.impl.llama_c.ggml_view_2d(ctx0, bcx, n_embd, n_pos, mtmd.graph.nb(bcx, 1), llama.impl.llama_c.ggml_row_size(bcx.?.type, 2 * @as(i64, n_embd)));

            var bx = llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, llama.impl.llama_c.ggml_mul(ctx0, bb, x)));
            const d_conv = mtmd.graph.ne(layer.conv_dw_w, 0) - 1;
            const name = try mtmd.impl.stringFormat(arena, "conv_%d", .{il});
            bx = llama.impl.llama_c.ggml_concat(ctx0, b.at(name), bx, 0);
            try b.out(name, llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, bx, d_conv, n_embd, mtmd.graph.nb(bx, 1), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(bx, 0))));
            bx = llama.impl.llama_c.ggml_reshape_3d(ctx0, bx, mtmd.graph.ne(bx, 0), n_embd, 1);

            cur = llama.impl.llama_c.ggml_ssm_conv(ctx0, bx, layer.conv_dw_w);
            cur = llama.impl.llama_c.ggml_reshape_2d(ctx0, cur, n_embd, n_pos);
            g.cb(cur, "conv_conv", il);

            cur = llama.impl.llama_c.ggml_mul(ctx0, cc, cur);
            cur = g.buildMm(layer.conv_pw2_w, cur);
            g.cb(cur, "conv_out_proj", il);
        } else {
            var q_cur = g.buildMm(layer.q_w, cur);
            var k_cur = g.buildMm(layer.k_w, cur);
            var v_cur = g.buildMm(layer.v_w, cur);

            q_cur = llama.impl.llama_c.ggml_reshape_3d(ctx0, q_cur, d_head, n_head, n_pos);
            k_cur = llama.impl.llama_c.ggml_reshape_3d(ctx0, k_cur, d_head, n_head_kv, n_pos);
            v_cur = llama.impl.llama_c.ggml_reshape_3d(ctx0, v_cur, d_head, n_head_kv, n_pos);

            q_cur = g.buildNorm(q_cur, layer.q_norm, null, norm_t, eps, il);
            k_cur = g.buildNorm(k_cur, layer.k_norm, null, norm_t, eps, il);

            q_cur = llama.impl.llama_c.ggml_rope_ext(ctx0, q_cur, positions, null, d_head, llama.impl.llama_c.GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);
            k_cur = llama.impl.llama_c.ggml_rope_ext(ctx0, k_cur, positions, null, d_head, llama.impl.llama_c.GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0, 0.0, 1.0, 0.0, 0.0);

            // the cached window goes in front, this chunk's tail is kept for the next call
            const k_name = try mtmd.impl.stringFormat(arena, "k_%d", .{il});
            const v_name = try mtmd.impl.stringFormat(arena, "v_%d", .{il});
            const n_embd_kv: i64 = @as(i64, d_head) * n_head_kv;
            const k_all = llama.impl.llama_c.ggml_concat(ctx0, b.at(k_name), llama.impl.llama_c.ggml_reshape_2d(ctx0, k_cur, n_embd_kv, n_pos), 1);
            const v_all = llama.impl.llama_c.ggml_concat(ctx0, b.at(v_name), llama.impl.llama_c.ggml_reshape_2d(ctx0, v_cur, n_embd_kv, n_pos), 1);
            try b.out(k_name, llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, k_all, n_embd_kv, prefix, mtmd.graph.nb(k_all, 1), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(k_all, 1))));
            try b.out(v_name, llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_view_2d(ctx0, v_all, n_embd_kv, prefix, mtmd.graph.nb(v_all, 1), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(v_all, 1))));
            k_cur = llama.impl.llama_c.ggml_reshape_3d(ctx0, k_all, d_head, n_head_kv, prefix + n_pos);
            v_cur = llama.impl.llama_c.ggml_reshape_3d(ctx0, v_all, d_head, n_head_kv, prefix + n_pos);

            cur = g.buildAttn(layer.o_w, null, q_cur, k_cur, v_cur, kq_mask, g.kq_scale, il, null);
            g.cb(cur, "attn_out", il);
        }

        cur = llama.impl.llama_c.ggml_add(ctx0, cur, inp_l);
        inp_l = cur;

        cur = g.buildNorm(cur, layer.ln_2_w, null, norm_t, eps, il);
        g.cb(cur, "ffn_norm", il);

        cur = g.buildFfn(cur, layer.ff_up_w, null, layer.ff_gate_w, null, layer.ff_down_w, null, .silu, il);
        g.cb(cur, "ffn_out", il);

        cur = llama.impl.llama_c.ggml_add(ctx0, inp_l, cur);
        inp_l = cur;
    }

    cur = g.buildNorm(inp_l, model.post_ln_w, null, norm_t, eps, -1);
    g.cb(cur, "embedding_norm", -1);

    // projector: LayerNorm -> Linear -> GELU -> Linear
    cur = g.buildNorm(cur, model.mm_0_w, model.mm_0_b, .normal, 1e-5, -1);
    cur = g.buildFfn(cur, model.mm_1_w, model.mm_1_b, null, null, model.mm_3_w, model.mm_3_b, .gelu_erf, -1);
    g.cb(cur, "projected", -1);

    // the state outputs go first so that the embeddings stay the last node of the graph
    for (b.state_out.items) |s| {
        const name = try mtmd.impl.stringFormat(arena, "state_out_%s", .{s.name.ptr});
        _ = llama.impl.llama_c.ggml_set_name(s.t, name.ptr);
        llama.impl.llama_c.ggml_set_output(s.t);
        llama.impl.llama_c.ggml_build_forward_expand(g.gf, s.t);
    }

    llama.impl.llama_c.ggml_build_forward_expand(g.gf, cur);

    return g.gf;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
