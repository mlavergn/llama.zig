//! The Gemma 4 audio Conformer encoder: dual half-step FFNs, chunked local
//! self-attention with sinusoidal relative positions, a depthwise light
//! conv, and an output projection.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/models/gemma4a.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration names what it ports and the line it began at.
//!
//! Since v1.0.0 the encoder streams (20c2e9665): the subsampling convs'
//! left column, each attention layer's last `max_past` K/V rows and each
//! light conv's taps come from named state inputs and go out as state
//! outputs; a zero state is a cold start. `state_in` is a `std::map` read
//! only through `at()`, so a hash map stands in; `state_out` is a vector,
//! iterated in insertion order, and stays a list.
//!
//! Upstream overrides the virtual `build_mm`; here `build` installs
//! `buildMm` as the graph's `mm_hook`, so `graph.Graph.buildFfn` and the
//! rest reach it the way the C++ base class reaches the override.

const std = @import("std");
const mtmd = @import("../module.zig");
const stdlib = @import("stdlib");

/// `chunk_size` (mtmd/models/models.h:278 @b1b770d90), a static member of
/// `mtmd_graph_gemma4a`: the model's `attention_chunk_size`.
pub const chunk_size: i64 = 12;
/// `max_past` (mtmd/models/models.h:279 @b1b770d90), a static member of
/// `mtmd_graph_gemma4a`: the model's `attention_context_left - 1`.
pub const max_past: i64 = 12;

/// Ports `list_gemma4a_state_slots` (gemma4a.cpp:13 @b1b770d90): the state of
/// the streaming encoder, a zero state is a cold start. The caller frees the
/// slice with `module.freeSlots`.
pub fn listStateSlots(hparams: *const mtmd.model.Hparams, model: *const mtmd.model.Model) mtmd.clip.Error![]mtmd.models.StateSlot {
    var slots: std.ArrayList(mtmd.models.StateSlot) = .empty;
    errdefer {
        for (slots.items) |sl| std.heap.c_allocator.free(sl.name);
        slots.deinit(std.heap.c_allocator);
    }

    const n_mel: i64 = hparams.n_mel_bins;
    const n_ch = mtmd.graph.ne(model.sscp_conv_w[0], 3);

    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "pos", 0), .ne0 = 1, .ne1 = 1 });
    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "sscp_0", 0), .ne0 = n_mel, .ne1 = 1 });
    try slots.append(std.heap.c_allocator, .{ .name = try std.heap.c_allocator.dupeSentinel(u8, "sscp_1", 0), .ne0 = @divTrunc(n_mel - 1, 2) + 1, .ne1 = n_ch });
    for (model.layers.items, 0..) |*layer, il| {
        if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "k_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.k_w, 1), .ne1 = max_past });
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "v_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.v_w, 1), .ne1 = max_past });
        }
        if (layer.norm_conv_w != null and layer.conv_pw1_w != null and layer.conv_dw_w != null and layer.conv_pw2_w != null) {
            try slots.append(std.heap.c_allocator, .{ .name = try mtmd.impl.stringFormat(std.heap.c_allocator, "conv_%zu", .{il}), .ne0 = mtmd.graph.ne(layer.conv_dw_w, 0) - 1, .ne1 = mtmd.graph.ne(layer.conv_dw_w, 1) });
        }
    }

    return slots.toOwnedSlice(std.heap.c_allocator);
}

/// A named `state_out` entry.
const StateOut = struct { name: [:0]const u8, t: mtmd.graph.T };

/// The graph's state tensors, as the C++'s `build` and its `extract_blocks`
/// lambda capture them.
const State = struct {
    arena: std.mem.Allocator,
    state_in: std.StringHashMapUnmanaged(mtmd.graph.T) = .empty,
    state_out: std.ArrayList(StateOut) = .empty,

    /// `std::map::at` on a missing key: an uncaught `std::out_of_range`.
    fn at(self: *const State, name: []const u8) mtmd.graph.T {
        return self.state_in.get(name) orelse mtmd.impl.abortUncaught("std::out_of_range", "map::at:  key not found");
    }

    fn out(self: *State, name: [:0]const u8, t: mtmd.graph.T) mtmd.clip.Error!void {
        try self.state_out.append(self.arena, .{ .name = name, .t = t });
    }
};

/// The `extract_blocks` lambda (gemma4a.cpp:163 @b1b770d90) of `build`:
/// K/V block context through an overlapping view. The P cached positions go
/// in front, the right is padded to S*B, then the view strides C in the
/// block dimension.
fn extractBlocks(g: *const mtmd.graph.Graph, st: *State, t_in: mtmd.graph.T, name: [:0]const u8, S: i64, B: i64, P: i64, C: i64, n_pos: i64) mtmd.clip.Error!mtmd.graph.T {
    const ctx0 = g.ctx0;
    var t = mtmd.impl.c.ggml_concat(ctx0, mtmd.impl.c.ggml_reshape_3d(ctx0, st.at(name), g.d_head, g.n_head, P), t_in, 2);
    const tail = mtmd.impl.c.ggml_view_3d(ctx0, t, g.d_head, g.n_head, P, mtmd.graph.nb(t, 1), mtmd.graph.nb(t, 2), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(t, 2));
    try st.out(name, mtmd.impl.c.ggml_reshape_2d(ctx0, mtmd.impl.c.ggml_cont(ctx0, tail), @as(i64, g.d_head) * g.n_head, P));
    t = mtmd.impl.c.ggml_pad_ext(ctx0, t, 0, 0, 0, 0, 0, @truncate(S * B - P - n_pos), 0, 0); // [D, H, S*B]
    // Overlapping view: stride for B dim is C positions, not S
    // ne = [D, H, S, B], data_size = D*H*S*B*sizeof = source_nbytes (exact fit)
    // nb1=D*sizeof, nb2=D*H*sizeof, nb3=C*D*H*sizeof (overlap: C < S)
    t = mtmd.impl.c.ggml_view_4d(ctx0, t, g.d_head, g.n_head, S, B, mtmd.graph.nb(t, 1), mtmd.graph.nb(t, 2), @as(usize, @intCast(C)) * mtmd.graph.nb(t, 2), 0);
    t = mtmd.impl.c.ggml_cont(ctx0, t); // materialize overlapping windows
    return t;
}

/// Ports `build` (gemma4a.cpp:36 @b1b770d90), a member of
/// `mtmd_graph_gemma4a`.
pub fn build(g: *mtmd.graph.Graph, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!?*mtmd.impl.c.ggml_cgraph {
    _ = params;
    g.mm_hook = buildMm;
    const ctx0 = g.ctx0;
    const model = g.model;
    const hparams = g.hparams;
    const d_head = g.d_head;
    const n_head = g.n_head;

    const res_weight: f32 = 0.5;
    const norm_eps: f32 = 1e-6;

    var arena_state: std.heap.ArenaAllocator = .init(std.heap.c_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var st: State = .{ .arena = arena };
    {
        const slots = try listStateSlots(hparams, model);
        defer mtmd.models.freeSlots(slots);
        for (slots) |slot| {
            const t = mtmd.impl.c.ggml_new_tensor_2d(ctx0, mtmd.impl.c.GGML_TYPE_F32, slot.ne0, slot.ne1);
            const name = try mtmd.impl.stringFormat(arena, "state_in_%s", .{slot.name.ptr});
            _ = mtmd.impl.c.ggml_set_name(t, name.ptr);
            mtmd.impl.c.ggml_set_input(t);
            try st.state_in.put(arena, try arena.dupe(u8, slot.name), t);
        }
    }

    // 1. Input
    const inp = g.buildInpRaw(1);
    var cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, inp));

    // 2. Subsampling Conv2D (padding=1, matching PyTorch)
    // in time, the left pad comes from the state
    {
        for (0..2) |i| {
            const name = try mtmd.impl.stringFormat(arena, "sscp_%zu", .{i});
            const left = mtmd.impl.c.ggml_reshape_3d(ctx0, st.at(name), mtmd.graph.ne(cur, 0), 1, mtmd.graph.ne(cur, 2));
            cur = mtmd.impl.c.ggml_concat(ctx0, left, cur, 1);
            const last = mtmd.impl.c.ggml_view_3d(ctx0, cur, mtmd.graph.ne(cur, 0), 1, mtmd.graph.ne(cur, 2), mtmd.graph.nb(cur, 1), mtmd.graph.nb(cur, 2), @as(usize, @intCast(mtmd.graph.ne(cur, 1) - 1)) * mtmd.graph.nb(cur, 1));
            try st.out(name, mtmd.impl.c.ggml_reshape_2d(ctx0, mtmd.impl.c.ggml_cont(ctx0, last), mtmd.graph.ne(cur, 0), mtmd.graph.ne(cur, 2)));
            cur = mtmd.impl.c.ggml_pad_ext(ctx0, cur, 0, 0, 0, 1, 0, 0, 0, 0);
            cur = mtmd.impl.c.ggml_conv_2d(ctx0, model.sscp_conv_w[i], cur, 2, 2, 1, 0, 1, 1);
            if (model.sscp_conv_b[i] != null) {
                cur = mtmd.impl.c.ggml_add(ctx0, cur, model.sscp_conv_b[i]);
            }
            // nn.LayerNorm(channels): permute ch to ne[0], normalize, permute back
            if (model.sscp_norm_w[i] != null) {
                cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, cur, 1, 2, 0, 3));
                cur = mtmd.impl.c.ggml_norm(ctx0, cur, norm_eps);
                cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.sscp_norm_w[i]);
                cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, cur, 2, 0, 1, 3));
            }
            cur = mtmd.impl.c.ggml_relu(ctx0, cur);
        }
        // Flatten [freq, time, ch, 1] -> [ch*freq, time]
        cur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = mtmd.impl.c.ggml_reshape_2d(ctx0, cur, mtmd.graph.ne(cur, 0) * mtmd.graph.ne(cur, 1), mtmd.graph.ne(cur, 2));
        if (model.sscp_inp_proj_w != null) {
            cur = g.buildMm(model.sscp_inp_proj_w, cur);
            if (model.sscp_inp_proj_b != null) {
                cur = mtmd.impl.c.ggml_add(ctx0, cur, model.sscp_inp_proj_b);
            }
        }
    }

    const n_pos = mtmd.graph.ne(cur, 1);
    try st.out("pos", mtmd.impl.c.ggml_scale_bias(ctx0, st.at("pos"), 1.0, @floatFromInt(n_pos)));

    // Chunked local attention parameters
    const C: i64 = chunk_size;
    const P: i64 = max_past;
    const S: i64 = C + P; // context_size = 24
    const R: i64 = P + 1; // RPE positions = 13
    const B: i64 = @divTrunc(n_pos + C - 1, C); // num_blocks
    const Np: i64 = B * C; // padded sequence length
    const pad_seq = Np - n_pos;

    // Input tensors: blocked RPE and blocked attention mask
    const pos_emb = mtmd.impl.c.ggml_new_tensor_2d(ctx0, mtmd.impl.c.GGML_TYPE_F32, @as(i64, n_head) * d_head, R);
    _ = mtmd.impl.c.ggml_set_name(pos_emb, "pos_emb");
    mtmd.impl.c.ggml_set_input(pos_emb);

    const kq_mask = mtmd.impl.c.ggml_new_tensor_3d(ctx0, mtmd.impl.c.GGML_TYPE_F32, S, C, B);
    _ = mtmd.impl.c.ggml_set_name(kq_mask, "kq_mask");
    mtmd.impl.c.ggml_set_input(kq_mask);

    // 3. Conformer Blocks
    var il: c_int = 0;
    while (il < hparams.n_layer) : (il += 1) {
        const layer = &model.layers.items[@intCast(il)];
        var residual = cur;

        // FFN 1 (half-step)
        if (layer.ff_norm_w != null and layer.ff_up_w != null and layer.ff_down_w != null) {
            cur = g.buildNorm(cur, layer.ff_norm_w, null, .rms, norm_eps, il);
            cur = g.buildFfn(cur, layer.ff_up_w, null, null, null, layer.ff_down_w, null, .silu, il);
            if (layer.ff_post_norm_w != null) {
                cur = g.buildNorm(cur, layer.ff_post_norm_w, null, .rms, norm_eps, il);
            }
            residual = mtmd.impl.c.ggml_add(ctx0, residual, mtmd.impl.c.ggml_scale(ctx0, cur, res_weight));
        }

        // Chunked local self-attention with RPE
        if (layer.q_w != null and layer.k_w != null and layer.v_w != null and layer.o_w != null) {
            const q_scale = (1.0 / stdlib.libm.sqrtfInt(d_head)) / stdlib.libm.logf(2.0);
            const k_scale = stdlib.libm.logf(1.0 + stdlib.libm.expf(1.0)) / stdlib.libm.logf(2.0);
            const softcap: f32 = 50.0;

            const attn_norm_w = if (layer.attn_pre_norm_w != null) layer.attn_pre_norm_w else layer.ln_1_w;
            cur = if (attn_norm_w != null) g.buildNorm(residual, attn_norm_w, null, .rms, norm_eps, il) else residual;

            var Qcur = g.buildMm(layer.q_w, cur);
            var Kcur = g.buildMm(layer.k_w, cur);
            var Vcur = g.buildMm(layer.v_w, cur);

            // [n_embd, n_pos] -> [D, H, N]
            Qcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = mtmd.impl.c.ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            // Q/K scaling
            Qcur = mtmd.impl.c.ggml_scale(ctx0, Qcur, q_scale);
            if (layer.per_dim_scale_w != null) {
                Qcur = mtmd.impl.c.ggml_mul(ctx0, Qcur, mtmd.impl.c.ggml_reshape_3d(ctx0, layer.per_dim_scale_w, d_head, 1, 1));
            }
            Kcur = mtmd.impl.c.ggml_scale(ctx0, Kcur, k_scale);
            if (layer.per_dim_k_scale_w != null) {
                Kcur = mtmd.impl.c.ggml_mul(ctx0, Kcur, mtmd.impl.c.ggml_reshape_3d(ctx0, layer.per_dim_k_scale_w, d_head, 1, 1));
            }

            // Q blocking: [D, H, N] -> pad to Np -> reshape [D, H, C, B]
            // ggml permute: ne[ax_i] = src->ne[i], so (0,3,1,2) sends H->3, C->1, B->2
            Qcur = mtmd.impl.c.ggml_pad(ctx0, Qcur, 0, 0, @intCast(pad_seq), 0); // [D, H, Np]
            Qcur = mtmd.impl.c.ggml_reshape_4d(ctx0, Qcur, d_head, n_head, C, B); // [D, H, C, B]
            Qcur = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, Qcur, 0, 3, 1, 2)); // [D, C, B, H]

            var Kblk = try extractBlocks(g, &st, Kcur, try mtmd.impl.stringFormat(arena, "k_%d", .{il}), S, B, P, C, n_pos);
            // [D, H, S, B] -> [D, S, B, H] via permute(0,3,1,2)
            Kblk = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, Kblk, 0, 3, 1, 2));

            var Vblk = try extractBlocks(g, &st, Vcur, try mtmd.impl.stringFormat(arena, "v_%d", .{il}), S, B, P, C, n_pos);
            // [D, H, S, B] -> [S, D, B, H] via permute(1,3,0,2)
            Vblk = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, Vblk, 1, 3, 0, 2));

            // Content attention: Q @ K^T
            // Kblk=[D,S,B,H], Qcur=[D,C,B,H] -> mul_mat contracts on D -> [S,C,B,H]
            var matrix_ac = mtmd.impl.c.ggml_mul_mat(ctx0, Kblk, Qcur);

            // Relative position attention
            if (layer.attn_k_rel_w != null) {
                // RPE: [n_embd, R] -> project -> [D, H, R] -> [D, R, H]
                var p = mtmd.impl.c.ggml_mul_mat(ctx0, layer.attn_k_rel_w, pos_emb);
                p = mtmd.impl.c.ggml_reshape_3d(ctx0, p, d_head, n_head, R);
                p = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, p, 0, 2, 1, 3)); // [D, R, H]

                // Q_flat @ RPE^T: [D, C*B, H] @ [D, R, H] -> [R, C*B, H]
                const Q_flat = mtmd.impl.c.ggml_reshape_3d(ctx0, Qcur, d_head, C * B, n_head);
                var matrix_bd = mtmd.impl.c.ggml_mul_mat(ctx0, p, Q_flat); // [R, C*B, H]
                matrix_bd = mtmd.impl.c.ggml_reshape_4d(ctx0, matrix_bd, R, C, B, n_head); // [R, C, B, H]

                // Blocked relative shift (appendix B of Transformer-XL)
                {
                    matrix_bd = mtmd.impl.c.ggml_pad(ctx0, matrix_bd, @intCast(S + 1 - R), 0, 0, 0); // [S+1, C, B, H]
                    matrix_bd = mtmd.impl.c.ggml_reshape_3d(ctx0, matrix_bd, (S + 1) * C, B, n_head);
                    matrix_bd = mtmd.impl.c.ggml_view_3d(ctx0, matrix_bd, C * S, B, n_head, mtmd.graph.nb(matrix_bd, 1), mtmd.graph.nb(matrix_bd, 2), 0);
                    matrix_bd = mtmd.impl.c.ggml_cont(ctx0, matrix_bd); // [C*S, B, H]
                    matrix_bd = mtmd.impl.c.ggml_reshape_4d(ctx0, matrix_bd, S, C, B, n_head); // [S, C, B, H]
                }

                matrix_ac = mtmd.impl.c.ggml_add(ctx0, matrix_ac, matrix_bd);
            }

            var scores = matrix_ac; // [S, C, B, H]

            // Softcap
            scores = mtmd.impl.c.ggml_scale(ctx0, scores, 1.0 / softcap);
            scores = mtmd.impl.c.ggml_tanh(ctx0, scores);
            scores = mtmd.impl.c.ggml_scale(ctx0, scores, softcap);

            // Blocked attention mask: [S, C, B] broadcasts over H
            scores = mtmd.impl.c.ggml_add(ctx0, scores, kq_mask);

            const attn = mtmd.impl.c.ggml_soft_max(ctx0, scores);

            // attn @ V: [S,C,B,H] @ [S,D,B,H] -> [D,C,B,H]
            var x = mtmd.impl.c.ggml_mul_mat(ctx0, Vblk, attn);

            // [D,C,B,H] -> [D,H,C,B] via permute(0,2,3,1) -> flatten -> trim
            x = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_permute(ctx0, x, 0, 2, 3, 1));
            x = mtmd.impl.c.ggml_cont_2d(ctx0, x, @as(i64, d_head) * n_head, C * B);
            if (pad_seq > 0) {
                x = mtmd.impl.c.ggml_view_2d(ctx0, x, @as(i64, d_head) * n_head, n_pos, mtmd.graph.nb(x, 1), 0);
                x = mtmd.impl.c.ggml_cont(ctx0, x);
            }

            x = g.buildMm(layer.o_w, x);
            if (layer.o_b != null) {
                x = mtmd.impl.c.ggml_add(ctx0, x, layer.o_b);
            }

            if (layer.attn_post_norm_w != null) {
                x = g.buildNorm(x, layer.attn_post_norm_w, null, .rms, norm_eps, il);
            }
            residual = mtmd.impl.c.ggml_add(ctx0, residual, x);
        }

        // Convolution Module
        if (layer.norm_conv_w != null and layer.conv_pw1_w != null and layer.conv_dw_w != null and layer.conv_pw2_w != null) {
            cur = g.buildNorm(residual, layer.norm_conv_w, null, .rms, norm_eps, il);
            var x = g.buildMm(layer.conv_pw1_w, cur);

            // GLU
            {
                const d = @divTrunc(mtmd.graph.ne(x, 0), 2);
                const gate = mtmd.impl.c.ggml_sigmoid(ctx0, mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_view_2d(ctx0, x, d, mtmd.graph.ne(x, 1), mtmd.graph.nb(x, 1), @as(usize, @intCast(d)) * mtmd.graph.nb(x, 0))));
                x = mtmd.impl.c.ggml_mul(ctx0, mtmd.impl.c.ggml_view_2d(ctx0, x, d, mtmd.graph.ne(x, 1), mtmd.graph.nb(x, 1), 0), gate);
                x = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, x));
            }

            // Causal depthwise Conv1D via ggml_ssm_conv, the left context comes from the state
            {
                const d_conv = mtmd.graph.ne(layer.conv_dw_w, 0) - 1;
                const name = try mtmd.impl.stringFormat(arena, "conv_%d", .{il});
                x = mtmd.impl.c.ggml_concat(ctx0, st.at(name), x, 0);
                try st.out(name, mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_view_2d(ctx0, x, d_conv, mtmd.graph.ne(x, 1), mtmd.graph.nb(x, 1), @as(usize, @intCast(n_pos)) * mtmd.graph.nb(x, 0))));
            }
            x = mtmd.impl.c.ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            if (layer.conv_dw_b != null) {
                x = mtmd.impl.c.ggml_add(ctx0, x, layer.conv_dw_b);
            }

            if (layer.conv_norm_w != null) {
                x = mtmd.impl.c.ggml_rms_norm(ctx0, x, norm_eps);
                x = mtmd.impl.c.ggml_mul(ctx0, x, layer.conv_norm_w);
            }
            x = mtmd.impl.c.ggml_silu(ctx0, x);
            x = g.buildMm(layer.conv_pw2_w, x);
            residual = mtmd.impl.c.ggml_add(ctx0, residual, x);
        }

        // FFN 2 (half-step)
        if (layer.ff_norm_1_w != null and layer.ff_up_1_w != null and layer.ff_down_1_w != null) {
            cur = g.buildNorm(residual, layer.ff_norm_1_w, null, .rms, norm_eps, il);
            cur = g.buildFfn(cur, layer.ff_up_1_w, null, null, null, layer.ff_down_1_w, null, .silu, il);
            if (layer.ff_post_norm_1_w != null) {
                cur = g.buildNorm(cur, layer.ff_post_norm_1_w, null, .rms, norm_eps, il);
            }
            residual = mtmd.impl.c.ggml_add(ctx0, residual, mtmd.impl.c.ggml_scale(ctx0, cur, res_weight));
        }

        // Layer output norm
        cur = if (layer.ln_2_w != null) g.buildNorm(residual, layer.ln_2_w, null, .rms, norm_eps, il) else residual;
    }

    // 4. Output Projection
    if (model.audio_out_proj_w != null) {
        cur = g.buildMm(model.audio_out_proj_w, cur);
        if (model.audio_out_proj_b != null) {
            cur = mtmd.impl.c.ggml_add(ctx0, cur, model.audio_out_proj_b);
        }
    }

    // 5. Audio Multimodal Embedder
    cur = mtmd.impl.c.ggml_rms_norm(ctx0, cur, norm_eps);
    if (model.mm_soft_emb_norm_w != null) {
        cur = mtmd.impl.c.ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
    }
    if (model.mm_input_proj_w != null) {
        cur = g.buildMm(model.mm_input_proj_w, cur);
    }

    // the embeddings must stay the last node
    for (st.state_out.items) |so| {
        const name = try mtmd.impl.stringFormat(arena, "state_out_%s", .{so.name.ptr});
        _ = mtmd.impl.c.ggml_set_name(so.t, name.ptr);
        mtmd.impl.c.ggml_set_output(so.t);
        mtmd.impl.c.ggml_build_forward_expand(g.gf, so.t);
    }

    mtmd.impl.c.ggml_build_forward_expand(g.gf, cur);
    return g.gf;
}

/// Ports `build_mm` (gemma4a.cpp:330 @b1b770d90), a member of
/// `mtmd_graph_gemma4a`: a Gemma4ClippableLinear clamps its input and
/// output to the bounds the model file records for that weight.
///
/// Parameters:
///   - `g`: the graph builder.
///   - `w`: the weight; its name keys `clamp_info_map`.
///   - `x`: the input.
///
/// Return: `w @ x`, clamped where the model says so.
pub fn buildMm(g: *const mtmd.graph.Graph, w: mtmd.graph.T, x: mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const ci = g.model.clamp_info_map.get(std.mem.sliceTo(&w.?.name, 0)) orelse return mtmd.impl.c.ggml_mul_mat(ctx0, w, x);
    const clamped = mtmd.impl.c.ggml_clamp(ctx0, x, ci.inp_min, ci.inp_max);
    const out = mtmd.impl.c.ggml_mul_mat(ctx0, w, clamped);
    return mtmd.impl.c.ggml_clamp(ctx0, out, ci.out_min, ci.out_max);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
