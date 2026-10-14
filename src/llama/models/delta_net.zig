//! The gated delta net: linear attention with a per-head state, shared by
//! the architectures built on it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/delta-net-base.cpp` at v0.3.0
//! (`c1d0e7a00`): `get_slice_2d` and every member of
//! `llm_build_delta_net_base`.
//!
//! The C++ makes these members of a subclass of `llm_graph_context`; here
//! they are functions over the builder.

const std = @import("std");
const impl = @import("../impl.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const Builder = graph.Builder;

/// The output and the new state.
pub const OutState = struct { out: *Tensor, state: *Tensor };

/// Ports `get_slice_2d` (delta-net-base.cpp:9 @c1d0e7a00): one slice of
/// the third dimension, `[x, y, c, b]` to `[x, y, 1, b]`.
fn getSlice2d(ctx: *c.ggml_context, t: *Tensor, ch: i64) *Tensor {
    return c.ggml_view_4d(ctx, t, t.ne[0], t.ne[1], 1, t.ne[3], t.nb[1], t.nb[2], t.nb[3], t.nb[2] * @as(usize, @intCast(ch)));
}

fn checkShapes(q: *const Tensor, k: *const Tensor, v: *const Tensor, g: *const Tensor, b: *const Tensor, s: *const Tensor, comptime line: c_int) void {
    const S_k = q.ne[0];
    const H_k = q.ne[1];
    const n_tokens = q.ne[2];
    const n_seqs = q.ne[3];
    const S_v = v.ne[0];
    const H_v = v.ne[1];
    impl.assert(S_k == S_v, "delta-net-base.cpp", line, "S_k == S_v");
    impl.assert(@rem(H_v, H_k) == 0, "delta-net-base.cpp", line + 1, "H_v % H_k == 0");
    impl.assert(q.ne[0] == S_k and q.ne[1] == H_k and q.ne[2] == n_tokens and q.ne[3] == n_seqs, "delta-net-base.cpp", line + 3, "q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs");
    impl.assert(k.ne[0] == S_k and k.ne[1] == H_k and k.ne[2] == n_tokens and k.ne[3] == n_seqs, "delta-net-base.cpp", line + 4, "k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs");
    impl.assert(v.ne[0] == S_v and v.ne[1] == H_v and v.ne[2] == n_tokens and v.ne[3] == n_seqs, "delta-net-base.cpp", line + 5, "v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs");
    impl.assert(g.ne[0] == 1 or g.ne[0] == S_v, "delta-net-base.cpp", line + 7, "g->ne[0] == 1   || g->ne[0] == S_v");
    impl.assert(g.ne[1] == H_v and g.ne[2] == n_tokens and g.ne[3] == n_seqs, "delta-net-base.cpp", line + 8, "g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs");
    impl.assert(b.ne[0] == 1 and b.ne[1] == H_v and b.ne[2] == n_tokens and b.ne[3] == n_seqs, "delta-net-base.cpp", line + 9, "b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs");
    impl.assert(s.ne[0] == S_v and s.ne[1] == S_v and s.ne[2] == H_v and s.ne[3] == n_seqs, "delta-net-base.cpp", line + 10, "s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs");
}

/// Ports `llm_build_delta_net_base`'s `build_delta_net_chunking`
/// (delta-net-base.cpp:16 @c1d0e7a00): many tokens at once, in chunks.
pub fn buildDeltaNetChunking(self: *const Builder, q_: *Tensor, k_: *Tensor, v_: *Tensor, g_: *Tensor, b_: *Tensor, s_: *Tensor, il: c_int) OutState {
    const ctx = self.ctx0;
    var q = q_;
    var k = k_;
    var v = v_;
    var g = g_;
    var b = b_;
    var s = s_;

    const S_k = q.ne[0];
    const H_k = q.ne[1];
    const n_tokens = q.ne[2];
    const n_seqs = q.ne[3];
    const S_v = v.ne[0];
    const H_v = v.ne[1];
    const kda = g.ne[0] == S_k and g.ne[1] == H_k;

    checkShapes(q, k, v, g, b, s, 33);

    const scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(S_k));
    q = c.ggml_scale(ctx, q, scale);

    self.cb(q, "q_in", il);
    self.cb(k, "k_in", il);
    self.cb(v, "v_in", il);
    self.cb(b, "b_in", il);
    self.cb(g, "g_in", il);

    q = c.ggml_permute(ctx, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = c.ggml_permute(ctx, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = c.ggml_permute(ctx, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = c.ggml_permute(ctx, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = c.ggml_permute(ctx, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const CS: i64 = if (kda) 16 else 64; // the chunk size
    const pad: i64 = @rem(CS - @rem(n_tokens, CS), CS);
    const n_chunks: i64 = @divTrunc(n_tokens + pad, CS);

    q = c.ggml_pad(ctx, q, 0, @intCast(pad), 0, 0);
    k = c.ggml_pad(ctx, k, 0, @intCast(pad), 0, 0);
    v = c.ggml_pad(ctx, v, 0, @intCast(pad), 0, 0);
    g = c.ggml_pad(ctx, g, 0, @intCast(pad), 0, 0);
    b = c.ggml_pad(ctx, b, 0, @intCast(pad), 0, 0);

    const v_b0 = c.ggml_mul(ctx, v, b);
    const k_b0 = c.ggml_mul(ctx, k, b);
    self.cb(v_b0, "v_b", il);
    self.cb(k_b0, "k_b", il);

    q = c.ggml_reshape_4d(ctx, q, S_k, CS, n_chunks, H_k * n_seqs);
    k = c.ggml_reshape_4d(ctx, k, S_k, CS, n_chunks, H_k * n_seqs);
    var k_b = c.ggml_reshape_4d(ctx, k_b0, S_k, CS, n_chunks, H_v * n_seqs);
    v = c.ggml_reshape_4d(ctx, v, S_v, CS, n_chunks, H_v * n_seqs);
    const v_b = c.ggml_reshape_4d(ctx, v_b0, S_v, CS, n_chunks, H_v * n_seqs);

    g = c.ggml_reshape_4d(ctx, g, g.ne[0], CS, n_chunks, H_v * n_seqs);
    b = c.ggml_reshape_4d(ctx, b, 1, CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    const g_cs = c.ggml_cumsum(ctx, c.ggml_cont(ctx, c.ggml_transpose(ctx, g)));
    self.cb(g_cs, "g_cs", il);

    var kb: *Tensor = undefined;
    var kq: *Tensor = undefined;
    if (kda) {
        const CHB = n_chunks * H_k * n_seqs;
        const g_cs_i = c.ggml_reshape_4d(ctx, g_cs, CS, 1, S_k, CHB); // [chunk_size, 1, S_k, CHB]
        var g_cs_j = c.ggml_reshape_4d(ctx, g_cs, 1, CS, S_k, CHB); // [1, chunk_size, S_k, CHB]
        g_cs_j = c.ggml_repeat_4d(ctx, g_cs_j, CS, CS, S_k, CHB);

        // [chunk_size, chunk_size, S_k, CHB]
        var decay_mask = c.ggml_sub(ctx, g_cs_j, g_cs_i);
        decay_mask = c.ggml_tri(ctx, decay_mask, c.GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = c.ggml_exp(ctx, decay_mask);
        self.cb(decay_mask, "decay_mask", il);

        // [S_k, BT_j, BT_i, CHB]: the second and third chunk sizes swap.
        decay_mask = c.ggml_cont_4d(ctx, c.ggml_permute(ctx, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        const k_b_i = c.ggml_reshape_4d(ctx, k_b, S_k, CS, 1, CHB);
        const k_j = c.ggml_reshape_4d(ctx, k, S_k, 1, CS, CHB);
        const q_i = c.ggml_reshape_4d(ctx, q, S_k, CS, 1, CHB);

        const decay_k_b_i = c.ggml_mul(ctx, decay_mask, k_b_i);
        const decay_q_i = c.ggml_mul(ctx, decay_mask, q_i);

        // [S, BT, BT, CHB] @ [S, 1, BT, CHB] = [BT, 1, BT, CHB]
        kb = c.ggml_mul_mat(ctx, decay_k_b_i, k_j);
        kq = c.ggml_mul_mat(ctx, decay_q_i, k_j);

        kb = c.ggml_cont(ctx, c.ggml_transpose(ctx, c.ggml_reshape_4d(ctx, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = c.ggml_cont(ctx, c.ggml_transpose(ctx, c.ggml_reshape_4d(ctx, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        const g_cs_i = g_cs;
        var g_cs_j = c.ggml_reshape_4d(ctx, g_cs, 1, CS, n_chunks, H_v * n_seqs);
        g_cs_j = c.ggml_repeat_4d(ctx, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        // [CS, CS, n_chunks, H_v * n_seqs]
        var decay_mask = c.ggml_sub(ctx, g_cs_j, g_cs_i);
        decay_mask = c.ggml_tri(ctx, decay_mask, c.GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = c.ggml_exp(ctx, decay_mask);
        self.cb(decay_mask, "decay_mask", il);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kb = c.ggml_mul_mat(ctx, k, k_b);
        kb = c.ggml_mul(ctx, kb, decay_mask);

        kq = c.ggml_mul_mat(ctx, k, q);
        kq = c.ggml_mul(ctx, kq, decay_mask);
    }

    kq = c.ggml_tri(ctx, kq, c.GGML_TRI_TYPE_LOWER_DIAG);
    self.cb(kq, "kq", il);

    // [CS, CS, n_chunks, H_k * n_seqs]
    var attn = c.ggml_tri(ctx, kb, c.GGML_TRI_TYPE_LOWER);
    self.cb(attn, "attn", il);

    var identity = c.ggml_view_1d(ctx, attn, CS, 0);
    identity = c.ggml_fill(ctx, identity, 1.0);
    identity = c.ggml_diag(ctx, identity);

    const lhs = c.ggml_add(ctx, attn, identity);
    self.cb(lhs, "dnet_add_ch_lhs", il);

    attn = c.ggml_neg(ctx, attn);
    self.cb(attn, "attn_pre_solve", il);

    const lin_solve = c.ggml_solve_tri(ctx, lhs, attn, true, true, false);
    attn = c.ggml_add(ctx, lin_solve, identity);
    self.cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = c.ggml_mul_mat(ctx, c.ggml_cont(ctx, c.ggml_transpose(ctx, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs]; KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    const g_exp = c.ggml_exp(ctx, g_cs);

    k_b = c.ggml_cont(ctx, c.ggml_transpose(ctx, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    const kbg = c.ggml_mul(ctx, k_b, g_exp);
    self.cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    const k_cd = c.ggml_mul_mat(ctx, kbg, attn);
    self.cb(k_cd, "k_cumdecay", il);

    // [1, CS, n_chunks, H_k * n_seqs]; KDA: [S_k, CS, n_chunks, H_k * n_seqs]
    const g_exp_t = c.ggml_cont(ctx, c.ggml_transpose(ctx, g_exp));
    const q_g_exp = c.ggml_mul(ctx, q, g_exp_t);

    // The last element of g_cumsum along CS: [1, 1, n_chunks, H_v * n_seqs];
    // KDA: [1, S_k, n_chunks, H_v * n_seqs].
    var g_last = c.ggml_view_4d(ctx, g_cs, 1, g_cs.*.ne[1], g_cs.*.ne[2], g_cs.*.ne[3], g_cs.*.nb[1], g_cs.*.nb[2], g_cs.*.nb[3], c.ggml_row_size(g_cs.*.type, g_cs.*.ne[0] - 1));
    self.cb(g_last, "g_last", il);

    // CUDA does not support non-contiguous unary ops yet.
    g_last = c.ggml_cont(ctx, g_last);

    // [1, 1, n_chunks, H_v * n_seqs]; KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    const g_last_exp_t = c.ggml_transpose(ctx, c.ggml_exp(ctx, g_last));
    self.cb(g_last_exp_t, "g_last_exp_t", il);

    // [CS, 1, n_chunks, H_v * n_seqs]; KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    const g_diff = c.ggml_neg(ctx, c.ggml_sub(ctx, g_cs, g_last));
    self.cb(g_diff, "g_diff", il);

    const g_diff_exp_t = c.ggml_cont(ctx, c.ggml_transpose(ctx, c.ggml_exp(ctx, g_diff)));

    // [S_k, CS, n_chunks, H_v * n_seqs]
    const kg = c.ggml_mul(ctx, k, g_diff_exp_t);
    self.cb(kg, "key_gdiff", il);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    const kg_t = c.ggml_cont(ctx, c.ggml_transpose(ctx, kg));
    self.cb(kg_t, "key_gdiff_t", il);

    s = c.ggml_reshape_4d(ctx, s, S_v, S_v, 1, H_v * n_seqs);
    self.cb(s, "dnet_add_ch_state", il);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    const v_t = c.ggml_cont(ctx, c.ggml_transpose(ctx, v));

    var chunk: i64 = 0;
    while (chunk < n_chunks) : (chunk += 1) {
        const ch_k_cd = getSlice2d(ctx, k_cd, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        const ch_v_t = getSlice2d(ctx, v_t, chunk); // [ CS, S_v, 1, H_v * n_seqs]
        const ch_kq = getSlice2d(ctx, kq, chunk); // [ CS,  CS, 1, H_k * n_seqs]
        const ch_q_g_exp = getSlice2d(ctx, q_g_exp, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        const ch_kg_t = getSlice2d(ctx, kg_t, chunk); // [ CS, S_k, 1, H_v * n_seqs]

        // [CS, S_v, 1, H_v * n_seqs]
        const v_t_p = c.ggml_mul_mat(ctx, ch_k_cd, s);
        self.cb(v_t_p, "v_prime", il);

        const v_t_new = c.ggml_sub(ctx, ch_v_t, v_t_p);
        self.cb(v_t_new, "v_t_new", il);

        // [S_v, CS, 1, H_v * n_seqs]
        const v_attn = c.ggml_mul_mat(ctx, v_t_new, ch_kq);
        self.cb(v_attn, "v_attn", il);

        const attn_inter = c.ggml_mul_mat(ctx, s, ch_q_g_exp);
        self.cb(attn_inter, "attn_inter", il);

        const o_ch = c.ggml_add(ctx, attn_inter, v_attn);
        self.cb(o_ch, "dnet_add_ch_attn_out", il);

        v = c.ggml_set_inplace(ctx, v, o_ch, v.nb[1], v.nb[2], v.nb[3], @as(usize, @intCast(chunk)) * v.nb[2]);

        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        const kgv = c.ggml_mul_mat(ctx, ch_kg_t, v_t_new); // [S_k, S_v, 1, H_k * n_seqs]

        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
        const ch_g_last_exp_t = getSlice2d(ctx, g_last_exp_t, chunk);
        s = c.ggml_mul(ctx, s, ch_g_last_exp_t);
        s = c.ggml_add(ctx, s, kgv);
        self.cb(s, "dnet_add_ch_state", il);
    }

    // Truncate the padded tokens.
    var o = c.ggml_view_4d(ctx, v, S_v, n_tokens, H_v, n_seqs, c.ggml_row_size(v.type, S_v), c.ggml_row_size(v.type, S_v * CS * n_chunks), c.ggml_row_size(v.type, S_v * CS * n_chunks * H_v), 0);
    o = c.ggml_permute(ctx, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = c.ggml_reshape_4d(ctx, s, S_v, S_v, H_v, n_seqs);
    self.cb(s, "output_state", il);

    return .{ .out = o, .state = s };
}

/// Ports `llm_build_delta_net_base`'s `build_delta_net_autoregressive`
/// (delta-net-base.cpp:289 @c1d0e7a00): one token.
pub fn buildDeltaNetAutoregressive(self: *const Builder, q_: *Tensor, k_: *Tensor, v_: *Tensor, g_: *Tensor, b_: *Tensor, s_: *Tensor, il: c_int) OutState {
    const ctx = self.ctx0;
    var q = q_;
    var k = k_;
    var v = v_;
    var g = g_;
    var b = b_;
    var s = s_;

    const S_k = q.ne[0];
    const n_tokens = q.ne[2];
    const n_seqs = q.ne[3];
    const H_v = v.ne[1];

    impl.assert(n_tokens == 1, "delta-net-base.cpp", 305, "n_tokens == 1");
    checkShapes(q, k, v, g, b, s, 307);

    const scale: f32 = 1.0 / impl.sqrtf(@floatFromInt(S_k));
    q = c.ggml_scale(ctx, q, scale);

    q = c.ggml_permute(ctx, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = c.ggml_permute(ctx, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = c.ggml_permute(ctx, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

    self.cb(q, "q_in", il);
    self.cb(k, "k_in", il);
    self.cb(v, "v_in", il);
    self.cb(b, "b_in", il);
    self.cb(g, "g_in", il);

    // GDA: [1, 1, H_v, n_seqs]; KDA: [1, S_k, H_v, n_seqs]
    g = c.ggml_reshape_4d(ctx, g, 1, g.ne[0], H_v, n_seqs);
    b = c.ggml_reshape_4d(ctx, b, 1, 1, H_v, n_seqs);

    // [S_v, S_v, H_v, n_seqs]
    g = c.ggml_exp(ctx, g);
    s = c.ggml_mul(ctx, s, g);

    // [1, S_v, H_v, n_seqs]
    var sk = c.ggml_mul(ctx, s, k);
    sk = c.ggml_sum_rows(ctx, sk);

    // [S_v, 1, H_v, n_seqs]
    var d = c.ggml_sub(ctx, v, c.ggml_transpose(ctx, sk));
    d = c.ggml_mul(ctx, d, b);

    // [1, S_v, H_v, n_seqs]
    const d_t = c.ggml_transpose(ctx, d);

    // [S_v, S_v, H_v, n_seqs]
    k = c.ggml_repeat(ctx, k, s);
    const kd = c.ggml_mul(ctx, k, d_t);

    s = c.ggml_add(ctx, s, kd);
    self.cb(s, "dnet_add_ar_state", il);

    const s_q = c.ggml_mul(ctx, s, q);
    var o = c.ggml_sum_rows(ctx, s_q);
    o = c.ggml_permute(ctx, o, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]

    return .{ .out = o, .state = s };
}

/// Ports `llm_build_delta_net_base`'s `build_delta_net_fused`
/// (delta-net-base.cpp:373 @c1d0e7a00): the fused operator, K = 1, which
/// carries the final state only.
pub fn buildDeltaNetFused(self: *const Builder, q: *Tensor, k: *Tensor, v: *Tensor, g: *Tensor, b: *Tensor, s: *Tensor, il: c_int) OutState {
    const ctx = self.ctx0;
    const n_tokens = q.ne[2];
    const n_seqs = q.ne[3];
    const S_v = v.ne[0];
    const H_v = v.ne[1];

    checkShapes(q, k, v, g, b, s, 389);

    const result = c.ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
    self.res.addFusedNode(.{ .op = if (n_tokens == 1) .gdn_ar else .gdn_ch, .tensor = result, .il = il });

    const output = c.ggml_view_4d(ctx, result, S_v, H_v, n_tokens, n_seqs, c.ggml_row_size(result.*.type, S_v), c.ggml_row_size(result.*.type, S_v * H_v), c.ggml_row_size(result.*.type, S_v * H_v * n_tokens), 0);
    const new_state = c.ggml_view_4d(ctx, result, S_v, S_v, H_v, n_seqs, c.ggml_row_size(result.*.type, S_v), c.ggml_row_size(result.*.type, S_v * S_v), c.ggml_row_size(result.*.type, S_v * S_v * H_v), c.ggml_row_size(result.*.type, S_v * H_v * n_tokens * n_seqs));

    return .{ .out = output, .state = new_state };
}

/// Ports `llm_build_delta_net_base`'s `build_delta_net` (delta-net-base.cpp:425
/// @c1d0e7a00): fused or not, autoregressive or chunked, by the token count.
pub fn buildDeltaNet(self: *const Builder, q: *Tensor, k: *Tensor, v: *Tensor, g: *Tensor, b: *Tensor, s: *Tensor, il: c_int) OutState {
    const n_seq_tokens = q.ne[2];
    if (n_seq_tokens == 1) {
        if (self.cparams.fused_gdn_ar) return buildDeltaNetFused(self, q, k, v, g, b, s, il);
        return buildDeltaNetAutoregressive(self, q, k, v, g, b, s, il);
    }
    if (self.cparams.fused_gdn_ch) return buildDeltaNetFused(self, q, k, v, g, b, s, il);
    return buildDeltaNetChunking(self, q, k, v, g, b, s, il);
}

/// Ports `llm_build_delta_net_base`'s `build_conv_state` (delta-net-base.cpp:449
/// @c1d0e7a00): the convolution's input with the saved state in front, and
/// the state written back.
pub fn buildConvState(self: *const Builder, inp: *const graph.InpRs, conv_states_all: *Tensor, qkv_mixed_: *Tensor, conv_kernel_size: i64, conv_channels: i64, il: c_int) *Tensor {
    const ctx = self.ctx0;
    const mctx_cur = inp.mctx;
    const kv_head: i64 = mctx_cur.getHead();
    const mem_size: i64 = mctx_cur.getSize();
    const n_seqs: i64 = self.ubatch.n_seqs;

    var conv_states = self.buildRs(inp, conv_states_all, @intCast(self.hparams.nEmbdR()), @intCast(n_seqs), null);
    self.cb(conv_states, "conv_states", il);

    conv_states = c.ggml_reshape_3d(ctx, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    self.cb(conv_states, "conv_states_reshaped", il);

    const qkv_mixed = c.ggml_transpose(ctx, qkv_mixed_);
    self.cb(qkv_mixed, "qkv_mixed_transposed", il);

    const conv_input = c.ggml_concat(ctx, conv_states, qkv_mixed, 0);
    self.cb(conv_input, "conv_input", il);

    const row_count = (conv_kernel_size - 1) * conv_channels;
    const row_size = c.ggml_row_size(conv_states_all.type, row_count);

    if (self.cparams.n_rs_seq == 0) {
        const s_idx = conv_input.*.ne[0] - conv_states.*.ne[0];
        const s_slot: i64 = 0;
        const conv_state_last = c.ggml_view_3d(ctx, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input.*.nb[1], conv_input.*.nb[2], c.ggml_row_size(conv_input.*.type, s_idx));
        self.cb(conv_state_last, "conv_state_last", il);
        const conv_state_update = c.ggml_view_2d(ctx, conv_states_all, row_count, n_seqs, conv_states_all.nb[1], @as(usize, @intCast(s_slot * mem_size + kv_head)) * row_size);
        self.cb(conv_state_update, "conv_state_update", il);
        c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, conv_state_last, conv_state_update));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last n_rs_seq + 1 tokens of
        // a sequence are in the same ubatch, as split_equal guarantees.
        const K: i64 = @as(i64, self.cparams.n_rs_seq) + 1;
        var t: i64 = 1;
        while (t <= K) : (t += 1) {
            const s_idx = @max(0, conv_input.*.ne[0] - conv_states.*.ne[0] - K + t);
            const s_slot = K - t;
            const conv_state_last = c.ggml_view_3d(ctx, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input.*.nb[1], conv_input.*.nb[2], c.ggml_row_size(conv_input.*.type, s_idx));
            const conv_state_update = c.ggml_view_2d(ctx, conv_states_all, row_count, n_seqs, conv_states_all.nb[1], @as(usize, @intCast(s_slot * mem_size + kv_head)) * row_size);
            c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, conv_state_last, conv_state_update));
        }
    }
    return conv_input;
}

/// Ports `llm_build_delta_net_base`'s `build_recurrent_attn` (delta-net-base.cpp:527
/// @c1d0e7a00): the delta net over the saved state, with the new state, or
/// its rollback snapshots, written back.
pub fn buildRecurrentAttn(self: *const Builder, inp: *const graph.InpRs, ssm_states_all: *Tensor, q: *Tensor, k: *Tensor, v: *Tensor, g: *Tensor, b: *Tensor, s: *Tensor, il: c_int) *Tensor {
    const ctx = self.ctx0;
    const mctx_cur = inp.mctx;
    const kv_head: usize = mctx_cur.getHead();
    const mem_size: usize = mctx_cur.getSize();

    const S_v = s.ne[0];
    const H_v = s.ne[2];
    const n_seqs = s.ne[3];
    const n_seq_tokens = q.ne[2];

    const keep = self.cparams.n_rs_seq > 0;
    const n_embd_s: i64 = self.hparams.nEmbdS();

    if (!keep) {
        const attn_out = buildDeltaNet(self, q, k, v, g, b, s, il);
        self.cb(attn_out.out, "attn_output", il);
        self.cb(attn_out.state, "new_state", il);
        c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, attn_out.state, c.ggml_view_2d(ctx, ssm_states_all, n_embd_s, n_seqs, ssm_states_all.nb[1], kv_head * @as(usize, @intCast(n_embd_s)) * c.ggml_element_size(ssm_states_all))));
        return attn_out.out;
    }

    const D = S_v * S_v * H_v;
    const K: i64 = @as(i64, self.cparams.n_rs_seq) + 1;

    // K snapshot slots are written into the output.
    const gdn_out = c.ggml_gated_delta_net(ctx, q, k, v, g, b, s, K);
    self.res.addFusedNode(.{ .op = if (n_seq_tokens > 1) .gdn_ch else .gdn_ar, .tensor = gdn_out, .il = il });

    const attn_score_elems = S_v * H_v * n_seq_tokens * n_seqs;
    const state_size_per_snap = S_v * S_v * H_v * n_seqs;

    const output = c.ggml_view_4d(ctx, gdn_out, S_v, H_v, n_seq_tokens, n_seqs, c.ggml_row_size(gdn_out.*.type, S_v), c.ggml_row_size(gdn_out.*.type, S_v * H_v), c.ggml_row_size(gdn_out.*.type, S_v * H_v * n_seq_tokens), 0);
    self.cb(output, "attn_output", il);

    const row_size: usize = @as(usize, @intCast(n_embd_s)) * c.ggml_element_size(ssm_states_all);

    // The op writes the last min(n_seq_tokens, K) snapshots.
    const n_written = @min(n_seq_tokens, K);

    // Snapshot slot i goes to rollback group i.
    const src = c.ggml_view_3d(ctx, gdn_out, D, n_seqs, n_written, c.ggml_row_size(gdn_out.*.type, D), c.ggml_row_size(gdn_out.*.type, state_size_per_snap), c.ggml_row_size(gdn_out.*.type, attn_score_elems));
    const dst = c.ggml_view_3d(ctx, ssm_states_all, D, n_seqs, n_written, ssm_states_all.nb[1], mem_size * row_size, kv_head * row_size);
    c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, src, dst));

    return output;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
