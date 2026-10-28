//! The Mamba and Mamba-2 selective state-space layers, shared by the
//! architectures built on them: the pure state-space models and the hybrids
//! that interleave them with attention.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/mamba-base.cpp` at v0.4.0
//! (`5266f24da`): every member of `llm_build_mamba_base`.
//!
//! The C++ makes these members of a subclass of `llm_graph_context`; here
//! they are functions over the builder.

const std = @import("std");
const impl = @import("../impl.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const Builder = graph.Builder;
const Model = model_mod.Model;

/// The `get_ssm_rows` lambda the layers hand `build_rs`: reshape the
/// gathered states and scan them. Its fields are what the lambda captures.
///
/// The states and the indices `build_rs` provides are used so that the
/// states are read before they are overwritten, without copying them.
pub const SsmRows = struct {
    d_state: i64,
    head_dim: i64,
    n_head: i64,
    /// The states' slot count, the reshape's last dimension.
    n_slots: i64,
    x: *Tensor,
    dt: *Tensor,
    A: *Tensor,
    B: *Tensor,
    C: *Tensor,
    /// Above 1, the backend also returns rollback snapshots.
    K: i64,

    fn call(context: ?*const anyopaque, ctx: *c.ggml_context, states: *Tensor, ids: *Tensor) *Tensor {
        const self: *const SsmRows = @ptrCast(@alignCast(context.?));
        const ssm: *Tensor = c.ggml_reshape_4d(ctx, states, self.d_state, self.head_dim, self.n_head, self.n_slots);

        // The custom operator for the parallel associative scan of the
        // Mamba paper's Annex D.
        // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
        return c.ggml_ssm_scan(ctx, ssm, self.x, self.dt, self.A, self.B, self.C, ids, self.K);
    }

    /// Return: the closure, borrowing `self` until `build_rs` returns.
    pub fn rowsFn(self: *const SsmRows) graph.GetRowsFn {
        return .{ .context = self, .func = call };
    }
};

/// Ports `llm_build_mamba_base`'s `build_mamba_layer` (models/mamba-base.cpp:9
/// @5266f24da): a Mamba-1 block over the recurrent states.
///
/// Return: `{n_embd, n_tokens}`.
pub fn buildMambaLayer(b: *const Builder, inp: *const graph.InpRs, cur_: *Tensor, model: *const Model, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    const mctx_cur = inp.mctx;

    const kv_head: usize = mctx_cur.getHead();

    const layer = &model.layers[@intCast(il)];

    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const dt_rank: i64 = hp.ssm_dt_rank;
    const n_head: i64 = d_inner;
    const head_dim: i64 = 1;
    const n_seqs: i64 = ubatch.n_seqs;
    // Some variants of the Mamba architecture (FalconMamba) apply a layer
    // norm on B and Dt.
    const ssm_dt_b_c_rms = hp.ssm_dt_b_c_rms;

    const n_seq_tokens: i64 = ubatch.n_seq_tokens;

    impl.assert(n_seqs != 0, "models/mamba-base.cpp", 32, "n_seqs != 0");
    impl.assert(ubatch.equalSeqs(), "models/mamba-base.cpp", 33, "ubatch.equal_seqs()");
    impl.assert(ubatch.n_tokens == n_seq_tokens * n_seqs, "models/mamba-base.cpp", 34, "ubatch.n_tokens == n_seq_tokens * n_seqs");
    impl.assert(@rem(d_inner, n_head) == 0, "models/mamba-base.cpp", 35, "d_inner % n_head == 0");

    const conv_states_all = mctx_cur.getRL(il);
    const ssm_states_all = mctx_cur.getSL(il);

    var conv = b.buildRs(inp, conv_states_all, @bitCast(hp.nEmbdR()), @intCast(n_seqs), null);
    conv = c.ggml_reshape_3d(ctx, conv, d_conv - 1, d_inner, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    var cur: *Tensor = c.ggml_reshape_3d(ctx, cur_, cur_.ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    const xz = b.buildLoraMm(layer.ssm_in.?, cur, layer.ssm_in_s);
    // Split in two: => {d_inner, n_seq_tokens, n_seqs}
    var x: *Tensor = c.ggml_view_3d(ctx, xz, d_inner, xz.ne[1], xz.ne[2], xz.nb[1], xz.nb[2], 0);
    const z: *Tensor = c.ggml_view_3d(ctx, xz, d_inner, xz.ne[1], xz.ne[2], xz.nb[1], xz.nb[2], @as(usize, @intCast(d_inner)) * c.ggml_element_size(xz));

    // The convolution.
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        const conv_x: *Tensor = c.ggml_concat(ctx, conv, c.ggml_transpose(ctx, x), 0);

        // Copy the last (d_conv - 1) columns back into the state cache.
        const last_conv = c.ggml_view_3d(ctx, conv_x, d_conv - 1, d_inner, n_seqs, conv_x.nb[1], conv_x.nb[2], @as(usize, @intCast(n_seq_tokens)) * conv_x.nb[0]);

        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, last_conv, c.ggml_view_1d(ctx, conv_states_all, (d_conv - 1) * d_inner * n_seqs, kv_head * @as(usize, @intCast((d_conv - 1) * d_inner)) * c.ggml_element_size(conv_states_all))));

        // The 1D convolution: a self-overlapping view of conv_x over d_conv
        // columns at each stride of the third dimension, multiplied
        // element-wise by the conv1d weight, each row summed, and the ne[0]
        // dimension permuted away. Simultaneous sequences must all have the
        // same length.
        x = c.ggml_ssm_conv(ctx, conv_x, layer.ssm_conv1d);

        // The bias.
        x = c.ggml_add(ctx, x, layer.ssm_conv1d_b);

        x = c.ggml_silu(ctx, x);
    }

    // The SSM.
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        const x_db = b.buildLoraMm(layer.ssm_x.?, x, null);
        // Split.
        var dt: *Tensor = c.ggml_view_3d(ctx, x_db, dt_rank, n_seq_tokens, n_seqs, x_db.nb[1], x_db.nb[2], 0);
        var B: *Tensor = c.ggml_view_4d(ctx, x_db, d_state, 1, n_seq_tokens, n_seqs, @as(usize, @intCast(d_state)) * x_db.nb[0], x_db.nb[1], x_db.nb[2], c.ggml_element_size(x_db) * @as(usize, @intCast(dt_rank)));
        var C: *Tensor = c.ggml_view_4d(ctx, x_db, d_state, 1, n_seq_tokens, n_seqs, @as(usize, @intCast(d_state)) * x_db.nb[0], x_db.nb[1], x_db.nb[2], c.ggml_element_size(x_db) * @as(usize, @intCast(dt_rank + d_state)));

        // Some Mamba variants (FalconMamba, Jamba) apply an RMS norm to B,
        // C and Dt.
        if (ssm_dt_b_c_rms or (layer.ssm_dt_norm != null and layer.ssm_b_norm != null and layer.ssm_c_norm != null)) {
            dt = b.buildNorm(dt, layer.ssm_dt_norm, null, .rms, il);
            B = b.buildNorm(B, layer.ssm_b_norm, null, .rms, il);
            C = b.buildNorm(C, layer.ssm_c_norm, null, .rms, il);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = b.buildLoraMm(layer.ssm_dt.?, dt, null);
        dt = c.ggml_add(ctx, dt, layer.ssm_dt_b);

        cur = x;
        x = c.ggml_reshape_4d(ctx, x, head_dim, n_head, n_seq_tokens, n_seqs);

        const rows: SsmRows = .{ .d_state = d_state, .head_dim = head_dim, .n_head = n_head, .n_slots = mctx_cur.getSize(), .x = x, .dt = dt, .A = layer.ssm_a.?, .B = B, .C = C, .K = 1 };
        const y_ssm = b.buildRs(inp, ssm_states_all, @bitCast(hp.nEmbdS()), @intCast(ubatch.n_seqs), rows.rowsFn());

        // Store the last states.
        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, c.ggml_view_1d(ctx, y_ssm, d_state * d_inner * n_seqs, x.nb[3] * @as(usize, @intCast(x.ne[3]))), c.ggml_view_1d(ctx, ssm_states_all, d_state * d_inner * n_seqs, kv_head * @as(usize, @intCast(d_state * d_inner)) * c.ggml_element_size(ssm_states_all))));

        var y: *Tensor = c.ggml_view_3d(ctx, y_ssm, d_inner, n_seq_tokens, n_seqs, x.nb[2], x.nb[3], 0);

        // TODO: skip computing output earlier for unused tokens

        y = c.ggml_add(ctx, y, c.ggml_mul(ctx, cur, layer.ssm_d));
        y = c.ggml_swiglu_split(ctx, c.ggml_cont(ctx, z), y);

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = b.buildLoraMm(layer.ssm_out.?, y, layer.ssm_out_s);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = c.ggml_reshape_2d(ctx, cur, cur.ne[0], n_seq_tokens * n_seqs);

    return cur;
}

/// Ports `llm_build_mamba_base`'s `build_mamba2_layer` (models/mamba-base.cpp:151
/// @5266f24da): a Mamba-2 block over the recurrent states, with rollback
/// snapshots when the context keeps them.
///
/// Return: `{n_embd, n_tokens}`.
pub fn buildMamba2Layer(b: *const Builder, inp: *const graph.InpRs, cur_: *Tensor, model: *const Model, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const ubatch = b.ubatch;
    const mctx_cur = inp.mctx;
    const layer = &model.layers[@intCast(il)];

    const kv_head: usize = mctx_cur.getHead();
    const mem_size: usize = mctx_cur.getSize();

    const d_conv: i64 = hp.ssm_d_conv;
    const d_inner: i64 = hp.ssm_d_inner;
    const d_state: i64 = hp.ssm_d_state;
    const n_head: i64 = hp.ssm_dt_rank;
    const head_dim: i64 = @divTrunc(d_inner, n_head);
    const n_group: i64 = hp.ssm_n_group;
    const n_seqs: i64 = ubatch.n_seqs;

    const n_seq_tokens: i64 = ubatch.n_seq_tokens;
    const K: i64 = if (b.cparams.n_rs_seq > 0) @as(i64, b.cparams.n_rs_seq) + 1 else 1;

    impl.assert(n_seqs != 0, "models/mamba-base.cpp", 172, "n_seqs != 0");
    impl.assert(ubatch.equalSeqs(), "models/mamba-base.cpp", 173, "ubatch.equal_seqs()");
    impl.assert(ubatch.n_tokens == n_seq_tokens * n_seqs, "models/mamba-base.cpp", 174, "ubatch.n_tokens == n_seq_tokens * n_seqs");
    impl.assert(@rem(d_inner, n_head) == 0, "models/mamba-base.cpp", 175, "d_inner % n_head  == 0");
    impl.assert(@rem(d_inner, n_group) == 0, "models/mamba-base.cpp", 176, "d_inner % n_group == 0");

    const conv_states_all = mctx_cur.getRL(il);
    const ssm_states_all = mctx_cur.getSL(il);
    const state_slots: i64 = ssm_states_all.ne[1];

    var conv = b.buildRs(inp, conv_states_all, @bitCast(hp.nEmbdR()), @intCast(n_seqs), null);
    conv = c.ggml_reshape_3d(ctx, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs);

    // d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    // The projection stays 2D: with a {n_embd, 1, n_seqs} batch the CUDA
    // backend dispatches a column-batched GEMV for what is a large dense GEMM.
    // {n_embd, d_in_proj} @ {n_embd, n_tokens} => {d_in_proj, n_tokens}
    var zxBCdt = b.buildLoraMm(layer.ssm_in.?, cur_, layer.ssm_in_s);
    // {d_in_proj, n_tokens} => {d_in_proj, n_seq_tokens, n_seqs}
    zxBCdt = c.ggml_reshape_3d(ctx, zxBCdt, zxBCdt.ne[0], n_seq_tokens, n_seqs);

    // Split in three.
    const z: *Tensor = c.ggml_view_4d(ctx, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, @as(usize, @intCast(head_dim)) * zxBCdt.nb[0], zxBCdt.nb[1], zxBCdt.nb[2], 0);
    var xBC: *Tensor = c.ggml_view_3d(ctx, zxBCdt, d_inner + 2 * n_group * d_state, n_seq_tokens, n_seqs, zxBCdt.nb[1], zxBCdt.nb[2], @as(usize, @intCast(d_inner)) * c.ggml_element_size(zxBCdt));
    var dt: *Tensor = c.ggml_view_3d(ctx, zxBCdt, n_head, n_seq_tokens, n_seqs, zxBCdt.nb[1], zxBCdt.nb[2], @as(usize, @intCast(2 * d_inner + 2 * n_group * d_state)) * c.ggml_element_size(zxBCdt));

    // The convolution.
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner + 2*n_group*d_state, n_seqs}
        const conv_x: *Tensor = c.ggml_concat(ctx, conv, c.ggml_transpose(ctx, xBC), 0);

        const row_count: i64 = (d_conv - 1) * (d_inner + 2 * n_group * d_state);
        const row_size: usize = c.ggml_row_size(conv_states_all.type, row_count);
        const n_written: i64 = @min(n_seq_tokens, K);

        var slot: i64 = 0;
        while (slot < n_written) : (slot += 1) {
            const last_conv = c.ggml_view_3d(ctx, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs, conv_x.nb[1], conv_x.nb[2], @as(usize, @intCast(n_seq_tokens - slot)) * conv_x.nb[0]);

            c.ggml_build_forward_expand(b.gf, c.ggml_cpy(ctx, last_conv, c.ggml_view_2d(ctx, conv_states_all, row_count, n_seqs, conv_states_all.nb[1], (@as(usize, @intCast(slot)) * mem_size + kv_head) * row_size)));
        }

        // The 1D convolution: a self-overlapping view of conv_x over d_conv
        // columns at each stride of the third dimension, multiplied
        // element-wise by the conv1d weight, each row summed, and the ne[0]
        // dimension permuted away. Simultaneous sequences must all have the
        // same length.
        xBC = c.ggml_ssm_conv(ctx, conv_x, layer.ssm_conv1d);

        // The bias.
        xBC = c.ggml_add(ctx, xBC, layer.ssm_conv1d_b);

        xBC = c.ggml_silu(ctx, xBC);
    }

    // The SSM.
    var cur: *Tensor = undefined;
    {
        // These correspond to V, K and Q in the SSM/attention duality.
        const x: *Tensor = c.ggml_view_4d(ctx, xBC, head_dim, n_head, n_seq_tokens, n_seqs, @as(usize, @intCast(head_dim)) * xBC.nb[0], xBC.nb[1], xBC.nb[2], 0);
        const B: *Tensor = c.ggml_view_4d(ctx, xBC, d_state, n_group, n_seq_tokens, n_seqs, @as(usize, @intCast(d_state)) * xBC.nb[0], xBC.nb[1], xBC.nb[2], @as(usize, @intCast(d_inner)) * c.ggml_element_size(xBC));
        const C: *Tensor = c.ggml_view_4d(ctx, xBC, d_state, n_group, n_seq_tokens, n_seqs, @as(usize, @intCast(d_state)) * xBC.nb[0], xBC.nb[1], xBC.nb[2], @as(usize, @intCast(d_inner + n_group * d_state)) * c.ggml_element_size(xBC));

        // {n_head, n_seq_tokens, n_seqs}
        dt = c.ggml_add(ctx, c.ggml_cont(ctx, dt), layer.ssm_dt_b);

        // TODO: use semistructured matrices to implement state-space duality
        // K > 1 asks the backend for rollback snapshots besides the final state.
        const rows: SsmRows = .{ .d_state = d_state, .head_dim = head_dim, .n_head = n_head, .n_slots = state_slots, .x = x, .dt = dt, .A = layer.ssm_a.?, .B = B, .C = C, .K = K };
        const y_ssm = b.buildRs(inp, ssm_states_all, @bitCast(hp.nEmbdS()), @intCast(ubatch.n_seqs), rows.rowsFn());
        const D: i64 = d_state * d_inner;
        const n_written: i64 = @min(n_seq_tokens, K);
        const row_size: usize = c.ggml_row_size(ssm_states_all.type, D);
        const y_row_size: usize = c.ggml_row_size(y_ssm.type, D);
        const state_offset: usize = @as(usize, @intCast(c.ggml_nelements(x))) * c.ggml_element_size(x);

        c.ggml_build_forward_expand(b.gf, c.ggml_cpy(
            ctx,
            c.ggml_view_3d(ctx, y_ssm, D, n_seqs, n_written, y_row_size, y_row_size * @as(usize, @intCast(n_seqs)), state_offset),
            c.ggml_view_3d(ctx, ssm_states_all, D, n_seqs, n_written, ssm_states_all.nb[1], mem_size * row_size, kv_head * row_size),
        ));

        var y: *Tensor = c.ggml_view_4d(ctx, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x.nb[1], @as(usize, @intCast(n_head)) * x.nb[1], @as(usize, @intCast(n_seq_tokens * n_head)) * x.nb[1], 0);

        // TODO: skip computing output earlier for unused tokens

        y = c.ggml_add(ctx, y, c.ggml_mul(ctx, x, layer.ssm_d));
        b.cb(y, "mamba2_y_add_d", il);
        y = c.ggml_swiglu_split(ctx, c.ggml_cont(ctx, z), y);

        // The grouped RMS norm.
        if (layer.ssm_norm != null) {
            y = c.ggml_reshape_4d(ctx, y, @divTrunc(d_inner, n_group), n_group, n_seq_tokens, n_seqs);
            y = b.buildNorm(y, layer.ssm_norm, null, .rms, il);
        }

        y = c.ggml_reshape_2d(ctx, y, d_inner, n_seq_tokens * n_seqs);

        // {d_inner, n_embd} @ {d_inner, n_tokens} => {n_embd, n_tokens}
        cur = b.buildLoraMm(layer.ssm_out.?, y, layer.ssm_out_s);
    }

    b.cb(cur, "mamba_out", il);
    return cur;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
