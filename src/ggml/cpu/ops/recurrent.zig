//! The linear-recurrence kernels: `rwkv_wkv6`, `gla`, `gated_delta_net`, the
//! three `dsv4_hc_*` hyper-connection ops, and `rwkv_wkv7`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! `solve_tri` sits between `gla` and `gated_delta_net` in the C; it belongs to
//! the custom-ops family and is not here.
//!
//! # `gated_delta_net` is on the deliverable's path
//!
//! Qwen3.5's linear-attention layers are built from it. On a Metal machine the
//! CPU kernel does not run during inference, but it is the reference
//! semantics, so it is ported with the same care as the rest.
//!
//! # Which arms compile here
//!
//! `__ARM_NEON && __aarch64__` and `GGML_SIMD` are defined; SVE, AVX and RVV
//! are not. So:
//!
//! - `rwkv_wkv6` and `gla` take their `WKV_VECTOR_SIZE` / `GLA_VECTOR_SIZE`
//!   NEON arm, four lanes. Every lane is independent — there is no reduction
//!   across `j` — and each lane runs exactly the fused operations the scalar
//!   remainder writes as `a*b + c` expressions, which clang fuses too. So body
//!   and remainder are the same function of each element, and the port is one
//!   elementwise loop that names each fusion once. Changing the vector width
//!   could not change a bit.
//! - `rwkv_wkv7` takes the `GGML_SIMD` non-SVE arm. That one **does** reduce
//!   across `j` — twice, through four `float32x4_t` accumulators, a halving
//!   tree and the pairwise `vaddvq_f32` — so it is ported vector for vector.
//!
//! # Loop index names
//!
//! The C uses `i`, `j`, `t`, `h` — none of them reserved — plus `i0`, which is
//! renamed `j0` where it appears.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const neon = @import("../quants/arm/neon.zig");
const threading = @import("../threading.zig");
const defs = @import("../defs.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

const f32x4 = neon.f32x4;

extern fn expf(x: f32) f32;
extern fn sqrtf(x: f32) f32;

inline fn barrier(params: *const ComputeParams) void {
    threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
}

inline fn f32Data(t: *const Tensor) [*]f32 {
    return @ptrCast(@alignCast(t.data.?));
}

inline fn at(comptime T: type, base: ?*anyopaque, off: usize) *T {
    const b: [*]u8 = @ptrCast(base.?);
    return @ptrCast(@alignCast(b + off));
}

/// The head range `[h_start, h_end)` thread `ith` owns, as `rwkv_wkv6`, `gla`
/// and `rwkv_wkv7` each compute it inline. The C does the arithmetic in
/// `int64_t` and narrows to `int`; head counts fit either way.
fn headRange(params: *const ComputeParams, heads: i64) struct { i64, i64 } {
    const ith: i64 = params.ith;
    const nth: i64 = params.nth;
    const h_start = @divTrunc(heads * ith, nth);
    const h_end = if (@divTrunc(heads * (ith + 1), nth) < heads) @divTrunc(heads * (ith + 1), nth) else heads;
    return .{ h_start, h_end };
}

/// Ports `ggml_compute_forward_rwkv_wkv6_f32` (ops.cpp:10270 @c1d0e7a00), the
/// `__ARM_NEON && __aarch64__` arm (`WKV_VECTOR_SIZE` 4).
///
/// Per element, with `kv = v * k` rounded on its own (`vmulq_f32`):
///
/// - `temp = prev_state + kv * time_faaaa`, fused (`vfmaq_f32`);
/// - `dst += temp * r`, fused;
/// - `state = kv + prev_state * time_decay`, fused.
///
/// The scalar remainder writes the same three as `a*b + c` and `+=` and clang
/// fuses them identically, so one loop over `j` covers both. See the file
/// header.
fn rwkvWkv6F32(params: *const ComputeParams, dst: *Tensor) void {
    const src1 = impl.one(Tensor, dst.src[1]);
    const src5 = impl.one(Tensor, dst.src[5]);

    const T = src1.ne[2];
    const C = dst.ne[0];
    const HEADS = src1.ne[1];
    const n_seqs = src5.ne[1];
    const head_size = @divTrunc(C, HEADS);

    const dst_data = f32Data(dst);
    const state = dst_data + @as(usize, @intCast(C * T));

    const h_start, const h_end = headRange(params, HEADS);

    const k = f32Data(impl.one(Tensor, dst.src[0]));
    const v = f32Data(src1);
    const r = f32Data(impl.one(Tensor, dst.src[2]));
    const time_faaaa = f32Data(impl.one(Tensor, dst.src[3]));
    const time_decay = f32Data(impl.one(Tensor, dst.src[4]));

    const t_stride: usize = @intCast(HEADS * head_size); // Same to C

    const h_stride: usize = @intCast(@divTrunc(C, HEADS));
    impl.assert(@rem(C, HEADS) == 0, "C % HEADS == 0"); // C must be divisible by HEADS
    const h_stride_2d: usize = @intCast(head_size * head_size);

    if (params.ith == 0) {
        @memset(dst_data[0..@intCast(T * C)], 0);
    }
    barrier(params);

    const hs: usize = @intCast(head_size);
    const seq_len = @divTrunc(T, n_seqs);

    var t: i64 = 0;
    while (t < T) : (t += 1) {
        const t_offset = @as(usize, @intCast(t)) * t_stride;
        const state_offset: usize = @intCast(head_size * C * @divTrunc(t, seq_len));
        const state_cur = state + state_offset;
        const state_prev = if (@rem(t, seq_len) != 0) state_cur else f32Data(src5) + state_offset;

        var h: usize = @intCast(h_start);
        while (h < h_end) : (h += 1) {
            const h_offset = h * h_stride;
            const t_h_offset = t_offset + h_offset;
            const h_2d_offset = h * h_stride_2d;

            for (0..hs) |i| {
                const t_h_i_offset = t_h_offset + i;
                const h_i_offset = h_offset + i;
                const h_2d_i_offset = h_2d_offset + i * h_stride;

                const k_val = k[t_h_i_offset];
                const r_val = r[t_h_i_offset];
                const time_faaaa_val = time_faaaa[h_i_offset];
                const time_decay_val = time_decay[t_h_i_offset];

                for (0..hs) |j| {
                    const t_h_j_offset = t_h_offset + j;
                    const h_2d_i_j_offset = h_2d_i_offset + j;

                    const kv_val = v[t_h_j_offset] * k_val;
                    const prev_state_val = state_prev[h_2d_i_j_offset];
                    // temp = kv * time_faaaa + prev_state, fused
                    const temp_val = @mulAdd(f32, kv_val, time_faaaa_val, prev_state_val);
                    // dst += temp * r, fused
                    dst_data[t_h_j_offset] = @mulAdd(f32, temp_val, r_val, dst_data[t_h_j_offset]);
                    // state = prev_state * time_decay + kv, fused
                    state_cur[h_2d_i_j_offset] = @mulAdd(f32, prev_state_val, time_decay_val, kv_val);
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_rwkv_wkv6` (ops.cpp:10462 @c1d0e7a00).
pub export fn ggml_compute_forward_rwkv_wkv6(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => rwkvWkv6F32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_gla_f32` (ops.cpp:10482 @c1d0e7a00), the
/// `__ARM_NEON && __aarch64__` arm (`GLA_VECTOR_SIZE` 4).
///
/// Per element: `kv = v * k` on its own, `temp = kv + prev_state * g` fused,
/// `dst += temp * q` fused. As with `rwkv_wkv6`, the vector body and the
/// scalar remainder agree element for element. `q` is pre-scaled by
/// `scale` once per `i`, outside the fused steps.
fn glaF32(params: *const ComputeParams, dst: *Tensor) void {
    const src1 = impl.one(Tensor, dst.src[1]);
    const src4 = impl.one(Tensor, dst.src[4]);

    const T = src1.ne[2];
    const C = dst.ne[0];
    const HEADS = src1.ne[1];
    const n_seqs = src4.ne[1];
    const head_size = @divTrunc(C, HEADS);
    const scale = impl.getOpParamsF32(dst, 0);

    const dst_data = f32Data(dst);
    const state = dst_data + @as(usize, @intCast(C * T));

    const h_start, const h_end = headRange(params, HEADS);

    const k = f32Data(impl.one(Tensor, dst.src[0]));
    const v = f32Data(src1);
    const q = f32Data(impl.one(Tensor, dst.src[2]));
    const g = f32Data(impl.one(Tensor, dst.src[3]));

    const t_stride: usize = @intCast(HEADS * head_size); // Same to C

    const h_stride: usize = @intCast(@divTrunc(C, HEADS));
    impl.assert(@rem(C, HEADS) == 0, "C % HEADS == 0"); // C must be divisible by HEADS
    const h_stride_2d: usize = @intCast(head_size * head_size);

    if (params.ith == 0) {
        @memset(dst_data[0..@intCast(T * C)], 0);
    }
    barrier(params);

    const hs: usize = @intCast(head_size);
    const seq_len = @divTrunc(T, n_seqs);

    var t: i64 = 0;
    while (t < T) : (t += 1) {
        const t_offset = @as(usize, @intCast(t)) * t_stride;
        const state_offset: usize = @intCast(head_size * C * @divTrunc(t, seq_len));
        const state_cur = state + state_offset;
        const state_prev = if (@rem(t, seq_len) != 0) state_cur else f32Data(src4) + state_offset;

        var h: usize = @intCast(h_start);
        while (h < h_end) : (h += 1) {
            const h_offset = h * h_stride;
            const t_h_offset = t_offset + h_offset;
            const h_2d_offset = h * h_stride_2d;

            for (0..hs) |i| {
                const t_h_i_offset = t_h_offset + i;
                const h_2d_i_offset = h_2d_offset + i * h_stride;

                const k_val = k[t_h_i_offset];
                const q_val = q[t_h_i_offset] * scale;
                const g_val = g[t_h_i_offset];

                for (0..hs) |j| {
                    const t_h_j_offset = t_h_offset + j;
                    const h_2d_i_j_offset = h_2d_i_offset + j;

                    const kv_val = v[t_h_j_offset] * k_val;
                    const prev_state_val = state_prev[h_2d_i_j_offset];
                    // temp = kv + prev_state * g, fused
                    const temp_val = @mulAdd(f32, prev_state_val, g_val, kv_val);
                    // dst += temp * q, fused
                    dst_data[t_h_j_offset] = @mulAdd(f32, temp_val, q_val, dst_data[t_h_j_offset]);
                    state_cur[h_2d_i_j_offset] = temp_val;
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_gla` (ops.cpp:10663 @c1d0e7a00).
pub export fn ggml_compute_forward_gla(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => glaF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_gated_delta_net_one_chunk`
/// (ops.cpp:10752 @c1d0e7a00).
///
/// Rows `[ir0, ir1)` are (head, sequence) pairs. The state is stored
/// transposed — `s_out[j*S_v + i] = S[i][j]` — so every per-token step below
/// is a contiguous row operation through the `vec.h` helpers: `scale_f32`
/// (vDSP), `mul_f32`, the NEON `ggml_vec_dot_f32`, and `mad_f32` (fused).
/// The arithmetic outside those helpers has no multiply feeding an add, so
/// nothing here is fused beyond what the helpers do.
fn gatedDeltaNetOneChunk(params: *const ComputeParams, dst: *Tensor, ir0: i64, ir1: i64) void {
    const src_q = impl.one(Tensor, dst.src[0]);
    const src_k = impl.one(Tensor, dst.src[1]);
    const src_v = impl.one(Tensor, dst.src[2]);
    const src_g = impl.one(Tensor, dst.src[3]);
    const src_beta = impl.one(Tensor, dst.src[4]);
    const src_state = impl.one(Tensor, dst.src[5]);

    const S_v = src_v.ne[0];
    const H = src_v.ne[1];
    const n_tokens = src_v.ne[2];
    const n_seqs = src_v.ne[3];

    impl.assert(c.ggml_is_contiguous_rows(src_q), "ggml_is_contiguous_rows(src_q)");
    impl.assert(c.ggml_is_contiguous_rows(src_k), "ggml_is_contiguous_rows(src_k)");
    impl.assert(c.ggml_is_contiguous_rows(src_v), "ggml_is_contiguous_rows(src_v)");
    impl.assert(c.ggml_is_contiguous(src_g), "ggml_is_contiguous(src_g)");
    impl.assert(c.ggml_is_contiguous(src_beta), "ggml_is_contiguous(src_beta)");
    impl.assert(c.ggml_is_contiguous(src_state), "ggml_is_contiguous(src_state)");

    impl.assert(src_g.ne[0] == 1 or src_g.ne[0] == S_v, "src_g->ne[0] == 1 || src_g->ne[0] == S_v");
    impl.assert(src_beta.ne[0] == 1, "src_beta->ne[0] == 1");

    // GGML_TENSOR_LOCALS for q, k, v, g and beta: only the names used.
    const neq1 = src_q.ne[1];
    const neq3 = src_q.ne[3];
    const nbq1 = src_q.nb[1];
    const nbq2 = src_q.nb[2];
    const nbq3 = src_q.nb[3];
    const nek1 = src_k.ne[1];
    const nek3 = src_k.ne[3];
    const nbk1 = src_k.nb[1];
    const nbk2 = src_k.nb[2];
    const nbk3 = src_k.nb[3];
    const nev3 = src_v.ne[3];
    const nbv1 = src_v.nb[1];
    const nbv2 = src_v.nb[2];
    const nbv3 = src_v.nb[3];
    const neg0 = src_g.ne[0];
    const nbg1 = src_g.nb[1];
    const nbg2 = src_g.nb[2];
    const nbg3 = src_g.nb[3];
    const nbb1 = src_beta.nb[1];
    const nbb2 = src_beta.nb[2];
    const nbb3 = src_beta.nb[3];

    const kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const K: i64 = impl.getOpParamsI32(dst, 0);
    impl.assert(K >= 1, "K >= 1");
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    const state_seq_stride: i64 = @intCast(src_state.nb[3] / @sizeOf(f32));

    const per_thread = S_v + (if (K > 1) S_v * S_v else 0);
    const ith: i64 = params.ith;

    const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const delta: [*]f32 = wbase + @as(usize, @intCast(ith * per_thread)) + common.cache_line_size_f32;
    const state_work: ?[*]f32 = if (K > 1) delta + @as(usize, @intCast(S_v)) else null;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    const attn_score_elems = S_v * H * n_tokens * n_seqs;
    const state_size_per_snap = S_v * S_v * H * n_seqs;
    const attn_out_base = f32Data(dst);
    const state_out_base = f32Data(dst) + @as(usize, @intCast(attn_score_elems));

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    const state_in_base: [*]const f32 = f32Data(src_state);

    //const int64_t rq1 = nev1 / neq1;
    //const int64_t rk1 = nev1 / nek1;
    const rq3 = @divTrunc(nev3, neq3);
    const rk3 = @divTrunc(nev3, nek3);

    const scale = 1.0 / sqrtf(@floatFromInt(S_v));

    const sv: usize = @intCast(S_v);
    const sv2 = sv * sv;

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const iv1 = @rem(ir, H); // head_index
        const iv3 = @divTrunc(ir, H); // sequence

        const iq1 = @rem(iv1, neq1);
        const ik1 = @rem(iv1, nek1);

        const iq3 = @divTrunc(iv3, rq3);
        const ik3 = @divTrunc(iv3, rk3);

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        const s_out: [*]f32 = if (K > 1)
            state_work.?
        else
            state_out_base + @as(usize, @intCast((iv3 * H + iv1) * S_v * S_v));

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        const s_in = state_in_base + @as(usize, @intCast(iv3 * state_seq_stride + iv1 * S_v * S_v));
        @memcpy(s_out[0..sv2], s_in[0..sv2]);

        // attn output pointer for first token of this (head, seq)
        var attn_data = attn_out_base + @as(usize, @intCast((iv3 * n_tokens * H + iv1) * S_v));

        var t: i64 = 0;
        while (t < n_tokens) : (t += 1) {
            const tu: usize = @intCast(t);
            const q_d: [*]const f32 = @ptrCast(at(f32, src_q.data, @as(usize, @intCast(iq3)) * nbq3 + tu * nbq2 + @as(usize, @intCast(iq1)) * nbq1));
            const k_d: [*]const f32 = @ptrCast(at(f32, src_k.data, @as(usize, @intCast(ik3)) * nbk3 + tu * nbk2 + @as(usize, @intCast(ik1)) * nbk1));
            const v_d: [*]const f32 = @ptrCast(at(f32, src_v.data, @as(usize, @intCast(iv3)) * nbv3 + tu * nbv2 + @as(usize, @intCast(iv1)) * nbv1));

            const beta_val = at(f32, src_beta.data, @as(usize, @intCast(iv3)) * nbb3 + tu * nbb2 + @as(usize, @intCast(iv1)) * nbb1).*;
            const g_d: [*]const f32 = @ptrCast(at(f32, src_g.data, @as(usize, @intCast(iv3)) * nbg3 + tu * nbg2 + @as(usize, @intCast(iv1)) * nbg1));

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            if (kda) {
                // precompute exp(g) into delta scratch (reused below)
                for (0..sv) |i| delta[i] = expf(g_d[i]);
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                for (0..sv) |j| vec.mul_f32(S_v, s_out + j * sv, s_out + j * sv, delta);
            } else {
                vec.scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            for (0..sv) |j| {
                var sum: f32 = 0.0;
                vec.dot_f32(@intCast(S_v), &sum, 0, s_out + j * sv, 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            for (0..sv) |j| vec.mad_f32(S_v, s_out + j * sv, k_d, delta[j]);

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            for (0..sv) |j| {
                var sum: f32 = 0.0;
                vec.dot_f32(@intCast(S_v), &sum, 0, s_out + j * sv, 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += @as(usize, @intCast(S_v * H)); // advance to next token

            if (K > 1) {
                const target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 and target_slot < K) {
                    const curr_state_o = state_out_base + @as(usize, @intCast(target_slot * state_size_per_snap +
                        (iv3 * H + iv1) * S_v * S_v));
                    @memcpy(curr_state_o[0..sv2], s_out[0..sv2]);
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_gated_delta_net_f32` (ops.cpp:10906 @c1d0e7a00).
///
/// Chunked work-stealing over the (head, sequence) rows, four chunks per
/// thread, through the threadpool's shared chunk counter — the same scheme
/// `mul_mat` uses.
fn gatedDeltaNetF32(params: *const ComputeParams, dst: *Tensor) void {
    const V = impl.one(Tensor, dst.src[2]);
    const nr = V.ne[1] * V.ne[3];

    // disable for NUMA
    const disable_chunking = threading.ggml_is_numa();

    const nth: i64 = params.nth;
    const ith: i64 = params.ith;

    // 4x chunks per thread
    const nth_scaled = nth * 4;
    const chunk_size = @divTrunc(nr + nth_scaled - 1, nth_scaled);
    var nchunk = @divTrunc(nr + chunk_size - 1, chunk_size);

    if (nth == 1 or nchunk < nth or disable_chunking) {
        nchunk = nth;
    }

    const tp: *defs.Threadpool = @ptrCast(@alignCast(params.threadpool.?));

    if (ith == 0) {
        threading.ggml_threadpool_chunk_set(tp, params.nth);
    }

    threading.ggml_barrier(tp);

    const dr = @divTrunc(nr + nchunk - 1, nchunk);

    var current_chunk: i64 = ith;

    while (current_chunk < nchunk) {
        const ir0 = dr * current_chunk;
        const ir1 = @min(ir0 + dr, nr);

        gatedDeltaNetOneChunk(params, dst, ir0, ir1);
        current_chunk = threading.ggml_threadpool_chunk_add(tp, 1);
    }
}

/// Ports `ggml_compute_forward_gated_delta_net` (ops.cpp:10947 @c1d0e7a00).
pub export fn ggml_compute_forward_gated_delta_net(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => gatedDeltaNetF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// The hyper-connection width the three `dsv4_hc_*` kernels hard-code as
/// `constexpr int64_t hc = 4`.
const hc_width: usize = 4;

/// Ports `ggml_dsv4_hc_comb_norm_cols` (ops.cpp:10967 @c1d0e7a00).
fn dsv4HcCombNormCols(comb: *[hc_width * hc_width]f32, eps: f32) void {
    const hc = hc_width;

    for (0..hc) |idst| {
        var sum: f32 = eps;
        for (0..hc) |isrc| sum += comb[idst + hc * isrc];

        const inv_sum = 1.0 / sum;
        for (0..hc) |isrc| comb[idst + hc * isrc] *= inv_sum;
    }
}

/// Ports `ggml_dsv4_hc_comb_norm_rows` (ops.cpp:10983 @c1d0e7a00).
fn dsv4HcCombNormRows(comb: *[hc_width * hc_width]f32, eps: f32) void {
    const hc = hc_width;

    for (0..hc) |isrc| {
        var sum: f32 = eps;
        for (0..hc) |idst| sum += comb[idst + hc * isrc];

        const inv_sum = 1.0 / sum;
        for (0..hc) |idst| comb[idst + hc * isrc] *= inv_sum;
    }
}

/// Ports `ggml_compute_forward_dsv4_hc_comb_f32` (ops.cpp:10999 @c1d0e7a00).
///
/// Two fusions: `xv * scale_comb + bv` and `comb * inv_sum + eps`, each one
/// expression in the C.
fn dsv4HcCombF32(params: *const ComputeParams, dst: *Tensor) void {
    const mixes = impl.one(Tensor, dst.src[0]);
    const scale = impl.one(Tensor, dst.src[1]);
    const base = impl.one(Tensor, dst.src[2]);

    impl.assert(mixes.type == c.GGML_TYPE_F32, "mixes->type == GGML_TYPE_F32");
    impl.assert(scale.type == c.GGML_TYPE_F32, "scale->type == GGML_TYPE_F32");
    impl.assert(base.type == c.GGML_TYPE_F32, "base->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const hc = hc_width;
    const comb_offset = 2 * hc;
    const hc_mix_dim = (2 + hc) * hc;

    const n_tokens = mixes.ne[1];

    impl.assert(mixes.ne[0] == hc_mix_dim, "mixes->ne[0] == hc_mix_dim");
    impl.assert(dst.ne[0] == hc, "dst->ne[0] == hc");
    impl.assert(dst.ne[1] == hc, "dst->ne[1] == hc");
    impl.assert(dst.ne[2] == n_tokens, "dst->ne[2] == n_tokens");
    impl.assert(scale.ne[0] >= 3, "scale->ne[0] >= 3");
    impl.assert(base.ne[0] == hc_mix_dim, "base->ne[0] == hc_mix_dim");

    const nbm0 = mixes.nb[0];
    const nbm1 = mixes.nb[1];
    const nbs0 = scale.nb[0];
    const nbb0 = base.nb[0];
    const nbd0 = dst.nb[0];
    const nbd1 = dst.nb[1];
    const nbd2 = dst.nb[2];

    const eps = impl.getOpParamsF32(dst, 0);
    const n_iter = impl.getOpParamsI32(dst, 1);
    impl.assert(n_iter > 0, "n_iter > 0");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const dr = @divTrunc(n_tokens + nth - 1, nth);
    const it0 = dr * ith;
    const it1 = @min(it0 + dr, n_tokens);

    const scale_comb = at(f32, scale.data, 2 * nbs0).*;

    var it: i64 = it0;
    while (it < it1) : (it += 1) {
        const itu: usize = @intCast(it);
        var comb: [hc * hc]f32 = undefined;

        for (0..hc) |isrc| {
            var max: f32 = -std.math.inf(f32);
            for (0..hc) |idst| {
                const idx = idst + hc * isrc;
                const xv = at(f32, mixes.data, (comb_offset + idx) * nbm0 + itu * nbm1).*;
                const bv = at(f32, base.data, (comb_offset + idx) * nbb0).*;
                // xv * scale_comb + bv, fused
                const v = @mulAdd(f32, xv, scale_comb, bv);
                comb[idx] = v;
                // MAX(max, v): `max > v ? max : v`, which keeps `v` when
                // either is NaN -- not `@max`.
                max = if (max > v) max else v;
            }

            var sum: f32 = 0.0;
            for (0..hc) |idst| {
                const idx = idst + hc * isrc;
                const v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            const inv_sum = 1.0 / sum;
            for (0..hc) |idst| {
                const idx = idst + hc * isrc;
                // comb * inv_sum + eps, fused
                comb[idx] = @mulAdd(f32, comb[idx], inv_sum, eps);
            }
        }

        dsv4HcCombNormCols(&comb, eps);
        var i: i32 = 1;
        while (i < n_iter) : (i += 1) {
            dsv4HcCombNormRows(&comb, eps);
            dsv4HcCombNormCols(&comb, eps);
        }

        for (0..hc) |isrc| {
            for (0..hc) |idst| {
                const idx = idst + hc * isrc;
                at(f32, dst.data, idst * nbd0 + isrc * nbd1 + itu * nbd2).* = comb[idx];
            }
        }
    }
}

/// Ports `ggml_compute_forward_dsv4_hc_comb` (ops.cpp:11086 @c1d0e7a00).
pub export fn ggml_compute_forward_dsv4_hc_comb(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => dsv4HcCombF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_dsv4_hc_pre_f32` (ops.cpp:11105 @c1d0e7a00).
///
/// `sum += xv * wv` is fused, one rounding per step.
fn dsv4HcPreF32(params: *const ComputeParams, dst: *Tensor) void {
    const x = impl.one(Tensor, dst.src[0]);
    const weights = impl.one(Tensor, dst.src[1]);

    impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    impl.assert(weights.type == c.GGML_TYPE_F32, "weights->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const n_embd = x.ne[0];
    const hc = x.ne[1];
    const n_tokens = x.ne[2];

    impl.assert(dst.ne[0] == n_embd, "dst->ne[0] == n_embd");
    impl.assert(dst.ne[1] == n_tokens, "dst->ne[1] == n_tokens");
    impl.assert(weights.ne[0] == hc, "weights->ne[0] == hc");
    impl.assert(weights.ne[1] == n_tokens, "weights->ne[1] == n_tokens");

    const nbx0 = x.nb[0];
    const nbx1 = x.nb[1];
    const nbx2 = x.nb[2];
    const nbw0 = weights.nb[0];
    const nbw1 = weights.nb[1];
    const nbd0 = dst.nb[0];
    const nbd1 = dst.nb[1];

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr = n_embd * n_tokens;
    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j0: usize = @intCast(@rem(ir, n_embd));
        const it: usize = @intCast(@divTrunc(ir, n_embd));

        var sum: f32 = 0.0;
        var ih: usize = 0;
        while (ih < hc) : (ih += 1) {
            const xv = at(f32, x.data, j0 * nbx0 + ih * nbx1 + it * nbx2).*;
            const wv = at(f32, weights.data, ih * nbw0 + it * nbw1).*;
            // sum += xv * wv, fused
            sum = @mulAdd(f32, xv, wv, sum);
        }

        at(f32, dst.data, j0 * nbd0 + it * nbd1).* = sum;
    }
}

/// Ports `ggml_compute_forward_dsv4_hc_pre` (ops.cpp:11151 @c1d0e7a00).
pub export fn ggml_compute_forward_dsv4_hc_pre(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => dsv4HcPreF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_dsv4_hc_post_f32` (ops.cpp:11170 @c1d0e7a00).
///
/// `sum = xv * pv` rounds on its own; each `sum += rv * cv` is fused.
fn dsv4HcPostF32(params: *const ComputeParams, dst: *Tensor) void {
    const x = impl.one(Tensor, dst.src[0]);
    const residual = impl.one(Tensor, dst.src[1]);
    const post = impl.one(Tensor, dst.src[2]);
    const comb = impl.one(Tensor, dst.src[3]);

    impl.assert(x.type == c.GGML_TYPE_F32, "x->type == GGML_TYPE_F32");
    impl.assert(residual.type == c.GGML_TYPE_F32, "residual->type == GGML_TYPE_F32");
    impl.assert(post.type == c.GGML_TYPE_F32, "post->type == GGML_TYPE_F32");
    impl.assert(comb.type == c.GGML_TYPE_F32, "comb->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const n_embd = x.ne[0];
    const n_tokens = x.ne[1];
    const hc = residual.ne[1];

    impl.assert(dst.ne[0] == n_embd, "dst->ne[0] == n_embd");
    impl.assert(dst.ne[1] == hc, "dst->ne[1] == hc");
    impl.assert(dst.ne[2] == n_tokens, "dst->ne[2] == n_tokens");
    impl.assert(residual.ne[0] == n_embd, "residual->ne[0] == n_embd");
    impl.assert(residual.ne[2] == n_tokens, "residual->ne[2] == n_tokens");
    impl.assert(post.ne[0] == hc, "post->ne[0] == hc");
    impl.assert(post.ne[1] == n_tokens, "post->ne[1] == n_tokens");
    impl.assert(comb.ne[0] == hc, "comb->ne[0] == hc");
    impl.assert(comb.ne[1] == hc, "comb->ne[1] == hc");
    impl.assert(comb.ne[2] == n_tokens, "comb->ne[2] == n_tokens");

    const nbx0 = x.nb[0];
    const nbx1 = x.nb[1];
    const nbr0 = residual.nb[0];
    const nbr1 = residual.nb[1];
    const nbr2 = residual.nb[2];
    const nbp0 = post.nb[0];
    const nbp1 = post.nb[1];
    const nbc0 = comb.nb[0];
    const nbc1 = comb.nb[1];
    const nbc2 = comb.nb[2];
    const nbd0 = dst.nb[0];
    const nbd1 = dst.nb[1];
    const nbd2 = dst.nb[2];

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr = n_embd * hc * n_tokens;
    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j0: usize = @intCast(@rem(ir, n_embd));
        const idst: usize = @intCast(@rem(@divTrunc(ir, n_embd), hc));
        const it: usize = @intCast(@divTrunc(ir, n_embd * hc));

        const xv = at(f32, x.data, j0 * nbx0 + it * nbx1).*;
        const pv = at(f32, post.data, idst * nbp0 + it * nbp1).*;

        var sum = xv * pv;
        var isrc: usize = 0;
        while (isrc < hc) : (isrc += 1) {
            const rv = at(f32, residual.data, j0 * nbr0 + isrc * nbr1 + it * nbr2).*;
            const cv = at(f32, comb.data, idst * nbc0 + isrc * nbc1 + it * nbc2).*;
            // sum += rv * cv, fused
            sum = @mulAdd(f32, rv, cv, sum);
        }

        at(f32, dst.data, j0 * nbd0 + idst * nbd1 + it * nbd2).* = sum;
    }
}

/// Ports `ggml_compute_forward_dsv4_hc_post` (ops.cpp:11232 @c1d0e7a00).
pub export fn ggml_compute_forward_dsv4_hc_post(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => dsv4HcPostF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// `GGML_F32_STEP` and `GGML_F32_EPR` (simd-mappings.h:337, 338 @c1d0e7a00),
/// the NEON arm: four `float32x4_t` per step.
const f32_step: usize = 16;
const f32_epr: usize = 4;

/// Ports the NEON `GGML_F32x4_REDUCE` (simd-mappings.h:349 @c1d0e7a00): halve,
/// halve, then one **pairwise** `vaddvq_f32`. The C widens the result to
/// `ggml_float` and every caller here narrows it straight back to `float`,
/// which is lossless, so the round trip is left out.
inline fn f32VecReduce(x: *[4]f32x4) f32 {
    x[0] = x[0] + x[2];
    x[1] = x[1] + x[3];
    x[0] = x[0] + x[1];
    return neon.addvq_f32(x[0]);
}

inline fn load4(p: [*]const f32) f32x4 {
    return p[0..4].*;
}

/// Ports `ggml_compute_forward_rwkv_wkv7_f32` (ops.cpp:11251 @c1d0e7a00), the
/// `GGML_SIMD` arm without SVE or RVV — the NEON one.
///
/// Two reductions over `j` per `(t, h, ii)`: `sa = a · state_prev[ii]` and
/// the output `Σ state_cur * r`. Both accumulate into four `float32x4_t`
/// with `vfmaq_f32` over steps of 16 and reduce through `f32VecReduce`, so
/// both are ported vector for vector — an elementwise rewrite would change
/// the summation order.
///
/// The state update per lane is `state = kv + state * w` then
/// `state = state + sa * b`, each fused, `kv = v * k` rounded on its own.
///
/// The step loops run `j` in strides of 16 to *past* `head_size` and load
/// whole vectors, so the C assumes `head_size % 16 == 0` (RWKV7 uses 64). The
/// remainder loop after them therefore never runs; it is kept, as in the C,
/// fused as the vector body is.
fn rwkvWkv7F32(params: *const ComputeParams, dst: *Tensor) void {
    const src1 = impl.one(Tensor, dst.src[1]);
    const src6 = impl.one(Tensor, dst.src[6]);

    const T = src1.ne[2];
    const C = dst.ne[0];
    const HEADS = src1.ne[1];
    const n_seqs = src6.ne[1];
    const head_size = @divTrunc(C, HEADS);

    const dst_data = f32Data(dst);
    const state = dst_data + @as(usize, @intCast(C * T));

    const h_start, const h_end = headRange(params, HEADS);

    const r = f32Data(impl.one(Tensor, dst.src[0]));
    const w = f32Data(src1);
    const k = f32Data(impl.one(Tensor, dst.src[2]));
    const v = f32Data(impl.one(Tensor, dst.src[3]));
    const a = f32Data(impl.one(Tensor, dst.src[4]));
    const b = f32Data(impl.one(Tensor, dst.src[5]));

    const t_stride = HEADS * head_size; // Same to C

    const h_stride = @divTrunc(C, HEADS);
    impl.assert(@rem(C, HEADS) == 0, "C % HEADS == 0"); // C must be divisible by HEADS
    const h_stride_2d = head_size * head_size;

    const seq_len = @divTrunc(T, n_seqs);
    const zero: f32x4 = @splat(0.0);

    var t: i64 = 0;
    while (t < T) : (t += 1) {
        const t_offset = t * t_stride;
        const state_offset: usize = @intCast(head_size * C * @divTrunc(t, seq_len));
        const state_cur = state + state_offset;
        const state_prev = if (@rem(t, seq_len) != 0) state_cur else f32Data(src6) + state_offset;

        var h: i64 = h_start;
        while (h < h_end) : (h += 1) {
            const h_offset = h * h_stride;
            const t_h_offset: usize = @intCast(t_offset + h_offset);
            const h_2d_offset = h * h_stride_2d;

            var ii: i64 = 0;
            while (ii < head_size) : (ii += 1) {
                const t_h_i_offset = t_h_offset + @as(usize, @intCast(ii));
                const h_2d_i_offset: usize = @intCast(h_2d_offset + ii * h_stride);

                const v_vec: f32x4 = @splat(v[t_h_i_offset]);

                var sa: f32 = 0;
                {
                    var sum: [4]f32x4 = .{zero} ** 4;
                    var j: usize = 0;
                    while (j < head_size) : (j += f32_step) {
                        for (0..4) |kk| {
                            const ax = load4(a + t_h_offset + j + kk * f32_epr);
                            const ay = load4(state_prev + h_2d_i_offset + j + kk * f32_epr);
                            sum[kk] = neon.fma_f32(sum[kk], ax, ay);
                        }
                    }
                    sa = f32VecReduce(&sum);
                }

                const sa_vec: f32x4 = @splat(sa);

                var j: usize = 0;
                var result_vec: [4]f32x4 = .{zero} ** 4;
                while (j < head_size) : (j += f32_step) {
                    for (0..4) |kk| {
                        const t_h_j_offset = t_h_offset + j + kk * f32_epr;
                        const h_2d_i_j_offset = h_2d_i_offset + j + kk * f32_epr;

                        const r_vec = load4(r + t_h_j_offset);
                        const w_vec = load4(w + t_h_j_offset);
                        var k_vec = load4(k + t_h_j_offset);
                        const b_vec = load4(b + t_h_j_offset);

                        k_vec = neon.mul_f32(v_vec, k_vec);

                        var state_vec = load4(state_prev + h_2d_i_j_offset);
                        // kv + s * decay + sa * b
                        state_vec = neon.fma_f32(k_vec, state_vec, w_vec);
                        state_vec = neon.fma_f32(state_vec, sa_vec, b_vec);
                        (state_cur + h_2d_i_j_offset)[0..4].* = state_vec;

                        result_vec[kk] = neon.fma_f32(result_vec[kk], state_vec, r_vec);
                    }
                }
                dst_data[t_h_i_offset] = f32VecReduce(&result_vec);

                // There shouldn't be left-overs though.
                while (j < head_size) : (j += 1) {
                    const t_h_j_offset = t_h_offset + j;
                    const h_2d_i_j_offset = h_2d_i_offset + j;

                    const r_val = r[t_h_j_offset];
                    const w_val = w[t_h_j_offset];
                    const k_val = k[t_h_j_offset];
                    const b_val = b[t_h_j_offset];
                    const kv_val = v[t_h_i_offset] * k_val;

                    const prev_state_val = state_prev[h_2d_i_j_offset];
                    // (prev_state * w + kv) + sa * b: each add fuses the
                    // product beside it.
                    state_cur[h_2d_i_j_offset] = @mulAdd(f32, sa, b_val, @mulAdd(f32, prev_state_val, w_val, kv_val));
                    // dst += state * r, fused
                    dst_data[t_h_i_offset] = @mulAdd(f32, state_cur[h_2d_i_j_offset], r_val, dst_data[t_h_i_offset]);
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_rwkv_wkv7` (ops.cpp:11448 @c1d0e7a00).
pub export fn ggml_compute_forward_rwkv_wkv7(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => rwkvWkv7F32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the halving reduce is pairwise at the last step" {
    // Pairwise: (1e8 + 1) + (-1e8 + 1); each inner sum rounds back to +-1e8,
    // so the result is 0. An ordered sum would give ((1e8 + 1) - 1e8) + 1 = 1.
    var x: [4]f32x4 = .{ .{ 1e8, 1, -1e8, 1 }, @splat(0), @splat(0), @splat(0) };
    try std.testing.expectEqual(@as(f32, 0), f32VecReduce(&x));
}
