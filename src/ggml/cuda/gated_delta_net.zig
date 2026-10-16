//! `GGML_OP_GATED_DELTA_NET`: the gated delta rule recurrence of Qwen3.5's
//! linear-attention layers (and Kimi's KDA variant, a per-channel gate),
//! plus the entry point the compute loop's fusion uses to write the
//! recurrent-state snapshots straight into the cache.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cuh` at v0.3.0
//! (`c1d0e7a00`). The kernel, `gated_delta_net_cuda`, stays CUDA.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// Ports `ggml_cuda_gated_delta_net_fused_cache` (gated_delta_net.cuh:5
/// @c1d0e7a00): the fused kernel's recurrent-state output. Strides in
/// elements; the per-sequence stride is always `D`, set in-kernel.
pub const FusedCache = struct {
    /// Rollback slot 0.
    data: ?*f32,
    /// Between rollback slots (0 when K == 1).
    slot_stride: i64,
};

/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `launch_gated_delta_net` (gated_delta_net.cu:170 @c1d0e7a00).
fn launchGatedDeltaNet(
    comptime kda: bool,
    comptime keep_rs_t: bool,
    q_d: ?*const anyopaque,
    k_d: ?*const anyopaque,
    v_d: ?*const anyopaque,
    g_d: ?*const anyopaque,
    b_d: ?*const anyopaque,
    s_d: ?*const anyopaque,
    dst_d: ?*anyopaque,
    state_d: ?*anyopaque,
    S_v: i64,
    H: i64,
    n_tokens: i64,
    n_seqs: i64,
    sq1: i64,
    sq2: i64,
    sq3: i64,
    sv1: i64,
    sv2: i64,
    sv3: i64,
    sb1: i64,
    sb2: i64,
    sb3: i64,
    neqk1: i64,
    rq3: i64,
    scale: f32,
    state_slot_stride: i64,
    K: c_int,
    stream: cu.cudaStream_t,
) void {
    //TODO: Add chunked kernel for even faster pre-fill
    const warp_size: c_int = common.info().devices[@intCast(common.getDevice())].warp_size;
    const num_warps = 4;
    const grid_dims: kernels.Dim3 = .{ .x = dim(H), .y = dim(n_seqs), .z = dim(@divTrunc(S_v + num_warps - 1, num_warps)) };
    const block_dims: kernels.Dim3 = .{ .x = if (warp_size <= S_v) @bitCast(warp_size) else dim(S_v), .y = num_warps, .z = 1 };

    const neqk1_magic = common.initFastdivValues(@bitCast(neqk1));
    const rq3_magic = common.initFastdivValues(@bitCast(rq3));

    const params: kernels.LaunchParams = .{ .grid = grid_dims, .block = block_dims, .shmem = 0, .stream = stream };
    const args = .{
        q_d,      k_d,    v_d,         g_d,       b_d,   s_d,               dst_d, state_d, H,
        n_tokens, n_seqs, sq1,         sq2,       sq3,   sv1,               sv2,   sv3,     sb1,
        sb2,      sb3,    neqk1_magic, rq3_magic, scale, state_slot_stride, K,
    };
    switch (S_v) {
        inline 16, 32, 64, 128 => |s| kernels.launchPdl(kernels.get("gated_delta_net", kernels.tmpl("gated_delta_net_cuda", &.{ kernels.int(s), kernels.boolean(kda), kernels.boolean(keep_rs_t) })), params, args),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_cuda_op_gated_delta_net_impl` (gated_delta_net.cu:223
/// @c1d0e7a00).
fn opGatedDeltaNetImpl(ctx: *common.Context, dst: *Tensor, cache: ?*const FusedCache) void {
    const src_q = impl.one(Tensor, dst.src[0]);
    const src_k = impl.one(Tensor, dst.src[1]);
    const src_v = impl.one(Tensor, dst.src[2]);
    const src_g = impl.one(Tensor, dst.src[3]);
    const src_beta = impl.one(Tensor, dst.src[4]);
    const src_state = impl.one(Tensor, dst.src[5]);

    const neq1 = src_q.ne[1];
    const neq3 = src_q.ne[3];
    const nbq1 = src_q.nb[1];
    const nbq2 = src_q.nb[2];
    const nbq3 = src_q.nb[3];
    const nek1 = src_k.ne[1];
    const nev0 = src_v.ne[0];
    const nev1 = src_v.ne[1];
    const nev2 = src_v.ne[2];
    const nev3 = src_v.ne[3];
    const nbv1 = src_v.nb[1];
    const nbv2 = src_v.nb[2];
    const nbv3 = src_v.nb[3];
    const nbb1 = src_beta.nb[1];
    const nbb2 = src_beta.nb[2];
    const nbb3 = src_beta.nb[3];

    const S_v = nev0;
    const H = nev1;
    const n_tokens = nev2;
    const n_seqs = nev3;

    const kda = src_g.ne[0] == S_v;

    impl.assert(neq1 == nek1, "neq1 == nek1");
    const neqk1 = neq1;

    const rq3 = @divTrunc(nev3, neq3);

    const q_d: ?*const anyopaque = src_q.data;
    const k_d: ?*const anyopaque = src_k.data;
    const v_d: ?*const anyopaque = src_v.data;
    const g_d: ?*const anyopaque = src_g.data;
    const b_d: ?*const anyopaque = src_beta.data;

    const s_d: ?*const anyopaque = src_state.data;
    const dst_d: [*]f32 = @ptrCast(@alignCast(dst.data));

    impl.assert(c.ggml_is_contiguous_rows(src_q), "ggml_is_contiguous_rows(src_q)");
    impl.assert(c.ggml_is_contiguous_rows(src_k), "ggml_is_contiguous_rows(src_k)");
    impl.assert(c.ggml_is_contiguous_rows(src_v), "ggml_is_contiguous_rows(src_v)");
    impl.assert(c.ggml_are_same_stride(src_q, src_k), "ggml_are_same_stride(src_q, src_k)");
    impl.assert(src_g.ne[0] == 1 or kda, "src_g->ne[0] == 1 || kda");
    impl.assert(c.ggml_is_contiguous(src_g), "ggml_is_contiguous(src_g)");
    impl.assert(c.ggml_is_contiguous(src_beta), "ggml_is_contiguous(src_beta)");
    impl.assert(c.ggml_is_contiguous(src_state), "ggml_is_contiguous(src_state)");

    // strides in floats (beta strides used for both g and beta offset computation)
    const sq1: i64 = @intCast(nbq1 / @sizeOf(f32));
    const sq2: i64 = @intCast(nbq2 / @sizeOf(f32));
    const sq3: i64 = @intCast(nbq3 / @sizeOf(f32));
    const sv1: i64 = @intCast(nbv1 / @sizeOf(f32));
    const sv2: i64 = @intCast(nbv2 / @sizeOf(f32));
    const sv3: i64 = @intCast(nbv3 / @sizeOf(f32));
    const sb1: i64 = @intCast(nbb1 / @sizeOf(f32));
    const sb2: i64 = @intCast(nbb2 / @sizeOf(f32));
    const sb3: i64 = @intCast(nbb3 / @sizeOf(f32));

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(S_v)));

    const stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const K = impl.getOpParamsI32(dst, 0);
    const keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    var state_d: ?*anyopaque = dst_d + @as(usize, @intCast(S_v * H * n_tokens * n_seqs));
    var state_slot_stride: i64 = S_v * S_v * H * n_seqs;
    if (cache) |fc| {
        state_d = fc.data;
        state_slot_stride = fc.slot_stride;
    }

    const args = .{ q_d, k_d, v_d, g_d, b_d, s_d, @as(?*anyopaque, dst_d), state_d, S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream };
    if (kda) {
        if (keep_rs) {
            @call(.auto, launchGatedDeltaNet, .{ true, true } ++ args);
        } else {
            @call(.auto, launchGatedDeltaNet, .{ true, false } ++ args);
        }
    } else {
        if (keep_rs) {
            @call(.auto, launchGatedDeltaNet, .{ false, true } ++ args);
        } else {
            @call(.auto, launchGatedDeltaNet, .{ false, false } ++ args);
        }
    }
}

/// Ports `ggml_cuda_op_gated_delta_net` (gated_delta_net.cu:320 @c1d0e7a00).
pub fn opGatedDeltaNet(ctx: *common.Context, dst: *Tensor) void {
    opGatedDeltaNetImpl(ctx, dst, null);
}

/// Ports `ggml_cuda_op_gated_delta_net_fused_cache` (gated_delta_net.cu:324
/// @c1d0e7a00): the same op, writing the snapshot(s) into the cache
/// instead of `dst`.
pub fn opGatedDeltaNetFusedCache(ctx: *common.Context, dst: *Tensor, cache: FusedCache) void {
    opGatedDeltaNetImpl(ctx, dst, &cache);
}
