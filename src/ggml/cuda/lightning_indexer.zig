//! `GGML_OP_LIGHTNING_INDEXER`: DeepSeek's sparse-attention indexer scores,
//! `sum_h w[h] * relu(q[h] . k)` plus the mask, per key.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/lightning-indexer.cu` at v0.7.0
//! (`620d2225c`), the host half. The kernels,
//! `lightning_indexer_kernel_wmma`, `lightning_indexer_kernel_vec` and
//! (v0.6.0, for four heads) `lightning_indexer_kernel_tile`, stay CUDA. The
//! HIP/MUSA arm, which always takes the vector kernel, is not compiled
//! here.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// The `LIGHTNING_INDEXER_CASE` instantiation for one kernel family, shape
/// and K type.
fn kernel(comptime family: []const u8, comptime warps_per_block: c_int, comptime k_vecs_per_block: c_int, comptime n_head: i64, comptime type_k: c.enum_ggml_type) kernels.Kernel {
    return kernels.get("lightning-indexer", kernels.tmpl("lightning_indexer_kernel_" ++ family, &.{
        kernels.int(warps_per_block), kernels.int(k_vecs_per_block), kernels.long(128), kernels.long(n_head), kernels.ggmlType(type_k),
    }));
}

const wmma_types = [_]c.enum_ggml_type{ c.GGML_TYPE_F16, c.GGML_TYPE_Q4_0, c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q8_0 };
const vec_types = wmma_types ++ [_]c.enum_ggml_type{ c.GGML_TYPE_BF16, c.GGML_TYPE_F32 };

/// One `if (K->type == type_K) … else` chain of `LIGHTNING_INDEXER_CASE`s
/// ending in `GGML_ABORT`.
fn launchCase(comptime family: []const u8, comptime k_vecs_per_block: c_int, comptime n_head: i64, comptime types: []const c.enum_ggml_type, k_type: c.enum_ggml_type, p: kernels.LaunchParams, args: anytype) void {
    inline for (types) |t| {
        if (k_type == t) {
            kernels.launch(kernel(family, 8, k_vecs_per_block, n_head, t), p, args);
            return;
        }
    }
    impl.abort("fatal error");
}

/// `n_head == 4` has no wmma instantiation (upstream's four-head arm names
/// only the tile and vec kernels), so its wmma branch must not be analysed.
fn shapeCase(comptime n_head: i64, use_wmma: bool, k_type: c.enum_ggml_type, n_kv: c_int, n_batch: c_int, n_stream: c_int, stream: common.cu.cudaStream_t, args: anytype) void {
    const warps_per_block = 8;
    if (n_head != 4 and use_wmma) {
        // use wmma kernel
        const k_vecs_per_block = 32;
        const num_kv_blocks = @divTrunc(n_kv + k_vecs_per_block - 1, k_vecs_per_block);
        const p: kernels.LaunchParams = .{
            .grid = .{ .x = @bitCast(num_kv_blocks), .y = @bitCast(n_batch), .z = @bitCast(n_stream) },
            .block = .{ .x = 32, .y = warps_per_block },
            .stream = stream,
        };
        launchCase("wmma", k_vecs_per_block, n_head, &wmma_types, k_type, p, args);
    } else {
        // use vector kernel
        const k_vecs_per_warp = 8;
        const k_vecs_per_block = k_vecs_per_warp * warps_per_block;
        const num_kv_blocks = @divTrunc(n_kv + k_vecs_per_block - 1, k_vecs_per_block);
        const p: kernels.LaunchParams = .{
            .grid = .{ .x = @bitCast(num_kv_blocks), .y = @bitCast(n_batch), .z = @bitCast(n_stream) },
            .block = .{ .x = 32, .y = warps_per_block },
            .stream = stream,
        };
        launchCase("vec", k_vecs_per_block, n_head, &vec_types, k_type, p, args);
    }
}

/// Ports `LIGHTNING_INDEXER_TILE_TOKENS` (lightning-indexer.cu:240 @620d2225c):
/// tokens scored per block by the tile kernel.
const tile_tokens = 8;

/// Ports `ggml_cuda_lightning_indexer` (lightning-indexer.cu:569 @620d2225c).
/// v0.7.0 dropped the tile arm's `#ifndef GGML_USE_MUSA` (MUSA now takes the tile
/// kernel too, staging fewer heads per pass); nothing changed on CUDA.
pub fn opLightningIndexer(ctx: *common.Context, dst: *Tensor) void {
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);
    const w = impl.one(Tensor, dst.src[2]); // weights
    const m = impl.one(Tensor, dst.src[3]); // mask

    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    impl.assert(q.type == c.GGML_TYPE_F32, "  q->type == GGML_TYPE_F32");
    impl.assert(w.type == c.GGML_TYPE_F32, "  w->type == GGML_TYPE_F32");
    impl.assert(m.type == c.GGML_TYPE_F16, "  m->type == GGML_TYPE_F16");

    // input tensor rows must be contiguous
    impl.assert(q.nb[0] == c.ggml_type_size(q.type), "nbq0 == ggml_type_size(q->type)");
    impl.assert(k.nb[0] == c.ggml_type_size(k.type), "nbk0 == ggml_type_size(k->type)");
    impl.assert(w.nb[0] == c.ggml_type_size(w.type), "nbw0 == ggml_type_size(w->type)");
    impl.assert(m.nb[0] == c.ggml_type_size(m.type), "nbm0 == ggml_type_size(m->type)");

    // dst cannot be transposed or permuted
    impl.assert(dst.nb[0] == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(dst.nb[0] <= dst.nb[1], "nb0 <= nb1");
    impl.assert(dst.nb[1] <= dst.nb[2], "nb1 <= nb2");
    impl.assert(dst.nb[2] <= dst.nb[3], "nb2 <= nb3");

    const n_embd: c_int = @truncate(q.ne[0]);
    const n_head: c_int = @truncate(q.ne[1]);
    const n_batch: c_int = @truncate(q.ne[2]);
    const n_stream: c_int = @truncate(q.ne[3]);
    const n_kv: c_int = @truncate(k.ne[2]);

    const q_d: ?*const f32 = @ptrCast(@alignCast(q.data));
    const k_d: ?*const u8 = @ptrCast(k.data);
    const w_d: ?*const f32 = @ptrCast(@alignCast(w.data));
    const m_d: ?*const anyopaque = m.data; // `const half *`
    const dst_d: ?*f32 = @ptrCast(@alignCast(dst.data));

    const device = common.getDevice();
    const x = common.info().devices[@intCast(device)].cc;

    // The kernel's `int64_t` counts are the `int` locals, widened.
    const args = .{
        q_d,                k_d,               w_d,            m_d,       dst_d,
        @as(i64, n_stream), @as(i64, n_batch), @as(i64, n_kv), dst.nb[1], dst.nb[2],
        dst.nb[3],          q.nb[1],           q.nb[2],        q.nb[3],   k.nb[1],
        k.nb[2],            k.nb[3],           w.nb[1],        w.nb[2],   w.nb[3],
        m.nb[1],            m.nb[2],           m.nb[3],        m.ne[3],
    };

    const use_wmma = common.cc.isNvidia(x) and common.turingMmaAvailable(x) and k.type != c.GGML_TYPE_F32 and k.type != c.GGML_TYPE_BF16;
    const stream = ctx.stream();

    if (n_embd == 128 and n_head == 64) {
        shapeCase(64, use_wmma, k.type, n_kv, n_batch, n_stream, stream, args);
    } else if (n_embd == 128 and n_head == 32) {
        shapeCase(32, use_wmma, k.type, n_kv, n_batch, n_stream, stream, args);
    } else if (n_embd == 128 and n_head == 4 and n_batch >= tile_tokens) {
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
        const warps_per_block = 8;
        const k_vecs_per_block = 64;

        const num_kv_blocks = @divTrunc(n_kv + k_vecs_per_block - 1, k_vecs_per_block);
        const num_batch_blocks = @divTrunc(n_batch + tile_tokens - 1, tile_tokens);
        const p: kernels.LaunchParams = .{
            .grid = .{ .x = @bitCast(num_kv_blocks), .y = @bitCast(num_batch_blocks), .z = @bitCast(n_stream) },
            .block = .{ .x = 32, .y = warps_per_block },
            .stream = stream,
        };
        launchCase("tile", k_vecs_per_block, 4, &vec_types, k.type, p, args);
    } else if (n_embd == 128 and n_head == 4) {
        // a batch smaller than a token tile, use vector kernel
        shapeCase(4, false, k.type, n_kv, n_batch, n_stream, stream, args);
    } else {
        impl.abort("fatal error");
    }
}

/// Ports `ggml_cuda_lightning_indexer_supported` (lightning-indexer.cu:743 @620d2225c).
pub fn supported(device: c_int, dst: *const Tensor) bool {
    _ = device;
    const q = impl.one(Tensor, dst.src[0]);
    const k = impl.one(Tensor, dst.src[1]);

    if (q.ne[0] != 128) return false;

    if (q.ne[1] != 64 and q.ne[1] != 32 and q.ne[1] != 4) return false;

    // alignment checks
    for ([_]*const Tensor{ q, k }) |t| {
        if (c.ggml_is_quantized(t.type)) continue;
        for (1..c.GGML_MAX_DIMS) |i| {
            if (t.nb[i] % 16 != 0) return false;
        }
    }

    return switch (k.type) {
        c.GGML_TYPE_F32, c.GGML_TYPE_BF16, c.GGML_TYPE_F16, c.GGML_TYPE_Q8_0, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q4_1, c.GGML_TYPE_Q4_0 => true,
        else => false,
    };
}
