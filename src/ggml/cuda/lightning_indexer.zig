//! `GGML_OP_LIGHTNING_INDEXER`: DeepSeek's sparse-attention indexer scores,
//! `sum_h w[h] * relu(q[h] . k)` plus the mask, per key.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/lightning-indexer.cu` at v1.0.0
//! (`b1b770d90`), the host half. The kernels,
//! `lightning_indexer_kernel_mma` (on
//! `mma.cuh` with four warps a block), `lightning_indexer_kernel_vec` and
//! (for four heads) `lightning_indexer_kernel_tile`, stay CUDA. The
//! HIP/MUSA arms (AMD MFMA and WMMA) are not compiled here.

const ggml = @import("../module.zig");
/// The `LIGHTNING_INDEXER_CASE` instantiation for one kernel family, shape
/// and K type.
fn kernel(comptime family: []const u8, comptime warps_per_block: c_int, comptime k_vecs_per_block: c_int, comptime n_head: i64, comptime type_k: ggml.impl.c.enum_ggml_type) ggml.cuda.kernels.Kernel {
    return ggml.cuda.kernels.get("lightning-indexer", ggml.cuda.kernels.tmpl("lightning_indexer_kernel_" ++ family, &.{
        ggml.cuda.kernels.int(warps_per_block), ggml.cuda.kernels.int(k_vecs_per_block), ggml.cuda.kernels.long(128), ggml.cuda.kernels.long(n_head), ggml.cuda.kernels.ggmlType(type_k),
    }));
}

const mma_types = [_]ggml.impl.c.enum_ggml_type{ ggml.impl.c.GGML_TYPE_F16, ggml.impl.c.GGML_TYPE_Q4_0, ggml.impl.c.GGML_TYPE_Q4_1, ggml.impl.c.GGML_TYPE_Q5_0, ggml.impl.c.GGML_TYPE_Q5_1, ggml.impl.c.GGML_TYPE_Q8_0 };
const vec_types = mma_types ++ [_]ggml.impl.c.enum_ggml_type{ ggml.impl.c.GGML_TYPE_BF16, ggml.impl.c.GGML_TYPE_F32 };

/// One `if (K->type == type_K) … else` chain of `LIGHTNING_INDEXER_CASE`s
/// ending in `GGML_ABORT`.
fn launchCase(comptime family: []const u8, comptime warps_per_block: c_int, comptime k_vecs_per_block: c_int, comptime n_head: i64, comptime types: []const ggml.impl.c.enum_ggml_type, k_type: ggml.impl.c.enum_ggml_type, p: ggml.cuda.kernels.LaunchParams, args: anytype) void {
    inline for (types) |t| {
        if (k_type == t) {
            ggml.cuda.kernels.launch(kernel(family, warps_per_block, k_vecs_per_block, n_head, t), p, args);
            return;
        }
    }
    ggml.impl.abort("fatal error");
}

/// `n_head == 4` has no mma instantiation (upstream's four-head arm names
/// only the tile and vec kernels), so its mma branch must not be analysed.
/// The mma kernel runs four warps of the device's warp size.
fn shapeCase(comptime n_head: i64, use_mma: bool, k_type: ggml.impl.c.enum_ggml_type, n_kv: c_int, n_batch: c_int, n_stream: c_int, warp_size: c_int, stream: ggml.cuda.common.cu.cudaStream_t, args: anytype) void {
    if (n_head != 4 and use_mma) {
        const k_vecs_per_block = 32;
        const warps_per_block = 4;
        const num_kv_blocks = @divTrunc(n_kv + k_vecs_per_block - 1, k_vecs_per_block);
        const p: ggml.cuda.kernels.LaunchParams = .{
            .grid = .{ .x = @bitCast(num_kv_blocks), .y = @bitCast(n_batch), .z = @bitCast(n_stream) },
            .block = .{ .x = @bitCast(warp_size), .y = warps_per_block },
            .stream = stream,
        };
        launchCase("mma", warps_per_block, k_vecs_per_block, n_head, &mma_types, k_type, p, args);
    } else {
        // use vector kernel
        const k_vecs_per_warp = 8;
        const warps_per_block = 8;
        const k_vecs_per_block = k_vecs_per_warp * warps_per_block;
        const num_kv_blocks = @divTrunc(n_kv + k_vecs_per_block - 1, k_vecs_per_block);
        const p: ggml.cuda.kernels.LaunchParams = .{
            .grid = .{ .x = @bitCast(num_kv_blocks), .y = @bitCast(n_batch), .z = @bitCast(n_stream) },
            .block = .{ .x = 32, .y = warps_per_block },
            .stream = stream,
        };
        launchCase("vec", warps_per_block, k_vecs_per_block, n_head, &vec_types, k_type, p, args);
    }
}

/// Ports `LIGHTNING_INDEXER_TILE_TOKENS` (lightning-indexer.cu:8 @b1b770d90):
/// tokens scored per block by the tile kernel.
const tile_tokens = 8;

/// Ports `ggml_cuda_lightning_indexer` (lightning-indexer.cu:504 @b1b770d90).
/// It asks `turing_mma_available` alone on NVIDIA (the AMD MFMA and WMMA
/// arms join it on HIP).
pub fn opLightningIndexer(ctx: *ggml.cuda.common.Context, dst: *ggml.impl.c.ggml_tensor) void {
    const q = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const k = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);
    const w = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[2]); // weights
    const m = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[3]); // mask

    ggml.impl.assert(dst.type == ggml.impl.c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    ggml.impl.assert(q.type == ggml.impl.c.GGML_TYPE_F32, "  q->type == GGML_TYPE_F32");
    ggml.impl.assert(w.type == ggml.impl.c.GGML_TYPE_F32, "  w->type == GGML_TYPE_F32");
    ggml.impl.assert(m.type == ggml.impl.c.GGML_TYPE_F16, "  m->type == GGML_TYPE_F16");

    // input tensor rows must be contiguous
    ggml.impl.assert(q.nb[0] == ggml.impl.c.ggml_type_size(q.type), "nbq0 == ggml_type_size(q->type)");
    ggml.impl.assert(k.nb[0] == ggml.impl.c.ggml_type_size(k.type), "nbk0 == ggml_type_size(k->type)");
    ggml.impl.assert(w.nb[0] == ggml.impl.c.ggml_type_size(w.type), "nbw0 == ggml_type_size(w->type)");
    ggml.impl.assert(m.nb[0] == ggml.impl.c.ggml_type_size(m.type), "nbm0 == ggml_type_size(m->type)");

    // dst cannot be transposed or permuted
    ggml.impl.assert(dst.nb[0] == @sizeOf(f32), "nb0 == sizeof(float)");
    ggml.impl.assert(dst.nb[0] <= dst.nb[1], "nb0 <= nb1");
    ggml.impl.assert(dst.nb[1] <= dst.nb[2], "nb1 <= nb2");
    ggml.impl.assert(dst.nb[2] <= dst.nb[3], "nb2 <= nb3");

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

    const device = ggml.cuda.common.getDevice();
    const x = ggml.cuda.common.info().devices[@intCast(device)].cc;

    // The kernel's `int64_t` counts are the `int` locals, widened.
    const args = .{
        q_d,                k_d,               w_d,            m_d,       dst_d,
        @as(i64, n_stream), @as(i64, n_batch), @as(i64, n_kv), dst.nb[1], dst.nb[2],
        dst.nb[3],          q.nb[1],           q.nb[2],        q.nb[3],   k.nb[1],
        k.nb[2],            k.nb[3],           w.nb[1],        w.nb[2],   w.nb[3],
        m.nb[1],            m.nb[2],           m.nb[3],        m.ne[3],
    };

    const use_mma = ggml.cuda.common.turingMmaAvailable(x) and k.type != ggml.impl.c.GGML_TYPE_F32 and k.type != ggml.impl.c.GGML_TYPE_BF16;
    const warp_size = ggml.cuda.common.info().devices[@intCast(device)].warp_size;
    const stream = ctx.stream();

    if (n_embd == 128 and n_head == 64) {
        shapeCase(64, use_mma, k.type, n_kv, n_batch, n_stream, warp_size, stream, args);
    } else if (n_embd == 128 and n_head == 32) {
        shapeCase(32, use_mma, k.type, n_kv, n_batch, n_stream, warp_size, stream, args);
    } else if (n_embd == 128 and n_head == 4 and n_batch >= tile_tokens) {
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
        const warps_per_block = 8;
        const k_vecs_per_block = 64;

        const num_kv_blocks = @divTrunc(n_kv + k_vecs_per_block - 1, k_vecs_per_block);
        const num_batch_blocks = @divTrunc(n_batch + tile_tokens - 1, tile_tokens);
        const p: ggml.cuda.kernels.LaunchParams = .{
            .grid = .{ .x = @bitCast(num_kv_blocks), .y = @bitCast(num_batch_blocks), .z = @bitCast(n_stream) },
            .block = .{ .x = 32, .y = warps_per_block },
            .stream = stream,
        };
        launchCase("tile", warps_per_block, k_vecs_per_block, 4, &vec_types, k.type, p, args);
    } else if (n_embd == 128 and n_head == 4) {
        // a batch smaller than a token tile, use vector kernel
        shapeCase(4, false, k.type, n_kv, n_batch, n_stream, warp_size, stream, args);
    } else {
        ggml.impl.abort("fatal error");
    }
}

/// Ports `ggml_cuda_lightning_indexer_supported` (lightning-indexer.cu:668 @b1b770d90).
pub fn supported(device: c_int, dst: *const ggml.impl.c.ggml_tensor) bool {
    _ = device;
    const q = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[0]);
    const k = ggml.impl.one(ggml.impl.c.ggml_tensor, dst.src[1]);

    if (q.ne[0] != 128) return false;

    if (q.ne[1] != 64 and q.ne[1] != 32 and q.ne[1] != 4) return false;

    // alignment checks
    for ([_]*const ggml.impl.c.ggml_tensor{ q, k }) |t| {
        if (ggml.impl.c.ggml_is_quantized(t.type)) continue;
        for (1..ggml.impl.c.GGML_MAX_DIMS) |i| {
            if (t.nb[i] % 16 != 0) return false;
        }
    }

    return switch (k.type) {
        ggml.impl.c.GGML_TYPE_F32, ggml.impl.c.GGML_TYPE_BF16, ggml.impl.c.GGML_TYPE_F16, ggml.impl.c.GGML_TYPE_Q8_0, ggml.impl.c.GGML_TYPE_Q5_1, ggml.impl.c.GGML_TYPE_Q5_0, ggml.impl.c.GGML_TYPE_Q4_1, ggml.impl.c.GGML_TYPE_Q4_0 => true,
        else => false,
    };
}
