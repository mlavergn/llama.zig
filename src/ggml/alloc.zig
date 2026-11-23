//! Tensor and graph memory allocation for ggml.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-alloc.c` (v1.0.0, `b1b770d90`),
//! 1,249 lines of C. Each function below names the C function it replaces and
//! the line it began at, so behaviour can be cross-referenced against the
//! original. This file exports the same C symbols with the same signatures, so
//! the C++ that calls into it links unchanged.
//!
//! # Structure
//!
//! Three layers, in the order the C file defines them:
//!
//! 1. `Tallocr` — a bump allocator over one backend buffer.
//! 2. `DynTallocr` — a free-list allocator over chunks, used to plan a graph's
//!    memory before any buffer exists.
//! 3. `Gallocr` — the graph allocator, which walks a compute graph, works out
//!    when each tensor dies, and reuses the space.
//!
//! # Fidelity notes
//!
//! - Allocation uses libc `malloc`/`calloc`/`free` rather than a Zig allocator.
//!   These structures are handed across the C ABI and freed by paths that may
//!   still be C, so the allocator has to be the one C uses.
//! - The `GGML_ALLOCATOR_DEBUG` blocks in the original are compiled out and are
//!   not reproduced; where they had an effect on control flow, that is noted.
//! - Unsigned wrapping in `dynTallocrAlloc`'s reuse calculation is deliberate
//!   and is reproduced exactly. See the comment there.

const std = @import("std");
const mod = @import("module.zig");
const config = @import("config");
const stdlib = @import("stdlib");
const builtin = @import("builtin");
// The hash-set lifecycle used to be reached through `impl.zig` as an extern,
// because it was still C. It is ported now, so call it directly.

/// Build-time switch that makes this file abort on a hot path.
///
/// Off by default and compiled out entirely. See `probe` below.
/// The marker `scripts/probe-ported` looks for.
pub const probe_marker = "LLAMAZIG PORTED CODE REACHED";

/// Aborts if the probe is enabled, proving this file is on the execution path.
///
/// Output comparison cannot tell correct code from code that never runs. This
/// can: build with `-Dprobe-ported`, and a run that does *not* abort means the
/// ported implementation was bypassed, whatever the symbol tables say.
inline fn probe() void {
    if (config.probe_ported) @panic(probe_marker);
}

/// Ports `MAX_FREE_BLOCKS` (ggml-alloc.c:14 @b1b770d90).
const max_free_blocks = 256;

/// Ports `GGML_VBUFFER_MAX_CHUNKS` (ggml-alloc.c:93 @b1b770d90).
const max_chunks = 16;

/// Debug tracing in the original, compiled out by default: `AT_PRINTF`
/// (ggml-alloc.c:19 @b1b770d90).
/// Kept as a flag so the trace points stay visible in the ported code.
const allocator_debug = false;

/// True when the C build would have `NDEBUG` unset, gating its debug logging.
const debug_logging = builtin.mode == .debug;

// -----------------------------------------------------------------------------
// Op predicates

/// Ports `ggml_op_can_inplace` (ggml-alloc.c:22 @b1b770d90).
///
/// Ops listed here may write their output over an input's memory. Backends
/// implementing them must not use `restrict` pointers.
///
/// Parameters:
/// - `op`: the operation to classify.
///
/// Return: true when the op is safe to run in place.
pub fn ggml_op_can_inplace(op: mod.impl.ggml_c.enum_ggml_op) callconv(.c) bool {
    return switch (op) {
        mod.impl.ggml_c.GGML_OP_FILL,
        mod.impl.ggml_c.GGML_OP_SCALE,
        mod.impl.ggml_c.GGML_OP_ADD,
        mod.impl.ggml_c.GGML_OP_ADD_ID,
        mod.impl.ggml_c.GGML_OP_SUB,
        mod.impl.ggml_c.GGML_OP_MUL,
        mod.impl.ggml_c.GGML_OP_DIV,
        mod.impl.ggml_c.GGML_OP_SQR,
        mod.impl.ggml_c.GGML_OP_SQRT,
        mod.impl.ggml_c.GGML_OP_LOG,
        mod.impl.ggml_c.GGML_OP_UNARY,
        mod.impl.ggml_c.GGML_OP_ROPE,
        mod.impl.ggml_c.GGML_OP_ROPE_BACK,
        mod.impl.ggml_c.GGML_OP_SILU_BACK,
        mod.impl.ggml_c.GGML_OP_RMS_NORM,
        mod.impl.ggml_c.GGML_OP_RMS_NORM_BACK,
        mod.impl.ggml_c.GGML_OP_CLAMP,
        mod.impl.ggml_c.GGML_OP_SOFT_MAX,
        mod.impl.ggml_c.GGML_OP_SOFT_MAX_BACK,
        => true,
        else => false,
    };
}

/// Ports `aligned_offset` (ggml-alloc.c:50 @b1b770d90).
///
/// Parameters:
/// - `buffer`: base address the offset is relative to; null when aligning a
///   size rather than an address, which is how most callers use it.
/// - `offset`: the offset to align.
/// - `alignment`: must be a power of two.
///
/// Return: `offset` advanced to the next aligned position.
fn alignedOffset(buffer: ?*const anyopaque, offset: usize, alignment: usize) usize {
    std.debug.assert(alignment != 0 and (alignment & (alignment - 1)) == 0);
    const base = @intFromPtr(buffer);
    const alignment_gap = (alignment - ((base + offset) % alignment)) % alignment;
    return offset + alignment_gap;
}

// -----------------------------------------------------------------------------
// tallocr -- bump allocation within a single backend buffer

/// Ports `ggml_tallocr_new` (ggml-alloc.c:58 @b1b770d90).
pub fn ggml_tallocr_new(buffer: mod.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) mod.impl.ggml_c.struct_ggml_tallocr {
    const base = mod.backend.ggml_backend_buffer_get_base(buffer);
    const alignment = mod.backend.ggml_backend_buffer_get_alignment(buffer);

    std.debug.assert(alignment != 0 and (alignment & (alignment - 1)) == 0);

    return .{
        .buffer = buffer,
        .base = base,
        .alignment = alignment,
        .offset = alignedOffset(base, 0, alignment),
    };
}

/// Ports `ggml_tallocr_alloc` (ggml-alloc.c:73 @b1b770d90).
///
/// Aborts rather than returning an error when the buffer is exhausted, as the
/// C version does; callers are not written to recover from it.
pub fn ggml_tallocr_alloc(
    talloc: *mod.impl.ggml_c.struct_ggml_tallocr,
    tensor: *mod.impl.ggml_c.ggml_tensor,
) callconv(.c) mod.impl.ggml_c.enum_ggml_status {
    var size = mod.backend.ggml_backend_buffer_get_alloc_size(talloc.buffer, tensor);
    size = stdlib.num.pad(usize, size, talloc.alignment);

    if (talloc.offset + size > mod.backend.ggml_backend_buffer_get_size(talloc.buffer)) {
        mod.impl.logError(
            "%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
            .{
                "ggml_tallocr_alloc",
                &tensor.name,
                size,
                mod.backend.ggml_backend_buffer_get_size(talloc.buffer) - talloc.offset,
            },
        );
        mod.impl.abort("not enough space in the buffer");
    }

    const addr = @as([*]u8, @ptrCast(mod.backend.ggml_backend_buffer_get_base(talloc.buffer))) + talloc.offset;
    talloc.offset += size;

    std.debug.assert(@intFromPtr(addr) % talloc.alignment == 0);

    return mod.backend.ggml_backend_tensor_alloc(talloc.buffer, tensor, addr);
}

// -----------------------------------------------------------------------------
// dynamic tensor allocator

/// Ports `struct buffer_address` (ggml-alloc.c:96 @b1b770d90).
///
/// A memory address relative to an allocation that may be split across several
/// backend buffers, identified by chunk index plus offset within it.
const BufferAddress = extern struct {
    chunk: c_int,
    offset: usize,

    /// Ports `GGML_BUFFER_ADDRESS_INVALID` (ggml-alloc.c:101 @b1b770d90).
    const invalid: BufferAddress = .{ .chunk = -1, .offset = std.math.maxInt(usize) };

    /// Ports `ggml_buffer_address_less` (ggml-alloc.c:103 @b1b770d90).
    ///
    /// Only reachable from the compiled-out debug tracing in the original.
    fn less(a: BufferAddress, b: BufferAddress) bool {
        return if (a.chunk != b.chunk) a.chunk < b.chunk else a.offset < b.offset;
    }
};

/// Ports `struct free_block` (ggml-alloc.c:107 @b1b770d90).
const FreeBlock = extern struct {
    offset: usize,
    size: usize,
};

/// Ports `struct tallocr_chunk` (ggml-alloc.c:112 @b1b770d90).
const TallocrChunk = extern struct {
    free_blocks: [max_free_blocks]FreeBlock,
    n_free_blocks: c_int,
    max_size: usize,

    /// Ports `ggml_dyn_tallocr_insert_block` (ggml-alloc.c:132 @b1b770d90).
    ///
    /// Keeps `free_blocks` sorted by offset, which is what makes merging
    /// adjacent blocks in `freeBytes` a local check rather than a search.
    fn insertBlock(chunk: *TallocrChunk, offset: usize, size: usize) void {
        mod.impl.assert(chunk.n_free_blocks < max_free_blocks, "chunk->n_free_blocks < MAX_FREE_BLOCKS && \"out of free blocks\"");

        var insert_pos: usize = 0;
        while (insert_pos < @as(usize, @intCast(chunk.n_free_blocks)) and
            chunk.free_blocks[insert_pos].offset < offset) : (insert_pos += 1)
        {}

        var i: usize = @intCast(chunk.n_free_blocks);
        while (i > insert_pos) : (i -= 1) {
            chunk.free_blocks[i] = chunk.free_blocks[i - 1];
        }

        chunk.free_blocks[insert_pos] = .{ .offset = offset, .size = size };
        chunk.n_free_blocks += 1;
    }

    /// Ports `ggml_dyn_tallocr_remove_block` (ggml-alloc.c:149 @b1b770d90).
    fn removeBlock(chunk: *TallocrChunk, idx: usize) void {
        var i = idx;
        while (i + 1 < @as(usize, @intCast(chunk.n_free_blocks))) : (i += 1) {
            chunk.free_blocks[i] = chunk.free_blocks[i + 1];
        }
        chunk.n_free_blocks -= 1;
    }
};

/// Ports `struct ggml_dyn_tallocr` (ggml-alloc.c:118 @b1b770d90).
///
/// Plans allocations without owning memory: it decides offsets, and a
/// `VBuffer` later turns those into real backend buffers.
const DynTallocr = extern struct {
    alignment: usize,
    max_chunk_size: usize,
    chunks: [max_chunks]?*TallocrChunk,
    n_chunks: c_int,

    /// Ports `ggml_dyn_tallocr_new` (ggml-alloc.c:363 @b1b770d90).
    fn create(alignment: usize, max_buffer_size: usize) *DynTallocr {
        const alloc: *DynTallocr = @ptrCast(@alignCast(std.c.malloc(@sizeOf(DynTallocr)).?));
        alloc.* = .{
            .alignment = alignment,
            // Clamped to avoid overflow when a chunk's size is used in
            // arithmetic that can legitimately exceed it.
            .max_chunk_size = @min(max_buffer_size, std.math.maxInt(usize) / 2),
            .chunks = @splat(null),
            .n_chunks = 0,
        };
        alloc.reset();
        return alloc;
    }

    /// Ports `ggml_dyn_tallocr_free` (ggml-alloc.c:381 @b1b770d90).
    fn destroy(alloc: *DynTallocr) void {
        for (0..@intCast(alloc.n_chunks)) |i| std.c.free(alloc.chunks[i]);
        std.c.free(alloc);
    }

    /// Ports `ggml_dyn_tallocr_reset` (ggml-alloc.c:349 @b1b770d90).
    ///
    /// Frees every chunk slot, not just the live ones, matching the C loop.
    fn reset(alloc: *DynTallocr) void {
        for (0..max_chunks) |i| {
            std.c.free(alloc.chunks[i]);
            alloc.chunks[i] = null;
        }
        alloc.n_chunks = 0;
    }

    /// Ports `ggml_dyn_tallocr_new_chunk` (ggml-alloc.c:157 @b1b770d90).
    ///
    /// Return: the new chunk's index, or -1 when no slots remain.
    fn newChunk(alloc: *DynTallocr, min_size: usize) c_int {
        if (alloc.n_chunks >= max_chunks) return -1;

        const chunk: *TallocrChunk = @ptrCast(@alignCast(std.c.calloc(1, @sizeOf(TallocrChunk)).?));
        chunk.n_free_blocks = 1;
        chunk.free_blocks[0].offset = 0;
        // A chunk is normally capped at max_chunk_size, but may exceed it when
        // a single tensor does not fit otherwise, or when chunks are running
        // out. The backend either satisfies the larger request or reports it.
        chunk.free_blocks[0].size = @max(min_size, alloc.max_chunk_size);
        if (alloc.n_chunks == max_chunks - 1) {
            chunk.free_blocks[0].size = std.math.maxInt(usize) / 2;
        }

        alloc.chunks[@intCast(alloc.n_chunks)] = chunk;
        alloc.n_chunks += 1;
        return alloc.n_chunks - 1;
    }

    /// Ports `ggml_dyn_tallocr_alloc` (ggml-alloc.c:199 @b1b770d90).
    ///
    /// Best-fit across every chunk, falling back to the last block of a chunk
    /// (which may grow it), then to a new chunk.
    ///
    /// Parameters:
    /// - `alloc`: the allocator to take space from.
    /// - `size`: bytes needed, before alignment.
    /// - `tensor`: only used by the compiled-out debug tracing; kept so the
    ///   signature matches the original.
    ///
    /// Return: where the tensor should live. Aborts if even a fresh chunk
    /// cannot satisfy the request, which the last chunk's effectively
    /// unbounded size makes unreachable in practice.
    fn allocate(alloc: *DynTallocr, requested: usize, tensor: *const mod.impl.ggml_c.ggml_tensor) BufferAddress {
        _ = tensor;
        const size = alignedOffset(null, requested, alloc.alignment);

        var best_fit_chunk: c_int = -1;
        var best_fit_block: c_int = -1;
        var max_avail: usize = 0;

        // Best fit among all blocks except each chunk's last, which is handled
        // separately below because taking from it can grow the chunk.
        for (0..@intCast(alloc.n_chunks)) |chunk_index| {
            const chunk = alloc.chunks[chunk_index].?;
            var best_fit_size: usize = std.math.maxInt(usize);
            var i: usize = 0;
            while (i + 1 < @as(usize, @intCast(chunk.n_free_blocks))) : (i += 1) {
                const block = &chunk.free_blocks[i];
                max_avail = @max(max_avail, block.size);
                if (block.size >= size and block.size <= best_fit_size) {
                    best_fit_chunk = @intCast(chunk_index);
                    best_fit_block = @intCast(i);
                    best_fit_size = block.size;
                }
            }
        }

        if (best_fit_block == -1) {
            // Nothing fit, so consider each chunk's last block, which may grow.
            var best_reuse: i64 = std.math.minInt(i64);
            for (0..@intCast(alloc.n_chunks)) |chunk_index| {
                const chunk = alloc.chunks[chunk_index].?;
                if (chunk.n_free_blocks > 0) {
                    const block = &chunk.free_blocks[@intCast(chunk.n_free_blocks - 1)];
                    max_avail = @max(max_avail, block.size);

                    // The C computes this in size_t and assigns to int64_t, so
                    // a "negative" result arrives via unsigned wraparound. The
                    // wrapping subtraction and bitcast reproduce that exactly;
                    // computing in i64 would give different results when the
                    // intermediate exceeds i64 range.
                    const reuse_factor: i64 = @bitCast(chunk.max_size -% block.offset -% size);

                    // reuse_factor < 0: extra memory must be allocated
                    // reuse_factor = 0: the free space matches the tensor
                    // reuse_factor > 0: space will be left unused
                    const better_reuse = best_reuse < 0 and reuse_factor > best_reuse;
                    const better_fit = reuse_factor >= 0 and reuse_factor < best_reuse;
                    if (block.size >= size and (better_reuse or better_fit)) {
                        best_fit_chunk = @intCast(chunk_index);
                        best_fit_block = chunk.n_free_blocks - 1;
                        best_reuse = reuse_factor;
                    }
                }
            }
        }

        if (best_fit_block == -1) {
            // No existing chunk has room.
            best_fit_chunk = alloc.newChunk(size);
            best_fit_block = 0;
        }
        if (best_fit_chunk == -1) {
            // The last chunk has virtually endless memory, so this is
            // unreachable short of a bug.
            mod.impl.logError(
                "%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
                .{ "ggml_dyn_tallocr_alloc", size, max_avail },
            );
            mod.impl.abort("graph allocation: failed to reserve memory");
        }

        const chunk = alloc.chunks[@intCast(best_fit_chunk)].?;
        const block = &chunk.free_blocks[@intCast(best_fit_block)];
        const addr: BufferAddress = .{ .chunk = best_fit_chunk, .offset = block.offset };
        block.offset += size;
        block.size -= size;
        if (block.size == 0) {
            chunk.removeBlock(@intCast(best_fit_block));
        }

        chunk.max_size = @max(chunk.max_size, addr.offset + size);
        return addr;
    }

    /// Ports `ggml_dyn_tallocr_free_bytes` (ggml-alloc.c:309 @b1b770d90).
    ///
    /// Naive by design: the free-block count stays small enough that a linear
    /// scan with adjacent-block merging is the cheapest thing that works.
    fn freeBytes(alloc: *DynTallocr, addr: BufferAddress, requested: usize) void {
        const size = alignedOffset(null, requested, alloc.alignment);
        const chunk = alloc.chunks[@intCast(addr.chunk)].?;

        for (0..@intCast(chunk.n_free_blocks)) |i| {
            const block = &chunk.free_blocks[i];

            // The freed range sits immediately after this block.
            if (block.offset + block.size == addr.offset) {
                block.size += size;
                if (i + 1 < @as(usize, @intCast(chunk.n_free_blocks))) {
                    const next = &chunk.free_blocks[i + 1];
                    if (block.offset + block.size == next.offset) {
                        block.size += next.size;
                        chunk.removeBlock(i + 1);
                    }
                }
                return;
            }

            // The freed range sits immediately before this block.
            if (addr.offset + size == block.offset) {
                block.offset = addr.offset;
                block.size += size;
                if (i > 0) {
                    const prev = &chunk.free_blocks[i - 1];
                    if (prev.offset + prev.size == block.offset) {
                        prev.size += block.size;
                        chunk.removeBlock(i);
                    }
                }
                return;
            }
        }

        chunk.insertBlock(addr.offset, size);
    }

    /// Ports `ggml_dyn_tallocr_max_size` (ggml-alloc.c:388 @b1b770d90).
    fn maxSize(alloc: *const DynTallocr, chunk: c_int) usize {
        return if (chunk < alloc.n_chunks) alloc.chunks[@intCast(chunk)].?.max_size else 0;
    }
};

// -----------------------------------------------------------------------------
// vbuffer -- a contiguous logical range split across backend buffers

/// Ports `struct vbuffer` (ggml-alloc.c:395 @b1b770d90).
const VBuffer = extern struct {
    chunks: [max_chunks]mod.impl.ggml_c.ggml_backend_buffer_t,

    /// Ports `ggml_vbuffer_alloc` (ggml-alloc.c:421 @b1b770d90).
    ///
    /// Return: the buffer, or null if any chunk failed to allocate. On failure
    /// every chunk allocated so far is released.
    fn create(
        buft: mod.impl.ggml_c.ggml_backend_buffer_type_t,
        talloc: *const DynTallocr,
        usage: mod.impl.ggml_c.enum_ggml_backend_buffer_usage,
    ) ?*VBuffer {
        const buf: *VBuffer = @ptrCast(@alignCast(std.c.calloc(1, @sizeOf(VBuffer)) orelse return null));

        for (0..@intCast(talloc.n_chunks)) |n| {
            const chunk_size = talloc.chunks[n].?.max_size;
            buf.chunks[n] = mod.backend.ggml_backend_buft_alloc_buffer(buft, chunk_size);
            if (buf.chunks[n] == null) {
                buf.destroy();
                return null;
            }
            mod.backend.ggml_backend_buffer_set_usage(buf.chunks[n], usage);
        }
        return buf;
    }

    /// Ports `ggml_vbuffer_free` (ggml-alloc.c:399 @b1b770d90).
    fn destroy(buf: ?*VBuffer) void {
        const b = buf orelse return;
        for (0..max_chunks) |i| mod.backend.ggml_backend_buffer_free(b.chunks[i]);
        std.c.free(b);
    }

    /// Ports `ggml_vbuffer_chunk_size` (ggml-alloc.c:409 @b1b770d90).
    fn chunkSize(buf: *VBuffer, chunk: c_int) usize {
        const b = buf.chunks[@intCast(chunk)];
        return if (b != null) mod.backend.ggml_backend_buffer_get_size(b) else 0;
    }

    /// Ports `ggml_vbuffer_size` (ggml-alloc.c:413 @b1b770d90).
    fn totalSize(buf: *VBuffer) usize {
        var size: usize = 0;
        for (0..max_chunks) |i| {
            if (buf.chunks[i] == null) break;
            size += mod.backend.ggml_backend_buffer_get_size(buf.chunks[i]);
        }
        return size;
    }

    /// Ports `ggml_vbuffer_tensor_alloc` (ggml-alloc.c:439 @b1b770d90).
    fn tensorAlloc(buf: *VBuffer, tensor: *mod.impl.ggml_c.ggml_tensor, buf_addr: BufferAddress) void {
        const chunk = buf.chunks[@intCast(buf_addr.chunk)];
        const base: [*]u8 = @ptrCast(mod.backend.ggml_backend_buffer_get_base(chunk));
        _ = mod.backend.ggml_backend_tensor_alloc(chunk, tensor, base + buf_addr.offset);
    }

    /// Ports `ggml_vbuffer_reset` (ggml-alloc.c:445 @b1b770d90).
    fn reset(buf: *VBuffer) void {
        for (0..max_chunks) |i| {
            if (buf.chunks[i] == null) break;
            mod.backend.ggml_backend_buffer_reset(buf.chunks[i]);
        }
    }
};

// -----------------------------------------------------------------------------
// graph allocator

/// Ports `struct hash_node` (ggml-alloc.c:456 @b1b770d90).
///
/// Per-tensor bookkeeping for one graph: how many consumers remain, so the
/// allocator knows when a tensor's memory can be reused.
const HashNode = extern struct {
    n_children: c_int,
    n_views: c_int,
    buffer_id: c_int,
    addr: BufferAddress,
    allocated: bool,
};

/// Ports `struct tensor_alloc` (ggml-alloc.c:464 @b1b770d90).
const TensorAlloc = extern struct {
    buffer_id: c_int,
    addr: BufferAddress,
    /// 0 means pre-allocated, unused, or a view.
    size_max: usize,
};

/// Ports `struct leaf_alloc` (ggml-alloc.c:470 @b1b770d90).
const LeafAlloc = extern struct {
    leaf: TensorAlloc,
};

/// Ports `struct node_alloc` (ggml-alloc.c:474 @b1b770d90).
const NodeAlloc = extern struct {
    dst: TensorAlloc,
    src: [mod.impl.ggml_c.GGML_MAX_SRC]TensorAlloc,
};

/// Ports `struct ggml_gallocr` (ggml-alloc.c:479 @b1b770d90).
///
/// Opaque to callers: only a pointer to it crosses the ABI, so the layout is
/// this file's business alone.
pub const Gallocr = extern struct {
    bufts: [*c]mod.impl.ggml_c.ggml_backend_buffer_type_t,
    buffers: [*c]?*VBuffer,
    buf_tallocs: [*c]?*DynTallocr,
    n_buffers: c_int,

    hash_set: mod.impl.HashSet,
    hash_values: [*c]HashNode,

    node_allocs: [*c]NodeAlloc,
    n_nodes: c_int,

    leaf_allocs: [*c]LeafAlloc,
    n_leafs: c_int,

    /// Ports `ggml_gallocr_hash_get` (ggml-alloc.c:581 @b1b770d90).
    fn hashGet(galloc: *Gallocr, t: *mod.impl.ggml_c.ggml_tensor) *HashNode {
        const i = mod.impl.hashFindOrInsert(&galloc.hash_set, t);
        return &mod.impl.many(HashNode, galloc.hash_values)[i];
    }

    /// Ports `ggml_gallocr_is_own` (ggml-alloc.c:586 @b1b770d90).
    fn isOwn(galloc: *Gallocr, t: *mod.impl.ggml_c.ggml_tensor) bool {
        return galloc.hashGet(t).allocated;
    }

    /// Ports `ggml_gallocr_is_allocated` (ggml-alloc.c:590 @b1b770d90).
    fn isAllocated(galloc: *Gallocr, t: *mod.impl.ggml_c.ggml_tensor) bool {
        return t.data != null // set externally
        or t.buffer != null // on an external buffer, not yet allocated
        or galloc.isOwn(t); // this allocator will place it
    }

    /// Ports `ggml_gallocr_free_extra_space` (ggml-alloc.c:597 @b1b770d90).
    ///
    /// When a node reuses a parent's memory and needs less of it, the tail is
    /// returned to the allocator rather than left stranded.
    fn freeExtraSpace(galloc: *Gallocr, node: *mod.impl.ggml_c.ggml_tensor, parent: *mod.impl.ggml_c.ggml_tensor) void {
        const hn = galloc.hashGet(node);
        const p_hn = galloc.hashGet(parent);

        var parent_size = mod.backend.ggml_backend_buft_get_alloc_size(galloc.bufts[@intCast(p_hn.buffer_id)], parent);
        var node_size = mod.backend.ggml_backend_buft_get_alloc_size(galloc.bufts[@intCast(hn.buffer_id)], node);

        mod.impl.assert(parent_size >= node_size, "parent_size >= node_size");

        // Align both sizes so that what remains after the free is still
        // aligned, which the next allocation from this chunk depends on.
        const p_alloc = galloc.buf_tallocs[@intCast(p_hn.buffer_id)].?;
        parent_size = alignedOffset(null, parent_size, p_alloc.alignment);
        node_size = alignedOffset(null, node_size, p_alloc.alignment);

        if (parent_size > node_size) {
            var p_addr = p_hn.addr;
            p_addr.offset += node_size;
            p_alloc.freeBytes(p_addr, parent_size - node_size);
        }
    }

    /// Ports `ggml_gallocr_allocate_node` (ggml-alloc.c:620 @b1b770d90).
    ///
    /// Prefers writing over a parent whose last consumer this node is, which is
    /// where most of the memory saving in a graph comes from.
    fn allocateNode(galloc: *Gallocr, node: *mod.impl.ggml_c.ggml_tensor, buffer_id: c_int) void {
        mod.impl.assert(buffer_id >= 0, "buffer_id >= 0");
        const hn = galloc.hashGet(node);

        if (galloc.isAllocated(node) or mod.impl.isView(node)) return;

        hn.allocated = true;
        std.debug.assert(hn.addr.offset == 0);

        if (ggml_op_can_inplace(node.op)) {
            for (0..mod.impl.ggml_c.GGML_MAX_SRC) |i| {
                const parent = mod.impl.one(mod.impl.ggml_c.ggml_tensor, node.src[i] orelse continue);

                // External data cannot be written over.
                if (!galloc.isOwn(parent)) continue;

                // Outputs must survive the graph, so they cannot be reused.
                const parent_is_output = (parent.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_OUTPUT) != 0;
                const views_an_output = if (parent.view_src != null)
                    (mod.impl.one(mod.impl.ggml_c.ggml_tensor, parent.view_src).flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_OUTPUT) != 0
                else
                    false;
                if (parent_is_output or views_an_output) continue;

                if (!mod.impl.areSameLayout(node, parent)) continue;

                const p_hn = galloc.hashGet(parent);
                if (p_hn.n_children != 1 or p_hn.n_views != 0) continue;

                if (mod.impl.isView(parent)) {
                    const view_src = mod.impl.one(mod.impl.ggml_c.ggml_tensor, parent.view_src);
                    const view_src_hn = galloc.hashGet(view_src);
                    if (view_src_hn.n_views == 1 and view_src_hn.n_children == 0 and
                        view_src.*.data == parent.data)
                    {
                        std.debug.assert(view_src_hn.addr.chunk == p_hn.addr.chunk and
                            view_src_hn.addr.offset == p_hn.addr.offset);
                        hn.buffer_id = p_hn.buffer_id;
                        hn.addr = p_hn.addr;
                        // Clearing both keeps the parent from being freed while
                        // this node is still using its memory.
                        p_hn.allocated = false;
                        view_src_hn.allocated = false;
                        galloc.freeExtraSpace(node, view_src);
                        return;
                    }
                } else {
                    hn.buffer_id = p_hn.buffer_id;
                    hn.addr = p_hn.addr;
                    p_hn.allocated = false;
                    galloc.freeExtraSpace(node, parent);
                    return;
                }
            }
        }

        const alloc = galloc.buf_tallocs[@intCast(buffer_id)].?;
        const buft = galloc.bufts[@intCast(buffer_id)];
        const size = mod.backend.ggml_backend_buft_get_alloc_size(buft, node);
        hn.buffer_id = buffer_id;
        hn.addr = alloc.allocate(size, node);
    }

    /// Ports `ggml_gallocr_free_node` (ggml-alloc.c:688 @b1b770d90).
    fn freeNode(galloc: *Gallocr, node: *mod.impl.ggml_c.ggml_tensor) void {
        // Graph outputs outlive the graph and are never freed.
        if ((node.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_OUTPUT) != 0) return;

        const hn = galloc.hashGet(node);
        const buffer_id = hn.buffer_id;
        const alloc = galloc.buf_tallocs[@intCast(buffer_id)].?;
        const buft = galloc.bufts[@intCast(buffer_id)];
        const size = mod.backend.ggml_backend_buft_get_alloc_size(buft, node);

        alloc.freeBytes(hn.addr, size);
        hn.allocated = false;
    }
};

/// Ports `get_node_buffer_id` (ggml-alloc.c:711 @b1b770d90).
fn getNodeBufferId(node_buffer_ids: [*c]const c_int, i: usize) c_int {
    return if (node_buffer_ids != null) node_buffer_ids[i] else 0;
}

/// Ports `ggml_gallocr_new_n` (ggml-alloc.c:495 @b1b770d90).
pub fn ggml_gallocr_new_n(bufts: [*c]mod.impl.ggml_c.ggml_backend_buffer_type_t, n_bufs: c_int) callconv(.c) ?*Gallocr {
    const galloc: *Gallocr = @ptrCast(@alignCast(std.c.calloc(1, @sizeOf(Gallocr)) orelse
        mod.impl.abort("galloc != NULL")));

    const n: usize = @intCast(n_bufs);
    galloc.bufts = @ptrCast(@alignCast(std.c.calloc(n, @sizeOf(mod.impl.ggml_c.ggml_backend_buffer_type_t)) orelse
        mod.impl.abort("galloc->bufts != NULL")));
    galloc.buffers = @ptrCast(@alignCast(std.c.calloc(n, @sizeOf(?*VBuffer)) orelse
        mod.impl.abort("galloc->buffers != NULL")));
    galloc.buf_tallocs = @ptrCast(@alignCast(std.c.calloc(n, @sizeOf(?*DynTallocr)) orelse
        mod.impl.abort("galloc->buf_tallocs != NULL")));

    for (0..n) |i| {
        galloc.bufts[i] = bufts[i];
        galloc.buffers[i] = null;

        // One allocator per distinct buffer type: repeats share, so that space
        // freed by one is visible to the others.
        for (0..i) |j| {
            if (bufts[i] == bufts[j]) {
                galloc.buf_tallocs[i] = galloc.buf_tallocs[j];
                break;
            }
        }

        if (galloc.buf_tallocs[i] == null) {
            const alignment = mod.backend.ggml_backend_buft_get_alignment(bufts[i]);
            const max_size = mod.backend.ggml_backend_buft_get_max_size(bufts[i]);
            galloc.buf_tallocs[i] = DynTallocr.create(alignment, max_size);
        }
    }
    galloc.n_buffers = n_bufs;

    return galloc;
}

/// Ports `ggml_gallocr_new` (ggml-alloc.c:531 @b1b770d90).
pub fn ggml_gallocr_new(buft: mod.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) ?*Gallocr {
    var one = buft;
    return ggml_gallocr_new_n(&one, 1);
}

/// Ports `ggml_gallocr_free` (ggml-alloc.c:535 @b1b770d90).
///
/// Buffers and allocators may be shared between slots when a buffer type
/// repeats, so each is freed only at its first occurrence.
pub fn ggml_gallocr_free(galloc_opt: ?*Gallocr) callconv(.c) void {
    const galloc = galloc_opt orelse return;

    for (0..@intCast(galloc.n_buffers)) |i| {
        if (galloc.buffers != null) {
            var freed = false;
            for (0..i) |j| {
                if (galloc.buffers[j] == galloc.buffers[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) VBuffer.destroy(galloc.buffers[i]);
        }
        if (galloc.buf_tallocs != null) {
            var freed = false;
            for (0..i) |j| {
                if (galloc.buf_tallocs[j] == galloc.buf_tallocs[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                if (galloc.buf_tallocs[i]) |t| t.destroy();
            }
        }
    }

    mod.graph.ggml_hash_set_free(&galloc.hash_set);
    std.c.free(@ptrCast(galloc.hash_values));
    std.c.free(@ptrCast(galloc.bufts));
    std.c.free(@ptrCast(galloc.buffers));
    std.c.free(@ptrCast(galloc.buf_tallocs));
    std.c.free(@ptrCast(galloc.node_allocs));
    std.c.free(@ptrCast(galloc.leaf_allocs));
    std.c.free(galloc);
}

/// Ports `ggml_gallocr_alloc_graph_impl` (ggml-alloc.c:715 @b1b770d90).
///
/// Walks the graph twice: once to count each tensor's consumers, then once to
/// place tensors and free them as their last consumer is reached.
fn allocGraphImpl(
    galloc: *Gallocr,
    graph: *mod.impl.CGraph,
    node_buffer_ids: [*c]const c_int,
    leaf_buffer_ids: [*c]const c_int,
) void {
    mod.graph.ggml_hash_set_reset(&galloc.hash_set);
    @memset(galloc.hash_values[0..galloc.hash_set.size], std.mem.zeroes(HashNode));

    // Leafs may be tensors the application wants allocated even though the
    // graph does not consume them.
    for (0..@intCast(graph.n_leafs)) |i| {
        const leaf = graph.leafs[i].?;
        galloc.allocateNode(leaf, getNodeBufferId(leaf_buffer_ids, i));
    }

    // Count children and views, and place graph inputs first so that later
    // allocations cannot land on top of them.
    for (0..@intCast(graph.n_nodes)) |i| {
        const node = graph.nodes[i].?;

        // GGML_OP_NONE nodes are how ggml-backend expresses ordering
        // constraints: the sources matter, the node itself is never used.
        if (mod.impl.isView(node) and node.op != mod.impl.ggml_c.GGML_OP_NONE) {
            galloc.hashGet(node.view_src.?).n_views += 1;
        }

        if ((node.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_INPUT) != 0) {
            galloc.allocateNode(node, getNodeBufferId(node_buffer_ids, i));
        }

        for (0..mod.impl.ggml_c.GGML_MAX_SRC) |j| {
            const src = mod.impl.one(mod.impl.ggml_c.ggml_tensor, node.src[j] orelse continue);
            galloc.hashGet(src).n_children += 1;
            if ((src.*.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_INPUT) != 0) {
                galloc.allocateNode(src, getNodeBufferId(node_buffer_ids, i));
            }
        }
    }

    for (0..@intCast(graph.n_nodes)) |i| {
        const node = graph.nodes[i].?;
        const buffer_id = getNodeBufferId(node_buffer_ids, i);

        // Parents first: at this point only leafs still need placing.
        for (0..mod.impl.ggml_c.GGML_MAX_SRC) |j| {
            const parent = mod.impl.one(mod.impl.ggml_c.ggml_tensor, node.src[j] orelse continue);
            galloc.allocateNode(parent, buffer_id);
        }

        galloc.allocateNode(node, buffer_id);

        // A parent whose last consumer this node was can now be released.
        for (0..mod.impl.ggml_c.GGML_MAX_SRC) |j| {
            const parent = mod.impl.one(mod.impl.ggml_c.ggml_tensor, node.src[j] orelse continue);
            const p_hn = galloc.hashGet(parent);
            p_hn.n_children -= 1;

            if (p_hn.n_children != 0 or p_hn.n_views != 0) continue;

            if (mod.impl.isView(parent)) {
                const view_src = mod.impl.one(mod.impl.ggml_c.ggml_tensor, parent.view_src);
                const view_src_hn = galloc.hashGet(view_src);
                view_src_hn.n_views -= 1;
                if (view_src_hn.n_views == 0 and view_src_hn.n_children == 0 and
                    view_src_hn.allocated)
                {
                    galloc.freeNode(view_src);
                }
            } else if (p_hn.allocated) {
                galloc.freeNode(parent);
            }
        }
    }
}

/// Ports `ggml_gallocr_reserve_n_impl` (ggml-alloc.c:822 @b1b770d90).
///
/// Plans the graph's memory and, unless `no_alloc`, materialises the buffers.
///
/// Return: false when a buffer could not be allocated.
fn reserveNImpl(
    galloc: *Gallocr,
    graph: *mod.impl.CGraph,
    node_buffer_ids: [*c]const c_int,
    leaf_buffer_ids: [*c]const c_int,
    no_alloc: bool,
) bool {
    var min_hash_size: usize = @intCast(graph.n_nodes + graph.n_leafs);
    // 25% margin, to keep the linear probing in the hash set short.
    min_hash_size += min_hash_size / 4;

    if (galloc.hash_set.size < min_hash_size) {
        mod.graph.ggml_hash_set_free(&galloc.hash_set);
        galloc.hash_set = mod.graph.ggml_hash_set_new(min_hash_size);
        mod.impl.assert(galloc.hash_set.keys != null, "galloc->hash_set.keys != NULL");

        std.c.free(@ptrCast(galloc.hash_values));
        galloc.hash_values = @ptrCast(@alignCast(std.c.malloc(@sizeOf(HashNode) * galloc.hash_set.size) orelse
            mod.impl.abort("galloc->hash_values != NULL")));
    }

    for (0..@intCast(galloc.n_buffers)) |i| galloc.buf_tallocs[i].?.reset();

    allocGraphImpl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    if (galloc.n_nodes < graph.n_nodes) {
        std.c.free(@ptrCast(galloc.node_allocs));
        galloc.node_allocs = @ptrCast(@alignCast(std.c.calloc(@intCast(graph.n_nodes), @sizeOf(NodeAlloc)) orelse
            mod.impl.abort("galloc->node_allocs != NULL")));
    }
    galloc.n_nodes = graph.n_nodes;
    for (0..@intCast(graph.n_nodes)) |i| {
        const node = graph.nodes[i].?;
        const node_alloc = &mod.impl.many(NodeAlloc, galloc.node_allocs)[i];
        node_alloc.dst = tensorAllocFor(galloc, node);
        for (0..mod.impl.ggml_c.GGML_MAX_SRC) |j| {
            node_alloc.src[j] = if (node.src[j]) |src|
                tensorAllocFor(galloc, src)
            else
                .{ .buffer_id = -1, .addr = BufferAddress.invalid, .size_max = 0 };
        }
    }

    if (galloc.n_leafs < graph.n_leafs) {
        std.c.free(@ptrCast(galloc.leaf_allocs));
        galloc.leaf_allocs = @ptrCast(@alignCast(std.c.calloc(@intCast(graph.n_leafs), @sizeOf(LeafAlloc)) orelse
            mod.impl.abort("galloc->leaf_allocs != NULL")));
    }
    galloc.n_leafs = graph.n_leafs;
    for (0..@intCast(graph.n_leafs)) |i| {
        mod.impl.many(LeafAlloc, galloc.leaf_allocs)[i].leaf = tensorAllocFor(galloc, graph.leafs[i].?);
    }

    for (0..@intCast(galloc.n_buffers)) |i| {
        // A repeated buffer type shares its buffer with the first slot using it.
        for (0..i) |j| {
            if (galloc.buf_tallocs[j] == galloc.buf_tallocs[i]) {
                galloc.buffers[i] = galloc.buffers[j];
                break;
            }
        }

        // Even with no tensors in it, the buffer must exist so views into it
        // can be initialised.
        var needs_realloc = galloc.buffers[i] == null;
        var new_size: usize = 0;
        const talloc = galloc.buf_tallocs[i].?;
        for (0..@intCast(talloc.n_chunks)) |chunk| {
            const cur_chunk_size = if (galloc.buffers[i]) |b| b.chunkSize(@intCast(chunk)) else 0;
            const new_chunk_size = talloc.maxSize(@intCast(chunk));
            new_size += new_chunk_size;
            if (new_chunk_size > cur_chunk_size) needs_realloc = true;
        }

        if (!needs_realloc) continue;

        if (debug_logging) {
            const cur_size = if (galloc.buffers[i]) |b| b.totalSize() else 0;
            if (cur_size > 0) {
                mod.impl.logDebug(
                    "%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                    .{
                        "ggml_gallocr_reserve_n",
                        mod.backend.ggml_backend_buft_name(galloc.bufts[i]),
                        @as(f64, @floatFromInt(cur_size)) / 1024.0 / 1024.0,
                        @as(f64, @floatFromInt(new_size)) / 1024.0 / 1024.0,
                    },
                );
            }
        }

        VBuffer.destroy(galloc.buffers[i]);
        if (no_alloc) {
            galloc.buffers[i] = null;
        } else {
            galloc.buffers[i] = VBuffer.create(galloc.bufts[i], talloc, mod.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            if (galloc.buffers[i] == null) {
                mod.impl.logError(
                    "%s: failed to allocate %s buffer of size %zu\n",
                    .{ "ggml_gallocr_reserve_n", mod.backend.ggml_backend_buft_name(galloc.bufts[i]), new_size },
                );
                return false;
            }
        }
    }

    return true;
}

/// Builds the `TensorAlloc` record for one tensor.
///
/// Factored out of `ggml_gallocr_reserve_n_impl` (ggml-alloc.c:857 and :871),
/// where the identical body appears once for nodes, once for sources, and once
/// for leafs (:895).
///
/// Parameters:
/// - `galloc`: the allocator holding the placement decisions.
/// - `t`: the tensor to describe.
///
/// Return: the record, marked unallocated when the tensor is a view or already
/// has data.
fn tensorAllocFor(galloc: *Gallocr, t: *mod.impl.ggml_c.ggml_tensor) TensorAlloc {
    if (t.view_src != null or t.data != null) {
        return .{ .buffer_id = -1, .addr = BufferAddress.invalid, .size_max = 0 };
    }
    const hn = galloc.hashGet(t);
    return .{
        .buffer_id = hn.buffer_id,
        .addr = hn.addr,
        .size_max = mod.backend.ggml_backend_buft_get_alloc_size(galloc.bufts[@intCast(hn.buffer_id)], t),
    };
}

/// Ports `ggml_gallocr_reserve_n_size` (ggml-alloc.c:948 @b1b770d90).
pub fn ggml_gallocr_reserve_n_size(
    galloc: *Gallocr,
    graph: *mod.impl.CGraph,
    node_buffer_ids: [*c]const c_int,
    leaf_buffer_ids: [*c]const c_int,
    sizes: [*c]usize,
) callconv(.c) void {
    mod.impl.assert(
        reserveNImpl(galloc, graph, node_buffer_ids, leaf_buffer_ids, true),
        "ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, true)",
    );
    for (0..@intCast(galloc.n_buffers)) |i| {
        sizes[i] = 0;
        const talloc = galloc.buf_tallocs[i].?;
        for (0..@intCast(talloc.n_chunks)) |chunk| {
            sizes[i] += talloc.chunks[chunk].?.max_size;
        }
    }
}

/// Ports `ggml_gallocr_reserve_n` (ggml-alloc.c:959 @b1b770d90).
pub fn ggml_gallocr_reserve_n(
    galloc: *Gallocr,
    graph: *mod.impl.CGraph,
    node_buffer_ids: [*c]const c_int,
    leaf_buffer_ids: [*c]const c_int,
) callconv(.c) bool {
    return reserveNImpl(galloc, graph, node_buffer_ids, leaf_buffer_ids, false);
}

/// Ports `ggml_gallocr_reserve` (ggml-alloc.c:963 @b1b770d90).
pub fn ggml_gallocr_reserve(galloc: *Gallocr, graph: *mod.impl.CGraph) callconv(.c) bool {
    return ggml_gallocr_reserve_n(galloc, graph, null, null);
}

/// Ports `ggml_gallocr_init_tensor` (ggml-alloc.c:967 @b1b770d90).
///
/// Turns a planned placement into a real one, now that buffers exist.
fn initTensor(galloc: *Gallocr, tensor: *mod.impl.ggml_c.ggml_tensor, tensor_alloc: *const TensorAlloc) void {
    const buffer_id = tensor_alloc.buffer_id;

    if (tensor.view_src != null) {
        if (tensor.buffer == null) {
            std.debug.assert(tensor_alloc.addr.offset == std.math.maxInt(usize));
            // A tensor allocated without ggml-backend has nothing to view into.
            if (mod.impl.one(mod.impl.ggml_c.ggml_tensor, tensor.view_src).buffer == null) return;
            _ = mod.backend.ggml_backend_view_init(tensor);
        }
        return;
    }

    if (tensor.data == null) {
        std.debug.assert(tensor_alloc.addr.offset != std.math.maxInt(usize));
        std.debug.assert(mod.backend.ggml_backend_buft_get_alloc_size(galloc.bufts[@intCast(buffer_id)], tensor) <=
            tensor_alloc.size_max);
        galloc.buffers[@intCast(buffer_id)].?.tensorAlloc(tensor, tensor_alloc.addr);
    }
    // Otherwise the tensor already has data; if it also has no buffer it was
    // allocated outside ggml-backend and is left alone.
}

/// Ports `ggml_gallocr_node_needs_realloc` (ggml-alloc.c:994 @b1b770d90).
///
/// Note the sense: this returns true when the existing placement is still
/// usable, despite the name. Kept as-is so the call sites read like the C.
fn nodeNeedsRealloc(galloc: *Gallocr, node: *mod.impl.ggml_c.ggml_tensor, talloc: *const TensorAlloc) bool {
    var node_size: usize = 0;
    if (node.data == null and node.view_src == null) {
        // Data present before but not now means the placement is stale.
        if (talloc.buffer_id < 0) return false;
        node_size = mod.backend.ggml_backend_buft_get_alloc_size(galloc.bufts[@intCast(talloc.buffer_id)], node);
    }
    return talloc.size_max >= node_size;
}

/// Ports `ggml_gallocr_needs_realloc` (ggml-alloc.c:1006 @b1b770d90).
fn needsRealloc(galloc: *Gallocr, graph: *mod.impl.CGraph) bool {
    if (galloc.n_nodes != graph.n_nodes) {
        if (debug_logging) mod.impl.logDebug("%s: graph has different number of nodes\n", .{"ggml_gallocr_needs_realloc"});
        return true;
    }
    if (galloc.n_leafs != graph.n_leafs) {
        if (debug_logging) mod.impl.logDebug("%s: graph has different number of leafs\n", .{"ggml_gallocr_needs_realloc"});
        return true;
    }

    for (0..@intCast(graph.n_nodes)) |i| {
        const node = graph.nodes[i].?;
        const node_alloc = &mod.impl.many(NodeAlloc, galloc.node_allocs)[i];

        if (!nodeNeedsRealloc(galloc, node, &node_alloc.dst)) {
            if (debug_logging) mod.impl.logDebug("%s: node %s is not valid\n", .{ "ggml_gallocr_needs_realloc", &node.name });
            return true;
        }

        for (0..mod.impl.ggml_c.GGML_MAX_SRC) |j| {
            const src = mod.impl.one(mod.impl.ggml_c.ggml_tensor, node.src[j] orelse continue);
            if (!nodeNeedsRealloc(galloc, src, &node_alloc.src[j])) {
                if (debug_logging) mod.impl.logDebug("%s: src %d (%s) of node %s is not valid\n", .{ "ggml_gallocr_needs_realloc", @as(c_int, @intCast(j)), &src.*.name, &node.name });
                return true;
            }
        }
    }

    return false;
}

/// Ports `ggml_gallocr_alloc_graph` (ggml-alloc.c:1049 @b1b770d90).
pub fn ggml_gallocr_alloc_graph(galloc: *Gallocr, graph: *mod.impl.CGraph) callconv(.c) bool {
    // Called for every graph evaluation, so a run that never trips this never
    // reached our allocator at all.
    probe();

    if (needsRealloc(galloc, graph)) {
        if (galloc.n_buffers != 1) {
            if (debug_logging) mod.impl.logDebug("%s: cannot reallocate multi buffer graph automatically, call reserve\n", .{"ggml_gallocr_alloc_graph"});
            return false;
        }
        if (debug_logging) mod.impl.logDebug("%s: reallocating buffers automatically\n", .{"ggml_gallocr_alloc_graph"});
        if (!ggml_gallocr_reserve(galloc, graph)) return false;
    }

    for (0..@intCast(galloc.n_buffers)) |i| {
        if (galloc.buffers[i]) |b| b.reset();
    }

    for (0..@intCast(graph.n_leafs)) |i| {
        initTensor(galloc, graph.leafs[i].?, &mod.impl.many(LeafAlloc, galloc.leaf_allocs)[i].leaf);
    }
    for (0..@intCast(graph.n_nodes)) |i| {
        const node = graph.nodes[i].?;
        const node_alloc = &mod.impl.many(NodeAlloc, galloc.node_allocs)[i];
        for (0..mod.impl.ggml_c.GGML_MAX_SRC) |j| {
            const src = mod.impl.one(mod.impl.ggml_c.ggml_tensor, node.src[j] orelse continue);
            initTensor(galloc, src, &node_alloc.src[j]);
        }
        initTensor(galloc, node, &node_alloc.dst);
    }

    return true;
}

/// Ports `ggml_gallocr_get_buffer_size` (ggml-alloc.c:1097 @b1b770d90).
///
/// Returns 0 for a buffer shared with an earlier slot, so summing across
/// buffers does not double count.
pub fn ggml_gallocr_get_buffer_size(galloc: *Gallocr, buffer_id: c_int) callconv(.c) usize {
    mod.impl.assert(
        buffer_id >= 0 and buffer_id < galloc.n_buffers,
        "buffer_id >= 0 && buffer_id < galloc->n_buffers",
    );

    const buf = galloc.buffers[@intCast(buffer_id)] orelse return 0;

    for (0..@intCast(buffer_id)) |i| {
        if (galloc.buffers[i] == galloc.buffers[@intCast(buffer_id)]) return 0;
    }

    return buf.totalSize();
}

// -----------------------------------------------------------------------------
// utils

/// Ports `ggml_backend_alloc_ctx_tensors_from_buft_collect` (ggml-alloc.c:1117 @b1b770d90).
///
/// Lists every tensor of a context, in its order.
///
/// Return: a `malloc`ed array of `n_tensors.*` pointers the caller `free`s, or
/// null when the context is empty or the array could not be allocated.
fn allocCtxTensorsFromBuftCollect(ctx: *mod.context.Context, n_tensors: *c_int) ?[*]?*mod.impl.ggml_c.ggml_tensor {
    var n: c_int = 0;
    var t = mod.context.ggml_get_first_tensor(ctx);
    while (t != null) : (t = mod.context.ggml_get_next_tensor(ctx, t.?)) n += 1;
    n_tensors.* = n;
    if (n == 0) return null;

    const nbytes = @as(usize, @intCast(n)) * @sizeOf(?*mod.impl.ggml_c.ggml_tensor);
    const tensors: [*]?*mod.impl.ggml_c.ggml_tensor = @ptrCast(@alignCast(std.c.malloc(nbytes) orelse {
        mod.impl.logError("%s: failed to allocate %zu bytes\n", .{ "ggml_backend_alloc_ctx_tensors_from_buft_collect", nbytes });
        return null;
    }));
    var i: usize = 0;
    t = mod.context.ggml_get_first_tensor(ctx);
    while (t != null) : (t = mod.context.ggml_get_next_tensor(ctx, t.?)) {
        tensors[i] = t;
        i += 1;
    }
    return tensors;
}

/// Ports `ggml_backend_alloc_ctx_tensors_from_buft` (ggml-alloc.c:1140 @b1b770d90).
pub fn ggml_backend_alloc_ctx_tensors_from_buft(
    ctx: *mod.impl.ggml_c.ggml_context,
    buft: mod.impl.ggml_c.ggml_backend_buffer_type_t,
) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    mod.impl.assert(mod.context.ggml_get_no_alloc(@ptrCast(@alignCast(ctx))), "ggml_get_no_alloc(ctx) == true");

    var n_tensors: c_int = 0;
    const tensors = allocCtxTensorsFromBuftCollect(@ptrCast(@alignCast(ctx)), &n_tensors) orelse return null;

    const buffer = mod.backend.ggml_backend_buft_alloc_buffer_n(buft, @ptrCast(tensors), n_tensors);
    std.c.free(@ptrCast(tensors));
    return buffer;
}

/// Ports `ggml_backend_alloc_ctx_tensors_from_buft_size` (ggml-alloc.c:1154 @b1b770d90).
pub fn ggml_backend_alloc_ctx_tensors_from_buft_size(
    ctx: *mod.impl.ggml_c.ggml_context,
    buft: mod.impl.ggml_c.ggml_backend_buffer_type_t,
) callconv(.c) usize {
    mod.impl.assert(mod.context.ggml_get_no_alloc(@ptrCast(@alignCast(ctx))), "ggml_get_no_alloc(ctx) == true");

    var n_tensors: c_int = 0;
    const tensors = allocCtxTensorsFromBuftCollect(@ptrCast(@alignCast(ctx)), &n_tensors) orelse return 0;

    const nbytes_total = mod.backend.ggml_backend_buft_get_alloc_size_n(buft, @ptrCast(tensors), n_tensors);
    std.c.free(@ptrCast(tensors));
    return nbytes_total;
}

/// Ports `ggml_backend_alloc_ctx_tensors` (ggml-alloc.c:1168 @b1b770d90).
pub fn ggml_backend_alloc_ctx_tensors(
    ctx: *mod.impl.ggml_c.ggml_context,
    backend: mod.impl.ggml_c.ggml_backend_t,
) callconv(.c) mod.impl.ggml_c.ggml_backend_buffer_t {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, mod.backend.ggml_backend_get_default_buffer_type(backend));
}

// -----------------------------------------------------------------------------
// Unit Tests

test "aligned offset rounds an address up" {
    // With a null base this aligns a size, which is how most callers use it.
    try std.testing.expectEqual(@as(usize, 0), alignedOffset(null, 0, 16));
    try std.testing.expectEqual(@as(usize, 16), alignedOffset(null, 1, 16));
    try std.testing.expectEqual(@as(usize, 16), alignedOffset(null, 16, 16));
    try std.testing.expectEqual(@as(usize, 32), alignedOffset(null, 17, 16));
}

test "ops that can run in place" {
    try std.testing.expect(ggml_op_can_inplace(mod.impl.ggml_c.GGML_OP_ADD));
    try std.testing.expect(ggml_op_can_inplace(mod.impl.ggml_c.GGML_OP_SOFT_MAX));
    try std.testing.expect(!ggml_op_can_inplace(mod.impl.ggml_c.GGML_OP_MUL_MAT));
    try std.testing.expect(!ggml_op_can_inplace(mod.impl.ggml_c.GGML_OP_NONE));
}

test "free blocks stay sorted by offset" {
    var chunk = std.mem.zeroes(TallocrChunk);
    chunk.insertBlock(100, 10);
    chunk.insertBlock(0, 10);
    chunk.insertBlock(50, 10);

    try std.testing.expectEqual(@as(c_int, 3), chunk.n_free_blocks);
    try std.testing.expectEqual(@as(usize, 0), chunk.free_blocks[0].offset);
    try std.testing.expectEqual(@as(usize, 50), chunk.free_blocks[1].offset);
    try std.testing.expectEqual(@as(usize, 100), chunk.free_blocks[2].offset);

    chunk.removeBlock(1);
    try std.testing.expectEqual(@as(c_int, 2), chunk.n_free_blocks);
    try std.testing.expectEqual(@as(usize, 0), chunk.free_blocks[0].offset);
    try std.testing.expectEqual(@as(usize, 100), chunk.free_blocks[1].offset);
}

test "buffer address ordering is chunk-major" {
    const a: BufferAddress = .{ .chunk = 0, .offset = 100 };
    const b: BufferAddress = .{ .chunk = 1, .offset = 0 };
    try std.testing.expect(BufferAddress.less(a, b));
    try std.testing.expect(!BufferAddress.less(b, a));

    const c1: BufferAddress = .{ .chunk = 1, .offset = 50 };
    try std.testing.expect(BufferAddress.less(b, c1));
}
