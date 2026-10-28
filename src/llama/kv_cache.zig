//! The KV cache: the keys and values of every attention layer, per cell.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-kv-cache.h` -- `llama_kv_cache` with its
//!   `stream_copy_info` and `slot_info`, and `llama_kv_cache_context`
//! - `llama.cpp/src/llama-kv-cache.cpp` -- their bodies, `ggml_gen_hadamard`,
//!   `args_set_input_kq_mask`, `set_input_kq_mask_impl` and
//!   `llm_graph_input_k_shift`
//!
//! All at v0.4.0 (`5266f24da`). The cell bookkeeping is `kv_cells.zig`'s.
//!
//! # Shared cells
//!
//! A cache built with `mem_other` views another cache's cells and tensors
//! (`[TAG_KV_CACHE_SHARE_CELLS]`); the C++ shares the cells through a
//! `std`'s `shared_ptr`. Here the cells live in a reference-counted `Shared`
//! the owning cache allocates and the viewing one retains.

const std = @import("std");
const impl = @import("impl.zig");
const arch_mod = @import("arch.zig");
const hparams_mod = @import("hparams.zig");
const cparams_mod = @import("cparams.zig");
const kv_cells = @import("kv_cells.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const graph = @import("graph.zig");
const model_mod = @import("model.zig");
const context_mod = @import("context.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Cparams = cparams_mod.Cparams;
const KvCells = kv_cells.KvCells;
const Ubatch = batch.Ubatch;
const Tensor = c.ggml_tensor;
const max_seq = cparams_mod.max_seq;

fn oom() noreturn {
    impl.abort("llama-kv-cache.cpp", 0, "out of memory");
}

fn swaType(hp: *const Hparams) hparams_mod.SwaType {
    return @fromBackingInt(@intCast(hp.swa_type));
}

/// Ports `ggml_is_power_of_2` (llama-kv-cache.cpp:17 @5266f24da).
fn isPowerOf2(n: c_int) bool {
    return (n & (n - 1)) == 0;
}

/// Ports `ggml_gen_hadamard` (llama-kv-cache.cpp:23 @5266f24da): the
/// orthonormal Walsh-Hadamard rotation of size `n`, whose square is the
/// identity. The C++ handles a non-F32 tensor that it never creates; the
/// matrix is always F32 here.
fn genHadamard(data: []f32, n: usize) void {
    std.debug.assert(isPowerOf2(@intCast(n)));
    data[0] = @floatCast(1.0 / @as(f64, impl.sqrtf(@floatFromInt(n))));
    var s: usize = 1;
    while (s < n) : (s *= 2) {
        for (0..s) |i| {
            for (0..s) |j| {
                const val = data[i * n + j];
                data[(i + s) * n + j] = val;
                data[i * n + (j + s)] = val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
    }
}

/// `LLAMA_NON_CAUSAL_TYPE_SWA_ONLY` and `LLAMA_NON_CAUSAL_TYPE_SWA_FULL`,
/// of `llama_non_causal_type` (llama-hparams.h:33 @5266f24da): with a
/// non-causal context, whether only the SWA layers drop causality, and
/// whether SWA still applies within the current ubatch.
const non_causal_type_swa_only: c_int = 1;
const non_causal_type_swa_full: c_int = 2;

/// Mirrors `llama_kv_cache`'s `stream_copy_info` (llama-kv-cache.h:22 @5266f24da):
/// stream copies queued for the next update.
pub const StreamCopyInfo = struct {
    ssrc: std.ArrayList(u32) = .empty,
    sdst: std.ArrayList(u32) = .empty,

    pub fn empty(self: *const StreamCopyInfo) bool {
        std.debug.assert(self.ssrc.items.len == self.sdst.items.len);
        return self.ssrc.items.len == 0;
    }

    pub fn deinit(self: *StreamCopyInfo) void {
        self.ssrc.deinit(allocator);
        self.sdst.deinit(allocator);
    }
};

/// Mirrors `llama_kv_cache`'s `slot_info` (llama-kv-cache.h:34 @5266f24da):
/// where a ubatch's tokens go, per stream.
pub const SlotInfo = struct {
    s0: u32 = 0,
    s1: u32 = 0,
    /// `[ns]`.
    strm: std.ArrayList(i32) = .empty,
    /// `[ns]`.
    idxs: std.ArrayList(std.ArrayList(u32)) = .empty,

    pub fn deinit(self: *SlotInfo) void {
        self.strm.deinit(allocator);
        for (self.idxs.items) |*v| v.deinit(allocator);
        self.idxs.deinit(allocator);
    }

    /// Return: a deep copy, the caller's to `deinit`.
    pub fn clone(self: *const SlotInfo) SlotInfo {
        var out: SlotInfo = .{ .s0 = self.s0, .s1 = self.s1 };
        out.strm.appendSlice(allocator, self.strm.items) catch oom();
        for (self.idxs.items) |v| {
            var copy: std.ArrayList(u32) = .empty;
            copy.appendSlice(allocator, v.items) catch oom();
            out.idxs.append(allocator, copy) catch oom();
        }
        return out;
    }

    /// Ports `slot_info`'s `head` (llama-kv-cache.h:45 @5266f24da).
    pub fn head(self: *const SlotInfo) u32 {
        impl.assert(self.idxs.items.len == 1, "llama-kv-cache.h", 46, "idxs.size() == 1");
        impl.assert(self.idxs.items[0].items.len != 0, "llama-kv-cache.h", 47, "!idxs[0].empty()");
        return self.idxs.items[0].items[0];
    }

    /// Ports `slot_info`'s `resize` (llama-kv-cache.h:54 @5266f24da).
    pub fn resize(self: *SlotInfo, n: usize) void {
        self.strm.resize(allocator, n) catch oom();
        while (self.idxs.items.len > n) {
            var v = self.idxs.pop().?;
            v.deinit(allocator);
        }
        while (self.idxs.items.len < n) self.idxs.append(allocator, .empty) catch oom();
    }

    /// Ports `slot_info`'s `size` (llama-kv-cache.h:57 @5266f24da).
    pub fn size(self: *const SlotInfo) usize {
        impl.assert(self.idxs.items.len == self.strm.items.len, "llama-kv-cache.h", 58, "idxs.size() == strm.size()");
        impl.assert(self.idxs.items.len != 0, "llama-kv-cache.h", 59, "!idxs.empty()");
        return self.idxs.items[0].items.len;
    }

    /// Ports `slot_info`'s `n_stream` (llama-kv-cache.h:64 @5266f24da).
    pub fn nStream(self: *const SlotInfo) usize {
        return self.strm.items.len;
    }

    /// Ports `slot_info`'s `empty` (llama-kv-cache.h:68 @5266f24da).
    pub fn isEmpty(self: *const SlotInfo) bool {
        return self.idxs.items.len == 0;
    }

    /// Ports `slot_info`'s `clear` (llama-kv-cache.h:72 @5266f24da).
    pub fn clear(self: *SlotInfo) void {
        for (self.idxs.items) |*v| v.deinit(allocator);
        self.idxs.clearRetainingCapacity();
    }

    /// Ports `slot_info`'s `is_contiguous` (llama-kv-cache.h:77 @5266f24da).
    pub fn isContiguous(self: *const SlotInfo) bool {
        if (self.idxs.items.len == 0 or self.idxs.items[0].items.len == 0) return true;
        if (self.idxs.items.len > 1) return false;
        const h = self.idxs.items[0].items[0];
        for (self.idxs.items[0].items, 0..) |idx, i| {
            if (idx != h + i) return false;
        }
        return true;
    }
};

/// `kv_layer` (llama-kv-cache.h:250 @5266f24da): one layer's K and V, and
/// their per-stream views.
const KvLayer = struct {
    il: u32,
    k: *Tensor,
    v: ?*Tensor,
    k_stream: std.ArrayList(*Tensor) = .empty,
    v_stream: std.ArrayList(?*Tensor) = .empty,
};

/// The cells, shared between a cache and the caches viewing it.
const Shared = struct {
    refs: u32 = 1,
    cells: std.ArrayList(KvCells) = .empty,
};

/// Ports `llama_kv_cache` (llama-kv-cache.h:20 @5266f24da).
pub const KvCache = struct {
    model: *const model_mod.Model,
    hparams: *const Hparams,

    /// Whether the value tensor is transposed.
    v_trans: bool = true,
    n_seq_max: u32 = 1,
    n_stream: u32 = 1,
    n_pad: u32 = 1,
    n_swa: u32 = 0,

    attn_rot_k: bool = false,
    attn_rot_v: bool = false,
    n_embd_head_k_all: i32 = 0,
    n_embd_head_v_all: i32 = 0,
    /// The Hadamard matrices by size, kept in host memory.
    attn_rot_hadamard: std.AutoHashMapUnmanaged(i64, []f32) = .empty,

    debug: c_int = 0,
    swa_type: hparams_mod.SwaType = .none,

    /// One context and buffer per buffer type, in the buffer types' name order.
    ctxs_bufs: std.ArrayList(struct { ctx: *c.ggml_context, buf: c.ggml_backend_buffer_t }) = .empty,

    v_heads: std.ArrayList(u32) = .empty,
    other: ?*KvCache = null,
    shared: *Shared,

    seq_to_stream: std.ArrayList(u32) = .empty,
    sc_info: StreamCopyInfo = .{},

    layers: std.ArrayList(KvLayer) = .empty,
    /// `map_layer_ids`: a model layer to its index in `layers`.
    map_layer_ids: std.AutoHashMapUnmanaged(i32, i32) = .empty,

    fn cells(self: *const KvCache) []KvCells {
        return self.shared.cells.items;
    }

    /// Ports the constructor of `llama_kv_cache` (llama-kv-cache.cpp:65
    /// @5266f24da) at its default `name_tag = ""`.
    ///
    /// Parameters mirror the C++'s; `filter`, `reuse` and `share` may be
    /// empty.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        hparams: *const Hparams,
        type_k: c.enum_ggml_type,
        type_v: c.enum_ggml_type,
        v_trans: bool,
        offload: bool,
        unified: bool,
        kv_size_: u32,
        n_seq_max: u32,
        n_pad: u32,
        n_swa: u32,
        swa_type: hparams_mod.SwaType,
        mem_other: ?*KvCache,
        filter: memory.LayerFilter,
        reuse: memory.LayerMap,
        share: memory.LayerMap,
        exc: *Exception,
    ) Thrown!*KvCache {
        return createTagged(model, hparams, type_k, type_v, v_trans, offload, unified, kv_size_, n_seq_max, n_pad, n_swa, swa_type, mem_other, filter, reuse, share, "", exc);
    }

    /// Ports the constructor of `llama_kv_cache` (llama-kv-cache.cpp:65
    /// @5266f24da) with its `name_tag`: a model can hold more than one
    /// cache, so the tag keeps the tensor names unique (`cache_<tag>k_l<il>`).
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn createTagged(
        model: *const model_mod.Model,
        hparams: *const Hparams,
        type_k: c.enum_ggml_type,
        type_v: c.enum_ggml_type,
        v_trans: bool,
        offload: bool,
        unified: bool,
        kv_size_: u32,
        n_seq_max: u32,
        n_pad: u32,
        n_swa: u32,
        swa_type: hparams_mod.SwaType,
        mem_other: ?*KvCache,
        filter: memory.LayerFilter,
        reuse: memory.LayerMap,
        share: memory.LayerMap,
        name_tag: [*:0]const u8,
        exc: *Exception,
    ) Thrown!*KvCache {
        const func = "llama_kv_cache";
        var kv_size = kv_size_;
        const self = allocator.create(KvCache) catch oom();
        self.* = .{
            .model = model,
            .hparams = hparams,
            .v_trans = v_trans,
            .n_seq_max = n_seq_max,
            .n_stream = if (unified) 1 else n_seq_max,
            .n_pad = n_pad,
            .n_swa = n_swa,
            .swa_type = swa_type,
            .other = mem_other,
            .shared = undefined,
        };
        if (mem_other) |o| {
            o.shared.refs += 1;
            self.shared = o.shared;
        } else {
            self.shared = allocator.create(Shared) catch oom();
            self.shared.* = .{};
        }
        errdefer self.destroy();

        // Shared cells view the source's tensors, so their count follows
        // the source's allocation.
        if (self.other) |o| {
            const size_other = o.getSize();
            if (kv_size != size_other) {
                impl.logWarn("%s: kv_size = %u overridden to %u to match the shared source cache\n", .{ func, kv_size, size_other });
                kv_size = size_other;
            }
        }

        impl.assert(kv_size % n_pad == 0, "llama-kv-cache.cpp", 100, "kv_size % n_pad == 0");

        const n_layer = hparams.n_layer_all;
        const n_stream = self.n_stream;

        // One context per buffer type, ordered by the buffer type's name.
        var ctx_map: std.ArrayList(struct { buft: c.ggml_backend_buffer_type_t, ctx: *c.ggml_context }) = .empty;
        // The contexts not yet handed to `ctxs_bufs` are freed with the map,
        // as the C++'s `ggml_context_ptr`s are.
        var n_moved: usize = 0;
        defer {
            for (ctx_map.items[n_moved..]) |e| c.ggml_free(e.ctx);
            ctx_map.deinit(allocator);
        }

        impl.assert(n_stream == 1 or n_stream == n_seq_max, "llama-kv-cache.cpp", 135, "n_stream == 1 || n_stream == n_seq_max");

        self.v_heads.appendNTimes(allocator, 0, n_stream) catch oom();

        // A viewing cache shares the source's cells; the C++ resizes them
        // through the shared vector all the same.
        const old_len = self.shared.cells.items.len;
        if (n_stream < old_len) {
            for (self.shared.cells.items[n_stream..]) |*cl| cl.deinit();
        }
        self.shared.cells.resize(allocator, n_stream) catch oom();
        if (n_stream > old_len) {
            for (self.shared.cells.items[old_len..]) |*cl| cl.* = .{};
        }
        for (self.shared.cells.items) |*cl| cl.resize(kv_size);

        // By default, every sequence maps to stream 0.
        self.seq_to_stream.appendNTimes(allocator, 0, max_seq) catch oom();
        if (n_stream > 1) {
            self.seq_to_stream.shrinkRetainingCapacity(n_stream);
            for (self.seq_to_stream.items, 0..) |*s, i| s.* = @intCast(i);
        }

        // [TAG_V_CACHE_VARIABLE]
        if (v_trans and hparams.isNEmbdVGqaVariable()) {
            impl.logWarn("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n", .{ func, hparams.nEmbdVGqaMax() });
        }

        const is_mla = hparams.isMla();

        var il: u32 = 0;
        while (il < n_layer) : (il += 1) {
            if (!hparams.hasKv(il)) {
                impl.logDebug("%s: layer %3d: does not have KV cache\n", .{ func, il });
                continue;
            }
            if (filter.func != null and !filter.call(@intCast(il))) {
                impl.logDebug("%s: layer %3d: filtered\n", .{ func, il });
                continue;
            }
            if (share.func != null and self.other != null) {
                const il_share = share.call(@intCast(il));
                if (il_share >= 0) {
                    const o = self.other.?;
                    // `map_layer_ids[il_share]` is `operator[]`: a layer the
                    // other cache lacks is inserted as 0, and layer 0 shared.
                    const slot = o.map_layer_ids.getOrPut(allocator, il_share) catch oom();
                    if (!slot.found_existing) slot.value_ptr.* = 0;
                    const layer_share = o.layers.items[@intCast(slot.value_ptr.*)];
                    impl.logWarn("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", .{ func, il, il_share, layer_share.k.data, if (layer_share.v) |v| v.data else null });
                    self.map_layer_ids.put(allocator, @intCast(il), @intCast(self.layers.items.len)) catch oom();
                    var copy: KvLayer = .{ .il = il, .k = layer_share.k, .v = layer_share.v };
                    copy.k_stream.appendSlice(allocator, layer_share.k_stream.items) catch oom();
                    copy.v_stream.appendSlice(allocator, layer_share.v_stream.items) catch oom();
                    self.layers.append(allocator, copy) catch oom();
                    continue;
                }
            }

            const hk: i32 = @intCast(hparams.nEmbdHeadK(il));
            if (self.n_embd_head_k_all == 0) {
                self.n_embd_head_k_all = hk;
            } else if (self.n_embd_head_k_all > 0 and self.n_embd_head_k_all != hk) {
                self.n_embd_head_k_all = -1;
            }
            if (!is_mla) {
                const hv: i32 = @intCast(hparams.nEmbdHeadV(il));
                if (self.n_embd_head_v_all == 0) {
                    self.n_embd_head_v_all = hv;
                } else if (self.n_embd_head_v_all > 0 and self.n_embd_head_v_all != hv) {
                    self.n_embd_head_v_all = -1;
                }
            }

            // [TAG_V_CACHE_VARIABLE]
            const n_embd_k_gqa: i64 = hparams.nEmbdKGqa(il);
            const n_embd_v_gqa: i64 = if (!v_trans) hparams.nEmbdVGqa(il) else hparams.nEmbdVGqaMax();

            var dev_name: [*:0]const u8 = "CPU";
            var buft = c.ggml_backend_cpu_buffer_type();
            if (offload) {
                const dev = model.devLayer(il);
                buft = c.ggml_backend_dev_buffer_type(dev);
                dev_name = c.ggml_backend_dev_name(dev);
            }

            impl.logDebug("%s: layer %3d: dev = %s\n", .{ func, il, dev_name });

            const ctx = ctxForBuft(&ctx_map, buft, n_stream, n_layer) orelse
                return exc.throw(.runtime_error, "failed to create ggml context for kv cache", .{});

            const has_v = !is_mla;
            const k = c.ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
            const v: ?*Tensor = if (has_v) c.ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) else null;

            _ = c.ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
            if (v) |t| _ = c.ggml_format_name(t, "cache_%sv_l%d", name_tag, il);

            var layer: KvLayer = .{ .il = il, .k = k, .v = v };
            var s: u32 = 0;
            while (s < n_stream) : (s += 1) {
                layer.k_stream.append(allocator, c.ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k.*.nb[1], s * k.*.nb[2])) catch oom();
                layer.v_stream.append(allocator, if (v) |t| c.ggml_view_2d(ctx, t, n_embd_v_gqa, kv_size, t.*.nb[1], s * t.*.nb[2]) else null) catch oom();
            }

            self.map_layer_ids.put(allocator, @intCast(il), @intCast(self.layers.items.len)) catch oom();
            self.layers.append(allocator, layer) catch oom();
        }

        if (reuse.func != null) {
            impl.logDebug("%s: reusing layers:\n", .{func});
            il = 0;
            while (il < n_layer) : (il += 1) {
                const il_reuse = reuse.call(@intCast(il));
                if (il_reuse < 0) {
                    impl.logDebug("%s: - layer %3d: no reuse\n", .{ func, il });
                    continue;
                }
                if (filter.func != null and !filter.call(@intCast(il))) {
                    impl.logDebug("%s: - layer %3d: filtered\n", .{ func, il });
                    continue;
                }
                impl.assert(self.map_layer_ids.contains(il_reuse), "llama-kv-cache.cpp", 268, "map_layer_ids.find(il_reuse) != map_layer_ids.end()");
                self.map_layer_ids.put(allocator, @intCast(il), self.map_layer_ids.get(il_reuse).?) catch oom();
                impl.logDebug("%s: - layer %3d: reuse layer %d, is_swa = %d\n", .{ func, il, il_reuse, @as(c_int, @intFromBool(hparams.isSwa(il))) });
            }
        }

        // Allocate the tensors and clear the buffers, so the padding holds
        // no NaNs.
        for (ctx_map.items) |e| {
            var buf: c.ggml_backend_buffer_t = null;
            if (hparams.no_alloc) {
                buf = c.ggml_backend_buft_alloc_buffer(e.buft, 0); // a dummy buffer
                var t: ?*Tensor = c.ggml_get_first_tensor(e.ctx);
                while (t) |cur| : (t = c.ggml_get_next_tensor(e.ctx, cur)) cur.buffer = buf;
            } else {
                buf = c.ggml_backend_alloc_ctx_tensors_from_buft(e.ctx, e.buft);
            }
            if (buf == null) return exc.throw(.runtime_error, "failed to allocate buffer for kv cache", .{});
            impl.logInfo("%s: %10s KV buffer size = %8.2f MiB\n", .{ func, c.ggml_backend_buffer_name(buf), @as(f64, @floatFromInt(c.ggml_backend_buffer_get_size(buf))) / 1024.0 / 1024.0 });
            c.ggml_backend_buffer_clear(buf, 0);
            self.ctxs_bufs.append(allocator, .{ .ctx = e.ctx, .buf = buf }) catch oom();
            n_moved += 1;
        }

        {
            const memory_size_k = self.sizeKBytes();
            const memory_size_v = self.sizeVBytes();
            impl.logInfo("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", .{
                func,
                @as(f64, @as(f32, @floatFromInt(memory_size_k + memory_size_v)) / (1024.0 * 1024.0)),
                kv_size,
                @as(c_int, @intCast(self.layers.items.len)),
                n_seq_max,
                n_stream,
                c.ggml_type_name(type_k),
                @as(f64, @as(f32, @floatFromInt(memory_size_k)) / (1024.0 * 1024.0)),
                c.ggml_type_name(type_v),
                @as(f64, @as(f32, @floatFromInt(memory_size_v)) / (1024.0 * 1024.0)),
            });
        }

        // [TAG_KV_CACHE_SHARE_CELLS]
        if (self.other) |o| {
            self.n_embd_head_k_all = o.n_embd_head_k_all;
            self.n_embd_head_v_all = o.n_embd_head_v_all;
            self.attn_rot_k = o.attn_rot_k;
            self.attn_rot_v = o.attn_rot_v;
        } else {
            const attn_rot_disable = if (std.c.getenv("LLAMA_ATTN_ROT_DISABLE")) |v| impl.atoi(v) != 0 else false;
            if (attn_rot_disable) impl.logWarn("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", .{func});

            self.attn_rot_k = !attn_rot_disable and self.n_embd_head_k_all > 0 and c.ggml_is_quantized(type_k) and hparams.nEmbdHeadK(0) % 64 == 0;

            // Always rotate for the DeepSeek lightning indexers.
            const a = model.arch;
            if ((a == .deepseek32 or a == .deepseek4 or a == .glm_dsa or a == .dots3note) and hparams.n_embd_head_k_full == hparams.indexer_head_size) {
                self.attn_rot_k = true;
            }

            self.attn_rot_v = !attn_rot_disable and self.n_embd_head_v_all > 0 and c.ggml_is_quantized(type_v) and hparams.nEmbdHeadV(0) % 64 == 0;
        }

        impl.logInfo("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", .{ func, @as(c_int, @intFromBool(self.attn_rot_k)), self.n_embd_head_k_all });
        impl.logInfo("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", .{ func, @as(c_int, @intFromBool(self.attn_rot_v)), self.n_embd_head_v_all });

        // Precompute the Hadamard matrices, in host memory.
        if (self.attn_rot_k or self.attn_rot_v) {
            var n: i64 = 64;
            while (n <= @max(self.n_embd_head_k_all, self.n_embd_head_v_all)) : (n *= 2) {
                const nn: usize = @intCast(n);
                const data = allocator.alloc(f32, nn * nn) catch oom();
                @memset(data, 0);
                genHadamard(data, nn);
                self.attn_rot_hadamard.put(allocator, n, data) catch oom();
            }
        }

        if (std.c.getenv("LLAMA_KV_CACHE_DEBUG")) |v| self.debug = impl.atoi(v);
        return self;
    }

    fn ctxForBuft(map: anytype, buft: c.ggml_backend_buffer_type_t, n_stream: u32, n_layer: u32) ?*c.ggml_context {
        const name = std.mem.span(c.ggml_backend_buft_name(buft));
        var pos: usize = 0;
        while (pos < map.items.len) : (pos += 1) {
            const order = std.mem.order(u8, std.mem.span(c.ggml_backend_buft_name(map.items[pos].buft)), name);
            if (order == .eq) return map.items[pos].ctx;
            if (order == .gt) break;
        }
        const ctx = c.ggml_init(.{
            .mem_size = 2 * (1 + @as(usize, n_stream)) * n_layer * c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        }) orelse return null;
        map.insert(allocator, pos, .{ .buft = buft, .ctx = ctx }) catch oom();
        return ctx;
    }

    /// Frees the cache: its buffers, and the cells when it was their last
    /// holder.
    ///
    /// Return: nothing.
    pub fn destroy(self: *KvCache) void {
        for (self.ctxs_bufs.items) |e| {
            c.ggml_backend_buffer_free(e.buf);
            c.ggml_free(e.ctx);
        }
        self.ctxs_bufs.deinit(allocator);
        var it = self.attn_rot_hadamard.valueIterator();
        while (it.next()) |v| allocator.free(v.*);
        self.attn_rot_hadamard.deinit(allocator);
        self.v_heads.deinit(allocator);
        self.shared.refs -= 1;
        if (self.shared.refs == 0) {
            for (self.shared.cells.items) |*cl| cl.deinit();
            self.shared.cells.deinit(allocator);
            allocator.destroy(self.shared);
        }
        self.seq_to_stream.deinit(allocator);
        self.sc_info.deinit();
        for (self.layers.items) |*l| {
            l.k_stream.deinit(allocator);
            l.v_stream.deinit(allocator);
        }
        self.layers.deinit(allocator);
        self.map_layer_ids.deinit(allocator);
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache`'s `clear` (llama-kv-cache.cpp:369 @5266f24da).
    pub fn clear(self: *KvCache, data: bool) void {
        for (0..self.n_stream) |s| {
            self.cells()[s].reset();
            self.v_heads.items[s] = 0;
        }
        if (data) {
            for (self.ctxs_bufs.items) |e| c.ggml_backend_buffer_clear(e.buf, 0);
        }
    }

    /// Ports `llama_kv_cache`'s `seq_rm` (llama-kv-cache.cpp:382 @5266f24da).
    pub fn seqRm(self: *KvCache, seq_id: i32, p0_: i32, p1_: i32) bool {
        // [TAG_KV_CACHE_SHARE_CELLS]
        if (self.other != null) return true;

        impl.assert(seq_id == -1 or (seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len), "llama-kv-cache.cpp", 389, "seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size())");

        const p0 = if (p0_ < 0) 0 else p0_;
        const p1 = if (p1_ < 0) std.math.maxInt(i32) else p1_;

        if (seq_id >= 0) {
            const strm = self.seq_to_stream.items[@intCast(seq_id)];
            const cl = &self.cells()[strm];
            const head = &self.v_heads.items[strm];
            var new_head: u32 = cl.size();
            for (0..cl.size()) |i_| {
                const i: u32 = @intCast(i_);
                if (!cl.posIn(i, p0, p1)) continue;
                if (cl.seqHas(i, seq_id) and cl.seqRm(i, seq_id)) {
                    if (new_head == cl.size()) new_head = i;
                }
            }
            // A freed slot is where the next search starts.
            if (new_head != cl.size() and new_head < head.*) head.* = new_head;
        } else {
            // Match any sequence.
            for (0..self.n_stream) |s| {
                const cl = &self.cells()[s];
                const head = &self.v_heads.items[s];
                var new_head: u32 = cl.size();
                for (0..cl.size()) |i_| {
                    const i: u32 = @intCast(i_);
                    if (!cl.posIn(i, p0, p1)) continue;
                    cl.rm(i);
                    if (new_head == cl.size()) new_head = i;
                }
                if (new_head != cl.size() and new_head < head.*) head.* = new_head;
            }
        }
        return true;
    }

    /// Ports `llama_kv_cache`'s `seq_cp` (llama-kv-cache.cpp:451 @5266f24da).
    pub fn seqCp(self: *KvCache, seq_id_src: i32, seq_id_dst: i32, p0_: i32, p1_: i32) void {
        if (self.other != null) return;

        impl.assert(seq_id_src >= 0 and @as(usize, @intCast(seq_id_src)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 457, "seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size()");
        impl.assert(seq_id_dst >= 0 and @as(usize, @intCast(seq_id_dst)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 458, "seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size()");

        const s0 = self.seq_to_stream.items[@intCast(seq_id_src)];
        const s1 = self.seq_to_stream.items[@intCast(seq_id_dst)];

        if (s0 == s1) {
            // In the same stream only the metadata changes.
            const cl = &self.cells()[s0];
            if (seq_id_src == seq_id_dst) return;
            const p0 = if (p0_ < 0) 0 else p0_;
            const p1 = if (p1_ < 0) std.math.maxInt(i32) else p1_;
            for (0..cl.size()) |i_| {
                const i: u32 = @intCast(i_);
                if (!cl.posIn(i, p0, p1)) continue;
                if (cl.seqHas(i, seq_id_src)) cl.seqAdd(i, seq_id_dst);
            }
            return;
        }

        // A copy across streams copies the buffer data too.
        var is_full = true;
        const size: i32 = @intCast(self.getSize());
        if (p0_ > 0 and p0_ + 1 < size) is_full = false;
        if (p1_ > 0 and p1_ + 1 < size) is_full = false;
        impl.assert(is_full, "llama-kv-cache.cpp", 506, "is_full && \"seq_cp() is only supported for full KV buffers\"");

        // The buffer copy happens on the next update.
        self.sc_info.ssrc.append(allocator, s0) catch oom();
        self.sc_info.sdst.append(allocator, s1) catch oom();

        const src = &self.cells()[s0];
        const dst = &self.cells()[s1];
        dst.reset();
        for (0..src.size()) |i_| {
            const i: u32 = @intCast(i_);
            if (src.seqHas(i, seq_id_src)) {
                var pos = src.posGet(i);
                const shift = src.getShift(i);
                const ext = src.extGet(i);
                if (shift != 0) {
                    pos -= shift;
                    std.debug.assert(pos >= 0);
                }
                dst.posSet(i, pos);
                dst.seqAdd(i, seq_id_dst);
                if (shift != 0) _ = dst.posAdd(i, shift);
                dst.extSet(i, ext);
            }
        }
        self.v_heads.items[s1] = self.v_heads.items[s0];
    }

    /// Ports `llama_kv_cache`'s `seq_keep` (llama-kv-cache.cpp:543 @5266f24da).
    pub fn seqKeep(self: *KvCache, seq_id: i32) void {
        if (self.other != null) return;
        impl.assert(seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 549, "seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()");
        const strm = self.seq_to_stream.items[@intCast(seq_id)];
        const cl = &self.cells()[strm];
        const head = &self.v_heads.items[strm];
        var new_head: u32 = cl.size();
        for (0..cl.size()) |i_| {
            const i: u32 = @intCast(i_);
            if (cl.seqKeep(i, seq_id)) {
                if (new_head == cl.size()) new_head = i;
            }
        }
        if (new_head != cl.size() and new_head < head.*) head.* = new_head;
    }

    /// Ports `llama_kv_cache`'s `seq_add` (llama-kv-cache.cpp:570 @5266f24da).
    pub fn seqAdd(self: *KvCache, seq_id: i32, p0_: i32, p1_: i32, shift: i32) void {
        if (self.other != null) return;
        impl.assert(seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 576, "seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()");
        impl.assert(self.hparams.nPosPerEmbd() == 1, "llama-kv-cache.cpp", 577, "hparams.n_pos_per_embd() == 1 && \"seq_add() is only supported for n_pos_per_embd() == 1\"");

        const strm = self.seq_to_stream.items[@intCast(seq_id)];
        const cl = &self.cells()[strm];
        const head = &self.v_heads.items[strm];

        if (shift == 0) return;

        var new_head: u32 = cl.size();
        const p0 = if (p0_ < 0) 0 else p0_;
        const p1 = if (p1_ < 0) std.math.maxInt(i32) else p1_;

        // No range: return before looping over every cell.
        if (p0 == p1) return;

        for (0..cl.size()) |i_| {
            const i: u32 = @intCast(i_);
            if (!cl.posIn(i, p0, p1)) continue;
            if (cl.seqHas(i, seq_id)) {
                if (cl.posAdd(i, shift)) {
                    if (new_head == cl.size()) new_head = i;
                }
            }
        }
        // A freed slot is where the next search starts; else the beginning.
        head.* = if (new_head != cl.size()) new_head else 0;
    }

    /// Ports `llama_kv_cache`'s `seq_div` (llama-kv-cache.cpp:620 @5266f24da).
    pub fn seqDiv(self: *KvCache, seq_id: i32, p0_: i32, p1_: i32, d: c_int) void {
        if (self.other != null) return;
        impl.assert(seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 626, "seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()");
        impl.assert(self.hparams.nPosPerEmbd() == 1, "llama-kv-cache.cpp", 627, "hparams.n_pos_per_embd() == 1 && \"seq_div() is only supported for n_pos_per_embd() == 1\"");

        const cl = &self.cells()[self.seq_to_stream.items[@intCast(seq_id)]];
        if (d == 1) return;
        const p0 = if (p0_ < 0) 0 else p0_;
        const p1 = if (p1_ < 0) std.math.maxInt(i32) else p1_;
        if (p0 == p1) return;
        for (0..cl.size()) |i_| {
            const i: u32 = @intCast(i_);
            if (!cl.posIn(i, p0, p1)) continue;
            if (cl.seqHas(i, seq_id)) cl.posDiv(i, d);
        }
    }

    /// Ports `llama_kv_cache`'s `seq_pos_min` (llama-kv-cache.cpp:659 @5266f24da).
    pub fn seqPosMin(self: *const KvCache, seq_id: i32) i32 {
        if (self.other) |o| return o.seqPosMin(seq_id);
        impl.assert(seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 665, "seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()");
        return self.cells()[self.seq_to_stream.items[@intCast(seq_id)]].seqPosMin(seq_id);
    }

    /// Ports `llama_kv_cache`'s `seq_pos_max` (llama-kv-cache.cpp:672 @5266f24da).
    pub fn seqPosMax(self: *const KvCache, seq_id: i32) i32 {
        if (self.other) |o| return o.seqPosMax(seq_id);
        impl.assert(seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 678, "seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()");
        return self.cells()[self.seq_to_stream.items[@intCast(seq_id)]].seqPosMax(seq_id);
    }

    /// Ports `llama_kv_cache`'s `memory_breakdown` (llama-kv-cache.cpp:685 @5266f24da).
    pub fn memoryBreakdown(self: *const KvCache, out: *std.ArrayList(memory.BuftSize)) void {
        for (self.ctxs_bufs.items) |e| {
            const buft = c.ggml_backend_buffer_get_type(e.buf);
            if (self.hparams.no_alloc) {
                impl.assert(c.ggml_backend_buffer_get_base(e.buf) == null, "llama-kv-cache.cpp", 691, "ggml_backend_buffer_get_base(buf.get()) == nullptr");
                memory.addBuftSize(out, buft, c.ggml_backend_alloc_ctx_tensors_from_buft_size(e.ctx, buft));
            } else {
                memory.addBuftSize(out, buft, c.ggml_backend_buffer_get_size(e.buf));
            }
        }
    }

    /// Ports `llama_kv_cache`'s `init_batch` (llama-kv-cache.cpp:702 @5266f24da):
    /// split the batch and find a slot for every ubatch.
    pub fn initBatch(self: *KvCache, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *KvCacheContext {
        _ = embd_all;
        blk: {
            balloc.splitReset();
            var ubatches: std.ArrayList(Ubatch) = .empty;
            while (true) {
                const ub = if (self.n_stream == 1) balloc.splitSimple(n_ubatch) else balloc.splitEqual(n_ubatch, true, 0);
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }
            if (balloc.getNUsed() < balloc.getNTokens()) {
                // No suitable split.
                releaseUbatches(&ubatches);
                break :blk;
            }
            const sinfos = self.prepare(ubatches.items) orelse {
                releaseUbatches(&ubatches);
                break :blk;
            };
            return KvCacheContext.createBatch(self, sinfos, ubatches);
        }
        return KvCacheContext.createStatus(.failed_prepare);
    }

    /// Ports `llama_kv_cache`'s `init_full` (llama-kv-cache.cpp:739 @5266f24da).
    pub fn initFull(self: *KvCache) *KvCacheContext {
        return KvCacheContext.createFull(self);
    }

    /// Ports `llama_kv_cache`'s `init_update` (llama-kv-cache.cpp:743 @5266f24da).
    pub fn initUpdate(self: *KvCache, lctx: *anyopaque, optimize: bool) *KvCacheContext {
        _ = optimize;
        const do_shift = self.getHasShift();
        const sc = self.sc_info;
        self.sc_info = .{};
        return KvCacheContext.createUpdate(self, lctx, do_shift, sc);
    }

    /// Ports `llama_kv_cache`'s `prepare` (llama-kv-cache.cpp:751 @5266f24da):
    /// place every ubatch in turn to find their slots, then put the cells
    /// back as they were.
    ///
    /// Return: the slots, the caller's; null when one does not fit.
    pub fn prepare(self: *KvCache, ubatches: []const Ubatch) ?std.ArrayList(SlotInfo) {
        var res: std.ArrayList(SlotInfo) = .empty;

        const State = struct {
            sinfo: SlotInfo,
            v_heads_old: std.ArrayList(u32),
            v_cells: std.ArrayList(KvCells),
        };
        var states: std.ArrayList(State) = .empty;
        defer {
            for (states.items) |*st| {
                st.sinfo.deinit();
                st.v_heads_old.deinit(allocator);
                for (st.v_cells.items) |*cl| cl.deinit();
                st.v_cells.deinit(allocator);
            }
            states.deinit(allocator);
        }

        var success = true;
        for (ubatches) |*ubatch| {
            // Only find a slot; do not modify the cells yet.
            var sinfo_new = self.findSlot(ubatch, false);
            if (sinfo_new.isEmpty()) {
                sinfo_new.deinit();
                success = false;
                break;
            }
            res.append(allocator, sinfo_new) catch oom();

            // Remember the old state of the cells.
            var st: State = .{ .sinfo = sinfo_new.clone(), .v_heads_old = .empty, .v_cells = .empty };
            st.v_heads_old.appendSlice(allocator, self.v_heads.items) catch oom();
            for (0..sinfo_new.nStream()) |s| {
                const cl = &self.cells()[@intCast(sinfo_new.strm.items[s])];
                st.v_cells.append(allocator, cl.cpIdxs(sinfo_new.idxs.items[s].items)) catch oom();
            }
            states.append(allocator, st) catch oom();

            // Now place the ubatch.
            self.applyUbatch(&sinfo_new, ubatch);
        }

        impl.assert(states.items.len != 0 or !success, "llama-kv-cache.cpp", 795, "!states.empty() || !success");

        // Restore the cells, most recent first.
        var i = states.items.len;
        while (i > 0) {
            i -= 1;
            const st = &states.items[i];
            for (0..st.sinfo.nStream()) |s| {
                const strm: usize = @intCast(st.sinfo.strm.items[s]);
                self.cells()[strm].setIdxs(st.sinfo.idxs.items[s].items, &st.v_cells.items[s]);
                self.v_heads.items[strm] = st.v_heads_old.items[strm];
            }
        }

        if (!success) {
            for (res.items) |*si| si.deinit();
            res.deinit(allocator);
            return null;
        }
        return res;
    }

    /// Ports `llama_kv_cache`'s `update` (llama-kv-cache.cpp:817 @5266f24da):
    /// pending stream copies, then a K-shift if positions moved.
    ///
    /// Return: whether a graph was computed.
    pub fn update(self: *KvCache, lctx_: *anyopaque, do_shift: bool, sc_info: *const StreamCopyInfo) bool {
        if (self.other != null) return true;
        const lctx: *context_mod.Context = @ptrCast(@alignCast(lctx_));
        var updated = false;
        const sched = lctx.getSched();

        if (!sc_info.empty()) {
            std.debug.assert(self.n_stream > 1);
            lctx.synchronize();
            for (sc_info.ssrc.items, sc_info.sdst.items) |ssrc, sdst| {
                std.debug.assert(ssrc < self.n_stream and sdst < self.n_stream and ssrc != sdst);
                impl.logDebug("%s: copying KV buffer: stream %d to stream %d\n", .{ "update", ssrc, sdst });
                for (self.layers.items) |*layer| {
                    c.ggml_backend_tensor_copy(layer.k_stream.items[ssrc], layer.k_stream.items[sdst]);
                    if (layer.v_stream.items[ssrc] != null) c.ggml_backend_tensor_copy(layer.v_stream.items[ssrc], layer.v_stream.items[sdst]);
                }
            }
        }

        if (do_shift) {
            if (!self.getCanShift()) impl.abort("llama-kv-cache.cpp", 859, "The current KV cache / model configuration does not support K-shift");
            impl.logDebug("%s: applying K-shift\n", .{"update"});

            if (self.hparams.rope_type != c.LLAMA_ROPE_TYPE_NONE) {
                c.ggml_backend_sched_reset(sched);
                const res = lctx.getGfResReserve();
                res.reset();
                const gf = self.buildGraphShift(res, lctx);
                if (!c.ggml_backend_sched_alloc_graph(sched, gf)) {
                    impl.logError("%s: failed to allocate compute graph for K-shift\n", .{"update"});
                    return updated;
                }
                res.setInputs(null);
                if (lctx.graphCompute(gf, false) != c.GGML_STATUS_SUCCESS) {
                    impl.logError("%s: failed to compute K-shift\n", .{"update"});
                    return updated;
                }
                updated = true;
            }
            for (self.cells()[0..self.n_stream]) |*cl| cl.resetShift();
        }
        return updated;
    }

    /// Ports `llama_kv_cache`'s `find_slot` (llama-kv-cache.cpp:898 @5266f24da).
    ///
    /// Return: the slot; empty when the ubatch does not fit.
    pub fn findSlot(self: *const KvCache, ubatch: *const Ubatch, cont: bool) SlotInfo {
        if (self.debug > 0) self.debugCells(ubatch);

        var n_tokens: u32 = ubatch.n_tokens;
        var n_seqs: u32 = 1;
        if (self.n_stream > 1) {
            impl.assert(n_tokens % ubatch.n_seqs_unq == 0, "llama-kv-cache.cpp", 970, "n_tokens % ubatch.n_seqs_unq == 0");
            n_seqs = ubatch.n_seqs_unq;
            n_tokens = n_tokens / n_seqs;
        }

        var res: SlotInfo = .{ .s0 = max_seq, .s1 = 0 };
        res.resize(n_seqs);

        for (0..n_seqs) |s| {
            const seq_id = ubatch.seq_id_unq[s];
            if (self.n_stream > 1) {
                impl.assert(ubatch.n_seq_id[s * n_tokens] == 1, "llama-kv-cache.cpp", 989, "ubatch.n_seq_id[s*n_tokens]    == 1");
                impl.assert(ubatch.seq_id[s * n_tokens][0] == seq_id, "llama-kv-cache.cpp", 990, "ubatch.seq_id  [s*n_tokens][0] == seq_id");
            }
            const strm = self.seq_to_stream.items[@intCast(seq_id)];
            res.s0 = @min(res.s0, strm);
            res.s1 = @max(res.s1, strm);
            res.strm.items[s] = @intCast(strm);
            res.idxs.items[s].ensureTotalCapacity(allocator, n_tokens) catch oom();

            const cl = &self.cells()[strm];
            var head_cur = self.v_heads.items[strm];

            // With enough unused cells before the head, search from the start.
            if (head_cur > cl.getUsed() + 2 * n_tokens) head_cur = 0;

            if (n_tokens > cl.size()) {
                impl.logError("%s: n_tokens = %d > size = %u\n", .{ "find_slot", n_tokens, cl.size() });
                res.deinit();
                return .{};
            }

            var n_tested: u32 = 0;
            // A continuous slot tests all tokens from the head; otherwise one by one.
            const n_test: u32 = if (cont) n_tokens else 1;

            while (true) {
                if (head_cur + n_test > cl.size()) {
                    n_tested += cl.size() - head_cur;
                    head_cur = 0;
                    continue;
                }
                for (0..n_test) |_| {
                    const idx = head_cur;
                    head_cur += 1;
                    n_tested += 1;

                    // A cell is usable when empty, or held by one sequence
                    // whose SWA masks it.
                    var can_use = cl.isEmpty(idx);
                    if (!can_use and cl.seqCount(idx) == 1) {
                        const pos_cell = cl.posGet(idx);
                        const seq_id_cell = cl.seqGet(idx);
                        if (Hparams.isMaskedSwa(self.n_swa, self.swa_type, pos_cell, cl.seqPosMax(seq_id_cell) + 1)) can_use = true;
                    }
                    if (can_use) {
                        res.idxs.items[s].append(allocator, idx) catch oom();
                    } else if (cont) {
                        break;
                    }
                }
                if (res.idxs.items[s].items.len == n_tokens) break;
                if (cont) res.idxs.items[s].clearRetainingCapacity();
                if (n_tested >= cl.size()) {
                    res.deinit();
                    return .{};
                }
            }
            // No suitable slot.
            if (res.idxs.items[s].items.len < n_tokens) {
                res.deinit();
                return .{};
            }
        }
        std.debug.assert(res.s1 >= res.s0);
        return res;
    }

    fn debugCells(self: *const KvCache, ubatch: *const Ubatch) void {
        var ss: std.ArrayList(u8) = .empty;
        defer ss.deinit(allocator);
        for (0..ubatch.n_seqs_unq) |s| {
            const seq_id = ubatch.seq_id_unq[s];
            const stream_id = self.seq_to_stream.items[@intCast(seq_id)];
            const cl = &self.cells()[stream_id];
            const head_cur = self.v_heads.items[stream_id];

            impl.logDebug("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n", .{
                "find_slot", stream_id, cl.usedMaxP1(), cl.getUsed(), head_cur, self.getSize(), self.n_swa,
            });

            if ((self.debug == 2 and self.n_swa > 0) or self.debug > 2) {
                ss.clearRetainingCapacity();
                for (0..cl.size()) |i_| {
                    const i: u32 = @intCast(i_);
                    if (cl.isEmpty(i)) {
                        ss.append(allocator, '.') catch oom();
                    } else if (cl.seqCount(i) == 1) {
                        ss.print(allocator, "{d}", .{cl.seqGet(i)}) catch oom();
                    } else {
                        ss.append(allocator, 'M') catch oom();
                    }
                    if (i % 256 == 255) ss.appendSlice(allocator, " *\n") catch oom();
                }
                ss.append(allocator, 0) catch oom();
                impl.logDebug("\n%s\n", .{@as([*:0]const u8, @ptrCast(ss.items.ptr))});
            }

            if ((self.debug == 2 and self.n_swa > 0) or self.debug > 2) {
                ss.clearRetainingCapacity();
                for (0..cl.size()) |i_| {
                    const i: u32 = @intCast(i_);
                    const start = ss.items.len;
                    if (cl.isEmpty(i)) {
                        ss.append(allocator, '.') catch oom();
                    } else {
                        ss.print(allocator, "{d}", .{cl.posGet(i)}) catch oom();
                    }
                    const n = ss.items.len - start;
                    if (n < 5) ss.appendNTimes(allocator, ' ', 5 - n) catch oom();
                    if (i % 256 == 255) ss.appendSlice(allocator, " *") catch oom();
                    if (i % 64 == 63) ss.append(allocator, '\n') catch oom();
                }
                ss.append(allocator, 0) catch oom();
                impl.logDebug("\n%s\n", .{@as([*:0]const u8, @ptrCast(ss.items.ptr))});
            }

            for (0..max_seq) |sq| {
                const sid: i32 = @intCast(sq);
                if (cl.seqPosMin(sid) < 0) continue;
                impl.logDebug("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", .{ "find_slot", stream_id, sid, cl.seqPosMin(sid), sid, cl.seqPosMax(sid) });
            }
        }
    }

    /// Ports `llama_kv_cache`'s `apply_ubatch` (llama-kv-cache.cpp:1097 @5266f24da).
    pub fn applyUbatch(self: *KvCache, sinfo: *const SlotInfo, ubatch: *const Ubatch) void {
        if (self.other != null) return;

        // The largest position of each sequence this ubatch overwrites;
        // always empty without SWA.
        var seq_pos_max_rm: [max_seq]i32 = @splat(-1);

        std.debug.assert(ubatch.n_tokens == sinfo.nStream() * sinfo.size());

        for (0..sinfo.nStream()) |s| {
            for (0..sinfo.size()) |ii| {
                const i = s * sinfo.size() + ii;
                const cl = &self.cells()[@intCast(sinfo.strm.items[s])];
                const idx = sinfo.idxs.items[s].items[ii];

                if (!cl.isEmpty(idx)) {
                    std.debug.assert(cl.seqCount(idx) == 1);
                    const seq_id = cl.seqGet(idx);
                    const pos = cl.posGet(idx);
                    seq_pos_max_rm[@intCast(seq_id)] = @max(seq_pos_max_rm[@intCast(seq_id)], pos);
                    cl.rm(idx);
                }

                cl.posSet(idx, ubatch.pos[i]);

                const hp = self.hparams;
                if (ubatch.isPos2d() or ubatch.token != null or hp.ple_n_heads > 0) {
                    var ext: kv_cells.CellExt = .{};
                    if (ubatch.isPos2d()) {
                        ext.x = ubatch.pos[i + ubatch.n_tokens * 2];
                        ext.y = ubatch.pos[i + ubatch.n_tokens];
                    }
                    if (ubatch.token != null) {
                        ext.tok = ubatch.token[i];
                    } else if (hp.ple_n_heads > 0) {
                        // An embedding (multimodal) batch has no token ids;
                        // the PLE layers want one all the same.
                        ext.tok = @bitCast(if (hp.ple_image_token_id != 0) hp.ple_image_token_id else hp.ple_eos_token_id);
                    }
                    cl.extSet(idx, ext);
                }

                for (0..@intCast(ubatch.n_seq_id[i])) |sq| cl.seqAdd(idx, ubatch.seq_id[i][sq]);
            }
        }

        // Keep every position in [pos_min, pos_max] of each sequence present:
        // purge positions below the ones overwritten.
        for (0..max_seq) |s| {
            if (seq_pos_max_rm[s] == -1) continue;
            impl.assert(s < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 1168, "s < seq_to_stream.size()");
            const cl = &self.cells()[self.seq_to_stream.items[s]];
            if (cl.seqPosMin(@intCast(s)) <= seq_pos_max_rm[s]) {
                impl.logDebug("%s: purging positions [%d, %d] of sequence %d from KV cache\n", .{ "apply_ubatch", cl.seqPosMin(@intCast(s)), seq_pos_max_rm[s], @as(c_int, @intCast(s)) });
                _ = self.seqRm(@intCast(s), cl.seqPosMin(@intCast(s)), seq_pos_max_rm[s] + 1);
            }
        }

        // Move the head past the slot.
        for (0..sinfo.nStream()) |s| {
            const idxs = sinfo.idxs.items[s].items;
            self.v_heads.items[@intCast(sinfo.strm.items[s])] = idxs[idxs.len - 1] + 1;
        }
    }

    /// Ports `llama_kv_cache`'s `get_can_shift` (llama-kv-cache.cpp:1188 @5266f24da).
    pub fn getCanShift(self: *const KvCache) bool {
        // Step35's per-layer RoPE dims break the single global n_rot.
        if (self.model.arch == .step35) return false;
        if (self.hparams.nPosPerEmbd() > 1) return false;
        return true;
    }

    /// Ports `llama_kv_cache`'s `get_size` (llama-kv-cache.cpp:1199 @5266f24da).
    pub fn getSize(self: *const KvCache) u32 {
        return self.cells()[self.seq_to_stream.items[0]].size();
    }

    /// Ports `llama_kv_cache`'s `get_n_stream` (llama-kv-cache.cpp:1205 @5266f24da).
    pub fn getNStream(self: *const KvCache) u32 {
        return self.n_stream;
    }

    /// Ports `llama_kv_cache`'s `get_has_shift` (llama-kv-cache.cpp:1209 @5266f24da).
    pub fn getHasShift(self: *const KvCache) bool {
        var result = false;
        for (self.cells()[0..self.n_stream]) |*cl| result = result or cl.getHasShift();
        return result;
    }

    /// Ports `llama_kv_cache`'s `type_k` (llama-kv-cache.cpp:1219 @5266f24da).
    pub fn typeK(self: *const KvCache) c.enum_ggml_type {
        return self.layers.items[0].k.type;
    }

    /// Ports `llama_kv_cache`'s `type_v` (llama-kv-cache.cpp:1223 @5266f24da).
    pub fn typeV(self: *const KvCache) c.enum_ggml_type {
        return self.layers.items[0].v.?.type;
    }

    /// Ports `llama_kv_cache`'s `get_layer_ids` (llama-kv-cache.cpp:1227
    /// @5266f24da): the model layers the cache holds, in its order.
    ///
    /// Return: the ids, the caller's to free.
    pub fn getLayerIds(self: *const KvCache) std.ArrayList(u32) {
        var res: std.ArrayList(u32) = .empty;
        res.ensureTotalCapacity(allocator, self.layers.items.len) catch oom();
        for (self.layers.items) |*layer| res.appendAssumeCapacity(layer.il);
        return res;
    }

    /// Ports `llama_kv_cache`'s `get_k_storage` (llama-kv-cache.cpp:1238 @5266f24da).
    pub fn getKStorage(self: *const KvCache, il: i32) *Tensor {
        return self.layers.items[@intCast(self.map_layer_ids.get(il).?)].k;
    }

    /// Ports `llama_kv_cache`'s `get_cells` (llama-kv-cache.cpp:1244 @5266f24da).
    pub fn getCells(self: *const KvCache, seq_id: i32) *const KvCells {
        impl.assert(seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len, "llama-kv-cache.cpp", 1245, "seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()");
        return &self.cells()[self.seq_to_stream.items[@intCast(seq_id)]];
    }

    /// Ports `llama_kv_cache`'s `get_n_kv` (llama-kv-cache.cpp:1250 @5266f24da):
    /// the cells to attend over, padded so the graph stays constant.
    pub fn getNKv(self: *const KvCache, sinfo: *const SlotInfo) u32 {
        var result: u32 = 0;
        const n_pad_cur = @max(self.n_pad, 256);
        for (0..sinfo.nStream()) |s| {
            const cl = &self.cells()[@intCast(sinfo.strm.items[s])];
            const padded = (cl.usedMaxP1() + n_pad_cur - 1) / n_pad_cur * n_pad_cur;
            result = @max(@min(cl.size(), @max(n_pad_cur, padded)), result);
        }
        return result;
    }

    fn layerAt(self: *const KvCache, il: i32) *const KvLayer {
        const ikv = self.map_layer_ids.get(il) orelse impl.abort("llama-kv-cache.cpp", 1267, "map::at:  key not found");
        return &self.layers.items[@intCast(ikv)];
    }

    /// Ports `llama_kv_cache`'s `get_k` (llama-kv-cache.cpp:1266 @5266f24da).
    pub fn getK(self: *const KvCache, ctx: *c.ggml_context, il: i32, n_kv: u32, sinfo: *const SlotInfo) *Tensor {
        const k = self.layerAt(il).k;
        const kv_size: u64 = self.getSize();
        const n_embd_k_gqa: u64 = @intCast(k.ne[0]);
        const ns: u32 = sinfo.s1 - sinfo.s0 + 1;
        const ilu: u32 = @intCast(il);
        return c.ggml_view_4d(ctx, k, self.hparams.nEmbdHeadK(ilu), self.hparams.nHeadKv(ilu), n_kv, ns, c.ggml_row_size(k.type, self.hparams.nEmbdHeadK(ilu)), c.ggml_row_size(k.type, @intCast(n_embd_k_gqa)), c.ggml_row_size(k.type, @intCast(n_embd_k_gqa * kv_size)), c.ggml_row_size(k.type, @intCast(n_embd_k_gqa * kv_size)) * sinfo.s0);
    }

    /// Ports `llama_kv_cache`'s `get_v` (llama-kv-cache.cpp:1286 @5266f24da).
    pub fn getV(self: *const KvCache, ctx: *c.ggml_context, il: i32, n_kv: u32, sinfo: *const SlotInfo) *Tensor {
        const v = self.layerAt(il).v.?;
        const kv_size: u64 = self.getSize();
        const n_embd_v_gqa: u64 = @intCast(v.ne[0]);
        const ns: u32 = sinfo.s1 - sinfo.s0 + 1;
        const ilu: u32 = @intCast(il);
        const hv = self.hparams.nEmbdHeadV(ilu);
        if (!self.v_trans) {
            // v->nb[1] <= v->nb[2]
            return c.ggml_view_4d(ctx, v, hv, self.hparams.nHeadKv(ilu), n_kv, ns, c.ggml_row_size(v.type, hv), c.ggml_row_size(v.type, @intCast(n_embd_v_gqa)), c.ggml_row_size(v.type, @intCast(n_embd_v_gqa * kv_size)), c.ggml_row_size(v.type, @intCast(n_embd_v_gqa * kv_size)) * sinfo.s0);
        }
        // v->nb[1] > v->nb[2]
        return c.ggml_view_4d(ctx, v, n_kv, self.hparams.nHeadKv(ilu), hv, ns, c.ggml_row_size(v.type, @intCast(kv_size * hv)), c.ggml_row_size(v.type, @intCast(kv_size)), c.ggml_row_size(v.type, @intCast(kv_size * n_embd_v_gqa)), c.ggml_row_size(v.type, @intCast(kv_size * n_embd_v_gqa)) * sinfo.s0);
    }

    /// Ports `llama_kv_cache`'s `cpy_k` (llama-kv-cache.cpp:1318 @5266f24da):
    /// store the current K values into the cache.
    pub fn cpyK(self: *const KvCache, ctx: *c.ggml_context, k_cur_: *Tensor, k_idxs: *Tensor, il: i32, sinfo: *const SlotInfo) *Tensor {
        _ = sinfo;
        var k = self.layerAt(il).k;
        const n_embd_head = k_cur_.ne[0];
        const n_head = k_cur_.ne[1];
        const n_tokens = k_cur_.ne[2];
        const n_embd_gqa = n_embd_head * n_head;

        // Dims 0 and 1 merge.
        impl.assert(c.ggml_row_size(k_cur_.type, n_embd_head) == k_cur_.nb[1], "llama-kv-cache.cpp", 1333, "ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]");
        const k_cur = c.ggml_view_2d(ctx, k_cur_, n_embd_gqa, n_tokens, k_cur_.nb[2], 0);

        const n_stream = k.ne[2];
        if (n_stream > 1) {
            const kv_size: i64 = self.getSize();
            // The indices are global across streams.
            k = c.ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size * n_stream);
        }
        return c.ggml_set_rows(ctx, k, k_cur, k_idxs);
    }

    /// Ports `llama_kv_cache`'s `cpy_v` (llama-kv-cache.cpp:1353 @5266f24da).
    pub fn cpyV(self: *const KvCache, ctx: *c.ggml_context, v_cur_: *Tensor, v_idxs: *Tensor, il: i32, sinfo: *const SlotInfo) *Tensor {
        _ = sinfo;
        var v = self.layerAt(il).v.?;
        var v_cur = v_cur_;
        const n_embd_head = v_cur.ne[0];
        const n_head = v_cur.ne[1];
        const n_tokens = v_cur.ne[2];
        const n_embd_gqa = n_embd_head * n_head;

        impl.assert(c.ggml_row_size(v_cur.type, n_embd_head) == v_cur.nb[1], "llama-kv-cache.cpp", 1367, "ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]");

        const n_stream = v.ne[2];

        // With FA, the V cache is not transposed.
        if (!self.v_trans) {
            v_cur = c.ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur.nb[2], 0);
            if (n_stream > 1) {
                const kv_size: i64 = self.getSize();
                v = c.ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size * n_stream);
            }
            return c.ggml_set_rows(ctx, v, v_cur, v_idxs);
        }

        if (c.ggml_row_size(v_cur.type, n_embd_gqa) == v_cur.nb[2]) {
            // Dims 0, 1 and 2 merge.
            v_cur = c.ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
        } else {
            v_cur = c.ggml_cont_2d(ctx, v_cur, n_embd_gqa, n_tokens);
        }

        // [TAG_V_CACHE_VARIABLE]
        if (n_embd_gqa < v.ne[0]) v_cur = c.ggml_pad(ctx, v_cur, @intCast(v.ne[0] - n_embd_gqa), 0, 0, 0);

        // Each row of the indices is a single head element.
        const v_view = c.ggml_reshape_2d(ctx, v, 1, c.ggml_nelements(v));
        v_cur = c.ggml_reshape_2d(ctx, v_cur, 1, c.ggml_nelements(v_cur));
        return c.ggml_set_rows(ctx, v_view, v_cur, v_idxs);
    }

    /// Ports `llama_kv_cache`'s `build_input_k_idxs` (llama-kv-cache.cpp:1409 @5266f24da).
    pub fn buildInputKIdxs(self: *const KvCache, ctx: *c.ggml_context, ubatch: *const Ubatch) *Tensor {
        _ = self;
        const k_idxs = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I64, ubatch.n_tokens);
        c.ggml_set_input(k_idxs);
        return k_idxs;
    }

    /// Ports `llama_kv_cache`'s `build_input_v_idxs` (llama-kv-cache.cpp:1419 @5266f24da).
    pub fn buildInputVIdxs(self: *const KvCache, ctx: *c.ggml_context, ubatch: *const Ubatch) *Tensor {
        const n_tokens: i64 = ubatch.n_tokens;
        const v_idxs = if (!self.v_trans)
            c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I64, n_tokens)
        else
            c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I64, n_tokens * self.hparams.nEmbdVGqaMax());
        c.ggml_set_input(v_idxs);
        return v_idxs;
    }

    /// Ports `llama_kv_cache`'s `build_input_k_rot` (llama-kv-cache.cpp:1435 @5266f24da).
    pub fn buildInputKRot(self: *const KvCache, ctx: *c.ggml_context) ?*Tensor {
        if (!self.attn_rot_k) return null;
        var nrot: c_int = 64;
        while (true) {
            nrot *= 2;
            if (@rem(self.n_embd_head_k_all, nrot) != 0) break;
        }
        nrot = @divTrunc(nrot, 2);
        const res = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, nrot, nrot);
        c.ggml_set_input(res);
        _ = c.ggml_set_name(res, "attn_inp_k_rot");
        return res;
    }

    /// Ports `llama_kv_cache`'s `build_input_v_rot` (llama-kv-cache.cpp:1456
    /// @5266f24da): V uses the smallest rotation.
    pub fn buildInputVRot(self: *const KvCache, ctx: *c.ggml_context) ?*Tensor {
        if (!self.attn_rot_v) return null;
        const nrot: c_int = 64;
        const res = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, nrot, nrot);
        c.ggml_set_input(res);
        _ = c.ggml_set_name(res, "attn_inp_v_rot");
        return res;
    }

    /// Ports `llama_kv_cache`'s `set_input_k_idxs` (llama-kv-cache.cpp:1476 @5266f24da).
    pub fn setInputKIdxs(self: *const KvCache, dst: *Tensor, ubatch: *const Ubatch, sinfo: *const SlotInfo) void {
        impl.assert(ubatch.n_tokens == sinfo.size() * sinfo.nStream(), "llama-kv-cache.cpp", 1478, "n_tokens == (int64_t) sinfo.size()*sinfo.n_stream()");
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache.cpp", 1480, "ggml_backend_buffer_is_host(dst->buffer)");
        const data: [*]i64 = @ptrCast(@alignCast(dst.data));
        for (0..sinfo.nStream()) |s| {
            const offs: i64 = @as(i64, sinfo.strm.items[s]) * self.getSize();
            for (0..sinfo.size()) |i| data[s * sinfo.size() + i] = offs + sinfo.idxs.items[s].items[i];
        }
    }

    /// Ports `llama_kv_cache`'s `set_input_v_idxs` (llama-kv-cache.cpp:1492 @5266f24da).
    pub fn setInputVIdxs(self: *const KvCache, dst: *Tensor, ubatch: *const Ubatch, sinfo: *const SlotInfo) void {
        impl.assert(ubatch.n_tokens == sinfo.size() * sinfo.nStream(), "llama-kv-cache.cpp", 1494, "n_tokens == (int64_t) sinfo.size()*sinfo.n_stream()");
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache.cpp", 1496, "ggml_backend_buffer_is_host(dst->buffer)");
        const data: [*]i64 = @ptrCast(@alignCast(dst.data));
        if (!self.v_trans) {
            for (0..sinfo.nStream()) |s| {
                const offs: i64 = @as(i64, sinfo.strm.items[s]) * self.getSize();
                for (0..sinfo.size()) |i| data[s * sinfo.size() + i] = offs + sinfo.idxs.items[s].items[i];
            }
        } else {
            // The V cache is transposed without flash attention.
            const kv_size: i64 = self.getSize();
            const n_embd_v_gqa: usize = self.hparams.nEmbdVGqaMax();
            for (0..sinfo.nStream()) |s| {
                const offs: i64 = @as(i64, sinfo.strm.items[s]) * kv_size * @as(i64, @intCast(n_embd_v_gqa));
                for (0..sinfo.size()) |i| {
                    for (0..n_embd_v_gqa) |j| {
                        data[s * sinfo.size() * n_embd_v_gqa + i * n_embd_v_gqa + j] = offs + @as(i64, @intCast(j)) * kv_size + sinfo.idxs.items[s].items[i];
                    }
                }
            }
        }
    }

    /// Ports `llama_kv_cache`'s `set_input_k_shift` (llama-kv-cache.cpp:1525 @5266f24da).
    pub fn setInputKShift(self: *const KvCache, dst: *Tensor) void {
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache.cpp", 1526, "ggml_backend_buffer_is_host(dst->buffer)");
        const data: [*]i32 = @ptrCast(@alignCast(dst.data));
        for (0..self.n_stream) |s| {
            const cl = &self.cells()[s];
            for (0..cl.size()) |i_| {
                const i: u32 = @intCast(i_);
                data[s * cl.size() + i] = if (cl.isEmpty(i)) 0 else cl.getShift(i);
            }
        }
    }

    /// Ports `llama_kv_cache`'s `set_input_kq_mask` (llama-kv-cache.cpp:1744
    /// @5266f24da), dispatching to `set_input_kq_mask_impl` (llama-kv-cache.cpp:1736
    /// @5266f24da) over the element type and the four flags.
    pub fn setInputKqMask(self: *const KvCache, dst: *Tensor, ubatch: *const Ubatch, causal_attn_: bool) void {
        const n_tokens: i64 = ubatch.n_tokens;
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache.cpp", 1747, "ggml_backend_buffer_is_host(dst->buffer)");
        const n_kv = dst.ne[0];
        // The streams in the current ubatch.
        const n_stream = dst.ne[3];
        impl.assert(@rem(n_tokens, n_stream) == 0, "llama-kv-cache.cpp", 1752, "n_tokens%n_stream == 0");

        // See `llama_non_causal_type`: with SWA_ONLY only the SWA cache, or
        // the SWA layers of a single cache, become non-causal.
        var causal_attn = causal_attn_;
        if (!causal_attn and self.hparams.non_causal_type == non_causal_type_swa_only) {
            causal_attn = self.swa_type == .none;
        }

        const args: MaskArgs = .{
            .hparams = self.hparams,
            .ubatch = ubatch,
            .cells = self.cells(),
            .seq_to_stream = self.seq_to_stream.items,
            .n_swa = self.n_swa,
            .swa_type = self.swa_type,
            .n_kv = n_kv,
            .n_stream = n_stream,
            .n_tps = @divTrunc(n_tokens, n_stream),
        };
        const swa = self.swa_type != .none;
        const is_2d = ubatch.isPos2d();
        const alibi = self.hparams.use_alibi;
        if (dst.type == c.GGML_TYPE_F16) {
            maskDispatch(u16, args, @ptrCast(@alignCast(dst.data)), causal_attn, swa, is_2d, alibi);
        } else {
            maskDispatch(f32, args, @ptrCast(@alignCast(dst.data)), causal_attn, swa, is_2d, alibi);
        }
    }

    /// Ports `llama_kv_cache`'s `set_input_pos_bucket` (llama-kv-cache.cpp:1788 @5266f24da).
    pub fn setInputPosBucket(self: *const KvCache, dst: *Tensor, ubatch: *const Ubatch) void {
        const n_tokens: usize = ubatch.n_tokens;
        impl.assert(self.n_stream == 1, "llama-kv-cache.cpp", 1791, "n_stream == 1 && \"TODO: support multiple streams\"");
        const cl = &self.cells()[0];
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache.cpp", 1794, "ggml_backend_buffer_is_host(dst->buffer)");
        impl.assert(!ubatch.equalSeqs(), "llama-kv-cache.cpp", 1795, "!ubatch->equal_seqs()");
        const data: [*]i32 = @ptrCast(@alignCast(dst.data));
        const n_kv: usize = @intCast(dst.ne[0]);
        for (0..n_tokens) |i| {
            for (0..n_kv) |j| {
                // An empty cell's position is irrelevant; attention masks it.
                const p0: i32 = if (cl.isEmpty(@intCast(j))) -1 else cl.posGet(@intCast(j));
                data[i * n_kv + j] = graph.relativePositionBucket(p0, ubatch.pos[i], self.hparams.n_rel_attn_bkts, false);
            }
        }
    }

    /// Ports `llama_kv_cache`'s `set_input_k_rot` (llama-kv-cache.cpp:1813 @5266f24da).
    pub fn setInputKRot(self: *const KvCache, dst: *Tensor) void {
        self.setInputRot(dst, 1817);
    }

    /// Ports `llama_kv_cache`'s `set_input_v_rot` (llama-kv-cache.cpp:1822 @5266f24da).
    pub fn setInputVRot(self: *const KvCache, dst: *Tensor) void {
        self.setInputRot(dst, 1826);
    }

    fn setInputRot(self: *const KvCache, dst: *Tensor, comptime line: c_int) void {
        impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-kv-cache.cpp", line - 3, "ggml_backend_buffer_is_host(dst->buffer)");
        const m = self.attn_rot_hadamard.get(dst.ne[0]) orelse impl.abort("llama-kv-cache.cpp", line, "GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0])) failed");
        @memcpy(@as([*]u8, @ptrCast(dst.data))[0..c.ggml_nbytes(dst)], std.mem.sliceAsBytes(m)[0..c.ggml_nbytes(dst)]);
    }

    /// Ports `llama_kv_cache`'s `has_cell_ext` (llama-kv-cache.cpp:1831
    /// @5266f24da): whether `llama_kv_cell_ext` holds what a state save and
    /// restore must keep -- M-RoPE's 2D position, or the token the PLE
    /// n-gram hash needs.
    pub fn hasCellExt(self: *const KvCache) bool {
        return self.hparams.nPosPerEmbd() > 1 or self.hparams.ple_n_heads > 0;
    }

    /// Ports `llama_kv_cache`'s `get_prev_tokens` (llama-kv-cache.cpp:1836
    /// @5266f24da): for every token of the ubatch, the ids of the `n`
    /// tokens before it in its sequence, `LLAMA_TOKEN_NULL` where there are
    /// none. `apply_ubatch` has already stored this ubatch, so the cells
    /// cover its own tokens too; the nearest cell at or before a position
    /// resolves M-RoPE's shared positions. The n-gram input embeddings use
    /// it.
    ///
    /// Parameters:
    /// - `res`: replaced with `n_tokens * n` ids, token-major.
    ///
    /// Return: nothing.
    pub fn getPrevTokens(self: *const KvCache, ubatch: *const Ubatch, n: u32, res: *std.ArrayList(i32)) void {
        const n_tokens: u32 = ubatch.n_tokens;

        res.clearRetainingCapacity();
        res.appendNTimes(allocator, -1, @as(usize, n_tokens) * n) catch oom();

        if (n == 0) return;

        // An embedding (multimodal) ubatch can repeat one position for a
        // whole image, so positions do not give the token order: resolve
        // its predecessors by ubatch order instead. The C++ keys an
        // `unordered_map` by sequence and only looks it up.
        var ord: std.ArrayList(u32) = .empty;
        defer ord.deinit(allocator);
        var seq_idx: [max_seq]std.ArrayList(u32) = @splat(.empty);
        defer for (&seq_idx) |*v| v.deinit(allocator);

        if (ubatch.token == null) {
            ord.resize(allocator, n_tokens) catch oom();
            for (0..n_tokens) |i| {
                const v = &seq_idx[@intCast(ubatch.seq_id[i][0])];
                ord.items[i] = @intCast(v.items.len);
                v.append(allocator, @intCast(i)) catch oom();
            }
        }

        for (0..n_tokens) |i| {
            // A token of more than one sequence has an ambiguous history;
            // the n-gram architectures reject such batches.
            const seq_id = ubatch.seq_id[i][0];

            for (0..n) |j| {
                const d: i32 = @intCast(n - j);

                var p: i32 = undefined;
                if (ubatch.token == null) {
                    const v = seq_idx[@intCast(seq_id)].items;
                    const k: i64 = @as(i64, ord.items[i]) - d;
                    // k >= 0: an earlier token of this ubatch; else before the chunk.
                    p = if (k >= 0) ubatch.pos[v[@intCast(k)]] else ubatch.pos[v[0]] +% @as(i32, @intCast(k));
                } else {
                    p = ubatch.pos[i] -% d;
                }

                if (p < 0) continue;

                res.items[i * n + j] = self.cells()[self.seq_to_stream.items[@intCast(seq_id)]].seqPosTokLe(seq_id, p);
            }
        }
    }

    /// Ports `llama_kv_cache`'s `total_size` (llama-kv-cache.cpp:1892 @5266f24da).
    pub fn totalSize(self: *const KvCache) usize {
        var size: usize = 0;
        for (self.ctxs_bufs.items) |e| size += c.ggml_backend_buffer_get_size(e.buf);
        return size;
    }

    /// Ports `llama_kv_cache`'s `size_k_bytes` (llama-kv-cache.cpp:1902 @5266f24da).
    fn sizeKBytes(self: *const KvCache) usize {
        var size: usize = 0;
        for (self.layers.items) |l| size += c.ggml_nbytes(l.k);
        return size;
    }

    /// Ports `llama_kv_cache`'s `size_v_bytes` (llama-kv-cache.cpp:1912 @5266f24da).
    fn sizeVBytes(self: *const KvCache) usize {
        var size: usize = 0;
        for (self.layers.items) |l| size += if (l.v) |v| c.ggml_nbytes(v) else 0;
        return size;
    }

    /// Ports `llama_kv_cache`'s `build_rope_shift` (llama-kv-cache.cpp:1922 @5266f24da).
    fn buildRopeShift(self: *const KvCache, cparams: *const Cparams, ctx: *c.ggml_context, cur: *Tensor, shift: *Tensor, rot: ?*Tensor, factors: ?*Tensor, freq_base: f32, freq_scale: f32, il: u32) *Tensor {
        const n_ctx_orig: c_int = @bitCast(cparams.n_ctx_orig_yarn);
        const n_rot: c_int = @intCast(self.hparams.nRot(il));
        // For M-RoPE the whole vector rotates on a shift; NEOX ordering does it.
        const rope_type: c_int = if (self.hparams.rope_type == c.LLAMA_ROPE_TYPE_MROPE or self.hparams.rope_type == c.LLAMA_ROPE_TYPE_IMROPE) c.LLAMA_ROPE_TYPE_NEOX else self.hparams.rope_type;
        if (c.ggml_is_quantized(cur.type)) {
            // Dequantize, RoPE, quantize back.
            var tmp = c.ggml_cast(ctx, cur, c.GGML_TYPE_F32);
            tmp = graph.mulMatHadamard(ctx, tmp, rot.?);
            tmp = c.ggml_rope_ext(ctx, tmp, shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, cparams.yarn_ext_factor, cparams.yarn_attn_factor, cparams.yarn_beta_fast, cparams.yarn_beta_slow);
            tmp = graph.mulMatHadamard(ctx, tmp, rot.?);
            return c.ggml_cpy(ctx, tmp, cur);
        }
        // Only the first n_rot dimensions rotate.
        return c.ggml_rope_ext_inplace(ctx, cur, shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, cparams.yarn_ext_factor, cparams.yarn_attn_factor, cparams.yarn_beta_fast, cparams.yarn_beta_slow);
    }

    /// Ports `llama_kv_cache`'s `build_graph_shift` (llama-kv-cache.cpp:2001 @5266f24da).
    fn buildGraphShift(self: *const KvCache, res: *graph.Result, lctx: *context_mod.Context) *c.ggml_cgraph {
        impl.assert(self.other == null, "llama-kv-cache.cpp", 2003, "!other");
        const ctx = res.getCtx();
        const gf = res.getGf();

        const inp = allocator.create(InpKShift) catch oom();
        inp.* = .{ .kv_self = self };
        inp.k_shift = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, @as(i64, self.getSize()) * self.n_stream);
        c.ggml_set_input(inp.k_shift);
        inp.k_rot = self.buildInputKRot(ctx);

        const cparams = lctx.getCparams();
        for (self.layers.items) |*layer| {
            const il = layer.il;
            if (!self.hparams.hasRope(il)) continue;
            const n_head_kv: i64 = self.hparams.nHeadKv(il);
            const n_embd_k_gqa: i64 = self.hparams.nEmbdKGqa(il);
            const n_rot: i64 = self.hparams.nRot(il);
            const n_embd_head_k: i64 = self.hparams.nEmbdHeadK(il);
            const n_embd_nope: i64 = if (self.hparams.n_lora_kv > 0) n_embd_head_k - n_rot else 0;

            const freq_base_l = self.model.getRopeFreqBase(cparams, il);
            const freq_scale_l = self.model.getRopeFreqScale(cparams, il);
            const rope_factors = self.model.getRopeFactors(cparams, il);

            const k = c.ggml_view_3d(ctx, layer.k, n_rot, n_head_kv, @as(i64, self.getSize()) * self.n_stream, c.ggml_row_size(layer.k.type, n_embd_head_k), c.ggml_row_size(layer.k.type, n_embd_k_gqa), c.ggml_row_size(layer.k.type, n_embd_nope));
            const cur = self.buildRopeShift(cparams, ctx, k, inp.k_shift.?, inp.k_rot, rope_factors, freq_base_l, freq_scale_l, il);
            c.ggml_build_forward_expand(gf, cur);
        }
        _ = res.addInput(InpKShift, inp);
        return gf;
    }

    /// Ports `llama_kv_cache`'s `state_write` (llama-kv-cache.cpp:2053 @5266f24da).
    pub fn stateWrite(self: *const KvCache, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        _ = flags;
        if (self.other != null) return;
        try w.writeValue(self.n_stream, exc);
        for (0..self.n_stream) |s| {
            var cr: CellRanges = .{ .strm = @intCast(s) };
            defer cr.data.deinit(allocator);
            var cell_count: u32 = 0;
            const cl = &self.cells()[s];

            // The ranges of cells holding the sequence, or any when -1.
            var cell_range_begin: u32 = cl.size();
            for (0..cl.size()) |i_| {
                const i: u32 = @intCast(i_);
                var add_cell = !cl.isEmpty(i) and (seq_id == -1 or cl.seqHas(i, seq_id));
                // A cell the SWA masks is not saved.
                if (add_cell and seq_id != -1) {
                    add_cell = !Hparams.isMaskedSwa(self.n_swa, self.swa_type, cl.posGet(i), cl.seqPosMax(seq_id));
                }
                if (add_cell) {
                    cell_count += 1;
                    if (cell_range_begin == cl.size()) cell_range_begin = i;
                } else if (cell_range_begin != cl.size()) {
                    cr.data.append(allocator, .{ cell_range_begin, i }) catch oom();
                    cell_range_begin = cl.size();
                }
            }
            if (cell_range_begin != cl.size()) cr.data.append(allocator, .{ cell_range_begin, cl.size() }) catch oom();

            var check: u32 = 0;
            for (cr.data.items) |r| check += r[1] - r[0];
            impl.assert(cell_count == check, "llama-kv-cache.cpp", 2109, "cell_count == cell_count_check");

            try w.writeValue(cell_count, exc);
            if (cell_count == 0) continue;
            try self.stateWriteMeta(w, &cr, seq_id, exc);
            try self.stateWriteData(w, &cr, exc);
        }
    }

    const CellRanges = struct {
        strm: u32,
        /// `[from, to)`.
        data: std.ArrayList([2]u32) = .empty,
    };

    /// Ports `llama_kv_cache`'s `state_read` (llama-kv-cache.cpp:2123 @5266f24da).
    pub fn stateRead(self: *KvCache, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        return self.stateReadSinfo(r, seq_id, flags, null, null, exc);
    }

    /// Ports `llama_kv_cache`'s `state_read_sinfo` (llama-kv-cache.cpp:2127
    /// @5266f24da): `state_read`, plus the cells the restored tokens went
    /// to. A cache mirroring another (the qwen4exp indexer) must not search
    /// for cells of its own: two searches agree only by luck.
    ///
    /// Parameters:
    /// - `sinfos_out`: when set, replaced with the layout used, one entry
    ///   per stream; a stream with no cells leaves an empty one. The
    ///   caller's to `deinit`.
    /// - `sinfos_in`: when set, the layout to use instead of searching, one
    ///   entry per stream, each holding the blob's cell count.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn stateReadSinfo(self: *KvCache, r: io.Reader, seq_id: i32, flags: u32, sinfos_out: ?*std.ArrayList(SlotInfo), sinfos_in: ?[]const SlotInfo, exc: *Exception) Thrown!void {
        _ = flags;
        if (self.other != null) return;
        impl.assert(seq_id == -1 or (seq_id >= 0 and @as(usize, @intCast(seq_id)) < self.seq_to_stream.items.len), "llama-kv-cache.cpp", 2141, "seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size())");

        if (sinfos_out) |out| {
            for (out.items) |*si| si.deinit();
            out.clearRetainingCapacity();
            out.appendNTimes(allocator, .{}, self.n_stream) catch oom();
        }

        if (sinfos_in) |in| {
            if (in.len != self.n_stream) return exc.throw(.runtime_error, "failed to restore kv cache: mirrored slot layout has the wrong stream count", .{});
        }

        const n_stream_cur = try r.readValue(u32, exc);
        if (n_stream_cur != self.n_stream) return exc.throw(.runtime_error, "n_stream mismatch", .{});

        // A whole-context restore replaces every stream: empty the cache
        // once here, since `clear` resets all streams at once.
        if (seq_id == -1) self.clear(true);

        for (0..self.n_stream) |s| {
            const cell_count = try r.readValue(u32, exc);
            if (cell_count == 0) {
                // A mirrored cache must be empty here too, or the two no
                // longer agree cell for cell.
                if (sinfos_in) |in| {
                    if (!in[s].isEmpty()) return exc.throw(.runtime_error, "failed to restore kv cache: mirrored cache holds cells this one does not", .{});
                }
                continue;
            }
            const strm: u32 = if (seq_id == -1) @intCast(s) else self.seq_to_stream.items[@intCast(seq_id)];

            var sinfo: SlotInfo = .{};
            defer sinfo.deinit();

            var res = try self.stateReadMeta(r, strm, cell_count, &sinfo, seq_id, if (sinfos_in) |in| &in[s] else null, exc);
            if (res) {
                // The C++ catches any exception from reading the data.
                var inner: Exception = .{};
                res = self.stateReadData(r, strm, cell_count, &sinfo, &inner) catch false;
            }
            if (!res) {
                if (seq_id == -1) self.clear(true) else _ = self.seqRm(seq_id, -1, -1);
                return exc.throw(.runtime_error, "failed to restore kv cache", .{});
            }

            if (sinfos_out) |out| {
                out.items[s] = sinfo;
                sinfo = .{};
            }
        }
    }

    /// Ports `llama_kv_cache`'s `state_write_meta` (llama-kv-cache.cpp:2203 @5266f24da).
    fn stateWriteMeta(self: *const KvCache, w: io.Writer, cr: *const CellRanges, seq_id: i32, exc: *Exception) Thrown!void {
        const cl = &self.cells()[cr.strm];
        var seq_ids: std.ArrayList(i32) = .empty;
        defer seq_ids.deinit(allocator);
        for (cr.data.items) |range| {
            var i = range[0];
            while (i < range[1]) : (i += 1) {
                seq_ids.clearRetainingCapacity();
                var cur: i32 = 0;
                while (cur < @as(i32, @intCast(self.n_seq_max))) : (cur += 1) {
                    if (cur == seq_id or seq_id == -1) {
                        if (cl.seqHas(i, cur)) seq_ids.append(allocator, cur) catch oom();
                    }
                }
                const pos = cl.posGet(i);
                const n_seq_id: u32 = @intCast(seq_ids.items.len);
                try w.writeValue(pos, exc);
                try w.writeValue(n_seq_id, exc);
                if (self.hasCellExt()) try w.writeValue(cl.extGet(i), exc);
                for (seq_ids.items) |sid| try w.writeValue(sid, exc);
            }
        }
    }

    /// Ports `llama_kv_cache`'s `state_write_data` (llama-kv-cache.cpp:2236 @5266f24da).
    fn stateWriteData(self: *const KvCache, w: io.Writer, cr: *const CellRanges, exc: *Exception) Thrown!void {
        const cl = &self.cells()[cr.strm];
        const v_trans: u32 = @intFromBool(self.v_trans);
        const n_layer: u32 = @intCast(self.layers.items.len);
        try w.writeValue(v_trans, exc);
        try w.writeValue(n_layer, exc);

        // All the keys first, a row per cell, a range at a time.
        for (self.layers.items) |*layer| {
            const n_embd_k_gqa: i64 = self.hparams.nEmbdKGqa(layer.il);
            const k = layer.k_stream.items[cr.strm];
            const k_type_i: i32 = @intCast(k.type);
            try w.writeValue(k_type_i, exc);
            const k_size_row: u64 = c.ggml_row_size(k.type, n_embd_k_gqa);
            try w.writeValue(k_size_row, exc);
            for (cr.data.items) |range| {
                const range_size: usize = range[1] - range[0];
                try w.writeTensor(k, range[0] * k_size_row, range_size * k_size_row, exc);
            }
        }

        if (!self.v_trans) {
            for (self.layers.items) |*layer| {
                const n_embd_v_gqa: i64 = self.hparams.nEmbdVGqa(layer.il);
                const v = layer.v_stream.items[cr.strm] orelse continue;
                const v_type_i: i32 = @intCast(v.type);
                try w.writeValue(v_type_i, exc);
                const v_size_row: u64 = c.ggml_row_size(v.type, n_embd_v_gqa);
                try w.writeValue(v_size_row, exc);
                for (cr.data.items) |range| {
                    const range_size: usize = range[1] - range[0];
                    try w.writeTensor(v, range[0] * v_size_row, range_size * v_size_row, exc);
                }
            }
        } else {
            // Transposed: each row's elements for every range.
            const kv_size: u32 = cl.size();
            for (self.layers.items) |*layer| {
                const n_embd_v_gqa: u32 = self.hparams.nEmbdVGqa(layer.il);
                const v = layer.v_stream.items[cr.strm] orelse continue;
                const v_type_i: i32 = @intCast(v.type);
                try w.writeValue(v_type_i, exc);
                const v_size_el: u32 = @intCast(c.ggml_type_size(v.type));
                try w.writeValue(v_size_el, exc);
                try w.writeValue(n_embd_v_gqa, exc);
                for (0..n_embd_v_gqa) |j| {
                    for (cr.data.items) |range| {
                        const range_size: usize = range[1] - range[0];
                        const src_offset = (range[0] + j * kv_size) * v_size_el;
                        try w.writeTensor(v, src_offset, range_size * v_size_el, exc);
                    }
                }
            }
        }
    }

    /// Ports `llama_kv_cache`'s `state_read_meta` (llama-kv-cache.cpp:2335
    /// @5266f24da). `sinfo_in`, when set, replaces the `find_slot`: the
    /// caller gives the cells.
    ///
    /// Return: false on an inconsistency, which the C++ logs; `error.Thrown`
    /// when the stream runs out.
    fn stateReadMeta(self: *KvCache, r: io.Reader, strm: u32, cell_count: u32, sinfo: *SlotInfo, dest_seq_id_: i32, sinfo_in: ?*const SlotInfo, exc: *Exception) Thrown!bool {
        const func = "state_read_meta";
        var dest_seq_id = dest_seq_id_;
        const cl = &self.cells()[strm];
        const head = &self.v_heads.items[strm];

        if (dest_seq_id != -1) {
            // A single sequence.
            _ = self.seqRm(dest_seq_id, -1, -1);

            var balloc = batch.BatchAllocr.init(self.hparams.nPosPerEmbd());
            defer balloc.deinit();
            var ubatch = balloc.ubatchReserve(cell_count, 1);
            defer ubatch.release();
            ubatch.seq_id_unq[0] = dest_seq_id;

            // The ext as it was saved, put back after `applyUbatch`.
            var exts: std.ArrayList(kv_cells.CellExt) = .empty;
            defer exts.deinit(allocator);
            if (self.hasCellExt()) exts.appendNTimes(allocator, .{}, cell_count) catch oom();

            for (0..cell_count) |i| {
                const pos = try r.readValue(i32, exc);
                const n_seq_id = try r.readValue(u32, exc);
                if (n_seq_id != 1) {
                    impl.logError("%s: invalid seq_id-agnostic kv cell\n", .{func});
                    return false;
                }
                if (self.hasCellExt()) {
                    const ext = try r.readValue(kv_cells.CellExt, exc);
                    if (self.hparams.nPosPerEmbd() > 1) {
                        ubatch.pos[i + ubatch.n_tokens] = ext.y;
                        ubatch.pos[i + ubatch.n_tokens * 2] = ext.x;
                    }
                    // `applyUbatch` restores ext.tok from the ubatch's tokens.
                    ubatch.token[i] = ext.tok;
                    exts.items[i] = ext;
                }
                // Read the sequence id and discard it: dest_seq_id replaces it.
                _ = try r.readValue(i32, exc);
                ubatch.pos[i] = pos;
                ubatch.n_seq_id[i] = @intCast(n_seq_id);
                ubatch.seq_id[i] = @ptrCast(&dest_seq_id);
            }

            if (sinfo_in) |in| {
                // This cache mirrors another: it takes that cache's layout
                // instead of searching for its own cells.
                if (in.isEmpty() or in.nStream() != 1 or in.idxs.items[0].items.len != cell_count) {
                    impl.logError("%s: mirrored slot layout holds %d cells, this cache restores %d\n", .{ func, @as(c_int, if (in.isEmpty()) 0 else @intCast(in.idxs.items[0].items.len)), cell_count });
                    return false;
                }

                sinfo.deinit();
                sinfo.* = in.clone();

                // The layout is cell indices: it means the same in both
                // caches only while their streams line up.
                sinfo.s0 = strm;
                sinfo.s1 = strm;
                sinfo.strm.items[0] = @intCast(strm);

                // `seqRm` above freed exactly the cells this sequence held;
                // anything else in the way is a cache already drifted, which
                // this restore must not hide.
                for (0..cell_count) |i| {
                    const idx = sinfo.idxs.items[0].items[i];
                    if (idx >= cl.size() or !cl.isEmpty(idx)) {
                        impl.logError("%s: cell %u of the mirrored slot layout is not free\n", .{ func, idx });
                        return false;
                    }
                }
            } else {
                sinfo.* = self.findSlot(&ubatch, false);
                if (sinfo.isEmpty()) {
                    impl.logError("%s: failed to find %d available cells in kv cache\n", .{ func, cell_count });
                    return false;
                }
            }

            // `applyUbatch` rebuilds the ext from the ubatch; only ext.tok
            // and M-RoPE's 2D position round-trip through it.
            self.applyUbatch(sinfo, &ubatch);

            // That ubatch is built with this cache's own n_pos_per_embd: a
            // cache without M-RoPE mirroring one with it (the qwen4exp QSA
            // indexer) would drop x and y. Put the saved ext back, as the
            // whole-context path does.
            for (exts.items, 0..) |ext, i| cl.extSet(sinfo.idxs.items[0].items[i], ext);

            impl.logDebug("%s: cell_count = %d, dest_seq_id = %d\n", .{ func, cell_count, dest_seq_id });

            // Every cell was allocated with the right sequence and position.
            impl.assert(sinfo.nStream() == 1, "llama-kv-cache.cpp", 2442, "sinfo.n_stream() == 1");
            impl.assert(sinfo.idxs.items[0].items.len == cell_count, "llama-kv-cache.cpp", 2443, "sinfo.idxs[0].size() == cell_count");
            for (0..cell_count) |i| {
                const idx = sinfo.idxs.items[0].items[i];
                impl.assert(cl.posGet(idx) == ubatch.pos[i], "llama-kv-cache.cpp", 2446, "cells.pos_get(idx) == ubatch.pos[i]");
                impl.assert(cl.seqHas(idx, dest_seq_id), "llama-kv-cache.cpp", 2447, "cells.seq_has(idx, dest_seq_id)");
            }
        } else {
            // The whole cache.
            if (cell_count > cl.size()) {
                impl.logError("%s: not enough cells in kv cache\n", .{func});
                return false;
            }

            // The cells go in from 0, so a mirrored cache lands on the same
            // ones as long as it restores the same count.
            if (sinfo_in) |in| {
                if (in.isEmpty() or in.nStream() != 1 or in.idxs.items[0].items.len != cell_count) {
                    impl.logError("%s: mirrored slot layout holds %d cells, this cache restores %d\n", .{ func, @as(c_int, if (in.isEmpty()) 0 else @intCast(in.idxs.items[0].items.len)), cell_count });
                    return false;
                }
            }

            for (0..cell_count) |i_| {
                const i: u32 = @intCast(i_);
                const pos = try r.readValue(i32, exc);
                const n_seq_id = try r.readValue(u32, exc);
                cl.posSet(i, pos);
                if (self.hasCellExt()) cl.extSet(i, try r.readValue(kv_cells.CellExt, exc));
                for (0..n_seq_id) |_| {
                    const seq_id = try r.readValue(i32, exc);
                    if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= self.n_seq_max) {
                        impl.logError("%s: invalid seq_id, %d is out of range [0, %u)\n", .{ func, seq_id, self.n_seq_max });
                        return false;
                    }
                    cl.seqAdd(i, seq_id);
                }
            }
            // A contiguous slot for the whole cache.
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.resize(1);
            sinfo.strm.items[0] = @intCast(strm);
            sinfo.idxs.items[0].resize(allocator, cell_count) catch oom();
            for (sinfo.idxs.items[0].items, 0..) |*x, i| x.* = @intCast(i);
            head.* = 0;
        }
        return true;
    }

    /// Ports `llama_kv_cache`'s `state_read_data` (llama-kv-cache.cpp:2508
    /// @5266f24da): the scatter reads are batched per contiguous run of
    /// destination cells, `[from, to)` as the cell ranges are.
    fn stateReadData(self: *KvCache, r: io.Reader, strm: u32, cell_count: u32, sinfo: *const SlotInfo, exc: *Exception) Thrown!bool {
        const func = "state_read_data";
        const cl = &self.cells()[strm];

        var runs: std.ArrayList([2]u32) = .empty;
        defer runs.deinit(allocator);
        if (cell_count > 0) {
            const idxs = sinfo.idxs.items[0].items;
            var i_0: u32 = 0;
            while (i_0 < cell_count) {
                var i_1 = i_0 + 1;
                while (i_1 < cell_count and idxs[i_1] == idxs[i_1 - 1] +% 1) i_1 += 1;
                runs.append(allocator, .{ idxs[i_0], idxs[i_1 - 1] +% 1 }) catch oom();
                i_0 = i_1;
            }
        }

        const v_trans = try r.readValue(u32, exc);
        const n_layer = try r.readValue(u32, exc);

        if (n_layer != self.layers.items.len) {
            impl.logError("%s: mismatched layer count (%u instead of %u)\n", .{ func, n_layer, @as(u32, @intCast(self.layers.items.len)) });
            return false;
        }
        if (cell_count > cl.size()) {
            impl.logError("%s: not enough cells in kv cache to restore state (%u > %u)\n", .{ func, cell_count, cl.size() });
            return false;
        }
        if (self.v_trans != (v_trans != 0)) {
            impl.logError("%s: incompatible V transposition\n", .{func});
            return false;
        }

        // Each layer's keys, a row per cell.
        for (self.layers.items) |*layer| {
            const il = layer.il;
            const n_embd_k_gqa: i64 = self.hparams.nEmbdKGqa(il);
            const k = layer.k_stream.items[strm];
            const k_type_i_ref = try r.readValue(i32, exc);
            const k_type_i: i32 = @intCast(k.type);
            if (k_type_i != k_type_i_ref) {
                impl.logError("%s: mismatched key type (%d != %d, layer %d)\n", .{ func, k_type_i, k_type_i_ref, il });
                return false;
            }
            const k_size_row_ref = try r.readValue(u64, exc);
            const k_size_row = c.ggml_row_size(k.type, n_embd_k_gqa);
            if (k_size_row != k_size_row_ref) {
                impl.logError("%s: mismatched key row size (%zu != %zu, layer %d)\n", .{ func, k_size_row, @as(usize, @intCast(k_size_row_ref)), il });
                return false;
            }
            for (runs.items) |run| try r.readTensor(k, @as(usize, run[0]) * k_size_row, @as(usize, run[1] - run[0]) * k_size_row, exc);
        }

        if (!self.v_trans) {
            for (self.layers.items) |*layer| {
                const il = layer.il;
                const n_embd_v_gqa: i64 = self.hparams.nEmbdVGqa(il);
                const v = layer.v_stream.items[strm] orelse continue;
                const v_type_i_ref = try r.readValue(i32, exc);
                const v_type_i: i32 = @intCast(v.type);
                if (v_type_i != v_type_i_ref) {
                    impl.logError("%s: mismatched value type (%d != %d, layer %d)\n", .{ func, v_type_i, v_type_i_ref, il });
                    return false;
                }
                const v_size_row_ref = try r.readValue(u64, exc);
                const v_size_row = c.ggml_row_size(v.type, n_embd_v_gqa);
                if (v_size_row != v_size_row_ref) {
                    impl.logError("%s: mismatched value row size (%zu != %zu, layer %d)\n", .{ func, v_size_row, @as(usize, @intCast(v_size_row_ref)), il });
                    return false;
                }
                for (runs.items) |run| try r.readTensor(v, @as(usize, run[0]) * v_size_row, @as(usize, run[1] - run[0]) * v_size_row, exc);
            }
        } else {
            // Each layer's values, transposed.
            for (self.layers.items) |*layer| {
                const il = layer.il;
                const n_embd_v_gqa: u32 = self.hparams.nEmbdVGqa(il);
                const v = layer.v_stream.items[strm] orelse continue;
                const v_type_i_ref = try r.readValue(i32, exc);
                const v_type_i: i32 = @intCast(v.type);
                if (v_type_i != v_type_i_ref) {
                    impl.logError("%s: mismatched value type (%d != %d, layer %d)\n", .{ func, v_type_i, v_type_i_ref, il });
                    return false;
                }
                const v_size_el_ref = try r.readValue(u32, exc);
                const v_size_el = c.ggml_type_size(v.type);
                if (v_size_el != v_size_el_ref) {
                    impl.logError("%s: mismatched value element size (%zu != %zu, layer %d)\n", .{ func, v_size_el, @as(usize, v_size_el_ref), il });
                    return false;
                }
                const n_embd_v_gqa_ref = try r.readValue(u32, exc);
                if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                    impl.logError("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", .{ func, n_embd_v_gqa, n_embd_v_gqa_ref, il });
                    return false;
                }
                for (0..n_embd_v_gqa) |j| {
                    for (runs.items) |run| try r.readTensor(v, (@as(usize, run[0]) + j * cl.size()) * v_size_el, @as(usize, run[1] - run[0]) * v_size_el, exc);
                }
            }
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *KvCache) memory.Memory {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: memory.Memory.VTable = .{
        .initBatch = vtInitBatch,
        .initFull = vtInitFull,
        .initUpdate = vtInitUpdate,
        .getCanShift = vtGetCanShift,
        .clear = vtClear,
        .seqRm = vtSeqRm,
        .seqCp = vtSeqCp,
        .seqKeep = vtSeqKeep,
        .seqAdd = vtSeqAdd,
        .seqDiv = vtSeqDiv,
        .seqPosMin = vtSeqPosMin,
        .seqPosMax = vtSeqPosMax,
        .memoryBreakdown = vtMemoryBreakdown,
        .stateWrite = vtStateWrite,
        .stateRead = vtStateRead,
        .destroy = vtDestroy,
    };

    fn from(p: *anyopaque) *KvCache {
        return @ptrCast(@alignCast(p));
    }
    fn vtInitBatch(p: *anyopaque, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) memory.Context {
        return from(p).initBatch(balloc, n_ubatch, embd_all).contextOf();
    }
    fn vtInitFull(p: *anyopaque) memory.Context {
        return from(p).initFull().contextOf();
    }
    fn vtInitUpdate(p: *anyopaque, lctx: *anyopaque, optimize: bool) memory.Context {
        return from(p).initUpdate(lctx, optimize).contextOf();
    }
    fn vtGetCanShift(p: *anyopaque) bool {
        return from(p).getCanShift();
    }
    fn vtClear(p: *anyopaque, data: bool) void {
        from(p).clear(data);
    }
    fn vtSeqRm(p: *anyopaque, seq_id: i32, p0: i32, p1: i32) bool {
        return from(p).seqRm(seq_id, p0, p1);
    }
    fn vtSeqCp(p: *anyopaque, a: i32, b: i32, p0: i32, p1: i32) void {
        from(p).seqCp(a, b, p0, p1);
    }
    fn vtSeqKeep(p: *anyopaque, seq_id: i32) void {
        from(p).seqKeep(seq_id);
    }
    fn vtSeqAdd(p: *anyopaque, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        from(p).seqAdd(seq_id, p0, p1, shift);
    }
    fn vtSeqDiv(p: *anyopaque, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        from(p).seqDiv(seq_id, p0, p1, d);
    }
    fn vtSeqPosMin(p: *anyopaque, seq_id: i32) i32 {
        return from(p).seqPosMin(seq_id);
    }
    fn vtSeqPosMax(p: *anyopaque, seq_id: i32) i32 {
        return from(p).seqPosMax(seq_id);
    }
    fn vtMemoryBreakdown(p: *anyopaque, out: *std.ArrayList(memory.BuftSize)) void {
        from(p).memoryBreakdown(out);
    }
    fn vtStateWrite(p: *anyopaque, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        return from(p).stateWrite(w, seq_id, flags, exc);
    }
    fn vtStateRead(p: *anyopaque, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        return from(p).stateRead(r, seq_id, flags, exc);
    }
    fn vtDestroy(p: *anyopaque) void {
        from(p).destroy();
    }
};

fn releaseUbatches(list: *std.ArrayList(Ubatch)) void {
    for (list.items) |*u| u.release();
    list.deinit(allocator);
}

/// Mirrors `llm_graph_input_k_shift` (llama-kv-cache.cpp:1974 @5266f24da).
const InpKShift = struct {
    /// I32 `[kv_size*n_stream]`.
    k_shift: ?*Tensor = null,
    /// Assumes `k_rot^2 == I`.
    k_rot: ?*Tensor = null,
    kv_self: *const KvCache,

    /// Ports `llm_graph_input_k_shift`'s `set_input` (llama-kv-cache.cpp:1989
    /// @5266f24da): a rotation the scheduler gave no buffer is not set.
    pub fn setInput(self: *InpKShift, _: ?*const Ubatch) void {
        if (self.k_shift) |t| self.kv_self.setInputKShift(t);
        if (self.k_rot) |t| {
            if (t.buffer != null) self.kv_self.setInputKRot(t);
        }
    }
};

// -----------------------------------------------------------------------------
// The KQ mask

/// Mirrors `args_set_input_kq_mask` (llama-kv-cache.cpp:1539 @5266f24da).
const MaskArgs = struct {
    hparams: *const Hparams,
    ubatch: *const Ubatch,
    cells: []const KvCells,
    seq_to_stream: []const u32,
    n_swa: u32,
    swa_type: hparams_mod.SwaType,
    n_kv: i64,
    n_stream: i64,
    n_tps: i64,
};

fn maskDispatch(comptime T: type, args: MaskArgs, data: [*]T, causal: bool, swa: bool, is_2d: bool, alibi: bool) void {
    inline for (.{ false, true }) |cz| {
        inline for (.{ false, true }) |sw| {
            inline for (.{ false, true }) |d2| {
                inline for (.{ false, true }) |al| {
                    if (causal == cz and swa == sw and is_2d == d2 and alibi == al) {
                        return setInputKqMaskImpl(T, cz, sw, d2, al, args, data);
                    }
                }
            }
        }
    }
}

fn maskCast(comptime T: type, v: f32) T {
    if (T == u16) return c.ggml_fp32_to_fp16(v);
    return v;
}

/// Ports `set_input_kq_mask_impl` (llama-kv-cache.cpp:1555 @5266f24da).
///
/// For tokens of the same sequence the mask is mostly the same, so the row
/// of the first token of each sequence is copied and only the cells near
/// the batch's positions are revisited -- not with ALiBi.
fn setInputKqMaskImpl(comptime T: type, comptime causal: bool, comptime swa: bool, comptime is_2d: bool, comptime alibi: bool, args: MaskArgs, data: [*]T) void {
    const ubatch = args.ubatch;
    const n_kv: usize = @intCast(args.n_kv);
    const n_tps: usize = @intCast(args.n_tps);
    const mask_keep = maskCast(T, 0.0);
    const mask_drop = maskCast(T, -std.math.inf(f32));

    // The smallest position of each sequence in the batch.
    var seq_pos_min: [max_seq]i32 = @splat(std.math.maxInt(i32));
    for (0..ubatch.n_tokens) |i| {
        const seq_id: usize = @intCast(ubatch.seq_id[i][0]);
        seq_pos_min[seq_id] = @min(seq_pos_min[seq_id], ubatch.pos[i]);
    }

    // Per stream: each sequence's source row and its revisited cells. The
    // C++ keys `std`'s `unordered_map`s by sequence and only looks them up.
    var seq_srct: [max_seq]?u32 = undefined;
    var seq_idxs: [max_seq]std.ArrayList(u32) = @splat(.empty);
    defer for (&seq_idxs) |*v| v.deinit(allocator);

    for (0..@intCast(args.n_stream)) |s| {
        seq_srct = @splat(null);
        for (&seq_idxs) |*v| v.clearRetainingCapacity();

        for (0..n_tps) |ii| {
            const i = s * n_tps + ii;
            const seq_id = ubatch.seq_id[i][0];
            const sid: usize = @intCast(seq_id);
            const cl = &args.cells[args.seq_to_stream[sid]];

            var p0: i32 = -1;
            const p1 = ubatch.pos[i];

            // For M-RoPE.
            const p1_x: i32 = if (is_2d) ubatch.pos[i + ubatch.n_tokens * 2] else 0;
            const p1_y: i32 = if (is_2d) ubatch.pos[i + ubatch.n_tokens] else 0;

            const idst = n_kv * i;
            var prev = false;
            const idxs = &seq_idxs[sid];

            if (!alibi) {
                if (seq_srct[sid]) |srct| {
                    const idst_prev = n_kv * srct;
                    @memcpy(data[idst .. idst + n_kv], data[idst_prev .. idst_prev + n_kv]);
                    prev = true;
                } else {
                    idxs.clearRetainingCapacity();
                    idxs.ensureTotalCapacity(allocator, ubatch.n_tokens + args.n_swa + 32) catch oom();
                    seq_srct[sid] = @intCast(i);
                }
            }

            var jj: usize = 0;
            while (jj < n_kv) : (jj += 1) {
                var j: u32 = @intCast(jj);
                // An existing mask for this sequence: update only its cells.
                if (!alibi and prev) {
                    if (jj >= idxs.items.len) break;
                    j = idxs.items[jj];
                }

                const keep: bool = blk: {
                    if (cl.isEmpty(j)) break :blk false;
                    // Mask a token of another sequence.
                    if (!cl.seqHas(j, seq_id)) break :blk false;
                    p0 = cl.posGet(j);
                    if (!alibi and !prev) {
                        // Record every cell with p0 >= seq_pos_min - n_swa - 32.
                        if (p0 +% @as(i32, @intCast(args.n_swa + 32)) >= seq_pos_min[sid]) idxs.append(allocator, j) catch oom();
                    }
                    if (causal) {
                        // Mask future tokens.
                        if (p0 > p1) break :blk false;
                        // The M-RoPE causal mask.
                        if (is_2d and p0 == p1) {
                            if (cl.extGet(j).is2dGt(p1_x, p1_y)) break :blk false;
                        }
                    }
                    if (swa) {
                        // See `llama_non_causal_type`: SWA_FULL applies no
                        // SWA between tokens of the current ubatch.
                        const in_span = !causal and args.hparams.non_causal_type == non_causal_type_swa_full and p0 >= seq_pos_min[sid];
                        if (!in_span and Hparams.isMaskedSwa(args.n_swa, args.swa_type, p0, p1)) break :blk false;
                    }
                    break :blk true;
                };

                if (!keep) {
                    data[idst + j] = mask_drop;
                } else if (alibi) {
                    data[idst + j] = maskCast(T, @floatFromInt(-@as(i32, @intCast(@abs(p0 - p1)))));
                } else {
                    data[idst + j] = mask_keep;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// The per-batch context

/// Ports `llama_kv_cache_context` (llama-kv-cache.h:350 @5266f24da).
pub const KvCacheContext = struct {
    status: memory.Status,
    kv: ?*KvCache = null,
    lctx: ?*anyopaque = null,
    do_shift: bool = false,
    sc_info: StreamCopyInfo = .{},
    i_cur: usize = 0,
    sinfos: std.ArrayList(SlotInfo) = .empty,
    ubatches: std.ArrayList(Ubatch) = .empty,
    n_kv: u32 = 0,

    /// Ports the status constructor of `llama_kv_cache_context`
    /// (llama-kv-cache.cpp:2668 @5266f24da).
    pub fn createStatus(status: memory.Status) *KvCacheContext {
        const self = allocator.create(KvCacheContext) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-cache constructor of `llama_kv_cache_context`
    /// (llama-kv-cache.cpp:2670 @5266f24da): a dummy slot, enough to build
    /// the worst-case graph.
    pub fn createFull(kv: *KvCache) *KvCacheContext {
        const self = allocator.create(KvCacheContext) catch oom();
        self.* = .{ .status = .success, .kv = kv };
        self.n_kv = kv.getSize();
        const n_stream = kv.getNStream();
        var si: SlotInfo = .{ .s0 = 0, .s1 = n_stream - 1 };
        si.idxs.resize(allocator, n_stream) catch oom();
        for (0..n_stream) |s| {
            si.strm.append(allocator, @intCast(s)) catch oom();
            si.idxs.items[s] = .empty;
            si.idxs.items[s].append(allocator, 0) catch oom();
        }
        self.sinfos.append(allocator, si) catch oom();
        return self;
    }

    /// Ports the update constructor of `llama_kv_cache_context`
    /// (llama-kv-cache.cpp:2687 @5266f24da).
    pub fn createUpdate(kv: *KvCache, lctx: *anyopaque, do_shift: bool, sc_info: StreamCopyInfo) *KvCacheContext {
        const self = allocator.create(KvCacheContext) catch oom();
        self.* = .{ .status = .success, .kv = kv, .lctx = lctx, .do_shift = do_shift, .sc_info = sc_info };
        if (!do_shift and self.sc_info.empty()) self.status = .no_update;
        return self;
    }

    /// Ports the batch constructor of `llama_kv_cache_context`
    /// (llama-kv-cache.cpp:2697 @5266f24da): takes both lists.
    pub fn createBatch(kv: *KvCache, sinfos: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *KvCacheContext {
        const self = allocator.create(KvCacheContext) catch oom();
        self.* = .{ .status = .success, .kv = kv, .sinfos = sinfos, .ubatches = ubatches };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *KvCacheContext) void {
        self.sc_info.deinit();
        for (self.sinfos.items) |*si| si.deinit();
        self.sinfos.deinit(allocator);
        releaseUbatches(&self.ubatches);
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_context`'s `next` (llama-kv-cache.cpp:2705 @5266f24da).
    pub fn next(self: *KvCacheContext) bool {
        std.debug.assert(self.status == .success);
        self.i_cur += 1;
        return self.i_cur < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_context`'s `apply` (llama-kv-cache.cpp:2715 @5266f24da).
    pub fn apply(self: *KvCacheContext) bool {
        std.debug.assert(!memory.statusIsFail(self.status));
        // No ubatches: a cache update.
        if (self.ubatches.items.len == 0) {
            _ = self.kv.?.update(self.lctx.?, self.do_shift, &self.sc_info);
            return true;
        }
        self.kv.?.applyUbatch(&self.sinfos.items[self.i_cur], &self.ubatches.items[self.i_cur]);
        self.n_kv = self.kv.?.getNKv(&self.sinfos.items[self.i_cur]);
        return true;
    }

    /// Ports `llama_kv_cache_context`'s `get_status` (llama-kv-cache.cpp:2731 @5266f24da).
    pub fn getStatus(self: *const KvCacheContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_kv_cache_context`'s `get_ubatch` (llama-kv-cache.cpp:2735 @5266f24da).
    pub fn getUbatch(self: *const KvCacheContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_cur];
    }

    /// Ports `llama_kv_cache_context`'s `get_n_kv` (llama-kv-cache.cpp:2741 @5266f24da).
    pub fn getNKv(self: *const KvCacheContext) u32 {
        return self.n_kv;
    }

    pub fn typeK(self: *const KvCacheContext) c.enum_ggml_type {
        return self.kv.?.typeK();
    }
    pub fn typeV(self: *const KvCacheContext) c.enum_ggml_type {
        return self.kv.?.typeV();
    }

    fn sinfo(self: *const KvCacheContext) *const SlotInfo {
        return &self.sinfos.items[self.i_cur];
    }

    pub fn getK(self: *const KvCacheContext, ctx: *c.ggml_context, il: i32) *Tensor {
        return self.kv.?.getK(ctx, il, self.n_kv, self.sinfo());
    }
    pub fn getV(self: *const KvCacheContext, ctx: *c.ggml_context, il: i32) *Tensor {
        return self.kv.?.getV(ctx, il, self.n_kv, self.sinfo());
    }
    pub fn cpyK(self: *const KvCacheContext, ctx: *c.ggml_context, k_cur: *Tensor, k_idxs: *Tensor, il: i32) *Tensor {
        return self.kv.?.cpyK(ctx, k_cur, k_idxs, il, self.sinfo());
    }
    pub fn cpyV(self: *const KvCacheContext, ctx: *c.ggml_context, v_cur: *Tensor, v_idxs: *Tensor, il: i32) *Tensor {
        return self.kv.?.cpyV(ctx, v_cur, v_idxs, il, self.sinfo());
    }
    pub fn buildInputKIdxs(self: *const KvCacheContext, ctx: *c.ggml_context, ubatch: *const Ubatch) *Tensor {
        return self.kv.?.buildInputKIdxs(ctx, ubatch);
    }
    pub fn buildInputVIdxs(self: *const KvCacheContext, ctx: *c.ggml_context, ubatch: *const Ubatch) *Tensor {
        return self.kv.?.buildInputVIdxs(ctx, ubatch);
    }
    pub fn buildInputKRot(self: *const KvCacheContext, ctx: *c.ggml_context) ?*Tensor {
        return self.kv.?.buildInputKRot(ctx);
    }
    pub fn buildInputVRot(self: *const KvCacheContext, ctx: *c.ggml_context) ?*Tensor {
        return self.kv.?.buildInputVRot(ctx);
    }
    pub fn setInputKShift(self: *const KvCacheContext, dst: *Tensor) void {
        self.kv.?.setInputKShift(dst);
    }
    pub fn setInputKIdxs(self: *const KvCacheContext, dst: *Tensor, ubatch: *const Ubatch) void {
        self.kv.?.setInputKIdxs(dst, ubatch, self.sinfo());
    }
    pub fn setInputVIdxs(self: *const KvCacheContext, dst: *Tensor, ubatch: *const Ubatch) void {
        self.kv.?.setInputVIdxs(dst, ubatch, self.sinfo());
    }
    pub fn setInputKqMask(self: *const KvCacheContext, dst: *Tensor, ubatch: *const Ubatch, causal_attn: bool) void {
        self.kv.?.setInputKqMask(dst, ubatch, causal_attn);
    }
    pub fn setInputPosBucket(self: *const KvCacheContext, dst: *Tensor, ubatch: *const Ubatch) void {
        self.kv.?.setInputPosBucket(dst, ubatch);
    }
    pub fn setInputKRot(self: *const KvCacheContext, dst: *Tensor) void {
        self.kv.?.setInputKRot(dst);
    }
    pub fn setInputVRot(self: *const KvCacheContext, dst: *Tensor) void {
        self.kv.?.setInputVRot(dst);
    }

    /// Ports `llama_kv_cache_context`'s `get_prev_tokens`
    /// (llama-kv-cache.cpp:2813 @5266f24da); see `KvCache.getPrevTokens`.
    pub fn getPrevTokens(self: *const KvCacheContext, ubatch: *const Ubatch, n: u32, res: *std.ArrayList(i32)) void {
        self.kv.?.getPrevTokens(ubatch, n, res);
    }

    pub fn contextOf(self: *KvCacheContext) memory.Context {
        return .{ .ptr = self, .vtable = &ctx_vtable };
    }

    const ctx_vtable: memory.Context.VTable = .{
        .next = struct {
            fn f(p: *anyopaque) bool {
                return cast(p).next();
            }
        }.f,
        .apply = struct {
            fn f(p: *anyopaque) bool {
                return cast(p).apply();
            }
        }.f,
        .getUbatch = struct {
            fn f(p: *anyopaque) *const Ubatch {
                return cast(p).getUbatch();
            }
        }.f,
        .getStatus = struct {
            fn f(p: *anyopaque) memory.Status {
                return cast(p).getStatus();
            }
        }.f,
        .destroy = struct {
            fn f(p: *anyopaque) void {
                cast(p).destroy();
            }
        }.f,
    };

    fn cast(p: *anyopaque) *KvCacheContext {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(StreamCopyInfo);
    std.testing.refAllDecls(SlotInfo);
    std.testing.refAllDecls(KvCache);
    std.testing.refAllDecls(KvCacheContext);
}

test "the Hadamard matrix squares to the identity" {
    var m: [16]f32 = @splat(0);
    genHadamard(&m, 4);
    for (0..4) |i| {
        for (0..4) |j| {
            var acc: f32 = 0;
            for (0..4) |k| acc += m[i * 4 + k] * m[k * 4 + j];
            try std.testing.expectApproxEqAbs(@as(f32, if (i == j) 1 else 0), acc, 1e-6);
        }
    }
}
