//! DeepSeek-V4's memory: a raw sliding-window token cache, three compressed
//! K-only block caches -- CSA, one row per 4 tokens; HCA, one row per 128;
//! and the lightning indexer's, one per 4 -- and the compressor states the
//! graph builds their rows from.
//!
//! The compressed caches are storage only: which rows each query sees, and
//! which state rows a ubatch reads and writes, is planned per ubatch here
//! (`CompPlan`) and handed to the graph as inputs (`graph.InpDsv4`). The
//! C++'s two FIXMEs stand: the cache only supports non-unified mode even
//! when the unified flag is passed, and it conflates token positions with
//! buffer contents
//! (https://github.com/ggml-org/llama.cpp/pull/25521#discussion_r3558173819).
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-kv-cache-dsv4.h` -- `llama_dsv4_comp_state`,
//!   `llama_kv_cache_dsv4`, `llama_kv_cache_dsv4_raw_context`,
//!   `llama_kv_cache_dsv4_comp_context` and `llama_kv_cache_dsv4_context`
//! - `llama.cpp/src/llama-kv-cache-dsv4.cpp` -- their bodies and the static
//!   planners
//!
//! All at v0.4.1 (`b29c606e2`).

const std = @import("std");
const impl = @import("impl.zig");
const hparams_mod = @import("hparams.zig");
const cparams_mod = @import("cparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const io = @import("io.zig");
const kv_cache = @import("kv_cache.zig");
const kv_cache_iswa = @import("kv_cache_iswa.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;
const Ubatch = batch.Ubatch;
const KvCache = kv_cache.KvCache;
const KvCacheContext = kv_cache.KvCacheContext;
const KvCacheIswa = kv_cache_iswa.KvCacheIswa;
const SlotInfo = kv_cache.SlotInfo;
const StreamCopyInfo = kv_cache.StreamCopyInfo;
const Tensor = c.ggml_tensor;
const max_seq = cparams_mod.max_seq;

const file = "llama-kv-cache-dsv4.cpp";

fn oom() noreturn {
    impl.abort(file, 0, "out of memory");
}

/// `DSV4_CSA_RATIO` (llama-kv-cache-dsv4.cpp:18 @b29c606e2).
pub const csa_ratio: u32 = 4;
/// `DSV4_HCA_RATIO` (llama-kv-cache-dsv4.cpp:19 @b29c606e2).
pub const hca_ratio: u32 = 128;

/// `DSV4_STATE_MAGIC` (llama-kv-cache-dsv4.cpp:21 @b29c606e2): "DSV4".
const state_magic: u32 = 0x34565344;
/// `DSV4_STATE_VERSION` (llama-kv-cache-dsv4.cpp:22 @b29c606e2).
const state_version: u32 = 1;
/// `DSV4_STATE_MODE_FULL` (llama-kv-cache-dsv4.cpp:23 @b29c606e2).
const state_mode_full: u32 = 0;
/// `DSV4_STATE_MODE_PARTIAL` (llama-kv-cache-dsv4.cpp:24 @b29c606e2).
const state_mode_partial: u32 = 1;
/// `DSV4_K_CACHE_STATE_VER` (llama-kv-cache-dsv4.cpp:25 @b29c606e2).
const k_cache_state_ver: u32 = 2;
/// `DSV4_COMP_STATE_VER` (llama-kv-cache-dsv4.cpp:26 @b29c606e2).
const comp_state_ver: u32 = 1;

/// Ports `dsv4_comp_size` (llama-kv-cache-dsv4.cpp:28 @b29c606e2): the
/// compressed rows `kv_size` tokens make, at least one.
fn compSize(kv_size: u32, ratio: u32) u32 {
    return @max(1, (kv_size +% ratio -% 1) / ratio);
}

/// `GGML_PAD` (ggml.h:267 @b29c606e2) for a power-of-two `n`.
fn padTo(x: u32, comptime n: u32) u32 {
    return (x +% (n - 1)) & ~@as(u32, n - 1);
}

/// Ports `dsv4_clear_tensor_stream` (llama-kv-cache-dsv4.cpp:32 @b29c606e2):
/// zero one stream's plane of a contiguous 3-D tensor.
fn clearTensorStream(tensor: *Tensor, stream: u32) void {
    impl.assert(c.ggml_is_contiguous(tensor), file, 33, "ggml_is_contiguous(tensor)");
    impl.assert(tensor.ne[3] == 1, file, 34, "tensor->ne[3] == 1");
    impl.assert(stream < @as(u32, @truncate(@as(u64, @bitCast(tensor.ne[2])))), file, 35, "stream < (uint32_t) tensor->ne[2]");

    const stream_size = tensor.nb[2];
    c.ggml_backend_tensor_memset(tensor, 0, stream * stream_size, stream_size);
}

/// Ports `dsv4_state_n_used_k_rows` (llama-kv-cache-dsv4.cpp:41 @b29c606e2):
/// the compressed rows a sequence ending at `pos_max` has completed.
fn stateNUsedKRows(pos_max: i32, ratio: u32, kv_size: u32) u32 {
    if (pos_max < 0) return 0;
    const n_rows: u64 = (@as(u64, @intCast(pos_max)) + 1) / ratio;
    return @intCast(@min(@as(u64, kv_size), n_rows));
}

/// Ports `dsv4_stream_offset` (llama-kv-cache-dsv4.cpp:51 @b29c606e2): where
/// a sequence's stream starts, in rows of `size`. Its only callers plan a
/// batch, where the C++'s exception escapes `llama_decode` uncaught.
fn streamOffset(n_stream: u32, seq_id: i32, size: u32) i64 {
    if (n_stream <= 1) return 0;
    if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= n_stream) {
        impl.throwRuntime(file, 56, "DSV4 sequence id out of stream range");
    }
    return @as(i64, seq_id) * size;
}

/// Ports `dsv4_ubatch_has_coupled` (llama-kv-cache-dsv4.cpp:62 @b29c606e2):
/// whether a token belongs to several sequences at once.
fn ubatchHasCoupled(ubatch: *const Ubatch) bool {
    for (0..ubatch.n_tokens) |i| {
        if (ubatch.n_seq_id[i] > 1) return true;
    }
    return false;
}

/// Ports `dsv4_token_has_seq` (llama-kv-cache-dsv4.cpp:72 @b29c606e2).
fn tokenHasSeq(ubatch: *const Ubatch, i: usize, seq_id: i32) bool {
    var s: usize = 0;
    while (s < @as(usize, @intCast(ubatch.n_seq_id[i]))) : (s += 1) {
        if (ubatch.seq_id[i][s] == seq_id) return true;
    }
    return false;
}

/// Ports `dsv4_build_raw_write_ubatch` (llama-kv-cache-dsv4.cpp:82
/// @b29c606e2): a coupled ubatch, with tokens shared between sequences,
/// unrolled into one copy of each token per sequence, so the raw cache
/// writes each sequence's stream separately.
///
/// Return: a ubatch holding its own reference, the caller's to `release`.
fn buildRawWriteUbatch(ubatch: *const Ubatch) Ubatch {
    if (!ubatchHasCoupled(ubatch)) {
        ubatch.retain();
        return ubatch.*;
    }
    if (ubatch.embd != null) {
        impl.throwRuntime(file, 87, "DSV4 coupled embedding ubatches are not supported");
    }

    var counts: std.ArrayList(u32) = .empty;
    defer counts.deinit(allocator);
    counts.appendNTimes(allocator, 0, ubatch.n_seqs_unq) catch oom();
    var n_tokens: u32 = 0;
    for (0..ubatch.n_seqs_unq) |s| {
        const seq_id = ubatch.seq_id_unq[s];
        for (0..ubatch.n_tokens) |i| {
            if (tokenHasSeq(ubatch, i, seq_id)) {
                counts.items[s] += 1;
                n_tokens += 1;
            }
        }
    }

    if (n_tokens == 0) {
        ubatch.retain();
        return ubatch.*;
    }

    const n_seq_tokens = counts.items[0];
    for (counts.items[1..]) |n| {
        if (n != n_seq_tokens) impl.throwRuntime(file, 109, "DSV4 coupled raw writes require equal sequence lengths");
    }

    const data = allocator.create(batch.UbatchData) catch oom();
    data.* = .{};
    data.pos.appendNTimes(allocator, 0, @as(usize, n_tokens) * ubatch.n_pos) catch oom();
    data.n_seq_id.ensureTotalCapacity(allocator, n_tokens) catch oom();
    data.seq_id.ensureTotalCapacity(allocator, n_tokens) catch oom();
    data.seq_id_data.ensureTotalCapacity(allocator, n_tokens) catch oom();
    data.seq_id_unq.appendSlice(allocator, ubatch.seq_id_unq[0..ubatch.n_seqs_unq]) catch oom();
    data.seq_idx.appendNTimes(allocator, -1, max_seq) catch oom();
    data.output.appendNTimes(allocator, 0, n_tokens) catch oom();
    if (ubatch.token != null) data.token.ensureTotalCapacity(allocator, n_tokens) catch oom();

    for (data.seq_id_unq.items, 0..) |seq_id, s| data.seq_idx.items[@intCast(seq_id)] = @intCast(s);

    for (0..ubatch.n_seqs_unq) |s| {
        const seq_id = ubatch.seq_id_unq[s];
        for (0..ubatch.n_tokens) |i| {
            if (!tokenHasSeq(ubatch, i, seq_id)) continue;

            const dst = data.n_seq_id.items.len;
            if (ubatch.token != null) data.token.appendAssumeCapacity(ubatch.token[i]);
            for (0..ubatch.n_pos) |p| {
                data.pos.items[p * n_tokens + dst] = ubatch.pos[p * ubatch.n_tokens + i];
            }
            data.n_seq_id.appendAssumeCapacity(1);
            data.seq_id_data.appendAssumeCapacity(seq_id);
        }
    }

    // The capacity was reserved, so these pointers stay put.
    for (0..n_tokens) |i| data.seq_id.appendAssumeCapacity(@ptrCast(&data.seq_id_data.items[i]));

    return .{
        .b_equal_seqs = 1,
        .n_tokens = n_tokens,
        .n_seq_tokens = n_seq_tokens,
        .n_seqs = ubatch.n_seqs_unq,
        .n_seqs_unq = ubatch.n_seqs_unq,
        .n_pos = ubatch.n_pos,
        .token = if (data.token.items.len == 0) null else data.token.items.ptr,
        .embd = null,
        .pos = data.pos.items.ptr,
        .n_seq_id = data.n_seq_id.items.ptr,
        .seq_id = data.seq_id.items.ptr,
        .seq_id_unq = data.seq_id_unq.items.ptr,
        .seq_idx = data.seq_idx.items.ptr,
        .output = data.output.items.ptr,
        .data = data,
    };
}

/// Ports `dsv4_build_raw_write_ubatches` (llama-kv-cache-dsv4.cpp:173 @b29c606e2).
///
/// Return: one raw-write ubatch per ubatch, the caller's to release.
fn buildRawWriteUbatches(ubatches: []const Ubatch) std.ArrayList(Ubatch) {
    var res: std.ArrayList(Ubatch) = .empty;
    res.ensureTotalCapacity(allocator, ubatches.len) catch oom();
    for (ubatches) |*u| res.appendAssumeCapacity(buildRawWriteUbatch(u));
    return res;
}

/// Ports `dsv4_batch_has_coupled` (llama-kv-cache-dsv4.cpp:182 @b29c606e2).
fn batchHasCoupled(b: *const c.llama_batch) bool {
    if (b.n_seq_id == null) return false;
    var i: i32 = 0;
    while (i < b.n_tokens) : (i += 1) {
        if (b.n_seq_id[@intCast(i)] > 1) return true;
    }
    return false;
}

/// Ports `dsv4_comp_graph_n_stream` (llama-kv-cache-dsv4.cpp:196 @b29c606e2):
/// coupled sequence sets must stay in one graph stream because their
/// compressed state is shared; independent per-sequence state can fan out.
fn compGraphNStream(ubatch: *const Ubatch, n_stream: u32) i64 {
    if (n_stream <= 1 or ubatch.n_seqs_unq <= 1 or ubatchHasCoupled(ubatch)) return 1;
    return ubatch.n_seqs_unq;
}

/// Ports `dsv4_state_src_stream_range` (llama-kv-cache-dsv4.cpp:206
/// @b29c606e2): the streams a state write saves.
fn stateSrcStreamRange(n_stream: u32, seq_id: i32, s0: *u32, ns: *u32, exc: *Exception) Thrown!void {
    if (seq_id >= 0 and n_stream > 1) {
        if (@as(u32, @bitCast(seq_id)) >= n_stream) {
            return exc.throw(.runtime_error, "DSV4 state sequence id out of stream range", .{});
        }
        s0.* = @bitCast(seq_id);
        ns.* = 1;
        return;
    }
    s0.* = 0;
    ns.* = if (seq_id >= 0) 1 else n_stream;
}

/// Ports `dsv4_state_dst_stream_range` (llama-kv-cache-dsv4.cpp:225
/// @b29c606e2): the first stream a state read restores into.
fn stateDstStreamRange(n_stream: u32, seq_id: i32, ns: u32, s0: *u32, exc: *Exception) Thrown!void {
    if (seq_id >= 0) {
        if (ns != 1) return exc.throw(.runtime_error, "DSV4 sequence state stream count mismatch", .{});
        if (n_stream > 1 and @as(u32, @bitCast(seq_id)) >= n_stream) {
            return exc.throw(.runtime_error, "DSV4 state sequence id out of stream range", .{});
        }
        s0.* = if (n_stream > 1) @bitCast(seq_id) else 0;
        return;
    }
    if (ns != n_stream) return exc.throw(.runtime_error, "DSV4 full state stream count mismatch", .{});
    s0.* = 0;
}

/// Ports `dsv4_state_write_tensor_streams` (llama-kv-cache-dsv4.cpp:249
/// @b29c606e2): a tensor's metadata, then the first `n_rows` rows of each
/// stream -- `stream_ids`' streams when given, else `ns` from `s0`.
fn stateWriteTensorStreams(w: io.Writer, tensor: *Tensor, tensor_rows: u32, n_rows: u32, s0: u32, ns: u32, stream_ids: ?[]const u32, exc: *Exception) Thrown!void {
    const type_i: i32 = @intCast(tensor.type);
    const ne0: u64 = @bitCast(tensor.ne[0]);
    const rows: u64 = n_rows;
    const row_size: u64 = c.ggml_row_size(tensor.type, tensor.ne[0]);

    if (n_rows > tensor_rows) return exc.throw(.runtime_error, "DSV4 state tensor row count exceeds storage", .{});

    try w.writeValue(type_i, exc);
    try w.writeValue(ne0, exc);
    try w.writeValue(rows, exc);
    try w.writeValue(row_size, exc);

    const stream_stride: usize = @as(usize, tensor_rows) * row_size;
    const size: usize = @as(usize, n_rows) * row_size;
    if (size == 0) return;

    if (stream_ids) |ids| {
        if (ids.len != ns) return exc.throw(.runtime_error, "DSV4 state tensor stream map size mismatch", .{});
    }

    for (0..ns) |s| {
        const stream: u32 = if (stream_ids) |ids| ids[s] else s0 + @as(u32, @intCast(s));
        if (@as(i64, stream) >= tensor.ne[2]) return exc.throw(.runtime_error, "DSV4 state tensor stream out of range", .{});
        const offset: usize = @as(usize, stream) * stream_stride;
        try w.writeTensor(tensor, offset, size, exc);
    }
}

/// Ports `dsv4_state_read_tensor_streams` (llama-kv-cache-dsv4.cpp:291 @b29c606e2).
fn stateReadTensorStreams(r: io.Reader, tensor: *Tensor, tensor_rows: u32, n_rows: u32, s0: u32, ns: u32, exc: *Exception) Thrown!void {
    const type_i_ref = try r.readValue(i32, exc);
    const ne0_ref = try r.readValue(u64, exc);
    const rows_ref = try r.readValue(u64, exc);
    const row_size_ref = try r.readValue(u64, exc);

    const type_i: i32 = @intCast(tensor.type);
    const ne0: u64 = @bitCast(tensor.ne[0]);
    const rows: u64 = n_rows;
    const row_size: u64 = c.ggml_row_size(tensor.type, tensor.ne[0]);

    if (type_i != type_i_ref or ne0 != ne0_ref or rows != rows_ref or row_size != row_size_ref) {
        return exc.throw(.runtime_error, "DSV4 state tensor metadata mismatch", .{});
    }
    if (n_rows > tensor_rows) return exc.throw(.runtime_error, "DSV4 state tensor row count exceeds storage", .{});

    const stream_stride: usize = @as(usize, tensor_rows) * row_size;
    const size: usize = @as(usize, n_rows) * row_size;
    if (size == 0) return;

    for (0..ns) |s| {
        const offset: usize = (@as(usize, s0) + s) * stream_stride;
        try r.readTensor(tensor, offset, size, exc);
    }
}

/// Ports `dsv4_state_write_k_cache` (llama-kv-cache-dsv4.cpp:332
/// @b29c606e2): a compressed cache's first `n_rows` rows, layer by layer.
fn stateWriteKCache(w: io.Writer, kv: *const KvCache, seq_id: i32, flags: u32, n_rows: u32, exc: *Exception) Thrown!void {
    _ = flags;

    var s0: u32 = undefined;
    var ns: u32 = undefined;
    try stateSrcStreamRange(kv.getNStream(), seq_id, &s0, &ns, exc);

    const version: u32 = k_cache_state_ver;
    const kv_size = kv.getSize();
    var layer_ids = kv.getLayerIds();
    defer layer_ids.deinit(allocator);
    const n_layer: u32 = @intCast(layer_ids.items.len);

    if (n_rows > kv_size) return exc.throw(.runtime_error, "DSV4 K-cache state row count exceeds cache size", .{});

    try w.writeValue(version, exc);
    try w.writeValue(n_rows, exc);
    try w.writeValue(ns, exc);
    try w.writeValue(n_layer, exc);

    for (layer_ids.items) |il| {
        try w.writeValue(il, exc);
        try stateWriteTensorStreams(w, kv.getKStorage(@intCast(il)), kv_size, n_rows, s0, ns, null, exc);
    }
}

/// Ports `dsv4_state_read_k_cache` (llama-kv-cache-dsv4.cpp:364 @b29c606e2).
fn stateReadKCache(r: io.Reader, kv: *KvCache, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
    _ = flags;

    const version = try r.readValue(u32, exc);
    const n_rows_ref = try r.readValue(u32, exc);
    const ns = try r.readValue(u32, exc);
    const n_layer_ref = try r.readValue(u32, exc);

    if (version != 1 and version != k_cache_state_ver) return exc.throw(.runtime_error, "DSV4 K-cache state version mismatch", .{});

    const kv_size = kv.getSize();
    if (version == 1 and n_rows_ref != kv_size) {
        impl.logInfo("kv size ref %d kv %d\n", .{ n_rows_ref, kv_size });
        return exc.throw(.runtime_error, "DSV4 K-cache state size mismatch", .{});
    }
    if (n_rows_ref > kv_size) {
        impl.logInfo("kv rows ref %d kv %d\n", .{ n_rows_ref, kv_size });
        return exc.throw(.runtime_error, "DSV4 K-cache state size mismatch", .{});
    }

    var s0: u32 = undefined;
    try stateDstStreamRange(kv.getNStream(), seq_id, ns, &s0, exc);

    var layer_ids = kv.getLayerIds();
    defer layer_ids.deinit(allocator);
    if (n_layer_ref != layer_ids.items.len) return exc.throw(.runtime_error, "DSV4 K-cache layer count mismatch", .{});

    for (layer_ids.items) |il| {
        const il_ref = try r.readValue(u32, exc);
        if (il_ref != il) return exc.throw(.runtime_error, "DSV4 K-cache layer id mismatch", .{});
        try stateReadTensorStreams(r, kv.getKStorage(@intCast(il)), kv_size, n_rows_ref, s0, ns, exc);
    }
}

/// Ports `dsv4_plan_positions` (llama-kv-cache-dsv4.cpp:414 @b29c606e2):
/// `[a, b, c]`, for the debug log.
///
/// Return: the text, NUL-terminated, the caller's to free.
pub fn planPositions(values: []const i32) [:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    out.append(allocator, '[') catch oom();
    for (values, 0..) |v, i| {
        if (i > 0) out.appendSlice(allocator, ", ") catch oom();
        out.print(allocator, "{d}", .{v}) catch oom();
    }
    out.append(allocator, ']') catch oom();
    return out.toOwnedSliceSentinel(allocator, 0) catch oom();
}

/// `LLAMA_DSV4_COMPRESS_DEBUG`, read into `dsv4_build_comp_plan`'s static
/// `debug` (llama-kv-cache-dsv4.cpp:718 @b29c606e2).
pub fn compressDebug() bool {
    const env = std.c.getenv("LLAMA_DSV4_COMPRESS_DEBUG") orelse return false;
    return impl.atoi(env) > 0;
}

/// Mirrors `llama_kv_cache_dsv4_context`'s `comp_plan` (llama-kv-cache-dsv4.h:273
/// @b29c606e2): the per-ubatch recipe for updating a compressor state,
/// committing completed compressed rows, and masking the compressed
/// attention source.
pub const CompPlan = struct {
    /// APE row ids, `pos % ratio`, for the compressor-state updates.
    state_pos: std.ArrayList(i32) = .empty,

    /// Current-ubatch source rows and unique persistent-state destination
    /// rows, for deterministic ring-state updates.
    state_persist_src_idxs: std.ArrayList(i32) = .empty,
    state_persist_dst_idxs: std.ArrayList(i32) = .empty,

    /// Rollback restores: snapshot planes copied back to the current
    /// compressor-state plane before the graph reads it.
    state_restore_src_idxs: std.ArrayList(i32) = .empty,
    state_restore_dst_idxs: std.ArrayList(i32) = .empty,

    /// Rollback snapshots: rows of the graph-local `[persistent_state |
    /// current_ubatch_scratch]` tensor copied into the rollback planes after
    /// the graph has computed the current tokens' compressor state.
    state_snapshot_src_idxs: std.ArrayList(i32) = .empty,
    state_snapshot_dst_idxs: std.ArrayList(i32) = .empty,

    /// Flattened source rows for state-backed commits, into the graph-local
    /// `[persistent_state | current_ubatch_scratch]` tensor. For overlapped
    /// compression the first half is previous rows and the second current
    /// rows; a final synthetic zero/-inf row may be addressed for the first
    /// block's previous half.
    state_read_idxs: std.ArrayList(i32) = .empty,

    /// The compressed-cache rows state-backed commits write. A
    /// non-boundary CSA/LID decode step can target a masked scratch row.
    state_write_idxs: std.ArrayList(i64) = .empty,

    /// RoPE positions for state-backed commits.
    state_write_pos: std.ArrayList(i32) = .empty,

    /// Completed compressed rows visible to each query token.
    n_visible: std.ArrayList(i32) = .empty,

    /// Streams the attention graph uses for this ubatch.
    n_stream: i64 = 1,

    /// The graph's width in compressed rows. It can exceed `n_visible`, so
    /// masked padding rows do not force a new graph at every CSA block.
    n_kv: i64 = 0,

    pub fn deinit(self: *CompPlan) void {
        self.state_pos.deinit(allocator);
        self.state_persist_src_idxs.deinit(allocator);
        self.state_persist_dst_idxs.deinit(allocator);
        self.state_restore_src_idxs.deinit(allocator);
        self.state_restore_dst_idxs.deinit(allocator);
        self.state_snapshot_src_idxs.deinit(allocator);
        self.state_snapshot_dst_idxs.deinit(allocator);
        self.state_read_idxs.deinit(allocator);
        self.state_write_idxs.deinit(allocator);
        self.state_write_pos.deinit(allocator);
        self.n_visible.deinit(allocator);
        self.* = .{};
    }
};

fn push(comptime T: type, list: *std.ArrayList(T), v: T) void {
    list.append(allocator, v) catch oom();
}

/// The planner's `state_source_idx` lambda (llama-kv-cache-dsv4.cpp:473
/// @b29c606e2): a state source's row in the graph-local `[persistent_state
/// | current_ubatch_scratch | zero]` tensor.
const SourceIdx = struct {
    const Key = struct { seq: i32, pos: i32 };

    state_rows: i64,
    n_tokens: u32,
    n_stream: u32,
    state_size: u32,
    curr_token_idx_map: std.AutoHashMapUnmanaged(Key, i64) = .empty,

    fn at(self: *const SourceIdx, seq_id: i32, pos: i32) i32 {
        if (pos < 0) {
            // The overlap compressor needs a zero/-inf source for the first
            // block's previous half. The graph appends that row after the
            // current-ubatch scratch rows.
            return @truncate(self.state_rows + self.n_tokens);
        }
        if (self.curr_token_idx_map.get(.{ .seq = seq_id, .pos = pos })) |i| {
            return @truncate(self.state_rows + i);
        }
        const stream_off = streamOffset(self.n_stream, seq_id, self.state_size);
        return @truncate(stream_off + @as(u32, @bitCast(pos)) % self.state_size);
    }
};

/// Ports `dsv4_build_comp_plan` (llama-kv-cache-dsv4.cpp:427 @b29c606e2).
///
/// `n_stream` is the persistent cache and state layout; `plan.n_stream` is
/// the graph's view for this ubatch, which can be a subset of those
/// streams.
///
/// Return: the plan, the caller's to `deinit`.
fn buildCompPlan(ubatch: *const Ubatch, ratio: u32, overlap: bool, state_size: u32, kv_size: u32, n_stream: u32, n_rs_seq: u32, rs_idx: []const u32) CompPlan {
    var plan: CompPlan = .{};
    plan.n_visible.appendNTimes(allocator, 0, ubatch.n_tokens) catch oom();
    plan.n_stream = compGraphNStream(ubatch, n_stream);

    if (n_stream <= 1 and ubatch.n_seqs_unq > 1) {
        impl.throwRuntime(file, 443, "DSV4 single compressed stream cannot serve multiple sequences");
    }

    const state_rows: i64 = @as(i64, state_size) * n_stream;

    const PersistRow = struct { dst: i32, src: i32, pos: i32 };
    var persist_rows: std.ArrayList(PersistRow) = .empty;
    defer persist_rows.deinit(allocator);

    // For the overlap compressor, `build_overlap_compressed_kv_from_state`
    // reads `state_read_idxs` as two contiguous halves: the first
    // `ratio*n_blocks` entries are every block's previous-window rows, then
    // every block's current-window rows. They are collected apart here and
    // joined once every completed block has been visited.
    var overlap_prev_reads: std.ArrayList(i32) = .empty;
    defer overlap_prev_reads.deinit(allocator);
    var overlap_cur_reads: std.ArrayList(i32) = .empty;
    defer overlap_cur_reads.deinit(allocator);

    var src: SourceIdx = .{ .state_rows = state_rows, .n_tokens = ubatch.n_tokens, .n_stream = n_stream, .state_size = state_size };
    defer src.curr_token_idx_map.deinit(allocator);
    var state_write_counts: std.AutoHashMapUnmanaged(i32, u32) = .empty;
    defer state_write_counts.deinit(allocator);

    for (0..ubatch.n_tokens) |i| {
        var s: usize = 0;
        while (s < @as(usize, @intCast(ubatch.n_seq_id[i]))) : (s += 1) {
            src.curr_token_idx_map.put(allocator, .{ .seq = ubatch.seq_id[i][s], .pos = ubatch.pos[i] }, @intCast(i)) catch oom();
        }
    }

    for (0..ubatch.n_tokens) |i| {
        const pos = ubatch.pos[i];
        if (pos < 0) continue;
        const upos: u32 = @bitCast(pos);

        push(i32, &plan.state_pos, @intCast(upos % ratio));

        const n_visible: i64 = @divTrunc(@as(i64, pos + 1), @as(i64, ratio));
        plan.n_visible.items[i] = @truncate(n_visible);
        plan.n_kv = @max(plan.n_kv, n_visible);

        var s: usize = 0;
        while (s < @as(usize, @intCast(ubatch.n_seq_id[i]))) : (s += 1) {
            const seq_id = ubatch.seq_id[i][s];
            const stream_off = streamOffset(n_stream, seq_id, state_size);
            const state_idx: i32 = @truncate(stream_off + upos % state_size);

            var found: ?*PersistRow = null;
            for (persist_rows.items) |*row| {
                if (row.dst == state_idx) {
                    found = row;
                    break;
                }
            }
            if (found) |row| {
                if (pos > row.pos) {
                    row.src = @intCast(i);
                    row.pos = pos;
                }
            } else {
                persist_rows.append(allocator, .{ .dst = state_idx, .src = @intCast(i), .pos = pos }) catch oom();
            }

            if ((upos + 1) % ratio != 0) continue;

            const source_start: i32 = @bitCast(upos + 1 -% ratio);
            const cache_off = streamOffset(n_stream, seq_id, kv_size);

            push(i64, &plan.state_write_idxs, cache_off + upos / ratio);
            push(i32, &plan.state_write_pos, source_start);
            const gop = state_write_counts.getOrPut(allocator, seq_id) catch oom();
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;

            if (overlap) {
                const prev_start: i32 = source_start -% @as(i32, @bitCast(ratio));
                var j: u32 = 0;
                while (j < ratio) : (j += 1) push(i32, &overlap_prev_reads, src.at(seq_id, prev_start +% @as(i32, @bitCast(j))));
                j = 0;
                while (j < ratio) : (j += 1) push(i32, &overlap_cur_reads, src.at(seq_id, source_start +% @as(i32, @bitCast(j))));
            } else {
                var j: u32 = 0;
                while (j < ratio) : (j += 1) push(i32, &plan.state_read_idxs, src.at(seq_id, source_start +% @as(i32, @bitCast(j))));
            }
        }
    }

    if (ratio == csa_ratio and plan.state_pos.items.len != 0) {
        std.debug.assert(kv_size > 0);

        // Pad each stream to the reserve plan's block count.
        const Dummy = struct {
            plan: *CompPlan,
            src: *const SourceIdx,
            prev: *std.ArrayList(i32),
            cur: *std.ArrayList(i32),
            ubatch: *const Ubatch,
            overlap: bool,
            ratio: u32,
            n_stream: u32,
            kv_size: u32,

            /// The planner's `append_dummy_block` lambda (llama-kv-cache-dsv4.cpp:551 @b29c606e2).
            fn append(d: @This(), seq_id: i32, i: usize) void {
                const cache_off = streamOffset(d.n_stream, seq_id, d.kv_size);
                const source_idx = d.src.at(seq_id, d.ubatch.pos[i]);

                push(i64, &d.plan.state_write_idxs, cache_off + d.kv_size - 1);
                push(i32, &d.plan.state_write_pos, 0);

                var j: u32 = 0;
                if (d.overlap) {
                    while (j < d.ratio) : (j += 1) {
                        push(i32, d.prev, source_idx);
                        push(i32, d.cur, source_idx);
                    }
                } else {
                    while (j < d.ratio) : (j += 1) push(i32, &d.plan.state_read_idxs, source_idx);
                }
            }
        };
        const dummy: Dummy = .{ .plan = &plan, .src = &src, .prev = &overlap_prev_reads, .cur = &overlap_cur_reads, .ubatch = ubatch, .overlap = overlap, .ratio = ratio, .n_stream = n_stream, .kv_size = kv_size };

        if (ubatchHasCoupled(ubatch)) {
            if (plan.state_write_idxs.items.len == 0) {
                var i: usize = 0;
                while (i < ubatch.n_tokens and ubatch.pos[i] < 0) i += 1;
                std.debug.assert(i < ubatch.n_tokens);
                dummy.append(ubatch.seq_id[i][0], i);
            }
        } else {
            const n_blocks: u32 = (@max(1, ubatch.n_seq_tokens) + ratio - 1) / ratio;

            for (0..ubatch.n_seqs_unq) |s| {
                const seq_id = ubatch.seq_id_unq[s];
                const n_writes: u32 = state_write_counts.get(seq_id) orelse 0;
                if (n_writes >= n_blocks) continue;
                if (n_writes + 1 != n_blocks) {
                    impl.throwRuntime(file, 589, "DSV4 CSA sequence positions are not contiguous");
                }

                var i: usize = 0;
                while (i < ubatch.n_tokens and (ubatch.pos[i] < 0 or !tokenHasSeq(ubatch, i, seq_id))) i += 1;
                std.debug.assert(i < ubatch.n_tokens);
                dummy.append(seq_id, i);
            }
        }
    }

    if (ratio == hca_ratio and plan.state_pos.items.len != 0 and plan.state_write_idxs.items.len == 0) {
        std.debug.assert(kv_size > 0);
        // The last slot must not be live, or the dummy write would corrupt
        // it; a full stream implies a completed block, which implies real
        // writes.
        std.debug.assert(plan.n_kv < @as(i64, kv_size));

        // Keep the compress and write ops in the graph when no HCA block
        // completes in this ubatch. The dummy block writes to the last
        // cache slot and is masked out.
        var i: usize = 0;
        while (i < ubatch.n_tokens and ubatch.pos[i] < 0) i += 1;
        std.debug.assert(i < ubatch.n_tokens);

        const seq_id = ubatch.seq_id[i][0];
        const cache_off = streamOffset(n_stream, seq_id, kv_size);
        const source_idx = src.at(seq_id, ubatch.pos[i]);

        push(i64, &plan.state_write_idxs, cache_off + kv_size - 1);
        push(i32, &plan.state_write_pos, 0);

        var j: u32 = 0;
        while (j < ratio) : (j += 1) push(i32, &plan.state_read_idxs, source_idx);
    }

    if (overlap) {
        // [ all blocks' prev-window rows | all blocks' cur-window rows ]
        plan.state_read_idxs.appendSlice(allocator, overlap_prev_reads.items) catch oom();
        plan.state_read_idxs.appendSlice(allocator, overlap_cur_reads.items) catch oom();
    }

    // Keep the mask, and with it the compressed-attention branch, present
    // even before the first block is visible, so the graph's topology never
    // changes. Padded slots are masked out; the compressed buffers are
    // zero-initialized.
    plan.n_kv = @max(padI64(plan.n_kv, 256), 256);

    std.mem.sort(PersistRow, persist_rows.items, {}, struct {
        fn lt(_: void, a: PersistRow, b: PersistRow) bool {
            return a.dst < b.dst;
        }
    }.lt);

    for (persist_rows.items) |row| {
        push(i32, &plan.state_persist_src_idxs, row.src);
        push(i32, &plan.state_persist_dst_idxs, row.dst);
    }

    if (n_rs_seq > 0) {
        // Restore and snapshot entries for every stream of the layout, so
        // the graph's tensor sizes do not depend on the ubatch's sequence
        // count. Streams absent from the ubatch get no-op entries.
        var token_idxs: std.ArrayList(u32) = .empty;
        defer token_idxs.deinit(allocator);

        var stream: u32 = 0;
        while (stream < n_stream) : (stream += 1) {
            var seq_id: i32 = -1;
            if (n_stream == 1) {
                // A unified stream serves any single sequence.
                seq_id = if (ubatch.n_seqs_unq > 0) ubatch.seq_id_unq[0] else -1;
            } else {
                for (0..ubatch.n_seqs_unq) |s| {
                    if (ubatch.seq_id_unq[s] == @as(i32, @bitCast(stream))) {
                        seq_id = ubatch.seq_id_unq[s];
                        break;
                    }
                }
            }

            const stream_off: i64 = @as(i64, stream) * state_size;
            const rollback: u32 = if (seq_id >= 0 and @as(u32, @bitCast(seq_id)) < rs_idx.len) rs_idx[@intCast(seq_id)] else 0;
            // Keep the restore graph fixed-width when no rollback is pending.
            const src_plane: i64 = if (rollback > 0 and rollback <= n_rs_seq) @as(i64, rollback) * state_rows else 0;
            var r: u32 = 0;
            while (r < state_size) : (r += 1) {
                push(i32, &plan.state_restore_src_idxs, @truncate(src_plane + stream_off + r));
                push(i32, &plan.state_restore_dst_idxs, @truncate(stream_off + r));
            }

            token_idxs.clearRetainingCapacity();
            if (seq_id >= 0) {
                for (0..ubatch.n_tokens) |i| {
                    if (tokenHasSeq(ubatch, i, seq_id)) token_idxs.append(allocator, @intCast(i)) catch oom();
                }
            }

            const n_seq_tokens: u32 = @intCast(token_idxs.items.len);
            const scratch_off: i64 = state_rows * (1 + @as(i64, n_rs_seq));
            var d: u32 = 1;
            while (d <= n_rs_seq) : (d += 1) {
                const dst_plane: i64 = @as(i64, d) * state_rows;
                const prefix: u32 = if (d <= n_seq_tokens) n_seq_tokens - d else 0;

                r = 0;
                while (r < state_size) : (r += 1) {
                    var srow: i32 = @truncate(stream_off + r);

                    for (token_idxs.items[0..prefix]) |i_tok| {
                        if (ubatch.pos[i_tok] >= 0 and @as(u32, @bitCast(ubatch.pos[i_tok])) % state_size == r) {
                            srow = @truncate(scratch_off + i_tok);
                        }
                    }

                    if (n_seq_tokens == 0) {
                        // A no-op: copy the snapshot plane onto itself.
                        srow = @truncate(dst_plane + stream_off + r);
                    }

                    push(i32, &plan.state_snapshot_src_idxs, srow);
                    push(i32, &plan.state_snapshot_dst_idxs, @truncate(dst_plane + stream_off + r));
                }
            }
        }
    }

    if (compressDebug()) {
        const persist = planPositions(plan.state_persist_dst_idxs.items);
        defer allocator.free(persist);
        const write_pos = planPositions(plan.state_write_pos.items);
        defer allocator.free(write_pos);
        impl.logDebug("%s: ratio=%u, n_tokens=%u, n_seqs_unq=%u, state_persist_dst=%s, state_write_pos=%s\n", .{ "dsv4_build_comp_plan", ratio, ubatch.n_tokens, ubatch.n_seqs_unq, persist.ptr, write_pos.ptr });
        for (0..ubatch.n_seqs_unq) |s| {
            const seq_id = ubatch.seq_id_unq[s];
            const rollback: u32 = if (seq_id >= 0 and @as(u32, @bitCast(seq_id)) < rs_idx.len) rs_idx[@intCast(seq_id)] else 0;
            impl.logDebug("%s:   seq %d pos [%d, %d] rollback=%u\n", .{ "dsv4_build_comp_plan", seq_id, ubatch.pos[0], ubatch.pos[ubatch.n_tokens - 1], rollback });
        }
    }

    return plan;
}

/// `GGML_PAD` (ggml.h:267 @b29c606e2) on a non-negative `int64_t`.
fn padI64(x: i64, comptime n: i64) i64 {
    return (x + (n - 1)) & ~@as(i64, n - 1);
}

/// Ports `dsv4_build_comp_plans` (llama-kv-cache-dsv4.cpp:739 @b29c606e2):
/// one plan per ubatch; the first ubatch touching a sequence consumes its
/// rollback restore.
///
/// Return: the plans, the caller's to free.
fn buildCompPlans(ubatches: []const Ubatch, ratio: u32, overlap: bool, state_size: u32, kv_size: u32, n_stream: u32, n_rs_seq: u32, rs_idx: []const u32) std.ArrayList(CompPlan) {
    var plans: std.ArrayList(CompPlan) = .empty;
    plans.ensureTotalCapacity(allocator, ubatches.len) catch oom();

    const rs = allocator.dupe(u32, rs_idx) catch oom();
    defer allocator.free(rs);
    for (ubatches) |*ubatch| {
        plans.appendAssumeCapacity(buildCompPlan(ubatch, ratio, overlap, state_size, kv_size, n_stream, n_rs_seq, rs));

        for (0..ubatch.n_seqs_unq) |s| {
            const seq_id = ubatch.seq_id_unq[s];
            if (seq_id >= 0 and @as(usize, @intCast(seq_id)) < rs.len) rs[@intCast(seq_id)] = 0;
        }
    }
    return plans;
}

fn freePlans(list: *std.ArrayList(CompPlan)) void {
    for (list.items) |*p| p.deinit();
    list.deinit(allocator);
}

/// Ports `dsv4_build_comp_sinfos` (llama-kv-cache-dsv4.cpp:767 @b29c606e2):
/// a compressed cache's view per ubatch -- its streams, with a dummy index,
/// since compressed rows are graph outputs, not token writes.
///
/// Return: the slots, the caller's to free.
fn buildCompSinfos(ubatches: []const Ubatch, n_stream: u32) std.ArrayList(SlotInfo) {
    var sinfos: std.ArrayList(SlotInfo) = .empty;
    sinfos.ensureTotalCapacity(allocator, ubatches.len) catch oom();

    for (ubatches) |*ubatch| {
        if (n_stream <= 1 and ubatch.n_seqs_unq > 1) {
            impl.throwRuntime(file, 775, "DSV4 single compressed stream cannot serve multiple sequences");
        }

        const ns: u32 = @intCast(compGraphNStream(ubatch, n_stream));
        var sinfo: SlotInfo = .{};
        sinfo.s0 = if (n_stream > 1) max_seq else 0;
        sinfo.s1 = 0;
        sinfo.resize(ns);

        for (0..ns) |s| {
            const seq_id: i32 = if (n_stream > 1) ubatch.seq_id_unq[s] else 0;
            const strm: u32 = @intCast(streamOffset(n_stream, seq_id, 1));

            sinfo.s0 = @min(sinfo.s0, strm);
            sinfo.s1 = @max(sinfo.s1, strm);
            sinfo.strm.items[s] = @intCast(strm);
            sinfo.idxs.items[s].appendNTimes(allocator, 0, 1) catch oom();
        }

        if (n_stream > 1 and sinfo.s1 -% sinfo.s0 +% 1 != ns) {
            impl.throwRuntime(file, 795, "DSV4 compressed streams are not contiguous in ubatch");
        }

        sinfos.appendAssumeCapacity(sinfo);
    }
    return sinfos;
}

/// Ports `dsv4_build_raw_read_sinfos` (llama-kv-cache-dsv4.cpp:804
/// @b29c606e2): the raw cache's read view per ubatch -- the write slots,
/// except that a coupled ubatch reads its first sequence's stream alone.
///
/// Return: the slots, the caller's to free.
fn buildRawReadSinfos(sinfos_write: []const SlotInfo, ubatches: []const Ubatch) std.ArrayList(SlotInfo) {
    var sinfos: std.ArrayList(SlotInfo) = .empty;
    sinfos.ensureTotalCapacity(allocator, ubatches.len) catch oom();

    for (ubatches, 0..) |*ubatch, i| {
        const sinfo_write = &sinfos_write[i];

        if (!ubatchHasCoupled(ubatch)) {
            sinfos.appendAssumeCapacity(sinfo_write.clone());
            continue;
        }

        const seq_id = ubatch.seq_id[0][0];
        var i_stream: usize = 0;
        while (i_stream < sinfo_write.nStream()) : (i_stream += 1) {
            if (sinfo_write.strm.items[i_stream] == seq_id) break;
        }
        if (i_stream == sinfo_write.nStream()) {
            impl.throwRuntime(file, 827, "DSV4 raw write stream not found for coupled read");
        }

        var sinfo: SlotInfo = .{};
        sinfo.s0 = @intCast(sinfo_write.strm.items[i_stream]);
        sinfo.s1 = @intCast(sinfo_write.strm.items[i_stream]);
        sinfo.resize(1);
        sinfo.strm.items[0] = sinfo_write.strm.items[i_stream];
        sinfo.idxs.items[0].appendSlice(allocator, sinfo_write.idxs.items[i_stream].items) catch oom();
        sinfos.appendAssumeCapacity(sinfo);
    }
    return sinfos;
}

/// Ports `dsv4_build_reserve_comp_plan` (llama-kv-cache-dsv4.cpp:842
/// @b29c606e2): the worst-case plan a reserve graph is sized by, its
/// indices all zero.
///
/// Return: the plan, the caller's to `deinit`.
fn buildReserveCompPlan(ubatch: *const Ubatch, ratio: u32, overlap: bool, state_size: u32, kv_size: u32, n_stream: u32, n_rs_seq: u32) CompPlan {
    var plan: CompPlan = .{};
    plan.n_visible.appendNTimes(allocator, 0, ubatch.n_tokens) catch oom();
    plan.n_stream = compGraphNStream(ubatch, n_stream);
    plan.n_kv = kv_size;

    if (ubatch.n_tokens == 0) return plan;

    // The worst case over every sequence split: the sum of each sequence's
    // ceil(tokens/ratio) is at most floor(n_tokens/ratio) + n_seqs.
    const n_seqs: u32 = @max(1, ubatch.n_seqs);
    const n_blocks: usize = @as(usize, ubatch.n_tokens / ratio) + n_seqs;

    const state_rows: u64 = @as(u64, state_size) * n_stream;
    const n_persist: usize = @intCast(@min(@as(u64, ubatch.n_tokens), state_rows));
    const n_restore: usize = if (n_rs_seq > 0) @as(usize, state_size) * n_stream else 0;
    const n_snapshot: usize = @as(usize, n_rs_seq) * state_size * n_stream;

    plan.state_pos.appendNTimes(allocator, 0, ubatch.n_tokens) catch oom();
    plan.state_persist_src_idxs.appendNTimes(allocator, 0, n_persist) catch oom();
    plan.state_persist_dst_idxs.appendNTimes(allocator, 0, n_persist) catch oom();
    plan.state_restore_src_idxs.appendNTimes(allocator, 0, n_restore) catch oom();
    plan.state_restore_dst_idxs.appendNTimes(allocator, 0, n_restore) catch oom();
    plan.state_snapshot_src_idxs.appendNTimes(allocator, 0, n_snapshot) catch oom();
    plan.state_snapshot_dst_idxs.appendNTimes(allocator, 0, n_snapshot) catch oom();
    plan.state_read_idxs.appendNTimes(allocator, 0, @as(usize, if (overlap) 2 else 1) * ratio * n_blocks) catch oom();
    plan.state_write_idxs.appendNTimes(allocator, 0, n_blocks) catch oom();
    plan.state_write_pos.appendNTimes(allocator, 0, n_blocks) catch oom();

    return plan;
}

/// Ports `dsv4_make_k_only` (llama-kv-cache-dsv4.cpp:883 @b29c606e2):
/// `llama_kv_cache` allocates K-only storage when `is_mla()`.
fn makeKOnly(hparams: *Hparams) void {
    hparams.n_embd_head_k_mla_impl = hparams.nEmbdHeadK(0);
    hparams.n_embd_head_v_mla_impl = hparams.nEmbdHeadK(0);
}

// -----------------------------------------------------------------------------
// llama_dsv4_comp_state

/// `llama_dsv4_comp_state`'s `layer` (llama-kv-cache-dsv4.h:53 @b29c606e2).
const CompLayer = struct {
    il: u32,
    kv: *Tensor,
    score: *Tensor,
    kv_stream: std.ArrayList(*Tensor) = .empty,
    score_stream: std.ArrayList(*Tensor) = .empty,
};

/// Ports `llama_dsv4_comp_state` (llama-kv-cache-dsv4.h:11 @b29c606e2): a
/// compressor's running state per layer -- `state_size` rows of the
/// pending window's projected K and gate scores, per stream, with
/// `n_rs_seq` rollback snapshot planes after the current one.
pub const CompState = struct {
    sc_info: StreamCopyInfo = .{},

    ratio: u32,
    state_size: u32,
    n_embd_state: u32,
    n_stream: u32,
    n_rs_seq: u32,

    /// One context and buffer per buffer type, in the buffer types' name order.
    ctxs_bufs: std.ArrayList(struct { ctx: *c.ggml_context, buf: c.ggml_backend_buffer_t }) = .empty,

    layers: std.ArrayList(CompLayer) = .empty,

    /// `map_layer_ids`: a model layer to its index in `layers`.
    map_layer_ids: std.AutoHashMapUnmanaged(i32, i32) = .empty,

    /// Ports the constructor of `llama_dsv4_comp_state`
    /// (llama-kv-cache-dsv4.cpp:893 @b29c606e2).
    ///
    /// Parameters mirror the C++'s; `name` names the state in its tensors
    /// and logs, and `filter` selects its layers.
    ///
    /// Return: the state, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        offload: bool,
        unified: bool,
        n_seq_max: u32,
        ratio: u32,
        state_size: u32,
        n_embd_state: u32,
        n_rs_seq: u32,
        name: [*:0]const u8,
        filter: memory.LayerFilter,
        exc: *Exception,
    ) Thrown!*CompState {
        const func = "llama_dsv4_comp_state";
        const hparams = &model.hparams;

        const self = allocator.create(CompState) catch oom();
        self.* = .{
            .ratio = ratio,
            .state_size = state_size,
            .n_embd_state = n_embd_state,
            .n_stream = if (unified) 1 else n_seq_max,
            .n_rs_seq = n_rs_seq,
        };
        errdefer self.destroy();

        // One context per buffer type, ordered by the buffer type's name.
        var ctx_map: std.ArrayList(struct { buft: c.ggml_backend_buffer_type_t, ctx: *c.ggml_context }) = .empty;
        var n_moved: usize = 0;
        defer {
            for (ctx_map.items[n_moved..]) |e| c.ggml_free(e.ctx);
            ctx_map.deinit(allocator);
        }

        const n_stream = self.n_stream;

        var il: u32 = 0;
        while (il < hparams.nLayer()) : (il += 1) {
            if (filter.func != null and !filter.call(@intCast(il))) continue;

            var dev_name: [*:0]const u8 = "CPU";
            var buft = c.ggml_backend_cpu_buffer_type();
            if (offload) {
                const dev = model.devLayer(il);
                buft = c.ggml_backend_dev_buffer_type(dev);
                dev_name = c.ggml_backend_dev_name(dev);
            }

            impl.logDebug("%s: layer %3d: dev = %s\n", .{ func, il, dev_name });

            // The `ctx_for_buft` lambda.
            const ctx: *c.ggml_context = blk: {
                const bname = std.mem.span(c.ggml_backend_buft_name(buft));
                var pos: usize = 0;
                while (pos < ctx_map.items.len) : (pos += 1) {
                    const order = std.mem.order(u8, std.mem.span(c.ggml_backend_buft_name(ctx_map.items[pos].buft)), bname);
                    if (order == .eq) break :blk ctx_map.items[pos].ctx;
                    if (order == .gt) break;
                }
                const ctx = c.ggml_init(.{
                    .mem_size = 2 * (1 + @as(usize, n_stream)) * hparams.nLayer() * c.ggml_tensor_overhead(),
                    .mem_buffer = null,
                    .no_alloc = true,
                }) orelse return exc.throw(.runtime_error, "failed to create ggml context for DSV4 compressor state", .{});
                ctx_map.insert(allocator, pos, .{ .buft = buft, .ctx = ctx }) catch oom();
                break :blk ctx;
            };

            const n_planes: i64 = @as(i64, n_stream) * (1 + n_rs_seq);
            const kv: *Tensor = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, n_embd_state, state_size, n_planes);
            const score: *Tensor = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, n_embd_state, state_size, n_planes);

            _ = c.ggml_format_name(kv, "dsv4_%s_state_kv_l%d", name, @as(c_int, @intCast(il)));
            _ = c.ggml_format_name(score, "dsv4_%s_state_score_l%d", name, @as(c_int, @intCast(il)));

            var layer: CompLayer = .{ .il = il, .kv = kv, .score = score };
            var s: u32 = 0;
            while (s < n_stream) : (s += 1) {
                layer.kv_stream.append(allocator, c.ggml_view_2d(ctx, kv, n_embd_state, state_size, kv.nb[1], s * kv.nb[2])) catch oom();
                layer.score_stream.append(allocator, c.ggml_view_2d(ctx, score, n_embd_state, state_size, score.nb[1], s * score.nb[2])) catch oom();
            }

            self.map_layer_ids.put(allocator, @intCast(il), @intCast(self.layers.items.len)) catch oom();
            self.layers.append(allocator, layer) catch oom();
        }

        for (ctx_map.items) |e| {
            const buf = c.ggml_backend_alloc_ctx_tensors_from_buft(e.ctx, e.buft);
            if (buf == null) return exc.throw(.runtime_error, "failed to allocate buffer for DSV4 compressor state", .{});

            c.ggml_backend_buffer_clear(buf, 0);

            impl.logInfo("%s: %10s DSV4 %s state buffer size = %8.2f MiB\n", .{ func, c.ggml_backend_buffer_name(buf), name, @as(f64, @floatFromInt(c.ggml_backend_buffer_get_size(buf))) / 1024.0 / 1024.0 });

            self.ctxs_bufs.append(allocator, .{ .ctx = e.ctx, .buf = buf }) catch oom();
            n_moved += 1;
        }

        impl.logInfo("%s: %s ratio = %u, state = %u x %u, streams = %u, rs_seq = %u, layers = %zu, size = %7.2f MiB\n", .{ func, name, ratio, state_size, n_embd_state, n_stream, n_rs_seq, self.layers.items.len, @as(f64, @floatFromInt(self.totalSize())) / 1024.0 / 1024.0 });

        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *CompState) void {
        for (self.ctxs_bufs.items) |e| {
            c.ggml_backend_buffer_free(e.buf);
            c.ggml_free(e.ctx);
        }
        self.ctxs_bufs.deinit(allocator);
        for (self.layers.items) |*l| {
            l.kv_stream.deinit(allocator);
            l.score_stream.deinit(allocator);
        }
        self.layers.deinit(allocator);
        self.map_layer_ids.deinit(allocator);
        self.sc_info.deinit();
        allocator.destroy(self);
    }

    /// Ports `llama_dsv4_comp_state`'s `clear` (llama-kv-cache-dsv4.cpp:1002
    /// @b29c606e2): one sequence's planes, or every buffer.
    pub fn clear(self: *CompState, seq_id: i32, data: bool) void {
        if (!data) return;

        if (seq_id >= 0) {
            impl.assert(@as(u32, @bitCast(seq_id)) < self.n_stream, file, 1008, "(uint32_t) seq_id < n_stream");

            for (self.layers.items) |*layer| {
                var d: u32 = 0;
                while (d <= self.n_rs_seq) : (d += 1) {
                    const stream = d * self.n_stream + @as(u32, @bitCast(seq_id));
                    clearTensorStream(layer.kv, stream);
                    clearTensorStream(layer.score, stream);
                }
            }
            return;
        }

        for (self.ctxs_bufs.items) |e| c.ggml_backend_buffer_clear(e.buf, 0);
    }

    /// Ports `llama_dsv4_comp_state`'s `seq_cp` (llama-kv-cache-dsv4.cpp:1025
    /// @b29c606e2): clear the destination now and queue the stream copy for
    /// the next update.
    pub fn seqCp(self: *CompState, seq_id_src: i32, seq_id_dst: i32) void {
        impl.assert(seq_id_src >= 0 and @as(u32, @bitCast(seq_id_src)) < self.n_stream, file, 1026, "seq_id_src >= 0 && (uint32_t) seq_id_src < n_stream");
        impl.assert(seq_id_dst >= 0 and @as(u32, @bitCast(seq_id_dst)) < self.n_stream, file, 1027, "seq_id_dst >= 0 && (uint32_t) seq_id_dst < n_stream");

        if (seq_id_src == seq_id_dst) return;

        self.clear(seq_id_dst, true);

        self.sc_info.ssrc.append(allocator, @bitCast(seq_id_src)) catch oom();
        self.sc_info.sdst.append(allocator, @bitCast(seq_id_dst)) catch oom();
    }

    /// Ports `llama_dsv4_comp_state`'s `apply_copies` (llama-kv-cache-dsv4.cpp:1039 @b29c606e2).
    pub fn applyCopies(self: *const CompState, sc_info: *const StreamCopyInfo) void {
        for (sc_info.ssrc.items, sc_info.sdst.items) |ssrc, sdst| {
            for (self.layers.items) |*layer| {
                c.ggml_backend_tensor_copy(layer.kv_stream.items[ssrc], layer.kv_stream.items[sdst]);
                c.ggml_backend_tensor_copy(layer.score_stream.items[ssrc], layer.score_stream.items[sdst]);
            }
        }
    }

    /// Ports `llama_dsv4_comp_state`'s `get_ratio` (llama-kv-cache-dsv4.cpp:1051 @b29c606e2).
    pub fn getRatio(self: *const CompState) u32 {
        return self.ratio;
    }

    /// Ports `llama_dsv4_comp_state`'s `get_state_size` (llama-kv-cache-dsv4.cpp:1055 @b29c606e2).
    pub fn getStateSize(self: *const CompState) u32 {
        return self.state_size;
    }

    /// Ports `llama_dsv4_comp_state`'s `get_n_stream` (llama-kv-cache-dsv4.cpp:1059 @b29c606e2).
    pub fn getNStream(self: *const CompState) u32 {
        return self.n_stream;
    }

    /// Ports `llama_dsv4_comp_state`'s `get_n_rs_seq` (llama-kv-cache-dsv4.cpp:1063 @b29c606e2).
    pub fn getNRsSeq(self: *const CompState) u32 {
        return self.n_rs_seq;
    }

    /// Ports `llama_dsv4_comp_state`'s `get_n_rows` (llama-kv-cache-dsv4.cpp:1067
    /// @b29c606e2): the current plane's rows across every stream.
    pub fn getNRows(self: *const CompState) u32 {
        return self.state_size *% self.n_stream;
    }

    /// Ports `llama_dsv4_comp_state`'s `memory_breakdown` (llama-kv-cache-dsv4.cpp:1071 @b29c606e2).
    pub fn memoryBreakdown(self: *const CompState, out: *std.ArrayList(memory.BuftSize)) void {
        for (self.ctxs_bufs.items) |e| {
            memory.addBuftSize(out, c.ggml_backend_buffer_get_type(e.buf), c.ggml_backend_buffer_get_size(e.buf));
        }
    }

    /// Ports `llama_dsv4_comp_state`'s `state_write` (llama-kv-cache-dsv4.cpp:1080
    /// @b29c606e2): each saved stream's plane is the one its pending
    /// rollback, if any, points at.
    pub fn stateWrite(self: *const CompState, w: io.Writer, seq_id: i32, flags: u32, rs_idx: []const u32, exc: *Exception) Thrown!void {
        _ = flags;

        var s0: u32 = undefined;
        var ns: u32 = undefined;
        try stateSrcStreamRange(self.n_stream, seq_id, &s0, &ns, exc);

        const stream_ids = allocator.alloc(u32, ns) catch oom();
        defer allocator.free(stream_ids);
        for (0..ns) |s| {
            const seq: u32 = if (seq_id >= 0) @bitCast(seq_id) else s0 + @as(u32, @intCast(s));
            if (seq >= rs_idx.len or rs_idx[seq] > self.n_rs_seq) {
                return exc.throw(.runtime_error, "DSV4 recurrent state rollback index out of range", .{});
            }
            stream_ids[s] = rs_idx[seq] *% self.n_stream +% s0 +% @as(u32, @intCast(s));
        }

        const version: u32 = comp_state_ver;
        const n_layer: u32 = @intCast(self.layers.items.len);

        try w.writeValue(version, exc);
        try w.writeValue(self.ratio, exc);
        try w.writeValue(self.state_size, exc);
        try w.writeValue(self.n_embd_state, exc);
        try w.writeValue(ns, exc);
        try w.writeValue(n_layer, exc);

        for (self.layers.items) |*layer| {
            try w.writeValue(layer.il, exc);

            try stateWriteTensorStreams(w, layer.kv, self.state_size, self.state_size, s0, ns, stream_ids, exc);
            try stateWriteTensorStreams(w, layer.score, self.state_size, self.state_size, s0, ns, stream_ids, exc);
        }
    }

    /// Ports `llama_dsv4_comp_state`'s `state_read` (llama-kv-cache-dsv4.cpp:1118 @b29c606e2).
    pub fn stateRead(self: *CompState, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        _ = flags;

        const version = try r.readValue(u32, exc);
        const ratio_ref = try r.readValue(u32, exc);
        const state_size_ref = try r.readValue(u32, exc);
        const n_embd_state_ref = try r.readValue(u32, exc);
        const ns = try r.readValue(u32, exc);
        const n_layer_ref = try r.readValue(u32, exc);

        if (version != comp_state_ver) return exc.throw(.runtime_error, "DSV4 compressor state version mismatch", .{});
        if (ratio_ref != self.ratio or state_size_ref != self.state_size or n_embd_state_ref != self.n_embd_state) {
            return exc.throw(.runtime_error, "DSV4 compressor state metadata mismatch", .{});
        }
        if (n_layer_ref != self.layers.items.len) return exc.throw(.runtime_error, "DSV4 compressor state layer count mismatch", .{});

        var s0: u32 = undefined;
        try stateDstStreamRange(self.n_stream, seq_id, ns, &s0, exc);

        for (self.layers.items) |*layer| {
            const il_ref = try r.readValue(u32, exc);
            if (il_ref != layer.il) return exc.throw(.runtime_error, "DSV4 compressor state layer id mismatch", .{});

            try stateReadTensorStreams(r, layer.kv, self.state_size, self.state_size, s0, ns, exc);
            try stateReadTensorStreams(r, layer.score, self.state_size, self.state_size, s0, ns, exc);
        }
    }

    fn layerAt(self: *const CompState, il: i32) *const CompLayer {
        const ids = self.map_layer_ids.get(il) orelse impl.abort(file, 1161, "map::at:  key not found");
        return &self.layers.items[@intCast(ids)];
    }

    /// Ports `llama_dsv4_comp_state`'s `get_kv_all` (llama-kv-cache-dsv4.cpp:1160
    /// @b29c606e2): every plane of every stream, as rows.
    pub fn getKvAll(self: *const CompState, ctx: *c.ggml_context, il: i32) *Tensor {
        const state = self.layerAt(il).kv;
        return c.ggml_view_2d(ctx, state, state.ne[0], @as(i64, self.getNRows()) * (1 + self.n_rs_seq), state.nb[1], 0);
    }

    /// Ports `llama_dsv4_comp_state`'s `get_score_all` (llama-kv-cache-dsv4.cpp:1167 @b29c606e2).
    pub fn getScoreAll(self: *const CompState, ctx: *c.ggml_context, il: i32) *Tensor {
        const state = self.layerAt(il).score;
        return c.ggml_view_2d(ctx, state, state.ne[0], @as(i64, self.getNRows()) * (1 + self.n_rs_seq), state.nb[1], 0);
    }

    /// Ports `llama_dsv4_comp_state`'s `get_kv` (llama-kv-cache-dsv4.cpp:1174
    /// @b29c606e2): the current plane alone.
    pub fn getKv(self: *const CompState, ctx: *c.ggml_context, il: i32) *Tensor {
        const state: *Tensor = self.getKvAll(ctx, il);
        const row_size = c.ggml_row_size(state.type, state.ne[0]);
        return c.ggml_view_2d(ctx, state, state.ne[0], self.getNRows(), state.nb[1], 0 * row_size);
    }

    /// Ports `llama_dsv4_comp_state`'s `get_score` (llama-kv-cache-dsv4.cpp:1181 @b29c606e2).
    pub fn getScore(self: *const CompState, ctx: *c.ggml_context, il: i32) *Tensor {
        const state: *Tensor = self.getScoreAll(ctx, il);
        const row_size = c.ggml_row_size(state.type, state.ne[0]);
        return c.ggml_view_2d(ctx, state, state.ne[0], self.getNRows(), state.nb[1], 0 * row_size);
    }

    /// Ports `llama_dsv4_comp_state`'s `cpy_kv` (llama-kv-cache-dsv4.cpp:1188 @b29c606e2).
    pub fn cpyKv(self: *const CompState, ctx: *c.ggml_context, cur: *Tensor, idxs: *Tensor, il: i32) *Tensor {
        return c.ggml_set_rows(ctx, self.getKvAll(ctx, il), cur, idxs);
    }

    /// Ports `llama_dsv4_comp_state`'s `cpy_score` (llama-kv-cache-dsv4.cpp:1192 @b29c606e2).
    pub fn cpyScore(self: *const CompState, ctx: *c.ggml_context, cur: *Tensor, idxs: *Tensor, il: i32) *Tensor {
        return c.ggml_set_rows(ctx, self.getScoreAll(ctx, il), cur, idxs);
    }

    /// Ports `llama_dsv4_comp_state`'s `total_size` (llama-kv-cache-dsv4.cpp:1196 @b29c606e2).
    pub fn totalSize(self: *const CompState) usize {
        var size: usize = 0;
        for (self.ctxs_bufs.items) |e| size += c.ggml_backend_buffer_get_size(e.buf);
        return size;
    }
};

// -----------------------------------------------------------------------------
// llama_kv_cache_dsv4

/// The constructor's `filter_raw` lambda (llama-kv-cache-dsv4.cpp:1233
/// @b29c606e2): the caller's filter alone. It lives on the constructor's
/// stack, as the C++'s by-reference captures do; the caches read their
/// filters only while they are built.
const FilterRaw = struct {
    outer: memory.LayerFilter,

    fn call(ctx: ?*const anyopaque, il: i32) bool {
        const self: *const FilterRaw = @ptrCast(@alignCast(ctx.?));
        if (self.outer.func != null and !self.outer.call(il)) return false;
        return true;
    }

    fn filter(self: *const FilterRaw) memory.LayerFilter {
        return .{ .ctx = self, .func = call };
    }
};

/// The constructor's `filter_csa` and `filter_hca` lambdas
/// (llama-kv-cache-dsv4.cpp:1271, 1279 @b29c606e2): the caller's filter,
/// then the layer's compression ratio.
const FilterRatio = struct {
    outer: memory.LayerFilter,
    hparams: *const Hparams,
    ratio: u32,

    fn call(ctx: ?*const anyopaque, il: i32) bool {
        const self: *const FilterRatio = @ptrCast(@alignCast(ctx.?));
        if (self.outer.func != null and !self.outer.call(il)) return false;
        return self.hparams.dsv4_compress_ratios[@intCast(il)] == self.ratio;
    }

    fn filter(self: *const FilterRatio) memory.LayerFilter {
        return .{ .ctx = self, .func = call };
    }
};

/// Ports `llama_kv_cache_dsv4` (llama-kv-cache-dsv4.h:88 @b29c606e2): the
/// raw iSWA token cache, the three compressed K-only caches and their
/// compressor states, driven as one memory.
///
/// The four hyperparameter copies are fields because each cache keeps a
/// pointer to its own; the cache must not move once built.
pub const KvCacheDsv4 = struct {
    hparams_raw: Hparams,
    hparams_csa: Hparams,
    hparams_hca: Hparams,
    hparams_lid: Hparams,

    n_seq_max: u32,
    n_rs_seq: u32,

    /// Per sequence, the pending rollback depth; 0 for none.
    rs_idx: std.ArrayList(u32) = .empty,

    kv_raw: ?*KvCacheIswa = null,
    kv_csa: ?*KvCache = null,
    kv_hca: ?*KvCache = null,
    kv_lid: ?*KvCache = null,
    csa_state: ?*CompState = null,
    hca_state: ?*CompState = null,
    lid_state: ?*CompState = null,

    /// Ports the constructor of `llama_kv_cache_dsv4`
    /// (llama-kv-cache-dsv4.cpp:1210 @b29c606e2).
    ///
    /// Parameters mirror the C++'s; `filter` and `reuse` may be empty.
    ///
    /// Return: the cache, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(
        model: *const model_mod.Model,
        type_k: c.enum_ggml_type,
        type_v: c.enum_ggml_type,
        v_trans: bool,
        offload: bool,
        swa_full: bool,
        unified: bool,
        kv_size: u32,
        n_seq_max: u32,
        n_ubatch: u32,
        n_pad: u32,
        n_rs_seq: u32,
        filter: memory.LayerFilter,
        reuse: memory.LayerMap,
        exc: *Exception,
    ) Thrown!*KvCacheDsv4 {
        const func = "llama_kv_cache_dsv4";
        const mhp = &model.hparams;

        const self = allocator.create(KvCacheDsv4) catch oom();
        self.* = .{
            .hparams_raw = mhp.*,
            .hparams_csa = mhp.*,
            .hparams_hca = mhp.*,
            .hparams_lid = mhp.*,
            .n_seq_max = n_seq_max,
            .n_rs_seq = n_rs_seq,
        };
        errdefer self.destroy();
        self.rs_idx.appendNTimes(allocator, 0, n_seq_max) catch oom();

        const filter_raw: FilterRaw = .{ .outer = filter };

        // The public KV mode may be unified, but the C++ keeps DSV4's KV and
        // state streams per sequence regardless.
        _ = unified;
        const unified_raw = false;

        self.hparams_raw.n_layer_nextn = 0;
        self.hparams_csa.n_layer_nextn = 0;
        self.hparams_hca.n_layer_nextn = 0;
        self.hparams_lid.n_layer_nextn = 0;

        impl.logInfo("%s: creating DSV4 raw KV cache\n", .{func});

        makeKOnly(&self.hparams_raw);

        self.kv_raw = try KvCacheIswa.create(model, &self.hparams_raw, type_k, type_v, v_trans, offload, swa_full, unified_raw, kv_size, n_seq_max, n_ubatch, n_pad, null, filter_raw.filter(), reuse, .{}, exc);

        makeKOnly(&self.hparams_csa);
        makeKOnly(&self.hparams_hca);

        @memset(&self.hparams_lid.n_head_kv_arr, 1);
        self.hparams_lid.n_embd_head_k_full = mhp.indexer_head_size;
        self.hparams_lid.n_embd_head_v_full = mhp.indexer_head_size;
        self.hparams_lid.n_embd_head_k_swa = mhp.indexer_head_size;
        self.hparams_lid.n_embd_head_v_swa = mhp.indexer_head_size;
        self.hparams_lid.rope_type = c.LLAMA_ROPE_TYPE_NEOX;
        makeKOnly(&self.hparams_lid);

        const filter_csa: FilterRatio = .{ .outer = filter, .hparams = mhp, .ratio = csa_ratio };
        const filter_hca: FilterRatio = .{ .outer = filter, .hparams = mhp, .ratio = hca_ratio };

        const unified_compressed = false;

        impl.logInfo("%s: creating DSV4 CSA compressed KV cache, size = %u cells\n", .{ func, compSize(kv_size, csa_ratio) });

        self.kv_csa = try KvCache.create(model, &self.hparams_csa, type_k, type_v, v_trans, offload, unified_compressed, padTo(compSize(kv_size, csa_ratio), 256), n_seq_max, n_pad, 0, .none, null, filter_csa.filter(), .{}, .{}, exc);

        impl.logInfo("%s: creating DSV4 HCA compressed KV cache, size = %u cells\n", .{ func, compSize(kv_size, hca_ratio) });

        self.kv_hca = try KvCache.create(model, &self.hparams_hca, type_k, type_v, v_trans, offload, unified_compressed, padTo(compSize(kv_size, hca_ratio), 256), n_seq_max, n_pad, 0, .none, null, filter_hca.filter(), .{}, .{}, exc);

        impl.logInfo("%s: creating DSV4 lightning-indexer KV cache, size = %u cells\n", .{ func, compSize(kv_size, csa_ratio) });

        self.kv_lid = try KvCache.create(model, &self.hparams_lid, type_k, type_v, v_trans, offload, unified_compressed, padTo(compSize(kv_size, csa_ratio), 256), n_seq_max, n_pad, 0, .none, null, filter_csa.filter(), .{}, .{}, exc);

        impl.logInfo("%s: creating DSV4 CSA compressor state\n", .{func});

        self.csa_state = try CompState.create(model, offload, unified_compressed, n_seq_max, csa_ratio, 2 * csa_ratio, 2 *% mhp.nEmbdHeadK(0), n_rs_seq, "csa", filter_csa.filter(), exc);

        impl.logInfo("%s: creating DSV4 HCA compressor state\n", .{func});

        self.hca_state = try CompState.create(model, offload, unified_compressed, n_seq_max, hca_ratio, hca_ratio, mhp.nEmbdHeadK(0), n_rs_seq, "hca", filter_hca.filter(), exc);

        impl.logInfo("%s: creating DSV4 lightning-indexer compressor state\n", .{func});

        self.lid_state = try CompState.create(model, offload, unified_compressed, n_seq_max, csa_ratio, 2 * csa_ratio, 2 *% mhp.indexer_head_size, n_rs_seq, "lid", filter_csa.filter(), exc);

        // DSV4 attention reads compressed-K and compressor-state rows the
        // current graph need not overwrite; uninitialized buffer contents
        // would leak in (instance-specific garbage) and corrupt recall.
        // Zero every compressed buffer up front, so reads of rows never
        // written are deterministic.
        self.clearCompressed(-1, true);

        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *KvCacheDsv4) void {
        if (self.kv_raw) |k| k.destroy();
        if (self.kv_csa) |k| k.destroy();
        if (self.kv_hca) |k| k.destroy();
        if (self.kv_lid) |k| k.destroy();
        if (self.csa_state) |s| s.destroy();
        if (self.hca_state) |s| s.destroy();
        if (self.lid_state) |s| s.destroy();
        self.rs_idx.deinit(allocator);
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsv4`'s `init_batch` (llama-kv-cache-dsv4.cpp:1338
    /// @b29c606e2): the iSWA cache's simple split when neither the raw nor
    /// the compressed state needs a per-sequence layout, then an equal split
    /// -- or a sequence split, for coupled sequence sets, which serialize
    /// until DSV4 has explicit shared-state handling for compressed streams.
    pub fn initBatch(self: *KvCacheDsv4, balloc: *batch.BatchAllocr, n_ubatch: u32, embd_all: bool) *Dsv4Context {
        _ = embd_all;

        const raw_per_seq = self.getRaw().getBase().getNStream() != 1;
        const comp_per_seq = self.getCsaState().getNStream() > 1;
        const has_coupled = batchHasCoupled(balloc.getBatch());

        // The iSWA cache's split, when the compressed state does not need a
        // per-sequence graph layout.
        if (!(raw_per_seq or comp_per_seq)) {
            balloc.splitReset();

            var ubatches: std.ArrayList(Ubatch) = .empty;
            while (true) {
                const ub = balloc.splitSimple(n_ubatch);
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }

            if (balloc.getNUsed() >= balloc.getNTokens()) {
                if (self.makeContext(ubatches)) |ctx| return ctx;
            } else {
                releaseAll(&ubatches);
            }
        }

        // With per-sequence raw or compressed state, independent sequences
        // can share an equal-length ubatch.
        {
            balloc.splitReset();

            var ubatches: std.ArrayList(Ubatch) = .empty;
            while (true) {
                const ub = if (has_coupled)
                    balloc.splitSeq(n_ubatch)
                else
                    // [TAG_RECURRENT_ROLLBACK_SPLITS] The trailing
                    // (1 + n_rs_seq) tokens of each sequence must stay in
                    // the same ubatch.
                    balloc.splitEqual(n_ubatch, raw_per_seq or comp_per_seq, if (self.n_rs_seq > 0) self.n_rs_seq + 1 else 0);
                if (ub.n_tokens == 0) break;
                ubatches.append(allocator, ub) catch oom();
            }

            if (balloc.getNUsed() >= balloc.getNTokens()) {
                if (self.makeContext(ubatches)) |ctx| return ctx;
            } else {
                releaseAll(&ubatches);
            }
        }

        return Dsv4Context.createStatus(.failed_prepare);
    }

    /// `init_batch`'s `make_context` lambda (llama-kv-cache-dsv4.cpp:1348
    /// @b29c606e2): place the raw writes in both halves of the raw cache.
    /// Takes `ubatches`, and releases them when they do not fit.
    fn makeContext(self: *KvCacheDsv4, ubatches_: std.ArrayList(Ubatch)) ?*Dsv4Context {
        var ubatches = ubatches_;
        var ubatches_raw = buildRawWriteUbatches(ubatches.items);

        const kv_raw = self.getRaw();
        var sinfos_base = kv_raw.getBase().prepare(ubatches_raw.items) orelse {
            releaseAll(&ubatches_raw);
            releaseAll(&ubatches);
            return null;
        };
        if (sinfos_base.items.len == 0) {
            sinfos_base.deinit(allocator);
            releaseAll(&ubatches_raw);
            releaseAll(&ubatches);
            return null;
        }

        var sinfos_swa = kv_raw.getSwa().prepare(ubatches_raw.items) orelse {
            freeSlotInfos(&sinfos_base);
            releaseAll(&ubatches_raw);
            releaseAll(&ubatches);
            return null;
        };
        if (sinfos_swa.items.len == 0) {
            sinfos_swa.deinit(allocator);
            freeSlotInfos(&sinfos_base);
            releaseAll(&ubatches_raw);
            releaseAll(&ubatches);
            return null;
        }

        const sinfos_swa_read = buildRawReadSinfos(sinfos_swa.items, ubatches.items);

        return Dsv4Context.createBatch(self, sinfos_base, sinfos_swa, sinfos_swa_read, ubatches, ubatches_raw);
    }

    /// Ports `llama_kv_cache_dsv4`'s `init_full` (llama-kv-cache-dsv4.cpp:1434 @b29c606e2).
    pub fn initFull(self: *KvCacheDsv4) *Dsv4Context {
        return Dsv4Context.createFull(self);
    }

    /// Ports `llama_kv_cache_dsv4`'s `init_update` (llama-kv-cache-dsv4.cpp:1438
    /// @b29c606e2): the context takes the compressor states' queued copies.
    pub fn initUpdate(self: *KvCacheDsv4, lctx: *anyopaque, optimize: bool) *Dsv4Context {
        const sc_csa = self.getCsaState().sc_info;
        self.getCsaState().sc_info = .{};
        const sc_hca = self.getHcaState().sc_info;
        self.getHcaState().sc_info = .{};
        const sc_lid = self.getLidState().sc_info;
        self.getLidState().sc_info = .{};
        return Dsv4Context.createUpdate(self, lctx, optimize, sc_csa, sc_hca, sc_lid);
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_can_shift` (llama-kv-cache-dsv4.cpp:1448
    /// @b29c606e2): compressed rows carry block-derived positions, so
    /// shifting stays off until DSV4's compressed-cache shift semantics are
    /// wired.
    pub fn getCanShift(self: *const KvCacheDsv4) bool {
        _ = self;
        return false;
    }

    /// Ports `llama_kv_cache_dsv4`'s `clear` (llama-kv-cache-dsv4.cpp:1454
    /// @b29c606e2): the compressed buffers are cleared with their data
    /// always, so they never expose stale or uninitialized rows.
    pub fn clear(self: *KvCacheDsv4, data: bool) void {
        self.getRaw().clear(data);
        self.clearCompressed(-1, true);
    }

    /// Ports `llama_kv_cache_dsv4`'s `seq_rm` (llama-kv-cache-dsv4.cpp:1459
    /// @b29c606e2): only a tail can go. Past a sequence's end every cache
    /// drops it; within the last `n_rs_seq` positions the compressor states
    /// roll back to a snapshot on the next ubatch; from the start the whole
    /// sequence goes.
    pub fn seqRm(self: *KvCacheDsv4, seq_id: i32, p0: i32, p1: i32) bool {
        if (p1 >= 0) return false;

        if (p0 > 0) {
            if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= self.n_seq_max) return false;

            const pos_max = self.getRaw().seqPosMax(seq_id);
            if (p0 > pos_max) {
                var res = true;

                const r_raw = self.getRaw().seqRm(seq_id, p0, -1);
                res = res and r_raw;
                const r_csa = self.getCsa().seqRm(seq_id, @divTrunc(p0, @as(i32, csa_ratio)), -1);
                res = res and r_csa;
                const r_hca = self.getHca().seqRm(seq_id, @divTrunc(p0, @as(i32, hca_ratio)), -1);
                res = res and r_hca;
                const r_lid = self.getLid().seqRm(seq_id, @divTrunc(p0, @as(i32, csa_ratio)), -1);
                res = res and r_lid;

                return res;
            }

            if (self.n_rs_seq == 0) return false;

            const rollback: i32 = pos_max -% (p0 - 1);
            if (rollback < 1 or rollback > @as(i32, @bitCast(self.n_rs_seq))) return false;

            // A pending rollback is single-use: stacked partial removals do
            // not compose.
            if (self.rs_idx.items[@intCast(seq_id)] != 0) return false;

            const res = self.getRaw().seqRm(seq_id, p0, p1);
            if (res) self.rs_idx.items[@intCast(seq_id)] = @bitCast(rollback);

            return res;
        }

        const res = self.getRaw().seqRm(seq_id, p0, p1);

        if (res) self.clearCompressed(seq_id, true);

        return res;
    }

    /// Ports `llama_kv_cache_dsv4`'s `seq_cp` (llama-kv-cache-dsv4.cpp:1512
    /// @b29c606e2): whole sequences only.
    pub fn seqCp(self: *KvCacheDsv4, seq_id_src: i32, seq_id_dst: i32, p0: i32, p1: i32) void {
        impl.assert(p0 <= 0 and p1 < 0, file, 1513, "p0 <= 0 && p1 < 0 && \"DSV4 only supports full sequence copies\"");

        self.getRaw().seqCp(seq_id_src, seq_id_dst, p0, p1);
        self.getCsa().seqCp(seq_id_src, seq_id_dst, -1, -1);
        self.getHca().seqCp(seq_id_src, seq_id_dst, -1, -1);
        self.getLid().seqCp(seq_id_src, seq_id_dst, -1, -1);

        self.getCsaState().seqCp(seq_id_src, seq_id_dst);
        self.getHcaState().seqCp(seq_id_src, seq_id_dst);
        self.getLidState().seqCp(seq_id_src, seq_id_dst);

        if (seq_id_src != seq_id_dst) self.rs_idx.items[@intCast(seq_id_dst)] = 0;
    }

    /// Ports `llama_kv_cache_dsv4`'s `seq_keep` (llama-kv-cache-dsv4.cpp:1529 @b29c606e2).
    pub fn seqKeep(self: *KvCacheDsv4, seq_id: i32) void {
        impl.assert(seq_id >= 0 and @as(u32, @bitCast(seq_id)) < self.n_seq_max, file, 1530, "seq_id >= 0 && (uint32_t) seq_id < n_seq_max");

        self.getRaw().seqKeep(seq_id);

        var id: i32 = 0;
        while (id < @as(i32, @bitCast(self.n_seq_max))) : (id += 1) {
            if (id == seq_id) continue;

            _ = self.getRaw().seqRm(id, -1, -1);
            self.clearCompressed(id, true);
        }
    }

    /// Ports `llama_kv_cache_dsv4`'s `seq_add` (llama-kv-cache-dsv4.cpp:1544 @b29c606e2).
    pub fn seqAdd(self: *KvCacheDsv4, seq_id: i32, p0: i32, p1: i32, shift: i32) void {
        self.getRaw().seqAdd(seq_id, p0, p1, shift);
    }

    /// Ports `llama_kv_cache_dsv4`'s `seq_div` (llama-kv-cache-dsv4.cpp:1548 @b29c606e2).
    pub fn seqDiv(self: *KvCacheDsv4, seq_id: i32, p0: i32, p1: i32, d: c_int) void {
        self.getRaw().seqDiv(seq_id, p0, p1, d);
    }

    /// Ports `llama_kv_cache_dsv4`'s `seq_pos_min` (llama-kv-cache-dsv4.cpp:1552
    /// @b29c606e2): the raw SWA cache may hold a wider window, but the
    /// compressed state cannot roll back within it, so only the current
    /// boundary is reported, and the server uses checkpoints for rollback.
    pub fn seqPosMin(self: *const KvCacheDsv4, seq_id: i32) i32 {
        if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= self.n_seq_max) return -1;
        return self.getRaw().seqPosMax(seq_id);
    }

    /// Ports `llama_kv_cache_dsv4`'s `seq_pos_max` (llama-kv-cache-dsv4.cpp:1563 @b29c606e2).
    pub fn seqPosMax(self: *const KvCacheDsv4, seq_id: i32) i32 {
        if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= self.n_seq_max) return -1;
        return self.getRaw().seqPosMax(seq_id);
    }

    /// Ports `llama_kv_cache_dsv4`'s `memory_breakdown` (llama-kv-cache-dsv4.cpp:1571 @b29c606e2).
    pub fn memoryBreakdown(self: *const KvCacheDsv4, out: *std.ArrayList(memory.BuftSize)) void {
        self.getRaw().memoryBreakdown(out);
        self.getCsa().memoryBreakdown(out);
        self.getHca().memoryBreakdown(out);
        self.getLid().memoryBreakdown(out);
        self.getCsaState().memoryBreakdown(out);
        self.getHcaState().memoryBreakdown(out);
        self.getLidState().memoryBreakdown(out);
    }

    /// Ports `llama_kv_cache_dsv4`'s `state_write` (llama-kv-cache-dsv4.cpp:1594
    /// @b29c606e2): a header, the raw cache, the used rows of the compressed
    /// caches unless partial, then the compressor states.
    pub fn stateWrite(self: *const KvCacheDsv4, w: io.Writer, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        const partial_only = flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY != 0;

        const magic: u32 = state_magic;
        const version: u32 = state_version;
        const mode: u32 = if (partial_only) state_mode_partial else state_mode_full;

        try w.writeValue(magic, exc);
        try w.writeValue(version, exc);
        try w.writeValue(mode, exc);

        try self.getRaw().stateWrite(w, seq_id, flags, exc);

        if (!partial_only) {
            const pos_max: i32 = if (seq_id >= 0) self.getRaw().seqPosMax(seq_id) else -1;

            // The C++'s FIXME: token positions are conflated with rows,
            // which is not true for the multi-modal case.
            const n_rows_csa = if (seq_id >= 0) stateNUsedKRows(pos_max, csa_ratio, self.getCsa().getSize()) else self.getCsa().getSize();
            const n_rows_hca = if (seq_id >= 0) stateNUsedKRows(pos_max, hca_ratio, self.getHca().getSize()) else self.getHca().getSize();
            const n_rows_lid = if (seq_id >= 0) stateNUsedKRows(pos_max, csa_ratio, self.getLid().getSize()) else self.getLid().getSize();

            try stateWriteKCache(w, self.getCsa(), seq_id, flags, n_rows_csa, exc);
            try stateWriteKCache(w, self.getHca(), seq_id, flags, n_rows_hca, exc);
            try stateWriteKCache(w, self.getLid(), seq_id, flags, n_rows_lid, exc);
        }

        try self.getCsaState().stateWrite(w, seq_id, flags, self.rs_idx.items, exc);
        try self.getHcaState().stateWrite(w, seq_id, flags, self.rs_idx.items, exc);
        try self.getLidState().stateWrite(w, seq_id, flags, self.rs_idx.items, exc);
    }

    /// Ports `llama_kv_cache_dsv4`'s `state_read` (llama-kv-cache-dsv4.cpp:1628
    /// @b29c606e2): a restored sequence has no rollback pending.
    pub fn stateRead(self: *KvCacheDsv4, r: io.Reader, seq_id: i32, flags: u32, exc: *Exception) Thrown!void {
        const magic = try r.readValue(u32, exc);
        const version = try r.readValue(u32, exc);

        if (magic != state_magic) return exc.throw(.runtime_error, "DSV4 state magic mismatch", .{});
        if (version != state_version) return exc.throw(.runtime_error, "DSV4 state version mismatch", .{});

        const mode = try r.readValue(u32, exc);
        if (mode != state_mode_full and mode != state_mode_partial) return exc.throw(.runtime_error, "DSV4 state mode mismatch", .{});

        const partial_only = mode == state_mode_partial;
        if (partial_only != (flags & c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY != 0)) {
            return exc.throw(.runtime_error, "DSV4 state flags mismatch", .{});
        }

        try self.getRaw().stateRead(r, seq_id, flags, exc);

        if (!partial_only) {
            self.clearCompressed(seq_id, true);

            try stateReadKCache(r, self.getCsa(), seq_id, flags, exc);
            try stateReadKCache(r, self.getHca(), seq_id, flags, exc);
            try stateReadKCache(r, self.getLid(), seq_id, flags, exc);
        }

        try self.getCsaState().stateRead(r, seq_id, flags, exc);
        try self.getHcaState().stateRead(r, seq_id, flags, exc);
        try self.getLidState().stateRead(r, seq_id, flags, exc);

        if (seq_id >= 0) {
            impl.assert(@as(u32, @bitCast(seq_id)) < self.n_seq_max, file, 1668, "(uint32_t) seq_id < n_seq_max");
            self.rs_idx.items[@intCast(seq_id)] = 0;
        } else {
            @memset(self.rs_idx.items, 0);
        }
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_raw` (llama-kv-cache-dsv4.cpp:1675 @b29c606e2).
    pub fn getRaw(self: *const KvCacheDsv4) *KvCacheIswa {
        return self.kv_raw.?;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_csa` (llama-kv-cache-dsv4.cpp:1679 @b29c606e2).
    pub fn getCsa(self: *const KvCacheDsv4) *KvCache {
        return self.kv_csa.?;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_hca` (llama-kv-cache-dsv4.cpp:1683 @b29c606e2).
    pub fn getHca(self: *const KvCacheDsv4) *KvCache {
        return self.kv_hca.?;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_lid` (llama-kv-cache-dsv4.cpp:1687 @b29c606e2).
    pub fn getLid(self: *const KvCacheDsv4) *KvCache {
        return self.kv_lid.?;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_csa_state` (llama-kv-cache-dsv4.cpp:1691 @b29c606e2).
    pub fn getCsaState(self: *const KvCacheDsv4) *CompState {
        return self.csa_state.?;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_hca_state` (llama-kv-cache-dsv4.cpp:1695 @b29c606e2).
    pub fn getHcaState(self: *const KvCacheDsv4) *CompState {
        return self.hca_state.?;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_lid_state` (llama-kv-cache-dsv4.cpp:1699 @b29c606e2).
    pub fn getLidState(self: *const KvCacheDsv4) *CompState {
        return self.lid_state.?;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_n_rs_seq` (llama-kv-cache-dsv4.cpp:1703 @b29c606e2).
    pub fn getNRsSeq(self: *const KvCacheDsv4) u32 {
        return self.n_rs_seq;
    }

    /// Ports `llama_kv_cache_dsv4`'s `get_rs_idx` (llama-kv-cache-dsv4.cpp:1707 @b29c606e2).
    pub fn getRsIdx(self: *const KvCacheDsv4) []const u32 {
        return self.rs_idx.items;
    }

    /// Ports `llama_kv_cache_dsv4`'s `reset_rs_idx_for_ubatches`
    /// (llama-kv-cache-dsv4.cpp:1711 @b29c606e2): a ubatch that touches a
    /// sequence consumes its pending rollback.
    pub fn resetRsIdxForUbatches(self: *KvCacheDsv4, ubatches: []const Ubatch) void {
        if (self.n_rs_seq == 0) return;

        for (ubatches) |*ubatch| {
            for (0..ubatch.n_tokens) |i| {
                var s: usize = 0;
                while (s < @as(usize, @intCast(ubatch.n_seq_id[i]))) : (s += 1) {
                    const seq_id = ubatch.seq_id[i][s];
                    if (seq_id >= 0 and @as(u32, @bitCast(seq_id)) < self.n_seq_max) self.rs_idx.items[@intCast(seq_id)] = 0;
                }
            }
        }
    }

    /// Ports `llama_kv_cache_dsv4`'s `clear_compressed` (llama-kv-cache-dsv4.cpp:1728
    /// @b29c606e2): every compressed cache and state, or one sequence's.
    fn clearCompressed(self: *KvCacheDsv4, seq_id: i32, data: bool) void {
        if (seq_id < 0) {
            self.getCsa().clear(data);
            self.getHca().clear(data);
            self.getLid().clear(data);
        } else {
            impl.assert(@as(u32, @bitCast(seq_id)) < self.n_seq_max, file, 1734, "(uint32_t) seq_id < n_seq_max");

            for ([_]*KvCache{ self.getCsa(), self.getHca(), self.getLid() }) |kv| {
                _ = kv.seqRm(seq_id, -1, -1);

                if (data) {
                    // The C++'s TODO: do not clear the KV cache during
                    // `seq_rm`
                    // (https://github.com/ggml-org/llama.cpp/pull/26490#discussion_r3798143663).
                    var layer_ids = kv.getLayerIds();
                    defer layer_ids.deinit(allocator);
                    for (layer_ids.items) |il| clearTensorStream(kv.getKStorage(@intCast(il)), @bitCast(seq_id));
                }
            }
        }

        self.getCsaState().clear(seq_id, data);
        self.getHcaState().clear(seq_id, data);
        self.getLidState().clear(seq_id, data);

        if (seq_id >= 0) {
            self.rs_idx.items[@intCast(seq_id)] = 0;
        } else {
            @memset(self.rs_idx.items, 0);
        }
    }

    // -------------------------------------------------------------------------
    // The memory interface

    pub fn memoryOf(self: *KvCacheDsv4) memory.Memory {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: memory.Memory.VTable = .{
        .initBatch = struct {
            fn f(p: *anyopaque, b: *batch.BatchAllocr, n: u32, e: bool) memory.Context {
                return from(p).initBatch(b, n, e).contextOf();
            }
        }.f,
        .initFull = struct {
            fn f(p: *anyopaque) memory.Context {
                return from(p).initFull().contextOf();
            }
        }.f,
        .initUpdate = struct {
            fn f(p: *anyopaque, l: *anyopaque, o: bool) memory.Context {
                return from(p).initUpdate(l, o).contextOf();
            }
        }.f,
        .getCanShift = struct {
            fn f(p: *anyopaque) bool {
                return from(p).getCanShift();
            }
        }.f,
        .clear = struct {
            fn f(p: *anyopaque, d: bool) void {
                from(p).clear(d);
            }
        }.f,
        .seqRm = struct {
            fn f(p: *anyopaque, s: i32, p0: i32, p1: i32) bool {
                return from(p).seqRm(s, p0, p1);
            }
        }.f,
        .seqCp = struct {
            fn f(p: *anyopaque, a: i32, b: i32, p0: i32, p1: i32) void {
                from(p).seqCp(a, b, p0, p1);
            }
        }.f,
        .seqKeep = struct {
            fn f(p: *anyopaque, s: i32) void {
                from(p).seqKeep(s);
            }
        }.f,
        .seqAdd = struct {
            fn f(p: *anyopaque, s: i32, p0: i32, p1: i32, sh: i32) void {
                from(p).seqAdd(s, p0, p1, sh);
            }
        }.f,
        .seqDiv = struct {
            fn f(p: *anyopaque, s: i32, p0: i32, p1: i32, d: c_int) void {
                from(p).seqDiv(s, p0, p1, d);
            }
        }.f,
        .seqPosMin = struct {
            fn f(p: *anyopaque, s: i32) i32 {
                return from(p).seqPosMin(s);
            }
        }.f,
        .seqPosMax = struct {
            fn f(p: *anyopaque, s: i32) i32 {
                return from(p).seqPosMax(s);
            }
        }.f,
        .memoryBreakdown = struct {
            fn f(p: *anyopaque, out: *std.ArrayList(memory.BuftSize)) void {
                from(p).memoryBreakdown(out);
            }
        }.f,
        .stateWrite = struct {
            fn f(p: *anyopaque, w: io.Writer, s: i32, fl: u32, e: *Exception) Thrown!void {
                return from(p).stateWrite(w, s, fl, e);
            }
        }.f,
        .stateRead = struct {
            fn f(p: *anyopaque, r: io.Reader, s: i32, fl: u32, e: *Exception) Thrown!void {
                return from(p).stateRead(r, s, fl, e);
            }
        }.f,
        .destroy = struct {
            fn f(p: *anyopaque) void {
                from(p).destroy();
            }
        }.f,
    };

    fn from(p: *anyopaque) *KvCacheDsv4 {
        return @ptrCast(@alignCast(p));
    }
};

fn releaseAll(list: *std.ArrayList(Ubatch)) void {
    for (list.items) |*u| u.release();
    list.deinit(allocator);
}

fn freeSlotInfos(list: *std.ArrayList(SlotInfo)) void {
    for (list.items) |*si| si.deinit();
    list.deinit(allocator);
}

fn retainedCopy(list: []const Ubatch) std.ArrayList(Ubatch) {
    var out: std.ArrayList(Ubatch) = .empty;
    out.ensureTotalCapacity(allocator, list.len) catch oom();
    for (list) |u| {
        u.retain();
        out.appendAssumeCapacity(u);
    }
    return out;
}

// -----------------------------------------------------------------------------
// llama_kv_cache_dsv4_raw_context

/// Ports `dsv4_build_full_sinfo` (llama-kv-cache-dsv4.cpp:1767 @b29c606e2):
/// every stream, with a dummy index.
fn buildFullSinfo(kv: *const KvCache) SlotInfo {
    const n_stream = kv.getNStream();

    var sinfo: SlotInfo = .{};
    sinfo.s0 = 0;
    sinfo.s1 = n_stream - 1;
    sinfo.resize(n_stream);
    for (0..n_stream) |s| {
        sinfo.strm.items[s] = @intCast(s);
        sinfo.idxs.items[s].appendNTimes(allocator, 0, 1) catch oom();
    }
    return sinfo;
}

/// Ports `llama_kv_cache_dsv4_raw_context` (llama-kv-cache-dsv4.h:179
/// @b29c606e2): DSV4's raw attention uses only the SWA half of the raw
/// cache. The base half is kept for the iSWA cache's bookkeeping, but has
/// no DSV4 layers to expose.
pub const RawContext = struct {
    i_next: usize = 0,

    kv_swa: *KvCache,

    sinfos_write: std.ArrayList(SlotInfo) = .empty,
    sinfos_read: std.ArrayList(SlotInfo) = .empty,
    ubatches: std.ArrayList(Ubatch) = .empty,
    ubatches_write: std.ArrayList(Ubatch) = .empty,

    ctx_base_mem: ?*KvCacheContext = null,
    ctx_swa_mem: ?*KvCacheContext = null,

    n_kv: u32 = 0,

    status: memory.Status,

    /// Ports the full-cache constructor of `llama_kv_cache_dsv4_raw_context`
    /// (llama-kv-cache-dsv4.cpp:1782 @b29c606e2).
    pub fn createFull(kv: *KvCacheIswa) *RawContext {
        const self = allocator.create(RawContext) catch oom();
        self.* = .{ .kv_swa = kv.getSwa(), .status = .success };
        self.n_kv = self.kv_swa.getSize();
        self.sinfos_read.append(allocator, buildFullSinfo(self.kv_swa)) catch oom();
        self.sinfos_write.append(allocator, self.sinfos_read.items[0].clone()) catch oom();
        return self;
    }

    /// Ports the update constructor of `llama_kv_cache_dsv4_raw_context`
    /// (llama-kv-cache-dsv4.cpp:1792 @b29c606e2).
    pub fn createUpdate(kv: *KvCacheIswa, lctx: *anyopaque, optimize: bool) *RawContext {
        const self = allocator.create(RawContext) catch oom();
        const b = kv.getBase().initUpdate(lctx, optimize);
        const s = kv.getSwa().initUpdate(lctx, optimize);
        self.* = .{ .kv_swa = kv.getSwa(), .ctx_base_mem = b, .ctx_swa_mem = s, .status = memory.statusCombine(b.getStatus(), s.getStatus()) };
        self.n_kv = self.kv_swa.getSize();
        return self;
    }

    /// Ports the batch constructor of `llama_kv_cache_dsv4_raw_context`
    /// (llama-kv-cache-dsv4.cpp:1803 @b29c606e2): the base half places the
    /// raw writes through an ordinary KV cache context; the SWA half's are
    /// applied here, so its reads can differ from its writes.
    pub fn createBatch(
        kv: *KvCacheIswa,
        sinfos_base_write: std.ArrayList(SlotInfo),
        sinfos_swa_write: std.ArrayList(SlotInfo),
        sinfos_swa_read: std.ArrayList(SlotInfo),
        ubatches: std.ArrayList(Ubatch),
        ubatches_write: std.ArrayList(Ubatch),
    ) *RawContext {
        const self = allocator.create(RawContext) catch oom();
        self.* = .{
            .kv_swa = kv.getSwa(),
            .sinfos_write = sinfos_swa_write,
            .sinfos_read = sinfos_swa_read,
            .ubatches = ubatches,
            .ubatches_write = ubatches_write,
            .status = .success,
        };
        self.ctx_base_mem = KvCacheContext.createBatch(kv.getBase(), sinfos_base_write, retainedCopy(self.ubatches_write.items));
        self.n_kv = self.kv_swa.getSize();
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *RawContext) void {
        freeSlotInfos(&self.sinfos_write);
        freeSlotInfos(&self.sinfos_read);
        releaseAll(&self.ubatches);
        releaseAll(&self.ubatches_write);
        if (self.ctx_base_mem) |m| m.destroy();
        if (self.ctx_swa_mem) |m| m.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `next` (llama-kv-cache-dsv4.cpp:1822 @b29c606e2).
    pub fn next(self: *RawContext) bool {
        if (self.ubatches.items.len == 0) return true;

        if (self.ctx_base_mem) |m| _ = m.next();

        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `apply` (llama-kv-cache-dsv4.cpp:1838 @b29c606e2).
    pub fn apply(self: *RawContext) bool {
        var res = true;

        if (self.ctx_base_mem) |m| {
            const r = m.apply();
            res = res and r;
        }
        if (self.ctx_swa_mem) |m| {
            const r = m.apply();
            res = res and r;
        }
        if (self.ubatches_write.items.len != 0) {
            self.kv_swa.applyUbatch(&self.sinfos_write.items[self.i_next], &self.ubatches_write.items[self.i_next]);
            self.n_kv = self.kv_swa.getNKv(&self.sinfos_read.items[self.i_next]);
        }

        return res;
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `get_status` (llama-kv-cache-dsv4.cpp:1855 @b29c606e2).
    pub fn getStatus(self: *const RawContext) memory.Status {
        return self.status;
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `get_ubatch` (llama-kv-cache-dsv4.cpp:1859 @b29c606e2).
    pub fn getUbatch(self: *const RawContext) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `get_n_kv` (llama-kv-cache-dsv4.cpp:1865 @b29c606e2).
    pub fn getNKv(self: *const RawContext) u32 {
        return self.n_kv;
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `get_n_write` (llama-kv-cache-dsv4.cpp:1869
    /// @b29c606e2): the current ubatch's raw writes.
    pub fn getNWrite(self: *const RawContext) u32 {
        if (self.ubatches_write.items.len == 0) return 0;
        return self.ubatches_write.items[self.i_next].n_tokens;
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `get_k` (llama-kv-cache-dsv4.cpp:1877 @b29c606e2).
    pub fn getK(self: *const RawContext, ctx: *c.ggml_context, il: i32) *Tensor {
        return self.kv_swa.getK(ctx, il, self.n_kv, &self.sinfos_read.items[self.i_next]);
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `cpy_k` (llama-kv-cache-dsv4.cpp:1881
    /// @b29c606e2): `k_idxs` may be expanded to one block per stream while
    /// `k_cur` is the token block alone; each copy then gets a zero
    /// dependency on the others, so every write executes.
    pub fn cpyK(self: *const RawContext, ctx: *c.ggml_context, k_cur: *Tensor, k_idxs: *Tensor, il: i32) *Tensor {
        const sinfo = &self.sinfos_write.items[self.i_next];

        if (k_cur.ne[2] == k_idxs.ne[0]) return self.kv_swa.cpyK(ctx, k_cur, k_idxs, il, sinfo);

        const n_fanout: i64 = @intCast(sinfo.size() * sinfo.nStream());

        impl.assert(sinfo.nStream() > 1, file, 1892, "sinfo.n_stream() > 1");
        impl.assert(k_cur.ne[2] == @as(i64, @intCast(sinfo.size())), file, 1893, "k_cur->ne[2] == (int64_t) sinfo.size()");
        impl.assert(k_idxs.ne[0] == n_fanout, file, 1894, "k_idxs->ne[0] == n_fanout");

        var res: ?*Tensor = null;
        for (0..sinfo.nStream()) |s| {
            const k_idxs_s: *Tensor = c.ggml_view_1d(ctx, k_idxs, @intCast(sinfo.size()), s * sinfo.size() * c.ggml_element_size(k_idxs));
            const cur = self.kv_swa.cpyK(ctx, k_cur, k_idxs_s, il, sinfo);
            if (res) |r| {
                res = c.ggml_add(ctx, r, c.ggml_sub(ctx, cur, cur));
            } else {
                res = cur;
            }
        }
        return res.?;
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `build_input_k_idxs`
    /// (llama-kv-cache-dsv4.cpp:1910 @b29c606e2): one index per raw write.
    pub fn buildInputKIdxs(self: *const RawContext, ctx: *c.ggml_context, ubatch: *const Ubatch) *Tensor {
        const n_tokens: u32 = if (self.ubatches_write.items.len == 0) ubatch.n_tokens else self.ubatches_write.items[self.i_next].n_tokens;

        const k_idxs: *Tensor = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I64, n_tokens);
        c.ggml_set_input(k_idxs);
        return k_idxs;
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `build_input_k_rot` (llama-kv-cache-dsv4.cpp:1919 @b29c606e2).
    pub fn buildInputKRot(self: *const RawContext, ctx: *c.ggml_context) ?*Tensor {
        return self.kv_swa.buildInputKRot(ctx);
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `set_input_k_idxs` (llama-kv-cache-dsv4.cpp:1923 @b29c606e2).
    pub fn setInputKIdxs(self: *const RawContext, dst: *Tensor) void {
        self.kv_swa.setInputKIdxs(dst, &self.ubatches_write.items[self.i_next], &self.sinfos_write.items[self.i_next]);
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `set_input_kq_mask` (llama-kv-cache-dsv4.cpp:1927 @b29c606e2).
    pub fn setInputKqMask(self: *const RawContext, dst: *Tensor, ubatch: *const Ubatch, causal_attn: bool) void {
        self.kv_swa.setInputKqMask(dst, ubatch, causal_attn);
    }

    /// Ports `llama_kv_cache_dsv4_raw_context`'s `set_input_k_rot` (llama-kv-cache-dsv4.cpp:1931 @b29c606e2).
    pub fn setInputKRot(self: *const RawContext, dst: *Tensor) void {
        self.kv_swa.setInputKRot(dst);
    }
};

// -----------------------------------------------------------------------------
// llama_kv_cache_dsv4_comp_context

/// Ports `llama_kv_cache_dsv4_comp_context` (llama-kv-cache-dsv4.h:237
/// @b29c606e2): compressed rows are graph outputs, not token writes, so
/// this exposes a compressed cache's K without `apply` semantics.
pub const CompContext = struct {
    kv: *KvCache,
    i_cur: usize = 0,
    sinfos: std.ArrayList(SlotInfo) = .empty,
    ubatches: std.ArrayList(Ubatch) = .empty,
    n_kv: u32,

    /// Ports the full-cache constructor of `llama_kv_cache_dsv4_comp_context`
    /// (llama-kv-cache-dsv4.cpp:1939 @b29c606e2).
    pub fn createFull(kv: *KvCache) *CompContext {
        const self = allocator.create(CompContext) catch oom();
        self.* = .{ .kv = kv, .n_kv = kv.getSize() };
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

    /// Ports the batch constructor of `llama_kv_cache_dsv4_comp_context`
    /// (llama-kv-cache-dsv4.cpp:1952 @b29c606e2): takes both lists.
    pub fn createBatch(kv: *KvCache, sinfos: std.ArrayList(SlotInfo), ubatches: std.ArrayList(Ubatch)) *CompContext {
        const self = allocator.create(CompContext) catch oom();
        self.* = .{ .kv = kv, .sinfos = sinfos, .ubatches = ubatches, .n_kv = kv.getSize() };
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *CompContext) void {
        freeSlotInfos(&self.sinfos);
        releaseAll(&self.ubatches);
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsv4_comp_context`'s `next` (llama-kv-cache-dsv4.cpp:1962 @b29c606e2).
    pub fn next(self: *CompContext) bool {
        if (self.ubatches.items.len == 0) return true;
        self.i_cur += 1;
        return self.i_cur < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_dsv4_comp_context`'s `get_n_kv` (llama-kv-cache-dsv4.cpp:1974 @b29c606e2).
    pub fn getNKv(self: *const CompContext) u32 {
        return self.n_kv;
    }

    /// Ports `llama_kv_cache_dsv4_comp_context`'s `get_k` (llama-kv-cache-dsv4.cpp:1978 @b29c606e2).
    pub fn getK(self: *const CompContext, ctx: *c.ggml_context, il: i32) *Tensor {
        return self.kv.getK(ctx, il, self.n_kv, &self.sinfos.items[self.i_cur]);
    }

    /// Ports `llama_kv_cache_dsv4_comp_context`'s `cpy_k` (llama-kv-cache-dsv4.cpp:1982 @b29c606e2).
    pub fn cpyK(self: *const CompContext, ctx: *c.ggml_context, k_cur: *Tensor, k_idxs: *Tensor, il: i32) *Tensor {
        return self.kv.cpyK(ctx, k_cur, k_idxs, il, &self.sinfos.items[self.i_cur]);
    }

    /// Ports `llama_kv_cache_dsv4_comp_context`'s `build_input_k_rot` (llama-kv-cache-dsv4.cpp:1986 @b29c606e2).
    pub fn buildInputKRot(self: *const CompContext, ctx: *c.ggml_context) ?*Tensor {
        return self.kv.buildInputKRot(ctx);
    }

    /// Ports `llama_kv_cache_dsv4_comp_context`'s `set_input_k_rot` (llama-kv-cache-dsv4.cpp:1990 @b29c606e2).
    pub fn setInputKRot(self: *const CompContext, dst: *Tensor) void {
        self.kv.setInputKRot(dst);
    }
};

// -----------------------------------------------------------------------------
// llama_kv_cache_dsv4_context

/// The empty plan `get_csa_plan` and its siblings return without ubatches.
const empty_plan: CompPlan = .{};

/// Ports `llama_kv_cache_dsv4_context` (llama-kv-cache-dsv4.h:268 @b29c606e2):
/// the DSV4 memory's view for one batch -- the raw context, a view of each
/// compressed cache, the compressor states, and a plan per ubatch for each.
pub const Dsv4Context = struct {
    i_next: usize = 0,

    ubatches: std.ArrayList(Ubatch) = .empty,

    plans_csa: std.ArrayList(CompPlan) = .empty,
    plans_hca: std.ArrayList(CompPlan) = .empty,
    plans_lid: std.ArrayList(CompPlan) = .empty,

    ctx_raw: ?*RawContext = null,
    ctx_csa_mem: ?*KvCacheContext = null,
    ctx_hca_mem: ?*KvCacheContext = null,
    ctx_lid_mem: ?*KvCacheContext = null,

    ctx_csa: ?*CompContext = null,
    ctx_hca: ?*CompContext = null,
    ctx_lid: ?*CompContext = null,

    csa_state: ?*CompState = null,
    hca_state: ?*CompState = null,
    lid_state: ?*CompState = null,

    sc_info_csa: StreamCopyInfo = .{},
    sc_info_hca: StreamCopyInfo = .{},
    sc_info_lid: StreamCopyInfo = .{},

    /// Whether the plans are the reserve graph's worst cases, rebuilt per
    /// ubatch -- the C++'s `mutable` reserve plans.
    reserve_plans: bool = false,
    reserve_plan_csa: CompPlan = .{},
    reserve_plan_hca: CompPlan = .{},
    reserve_plan_lid: CompPlan = .{},

    status: memory.Status,

    /// Ports the status constructor of `llama_kv_cache_dsv4_context`
    /// (llama-kv-cache-dsv4.cpp:1998 @b29c606e2).
    pub fn createStatus(status: memory.Status) *Dsv4Context {
        const self = allocator.create(Dsv4Context) catch oom();
        self.* = .{ .status = status };
        return self;
    }

    /// Ports the full-cache constructor of `llama_kv_cache_dsv4_context`
    /// (llama-kv-cache-dsv4.cpp:2000 @b29c606e2): the reserve graph's view,
    /// with worst-case plans.
    pub fn createFull(kv: *KvCacheDsv4) *Dsv4Context {
        const self = allocator.create(Dsv4Context) catch oom();
        const raw = RawContext.createFull(kv.getRaw());
        const csa = kv.getCsa().initFull();
        const hca = kv.getHca().initFull();
        const lid = kv.getLid().initFull();
        self.* = .{
            .ctx_raw = raw,
            .ctx_csa_mem = csa,
            .ctx_hca_mem = hca,
            .ctx_lid_mem = lid,
            .ctx_csa = CompContext.createFull(kv.getCsa()),
            .ctx_hca = CompContext.createFull(kv.getHca()),
            .ctx_lid = CompContext.createFull(kv.getLid()),
            .csa_state = kv.getCsaState(),
            .hca_state = kv.getHcaState(),
            .lid_state = kv.getLidState(),
            .reserve_plans = true,
            .status = memory.statusCombine(memory.statusCombine(raw.getStatus(), csa.getStatus()), memory.statusCombine(hca.getStatus(), lid.getStatus())),
        };
        return self;
    }

    /// Ports the update constructor of `llama_kv_cache_dsv4_context`
    /// (llama-kv-cache-dsv4.cpp:2018 @b29c606e2): an update when any of the
    /// caches has one, or any compressor state has queued copies.
    pub fn createUpdate(kv: *KvCacheDsv4, lctx: *anyopaque, optimize: bool, sc_info_csa: StreamCopyInfo, sc_info_hca: StreamCopyInfo, sc_info_lid: StreamCopyInfo) *Dsv4Context {
        const self = allocator.create(Dsv4Context) catch oom();
        const raw = RawContext.createUpdate(kv.getRaw(), lctx, optimize);
        const csa = kv.getCsa().initUpdate(lctx, optimize);
        const hca = kv.getHca().initUpdate(lctx, optimize);
        const lid = kv.getLid().initUpdate(lctx, optimize);
        self.* = .{
            .ctx_raw = raw,
            .ctx_csa_mem = csa,
            .ctx_hca_mem = hca,
            .ctx_lid_mem = lid,
            .csa_state = kv.getCsaState(),
            .hca_state = kv.getHcaState(),
            .lid_state = kv.getLidState(),
            .sc_info_csa = sc_info_csa,
            .sc_info_hca = sc_info_hca,
            .sc_info_lid = sc_info_lid,
            .status = undefined,
        };
        const copies: memory.Status = if (self.sc_info_csa.empty() and self.sc_info_hca.empty() and self.sc_info_lid.empty()) .no_update else .success;
        self.status = memory.statusCombine(
            memory.statusCombine(memory.statusCombine(raw.getStatus(), csa.getStatus()), memory.statusCombine(hca.getStatus(), lid.getStatus())),
            copies,
        );
        return self;
    }

    /// Ports the batch constructor of `llama_kv_cache_dsv4_context`
    /// (llama-kv-cache-dsv4.cpp:2043 @b29c606e2): plans every ubatch for
    /// each compressor, then consumes the rollbacks the ubatches touch.
    /// Takes every list.
    pub fn createBatch(
        kv: *KvCacheDsv4,
        sinfos_raw_base_write: std.ArrayList(SlotInfo),
        sinfos_raw_swa_write: std.ArrayList(SlotInfo),
        sinfos_raw_swa_read: std.ArrayList(SlotInfo),
        ubatches: std.ArrayList(Ubatch),
        ubatches_raw: std.ArrayList(Ubatch),
    ) *Dsv4Context {
        const self = allocator.create(Dsv4Context) catch oom();
        self.* = .{ .ubatches = ubatches, .status = .success };
        const ub = self.ubatches.items;

        self.plans_csa = buildCompPlans(ub, csa_ratio, true, kv.getCsaState().getStateSize(), kv.getCsa().getSize(), kv.getCsaState().getNStream(), kv.getNRsSeq(), kv.getRsIdx());
        self.plans_hca = buildCompPlans(ub, hca_ratio, false, kv.getHcaState().getStateSize(), kv.getHca().getSize(), kv.getHcaState().getNStream(), kv.getNRsSeq(), kv.getRsIdx());
        self.plans_lid = buildCompPlans(ub, csa_ratio, true, kv.getLidState().getStateSize(), kv.getLid().getSize(), kv.getLidState().getNStream(), kv.getNRsSeq(), kv.getRsIdx());

        self.ctx_raw = RawContext.createBatch(kv.getRaw(), sinfos_raw_base_write, sinfos_raw_swa_write, sinfos_raw_swa_read, retainedCopy(ub), ubatches_raw);

        self.ctx_csa = CompContext.createBatch(kv.getCsa(), buildCompSinfos(ub, kv.getCsa().getNStream()), retainedCopy(ub));
        self.ctx_hca = CompContext.createBatch(kv.getHca(), buildCompSinfos(ub, kv.getHca().getNStream()), retainedCopy(ub));
        self.ctx_lid = CompContext.createBatch(kv.getLid(), buildCompSinfos(ub, kv.getLid().getNStream()), retainedCopy(ub));

        self.csa_state = kv.getCsaState();
        self.hca_state = kv.getHcaState();
        self.lid_state = kv.getLidState();

        self.status = self.ctx_raw.?.getStatus();

        kv.resetRsIdxForUbatches(ub);
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *Dsv4Context) void {
        releaseAll(&self.ubatches);
        freePlans(&self.plans_csa);
        freePlans(&self.plans_hca);
        freePlans(&self.plans_lid);
        if (self.ctx_raw) |x| x.destroy();
        if (self.ctx_csa_mem) |x| x.destroy();
        if (self.ctx_hca_mem) |x| x.destroy();
        if (self.ctx_lid_mem) |x| x.destroy();
        if (self.ctx_csa) |x| x.destroy();
        if (self.ctx_hca) |x| x.destroy();
        if (self.ctx_lid) |x| x.destroy();
        self.sc_info_csa.deinit();
        self.sc_info_hca.deinit();
        self.sc_info_lid.deinit();
        self.reserve_plan_csa.deinit();
        self.reserve_plan_hca.deinit();
        self.reserve_plan_lid.deinit();
        allocator.destroy(self);
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `next` (llama-kv-cache-dsv4.cpp:2091 @b29c606e2).
    pub fn next(self: *Dsv4Context) bool {
        std.debug.assert(self.status == .success);

        _ = self.ctx_raw.?.next();
        _ = self.ctx_csa.?.next();
        _ = self.ctx_hca.?.next();
        _ = self.ctx_lid.?.next();

        self.i_next += 1;
        return self.i_next < self.ubatches.items.len;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `apply` (llama-kv-cache-dsv4.cpp:2106
    /// @b29c606e2): without ubatches, an update, which also applies the
    /// compressor states' queued stream copies.
    pub fn apply(self: *Dsv4Context) bool {
        std.debug.assert(!memory.statusIsFail(self.status));

        var res = true;

        const r_raw = self.ctx_raw.?.apply();
        res = res and r_raw;

        if (self.ctx_csa_mem) |csa| {
            const r_csa = csa.apply();
            res = res and r_csa;
            const r_hca = self.ctx_hca_mem.?.apply();
            res = res and r_hca;
            const r_lid = self.ctx_lid_mem.?.apply();
            res = res and r_lid;
        }

        if (self.ubatches.items.len == 0) {
            self.csa_state.?.applyCopies(&self.sc_info_csa);
            self.hca_state.?.applyCopies(&self.sc_info_hca);
            self.lid_state.?.applyCopies(&self.sc_info_lid);
        }

        return res;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_status` (llama-kv-cache-dsv4.cpp:2128 @b29c606e2).
    pub fn getStatus(self: *const Dsv4Context) memory.Status {
        return self.status;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_ubatch` (llama-kv-cache-dsv4.cpp:2132 @b29c606e2).
    pub fn getUbatch(self: *const Dsv4Context) *const Ubatch {
        std.debug.assert(self.status == .success);
        return &self.ubatches.items[self.i_next];
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_raw` (llama-kv-cache-dsv4.cpp:2138 @b29c606e2).
    pub fn getRaw(self: *const Dsv4Context) *const RawContext {
        std.debug.assert(self.status == .success);
        return self.ctx_raw.?;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_csa` (llama-kv-cache-dsv4.cpp:2144 @b29c606e2).
    pub fn getCsa(self: *const Dsv4Context) *const CompContext {
        std.debug.assert(self.status == .success);
        return self.ctx_csa.?;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_hca` (llama-kv-cache-dsv4.cpp:2150 @b29c606e2).
    pub fn getHca(self: *const Dsv4Context) *const CompContext {
        std.debug.assert(self.status == .success);
        return self.ctx_hca.?;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_lid` (llama-kv-cache-dsv4.cpp:2156 @b29c606e2).
    pub fn getLid(self: *const Dsv4Context) *const CompContext {
        std.debug.assert(self.status == .success);
        return self.ctx_lid.?;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_csa_state` (llama-kv-cache-dsv4.cpp:2162 @b29c606e2).
    pub fn getCsaState(self: *const Dsv4Context) *const CompState {
        std.debug.assert(self.status == .success);
        return self.csa_state.?;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_hca_state` (llama-kv-cache-dsv4.cpp:2168 @b29c606e2).
    pub fn getHcaState(self: *const Dsv4Context) *const CompState {
        std.debug.assert(self.status == .success);
        return self.hca_state.?;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_lid_state` (llama-kv-cache-dsv4.cpp:2174 @b29c606e2).
    pub fn getLidState(self: *const Dsv4Context) *const CompState {
        std.debug.assert(self.status == .success);
        return self.lid_state.?;
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_csa_plan` (llama-kv-cache-dsv4.cpp:2180 @b29c606e2).
    pub fn getCsaPlan(self: *const Dsv4Context) *const CompPlan {
        std.debug.assert(self.status == .success);
        if (self.plans_csa.items.len == 0) return &empty_plan;
        return &self.plans_csa.items[self.i_next];
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_hca_plan` (llama-kv-cache-dsv4.cpp:2191 @b29c606e2).
    pub fn getHcaPlan(self: *const Dsv4Context) *const CompPlan {
        std.debug.assert(self.status == .success);
        if (self.plans_hca.items.len == 0) return &empty_plan;
        return &self.plans_hca.items[self.i_next];
    }

    /// Ports `llama_kv_cache_dsv4_context`'s `get_lid_plan` (llama-kv-cache-dsv4.cpp:2202 @b29c606e2).
    pub fn getLidPlan(self: *const Dsv4Context) *const CompPlan {
        std.debug.assert(self.status == .success);
        if (self.plans_lid.items.len == 0) return &empty_plan;
        return &self.plans_lid.items[self.i_next];
    }

    /// Ports the overload of `llama_kv_cache_dsv4_context`'s `get_csa_plan`
    /// that takes a ubatch (llama-kv-cache-dsv4.cpp:2213 @b29c606e2): the
    /// reserve graph's worst case, rebuilt for `ubatch`, else the current
    /// plan. The returned plan is valid until the next call.
    pub fn getCsaPlanFor(self: *const Dsv4Context, ubatch: *const Ubatch) *const CompPlan {
        std.debug.assert(self.status == .success);
        if (!self.reserve_plans) return self.getCsaPlan();

        const m = @constCast(self);
        m.reserve_plan_csa.deinit();
        m.reserve_plan_csa = buildReserveCompPlan(ubatch, csa_ratio, true, self.csa_state.?.getStateSize(), self.getCsa().getNKv(), self.csa_state.?.getNStream(), self.csa_state.?.getNRsSeq());
        return &m.reserve_plan_csa;
    }

    /// Ports the overload of `llama_kv_cache_dsv4_context`'s `get_hca_plan`
    /// that takes a ubatch (llama-kv-cache-dsv4.cpp:2227 @b29c606e2).
    pub fn getHcaPlanFor(self: *const Dsv4Context, ubatch: *const Ubatch) *const CompPlan {
        std.debug.assert(self.status == .success);
        if (!self.reserve_plans) return self.getHcaPlan();

        const m = @constCast(self);
        m.reserve_plan_hca.deinit();
        m.reserve_plan_hca = buildReserveCompPlan(ubatch, hca_ratio, false, self.hca_state.?.getStateSize(), self.getHca().getNKv(), self.hca_state.?.getNStream(), self.hca_state.?.getNRsSeq());
        return &m.reserve_plan_hca;
    }

    /// Ports the overload of `llama_kv_cache_dsv4_context`'s `get_lid_plan`
    /// that takes a ubatch (llama-kv-cache-dsv4.cpp:2241 @b29c606e2).
    pub fn getLidPlanFor(self: *const Dsv4Context, ubatch: *const Ubatch) *const CompPlan {
        std.debug.assert(self.status == .success);
        if (!self.reserve_plans) return self.getLidPlan();

        const m = @constCast(self);
        m.reserve_plan_lid.deinit();
        m.reserve_plan_lid = buildReserveCompPlan(ubatch, csa_ratio, true, self.lid_state.?.getStateSize(), self.getLid().getNKv(), self.lid_state.?.getNStream(), self.lid_state.?.getNRsSeq());
        return &m.reserve_plan_lid;
    }

    pub fn contextOf(self: *Dsv4Context) memory.Context {
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

    fn cast(p: *anyopaque) *Dsv4Context {
        return @ptrCast(@alignCast(p));
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(CompPlan);
    std.testing.refAllDecls(CompState);
    std.testing.refAllDecls(KvCacheDsv4);
    std.testing.refAllDecls(RawContext);
    std.testing.refAllDecls(CompContext);
    std.testing.refAllDecls(Dsv4Context);
}

/// A single-sequence ubatch over `pos`, all in sequence `seq`, backed by
/// the caller's arrays.
const TestUbatch = struct {
    pos: [16]i32 = undefined,
    n_seq_id: [16]i32 = @splat(1),
    seq_data: [16]i32 = undefined,
    seq_ptr: [16][*c]i32 = undefined,
    seq_unq: [1]i32 = undefined,

    fn ubatch(self: *TestUbatch, seq: i32, pos: []const i32) Ubatch {
        for (pos, 0..) |p, i| {
            self.pos[i] = p;
            self.seq_data[i] = seq;
            self.seq_ptr[i] = &self.seq_data[i];
        }
        self.seq_unq[0] = seq;
        return .{
            .b_equal_seqs = 1,
            .n_tokens = @intCast(pos.len),
            .n_seq_tokens = @intCast(pos.len),
            .n_seqs = 1,
            .n_seqs_unq = 1,
            .n_pos = 1,
            .pos = &self.pos,
            .n_seq_id = &self.n_seq_id,
            .seq_id = &self.seq_ptr,
            .seq_id_unq = &self.seq_unq,
        };
    }
};

test "the compressed caches have a row per block, at least one" {
    try std.testing.expectEqual(@as(u32, 1), compSize(0, csa_ratio));
    try std.testing.expectEqual(@as(u32, 1), compSize(3, csa_ratio));
    try std.testing.expectEqual(@as(u32, 2), compSize(5, csa_ratio));
    try std.testing.expectEqual(@as(u32, 1), compSize(128, hca_ratio));
    try std.testing.expectEqual(@as(u32, 2), compSize(129, hca_ratio));
}

test "a sequence's used compressed rows are its completed blocks" {
    try std.testing.expectEqual(@as(u32, 0), stateNUsedKRows(-1, csa_ratio, 64));
    try std.testing.expectEqual(@as(u32, 0), stateNUsedKRows(2, csa_ratio, 64));
    try std.testing.expectEqual(@as(u32, 1), stateNUsedKRows(3, csa_ratio, 64));
    try std.testing.expectEqual(@as(u32, 64), stateNUsedKRows(1000, csa_ratio, 64));
}

test "a prompt's CSA plan commits each completed block from its two windows" {
    var tu: TestUbatch = .{};
    const ub = tu.ubatch(0, &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    var plan = buildCompPlan(&ub, csa_ratio, true, 2 * csa_ratio, 256, 1, 0, &.{0});
    defer plan.deinit();

    // Two blocks complete, at rows 0 and 1, positioned at their first token.
    try std.testing.expectEqualSlices(i64, &.{ 0, 1 }, plan.state_write_idxs.items);
    try std.testing.expectEqualSlices(i32, &.{ 0, 4 }, plan.state_write_pos.items);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 0, 1, 2, 3 }, plan.state_pos.items);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 1, 1, 2 }, plan.n_visible.items);

    // All previous halves, then all current halves. The first block's
    // previous window is the synthetic zero row past the scratch rows; the
    // rest are the ubatch's own rows, after the 8 state rows.
    const zero: i32 = 8 + 8;
    try std.testing.expectEqualSlices(i32, &.{ zero, zero, zero, zero, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15 }, plan.state_read_idxs.items);

    // The graph is at least 256 rows wide, so its topology never changes.
    try std.testing.expectEqual(@as(i64, 256), plan.n_kv);

    // Each ring row is persisted once, from the latest token that maps to it.
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, plan.state_persist_dst_idxs.items);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, plan.state_persist_src_idxs.items);
}

test "a CSA decode step between boundaries writes a masked dummy block" {
    var tu: TestUbatch = .{};
    const ub = tu.ubatch(0, &.{9});
    var plan = buildCompPlan(&ub, csa_ratio, true, 2 * csa_ratio, 256, 1, 0, &.{0});
    defer plan.deinit();

    // No block completes, so the reserve's block count is met by a write to
    // the last cache row, reading the token's own scratch row.
    try std.testing.expectEqualSlices(i64, &.{255}, plan.state_write_idxs.items);
    try std.testing.expectEqualSlices(i32, &.{0}, plan.state_write_pos.items);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 8, 8, 8, 8, 8, 8, 8, 8 }, plan.state_read_idxs.items);
    try std.testing.expectEqualSlices(i32, &.{1}, plan.state_persist_dst_idxs.items);
}

test "an HCA step that completes no block still keeps its write" {
    var tu: TestUbatch = .{};
    const ub = tu.ubatch(1, &.{ 5, 6 });
    var plan = buildCompPlan(&ub, hca_ratio, false, hca_ratio, 256, 2, 0, &.{ 0, 0 });
    defer plan.deinit();

    // Sequence 1's stream starts a cache's length in.
    try std.testing.expectEqualSlices(i64, &.{256 + 255}, plan.state_write_idxs.items);
    try std.testing.expectEqual(@as(usize, hca_ratio), plan.state_read_idxs.items.len);
    try std.testing.expectEqualSlices(i32, &.{ 128 + 5, 128 + 6 }, plan.state_persist_dst_idxs.items);
    try std.testing.expectEqual(@as(i64, 1), plan.n_stream);
}

test "a pending rollback restores from its snapshot plane once" {
    var tu: TestUbatch = .{};
    const ub = tu.ubatch(0, &.{ 4, 5 });
    const ubs = [_]Ubatch{ ub, ub };
    var plans = buildCompPlans(&ubs, csa_ratio, true, 2 * csa_ratio, 256, 1, 2, &.{2});
    defer freePlans(&plans);

    const state_rows: i32 = 8;
    // The first ubatch restores from plane 2; the second, with the
    // rollback consumed, copies the current plane onto itself.
    try std.testing.expectEqual(2 * state_rows, plans.items[0].state_restore_src_idxs.items[0]);
    try std.testing.expectEqual(@as(i32, 0), plans.items[1].state_restore_src_idxs.items[0]);
    try std.testing.expectEqual(@as(usize, 2 * 8), plans.items[0].state_snapshot_src_idxs.items.len);
}

test "a reserve plan is sized for the worst sequence split" {
    var tu: TestUbatch = .{};
    const ub = tu.ubatch(0, &.{ 0, 1, 2, 3, 4, 5 });
    var plan = buildReserveCompPlan(&ub, csa_ratio, true, 2 * csa_ratio, 512, 1, 0);
    defer plan.deinit();

    // floor(6/4) + 1 blocks, two windows of four rows each.
    try std.testing.expectEqual(@as(usize, 2), plan.state_write_idxs.items.len);
    try std.testing.expectEqual(@as(usize, 2 * 4 * 2), plan.state_read_idxs.items.len);
    try std.testing.expectEqual(@as(usize, 6), plan.state_persist_src_idxs.items.len);
    try std.testing.expectEqual(@as(i64, 512), plan.n_kv);
}

test "plan positions print as a list" {
    const s = planPositions(&.{ 1, -2, 3 });
    defer allocator.free(s);
    try std.testing.expectEqualStrings("[1, -2, 3]", s);
}
