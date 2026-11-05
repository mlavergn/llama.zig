//! `llama-imatrix`: collect the importance matrix a quantizer weighs its
//! rounding by, merge saved ones, or print statistics about one.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/imatrix/imatrix.cpp` at v0.7.0 (`620d2225c`).
//! Each declaration names the C++ it replaces and the line it began at.
//!
//! # Orders that come from libc++
//!
//! `IMatrixCollector::m_stats` is a `std::unordered_map`, and both writers
//! and `--show-statistics` walk it: the partial-data warnings come out in
//! its order, and the statistics are pushed in it before an unstable
//! `std::sort`. It is `libcxx.StringMap`, which reproduces that order, and
//! the sort is `libcxx.sort`.
//!
//! # The perplexity's threads
//!
//! `process_logits` hands tokens to `hardware_concurrency()` threads off a
//! shared counter, and each adds its partial sums into the total as it
//! finishes -- so upstream's last bits depend on scheduling, below the four
//! decimals it prints. The port keeps the threads and the counter, and adds
//! the partial sums in thread order.
//!
//! # Contraction
//!
//! The statistics are double-precision accumulations written into the GGUF
//! (`<name>.stats`), so `tools-diff`'s file hashes see their last bits: each
//! `acc += a * b` the C++ writes in one expression is a `@mulAdd`, the
//! outer multiply fused with the add, as clang contracts it.

const std = @import("std");
const tc = @import("tools_common");
const libcxx = @import("llama").libcxx;
const c = tc.c;
const io = tc.io;
const log = tc.log;
const arg = tc.arg;
const Params = tc.params.Params;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

extern fn expf(x: f32) f32;
extern fn log2f(x: f32) f32;
extern fn sqrtf(x: f32) f32;
extern fn exp(x: f64) f64;
const libm = struct {
    extern "c" fn log(x: f64) f64;
};
const logD = libm.log;
extern fn sqrt(x: f64) f64;

const gpa = std.heap.c_allocator;

const int_max: c_int = std.math.maxInt(c_int);
const fnan = std.math.nan(f32);

/// `GGML_ASSERT`, which upstream keeps in every build mode.
fn ggmlAssert(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort("imatrix.cpp", line, "GGML_ASSERT(%s) failed", expr);
}

/// `std::clamp(v, lo, hi)`: `v < lo ? lo : hi < v ? hi : v`, a NaN kept.
fn cClamp(v: f32, lo: f32, hi: f32) f32 {
    return if (v < lo) lo else if (hi < v) hi else v;
}

/// `std::stoi` inside `try { ... } catch (...)`: null on either throw.
fn stoiOr(s: []const u8) ?c_int {
    var buf: [64]u8 = undefined;
    if (s.len >= buf.len) {
        const z = gpa.dupeSentinel(u8, s, 0) catch return null;
        defer gpa.free(z);
        return switch (tc.cxx.stoi(z)) {
            .ok => |v| v,
            else => null,
        };
    }
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    const z: [:0]const u8 = buf[0..s.len :0];
    return switch (tc.cxx.stoi(z)) {
        .ok => |v| v,
        else => null,
    };
}

/// Ports `print_usage` (imatrix.cpp:26 @620d2225c).
fn printUsage(argv: []const [:0]const u8) void {
    log.out("\nexample usage:\n", .{});
    log.out("\n    %s \\\n" ++
        "       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \\\n" ++
        "       [--process-output] [--nextn] [--model-draft draft.gguf] [--chunk 123] [--save-frequency 0] \\\n" ++
        "       [--output-frequency 10] [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] \\\n" ++
        "       [--parse-special] [--show-statistics] [...]\n", .{argv[0].ptr});
    log.out("\n", .{});
}

/// Ports `Stats` (imatrix.cpp:36 @620d2225c).
const Stats = struct {
    activations: std.ArrayList(f32) = .empty,
    values: std.ArrayList(f32) = .empty,
    counts: std.ArrayList(i64) = .empty,
};

/// Ports `tensor_statistics` (imatrix.cpp:42 @620d2225c).
const TensorStatistics = struct {
    tensor: []const u8,
    legacy: bool = true,
    sum: f64 = 0.0,
    mean: f32 = 0.0,
    elements: i64 = 0,
    std_deviation: f32 = 0.0,
    skewness: f32 = 0.0,
    kurtosis: f32 = 0.0,
    gain: f32 = fnan,
    entropy: f32 = 0.0,
    l2_dist: f32 = fnan,
    cossim: f32 = fnan,
    pearson: f32 = fnan,
    covariance: f32 = fnan,
    cov_sum: f64 = 0.0,
    var_c_sum: f64 = 0.0,
    var_p_sum: f64 = 0.0,
    dot_prod: f64 = 0.0,
    norm1_sq: f64 = 0.0,
    norm2_sq: f64 = 0.0,
    l2_dist_sq: f64 = 0.0,
    sum_prev: f64 = 0.0,
    elements_prev: i64 = 0,
    n_features: i64 = 0,
};

/// What `IMatrixCollector` keeps of `common_params`: `set_params` copies
/// the whole struct, so later changes to the caller's do not reach it.
const CollectorParams = struct {
    n_ctx: i32 = 0,
    n_parallel: i32 = 1,
    process_output: bool = false,
    load_mtp: bool = false,
    n_out_freq: i32 = 10,
    n_save_freq: i32 = 0,
    imat_dat: i8 = 0,
    out_file: []const u8 = "",
    prompt_file: []const u8 = "",

    fn of(p: *const Params) CollectorParams {
        return .{
            .n_ctx = p.n_ctx,
            .n_parallel = p.n_parallel,
            .process_output = p.process_output,
            .load_mtp = p.load_mtp,
            .n_out_freq = p.n_out_freq,
            .n_save_freq = p.n_save_freq,
            .imat_dat = p.imat_dat,
            .out_file = p.out_file,
            .prompt_file = p.prompt_file,
        };
    }
};

/// Ports `IMatrixCollector` (imatrix.cpp:69 @620d2225c). The C++ locks a
/// mutex in `collect_imatrix`; the scheduler calls it from the thread
/// computing the graph, one node at a time, so there is nothing to lock.
const Collector = struct {
    stats: libcxx.StringMap(Stats),
    params: CollectorParams = .{},
    datasets: std.ArrayList([]const u8) = .empty,
    last_chunk: i32 = 0,
    chunk_size: i32 = 0,
    n_layer_nextn: i32 = 0,
    src1_data: std.ArrayList(u8) = .empty,
    ids: std.ArrayList(u8) = .empty, // the expert ids from ggml_mul_mat_id

    /// Ports `e_chunk_size` (imatrix.cpp:90 @620d2225c), a member of `IMatrixCollector`.
    fn eChunkSize(self: *const Collector) i32 {
        return if (self.chunk_size > 0) self.chunk_size else @divTrunc(self.params.n_ctx, self.params.n_parallel);
    }

    /// The `n_chunk > m_last_chunk` step both arms of `collect_imatrix` end on.
    fn maybeSave(self: *Collector, n_chunk: i32) !void {
        if (n_chunk > self.last_chunk) {
            const chunk_step = n_chunk - self.last_chunk;
            self.last_chunk = n_chunk;
            if (@divTrunc(@rem(self.last_chunk, self.params.n_out_freq), chunk_step) == 0) try self.save(-1);
            if (self.params.n_save_freq > 0 and @divTrunc(@rem(self.last_chunk, self.params.n_save_freq), chunk_step) == 0) try self.save(self.last_chunk);
        }
    }

    /// Ports `collect_imatrix` (imatrix.cpp:561 @620d2225c), a member of `IMatrixCollector`.
    fn collect(self: *Collector, t: *c.ggml_tensor, ask: bool) !bool {
        const func = "collect_imatrix";
        const src0: *c.ggml_tensor = t.src[0];
        const src1: *c.ggml_tensor = t.src[1];
        const wname = filterTensorName(std.mem.sliceTo(&src0.name, 0));

        const chunk_size: i32 = @divTrunc(self.params.n_ctx, self.params.n_parallel);

        // when ask is true, the scheduler wants to know if we are interested in data from this tensor
        // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
        if (ask) {
            if (t.op == c.GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
            if (t.op != c.GGML_OP_MUL_MAT) return false;
            // why are small batches ignored (<16 tokens)?
            if (src1.ne[1] < 16 or src1.type != c.GGML_TYPE_F32) return false;
            const is_output = std.mem.eql(u8, wname, "output.weight") or std.mem.eql(u8, wname, "token_embd.weight") or std.mem.eql(u8, wname, "nextn.post_projection.weight");
            const is_nextn = std.mem.eql(u8, wname, "nextn.pre_projection.weight");
            if (!(std.mem.startsWith(u8, wname, "blk.") or (self.params.process_output and is_output) or (self.params.load_mtp and is_nextn))) return false;
            return true;
        }

        // copy the data from the GPU memory if needed
        const is_host = c.ggml_backend_buffer_is_host(src1.buffer);

        if (!is_host) {
            const src1_nbytes = c.ggml_nbytes(src1);
            try self.src1_data.resize(gpa, src1_nbytes);
            c.ggml_backend_tensor_get(src1, self.src1_data.items.ptr, 0, src1_nbytes);
        }

        const data: [*]const u8 = if (is_host) @ptrCast(src1.data) else self.src1_data.items.ptr;
        ggmlAssert(src1.nb[0] == c.ggml_element_size(src1), 596, "src1->nb[0] == ggml_element_size(src1)");

        // this has been adapted to the new format of storing merged experts in a single 3d tensor
        if (t.op == c.GGML_OP_MUL_MAT_ID) {
            //   ids  -> [n_experts_used, n_tokens]
            //   src1 -> [cols, n_expert_used, n_tokens]
            const ids: *c.ggml_tensor = t.src[2];
            const n_as = src0.ne[2];
            const n_ids = ids.ne[0];

            ggmlAssert(ids.ne[1] == src1.ne[2], 611, "ids->ne[1] == src1->ne[2]");

            // the extra dimension would need to be stored somewhere to be reflected in the imatrix file
            if (c.ggml_nrows(src1) != src1.ne[1] * src1.ne[2]) {
                log.err("%s: tensor has more than 3 dimensions: %.*s", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr });
                ggmlAssert(false, 616, "false");
            }

            try self.ids.resize(gpa, c.ggml_nbytes(ids));
            c.ggml_backend_tensor_get(ids, self.ids.items.ptr, 0, c.ggml_nbytes(ids));

            const e = (try self.stats.getOrPut(wname, .{})).value;
            const nas: usize = @intCast(n_as);

            if (e.counts.items.len == 1 and n_as > 1) {
                // broadcast, when loading an old imatrix
                try e.counts.appendNTimes(gpa, e.counts.items[0], nas - 1);
            }
            if (e.values.items.len == 0) {
                try resizeF32(&e.activations, @intCast(src1.ne[0] * n_as));
                try e.values.appendNTimes(gpa, 0, @intCast(src1.ne[0] * n_as));
                if (e.counts.items.len < nas) try e.counts.appendNTimes(gpa, 0, nas - e.counts.items.len) else e.counts.shrinkRetainingCapacity(nas);
            } else if (e.values.items.len != @as(usize, @intCast(src1.ne[0] * n_as))) {
                log.err("%s: inconsistent size for %.*s (%d vs %d)\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr, @as(c_int, @truncate(@as(i64, @intCast(e.values.items.len)))), @as(c_int, @truncate(src1.ne[0] * n_as)) });
                tc.cExit(1);
            } else if (e.counts.items.len != nas) {
                log.err("%s: inconsistent expert count for %.*s (%d vs %d)\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr, @as(c_int, @intCast(e.counts.items.len)), @as(c_int, @truncate(n_as)) });
                tc.cExit(1);
            }
            log.tmpl(.debug, 2, "%s[%d]: %32.*s, %s, %5d x %5d, %d\n", .{ func, self.last_chunk, @as(c_int, @intCast(wname.len)), wname.ptr, c.ggml_op_name(t.op), @as(c_int, @intCast(src1.ne[0])), @as(c_int, @intCast(src1.ne[2])), @as(c_int, @intCast(src1.type)) });

            const ne0: usize = @intCast(src1.ne[0]);
            const n_tokens = src1.ne[2];
            const has_act = e.activations.items.len != 0;

            // single pass over the routing ids
            const touched = try gpa.alloc(u8, nas);
            defer gpa.free(touched);
            @memset(touched, 0);
            var idx: i64 = 0;
            while (idx < n_ids) : (idx += 1) {
                var row: i64 = 0;
                while (row < n_tokens) : (row += 1) {
                    const off: usize = @intCast(row * @as(i64, @intCast(ids.nb[1])) + idx * @as(i64, @intCast(ids.nb[0])));
                    const ex = std.mem.readInt(i32, self.ids.items[off..][0..4], .little);
                    ggmlAssert(ex >= 0 and ex < n_as, 652, "ex >= 0 && ex < n_as"); // sanity check
                    const j11 = @rem(idx, src1.ne[1]);
                    const x: [*]const f32 = @ptrCast(@alignCast(data + @as(usize, @intCast(j11 * @as(i64, @intCast(src1.nb[1])) + row * @as(i64, @intCast(src1.nb[2]))))));
                    const exu: usize = @intCast(ex);
                    const acc = e.values.items[exu * ne0 ..];

                    e.counts.items[exu] += 1;
                    touched[exu] = 1;
                    for (0..ne0) |j| acc[j] = @mulAdd(f32, x[j], x[j], acc[j]);
                    // legacy imatrix tensors do not have activation sums
                    if (has_act) {
                        const act = e.activations.items[exu * ne0 ..];
                        for (0..ne0) |j| act[j] += x[j];
                    }
                }
            }

            // check for non-finite values, only checking experts that were routed to and touched
            for (0..nas) |ex| {
                if (touched[ex] != 0 and !allFinite(e.values.items[ex * ne0 ..][0..ne0])) {
                    log.err("%s: non-finite values detected in %.*s\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr });
                    tc.cExit(1);
                }
            }

            for (0..nas) |ex| {
                const counts = (self.stats.get(wname).?).counts.items;
                const n_chunk = rowsToChunks(counts[ex], chunk_size);
                try self.maybeSave(n_chunk);
            }
        } else {
            const e = (try self.stats.getOrPut(wname, .{})).value;
            const n_mat = src0.ne[2] * src0.ne[3];

            // use a single count per dense tensor
            // (necessary when merging older GGUF-imatrix files with 3d tensors)
            if (e.counts.items.len > 1) {
                var all_equal = true;
                for (e.counts.items[1..]) |v| {
                    if (e.counts.items[0] != v) {
                        all_equal = false;
                        break;
                    }
                }
                if (all_equal) e.counts.shrinkRetainingCapacity(1);
            }
            if (e.values.items.len == 0) {
                try resizeF32(&e.activations, @intCast(src1.ne[0] * n_mat));
                try e.values.appendNTimes(gpa, 0, @intCast(src1.ne[0] * n_mat));
                if (e.counts.items.len == 0) try e.counts.append(gpa, 0) else e.counts.shrinkRetainingCapacity(1);
            } else if (e.values.items.len != @as(usize, @intCast(src1.ne[0] * n_mat))) {
                log.err("%s: inconsistent size for %.*s (%d vs %d)\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr, @as(c_int, @truncate(@as(i64, @intCast(e.values.items.len)))), @as(c_int, @truncate(src1.ne[0] * n_mat)) });
                tc.cExit(1);
            }
            log.tmpl(.debug, 2, "%s[%d]: %32.*s, %s, %5d x %5d x %5d, %d\n", .{ func, self.last_chunk, @as(c_int, @intCast(wname.len)), wname.ptr, c.ggml_op_name(t.op), @as(c_int, @intCast(src1.ne[0])), @as(c_int, @intCast(src1.ne[1])), @as(c_int, @intCast(src1.ne[2])), @as(c_int, @intCast(src1.type)) });

            const ne0: usize = @intCast(src1.ne[0]);
            const has_act = e.activations.items.len != 0;

            var j3: i64 = 0;
            while (j3 < src1.ne[3]) : (j3 += 1) {
                var j2: i64 = 0;
                while (j2 < src1.ne[2]) : (j2 += 1) {
                    // handle 3D+ tensors, but flatten 3D+ activations when model tensor is 2D
                    const mat_id: usize = @intCast(@rem(j3, src0.ne[3]) * src0.ne[2] + @rem(j2, src0.ne[2]));
                    const acc = e.values.items[mat_id * ne0 ..];

                    var row: i64 = 0;
                    while (row < src1.ne[1]) : (row += 1) {
                        const off = row * @as(i64, @intCast(src1.nb[1])) + j2 * @as(i64, @intCast(src1.nb[2])) + j3 * @as(i64, @intCast(src1.nb[3]));
                        const x: [*]const f32 = @ptrCast(@alignCast(data + @as(usize, @intCast(off))));
                        for (0..ne0) |j| acc[j] = @mulAdd(f32, x[j], x[j], acc[j]);
                        if (has_act) {
                            const act = e.activations.items[mat_id * ne0 ..];
                            for (0..ne0) |j| act[j] += x[j];
                        }
                    }
                }
            }

            // check for non-finite values
            if (!allFinite(e.values.items)) {
                log.err("%s: non-finite values detected in %.*s\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr });
                tc.cExit(1);
            }
            // only 1 count in practice, except when a tensor is used for both MUL_MAT_ID and MUL_MAT
            const n_counts = e.counts.items.len;
            for (0..n_counts) |i| {
                const counts = (self.stats.get(wname).?).counts.items;
                counts[i] += @divTrunc(c.ggml_nrows(src1), n_mat);
                const n_chunk = rowsToChunks(counts[i], chunk_size);
                try self.maybeSave(n_chunk);
            }
        }

        return true;
    }

    /// Ports `save_imatrix_legacy` (imatrix.cpp:762 @620d2225c), a member of `IMatrixCollector`.
    fn saveLegacy(self: *const Collector, ncall_in: i32) !void {
        const func = "save_imatrix_legacy";
        var fname: std.ArrayList(u8) = .empty;
        defer fname.deinit(gpa);
        try fname.appendSlice(gpa, self.params.out_file);
        if (ncall_in > 0) try fname.print(gpa, ".at_{d}", .{ncall_in});
        try fname.append(gpa, 0);

        // warn when writing imatrix entries that do not have full data
        // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

        var n_entries: i32 = 0;
        var to_store: std.ArrayList([]const u8) = .empty;
        defer to_store.deinit(gpa);

        var is_first = true; // for printing
        var it = self.stats.iterator();
        while (it.next()) |kv| {
            const n_all: c_int = @intCast(kv.value.counts.items.len);
            if (n_all == 0) continue;

            var n_zeros: c_int = 0;
            for (kv.value.counts.items) |cnt| {
                if (@as(c_int, @truncate(cnt)) == 0) n_zeros += 1;
            }

            if (n_zeros != 0 and is_first) {
                log.inf("\n", .{});
                is_first = false;
            }

            if (n_zeros == n_all) {
                log.wrn("%s: entry '%40.*s' has no data - skipping\n", .{ func, @as(c_int, @intCast(kv.key.len)), kv.key.ptr });
                continue;
            }

            if (n_zeros > 0) {
                const pct: f32 = 100.0 * @as(f32, @floatFromInt(n_all - n_zeros)) / @as(f32, @floatFromInt(n_all));
                log.wrn("%s: entry '%40.*s' has partial data (%.2f%%)\n", .{ func, @as(c_int, @intCast(kv.key.len)), kv.key.ptr, @as(f64, pct) });
            }

            n_entries += 1;
            try to_store.append(gpa, kv.key);
        }

        if (to_store.items.len < self.stats.count()) {
            log.wrn("%s: storing only %zu out of %zu entries\n", .{ func, to_store.items.len, self.stats.count() });
        }

        // deterministic tensor name order
        std.mem.sort([]const u8, to_store.items, {}, lessThan);

        const chunk_size = self.eChunkSize();

        const out = std.c.fopen(@ptrCast(fname.items.ptr), "wb");
        defer if (out) |f| {
            _ = std.c.fclose(f);
        };
        const W = struct {
            fn bytes(f: ?*std.c.FILE, b: []const u8) void {
                if (f) |ff| _ = std.c.fwrite(b.ptr, 1, b.len, ff);
            }
            fn int(f: ?*std.c.FILE, v: i32) void {
                bytes(f, std.mem.asBytes(&v));
            }
        };
        W.int(out, n_entries);
        for (to_store.items) |name| {
            const stat = self.stats.get(name).?;
            W.int(out, @intCast(name.len));
            W.bytes(out, name);
            // ceiling division to avoid accidental zeros
            const maxc = std.mem.max(i64, stat.counts.items);
            const ncall: i32 = @truncate(@divTrunc(maxc + (chunk_size - 1), chunk_size));
            W.int(out, ncall);
            const nval: i32 = @intCast(stat.values.items.len);
            const nmat: i32 = @intCast(stat.counts.items.len);
            W.int(out, nval);
            if (nval > 0 and nmat > 0) {
                const tmp = try gpa.alloc(f32, @intCast(nval));
                defer gpa.free(tmp);
                var i: i32 = 0;
                while (i < nval) : (i += 1) {
                    var count: f32 = @floatFromInt(stat.counts.items[@intCast(@divTrunc(i, @divTrunc(nval, nmat)))]);
                    var value = stat.values.items[@intCast(i)];
                    if (count == 0.0) {
                        // store 1 for partial data
                        value = 1.0;
                        count = 1.0;
                    }
                    tmp[@intCast(i)] = (value / count) * @as(f32, @floatFromInt(ncall));
                }
                W.bytes(out, std.mem.sliceAsBytes(tmp));
            }
        }

        // Write the number of call the matrix was computed with
        W.int(out, self.last_chunk);

        // Write the input filename at the end of the file to later on specify it in quantize
        {
            var dataset_file = self.params.prompt_file;
            // When there is no prompt but there were other imatrix files loaded, use the last dataset
            if (self.params.prompt_file.len == 0 and self.datasets.items.len != 0) dataset_file = self.datasets.items[self.datasets.items.len - 1];
            W.int(out, @intCast(dataset_file.len));
            W.bytes(out, dataset_file);
        }

        log.tmpl(.none, 1, "\n", .{});
        log.tmpl(.debug, 1, "%s: stored collected data after %d chunks in %s\n", .{ func, self.last_chunk, fname.items.ptr });
    }

    /// Ports `save_imatrix` (imatrix.cpp:867 @620d2225c), a member of `IMatrixCollector`.
    fn save(self: *const Collector, n_chunk: i32) !void {
        const func = "save_imatrix";
        const use_legacy_format = self.params.imat_dat;

        if (use_legacy_format > 0) return self.saveLegacy(n_chunk);

        // only warn when `--output-format gguf` is not specified
        if (use_legacy_format == 0 and !std.mem.endsWith(u8, self.params.out_file, ".gguf")) {
            log.wrn("\n%s: saving imatrix using GGUF format with a different suffix than .gguf\n", .{func});
            log.wrn("%s: if you want the previous imatrix format, use --output-format dat\n", .{func});
        }

        var fname: std.ArrayList(u8) = .empty;
        defer fname.deinit(gpa);
        try fname.appendSlice(gpa, self.params.out_file);
        if (n_chunk > 0) try fname.print(gpa, ".at_{d}", .{n_chunk});
        try fname.append(gpa, 0);

        // write imatrix entries even if they don't have full data. (can be corrected when reading)

        var to_store: std.ArrayList([]const u8) = .empty;
        defer to_store.deinit(gpa);
        var data_size: usize = 0;

        const overhead = c.ggml_tensor_overhead();
        var is_first = true; // for printing
        var it = self.stats.iterator();
        while (it.next()) |kv| {
            const n_all: c_int = @intCast(kv.value.counts.items.len);

            var n_zeros: c_int = 0;
            for (kv.value.counts.items) |cnt| {
                if (cnt == 0) n_zeros += 1;
            }

            if (n_zeros != 0 and is_first) {
                log.inf("\n", .{});
                is_first = false;
            }

            if (n_zeros > 0) {
                const pct: f32 = 100.0 * @as(f32, @floatFromInt(n_all - n_zeros)) / @as(f32, @floatFromInt(n_all));
                log.wrn("%s: entry '%40.*s' has partial data (%.2f%%)\n", .{ func, @as(c_int, @intCast(kv.key.len)), kv.key.ptr, @as(f64, pct) });
            }

            try to_store.append(gpa, kv.key);
            data_size += pad(overhead + @sizeOf(f32) * kv.value.activations.items.len, c.GGML_MEM_ALIGN);
            data_size += pad(overhead + @sizeOf(f32) * kv.value.values.items.len, c.GGML_MEM_ALIGN);
            data_size += pad(overhead + @sizeOf(f32) * kv.value.counts.items.len, c.GGML_MEM_ALIGN);
            data_size += pad(overhead + @sizeOf(f32) * 10, c.GGML_MEM_ALIGN);
        }

        // deterministic tensor name order
        std.mem.sort([]const u8, to_store.items, {}, lessThan);

        // Compute per-tensor statistics
        var tstats: std.ArrayList(TensorStatistics) = .empty;
        defer tstats.deinit(gpa);
        try tstats.ensureTotalCapacity(gpa, self.stats.count());
        var it2 = self.stats.iterator();
        while (it2.next()) |kv| _ = try computeVectorStatistics(&tstats, kv.key, kv.value);
        if (tstats.items.len != 0) try computeTensorStatistics(tstats.items, &self.stats, nextnLayerStart(tstats.items, self.n_layer_nextn));

        const params: c.ggml_init_params = .{ .mem_size = data_size, .mem_buffer = null, .no_alloc = false };
        const ctx = c.ggml_init(params);
        const ctx_gguf = c.gguf_init_empty();

        {
            var datasets: std.ArrayList([*:0]const u8) = .empty;
            defer datasets.deinit(gpa);
            for (self.datasets.items) |d| try datasets.append(gpa, (try gpa.dupeSentinel(u8, d, 0)).ptr);
            if (self.params.prompt_file.len != 0) try datasets.append(gpa, (try gpa.dupeSentinel(u8, self.params.prompt_file, 0)).ptr);

            c.gguf_set_val_str(ctx_gguf, "general.type", "imatrix");
            // Write the dataset paths
            c.gguf_set_arr_str(ctx_gguf, tc.imatrix_loader.kv_datasets, @ptrCast(datasets.items.ptr), datasets.items.len);
            // Write the number of chunks the matrix was computed with
            c.gguf_set_val_u32(ctx_gguf, tc.imatrix_loader.kv_chunk_count, @bitCast(self.last_chunk));
            c.gguf_set_val_u32(ctx_gguf, tc.imatrix_loader.kv_chunk_size, @bitCast(self.eChunkSize()));
            // Write how many of the top layers are NextN layers, so statistics can tell them apart
            if (self.n_layer_nextn > 0) c.gguf_set_val_u32(ctx_gguf, tc.imatrix_loader.kv_n_layer_nextn, @bitCast(self.n_layer_nextn));
            // Define the schema for the tensor statistics (for use in quantize.cpp)
            const stats_schema = [_][*:0]const u8{ "sum_sq", "mean", "elements", "std_deviation", "skewness", "kurtosis", "gain", "h_norm", "l2_dist", "cossim", "pearson", "covariance" };
            c.gguf_set_arr_str(ctx_gguf, tc.imatrix_loader.kv_stats_schema, @ptrCast(@constCast(&stats_schema)), 12);
            for (datasets.items) |d| gpa.free(std.mem.span(d));
        }

        for (to_store.items) |name| {
            const stat = self.stats.get(name).?;
            const nval: i32 = @intCast(stat.values.items.len);
            const nmat: i32 = @intCast(stat.counts.items.len);
            if (nval > 0 and nmat > 0) {
                const in_sum2 = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, @divTrunc(nval, nmat), nmat);
                const counts = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 1, nmat);
                _ = c.ggml_format_name(in_sum2, "%.*s.in_sum2", @as(c_int, @intCast(name.len)), name.ptr);
                _ = c.ggml_format_name(counts, "%.*s.counts", @as(c_int, @intCast(name.len)), name.ptr);

                const sd: [*]f32 = @ptrCast(@alignCast(in_sum2.*.data));
                for (stat.values.items, 0..) |v, j| sd[j] = v;
                const cd: [*]f32 = @ptrCast(@alignCast(counts.*.data));
                for (stat.counts.items, 0..) |v, j| cd[j] = @floatFromInt(v);

                c.gguf_add_tensor(ctx_gguf, in_sum2);
                c.gguf_add_tensor(ctx_gguf, counts);

                if (stat.activations.items.len != 0) {
                    const nact: i32 = @intCast(stat.activations.items.len);
                    const in_sum = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, @divTrunc(nact, nmat), nmat);
                    _ = c.ggml_format_name(in_sum, "%.*s.in_sum", @as(c_int, @intCast(name.len)), name.ptr);
                    const ad: [*]f32 = @ptrCast(@alignCast(in_sum.*.data));
                    for (stat.activations.items, 0..) |v, j| ad[j] = v;
                    c.gguf_add_tensor(ctx_gguf, in_sum);
                }
            } else {
                log.wrn("%s: no data for tensor %.*s\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr });
            }

            // Store per-tensor statistics as a small 1D tensor
            {
                var sum_sq: f64 = 0.0;
                var mean: f32 = 0.0;
                var elements: f32 = 0.0;
                var std_deviation: f32 = 0.0;
                var skewness: f32 = 0.0;
                var kurtosis: f32 = 0.0;
                var gain: f32 = 0.0;
                var h_norm: f32 = 0.0;
                var l2_dist: f32 = 0.0;
                var cossim: f32 = 0.0;
                var pearson: f32 = 0.0;
                var covariance: f32 = 0.0;
                // `tstat_index`, an index by tensor name: the names are unique
                for (tstats.items) |*ts| {
                    if (!std.mem.eql(u8, ts.tensor, name)) continue;
                    sum_sq = ts.sum;
                    mean = ts.mean;
                    elements = @floatFromInt(ts.elements);
                    std_deviation = ts.std_deviation;
                    skewness = ts.skewness;
                    kurtosis = ts.kurtosis;
                    gain = ts.gain;
                    h_norm = if (ts.elements > 1) 100.0 * (ts.entropy / log2f(@floatFromInt(ts.elements))) else fnan;
                    l2_dist = ts.l2_dist;
                    cossim = ts.cossim;
                    pearson = ts.pearson;
                    covariance = ts.covariance;
                    break;
                }

                const stats = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 12);
                _ = c.ggml_format_name(stats, "%.*s.stats", @as(c_int, @intCast(name.len)), name.ptr);
                // Store the statistics in the same order as defined in stats_schema[]
                const st: [*]f32 = @ptrCast(@alignCast(stats.*.data));
                st[0] = @floatCast(sum_sq);
                st[1] = mean;
                st[2] = elements;
                st[3] = std_deviation;
                st[4] = skewness;
                st[5] = kurtosis;
                st[6] = gain;
                st[7] = h_norm;
                st[8] = l2_dist;
                st[9] = cossim;
                st[10] = pearson;
                st[11] = covariance;
                c.gguf_add_tensor(ctx_gguf, stats);
            }
        }

        _ = c.gguf_write_to_file(ctx_gguf, @ptrCast(fname.items.ptr), false);

        log.tmpl(.none, 1, "\n", .{});
        log.tmpl(.debug, 1, "%s: stored collected data after %d chunks in %s\n", .{ func, self.last_chunk, fname.items.ptr });

        c.gguf_free(ctx_gguf);
        c.ggml_free(ctx);
    }

    /// Ports `load_imatrix` (imatrix.cpp:1058 @620d2225c), a member of `IMatrixCollector`.
    fn load(self: *Collector, a: std.mem.Allocator, file_name: [:0]const u8) !bool {
        const func = "load_imatrix";
        var loaded = try tc.imatrix_loader.Imatrix.init(a);
        if (!try tc.imatrix_loader.load(file_name, &loaded)) return false;

        const is_legacy = loaded.is_legacy;
        const loaded_nextn: i32 = loaded.n_layer_nextn;
        if (loaded_nextn > 0) {
            if (self.n_layer_nextn == 0) {
                self.n_layer_nextn = loaded_nextn;
            } else if (self.n_layer_nextn != loaded_nextn) {
                log.wrn("%s: NextN layer count mismatch in %s: %d != %d, using %d\n", .{ func, file_name.ptr, loaded_nextn, self.n_layer_nextn, self.n_layer_nextn });
            }
        }

        if (!is_legacy and loaded.chunk_size > 0) {
            if (self.chunk_size == 0) {
                self.chunk_size = loaded.chunk_size;
            } else if (self.chunk_size != loaded.chunk_size) {
                log.wrn("%s: chunk size mismatch in %s: %d != %d, using %d\n", .{ func, file_name.ptr, loaded.chunk_size, self.chunk_size, self.chunk_size });
            }
        }

        const chunk_size = self.eChunkSize();

        // `common_imatrix::entries` is a `std::map`: name order
        const names = try loaded.sortedNames(a);
        for (names) |name| {
            const entry = loaded.entries.get(name).?;
            const e = (try self.stats.getOrPut(name, .{})).value;

            if (is_legacy) {
                if (e.values.items.len == 0) {
                    try e.values.appendNTimes(gpa, 0, entry.sums.items.len);
                    if (e.counts.items.len == 0) try e.counts.append(gpa, 0) else e.counts.shrinkRetainingCapacity(1);
                }
                for (entry.sums.items, 0..) |s, j| e.values.items[j] = @mulAdd(f32, s, @floatFromInt(chunk_size), e.values.items[j]);
                for (e.counts.items) |*cnt| cnt.* += entry.counts.items[0] * chunk_size;
                e.activations.clearRetainingCapacity();
            } else {
                // GGUF format: raw sums and counts, accumulate directly
                const nval = entry.sums.items.len;
                const ncounts = entry.counts.items.len;
                const entry_act: []const f32 = entry.activations.items;
                const nact = entry_act.len;
                const first_contribution = e.values.items.len == 0;

                if (e.values.items.len == 0) {
                    try e.values.appendNTimes(gpa, 0, nval);
                } else if (nval != e.values.items.len) {
                    log.err("%s: mismatched sums size for %.*s: %zu != %zu\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr, nval, e.values.items.len });
                    return false;
                }

                if (e.counts.items.len == 0) {
                    try e.counts.appendNTimes(gpa, 0, ncounts);
                } else if (e.counts.items.len == 1 and ncounts > 1) {
                    try e.counts.appendNTimes(gpa, e.counts.items[0], ncounts - 1);
                } else if (ncounts != e.counts.items.len) {
                    log.err("%s: mismatched counts size for %.*s: %zu != %zu\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr, ncounts, e.counts.items.len });
                    return false;
                }

                for (entry.sums.items, 0..) |s, j| e.values.items[j] += s;
                for (entry.counts.items, 0..) |cnt, j| e.counts.items[j] += cnt;

                if (nact > 0 and (first_contribution or e.activations.items.len != 0)) {
                    if (nact != nval) {
                        log.err("%s: mismatched activations size for %.*s: %zu != %zu\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr, nact, nval });
                        return false;
                    }

                    if (e.activations.items.len == 0) {
                        try resizeF32(&e.activations, nact);
                    } else if (nact != e.activations.items.len) {
                        log.err("%s: mismatched activations size for %.*s: %zu != %zu\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr, nact, e.activations.items.len });
                        return false;
                    }

                    for (entry_act, 0..) |v, j| e.activations.items[j] += v;
                } else {
                    e.activations.clearRetainingCapacity();
                }
            }
        }

        for (loaded.datasets.items) |d| try self.datasets.append(gpa, try gpa.dupe(u8, d));

        // Calculate the last chunk count
        var max_count: i64 = 0;
        var it = self.stats.iterator();
        while (it.next()) |kv| {
            for (kv.value.counts.items) |cnt| max_count = @max(max_count, cnt);
        }

        self.last_chunk = rowsToChunks(max_count, chunk_size);

        return true;
    }
};

/// `v.resize(n, 0)` from empty.
fn resizeF32(v: *std.ArrayList(f32), n: usize) !void {
    v.clearRetainingCapacity();
    try v.appendNTimes(gpa, 0, n);
}

fn pad(x: usize, n: usize) usize {
    return (x + n - 1) & ~(n - 1);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Ports `filter_tensor_name` (imatrix.cpp:97 @620d2225c): remove any prefix
/// and suffix, `CUDA0#blk.0.attn_k.weight#0` => `blk.0.attn_k.weight`.
fn filterTensorName(name: []const u8) []const u8 {
    const p = std.mem.indexOfScalar(u8, name, '#') orelse return name;
    const rest = name[p + 1 ..];
    const q = std.mem.indexOfScalar(u8, rest, '#') orelse return rest;
    return rest[0..q];
}

/// Ports `process_tensor_name` (imatrix.cpp:114 @620d2225c). `tensor` is a
/// join of the name's parts before `weight`, the block and its index left
/// out -- by the C++'s own skip, which over a four-part name jumps straight
/// to `weight` and leaves `tensor` empty, so it falls back to the input.
/// The joined form is written into `buf`, which outlives the result.
fn processTensorName(input: []const u8, layer: *[]const u8, tensor: *[]const u8, buf: *std.ArrayList(u8)) void {
    layer.* = "";
    tensor.* = "";
    buf.clearRetainingCapacity();

    var parts: std.ArrayList([]const u8) = .empty;
    defer parts.deinit(gpa);
    // `std::getline(stream, item, '.')`: no empty last item after a final '.'
    var it = std.mem.splitScalar(u8, input, '.');
    while (it.next()) |item| {
        if (it.index == null and item.len == 0) break;
        parts.append(gpa, item) catch oom();
    }
    const name = parts.items;
    for (0..name.len) |i| {
        if (std.mem.eql(u8, name[i], "blk") and i + 1 < name.len) {
            layer.* = name[i + 1];
            break;
        }
    }
    for (0..name.len) |i| {
        if (std.mem.eql(u8, name[i], "weight") and i > 0) {
            var j: usize = 0;
            while (j < name.len) : (j += 1) {
                if (std.mem.eql(u8, name[j], "blk")) {
                    j += if (name.len > 4) 1 else 2;
                    continue;
                }
                if (j == i) break;
                if (buf.items.len != 0) buf.append(gpa, '.') catch oom();
                buf.appendSlice(gpa, name[j]) catch oom();
            }
            break;
        }
    }

    tensor.* = buf.items;
    if (tensor.len == 0) tensor.* = input;
    if (layer.len == 0) layer.* = "-";
}

fn oom() noreturn {
    _ = c.ggml_abort("imatrix.cpp", 0, "out of memory");
    unreachable;
}

/// The block number of `tensor`, through `process_tensor_name` and `std::stoi`.
fn tensorBlk(tensor: []const u8) ?c_int {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var layer: []const u8 = "";
    var name: []const u8 = "";
    processTensorName(tensor, &layer, &name, &buf);
    return stoiOr(layer);
}

/// Ports `compute_tensor_averages` (imatrix.cpp:150 @620d2225c). An empty
/// result is the C++'s empty vector.
fn computeTensorAverages(tstats: *const Stats, use_activations: bool) ![]f32 {
    if (tstats.counts.items.len == 0) return &.{};

    const n_mat = tstats.counts.items.len;
    const len = if (use_activations) tstats.activations.items.len else tstats.values.items.len;
    if (len == 0 or n_mat == 0 or len % n_mat != 0) return &.{};

    const row = len / n_mat;
    const vec = try gpa.alloc(f32, len);
    @memset(vec, fnan);

    var has_valid = false;
    for (0..n_mat) |m| {
        const cnt: f32 = @floatFromInt(tstats.counts.items[m]);
        const off = m * row;

        if (cnt <= 0.0) continue;

        has_valid = true;
        const scale = 1.0 / cnt;
        const src = if (use_activations) tstats.activations.items[off..] else tstats.values.items[off..];
        const dst = vec[off..];

        for (0..row) |j| dst[j] = src[j] * scale;
    }

    if (!has_valid) {
        gpa.free(vec);
        return &.{};
    }

    return vec;
}

/// Ports `compute_vector_statistics` (imatrix.cpp:180 @620d2225c).
fn computeVectorStatistics(tstats: *std.ArrayList(TensorStatistics), name: []const u8, e: *const Stats) !bool {
    const func = "compute_vector_statistics";
    const legacy = e.activations.items.len == 0;
    const n_mat = e.counts.items.len;
    const len = if (legacy) e.values.items.len else e.activations.items.len;

    if (n_mat == 0 or len == 0 or len % n_mat != 0) {
        log.err("%s: data size mismatch or empty for tensor %.*s\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr });
        return false;
    }
    if (!legacy and e.values.items.len != len) {
        log.err("%s: activations/values size mismatch for %.*s\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr });
        return false;
    }

    const row_size = len / n_mat;
    var sum: f64 = 0.0;
    var mean: f64 = 0.0;
    var sum_sq_diff: f64 = 0.0;
    var sum_cu_diff: f64 = 0.0;
    var sum_qd_diff: f64 = 0.0;
    var sum_energy: f64 = 0.0;
    var valid_n: usize = 0;

    // Mean
    for (0..n_mat) |i| {
        const cnt: f32 = @floatFromInt(e.counts.items[i]);
        if (cnt <= 0.0) continue;

        const inv_c = 1.0 / @as(f64, cnt);
        const off = i * row_size;

        for (0..row_size) |j| {
            const v_act: f64 = if (legacy) 0.0 else @as(f64, e.activations.items[off + j]) * inv_c;
            const v_val: f64 = @as(f64, e.values.items[off + j]) * inv_c;
            const v = if (legacy) v_val else v_act; // Use activation average for non-legacy
            if (!std.math.isFinite(v) or !std.math.isFinite(v_val)) continue;

            sum += v_val;
            valid_n += 1;
            const delta = v - mean;
            mean += delta / @as(f64, @floatFromInt(valid_n));

            if (v_val > 0.0) sum_energy += v_val;
        }
    }

    if (valid_n == 0) return false;

    var std_deviation: f32 = 0.0;
    var entropy: f64 = 0.0;

    // Std Dev, Skew, Kurtosis, Entropy
    const inv_sum_energy: f64 = if (sum_energy > 0.0) 1.0 / sum_energy else 0.0;
    const log2_inv = 1.0 / logD(2.0);

    for (0..n_mat) |i| {
        const cnt: f32 = @floatFromInt(e.counts.items[i]);
        if (cnt <= 0.0) continue;
        const inv_c = 1.0 / @as(f64, cnt);
        const off = i * row_size;

        for (0..row_size) |j| {
            const v_act: f64 = if (legacy) 0.0 else @as(f64, e.activations.items[off + j]) * inv_c;
            const v_val: f64 = @as(f64, e.values.items[off + j]) * inv_c;
            const v = if (legacy) v_val else v_act;
            if (!std.math.isFinite(v) or !std.math.isFinite(v_val)) continue;
            const diff = v - mean;

            sum_sq_diff = @mulAdd(f64, diff, diff, sum_sq_diff);
            sum_cu_diff = @mulAdd(f64, diff * diff, diff, sum_cu_diff);
            sum_qd_diff = @mulAdd(f64, diff * diff * diff, diff, sum_qd_diff);

            // Entropy (Distribution of Energy)
            if (inv_sum_energy > 0.0) {
                const v_energy: f64 = @as(f64, e.values.items[off + j]) * inv_c;
                const p = (if (0.0 < v_energy) v_energy else 0.0) * inv_sum_energy;
                if (p > 1e-10) entropy = @mulAdd(f64, -(p * logD(p)), log2_inv, entropy);
            }
        }
    }

    const vn: f64 = @floatFromInt(valid_n);
    const variance: f64 = if (valid_n > 1) sum_sq_diff / vn else 0.0;
    std_deviation = sqrtf(@floatCast(if (variance < 0.0) 0.0 else variance));
    var skewness: f32 = 0.0;
    var kurtosis: f32 = 0.0;
    if (std_deviation > 1e-10) {
        const m2 = sum_sq_diff / vn;
        skewness = @floatCast(sum_cu_diff / vn / (m2 * sqrt(m2)));
        kurtosis = @floatCast(sum_qd_diff / vn / (m2 * m2) - 3.0);
    }

    try tstats.append(gpa, .{
        .tensor = name,
        .legacy = legacy,
        .sum = sum,
        .mean = @floatCast(mean),
        .elements = @intCast(valid_n),
        .std_deviation = std_deviation,
        .skewness = skewness,
        .kurtosis = kurtosis,
        .gain = fnan,
        .entropy = @floatCast(entropy),
        .l2_dist = fnan,
        .cossim = fnan,
        .pearson = fnan,
        .covariance = fnan,
    });

    return true;
}

/// Ports `nextn_layer_start` (imatrix.cpp:291 @620d2225c).
fn nextnLayerStart(tstats: []const TensorStatistics, n_layer_nextn: i32) c_int {
    var max_blk: c_int = -1;
    var first: c_int = int_max;

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    for (tstats) |ts| {
        var layer_str: []const u8 = "";
        var name: []const u8 = "";
        processTensorName(ts.tensor, &layer_str, &name, &buf);

        const blk = stoiOr(layer_str) orelse continue;

        max_blk = @max(max_blk, blk);
        if (std.mem.startsWith(u8, name, "nextn.")) first = @min(first, blk);
    }

    if (n_layer_nextn > 0 and max_blk >= 0) return @max(0, max_blk + 1 - n_layer_nextn);

    return first;
}

/// Ports `layer_label` (imatrix.cpp:312 @620d2225c). Written into `buf`.
fn layerLabel(buf: *[32]u8, blk: c_int, nextn_start: c_int) [:0]const u8 {
    if (blk < 0 or blk == int_max) return "-";
    if (blk >= nextn_start) return std.fmt.bufPrintSentinel(buf, "mtp{d}", .{blk - nextn_start}, 0) catch |err| switch (err) {
        error.NoSpaceLeft => unreachable,
    };

    return std.fmt.bufPrintSentinel(buf, "{d}", .{blk}, 0) catch |err| switch (err) {
        error.NoSpaceLeft => unreachable,
    };
}

/// Ports `compute_tensor_statistics` (imatrix.cpp:319 @620d2225c).
fn computeTensorStatistics(tstats: []TensorStatistics, mstats: *const libcxx.StringMap(Stats), nextn_start: c_int) !void {
    const func = "compute_tensor_statistics";
    var tensor_map: std.StringHashMapUnmanaged(usize) = .empty;
    defer tensor_map.deinit(gpa);
    try tensor_map.ensureTotalCapacity(gpa, @intCast(tstats.len));
    for (tstats, 0..) |ts, i| tensor_map.putAssumeCapacity(ts.tensor, i);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var tname: std.ArrayList(u8) = .empty;
    defer tname.deinit(gpa);

    for (tstats) |*ts| {
        var layer_str: []const u8 = "";
        var dummy_tensor: []const u8 = "";
        processTensorName(ts.tensor, &layer_str, &dummy_tensor, &buf);

        const blk = stoiOr(layer_str) orelse continue;
        if (blk <= 0) continue;
        if (blk == nextn_start) continue;

        const blk_first: c_int = if (blk > nextn_start) nextn_start else 0;
        var needle: std.ArrayList(u8) = .empty;
        defer needle.deinit(gpa);
        try needle.print(gpa, "blk.{s}", .{layer_str});
        const blk_start_pos = std.mem.indexOf(u8, ts.tensor, needle.items) orelse continue;

        var found: ?usize = null;
        var prev: c_int = blk - 1;
        while (prev >= blk_first and found == null) : (prev -= 1) {
            tname.clearRetainingCapacity();
            const end = @min(ts.tensor.len, blk_start_pos + layer_str.len + 4);
            try tname.appendSlice(gpa, ts.tensor[0..blk_start_pos]);
            try tname.print(gpa, "blk.{d}", .{prev});
            try tname.appendSlice(gpa, ts.tensor[end..]);
            found = tensor_map.get(tname.items);
        }

        const pi = found orelse {
            log.wrn("%s: no preceding-layer tensor for '%.*s'\n", .{ func, @as(c_int, @intCast(ts.tensor.len)), ts.tensor.ptr });
            continue;
        };

        const prev_ts = &tstats[pi];
        const curr_e = mstats.get(ts.tensor) orelse continue;
        const prev_e = mstats.get(prev_ts.tensor) orelse continue;

        // one side may not have no activation sums so compare both on the energy
        const use_activations = !ts.legacy and !prev_ts.legacy;
        const curr_avg = try computeTensorAverages(curr_e, use_activations);
        defer if (curr_avg.len != 0) gpa.free(curr_avg);
        const prev_avg = try computeTensorAverages(prev_e, use_activations);
        defer if (prev_avg.len != 0) gpa.free(prev_avg);

        if (curr_avg.len == 0 or curr_avg.len != prev_avg.len) continue;

        var dot_prod: f64 = 0.0;
        var norm1_sq: f64 = 0.0;
        var norm2_sq: f64 = 0.0;
        var l2_dist_sq: f64 = 0.0;
        var sum_c: f64 = 0.0;
        var sum_p: f64 = 0.0;
        const n = curr_avg.len;
        var valid_n: usize = 0;

        // Sums for Means
        for (0..n) |i| {
            const c_val: f64 = curr_avg[i];
            const p_val: f64 = prev_avg[i];
            if (std.math.isFinite(c_val) and std.math.isFinite(p_val)) {
                sum_c += c_val;
                sum_p += p_val;
                valid_n += 1;
            }
        }

        if (valid_n == 0) continue;
        const vn: f64 = @floatFromInt(valid_n);
        const mean_c = sum_c / vn;
        const mean_p = sum_p / vn;

        var cov_sum: f64 = 0.0;
        var var_c_sum: f64 = 0.0;
        var var_p_sum: f64 = 0.0;

        // Metrics
        for (0..n) |i| {
            const c_val: f64 = curr_avg[i];
            const p_val: f64 = prev_avg[i];
            if (!std.math.isFinite(c_val) or !std.math.isFinite(p_val)) continue;

            // Cosine Similarity & L2 Distance
            dot_prod = @mulAdd(f64, c_val, p_val, dot_prod);
            norm1_sq = @mulAdd(f64, c_val, c_val, norm1_sq);
            norm2_sq = @mulAdd(f64, p_val, p_val, norm2_sq);
            const diff = c_val - p_val;
            l2_dist_sq = @mulAdd(f64, diff, diff, l2_dist_sq);

            // Pearson (Centered stats)
            const dc = c_val - mean_c;
            const dp = p_val - mean_p;
            cov_sum = @mulAdd(f64, dc, dp, cov_sum);
            var_c_sum = @mulAdd(f64, dc, dc, var_c_sum);
            var_p_sum = @mulAdd(f64, dp, dp, var_p_sum);
        }

        ts.n_features = @intCast(valid_n);
        ts.dot_prod = dot_prod;
        ts.norm1_sq = norm1_sq;
        ts.norm2_sq = norm2_sq;
        ts.cov_sum = cov_sum;
        ts.var_c_sum = var_c_sum;
        ts.var_p_sum = var_p_sum;
        ts.l2_dist_sq = l2_dist_sq;
        ts.l2_dist = @floatCast(sqrt(l2_dist_sq));

        if (valid_n > 1) {
            ts.covariance = @floatCast(cov_sum / vn);
        }

        if (norm1_sq > 0.0 and norm2_sq > 0.0) {
            ts.cossim = @floatCast(dot_prod / (sqrt(norm1_sq) * sqrt(norm2_sq)));
            ts.cossim = cClamp(ts.cossim, -1.0, 1.0);
        } else {
            ts.cossim = if (norm1_sq == 0.0 and norm2_sq == 0.0) fnan else 0.0;
        }

        if (var_c_sum > 0.0 and var_p_sum > 0.0) {
            ts.pearson = @floatCast(cov_sum / (sqrt(var_c_sum) * sqrt(var_p_sum)));
            ts.pearson = cClamp(ts.pearson, -1.0, 1.0);
        } else {
            ts.pearson = if (var_c_sum == 0.0 and var_p_sum == 0.0) fnan else 0.0;
        }

        // `1e-10f`, a float literal compared as a double
        const eps_f32: f32 = 1e-10;
        const eps: f64 = eps_f32;
        if (prev_ts.sum > eps) {
            ts.gain = @floatCast(sqrt(ts.sum / @as(f64, @floatFromInt(ts.elements))) / sqrt(prev_ts.sum / @as(f64, @floatFromInt(prev_ts.elements))));
            ts.sum_prev = prev_ts.sum;
            ts.elements_prev = prev_ts.elements;
        } else {
            ts.gain = if (ts.sum <= eps) 1.0 else fnan;
        }
    }
}

/// A `std::map<int, float>`: kept sorted by key, read by `find`.
const LayerMap = struct {
    keys: std.ArrayList(c_int) = .empty,
    vals: std.ArrayList(f32) = .empty,

    fn put(self: *LayerMap, k: c_int, v: f32) !void {
        var pos: usize = 0;
        while (pos < self.keys.items.len and self.keys.items[pos] < k) pos += 1;
        if (pos < self.keys.items.len and self.keys.items[pos] == k) {
            self.vals.items[pos] = v;
            return;
        }
        try self.keys.insert(gpa, pos, k);
        try self.vals.insert(gpa, pos, v);
    }

    fn get(self: *const LayerMap, k: c_int) ?f32 {
        for (self.keys.items, self.vals.items) |kk, v| {
            if (kk == k) return v;
        }
        return null;
    }

    fn deinit(self: *LayerMap) void {
        self.keys.deinit(gpa);
        self.vals.deinit(gpa);
    }
};

/// Mirrors `compute_layer_statistics`'s `layer_aggregation` (imatrix.cpp:459 @620d2225c).
const LayerAggregation = struct {
    sum_dot_prod: f64 = 0.0,
    sum_norm1_sq: f64 = 0.0,
    sum_norm2_sq: f64 = 0.0,
    sum_l2_dist_sq: f64 = 0.0,
    sum_cov: f64 = 0.0,
    sum_var_c: f64 = 0.0,
    sum_var_p: f64 = 0.0,
    sum_energy_curr: f64 = 0.0,
    sum_energy_prev: f64 = 0.0,
    sum_elements_curr: i64 = 0,
    sum_elements_prev: i64 = 0,
    sum_n_features: i64 = 0,
    n_tensors: c_int = 0,
};

/// The five maps `compute_layer_statistics` fills.
const LayerStatistics = struct {
    cossim: LayerMap = .{},
    l2_dist: LayerMap = .{},
    pearson: LayerMap = .{},
    covariance: LayerMap = .{},
    gain: LayerMap = .{},

    fn deinit(self: *LayerStatistics) void {
        self.cossim.deinit();
        self.l2_dist.deinit();
        self.pearson.deinit();
        self.covariance.deinit();
        self.gain.deinit();
    }
};

/// Ports `compute_layer_statistics` (imatrix.cpp:451 @620d2225c).
fn computeLayerStatistics(tstats: []const TensorStatistics, out: *LayerStatistics) !void {
    // `std::map<int, layer_aggregation>`: sorted by layer
    var keys: std.ArrayList(c_int) = .empty;
    defer keys.deinit(gpa);
    var aggs: std.ArrayList(LayerAggregation) = .empty;
    defer aggs.deinit(gpa);

    for (tstats) |ts| {
        const blk: c_int = tensorBlk(ts.tensor) orelse -1;

        if (blk <= 0) continue;

        if (ts.norm1_sq == 0.0 and ts.norm2_sq == 0.0 and ts.l2_dist_sq == 0.0) continue;
        var pos: usize = 0;
        while (pos < keys.items.len and keys.items[pos] < blk) pos += 1;
        if (pos == keys.items.len or keys.items[pos] != blk) {
            try keys.insert(gpa, pos, blk);
            try aggs.insert(gpa, pos, .{});
        }
        const entry = &aggs.items[pos];
        entry.sum_dot_prod += ts.dot_prod;
        entry.sum_norm1_sq += ts.norm1_sq;
        entry.sum_norm2_sq += ts.norm2_sq;
        entry.sum_l2_dist_sq += ts.l2_dist_sq;
        entry.sum_cov += ts.cov_sum;
        entry.sum_var_c += ts.var_c_sum;
        entry.sum_var_p += ts.var_p_sum;
        entry.n_tensors += 1;
        if (ts.n_features > 0) entry.sum_n_features += ts.n_features;

        // skip tensors with no match in a previous layer
        if (ts.elements_prev > 0) {
            entry.sum_energy_curr += ts.sum;
            entry.sum_energy_prev += ts.sum_prev;
            entry.sum_elements_curr += ts.elements;
            entry.sum_elements_prev += ts.elements_prev;
        }
    }

    for (keys.items, aggs.items) |layer, agg| {
        if (agg.n_tensors == 0) continue;

        var cossim: f32 = 0.0;
        if (agg.sum_norm1_sq > 0.0 and agg.sum_norm2_sq > 0.0) {
            cossim = @floatCast(agg.sum_dot_prod / (sqrt(agg.sum_norm1_sq) * sqrt(agg.sum_norm2_sq)));
            cossim = cClamp(cossim, -1.0, 1.0);
        } else if (agg.sum_norm1_sq == 0.0 and agg.sum_norm2_sq == 0.0) {
            cossim = fnan;
        }

        var gain: f32 = fnan;
        if (agg.sum_elements_curr > 0 and agg.sum_elements_prev > 0 and agg.sum_energy_prev > 0.0) {
            const rms_curr = sqrt(agg.sum_energy_curr / @as(f64, @floatFromInt(agg.sum_elements_curr)));
            const rms_prev = sqrt(agg.sum_energy_prev / @as(f64, @floatFromInt(agg.sum_elements_prev)));
            gain = @floatCast(rms_curr / rms_prev);
        }

        try out.cossim.put(layer, cossim);
        try out.l2_dist.put(layer, @floatCast(sqrt(agg.sum_l2_dist_sq)));
        try out.gain.put(layer, gain);

        if (agg.sum_n_features > 0) {
            try out.covariance.put(layer, @floatCast(agg.sum_cov / @as(f64, @floatFromInt(agg.sum_n_features))));
        } else {
            try out.covariance.put(layer, fnan);
        }

        if (agg.sum_var_c > 0.0 and agg.sum_var_p > 0.0) {
            const pearson: f32 = @floatCast(agg.sum_cov / (sqrt(agg.sum_var_c) * sqrt(agg.sum_var_p)));
            try out.pearson.put(layer, cClamp(pearson, -1.0, 1.0));
        } else if (agg.sum_var_c == 0.0 and agg.sum_var_p == 0.0) {
            try out.pearson.put(layer, fnan);
        } else {
            try out.pearson.put(layer, 0.0);
        }
    }
}

/// Ports `all_finite` (imatrix.cpp:548 @620d2225c).
fn allFinite(v: []const f32) bool {
    for (v) |x| {
        if (!std.math.isFinite(x)) return false;
    }
    return true;
}

/// Ports `rows_to_chunks` (imatrix.cpp:557 @620d2225c): round to nearest
/// instead of truncating.
fn rowsToChunks(n_rows: i64, chunk_size: i32) i32 {
    return @truncate(@divTrunc(n_rows + @divTrunc(chunk_size, 2), chunk_size));
}

var g_collector: Collector = undefined;

/// Ports `ik_collect_imatrix` (imatrix.cpp:1162 @620d2225c).
fn ikCollectImatrix(t: [*c]c.ggml_tensor, ask: bool, _: ?*anyopaque) callconv(.c) bool {
    return g_collector.collect(t, ask) catch false;
}

/// Ports `results_log_softmax` and `log_softmax` (imatrix.cpp:1166, 1192 @620d2225c).
const LogSoftmax = struct { log_softmax: f64, logit: f32, prob: f32 };

fn logSoftmax(n_vocab: usize, logits: [*]const f32, tok: usize) LogSoftmax {
    var max_logit = logits[0];
    for (1..n_vocab) |i| max_logit = @max(max_logit, logits[i]);
    var sum_exp: f64 = 0.0;
    for (0..n_vocab) |i| sum_exp += expf(logits[i] - max_logit);
    return .{
        .log_softmax = @as(f64, logits[tok] - max_logit) - logD(sum_exp),
        .logit = logits[tok],
        .prob = expf(logits[tok] - max_logit) / @as(f32, @floatCast(sum_exp)),
    };
}

/// What `process_logits`'s threads share.
const LogitsWork = struct {
    n_vocab: usize,
    logits: [*]const f32,
    tokens: [*]const c.llama_token,
    n_token: usize,
    logit_history: [*]f32,
    prob_history: [*]f32,
    counter: std.atomic.Value(usize) = .init(0),

    fn compute(self: *LogitsWork, sums: *[2]f64) void {
        var local_nll: f64 = 0;
        var local_nll2: f64 = 0;
        while (true) {
            const i = self.counter.fetchAdd(1, .seq_cst);
            if (i >= self.n_token) break;
            const results = logSoftmax(self.n_vocab, self.logits + i * self.n_vocab, @intCast(self.tokens[i + 1]));
            const v = -results.log_softmax;
            local_nll += v;
            local_nll2 = @mulAdd(f64, v, v, local_nll2);
            self.logit_history[i] = results.logit;
            self.prob_history[i] = results.prob;
        }
        sums.* = .{ local_nll, local_nll2 };
    }
};

/// Ports `process_logits` (imatrix.cpp:1204 @620d2225c).
fn processLogits(n_vocab: usize, logits: [*]const f32, tokens: [*]const c.llama_token, n_token: usize, n_workers: usize, nll: *f64, nll2: *f64, logit_history: [*]f32, prob_history: [*]f32) !void {
    var work: LogitsWork = .{ .n_vocab = n_vocab, .logits = logits, .tokens = tokens, .n_token = n_token, .logit_history = logit_history, .prob_history = prob_history };
    const sums = try gpa.alloc([2]f64, n_workers + 1);
    defer gpa.free(sums);
    const threads = try gpa.alloc(std.Thread, n_workers);
    defer gpa.free(threads);
    for (threads, 0..) |*w, k| w.* = try std.Thread.spawn(.{}, LogitsWork.compute, .{ &work, &sums[k] });
    work.compute(&sums[n_workers]);
    for (threads) |w| w.join();
    for (sums) |s| {
        nll.* += s[0];
        nll2.* += s[1];
    }
}

/// Ports `nextn_collector` (imatrix.cpp:1238 @620d2225c): the MTP context
/// the trunk's NextN hidden states are decoded through, so the NextN
/// layers' matrix multiplications reach the collector.
const NextnCollector = struct {
    ctx: ?*c.llama_context,
    batch: tc.common.Batch,
    n_embd: i32 = 0,
    own_lm_head: bool = false,
    pending_h: std.ArrayList(f32) = .empty,

    /// Ports `clear_memory` (imatrix.cpp:1244 @620d2225c), a member of `nextn_collector`.
    fn clearMemory(self: *NextnCollector) void {
        c.llama_memory_clear(c.llama_get_memory(self.ctx), true);
        self.pending_h.clearRetainingCapacity(); // a chunk's last h-row has no next token, the trunk restarts at position 0
    }

    /// Ports `decode` (imatrix.cpp:1387 @620d2225c), a member of
    /// `nextn_collector`: each trunk row's NextN hidden state goes in with
    /// the token after it, the last held back for the next batch.
    fn decode(self: *NextnCollector, ctx_trunk: ?*c.llama_context, batch_trunk: *const tc.common.Batch) !bool {
        const func = "decode";
        const n_embd: usize = @intCast(self.n_embd);
        const n_last = batch_trunk.size() - 1;
        self.batch.clear();
        if (self.pending_h.items.len != 0) {
            const t0 = batch_trunk.tokens.items[0];
            const idx = self.batch.add(t0.id, t0.pos[0], 0, self.own_lm_head);
            _ = self.batch.setEmbd(idx, .{ .data = self.pending_h.items.ptr, .n_rows = 1, .n_embd = n_embd });
        }

        var h_last: ?[*]const f32 = null;
        var i: i32 = 0;
        while (i <= n_last) : (i += 1) {
            const h = tc.speculative.ext.getEmbeddingsNextnIth(ctx_trunk, i) orelse {
                log.err("%s: no NextN hidden state for row %d\n", .{ func, i });

                return false;
            };

            if (i == n_last) {
                h_last = h;
                break;
            }

            const tn = batch_trunk.tokens.items[@intCast(i + 1)];
            const idx = self.batch.add(tn.id, tn.pos[0], 0, self.own_lm_head);
            _ = self.batch.setEmbd(idx, .{ .data = h, .n_rows = 1, .n_embd = n_embd });
        }

        if (self.batch.size() > 0) {
            if (c.llama_process(self.ctx, c.LLAMA_PROCESS_TYPE_DECODE, self.batch.get()) != 0) {
                log.err("%s: failed to decode the NextN layer\n", .{func});

                return false;
            }
        }

        if (h_last) |h| {
            self.pending_h.clearRetainingCapacity();
            try self.pending_h.appendSlice(gpa, h[0..n_embd]);
        }

        return true;
    }

    fn deinit(self: *NextnCollector) void {
        self.batch.deinit();
        c.llama_free(self.ctx);
        self.pending_h.deinit(gpa);
    }
};

/// Ports `nextn_model_info` (imatrix.cpp:1251 @620d2225c).
const NextnModelInfo = struct {
    has_layers: bool = false,
    own_lm_head: []bool = &.{},
};

/// Ports `nextn_read_model_info` (imatrix.cpp:1256 @620d2225c).
fn nextnReadModelInfo(a: std.mem.Allocator, model_path: []const u8, n_layer: i32, n_heads: i32) !NextnModelInfo {
    const func = "nextn_read_model_info";
    var info: NextnModelInfo = .{};
    info.own_lm_head = try a.alloc(bool, @intCast(n_heads));
    @memset(info.own_lm_head, false);

    const path = try a.dupeSentinel(u8, model_path, 0);
    const gguf_params: c.gguf_init_params = .{ .no_alloc = true, .ctx = null };
    const ctx_gguf = c.gguf_init_from_file(path.ptr, gguf_params);
    if (ctx_gguf == null) {
        log.wrn("%s: cannot read '%s'\n", .{ func, path.ptr });
        return info;
    }

    const n_tensors = c.gguf_get_n_tensors(ctx_gguf);

    var head: i32 = 0;
    while (head < n_heads) : (head += 1) {
        const prefix = try std.fmt.allocPrint(a, "blk.{d}.nextn.", .{n_layer + head});
        const lm_head = try std.fmt.allocPrint(a, "{s}shared_head_head.weight", .{prefix});
        var i: i64 = 0;
        while (i < n_tensors) : (i += 1) {
            const name = std.mem.span(c.gguf_get_tensor_name(ctx_gguf, i));
            if (!std.mem.startsWith(u8, name, prefix)) continue;

            info.has_layers = true;
            if (std.mem.eql(u8, name, lm_head)) info.own_lm_head[@intCast(head)] = true;
        }
    }

    c.gguf_free(ctx_gguf);

    return info;
}

/// Ports `model_file_shape` (imatrix.cpp:1286 @620d2225c).
const ModelFileShape = struct {
    valid: bool = false,
    is_split: bool = false,
    arch: []const u8 = "",
    n_layer_all: u32 = 0,
    n_embd_out: u32 = 0,
    /// `std::set<uint32_t>`, read by `count` alone.
    nextn_idx: std.ArrayList(u32) = .empty,
    has_trunk: bool = false,
    has_nextn: bool = false,
    n_nextn_layers: u32 = 0,
    first_nextn: []const u8 = "",

    /// Ports `n_trunk` (imatrix.cpp:1298 @620d2225c), a member of `model_file_shape`.
    fn nTrunk(self: *const ModelFileShape) u32 {
        return if (self.n_layer_all >= self.n_nextn_layers) self.n_layer_all - self.n_nextn_layers else 0;
    }

    fn hasNextnIdx(self: *const ModelFileShape, idx: u32) bool {
        return std.mem.indexOfScalar(u32, self.nextn_idx.items, idx) != null;
    }
};

/// Ports `model_read_file_shape` (imatrix.cpp:1301 @620d2225c).
fn modelReadFileShape(a: std.mem.Allocator, model_path: []const u8) !ModelFileShape {
    var shape: ModelFileShape = .{};

    const path = try a.dupeSentinel(u8, model_path, 0);
    const gguf_params: c.gguf_init_params = .{ .no_alloc = true, .ctx = null };
    const ctx_gguf = c.gguf_init_from_file(path.ptr, gguf_params);
    if (ctx_gguf == null) return shape;
    shape.valid = true;

    const arch_key = c.gguf_find_key(ctx_gguf, "general.architecture");
    if (arch_key >= 0 and c.gguf_get_kv_type(ctx_gguf, arch_key) == c.GGUF_TYPE_STRING) {
        shape.arch = try a.dupe(u8, std.mem.span(c.gguf_get_val_str(ctx_gguf, arch_key)));

        const block_key = c.gguf_find_key(ctx_gguf, (try std.fmt.allocPrintSentinel(a, "{s}.block_count", .{shape.arch}, 0)).ptr);
        if (block_key >= 0 and c.gguf_get_kv_type(ctx_gguf, block_key) == c.GGUF_TYPE_UINT32) {
            shape.n_layer_all = c.gguf_get_val_u32(ctx_gguf, block_key);
        }

        const nextn_key = c.gguf_find_key(ctx_gguf, (try std.fmt.allocPrintSentinel(a, "{s}.nextn_predict_layers", .{shape.arch}, 0)).ptr);
        if (nextn_key >= 0 and c.gguf_get_kv_type(ctx_gguf, nextn_key) == c.GGUF_TYPE_UINT32) {
            shape.n_nextn_layers = c.gguf_get_val_u32(ctx_gguf, nextn_key);
        }

        const embd_out_key = c.gguf_find_key(ctx_gguf, (try std.fmt.allocPrintSentinel(a, "{s}.embedding_length_out", .{shape.arch}, 0)).ptr);
        const embd_key = c.gguf_find_key(ctx_gguf, (try std.fmt.allocPrintSentinel(a, "{s}.embedding_length", .{shape.arch}, 0)).ptr);
        if (embd_out_key >= 0 and c.gguf_get_kv_type(ctx_gguf, embd_out_key) == c.GGUF_TYPE_UINT32) {
            shape.n_embd_out = c.gguf_get_val_u32(ctx_gguf, embd_out_key);
        } else if (embd_key >= 0 and c.gguf_get_kv_type(ctx_gguf, embd_key) == c.GGUF_TYPE_UINT32) {
            shape.n_embd_out = c.gguf_get_val_u32(ctx_gguf, embd_key);
        }
    }

    const split_key = c.gguf_find_key(ctx_gguf, "split.count");
    if (split_key >= 0 and c.gguf_get_kv_type(ctx_gguf, split_key) == c.GGUF_TYPE_UINT16) {
        shape.is_split = c.gguf_get_val_u16(ctx_gguf, split_key) > 1;
    }

    const n_tensors = c.gguf_get_n_tensors(ctx_gguf);
    var i: i64 = 0;
    while (i < n_tensors) : (i += 1) {
        const name = std.mem.span(c.gguf_get_tensor_name(ctx_gguf, i));
        if (std.mem.indexOf(u8, name, ".nextn.") != null) {
            shape.has_nextn = true;
            if (shape.first_nextn.len == 0) shape.first_nextn = try a.dupe(u8, name);

            if (std.mem.startsWith(u8, name, "blk.")) {
                if (std.mem.indexOfScalarPos(u8, name, 4, '.')) |dot| {
                    if (std.mem.startsWith(u8, name[dot + 1 ..], "nextn.")) {
                        var idx: u32 = 0;
                        var ok = dot > 4;
                        var j: usize = 4;
                        while (j < dot and ok) : (j += 1) {
                            ok = name[j] >= '0' and name[j] <= '9';
                            idx = 10 *% idx +% (name[j] -% '0');
                        }

                        if (ok and !shape.hasNextnIdx(idx)) try shape.nextn_idx.append(a, idx);
                    }
                }
            }
        } else if (std.mem.startsWith(u8, name, "blk.0.")) {
            shape.has_trunk = true;
        }
    }

    c.gguf_free(ctx_gguf);

    return shape;
}

/// Ports `nextn_collector_init` (imatrix.cpp:1368 @620d2225c).
fn nextnCollectorInit(model: ?*c.llama_model, params: *const Params, own_lm_head: bool) ?NextnCollector {
    const func = "nextn_collector_init";
    var cparams = tc.common.contextParamsToLlama(params);
    cparams.ctx_type = c.LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq = 0;
    const ctx = c.llama_init_from_model(model, cparams);
    if (ctx == null) {
        log.err("%s: failed to create the NextN context\n", .{func});
        return null;
    }

    return .{
        .ctx = ctx,
        .batch = tc.common.Batch.init(gpa, ctx),
        .n_embd = c.llama_model_n_embd_out(model),
        .own_lm_head = own_lm_head,
    };
}

fn nowNs() i128 {
    return @import("tools_common").clock.monotonicNs();
}

/// Ports `compute_imatrix` (imatrix.cpp:1423 @620d2225c).
fn computeImatrix(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params, n_ctx: i32, nextn: ?*NextnCollector) !bool {
    const func = "compute_imatrix";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    const add_bos = c.llama_vocab_get_add_bos(vocab);

    if (c.llama_pooling_type(ctx) != c.LLAMA_POOLING_TYPE_LAST) {
        ggmlAssert(!c.llama_vocab_get_add_eos(vocab), 1430, "!llama_vocab_get_add_eos(vocab)");
    }

    const tim1 = nowNs();
    log.inf("%s: tokenizing the input ..\n", .{func});

    var tokens = try tc.common.tokenize(a, vocab, params.prompt, true, params.parse_special);

    const tim2 = nowNs();
    log.inf("%s: tokenization took %g ms\n", .{ func, 1e-3 * @as(f64, @floatFromInt(@divTrunc(tim2 - tim1, 1000))) });

    if (params.i_chunk > 0) {
        if (@as(usize, @intCast((params.i_chunk + 2) * n_ctx)) >= tokens.len) {
            log.err("%s: there will be not enough tokens left after removing %d chunks\n", .{ func, params.i_chunk });
            return false;
        }
        log.inf("%s: removing initial %d chunks (%d tokens)\n", .{ func, params.i_chunk, params.i_chunk * n_ctx });
        tokens = tokens[@intCast(params.i_chunk * n_ctx)..];
    }

    if (@as(c_int, @intCast(tokens.len)) < 2 * n_ctx) {
        log.err("%s: you need at least %d tokens for a context of %d tokens\n", .{ func, 2 * n_ctx, n_ctx });
        log.err("%s: the data file you provided tokenizes to only %zu tokens\n", .{ func, tokens.len });
        return false;
    }

    var logit_history: []f32 = &.{};
    var prob_history: []f32 = &.{};

    if (params.compute_ppl) {
        logit_history = try a.alloc(f32, tokens.len);
        @memset(logit_history, 0);
        prob_history = try a.alloc(f32, tokens.len);
        @memset(prob_history, 0);
    }

    const n_chunk_max: c_int = @intCast(tokens.len / @as(usize, @intCast(n_ctx)));

    const n_chunk = if (params.n_chunks < 0) n_chunk_max else @min(params.n_chunks, n_chunk_max);
    const n_vocab: c_int = c.llama_vocab_n_tokens(vocab);
    const n_batch: c_int = params.n_batch;

    var count: c_int = 0;
    var nll: f64 = 0.0;
    var nll2: f64 = 0.0;

    const num_batches = @divTrunc(n_ctx + n_batch - 1, n_batch);
    const n_seq: c_int = @max(1, @divTrunc(n_batch, n_ctx));

    ggmlAssert(n_batch < n_ctx or @rem(n_batch, n_ctx) == 0, 1477, "n_batch < n_ctx || n_batch % n_ctx == 0");
    ggmlAssert(params.n_ctx == n_seq * n_ctx, 1478, "params.n_ctx == n_seq * n_ctx");

    var batch = tc.common.Batch.init(gpa, ctx);
    defer batch.deinit();

    var logits: std.ArrayList(f32) = .empty;
    defer logits.deinit(gpa);
    if (params.compute_ppl and num_batches > 1) try logits.ensureTotalCapacity(gpa, @as(usize, @intCast(n_ctx)) * @as(usize, @intCast(n_vocab)));

    log.inf("%s: computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", .{ func, n_chunk, n_ctx, n_batch, n_seq });

    const n_workers: usize = @as(usize, @intCast(std.Thread.getCpuCount() catch 1)) - 1;

    const nv: usize = @intCast(n_vocab);
    var i: c_int = 0;
    while (i < n_chunk) : (i += n_seq) {
        const start = i * n_ctx;
        const end = start + n_ctx;

        const n_seq_batch = @min(n_seq, n_chunk - i);

        const t_start = nowNs();

        // clear the KV cache
        c.llama_memory_clear(c.llama_get_memory(ctx), true);
        if (nextn) |n| n.clearMemory();

        var j: c_int = 0;
        while (j < num_batches) : (j += 1) {
            const batch_start = start + j * n_batch;
            const batch_size = @min(end - batch_start, n_batch);

            // clear the batch
            batch.clear();

            var seq: c_int = 0;
            while (seq < n_seq_batch) : (seq += 1) {
                const seq_start: usize = @intCast(batch_start + seq * n_ctx);

                // save original token and restore it after eval
                const token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos and j == 0) tokens[seq_start] = c.llama_vocab_bos(vocab);
                var k: c_int = 0;
                while (k < batch_size) : (k += 1) {
                    // NOTE: specifying all logits to get activations for the output.weight tensor
                    //       and also for the perplexity calculation.
                    _ = batch.add(tokens[seq_start + @as(usize, @intCast(k))], j * n_batch + k, seq, true);
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (c.llama_process(ctx, c.LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                log.err("%s : failed to eval\n", .{func});

                return false;
            }

            if (nextn) |n| {
                if (!try n.decode(ctx, &batch)) return false;
            }

            if (params.compute_ppl and num_batches > 1) {
                const batch_logits = c.llama_get_logits(ctx);
                try logits.appendSlice(gpa, batch_logits[0 .. @as(usize, @intCast(batch_size)) * nv]);
            }
        }

        if (i == 0) {
            c.llama_synchronize(ctx);
            const t_end = nowNs();
            const t_total: f32 = @floatCast(@as(f64, @floatFromInt(t_end - t_start)) / 1e9);
            log.inf("%s: %.2f seconds per pass - ETA ", .{ func, @as(f64, t_total) });
            var total_seconds: c_int = @intFromFloat(t_total * @as(f32, @floatFromInt(n_chunk)) / @as(f32, @floatFromInt(n_seq)));
            if (total_seconds >= 60 * 60) {
                log.out("%d hours ", .{@divTrunc(total_seconds, 60 * 60)});
                total_seconds = @rem(total_seconds, 60 * 60);
            }
            log.out("%.2f minutes\n", .{@as(f64, @floatFromInt(total_seconds)) / 60.0});
        }

        if (params.compute_ppl) {
            const first = @divTrunc(n_ctx, 2);
            var seq: c_int = 0;
            while (seq < n_seq_batch) : (seq += 1) {
                const all_logits: [*]const f32 = if (num_batches > 1) logits.items.ptr else c.llama_get_logits_ith(ctx, seq * n_ctx);

                const base: usize = @intCast(start + seq * n_ctx + first);
                try processLogits(nv, all_logits + @as(usize, @intCast(first)) * nv, tokens.ptr + base, @intCast(n_ctx - 1 - first), n_workers, &nll, &nll2, logit_history.ptr + base, prob_history.ptr + base);

                count += n_ctx - first - 1;

                log.out("[%d]%.4lf,", .{ i + seq + 1, exp(nll / @as(f64, @floatFromInt(count))) });
            }
            io.flushStdout();

            logits.clearRetainingCapacity();
        }
    }

    log.out("\n", .{});

    if (params.compute_ppl) {
        nll2 /= @floatFromInt(count);
        nll /= @floatFromInt(count);
        const ppl = exp(nll);
        nll2 = @mulAdd(f64, -nll, nll, nll2);
        if (nll2 > 0) {
            nll2 = sqrt(nll2 / @as(f64, @floatFromInt(count - 1)));
            log.out("Final estimate: PPL = %.4lf +/- %.5lf\n", .{ ppl, nll2 * ppl });
        } else {
            log.out("Unexpected negative standard deviation of log(prob)\n", .{});
        }
    }

    return true;
}

/// Ports `show_statistics`'s `tensor_comparer` (imatrix.cpp:1633 @620d2225c).
fn tensorLess(_: void, a: TensorStatistics, b: TensorStatistics) bool {
    var buf_a: std.ArrayList(u8) = .empty;
    defer buf_a.deinit(gpa);
    var buf_b: std.ArrayList(u8) = .empty;
    defer buf_b.deinit(gpa);
    var lay_a: []const u8 = "";
    var lay_b: []const u8 = "";
    var name_a: []const u8 = "";
    var name_b: []const u8 = "";
    processTensorName(a.tensor, &lay_a, &name_a, &buf_a);
    processTensorName(b.tensor, &lay_b, &name_b, &buf_b);

    // Handle non-numeric layers (e.g., "output")
    const blk_a: c_int = stoiOr(lay_a) orelse if (std.mem.indexOf(u8, a.tensor, "output") != null) int_max else int_max - 1;
    const blk_b: c_int = stoiOr(lay_b) orelse if (std.mem.indexOf(u8, b.tensor, "output") != null) int_max else int_max - 1;

    if (blk_a != blk_b) return blk_a < blk_b;
    return std.mem.order(u8, name_a, name_b) == .lt;
}

/// Mirrors `show_statistics`'s `layer_stats` (imatrix.cpp:1662 @620d2225c).
const LayerStats = struct {
    layer_sum: f32 = 0.0,
    n: c_int = 0,
};

/// `label_fmt` in `show_statistics`: shorten names for table formatting.
fn labelFmt(a: std.mem.Allocator, s: []const u8, w: usize) ![:0]const u8 {
    if (s.len <= w) return a.dupeSentinel(u8, s, 0);
    return std.fmt.allocPrintSentinel(a, "..{s}", .{s[s.len - (w - 2) ..]}, 0);
}

/// Ports `show_statistics` (imatrix.cpp:1604 @620d2225c).
fn showStatistics(a: std.mem.Allocator, params: *const Params) !bool {
    const func = "show_statistics";
    g_collector.params = .of(params);
    var ts: std.ArrayList(TensorStatistics) = .empty;

    if (params.in_files.items.len == 0) return false;

    // Load and process data
    const in0 = try a.dupeSentinel(u8, params.in_files.items[0], 0);
    if (try g_collector.load(a, in0)) {
        try ts.ensureTotalCapacity(gpa, g_collector.stats.count());
        var it = g_collector.stats.iterator();
        while (it.next()) |kv| _ = try computeVectorStatistics(&ts, kv.key, kv.value);
    } else {
        return false;
    }

    if (ts.items.len == 0) return false;

    var n_legacy: usize = 0;
    for (ts.items) |t| {
        if (t.legacy) n_legacy += 1;
    }
    if (n_legacy > 0 and n_legacy < ts.items.len) {
        log.wrn("%s: %zu of %zu tensors have no activation data, using the legacy layout\n", .{ func, n_legacy, ts.items.len });
    }

    const legacy = n_legacy > 0;
    const nextn_start = nextnLayerStart(ts.items, g_collector.n_layer_nextn);
    try computeTensorStatistics(ts.items, &g_collector.stats, nextn_start);

    // Sorting logic (Layer index -> Tensor Name)
    libcxx.sort(TensorStatistics, ts.items, {}, tensorLess);

    // `std::map<int, layer_stats>`: kept sorted by key
    var ls_keys: std.ArrayList(c_int) = .empty;
    var ls_vals: std.ArrayList(LayerStats) = .empty;

    const w_lay: c_int = 6;
    const w_nam: c_int = 40; // Should be wide enough for most tensor names
    const sep = " | ";

    io.printf("\nComputing tensor statistics for %s (%d tensors)\n", .{ in0.ptr, @as(c_int, @intCast(ts.items.len)) });

    if (legacy) {
        io.printf("\n%*s%s%-*s%s%10s%10s%12s%12s%9s%s%17s%8s%s%10s%10s\n", .{
            w_lay,  "Layer",  sep,
            w_nam,  "Tensor", sep,
            "Mean", "StdDev", "Skew",
            "Kurt", "H Norm", sep,
            "∑ E[A²]",
            "Gain", sep,      "PCC",
            "Cov",
        });
        io.printf("%s\n", .{(try dashes(a, 153)).ptr});
    } else {
        io.printf("\n%*s%s%-*s%s%10s%10s%12s%12s%9s%s%17s%8s%s%12s%10s%10s\n", .{
            w_lay,  "Layer",  sep,
            w_nam,  "Tensor", sep,
            "Mean", "StdDev", "Skew",
            "Kurt", "H Norm", sep,
            "∑ E[A²]",
            "Gain", sep,      "L2 Dist",
            "PCC",  "Cov",
        });
        io.printf("%s\n", .{(try dashes(a, 165)).ptr});
    }

    // Tensor Statistics
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    for (ts.items) |tstat| {
        var layer: []const u8 = "";
        var name: []const u8 = "";

        processTensorName(tstat.tensor, &layer, &name, &buf);
        const h_norm: f32 = if (tstat.elements > 1) 100.0 * (tstat.entropy / log2f(@floatFromInt(tstat.elements))) else fnan;

        const blk: c_int = stoiOr(layer) orelse if (std.mem.indexOf(u8, tstat.tensor, "output") != null) int_max else -1;

        var lbuf: [32]u8 = undefined;
        const label = layerLabel(&lbuf, blk, nextn_start);
        const tname = try labelFmt(a, tstat.tensor, @intCast(w_nam));
        if (legacy) {
            io.printf("%*s%s%-*s%s%10.4f%10.4f%12.4f%12.4f%8.2f%%%s%14.4f%8.2f%s%10.4f%10.4f\n", .{
                w_lay,                    label.ptr,                     sep,
                w_nam,                    tname.ptr,                     sep,
                @as(f64, tstat.mean),     @as(f64, tstat.std_deviation), @as(f64, tstat.skewness),
                @as(f64, tstat.kurtosis), @as(f64, h_norm),              sep,
                tstat.sum,                @as(f64, tstat.gain),          sep,
                @as(f64, tstat.pearson),  @as(f64, tstat.covariance),
            });
        } else {
            io.printf("%*s%s%-*s%s%10.4f%10.4f%12.4f%12.4f%8.2f%%%s%14.4f%8.2f%s%12.4f%10.4f%10.4f\n", .{
                w_lay,                    label.ptr,                     sep,
                w_nam,                    tname.ptr,                     sep,
                @as(f64, tstat.mean),     @as(f64, tstat.std_deviation), @as(f64, tstat.skewness),
                @as(f64, tstat.kurtosis), @as(f64, h_norm),              sep,
                tstat.sum,                @as(f64, tstat.gain),          sep,
                @as(f64, tstat.l2_dist),  @as(f64, tstat.pearson),       @as(f64, tstat.covariance),
            });
        }

        // Aggregate Layer Stats
        var pos: usize = 0;
        while (pos < ls_keys.items.len and ls_keys.items[pos] < blk) pos += 1;
        if (pos == ls_keys.items.len or ls_keys.items[pos] != blk) {
            try ls_keys.insert(a, pos, blk);
            try ls_vals.insert(a, pos, .{});
        }
        const l = &ls_vals.items[pos];
        l.layer_sum = @floatCast(@as(f64, l.layer_sum) + tstat.sum);
        l.n = @truncate(@as(i64, l.n) + tstat.elements);
    }

    // Layer Statistics
    var lstats: LayerStatistics = .{};
    defer lstats.deinit();
    try computeLayerStatistics(ts.items, &lstats);

    var layers: usize = 0;
    var trunk_layers: usize = 0;
    var min: c_int = int_max;
    var max: c_int = -1;

    for (ls_keys.items, ls_vals.items) |layer, stats| {
        if (layer >= 0 and layer < 9999 and stats.n > 0) {
            layers += 1;
            if (layer < nextn_start) {
                trunk_layers += 1;
                min = @min(layer, min);
                max = @max(layer, max);
            }
        }
    }

    if (trunk_layers > 0) {
        const expected: usize = @intCast(max - min + 1);
        if (trunk_layers != expected) {
            log.wrn("\n%s: layer sequence gap detected (found %zu layers in range %d-%d, expected %zu); layer statistics will not be shown\n", .{ func, trunk_layers, min, max, expected });
            return false;
        }
    }

    io.printf("\nComputing layer statistics for %s (%zu layers)\n\n", .{ in0.ptr, layers });

    if (legacy) {
        io.printf("%*s%s%17s%8s%s%9s%9s%12s\n", .{ w_lay, "Layer", sep, "∑ E[A²]", "Gain", sep, "CosSim", "PCC", "Cov" });
        io.printf("%s\n", .{(try dashes(a, 64)).ptr});
    } else {
        io.printf("%*s%s%17s%8s%s%12s%9s%9s%12s\n", .{ w_lay, "Layer", sep, "∑ E[A²]", "Gain", sep, "L2 Dist", "CosSim", "PCC", "Cov" });
        io.printf("%s\n", .{(try dashes(a, 76)).ptr});
    }

    for (ls_keys.items, ls_vals.items) |layer, stats| {
        if (layer < 0 or stats.n == 0) continue;

        const edge = layer == 0 or layer == int_max;
        const lgn: f32 = if (edge) fnan else lstats.gain.get(layer) orelse fnan;
        const ll2: f32 = if (edge) fnan else lstats.l2_dist.get(layer) orelse fnan;
        const lcs: f32 = if (edge) fnan else lstats.cossim.get(layer) orelse fnan;
        const lpc: f32 = if (edge) fnan else lstats.pearson.get(layer) orelse fnan;
        const lcv: f32 = if (edge) fnan else lstats.covariance.get(layer) orelse fnan;
        var lbuf: [32]u8 = undefined;
        const lyr = layerLabel(&lbuf, layer, nextn_start);

        if (legacy) {
            io.printf("%*s%s%14.4f%8.2f%s%9.4f%9.4f%12.4f\n", .{ w_lay, lyr.ptr, sep, @as(f64, stats.layer_sum), @as(f64, lgn), sep, @as(f64, lcs), @as(f64, lpc), @as(f64, lcv) });
        } else {
            io.printf("%*s%s%14.4f%8.2f%s%12.4f%9.4f%9.4f%12.4f\n", .{ w_lay, lyr.ptr, sep, @as(f64, stats.layer_sum), @as(f64, lgn), sep, @as(f64, ll2), @as(f64, lcs), @as(f64, lpc), @as(f64, lcv) });
        }
    }

    io.printf("\n", .{});
    return true;
}

/// `std::string(n, '-')`.
fn dashes(a: std.mem.Allocator, n: usize) ![:0]u8 {
    const s = try a.allocSentinel(u8, n, 0);
    @memset(s, '-');
    return s;
}

/// Ports `main` (imatrix.cpp:1820 @620d2225c).
fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "main";
    var params = try Params.init(a);

    params.out_file = "imatrix.gguf";

    params.n_ctx = 512;
    params.escape = false;

    tc.common.init();

    if (!try arg.parse(argv, &params, .imatrix, printUsage)) return 1;

    const use_draft = params.speculative.draft.mparams.path.len != 0;
    if (use_draft) {
        params.load_mtp = true;
        params.speculative.draft.n_max = 0; // keep trunk's context identical to a plain run
    }

    g_collector = .{ .stats = try libcxx.StringMap(Stats).init(gpa) };

    // set_params before show_statistics so load_imatrix has valid n_ctx/n_parallel
    g_collector.params = .of(&params);

    if (params.show_statistics) {
        if (!try showStatistics(a, &params)) return 1;
        return 0;
    }

    const n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        log.err("%s: imatrix tool requires '--ctx-size' > 0\n", .{func});
        return 1;
    }

    {
        const n_seq: i32 = @max(1, @divTrunc(params.n_batch, n_ctx));
        const n_kv = n_seq * n_ctx;

        if (params.load_mtp and n_seq > 1) {
            log.err("%s: '--nextn' and '--model-draft' need a single sequence per batch, set '--batch-size' to at most '--ctx-size' (%d)\n", .{ func, n_ctx });
            return 1;
        }

        params.n_parallel = n_seq;
        params.n_ctx = n_kv;

        params.n_batch = @min(params.n_batch, n_kv);
    }

    g_collector.params = .of(&params);

    for (params.in_files.items) |in_file| {
        const z = try a.dupeSentinel(u8, in_file, 0);
        log.inf("%s : loading imatrix from '%s'\n", .{ func, z.ptr });
        if (!try g_collector.load(a, z)) {
            log.err("%s : failed to load %s\n", .{ func, z.ptr });
            return 1;
        }
    }

    const out_file = try a.dupeSentinel(u8, params.out_file, 0);

    if (params.prompt.len == 0) {
        log.inf("No prompt provided; combining precomputed matrices only.\n", .{});

        if (params.in_files.items.len == 0) {
            log.err("Error: No prompt provided and no precomputed matrices (--in-file) to combine.\n", .{});
            return 1;
        }

        if (params.in_files.items.len == 1) {
            log.inf("%s : saving imatrix to '%s'\n", .{ func, out_file.ptr });
        } else if (params.in_files.items.len > 1) {
            log.inf("%s : saving combined imatrix to '%s'\n", .{ func, out_file.ptr });
        }

        try g_collector.save(-1);

        return 0;
    }

    {
        const shape = try modelReadFileShape(a, params.model.path);
        if (use_draft and shape.n_nextn_layers > 0) {
            log.err("%s: model already includes NextN layers\n", .{func});
            return 1;
        }
        if (use_draft and shape.has_nextn) {
            log.err("%s: the model already includes NextN layers (NextN tensor '%s')\n", .{ func, (try a.dupeSentinel(u8, shape.first_nextn, 0)).ptr });
            return 1;
        }
        if (!use_draft and !shape.has_trunk and (shape.n_nextn_layers > 0 or shape.has_nextn)) {
            log.err("%s: '%s' is a NextN draft, use with --model-draft/-md\n", .{ func, (try a.dupeSentinel(u8, params.model.path, 0)).ptr });
            return 1;
        }
        if (use_draft) {
            const path_md = try a.dupeSentinel(u8, params.speculative.draft.mparams.path, 0);
            const shape_md = try modelReadFileShape(a, path_md);
            if (!shape_md.valid or shape_md.is_split) {
                log.err("%s: '%s' is not a readable single-shard GGUF\n", .{ func, path_md.ptr });
                return 1;
            }
            if (shape.valid and !std.mem.eql(u8, shape_md.arch, shape.arch)) {
                log.err("%s: the draft architecture '%s' does not match the target architecture '%s'\n", .{ func, (try a.dupeSentinel(u8, shape_md.arch, 0)).ptr, (try a.dupeSentinel(u8, shape.arch, 0)).ptr });
                return 1;
            }
            if (shape_md.n_nextn_layers == 0 or shape_md.has_trunk) {
                log.err("%s: '%s' is not a draft (no NextN layers or trunk tensors present)\n", .{ func, path_md.ptr });
                return 1;
            }
            if (shape_md.n_nextn_layers > 1) {
                log.err("%s: multi-layer NextN drafts are not supported (found %u)\n", .{ func, shape_md.n_nextn_layers });
                return 1;
            }
            if (shape_md.nTrunk() == 0) {
                log.err("%s: NextN layers sharing the trunk's KV cache are not supported\n", .{func});
                return 1;
            }
            if (shape.valid and !shape_md.hasNextnIdx(shape.nTrunk())) {
                log.err("%s: no NextN tensors at layer index %u in '%s' found\n", .{ func, shape.nTrunk(), path_md.ptr });
                return 1;
            }
            if (shape.valid and shape_md.n_embd_out != shape.n_embd_out) {
                log.err("%s: the draft output width %u does not match the target's %u\n", .{ func, shape_md.n_embd_out, shape.n_embd_out });
                return 1;
            }
            for (params.speculative.types.items) |ty| {
                if (ty != .none and ty != .draft_mtp) {
                    log.err("%s: speculative type '%s' is not supported with --model-draft/-md (only 'draft-mtp')\n", .{ func, (try a.dupeSentinel(u8, ty.toStr(), 0)).ptr });
                    return 1;
                }
            }
        }
    }

    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ikCollectImatrix;
    params.cb_eval_user_data = null;
    params.warmup = false;

    // init
    var llama_init = try tc.init.initFromParams(init.gpa, &params, false);
    defer llama_init.deinit();

    const model = llama_init.model;
    const ctx = llama_init.context;

    if (model == null or ctx == null) {
        log.err("%s : failed to init\n", .{func});
        return 1;
    }

    const n_ctx_train = c.llama_model_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        log.wrn("%s: model was trained on only %d context tokens (%d specified)\n", .{ func, n_ctx_train, params.n_ctx });
    }

    var model_draft: ?*c.llama_model = null;
    defer if (model_draft) |m| c.llama_model_free(m);
    if (use_draft) {
        var mparams_sidecar = tc.common.modelParamsToLlama(&params);
        mparams_sidecar.load_mtp = true;
        const path_md = try a.dupeSentinel(u8, params.speculative.draft.mparams.path, 0);
        model_draft = c.llama_model_load_from_file(path_md.ptr, mparams_sidecar);
        if (model_draft == null) {
            log.err("%s: failed to load the draft model '%s'\n", .{ func, path_md.ptr });
            return 1;
        }
        if (!tc.speculative.areCompatible(model, model_draft)) {
            log.err("%s: the target and draft vocab are not compatible\n", .{func});
            return 1;
        }
    }

    var nextn: ?NextnCollector = null;
    defer if (nextn) |*n| n.deinit();

    if (params.load_mtp) {
        const model_src = if (use_draft) model_draft else model;
        const path_src = if (use_draft) params.speculative.draft.mparams.path else params.model.path;
        const flag_src: [*:0]const u8 = if (use_draft) "'-md'" else "'--nextn'";

        const n_heads = c.llama_model_n_layer_nextn(model_src);
        const n_trunk_src = c.llama_model_n_layer(model_src);
        const n_trunk = c.llama_model_n_layer(model);
        const info: NextnModelInfo = if (n_heads > 0) try nextnReadModelInfo(a, path_src, n_trunk, n_heads) else .{};

        if (n_heads == 0) {
            log.wrn("%s: the model has no NextN layers, %s has no effect\n", .{ func, flag_src });
        } else if (n_trunk_src == 0) {
            log.err("%s: NextN layers sharing the trunk's KV cache are not supported\n", .{func});
            return 1;
        } else if (n_heads > 1) {
            log.err("%s: multi-layer NextN drafts are not supported (found %u)\n", .{ func, n_heads });
            return 1;
        } else if (!info.has_layers) {
            log.wrn("%s: no NextN tensor in '%s', %s has no effect\n", .{ func, (try a.dupeSentinel(u8, path_src, 0)).ptr, flag_src });
        } else {
            nextn = nextnCollectorInit(model_src, &params, info.own_lm_head[0]) orelse return 1;
            tc.speculative.ext.setEmbeddingsNextn(ctx, true, false); // masked = false
            g_collector.n_layer_nextn = n_heads;

            log.inf("%s: processing %d NextN layer(s) from block %d\n", .{ func, n_heads, n_trunk });
        }
    }

    // print system information
    {
        log.inf("\n", .{});
        log.inf("%s\n", .{(try tc.common.paramsGetSystemInfo(a, &params)).ptr});
    }

    if (!try computeImatrix(a, ctx, &params, n_ctx, if (nextn) |*n| n else null)) return 1;

    try g_collector.save(-1);

    log.out("\n", .{});
    c.llama_perf_context_print(ctx);

    c.llama_backend_free();

    return 0;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
