//! `llama-imatrix`: collect the importance matrix a quantizer weighs its
//! rounding by, merge saved ones, or print statistics about one.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/imatrix/imatrix.cpp` at v0.3.0 (`c1d0e7a00`).
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

const std = @import("std");
const builtin = @import("builtin");
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
extern fn log2(x: f64) f64;
extern fn sqrt(x: f64) f64;

const gpa = std.heap.c_allocator;

/// `GGML_ASSERT`, which upstream keeps in every build mode.
fn ggmlAssert(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort("imatrix.cpp", line, "GGML_ASSERT(%s) failed", expr);
}

/// Ports `print_usage` (imatrix.cpp:28 @c1d0e7a00).
fn printUsage(argv: []const [:0]const u8) void {
    log.out("\nexample usage:\n", .{});
    log.out("\n    %s \\\n" ++
        "       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \\\n" ++
        "       [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \\\n" ++
        "       [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \\\n" ++
        "       [--show-statistics] [...]\n", .{argv[0].ptr});
    log.out("\n", .{});
}

/// Ports `Stats` (imatrix.cpp:38 @c1d0e7a00).
const Stats = struct {
    values: std.ArrayList(f32) = .empty,
    counts: std.ArrayList(i64) = .empty,
};

/// Ports `tensor_statistics` (imatrix.cpp:43 @c1d0e7a00).
const TensorStatistics = struct {
    tensor: []const u8,
    stats: *const Stats,
    total_sqract: f32 = 0.0,
    mean_sqract: f32 = 0.0,
    max_sqract: f32 = 0.0,
    min_sqract: f32 = 0.0,
    elements: c_int = 0,
    stddev: f32 = 0.0,
    active: f32 = 0.0,
    entropy: f32 = 0.0,
    zd: f32 = 0.0,
    cossim: f32 = 0.0,
};

/// The reference compiler's shape for `std::inner_product` over floats --
/// `init = init + a*b`, contracted, then a strict in-order reduction by the
/// loop vectorizer: products rounded in whole groups, the rest fused. The
/// group is `src/ggml/cpu/simd.zig`'s `strict_loop.group`, measured there.
const strict_group: ?usize = blk: {
    const model = builtin.cpu.model.name;
    if (builtin.cpu.arch == .aarch64 and std.mem.startsWith(u8, model, "apple_")) break :blk 4;
    if (builtin.cpu.arch == .aarch64 and std.mem.eql(u8, model, "generic")) break :blk 8;
    break :blk null;
};

fn innerProduct(x: []const f32, y: []const f32) f32 {
    var sum: f32 = 0.0;
    const t = x.len;
    const group = strict_group orelse t + 1;
    const split = t - t % group;
    var k: usize = 0;
    while (k < split) : (k += 1) sum += x[k] * y[k];
    while (k < t) : (k += 1) sum = @mulAdd(f32, x[k], y[k], sum);
    return sum;
}

/// What `IMatrixCollector` keeps of `common_params`: `set_params` copies
/// the whole struct, so later changes to the caller's do not reach it.
const CollectorParams = struct {
    n_ctx: i32 = 0,
    n_parallel: i32 = 1,
    process_output: bool = false,
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
            .n_out_freq = p.n_out_freq,
            .n_save_freq = p.n_save_freq,
            .imat_dat = p.imat_dat,
            .out_file = p.out_file,
            .prompt_file = p.prompt_file,
        };
    }
};

/// Ports `IMatrixCollector` (imatrix.cpp:58 @c1d0e7a00). The C++ locks a
/// mutex in `collect_imatrix`; the scheduler calls it from the thread
/// computing the graph, one node at a time, so there is nothing to lock.
const Collector = struct {
    stats: libcxx.StringMap(Stats),
    params: CollectorParams = .{},
    datasets: std.ArrayList([]const u8) = .empty,
    last_chunk: i32 = 0,
    src1_data: std.ArrayList(u8) = .empty,
    ids: std.ArrayList(u8) = .empty, // the expert ids from ggml_mul_mat_id

    fn chunkSize(self: *const Collector) i32 {
        return @divTrunc(self.params.n_ctx, self.params.n_parallel);
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

    /// Ports `collect_imatrix` (imatrix.cpp:234 @c1d0e7a00), a member of `IMatrixCollector`.
    fn collect(self: *Collector, t: *c.ggml_tensor, ask: bool) !bool {
        const func = "collect_imatrix";
        const src0: *c.ggml_tensor = t.src[0];
        const src1: *c.ggml_tensor = t.src[1];
        const wname = filterTensorName(std.mem.sliceTo(&src0.name, 0));

        const chunk_size = self.chunkSize();

        // when ask is true, the scheduler wants to know if we are interested in data from this tensor
        // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
        if (ask) {
            if (t.op == c.GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
            if (t.op != c.GGML_OP_MUL_MAT) return false;
            // why are small batches ignored (<16 tokens)?
            if (src1.ne[1] < 16 or src1.type != c.GGML_TYPE_F32) return false;
            if (!(std.mem.startsWith(u8, wname, "blk.") or (self.params.process_output and std.mem.eql(u8, wname, "output.weight")))) return false;
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
        ggmlAssert(src1.nb[0] == c.ggml_element_size(src1), 266, "src1->nb[0] == ggml_element_size(src1)");

        // this has been adapted to the new format of storing merged experts in a single 3d tensor
        if (t.op == c.GGML_OP_MUL_MAT_ID) {
            //   ids  -> [n_experts_used, n_tokens]
            //   src1 -> [cols, n_expert_used, n_tokens]
            const ids: *c.ggml_tensor = t.src[2];
            const n_as = src0.ne[2];
            const n_ids = ids.ne[0];

            ggmlAssert(ids.ne[1] == src1.ne[2], 281, "ids->ne[1] == src1->ne[2]");

            // the extra dimension would need to be stored somewhere to be reflected in the imatrix file
            if (c.ggml_nrows(src1) != src1.ne[1] * src1.ne[2]) {
                log.err("%s: tensor has more than 3 dimensions: %.*s", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr });
                ggmlAssert(false, 286, "false");
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
                try e.values.appendNTimes(gpa, 0, @intCast(src1.ne[0] * n_as));
                if (e.counts.items.len < nas) try e.counts.appendNTimes(gpa, 0, nas - e.counts.items.len) else e.counts.shrinkRetainingCapacity(nas);
            } else if (e.values.items.len != @as(usize, @intCast(src1.ne[0] * n_as))) {
                log.err("%s: inconsistent size for %.*s (%d vs %d)\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr, @as(c_int, @intCast(e.values.items.len)), @as(c_int, @intCast(src1.ne[0] * n_as)) });
                tc.cExit(1);
            } else if (e.counts.items.len != nas) {
                log.err("%s: inconsistent expert count for %.*s (%d vs %d)\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr, @as(c_int, @intCast(e.counts.items.len)), @as(c_int, @intCast(n_as)) });
                tc.cExit(1);
            }
            log.tmpl(.debug, 2, "%s[%d]: %32.*s, %s, %5d x %5d, %d\n", .{ func, self.last_chunk, @as(c_int, @intCast(wname.len)), wname.ptr, c.ggml_op_name(t.op), @as(c_int, @intCast(src1.ne[0])), @as(c_int, @intCast(src1.ne[2])), @as(c_int, @intCast(src1.type)) });

            const ne0: usize = @intCast(src1.ne[0]);
            const n_tokens = src1.ne[2];

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

                    ggmlAssert(ex >= 0 and ex < n_as, 321, "ex >= 0 && ex < n_as"); // sanity check

                    const j11 = @rem(idx, src1.ne[1]);
                    const x: [*]const f32 = @ptrCast(@alignCast(data + @as(usize, @intCast(j11 * @as(i64, @intCast(src1.nb[1])) + row * @as(i64, @intCast(src1.nb[2]))))));
                    const acc = e.values.items[@as(usize, @intCast(ex)) * ne0 ..];

                    e.counts.items[@intCast(ex)] += 1;
                    touched[@intCast(ex)] = 1;
                    for (0..ne0) |j| acc[j] = @mulAdd(f32, x[j], x[j], acc[j]);
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
                const n_chunk: i32 = @truncate(@divTrunc(counts[ex], chunk_size));
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
                try e.values.appendNTimes(gpa, 0, @intCast(src1.ne[0] * n_mat));
                if (e.counts.items.len == 0) try e.counts.append(gpa, 0) else e.counts.shrinkRetainingCapacity(1);
            } else if (e.values.items.len != @as(usize, @intCast(src1.ne[0] * n_mat))) {
                log.err("%s: inconsistent size for %.*s (%d vs %d)\n", .{ func, @as(c_int, @intCast(wname.len)), wname.ptr, @as(c_int, @intCast(e.values.items.len)), @as(c_int, @intCast(src1.ne[0] * n_mat)) });
                tc.cExit(1);
            }
            log.tmpl(.debug, 2, "%s[%d]: %32.*s, %s, %5d x %5d x %5d, %d\n", .{ func, self.last_chunk, @as(c_int, @intCast(wname.len)), wname.ptr, c.ggml_op_name(t.op), @as(c_int, @intCast(src1.ne[0])), @as(c_int, @intCast(src1.ne[1])), @as(c_int, @intCast(src1.ne[2])), @as(c_int, @intCast(src1.type)) });

            const ne0: usize = @intCast(src1.ne[0]);

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
                const n_chunk: i32 = @truncate(@divTrunc(counts[i], chunk_size));
                try self.maybeSave(n_chunk);
            }
        }

        return true;
    }

    /// Ports `save_imatrix_legacy` (imatrix.cpp:426 @c1d0e7a00), a member of `IMatrixCollector`.
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

        const chunk_size = self.chunkSize();

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

    /// Ports `save_imatrix` (imatrix.cpp:532 @c1d0e7a00), a member of `IMatrixCollector`.
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
            data_size += pad(c.ggml_tensor_overhead() + @sizeOf(f32) * kv.value.values.items.len, c.GGML_MEM_ALIGN);
            data_size += pad(c.ggml_tensor_overhead() + @sizeOf(f32) * kv.value.counts.items.len, c.GGML_MEM_ALIGN);
        }

        // deterministic tensor name order
        std.mem.sort([]const u8, to_store.items, {}, lessThan);

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
            c.gguf_set_val_u32(ctx_gguf, tc.imatrix_loader.kv_chunk_size, @bitCast(self.chunkSize()));
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
            }
        }

        _ = c.gguf_write_to_file(ctx_gguf, @ptrCast(fname.items.ptr), false);

        log.tmpl(.none, 1, "\n", .{});
        log.tmpl(.debug, 1, "%s: stored collected data after %d chunks in %s\n", .{ func, self.last_chunk, fname.items.ptr });

        c.gguf_free(ctx_gguf);
        c.ggml_free(ctx);
    }

    /// Ports `load_imatrix` (imatrix.cpp:642 @c1d0e7a00), a member of `IMatrixCollector`.
    fn load(self: *Collector, a: std.mem.Allocator, file_name: [:0]const u8) !bool {
        const func = "load_imatrix";
        var loaded = try tc.imatrix_loader.Imatrix.init(a);
        if (!try tc.imatrix_loader.load(file_name, &loaded)) return false;

        const chunk_size = self.chunkSize();
        const is_legacy = loaded.is_legacy;

        // `common_imatrix::entries` is a `std::map`: name order
        const names = try loaded.sortedNames(a);
        for (names) |name| {
            const entry = loaded.entries.get(name).?;
            const e = (try self.stats.getOrPut(name, .{})).value;

            if (is_legacy) {
                // Legacy format: sums contain (raw_sum/raw_count)*ncall, counts contain {ncall}
                // Reconstruct raw form by multiplying by chunk_size
                if (e.values.items.len == 0) {
                    try e.values.appendNTimes(gpa, 0, entry.sums.items.len);
                    if (e.counts.items.len == 0) try e.counts.append(gpa, 0) else e.counts.shrinkRetainingCapacity(1);
                }
                for (entry.sums.items, 0..) |s, j| e.values.items[j] = @mulAdd(f32, s, @floatFromInt(chunk_size), e.values.items[j]);
                for (e.counts.items) |*cnt| cnt.* += entry.counts.items[0] * chunk_size;
            } else {
                // GGUF format: raw sums and counts, accumulate directly
                const nval = entry.sums.items.len;
                const ncounts = entry.counts.items.len;

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
            }
        }

        for (loaded.datasets.items) |d| try self.datasets.append(gpa, try gpa.dupe(u8, d));

        // Calculate the last chunk count
        var max_count: i64 = 0;
        var it = self.stats.iterator();
        while (it.next()) |kv| {
            for (kv.value.counts.items) |cnt| max_count = @max(max_count, cnt);
        }
        self.last_chunk = @truncate(@divTrunc(max_count, chunk_size));

        return true;
    }
};

fn pad(x: usize, n: usize) usize {
    return (x + n - 1) & ~(n - 1);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Ports `filter_tensor_name` (imatrix.cpp:79 @c1d0e7a00): remove any prefix
/// and suffix, `CUDA0#blk.0.attn_k.weight#0` => `blk.0.attn_k.weight`.
fn filterTensorName(name: []const u8) []const u8 {
    const p = std.mem.indexOfScalar(u8, name, '#') orelse return name;
    const rest = name[p + 1 ..];
    const q = std.mem.indexOfScalar(u8, rest, '#') orelse return rest;
    return rest[0..q];
}

/// Ports `process_tensor_name` (imatrix.cpp:96 @c1d0e7a00). `layer` is kept
/// across calls, as the comparator's shared string is.
fn processTensorName(input: []const u8, layer: *[]const u8, tensor: *[]const u8) void {
    var parts: [64][]const u8 = undefined;
    var n: usize = 0;
    // `std::getline(stream, item, '.')`: no empty last item after a final '.'
    var it = std.mem.splitScalar(u8, input, '.');
    while (it.next()) |item| {
        if (it.index == null and item.len == 0) break;
        if (n < parts.len) {
            parts[n] = item;
            n += 1;
        }
    }
    for (0..n) |i| {
        if (std.mem.eql(u8, parts[i], "blk") and i + 1 < n) {
            layer.* = parts[i + 1];
            break;
        }
    }
    for (0..n) |i| {
        if (std.mem.eql(u8, parts[i], "weight") and i > 0) {
            tensor.* = parts[i - 1];
            break;
        }
    }
    if (tensor.len == 0) tensor.* = input;
    if (layer.len == 0) layer.* = "-";
}

/// Ports `compute_statistics` (imatrix.cpp:125 @c1d0e7a00).
fn computeStatistics(a: std.mem.Allocator, tstats: *std.ArrayList(TensorStatistics), name: []const u8, e: *const Stats) !void {
    const func = "compute_statistics";
    // `%` by an empty `counts` is a division by zero; aarch64's `udiv` gives
    // zero, so the remainder is the whole size
    const rem = if (e.counts.items.len == 0) e.values.items.len else e.values.items.len % e.counts.items.len;
    if (rem != 0) {
        log.err("%s: activation size mismatch for tensor %.*s (%zu vs %zu)\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr, e.counts.items.len, e.values.items.len });
        return;
    }
    if (e.counts.items.len == 0) {
        log.err("%s: there are no activations for tensor %.*s. The imatrix may be suboptimal\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr });
        return;
    }

    const n_mat: c_int = @intCast(e.counts.items.len);
    const row_size: c_int = @intCast(@divTrunc(@as(c_int, @intCast(e.values.items.len)), n_mat));

    var activations: std.ArrayList(f32) = .empty;
    try activations.ensureTotalCapacity(a, e.values.items.len);

    var i: c_int = 0;
    while (i < n_mat) : (i += 1) {
        if (e.counts.items[@intCast(i)] == 0) {
            log.dbg("%s: skipping tensor %.*s due to zero count at index %d\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr, i });
            continue;
        }
        var j: c_int = 0;
        while (j < row_size) : (j += 1) {
            try activations.append(a, e.values.items[@intCast(i * row_size + j)] / @as(f32, @floatFromInt(e.counts.items[@intCast(i)])));
        }
    }

    if (activations.items.len == 0) {
        log.err("%s: all counts are zero for tensor %.*s, skipping statistics computation\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr });
        return;
    }

    const acts = activations.items;
    const size: f32 = @floatFromInt(acts.len);
    var act_total: f32 = 0.0;
    for (acts) |v| act_total += v;
    var act_max = acts[0];
    for (acts[1..]) |v| {
        if (act_max < v) act_max = v;
    }
    var act_min = acts[0];
    for (acts[1..]) |v| {
        if (v < act_min) act_min = v;
    }
    const act_mean = act_total / size;
    const act_sqr_total = innerProduct(acts, acts);
    const act_var = @mulAdd(f32, -act_mean, act_mean, act_sqr_total / size);
    const act_dev = sqrtf(@max(0.0, act_var));
    const threshold: f32 = 1e-5;
    var inactive_count: c_int = 0;
    for (acts) |v| {
        if (@abs(v) <= threshold) inactive_count += 1;
    }
    const active_ratio = 1 - @as(f32, @floatFromInt(inactive_count)) / size;

    var entropy: f32 = 0;
    if (act_total > 0) {
        for (acts) |act| {
            const p = act / act_total;
            if (p > 0) entropy = @mulAdd(f32, -p, log2f(p), entropy);
        }
    }

    var z_score: c_int = 0;
    if (act_dev > 0.0) {
        for (acts) |act| {
            const p = (act - act_mean) / act_dev;
            if (p > 1) z_score += 1;
        }
    }

    const elements: c_int = @intCast(acts.len);
    try tstats.append(a, .{
        .tensor = name,
        .stats = e,
        .total_sqract = act_total,
        .mean_sqract = act_mean,
        .max_sqract = act_max,
        .min_sqract = act_min,
        .elements = elements,
        .stddev = act_dev,
        .active = active_ratio,
        .entropy = entropy,
        .zd = @as(f32, @floatFromInt(z_score)) / @as(f32, @floatFromInt(elements)),
    });
}

/// `std::regex_search(s, m, std::regex(R"(blk\.(\d+)\.)"))`: the leftmost
/// match, as the start and length of the group.
fn findBlk(s: []const u8) ?struct { pos: usize, len: usize } {
    var i: usize = 0;
    while (i + 4 <= s.len) : (i += 1) {
        if (!std.mem.eql(u8, s[i .. i + 4], "blk.")) continue;
        var j = i + 4;
        while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
        if (j > i + 4 and j < s.len and s[j] == '.') return .{ .pos = i + 4, .len = j - (i + 4) };
    }
    return null;
}

/// Ports `compute_cossim` (imatrix.cpp:200 @c1d0e7a00).
fn computeCossim(a: std.mem.Allocator, tstats: []TensorStatistics) !void {
    for (tstats) |*ts| {
        if (findBlk(ts.tensor)) |m| {
            const blk = switch (tc.cxx.stoi(try a.dupeSentinel(u8, ts.tensor[m.pos..][0..m.len], 0))) {
                .ok => |v| v,
                // `\d+` that does not fit an int: the C++ throws out of main
                else => |e| {
                    io.eprintf("libc++abi: terminating due to uncaught exception of type std::out_of_range: %s\n", .{e.what()});
                    std.process.abort();
                },
            };
            const tname = try std.fmt.allocPrint(a, "{s}{d}{s}", .{ ts.tensor[0..m.pos], blk - 1, ts.tensor[m.pos + m.len ..] });
            for (tstats) |*prev| {
                if (!std.mem.eql(u8, prev.tensor, tname)) continue;
                const cur = ts.stats.values.items;
                const pv = prev.stats.values.items;
                const n = @min(pv.len, cur.len);
                const dp = innerProduct(cur[0..n], pv[0..n]);
                const curr_mag = sqrtf(innerProduct(cur, cur));
                const prev_mag = sqrtf(innerProduct(pv, pv));
                ts.cossim = dp / (curr_mag * prev_mag);
                break;
            }
        } else {
            ts.cossim = 0;
        }
    }
}

fn allFinite(v: []const f32) bool {
    for (v) |x| {
        if (!std.math.isFinite(x)) return false;
    }
    return true;
}

var g_collector: Collector = undefined;

/// Ports `ik_collect_imatrix` (imatrix.cpp:715 @c1d0e7a00).
fn ikCollectImatrix(t: [*c]c.ggml_tensor, ask: bool, _: ?*anyopaque) callconv(.c) bool {
    return g_collector.collect(t, ask) catch false;
}

/// Ports `results_log_softmax` and `log_softmax` (imatrix.cpp:719, 745 @c1d0e7a00).
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

/// Ports `process_logits` (imatrix.cpp:757 @c1d0e7a00).
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

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// Ports `compute_imatrix` (imatrix.cpp:791 @c1d0e7a00).
fn computeImatrix(a: std.mem.Allocator, ctx: ?*c.llama_context, params: *const Params, n_ctx: i32) !bool {
    const func = "compute_imatrix";
    const model = c.llama_get_model(ctx);
    const vocab = c.llama_model_get_vocab(model);

    const add_bos = c.llama_vocab_get_add_bos(vocab);

    if (c.llama_pooling_type(ctx) != c.LLAMA_POOLING_TYPE_LAST) {
        ggmlAssert(!c.llama_vocab_get_add_eos(vocab), 798, "!llama_vocab_get_add_eos(vocab)");
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

    ggmlAssert(n_batch < n_ctx or @rem(n_batch, n_ctx) == 0, 845, "n_batch < n_ctx || n_batch % n_ctx == 0");
    ggmlAssert(params.n_ctx == n_seq * n_ctx, 846, "params.n_ctx == n_seq * n_ctx");

    var batch = c.llama_batch_init(@min(n_batch, n_ctx * n_seq), 0, 1);

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

        var j: c_int = 0;
        while (j < num_batches) : (j += 1) {
            const batch_start = start + j * n_batch;
            const batch_size = @min(end - batch_start, n_batch);

            // clear the batch
            tc.common.batchClear(&batch);

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
                    tc.common.batchAdd(&batch, tokens[seq_start + @as(usize, @intCast(k))], j * n_batch + k, &.{seq}, true);
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (c.llama_decode(ctx, batch) != 0) {
                log.err("%s : failed to eval\n", .{func});
                c.llama_batch_free(batch);
                return false;
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

    c.llama_batch_free(batch);

    return true;
}

/// Mirrors `show_statistics`'s `weighted_stats` (imatrix.cpp:1000 @c1d0e7a00).
const WeightedStats = struct {
    weighted_bias: f32 = 0.0,
    weighted_zd: f32 = 0.0,
    weighted_cossim: f32 = 0.0,
    total_elements: c_int = 0,
};

/// `tensor_comparer` (imatrix.cpp:989 @c1d0e7a00).
fn tensorLess(_: void, a: TensorStatistics, b: TensorStatistics) bool {
    var layer: []const u8 = "";
    var name_a: []const u8 = "";
    var name_b: []const u8 = "";
    processTensorName(a.tensor, &layer, &name_a);
    processTensorName(b.tensor, &layer, &name_b);
    const o = std.mem.order(u8, name_a, name_b);
    return o == .lt or (o == .eq and a.total_sqract > b.total_sqract);
}

/// Ports `show_statistics` (imatrix.cpp:968 @c1d0e7a00).
fn showStatistics(a: std.mem.Allocator, params: *const Params) !bool {
    var ts: std.ArrayList(TensorStatistics) = .empty;
    if (params.in_files.items.len != 1) {
        log.err("\nError: a single imatrix file is required to compute tensor statistics\n\n", .{});
        return false;
    }
    const in0 = try a.dupeSentinel(u8, params.in_files.items[0], 0);
    if (try g_collector.load(a, in0)) {
        var it = g_collector.stats.iterator();
        while (it.next()) |kv| try computeStatistics(a, &ts, kv.key, kv.value);
    } else {
        log.err("\nError: %s is not a valid imatrix file\n\n", .{in0.ptr});
        return false;
    }
    if (ts.items.len != 0) {
        try computeCossim(a, ts.items);
    } else {
        log.err("Error: cannot compute statistics for %s\n\n", .{in0.ptr});
        return false;
    }

    libcxx.sort(TensorStatistics, ts.items, {}, tensorLess);

    // `std::map<int, weighted_stats>`: kept sorted by key
    var ws_keys: std.ArrayList(c_int) = .empty;
    var ws_vals: std.ArrayList(WeightedStats) = .empty;

    log.inf("\nComputing statistics for %s (%d tensors)\n", .{ in0.ptr, @as(c_int, @intCast(ts.items.len)) });
    log.inf("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", .{
        " Layer",     "       Tensor",
        "          Σ(Act²)",
        "  Min",      "            Max",
        "           μ",
        "   σ",
        " % Active",  "N",
        "   Entropy", "E (norm)",
        "ZD",         "  CosSim",
    });
    log.inf("=============================================================================================================" ++
        "============================================================\n", .{});
    for (ts.items) |tstat| {
        var layer: []const u8 = "";
        var name: []const u8 = "";
        processTensorName(tstat.tensor, &layer, &name);

        const blk: c_int = switch (tc.cxx.stoi(try a.dupeSentinel(u8, layer, 0))) {
            .ok => |v| v,
            else => -1, // not a block layer
        };

        const entropy_norm: f32 = if (tstat.elements > 0)
            @floatCast(100.0 * (@as(f64, tstat.entropy) / log2(@floatFromInt(tstat.elements))))
        else
            0.0;

        log.inf("%5.*s\t%-20.*s\t%10.2f\t%8.4f\t%11.4f\t%6.2f\t%6.2f\t%8.2f%%\t%6d\t%10.4f\t%6.2f%%\t%10.2f%%\t%8.4f\n", .{
            @as(c_int, @intCast(layer.len)), layer.ptr,
            @as(c_int, @intCast(name.len)),  name.ptr,
            @as(f64, tstat.total_sqract),    @as(f64, tstat.min_sqract),
            @as(f64, tstat.max_sqract),      @as(f64, tstat.mean_sqract),
            @as(f64, tstat.stddev),          @as(f64, tstat.active * 100.0),
            tstat.elements,                  @as(f64, tstat.entropy),
            @as(f64, entropy_norm),          @as(f64, 100.0 * tstat.zd),
            @as(f64, tstat.cossim),
        });

        const el: f32 = @floatFromInt(tstat.elements);
        const weighted_bias = el * tstat.total_sqract;
        const weighted_zd = el * tstat.zd;
        const weighted_cossim = el * tstat.cossim;

        var pos: usize = 0;
        while (pos < ws_keys.items.len and ws_keys.items[pos] < blk) pos += 1;
        if (pos < ws_keys.items.len and ws_keys.items[pos] == blk) {
            const w = &ws_vals.items[pos];
            w.weighted_bias += weighted_bias;
            w.weighted_zd += weighted_zd;
            w.weighted_cossim += weighted_cossim;
            w.total_elements += tstat.elements;
        } else {
            try ws_keys.insert(a, pos, blk);
            try ws_vals.insert(a, pos, .{ .weighted_bias = weighted_bias, .weighted_zd = weighted_zd, .weighted_cossim = weighted_cossim, .total_elements = tstat.elements });
        }
    }

    var layers: c_int = 0;
    for (ws_keys.items) |k| {
        if (k >= 0) layers += 1;
    }
    log.inf("\nComputing weighted average statistics per layer (%d layers)\n", .{layers});
    log.inf("\n%s\t%s\t%s\t%s\n", .{ "  Layer", "     μΣ(Act²)", "      μZD", "μCosSim" });
    log.inf("================================================\n", .{});
    for (ws_keys.items, ws_vals.items) |layer, stats| {
        if (stats.total_elements == 0) continue;
        if (layer >= 0) {
            const te: f32 = @floatFromInt(stats.total_elements);
            const bias = stats.weighted_bias / te;
            const zd = stats.weighted_zd / te;
            const cossim = stats.weighted_cossim / te;
            log.inf("%5d\t%14.2f\t%10.4f%%\t%6.4f\n", .{ layer, @as(f64, bias), @as(f64, 100.0 * zd), @as(f64, cossim) });
        }
    }
    log.inf("\n", .{});

    return true;
}

/// Ports `common_params_get_system_info` (common/common.cpp:421 @c1d0e7a00).
fn systemInfo(a: std.mem.Allocator, params: *const Params) ![:0]u8 {
    var os: std.ArrayList(u8) = .empty;
    try os.print(a, "system_info: n_threads = {d}", .{params.cpuparams.n_threads});
    if (params.cpuparams_batch.n_threads != -1) try os.print(a, " (n_threads_batch = {d})", .{params.cpuparams_batch.n_threads});
    try os.print(a, " / {d} | {s}", .{ std.Thread.getCpuCount() catch 0, c.llama_print_system_info() });
    return a.dupeSentinel(u8, os.items, 0);
}

/// Ports `main` (imatrix.cpp:1077 @c1d0e7a00).
fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "main";
    var params = try Params.init(a);

    params.out_file = "imatrix.gguf";

    params.n_ctx = 512;
    params.escape = false;

    tc.common.init();

    if (!try arg.parse(argv, &params, .imatrix, printUsage)) return 1;

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

    // print system information
    {
        log.inf("\n", .{});
        log.inf("%s\n", .{(try systemInfo(a, &params)).ptr});
    }

    if (!try computeImatrix(a, ctx, &params, n_ctx)) return 1;

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
