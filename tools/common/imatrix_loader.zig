//! `common_imatrix_load`: an importance matrix from a GGUF or the legacy
//! binary format.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/imatrix-loader.cpp` and the struct and keys of
//! `common/imatrix-loader.h` (v0.3.0, `c1d0e7a00`). Each declaration names
//! what it ports and the line it began at.
//!
//! # `std::map` order
//!
//! `common_imatrix::entries` is a `std::map`, so its order is the names'
//! bytewise order; `Imatrix.names` is kept sorted to match, and the GGUF
//! path's pairing map, also a `std::map`, is walked in that order too.

const std = @import("std");
const c = @import("tools_h");
const log = @import("log.zig");
const strings = @import("strings.zig");

/// `LLM_KV_IMATRIX_DATASETS`, `LLM_KV_IMATRIX_CHUNK_COUNT`, `LLM_KV_IMATRIX_CHUNK_SIZE`
/// (imatrix-loader.h:8, 9, 10 @c1d0e7a00).
pub const kv_datasets = "imatrix.datasets";
pub const kv_chunk_count = "imatrix.chunk_count";
pub const kv_chunk_size = "imatrix.chunk_size";

/// Ports `common_imatrix_entry` (imatrix-loader.h:12 @c1d0e7a00).
pub const Entry = struct {
    sums: std.ArrayList(f32) = .empty,
    counts: std.ArrayList(i64) = .empty,
};

/// Ports `common_imatrix` (imatrix-loader.h:17 @c1d0e7a00).
pub const Imatrix = struct {
    allocator: std.mem.Allocator,
    /// `entries`, by name.
    entries: std.StringHashMapUnmanaged(Entry) = .empty,
    datasets: std.ArrayList([]const u8) = .empty,
    chunk_count: i32 = 0,
    chunk_size: i32 = 0,
    is_legacy: bool = false,
    has_metadata: bool = false,

    /// Reserved for the lifecycle convention; cannot fail.
    pub fn init(allocator: std.mem.Allocator) !Imatrix {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Imatrix) void {
        var it = self.entries.iterator();
        while (it.next()) |kv| {
            self.allocator.free(kv.key_ptr.*);
            kv.value_ptr.sums.deinit(self.allocator);
            kv.value_ptr.counts.deinit(self.allocator);
        }
        self.entries.deinit(self.allocator);
        for (self.datasets.items) |d| self.allocator.free(d);
        self.datasets.deinit(self.allocator);
    }

    /// `entries[name]`: the entry, made empty if it was not there.
    pub fn entry(self: *Imatrix, name: []const u8) !*Entry {
        const gop = try self.entries.getOrPut(self.allocator, name);
        if (!gop.found_existing) {
            gop.key_ptr.* = try self.allocator.dupe(u8, name);
            gop.value_ptr.* = .{};
        }
        return gop.value_ptr;
    }

    /// The names in `std::map` order. Borrowed until the next call or
    /// `deinit`; the caller frees the slice.
    pub fn sortedNames(self: *const Imatrix, allocator: std.mem.Allocator) ![][]const u8 {
        const names = try allocator.alloc([]const u8, self.entries.count());
        var it = self.entries.keyIterator();
        var i: usize = 0;
        while (it.next()) |k| : (i += 1) names[i] = k.*;
        std.mem.sort([]const u8, names, {}, lessThan);
        return names;
    }
};

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: *anyopaque) c_int;
extern fn fread(ptr: [*]u8, size: usize, n: usize, f: *anyopaque) usize;
extern fn fgetc(f: *anyopaque) c_int;
extern fn ungetc(ch: c_int, f: *anyopaque) c_int;

/// `std::ifstream` in binary mode, as far as the legacy reader uses it:
/// `read` sets a sticky fail flag on a short read, `peek` looks at one byte.
const In = struct {
    f: *anyopaque,
    failed: bool = false,

    fn read(self: *In, buf: []u8) void {
        if (self.failed) return;
        if (buf.len == 0) return;
        if (fread(buf.ptr, 1, buf.len, self.f) != buf.len) self.failed = true;
    }

    fn readInt(self: *In, comptime T: type) T {
        var v: T = 0;
        self.read(std.mem.asBytes(&v));
        return v;
    }

    fn peekEof(self: *In) bool {
        if (self.failed) return true;
        const ch = fgetc(self.f);
        if (ch == -1) return true;
        _ = ungetc(ch, self.f);
        return false;
    }
};

/// Ports `common_imatrix_load_legacy` (imatrix-loader.cpp:10 @c1d0e7a00).
fn loadLegacy(fname: [:0]const u8, imatrix: *Imatrix) !bool {
    const func = "common_imatrix_load_legacy";
    const f = fopen(fname.ptr, "rb") orelse {
        log.err("%s: failed to open %s\n", .{ func, fname.ptr });
        return false;
    };
    defer _ = fclose(f);
    var in: In = .{ .f = f };

    const n_entries = in.readInt(c_int);
    if (in.failed or n_entries < 1) {
        log.err("%s: no data in file %s\n", .{ func, fname.ptr });
        return false;
    }

    const a = imatrix.allocator;
    var i: c_int = 0;
    while (i < n_entries) : (i += 1) {
        const len = in.readInt(i32);
        const name_buf = try a.alloc(u8, @intCast(@max(len, 0)));
        defer a.free(name_buf);
        in.read(name_buf);
        if (in.failed) {
            log.err("%s: failed reading name for entry %d from %s\n", .{ func, i + 1, fname.ptr });
            return false;
        }
        // `std::string{ name_as_vec.data() }`: up to the first NUL.
        const name = name_buf[0 .. std.mem.indexOfScalar(u8, name_buf, 0) orelse name_buf.len];

        const ncall = in.readInt(i32);
        const nval = in.readInt(i32);
        if (in.failed or nval < 1) {
            log.err("%s: failed reading number of values for entry %d\n", .{ func, i });
            return false;
        }

        const e = try imatrix.entry(name);
        try e.sums.resize(a, @intCast(nval));
        in.read(std.mem.sliceAsBytes(e.sums.items));
        if (in.failed) {
            log.err("%s: failed reading data for entry %d\n", .{ func, i });
            return false;
        }

        try e.counts.resize(a, 1);
        e.counts.items[0] = ncall;
    }

    // the trailing data (chunk count + dataset name) is optional
    if (!in.peekEof()) {
        const n_calls = in.readInt(i32);
        imatrix.chunk_count = n_calls;

        if (!in.failed) {
            const len = in.readInt(i32);
            if (!in.failed and len > 0) {
                const dataset = try a.alloc(u8, @intCast(len));
                defer a.free(dataset);
                @memset(dataset, 0);
                in.read(dataset);
                if (!in.failed) {
                    const d = dataset[0 .. std.mem.indexOfScalar(u8, dataset, 0) orelse dataset.len];
                    try imatrix.datasets.append(a, try a.dupe(u8, d));
                }
            }
        }
    }

    imatrix.chunk_size = 0;
    imatrix.is_legacy = true;

    return true;
}

/// Ports `common_imatrix_load` (imatrix-loader.cpp:82 @c1d0e7a00).
///
/// Parameters:
/// - `fname`: a GGUF imatrix, or the legacy format if it does not parse as one.
/// - `imatrix`: filled in; owns everything it holds.
///
/// Return: whether it loaded. Failures are logged as the C logs them.
pub fn load(fname: [:0]const u8, imatrix: *Imatrix) !bool {
    const func = "common_imatrix_load";
    var ctx: ?*c.ggml_context = null;
    const meta_gguf_params: c.gguf_init_params = .{ .no_alloc = false, .ctx = &ctx };
    const ctx_gguf = c.gguf_init_from_file(fname.ptr, meta_gguf_params) orelse return loadLegacy(fname, imatrix);
    defer {
        c.gguf_free(ctx_gguf);
        c.ggml_free(ctx);
    }

    const n_entries: i32 = @truncate(c.gguf_get_n_tensors(ctx_gguf));
    if (n_entries < 1) {
        log.err("%s: no data in file %s\n", .{ func, fname.ptr });
        return false;
    }

    const a = imatrix.allocator;
    const datasets_key = c.gguf_find_key(ctx_gguf, kv_datasets);
    const chunk_count_key = c.gguf_find_key(ctx_gguf, kv_chunk_count);
    const chunk_size_key = c.gguf_find_key(ctx_gguf, kv_chunk_size);

    if (datasets_key != -1 and c.gguf_get_kv_type(ctx_gguf, datasets_key) == c.GGUF_TYPE_ARRAY and
        c.gguf_get_arr_type(ctx_gguf, datasets_key) == c.GGUF_TYPE_STRING)
    {
        const n = c.gguf_get_arr_n(ctx_gguf, datasets_key);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            try imatrix.datasets.append(a, try a.dupe(u8, std.mem.span(c.gguf_get_arr_str(ctx_gguf, datasets_key, i))));
        }
    }

    imatrix.has_metadata = (datasets_key != -1 and chunk_count_key != -1 and chunk_size_key != -1);
    imatrix.chunk_count = if (chunk_count_key != -1) @bitCast(c.gguf_get_val_u32(ctx_gguf, chunk_count_key)) else 0;
    imatrix.chunk_size = if (chunk_size_key != -1) @bitCast(c.gguf_get_val_u32(ctx_gguf, chunk_size_key)) else 0;

    // `sums_counts_for`: name -> (in_sum2, counts), walked in name order.
    const Pair = struct { sums: ?*c.ggml_tensor = null, counts: ?*c.ggml_tensor = null };
    var sums_counts_for: std.StringArrayHashMapUnmanaged(Pair) = .empty;
    defer {
        for (sums_counts_for.keys()) |k| a.free(k);
        sums_counts_for.deinit(a);
    }

    var cur = c.ggml_get_first_tensor(ctx);
    while (cur) |t| : (cur = c.ggml_get_next_tensor(ctx, t)) {
        const name = std.mem.sliceTo(&t.*.name, 0);
        if (name.len == 0) continue;

        if (strings.removeSuffix(name, ".in_sum2")) |base| {
            const gop = try sums_counts_for.getOrPut(a, base);
            if (!gop.found_existing) {
                gop.key_ptr.* = try a.dupe(u8, base);
                gop.value_ptr.* = .{};
            }
            gop.value_ptr.sums = t;
        } else if (strings.removeSuffix(name, ".counts")) |base| {
            const gop = try sums_counts_for.getOrPut(a, base);
            if (!gop.found_existing) {
                gop.key_ptr.* = try a.dupe(u8, base);
                gop.value_ptr.* = .{};
            }
            gop.value_ptr.counts = t;
        }
    }

    const SortCtx = struct {
        keys: []const []const u8,
        pub fn lessThan(self: @This(), x: usize, y: usize) bool {
            return std.mem.order(u8, self.keys[x], self.keys[y]) == .lt;
        }
    };
    sums_counts_for.sort(SortCtx{ .keys = sums_counts_for.keys() });

    for (sums_counts_for.keys(), sums_counts_for.values()) |name, sc| {
        const in_sum2 = sc.sums orelse null;
        const counts = sc.counts orelse null;

        if (in_sum2 == null or counts == null) {
            log.err("%s: mismatched sums and counts for %.*s\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr });
            return false;
        }

        if (in_sum2.?.*.type != c.GGML_TYPE_F32 or counts.?.*.type != c.GGML_TYPE_F32) {
            log.err("%s: sums and counts for %.*s must be F32\n", .{ func, @as(c_int, @intCast(name.len)), name.ptr });
            return false;
        }

        const e = try imatrix.entry(name);

        const nval: usize = @intCast(c.ggml_nelements(in_sum2));
        const ncounts: usize = @intCast(c.ggml_nelements(counts));

        const sums_data: [*]const f32 = @ptrCast(@alignCast(in_sum2.?.*.data));
        try e.sums.resize(a, nval);
        @memcpy(e.sums.items, sums_data[0..nval]);

        const counts_data: [*]const f32 = @ptrCast(@alignCast(counts.?.*.data));
        try e.counts.resize(a, ncounts);
        // `std::lround`: half away from zero.
        for (e.counts.items, counts_data[0..ncounts]) |*dst, v| dst.* = @intFromFloat(@round(v));
    }

    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
