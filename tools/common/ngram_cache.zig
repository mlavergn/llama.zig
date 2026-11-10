//! The n-gram caches of lookup decoding: n-grams of up to four tokens, each
//! mapped to how often every following token was seen.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/ngram-cache.h` and
//! `llama.cpp/common/ngram-cache.cpp` at v0.9.0 (`cd959bae9`). Each
//! declaration names the C++ it replaces and the line it began at.
//!
//! # Iteration order
//!
//! Both levels are `std::unordered_map`s, and their order is observable: a
//! draft takes the *first* token with the highest count, so ties go to
//! whichever comes first in the part, and `save` writes the cache in
//! iteration order. Both are `cxx.HashMap`, libc++'s layout. Where the C++
//! copies a part before iterating it, the copy keeps the source's structure
//! (libc++ 22's `__copy_construct`), so iterating the original is the same.

const std = @import("std");
const c = @import("tools_h");
const cxx = @import("cxx.zig");
const log = @import("log.zig");
const io = @import("io.zig");

/// Ports `LLAMA_NGRAM_MIN`, `LLAMA_NGRAM_MAX`, `LLAMA_NGRAM_STATIC`
/// (ngram-cache.h:9, 10, 11 @cd959bae9).
pub const ngram_min = 1;
pub const ngram_max = 4;
pub const ngram_static = 2;

/// Ports `common_ngram` (ngram-cache.h:15 @cd959bae9): up to four tokens,
/// padded with `LLAMA_TOKEN_NULL`. Saved to disk as its 16 bytes.
pub const Ngram = extern struct {
    tokens: [ngram_max]c.llama_token,

    /// The default constructor: every token `LLAMA_TOKEN_NULL`.
    pub fn empty() Ngram {
        return .{ .tokens = @splat(c.LLAMA_TOKEN_NULL) };
    }

    /// The constructor from `ngram_size` tokens at `input`.
    ///
    /// Parameters:
    /// - `input`: at least `ngram_size` tokens.
    /// - `ngram_size`: how many to take; the rest are `LLAMA_TOKEN_NULL`.
    ///
    /// Return: the n-gram.
    pub fn fromSlice(input: [*]const c.llama_token, ngram_size: i32) Ngram {
        var r: Ngram = undefined;
        for (0..ngram_max) |i| {
            r.tokens[i] = if (@as(i32, @intCast(i)) < ngram_size) input[i] else c.LLAMA_TOKEN_NULL;
        }
        return r;
    }

    /// `operator==`.
    pub fn eql(a: Ngram, b: Ngram) bool {
        for (0..ngram_max) |i| {
            if (a.tokens[i] != b.tokens[i]) return false;
        }
        return true;
    }
};

/// Ports `common_token_hash_function` (ngram-cache.h:40 @cd959bae9):
/// Fibonacci hashing, the token widened to 64 bits with its sign.
pub fn tokenHash(token: c.llama_token) u64 {
    return @as(u64, @bitCast(@as(i64, token))) *% 11400714819323198485;
}

/// Ports `common_ngram_hash_function` (ngram-cache.h:47 @cd959bae9).
pub fn ngramHash(ngram: Ngram) u64 {
    var hash = tokenHash(ngram.tokens[0]);
    for (1..ngram_max) |i| {
        hash ^= tokenHash(ngram.tokens[i]);
    }
    return hash;
}

/// Ports `common_ngram_cache_part` (ngram-cache.h:58 @cd959bae9): token to
/// the number of times it has been seen, keyed by `std::hash<int>`.
pub const CachePart = cxx.HashMap(c.llama_token, i32, cxx.hashInt, cxx.eqlInt);

/// Ports `common_ngram_cache` (ngram-cache.h:61 @cd959bae9): n-gram to the
/// empirical distribution of the tokens following it. Owns its parts.
pub const Cache = cxx.HashMap(Ngram, CachePart, ngramHash, Ngram.eql);

/// Frees a cache and its parts.
pub fn deinitCache(cache: *Cache) void {
    cache.deinit();
}

/// A copy of a cache, as the C++ copy constructor makes it.
///
/// Return: the copy, freed with `deinitCache`.
pub fn cloneCache(allocator: std.mem.Allocator, cache: *const Cache) error{OutOfMemory}!Cache {
    return cache.clone(allocator);
}

/// Ports `common_ngram_cache_update` (ngram-cache.cpp:12 @cd959bae9): count
/// every n-gram ending in the last `nnew` tokens of `inp`.
///
/// Parameters:
/// - `cache`: the cache to update.
/// - `ngram_min_`, `ngram_max_`: the n-gram sizes to count.
/// - `inp`: the history; it may only ever be appended to.
/// - `nnew`: how many tokens were appended since the last call.
/// - `print_progress`: report every ten million n-grams on stderr.
pub fn update(cache: *Cache, ngram_min_: i32, ngram_max_: i32, inp: []const c.llama_token, nnew: i32, print_progress: bool) error{OutOfMemory}!void {
    const t_start_ms = c.ggml_time_ms();
    const inp_size: i64 = @intCast(inp.len);

    const n_todo: i64 = inp_size * (ngram_max_ - ngram_min_ + 1);
    var n_done: i64 = 0;

    var ngram_size: i64 = ngram_min_;
    while (ngram_size <= ngram_max_) : (ngram_size += 1) {
        const i_start = @max(inp_size - nnew, ngram_size);
        var i: i64 = i_start;
        while (i < inp_size) : (i += 1) {
            const ngram_start = i - ngram_size;
            const ngram = Ngram.fromSlice(inp.ptr + @as(usize, @intCast(ngram_start)), @intCast(ngram_size));
            const token = inp[@intCast(i)];

            if (cache.get(ngram)) |part| {
                if (part.get(token)) |token_count| {
                    token_count.* += 1;
                } else {
                    _ = try part.emplace(token, 1);
                }
            } else {
                var part = try CachePart.init(cache.allocator);
                errdefer part.deinit();
                _ = try part.emplace(token, 1);
                _ = try cache.emplace(ngram, part);
            }
            n_done += 1;

            if (print_progress and @mod(n_done, 10000000) == 0) {
                const t_now_ms = c.ggml_time_ms();
                const eta_ms = @divTrunc((inp_size * (ngram_max_ - ngram_min_ + 1) - n_done) * (t_now_ms - t_start_ms), n_done);
                const eta_min = @divTrunc(eta_ms, 60 * 1000);
                const eta_s = @divTrunc(eta_ms - 60 * 1000 * eta_min, 1000);

                io.eprintf("%s: %lld/%lld done, ETA: %02lld:%02lld\n", .{ @as([*:0]const u8, "common_ngram_cache_update"), @as(c_longlong, n_done), @as(c_longlong, n_todo), @as(c_longlong, eta_min), @as(c_longlong, eta_s) });
            }
        }
    }
}

/// Ports `get_token` (ngram-cache.cpp:55 @cd959bae9): token `i` of `inp`
/// followed by `draft` without its first token. An index the C++ would read
/// out of bounds -- `i` wrapped from a negative start, when `inp` is
/// shorter than an n-gram -- is undefined there and `LLAMA_TOKEN_NULL` here.
fn getToken(inp: []const c.llama_token, draft_: []const c.llama_token, i: u64) c.llama_token {
    if (i < inp.len) return inp[@intCast(i)];
    const j = 1 +% i -% inp.len;
    return if (j < draft_.len) draft_[@intCast(j)] else c.LLAMA_TOKEN_NULL;
}

/// Ports `draft_min_sample_size_lax`, `draft_min_percent_lax`,
/// `draft_min_sample_size_strict`, `draft_min_percent_strict`
/// (ngram-cache.cpp:60, 61, 62, 63 @cd959bae9): below these the draft stops.
const draft_min_sample_size_lax = [ngram_max]i32{ 2, 2, 1, 1 };
const draft_min_percent_lax = [ngram_max]i32{ 66, 50, 50, 50 };
const draft_min_sample_size_strict = [ngram_max]i32{ 4, 3, 2, 2 };
const draft_min_percent_strict = [ngram_max]i32{ 75, 66, 66, 66 };

/// Ports `try_draft` (ngram-cache.cpp:66 @cd959bae9), the overload over the
/// static cache alone: its most frequent token, if seen often enough.
fn tryDraftStatic(nc_static: *const Cache, ngram_static_: Ngram) c.llama_token {
    const part_static = nc_static.get(ngram_static_) orelse return c.LLAMA_TOKEN_NULL;

    var max_count_static: i32 = 0;
    var sum_count_static: i32 = 0;
    var max_token: c.llama_token = c.LLAMA_TOKEN_NULL;

    var it = part_static.iterator();
    while (it.next()) |e| {
        const token = e.key;
        const count_static = e.value.*;

        if (count_static > max_count_static) {
            max_token = token;
            max_count_static = count_static;
        }
        sum_count_static +%= count_static;
    }

    if (sum_count_static < draft_min_sample_size_lax[ngram_static - 1]) {
        return c.LLAMA_TOKEN_NULL;
    }
    if (100 *% max_count_static < draft_min_percent_lax[ngram_static - 1] *% sum_count_static) {
        return c.LLAMA_TOKEN_NULL;
    }
    return max_token;
}

/// Ports `try_draft` (ngram-cache.cpp:98 @cd959bae9), the overload over a
/// primary cache (context or dynamic), validated against the static part:
/// the longest n-gram first.
fn tryDraft(nc_primary: *const Cache, ngrams_primary: []const Ngram, part_static: ?*const CachePart, min_sample_size: *const [ngram_max]i32, min_percent: *const [ngram_max]i32) c.llama_token {
    var drafted_token: c.llama_token = c.LLAMA_TOKEN_NULL;

    var i: i32 = @as(i32, @intCast(ngrams_primary.len)) - 1;
    while (i >= 0 and drafted_token == c.LLAMA_TOKEN_NULL) : (i -= 1) {
        const ngram_primary = ngrams_primary[@intCast(i)];

        const part_primary = nc_primary.get(ngram_primary) orelse continue;

        var max_count_primary: i32 = 0;
        var max_count_static: i32 = 0;
        var sum_count_primary: i32 = 0;
        var max_token: c.llama_token = c.LLAMA_TOKEN_NULL;

        var it = part_primary.iterator();
        while (it.next()) |e| {
            const token = e.key;

            const token_count_static: ?*i32 = if (part_static) |ps| ps.get(token) else null;

            const count_primary = e.value.*;
            const count_static: i32 = if (token_count_static) |cs| 100 *% cs.* else 1;

            if (count_primary *% count_static > max_count_primary *% max_count_static) {
                max_token = token;
                max_count_primary = count_primary;
                max_count_static = count_static;
            }
            sum_count_primary +%= count_primary;
        }

        if (sum_count_primary < min_sample_size[@intCast(i)]) {
            continue;
        }
        if (100 *% max_count_primary < min_percent[@intCast(i)] *% sum_count_primary) {
            continue;
        }
        drafted_token = max_token;
    }

    return drafted_token;
}

/// Ports `common_ngram_cache_draft` (ngram-cache.cpp:146 @cd959bae9): extend
/// `draft_out` one token at a time from the context cache, then the dynamic
/// one, then the static one, until none answers or `n_draft` are drafted.
///
/// Parameters:
/// - `inp`: the tokens generated so far.
/// - `draft_out`: holds exactly the last sampled token on entry; the drafted
///   tokens are appended.
/// - `allocator`: for `draft_out` and scratch.
/// - `n_draft`: at most this many tokens are added.
/// - `ngram_min_`, `ngram_max_`: the n-gram sizes of the context and dynamic caches.
/// - `nc_context`, `nc_dynamic`, `nc_static`: the three caches, read only.
pub fn draft(inp: []const c.llama_token, draft_out: *std.ArrayList(c.llama_token), allocator: std.mem.Allocator, n_draft: i32, ngram_min_: i32, ngram_max_: i32, nc_context: *Cache, nc_dynamic: *Cache, nc_static: *Cache) error{OutOfMemory}!void {
    if (draft_out.items.len != 1) _ = c.ggml_abort("ngram-cache.cpp", 150, "GGML_ASSERT(%s) failed", "draft.size() == 1");
    const inp_size: i64 = @intCast(inp.len);

    if (inp_size < ngram_static) {
        return;
    }

    var ngrams_cd: std.ArrayList(Ngram) = .empty;
    defer ngrams_cd.deinit(allocator);

    while (@as(i64, @intCast(draft_out.items.len)) - 1 < n_draft) {
        var drafted_token: c.llama_token = c.LLAMA_TOKEN_NULL;

        const ngram_start_static: i64 = inp_size - ngram_static + @as(i64, @intCast(draft_out.items.len)) - 1;
        var ngram_static_ = Ngram.empty();
        var j = ngram_start_static;
        while (j < ngram_start_static + ngram_static) : (j += 1) {
            ngram_static_.tokens[@intCast(j - ngram_start_static)] = getToken(inp, draft_out.items, @bitCast(j));
        }
        // the C++ copies the part; the copy is only searched
        const part_static: ?*const CachePart = nc_static.get(ngram_static_);

        // cd = context + dynamic
        ngrams_cd.clearRetainingCapacity();
        var ngram_size_cd: i32 = ngram_min_;
        while (ngram_size_cd <= ngram_max_) : (ngram_size_cd += 1) {
            const ngram_start_cd: i64 = inp_size - ngram_size_cd + @as(i64, @intCast(draft_out.items.len)) - 1;
            var ngram_cd = Ngram.empty();
            var k = ngram_start_cd;
            while (k < ngram_start_cd + ngram_size_cd) : (k += 1) {
                ngram_cd.tokens[@intCast(k - ngram_start_cd)] = getToken(inp, draft_out.items, @bitCast(k));
            }
            try ngrams_cd.append(allocator, ngram_cd);
        }
        if (drafted_token == c.LLAMA_TOKEN_NULL) {
            drafted_token = tryDraft(nc_context, ngrams_cd.items, part_static, &draft_min_sample_size_lax, &draft_min_percent_lax);
        }
        if (drafted_token == c.LLAMA_TOKEN_NULL) {
            drafted_token = tryDraft(nc_dynamic, ngrams_cd.items, part_static, &draft_min_sample_size_strict, &draft_min_percent_strict);
        }
        if (drafted_token == c.LLAMA_TOKEN_NULL) {
            drafted_token = tryDraftStatic(nc_static, ngram_static_);
        }

        if (drafted_token == c.LLAMA_TOKEN_NULL) {
            break;
        }

        log.dbg(" - draft candidate: token=%d\n", .{@as(c_int, drafted_token)});
        try draft_out.append(allocator, drafted_token);
    }
}

const FILE = opaque {};
extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*FILE;
extern fn fclose(f: *FILE) c_int;
extern fn fwrite(ptr: *const anyopaque, size: usize, n: usize, f: *FILE) usize;
extern fn fread(ptr: *anyopaque, size: usize, n: usize, f: *FILE) usize;

/// Ports `common_ngram_cache_save` (ngram-cache.cpp:198 @cd959bae9): each
/// n-gram's 16 bytes, its token count, and its `(token, count)` pairs, in
/// iteration order. A file that cannot be opened is written to as the
/// C++'s failed `std::ofstream` is: not at all, silently.
///
/// Parameters:
/// - `cache`: the cache.
/// - `filename`: where to write it.
pub fn save(cache: *const Cache, filename: [:0]const u8) void {
    const f = fopen(filename.ptr, "wb") orelse return;
    defer _ = fclose(f);
    var it = cache.iterator();
    while (it.next()) |item| {
        const ngram = item.key;
        const token_counts = item.value;
        if (token_counts.count() == 0) _ = c.ggml_abort("ngram-cache.cpp", 203, "GGML_ASSERT(%s) failed", "!token_counts.empty()");
        const ntokens: i32 = @intCast(token_counts.count());
        if (!(ntokens > 0)) _ = c.ggml_abort("ngram-cache.cpp", 205, "GGML_ASSERT(%s) failed", "ntokens > 0");

        _ = fwrite(&ngram, @sizeOf(Ngram), 1, f);
        _ = fwrite(&ntokens, @sizeOf(i32), 1, f);
        var it2 = token_counts.iterator();
        while (it2.next()) |item2| {
            const token = item2.key;
            const count_ = item2.value.*;
            if (!(count_ > 0)) _ = c.ggml_abort("ngram-cache.cpp", 212, "GGML_ASSERT(%s) failed", "count > 0");

            _ = fwrite(&token, @sizeOf(c.llama_token), 1, f);
            _ = fwrite(&count_, @sizeOf(i32), 1, f);
        }
    }
}

/// `std::istream::read`'s answer: all `n` bytes or a failed stream at
/// end of file, which is all the C++ asks of it.
const Reader = struct {
    f: *FILE,
    eof: bool = false,

    fn read(self: *Reader, dst: []u8) bool {
        if (self.eof) return false;
        if (fread(dst.ptr, 1, dst.len, self.f) == dst.len) return true;
        self.eof = true;
        return false;
    }
};

/// Ports `common_ngram_cache_load` (ngram-cache.cpp:220 @cd959bae9). Records
/// end at the first n-gram that does not read whole; a record cut short
/// after its n-gram aborts, as the C++'s assertions do.
///
/// Parameters:
/// - `allocator`: owns the cache.
/// - `filename`: the file `save` wrote.
///
/// Return: the cache, or `error.OpenFailed` where the C++ throws
/// `std::ifstream::failure("Unable to open file " + filename)`.
pub fn load(allocator: std.mem.Allocator, filename: [:0]const u8) error{ OutOfMemory, OpenFailed }!Cache {
    const f = fopen(filename.ptr, "rb") orelse return error.OpenFailed;
    defer _ = fclose(f);
    var hashmap_file: Reader = .{ .f = f };
    var ngram_cache = try Cache.init(allocator);
    errdefer ngram_cache.deinit();

    var ngram: Ngram = Ngram.empty();
    var ntokens: i32 = 0;
    var token: c.llama_token = 0;
    var count_: i32 = 0;

    while (hashmap_file.read(std.mem.asBytes(&ngram))) {
        if (hashmap_file.eof) _ = c.ggml_abort("ngram-cache.cpp", 237, "GGML_ASSERT(%s) failed", "!hashmap_file.eof()");
        if (!hashmap_file.read(std.mem.asBytes(&ntokens))) _ = c.ggml_abort("ngram-cache.cpp", 238, "GGML_ASSERT(%s) failed", "hashmap_file.read(ntokensc, sizeof(int32_t))");
        if (!(ntokens > 0)) _ = c.ggml_abort("ngram-cache.cpp", 239, "GGML_ASSERT(%s) failed", "ntokens > 0");
        var token_counts = try CachePart.init(allocator);
        errdefer token_counts.deinit();

        var i: i32 = 0;
        while (i < ntokens) : (i += 1) {
            if (hashmap_file.eof) _ = c.ggml_abort("ngram-cache.cpp", 243, "GGML_ASSERT(%s) failed", "!hashmap_file.eof()");
            if (!hashmap_file.read(std.mem.asBytes(&token))) _ = c.ggml_abort("ngram-cache.cpp", 244, "GGML_ASSERT(%s) failed", "hashmap_file.read(tokenc, sizeof(llama_token))");
            if (hashmap_file.eof) _ = c.ggml_abort("ngram-cache.cpp", 245, "GGML_ASSERT(%s) failed", "!hashmap_file.eof()");
            if (!hashmap_file.read(std.mem.asBytes(&count_))) _ = c.ggml_abort("ngram-cache.cpp", 246, "GGML_ASSERT(%s) failed", "hashmap_file.read(countc, sizeof(int32_t))");
            if (!(count_ > 0)) _ = c.ggml_abort("ngram-cache.cpp", 247, "GGML_ASSERT(%s) failed", "count > 0");
            _ = try token_counts.emplace(token, count_);
        }

        // the C++ emplaces a copy; a copy keeps the part's structure
        const r = try ngram_cache.emplace(ngram, token_counts);
        if (!r.inserted) token_counts.deinit();
    }
    if (!hashmap_file.eof) _ = c.ggml_abort("ngram-cache.cpp", 253, "GGML_ASSERT(%s) failed", "hashmap_file.eof()");

    return ngram_cache;
}

/// Ports `common_ngram_cache_merge` (ngram-cache.cpp:258 @cd959bae9): add
/// every count in `add` to `target`.
///
/// Parameters:
/// - `target`: the cache added to.
/// - `add`: the cache added from, read only.
pub fn merge(target: *Cache, add: *const Cache) error{OutOfMemory}!void {
    var it = add.iterator();
    while (it.next()) |ngram_part| {
        const ngram = ngram_part.key;
        const part = ngram_part.value;

        const part_merged = target.get(ngram) orelse {
            var copy = try part.clone(target.allocator);
            errdefer copy.deinit();
            _ = try target.emplace(ngram, copy);
            continue;
        };

        var it2 = part.iterator();
        while (it2.next()) |token_count| {
            const token = token_count.key;
            const count_ = token_count.value.*;
            if (!(count_ > 0)) _ = c.ggml_abort("ngram-cache.cpp", 272, "GGML_ASSERT(%s) failed", "count > 0");

            const token_count_merged = part_merged.get(token) orelse {
                _ = try part_merged.emplace(token, count_);
                continue;
            };

            token_count_merged.* +%= count_;
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the token hash widens with its sign and wraps" {
    try std.testing.expectEqual(@as(u64, 11400714819323198485), tokenHash(1));
    try std.testing.expectEqual(@as(u64, 0) -% 11400714819323198485, tokenHash(-1));
    const n = Ngram.empty();
    // four equal hashes xor to zero
    try std.testing.expectEqual(@as(u64, 0), ngramHash(n));
}

test "update counts every n-gram and a draft follows the most frequent continuation" {
    const a = std.testing.allocator;
    var ctx = try Cache.init(a);
    defer ctx.deinit();
    var dyn = try Cache.init(a);
    defer dyn.deinit();
    var sta = try Cache.init(a);
    defer sta.deinit();
    const inp = [_]c.llama_token{ 1, 2, 3, 1, 2, 3, 1, 2 };
    try update(&ctx, 1, 4, &inp, inp.len, false);
    // 1-grams: 7 distinct positions over tokens {1,2,3}
    const one = ctx.get(Ngram.fromSlice(&[_]c.llama_token{1}, 1)).?;
    try std.testing.expectEqual(@as(i32, 3), one.get(2).?.*);

    var d: std.ArrayList(c.llama_token) = .empty;
    defer d.deinit(a);
    try d.append(a, 3);
    try draft(&inp, &d, a, 3, 1, 4, &ctx, &dyn, &sta);
    // inp ends with the sampled token, which draft[0] repeats and the
    // n-grams skip: (2,3,1,2) -> 3, (3,1,2,3) -> 1, (1,2,3,1) -> 2
    try std.testing.expectEqualSlices(c.llama_token, &.{ 3, 3, 1, 2 }, d.items);
}

test "save and load round-trip in iteration order, and merge adds counts" {
    const a = std.testing.allocator;
    var ctx = try Cache.init(a);
    defer ctx.deinit();
    const inp = [_]c.llama_token{ 5, 6, 7, 5, 6, 8, 5, 6, 7 };
    try update(&ctx, 1, 2, &inp, inp.len, false);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/cache.bin", .{tmp.sub_path}, 0);
    defer a.free(path);
    save(&ctx, path);
    var back = try load(a, path);
    defer back.deinit();
    // loading inserts in the file's order, so the loaded map iterates
    // differently, as libc++'s does: compare contents
    try std.testing.expectEqual(ctx.count(), back.count());
    var it1 = ctx.iterator();
    while (it1.next()) |e1| {
        const p2 = back.get(e1.key).?;
        try std.testing.expectEqual(e1.value.count(), p2.count());
        var it2 = e1.value.iterator();
        while (it2.next()) |tc| try std.testing.expectEqual(tc.value.*, p2.get(tc.key).?.*);
    }
    try std.testing.expectError(error.OpenFailed, load(a, "/nonexistent/x/y"));

    try merge(&back, &ctx);
    const p = back.get(Ngram.fromSlice(&[_]c.llama_token{ 5, 6 }, 2)).?;
    try std.testing.expectEqual(@as(i32, 4), p.get(7).?.*);
}

test "a map keyed by the Fibonacci hash iterates as libc++'s does" {
    // measured: libc++ 22 (zig c++) iterates these keys emplaced in this
    // order as 11 2 9 3 100 7 -1 5 over 11 buckets
    const M = cxx.HashMap(c.llama_token, i32, tokenHash, cxx.eqlInt);
    var m = try M.init(std.testing.allocator);
    defer m.deinit();
    for ([_]c.llama_token{ 5, -1, 7, 100, 3, 9, 11, 2 }) |k| _ = try m.emplace(k, 1);
    try std.testing.expectEqual(@as(usize, 11), m.bucketCount());
    var it = m.iterator();
    for ([_]c.llama_token{ 11, 2, 9, 3, 100, 7, -1, 5 }) |k| try std.testing.expectEqual(k, it.next().?.key);
}
