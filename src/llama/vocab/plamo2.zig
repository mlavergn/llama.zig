//! The PLaMo-2 tokenizer: a suffix table over the vocabulary, then a
//! dynamic program for the cheapest segmentation, with byte fallback.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v0.3.0 (`c1d0e7a00`).
//!
//! The C++ builds its table from `std::unordered_map`s, but iterates only
//! one of them, and only to collect its keys, which it then sorts by their
//! reversed bytes: distinct strings under a total order. So no hash order
//! reaches the result, and plain Zig maps stand in for them. Where the C++
//! reads a map with `operator[]`, which inserts 0 for a missing key, so
//! does this.

const std = @import("std");
const impl = @import("../impl.zig");
const unicode = @import("../unicode.zig");
const vm = @import("module.zig");

const allocator = impl.allocator;
const Vocab = vm.Vocab;
const Token = vm.Token;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

const table_piece_length = 0;
const table_token_id = 1;
const table_score = 2;
const table_piece_id = 3;

const path_token_length = 0;
const path_token_id = 1;
const path_num_tokens = 2;

const invalid_score: i32 = -20000000;
const unknown_score: i32 = -10000000;

/// Ports `llm_tokenizer_plamo2` (llama-vocab.cpp:1351 @c1d0e7a00).
pub const Tokenizer = struct {
    /// `bytes_`: the token for each byte value.
    bytes: [256]Token,
    /// `to_suffix_id_`: (code point, piece id) to suffix id.
    to_suffix_id: std.AutoHashMapUnmanaged(i64, i32) = .empty,
    /// `table_`: [piece_length, token_id, score, piece_id] rows.
    table: std.ArrayList([4]i32) = .empty,

    /// Ports `llm_tokenizer_plamo2`'s `build`.
    ///
    /// Return: the tokenizer, or `error.Thrown` -- a byte with no token, or
    /// a code point `unicode_cpt_to_utf8` refuses.
    pub fn init(vocab: *const Vocab, exc: *impl.Exception) impl.Thrown!Tokenizer {
        var t: Tokenizer = .{ .bytes = @splat(0) };
        errdefer t.deinit();

        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Build token list and byte mapping
        var suffix_to_score: std.StringHashMapUnmanaged(f32) = .empty;
        var token_to_id: std.StringHashMapUnmanaged(Token) = .empty;

        const n = vocab.nTokens();
        var token_id: u32 = 0;
        while (token_id < n) : (token_id += 1) {
            const tid: Token = @intCast(token_id);
            const entry = try vocab.getTokenData(tid, exc);
            const text = vocab.text(tid);
            (token_to_id.getOrPut(arena, text) catch oom()).value_ptr.* = tid;

            // Handle byte tokens
            if (vocab.isByte(tid)) {
                if (text.len == 6 and std.mem.eql(u8, text[0..3], "<0x") and text[5] == '>') {
                    if (std.ascii.isHex(text[3]) and std.ascii.isHex(text[4])) {
                        const byte_val = std.fmt.parseInt(u8, text[3..5], 16) catch unreachable;
                        t.bytes[byte_val] = tid;
                    }
                }
                continue;
            }

            // Add token and all its suffixes to suffix_to_score
            (suffix_to_score.getOrPut(arena, text) catch oom()).value_ptr.* = entry.score;

            // Extract suffixes character by character (UTF-8 aware)
            const cpts = unicode.cptsFromUtf8(text);
            defer allocator.free(cpts);
            var i: usize = 1;
            while (i < cpts.len) : (i += 1) {
                const suffix = try utf8Of(arena, cpts[i..], exc);
                const g = suffix_to_score.getOrPut(arena, suffix) catch oom();
                if (!g.found_existing) g.value_ptr.* = std.math.nan(f32);
            }
        }

        // Check that all byte tokens are set
        for (t.bytes, 0..) |b, i| {
            if (b == 0) return exc.throw(.runtime_error, "Byte token for <0x{d}> is not set", .{i});
        }

        // Build suffix list in lexicographical order of reversed strings
        var suffixes: std.ArrayList([]const u8) = .empty;
        suffixes.ensureTotalCapacity(arena, suffix_to_score.count() + 1) catch oom();
        var kit = suffix_to_score.keyIterator();
        while (kit.next()) |k| suffixes.appendAssumeCapacity(k.*);
        suffixes.appendAssumeCapacity(""); // Empty suffix
        std.mem.sort([]const u8, suffixes.items, {}, reversedLess);

        // Build suffix_to_id and to_suffix_id_
        var suffix_to_id: std.StringHashMapUnmanaged(i32) = .empty;
        var num_pieces: i32 = 0;

        for (suffixes.items) |suffix| {
            (suffix_to_id.getOrPut(arena, suffix) catch oom()).value_ptr.* = num_pieces;
            if (suffix.len != 0) {
                const cpts = unicode.cptsFromUtf8(suffix);
                defer allocator.free(cpts);

                const remaining = try utf8Of(arena, cpts[1..], exc);
                const rid = suffix_to_id.getOrPut(arena, remaining) catch oom();
                if (!rid.found_existing) rid.value_ptr.* = 0;
                const piece_code: i64 = (@as(i64, cpts[0]) << 32) | @as(i64, rid.value_ptr.*);
                (t.to_suffix_id.getOrPut(allocator, piece_code) catch oom()).value_ptr.* = num_pieces;

                // Count number of pieces for this suffix
                var pieces_for_suffix: i32 = 1; // sentinel row
                var piece_length: usize = cpts.len;
                while (piece_length > 0) : (piece_length -= 1) {
                    const piece = try utf8Of(arena, cpts[0..piece_length], exc);
                    if (suffix_to_score.contains(piece)) pieces_for_suffix += 1;
                }
                num_pieces += pieces_for_suffix;
            } else {
                num_pieces += 1; // Empty suffix contributes one piece (sentinel row)
            }
        }

        // Build flattened table
        t.table.appendNTimes(allocator, .{ 0, 0, 0, 0 }, @intCast(num_pieces)) catch oom();
        var table_idx: usize = 0;

        for (suffixes.items) |suffix| {
            // Add all prefixes of the suffix to the table (in decreasing order of length)
            const cpts = unicode.cptsFromUtf8(suffix);
            defer allocator.free(cpts);
            var piece_length: usize = cpts.len;
            while (piece_length > 0) : (piece_length -= 1) {
                const piece = try utf8Of(arena, cpts[0..piece_length], exc);
                const score = suffix_to_score.get(piece) orelse continue;

                const row = &t.table.items[table_idx];
                row[table_piece_length] = @intCast(piece_length);
                row[table_token_id] = token_to_id.get(piece) orelse -1;
                row[table_score] = if (std.math.isFinite(score)) @intFromFloat(@round(@as(f64, score) * 1e4)) else invalid_score;
                const pid = suffix_to_id.getOrPut(arena, piece) catch oom();
                if (!pid.found_existing) pid.value_ptr.* = 0;
                row[table_piece_id] = pid.value_ptr.*;

                table_idx += 1;
            }

            // Add sentinel row
            const row = &t.table.items[table_idx];
            row[table_piece_length] = 1;
            row[table_token_id] = -1;
            row[table_score] = unknown_score;
            table_idx += 1;
        }

        return t;
    }

    /// Return: nothing.
    pub fn deinit(self: *Tokenizer) void {
        self.to_suffix_id.deinit(allocator);
        self.table.deinit(allocator);
    }

    /// Ports `llm_tokenizer_plamo2`'s `encode` (llama-vocab.cpp:1493
    /// @c1d0e7a00), appending to `output` as the session does.
    ///
    /// Return: nothing; `error.Thrown` is not raised but kept for the
    /// tokenizers' common shape.
    pub fn encode(self: *const Tokenizer, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
        _ = exc;
        const all = unicode.cptsFromUtf8(text);
        defer allocator.free(all);
        // Skip the first code point if it is a BOM (Byte Order Mark)
        const unicode_data = if (all.len != 0 and all[0] == 0xFEFF) all[1..] else all;
        if (unicode_data.len == 0) return;

        const data_len = unicode_data.len;

        // Initialize scores array (dynamic programming)
        const scores = allocator.alloc(i64, data_len + 1) catch oom();
        defer allocator.free(scores);
        @memset(scores, @as(i64, 1) << 60);
        scores[data_len] = 0;

        // Path array to track best tokenization
        const path = allocator.alloc([3]i32, data_len + 1) catch oom();
        defer allocator.free(path);
        @memset(path, .{ 0, 0, 0 });

        const table = self.table.items;
        var suffix_id: i32 = 0;

        // Process from end to beginning
        var ii: isize = @as(isize, @intCast(data_len)) - 1;
        while (ii >= 0) : (ii -= 1) {
            const i: usize = @intCast(ii);
            const ch = unicode_data[i];

            // Find next suffix ID
            var p: usize = @intCast(suffix_id);
            while (p < table.len) : (p += 1) {
                const piece_code: i64 = (@as(i64, ch) << 32) | @as(i64, table[p][table_piece_id]);
                suffix_id = self.to_suffix_id.get(piece_code) orelse 0;
                if (suffix_id > 0 or table[p][table_score] == unknown_score) break;
            }

            // Update best path
            p = @intCast(suffix_id);
            while (p < table.len) : (p += 1) {
                const score = table[p][table_score];
                if (score > invalid_score) {
                    const piece_length: usize = @intCast(table[p][table_piece_length]);
                    const s: i64 = scores[i + piece_length] - score;
                    if (s < scores[i]) {
                        scores[i] = s;
                        path[i][path_token_length] = @intCast(piece_length);
                        path[i][path_token_id] = table[p][table_token_id];
                        path[i][path_num_tokens] = path[i + piece_length][path_num_tokens] + 1;
                        if (score == unknown_score) {
                            // Add UTF-8 byte count
                            path[i][path_num_tokens] += @as(i32, @intFromBool(ch >= 0x80)) + @intFromBool(ch >= 0x800) + @intFromBool(ch >= 0x10000);
                        }
                    }
                }
                if (score == unknown_score) break;
            }
        }

        // Decode the best path
        output.ensureUnusedCapacity(allocator, @intCast(path[0][path_num_tokens])) catch oom();
        var pos: usize = 0;
        while (pos < data_len) {
            if (path[pos][path_token_id] >= 0) {
                output.append(allocator, path[pos][path_token_id]) catch oom();
            } else {
                // Fall back to byte tokens
                const ch = unicode_data[pos];
                const s: u32 = 1 + @as(u32, @intFromBool(ch >= 0x80)) + @intFromBool(ch >= 0x800) + @intFromBool(ch >= 0x10000);
                var i: u32 = 0;
                while (i < s) : (i += 1) {
                    var b: u8 = undefined;
                    if (s == 1) {
                        b = @truncate(ch);
                    } else if (i == 0) {
                        b = @truncate((@as(u32, 0xF00) >> @intCast(s)) & 0xFF);
                    } else {
                        b = 0x80;
                    }
                    const idx = b | @as(u8, @truncate((ch >> @intCast((s - i - 1) * 6)) & 0x3F));
                    output.append(allocator, self.bytes[idx]) catch oom();
                }
            }
            pos += @intCast(path[pos][path_token_length]);
        }
    }
};

/// `std`'s `string` comparison of the reversed strings: bytewise, unsigned.
fn reversedLess(_: void, a: []const u8, b: []const u8) bool {
    var i: usize = 0;
    while (i < a.len and i < b.len) : (i += 1) {
        const x = a[a.len - 1 - i];
        const y = b[b.len - 1 - i];
        if (x != y) return x < y;
    }
    return a.len < b.len;
}

/// The code points concatenated as UTF-8, as the C++'s `+=` of
/// `unicode_cpt_to_utf8` builds them.
fn utf8Of(arena: std.mem.Allocator, cpts: []const u32, exc: *impl.Exception) impl.Thrown![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (cpts) |cp| {
        var b: [4]u8 = undefined;
        const s = unicode.cptToUtf8(cp, &b) catch return exc.throw(.invalid_argument, "invalid codepoint", .{});
        out.appendSlice(arena, s) catch oom();
    }
    return out.items;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "suffixes order by their reversed bytes" {
    try std.testing.expect(reversedLess({}, "", "a"));
    try std.testing.expect(reversedLess({}, "ba", "ab"));
    try std.testing.expect(!reversedLess({}, "ab", "ba"));
    try std.testing.expect(reversedLess({}, "a", "ba"));
}
