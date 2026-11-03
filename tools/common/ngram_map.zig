//! Self-speculation from the token history: find the n-gram just generated
//! earlier in the history and propose the m-gram that followed it.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/ngram-map.h` and `llama.cpp/common/ngram-map.cpp`
//! at v0.6.0 (`d81235049`). Each declaration names the C++ it replaces and
//! the line it began at.
//!
//! # Two algorithms
//!
//! `simpleDraft` (`ngram-simple`) searches backwards for the last n-gram and
//! copies what followed it. `Map` (`ngram-map-k`, `ngram-map-k4v`) keeps the
//! keys it has drafted from, with up to four m-gram values each and how many
//! tokens of the last draft were accepted, and a hash table from n-gram to
//! history index to find a key without the scan.
//!
//! # C arithmetic
//!
//! Indices are `size_t`, and the C++ subtracts them freely where its own
//! guards keep the result in range; the subtractions wrap here (`-%`) so a
//! guard that does not hold behaves as the C++'s unsigned arithmetic does.
//! The hash is `uint32_t`, wrapping.

const std = @import("std");
const c = @import("tools_h");
const log = @import("log.zig");

/// Ports `LCG_FACTOR` (ngram-map.cpp:11 @d81235049): the multiplier, near
/// `(sqrt(5) - 1) / 2 * 2^32`. `unsigned long` in the C++; only its low 32
/// bits reach the `uint32_t` hash.
const lcg_factor: u32 = 2654435761;

/// Ports `common_ngram_map_hash` (ngram-map.cpp:14 @d81235049): the LCG hash
/// of `len` tokens at `start`.
fn mapHash(tokens: []const c.llama_token, start: usize, len: usize) u32 {
    var hash: u32 = 0;
    for (0..len) |i| {
        hash = hash *% lcg_factor +% @as(u32, @bitCast(tokens[start + i]));
    }
    return hash;
}

/// Ports `common_tokens_to_str` (ngram-map.cpp:23 @d81235049): `length`
/// tokens at `start` as `[v0, v1, ...]`.
///
/// Return: the text, NUL-terminated, the allocator's.
fn tokensToStr(allocator: std.mem.Allocator, inp: []const c.llama_token, start: usize, length: usize) error{OutOfMemory}![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '[');
    for (0..length) |i| {
        if (i > 0) {
            try out.appendSlice(allocator, ", ");
        }
        try out.print(allocator, "{d}", .{inp[start + i]});
    }
    try out.append(allocator, ']');
    return out.toOwnedSliceSentinel(allocator, 0);
}

/// Ports `common_ngram_simple_config` (ngram-map.h:24 @d81235049).
pub const SimpleConfig = struct {
    /// size of n-grams to lookup in self-mode
    size_ngram: u16,
    /// size of m-grams to draft in self-mode
    size_mgram: u16,
};

/// Ports `common_ngram_simple_draft` (ngram-map.cpp:49 @d81235049): search
/// the history backwards for its last `size_ngram` tokens (the sampled one
/// last) and return up to `size_mgram` tokens that followed.
///
/// Parameters:
/// - `allocator`: owns the result.
/// - `config`: the n-gram and m-gram sizes.
/// - `tokens`: the history.
/// - `sampled`: the token just sampled, not yet in `tokens`.
///
/// Return: the draft, empty if there is no match; the caller frees it.
pub fn simpleDraft(allocator: std.mem.Allocator, config: SimpleConfig, tokens: []const c.llama_token, sampled: c.llama_token) error{OutOfMemory}!std.ArrayList(c.llama_token) {
    // Simple implementation of self-speculative decoding without a draft model.
    //
    const cur_len = tokens.len;

    const n_draft_min: usize = config.size_ngram; // size of n-gram to lookup in token history
    const n_draft_max: usize = config.size_mgram; // the m-gram following the found n-gram is used for draft

    // vector for tokens we want to verify.
    // return empty vector if there is no match.
    var draft_tokens: std.ArrayList(c.llama_token) = .empty;
    errdefer draft_tokens.deinit(allocator);

    // We need at least n_draft_min + n_draft_max + 1 tokens.
    if (cur_len <= n_draft_min + n_draft_max + 1) {
        return draft_tokens;
    }

    // pattern search
    var pattern: std.ArrayList(c.llama_token) = .empty;
    defer pattern.deinit(allocator);
    try pattern.ensureTotalCapacity(allocator, n_draft_min);
    var j = cur_len - n_draft_min + 1;
    while (j < cur_len) : (j += 1) {
        try pattern.append(allocator, tokens[j]);
    }
    try pattern.append(allocator, sampled); // add the last token to the pattern

    var match_pos: usize = 0; // we ignore position 0, position 0 == no match
    // search backwards, but skip the current match (we are currently there)
    j = cur_len - n_draft_min - 1;
    while (j > 0) : (j -= 1) {
        var match = true;
        for (pattern.items, 0..) |p, k| {
            if (tokens[j + k] != p) {
                match = false;
                break;
            }
        }
        if (match) {
            match_pos = j;
            break;
        }
    }
    if (match_pos == 0) {
        return draft_tokens;
    }

    const copy_max = @min(n_draft_max, cur_len - (match_pos + n_draft_min));
    if (copy_max < n_draft_min) {
        return draft_tokens;
    }
    log.dbg("%s: #tokens = %zu: found matching pattern at pos %zu, length %zu, draft length %zu\n", .{ @as([*:0]const u8, "common_ngram_simple_draft"), cur_len, match_pos, pattern.items.len, copy_max });

    try draft_tokens.ensureTotalCapacity(allocator, copy_max);
    for (0..copy_max) |k| {
        draft_tokens.appendAssumeCapacity(tokens[match_pos + n_draft_min + k]);
    }
    return draft_tokens;
}

/// Ports `COMMON_NGRAM_MAX_VALUES` (ngram-map.h:39 @d81235049): the m-gram
/// values kept per key.
pub const max_values = 4;

/// Ports `COMMON_NGRAM_HASH_MAP_SIZE` (ngram-map.h:42 @d81235049): entries in
/// the map from n-gram hash to history index.
pub const hash_map_size = 262144;

/// Ports `COMMON_NGRAM_MAX_VALUE_COUNT` (ngram-map.cpp:119 @d81235049): the
/// most a key or value count reaches.
const max_value_count = 16380;

/// Ports `common_ngram_map_value` (ngram-map.h:45 @d81235049): one m-gram
/// seen after a key.
pub const Value = struct {
    /// index of value m-gram in token-history (0 if unused)
    value_idx: usize = 0,
    /// number of occurrences of this value m-gram after the key n-gram (0 in an unused values-slot)
    value_num: u16 = 0,
    /// number of accepted tokens at last draft (-1 if unused)
    n_accepted: i16 = -1,
};

/// Ports `common_ngram_map_key` (ngram-map.h:52 @d81235049): one n-gram seen
/// more than once.
pub const Key = struct {
    /// index of key n-gram in token-history
    key_idx: usize,
    /// index of last token of statistics computation (key_num, values)
    stat_idx: usize,
    /// number of occurrences of this key n-gram in token-history
    key_num: u16,
    /// some known values after the key
    values: [max_values]Value = @splat(.{}),
};

/// Ports `common_ngram_map` (ngram-map.h:61 @d81235049).
pub const Map = struct {
    allocator: std.mem.Allocator,

    /// size of key n-grams
    size_key: u16,
    /// size of value m-grams
    size_value: u16,

    /// true if only key n-grams are used, no values.
    key_only: bool,

    /// key n-grams which occur several times in token-history
    keys: std.ArrayList(Key) = .empty,
    /// minimum number of key hits to consider a draft
    min_hits: u16,

    /// true, if statistics of the key_map should be printed.
    show_key_map_stats: bool = false,

    /// number of tokens at previous start of generation
    size_last_begin: usize = 0,

    /// true if a draft was created at last call.
    last_draft_created: bool = false,
    /// index of last key used for draft generation (0 = no draft)
    last_draft_key_idx: usize = 0,
    /// index of last value used for draft generation.
    last_draft_value_idx: u16 = 0,

    /// index of last check in context history
    idx_last_check: usize = 0,

    /// key_map[hash] = index of ngram in context window
    key_map: []u32,
    /// index of the last ngram added to key_map
    key_map_last_idx: u32 = 0,

    /// Ports `common_ngram_map` (ngram-map.h:72 @d81235049), the
    /// constructor: an empty map with every hash entry 0.
    ///
    /// Parameters:
    /// - `allocator`: owns the keys and the hash table.
    /// - `sz_key`, `sz_value`: the n-gram and m-gram sizes.
    /// - `only_keys`: draft from keys alone (`ngram-map-k`).
    /// - `min_hits`: key hits needed before a draft, with values.
    ///
    /// Return: the map, freed with `deinit`.
    pub fn init(allocator: std.mem.Allocator, sz_key: u16, sz_value: u16, only_keys: bool, min_hits: u16) error{OutOfMemory}!Map {
        const key_map = try allocator.alloc(u32, hash_map_size); // 2^18 hash entries, 0 entries if key_map shouldn't be used
        @memset(key_map, 0);
        return .{ .allocator = allocator, .size_key = sz_key, .size_value = sz_value, .key_only = only_keys, .min_hits = min_hits, .key_map = key_map };
    }

    /// Frees the keys and the hash table.
    pub fn deinit(self: *Map) void {
        self.keys.deinit(self.allocator);
        self.allocator.free(self.key_map);
    }

    /// A copy, as the C++ copies the struct.
    ///
    /// Return: the copy, freed with `deinit`.
    pub fn clone(self: *const Map, allocator: std.mem.Allocator) error{OutOfMemory}!Map {
        var m = self.*;
        m.allocator = allocator;
        m.keys = .empty;
        try m.keys.appendSlice(allocator, self.keys.items);
        errdefer m.keys.deinit(allocator);
        m.key_map = try allocator.dupe(u32, self.key_map);
        return m;
    }
};

fn matchAt(inp: []const c.llama_token, at: usize, key_tokens: []const c.llama_token) bool {
    for (key_tokens, 0..) |kt, k| {
        if (inp[at + k] != kt) return false;
    }
    return true;
}

/// Ports `common_ngram_map_begin` (ngram-map.cpp:121 @d81235049): a new
/// generation over `tokens`; when the history shrank, drop the keys, values
/// and hash entries past the point it shrank to.
///
/// Parameters:
/// - `map`: the map.
/// - `tokens`: the history the generation starts from.
pub fn begin(map: *Map, tokens: []const c.llama_token) void {
    const func: [*:0]const u8 = "common_ngram_map_begin";
    const size_begin = tokens.len;

    log.dbg("%s: begin, idx_last_draft=%zu, new begin=%zu, #keys=%zu\n", .{ func, map.idx_last_check, size_begin, map.keys.items.len });

    var idx_begin_cleanup = map.size_last_begin;
    if (idx_begin_cleanup > size_begin) {
        if (size_begin > @as(usize, map.size_key) + map.size_value) {
            idx_begin_cleanup = size_begin - map.size_key - map.size_value;
        } else {
            idx_begin_cleanup = 0;
        }
        log.inf("%s: shrink cleanup begin: %zu -> %zu\n", .{ func, map.size_last_begin, idx_begin_cleanup });
    }

    var count_map_entries_upd: usize = 0;
    if (map.key_map.len != 0 and size_begin < map.idx_last_check) {
        if (map.show_key_map_stats) {
            // Print statistics of hash map map_key.
            var count_nonzero: usize = 0;
            var min_idx: u32 = std.math.maxInt(u32);
            var max_idx: u32 = 0;
            for (map.key_map) |key_idx| {
                if (key_idx != 0) {
                    count_nonzero += 1;
                    if (key_idx < min_idx) min_idx = key_idx;
                    if (key_idx > max_idx) max_idx = key_idx;
                }
            }
            if (count_nonzero == 0) {
                min_idx = 0;
            }
            log.inf("%s: key_map stats: entries=%zu, min_idx=%u, max_idx=%u, key_map_last_idx=%u\n", .{ func, count_nonzero, min_idx, max_idx, map.key_map_last_idx });
        }

        // Update the map from hash to key index (clear outdated entries).
        for (map.key_map) |*key_idx| {
            if (key_idx.* != 0 and key_idx.* >= idx_begin_cleanup) {
                key_idx.* = 0;
                count_map_entries_upd += 1;
            }
        }
        map.key_map_last_idx = if (idx_begin_cleanup > 0) @truncate(idx_begin_cleanup - 1) else 0;
    }

    if (size_begin < map.idx_last_check and map.keys.items.len != 0) {
        const count_keys = map.keys.items.len;
        var count_keys_del: usize = 0;
        var count_values_del: usize = 0;
        var i: i32 = @as(i32, @intCast(map.keys.items.len)) - 1;
        while (i >= 0) : (i -= 1) {
            const key = &map.keys.items[@intCast(i)];
            if (key.key_idx >= idx_begin_cleanup) {
                // Delete the key.
                log.dbg("%s: delete key %d at index %zu (>= idx_begin_cleanup=%zu)\n", .{ func, i, key.key_idx, idx_begin_cleanup });
                _ = map.keys.orderedRemove(@intCast(i));
                count_keys_del += 1;
                continue;
            }
            if (map.key_only) {
                continue;
            }

            // Check the indices of the values.
            var j: i16 = max_values - 1;
            while (j >= 0) : (j -= 1) {
                const value = &key.values[@intCast(j)];
                if (value.value_idx != 0 and value.value_idx >= idx_begin_cleanup) {
                    // Delete the value.
                    count_values_del += 1;

                    // Move all values after this value to the left.
                    var k: usize = @intCast(j);
                    while (k < max_values - 1) : (k += 1) {
                        key.values[k] = key.values[k + 1];
                    }
                    // Clear the last value.
                    key.values[max_values - 1].value_idx = 0;
                    key.values[max_values - 1].value_num = 0;
                }
            }
            if (key.values[0].value_idx == 0) {
                // No values left, delete the key.
                log.dbg("%s: delete key %d at index %zu (no values left)\n", .{ func, i, key.key_idx });
                _ = map.keys.orderedRemove(@intCast(i));
                count_keys_del += 1;
            }
        }

        log.inf("%s: refresh map: idx_last_draft=%zu, new begin=%zu, #keys_checked=%zu, #keys_del=%zu, #values_del=%zu, #hashes_upd=%zu\n", .{ func, map.idx_last_check, size_begin, count_keys, count_keys_del, count_values_del, count_map_entries_upd });
    }

    map.idx_last_check = size_begin;
    map.size_last_begin = size_begin;
}

/// Ports `common_ngram_map_draft` (ngram-map.cpp:221 @d81235049): find the
/// key ending in `sampled`, update its statistics, and draft the m-gram
/// after it -- its first value with keys alone, otherwise the most frequent
/// value when it dominates the others.
///
/// Parameters:
/// - `map`: the map.
/// - `inp`: the tokens generated so far.
/// - `sampled`: the token just sampled.
/// - `draft_out`: empty on entry; the draft is appended, with `map.allocator`.
pub fn draft(map: *Map, inp: []const c.llama_token, sampled: c.llama_token, draft_out: *std.ArrayList(c.llama_token)) error{OutOfMemory}!void {
    const func: [*:0]const u8 = "common_ngram_map_draft";
    // reset last key and value.
    map.last_draft_created = false;
    map.last_draft_key_idx = 0;
    map.last_draft_value_idx = 0;

    const cur_len = inp.len;
    const n: u16 = map.size_key;
    const m: u16 = map.size_value;
    if (cur_len < @as(usize, @intCast(2 * @as(i32, n) + m))) {
        return;
    }
    if (cur_len >= @as(usize, std.math.maxInt(u32))) {
        // key_map uses uint32_t instead of size_t.
        _ = c.ggml_abort("ngram-map.cpp", 237, "%s: cur_len exceeds UINT32_MAX: %zu", func, cur_len);
    }

    if (map.idx_last_check > cur_len) {
        // Should not happen because of common_ngram_map_begin().
        _ = c.ggml_abort("ngram-map.cpp", 242, "%s: map.idx_last_check > cur_len: %zu > %zu", func, map.idx_last_check, cur_len);
    }
    map.idx_last_check = cur_len;

    // search pattern, the key n-gram
    var key_tokens: std.ArrayList(c.llama_token) = .empty;
    defer key_tokens.deinit(map.allocator);
    try key_tokens.ensureTotalCapacity(map.allocator, n);
    var jj = cur_len -% n +% 1;
    while (jj < cur_len) : (jj += 1) {
        try key_tokens.append(map.allocator, inp[jj]);
    }
    try key_tokens.append(map.allocator, sampled);

    // search for the key in the map
    var match_pos: usize = 0;
    if (map.size_last_begin > cur_len) {
        _ = c.ggml_abort("ngram-map.cpp", 257, "%s: map.size_last_begin > cur_len: %zu > %zu", func, map.size_last_begin, cur_len);
    }
    const nm1: usize = @as(usize, n) + m + 1;
    if (map.key_map.len != 0) {
        // Search for the key in the map key_map from hash of ngrams to index of ngram.
        const idx_hash: u32 = @intCast(mapHash(key_tokens.items, 0, n) % map.key_map.len);
        const idx_key = map.key_map[idx_hash];
        if (idx_key != 0 and idx_key < cur_len -% nm1) {
            // Check if the key matches the key at idx_key (because of possible collisions).
            var match = true;
            for (0..n) |k| {
                if (inp[idx_key + k] != key_tokens.items[k]) {
                    match = false;
                    break;
                }
            }
            log.dbg("%s: key hash %x -> idx_key %d: match %d\n", .{ func, idx_hash, idx_key, @as(c_int, if (match) 1 else 0) });
            if (match) {
                match_pos = idx_key;
            }
        }
    }
    if (match_pos == 0 and map.size_last_begin > nm1) {
        // Search for the key in [1, map.size_last_begin - n - m -1], descending.
        var j = map.size_last_begin - nm1;
        while (j > map.key_map_last_idx) : (j -= 1) {
            // Check if the key matches the key.
            if (matchAt(inp, j, key_tokens.items[0..n])) {
                match_pos = j;
                break;
            }
        }
    }
    if (match_pos == 0) {
        // In case of a reasoning chat, the part after size_last_begin may be deleted/reordered later.
        //
        // Search in [size_last_begin, cur_len - n - m - 1], descending.
        var j = cur_len -% nm1;
        while (j > map.size_last_begin and j > map.key_map_last_idx) : (j -= 1) {
            if (matchAt(inp, j, key_tokens.items[0..n])) {
                match_pos = j;
                break;
            }
        }
    }
    if (match_pos > 0) {
        log.dbg("%s: cur_len = %zu, n = %d, m = %d, sz_tkns = %zu, sampled = %d, match_pos = %zu\n", .{ func, cur_len, @as(c_int, n), @as(c_int, m), key_tokens.items.len, @as(c_int, sampled), match_pos });
    }

    if (map.key_map.len != 0) {
        // Add hashes of new ngrams in key_map.
        //
        // Use the same order as above.
        if (map.size_last_begin > nm1) {
            var j = map.size_last_begin - nm1;
            while (j > map.key_map_last_idx) : (j -= 1) {
                // compute hash and store index of ngram at idx j in the map.
                const idx_hash: usize = mapHash(inp, j, n) % map.key_map.len;
                if (map.key_map[idx_hash] == 0) {
                    map.key_map[idx_hash] = @truncate(j); // collisions may occur
                }
            }
        }

        var j = cur_len -% nm1;
        while (j > map.size_last_begin and j > map.key_map_last_idx) : (j -= 1) {
            // compute hash and store index of ngram at idx j in the map.
            const idx_hash: usize = mapHash(inp, j, n) % map.key_map.len;
            if (map.key_map[idx_hash] == 0) {
                map.key_map[idx_hash] = @truncate(j);
            }
        }
        map.key_map_last_idx = @max(@as(u32, @truncate(cur_len -% nm1)), map.key_map_last_idx);
    }

    if (match_pos == 0) {
        return;
    }

    // We have a match, now we look for the statistics of the key.
    var key_offset = map.keys.items.len; // offset in the map
    // We iterate through the std::vector<common_ngram_map_key> map->keys.
    for (map.keys.items, 0..) |key, i| {
        if (matchAt(inp, key.key_idx, key_tokens.items[0..n])) {
            key_offset = i;
            break;
        }
    }
    if (key_offset == map.keys.items.len) {
        // We create a new key-entry, it will get offset key_offset.
        var new_key: Key = .{ .key_idx = match_pos, .stat_idx = 0, .key_num = 0 };
        for (&new_key.values) |*v| {
            v.value_num = 0;
            v.n_accepted = @bitCast(m);
        }
        try map.keys.append(map.allocator, new_key);
    }

    // our key n-gram:
    const curr_key = &map.keys.items[key_offset];

    // update number of key hits
    curr_key.key_num = @intCast(@min(@as(i32, map.keys.items[key_offset].key_num) + 1, max_value_count));

    if (map.key_only) {
        // simple mode:
        // Fill in the draft with the m tokens following the key.
        // We work with value values[0] only.
        const n_draft_tokens: i32 = @min(@as(i32, m), @as(i32, curr_key.values[0].n_accepted));

        var i: i32 = 0;
        while (i < n_draft_tokens) : (i += 1) {
            try draft_out.append(map.allocator, inp[match_pos + n + @as(usize, @intCast(i))]);
        }

        log.dbg("%s: key_idx = %zu, key_offset = %zu, key_num = %d, draft.size = %zu\n", .{ func, curr_key.key_idx, key_offset, @as(c_int, curr_key.key_num), draft_out.items.len });

        map.last_draft_created = true;
        map.last_draft_key_idx = key_offset;
        map.last_draft_value_idx = 0; // value 0 is used for simple mode
        return;
    }

    if (curr_key.key_num < map.min_hits) {
        // not enough hits to consider this a good draft
        log.dbg("%s: key_offset = %zu, key_num = %d, min_hits = %d, no draft\n", .{ func, key_offset, @as(c_int, curr_key.key_num), @as(c_int, map.min_hits) });
        return;
    }

    // complex mode: examine the different m-grams after this key n-gram.
    //

    // determine all (max COMMON_NGRAM_MAX_VALUES) m-grams after the key n-gram.
    var i = curr_key.stat_idx;
    while (i <= match_pos) : (i += 1) {
        // begins the key n-gram at index i?
        if (!matchAt(inp, i, key_tokens.items[0..n])) {
            continue;
        }

        // Do we haven a existing value m-gram or a new one after the key at index i?
        const idx_begin_value_key = i + n;
        var idx_value: i32 = -1;
        for (0..max_values) |v| {
            const idx_begin_value_v = curr_key.values[v].value_idx;
            if (idx_begin_value_v == 0) {
                // We found an empty value slot => we found a new value m-gram after the key n-gram.
                curr_key.values[v].value_idx = idx_begin_value_key;
                curr_key.values[v].value_num = 0;
                curr_key.values[v].n_accepted = @bitCast(m);
                idx_value = @intCast(v);
                break;
            }
            var match = true;
            for (0..m) |j| {
                if (inp[idx_begin_value_key + j] != inp[idx_begin_value_v + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // We found an existing value m-gram after the key n-gram.
                idx_value = @intCast(v);
                break;
            }
        }
        if (idx_value >= 0) {
            // We found a value m-gram of the key n-gram.
            const vv = &curr_key.values[@intCast(idx_value)];
            vv.value_num = @intCast(@min(@as(i32, vv.value_num) + 1, max_value_count));
        }
    }
    // the statistics are updated up to match_pos.
    curr_key.stat_idx = match_pos;

    // Do we have a value we could use for the draft?
    var max_occur: u16 = 0;
    var slot_max: i32 = 0;
    for (0..max_values) |v| {
        const curr_occur = curr_key.values[v].value_num;
        if (curr_occur > max_occur) {
            max_occur = curr_occur;
            slot_max = @intCast(v);
        }
    }
    // What is sum of the other occurrences?
    var sum_occur: u32 = 0;
    for (0..max_values) |v| {
        if (v == slot_max) {
            continue;
        }
        const curr_occur = curr_key.values[v].value_num;
        sum_occur += curr_occur;
    }

    const vs = &curr_key.values;
    log.dbg("%s: key_offset = %zu, max_occur = %d, sum_occur = %d, slot_max = %d [%zu/%d, %zu/%d, %zu/%d, %zu/%d]\n", .{
        func,                        key_offset,                  @as(c_int, max_occur),       sum_occur,                   slot_max,
        vs[0].value_idx,             @as(c_int, vs[0].value_num), vs[1].value_idx,             @as(c_int, vs[1].value_num), vs[2].value_idx,
        @as(c_int, vs[2].value_num), vs[3].value_idx,             @as(c_int, vs[3].value_num),
    });
    // Print the tokens of the four values (if idx != 0), use LOG_INF
    if (log.verbosity_thold >= log.level_debug) {
        for (0..max_values) |v| {
            if (curr_key.values[v].value_idx != 0) {
                const s = try tokensToStr(map.allocator, inp, curr_key.values[v].value_idx, m);
                defer map.allocator.free(s);
                log.dbg("%s: value[%d] = %s\n", .{ func, @as(c_int, @intCast(v)), s.ptr });
            }
        }
    }

    if (sum_occur > 0 and max_occur < 2 * sum_occur) {
        // The most frequent value is not much more frequent than the other values.
        // We do not use the draft.
        return;
    }

    // We use the most frequent value values[slot_max] for the draft.
    // Fill in the draft with the m tokens following the key.
    const n_draft_tokens: i32 = @min(@as(i32, m), @as(i32, curr_key.values[@intCast(slot_max)].n_accepted));

    var k: i32 = 0;
    while (k < n_draft_tokens) : (k += 1) {
        try draft_out.append(map.allocator, inp[match_pos + n + @as(usize, @intCast(k))]);
    }

    log.dbg("%s: key_offset = %zu, slot_max = %d, key_num = %d, draft.size = %zu\n", .{ func, key_offset, slot_max, @as(c_int, curr_key.key_num), draft_out.items.len });

    map.last_draft_created = true;
    map.last_draft_key_idx = key_offset;
    map.last_draft_value_idx = @intCast(slot_max); // value used for draft generation.
}

/// Ports `common_ngram_map_accept` (ngram-map.cpp:518 @d81235049): record
/// how many tokens of the last draft were accepted, which caps the next
/// draft from the same value.
///
/// Parameters:
/// - `map`: the map.
/// - `n_accepted`: the accepted count; stored as `int16_t`, truncated.
pub fn accept(map: *Map, n_accepted: u16) void {
    if (!map.last_draft_created) {
        return;
    }

    // find the key and its chosen value.
    const key_idx = map.last_draft_key_idx;
    const val_idx = map.last_draft_value_idx;

    // find key corresponding to key_idx.
    const curr_key = &map.keys.items[key_idx];
    // find value corresponding to val_idx.
    const curr_value = &curr_key.values[val_idx]; // value used for draft generation.

    // update the value statistics
    log.dbg("common_ngram_map_send_accepted: n_accepted = %d, prev value_num = %d\n", .{ @as(c_int, n_accepted), @as(c_int, curr_value.n_accepted) });
    curr_value.n_accepted = @bitCast(n_accepted);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the map hash wraps in 32 bits" {
    const toks = [_]c.llama_token{ -1, 3 };
    const want: u32 = (@as(u32, 0) *% lcg_factor +% 0xffffffff) *% lcg_factor +% 3;
    try std.testing.expectEqual(want, mapHash(&toks, 0, 2));
}

test "simple draft copies what followed the last n-gram" {
    const a = std.testing.allocator;
    const toks = [_]c.llama_token{ 9, 1, 2, 3, 4, 5, 6, 7, 1 };
    var d = try simpleDraft(a, .{ .size_ngram = 2, .size_mgram = 3 }, &toks, 2);
    defer d.deinit(a);
    try std.testing.expectEqualSlices(c.llama_token, &.{ 3, 4, 5 }, d.items);
    const str = try tokensToStr(a, &toks, 1, 3);
    defer a.free(str);
    try std.testing.expectEqualStrings("[1, 2, 3]", str);
}

test "the keyed map drafts, records acceptance, and forgets on shrink" {
    const a = std.testing.allocator;
    var map = try Map.init(a, 2, 2, true, 1);
    defer map.deinit();
    const toks = [_]c.llama_token{ 0, 1, 2, 3, 4, 8, 9, 1 };
    begin(&map, toks[0..0]);
    var d: std.ArrayList(c.llama_token) = .empty;
    defer d.deinit(a);
    try draft(&map, &toks, 2, &d);
    try std.testing.expectEqualSlices(c.llama_token, &.{ 3, 4 }, d.items);
    try std.testing.expect(map.last_draft_created);
    accept(&map, 1);
    try std.testing.expectEqual(@as(i16, 1), map.keys.items[0].values[0].n_accepted);
    var copy = try map.clone(a);
    defer copy.deinit();
    try std.testing.expectEqual(@as(usize, 1), copy.keys.items.len);
    begin(&map, toks[0..1]);
    try std.testing.expectEqual(@as(usize, 0), map.keys.items.len);
}
