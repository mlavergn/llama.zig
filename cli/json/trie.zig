//! `common_trie`: a trie over Unicode code points, and the two UTF-8
//! helpers it is built with.
//!
//! # Provenance
//!
//! Ports `common_trie` from `llama.cpp/common/trie.h` and
//! `llama.cpp/common/trie.cpp`, and `common_parse_utf8_codepoint` and
//! `common_unicode_cpt_to_utf8` from `llama.cpp/common/unicode.cpp`
//! (llama.cpp v0.9.0, `cd959bae9`). Each declaration names what it ports
//! and the line it began at.
//!
//! # Scope
//!
//! All of `trie.cpp`: what `json-schema-to-grammar.cpp`'s `_not_strings`
//! reaches -- building the trie and walking its nodes -- and what the PEG
//! parser does (`cli/peg/`): `check_at` for `until`, and the Aho-Corasick
//! automaton its GBNF export emits.
//!
//! A node's children are a `std::map<uint32_t, size_t>`, iterated in key
//! order, and `_not_strings` writes them in that order; here they are a
//! list kept sorted by code point.

const std = @import("std");

/// Ports `utf8_parse_result` (common/unicode.h:10 @cd959bae9). Since
/// v0.5.0 `bytes_consumed` is also set when decoding fails: it is then the
/// length of the valid prefix, at least one byte, which the PEG `until`
/// consumes as one invalid run.
pub const Utf8Parse = struct {
    pub const Status = enum { success, incomplete, invalid };

    codepoint: u32 = 0,
    bytes_consumed: usize = 0,
    status: Status,
};

/// Ports `common_parse_utf8_codepoint` (common/unicode.cpp:17 @cd959bae9). It
/// checks the lead byte's shape and the continuation bytes' tags and nothing
/// else: overlong forms, surrogates and lead bytes up to 0xF7 decode. A
/// longer sequence is checked one byte at a time, so a bad continuation byte
/// is reported as `invalid` before a short input is `incomplete`.
///
/// Parameters:
/// - `input`: the text.
/// - `offset`: where the code point starts.
///
/// Return: the code point and its length, or why there is none and the
/// length of the valid prefix (zero only at the end of the input).
pub fn parseUtf8Codepoint(input: []const u8, offset: usize) Utf8Parse {
    if (offset >= input.len) return .{ .status = .incomplete };
    const b0: u32 = input[offset];
    if (b0 & 0x80 == 0) return .{ .status = .success, .codepoint = b0, .bytes_consumed = 1 };
    if (b0 & 0x40 == 0) return .{ .status = .invalid, .bytes_consumed = 1 };
    if (b0 & 0x20 == 0) {
        if (offset + 1 >= input.len) return .{ .status = .incomplete, .bytes_consumed = 1 };
        const b1: u32 = input[offset + 1];
        if (b1 & 0xc0 != 0x80) return .{ .status = .invalid, .bytes_consumed = 1 };
        return .{ .status = .success, .codepoint = ((b0 & 0x1f) << 6) | (b1 & 0x3f), .bytes_consumed = 2 };
    }
    if (b0 & 0x10 == 0) {
        if (checkContinuation(input, offset, 3)) |bad| return bad;
        const b1: u32 = input[offset + 1];
        const b2: u32 = input[offset + 2];
        return .{ .status = .success, .codepoint = ((b0 & 0x0f) << 12) | ((b1 & 0x3f) << 6) | (b2 & 0x3f), .bytes_consumed = 3 };
    }
    if (b0 & 0x08 == 0) {
        if (checkContinuation(input, offset, 4)) |bad| return bad;
        const b1: u32 = input[offset + 1];
        const b2: u32 = input[offset + 2];
        const b3: u32 = input[offset + 3];
        return .{
            .status = .success,
            .codepoint = ((b0 & 0x07) << 18) | ((b1 & 0x3f) << 12) | ((b2 & 0x3f) << 6) | (b3 & 0x3f),
            .bytes_consumed = 4,
        };
    }
    return .{ .status = .invalid, .bytes_consumed = 1 };
}

/// The `for (size_t i = 1; i < n; i++)` loop the three- and four-byte arms
/// of `parseUtf8Codepoint` above each spell out.
///
/// Return: null when all `n - 1` continuation bytes are present and tagged,
/// otherwise the failure and the valid prefix's length.
fn checkContinuation(input: []const u8, offset: usize, n: usize) ?Utf8Parse {
    var i: usize = 1;
    while (i < n) : (i += 1) {
        if (offset + i >= input.len) return .{ .status = .incomplete, .bytes_consumed = i };
        if (input[offset + i] & 0xc0 != 0x80) return .{ .status = .invalid, .bytes_consumed = i };
    }
    return null;
}

/// Ports `common_unicode_cpt_to_utf8` (common/unicode.cpp:99 @cd959bae9).
///
/// Parameters:
/// - `cpt`: the code point.
/// - `buf`: receives the bytes.
///
/// Return: the encoded bytes, a slice of `buf`, or `error.InvalidCodepoint`
/// above U+10FFFF, where the C++ throws `std::invalid_argument("invalid
/// codepoint")`.
pub fn cptToUtf8(cpt: u32, buf: *[4]u8) error{InvalidCodepoint}![]const u8 {
    if (cpt <= 0x7f) {
        buf[0] = @intCast(cpt);
        return buf[0..1];
    }
    if (cpt <= 0x7ff) {
        buf[0] = @intCast(0xc0 | ((cpt >> 6) & 0x1f));
        buf[1] = @intCast(0x80 | (cpt & 0x3f));
        return buf[0..2];
    }
    if (cpt <= 0xffff) {
        buf[0] = @intCast(0xe0 | ((cpt >> 12) & 0x0f));
        buf[1] = @intCast(0x80 | ((cpt >> 6) & 0x3f));
        buf[2] = @intCast(0x80 | (cpt & 0x3f));
        return buf[0..3];
    }
    if (cpt <= 0x10ffff) {
        buf[0] = @intCast(0xf0 | ((cpt >> 18) & 0x07));
        buf[1] = @intCast(0x80 | ((cpt >> 12) & 0x3f));
        buf[2] = @intCast(0x80 | ((cpt >> 6) & 0x3f));
        buf[3] = @intCast(0x80 | (cpt & 0x3f));
        return buf[0..4];
    }
    return error.InvalidCodepoint;
}

/// Ports `common_trie` (trie.h:12 @cd959bae9).
pub const Trie = struct {
    /// One child edge: the code point and the node it leads to.
    pub const Edge = struct { cpt: u32, child: usize };

    /// Ports `node` (trie.h:13 @cd959bae9), a member of `common_trie`.
    pub const Node = struct {
        /// sorted by `cpt`, as the `std::map` iterates
        children: std.ArrayList(Edge) = .empty,
        /// index of the pattern ending here, -1 if none
        pattern: i32 = -1,

        fn find(self: *const Node, cpt: u32) ?usize {
            for (self.children.items) |e| {
                if (e.cpt == cpt) return e.child;
            }
            return null;
        }
    };

    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    n_patterns: i32 = 0,

    /// Ports `common_trie` (trie.h:24 @cd959bae9), the constructor taking
    /// words: a root, then every word inserted in order.
    ///
    /// Parameters:
    /// - `allocator`: owns the nodes.
    /// - `words`: inserted in order.
    ///
    /// Return: the trie, freed with `deinit`.
    pub fn init(allocator: std.mem.Allocator, words: []const []const u8) error{OutOfMemory}!Trie {
        var t: Trie = .{ .allocator = allocator };
        errdefer t.deinit();
        _ = try t.createNode();
        for (words) |w| _ = try t.insertWord(w);
        return t;
    }

    /// Frees the nodes.
    pub fn deinit(self: *Trie) void {
        for (self.nodes.items) |*n| n.children.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
    }

    /// Ports `match_result` (trie.h:30 @cd959bae9), a member of `common_trie`.
    pub const MatchResult = enum { no_match, partial_match, complete_match };

    /// Ports `check_at` (trie.cpp:7 @cd959bae9): whether a word starts at
    /// `start_pos` -- complete as soon as one ends, partial when the input
    /// ends inside the trie. A code point that fails to decode ends the walk
    /// as the end of input does.
    ///
    /// Parameters:
    /// - `sv`: the text.
    /// - `start_pos`: where to look.
    ///
    /// Return: the match.
    pub fn checkAt(self: *const Trie, sv: []const u8, start_pos: usize) MatchResult {
        var current: usize = 0;
        var pos = start_pos;
        while (pos < sv.len) {
            const r = parseUtf8Codepoint(sv, pos);
            if (r.status != .success) break;
            current = self.nodes.items[current].find(r.codepoint) orelse return .no_match;
            pos += r.bytes_consumed;
            if (self.nodes.items[current].pattern >= 0) return .complete_match;
        }
        if (current != 0) return .partial_match;
        return .no_match;
    }

    /// Ports `create_node` (trie.h:45 @cd959bae9).
    fn createNode(self: *Trie) error{OutOfMemory}!usize {
        const index = self.nodes.items.len;
        try self.nodes.append(self.allocator, .{});
        return index;
    }

    /// Ports `insert` (trie.cpp:44 @cd959bae9), the overload taking a
    /// string: its code points up to the first that fails to decode.
    ///
    /// Return: the word's pattern index.
    pub fn insertWord(self: *Trie, word: []const u8) error{OutOfMemory}!i32 {
        var symbols: std.ArrayList(u32) = .empty;
        defer symbols.deinit(self.allocator);
        var pos: usize = 0;
        while (pos < word.len) {
            const r = parseUtf8Codepoint(word, pos);
            if (r.status != .success) break;
            try symbols.append(self.allocator, r.codepoint);
            pos += r.bytes_consumed;
        }
        return self.insertSymbols(symbols.items);
    }

    /// Ports `insert` (trie.cpp:59 @cd959bae9), the overload taking code
    /// points.
    ///
    /// Return: the pattern index; a duplicate keeps the first one's.
    pub fn insertSymbols(self: *Trie, symbols: []const u32) error{OutOfMemory}!i32 {
        var current: usize = 0;
        for (symbols) |ch| {
            if (self.nodes.items[current].find(ch)) |child| {
                current = child;
            } else {
                const child = try self.createNode();
                const kids = &self.nodes.items[current].children;
                var at: usize = 0;
                while (at < kids.items.len and kids.items[at].cpt < ch) at += 1;
                try kids.insert(self.allocator, at, .{ .cpt = ch, .child = child });
                current = child;
            }
        }
        if (self.nodes.items[current].pattern < 0) {
            self.nodes.items[current].pattern = self.n_patterns;
            self.n_patterns += 1;
        }
        return self.nodes.items[current].pattern;
    }
};

/// Ports `common_aho_corasick` (trie.h:53 @cd959bae9): the automaton over a
/// trie's words, with failure links, the longest word ending at each state,
/// and every code point that has a transition.
pub const AhoCorasick = struct {
    t: Trie,
    /// failure links
    fail: []usize,
    /// states in breadth-first order
    order: []usize,
    /// the longest pattern ending at each state, -1 if none
    match: []i32,
    /// every code point with a transition, ascending, as `std::set` holds it
    alphabet: []u32,

    /// Ports `common_aho_corasick` (trie.cpp:77 @cd959bae9), the
    /// constructor: the trie of `strings` and its automaton.
    ///
    /// Parameters:
    /// - `allocator`: owns everything.
    /// - `strings`: the patterns, inserted in order.
    ///
    /// Return: the automaton, freed with `deinit`.
    pub fn init(allocator: std.mem.Allocator, strings: []const []const u8) error{OutOfMemory}!AhoCorasick {
        var t = try Trie.init(allocator, strings);
        errdefer t.deinit();
        const nodes = t.nodes.items;
        const n = nodes.len;

        const fail = try allocator.alloc(usize, n);
        errdefer allocator.free(fail);
        @memset(fail, 0);
        var order: std.ArrayList(usize) = .empty;
        errdefer order.deinit(allocator);
        try order.ensureTotalCapacityPrecise(allocator, n);

        // a FIFO over `order` itself: every state is queued exactly once
        var queue: std.ArrayList(usize) = .empty;
        defer queue.deinit(allocator);
        try queue.append(allocator, 0);
        var head: usize = 0;
        while (head < queue.items.len) {
            const u = queue.items[head];
            head += 1;
            order.appendAssumeCapacity(u);
            for (nodes[u].children.items) |e| {
                const v = e.child;
                if (u != 0) {
                    var f = fail[u];
                    while (f != 0 and nodes[f].find(e.cpt) == null) f = fail[f];
                    const it = nodes[f].find(e.cpt);
                    fail[v] = if (it != null and it.? != v) it.? else 0;
                }
                try queue.append(allocator, v);
            }
        }

        const match = try allocator.alloc(i32, n);
        errdefer allocator.free(match);
        @memset(match, -1);
        for (order.items) |u| {
            match[u] = if (nodes[u].pattern >= 0) nodes[u].pattern else if (u != 0) match[fail[u]] else -1;
        }

        var alphabet: std.ArrayList(u32) = .empty;
        errdefer alphabet.deinit(allocator);
        for (nodes) |node| {
            for (node.children.items) |e| {
                var at: usize = 0;
                while (at < alphabet.items.len and alphabet.items[at] < e.cpt) at += 1;
                if (at < alphabet.items.len and alphabet.items[at] == e.cpt) continue;
                try alphabet.insert(allocator, at, e.cpt);
            }
        }

        return .{
            .t = t,
            .fail = fail,
            .order = try order.toOwnedSlice(allocator),
            .match = match,
            .alphabet = try alphabet.toOwnedSlice(allocator),
        };
    }

    /// Frees the trie and the tables.
    pub fn deinit(self: *AhoCorasick) void {
        const a = self.t.allocator;
        a.free(self.fail);
        a.free(self.order);
        a.free(self.match);
        a.free(self.alphabet);
        self.t.deinit();
    }

    /// Ports `num_states` (trie.h:65 @cd959bae9).
    pub fn numStates(self: *const AhoCorasick) usize {
        return self.t.nodes.items.len;
    }

    /// Ports `is_terminal` (trie.h:66 @cd959bae9).
    pub fn isTerminal(self: *const AhoCorasick, s: usize) bool {
        return self.match[s] >= 0;
    }

    /// Ports `match_pattern` (trie.h:69 @cd959bae9).
    pub fn matchPattern(self: *const AhoCorasick, s: usize) i32 {
        return self.match[s];
    }

    /// Ports `next` (trie.cpp:116 @cd959bae9): the transition on `ch`,
    /// following failure links until one exists.
    ///
    /// Return: the next state; the root when nothing matches.
    pub fn next(self: *const AhoCorasick, state_in: usize, ch: u32) usize {
        const nodes = self.t.nodes.items;
        var state = state_in;
        while (state != 0 and nodes[state].find(ch) == null) state = self.fail[state];
        return nodes[state].find(ch) orelse 0;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the trie keeps children in code point order and stops at bad UTF-8" {
    var t = try Trie.init(std.testing.allocator, &.{ "b", "a\xc3\xa9", "a", "x\xff" });
    defer t.deinit();
    const root = t.nodes.items[0];
    try std.testing.expectEqual(@as(usize, 3), root.children.items.len);
    try std.testing.expectEqual(@as(u32, 'a'), root.children.items[0].cpt);
    try std.testing.expectEqual(@as(u32, 'x'), root.children.items[2].cpt);
    const a = t.nodes.items[root.children.items[0].child];
    try std.testing.expectEqual(@as(i32, 2), a.pattern);
    try std.testing.expectEqual(@as(u32, 0xe9), a.children.items[0].cpt);
    var buf: [4]u8 = undefined;
    try std.testing.expectEqualStrings("\xf4\x8f\xbf\xbf", try cptToUtf8(0x10ffff, &buf));
    try std.testing.expectError(error.InvalidCodepoint, cptToUtf8(0x110000, &buf));
}

test "check_at and the automaton" {
    var t = try Trie.init(std.testing.allocator, &.{ "</a>", "</b" });
    defer t.deinit();
    try std.testing.expectEqual(Trie.MatchResult.complete_match, t.checkAt("x</a>", 1));
    try std.testing.expectEqual(Trie.MatchResult.partial_match, t.checkAt("x</", 1));
    try std.testing.expectEqual(Trie.MatchResult.no_match, t.checkAt("x</c", 1));
    try std.testing.expectEqual(Trie.MatchResult.no_match, t.checkAt("x", 1));

    var ac = try AhoCorasick.init(std.testing.allocator, &.{ "he", "she", "hers" });
    defer ac.deinit();
    var s: usize = 0;
    for ("ushe") |c| s = ac.next(s, c);
    // "she" ends here, and through its failure link so does "he"
    try std.testing.expectEqual(@as(i32, 1), ac.matchPattern(s));
    try std.testing.expectEqualSlices(u32, &.{ 'e', 'h', 'r', 's' }, ac.alphabet);
}
