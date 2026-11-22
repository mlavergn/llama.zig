//! TTS normalization patterns: the small matching language a task
//! template (e.g. "tts") normalizes the text to speak with, through the two
//! functions `tts_norm_scan(text, rules)` and `tts_norm_sub(text, pattern,
//! repl)`. It is not a regex engine: PEG-style, alternatives are ordered and
//! quantifiers greedy and possessive, so nothing backtracks.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/chat-tts-norm.cpp` and
//! `llama.cpp/common/chat-tts-norm.h` (llama.cpp v1.0.0, `b1b770d90`):
//! the character classes, the UTF-8 decoder, the pattern parser and
//! matcher, `common_tts_norm_pattern`, `common_tts_norm_scan`,
//! `common_tts_norm_sub` and `common_chat_tts_norm_add_builtins` with its two
//! builtins. Each declaration names what it ports and the line it began at.
//!
//! # Ownership
//!
//! Every function takes an arena `a` and returns what it built in it.
//! A malformed pattern or replacement is `error.RuntimeError`, its
//! `std::runtime_error` text in `exception.what`.

const std = @import("std");
const mod = @import("module.zig");
const jinja = @import("zigjinja");

// -----------------------------------------------------------------------------
// character classes

const Range = struct { lo: u32, hi: u32 };

/// Ports `TTS_NORM_SPACE` (chat-tts-norm.cpp:28 @b1b770d90).
const space_ranges = [_]Range{
    .{ .lo = 0x0009, .hi = 0x000D }, .{ .lo = 0x0020, .hi = 0x0020 }, .{ .lo = 0x0085, .hi = 0x0085 }, .{ .lo = 0x00A0, .hi = 0x00A0 }, .{ .lo = 0x1680, .hi = 0x1680 },
    .{ .lo = 0x2000, .hi = 0x200A }, .{ .lo = 0x2028, .hi = 0x2029 }, .{ .lo = 0x202F, .hi = 0x202F }, .{ .lo = 0x205F, .hi = 0x205F }, .{ .lo = 0x3000, .hi = 0x3000 },
};

/// Ports `TTS_NORM_PUNCT` (chat-tts-norm.cpp:33 @b1b770d90).
const punct_ranges = [_]Range{
    .{ .lo = 0x0021, .hi = 0x002F }, .{ .lo = 0x003A, .hi = 0x0040 }, .{ .lo = 0x005B, .hi = 0x0060 }, .{ .lo = 0x007B, .hi = 0x007E },
    .{ .lo = 0x00A1, .hi = 0x00A9 }, .{ .lo = 0x00AB, .hi = 0x00B4 }, .{ .lo = 0x00B6, .hi = 0x00B9 }, .{ .lo = 0x00BB, .hi = 0x00BF },
    .{ .lo = 0x00D7, .hi = 0x00D7 }, .{ .lo = 0x00F7, .hi = 0x00F7 }, .{ .lo = 0x2010, .hi = 0x2027 }, .{ .lo = 0x2030, .hi = 0x205E },
    .{ .lo = 0x20A0, .hi = 0x20CF }, .{ .lo = 0x2100, .hi = 0x218F }, .{ .lo = 0x2190, .hi = 0x2BFF }, .{ .lo = 0x3001, .hi = 0x3003 },
    .{ .lo = 0x3008, .hi = 0x3020 }, .{ .lo = 0x3030, .hi = 0x3030 }, .{ .lo = 0x303D, .hi = 0x303D }, .{ .lo = 0x30FB, .hi = 0x30FB },
    .{ .lo = 0xFE10, .hi = 0xFE19 }, .{ .lo = 0xFE30, .hi = 0xFE4F }, .{ .lo = 0xFE50, .hi = 0xFE6B }, .{ .lo = 0xFF01, .hi = 0xFF0F },
    .{ .lo = 0xFF1A, .hi = 0xFF20 }, .{ .lo = 0xFF3B, .hi = 0xFF40 }, .{ .lo = 0xFF5B, .hi = 0xFF65 }, .{ .lo = 0x1F000, .hi = 0x1FAFF },
};

/// Ports `TTS_NORM_CONTROL` (chat-tts-norm.cpp:43 @b1b770d90).
const control_ranges = [_]Range{
    .{ .lo = 0x0000, .hi = 0x001F }, .{ .lo = 0x007F, .hi = 0x009F }, .{ .lo = 0x200B, .hi = 0x200F }, .{ .lo = 0x2060, .hi = 0x206F }, .{ .lo = 0xFEFF, .hi = 0xFEFF },
};

/// Ports `TTS_NORM_HAN` (chat-tts-norm.cpp:47 @b1b770d90).
const han_ranges = [_]Range{
    .{ .lo = 0x2E80, .hi = 0x2FDF }, .{ .lo = 0x3005, .hi = 0x3005 }, .{ .lo = 0x3007, .hi = 0x3007 }, .{ .lo = 0x3021, .hi = 0x3029 },   .{ .lo = 0x3038, .hi = 0x303B },
    .{ .lo = 0x3400, .hi = 0x4DBF }, .{ .lo = 0x4E00, .hi = 0x9FFF }, .{ .lo = 0xF900, .hi = 0xFAFF }, .{ .lo = 0x20000, .hi = 0x323AF },
};

/// Ports `tts_norm_in` (chat-tts-norm.cpp:53 @b1b770d90).
fn in(ranges: []const Range, c: u32) bool {
    for (ranges) |r| {
        if (c >= r.lo and c <= r.hi) return true;
    }
    return false;
}

/// Ports `tts_norm_is_alpha` (chat-tts-norm.cpp:62 @b1b770d90).
fn isAlpha(c: u32) bool {
    if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z')) return true;
    if (c == 0xAA or c == 0xB5 or c == 0xBA) return true;
    return c >= 0xC0 and !in(&space_ranges, c) and !in(&punct_ranges, c) and !in(&control_ranges, c);
}

/// Ports `tts_norm_class_match` (chat-tts-norm.cpp:73 @b1b770d90): `kind`
/// is one of "dspawulhc".
fn classMatch(kind: u8, c: u32) bool {
    return switch (kind) {
        'd' => c >= '0' and c <= '9',
        's' => in(&space_ranges, c),
        'p' => in(&punct_ranges, c),
        'a' => isAlpha(c),
        'w' => isAlpha(c) or (c >= '0' and c <= '9'),
        'u' => c >= 'A' and c <= 'Z',
        'l' => c >= 'a' and c <= 'z',
        'h' => in(&han_ranges, c),
        'c' => in(&control_ranges, c),
        else => false,
    };
}

/// Ports `tts_norm_is_class_letter` (chat-tts-norm.cpp:88 @b1b770d90).
fn isClassLetter(c_in: u32) bool {
    var c = c_in;
    if (c >= 'A' and c <= 'Z') c += 'a' - 'A';
    return c < 0x80 and std.mem.indexOfScalar(u8, "dspawulhc", @intCast(c)) != null;
}

/// Ports `tts_norm_ascii_lower` (chat-tts-norm.cpp:95 @b1b770d90).
fn asciiLower(c: u32) u32 {
    return if (c >= 'A' and c <= 'Z') c + ('a' - 'A') else c;
}

/// Ports `tts_norm_ascii_swapcase` (chat-tts-norm.cpp:99 @b1b770d90).
fn asciiSwapcase(c: u32) u32 {
    if (c >= 'A' and c <= 'Z') return c + ('a' - 'A');
    if (c >= 'a' and c <= 'z') return c - ('a' - 'A');
    return c;
}

/// Ports `tts_norm_is_ascii_alnum` (chat-tts-norm.cpp:105 @b1b770d90).
fn isAsciiAlnum(c: u32) bool {
    return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
}

/// A text as code points, with each one's byte offset.
pub const Decoded = struct {
    cpts: []u32,
    /// `offs[i]` is the byte offset of code point `i`, `offs[n]` the text's size
    offs: []usize,
};

/// Ports `tts_norm_decode_utf8` (chat-tts-norm.cpp:110 @b1b770d90): invalid
/// bytes are decoded as U+FFFD, one per byte.
pub fn decodeUtf8(a: std.mem.Allocator, s: []const u8) mod.exception.Error!Decoded {
    var cpts: std.ArrayList(u32) = .empty;
    var offs: std.ArrayList(usize) = .empty;
    var i: usize = 0;
    const n = s.len;
    while (i < n) {
        try offs.append(a, i);
        const c = s[i];
        var cp: u32 = 0;
        var len: usize = 0;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        }
        var ok = len > 0 and i + len <= n;
        var k: usize = 1;
        while (ok and k < len) : (k += 1) {
            const cc = s[i + k];
            if ((cc & 0xC0) != 0x80) {
                ok = false;
            } else {
                cp = (cp << 6) | (cc & 0x3F);
            }
        }
        if (!ok) {
            try cpts.append(a, 0xFFFD);
            i += 1;
        } else {
            try cpts.append(a, cp);
            i += len;
        }
    }
    try offs.append(a, n);
    return .{ .cpts = cpts.items, .offs = offs.items };
}

// -----------------------------------------------------------------------------
// AST

/// Ports `tts_norm_op` (chat-tts-norm.cpp:153 @b1b770d90).
const Op = enum {
    /// one codepoint
    literal,
    /// one codepoint from a set; class escapes like %d are sets with a single class
    set,
    /// any codepoint
    any,
    /// ^
    start,
    /// $
    end,
    /// %b
    boundary,
    /// (...), (?:...)
    group,
    /// (?=...), (?!...)
    lookahead,
    /// (?<=c), (?<!c) - exactly one codepoint
    lookbehind,
};

/// Ports `tts_norm_class` (chat-tts-norm.cpp:165 @b1b770d90).
const Class = struct { kind: u8, neg: bool };

/// Ports `tts_norm_set` (chat-tts-norm.cpp:170 @b1b770d90).
const Set = struct {
    neg: bool = false,
    ranges: std.ArrayList(Range) = .empty,
    classes: std.ArrayList(Class) = .empty,
};

const Seq = []Item;

/// Ports `tts_norm_pattern_node` (chat-tts-norm.cpp:179 @b1b770d90).
const PatternNode = struct {
    op: Op = .group,
    /// literal
    cp: u32 = 0,
    /// set
    set: Set = .{},
    /// group, lookahead; lookbehind holds a single item
    alts: []Seq = &.{},
    /// group: capture index, -1 for non-capturing
    capture: c_int = -1,
    /// lookahead, lookbehind
    neg: bool = false,
};

/// Ports `TTS_NORM_INF` (chat-tts-norm.cpp:188 @b1b770d90).
const inf: c_int = std.math.maxInt(c_int);
/// Ports `TTS_NORM_MAX_REPEAT` (chat-tts-norm.cpp:189 @b1b770d90).
const max_repeat: c_int = 1000;
/// Ports `TTS_NORM_MAX_PATTERN_LEN` (chat-tts-norm.cpp:190 @b1b770d90).
const max_pattern_len: usize = 4096;

/// Ports `tts_norm_item` (chat-tts-norm.cpp:192 @b1b770d90).
const Item = struct {
    atom: PatternNode,
    min: c_int = 1,
    max: c_int = 1,
};

/// Ports `tts_norm_is_zero_width` (chat-tts-norm.cpp:198 @b1b770d90).
fn isZeroWidth(op: Op) bool {
    return op == .start or op == .end or op == .boundary or op == .lookahead or op == .lookbehind;
}

// -----------------------------------------------------------------------------
// parser

/// Ports `tts_norm_parser` (chat-tts-norm.cpp:207 @b1b770d90).
const Parser = struct {
    a: std.mem.Allocator,
    s: []const u32,
    p: usize = 0,
    n_caps: *usize,

    /// Ports `fail` (chat-tts-norm.cpp:214 @b1b770d90).
    fn fail(self: *const Parser, msg: []const u8) mod.exception.Error {
        return mod.exception.throw(error.RuntimeError, "invalid TTS norm pattern: {s} at position {d}", .{ msg, self.p });
    }

    fn eof(self: *const Parser) bool {
        return self.p >= self.s.len;
    }

    fn peek(self: *const Parser) u32 {
        return self.s[self.p];
    }

    fn peekIs(self: *const Parser, c: u32) bool {
        return !self.eof() and self.s[self.p] == c;
    }

    /// Ports `expect` (chat-tts-norm.cpp:222 @b1b770d90).
    fn expect(self: *Parser, c: u32, what: []const u8) mod.exception.Error!void {
        if (!self.peekIs(c)) {
            return self.fail(try std.mem.concat(self.a, u8, &.{ "expected ", what }));
        }
        self.p += 1;
    }

    /// Ports `parse_alts` (chat-tts-norm.cpp:229 @b1b770d90).
    fn parseAlts(self: *Parser) mod.exception.Error![]Seq {
        var alts: std.ArrayList(Seq) = .empty;
        try alts.append(self.a, try self.parseSeq());
        while (self.peekIs('|')) {
            self.p += 1;
            try alts.append(self.a, try self.parseSeq());
        }
        return alts.items;
    }

    /// Ports `parse_seq` (chat-tts-norm.cpp:239 @b1b770d90).
    fn parseSeq(self: *Parser) mod.exception.Error!Seq {
        var seq: std.ArrayList(Item) = .empty;
        while (!self.eof() and self.peek() != '|' and self.peek() != ')') {
            var item: Item = .{ .atom = try self.parseAtom() };
            try self.parseQuantifier(&item);
            try seq.append(self.a, item);
        }
        return seq.items;
    }

    /// Ports `parse_int` (chat-tts-norm.cpp:250 @b1b770d90).
    fn parseInt(self: *Parser) mod.exception.Error!c_int {
        if (self.eof() or self.peek() < '0' or self.peek() > '9') {
            return self.fail("expected number in quantifier");
        }
        var v: c_int = 0;
        while (!self.eof() and self.peek() >= '0' and self.peek() <= '9') {
            v = v * 10 + @as(c_int, @intCast(self.peek() - '0'));
            if (v > max_repeat) {
                return self.fail("repetition count exceeds " ++ std.fmt.comptimePrint("{d}", .{max_repeat}));
            }
            self.p += 1;
        }
        return v;
    }

    /// Ports `parse_quantifier` (chat-tts-norm.cpp:265 @b1b770d90).
    fn parseQuantifier(self: *Parser, item: *Item) mod.exception.Error!void {
        if (self.eof()) return;
        const c = self.peek();
        if (c != '?' and c != '*' and c != '+' and c != '{') return;
        if (isZeroWidth(item.atom.op)) {
            return self.fail("zero-width assertion cannot be repeated");
        }
        self.p += 1;
        if (c == '?') {
            item.min = 0;
            item.max = 1;
        } else if (c == '*') {
            item.min = 0;
            item.max = inf;
        } else if (c == '+') {
            item.min = 1;
            item.max = inf;
        } else {
            item.min = try self.parseInt();
            item.max = item.min;
            if (self.peekIs(',')) {
                self.p += 1;
                item.max = if (self.peekIs('}')) inf else try self.parseInt();
            }
            try self.expect('}', "'}' to close quantifier");
            if (item.max < item.min) {
                return self.fail("quantifier max is smaller than min");
            }
        }
        if (!self.eof() and (self.peek() == '?' or self.peek() == '*' or self.peek() == '+' or self.peek() == '{')) {
            return self.fail("multiple quantifiers");
        }
    }

    /// Ports `parse_group` (chat-tts-norm.cpp:300 @b1b770d90); '(' already
    /// consumed.
    fn parseGroup(self: *Parser) mod.exception.Error!PatternNode {
        var node: PatternNode = .{ .op = .group };
        if (self.peekIs('?')) {
            self.p += 1;
            if (self.eof()) return self.fail("unterminated group");
            const c = self.peek();
            if (c == ':') {
                self.p += 1;
            } else if (c == '=' or c == '!') {
                self.p += 1;
                node.op = .lookahead;
                node.neg = c == '!';
            } else if (c == '<' and self.p + 1 < self.s.len and (self.s[self.p + 1] == '=' or self.s[self.p + 1] == '!')) {
                node.op = .lookbehind;
                node.neg = self.s[self.p + 1] == '!';
                self.p += 2;
                if (self.eof() or self.peek() == ')') {
                    return self.fail("lookbehind must contain exactly one codepoint matcher");
                }
                const atom = try self.parseAtom();
                if (atom.op != .literal and atom.op != .set and atom.op != .any) {
                    return self.fail("lookbehind must contain exactly one codepoint matcher");
                }
                if (!self.peekIs(')')) {
                    return self.fail("lookbehind must contain exactly one codepoint matcher");
                }
                self.p += 1;
                const seq = try self.a.alloc(Item, 1);
                seq[0] = .{ .atom = atom };
                const alts = try self.a.alloc(Seq, 1);
                alts[0] = seq;
                node.alts = alts;
                return node;
            } else if (c == 'i') {
                return self.fail("(?i) is only allowed at the start of the pattern");
            } else {
                return self.fail("unknown group type");
            }
        } else {
            node.capture = @intCast(self.n_caps.*);
            self.n_caps.* += 1;
        }
        node.alts = try self.parseAlts();
        try self.expect(')', "')' to close group");
        return node;
    }

    const Escape = union(enum) { class: Class, literal: u32, boundary };

    /// Ports `parse_escape` (chat-tts-norm.cpp:350 @b1b770d90): after '%'
    /// inside or outside of a set.
    fn parseEscape(self: *Parser, in_set: bool) mod.exception.Error!Escape {
        if (self.eof()) return self.fail("dangling '%'");
        const e = self.s[self.p];
        self.p += 1;
        if (isClassLetter(e)) {
            return .{ .class = .{ .kind = @intCast(asciiLower(e)), .neg = e >= 'A' and e <= 'Z' } };
        }
        if (e == 'b' and !in_set) return .boundary;
        if (isAsciiAlnum(e)) {
            self.p -= 1;
            return self.fail(try std.fmt.allocPrint(self.a, "unknown escape '%{c}'", .{@as(u8, @intCast(e))}));
        }
        return .{ .literal = e };
    }

    /// Ports `parse_set` (chat-tts-norm.cpp:373 @b1b770d90); '[' already
    /// consumed.
    fn parseSet(self: *Parser) mod.exception.Error!PatternNode {
        var node: PatternNode = .{ .op = .set };
        if (self.peekIs('^')) {
            node.set.neg = true;
            self.p += 1;
        }
        var empty = true;
        while (true) {
            if (self.eof()) return self.fail("unterminated set");
            if (self.peek() == ']') {
                self.p += 1;
                break;
            }
            empty = false;
            var lo: u32 = 0;
            if (self.peek() == '%') {
                self.p += 1;
                switch (try self.parseEscape(true)) {
                    .class => |cls| {
                        try node.set.classes.append(self.a, cls);
                        continue;
                    },
                    .literal => |l| lo = l,
                    .boundary => unreachable,
                }
            } else {
                lo = self.s[self.p];
                self.p += 1;
            }
            var hi = lo;
            if (self.peekIs('-') and self.p + 1 < self.s.len and self.s[self.p + 1] != ']') {
                self.p += 1;
                if (self.peek() == '%') {
                    self.p += 1;
                    switch (try self.parseEscape(true)) {
                        .class => return self.fail("character class cannot be used in a range"),
                        .literal => |l| hi = l,
                        .boundary => unreachable,
                    }
                } else {
                    hi = self.s[self.p];
                    self.p += 1;
                }
                if (hi < lo) return self.fail("invalid range in set");
            }
            try node.set.ranges.append(self.a, .{ .lo = lo, .hi = hi });
        }
        if (empty) return self.fail("empty set");
        return node;
    }

    /// Ports `parse_atom` (chat-tts-norm.cpp:428 @b1b770d90).
    fn parseAtom(self: *Parser) mod.exception.Error!PatternNode {
        const c = self.s[self.p];
        self.p += 1;
        switch (c) {
            '(' => return self.parseGroup(),
            '[' => return self.parseSet(),
            '.' => return .{ .op = .any },
            '^' => return .{ .op = .start },
            '$' => return .{ .op = .end },
            '%' => {
                var node: PatternNode = .{};
                switch (try self.parseEscape(false)) {
                    .class => |cls| {
                        node.op = .set;
                        try node.set.classes.append(self.a, cls);
                    },
                    .boundary => node.op = .boundary,
                    .literal => |l| {
                        node.op = .literal;
                        node.cp = l;
                    },
                }
                return node;
            },
            '?', '*', '+', '{' => {
                self.p -= 1;
                return self.fail("nothing to repeat");
            },
            ']', '}' => {
                self.p -= 1;
                const ch: u8 = @intCast(c);
                return self.fail(try std.fmt.allocPrint(self.a, "unescaped '{c}', use '%{c}'", .{ ch, ch }));
            },
            else => return .{ .op = .literal, .cp = c },
        }
    }
};

// -----------------------------------------------------------------------------
// matcher

const Cap = struct { start: i64, end: i64 };

/// Ports `tts_norm_matcher` (chat-tts-norm.cpp:484 @b1b770d90): PEG-style,
/// alternatives are ordered (the first one that matches wins) and
/// quantifiers are greedy and possessive (they never give back what they
/// consumed), so there is no backtracking.
const Matcher = struct {
    a: std.mem.Allocator,
    t: []const u32,
    icase: bool,
    caps: []Cap,

    fn saveCaps(self: *Matcher) mod.exception.Error![]Cap {
        return self.a.dupe(Cap, self.caps);
    }

    fn restoreCaps(self: *Matcher, saved: []const Cap) void {
        @memcpy(self.caps, saved);
    }

    /// Ports `is_w` (chat-tts-norm.cpp:489 @b1b770d90).
    fn isW(self: *const Matcher, i: usize) bool {
        return i < self.t.len and classMatch('w', self.t[i]);
    }

    fn setTest(st: *const Set, x: u32) bool {
        for (st.ranges.items) |r| {
            if (x >= r.lo and x <= r.hi) return true;
        }
        for (st.classes.items) |k| {
            if (classMatch(k.kind, x) != k.neg) return true;
        }
        return false;
    }

    /// Ports `set_match` (chat-tts-norm.cpp:493 @b1b770d90).
    fn setMatch(self: *const Matcher, st: *const Set, c: u32) bool {
        var m = setTest(st, c);
        if (!m and self.icase) {
            const o = asciiSwapcase(c);
            if (o != c) m = setTest(st, o);
        }
        return m != st.neg;
    }

    /// Ports `match_alts` (chat-tts-norm.cpp:517 @b1b770d90).
    fn matchAlts(self: *Matcher, alts: []const Seq, i: usize, end: *usize) mod.exception.Error!bool {
        for (alts) |seq| {
            const saved = try self.saveCaps();
            if (try self.matchSeq(seq, i, end)) return true;
            self.restoreCaps(saved);
        }
        return false;
    }

    /// Ports `match_seq` (chat-tts-norm.cpp:528 @b1b770d90).
    fn matchSeq(self: *Matcher, seq: Seq, i: usize, end: *usize) mod.exception.Error!bool {
        var pos = i;
        for (seq) |*it| {
            var count: c_int = 0;
            while (count < it.max) {
                const is_group = it.atom.op == .group;
                var saved: []Cap = &.{};
                if (is_group) saved = try self.saveCaps();
                var e = pos;
                if (!try self.matchOnce(&it.atom, pos, &e)) {
                    if (is_group) self.restoreCaps(saved);
                    break;
                }
                if (e == pos) {
                    // an empty iteration would repeat forever, it satisfies any remaining minimum
                    count = @max(count + 1, it.min);
                    break;
                }
                pos = e;
                count += 1;
            }
            if (count < it.min) return false;
        }
        end.* = pos;
        return true;
    }

    /// Ports `match_once` (chat-tts-norm.cpp:561 @b1b770d90).
    fn matchOnce(self: *Matcher, node: *const PatternNode, i: usize, end: *usize) mod.exception.Error!bool {
        const n = self.t.len;
        switch (node.op) {
            .literal => {
                if (i < n and (self.t[i] == node.cp or (self.icase and asciiLower(self.t[i]) == asciiLower(node.cp)))) {
                    end.* = i + 1;
                    return true;
                }
                return false;
            },
            .set => {
                if (i < n and self.setMatch(&node.set, self.t[i])) {
                    end.* = i + 1;
                    return true;
                }
                return false;
            },
            .any => {
                if (i < n) {
                    end.* = i + 1;
                    return true;
                }
                return false;
            },
            .start => {
                end.* = i;
                return i == 0;
            },
            .end => {
                end.* = i;
                return i == n;
            },
            .boundary => {
                end.* = i;
                return (i > 0 and self.isW(i - 1)) != self.isW(i);
            },
            .group => {
                var e = i;
                if (!try self.matchAlts(node.alts, i, &e)) return false;
                if (node.capture >= 0) {
                    self.caps[@intCast(node.capture)] = .{ .start = @intCast(i), .end = @intCast(e) };
                }
                end.* = e;
                return true;
            },
            .lookahead => {
                // captures made inside a lookahead are discarded
                const saved = try self.saveCaps();
                var e = i;
                const ok = try self.matchAlts(node.alts, i, &e);
                self.restoreCaps(saved);
                end.* = i;
                return ok != node.neg;
            },
            .lookbehind => {
                var e: usize = 0;
                const ok = i > 0 and try self.matchOnce(&node.alts[0][0].atom, i - 1, &e) and e == i;
                end.* = i;
                return ok != node.neg;
            },
        }
    }
};

// -----------------------------------------------------------------------------
// common_tts_norm_pattern

/// Ports `common_tts_norm_pattern` (chat-tts-norm.h:24 @b1b770d90).
pub const Pattern = struct {
    root: PatternNode,
    n_caps: usize = 0,
    icase: bool = false,

    /// Ports `common_tts_norm_pattern`'s constructor (chat-tts-norm.cpp:626
    /// @b1b770d90).
    ///
    /// Return: the pattern, in `a`; `error.RuntimeError` on an invalid one.
    pub fn init(a: std.mem.Allocator, src: []const u8) mod.exception.Error!Pattern {
        if (src.len > max_pattern_len) {
            return mod.exception.throw(error.RuntimeError, "invalid TTS norm pattern: pattern is longer than {d} bytes", .{max_pattern_len});
        }
        const decoded = try decodeUtf8(a, src);
        var self: Pattern = .{ .root = .{ .op = .group } };
        var parser: Parser = .{ .a = a, .s = decoded.cpts, .n_caps = &self.n_caps };
        const icase_prefix = [_]u32{ '(', '?', 'i', ')' };
        if (parser.s.len >= 4 and std.mem.eql(u32, &icase_prefix, parser.s[0..4])) {
            self.icase = true;
            parser.p = 4;
        }
        self.root.alts = try parser.parseAlts();
        if (!parser.eof()) return parser.fail("unbalanced ')'");
        return self;
    }

    /// Ports `match_at` (chat-tts-norm.cpp:653 @b1b770d90): a match anchored
    /// at code point index `pos`.
    ///
    /// Parameters:
    /// - `end`: on success, the code point index after the match.
    /// - `groups`: on success, `[start, end)` of each capture, `-1, -1` for one
    ///   that did not participate; owned by `a`.
    pub fn matchAt(self: *const Pattern, a: std.mem.Allocator, cpts: []const u32, pos: usize, end: *usize, groups: *[]Cap) mod.exception.Error!bool {
        const caps = try a.alloc(Cap, self.n_caps);
        @memset(caps, .{ .start = -1, .end = -1 });
        var m: Matcher = .{ .a = a, .t = cpts, .icase = self.icase, .caps = caps };
        var e = pos;
        if (!try m.matchAlts(self.root.alts, pos, &e)) return false;
        end.* = e;
        groups.* = m.caps;
        return true;
    }
};

// -----------------------------------------------------------------------------
// scan / sub

/// Ports `common_tts_norm_segment` (chat-tts-norm.h:45 @b1b770d90): byte
/// offsets into the input text.
pub const Segment = struct {
    /// index of the matched rule, -1 for unmatched text
    rule: c_int = -1,
    start: usize = 0,
    end: usize = 0,
    /// byte offsets; null for groups that did not participate
    groups: []?[2]usize = &.{},
};

/// Ports `common_tts_norm_scan` (chat-tts-norm.cpp:672 @b1b770d90): the text
/// split into matched and unmatched segments. At each position the rules
/// are tried in order and the first non-empty match wins; otherwise one
/// code point is skipped.
pub fn scan(a: std.mem.Allocator, text: []const u8, rules: []const *const Pattern) mod.exception.Error![]Segment {
    const d = try decodeUtf8(a, text);
    var out: std.ArrayList(Segment) = .empty;
    const n = d.cpts.len;
    var i: usize = 0;
    var pending: usize = 0; // start of the current unmatched run
    var groups: []Cap = &.{};

    while (i < n) {
        var matched = false;
        for (rules, 0..) |rule, r| {
            var end = i;
            if (try rule.matchAt(a, d.cpts, i, &end, &groups) and end > i) {
                if (pending < i) try out.append(a, .{ .start = d.offs[pending], .end = d.offs[i] });
                var seg: Segment = .{ .rule = @intCast(r), .start = d.offs[i], .end = d.offs[end] };
                const gs = try a.alloc(?[2]usize, groups.len);
                for (groups, 0..) |g, k| {
                    gs[k] = if (g.start < 0) null else .{ d.offs[@intCast(g.start)], d.offs[@intCast(g.end)] };
                }
                seg.groups = gs;
                try out.append(a, seg);
                i = end;
                pending = i;
                matched = true;
                break;
            }
        }
        if (!matched) i += 1;
    }
    if (pending < n) try out.append(a, .{ .start = d.offs[pending], .end = d.offs[n] });
    return out.items;
}

/// Ports `tts_norm_repl_piece` (chat-tts-norm.cpp:719 @b1b770d90).
const ReplPiece = struct {
    literal: []const u8 = "",
    /// 0 = whole match, 1.. = groups, -1 a literal
    group: c_int = -1,
};

/// Ports `tts_norm_parse_repl` (chat-tts-norm.cpp:724 @b1b770d90).
fn parseRepl(a: std.mem.Allocator, repl: []const u8, pattern: *const Pattern) mod.exception.Error![]ReplPiece {
    var pieces: std.ArrayList(ReplPiece) = .empty;
    var lit: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < repl.len) : (i += 1) {
        if (repl[i] != '%') {
            try lit.append(a, repl[i]);
            continue;
        }
        if (i + 1 >= repl.len) {
            return mod.exception.throw(error.RuntimeError, "invalid TTS norm replacement: dangling '%'", .{});
        }
        i += 1;
        const c = repl[i];
        if (c == '%') {
            try lit.append(a, '%');
        } else if (c >= '0' and c <= '9') {
            const g: usize = c - '0';
            if (g > pattern.n_caps) {
                return mod.exception.throw(error.RuntimeError, "invalid TTS norm replacement: group %{c} does not exist", .{c});
            }
            if (lit.items.len != 0) {
                try pieces.append(a, .{ .literal = lit.items });
                lit = .empty;
            }
            try pieces.append(a, .{ .group = @intCast(g) });
        } else {
            return mod.exception.throw(error.RuntimeError, "invalid TTS norm replacement: unknown escape '%{c}'", .{c});
        }
    }
    if (lit.items.len != 0) try pieces.append(a, .{ .literal = lit.items });
    return pieces.items;
}

/// Ports `common_tts_norm_sub` (chat-tts-norm.cpp:759 @b1b770d90): every
/// non-overlapping match of `pattern` replaced by `repl`, whose references
/// are %0 (the whole match), %1..%9 (the groups) and %% (a percent).
pub fn sub(a: std.mem.Allocator, text: []const u8, pattern: *const Pattern, repl: []const u8) mod.exception.Error![]u8 {
    const pieces = try parseRepl(a, repl, pattern);
    var out: std.ArrayList(u8) = .empty;
    for (try scan(a, text, &.{pattern})) |seg| {
        if (seg.rule < 0) {
            try out.appendSlice(a, text[seg.start..seg.end]);
            continue;
        }
        for (pieces) |piece| {
            if (piece.group < 0) {
                try out.appendSlice(a, piece.literal);
            } else if (piece.group == 0) {
                try out.appendSlice(a, text[seg.start..seg.end]);
            } else if (seg.groups[@intCast(piece.group - 1)]) |g| {
                try out.appendSlice(a, text[g[0]..g[1]]);
            }
        }
    }
    return out.items;
}

// -----------------------------------------------------------------------------
// template builtins

const Rule = struct { name: []const u8, pattern: []const u8 };

/// Ports `tts_norm_rules_from_value` (chat-tts-norm.cpp:788 @b1b770d90): a
/// pattern string (rule name "match"), a dict {name: pattern}, or a list of
/// [name, pattern] pairs.
fn rulesFromValue(a: std.mem.Allocator, v: jinja.Value) mod.exception.Error![]Rule {
    var rules: std.ArrayList(Rule) = .empty;
    switch (v) {
        .string => |s| try rules.append(a, .{ .name = "match", .pattern = s }),
        .dict => |d| {
            var it = d.map.iterator();
            while (it.next()) |kv| {
                const pat = switch (kv.value_ptr.*) {
                    .string => |s| s,
                    else => return mod.exception.throw(error.RuntimeError, "tts_norm_scan: rule pattern must be a string", .{}),
                };
                try rules.append(a, .{ .name = kv.key_ptr.*, .pattern = pat });
            }
        },
        .list => |l| for (l.items.items) |item| {
            const pair = switch (item) {
                .list => |p| p.items.items,
                else => &.{},
            };
            if (pair.len != 2 or pair[0] != .string or pair[1] != .string) {
                return mod.exception.throw(error.RuntimeError, "tts_norm_scan: each rule must be a [name, pattern] pair of strings", .{});
            }
            try rules.append(a, .{ .name = pair[0].string, .pattern = pair[1].string });
        },
        else => return mod.exception.throw(error.RuntimeError, "tts_norm_scan: rules must be a string, a dict or a list of [name, pattern] pairs", .{}),
    }
    return rules.items;
}

/// `string(s, is_input)`: `s` marked input when `is_input`, else template text.
fn mkStr(a: std.mem.Allocator, s: []const u8, is_input: bool) jinja.value.CallError!jinja.Value {
    if (is_input) {
        if (jinja.provenance.active) |record| return .{ .string = try record.markInput(s) };
    }
    return .{ .string = try a.dupe(u8, s) };
}

/// `all_parts_are_input()` of a bound string.
fn allInput(s: []const u8) bool {
    const record = jinja.provenance.active orelse return false;
    return record.allInput(s);
}

/// A builtin's failure, as the engine reports a throw from a function.
fn callFailure(e: mod.exception.Error) jinja.value.CallError {
    return if (e == error.OutOfMemory) error.OutOfMemory else error.RuntimeError;
}

/// Ports `tts_norm_builtin_scan` (chat-tts-norm.cpp:813 @b1b770d90).
fn builtinScan(allocator: std.mem.Allocator, args: []jinja.Value, ctx: ?*anyopaque, env: ?*anyopaque) jinja.value.CallError!jinja.Value {
    _ = ctx;
    _ = env;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    if (args.len != 2) return error.RuntimeError;
    const text = switch (args[0]) {
        .string => |s| s,
        else => return error.RuntimeError,
    };
    const is_input = allInput(text);

    const rule_srcs = rulesFromValue(a, args[1]) catch |e| return callFailure(e);
    const patterns = try a.alloc(Pattern, rule_srcs.len);
    const ptrs = try a.alloc(*const Pattern, rule_srcs.len);
    for (rule_srcs, 0..) |r, i| patterns[i] = Pattern.init(a, r.pattern) catch |e| return callFailure(e);
    for (patterns, 0..) |*p, i| ptrs[i] = p;

    const segments = scan(a, text, ptrs) catch |e| return callFailure(e);
    const out = try allocator.create(jinja.value.List);
    out.* = jinja.value.List.init(allocator);
    for (segments) |seg| {
        const groups = try allocator.create(jinja.value.List);
        groups.* = jinja.value.List.init(allocator);
        for (seg.groups) |g| {
            // groups that did not participate are empty strings, so they render as nothing and are falsy
            try groups.append(if (g) |x| try mkStr(allocator, text[x[0]..x[1]], is_input) else try mkStr(allocator, text[0..0], is_input));
        }
        const obj = try allocator.create(jinja.value.Dict);
        obj.* = jinja.value.Dict.init(allocator);
        try obj.set("rule", if (seg.rule < 0) .null else .{ .string = try allocator.dupe(u8, rule_srcs[@intCast(seg.rule)].name) });
        try obj.set("text", try mkStr(allocator, text[seg.start..seg.end], is_input));
        try obj.set("groups", .{ .list = groups });
        try out.append(.{ .dict = obj });
    }
    return .{ .list = out };
}

/// Ports `tts_norm_builtin_sub` (chat-tts-norm.cpp:851 @b1b770d90).
fn builtinSub(allocator: std.mem.Allocator, args: []jinja.Value, ctx: ?*anyopaque, env: ?*anyopaque) jinja.value.CallError!jinja.Value {
    _ = ctx;
    _ = env;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    if (args.len != 3) return error.RuntimeError;
    for (args) |arg| if (arg != .string) return error.RuntimeError;
    const text = args[0].string;
    const pattern = Pattern.init(a, args[1].string) catch |e| return callFailure(e);
    const res = sub(a, text, &pattern, args[2].string) catch |e| return callFailure(e);
    return mkStr(allocator, res, allInput(text));
}

/// Ports `common_chat_tts_norm_add_builtins` (chat-tts-norm.cpp:860
/// @b1b770d90): `tts_norm_scan()` and `tts_norm_sub()` added to the context
/// of a task template. They are not global builtins on purpose: chat
/// templates are not affected.
///
/// Parameters:
/// - `a`: owns the two callables, for as long as the render runs.
/// - `set`: binds one variable, as `ctx.set_val` does.
pub fn addBuiltins(a: std.mem.Allocator, ctx: anytype, comptime set: fn (@TypeOf(ctx), []const u8, jinja.Value) mod.exception.Error!void) mod.exception.Error!void {
    const scan_fn = try a.create(jinja.value.Callable);
    scan_fn.* = jinja.value.Callable.initWithFunc(try a.dupe(u8, "tts_norm_scan"), builtinScan, false);
    try set(ctx, "tts_norm_scan", .{ .callable = scan_fn });
    const sub_fn = try a.create(jinja.value.Callable);
    sub_fn.* = jinja.value.Callable.initWithFunc(try a.dupe(u8, "tts_norm_sub"), builtinSub, false);
    try set(ctx, "tts_norm_sub", .{ .callable = sub_fn });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "sub follows upstream's test-jinja cases" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    // tests/test-jinja.cpp's test_tts_norm_patterns
    const cases = [_][4][]const u8{
        .{ "100 20", "%d+0", "X", "100 20" },
        .{ "100 20", "%d+", "X", "X X" },
        .{ "123-4 123-45 123-456789", "%d{3}-%d{2,}", "#", "123-4 # #" },
        .{ "GPT-5 is -5, (-3)", "(?<!%w)-(%d+)", "minus %1", "GPT-5 is minus 5, (minus 3)" },
        .{ "at 10:30 or 1:234", "(%d%d?):(%d%d)(?!%d)", "%1h%2", "at 10h30 or 1:234" },
        .{ "up 50%", "(%d+)%%", "%1 percent (%0) 100%%", "up 50 percent (50%) 100%" },
        .{ "Dr. Who and dr. No, not Mdr.", "(?i)%bdr%.", "doctor", "doctor Who and doctor No, not Mdr." },
        .{ "€ 5 & $6!", "[$€£]%s*(%d+)", "<%1>", "<5> & <6>!" },
        .{ "€ 5 & $6!", "[^%w%s]", "", " 5  6" },
        .{ "你 好 world 世界", "(%h) +(%h)", "%1%2", "你好 world 世界" },
    };
    for (cases) |c| {
        const p = try Pattern.init(a, c[1]);
        try std.testing.expectEqualStrings(c[3], try sub(a, c[0], &p, c[2]));
    }
}

test "scan splits text into matched and unmatched segments" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const num = try Pattern.init(a, "%d+");
    const word = try Pattern.init(a, "%a+");
    // "[word:abc][-: ][num:123][-:, ][word:x][num:9]"
    const segs = try scan(a, "abc 123, x9", &.{ &num, &word });
    const rules = [_]c_int{ 1, -1, 0, -1, 1, 0 };
    try std.testing.expectEqual(rules.len, segs.len);
    for (rules, segs) |r, seg| try std.testing.expectEqual(r, seg.rule);
    try std.testing.expectEqualStrings(", ", "abc 123, x9"[segs[3].start..segs[3].end]);
}

test "invalid patterns throw upstream's text" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try std.testing.expectError(error.RuntimeError, Pattern.init(a, "(a"));
    try std.testing.expectEqualStrings("invalid TTS norm pattern: expected ')' to close group at position 2", mod.exception.what(error.RuntimeError));
    try std.testing.expectError(error.RuntimeError, Pattern.init(a, "a]"));
    try std.testing.expectEqualStrings("invalid TTS norm pattern: unescaped ']', use '%]' at position 1", mod.exception.what(error.RuntimeError));
    try std.testing.expectError(error.RuntimeError, Pattern.init(a, "%q"));
    try std.testing.expectEqualStrings("invalid TTS norm pattern: unknown escape '%q' at position 1", mod.exception.what(error.RuntimeError));
}
