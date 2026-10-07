//! Compares `src/llama/regex.zig` with libc++'s `std::regex`.
//!
//! **Not a port.** The harness behind `scripts/regex-diff`: random
//! ECMAScript patterns, malformed ones included, run through both engines
//! in one process over random inputs -- `regex_search` and `regex_match`,
//! the two calls `llama-grammar.cpp` makes, and `regex_iterator` under
//! `optimize | nosubs` over a sub-range, the call `unicode.cpp` makes --
//! comparing whether the pattern compiled (and with which error code),
//! whether it matched, and every sub-match's position and length. Each
//! runs as `std::regex` and as `std::wregex`, the second over an alphabet
//! with the code points where a wide engine can differ.
//!
//! The alphabet is small on purpose: a handful of word, space, newline and
//! punctuation bytes makes repeats, ties between alternatives and empty
//! loop iterations common, which is where engines disagree. The real trigger
//! patterns upstream's chat code generates are run first, over inputs built
//! to trigger them.

const std = @import("std");
const regex = @import("regex");

extern fn ref_regex(pat: [*]const u8, plen: usize, in: [*]const u8, ilen: usize, mode: c_int, out: [*]c_long, max_groups: c_int) c_int;
extern fn ref_regex2(pat: *const anyopaque, plen: usize, in: *const anyopaque, ilen: usize, wide: c_int, nosubs: c_int, mode: c_int, begin: usize, end: usize, out: [*]c_long, max_out: c_int) c_int;
extern fn ref_wsweep(pat: [*]const u32, plen: usize, n: u32, out: [*]u8) c_int;
extern fn ref_wcompile_sweep(pre: [*]const u32, prelen: usize, post: [*]const u32, postlen: usize, n: u32, out: [*]u8) void;

const max_groups = 32;

var checks: usize = 0;
var failures: usize = 0;

/// The Zig engine's answer, in `ref_regex`'s encoding.
fn zigRegex(pat: []const u8, in: []const u8, mode: c_int, out: []c_long) c_int {
    var re = regex.Regex.compile(pat) catch |err| return -@as(c_int, regex.errorCode(err));
    defer re.deinit();
    var m = regex.Match.init();
    defer m.deinit();
    const ok = (if (mode == 0) re.trySearch(in, &m) else re.tryMatch(in, &m)) catch |err| {
        return -100 - @as(c_int, regex.errorCode(err));
    };
    if (!ok) return 0;
    const n = m.size();
    for (0..@min(n, max_groups)) |i| {
        out[2 * i] = @intCast(m.position(i));
        out[2 * i + 1] = if (m.matched(i)) @intCast(m.length(i)) else -1;
    }
    return @intCast(1 + n);
}

fn compare(pat: []const u8, in: []const u8) void {
    for ([_]c_int{ 0, 1 }) |mode| {
        var ro: [2 * max_groups]c_long = undefined;
        var zo: [2 * max_groups]c_long = undefined;
        const rr = ref_regex(pat.ptr, pat.len, in.ptr, in.len, mode, &ro, max_groups);
        const zr = zigRegex(pat, in, mode, &zo);
        checks += 1;
        var same = rr == zr;
        if (same and rr > 0) {
            const n: usize = @min(@as(usize, @intCast(rr - 1)), max_groups);
            same = std.mem.eql(c_long, ro[0 .. 2 * n], zo[0 .. 2 * n]);
        }
        if (!same) {
            failures += 1;
            if (failures <= 12) {
                std.debug.print("FAIL {s} pattern=\"{f}\" input=\"{f}\": ref {d} zig {d}", .{
                    if (mode == 0) "search" else "match", std.zig.fmtString(pat), std.zig.fmtString(in), rr, zr,
                });
                if (rr > 0 and zr > 0) {
                    const n: usize = @min(@as(usize, @intCast(rr - 1)), max_groups);
                    std.debug.print(" ref groups", .{});
                    for (0..n) |i| std.debug.print(" ({d},{d})", .{ ro[2 * i], ro[2 * i + 1] });
                    std.debug.print(" zig groups", .{});
                    for (0..n) |i| std.debug.print(" ({d},{d})", .{ zo[2 * i], zo[2 * i + 1] });
                }
                std.debug.print("\n", .{});
            }
        }
    }
}

/// The general comparison: `CharT` is `u8` for `std::regex` or `u32` for
/// `std::wregex`; mode 0 searches, 1 matches, 2 iterates over
/// `in[begin..end]`.
fn zigRun(comptime CharT: type, pat: []const CharT, in: []const CharT, nosubs: bool, mode: c_int, begin: usize, end: usize, out: []c_long) c_int {
    const Re = regex.BasicRegex(CharT);
    var re = Re.compileWith(pat, .{ .nosubs = nosubs }) catch |err| return -@as(c_int, regex.errorCode(err));
    defer re.deinit();
    if (mode == 2) {
        var it = Re.Iterator.init(&re, in, begin, end) catch |err| return -100 - @as(c_int, regex.errorCode(err));
        defer it.deinit();
        var n: usize = 0;
        while (it.valid() and n < max_groups) {
            out[2 * n] = @intCast(it.position());
            out[2 * n + 1] = @intCast(it.length());
            n += 1;
            it.next() catch |err| return -100 - @as(c_int, regex.errorCode(err));
        }
        return @intCast(1 + n);
    }
    var m = regex.Match.init();
    defer m.deinit();
    const ok = (if (mode == 0) re.trySearch(in, &m) else re.tryMatch(in, &m)) catch |err| {
        return -100 - @as(c_int, regex.errorCode(err));
    };
    if (!ok) return 0;
    const n = m.size();
    for (0..@min(n, max_groups)) |i| {
        out[2 * i] = @intCast(m.position(i));
        out[2 * i + 1] = if (m.matched(i)) @intCast(m.length(i)) else -1;
    }
    return @intCast(1 + n);
}

fn printUnits(comptime CharT: type, s: []const CharT) void {
    if (CharT == u8) {
        std.debug.print("\"{f}\"", .{std.zig.fmtString(s)});
    } else {
        std.debug.print("[", .{});
        for (s, 0..) |c, i| {
            if (i != 0) std.debug.print(" ", .{});
            if (c >= 0x20 and c < 0x7f) std.debug.print("'{c}'", .{@as(u8, @intCast(c))}) else std.debug.print("U+{X:0>4}", .{c});
        }
        std.debug.print("]", .{});
    }
}

fn compare2(comptime CharT: type, pat: []const CharT, in: []const CharT, nosubs: bool, mode: c_int, begin: usize, end: usize) void {
    var ro: [2 * max_groups]c_long = undefined;
    var zo: [2 * max_groups]c_long = undefined;
    const rr = ref_regex2(pat.ptr, pat.len, in.ptr, in.len, @intFromBool(CharT != u8), @intFromBool(nosubs), mode, begin, end, &ro, max_groups);
    const zr = zigRun(CharT, pat, in, nosubs, mode, begin, end, &zo);
    checks += 1;
    var same = rr == zr;
    if (same and rr > 0) {
        const n: usize = @min(@as(usize, @intCast(rr - 1)), max_groups);
        same = std.mem.eql(c_long, ro[0 .. 2 * n], zo[0 .. 2 * n]);
    }
    if (!same) {
        failures += 1;
        if (failures <= 12) {
            std.debug.print("FAIL {s}{s}{s} [{d},{d}) pattern=", .{
                if (CharT == u8) "" else "w",
                ([_][]const u8{ "search", "match", "iterate" })[@intCast(mode)],
                if (nosubs) " nosubs" else "",
                begin,
                end,
            });
            printUnits(CharT, pat);
            std.debug.print(" input=", .{});
            printUnits(CharT, in);
            std.debug.print(": ref {d} zig {d}", .{ rr, zr });
            if (rr > 0 and zr > 0) {
                std.debug.print(" ref", .{});
                for (0..@min(@as(usize, @intCast(rr - 1)), max_groups)) |i| std.debug.print(" ({d},{d})", .{ ro[2 * i], ro[2 * i + 1] });
                std.debug.print(" zig", .{});
                for (0..@min(@as(usize, @intCast(zr - 1)), max_groups)) |i| std.debug.print(" ({d},{d})", .{ zo[2 * i], zo[2 * i + 1] });
            }
            std.debug.print("\n", .{});
        }
    }
}

/// The alphabet a wide pattern and input draw from: ASCII word, space and
/// punctuation, plus the code points where `wchar_t` can go wrong -- the
/// two separators `.` rejects, U+0130 (low byte `'0'`), U+0161 (low byte
/// `'a'`), NEL, NBSP, Latin-1, CJK, a fullwidth digit and one past the BMP.
const wide_alphabet = [_]u32{ 'a', 'b', '_', ' ', '\n', '\r', '1', '\'', 0x0B, 0xE9, 0x85, 0xA0, 0x130, 0x161, 0x2028, 0x2029, 0x4E00, 0x9FA5, 0xFF11, 0x1F600 };

fn Gen(comptime CharT: type) type {
    return struct {
        const Self = @This();

        r: std.Random,
        buf: std.ArrayList(CharT) = .empty,
        groups: u32 = 0,

        fn lit(self: *Self) void {
            if (CharT != u8 and self.r.uintLessThan(u8, 3) == 0) {
                self.putc(@intCast(wide_alphabet[self.r.uintLessThan(usize, wide_alphabet.len)]));
                return;
            }
            const chars = "ab_ <>{}\n1:";
            const ch = chars[self.r.uintLessThan(usize, chars.len)];
            switch (ch) {
                '{', '}' => {
                    // Unescaped braces are mostly errors; escape them most of the time.
                    if (self.r.uintLessThan(u8, 4) != 0) self.put("\\");
                    self.putc(ch);
                },
                else => self.putc(ch),
            }
        }

        fn put(self: *Self, s: []const u8) void {
            for (s) |ch| self.putc(ch);
        }

        fn putc(self: *Self, ch: CharT) void {
            self.buf.append(std.heap.c_allocator, ch) catch unreachable;
        }

        fn class(self: *Self) void {
            self.put("[");
            if (self.r.boolean()) self.put("^");
            const n = self.r.uintLessThan(usize, 4) + 1;
            for (0..n) |_| {
                switch (self.r.uintLessThan(u8, if (CharT == u8) 9 else 11)) {
                    0 => self.put("a-c"),
                    1 => self.put("\\s"),
                    2 => self.put("\\w"),
                    3 => self.put("\\W"),
                    4 => self.put("[:alpha:]"),
                    5 => self.put("[:digit:]"),
                    6 => self.put("[.space.]"),
                    7 => self.put("\\]"),
                    9 => self.put("\\u4e00-\\u9fa5"),
                    10 => if (CharT != u8) {
                        self.putc(0xE0);
                        self.put("-");
                        self.putc(0x4E00);
                    },
                    else => self.lit(),
                }
            }
            if (self.r.uintLessThan(u8, 6) == 0) self.put("-");
            self.put("]");
        }

        fn atom(self: *Self, depth: u32) void {
            switch (self.r.uintLessThan(u8, if (depth > 2) 9 else 13)) {
                0, 1, 2 => self.lit(),
                3 => self.put("."),
                4 => self.class(),
                5 => self.put(([_][]const u8{ "\\d", "\\w", "\\s", "\\D", "\\W", "\\S", "\\.", "\\(", "\\n", "\\x41", "\\u0062" })[self.r.uintLessThan(usize, 11)]),
                6 => self.put(([_][]const u8{ "^", "$", "\\b", "\\B" })[self.r.uintLessThan(usize, 4)]),
                7 => if (self.groups > 0) {
                    self.put("\\");
                    self.putc('1' + @as(u8, @intCast(self.r.uintLessThan(u32, @min(self.groups, 9)))));
                } else self.lit(),
                8 => self.lit(),
                9, 10 => {
                    self.put("(");
                    self.groups += 1;
                    self.alt(depth + 1);
                    self.put(")");
                },
                11 => {
                    self.put("(?:");
                    self.alt(depth + 1);
                    self.put(")");
                },
                else => {
                    self.put(if (self.r.boolean()) "(?=" else "(?!");
                    self.alt(depth + 1);
                    self.put(")");
                },
            }
        }

        fn quant(self: *Self) void {
            switch (self.r.uintLessThan(u8, 14)) {
                0 => self.put("*"),
                1 => self.put("+"),
                2 => self.put("?"),
                3 => self.put("{2}"),
                4 => self.put("{1,}"),
                5 => self.put("{0,2}"),
                6 => self.put("*?"),
                7 => self.put("+?"),
                8 => self.put("??"),
                9 => self.put("{1,3}?"),
                else => {},
            }
        }

        fn seq(self: *Self, depth: u32) void {
            const n = self.r.uintLessThan(usize, 5);
            for (0..n) |_| {
                self.atom(depth);
                self.quant();
            }
        }

        fn alt(self: *Self, depth: u32) void {
            self.seq(depth);
            while (self.r.uintLessThan(u8, 4) == 0) {
                self.put("|");
                self.seq(depth);
            }
        }

        /// One random pattern; about one in eight is deliberately broken.
        fn pattern(self: *Self) []const CharT {
            self.buf.clearRetainingCapacity();
            self.groups = 0;
            self.alt(0);
            if (self.r.uintLessThan(u8, 8) == 0) {
                const breakers = [_][]const u8{ "(", ")", "[", "]", "}", "{", "*", "\\", "\\q", "[b-a]", "a{3,1}", "[[:nope:]]", "[[.ab.]]", "\\9", "(?" };
                const b = breakers[self.r.uintLessThan(usize, breakers.len)];
                const at = self.r.uintLessThan(usize, self.buf.items.len + 1);
                var tmp: [16]CharT = undefined;
                for (b, 0..) |ch, i| tmp[i] = ch;
                self.buf.insertSlice(std.heap.c_allocator, at, tmp[0..b.len]) catch unreachable;
            }
            return self.buf.items;
        }

        fn input(self: *Self, buf: []CharT) []const CharT {
            const chars = "ab_ <>{}\n1:\r\x00";
            const n = self.r.uintLessThan(usize, buf.len);
            for (buf[0..n]) |*ch| {
                if (CharT != u8 and self.r.uintLessThan(u8, 3) == 0) {
                    ch.* = @intCast(wide_alphabet[self.r.uintLessThan(usize, wide_alphabet.len)]);
                } else {
                    ch.* = chars[self.r.uintLessThan(usize, chars.len)];
                }
            }
            return buf[0..n];
        }
    };
}

/// Random patterns through search, match and -- with and without `nosubs`
/// -- a `regex_iterator` over a random sub-range.
fn randomRun(comptime CharT: type, seed: u64, n_patterns: usize) void {
    var prng = std.Random.DefaultPrng.init(seed);
    var gen: Gen(CharT) = .{ .r = prng.random() };
    defer gen.buf.deinit(std.heap.c_allocator);
    var inbuf: [40]CharT = undefined;
    for (0..n_patterns) |_| {
        const pat = gen.pattern();
        for (0..6) |_| {
            const in = gen.input(&inbuf);
            compare2(CharT, pat, in, false, 0, 0, in.len);
            compare2(CharT, pat, in, false, 1, 0, in.len);
            const b = gen.r.uintLessThan(usize, in.len + 1);
            const e = b + gen.r.uintLessThan(usize, in.len - b + 1);
            compare2(CharT, pat, in, gen.r.boolean(), 2, b, e);
        }
    }
}

fn toWide(s: []const u8, buf: []u32) []const u32 {
    for (s, 0..) |ch, i| buf[i] = ch;
    return buf[0..s.len];
}

/// Every code point, one at a time, through `regex_match` of each wide
/// pattern: classification, `.`, escapes and ranges as `wchar_t` sees them.
fn wideSweep() void {
    const n: u32 = 0x110000;
    const ref = std.heap.c_allocator.alloc(u8, n) catch unreachable;
    defer std.heap.c_allocator.free(ref);
    const pats = [_][]const u8{
        ".",           "\\s",         "\\S",               "\\w",         "\\W",           "\\d",          "\\D",
        "[\\s]",       "[^\\s]",      "[\\S]",             "[\\W]",       "[^\\W]",        "[\\D]",        "\\b.",
        "\\B.",        "(?=\\w).",    "[\\u4e00-\\u9fa5]", "[^a-z]",      "[\\x80-\\xff]", "[[=a=]]",      "[[.a.]-z]",
        "[[:alnum:]]", "[[:alpha:]]", "[[:blank:]]",       "[[:cntrl:]]", "[[:digit:]]",   "[[:graph:]]",  "[[:lower:]]",
        "[[:print:]]", "[[:punct:]]", "[[:space:]]",       "[[:upper:]]", "[[:w:]]",       "[[:xdigit:]]", "[^[:alpha:]]",
    };
    var wbuf: [64]u32 = undefined;
    var one: [1]u32 = undefined;
    for (pats) |p8| {
        const p = toWide(p8, &wbuf);
        const rc = ref_wsweep(p.ptr, p.len, n, ref.ptr);
        var re = regex.WRegex.compile(p) catch |err| {
            checks += 1;
            if (rc != -@as(c_int, regex.errorCode(err))) {
                failures += 1;
                std.debug.print("FAIL wsweep compile \"{s}\": ref {d} zig {d}\n", .{ p8, rc, -@as(c_int, regex.errorCode(err)) });
            }
            continue;
        };
        defer re.deinit();
        var m = regex.Match.init();
        defer m.deinit();
        var bad: usize = 0;
        var c: u32 = 0;
        while (c < n) : (c += 1) {
            one[0] = c;
            const z: u8 = @intFromBool(re.match(&one, &m));
            if (z != ref[c]) {
                bad += 1;
                if (bad <= 4) std.debug.print("FAIL wsweep \"{s}\" U+{X:0>4}: ref {d} zig {d}\n", .{ p8, c, ref[c], z });
            }
        }
        checks += n;
        failures += bad;
    }
}

/// Every code point in each position the parser reads through the traits,
/// compiled in both engines: the error codes must agree.
fn wideCompileSweep() void {
    const Case = struct { pre: []const u8, post: []const u8, n: u32 };
    const cases = [_]Case{
        .{ .pre = "a{", .post = "}", .n = 0x110000 },
        .{ .pre = "a{1,", .post = "}", .n = 0x3000 },
        .{ .pre = "\\x", .post = "0", .n = 0x110000 },
        .{ .pre = "\\u000", .post = "", .n = 0x3000 },
        .{ .pre = "[[.", .post = ".]]", .n = 0x110000 },
        .{ .pre = "[[=", .post = "=]]", .n = 0x3000 },
        .{ .pre = "[[:", .post = ":]]", .n = 0x3000 },
        .{ .pre = "[[:alph", .post = ":]]", .n = 0x3000 },
        .{ .pre = "\\c", .post = "", .n = 0x3000 },
        .{ .pre = "\\", .post = "", .n = 0x3000 },
        .{ .pre = "[\\", .post = "]", .n = 0x3000 },
        .{ .pre = "[b-", .post = "]", .n = 0x3000 },
        .{ .pre = "(?", .post = "a)", .n = 0x3000 },
    };
    const ref = std.heap.c_allocator.alloc(u8, 0x110000) catch unreachable;
    defer std.heap.c_allocator.free(ref);
    var pre_b: [16]u32 = undefined;
    var post_b: [16]u32 = undefined;
    var pat: [40]u32 = undefined;
    for (cases) |cs| {
        const pre = toWide(cs.pre, &pre_b);
        const post = toWide(cs.post, &post_b);
        ref_wcompile_sweep(pre.ptr, pre.len, post.ptr, post.len, cs.n, ref.ptr);
        var bad: usize = 0;
        var c: u32 = 0;
        while (c < cs.n) : (c += 1) {
            @memcpy(pat[0..pre.len], pre);
            pat[pre.len] = c;
            @memcpy(pat[pre.len + 1 ..][0..post.len], post);
            const p = pat[0 .. pre.len + 1 + post.len];
            const z: u8 = if (regex.WRegex.compile(p)) |re_c| blk: {
                var re = re_c;
                re.deinit();
                break :blk 0;
            } else |err| regex.errorCode(err);
            if (z != ref[c]) {
                bad += 1;
                if (bad <= 4) std.debug.print("FAIL wcompile \"{s}\" U+{X:0>4} \"{s}\": ref {d} zig {d}\n", .{ cs.pre, c, cs.post, ref[c], z });
            }
        }
        checks += cs.n;
        failures += bad;
    }
}

pub fn main() !void {
    // The trigger patterns llama.cpp's own chat code builds.
    const real = [_][]const u8{
        "[\\s\\S]*?(<tool_call>)[\\s\\S]*",
        "[\\s\\S]*?<tool_call>(\\{)[\\s\\S]*",
        "^[\\s\\S]*?<tool_call>(\\{)[\\s\\S]*$",
        "^\\s+to$",
        "^<\\|channel\\|>(?:commentary|analysis)\\s+to=functions$",
        "<\\|start\\|>assistant(\\s+to)",
        "<\\|start\\|>assistant(<\\|channel\\|>(?:commentary|analysis)\\s+to)",
        ">>>(?!all)",
        "[\\s\\S]*?(<tool_call>|a\\.b\\*\\(c\\))[\\s\\S]*",
        "(\\{\")",
    };
    const real_inputs = [_][]const u8{
        "",                                         "Let me check. <tool_call>{\"name\": \"x\"}",
        "<tool_call>",                              "  to",
        "<|channel|>commentary  to=functions",      "<|start|>assistant to",
        "<|start|>assistant<|channel|>analysis to", ">>>all",
        ">>>python",                                "a.b*(c){",
        "{\"k\": 1}",                               "no trigger here at all, just text\n\n",
    };
    for (real) |p| {
        for (real_inputs) |in| compare(p, in);
    }

    // The traits tables, swept: every name the measured tables hold, some
    // they do not, each as a collating symbol, an equivalence class and a
    // character class, against every single byte. Random patterns reach
    // almost none of these -- dropping `NUL` from the collating table passed
    // them all.
    {
        const names = [_][]const u8{
            "NUL",            "alert",             "backspace",            "tab",           "newline",
            "vertical-tab",   "form-feed",         "carriage-return",      "space",         "exclamation-mark",
            "quotation-mark", "number-sign",       "dollar-sign",          "percent-sign",  "ampersand",
            "apostrophe",     "left-parenthesis",  "right-parenthesis",    "asterisk",      "plus-sign",
            "comma",          "hyphen",            "hyphen-minus",         "period",        "full-stop",
            "slash",          "solidus",           "zero",                 "one",           "two",
            "three",          "four",              "five",                 "six",           "seven",
            "eight",          "nine",              "colon",                "semicolon",     "less-than-sign",
            "equals-sign",    "greater-than-sign", "question-mark",        "commercial-at", "left-square-bracket",
            "backslash",      "reverse-solidus",   "right-square-bracket", "circumflex",    "circumflex-accent",
            "underscore",     "low-line",          "grave-accent",         "left-brace",    "left-curly-bracket",
            "vertical-line",  "right-brace",       "right-curly-bracket",  "tilde",         "a",
            "Z",              "ab",                "SOH",                  "DEL",           "SPACE",
            "alnum",          "alpha",             "blank",                "cntrl",         "d",
            "digit",          "graph",             "lower",                "print",         "punct",
            "s",              "upper",             "w",                    "xdigit",        "ALPHA",
            "Digit",          "word",              "",
        };
        var pat: [64]u8 = undefined;
        for (names) |name| {
            for ([_][2][]const u8{ .{ "[[.", ".]]" }, .{ "[[=", "=]]" }, .{ "[[:", ":]]" }, .{ "[^[:", ":]]" } }) |wrap| {
                const p = std.fmt.bufPrint(&pat, "{s}{s}{s}", .{ wrap[0], name, wrap[1] }) catch unreachable;
                var b: usize = 0;
                while (b < 256) : (b += 1) {
                    const in = [_]u8{@intCast(b)};
                    compare(p, &in);
                }
            }
        }
        // Escapes and `.` over every byte.
        for ([_][]const u8{ ".", "\\d", "\\D", "\\s", "\\S", "\\w", "\\W", "[\\d]", "[\\D]", "[\\W]", "[^\\W]", "[\\x7f-\\xff]", "[]", "[^]", "\\0", "\\cJ" }) |p| {
            var b: usize = 0;
            while (b < 256) : (b += 1) {
                const in = [_]u8{@intCast(b)};
                compare(p, &in);
            }
        }
    }

    randomRun(u8, 0x7e6e, 30000);
    randomRun(u32, 0x3e6e, 20000);
    wideSweep();
    wideCompileSweep();

    if (failures == 0) {
        std.debug.print("PASS: {d} std::regex and std::wregex results identical, Zig vs libc++\n", .{checks});
        return;
    }
    std.debug.print("FAIL: {d} of {d} checks differ\n", .{ failures, checks });
    std.process.exit(1);
}
