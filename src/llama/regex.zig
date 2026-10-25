//! `std::regex` with ECMAScript grammar, as libc++ implements it.
//!
//! # Provenance
//!
//! Ported from Apple's libc++ 22.1.6 (`_LIBCPP_VERSION 220106`), the header
//! `usr/include/c++/v1/regex` in the macOS 27 SDK: the node types from
//! `__end_state` through `__lookahead`, `basic_regex`'s ECMAScript parser,
//! `__match_at_start_ecma`, `__search`, and `regex_search`/`regex_match`.
//!
//! Two pieces live in the libc++ dylib rather than the header, and were
//! **measured** rather than read: `__match_any_but_newline<char>::__exec`
//! (`.` rejects exactly `\n` and `\r`), and the two name tables behind
//! `regex_traits<char>::lookup_classname` and `lookup_collatename`. A probe
//! instantiated `[[:name:]]`, `[[.name.]]` and `[[=name=]]` for every
//! POSIX name and tested each against all 256 bytes; the tables below are
//! its output. `scripts/regex-diff` keeps them honest.
//!
//! **Not upstream llama.cpp code**, so citations name the libc++ template
//! rather than a file and line, as in `libcxx.zig`.
//!
//! # Why port a regex engine
//!
//! A lazy grammar's trigger patterns are `std::regex`es, compiled in
//! `llama_grammar_init_impl` (llama-grammar.cpp:1210 @c1d0e7a00), and *where* a match starts and
//! which capture group is first non-empty decide what text the grammar is
//! fed. ECMAScript fixes most of that, but libc++ is the reference, quirks
//! included: an unescaped `]` or `}` outside a class is a parse error, `.`
//! accepts NUL, `\b` never matches at the end of an empty remainder, loop
//! iterations reset their captures, and a match taking more than 4096 steps
//! per input byte throws `error_complexity`. A different ECMAScript engine
//! would be correct and still disagree.
//!
//! # Scope
//!
//! The configuration `llama-grammar.cpp` uses and no other: `char`, the
//! `"C"` locale, flags `ECMAScript` only -- no `icase`, `collate`,
//! `nosubs` or `multiline`. With the `"C"` locale `__might_have_digraph_` is
//! false, so a bracket's digraphs are parsed but never matched, as there.

const std = @import("std");
const impl = @import("impl.zig");

const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("regex", 0, "out of memory");
}

// -----------------------------------------------------------------------------
// regex_traits<char> in the "C" locale

const Mask = u32;
const m_space: Mask = 1 << 0;
const m_print: Mask = 1 << 1;
const m_cntrl: Mask = 1 << 2;
const m_upper: Mask = 1 << 3;
const m_lower: Mask = 1 << 4;
const m_alpha: Mask = 1 << 5;
const m_digit: Mask = 1 << 6;
const m_punct: Mask = 1 << 7;
const m_xdigit: Mask = 1 << 8;
const m_blank: Mask = 1 << 9;
const m_regex_word: Mask = 1 << 10;
const m_alnum: Mask = m_alpha | m_digit;
const m_graph: Mask = m_alnum | m_punct;

/// Whether `\t` is `blank` in the `"C"` locale's table. libc++'s classic
/// `ctype<char>` table is the C library's, and its classic
/// `ctype<wchar_t>::do_is` reads that same table for ASCII (line 694 of
/// `src/locale.cpp` in the libc++ the reference links -- libc++'s, not a
/// citation of upstream). On Windows the table is the UCRT's,
/// whose `_BLANK` marks the space alone -- measured by `scripts/regex-diff`
/// built for Windows, which answered `PLAN.md`'s open question: `[[:blank:]]`
/// misses `\t` there in `std::regex` and `std::wregex` alike, and no other
/// byte, code point or class differs.
const tab_is_blank = @import("builtin").os.tag != .windows;

/// The `"C"` locale's `ctype` classification of a character. Apple's
/// `ctype<char>::is` answers false for every byte above 0x7f, and
/// `ctype<wchar_t>::is` for every code point above it -- measured over all
/// 0x110000 by `scripts/regex-diff`, on Windows too.
fn ctypeOf(ch: u32) Mask {
    if (ch >= 0x80) return 0;
    var m: Mask = 0;
    if (ch == ' ' or (ch >= '\t' and ch <= '\r')) m |= m_space;
    if (ch == ' ' or (tab_is_blank and ch == '\t')) m |= m_blank;
    if (ch < 0x20 or ch == 0x7f) m |= m_cntrl;
    if (ch >= 0x20 and ch < 0x7f) m |= m_print;
    if (ch >= 'A' and ch <= 'Z') m |= m_upper | m_alpha;
    if (ch >= 'a' and ch <= 'z') m |= m_lower | m_alpha;
    if (ch >= '0' and ch <= '9') m |= m_digit;
    if ((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f') or (ch >= 'A' and ch <= 'F')) m |= m_xdigit;
    if (ch > 0x20 and ch < 0x7f and (m & m_alnum) == 0) m |= m_punct;
    return m;
}

/// Ports `regex_traits<CharT>::isctype`.
fn isctype(ch: u32, m: Mask) bool {
    if ((ctypeOf(ch) & m) != 0) return true;
    return ch == '_' and (m & m_regex_word) != 0;
}

/// Ports `regex_traits<char>::__regex_traits_value`: a digit's value in
/// `radix`, or -1.
fn traitsValue(ch: u8, radix: u8) i32 {
    if ((ch & 0xF8) == 0x30) return ch - '0';
    if (radix != 8) {
        if ((ch & 0xFE) == 0x38) return ch - '0';
        if (radix == 16) {
            const lc = ch | 0x20;
            if ('a' <= lc and lc <= 'f') return lc - ('a' - 10);
        }
    }
    return -1;
}

/// Ports both `__regex_traits_value` overloads. The `wchar_t` one narrows
/// first, and `ctype<wchar_t>::narrow` in the `"C"` locale never maps a
/// code point above 0x7f onto an ASCII digit or letter -- measured -- so
/// the answer there is always -1.
fn traitsValueOf(comptime CharT: type, ch: u32, radix: u8) i32 {
    if (CharT != u8 and ch >= 0x80) return -1;
    return traitsValue(@intCast(ch), radix);
}

/// Narrows a `wchar_t` name the way the wide `__lookup_classname` and
/// `__lookup_collatename` do: any character at or above 127 means no such
/// name. A `char` name passes through.
///
/// Return: the name as bytes in `buf`, or null.
fn narrowName(comptime CharT: type, name: []const CharT, buf: []u8) ?[]const u8 {
    if (CharT == u8) return name;
    if (name.len > buf.len) return null;
    for (name, 0..) |ch, i| {
        if (ch >= 127) return null;
        buf[i] = @intCast(ch);
    }
    return buf[0..name.len];
}

/// `regex_traits<CharT>::lookup_classname` with `icase` false, measured:
/// the name is matched case-insensitively against this table.
fn lookupClassname(comptime CharT: type, name_in: []const CharT) Mask {
    const table = [_]struct { []const u8, Mask }{
        .{ "alnum", m_alnum }, .{ "alpha", m_alpha },            .{ "blank", m_blank },
        .{ "cntrl", m_cntrl }, .{ "d", m_digit },                .{ "digit", m_digit },
        .{ "graph", m_graph }, .{ "lower", m_lower },            .{ "print", m_print },
        .{ "punct", m_punct }, .{ "s", m_space },                .{ "space", m_space },
        .{ "upper", m_upper }, .{ "w", m_alnum | m_regex_word }, .{ "xdigit", m_xdigit },
    };
    var nbuf: [16]u8 = undefined;
    const name = narrowName(CharT, name_in, &nbuf) orelse return 0;
    var buf: [16]u8 = undefined;
    if (name.len > buf.len) return 0;
    const lower = std.ascii.lowerString(&buf, name);
    for (table) |e| {
        if (std.mem.eql(u8, e[0], lower)) return e[1];
    }
    return 0;
}

/// `regex_traits<CharT>::lookup_collatename`, measured: a case-sensitive
/// table of POSIX names, else a one-character name is itself; anything
/// else is not a collating element. A wide name with a character at or
/// above 127 is never one.
///
/// Return: the element, or null.
fn lookupCollatename(comptime CharT: type, name_in: []const CharT) ?u32 {
    const table = [_]struct { []const u8, u8 }{
        .{ "NUL", 0x00 },                 .{ "alert", 0x07 },             .{ "backspace", 0x08 },
        .{ "tab", 0x09 },                 .{ "newline", 0x0a },           .{ "vertical-tab", 0x0b },
        .{ "form-feed", 0x0c },           .{ "carriage-return", 0x0d },   .{ "space", 0x20 },
        .{ "exclamation-mark", 0x21 },    .{ "quotation-mark", 0x22 },    .{ "number-sign", 0x23 },
        .{ "dollar-sign", 0x24 },         .{ "percent-sign", 0x25 },      .{ "ampersand", 0x26 },
        .{ "apostrophe", 0x27 },          .{ "left-parenthesis", 0x28 },  .{ "right-parenthesis", 0x29 },
        .{ "asterisk", 0x2a },            .{ "plus-sign", 0x2b },         .{ "comma", 0x2c },
        .{ "hyphen", 0x2d },              .{ "hyphen-minus", 0x2d },      .{ "period", 0x2e },
        .{ "full-stop", 0x2e },           .{ "slash", 0x2f },             .{ "solidus", 0x2f },
        .{ "zero", 0x30 },                .{ "one", 0x31 },               .{ "two", 0x32 },
        .{ "three", 0x33 },               .{ "four", 0x34 },              .{ "five", 0x35 },
        .{ "six", 0x36 },                 .{ "seven", 0x37 },             .{ "eight", 0x38 },
        .{ "nine", 0x39 },                .{ "colon", 0x3a },             .{ "semicolon", 0x3b },
        .{ "less-than-sign", 0x3c },      .{ "equals-sign", 0x3d },       .{ "greater-than-sign", 0x3e },
        .{ "question-mark", 0x3f },       .{ "commercial-at", 0x40 },     .{ "left-square-bracket", 0x5b },
        .{ "backslash", 0x5c },           .{ "reverse-solidus", 0x5c },   .{ "right-square-bracket", 0x5d },
        .{ "circumflex", 0x5e },          .{ "circumflex-accent", 0x5e }, .{ "underscore", 0x5f },
        .{ "low-line", 0x5f },            .{ "grave-accent", 0x60 },      .{ "left-brace", 0x7b },
        .{ "left-curly-bracket", 0x7b },  .{ "vertical-line", 0x7c },     .{ "right-brace", 0x7d },
        .{ "right-curly-bracket", 0x7d }, .{ "tilde", 0x7e },
    };
    var nbuf: [32]u8 = undefined;
    const name = narrowName(CharT, name_in, &nbuf) orelse return null;
    if (name.len == 0) return null;
    for (table) |e| {
        if (std.mem.eql(u8, e[0], name)) return e[1];
    }
    if (name.len == 1) return name[0];
    return null;
}

// -----------------------------------------------------------------------------
// Errors

/// `std::regex_constants::error_type`, plus `__re_err_parse`.
pub const Error = error{
    Collate,
    Ctype,
    Escape,
    Backref,
    Brack,
    Paren,
    Brace,
    BadBrace,
    Range,
    BadRepeat,
    Complexity,
    Parse,
};

/// The `std::regex_constants::error_type` value libc++ gives each error,
/// for `scripts/regex-diff` to compare against `regex_error::code()`.
///
/// Parameters:
/// - `err`: the error.
///
/// Return: the code, 1 for `error_collate` through 17 for `__re_err_parse`.
pub fn errorCode(err: Error) u8 {
    return switch (err) {
        error.Collate => 1,
        error.Ctype => 2,
        error.Escape => 3,
        error.Backref => 4,
        error.Brack => 5,
        error.Paren => 6,
        error.Brace => 7,
        error.BadBrace => 8,
        error.Range => 9,
        error.BadRepeat => 11,
        error.Complexity => 12,
        error.Parse => 17,
    };
}

/// The `what()` libc++ gives each error.
///
/// Parameters:
/// - `err`: the error.
///
/// Return: the message, static.
pub fn errorText(err: Error) [:0]const u8 {
    return switch (err) {
        error.Collate => "The expression contained an invalid collating element name.",
        error.Ctype => "The expression contained an invalid character class name.",
        error.Escape => "The expression contained an invalid escaped character, or a trailing escape.",
        error.Backref => "The expression contained an invalid back reference.",
        error.Brack => "The expression contained mismatched [ and ].",
        error.Paren => "The expression contained mismatched ( and ).",
        error.Brace => "The expression contained mismatched { and }.",
        error.BadBrace => "The expression contained an invalid range in a {} expression.",
        error.Range => "The expression contained an invalid character range, such as [b-a] in most encodings.",
        error.BadRepeat => "One of *?+{ was not preceded by a valid regular expression.",
        error.Complexity => "The complexity of an attempted match against a regular expression exceeded a pre-set level.",
        error.Parse => "The parser did not consume the entire regular expression.",
    };
}

/// Stands in for a `std::regex_error` escaping through the C ABI: the C++
/// terminates, and so does this.
///
/// Parameters:
/// - `err`: the error.
///
/// Return: never.
pub fn throwRegexError(err: Error) noreturn {
    impl.c.ggml_abort("regex", 0, "terminating due to uncaught exception of type std::__1::regex_error: %s", errorText(err).ptr);
}

// -----------------------------------------------------------------------------
// Nodes

const none = std.math.maxInt(u32);

/// The node types libc++ builds for an ECMAScript `char` regex, one tag
/// each in place of a virtual `__exec`.
const Kind = enum {
    end_state,
    empty_state,
    empty_non_own,
    repeat_one_loop,
    loop,
    alternate,
    begin_marked,
    end_marked,
    back_ref,
    word_boundary,
    l_anchor,
    r_anchor,
    match_any_but_newline,
    match_char,
    bracket,
    lookahead,
};

const Node = struct {
    kind: Kind,
    /// `__has_one_state::__first_`.
    first: u32 = none,
    /// `__owns_two_states::__second_`.
    second: u32 = none,

    ch: u32 = 0,
    mexp: u32 = 0,
    invert: bool = false,

    loop_id: u32 = 0,
    min: usize = 0,
    max: usize = 0,
    mexp_begin: u32 = 0,
    mexp_end: u32 = 0,
    greedy: bool = true,

    /// The bracket's or the lookahead's index.
    aux: u32 = 0,
};

/// Ports `__bracket_expression` (non-`icase`, non-`collate`).
const Bracket = struct {
    chars: std.ArrayList(u32) = .empty,
    neg_chars: std.ArrayList(u32) = .empty,
    ranges: std.ArrayList([2]u32) = .empty,
    digraphs: std.ArrayList([2]u32) = .empty,
    equivalences: std.ArrayList(u32) = .empty,
    mask: Mask = 0,
    neg_mask: Mask = 0,
    negate: bool,

    fn deinit(self: *Bracket) void {
        self.chars.deinit(allocator);
        self.neg_chars.deinit(allocator);
        self.ranges.deinit(allocator);
        self.digraphs.deinit(allocator);
        self.equivalences.deinit(allocator);
    }

    fn clone(self: *const Bracket) Bracket {
        var b: Bracket = .{ .mask = self.mask, .neg_mask = self.neg_mask, .negate = self.negate };
        b.chars.appendSlice(allocator, self.chars.items) catch oom();
        b.neg_chars.appendSlice(allocator, self.neg_chars.items) catch oom();
        b.ranges.appendSlice(allocator, self.ranges.items) catch oom();
        b.digraphs.appendSlice(allocator, self.digraphs.items) catch oom();
        b.equivalences.appendSlice(allocator, self.equivalences.items) catch oom();
        return b;
    }

    /// Ports `__bracket_expression::__add_range`: both ends one character,
    /// in unsigned order.
    fn addRange(self: *Bracket, b: []const u32, e: []const u32) Error!void {
        if (b.len != 1 or e.len != 1 or e[0] < b[0]) return error.Range;
        self.ranges.append(allocator, .{ b[0], e[0] }) catch oom();
    }

    fn addStart(self: *Bracket, r: []const u32) void {
        if (r.len == 1) {
            self.chars.append(allocator, r[0]) catch oom();
        } else {
            self.digraphs.append(allocator, .{ r[0], r[1] }) catch oom();
        }
    }

    /// Ports `__bracket_expression::__exec`'s test, with no digraphs.
    fn matches(self: *const Bracket, ch: u32) bool {
        for (self.chars.items) |cc| {
            if (ch == cc) return true;
        }
        if (!(self.neg_mask == 0 and self.neg_chars.items.len == 0)) {
            const in_neg_mask = isctype(ch, self.neg_mask);
            const in_neg_chars = std.mem.findScalar(u32, self.neg_chars.items, ch) != null;
            if (!(in_neg_mask or in_neg_chars)) return true;
        }
        for (self.ranges.items) |r| {
            if (r[0] <= ch and ch <= r[1]) return true;
        }
        for (self.equivalences.items) |e| {
            if (ch == e) return true;
        }
        return isctype(ch, self.mask);
    }
};

// -----------------------------------------------------------------------------
// basic_regex

/// A compiled `std::regex`.
pub fn BasicRegex(comptime CharT: type) type {
    return struct {
        const Self = @This();

        nodes: std.ArrayList(Node) = .empty,
        brackets: std.ArrayList(Bracket) = .empty,
        lookaheads: std.ArrayList(*Self) = .empty,

        /// `__start_`, an `__empty_state`.
        start: u32 = none,
        /// `__end_`, the node new ones are spliced after while parsing.
        end: u32 = none,
        marked_count: u32 = 0,
        loop_count: u32 = 0,
        open_count: i32 = 0,
        /// `regex_constants::nosubs`: groups are parsed but record nothing.
        nosubs: bool = false,

        /// Ports `basic_regex(const std::string &)` and `__init`: the pattern
        /// must parse completely.
        ///
        /// Parameters:
        /// - `pattern`: the ECMAScript pattern.
        ///
        /// Return: the compiled regex, or the `regex_error` the constructor
        /// would throw.
        pub fn compile(pattern: []const CharT) Error!Self {
            return compileWith(pattern, .{});
        }

        /// `compile` with syntax flags, as `basic_regex(s, flags)`.
        ///
        /// Parameters:
        /// - `pattern`: the ECMAScript pattern.
        /// - `syntax`: the flags beyond `ECMAScript`.
        ///
        /// Return: the compiled regex, or the `regex_error`.
        pub fn compileWith(pattern: []const CharT, syntax: Syntax) Error!Self {
            var re: Self = .{ .nosubs = syntax.nosubs };
            errdefer re.deinit();
            var p: Parser(CharT) = .{ .re = &re, .pat = pattern };
            const temp = try p.parse(0);
            if (temp != pattern.len) return error.Parse;
            return re;
        }

        /// Frees the nodes, brackets and nested lookahead regexes.
        ///
        /// Return: nothing.
        pub fn deinit(self: *Self) void {
            for (self.brackets.items) |*b| b.deinit();
            self.brackets.deinit(allocator);
            for (self.lookaheads.items) |la| {
                la.deinit();
                allocator.destroy(la);
            }
            self.lookaheads.deinit(allocator);
            self.nodes.deinit(allocator);
            self.* = .{};
        }

        /// A deep copy. libc++ shares the node graph between copies through a
        /// `shared_ptr`; it is immutable after construction, so a copy is
        /// indistinguishable.
        ///
        /// Return: the copy, freed with `deinit`.
        pub fn clone(self: *const Self) Self {
            var out: Self = .{
                .nosubs = self.nosubs,
                .start = self.start,
                .end = self.end,
                .marked_count = self.marked_count,
                .loop_count = self.loop_count,
                .open_count = self.open_count,
            };
            out.nodes.appendSlice(allocator, self.nodes.items) catch oom();
            for (self.brackets.items) |*b| out.brackets.append(allocator, b.clone()) catch oom();
            for (self.lookaheads.items) |la| {
                const p = allocator.create(Self) catch oom();
                p.* = la.clone();
                out.lookaheads.append(allocator, p) catch oom();
            }
            return out;
        }

        /// `mark_count()`.
        pub fn markCount(self: *const Self) u32 {
            return self.marked_count;
        }

        fn addNode(self: *Self, n: Node) u32 {
            self.nodes.append(allocator, n) catch oom();
            return @intCast(self.nodes.items.len - 1);
        }

        /// The shared shape of every `__push_*`: the new node takes over
        /// `__end_`'s successor and becomes `__end_`.
        fn pushNode(self: *Self, n_in: Node) u32 {
            var n = n_in;
            n.first = self.nodes.items[self.end].first;
            const id = self.addNode(n);
            self.nodes.items[self.end].first = id;
            self.end = id;
            return id;
        }

        /// Ports `regex_search(const std::string &, std::smatch &, const regex &)`.
        ///
        /// Parameters:
        /// - `input`: the text.
        /// - `m`: receives the match; cleared when there is none.
        ///
        /// Return: whether a match was found anywhere.
        pub fn search(self: *const Self, input: []const CharT, m: *Match) bool {
            return self.searchImpl(input, 0, input.len, m, 0) catch |err| throwRegexError(err);
        }

        /// Ports `regex_match(const std::string &, std::smatch &, const regex &)`.
        ///
        /// Parameters:
        /// - `input`: the text.
        /// - `m`: receives the match; cleared when there is none.
        ///
        /// Return: whether the whole input matches.
        pub fn match(self: *const Self, input: []const CharT, m: *Match) bool {
            const r = self.searchImpl(input, 0, input.len, m, flag_match_continuous | flag_full_match) catch |err| throwRegexError(err);
            if (r) {
                // `!__m.suffix().matched`; `__full_match` already guarantees it.
                if (m.subs.items[0].second != input.len) {
                    m.subs.clearRetainingCapacity();
                    return false;
                }
            }
            return r;
        }

        /// `search`, returning the `regex_error` a match can throw instead of
        /// terminating. For `scripts/regex-diff`.
        ///
        /// Return: whether a match was found, or `error.Complexity`.
        pub fn trySearch(self: *const Self, input: []const CharT, m: *Match) Error!bool {
            return self.searchImpl(input, 0, input.len, m, 0);
        }

        /// `match`, returning the `regex_error` instead of terminating.
        ///
        /// Return: whether the whole input matches, or `error.Complexity`.
        pub fn tryMatch(self: *const Self, input: []const CharT, m: *Match) Error!bool {
            const r = try self.searchImpl(input, 0, input.len, m, flag_match_continuous | flag_full_match);
            if (r and m.subs.items[0].second != input.len) {
                m.subs.clearRetainingCapacity();
                return false;
            }
            return r;
        }

        /// Ports `regex_iterator` over `input[begin..end]`: the successive
        /// non-overlapping matches, with libc++'s rule for stepping past an
        /// empty one.
        pub const Iterator = struct {
            re: *const Self,
            input: []const CharT,
            end: usize,
            /// `match_results::__position_start_`: the range's start, fixed
            /// by the first search because every later one passes
            /// `__no_update_pos`.
            position_start: usize,
            /// `__flags_`, which accumulates `__no_update_pos` and
            /// `match_prev_avail` as the iterator advances.
            flags: u32 = 0,
            m: Match = .{},

            /// Ports the constructor: the first `regex_search`.
            ///
            /// Parameters:
            /// - `re`: the regex; borrowed for the iterator's lifetime.
            /// - `input`: the whole text; borrowed likewise.
            /// - `begin`, `end`: the range to search.
            ///
            /// Return: the iterator, at the end if nothing matched, or the
            /// `regex_error` a search can throw.
            pub fn init(re: *const Self, input: []const CharT, begin: usize, end: usize) Error!Iterator {
                var it: Iterator = .{ .re = re, .input = input, .end = end, .position_start = begin };
                errdefer it.deinit();
                _ = try re.searchImpl(input, begin, end, &it.m, it.flags);
                return it;
            }

            /// Return: nothing.
            pub fn deinit(self: *Iterator) void {
                self.m.deinit();
            }

            /// Whether the iterator is not yet `regex_iterator()`.
            pub fn valid(self: *const Iterator) bool {
                return self.m.subs.items.len != 0;
            }

            /// `match.position()`: from the range's start.
            pub fn position(self: *const Iterator) usize {
                return self.m.subs.items[0].first - self.position_start;
            }

            /// `match.length()`.
            pub fn length(self: *const Iterator) usize {
                return self.m.length(0);
            }

            /// Ports `operator++`.
            ///
            /// Return: nothing, or the `regex_error` a search can throw.
            pub fn next(self: *Iterator) Error!void {
                self.flags |= flag_no_update_pos;
                var start = self.m.subs.items[0].second;
                if (self.m.subs.items[0].first == self.m.subs.items[0].second) {
                    if (start == self.end) {
                        self.m.subs.clearRetainingCapacity();
                        return;
                    } else if (try self.re.searchImpl(self.input, start, self.end, &self.m, self.flags | flag_match_not_null | flag_match_continuous)) {
                        return;
                    } else {
                        start += 1;
                    }
                }
                self.flags |= flag_match_prev_avail;
                // On success libc++ moves the prefix's start back to the
                // previous match's end, which only `prefix()` can see.
                _ = try self.re.searchImpl(self.input, start, self.end, &self.m, self.flags);
            }
        };

        /// Ports `basic_regex::__search` over `input[first_in..last]`.
        /// Sub-match offsets are absolute indices into `input`, so with
        /// `match_prev_avail` the character before `first_in` is visible to
        /// `\b`, as through a `string::const_iterator`.
        fn searchImpl(self: *const Self, input: []const CharT, first_in: usize, last: usize, m: *Match, flags_in: u32) Error!bool {
            var flags = flags_in;
            if ((flags & flag_match_prev_avail) != 0) flags &= ~(flag_match_not_bol | flag_match_not_bow);
            m.reset(1 + self.marked_count, last);
            if (try self.matchAtStartEcma(input, first_in, last, m, flags, (flags & flag_no_update_pos) == 0)) return true;
            if (first_in != last and (flags & flag_match_continuous) == 0) {
                flags |= flag_match_prev_avail;
                var first = first_in + 1;
                while (first != last) : (first += 1) {
                    m.reset(1 + self.marked_count, last);
                    if (try self.matchAtStartEcma(input, first, last, m, flags, false)) return true;
                }
                m.reset(1 + self.marked_count, last);
                if (try self.matchAtStartEcma(input, last, last, m, flags, false)) return true;
            }
            m.subs.clearRetainingCapacity();
            return false;
        }

        /// Ports `basic_regex::__match_at_start_ecma`.
        ///
        /// Depth-first over a stack of states: a split pushes the preferred
        /// branch on top, a reject pops, and the first state to reach the end
        /// node is the match.
        fn matchAtStartEcma(self: *const Self, input: []const CharT, first: usize, last: usize, m: *Match, flags: u32, at_first: bool) Error!bool {
            if (self.start == none) return false;

            var states: std.ArrayList(State) = .empty;
            defer {
                for (states.items) |*s| s.deinit();
                states.deinit(allocator);
            }

            const s0: State = .{
                .do_ = .none,
                .first = first,
                .current = first,
                .last = last,
                .sub_matches = allocator.alloc(SubMatch, self.marked_count) catch oom(),
                .loop_data = allocator.alloc(LoopData, self.loop_count) catch oom(),
                .node = self.start,
                .flags = flags,
                .at_first = at_first,
            };
            @memset(s0.sub_matches, .{ .first = last, .second = last, .matched = false });
            @memset(s0.loop_data, .{ .count = 0, .pos = 0 });
            states.append(allocator, s0) catch oom();

            var counter: i32 = 0;
            const length: i32 = @intCast(last - first);
            while (true) {
                counter += 1;
                if (@rem(counter, complexity_factor) == 0 and @divTrunc(counter, complexity_factor) >= length) {
                    return error.Complexity;
                }
                const s = &states.items[states.items.len - 1];
                if (s.node != none) try self.exec(input, s);
                switch (s.do_) {
                    .end_state => {
                        if ((flags & flag_match_not_null) != 0 and s.current == first) {
                            var popped = states.pop().?;
                            popped.deinit();
                        } else if ((flags & flag_full_match) != 0 and s.current != last) {
                            var popped = states.pop().?;
                            popped.deinit();
                        } else {
                            m.subs.items[0] = .{ .first = first, .second = s.current, .matched = true };
                            for (s.sub_matches, 0..) |sm, i| m.subs.items[i + 1] = sm;
                            return true;
                        }
                    },
                    .accept_and_consume, .repeat, .accept_but_not_consume => {},
                    .split => {
                        var snext = s.clone();
                        self.execSplit(true, s);
                        self.execSplit(false, &snext);
                        states.append(allocator, snext) catch oom();
                    },
                    .reject => {
                        var popped = states.pop().?;
                        popped.deinit();
                    },
                    .none => impl.abort("regex", 0, "unknown regex state"),
                }
                if (states.items.len == 0) break;
            }
            return false;
        }

        /// The `__exec` of each node kind.
        fn exec(self: *const Self, input: []const CharT, s: *State) Error!void {
            const n = &self.nodes.items[s.node];
            switch (n.kind) {
                .end_state => s.do_ = .end_state,
                .empty_state, .empty_non_own => {
                    s.do_ = .accept_but_not_consume;
                    s.node = n.first;
                },
                .repeat_one_loop => {
                    s.do_ = .repeat;
                    s.node = n.first;
                },
                .loop => {
                    const ld = &s.loop_data[n.loop_id];
                    if (s.do_ == .repeat) {
                        ld.count += 1;
                        var do_repeat = ld.count < n.max;
                        const do_alt = ld.count >= n.min;
                        if (do_repeat and do_alt and ld.pos == s.current) do_repeat = false;
                        if (do_repeat and do_alt) {
                            s.do_ = .split;
                        } else if (do_repeat) {
                            s.do_ = .accept_but_not_consume;
                            s.node = n.first;
                            initRepeat(n, s);
                        } else {
                            s.do_ = .accept_but_not_consume;
                            s.node = n.second;
                        }
                    } else {
                        ld.count = 0;
                        const do_repeat = 0 < n.max;
                        const do_alt = 0 >= n.min;
                        if (do_repeat and do_alt) {
                            s.do_ = .split;
                        } else if (do_repeat) {
                            s.do_ = .accept_but_not_consume;
                            s.node = n.first;
                            initRepeat(n, s);
                        } else {
                            s.do_ = .accept_but_not_consume;
                            s.node = n.second;
                        }
                    }
                },
                .alternate => s.do_ = .split,
                .begin_marked => {
                    s.do_ = .accept_but_not_consume;
                    s.sub_matches[n.mexp - 1].first = s.current;
                    s.node = n.first;
                },
                .end_marked => {
                    s.do_ = .accept_but_not_consume;
                    s.sub_matches[n.mexp - 1].second = s.current;
                    s.sub_matches[n.mexp - 1].matched = true;
                    s.node = n.first;
                },
                .back_ref => {
                    if (n.mexp > s.sub_matches.len) return error.Backref;
                    const sm = s.sub_matches[n.mexp - 1];
                    if (sm.matched) {
                        const len = sm.second - sm.first;
                        if (s.last - s.current >= len and std.mem.eql(CharT, input[sm.first..sm.second], input[s.current .. s.current + len])) {
                            s.do_ = .accept_but_not_consume;
                            s.current += len;
                            s.node = n.first;
                        } else {
                            reject(s);
                        }
                    } else {
                        reject(s);
                    }
                },
                .word_boundary => {
                    var is_word_b = false;
                    if (s.first != s.last) {
                        if (s.current == s.last) {
                            if ((s.flags & flag_match_not_eow) == 0) {
                                const ch = input[s.current - 1];
                                is_word_b = ch == '_' or isctype(ch, m_alnum);
                            }
                        } else if (s.current == s.first and (s.flags & flag_match_prev_avail) == 0) {
                            if ((s.flags & flag_match_not_bow) == 0) {
                                const ch = input[s.current];
                                is_word_b = ch == '_' or isctype(ch, m_alnum);
                            }
                        } else {
                            const c1 = input[s.current - 1];
                            const c2 = input[s.current];
                            const is_c1_b = c1 == '_' or isctype(c1, m_alnum);
                            const is_c2_b = c2 == '_' or isctype(c2, m_alnum);
                            is_word_b = is_c1_b != is_c2_b;
                        }
                    }
                    if (is_word_b != n.invert) {
                        s.do_ = .accept_but_not_consume;
                        s.node = n.first;
                    } else {
                        reject(s);
                    }
                },
                .l_anchor => {
                    if (s.at_first and s.current == s.first and (s.flags & flag_match_not_bol) == 0) {
                        s.do_ = .accept_but_not_consume;
                        s.node = n.first;
                    } else {
                        reject(s);
                    }
                },
                .r_anchor => {
                    if (s.current == s.last and (s.flags & flag_match_not_eol) == 0) {
                        s.do_ = .accept_but_not_consume;
                        s.node = n.first;
                    } else {
                        reject(s);
                    }
                },
                .match_any_but_newline => {
                    if (s.current != s.last) {
                        // Measured: `__match_any_but_newline<wchar_t>`
                        // also rejects U+2028 and U+2029.
                        const ch: u32 = input[s.current];
                        if (ch == '\r' or ch == '\n' or (CharT != u8 and (ch == 0x2028 or ch == 0x2029))) {
                            reject(s);
                        } else {
                            s.do_ = .accept_and_consume;
                            s.current += 1;
                            s.node = n.first;
                        }
                    } else {
                        reject(s);
                    }
                },
                .match_char => {
                    if (s.current != s.last and input[s.current] == n.ch) {
                        s.do_ = .accept_and_consume;
                        s.current += 1;
                        s.node = n.first;
                    } else {
                        reject(s);
                    }
                },
                .bracket => {
                    const b = &self.brackets.items[n.aux];
                    var found = false;
                    var consumed: usize = 0;
                    if (s.current != s.last) {
                        consumed = 1;
                        found = b.matches(input[s.current]);
                    } else {
                        found = b.negate; // force reject
                    }
                    if (found != b.negate) {
                        s.do_ = .accept_and_consume;
                        s.current += consumed;
                        s.node = n.first;
                    } else {
                        reject(s);
                    }
                },
                .lookahead => {
                    const la = self.lookaheads.items[n.aux];
                    var m = Match.init();
                    defer m.deinit();
                    m.reset(1 + la.marked_count, s.last);
                    const matched = try la.matchAtStartEcma(
                        input,
                        s.current,
                        s.last,
                        &m,
                        (s.flags | flag_match_continuous) & ~flag_full_match,
                        s.at_first and s.current == s.first,
                    );
                    if (matched != n.invert) {
                        s.do_ = .accept_but_not_consume;
                        s.node = n.first;
                        var i: usize = 1;
                        while (i < m.subs.items.len) : (i += 1) {
                            s.sub_matches[n.mexp + i - 1] = m.subs.items[i];
                        }
                    } else {
                        reject(s);
                    }
                },
            }
        }

        /// The `__exec_split` of `__loop` and `__alternate`.
        fn execSplit(self: *const Self, second: bool, s: *State) void {
            const n = &self.nodes.items[s.node];
            s.do_ = .accept_but_not_consume;
            switch (n.kind) {
                .loop => {
                    if (n.greedy != second) {
                        s.node = n.first;
                        initRepeat(n, s);
                    } else {
                        s.node = n.second;
                    }
                },
                .alternate => s.node = if (second) n.second else n.first,
                else => unreachable,
            }
        }
    };
}

fn reject(s: *State) void {
    s.do_ = .reject;
    s.node = none;
}

/// Ports `__loop::__init_repeat`: an iteration starts with its own
/// captures unset.
fn initRepeat(n: *const Node, s: *State) void {
    s.loop_data[n.loop_id].pos = s.current;
    var i: usize = n.mexp_begin - 1;
    while (i != n.mexp_end - 1) : (i += 1) {
        s.sub_matches[i] = .{ .first = s.last, .second = s.last, .matched = false };
    }
}

/// `_LIBCPP_REGEX_COMPLEXITY_FACTOR`.
const complexity_factor: i32 = 4096;

// `regex_constants::match_flag_type`, with libc++'s values.
const flag_match_not_bol: u32 = 1 << 0;
const flag_match_not_eol: u32 = 1 << 1;
const flag_match_not_bow: u32 = 1 << 2;
const flag_match_not_eow: u32 = 1 << 3;
const flag_match_not_null: u32 = 1 << 5;
const flag_match_continuous: u32 = 1 << 6;
const flag_match_prev_avail: u32 = 1 << 7;
const flag_no_update_pos: u32 = 1 << 11;
const flag_full_match: u32 = 1 << 12;

/// The `regex_constants::syntax_option_type` flags beyond `ECMAScript`
/// that are supported. libc++ reads `optimize` nowhere, so it is not here.
pub const Syntax = struct {
    nosubs: bool = false,
};

/// `std::regex`.
pub const Regex = BasicRegex(u8);
/// `std::wregex`, over code points: `wchar_t` is 32 bits here and every
/// value it holds is non-negative.
pub const WRegex = BasicRegex(u32);

/// `sub_match<const char *>`, as offsets.
pub const SubMatch = struct {
    first: usize,
    second: usize,
    matched: bool,
};

const LoopData = struct {
    count: usize,
    pos: usize,
};

/// Ports `__state<char>`.
const State = struct {
    do_: Do,
    first: usize,
    current: usize,
    last: usize,
    sub_matches: []SubMatch,
    loop_data: []LoopData,
    node: u32,
    flags: u32,
    at_first: bool,

    fn deinit(self: *State) void {
        allocator.free(self.sub_matches);
        allocator.free(self.loop_data);
    }

    fn clone(self: *const State) State {
        var out = self.*;
        out.sub_matches = allocator.dupe(SubMatch, self.sub_matches) catch oom();
        out.loop_data = allocator.dupe(LoopData, self.loop_data) catch oom();
        return out;
    }
};

/// `__state::__do_`.
const Do = enum {
    none,
    end_state,
    accept_and_consume,
    accept_but_not_consume,
    reject,
    split,
    repeat,
};

/// `std::smatch`, as offsets into the input.
pub const Match = struct {
    subs: std.ArrayList(SubMatch) = .empty,

    /// Return: an empty match.
    pub fn init() Match {
        return .{};
    }

    /// Return: nothing.
    pub fn deinit(self: *Match) void {
        self.subs.deinit(allocator);
    }

    fn reset(self: *Match, n: usize, last: usize) void {
        self.subs.resize(allocator, n) catch oom();
        @memset(self.subs.items, .{ .first = last, .second = last, .matched = false });
    }

    /// `match_results::size()`.
    pub fn size(self: *const Match) usize {
        return self.subs.items.len;
    }

    /// `match_results::position(i)`.
    pub fn position(self: *const Match, i: usize) usize {
        return self.subs.items[i].first;
    }

    /// `match_results::length(i)`.
    pub fn length(self: *const Match, i: usize) usize {
        const sm = self.subs.items[i];
        return if (sm.matched) sm.second - sm.first else 0;
    }

    /// Whether sub-match `i` participated.
    pub fn matched(self: *const Match, i: usize) bool {
        return self.subs.items[i].matched;
    }
};

// -----------------------------------------------------------------------------
// The ECMAScript parser

fn Parser(comptime CharT: type) type {
    return struct {
        const Self = @This();

        re: *BasicRegex(CharT),
        pat: []const CharT,

        fn at(self: *const Self, i: usize) u32 {
            return self.pat[i];
        }

        /// Ports `basic_regex::__parse`: a fresh `__start_` and end node, then
        /// the expression.
        fn parse(self: *Self, first: usize) Error!usize {
            const h = self.re.addNode(.{ .kind = .end_state });
            self.re.start = self.re.addNode(.{ .kind = .empty_state, .first = h });
            self.re.end = self.re.start;
            return self.parseEcmaExp(first);
        }

        fn pushChar(self: *Self, ch: u32) void {
            _ = self.re.pushNode(.{ .kind = .match_char, .ch = ch });
        }

        fn pushEmpty(self: *Self) void {
            _ = self.re.pushNode(.{ .kind = .empty_state });
        }

        /// Ports `basic_regex::__push_alternation`.
        fn pushAlternation(self: *Self, sa: u32, ea: u32) void {
            const re = self.re;
            const alt = re.addNode(.{ .kind = .alternate, .first = re.nodes.items[sa].first, .second = re.nodes.items[ea].first });
            re.nodes.items[sa].first = alt;
            const e = re.addNode(.{ .kind = .empty_state, .first = re.nodes.items[re.end].first });
            re.nodes.items[ea].first = e;
            const non_own = re.addNode(.{ .kind = .empty_non_own, .first = e });
            re.nodes.items[re.end].first = non_own;
            re.end = e;
        }

        /// Ports `basic_regex::__push_loop`.
        fn pushLoop(self: *Self, min: usize, max: usize, s: u32, mexp_begin: u32, mexp_end: u32, greedy: bool) void {
            const re = self.re;
            const e1 = re.addNode(.{ .kind = .empty_state, .first = re.nodes.items[re.end].first });
            re.nodes.items[re.end].first = none;
            const e2 = re.addNode(.{
                .kind = .loop,
                .first = re.nodes.items[s].first,
                .second = e1,
                .loop_id = re.loop_count,
                .min = min,
                .max = max,
                .mexp_begin = mexp_begin,
                .mexp_end = mexp_end,
                .greedy = greedy,
            });
            re.nodes.items[s].first = none;
            const r1 = re.addNode(.{ .kind = .repeat_one_loop, .first = e2 });
            re.nodes.items[re.end].first = r1;
            re.end = e1;
            re.nodes.items[s].first = e2;
            re.loop_count += 1;
        }

        fn startMatchingList(self: *Self, negate: bool) u32 {
            const re = self.re;
            re.brackets.append(allocator, .{ .negate = negate }) catch oom();
            const idx: u32 = @intCast(re.brackets.items.len - 1);
            _ = re.pushNode(.{ .kind = .bracket, .aux = idx });
            return idx;
        }

        fn bracket(self: *Self, idx: u32) *Bracket {
            return &self.re.brackets.items[idx];
        }

        /// Ports `basic_regex::__parse_ecma_exp`.
        fn parseEcmaExp(self: *Self, first_in: usize) Error!usize {
            var first = first_in;
            const sa = self.re.end;
            var temp = try self.parseAlternative(first);
            if (temp == first) self.pushEmpty();
            first = temp;
            while (first != self.pat.len and self.at(first) == '|') {
                const sb = self.re.end;
                first += 1;
                temp = try self.parseAlternative(first);
                if (temp == first) self.pushEmpty();
                self.pushAlternation(sa, sb);
                first = temp;
            }
            return first;
        }

        /// Ports `basic_regex::__parse_alternative`.
        fn parseAlternative(self: *Self, first_in: usize) Error!usize {
            var first = first_in;
            while (true) {
                const temp = try self.parseTerm(first);
                if (temp == first) break;
                first = temp;
            }
            return first;
        }

        /// Ports `basic_regex::__parse_term`.
        fn parseTerm(self: *Self, first: usize) Error!usize {
            var temp = try self.parseAssertion(first);
            if (temp == first) {
                const e = self.re.end;
                const mexp_begin = self.re.marked_count;
                temp = try self.parseAtom(first);
                if (temp != first) {
                    return self.parseDuplSymbol(temp, e, mexp_begin + 1, self.re.marked_count + 1);
                }
                return first;
            }
            return temp;
        }

        /// Ports `basic_regex::__parse_assertion`.
        fn parseAssertion(self: *Self, first: usize) Error!usize {
            const last = self.pat.len;
            if (first == last) return first;
            switch (self.at(first)) {
                '^' => {
                    _ = self.re.pushNode(.{ .kind = .l_anchor });
                    return first + 1;
                },
                '$' => {
                    _ = self.re.pushNode(.{ .kind = .r_anchor });
                    return first + 1;
                },
                '\\' => {
                    const temp = first + 1;
                    if (temp != last) {
                        if (self.at(temp) == 'b') {
                            _ = self.re.pushNode(.{ .kind = .word_boundary, .invert = false });
                            return temp + 1;
                        } else if (self.at(temp) == 'B') {
                            _ = self.re.pushNode(.{ .kind = .word_boundary, .invert = true });
                            return temp + 1;
                        }
                    }
                },
                '(' => {
                    var temp = first + 1;
                    if (temp != last and self.at(temp) == '?') {
                        temp += 1;
                        if (temp != last) {
                            const ch = self.at(temp);
                            if (ch == '=' or ch == '!') {
                                const sub = allocator.create(BasicRegex(CharT)) catch oom();
                                sub.* = .{ .nosubs = self.re.nosubs };
                                var sp: Self = .{ .re = sub, .pat = self.pat };
                                temp = sp.parse(temp + 1) catch |err| {
                                    sub.deinit();
                                    allocator.destroy(sub);
                                    return err;
                                };
                                const mexp = sub.marked_count;
                                self.re.lookaheads.append(allocator, sub) catch oom();
                                _ = self.re.pushNode(.{
                                    .kind = .lookahead,
                                    .invert = ch == '!',
                                    .mexp = self.re.marked_count,
                                    .aux = @intCast(self.re.lookaheads.items.len - 1),
                                });
                                self.re.marked_count += mexp;
                                if (temp == last or self.at(temp) != ')') return error.Paren;
                                return temp + 1;
                            }
                        }
                    }
                },
                else => {},
            }
            return first;
        }

        /// Ports `basic_regex::__parse_atom`.
        fn parseAtom(self: *Self, first_in: usize) Error!usize {
            var first = first_in;
            const last = self.pat.len;
            if (first == last) return first;
            switch (self.at(first)) {
                '.' => {
                    _ = self.re.pushNode(.{ .kind = .match_any_but_newline });
                    return first + 1;
                },
                '\\' => return self.parseAtomEscape(first),
                '[' => return self.parseBracketExpression(first),
                '(' => {
                    first += 1;
                    if (first == last) return error.Paren;
                    const temp = first + 1;
                    if (temp != last and self.at(first) == '?' and self.at(temp) == ':') {
                        self.re.open_count += 1;
                        first = try self.parseEcmaExp(temp + 1);
                        if (first == last or self.at(first) != ')') return error.Paren;
                        self.re.open_count -= 1;
                        return first + 1;
                    } else {
                        // `__push_begin_marked_subexpression`: nothing
                        // under `nosubs`, and the count does not move.
                        if (!self.re.nosubs) {
                            self.re.marked_count += 1;
                            _ = self.re.pushNode(.{ .kind = .begin_marked, .mexp = self.re.marked_count });
                        }
                        const temp_count = self.re.marked_count;
                        self.re.open_count += 1;
                        first = try self.parseEcmaExp(first);
                        if (first == last or self.at(first) != ')') return error.Paren;
                        if (!self.re.nosubs) _ = self.re.pushNode(.{ .kind = .end_marked, .mexp = temp_count });
                        self.re.open_count -= 1;
                        return first + 1;
                    }
                },
                '*', '+', '?', '{' => return error.BadRepeat,
                else => return self.parsePatternCharacter(first),
            }
        }

        /// Ports `basic_regex::__parse_atom_escape`.
        fn parseAtomEscape(self: *Self, first: usize) Error!usize {
            const last = self.pat.len;
            if (first != last and self.at(first) == '\\') {
                const t1 = first + 1;
                if (t1 == last) return error.Escape;

                var t2 = try self.parseDecimalEscape(t1);
                if (t2 != t1) return t2;
                t2 = self.parseCharacterClassEscape(t1);
                if (t2 != t1) return t2;
                t2 = try self.parseCharacterEscape(t1, null);
                if (t2 != t1) return t2;
            }
            return first;
        }

        /// Ports `basic_regex::__parse_decimal_escape`.
        fn parseDecimalEscape(self: *Self, first_in: usize) Error!usize {
            var first = first_in;
            const last = self.pat.len;
            if (first != last) {
                if (self.at(first) == '0') {
                    self.pushChar(0);
                    first += 1;
                } else if ('1' <= self.at(first) and self.at(first) <= '9') {
                    var v: u32 = self.at(first) - '0';
                    first += 1;
                    while (first != last and '0' <= self.at(first) and self.at(first) <= '9') : (first += 1) {
                        if (v >= std.math.maxInt(u32) / 10) return error.Backref;
                        v = 10 * v + self.at(first) - '0';
                    }
                    if (v == 0 or v > self.re.marked_count) return error.Backref;
                    _ = self.re.pushNode(.{ .kind = .back_ref, .mexp = v });
                }
            }
            return first;
        }

        /// Ports `basic_regex::__parse_character_class_escape`.
        fn parseCharacterClassEscape(self: *Self, first: usize) usize {
            if (first == self.pat.len) return first;
            switch (self.at(first)) {
                'd', 'D' => {
                    const ml = self.startMatchingList(self.at(first) == 'D');
                    self.bracket(ml).mask |= m_digit;
                },
                's', 'S' => {
                    const ml = self.startMatchingList(self.at(first) == 'S');
                    self.bracket(ml).mask |= m_space;
                },
                'w', 'W' => {
                    const ml = self.startMatchingList(self.at(first) == 'W');
                    self.bracket(ml).mask |= m_alnum;
                    self.bracket(ml).chars.append(allocator, '_') catch oom();
                },
                else => return first,
            }
            return first + 1;
        }

        /// Ports `basic_regex::__parse_character_escape`. With `str`, the
        /// character is returned through it rather than pushed.
        fn parseCharacterEscape(self: *Self, first_in: usize, str: ?*Str) Error!usize {
            var first = first_in;
            const last = self.pat.len;
            if (first == last) return first;

            const emit = struct {
                fn f(p: *Self, s: ?*Str, ch: u32) void {
                    if (s) |o| o.set1(ch) else p.pushChar(ch);
                }
            }.f;

            switch (self.at(first)) {
                'f' => {
                    emit(self, str, 0xC);
                    first += 1;
                },
                'n' => {
                    emit(self, str, 0xA);
                    first += 1;
                },
                'r' => {
                    emit(self, str, 0xD);
                    first += 1;
                },
                't' => {
                    emit(self, str, 0x9);
                    first += 1;
                },
                'v' => {
                    emit(self, str, 0xB);
                    first += 1;
                },
                'c' => {
                    const t = first + 1;
                    if (t != last) {
                        const ch = self.at(t);
                        if (('A' <= ch and ch <= 'Z') or ('a' <= ch and ch <= 'z')) {
                            emit(self, str, ch % 32);
                            first = t + 1;
                        } else return error.Escape;
                    } else return error.Escape;
                },
                'u', 'x' => {
                    var sum: u32 = 0;
                    const n_digits: usize = if (self.at(first) == 'u') 4 else 2;
                    first += 1;
                    for (0..n_digits) |_| {
                        if (first == last) return error.Escape;
                        const hd = traitsValueOf(CharT, self.at(first), 16);
                        if (hd == -1) return error.Escape;
                        sum = 16 * sum + @as(u32, @intCast(hd));
                        first += 1;
                    }
                    // `_CharT(__sum)`: the low byte of a `char`, all of a `wchar_t`.
                    emit(self, str, if (CharT == u8) sum & 0xFF else sum);
                },
                '0' => {
                    emit(self, str, 0);
                    first += 1;
                },
                else => {
                    if (!isctype(self.at(first), m_alnum)) {
                        emit(self, str, self.at(first));
                        first += 1;
                    } else return error.Escape;
                },
            }
            return first;
        }

        /// Ports `basic_regex::__parse_pattern_character`.
        fn parsePatternCharacter(self: *Self, first: usize) usize {
            if (first == self.pat.len) return first;
            switch (self.at(first)) {
                '^', '$', '\\', '.', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|' => return first,
                else => {
                    self.pushChar(self.at(first));
                    return first + 1;
                },
            }
        }

        /// Ports `basic_regex::__parse_DUP_COUNT`.
        fn parseDupCount(self: *Self, first_in: usize, out: *i32) Error!usize {
            var first = first_in;
            const last = self.pat.len;
            if (first != last) {
                var val = traitsValueOf(CharT, self.at(first), 10);
                if (val != -1) {
                    out.* = val;
                    first += 1;
                    while (first != last) : (first += 1) {
                        val = traitsValueOf(CharT, self.at(first), 10);
                        if (val == -1) break;
                        if (out.* >= std.math.maxInt(i32) / 10) return error.BadBrace;
                        out.* *= 10;
                        out.* += val;
                    }
                }
            }
            return first;
        }

        /// Ports `basic_regex::__parse_ERE_dupl_symbol` under ECMAScript, where a
        /// trailing `?` makes a quantifier lazy.
        fn parseDuplSymbol(self: *Self, first_in: usize, s: u32, mexp_begin: u32, mexp_end: u32) Error!usize {
            var first = first_in;
            const last = self.pat.len;
            const inf = std.math.maxInt(usize);
            if (first == last) return first;
            switch (self.at(first)) {
                '*', '+', '?' => {
                    const ch = self.at(first);
                    first += 1;
                    const lazy = first != last and self.at(first) == '?';
                    if (lazy) first += 1;
                    const min: usize = if (ch == '+') 1 else 0;
                    const max: usize = if (ch == '?') 1 else inf;
                    self.pushLoop(min, max, s, mexp_begin, mexp_end, !lazy);
                },
                '{' => {
                    var min: i32 = 0;
                    first += 1;
                    var temp = try self.parseDupCount(first, &min);
                    if (temp == first) return error.BadBrace;
                    first = temp;
                    if (first == last) return error.Brace;
                    switch (self.at(first)) {
                        '}' => {
                            first += 1;
                            const lazy = first != last and self.at(first) == '?';
                            if (lazy) first += 1;
                            self.pushLoop(@intCast(min), @intCast(min), s, mexp_begin, mexp_end, !lazy);
                        },
                        ',' => {
                            first += 1;
                            if (first == last) return error.BadBrace;
                            if (self.at(first) == '}') {
                                first += 1;
                                const lazy = first != last and self.at(first) == '?';
                                if (lazy) first += 1;
                                self.pushLoop(@intCast(min), inf, s, mexp_begin, mexp_end, !lazy);
                            } else {
                                var max: i32 = -1;
                                temp = try self.parseDupCount(first, &max);
                                if (temp == first) return error.Brace;
                                first = temp;
                                if (first == last or self.at(first) != '}') return error.Brace;
                                first += 1;
                                if (max < min) return error.BadBrace;
                                const lazy = first != last and self.at(first) == '?';
                                if (lazy) first += 1;
                                self.pushLoop(@intCast(min), @intCast(max), s, mexp_begin, mexp_end, !lazy);
                            }
                        },
                        else => return error.BadBrace,
                    }
                },
                else => {},
            }
            return first;
        }

        /// Ports `basic_regex::__parse_bracket_expression` under ECMAScript,
        /// where `]` first in a class closes it.
        fn parseBracketExpression(self: *Self, first_in: usize) Error!usize {
            var first = first_in;
            const last = self.pat.len;
            if (first != last and self.at(first) == '[') {
                first += 1;
                if (first == last) return error.Brack;
                var negate = false;
                if (self.at(first) == '^') {
                    first += 1;
                    negate = true;
                }
                const ml = self.startMatchingList(negate);
                if (first == last) return error.Brack;
                while (true) {
                    const temp = try self.parseExpressionTerm(first, ml);
                    if (temp == first) break;
                    first = temp;
                }
                if (first == last) return error.Brack;
                if (self.at(first) == '-') {
                    self.bracket(ml).chars.append(allocator, '-') catch oom();
                    first += 1;
                }
                if (first == last or self.at(first) != ']') return error.Brack;
                first += 1;
            }
            return first;
        }

        /// Ports `basic_regex::__parse_expression_term` under ECMAScript.
        fn parseExpressionTerm(self: *Self, first_in: usize, ml: u32) Error!usize {
            var first = first_in;
            const last = self.pat.len;
            if (first == last or self.at(first) == ']') return first;

            var temp = first + 1;
            var start_range: Str = .{};
            if (temp != last and self.at(first) == '[') {
                if (self.at(temp) == '=') {
                    return self.parseEquivalenceClass(temp + 1, ml);
                } else if (self.at(temp) == ':') {
                    return self.parseCharacterClass(temp + 1, ml);
                } else if (self.at(temp) == '.') {
                    first = try self.parseCollatingSymbol(temp + 1, &start_range);
                }
            }
            if (start_range.len == 0) {
                if (self.at(first) == '\\') {
                    first = try self.parseClassEscape(first + 1, &start_range, ml);
                } else {
                    start_range.set1(self.at(first));
                    first += 1;
                }
            }
            if (first != last and self.at(first) != ']') {
                temp = first + 1;
                if (temp != last and self.at(first) == '-' and self.at(temp) != ']') {
                    var end_range: Str = .{};
                    first = temp;
                    temp += 1;
                    if (temp != last and self.at(first) == '[' and self.at(temp) == '.') {
                        first = try self.parseCollatingSymbol(temp + 1, &end_range);
                    } else if (self.at(first) == '\\') {
                        first = try self.parseClassEscape(first + 1, &end_range, ml);
                    } else {
                        end_range.set1(self.at(first));
                        first += 1;
                    }
                    try self.bracket(ml).addRange(start_range.slice(), end_range.slice());
                } else if (start_range.len != 0) {
                    self.bracket(ml).addStart(start_range.slice());
                }
            } else if (start_range.len != 0) {
                self.bracket(ml).addStart(start_range.slice());
            }
            return first;
        }

        /// Ports `basic_regex::__parse_class_escape`.
        fn parseClassEscape(self: *Self, first: usize, str: *Str, ml: u32) Error!usize {
            if (first == self.pat.len) return error.Escape;
            const b = self.bracket(ml);
            switch (self.at(first)) {
                0 => {
                    str.set1(0);
                    return first + 1;
                },
                'b' => {
                    str.set1(8);
                    return first + 1;
                },
                'd' => {
                    b.mask |= m_digit;
                    return first + 1;
                },
                'D' => {
                    b.neg_mask |= m_digit;
                    return first + 1;
                },
                's' => {
                    b.mask |= m_space;
                    return first + 1;
                },
                'S' => {
                    b.neg_mask |= m_space;
                    return first + 1;
                },
                'w' => {
                    b.mask |= m_alnum;
                    b.chars.append(allocator, '_') catch oom();
                    return first + 1;
                },
                'W' => {
                    b.neg_mask |= m_alnum;
                    b.neg_chars.append(allocator, '_') catch oom();
                    return first + 1;
                },
                else => return self.parseCharacterEscape(first, str),
            }
        }

        /// `std::search` for a two-byte closer.
        fn searchClose(self: *const Self, first: usize, a: u8, b: u8) ?usize {
            var i = first;
            while (i + 1 < self.pat.len) : (i += 1) {
                if (self.at(i) == a and self.at(i + 1) == b) return i;
            }
            return null;
        }

        /// Ports `basic_regex::__parse_equivalence_class`.
        fn parseEquivalenceClass(self: *Self, first: usize, ml: u32) Error!usize {
            const temp = self.searchClose(first, '=', ']') orelse return error.Brack;
            const collate_name = lookupCollatename(CharT, self.pat[first..temp]) orelse return error.Collate;
            // In the "C" locale `transform_primary` of one byte is that byte.
            self.bracket(ml).equivalences.append(allocator, collate_name) catch oom();
            return temp + 2;
        }

        /// Ports `basic_regex::__parse_character_class`.
        fn parseCharacterClass(self: *Self, first: usize, ml: u32) Error!usize {
            const temp = self.searchClose(first, ':', ']') orelse return error.Brack;
            const class_type = lookupClassname(CharT, self.pat[first..temp]);
            if (class_type == 0) return error.Ctype;
            self.bracket(ml).mask |= class_type;
            return temp + 2;
        }

        /// Ports `basic_regex::__parse_collating_symbol`.
        fn parseCollatingSymbol(self: *Self, first: usize, col_sym: *Str) Error!usize {
            const temp = self.searchClose(first, '.', ']') orelse return error.Brack;
            const ch = lookupCollatename(CharT, self.pat[first..temp]) orelse return error.Collate;
            col_sym.set1(ch);
            return temp + 2;
        }
    };
}

/// A `basic_string<char>` of at most two bytes: a bracket's range end or
/// collating element.
const Str = struct {
    buf: [2]u32 = undefined,
    len: usize = 0,

    fn set1(self: *Str, ch: u32) void {
        self.buf[0] = ch;
        self.len = 1;
    }

    fn slice(self: *const Str) []const u32 {
        return self.buf[0..self.len];
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "search finds the leftmost match and its first group" {
    var re = try Regex.compile("[\\s\\S]*?(<tool_call>)[\\s\\S]*");
    defer re.deinit();
    var m = Match.init();
    defer m.deinit();
    try std.testing.expect(re.search("say <tool_call>{}", &m));
    try std.testing.expectEqual(@as(usize, 0), m.position(0));
    try std.testing.expectEqual(@as(usize, 4), m.position(1));
    try std.testing.expectEqual(@as(usize, 11), m.length(1));
}

test "an unescaped closing bracket is a parse error, as in libc++" {
    try std.testing.expectError(error.Parse, Regex.compile("a]"));
    try std.testing.expectError(error.Paren, Regex.compile("(a"));
    try std.testing.expectError(error.BadRepeat, Regex.compile("*a"));
}

test "match requires the whole input" {
    var re = try Regex.compile("a+");
    defer re.deinit();
    var m = Match.init();
    defer m.deinit();
    try std.testing.expect(re.match("aaa", &m));
    try std.testing.expect(!re.match("aab", &m));
}
