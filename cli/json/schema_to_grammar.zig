//! `json_schema_to_grammar`: a JSON schema to a GBNF grammar.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/json-schema-to-grammar.h` and
//! `llama.cpp/common/json-schema-to-grammar.cpp`, with the three string
//! helpers it takes from `llama.cpp/common/common.cpp` (llama.cpp v0.8.0,
//! `8642541b2`). Each declaration names what it ports and the line it
//! began at.
//!
//! # What it stands on
//!
//! The schema arrives as a `schema.Document` (`json-schema.cpp`), the JSON
//! as `cli/json/`'s `Value`, and its three `std::regex`es run through
//! `src/llama/regex.zig`, which reproduces the libc++ the reference links.
//! The regexes are single character classes, but `regex.zig` is how the
//! port answers every `std::regex` question, bytes above 0x7F included.
//!
//! # Built with `LLAMA_USE_LLGUIDANCE` off
//!
//! As the reference builds it: `force_gbnf` is ignored and the grammar is
//! always GBNF.
//!
//! # Allocation
//!
//! A converter keeps every rule, name and intermediate string in one arena
//! and hands back a copy of the grammar on the caller's allocator. Strings
//! in the rule map are never changed in place, so the snapshot
//! `_visit_pattern` takes before a pattern and restores after a failed one
//! is a shallow copy of the map.

const std = @import("std");
const json = @import("module.zig");
const schema = @import("schema.zig");
const trie_mod = @import("trie.zig");
const regex = @import("llama").regex;

const Value = json.Value;
const Node = schema.Node;
pub const Error = schema.Error;
const raise = schema.raise;

/// Ports `common_grammar_options` (json-schema-to-grammar.h:17 @cd959bae9).
pub const Options = struct {
    dotall: bool = false,
};

/// Where `check_errors`' warning goes: the whole line, `WARNING: ...\n`.
/// Null writes it to stderr, as the C++'s `fprintf(stderr, ...)` does; a
/// harness sets it to capture the line.
pub var warning_sink: ?*const fn (line: []const u8) void = null;

// -----------------------------------------------------------------------------
// common.cpp's string helpers

/// Ports `string_join` (common/common.cpp:550 @cd959bae9).
fn stringJoin(a: std.mem.Allocator, values: []const []const u8, sep: []const u8) Error![]u8 {
    return std.mem.join(a, sep, values);
}

/// Ports `string_repeat` (common/common.cpp:577 @cd959bae9).
fn stringRepeat(a: std.mem.Allocator, s: []const u8, n: usize) Error![]u8 {
    const out = try a.alloc(u8, s.len * n);
    for (0..n) |k| @memcpy(out[k * s.len ..][0..s.len], s);
    return out;
}

/// Ports `string_split` (common/common.cpp:561 @cd959bae9), the overload
/// taking a string delimiter: every piece, empty ones included.
fn stringSplit(a: std.mem.Allocator, s: []const u8, delim: []const u8) Error![][]const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitSequence(u8, s, delim);
    while (it.next()) |p| try parts.append(a, p);
    return parts.items;
}

/// `std::stoi(s)`: `strtol` in base 10 -- leading white space, an optional
/// sign, at least one digit, the rest ignored -- then a range check to
/// `int`.
///
/// Return: the value, or null where `stoi` throws (`invalid_argument` or
/// `out_of_range`, both `std::logic_error`).
fn stoi(s: []const u8) ?c_int {
    var k: usize = 0;
    while (k < s.len and schema.isCSpace(s[k])) k += 1;
    var negative = false;
    if (k < s.len and (s[k] == '+' or s[k] == '-')) {
        negative = s[k] == '-';
        k += 1;
    }
    const start = k;
    var r: i128 = 0;
    while (k < s.len and s[k] >= '0' and s[k] <= '9') : (k += 1) {
        // past `long`'s range strtol reports ERANGE; keep reading digits
        if (r <= std.math.maxInt(i64) + 1) r = r * 10 + (s[k] - '0');
    }
    if (k == start) return null;
    if (negative) r = -r;
    if (r < std.math.minInt(c_int) or r > std.math.maxInt(c_int)) return null;
    return @intCast(r);
}

fn intText(a: std.mem.Allocator, v: anytype) Error![]u8 {
    return std.fmt.allocPrint(a, "{d}", .{v});
}

// -----------------------------------------------------------------------------
// Repetition and integer ranges

/// Ports `build_repetition` (json-schema-to-grammar.cpp:18 @cd959bae9).
/// `max_items == maxInt(c_int)` means unbounded. The C++'s `int`
/// arithmetic wraps here, where a pattern's `{-1,}` reaches it.
fn buildRepetition(a: std.mem.Allocator, item_rule: []const u8, min_items: c_int, max_items: c_int, separator_rule: []const u8) Error![]u8 {
    const has_max = max_items != std.math.maxInt(c_int);

    if (max_items == 0) return a.dupe(u8, "");
    if (min_items == 0 and max_items == 1) return std.mem.concat(a, u8, &.{ item_rule, "?" });

    if (separator_rule.len == 0) {
        if (min_items == 1 and !has_max) return std.mem.concat(a, u8, &.{ item_rule, "+" });
        if (min_items == 0 and !has_max) return std.mem.concat(a, u8, &.{ item_rule, "*" });
        const max_s = if (has_max) try intText(a, max_items) else "";
        return std.mem.concat(a, u8, &.{ item_rule, "{", try intText(a, min_items), ",", max_s, "}" });
    }

    const inner = try std.mem.concat(a, u8, &.{ "(", separator_rule, " ", item_rule, ")" });
    const rest = try buildRepetition(
        a,
        inner,
        if (min_items == 0) 0 else min_items -% 1,
        if (has_max) max_items -% 1 else max_items,
        "",
    );
    var result = try std.mem.concat(a, u8, &.{ item_rule, " ", rest });
    if (min_items == 0) result = try std.mem.concat(a, u8, &.{ "(", result, ")?" });
    return result;
}

/// `build_min_max_int`'s output stream.
const Out = struct {
    a: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,

    fn s(self: *Out, text: []const u8) Error!void {
        try self.buf.appendSlice(self.a, text);
    }

    fn ch(self: *Out, c: u8) Error!void {
        try self.buf.append(self.a, c);
    }

    fn int(self: *Out, v: anytype) Error!void {
        try self.buf.print(self.a, "{d}", .{v});
    }

    /// the `digit_range` lambda (json-schema-to-grammar.cpp:49 @cd959bae9)
    fn digitRange(self: *Out, from: u8, to: u8) Error!void {
        try self.ch('[');
        if (from == to) {
            try self.ch(from);
        } else {
            try self.ch(from);
            try self.ch('-');
            try self.ch(to);
        }
        try self.ch(']');
    }

    /// the `more_digits` lambda (json-schema-to-grammar.cpp:58 @cd959bae9)
    fn moreDigits(self: *Out, min_digits: c_int, max_digits: c_int) Error!void {
        try self.s("[0-9]");
        if (min_digits == max_digits and min_digits == 1) return;
        try self.ch('{');
        try self.int(min_digits);
        if (max_digits != min_digits) {
            try self.ch(',');
            if (max_digits != std.math.maxInt(c_int)) try self.int(max_digits);
        }
        try self.ch('}');
    }

    /// the `uniform_range` lambda (json-schema-to-grammar.cpp:73 @cd959bae9)
    fn uniformRange(self: *Out, from: []const u8, to: []const u8) Error!void {
        var i: usize = 0;
        while (i < from.len and i < to.len and from[i] == to[i]) i += 1;
        if (i > 0) {
            try self.ch('"');
            try self.s(from[0..i]);
            try self.ch('"');
        }
        if (i < from.len and i < to.len) {
            if (i > 0) try self.ch(' ');
            const sub_len = from.len - i - 1;
            if (sub_len > 0) {
                const from_sub = from[i + 1 ..];
                const to_sub = to[i + 1 ..];
                const sub_zeros = try stringRepeat(self.a, "0", sub_len);
                const sub_nines = try stringRepeat(self.a, "9", sub_len);
                const n: c_int = @truncate(@as(i64, @intCast(sub_len)));

                var to_reached = false;
                try self.ch('(');
                if (std.mem.eql(u8, from_sub, sub_zeros)) {
                    try self.digitRange(from[i], to[i] -% 1);
                    try self.ch(' ');
                    try self.moreDigits(n, n);
                } else {
                    try self.ch('[');
                    try self.ch(from[i]);
                    try self.s("] ");
                    try self.ch('(');
                    try self.uniformRange(from_sub, sub_nines);
                    try self.ch(')');
                    if (@as(i32, from[i]) < @as(i32, to[i]) - 1) {
                        try self.s(" | ");
                        if (std.mem.eql(u8, to_sub, sub_nines)) {
                            try self.digitRange(from[i] +% 1, to[i]);
                            to_reached = true;
                        } else {
                            try self.digitRange(from[i] +% 1, to[i] -% 1);
                        }
                        try self.ch(' ');
                        try self.moreDigits(n, n);
                    }
                }
                if (!to_reached) {
                    try self.s(" | ");
                    try self.digitRange(to[i], to[i]);
                    try self.ch(' ');
                    try self.uniformRange(sub_zeros, to_sub);
                }
                try self.ch(')');
            } else {
                try self.ch('[');
                try self.ch(from[i]);
                try self.ch('-');
                try self.ch(to[i]);
                try self.ch(']');
            }
        }
    }
};

/// Ports `build_min_max_int` (json-schema-to-grammar.cpp:45 @cd959bae9).
/// `minInt(i64)` and `maxInt(i64)` mean unbounded. Negating a bound is
/// signed overflow in the C++ at `minInt(i64)`, which wraps; so it does
/// here.
fn buildMinMaxInt(min_in: i64, max_value: i64, out: *Out, decimals_left: c_int, top_level: bool) Error!void {
    var min_value = min_in;
    const has_min = min_value != std.math.minInt(i64);
    const has_max = max_value != std.math.maxInt(i64);
    const a = out.a;

    if (has_min and has_max) {
        if (min_value < 0 and max_value < 0) {
            try out.s("\"-\" (");
            try buildMinMaxInt(0 -% max_value, 0 -% min_value, out, decimals_left, true);
            try out.ch(')');
            return;
        }

        if (min_value < 0) {
            try out.s("\"-\" (");
            try buildMinMaxInt(0, 0 -% min_value, out, decimals_left, true);
            try out.s(") | ");
            min_value = 0;
        }

        var min_s: []const u8 = try intText(a, min_value);
        const max_s = try intText(a, max_value);
        const min_digits = min_s.len;
        const max_digits = max_s.len;

        var digits = min_digits;
        while (digits < max_digits) : (digits += 1) {
            try out.uniformRange(min_s, try stringRepeat(a, "9", digits));
            min_s = try std.mem.concat(a, u8, &.{ "1", try stringRepeat(a, "0", digits) });
            try out.s(" | ");
        }
        try out.uniformRange(min_s, max_s);
        return;
    }

    const less_decimals = @max(decimals_left -% 1, 1);

    if (has_min) {
        if (min_value < 0) {
            try out.s("\"-\" (");
            try buildMinMaxInt(std.math.minInt(i64), 0 -% min_value, out, decimals_left, false);
            try out.s(") | [0] | [1-9] ");
            try out.moreDigits(0, decimals_left -% 1);
        } else if (min_value == 0) {
            if (top_level) {
                try out.s("[0] | [1-9] ");
                try out.moreDigits(0, less_decimals);
            } else {
                try out.moreDigits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            const c: u8 = '0' + @as(u8, @intCast(min_value));
            const range_start: u8 = if (top_level) '1' else '0';
            if (c > range_start) {
                try out.digitRange(range_start, c - 1);
                try out.ch(' ');
                try out.moreDigits(1, less_decimals);
                try out.s(" | ");
            }
            try out.digitRange(c, '9');
            try out.ch(' ');
            try out.moreDigits(0, less_decimals);
        } else {
            const min_s = try intText(a, min_value);
            const len: c_int = @intCast(min_s.len);
            const c = min_s[0];

            if (c > '1') {
                try out.digitRange(if (top_level) '1' else '0', c - 1);
                try out.ch(' ');
                try out.moreDigits(len, less_decimals);
                try out.s(" | ");
            }
            try out.digitRange(c, c);
            try out.s(" (");
            const rest = std.fmt.parseInt(i64, min_s[1..], 10) catch |err| switch (err) {
                error.Overflow, error.InvalidCharacter => unreachable,
            };
            try buildMinMaxInt(rest, std.math.maxInt(i64), out, less_decimals, false);
            try out.ch(')');
            if (c < '9') {
                try out.s(" | ");
                try out.digitRange(c + 1, '9');
                try out.ch(' ');
                try out.moreDigits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                try out.s("\"-\" [1-9] ");
                try out.moreDigits(0, less_decimals);
                try out.s(" | ");
            }
            try buildMinMaxInt(0, max_value, out, decimals_left, true);
        } else {
            try out.s("\"-\" (");
            try buildMinMaxInt(0 -% max_value, std.math.maxInt(i64), out, decimals_left, false);
            try out.ch(')');
        }
        return;
    }

    return raise(error.RuntimeError, &.{"At least one of min_value or max_value must be set"});
}

// -----------------------------------------------------------------------------
// The built-in rules

/// Ports `SPACE_RULE` (json-schema-to-grammar.cpp:229 @cd959bae9).
const space_rule = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

/// Ports `BuiltinRule` (json-schema-to-grammar.cpp:231 @cd959bae9).
const BuiltinRule = struct {
    name: []const u8,
    content: []const u8,
    deps: []const []const u8 = &.{},
};

/// Ports `PRIMITIVE_RULES` (json-schema-to-grammar.cpp:236 @cd959bae9).
/// The C++ map is unordered and only ever looked up.
const primitive_rules = [_]BuiltinRule{
    .{ .name = "boolean", .content = "(\"true\" | \"false\")" },
    .{ .name = "decimal-part", .content = "[0-9]{1,16}" },
    .{ .name = "integral-part", .content = "[0] | [1-9] [0-9]{0,15}" },
    .{ .name = "number", .content = "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", .deps = &.{ "integral-part", "decimal-part" } },
    .{ .name = "integer", .content = "(\"-\"? integral-part)", .deps = &.{"integral-part"} },
    .{ .name = "value", .content = "object | array | string | number | boolean | null", .deps = &.{ "object", "array", "string", "number", "boolean", "null" } },
    .{ .name = "object", .content = "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", .deps = &.{ "string", "value" } },
    .{ .name = "array", .content = "\"[\" space ( value (\",\" space value)* )? space \"]\"", .deps = &.{"value"} },
    .{ .name = "uuid", .content = "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"" },
    .{ .name = "char", .content = "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})" },
    .{ .name = "string", .content = "\"\\\"\" char* \"\\\"\"", .deps = &.{"char"} },
    .{ .name = "null", .content = "\"null\"" },
};

/// Ports `STRING_FORMAT_RULES` (json-schema-to-grammar.cpp:251 @cd959bae9).
const string_format_rules = [_]BuiltinRule{
    .{ .name = "date", .content = "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )" },
    .{ .name = "time", .content = "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )" },
    .{ .name = "date-time", .content = "date \"T\" time", .deps = &.{ "date", "time" } },
    .{ .name = "date-string", .content = "\"\\\"\" date \"\\\"\"", .deps = &.{"date"} },
    .{ .name = "time-string", .content = "\"\\\"\" time \"\\\"\"", .deps = &.{"time"} },
    .{ .name = "date-time-string", .content = "\"\\\"\" date-time \"\\\"\"", .deps = &.{"date-time"} },
};

fn findRule(table: []const BuiltinRule, name: []const u8) ?*const BuiltinRule {
    for (table) |*r| {
        if (std.mem.eql(u8, r.name, name)) return r;
    }
    return null;
}

fn primitive(name: []const u8) *const BuiltinRule {
    return findRule(&primitive_rules, name).?;
}

/// Ports `is_reserved_name` (json-schema-to-grammar.cpp:260 @cd959bae9).
fn isReservedName(name: []const u8) bool {
    if (std.mem.eql(u8, name, "root")) return true;
    return findRule(&primitive_rules, name) != null or findRule(&string_format_rules, name) != null;
}

/// Ports `NON_LITERAL_SET` (json-schema-to-grammar.cpp:281 @cd959bae9).
fn isNonLiteral(c: u8) bool {
    return std.mem.indexOfScalar(u8, "|.()[]{}*+?^$", c) != null;
}

/// Ports `ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS` (json-schema-to-grammar.cpp:282 @cd959bae9).
fn isEscapedInRegexpsButNotInLiterals(c: u8) bool {
    return std.mem.indexOfScalar(u8, "^$.[]()|{}*+?", c) != null;
}

/// Ports `GRAMMAR_LITERAL_ESCAPE_RE` (json-schema-to-grammar.cpp:275 @cd959bae9).
/// `GRAMMAR_RANGE_LITERAL_ESCAPE_RE` beside it is never used.
const grammar_literal_escape_re = "[\r\n\"\\\\]";

/// Ports `GRAMMAR_LITERAL_ESCAPES` (json-schema-to-grammar.cpp:277 @cd959bae9).
fn grammarLiteralEscape(c: u8) []const u8 {
    return switch (c) {
        '\r' => "\\r",
        '\n' => "\\n",
        '"' => "\\\"",
        '-' => "\\-",
        ']' => "\\]",
        '\\' => "\\\\",
        else => unreachable,
    };
}

fn compileRegex(pattern: []const u8) regex.Regex {
    return regex.Regex.compile(pattern) catch |e| regex.throwRegexError(e);
}

/// Ports `replacePattern` (json-schema-to-grammar.cpp:284 @cd959bae9):
/// `regex_search` from each match's end, each match replaced by
/// `replacement` of its first character.
fn replacePattern(a: std.mem.Allocator, input: []const u8, re: *const regex.Regex, comptime replacement: fn (u8) []const u8) Error![]u8 {
    var m: regex.Match = .{};
    defer m.deinit();
    var out: std.ArrayList(u8) = .empty;
    var start: usize = 0;
    while (re.search(input[start..], &m)) {
        const sm = m.subs.items[0];
        try out.appendSlice(a, input[start..][0..sm.first]);
        try out.appendSlice(a, replacement(input[start + sm.first]));
        start += sm.second;
    }
    try out.appendSlice(a, input[start..]);
    return out.items;
}

/// Ports `format_literal` (json-schema-to-grammar.cpp:302 @cd959bae9).
fn formatLiteral(a: std.mem.Allocator, re: *const regex.Regex, literal: []const u8) Error![]u8 {
    const escaped = try replacePattern(a, literal, re, grammarLiteralEscape);
    return std.mem.concat(a, u8, &.{ "\"", escaped, "\"" });
}

/// Ports `gbnf_format_literal` (json-schema-to-grammar.cpp:310 @cd959bae9).
///
/// Parameters:
/// - `gpa`: allocates the result.
/// - `literal`: any bytes.
///
/// Return: the GBNF string literal, owned by the caller.
pub fn gbnfFormatLiteral(gpa: std.mem.Allocator, literal: []const u8) Error![]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var re = compileRegex(grammar_literal_escape_re);
    defer re.deinit();
    return gpa.dupe(u8, try formatLiteral(arena.allocator(), &re, literal));
}

/// Ports `gbnf_escape_length` (json-schema-to-grammar.cpp:328 @cd959bae9):
/// the length of a GBNF-compatible escape at `pos`, 0 if there is none.
/// The two-byte escapes are the ones `parse_char` in `llama-grammar.cpp`
/// reads; `\-` joined them at v0.5.0.
fn gbnfEscapeLength(pattern: []const u8, pos: usize) usize {
    if (pos + 1 >= pattern.len or pattern[pos] != '\\') return 0;
    const n_hex: usize = switch (pattern[pos + 1]) {
        'x' => 2,
        'u' => 4,
        'U' => 8,
        't', 'r', 'n', '\\', '"', '[', ']', '-' => return 2,
        else => return 0,
    };
    if (pos + 2 + n_hex > pattern.len) return 0;
    for (pattern[pos + 2 .. pos + 2 + n_hex]) |h| {
        if (!std.ascii.isHex(h)) return 0;
    }
    return 2 + n_hex;
}

// -----------------------------------------------------------------------------
// The converter

/// A rule of the sequence `_pattern_to_rule` builds: `literal_or_rule`.
const LitOrRule = struct {
    s: []const u8,
    literal: bool,
};

/// A property as `_build_object_rule` takes it.
const PropRef = struct {
    name: []const u8,
    schema: *const Node,
};

const StringSet = std.StringHashMapUnmanaged(void);
const RuleMap = std.StringHashMapUnmanaged([]const u8);

/// Ports `gbnf_format_rule_name` (json-schema-to-grammar.cpp:313 @cd959bae9):
/// every run of bytes outside `[a-zA-Z0-9-]` becomes one `-`.
///
/// Parameters:
/// - `a`: owns the result.
/// - `name`: the name.
///
/// Return: the rule name, owned by `a`.
pub fn gbnfFormatRuleName(a: std.mem.Allocator, name: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var in_invalid = false;
    for (name) |c| {
        if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '-') {
            try out.append(a, c);
            in_invalid = false;
        } else if (!in_invalid) {
            try out.append(a, '-');
            in_invalid = true;
        }
    }
    return out.items;
}

/// Ports `common_chat_schema_converter` (json-schema-to-grammar.cpp:355 @cd959bae9).
pub const Converter = struct {
    a: std.mem.Allocator,
    dotall: bool,
    rules: RuleMap = .empty,
    refs_being_resolved: StringSet = .empty,
    errors: std.ArrayList([]const u8) = .empty,
    warnings: std.ArrayList([]const u8) = .empty,
    /// `_depth`: how deep `visit` is, `depth_guard` (json-schema-to-grammar.cpp:365 @cd959bae9)
    depth: c_int = 0,
    re_literal_escape: regex.Regex,

    /// Ports `common_chat_schema_converter` (json-schema-to-grammar.cpp:843 @cd959bae9),
    /// the constructor.
    ///
    /// Parameters:
    /// - `a`: an arena; every string the converter makes lives in it.
    /// - `dotall`: whether a pattern's `.` matches newlines.
    ///
    /// Return: the converter, freed with `deinit` and its arena.
    pub fn init(a: std.mem.Allocator, dotall: bool) Error!Converter {
        var self: Converter = .{
            .a = a,
            .dotall = dotall,
            .re_literal_escape = compileRegex(grammar_literal_escape_re),
        };
        try self.rules.put(a, "space", space_rule);
        return self;
    }

    /// Frees the compiled regexes; the arena owns the rest.
    pub fn deinit(self: *Converter) void {
        self.re_literal_escape.deinit();
    }

    fn cat(self: *Converter, parts: []const []const u8) Error![]u8 {
        return std.mem.concat(self.a, u8, parts);
    }

    fn cati(self: *Converter, prefix: []const u8, i: anytype) Error![]u8 {
        return std.fmt.allocPrint(self.a, "{s}{d}", .{ prefix, i });
    }

    /// Ports `_add_rule` (json-schema-to-grammar.cpp:380 @cd959bae9).
    ///
    /// Return: the rule's name, borrowed from the converter's arena.
    pub fn addRule(self: *Converter, name: []const u8, rule: []const u8) Error![]const u8 {
        const esc_name = try gbnfFormatRuleName(self.a, name);
        if (self.rules.get(esc_name)) |existing| {
            if (!std.mem.eql(u8, existing, rule)) {
                var i: usize = 0;
                while (true) : (i += 1) {
                    const k = try self.cati(esc_name, i);
                    const got = self.rules.get(k) orelse break;
                    if (std.mem.eql(u8, got, rule)) break;
                }
                const key = try self.cati(esc_name, i);
                try self.rules.put(self.a, key, try self.a.dupe(u8, rule));
                return key;
            }
        }
        try self.rules.put(self.a, esc_name, try self.a.dupe(u8, rule));
        return esc_name;
    }

    /// Ports `_generate_union_rule` (json-schema-to-grammar.cpp:395 @cd959bae9).
    fn generateUnionRule(self: *Converter, name: []const u8, alts: []const *Node) Error![]u8 {
        const rules = try self.a.alloc([]const u8, alts.len);
        for (alts, 0..) |alt, i| {
            const prefix = try self.cat(&.{ name, if (name.len == 0) "alternative-" else "-" });
            rules[i] = try self.visit(alt, try self.cati(prefix, i));
        }
        return stringJoin(self.a, rules, " | ");
    }

    /// Ports `_visit_pattern` (json-schema-to-grammar.cpp:414 @cd959bae9):
    /// a pattern with no grammar equivalent degrades to any string with a
    /// warning; an invalid one is an error.
    fn visitPattern(self: *Converter, pattern: []const u8, name: []const u8) Error![]const u8 {
        const snapshot = try self.rules.clone(self.a);
        var p: PatternState = .{ .conv = self, .name = name, .sub = undefined };
        if (p.toRule(pattern)) |rule| {
            return rule;
        } else |e| switch (e) {
            error.UnsupportedPattern => {
                self.rules = snapshot;
                try self.warnings.append(self.a, try self.cat(&.{ "pattern ", pattern, " is not supported (", cstr(p.message), "), accepting any string" }));
                return self.addRule(name, try self.addPrimitive("string", primitive("string")));
            },
            error.InvalidPattern => {
                self.rules = snapshot;
                try self.errors.append(self.a, try self.cat(&.{ "Invalid pattern ", pattern, ": ", cstr(p.message) }));
                return "";
            },
            else => |other| return @as(Error, @errorCast(other)),
        }
    }

    /// Ports `_not_strings` (json-schema-to-grammar.cpp:665 @cd959bae9): a
    /// rule matching a JSON string that is none of `strings`.
    fn notStrings(self: *Converter, strings: []const []const u8) Error![]u8 {
        var t = try trie_mod.Trie.init(self.a, strings);
        const char_rule = try self.addPrimitive("char", primitive("char"));
        var out: Out = .{ .a = self.a };
        try out.s("[\"] ( ");
        try self.notStringsVisit(&t, 0, 0, char_rule, &out);
        try out.s(" )");
        if (t.nodes.items[0].pattern < 0) try out.ch('?');
        try out.s(" [\"]");
        return out.buf.items;
    }

    /// Ports the lambda `visit` (json-schema-to-grammar.cpp:672 @cd959bae9), in
    /// `_not_strings`: one level per char of the longest name.
    fn notStringsVisit(self: *Converter, t: *const trie_mod.Trie, idx: usize, depth: c_int, char_rule: []const u8, out: *Out) Error!void {
        if (depth >= schema.recursive_limit) {
            return raise(error.InvalidArgument, &.{"JSON schema property name is longer than " ++ schema.recursive_limit_text ++ " chars"});
        }
        const node = &t.nodes.items[idx];
        var rejects: std.ArrayList(u8) = .empty;
        var first = true;
        for (node.children.items) |edge| {
            var buf: [4]u8 = undefined;
            const c = trie_mod.cptToUtf8(edge.cpt, &buf) catch
                return raise(error.InvalidArgument, &.{"invalid codepoint"});
            try rejects.appendSlice(self.a, c);
            if (first) {
                first = false;
            } else {
                try out.s(" | ");
            }
            try out.ch('[');
            try out.s(c);
            try out.ch(']');
            if (t.nodes.items[edge.child].children.items.len != 0) {
                try out.s(" (");
                try self.notStringsVisit(t, edge.child, depth + 1, char_rule, out);
                try out.ch(')');
            } else {
                try out.ch(' ');
                try out.s(char_rule);
                try out.ch('+');
            }
        }
        if (node.children.items.len != 0) {
            try out.s(" | [^\"");
            try out.s(rejects.items);
            try out.s("] ");
            try out.s(char_rule);
            try out.ch('*');
        }
    }

    /// Ports `_resolve_ref` (json-schema-to-grammar.cpp:710 @cd959bae9).
    fn resolveRef(self: *Converter, ref: *const schema.Ref) Error![]const u8 {
        const fragment = if (std.mem.indexOfScalar(u8, ref.ref, '#')) |at| ref.ref[at + 1 ..] else ref.ref;
        var ref_name: []const u8 = try self.cat(&.{ "ref", try gbnfFormatRuleName(self.a, fragment) });
        if (!self.rules.contains(ref_name) and !self.refs_being_resolved.contains(ref.ref)) {
            const target = ref.target orelse {
                try self.errors.append(self.a, try self.cat(&.{ "Unresolved $ref ", ref.ref }));
                return "";
            };
            try self.refs_being_resolved.put(self.a, ref.ref, {});
            ref_name = try self.visit(target, ref_name);
            _ = self.refs_being_resolved.remove(ref.ref);
        }
        return ref_name;
    }

    fn contains(set: []const []const u8, s: []const u8) bool {
        for (set) |x| {
            if (std.mem.eql(u8, x, s)) return true;
        }
        return false;
    }

    /// Ports `_build_object_rule` (json-schema-to-grammar.cpp:726 @cd959bae9).
    fn buildObjectRule(
        self: *Converter,
        properties: []const PropRef,
        required: []const []const u8,
        name: []const u8,
        additional_properties: ?*const Node,
    ) Error![]u8 {
        var required_props: std.ArrayList([]const u8) = .empty;
        var optional_props: std.ArrayList([]const u8) = .empty;
        var kv_names: RuleMap = .empty;
        var prop_names: std.ArrayList([]const u8) = .empty;
        const dash: []const u8 = if (name.len == 0) "" else "-";

        for (properties) |kv| {
            const prop_rule_name = try self.visit(kv.schema, try self.cat(&.{ name, dash, kv.name }));
            const key_text = try self.dumpString(kv.name);
            const kv_rule = try self.addRule(
                try self.cat(&.{ name, dash, kv.name, "-kv" }),
                try self.cat(&.{ try formatLiteral(self.a, &self.re_literal_escape, key_text), " space \":\" space ", prop_rule_name }),
            );
            try kv_names.put(self.a, kv.name, kv_rule);
            if (contains(required, kv.name)) {
                try required_props.append(self.a, kv.name);
            } else {
                try optional_props.append(self.a, kv.name);
            }
            try prop_names.append(self.a, kv.name);
        }
        if (additional_properties) |additional| {
            const sub_name = try self.cat(&.{ name, dash, "additional" });
            const value_rule = if (additional.* != .any)
                try self.visit(additional, try self.cat(&.{ sub_name, "-value" }))
            else
                try self.addPrimitive("value", primitive("value"));

            const key_rule = if (prop_names.items.len == 0)
                try self.addPrimitive("string", primitive("string"))
            else
                try self.addRule(try self.cat(&.{ sub_name, "-k" }), try self.notStrings(prop_names.items));
            const kv_rule = try self.addRule(try self.cat(&.{ sub_name, "-kv" }), try self.cat(&.{ key_rule, " \":\" space ", value_rule }));
            try kv_names.put(self.a, "*", kv_rule);
            try optional_props.append(self.a, "*");
        }

        if (required_props.items.len == 0 and optional_props.items.len == 0) {
            return self.a.dupe(u8, "\"{\" space \"}\"");
        }

        var rule: std.ArrayList(u8) = .empty;
        try rule.appendSlice(self.a, "\"{\" space ");
        for (required_props.items, 0..) |p, i| {
            if (i > 0) try rule.appendSlice(self.a, " \",\" space ");
            try rule.appendSlice(self.a, kv_names.get(p) orelse "");
        }

        if (optional_props.items.len != 0) {
            try rule.appendSlice(self.a, " (");
            if (required_props.items.len != 0) try rule.appendSlice(self.a, " \",\" space ( ");

            for (0..optional_props.items.len) |i| {
                if (i > 0) try rule.appendSlice(self.a, " | ");
                try rule.appendSlice(self.a, try self.recursiveRefs(&kv_names, name, optional_props.items[i..], false));
            }
            if (required_props.items.len != 0) try rule.appendSlice(self.a, " )");
            try rule.appendSlice(self.a, " )?");
        }

        try rule.appendSlice(self.a, " space \"}\"");
        return rule.items;
    }

    /// Ports the lambda `get_recursive_refs` (json-schema-to-grammar.cpp:784 @cd959bae9),
    /// in `_build_object_rule`.
    fn recursiveRefs(self: *Converter, kv_names: *const RuleMap, name: []const u8, ks: []const []const u8, first_is_optional: bool) Error![]u8 {
        if (ks.len == 0) return self.a.dupe(u8, "");
        const k = ks[0];
        const star = std.mem.eql(u8, k, "*");
        const kv_rule_name = kv_names.get(k) orelse "";
        const comma_ref = try self.cat(&.{ "( \",\" space ", kv_rule_name, " )" });
        var res = if (first_is_optional)
            try self.cat(&.{ comma_ref, if (star) "*" else "?" })
        else
            try self.cat(&.{ kv_rule_name, if (star) try self.cat(&.{ " ", comma_ref, "*" }) else "" });
        if (ks.len > 1) {
            const rest = try self.recursiveRefs(kv_names, name, ks[1..], true);
            const dash: []const u8 = if (name.len == 0) "" else "-";
            const rest_name = try self.addRule(try self.cat(&.{ name, dash, k, "-rest" }), rest);
            res = try self.cat(&.{ res, " ", rest_name });
        }
        return res;
    }

    /// Ports `_add_primitive` (json-schema-to-grammar.cpp:823 @cd959bae9).
    fn addPrimitive(self: *Converter, name: []const u8, rule: *const BuiltinRule) Error![]const u8 {
        const n = try self.addRule(name, rule.content);
        for (rule.deps) |dep| {
            const dep_rule = findRule(&primitive_rules, dep) orelse findRule(&string_format_rules, dep) orelse {
                try self.errors.append(self.a, try self.cat(&.{ "Rule ", dep, " not known" }));
                continue;
            };
            if (!self.rules.contains(dep)) _ = try self.addPrimitive(dep, dep_rule);
        }
        return n;
    }

    /// Ports `add_schema` (json-schema-to-grammar.cpp:847 @cd959bae9).
    ///
    /// Return: the rule's name, borrowed from the converter's arena.
    pub fn addSchema(self: *Converter, name: []const u8, node: *const Node) Error![]const u8 {
        return self.visit(node, name);
    }

    /// `common_json(s).dump()` of a string: its JSON text, which fails on
    /// invalid UTF-8.
    fn dumpString(self: *Converter, s: []const u8) Error![]u8 {
        const v: Value = .{ .string = @constCast(s) };
        return json.dump(self.a, &v, -1) catch |e| schema.fromJson(e);
    }

    /// Ports `_generate_constant_rule` (json-schema-to-grammar.cpp:851 @cd959bae9).
    fn generateConstantRule(self: *Converter, value: *const Value) Error![]u8 {
        const text = json.dump(self.a, value, -1) catch |e| return schema.fromJson(e);
        return formatLiteral(self.a, &self.re_literal_escape, text);
    }

    /// Ports `_visit_primitive` (json-schema-to-grammar.cpp:855 @cd959bae9).
    fn visitPrimitive(self: *Converter, rule_name: []const u8, type_name: []const u8) Error![]const u8 {
        const n: []const u8 = if (std.mem.eql(u8, rule_name, "root")) "root" else type_name;
        return self.addPrimitive(n, primitive(type_name));
    }

    /// Ports `_visit_all_of` (json-schema-to-grammar.cpp:859 @cd959bae9).
    fn visitAllOf(self: *Converter, children: []const *Node, name: []const u8, rule_name: []const u8) Error![]const u8 {
        var st: AllOfState = .{ .conv = self };
        for (children) |child| {
            if (child.* == .any_of) {
                for (child.any_of) |alt| try st.addComponent(alt, false, 0);
            } else {
                try st.addComponent(child, true, 0);
            }
        }
        if (st.enum_values.count() != 0) {
            // std::map order: bytewise
            var keys: std.ArrayList([]const u8) = .empty;
            var it = st.enum_values.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* == children.len) try keys.append(self.a, e.key_ptr.*);
            }
            std.mem.sort([]const u8, keys.items, {}, lessBytes);
            if (keys.items.len != 0) {
                return self.addRule(rule_name, try self.cat(&.{ "(", try stringJoin(self.a, keys.items, " | "), ")" }));
            }
        }
        return self.addRule(rule_name, try self.buildObjectRule(st.properties.items, st.required.items, name, null));
    }

    /// Ports `visit` (json-schema-to-grammar.cpp:907 @cd959bae9).
    pub fn visit(self: *Converter, node: *const Node, name: []const u8) Error![]const u8 {
        // the $ref targets are visited too, a $ref chain is not bounded by the builder
        try schema.depthEnter(&self.depth);
        defer self.depth -= 1;

        const rule_name: []const u8 = if (isReservedName(name))
            try self.cat(&.{ name, "-" })
        else if (name.len == 0) "root" else name;
        const sub_name = try self.cat(&.{ name, if (name.len == 0) "" else "-" });

        switch (node.*) {
            .ref => |*r| return self.addRule(rule_name, try self.resolveRef(r)),
            .any_of => |alts| return self.addRule(rule_name, try self.generateUnionRule(name, alts)),
            .all_of => |children| return self.visitAllOf(children, name, rule_name),
            .@"const" => |*v| return self.addRule(rule_name, try self.generateConstantRule(v)),
            .@"enum" => |vs| {
                const rules = try self.a.alloc([]const u8, vs.len);
                for (vs, 0..) |*v, i| rules[i] = try self.generateConstantRule(v);
                return self.addRule(rule_name, try self.cat(&.{ "(", try stringJoin(self.a, rules, " | "), ")" }));
            },
            .object => |obj| {
                if (obj.properties.len == 0) {
                    if (obj.additional_properties) |ap| {
                        if (ap.* == .any) return self.addRule(rule_name, try self.addPrimitive("object", primitive("object")));
                    }
                }
                const props = try self.a.alloc(PropRef, obj.properties.len);
                var required: std.ArrayList([]const u8) = .empty;
                for (obj.properties, 0..) |p, i| {
                    props[i] = .{ .name = p.name, .schema = p.schema };
                    if (p.required) try required.append(self.a, p.name);
                }
                return self.addRule(rule_name, try self.buildObjectRule(props, required.items, name, obj.additional_properties));
            },
            .tuple => |items| {
                var rule: std.ArrayList(u8) = .empty;
                try rule.appendSlice(self.a, "\"[\" space ");
                for (items, 0..) |item, i| {
                    if (i > 0) try rule.appendSlice(self.a, " \",\" space ");
                    try rule.appendSlice(self.a, try self.visit(item, try self.cati(try self.cat(&.{ sub_name, "tuple-" }), i)));
                }
                try rule.appendSlice(self.a, " space \"]\"");
                return self.addRule(rule_name, rule.items);
            },
            .array => |arr| {
                if (arr.items.* == .any and arr.min_items == 0 and arr.max_items < 0) {
                    return self.visitPrimitive(rule_name, "array");
                }
                const item_rule_name = try self.visit(arr.items, try self.cat(&.{ sub_name, "item" }));
                const max_items: c_int = if (arr.max_items < 0) std.math.maxInt(c_int) else arr.max_items;
                const rep = try buildRepetition(self.a, item_rule_name, arr.min_items, max_items, "\",\" space");
                return self.addRule(rule_name, try self.cat(&.{ "\"[\" space ", rep, " space \"]\"" }));
            },
            .string => |str| {
                if (str.pattern.len != 0) return self.visitPattern(str.pattern, rule_name);
                if (str.format == .uuid) return self.visitPrimitive(rule_name, "uuid");
                if (str.format != .none) {
                    const prim_name: []const u8 = switch (str.format) {
                        .date => "date-string",
                        .time => "time-string",
                        else => "date-time-string",
                    };
                    return self.addRule(rule_name, try self.addPrimitive(prim_name, findRule(&string_format_rules, prim_name).?));
                }
                if (str.min_length > 0 or str.max_length >= 0) {
                    const char_rule = try self.addPrimitive("char", primitive("char"));
                    const max_len: c_int = if (str.max_length < 0) std.math.maxInt(c_int) else str.max_length;
                    const rep = try buildRepetition(self.a, char_rule, str.min_length, max_len, "");
                    return self.addRule(rule_name, try self.cat(&.{ "\"\\\"\" ", rep, " \"\\\"\"" }));
                }
                return self.visitPrimitive(rule_name, "string");
            },
            .integer => |i| {
                if (i.minimum == std.math.minInt(i64) and i.maximum == std.math.maxInt(i64)) {
                    return self.visitPrimitive(rule_name, "integer");
                }
                var out: Out = .{ .a = self.a };
                try out.ch('(');
                try buildMinMaxInt(i.minimum, i.maximum, &out, 16, true);
                try out.ch(')');
                return self.addRule(rule_name, out.buf.items);
            },
            .number => return self.visitPrimitive(rule_name, "number"),
            .boolean => return self.visitPrimitive(rule_name, "boolean"),
            .null => return self.visitPrimitive(rule_name, "null"),
            .any => return self.addRule(rule_name, try self.addPrimitive("value", primitive("value"))),
        }
    }

    /// Ports `check_errors` (json-schema-to-grammar.cpp:1007 @cd959bae9).
    pub fn checkErrors(self: *Converter) Error!void {
        if (self.errors.items.len != 0) {
            return raise(error.InvalidArgument, &.{ "JSON schema conversion failed:\n", try stringJoin(self.a, self.errors.items, "\n") });
        }
        if (self.warnings.items.len != 0) {
            // `fprintf(stderr, "...%s\n", ...c_str())`: the `%s` stops at a NUL
            const joined = try stringJoin(self.a, self.warnings.items, "; ");
            const line = try self.cat(&.{ "WARNING: JSON schema conversion was incomplete: ", cstr(joined), "\n" });
            if (warning_sink) |sink| sink(line) else json.errors.writeStderr(line);
        }
    }

    /// Ports `format_grammar` (json-schema-to-grammar.cpp:1016 @cd959bae9):
    /// every rule, in `std::map` order.
    ///
    /// Return: the grammar, in the converter's arena.
    pub fn formatGrammar(self: *Converter) Error![]u8 {
        var keys: std.ArrayList([]const u8) = .empty;
        var it = self.rules.keyIterator();
        while (it.next()) |k| try keys.append(self.a, k.*);
        std.mem.sort([]const u8, keys.items, {}, lessBytes);
        var out: std.ArrayList(u8) = .empty;
        for (keys.items) |k| {
            try out.appendSlice(self.a, k);
            try out.appendSlice(self.a, " ::= ");
            try out.appendSlice(self.a, self.rules.get(k).?);
            try out.append(self.a, '\n');
        }
        return out.items;
    }
};

/// The state `_visit_all_of` gathers its components into.
const AllOfState = struct {
    conv: *Converter,
    required: std.ArrayList([]const u8) = .empty,
    properties: std.ArrayList(PropRef) = .empty,
    enum_values: std.StringHashMapUnmanaged(usize) = .empty,

    /// Ports the lambda `add_component` (json-schema-to-grammar.cpp:863 @cd959bae9),
    /// in `_visit_all_of`.
    fn addComponent(self: *AllOfState, comp: *const Node, is_required: bool, depth: c_int) Error!void {
        if (depth >= schema.recursive_limit) {
            return raise(error.InvalidArgument, &.{"JSON schema is nested more than " ++ schema.recursive_limit_text ++ " levels deep"});
        }
        const a = self.conv.a;
        switch (comp.*) {
            .ref => |r| if (r.target) |target| try self.addComponent(target, is_required, depth + 1),
            .object => |obj| for (obj.properties) |p| {
                try self.properties.append(a, .{ .name = p.name, .schema = p.schema });
                if (is_required and !Converter.contains(self.required.items, p.name)) {
                    try self.required.append(a, p.name);
                }
            },
            .@"enum" => |vs| for (vs) |*v| {
                const gop = try self.enum_values.getOrPut(a, try self.conv.generateConstantRule(v));
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += 1;
            },
            else => {},
        }
    }
};

/// The state of one `_pattern_to_rule`: the cursor and depth its
/// `transform` lambda shares across recursion.
const PatternState = struct {
    conv: *Converter,
    name: []const u8,
    sub: []const u8,
    i: usize = 0,
    paren_depth: c_int = 0,
    sub_rule_ids: RuleMap = .empty,
    /// the `what()` of the `unsupported_pattern` or `invalid_pattern` thrown
    message: []const u8 = "",

    const PatternError = Error || error{ UnsupportedPattern, InvalidPattern };

    fn unsupported(self: *PatternState, parts: []const []const u8) PatternError {
        self.message = try std.mem.concat(self.conv.a, u8, parts);
        return error.UnsupportedPattern;
    }

    fn invalid(self: *PatternState, msg: []const u8) PatternError {
        self.message = msg;
        return error.InvalidPattern;
    }

    fn toRuleText(ls: LitOrRule, a: std.mem.Allocator) Error![]const u8 {
        return if (ls.literal) std.mem.concat(a, u8, &.{ "\"", ls.s, "\"" }) else ls.s;
    }

    /// Ports `_pattern_to_rule` (json-schema-to-grammar.cpp:430 @cd959bae9).
    fn toRule(self: *PatternState, pattern: []const u8) PatternError![]const u8 {
        if (pattern.len < 2 or pattern[0] != '^' or pattern[pattern.len - 1] != '$') {
            return self.unsupported(&.{"not anchored with '^' and '$'"});
        }
        self.sub = pattern[1 .. pattern.len - 1];
        const rule = try toRuleText(try self.transform(), self.conv.a);
        if (self.paren_depth != 0) return self.invalid("unbalanced parentheses");
        return self.conv.addRule(self.name, try std.mem.concat(self.conv.a, u8, &.{ "\"\\\"\" (", rule, ") \"\\\"\"" }));
    }

    /// the `get_dot` lambda (json-schema-to-grammar.cpp:450 @cd959bae9)
    fn getDot(self: *PatternState) Error![]const u8 {
        const rule: []const u8 = if (self.conv.dotall) "[\\U00000000-\\U0010FFFF]" else "[^\\x0A\\x0D]";
        return self.conv.addRule("dot", rule);
    }

    /// the `join_seq` lambda (json-schema-to-grammar.cpp:461 @cd959bae9):
    /// consecutive literals merge into one.
    fn joinSeq(self: *PatternState, seq: []const LitOrRule) Error!LitOrRule {
        const a = self.conv.a;
        var results: std.ArrayList([]const u8) = .empty;
        var literal: std.ArrayList(u8) = .empty;
        for (seq) |item| {
            if (item.literal) {
                try literal.appendSlice(a, item.s);
            } else {
                if (literal.items.len != 0) {
                    try results.append(a, try toRuleText(.{ .s = literal.items, .literal = true }, a));
                    literal = .empty;
                }
                try results.append(a, item.s);
            }
        }
        if (literal.items.len != 0) try results.append(a, try toRuleText(.{ .s = literal.items, .literal = true }, a));
        return .{ .s = try stringJoin(a, results.items, " "), .literal = false };
    }

    /// the `transform` lambda (json-schema-to-grammar.cpp:447 @cd959bae9)
    fn transform(self: *PatternState) PatternError!LitOrRule {
        const a = self.conv.a;
        const sub = self.sub;
        const length = sub.len;
        var seq: std.ArrayList(LitOrRule) = .empty;

        while (self.i < length) {
            const c = sub[self.i];
            if (c == '.') {
                try seq.append(a, .{ .s = try self.getDot(), .literal = false });
                self.i += 1;
            } else if (c == '(') {
                self.i += 1;
                if (self.i < length and sub[self.i] == '?') {
                    if (self.i + 1 < length and sub[self.i + 1] == ':') {
                        self.i += 2; // a non-capturing group is a group
                    } else {
                        // lookaround, named group, inline flags, ...
                        return self.unsupported(&.{"unsupported group syntax"});
                    }
                }
                self.paren_depth += 1;
                if (self.paren_depth > schema.recursive_limit) return self.unsupported(&.{"pattern nesting too deep"});
                const inner = try toRuleText(try self.transform(), a);
                try seq.append(a, .{ .s = try std.mem.concat(a, u8, &.{ "(", inner, ")" }), .literal = false });
            } else if (c == ')') {
                self.i += 1;
                if (self.paren_depth == 0) return self.invalid("unbalanced parentheses");
                self.paren_depth -= 1;
                return self.joinSeq(seq.items);
            } else if (c == '^' or c == '$') {
                return self.unsupported(&.{"anchor inside the pattern"});
            } else if (c == '[') {
                var square: std.ArrayList(u8) = .empty;
                try square.append(a, c);
                self.i += 1;
                while (self.i < length and sub[self.i] != ']') {
                    if (sub[self.i] == '\\') {
                        const n = gbnfEscapeLength(sub, self.i);
                        if (n == 0) {
                            return self.unsupported(&.{ "unsupported escape in character class: ", sub[self.i..@min(self.i + 2, length)] });
                        }
                        try square.appendSlice(a, sub[self.i..][0..n]);
                        self.i += n;
                    } else {
                        try square.append(a, sub[self.i]);
                        self.i += 1;
                    }
                }
                if (self.i >= length) return self.invalid("unterminated character class");
                try square.append(a, ']');
                self.i += 1;
                try seq.append(a, .{ .s = square.items, .literal = false });
            } else if (c == '|') {
                try seq.append(a, .{ .s = "|", .literal = false });
                self.i += 1;
            } else if (c == '*' or c == '+' or c == '?') {
                if (seq.items.len == 0) return self.invalid("nothing to repeat");
                const last = &seq.items[seq.items.len - 1];
                last.* = .{ .s = try std.mem.concat(a, u8, &.{ try toRuleText(last.*, a), &.{c} }), .literal = false };
                self.i += 1;
            } else if (c == '{') {
                var curly: std.ArrayList(u8) = .empty;
                try curly.append(a, c);
                self.i += 1;
                while (self.i < length and sub[self.i] != '}') {
                    try curly.append(a, sub[self.i]);
                    self.i += 1;
                }
                if (self.i >= length) return self.unsupported(&.{"unterminated curly brackets"});
                try curly.append(a, '}');
                self.i += 1;
                const nums = try stringSplit(a, curly.items[1 .. curly.items.len - 1], ",");
                var min_times: c_int = 0;
                var max_times: c_int = std.math.maxInt(c_int);
                if (nums.len != 1 and nums.len != 2) return self.unsupported(&.{"wrong number of values in curly brackets"});
                if (nums.len == 1) {
                    min_times = stoi(nums[0]) orelse return self.unsupported(&.{"invalid number in curly brackets"});
                    max_times = min_times;
                } else {
                    if (nums[0].len != 0) min_times = stoi(nums[0]) orelse return self.unsupported(&.{"invalid number in curly brackets"});
                    if (nums[1].len != 0) max_times = stoi(nums[1]) orelse return self.unsupported(&.{"invalid number in curly brackets"});
                }
                if (seq.items.len == 0) return self.invalid("nothing to repeat");
                const last = &seq.items[seq.items.len - 1];
                var sub_rule = last.s;
                const sub_is_literal = last.literal;
                if (!sub_is_literal) {
                    const gop = try self.sub_rule_ids.getOrPut(a, sub_rule);
                    if (!gop.found_existing) gop.value_ptr.* = "";
                    if (gop.value_ptr.*.len == 0) {
                        const id_name = try std.fmt.allocPrint(a, "{s}-{d}", .{ self.name, self.sub_rule_ids.count() });
                        const id = try self.conv.addRule(id_name, sub_rule);
                        self.sub_rule_ids.getPtr(sub_rule).?.* = id;
                    }
                    sub_rule = self.sub_rule_ids.get(sub_rule).?;
                }
                const item = if (sub_is_literal) try std.mem.concat(a, u8, &.{ "\"", sub_rule, "\"" }) else sub_rule;
                last.* = .{ .s = try buildRepetition(a, item, min_times, max_times, ""), .literal = false };
            } else {
                var literal: std.ArrayList(u8) = .empty;
                while (self.i < length) {
                    if (sub[self.i] == '\\') {
                        if (self.i == length - 1) return self.invalid("trailing backslash");
                        const next = sub[self.i + 1];
                        if (isEscapedInRegexpsButNotInLiterals(next)) {
                            self.i += 1;
                            try literal.append(a, sub[self.i]);
                            self.i += 1;
                        } else {
                            const n = gbnfEscapeLength(sub, self.i);
                            if (n == 0) return self.unsupported(&.{ "unsupported escape: ", sub[self.i..@min(self.i + 2, length)] });
                            try literal.appendSlice(a, sub[self.i..][0..n]);
                            self.i += n;
                        }
                    } else if (sub[self.i] == '"') {
                        try literal.appendSlice(a, "\\\"");
                        self.i += 1;
                    } else if (!isNonLiteral(sub[self.i]) and
                        (self.i == length - 1 or literal.items.len == 0 or sub[self.i + 1] == '.' or !isNonLiteral(sub[self.i + 1])))
                    {
                        try literal.append(a, sub[self.i]);
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                // nothing was consumed, ex. a stray ']' or '}'
                if (literal.items.len == 0) return self.unsupported(&.{ "unsupported character: ", &.{c} });
                try seq.append(a, .{ .s = literal.items, .literal = true });
            }
        }
        return self.joinSeq(seq.items);
    }
};

/// A string as `what()` returns it: up to its first NUL.
fn cstr(s: []const u8) []const u8 {
    return if (std.mem.indexOfScalar(u8, s, 0)) |n| s[0..n] else s;
}

fn lessBytes(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

// -----------------------------------------------------------------------------
// Entry points

/// Ports `json_schema_to_grammar` (json-schema-to-grammar.cpp:1025 @cd959bae9),
/// the overload taking JSON: every `std::runtime_error` on the way --
/// the schema outside the subset, a JSON error -- becomes a
/// `std::invalid_argument` naming the conversion.
///
/// Parameters:
/// - `gpa`: allocates the result and the working state.
/// - `value`: the schema.
/// - `force_gbnf`: ignored; the reference builds without llguidance.
///
/// Return: the grammar, owned by the caller; `error.InvalidArgument` with
/// `schema.what()` set when conversion fails.
pub fn jsonSchemaToGrammar(gpa: std.mem.Allocator, value: *const Value, force_gbnf: bool) Error![]u8 {
    _ = force_gbnf;
    const wrapped = blk: {
        var doc = schema.fromJsonSchema(gpa, value) catch |e| break :blk e;
        defer doc.deinit();
        return documentToGrammar(gpa, &doc) catch |e| break :blk e;
    };
    return switch (wrapped) {
        error.RuntimeError, error.JsonError => raise(error.InvalidArgument, &.{ "JSON schema conversion failed:\n", schema.what() }),
        else => |e| e,
    };
}

/// Ports `json_schema_to_grammar` (json-schema-to-grammar.cpp:1040 @cd959bae9),
/// the overload taking a parsed document.
///
/// Parameters:
/// - `gpa`: allocates the result and the working state.
/// - `doc`: the schema.
///
/// Return: the grammar, owned by the caller; on failure the error and
/// `schema.what()`, unwrapped.
pub fn documentToGrammar(gpa: std.mem.Allocator, doc: *const schema.Document) Error![]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var conv = try Converter.init(arena.allocator(), false);
    defer conv.deinit();
    _ = try conv.visit(doc.root, "");
    try conv.checkErrors();
    return gpa.dupe(u8, try conv.formatGrammar());
}

/// Ports `common_grammar_builder` (json-schema-to-grammar.h:12 @cd959bae9):
/// what `build_grammar`'s callback is handed.
pub const GrammarBuilder = struct {
    conv: *Converter,

    /// The `add_rule` member: `_add_rule`.
    ///
    /// Return: the rule's name, valid until `buildGrammar` returns.
    pub fn addRule(self: *GrammarBuilder, name: []const u8, rule: []const u8) Error![]const u8 {
        return self.conv.addRule(name, rule);
    }

    /// The `add_schema` member: `add_schema`, with "root" meaning the root.
    ///
    /// Return: the rule's name, valid until `buildGrammar` returns.
    pub fn addSchema(self: *GrammarBuilder, name: []const u8, node: *const Node) Error![]const u8 {
        return self.conv.addSchema(if (std.mem.eql(u8, name, "root")) "" else name, node);
    }
};

/// Ports `build_grammar` (json-schema-to-grammar.cpp:1047 @cd959bae9): a
/// grammar assembled by a callback. Errors the callback returns are not
/// wrapped.
///
/// Parameters:
/// - `gpa`: allocates the result and the working state.
/// - `options`: `dotall` for patterns.
/// - `ctx`: passed to `cb`.
/// - `cb`: adds rules and schemas through the builder.
///
/// Return: the grammar, owned by the caller.
pub fn buildGrammar(
    gpa: std.mem.Allocator,
    options: Options,
    ctx: anytype,
    comptime cb: fn (@TypeOf(ctx), *GrammarBuilder) Error!void,
) Error![]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var conv = try Converter.init(arena.allocator(), options.dotall);
    defer conv.deinit();
    var builder: GrammarBuilder = .{ .conv = &conv };
    try cb(ctx, &builder);
    try conv.checkErrors();
    return gpa.dupe(u8, try conv.formatGrammar());
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

fn expectGrammar(schema_text: []const u8, expected: []const u8) !void {
    const a = std.testing.allocator;
    var v = try json.parse(a, schema_text);
    defer v.deinit(a);
    const g = try jsonSchemaToGrammar(a, &v, true);
    defer a.free(g);
    try std.testing.expectEqualStrings(expected, g);
}

test "integer bounds become digit ranges" {
    try expectGrammar("{\"type\": \"integer\", \"minimum\": 10}",
        \\root ::= ([1] ([0-9]{1,15}) | [2-9] [0-9]{1,15})
        \\space ::= | " " | "\n"{1,2} [ \t]{0,20}
        \\
    );
    try expectGrammar("{\"type\": \"integer\", \"minimum\": -5, \"maximum\": 30}",
        \\root ::= ("-" ([0-5]) | [0-9] | ([1-2] [0-9] | [3] "0"))
        \\space ::= | " " | "\n"{1,2} [ \t]{0,20}
        \\
    );
}

test "an object's optional properties chain through rest rules" {
    try expectGrammar(
        \\{"type": "object", "properties": {"a": {"type": "string"}, "b": {"const": 1}}, "additionalProperties": false}
    ,
        \\a-kv ::= "\"a\"" space ":" space string
        \\a-rest ::= ( "," space b-kv )?
        \\b ::= "1"
        \\b-kv ::= "\"b\"" space ":" space b
        \\char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
        \\root ::= "{" space  (a-kv a-rest | b-kv )? space "}"
        \\space ::= | " " | "\n"{1,2} [ \t]{0,20}
        \\string ::= "\"" char* "\""
        \\
    );
}
