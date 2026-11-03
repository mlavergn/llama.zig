//! The PEG builder: combinators that add parsers to an arena and hand back
//! their ids, the JSON and Python value grammars built from them, and the
//! operators the C++ spells with `+`, `<<` and `|`.
//!
//! # Provenance
//!
//! Ports `common_peg_parser_builder`, the operators of `common_peg_parser`,
//! `parse_char_classes` and its helpers, `rule_name` and `build_peg_parser`
//! from `llama.cpp/common/peg-parser.h` and `llama.cpp/common/peg-parser.cpp`
//! (llama.cpp v0.6.0, `d81235049`). Each declaration names what it ports and
//! the line it began at.
//!
//! # Shape
//!
//! The C++ hands back a `common_peg_parser`, an id beside a reference to the
//! builder, so `a + b` can reach the builder; here every combinator takes
//! and returns a bare `ParserId`, and the operators are methods:
//! `a + b` is `b.plus(a, b)`, `a << "x"` is `b.shlLit(a, "x")`, `"x" | a` is
//! `b.litOr("x", a)`, `a += b` is `b.plusAssign(&a, b)`. A `rule` built by a
//! function takes the function and a context for it.
//!
//! `rule_name`'s `std::regex_replace` runs through libllama's `regex.zig`.

const std = @import("std");
const json_mod = @import("json");
const regex = @import("llama").regex;
const errors = @import("errors.zig");
const arena_mod = @import("arena.zig");

const Allocator = std.mem.Allocator;
const Error = errors.Error;
const Arena = arena_mod.Arena;
const ParserId = arena_mod.ParserId;
const Parser = arena_mod.Parser;
const CharRange = arena_mod.CharRange;
const SharedDoc = arena_mod.SharedDoc;
const schema = json_mod.schema;

/// Ports `parse_hex_escape` (peg-parser.cpp:34 @d81235049).
///
/// Return: the value and how many digits it took, or `{0, 0}`.
fn parseHexEscape(str: []const u8, pos: usize, hex_count: usize) struct { u32, usize } {
    if (pos + hex_count > str.len) return .{ 0, 0 };
    var value: u32 = 0;
    for (0..hex_count) |i| {
        const c = str[pos + i];
        if (!arena_mod.isHexDigit(c)) return .{ 0, 0 };
        value <<= 4;
        if ('a' <= c and c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c and c <= 'F') {
            value += c - 'A' + 10;
        } else {
            value += c - '0';
        }
    }
    return .{ value, hex_count };
}

/// `static_cast<uint32_t>(c)` of a plain `char`: sign-extended where `char`
/// is signed, as on Darwin, Windows and x86 Linux.
fn charToU32(c: u8) u32 {
    const cc: c_char = @bitCast(c);
    return @bitCast(@as(c_int, cc));
}

/// Ports `parse_char_class_char` (peg-parser.cpp:59 @d81235049).
///
/// Return: the code point and how many bytes it took.
fn parseCharClassChar(content: []const u8, pos: usize) struct { u32, usize } {
    if (content[pos] == '\\' and pos + 1 < content.len) {
        switch (content[pos + 1]) {
            'x', 'u', 'U' => |e| {
                const n: usize = switch (e) {
                    'x' => 2,
                    'u' => 4,
                    else => 8,
                };
                const r = parseHexEscape(content, pos + 2, n);
                if (r[1] > 0) return .{ r[0], 2 + r[1] };
                return .{ e, 2 };
            },
            'n' => return .{ '\n', 2 },
            't' => return .{ '\t', 2 },
            'r' => return .{ '\r', 2 },
            '\\' => return .{ '\\', 2 },
            ']' => return .{ ']', 2 },
            '[' => return .{ '[', 2 },
            else => |c| return .{ charToU32(c), 2 },
        }
    }
    return .{ content[pos], 1 };
}

/// Ports `parse_char_classes` (peg-parser.cpp:100 @d81235049): `[a-z\n]`,
/// `^` negating, with or without the brackets.
///
/// An empty class -- `""` or `"["` -- is undefined in the C++ (`front()`
/// and `back()` of an empty string); here it is no ranges.
///
/// Return: the ranges, allocated with `a`, and whether the class is
/// negated.
fn parseCharClasses(a: Allocator, classes: []const u8) Error!struct { []CharRange, bool } {
    var ranges: std.ArrayList(CharRange) = .empty;
    var negated = false;
    var content = classes;
    if (content.len > 0 and content[0] == '[') content = content[1..];
    if (content.len > 0 and content[content.len - 1] == ']') content = content[0 .. content.len - 1];
    if (content.len > 0 and content[0] == '^') {
        negated = true;
        content = content[1..];
    }
    var i: usize = 0;
    while (i < content.len) {
        const start = parseCharClassChar(content, i);
        i += start[1];
        if (i + 1 < content.len and content[i] == '-') {
            const end = parseCharClassChar(content, i + 1);
            try ranges.append(a, .{ .start = start[0], .end = end[0] });
            i += 1 + end[1];
        } else {
            try ranges.append(a, .{ .start = start[0], .end = start[0] });
        }
    }
    return .{ ranges.items, negated };
}

threadlocal var invalid_rule_chars_re: ?regex.Regex = null;

/// Ports `rule_name` (peg-parser.cpp:1072 @d81235049): every run of
/// characters outside `[a-zA-Z0-9-]` becomes one `-`.
///
/// Return: allocated with `a`.
pub fn ruleName(a: Allocator, name: []const u8) Error![]u8 {
    if (invalid_rule_chars_re == null) {
        invalid_rule_chars_re = regex.Regex.compile("[^a-zA-Z0-9-]+") catch |e| regex.throwRegexError(e);
    }
    const re = &invalid_rule_chars_re.?;
    var it = regex.Regex.Iterator.init(re, name, 0, name.len) catch |e| regex.throwRegexError(e);
    defer it.deinit();
    var out: std.ArrayList(u8) = .empty;
    var last: usize = 0;
    while (it.valid()) {
        const m = it.m.subs.items[0];
        try out.appendSlice(a, name[last..m.first]);
        try out.append(a, '-');
        last = m.second;
        it.next() catch |e| regex.throwRegexError(e);
    }
    try out.appendSlice(a, name[last..]);
    return out.items;
}

/// Ports `common_peg_parser_builder` (peg-parser.h:370 @d81235049).
pub const Builder = struct {
    arena: Arena,

    /// Ports `common_peg_parser_builder` (peg-parser.cpp:1077 @d81235049),
    /// the constructor.
    ///
    /// Return: a builder over an empty arena; `deinit` frees it unless
    /// `build` took it.
    pub fn init(gpa: Allocator) Error!Builder {
        return .{ .arena = try Arena.init(gpa) };
    }

    /// Frees the arena, unless `build` handed it out.
    pub fn deinit(self: *Builder) void {
        self.arena.deinit();
    }

    fn mem(self: *Builder) Allocator {
        return self.arena.alloc();
    }

    fn add(self: *Builder, p: Parser) Error!ParserId {
        return self.arena.addParser(p);
    }

    fn dupe(self: *Builder, s: []const u8) Error![]u8 {
        return self.mem().dupe(u8, s);
    }

    fn dupeAll(self: *Builder, list: []const []const u8) Error![]const []const u8 {
        const out = try self.mem().alloc([]const u8, list.len);
        for (list, 0..) |s, i| out[i] = try self.dupe(s);
        return out;
    }

    /// Ports `eps` (peg-parser.h:381 @d81235049).
    pub fn eps(self: *Builder) Error!ParserId {
        return self.add(.epsilon);
    }

    /// Ports `start` (peg-parser.h:385 @d81235049).
    pub fn start(self: *Builder) Error!ParserId {
        return self.add(.start);
    }

    /// Ports `end` (peg-parser.h:389 @d81235049).
    pub fn end(self: *Builder) Error!ParserId {
        return self.add(.end);
    }

    /// Ports `literal` (peg-parser.h:393 @d81235049).
    pub fn literal(self: *Builder, lit: []const u8) Error!ParserId {
        return self.add(.{ .literal = try self.dupe(lit) });
    }

    /// Ports `sequence` (peg-parser.h:397 @d81235049), the overload taking
    /// nothing: an empty sequence.
    pub fn sequenceEmpty(self: *Builder) Error!ParserId {
        return self.add(.{ .sequence = &.{} });
    }

    /// Ports `sequence` (peg-parser.cpp:1079 @d81235049): nested sequences
    /// are flattened into this one.
    pub fn sequence(self: *Builder, parsers: []const ParserId) Error!ParserId {
        return self.add(.{ .sequence = try self.flatten(parsers, .sequence) });
    }

    /// Ports `choice` (peg-parser.h:404 @d81235049), the overload taking
    /// nothing: an empty choice.
    pub fn choiceEmpty(self: *Builder) Error!ParserId {
        return self.add(.{ .choice = &.{} });
    }

    /// Ports `choice` (peg-parser.cpp:1111 @d81235049): nested choices are
    /// flattened into this one.
    pub fn choice(self: *Builder, parsers: []const ParserId) Error!ParserId {
        return self.add(.{ .choice = try self.flatten(parsers, .choice) });
    }

    fn flatten(self: *Builder, parsers: []const ParserId, comptime kind: std.meta.Tag(Parser)) Error![]ParserId {
        var out: std.ArrayList(ParserId) = .empty;
        for (parsers) |p| {
            const parser = try self.arena.get(p);
            if (parser.* == kind) {
                try out.appendSlice(self.mem(), @field(parser.*, @tagName(kind)));
            } else {
                try out.append(self.mem(), p);
            }
        }
        return out.items;
    }

    /// Ports `one_or_more` (peg-parser.h:411 @d81235049).
    pub fn oneOrMore(self: *Builder, p: ParserId) Error!ParserId {
        return self.repeat(p, 1, -1);
    }

    /// Ports `zero_or_more` (peg-parser.h:415 @d81235049).
    pub fn zeroOrMore(self: *Builder, p: ParserId) Error!ParserId {
        return self.repeat(p, 0, -1);
    }

    /// Ports `optional` (peg-parser.h:419 @d81235049).
    pub fn optional(self: *Builder, p: ParserId) Error!ParserId {
        return self.repeat(p, 0, 1);
    }

    /// Ports `peek` (peg-parser.h:423 @d81235049).
    pub fn peek(self: *Builder, p: ParserId) Error!ParserId {
        return self.add(.{ .and_ = p });
    }

    /// Ports `negate` (peg-parser.h:427 @d81235049).
    pub fn negate(self: *Builder, p: ParserId) Error!ParserId {
        return self.add(.{ .not_ = p });
    }

    /// Ports `any` (peg-parser.h:431 @d81235049).
    pub fn any(self: *Builder) Error!ParserId {
        return self.add(.any);
    }

    /// Ports `chars` (peg-parser.cpp:1143 @d81235049). A max of -1 is
    /// unbounded.
    pub fn chars(self: *Builder, classes: []const u8, min: c_int, max: c_int) Error!ParserId {
        const parsed = try parseCharClasses(self.mem(), classes);
        return self.add(.{ .chars = .{
            .pattern = try self.dupe(classes),
            .ranges = parsed[0],
            .negated = parsed[1],
            .min_count = min,
            .max_count = max,
        } });
    }

    /// Ports `ref` (peg-parser.h:442 @d81235049).
    pub fn ref(self: *Builder, name: []const u8) Error!ParserId {
        return self.add(.{ .ref = try self.dupe(name) });
    }

    /// Ports `space` (peg-parser.h:446 @d81235049).
    pub fn space(self: *Builder) Error!ParserId {
        return self.add(.space);
    }

    /// Ports `until` (peg-parser.h:451 @d81235049).
    pub fn until(self: *Builder, delimiter: []const u8) Error!ParserId {
        return self.untilOneOf(&.{delimiter});
    }

    /// Ports `until_one_of` (peg-parser.h:455 @d81235049).
    pub fn untilOneOf(self: *Builder, delimiters: []const []const u8) Error!ParserId {
        return self.add(.{ .until = try self.dupeAll(delimiters) });
    }

    /// Ports `rest` (peg-parser.h:459 @d81235049).
    pub fn rest(self: *Builder) Error!ParserId {
        return self.untilOneOf(&.{});
    }

    /// Ports `repeat` (peg-parser.h:464 @d81235049), the overload taking a
    /// range.
    pub fn repeat(self: *Builder, p: ParserId, min: c_int, max: c_int) Error!ParserId {
        return self.add(.{ .repetition = .{ .child = p, .min_count = min, .max_count = max } });
    }

    /// Ports `repeat` (peg-parser.h:468 @d81235049), the overload taking a
    /// count.
    pub fn repeatN(self: *Builder, p: ParserId, n: c_int) Error!ParserId {
        return self.repeat(p, n, n);
    }

    /// Ports `string_content` (peg-parser.cpp:1196 @d81235049).
    pub fn stringContent(self: *Builder, delimiter: u8) Error!ParserId {
        return self.add(.{ .string = delimiter });
    }

    /// Ports `double_quoted_string` (peg-parser.cpp:1200 @d81235049).
    pub fn doubleQuotedString(self: *Builder) Error!ParserId {
        return self.ruleFn("double-quoted-string", self, quotedBody('"'), false);
    }

    /// Ports `single_quoted_string` (peg-parser.cpp:1206 @d81235049).
    pub fn singleQuotedString(self: *Builder) Error!ParserId {
        return self.ruleFn("single-quoted-string", self, quotedBody('\''), false);
    }

    fn quotedBody(comptime q: u8) fn (*Builder) Error!ParserId {
        return struct {
            fn body(b: *Builder) Error!ParserId {
                return b.sequence(&.{ try b.literal(&.{q}), try b.stringContent(q), try b.literal(&.{q}) });
            }
        }.body;
    }

    /// Ports `quoted_string` (peg-parser.cpp:1212 @d81235049).
    pub fn quotedString(self: *Builder) Error!ParserId {
        return self.ruleFn("quoted-string", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.choice(&.{ try b.doubleQuotedString(), try b.singleQuotedString() });
            }
        }.body, false);
    }

    /// Ports `json_number` (peg-parser.cpp:1220 @d81235049): committed only
    /// when the next character cannot extend it.
    pub fn jsonNumber(self: *Builder) Error!ParserId {
        return self.ruleFn("json-number", self, struct {
            fn body(b: *Builder) Error!ParserId {
                const digit1_9 = try b.chars("[1-9]", 1, 1);
                const digits = try b.chars("[0-9]", 1, -1);
                const int_part = try b.choice(&.{ try b.literal("0"), try b.sequence(&.{ digit1_9, try b.chars("[0-9]", 0, -1) }) });
                const frac = try b.sequence(&.{ try b.literal("."), digits });
                const exp = try b.sequence(&.{ try b.choice(&.{ try b.literal("e"), try b.literal("E") }), try b.optional(try b.chars("[+-]", 1, 1)), digits });
                const not_number_continuation = try b.negate(try b.chars("[0-9.eE+-]", 1, 1));
                return b.sequence(&.{ try b.optional(try b.literal("-")), int_part, try b.optional(frac), try b.optional(exp), not_number_continuation });
            }
        }.body, false);
    }

    /// Ports `json_string` (peg-parser.cpp:1235 @d81235049).
    pub fn jsonString(self: *Builder) Error!ParserId {
        return self.ruleFn("json-string", self, quotedBody('"'), false);
    }

    /// Ports `json_bool` (peg-parser.cpp:1241 @d81235049).
    pub fn jsonBool(self: *Builder) Error!ParserId {
        return self.ruleFn("json-bool", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.choice(&.{ try b.literal("true"), try b.literal("false") });
            }
        }.body, false);
    }

    /// Ports `json_null` (peg-parser.cpp:1247 @d81235049).
    pub fn jsonNull(self: *Builder) Error!ParserId {
        return self.ruleFn("json-null", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.literal("null");
            }
        }.body, false);
    }

    /// Ports `json_object` (peg-parser.cpp:1253 @d81235049).
    pub fn jsonObject(self: *Builder) Error!ParserId {
        return self.ruleFn("json-object", self, struct {
            fn body(b: *Builder) Error!ParserId {
                const ws = try b.space();
                const member = try b.sequence(&.{ try b.jsonString(), ws, try b.literal(":"), ws, try b.json() });
                const members = try b.sequence(&.{ member, try b.zeroOrMore(try b.sequence(&.{ ws, try b.literal(","), ws, member })) });
                return b.sequence(&.{
                    try b.literal("{"),
                    ws,
                    try b.choice(&.{ try b.literal("}"), try b.sequence(&.{ members, ws, try b.literal("}") }) }),
                });
            }
        }.body, false);
    }

    /// Ports `json_array` (peg-parser.cpp:1269 @d81235049).
    pub fn jsonArray(self: *Builder) Error!ParserId {
        return self.ruleFn("json-array", self, struct {
            fn body(b: *Builder) Error!ParserId {
                const ws = try b.space();
                const elements = try b.sequence(&.{ try b.json(), try b.zeroOrMore(try b.sequence(&.{ ws, try b.literal(","), ws, try b.json() })) });
                return b.sequence(&.{
                    try b.literal("["),
                    ws,
                    try b.choice(&.{ try b.literal("]"), try b.sequence(&.{ elements, ws, try b.literal("]") }) }),
                });
            }
        }.body, false);
    }

    /// Ports `json` (peg-parser.cpp:1284 @d81235049): any JSON value.
    pub fn json(self: *Builder) Error!ParserId {
        return self.ruleFn("json-value", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.choice(&.{
                    try b.jsonObject(),
                    try b.jsonArray(),
                    try b.jsonString(),
                    try b.jsonNumber(),
                    try b.jsonBool(),
                    try b.jsonNull(),
                });
            }
        }.body, false);
    }

    /// Ports `python_string` (peg-parser.cpp:1297 @d81235049).
    pub fn pythonString(self: *Builder) Error!ParserId {
        return self.ruleFn("python-string", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.choice(&.{ try b.doubleQuotedString(), try b.singleQuotedString() });
            }
        }.body, false);
    }

    /// Ports `python_number` (peg-parser.cpp:1303 @d81235049).
    pub fn pythonNumber(self: *Builder) Error!ParserId {
        return self.jsonNumber();
    }

    /// Ports `python_bool` (peg-parser.cpp:1307 @d81235049).
    pub fn pythonBool(self: *Builder) Error!ParserId {
        return self.ruleFn("python-bool", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.choice(&.{ try b.literal("True"), try b.literal("False") });
            }
        }.body, false);
    }

    /// Ports `python_null` (peg-parser.cpp:1313 @d81235049).
    pub fn pythonNull(self: *Builder) Error!ParserId {
        return self.ruleFn("python-none", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.literal("None");
            }
        }.body, false);
    }

    /// Ports `python_dict` (peg-parser.cpp:1319 @d81235049).
    pub fn pythonDict(self: *Builder) Error!ParserId {
        return self.ruleFn("python-dict", self, struct {
            fn body(b: *Builder) Error!ParserId {
                const ws = try b.space();
                const member = try b.sequence(&.{ try b.pythonString(), ws, try b.literal(":"), ws, try b.pythonValue() });
                const members = try b.sequence(&.{ member, try b.zeroOrMore(try b.sequence(&.{ ws, try b.literal(","), ws, member })) });
                return b.sequence(&.{
                    try b.literal("{"),
                    ws,
                    try b.choice(&.{ try b.literal("}"), try b.sequence(&.{ members, ws, try b.literal("}") }) }),
                    ws,
                });
            }
        }.body, false);
    }

    /// Ports `python_array` (peg-parser.cpp:1336 @d81235049).
    pub fn pythonArray(self: *Builder) Error!ParserId {
        return self.ruleFn("python-array", self, struct {
            fn body(b: *Builder) Error!ParserId {
                const ws = try b.space();
                const elements = try b.sequence(&.{ try b.pythonValue(), try b.zeroOrMore(try b.sequence(&.{ try b.literal(","), ws, try b.pythonValue() })) });
                return b.sequence(&.{
                    try b.literal("["),
                    ws,
                    try b.choice(&.{ try b.literal("]"), try b.sequence(&.{ elements, ws, try b.literal("]") }) }),
                    ws,
                });
            }
        }.body, false);
    }

    /// Ports `python_value` (peg-parser.cpp:1352 @d81235049).
    pub fn pythonValue(self: *Builder) Error!ParserId {
        return self.ruleFn("python-value", self, struct {
            fn body(b: *Builder) Error!ParserId {
                return b.choice(&.{
                    try b.pythonDict(),
                    try b.pythonArray(),
                    try b.pythonString(),
                    try b.pythonNumber(),
                    try b.pythonBool(),
                    try b.pythonNull(),
                });
            }
        }.body, false);
    }

    /// Ports `marker` (peg-parser.cpp:1365 @d81235049): `<...>` or `[...]`.
    pub fn marker(self: *Builder) Error!ParserId {
        const sharp = try self.plus(try self.plus(try self.literal("<"), try self.until(">")), try self.literal(">"));
        const square = try self.plus(try self.plus(try self.literal("["), try self.until("]")), try self.literal("]"));
        return self.choice(&.{ sharp, square });
    }

    /// Ports `json_member` (peg-parser.cpp:1371 @d81235049).
    pub fn jsonMember(self: *Builder, key: []const u8, p: ParserId) Error!ParserId {
        const ws = try self.space();
        const quoted = try std.mem.concat(self.mem(), u8, &.{ "\"", key, "\"" });
        return self.sequence(&.{ try self.literal(quoted), ws, try self.literal(":"), ws, p });
    }

    /// Ports `schema` (peg-parser.cpp:1148 @d81235049), the overload taking
    /// a document: `node` lives in `doc`, which the arena now also holds.
    pub fn schemaDoc(self: *Builder, p: ParserId, name: []const u8, doc: *SharedDoc, node: *const schema.Node, raw: bool) Error!ParserId {
        const n = try self.dupe(name);
        const id = try self.add(.{ .schema = .{ .child = p, .name = n, .doc = doc, .node = node, .raw = raw } });
        _ = doc.retain();
        return id;
    }

    /// Ports `schema` (peg-parser.cpp:1152 @d81235049), the overload taking
    /// JSON: parsed into a document of its own.
    pub fn schemaJson(self: *Builder, p: ParserId, name: []const u8, value: *const json_mod.Value, raw: bool) Error!ParserId {
        const doc = try SharedDoc.fromJson(self.arena.gpa, value);
        defer doc.release();
        return self.schemaDoc(p, name, doc, doc.doc.root, raw);
    }

    /// Ports `rule` (peg-parser.cpp:1157 @d81235049), the overload taking a
    /// parser: the rule, stored under its cleaned name.
    ///
    /// Return: a reference to it.
    pub fn rule(self: *Builder, name: []const u8, p: ParserId, trigger: bool) Error!ParserId {
        const clean_name = try ruleName(self.mem(), name);
        const rule_id = try self.add(.{ .rule = .{ .name = clean_name, .child = p, .trigger = trigger } });
        try self.arena.addRule(clean_name, rule_id);
        return self.ref(clean_name);
    }

    /// Ports `rule` (peg-parser.cpp:1164 @d81235049), the overload taking a
    /// builder function: an existing rule of that name is reused; otherwise
    /// a placeholder is registered so `f` can refer to the rule, then the
    /// rule `f` builds replaces it.
    ///
    /// Parameters:
    /// - `ctx`, `f`: builds the rule's parser.
    ///
    /// Return: a reference to the rule.
    pub fn ruleFn(self: *Builder, name: []const u8, ctx: anytype, comptime f: fn (@TypeOf(ctx)) Error!ParserId, trigger: bool) Error!ParserId {
        const clean_name = try ruleName(self.mem(), name);
        if (self.arena.hasRule(clean_name)) return self.ref(clean_name);

        const placeholder = try self.any();
        const placeholder_rule_id = try self.add(.{ .rule = .{ .name = clean_name, .child = placeholder, .trigger = trigger } });
        try self.arena.addRule(clean_name, placeholder_rule_id);

        const parser = try f(ctx);

        const rule_id = try self.add(.{ .rule = .{ .name = clean_name, .child = parser, .trigger = trigger } });
        try self.arena.addRule(clean_name, rule_id);
        return self.ref(clean_name);
    }

    /// Ports `trigger_rule` (peg-parser.h:528 @d81235049), the overload
    /// taking a parser.
    pub fn triggerRule(self: *Builder, name: []const u8, p: ParserId) Error!ParserId {
        return self.rule(name, p, true);
    }

    /// Ports `trigger_rule` (peg-parser.h:529 @d81235049), the overload
    /// taking a builder function.
    pub fn triggerRuleFn(self: *Builder, name: []const u8, ctx: anytype, comptime f: fn (@TypeOf(ctx)) Error!ParserId) Error!ParserId {
        return self.ruleFn(name, ctx, f, true);
    }

    /// Ports `atomic` (peg-parser.h:534 @d81235049).
    pub fn atomic(self: *Builder, p: ParserId) Error!ParserId {
        return self.add(.{ .atomic = p });
    }

    /// Ports `tag` (peg-parser.h:538 @d81235049).
    pub fn tag(self: *Builder, t: []const u8, p: ParserId) Error!ParserId {
        return self.add(.{ .tag = .{ .child = p, .tag = try self.dupe(t) } });
    }

    /// Ports `gbnf` (peg-parser.h:542 @d81235049).
    pub fn gbnf(self: *Builder, p: ParserId, grammar: []const u8) Error!ParserId {
        return self.add(.{ .gbnf = .{ .child = p, .grammar = try self.dupe(grammar) } });
    }

    /// Ports `ac` (peg-parser.cpp:1382 @d81235049), the overload taking a
    /// list.
    pub fn ac(self: *Builder, p: ParserId, delimiters: []const []const u8) Error!ParserId {
        if (delimiters.len == 0) return errors.raise(error.RuntimeError, &.{"ac parser requires at least one delimiter"});
        return self.add(.{ .ac = .{ .child = p, .delimiters = try self.dupeAll(delimiters) } });
    }

    /// Ports `ac` (peg-parser.h:549 @d81235049), the overload taking one
    /// delimiter.
    pub fn acOne(self: *Builder, p: ParserId, delimiter: []const u8) Error!ParserId {
        return self.ac(p, &.{delimiter});
    }

    /// Ports `set_root` (peg-parser.cpp:1185 @d81235049).
    pub fn setRoot(self: *Builder, p: ParserId) void {
        self.arena.root = p;
    }

    /// Ports `build` (peg-parser.cpp:1189 @d81235049): references resolved,
    /// the arena moved out. The builder is left empty, and is not used
    /// again; a failure leaves the arena in the builder.
    ///
    /// Return: the arena, owned by the caller.
    pub fn build(self: *Builder) Error!Arena {
        try self.arena.resolveRefs();
        const out = self.arena;
        self.arena = try Arena.init(out.gpa);
        return out;
    }

    // -------------------------------------------------------------------------
    // the operators of `common_peg_parser`

    /// Ports `operator`+ (peg-parser.cpp:1012 @d81235049): `a + b`.
    pub fn plus(self: *Builder, a: ParserId, b: ParserId) Error!ParserId {
        return self.sequence(&.{ a, b });
    }

    /// Ports `operator`| (peg-parser.cpp:1016 @d81235049): `a | b`.
    pub fn orr(self: *Builder, a: ParserId, b: ParserId) Error!ParserId {
        return self.choice(&.{ a, b });
    }

    /// Ports `operator`<< (peg-parser.cpp:1020 @d81235049): `a << b`, a
    /// sequence with optional space between.
    pub fn shl(self: *Builder, a: ParserId, b: ParserId) Error!ParserId {
        return self.sequence(&.{ a, try self.space(), b });
    }

    /// Ports `operator`+ (peg-parser.cpp:1024 @d81235049): `a + "str"`.
    pub fn plusLit(self: *Builder, a: ParserId, s: []const u8) Error!ParserId {
        return self.plus(a, try self.literal(s));
    }

    /// Ports `operator`<< (peg-parser.cpp:1032 @d81235049): `a << "str"`.
    pub fn shlLit(self: *Builder, a: ParserId, s: []const u8) Error!ParserId {
        return self.shl(a, try self.literal(s));
    }

    /// Ports `operator`| (peg-parser.cpp:1040 @d81235049): `a | "str"`.
    pub fn orLit(self: *Builder, a: ParserId, s: []const u8) Error!ParserId {
        return self.orr(a, try self.literal(s));
    }

    /// Ports `operator`+ (peg-parser.cpp:1048 @d81235049): `"str" + a`.
    pub fn litPlus(self: *Builder, s: []const u8, a: ParserId) Error!ParserId {
        return self.plus(try self.literal(cstr(s)), a);
    }

    /// Ports `operator`<< (peg-parser.cpp:1056 @d81235049): `"str" << a`.
    pub fn litShl(self: *Builder, s: []const u8, a: ParserId) Error!ParserId {
        return self.shl(try self.literal(cstr(s)), a);
    }

    /// Ports `operator`| (peg-parser.cpp:1064 @d81235049): `"str" | a`.
    pub fn litOr(self: *Builder, s: []const u8, a: ParserId) Error!ParserId {
        return self.orr(try self.literal(cstr(s)), a);
    }

    /// Ports `operator`+= (peg-parser.cpp:1002 @d81235049): `a += b`.
    pub fn plusAssign(self: *Builder, a: *ParserId, b: ParserId) Error!void {
        a.* = try self.sequence(&.{ a.*, b });
    }

    /// Ports `operator`|= (peg-parser.cpp:1007 @d81235049): `a |= b`.
    pub fn orAssign(self: *Builder, a: *ParserId, b: ParserId) Error!void {
        a.* = try self.choice(&.{ a.*, b });
    }
};

/// The `std::string` overloads of the left-hand operators go through
/// `str.c_str()`: the string ends at its first NUL.
fn cstr(s: []const u8) []const u8 {
    return if (std.mem.indexOfScalar(u8, s, 0)) |n| s[0..n] else s;
}

/// Ports `build_peg_parser` (peg-parser.cpp:2120 @d81235049): a builder,
/// its root from `f`, built.
///
/// Return: the arena, owned by the caller.
pub fn buildPegParser(gpa: Allocator, ctx: anytype, comptime f: fn (@TypeOf(ctx), *Builder) Error!ParserId) Error!Arena {
    var b = try Builder.init(gpa);
    defer b.deinit();
    b.setRoot(try f(ctx, &b));
    return b.build();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "char classes, escapes and a signed char" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const r = try parseCharClasses(arena.allocator(), "[^a-z\\n\\x41\\u00e9-\\u00ff\\q]");
    try std.testing.expect(r[1]);
    try std.testing.expectEqual(@as(usize, 5), r[0].len);
    try std.testing.expectEqual(CharRange{ .start = 0xe9, .end = 0xff }, r[0][3]);
    const s = try parseCharClasses(arena.allocator(), "\\\xc3");
    try std.testing.expectEqual(charToU32(0xc3), s[0][0].start);
    try std.testing.expectEqualStrings("a-b-", try ruleName(arena.allocator(), "a b!?"));
}

test "a JSON parser built and run" {
    const a = std.testing.allocator;
    var arena = try buildPegParser(a, {}, struct {
        fn f(_: void, b: *Builder) Error!ParserId {
            return b.plus(try b.litPlus("<tool_call>", try b.json()), try b.literal("</tool_call>"));
        }
    }.f);
    defer arena.deinit();
    var ctx = try arena_mod.ParseContext.init(a, "<tool_call>{\"a\": [1, 2.5e3, true]}</tool_call>", arena_mod.Flags.none);
    defer ctx.deinit();
    const r = try arena.parse(&ctx, 0);
    try std.testing.expect(r.success());
    try std.testing.expectEqual(ctx.input.len, r.end);
}
