//! The chat side of the PEG parser: builders for reasoning, content and
//! tool-call grammars, and the mappers that turn a parse's AST into a
//! `common_chat_msg`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/chat-peg-parser.h` and
//! `llama.cpp/common/chat-peg-parser.cpp` (llama.cpp v0.9.0, `cd959bae9`):
//! `common_chat_peg_builder`, `common_chat_peg_mapper` and its gemma4 and
//! minimax-m3 variants, `tag_based_peg_mapper`, `tagged_peg_parser`, and the
//! string helpers they share. Each declaration names what it ports and the
//! line it began at.
//!
//! # Shape
//!
//! `common_chat_peg_builder` derives from `common_peg_parser_builder`; here
//! `ChatBuilder` holds a `peg.Builder` as `b`, and the base combinators are
//! called through it. The three mappers are one `Mapper` and a `Kind`, since
//! the C++ overrides only `from_ast`.
//!
//! The order the C++ creates parsers in is part of the result -- a parser's
//! id is its position -- and these builders evaluate every compound
//! expression left to right, as clang does the C++ operators.
//!
//! # After a `tool-close`
//!
//! Until v0.6.0 the base mapper's `current_tool` still pointed at the
//! `std::optional` a `tool-close` had just reset, and a tool-related node
//! before the next `tool-open` wrote a destroyed `common_chat_tool_call`;
//! upstream's `map` (chat-peg-parser.cpp:315 @cd959bae9) now clears it, so
//! such a node is ignored. `Mapper.after_close` records that one arrived, for
//! `make peg-diff`, which must see some.

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const libcxx = @import("llama").libcxx;
const msg_mod = @import("msg.zig");
const exception = @import("exception.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Msg = msg_mod.Msg;
const ToolCall = msg_mod.ToolCall;
const ParserId = peg.ParserId;
const AstId = peg.AstId;
const AstNode = peg.AstNode;
const AstArena = peg.AstArena;
const ParseResult = peg.ParseResult;
const invalid_ast_id = peg.invalid_ast_id;

/// Every fallible call here: `peg.Error`, the text in `peg.what(e)`.
pub const Error = peg.Error;

fn fromJson(e: json.Error) Error {
    return peg.errors.fromJson(e);
}

/// An error from `msg.zig` as one of ours, its text carried over.
fn fromChat(e: exception.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => error.JsonError,
        error.RuntimeError, error.InvalidArgument => |x| peg.errors.raise(x, &.{exception.what(x)}),
    };
}

// -----------------------------------------------------------------------------
// String helpers

fn isCSpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r');
}

fn isCAlnum(c: u8) bool {
    return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
}

/// Ports `trim_trailing_space` (chat-peg-parser.cpp:12 @cd959bae9): at most
/// `max` characters, -1 for all.
fn trimTrailingSpace(sv_in: []const u8, max: c_int) []const u8 {
    var sv = sv_in;
    var count: c_int = 0;
    while (sv.len > 0 and isCSpace(sv[sv.len - 1])) {
        if (max != -1 and count >= max) break;
        sv = sv[0 .. sv.len - 1];
        count += 1;
    }
    return sv;
}

/// Ports `trim_leading_space` (chat-peg-parser.cpp:24 @cd959bae9).
fn trimLeadingSpace(sv_in: []const u8, max: c_int) []const u8 {
    var sv = sv_in;
    var count: c_int = 0;
    while (sv.len > 0 and isCSpace(sv[0])) {
        if (max != -1 and count >= max) break;
        sv = sv[1..];
        count += 1;
    }
    return sv;
}

/// Ports `trim` (chat-peg-parser.cpp:36 @cd959bae9): one leading space,
/// every trailing one.
fn trim(sv: []const u8) []const u8 {
    return trimTrailingSpace(trimLeadingSpace(sv, 1), -1);
}

/// Ports `json_brace_depth` (chat-peg-parser.cpp:42 @cd959bae9): unclosed
/// `{` outside strings.
fn jsonBraceDepth(s: []const u8) c_int {
    var depth: c_int = 0;
    var in_string = false;
    var escaped = false;
    for (s) |c| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' and in_string) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string) {
            if (c == '{') {
                depth += 1;
            } else if (c == '}') {
                depth -= 1;
            }
        }
    }
    return depth;
}

/// `common_json(s).dump()`: `s` as a JSON string, quoted.
///
/// Return: owned by the caller; `JsonError` when `s` is not UTF-8.
fn dumpString(gpa: Allocator, s: []const u8) Error![]u8 {
    var v = Value.initString(gpa, s) catch |e| return fromJson(e);
    defer v.deinit(gpa);
    return json.dump(gpa, &v, -1) catch |e| fromJson(e);
}

/// Ports `escape_json_string_inner` (chat-peg-parser.cpp:71 @cd959bae9):
/// `s` JSON-escaped, without the quotes.
///
/// Return: owned by the caller.
fn escapeJsonStringInner(gpa: Allocator, s: []const u8) Error![]u8 {
    const escaped = try dumpString(gpa, s);
    if (escaped.len >= 2 and escaped[0] == '"' and escaped[escaped.len - 1] == '"') {
        defer gpa.free(escaped);
        return gpa.dupe(u8, escaped[1 .. escaped.len - 1]);
    }
    return escaped;
}

/// Ports `normalize_quotes_to_json` (chat-peg-parser.cpp:84 @cd959bae9):
/// Python's single-quoted strings as JSON's double-quoted ones, and
/// `True`, `False` and `None` -- or a prefix of one, as a stream ends --
/// as JSON's literals.
///
/// Return: owned by the caller.
pub fn normalizeQuotesToJson(gpa: Allocator, input: []const u8) Error![]u8 {
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(gpa);
    try result.ensureTotalCapacity(gpa, input.len + 16);

    var in_single_quoted = false;
    var in_double_quoted = false;
    const isWordChar = struct {
        fn f(ch: u8) bool {
            return isCAlnum(ch) or ch == '_';
        }
    }.f;
    const literals = [_][2][]const u8{ .{ "True", "true" }, .{ "False", "false" }, .{ "None", "null" } };

    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        const c = input[i];
        if (c == '\\' and i + 1 < input.len) {
            const next = input[i + 1];
            if (in_single_quoted) {
                if (next == '\'') {
                    try result.append(gpa, '\'');
                    i += 1;
                    continue;
                }
                if (next == '"') {
                    try result.appendSlice(gpa, "\\\"");
                    i += 1;
                    continue;
                }
                try result.appendSlice(gpa, &.{ c, next });
                i += 1;
                continue;
            }
            if (in_double_quoted) {
                try result.appendSlice(gpa, &.{ c, next });
                i += 1;
                continue;
            }
            try result.append(gpa, c);
            continue;
        }

        if (c == '"') {
            if (in_single_quoted) {
                try result.appendSlice(gpa, "\\\"");
            } else {
                in_double_quoted = !in_double_quoted;
                try result.append(gpa, c);
            }
        } else if (c == '\'') {
            if (in_double_quoted) {
                try result.append(gpa, c);
            } else if (in_single_quoted) {
                in_single_quoted = false;
                try result.append(gpa, '"');
            } else {
                in_single_quoted = true;
                try result.append(gpa, '"');
            }
        } else if (!in_single_quoted and !in_double_quoted and (c == 'T' or c == 'F' or c == 'N') and
            (i == 0 or !isWordChar(input[i - 1])))
        {
            var n: usize = 0;
            while (i + n < input.len and isWordChar(input[i + n])) n += 1;
            const token = input[i .. i + n];
            var matched = false;
            for (literals) |lit| {
                const py = lit[0][0..@min(n, lit[0].len)];
                if (std.mem.eql(u8, py, token)) {
                    try result.appendSlice(gpa, lit[1][0..@min(n, lit[1].len)]);
                    i += n - 1;
                    matched = true;
                    break;
                }
            }
            if (!matched) try result.append(gpa, c);
        } else {
            try result.append(gpa, c);
        }
    }
    return result.toOwnedSlice(gpa);
}

/// Replaces `field` with a copy of `s`.
fn setStr(gpa: Allocator, field: *[]const u8, s: []const u8) Error!void {
    const copy = try gpa.dupe(u8, s);
    gpa.free(field.*);
    field.* = copy;
}

/// `field += s`.
fn appendStr(gpa: Allocator, field: *[]const u8, s: []const u8) Error!void {
    if (s.len == 0) return;
    const out = try gpa.alloc(u8, field.len + s.len);
    @memcpy(out[0..field.len], field.*);
    @memcpy(out[field.len..], s);
    gpa.free(field.*);
    field.* = out;
}

/// `field += node.sanitized_text()`: the node's text with every invalid
/// UTF-8 run replaced by U+FFFD.
fn appendSanitized(gpa: Allocator, field: *[]const u8, node: *const AstNode) Error!void {
    if (node.invalid_utf8.len == 0) return appendStr(gpa, field, node.text);
    const text = try node.sanitizedText(gpa);
    defer gpa.free(text);
    return appendStr(gpa, field, text);
}

// -----------------------------------------------------------------------------
// The tagged parser

/// Ports `tag_based_peg_mapper` (chat-peg-parser.h:189 @cd959bae9): every
/// tag's last text, keyed as a `std::map` orders them.
pub const TagMap = struct {
    pub const Entry = struct { key: []const u8, value: []const u8 };

    gpa: Allocator,
    /// sorted bytewise by key
    entries: std.ArrayList(Entry) = .empty,

    pub fn deinit(self: *TagMap) void {
        for (self.entries.items) |e| {
            self.gpa.free(e.key);
            self.gpa.free(e.value);
        }
        self.entries.deinit(self.gpa);
    }

    /// `tags[key] = value`.
    pub fn put(self: *TagMap, key: []const u8, value: []const u8) Error!void {
        var at: usize = 0;
        while (at < self.entries.items.len and std.mem.order(u8, self.entries.items[at].key, key) == .lt) at += 1;
        if (at < self.entries.items.len and std.mem.eql(u8, self.entries.items[at].key, key)) {
            const v = try self.gpa.dupe(u8, value);
            self.gpa.free(self.entries.items[at].value);
            self.entries.items[at].value = v;
            return;
        }
        const k = try self.gpa.dupe(u8, key);
        errdefer self.gpa.free(k);
        const v = try self.gpa.dupe(u8, value);
        errdefer self.gpa.free(v);
        try self.entries.insert(self.gpa, at, .{ .key = k, .value = v });
    }

    /// Ports `from_ast` (chat-peg-parser.cpp:188 @cd959bae9), a member of
    /// `tag_based_peg_mapper`.
    pub fn fromAst(self: *TagMap, arena: *const AstArena, result: *const ParseResult) Error!void {
        try arena.visitResult(result, self, struct {
            fn f(m: *TagMap, node: *const AstNode) Error!void {
                if (node.tag.len > 0) try m.put(node.tag, node.text);
            }
        }.f);
    }
};

/// Ports `tagged_parse_result` (chat-peg-parser.h:196 @cd959bae9).
pub const TaggedParseResult = struct {
    /// its node list and invalid runs owned here; the AST it named is gone
    result: ParseResult,
    tags: TagMap,

    pub fn deinit(self: *TaggedParseResult) void {
        self.tags.gpa.free(self.result.nodes);
        self.tags.gpa.free(self.result.invalid_utf8);
        self.tags.deinit();
    }
};

/// Ports `tagged_peg_parser` (chat-peg-parser.h:201 @cd959bae9).
pub const TaggedPegParser = struct {
    arena: peg.Arena,
    flags: u32 = peg.Flags.none,

    pub fn deinit(self: *TaggedPegParser) void {
        self.arena.deinit();
    }

    /// Ports `withDebug` (chat-peg-parser.h:205 @cd959bae9).
    pub fn withDebug(self: *TaggedPegParser) *TaggedPegParser {
        self.flags |= peg.Flags.debug;
        return self;
    }

    /// Ports `withoutDebug` (chat-peg-parser.h:210 @cd959bae9).
    pub fn withoutDebug(self: *TaggedPegParser) *TaggedPegParser {
        self.flags &= ~peg.Flags.debug;
        return self;
    }

    fn extract(gpa: Allocator, ctx: *peg.ParseContext, result: ParseResult) Error!TaggedParseResult {
        var tags: TagMap = .{ .gpa = gpa };
        errdefer tags.deinit();
        try tags.fromAst(&ctx.ast, &result);
        var r = result;
        r.nodes = try gpa.dupe(AstId, result.nodes);
        errdefer gpa.free(r.nodes);
        r.invalid_utf8 = try gpa.dupe(peg.InvalidUtf8, result.invalid_utf8);
        return .{ .result = r, .tags = tags };
    }

    /// Ports `parse_and_extract` (chat-peg-parser.cpp:196 @cd959bae9).
    ///
    /// Return: owned by the caller; free with `deinit`.
    pub fn parseAndExtract(self: *const TaggedPegParser, gpa: Allocator, input: []const u8, extra_flags: u32) Error!TaggedParseResult {
        var ctx = try peg.ParseContext.init(gpa, input, self.flags | extra_flags);
        defer ctx.deinit();
        const result = try self.arena.parse(&ctx, 0);
        return extract(gpa, &ctx, result);
    }

    /// Ports `parse_anywhere_and_extract` (chat-peg-parser.cpp:206 @cd959bae9):
    /// the first position the parser succeeds at, or the last position's
    /// result.
    ///
    /// Return: owned by the caller; free with `deinit`.
    pub fn parseAnywhereAndExtract(self: *const TaggedPegParser, gpa: Allocator, input: []const u8) Error!TaggedParseResult {
        if (input.len == 0) return self.parseAndExtract(gpa, input, peg.Flags.none);
        for (0..input.len) |i| {
            var ctx = try peg.ParseContext.init(gpa, input, self.flags);
            defer ctx.deinit();
            const result = try self.arena.parse(&ctx, i);
            if (result.success() or i == input.len - 1) return extract(gpa, &ctx, result);
        }
        unreachable;
    }
};

/// Ports `build_tagged_peg_parser` (chat-peg-parser.cpp:222 @cd959bae9).
pub fn buildTaggedPegParser(gpa: Allocator, ctx: anytype, comptime f: fn (@TypeOf(ctx), *peg.Builder) Error!ParserId) Error!TaggedPegParser {
    return .{ .arena = try peg.buildPegParser(gpa, ctx, f) };
}

// -----------------------------------------------------------------------------
// The mappers

/// Ports `common_chat_peg_mapper` (chat-peg-parser.h:10 @cd959bae9) and its
/// two subclasses, `common_chat_peg_gemma4_mapper` (chat-peg-parser.h:35 @cd959bae9)
/// and `common_chat_peg_minimax_m3_mapper` (chat-peg-parser.h:43 @cd959bae9).
pub const Mapper = struct {
    pub const Kind = enum { base, gemma4, minimax_m3 };

    /// Ports `TOOL_ARG_OBJECT` (chat-peg-parser.h:45 @cd959bae9).
    pub const TOOL_ARG_OBJECT = "tool-arg-object";
    /// Ports `TOOL_ARG_ARRAY` (chat-peg-parser.h:46 @cd959bae9).
    pub const TOOL_ARG_ARRAY = "tool-arg-array";
    /// Ports `TOOL_ARG_ITEM` (chat-peg-parser.h:47 @cd959bae9).
    pub const TOOL_ARG_ITEM = "tool-arg-item";

    /// Where `current_tool` points; `closed` is null, set by a `tool-close`.
    const Current = union(enum) { none, pending, result: usize, closed };

    gpa: Allocator,
    result: *Msg,
    kind: Kind = .base,
    pending_tool_call: ?ToolCall = null,
    current: Current = .none,
    arg_count: c_int = 0,
    closing_quote_pending: bool = false,
    args_buffer: []const u8 = "",
    /// whether a tool-related node arrived after a `tool-close` and before
    /// the next `tool-open` -- a harness's question, never the C++'s
    after_close: bool = false,

    /// Return: a mapper writing into `result`, whose strings `gpa` owns.
    pub fn init(gpa: Allocator, result: *Msg, kind: Kind) Mapper {
        return .{ .gpa = gpa, .result = result, .kind = kind };
    }

    pub fn deinit(self: *Mapper) void {
        if (self.pending_tool_call) |*p| p.deinit(self.gpa);
        self.gpa.free(self.args_buffer);
    }

    fn currentTool(self: *Mapper) ?*ToolCall {
        return switch (self.current) {
            .none, .closed => null,
            .pending => &self.pending_tool_call.?,
            .result => |i| &self.result.tool_calls.items[i],
        };
    }

    /// `current_tool` is not null; asked only of a tool-related node.
    fn live(self: *Mapper) bool {
        return switch (self.current) {
            .none => false,
            .closed => blk: {
                self.after_close = true;
                break :blk false;
            },
            else => true,
        };
    }

    /// Ports `args_target` (chat-peg-parser.cpp:276 @cd959bae9): the
    /// arguments once the tool's name is known, the buffer before.
    fn argsTarget(self: *Mapper) *[]const u8 {
        if (self.currentTool()) |t| {
            if (t.name.len > 0) return &t.arguments;
        }
        return &self.args_buffer;
    }

    /// Ports `normalize_container_value` (chat-peg-parser.cpp:280 @cd959bae9).
    fn normalizeContainerValue(self: *Mapper, input: []const u8) Error![]u8 {
        return normalizeQuotesToJson(self.gpa, input);
    }

    /// Ports `from_ast` (chat-peg-parser.cpp:284 @cd959bae9), and the two
    /// overrides, `common_chat_peg_gemma4_mapper`'s `from_ast` (chat-peg-parser.cpp:963 @cd959bae9)
    /// and `common_chat_peg_minimax_m3_mapper`'s `from_ast` (chat-peg-parser.cpp:1205 @cd959bae9).
    pub fn fromAst(self: *Mapper, arena: *const AstArena, parse_result: *const ParseResult) Error!void {
        switch (self.kind) {
            .gemma4 => for (parse_result.nodes) |n| try self.gemma4Visit(arena, n),
            .minimax_m3 => for (parse_result.nodes) |n| try self.minimaxVisit(arena, n),
            .base => {
                try arena.visitResult(parse_result, self, struct {
                    fn f(m: *Mapper, node: *const AstNode) Error!void {
                        try m.map(node);
                    }
                }.f);
                if (self.pending_tool_call) |*p| {
                    if (p.name.len > 0) {
                        if (self.args_buffer.len > 0) try setStr(self.gpa, &p.arguments, self.args_buffer);
                        if (self.closing_quote_pending and p.arguments.len > 0) try appendStr(self.gpa, &p.arguments, "\"");
                        try self.result.addToolCall(self.gpa, p.*);
                        p.deinit(self.gpa);
                        self.pending_tool_call = null;
                        self.current = .none;
                    }
                }
                const rc = self.result.reasoning_content;
                if (rc.len > 0) {
                    for (rc) |c| {
                        if (c != ' ' and c != '\n' and c != '\r' and c != '\t') return;
                    }
                    try setStr(self.gpa, &self.result.reasoning_content, "");
                }
            },
        }
    }

    /// Ports `map` (chat-peg-parser.cpp:315 @cd959bae9), a member of
    /// `common_chat_peg_mapper`: one node's effect on the message.
    pub fn map(self: *Mapper, node: *const AstNode) Error!void {
        const gpa = self.gpa;
        const T = ChatBuilder;
        const eq = std.mem.eql;
        const tag = node.tag;

        if (eq(u8, tag, T.REASONING)) try appendSanitized(gpa, &self.result.reasoning_content, node);
        if (eq(u8, tag, T.CONTENT)) try appendSanitized(gpa, &self.result.content, node);

        const is_tool_open = eq(u8, tag, T.TOOL_OPEN);
        const is_tool_close = eq(u8, tag, T.TOOL_CLOSE);
        const is_tool_name = eq(u8, tag, T.TOOL_NAME);
        const is_tool_id = eq(u8, tag, T.TOOL_ID);
        const is_tool_args = eq(u8, tag, T.TOOL_ARGS);
        const is_arg_open = eq(u8, tag, T.TOOL_ARG_OPEN);
        const is_arg_close = eq(u8, tag, T.TOOL_ARG_CLOSE);
        const is_arg_name = eq(u8, tag, T.TOOL_ARG_NAME);
        const is_arg_value = eq(u8, tag, T.TOOL_ARG_VALUE);
        const is_arg_string_value = eq(u8, tag, T.TOOL_ARG_STRING_VALUE);

        if (is_tool_open) {
            if (self.pending_tool_call) |*p| p.deinit(gpa);
            self.pending_tool_call = .{};
            self.current = .pending;
            self.arg_count = 0;
            try setStr(gpa, &self.args_buffer, "");
            self.closing_quote_pending = false;
        }

        if (is_tool_id and self.live()) {
            var text = trimTrailingSpace(node.text, -1);
            if (text.len >= 2 and text[0] == '"' and text[text.len - 1] == '"') text = text[1 .. text.len - 1];
            try setStr(gpa, &self.currentTool().?.id, text);
        }

        if (is_tool_name and self.live()) {
            const t = self.currentTool().?;
            try setStr(gpa, &t.name, trimTrailingSpace(node.text, -1));
            if (self.args_buffer.len > 0) {
                try setStr(gpa, &t.arguments, self.args_buffer);
                try setStr(gpa, &self.args_buffer, "");
            } else if (t.arguments.len == 0) {
                try setStr(gpa, &t.arguments, "{");
            }
            if (self.pending_tool_call) |*p| {
                try self.result.addToolCall(gpa, p.*);
                p.deinit(gpa);
                self.pending_tool_call = null;
                self.current = .{ .result = self.result.tool_calls.items.len - 1 };
            }
        }

        if (is_tool_args and self.live()) {
            const text = trimTrailingSpace(node.text, -1);
            if (text.len > 0 and text[0] == '{') try setStr(gpa, self.argsTarget(), text);
        }

        if (is_arg_open) self.closing_quote_pending = false;

        if (is_arg_name and self.live()) {
            var entry: std.ArrayList(u8) = .empty;
            defer entry.deinit(gpa);
            if (self.arg_count > 0) try entry.append(gpa, ',');
            const name = try dumpString(gpa, trim(node.text));
            defer gpa.free(name);
            try entry.appendSlice(gpa, name);
            try entry.append(gpa, ':');
            self.arg_count += 1;
            const target = self.argsTarget();
            if (target.len == 0) try setStr(gpa, target, "{");
            try appendStr(gpa, target, entry.items);
        }

        if ((is_arg_value or is_arg_string_value) and self.live()) {
            const value_content = node.text;
            var value_to_add: std.ArrayList(u8) = .empty;
            defer value_to_add.deinit(gpa);
            if (value_content.len == 0 and is_arg_string_value) {
                try value_to_add.append(gpa, '"');
                self.closing_quote_pending = true;
            } else if (value_content.len > 0 and is_arg_string_value) {
                if (!self.closing_quote_pending) {
                    try value_to_add.append(gpa, '"');
                    self.closing_quote_pending = true;
                }
                const esc = try escapeJsonStringInner(gpa, value_content);
                defer gpa.free(esc);
                try value_to_add.appendSlice(gpa, esc);
            } else if (value_content.len > 0) {
                const norm = try self.normalizeContainerValue(value_content);
                defer gpa.free(norm);
                try value_to_add.appendSlice(gpa, norm);
            }
            try appendStr(gpa, self.argsTarget(), value_to_add.items);
        }

        if (is_arg_close and self.live()) {
            if (self.closing_quote_pending) {
                try appendStr(gpa, self.argsTarget(), "\"");
                self.closing_quote_pending = false;
            }
        }

        if (is_tool_close and self.live()) {
            const t = self.currentTool().?;
            if (t.name.len == 0 and self.args_buffer.len > 0) {
                try setStr(gpa, &t.arguments, self.args_buffer);
                try setStr(gpa, &self.args_buffer, "");
            }
            if (self.closing_quote_pending) {
                try appendStr(gpa, &t.arguments, "\"");
                self.closing_quote_pending = false;
            }
            var d = jsonBraceDepth(t.arguments);
            while (d > 0) : (d -= 1) try appendStr(gpa, &t.arguments, "}");
            if (self.pending_tool_call) |*p| {
                if (t.name.len > 0) try self.result.addToolCall(gpa, p.*);
                p.deinit(gpa);
                self.pending_tool_call = null;
                self.current = .closed;
            }
        }
    }

    // -------------------------------------------------------------------------
    // gemma4

    /// Ports `gemma4_to_json` (chat-peg-parser.cpp:969 @cd959bae9): a value
    /// of Gemma 4's tool-call syntax as JSON, left open while partial.
    ///
    /// Return: owned by the caller.
    fn gemma4ToJson(self: *Mapper, arena: *const AstArena, id: AstId) Error![]u8 {
        const gpa = self.gpa;
        const node = try arena.get(id);
        const eq = std.mem.eql;
        if (node.text.len == 0) return gpa.dupe(u8, "");
        const r = node.rule;
        if (eq(u8, r, "gemma4-number") or eq(u8, r, "gemma4-bool") or eq(u8, r, "gemma4-null")) return gpa.dupe(u8, node.text);
        if (eq(u8, r, "gemma4-string-content")) return escapeJsonStringInner(gpa, node.text);

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        if (eq(u8, r, "gemma4-string")) {
            try out.append(gpa, '"');
            if (node.children.len > 0) {
                const inner = try self.gemma4ToJson(arena, node.children[0]);
                defer gpa.free(inner);
                try out.appendSlice(gpa, inner);
                if (!node.is_partial) try out.append(gpa, '"');
            }
            return out.toOwnedSlice(gpa);
        }
        if (eq(u8, r, "gemma4-array") or eq(u8, r, "gemma4-dict")) {
            const dict = r[7] == 'd';
            try out.append(gpa, if (dict) '{' else '[');
            for (node.children, 0..) |child_id, i| {
                if (i > 0) try out.append(gpa, ',');
                const inner = try self.gemma4ToJson(arena, child_id);
                defer gpa.free(inner);
                try out.appendSlice(gpa, inner);
            }
            if (!node.is_partial) try out.append(gpa, if (dict) '}' else ']');
            return out.toOwnedSlice(gpa);
        }
        if (eq(u8, r, "gemma4-dict-key-name")) return gpa.dupe(u8, node.text);
        if (eq(u8, r, "gemma4-dict-key")) {
            try out.append(gpa, '"');
            if (node.children.len > 0) {
                const inner = try self.gemma4ToJson(arena, node.children[0]);
                defer gpa.free(inner);
                const esc = try escapeJsonStringInner(gpa, inner);
                defer gpa.free(esc);
                try out.appendSlice(gpa, esc);
            }
            if (!node.is_partial) try out.appendSlice(gpa, "\":");
            return out.toOwnedSlice(gpa);
        }
        if (eq(u8, r, "gemma4-dict-kv")) {
            for (node.children) |child_id| {
                const inner = try self.gemma4ToJson(arena, child_id);
                defer gpa.free(inner);
                try out.appendSlice(gpa, inner);
            }
            return out.toOwnedSlice(gpa);
        }
        if (eq(u8, r, "gemma4-value")) {
            if (node.children.len > 0) return self.gemma4ToJson(arena, node.children[0]);
            return gpa.dupe(u8, "");
        }
        return gpa.dupe(u8, "");
    }

    /// Ports `common_chat_peg_gemma4_mapper`'s `visit` (chat-peg-parser.cpp:1064 @cd959bae9).
    fn gemma4Visit(self: *Mapper, arena: *const AstArena, id: AstId) Error!void {
        const gpa = self.gpa;
        const node = try arena.get(id);
        if (std.mem.eql(u8, node.tag, "reasoning")) return appendSanitized(gpa, &self.result.reasoning_content, node);
        if (std.mem.eql(u8, node.tag, "content")) return appendSanitized(gpa, &self.result.content, node);
        if (std.mem.eql(u8, node.tag, "tool")) {
            const name_id = try arena.findByTag(node, "tool-name", 3);
            const args_id = try arena.findByTag(node, "tool-args", 3);
            if (name_id != invalid_ast_id and args_id != invalid_ast_id) {
                const name_node = try arena.get(name_id);
                const args_node = try arena.get(args_id);
                if (!name_node.is_partial) {
                    var args: []u8 = "";
                    defer gpa.free(args);
                    if (args_node.children.len > 0) args = try self.gemma4ToJson(arena, args_node.children[0]);
                    try self.result.addToolCall(gpa, .{ .name = name_node.text, .arguments = args });
                }
            }
            return;
        }
        for (node.children) |child| try self.gemma4Visit(arena, child);
    }

    // -------------------------------------------------------------------------
    // minimax-m3

    /// Ports `minimax_m3_collect` (chat-peg-parser.cpp:1103 @cd959bae9): the
    /// nearest descendants tagged `tag`.
    fn minimaxCollect(gpa: Allocator, arena: *const AstArena, node: *const AstNode, tag: []const u8, out: *std.ArrayList(AstId)) Error!void {
        for (node.children) |child_id| {
            const child = try arena.get(child_id);
            if (std.mem.eql(u8, child.tag, tag)) {
                try out.append(gpa, child_id);
            } else {
                try minimaxCollect(gpa, arena, child, tag, out);
            }
        }
    }

    /// Ports `minimax_m3_value_of` (chat-peg-parser.cpp:1117 @cd959bae9).
    fn minimaxValueOf(arena: *const AstArena, node: *const AstNode) Error!AstId {
        for (node.children) |child_id| {
            const tag = (try arena.get(child_id)).tag;
            const eq = std.mem.eql;
            if (eq(u8, tag, ChatBuilder.TOOL_ARG_VALUE) or eq(u8, tag, ChatBuilder.TOOL_ARG_STRING_VALUE) or
                eq(u8, tag, TOOL_ARG_OBJECT) or eq(u8, tag, TOOL_ARG_ARRAY)) return child_id;
        }
        return invalid_ast_id;
    }

    /// Ports `minimax_m3_member_to_json` (chat-peg-parser.cpp:1132 @cd959bae9).
    fn minimaxMemberToJson(self: *Mapper, arena: *const AstArena, node: *const AstNode) Error![]u8 {
        const gpa = self.gpa;
        const name_id = try arena.findByTag(node, ChatBuilder.TOOL_ARG_NAME, 3);
        if (name_id == invalid_ast_id) return gpa.dupe(u8, "");
        const name = try dumpString(gpa, (try arena.get(name_id)).text);
        defer gpa.free(name);
        const value = try self.minimaxValueToJson(arena, try minimaxValueOf(arena, node), !node.is_partial);
        defer gpa.free(value);
        return std.mem.concat(gpa, u8, &.{ name, ":", value });
    }

    /// Ports `minimax_m3_container_to_json` (chat-peg-parser.cpp:1142 @cd959bae9).
    fn minimaxContainerToJson(self: *Mapper, arena: *const AstArena, node: *const AstNode, is_object: bool, closed: bool) Error![]u8 {
        const gpa = self.gpa;
        const tag = if (is_object) ChatBuilder.TOOL_ARG else TOOL_ARG_ITEM;
        var entries: std.ArrayList(AstId) = .empty;
        defer entries.deinit(gpa);
        try minimaxCollect(gpa, arena, node, tag, &entries);

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.append(gpa, if (is_object) '{' else '[');
        var add_comma = false;
        for (entries.items) |entry_id| {
            const entry = try arena.get(entry_id);
            const text = if (is_object)
                try self.minimaxMemberToJson(arena, entry)
            else
                try self.minimaxValueToJson(arena, try minimaxValueOf(arena, entry), !entry.is_partial);
            defer gpa.free(text);
            if (text.len == 0) continue;
            if (add_comma) try out.append(gpa, ',');
            add_comma = true;
            try out.appendSlice(gpa, text);
        }
        if (closed) try out.append(gpa, if (is_object) '}' else ']');
        return out.toOwnedSlice(gpa);
    }

    /// Ports `minimax_m3_value_to_json` (chat-peg-parser.cpp:1182 @cd959bae9).
    fn minimaxValueToJson(self: *Mapper, arena: *const AstArena, id: AstId, closed: bool) Error![]u8 {
        const gpa = self.gpa;
        if (id == invalid_ast_id) return gpa.dupe(u8, "");
        const node = try arena.get(id);
        if (std.mem.eql(u8, node.tag, TOOL_ARG_OBJECT)) return self.minimaxContainerToJson(arena, node, true, closed);
        if (std.mem.eql(u8, node.tag, TOOL_ARG_ARRAY)) return self.minimaxContainerToJson(arena, node, false, closed);
        if (std.mem.eql(u8, node.tag, ChatBuilder.TOOL_ARG_STRING_VALUE)) {
            const esc = try escapeJsonStringInner(gpa, node.text);
            defer gpa.free(esc);
            return std.mem.concat(gpa, u8, &.{ "\"", esc, if (closed) "\"" else "" });
        }
        return gpa.dupe(u8, node.text);
    }

    /// Ports `common_chat_peg_minimax_m3_mapper`'s `visit` (chat-peg-parser.cpp:1212 @cd959bae9).
    fn minimaxVisit(self: *Mapper, arena: *const AstArena, id: AstId) Error!void {
        const gpa = self.gpa;
        const node = try arena.get(id);
        if (std.mem.eql(u8, node.tag, ChatBuilder.REASONING)) return appendSanitized(gpa, &self.result.reasoning_content, node);
        if (std.mem.eql(u8, node.tag, ChatBuilder.CONTENT)) return appendSanitized(gpa, &self.result.content, node);
        if (std.mem.eql(u8, node.tag, ChatBuilder.TOOL)) {
            const name_id = try arena.findByTag(node, ChatBuilder.TOOL_NAME, 3);
            if (name_id != invalid_ast_id) {
                const name = (try arena.get(name_id)).text;
                const args = try self.minimaxContainerToJson(arena, node, true, !node.is_partial);
                defer gpa.free(args);
                try self.result.addToolCall(gpa, .{ .name = name, .arguments = args });
            }
            return;
        }
        for (node.children) |child| try self.minimaxVisit(arena, child);
    }
};

// -----------------------------------------------------------------------------
// The builder

/// Ports `COMMON_CHAT_MAX_PERMUTE` (chat-peg-parser.h:58 @cd959bae9).
pub const max_permute: usize = 6;

/// One entry of `standard_constructed_tools`' `std::map` of markers.
pub const Marker = struct { key: []const u8, value: []const u8 };

/// Ports `parse_key_spec` (chat-peg-parser.cpp:623 @cd959bae9): `a.b` as
/// `{a, b}`, a key without a dot as `{"", key}`.
fn parseKeySpec(key: []const u8) struct { []const u8, []const u8 } {
    const dot = std.mem.indexOfScalar(u8, key, '.') orelse return .{ "", key };
    return .{ key[0..dot], key[dot + 1 ..] };
}

/// Ports `common_chat_peg_builder` (chat-peg-parser.h:60 @cd959bae9).
pub const ChatBuilder = struct {
    b: peg.Builder,

    pub const REASONING_BLOCK = "reasoning-block";
    pub const REASONING = "reasoning";
    pub const CONTENT = "content";
    pub const TOOL = "tool";
    pub const TOOL_OPEN = "tool-open";
    pub const TOOL_CLOSE = "tool-close";
    pub const TOOL_ID = "tool-id";
    pub const TOOL_NAME = "tool-name";
    pub const TOOL_ARGS = "tool-args";
    pub const TOOL_ARG = "tool-arg";
    pub const TOOL_ARG_OPEN = "tool-arg-open";
    pub const TOOL_ARG_CLOSE = "tool-arg-close";
    pub const TOOL_ARG_NAME = "tool-arg-name";
    pub const TOOL_ARG_VALUE = "tool-arg-value";
    pub const TOOL_ARG_STRING_VALUE = "tool-arg-string-value";

    pub fn init(gpa: Allocator) Error!ChatBuilder {
        return .{ .b = try peg.Builder.init(gpa) };
    }

    pub fn deinit(self: *ChatBuilder) void {
        self.b.deinit();
    }

    fn alloc(self: *ChatBuilder) Allocator {
        return self.b.arena.gpa;
    }

    fn tmp(self: *ChatBuilder) Allocator {
        return self.b.arena.alloc();
    }

    fn cat(self: *ChatBuilder, parts: []const []const u8) Error![]u8 {
        return std.mem.concat(self.tmp(), u8, parts);
    }

    fn quoted(self: *ChatBuilder, s: []const u8) Error!ParserId {
        return self.b.literal(try self.cat(&.{ "\"", s, "\"" }));
    }

    /// Ports `reasoning_block` (chat-peg-parser.h:82 @cd959bae9).
    pub fn reasoningBlock(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(REASONING_BLOCK, p);
    }

    /// Ports `reasoning` (chat-peg-parser.h:84 @cd959bae9).
    pub fn reasoning(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(REASONING, p);
    }

    /// Ports `content` (chat-peg-parser.h:86 @cd959bae9).
    pub fn content(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(CONTENT, p);
    }

    /// Ports `tool` (chat-peg-parser.h:93 @cd959bae9).
    pub fn tool(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(TOOL, p);
    }

    /// Ports `tool_open` (chat-peg-parser.h:94 @cd959bae9).
    pub fn toolOpen(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.atomic(try self.b.tag(TOOL_OPEN, p));
    }

    /// Ports `tool_close` (chat-peg-parser.h:95 @cd959bae9).
    pub fn toolClose(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.atomic(try self.b.tag(TOOL_CLOSE, p));
    }

    /// Ports `tool_id` (chat-peg-parser.h:96 @cd959bae9).
    pub fn toolId(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.atomic(try self.b.tag(TOOL_ID, p));
    }

    /// Ports `tool_name` (chat-peg-parser.h:97 @cd959bae9).
    pub fn toolName(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.atomic(try self.b.tag(TOOL_NAME, p));
    }

    /// Ports `tool_args` (chat-peg-parser.h:98 @cd959bae9).
    pub fn toolArgs(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(TOOL_ARGS, p);
    }

    /// Ports `tool_arg` (chat-peg-parser.h:99 @cd959bae9).
    pub fn toolArg(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(TOOL_ARG, p);
    }

    /// Ports `tool_arg_open` (chat-peg-parser.h:100 @cd959bae9).
    pub fn toolArgOpen(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.atomic(try self.b.tag(TOOL_ARG_OPEN, p));
    }

    /// Ports `tool_arg_close` (chat-peg-parser.h:101 @cd959bae9).
    pub fn toolArgClose(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.atomic(try self.b.tag(TOOL_ARG_CLOSE, p));
    }

    /// Ports `tool_arg_name` (chat-peg-parser.h:102 @cd959bae9).
    pub fn toolArgName(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.atomic(try self.b.tag(TOOL_ARG_NAME, p));
    }

    /// Ports `tool_arg_value` (chat-peg-parser.h:103 @cd959bae9).
    pub fn toolArgValue(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(TOOL_ARG_VALUE, p);
    }

    /// Ports `tool_arg_string_value` (chat-peg-parser.h:106 @cd959bae9).
    pub fn toolArgStringValue(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(TOOL_ARG_STRING_VALUE, p);
    }

    /// Ports `tool_arg_json_value` (chat-peg-parser.h:107 @cd959bae9).
    pub fn toolArgJsonValue(self: *ChatBuilder, p: ParserId) Error!ParserId {
        return self.b.tag(TOOL_ARG_VALUE, p);
    }

    /// Ports `tag_with_safe_content` (chat-peg-parser.cpp:229 @cd959bae9):
    /// `p` wherever it matches, content everywhere else.
    pub fn tagWithSafeContent(self: *ChatBuilder, tag_name: []const u8, marker: []const u8, p: ParserId) Error!ParserId {
        const b = &self.b;
        if (marker.len == 0) {
            return b.zeroOrMore(try b.choice(&.{ p, try b.rule(tag_name, try self.content(try b.any()), false) }));
        }
        const chunk = try b.plus(try b.plus(try b.negate(try b.literal(marker)), try b.any()), try b.until(marker));
        const content_chunk = try b.rule(tag_name, try self.content(chunk), false);
        return b.zeroOrMore(try b.choice(&.{ p, content_chunk }));
    }

    /// Ports `permute` (chat-peg-parser.cpp:239 @cd959bae9): every parser
    /// once, in any order -- one rule per subset still to match, so shared
    /// tails are written once. Past `max_permute` parsers, the given order.
    pub fn permute(self: *ChatBuilder, rule_prefix: []const u8, parsers: []const ParserId) Error!ParserId {
        const b = &self.b;
        if (parsers.len == 0) return b.eps();
        if (parsers.len == 1 or parsers.len > max_permute) return b.sequence(parsers);

        const P = struct {
            cb: *ChatBuilder,
            prefix: []const u8,
            parsers: []const ParserId,
            rules: std.AutoHashMapUnmanaged(u32, ParserId) = .empty,

            fn remainingOf(st: *@This(), remaining: u32) Error!ParserId {
                const bb = &st.cb.b;
                if (remaining == 0) return bb.eps();
                if (st.rules.get(remaining)) |cached| return cached;
                var alternatives = try bb.choiceEmpty();
                for (st.parsers, 0..) |p, i| {
                    const bit = @as(u32, 1) << @intCast(i);
                    if (remaining & bit != 0) {
                        const rest = try st.remainingOf(remaining & ~bit);
                        try bb.orAssign(&alternatives, try bb.plus(p, rest));
                    }
                }
                const name = try std.fmt.allocPrint(st.cb.tmp(), "{s}-{d}", .{ st.prefix, remaining });
                const r = try bb.rule(name, alternatives, false);
                try st.rules.put(st.cb.tmp(), remaining, r);
                return r;
            }
        };
        var st: P = .{ .cb = self, .prefix = rule_prefix, .parsers = parsers };
        return st.remainingOf((@as(u32, 1) << @intCast(parsers.len)) - 1);
    }

    /// Ports `prefix` (chat-peg-parser.cpp:876 @cd959bae9): `s` up to the
    /// first `delimiter`.
    pub fn prefix(self: *ChatBuilder, s: []const u8, delimiter: []const u8) Error!ParserId {
        if (s.len == 0) return self.b.eps();
        if (delimiter.len == 0) return self.b.literal(s);
        const at = std.mem.indexOf(u8, s, delimiter) orelse s.len;
        return self.b.literal(s[0..at]);
    }

    /// Ports `optspace` (chat-peg-parser.cpp:886 @cd959bae9): `tag` with its
    /// leading and trailing spaces each optional.
    pub fn optspace(self: *ChatBuilder, tag: []const u8) Error!ParserId {
        const b = &self.b;
        var parser = try b.eps();
        var end_of_prefix_space = tag.len;
        var start_of_suffix_space = tag.len;
        for (tag, 0..) |c, i| {
            if (!isCSpace(c)) {
                end_of_prefix_space = i;
                break;
            }
        }
        var i = tag.len;
        while (i > 0) : (i -= 1) {
            if (!isCSpace(tag[i - 1])) {
                start_of_suffix_space = i;
                break;
            }
        }
        for (0..end_of_prefix_space) |k| try b.plusAssign(&parser, try b.optional(try b.literal(tag[k .. k + 1])));
        const mid_len = start_of_suffix_space -% end_of_prefix_space;
        const mid = tag[end_of_prefix_space..][0..@min(mid_len, tag.len - end_of_prefix_space)];
        try b.plusAssign(&parser, try b.literal(mid));
        for (start_of_suffix_space..tag.len) |k| try b.plusAssign(&parser, try b.optional(try b.literal(tag[k .. k + 1])));
        return parser;
    }

    /// One function of `tools`: its name and parameters, or null when the
    /// entry has no `function`.
    const Function = struct { name: []const u8, params: Value };

    fn function(self: *ChatBuilder, tool_def: *const Value) Error!?Function {
        if (!tool_def.contains("function")) return null;
        const f = tool_def.at("function") catch |e| return fromJson(e);
        const name_v = f.at("name") catch |e| return fromJson(e);
        const name = name_v.get([]const u8) catch |e| return fromJson(e);
        const params = msg_mod.toolParameters(self.alloc(), f) catch |e| return fromChat(e);
        return .{ .name = try self.tmp().dupe(u8, name), .params = params };
    }

    /// `params.contains("properties") && !params["properties"].empty()`.
    fn properties(params: *Value) ?*Value {
        if (!params.contains("properties")) return null;
        const p = params.at("properties") catch return null;
        if (p.empty()) return null;
        return p;
    }

    /// Ports `standard_constructed_tools` (chat-peg-parser.cpp:459 @cd959bae9):
    /// XML-like tool calls, `<function=name><param=k>v</param></function>`
    /// by default, each marker replaceable.
    pub fn standardConstructedTools(self: *ChatBuilder, markers: []const Marker, tools: *const Value, parallel_tool_calls: bool, force_tool_calls: bool) Error!ParserId {
        const b = &self.b;
        if (!tools.isArray() or tools.empty()) return b.eps();

        const getMarker = struct {
            fn f(ms: []const Marker, key: []const u8, default: []const u8) []const u8 {
                for (ms) |m| if (std.mem.eql(u8, m.key, key)) return m.value;
                return default;
            }
        }.f;
        const section_start = getMarker(markers, "tool_call_start_marker", "<tool_call>");
        const section_end = getMarker(markers, "tool_call_end_marker", "</tool_call>");
        const func_opener = getMarker(markers, "function_opener", "<function=");
        const func_name_suffix = getMarker(markers, "function_name_suffix", ">");
        const func_closer = getMarker(markers, "function_closer", "</function>");
        const param_key_prefix = getMarker(markers, "parameter_key_prefix", "<param=");
        const param_key_suffix = getMarker(markers, "parameter_key_suffix", ">");
        const param_closer = getMarker(markers, "parameter_closer", "</param>");

        var tool_choices = try b.choiceEmpty();
        for (tools.array.items, 0..) |*tool_def, tool_index| {
            var f = (try self.function(tool_def)) orelse continue;
            defer f.params.deinit(self.alloc());

            var args = try b.eps();
            if (properties(&f.params)) |props| {
                var arg_choice = try b.choiceEmpty();
                var it = props.items();
                while (it.next()) |el| {
                    const prop_name = try self.tmp().dupe(u8, el.key);
                    const arg_name_parser = try b.choice(&.{
                        try b.literal(prop_name),
                        try self.quoted(prop_name),
                        try b.literal(try self.cat(&.{ "'", prop_name, "'" })),
                    });
                    var arg = try self.toolArgOpen(try b.literal(param_key_prefix));
                    arg = try b.plus(arg, try self.toolArgName(arg_name_parser));
                    arg = try b.plus(arg, try b.literal(param_key_suffix));
                    arg = try b.plus(arg, try self.toolArgValue(try b.until(param_closer)));
                    arg = try b.plus(arg, try self.toolArgClose(try b.literal(param_closer)));
                    const arg_rule = try self.toolArg(arg);
                    try b.orAssign(&arg_choice, arg_rule);
                }
                args = try b.zeroOrMore(try b.plus(arg_choice, try b.space()));
            }

            var open = try b.plus(try b.literal(func_opener), try self.toolName(try b.literal(f.name)));
            open = try b.plus(open, try b.literal(func_name_suffix));
            var body = try b.plus(try self.toolOpen(open), try b.space());
            body = try b.plus(body, try self.toolArgs(args));
            body = try b.plus(body, try b.space());
            body = try b.plus(body, try self.toolClose(try b.literal(func_closer)));
            const tool_parser = try self.tool(body);
            try b.orAssign(&tool_choices, try b.rule(try std.fmt.allocPrint(self.tmp(), "tool-{d}", .{tool_index}), tool_parser, false));
        }

        var section: ParserId = undefined;
        if (parallel_tool_calls) {
            var s = try b.plus(try b.literal(section_start), try b.space());
            s = try b.plus(s, try b.oneOrMore(try b.plus(tool_choices, try b.space())));
            s = try b.plus(s, try b.literal(section_end));
            section = try b.triggerRule("tool-call", s);
        } else {
            var s = try b.plus(try b.literal(section_start), try b.space());
            s = try b.plus(s, tool_choices);
            s = try b.plus(s, try b.space());
            s = try b.plus(s, try b.literal(section_end));
            section = try b.triggerRule("tool-call", s);
        }
        return if (force_tool_calls) section else b.optional(section);
    }

    /// Ports `python_or_json_value` (chat-peg-parser.cpp:532 @cd959bae9):
    /// `python_value` whose leaves also take JSON's `true`, `false` and
    /// `null`.
    fn pythonOrJsonValue(self: *ChatBuilder) Error!ParserId {
        return self.b.ruleFn("python-or-json-value", self, struct {
            fn body(cb: *ChatBuilder) Error!ParserId {
                const b = &cb.b;
                const Ctx = struct { b: *peg.Builder, ws: ParserId, members: ParserId = undefined, elements: ParserId = undefined };
                var ctx: Ctx = .{ .b = b, .ws = try b.space() };
                const value = try cb.pythonOrJsonValue();
                const ws = ctx.ws;

                const member = try b.sequence(&.{ try b.pythonString(), ws, try b.literal(":"), ws, value });
                ctx.members = try b.sequence(&.{ member, try b.zeroOrMore(try b.sequence(&.{ ws, try b.literal(","), ws, member })) });
                const dict = try b.ruleFn("python-or-json-dict", &ctx, struct {
                    fn f(c: *Ctx) Error!ParserId {
                        const bb = c.b;
                        return bb.sequence(&.{ try bb.literal("{"), c.ws, try bb.choice(&.{ try bb.literal("}"), try bb.sequence(&.{ c.members, c.ws, try bb.literal("}") }) }), c.ws });
                    }
                }.f, false);

                ctx.elements = try b.sequence(&.{ value, try b.zeroOrMore(try b.sequence(&.{ try b.literal(","), ws, value })) });
                const array = try b.ruleFn("python-or-json-array", &ctx, struct {
                    fn f(c: *Ctx) Error!ParserId {
                        const bb = c.b;
                        return bb.sequence(&.{ try bb.literal("["), c.ws, try bb.choice(&.{ try bb.literal("]"), try bb.sequence(&.{ c.elements, c.ws, try bb.literal("]") }) }), c.ws });
                    }
                }.f, false);

                return b.choice(&.{ dict, array, try b.pythonString(), try b.pythonNumber(), try b.pythonBool(), try b.pythonNull(), try b.jsonBool(), try b.jsonNull() });
            }
        }.body, false);
    }

    /// Ports `python_style_tool_calls` (chat-peg-parser.cpp:555 @cd959bae9):
    /// `[name(arg="v", n=1), ...]`, as LFM2 writes them.
    pub fn pythonStyleToolCalls(self: *ChatBuilder, tools: *const Value, parallel_tool_calls: bool, allow_json_literals: bool) Error!ParserId {
        const b = &self.b;
        if (!tools.isArray() or tools.empty()) return b.eps();

        var tool_choices = try b.choiceEmpty();
        for (tools.array.items, 0..) |*tool_def, tool_index| {
            var f = (try self.function(tool_def)) orelse continue;
            defer f.params.deinit(self.alloc());

            var args = try b.eps();
            if (properties(&f.params)) |props| {
                var arg_choice = try b.choiceEmpty();
                var it = props.items();
                while (it.next()) |el| {
                    const prop_name = try self.tmp().dupe(u8, el.key);
                    const prop_def = el.value;
                    var is_string_type = false;
                    if (prop_def.contains("type")) {
                        const t = prop_def.at("type") catch |e| return fromJson(e);
                        is_string_type = t.isString() and std.mem.eql(u8, t.string, "string");
                    }

                    const arg_name_parser = try b.literal(prop_name);
                    var arg_value_parser = try b.eps();
                    const dq = try b.plusLit(try b.plus(try b.literal("\""), try b.stringContent('"')), "\"");
                    const sq = try b.plusLit(try b.plus(try b.literal("'"), try b.stringContent('\'')), "'");
                    const string_value_parser = try self.toolArgValue(try b.choice(&.{ dq, sq }));

                    if (is_string_type) {
                        arg_value_parser = string_value_parser;
                    } else {
                        arg_value_parser = try self.toolArgValue(if (allow_json_literals) try self.pythonOrJsonValue() else try b.pythonValue());
                    }

                    const open = try self.toolArgOpen(try b.plus(try self.toolArgName(arg_name_parser), try b.literal("=")));
                    var arg = try b.plus(open, arg_value_parser);
                    arg = try b.plus(arg, try self.toolArgClose(try b.eps()));
                    const arg_rule = try self.toolArg(arg);
                    try b.orAssign(&arg_choice, arg_rule);
                }
                args = try b.plus(arg_choice, try b.zeroOrMore(try b.plus(try b.litPlus(",", try b.space()), arg_choice)));
            }

            const open = try self.toolOpen(try b.plus(try self.toolName(try b.literal(f.name)), try b.literal("(")));
            var body = try b.plus(open, try b.space());
            body = try b.plus(body, try self.toolArgs(args));
            body = try b.plus(body, try b.space());
            body = try b.plus(body, try self.toolClose(try b.literal(")")));
            const tool_parser = try self.tool(body);
            try b.orAssign(&tool_choices, try b.rule(try std.fmt.allocPrint(self.tmp(), "tool-{d}", .{tool_index}), tool_parser, false));
        }

        if (parallel_tool_calls) {
            var s = try b.litPlus("[", try b.space());
            s = try b.plus(s, tool_choices);
            s = try b.plus(s, try b.zeroOrMore(try b.plus(try b.litPlus(",", try b.space()), tool_choices)));
            s = try b.plus(s, try b.space());
            return b.plusLit(s, "]");
        }
        var s = try b.litPlus("[", try b.space());
        s = try b.plus(s, tool_choices);
        s = try b.plus(s, try b.space());
        return b.plusLit(s, "]");
    }

    /// A left fold of `+`, as the C++ writes `a + b + c`: each operand is
    /// built just before it is added, which is the order clang builds them.
    const Chain = struct {
        b: *peg.Builder,
        acc: ParserId,

        fn add(self: *Chain, p: ParserId) Error!void {
            self.acc = try self.b.plus(self.acc, p);
        }
    };

    fn chain(self: *ChatBuilder, first: ParserId) Chain {
        return .{ .b = &self.b, .acc = first };
    }

    /// `literal("\"" + key + "\"") + space() + literal(":") + space()`.
    fn keyChain(self: *ChatBuilder, key: []const u8) Error!Chain {
        const b = &self.b;
        var c = self.chain(try self.quoted(key));
        try c.add(try b.space());
        try c.add(try b.literal(":"));
        try c.add(try b.space());
        return c;
    }

    /// `literal("\"") + tool_id(string_content('"')) + literal("\"")`.
    fn quotedId(self: *ChatBuilder) Error!ParserId {
        const b = &self.b;
        var c = self.chain(try b.literal("\""));
        try c.add(try self.toolId(try b.stringContent('"')));
        try c.add(try b.literal("\""));
        return c.acc;
    }

    /// `atomic(key + "\"" + tool_id(string_content('"')) + "\"")`.
    fn idQuoted(self: *ChatBuilder, key: []const u8) Error!ParserId {
        const b = &self.b;
        var c = try self.keyChain(key);
        try c.add(try b.literal("\""));
        try c.add(try self.toolId(try b.stringContent('"')));
        try c.add(try b.literal("\""));
        return b.atomic(c.acc);
    }

    /// `atomic(key + choice({"\"" id "\"", tool_id(json_number())}))`.
    fn idOrNumber(self: *ChatBuilder, key: []const u8) Error!ParserId {
        const b = &self.b;
        var c = try self.keyChain(key);
        try c.add(try b.choice(&.{ try self.quotedId(), try self.toolId(try b.jsonNumber()) }));
        return b.atomic(c.acc);
    }

    /// `p + space() + optional(literal(",") + space())`, optional.
    fn optionalWithComma(self: *ChatBuilder, p: ParserId) Error!ParserId {
        const b = &self.b;
        var c = self.chain(p);
        try c.add(try b.space());
        try c.add(try b.optional(try b.plus(try b.literal(","), try b.space())));
        return b.optional(c.acc);
    }

    /// `optional(p + space() + literal(",") + space())`.
    fn optionalThenComma(self: *ChatBuilder, p: ParserId) Error!ParserId {
        const b = &self.b;
        var c = self.chain(p);
        try c.add(try b.space());
        try c.add(try b.literal(","));
        try c.add(try b.space());
        return b.optional(c.acc);
    }

    /// `tool_args(schema(json(), "tool-" + std::to_string(i) + "-schema", params))`.
    fn schemaArgs(self: *ChatBuilder, tool_index: usize, params: *const Value) Error!ParserId {
        const b = &self.b;
        const j = try b.json();
        return self.toolArgs(try b.schemaJson(j, try std.fmt.allocPrint(self.tmp(), "tool-{d}-schema", .{tool_index}), params, false));
    }

    /// `atomic(literal("\"") + tool_name(literal(name)) + literal("\""))`.
    fn quotedName(self: *ChatBuilder, name: []const u8) Error!ParserId {
        const b = &self.b;
        var c = self.chain(try b.literal("\""));
        try c.add(try self.toolName(try b.literal(name)));
        try c.add(try b.literal("\""));
        return b.atomic(c.acc);
    }

    /// Ports `build_json_tools_function_is_key` (chat-peg-parser.cpp:632 @cd959bae9):
    /// `{"name": {...}}`.
    fn buildJsonToolsFunctionIsKey(self: *ChatBuilder, tools: *const Value, args_key: []const u8, effective_args_key: []const u8, call_id_key: []const u8, gen_call_id_key: []const u8) Error!ParserId {
        const b = &self.b;
        var tool_choices = try b.choiceEmpty();
        for (tools.array.items, 0..) |*tool_def, tool_index| {
            var f = (try self.function(tool_def)) orelse continue;
            defer f.params.deinit(self.alloc());

            var inner_fields: std.ArrayList(ParserId) = .empty;
            if (call_id_key.len > 0) {
                const id_parser = try self.idQuoted(call_id_key);
                try inner_fields.append(self.tmp(), try self.optionalWithComma(id_parser));
            }
            if (gen_call_id_key.len > 0) {
                const gen_id_parser = try self.idOrNumber(gen_call_id_key);
                try inner_fields.append(self.tmp(), try self.optionalWithComma(gen_id_parser));
            }

            var args_parser = try b.eps();
            if (args_key.len == 0) {
                args_parser = try self.schemaArgs(tool_index, &f.params);
            } else {
                var c = try self.keyChain(effective_args_key);
                try c.add(try self.schemaArgs(tool_index, &f.params));
                args_parser = c.acc;
            }
            try inner_fields.append(self.tmp(), args_parser);

            var inner_object = try b.eps();
            if (args_key.len == 0 and inner_fields.items.len == 1) {
                inner_object = inner_fields.items[0];
            } else {
                var c = self.chain(try b.literal("{"));
                try c.add(try b.space());
                for (inner_fields.items, 0..) |field, i| {
                    try c.add(field);
                    if (i < inner_fields.items.len - 1) try c.add(try b.space());
                }
                try c.add(try b.space());
                try c.add(try b.literal("}"));
                inner_object = c.acc;
            }

            var c = self.chain(try self.toolOpen(try b.literal("{")));
            try c.add(try b.space());
            try c.add(try b.literal("\""));
            try c.add(try self.toolName(try b.literal(f.name)));
            try c.add(try b.literal("\""));
            try c.add(try b.space());
            try c.add(try b.literal(":"));
            try c.add(try b.space());
            try c.add(inner_object);
            try c.add(try b.space());
            try c.add(try self.toolClose(try b.literal("}")));
            const tool_parser = try self.tool(c.acc);
            try b.orAssign(&tool_choices, try b.rule(try std.fmt.allocPrint(self.tmp(), "tool-{d}", .{tool_index}), tool_parser, false));
        }
        return tool_choices;
    }

    /// Ports `build_json_tools_nested_keys` (chat-peg-parser.cpp:712 @cd959bae9):
    /// `{"function": {"name": ..., "arguments": ...}}`, keys in dot
    /// notation.
    fn buildJsonToolsNestedKeys(self: *ChatBuilder, tools: *const Value, effective_name_key: []const u8, effective_args_key: []const u8, call_id_key: []const u8, gen_call_id_key: []const u8) Error!ParserId {
        const b = &self.b;
        var tool_choices = try b.choiceEmpty();
        const name_spec = parseKeySpec(effective_name_key);
        const args_spec = parseKeySpec(effective_args_key);
        const nested_prefix = if (name_spec[0].len > 0) name_spec[0] else args_spec[0];
        const nested_name_field = if (name_spec[0].len > 0) name_spec[1] else effective_name_key;
        const nested_args_field = if (args_spec[0].len > 0) args_spec[1] else effective_args_key;

        for (tools.array.items, 0..) |*tool_def, tool_index| {
            var f = (try self.function(tool_def)) orelse continue;
            defer f.params.deinit(self.alloc());

            var nn = try self.keyChain(nested_name_field);
            try nn.add(try self.quotedName(f.name));
            const nested_name = nn.acc;
            var na = try self.keyChain(nested_args_field);
            try na.add(try self.schemaArgs(tool_index, &f.params));
            const nested_args = na.acc;

            var no = self.chain(try b.literal("{"));
            try no.add(try b.space());
            try no.add(nested_name);
            try no.add(try b.space());
            try no.add(try b.literal(","));
            try no.add(try b.space());
            try no.add(nested_args);
            try no.add(try b.space());
            try no.add(try b.literal("}"));
            const nested_object = no.acc;

            var body = self.chain(try self.toolOpen(try b.literal("{")));
            try body.add(try b.space());

            if (call_id_key.len > 0 and parseKeySpec(call_id_key)[0].len == 0) {
                const id_parser = try self.idQuoted(call_id_key);
                try body.add(try self.optionalThenComma(id_parser));
            }
            if (gen_call_id_key.len > 0 and parseKeySpec(gen_call_id_key)[0].len == 0) {
                const gen_id_parser = try self.idOrNumber(gen_call_id_key);
                try body.add(try self.optionalThenComma(gen_id_parser));
            }

            var nf = try self.keyChain(nested_prefix);
            try nf.add(nested_object);
            try body.add(nf.acc);
            try body.add(try b.space());
            try body.add(try self.toolClose(try b.literal("}")));
            try b.orAssign(&tool_choices, try b.rule(try std.fmt.allocPrint(self.tmp(), "tool-{d}", .{tool_index}), try self.tool(body.acc), false));
        }
        return tool_choices;
    }

    const Pair = struct { parser: ParserId, key: []const u8 };

    fn orderIndex(order: []const []const u8, key: []const u8) usize {
        for (order, 0..) |o, i| if (std.mem.eql(u8, o, key)) return i;
        return order.len;
    }

    /// Ports `build_json_tools_flat_keys` (chat-peg-parser.cpp:785 @cd959bae9):
    /// `{"name": ..., "arguments": ...}`, the fields in `parameters_order`,
    /// sorted by libc++'s `std::sort`.
    fn buildJsonToolsFlatKeys(self: *ChatBuilder, tools: *const Value, effective_name_key: []const u8, effective_args_key: []const u8, call_id_key: []const u8, gen_call_id_key: []const u8, parameters_order: []const []const u8, accept_openai_wrapper: bool) Error!ParserId {
        const b = &self.b;
        var tool_choices = try b.choiceEmpty();
        const name_key_parser = try self.quoted(effective_name_key);
        const args_key_parser = try self.quoted(effective_args_key);

        for (tools.array.items, 0..) |*tool_def, tool_index| {
            var f = (try self.function(tool_def)) orelse continue;
            defer f.params.deinit(self.alloc());

            var tn = self.chain(name_key_parser);
            try tn.add(try b.space());
            try tn.add(try b.literal(":"));
            try tn.add(try b.space());
            try tn.add(try self.quotedName(f.name));
            const tool_name_ = tn.acc;
            var ta = self.chain(args_key_parser);
            try ta.add(try b.space());
            try ta.add(try b.literal(":"));
            try ta.add(try b.space());
            try ta.add(try self.schemaArgs(tool_index, &f.params));
            const tool_args_ = ta.acc;

            var id_parser = try b.eps();
            if (call_id_key.len > 0) id_parser = try self.idOrNumber(call_id_key);
            var gen_id_parser = try b.eps();
            if (gen_call_id_key.len > 0) gen_id_parser = try self.idOrNumber(gen_call_id_key);

            var pairs: std.ArrayList(Pair) = .empty;
            try pairs.append(self.tmp(), .{ .parser = tool_name_, .key = effective_name_key });
            try pairs.append(self.tmp(), .{ .parser = tool_args_, .key = effective_args_key });
            if (call_id_key.len > 0) try pairs.append(self.tmp(), .{ .parser = try b.optional(id_parser), .key = call_id_key });
            if (gen_call_id_key.len > 0) try pairs.append(self.tmp(), .{ .parser = try b.optional(gen_id_parser), .key = gen_call_id_key });

            libcxx.sort(Pair, pairs.items, parameters_order, struct {
                fn less(order: []const []const u8, x: Pair, y: Pair) bool {
                    return orderIndex(order, x.key) < orderIndex(order, y.key);
                }
            }.less);

            var type_field = try b.eps();
            if (accept_openai_wrapper) {
                var t = self.chain(try b.literal("\"type\""));
                try t.add(try b.space());
                try t.add(try b.literal(":"));
                try t.add(try b.space());
                try t.add(try b.literal("\"function\""));
                try t.add(try b.space());
                try t.add(try b.literal(","));
                try t.add(try b.space());
                type_field = try b.optional(t.acc);
            }
            var ob = self.chain(try self.toolOpen(try b.literal("{")));
            try ob.add(try b.space());
            try ob.add(type_field);
            for (pairs.items, 0..) |pp, i| {
                try ob.add(pp.parser);
                if (i < pairs.items.len - 1) {
                    try ob.add(try b.space());
                    try ob.add(try b.literal(","));
                    try ob.add(try b.space());
                }
            }
            try ob.add(try b.space());
            try ob.add(try self.toolClose(try b.literal("}")));
            try b.orAssign(&tool_choices, try b.rule(try std.fmt.allocPrint(self.tmp(), "tool-{d}", .{tool_index}), try self.tool(ob.acc), false));
        }
        return tool_choices;
    }

    /// The trailing arguments of `standard_json_tools`, by their C++
    /// defaults.
    pub const JsonToolsOptions = struct {
        name_key: []const u8 = "",
        args_key: []const u8 = "",
        array_wrapped: bool = false,
        function_is_key: bool = false,
        call_id_key: []const u8 = "",
        gen_call_id_key: []const u8 = "",
        parameters_order: []const []const u8 = &.{},
        accept_openai_wrapper: bool = false,
    };

    /// Ports `standard_json_tools` (chat-peg-parser.cpp:912 @cd959bae9): JSON
    /// tool calls between `section_start` and `section_end`, in whichever
    /// of the three layouts the keys describe.
    pub fn standardJsonTools(self: *ChatBuilder, section_start: []const u8, section_end: []const u8, tools: *const Value, parallel_tool_calls: bool, force_tool_calls: bool, o: JsonToolsOptions) Error!ParserId {
        const b = &self.b;
        if (!tools.isArray() or tools.empty()) return b.eps();

        const effective_name_key = if (o.name_key.len == 0) "name" else o.name_key;
        const effective_args_key = if (o.args_key.len == 0) "arguments" else o.args_key;

        var tool_choices = try b.eps();
        if (o.function_is_key) {
            tool_choices = try self.buildJsonToolsFunctionIsKey(tools, o.args_key, effective_args_key, o.call_id_key, o.gen_call_id_key);
        } else {
            const name_spec = parseKeySpec(effective_name_key);
            const args_spec = parseKeySpec(effective_args_key);
            if (name_spec[0].len > 0 or args_spec[0].len > 0) {
                tool_choices = try self.buildJsonToolsNestedKeys(tools, effective_name_key, effective_args_key, o.call_id_key, o.gen_call_id_key);
            } else {
                tool_choices = try self.buildJsonToolsFlatKeys(tools, effective_name_key, effective_args_key, o.call_id_key, o.gen_call_id_key, o.parameters_order, o.accept_openai_wrapper);
            }
        }

        var tool_calls = tool_choices;
        if (parallel_tool_calls) {
            var more = self.chain(try b.space());
            try more.add(try b.literal(","));
            try more.add(try b.space());
            try more.add(tool_choices);
            tool_calls = try b.plus(tool_calls, try b.zeroOrMore(more.acc));
        }
        if (o.array_wrapped) {
            var w = self.chain(try b.literal("["));
            try w.add(try b.space());
            try w.add(tool_calls);
            try w.add(try b.space());
            try w.add(try b.literal("]"));
            tool_calls = w.acc;
        }
        var sec = self.chain(try b.literal(section_start));
        try sec.add(try b.space());
        try sec.add(tool_calls);
        try sec.add(try b.space());
        try sec.add(try b.literal(section_end));
        const section = try b.triggerRule("tool-call", sec.acc);
        return if (force_tool_calls) section else b.optional(section);
    }

    /// Ports `build` (peg-parser.cpp:1202 @cd959bae9), through the base.
    pub fn build(self: *ChatBuilder) Error!peg.Arena {
        return self.b.build();
    }
};

/// Ports `build_chat_peg_parser` (chat-peg-parser.h:182 @cd959bae9).
///
/// Return: the arena, owned by the caller.
pub fn buildChatPegParser(gpa: Allocator, ctx: anytype, comptime f: fn (@TypeOf(ctx), *ChatBuilder) Error!ParserId) Error!peg.Arena {
    var cb = try ChatBuilder.init(gpa);
    defer cb.deinit();
    cb.b.setRoot(try f(ctx, &cb));
    return cb.build();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "Python quotes and literals become JSON" {
    const a = std.testing.allocator;
    const out = try normalizeQuotesToJson(a, "{'code': 'print(\\'hi\\')', 'ok': Tru, 'n': None, 'x': Nonsense}");
    defer a.free(out);
    try std.testing.expectEqualStrings("{\"code\": \"print('hi')\", \"ok\": tru, \"n\": null, \"x\": Nonsense}", out);
}

test "a tagged tool call maps to a message" {
    const a = std.testing.allocator;
    var arena = try buildChatPegParser(a, {}, struct {
        fn f(_: void, p: *ChatBuilder) Error!ParserId {
            const b = &p.b;
            const content = try p.content(try b.until("<tool_call>"));
            var tools = try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]");
            defer tools.deinit(a);
            const calls = try p.standardConstructedTools(&.{}, &tools, false, false);
            return b.plus(try b.plus(content, calls), try b.end());
        }
    }.f);
    defer arena.deinit();
    var ctx = try peg.ParseContext.init(a, "Hi<tool_call><function=get><param=city>Paris</param></function></tool_call>", peg.Flags.none);
    defer ctx.deinit();
    const r = try arena.parse(&ctx, 0);
    try std.testing.expect(r.success());
    var m: Msg = .{};
    defer m.deinit(a);
    var mapper = Mapper.init(a, &m, .base);
    defer mapper.deinit();
    try mapper.fromAst(&ctx.ast, &r);
    try std.testing.expectEqualStrings("Hi", m.content);
    try std.testing.expectEqual(@as(usize, 1), m.tool_calls.items.len);
    try std.testing.expectEqualStrings("{\"city\":Paris}", m.tool_calls.items[0].arguments);
}
