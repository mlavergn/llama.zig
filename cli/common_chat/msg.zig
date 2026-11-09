//! The chat data model: a message, its content parts and tool calls, the
//! difference between two partial parses of one, and the OpenAI-compatible
//! JSON both directions.
//!
//! # Provenance
//!
//! Ports the data half of `llama.cpp/common/chat.h` and `llama.cpp/common/chat.cpp`
//! (llama.cpp v0.8.0, `8642541b2`): `common_chat_tool_call`,
//! `common_chat_msg_content_part`, `common_chat_msg` and its methods,
//! `common_chat_msg_diff::compute_diffs`, the role, span and delimiter
//! types, `common_chat_tool`, the tool-choice, format, continuation and
//! reasoning-format enums with their names, and the `*_oaicompat`
//! conversions. Rendering a template and parsing a model's output -- the
//! rest of `chat.cpp`, over Jinja and the PEG parser -- are Stage 9 step 4.
//! Each declaration names what it ports and the line it began at.
//!
//! # Ownership
//!
//! A `Msg` owns every string and list in it, allocated by the allocator its
//! constructor took; `deinit` frees them. Functions returning a `json.Value`
//! return one the caller owns.
//!
//! # Errors
//!
//! A throw is an `exception.Error` with `exception.what()` holding the
//! text the C++ would have put in `e.what()`.

const std = @import("std");
const json = @import("json");
const exception = @import("exception.zig");

const Error = exception.Error;
const Value = json.Value;
const throw = exception.throw;

/// `LOG_WRN` and `LOG_ERR`, wherever the host sends them: the server sets
/// this to its `common_log`. Null drops them.
pub var log_sink: ?*const fn (level: Level, text: []const u8) void = null;

/// The two levels this module logs at.
pub const Level = enum { warn, err };

fn logf(level: Level, comptime fmt: []const u8, args: anytype) void {
    const sink = log_sink orelse return;
    var buf: [4096]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..];
    sink(level, text);
}

/// Ports `common_chat_tool_call` (chat.h:35 @8642541b2).
pub const ToolCall = struct {
    name: []const u8 = "",
    arguments: []const u8 = "",
    id: []const u8 = "",

    /// Ports its `operator`== (chat.h:40 @8642541b2).
    pub fn eql(a: ToolCall, b: ToolCall) bool {
        return std.mem.eql(u8, a.name, b.name) and std.mem.eql(u8, a.arguments, b.arguments) and std.mem.eql(u8, a.id, b.id);
    }

    /// A copy whose strings `gpa` owns.
    pub fn clone(self: ToolCall, gpa: std.mem.Allocator) !ToolCall {
        const name = try gpa.dupe(u8, self.name);
        errdefer gpa.free(name);
        const arguments = try gpa.dupe(u8, self.arguments);
        errdefer gpa.free(arguments);
        return .{ .name = name, .arguments = arguments, .id = try gpa.dupe(u8, self.id) };
    }

    /// Frees what `clone` allocated.
    pub fn deinit(self: *ToolCall, gpa: std.mem.Allocator) void {
        gpa.free(self.name);
        gpa.free(self.arguments);
        gpa.free(self.id);
        self.* = .{};
    }
};

/// Ports `common_chat_msg_content_part` (chat.h:45 @8642541b2).
pub const ContentPart = struct {
    type: []const u8 = "",
    text: []const u8 = "",
    name: []const u8 = "", // tool_reference: name of a tool from the request tools

    /// Ports its `operator`== (chat.h:55 @8642541b2).
    pub fn eql(a: ContentPart, b: ContentPart) bool {
        return std.mem.eql(u8, a.type, b.type) and std.mem.eql(u8, a.text, b.text) and std.mem.eql(u8, a.name, b.name);
    }
};

/// Ports `common_chat_msg` (chat.h:90 @8642541b2).
pub const Msg = struct {
    role: []const u8 = "",
    content: []const u8 = "",
    content_parts: std.ArrayList(ContentPart) = .empty,
    tool_calls: std.ArrayList(ToolCall) = .empty,
    reasoning_content: []const u8 = "",
    tool_name: []const u8 = "",
    tool_call_id: []const u8 = "",
    /// non-OpenAI message keys as a JSON object string, only for templates that read them (DeepSeek V4.1)
    extra_json: []const u8 = "",

    /// Frees every string and list; the message is empty afterwards.
    ///
    /// Parameters:
    /// - `gpa`: the allocator that built the message.
    pub fn deinit(self: *Msg, gpa: std.mem.Allocator) void {
        gpa.free(self.role);
        gpa.free(self.content);
        for (self.content_parts.items) |p| {
            gpa.free(p.type);
            gpa.free(p.text);
            gpa.free(p.name);
        }
        self.content_parts.deinit(gpa);
        for (self.tool_calls.items) |*tc| tc.deinit(gpa);
        self.tool_calls.deinit(gpa);
        gpa.free(self.reasoning_content);
        gpa.free(self.tool_name);
        gpa.free(self.tool_call_id);
        gpa.free(self.extra_json);
        self.* = .{};
    }

    /// A deep copy, as the C++ copy constructor makes one.
    ///
    /// Return: owned by the caller; free with `deinit(gpa)`.
    pub fn clone(self: *const Msg, gpa: std.mem.Allocator) !Msg {
        var out: Msg = .{};
        errdefer out.deinit(gpa);
        out.role = try gpa.dupe(u8, self.role);
        out.content = try gpa.dupe(u8, self.content);
        for (self.content_parts.items) |p| try out.addPartNamed(gpa, p.type, p.text, p.name);
        for (self.tool_calls.items) |tc| try out.addToolCall(gpa, tc);
        out.reasoning_content = try gpa.dupe(u8, self.reasoning_content);
        out.tool_name = try gpa.dupe(u8, self.tool_name);
        out.tool_call_id = try gpa.dupe(u8, self.tool_call_id);
        out.extra_json = try gpa.dupe(u8, self.extra_json);
        return out;
    }

    /// Appends a content part, copying both strings.
    pub fn addPart(self: *Msg, gpa: std.mem.Allocator, type_: []const u8, text: []const u8) !void {
        return self.addPartNamed(gpa, type_, text, "");
    }

    /// Appends a content part, copying its strings; `name` is a
    /// `tool_reference`'s.
    pub fn addPartNamed(self: *Msg, gpa: std.mem.Allocator, type_: []const u8, text: []const u8, name: []const u8) !void {
        const t = try gpa.dupe(u8, type_);
        errdefer gpa.free(t);
        const x = try gpa.dupe(u8, text);
        errdefer gpa.free(x);
        const n = try gpa.dupe(u8, name);
        errdefer gpa.free(n);
        try self.content_parts.append(gpa, .{ .type = t, .text = x, .name = n });
    }

    /// Appends a tool call, copying its strings.
    pub fn addToolCall(self: *Msg, gpa: std.mem.Allocator, tc: ToolCall) !void {
        var copy = try tc.clone(gpa);
        errdefer copy.deinit(gpa);
        try self.tool_calls.append(gpa, copy);
    }

    /// Replaces one of the message's strings with a copy of `s`.
    pub fn setField(gpa: std.mem.Allocator, field: *[]const u8, s: []const u8) !void {
        const copy = try gpa.dupe(u8, s);
        gpa.free(field.*);
        field.* = copy;
    }

    /// Ports `common_chat_msg`'s `empty` (chat.h:106 @8642541b2).
    pub fn empty(self: *const Msg) bool {
        return self.content.len == 0 and self.content_parts.items.len == 0 and self.tool_calls.items.len == 0 and
            self.reasoning_content.len == 0 and self.tool_name.len == 0 and self.tool_call_id.len == 0;
    }

    /// Ports `common_chat_msg`'s `contains_media` (chat.h:110 @8642541b2).
    pub fn containsMedia(self: *const Msg) bool {
        for (self.content_parts.items) |p| {
            if (std.mem.eql(u8, p.type, "media_marker")) return true;
        }
        return false;
    }

    /// Ports `common_chat_msg`'s `set_tool_call_ids` (chat.h:119 @8642541b2):
    /// gives every tool call the id cached for its position, drawing a new
    /// one for a call that has none when the cache is short.
    ///
    /// Parameters:
    /// - `ids_cache`: the ids handed out so far; grows; strings owned by `gpa`.
    /// - `ctx`, `gen`: draws a fresh id, owned by `gpa`.
    pub fn setToolCallIds(
        self: *Msg,
        gpa: std.mem.Allocator,
        ids_cache: *std.ArrayList([]const u8),
        ctx: anytype,
        comptime gen: fn (@TypeOf(ctx), std.mem.Allocator) anyerror![]const u8,
    ) !void {
        for (self.tool_calls.items, 0..) |*tc, i| {
            if (ids_cache.items.len <= i) {
                const id = if (tc.id.len == 0) try gen(ctx, gpa) else try gpa.dupe(u8, tc.id);
                errdefer gpa.free(id);
                try ids_cache.append(gpa, id);
            }
            try setField(gpa, &tc.id, ids_cache.items[i]);
        }
    }

    /// Ports `common_chat_msg`'s `operator`== (chat.h:133 @8642541b2).
    pub fn eql(a: *const Msg, b: *const Msg) bool {
        if (!std.mem.eql(u8, a.role, b.role) or !std.mem.eql(u8, a.content, b.content)) return false;
        if (a.content_parts.items.len != b.content_parts.items.len) return false;
        for (a.content_parts.items, b.content_parts.items) |x, y| if (!x.eql(y)) return false;
        if (a.tool_calls.items.len != b.tool_calls.items.len) return false;
        for (a.tool_calls.items, b.tool_calls.items) |x, y| if (!x.eql(y)) return false;
        return std.mem.eql(u8, a.reasoning_content, b.reasoning_content) and
            std.mem.eql(u8, a.tool_name, b.tool_name) and std.mem.eql(u8, a.tool_call_id, b.tool_call_id);
    }

    /// Ports `common_chat_msg`'s `render_content` (chat.cpp:77 @8642541b2): the
    /// content, or the text parts joined by `delimiter`.
    ///
    /// Return: owned by the caller.
    pub fn renderContent(self: *const Msg, gpa: std.mem.Allocator, delimiter: []const u8) Error![]u8 {
        if (self.content.len != 0 and self.content_parts.items.len != 0) {
            return throw(error.RuntimeError, "Cannot specify both content and content_parts", .{});
        }
        if (self.content.len != 0) return gpa.dupe(u8, self.content);

        var text: std.ArrayList(u8) = .empty;
        errdefer text.deinit(gpa);
        for (self.content_parts.items) |part| {
            if (std.mem.eql(u8, part.type, "text")) {
                if (text.items.len != 0) try text.appendSlice(gpa, delimiter);
                try text.appendSlice(gpa, part.text);
            }
        }
        return text.toOwnedSlice(gpa);
    }

    /// Ports `common_chat_msg`'s `to_json_oaicompat` (chat.cpp:156 @8642541b2).
    ///
    /// Return: owned by the caller.
    pub fn toJsonOaicompat(self: *const Msg, gpa: std.mem.Allocator, concat_typed_text: bool) Error!Value {
        if (self.content.len != 0 and self.content_parts.items.len != 0) {
            return throw(error.RuntimeError, "Cannot specify both content and content_parts", .{});
        }
        var jmsg = Value.emptyObject();
        errdefer jmsg.deinit(gpa);
        try jmsg.set(gpa, "role", try Value.from(gpa, self.role));
        if (self.content.len != 0) {
            try jmsg.set(gpa, "content", try Value.from(gpa, self.content));
        } else if (self.content_parts.items.len != 0) {
            if (concat_typed_text or self.containsMedia()) {
                var text: std.ArrayList(u8) = .empty;
                defer text.deinit(gpa);
                var last_was_media_marker = false;
                // join parts with newline, do not add newline before or after media markers
                for (self.content_parts.items) |part| {
                    var add_new_line = true;
                    if (std.mem.eql(u8, part.type, "text")) {
                        add_new_line = !last_was_media_marker and text.items.len != 0;
                        last_was_media_marker = false;
                    } else if (std.mem.eql(u8, part.type, "media_marker")) {
                        add_new_line = false;
                        last_was_media_marker = true;
                    } else {
                        logf(.warn, "Ignoring content part type: {s}\n", .{part.type});
                        continue;
                    }
                    if (add_new_line) try text.append(gpa, '\n');
                    try text.appendSlice(gpa, part.text);
                }
                try jmsg.set(gpa, "content", try Value.from(gpa, text.items));
            } else {
                var parts = Value.emptyArray();
                errdefer parts.deinit(gpa);
                for (self.content_parts.items) |part| {
                    var p = Value.emptyObject();
                    errdefer p.deinit(gpa);
                    try p.set(gpa, "type", try Value.from(gpa, part.type));
                    if (std.mem.eql(u8, part.type, "tool_reference")) {
                        try p.set(gpa, "name", try Value.from(gpa, part.name));
                    } else {
                        try p.set(gpa, "text", try Value.from(gpa, part.text));
                    }
                    try parts.pushBack(gpa, p);
                }
                try jmsg.set(gpa, "content", parts);
            }
        } else {
            try jmsg.set(gpa, "content", try Value.from(gpa, ""));
        }
        if (self.reasoning_content.len != 0) try jmsg.set(gpa, "reasoning_content", try Value.from(gpa, self.reasoning_content));
        if (self.tool_name.len != 0) try jmsg.set(gpa, "name", try Value.from(gpa, self.tool_name));
        if (self.tool_call_id.len != 0) try jmsg.set(gpa, "tool_call_id", try Value.from(gpa, self.tool_call_id));
        if (self.tool_calls.items.len != 0) {
            var jtool_calls = Value.emptyArray();
            errdefer jtool_calls.deinit(gpa);
            for (self.tool_calls.items) |tool_call| {
                var func = Value.emptyObject();
                errdefer func.deinit(gpa);
                try func.set(gpa, "name", try Value.from(gpa, tool_call.name));
                try func.set(gpa, "arguments", try Value.from(gpa, tool_call.arguments));
                var tc = Value.emptyObject();
                errdefer tc.deinit(gpa);
                try tc.set(gpa, "type", try Value.from(gpa, "function"));
                try tc.set(gpa, "function", func);
                if (tool_call.id.len != 0) try tc.set(gpa, "id", try Value.from(gpa, tool_call.id));
                try jtool_calls.pushBack(gpa, tc);
            }
            try jmsg.set(gpa, "tool_calls", jtool_calls);
        }
        return jmsg;
    }
};

/// Ports `string_diff` (chat.cpp:58 @8642541b2): what `current` adds to
/// `last`.
///
/// Return: a slice of `current`.
pub fn stringDiff(last: []const u8, current: []const u8) Error![]const u8 {
    if (last.len == 0) return current;
    if (!std.mem.startsWith(u8, current, last)) {
        if (std.mem.startsWith(u8, last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        return throw(error.RuntimeError, "Invalid diff: '{s}' not found at start of '{s}'", .{ last, current });
    }
    return current[last.len..];
}

/// `std::string::npos`, `tool_call_index`'s "no tool call".
pub const npos: usize = std.math.maxInt(usize);

/// Ports `common_chat_msg_diff` (chat.h:142 @8642541b2). The strings are
/// borrowed from the messages `computeDiffs` compared.
pub const MsgDiff = struct {
    reasoning_content_delta: []const u8 = "",
    content_delta: []const u8 = "",
    tool_call_index: usize = npos,
    tool_call_delta: ToolCall = .{},

    /// Ports its `operator`== (chat.h:151 @8642541b2), which leaves out the
    /// reasoning delta.
    pub fn eql(a: MsgDiff, b: MsgDiff) bool {
        return std.mem.eql(u8, a.content_delta, b.content_delta) and a.tool_call_index == b.tool_call_index and
            a.tool_call_delta.eql(b.tool_call_delta);
    }
};

/// Ports `common_chat_msg_diff`'s `compute_diffs` (chat.cpp:243 @8642541b2).
///
/// Return: owned by the caller (the list only; the strings are borrowed
/// from `msg_prv` and `msg_new`).
pub fn computeDiffs(gpa: std.mem.Allocator, msg_prv: *const Msg, msg_new: *const Msg) Error!std.ArrayList(MsgDiff) {
    var diffs: std.ArrayList(MsgDiff) = .empty;
    errdefer diffs.deinit(gpa);
    const prv_calls = msg_prv.tool_calls.items;
    const new_calls = msg_new.tool_calls.items;

    // TODO: these can become expensive for long messages - how to optimize?
    if (!std.mem.eql(u8, msg_prv.reasoning_content, msg_new.reasoning_content)) {
        try diffs.append(gpa, .{ .reasoning_content_delta = try stringDiff(msg_prv.reasoning_content, msg_new.reasoning_content) });
    }
    if (!std.mem.eql(u8, msg_prv.content, msg_new.content)) {
        try diffs.append(gpa, .{ .content_delta = try stringDiff(msg_prv.content, msg_new.content) });
    }

    if (new_calls.len < prv_calls.len) {
        var err: std.ArrayList(u8) = .empty;
        defer err.deinit(gpa);
        try err.print(gpa, "Invalid diff: now finding less tool calls!\n", .{});
        try err.print(gpa, "  Previous ({d}):\n", .{prv_calls.len});
        for (prv_calls) |tc| try err.print(gpa, "    - name: '{s}', args: '{s}'\n", .{ tc.name, tc.arguments });
        try err.print(gpa, "  Current ({d}):\n", .{new_calls.len});
        for (new_calls) |tc| try err.print(gpa, "    - name: '{s}', args: '{s}'\n", .{ tc.name, tc.arguments });
        try err.print(gpa, "  Current msg text content:\n{s}\n", .{msg_new.content});
        return throw(error.RuntimeError, "{s}", .{err.items});
    }

    if (prv_calls.len != 0) {
        const idx = prv_calls.len - 1;
        const pref = prv_calls[idx];
        const newf = new_calls[idx];
        // Allow tool name to change during incremental parsing:
        // - empty -> non-empty (initial discovery)
        // - prefix -> longer string (name grows as more input is parsed)
        if (!std.mem.eql(u8, pref.name, newf.name) and pref.name.len != 0 and newf.name.len != 0) {
            // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
            const is_prefix = std.mem.startsWith(u8, newf.name, pref.name);
            if (!is_prefix) {
                logf(.err, "Tool call mismatch: prev='{s}' new='{s}'\n", .{ pref.name, newf.name });
                return throw(error.RuntimeError, "Invalid diff: tool call mismatch!", .{});
            }
        }
        const args_diff = try stringDiff(pref.arguments, newf.arguments);
        const id_or_name_changed = !std.mem.eql(u8, pref.id, newf.id) or !std.mem.eql(u8, pref.name, newf.name);
        if (args_diff.len != 0 or id_or_name_changed) {
            var diff: MsgDiff = .{ .tool_call_index = idx };
            if (id_or_name_changed) {
                diff.tool_call_delta.id = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
            try diffs.append(gpa, diff);
        }
    }
    for (prv_calls.len..new_calls.len) |idx| {
        try diffs.append(gpa, .{ .tool_call_index = idx, .tool_call_delta = new_calls[idx] });
    }
    return diffs;
}

/// Ports `common_chat_role` (chat.h:157 @8642541b2).
pub const Role = enum(c_int) {
    unknown,
    system,
    assistant,
    user,
    tool,

    /// Ports `common_chat_role_from_string` (chat.cpp:97 @8642541b2).
    pub fn fromString(role: []const u8) Role {
        if (std.mem.eql(u8, role, "system")) return .system;
        if (std.mem.eql(u8, role, "assistant")) return .assistant;
        if (std.mem.eql(u8, role, "user")) return .user;
        if (std.mem.eql(u8, role, "tool")) return .tool;
        return .unknown;
    }

    /// Ports `common_chat_role_to_string` (chat.cpp:105 @8642541b2).
    pub fn toString(role: Role) []const u8 {
        return switch (role) {
            .system => "system",
            .assistant => "assistant",
            .user => "user",
            .tool => "tool",
            .unknown => "",
        };
    }
};

/// Ports `common_chat_msg_span` (chat.h:168 @8642541b2).
pub const MsgSpan = struct {
    role: Role = .unknown,
    pos: usize = 0,
    len: usize = 0,

    /// Ports `valid` (chat.h:173 @8642541b2).
    pub fn valid(self: MsgSpan) bool {
        return self.role != .unknown;
    }
};

/// Ports `common_chat_msg_spans` (chat.h:178 @8642541b2).
pub const MsgSpans = struct {
    spans: std.ArrayList(MsgSpan) = .empty,

    pub fn deinit(self: *MsgSpans, gpa: std.mem.Allocator) void {
        self.spans.deinit(gpa);
    }

    /// Ports `add` (chat.h:181 @8642541b2).
    pub fn add(self: *MsgSpans, gpa: std.mem.Allocator, role: Role, pos: usize, len: usize) !void {
        try self.spans.append(gpa, .{ .role = role, .pos = pos, .len = len });
    }

    /// Ports `is_user_start` (chat.h:185 @8642541b2).
    pub fn isUserStart(self: *const MsgSpans, pos: i32) bool {
        for (self.spans.items) |s| {
            if (s.role == .user and pos == @as(i32, @truncate(@as(isize, @bitCast(s.pos))))) return true;
        }
        return false;
    }

    /// Ports `last_user_message_pos` (chat.h:194 @8642541b2).
    pub fn lastUserMessagePos(self: *const MsgSpans) i32 {
        var i = self.spans.items.len;
        while (i > 0) {
            i -= 1;
            const s = self.spans.items[i];
            if (s.role == .user) return @truncate(@as(isize, @bitCast(s.pos)));
        }
        return -1;
    }
};

/// Ports `common_chat_msg_delimiter` (chat.h:204 @8642541b2).
pub const MsgDelimiter = struct {
    role: Role = .unknown,
    delimiter: []const u8 = "",
    tokens: std.ArrayList(i32) = .empty,
};

/// Ports `common_chat_msg_delimiters` (chat.h:210 @8642541b2). Owns its
/// strings and token lists.
pub const MsgDelimiters = struct {
    delimiters: std.ArrayList(MsgDelimiter) = .empty,

    pub fn deinit(self: *MsgDelimiters, gpa: std.mem.Allocator) void {
        for (self.delimiters.items) |*d| {
            gpa.free(d.delimiter);
            d.tokens.deinit(gpa);
        }
        self.delimiters.deinit(gpa);
    }

    /// Ports `add` (chat.h:216 @8642541b2).
    pub fn add(self: *MsgDelimiters, gpa: std.mem.Allocator, role: Role, delimiter: []const u8) !void {
        const d = try gpa.dupe(u8, delimiter);
        errdefer gpa.free(d);
        try self.delimiters.append(gpa, .{ .role = role, .delimiter = d });
    }

    /// Ports `common_chat_msg_delimiters`'s `tokenize` (chat.cpp:116 @8642541b2):
    /// each delimiter through `common_tokenize(vocab, delimiter, false, true)`,
    /// which the caller supplies -- this module links no libllama.
    pub fn tokenize(
        self: *MsgDelimiters,
        gpa: std.mem.Allocator,
        ctx: anytype,
        comptime tok: fn (@TypeOf(ctx), std.mem.Allocator, []const u8) anyerror!std.ArrayList(i32),
    ) !void {
        for (self.delimiters.items) |*d| {
            const t = try tok(ctx, gpa, d.delimiter);
            d.tokens.deinit(gpa);
            d.tokens = t;
        }
    }

    /// Ports `common_chat_msg_delimiters`'s `split` (chat.cpp:122 @8642541b2):
    /// the tokens cut into message spans at each delimiter, jumping over the
    /// regions `skips` names (start, length), which must be sorted by start.
    ///
    /// Return: owned by the caller.
    pub fn split(self: *const MsgDelimiters, gpa: std.mem.Allocator, tokens: []const i32, skips: []const [2]usize) !MsgSpans {
        const Match = struct { role: Role, pos: usize };
        var matches: std.ArrayList(Match) = .empty;
        defer matches.deinit(gpa);

        var skip: usize = 0;
        var i: usize = 0;
        while (i < tokens.len) {
            if (skip < skips.len and i == skips[skip][0]) {
                i += skips[skip][1];
                skip += 1;
                continue;
            }
            for (self.delimiters.items) |d| {
                if (i + d.tokens.items.len > tokens.len) continue;
                if (std.mem.eql(i32, d.tokens.items, tokens[i .. i + d.tokens.items.len])) {
                    try matches.append(gpa, .{ .role = d.role, .pos = i });
                    break;
                }
            }
            i += 1;
        }

        try matches.append(gpa, .{ .role = .unknown, .pos = tokens.len });

        var spans: MsgSpans = .{};
        errdefer spans.deinit(gpa);
        var k: usize = 0;
        while (k + 1 < matches.items.len) : (k += 1) {
            const curr = matches.items[k];
            const next = matches.items[k + 1];
            try spans.add(gpa, curr.role, curr.pos, next.pos -% curr.pos);
        }
        return spans;
    }
};

/// Ports `common_chat_tool` (chat.h:226 @8642541b2).
pub const Tool = struct {
    name: []const u8 = "",
    description: []const u8 = "",
    parameters: []const u8 = "",
    /// withheld from the system prompt, a tool_reference reveals it later in the conversation
    defer_loading: bool = false,

    pub fn deinit(self: *Tool, gpa: std.mem.Allocator) void {
        gpa.free(self.name);
        gpa.free(self.description);
        gpa.free(self.parameters);
        self.* = .{};
    }
};

/// Ports `common_chat_tool_choice` (chat.h:235 @8642541b2).
pub const ToolChoice = enum(c_int) { auto, required, none };

/// Ports `common_chat_tool_choice_parse_oaicompat` (chat.cpp:322 @8642541b2).
pub fn toolChoiceParseOaicompat(tool_choice: []const u8) Error!ToolChoice {
    if (std.mem.eql(u8, tool_choice, "auto")) return .auto;
    if (std.mem.eql(u8, tool_choice, "none")) return .none;
    if (std.mem.eql(u8, tool_choice, "required")) return .required;
    return throw(error.InvalidArgument, "Invalid tool_choice: {s}", .{tool_choice});
}

/// Ports `common_chat_format` (chat.h:241 @8642541b2).
pub const Format = enum(c_int) {
    content_only,
    peg_simple,
    peg_native,
    peg_gemma4,
    peg_minimax_m3,

    /// Ports `common_chat_format_name` (chat.cpp:994 @8642541b2). Every
    /// value has a name, so the C++'s `default:` throw is unreachable.
    pub fn name(self: Format) []const u8 {
        return switch (self) {
            .content_only => "Content-only",
            .peg_simple => "peg-simple",
            .peg_native => "peg-native",
            .peg_gemma4 => "peg-gemma4",
            .peg_minimax_m3 => "peg-minimax-m3",
        };
    }
};

/// Ports `common_chat_continuation` (chat.h:255 @8642541b2).
pub const Continuation = enum(c_int) { none, auto, reasoning, content };

/// Ports `common_chat_continuation_parse` (chat.cpp:695 @8642541b2): null
/// and false are none, true is auto; anything but those and the two
/// field names throws `invalid_argument`.
pub fn continuationParse(value: *const Value) Error!Continuation {
    if (value.* == .null) return .none;
    if (value.* == .boolean) return if (value.boolean) .auto else .none;
    if (value.* == .string) {
        if (std.mem.eql(u8, value.string, "reasoning_content")) return .reasoning;
        if (std.mem.eql(u8, value.string, "content")) return .content;
    }
    return throw(error.InvalidArgument, "Invalid continue_final_message: expected a boolean, \"content\" or \"reasoning_content\"", .{});
}

/// Ports `common_reasoning_format` (common/common.h:428 @8642541b2).
pub const ReasoningFormat = enum(c_int) {
    none,
    auto,
    deepseek_legacy,
    deepseek,

    /// Ports `common_reasoning_format_name` (chat.cpp:1011 @8642541b2).
    pub fn name(self: ReasoningFormat) []const u8 {
        return switch (self) {
            .none => "none",
            .auto => "auto",
            .deepseek => "deepseek",
            .deepseek_legacy => "deepseek-legacy",
        };
    }

    /// Ports `common_reasoning_format_from_name` (chat.cpp:1026 @8642541b2).
    pub fn fromName(format: []const u8) Error!ReasoningFormat {
        if (std.mem.eql(u8, format, "none")) return .none;
        if (std.mem.eql(u8, format, "auto")) return .auto;
        if (std.mem.eql(u8, format, "deepseek")) return .deepseek;
        if (std.mem.eql(u8, format, "deepseek-legacy")) return .deepseek_legacy;
        return throw(error.RuntimeError, "Unknown reasoning format: {s}", .{format});
    }
};

/// `get<std::string>()` into a fresh copy.
fn getString(gpa: std.mem.Allocator, v: *const Value) Error![]const u8 {
    return gpa.dupe(u8, try v.get([]const u8));
}

/// `dump()` at the default indent, -1.
fn dumpCompact(gpa: std.mem.Allocator, v: *const Value) Error![]u8 {
    return json.dump(gpa, v, -1);
}

/// `j == "s"`: `common_json`'s comparison with a string value.
fn isString(v: *const Value, s: []const u8) bool {
    return v.* == .string and std.mem.eql(u8, v.string, s);
}

/// Ports `common_chat_msgs_parse_oaicompat` (chat.cpp:350 @8642541b2).
///
/// Return: owned by the caller; free each message with `deinit(gpa)`.
pub fn msgsParseOaicompat(gpa: std.mem.Allocator, messages: *Value) Error!std.ArrayList(Msg) {
    var msgs: std.ArrayList(Msg) = .empty;
    errdefer {
        for (msgs.items) |*m| m.deinit(gpa);
        msgs.deinit(gpa);
    }
    msgsParseInner(gpa, messages, &msgs) catch |e| {
        // @ngxson : disable otherwise it's bloating the API response
        return exception.rethrow(error.RuntimeError, e, "Failed to parse messages: ", "");
    };
    return msgs;
}

fn msgsParseInner(gpa: std.mem.Allocator, messages: *Value, msgs: *std.ArrayList(Msg)) Error!void {
    if (!messages.isArray()) {
        const d = try dumpCompact(gpa, messages);
        defer gpa.free(d);
        return throw(error.InvalidArgument, "Expected 'messages' to be an array, got {s}", .{d});
    }

    var it = messages.items();
    while (it.next()) |item| {
        const message = item.value;
        if (!message.isObject()) {
            const d = try dumpCompact(gpa, message);
            defer gpa.free(d);
            return throw(error.InvalidArgument, "Expected 'message' to be an object, got {s}", .{d});
        }

        var msg: Msg = .{};
        errdefer msg.deinit(gpa);
        if (!message.contains("role")) {
            const d = try dumpCompact(gpa, message);
            defer gpa.free(d);
            return throw(error.InvalidArgument, "Missing 'role' in message: {s}", .{d});
        }
        msg.role = try getString(gpa, try message.at("role"));

        const has_content = message.contains("content");
        const has_tool_calls = message.contains("tool_calls");
        if (has_content) {
            const content = try message.at("content");
            if (content.isString()) {
                try Msg.setField(gpa, &msg.content, content.string);
            } else if (content.isArray()) {
                var pit = content.items();
                while (pit.next()) |pitem| {
                    const part = pitem.value;
                    if (!part.contains("type")) {
                        const d = try dumpCompact(gpa, part);
                        defer gpa.free(d);
                        return throw(error.InvalidArgument, "Missing content part type: {s}", .{d});
                    }
                    const type_ = try part.at("type");
                    const t = try type_.get([]const u8);
                    if (isString(type_, "text") or isString(type_, "media_marker")) {
                        try msg.addPart(gpa, t, try (try part.at("text")).get([]const u8));
                    } else if (isString(type_, "tool_reference")) {
                        // names a request tool, the template renders its definition
                        try msg.addPartNamed(gpa, t, "", try (try part.at("name")).get([]const u8));
                    } else {
                        const d = try dumpCompact(gpa, type_);
                        defer gpa.free(d);
                        return throw(error.InvalidArgument, "Unsupported content part type: {s}", .{d});
                    }
                }
            } else if (!content.isNull()) {
                const d = try dumpCompact(gpa, content);
                defer gpa.free(d);
                return throw(error.InvalidArgument, "Invalid 'content' type: expected string or array, got {s} (ref: https://github.com/ggml-org/llama.cpp/issues/8367)", .{d});
            }
        }
        if (has_tool_calls) {
            var tit = (try message.at("tool_calls")).items();
            while (tit.next()) |titem| {
                const tool_call = titem.value;
                if (!tool_call.contains("type")) {
                    const d = try dumpCompact(gpa, tool_call);
                    defer gpa.free(d);
                    return throw(error.InvalidArgument, "Missing tool call type: {s}", .{d});
                }
                const type_ = try tool_call.at("type");
                if (!isString(type_, "function")) {
                    const d = try dumpCompact(gpa, tool_call);
                    defer gpa.free(d);
                    return throw(error.InvalidArgument, "Unsupported tool call type: {s}", .{d});
                }
                if (!tool_call.contains("function")) {
                    const d = try dumpCompact(gpa, tool_call);
                    defer gpa.free(d);
                    return throw(error.InvalidArgument, "Missing tool call function: {s}", .{d});
                }
                const fc = try tool_call.at("function");
                if (!fc.contains("name")) {
                    const d = try dumpCompact(gpa, tool_call);
                    defer gpa.free(d);
                    return throw(error.InvalidArgument, "Missing tool call name: {s}", .{d});
                }
                var tc: ToolCall = .{};
                tc.name = try (try fc.at("name")).get([]const u8);
                const args = try fc.at("arguments");
                var args_dump: ?[]u8 = null;
                defer if (args_dump) |s| gpa.free(s);
                if (args.isString()) {
                    tc.arguments = args.string;
                } else {
                    args_dump = try dumpCompact(gpa, args);
                    tc.arguments = args_dump.?;
                }
                if (tool_call.contains("id")) tc.id = try (try tool_call.at("id")).get([]const u8);
                try msg.addToolCall(gpa, tc);
            }
        }
        if (!has_content and !has_tool_calls) {
            return throw(error.InvalidArgument, "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & https://github.com/ggml-org/llama.cpp/issues/12279)", .{});
        }
        if (message.contains("reasoning_content")) try Msg.setField(gpa, &msg.reasoning_content, try (try message.at("reasoning_content")).get([]const u8));
        if (message.contains("name")) try Msg.setField(gpa, &msg.tool_name, try (try message.at("name")).get([]const u8));
        if (message.contains("tool_call_id")) try Msg.setField(gpa, &msg.tool_call_id, try (try message.at("tool_call_id")).get([]const u8));
        try collectExtraJson(gpa, message, has_tool_calls, &msg);

        try msgs.append(gpa, msg);
    }
}

/// The block of `common_chat_msgs_parse_oaicompat` (chat.cpp:350 @8642541b2)
/// that keeps a message's keys beyond OpenAI's in `extra_json`, a tool call's
/// beyond its own under `__tool_call_extras`.
fn collectExtraJson(gpa: std.mem.Allocator, message: *Value, has_tool_calls: bool, msg: *Msg) Error!void {
    var extra = Value.emptyObject();
    defer extra.deinit(gpa);
    var it = message.items();
    while (it.next()) |item| {
        const k = item.key;
        if (!isOneOf(k, &.{ "role", "content", "tool_calls", "reasoning_content", "name", "tool_call_id" })) {
            try extra.set(gpa, k, try item.value.clone(gpa));
        }
    }
    if (has_tool_calls) {
        // per tool call: keys besides type/function/id, and function keys besides name/arguments
        var tc_extra = Value.emptyArray();
        defer tc_extra.deinit(gpa);
        var any = false;
        var tit = (try message.at("tool_calls")).items();
        while (tit.next()) |titem| {
            const tool_call = titem.value;
            var e = Value.emptyObject();
            errdefer e.deinit(gpa);
            var kit = tool_call.items();
            while (kit.next()) |kv| {
                if (!isOneOf(kv.key, &.{ "type", "function", "id" })) {
                    try e.set(gpa, kv.key, try kv.value.clone(gpa));
                }
            }
            var fit = (try tool_call.at("function")).items();
            while (fit.next()) |kv| {
                if (!isOneOf(kv.key, &.{ "name", "arguments" })) {
                    const slot = try e.index(gpa, "function");
                    try slot.set(gpa, kv.key, try kv.value.clone(gpa));
                }
            }
            any = any or !e.empty();
            try tc_extra.pushBack(gpa, e);
        }
        if (any) {
            const moved = tc_extra;
            tc_extra = .null;
            try extra.set(gpa, "__tool_call_extras", moved);
        }
    }
    if (!extra.empty()) {
        const d = try dumpCompact(gpa, &extra);
        gpa.free(msg.extra_json);
        msg.extra_json = d;
    }
}

fn isOneOf(k: []const u8, set: []const []const u8) bool {
    for (set) |s| if (std.mem.eql(u8, k, s)) return true;
    return false;
}

/// The two capabilities of `jinja::caps` that rendering messages reads.
pub const ContentCaps = struct {
    supports_string_content: bool = true,
    supports_typed_content: bool = true,
};

/// Ports `messages_inp_normalizer`'s `concat_content_parts` (chat.cpp:527 @8642541b2).
fn concatContentParts(gpa: std.mem.Allocator, parts: *Value) Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(gpa);
    var last_was_media_marker = false;
    var it = parts.items();
    while (it.next()) |item| {
        const part = item.value;
        const type_ = try part.value("type", []const u8, "");
        var add_new_line = true;
        if (std.mem.eql(u8, type_, "text")) {
            add_new_line = !last_was_media_marker and text.items.len != 0;
            last_was_media_marker = false;
        } else if (std.mem.eql(u8, type_, "media_marker")) {
            add_new_line = false;
            last_was_media_marker = true;
        } else {
            logf(.warn, "Ignoring content part type: {s}\n", .{type_});
            continue;
        }
        if (add_new_line) try text.append(gpa, '\n');
        try text.appendSlice(gpa, try part.value("text", []const u8, ""));
    }
    return text.toOwnedSlice(gpa);
}

/// Ports `messages_inp_normalizer`'s `normalize` (chat.cpp:499 @8642541b2): a
/// message's content turned into the one form the template takes, when it
/// takes only one.
///
/// Return: owned by the caller.
pub fn normalizeMessages(gpa: std.mem.Allocator, caps: ContentCaps, messages: *const Value) Error!Value {
    const only_string = caps.supports_string_content and !caps.supports_typed_content;
    const only_typed = !caps.supports_string_content and caps.supports_typed_content;
    if ((!only_string and !only_typed) or !messages.isArray()) return messages.clone(gpa);
    var normalized = Value.emptyArray();
    errdefer normalized.deinit(gpa);
    for (messages.array.items) |*msg| {
        var copy = try msg.clone(gpa);
        errdefer copy.deinit(gpa);
        if (copy.contains("content")) {
            const it = try copy.at("content");
            if (only_typed and it.isString()) {
                var part = Value.emptyObject();
                errdefer part.deinit(gpa);
                try part.set(gpa, "type", try Value.from(gpa, "text"));
                try part.set(gpa, "text", try Value.from(gpa, it.string));
                var arr = Value.emptyArray();
                errdefer arr.deinit(gpa);
                try arr.pushBack(gpa, part);
                it.assign(gpa, arr);
            } else if (only_string and it.isArray()) {
                const s = try concatContentParts(gpa, it);
                defer gpa.free(s);
                it.assign(gpa, try Value.from(gpa, s));
            }
        }
        try normalized.pushBack(gpa, copy);
    }
    return normalized;
}

/// Ports `render_message_to_json` (chat.cpp:554 @8642541b2).
///
/// Return: owned by the caller.
pub fn renderMessagesToJson(gpa: std.mem.Allocator, msgs: []const Msg, caps: ContentCaps) Error!Value {
    if (!caps.supports_string_content and !caps.supports_typed_content) {
        logf(.warn, "render_message_to_json: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\n", .{});
    }
    var messages = Value.emptyArray();
    defer messages.deinit(gpa);
    for (msgs) |*m| try messages.pushBack(gpa, try m.toJsonOaicompat(gpa, false));
    return normalizeMessages(gpa, caps, &messages);
}

/// Ports `render_message_to_json_deepseek_v4_1` (chat.cpp:568 @8642541b2):
/// DeepSeek V4.1 reads non-OpenAI message keys and joins content blocks with
/// "\n\n" around images, so media markers stay separate text parts and the
/// extra keys are restored.
///
/// Return: owned by the caller.
pub fn renderMessagesToJsonDeepseekV41(gpa: std.mem.Allocator, msgs: []const Msg, caps: ContentCaps) Error!Value {
    var messages = Value.emptyArray();
    defer messages.deinit(gpa);
    for (msgs) |*m| {
        var jmsg: Value = undefined;
        if (m.containsMedia() and m.content.len == 0) {
            var typed = try m.clone(gpa);
            defer typed.deinit(gpa);
            for (typed.content_parts.items) |*part| {
                const t = try gpa.dupe(u8, "text");
                gpa.free(part.type);
                part.type = t;
            }
            jmsg = try typed.toJsonOaicompat(gpa, false);
        } else {
            jmsg = try m.toJsonOaicompat(gpa, false);
        }
        errdefer jmsg.deinit(gpa);
        if (m.extra_json.len != 0) {
            var extra = try json.parse(gpa, m.extra_json);
            defer extra.deinit(gpa);
            var it = extra.items();
            while (it.next()) |kv| {
                if (std.mem.eql(u8, kv.key, "__tool_call_extras")) {
                    if (!jmsg.contains("tool_calls")) continue;
                    const tcs = try jmsg.at("tool_calls");
                    var i: usize = 0;
                    while (i < tcs.size() and i < kv.value.size()) : (i += 1) {
                        const tc = try tcs.atIndex(i);
                        var eit = (try kv.value.atIndex(i)).items();
                        while (eit.next()) |tkv| {
                            if (std.mem.eql(u8, tkv.key, "function")) {
                                var fit = tkv.value.items();
                                while (fit.next()) |fkv| {
                                    const func = try tc.index(gpa, "function");
                                    try func.set(gpa, fkv.key, try fkv.value.clone(gpa));
                                }
                            } else {
                                try tc.set(gpa, tkv.key, try tkv.value.clone(gpa));
                            }
                        }
                    }
                } else if (!jmsg.contains(kv.key)) {
                    try jmsg.set(gpa, kv.key, try kv.value.clone(gpa));
                }
            }
        }
        try messages.pushBack(gpa, jmsg);
    }
    return normalizeMessages(gpa, caps, &messages);
}

/// Ports `common_chat_msgs_to_json_oaicompat` (chat.cpp:611 @8642541b2).
///
/// Return: owned by the caller.
pub fn msgsToJsonOaicompat(gpa: std.mem.Allocator, msgs: []const Msg, concat_typed_text: bool) Error!Value {
    return renderMessagesToJson(gpa, msgs, .{ .supports_string_content = true, .supports_typed_content = !concat_typed_text });
}

/// Ports `common_chat_tools_to_json_oaicompat` (chat.cpp:618 @8642541b2).
///
/// Return: owned by the caller; null for no tools.
pub fn toolsToJsonOaicompat(gpa: std.mem.Allocator, tools: []const Tool) Error!Value {
    if (tools.len == 0) return .null;
    var result = Value.emptyArray();
    errdefer result.deinit(gpa);
    for (tools) |tool| {
        var params = try json.parse(gpa, tool.parameters);
        errdefer params.deinit(gpa);
        var func = Value.emptyObject();
        errdefer func.deinit(gpa);
        try func.set(gpa, "name", try Value.from(gpa, tool.name));
        try func.set(gpa, "description", try Value.from(gpa, tool.description));
        try func.set(gpa, "parameters", params);
        if (tool.defer_loading) try func.set(gpa, "defer_loading", try Value.from(gpa, true));
        var t = Value.emptyObject();
        errdefer t.deinit(gpa);
        try t.set(gpa, "type", try Value.from(gpa, "function"));
        try t.set(gpa, "function", func);
        try result.pushBack(gpa, t);
    }
    return result;
}

/// Ports `common_chat_tool_parameters` (chat.cpp:641 @8642541b2): a
/// function's parameters, or an empty object schema.
///
/// Return: owned by the caller.
pub fn toolParameters(gpa: std.mem.Allocator, function: *const Value) Error!Value {
    if (function.contains("parameters")) {
        const params = try function.at("parameters");
        if (!params.isNull() and !(params.isObject() and params.empty())) return params.clone(gpa);
    }
    var out = Value.emptyObject();
    errdefer out.deinit(gpa);
    try out.set(gpa, "type", try Value.from(gpa, "object"));
    try out.set(gpa, "properties", Value.emptyObject());
    return out;
}

/// Ports `common_chat_tools_parse_oaicompat` (chat.cpp:651 @8642541b2).
///
/// Return: owned by the caller; free each tool with `deinit(gpa)`.
pub fn toolsParseOaicompat(gpa: std.mem.Allocator, tools: *Value) Error!std.ArrayList(Tool) {
    var result: std.ArrayList(Tool) = .empty;
    errdefer {
        for (result.items) |*t| t.deinit(gpa);
        result.deinit(gpa);
    }
    toolsParseInner(gpa, tools, &result) catch |e| {
        if (e == error.OutOfMemory) return e;
        const d = try json.dump(gpa, tools, 2);
        defer gpa.free(d);
        const suffix = try std.fmt.allocPrint(gpa, "; tools = {s}", .{d});
        defer gpa.free(suffix);
        return exception.rethrow(error.RuntimeError, e, "Failed to parse tools: ", suffix);
    };
    return result;
}

fn toolsParseInner(gpa: std.mem.Allocator, tools: *Value, result: *std.ArrayList(Tool)) Error!void {
    if (tools.isNull()) return;
    if (!tools.isArray()) {
        const d = try dumpCompact(gpa, tools);
        defer gpa.free(d);
        return throw(error.InvalidArgument, "Expected 'tools' to be an array, got {s}", .{d});
    }
    var it = tools.items();
    while (it.next()) |item| {
        const tool = item.value;
        if (!tool.contains("type")) {
            const d = try dumpCompact(gpa, tool);
            defer gpa.free(d);
            return throw(error.InvalidArgument, "Missing tool type: {s}", .{d});
        }
        const type_ = try tool.at("type");
        if (!isString(type_, "function")) {
            const d = try dumpCompact(gpa, tool);
            defer gpa.free(d);
            return throw(error.InvalidArgument, "Unsupported tool type: {s}", .{d});
        }
        if (!tool.contains("function")) {
            const d = try dumpCompact(gpa, tool);
            defer gpa.free(d);
            return throw(error.InvalidArgument, "Missing tool function: {s}", .{d});
        }
        const function = try tool.at("function");

        // "defer_loading" sits on the function, or on the tool next to "type"; the function wins
        var defer_false: Value = .{ .boolean = false };
        const defer_v: *const Value = if (function.contains("defer_loading"))
            try function.at("defer_loading")
        else if (tool.contains("defer_loading"))
            try tool.at("defer_loading")
        else
            &defer_false;
        if (!defer_v.isBoolean()) {
            const d = try dumpCompact(gpa, defer_v);
            defer gpa.free(d);
            return throw(error.InvalidArgument, "Expected 'defer_loading' to be a boolean, got {s}", .{d});
        }

        var t: Tool = .{};
        errdefer t.deinit(gpa);
        t.defer_loading = defer_v.boolean;
        t.name = try getString(gpa, try function.at("name"));
        t.description = try gpa.dupe(u8, try function.value("description", []const u8, ""));
        // `value("parameters", json::object())`: the member when there is
        // one, whatever it is, else `{}`
        if (function.contains("parameters")) {
            t.parameters = try dumpCompact(gpa, try function.at("parameters"));
        } else {
            t.parameters = try gpa.dupe(u8, "{}");
        }
        try result.append(gpa, t);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a message round-trips through the OpenAI form" {
    const a = std.testing.allocator;
    var doc = try json.parse(a,
        \\[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"text","text":"there"}]},
        \\ {"role":"assistant","content":null,"tool_calls":[{"type":"function","id":"c1","function":{"name":"f","arguments":{"x":1}}}]}]
    );
    defer doc.deinit(a);
    var msgs = try msgsParseOaicompat(a, &doc);
    defer {
        for (msgs.items) |*m| m.deinit(a);
        msgs.deinit(a);
    }
    try std.testing.expectEqual(@as(usize, 2), msgs.items.len);
    try std.testing.expectEqualStrings("{\"x\":1}", msgs.items[1].tool_calls.items[0].arguments);
    var back = try msgsToJsonOaicompat(a, msgs.items, true);
    defer back.deinit(a);
    const s = try json.dump(a, &back, -1);
    defer a.free(s);
    try std.testing.expectEqualStrings(
        \\[{"role":"user","content":"hi\nthere"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{\"x\":1}"},"id":"c1"}]}]
    , s);
}

test "a bad message names itself in the error" {
    const a = std.testing.allocator;
    var doc = try json.parse(a, "[{\"content\":\"x\"}]");
    defer doc.deinit(a);
    try std.testing.expectError(error.RuntimeError, msgsParseOaicompat(a, &doc));
    try std.testing.expectEqualStrings("Failed to parse messages: Missing 'role' in message: {\"content\":\"x\"}", exception.what(error.RuntimeError));
}

test "diffs carry only what grew" {
    const a = std.testing.allocator;
    var prv: Msg = .{};
    defer prv.deinit(a);
    var cur: Msg = .{};
    defer cur.deinit(a);
    try Msg.setField(a, &prv.content, "Hello");
    try Msg.setField(a, &cur.content, "Hello, world");
    try cur.addToolCall(a, .{ .name = "f", .arguments = "{" });
    var diffs = try computeDiffs(a, &prv, &cur);
    defer diffs.deinit(a);
    try std.testing.expectEqual(@as(usize, 2), diffs.items.len);
    try std.testing.expectEqualStrings(", world", diffs.items[0].content_delta);
    try std.testing.expectEqual(@as(usize, 0), diffs.items[1].tool_call_index);
}
