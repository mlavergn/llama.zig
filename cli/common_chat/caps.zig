//! What a chat template can do, read off it by rendering it over probe
//! conversations and watching which values it touches.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/jinja/caps.h` and `llama.cpp/common/jinja/caps.cpp`
//! (llama.cpp v0.9.0, `cd959bae9`): `jinja`'s `caps`, `caps_get` and its
//! `caps_try_execute`. `caps_apply_preserve_reasoning` and
//! `caps_apply_reasoning_effort` live in `template.zig`, beside the
//! rendering they set up. Each declaration names what it ports and the line
//! it began at.
//!
//! # How the touching is seen
//!
//! Upstream renders each probe with `context`'s `is_get_stats` on, and its
//! runtime records, on the value itself, that it was used and which
//! operations it met. zigjinja records the same at the same points when a
//! `jinja.stats.Stats` is active (`zigjinja/src/stats.zig`), keyed by the
//! value's identity, which a copy keeps; the questions below are asked of
//! the values this file bound.

const std = @import("std");
const json = @import("json");
const jinja = @import("zigjinja");
const exception = @import("exception.zig");
const msg_mod = @import("msg.zig");
const template = @import("template.zig");
const log = @import("log.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const Template = template.Template;
const Render = template.Render;

/// `g_jinja_debug` (runtime.h:15 @cd959bae9): the engine's debug switch,
/// off unless a debugger sets it.
pub var jinja_debug: bool = false;

/// Ports `jinja`'s `caps` (caps.h:11 @cd959bae9).
pub const Caps = struct {
    supports_tools: bool = true,
    supports_tool_calls: bool = true,
    supports_system_role: bool = true,
    supports_parallel_tool_calls: bool = true,
    supports_preserve_reasoning: bool = false,
    supports_reasoning_effort: bool = false,
    /// reasoning effort levels the template renders distinctly, from lowest
    /// to highest; the template's `gpa` owns them, and `Template.deinit` frees
    /// them
    reasoning_effort_levels: []const []const u8 = &.{},
    supports_string_content: bool = true,
    supports_typed_content: bool = false,
    supports_object_arguments: bool = false,

    /// One entry of `to_map`.
    pub const Entry = struct { name: []const u8, value: bool };

    /// Ports `to_map` (caps.cpp:108 @cd959bae9), in the `std`'s `map`'s order:
    /// sorted by name.
    pub fn toMap(self: Caps) [9]Entry {
        return .{
            .{ .name = "supports_object_arguments", .value = self.supports_object_arguments },
            .{ .name = "supports_parallel_tool_calls", .value = self.supports_parallel_tool_calls },
            .{ .name = "supports_preserve_reasoning", .value = self.supports_preserve_reasoning },
            .{ .name = "supports_reasoning_effort", .value = self.supports_reasoning_effort },
            .{ .name = "supports_string_content", .value = self.supports_string_content },
            .{ .name = "supports_system_role", .value = self.supports_system_role },
            .{ .name = "supports_tool_calls", .value = self.supports_tool_calls },
            .{ .name = "supports_tools", .value = self.supports_tools },
            .{ .name = "supports_typed_content", .value = self.supports_typed_content },
        };
    }

    /// Ports `to_string` (caps.cpp:122 @cd959bae9).
    ///
    /// Return: owned by `a`.
    pub fn toString(self: Caps, a: Allocator) Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(a, "Caps(\n");
        for (self.toMap()) |e| try out.print(a, "  {s}={s}\n", .{ e.name, if (e.value) "true" else "false" });
        try out.appendSlice(a, ")");
        return out.toOwnedSlice(a);
    }

    /// The two of these that `messages_inp_normalizer` reads.
    pub fn contentCaps(self: Caps) msg_mod.ContentCaps {
        return .{ .supports_string_content = self.supports_string_content, .supports_typed_content = self.supports_typed_content };
    }
};

/// One probe's render: its outcome, and the record of what it touched.
const Probe = struct {
    r: Render,
    stats: jinja.stats.Stats,
    messages: jinja.Value = .null,
    tools: jinja.Value = .null,
    success: bool = false,
    result: []const u8 = "",

    fn deinit(self: *Probe) void {
        self.stats.deinit();
        self.r.deinit();
    }

    fn used(self: *const Probe, v: ?jinja.Value) bool {
        return if (v) |x| self.stats.isUsed(x) else false;
    }

    fn hasOp(self: *const Probe, v: ?jinja.Value, op: []const u8) bool {
        return if (v) |x| self.stats.hasOp(x, op) else false;
    }
};

/// What a probe sets on the context after the JSON, as `ctx_fn` does.
const CtxFn = union(enum) {
    none,
    preserve_reasoning,
    /// the effort `caps_apply_reasoning_effort` binds
    reasoning_effort: []const u8,
};

/// Ports `caps_try_execute` (caps.cpp:37 @cd959bae9): `messages` and
/// `tools` bound with empty BOS and EOS and a generation prompt, the
/// context adjusted, the template run with statistics on. A render that
/// fails leaves an empty result, as upstream catches and ignores it -- but
/// first it is retried once with an empty `reasoning_content` on every
/// assistant turn that lacks one, for templates that require a thinking
/// field on each (K2 Horizon's). `p` holds the last attempt.
fn tryExecute(p: *Probe, gpa: Allocator, tmpl: *const Template, messages_json: []const u8, tools_json: ?[]const u8, ctx_fn: CtxFn) Error!?jinja.Value {
    var attempt: usize = 0;
    while (true) : (attempt += 1) {
        p.* = .{ .r = undefined, .stats = jinja.stats.Stats.init(gpa) };
        p.r.init(gpa, true);
        const a = p.r.alloc();
        var msgs = try json.parse(a, messages_json);
        if (attempt == 1) _ = try addEmptyReasoning(a, &msgs);
        // whether a failed first attempt has anything to retry with
        const retryable = attempt == 0 and try addEmptyReasoning(null, &msgs);

        var vars = Value.emptyObject();
        try vars.set(a, "messages", msgs);
        try vars.set(a, "tools", if (tools_json) |t| try json.parse(a, t) else Value.emptyArray());
        try vars.set(a, "bos_token", try Value.from(a, ""));
        try vars.set(a, "eos_token", try Value.from(a, ""));
        try vars.set(a, "add_generation_prompt", try Value.from(a, true));
        try p.r.bindJson(&vars, true);

        var effort: ?jinja.Value = null;
        switch (ctx_fn) {
            .none => {},
            .preserve_reasoning => {
                try p.r.set("enable_thinking", .{ .boolean = true });
                try template.capsApplyPreserveReasoning(&p.r, true);
            },
            .reasoning_effort => |level| {
                try p.r.set("enable_thinking", .{ .boolean = true });
                effort = try template.capsApplyReasoningEffort(&p.r, level);
            },
        }
        p.messages = p.r.vars.get("messages").?;
        p.tools = p.r.vars.get("tools").?;

        jinja.stats.active = &p.stats;
        defer jinja.stats.active = null;
        if (p.r.run(tmpl, a)) |text| {
            p.result = text;
            p.success = true;
        } else |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            p.result = "";
            p.success = false;
        }

        // some templates require a thinking field on every assistant turn (e.g. K2 Horizon):
        // retry once with an empty reasoning_content on the assistant turns that lack one
        if (!p.success and retryable) {
            p.deinit();
            continue;
        }
        return effort;
    }
}

/// The retry's edit inside `caps_try_execute` (caps.cpp:37 @cd959bae9):
/// `reasoning_content: ""` on every assistant message that has none.
/// With `a` null it only reports whether there is one to add.
///
/// Return: whether a message lacked it.
fn addEmptyReasoning(a: ?Allocator, msgs: *Value) Error!bool {
    var added = false;
    var it = msgs.items();
    while (it.next()) |item| {
        const msg = item.value;
        if (msg.isObject() and std.mem.eql(u8, try msg.value("role", []const u8, ""), "assistant") and !msg.contains("reasoning_content")) {
            if (a) |alloc| try msg.set(alloc, "reasoning_content", try Value.from(alloc, ""));
            added = true;
        }
    }
    return added;
}

/// `v->at(i)` on a list, null when it is not one or too short.
fn atIndex(v: ?jinja.Value, i: usize) ?jinja.Value {
    const x = v orelse return null;
    if (x != .list or i >= x.list.items.items.len) return null;
    return x.list.items.items[i];
}

/// `v->at(key)` on a mapping, null when it is not one or lacks the key.
fn atKey(v: ?jinja.Value, key: []const u8) ?jinja.Value {
    const x = v orelse return null;
    if (x != .dict) return null;
    return x.dict.get(key);
}

const content_marker = "STRING_MARKER";

const probe_tools =
    \\[{"name":"tool","type":"function","function":{"name":"tool1","description":"Tool description",
    \\"parameters":{"type":"object","properties":{"arg":{"type":"string","description":"Arg description"}},"required":["arg"]}}}]
;

fn toolCallMessages(comptime args: []const u8, comptime second_call: bool) []const u8 {
    const call1 = "{\"id\":\"call00001\",\"type\":\"function\",\"function\":{\"name\":\"tool1\",\"arguments\":" ++ args ++ "}}";
    const call2 = if (second_call) ",{\"id\":\"call00002\",\"type\":\"function\",\"function\":{\"name\":\"tool1\",\"arguments\":" ++ args ++ "}}" else "";
    return "[{\"role\":\"user\",\"content\":\"User message\"}," ++
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[" ++ call1 ++ call2 ++ "]}," ++
        "{\"role\":\"tool\",\"content\":\"Tool response\",\"tool_call_id\":\"call00001\"}," ++
        "{\"role\":\"assistant\",\"content\":\"The tool response was 'tool response'\"}," ++
        "{\"role\":\"user\",\"content\":\"User message\"}]";
}

/// `render` in `caps_get` (caps.cpp:132 @cd959bae9): a single user turn with
/// the given effort, nothing when the template raises.
///
/// Return: the text, owned by `gpa`, or null.
fn renderEffort(gpa: Allocator, tmpl: *const Template, effort: []const u8) Error!?[]u8 {
    var p: Probe = undefined;
    _ = try tryExecute(&p, gpa, tmpl, "[{\"role\":\"user\",\"content\":\"User message\"}]", null, .{ .reasoning_effort = effort });
    defer p.deinit();
    return if (p.success) try gpa.dupe(u8, p.result) else null;
}

/// The reasoning effort levels of `caps_get` (caps.cpp:132 @cd959bae9): an
/// unknown level shows the render of the template fallback, a level that
/// renders the same is not honored; levels sharing a render are aliases and
/// the highest one is kept.
///
/// Return: the levels, owned by `gpa`; the strings are static.
fn reasoningEffortLevels(gpa: Allocator, tmpl: *const Template) Error![]const []const u8 {
    const candidates = [_][]const u8{ "minimal", "low", "medium", "high", "xhigh", "max" };

    const unknown = try renderEffort(gpa, tmpl, "REASONING_EFFORT_UNKNOWN");
    defer if (unknown) |u| gpa.free(u);
    var renders: std.ArrayList([]u8) = .empty;
    defer {
        for (renders.items) |r| gpa.free(r);
        renders.deinit(gpa);
    }
    var levels: std.ArrayList([]const u8) = .empty;
    errdefer levels.deinit(gpa);
    next: for (candidates) |level| {
        const rendered = (try renderEffort(gpa, tmpl, level)) orelse continue;
        if (unknown) |u| {
            if (std.mem.eql(u8, rendered, u)) {
                gpa.free(rendered);
                continue;
            }
        }
        for (renders.items, 0..) |r, i| {
            if (std.mem.eql(u8, r, rendered)) {
                levels.items[i] = level;
                gpa.free(rendered);
                continue :next;
            }
        }
        try renders.append(gpa, rendered);
        try levels.append(gpa, level);
    }
    return levels.toOwnedSlice(gpa);
}

/// Ports `caps_get` (caps.cpp:132 @cd959bae9): the template rendered over
/// the probes, each capability read off what the render touched.
pub fn capsGet(tmpl: *const Template) Error!Caps {
    const gpa = tmpl.gpa;
    var result: Caps = .{};

    // case: typed content support
    var checks_for_string = false;
    {
        var p: Probe = undefined;
        _ = try tryExecute(&p, gpa, tmpl, "[{\"role\":\"user\",\"content\":\"" ++ content_marker ++ "\"}]", null, .none);
        defer p.deinit();
        const content = atKey(atIndex(p.messages, 0), "content");
        if (p.hasOp(content, "test_is_string")) checks_for_string = true;
        const used_as_array = p.hasOp(content, "selectattr") or p.hasOp(content, "array_access");
        if (used_as_array) result.supports_typed_content = true;
        if (!p.success) {
            result.supports_string_content = false;
        } else if (used_as_array and std.mem.indexOf(u8, p.result, content_marker) == null) {
            result.supports_string_content = false;
        }
    }

    if (checks_for_string) {
        var p: Probe = undefined;
        _ = try tryExecute(&p, gpa, tmpl, "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"" ++ content_marker ++ "\"}]}]", null, .none);
        defer p.deinit();
        const content = atKey(atIndex(p.messages, 0), "content");
        const used_as_array = p.hasOp(content, "selectattr") or p.hasOp(content, "array_access");
        if (used_as_array and p.success) result.supports_typed_content = true;
    }

    // case: system prompt support
    {
        var p: Probe = undefined;
        _ = try tryExecute(&p, gpa, tmpl, "[{\"role\":\"system\",\"content\":\"System message\"},{\"role\":\"user\",\"content\":\"User message\"}]", null, .none);
        defer p.deinit();
        const content = atKey(atIndex(p.messages, 0), "content");
        if (!p.used(content)) result.supports_system_role = false;
    }

    // case: tools support: single call with object arguments
    {
        var p: Probe = undefined;
        _ = try tryExecute(&p, gpa, tmpl, comptime toolCallMessages("{\"arg\":\"value\"}", false), probe_tools, .none);
        defer p.deinit();
        if (p.success) blk: {
            const tool_name = atKey(atKey(atIndex(p.tools, 0), "function"), "name");
            if (!p.used(tool_name)) result.supports_tools = false;
            const tool_calls = atKey(atIndex(p.messages, 1), "tool_calls");
            if (!p.used(tool_calls)) {
                result.supports_tool_calls = false;
                break :blk;
            }
            const tool_arg = atKey(atKey(atKey(atIndex(tool_calls, 0), "function"), "arguments"), "arg");
            if (p.used(tool_arg)) result.supports_object_arguments = true;
        }
    }

    if (!result.supports_object_arguments) {
        // case: tools support: single call with string arguments
        var p: Probe = undefined;
        _ = try tryExecute(&p, gpa, tmpl, comptime toolCallMessages("\"{\\\"arg\\\": \\\"value\\\"}\"", false), probe_tools, .none);
        defer p.deinit();
        if (!p.success) {
            result.supports_tool_calls = false;
            result.supports_tools = false;
        } else blk: {
            const tool_name = atKey(atKey(atIndex(p.tools, 0), "function"), "name");
            if (!p.used(tool_name)) result.supports_tools = false;
            const tool_calls = atKey(atIndex(p.messages, 1), "tool_calls");
            if (!p.used(tool_calls)) {
                result.supports_tool_calls = false;
                break :blk;
            }
        }
    }

    // case: tools support: parallel calls
    {
        var p: Probe = undefined;
        if (result.supports_object_arguments) {
            _ = try tryExecute(&p, gpa, tmpl, comptime toolCallMessages("{\"arg\":\"value\"}", true), probe_tools, .none);
        } else {
            _ = try tryExecute(&p, gpa, tmpl, comptime toolCallMessages("\"{\\\"arg\\\": \\\"value\\\"}\"", true), probe_tools, .none);
        }
        defer p.deinit();
        if (!p.success) {
            result.supports_parallel_tool_calls = false;
        } else {
            const tool_calls = atKey(atIndex(p.messages, 1), "tool_calls");
            const tool_call_1 = atKey(atIndex(tool_calls, 1), "function");
            if (!p.used(tool_call_1)) result.supports_parallel_tool_calls = false;
        }
    }

    // case: preserve reasoning content in chat history
    {
        const reasoning_placeholder = "<REASONING_CONTENT_PLACEHOLDER>";
        var p: Probe = undefined;
        _ = try tryExecute(&p, gpa, tmpl, "[{\"role\":\"user\",\"content\":\"User message\"}," ++
            "{\"role\":\"assistant\",\"content\":\"Assistant message\",\"reasoning_content\":\"" ++ reasoning_placeholder ++ "\"}," ++
            "{\"role\":\"user\",\"content\":\"User message\"}," ++
            "{\"role\":\"assistant\",\"content\":\"Assistant message\",\"reasoning_content\":\"Reasoning content\"}," ++
            "{\"role\":\"user\",\"content\":\"User message\"}]", null, .preserve_reasoning);
        defer p.deinit();
        if (std.mem.indexOf(u8, p.result, reasoning_placeholder) != null) result.supports_preserve_reasoning = true;
    }

    // case: reasoning effort level
    {
        var p: Probe = undefined;
        const effort = try tryExecute(&p, gpa, tmpl, "[{\"role\":\"user\",\"content\":\"User message\"}]", null, .{ .reasoning_effort = "low" });
        defer p.deinit();
        result.supports_reasoning_effort = p.used(effort);
    }

    if (result.supports_reasoning_effort) {
        result.reasoning_effort_levels = try reasoningEffortLevels(gpa, tmpl);
    }

    // `JJ_DEBUG` (runtime.h:13 @cd959bae9): printed only under the engine's own debug switch
    // (`g_jinja_debug`), which nothing sets -- never through `common_log`
    if (jinja_debug) {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        log.dbg("{s}\n", .{try result.toString(arena_state.allocator())});
    }
    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a template that tests content for a string checks typed content too" {
    var t = try Template.init(std.testing.allocator,
        \\{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{{ p.text }}{% endfor %}{% endif %}{% endfor %}
    , "", "");
    defer t.deinit();
    try std.testing.expect(t.caps.supports_string_content);
    try std.testing.expect(t.caps.supports_typed_content);
    try std.testing.expect(t.caps.supports_system_role);
    try std.testing.expect(!t.caps.supports_tools);
}
