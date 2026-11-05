//! The port's half of `scripts/chat-msg-diff`: the records
//! `harness/chat_msg_ref.cpp` runs through upstream's `chat.cpp`, run
//! through `cli/common_chat/`, written as the same transcript.
//!
//! # Provenance
//!
//! **Not a port.** A test driver; it mirrors `harness/chat_msg_ref.cpp`
//! line for line, and the two must stay in step.

const std = @import("std");
const json = @import("json");
const chat = @import("common_chat");

const Msg = chat.Msg;
const Value = json.Value;

var gpa_state: std.heap.DebugAllocator(.{}) = .init;
const gpa = gpa_state.allocator();

var out: std.ArrayList(u8) = .empty;

fn str(label: []const u8, s: []const u8) !void {
    try out.print(gpa, "{s}({d}):{s}\n", .{ label, s.len, s });
}

fn line(comptime fmt: []const u8, args: anytype) !void {
    try out.print(gpa, fmt ++ "\n", args);
}

/// The C++ driver's `guarded`: a throw is printed as its class and text.
fn guarded(r: anyerror!void) !void {
    r catch |e| switch (e) {
        error.JsonError => try str("json_error", chat.what(e)),
        error.InvalidArgument => try str("invalid_argument", chat.what(e)),
        error.RuntimeError => try str("runtime_error", chat.what(e)),
        else => return e,
    };
}

fn dumpOwned(v: Value, indent: i32) ![]u8 {
    var x = v;
    defer x.deinit(gpa);
    return json.dump(gpa, &x, indent);
}

fn showMsg(m: *const Msg) !void {
    try str("role", m.role);
    try str("content", m.content);
    for (m.content_parts.items) |p| {
        try str("part.type", p.type);
        try str("part.text", p.text);
    }
    for (m.tool_calls.items) |tc| {
        try str("tc.name", tc.name);
        try str("tc.arguments", tc.arguments);
        try str("tc.id", tc.id);
    }
    try str("reasoning_content", m.reasoning_content);
    try str("tool_name", m.tool_name);
    try str("tool_call_id", m.tool_call_id);
    try line("empty {d} media {d}", .{ @intFromBool(m.empty()), @intFromBool(m.containsMedia()) });
    try guarded(oai(m, false, "oai"));
    try guarded(oai(m, true, "oai_concat"));
    try guarded(render(m, "\n\n", "render"));
    try guarded(render(m, "|", "render_bar"));
}

fn oai(m: *const Msg, concat: bool, label: []const u8) !void {
    const s = try dumpOwned(try m.toJsonOaicompat(gpa, concat), -1);
    defer gpa.free(s);
    try str(label, s);
}

fn render(m: *const Msg, delim: []const u8, label: []const u8) !void {
    const s = try m.renderContent(gpa, delim);
    defer gpa.free(s);
    try str(label, s);
}

fn freeMsgs(msgs: *std.ArrayList(Msg)) void {
    for (msgs.items) |*m| m.deinit(gpa);
    msgs.deinit(gpa);
}

fn oneMsg(j: *const Value) !Msg {
    var arr = Value.emptyArray();
    defer arr.deinit(gpa);
    try arr.pushBack(gpa, try j.clone(gpa));
    var msgs = try chat.msg.msgsParseOaicompat(gpa, &arr);
    defer freeMsgs(&msgs);
    if (msgs.items.len == 0) return error.Unreachable;
    return msgs.items[0].clone(gpa);
}

fn opM(j: *Value) !void {
    var msgs = try chat.msg.msgsParseOaicompat(gpa, j);
    defer freeMsgs(&msgs);
    try line("n {d}", .{msgs.items.len});
    for (msgs.items) |*m| try showMsg(m);
    try guarded(all(msgs.items, false));
    try guarded(all(msgs.items, true));
    for (msgs.items) |*x| for (msgs.items) |*y| try line("eq {d}", .{@intFromBool(x.eql(y))});
}

fn all(msgs: []const Msg, concat: bool) !void {
    const s = try dumpOwned(try chat.msg.msgsToJsonOaicompat(gpa, msgs, concat), if (concat) 2 else -1);
    defer gpa.free(s);
    try str(if (concat) "all_concat" else "all", s);
}

fn opT(j: *Value) !void {
    var tools = try chat.msg.toolsParseOaicompat(gpa, j);
    defer {
        for (tools.items) |*t| t.deinit(gpa);
        tools.deinit(gpa);
    }
    try line("n {d}", .{tools.items.len});
    for (tools.items) |t| {
        try str("name", t.name);
        try str("description", t.description);
        try str("parameters", t.parameters);
    }
    try guarded(toolsBack(tools.items));
}

fn toolsBack(tools: []const chat.Tool) !void {
    const s = try dumpOwned(try chat.msg.toolsToJsonOaicompat(gpa, tools), -1);
    defer gpa.free(s);
    try str("back", s);
}

fn opC(j: *Value) !void {
    const choice = try chat.msg.toolChoiceParseOaicompat(try j.get([]const u8));
    try line("choice {d}", .{@backingInt(choice)});
}

fn opK(j: *Value) !void {
    try line("continuation {d}", .{@backingInt(try chat.msg.continuationParse(j))});
}

fn ints(gpa_: std.mem.Allocator, v: *Value) !std.ArrayList(i32) {
    var r: std.ArrayList(i32) = .empty;
    const xs = try v.getSlice(gpa_, i32);
    defer gpa_.free(xs);
    try r.appendSlice(gpa_, xs);
    return r;
}

fn opS(j: *Value) !void {
    var d: chat.MsgDelimiters = .{};
    defer d.deinit(gpa);
    var it = (try j.at("delims")).items();
    while (it.next()) |item| {
        const e = item.value;
        const role = chat.Role.fromString(try (try e.at("role")).get([]const u8));
        try d.add(gpa, role, try (try e.at("delimiter")).get([]const u8));
        d.delimiters.items[d.delimiters.items.len - 1].tokens = try ints(gpa, try e.at("tokens"));
    }
    var skips: std.ArrayList([2]usize) = .empty;
    defer skips.deinit(gpa);
    var sit = (try j.at("skips")).items();
    while (sit.next()) |item| {
        const s = item.value;
        const start = try (try s.atIndex(0)).get(usize);
        const len = try (try s.atIndex(1)).get(usize);
        // `std::map`: a repeated start keeps the last length, sorted by start
        var found = false;
        for (skips.items) |*k| if (k[0] == start) {
            k[1] = len;
            found = true;
        };
        if (!found) try skips.append(gpa, .{ start, len });
    }
    std.mem.sort([2]usize, skips.items, {}, struct {
        fn lt(_: void, x: [2]usize, y: [2]usize) bool {
            return x[0] < y[0];
        }
    }.lt);
    var toks = try ints(gpa, try j.at("tokens"));
    defer toks.deinit(gpa);
    var spans = try d.split(gpa, toks.items, skips.items);
    defer spans.deinit(gpa);
    for (spans.spans.items) |s| try line("span {s} {d} {d} {d}", .{ s.role.toString(), s.pos, s.len, @intFromBool(s.valid()) });
    var p: i32 = -1;
    while (p < 12) : (p += 1) try line("user_start {d} {d}", .{ p, @intFromBool(spans.isUserStart(p)) });
    try line("last_user {d}", .{spans.lastUserMessagePos()});
}

fn opF(j: *Value) !void {
    var prv = try oneMsg(try j.atIndex(0));
    defer prv.deinit(gpa);
    var cur = try oneMsg(try j.atIndex(1));
    defer cur.deinit(gpa);
    var diffs = try chat.msg.computeDiffs(gpa, &prv, &cur);
    defer diffs.deinit(gpa);
    try line("n {d}", .{diffs.items.len});
    for (diffs.items) |d| {
        try str("reasoning_delta", d.reasoning_content_delta);
        try str("content_delta", d.content_delta);
        try line("index {d}", .{d.tool_call_index});
        try str("delta.name", d.tool_call_delta.name);
        try str("delta.arguments", d.tool_call_delta.arguments);
        try str("delta.id", d.tool_call_delta.id);
    }
    for (diffs.items) |x| for (diffs.items) |y| try line("eq {d}", .{@intFromBool(x.eql(y))});
}

fn opR(j: *Value) !void {
    const f = try chat.ReasoningFormat.fromName(try j.get([]const u8));
    try line("format {d} {s}", .{ @backingInt(f), f.name() });
}

fn opP(j: *Value) !void {
    const s = try dumpOwned(try chat.msg.toolParameters(gpa, j), -1);
    defer gpa.free(s);
    try str("params", s);
}

fn run(op: u8, payload: []const u8) !void {
    var j = json.parse(gpa, payload) catch |e| switch (e) {
        error.JsonError => return str("bad_input", json.what()),
        else => return e,
    };
    defer j.deinit(gpa);
    switch (op) {
        'M' => try guarded(opM(&j)),
        'T' => try guarded(opT(&j)),
        'C' => try guarded(opC(&j)),
        'K' => try guarded(opK(&j)),
        'S' => try guarded(opS(&j)),
        'F' => try guarded(opF(&j)),
        'R' => try guarded(opR(&j)),
        'P' => try guarded(opP(&j)),
        else => try line("unknown op", .{}),
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("usage: chat_msg_diff <in> <out>\n", .{});
        std.process.exit(2);
    }
    const io = init.io;
    const input = try std.Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .unlimited);
    var n: usize = 0;
    var lines = std.mem.splitScalar(u8, input, '\n');
    while (lines.next()) |rec| {
        if (rec.len < 2) continue;
        try line("## {d} {c}", .{ n, rec[0] });
        n += 1;
        try run(rec[0], rec[2..]);
    }
    for (std.enums.values(chat.Format)) |f| try line("format_name {s}", .{f.name()});
    for ([_][]const u8{ "system", "assistant", "user", "tool", "", "User", "developer" }) |r| {
        const role = chat.Role.fromString(r);
        try line("role {s} {d} {s}", .{ r, @backingInt(role), role.toString() });
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = out.items });
    out.deinit(gpa);
    gpa.free(input);
    if (gpa_state.deinit() == .leak) std.process.exit(3);
}
