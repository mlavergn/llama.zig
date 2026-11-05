//! The port's half of `scripts/chat-parse-diff`: the records
//! `harness/chat_parse_ref.cpp` replays through upstream's `common/chat.cpp`,
//! auto-parser and per-model parsers, replayed through `cli/common_chat/`
//! and written as the same transcript.
//!
//! # Provenance
//!
//! **Not a port.** A test driver; it mirrors `harness/chat_parse_ref.cpp`
//! line for line, and the two must stay in step. See that file for the
//! record and transcript formats.
//!
//! Usage: chat_parse_diff <records> <transcript>

const std = @import("std");
const json = @import("json");
const peg = @import("peg");
const jinja = @import("zigjinja");
const chat = @import("common_chat");
const llama = @import("llama");
const peg_serialize = @import("peg_serialize.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Msg = chat.msg.Msg;
const helpers = chat.autoparser.helpers;

const gpa = std.heap.c_allocator;

/// What `chat_parse_ref.cpp` pins `time` and `clock_gettime` to.
const fixed_time: i64 = 1700000000;

var out: std.ArrayList(u8) = .empty;
var logs: std.ArrayList(u8) = .empty;

fn esc(w: *std.ArrayList(u8), s: []const u8) !void {
    for (s) |c| {
        if (c == '\\') {
            try w.appendSlice(gpa, "\\\\");
        } else if (c < 0x20 or c >= 0x7f) {
            try w.print(gpa, "\\x{x:0>2}", .{c});
        } else {
            try w.append(gpa, c);
        }
    }
}

fn line(comptime fmt: []const u8, args: anytype) !void {
    try out.print(gpa, fmt ++ "\n", args);
}

fn kv(key: []const u8, s: []const u8) !void {
    try out.appendSlice(gpa, key);
    try out.append(gpa, ' ');
    try esc(&out, s);
    try out.append(gpa, '\n');
}

fn flag(key: []const u8, b: bool) !void {
    try line("{s} {s}", .{ key, if (b) "1" else "0" });
}

/// The C++ driver's `guarded`: a throw is printed as its class and text.
fn guarded(r: anyerror!void) !void {
    r catch |e| {
        const class = switch (e) {
            error.JsonError => "json_error",
            error.InvalidArgument => "invalid_argument",
            error.RuntimeError => "runtime_error",
            else => return e,
        };
        try out.print(gpa, "E {s} ", .{class});
        try esc(&out, chat.exception.what(e));
        try out.append(gpa, '\n');
    };
}

// -----------------------------------------------------------------------------
// logs, as upstream's common_log writes them in JSONL: one line per call,
// cut at a NUL (`%s`), bad UTF-8 replaced (`dump_safe`)

fn logLine(tag: []const u8, text_in: []const u8) void {
    const text = if (std.mem.indexOfScalar(u8, text_in, 0)) |nul| text_in[0..nul] else text_in;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const v = Value.from(a, text) catch return;
    const dumped = json.dumpSafe(a, &v, -1) catch return;
    var back = json.parse(a, dumped) catch return;
    const clean = back.get([]const u8) catch return;
    logs.print(gpa, "L {s} ", .{tag}) catch return;
    var tmp: std.ArrayList(u8) = .empty;
    esc(&tmp, clean) catch return;
    logs.appendSlice(gpa, tmp.items) catch return;
    tmp.deinit(gpa);
    logs.append(gpa, '\n') catch return;
}

fn chatLog(level: chat.log.Level, text: []const u8) void {
    switch (level) {
        .debug => {},
        .info => logLine("I", text),
        .warn => logLine("W", text),
        .err => logLine("E", text),
    }
}

fn msgLog(level: chat.msg.Level, text: []const u8) void {
    switch (level) {
        .warn => logLine("W", text),
        .err => logLine("E", text),
    }
}

fn flushLogs() !void {
    try out.appendSlice(gpa, logs.items);
    logs.clearRetainingCapacity();
}

fn silent(_: []const u8) void {}

// -----------------------------------------------------------------------------
// records

fn str(v: *const Value, key: []const u8) ![]const u8 {
    return (try v.at(key)).get([]const u8);
}

fn boolean(v: *const Value, key: []const u8) !bool {
    return (try v.at(key)).get(bool);
}

fn int(v: *const Value, key: []const u8) !i64 {
    return (try v.at(key)).get(i64);
}

fn unhex(a: Allocator, h: []const u8) ![]u8 {
    const s = try a.alloc(u8, h.len / 2);
    for (s, 0..) |*c, i| c.* = try std.fmt.parseInt(u8, h[2 * i ..][0..2], 16);
    return s;
}

fn readMsg(a: Allocator, j: *const Value) !Msg {
    var m: Msg = .{};
    m.role = try str(j, "role");
    m.content = try str(j, "content");
    for ((try j.at("parts")).array.items) |*p| {
        try m.content_parts.append(a, .{ .type = try (try p.atIndex(0)).get([]const u8), .text = try (try p.atIndex(1)).get([]const u8) });
    }
    for ((try j.at("tool_calls")).array.items) |*tc| {
        try m.tool_calls.append(a, .{
            .name = try (try tc.atIndex(0)).get([]const u8),
            .arguments = try (try tc.atIndex(1)).get([]const u8),
            .id = try (try tc.atIndex(2)).get([]const u8),
        });
    }
    m.reasoning_content = try str(j, "reasoning_content");
    m.tool_name = try str(j, "tool_name");
    m.tool_call_id = try str(j, "tool_call_id");
    return m;
}

fn readInputs(a: Allocator, j: *const Value) !chat.templates.TemplatesInputs {
    var msgs: std.ArrayList(Msg) = .empty;
    for ((try j.at("messages")).array.items) |*m| try msgs.append(a, try readMsg(a, m));
    var tools: std.ArrayList(chat.msg.Tool) = .empty;
    for ((try j.at("tools")).array.items) |*t| {
        try tools.append(a, .{
            .name = try (try t.atIndex(0)).get([]const u8),
            .description = try (try t.atIndex(1)).get([]const u8),
            .parameters = try (try t.atIndex(2)).get([]const u8),
        });
    }
    var kwargs: std.ArrayList(chat.templates.KwArg) = .empty;
    for ((try j.at("kwargs")).array.items) |*k| {
        try kwargs.append(a, .{ .key = try (try k.atIndex(0)).get([]const u8), .value = try (try k.atIndex(1)).get([]const u8) });
    }
    return .{
        .messages = msgs.items,
        .grammar = try str(j, "grammar"),
        .json_schema = try str(j, "json_schema"),
        .add_generation_prompt = try boolean(j, "add_generation_prompt"),
        .continue_final_message = @fromBackingInt(@intCast(try int(j, "continue"))),
        .use_jinja = try boolean(j, "use_jinja"),
        .tools = tools.items,
        .tool_choice = @fromBackingInt(@intCast(try int(j, "tool_choice"))),
        .parallel_tool_calls = try boolean(j, "parallel"),
        .reasoning_format = @fromBackingInt(@intCast(try int(j, "reasoning_format"))),
        .enable_thinking = try boolean(j, "enable_thinking"),
        .chat_template_kwargs = kwargs.items,
        .add_bos = try boolean(j, "add_bos"),
        .add_eos = try boolean(j, "add_eos"),
        .force_pure_content = try boolean(j, "force_pure_content"),
    };
}

fn readTemplateParams(a: Allocator, j: *const Value) !chat.autoparser.TemplateParams {
    return .{
        .messages = try (try j.at("messages")).clone(a),
        .tools = try (try j.at("tools")).clone(a),
        .add_generation_prompt = try boolean(j, "add_generation_prompt"),
        .enable_thinking = try boolean(j, "enable_thinking"),
        .extra_context = if (j.contains("extra_context")) try (try j.at("extra_context")).clone(a) else null,
    };
}

/// `show_prompt`: the text, then every part, an empty one included.
fn showPrompt(a: Allocator, p: *const chat.prompt.Prompt) !void {
    try kv("prompt", try p.str(a));
    for (p.parts.items) |part| try kv(if (part.is_input) "prompt.input" else "prompt.template", part.text);
}

fn showParams(a: Allocator, p: *const chat.params.ChatParams) !void {
    try kv("format", p.format.name());
    try showPrompt(a, &p.prompt);
    try kv("grammar", p.grammar);
    try flag("lazy", p.grammar_lazy);
    try kv("gen", p.generation_prompt);
    try flag("thinking", p.supports_thinking);
    try kv("think_start", p.thinking_start_tag);
    for (p.thinking_end_tags.items) |t| try kv("think_end", t);
    for (p.grammar_triggers.items) |t| {
        try out.print(gpa, "trigger {d} {d} ", .{ @backingInt(t.type), t.token });
        try esc(&out, t.value);
        try out.append(gpa, '\n');
    }
    for (p.preserved_tokens.items) |t| try kv("preserved", t);
    for (p.additional_stops.items) |t| try kv("stop", t);
    try guarded(showParser(a, p));
    try guarded(showDelims(a, p));
}

fn showParser(a: Allocator, p: *const chat.params.ChatParams) !void {
    const text = if (p.parser) |*arena| (if (arena.empty()) "" else try peg_serialize.save(a, arena)) else "";
    try kv("parser", text);
}

/// The form `common_chat_msg_delimiters::to_json` wrote until v0.7.0.
fn showDelims(a: Allocator, p: *const chat.params.ChatParams) !void {
    var v = Value.emptyArray();
    for (p.message_delimiters.delimiters.items) |d| {
        var o = Value.emptyObject();
        try o.set(a, "role", try Value.from(a, d.role.toString()));
        try o.set(a, "delimiter", try Value.from(a, d.delimiter));
        try v.pushBack(a, o);
    }
    try kv("delims", try json.dump(a, &v, -1));
}

fn showMsg(m: *const Msg) !void {
    try kv("role", m.role);
    try kv("content", m.content);
    for (m.content_parts.items) |p| {
        try kv("part.type", p.type);
        try kv("part.text", p.text);
    }
    for (m.tool_calls.items) |tc| {
        try kv("tc.name", tc.name);
        try kv("tc.arguments", tc.arguments);
        try kv("tc.id", tc.id);
    }
    try kv("reasoning", m.reasoning_content);
    try kv("tool_name", m.tool_name);
    try kv("tool_call_id", m.tool_call_id);
}

fn showAnalysis(ap: *const chat.autoparser.Autoparser) !void {
    try kv("an.user_start", ap.user_start);
    try kv("an.assistant_start", ap.assistant_start);
    try kv("an.reasoning.mode", ap.reasoning.mode.name());
    try kv("an.reasoning.start", ap.reasoning.start);
    try kv("an.reasoning.end", ap.reasoning.end);
    try kv("an.content.mode", ap.content.mode.name());
    try kv("an.content.start", ap.content.start);
    try kv("an.content.end", ap.content.end);
    const f = &ap.tools.format;
    try kv("an.tools.mode", f.mode.name());
    try kv("an.tools.section_start", f.section_start);
    try kv("an.tools.section_end", f.section_end);
    try kv("an.tools.per_call_start", f.per_call_start);
    try kv("an.tools.per_call_end", f.per_call_end);
    try flag("an.tools.fun_name_is_key", f.fun_name_is_key);
    try flag("an.tools.tools_array_wrapped", f.tools_array_wrapped);
    try flag("an.tools.openai_wrapper_trigger", f.openai_wrapper_trigger);
    try kv("an.tools.function_field", f.function_field);
    try kv("an.tools.name_field", f.name_field);
    try kv("an.tools.args_field", f.args_field);
    try kv("an.tools.id_field", f.id_field);
    try kv("an.tools.gen_id_field", f.gen_id_field);
    for (f.parameter_order.items) |p| try kv("an.tools.parameter_order", p);
    const fn_ = &ap.tools.function;
    try kv("an.function.name_prefix", fn_.name_prefix);
    try kv("an.function.name_suffix", fn_.name_suffix);
    try kv("an.function.args_separator", fn_.args_separator);
    try kv("an.function.close", fn_.close);
    const ar = &ap.tools.arguments;
    try kv("an.args.start", ar.start);
    try kv("an.args.end", ar.end);
    try kv("an.args.name_prefix", ar.name_prefix);
    try kv("an.args.name_suffix", ar.name_suffix);
    try kv("an.args.value_prefix", ar.value_prefix);
    try kv("an.args.value_suffix", ar.value_suffix);
    try kv("an.args.separator", ar.separator);
    try flag("an.args.tolerate_intertag_whitespace", ar.tolerate_intertag_whitespace);
    try kv("an.call_id.pos", ap.tools.call_id.pos.name());
    try kv("an.call_id.prefix", ap.tools.call_id.prefix);
    try kv("an.call_id.suffix", ap.tools.call_id.suffix);
    for (ap.preserved_tokens.items) |t| try kv("an.preserved", t);
    for (ap.additional_stops.items) |t| try kv("an.stop", t);
}

fn showDiff(d: chat.autoparser.DiffSplit) !void {
    try kv("prefix", d.prefix);
    try kv("suffix", d.suffix);
    try kv("left", d.left);
    try kv("right", d.right);
}

const Kept = struct {
    params: chat.params.ChatParams,
};

const State = struct {
    tmpls: std.AutoHashMapUnmanaged(i64, chat.templates.Templates) = .empty,
    applies: std.AutoHashMapUnmanaged(i64, Kept) = .empty,
    /// Owns every kept apply's result.
    keep_arena: std.heap.ArenaAllocator = .init(gpa),
};

var models: std.StringHashMapUnmanaged(*anyopaque) = .empty;

fn quietLog(_: c_uint, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {}

fn tmplRecord(st: *State, a: Allocator, r: *const Value) !void {
    const id = try int(r, "id");
    const src = try str(r, "src");
    const bos = try str(r, "bos");
    const eos = try str(r, "eos");
    // With a "model", its vocabulary's BOS, EOS and flags, read through
    // libllama's port, as `common_chat_templates_init` reads them.
    var info: ?chat.templates.ModelInfo = null;
    if (r.contains("model")) {
        const path = try str(r, "model");
        const model = models.get(path) orelse blk: {
            llama.api.impl.llama_log_set(quietLog, null);
            var mp = llama.api.model.llama_model_default_params();
            mp.vocab_only = true;
            const z = try gpa.dupeSentinel(u8, path, 0);
            const m = llama.api.llama.llama_model_load_from_file(z.ptr, mp) orelse return error.ModelLoad;
            // the key outlives the record it was read from
            try models.put(gpa, try gpa.dupe(u8, path), @ptrCast(m));
            break :blk @as(*anyopaque, m);
        };
        info = try chat.templates.modelInfo(gpa, model);
    }
    const t = chat.templates.init(gpa, if (info) |*i| i else null, src, bos, eos) catch |e| {
        if (e == error.OutOfMemory) return e;
        try line("init_failed", .{});
        return;
    };
    try st.tmpls.put(gpa, id, t);
    const tp = st.tmpls.getPtr(id).?;
    for (chat.templates.getCaps(tp)) |e| try line("caps {s}={s}", .{ e.name, if (e.value) "1" else "0" });
    try kv("source", chat.templates.source(tp, ""));
    try kv("tool_use_source", chat.templates.source(tp, "tool_use"));
    try flag("explicit", chat.templates.wasExplicit(tp));
    const asr = chat.templates.getAsrPrompt(tp);
    try kv("asr.system", asr.system);
    try kv("asr.user", asr.user);
    try guarded(blk: {
        const b = chat.templates.supportEnableThinking(gpa, tp) catch |e| break :blk e;
        break :blk flag("enable_thinking", b);
    });
    try guarded(blk: {
        const ex = chat.templates.formatExample(a, tp, true, &.{}) catch |e| break :blk e;
        break :blk kv("example", ex);
    });
    try flag("verify", try chat.templates.verifyTemplate(gpa, src, true));
    try guarded(analysis(a, src, bos, eos));
}

fn analysis(a: Allocator, src: []const u8, bos: []const u8, eos: []const u8) !void {
    var tmpl = try chat.template.Template.init(gpa, src, bos, eos);
    defer tmpl.deinit();
    var ap: chat.autoparser.Autoparser = .{};
    try chat.autoparser.analyzeTemplate(&ap, a, &tmpl);
    try showAnalysis(&ap);
}

fn applyRecord(st: *State, a: Allocator, n: usize, r: *const Value) !void {
    const tp = st.tmpls.getPtr(try int(r, "t")) orelse return line("no_templates", .{});
    const keep = r.contains("keep") and try boolean(r, "keep");
    const pa = if (keep) st.keep_arena.allocator() else a;
    try guarded(blk: {
        const in = readInputs(pa, try r.at("in")) catch |e| break :blk e;
        const params = chat.templates.apply(pa, tp, &in) catch |e| break :blk e;
        showParams(a, &params) catch |e| break :blk e;
        if (keep) st.applies.put(gpa, @intCast(n), .{ .params = params }) catch |e| break :blk e;
    });
}

fn parseRecord(st: *State, a: Allocator, r: *const Value) !void {
    const text = try unhex(a, try str(r, "input_hex"));
    const partial = try boolean(r, "partial");
    const kept = st.applies.getPtr(try int(r, "a")) orelse return line("no_apply", .{});
    // `common_chat_parser_params(params)`: the format, and the generation
    // prompt as input without tokens
    var gen = try chat.input.Input.fromText(a, kept.params.generation_prompt);
    const params: chat.parse.ParserParams = .{ .format = kept.params.format, .generation_prompt = &gen };
    var input = try chat.input.Input.fromText(a, text);
    const arena: ?*const peg.Arena = if (kept.params.parser) |*p| p else null;
    try guarded(blk: {
        var m = chat.parse.pegParse(gpa, arena, &input, partial, &params) catch |e| break :blk e;
        defer m.deinit(gpa);
        break :blk showMsg(&m);
    });
}

fn cvRecord(a: Allocator, r: *const Value) !void {
    var tmpl = try chat.template.Template.init(gpa, try str(r, "src"), try str(r, "bos"), try str(r, "eos"));
    defer tmpl.deinit();
    const pa = try readTemplateParams(a, try r.at("a"));
    const pb = try readTemplateParams(a, try r.at("b"));
    const res = try helpers.compareVariants(a, &tmpl, &pa, &pb) orelse return line("none", .{});
    try showDiff(res.diff);
    try kv("output_A", res.output_A);
    try kv("output_B", res.output_B);
}

fn run(st: *State, a: Allocator, n: usize, r: *const Value) !void {
    const op = try str(r, "op");
    try line("== {d} {s}", .{ n, op });
    if (std.mem.eql(u8, op, "tmpl")) {
        try tmplRecord(st, a, r);
    } else if (std.mem.eql(u8, op, "apply")) {
        try applyRecord(st, a, n, r);
    } else if (std.mem.eql(u8, op, "parse")) {
        try parseRecord(st, a, r);
    } else if (std.mem.eql(u8, op, "diff")) {
        try showDiff(try helpers.calculateDiffSplit(a, try str(r, "l"), try str(r, "r")));
    } else if (std.mem.eql(u8, op, "ucp")) {
        try kv("r", helpers.untilCommonPrefix(try str(r, "full"), try str(r, "l"), try str(r, "r")));
    } else if (std.mem.eql(u8, op, "acs")) {
        try kv("r", helpers.afterCommonSuffix(try str(r, "full"), try str(r, "l"), try str(r, "r")));
    } else if (std.mem.eql(u8, op, "seg")) {
        for (try helpers.segmentizeMarkers(a, try str(r, "text"))) |s| try kv(s.type.name(), s.value);
    } else if (std.mem.eql(u8, op, "cv")) {
        try guarded(cvRecord(a, r));
    } else {
        try line("unknown_op", .{});
    }
    try flushLogs();
}

const FILE = opaque {};
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*FILE;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *FILE) usize;
extern "c" fn fflush(f: *FILE) c_int;
extern "c" fn fclose(f: *FILE) c_int;

/// Whether a record before the start of a resumed run rebuilds state the
/// records after it read: templates, kept applies.
fn buildsState(r: *const Value) bool {
    const op = str(r, "op") catch return false;
    if (std.mem.eql(u8, op, "tmpl")) return true;
    return std.mem.eql(u8, op, "apply") and r.contains("keep");
}

/// Usage: chat_parse_diff <records> <transcript> [start]. From `start` the
/// transcript is appended to, the records before it replayed without output
/// where they build state; `CHAT_PARSE_SKIP` lists records (comma-separated)
/// a previous run crashed on, never run again. Each record's lines are
/// written as it finishes, so a crash leaves every finished record behind.
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 4) {
        std.debug.print("usage: {s} <records> <transcript> [start]\n", .{args[0]});
        std.process.exit(2);
    }
    const start: usize = if (args.len == 4) try std.fmt.parseInt(usize, args[3], 10) else 0;
    var skip: std.AutoHashMapUnmanaged(usize, void) = .empty;
    if (init.environ_map.get("CHAT_PARSE_SKIP")) |list| {
        var it = std.mem.tokenizeScalar(u8, list, ',');
        while (it.next()) |x| try skip.put(gpa, try std.fmt.parseInt(usize, x, 10), {});
    }
    chat.template.now_override = fixed_time;
    jinja.time.fixed_timestamp = fixed_time;
    chat.log.sink = chatLog;
    chat.msg.log_sink = msgLog;
    json.schema_to_grammar.warning_sink = silent;

    const io = init.io;
    const records = try std.Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .unlimited);
    defer gpa.free(records);
    const path = try gpa.dupeSentinel(u8, args[2], 0);
    const file = fopen(path, if (start == 0) "wb" else "ab") orelse return error.CannotOpen;

    var st: State = .{};
    var lines = std.mem.splitScalar(u8, records, '\n');
    var n: usize = 0;
    while (lines.next()) |text| {
        if (text.len == 0) continue;
        defer n += 1;
        if (skip.contains(n)) continue;
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const a = arena_state.allocator();
        const r = try json.parse(a, text);
        if (n < start and !buildsState(&r)) continue;
        try run(&st, a, n, &r);
        if (n >= start) {
            _ = fwrite(out.items.ptr, 1, out.items.len, file);
            _ = fflush(file);
        }
        out.clearRetainingCapacity();
    }
    _ = fclose(file);
}
