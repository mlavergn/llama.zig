//! A chat template, and rendering one over a conversation.
//!
//! # Provenance
//!
//! Ports `common_chat_template` (llama.cpp v0.6.0, `d81235049`,
//! `llama.cpp/common/chat.h`), the rendering half of
//! `llama.cpp/common/chat.cpp` -- `format_time`,
//! `common_chat_template_direct_apply_impl`,
//! `common_chat_template_generation_prompt_impl` and their two wrappers,
//! `common_chat_extra_context` -- `autoparser`'s `generation_params` from
//! `llama.cpp/common/chat-auto-parser.h`, the source normalisation of
//! `lexer`'s `tokenize` in `llama.cpp/common/jinja/lexer.cpp`, and
//! `jinja`'s `global_from_json` and its `from_json` in
//! `llama.cpp/common/jinja/value.cpp`. Each declaration names what it ports
//! and the line it began at.
//!
//! # The engine
//!
//! Upstream renders through its own `common/jinja`; this renders through
//! `zigjinja`, set up as `cli/chat.zig` sets it up -- both block options on,
//! `upstreamOutput` as the finalizer, the AST path -- which is what `make
//! jinja-diff` holds to upstream, and with every bound string marked input,
//! as `global_from_json(ctx, json, true)` marks it: zigjinja follows
//! llama.cpp's builtins for a marked string, so marking is part of getting
//! the same text, not only of trusting it. The marks are dropped from the
//! result; upstream's `common_chat_params`'s `prompt` is a plain string too.
//!
//! A string bound as input loses any provenance sentinel byte it held
//! (`cli/chat.zig`'s `sanitize`), where upstream keeps it; a template whose
//! own text holds one is refused. Those three bytes are the one place the
//! text can differ.
//!
//! A render that fails is `error.RuntimeError` with `what()` beginning
//! `jinja_failure_prefix`: upstream's text there is its own engine's
//! message, which zigjinja does not reproduce.

const std = @import("std");
const json = @import("json");
const jinja = @import("zigjinja");
const exception = @import("exception.zig");
const msg_mod = @import("msg.zig");
const caps_mod = @import("caps.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const throw = exception.throw;

pub const Caps = caps_mod.Caps;

/// What a failed render's `what()` begins with; see the file's header.
pub const jinja_failure_prefix = "jinja: ";

// -----------------------------------------------------------------------------
// generation_params

/// Ports `autoparser`'s `generation_params` (chat-auto-parser.h:54
/// @d81235049): what a template is rendered with and a parser built for.
/// Every JSON value and string is owned by the allocator of the call that
/// built it, an arena.
pub const GenerationParams = struct {
    messages: Value = .null,
    tools: Value = .null,
    tool_choice: msg_mod.ToolChoice = .auto,
    json_schema: Value = .null,
    parallel_tool_calls: bool = true,
    reasoning_format: msg_mod.ReasoningFormat = .auto,
    stream: bool = true,
    grammar: []const u8 = "",
    add_generation_prompt: bool = false,
    continue_final_message: msg_mod.Continuation = .none,
    continue_msg: msg_mod.Msg = .{},
    enable_thinking: bool = true,
    extra_context: Value = .null,
    add_bos: bool = false,
    add_eos: bool = false,
    is_inference: bool = true,
    add_inference: bool = false,
    mark_input: bool = true,

    /// Ports `has_continuation` (chat-auto-parser.h:75 @d81235049).
    pub fn hasContinuation(self: *const GenerationParams) bool {
        return self.continue_final_message != .none and !self.continue_msg.empty();
    }

    /// A copy whose JSON is `a`'s, as the C++ copy constructor makes one.
    /// The strings and the message are shared.
    pub fn clone(self: *const GenerationParams, a: Allocator) Error!GenerationParams {
        var c = self.*;
        c.messages = try self.messages.clone(a);
        c.tools = try self.tools.clone(a);
        c.json_schema = try self.json_schema.clone(a);
        c.extra_context = try self.extra_context.clone(a);
        return c;
    }
};

// -----------------------------------------------------------------------------
// common_chat_template

/// Ports `lexer`'s `tokenize`'s source normalisation (lexer.cpp:32
/// @d81235049): `\r\n` and `\r` as `\n`, then one trailing newline of the
/// original dropped. This is `common_chat_template::source()`, which every
/// template-detection `find` reads.
///
/// Return: owned by `a`.
pub fn normalizeSource(a: Allocator, source: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(a, source.len);
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        const ch = source[i];
        if (ch == '\r') {
            if (i + 1 < source.len and source[i + 1] == '\n') continue;
            out.appendAssumeCapacity('\n');
        } else out.appendAssumeCapacity(ch);
    }
    if (source.len > 0 and source[source.len - 1] == '\n') _ = out.pop();
    return out.toOwnedSlice(a);
}

/// Ports `common_chat_template` (chat.h:51 @d81235049): a template's source
/// parsed, its BOS and EOS spellings, and the capabilities `caps_get` reads
/// off it.
pub const Template = struct {
    gpa: Allocator,
    /// The source as given, which zigjinja renders.
    raw: []u8,
    /// `source()`: the lexer's normalised source.
    src: []u8,
    bos_tok: []u8,
    eos_tok: []u8,
    caps: Caps,

    /// Ports `common_chat_template`'s constructor (chat.h:58 @d81235049):
    /// parses the source, failing as upstream's lexer and parser fail, and
    /// runs `caps_get` over it.
    ///
    /// Return: owned by `gpa`; release with `deinit`. `error.RuntimeError`
    /// when the source does not parse, its `what()` zigjinja's rather than
    /// upstream's.
    pub fn init(gpa: Allocator, text: []const u8, bos_token: []const u8, eos_token: []const u8) Error!Template {
        var self: Template = .{
            .gpa = gpa,
            .raw = try gpa.dupe(u8, text),
            .src = &.{},
            .bos_tok = &.{},
            .eos_tok = &.{},
            .caps = .{},
        };
        errdefer self.deinit();
        self.src = try normalizeSource(gpa, text);
        self.bos_tok = try gpa.dupe(u8, bos_token);
        self.eos_tok = try gpa.dupe(u8, eos_token);
        try checkParses(gpa, self.raw);
        self.caps = try caps_mod.capsGet(&self);
        return self;
    }

    pub fn deinit(self: *Template) void {
        self.gpa.free(self.raw);
        self.gpa.free(self.src);
        self.gpa.free(self.bos_tok);
        self.gpa.free(self.eos_tok);
        self.* = undefined;
    }

    /// Ports `source` (chat.h:71 @d81235049).
    pub fn source(self: *const Template) []const u8 {
        return self.src;
    }

    /// Ports `bos_token` (chat.h:72 @d81235049).
    pub fn bosToken(self: *const Template) []const u8 {
        return self.bos_tok;
    }

    /// Ports `eos_token` (chat.h:73 @d81235049).
    pub fn eosToken(self: *const Template) []const u8 {
        return self.eos_tok;
    }

    /// Ports `original_caps` (chat.h:75 @d81235049).
    pub fn originalCaps(self: *const Template) Caps {
        return self.caps;
    }
};

/// Whether `source` parses, as `jinja`'s `parse_from_tokens` decides.
fn checkParses(gpa: Allocator, text: []const u8) Error!void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (std.mem.findAny(u8, text, &jinja.provenance.sentinels) != null)
        return throw(error.RuntimeError, jinja_failure_prefix ++ "the template holds a provenance sentinel byte", .{});
    var env = jinja.Environment.init(arena);
    defer env.deinit();
    env.trim_blocks = true;
    env.lstrip_blocks = true;
    // the parse itself follows llama.cpp's grammar: unary `+`/`-` nest and
    // bind tighter than a filter (zigjinja, `Parser.llama_cpp_semantics`)
    env.llama_cpp_semantics = true;
    _ = env.fromString(text, "chat") catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return throw(error.RuntimeError, jinja_failure_prefix ++ "parse: {s} near line {d}", .{ @errorName(e), env.last_syntax_error_line }),
    };
}

// -----------------------------------------------------------------------------
// Rendering

/// An allocator that never gives memory back, so no address is reused
/// while a usage-statistics record keys on addresses (`zigjinja/src/stats.zig`).
const NoFree = struct {
    child: Allocator,

    fn allocator(self: *NoFree) Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *NoFree = @ptrCast(@alignCast(ctx));
        return self.child.rawAlloc(len, alignment, ret_addr);
    }

    fn resize(_: *anyopaque, memory: []u8, _: std.mem.Alignment, new_len: usize, _: usize) bool {
        return new_len <= memory.len;
    }

    fn remap(_: *anyopaque, memory: []u8, _: std.mem.Alignment, new_len: usize, _: usize) ?[*]u8 {
        return if (new_len <= memory.len) memory.ptr else null;
    }

    fn free(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {}
};

/// Upstream's rendering of a non-string `{{ }}` value; `cli/chat.zig`'s
/// `upstreamOutput`, which `make jinja-diff` holds to
/// `gather_string_parts_recursive` (runtime.h:717 @d81235049).
fn upstreamOutput(allocator: Allocator, val: jinja.Value) jinja.Value {
    return switch (val) {
        .boolean => |b| .{ .string = if (b) "True" else "False" },
        .null, .undefined, .dict => .{ .string = "" },
        .list => gatherList(allocator, val) catch val,
        // a float's `as_string`: `std::to_string`, trailing zeros trimmed
        .float => .{ .string = jinja.llama_repr.LlamaRepr.asString(allocator, val) catch return val },
        else => val,
    };
}

fn gatherList(allocator: Allocator, val: jinja.Value) !jinja.Value {
    var pieces: std.ArrayList([]const u8) = .empty;
    try gather(allocator, val, &pieces);
    if (jinja.provenance.active) |record| {
        if (try record.concat(pieces.items)) |joined| return .{ .string = joined };
    }
    return .{ .string = try std.mem.concat(allocator, u8, pieces.items) };
}

fn gather(allocator: Allocator, val: jinja.Value, pieces: *std.ArrayList([]const u8)) !void {
    switch (val) {
        .list => |list| for (list.items.items) |item| try gather(allocator, item, pieces),
        .string => |str| try pieces.append(allocator, str),
        .boolean => |b| try pieces.append(allocator, if (b) "True" else "False"),
        .integer => try pieces.append(allocator, try val.toString(allocator)),
        .float => try pieces.append(allocator, try jinja.llama_repr.LlamaRepr.asString(allocator, val)),
        else => {},
    }
}

/// One render: the variables bound, in order, and the engine run once.
/// Ports the shape of a `jinja`'s `context` filled by `set_val` and
/// `global_from_json`, then `runtime`'s `execute` and `gather_string_parts`.
pub const Render = struct {
    arena_state: std.heap.ArenaAllocator,
    no_free: NoFree,
    record: jinja.provenance.Provenance,
    vars: std.StringHashMap(jinja.Value),
    /// The marked runs of the last `run`'s text, kept only while
    /// `mark_capture` is set; owned by the capture's allocator.
    marked_parts: []const jinja.provenance.Part,

    /// Parameters:
    /// - `gpa`: backs everything the render allocates.
    /// - `keep_addresses`: never free during the render, for a statistics record.
    ///
    /// Return: release with `deinit`. Pinned: take its address only after it is in place.
    pub fn init(self: *Render, gpa: Allocator, keep_addresses: bool) void {
        self.arena_state = std.heap.ArenaAllocator.init(gpa);
        self.no_free = .{ .child = self.arena_state.allocator() };
        const a = if (keep_addresses) self.no_free.allocator() else self.arena_state.allocator();
        self.record = jinja.provenance.Provenance.init(gpa);
        self.vars = std.StringHashMap(jinja.Value).init(a);
        self.marked_parts = &.{};
    }

    pub fn deinit(self: *Render) void {
        self.record.deinit();
        self.arena_state.deinit();
    }

    /// The allocator every value of this render comes from.
    pub fn alloc(self: *Render) Allocator {
        return self.vars.allocator;
    }

    /// `ctx.set_val(name, v)`. A variable named "" is not bound: no
    /// template can name it, and zigjinja spells the missing `else` of
    /// `a if b` as a lookup of that name -- so a `chat_template_kwargs`
    /// member "" (a request whose kwargs are a string has one) would
    /// otherwise fill every such branch.
    pub fn set(self: *Render, name: []const u8, v: jinja.Value) Error!void {
        if (name.len == 0) return;
        try self.vars.put(try self.alloc().dupe(u8, name), v);
    }

    /// `mk_val<value_string>(s)`: a string, not input.
    pub fn string(self: *Render, s: []const u8) Error!jinja.Value {
        return .{ .string = try self.alloc().dupe(u8, try sanitize(self.alloc(), s)) };
    }

    /// Ports `from_json` (value.cpp:1412 @d81235049): a JSON value as the
    /// engine's, every string marked input when `mark_input`.
    pub fn fromJson(self: *Render, v: *const Value, mark_input: bool) Error!jinja.Value {
        const a = self.alloc();
        return switch (v.*) {
            .null, .discarded => .null,
            .boolean => |b| .{ .boolean = b },
            .number_integer => |n| .{ .integer = n },
            .number_unsigned => |n| .{ .integer = @bitCast(n) },
            .number_float => |f| .{ .float = f },
            .string => |s| blk: {
                const clean = try sanitize(a, s);
                break :blk .{ .string = if (mark_input) try self.record.markInput(clean) else try a.dupe(u8, clean) };
            },
            .array => |arr| blk: {
                const list = try a.create(jinja.value.List);
                list.* = jinja.value.List.init(a);
                for (arr.items) |*item| try list.append(try self.fromJson(item, mark_input));
                break :blk .{ .list = list };
            },
            .object => |m| blk: {
                const dict = try a.create(jinja.value.Dict);
                dict.* = jinja.value.Dict.init(a);
                for (m.members.items) |*member| try dict.set(member.key, try self.fromJson(&member.value, mark_input));
                break :blk .{ .dict = dict };
            },
        };
    }

    /// Ports `global_from_json` (value.cpp:1509 @d81235049): each member of
    /// the object bound in order, a later one replacing an earlier binding.
    pub fn bindJson(self: *Render, obj: *const Value, mark_input: bool) Error!void {
        if (!obj.isObject()) return throw(error.RuntimeError, "global_from_json: input JSON value must be an object", .{});
        for (obj.object.members.items) |*member| try self.set(member.key, try self.fromJson(&member.value, mark_input));
    }

    /// Runs `tmpl` over the bound variables.
    ///
    /// Return: the text, owned by `out`. `error.RuntimeError` with
    /// `jinja_failure_prefix` when the template fails or raises.
    pub fn run(self: *Render, tmpl: *const Template, out: Allocator) Error![]u8 {
        const a = self.alloc();
        if (std.mem.findAny(u8, tmpl.raw, &jinja.provenance.sentinels) != null)
            return throw(error.RuntimeError, jinja_failure_prefix ++ "the template holds a provenance sentinel byte", .{});
        var env = jinja.Environment.init(a);
        defer env.deinit();
        env.finalize = upstreamOutput;
        env.trim_blocks = true;
        env.lstrip_blocks = true;
        // tojson, sort and a container as a string, as llama.cpp writes them.
        env.llama_cpp_semantics = true;
        const parsed = env.fromString(tmpl.raw, "chat") catch |e| return failure(e);

        jinja.provenance.active = &self.record;
        defer jinja.provenance.active = null;

        var ctx = jinja.context.Context.init(&env, self.vars, "chat", a) catch |e| return failure(e);
        defer ctx.deinit();
        var compiler_inst = jinja.compiler.Compiler.init(&env, "chat", a);
        defer compiler_inst.deinit();
        var compiled = compiler_inst.compile(parsed, false) catch |e| return failure(e);
        defer compiled.deinit();
        const rendered = compiled.render(&ctx, a) catch |e| return failure(e);
        const text = self.record.decode(rendered) catch |e| return failure(e);
        if (mark_capture) |cap| {
            var one: [1]jinja.provenance.Part = undefined;
            self.marked_parts = try cap.allocator.dupe(jinja.provenance.Part, self.record.partsOrTemplate(text, &one));
        }
        return out.dupe(u8, text);
    }
};

fn failure(e: anyerror) Error {
    if (e == error.OutOfMemory) return error.OutOfMemory;
    return throw(error.RuntimeError, jinja_failure_prefix ++ "{s}", .{@errorName(e)});
}

/// `cli/chat.zig`'s `sanitize`: the provenance sentinels removed from text
/// about to be bound, so it cannot open or close an input run.
fn sanitize(a: Allocator, text: []const u8) Error![]const u8 {
    const sentinels = &jinja.provenance.sentinels;
    if (std.mem.findAny(u8, text, sentinels) == null) return text;
    var out: std.ArrayList(u8) = .empty;
    for (text) |ch| {
        if (std.mem.findScalar(u8, sentinels, ch) != null) continue;
        try out.append(a, ch);
    }
    return out.items;
}

/// Ports `caps_apply_preserve_reasoning` (caps.cpp:22 @d81235049).
pub fn capsApplyPreserveReasoning(r: *Render, enabled: bool) Error!void {
    try r.set("preserve_thinking", .{ .boolean = enabled });
    try r.set("clear_thinking", .{ .boolean = !enabled });
    try r.set("truncate_history_thinking", .{ .boolean = !enabled });
    try r.set("drop_thinking", .{ .boolean = !enabled });
}

/// Ports `caps_apply_reasoning_effort` (caps.cpp:29 @d81235049): one value
/// under both names, so a use of either counts against it.
///
/// Return: the value bound.
pub fn capsApplyReasoningEffort(r: *Render, effort: []const u8) Error!jinja.Value {
    const v = try r.string(effort);
    try r.set("reasoning_effort", v);
    try r.set("reasoning_strength", v);
    return v;
}

// -----------------------------------------------------------------------------
// direct_apply

/// What a time stamp reads as, for `format_time` and the gate that pins it.
/// Null reads the clock.
pub var now_override: ?i64 = null;

/// `std::chrono::system_clock::now()` in whole seconds.
pub fn now() i64 {
    return now_override orelse jinja.time.timestamp();
}

const month_abbrev = [_][]const u8{ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

/// Ports `format_time` (chat.cpp:36 @d81235049) for the two formats it is
/// given, `%b %d %Y` and `%d %b %Y`, in the C locale and local time.
///
/// Return: owned by `a`.
pub fn formatTime(a: Allocator, t: i64, comptime day_first: bool) Error![]u8 {
    const local = t + jinja.localtime.LocalTime.utcOffset(a, t);
    const secs: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(local, 0)) };
    const year_day = secs.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const month = month_abbrev[@backingInt(month_day.month) - 1];
    const day = month_day.day_index + 1;
    return if (day_first)
        std.fmt.allocPrint(a, "{d:0>2} {s} {d}", .{ day, month, year_day.year })
    else
        std.fmt.allocPrint(a, "{s} {d:0>2} {d}", .{ month, day, year_day.year });
}

/// Ports `common_chat_extra_context` (chat.cpp:1080 @d81235049):
/// `datetime` and `date_string`, today's date two ways.
///
/// Return: an object owned by `a`.
pub fn extraContext(a: Allocator) Error!Value {
    var ctx = Value.emptyObject();
    const t = now();
    try ctx.set(a, "datetime", try Value.from(a, try formatTime(a, t, false)));
    try ctx.set(a, "date_string", try Value.from(a, try formatTime(a, t, true)));
    return ctx;
}

/// Ports `common_chat_template_direct_apply_impl` (chat.cpp:905
/// @d81235049): `inputs` rendered through `tmpl`, each override replacing
/// what `inputs` would have bound, and a BOS or EOS the vocabulary adds
/// itself cut from the ends.
///
/// Parameters:
/// - `a`: an arena; the result and every temporary are its.
/// - `messages_override`: messages already built for this template; when
///   null, `inputs.messages` normalised for the template's content caps.
/// - `tools_override`, `additional_context`: likewise.
///
/// Return: the text.
pub fn directApplyImpl(
    a: Allocator,
    tmpl: *const Template,
    inputs: *const GenerationParams,
    messages_override: ?*const Value,
    tools_override: ?*const Value,
    additional_context: ?*const Value,
) Error![]u8 {
    var inp = Value.emptyObject();
    if (messages_override) |m| {
        try inp.set(a, "messages", try m.clone(a));
    } else {
        try inp.set(a, "messages", try msg_mod.normalizeMessages(a, tmpl.caps.contentCaps(), &inputs.messages));
    }
    try inp.set(a, "bos_token", try Value.from(a, tmpl.bos_tok));
    try inp.set(a, "eos_token", try Value.from(a, tmpl.eos_tok));
    try inp.set(a, "enable_thinking", try Value.from(a, inputs.enable_thinking));
    if (tools_override != null or !inputs.tools.empty()) {
        try inp.set(a, "tools", try (tools_override orelse &inputs.tools).clone(a));
    }
    if (inputs.extra_context.isObject()) {
        for (inputs.extra_context.object.members.items) |*m| try inp.set(a, m.key, try m.value.clone(a));
    }
    if (additional_context) |extra| {
        var it = @constCast(extra).items();
        while (it.next()) |item| try inp.set(a, item.key, try item.value.clone(a));
    }
    if (inputs.add_generation_prompt) {
        try inp.set(a, "add_generation_prompt", try Value.from(a, true));
    }

    var r: Render = undefined;
    r.init(a, false);
    defer r.deinit();
    if (inp.contains("preserve_reasoning")) {
        const v = try inp.at("preserve_reasoning");
        if (v.isBoolean()) try capsApplyPreserveReasoning(&r, v.boolean);
    }
    if (inp.contains("reasoning_effort")) {
        const v = try inp.at("reasoning_effort");
        if (v.isString() and !v.empty()) _ = try capsApplyReasoningEffort(&r, v.string);
    }
    try r.bindJson(&inp, inputs.mark_input);
    if (mark_capture != null) try markTokens(&r, &inp);

    const full = try r.run(tmpl, a);
    var result = full;
    if (inputs.add_bos and std.mem.startsWith(u8, result, tmpl.bos_tok)) {
        result = result[tmpl.bos_tok.len..];
    }
    if (inputs.add_eos and std.mem.endsWith(u8, result, tmpl.eos_tok)) {
        result = result[0 .. result.len - tmpl.eos_tok.len];
    }
    if (mark_capture) |cap| {
        const lo = @intFromPtr(result.ptr) - @intFromPtr(full.ptr);
        try cap.renders.append(cap.allocator, .{
            .text = result,
            .parts = try clipParts(cap.allocator, r.marked_parts, lo, lo + result.len),
        });
    }
    return result;
}

// -----------------------------------------------------------------------------
// Trust marks, for a host that tokenizes by them

/// **Not a port.** Upstream's `common_chat_params::prompt` is a plain
/// string, and its server tokenizes the whole of it with `parse_special`.
/// `llama-cli` does not (SPEC.md 5.2): it tokenizes every run of message
/// text with `parse_special` off, so it needs the marks the render drops.
/// While this is set, every `directApplyImpl` records its text and the runs
/// the engine marked as input, and binds `bos_token` and `eos_token` as
/// provenance tokens -- input to every propagation rule, so the text is the
/// same; the host decides what an exact token run is worth, as `cli/chat.zig`
/// does. Null, the default and every other host's, records nothing.
pub threadlocal var mark_capture: ?*MarkCapture = null;

/// What `mark_capture` collects over one `templates.apply`.
pub const MarkCapture = struct {
    /// Owns `renders` and each record's `parts`; an arena in practice.
    allocator: Allocator,
    /// Every render, in the order `directApplyImpl` returned them -- the
    /// analysis probes first, the prompt's own last.
    renders: std.ArrayList(Marked) = .empty,
};

/// One render's text and its runs, which cover it exactly.
pub const Marked = struct {
    /// Borrowed from the render's arena.
    text: []const u8,
    parts: []const jinja.provenance.Part,
};

/// `bos_token` and `eos_token` as provenance tokens, when they are the
/// strings `inp` bound -- a kwarg of either name replaces them, and stays
/// plain input.
fn markTokens(r: *Render, inp: *const Value) Error!void {
    for ([_][]const u8{ "bos_token", "eos_token" }) |name| {
        const v = inp.at(name) catch continue;
        if (!v.isString()) continue;
        try r.set(name, .{ .string = try r.record.markToken(try sanitize(r.alloc(), v.string)) });
    }
}

/// The runs of `parts` that fall in `[lo, hi)`, clipped to it.
fn clipParts(a: Allocator, parts: []const jinja.provenance.Part, lo: usize, hi: usize) Error![]const jinja.provenance.Part {
    var out: std.ArrayList(jinja.provenance.Part) = .empty;
    var at: usize = 0;
    for (parts) |p| {
        const start = @max(at, lo);
        const end = @min(at + p.len, hi);
        at += p.len;
        if (end <= start) continue;
        var q = p;
        q.len = end - start;
        try out.append(a, q);
    }
    return out.items;
}

/// Ports `common_chat_template_direct_apply` (chat.cpp:968 @d81235049).
pub fn directApply(a: Allocator, tmpl: *const Template, inputs: *const GenerationParams) Error![]u8 {
    return directApplyImpl(a, tmpl, inputs, null, null, null);
}

/// Ports `common_chat_template_generation_prompt_impl` (chat.cpp:974
/// @d81235049): what `add_generation_prompt` appends, as the difference
/// between a render without it and one with.
///
/// Return: owned by `a`.
pub fn generationPromptImpl(
    a: Allocator,
    tmpl: *const Template,
    inputs: *const GenerationParams,
    messages_override: ?*const Value,
    tools_override: ?*const Value,
    additional_context: ?*const Value,
) Error![]u8 {
    var params = inputs.*;
    params.add_generation_prompt = false;
    params.continue_final_message = .none;
    const no_gen_prompt = try directApplyImpl(a, tmpl, &params, messages_override, tools_override, additional_context);
    params.add_generation_prompt = true;
    const gen_prompt = try directApplyImpl(a, tmpl, &params, messages_override, tools_override, additional_context);
    var prefix_len: usize = 0;
    const min_size = @min(no_gen_prompt.len, gen_prompt.len);
    while (prefix_len < min_size and no_gen_prompt[prefix_len] == gen_prompt[prefix_len]) prefix_len += 1;
    return gen_prompt[prefix_len..];
}

/// Ports `common_chat_template_generation_prompt` (chat.cpp:996 @d81235049).
pub fn generationPrompt(a: Allocator, tmpl: *const Template, inputs: *const GenerationParams) Error![]u8 {
    return generationPromptImpl(a, tmpl, inputs, null, null, null);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the lexer's source drops one trailing newline and folds carriage returns" {
    const a = std.testing.allocator;
    const s = try normalizeSource(a, "a\r\nb\rc\n\n");
    defer a.free(s);
    try std.testing.expectEqualStrings("a\nb\nc\n", s);
}

test "a template renders its messages" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var t = try Template.init(std.testing.allocator, "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<a>{% endif %}", "<s>", "</s>");
    defer t.deinit();
    var p: GenerationParams = .{ .add_generation_prompt = true };
    p.messages = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]");
    try std.testing.expectEqualStrings("<user>hi<a>", try directApply(a, &t, &p));
    try std.testing.expectEqualStrings("<a>", try generationPrompt(a, &t, &p));
}
