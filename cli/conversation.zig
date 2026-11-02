//! The chat side of `llama-cli`: a conversation rendered through the model's
//! template by `cli/common_chat/`, each reply read back by the parser the
//! template asked for, and the reasoning shown apart from the answer.
//!
//! # Provenance
//!
//! **Mostly not a port.** Upstream's `llama-cli` (v0.5.0, `7fe450e19`) is a
//! client of a `llama-server` it runs in-process: it posts the conversation
//! to `/v1/chat/completions` and prints the `reasoning_content` and `content`
//! deltas of the stream (`llama.cpp/tools/cli/cli-context.cpp`). This file
//! does in one process what that pair does across HTTP:
//!
//! - **the request**: what upstream's server makes of a body holding only
//!   `messages` -- `Chat.init` and `Chat.inputs`, after
//!   `oaicompat_chat_params_parse` (tools/server/server-common.cpp:1151 @7fe450e19)
//!   and the chat options `load_model` (tools/server/server-context.cpp:1006 @7fe450e19)
//!   sets from the command line;
//! - **the stream**: every piece of reply parsed partially and diffed
//!   against the last parse, then parsed whole at the end -- `Reply`, after
//!   `update_chat_msg` (tools/server/server-task.cpp:162 @7fe450e19);
//! - **the display**: `ui::assistant_turn`, whose port is `Display`.
//!
//! One deliberate difference: upstream's server tokenizes the rendered
//! prompt whole with `parse_special`, so a message spelling a control token
//! becomes one. This tokenizes every run the template marked as message
//! text with `parse_special` off (SPEC.md 5.2), using the marks
//! `template.zig` records under `mark_capture`.

const std = @import("std");
const common_chat = @import("common_chat");
const json = @import("json");
const args_mod = @import("args.zig");
const chat = @import("chat.zig");
const session_mod = @import("session.zig");
const reasoning_budget = @import("reasoning_budget.zig");

const c = session_mod.c;
const Allocator = std.mem.Allocator;
const Msg = common_chat.Msg;
const templates = common_chat.templates;
const template = common_chat.template;
const KwArg = templates.KwArg;

/// Failures of the chat layer, told apart from the session's.
pub const Error = error{
    /// The template's render could not be matched with its trust marks.
    MarkingLost,
    /// The template produced nothing from a non-empty conversation; see
    /// `chat.Error.EmptyRender`.
    EmptyRender,
    /// The template could not be loaded or applied; `last_error` says why.
    ChatFailed,
};

/// The `what()` of the last `ChatFailed`, or of a kwarg that did not parse.
pub var last_error: std.ArrayList(u8) = .empty;

/// Where `last_error` lives: it outlives every chat.
const error_allocator = std.heap.smp_allocator;

fn fail(e: anyerror) anyerror {
    if (e == error.OutOfMemory) return e;
    setError(common_chat.what(e)) catch return error.OutOfMemory;
    return Error.ChatFailed;
}

fn setError(text: []const u8) !void {
    last_error.clearRetainingCapacity();
    try last_error.appendSlice(error_allocator, text);
}

// -----------------------------------------------------------------------------
// The kwargs

/// Replays the command line's writes to upstream's `default_template_kwargs`
/// -- a `std::map` from a name to the JSON text of its value -- and adds the
/// `preserve_reasoning` default added when nothing set it, at the end of
/// `common_params_parse_ex` (common/arg.cpp:762 @7fe450e19).
///
/// Parameters:
/// - `a`: owns the result; an arena in practice.
/// - `ops`: `Args.kwargOps()`, in command-line order.
///
/// Return: the map's entries, one per name, the last write winning. On a
/// `--chat-template-kwargs` value that is not JSON, `error.JsonError` with
/// the parser's `what()` in `last_error`.
pub fn resolveKwargs(a: Allocator, ops: []const args_mod.KwargOp) ![]KwArg {
    var map: std.ArrayList(KwArg) = .empty;
    for (ops) |op| switch (op) {
        .object => |text| {
            var v = json.parse(a, text) catch |e| {
                if (e == error.OutOfMemory) return e;
                try setError(json.what());
                return e;
            };
            var it = v.items();
            while (it.next()) |item| try put(a, &map, try a.dupe(u8, item.key), try json.dump(a, item.value, -1));
        },
        .set => |kv| try put(a, &map, kv.key, kv.value),
        .effort => |level| {
            if (std.mem.eql(u8, level, "default")) {
                for (map.items, 0..) |kv, i| if (std.mem.eql(u8, kv.key, "reasoning_effort")) {
                    _ = map.orderedRemove(i);
                    break;
                };
            } else {
                const v = try json.Value.from(a, level);
                try put(a, &map, "reasoning_effort", try json.dump(a, &v, -1));
            }
        },
    };
    if (find(map.items, "preserve_reasoning") == null) try put(a, &map, "preserve_reasoning", "true");
    return map.items;
}

fn put(a: Allocator, map: *std.ArrayList(KwArg), key: []const u8, value: []const u8) !void {
    for (map.items) |*kv| if (std.mem.eql(u8, kv.key, key)) {
        kv.value = value;
        return;
    };
    try map.append(a, .{ .key = key, .value = value });
}

fn find(map: []const KwArg, key: []const u8) ?[]const u8 {
    for (map) |kv| if (std.mem.eql(u8, kv.key, key)) return kv.value;
    return null;
}

// -----------------------------------------------------------------------------
// The display

/// Ports `assistant_turn` (tools/cli/cli-ui.h:187 @7fe450e19), in namespace
/// `ui`, with the part of `console` it drives: reasoning opens with
/// `[Start thinking]` and closes with `[End thinking]` when the answer
/// begins, and under `--color` is written grey -- the markers too -- as
/// `DISPLAY_TYPE_REASONING`'s colour in `set_display` (common/console.cpp:173 @7fe450e19).
pub const Display = struct {
    out: *std.Io.Writer,
    /// `console`'s `advanced_display`: whether set_display writes anything.
    color: bool,
    /// `console`'s `current_display`, which outlives a turn.
    current: Kind = .reset,
    /// The turn's state, as `assistant_turn` holds it.
    mode: Mode = .content,
    trailing_newline: bool = true,
    is_inside_reasoning: bool = false,

    /// `assistant_display_mode`.
    pub const Mode = enum { reasoning, content };

    /// The `display_type`s a reply can switch to.
    pub const Kind = enum { reset, reasoning, err };

    /// The `assistant_turn` constructor.
    pub fn begin(self: *Display) !void {
        self.mode = .content;
        self.trailing_newline = true;
        self.is_inside_reasoning = false;
        try self.setDisplay(.reset);
    }

    /// Ports `assistant_turn`'s `push` (tools/cli/cli-ui.h:198 @7fe450e19).
    pub fn push(self: *Display, m: Mode, buffer: []const u8) !void {
        if (m != self.mode) {
            try self.addNewlineIfNeeded();
            switch (m) {
                .content => {
                    if (self.is_inside_reasoning) {
                        try self.log("[End thinking]\n\n");
                        self.is_inside_reasoning = false;
                    }
                    try self.setDisplay(.reset);
                },
                .reasoning => {
                    try self.setDisplay(.reasoning);
                    self.is_inside_reasoning = true;
                    try self.log("\n[Start thinking]\n\n");
                },
            }
        }
        self.mode = m;
        if (buffer.len == 0) return;
        self.trailing_newline = buffer[buffer.len - 1] == '\n';
        try self.log(buffer);
        try self.out.flush();
    }

    /// The `assistant_turn` destructor.
    pub fn end(self: *Display) !void {
        try self.setDisplay(.reset);
        try self.addNewlineIfNeeded();
    }

    /// Ports `assistant_turn`'s `add_newline_if_needed` (tools/cli/cli-ui.h:226 @7fe450e19).
    fn addNewlineIfNeeded(self: *Display) !void {
        if (!self.trailing_newline) {
            try self.log("\n");
            try self.out.flush();
        }
    }

    /// Ports `ui`'s `show_error` (tools/cli/cli-ui.h:234 @7fe450e19) as a reply
    /// shows it: `console::error`, in red under `--color`, the colour before
    /// it restored.
    pub fn showError(self: *Display, title: []const u8) !void {
        const prev = self.current;
        try self.setDisplay(.err);
        try self.log("Error: ");
        try self.log(title);
        try self.log("\n");
        try self.setDisplay(prev);
        try self.out.flush();
    }

    /// `console::log("%s", s)`: up to the first NUL, as `c_str()` reads.
    fn log(self: *Display, s: []const u8) !void {
        const end_at = std.mem.findScalar(u8, s, 0) orelse s.len;
        try self.out.writeAll(s[0..end_at]);
    }

    /// Ports `console`'s `set_display` (common/console.cpp:173 @7fe450e19) for
    /// the three kinds a reply uses.
    fn setDisplay(self: *Display, kind: Kind) !void {
        if (!self.color or self.current == kind) return;
        try self.out.writeAll(switch (kind) {
            .reset => "\x1b[0m",
            .reasoning => "\x1b[90m",
            .err => "\x1b[1m\x1b[31m",
        });
        self.current = kind;
        try self.out.flush();
    }
};

// -----------------------------------------------------------------------------
// The reply

/// One reply, read as the server's stream would carry it: each piece joins
/// the text, the text is parsed as a partial message and the difference from
/// the last parse is shown; `finish` parses it whole. The reasoning and the
/// answer shown are what the history keeps, as `generate_completion`
/// (tools/cli/cli-context.cpp:350 @7fe450e19) collects them.
pub const Reply = struct {
    gpa: Allocator,
    display: *Display,
    params: *const common_chat.parse.ParserParams,
    /// Every piece sent so far: `task_result_state::generated_text`.
    text: std.ArrayList(u8) = .empty,
    /// The last parse: `task_result_state::chat_msg`.
    msg: Msg = .{},
    /// What the display was given: `generated_content`.
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    /// Set once a parse has failed and the error has been shown.
    failed: bool = false,

    pub fn deinit(self: *Reply) void {
        self.text.deinit(self.gpa);
        self.msg.deinit(self.gpa);
        self.content.deinit(self.gpa);
        self.reasoning.deinit(self.gpa);
    }

    /// The `Session.Sink` this reply is.
    pub fn sink(self: *Reply) session_mod.Sink {
        return .{ .ctx = self, .text = onText };
    }

    fn onText(ctx: *anyopaque, text: []const u8) anyerror!bool {
        const self: *Reply = @ptrCast(@alignCast(ctx));
        try self.text.appendSlice(self.gpa, text);
        return self.update(true);
    }

    /// The final, whole parse.
    pub fn finish(self: *Reply) !void {
        if (self.failed) return;
        _ = try self.update(false);
    }

    /// Ports `task_result_state`'s `update_chat_msg` (tools/server/server-task.cpp:162 @7fe450e19)
    /// and what `generate_completion` does with the diffs: a
    /// reasoning delta, then a content delta, each shown when not empty; tool
    /// call deltas reach nobody, as upstream's CLI reads neither. A parse that
    /// throws ends the stream with its error, as the server's stream does.
    ///
    /// Return: false when the parse failed.
    fn update(self: *Reply, is_partial: bool) !bool {
        var new_msg = common_chat.parse.parse(self.gpa, self.text.items, is_partial, self.params) catch |e| {
            if (e == error.OutOfMemory) return e;
            return self.reject(e);
        };
        defer new_msg.deinit(self.gpa);
        if (new_msg.empty()) return true;
        var diffs = common_chat.msg.computeDiffs(self.gpa, &self.msg, &new_msg) catch |e| {
            if (e == error.OutOfMemory) return e;
            return self.reject(e);
        };
        defer diffs.deinit(self.gpa);
        for (diffs.items) |d| {
            if (d.reasoning_content_delta.len != 0) {
                try self.reasoning.appendSlice(self.gpa, d.reasoning_content_delta);
                try self.display.push(.reasoning, d.reasoning_content_delta);
            }
            if (d.content_delta.len != 0) {
                try self.content.appendSlice(self.gpa, d.content_delta);
                try self.display.push(.content, d.content_delta);
            }
        }
        // After the deltas are shown: they borrow from both messages.
        std.mem.swap(Msg, &self.msg, &new_msg);
        return true;
    }

    fn reject(self: *Reply, e: anyerror) !bool {
        self.failed = true;
        try self.display.showError(common_chat.what(e));
        return false;
    }
};

// -----------------------------------------------------------------------------
// The conversation

/// A conversation: the templates, the server's chat options, and the history.
pub const Chat = struct {
    gpa: Allocator,
    /// Owns the kwargs and anything else that lives as long as the chat.
    arena: std.heap.ArenaAllocator,
    tmpls: templates.Templates,
    settings: *const args_mod.Args,
    kwargs: []const KwArg,
    /// `server_chat_params::enable_thinking`: on unless `--reasoning off`,
    /// and only for a template that thinks.
    enable_thinking: bool,
    /// The system prompt, re-added by `/clear`.
    system: []const u8,
    history: std.ArrayList(Msg) = .empty,

    /// Loads the templates and works out the chat options, as the server does
    /// when it loads a model.
    ///
    /// Parameters:
    /// - `gpa`: owns the chat; release with `deinit`.
    /// - `settings`: the command line; must outlive the chat.
    /// - `kwargs`: from `resolveKwargs`; must outlive the chat.
    /// - `model`: the loaded model, for its templates and BOS and EOS.
    /// - `override`: `--chat-template`'s text or `--chat-template-file`'s
    ///   contents, or "" for the model's own.
    /// - `system`: the system prompt, or "".
    ///
    /// Return: `Error.ChatFailed` with `last_error` set when the template
    /// does not load.
    pub fn init(
        gpa: Allocator,
        settings: *const args_mod.Args,
        kwargs: []const KwArg,
        model: *const c.llama_model,
        override: []const u8,
        system: []const u8,
    ) !Chat {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const info = templates.modelInfo(a, model) catch |e| return fail(e);
        var tmpls = templates.init(gpa, &info, override, "", "") catch |e| return fail(e);
        errdefer tmpls.deinit();
        const supports = templates.supportEnableThinking(gpa, &tmpls) catch |e| return fail(e);
        var self: Chat = .{
            .gpa = gpa,
            .arena = arena,
            .tmpls = tmpls,
            .settings = settings,
            .kwargs = kwargs,
            .enable_thinking = settings.enable_reasoning != 0 and supports,
            .system = try a.dupe(u8, system),
        };
        try self.addSystem();
        return self;
    }

    pub fn deinit(self: *Chat) void {
        self.clearHistory();
        self.history.deinit(self.gpa);
        self.tmpls.deinit();
        self.arena.deinit();
    }

    fn clearHistory(self: *Chat) void {
        for (self.history.items) |*m| m.deinit(self.gpa);
        self.history.clearRetainingCapacity();
    }

    /// `add_system_prompt` (tools/cli/cli-context.cpp:278 @7fe450e19).
    fn addSystem(self: *Chat) !void {
        if (self.system.len != 0) try self.push("system", self.system, "");
    }

    /// `/clear`: the history dropped and the system prompt added again.
    pub fn clear(self: *Chat) !void {
        self.clearHistory();
        try self.addSystem();
    }

    /// Adds a user turn.
    pub fn addUser(self: *Chat, text: []const u8) !void {
        try self.push("user", text, "");
    }

    /// `/regen`: drops the last reply.
    ///
    /// Return: false when the history does not end in one.
    pub fn dropReply(self: *Chat) bool {
        const last = self.history.getLastOrNull() orelse return false;
        if (!std.mem.eql(u8, last.role, "assistant")) return false;
        var m = self.history.pop().?;
        m.deinit(self.gpa);
        return true;
    }

    fn push(self: *Chat, role: []const u8, content: []const u8, reasoning: []const u8) !void {
        var m: Msg = .{};
        errdefer m.deinit(self.gpa);
        m.role = try self.gpa.dupe(u8, role);
        m.content = try self.gpa.dupe(u8, content);
        m.reasoning_content = try self.gpa.dupe(u8, reasoning);
        try self.history.append(self.gpa, m);
    }

    /// The request `oaicompat_chat_params_parse` (tools/server/server-common.cpp:1151
    /// @7fe450e19) builds for a body of `messages` alone: no tools, no
    /// schema, a generation prompt, the server's reasoning format, its
    /// thinking switch -- which an `enable_thinking` kwarg overrides -- and
    /// its kwargs.
    fn inputs(self: *const Chat) !templates.TemplatesInputs {
        var enable_thinking = self.enable_thinking;
        if (find(self.kwargs, "enable_thinking")) |v| {
            if (std.mem.eql(u8, v, "true")) {
                enable_thinking = true;
            } else if (std.mem.eql(u8, v, "false")) {
                enable_thinking = false;
            } else if (v.len != 0 and v[0] == '"') {
                try setError("invalid type for \"enable_thinking\" (expected boolean, got string)");
                return Error.ChatFailed;
            }
        }
        const caps = templates.getCaps(&self.tmpls);
        return .{
            .messages = self.history.items,
            .add_generation_prompt = true,
            .use_jinja = true,
            .parallel_tool_calls = templates.capsValue(caps, "supports_parallel_tool_calls"),
            .reasoning_format = self.settings.reasoning_format,
            .enable_thinking = enable_thinking,
            .chat_template_kwargs = self.kwargs,
            .force_pure_content = self.settings.skip_chat_parsing,
        };
    }

    /// One reply to the conversation as it stands, shown on `display` and
    /// added to the history.
    ///
    /// Parameters:
    /// - `session`: the loaded model.
    /// - `display`: where the reply goes.
    /// - `echo`: when not null, the rendered prompt is written there first --
    ///   the one-shot path's `--display-prompt`.
    ///
    /// Return: the number of tokens generated; `session.last_stop` says why
    /// the reply ended.
    pub fn reply(self: *Chat, session: *session_mod.Session, display: *Display, echo: ?*std.Io.Writer) !usize {
        var arena_state = std.heap.ArenaAllocator.init(self.gpa);
        defer arena_state.deinit();
        const a = arena_state.allocator();

        const in = try self.inputs();
        var capture: template.MarkCapture = .{ .allocator = a };
        template.mark_capture = &capture;
        const applied = templates.apply(a, &self.tmpls, &in);
        template.mark_capture = null;
        const params = applied catch |e| return fail(e);

        if (params.prompt.len == 0 and self.history.items.len != 0) return Error.EmptyRender;
        const segments = try segmentsFor(a, &capture, params.prompt);
        if (echo) |w| {
            try w.writeAll(params.prompt);
            try w.flush();
        }

        // The server task's token-level `preserved_tokens`
        // (tools/server/server-schema.cpp:341 @7fe450e19): a preserved token
        // counts only when it is one token.
        var preserved: std.ArrayList(c.llama_token) = .empty;
        for (params.preserved_tokens.items) |t| {
            const ids = try session.tokenizeText(t, .{ .add_special = false, .parse_special = true });
            defer session.allocator.free(ids);
            if (ids.len == 1) try preserved.append(a, ids[0]);
        }

        const rbudget = try self.budget(a, session, &params);
        defer if (rbudget) |rb| c.llama_sampler_free(rb);

        var parser = common_chat.parse.loadParser(self.gpa, params.parser) catch |e| return fail(e);
        defer parser.deinit();
        var pp = common_chat.parse.ParserParams.fromChatParams(&params);
        pp.parser = &parser;
        pp.reasoning_format = self.settings.reasoning_format;
        // `params_from_json_cmpl`'s post-processing: a stream, so only
        // deepseek-legacy keeps the reasoning in the content.
        pp.reasoning_in_content = self.settings.reasoning_format == .deepseek_legacy;

        var r: Reply = .{ .gpa = self.gpa, .display = display, .params = &pp };
        defer r.deinit();
        try display.begin();
        const n = session.generateTurn(segments, .{
            .preserved = preserved.items,
            .stops = params.additional_stops.items,
            .rbudget = rbudget,
        }, r.sink()) catch |e| {
            try display.end();
            return e;
        };
        try r.finish();
        try display.end();

        try self.push("assistant", r.content.items, r.reasoning.items);
        return n;
    }

    /// The reasoning budget sampler `common_sampler_init`
    /// (common/sampling.cpp:187 @7fe450e19) builds from the server task's
    /// budget fields: only for a template with an end-of-thinking tag, only
    /// when there is a budget (or a lazy grammar), and fed the generation
    /// prompt's tokens so a template that opens the block itself starts it
    /// counting.
    ///
    /// Return: the sampler, or null.
    fn budget(self: *const Chat, a: Allocator, session: *session_mod.Session, params: *const common_chat.params.ChatParams) !?*c.llama_sampler {
        if (params.thinking_end_tags.items.len == 0) return null;
        const n_budget = self.settings.reasoning_budget;
        if (!(params.grammar_lazy or n_budget >= 0)) return null;

        const tok = struct {
            fn run(al: Allocator, s: *session_mod.Session, text: []const u8) ![]c.llama_token {
                const ids = try s.tokenizeText(text, .{ .add_special = false, .parse_special = true });
                defer s.allocator.free(ids);
                return al.dupe(c.llama_token, ids);
            }
        }.run;

        const start = try tok(a, session, params.thinking_start_tag);
        var ends: std.ArrayList([]const c.llama_token) = .empty;
        for (params.thinking_end_tags.items) |tag| {
            if (tag.len != 0) try ends.append(a, try tok(a, session, tag));
        }
        if (start.len == 0 or ends.items.len == 0) return null;

        var forced: std.ArrayList(c.llama_token) = .empty;
        if (self.settings.reasoning_budget_message.len != 0) {
            try forced.appendSlice(a, try tok(a, session, self.settings.reasoning_budget_message));
        }
        try forced.appendSlice(a, ends.items[0]);

        const rb = reasoning_budget.init(
            session.vocab,
            &.{start},
            ends.items,
            forced.items,
            if (n_budget < 0) std.math.maxInt(i32) else n_budget,
            .idle,
        ) catch return error.OutOfMemory;

        // The generation prompt's tokens, less a leading one some tokenizers
        // add as a space before a special token.
        const gp = params.generation_prompt;
        if (gp.len != 0) {
            const ids = try tok(a, session, gp);
            for (ids, 0..) |id, i| {
                if (i == 0) {
                    var buf: [256]u8 = undefined;
                    const n = c.llama_token_to_piece(session.vocab, id, &buf, buf.len, 0, true);
                    const first: u8 = if (n > 0) buf[0] else 0;
                    if (isSpace(first) and !isSpace(gp[0])) continue;
                }
                c.llama_sampler_accept(rb, id);
            }
        }
        return rb;
    }
};

/// `std::isspace` in the C locale.
fn isSpace(ch: u8) bool {
    return ch == ' ' or (ch >= '\t' and ch <= '\r');
}

/// The rendered prompt split into runs of one trust level, from the marks
/// `template.zig` recorded while `templates.apply` ran: the longest render
/// the prompt begins with -- the prompt's own, which the analysis probes
/// before it cannot outgrow -- and, after it, whatever a parser appended
/// itself (Gemma 4's `<|turn>model\n`), which is template text.
///
/// Return: segments borrowing from `capture` and `prompt`, allocated in `a`.
/// `Error.MarkingLost` when no render matches, which fails closed.
fn segmentsFor(a: Allocator, capture: *const template.MarkCapture, prompt: []const u8) ![]chat.Segment {
    var best: ?template.Marked = null;
    var i = capture.renders.items.len;
    while (i > 0) {
        i -= 1;
        const r = capture.renders.items[i];
        if (!std.mem.startsWith(u8, prompt, r.text)) continue;
        if (best == null or r.text.len > best.?.text.len) best = r;
    }
    const m = best orelse return Error.MarkingLost;
    var out: std.ArrayList(chat.Segment) = .empty;
    var at: usize = 0;
    for (m.parts) |part| {
        const run = m.text[at..][0..part.len];
        at += part.len;
        if (run.len == 0) continue;
        // The only input runs that are control tokens are the two bound as tokens.
        try out.append(a, .{ .text = run, .trusted = !part.input or part.token, .token = part.token });
    }
    if (at != m.text.len) return Error.MarkingLost;
    if (prompt.len > m.text.len) try out.append(a, .{ .text = prompt[m.text.len..], .trusted = true });
    return out.items;
}

/// Ports `tty_can_use_colors` (common/common.cpp:1185 @7fe450e19): `--color
/// auto`'s answer.
///
/// Parameters:
/// - `io`: for asking the two streams.
/// - `environ`: the process environment, `std.process.Init`'s map.
pub fn ttyCanUseColors(io: std.Io, environ: anytype) bool {
    if (environ.get("NO_COLOR")) |v| if (v.len != 0) return false;
    if (environ.get("TERM")) |v| if (std.mem.eql(u8, v, "dumb")) return false;
    const out = std.Io.File.stdout().isTty(io) catch false;
    const err = std.Io.File.stderr().isTty(io) catch false;
    return out or err;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the display brackets reasoning as upstream's assistant_turn does" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    var d: Display = .{ .out = &w, .color = false };
    try d.begin();
    try d.push(.reasoning, "hmm");
    try d.push(.reasoning, " ok");
    try d.push(.content, "Paris.");
    try d.end();
    try std.testing.expectEqualStrings("\n[Start thinking]\n\nhmm ok\n[End thinking]\n\nParis.\n", w.buffered());
}

test "a reply with no answer never closes its thinking, and colour is grey" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    var d: Display = .{ .out = &w, .color = true };
    try d.begin();
    try d.push(.reasoning, "x\n");
    try d.end();
    try std.testing.expectEqualStrings("\x1b[90m\n[Start thinking]\n\nx\n\x1b[0m", w.buffered());
}

test "kwargs replay in order, the last write wins, and preserve_reasoning defaults on" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ops = [_]args_mod.KwargOp{
        .{ .object = "{\"enable_thinking\": false, \"x\": [1, \"a\"]}" },
        .{ .set = .{ .key = "enable_thinking", .value = "true" } },
        .{ .effort = "high" },
        .{ .effort = "default" },
    };
    const kw = try resolveKwargs(a, &ops);
    try std.testing.expectEqual(@as(usize, 3), kw.len);
    try std.testing.expectEqualStrings("true", find(kw, "enable_thinking").?);
    try std.testing.expectEqualStrings("[1,\"a\"]", find(kw, "x").?);
    try std.testing.expectEqualStrings("true", find(kw, "preserve_reasoning").?);
    try std.testing.expect(find(kw, "reasoning_effort") == null);
}

test "the stop-string search holds back a partial stop and cuts at a full one" {
    const stops = [_][]const u8{"</s>"};
    try std.testing.expectEqual(@as(?usize, 5), session_mod.findStop(&stops, "hello</", 2, false));
    try std.testing.expectEqual(@as(?usize, 5), session_mod.findStop(&stops, "hello</s>", 2, true));
    try std.testing.expectEqual(@as(?usize, null), session_mod.findStop(&stops, "hello", 2, false));
    try std.testing.expectEqual(@as(usize, 1), session_mod.validUtf8Len("a\xc3"));
}
