//! A realtime conversation, modeled after the OpenAI realtime API (GA).
//!
//! One slot holds the conversation for the whole session: text and audio go
//! into it as they arrive, so a response starts right after the last audio.
//! session.type "realtime" holds a conversation, "transcription" transcribes
//! each committed audio buffer on its own.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-realtime.cpp` and
//! `llama.cpp/tools/server/server-realtime.h` (llama.cpp v1.0.0,
//! `b1b770d90`) whole: `server_realtime_deps`,
//! `server_realtime_conversation`, `realtime_conversation`,
//! `server_realtime_op` and `server_realtime_slot`. Each declaration names
//! what it ports and the line it began at.
//!
//! # Threads
//!
//! As upstream: `conversation` runs on the connection's HTTP thread, reading
//! results; client events are handled on a thread of its own; the slot half
//! runs on the server thread. The two meet through control tasks, and share
//! what `mutex` guards.
//!
//! # Exceptions
//!
//! As upstream: an exception from a client event or from a result is sent
//! to the client as an `error` event with type `server_error`; one from a
//! result ends the session. What escapes the reader thread past that (only
//! a failed send) is logged and closes this connection (`readerLoop`).
//!
//! # Memory
//!
//! Session configuration lives in `arena`, which only the reader thread
//! touches once the session runs; the current response's items in
//! `resp_arena`, under `mutex`; the ids and the transcript both threads read
//! are `gpa`'s; an event lives as long as it takes to send it.

const std = @import("std");
const mtmd = @import("mtmd");
const llama = @import("llama");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const json = @import("json");
const common_chat = @import("common_chat");
const tools_common = @import("tools_common");

/// `nullptr`, in an object's list.
const jnull: json.Value = .null;

/// `json { {k0, v0}, {k1, v1}, ... }`: an object of the pairs in order.
fn object(a: std.mem.Allocator, pairs: anytype) common_chat.exception.Error!json.Value {
    var o = json.Value.emptyObject();
    inline for (pairs) |p| try o.set(a, p[0], try mod.common.from(a, p[1]));
    return o;
}

/// `json::array({ v... })`.
fn array(a: std.mem.Allocator, values: anytype) common_chat.exception.Error!json.Value {
    var o = json.Value.emptyArray();
    inline for (values) |v| try o.pushBack(a, try mod.common.from(a, v));
    return o;
}

/// A string, or null when it is empty: `s.empty() ? json(nullptr) : json(s)`.
fn orNull(a: std.mem.Allocator, s: []const u8) common_chat.exception.Error!json.Value {
    return if (s.len == 0) .null else json.Value.initString(a, s);
}

fn valueIs(v: *const json.Value, s: []const u8) bool {
    return v.isString() and std.mem.eql(u8, v.string, s);
}

/// `"item_" + random_string()`, and the others.
fn randomId(a: std.mem.Allocator, prefix: []const u8) common_chat.exception.Error![]u8 {
    return std.mem.concat(a, u8, &.{ prefix, try mod.common.randomString(a) });
}

// -----------------------------------------------------------------------------
// request values

/// Ports `parse_output_modalities` (server-realtime.cpp:35 @b1b770d90):
/// output_modalities of a response, unset: the model switches between text
/// and audio itself.
///
/// Return: null, or the error's text in `a`.
fn parseOutputModalities(a: std.mem.Allocator, j: *const json.Value, mode: *mtmd.helper_gen.OutMode) common_chat.exception.Error!?[]const u8 {
    if (j.isNull()) {
        mode.* = mtmd.helper_gen.OutMode.auto;
        return null;
    }
    if (!j.isArray() or j.empty()) {
        return "output_modalities must be a non-empty array";
    }
    var text = false;
    var audio = false;
    for (j.array.items) |*m| {
        if (valueIs(m, "text")) {
            text = true;
        } else if (valueIs(m, "audio")) {
            audio = true;
        } else {
            return try std.mem.concat(a, u8, &.{ "output_modalities: unknown modality ", try json.dump(a, m, -1) });
        }
    }
    if (text and audio) {
        mode.* = mtmd.helper_gen.OutMode.mixed;
    } else {
        mode.* = if (audio) mtmd.helper_gen.OutMode.audio else mtmd.helper_gen.OutMode.text;
    }
    return null;
}

/// Ports `parse_max_output_tokens` (server-realtime.cpp:64 @b1b770d90).
fn parseMaxOutputTokens(j: *const json.Value, n: *i32) bool {
    if (valueIs(j, "inf")) {
        n.* = -1;
        return true;
    }
    if (!j.isNumberInteger()) {
        return false;
    }
    // j.get<int64_t>(): an unsigned value wraps
    const v: i64 = switch (j.*) {
        .number_integer => |x| x,
        .number_unsigned => |x| @bitCast(x),
        else => unreachable,
    };
    if (v < 1) {
        return false;
    }
    n.* = @intCast(@min(v, std.math.maxInt(i32)));
    return true;
}

/// Ports `tool_choice_str` (server-realtime.cpp:106 @b1b770d90).
fn toolChoiceStr(choice: common_chat.ToolChoice) []const u8 {
    return switch (choice) {
        .none => "none",
        .required => "required",
        else => "auto",
    };
}

/// Ports `output_modalities_json` (server-realtime.cpp:114 @b1b770d90).
fn outputModalitiesJson(a: std.mem.Allocator, mode: mtmd.helper_gen.OutMode) common_chat.exception.Error!json.Value {
    return switch (mode) {
        mtmd.helper_gen.OutMode.text => array(a, .{"text"}),
        mtmd.helper_gen.OutMode.audio => array(a, .{"audio"}),
        mtmd.helper_gen.OutMode.mixed => array(a, .{ "text", "audio" }),
        else => .null,
    };
}

/// Ports `parse_tool_choice` (server-realtime.cpp:92 @b1b770d90).
///
/// Return: null, or the error's text in `a`.
fn parseToolChoice(a: std.mem.Allocator, j: *const json.Value, choice: *common_chat.ToolChoice) common_chat.exception.Error!?[]const u8 {
    if (!j.isString()) {
        return "tool_choice must be \"auto\", \"none\" or \"required\"";
    }
    choice.* = common_chat.msg.toolChoiceParseOaicompat(j.string) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return try a.dupe(u8, mod.context.whatOf(e)),
    };
    return null;
}

// -----------------------------------------------------------------------------
// server_realtime_deps

/// Ports `server_realtime_deps` (server-realtime.h:23 @b1b770d90).
pub const Deps = struct {
    queue_tasks: *mod.context.TaskQueue,
    queue_results: *mod.context.ResultQueue,
    meta: *const mod.context.Meta,
    params: *const tools_common.params.Params,
    mctx: ?*mtmd.mtmd.Context,
    cap: *const mtmd.helper.Cap,
    vocab: ?*const llama.impl.llama_c.llama_vocab,
};

/// `req.should_stop`'s kin: the reader's stop test.
const Stop = struct {
    conv: *Conversation,
    /// the slot's controls also stop on `closed`
    with_closed: bool,

    pub fn shouldStop(self: Stop) bool {
        return (self.with_closed and self.conv.closed.load(.seq_cst)) or !self.conv.conn.isOpen();
    }
};

/// `tool_call_index -> output index`, a `std::map`: walked in key order.
const CallItem = struct { index: usize, idx: usize };

/// Ports `realtime_conversation` (server-realtime.cpp:123 @b1b770d90).
const Conversation = struct {
    deps: Deps,
    conn: *mod.http.WsConn,

    arena: std.heap.ArenaAllocator,
    /// what `response_create` resets
    resp_arena: std.heap.ArenaAllocator,

    id: []const u8 = "",

    // session config
    type: []const u8 = "realtime",
    instructions: []const u8 = "",
    language: []const u8 = "", // transcription
    out_language: []const u8 = "", // `deps.params.tts_lang` at first
    out_voice: []const u8 = "", // `deps.params.tts_voice` at first
    out_mode: mtmd.helper_gen.OutMode = mtmd.helper_gen.OutMode.auto,
    max_output_tokens: i32 = -1, // "inf"
    input_rate: c_int = 16000,
    output_rate: c_int = 24000,

    // tools, in the system turn
    tools_json: json.Value = json.Value.emptyArray(), // as the client sent them
    tools: std.ArrayList(common_chat.Tool) = .empty,
    tool_choice: common_chat.ToolChoice = .auto,

    // the text around a turn, from the chat template
    first_header: []const u8 = "", // system prompt + first user header
    user_header: []const u8 = "",
    user_footer: []const u8 = "",
    assistant_header: []const u8 = "",
    assistant_footer: []const u8 = "",
    tool_header: []const u8 = "", // around a tool result
    tool_footer: []const u8 = "",

    // conversation state, owned by the reader thread
    started: bool = false, // something went into the slot, the session config is fixed
    turn_open: bool = false, // a user turn is being filled
    has_audio: bool = false, // the open turn has audio not committed yet
    tool_results: bool = false, // tool results went in after the last response, the next one answers them
    n_turns: c_int = 0,
    next_header: []const u8 = "", // text that goes in before the next user turn
    resampler: ?*mtmd.vendor.miniaudio.HelperResampler = null, // input_rate -> model rate

    // the item or response the results belong to, shared with the result loop
    mutex: stdlib.sync.Mutex = .{},
    last_item_id: []const u8 = "",
    transcribe_item_id: []const u8 = "", // transcription: the item being transcribed
    transcript: std.ArrayList(u8) = .empty,
    response_id: []const u8 = "",
    response_item_id: []const u8 = "",
    response_text: std.ArrayList(u8) = .empty,
    idx_text: i32 = -1,
    idx_audio: i32 = -1,
    parts: json.Value = json.Value.emptyArray(),
    output: json.Value = json.Value.emptyArray(), // items of the response
    idx_message: i32 = -1, // output index of the message, added with its first delta
    call_items: std.ArrayList(CallItem) = .empty, // tool call -> output index
    parse: ?mod.task.TaskResultState = null, // the tool call parser, with tools

    closed: std.atomic.Value(bool) = .init(false),

    fn lock(self: *Conversation) void {
        self.mutex.lock();
    }

    fn unlock(self: *Conversation) void {
        self.mutex.unlock();
    }

    fn a(self: *Conversation) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// `last_item_id = id`, with `mutex` held.
    fn setLastItem(self: *Conversation, id: []const u8) !void {
        const copy = try std.heap.c_allocator.dupe(u8, id);
        std.heap.c_allocator.free(self.last_item_id);
        self.last_item_id = copy;
    }

    /// `transcribe_item_id = id`, with `mutex` held.
    fn setTranscribeItem(self: *Conversation, id: []const u8) !void {
        const copy = try std.heap.c_allocator.dupe(u8, id);
        std.heap.c_allocator.free(self.transcribe_item_id);
        self.transcribe_item_id = copy;
    }

    fn ra(self: *Conversation) std.mem.Allocator {
        return self.resp_arena.allocator();
    }

    //
    // events
    //

    /// Ports `send_event` (server-realtime.cpp:188 @b1b770d90). `ev` is
    /// allocated in `ea`, an arena that outlives the call.
    fn sendEvent(self: *Conversation, ea: std.mem.Allocator, ev: json.Value) !void {
        var e = ev;
        try e.set(ea, "event_id", try json.Value.initString(ea, try randomId(ea, "event_")));
        _ = self.conn.send(try mod.common.safeJsonToStr(ea, &e));
    }

    /// Ports `send_error` (server-realtime.cpp:193 @b1b770d90).
    fn sendError(self: *Conversation, ea: std.mem.Allocator, message: []const u8, type_err: []const u8) !void {
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "error" },
            .{ "error", try object(ea, .{
                .{ "type", type_err },
                .{ "code", jnull },
                .{ "message", message },
                .{ "param", jnull },
            }) },
        }));
    }

    fn sendInvalid(self: *Conversation, ea: std.mem.Allocator, message: []const u8) !void {
        return self.sendError(ea, message, "invalid_request_error");
    }

    /// Ports `session_json` (server-realtime.cpp:200 @b1b770d90).
    fn sessionJson(self: *Conversation, ea: std.mem.Allocator) !json.Value {
        var audio_in = try object(ea, .{
            .{ "format", try object(ea, .{ .{ "type", "audio/pcm" }, .{ "rate", self.input_rate } }) },
            .{ "turn_detection", jnull },
        });
        if (std.mem.eql(u8, self.type, "transcription")) {
            try audio_in.set(ea, "transcription", try object(ea, .{
                .{ "model", self.deps.meta.model_name },
                .{ "language", try orNull(ea, self.language) },
            }));
            return object(ea, .{
                .{ "id", self.id },
                .{ "object", "realtime.transcription_session" },
                .{ "type", "transcription" },
                .{ "audio", try object(ea, .{.{ "input", audio_in }}) },
            });
        }
        var audio = try object(ea, .{.{ "input", audio_in }});
        if (self.deps.cap.audio_gen) {
            try audio.set(ea, "output", try object(ea, .{
                .{ "format", try object(ea, .{ .{ "type", "audio/pcm" }, .{ "rate", self.output_rate } }) },
                .{ "voice", try orNull(ea, self.out_voice) },
                .{ "language", try orNull(ea, self.out_language) },
            }));
        }
        const max_tokens: json.Value = if (self.max_output_tokens < 0) try json.Value.initString(ea, "inf") else .{ .number_integer = self.max_output_tokens };
        return object(ea, .{
            .{ "id", self.id },
            .{ "object", "realtime.session" },
            .{ "type", "realtime" },
            .{ "model", self.deps.meta.model_name },
            .{ "instructions", self.instructions },
            .{ "output_modalities", try outputModalitiesJson(ea, self.out_mode) },
            .{ "max_output_tokens", max_tokens },
            .{ "tools", try self.tools_json.clone(ea) },
            .{ "tool_choice", toolChoiceStr(self.tool_choice) },
            .{ "audio", audio },
        });
    }

    //
    // slot controls, run on the slot in the order they are posted
    //

    /// Ports `control` (server-realtime.cpp:237 @b1b770d90).
    fn control(self: *Conversation, action: mod.task.ControlAction, text: []const u8, audio: []const f32, response: ?*const mod.task.ControlResponse) !bool {
        var ctl = try mod.context.Reader.init(std.heap.c_allocator, self.deps.queue_tasks, self.deps.queue_results, 1);
        defer ctl.deinit();
        var task = mod.task.Task.initType(.control);
        task.id = ctl.getNewId();
        const ta = try task.arenaAlloc(std.heap.c_allocator);
        task.params.control_action = action;
        task.params.control_cmpl_id = try ta.dupe(u8, self.id);
        task.params.control_text = try ta.dupe(u8, text);
        task.params.control_audio = try ta.dupe(f32, audio);
        if (response) |r| {
            task.params.control_response = try mod.task.deepClone(ta, r.*);
        }
        try ctl.postTask(task, false);
        const result = (try ctl.next(Stop{ .conv = self, .with_closed = true })) orelse return false;
        defer result.destroy(std.heap.c_allocator);
        return switch (result.kind) {
            .control => |res| res.success,
            else => false,
        };
    }

    /// Ports `response_sampling` (server-realtime.cpp:254 @b1b770d90):
    /// sampling of a response: the server defaults, with the tool call
    /// grammar of `session`.
    fn responseSampling(self: *Conversation, ea: std.mem.Allocator, session: ?*const common_chat.session.Session) !tools_common.params.Sampling {
        var s = try mod.task.deepClone(ea, self.deps.params.sampling);
        if (session) |ss| {
            try ss.applySampling(ea, &s);
        }
        return s;
    }

    //
    // chat template
    //

    /// Ports `chat_inputs` (server-realtime.cpp:266 @b1b770d90).
    fn chatInputs(self: *Conversation, ea: std.mem.Allocator, messages: []const common_chat.Msg, add_generation_prompt: bool) !common_chat.templates.TemplatesInputs {
        const params = self.deps.meta.chat_params;
        return .{
            .messages = messages,
            .add_generation_prompt = add_generation_prompt,
            .use_jinja = params.use_jinja,
            .enable_thinking = false,
            .reasoning_format = params.reasoning_format,
            .chat_template_kwargs = try params.chat_template_kwargs.kwArgs(ea),
            .tools = self.tools.items,
        };
    }

    /// Ports `render` (server-realtime.cpp:279 @b1b770d90).
    fn render(self: *Conversation, ea: std.mem.Allocator, messages: []const common_chat.Msg, add_generation_prompt: bool) ![]const u8 {
        const inputs = try self.chatInputs(ea, messages, add_generation_prompt);
        const params = try common_chat.templates.apply(ea, self.deps.meta.chat_params.tmpls, &inputs);
        return params.prompt.str(ea);
    }

    /// The `with` lambda of `derive_turn_text`.
    fn renderWith(self: *Conversation, ea: std.mem.Allocator, sys: []const common_chat.Msg, rest: []const common_chat.Msg, add_generation_prompt: bool) ![]const u8 {
        const all = try std.mem.concat(ea, common_chat.Msg, &.{ sys, rest });
        return self.render(ea, all, add_generation_prompt);
    }

    /// Ports `derive_turn_text` (server-realtime.cpp:284 @b1b770d90): the
    /// text around a user and an assistant turn, from placeholder messages.
    fn deriveTurnText(self: *Conversation, instructions: []const u8) !bool {
        var tmp = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer tmp.deinit();
        const ta = tmp.allocator();
        const sa = self.a();

        const user = "@@USER@@";
        const reply = "@@REPLY@@";
        const u: common_chat.Msg = .{ .role = "user", .content = user };
        const r: common_chat.Msg = .{ .role = "assistant", .content = reply };

        var sys_buf: [1]common_chat.Msg = undefined;
        var sys: []const common_chat.Msg = &.{};
        if (instructions.len != 0) {
            sys_buf[0] = .{ .role = "system", .content = instructions };
            sys = sys_buf[0..1];
        }
        const ax = try self.renderWith(ta, sys, &.{u}, false);
        const g = try self.renderWith(ta, sys, &.{u}, true);
        const cx = try self.renderWith(ta, sys, &.{ u, r }, false);
        const d = try self.renderWith(ta, sys, &.{ u, r, u }, false);

        const pa = std.mem.indexOf(u8, ax, user);
        const pc = std.mem.indexOf(u8, cx, reply);
        const pd = if (cx.len <= d.len) std.mem.indexOfPos(u8, d, cx.len, user) else null;
        if (pa == null or pc == null or pd == null or
            !std.mem.startsWith(u8, cx, ax) or !std.mem.startsWith(u8, d, cx) or !std.mem.startsWith(u8, g, ax) or pc.? < ax.len)
        {
            return false;
        }
        // the response opens with the generation prompt, a past turn may render differently (e.g. its think block)
        self.first_header = try sa.dupe(u8, ax[0..pa.?]);
        self.user_footer = try sa.dupe(u8, ax[pa.? + user.len ..]);
        self.assistant_header = try sa.dupe(u8, g[ax.len..]);
        self.assistant_footer = try sa.dupe(u8, cx[pc.? + reply.len ..]);
        self.user_header = try sa.dupe(u8, d[cx.len..pd.?]);

        // a tool result follows the assistant turn that called the tool
        if (self.tools.items.len != 0) {
            const result = "@@RESULT@@";
            var call: common_chat.Msg = .{ .role = "assistant", .content = "" };
            try call.tool_calls.append(ta, .{ .name = self.tools.items[0].name, .arguments = "{}", .id = "call_0" });
            const res: common_chat.Msg = .{ .role = "tool", .content = result, .tool_call_id = "call_0" };
            const e = try self.renderWith(ta, sys, &.{ u, call }, false);
            const f = try self.renderWith(ta, sys, &.{ u, call, res }, false);
            const pf = if (e.len <= f.len) std.mem.indexOfPos(u8, f, e.len, result) else null;
            if (pf == null or !std.mem.startsWith(u8, f, e)) {
                return false;
            }
            self.tool_header = try sa.dupe(u8, f[e.len..pf.?]);
            self.tool_footer = try sa.dupe(u8, f[pf.? + result.len ..]);
        }
        return true;
    }

    //
    // conversation
    //

    /// Ports `ensure_turn` (server-realtime.cpp:345 @b1b770d90): the header
    /// of a user turn goes in before its first audio or text.
    fn ensureTurn(self: *Conversation, ea: std.mem.Allocator) !bool {
        if (self.turn_open) {
            return true;
        }
        const header = if (self.n_turns == 0) self.first_header else try std.mem.concat(ea, u8, &.{ self.next_header, self.user_header });
        if (!try self.control(.text_append, header, &.{}, null)) {
            return false;
        }
        self.next_header = "";
        self.started = true;
        self.turn_open = true;
        return true;
    }

    /// Ports `on_append` (server-realtime.cpp:359 @b1b770d90).
    fn onAppend(self: *Conversation, ea: std.mem.Allocator, ev: *const json.Value) !void {
        const pcm = mod.tokens.base64HppDecode(ea, mod.common.jsonValue(ev, "audio", []const u8, "")) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return self.sendInvalid(ea, "invalid base64 audio"),
        };
        var samples = try pcm16ToF32(ea, pcm);
        if (samples.len == 0) {
            return;
        }
        if (!try self.resample(ea, &samples) or samples.len == 0 or !try self.ensureTurn(ea)) {
            return;
        }
        if (try self.control(.audio_append, "", samples, null)) {
            self.has_audio = true;
        }
    }

    /// Ports `resample` (server-realtime.cpp:384 @b1b770d90): `samples` at
    /// `input_rate` to the model's rate, the resampler made on first use;
    /// false after sending the error.
    fn resample(self: *Conversation, ea: std.mem.Allocator, samples: *[]f32) !bool {
        if (self.resampler == null) {
            self.resampler = mtmd.helper.mtmd_helper_resampler_init(self.input_rate, mtmd.mtmd.mtmd_get_audio_sample_rate(self.deps.mctx.?));
            if (self.resampler == null) {
                try self.sendError(ea, "failed to initialize the audio resampler", "server_error");
                return false;
            }
        }
        var out: ?[*]const f32 = null;
        const n = mtmd.helper.mtmd_helper_resampler_process(self.resampler.?, samples.ptr, samples.len, &out);
        if (n < 0) {
            try self.sendError(ea, "failed to resample the audio", "server_error");
            return false;
        }
        if (out != samples.ptr) {
            samples.* = try ea.dupe(f32, out.?[0..@intCast(n)]);
        }
        return true;
    }

    /// Ports `commit` (server-realtime.cpp:405 @b1b770d90): ends the audio
    /// of the open turn, it becomes a user item.
    fn commit(self: *Conversation, ea: std.mem.Allocator) !bool {
        if (!self.has_audio) {
            try self.sendError(ea, "input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty");
            return false;
        }
        if (!try self.control(.audio_end, "", &.{}, null)) {
            return false;
        }
        self.has_audio = false;

        self.lock();
        defer self.unlock();
        const prev = try orNull(ea, self.last_item_id);
        try self.setLastItem(try randomId(ea, "item_"));
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "input_audio_buffer.committed" },
            .{ "previous_item_id", try prev.clone(ea) },
            .{ "item_id", self.last_item_id },
        }));
        const item = try object(ea, .{
            .{ "id", self.last_item_id },
            .{ "object", "realtime.item" },
            .{ "type", "message" },
            .{ "role", "user" },
            .{ "status", "completed" },
            .{ "content", try array(ea, .{try object(ea, .{.{ "type", "input_audio" }})}) },
        });
        try self.sendItemEvents(ea, &prev, &item);
        return true;
    }

    /// `conversation.item.added`, then `conversation.item.done`.
    fn sendItemEvents(self: *Conversation, ea: std.mem.Allocator, prev: *const json.Value, item: *const json.Value) !void {
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "conversation.item.added" },
            .{ "previous_item_id", try prev.clone(ea) },
            .{ "item", try item.clone(ea) },
        }));
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "conversation.item.done" },
            .{ "previous_item_id", try prev.clone(ea) },
            .{ "item", try item.clone(ea) },
        }));
    }

    /// Ports `on_commit` (server-realtime.cpp:426 @b1b770d90).
    fn onCommit(self: *Conversation, ea: std.mem.Allocator) !void {
        if (!try self.commit(ea) or !std.mem.eql(u8, self.type, "transcription")) {
            return;
        }
        // transcription: the committed audio is transcribed on its own, then the slot is cleared for the next one
        {
            self.lock();
            defer self.unlock();
            try self.setTranscribeItem(self.last_item_id);
            self.transcript.clearRetainingCapacity();
        }
        var response: mod.task.ControlResponse = .{};
        response.n_predict = self.max_output_tokens;
        response.out_mode = mtmd.helper_gen.OutMode.text;
        response.sampling = try self.responseSampling(ea, null);
        const prompt = try std.mem.concat(ea, u8, &.{ self.user_footer, self.assistant_header });
        if (!try self.control(.response_create, prompt, &.{}, &response)) {
            return;
        }
        _ = try self.control(.clear, "", &.{}, null);
        self.turn_open = false;
        self.n_turns = 0;
    }

    /// `item.contains("id") && item["id"].is_string() ? item["id"] : "item_" + random_string()`.
    fn itemId(ea: std.mem.Allocator, item: *const json.Value) ![]const u8 {
        if (item.contains("id")) {
            const v = try item.at("id");
            if (v.isString()) return v.string;
        }
        return randomId(ea, "item_");
    }

    /// Ports `on_item_create` (server-realtime.cpp:448 @b1b770d90).
    fn onItemCreate(self: *Conversation, ea: std.mem.Allocator, ev: *const json.Value) !void {
        if (std.mem.eql(u8, self.type, "transcription")) {
            return self.sendInvalid(ea, "conversation.item.create: not supported in a transcription session");
        }
        const empty = json.Value.emptyObject();
        const item: *const json.Value = if (ev.contains("item")) try ev.at("item") else &empty;
        const item_type = mod.common.jsonValue(item, "type", []const u8, "");
        if (std.mem.eql(u8, item_type, "function_call_output")) {
            return self.onToolOutput(ea, item);
        }
        if (!std.mem.eql(u8, item_type, "message") or !std.mem.eql(u8, mod.common.jsonValue(item, "role", []const u8, ""), "user")) {
            return self.sendInvalid(ea, "conversation.item.create: only user messages and function call outputs are supported");
        }
        var text: std.ArrayList(u8) = .empty;
        if (item.contains("content")) {
            const content = try item.at("content");
            // a range-for: an array's elements, an object's values, a scalar once, null never
            switch (content.*) {
                .array => |arr| for (arr.items) |*part| {
                    if (!try self.appendPart(ea, &text, part)) return;
                },
                .object => |m| for (m.members.items) |*member| {
                    if (!try self.appendPart(ea, &text, &member.value)) return;
                },
                .null => {},
                else => if (!try self.appendPart(ea, &text, content)) return,
            }
        }
        if (!try self.ensureTurn(ea) or !try self.control(.text_append, text.items, &.{}, null)) {
            return;
        }
        self.lock();
        defer self.unlock();
        const prev = try orNull(ea, self.last_item_id);
        try self.setLastItem(try itemId(ea, item));
        const out = try object(ea, .{
            .{ "id", self.last_item_id },
            .{ "object", "realtime.item" },
            .{ "type", "message" },
            .{ "role", "user" },
            .{ "status", "completed" },
            .{ "content", try array(ea, .{try object(ea, .{ .{ "type", "input_text" }, .{ "text", text.items } })}) },
        });
        try self.sendItemEvents(ea, &prev, &out);
    }

    fn appendPart(self: *Conversation, ea: std.mem.Allocator, text: *std.ArrayList(u8), part: *const json.Value) !bool {
        const t = mod.common.jsonValue(part, "type", []const u8, "");
        if (!std.mem.eql(u8, t, "input_text")) {
            try self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "conversation.item.create: unsupported content type: ", t }));
            return false;
        }
        try text.appendSlice(ea, mod.common.jsonValue(part, "text", []const u8, ""));
        return true;
    }

    /// Ports `on_tool_output` (server-realtime.cpp:484 @b1b770d90): the
    /// result of a tool call of the last response, the next response
    /// answers it.
    fn onToolOutput(self: *Conversation, ea: std.mem.Allocator, item: *const json.Value) !void {
        if (self.tools.items.len == 0 or self.n_turns == 0 or self.turn_open) {
            return self.sendInvalid(ea, "conversation.item.create: a function call output must follow the response that called it");
        }
        const output = mod.common.jsonValue(item, "output", []const u8, "");
        const text = try std.mem.concat(ea, u8, &.{ self.next_header, self.tool_header, output, self.tool_footer });
        if (!try self.control(.text_append, text, &.{}, null)) {
            return;
        }
        self.next_header = "";
        self.tool_results = true;

        self.lock();
        defer self.unlock();
        const prev = try orNull(ea, self.last_item_id);
        try self.setLastItem(try itemId(ea, item));
        const out = try object(ea, .{
            .{ "id", self.last_item_id },
            .{ "object", "realtime.item" },
            .{ "type", "function_call_output" },
            .{ "call_id", mod.common.jsonValue(item, "call_id", []const u8, "") },
            .{ "output", output },
            .{ "status", "completed" },
        });
        try self.sendItemEvents(ea, &prev, &out);
    }

    /// Ports `on_response_create` (server-realtime.cpp:505 @b1b770d90).
    fn onResponseCreate(self: *Conversation, ea: std.mem.Allocator, ev: *const json.Value) !void {
        if (std.mem.eql(u8, self.type, "transcription")) {
            return self.sendInvalid(ea, "response.create: not supported in a transcription session");
        }
        if (!self.turn_open and !self.tool_results) {
            return self.sendError(ea, "response.create: commit audio or create a message first", "conversation_empty");
        }
        var mode = self.out_mode;
        const empty = json.Value.emptyObject();
        const resp: *const json.Value = if (ev.contains("response")) try ev.at("response") else &empty;
        if (resp.contains("output_modalities")) {
            if (try parseOutputModalities(ea, try resp.at("output_modalities"), &mode)) |err| {
                return self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "response.create: ", err }));
            }
        }
        if (!self.deps.cap.audio_gen and mode != mtmd.helper_gen.OutMode.text and mode != mtmd.helper_gen.OutMode.auto) {
            return self.sendInvalid(ea, "response.create: the model cannot generate audio");
        }
        if (!self.deps.cap.text_gen and mode != mtmd.helper_gen.OutMode.audio and mode != mtmd.helper_gen.OutMode.auto) {
            return self.sendInvalid(ea, "response.create: the model cannot generate text");
        }
        // audio not committed yet belongs to this turn
        if (self.has_audio and !try self.commit(ea)) {
            return;
        }

        var choice = self.tool_choice;
        if (resp.contains("tool_choice")) {
            if (try parseToolChoice(ea, try resp.at("tool_choice"), &choice)) |err| {
                return self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "response.create: ", err }));
            }
        }

        // the gen-audio pipeline opens the response in the asked mode, after tool results the assistant turn opens right away
        const prompt = try std.mem.concat(ea, u8, &.{ if (self.turn_open) self.user_footer else "", self.assistant_header });
        var response: mod.task.ControlResponse = .{};
        response.n_predict = self.max_output_tokens;
        if (resp.contains("max_output_tokens") and !parseMaxOutputTokens(try resp.at("max_output_tokens"), &response.n_predict)) {
            return self.sendInvalid(ea, "response.create: max_output_tokens must be an integer or \"inf\"");
        }
        response.out_mode = mode;
        response.lang = self.out_language;
        response.voice = self.out_voice;

        // with tools, the output goes through the tool call parser of the chat template and its grammar
        var session: ?common_chat.session.Session = null;
        defer if (session) |*ss| ss.deinit();
        if (self.tools.items.len != 0) {
            const user: common_chat.Msg = .{ .role = "user", .content = "@@USER@@" };
            var inputs = try self.chatInputs(ea, &.{user}, true);
            inputs.tool_choice = choice;
            inputs.parallel_tool_calls = true;
            session = common_chat.session.Session.init(std.heap.c_allocator, self.deps.meta.chat_params.tmpls, self.deps.vocab, &inputs, .{}) catch |e| {
                if (e == error.OutOfMemory) return error.OutOfMemory;
                return self.sendError(ea, try std.mem.concat(ea, u8, &.{ "response.create: ", mod.context.whatOf(e) }), "server_error");
            };
        }
        response.sampling = try self.responseSampling(ea, if (session) |*ss| ss else null);

        {
            self.lock();
            defer self.unlock();
            _ = self.resp_arena.reset(.retain_capacity);
            const rsa = self.ra();
            self.response_id = try randomId(rsa, "resp_");
            self.response_item_id = try randomId(rsa, "item_");
            self.response_text = .empty;
            self.parts = json.Value.emptyArray();
            self.idx_text = -1;
            self.idx_audio = -1;
            self.output = json.Value.emptyArray();
            self.idx_message = -1;
            self.call_items = .empty;
            if (self.parse) |*p| p.deinit(std.heap.c_allocator);
            self.parse = null;
            if (session) |*ss| {
                self.parse = try mod.task.TaskResultState.init(std.heap.c_allocator, ss);
            }
            try self.sendEvent(ea, try object(ea, .{
                .{ "type", "response.created" },
                .{ "response", try object(ea, .{
                    .{ "id", self.response_id },
                    .{ "object", "realtime.response" },
                    .{ "status", "in_progress" },
                    .{ "output_modalities", try outputModalitiesJson(ea, mode) },
                    .{ "output", json.Value.emptyArray() },
                }) },
            }));
        }
        if (!try self.control(.response_create, prompt, &.{}, &response)) {
            return;
        }
        // the response ends the turn, its tokens are in the slot; the footer goes in before the next turn
        self.next_header = self.assistant_footer;
        self.turn_open = false;
        self.tool_results = false;
        self.n_turns += 1;
    }

    /// Ports `on_session_update` (server-realtime.cpp:603 @b1b770d90).
    fn onSessionUpdate(self: *Conversation, ea: std.mem.Allocator, ev: *const json.Value) !void {
        const session = try valueOr(ev, "session");
        if (!session.isObject()) {
            return self.sendInvalid(ea, "session.update: session must be an object");
        }
        if (session.contains("type")) {
            const st = try session.at("type");
            if (st.isString() and !std.mem.eql(u8, st.string, self.type)) {
                const t = st.string;
                if (!std.mem.eql(u8, t, "realtime") and !std.mem.eql(u8, t, "transcription")) {
                    return self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "session.update: unknown session type ", t }));
                }
                if (self.started) {
                    return self.sendInvalid(ea, "session.update: the session type cannot change after the first audio or message");
                }
                self.type = try self.a().dupe(u8, t);
            }
        }
        if (session.contains("instructions") and (try session.at("instructions")).isString()) {
            if (self.started) {
                try self.sendInvalid(ea, "session.update: instructions cannot change after the first audio or message");
            } else {
                self.instructions = try self.a().dupe(u8, (try session.at("instructions")).string);
            }
        }
        if (session.contains("output_modalities")) {
            if (try parseOutputModalities(ea, try session.at("output_modalities"), &self.out_mode)) |err| {
                return self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "session.update: ", err }));
            }
        }
        if (session.contains("tools")) {
            if (self.started) {
                try self.sendInvalid(ea, "session.update: tools cannot change after the first audio or message");
            } else {
                const tools_in = try session.at("tools");
                if (try self.parseTools(ea, tools_in)) |err| {
                    return self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "session.update: ", err }));
                }
                self.tools_json = try tools_in.clone(self.a());
            }
        }
        if (session.contains("tool_choice")) {
            if (try parseToolChoice(ea, try session.at("tool_choice"), &self.tool_choice)) |err| {
                return self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "session.update: ", err }));
            }
        }
        if (session.contains("max_output_tokens") and !parseMaxOutputTokens(try session.at("max_output_tokens"), &self.max_output_tokens)) {
            return self.sendInvalid(ea, "session.update: max_output_tokens must be an integer or \"inf\"");
        }
        const audio = try valueOr(&session, "audio");
        const in = try valueOr(&audio, "input");
        if (in.contains("format") and (try in.at("format")).isObject()) {
            const rate = mod.common.jsonValue(try in.at("format"), "rate", c_int, self.input_rate);
            if (rate < 8000 or rate > 192000) {
                return self.sendInvalid(ea, "session.update: the input audio rate must be between 8000 and 192000 Hz");
            }
            if (rate != self.input_rate) {
                self.input_rate = rate;
                if (self.resampler) |r| mtmd.helper.mtmd_helper_resampler_free(r);
                self.resampler = null;
            }
        }
        if (in.contains("transcription") and (try in.at("transcription")).isObject()) {
            self.language = try self.a().dupe(u8, mod.common.jsonValue(try in.at("transcription"), "language", []const u8, self.language));
        }
        const out = try valueOr(&audio, "output");
        if (out.contains("voice") or out.contains("language")) {
            var voice = self.out_voice;
            if (out.contains("voice")) {
                // string or {"id": name}
                const v = try out.at("voice");
                voice = if (v.isObject()) mod.common.jsonValue(v, "id", []const u8, "") else if (v.isString()) v.string else "";
            }
            const lang = mod.common.jsonValue(&out, "language", []const u8, self.out_language);
            const voices = self.deps.meta.tts_voices;
            if (voice.len != 0 and voices.len != 0) {
                const known = for (voices) |name| {
                    if (std.mem.eql(u8, name, voice)) break true;
                } else false;
                if (!known) {
                    return self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "session.update: unknown voice \"", voice, "\", see \"voices\" in GET /props" }));
                }
            }
            if (self.started and (!std.mem.eql(u8, voice, self.out_voice) or !std.mem.eql(u8, lang, self.out_language))) {
                return self.sendInvalid(ea, "session.update: the output voice and language cannot change after the first audio or message");
            }
            self.out_voice = try self.a().dupe(u8, voice);
            self.out_language = try self.a().dupe(u8, lang);
        }
        if (!self.started and !try self.configure(ea)) {
            return self.sendError(ea, "the chat template does not fit the realtime turn format", "server_error");
        }
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "session.updated" },
            .{ "session", try self.sessionJson(ea) },
        }));
    }

    /// Ports `parse_tools` (server-realtime.cpp:77 @b1b770d90): realtime
    /// tools have the format of the Responses API. A schema that would
    /// fail later, when the chat template builds the tool call grammar, is
    /// rejected here.
    ///
    /// Return: null, or the error's text in `ea`.
    fn parseTools(self: *Conversation, ea: std.mem.Allocator, j: *const json.Value) !?[]const u8 {
        var converted = mod.convert.responsesTools(ea, j) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            return try std.mem.concat(ea, u8, &.{ "tools: ", mod.context.whatOf(e) });
        };
        const parsed = common_chat.msg.toolsParseOaicompat(self.a(), &converted) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            return try std.mem.concat(ea, u8, &.{ "tools: ", mod.context.whatOf(e) });
        };
        // reject the schemas that would fail later, when the chat template builds the tool call grammar
        for (parsed.items) |*tool| {
            const schema = json.parse(ea, tool.parameters) catch |e| {
                if (e == error.OutOfMemory) return error.OutOfMemory;
                return try std.mem.concat(ea, u8, &.{ "tools: ", mod.context.whatOf(e) });
            };
            _ = mod.oaicompat.schemaToGrammar(ea, &schema) catch |e| {
                if (e == error.OutOfMemory) return error.OutOfMemory;
                return try std.mem.concat(ea, u8, &.{ "tools: ", mod.context.whatOf(e) });
            };
        }
        self.tools = parsed;
        return null;
    }

    /// Ports `configure` (server-realtime.cpp:702 @b1b770d90): turn text of
    /// the session type, before anything went into the slot; a model that
    /// speaks takes the speech system prompt of its chat template before
    /// the instructions.
    fn configure(self: *Conversation, ea: std.mem.Allocator) !bool {
        var instructions = self.instructions;
        if (std.mem.eql(u8, self.type, "transcription")) {
            const preset = try common_chat.templates.getAsrPrompt(ea, self.deps.meta.chat_params.tmpls, self.language);
            instructions = preset.system;
        } else if (self.deps.cap.audio_gen) {
            const speech = try common_chat.templates.getSpeechSystemPrompt(ea, self.deps.meta.chat_params.tmpls, self.out_language, self.out_voice);
            if (speech.len != 0) {
                instructions = if (instructions.len == 0) speech else try std.mem.concat(ea, u8, &.{ speech, "\n", instructions });
            }
        }
        return self.deriveTurnText(instructions);
    }

    /// Ports `handle` (server-realtime.cpp:718 @b1b770d90).
    fn handle(self: *Conversation, ea: std.mem.Allocator, msg: []const u8) !void {
        var ev = json.parse(ea, msg) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return self.sendInvalid(ea, "invalid JSON"),
        };
        const t = mod.common.jsonValue(&ev, "type", []const u8, "");
        if (std.mem.eql(u8, t, "session.update") or std.mem.eql(u8, t, "transcription_session.update")) {
            try self.onSessionUpdate(ea, &ev);
        } else if (std.mem.eql(u8, t, "input_audio_buffer.append")) {
            try self.onAppend(ea, &ev);
        } else if (std.mem.eql(u8, t, "input_audio_buffer.commit")) {
            try self.onCommit(ea);
        } else if (std.mem.eql(u8, t, "input_audio_buffer.clear")) {
            // the audio already went into the conversation
            try self.sendEvent(ea, try object(ea, .{.{ "type", "input_audio_buffer.cleared" }}));
        } else if (std.mem.eql(u8, t, "conversation.item.create")) {
            try self.onItemCreate(ea, &ev);
        } else if (std.mem.eql(u8, t, "response.create")) {
            try self.onResponseCreate(ea, &ev);
        } else if (std.mem.eql(u8, t, "response.cancel")) {
            _ = try self.control(.response_cancel, "", &.{}, null);
        } else {
            try self.sendInvalid(ea, try std.mem.concat(ea, u8, &.{ "unknown event type: ", t }));
        }
    }

    //
    // results of the slot
    //

    /// Ports `open_message` (server-realtime.cpp:756 @b1b770d90): the message
    /// item of the response is added with its first delta.
    fn openMessage(self: *Conversation, ea: std.mem.Allocator) !void {
        if (self.idx_message >= 0) {
            return;
        }
        const rsa = self.ra();
        self.idx_message = @intCast(self.output.array.items.len);
        try self.output.pushBack(rsa, try object(rsa, .{
            .{ "id", self.response_item_id },
            .{ "object", "realtime.item" },
            .{ "type", "message" },
            .{ "role", "assistant" },
            .{ "status", "in_progress" },
            .{ "content", json.Value.emptyArray() },
        }));
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "response.output_item.added" },
            .{ "response_id", self.response_id },
            .{ "output_index", self.idx_message },
            .{ "item", try self.output.array.items[@intCast(self.idx_message)].clone(ea) },
        }));
    }

    /// Ports `open_part` (server-realtime.cpp:768 @b1b770d90): a part of the
    /// message is opened on its first delta, in order of appearance.
    fn openPart(self: *Conversation, ea: std.mem.Allocator, part_type: []const u8) !i32 {
        try self.openMessage(ea);
        const rsa = self.ra();
        const part = if (std.mem.eql(u8, part_type, "output_text"))
            try object(rsa, .{ .{ "type", part_type }, .{ "text", "" } })
        else
            try object(rsa, .{ .{ "type", part_type }, .{ "transcript", "" } });
        try self.parts.pushBack(rsa, part);
        const n = self.parts.array.items.len;
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "response.content_part.added" },
            .{ "response_id", self.response_id },
            .{ "item_id", self.response_item_id },
            .{ "output_index", self.idx_message },
            .{ "content_index", n - 1 },
            .{ "part", try part.clone(ea) },
        }));
        return @intCast(n - 1);
    }

    /// Ports `on_text` (server-realtime.cpp:777 @b1b770d90).
    fn onText(self: *Conversation, ea: std.mem.Allocator, text: []const u8) !void {
        if (self.idx_text < 0) {
            self.idx_text = try self.openPart(ea, "output_text");
        }
        try self.response_text.appendSlice(self.ra(), text);
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "response.output_text.delta" },
            .{ "response_id", self.response_id },
            .{ "item_id", self.response_item_id },
            .{ "output_index", self.idx_message },
            .{ "content_index", self.idx_text },
            .{ "delta", text },
        }));
    }

    /// Ports `on_tool_call` (server-realtime.cpp:787 @b1b770d90): a tool call
    /// is an output item of its own: its name comes first, then its
    /// arguments.
    fn onToolCall(self: *Conversation, ea: std.mem.Allocator, diff: *const common_chat.MsgDiff) !void {
        const rsa = self.ra();
        var pos: usize = 0;
        var found: ?usize = null;
        while (pos < self.call_items.items.len and self.call_items.items[pos].index <= diff.tool_call_index) : (pos += 1) {
            if (self.call_items.items[pos].index == diff.tool_call_index) {
                found = self.call_items.items[pos].idx;
                break;
            }
        }
        const idx = found orelse blk: {
            const new_idx = self.output.array.items.len;
            try self.call_items.insert(rsa, pos, .{ .index = diff.tool_call_index, .idx = new_idx });
            try self.output.pushBack(rsa, try object(rsa, .{
                .{ "id", try randomId(rsa, "item_") },
                .{ "object", "realtime.item" },
                .{ "type", "function_call" },
                .{ "status", "in_progress" },
                .{ "name", diff.tool_call_delta.name },
                .{ "call_id", diff.tool_call_delta.id },
                .{ "arguments", "" },
            }));
            try self.sendEvent(ea, try object(ea, .{
                .{ "type", "response.output_item.added" },
                .{ "response_id", self.response_id },
                .{ "output_index", new_idx },
                .{ "item", try self.output.array.items[new_idx].clone(ea) },
            }));
            break :blk new_idx;
        };
        if (diff.tool_call_delta.arguments.len == 0) {
            return;
        }
        const item = &self.output.array.items[idx];
        const args = try std.mem.concat(rsa, u8, &.{ (try item.at("arguments")).string, diff.tool_call_delta.arguments });
        try item.set(rsa, "arguments", try json.Value.initString(rsa, args));
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "response.function_call_arguments.delta" },
            .{ "response_id", self.response_id },
            .{ "item_id", try (try item.at("id")).clone(ea) },
            .{ "output_index", idx },
            .{ "call_id", try (try item.at("call_id")).clone(ea) },
            .{ "delta", diff.tool_call_delta.arguments },
        }));
    }

    /// Ports `on_result` (server-realtime.cpp:807 @b1b770d90).
    fn onResult(self: *Conversation, ea: std.mem.Allocator, res: *const mod.result.Realtime) !void {
        self.lock();
        defer self.unlock();
        const res_text = res.content.text.items;
        if (self.transcribe_item_id.len != 0) {
            if (!res.content.empty()) {
                try self.transcript.appendSlice(std.heap.c_allocator, res_text);
                try self.sendEvent(ea, try object(ea, .{
                    .{ "type", "conversation.item.input_audio_transcription.delta" },
                    .{ "item_id", self.transcribe_item_id },
                    .{ "content_index", 0 },
                    .{ "delta", res_text },
                }));
            }
            if (res.response_done) {
                try self.sendEvent(ea, try object(ea, .{
                    .{ "type", "conversation.item.input_audio_transcription.completed" },
                    .{ "item_id", self.transcribe_item_id },
                    .{ "content_index", 0 },
                    .{ "transcript", self.transcript.items },
                }));
                try self.setTranscribeItem("");
            }
            return;
        }

        // with tools, the text goes through the parser: content and tool calls
        var diffs: std.ArrayList(common_chat.MsgDiff) = .empty;
        if (self.parse) |*p| {
            if (!res.content.empty() or res.response_done) {
                var added = try res.content.clone(ea);
                _ = try p.updateChatMsg(ea, &added, !res.response_done, &diffs, true);
            }
        } else if (!res.content.empty()) {
            try diffs.append(ea, .{ .content_delta = res_text });
        }
        for (diffs.items) |*diff| {
            if (diff.content_delta.len != 0) {
                try self.onText(ea, diff.content_delta);
            }
            if (diff.tool_call_index != common_chat.msg.npos) {
                try self.onToolCall(ea, diff);
            }
        }
        if (res.audio.len != 0) {
            if (self.idx_audio < 0) {
                self.idx_audio = try self.openPart(ea, "output_audio");
            }
            try self.sendEvent(ea, try object(ea, .{
                .{ "type", "response.output_audio.delta" },
                .{ "response_id", self.response_id },
                .{ "item_id", self.response_item_id },
                .{ "output_index", self.idx_message },
                .{ "content_index", self.idx_audio },
                .{ "delta", try encodePcm16(ea, res.audio) },
            }));
        }
        if (!res.response_done) {
            return;
        }

        const rsa = self.ra();
        const status: []const u8 = if (res.cancelled) "cancelled" else "completed";
        if (self.idx_message >= 0) {
            for (self.parts.array.items, 0..) |*part, i| {
                var ev = try object(ea, .{
                    .{ "response_id", self.response_id },
                    .{ "item_id", self.response_item_id },
                    .{ "output_index", self.idx_message },
                    .{ "content_index", i },
                });
                if (@as(i32, @intCast(i)) == self.idx_text) {
                    try part.set(rsa, "text", try json.Value.initString(rsa, self.response_text.items));
                    try ev.set(ea, "type", try json.Value.initString(ea, "response.output_text.done"));
                    try ev.set(ea, "text", try json.Value.initString(ea, self.response_text.items));
                } else {
                    try ev.set(ea, "type", try json.Value.initString(ea, "response.output_audio.done"));
                }
                try self.sendEvent(ea, try ev.clone(ea));
                try ev.erase(ea, "text");
                try ev.set(ea, "type", try json.Value.initString(ea, "response.content_part.done"));
                try ev.set(ea, "part", try part.clone(ea));
                try self.sendEvent(ea, ev);
            }
            try self.output.array.items[@intCast(self.idx_message)].set(rsa, "content", try self.parts.clone(rsa));
        }
        for (self.call_items.items) |ci| {
            const item = &self.output.array.items[ci.idx];
            try self.sendEvent(ea, try object(ea, .{
                .{ "type", "response.function_call_arguments.done" },
                .{ "response_id", self.response_id },
                .{ "item_id", try (try item.at("id")).clone(ea) },
                .{ "output_index", ci.idx },
                .{ "call_id", try (try item.at("call_id")).clone(ea) },
                .{ "name", try (try item.at("name")).clone(ea) },
                .{ "arguments", try (try item.at("arguments")).clone(ea) },
            }));
        }
        for (self.output.array.items, 0..) |*item, i| {
            try item.set(rsa, "status", try json.Value.initString(rsa, status));
            try self.sendEvent(ea, try object(ea, .{
                .{ "type", "response.output_item.done" },
                .{ "response_id", self.response_id },
                .{ "output_index", i },
                .{ "item", try item.clone(ea) },
            }));
            try self.setLastItem((try item.at("id")).string);
        }
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "response.done" },
            .{ "response", try object(ea, .{
                .{ "id", self.response_id },
                .{ "object", "realtime.response" },
                .{ "status", status },
                .{ "output", try self.output.clone(ea) },
                .{ "usage", try object(ea, .{.{ "output_tokens", res.n_decoded }}) },
            }) },
        }));
        if (self.parse) |*p| p.deinit(std.heap.c_allocator);
        self.parse = null;
    }

    //
    // session
    //

    /// The reader thread of `run`: client events, handled in order.
    fn readerLoop(self: *Conversation) void {
        self.readEvents() catch |e| {
            const z = mod.common.dupZ(mod.context.whatOf(e));
            defer mod.common.freeZ(z);
            mod.common.srv(.err, "operator()", "realtime session failed: %s\n", .{z});
        };
        self.closed.store(true, .seq_cst);
        self.conn.close();
    }

    fn readEvents(self: *Conversation) !void {
        var msg: std.ArrayList(u8) = .empty;
        defer msg.deinit(std.heap.c_allocator);
        var ev_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer ev_arena.deinit();
        while (try self.conn.read(std.heap.c_allocator, &msg) != 0) {
            _ = ev_arena.reset(.retain_capacity);
            const ea = ev_arena.allocator();
            self.handle(ea, msg.items) catch |e| {
                try self.sendError(ea, mod.context.whatOf(e), "server_error");
            };
        }
    }

    /// Ports `run` (server-realtime.cpp:892 @b1b770d90).
    fn run(self: *Conversation) !void {
        var ev_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer ev_arena.deinit();
        const ea = ev_arena.allocator();

        self.id = try randomId(self.a(), "sess_");
        self.input_rate = mtmd.mtmd.mtmd_get_audio_sample_rate(self.deps.mctx.?);
        if (self.deps.cap.audio_gen) {
            self.output_rate = mtmd.mtmd.mtmd_gen_audio_get_info(self.deps.mctx.?).sample_rate;
        }
        if (!try self.configure(ea)) {
            try self.sendError(ea, "the chat template does not fit the realtime turn format", "server_error");
            self.conn.close();
            return;
        }

        var rd = try mod.context.Reader.init(std.heap.c_allocator, self.deps.queue_tasks, self.deps.queue_results, 1);
        defer rd.deinit();
        {
            var task = mod.task.Task.initType(.realtime);
            task.id = rd.getNewId();
            const ta = try task.arenaAlloc(std.heap.c_allocator);
            task.params = try mod.task.TaskParams.init(ta);
            task.params.sampling = try mod.task.deepClone(ta, self.deps.params.sampling);
            task.params.stream = true;
            task.params.oaicompat_cmpl_id = try ta.dupe(u8, self.id);
            task.params.oaicompat_model = try ta.dupe(u8, self.deps.meta.model_name);
            try rd.postTask(task, false);
        }
        // the slot answers once it holds the session
        {
            const stop: Stop = .{ .conv = self, .with_closed = false };
            const result = (try rd.next(stop)) orelse {
                self.conn.close();
                return;
            };
            defer result.destroy(std.heap.c_allocator);
            if (result.isError()) {
                var j = try result.toJson(ea);
                try self.sendError(ea, mod.common.jsonValue(&j, "message", []const u8, "server error"), "server_error");
                self.conn.close();
                return;
            }
        }
        try self.sendEvent(ea, try object(ea, .{
            .{ "type", "session.created" },
            .{ "session", try self.sessionJson(ea) },
        }));

        // client events are handled on this thread, results are sent from the loop below
        const reader = try std.Thread.spawn(.{}, readerLoop, .{self});

        const stop: Stop = .{ .conv = self, .with_closed = false };
        while (true) {
            _ = ev_arena.reset(.retain_capacity);
            const result = (try rd.next(stop)) orelse break;
            defer result.destroy(std.heap.c_allocator);
            if (result.isError()) {
                var j = try result.toJson(ea);
                try self.sendError(ea, mod.common.jsonValue(&j, "message", []const u8, "server error"), "server_error");
                break;
            }
            switch (result.kind) {
                .realtime => |*res| self.onResult(ea, res) catch |e| {
                    try self.sendError(ea, mod.context.whatOf(e), "server_error");
                    break;
                },
                else => {},
            }
        }

        self.closed.store(true, .seq_cst);
        self.conn.close();
        reader.join();
    }
};

/// `json.value(key, json::object())`: the member, or an empty object; of
/// anything but an object it throws.
fn valueOr(v: *const json.Value, key: []const u8) !json.Value {
    if (!v.isObject()) {
        return json.errors.fail(.type_error, 306, "cannot use value() with {s}", .{v.typeName()});
    }
    if (v.contains(key)) return (try v.at(key)).*;
    return json.Value.emptyObject();
}

/// Ports `encode_pcm16` (server-realtime.cpp:751 @b1b770d90).
///
/// Return: owned by `a`.
fn encodePcm16(a: std.mem.Allocator, samples: []const f32) common_chat.exception.Error![]u8 {
    return mod.oaicompat.base64Encode(a, try mod.common.pcmF32ToS16(a, samples));
}

/// The PCM16 of an `input_audio_buffer.append`, as float samples.
///
/// Return: owned by `a`.
pub fn pcm16ToF32(a: std.mem.Allocator, pcm: []const u8) common_chat.exception.Error![]f32 {
    const samples = try a.alloc(f32, pcm.len / 2);
    for (samples, 0..) |*s, i| {
        const v = std.mem.readInt(i16, pcm[2 * i ..][0..2], .little);
        s.* = @as(f32, @floatFromInt(v)) / 32768.0;
    }
    return samples;
}

/// Ports `server_realtime_conversation` (server-realtime.cpp:967 @b1b770d90):
/// a realtime conversation over websocket, for models that answer after the
/// audio (e.g. LFM3-Audio): the audio is prefilled as it arrives, responses
/// follow output_modalities or the model's own switching.
pub fn conversation(deps: Deps, conn: *mod.http.WsConn) void {
    var session: Conversation = .{
        .deps = deps,
        .conn = conn,
        .arena = std.heap.ArenaAllocator.init(std.heap.c_allocator),
        .resp_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator),
        .out_language = deps.params.tts_lang,
        .out_voice = deps.params.tts_voice,
    };
    defer {
        if (session.resampler) |r| mtmd.helper.mtmd_helper_resampler_free(r);
        if (session.parse) |*p| p.deinit(std.heap.c_allocator);
        std.heap.c_allocator.free(session.last_item_id);
        std.heap.c_allocator.free(session.transcribe_item_id);
        session.transcript.deinit(std.heap.c_allocator);
        session.resp_arena.deinit();
        session.arena.deinit();
    }
    session.run() catch |e| {
        const z = mod.common.dupZ(mod.context.whatOf(e));
        defer mod.common.freeZ(z);
        mod.common.srv(.err, "server_realtime_conversation", "realtime session failed: %s\n", .{z});
        conn.close();
    };
}

// -----------------------------------------------------------------------------
// server_realtime_slot

/// Ports `server_realtime_op` (server-realtime.h:38 @b1b770d90): one control
/// of a realtime conversation. Its strings are `arena`'s, the control task's
/// own.
const Op = struct {
    action: mod.task.ControlAction,
    text: []const u8,
    audio: []const f32,
    response: mod.task.ControlResponse,
    arena: ?*std.heap.ArenaAllocator,

    fn deinit(self: *Op) void {
        if (self.arena) |ar| {
            ar.deinit();
            std.heap.c_allocator.destroy(ar);
        }
        self.arena = null;
    }
};

/// Ports `server_realtime_slot` (server-realtime.h:47 @b1b770d90): the
/// conversation of a slot that holds a realtime session
/// (SERVER_TASK_TYPE_REALTIME), it runs on the server thread. The controls
/// run in order; the sequence is decoded by its own calls, not in the shared
/// batch.
pub const RealtimeSlot = struct {
    err_msg: []const u8 = "",
    err_type: mod.common.ErrorType = .server,

    ctx: ?*llama.impl.llama_c.llama_context,
    model: ?*const llama.impl.llama_c.llama_model,
    vocab: ?*const llama.impl.llama_c.llama_vocab,
    mctx: ?*mtmd.mtmd.Context,
    seq_id: llama.impl.llama_c.llama_seq_id,
    n_ctx: i32,
    seed: u32,
    queue_results: *mod.context.ResultQueue,
    id_task: c_int,
    index: usize,

    ops: std.ArrayList(Op) = .empty,
    n_past: llama.impl.llama_c.llama_pos = 0, // positions of the conversation in the sequence
    responding: bool = false,
    cancelled: bool = false,
    n_predict: i32 = 0,
    n_decoded: i32 = 0,
    sampled: llama.impl.llama_c.llama_token = mod.common.token_null, // next text token of the response, sampled right after the slot's own decode

    stream: ?*mtmd.mtmd.AudioStream = null, // the audio of the open turn
    gen: ?*mtmd.helper_gen.GenAudio = null, // set if the model generates audio
    h_state: std.ArrayList(f32) = .empty, // hidden state that the gen-audio pipeline steps from
    smpl: ?*tools_common.sampling.Sampler = null,
    preserved: std.ArrayList(llama.impl.llama_c.llama_token) = .empty, // special tokens of the response that go out as text (tool calls)

    /// Ports the constructor of `server_realtime_slot` (server-realtime.cpp:976 @b1b770d90).
    pub fn create(ctx: ?*llama.impl.llama_c.llama_context, mctx: ?*mtmd.mtmd.Context, seq_id: llama.impl.llama_c.llama_seq_id, n_ctx: i32, params: *const tools_common.params.Sampling, queue_results: *mod.context.ResultQueue, id_task: c_int, index: usize) !*RealtimeSlot {
        const self = try std.heap.c_allocator.create(RealtimeSlot);
        const model = llama.api.context.llama_get_model(ctx);
        self.* = .{
            .ctx = ctx,
            .model = model,
            .vocab = llama.api.model.llama_model_get_vocab(model),
            .mctx = mctx,
            .seq_id = seq_id,
            .n_ctx = n_ctx,
            .seed = params.seed,
            .queue_results = queue_results,
            .id_task = id_task,
            .index = index,
        };
        if (mtmd.mtmd.mtmd_gen_audio_get_info(mctx.?).type != mtmd.mtmd.GenAudioType.none) {
            self.gen = mtmd.helper_gen.mtmd_helper_gen_audio_init(ctx, mctx.?);
        }
        return self;
    }

    pub fn destroy(self: *RealtimeSlot) void {
        for (self.ops.items) |*op| op.deinit();
        self.ops.deinit(std.heap.c_allocator);
        if (self.stream) |st| mtmd.mtmd.mtmd_audio_stream_free(st);
        if (self.gen) |g| mtmd.helper_gen.mtmd_helper_gen_audio_free(g);
        self.h_state.deinit(std.heap.c_allocator);
        if (self.smpl) |s| s.destroy();
        self.preserved.deinit(std.heap.c_allocator);
        std.heap.c_allocator.free(self.err_msg);
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `push` (server-realtime.cpp:985 @b1b770d90): takes the control
    /// task's parameters, its arena with them.
    pub fn push(self: *RealtimeSlot, task: *mod.task.Task) !void {
        try self.ops.append(std.heap.c_allocator, .{
            .action = task.params.control_action,
            .text = task.params.control_text,
            .audio = task.params.control_audio,
            .response = task.params.control_response,
            .arena = task.arena,
        });
        task.arena = null;
    }

    /// Ports `cancel` (server-realtime.cpp:989 @b1b770d90): stops the
    /// response being generated.
    pub fn cancel(self: *RealtimeSlot) void {
        self.cancelled = self.responding;
    }

    /// Ports `is_idle` (server-realtime.cpp:993 @b1b770d90): nothing to run
    /// until the next control.
    pub fn isIdle(self: *const RealtimeSlot) bool {
        return self.ops.items.len == 0 and !self.responding;
    }

    /// Ports `fail` (server-realtime.cpp:997 @b1b770d90).
    fn fail(self: *RealtimeSlot, msg: []const u8, ty: mod.common.ErrorType) !bool {
        std.heap.c_allocator.free(self.err_msg);
        self.err_msg = "";
        self.err_msg = try std.heap.c_allocator.dupe(u8, msg);
        self.err_type = ty;
        return false;
    }

    /// Ports `send_result` (server-realtime.cpp:1003 @b1b770d90): the text
    /// goes with its tokens.
    fn sendResult(self: *RealtimeSlot, content: *const common_chat.input.Input, audio: []const f32, response_done: bool) !void {
        const res = try mod.result.Result.create(std.heap.c_allocator, .realtime);
        errdefer res.destroy(std.heap.c_allocator);
        const r = &res.kind.realtime;
        res.id = self.id_task;
        res.index = self.index;
        r.content = try content.clone(res.alloc());
        r.audio = try res.alloc().dupe(f32, audio);
        r.response_done = response_done;
        r.cancelled = response_done and self.cancelled;
        r.n_decoded = self.n_decoded;
        try self.queue_results.send(res);
    }

    /// Ports `decode` (server-realtime.cpp:1016 @b1b770d90): appends tokens or
    /// embedding rows to the conversation, logits of the last one if asked.
    fn decode(self: *RealtimeSlot, tokens: ?[*]const llama.impl.llama_c.llama_token, rows: ?[*]const f32, n: i32, logits_last: bool) !bool {
        const n_embd: usize = @intCast(llama.api.model.llama_model_n_embd_inp(self.model));
        const n_batch: i32 = @intCast(llama.api.context.llama_n_batch(self.ctx));
        if (self.n_past + n >= self.n_ctx) {
            return self.fail("the conversation does not fit in the context", .exceed_context_size);
        }
        var c_i0: i32 = 0;
        while (c_i0 < n) : (c_i0 += n_batch) {
            const n_cur = @min(n_batch, n - c_i0);
            var b = llama.api.batch.llama_batch_init(n_cur, if (rows != null) @intCast(n_embd) else 0, 1);
            var i: i32 = 0;
            while (i < n_cur) : (i += 1) {
                const ui: usize = @intCast(i);
                const src: usize = @intCast(c_i0 + i);
                if (rows) |r| {
                    @memcpy(b.embd[ui * n_embd ..][0..n_embd], r[src * n_embd ..][0..n_embd]);
                } else {
                    b.token[ui] = tokens.?[src];
                }
                b.pos[ui] = self.n_past + c_i0 + i;
                b.n_seq_id[ui] = 1;
                b.seq_id[ui][0] = self.seq_id;
                b.logits[ui] = @intFromBool(logits_last and c_i0 + i == n - 1);
            }
            b.n_tokens = n_cur;
            const ret = llama.api.context.llama_decode(self.ctx, b);
            llama.api.batch.llama_batch_free(b);
            if (ret != 0) {
                return self.fail("failed to decode the conversation", .server);
            }
        }
        self.n_past += n;
        return true;
    }

    /// Ports `audio_rows` (server-realtime.cpp:1048 @b1b770d90): decodes the
    /// rows the audio stream has ready.
    fn audioRows(self: *RealtimeSlot) !bool {
        while (true) {
            const n = mtmd.mtmd.mtmd_audio_stream_step(self.stream.?, mod.common.token_null);
            if (n == -2) {
                return self.fail("audio stream error", .server);
            }
            if (n <= 0) {
                return true;
            }
            if (!try self.decode(null, mtmd.mtmd.mtmd_audio_stream_get_output_embd(self.stream.?), n, false)) {
                return false;
            }
        }
    }

    /// Ports `sample_next` (server-realtime.cpp:1065 @b1b770d90): samples the
    /// next text token of a response from the logits of the slot's own last
    /// decode; several slots respond in turns on the same context, so this
    /// runs before any other slot decodes.
    fn sampleNext(self: *RealtimeSlot) !void {
        self.sampled = mod.common.token_null;
        if (self.gen != null and !mtmd.helper_gen.mtmd_helper_gen_audio_next_is_text(self.gen.?)) {
            return;
        }
        self.sampled = try self.smpl.?.sample(self.ctx, -1, false);
        self.smpl.?.accept(self.sampled, true);
    }

    /// Ports `response_start` (server-realtime.cpp:1074 @b1b770d90).
    fn responseStart(self: *RealtimeSlot, op: *Op) !bool {
        const sparams = &op.response.sampling;
        self.n_predict = op.response.n_predict;
        self.n_decoded = 0;
        self.cancelled = false;

        if (self.gen != null) {
            // the gen-audio pipeline continues the conversation and follows the model's text/audio switching
            var inp = mod.task.TtsInp.init();
            defer inp.deinit();
            const ar = op.arena.?.allocator();
            inp.prompt_str = try ar.dupeSentinel(u8, op.text, 0);
            inp.data.seq_id = self.seq_id;
            inp.data.n_past = self.n_past;
            inp.data.seed = self.seed;
            inp.data.stream = true;
            inp.data.out_type = mtmd.helper_gen.OutType.pcm;
            inp.data.out_mode = op.response.out_mode;
            inp.lang_str = try ar.dupeSentinel(u8, op.response.lang, 0);
            // predefined voice -> speaker ref, LFM3 uses the system prompt instead
            if (op.response.voice.len != 0 and mtmd.mtmd.mtmd_gen_audio_get_info(self.mctx.?).n_voices > 0) {
                const voice = mtmd.mtmd.mtmd_gen_audio_get_voice(self.mctx.?, (try ar.dupeSentinel(u8, op.response.voice, 0)).ptr);
                if (voice == null) {
                    return self.fail(try std.mem.concat(ar, u8, &.{ "unknown voice \"", op.response.voice, "\"" }), .invalid_request);
                }
                inp.setSpeakerRef(voice);
            }
            if (mtmd.helper_gen.mtmd_helper_gen_audio_set_input(self.gen.?, inp.get()) != 0) {
                return self.fail("failed to start the response", .server);
            }
            // the pipeline may bias some text tokens, e.g. the switch to audio of a text-only response
            var n_bias: usize = 0;
            const bias = mtmd.helper_gen.mtmd_helper_gen_audio_get_logit_bias(self.gen.?, &n_bias);
            if (n_bias != 0) {
                try sparams.logit_bias.appendSlice(ar, bias.?[0..n_bias]);
            }
            var ret: i32 = mtmd.helper_gen.mtmd_helper_gen_audio_step_prompt(self.gen.?, @intCast(llama.api.context.llama_n_batch(self.ctx)));
            while (ret > 0) {
                ret = mtmd.helper_gen.mtmd_helper_gen_audio_step_prompt(self.gen.?, @intCast(llama.api.context.llama_n_batch(self.ctx)));
            }
            if (ret < 0) {
                return self.fail("failed to start the response", .server);
            }
            const h = llama.api.context.llama_get_embeddings_ith(self.ctx, -1).?;
            self.h_state.clearRetainingCapacity();
            try self.h_state.appendSlice(std.heap.c_allocator, h[0..@intCast(llama.api.model.llama_model_n_embd(self.model))]);
        } else {
            var tmp = std.heap.ArenaAllocator.init(std.heap.c_allocator);
            defer tmp.deinit();
            const toks = try mod.common.tokenize(tmp.allocator(), self.vocab, op.text, false, true);
            if (toks.len == 0) {
                return self.fail("failed to start the response", .server);
            }
            if (!try self.decode(toks.ptr, null, @intCast(toks.len), true)) {
                return false;
            }
        }
        if (self.smpl) |s| s.destroy();
        self.smpl = null;
        self.smpl = try tools_common.sampling.init(self.model, sparams);
        self.preserved.clearRetainingCapacity();
        try self.preserved.appendSlice(std.heap.c_allocator, sparams.preserved_tokens.items);
        // the logits of the last decode are this slot's only until another slot decodes
        try self.sampleNext();
        self.responding = true;
        return true;
    }

    /// Ports `piece` (server-realtime.cpp:1132 @b1b770d90): control tokens
    /// stay out of the text, except the preserved ones the client parses
    /// (tool calls).
    ///
    /// Return: owned by `a`.
    fn piece(self: *const RealtimeSlot, a: std.mem.Allocator, tok: llama.impl.llama_c.llama_token) ![]const u8 {
        const p = std.mem.indexOfScalar(llama.impl.llama_c.llama_token, self.preserved.items, tok) != null;
        if (p or !llama.api.vocab.llama_vocab_is_control(self.vocab, tok)) {
            return tools_common.common.tokenToPiece(a, self.vocab, tok, p);
        }
        return "";
    }

    /// Ports `response_step` (server-realtime.cpp:1138 @b1b770d90): one text
    /// token or audio frame of the response being generated.
    fn responseStep(self: *RealtimeSlot) !bool {
        var tmp = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer tmp.deinit();
        const ta = tmp.allocator();

        var stop = false;
        var content = common_chat.input.Input.init(ta);
        var audio: []const f32 = &.{};

        if (self.gen != null) {
            var h_next: [*c]const f32 = null;
            if (mtmd.helper_gen.mtmd_helper_gen_audio_step_gen(self.gen.?, self.sampled, self.h_state.items.ptr, &h_next, &stop) != 0) {
                return self.fail("response generation failed", .server);
            }
            if (self.sampled != mod.common.token_null and !stop) {
                try content.append(try self.piece(ta, self.sampled), self.sampled);
            }
            if (h_next != null) {
                const n: usize = @intCast(llama.api.model.llama_model_n_embd(self.model));
                self.h_state.clearRetainingCapacity();
                try self.h_state.appendSlice(std.heap.c_allocator, h_next[0..n]);
            }
        } else {
            stop = llama.api.vocab.llama_vocab_is_eog(self.vocab, self.sampled);
            if (!stop) {
                try content.append(try self.piece(ta, self.sampled), self.sampled);
                if (!try self.decode(@ptrCast(&self.sampled), null, 1, true)) {
                    return false;
                }
            }
        }
        self.n_decoded += 1;

        // the pipeline decodes on its own, the response ends before the context is full
        const ctx_full = if (self.gen != null)
            llama.api.context.llama_memory_seq_pos_max(llama.api.context.llama_get_memory(self.ctx), self.seq_id) + 2 >= self.n_ctx
        else
            self.n_past + 1 >= self.n_ctx;
        const done = stop or self.cancelled or ctx_full or (self.n_predict > 0 and self.n_decoded >= self.n_predict);
        if (!done) {
            try self.sampleNext();
        }
        if (self.gen != null) {
            if (done) {
                _ = mtmd.helper_gen.mtmd_helper_gen_audio_flush(self.gen.?);
            }
            var sample_rate: i32 = 0;
            var data: [*c]const u8 = null;
            var data_len: usize = 0;
            if (mtmd.helper_gen.mtmd_helper_gen_audio_get_output(self.gen.?, &sample_rate, @ptrCast(&data), &data_len, null) == 0 and data_len > 0) {
                const n = data_len / @sizeOf(f32);
                const samples = try ta.alloc(f32, n);
                @memcpy(std.mem.sliceAsBytes(samples), data[0 .. n * @sizeOf(f32)]);
                audio = samples;
            }
        }
        if (!content.empty() or audio.len != 0) {
            try self.sendResult(&content, audio, false);
        }
        if (done) {
            if (self.gen != null) {
                // the pipeline decoded the response, the sequence ends where it stopped
                self.n_past = llama.api.context.llama_memory_seq_pos_max(llama.api.context.llama_get_memory(self.ctx), self.seq_id) + 1;
            }
            const none = common_chat.input.Input.init(ta);
            try self.sendResult(&none, &.{}, true);
            self.responding = false;
            self.cancelled = false;
        }
        return true;
    }

    /// Ports `update` (server-realtime.cpp:1199 @b1b770d90): runs the queued
    /// controls, or one step of the response. Only the gen-audio pipeline
    /// steps from the hidden state, so only it turns embeddings on.
    ///
    /// Return: false on error, see `err_msg` and `err_type`.
    pub fn update(self: *RealtimeSlot) !bool {
        llama.api.context.llama_set_embeddings(self.ctx, self.gen != null);

        if (self.responding) {
            return self.responseStep();
        }

        while (self.ops.items.len != 0 and !self.responding) {
            var op = self.ops.orderedRemove(0);
            defer op.deinit();
            switch (op.action) {
                .text_append => {
                    var tmp = std.heap.ArenaAllocator.init(std.heap.c_allocator);
                    defer tmp.deinit();
                    const toks = try mod.common.tokenize(tmp.allocator(), self.vocab, op.text, false, true);
                    if (toks.len != 0 and !try self.decode(toks.ptr, null, @intCast(toks.len), false)) {
                        return false;
                    }
                },
                .audio_append => {
                    if (self.stream == null) {
                        self.stream = mtmd.mtmd.mtmd_audio_stream_init(self.mctx.?, mtmd.mtmd.mtmd_audio_stream_params_default());
                        if (self.stream == null) {
                            return self.fail("failed to initialize the audio stream", .server);
                        }
                    }
                    _ = mtmd.mtmd.mtmd_audio_stream_push(self.stream.?, op.audio.ptr, op.audio.len);
                    if (!try self.audioRows()) {
                        return false;
                    }
                },
                .audio_commit, .audio_end => {
                    // the end of the audio of a turn
                    if (self.stream != null) {
                        _ = mtmd.mtmd.mtmd_audio_stream_flush(self.stream.?, true);
                        if (!try self.audioRows()) {
                            return false;
                        }
                        mtmd.mtmd.mtmd_audio_stream_free(self.stream);
                        self.stream = null;
                    }
                },
                .response_create => {
                    if (!try self.responseStart(&op)) {
                        return false;
                    }
                },
                .clear => {
                    _ = llama.api.context.llama_memory_seq_rm(llama.api.context.llama_get_memory(self.ctx), self.seq_id, -1, -1);
                    if (self.stream) |st| mtmd.mtmd.mtmd_audio_stream_free(st);
                    self.stream = null;
                    self.n_past = 0;
                },
                else => {},
            }
        }
        return true;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "pcm16 samples are read as the C++ reads them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try pcm16ToF32(arena.allocator(), &.{ 0x00, 0x80, 0xff, 0x7f, 0x00, 0x00, 0x01 });
    try std.testing.expectEqual(@as(usize, 3), s.len);
    try std.testing.expectEqual(@as(f32, -1.0), s[0]);
    try std.testing.expectEqual(@as(f32, 32767.0 / 32768.0), s[1]);
    try std.testing.expectEqual(@as(f32, 0.0), s[2]);
}
