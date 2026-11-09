//! A task's results -- completions final and partial, embeddings, rerank
//! scores, errors, metrics, slots, LoRA -- and each one's JSON in every API
//! the server speaks: llama.cpp's own, OpenAI's completions, chat,
//! Responses and transcriptions, and Anthropic's messages, streamed or not.
//!
//! # Provenance
//!
//! Ports `server_task_result` and every `server_task_result_*` from
//! `llama.cpp/tools/server/server-task.cpp` and
//! `llama.cpp/tools/server/server-task.h`, and
//! `server_chat_msg_diff_to_json_oaicompat` from
//! `llama.cpp/tools/server/server-chat.cpp` (llama.cpp v0.8.0,
//! `8642541b2`). Each declaration names what it ports and the line it
//! began at.
//!
//! # One type for the class hierarchy
//!
//! The C++ results are subclasses of `server_task_result` behind a
//! `std::unique_ptr`, dispatched through five virtuals. Here a `Result` is
//! the base's members and a tagged union of the subclasses' -- the
//! `ResultPtr` `queue.zig` is generic over is `*Result` -- and each virtual
//! is a `switch`. A result owns an arena for everything in it, freed by
//! `destroy`.
//!
//! # What a client cannot compare
//!
//! `created`, `created_at` and `completed_at` are `std::time(0)`, and the
//! Responses API's `rs_`/`msg_` ids in a non-streamed answer are fresh
//! random strings; `server-task-diff` masks both.

const std = @import("std");
const json = @import("json");
const chat = @import("common_chat");
const tools = @import("tools_common");
const common = @import("common.zig");
const task = @import("task.zig");

const c = tools.c;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Error = common.Error;
const Obj = common.Obj;
const Token = common.Token;
const TaskParams = task.TaskParams;
const TaskResultState = task.TaskResultState;
const CompletionTokenOutput = task.CompletionTokenOutput;
const ResponseType = task.ResponseType;
const StopType = task.StopType;

extern fn time(t: ?*i64) i64;

/// Ports `llama_build_info` (build-info.cpp.in:27 @8642541b2): `b<number>-<commit>`.
pub const build_info = std.fmt.comptimePrint("b{d}-{s}", .{ tools.build_info.build_number, tools.build_info.commit });

/// `{"event": name, "data": data}`, the shape every streamed event has.
fn event(a: Allocator, name: []const u8, data: Value) Error!Value {
    var o = Obj.init(a);
    errdefer o.deinit();
    try o.put("event", name);
    try o.put("data", data);
    return o.done();
}

/// Ports `server_chat_msg_diff_to_json_oaicompat` (server-chat.cpp:643 @8642541b2):
/// one OpenAI chat delta.
///
/// Return: owned by `a`.
pub fn chatMsgDiffToJsonOaicompat(a: Allocator, diff: *const chat.MsgDiff) Error!Value {
    var delta = Obj.init(a);
    errdefer delta.deinit();
    if (diff.reasoning_content_delta.len != 0) {
        try delta.put("reasoning_content", diff.reasoning_content_delta);
    }
    if (diff.content_delta.len != 0) {
        try delta.put("content", diff.content_delta);
    }
    if (diff.tool_call_index != chat.msg.npos) {
        var tool_call = Obj.init(a);
        errdefer tool_call.deinit();
        try tool_call.put("index", diff.tool_call_index);
        if (diff.tool_call_delta.id.len != 0) {
            try tool_call.put("id", diff.tool_call_delta.id);
            try tool_call.put("type", "function");
        }
        if (diff.tool_call_delta.name.len != 0 or diff.tool_call_delta.arguments.len != 0) {
            var function = Obj.init(a);
            errdefer function.deinit();
            if (diff.tool_call_delta.name.len != 0) {
                try function.put("name", diff.tool_call_delta.name);
            }
            if (diff.tool_call_delta.arguments.len != 0) {
                try function.put("arguments", diff.tool_call_delta.arguments);
            }
            try tool_call.put("function", function.done());
        }
        var arr = Value.emptyArray();
        errdefer arr.deinit(a);
        try arr.pushBack(a, tool_call.done());
        try delta.put("tool_calls", arr);
    }
    return delta.done();
}

// -----------------------------------------------------------------------------
// the subclasses' members

/// Ports `server_task_result_cmpl_final` (server-task.h:406 @8642541b2).
/// `content` is in the result's arena.
pub const CmplFinal = struct {
    content: chat.input.Input = .{ .gpa = std.heap.c_allocator },
    tokens: []const Token = &.{},

    stream: bool = false,
    include_usage: bool = false,
    stats: common.SlotStats = .{},
    prompt: []const u8 = "",

    truncated: bool = false,
    n_decoded: i32 = 0,
    n_prompt_tokens: i32 = 0,
    n_prompt_tokens_cache: i32 = 0,
    n_tokens_cached: i32 = 0,
    has_new_line: bool = false,
    stopping_word: []const u8 = "",
    stop: StopType = .none,

    post_sampling_probs: bool = false,
    probs_output: []const CompletionTokenOutput = &.{},
    response_fields: []const []const u8 = &.{},

    generation_params: TaskParams = .{},

    // response formatting
    verbose: bool = false,
    res_type: ResponseType = .none,
    oaicompat_model: []const u8 = "",
    oaicompat_cmpl_id: []const u8 = "",
    oaicompat_msg: chat.Msg = .{}, // to be populated by update()

    oaicompat_msg_diffs: std.ArrayList(chat.MsgDiff) = .empty, // to be populated by update()
    is_updated: bool = false,

    // for OpenAI Responses API
    oai_resp_id: []const u8 = "",
    oai_resp_reasoning_id: []const u8 = "",
    oai_resp_message_id: []const u8 = "",
};

/// Ports `server_task_result_cmpl_partial` (server-task.h:481 @8642541b2).
/// `content` is in the result's arena.
pub const CmplPartial = struct {
    content: chat.input.Input = .{ .gpa = std.heap.c_allocator },
    tokens: []const Token = &.{},

    n_decoded: i32 = 0,
    n_prompt_tokens: i32 = 0,
    n_prompt_tokens_cache: i32 = 0,

    post_sampling_probs: bool = false,
    is_progress: bool = false,
    is_begin: bool = false, // whether to send 200 status to HTTP client (begin of SSE stream)
    // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    audio_commit_drained: bool = false, // realtime: non-final commit has fully decoded
    prob_output: CompletionTokenOutput = .{},
    stats: common.SlotStats = .{},
    progress: task.ResultPromptProgress = .{},

    // response formatting
    verbose: bool = false,
    res_type: ResponseType = .none,
    oaicompat_model: []const u8 = "",
    oaicompat_cmpl_id: []const u8 = "",
    oaicompat_msg_diffs: std.ArrayList(chat.MsgDiff) = .empty, // to be populated by update()
    is_updated: bool = false,

    // Streaming state copied from task_result_state for this chunk
    thinking_block_started: bool = false,
    text_block_started: bool = false,

    // for OpenAI Responses API
    oai_resp_created: bool = false,
    oai_resp_id: []const u8 = "",
    oai_resp_reasoning_id: []const u8 = "",
    oai_resp_message_id: []const u8 = "",
    oai_resp_fc_id: []const u8 = "",

    // for Anthropic API: track if any reasoning content has been generated
    anthropic_has_reasoning: bool = false,
};

/// Ports `server_task_result_embd` (server-task.h:541 @8642541b2).
pub const Embd = struct {
    embedding: []const []const f32 = &.{},

    n_tokens: i32 = 0,

    // response formatting
    res_type: ResponseType = .none,
};

/// Ports `server_task_result_realtime` (server-task.h:557 @8642541b2): one
/// event of a realtime conversation: text and audio of the response being
/// generated, or its end. `text` and `audio` are in the result's arena.
pub const Realtime = struct {
    text: []const u8 = "",
    audio: []const f32 = &.{}, // PCM F32 at the sample rate of mtmd_gen_audio_get_info()
    response_done: bool = false,
    cancelled: bool = false,
    n_decoded: i32 = 0,
};

/// Ports `server_task_result_tts` (server-task.h:571 @8642541b2). `audio` is
/// in the result's arena.
pub const Tts = struct {
    audio: []const u8 = "", // raw bytes for this chunk (WAV or PCM, per request's out_type)
    sample_rate: i32 = 0,
    final: bool = false, // true for the last chunk of a request
};

/// Ports `server_task_result_rerank` (server-task.h:581 @8642541b2).
pub const Rerank = struct {
    score: f32 = -1e6,

    n_tokens: i32 = 0,
};

/// Ports `server_task_result_decision` (server-task.h:589 @8642541b2).
pub const Decision = struct {
    scores: []const f32 = &.{}, // one raw model output per option

    n_tokens: i32 = 0,
};

/// Ports `server_task_result_error` (server-task.h:597 @8642541b2).
pub const ErrorResult = struct {
    err_type: common.ErrorType = .server,
    err_msg: []const u8 = "",

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    n_prompt_tokens: i32 = 0,
    n_ctx: i32 = 0,
};

/// Ports `server_task_result_metrics` (server-task.h:613 @8642541b2): used by
/// `/metrics`.
pub const MetricsResult = struct {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    n_processing_slots: c_int = 0,
    n_tasks_deferred: c_int = 0,

    metrics: common.Metrics = .{},
};

/// Ports `server_task_result_metrics`'s `metric_item` (server-task.h:622 @8642541b2).
pub const MetricItem = struct {
    name: []const u8,
    description: []const u8,
    value: f64, // prometheus values are always float64
};

/// Ports `server_task_result_slots` (server-task.h:631 @8642541b2): used by `/slots`.
pub const Slots = struct {
    n_idle_slots: c_int = 0,

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    slots_data: Value = .{ .array = .empty },
};

/// Ports `server_task_result_slot_save_load` (server-task.h:641 @8642541b2).
pub const SlotSaveLoad = struct {
    filename: []const u8 = "",
    is_save: bool = false, // true = save, false = load

    n_tokens: usize = 0,
    n_bytes: usize = 0,
    t_ms: f64 = 0,
};

/// Ports `server_task_result_slot_erase` (server-task.h:652 @8642541b2).
pub const SlotErase = struct {
    n_erased: usize = 0,
};

/// Ports `server_task_result_control` (server-task.h:658 @8642541b2).
pub const Control = struct {
    success: bool = false,
    message: []const u8 = "", // optional detail when success is false
};

/// Ports `server_task_result_get_lora` (server-task.h:671 @8642541b2).
pub const GetLora = struct {
    /// Ports `server_task_result_get_lora`'s `lora` (server-task.h:672 @8642541b2).
    pub const Lora = struct {
        info: common.LoraInfo,
        alora_invocation_string: []const u8 = "",
        alora_invocation_tokens: []const Token = &.{},
    };
    loras: []const Lora = &.{},
};

/// Ports `server_task_result_router` (server-task.h:758 @8642541b2): used
/// exclusively by router mode.
pub const Router = struct {
    data: Value = .null,
};

// -----------------------------------------------------------------------------
// server_task_result

/// Ports `server_task_result` (server-task.h:357 @8642541b2) and its
/// subclasses, as one type (see the file's doc comment).
pub const Result = struct {
    id: c_int = -1,
    id_slot: c_int = -1,

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    index: usize = 0, // to be used for batched tasks

    arena: std.heap.ArenaAllocator,
    kind: Kind,

    pub const Kind = union(enum) {
        cmpl_final: CmplFinal,
        cmpl_partial: CmplPartial,
        embd: Embd,
        realtime: Realtime,
        tts: Tts,
        rerank: Rerank,
        decision: Decision,
        err: ErrorResult,
        metrics: MetricsResult,
        slots: Slots,
        slot_save_load: SlotSaveLoad,
        slot_erase: SlotErase,
        control: Control,
        get_lora: GetLora,
        apply_lora: void,
        router: Router,
    };

    /// `new server_task_result_X()`: a result of the given kind, its members
    /// at their defaults; set them through `kind` and allocate with `alloc`.
    pub fn create(gpa: Allocator, comptime tag: std.meta.Tag(Kind)) Error!*Result {
        const r = try gpa.create(Result);
        const T = @FieldType(Kind, @tagName(tag));
        const v: T = if (T == void) {} else .{};
        r.* = .{ .arena = .init(gpa), .kind = @unionInit(Kind, @tagName(tag), v) };
        return r;
    }

    /// The result's arena: everything set into it should be allocated here.
    pub fn alloc(self: *Result) Allocator {
        return self.arena.allocator();
    }

    /// The `std::unique_ptr`'s delete.
    pub fn destroy(self: *Result, gpa: Allocator) void {
        self.arena.deinit();
        gpa.destroy(self);
    }

    /// Ports `is_error` (server-task.h:364 @8642541b2), a virtual member of `server_task_result`.
    pub fn isError(self: *Result) bool {
        return self.kind == .err;
    }

    /// Ports `is_stop` (server-task.h:368 @8642541b2), a virtual member of
    /// `server_task_result`: in stream mode a final result is a stop, a
    /// partial one is not.
    pub fn isStop(self: *Result) bool {
        return switch (self.kind) {
            .cmpl_partial => false,
            .realtime => false, // the session ends when the client leaves, the task is cancelled then
            .tts => |t| t.final,
            else => true,
        };
    }

    /// Ports `update` (server-task.h:372 @8642541b2), a virtual member of
    /// `server_task_result`: a completion result parses the text it carries
    /// into the task's chat state.
    pub fn update(self: *Result, state: *TaskResultState) Error!void {
        switch (self.kind) {
            .cmpl_final => try self.updateFinal(state),
            .cmpl_partial => try self.updatePartial(state),
            else => {},
        }
    }

    /// Ports `clone` (server-task.h:377 @8642541b2), a virtual member of
    /// `server_task_result`: only router results copy.
    pub fn clone(self: *Result, gpa: Allocator) Error!*Result {
        switch (self.kind) {
            .router => |r| {
                const out = try create(gpa, .router);
                errdefer out.destroy(gpa);
                out.id = self.id;
                out.id_slot = self.id_slot;
                out.index = self.index;
                out.kind.router.data = try r.data.clone(out.alloc());
                return out;
            },
            else => {
                _ = c.ggml_abort("server-task.h", 378, "not implemented for this task type");
                unreachable;
            },
        }
    }

    /// Ports `to_json` (server-task.h:662 @8642541b2), a pure virtual member of
    /// `server_task_result`, through each subclass's.
    ///
    /// Return: owned by `a`.
    pub fn toJson(self: *Result, a: Allocator) Error!Value {
        return switch (self.kind) {
            .cmpl_final => self.finalToJson(a),
            .cmpl_partial => self.partialToJson(a),
            .embd => self.embdToJson(a),
            .realtime => self.realtimeToJson(a),
            .tts => self.ttsToJson(a),
            .rerank => self.rerankToJson(a),
            .decision => self.decisionToJson(a),
            .err => self.errorToJson(a),
            .metrics => .null, // not used, /metrics renders prometheus text via to_metrics()
            .slots => |*s| s.slots_data.clone(a),
            .slot_save_load => self.slotSaveLoadToJson(a),
            .slot_erase => self.slotEraseToJson(a),
            .control => self.controlToJson(a),
            .get_lora => self.getLoraToJson(a),
            .apply_lora => applyLoraToJson(a),
            .router => |*r| r.data.clone(a),
        };
    }

    // -------------------------------------------------------------------------
    // server_task_result_cmpl_final

    /// Ports `server_task_result_cmpl_final`'s `update` (server-task.h:451 @8642541b2).
    fn updateFinal(self: *Result, state: *TaskResultState) Error!void {
        const ar = self.alloc();
        const f = &self.kind.cmpl_final;
        f.is_updated = true;
        const msg = try state.updateChatMsg(ar, &f.content, false, &f.oaicompat_msg_diffs, false);
        f.oaicompat_msg = try msg.clone(ar);

        f.oai_resp_id = try ar.dupe(u8, state.oai_resp_id);
        f.oai_resp_reasoning_id = try ar.dupe(u8, state.oai_resp_reasoning_id);
        f.oai_resp_message_id = try ar.dupe(u8, state.oai_resp_message_id);
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json` (server-task.cpp:312 @8642541b2).
    fn finalToJson(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        common.ggmlAssert(f.is_updated, "server-task.cpp", 313, "is_updated && \"update() must be called before to_json()\"");
        return switch (f.res_type) {
            .none => self.finalNonOaicompat(a),
            .oai_cmpl => self.finalOaicompat(a),
            .oai_chat => if (f.stream) self.finalOaicompatChatStream(a) else self.finalOaicompatChat(a),
            .oai_resp => if (f.stream) self.finalOaicompatRespStream(a) else self.finalOaicompatResp(a),
            .oai_asr => self.finalOaicompatAsr(a),
            .anthropic => if (f.stream) self.finalAnthropicStream(a) else self.finalAnthropic(a),
            else => {
                common.ggmlAssert(false, "server-task.cpp", 328, "false && \"Invalid task_response_type\"");
                unreachable;
            },
        };
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_non_oaicompat` (server-task.cpp:332 @8642541b2).
    fn finalNonOaicompat(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        var res = Obj.init(a);
        errdefer res.deinit();
        try res.put("index", self.index);
        try res.put("content", f.content.text.items);
        try res.put("tokens", f.tokens);
        try res.put("id_slot", self.id_slot);
        try res.put("stop", true);
        try res.put("model", f.oaicompat_model);
        try res.put("tokens_predicted", f.n_decoded);
        try res.put("tokens_evaluated", f.n_prompt_tokens);
        try res.put("generation_settings", try f.generation_params.toJson(a, false));
        try res.put("prompt", f.prompt);
        try res.put("has_new_line", f.has_new_line);
        try res.put("truncated", f.truncated);
        try res.put("stop_type", task.stopTypeToStr(f.stop));
        try res.put("stopping_word", f.stopping_word);
        try res.put("tokens_cached", f.n_tokens_cached);
        try res.put("timings", try f.stats.toJson(a));
        if (!f.stream and f.probs_output.len != 0) {
            try res.put("completion_probabilities", try CompletionTokenOutput.probsVectorToJson(a, f.probs_output, f.post_sampling_probs));
        }
        if (f.response_fields.len == 0) return res.done();
        var full = res.done();
        defer full.deinit(a);
        return common.jsonGetNestedValues(a, f.response_fields, &full);
    }

    /// Ports `server_task_result_cmpl_final`'s `usage_json_oaicompat` (server-task.cpp:357 @8642541b2).
    fn finalUsageJsonOaicompat(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("completion_tokens", f.n_decoded);
        try o.put("prompt_tokens", f.n_prompt_tokens);
        try o.put("total_tokens", f.n_decoded +% f.n_prompt_tokens);
        var details = Obj.init(a);
        errdefer details.deinit();
        try details.put("cached_tokens", f.n_prompt_tokens_cache);
        try o.put("prompt_tokens_details", details.done());
        return o.done();
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_oaicompat` (server-task.cpp:366 @8642541b2).
    fn finalOaicompat(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        const t = time(null);
        var logprobs: Value = .null; // OAI default to null
        errdefer logprobs.deinit(a);
        if (!f.stream and f.probs_output.len > 0) {
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("content", try CompletionTokenOutput.probsVectorToJson(a, f.probs_output, f.post_sampling_probs));
            logprobs = o.done();
        }
        var finish_reason: []const u8 = "length";
        if (f.stop == .word or f.stop == .eos) {
            finish_reason = "stop";
        }
        var choice = Obj.init(a);
        errdefer choice.deinit();
        try choice.put("text", f.content.text.items);
        try choice.put("index", self.index);
        try choice.put("logprobs", logprobs);
        logprobs = .null;
        try choice.put("finish_reason", finish_reason);
        var choices = Value.emptyArray();
        errdefer choices.deinit(a);
        try choices.pushBack(a, choice.done());

        var res = Obj.init(a);
        errdefer res.deinit();
        try res.put("choices", choices);
        choices = .null;
        try res.put("created", t);
        try res.put("model", f.oaicompat_model);
        try res.put("system_fingerprint", build_info);
        try res.put("object", "text_completion");
        try res.put("usage", try self.finalUsageJsonOaicompat(a));
        try res.put("id", f.oaicompat_cmpl_id);

        // extra fields for debugging purposes
        if (f.verbose) {
            try res.put("__verbose", try self.finalNonOaicompat(a));
        }
        if (f.stats.isSet()) {
            try res.put("timings", try f.stats.toJson(a));
        }

        return res.done();
    }

    /// The message a non-streamed answer reports: the parsed one, or the raw
    /// text as an assistant message when nothing was parsed.
    fn finalMsg(f: *const CmplFinal) chat.Msg {
        if (!f.oaicompat_msg.empty()) {
            return f.oaicompat_msg;
        }
        return .{ .role = "assistant", .content = f.content.text.items };
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_oaicompat_chat` (server-task.cpp:406 @8642541b2).
    fn finalOaicompatChat(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        var finish_reason: []const u8 = "length";
        const msg = finalMsg(f);
        if (f.stop == .word or f.stop == .eos) {
            finish_reason = if (msg.tool_calls.items.len == 0) "stop" else "tool_calls";
        }

        var choice = Obj.init(a);
        errdefer choice.deinit();
        try choice.put("finish_reason", finish_reason);
        try choice.put("index", self.index);
        try choice.put("message", try msg.toJsonOaicompat(a, false));

        if (!f.stream and f.probs_output.len > 0) {
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("content", try CompletionTokenOutput.probsVectorToJson(a, f.probs_output, f.post_sampling_probs));
            try choice.put("logprobs", o.done());
        }

        const t = time(null);

        var choices = Value.emptyArray();
        errdefer choices.deinit(a);
        try choices.pushBack(a, choice.done());
        var res = Obj.init(a);
        errdefer res.deinit();
        try res.put("choices", choices);
        choices = .null;
        try res.put("created", t);
        try res.put("model", f.oaicompat_model);
        try res.put("system_fingerprint", build_info);
        try res.put("object", "chat.completion");
        try res.put("usage", try self.finalUsageJsonOaicompat(a));
        try res.put("id", f.oaicompat_cmpl_id);

        // extra fields for debugging purposes
        if (f.verbose) {
            try res.put("__verbose", try self.finalNonOaicompat(a));
        }
        if (f.stats.isSet()) {
            try res.put("timings", try f.stats.toJson(a));
        }

        return res.done();
    }

    /// One `chat.completion.chunk`, as both stream formatters build it.
    fn chunk(self: *Result, a: Allocator, t: i64, finish_reason: Value, delta: Value, cmpl_id: []const u8, model: []const u8) Error!Value {
        var fr = finish_reason;
        var d = delta;
        errdefer fr.deinit(a);
        errdefer d.deinit(a);
        var choice = Obj.init(a);
        errdefer choice.deinit();
        try choice.put("finish_reason", fr);
        fr = .null;
        try choice.put("index", self.index);
        try choice.put("delta", d);
        d = .null;
        var choices = Value.emptyArray();
        errdefer choices.deinit(a);
        try choices.pushBack(a, choice.done());
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("choices", choices);
        choices = .null;
        try o.put("created", t);
        try o.put("id", cmpl_id);
        try o.put("model", model);
        try o.put("system_fingerprint", build_info);
        try o.put("object", "chat.completion.chunk");
        return o.done();
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_oaicompat_chat_stream` (server-task.cpp:454 @8642541b2).
    fn finalOaicompatChatStream(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        const t = time(null);
        var finish_reason: []const u8 = "length";
        if (f.stop == .word or f.stop == .eos) {
            finish_reason = if (f.oaicompat_msg.tool_calls.items.len == 0) "stop" else "tool_calls";
        }

        var deltas = Value.emptyArray();
        errdefer deltas.deinit(a);
        for (f.oaicompat_msg_diffs.items) |*diff| {
            try deltas.pushBack(a, try self.chunk(a, t, .null, try chatMsgDiffToJsonOaicompat(a, diff), f.oaicompat_cmpl_id, f.oaicompat_model));
        }

        try deltas.pushBack(a, try self.chunk(a, t, try Value.initString(a, finish_reason), Value.emptyObject(), f.oaicompat_cmpl_id, f.oaicompat_model));

        if (f.include_usage) {
            // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
            // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("choices", Value.emptyArray());
            try o.put("created", t);
            try o.put("id", f.oaicompat_cmpl_id);
            try o.put("model", f.oaicompat_model);
            try o.put("system_fingerprint", build_info);
            try o.put("object", "chat.completion.chunk");
            try o.put("usage", try self.finalUsageJsonOaicompat(a));
            try deltas.pushBack(a, o.done());
        }

        if (f.stats.isSet()) {
            try (try deltas.back()).set(a, "timings", try f.stats.toJson(a));
        }

        // extra fields for debugging purposes
        if (f.verbose and !deltas.empty()) {
            try (try deltas.front()).set(a, "__verbose", try self.finalNonOaicompat(a));
        }

        return deltas;
    }

    /// `{"type": "output_text", "annotations": [], "logprobs": [], "text": text}`.
    fn outputText(a: Allocator, text: []const u8) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("type", "output_text");
        try o.put("annotations", Value.emptyArray());
        try o.put("logprobs", Value.emptyArray());
        try o.put("text", text);
        return o.done();
    }

    /// `{"text": text, "type": "reasoning_text"}` in an array.
    fn reasoningContent(a: Allocator, text: []const u8) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("text", text);
        try o.put("type", "reasoning_text");
        var arr = Value.emptyArray();
        errdefer arr.deinit(a);
        try arr.pushBack(a, o.done());
        return arr;
    }

    fn functionCallItem(a: Allocator, tool_call: chat.ToolCall) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        const fc_id = try std.mem.concat(a, u8, &.{ "fc_", tool_call.id });
        defer a.free(fc_id);
        const call_id = try std.mem.concat(a, u8, &.{ "call_", tool_call.id });
        defer a.free(call_id);
        try o.put("id", fc_id);
        try o.put("type", "function_call");
        try o.put("status", "completed");
        try o.put("arguments", tool_call.arguments);
        try o.put("call_id", call_id);
        try o.put("name", tool_call.name);
        return o.done();
    }

    fn respUsage(f: *const CmplFinal, a: Allocator) Error!Value {
        var u = Obj.init(a);
        errdefer u.deinit();
        try u.put("input_tokens", f.n_prompt_tokens);
        try u.put("output_tokens", f.n_decoded);
        try u.put("total_tokens", f.n_decoded +% f.n_prompt_tokens);
        var details = Obj.init(a);
        errdefer details.deinit();
        try details.put("cached_tokens", f.n_prompt_tokens_cache);
        try u.put("input_tokens_details", details.done());
        return u.done();
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_oaicompat_resp` (server-task.cpp:520 @8642541b2).
    fn finalOaicompatResp(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        const msg = finalMsg(f);

        var output = Value.emptyArray();
        errdefer output.deinit(a);

        if (msg.reasoning_content.len != 0) {
            const rid = try common.randomString(a);
            defer a.free(rid);
            const id = try std.mem.concat(a, u8, &.{ "rs_", rid });
            defer a.free(id);
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("id", id);
            try o.put("summary", Value.emptyArray());
            try o.put("type", "reasoning");
            try o.put("content", try reasoningContent(a, msg.reasoning_content));
            try o.put("encrypted_content", "");
            try o.put("status", "completed");
            try output.pushBack(a, o.done());
        }

        if (msg.content.len != 0) {
            var content = Value.emptyArray();
            errdefer content.deinit(a);
            try content.pushBack(a, try outputText(a, msg.content));
            const rid = try common.randomString(a);
            defer a.free(rid);
            const id = try std.mem.concat(a, u8, &.{ "msg_", rid });
            defer a.free(id);
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("content", content);
            content = .null;
            try o.put("id", id);
            try o.put("role", msg.role);
            try o.put("status", "completed");
            try o.put("type", "message");
            try output.pushBack(a, o.done());
        }

        for (f.oaicompat_msg.tool_calls.items) |tool_call| {
            try output.pushBack(a, try functionCallItem(a, tool_call));
        }

        const t = time(null);
        var res = Obj.init(a);
        errdefer res.deinit();
        try res.put("completed_at", t);
        try res.put("created_at", t);
        try res.put("id", f.oai_resp_id);
        try res.put("model", f.oaicompat_model);
        try res.put("object", "response");
        try res.put("output", output);
        output = .null;
        try res.put("status", "completed");
        try res.put("usage", try respUsage(f, a));

        return res.done();
    }

    fn outputItemDone(a: Allocator, item: *const Value, output_index: usize) Error!Value {
        var d = Obj.init(a);
        errdefer d.deinit();
        try d.put("type", "response.output_item.done");
        try d.put("output_index", output_index);
        try d.put("item", try item.clone(a));
        return event(a, "response.output_item.done", d.done());
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_oaicompat_resp_stream` (server-task.cpp:591 @8642541b2).
    fn finalOaicompatRespStream(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        var server_sent_events = Value.emptyArray();
        errdefer server_sent_events.deinit(a);
        var output = Value.emptyArray();
        errdefer output.deinit(a);
        var output_index: usize = 0;

        if (f.oaicompat_msg.reasoning_content.len != 0) {
            const item_output_index = output_index;
            output_index += 1;
            var item = Obj.init(a);
            errdefer item.deinit();
            try item.put("id", f.oai_resp_reasoning_id);
            try item.put("summary", Value.emptyArray());
            try item.put("type", "reasoning");
            try item.put("content", try reasoningContent(a, f.oaicompat_msg.reasoning_content));
            try item.put("encrypted_content", "");
            var output_item = item.done();
            defer output_item.deinit(a);

            try server_sent_events.pushBack(a, try outputItemDone(a, &output_item, item_output_index));
            try output.pushBack(a, try output_item.clone(a));
        }

        if (f.oaicompat_msg.content.len != 0) {
            const item_output_index = output_index;
            output_index += 1;
            {
                var d = Obj.init(a);
                errdefer d.deinit();
                try d.put("type", "response.output_text.done");
                try d.put("item_id", f.oai_resp_message_id);
                try d.put("output_index", item_output_index);
                try d.put("content_index", 0);
                try d.put("text", f.oaicompat_msg.content);
                try server_sent_events.pushBack(a, try event(a, "response.output_text.done", d.done()));
            }

            var content_part = try outputText(a, f.oaicompat_msg.content);
            defer content_part.deinit(a);

            {
                var d = Obj.init(a);
                errdefer d.deinit();
                try d.put("type", "response.content_part.done");
                try d.put("item_id", f.oai_resp_message_id);
                try d.put("output_index", item_output_index);
                try d.put("content_index", 0);
                try d.put("part", try content_part.clone(a));
                try server_sent_events.pushBack(a, try event(a, "response.content_part.done", d.done()));
            }
            var parts = Value.emptyArray();
            errdefer parts.deinit(a);
            try parts.pushBack(a, try content_part.clone(a));
            var item = Obj.init(a);
            errdefer item.deinit();
            try item.put("type", "message");
            try item.put("status", "completed");
            try item.put("id", f.oai_resp_message_id);
            try item.put("content", parts);
            parts = .null;
            try item.put("role", "assistant");
            var output_item = item.done();
            defer output_item.deinit(a);

            try server_sent_events.pushBack(a, try outputItemDone(a, &output_item, item_output_index));
            try output.pushBack(a, try output_item.clone(a));
        }

        for (f.oaicompat_msg.tool_calls.items) |tool_call| {
            const item_output_index = output_index;
            output_index += 1;
            var output_item = try functionCallItem(a, tool_call);
            defer output_item.deinit(a);
            try server_sent_events.pushBack(a, try outputItemDone(a, &output_item, item_output_index));
            try output.pushBack(a, try output_item.clone(a));
        }

        const t = time(null);
        var response = Obj.init(a);
        errdefer response.deinit();
        try response.put("id", f.oai_resp_id);
        try response.put("object", "response");
        try response.put("created_at", t);
        try response.put("status", "completed");
        try response.put("model", f.oaicompat_model);
        try response.put("output", output);
        output = .null;
        try response.put("usage", try respUsage(f, a));
        var d = Obj.init(a);
        errdefer d.deinit();
        try d.put("type", "response.completed");
        try d.put("response", response.done());
        try server_sent_events.pushBack(a, try event(a, "response.completed", d.done()));

        if (f.stats.isSet()) {
            try (try (try server_sent_events.back()).at("data")).set(a, "timings", try f.stats.toJson(a));
        }

        return server_sent_events;
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_oaicompat_asr` (server-task.cpp:719 @8642541b2).
    fn finalOaicompatAsr(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        var usage = Obj.init(a);
        errdefer usage.deinit();
        try usage.put("type", "tokens");
        try usage.put("input_tokens", f.n_prompt_tokens);
        try usage.put("output_tokens", f.n_decoded);
        try usage.put("total_tokens", f.n_decoded +% f.n_prompt_tokens);
        var details = Obj.init(a);
        errdefer details.deinit();
        try details.put("cached_tokens", f.n_prompt_tokens_cache);
        try usage.put("input_tokens_details", details.done());
        var ev = Obj.init(a);
        errdefer ev.deinit();
        try ev.put("type", "transcript.text.done");
        try ev.put("text", f.oaicompat_msg.content);
        try ev.put("usage", usage.done());
        return ev.done();
    }

    fn stopSequence(a: Allocator, stopping_word: []const u8) Error!Value {
        return if (stopping_word.len == 0) .null else Value.initString(a, stopping_word);
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_anthropic` (server-task.cpp:734 @8642541b2).
    fn finalAnthropic(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        var stop_reason: []const u8 = "max_tokens";
        if (f.stop == .word or f.stop == .eos) {
            stop_reason = if (f.oaicompat_msg.tool_calls.items.len == 0) "end_turn" else "tool_use";
        }

        var content_blocks = Value.emptyArray();
        errdefer content_blocks.deinit(a);

        const msg = finalMsg(f);

        // thinking block comes first (Anthropic extended thinking format)
        if (msg.reasoning_content.len != 0) {
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("type", "thinking");
            try o.put("thinking", msg.reasoning_content);
            try o.put("signature", ""); // empty signature for local models (no cryptographic verification)
            try content_blocks.pushBack(a, o.done());
        }

        if (msg.content.len != 0) {
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("type", "text");
            try o.put("text", msg.content);
            try content_blocks.pushBack(a, o.done());
        }

        for (msg.tool_calls.items) |tool_call| {
            var tool_use_block = Obj.init(a);
            errdefer tool_use_block.deinit();
            try tool_use_block.put("type", "tool_use");
            try tool_use_block.put("id", tool_call.id);
            try tool_use_block.put("name", tool_call.name);

            const input = json.parse(a, tool_call.arguments) catch |e| switch (e) {
                error.JsonError => Value.emptyObject(),
                error.OutOfMemory => return error.OutOfMemory,
            };
            try tool_use_block.put("input", input);

            try content_blocks.pushBack(a, tool_use_block.done());
        }

        var usage = Obj.init(a);
        errdefer usage.deinit();
        try usage.put("cache_read_input_tokens", f.n_prompt_tokens_cache);
        try usage.put("input_tokens", f.n_prompt_tokens -% f.n_prompt_tokens_cache);
        try usage.put("output_tokens", f.n_decoded);

        var res = Obj.init(a);
        errdefer res.deinit();
        try res.put("id", f.oaicompat_cmpl_id);
        try res.put("type", "message");
        try res.put("role", "assistant");
        try res.put("content", content_blocks);
        content_blocks = .null;
        try res.put("model", f.oaicompat_model);
        try res.put("stop_reason", stop_reason);
        try res.put("stop_sequence", try stopSequence(a, f.stopping_word));
        try res.put("usage", usage.done());

        return res.done();
    }

    /// `content_block_start`.
    fn blockStart(a: Allocator, index: usize, block: Value) Error!Value {
        var d = Obj.init(a);
        errdefer d.deinit();
        try d.put("type", "content_block_start");
        try d.put("index", index);
        try d.put("content_block", block);
        return event(a, "content_block_start", d.done());
    }

    /// `content_block_delta`.
    fn blockDelta(a: Allocator, index: usize, delta: Value) Error!Value {
        var d = Obj.init(a);
        errdefer d.deinit();
        try d.put("type", "content_block_delta");
        try d.put("index", index);
        try d.put("delta", delta);
        return event(a, "content_block_delta", d.done());
    }

    /// `content_block_stop`.
    fn blockStop(a: Allocator, index: usize) Error!Value {
        var d = Obj.init(a);
        errdefer d.deinit();
        try d.put("type", "content_block_stop");
        try d.put("index", index);
        return event(a, "content_block_stop", d.done());
    }

    fn pair(a: Allocator, k1: []const u8, v1: []const u8, k2: []const u8, v2: []const u8) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put(k1, v1);
        try o.put(k2, v2);
        return o.done();
    }

    fn toolUseBlock(a: Allocator, id: []const u8, name: []const u8) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("type", "tool_use");
        try o.put("id", id);
        try o.put("name", name);
        return o.done();
    }

    /// Ports `server_task_result_cmpl_final`'s `to_json_anthropic_stream` (server-task.cpp:800 @8642541b2).
    fn finalAnthropicStream(self: *Result, a: Allocator) Error!Value {
        const f = &self.kind.cmpl_final;
        var events = Value.emptyArray();
        errdefer events.deinit(a);

        var stop_reason: []const u8 = "max_tokens";
        if (f.stop == .word or f.stop == .eos) {
            stop_reason = if (f.oaicompat_msg.tool_calls.items.len == 0) "end_turn" else "tool_use";
        }

        const has_thinking = f.oaicompat_msg.reasoning_content.len != 0;
        const has_text = f.oaicompat_msg.content.len != 0;
        const num_tool_calls = f.oaicompat_msg.tool_calls.items.len;

        // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
        const thinking_block_index: usize = 0;
        const text_block_index: usize = if (has_thinking) 1 else 0;

        var thinking_block_started = false;
        var text_block_started = false;
        var tool_calls_started: std.AutoHashMapUnmanaged(usize, void) = .empty;
        defer tool_calls_started.deinit(a);

        for (f.oaicompat_msg_diffs.items) |diff| {
            // handle thinking/reasoning content
            if (diff.reasoning_content_delta.len != 0) {
                if (!thinking_block_started) {
                    try events.pushBack(a, try blockStart(a, thinking_block_index, try pair(a, "type", "thinking", "thinking", "")));
                    thinking_block_started = true;
                }

                try events.pushBack(a, try blockDelta(a, thinking_block_index, try pair(a, "type", "thinking_delta", "thinking", diff.reasoning_content_delta)));
            }

            // handle regular text content
            if (diff.content_delta.len != 0) {
                if (!text_block_started) {
                    try events.pushBack(a, try blockStart(a, text_block_index, try pair(a, "type", "text", "text", "")));
                    text_block_started = true;
                }

                try events.pushBack(a, try blockDelta(a, text_block_index, try pair(a, "type", "text_delta", "text", diff.content_delta)));
            }

            // handle tool calls
            if (diff.tool_call_index != chat.msg.npos) {
                const content_block_index: usize = @as(usize, @intFromBool(has_thinking)) + @as(usize, @intFromBool(has_text)) + diff.tool_call_index;

                if (!tool_calls_started.contains(diff.tool_call_index)) {
                    const full_tool_call = f.oaicompat_msg.tool_calls.items[diff.tool_call_index];

                    try events.pushBack(a, try blockStart(a, content_block_index, try toolUseBlock(a, full_tool_call.id, full_tool_call.name)));
                    try tool_calls_started.put(a, diff.tool_call_index, {});
                }

                if (diff.tool_call_delta.arguments.len != 0) {
                    try events.pushBack(a, try blockDelta(a, content_block_index, try pair(a, "type", "input_json_delta", "partial_json", diff.tool_call_delta.arguments)));
                }
            }
        }

        // close content blocks in order
        if (has_thinking) {
            // Anthropic API requires a signature_delta before closing thinking blocks
            // We use an empty signature since we can't generate a cryptographic signature for local models
            try events.pushBack(a, try blockDelta(a, thinking_block_index, try pair(a, "type", "signature_delta", "signature", "")));
            try events.pushBack(a, try blockStop(a, thinking_block_index));
        }

        if (has_text) {
            try events.pushBack(a, try blockStop(a, text_block_index));
        }

        for (0..num_tool_calls) |i| {
            const content_block_index: usize = @as(usize, @intFromBool(has_thinking)) + @as(usize, @intFromBool(has_text)) + i;
            try events.pushBack(a, try blockStop(a, content_block_index));
        }

        {
            var delta = Obj.init(a);
            errdefer delta.deinit();
            try delta.put("stop_reason", stop_reason);
            try delta.put("stop_sequence", try stopSequence(a, f.stopping_word));
            var usage = Obj.init(a);
            errdefer usage.deinit();
            try usage.put("output_tokens", f.n_decoded);
            var d = Obj.init(a);
            errdefer d.deinit();
            try d.put("type", "message_delta");
            try d.put("delta", delta.done());
            try d.put("usage", usage.done());
            try events.pushBack(a, try event(a, "message_delta", d.done()));
        }

        {
            var d = Obj.init(a);
            errdefer d.deinit();
            try d.put("type", "message_stop");
            try events.pushBack(a, try event(a, "message_stop", d.done()));
        }

        return events;
    }

    // -------------------------------------------------------------------------
    // server_task_result_cmpl_partial

    /// Ports `server_task_result_cmpl_partial`'s `update` (server-task.cpp:991 @8642541b2): the
    /// state as it was before this chunk is copied in, then advanced for
    /// the next one.
    fn updatePartial(self: *Result, state: *TaskResultState) Error!void {
        const ar = self.alloc();
        const p = &self.kind.cmpl_partial;
        p.is_updated = true;
        if (p.is_begin) {
            return; // begin marker only flushes headers, skip parsing
        }
        _ = try state.updateChatMsg(ar, &p.content, true, &p.oaicompat_msg_diffs, false);

        // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
        p.thinking_block_started = state.thinking_block_started;
        p.text_block_started = state.text_block_started;

        p.oai_resp_created = state.oai_resp_created;
        p.oai_resp_id = try ar.dupe(u8, state.oai_resp_id);
        p.oai_resp_reasoning_id = try ar.dupe(u8, state.oai_resp_reasoning_id);
        p.oai_resp_message_id = try ar.dupe(u8, state.oai_resp_message_id);
        p.oai_resp_fc_id = try ar.dupe(u8, state.oai_resp_fc_id);

        // track if the accumulated message has any reasoning content
        p.anthropic_has_reasoning = state.chat_msg.reasoning_content.len != 0;

        if (p.res_type == .oai_resp and !state.oai_resp_created and (p.is_progress or p.n_decoded == 1)) {
            state.oai_resp_created = true;
        }

        // Pre-compute state updates based on diffs (for next chunk)
        for (p.oaicompat_msg_diffs.items) |diff| {
            if (diff.reasoning_content_delta.len != 0 and !state.thinking_block_started) {
                state.thinking_block_started = true;
            }
            if (diff.content_delta.len != 0 and !state.text_block_started) {
                state.text_block_started = true;
            }
            if (diff.tool_call_delta.name.len != 0) {
                try state.setFcId(diff.tool_call_delta.id);
            }
        }
    }

    /// Ports `server_task_result_cmpl_partial`'s `to_json` (server-task.cpp:1029 @8642541b2).
    fn partialToJson(self: *Result, a: Allocator) Error!Value {
        const p = &self.kind.cmpl_partial;
        common.ggmlAssert(p.is_updated, "server-task.cpp", 1030, "is_updated && \"update() must be called before to_json()\"");
        if (p.is_begin) {
            return .null; // simply signal to HTTP handler to send the headers and status code
        }
        return switch (p.res_type) {
            .none => self.partialNonOaicompat(a),
            .oai_cmpl => self.partialOaicompat(a),
            .oai_chat => self.partialOaicompatChat(a),
            .oai_resp => self.partialOaicompatResp(a),
            .oai_asr => self.partialOaicompatAsr(a),
            .anthropic => self.partialAnthropic(a),
            else => {
                common.ggmlAssert(false, "server-task.cpp", 1048, "false && \"Invalid task_response_type\"");
                unreachable;
            },
        };
    }

    /// Ports `server_task_result_cmpl_partial`'s `to_json_non_oaicompat` (server-task.cpp:1052 @8642541b2).
    fn partialNonOaicompat(self: *Result, a: Allocator) Error!Value {
        const p = &self.kind.cmpl_partial;
        // non-OAI-compat JSON
        var res = Obj.init(a);
        errdefer res.deinit();
        try res.put("index", self.index);
        try res.put("content", p.content.text.items);
        try res.put("tokens", p.tokens);
        try res.put("stop", false);
        try res.put("id_slot", self.id_slot);
        try res.put("tokens_predicted", p.n_decoded);
        try res.put("tokens_evaluated", p.n_prompt_tokens);
        // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
        if (p.stats.isSet()) {
            try res.put("timings", try p.stats.toJson(a));
        }
        if (p.is_progress) {
            try res.put("prompt_progress", try p.progress.toJson(a));
        }
        if (p.prob_output.probs.len != 0) {
            try res.put("completion_probabilities", try CompletionTokenOutput.probsVectorToJson(a, (&p.prob_output)[0..1], p.post_sampling_probs));
        }
        return res.done();
    }

    /// Ports `server_task_result_cmpl_partial`'s `to_json_oaicompat` (server-task.cpp:1076 @8642541b2).
    fn partialOaicompat(self: *Result, a: Allocator) Error!Value {
        const p = &self.kind.cmpl_partial;
        const t = time(null);
        var logprobs: Value = .null; // OAI default to null
        errdefer logprobs.deinit(a);
        if (p.prob_output.probs.len > 0) {
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("content", try CompletionTokenOutput.probsVectorToJson(a, (&p.prob_output)[0..1], p.post_sampling_probs));
            logprobs = o.done();
        }
        var choice = Obj.init(a);
        errdefer choice.deinit();
        try choice.put("text", p.content.text.items);
        try choice.put("index", self.index);
        try choice.put("logprobs", logprobs);
        logprobs = .null;
        try choice.put("finish_reason", @as(Value, .null));
        var choices = Value.emptyArray();
        errdefer choices.deinit(a);
        try choices.pushBack(a, choice.done());
        var res = Obj.init(a);
        errdefer res.deinit();
        try res.put("choices", choices);
        choices = .null;
        try res.put("created", t);
        try res.put("model", p.oaicompat_model);
        try res.put("system_fingerprint", build_info);
        try res.put("object", "text_completion");
        try res.put("id", p.oaicompat_cmpl_id);

        // extra fields for debugging purposes
        if (p.verbose) {
            try res.put("__verbose", try self.partialNonOaicompat(a));
        }
        if (p.stats.isSet()) {
            try res.put("timings", try p.stats.toJson(a));
        }
        if (p.is_progress) {
            try res.put("prompt_progress", try p.progress.toJson(a));
        }

        return res.done();
    }

    /// Ports `server_task_result_cmpl_partial`'s `to_json_oaicompat_chat` (server-task.cpp:1114 @8642541b2).
    fn partialOaicompatChat(self: *Result, a: Allocator) Error!Value {
        const p = &self.kind.cmpl_partial;
        const first = p.n_decoded == 1;
        const t = time(null);

        var deltas = Value.emptyArray();
        errdefer deltas.deinit(a);
        // We have to send an initial update to conform to openai behavior
        if (first or p.is_progress) {
            var d = Obj.init(a);
            errdefer d.deinit();
            try d.put("role", "assistant");
            try d.put("content", @as(Value, .null));
            try deltas.pushBack(a, try self.chunk(a, t, .null, d.done(), p.oaicompat_cmpl_id, p.oaicompat_model));
        }

        for (p.oaicompat_msg_diffs.items) |*diff| {
            try deltas.pushBack(a, try self.chunk(a, t, .null, try chatMsgDiffToJsonOaicompat(a, diff), p.oaicompat_cmpl_id, p.oaicompat_model));
        }

        if (!deltas.empty()) {
            const last_json = try deltas.back();
            common.ggmlAssert((try last_json.at("choices")).size() >= 1, "server-task.cpp", 1150, "last_json.at(\"choices\").size() >= 1");

            if (p.prob_output.probs.len > 0) {
                var o = Obj.init(a);
                errdefer o.deinit();
                try o.put("content", try CompletionTokenOutput.probsVectorToJson(a, (&p.prob_output)[0..1], p.post_sampling_probs));
                try (try (try last_json.at("choices")).atIndex(0)).set(a, "logprobs", o.done());
            }

            if (p.stats.isSet()) {
                try last_json.set(a, "timings", try p.stats.toJson(a));
            }
            if (p.is_progress) {
                try last_json.set(a, "prompt_progress", try p.progress.toJson(a));
            }
        }

        return deltas;
    }

    fn inProgress(a: Allocator, name: []const u8, id: []const u8) Error!Value {
        var r = Obj.init(a);
        errdefer r.deinit();
        try r.put("id", id);
        try r.put("object", "response");
        try r.put("status", "in_progress");
        var d = Obj.init(a);
        errdefer d.deinit();
        try d.put("type", name);
        try d.put("response", r.done());
        return event(a, name, d.done());
    }

    /// The output index of a tool call in a stream: after the reasoning and
    /// the text. `size_t` arithmetic, as the C++'s.
    fn toolOutputIndex(p: *const CmplPartial, diff: *const chat.MsgDiff) usize {
        return @as(usize, if (p.anthropic_has_reasoning) 1 else 0) +% @as(usize, if (p.text_block_started) 1 else 0) +% diff.tool_call_index;
    }

    /// Ports `server_task_result_cmpl_partial`'s `to_json_oaicompat_resp` (server-task.cpp:1169 @8642541b2).
    fn partialOaicompatResp(self: *Result, a: Allocator) Error!Value {
        const p = &self.kind.cmpl_partial;
        var events = Value.emptyArray();
        errdefer events.deinit(a);

        if (!p.oai_resp_created) {
            try events.pushBack(a, try inProgress(a, "response.created", p.oai_resp_id));
            try events.pushBack(a, try inProgress(a, "response.in_progress", p.oai_resp_id));
        } else if (p.is_progress) {
            try events.pushBack(a, try inProgress(a, "response.in_progress", p.oai_resp_id));
        }

        for (p.oaicompat_msg_diffs.items) |diff| {
            if (diff.reasoning_content_delta.len != 0) {
                if (!p.thinking_block_started) {
                    var item = Obj.init(a);
                    errdefer item.deinit();
                    try item.put("id", p.oai_resp_reasoning_id);
                    try item.put("summary", Value.emptyArray());
                    try item.put("type", "reasoning");
                    try item.put("content", Value.emptyArray());
                    try item.put("encrypted_content", "");
                    try item.put("status", "in_progress");
                    var d = Obj.init(a);
                    errdefer d.deinit();
                    try d.put("type", "response.output_item.added");
                    try d.put("output_index", 0);
                    try d.put("item", item.done());
                    try events.pushBack(a, try event(a, "response.output_item.added", d.done()));
                    p.thinking_block_started = true;
                }
                var d = Obj.init(a);
                errdefer d.deinit();
                try d.put("type", "response.reasoning_text.delta");
                try d.put("output_index", 0);
                try d.put("content_index", 0);
                try d.put("delta", diff.reasoning_content_delta);
                try d.put("item_id", p.oai_resp_reasoning_id);
                try events.pushBack(a, try event(a, "response.reasoning_text.delta", d.done()));
            }

            if (diff.content_delta.len != 0) {
                const text_output_index: c_int = if (p.anthropic_has_reasoning) 1 else 0;
                if (!p.text_block_started) {
                    {
                        var item = Obj.init(a);
                        errdefer item.deinit();
                        try item.put("content", Value.emptyArray());
                        try item.put("id", p.oai_resp_message_id);
                        try item.put("role", "assistant");
                        try item.put("status", "in_progress");
                        try item.put("type", "message");
                        var d = Obj.init(a);
                        errdefer d.deinit();
                        try d.put("type", "response.output_item.added");
                        try d.put("output_index", text_output_index);
                        try d.put("item", item.done());
                        try events.pushBack(a, try event(a, "response.output_item.added", d.done()));
                    }
                    {
                        var d = Obj.init(a);
                        errdefer d.deinit();
                        try d.put("type", "response.content_part.added");
                        try d.put("item_id", p.oai_resp_message_id);
                        try d.put("output_index", text_output_index);
                        try d.put("content_index", 0);
                        try d.put("part", try pair(a, "type", "output_text", "text", ""));
                        try events.pushBack(a, try event(a, "response.content_part.added", d.done()));
                    }
                    p.text_block_started = true;
                }
                var d = Obj.init(a);
                errdefer d.deinit();
                try d.put("type", "response.output_text.delta");
                try d.put("item_id", p.oai_resp_message_id);
                try d.put("output_index", text_output_index);
                try d.put("content_index", 0);
                try d.put("delta", diff.content_delta);
                try events.pushBack(a, try event(a, "response.output_text.delta", d.done()));
            }

            if (diff.tool_call_delta.name.len != 0) {
                const fc_id = try std.mem.concat(a, u8, &.{ "fc_", diff.tool_call_delta.id });
                defer a.free(fc_id);
                const call_id = try std.mem.concat(a, u8, &.{ "call_", diff.tool_call_delta.id });
                defer a.free(call_id);
                var item = Obj.init(a);
                errdefer item.deinit();
                try item.put("id", fc_id);
                try item.put("arguments", "");
                try item.put("call_id", call_id);
                try item.put("name", diff.tool_call_delta.name);
                try item.put("type", "function_call");
                try item.put("status", "in_progress");
                var d = Obj.init(a);
                errdefer d.deinit();
                try d.put("type", "response.output_item.added");
                try d.put("output_index", toolOutputIndex(p, &diff));
                try d.put("item", item.done());
                try events.pushBack(a, try event(a, "response.output_item.added", d.done()));
                p.oai_resp_fc_id = try self.alloc().dupe(u8, diff.tool_call_delta.id);
            }

            if (diff.tool_call_delta.arguments.len != 0) {
                const item_id = try std.mem.concat(a, u8, &.{ "fc_", p.oai_resp_fc_id });
                defer a.free(item_id);
                var d = Obj.init(a);
                errdefer d.deinit();
                try d.put("type", "response.function_call_arguments.delta");
                try d.put("delta", diff.tool_call_delta.arguments);
                try d.put("item_id", item_id);
                try d.put("output_index", toolOutputIndex(p, &diff));
                try events.pushBack(a, try event(a, "response.function_call_arguments.delta", d.done()));
            }
        }

        if (!events.empty()) {
            const data = try (try events.back()).at("data");
            if (p.stats.isSet()) {
                try data.set(a, "timings", try p.stats.toJson(a));
            }
            if (p.is_progress) {
                try data.set(a, "prompt_progress", try p.progress.toJson(a));
            }
        }

        return events;
    }

    /// Ports `server_task_result_cmpl_partial`'s `to_json_oaicompat_asr` (server-task.cpp:1329 @8642541b2).
    fn partialOaicompatAsr(self: *Result, a: Allocator) Error!Value {
        const p = &self.kind.cmpl_partial;
        return pair(a, "type", "transcript.text.delta", "delta", p.content.text.items);
    }

    /// Ports `server_task_result_cmpl_partial`'s `to_json_anthropic` (server-task.cpp:1337 @8642541b2).
    fn partialAnthropic(self: *Result, a: Allocator) Error!Value {
        const p = &self.kind.cmpl_partial;
        var events = Value.emptyArray();
        errdefer events.deinit(a);
        const first = (p.n_decoded == 1);
        // use member variables to track block state across streaming calls
        // (anthropic_thinking_block_started, anthropic_text_block_started)

        if (first) {
            var usage = Obj.init(a);
            errdefer usage.deinit();
            try usage.put("cache_read_input_tokens", p.n_prompt_tokens_cache);
            try usage.put("input_tokens", p.n_prompt_tokens -% p.n_prompt_tokens_cache);
            try usage.put("output_tokens", @as(c_int, 0));
            var message = Obj.init(a);
            errdefer message.deinit();
            try message.put("id", p.oaicompat_cmpl_id);
            try message.put("type", "message");
            try message.put("role", "assistant");
            try message.put("content", Value.emptyArray());
            try message.put("model", p.oaicompat_model);
            try message.put("stop_reason", @as(Value, .null));
            try message.put("stop_sequence", @as(Value, .null));
            try message.put("usage", usage.done());
            var d = Obj.init(a);
            errdefer d.deinit();
            try d.put("type", "message_start");
            try d.put("message", message.done());
            try events.pushBack(a, try event(a, "message_start", d.done()));
        }

        // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
        const thinking_block_index: usize = 0;
        // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
        const text_block_index: usize = if (p.anthropic_has_reasoning) 1 else 0;

        // use local copies of streaming state (copied from task_result_state in update())
        // these reflect the state BEFORE this chunk was processed
        var thinking_started = p.thinking_block_started;
        var text_started = p.text_block_started;

        for (p.oaicompat_msg_diffs.items) |diff| {
            // handle thinking/reasoning content
            if (diff.reasoning_content_delta.len != 0) {
                if (!thinking_started) {
                    try events.pushBack(a, try blockStart(a, thinking_block_index, try pair(a, "type", "thinking", "thinking", "")));
                    thinking_started = true;
                }

                try events.pushBack(a, try blockDelta(a, thinking_block_index, try pair(a, "type", "thinking_delta", "thinking", diff.reasoning_content_delta)));
            }

            // handle regular text content
            if (diff.content_delta.len != 0) {
                if (!text_started) {
                    try events.pushBack(a, try blockStart(a, text_block_index, try pair(a, "type", "text", "text", "")));
                    text_started = true;
                }

                try events.pushBack(a, try blockDelta(a, text_block_index, try pair(a, "type", "text_delta", "text", diff.content_delta)));
            }

            // handle tool calls
            if (diff.tool_call_index != chat.msg.npos) {
                // use anthropic_has_reasoning for thinking block count (persists across calls)
                const content_block_index: usize = @as(usize, @intFromBool(p.anthropic_has_reasoning)) + @as(usize, @intFromBool(text_started)) + diff.tool_call_index;

                if (diff.tool_call_delta.name.len != 0) {
                    try events.pushBack(a, try blockStart(a, content_block_index, try toolUseBlock(a, diff.tool_call_delta.id, diff.tool_call_delta.name)));
                }

                if (diff.tool_call_delta.arguments.len != 0) {
                    try events.pushBack(a, try blockDelta(a, content_block_index, try pair(a, "type", "input_json_delta", "partial_json", diff.tool_call_delta.arguments)));
                }
            }
        }

        return events;
    }

    // -------------------------------------------------------------------------
    // the others

    /// Ports `server_task_result_embd`'s `to_json` (server-task.cpp:1479 @8642541b2)
    /// and its two formatters.
    fn embdToJson(self: *Result, a: Allocator) Error!Value {
        const e = &self.kind.embd;
        var o = Obj.init(a);
        errdefer o.deinit();
        if (e.res_type == .oai_embd) {
            // `to_json_oaicompat` (server-task.cpp:1492 @8642541b2)
            try o.put("index", self.index);
            try o.put("embedding", e.embedding[0]);
            try o.put("tokens_evaluated", e.n_tokens);
        } else {
            // `to_json_non_oaicompat` (server-task.cpp:1485 @8642541b2)
            try o.put("index", self.index);
            try o.put("embedding", e.embedding);
        }
        return o.done();
    }

    /// Ports `server_task_result_rerank`'s `to_json` (server-task.cpp:1503 @8642541b2).
    fn rerankToJson(self: *Result, a: Allocator) Error!Value {
        const r = &self.kind.rerank;
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("index", self.index);
        try o.put("score", r.score);
        try o.put("tokens_evaluated", r.n_tokens);
        return o.done();
    }

    /// Ports `server_task_result_realtime`'s `to_json` (server-task.cpp:1514 @8642541b2).
    fn realtimeToJson(self: *Result, a: Allocator) Error!Value {
        const r = &self.kind.realtime;
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("text", r.text);
        try o.put("n_samples", r.audio.len);
        try o.put("response_done", r.response_done);
        try o.put("cancelled", r.cancelled);
        try o.put("n_decoded", r.n_decoded);
        return o.done();
    }

    /// Ports `server_task_result_tts`'s `to_json` (server-task.cpp:1524 @8642541b2).
    fn ttsToJson(self: *Result, a: Allocator) Error!Value {
        const r = &self.kind.tts;
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("sample_rate", r.sample_rate);
        try o.put("n_bytes", r.audio.len);
        try o.put("final", r.final);
        return o.done();
    }

    /// Ports `server_task_result_decision`'s `to_json` (server-task.cpp:1535 @8642541b2).
    fn decisionToJson(self: *Result, a: Allocator) Error!Value {
        const r = &self.kind.decision;
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("index", self.index);
        try o.put("scores", r.scores);
        try o.put("tokens_evaluated", r.n_tokens);
        return o.done();
    }

    /// Ports `server_task_result_error`'s `to_json` (server-task.cpp:1546 @8642541b2).
    fn errorToJson(self: *Result, a: Allocator) Error!Value {
        const e = &self.kind.err;
        var res = try common.formatErrorResponse(a, e.err_msg, e.err_type);
        errdefer res.deinit(a);
        if (e.err_type == .exceed_context_size) {
            try res.set(a, "n_prompt_tokens", try Value.from(a, e.n_prompt_tokens));
            try res.set(a, "n_ctx", try Value.from(a, e.n_ctx));
        }
        return res;
    }

    /// Ports `server_task_result_metrics`'s `to_metrics` (server-task.cpp:1568 @8642541b2): the
    /// Prometheus text for `/metrics`.
    ///
    /// Return: owned by `a`.
    pub fn toMetrics(self: *Result, a: Allocator) Error![]u8 {
        const m = &self.kind.metrics;
        const metrics = &m.metrics;
        const counters = [_]MetricItem{
            .{ .name = "prompt_tokens_total", .description = "Number of prompt tokens processed, excluding cached tokens", .value = @floatFromInt(metrics.prompt.count) },
            .{ .name = "prompt_tokens_cached_total", .description = "Number of prompt tokens reused from the cache", .value = @floatFromInt(metrics.n_prompt_cached) },
            .{ .name = "prompt_seconds_total", .description = "Total time spent processing prompts", .value = @as(f64, @floatFromInt(metrics.prompt.time)) / 1.e6 },
            .{ .name = "tokens_predicted_total", .description = "Number of generation tokens processed", .value = @floatFromInt(metrics.predict.count) },
            .{ .name = "tokens_predicted_seconds_total", .description = "Total time spent generating tokens", .value = @as(f64, @floatFromInt(metrics.predict.time)) / 1.e6 },
            .{ .name = "n_decode_total", .description = "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding", .value = @floatFromInt(metrics.n_decode) },
            .{ .name = "n_tokens_max", .description = "Largest observed sequence length (prompt + generation)", .value = @floatFromInt(metrics.n_tokens_max) },
            .{ .name = "spec_decode_num_draft_tokens_total", .description = "Speculative: Total draft tokens generated", .value = @floatFromInt(metrics.n_draft_tokens) },
            .{ .name = "spec_decode_num_accepted_tokens_total", .description = "Speculative: Total draft tokens accepted by the target model", .value = @floatFromInt(metrics.n_draft_accepted) },
            .{ .name = "spec_decode_num_drafts_total", .description = "Speculative: Total speculative decoding verification steps", .value = @floatFromInt(metrics.n_draft_verif_steps) },
        };

        const gauges = [_]MetricItem{
            .{ .name = "prompt_tokens_seconds", .description = "Average prompt throughput in tokens/s", .value = metrics.prompt_bucket.nPerSecond() },
            .{ .name = "predicted_tokens_seconds", .description = "Average generation throughput in tokens/s", .value = metrics.predict_bucket.nPerSecond() },
            .{ .name = "requests_processing", .description = "Number of requests processing", .value = @floatFromInt(m.n_processing_slots) },
            .{ .name = "requests_deferred", .description = "Number of requests deferred", .value = @floatFromInt(m.n_tasks_deferred) },
            .{ .name = "n_busy_slots_per_decode", .description = "Average number of busy slots per llama_decode() call", .value = @as(f64, @floatFromInt(metrics.n_busy_slots)) / @max(@as(f64, @floatFromInt(metrics.n_decode)), 1.0) },
        };

        var prometheus: std.ArrayList(u8) = .empty;
        errdefer prometheus.deinit(a);

        try addItems(a, &prometheus, "counter", &counters);
        try addItems(a, &prometheus, "gauge", &gauges);

        // labeled counter: one time series per draft position
        if (metrics.n_accepted_per_pos.items.len != 0) {
            try prometheus.appendSlice(a, "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total" ++
                " Accepted tokens per draft position\n" ++
                "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n");
            for (metrics.n_accepted_per_pos.items, 0..) |n, i| {
                try prometheus.print(a, "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position=\"{d}\"}} {d}\n", .{ i, n });
            }
        }

        return prometheus.toOwnedSlice(a);
    }

    fn addItems(a: Allocator, prometheus: *std.ArrayList(u8), ty: []const u8, items: []const MetricItem) Error!void {
        for (items) |item| {
            var buf: [64]u8 = undefined;
            const v = ostreamDouble(&buf, item.value);
            try prometheus.print(a, "# HELP llamacpp:{s} {s}\n", .{ item.name, item.description });
            try prometheus.print(a, "# TYPE llamacpp:{s} {s}\n", .{ item.name, ty });
            try prometheus.print(a, "llamacpp:{s} {s}\n", .{ item.name, v });
        }
    }

    /// Ports `server_task_result_slot_save_load`'s `to_json` (server-task.cpp:1667 @8642541b2).
    fn slotSaveLoadToJson(self: *Result, a: Allocator) Error!Value {
        const s = &self.kind.slot_save_load;
        var timings = Obj.init(a);
        errdefer timings.deinit();
        try timings.put(if (s.is_save) "save_ms" else "restore_ms", s.t_ms);
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("id_slot", self.id_slot);
        try o.put("filename", s.filename);
        try o.put(if (s.is_save) "n_saved" else "n_restored", s.n_tokens);
        try o.put(if (s.is_save) "n_written" else "n_read", s.n_bytes);
        try o.put("timings", timings.done());
        return o.done();
    }

    /// Ports `server_task_result_slot_erase`'s `to_json` (server-task.cpp:1694 @8642541b2).
    fn slotEraseToJson(self: *Result, a: Allocator) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("id_slot", self.id_slot);
        try o.put("n_erased", self.kind.slot_erase.n_erased);
        return o.done();
    }

    /// Ports `server_task_result_control`'s `to_json` (server-task.h:662 @8642541b2).
    fn controlToJson(self: *Result, a: Allocator) Error!Value {
        const ctl = &self.kind.control;
        var out = Obj.init(a);
        errdefer out.deinit();
        try out.put("success", ctl.success);
        if (ctl.message.len != 0) {
            try out.put("message", ctl.message);
        }
        return out.done();
    }

    /// Ports `server_task_result_get_lora`'s `to_json` (server-task.cpp:1705 @8642541b2).
    fn getLoraToJson(self: *Result, a: Allocator) Error!Value {
        var result = Value.emptyArray();
        errdefer result.deinit(a);
        for (self.kind.get_lora.loras, 0..) |lora, i| {
            var entry = Obj.init(a);
            errdefer entry.deinit();
            try entry.put("id", i);
            try entry.put("path", lora.info.path);
            try entry.put("scale", lora.info.scale);
            try entry.put("task_name", lora.info.task_name);
            try entry.put("prompt_prefix", lora.info.prompt_prefix);
            if (lora.alora_invocation_tokens.len != 0) {
                try entry.put("alora_invocation_string", lora.alora_invocation_string);
                try entry.put("alora_invocation_tokens", lora.alora_invocation_tokens);
            }
            try result.pushBack(a, entry.done());
        }
        return result;
    }

    /// Ports `server_task_result_apply_lora`'s `to_json` (server-task.cpp:1729 @8642541b2).
    fn applyLoraToJson(a: Allocator) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("success", true);
        return o.done();
    }
};

extern fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

/// `std::ostream << double` at the default precision: `%g` with 6 digits.
fn ostreamDouble(buf: *[64]u8, v: f64) []const u8 {
    const n = snprintf(buf, buf.len, "%g", v);
    return buf[0..@intCast(n)];
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an error result carries the context size when it exceeded it" {
    const a = std.testing.allocator;
    const r = try Result.create(a, .err);
    defer r.destroy(a);
    r.kind.err = .{ .err_type = .exceed_context_size, .err_msg = "too long", .n_prompt_tokens = 9, .n_ctx = 8 };
    var v = try r.toJson(a);
    defer v.deinit(a);
    const s = try json.dump(a, &v, -1);
    defer a.free(s);
    try std.testing.expectEqualStrings("{\"code\":400,\"message\":\"too long\",\"type\":\"exceed_context_size_error\",\"n_prompt_tokens\":9,\"n_ctx\":8}", s);
}

test "metrics text formats values as an ostream does" {
    const a = std.testing.allocator;
    const r = try Result.create(a, .metrics);
    defer r.destroy(a);
    r.kind.metrics.metrics.prompt.count = 1234567;
    const s = try r.toMetrics(a);
    defer a.free(s);
    try std.testing.expect(std.mem.indexOf(u8, s, "llamacpp:prompt_tokens_total 1.23457e+06\n") != null);
}
