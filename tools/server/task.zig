//! A task -- what an HTTP thread hands the decode loop -- with its
//! parameters, the state an HTTP thread keeps while its results stream
//! back, a generated token's probabilities, and the prompt cache.
//!
//! # Provenance
//!
//! Ports `task_params`, `task_result_state`, `server_task`,
//! `result_prompt_progress`, `completion_token_output` and the prompt
//! cache (`server_prompt`, `server_prompt_data`,
//! `server_prompt_cache_state`, `server_prompt_cache`) from
//! `llama.cpp/tools/server/server-task.cpp` and
//! `llama.cpp/tools/server/server-task.h`; `common_grammar_value` and
//! `common_prompt_checkpoint` from
//! `llama.cpp/common/common.h` and `llama.cpp/common/common.cpp`
//! (llama.cpp v0.8.0, `8642541b2`). The result types are `result.zig`.
//! Each declaration names what it ports and the line it began at.
//!
//! # The chat parse
//!
//! Since v0.7.0 a `TaskResultState` owns a copy of the request's chat
//! session (`cli/common_chat`'s `common_chat_session`), which holds the
//! request's PEG parser and everything generated so far, text and tokens;
//! `update_chat_msg` feeds it each chunk.
//!
//! # Ownership
//!
//! `TaskParams` owns nothing itself: its strings and
//! lists live in the allocator their owner gave them, which is an arena
//! (a `Task`'s, a result's), and `clone` deep-copies into another one. A
//! `Task` owns its arena, its tokens and its children; a
//! `TaskResultState` owns everything in it through the allocator `init`
//! took.

const std = @import("std");
const json = @import("json");
const chat = @import("common_chat");
const tools = @import("tools_common");
const common = @import("common.zig");
const tokens_mod = @import("tokens.zig");
const queue = @import("queue.zig");

const c = tools.c;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Error = common.Error;
const Obj = common.Obj;
const Token = common.Token;
const GrammarTrigger = common.GrammarTrigger;
const ServerTokens = tokens_mod.ServerTokens;

pub const SamplerType = tools.params.SamplerType;
pub const SpeculativeType = tools.params.SpeculativeType;
pub const GrammarType = tools.params.GrammarType;

// -----------------------------------------------------------------------------
// common_params_sampling, as a task carries it

/// `common_params_sampling` (common/common.h:229 @8642541b2), as
/// `tools/common/params.zig` ports it: what a task's sampler is built
/// from. `common_grammar` is its `grammar_type` and `grammar`.
pub const Sampling = tools.params.Sampling;

/// Ports `common_grammar_value` (common/common.h:217 @8642541b2).
pub fn grammarValue(s: *const Sampling) []const u8 {
    return s.grammar;
}

/// `preserved_tokens.insert(tok)` on the `std::set` `Sampling` keeps sorted.
pub fn preserveToken(a: Allocator, s: *Sampling, tok: Token) Error!void {
    var i: usize = 0;
    while (i < s.preserved_tokens.items.len) : (i += 1) {
        const t = s.preserved_tokens.items[i];
        if (t == tok) return;
        if (t > tok) break;
    }
    try s.preserved_tokens.insert(a, i, tok);
}

/// A deep copy into `a`, as a C++ copy constructor makes one: every
/// string, slice and `std.ArrayList` duplicated, recursively; numbers,
/// enums and pointers to C objects copied. Used for the sampling and
/// speculative parameters, whose members `tools/common/` owns -- a member
/// added there is copied without a change here.
pub fn deepClone(a: Allocator, v: anytype) Error!@TypeOf(v) {
    const T = @TypeOf(v);
    switch (@typeInfo(T)) {
        .pointer => |p| {
            if (p.size != .slice) return v;
            const out = try a.alloc(p.child, v.len);
            for (v, 0..) |e, i| out[i] = try deepClone(a, e);
            return out;
        },
        .@"struct" => |st| {
            if (@hasField(T, "items") and @hasField(T, "capacity") and @hasDecl(T, "Slice")) {
                // a `std.ArrayList`
                const items = try deepClone(a, @as([]const @typeInfo(@FieldType(T, "items")).pointer.child, v.items));
                return .fromOwnedSlice(@constCast(items));
            }
            var out: T = v;
            inline for (st.field_names) |name| {
                @field(out, name) = try deepClone(a, @field(v, name));
            }
            return out;
        },
        .array => {
            var out = v;
            for (&out, v) |*o, e| o.* = try deepClone(a, e);
            return out;
        },
        else => return v,
    }
}

/// `params.speculative = params_base.speculative`: a deep copy into `a`.
fn cloneSpeculative(a: Allocator, s: *const tools.params.Speculative) Error!tools.params.Speculative {
    return deepClone(a, s.*);
}

/// Ports `common_speculative_type_name_str` (speculative.cpp:2387 @8642541b2).
///
/// Return: owned by `a`.
pub fn speculativeTypeNameStr(a: Allocator, types: []const SpeculativeType) Error![]u8 {
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(a);

    for (types, 0..) |t, i| {
        if (i > 0) {
            try result.append(a, ',');
        }
        try result.appendSlice(a, t.toStr());
    }
    return result.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// enums

/// Ports `server_task_type` (server-task.h:16 @8642541b2).
pub const TaskType = enum(c_int) {
    completion,
    embedding,
    rerank,
    decision,
    infill,
    transcribe_rt,
    realtime, // a conversation over /v1/realtime, see realtime.zig
    cancel,
    control,
    next_response,
    metrics,
    slot_get,
    slot_save,
    slot_restore,
    slot_erase,
    get_lora,
    set_lora,
    tts,
};

/// Ports `task_response_type` (server-task.h:38 @8642541b2).
pub const ResponseType = enum(c_int) {
    none, // llama.cpp native format
    oai_chat,
    oai_cmpl,
    oai_resp,
    oai_asr, // transcriptions API
    oai_embd,
    anthropic,
};

/// Ports `stop_type` (server-task.h:48 @8642541b2).
pub const StopType = enum(c_int) {
    none,
    eos,
    word,
    limit,
};

/// Ports `server_control_action` (server-task.h:55 @8642541b2).
pub const ControlAction = enum(c_int) {
    none,
    reasoning_end,
    audio_append, // push control_audio into the slot's audio stream
    audio_commit, // flush the audio stream, more audio can follow
    audio_end, // flush the audio stream and end it
    // realtime conversations (SERVER_TASK_TYPE_REALTIME), applied in order with the audio actions
    text_append, // append control_text to the conversation
    response_create, // generate after control_text, see control_response
    response_cancel, // stop the response being generated
    clear, // drop the conversation
};

/// Ports `server_control_response` (server-task.h:69 @8642541b2): a response
/// of a realtime conversation.
pub const ControlResponse = struct {
    out_mode: c.enum_mtmd_helper_gen_audio_out_mode = c.MTMD_HELPER_GEN_AUDIO_OUT_AUTO,
    sampling: Sampling = .{}, // of the text tokens: tool call grammar and preserved tokens
    n_predict: i32 = -1, // text tokens and audio frames, -1: until the context is full
};

/// Ports `stop_type_to_str` (server-task.cpp:243 @8642541b2).
pub fn stopTypeToStr(ty: StopType) []const u8 {
    return switch (ty) {
        .eos => "eos",
        .word => "word",
        .limit => "limit",
        else => "none",
    };
}

// -----------------------------------------------------------------------------
// task_params

/// Ports `task_params` (server-task.h:75 @8642541b2). Strings and lists are
/// the owner's arena's.
pub const TaskParams = struct {
    stream: bool = false,
    include_usage: bool = false,
    cache_prompt: bool = true, // remember the prompt to avoid reprocessing all prompt
    return_tokens: bool = false,
    return_progress: bool = false,

    sse_ping_interval: i32 = 30, // seconds between SSE comment pings while the stream stays silent, -1 disables

    n_keep: i32 = 0, // number of tokens to keep from initial prompt
    n_discard: i32 = 0, // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    n_predict: i32 = -1, // new tokens to predict
    n_indent: i32 = 0, // minimum line indentation for the generated text in number of whitespace characters
    n_cmpl: i32 = 1, // number of completions to generate from this prompt

    n_cache_reuse: i32 = 0, // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    t_max_prompt_ms: i64 = -1, // TODO: implement
    t_max_predict_ms: i64 = -1, // if positive, limit the generation phase to this time limit

    lora: common.LoraMap = .{}, // mapping adapter ID -> scale

    antiprompt: []const []const u8 = &.{},
    response_fields: []const []const u8 = &.{},

    timings_per_token: bool = false,
    post_sampling_probs: bool = false,

    sampling: Sampling = .{},
    speculative: tools.params.Speculative = .{},

    // response formatting
    verbose: bool = false,
    res_type: ResponseType = .none,
    oaicompat_model: []const u8 = "",
    oaicompat_cmpl_id: []const u8 = "",

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    control_action: ControlAction = .none,
    control_cmpl_id: []const u8 = "",
    control_audio: []const f32 = &.{}, // PCM F32 samples for SERVER_CONTROL_ACTION_AUDIO_APPEND
    control_text: []const u8 = "", // raw text, special tokens are parsed
    control_response: ControlResponse = .{},

    // reported in generation_settings, parsing itself is owned by the chat session
    chat_format: chat.Format = .content_only,
    reasoning_format: chat.ReasoningFormat = .none,

    // message spans for checkpointing
    message_spans: chat.MsgSpans = .{},

    // Embeddings
    embd_normalize: i32 = 2, // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    /// `task_params()`: the C++ default member initializers, whose sampling
    /// and speculative lists are not empty -- the DRY breakers, the sampler
    /// chain and the `{ none }` speculative type -- allocated in `a`. A
    /// `TaskParams` written `.{}` has those lists empty; only a cancel or
    /// control task, whose parameters nothing reads, should be one.
    pub fn init(a: Allocator) Error!TaskParams {
        const defaults = try tools.params.Params.init(a);
        return .{ .sampling = defaults.sampling, .speculative = defaults.speculative };
    }

    /// `task_params(const task_params &)`: a deep copy into `a`.
    pub fn clone(self: *const TaskParams, a: Allocator) Error!TaskParams {
        var p = self.*;
        p.lora = try self.lora.clone(a);
        p.antiprompt = try deepClone(a, self.antiprompt);
        p.response_fields = try deepClone(a, self.response_fields);
        p.sampling = try deepClone(a, self.sampling);
        p.speculative = try cloneSpeculative(a, &self.speculative);
        p.oaicompat_model = try a.dupe(u8, self.oaicompat_model);
        p.oaicompat_cmpl_id = try a.dupe(u8, self.oaicompat_cmpl_id);
        p.control_cmpl_id = try a.dupe(u8, self.control_cmpl_id);
        p.control_audio = try a.dupe(f32, self.control_audio);
        p.control_text = try a.dupe(u8, self.control_text);
        p.control_response.sampling = try deepClone(a, self.control_response.sampling);
        p.message_spans = .{ .spans = .fromOwnedSlice(try a.dupe(chat.MsgSpan, self.message_spans.spans.items)) };
        return p;
    }

    /// Ports `format_logit_bias` (server-task.cpp:19 @8642541b2), a member of `task_params`.
    ///
    /// Return: owned by `a`.
    pub fn formatLogitBias(self: *const TaskParams, a: Allocator, logit_bias: []const c.llama_logit_bias) Error!Value {
        _ = self;
        var data = Value.emptyArray();
        errdefer data.deinit(a);
        for (logit_bias) |lb| {
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("bias", lb.bias);
            try o.put("token", lb.token);
            try data.pushBack(a, o.done());
        }
        return data;
    }

    /// Ports `to_json` (server-task.cpp:30 @8642541b2), a member of
    /// `task_params`: the request's settings as `/completion` echoes them
    /// back; `only_metrics` leaves out the lists and the grammar.
    ///
    /// Return: owned by `a`.
    pub fn toJson(self: *const TaskParams, a: Allocator, only_metrics: bool) Error!Value {
        const sampling = &self.sampling;
        var samplers = try a.alloc([]const u8, sampling.samplers.items.len);
        defer a.free(samplers);
        for (sampling.samplers.items, 0..) |s, i| samplers[i] = s.toStr();

        var lora = Value.emptyArray();
        defer lora.deinit(a);
        for (self.lora.entries.items) |it| {
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("id", it.id);
            try o.put("scale", it.scale);
            try lora.pushBack(a, o.done());
        }

        const spec_types = try speculativeTypeNameStr(a, self.speculative.types.items);
        defer a.free(spec_types);

        var o = Obj.init(a);
        errdefer o.deinit();

        if (only_metrics) {
            try o.put("seed", sampling.seed);
            try o.put("temperature", sampling.temp);
            try o.put("dynatemp_range", sampling.dynatemp_range);
            try o.put("dynatemp_exponent", sampling.dynatemp_exponent);
            try o.put("top_k", sampling.top_k);
            try o.put("top_p", sampling.top_p);
            try o.put("min_p", sampling.min_p);
            try o.put("top_n_sigma", sampling.top_n_sigma);
            try o.put("xtc_probability", sampling.xtc_probability);
            try o.put("xtc_threshold", sampling.xtc_threshold);
            try o.put("typical_p", sampling.typ_p);
            try o.put("repeat_last_n", sampling.penalty_last_n);
            try o.put("repeat_penalty", sampling.penalty_repeat);
            try o.put("presence_penalty", sampling.penalty_present);
            try o.put("frequency_penalty", sampling.penalty_freq);
            try o.put("dry_multiplier", sampling.dry_multiplier);
            try o.put("dry_base", sampling.dry_base);
            try o.put("dry_allowed_length", sampling.dry_allowed_length);
            try o.put("dry_penalty_last_n", sampling.dry_penalty_last_n);
            try o.put("mirostat", sampling.mirostat);
            try o.put("mirostat_tau", sampling.mirostat_tau);
            try o.put("mirostat_eta", sampling.mirostat_eta);
            try o.put("adaptive_target", sampling.adaptive_target);
            try o.put("adaptive_decay", sampling.adaptive_decay);
            try o.put("max_tokens", self.n_predict);
            try o.put("n_predict", self.n_predict); // TODO: deduplicate?
            try o.put("n_keep", self.n_keep);
            try o.put("n_discard", self.n_discard);
            try o.put("ignore_eos", sampling.ignore_eos);
            try o.put("stream", self.stream);
            try o.put("n_probs", sampling.n_probs);
            try o.put("min_keep", sampling.min_keep);
            try o.put("chat_format", self.chat_format.name());
            try o.put("reasoning_format", self.reasoning_format.name());
            try o.put("reasoning_in_content", self.stream and self.reasoning_format == .deepseek_legacy);
            try o.put("generation_prompt", sampling.generation_prompt);
            try o.put("samplers", @as([]const []const u8, samplers));
            try o.put("speculative.types", spec_types);
            try o.put("timings_per_token", self.timings_per_token);
            try o.put("post_sampling_probs", self.post_sampling_probs);
            try o.put("backend_sampling", sampling.backend_sampling);
            try o.put("lora", try lora.clone(a));
            return o.done();
        }

        var grammar_triggers = Value.emptyArray();
        defer grammar_triggers.deinit(a);
        for (sampling.grammar_triggers.items) |trigger| {
            const ct: GrammarTrigger = .{ .type = @backingInt(trigger.type), .value = trigger.value, .token = trigger.token };
            try grammar_triggers.pushBack(a, try ct.toJson(a));
        }

        try o.put("seed", sampling.seed);
        try o.put("temperature", sampling.temp);
        try o.put("dynatemp_range", sampling.dynatemp_range);
        try o.put("dynatemp_exponent", sampling.dynatemp_exponent);
        try o.put("top_k", sampling.top_k);
        try o.put("top_p", sampling.top_p);
        try o.put("min_p", sampling.min_p);
        try o.put("top_n_sigma", sampling.top_n_sigma);
        try o.put("xtc_probability", sampling.xtc_probability);
        try o.put("xtc_threshold", sampling.xtc_threshold);
        try o.put("typical_p", sampling.typ_p);
        try o.put("repeat_last_n", sampling.penalty_last_n);
        try o.put("repeat_penalty", sampling.penalty_repeat);
        try o.put("presence_penalty", sampling.penalty_present);
        try o.put("frequency_penalty", sampling.penalty_freq);
        try o.put("dry_multiplier", sampling.dry_multiplier);
        try o.put("dry_base", sampling.dry_base);
        try o.put("dry_allowed_length", sampling.dry_allowed_length);
        try o.put("dry_penalty_last_n", sampling.dry_penalty_last_n);
        try o.put("dry_sequence_breakers", sampling.dry_sequence_breakers.items);
        try o.put("mirostat", sampling.mirostat);
        try o.put("mirostat_tau", sampling.mirostat_tau);
        try o.put("mirostat_eta", sampling.mirostat_eta);
        try o.put("adaptive_target", sampling.adaptive_target);
        try o.put("adaptive_decay", sampling.adaptive_decay);
        try o.put("stop", self.antiprompt);
        try o.put("max_tokens", self.n_predict);
        try o.put("n_predict", self.n_predict); // TODO: deduplicate?
        try o.put("n_keep", self.n_keep);
        try o.put("n_discard", self.n_discard);
        try o.put("ignore_eos", sampling.ignore_eos);
        try o.put("stream", self.stream);
        try o.put("logit_bias", try self.formatLogitBias(a, sampling.logit_bias.items));
        try o.put("n_probs", sampling.n_probs);
        try o.put("min_keep", sampling.min_keep);
        try o.put("grammar", grammarValue(sampling));
        try o.put("grammar_lazy", sampling.grammar_lazy);
        try o.put("grammar_triggers", try grammar_triggers.clone(a));
        try o.put("preserved_tokens", sampling.preserved_tokens.items);
        try o.put("chat_format", self.chat_format.name());
        try o.put("reasoning_format", self.reasoning_format.name());
        try o.put("reasoning_in_content", self.stream and self.reasoning_format == .deepseek_legacy);
        try o.put("generation_prompt", sampling.generation_prompt);
        try o.put("samplers", @as([]const []const u8, samplers));
        try o.put("speculative.types", spec_types);
        try o.put("timings_per_token", self.timings_per_token);
        try o.put("post_sampling_probs", self.post_sampling_probs);
        try o.put("backend_sampling", sampling.backend_sampling);
        try o.put("lora", try lora.clone(a));
        return o.done();
    }
};

// -----------------------------------------------------------------------------
// task_result_state

/// Ports `task_result_state` (server-task.h:134 @8642541b2): what an HTTP
/// thread tracks across a task's streamed results -- the text so far, the
/// message parsed from it, tool-call ids, and the Responses and Anthropic
/// streams' block state.
pub const TaskResultState = struct {
    gpa: Allocator,

    // tracking diffs for partial tool calls (kept, as upstream keeps it; unused there too)
    diffs: std.ArrayList(chat.MsgDiff) = .empty,
    chat_session: Session, // owns all parsing for this generation
    chat_msg: chat.Msg = .{},
    generated_tool_call_ids: std.ArrayList([]const u8) = .empty,
    /// `std::unordered_set<size_t>`, only ever tested and inserted into.
    sent_tool_call_names: std.AutoHashMapUnmanaged(usize, void) = .empty,

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    thinking_block_started: bool = false,
    text_block_started: bool = false,

    // for OpenAI Responses streaming API
    oai_resp_created: bool = false,
    oai_resp_id: []const u8 = "",
    oai_resp_reasoning_id: []const u8 = "",
    oai_resp_message_id: []const u8 = "",
    oai_resp_fc_id: []const u8 = "", // function call ID for current args delta

    /// The chat session a state owns.
    pub const Session = chat.session.Session;

    /// Ports the constructor of `task_result_state` (server-task.cpp:151 @8642541b2):
    /// a copy of the request's chat session (null: the default one, which
    /// parses everything as content), its message so far, and fresh
    /// Responses ids.
    pub fn init(gpa: Allocator, session: ?*const Session) Error!TaskResultState {
        var self: TaskResultState = .{
            .gpa = gpa,
            .chat_session = if (session) |ss| try ss.clone(gpa) else try Session.initEmpty(gpa),
        };
        errdefer self.deinit(gpa);
        self.chat_msg = try self.chat_session.msg().clone(gpa);
        self.oai_resp_id = try prefixed(gpa, "resp_");
        self.oai_resp_reasoning_id = try prefixed(gpa, "rs_");
        self.oai_resp_message_id = try prefixed(gpa, "msg_");
        self.oai_resp_fc_id = try gpa.dupe(u8, "");
        return self;
    }

    fn prefixed(gpa: Allocator, prefix: []const u8) Error![]u8 {
        const r = try common.randomString(gpa);
        defer gpa.free(r);
        return std.mem.concat(gpa, u8, &.{ prefix, r });
    }

    /// Frees everything; the `gpa` argument is the queue's, the same one.
    pub fn deinit(self: *TaskResultState, gpa: Allocator) void {
        _ = gpa;
        const a = self.gpa;
        self.diffs.deinit(a);
        self.chat_session.deinit();
        self.chat_msg.deinit(a);
        for (self.generated_tool_call_ids.items) |s| a.free(s);
        self.generated_tool_call_ids.deinit(a);
        self.sent_tool_call_names.deinit(a);
        a.free(self.oai_resp_id);
        a.free(self.oai_resp_reasoning_id);
        a.free(self.oai_resp_message_id);
        a.free(self.oai_resp_fc_id);
    }

    /// `oai_resp_fc_id = id`.
    pub fn setFcId(self: *TaskResultState, id: []const u8) Error!void {
        try chat.Msg.setField(self.gpa, &self.oai_resp_fc_id, id);
    }

    fn genId(_: void, a: Allocator) anyerror![]const u8 {
        return common.genToolCallId(a);
    }

    /// Ports `update_chat_msg` (server-task.cpp:159 @8642541b2), a member of
    /// `task_result_state`: feeds the new text and tokens to the chat
    /// session (`finish` on the last), and reports what changed. With `filter_tool_calls`
    /// a tool call's name is held back until its arguments start (or the
    /// stream ends), and each later delta carries only arguments.
    ///
    /// Parameters:
    /// - `out_a`: owns the strings of the diffs added to `diffs`.
    /// - `diffs`: replaced when not filtering, appended to when filtering.
    ///
    /// Return: the message parsed so far, borrowed from the state.
    pub fn updateChatMsg(
        self: *TaskResultState,
        out_a: Allocator,
        added: *const chat.input.Input,
        is_partial: bool,
        diffs: *std.ArrayList(chat.MsgDiff),
        filter_tool_calls: bool,
    ) Error!*const chat.Msg {
        const gpa = self.gpa;
        var msg_prv_copy = try self.chat_msg.clone(gpa);
        defer msg_prv_copy.deinit(gpa);
        var new_msg = try (if (is_partial) try self.chat_session.feed(added) else try self.chat_session.finish(added)).clone(gpa);
        if (new_msg.empty()) {
            new_msg.deinit(gpa);
            return &self.chat_msg;
        }
        {
            errdefer new_msg.deinit(gpa);
            new_msg.setToolCallIds(gpa, &self.generated_tool_call_ids, {}, genId) catch return error.OutOfMemory;
        }
        self.chat_msg.deinit(gpa);
        self.chat_msg = new_msg;
        var all_diffs = try chat.msg.computeDiffs(gpa, &msg_prv_copy, &self.chat_msg);
        defer all_diffs.deinit(gpa);

        if (!filter_tool_calls) {
            diffs.clearRetainingCapacity();
            for (all_diffs.items) |d| try diffs.append(out_a, try cloneDiff(out_a, d));
        } else {
            const calls = self.chat_msg.tool_calls.items;
            for (all_diffs.items) |d_in| {
                var d = d_in;
                // If this is a new type of delta, flush all currently pending tool call names
                for (calls, 0..) |tc, i| {
                    if (self.sent_tool_call_names.contains(i) or tc.name.len == 0) {
                        continue;
                    }
                    if (d.tool_call_index != i or d.tool_call_delta.arguments.len != 0) {
                        var header: chat.MsgDiff = .{};
                        header.tool_call_index = i;
                        header.tool_call_delta.id = tc.id;
                        header.tool_call_delta.name = tc.name;
                        try diffs.append(out_a, try cloneDiff(out_a, header));
                        try self.sent_tool_call_names.put(gpa, i, {});
                    }
                }

                if (d.tool_call_index == chat.msg.npos) {
                    try diffs.append(out_a, try cloneDiff(out_a, d));
                } else {
                    const i = d.tool_call_index;
                    if (self.sent_tool_call_names.contains(i)) {
                        if (d.tool_call_delta.arguments.len != 0) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id = "";
                            try diffs.append(out_a, try cloneDiff(out_a, d));
                        }
                    } else {
                        // Not sent yet.
                        if (d.tool_call_delta.arguments.len != 0 or !is_partial) {
                            d.tool_call_delta.name = calls[i].name;
                            d.tool_call_delta.id = calls[i].id;
                            try diffs.append(out_a, try cloneDiff(out_a, d));
                            try self.sent_tool_call_names.put(gpa, i, {});
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (calls, 0..) |tc, i| {
                    if (!self.sent_tool_call_names.contains(i) and tc.name.len != 0) {
                        var header: chat.MsgDiff = .{};
                        header.tool_call_index = i;
                        header.tool_call_delta.id = tc.id;
                        header.tool_call_delta.name = tc.name;
                        try diffs.append(out_a, try cloneDiff(out_a, header));
                        try self.sent_tool_call_names.put(gpa, i, {});
                    }
                }
            }
        }
        return &self.chat_msg;
    }
};

/// A diff whose strings `a` owns.
pub fn cloneDiff(a: Allocator, d: chat.MsgDiff) Error!chat.MsgDiff {
    return .{
        .reasoning_content_delta = try a.dupe(u8, d.reasoning_content_delta),
        .content_delta = try a.dupe(u8, d.content_delta),
        .tool_call_index = d.tool_call_index,
        .tool_call_delta = .{
            .name = try a.dupe(u8, d.tool_call_delta.name),
            .arguments = try a.dupe(u8, d.tool_call_delta.arguments),
            .id = try a.dupe(u8, d.tool_call_delta.id),
        },
    };
}

// -----------------------------------------------------------------------------
// server_task

/// Ports `gen_audio`'s `inp` (mtmd-helper.h:344 @8642541b2), the RAII wrapper
/// of `mtmd_helper_gen_audio_inp`. A `set_*` keeps its own
/// copy; `get` hands out pointers valid as long as this is alive.
pub const TtsInp = struct {
    data: c.mtmd_helper_gen_audio_inp = undefined,
    prompt_str: [:0]const u8 = "",
    lang_str: [:0]const u8 = "",
    voice_str: [:0]const u8 = "",
    speaker_ref_ptr: ?*c.mtmd_bitmap = null,

    /// The C++'s default member initializer.
    pub fn init() TtsInp {
        return .{ .data = c.mtmd_helper_gen_audio_inp_default() };
    }

    /// `set_speaker_ref`: takes the bitmap, the one it held is freed.
    pub fn setSpeakerRef(self: *TtsInp, bmp: ?*c.mtmd_bitmap) void {
        if (self.speaker_ref_ptr) |old| c.mtmd_bitmap_free(old);
        self.speaker_ref_ptr = bmp;
    }

    /// `get`.
    pub fn get(self: *TtsInp) *const c.mtmd_helper_gen_audio_inp {
        self.data.prompt = self.prompt_str.ptr;
        self.data.prompt_len = self.prompt_str.len;
        self.data.lang = if (self.lang_str.len == 0) null else self.lang_str.ptr;
        self.data.voice = if (self.voice_str.len == 0) null else self.voice_str.ptr;
        self.data.speaker_ref = self.speaker_ref_ptr;
        return &self.data;
    }

    /// The destructor: the strings are the task's arena's.
    pub fn deinit(self: *TtsInp) void {
        self.setSpeakerRef(null);
    }
};

/// Ports `server_task` (server-task.h:164 @8642541b2). Movable: everything it
/// owns is behind its arena pointer, its token list and its child list.
pub const Task = struct {
    /// Ports `server_task`'s `decision` (server-task.h:207 @8642541b2),
    /// used by SERVER_TASK_TYPE_DECISION: where to read the model output of
    /// each option, exactly one of the two lists is used. The lists are the
    /// task's own (`gpa`), freed by `Task.deinit`.
    pub const Decision = struct {
        labels: std.ArrayList(c.llama_token) = .empty, // logits of these tokens, at the last prompt token
        label_groups: std.ArrayList(i32) = .empty, // if set, number of labels per output, the output is their max
        markers: std.ArrayList(i32) = .empty, // embeddings[column] at these prompt positions
        column: i32 = 0,
        // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
        pointer: i32 = -1,
        // if set, the prompt must be the only sequence of its batch
        no_batching: bool = false,

        // if set, the outputs are the pooled embeddings of the sequence, one score per class
        pooled: bool = false,

        // for a joint head: one value per prompt token, see llama_batch_ext_set_decision_order()
        // the scores are the first n_scores rows of the embeddings
        order: std.ArrayList(i32) = .empty,
        n_scores: i32 = 0,

        /// Ports `pos_first` (server-task.h:221 @8642541b2): the first
        /// prompt position that is read, -1 if none.
        pub fn posFirst(self: *const Decision) i32 {
            var pos = self.pointer;
            for (self.markers.items) |marker| {
                pos = if (pos < 0) marker else @min(pos, marker);
            }
            for (self.order.items, 0..) |o, i| {
                if (o != 0) {
                    const ii: i32 = @intCast(i);
                    return if (pos < 0) ii else @min(pos, ii);
                }
            }
            return pos;
        }

        pub fn deinit(self: *Decision, gpa: Allocator) void {
            self.labels.deinit(gpa);
            self.label_groups.deinit(gpa);
            self.markers.deinit(gpa);
            self.order.deinit(gpa);
        }
    };

    /// Ports `server_task`'s `slot_action` (server-task.h:195 @8642541b2).
    pub const SlotAction = struct {
        id_slot: c_int = 0,
        filename: []const u8 = "",
        filepath: []const u8 = "",
    };

    id: c_int = -1, // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    index: usize = 0, // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    id_target: c_int = -1,
    id_slot: c_int = -1,

    // used by parallel sampling (multiple completions from same prompt)
    id_parent: c_int = -1,
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    child_tasks: std.ArrayList(Task) = .empty,
    // if set on a parent, the children have their own prompt and only share its first n_tokens_shared tokens
    n_tokens_shared: i32 = 0,

    // used by SERVER_TASK_TYPE_INFERENCE
    params: TaskParams = .{},
    tokens: ServerTokens = .{},

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    cli: bool = false,
    cli_prompt: []const u8 = "",
    cli_files: []const []const u8 = &.{},

    type: TaskType = .completion,

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    slot_action: SlotAction = .{},

    // used by SERVER_TASK_TYPE_METRICS
    metrics_reset_bucket: bool = false,

    // used by SERVER_TASK_TYPE_DECISION
    decision: Decision = .{},

    // used by SERVER_TASK_TYPE_SET_LORA
    set_lora: common.LoraMap = .{}, // mapping adapter ID -> scale

    // used by SERVER_TASK_TYPE_TTS
    tts_inp: TtsInp = .{},

    /// Owns `params`, the CLI strings, `slot_action`'s names and
    /// `set_lora`; null until `arenaAlloc` first runs.
    arena: ?*std.heap.ArenaAllocator = null,

    /// `server_task(type)`: a task that owns nothing yet.
    pub fn initType(ty: TaskType) Task {
        return .{ .type = ty };
    }

    /// The task's arena, made on first use with `gpa`.
    pub fn arenaAlloc(self: *Task, gpa: Allocator) Error!Allocator {
        if (self.arena == null) {
            const ar = try gpa.create(std.heap.ArenaAllocator);
            ar.* = .init(gpa);
            self.arena = ar;
        }
        return self.arena.?.allocator();
    }

    /// The destructor: the children, the tokens, the arena.
    pub fn deinit(self: *Task, gpa: Allocator) void {
        for (self.child_tasks.items) |*ch| ch.deinit(gpa);
        self.child_tasks.deinit(gpa);
        self.decision.deinit(gpa);
        self.tts_inp.deinit();
        self.tokens.deinit();
        if (self.arena) |ar| {
            ar.deinit();
            gpa.destroy(ar);
        }
        self.* = .{};
    }

    /// Ports `n_tokens` (server-task.h:251 @8642541b2).
    pub fn nTokens(self: *const Task) i32 {
        return @truncate(@as(i64, @intCast(self.tokens.size())));
    }

    /// Ports `need_embd` (server-task.h:255 @8642541b2).
    pub fn needEmbd(self: *const Task) bool {
        return switch (self.type) {
            .embedding, .rerank => true,
            .decision => self.decision.markers.items.len != 0 or self.decision.order.items.len != 0 or self.decision.pooled,
            else => false,
        };
    }

    /// Ports `need_logits` (server-task.h:267 @8642541b2).
    pub fn needLogits(self: *const Task) bool {
        return switch (self.type) {
            .completion, .infill, .transcribe_rt => true,
            .decision => self.decision.labels.items.len != 0,
            else => false,
        };
    }

    /// Ports `need_sampling` (server-task.h:280 @8642541b2).
    pub fn needSampling(self: *const Task) bool {
        return switch (self.type) {
            .completion, .infill, .tts, .transcribe_rt => true,
            else => false,
        };
    }

    /// Ports `get_list_id` (server-task.h:293 @8642541b2): every task's id and
    /// its children's, in a libc++ `unordered_set` reserved for `tasks.size()`.
    pub fn getListId(gpa: Allocator, tasks: []const Task) Error!queue.IdSet {
        var ids = try queue.IdSet.initCapacity(gpa, tasks.len);
        errdefer ids.deinit();
        for (tasks) |t| {
            try ids.insert(t.id);
            for (t.child_tasks.items) |child| {
                try ids.insert(child.id);
            }
        }
        return ids;
    }

    /// Ports `apply_chat_session` (server-task.h:304 @8642541b2): the
    /// session's sampling (grammar, triggers, preserved tokens, generation
    /// prompt), format, stop strings and message spans, for a templated
    /// session; an untemplated one changes nothing.
    pub fn applyChatSession(self: *Task, gpa: Allocator, session: *const chat.session.Session) Error!void {
        if (!session.hasTemplate()) {
            return;
        }

        const a = try self.arenaAlloc(gpa);
        try session.applySampling(a, &self.params.sampling);
        self.params.chat_format = session.format();
        const stops = session.additionalStops();
        const antiprompt = try a.alloc([]const u8, self.params.antiprompt.len + stops.len);
        @memcpy(antiprompt[0..self.params.antiprompt.len], self.params.antiprompt);
        for (stops, 0..) |stop, i| antiprompt[self.params.antiprompt.len + i] = try a.dupe(u8, stop);
        self.params.antiprompt = antiprompt;
        self.params.message_spans = try self.tokens.findMessageSpans(a, session.messageDelimiters());
    }

    /// Ports `add_child` (server-task.h:315 @8642541b2): a copy of this task
    /// as a child, with its own sampling seed.
    pub fn addChild(self: *Task, gpa: Allocator, id_parent: c_int, id_child: c_int) Error!void {
        var copy: Task = .{};
        errdefer copy.deinit(gpa);

        copy.id = id_child;
        copy.id_parent = id_parent;
        copy.params = try self.params.clone(try copy.arenaAlloc(gpa));
        copy.type = self.type;
        copy.tokens = try self.tokens.clone();
        copy.id_slot = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != c.LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed +%= @as(u32, @truncate(self.child_tasks.items.len)) +% 1;
        }

        try self.child_tasks.append(gpa, copy);
    }

    /// Ports `is_parent` (server-task.h:334 @8642541b2).
    pub fn isParent(self: *const Task) bool {
        return self.child_tasks.items.len > 0;
    }

    /// Ports `is_child` (server-task.h:338 @8642541b2).
    pub fn isChild(self: *const Task) bool {
        return self.id_parent != -1;
    }

    /// Ports `is_stateless` (server-task.h:343 @8642541b2): embedding and
    /// rerank tasks always process their whole prompt, their state is never
    /// reused.
    pub fn isStateless(self: *const Task) bool {
        return self.type == .embedding or self.type == .rerank;
    }
};

// -----------------------------------------------------------------------------
// result_prompt_progress, completion_token_output

/// Ports `result_prompt_progress` (server-task.h:348 @8642541b2).
pub const ResultPromptProgress = struct {
    total: i32 = 0,
    cache: i32 = 0,
    processed: i32 = 0,
    time_ms: i64 = 0,

    /// Ports `result_prompt_progress`'s `to_json` (server-task.cpp:234 @8642541b2).
    pub fn toJson(self: ResultPromptProgress, a: Allocator) Error!Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("total", self.total);
        try o.put("cache", self.cache);
        try o.put("processed", self.processed);
        try o.put("time_ms", self.time_ms);
        return o.done();
    }
};

/// Ports `completion_token_output` (server-task.h:385 @8642541b2): one
/// generated token, its probability and its candidates'.
pub const CompletionTokenOutput = struct {
    /// Ports `completion_token_output`'s `prob_info` (server-task.h:389 @8642541b2).
    pub const ProbInfo = struct {
        tok: Token = 0,
        txt: []const u8 = "",
        prob: f32 = 0,
    };

    tok: Token = 0,
    prob: f32 = 0,
    text_to_send: []const u8 = "",

    probs: []const ProbInfo = &.{},

    /// A deep copy into `a`.
    pub fn clone(self: *const CompletionTokenOutput, a: Allocator) Error!CompletionTokenOutput {
        var o = self.*;
        o.text_to_send = try a.dupe(u8, self.text_to_send);
        const ps = try a.alloc(ProbInfo, self.probs.len);
        for (self.probs, 0..) |p, i| ps[i] = .{ .tok = p.tok, .txt = try a.dupe(u8, p.txt), .prob = p.prob };
        o.probs = ps;
        return o;
    }

    /// Ports `completion_token_output`'s `to_json` (server-task.cpp:256 @8642541b2):
    /// the candidates, each with its text cut to whole characters.
    ///
    /// Return: owned by `a`.
    pub fn toJson(self: *const CompletionTokenOutput, a: Allocator, post_sampling_probs: bool) Error!Value {
        var probs_for_token = Value.emptyArray();
        errdefer probs_for_token.deinit(a);
        for (self.probs) |p| {
            const txt = p.txt[0..common.validateUtf8(p.txt)];
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("id", p.tok);
            try o.put("token", txt);
            try o.put("bytes", strToBytes(p.txt));
            try o.put(if (post_sampling_probs) "prob" else "logprob", if (post_sampling_probs) p.prob else logarithm(p.prob));
            try probs_for_token.pushBack(a, o.done());
        }
        return probs_for_token;
    }

    /// Ports `probs_vector_to_json` (server-task.cpp:274 @8642541b2), a member of `completion_token_output`.
    ///
    /// Return: owned by `a`.
    pub fn probsVectorToJson(a: Allocator, probs: []const CompletionTokenOutput, post_sampling_probs: bool) Error!Value {
        var out = Value.emptyArray();
        errdefer out.deinit(a);
        for (probs) |*p| {
            const txt = p.text_to_send[0..common.validateUtf8(p.text_to_send)];
            var o = Obj.init(a);
            errdefer o.deinit();
            try o.put("id", p.tok);
            try o.put("token", txt);
            try o.put("bytes", strToBytes(p.text_to_send));
            try o.put(if (post_sampling_probs) "prob" else "logprob", if (post_sampling_probs) p.prob else logarithm(p.prob));
            try o.put(if (post_sampling_probs) "top_probs" else "top_logprobs", try p.toJson(a, post_sampling_probs));
            try out.pushBack(a, o.done());
        }
        return out;
    }

    /// Ports `logarithm` (server-task.cpp:296 @8642541b2), a member of
    /// `completion_token_output`: the JSON library writes -inf as null, so
    /// log(0) is the lowest float instead.
    pub fn logarithm(x: f32) f32 {
        // the JSON library converts -inf to null, so we need to prevent that
        return if (x == 0.0) -std.math.floatMax(f32) else logf(x);
    }

    /// Ports `str_to_bytes` (server-task.cpp:301 @8642541b2), a member of `completion_token_output`.
    pub fn strToBytes(str: []const u8) common.Bytes {
        return .{ .s = str };
    }
};

extern fn logf(x: f32) f32;

// -----------------------------------------------------------------------------
// the prompt cache

/// Ports `common_prompt_checkpoint` (common/common.h:1293 @8642541b2), the
/// members the prompt cache reads. Its buffers are the owner's allocator's.
pub const PromptCheckpoint = struct {
    n_tokens: i64 = 0,

    // (optional) id of the task that created the checkpoint
    id_task: c_int = -1,

    pos_min: c.llama_pos = 0,
    pos_max: c.llama_pos = 0,

    data_tgt: []u8 = &.{},
    data_dft: []u8 = &.{},

    // (optional) speculative-decoding implementation state stashed with the checkpoint
    data_spec: []u8 = &.{},

    /// Ports `common_prompt_checkpoint`'s `size` (common/common.cpp:2385 @8642541b2).
    pub fn size(self: *const PromptCheckpoint) usize {
        return self.data_tgt.len + self.data_dft.len + self.data_spec.len;
    }

    /// Ports `common_prompt_checkpoint`'s `empty` (common/common.cpp:2389 @8642541b2).
    pub fn empty(self: *const PromptCheckpoint) bool {
        return self.data_tgt.len == 0;
    }

    /// Ports `common_prompt_checkpoint`'s `update_pos` (common/common.cpp:2404 @8642541b2).
    pub fn updatePos(self: *PromptCheckpoint, n_tokens: i64, pos_min: c.llama_pos, pos_max: c.llama_pos) void {
        self.n_tokens = n_tokens;
        self.pos_min = pos_min;
        self.pos_max = pos_max;
    }

    /// Ports `common_prompt_checkpoint`'s `clear` (common/common.cpp:2393 @8642541b2).
    pub fn clear(self: *PromptCheckpoint, a: Allocator) void {
        self.n_tokens = 0;
        self.pos_min = 0;
        self.pos_max = 0;
        a.free(self.data_tgt);
        a.free(self.data_dft);
        a.free(self.data_spec);
        self.data_tgt = &.{};
        self.data_dft = &.{};
        self.data_spec = &.{};
    }

    /// `common_prompt_checkpoint::update_tgt` and `update_dft`: the
    /// sequence's state into `buf`, resized to fit.
    fn updateBuf(buf: *[]u8, a: Allocator, ctx: ?*c.llama_context, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags, comptime line: c_int) Error!void {
        if (ctx == null) return;

        const ckpt_size = c.llama_state_seq_get_size_ext(ctx, seq_id, flags);

        buf.* = try a.realloc(buf.*, ckpt_size);

        const n = c.llama_state_seq_get_data_ext(ctx, buf.ptr, ckpt_size, seq_id, flags);
        if (n != ckpt_size) {
            _ = c.ggml_abort("common.cpp", line, "checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
        }
    }

    /// Ports `common_prompt_checkpoint`'s `update_tgt` (common/common.cpp:2413 @8642541b2).
    pub fn updateTgt(self: *PromptCheckpoint, a: Allocator, ctx: ?*c.llama_context, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) Error!void {
        try updateBuf(&self.data_tgt, a, ctx, seq_id, flags, 2427);
    }

    /// Ports `common_prompt_checkpoint`'s `update_dft` (common/common.cpp:2431 @8642541b2).
    pub fn updateDft(self: *PromptCheckpoint, a: Allocator, ctx: ?*c.llama_context, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) Error!void {
        try updateBuf(&self.data_dft, a, ctx, seq_id, flags, 2445);
    }

    fn loadBuf(buf: []const u8, ctx: ?*c.llama_context, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) bool {
        if (ctx == null) return true;
        if (buf.len == 0) return true;
        const n = c.llama_state_seq_set_data_ext(ctx, buf.ptr, buf.len, seq_id, flags);
        return n == buf.len;
    }

    /// Ports `common_prompt_checkpoint`'s `load_tgt` (common/common.cpp:2449 @8642541b2):
    /// false if the state could not be restored.
    pub fn loadTgt(self: *const PromptCheckpoint, ctx: ?*c.llama_context, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) bool {
        return loadBuf(self.data_tgt, ctx, seq_id, flags);
    }

    /// Ports `common_prompt_checkpoint`'s `load_dft` (common/common.cpp:2465 @8642541b2).
    pub fn loadDft(self: *const PromptCheckpoint, ctx: ?*c.llama_context, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) bool {
        return loadBuf(self.data_dft, ctx, seq_id, flags);
    }

    /// Ports `common_prompt_checkpoint`'s `clear_dft` (common/common.cpp:2485 @8642541b2).
    pub fn clearDft(self: *PromptCheckpoint, a: Allocator) void {
        a.free(self.data_dft);
        a.free(self.data_spec);
        self.data_dft = &.{};
        self.data_spec = &.{};
    }

    pub fn deinitWith(self: *PromptCheckpoint, a: Allocator) void {
        self.deinit(a);
    }

    fn clone(self: *const PromptCheckpoint, a: Allocator) Error!PromptCheckpoint {
        var p = self.*;
        p.data_tgt = try a.dupe(u8, self.data_tgt);
        p.data_dft = try a.dupe(u8, self.data_dft);
        p.data_spec = try a.dupe(u8, self.data_spec);
        return p;
    }

    fn deinit(self: *PromptCheckpoint, a: Allocator) void {
        a.free(self.data_tgt);
        a.free(self.data_dft);
        a.free(self.data_spec);
    }
};

/// Ports `server_prompt` (server-task.h:686 @8642541b2): a prompt's tokens and
/// its checkpoints (`std::list`, kept in order).
pub const ServerPrompt = struct {
    tokens: ServerTokens,
    checkpoints: std.ArrayList(PromptCheckpoint) = .empty,

    /// Ports `clear` (server-task.h:691 @8642541b2).
    pub fn clear(self: *ServerPrompt) void {
        self.tokens.clear();
        for (self.checkpoints.items) |*ck| ck.deinit(self.tokens.gpa);
        self.checkpoints.clearRetainingCapacity();
    }

    /// Ports `n_tokens` (server-task.h:696 @8642541b2).
    pub fn nTokens(self: *const ServerPrompt) c_int {
        return @truncate(@as(i64, @intCast(self.tokens.size())));
    }

    /// Ports `clone` (server-task.h:700 @8642541b2).
    pub fn clone(self: *const ServerPrompt) Error!ServerPrompt {
        var out: ServerPrompt = .{ .tokens = try self.tokens.clone() };
        errdefer out.deinit();
        for (self.checkpoints.items) |*ck| try out.checkpoints.append(out.tokens.gpa, try ck.clone(out.tokens.gpa));
        return out;
    }

    pub fn deinit(self: *ServerPrompt) void {
        const gpa = self.tokens.gpa;
        for (self.checkpoints.items) |*ck| ck.deinit(gpa);
        self.checkpoints.deinit(gpa);
        self.tokens.deinit();
    }
};

/// Ports `server_prompt_data` (server-task.h:708 @8642541b2).
pub const PromptData = struct {
    main: []u8 = &.{},
    drft: []u8 = &.{},

    /// Ports `size` (server-task.h:712 @8642541b2).
    pub fn size(self: *const PromptData) usize {
        return self.main.len + self.drft.len;
    }
};

/// Ports `server_prompt_cache_state` (server-task.h:717 @8642541b2).
pub const PromptCacheState = struct {
    prompt: ServerPrompt,
    data: PromptData = .{},

    /// Ports `size` (server-task.h:712 @8642541b2).
    pub fn size(self: *const PromptCacheState) usize {
        var res = self.data.size();

        for (self.prompt.checkpoints.items) |*ckpt| {
            res += ckpt.size();
        }

        return res;
    }

    fn deinit(self: *PromptCacheState) void {
        const gpa = self.prompt.tokens.gpa;
        gpa.free(self.data.main);
        gpa.free(self.data.drft);
        self.prompt.deinit();
    }
};

const mib: f64 = 1024.0 * 1024.0;

/// Ports `server_prompt_cache` (server-task.h:732 @8642541b2): prompts and
/// their saved states, oldest first, under a byte and a token limit.
pub const PromptCache = struct {
    gpa: Allocator,
    /// `std::list`: each state heap-allocated, so `alloc`'s pointer stays
    /// valid while others come and go.
    states: std.ArrayList(*PromptCacheState) = .empty,

    // in bytes, 0 = no limit
    limit_size: usize = 0,

    // in tokens, 0 = no limit
    limit_tokens: usize = 0,

    /// Ports the constructor of `server_prompt_cache` (server-task.h:732 @8642541b2).
    pub fn init(gpa: Allocator, limit_size_mib: i32, limit_tokens: usize) PromptCache {
        return .{
            .gpa = gpa,
            .limit_size = 1024 * 1024 * @as(usize, @intCast(if (limit_size_mib < 0) 0 else limit_size_mib)),
            .limit_tokens = limit_tokens,
        };
    }

    pub fn deinit(self: *PromptCache) void {
        for (self.states.items) |s| self.destroyState(s);
        self.states.deinit(self.gpa);
    }

    fn destroyState(self: *PromptCache, s: *PromptCacheState) void {
        s.deinit();
        self.gpa.destroy(s);
    }

    fn popFront(self: *PromptCache) void {
        const s = self.states.orderedRemove(0);
        self.destroyState(s);
    }

    /// Ports `server_prompt_cache`'s `size` (server-task.cpp:1736 @8642541b2).
    pub fn size(self: *const PromptCache) usize {
        var res: usize = 0;

        for (self.states.items) |state| {
            res += state.size();
        }

        return res;
    }

    /// Ports `server_prompt_cache`'s `n_tokens` (server-task.cpp:1746 @8642541b2).
    pub fn nTokens(self: *const PromptCache) usize {
        var res: usize = 0;

        for (self.states.items) |state| {
            res +%= @as(usize, @bitCast(@as(isize, state.prompt.nTokens())));
        }

        return res;
    }

    /// Ports `server_prompt_cache`'s `alloc` (server-task.cpp:1756 @8642541b2):
    /// room for a prompt's state, after dropping prompts it contains and
    /// the oldest ones over the size limit.
    ///
    /// Return: the new state, owned by the cache; null when the prompt is
    /// already cached, or too big, or memory ran out.
    pub fn alloc(self: *PromptCache, prompt: *const ServerPrompt, state_size_tgt: usize, state_size_dft: usize) Error!?*PromptCacheState {
        // first check if the current state is contained fully in the cache
        for (self.states.items) |s| {
            const cur_lcp_len: c_int = @truncate(@as(i64, @intCast(try s.prompt.tokens.getCommonPrefix(&prompt.tokens))));

            if (cur_lcp_len == @as(c_int, @truncate(@as(i64, @intCast(prompt.tokens.size()))))) {
                common.srv(.trc, "alloc", "%s", .{@as([*:0]const u8, " - prompt is already in the cache, skipping\n")});
                return null;
            }
        }

        // calculate checkpoints size to see if it will fit with the prompt
        var checkpoints_size: usize = 0;
        for (prompt.checkpoints.items) |*ckpt| {
            checkpoints_size += ckpt.size();
        }

        const state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

        // skip over-limit entries to avoid disturbing the cache
        if (self.limit_size > 0 and state_size_new > self.limit_size) {
            common.srv(.wrn, "alloc", " - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n", .{ @as(f64, @floatFromInt(state_size_new)) / mib, @as(f64, @floatFromInt(self.limit_size)) / mib });
            return null;
        }

        // remove any cached prompts that are fully contained in the current prompt
        var i: usize = 0;
        while (i < self.states.items.len) {
            const s = self.states.items[i];
            const len: c_int = @truncate(@as(i64, @intCast(try s.prompt.tokens.getCommonPrefix(&prompt.tokens))));

            if (len == @as(c_int, @truncate(@as(i64, @intCast(s.prompt.tokens.size()))))) {
                common.srv(.trc, "alloc", " - removing obsolete cached prompt with length %d\n", .{len});

                _ = self.states.orderedRemove(i);
                self.destroyState(s);
            } else {
                i += 1;
            }
        }

        if (self.limit_size > 0) {
            // make room before allocating the new vectors to avoid breaching the limit
            while (self.states.items.len != 0 and self.size() + state_size_new > self.limit_size) {
                common.srv(.wrn, "alloc", " - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n", .{@as(f64, @floatFromInt(self.states.items[0].size())) / mib});

                self.popFront();
            }
        }

        // check if we can allocate enough memory for the new state
        const state_data_tgt = self.gpa.alloc(u8, state_size_tgt) catch {
            return self.allocFailed();
        };
        const state_data_dft = self.gpa.alloc(u8, state_size_dft) catch {
            self.gpa.free(state_data_tgt);
            return self.allocFailed();
        };
        @memset(state_data_tgt, 0);
        @memset(state_data_dft, 0);

        const st = self.gpa.create(PromptCacheState) catch |e| {
            self.gpa.free(state_data_tgt);
            self.gpa.free(state_data_dft);
            return e;
        };
        st.* = .{ .prompt = prompt.clone() catch |e| {
            self.gpa.destroy(st);
            self.gpa.free(state_data_tgt);
            self.gpa.free(state_data_dft);
            return e;
        }, .data = .{ .main = state_data_tgt, .drft = state_data_dft } };
        self.states.append(self.gpa, st) catch |e| {
            self.destroyState(st);
            return e;
        };

        return st;
    }

    /// The `std::bad_alloc` arm of `alloc`: shrink the limit to 40% of what
    /// is cached and trim.
    fn allocFailed(self: *PromptCache) ?*PromptCacheState {
        common.srv(.err, "alloc", "failed to allocate memory for prompt cache state: %s\n", .{@as([*:0]const u8, "std::bad_alloc")});

        self.limit_size = @max(1, @as(usize, @intFromFloat(0.4 * @as(f64, @floatFromInt(self.size())))));

        common.srv(.wrn, "alloc", " - cache size limit reduced to %.3f MiB\n", .{@as(f64, @floatFromInt(self.limit_size)) / mib});

        self.update();

        return null;
    }

    /// Ports `server_prompt_cache`'s `load` (server-task.cpp:1838 @8642541b2): the
    /// cached prompt most like `tokens_new` that keeps the most of itself,
    /// restored into `id_slot` and moved into `prompt`.
    ///
    /// Return: false when a state failed to restore.
    pub fn load(self: *PromptCache, prompt: *ServerPrompt, tokens_new: *const ServerTokens, ctx_tgt: ?*c.llama_context, ctx_dft: ?*c.llama_context, id_slot: i32) Error!bool {
        const lcp_best: c_int = @truncate(@as(i64, @intCast(try prompt.tokens.getCommonPrefix(tokens_new))));

        var f_keep_best: f32 = if (prompt.tokens.size() > 0) @as(f32, @floatFromInt(lcp_best)) / @as(f32, @floatFromInt(prompt.tokens.size())) else -1.0; // empty slot: any cache entry wins
        var f_sim_best: f32 = @as(f32, @floatFromInt(lcp_best)) / @as(f32, @floatFromInt(tokens_new.size()));

        common.srv(.trc, "load", " - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", .{ @as(f64, f_keep_best), @as(f64, f_sim_best) });

        var it_best: ?usize = null;

        // find the most similar cached prompt, that would also preserve the most context
        for (self.states.items, 0..) |s, idx| {
            const lcp_cur: c_int = @truncate(@as(i64, @intCast(try s.prompt.tokens.getCommonPrefix(tokens_new))));

            const f_keep_cur = @as(f32, @floatFromInt(lcp_cur)) / @as(f32, @floatFromInt(s.prompt.tokens.size()));
            const f_sim_cur = @as(f32, @floatFromInt(lcp_cur)) / @as(f32, @floatFromInt(tokens_new.size()));

            common.srv(.trc, "load", "   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", .{ s.prompt.tokens.size(), lcp_cur, @as(f64, f_keep_cur), @as(f64, f_sim_cur) });

            // don't trash large prompts
            if (f_keep_cur < 0.25) {
                continue;
            }

            if (f_keep_best < f_keep_cur and f_sim_best < f_sim_cur) {
                f_keep_best = f_keep_cur;
                f_sim_best = f_sim_cur;

                it_best = idx;
            }
        }

        if (it_best) |bi| {
            const best = self.states.items[bi];
            common.srv(.trc, "load", " - found better prompt with f_keep = %.3f, f_sim = %.3f\n", .{ @as(f64, f_keep_best), @as(f64, f_sim_best) });

            {
                const data = best.data.main;

                const sz = data.len;
                const n = c.llama_state_seq_set_data_ext(ctx_tgt, data.ptr, sz, id_slot, 0);
                if (n != sz) {
                    common.srv(.err, "load", "failed to restore state with size %zu\n", .{sz});

                    return false;
                }

                self.gpa.free(best.data.main);
                best.data.main = &.{};
            }

            {
                const data = best.data.drft;

                if (data.len != 0) {
                    common.ggmlAssert(ctx_dft != null, "server-task.cpp", 1892, "ctx_dft");

                    const sz = data.len;
                    const n = c.llama_state_seq_set_data_ext(ctx_dft, data.ptr, sz, id_slot, 0);
                    if (n != sz) {
                        common.srv(.wrn, "load", "failed to restore state with size %zu\n", .{sz});

                        return false;
                    }

                    self.gpa.free(best.data.drft);
                    best.data.drft = &.{};
                }
            }

            prompt.deinit();
            prompt.* = best.prompt;
            best.prompt = .{ .tokens = .{ .gpa = self.gpa } };

            _ = self.states.orderedRemove(bi);
            self.destroyState(best);
        }

        return true;
    }

    /// Ports `server_prompt_cache`'s `update` (server-task.cpp:1915 @8642541b2):
    /// drops the oldest prompts over the size limit, then over the token
    /// limit -- raised to what the size limit holds at the average size per
    /// token.
    pub fn update(self: *PromptCache) void {
        if (self.limit_size > 0) {
            while (self.states.items.len != 0 and self.size() > self.limit_size) {
                common.srv(.wrn, "update", " - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", .{@as(f64, @floatFromInt(self.states.items[0].size())) / mib});

                self.popFront();
            }
        }

        // average size per token
        const size_per_token: f32 = @max(1.0, @as(f32, @floatFromInt(self.size())) / @as(f32, @floatFromInt(@max(1, self.nTokens()))));

        // dynamically increase the token limit if it can fit in the memory limit
        const limit_tokens_cur: usize = if (self.limit_size > 0) @max(self.limit_tokens, @as(usize, @intFromFloat(@as(f32, @floatFromInt(self.limit_size)) / size_per_token))) else self.limit_tokens;

        if (self.limit_tokens > 0) {
            while (self.states.items.len != 0 and self.nTokens() > limit_tokens_cur) {
                common.srv(.wrn, "update", " - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n", .{ self.limit_tokens, limit_tokens_cur, @as(f64, @floatFromInt(self.states.items[0].size())) / mib });

                self.popFront();
            }
        }

        common.srv(.trc, "update", " - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n", .{ self.states.items.len, @as(f64, @floatFromInt(self.size())) / mib, @as(f64, @floatFromInt(self.limit_size)) / mib, self.limit_tokens, limit_tokens_cur });

        for (self.states.items) |state| {
            common.srv(.trc, "update", "   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n", .{ @as(*const anyopaque, state), state.prompt.nTokens(), state.prompt.checkpoints.items.len, @as(f64, @floatFromInt(state.size())) / mib });
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a task's child copies its parameters with a new seed" {
    const a = std.testing.allocator;
    var t = Task.initType(.completion);
    defer t.deinit(a);
    t.params.sampling.seed = 7;
    t.params.antiprompt = &.{"x"};
    try t.addChild(a, 1, 2);
    try std.testing.expectEqual(@as(u32, 8), t.child_tasks.items[0].params.sampling.seed);
    try std.testing.expectEqualStrings("x", t.child_tasks.items[0].params.antiprompt[0]);
}
