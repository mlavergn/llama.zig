//! The server's core: the slots, the batch loop that drives them, and the
//! tasks the HTTP threads hand it -- `server_context`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-context.cpp` and
//! `llama.cpp/tools/server/server-context.h` (llama.cpp v0.8.0,
//! `8642541b2`) up to the public `server_context` and `get_meta`:
//! `server_output_limits`, `server_sample_and_accept_synth`, `slot_state`,
//! `server_batch`, `server_slot`, `process_mtmd_chunk`,
//! `server_context_impl` and `server_context`. The routes and the response
//! generator are `routes.zig`. Each declaration names what it ports and the
//! line it began at.
//!
//! # Threads
//!
//! As upstream: the decode loop runs on the thread that calls
//! `startLoop` (the main thread), the HTTP handlers on cpp-httplib's pool,
//! and the two meet only through `queue.zig`'s queues. `yield_to_queue`
//! lets the queue's worker thread answer metrics and slot queries while a
//! decode runs here.
//!
//! # Exceptions
//!
//! A C++ throw is an error; `whatOf` reads the text from whichever buffer
//! the thrower set (`common_chat`'s for the data layer, `cli/json`'s for
//! nlohmann, `tools_common.arg`'s for the sampler and the drafters).
//!
//! # Chat templates
//!
//! One template object, `common_chat.templates`' (Stage 9 step 4), as
//! upstream's `common_chat_templates`: parsed with the Jinja engine at
//! startup whether or not `--jinja` is on, for its capabilities and source
//! (what `/props` reports), and rendered by `oaicompat.chatTemplatesApply`
//! through Jinja or, with `--no-jinja`, through libllama's built-in
//! families. Its log lines (`LOG_DBG` and friends in `common/chat.cpp`, the
//! auto-parser and the per-model parsers) reach `common_log` through
//! `chatLogSink`.

const std = @import("std");
const tc = @import("tools_common");
const json = @import("json");
const chat = @import("common_chat");
const mtmd_mod = @import("mtmd");
const libcxx = @import("llama").libcxx;

const common = @import("common.zig");
const tokens_mod = @import("tokens.zig");
const task_mod = @import("task.zig");
const result_mod = @import("result.zig");
const queue_mod = @import("queue.zig");
const oaicompat = @import("oaicompat.zig");
const decision_mod = @import("decision.zig");
const convert = @import("convert.zig");
const realtime = @import("realtime.zig");

const c = tc.c;
const log = tc.log;
const sampling = tc.sampling;
const spec_mod = tc.speculative;
const Params = tc.params.Params;
const LoraInfo = tc.params.LoraInfo;

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Token = common.Token;
const Task = task_mod.Task;
const Result = result_mod.Result;
const ServerTokens = tokens_mod.ServerTokens;
const ServerPrompt = task_mod.ServerPrompt;
const PromptCheckpoint = task_mod.PromptCheckpoint;
const CompletionTokenOutput = task_mod.CompletionTokenOutput;
const ErrorType = common.ErrorType;

pub const TaskQueue = queue_mod.Queue(Task);
pub const ResultQueue = queue_mod.Response(*Result);
pub const Reader = queue_mod.ResponseReader(Task, *Result, task_mod.TaskResultState);

/// Every allocation that crosses threads: tasks, results, slot state.
pub const gpa = std.heap.c_allocator;

const token_null = common.token_null;

/// Ports `HTTP_POLLING_SECONDS` (server-context.cpp:46 @8642541b2).
pub const http_polling_seconds: c_int = 1;

// -----------------------------------------------------------------------------
// logging, assertions, exceptions

/// `GGML_ASSERT` with this file's name.
fn ggmlAssert(ok: bool, comptime line: c_int, comptime cond: [*:0]const u8) void {
    common.ggmlAssert(ok, "server-context.cpp", line, cond);
}

/// `GGML_ABORT(msg)`.
fn ggmlAbort(comptime line: c_int, comptime fmt: [*:0]const u8, args: anytype) noreturn {
    _ = @call(.auto, c.ggml_abort, .{ "server-context.cpp", line, fmt } ++ args);
    unreachable;
}

const srv = common.srv;

/// Ports `SLT_DBG`, `SLT_TRC`, `SLT_INF`, `SLT_WRN` and `SLT_ERR`
/// (server-common.h:28, 29, 30, 31, 32 @8642541b2): the message prefixed
/// with the calling function's name, the slot's id and its task's.
fn slt(comptime level: common.SrvLevel, comptime func: []const u8, slot: *const Slot, comptime fmt: []const u8, args: anytype) void {
    const f = "slot %12.*s: id %2d | task %d | " ++ fmt;
    const pre = .{ @as(c_int, 12), @as([*:0]const u8, func ++ ""), @as(c_int, slot.id), @as(c_int, if (slot.task) |*t| t.id else -1) };
    switch (level) {
        .dbg => log.dbg(f, pre ++ args),
        .trc => log.trc(f, pre ++ args),
        .inf => log.inf(f, pre ++ args),
        .wrn => log.wrn(f, pre ++ args),
        .err => log.err(f, pre ++ args),
    }
}

/// Whether a `LOG_*` line at `verbosity` prints: the macros evaluate their
/// arguments only then, and some arguments are expensive or throw.
fn logOn(verbosity: c_int) bool {
    return verbosity <= log.verbosity_thold;
}

/// A NUL-terminated copy for a `%s`, freed by the caller with `freeZ`.
const dupZ = common.dupZ;
const freeZ = common.freeZ;

/// `common/chat.cpp`'s `LOG_DBG`, `LOG_INF`, `LOG_WRN` and `LOG_ERR`, which
/// `cli/common_chat/log.zig` hands its host as finished lines: into
/// `common_log` at the macro's level and verbosity.
fn chatLogSink(level: chat.log.Level, text: []const u8) void {
    const z = dupZ(text);
    defer freeZ(z);
    switch (level) {
        .debug => log.dbg("%s", .{z}),
        .info => log.inf("%s", .{z}),
        .warn => log.wrn("%s", .{z}),
        .err => log.err("%s", .{z}),
    }
}

/// `chat.h`'s message helpers' `LOG_WRN` and `LOG_ERR` (`msg.zig`'s own
/// sink), likewise.
fn chatMsgLogSink(level: chat.msg.Level, text: []const u8) void {
    chatLogSink(if (level == .warn) .warn else .err, text);
}

/// Whether `LOG_DBG` prints at the current verbosity.
fn chatDebugOn() bool {
    return logOn(log.level_debug);
}

/// `common_chat_verify_template(tmpl, true)`, for `arg.zig`'s parse.
fn verifyJinjaTemplate(_: std.mem.Allocator, tmpl: []const u8) anyerror!bool {
    return chat.templates.verifyTemplate(gpa, tmpl, true);
}

/// Wires `cli/common_chat/` into the process, before the arguments are
/// parsed: its log lines into `common_log`, as upstream's `common/chat.cpp`
/// logs, and the Jinja arm of `common_chat_verify_template`, which
/// `--chat-template` runs at parse time.
pub fn installChat() void {
    chat.log.sink = chatLogSink;
    chat.log.debug_on = chatDebugOn;
    chat.msg.log_sink = chatMsgLogSink;
    tc.arg.verify_jinja_template = verifyJinjaTemplate;
}

/// `e.what()` for an error from any of the libraries the core calls: the
/// data layer's (`common_chat`'s buffer), nlohmann's, and `tools_common`'s
/// (`arg.what()`, the sampler's and the drafters').
pub fn whatOf(e: anyerror) []const u8 {
    return switch (e) {
        error.JsonError => json.what(),
        error.OutOfMemory => "std::bad_alloc",
        error.InvalidArgument, error.RuntimeError => common.what(e),
        error.Runtime, error.Throw => tc.arg.what(),
        else => @errorName(e),
    };
}

// -----------------------------------------------------------------------------
// free functions

/// Ports `server_output_limits` (server-context.cpp:48 @8642541b2): the
/// context's output buffer, sized for every slot verifying a full draft.
fn serverOutputLimits(params: *const Params) spec_mod.OutputLimits {
    if (params.mmproj.path.len != 0) {
        const path = dupZ(params.mmproj.path);
        defer freeZ(path);
        const mcaps = c.mtmd_get_cap_from_file(path);
        if (mcaps.gen_audio) {
            // some TTS models decode with embeddings output for the whole batch
            return .{ .total = params.n_batch, .per_seq = params.n_batch };
        }
    }

    if (params.embedding or
        (params.pooling_type != c.LLAMA_POOLING_TYPE_UNSPECIFIED and params.pooling_type != c.LLAMA_POOLING_TYPE_NONE))
    {
        return .{ .total = params.n_batch, .per_seq = 1 };
    }

    var result = spec_mod.getOutputLimits(params.n_batch, params.n_parallel, spec_mod.nMaxParams(&params.speculative));

    result.total = @max(1, result.total);
    result.per_seq = @max(1, result.per_seq);
    return result;
}

/// Ports `server_accept_replay` (server-context.cpp:71 @8642541b2): a
/// checkpoint restore dropped tokens the target had accepted -- re-accept
/// them rather than verify again.
///
/// Return: owned by `a`.
fn acceptReplay(
    a: Allocator,
    smpl: *sampling.Sampler,
    ctx: ?*c.llama_context,
    idxs: []const i32,
    draft: []const Token,
) ![]Token {
    ggmlAssert(idxs.len == draft.len + 1, 76, "idxs.size() == draft.size() + 1");

    var result: std.ArrayList(Token) = .empty;
    errdefer result.deinit(a);
    try result.ensureTotalCapacity(a, idxs.len);

    for (draft, 0..) |d, i| {
        // the token is discarded - the call is what advances the sampler over this position
        _ = try smpl.sample(ctx, idxs[i], false);
        smpl.accept(d, true);
        result.appendAssumeCapacity(d);
    }

    const id = try smpl.sample(ctx, idxs[draft.len], false);
    smpl.accept(id, true);
    result.appendAssumeCapacity(id);

    return result.toOwnedSlice(a);
}

/// Ports `server_sample_and_accept_synth` (server-context.cpp:97 @8642541b2):
/// synthetic draft verification for benchmarking -- accept draft tokens at
/// random instead of by match with the target; on replay the draft was
/// already accepted before a context checkpoint restore, so repeat the same
/// decisions.
///
/// Return: owned by `a`.
fn sampleAndAcceptSynth(
    a: Allocator,
    smpl: *sampling.Sampler,
    ctx: ?*c.llama_context,
    idxs: []const i32,
    draft: []const Token,
    synth_probs: []const f64,
    rng: *libcxx.Mt19937,
    is_replay: bool,
) ![]Token {
    ggmlAssert(idxs.len == draft.len + 1, 105, "idxs.size() == draft.size() + 1");
    ggmlAssert(synth_probs.len >= draft.len, 106, "synth_probs.size() >= draft.size()");

    var result: std.ArrayList(Token) = .empty;
    errdefer result.deinit(a);
    try result.ensureTotalCapacity(a, idxs.len);

    const vocab = c.llama_model_get_vocab(c.llama_get_model(ctx));
    for (draft, 0..) |d, i| {
        const id = try smpl.sample(ctx, idxs[i], false);
        const accept = is_replay or libcxx.uniformF64(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept and (is_replay or !c.llama_vocab_is_eog(vocab, d))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const is_replay_target = is_replay and i + 1 == draft.len;
            smpl.accept(d, is_replay_target);
            result.appendAssumeCapacity(d);
            continue;
        }

        smpl.accept(id, true);
        result.appendAssumeCapacity(id);
        return result.toOwnedSlice(a);
    }

    const id = try smpl.sample(ctx, idxs[draft.len], false);
    smpl.accept(id, true);
    result.appendAssumeCapacity(id);

    return result.toOwnedSlice(a);
}

/// Ports `slot_state` (server-context.cpp:140 @8642541b2). State diagram:
/// https://github.com/ggml-org/llama.cpp/pull/9283
pub const SlotState = enum {
    idle,
    wait_other, // after assigning a task, but waiting for parent slot to process prompt
    started, // after assigning a task and about to process prompt
    processing_prompt,
    done_prompt,
    generating,
};

// -----------------------------------------------------------------------------
// server_batch

/// Ports `server_batch` (server-context.cpp:151 @8642541b2): the entries of
/// one iteration, tokens and embeddings, rendered a sub-batch at a time into
/// `view`.
pub const Batch = struct {
    /// Ports `server_batch`'s `token` (server-context.cpp:154 @8642541b2).
    pub const Tok = struct {
        id_slot: i32,
        token: Token,
        pos: [c.GGML_MROPE_SECTIONS]c.llama_pos, // only pos[0] is used for text tokens
        output: bool,
        is_prompt: bool, // for stats tracking
        decision_order: i32 = 0,
        i_embd: i32 = -1, // row in embd, -1 if this is a token
    };

    view: tc.common.Batch = tc.common.Batch.initEmpty(gpa), // the rendered sub-batch [off, off + n_tokens), see render()

    tokens: std.ArrayList(Tok) = .empty,
    n_tokens_alloc: i32 = 0,
    n_embd: i32 = 0,

    // track if given slot can be batched with slots already in the batch
    slot_batched: ?*Slot = null,

    // embedding entries, they can be mixed with tokens if the context supports it
    embd: std.ArrayList(f32) = .empty, // n_embd per row

    alora_scale: f32 = -1.0,
    alora_disabled_id: usize = 0,

    /// The destructor of `server_batch`'s members.
    fn deinit(self: *Batch) void {
        self.view.deinit();
        self.tokens.deinit(gpa);
        self.embd.deinit(gpa);
    }

    /// Ports `server_batch`'s `init` (server-context.cpp:176 @8642541b2).
    fn init(self: *Batch, ctx: ?*c.llama_context, n_tokens_alloc: i32, n_embd: i32) !void {
        self.n_tokens_alloc = n_tokens_alloc;
        self.n_embd = n_embd;
        self.view.deinit();
        self.view = tc.common.Batch.init(gpa, ctx);
        try self.tokens.ensureTotalCapacity(gpa, @intCast(n_tokens_alloc));
    }

    /// Ports `server_batch`'s `add` (server-context.cpp:183 @8642541b2), a token.
    fn add(self: *Batch, id_slot: i32, token: Token, pos: c.llama_pos, output: bool, is_prompt: bool) !bool {
        if (@as(i32, @intCast(self.tokens.items.len)) >= self.n_tokens_alloc) {
            return false;
        }
        try self.tokens.append(gpa, .{ .id_slot = id_slot, .token = token, .pos = .{ pos, 0, 0, 0 }, .output = output, .is_prompt = is_prompt });
        return true;
    }

    /// Ports `server_batch`'s `add_embd` (server-context.cpp:192 @8642541b2):
    /// `embd_in` has `n_embd` values, `pos` has GGML_MROPE_SECTIONS values.
    fn addEmbd(self: *Batch, id_slot: i32, embd_in: [*]const f32, pos: *const [c.GGML_MROPE_SECTIONS]c.llama_pos, output: bool, is_prompt: bool) !bool {
        if (@as(i32, @intCast(self.tokens.items.len)) >= self.n_tokens_alloc) {
            return false;
        }
        const n_embd: usize = @intCast(self.n_embd);
        try self.tokens.append(gpa, .{ .id_slot = id_slot, .token = token_null, .pos = pos.*, .output = output, .is_prompt = is_prompt });
        self.tokens.items[self.tokens.items.len - 1].i_embd = @intCast(self.embd.items.len / n_embd);
        try self.embd.appendSlice(gpa, embd_in[0..n_embd]);
        return true;
    }

    /// Ports `server_batch`'s `has_embd` (server-context.cpp:202 @8642541b2).
    fn hasEmbd(self: *const Batch) bool {
        return self.embd.items.len != 0;
    }

    /// Ports `server_batch`'s `clear` (server-context.cpp:206 @8642541b2).
    fn clear(self: *Batch) void {
        self.tokens.clearRetainingCapacity();
        self.embd.clearRetainingCapacity();
        self.view.clear();
        self.slot_batched = null;
        self.alora_scale = -1.0;
        self.alora_disabled_id = 0;
    }

    /// Ports `server_batch`'s `size` (server-context.cpp:215 @8642541b2).
    fn size(self: *const Batch) i32 {
        return @intCast(self.tokens.items.len);
    }

    /// Ports `server_batch`'s `truncate` (server-context.cpp:220 @8642541b2):
    /// remove the entries after the first `n`.
    fn truncate(self: *Batch, n: i32) void {
        ggmlAssert(n >= 0 and n <= self.size(), 221, "n >= 0 && n <= size()");
        var i = n;
        while (i < self.size()) : (i += 1) {
            const t = &self.tokens.items[@intCast(i)];
            if (t.i_embd >= 0) {
                self.embd.shrinkRetainingCapacity(@as(usize, @intCast(t.i_embd)) * @as(usize, @intCast(self.n_embd)));
                break;
            }
        }
        self.tokens.shrinkRetainingCapacity(@intCast(n));
    }

    /// Ports `server_batch`'s `set_output` (server-context.cpp:231 @8642541b2).
    fn setOutput(self: *Batch, idx: i32, output: bool) void {
        ggmlAssert(idx >= 0 and idx < self.size(), 232, "idx >= 0 && idx < (int32_t)tokens.size()");
        self.tokens.items[@intCast(idx)].output = output;
    }

    /// Ports `server_batch`'s `set_decision_order` (server-context.cpp:236 @8642541b2).
    fn setDecisionOrder(self: *Batch, idx: i32, order: i32) void {
        ggmlAssert(idx >= 0 and idx < self.size(), 237, "idx >= 0 && idx < (int32_t)tokens.size()");
        self.tokens.items[@intCast(idx)].decision_order = order;
    }

    /// Ports `server_batch`'s `render` (server-context.cpp:242 @8642541b2):
    /// the sub-batch `[off, off + n_tokens)` into `view`; index `i` in `view`
    /// is index `off + i` here.
    fn render(self: *Batch, off: i32, n_tokens: i32) void {
        ggmlAssert(off >= 0 and off < self.size(), 243, "off >= 0 && off < size()");
        ggmlAssert(n_tokens > 0 and off + n_tokens <= self.size(), 244, "n_tokens > 0 && off + n_tokens <= size()");

        const n_embd: usize = @intCast(self.n_embd);
        self.view.clear();
        var i = off;
        while (i < off + n_tokens) : (i += 1) {
            const t = &self.tokens.items[@intCast(i)];
            if (t.i_embd >= 0) {
                const embd: c.llama_embd = .{ .data = self.embd.items.ptr + @as(usize, @intCast(t.i_embd)) * n_embd, .n_rows = 1, .n_embd = n_embd };
                _ = self.view.addEmbd(embd, &t.pos, t.id_slot, t.output);
            } else {
                _ = self.view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            self.view.tokens.items[self.view.tokens.items.len - 1].decision_order = t.decision_order;
        }
    }
};

// -----------------------------------------------------------------------------
// server_slot

/// `spec_draft_q.clear()`: the inner vectors freed too.
fn clearDraftQ(q: *spec_mod.Candidates) void {
    spec_mod.truncateCandidates(q, 0);
}

/// Ports `server_slot`'s `tts_ctx` (server-context.cpp:271 @8642541b2): the
/// generation state of a TTS slot.
const TtsCtx = struct {
    ctx: ?*c.mtmd_helper_gen_audio = null,
    h_state_prompt: std.ArrayList(f32) = .empty, // the first h_state after step_prompt done
    h_state: ?[*]const f32 = null,
    sampled: Token = token_null,
    n_decoded: i32 = 0,
    stop: bool = false,

    fn isSupported(self: *const TtsCtx) bool {
        return self.ctx != null;
    }

    fn reset(self: *TtsCtx) void {
        c.mtmd_helper_gen_audio_reset(self.ctx);
        self.h_state_prompt.clearRetainingCapacity();
        self.h_state = null;
        self.sampled = token_null;
        self.n_decoded = 0;
        self.stop = false;
    }

    fn deinit(self: *TtsCtx) void {
        self.h_state_prompt.deinit(gpa);
        if (self.ctx) |x| c.mtmd_helper_gen_audio_free(x);
        self.ctx = null;
    }
};

/// Ports `server_slot` (server-context.cpp:259 @8642541b2): one client's
/// sequence -- its task, its prompt in the KV cache, its sampler.
pub const Slot = struct {
    id: c_int = 0,

    ctx_tgt: ?*c.llama_context = null,
    ctx_dft: ?*c.llama_context = null,

    mem: tc.common.Memory = .{},

    // multimodal
    mctx: ?*c.mtmd_context = null,
    mbatch: ?*c.mtmd_batch = null,

    tts: TtsCtx = .{},

    // speculative decoding
    spec: ?*spec_mod.Speculative = null,

    spec_draft: std.ArrayList(Token) = .empty,

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    spec_draft_q: spec_mod.Candidates = .empty,
    spec_prompt: []Token = &.{},
    spec_i_batch: std.ArrayList(i32) = .empty,
    spec_ckpt: PromptCheckpoint = .{},
    spec_is_replay: bool = false,
    spec_synth_rng: libcxx.Mt19937 = libcxx.Mt19937.init(libcxx.Mt19937.default_seed),

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    task: ?Task = null,
    task_prev: ?Task = null, // used for debugging

    // used to determine the slot that has been used the longest
    t_last_used: i64 = -1,

    // generation props
    n_ctx: i32 = 0, // context size per slot
    n_keep: i32 = 0,
    i_batch: i32 = -1,

    // effective generation limit for the current task, -1 means unlimited
    n_predict_max: i32 = -1,

    last_nl_pos: usize = 0,

    generated: chat.input.Input = .{ .gpa = gpa },
    debug_generated_text: std.ArrayList(u8) = .empty,
    generated_tokens: std.ArrayList(Token) = .empty,
    n_sent_text: usize = 0, // number of sent text character (i.e. handle partial UTF-8 on streaming)

    generated_token_probs: std.ArrayList(CompletionTokenOutput) = .empty,
    /// Owns the strings in `generated_token_probs`; reset with them.
    probs_arena: std.heap.ArenaAllocator = std.heap.ArenaAllocator.init(std.heap.c_allocator),

    has_next_token: bool = true,
    has_new_line: bool = false,
    truncated: bool = false,

    stop: task_mod.StopType = .none,

    stopping_word: std.ArrayList(u8) = .empty,

    // state
    state: SlotState = .idle,

    prompt: ServerPrompt = .{ .tokens = .{} },

    lora: []LoraInfo = &.{},
    alora_invocation_start: i32 = -1,

    // sampling
    json_schema: Value = .null,

    smpl: ?*sampling.Sampler = null,

    sampled: Token = 0, // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    // for realtime transcription, these are the rows from the last audio stream step (size = n_rows * n_embd)
    inp_embd: std.ArrayList(f32) = .empty,

    // realtime transcription, see SERVER_TASK_TYPE_TRANSCRIBE_RT
    audio_stream: ?*c.mtmd_audio_stream = null,
    audio_wait: bool = false, // the stream needs more audio before it can produce rows
    audio_commit_rows: std.ArrayList(i64) = .empty, // end row of each non-final commit not yet completed

    // realtime conversation, see SERVER_TASK_TYPE_REALTIME
    rt: ?*realtime.RealtimeSlot = null,

    stats: common.SlotStats = .{},

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    n_accepted_per_pos: std.ArrayList(u64) = .empty,

    /// `callback_on_release` and `callback_on_reset`: both bound to the
    /// server context.
    owner: ?*Impl = null,

    // this is for printing timings with slot progress, not part of metrics
    t_print_last: i64 = 0,
    n_gen_last: i32 = 0,

    fn deinit(self: *Slot) void {
        if (self.task) |*t| t.deinit(gpa);
        if (self.task_prev) |*t| t.deinit(gpa);
        self.spec_draft.deinit(gpa);
        clearDraftQ(&self.spec_draft_q);
        self.spec_draft_q.deinit(gpa);
        gpa.free(self.spec_prompt);
        self.spec_i_batch.deinit(gpa);
        self.spec_ckpt.clear(gpa);
        self.generated.deinit();
        self.debug_generated_text.deinit(gpa);
        self.generated_tokens.deinit(gpa);
        self.generated_token_probs.deinit(gpa);
        self.probs_arena.deinit();
        self.stopping_word.deinit(gpa);
        self.prompt.deinit();
        gpa.free(self.lora);
        self.json_schema.deinit(gpa);
        if (self.smpl) |s| s.destroy();
        self.inp_embd.deinit(gpa);
        self.n_accepted_per_pos.deinit(gpa);
        if (self.mbatch) |b| c.mtmd_batch_free(b);
        self.tts.deinit();
        if (self.audio_stream) |st| c.mtmd_audio_stream_free(st);
        self.audio_commit_rows.deinit(gpa);
        if (self.rt) |r| r.destroy();
    }

    fn taskRef(self: *const Slot) *const Task {
        return &self.task.?;
    }

    /// Ports `server_slot`'s `prompt_save` (server-context.cpp:343 @8642541b2).
    fn promptSave(self: *const Slot, prompt_cache: *task_mod.PromptCache) !bool {
        if (self.prompt.tokens.size() == 0) {
            return false;
        }

        const cur_size_tgt = c.llama_state_seq_get_size_ext(self.ctx_tgt, self.id, c.LLAMA_STATE_SEQ_FLAGS_NONE);
        const cur_size_dft: usize = if (self.ctx_dft != null) c.llama_state_seq_get_size_ext(self.ctx_dft, self.id, c.LLAMA_STATE_SEQ_FLAGS_NONE) else 0;

        const cur_size = cur_size_tgt + cur_size_dft;

        srv(.trc, "prompt_save", " - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n", .{
            @as(c_int, @intCast(self.prompt.tokens.size())),
            @as(f64, @floatFromInt(cur_size)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(cur_size_dft)) / (1024.0 * 1024.0),
        });

        const cur = (try prompt_cache.alloc(&self.prompt, cur_size_tgt, cur_size_dft)) orelse return false;

        _ = c.llama_state_seq_get_data_ext(self.ctx_tgt, cur.data.main.ptr, cur_size_tgt, self.id, c.LLAMA_STATE_SEQ_FLAGS_NONE);
        if (self.ctx_dft != null) {
            _ = c.llama_state_seq_get_data_ext(self.ctx_dft, cur.data.drft.ptr, cur_size_dft, self.id, c.LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    /// Ports `server_slot`'s `prompt_load` (server-context.cpp:369 @8642541b2).
    fn promptLoad(self: *Slot, prompt_cache: *task_mod.PromptCache, tokens: *const ServerTokens) !bool {
        const res = try prompt_cache.load(&self.prompt, tokens, self.ctx_tgt, self.ctx_dft, self.id);
        if (!res) {
            slt(.wrn, "prompt_load", self, "%s", .{@as([*:0]const u8, "failed to load prompt from cache\n")});
        }

        return res;
    }

    /// Ports `server_slot`'s `prompt_clear` (server-context.cpp:378 @8642541b2).
    fn promptClear(self: *Slot) void {
        slt(.trc, "prompt_clear", self, "clearing prompt with %zu tokens\n", .{self.prompt.tokens.size()});

        self.mem.seqRm(self.id, -1, -1);

        self.prompt.clear();
    }

    /// Ports `server_slot`'s `reset` (server-context.cpp:422 @8642541b2).
    fn reset(self: *Slot) void {
        slt(.dbg, "reset", self, "%s", .{@as([*:0]const u8, "\n")});

        self.spec_is_replay = false;

        self.last_nl_pos = 0;
        self.generated.deinit();
        self.generated = .init(gpa);
        self.has_new_line = false;
        self.truncated = false;
        self.stop = .none;
        self.stopping_word.clearRetainingCapacity();
        self.n_sent_text = 0;

        if (self.canSpeculate()) {
            self.spec_draft.clearRetainingCapacity();
            self.spec_i_batch.clearRetainingCapacity();
            self.spec_ckpt.clear(gpa);
        }
        self.generated_tokens.clearRetainingCapacity();
        self.generated_token_probs.clearRetainingCapacity();
        _ = self.probs_arena.reset(.retain_capacity);
        self.json_schema.deinit(gpa);
        self.json_schema = .null;

        if (self.task_prev) |*t| t.deinit(gpa);
        self.task_prev = self.task;
        self.task = null;

        // note: callback_on_reset() must have run before this, see release()
        self.stats = .{};
        self.n_accepted_per_pos.clearRetainingCapacity();

        self.n_predict_max = -1;

        _ = c.llama_set_sampler(self.ctx_tgt, self.id, null);

        // clear alora start
        self.alora_invocation_start = -1;

        // clear multimodal state
        if (self.mbatch) |b| c.mtmd_batch_free(b);
        self.mbatch = null;

        self.tts.reset();

        if (self.audio_stream) |st| c.mtmd_audio_stream_free(st);
        self.audio_stream = null;
        self.audio_wait = false;
        self.audio_commit_rows.clearRetainingCapacity();
        self.inp_embd.clearRetainingCapacity();
        if (self.rt) |r| r.destroy();
        self.rt = null;
    }

    /// Ports `server_slot`'s `init_sampler` (server-context.cpp:470 @8642541b2).
    fn initSampler(self: *const Slot) void {
        if (self.smpl) |s| s.reset();

        if (!self.taskRef().needSampling()) {
            return;
        }

        const t_start = c.ggml_time_us();

        var n_text: c_int = 0;

        for (self.prompt.tokens.tokens.items) |id| {
            if (id != token_null) {
                if (self.smpl) |s| s.accept(id, false);
                n_text += 1;
            }
        }

        slt(.trc, "init_sampler", self, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n", .{
            @as(f64, @floatFromInt(c.ggml_time_us() - t_start)) / 1000.0, n_text, @as(c_int, @intCast(self.prompt.tokens.size())),
        });
    }

    /// Ports `server_slot`'s `need_embd` (server-context.cpp:494 @8642541b2).
    fn needEmbd(self: *const Slot) bool {
        ggmlAssert(self.task != null, 495, "task");
        return self.taskRef().needEmbd();
    }

    /// Ports `server_slot`'s `can_split` (server-context.cpp:499 @8642541b2).
    fn canSplit(self: *const Slot) bool {
        ggmlAssert(self.task != null, 500, "task");
        // a decision over a non-causal context reads the whole prompt in one batch
        if (self.taskRef().type == .decision and !c.llama_get_causal_attn(self.ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!self.taskRef().needEmbd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (c.llama_get_memory(self.ctx_tgt) == null) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const pooling = c.llama_pooling_type(self.ctx_tgt);
        if (pooling == c.LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == c.LLAMA_POOLING_TYPE_RANK and c.llama_get_causal_attn(self.ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (self.taskRef().type == .decision) {
            return true;
        }
        return false;
    }

    /// Ports `server_slot`'s `can_batch_with` (server-context.cpp:529 @8642541b2).
    fn canBatchWith(self: *const Slot, other_slot: *const Slot) bool {
        ggmlAssert(self.task != null, 530, "task");

        // a joint decision head reads the whole batch
        if (self.taskRef().decision.order.items.len != 0 or other_slot.taskRef().decision.order.items.len != 0) {
            return false;
        }

        // lfm3 d1 has no memory, its prompt is read as one flat ubatch and the short convs would mix the sequences
        // TODO: batch them along the sequence dimension, this needs ubatches of sequences with unequal lengths
        if (self.taskRef().decision.no_batching or other_slot.taskRef().decision.no_batching) {
            return false;
        }

        return self.taskRef().type == other_slot.taskRef().type and
            self.inp_embd.items.len == other_slot.inp_embd.items.len and
            common.areLoraEqual(self.lora, other_slot.lora);
    }

    /// Ports `server_slot`'s `n_remaining` (server-context.cpp:549 @8642541b2):
    /// -1 if the generation is limitless.
    fn nRemaining(self: *const Slot) i32 {
        return if (self.n_predict_max == -1) -1 else self.n_predict_max - @as(i32, @truncate(@as(i64, @bitCast(self.stats.n_gen))));
    }

    /// Ports `server_slot`'s `has_budget` (server-context.cpp:553 @8642541b2).
    fn hasBudget(self: *const Slot) bool {
        return self.n_predict_max == -1 or self.nRemaining() > 0;
    }

    /// Ports `server_slot`'s `is_processing` (server-context.cpp:557 @8642541b2).
    pub fn isProcessing(self: *const Slot) bool {
        return self.state != .idle;
    }

    /// Ports `server_slot`'s `can_speculate` (server-context.cpp:561 @8642541b2).
    fn canSpeculate(self: *const Slot) bool {
        return self.spec != null;
    }

    /// Ports `server_slot`'s `use_spec_rejection` (server-context.cpp:566
    /// @8642541b2): at temp 0 both p and q are point masses, so rejection is
    /// the same as sample-and-match.
    fn useSpecRejection(self: *const Slot) bool {
        return self.task != null and self.taskRef().params.sampling.temp > 0.0;
    }

    /// Ports `server_slot`'s `add_token` (server-context.cpp:570 @8642541b2).
    fn addToken(self: *Slot, token: *const CompletionTokenOutput) !void {
        if (!self.isProcessing()) {
            slt(.wrn, "add_token", self, "%s", .{@as([*:0]const u8, "slot is not processing\n")});
            return;
        }

        try self.generated_token_probs.append(gpa, try token.clone(self.probs_arena.allocator()));
    }

    /// Ports `server_slot`'s `get_n_draft_max` (server-context.cpp:579 @8642541b2).
    fn getNDraftMax(self: *const Slot) i32 {
        ggmlAssert(self.task != null, 580, "task");

        if (!self.canSpeculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        var n_draft_max = self.n_ctx - self.prompt.nTokens() - 2;

        if (self.nRemaining() > 0) {
            n_draft_max = @min(n_draft_max, self.nRemaining() - 1);
        }

        slt(.dbg, "get_n_draft_max", self, "max possible draft: %d\n", .{n_draft_max});

        return n_draft_max;
    }

    /// Ports `server_slot`'s `handle_last_sampled_token` (server-context.cpp:601
    /// @8642541b2): the sampled token of this slot into the batch, and the
    /// speculative draft tokens if any.
    fn handleLastSampledToken(self: *Slot, batch: *Batch) !void {
        const func = "handle_last_sampled_token";
        var add_ok = true;
        // the audio stream has no token before the first step, track the position with a dummy token
        const tok_track: Token = if (self.sampled == token_null) 0 else self.sampled;
        if (self.spec_draft.items.len == 0) {
            // no speculative decoding
            self.i_batch = batch.size();

            if (self.inp_embd.items.len != 0) {
                const n_rows: i32 = @divTrunc(@as(i32, @intCast(self.inp_embd.items.len)), batch.n_embd);
                ggmlAssert(@as(i32, @intCast(self.inp_embd.items.len)) == n_rows * batch.n_embd, 611, "(int32_t) inp_embd.size() == n_rows * batch.n_embd");
                self.i_batch = batch.size() + n_rows - 1;
                var i: i32 = 0;
                while (i < n_rows) : (i += 1) {
                    // text embeddings broadcast the same position across the M-RoPE sections
                    const p = self.prompt.tokens.posNext(-1);
                    const pos = [c.GGML_MROPE_SECTIONS]c.llama_pos{ p, p, p, 0 };
                    add_ok = (try batch.addEmbd(self.id, self.inp_embd.items.ptr + @as(usize, @intCast(i)) * @as(usize, @intCast(batch.n_embd)), &pos, i == n_rows - 1, false)) and add_ok;
                    if (i < n_rows - 1) {
                        try self.prompt.tokens.pushBackToken(tok_track);
                    }
                }
                if (self.audio_stream != null) {
                    self.inp_embd.clearRetainingCapacity(); // rows are consumed, the next step produces new ones
                }
            } else {
                add_ok = (try batch.add(self.id, self.sampled, self.prompt.tokens.posNext(-1), true, false)) and add_ok;
            }

            slt(.dbg, func, self, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n", .{
                self.sampled, self.n_ctx, self.prompt.nTokens(), @as(c_int, @intFromBool(self.truncated)),
            });
        } else {
            slt(.dbg, func, self, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n", .{
                self.sampled, self.prompt.tokens.size(), self.spec_draft.items.len, self.prompt.tokens.posNext(-1),
            });

            ggmlAssert(self.spec_i_batch.items.len == 0, 635, "spec_i_batch.empty()");

            try self.spec_i_batch.append(gpa, batch.size());
            for (0..self.spec_draft.items.len) |i| {
                try self.spec_i_batch.append(gpa, batch.size() + @as(i32, @intCast(i)) + 1);
            }

            var pos0 = self.prompt.tokens.posNext(-1);

            add_ok = (try batch.add(self.id, self.sampled, pos0, true, false)) and add_ok;
            pos0 += 1;
            for (self.spec_draft.items) |token| {
                add_ok = (try batch.add(self.id, token, pos0, true, false)) and add_ok;
                pos0 += 1;
            }
        }

        ggmlAssert(add_ok, 650, "add_ok && \"batch must be large enough to hold the sampled and draft tokens\"");

        try self.prompt.tokens.pushBackToken(tok_track);
        try self.prompt.tokens.insert(self.spec_draft.items);
    }

    /// Ports `server_slot`'s `release` (server-context.cpp:656 @8642541b2).
    fn release(self: *Slot) !void {
        if (self.isProcessing()) {
            ggmlAssert(self.task != null, 658, "task");

            slt(.inf, "release", self, "stop processing: n_tokens = %d, truncated = %d\n", .{ self.prompt.nTokens(), @as(c_int, @intFromBool(self.truncated)) });

            self.t_last_used = c.ggml_time_us();

            self.state = .idle;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (self.taskRef().isChild() or self.taskRef().isStateless() or self.audio_stream != null or self.taskRef().type == .realtime) {
                self.promptClear();
            }

            // mtmd_helper always re-eval the whole prompt
            if (self.taskRef().type == .tts) {
                self.promptClear();
            }

            // callback_on_reset: flush the generated token stats before reset()
            const owner = self.owner.?;
            if (self.stats.n_gen > 0) {
                try owner.metricsOnPrediction(self);
            }

            self.reset();

            // callback_on_release
            try owner.queue_tasks.popDeferredTask(self.id);
        }
    }

    /// Ports `server_slot`'s `find_stopping_strings` (server-context.cpp:685
    /// @8642541b2).
    ///
    /// Return: the position of the earliest stop word, or null.
    fn findStoppingStrings(self: *Slot, text: []const u8, last_token_size: usize, is_full_stop: bool) !?usize {
        ggmlAssert(self.task != null, 686, "task");

        var stop_pos: ?usize = null;

        for (self.taskRef().params.antiprompt) |word| {
            var pos: ?usize = undefined;

            if (is_full_stop) {
                const tmp = word.len + last_token_size;
                const from_pos = if (text.len > tmp) text.len - tmp else 0;

                pos = std.mem.indexOfPos(u8, text, from_pos, word);
            } else {
                // otherwise, partial stop
                pos = stringFindPartialStop(text, word);
            }

            if (pos != null and (stop_pos == null or pos.? < stop_pos.?)) {
                if (is_full_stop) {
                    self.stop = .word;
                    self.stopping_word.clearRetainingCapacity();
                    try self.stopping_word.appendSlice(gpa, word);
                    self.has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    /// Ports `server_slot`'s `print_timings_tg` (server-context.cpp:716 @8642541b2).
    fn printTimingsTg(self: *Slot) void {
        if (self.stats.n_gen < 100) {
            return;
        }

        const t_now = c.ggml_time_us();

        if (t_now - self.t_print_last < 3 * 1000 * 1000) {
            return;
        }

        const n_gen_second = self.stats.nGenTps();
        const n_gen_second_win = 1e6 / @as(f64, @floatFromInt(t_now - self.t_print_last)) * @as(f64, @floatFromInt(@as(i64, @intCast(self.stats.n_gen)) - self.n_gen_last));

        self.t_print_last = t_now;
        self.n_gen_last = @intCast(self.stats.n_gen);

        slt(.inf, "print_timings_tg", self, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", .{ @as(c_int, @intCast(self.stats.n_gen)), n_gen_second, n_gen_second_win });
    }

    /// Ports `server_slot`'s `print_timings_pp` (server-context.cpp:736 @8642541b2).
    fn printTimingsPp(self: *const Slot) void {
        const t_prompt_total = self.stats.tPromptMs();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const n_prompt_second = self.stats.nPromptTps();
        const n_task = self.taskRef().nTokens();
        const f_progress: f64 = if (n_task > 0) @as(f64, @floatFromInt(self.prompt.nTokens())) / @as(f64, @floatFromInt(n_task)) else 0.0;

        slt(.inf, "print_timings_pp", self, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n", .{
            @as(c_int, @intCast(self.stats.n_prompt_processed)), f_progress, t_prompt_total / 1e3, n_prompt_second,
        });
    }

    /// Ports `server_slot`'s `print_timings` (server-context.cpp:750 @8642541b2).
    fn printTimings(self: *const Slot) !void {
        const func = "print_timings";
        const t_prompt_total = self.stats.tPromptMs();
        const t_gen_total = self.stats.tGenMs();

        const t_prompt = self.stats.tPromptPerTokenMs();
        const n_prompt_second = self.stats.nPromptTps();

        const t_gen = self.stats.tGenPerTokenMs();
        const n_gen_second = self.stats.nGenTps();

        slt(.inf, func, self, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", .{
            t_prompt_total, @as(c_int, @intCast(self.stats.n_prompt_processed)), t_prompt, n_prompt_second,
        });

        slt(.inf, func, self, "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", .{
            t_gen_total, @as(c_int, @intCast(self.stats.n_gen)), t_gen, n_gen_second,
        });

        slt(.inf, func, self, "      total time = %10.2f ms / %5d tokens\n", .{
            t_prompt_total + t_gen_total, @as(c_int, @intCast(self.stats.n_prompt_processed + self.stats.n_gen)),
        });

        slt(.inf, func, self, "   graphs reused = %10d\n", .{c.llama_perf_context(self.ctx_tgt).n_reused});

        slt(.inf, func, self, "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n", .{ c.llama_perf_context(self.ctx_tgt).n_reallocs_same, c.llama_perf_context(self.ctx_tgt).n_reallocs_diff });

        const n_draft_total: i32 = @intCast(self.stats.n_draft_tokens);
        const n_draft_accepted: i32 = @intCast(self.stats.n_draft_accepted);
        const n_draft_verif_steps: i32 = @intCast(self.stats.n_draft_verif_steps);

        if (n_draft_total > 0) {
            const draft_ratio: f32 = @as(f32, @floatFromInt(n_draft_accepted)) / @as(f32, @floatFromInt(n_draft_total));
            const mean_acc_len: f64 = if (n_draft_verif_steps > 0) 1.0 + @as(f64, @floatFromInt(n_draft_accepted)) / @as(f64, @floatFromInt(n_draft_verif_steps)) else 1.0;

            var rates: std.ArrayList(u8) = .empty;
            defer rates.deinit(gpa);
            if (n_draft_verif_steps > 0) {
                for (self.n_accepted_per_pos.items, 0..) |n, i| {
                    if (i > 0) try rates.appendSlice(gpa, ", ");
                    var buf: [64]u8 = undefined;
                    const len = snprintf(&buf, buf.len, "%.3f", @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(n_draft_verif_steps)));
                    try rates.appendSlice(gpa, buf[0..@intCast(@min(len, buf.len - 1))]);
                }
            }
            try rates.append(gpa, 0);

            slt(.inf, func, self, "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n", .{
                @as(f64, draft_ratio), n_draft_accepted, n_draft_total, mean_acc_len,
            });
            slt(.trc, func, self, "     acc per pos = (%s)\n", .{@as([*:0]const u8, @ptrCast(rates.items.ptr))});
        }

        spec_mod.printStats(self.spec);
    }

    /// Ports `server_slot`'s `to_json` (server-context.cpp:808 @8642541b2).
    ///
    /// Return: owned by `a`.
    fn toJson(self: *const Slot, a: Allocator, only_metrics: bool) !Value {
        var res = common.Obj.init(a);
        errdefer res.deinit();

        try res.put("id", self.id);
        try res.put("n_ctx", self.n_ctx);
        try res.put("speculative", self.canSpeculate());
        try res.put("is_processing", self.isProcessing());

        const ptask: ?*const Task = if (self.task) |*t| t else if (self.task_prev) |*t| t else null;

        if (ptask) |pt| {
            try res.put("id_task", pt.id);
            try res.put("n_prompt_tokens", @as(i32, @intCast(self.prompt.tokens.size())));
            try res.put("n_prompt_tokens_processed", self.stats.n_prompt_processed);
            try res.put("n_prompt_tokens_cache", self.stats.n_prompt_cached);
            try res.put("params", try pt.params.toJson(a, only_metrics));
            var next = common.Obj.init(a);
            errdefer next.deinit();
            try next.put("has_next_token", self.has_next_token);
            try next.put("has_new_line", self.has_new_line);
            try next.put("n_remain", self.nRemaining());
            try next.put("n_decoded", self.stats.n_gen);
            var arr = Value.emptyArray();
            try arr.pushBack(a, next.done());
            try res.put("next_token", arr);

            if (!only_metrics) {
                try res.put("prompt", try pt.tokens.detokenize(a, self.ctx_tgt, true));
                try res.put("generated", if (self.generated.empty()) self.debug_generated_text.items else self.generated.text.items);
            }
        }

        return res.done();
    }

    /// Ports `server_slot`'s `copy_prompt_to` (server-context.cpp:845
    /// @8642541b2): the other slot continues from the tokens processed so far.
    fn copyPromptTo(self: *const Slot, other: *Slot) !void {
        self.mem.seqRm(other.id, -1, -1);
        self.mem.seqCp(self.id, other.id, -1, -1);

        var p = try self.prompt.clone();
        errdefer p.deinit();
        other.prompt.deinit();
        other.prompt = p;
    }

    /// Ports `server_slot`'s `copy_state_to` (server-context.cpp:852 @8642541b2).
    fn copyStateTo(self: *const Slot, other: *Slot) !void {
        ggmlAssert(self.state == .done_prompt, 853, "state == SLOT_STATE_DONE_PROMPT");

        self.mem.seqRm(other.id, -1, -1);
        self.mem.seqCp(self.id, other.id, -1, -1);

        other.i_batch = self.i_batch;

        other.stats = self.stats;

        var p = try self.prompt.clone();
        errdefer p.deinit();
        other.prompt.deinit();
        other.prompt = p;
        other.initSampler();
    }
};

extern fn snprintf(buf: [*]u8, n: usize, fmt: [*:0]const u8, ...) c_int;

/// Ports `string_find_partial_stop` (common/common.h:880 @8642541b2): where
/// a prefix of `stop` ends `str`.
fn stringFindPartialStop(str: []const u8, stop: []const u8) ?usize {
    if (str.len != 0 and stop.len != 0) {
        const max_len = @min(str.len, stop.len);
        const last_char = str[str.len - 1];
        var len = max_len;
        while (len > 0) : (len -= 1) {
            if (stop[len - 1] == last_char) {
                if (std.mem.endsWith(u8, str, stop[0..len])) {
                    return str.len - len;
                }
            }
        }
    }
    return null;
}

/// Ports `encode_mtmd_chunk` (server-context.cpp:869 @8642541b2): encode
/// the mtmd chunk at `idx`, batched with as many of the next chunks as
/// possible.
///
/// Return: 0 on success.
fn encodeMtmdChunk(slot: *const Slot, mbatch: *?*c.mtmd_batch, idx: usize) !i32 {
    const func = "encode_mtmd_chunk";
    const mctx = slot.mctx;
    const input_tokens = &slot.taskRef().tokens;
    const chunk = try input_tokens.findChunk(idx);

    if (mbatch.*) |b| c.mtmd_batch_free(b);
    mbatch.* = c.mtmd_batch_init(mctx);
    var res = c.mtmd_batch_add_chunk(mbatch.*, chunk);
    ggmlAssert(res == 0, 876, "res == 0"); // we should never have an empty batch

    // try batching as much as possible
    var n_added: c_int = 1;
    var idx_cur = idx;
    while (res == 0) {
        const next = input_tokens.findNextMediaChunk(idx_cur) orelse break;
        res = c.mtmd_batch_add_chunk(mbatch.*, next.chunk);
        n_added += if (res == 0) 1 else 0;
        idx_cur = next.idx;
        slt(.dbg, func, slot, "try adding media chunk idx = %zu to batch, res = %d\n", .{ next.idx, res });
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    slt(.trc, func, slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", .{ idx, n_added });

    res = c.mtmd_batch_encode(mbatch.*);
    if (res != 0) {
        slt(.err, func, slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", .{ idx, res });
        return -1;
    }

    return 0;
}

/// Ports `process_mtmd_chunk` (server-context.cpp:910 @8642541b2): the media
/// chunk at `idx` encoded -- with the media chunks after it batched in --
/// and decoded into the slot's sequence. The caller updates the prompt.
///
/// Return: 0 on success.
fn processMtmdChunk(slot: *const Slot, mbatch: *?*c.mtmd_batch, idx: usize, n_tokens_out: *usize) !i32 {
    ggmlAssert(slot.mctx != null, 911, "slot.mctx");
    const input_tokens = &slot.taskRef().tokens;
    const chunk = try input_tokens.findChunk(idx);
    var res: i32 = 0;

    const TryDecode = struct {
        const CbData = struct {
            spec: ?*spec_mod.Speculative,
            ctx: ?*c.llama_context,
        };

        fn cb(b_ptr: [*c]const c.mtmd_helper_embd_batch, user_data: ?*anyopaque) callconv(.c) i32 {
            const data: *const CbData = @ptrCast(@alignCast(user_data));
            const b = &b_ptr[0];

            var batch = tc.common.Batch.init(gpa, data.ctx);
            defer batch.deinit();
            const n_tokens: usize = @intCast(b.n_tokens);
            const n_embd: usize = @intCast(b.n_embd);
            for (0..n_tokens) |i| {
                var pos = [c.GGML_MROPE_SECTIONS]c.llama_pos{ 0, 0, 0, 0 };
                for (0..@intCast(b.n_pos)) |j| {
                    pos[j] = b.pos[j * n_tokens + i];
                }
                _ = batch.addEmbd(.{ .data = b.embd + i * n_embd, .n_rows = 1, .n_embd = n_embd }, &pos, b.seq_id, false);
            }

            return if (spec_mod.process(data.spec, &batch)) 0 else 1;
        }

        fn run(s: *const Slot, mb: *?*c.mtmd_batch, ch: *c.mtmd_input_chunk, i: usize, out: *usize) i32 {
            if (mb.*) |b| {
                const embd = c.mtmd_batch_get_output_embd(b, ch);
                if (embd != null) {
                    var cb_data: CbData = .{ .spec = s.spec, .ctx = s.ctx_tgt };
                    var new_n_past: c.llama_pos = 0; // unused for now
                    const r = c.mtmd_helper_decode_image_chunk(
                        s.mctx,
                        s.ctx_tgt,
                        ch,
                        embd,
                        s.prompt.tokens.posNext(-1),
                        s.id,
                        @intCast(c.llama_n_batch(s.ctx_tgt)),
                        &new_n_past,
                        cb,
                        &cb_data,
                    );
                    if (r != 0) {
                        slt(.err, "operator()", s, "failed to decode mtmd chunk, idx = %zu, res = %d\n", .{ i, r });
                        return -1;
                    }
                    out.* = c.mtmd_input_chunk_get_n_tokens(ch);
                    return 0; // success
                }
            }
            return 1; // (non-error) need to create & encode batch
        }
    };

    // if the batch is already exist, try searching & encode
    res = TryDecode.run(slot, mbatch, chunk, idx, n_tokens_out);
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (try encodeMtmdChunk(slot, mbatch, idx) != 0) {
        return -1;
    }

    return TryDecode.run(slot, mbatch, chunk, idx, n_tokens_out);
}

/// Ports `add_mtmd_chunk` (server-context.cpp:987 @8642541b2): an encoded
/// mtmd chunk added to the batch of the text tokens, instead of decoded on
/// its own like `processMtmdChunk`; `embd` is the encoder's output for it.
///
/// Return: false if the batch is full.
fn addMtmdChunk(slot: *const Slot, chunk: ?*const c.mtmd_input_chunk, embd: [*]const f32, batch: *Batch) !bool {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const image = c.mtmd_input_chunk_get_tokens_image(chunk);
    const is_mrope = c.mtmd_decode_use_mrope(slot.mctx);
    const n_tokens = c.mtmd_input_chunk_get_n_tokens(chunk);
    const n_embd: usize = @intCast(batch.n_embd);
    const pos_0 = slot.prompt.tokens.posNext(-1);

    for (0..n_tokens) |i| {
        const p: c.llama_pos = pos_0 + @as(c.llama_pos, @intCast(i));
        var pos = [c.GGML_MROPE_SECTIONS]c.llama_pos{ p, p, p, p };
        if (is_mrope and image != null) {
            const rel = c.mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = @bitCast(rel.t);
            pos[1] = @bitCast(rel.y);
            pos[2] = @bitCast(rel.x);
            pos[3] = @bitCast(rel.z);
        }
        if (!try batch.addEmbd(slot.id, embd + i * n_embd, &pos, slot.needEmbd(), true)) {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// server_state

/// Ports `server_state` (server-common.h:778 @8642541b2).
pub const ServerState = enum { downloading, measuring, loading, ready, sleeping };

/// Ports `server_state_to_str` (server-common.h:786 @8642541b2).
pub fn serverStateToStr(state: ServerState) []const u8 {
    return switch (state) {
        .downloading => "downloading",
        .measuring => "measuring",
        .loading => "loading",
        .ready => "ready",
        .sleeping => "sleeping",
    };
}

/// `server_state_callback_t` (server-context.h:61 @8642541b2): set only by
/// a child of router mode (step 7); the default is upstream's no-op lambda.
pub const StateCallback = struct {
    ctx: ?*anyopaque = null,
    func: ?*const fn (ctx: ?*anyopaque, state: ServerState, payload: *const Value) void = null,

    fn call(self: StateCallback, state: ServerState, payload: *const Value) void {
        if (self.func) |f| f(self.ctx, state, payload);
    }
};

/// Ports `server_context_meta` (server-context.h:16 @8642541b2): what the
/// routes read without touching the context, valid during sleep.
pub const Meta = struct {
    build_info: []const u8,
    model_name: []const u8,
    model_aliases: []const []const u8,
    model_tags: []const []const u8,
    model_path: []const u8,
    model_output_modalities: []const []const u8, // output modalities for GET /models
    has_mtmd: bool,
    has_inp_image: bool,
    has_inp_audio: bool,
    has_inp_video: bool,
    has_cap_chat: bool,
    has_cap_tts: bool,
    has_audio_stream: bool, // realtime transcription, see mtmd_support_audio_stream()
    json_ui_settings: *const Value,
    slot_n_ctx: c_int,
    pooling_type: c.enum_llama_pooling_type,

    // chat params
    chat_params: *const oaicompat.ServerChatParams,
    chat_template_caps: Value,
    chat_template: []const u8,
    chat_template_tool_use: []const u8,

    // tokens
    bos_token_str: []const u8,
    eos_token_str: []const u8,
    fim_pre_token: Token,
    fim_sub_token: Token,
    fim_mid_token: Token,
    fim_pad_token: Token,
    fim_rep_token: Token,
    fim_sep_token: Token,

    // sampling
    logit_bias_eog: []const c.llama_logit_bias,

    // model meta
    model_vocab_type: c.enum_llama_vocab_type,
    model_vocab_n_tokens: i32,
    model_n_ctx_train: i32,
    model_n_embd_inp: i32,
    model_n_params: u64,
    model_size: u64,
    model_ftype: []const u8,
};

// -----------------------------------------------------------------------------
// server_context_impl

/// Ports `load_progress_data` (server-context.cpp:1169 @8642541b2), a member
/// of `server_context_impl`.
const LoadProgress = struct {
    ctx: *Impl,
    stage: []const u8,
    stages: []const []const u8 = &.{},
    t_last_load_progress_ms: i64 = 0,
};

/// Ports `server_context_impl` (server-context.cpp:1016 @8642541b2).
pub const Impl = struct {
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    model_tgt: ?*c.llama_model = null,

    mctx: ?*c.mtmd_context = null,
    mtmd_cap: c.mtmd_helper_cap = std.mem.zeroes(c.mtmd_helper_cap),
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    init_opt: c.mtmd_helper_init_opt = undefined,
    vocab: ?*const c.llama_vocab = null,

    // prompt of the "asr" chat template, for live ASR
    asr_prompt: []Token = &.{},

    decision: decision_mod.DecisionContext = .{},

    queue_tasks: TaskQueue,
    queue_results: ResultQueue,

    // note: chat_params must not be refreshed upon existing sleeping state
    chat_params: oaicompat.ServerChatParams = undefined,
    /// The templates `chat_params.tmpls` points at (see the file's doc
    /// comment).
    tmpls: ?chat.templates.Templates = null,

    callback_state: StateCallback = .{},

    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    params_base: Params = undefined,

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    llama_init: ?tc.init.InitResult = null,

    ctx_tgt: ?*c.llama_context = null,

    batch: Batch = .{},

    model_dft: ?*c.llama_model = null,
    ctx_dft: ?*c.llama_context = null,

    spec_init: ?spec_mod.InitResult = null,

    ctx_tgt_seq_rm_type: tc.common.SeqRmType = .no,
    ctx_dft_seq_rm_type: tc.common.SeqRmType = .no,

    spec: ?*spec_mod.Speculative = null,

    add_bos_token: bool = true,

    n_ctx: i32 = 0, // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    n_swa: i32 = 0,

    // slots / clients
    slots: []Slot = &.{},

    trace: c_int = 0, // env: LLAMA_TRACE
    slots_debug: c_int = 0, // env: LLAMA_SERVER_SLOTS_DEBUG
    slots_n_diff: c_int = 0, // env: LLAMA_SERVER_SLOTS_N_DIFF

    n_empty_consecutive: c_int = 0,

    prompt_cache: ?*task_mod.PromptCache = null,

    metrics: common.Metrics = .{},

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    t_decode_start: i64 = 0, // start of the last submitted decode
    t_prompt_start: i64 = 0, // start of the oldest queued prompt decode
    n_prompt_queued: u64 = 0,

    json_ui_settings: Value = Value.emptyObject(),

    // Necessary similarity of prompt for slot selection
    slot_prompt_similarity: f32 = 0.0,

    model_name: []const u8 = "", // name of the loaded model, to be used by API
    model_aliases: []const []const u8 = &.{}, // additional names for the model
    model_tags: []const []const u8 = &.{}, // informational tags

    sleeping: bool = false,

    /// What the lifetime of the server owns: names, templates, settings.
    arena: std.heap.ArenaAllocator = std.heap.ArenaAllocator.init(std.heap.c_allocator),

    /// Ports `has_cap_tts` (server-context.cpp:1034 @8642541b2), a member of `server_context_impl`.
    pub fn hasCapTts(self: *const Impl) bool {
        return self.mtmd_cap.audio_gen;
    }

    /// Ports `has_cap_chat` (server-context.cpp:1038 @8642541b2), a member of `server_context_impl`.
    pub fn hasCapChat(self: *const Impl) bool {
        return self.mtmd_cap.chat;
    }

    /// Ports the constructor of `server_context_impl` (server-context.cpp:1052 @8642541b2).
    pub fn init(self: *Impl) !void {
        self.* = .{
            .queue_tasks = TaskQueue.init(gpa),
            .queue_results = try ResultQueue.init(gpa),
        };
        self.init_opt = c.mtmd_helper_init_opt_default();
        c.mtmd_helper_log_set(tc.common.defaultLogCallback, null);
    }

    /// Ports the destructor of `server_context_impl` (server-context.cpp:1056 @8642541b2).
    pub fn deinit(self: *Impl) void {
        if (!self.sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            self.destroy();
        }
        for (self.slots) |*s| s.deinit();
        gpa.free(self.slots);
        self.batch.deinit();
        self.decision.deinit();
        gpa.free(self.asr_prompt);
        if (self.tmpls) |*t| t.deinit();
        if (self.prompt_cache) |pc| {
            pc.deinit();
            gpa.destroy(pc);
        }
        self.metrics.deinit(gpa);
        self.queue_tasks.deinit();
        self.queue_results.deinit();
        self.arena.deinit();
    }

    /// Ports `server_context_impl`'s `get_metrics` (server-context.cpp:1064 @8642541b2).
    pub fn getMetrics(self: *const Impl, a: Allocator) !common.Metrics {
        return self.metrics.clone(a);
    }

    /// Ports `server_context_impl`'s `reset_metrics_bucket` (server-context.cpp:1068 @8642541b2).
    pub fn resetMetricsBucket(self: *Impl) void {
        self.metrics.resetBucket();
    }

    /// Ports `server_context_impl`'s `destroy` (server-context.cpp:1135 @8642541b2).
    fn destroy(self: *Impl) void {
        spec_mod.free(self.spec);
        self.spec = null;
        if (self.spec_init) |*si| si.deinit();
        self.spec_init = null;

        self.ctx_dft = null;
        self.model_dft = null;

        if (self.llama_init) |*li| li.deinit();
        self.llama_init = null;

        self.ctx_tgt = null;
        self.model_tgt = null;

        c.mtmd_free(self.mctx);
        self.mctx = null;
    }

    /// Ports `server_context_impl`'s `handle_sleeping_state` (server-context.cpp:1151 @8642541b2).
    fn handleSleepingState(self: *Impl, new_state: bool) !void {
        ggmlAssert(self.sleeping != new_state, 1152, "sleeping != new_state");
        if (new_state) {
            const empty: Value = .null;
            self.callback_state.call(.sleeping, &empty);
            // note: for sleeping == false, event is emitted by load_model()
            srv(.inf, "handle_sleeping_state", "%s", .{@as([*:0]const u8, "server is entering sleeping state\n")});
            self.destroy();
        } else {
            srv(.inf, "handle_sleeping_state", "%s", .{@as([*:0]const u8, "server is exiting sleeping state\n")});
            if (!try self.loadModel(&self.params_base)) {
                ggmlAbort(1163, "failed to reload model after sleeping", .{});
            }
        }
        self.sleeping = new_state;
    }

    /// Ports `server_context_impl`'s `load_progress_callback` (server-context.cpp:1176 @8642541b2).
    fn loadProgressCallback(progress: f32, user_data: ?*anyopaque) callconv(.c) bool {
        const d: *LoadProgress = @ptrCast(@alignCast(user_data.?));
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            const t_last = &d.t_last_load_progress_ms;
            const t_now = c.ggml_time_ms();
            const first = t_last.* == 0;
            const done = progress >= 1.0;
            const throttled = !first and !done and (t_now - t_last.*) < 200;
            if (throttled) {
                return true;
            }
            t_last.* = t_now;
        }
        if (d.ctx.callback_state.func != null) {
            var a = std.heap.ArenaAllocator.init(gpa);
            defer a.deinit();
            const ar = a.allocator();
            var o = common.Obj.init(ar);
            o.put("stages", d.stages) catch return true;
            o.put("current", d.stage) catch return true;
            o.put("value", progress) catch return true;
            const payload = o.done();
            d.ctx.callback_state.call(.loading, &payload);
        }
        return true;
    }

    /// Ports `server_context_impl`'s `load_model` (server-context.cpp:1203
    /// @8642541b2): load the model and initialize llama_context; this may
    /// also be called to resume from sleeping state.
    pub fn loadModel(self: *Impl, params: *Params) !bool {
        const func = "load_model";
        const a = self.arena.allocator();
        var load_progress_text: LoadProgress = .{ .ctx = self, .stage = "text_model" };
        var load_progress_mmproj: LoadProgress = .{ .ctx = self, .stage = "mmproj_model" };
        var load_progress_spec: LoadProgress = .{ .ctx = self, .stage = "spec_model" };

        const is_resume = self.sleeping;

        if (params != &self.params_base) self.params_base = params.*;
        const output_limits = serverOutputLimits(&self.params_base);
        self.params_base.n_outputs_max = output_limits.total;
        self.params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const has_mmproj = params.mmproj.path.len != 0;
        const has_draft = params.speculative.hasDft();
        const spec_mtp = for (self.params_base.speculative.types.items) |t| {
            if (t == .draft_mtp) break true;
        } else false;
        const has_spec = has_draft or spec_mtp;

        {
            var stages: std.ArrayList([]const u8) = .empty;
            try stages.append(a, "text_model");
            if (has_spec) try stages.append(a, "spec_model");
            if (has_mmproj) try stages.append(a, "mmproj_model");
            load_progress_text.stages = stages.items;
            load_progress_mmproj.stages = stages.items;
            load_progress_spec.stages = stages.items;

            // trigger 0% progress
            _ = loadProgressCallback(0.0, &load_progress_text);
        }

        {
            const name = dupZ(params.model.getName());
            defer freeZ(name);
            srv(.inf, func, "loading model '%s'\n", .{name});
        }
        if (logOn(log.level_trace)) {
            const path = dupZ(params.model.path);
            defer freeZ(path);
            srv(.trc, func, "local path '%s'\n", .{path});
        }

        const mmproj_path = try a.dupeSentinel(u8, self.params_base.mmproj.path, 0);
        var mparams = c.mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu = self.params_base.mmproj_use_gpu;
            mparams.device = self.params_base.mmproj_device;
            mparams.print_timings = false;
            mparams.n_threads = self.params_base.cpuparams.n_threads;
            mparams.flash_attn_type = self.params_base.flash_attn_type;
            mparams.warmup = self.params_base.warmup;
            mparams.image_min_tokens = self.params_base.image_min_tokens;
            mparams.image_max_tokens = self.params_base.image_max_tokens;
            mparams.batch_max_tokens = self.params_base.mtmd_batch_max_tokens;
            mparams.media_marker = common.getMediaMarker().ptr;
            // progress callback
            mparams.progress_callback = loadProgressCallback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        const MtmdParams = @typeInfo(@TypeOf(mtmd_mod.mtmd.getMemoryUsage)).@"fn".param_types[1].?;
        var mmproj_usage: mtmd_mod.mtmd.MemoryUsage = .{ .backend_mem_usage = .{}, .image_max_tokens = -1, .use_non_causal = false };
        defer mmproj_usage.deinit();
        var mmproj_usage_t_us: i64 = 0;
        if (has_mmproj) {
            const t_start = c.ggml_time_us();
            mmproj_usage.deinit();
            mmproj_usage = mtmd_mod.mtmd.getMemoryUsage(mmproj_path.ptr, @as(*const MtmdParams, @ptrCast(&mparams)).*);
            mmproj_usage_t_us = c.ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        // on resume params_base already holds the margin from the first load
        if (has_mmproj and self.params_base.fit_params and !is_resume) {
            try self.addMmprojMemory(&mmproj_usage, mmproj_usage_t_us);
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            self.params_base.load_progress_callback = loadProgressCallback;
            self.params_base.load_progress_callback_user_data = &load_progress_text;
        }

        if (self.llama_init) |*li| li.deinit();
        self.llama_init = try tc.init.initFromParams(gpa, &self.params_base, false);

        self.model_tgt = self.llama_init.?.model;
        self.ctx_tgt = self.llama_init.?.context;

        if (self.model_tgt == null) {
            const path = dupZ(self.params_base.model.path);
            defer freeZ(path);
            srv(.err, func, "failed to load model, '%s'\n", .{path});
            return false;
        }

        if (self.ctx_tgt == null) {
            const path = dupZ(self.params_base.model.path);
            defer freeZ(path);
            srv(.err, func, "failed to create_context with model '%s'\n", .{path});
            return false;
        }

        self.vocab = c.llama_model_get_vocab(self.model_tgt);

        self.decision.init(gpa, self.model_tgt) catch |e| {
            const w = dupZ(whatOf(e));
            defer freeZ(w);
            srv(.err, func, "failed to init decision model: %s\n", .{w});
            return false;
        };

        self.n_ctx = @intCast(c.llama_n_ctx(self.ctx_tgt));

        self.add_bos_token = c.llama_vocab_get_add_bos(self.vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            _ = loadProgressCallback(0.0, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0; // reset so internal cbs aren't delayed

            {
                var params_dft = spec_mod.baseParamsToSpeculative(&self.params_base);

                // progress callback
                params_dft.load_progress_callback = loadProgressCallback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                self.spec_init = try spec_mod.initFromParams(&params_dft, self.model_tgt, self.ctx_tgt);
                self.model_dft = self.spec_init.?.model();
                self.ctx_dft = self.spec_init.?.context();

                if (has_draft and self.model_dft == null) {
                    const path = dupZ(params_dft.model.path);
                    defer freeZ(path);
                    srv(.err, func, "failed to load draft model, '%s'\n", .{path});
                    return false;
                }

                if (self.ctx_dft == null) {
                    srv(.err, func, "%s", .{@as([*:0]const u8, "failed to create MTP context\n")});
                    return false;
                }

                self.params_base.speculative.draft.ctx_tgt = self.ctx_tgt;
                self.params_base.speculative.draft.ctx_dft = self.ctx_dft;
            }

            _ = loadProgressCallback(1.0, &load_progress_spec);
        }

        if (has_mmproj) {
            if (self.callback_state.func != null) {
                var o = common.Obj.init(a);
                try o.put("stage", "mmproj_model");
                const payload = o.done();
                self.callback_state.call(.loading, &payload);
            }

            if (!is_resume) {
                c.mtmd_helper_log_set(tc.common.defaultLogCallback, null);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const n_ubatch: c_int = @intCast(c.llama_n_ubatch(self.ctx_tgt));
                const is_mixed = self.useMixedBatch();
                if (is_mixed or mmproj_usage.use_non_causal) {
                    const n_max: c_int = if (is_mixed) @divTrunc(n_ubatch, 2) else n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        const why: [*:0]const u8 = if (is_mixed) "the model processes the prompt in one ubatch" else "model needs non-causal attention on image";
                        srv(.wrn, func, "cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", .{ mmproj_usage.image_max_tokens, n_max, n_ubatch, why });
                        srv(.wrn, func, "%s\n", .{@as([*:0]const u8, "increase n_ubatch (-ub) to increase vision token budget")});
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = @min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            self.mctx = c.mtmd_init_from_file(mmproj_path.ptr, self.model_tgt, mparams);
            if (self.mctx == null) {
                srv(.err, func, "failed to load multimodal model, '%s'\n", .{mmproj_path.ptr});
                return false;
            }
            srv(.inf, func, "loaded multimodal model, '%s'\n", .{mmproj_path.ptr});

            self.init_opt.video_params.fps_target = self.params_base.video_fps;
            self.init_opt.video_params.timestamp_interval_ms = self.params_base.video_timestamp_interval_ms;
            self.init_opt.video_params.ffmpeg_bin_dir = if (self.params_base.video_ffmpeg_bin_dir.len == 0)
                null
            else
                (try a.dupeSentinel(u8, self.params_base.video_ffmpeg_bin_dir, 0)).ptr;

            if (self.params_base.ctx_shift) {
                self.params_base.ctx_shift = false;
                srv(.wrn, func, "%s\n", .{@as([*:0]const u8, "ctx_shift is not supported by multimodal, it will be disabled")});
            }

            if (self.params_base.n_cache_reuse != 0) {
                self.params_base.n_cache_reuse = 0;
                srv(.wrn, func, "%s\n", .{@as([*:0]const u8, "cache_reuse is not supported by multimodal, it will be disabled")});
            }
        }

        self.mtmd_cap = c.mtmd_helper_get_capabilities(self.model_tgt, self.mctx);

        if (!c.llama_memory_can_shift(c.llama_get_memory(self.ctx_tgt))) {
            if (self.params_base.ctx_shift) {
                self.params_base.ctx_shift = false;
                srv(.wrn, func, "%s\n", .{@as([*:0]const u8, "ctx_shift is not supported by this context, it will be disabled")});
            }

            if (self.params_base.n_cache_reuse != 0) {
                self.params_base.n_cache_reuse = 0;
                srv(.wrn, func, "%s\n", .{@as([*:0]const u8, "cache_reuse is not supported by this context, it will be disabled")});
            }
        }

        if (c.llama_model_n_swa(self.model_tgt) == 0) {
            if (self.params_base.swa_full) {
                self.params_base.swa_full = false;
                srv(.wrn, func, "%s\n", .{@as([*:0]const u8, "swa_full is not supported by this model, it will be disabled")});
            }
        }

        self.n_swa = if (self.params_base.swa_full) 0 else c.llama_model_n_swa(self.model_tgt);

        // Necessary similarity of prompt for slot selection
        self.slot_prompt_similarity = self.params_base.slot_prompt_similarity;

        const n_ctx_train = c.llama_model_n_ctx_train(self.model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const n_ctx_seq: c_int = @intCast(c.llama_n_ctx_seq(self.ctx_tgt));
            const per_slot = self.params_base.kv_unified_per_slot;

            if (per_slot > 0) {
                if (n_ctx_seq > per_slot) {
                    srv(.inf, func, "capping per-slot context (%d) to --kv-unified-per-slot (%d)\n", .{ n_ctx_seq, per_slot });
                } else if (per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    srv(.wrn, func, "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, " ++
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to " ++
                        "n_parallel * kv_unified_per_slot)\n", .{ per_slot, n_ctx_seq, n_ctx_seq });
                }
            }

            const n_ctx_capped = if (per_slot > 0) @min(n_ctx_seq, per_slot) else n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                srv(.wrn, func, "the slot context (%d) exceeds the training context of the model (%d) - capping\n", .{ n_ctx_capped, n_ctx_train });
            }
        }

        for (self.slots) |*s| s.deinit();
        gpa.free(self.slots);
        self.slots = &.{};

        self.ctx_tgt_seq_rm_type = tc.common.contextCanSeqRm(self.ctx_tgt);
        if (self.ctx_tgt_seq_rm_type == .no) {
            srv(.wrn, func, "%s", .{@as([*:0]const u8, "speculative decoding not supported by this context\n")});
        }

        if (self.ctx_tgt_seq_rm_type == .full) {
            srv(.trc, func, "%s", .{@as([*:0]const u8, "speculative decoding will use checkpoints\n")});
        }

        // setup slots
        srv(.inf, func, "initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n", .{
            self.params_base.n_parallel, self.nCtxSlot(), @as([*:0]const u8, if (self.params_base.kv_unified) "true" else "false"),
        });

        // initialize slots
        self.slots = try gpa.alloc(Slot, @intCast(self.params_base.n_parallel));
        for (self.slots) |*s| s.* = .{};

        // try speculative decoding
        if (self.ctx_tgt_seq_rm_type != .no) {
            if (spec_mod.init(&self.params_base.speculative, @intCast(self.params_base.n_parallel))) |sp| {
                self.spec = sp;
            } else |e| {
                if (e == error.OutOfMemory) return e;
                const msg = dupZ(tc.arg.what());
                defer freeZ(msg);
                srv(.err, func, "failed to initialize speculative decoding context: %s\n", .{msg});
                if (self.params_base.speculative.hasSynth()) {
                    return false;
                }
            }
        }

        if (self.ctx_dft != null) {
            self.ctx_dft_seq_rm_type = tc.common.contextCanSeqRm(self.ctx_dft);
        }

        if (self.spec != null) {
            srv(.trc, func, "%s", .{@as([*:0]const u8, "speculative decoding context initialized\n")});
        } else {
            if (self.spec_init) |*si| si.deinit();
            self.spec_init = null;
            self.ctx_dft = null;
            self.model_dft = null;
        }

        if (self.spec == null and self.params_base.speculative.hasSynth()) {
            srv(.err, func, "%s", .{@as([*:0]const u8, "synthetic acceptance requires an initialized speculative decoding context\n")});
            return false;
        }

        for (self.slots, 0..) |*slot, i| {
            slot.id = @intCast(i);
            slot.ctx_tgt = self.ctx_tgt;
            slot.ctx_dft = self.ctx_dft;
            slot.mem.init(self.ctx_tgt, self.ctx_dft);
            slot.spec = self.spec;
            slot.n_ctx = self.nCtxSlot();

            slot.mctx = self.mctx;
            slot.prompt.tokens.has_mtmd = self.mctx != null;

            if (self.hasCapTts()) {
                slot.tts.ctx = c.mtmd_helper_gen_audio_init(self.ctx_tgt, self.mctx);
            }

            slt(.trc, func, slot, "new slot, n_ctx = %d\n", .{slot.n_ctx});

            // callback_on_release and callback_on_reset, see `Slot.release`
            slot.owner = self;

            slot.reset();
        }

        {
            self.trace = envInt("LLAMA_TRACE");

            if (self.trace != 0) {
                srv(.wrn, func, "LLAMA_TRACE = %d\n", .{self.trace});
            }
        }

        {
            self.slots_debug = envInt("LLAMA_SERVER_SLOTS_DEBUG");

            if (self.slots_debug != 0) {
                srv(.wrn, func, "LLAMA_SERVER_SLOTS_DEBUG = %d\n", .{self.slots_debug});
            }
        }

        {
            self.slots_n_diff = envInt("LLAMA_SERVER_SLOTS_N_DIFF");

            if (self.slots_n_diff != 0) {
                srv(.wrn, func, "LLAMA_SERVER_SLOTS_N_DIFF = %d\n", .{self.slots_n_diff});
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const n_batch: i32 = @intCast(c.llama_n_batch(self.ctx_tgt));
            const n_embd: i32 = c.llama_model_n_embd_inp(self.model_tgt);
            self.batch.deinit();
            self.batch = .{};
            try self.batch.init(self.ctx_tgt, @max(n_batch, self.params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (self.decision.type != .none and !self.params_base.cache_ram_explicit) {
            srv(.inf, func, "%s", .{@as([*:0]const u8, "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n")});
            self.params_base.cache_ram_mib = 0;
            self.params_base.cache_idle_slots = false;
        }

        if (self.params_base.cache_ram_mib != 0) {
            if (self.params_base.cache_ram_mib < 0) {
                srv(.trc, func, "prompt cache is enabled, size limit: %s\n", .{@as([*:0]const u8, "no limit")});
            } else {
                srv(.trc, func, "prompt cache is enabled, size limit: %d MiB\n", .{self.params_base.cache_ram_mib});
            }
            srv(.trc, func, "%s", .{@as([*:0]const u8, "use `--cache-ram 0` to disable the prompt cache\n")});

            if (self.prompt_cache) |pc| {
                pc.deinit();
                gpa.destroy(pc);
            }
            const pc = try gpa.create(task_mod.PromptCache);
            pc.* = task_mod.PromptCache.init(gpa, self.params_base.cache_ram_mib, @intCast(self.n_ctx));
            self.prompt_cache = pc;
        } else {
            srv(.trc, func, "%s", .{@as([*:0]const u8, "prompt cache is disabled - use `--cache-ram N` to enable it\n")});
        }
        srv(.trc, func, "%s", .{@as([*:0]const u8, "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n")});

        if (self.params_base.n_ctx_checkpoints > 0) {
            srv(.trc, func, "context checkpoints enabled, max = %d, min spacing = %d\n", .{ self.params_base.n_ctx_checkpoints, self.params_base.checkpoint_min_step });
        } else {
            srv(.trc, func, "%s", .{@as([*:0]const u8, "context checkpoints disabled\n")});
        }

        if (self.params_base.model_alias.items.items.len != 0) {
            // backward compat: use first alias as model name
            self.model_name = self.params_base.model_alias.items.items[0];
        } else if (self.params_base.model.getName().len != 0) {
            self.model_name = self.params_base.model.getName();
        } else {
            // fallback: derive model name from file name
            self.model_name = std.fs.path.basename(self.params_base.model.path);
        }

        self.model_aliases = self.params_base.model_alias.items.items;
        self.model_tags = self.params_base.model_tags.items.items;

        // propagate new defaults back to caller
        if (params != &self.params_base) params.* = self.params_base;

        if (!is_resume) {
            return self.initServer();
        }

        const empty: Value = .null;
        self.callback_state.call(.ready, &empty);

        return true;
    }

    /// The `has_mmproj && fit_params` block of `load_model`
    /// (server-context.cpp:1203 @8642541b2): the projector's worst-case
    /// memory added to the fit's target margin, per device.
    fn addMmprojMemory(self: *Impl, mmproj_usage: *const mtmd_mod.mtmd.MemoryUsage, t_elapsed: i64) !void {
        const func = "load_model";
        const mmproj_mem = &mmproj_usage.backend_mem_usage;
        if (mmproj_mem.items.items.len != 0) {
            var total: usize = 0;
            for (mmproj_mem.items.items) |e| total += e.size;
            srv(.trc, func, "[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", .{
                @as(f64, @floatFromInt(total)) / (1024.0 * 1024.0), @as(f64, @floatFromInt(t_elapsed)) / 1000.0,
            });
            ggmlAssert(self.params_base.fit_params_target.items.len != 0, 1280, "!params_base.fit_params_target.empty()");
            for (mmproj_mem.items.items) |e| {
                const dev = e.dev;
                const sz = e.size;
                var i: usize = 0;
                while (i < c.ggml_backend_dev_count()) : (i += 1) {
                    if (@intFromPtr(c.ggml_backend_dev_get(i)) == @intFromPtr(dev)) {
                        if (i < self.params_base.fit_params_target.items.len) {
                            srv(.dbg, func, "[mtmd] adding %.2f MiB to fit_params_target for device %s\n", .{
                                @as(f64, @floatFromInt(sz)) / (1024.0 * 1024.0), c.ggml_backend_dev_name(c.ggml_backend_dev_get(i)),
                            });
                            self.params_base.fit_params_target.items[i] += sz;
                        }
                        break;
                    }
                }
            }
        } else {
            srv(.err, func, "%s", .{@as([*:0]const u8, "[mtmd] failed to get memory usage of mmproj\n")});
        }
    }

    fn envInt(name: [*:0]const u8) c_int {
        const v = std.c.getenv(name) orelse return 0;
        return std.c.atoi(@ptrCast(v));
    }

    /// Ports `server_context_impl`'s `init` (server-context.cpp:1644
    /// @8642541b2): unlike `load_model`, this is only called once during
    /// initialization.
    fn initServer(self: *Impl) !bool {
        const func = "init";
        const a = self.arena.allocator();
        ggmlAssert(self.ctx_tgt != null, 1645, "ctx_tgt   != nullptr");
        ggmlAssert(self.model_tgt != null, 1646, "model_tgt != nullptr");

        ggmlAssert(!self.sleeping, 1648, "!sleeping");

        // wiring up server queues
        self.queue_tasks.onCallbacks(self, onNewTask, onUpdateSlots);
        try self.queue_tasks.onSleepingState(self, onSleeping);

        self.metrics.init();

        if (self.params_base.cache_idle_slots) {
            if (self.params_base.cache_ram_mib == 0) {
                srv(.wrn, func, "%s", .{@as([*:0]const u8, "--cache-idle-slots requires --cache-ram, disabling\n")});
                self.params_base.cache_idle_slots = false;
            } else {
                if (self.params_base.kv_unified) {
                    srv(.trc, func, "%s", .{@as([*:0]const u8, "idle slots will be saved to prompt cache and cleared upon starting a new task\n")});
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    srv(.trc, func, "%s", .{@as([*:0]const u8, "idle slots will be saved to prompt cache upon starting a new task\n")});
                }
                srv(.dbg, func, "%s", .{@as([*:0]const u8, "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n")});
            }
        }

        {
            const cfg = self.params_base.ui_config_json;
            if (cfg.len != 0) {
                if (json.parse(a, cfg)) |json_settings| {
                    self.json_ui_settings = json_settings;
                } else |e| {
                    const msg = dupZ(whatOf(e));
                    defer freeZ(msg);
                    srv(.err, func, "%s: failed to parse UI config: %s\n", .{ @as([*:0]const u8, func), msg });
                    return false;
                }
            }
        }

        // populate chat template params
        {
            var enable_thinking = false;

            const ok = self.initTemplates(&enable_thinking) catch |e| blk: {
                if (e == error.OutOfMemory) return e;
                const msg = dupZ(whatOf(e));
                defer freeZ(msg);
                srv(.err, func, "%s: chat template parsing error: %s\n", .{ @as([*:0]const u8, func), msg });
                srv(.err, func, "%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", .{@as([*:0]const u8, func)});
                srv(.err, func, "%s: for example: --no-jinja --chat-template chatml\n", .{@as([*:0]const u8, func)});
                break :blk false;
            };
            if (!ok) return false;

            gpa.free(self.asr_prompt);
            self.asr_prompt = &.{};
            self.asr_prompt = convert.getAsrPrompt(gpa, self.model_tgt) catch |e| {
                if (e == error.OutOfMemory) return e;
                const msg = dupZ(whatOf(e));
                defer freeZ(msg);
                srv(.err, func, "%s: \"asr\" chat template error: %s\n", .{ @as([*:0]const u8, func), msg });
                return false;
            };

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            var kwargs: oaicompat.KwMap = .{};
            for (self.params_base.default_template_kwargs.entries.items) |e| try kwargs.put(a, e.key, e.value);
            self.chat_params = .{
                .use_jinja = self.params_base.use_jinja,
                .prefill_assistant = self.params_base.prefill_assistant,
                .reasoning_format = @fromBackingInt(@intCast(@backingInt(self.params_base.reasoning_format))),
                .chat_template_kwargs = kwargs,
                .tmpls = &self.tmpls.?,
                .allow_image = self.mtmd_cap.vision,
                .allow_audio = self.mtmd_cap.audio,
                .allow_video = self.mtmd_cap.video,
                .enable_thinking = enable_thinking,
                .reasoning_budget = self.params_base.sampling.reasoning_budget_tokens,
                .reasoning_budget_message = self.params_base.sampling.reasoning_budget_message,
                .media_path = self.params_base.media_path,
                .force_pure_content = self.params_base.force_pure_content_parser,
            };

            {
                const caps = chat.templates.getCaps(&self.tmpls.?);
                const it: ?[]const u8 = for (self.params_base.default_template_kwargs.entries.items) |e| {
                    if (std.mem.eql(u8, e.key, "preserve_reasoning")) break e.value;
                } else null;
                const supported = chat.templates.capsValue(caps, "supports_preserve_reasoning");
                const specified = self.params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                const enabled = it != null and std.mem.eql(u8, it.?, "true");
                if (supported) {
                    const v = dupZ(it orelse "unset (template default)");
                    defer freeZ(v);
                    srv(.trc, func, "preserve_reasoning kwarg: %s\n", .{v});
                } else {
                    srv(.trc, func, "%s", .{@as([*:0]const u8, "preserve_reasoning kwarg: not supported by template\n")});
                }
                if (supported and !specified) {
                    srv(.wrn, func, "%s", .{@as([*:0]const u8, "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n")});
                }
                if (supported and !enabled) {
                    srv(.inf, func, "%s", .{@as([*:0]const u8, "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n")});
                }
                if (!supported and specified and enabled) {
                    srv(.wrn, func, "%s", .{@as([*:0]const u8, "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n")});
                }
            }
        }

        return true;
    }

    /// The `try` block of `init`'s chat-template section
    /// (server-context.cpp:1644 @8642541b2): `common_chat_templates_init`,
    /// the example's trace line, and whether thinking is on.
    fn initTemplates(self: *Impl, enable_thinking: *bool) !bool {
        const func = "init";
        const a = self.arena.allocator();
        var info = try chat.templates.modelInfo(a, @ptrCast(self.model_tgt.?));
        self.tmpls = try chat.templates.init(gpa, &info, self.params_base.chat_template, "", "");

        if (logOn(log.level_trace)) {
            var tmp = std.heap.ArenaAllocator.init(gpa);
            defer tmp.deinit();
            const ta = tmp.allocator();
            var kwargs: std.ArrayList(chat.templates.KwArg) = .empty;
            for (self.params_base.default_template_kwargs.entries.items) |e| try kwargs.append(ta, .{ .key = e.key, .value = e.value });
            const example = try chat.templates.formatExample(ta, &self.tmpls.?, self.params_base.use_jinja, kwargs.items);
            srv(.trc, func, "%s: chat template, example_format: '%s'\n", .{ @as([*:0]const u8, func), (try ta.dupeSentinel(u8, example, 0)).ptr });
        }

        // thinking is enabled if:
        // 1. It's not explicitly disabled via --reasoning off
        // 2. The chat template supports it
        const template_supports_thinking = self.params_base.use_jinja and try chat.templates.supportEnableThinking(gpa, &self.tmpls.?);
        enable_thinking.* = self.params_base.enable_reasoning != 0 and template_supports_thinking;
        srv(.trc, func, "%s: chat template, thinking = %d\n", .{ @as([*:0]const u8, func), @as(c_int, @intFromBool(enable_thinking.*)) });
        return true;
    }

    fn onNewTask(ctx: *anyopaque, task: *Task, is_yielding: bool) anyerror!bool {
        const self: *Impl = @ptrCast(@alignCast(ctx));
        return self.processSingleTask(task, is_yielding);
    }

    fn onUpdateSlots(ctx: *anyopaque) anyerror!void {
        const self: *Impl = @ptrCast(@alignCast(ctx));
        return self.updateSlots();
    }

    fn onSleeping(ctx: *anyopaque, sleeping: bool) void {
        const self: *Impl = @ptrCast(@alignCast(ctx));
        self.handleSleepingState(sleeping) catch |e| {
            const msg = dupZ(whatOf(e));
            defer freeZ(msg);
            ggmlAbort(1076, "failed to change the sleeping state: %s", .{msg});
        };
    }

    /// Ports `server_context_impl`'s `get_slot_by_id` (server-context.cpp:1770
    /// @8642541b2): an out-of-bounds id wraps around.
    fn getSlotById(self: *Impl, id_slot_in: c_int) ?*Slot {
        const n: c_int = @intCast(self.slots.len);
        const id_slot = @rem(id_slot_in, n);

        for (self.slots) |*slot| {
            if (slot.id == id_slot) {
                return slot;
            }
        }

        return null;
    }

    /// Ports `server_context_impl`'s `get_slot_by_cmpl_id` (server-context.cpp:1783 @8642541b2).
    fn getSlotByCmplId(self: *Impl, cmpl_id: []const u8) ?*Slot {
        if (cmpl_id.len == 0) {
            return null;
        }

        for (self.slots) |*slot| {
            if (slot.isProcessing() and slot.task != null and std.mem.eql(u8, slot.taskRef().params.oaicompat_cmpl_id, cmpl_id)) {
                return slot;
            }
        }

        return null;
    }

    /// Ports `server_context_impl`'s `get_available_slot` (server-context.cpp:1797 @8642541b2).
    fn getAvailableSlot(self: *Impl, task: *const Task) !?*Slot {
        const func = "get_available_slot";
        var ret: ?*Slot = null;

        var update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = self.getSlotById(task.id_slot);
            if (ret) |r| {
                // a busy slot is returned untouched, the caller defers the task
                if (r.isProcessing()) {
                    return r;
                }

                slt(.inf, func, r, "selected slot by id (%d)\n", .{task.id_slot});
            }
        }

        // find the slot that has at least n% prompt similarity
        if (self.slot_prompt_similarity != 0.0) {
            var f_sim_best: f32 = 0;

            for (self.slots) |*slot| {
                if (task.id_slot != -1 and slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.isProcessing()) {
                    slt(.trc, func, slot, " - skipping, is_processing = %d\n", .{@as(c_int, @intFromBool(slot.isProcessing()))});
                    continue;
                }

                const toks = &slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (toks.empty()) {
                    slt(.trc, func, slot, "%s", .{@as([*:0]const u8, " - skipping, slot is empty\n")});
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const lcp_len = try toks.getCommonPrefix(&task.tokens);
                const f_sim_cur: f32 = @as(f32, @floatFromInt(lcp_len)) / @as(f32, @floatFromInt(task.tokens.size()));

                slt(.trc, func, slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", .{ @as(f64, f_sim_cur), lcp_len, task.tokens.size(), @as(f64, self.slot_prompt_similarity) });

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best and f_sim_cur > self.slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = slot;
                }
            }

            if (ret) |r| {
                const f_keep: f32 = (f_sim_best * @as(f32, @floatFromInt(task.tokens.size()))) / @as(f32, @floatFromInt(r.prompt.tokens.size()));

                if (task.id_slot == -1) {
                    slt(.inf, func, r, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n", .{ @as(f64, f_sim_best), @as(f64, self.slot_prompt_similarity), @as(f64, f_keep) });
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == null) {
            const free_slots = try self.getFreeSlots(gpa, 1, -1);
            defer gpa.free(free_slots);

            if (free_slots.len != 0) {
                ret = free_slots[0];

                slt(.inf, func, ret.?, "selected slot by LRU, n_tokens = %d, t_last = %lld\n", .{ ret.?.prompt.nTokens(), @as(c_longlong, ret.?.t_last_used) });

                update_cache = true;
            }
        }

        if (ret) |r| {
            update_cache = update_cache and self.prompt_cache != null;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache and !task.isStateless();

            if (update_cache) {
                srv(.trc, func, "%s", .{@as([*:0]const u8, "updating prompt cache\n")});

                const t_start = c.ggml_time_us();

                _ = try r.promptSave(self.prompt_cache.?);

                if (!try r.promptLoad(self.prompt_cache.?, &task.tokens)) {
                    r.promptClear();
                }

                self.prompt_cache.?.update();

                srv(.trc, func, "prompt cache update took %.2f ms\n", .{@as(f64, @floatFromInt(c.ggml_time_us() - t_start)) / 1000.0});
            }
        }

        return ret;
    }

    /// Ports `server_context_impl`'s `try_clear_idle_slots` (server-context.cpp:1911
    /// @8642541b2): true if at least one slot has been cleared.
    fn tryClearIdleSlots(self: *Impl) bool {
        var res = false;

        if (!self.params_base.kv_unified) {
            return res;
        }

        for (self.slots) |*slot| {
            if (slot.isProcessing()) {
                continue;
            }

            if (slot.prompt.nTokens() > 0) {
                srv(.wrn, "try_clear_idle_slots", "purging slot %d with %zu tokens\n", .{ slot.id, slot.prompt.tokens.size() });

                slot.promptClear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    /// Ports `server_context_impl`'s `construct_lora_list` (server-context.cpp:1938 @8642541b2).
    ///
    /// Return: owned by `gpa`.
    fn constructLoraList(self: *const Impl, config: *const common.LoraMap) ![]LoraInfo {
        const output = try gpa.dupe(LoraInfo, self.params_base.lora_adapters.items); // copy
        for (output, 0..) |*o, i| {
            o.scale = 0.0;
            for (config.entries.items) |e| {
                if (e.id == @as(c_int, @intCast(i))) {
                    o.scale = e.scale;
                    break;
                }
            }
        }
        return output;
    }

    /// Ports `server_context_impl`'s `launch_slot_with_task` (server-context.cpp:1951
    /// @8642541b2). Takes the task: it ends in the slot, or is dropped.
    fn launchSlotWithTask(self: *Impl, slot: *Slot, task_in: Task) !bool {
        const func = "launch_slot_with_task";
        var task = task_in;
        var moved = false;
        defer if (!moved) task.deinit(gpa);

        // process per-request lora adapters
        if (task.params.lora.entries.items.len != 0) {
            const task_loras = try self.constructLoraList(&task.params.lora);
            if (!common.areLoraEqual(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (common.loraShouldClearCache(slot.lora, task_loras)) {
                    slt(.trc, func, slot, "clearing cache for lora change. %zu loras -> %zu loras\n", .{ slot.lora.len, task.params.lora.entries.items.len });
                    slot.prompt.clear();
                } else {
                    slt(.trc, func, slot, "keeping cache for alora. %zu target loras\n", .{task_loras.len});
                }
                gpa.free(slot.lora);
                slot.lora = task_loras;
            } else {
                gpa.free(task_loras);
            }
        } else {
            gpa.free(slot.lora);
            slot.lora = try gpa.dupe(LoraInfo, self.params_base.lora_adapters.items);
        }

        // if using alora, make sure it's only a single one requested and active
        var alora_invocation_start: usize = task.tokens.size();
        if (common.loraAllAlora(slot.lora)) {
            const enabled_ids = try common.loraGetEnabledIds(gpa, slot.lora);
            defer gpa.free(enabled_ids);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.len != 1) {
                try self.sendErrorTask(&task, "Cannot run multiple aLoRAs in a single request", .invalid_request);
                return false;
            }
            const lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const n_invocation_tokens = c.llama_adapter_get_alora_n_invocation_tokens(lora);
            const invocation_tokens = c.llama_adapter_get_alora_invocation_tokens(lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            var match_idx: c_int = @as(c_int, @intCast(n_invocation_tokens)) - 1;
            var i: c_int = @as(c_int, @intCast(task.tokens.size())) - 1;
            while (i >= 0) : (i -= 1) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens.at(@intCast(i)) == invocation_tokens[@intCast(match_idx)]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = @intCast(i);
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    match_idx -= 1;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = @as(c_int, @intCast(n_invocation_tokens)) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                slt(.dbg, func, slot, "alora %zu requested, but not found. deactivating\n", .{enabled_ids[0]});
                slot.lora[enabled_ids[0]].scale = 0.0;
            } else {
                slt(.dbg, func, slot, "alora %zu activated starting at %zu\n", .{ enabled_ids[0], alora_invocation_start });
                slot.alora_invocation_start = @intCast(alora_invocation_start);
            }
        }

        if (!task.tokens.validate(self.ctx_tgt)) {
            try self.sendErrorTask(&task, "Prompt contains invalid tokens", .invalid_request);
            return false;
        }

        if (logOn(log.level_debug)) {
            var tmp = std.heap.ArenaAllocator.init(gpa);
            defer tmp.deinit();
            var j = try slot.toJson(tmp.allocator(), false);
            const s = try tmp.allocator().dupeSentinel(u8, try common.safeJsonToStr(tmp.allocator(), &j), 0);
            slt(.dbg, func, slot, "launching slot : %s\n", .{s.ptr});
        }

        if (task.type == .tts) {
            ggmlAssert(self.hasCapTts(), 2025, "has_cap_tts()"); // should already checked in route handler
            if (!slot.tts.isSupported()) {
                slot.tts.ctx = c.mtmd_helper_gen_audio_init(self.ctx_tgt, slot.mctx);
            }
            // mtmd_helper always re-eval the whole prompt
            slot.promptClear();
            task.tts_inp.data.seq_id = slot.id;
            if (c.mtmd_helper_gen_audio_set_input(slot.tts.ctx, task.tts_inp.get()) != 0) {
                try self.sendErrorTask(&task, "failed to process TTS prompt", .server);
                return false;
            }
        }

        // realtime transcription: the rows come from the audio stream, the helper may bias some tokens
        if (task.type == .transcribe_rt) {
            var sparams = c.mtmd_audio_stream_params_default();
            sparams.prompt = self.asr_prompt.ptr;
            sparams.n_prompt = self.asr_prompt.len;
            if (slot.audio_stream) |st| c.mtmd_audio_stream_free(st);
            slot.audio_stream = c.mtmd_audio_stream_init(self.mctx, sparams);
            if (slot.audio_stream == null) {
                try self.sendErrorTask(&task, "failed to initialize the audio stream", .server);
                return false;
            }
            if (self.asr_prompt.len != 0) {
                const ta = try task.arenaAlloc(gpa);
                const n_bias = c.mtmd_helper_audio_stream_get_logit_bias(self.mctx, self.model_tgt, null, 0);
                const bias = try ta.alloc(c.llama_logit_bias, @intCast(n_bias));
                _ = c.mtmd_helper_audio_stream_get_logit_bias(self.mctx, self.model_tgt, bias.ptr, n_bias);
                try task.params.sampling.logit_bias.appendSlice(ta, bias);
            }
        }

        // initialize samplers
        if (task.needSampling()) {
            if (slot.smpl) |s| s.destroy();
            slot.smpl = null;
            slot.smpl = sampling.init(self.model_tgt, &task.params.sampling) catch |e| {
                if (e == error.OutOfMemory) return e;
                var buf: [4200]u8 = undefined;
                const err_msg = std.fmt.bufPrint(&buf, "Failed to initialize samplers: {s}", .{tc.arg.what()}) catch &buf;
                try self.sendErrorTask(&task, err_msg, .invalid_request);
                return false;
            };

            const need_pre_sample_logits = task.params.sampling.n_probs > 0 and !task.params.post_sampling_probs;

            var use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling = use_backend_sampling and !need_pre_sample_logits;

            // TODO: check verify if this actually works with TTS
            use_backend_sampling = use_backend_sampling and task.type != .tts;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                _ = c.llama_set_sampler(self.ctx_tgt, slot.id, slot.smpl.?.get());
            } else {
                _ = c.llama_set_sampler(self.ctx_tgt, slot.id, null);
            }

            if (logOn(log.level_trace)) {
                var tmp = std.heap.ArenaAllocator.init(gpa);
                defer tmp.deinit();
                const ta = tmp.allocator();
                const chain = try ta.dupeSentinel(u8, try slot.smpl.?.print(ta), 0);
                slt(.trc, func, slot, "sampler chain: %s\n", .{chain.ptr});
                const sp = try ta.dupeSentinel(u8, try sampling.paramsPrint(ta, &task.params.sampling), 0);
                slt(.trc, func, slot, "sampler params: \n%s\n", .{sp.ptr});
            }

            if (self.spec != null and spec_mod.getSynthProbs(self.spec).len != 0) {
                const seed: u32 = if (task.params.sampling.seed == c.LLAMA_DEFAULT_SEED)
                    common.randomDevice()
                else
                    task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            if (slot.smpl) |s| s.destroy();
            slot.smpl = null;
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = if (task.params.n_predict != -1) task.params.n_predict else self.params_base.n_predict;

        if (slot.task) |*old| old.deinit(gpa);
        slot.task = task;
        moved = true;

        if (slot.taskRef().type == .tts) {
            slot.state = .processing_prompt;
        } else {
            slot.state = if (slot.taskRef().isChild())
                .wait_other // wait for the parent to process prompt
            else
                .started;
        }

        // no prompt, the rows come from the audio stream, which audio controls reach from now on
        if (slot.taskRef().type == .transcribe_rt) {
            slot.promptClear();
            slot.initSampler();
            slot.sampled = token_null;
            slot.i_batch = -1;
            slot.stats.updatePromptStart();
            slot.state = .generating;
            const empty: CompletionTokenOutput = .{};
            try self.sendPartialResponse(slot, &empty, false, true);
        }

        // no prompt either, the session appends text and audio and asks for responses with its controls
        if (slot.taskRef().type == .realtime) {
            const t = slot.taskRef();
            slot.rt = try realtime.RealtimeSlot.create(self.ctx_tgt, self.mctx, slot.id, slot.n_ctx, &t.params.sampling, &self.queue_results, t.id, t.index);
            slot.promptClear();
            slot.stats.updatePromptStart();
            slot.state = .generating;
            slot.audio_wait = true;
            const empty: CompletionTokenOutput = .{};
            try self.sendPartialResponse(slot, &empty, false, true);
        }

        // reset server kill-switch counter
        self.n_empty_consecutive = 0;

        slt(.inf, func, slot, "processing task, is_child = %d\n", .{@as(c_int, @intFromBool(slot.taskRef().isChild()))});
        return true;
    }

    /// Ports `server_context_impl`'s `process_token` (server-context.cpp:2137 @8642541b2).
    ///
    /// Return: whether the slot continues.
    fn processToken(self: *Impl, result: *CompletionTokenOutput, slot: *Slot, a: Allocator) !bool {
        const func = "process_token";
        // remember which tokens were sampled - used for repetition penalties during sampling
        const token_str = result.text_to_send;
        slot.sampled = result.tok;

        try slot.generated.append(token_str, result.tok);
        if (slot.taskRef().params.return_tokens) {
            try slot.generated_tokens.append(gpa, result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        const incomplete = common.validateUtf8(slot.generated.text.items) < slot.generated.text.items.len;

        // search stop word and delete it
        if (!incomplete) {
            var pos = @min(slot.n_sent_text, slot.generated.text.items.len);

            const str_test = try a.dupe(u8, slot.generated.text.items[pos..]);
            var send_text = true;

            var stop_pos = try slot.findStoppingStrings(str_test, token_str.len, true);
            if (stop_pos) |sp| {
                slot.generated.truncate(pos + sp);
                pos = @min(slot.n_sent_text, slot.generated.text.items.len);
            } else if (slot.has_next_token and !c.llama_vocab_is_eog(self.vocab, result.tok)) {
                stop_pos = try slot.findStoppingStrings(str_test, token_str.len, false);
                send_text = stop_pos == null;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = try a.dupe(u8, slot.generated.text.items[pos..]);
                slot.n_sent_text += result.text_to_send.len;
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            try slot.addToken(result);
            if (slot.taskRef().params.stream) {
                try self.sendPartialResponse(slot, result, false, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!self.params_base.ctx_shift and slot.prompt.nTokens() + 1 >= slot.n_ctx) {
            slot.truncated = true;
            slot.stop = .limit;
            slot.has_next_token = false;

            slt(.dbg, func, slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n", .{
                slot.prompt.nTokens(), slot.taskRef().nTokens(), @as(c_int, @intCast(slot.stats.n_gen)), slot.n_ctx,
            });
        }

        // check the limits
        if (slot.stats.n_gen > 0 and slot.has_next_token and !slot.hasBudget()) {
            slot.stop = .limit;
            slot.has_next_token = false;

            slt(.dbg, func, slot, "stopped by limit, n_gen = %d, n_predict = %d\n", .{ @as(c_int, @intCast(slot.stats.n_gen)), slot.taskRef().params.n_predict });
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.taskRef().params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    var pos = slot.last_nl_pos;

                    var n_indent: c_int = 0;
                    const text = slot.generated.text.items;
                    while (pos < text.len and (text[pos] == ' ' or text[pos] == '\t')) {
                        n_indent += 1;
                        pos += 1;
                    }

                    if (pos < slot.generated.text.items.len and n_indent < slot.taskRef().params.n_indent) {
                        slot.stop = .limit;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        slt(.dbg, func, slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", .{ @as(c_int, @intCast(slot.stats.n_gen)), n_indent });
                    }
                }

                // find the next new line
                {
                    if (std.mem.indexOfScalarPos(u8, slot.generated.text.items, @min(slot.last_nl_pos, slot.generated.text.items.len), '\n')) |pos| {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (std.mem.indexOfScalar(u8, result.text_to_send, '\n') != null) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.taskRef().params.t_max_predict_ms > 0 and slot.stats.tGenMs() > @as(f64, @floatFromInt(slot.taskRef().params.t_max_predict_ms))) {
                slot.stop = .limit;
                slot.has_next_token = false;

                slt(.dbg, func, slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", .{ @as(c_int, @intCast(slot.stats.n_gen)), @as(c_int, @truncate(slot.taskRef().params.t_max_predict_ms)) });
            }
        }

        if (c.llama_vocab_is_eog(self.vocab, result.tok)) {
            slot.stop = .eos;
            slot.has_next_token = false;

            slt(.dbg, func, slot, "%s", .{@as([*:0]const u8, "stopped by EOS\n")});
        }

        if (logOn(log.level_debug)) {
            const z = try a.dupeSentinel(u8, token_str, 0);
            slt(.dbg, func, slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", .{ @as(c_int, @intCast(slot.stats.n_gen)), slot.nRemaining(), result.tok, z.ptr });
        }

        return slot.has_next_token; // continue
    }

    /// Ports `server_context_impl`'s `populate_token_probs` (server-context.cpp:2266 @8642541b2).
    fn populateTokenProbs(self: *const Impl, slot: *const Slot, result: *CompletionTokenOutput, post_sampling: bool, special: bool, idx: c_int, a: Allocator) !void {
        const n_probs_request: usize = @intCast(@max(slot.taskRef().params.sampling.n_probs, 0));
        var probs: std.ArrayList(CompletionTokenOutput.ProbInfo) = .empty;
        const vocab = c.llama_model_get_vocab(c.llama_get_model(self.ctx_tgt));

        if (post_sampling) {
            const cur_p = slot.smpl.?.getCandidates(true);
            const max_probs = cur_p.size;
            const n_probs = @min(max_probs, n_probs_request);

            // set probability for sampled token
            for (0..max_probs) |i| {
                if (cur_p.data[i].id == result.tok) {
                    result.prob = cur_p.data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            try probs.ensureTotalCapacity(a, n_probs);
            for (0..n_probs) |i| {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p.data[i].p == 0.0) {
                    break;
                }

                try probs.append(a, .{
                    .tok = cur_p.data[i].id,
                    .txt = try tc.common.tokenToPiece(a, vocab, cur_p.data[i].id, special),
                    .prob = cur_p.data[i].p,
                });
            }
        } else {
            const cur = try common.getTokenProbabilities(a, self.ctx_tgt, idx, n_probs_request);
            const max_probs = cur.len;
            const n_probs = @min(max_probs, n_probs_request);

            // set probability for sampled token
            for (0..max_probs) |i| {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            try probs.ensureTotalCapacity(a, n_probs);
            for (0..n_probs) |i| {
                try probs.append(a, .{
                    .tok = cur[i].id,
                    .txt = try tc.common.tokenToPiece(a, vocab, cur[i].id, special),
                    .prob = cur[i].p,
                });
            }
        }
        result.probs = probs.items;
    }

    /// Ports `server_context_impl`'s `send_error` (server-context.cpp:2323 @8642541b2), over a task.
    fn sendErrorTask(self: *Impl, task: *const Task, err: []const u8, ty: ErrorType) !void {
        try self.sendError(task.id, err, ty, 0, 0);
    }

    /// Ports `server_context_impl`'s `send_error` (server-context.cpp:2327 @8642541b2), over a slot.
    fn sendErrorSlot(self: *Impl, slot: *const Slot, err: []const u8, ty: ErrorType) !void {
        try self.sendError(slot.taskRef().id, err, ty, slot.taskRef().nTokens(), slot.n_ctx);
    }

    /// Ports `server_context_impl`'s `send_error` (server-context.cpp:2331 @8642541b2).
    fn sendError(self: *Impl, id_task: c_int, err: []const u8, ty: ErrorType, n_prompt_tokens: i32, n_ctx: i32) !void {
        {
            const z = dupZ(err);
            defer freeZ(z);
            srv(.err, "send_error", "task id = %d, error: %s\n", .{ id_task, z });
        }

        if (ty == .exceed_context_size) {
            ggmlAssert(n_ctx > 0 and n_prompt_tokens > 0, 2335, "n_ctx > 0 && n_prompt_tokens > 0");
        }

        const res = try Result.create(gpa, .err);
        errdefer res.destroy(gpa);
        res.id = id_task;
        res.kind.err.err_type = ty;
        res.kind.err.err_msg = try res.alloc().dupe(u8, err);
        res.kind.err.n_prompt_tokens = n_prompt_tokens;
        res.kind.err.n_ctx = n_ctx;

        try self.queue_results.send(res);
    }

    /// Ports `server_context_impl`'s `send_partial_response` (server-context.cpp:2348 @8642541b2).
    fn sendPartialResponse(self: *Impl, slot: *Slot, tkn: *const CompletionTokenOutput, is_progress: bool, is_begin: bool) !void {
        const res = try Result.create(gpa, .cmpl_partial);
        errdefer res.destroy(gpa);
        const ra = res.alloc();
        const p = &res.kind.cmpl_partial;
        const t = slot.taskRef();

        res.id = t.id;
        res.index = t.index;

        if (is_progress) {
            p.is_progress = true;
            p.progress.total = t.nTokens();
            p.progress.cache = @intCast(slot.stats.n_prompt_cached);
            p.progress.processed = @intCast(slot.prompt.tokens.size());
            p.progress.time_ms = @divTrunc(slot.stats.tElapsedUs(), 1000);
        }
        if (is_begin) {
            p.is_begin = true;
        } else {
            p.content = try slot.generated.substr(ra, slot.n_sent_text - tkn.text_to_send.len, tkn.text_to_send.len);
            p.tokens = try ra.dupe(Token, &.{tkn.tok});
        }

        p.n_decoded = @intCast(slot.stats.n_gen);
        p.n_prompt_tokens = t.nTokens();
        p.n_prompt_tokens_cache = @intCast(slot.stats.n_prompt_cached);
        p.post_sampling_probs = t.params.post_sampling_probs;

        p.verbose = t.params.verbose;
        p.res_type = t.params.res_type;
        p.oaicompat_model = try ra.dupe(u8, t.params.oaicompat_model);
        p.oaicompat_cmpl_id = try ra.dupe(u8, t.params.oaicompat_cmpl_id);

        // populate res.probs_output
        if (t.params.sampling.n_probs > 0) {
            p.prob_output = try tkn.clone(ra); // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != .none or t.params.timings_per_token) {
            p.stats = slot.stats;
        }

        try self.queue_results.send(res);
    }

    /// Ports `server_context_impl`'s `send_tts_result` (server-context.cpp:2391 @8642541b2).
    fn sendTtsResult(self: *Impl, slot: *const Slot, sample_rate: i32, data: [*c]const u8, data_len: usize, final: bool) !void {
        const res = try Result.create(gpa, .tts);
        errdefer res.destroy(gpa);

        res.id = slot.taskRef().id;
        res.index = slot.taskRef().index;
        res.kind.tts.sample_rate = sample_rate;
        res.kind.tts.audio = if (data_len == 0) "" else try res.alloc().dupe(u8, data[0..data_len]);
        res.kind.tts.final = final;

        try self.queue_results.send(res);
    }

    /// Ports `server_context_impl`'s `send_final_response` (server-context.cpp:2403 @8642541b2).
    fn sendFinalResponse(self: *Impl, slot: *Slot) !void {
        const res = try Result.create(gpa, .cmpl_final);
        errdefer res.destroy(gpa);
        const ra = res.alloc();
        const f = &res.kind.cmpl_final;
        const t = &slot.task.?;

        res.id = t.id;
        res.id_slot = slot.id;

        res.index = t.index;

        // keep copy of last generated text for debugging purposes
        if (self.slots_debug != 0) {
            slot.debug_generated_text.clearRetainingCapacity();
            try slot.debug_generated_text.appendSlice(gpa, slot.generated.text.items);
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (t.params.stream) {
            f.content = .init(ra);
            f.tokens = &.{};
        } else {
            f.content = try slot.generated.clone(ra);
            slot.generated.deinit(); // std::move
            slot.generated = .init(gpa);
            f.tokens = try ra.dupe(Token, slot.generated_tokens.items);
            slot.generated_tokens.clearRetainingCapacity(); // std::move
        }
        f.stats = slot.stats;
        f.prompt = try t.tokens.detokenize(ra, self.ctx_tgt, true);
        f.response_fields = try task_mod.deepClone(ra, t.params.response_fields);
        t.params.response_fields = &.{}; // std::move

        f.truncated = slot.truncated;
        f.n_decoded = @intCast(slot.stats.n_gen);
        f.n_prompt_tokens = t.nTokens();
        f.n_prompt_tokens_cache = @intCast(slot.stats.n_prompt_cached);
        f.n_tokens_cached = slot.prompt.nTokens();
        f.has_new_line = slot.has_new_line;
        f.stopping_word = try ra.dupe(u8, slot.stopping_word.items);
        f.stop = slot.stop;
        f.post_sampling_probs = t.params.post_sampling_probs;

        f.verbose = t.params.verbose;
        f.stream = t.params.stream;
        f.include_usage = t.params.include_usage;
        f.res_type = t.params.res_type;
        f.oaicompat_model = try ra.dupe(u8, t.params.oaicompat_model);
        f.oaicompat_cmpl_id = try ra.dupe(u8, t.params.oaicompat_cmpl_id);

        // populate res.probs_output
        if (t.params.sampling.n_probs > 0) {
            var n = slot.generated_token_probs.items.len;
            if (!t.params.stream and slot.stop == .word) {
                const stop_word_toks = try tc.common.tokenize(ra, self.vocab, slot.stopping_word.items, false, false);

                const safe_offset = @min(slot.generated_token_probs.items.len, stop_word_toks.len);
                n -= safe_offset;
            }
            const out = try ra.alloc(CompletionTokenOutput, n);
            for (slot.generated_token_probs.items[0..n], 0..) |*p, i| out[i] = try p.clone(ra);
            f.probs_output = out;
        }

        f.generation_params = try t.params.clone(ra); // copy the parameters

        try self.queue_results.send(res);
    }

    /// Ports `server_context_impl`'s `send_embedding` (server-context.cpp:2466 @8642541b2).
    fn sendEmbedding(self: *Impl, slot: *const Slot, batch: *const tc.common.Batch) !void {
        const res = try Result.create(gpa, .embd);
        errdefer res.destroy(gpa);
        const ra = res.alloc();
        const t = slot.taskRef();
        res.id = t.id;
        res.index = t.index;
        res.kind.embd.n_tokens = t.nTokens();
        res.kind.embd.res_type = t.params.res_type;

        const n_embd_out: usize = @intCast(c.llama_model_n_embd_out(self.model_tgt));

        const embd_res = try ra.alloc(f32, n_embd_out);
        @memset(embd_res, 0.0);

        var embeddings: std.ArrayList([]const f32) = .empty;

        var i: usize = 0;
        while (i < @as(usize, @intCast(batch.size()))) : (i += 1) {
            const bt = &batch.tokens.items[i];
            if (!bt.output or bt.seq_id != slot.id) {
                continue;
            }

            var embd: [*c]const f32 = null;
            if (c.llama_pooling_type(slot.ctx_tgt) == c.LLAMA_POOLING_TYPE_NONE) {
                embd = c.llama_get_embeddings_ith(slot.ctx_tgt, @intCast(i));
            } else {
                embd = c.llama_get_embeddings_seq(slot.ctx_tgt, bt.seq_id);
            }

            if (embd == null) {
                slt(.err, "send_embedding", slot, "failed to get embeddings, token = %d, seq_id = %d\n", .{ bt.id, bt.seq_id });

                const zeros = try ra.alloc(f32, n_embd_out);
                @memset(zeros, 0.0);
                try embeddings.append(ra, zeros);
                continue;
            }

            // normalize only when there is pooling
            if (c.llama_pooling_type(slot.ctx_tgt) != c.LLAMA_POOLING_TYPE_NONE) {
                tc.common.embdNormalize(embd, embd_res.ptr, @intCast(n_embd_out), t.params.embd_normalize);
                try embeddings.append(ra, embd_res);
                break;
            }

            try embeddings.append(ra, try ra.dupe(f32, embd[0..n_embd_out]));
        }
        res.kind.embd.embedding = embeddings.items;

        slt(.dbg, "send_embedding", slot, "%s", .{@as([*:0]const u8, "sending embeddings\n")});

        try self.queue_results.send(res);
    }

    /// Ports `server_context_impl`'s `send_decision` (server-context.cpp:2511
    /// @8642541b2): the raw model output of each option -- the logits of the
    /// labels, the scores of a joint head, or the embeddings at the markers.
    fn sendDecision(self: *Impl, slot: *const Slot, batch: *const tc.common.Batch, i_batch: i32) !void {
        const func = "send_decision";
        const res = try Result.create(gpa, .decision);
        errdefer res.destroy(gpa);
        const ra = res.alloc();
        const t = slot.taskRef();
        res.id = t.id;
        res.index = t.index;
        res.kind.decision.n_tokens = t.nTokens();

        const decision = &t.decision;
        var scores: std.ArrayList(f32) = .empty;

        if (decision.pooled) {
            const embd = c.llama_get_embeddings_seq(slot.ctx_tgt, slot.id);
            if (embd == null) {
                res.destroy(gpa);
                try self.sendErrorSlot(slot, "failed to get the class scores", .server);
                return;
            }
            try scores.appendSlice(ra, embd[0..@intCast(c.llama_model_n_embd_out(self.model_tgt))]);
        } else if (decision.labels.items.len != 0) {
            const logits = c.llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == null) {
                res.destroy(gpa);
                try self.sendErrorSlot(slot, "failed to get logits", .server);
                return;
            }
            const n_vocab = c.llama_vocab_n_tokens(self.vocab);
            for (decision.labels.items) |label| {
                ggmlAssert(label >= 0 and label < n_vocab, 2534, "label >= 0 && label < n_vocab");
                try scores.append(ra, logits[@intCast(label)]);
            }
            if (decision.label_groups.items.len != 0) {
                var grouped: std.ArrayList(f32) = .empty;
                var i: usize = 0;
                for (decision.label_groups.items) |n| {
                    ggmlAssert(n > 0 and i + @as(usize, @intCast(n)) <= scores.items.len, 2541, "n > 0 && i + n <= res->scores.size()");
                    // std::max_element: the first of the largest
                    const group = scores.items[i .. i + @as(usize, @intCast(n))];
                    var largest: usize = 0;
                    for (group[1..], 1..) |v, k| {
                        if (group[largest] < v) largest = k;
                    }
                    try grouped.append(ra, group[largest]);
                    i += @intCast(n);
                }
                scores = grouped;
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            var idx: std.ArrayList(i32) = .empty;
            for (batch.tokens.items, 0..) |*bt, i| {
                if (bt.output and bt.seq_id == slot.id) {
                    try idx.append(ra, @intCast(i));
                }
            }
            const n_idx: i32 = @intCast(idx.items.len);
            const pos_first: i32 = slot.prompt.nTokens() - n_idx;
            const GetEmbd = struct {
                fn run(ctx: ?*c.llama_context, ids: []const i32, first: i32, pos: i32) [*c]const f32 {
                    const i = pos - first;
                    return if (i >= 0 and i < @as(i32, @intCast(ids.len))) c.llama_get_embeddings_ith(ctx, ids[@intCast(i)]) else null;
                }
            };

            // joint head (decision model): the scores are the first rows
            var i: i32 = 0;
            while (i < decision.n_scores) : (i += 1) {
                const embd: [*c]const f32 = if (i < n_idx) c.llama_get_embeddings_ith(slot.ctx_tgt, idx.items[@intCast(i)]) else null;
                if (embd == null) {
                    res.destroy(gpa);
                    try self.sendErrorSlot(slot, "failed to get embeddings", .server);
                    return;
                }
                try scores.append(ra, embd[0]);
            }

            const n_embd_out = c.llama_model_n_embd_out(self.model_tgt);
            const n_pointer = @divTrunc(n_embd_out, 2);
            const embd_q: [*c]const f32 = if (decision.pointer >= 0) GetEmbd.run(slot.ctx_tgt, idx.items, pos_first, decision.pointer) else null;
            ggmlAssert(decision.column >= 0 and decision.column < n_embd_out, 2574, "decision.column >= 0 && decision.column < n_embd_out");

            for (decision.markers.items) |marker| {
                const embd = GetEmbd.run(slot.ctx_tgt, idx.items, pos_first, marker);
                if (embd == null or (decision.pointer >= 0 and embd_q == null)) {
                    res.destroy(gpa);
                    try self.sendErrorSlot(slot, "failed to get embeddings, the question and its options must fit in one batch", .server);
                    return;
                }
                if (decision.pointer < 0) {
                    try scores.append(ra, embd[@intCast(decision.column)]);
                    continue;
                }
                // as the reference compiles it: from 4 elements, four at a time
                // with each product rounded and added in order, and only the
                // `n % 4` left over fused (read from the disassembly)
                const np: usize = @intCast(n_pointer);
                const n_vec = if (np >= 4) np - np % 4 else 0;
                var dot: f32 = 0.0;
                for (0..np) |k| {
                    if (k < n_vec) {
                        dot += embd_q[k] * embd[np + k];
                    } else {
                        dot = @mulAdd(f32, embd_q[k], embd[np + k], dot);
                    }
                }
                try scores.append(ra, dot / @sqrt(@as(f32, @floatFromInt(n_pointer))));
            }
        }
        res.kind.decision.scores = scores.items;

        slt(.dbg, func, slot, "%s", .{@as([*:0]const u8, "sending decision result\n")});

        try self.queue_results.send(res);
    }

    /// Ports `server_context_impl`'s `send_rerank` (server-context.cpp:2599 @8642541b2).
    fn sendRerank(self: *Impl, slot: *const Slot, batch: *const tc.common.Batch) !void {
        const res = try Result.create(gpa, .rerank);
        errdefer res.destroy(gpa);
        const t = slot.taskRef();
        res.id = t.id;
        res.index = t.index;
        res.kind.rerank.n_tokens = t.nTokens();

        var i: usize = 0;
        while (i < @as(usize, @intCast(batch.size()))) : (i += 1) {
            const bt = &batch.tokens.items[i];
            if (!bt.output or bt.seq_id != slot.id) {
                continue;
            }

            var embd = c.llama_get_embeddings_seq(self.ctx_tgt, bt.seq_id);
            if (embd == null) {
                embd = c.llama_get_embeddings_ith(self.ctx_tgt, @intCast(i));
            }

            if (embd == null) {
                slt(.err, "send_rerank", slot, "failed to get embeddings, token = %d, seq_id = %d\n", .{ bt.id, bt.seq_id });

                res.kind.rerank.score = -1e6;
                continue;
            }

            res.kind.rerank.score = embd[0];
        }

        slt(.dbg, "send_rerank", slot, "sending rerank result, res.score = %f\n", .{@as(f64, res.kind.rerank.score)});

        try self.queue_results.send(res);
    }

    /// Ports `server_context_impl`'s `tokenize_cli_input` (server-context.cpp:2635
    /// @8642541b2): the input of a task set by the CLI. Unreached by the
    /// HTTP server, which tokenizes on its own threads.
    fn tokenizeCliInput(self: *Impl, task: *Task) !bool {
        const a = try task.arenaAlloc(gpa);
        const tokenized = blk: {
            const prompt = task.cli_prompt;
            if (self.mctx != null) {
                break :blk tokens_mod.processMtmdPrompt(gpa, self.mctx, prompt, task.cli_files, self.init_opt, false);
            } else {
                var pv = try Value.initString(a, prompt);
                var list = tokens_mod.tokenizeInputPrompts(gpa, self.vocab, self.mctx, &pv, true, true, self.init_opt) catch |e| break :blk @as(common.Error!ServerTokens, e);
                const first = list.items[0];
                for (list.items[1..]) |*t| t.deinit();
                list.deinit(gpa);
                break :blk @as(common.Error!ServerTokens, first);
            }
        };
        if (tokenized) |toks| {
            task.tokens.deinit();
            task.tokens = toks;
            task.cli_prompt = "";
            task.cli_files = &.{};
        } else |e| {
            if (e == error.OutOfMemory) return e;
            var buf: [4200]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "Failed to format input: {s}", .{whatOf(e)}) catch &buf;
            try self.sendErrorTask(task, msg, .invalid_request);
            return false;
        }
        return true;
    }

    /// Ports `server_context_impl`'s `get_free_slots` (server-context.cpp:2653 @8642541b2):
    /// idle slots that lose the least by being reused: empty slots first, then the least
    /// recently used ones.
    fn getFreeSlots(self: *Impl, a: Allocator, n_slots_needed: usize, exclude_id_slot: c_int) ![]*Slot {
        var free_slots: std.ArrayList(*Slot) = .empty;
        errdefer free_slots.deinit(a);
        for (self.slots) |*slot| {
            if (!slot.isProcessing() and slot.id != exclude_id_slot) {
                try free_slots.append(a, slot);
            }
        }

        // std::stable_sort: empty slots first, then the least recently used
        std.mem.sort(*Slot, free_slots.items, {}, struct {
            fn lt(_: void, x: *Slot, y: *Slot) bool {
                if ((x.prompt.nTokens() == 0) != (y.prompt.nTokens() == 0)) {
                    return x.prompt.nTokens() == 0;
                }
                return x.t_last_used < y.t_last_used;
            }
        }.lt);

        free_slots.shrinkRetainingCapacity(@min(free_slots.items.len, n_slots_needed));

        return free_slots.toOwnedSlice(a);
    }

    /// The `release_slots` lambda of `launch_slots_with_parent_task`
    /// (server-context.cpp:2674 @8642541b2).
    fn releaseSlotsOf(self: *Impl, id_parent: c_int) !void {
        for (self.slots) |*slot| {
            if (slot.isProcessing() and (slot.taskRef().id == id_parent or slot.taskRef().id_parent == id_parent)) {
                try slot.release();
            }
        }
    }

    /// Ports `server_context_impl`'s `launch_slots_with_parent_task`
    /// (server-context.cpp:2674 @8642541b2): launch multiple slots for
    /// parent + child tasks.
    fn launchSlotsWithParentTask(self: *Impl, parent_slot: *Slot, child_slots: []*Slot, parent_task_in: Task) !bool {
        const func = "launch_slots_with_parent_task";
        var parent_task = parent_task_in;
        ggmlAssert(!parent_slot.isProcessing(), 2675, "!parent_slot.is_processing()");
        ggmlAssert(parent_task.isParent(), 2676, "parent_task.is_parent()");
        ggmlAssert(child_slots.len == parent_task.child_tasks.items.len, 2677, "child_slots.size() == parent_task.child_tasks.size()");

        const id_parent = parent_task.id;

        srv(.trc, func, "launching slots for parent task id_task = %d with %zu child tasks\n", .{ id_parent, parent_task.child_tasks.items.len });

        // launch all child tasks first
        for (child_slots, 0..) |slot, idx| {
            // keep the state that the child task is about to overwrite
            if (self.prompt_cache) |pc| {
                if (try slot.promptSave(pc)) {
                    pc.update();
                }
            }

            const id_child = parent_task.child_tasks.items[idx].id;
            const child = parent_task.child_tasks.items[idx];
            // moved out: the parent keeps an empty task in its place
            parent_task.child_tasks.items[idx] = .{};
            if (!try self.launchSlotWithTask(slot, child)) {
                srv(.err, func, "failed to launch slot with child task, id_task = %d\n", .{id_child});
                try self.releaseSlotsOf(id_parent);
                parent_task.deinit(gpa);
                return false;
            }
        }

        // finally, launch the parent task
        if (!try self.launchSlotWithTask(parent_slot, parent_task)) {
            srv(.err, func, "failed to launch slot with task, id_task = %d\n", .{id_parent});
            try self.releaseSlotsOf(id_parent);
            return false;
        }

        return true;
    }

    /// Ports `server_context_impl`'s `create_checkpoint` (server-context.cpp:2724
    /// @8642541b2). `n_tokens_cur`: the number of tokens added to the batch
    /// for the current slot.
    fn createCheckpoint(self: *Impl, slot: *Slot, n_tokens_cur: i64, pos_min: c.llama_pos, pos_max: c.llama_pos) !void {
        const func = "create_checkpoint";
        const id_task = slot.taskRef().id;
        const cps = &slot.prompt.checkpoints;
        const ca = slot.prompt.tokens.gpa;
        const n_max: usize = @intCast(@max(self.params_base.n_ctx_checkpoints, 0));
        const mib = 1024.0 * 1024.0;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        var last: i64 = -1;
        var i: usize = 0;
        while (cps.items.len + 1 >= n_max and i < cps.items.len) {
            const it = &cps.items[i];
            if (it.id_task != id_task and last >= 0 and it.n_tokens <= last + self.params_base.checkpoint_min_step) {
                slt(.trc, func, slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %lld, size = %.3f MiB)\n", .{
                    it.pos_min, it.pos_max, @as(c_longlong, it.n_tokens), @as(f64, @as(f32, @floatFromInt(it.size())) / 1024 / 1024),
                });

                var removed = cps.orderedRemove(i);
                removed.clear(ca);
                continue;
            }

            last = it.n_tokens;
            i += 1;
        }

        while (cps.items.len >= n_max) {
            // make room for the new checkpoint, if needed
            const cur = &cps.items[0];

            slt(.wrn, func, slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %lld, size = %.3f MiB)\n", .{
                cur.pos_min, cur.pos_max, @as(c_longlong, cur.n_tokens), @as(f64, @as(f32, @floatFromInt(cur.size())) / 1024 / 1024),
            });

            var removed = cps.orderedRemove(0);
            removed.clear(ca);
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const n_tokens_new = @as(i64, slot.prompt.nTokens()) - n_tokens_cur;
            i = 0;
            while (i < cps.items.len) {
                if (cps.items[i].n_tokens == n_tokens_new) {
                    slt(.trc, func, slot, "superseding context checkpoint at n_tokens = %lld\n", .{@as(c_longlong, cps.items[i].n_tokens)});
                    var removed = cps.orderedRemove(i);
                    removed.clear(ca);
                } else {
                    i += 1;
                }
            }
        }

        try cps.append(ca, .{});
        const cur = &cps.items[cps.items.len - 1];

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.updatePos(@as(i64, slot.prompt.nTokens()) - n_tokens_cur, pos_min, pos_max);

        try cur.updateTgt(ca, self.ctx_tgt, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        try cur.updateDft(ca, self.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        {
            var data: std.ArrayList(u8) = .fromOwnedSlice(cur.data_spec);
            _ = spec_mod.getState(self.spec, slot.id, &data);
            cur.data_spec = try data.toOwnedSlice(gpa);
        }

        slt(.trc, func, slot, "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %lld, size = %.3f MiB)\n", .{
            @as(c_int, @intCast(cps.items.len)),                         self.params_base.n_ctx_checkpoints, cur.pos_min, cur.pos_max, @as(c_longlong, cur.n_tokens),
            @as(f64, @as(f32, @floatFromInt(cur.size())) / 1024 / 1024),
        });
        _ = mib;
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)

    /// Ports `SLOT_CKPT_MAGIC` and `SLOT_CKPT_VERSION` (server-context.cpp:2791, 2792 @8642541b2).
    const slot_ckpt_magic: u32 = 0x504b4353; // "SCKP"
    const slot_ckpt_version: u32 = 1;

    /// The `std::ifstream` / `std::ofstream` of the slot file's appendices,
    /// over C stdio: `ab` for `std::ios::binary | std::ios::app`, `rb` for
    /// reading.
    const CFile = opaque {};
    extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*CFile;
    extern fn fclose(f: *CFile) c_int;
    extern fn fread(ptr: [*]u8, size: usize, n: usize, f: *CFile) usize;
    extern fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *CFile) usize;
    extern fn fflush(f: *CFile) c_int;
    extern fn ferror(f: *CFile) c_int;
    extern fn fseeko(f: *CFile, off: i64, whence: c_int) c_int;
    extern fn ftello(f: *CFile) i64;

    /// A write stream: once a write fails, the stream stays failed, as
    /// `std::ofstream`'s badbit does.
    const OutFile = struct {
        f: *CFile,
        ok: bool = true,
    };

    /// Ports `ckpt_read` (server-context.cpp:2794 @8642541b2).
    fn ckptRead(f: *CFile, dst: []u8, n_read: *usize) bool {
        if (dst.len != 0 and fread(dst.ptr, 1, dst.len, f) != dst.len) {
            return false;
        }
        n_read.* += dst.len;
        return true;
    }

    fn ckptReadVal(comptime T: type, f: *CFile, dst: *T, n_read: *usize) bool {
        return ckptRead(f, std.mem.asBytes(dst), n_read);
    }

    /// Ports `ckpt_read_buf` (server-context.cpp:2802 @8642541b2). `buf` is
    /// replaced by a `gpa` allocation the caller owns.
    fn ckptReadBuf(f: *CFile, buf: *[]u8, n_avail: usize, n_read: *usize) !bool {
        var n: u64 = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckptReadVal(u64, f, &n, n_read) or n > n_avail - n_read.*) {
            return false;
        }
        gpa.free(buf.*);
        buf.* = try gpa.alloc(u8, @intCast(n));
        return n == 0 or ckptRead(f, buf.*, n_read);
    }

    /// Ports `ckpt_write` (server-context.cpp:2812 @8642541b2).
    fn ckptWrite(out: *OutFile, src: []const u8, n_written: *usize) void {
        if (out.ok and src.len != 0 and fwrite(src.ptr, 1, src.len, out.f) != src.len) {
            out.ok = false;
        }
        n_written.* += src.len;
    }

    fn ckptWriteVal(comptime T: type, out: *OutFile, src: *const T, n_written: *usize) void {
        ckptWrite(out, std.mem.asBytes(src), n_written);
    }

    /// Ports `ckpt_write_buf` (server-context.cpp:2817 @8642541b2).
    fn ckptWriteBuf(out: *OutFile, buf: []const u8, n_written: *usize) void {
        const n: u64 = buf.len;
        ckptWriteVal(u64, out, &n, n_written);
        if (n > 0) {
            ckptWrite(out, buf, n_written);
        }
    }

    /// `ofs.flush(); if (!ofs)`: false if the stream failed.
    fn ckptClose(out: *OutFile) bool {
        if (fflush(out.f) != 0 or ferror(out.f) != 0) out.ok = false;
        if (fclose(out.f) != 0) out.ok = false;
        return out.ok;
    }

    /// Opens `filepath` for reading at `offset`, as `std::ifstream(...,
    /// std::ios::ate)` then `seekg(offset)`; null where the C++ returns 0.
    ///
    /// Return: the file and the bytes after `offset`.
    fn ckptOpenAt(path_z: [*:0]const u8, offset: usize) ?struct { f: *CFile, n_avail: usize } {
        const f = fopen(path_z, "rb") orelse return null;
        if (fseeko(f, 0, 2) != 0) {
            _ = fclose(f);
            return null;
        }
        const end = ftello(f);
        const file_size: usize = if (end < 0) 0 else @intCast(end);
        if (file_size < offset or fseeko(f, @intCast(offset), 0) != 0) {
            _ = fclose(f);
            return null;
        }
        return .{ .f = f, .n_avail = file_size - offset }; // bytes after the llama state payload
    }

    /// Ports `server_context_impl`'s `save_slot_checkpoints` (server-context.cpp:2826
    /// @8642541b2): false if the appendix could not be written completely.
    fn saveSlotCheckpoints(self: *const Impl, filepath: []const u8, slot: *const Slot, n_written: *usize) !bool {
        _ = self;
        const func = "save_slot_checkpoints";
        n_written.* = 0;
        if (slot.prompt.checkpoints.items.len == 0) {
            return true;
        }
        const path_z = try gpa.dupeSentinel(u8, filepath, 0);
        defer gpa.free(path_z);
        var out: OutFile = .{ .f = fopen(path_z.ptr, "ab") orelse {
            slt(.wrn, func, slot, "failed to append context checkpoints to '%s'\n", .{path_z.ptr});
            return false;
        } };
        const magic: u32 = slot_ckpt_magic;
        const version: u32 = slot_ckpt_version;
        const count: u32 = @truncate(slot.prompt.checkpoints.items.len);
        ckptWriteVal(u32, &out, &magic, n_written);
        ckptWriteVal(u32, &out, &version, n_written);
        ckptWriteVal(u32, &out, &count, n_written);
        for (slot.prompt.checkpoints.items) |*cur| {
            ckptWriteVal(i64, &out, &cur.n_tokens, n_written);
            ckptWriteVal(c.llama_pos, &out, &cur.pos_min, n_written);
            ckptWriteVal(c.llama_pos, &out, &cur.pos_max, n_written);
            ckptWriteBuf(&out, cur.data_tgt, n_written);
            ckptWriteBuf(&out, cur.data_dft, n_written);
            ckptWriteBuf(&out, cur.data_spec, n_written);
        }
        if (!ckptClose(&out)) {
            slt(.wrn, func, slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", .{path_z.ptr});
            return false;
        }
        slt(.inf, func, slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n", .{
            count, @as(f64, @as(f32, @floatFromInt(n_written.*)) / 1024 / 1024), path_z.ptr,
        });
        return true;
    }

    /// Ports `server_context_impl`'s `load_slot_checkpoints` (server-context.cpp:2861
    /// @8642541b2).
    ///
    /// Return: the number of bytes consumed, 0 if there is no usable appendix.
    fn loadSlotCheckpoints(self: *const Impl, filepath: []const u8, offset: usize, slot: *Slot) !usize {
        const func = "load_slot_checkpoints";
        const path_z = try gpa.dupeSentinel(u8, filepath, 0);
        defer gpa.free(path_z);
        const opened = ckptOpenAt(path_z.ptr, offset) orelse return 0;
        const f = opened.f;
        defer _ = fclose(f);
        const n_avail = opened.n_avail;
        var n_read: usize = 0;
        var magic: u32 = 0;
        var version: u32 = 0;
        var count: u32 = 0;
        if (!ckptReadVal(u32, f, &magic, &n_read) or magic != slot_ckpt_magic) {
            return 0;
        }
        if (!ckptReadVal(u32, f, &version, &n_read) or version != slot_ckpt_version or
            !ckptReadVal(u32, f, &count, &n_read))
        {
            slt(.wrn, func, slot, "invalid context checkpoint appendix in '%s' - ignored\n", .{path_z.ptr});
            return 0;
        }
        const ca = slot.prompt.tokens.gpa;
        // std::list<common_prompt_checkpoint>, in order
        var checkpoints: std.ArrayList(task_mod.PromptCheckpoint) = .empty;
        defer {
            for (checkpoints.items) |*cp| cp.clear(ca);
            checkpoints.deinit(ca);
        }
        const n_max: usize = @bitCast(@as(isize, self.params_base.n_ctx_checkpoints));
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            var cur: task_mod.PromptCheckpoint = .{};
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            var bufs = [3][]u8{ &.{}, &.{}, &.{} };
            const ok = ckptReadVal(i64, f, &cur.n_tokens, &n_read) and
                ckptReadVal(c.llama_pos, f, &cur.pos_min, &n_read) and
                ckptReadVal(c.llama_pos, f, &cur.pos_max, &n_read) and
                try ckptReadBuf(f, &bufs[0], n_avail, &n_read) and
                try ckptReadBuf(f, &bufs[1], n_avail, &n_read) and
                try ckptReadBuf(f, &bufs[2], n_avail, &n_read);
            // the buffers are gpa's; the checkpoints hold the slot's allocator's
            cur.data_tgt = try ca.dupe(u8, bufs[0]);
            cur.data_dft = try ca.dupe(u8, bufs[1]);
            cur.data_spec = try ca.dupe(u8, bufs[2]);
            for (bufs) |b| gpa.free(b);
            if (!ok) {
                cur.clear(ca);
                slt(.wrn, func, slot, "truncated context checkpoint appendix in '%s' - ignored\n", .{path_z.ptr});
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.len == 0) {
                cur.clear(ca);
                slt(.wrn, func, slot, "invalid context checkpoint appendix in '%s' - ignored\n", .{path_z.ptr});
                return 0;
            }
            try checkpoints.append(ca, cur);
            if (checkpoints.items.len > n_max) {
                var front = checkpoints.orderedRemove(0);
                front.clear(ca);
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (self.ctx_dft != null and checkpoints.items.len != 0 and checkpoints.items[checkpoints.items.len - 1].data_dft.len != 0) {
            const ok = checkpoints.items[checkpoints.items.len - 1].loadDft(self.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            _ = c.llama_memory_seq_rm(c.llama_get_memory(self.ctx_dft), slot.id, -1, -1);
            if (!ok) {
                slt(.wrn, func, slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", .{path_z.ptr});
                for (checkpoints.items) |*cur| {
                    cur.clearDft(ca);
                }
            }
        }
        // slot.prompt.checkpoints = std::move(checkpoints)
        for (slot.prompt.checkpoints.items) |*cp| cp.clear(ca);
        slot.prompt.checkpoints.clearRetainingCapacity();
        try slot.prompt.checkpoints.appendSlice(ca, checkpoints.items);
        checkpoints.clearRetainingCapacity();
        slt(.inf, func, slot, "restored %zu context checkpoint(s) from '%s'\n", .{ slot.prompt.checkpoints.items.len, path_z.ptr });
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only

    /// Ports `SLOT_DRAFT_MAGIC` and `SLOT_DRAFT_VERSION` (server-context.cpp:2921, 2922 @8642541b2).
    const slot_draft_magic: u32 = 0x54464453; // "SDFT"
    const slot_draft_version: u32 = 1;

    /// Ports `server_context_impl`'s `save_slot_draft` (server-context.cpp:2925
    /// @8642541b2): false if the draft state could not be written completely.
    fn saveSlotDraft(self: *const Impl, filepath: []const u8, slot: *const Slot, n_written: *usize) !bool {
        const func = "save_slot_draft";
        n_written.* = 0;
        var data_dft: []u8 = &.{};
        defer gpa.free(data_dft);
        var data_spec: std.ArrayList(u8) = .empty;
        defer data_spec.deinit(gpa);
        if (self.ctx_dft != null) {
            data_dft = try gpa.alloc(u8, c.llama_state_seq_get_size_ext(self.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_NONE));
            _ = c.llama_state_seq_get_data_ext(self.ctx_dft, data_dft.ptr, data_dft.len, slot.id, c.LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        _ = spec_mod.getState(self.spec, slot.id, &data_spec);
        if (data_dft.len == 0 and data_spec.items.len == 0) {
            return true;
        }
        const path_z = try gpa.dupeSentinel(u8, filepath, 0);
        defer gpa.free(path_z);
        var out: OutFile = .{ .f = fopen(path_z.ptr, "ab") orelse {
            slt(.wrn, func, slot, "failed to append the draft state to '%s'\n", .{path_z.ptr});
            return false;
        } };
        const magic: u32 = slot_draft_magic;
        const version: u32 = slot_draft_version;
        ckptWriteVal(u32, &out, &magic, n_written);
        ckptWriteVal(u32, &out, &version, n_written);
        ckptWriteBuf(&out, data_dft, n_written);
        ckptWriteBuf(&out, data_spec.items, n_written);
        if (!ckptClose(&out)) {
            slt(.wrn, func, slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", .{path_z.ptr});
            return false;
        }
        slt(.inf, func, slot, "appended the draft state (%.3f MiB) to '%s'\n", .{ @as(f64, @as(f32, @floatFromInt(n_written.*)) / 1024 / 1024), path_z.ptr });
        return true;
    }

    /// Ports `server_context_impl`'s `load_slot_draft` (server-context.cpp:2959
    /// @8642541b2): the draft starts empty unless the file holds a draft
    /// state that fits the draft context.
    ///
    /// Return: the number of bytes consumed, 0 if there is no usable draft state.
    fn loadSlotDraft(self: *const Impl, filepath: []const u8, offset: usize, slot: *Slot) !usize {
        const func = "load_slot_draft";
        if (self.ctx_dft != null) {
            _ = c.llama_memory_seq_rm(c.llama_get_memory(self.ctx_dft), slot.id, -1, -1);
        }
        const path_z = try gpa.dupeSentinel(u8, filepath, 0);
        defer gpa.free(path_z);
        const opened = ckptOpenAt(path_z.ptr, offset) orelse return 0;
        const f = opened.f;
        defer _ = fclose(f);
        const n_avail = opened.n_avail;
        var n_read: usize = 0;
        var magic: u32 = 0;
        var version: u32 = 0;
        if (!ckptReadVal(u32, f, &magic, &n_read) or magic != slot_draft_magic) {
            return 0;
        }
        var data_dft: []u8 = &.{};
        defer gpa.free(data_dft);
        var data_spec: []u8 = &.{};
        defer gpa.free(data_spec);
        if (!ckptReadVal(u32, f, &version, &n_read) or version != slot_draft_version or
            !try ckptReadBuf(f, &data_dft, n_avail, &n_read) or
            !try ckptReadBuf(f, &data_spec, n_avail, &n_read))
        {
            slt(.wrn, func, slot, "invalid draft state in '%s' - ignored\n", .{path_z.ptr});
            return 0;
        }
        if (data_dft.len != 0) {
            if (self.ctx_dft == null or
                c.llama_state_seq_set_data_ext(self.ctx_dft, data_dft.ptr, data_dft.len, slot.id, c.LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.len)
            {
                if (self.ctx_dft != null) {
                    _ = c.llama_memory_seq_rm(c.llama_get_memory(self.ctx_dft), slot.id, -1, -1);
                }
                slt(.wrn, func, slot, "draft state in '%s' does not match the draft context - dropped\n", .{path_z.ptr});
                return n_read;
            }
        }
        if (data_spec.len != 0) {
            spec_mod.setState(self.spec, slot.id, data_spec);
        }
        slt(.inf, func, slot, "restored the draft state (%.3f MiB) from '%s'\n", .{ @as(f64, @as(f32, @floatFromInt(n_read)) / 1024 / 1024), path_z.ptr });
        return n_read;
    }

    /// Ports `server_context_impl`'s `process_single_task` (server-context.cpp:3001
    /// @8642541b2): false declines the task, offered again after the decode is
    /// done; otherwise the task is taken.
    fn processSingleTask(self: *Impl, task: *Task, is_yielding: bool) !bool {
        const func = "process_single_task";
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding and task.type != .metrics and task.type != .slot_get) {
            srv(.dbg, func, "decoding, decline task, id_task = %d\n", .{task.id});
            return false;
        }

        // the task is taken: moved somewhere below, or dropped at the end
        var owned = task.*;
        task.* = .{};
        var moved = false;
        defer if (!moved) owned.deinit(gpa);

        switch (owned.type) {
            .completion, .infill, .transcribe_rt, .realtime, .embedding, .rerank, .tts, .decision => blk: {
                if (owned.needSampling() and !self.decision.canGenerate()) {
                    try self.sendErrorTask(&owned, "This model is a decision model without text generation, use /v1/systemone", .not_supported);
                    break :blk;
                }

                // special case: if input is provided via CLI, tokenize it first
                // otherwise, no need to tokenize as it's already done inside the HTTP thread
                if (owned.cli) {
                    if (!try self.tokenizeCliInput(&owned)) {
                        break :blk;
                    }
                }

                const id_task = owned.id;

                const slot_opt = try self.getAvailableSlot(&owned);

                //
                // slot scheduling logic
                //

                const slot = slot_opt orelse {
                    // if no slot is available, we defer this task for processing later
                    srv(.dbg, func, "no slot is available, defer task, id_task = %d\n", .{id_task});
                    moved = true;
                    try self.queue_tasks.deferTask(owned);
                    break :blk;
                };

                if (slot.isProcessing()) {
                    // if requested slot is unavailable, we defer this task for processing later
                    srv(.dbg, func, "requested slot is unavailable, defer task, id_task = %d\n", .{id_task});
                    moved = true;
                    try self.queue_tasks.deferTask(owned);
                    break :blk;
                }

                if (owned.isParent()) {
                    // try getting free slots for all child tasks
                    const n_child_tasks = owned.child_tasks.items.len;
                    const child_slots = try self.getFreeSlots(gpa, n_child_tasks, slot.id);
                    defer gpa.free(child_slots);
                    if (child_slots.len < n_child_tasks) {
                        srv(.dbg, func, "not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", .{ child_slots.len, n_child_tasks, id_task });
                        moved = true;
                        try self.queue_tasks.deferTask(owned);
                        break :blk;
                    }
                    moved = true;
                    if (!try self.launchSlotsWithParentTask(slot, child_slots, owned)) {
                        srv(.err, func, "failed to launch slot with parent task, id_task = %d\n", .{id_task});
                        break :blk; // drop the task
                    }
                } else {
                    moved = true;
                    if (!try self.launchSlotWithTask(slot, owned)) {
                        srv(.err, func, "failed to launch slot with task, id_task = %d\n", .{id_task});
                        break :blk; // drop the task
                    }
                }

                if (self.params_base.cache_idle_slots) {
                    for (self.slots) |*s| {
                        if (!s.isProcessing()) {
                            slt(.trc, func, s, "%s", .{@as([*:0]const u8, "saving idle slot to prompt cache\n")});

                            if (try s.promptSave(self.prompt_cache.?)) {
                                slt(.dbg, func, s, "%s", .{@as([*:0]const u8, "__TEST_TAG_CACHE_IDLE_SLOT__\n")});
                                self.prompt_cache.?.update();
                            }

                            if (self.params_base.kv_unified) {
                                // [TAG_IDLE_SLOT_CLEAR]
                                s.promptClear();
                            }
                        }
                    }
                }
            },
            .cancel => {
                // release slot linked with the task id
                for (self.slots) |*slot| {
                    if (slot.task != null and slot.taskRef().id == owned.id_target) {
                        try slot.release();
                        break;
                    }
                }
            },
            .control => blk: {
                const res = try Result.create(gpa, .control);
                res.id = owned.id;

                const slot = self.getSlotByCmplId(owned.params.control_cmpl_id) orelse {
                    const id = dupZ(owned.params.control_cmpl_id);
                    defer freeZ(id);
                    srv(.wrn, func, "control %d on unknown completion id=%s, no live slot\n", .{ @as(c_int, @backingInt(owned.params.control_action)), id });
                    res.kind.control.success = false;
                    res.kind.control.message = "no active completion for this id";
                    try self.queue_results.send(res);
                    break :blk;
                };

                if (owned.params.control_action == .reasoning_end) {
                    // the budget sampler only exists when reasoning control was armed
                    if (!slot.taskRef().params.sampling.reasoning_control) {
                        res.kind.control.success = false;
                        res.kind.control.message = "reasoning control not enabled for this completion";
                        try self.queue_results.send(res);
                        break :blk;
                    }
                    // act on the live slot mid generation, never defer
                    _ = slot.smpl.?.reasoningBudgetForce();
                    res.kind.control.success = true;
                } else if (slot.rt) |rt| {
                    // a cancel acts now, everything else runs after what is already queued
                    if (owned.params.control_action == .response_cancel) {
                        rt.cancel();
                    } else {
                        try rt.push(&owned);
                    }
                    slot.audio_wait = false;
                    res.kind.control.success = true;
                } else if (owned.params.control_action == .audio_append or
                    owned.params.control_action == .audio_commit or
                    owned.params.control_action == .audio_end)
                {
                    if (slot.audio_stream) |audio_stream| {
                        if (owned.params.control_action == .audio_append) {
                            _ = c.mtmd_audio_stream_push(audio_stream, owned.params.control_audio.ptr, owned.params.control_audio.len);
                        } else {
                            const ret = c.mtmd_audio_stream_flush(audio_stream, owned.params.control_action == .audio_end);
                            // a non-final commit completes its item once the rows of its audio are decoded
                            if (ret == 0 and owned.params.control_action == .audio_commit) {
                                try slot.audio_commit_rows.append(gpa, c.mtmd_audio_stream_get_n_rows_flushed(audio_stream));
                            }
                        }
                        slot.audio_wait = false;
                        res.kind.control.success = true;
                    } else {
                        res.kind.control.success = false;
                        res.kind.control.message = "not a realtime transcription slot";
                    }
                } else {
                    res.kind.control.success = false;
                    res.kind.control.message = "unknown control action";
                }

                try self.queue_results.send(res);
            },
            .next_response => {
                // do nothing
            },
            .metrics => {
                var n_processing_slots: c_int = 0;

                for (self.slots) |*slot| {
                    if (slot.isProcessing()) {
                        n_processing_slots += 1;
                    }
                }
                srv(.dbg, func, "n_processing_slots = %d\n", .{n_processing_slots});

                const res = try Result.create(gpa, .metrics);
                errdefer res.destroy(gpa);
                res.id = owned.id;
                res.kind.metrics.n_processing_slots = n_processing_slots;
                res.kind.metrics.n_tasks_deferred = @intCast(self.queue_tasks.queueTasksDeferredSize());
                res.kind.metrics.metrics = try self.metrics.clone(res.alloc());

                if (owned.metrics_reset_bucket) {
                    self.metrics.resetBucket();
                }
                try self.queue_results.send(res);
            },
            .slot_get => {
                const res = try Result.create(gpa, .slots);
                errdefer res.destroy(gpa);
                const ra = res.alloc();
                var slots_data = Value.emptyArray();

                var n_idle_slots: c_int = 0;

                for (self.slots) |*slot| {
                    if (!slot.isProcessing()) {
                        n_idle_slots += 1;
                    }

                    try slots_data.pushBack(ra, try slot.toJson(ra, self.slots_debug == 0));
                }
                srv(.dbg, func, "n_idle_slots = %d\n", .{n_idle_slots});

                res.id = owned.id;
                res.kind.slots.slots_data = slots_data;
                res.kind.slots.n_idle_slots = n_idle_slots;

                try self.queue_results.send(res);
            },
            .slot_save => blk: {
                const id_slot = owned.slot_action.id_slot;
                const slot = self.getSlotById(id_slot) orelse {
                    try self.sendErrorTask(&owned, "Invalid slot ID", .invalid_request);
                    break :blk;
                };
                if (slot.isProcessing()) {
                    // if requested slot is unavailable, we defer this task for processing later
                    srv(.dbg, func, "requested slot is unavailable, defer task, id_task = %d\n", .{owned.id});
                    moved = true;
                    try self.queue_tasks.deferTask(owned);
                    break :blk;
                }

                const t_start = c.ggml_time_us();

                const filename = owned.slot_action.filename;
                const filepath = owned.slot_action.filepath;

                const packed_bytes = slot.prompt.tokens.serialize(gpa) catch |e| {
                    if (e == error.OutOfMemory) return e;
                    try self.sendErrorTask(&owned, whatOf(e), .not_supported);
                    break :blk;
                };
                defer gpa.free(packed_bytes);

                ggmlAssert(packed_bytes.len % @sizeOf(Token) == 0, 3242, "packed.size() % sizeof(llama_token) == 0");
                const path_z = try gpa.dupeSentinel(u8, filepath, 0);
                defer gpa.free(path_z);
                const nwrite = c.llama_state_seq_save_file(self.ctx_tgt, path_z.ptr, slot.id, @ptrCast(@alignCast(packed_bytes.ptr)), packed_bytes.len / @sizeOf(Token));
                if (nwrite == 0) {
                    try self.sendErrorTask(&owned, "Unable to save slot", .server);
                    break :blk;
                }

                var nwrite_ckpt: usize = 0;
                if (!try self.saveSlotCheckpoints(filepath, slot, &nwrite_ckpt)) {
                    try self.sendErrorTask(&owned, "Unable to save slot: incomplete context checkpoints", .server);
                    break :blk;
                }

                var nwrite_dft: usize = 0;
                if (!try self.saveSlotDraft(filepath, slot, &nwrite_dft)) {
                    try self.sendErrorTask(&owned, "Unable to save slot: incomplete draft state", .server);
                    break :blk;
                }

                const t_end = c.ggml_time_us();
                const t_save_ms = @as(f64, @floatFromInt(t_end - t_start)) / 1000.0;

                const res = try Result.create(gpa, .slot_save_load);
                errdefer res.destroy(gpa);
                res.id = owned.id;
                res.id_slot = id_slot;
                res.kind.slot_save_load.filename = try res.alloc().dupe(u8, filename);
                res.kind.slot_save_load.is_save = true;
                res.kind.slot_save_load.n_tokens = slot.prompt.tokens.size();
                res.kind.slot_save_load.n_bytes = nwrite + nwrite_ckpt + nwrite_dft;
                res.kind.slot_save_load.t_ms = t_save_ms;
                try self.queue_results.send(res);
            },
            .slot_restore => blk: {
                const id_slot = owned.slot_action.id_slot;
                const slot = self.getSlotById(id_slot) orelse {
                    try self.sendErrorTask(&owned, "Invalid slot ID", .invalid_request);
                    break :blk;
                };
                if (slot.isProcessing()) {
                    // if requested slot is unavailable, we defer this task for processing later
                    srv(.dbg, func, "requested slot is unavailable, defer task, id_task = %d\n", .{owned.id});
                    moved = true;
                    try self.queue_tasks.deferTask(owned);
                    break :blk;
                }

                const t_start = c.ggml_time_us();

                const filename = owned.slot_action.filename;
                const filepath = owned.slot_action.filepath;

                const nread = self.restoreSlot(slot, filepath) catch |e| {
                    if (e == error.OutOfMemory) return e;
                    slot.promptClear();
                    var buf: [4200]u8 = undefined;
                    const msg = std.fmt.bufPrint(&buf, "Unable to restore slot: {s}", .{whatOf(e)}) catch &buf;
                    try self.sendErrorTask(&owned, msg, .invalid_request);
                    break :blk;
                };

                // nread is the end offset of the llama state payload within the file
                const nread_ckpt = try self.loadSlotCheckpoints(filepath, nread, slot);
                const nread_dft = try self.loadSlotDraft(filepath, nread + nread_ckpt, slot);

                const t_end = c.ggml_time_us();
                const t_restore_ms = @as(f64, @floatFromInt(t_end - t_start)) / 1000.0;

                const res = try Result.create(gpa, .slot_save_load);
                errdefer res.destroy(gpa);
                res.id = owned.id;
                res.id_slot = id_slot;
                res.kind.slot_save_load.filename = try res.alloc().dupe(u8, filename);
                res.kind.slot_save_load.is_save = false;
                res.kind.slot_save_load.n_tokens = slot.prompt.tokens.size();
                res.kind.slot_save_load.n_bytes = nread + nread_ckpt + nread_dft;
                res.kind.slot_save_load.t_ms = t_restore_ms;
                try self.queue_results.send(res);
            },
            .slot_erase => blk: {
                const id_slot = owned.slot_action.id_slot;
                const slot = self.getSlotById(id_slot) orelse {
                    try self.sendErrorTask(&owned, "Invalid slot ID", .invalid_request);
                    break :blk;
                };
                if (slot.isProcessing()) {
                    // if requested slot is unavailable, we defer this task for processing later
                    srv(.dbg, func, "requested slot is unavailable, defer task, id_task = %d\n", .{owned.id});
                    moved = true;
                    try self.queue_tasks.deferTask(owned);
                    break :blk;
                }

                // Erase token cache
                const n_erased = slot.prompt.tokens.size();

                slot.promptClear();

                const res = try Result.create(gpa, .slot_erase);
                res.id = owned.id;
                res.id_slot = id_slot;
                res.kind.slot_erase.n_erased = n_erased;
                try self.queue_results.send(res);
            },
            .get_lora => {
                // TODO @ngxson : make lora_adapters a dedicated member of server_context
                const loras = self.params_base.lora_adapters.items;
                const res = try Result.create(gpa, .get_lora);
                errdefer res.destroy(gpa);
                const ra = res.alloc();
                res.id = owned.id;
                const out = try ra.alloc(result_mod.GetLora.Lora, loras.len);
                for (loras, 0..) |lora, i| {
                    var alora_invocation_string: std.ArrayList(u8) = .empty;
                    const n_alora_tokens = c.llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                    var alora_invocation_tokens: std.ArrayList(Token) = .empty;
                    if (n_alora_tokens != 0) {
                        const alora_tokens = c.llama_adapter_get_alora_invocation_tokens(lora.ptr);
                        for (0..@intCast(n_alora_tokens)) |j| {
                            try alora_invocation_string.appendSlice(ra, try tc.common.tokenToPiece(ra, self.vocab, alora_tokens[j], true));
                            try alora_invocation_tokens.append(ra, alora_tokens[j]);
                        }
                    }
                    out[i] = .{
                        .info = lora,
                        .alora_invocation_string = alora_invocation_string.items,
                        .alora_invocation_tokens = alora_invocation_tokens.items,
                    };
                }
                res.kind.get_lora.loras = out;
                try self.queue_results.send(res);
            },
            .set_lora => {
                const new_loras = try self.constructLoraList(&owned.set_lora);
                defer gpa.free(new_loras);
                // logging
                for (new_loras, 0..) |l, i| {
                    srv(.trc, func, "set lora adapter idx=%zu scale=%f\n", .{ i, @as(f64, l.scale) });
                }
                // TODO @ngxson : make lora_adapters a dedicated member of server_context
                self.params_base.lora_adapters = .empty;
                try self.params_base.lora_adapters.appendSlice(self.params_base.arena, new_loras);
                const res = try Result.create(gpa, .apply_lora);
                res.id = owned.id;
                try self.queue_results.send(res);
            },
        }

        return true;
    }

    /// The `try` block of the `SLOT_RESTORE` arm of `process_single_task`
    /// (server-context.cpp:3001 @8642541b2).
    ///
    /// Return: the bytes read.
    fn restoreSlot(self: *Impl, slot: *Slot, filepath: []const u8) !usize {
        const path_z = try gpa.dupeSentinel(u8, filepath, 0);
        defer gpa.free(path_z);
        var n_packed: usize = 0;
        var nread = c.llama_state_seq_load_file(self.ctx_tgt, path_z.ptr, slot.id, null, 0, &n_packed);
        var packed_tokens: []Token = &.{};
        defer gpa.free(packed_tokens);
        if (nread != 0) {
            packed_tokens = try gpa.alloc(Token, @max(1, n_packed));
            nread = c.llama_state_seq_load_file(self.ctx_tgt, path_z.ptr, slot.id, packed_tokens.ptr, packed_tokens.len, &n_packed);
        }
        if (nread == 0) {
            return common.throw(error.RuntimeError, "No available space in KV cache or invalid slot save file", .{});
        }

        var restored = try ServerTokens.deserialize(gpa, packed_tokens[0..n_packed], self.mctx != null);
        errdefer restored.deinit();

        if (restored.size() > @as(usize, @intCast(slot.n_ctx))) {
            return common.throw(error.RuntimeError, "Restored prompt does not fit in the slot context", .{});
        }

        if (!restored.validate(self.ctx_tgt)) {
            return common.throw(error.RuntimeError, "Invalid tokens in slot save file", .{});
        }

        slot.prompt.clear();
        slot.prompt.tokens.deinit();
        slot.prompt.tokens = restored;
        return nread;
    }

    /// Ports `server_context_impl`'s `iterate` (server-context.cpp:3415
    /// @8642541b2): an exception from one slot's step is that slot's error.
    fn slotException(self: *Impl, slot: *Slot, e: anyerror) void {
        const msg = gpa.dupe(u8, whatOf(e)) catch "std::bad_alloc";
        defer if (msg.ptr != "std::bad_alloc".ptr) gpa.free(msg);
        {
            const z = dupZ(msg);
            defer freeZ(z);
            slt(.err, "iterate", slot, "got exception: %s\n", .{z});
        }
        var buf: [4200]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "got exception: {s}", .{msg}) catch &buf;
        self.sendErrorSlot(slot, text, .server) catch |err| switch (err) {
            error.OutOfMemory, error.InvalidArgument, error.RuntimeError, error.JsonError => {},
        };
        slot.release() catch |err| switch (err) {
            error.OutOfMemory => {},
        };
    }

    /// Ports `server_context_impl`'s `abort_all_slots` (server-context.cpp:3439 @8642541b2).
    fn abortAllSlots(self: *Impl, reason: []const u8) !void {
        for (self.slots) |*slot| {
            if (slot.isProcessing()) {
                try self.sendErrorSlot(slot, reason, .server);
                try slot.release();
            }
        }
    }

    /// Ports `server_context_impl`'s `slot_audio_step` (server-context.cpp:3480
    /// @8642541b2): fill `slot.inp_embd` with the next rows of the audio
    /// stream.
    ///
    /// Return: false if the slot has nothing to decode now.
    fn slotAudioStep(self: *Impl, slot: *Slot) !bool {
        if (slot.inp_embd.items.len != 0) {
            return true; // rows from a previous step, not decoded yet
        }

        const audio_stream = slot.audio_stream.?;
        const n = c.mtmd_audio_stream_step(audio_stream, slot.sampled);

        // complete the commits whose rows have their text sampled, before the new rows: their text goes to the next item
        const n_rows = c.mtmd_audio_stream_get_n_rows(audio_stream);
        while (slot.audio_commit_rows.items.len != 0 and slot.audio_commit_rows.items[0] <= n_rows) {
            _ = slot.audio_commit_rows.orderedRemove(0);
            const res = try Result.create(gpa, .cmpl_partial);
            errdefer res.destroy(gpa);
            const ra = res.alloc();
            const p = &res.kind.cmpl_partial;
            const t = slot.taskRef();
            res.id = t.id;
            res.index = t.index;
            p.n_decoded = @intCast(slot.stats.n_gen);
            p.n_prompt_tokens = t.nTokens();
            p.res_type = t.params.res_type;
            p.oaicompat_model = try ra.dupe(u8, t.params.oaicompat_model);
            p.oaicompat_cmpl_id = try ra.dupe(u8, t.params.oaicompat_cmpl_id);
            p.audio_commit_drained = true;
            try self.queue_results.send(res);
        }

        if (n > 0) {
            // new sequence (e.g. live ASR restart): drop the old one, the text goes on in a new line
            if (c.mtmd_audio_stream_is_seq_start(audio_stream)) {
                slot.promptClear();
                if (slot.n_sent_text > 0) {
                    var sep: CompletionTokenOutput = .{};
                    sep.tok = token_null;
                    sep.text_to_send = "\n";
                    {
                        var chunk = try chat.input.Input.fromText(gpa, sep.text_to_send);
                        defer chunk.deinit();
                        try slot.generated.appendInput(&chunk);
                    }
                    slot.n_sent_text += sep.text_to_send.len;
                    try self.sendPartialResponse(slot, &sep, false, false);
                }
            }
            const rows = c.mtmd_audio_stream_get_output_embd(audio_stream);
            slot.inp_embd.clearRetainingCapacity();
            try slot.inp_embd.appendSlice(gpa, rows[0 .. @as(usize, @intCast(n)) * @as(usize, @intCast(self.batch.n_embd))]);
            slot.audio_wait = false;
            return true;
        }
        if (n == 0) {
            slot.audio_wait = true;
            return false;
        }
        if (n == -2) {
            try self.sendErrorSlot(slot, "audio stream error", .server);
            try slot.release();
            return false;
        }

        // end of stream
        slot.stop = .eos;
        slot.has_next_token = false;
        try slot.printTimings();
        try self.sendFinalResponse(slot);
        try slot.release();
        return false;
    }

    /// Ports `server_context_impl`'s `update_slots` (server-context.cpp:3540 @8642541b2).
    fn updateSlots(self: *Impl) !void {
        const func = "update_slots";
        // check if all slots are idle
        // a realtime slot waiting for audio has no work, the next audio control task wakes the loop
        {
            var all_idle = true;

            for (self.slots) |*slot| {
                if (slot.isProcessing() and !slot.audio_wait) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                srv(.trc, func, "%s", .{@as([*:0]const u8, "all slots are idle\n")});

                self.metricsFlushIdle();

                return; // skip further processing
            } else {
                srv(.dbg, func, "%s", .{@as([*:0]const u8, "posting NEXT_RESPONSE\n")});

                var task = Task.initType(.next_response);
                task.id = self.queue_tasks.getNewId();
                _ = try self.queue_tasks.post(task, false);
            }
        }

        self.preDecode() catch |e| {
            if (e == error.OutOfMemory) return e;
            const msg = try gpa.dupe(u8, whatOf(e));
            defer gpa.free(msg);
            const z = dupZ(msg);
            defer freeZ(z);
            srv(.err, func, "pre_decode() failed: %s\n", .{z});
            var buf: [4200]u8 = undefined;
            try self.abortAllSlots(std.fmt.bufPrint(&buf, "pre_decode() failed: {s}", .{msg}) catch &buf);

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        };

        // note: TTS slots bypass the shared batch entirely
        self.processTtsSlots() catch |e| {
            if (e == error.OutOfMemory) return e;
            const msg = try gpa.dupe(u8, whatOf(e));
            defer gpa.free(msg);
            const z = dupZ(msg);
            defer freeZ(z);
            srv(.err, func, "process_tts_slots() failed: %s\n", .{z});
            var buf: [4200]u8 = undefined;
            try self.abortAllSlots(std.fmt.bufPrint(&buf, "process_tts_slots() failed: {s}", .{msg}) catch &buf);
        };

        // so do the realtime conversations
        self.processRealtimeSlots() catch |e| {
            if (e == error.OutOfMemory) return e;
            const msg = try gpa.dupe(u8, whatOf(e));
            defer gpa.free(msg);
            const z = dupZ(msg);
            defer freeZ(z);
            srv(.err, func, "process_realtime_slots() failed: %s\n", .{z});
            var buf: [4200]u8 = undefined;
            try self.abortAllSlots(std.fmt.bufPrint(&buf, "process_realtime_slots() failed: {s}", .{msg}) catch &buf);
        };

        ggmlAssert(self.batch.slot_batched != null or self.batch.size() == 0, 3609, "batch.slot_batched || batch.size() == 0");

        if (self.batch.slot_batched) |slot_batched| {
            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            try tc.common.setAdapterLora(gpa, self.ctx_tgt, slot_batched.lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (self.batch.alora_scale > 0.0) {
                srv(.dbg, func, "re-enabling alora with scale %f\n", .{@as(f64, self.batch.alora_scale)});
                slot_batched.lora[self.batch.alora_disabled_id].scale = self.batch.alora_scale;
            }

            c.llama_set_embeddings(self.ctx_tgt, slot_batched.needEmbd());
        }

        var off_next: i32 = 0;
        var n_batch: i32 = @intCast(c.llama_n_batch(self.ctx_tgt));
        var off: i32 = 0;
        while (off < self.batch.size()) : (off = off_next) {
            const n_tokens = @min(n_batch, self.batch.size() - off);
            {
                self.batch.render(off, n_tokens);
                const ok = self.decode(&n_batch, off) catch |e| {
                    if (e == error.OutOfMemory) return e;
                    const msg = try gpa.dupe(u8, whatOf(e));
                    defer gpa.free(msg);
                    const z = dupZ(msg);
                    defer freeZ(z);
                    srv(.err, func, "decode() failed: %s\n", .{z});
                    var buf: [4200]u8 = undefined;
                    try self.abortAllSlots(std.fmt.bufPrint(&buf, "decode() failed: {s}", .{msg}) catch &buf);
                    break; // stop any further processing
                };

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = @intCast(c.llama_n_batch(self.ctx_tgt));
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            }

            self.postDecode(n_tokens, off) catch |e| {
                if (e == error.OutOfMemory) return e;
                const msg = try gpa.dupe(u8, whatOf(e));
                defer gpa.free(msg);
                const z = dupZ(msg);
                defer freeZ(z);
                srv(.err, func, "post_decode() failed: %s\n", .{z});
                var buf: [4200]u8 = undefined;
                try self.abortAllSlots(std.fmt.bufPrint(&buf, "post_decode() failed: {s}", .{msg}) catch &buf);
                break; // stop any further processing
            };
        }
    }

    /// Ports `server_context_impl`'s `process_realtime_slots`
    /// (server-context.cpp:3672 @8642541b2): realtime conversations
    /// (SERVER_TASK_TYPE_REALTIME) drive their own decode loop, see
    /// `realtime.RealtimeSlot`.
    fn processRealtimeSlots(self: *Impl) !void {
        for (self.slots) |*slot| {
            self.realtimeSlot(slot) catch |e| self.slotException(slot, e);
        }
    }

    fn realtimeSlot(self: *Impl, slot: *Slot) !void {
        if (!slot.isProcessing() or slot.rt == null) {
            return;
        }
        const rt = slot.rt.?;
        if (!try rt.update()) {
            try self.sendErrorSlot(slot, rt.err_msg, rt.err_type);
            try slot.release();
            return;
        }
        slot.audio_wait = rt.isIdle();
    }

    /// Ports `server_context_impl`'s `process_tts_slots` (server-context.cpp:3686 @8642541b2).
    fn processTtsSlots(self: *Impl) !void {
        for (self.slots) |*slot| {
            self.ttsSlot(slot) catch |e| self.slotException(slot, e);
        }
    }

    fn ttsSlot(self: *Impl, slot: *Slot) !void {
        if (!slot.isProcessing() or slot.taskRef().type != .tts) {
            return;
        }

        c.llama_set_embeddings(self.ctx_tgt, true);

        if (slot.state == .processing_prompt) {
            const ret = c.mtmd_helper_gen_audio_step_prompt(slot.tts.ctx, @intCast(c.llama_n_batch(self.ctx_tgt)));
            if (ret < 0) {
                try self.sendErrorSlot(slot, "TTS prompt processing failed", .server);
                try slot.release();
            } else if (ret == 0) {
                // done prompt, do sample and save h_state_prompt
                slot.tts.sampled = try slot.smpl.?.sample(self.ctx_tgt, -1, false);
                slot.smpl.?.accept(slot.tts.sampled, true);
                const h_embd = c.llama_get_embeddings_ith(self.ctx_tgt, -1);
                slot.tts.h_state_prompt.clearRetainingCapacity();
                try slot.tts.h_state_prompt.appendSlice(gpa, h_embd[0..@intCast(c.llama_model_n_embd(self.model_tgt))]);
                slot.tts.h_state = slot.tts.h_state_prompt.items.ptr;
                slot.state = .generating;
            }
            return;
        }

        const n_predict: i32 = if (slot.n_predict_max > 0) slot.n_predict_max else 512;
        if (slot.tts.stop or slot.tts.n_decoded >= n_predict) {
            var sample_rate: i32 = 0;
            var data: [*c]const u8 = null;
            var data_len: usize = 0;
            // generation truly ends here: force out any sub-window remainder still buffered
            if (c.mtmd_helper_gen_audio_flush(slot.tts.ctx) != 0 or c.mtmd_helper_gen_audio_get_output(slot.tts.ctx, &sample_rate, @ptrCast(&data), &data_len, null) != 0) {
                try self.sendErrorSlot(slot, "failed to finalize TTS output", .server);
            } else {
                try self.sendTtsResult(slot, sample_rate, data, data_len, true);
            }
            try slot.release();
            return;
        }

        var h_state_next: [*c]const f32 = null;
        var stop = false;
        if (c.mtmd_helper_gen_audio_step_gen(slot.tts.ctx, slot.tts.sampled, slot.tts.h_state, &h_state_next, &stop) != 0) {
            try self.sendErrorSlot(slot, "TTS generation failed", .server);
            try slot.release();
            return;
        }
        if (h_state_next == null) {
            // end-of-speech without a new frame (e.g. pocket-tts eos head)
            slot.tts.stop = true;
            return;
        }
        slot.tts.h_state = h_state_next;
        slot.tts.n_decoded += 1;
        slot.tts.stop = stop;

        if (!stop) {
            slot.tts.sampled = try slot.smpl.?.sample(self.ctx_tgt, -1, false);
            slot.smpl.?.accept(slot.tts.sampled, true);
        }

        if (slot.taskRef().params.stream) {
            var sample_rate: i32 = 0;
            var data: [*c]const u8 = null;
            var data_len: usize = 0;
            if (c.mtmd_helper_gen_audio_get_output(slot.tts.ctx, &sample_rate, @ptrCast(&data), &data_len, null) != 0) {
                try self.sendErrorSlot(slot, "TTS streaming output failed", .server);
                try slot.release();
            } else if (data_len > 0) {
                try self.sendTtsResult(slot, sample_rate, data, data_len, false);
            }
        }
    }

    /// The context-shift step of `pre_decode` (server-context.cpp:3761
    /// @8642541b2), one slot.
    fn contextShift(self: *Impl, slot: *Slot) !void {
        if (slot.task) |*t| {
            if (t.type == .tts or t.type == .realtime) {
                // TTS and realtime conversation slots drive their own decode loop, never enter the shared batch
                return;
            }
        }
        if (slot.state == .generating and slot.prompt.nTokens() + 1 >= slot.n_ctx) {
            if (!self.params_base.ctx_shift) {
                // this check is redundant (for good)
                // we should never get here, because generation should already stopped in process_token()
                try self.sendErrorSlot(slot, "context shift is disabled", .server);
                try slot.release();
                return;
            }

            if (self.mctx != null) {
                // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                // we don't support ctx_shift because an image chunk may contains multiple tokens
                ggmlAbort(3781, "not supported by multimodal", .{});
            }

            if (slot.taskRef().isParent() or slot.taskRef().isChild()) {
                try self.sendErrorSlot(slot, "context shift cannot be used for shared prompt", .server);
                try slot.release();
                return;
            }

            // Shift context
            var n_keep: i32 = if (slot.taskRef().params.n_keep < 0) slot.taskRef().nTokens() else slot.taskRef().params.n_keep;

            if (self.add_bos_token) {
                n_keep += 1;
            }

            n_keep = @min(slot.n_ctx - 4, n_keep);

            const n_left = slot.prompt.nTokens() - n_keep;
            var n_discard: i32 = if (slot.taskRef().params.n_discard != 0) slot.taskRef().params.n_discard else @divTrunc(n_left, 2);

            // ref: https://github.com/ggml-org/llama.cpp/pull/24786
            n_discard = std.math.clamp(n_discard, 0, @max(0, n_left - 1));

            slt(.wrn, "operator()", slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", .{ n_keep, n_left, n_discard });

            slot.mem.seqRm(slot.id, n_keep, n_keep + n_discard);
            slot.mem.seqAdd(slot.id, n_keep + n_discard, slot.prompt.tokens.posNext(-1), -n_discard);

            // add generated tokens to cache
            // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
            {
                ggmlAssert(!slot.prompt.tokens.has_mtmd, 3813, "!slot.prompt.tokens.has_mtmd");

                const new_tokens = try gpa.dupe(Token, slot.prompt.tokens.getTokens()); // copy
                defer gpa.free(new_tokens);
                const nk: usize = @intCast(n_keep);
                const nd: usize = @intCast(n_discard);
                var i = nk + nd;
                while (i < new_tokens.len) : (i += 1) {
                    new_tokens[i - nd] = new_tokens[i];
                }

                const new_len = slot.prompt.tokens.size() - nd;

                slot.prompt.clear();
                try slot.prompt.tokens.insert(new_tokens[0..new_len]);
            }

            slot.truncated = true;
        }
    }

    /// The drafting step of `pre_decode` (server-context.cpp:3761
    /// @8642541b2), one slot: whether it generates in this batch, and
    /// whether it drafts first.
    fn selectGenerating(self: *Impl, slot: *Slot, generating: *std.ArrayList(*Slot), drafting: *std.ArrayList(*Slot)) !void {
        if (slot.state != .generating or slot.taskRef().type == .tts or slot.taskRef().type == .realtime) {
            return;
        }

        if (slot.audio_stream != null and !try self.slotAudioStep(slot)) {
            return; // waiting for audio, or the stream ended
        }

        // check if we can batch this slot with the previous one
        if (self.batch.slot_batched == null) {
            self.batch.slot_batched = slot;
        } else if (!self.batch.slot_batched.?.canBatchWith(slot)) {
            return;
        }

        try generating.append(gpa, slot);

        if (self.spec != null) {
            spec_mod.getDraftParams(self.spec, slot.id).drafting = false;

            const use_ckpt_tgt = self.ctx_tgt_seq_rm_type == .full;
            const use_ckpt_dft = self.ctx_dft_seq_rm_type == .full;

            const n_draft_max = slot.getNDraftMax();

            if (n_draft_max > 0) {
                ggmlAssert(slot.canSpeculate(), 3867, "slot.can_speculate()");

                // stale candidates: a replay never reads them, a new draft refills them
                clearDraftQ(&slot.spec_draft_q);

                if (slot.spec_draft.items.len != 0) {
                    // we have a previous (partial) draft to reuse
                    if (use_ckpt_tgt) {
                        ggmlAssert(!slot.spec_ckpt.empty(), 3875, "!slot.spec_ckpt.empty()");
                    }
                } else {
                    ggmlAssert(slot.spec_i_batch.items.len == 0, 3878, "slot.spec_i_batch.empty()");

                    slot.spec_ckpt.updatePos(
                        slot.prompt.nTokens(),
                        c.llama_memory_seq_pos_min(c.llama_get_memory(self.ctx_tgt), slot.id),
                        c.llama_memory_seq_pos_max(c.llama_get_memory(self.ctx_tgt), slot.id),
                    );

                    if (use_ckpt_dft) {
                        try slot.spec_ckpt.updateDft(gpa, self.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }

                    gpa.free(slot.spec_prompt);
                    slot.spec_prompt = try slot.prompt.tokens.getTextTokens(gpa);

                    const spec_reject = slot.useSpecRejection();

                    spec_mod.getDraftParams(self.spec, slot.id).* = .{
                        .drafting = true,
                        .n_max = n_draft_max,
                        .pos0 = slot.prompt.tokens.posNext(-1),
                        .id_last = slot.sampled,
                        .prompt = slot.spec_prompt,
                        .result = &slot.spec_draft,
                        .result_q = if (spec_reject) &slot.spec_draft_q else null,
                        .temp = slot.taskRef().params.sampling.temp,
                        .seed = slot.taskRef().params.sampling.seed,
                    };

                    try drafting.append(gpa, slot);
                }
            }
        }
    }

    /// The checkpoint step of `pre_decode` (server-context.cpp:3761
    /// @8642541b2), one drafting slot.
    fn draftCheckpoint(self: *Impl, slot: *Slot) !void {
        const draft = &slot.spec_draft;
        const ckpt = &slot.spec_ckpt;

        slot.stats.n_draft_tokens += draft.items.len;

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        const use_ckpt_dft_full = self.ctx_dft_seq_rm_type == .full;

        if (self.ctx_dft != null) {
            if (use_ckpt_dft_full) {
                ggmlAssert(ckpt.loadDft(self.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY), 3930, "ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)");
            }

            if (!c.llama_memory_seq_rm(c.llama_get_memory(self.ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                ggmlAbort(3934, "failed to remove sequence %d\n", .{slot.id});
            }
        }

        if (draft.items.len != 0) {
            const use_ckpt_tgt = self.ctx_tgt_seq_rm_type == .full or
                (self.ctx_tgt_seq_rm_type == .rs and draft.items.len > c.llama_n_rs_seq(self.ctx_tgt));

            const use_ckpt_dft = (self.ctx_dft_seq_rm_type == .rs and draft.items.len > c.llama_n_rs_seq(self.ctx_dft));

            if (use_ckpt_tgt) {
                try ckpt.updateTgt(gpa, self.ctx_tgt, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                slt(.dbg, "operator()", slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n", .{
                    ckpt.pos_min,                                                 ckpt.pos_max,                                                       slot.prompt.nTokens(),
                    @as(f64, @as(f32, @floatFromInt(ckpt.size())) / 1024 / 1024), @as(f64, @as(f32, @floatFromInt(ckpt.data_dft.len)) / 1024 / 1024),
                });
            }

            if (use_ckpt_dft) {
                try ckpt.updateDft(gpa, self.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }
    }

    fn draftWork(self: *Impl) anyerror!void {
        spec_mod.draft(self.spec);
    }

    /// Ports `server_context_impl`'s `pre_decode` (server-context.cpp:3761
    /// @8642541b2): context shifts, drafts, and the batch -- the sampled and
    /// drafted tokens, then pending prompts up to `n_batch`.
    fn preDecode(self: *Impl) !void {
        // apply context-shift if needed
        // TODO: simplify and improve
        for (self.slots) |*slot| {
            self.contextShift(slot) catch |e| self.slotException(slot, e);
        }

        // start populating the batch for this iteration
        self.batch.clear();

        var generating: std.ArrayList(*Slot) = .empty;
        defer generating.deinit(gpa);
        var drafting: std.ArrayList(*Slot) = .empty;
        defer drafting.deinit(gpa);

        // determine which slots are generating and drafting
        for (self.slots) |*slot| {
            self.selectGenerating(slot, &generating, &drafting) catch |e| self.slotException(slot, e);
        }

        // generate the actual drafts (if any)
        if (drafting.items.len != 0) {
            try self.queue_tasks.yieldToQueue(self, draftWork);
        }

        // make checkpoints if needed
        for (drafting.items) |slot| {
            self.draftCheckpoint(slot) catch |e| self.slotException(slot, e);
        }

        // update the batch with the sampled/drafted tokens
        for (generating.items) |slot| {
            slot.handleLastSampledToken(&self.batch) catch |e| self.slotException(slot, e);
        }

        // process in chunks of params.n_batch
        const n_batch: i32 = @intCast(c.llama_n_batch(self.ctx_tgt));
        const n_ubatch: i32 = @intCast(c.llama_n_ubatch(self.ctx_tgt));

        // next, batch any pending prompts without exceeding n_batch
        if (self.params_base.cont_batching or self.batch.size() == 0) {
            var add_ok = true; // false means the batch is full, skip remaining slots

            for (self.slots) |*slot| {
                self.batchPrompt(slot, n_batch, n_ubatch, &add_ok) catch |e| self.slotException(slot, e);
            }
        }
    }

    const MtmdWork = struct {
        slot: *Slot,
        idx: usize,
        n_tokens_out: usize = 0,
        res: i32 = 0,

        fn run(w: *MtmdWork) anyerror!void {
            w.res = try processMtmdChunk(w.slot, &w.slot.mbatch, w.idx, &w.n_tokens_out);
        }
    };

    /// The prompt step of `pre_decode` (server-context.cpp:3761 @8642541b2),
    /// one slot: a new prompt's cache reuse and checkpoint restore, then its
    /// next tokens into the batch.
    fn batchPrompt(self: *Impl, slot: *Slot, n_batch: i32, n_ubatch: i32, add_ok: *bool) !void {
        const func = "operator()";
        const batch = &self.batch;
        const alora_scale = &batch.alora_scale;
        const alora_disabled_id = &batch.alora_disabled_id;

        if (!add_ok.* or batch.size() >= n_batch) {
            return; // batch is full, skip remaining slots
        }

        if (!slot.isProcessing() or slot.taskRef().type == .tts or slot.taskRef().type == .realtime) {
            return;
        }

        // check if we can batch this slot with the previous one
        if (batch.slot_batched != null and !batch.slot_batched.?.canBatchWith(slot)) {
            return;
        }

        // check if this is a child slot
        if (slot.state == .wait_other) {
            slt(.dbg, func, slot, "%s", .{@as([*:0]const u8, "waiting for parent slot to complete\n")});
            return;
        }

        // no prompt, the rows come from the audio stream
        if (slot.taskRef().type == .transcribe_rt) {
            return;
        }

        // this slot still has a prompt to be processed
        if (slot.state == .processing_prompt or slot.state == .started) {
            const input_tokens = &slot.task.?.tokens;

            // used to determine the number of tokens added to the batch for the current slot
            const n_tokens_prev = batch.size();

            // TODO: maybe move branch to outside of this loop in the future
            if (slot.state == .started) {
                slot.stats.updatePromptStart();

                slot.state = .processing_prompt;

                slt(.trc, func, slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n", .{ slot.n_ctx, slot.taskRef().params.n_keep, slot.taskRef().nTokens() });

                // keep track how many tokens we can reuse from the previous state
                var n_past: i32 = 0;

                // empty prompt passed -> release the slot and send empty response
                if (input_tokens.empty()) {
                    slt(.wrn, func, slot, "%s", .{@as([*:0]const u8, "empty prompt - releasing slot\n")});

                    try slot.printTimings();
                    try self.sendFinalResponse(slot);
                    try slot.release();

                    return;
                }

                // TODO: support memory-less logits computation
                // a decision reads its logits from the one batch of its prompt
                if (slot.taskRef().needLogits() and c.llama_get_memory(self.ctx_tgt) == null and slot.taskRef().type != .decision) {
                    try self.sendErrorSlot(slot, "the current context does not support logits computation. skipping", .server);
                    try slot.release();
                    return;
                }

                if (!slot.canSplit()) {
                    if (slot.taskRef().nTokens() > n_ubatch) {
                        var buf: [256]u8 = undefined;
                        const msg = std.fmt.bufPrint(&buf, "input ({d} tokens) is too large to process. increase the physical batch " ++
                            "size (current batch size: {d})", .{ slot.taskRef().nTokens(), n_ubatch }) catch |err| switch (err) {
                            error.NoSpaceLeft => unreachable,
                        };
                        try self.sendErrorSlot(slot, msg, .server);
                        try slot.release();
                        return;
                    }

                    if (slot.taskRef().nTokens() > slot.n_ctx) {
                        var buf: [256]u8 = undefined;
                        const msg = std.fmt.bufPrint(&buf, "input ({d} tokens) is larger than the max context size ({d} tokens). skipping", .{ slot.taskRef().nTokens(), slot.n_ctx }) catch |err| switch (err) {
                            error.NoSpaceLeft => unreachable,
                        };
                        try self.sendErrorSlot(slot, msg, .exceed_context_size);
                        try slot.release();
                        return;
                    }
                } else {
                    if (slot.taskRef().nTokens() >= slot.n_ctx) {
                        var buf: [256]u8 = undefined;
                        const msg = std.fmt.bufPrint(&buf, "request ({d} tokens) exceeds the available context size ({d} " ++
                            "tokens), try increasing it", .{ slot.taskRef().nTokens(), slot.n_ctx }) catch |err| switch (err) {
                            error.NoSpaceLeft => unreachable,
                        };
                        try self.sendErrorSlot(slot, msg, .exceed_context_size);
                        try slot.release();
                        return;
                    }

                    // the outputs of a decision are read from one batch
                    const n_decision_first: i32 = if (slot.taskRef().type == .decision) slot.taskRef().decision.posFirst() else -1;
                    if (n_decision_first >= 0 and slot.taskRef().nTokens() - n_decision_first > n_batch) {
                        var buf: [256]u8 = undefined;
                        const msg = std.fmt.bufPrint(&buf, "the question and its options ({d} tokens) are too large to process. " ++
                            "increase the batch size (current batch size: {d})", .{ slot.taskRef().nTokens() - n_decision_first, n_batch }) catch |err| switch (err) {
                            error.NoSpaceLeft => unreachable,
                        };
                        try self.sendErrorSlot(slot, msg, .invalid_request);
                        try slot.release();
                        return;
                    }

                    if (slot.taskRef().params.cache_prompt and !slot.taskRef().isStateless()) {
                        // reuse any previously computed tokens that are common with the new prompt
                        n_past = @intCast(try slot.prompt.tokens.getCommonPrefix(input_tokens));

                        // the children start from the shared prefix, do not go past it
                        if (slot.taskRef().n_tokens_shared > 0) {
                            n_past = @min(n_past, slot.taskRef().n_tokens_shared);
                        }

                        // the outputs of a decision are read from the batch, so its first position must be evaluated
                        if (n_decision_first >= 0) {
                            n_past = @min(n_past, n_decision_first);
                        }

                        // if there is an alora invoked, don't cache after the invocation start
                        if (slot.alora_invocation_start > 0) {
                            slt(.dbg, func, slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", .{ n_past, slot.alora_invocation_start });
                            n_past = @min(n_past, slot.alora_invocation_start - 1);
                        }

                        const n_cache_reuse = slot.taskRef().params.n_cache_reuse;

                        const can_cache_reuse =
                            c.llama_memory_can_shift(c.llama_get_memory(self.ctx_tgt)) and
                            !slot.prompt.tokens.has_mtmd;

                        if (!can_cache_reuse and n_cache_reuse > 0) {
                            slt(.wrn, func, slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", .{n_cache_reuse});
                        }

                        // reuse chunks from the cached prompt by shifting their KV cache in the new position
                        if (can_cache_reuse and n_cache_reuse > 0) {
                            ggmlAssert(!slot.prompt.tokens.has_mtmd, 4135, "!slot.prompt.tokens.has_mtmd");

                            var head_c: usize = @intCast(n_past); // cache
                            var head_p: usize = @intCast(n_past); // current prompt

                            if (self.mctx != null) {
                                // we should never reach this
                                ggmlAbort(4142, "not supported by multimodal", .{});
                            }

                            slt(.dbg, func, slot, "trying to reuse chunks with size > %d, n_past = %d\n", .{ n_cache_reuse, n_past });

                            while (head_c < slot.prompt.tokens.size() and
                                head_p < input_tokens.size())
                            {
                                var n_match: usize = 0;
                                while (head_c + n_match < slot.prompt.tokens.size() and
                                    head_p + n_match < input_tokens.size() and
                                    slot.prompt.tokens.at(head_c + n_match) == input_tokens.at(head_p + n_match))
                                {
                                    n_match += 1;
                                }

                                if (n_match >= @as(usize, @intCast(n_cache_reuse))) {
                                    slt(.trc, func, slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", .{ n_match, head_c, head_c + n_match, head_p, head_p + n_match });

                                    const kv_shift: i64 = @as(i64, @intCast(head_p)) - @as(i64, @intCast(head_c));

                                    slot.mem.seqRm(slot.id, @intCast(head_p), @intCast(head_c));
                                    slot.mem.seqAdd(slot.id, @intCast(head_c), @intCast(head_c + n_match), @truncate(kv_shift));

                                    for (0..n_match) |i| {
                                        slot.prompt.tokens.setToken(@intCast(head_p + i), slot.prompt.tokens.at(head_c + i));
                                        n_past += 1;
                                    }

                                    head_c += n_match;
                                    head_p += n_match;
                                } else {
                                    head_c += 1;
                                }
                            }

                            slt(.dbg, func, slot, "after context reuse, new n_past = %d\n", .{n_past});
                        }
                    } else {
                        // if we don't cache the prompt, we have to remove all previous tokens
                        n_past = 0;
                    }

                    var pos_next = slot.prompt.tokens.posNext(n_past);

                    // number of cached tokens shared with the new prompt, before any checkpoint is restored
                    const n_prefix = n_past;

                    // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                    const has_new_tokens = (n_past < slot.taskRef().nTokens());

                    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
                    const pos_eval: c.llama_pos = pos_next - @as(i32, if (has_new_tokens) 0 else 1);

                    // a state or a checkpoint is useful when its pos_min is below this threshold
                    const pos_min_thold: i32 = @max(0, if (self.n_swa > 0) pos_eval - self.n_swa + 2 else pos_eval);

                    if (n_past > 0 and n_past <= slot.prompt.nTokens()) {
                        const pos_min = c.llama_memory_seq_pos_min(c.llama_get_memory(self.ctx_tgt), slot.id);
                        if (pos_min == -1) {
                            slt(.err, func, slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", .{ n_past, @as(c_int, @intCast(slot.prompt.tokens.size())), slot.id, pos_min });
                            ggmlAbort(4206, "pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237", .{});
                        }

                        // when the prompt prefix does not match, print the tokens around the mismatch
                        // this is useful for debugging prompt caching
                        if (self.slots_debug != 0) {
                            try self.printPrefixMismatch(slot, n_past);
                        }

                        if (pos_min >= pos_min_thold) {
                            // search for a context checkpoint
                            var found: ?*PromptCheckpoint = null;
                            {
                                var k = slot.prompt.checkpoints.items.len;
                                while (k > 0) : (k -= 1) {
                                    const cur = &slot.prompt.checkpoints.items[k - 1];
                                    // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                    slt(.trc, func, slot, "checking checkpoint with [%d, %d] against %d...\n", .{ cur.pos_min, cur.pos_max, pos_min_thold });
                                    // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                    if (cur.pos_max > pos_next) {
                                        continue;
                                    }
                                    if (cur.pos_min < pos_min_thold or cur.pos_min == 0) {
                                        found = cur;
                                        break;
                                    }
                                }
                            }

                            var do_reset = found == null;

                            if (!do_reset) {
                                const it = found.?;
                                // restore the context checkpoint
                                if (!it.loadTgt(self.ctx_tgt, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) or
                                    !it.loadDft(self.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY))
                                {
                                    if (it.id_task != -1) {
                                        ggmlAbort(4275, "failed to restore context checkpoint\n", .{});
                                    }
                                    // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                    slt(.wrn, func, slot, "%s", .{@as([*:0]const u8, "failed to load context checkpoint restored from a slot file\n")});
                                    do_reset = true;
                                }
                            }

                            if (!do_reset) {
                                const it = found.?;
                                // restore the draft's speculative state
                                spec_mod.setState(self.spec, slot.id, it.data_spec);

                                pos_next = @min(pos_next, @max(it.pos_min + 1, it.pos_max));
                                n_past = @intCast(@min(slot.prompt.tokens.sizeUpToPos(pos_next), @as(usize, @intCast(it.n_tokens))));
                                slt(.trc, func, slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %lld, n_past = %d, size = %.3f MiB)\n", .{
                                    it.pos_min, it.pos_max, @as(c_longlong, it.n_tokens), n_past, @as(f64, @as(f32, @floatFromInt(it.size())) / 1024 / 1024),
                                });
                            }

                            if (do_reset) {
                                slt(.trc, func, slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n", .{@as([*:0]const u8, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055")});
                                pos_next = 0;
                                n_past = 0;
                            }
                        }
                    }

                    {
                        // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                        var k: usize = 0;
                        while (k < slot.prompt.checkpoints.items.len) {
                            const cur = &slot.prompt.checkpoints.items[k];
                            if (cur.pos_max > pos_next or cur.n_tokens > n_prefix) {
                                slt(.trc, func, slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %lld, n_swa = %d, pos_next = %d, size = %.3f MiB)\n", .{
                                    cur.pos_min, cur.pos_max, @as(c_longlong, cur.n_tokens), self.n_swa, pos_next, @as(f64, @as(f32, @floatFromInt(cur.size())) / 1024 / 1024),
                                });
                                var removed = slot.prompt.checkpoints.orderedRemove(k);
                                removed.clear(slot.prompt.tokens.gpa);
                            } else {
                                k += 1;
                            }
                        }
                    }
                }

                // [TAG_PROMPT_LOGITS]
                if (n_past == slot.taskRef().nTokens() and n_past > 0) {
                    slt(.wrn, func, slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", .{ n_past, slot.taskRef().nTokens() });
                    n_past -= 1;
                    slt(.wrn, func, slot, "n_past was set to %d\n", .{n_past});
                }

                slot.stats.n_prompt_cached = @intCast(n_past);
                slot.stats.n_prompt_processed = 0;

                self.metrics.addPromptCached(@intCast(n_past));

                try slot.prompt.tokens.keepFirst(@intCast(n_past));

                // this is to signal the client that the request has started processing
                if (slot.taskRef().params.stream) {
                    const empty: CompletionTokenOutput = .{};
                    if (slot.taskRef().params.return_progress) {
                        // send initial 0% progress update if needed
                        try self.sendPartialResponse(slot, &empty, true, false);
                    } else {
                        // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                        try self.sendPartialResponse(slot, &empty, false, true);
                    }
                }
            } // end of SLOT_STATE_STARTED

            if (!slot.canSplit()) {
                // cannot fit the prompt in the current batch - will try next iter
                if (batch.size() + slot.taskRef().nTokens() > n_batch) {
                    return;
                }
            }

            // note: the prompt timing is advanced in post_decode(), so it does not cover
            //       the tokens added to the batch below
            slot.printTimingsPp();

            // truncate any tokens that are beyond n_past for this slot
            const p0 = slot.prompt.tokens.posNext(-1);

            slt(.trc, func, slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", .{ slot.prompt.nTokens(), p0 });

            slot.mem.seqRm(slot.id, p0, -1);

            // shared prompt prefix: once it is processed, the children continue from it with their own prompt
            var wait_shared = false;
            if (slot.taskRef().n_tokens_shared > 0) {
                const is_shared_done = slot.prompt.nTokens() == slot.taskRef().n_tokens_shared;
                for (self.slots) |*other| {
                    if (other.state != .wait_other or other.taskRef().id_parent != slot.taskRef().id) {
                        continue;
                    }
                    if (is_shared_done) {
                        slt(.trc, func, slot, " - copying shared prompt (%d tokens) to child %d\n", .{ slot.prompt.nTokens(), other.id });
                        try slot.copyPromptTo(other);
                        other.state = .started;
                    } else {
                        wait_shared = true;
                    }
                }
            }

            // If using an alora, there may be uncached tokens that come
            // before the invocation sequence. When this happens, the
            // tokens before the invocation sequence need to be
            // processed without the adapter in a separate batch, then
            // the adapter needs to be enabled for the remaining tokens.
            if (common.loraAllAlora(slot.lora) and slot.alora_invocation_start - 1 > slot.prompt.nTokens()) {
                slt(.dbg, func, slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", .{ slot.prompt.nTokens(), slot.alora_invocation_start });
                const enabled_loras = try common.loraGetEnabledIds(gpa, slot.lora);
                defer gpa.free(enabled_loras);
                ggmlAssert(enabled_loras.len == 1, 4385, "enabled_loras.size() == 1");
                alora_scale.* = slot.lora[enabled_loras[0]].scale;
                slot.lora[enabled_loras[0]].scale = 0.0;
                alora_disabled_id.* = enabled_loras[0];
            }

            var do_checkpoint = self.params_base.n_ctx_checkpoints > 0;

            // make checkpoints only for completion tasks
            do_checkpoint = do_checkpoint and slot.taskRef().type == .completion;

            // make a checkpoint of the parts of the memory that cannot be rolled back.
            // checkpoints are created only if:
            // - the model does not support partial sequence removal
            // - the model uses SWA (and we are not using `swa_full`)
            // - the model supports partial sequence removal but only up to a fixed bound
            do_checkpoint = do_checkpoint and (self.ctx_tgt_seq_rm_type == .full or
                self.ctx_tgt_seq_rm_type == .rs or
                self.n_swa > 0);

            // TODO: do the same for all models, then remove process_mtmd_chunk()
            if (self.useMixedBatch() and !slot.canSplit() and input_tokens.has_mtmd) {
                if (!try self.addPromptMixed(slot)) {
                    return;
                }
            }

            var has_mtmd = false;

            // check if we should process the mtmd chunk
            while (true) {
                const cur_token_idx = slot.prompt.nTokens();
                if (cur_token_idx >= slot.taskRef().nTokens() or
                    input_tokens.at(@intCast(cur_token_idx)) != token_null // encountered a text token
                ) {
                    break;
                }

                // process the mtmd chunk
                // note: it submits its own decode, potentially be async
                //       so the timing is queued and flushed on the next sync
                self.metricsPreDecode();

                // encode on the worker thread, so we can still handle metrics tasks
                var work: MtmdWork = .{ .slot = slot, .idx = @intCast(cur_token_idx) };
                try self.queue_tasks.yieldToQueue(&work, MtmdWork.run);

                if (work.res != 0) {
                    slt(.err, func, slot, "failed to process mtmd chunk, res = %d\n", .{work.res});
                    try self.sendErrorSlot(slot, "failed to process mtmd chunk", .server);
                    try slot.release();
                    return; // the slot is done, skip it entirely
                }

                self.metricsQueuePrompt(work.n_tokens_out);
                slot.stats.n_prompt_processed += work.n_tokens_out;
                slot.stats.updatePromptLast();

                // add the mtmd chunk to cache
                {
                    const chunk = try input_tokens.findChunk(@intCast(cur_token_idx));
                    // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                    try slot.prompt.tokens.pushBackPlaceholder(chunk);
                }

                has_mtmd = true;
            }

            const spans = &slot.taskRef().params.message_spans;
            const last_user_pos = spans.lastUserMessagePos();

            const n_decision_first: i32 = if (slot.taskRef().type == .decision) slot.taskRef().decision.posFirst() else -1;

            // add prompt tokens for processing in the current batch
            while (slot.prompt.nTokens() < slot.taskRef().nTokens() and batch.size() < n_batch) {
                // get next token to process
                const cur_tok = input_tokens.at(@intCast(slot.prompt.nTokens()));
                if (cur_tok == token_null) {
                    break; // end of text chunk
                }

                // stop at the end of the shared prefix, the children are started from this state
                if (wait_shared and slot.prompt.nTokens() == slot.taskRef().n_tokens_shared) {
                    break;
                }

                // the outputs of a decision are read from one batch, do not split them
                if (slot.prompt.nTokens() == n_decision_first and batch.size() + slot.taskRef().nTokens() - n_decision_first > n_batch) {
                    break;
                }

                // if this is an alora request with pre-invocation
                // tokens that are not cached, we need to stop filling
                // this batch at those pre-invocation tokens.
                if (alora_scale.* > 0 and slot.prompt.nTokens() == slot.alora_invocation_start - 1) {
                    slt(.dbg, func, slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", .{ slot.prompt.nTokens(), slot.alora_invocation_start });
                    break;
                }

                // embedding requires all tokens in the batch to be output;
                // MTP also wants logits at every prompt position so the
                // streaming hook can mirror t_h_nextn into ctx_dft.
                add_ok.* = (try batch.add(slot.id, cur_tok, slot.prompt.tokens.posNext(-1), slot.needEmbd(), true)) and add_ok.*;
                if (slot.taskRef().decision.order.items.len != 0) {
                    batch.setDecisionOrder(batch.size() - 1, slot.taskRef().decision.order.items[@intCast(slot.prompt.nTokens())]);
                }
                try slot.prompt.tokens.pushBackToken(cur_tok);

                // break at the last user message, or at user messages at least min step past the last checkpoint
                if (do_checkpoint and spans.isUserStart(slot.prompt.nTokens())) {
                    const pos = slot.prompt.nTokens();
                    const checkpoints = slot.prompt.checkpoints.items;

                    if (pos == last_user_pos or checkpoints.len == 0 or pos > checkpoints[checkpoints.len - 1].n_tokens + self.params_base.checkpoint_min_step) {
                        break;
                    }
                }

                // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                // create checkpoints that many tokens before the end of the prompt:
                //  - 4 + n_ubatch
                //  - 4
                // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                if (do_checkpoint) {
                    // a function-local `static`: the first call's n_ubatch
                    if (checkpoint_offsets == null) checkpoint_offsets = .{ 4 + n_ubatch, 4 };

                    var should_break = false;
                    for (checkpoint_offsets.?) |offset| {
                        const n_last = @min(n_batch, offset);
                        if (slot.taskRef().nTokens() == slot.prompt.nTokens() + n_last) {
                            should_break = true;
                            break;
                        }
                    }
                    if (should_break) {
                        break;
                    }
                }
            }

            // the number of tokens added to the batch for the current slot
            const n_tokens_cur = batch.size() - n_tokens_prev;

            const n_tokens_start = slot.prompt.nTokens() - n_tokens_cur;

            const near_prompt_end = slot.taskRef().nTokens() < slot.prompt.nTokens() + n_ubatch;

            const is_user_start = spans.isUserStart(n_tokens_start);
            const is_last_user_message = n_tokens_start == last_user_pos;

            // entire prompt has been processed
            if (slot.prompt.nTokens() == slot.taskRef().nTokens()) {
                slot.state = .done_prompt;

                ggmlAssert(batch.size() > 0, 4548, "batch.size() > 0");

                // extract the logits only for the last token
                batch.setOutput(batch.size() - 1, true);

                slot.stats.n_gen = 0;
                slot.i_batch = batch.size() - 1;

                slot.initSampler();
            } else {
                // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                // message or we are near the end of the prompt
                if (!is_user_start and !near_prompt_end) {
                    do_checkpoint = false;
                }
            }

            const pos_min = c.llama_memory_seq_pos_min(c.llama_get_memory(self.ctx_tgt), slot.id);
            const pos_max = c.llama_memory_seq_pos_max(c.llama_get_memory(self.ctx_tgt), slot.id);

            // nothing to checkpoint yet
            // TODO: is this check needed?
            if (do_checkpoint and pos_min < 0) {
                do_checkpoint = false;
            }

            // do not checkpoint after mtmd chunks
            do_checkpoint = do_checkpoint and !has_mtmd;

            // no need to create checkpoints that are too close together, unless it's the last user message
            {
                const cps = slot.prompt.checkpoints.items;
                do_checkpoint = do_checkpoint and (cps.len == 0 or
                    is_last_user_message or near_prompt_end or
                    n_tokens_start > cps[cps.len - 1].n_tokens + self.params_base.checkpoint_min_step);
            }
            slt(.dbg, func, slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", .{ @as([*:0]const u8, if (do_checkpoint) "yes" else "no"), pos_min, pos_max });

            // note: we create the checkpoint before calling llama_decode(), so the current batch is not
            //       yet processed and therefore it is not part of the checkpoint.
            if (do_checkpoint) {
                try self.createCheckpoint(slot, n_tokens_cur, pos_min, pos_max);
            }
        }

        if (batch.slot_batched == null) {
            batch.slot_batched = slot;
        }
    }

    /// The `slots_debug` block of `pre_decode` (server-context.cpp:3761
    /// @8642541b2): the tokens around a prefix mismatch.
    fn printPrefixMismatch(self: *Impl, slot: *Slot, n_past: i32) !void {
        const func = "operator()";
        var tmp = std.heap.ArenaAllocator.init(gpa);
        defer tmp.deinit();
        const ta = tmp.allocator();
        const np0: i32 = @max(n_past - self.slots_n_diff, 0);
        const np1: i32 = @min(n_past + self.slots_n_diff + 2, @as(i32, @intCast(@min(slot.prompt.tokens.size(), slot.taskRef().tokens.size()))));

        var ss0: std.ArrayList(u8) = .empty;
        var ss1: std.ArrayList(u8) = .empty;
        var st0: std.ArrayList(u8) = .empty;
        var st1: std.ArrayList(u8) = .empty;

        try ss0.appendSlice(ta, "old: ... ");
        try ss1.appendSlice(ta, "new: ... ");

        var i = np0;
        while (i < np1) : (i += 1) {
            if (i == n_past) {
                try ss0.appendSlice(ta, " | ");
                try ss1.appendSlice(ta, " | ");
            }

            {
                const token = slot.prompt.tokens.at(@intCast(i));
                const p = if (token != token_null) try tc.common.tokenToPiece(ta, self.vocab, token, true) else "[mtmd]";
                try ss0.appendSlice(ta, p);
                try st0.print(ta, "{d:>8}", .{token});
            }

            {
                const token = slot.taskRef().tokens.at(@intCast(i));
                const p = if (token != token_null) try tc.common.tokenToPiece(ta, self.vocab, token, true) else "[mtmd]";
                try ss1.appendSlice(ta, p);
                try st1.print(ta, "{d:>8}", .{token});
            }
        }

        for ([_][]const u8{ ss0.items, ss1.items, st0.items, st1.items }) |line| {
            slt(.wrn, func, slot, "%s\n", .{(try ta.dupeSentinel(u8, line, 0)).ptr});
        }
    }

    const DecodeWork = struct {
        ctx: ?*c.llama_context,
        view: *tc.common.Batch,
        has_output: bool,
        ret: c_int = 0,

        fn run(w: *DecodeWork) anyerror!void {
            w.ret = c.llama_process(w.ctx, c.LLAMA_PROCESS_TYPE_DECODE, w.view.get());
            if (w.ret == 0 and w.has_output) {
                c.llama_synchronize(w.ctx);
            }
        }
    };

    const SpecProcessWork = struct {
        spec: ?*spec_mod.Speculative,
        view: *const tc.common.Batch,
        ok: bool = true,

        fn run(w: *SpecProcessWork) anyerror!void {
            w.ok = spec_mod.process(w.spec, w.view);
        }
    };

    /// Ports `server_context_impl`'s `use_mixed_batch` (server-context.cpp:4600
    /// @8642541b2): the mtmd chunks go in the batch as embeddings beside the
    /// text tokens. See https://github.com/ggml-org/llama.cpp/pull/29969
    fn useMixedBatch(self: *const Impl) bool {
        return c.llama_get_memory(self.ctx_tgt) == null or !c.llama_get_causal_attn(self.ctx_tgt);
    }

    const EncodeWork = struct {
        slot: *Slot,
        idx: usize,
        res: i32 = 0,

        fn run(w: *EncodeWork) anyerror!void {
            w.res = try encodeMtmdChunk(w.slot, &w.slot.mbatch, w.idx);
        }
    };

    /// Ports `server_context_impl`'s `add_prompt_mixed` (server-context.cpp:4607
    /// @8642541b2): the rest of the prompt into the batch, the mtmd chunks as
    /// embeddings next to the text tokens; the caller makes sure that it fits.
    ///
    /// Return: false on error, the slot is then released.
    fn addPromptMixed(self: *Impl, slot: *Slot) !bool {
        const func = "add_prompt_mixed";
        const input_tokens = &slot.task.?.tokens;
        const n_tokens_prev = self.batch.size();

        while (slot.prompt.nTokens() < slot.taskRef().nTokens()) {
            const cur_token_idx: usize = @intCast(slot.prompt.nTokens());
            const cur_tok = input_tokens.at(cur_token_idx);

            if (cur_tok != token_null) {
                const add_ok = try self.batch.add(slot.id, cur_tok, slot.prompt.tokens.posNext(-1), slot.needEmbd(), true);
                ggmlAssert(add_ok, 4621, "add_ok");
                if (slot.taskRef().decision.order.items.len != 0) {
                    self.batch.setDecisionOrder(self.batch.size() - 1, slot.taskRef().decision.order.items[cur_token_idx]);
                }
                try slot.prompt.tokens.pushBackToken(cur_tok);
                continue;
            }

            const chunk = try input_tokens.findChunk(cur_token_idx);

            var embd: [*c]f32 = if (slot.mbatch) |mb| c.mtmd_batch_get_output_embd(mb, chunk) else null;
            if (embd == null) {
                // encode on the worker thread, so we can still handle metrics tasks
                var work: EncodeWork = .{ .slot = slot, .idx = cur_token_idx };
                try self.queue_tasks.yieldToQueue(&work, EncodeWork.run);
                embd = if (work.res == 0) c.mtmd_batch_get_output_embd(slot.mbatch, chunk) else null;
            }

            if (embd == null or !try addMtmdChunk(slot, chunk, embd, &self.batch)) {
                slt(.err, func, slot, "%s", .{@as([*:0]const u8, "failed to process mtmd chunk\n")});
                // the batch must not keep the entries of a released slot
                self.batch.truncate(n_tokens_prev);
                try self.sendErrorSlot(slot, "failed to process mtmd chunk", .server);
                try slot.release();
                return false;
            }

            try slot.prompt.tokens.pushBackPlaceholder(chunk);
        }

        return true;
    }

    /// Ports `server_context_impl`'s `decode` (server-context.cpp:4658
    /// @8642541b2): true on success, false to retry with a smaller batch;
    /// an error is fatal.
    fn decode(self: *Impl, n_batch: *i32, off: i32) !bool {
        const func = "decode";
        srv(.dbg, func, "n_batch (effective) = %d, off = %d\n", .{ n_batch.*, off });

        self.metricsPreDecode();

        if (self.batch.size() == 0) {
            srv(.wrn, func, "%s", .{@as([*:0]const u8, "no tokens to decode\n")});

            self.n_empty_consecutive += 1;
            if (self.n_empty_consecutive > 3) {
                ggmlAbort(4667, "fatal error - please provide logs and repro in %s\n", .{@as([*:0]const u8, "https://github.com/ggml-org/llama.cpp/pull/20277")});
            }

            return true; // nothing to decode
        } else {
            self.n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (self.spec != null and self.batch.hasEmbd()) {
            if (c.llama_model_n_embd_inp(self.model_dft) != c.llama_model_n_embd_inp(self.model_tgt)) {
                srv(.err, func, "%s", .{@as([*:0]const u8, "unsupported batch.has_embd + spec case\n")});
                return common.throw(error.RuntimeError, "unsupported batch.has_embd + spec case", .{});
            }
        }

        var has_output = false;
        {
            var i = off;
            while (i < off + self.batch.view.size()) : (i += 1) {
                has_output = has_output or self.batch.tokens.items[@intCast(i)].output;
            }
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        var work: DecodeWork = .{ .ctx = self.ctx_tgt, .view = &self.batch.view, .has_output = has_output };
        try self.queue_tasks.yieldToQueue(&work, DecodeWork.run);
        const ret = work.ret;

        if (ret != 0) {
            {
                var err: []const u8 = "";

                if (n_batch.* == 1 and ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (err.len != 0) {
                    const z = dupZ(err);
                    defer freeZ(z);
                    srv(.err, func, "%s off = %d, n_batch = %d, ret = %d\n", .{ z, off, n_batch.*, ret });

                    for (self.slots) |*slot| {
                        if (slot.isProcessing()) {
                            try self.sendErrorSlot(slot, err, .server);
                            try slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.promptClear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    return common.throw(error.RuntimeError, "{s}", .{err});
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!self.tryClearIdleSlots()) {
                n_batch.* = @divTrunc(n_batch.*, 2);
            }

            srv(.wrn, func, "failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", .{ off, n_batch.*, ret });

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            self.metricsPostDecode(off, self.batch.view.size(), has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (self.spec != null) {
            var sw: SpecProcessWork = .{ .spec = self.spec, .view = &self.batch.view };
            try self.queue_tasks.yieldToQueue(&sw, SpecProcessWork.run);

            if (!sw.ok) {
                srv(.err, func, "%s", .{@as([*:0]const u8, "failed to process speculative batch\n")});

                // TODO: handle error
                return common.throw(error.RuntimeError, "failed to process speculative batch", .{});
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (self.slots) |*slot| {
            if (slot.state == .done_prompt and slot.taskRef().isParent()) {
                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (self.slots) |*child| {
                    if (child.state == .wait_other and slot.taskRef().id == child.taskRef().id_parent) {
                        slt(.trc, func, slot, " - copying state to child %d\n", .{child.id});

                        ggmlAssert(child.state == .wait_other, 4784, "child->state == SLOT_STATE_WAIT_OTHER");
                        // children with their own prompt are started at the end of the shared prefix
                        ggmlAssert(slot.taskRef().n_tokens_shared == 0, 4786, "slot.task->n_tokens_shared == 0");

                        try slot.copyStateTo(child);
                        child.state = .done_prompt;
                    }
                }
            }
        }

        return true;
    }

    /// The `accept_special_token` lambda of `post_decode` (server-context.cpp:4797 @8642541b2).
    fn acceptSpecialToken(self: *const Impl, slot: *const Slot, token: Token) bool {
        if (self.params_base.special) return true;
        for (slot.taskRef().params.sampling.preserved_tokens.items) |t| {
            if (t == token) return true;
        }
        return false;
    }

    /// The sampling step of `post_decode` (server-context.cpp:4797
    /// @8642541b2), one slot.
    fn sampleSlot(self: *Impl, slot: *Slot, off: i32, n_batch_tokens: i32, a: Allocator) !void {
        const batch_view = &self.batch.view;
        // optionally send prompt processing progress
        if (slot.state == .processing_prompt or slot.state == .done_prompt) {
            if (slot.taskRef().params.stream and slot.taskRef().params.return_progress) {
                const empty: CompletionTokenOutput = .{};
                try self.sendPartialResponse(slot, &empty, true, false);
            }
        }

        if (!(slot.i_batch >= off and slot.i_batch < off + n_batch_tokens)) {
            // the required token not in this sub-batch, skip
            return;
        }

        if (slot.state == .done_prompt) {
            if (slot.taskRef().type == .embedding) {
                // prompt evaluated for embedding
                try self.sendEmbedding(slot, batch_view);
                try slot.release();
                slot.i_batch = -1;
                return;
            }

            if (slot.taskRef().type == .rerank) {
                try self.sendRerank(slot, batch_view);
                try slot.release();
                slot.i_batch = -1;
                return;
            }

            if (slot.taskRef().type == .decision) {
                try self.sendDecision(slot, batch_view, slot.i_batch - off);
                try slot.release();
                slot.i_batch = -1;
                return;
            }

            ggmlAssert(slot.taskRef().needSampling(), 4854, "slot.task->need_sampling()");

            // prompt evaluated for next-token prediction
            slot.state = .generating;

            if (slot.canSpeculate()) {
                spec_mod.begin(self.spec, slot.id, try slot.prompt.tokens.getTextTokens(a));
            }
        } else if (slot.state != .generating) {
            return;
        }

        if (slot.canSpeculate() and slot.spec_draft.items.len != 0) {
            return; // sample using speculative decoding
        }

        // shifted according to the current sub-batch
        const tok_idx = slot.i_batch - off;

        const id = try slot.smpl.?.sample(slot.ctx_tgt, tok_idx, false);

        slot.i_batch = -1;

        slot.smpl.?.accept(id, true);

        // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
        const t_now = c.ggml_time_us();

        slot.stats.n_gen += 1;

        if (slot.stats.n_gen == 1) {
            slot.stats.updatePromptLast();
            slot.t_print_last = t_now;
            slot.n_gen_last = 0;
        }

        slot.stats.updateGenLast();

        const vocab = c.llama_model_get_vocab(c.llama_get_model(slot.ctx_tgt));
        var result: CompletionTokenOutput = .{};
        result.tok = id;
        result.text_to_send = try tc.common.tokenToPiece(a, vocab, result.tok, self.acceptSpecialToken(slot, result.tok));
        result.prob = 1.0; // TODO: set it here instead of doing inside populate_token_probs

        if (slot.taskRef().params.sampling.n_probs > 0) {
            try self.populateTokenProbs(slot, &result, slot.taskRef().params.post_sampling_probs, self.params_base.special, tok_idx, a);
        }

        if (!try self.processToken(&result, slot, a)) {
            // release slot because of stop condition
            try slot.printTimings();
            try self.sendFinalResponse(slot);
            try slot.release();

            return;
        }

        slot.printTimingsTg();
    }

    /// The speculative step of `post_decode` (server-context.cpp:4797
    /// @8642541b2), one slot: the main model samples and accepts.
    fn acceptDraft(self: *Impl, slot: *Slot, a: Allocator) !void {
        const func = "operator()";
        if (slot.state != .generating or !slot.canSpeculate() or
            slot.spec_draft.items.len == 0 or slot.spec_i_batch.items.len == 0)
        {
            return;
        }

        // save the original draft size
        const n_draft = slot.spec_draft.items.len;

        ggmlAssert(n_draft > 0, 4927, "n_draft > 0");

        // batch indices of the draft tokens, used to get the token probabilities below
        var spec_i_batch = slot.spec_i_batch;
        slot.spec_i_batch = .empty;
        defer spec_i_batch.deinit(gpa);

        // verify and try to accept the draft
        {
            const smpl_save = slot.smpl.?.clone();
            defer smpl_save.destroy();

            ggmlAssert(spec_i_batch.items.len == n_draft + 1, 4936, "spec_i_batch.size() == n_draft + 1");
            ggmlAssert(slot.spec_draft_q.items.len == 0 or (slot.spec_draft_q.items.len == slot.spec_draft.items.len), 4937, "slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size())");
            const synth_probs = spec_mod.getSynthProbs(self.spec);

            // drafters that fill no distribution fall back here
            const use_rejection = slot.useSpecRejection() and slot.spec_draft_q.items.len != 0;

            var accepted: []Token = undefined;
            if (synth_probs.len != 0) {
                // synthetic acceptance replaces verification entirely, so it comes first
                accepted = try sampleAndAcceptSynth(a, slot.smpl.?, slot.ctx_tgt, spec_i_batch.items, slot.spec_draft.items, synth_probs, &slot.spec_synth_rng, slot.spec_is_replay);
            } else if (slot.spec_is_replay and slot.useSpecRejection()) {
                accepted = try acceptReplay(a, slot.smpl.?, slot.ctx_tgt, spec_i_batch.items, slot.spec_draft.items);
            } else if (use_rejection) {
                const q = try a.alloc([]const c.llama_token_data, slot.spec_draft_q.items.len);
                for (slot.spec_draft_q.items, q) |*v, *o| o.* = v.items;
                accepted = try slot.smpl.?.sampleAndAcceptNRejection(a, slot.ctx_tgt, spec_i_batch.items, slot.spec_draft.items, q, false);
            } else {
                accepted = try slot.smpl.?.sampleAndAcceptNIdxs(a, slot.ctx_tgt, spec_i_batch.items, slot.spec_draft.items, false);
            }

            ggmlAssert(accepted.len >= 1, 4957, "accepted.size() >= 1");

            const n_rollback: u32 = @intCast(slot.spec_draft.items.len + 1 - accepted.len);

            const use_ckpt_tgt = self.ctx_tgt_seq_rm_type == .full or
                (self.ctx_tgt_seq_rm_type == .rs and n_rollback > c.llama_n_rs_seq(self.ctx_tgt));

            // check for partial draft acceptance
            if (n_rollback > 0) {
                if (use_ckpt_tgt) {
                    if (self.trace > 0) {
                        slt(.inf, func, slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", .{ accepted.len - 1, slot.spec_draft.items.len });
                    }

                    // partial acceptance is not supported by the context -> truncate the draft and restore the state
                    slot.spec_is_replay = true;
                    slot.spec_draft.clearRetainingCapacity();
                    try slot.spec_draft.appendSlice(gpa, accepted);

                    const ckpt = &slot.spec_ckpt;

                    slt(.dbg, func, slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", .{ ckpt.pos_min, ckpt.pos_max, ckpt.size() });

                    ggmlAssert(ckpt.loadTgt(slot.ctx_tgt, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY), 4980, "ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)");

                    if (slot.ctx_dft != null) {
                        ggmlAssert(ckpt.loadDft(slot.ctx_dft, slot.id, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY), 4983, "ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)");
                    }

                    slot.mem.seqRm(slot.id, ckpt.pos_max + 1, -1);

                    try slot.prompt.tokens.keepFirst(@intCast(ckpt.n_tokens));
                    sampling.Sampler.copy(smpl_save, slot.smpl.?);

                    return;
                }
            }

            if (self.trace > 0) {
                slt(.inf, func, slot, "accepted %2zu/%2zu draft tokens\n", .{ accepted.len - 1, n_draft });
            }

            spec_mod.accept(self.spec.?, slot.id, @intCast(accepted.len - 1));

            slot.spec_draft.clearRetainingCapacity();
            try slot.spec_draft.appendSlice(gpa, accepted);
        }

        const ids = try a.dupe(Token, slot.spec_draft.items);
        slot.spec_draft.clearRetainingCapacity(); // std::move

        var n_accepted = ids.len - 1;
        if (slot.spec_is_replay and n_accepted > 0) {
            n_accepted -= 1;
        }
        slot.spec_is_replay = false;

        slot.stats.updateGenLast();

        // update how many tokens out of those tested were accepted
        slot.stats.n_draft_accepted += n_accepted;
        slot.stats.n_draft_verif_steps += 1;

        const n_accepted_per_pos = &slot.n_accepted_per_pos;
        if (n_accepted_per_pos.items.len == 0) {
            try n_accepted_per_pos.appendNTimes(gpa, 0, @intCast(spec_mod.nMax(self.spec)));
        }
        {
            var i: usize = 0;
            while (i < n_accepted and i < n_accepted_per_pos.items.len) : (i += 1) {
                n_accepted_per_pos.items[i] += 1;
            }
        }

        // add accepted tokens to the prompt
        try slot.prompt.tokens.keepFirst(@intCast(slot.prompt.nTokens() - @as(i32, @intCast(n_draft))));
        try slot.prompt.tokens.insert(ids[0 .. ids.len - 1]);

        slot.sampled = ids[ids.len - 1]; // last accepted token
        slt(.dbg, func, slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", .{ slot.sampled, ids.len, n_draft });

        slot.mem.seqRm(slot.id, slot.prompt.tokens.posNext(-1), -1);

        const vocab = c.llama_model_get_vocab(c.llama_get_model(slot.ctx_tgt));
        for (ids, 0..) |tok, i| {
            var result: CompletionTokenOutput = .{};

            result.tok = tok;
            result.text_to_send = try tc.common.tokenToPiece(a, vocab, result.tok, self.acceptSpecialToken(slot, result.tok));
            result.prob = 1.0; // set later

            // post_sampling_probs is not supported with speculative decoding
            if (slot.taskRef().params.sampling.n_probs > 0 and !slot.taskRef().params.post_sampling_probs) {
                try self.populateTokenProbs(slot, &result, false, self.params_base.special, spec_i_batch.items[i], a);
            }

            slot.stats.n_gen += 1;

            if (!try self.processToken(&result, slot, a)) {
                try slot.printTimings();
                try self.sendFinalResponse(slot);
                try slot.release();

                return;
            }
        }

        slot.printTimingsTg();

        slt(.dbg, func, slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", .{ @as(c_int, @intCast(n_accepted)), @as(c_int, @intCast(n_draft)), slot.prompt.nTokens() });
    }

    /// Ports `server_context_impl`'s `post_decode` (server-context.cpp:4797 @8642541b2).
    fn postDecode(self: *Impl, n_batch_tokens: i32, off: i32) !void {
        var tmp = std.heap.ArenaAllocator.init(gpa);
        defer tmp.deinit();
        const a = tmp.allocator();

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        for (self.slots) |*slot| {
            for (slot.spec_i_batch.items) |i| {
                if (!(i >= off and i < off + n_batch_tokens)) {
                    self.slotException(slot, common.throw(error.RuntimeError, "speculative batch index {d} is not inside the current sub-batch [{d}, {d})", .{ i, off, off + n_batch_tokens }));
                    break;
                }
            }
        }

        for (self.slots) |*slot| {
            self.sampleSlot(slot, off, n_batch_tokens, a) catch |e| self.slotException(slot, e);
        }

        // speculative decoding - main model sample and accept
        for (self.slots) |*slot| {
            self.acceptDraft(slot, a) catch |e| self.slotException(slot, e);
        }
    }

    /// Ports `server_context_impl`'s `n_ctx_slot` (server-context.cpp:5065
    /// @8642541b2): context size of a single slot, capped by
    /// --kv-unified-per-slot and by the training context of the model.
    pub fn nCtxSlot(self: *const Impl) c_int {
        var res: c_int = @intCast(c.llama_n_ctx_seq(self.ctx_tgt));

        if (self.params_base.kv_unified_per_slot > 0) {
            res = @min(res, self.params_base.kv_unified_per_slot);
        }

        return @min(res, c.llama_model_n_ctx_train(self.model_tgt));
    }

    //
    // metrics helpers
    //

    /// Ports `server_context_impl`'s `metrics_pre_decode` (server-context.cpp:5084
    /// @8642541b2): call before submitting a decode, so that the queued
    /// prompt stats can be timed.
    fn metricsPreDecode(self: *Impl) void {
        self.t_decode_start = c.ggml_time_us();
    }

    /// Ports `server_context_impl`'s `metrics_queue_prompt` (server-context.cpp:5089
    /// @8642541b2): the batch is submitted, but its compute may not be done yet.
    fn metricsQueuePrompt(self: *Impl, n_tokens: u64) void {
        if (n_tokens == 0) {
            return;
        }
        if (self.n_prompt_queued == 0) {
            self.t_prompt_start = self.t_decode_start;
        }
        self.n_prompt_queued += n_tokens;
    }

    /// Ports `server_context_impl`'s `metrics_flush_prompt` (server-context.cpp:5100
    /// @8642541b2): call only after the context is synchronized, otherwise
    /// the time is meaningless.
    fn metricsFlushPrompt(self: *Impl) void {
        if (self.n_prompt_queued == 0) {
            return;
        }
        self.metrics.addPrompt(self.n_prompt_queued, @bitCast(c.ggml_time_us() - self.t_prompt_start));
        self.n_prompt_queued = 0;
    }

    /// Ports `server_context_impl`'s `metrics_post_decode` (server-context.cpp:5109
    /// @8642541b2): `has_output` is computed by the caller, which also
    /// already synchronized the context if it is set.
    fn metricsPostDecode(self: *Impl, off: i32, n_tokens: i32, has_output: bool) void {
        self.metrics.n_decode += 1;
        for (self.slots) |*slot| {
            if (slot.isProcessing()) {
                self.metrics.n_busy_slots += 1;
            }
            self.metrics.n_tokens_max = @max(self.metrics.n_tokens_max, @as(u64, @bitCast(@as(i64, slot.prompt.nTokens()))));
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        var n_prompt_tokens: u64 = 0;

        var i = off;
        while (i < off + n_tokens) : (i += 1) {
            const t = &self.batch.tokens.items[@intCast(i)];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens += 1;

            const slot = &self.slots[@intCast(t.id_slot)];
            if (slot.stats.isSet()) {
                slot.stats.n_prompt_processed += 1;
            }
        }

        self.metricsQueuePrompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            self.metricsFlushPrompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const t_now = c.ggml_time_us();
        i = off;
        while (i < off + n_tokens) : (i += 1) {
            const t = &self.batch.tokens.items[@intCast(i)];
            const slot = &self.slots[@intCast(t.id_slot)];
            if (t.is_prompt and slot.stats.isSet()) {
                slot.stats.setPromptLast(t_now);
            }
        }
    }

    /// Ports `server_context_impl`'s `metrics_flush_idle` (server-context.cpp:5158
    /// @8642541b2): flush any queued prompt metrics if all slots are now idle.
    fn metricsFlushIdle(self: *Impl) void {
        if (self.n_prompt_queued == 0) {
            return;
        }

        c.llama_synchronize(self.ctx_tgt);
        self.metricsFlushPrompt();
    }

    /// Ports `server_context_impl`'s `metrics_on_prediction` (server-context.cpp:5167 @8642541b2).
    fn metricsOnPrediction(self: *Impl, slot: *const Slot) !void {
        const t_us: u64 = @bitCast(slot.stats.tGenUs());
        const n = slot.stats.n_gen;
        const n_steps = slot.stats.nGenSteps();

        self.metrics.predict.add(n, n_steps, t_us);
        self.metrics.predict_bucket.add(n, n_steps, t_us);

        self.metrics.n_draft_tokens += slot.stats.n_draft_tokens;
        self.metrics.n_draft_accepted += slot.stats.n_draft_accepted;
        self.metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        const dst = &self.metrics.n_accepted_per_pos;
        const src = slot.n_accepted_per_pos.items;

        if (dst.items.len < src.len) {
            try dst.appendNTimes(gpa, 0, src.len - dst.items.len);
        }
        for (src, 0..) |v, i| {
            dst.items[i] += v;
        }
    }
};

/// The function-local `static const int checkpoint_offsets[]` of
/// `pre_decode` (server-context.cpp:3761 @8642541b2): set from the first
/// call's `n_ubatch`.
var checkpoint_offsets: ?[2]i32 = null;

// -----------------------------------------------------------------------------
// server_context

/// Ports `server_context` (server-context.h:63 @8642541b2): the public face
/// of `Impl`.
pub const ServerContext = struct {
    impl: *Impl,

    /// Ports the constructor of `server_context` (server-context.cpp:5195 @8642541b2).
    pub fn init() !ServerContext {
        const impl = try gpa.create(Impl);
        errdefer gpa.destroy(impl);
        try impl.init();
        return .{ .impl = impl };
    }

    pub fn deinit(self: *ServerContext) void {
        self.impl.deinit();
        gpa.destroy(self.impl);
    }

    /// Ports `server_context`'s `load_model` (server-context.cpp:5198
    /// @8642541b2): load the model and initialize llama_context.
    ///
    /// Return: true on success.
    pub fn loadModel(self: *ServerContext, params: *Params) !bool {
        return self.impl.loadModel(params);
    }

    /// Ports `server_context`'s `start_loop` (server-context.cpp:5202
    /// @8642541b2): blocks the main thread until termination.
    pub fn startLoop(self: *ServerContext) !void {
        const params = &self.impl.params_base;
        try self.impl.queue_tasks.startLoop(@as(i64, params.sleep_idle_seconds) * 1000);
    }

    /// Ports `server_context`'s `terminate` (server-context.cpp:5207
    /// @8642541b2): unblocks `startLoop`.
    pub fn terminate(self: *ServerContext) void {
        self.impl.queue_tasks.terminate();
    }

    /// Ports `server_context`'s `get_llama_context` (server-context.cpp:5211
    /// @8642541b2): null while sleeping.
    pub fn getLlamaContext(self: *const ServerContext) ?*c.llama_context {
        return self.impl.ctx_tgt;
    }

    /// Ports `server_context`'s `set_state_callback` (server-context.cpp:5291
    /// @8642541b2): set before `loadModel`.
    pub fn setStateCallback(self: *ServerContext, callback: StateCallback) void {
        self.impl.callback_state = callback;
    }

    /// Ports `server_context`'s `get_meta` (server-context.cpp:5219
    /// @8642541b2): read-only metadata, after `loadModel`.
    ///
    /// Return: strings owned by `a`, or by the context for its lifetime.
    pub fn getMeta(self: *const ServerContext, a: Allocator) !Meta {
        const impl = self.impl;
        const bos_id = c.llama_vocab_bos(impl.vocab);
        const eos_id = c.llama_vocab_eos(impl.vocab);
        const bos_token_str: []const u8 = if (bos_id != token_null) try tc.common.tokenToPiece(a, impl.vocab, bos_id, true) else "";
        const eos_token_str: []const u8 = if (eos_id != token_null) try tc.common.tokenToPiece(a, impl.vocab, eos_id, true) else "";

        const ftype_name = std.mem.span(c.llama_ftype_name(c.llama_model_ftype(impl.model_tgt)));

        const tmpls = &impl.tmpls.?;
        return .{
            .build_info = result_mod.build_info,
            .model_name = impl.model_name,
            .model_aliases = impl.model_aliases,
            .model_tags = impl.model_tags,
            .model_path = impl.params_base.model.path,
            .model_output_modalities = common.serverModelOutputModalities(tc.common.getDecisionType(impl.model_tgt), impl.hasCapChat(), impl.hasCapTts()),
            .has_mtmd = impl.mctx != null,
            .has_inp_image = impl.chat_params.allow_image,
            .has_inp_audio = impl.chat_params.allow_audio,
            .has_inp_video = impl.chat_params.allow_video,
            .has_cap_chat = impl.hasCapChat(),
            .has_cap_tts = impl.hasCapTts(),
            .has_audio_stream = impl.mtmd_cap.audio_stream,
            .json_ui_settings = &impl.json_ui_settings,
            .slot_n_ctx = impl.nCtxSlot(),
            .pooling_type = c.llama_pooling_type(impl.ctx_tgt),

            .chat_params = &impl.chat_params,
            .chat_template_caps = try chat.templates.getCapsJson(a, tmpls),
            .chat_template = chat.templates.source(tmpls, ""),
            .chat_template_tool_use = chat.templates.source(tmpls, "tool_use"),

            .bos_token_str = bos_token_str,
            .eos_token_str = eos_token_str,
            .fim_pre_token = c.llama_vocab_fim_pre(impl.vocab),
            .fim_sub_token = c.llama_vocab_fim_suf(impl.vocab),
            .fim_mid_token = c.llama_vocab_fim_mid(impl.vocab),
            .fim_pad_token = c.llama_vocab_fim_pad(impl.vocab),
            .fim_rep_token = c.llama_vocab_fim_rep(impl.vocab),
            .fim_sep_token = c.llama_vocab_fim_sep(impl.vocab),

            .logit_bias_eog = impl.params_base.sampling.logit_bias_eog.items,

            .model_vocab_type = c.llama_vocab_type(impl.vocab),
            .model_vocab_n_tokens = c.llama_vocab_n_tokens(impl.vocab),
            .model_n_ctx_train = c.llama_model_n_ctx_train(impl.model_tgt),
            .model_n_embd_inp = c.llama_model_n_embd(impl.model_tgt),
            .model_n_params = c.llama_model_n_params(impl.model_tgt),
            .model_size = c.llama_model_size(impl.model_tgt),
            .model_ftype = ftype_name,
        };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a partial stop word is found where its prefix ends the text" {
    try std.testing.expectEqual(@as(?usize, 3), stringFindPartialStop("abcxy", "xyz"));
    try std.testing.expectEqual(@as(?usize, null), stringFindPartialStop("abc", "xyz"));
    try std.testing.expectEqual(@as(?usize, null), stringFindPartialStop("", "x"));
    try std.testing.expectEqual(@as(?usize, 2), stringFindPartialStop("abx", "x"));
}
