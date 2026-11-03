//! The port's half of `scripts/server-task-diff`: the records
//! `harness/server_task_ref.cpp` runs through upstream's `server-common`,
//! `server-task` and `server-schema`, run through `tools/server/`'s data
//! layer and written as the same transcript.
//!
//! # Provenance
//!
//! **Not a port.** A test driver; it mirrors `harness/server_task_ref.cpp`
//! op for op, and the two must stay in step.
//!
//! Usage: server_task_port <records> <transcript> <text-model> <mmproj> <vocab>...

const std = @import("std");
const json = @import("json");
const chat = @import("common_chat");
const tools = @import("tools_common");
const sd = @import("server_data");

comptime {
    // libllama and libmtmd, compiled in: their exports are what `c.llama_*`
    // and `c.mtmd_*` link against.
    _ = @import("llama");
    _ = @import("mtmd");
}

const c = tools.c;
const log = tools.log;
const Value = json.Value;
const common = sd.common;
const tokens = sd.tokens;
const oaicompat = sd.oaicompat;
const task = sd.task;
const result = sd.result;
const schema = sd.schema;
const ServerTokens = tokens.ServerTokens;
const Token = common.Token;

var gpa_state: std.heap.DebugAllocator(.{ .thread_safe = false, .stack_trace_frames = 10 }) = .init;
const gpa = gpa_state.allocator();

/// The record's arena: everything a record allocates and does not free.
var rec_arena: std.heap.ArenaAllocator = undefined;
fn ra() std.mem.Allocator {
    return rec_arena.allocator();
}

var out: std.ArrayList(u8) = .empty;

/// `init.io`, for reading template files.
var g_io: std.Io = undefined;

fn line(s: []const u8) !void {
    try out.appendSlice(gpa, s);
    try out.append(gpa, '\n');
}

fn linef(comptime fmt: []const u8, args: anytype) !void {
    try out.print(gpa, fmt ++ "\n", args);
}

fn str(label: []const u8, s: []const u8) !void {
    try out.print(gpa, "{s}({d}):{s}\n", .{ label, s.len, s });
}

/// The C++ driver's `guarded`: a throw is printed as its class and text.
fn guarded(r: anyerror!void) !void {
    r catch |e| switch (e) {
        error.JsonError => try str("json_error", try oneLine(common.what(e))),
        error.InvalidArgument => try str("invalid_argument", try oneLine(common.what(e))),
        error.RuntimeError => try str("runtime_error", try oneLine(common.what(e))),
        else => return e,
    };
}

/// An exception's text on one line, as the reference's `one_line`.
fn oneLine(s: []const u8) ![]const u8 {
    return std.mem.replaceOwned(u8, ra(), s, "\n", "\\n");
}

fn hexBits(buf: *[8]u8, f: f32) []const u8 {
    return std.fmt.bufPrint(buf, "{x:0>8}", .{@as(u32, @bitCast(f))}) catch unreachable;
}

fn fnv(p: []const u8) u64 {
    var h: u64 = 1469598103934665603;
    for (p) |b| {
        h ^= b;
        h *%= 1099511628211;
    }
    return h;
}

fn toksStr(t: []const Token) ![]u8 {
    var s: std.ArrayList(u8) = .empty;
    try s.append(ra(), '[');
    for (t, 0..) |x, i| {
        if (i != 0) try s.append(ra(), ',');
        try s.print(ra(), "{d}", .{x});
    }
    try s.append(ra(), ']');
    return s.items;
}

fn dumpV(v: *const Value) ![]u8 {
    return json.dump(ra(), v, -1);
}

fn dumpOwned(v: Value) ![]u8 {
    var x = v;
    defer x.deinit(gpa);
    return json.dump(ra(), &x, -1);
}

fn unhex(hex: []const u8) ![]u8 {
    const b = try ra().alloc(u8, hex.len / 2);
    for (b, 0..) |*x, i| x.* = try std.fmt.parseInt(u8, hex[2 * i ..][0..2], 16);
    return b;
}

// the loaded models
var vocab_models: std.ArrayList(*c.llama_model) = .empty;
var text_model: ?*c.llama_model = null;
var text_ctx: ?*c.llama_context = null;
var mctx: ?*c.mtmd_context = null;
var params_base: tools.params.Params = undefined;
var base_arena: std.heap.ArenaAllocator = undefined;
var init_opt: c.mtmd_helper_init_opt = undefined;

fn vocabAt(j: *const Value) !?*const c.llama_vocab {
    const i = try j.value("v", usize, 0);
    if (i >= vocab_models.items.len) return error.JsonError;
    return c.llama_model_get_vocab(vocab_models.items[i]);
}

fn eogBias(vocab: ?*const c.llama_vocab) ![]c.llama_logit_bias {
    var res: std.ArrayList(c.llama_logit_bias) = .empty;
    const n = c.llama_vocab_n_tokens(vocab);
    var i: c_int = 0;
    while (i < n) : (i += 1) {
        if (c.llama_vocab_is_eog(vocab, i)) {
            try res.append(ra(), .{ .token = i, .bias = -std.math.inf(f32) });
        }
    }
    return res.items;
}

fn paramsExtra(p: *const task.TaskParams) !Value {
    const a = ra();
    var lists = common.Obj.init(a);
    try lists.put("reasoning_budget_start", p.sampling.reasoning_budget_start.items);
    var ends = Value.emptyArray();
    for (p.sampling.reasoning_budget_end.items) |e| try ends.pushBack(a, try common.from(a, e.items));
    try lists.put("reasoning_budget_end", ends);
    try lists.put("reasoning_budget_forced", p.sampling.reasoning_budget_forced.items);
    var o = common.Obj.init(a);
    try o.put("cache_prompt", p.cache_prompt);
    try o.put("include_usage", p.include_usage);
    try o.put("return_tokens", p.return_tokens);
    try o.put("return_progress", p.return_progress);
    try o.put("sse_ping_interval", p.sse_ping_interval);
    try o.put("n_indent", p.n_indent);
    try o.put("n_cmpl", p.n_cmpl);
    try o.put("n_cache_reuse", p.n_cache_reuse);
    try o.put("t_max_predict_ms", p.t_max_predict_ms);
    try o.put("verbose", p.verbose);
    try o.put("response_fields", p.response_fields);
    try o.put("grammar_type", @as(c_int, @backingInt(p.sampling.grammar_type)));
    try o.put("sampling_generation_prompt", p.sampling.generation_prompt);
    try o.put("reasoning_control", p.sampling.reasoning_control);
    try o.put("reasoning_budget_tokens", p.sampling.reasoning_budget_tokens);
    try o.put("lists", lists.done());
    try o.put("chat_format_int", p.chat_parser_params.format);
    try o.put("parse_tool_calls", p.chat_parser_params.parse_tool_calls);
    try o.put("is_continuation", p.chat_parser_params.is_continuation);
    try o.put("echo", p.chat_parser_params.echo);
    try o.put("n_logit_bias", p.sampling.logit_bias.items.len);
    return o.done();
}

fn evalParams(a: std.mem.Allocator, j: *const Value) !task.TaskParams {
    const vocab = try vocabAt(j);
    const eog = try eogBias(vocab);
    return schema.evalLlamaCmplSchema(a, vocab, &params_base, if (try j.value("eog", bool, true)) eog else &.{}, try j.at("data"));
}

// ---------------------------------------------------------------------------
// results

var states: std.AutoHashMapUnmanaged(i64, *task.TaskResultState) = .empty;

fn fillStats(s: *common.SlotStats, j: *const Value) !void {
    s.n_prompt_cached = try j.value("cached", u64, 0);
    s.n_prompt_processed = try j.value("processed", u64, 0);
    s.n_gen = try j.value("gen", u64, 0);
    s.n_draft_tokens = try j.value("draft", u64, 0);
    s.n_draft_accepted = try j.value("accepted", u64, 0);
    s.t_start = try j.value("t_start", i64, 0);
    s.t_prompt_last = try j.value("t_prompt", i64, 0);
    s.t_gen_last = try j.value("t_gen", i64, 0);
}

fn member(a: std.mem.Allocator, j: *const Value, key: []const u8, def: Value) !Value {
    return if (j.contains(key)) (try j.at(key)).clone(a) else def;
}

fn probsOf(a: std.mem.Allocator, j: *const Value) ![]task.CompletionTokenOutput {
    var res: std.ArrayList(task.CompletionTokenOutput) = .empty;
    var it = @constCast(j).items();
    while (it.next()) |e| {
        const p = e.value;
        var o: task.CompletionTokenOutput = .{};
        o.tok = try p.value("tok", Token, 0);
        o.prob = try p.value("prob", f32, 0.0);
        o.text_to_send = try a.dupe(u8, try p.value("text", []const u8, ""));
        var ps: std.ArrayList(task.CompletionTokenOutput.ProbInfo) = .empty;
        var qv = try member(a, p, "probs", Value.emptyArray());
        var qit = qv.items();
        while (qit.next()) |q| {
            try ps.append(a, .{ .tok = try q.value.value("tok", Token, 0), .txt = try a.dupe(u8, try q.value.value("txt", []const u8, "")), .prob = try q.value.value("prob", f32, 0.0) });
        }
        o.probs = ps.items;
        try res.append(a, o);
    }
    return res.items;
}

fn tokensOf(a: std.mem.Allocator, j: *const Value, key: []const u8) ![]const Token {
    return if (j.contains(key)) (try j.at(key)).getSlice(a, Token) else &.{};
}

fn stringsOf(a: std.mem.Allocator, j: *const Value, key: []const u8) ![]const []const u8 {
    if (!j.contains(key)) return &.{};
    const xs = try (try j.at(key)).getSlice(a, []const u8);
    for (xs) |*x| x.* = try a.dupe(u8, x.*);
    return xs;
}

fn bucket(b: *common.Metrics.Bucket, f: *const Value, k: []const u8) !void {
    if (!f.contains(k)) return;
    const bj = try f.at(k);
    b.count = try (try bj.atIndex(0)).get(u64);
    b.steps = try (try bj.atIndex(1)).get(u64);
    b.time = try (try bj.atIndex(2)).get(u64);
}

fn runResult(j: *const Value) !void {
    const kind = try (try j.at("kind")).get([]const u8);
    var fv = try member(ra(), j, "f", Value.emptyObject());
    const f = &fv;
    var holder: ?*result.Result = null;
    defer if (holder) |x| x.destroy(gpa);
    var r: *result.Result = undefined;
    if (std.mem.eql(u8, kind, "final")) {
        r = try result.Result.create(gpa, .cmpl_final);
        holder = r;
        const a = r.alloc();
        const x = &r.kind.cmpl_final;
        x.content = try a.dupe(u8, try f.value("content", []const u8, ""));
        x.tokens = try tokensOf(a, f, "tokens");
        x.stream = try f.value("stream", bool, false);
        x.include_usage = try f.value("include_usage", bool, false);
        if (f.contains("stats")) try fillStats(&x.stats, try f.at("stats"));
        x.prompt = try a.dupe(u8, try f.value("prompt", []const u8, ""));
        x.truncated = try f.value("truncated", bool, false);
        x.n_decoded = try f.value("n_decoded", i32, 0);
        x.n_prompt_tokens = try f.value("n_prompt_tokens", i32, 0);
        x.n_prompt_tokens_cache = try f.value("n_prompt_tokens_cache", i32, 0);
        x.n_tokens_cached = try f.value("n_tokens_cached", i32, 0);
        x.has_new_line = try f.value("has_new_line", bool, false);
        x.stopping_word = try a.dupe(u8, try f.value("stopping_word", []const u8, ""));
        x.stop = @fromBackingInt(@intCast(try f.value("stop", c_int, 0)));
        x.post_sampling_probs = try f.value("post_sampling_probs", bool, false);
        if (f.contains("probs")) x.probs_output = try probsOf(a, try f.at("probs"));
        x.response_fields = try stringsOf(a, f, "response_fields");
        x.generation_params = if (j.contains("gp")) try evalParams(a, try j.at("gp")) else try task.TaskParams.init(a);
        x.verbose = try f.value("verbose", bool, false);
        x.res_type = @fromBackingInt(@intCast(try f.value("res_type", c_int, 0)));
        x.oaicompat_model = try a.dupe(u8, try f.value("model", []const u8, ""));
        x.oaicompat_cmpl_id = try a.dupe(u8, try f.value("cmpl_id", []const u8, ""));
    } else if (std.mem.eql(u8, kind, "partial")) {
        r = try result.Result.create(gpa, .cmpl_partial);
        holder = r;
        const a = r.alloc();
        const x = &r.kind.cmpl_partial;
        x.content = try a.dupe(u8, try f.value("content", []const u8, ""));
        x.tokens = try tokensOf(a, f, "tokens");
        x.n_decoded = try f.value("n_decoded", i32, 0);
        x.n_prompt_tokens = try f.value("n_prompt_tokens", i32, 0);
        x.n_prompt_tokens_cache = try f.value("n_prompt_tokens_cache", i32, 0);
        x.post_sampling_probs = try f.value("post_sampling_probs", bool, false);
        x.is_progress = try f.value("is_progress", bool, false);
        x.is_begin = try f.value("is_begin", bool, false);
        if (f.contains("probs")) {
            const po = try probsOf(a, try f.at("probs"));
            if (po.len != 0) x.prob_output = po[0];
        }
        if (f.contains("stats")) try fillStats(&x.stats, try f.at("stats"));
        if (f.contains("progress")) {
            const pr = try f.at("progress");
            x.progress.total = try pr.value("total", i32, 0);
            x.progress.cache = try pr.value("cache", i32, 0);
            x.progress.processed = try pr.value("processed", i32, 0);
            x.progress.time_ms = try pr.value("time_ms", i64, 0);
        }
        x.verbose = try f.value("verbose", bool, false);
        x.res_type = @fromBackingInt(@intCast(try f.value("res_type", c_int, 0)));
        x.oaicompat_model = try a.dupe(u8, try f.value("model", []const u8, ""));
        x.oaicompat_cmpl_id = try a.dupe(u8, try f.value("cmpl_id", []const u8, ""));
    } else if (std.mem.eql(u8, kind, "embd")) {
        r = try result.Result.create(gpa, .embd);
        holder = r;
        const a = r.alloc();
        const x = &r.kind.embd;
        var embs: std.ArrayList([]const f32) = .empty;
        if (f.contains("embedding")) {
            var it = (try f.at("embedding")).items();
            while (it.next()) |e| try embs.append(a, try e.value.getSlice(a, f32));
        }
        x.embedding = embs.items;
        x.n_tokens = try f.value("n_tokens", i32, 0);
        x.res_type = @fromBackingInt(@intCast(try f.value("res_type", c_int, 0)));
    } else if (std.mem.eql(u8, kind, "rerank")) {
        r = try result.Result.create(gpa, .rerank);
        holder = r;
        if (f.contains("score")) r.kind.rerank.score = try f.value("score", f32, 0.0);
        r.kind.rerank.n_tokens = try f.value("n_tokens", i32, 0);
    } else if (std.mem.eql(u8, kind, "error")) {
        r = try result.Result.create(gpa, .err);
        holder = r;
        const a = r.alloc();
        r.kind.err.err_type = @fromBackingInt(@intCast(try f.value("type", c_int, 2)));
        r.kind.err.err_msg = try a.dupe(u8, try f.value("msg", []const u8, ""));
        r.kind.err.n_prompt_tokens = try f.value("n_prompt_tokens", i32, 0);
        r.kind.err.n_ctx = try f.value("n_ctx", i32, 0);
    } else if (std.mem.eql(u8, kind, "metrics")) {
        r = try result.Result.create(gpa, .metrics);
        holder = r;
        const a = r.alloc();
        const x = &r.kind.metrics;
        x.n_processing_slots = try f.value("processing", c_int, 0);
        x.n_tasks_deferred = try f.value("deferred", c_int, 0);
        const m = &x.metrics;
        try bucket(&m.prompt_bucket, f, "prompt_bucket");
        try bucket(&m.predict_bucket, f, "predict_bucket");
        try bucket(&m.prompt, f, "prompt");
        try bucket(&m.predict, f, "predict");
        m.n_prompt_cached = try f.value("n_prompt_cached", u64, 0);
        m.n_tokens_max = try f.value("n_tokens_max", u64, 0);
        m.n_decode = try f.value("n_decode", u64, 0);
        m.n_busy_slots = try f.value("n_busy_slots", u64, 0);
        m.n_draft_tokens = try f.value("n_draft_tokens", u64, 0);
        m.n_draft_accepted = try f.value("n_draft_accepted", u64, 0);
        m.n_draft_verif_steps = try f.value("n_draft_verif_steps", u64, 0);
        if (f.contains("per_pos")) {
            var it = (try f.at("per_pos")).items();
            while (it.next()) |v| try m.n_accepted_per_pos.append(a, try v.value.get(u64));
        }
        try str("metrics", try r.toMetrics(ra()));
    } else if (std.mem.eql(u8, kind, "slots")) {
        r = try result.Result.create(gpa, .slots);
        holder = r;
        r.kind.slots.slots_data = try member(r.alloc(), f, "data", Value.emptyArray());
    } else if (std.mem.eql(u8, kind, "save")) {
        r = try result.Result.create(gpa, .slot_save_load);
        holder = r;
        const x = &r.kind.slot_save_load;
        x.filename = try r.alloc().dupe(u8, try f.value("filename", []const u8, ""));
        x.is_save = try f.value("is_save", bool, false);
        x.n_tokens = try f.value("n_tokens", usize, 0);
        x.n_bytes = try f.value("n_bytes", usize, 0);
        x.t_ms = try f.value("t_ms", f64, 0.0);
    } else if (std.mem.eql(u8, kind, "erase")) {
        r = try result.Result.create(gpa, .slot_erase);
        holder = r;
        r.kind.slot_erase.n_erased = try f.value("n_erased", usize, 0);
    } else if (std.mem.eql(u8, kind, "control")) {
        r = try result.Result.create(gpa, .control);
        holder = r;
        r.kind.control.success = try f.value("success", bool, false);
        r.kind.control.message = try r.alloc().dupe(u8, try f.value("message", []const u8, ""));
    } else if (std.mem.eql(u8, kind, "get_lora")) {
        r = try result.Result.create(gpa, .get_lora);
        holder = r;
        const a = r.alloc();
        var ls: std.ArrayList(result.GetLora.Lora) = .empty;
        if (f.contains("loras")) {
            var it = (try f.at("loras")).items();
            while (it.next()) |e| {
                const l = e.value;
                try ls.append(a, .{
                    .info = .{
                        .path = try a.dupeSentinel(u8, try l.value("path", []const u8, ""), 0),
                        .scale = try l.value("scale", f32, 0.0),
                        .task_name = try a.dupe(u8, try l.value("task_name", []const u8, "")),
                        .prompt_prefix = try a.dupe(u8, try l.value("prompt_prefix", []const u8, "")),
                        .ptr = null,
                    },
                    .alora_invocation_string = try a.dupe(u8, try l.value("inv", []const u8, "")),
                    .alora_invocation_tokens = try tokensOf(a, l, "inv_tokens"),
                });
            }
        }
        r.kind.get_lora.loras = ls.items;
    } else if (std.mem.eql(u8, kind, "apply_lora")) {
        r = try result.Result.create(gpa, .apply_lora);
        holder = r;
    } else if (std.mem.eql(u8, kind, "router")) {
        r = try result.Result.create(gpa, .router);
        holder = r;
        r.kind.router.data = try member(r.alloc(), f, "data", .null);
        const copy = try r.clone(gpa);
        defer copy.destroy(gpa);
        try str("router_clone", try dumpOwned(try copy.toJson(gpa)));
    } else {
        return error.BadKind;
    }
    r.id = try j.value("id", c_int, -1);
    r.id_slot = try j.value("id_slot", c_int, -1);
    r.index = try j.value("index", usize, 0);
    try linef("is_error {d} is_stop {d}", .{ @intFromBool(r.isError()), @intFromBool(r.isStop()) });
    const st = try j.value("state", i64, -1);
    if (st >= 0) {
        const s = states.get(st).?;
        try guarded(r.update(s));
        try str("state.text", s.generated_text.items);
        try str("state.content", s.chat_msg.content);
        try linef("state.flags {d}{d}{d}", .{ @intFromBool(s.thinking_block_started), @intFromBool(s.text_block_started), @intFromBool(s.oai_resp_created) });
    }
    try guarded(showResultJson(r));
}

fn showResultJson(r: *result.Result) !void {
    var v = try r.toJson(gpa);
    defer v.deinit(gpa);
    try str("json", try dumpV(&v));
    try str("sse_oai", try common.formatOaiSse(ra(), &v));
    try str("sse_anthropic", try common.formatAnthropicSse(ra(), &v));
    try guarded(sseResp(&v));
}

fn sseResp(v: *const Value) !void {
    try str("sse_resp", try common.formatOaiRespSse(ra(), v));
}

// ---------------------------------------------------------------------------
// server_tokens

fn showTokens(label: []const u8, t: *const ServerTokens) !void {
    const s = try t.str(gpa);
    defer gpa.free(s);
    try str(label, s);
}

fn delimTok(vocab: ?*const c.llama_vocab, a: std.mem.Allocator, text: []const u8) anyerror!std.ArrayList(i32) {
    return .fromOwnedSlice(try common.tokenize(a, vocab, text, false, true));
}

fn tokensOp(a: *ServerTokens, op: *Value) !void {
    const o = try (try op.atIndex(0)).get([]const u8);
    if (std.mem.eql(u8, o, "pos_next")) {
        // the arguments are clamped to the size: past it is GGML_ASSERT
        const n = @min(try (try op.atIndex(1)).get(i64), @as(i64, @intCast(a.size())));
        try linef("pos_next {d} {d}", .{ n, a.posNext(n) });
    } else if (std.mem.eql(u8, o, "size_up_to_pos")) {
        try linef("size_up_to_pos {d}", .{a.sizeUpToPos(try (try op.atIndex(1)).get(c_int))});
    } else if (std.mem.eql(u8, o, "keep_first")) {
        var b = try a.clone();
        defer b.deinit();
        try b.keepFirst(@min(try (try op.atIndex(1)).get(usize), b.size()));
        try showTokens("kept", &b);
    } else if (std.mem.eql(u8, o, "validate")) {
        try linef("validate {d}", .{@intFromBool(a.validate(text_ctx))});
    } else if (std.mem.eql(u8, o, "text")) {
        const t = try a.getTextTokens(gpa);
        defer gpa.free(t);
        try line(try std.mem.concat(ra(), u8, &.{ "text ", try toksStr(t) }));
    } else if (std.mem.eql(u8, o, "detok")) {
        // only tokens of the text model's vocabulary: libllama throws out_of_range past it
        const tt = try a.getTextTokens(ra());
        var ok = true;
        for (tt) |t| ok = ok and t >= 0 and t < c.llama_vocab_n_tokens(c.llama_model_get_vocab(text_model));
        if (!ok) return line("detok skipped");
        const s = try a.detokenize(gpa, text_ctx, try (try op.atIndex(1)).get(bool));
        defer gpa.free(s);
        try str("detok", s);
    } else if (std.mem.eql(u8, o, "serialize")) {
        const s = try a.serialize(gpa);
        defer gpa.free(s);
        try linef("serialized {d} {d}", .{ s.len, fnv(s) });
        const packed_toks = try ra().alloc(Token, s.len / 4);
        @memcpy(std.mem.sliceAsBytes(packed_toks), s);
        try guarded(restored("restored", packed_toks, true));
        try guarded(restored("restored_nomtmd", packed_toks, false));
        if (op.size() > 1) {
            // a corrupted copy: truncated to n tokens, one token overwritten
            var bad = try ra().dupe(Token, packed_toks);
            const n = try (try op.atIndex(1)).get(usize);
            if (n < bad.len) bad = bad[0..n] else if (n > bad.len) {
                const grown = try ra().alloc(Token, n);
                @memset(grown, 0);
                @memcpy(grown[0..bad.len], bad);
                bad = grown;
            }
            if (op.size() > 3 and try (try op.atIndex(2)).get(usize) < bad.len) bad[try (try op.atIndex(2)).get(usize)] = try (try op.atIndex(3)).get(Token);
            try guarded(restored("restored_bad", bad, a.has_mtmd));
        }
    } else if (std.mem.eql(u8, o, "clone")) {
        var b = try a.clone();
        defer b.deinit();
        try showTokens("clone", &b);
    } else if (std.mem.eql(u8, o, "prefix")) {
        const other = try (try op.atIndex(1)).getSlice(ra(), Token);
        var b = try ServerTokens.fromTokens(gpa, other, a.has_mtmd);
        defer b.deinit();
        try linef("prefix {d}", .{try a.getCommonPrefix(&b)});
    } else if (std.mem.eql(u8, o, "prefix_self")) {
        var b = try a.clone();
        defer b.deinit();
        if (op.size() > 1) try b.keepFirst(@min(try (try op.atIndex(1)).get(usize), b.size()));
        try linef("prefix_self {d} {d}", .{ try a.getCommonPrefix(&b), try b.getCommonPrefix(a) });
    } else if (std.mem.eql(u8, o, "prefix_edit")) {
        // a copy cut at n and continued with another token
        var b = try a.clone();
        defer b.deinit();
        try b.keepFirst(@min(try (try op.atIndex(1)).get(usize), b.size()));
        try b.pushBackToken(try (try op.atIndex(2)).get(Token));
        try linef("prefix_edit {d} {d}", .{ try a.getCommonPrefix(&b), try b.getCommonPrefix(a) });
    } else if (std.mem.eql(u8, o, "next_media")) {
        const res = a.findNextMediaChunk(try (try op.atIndex(1)).get(usize));
        try linef("next_media {d} {d}", .{ @intFromBool(res != null), if (res) |m| m.idx else 0 });
    } else if (std.mem.eql(u8, o, "spans")) {
        var d: chat.MsgDelimiters = .{};
        defer d.deinit(gpa);
        var it = (try op.atIndex(1)).items();
        while (it.next()) |e| {
            try d.add(gpa, chat.Role.fromString(try (try e.value.atIndex(0)).get([]const u8)), try (try e.value.atIndex(1)).get([]const u8));
        }
        try d.tokenize(gpa, c.llama_model_get_vocab(text_model), delimTok);
        var spans = try a.findMessageSpans(gpa, &d);
        defer spans.deinit(gpa);
        var s: std.ArrayList(u8) = .empty;
        try s.appendSlice(ra(), "spans");
        for (spans.spans.items) |sp| try s.print(ra(), " {d}:{d}:{d}", .{ @backingInt(sp.role), sp.pos, sp.len });
        try line(s.items);
    } else if (std.mem.eql(u8, o, "push_back")) {
        var b = try a.clone();
        defer b.deinit();
        var cc = try a.clone();
        defer cc.deinit();
        try b.pushBackTokens(&cc);
        try showTokens("pushed", &b);
    } else if (std.mem.eql(u8, o, "push_token")) {
        var b = try a.clone();
        defer b.deinit();
        try b.pushBackToken(try (try op.atIndex(1)).get(Token));
        try showTokens("pushed_token", &b);
    } else if (std.mem.eql(u8, o, "set_token")) {
        var b = try a.clone();
        defer b.deinit();
        if (b.size() > 0 and !b.has_mtmd) {
            b.setToken(@min(try (try op.atIndex(1)).get(c_int), @as(c_int, @intCast(b.size())) - 1), try (try op.atIndex(2)).get(Token));
        }
        try showTokens("set", &b);
    } else if (std.mem.eql(u8, o, "find_chunk")) {
        const ch = try a.findChunk(try (try op.atIndex(1)).get(usize));
        try linef("find_chunk {d}", .{c.mtmd_input_chunk_get_n_tokens(ch)});
    } else if (std.mem.eql(u8, o, "insert")) {
        var b = try a.clone();
        defer b.deinit();
        try b.insert(try (try op.atIndex(1)).getSlice(ra(), Token));
        try showTokens("inserted", &b);
    } else {
        return error.BadOp;
    }
}

fn restored(label: []const u8, packed_toks: []const Token, has_mtmd: bool) !void {
    var t = try ServerTokens.deserialize(gpa, packed_toks, has_mtmd);
    defer t.deinit();
    try showTokens(label, &t);
}

fn tokensOps(a: *ServerTokens, ops: *Value) !void {
    var it = ops.items();
    while (it.next()) |op| {
        try guarded(tokensOp(a, op.value));
    }
}

// ---------------------------------------------------------------------------

fn runOp(op: []const u8, j: *Value) !void {
    if (std.mem.eql(u8, op, "B")) {
        const b = base_arena.allocator();
        if (j.contains("n_parallel")) params_base.n_parallel = try (try j.at("n_parallel")).get(i32);
        if (j.contains("n_predict")) params_base.n_predict = try (try j.at("n_predict")).get(i32);
        if (j.contains("n_keep")) params_base.n_keep = try (try j.at("n_keep")).get(i32);
        if (j.contains("n_cache_reuse")) params_base.n_cache_reuse = try (try j.at("n_cache_reuse")).get(i32);
        if (j.contains("cache_prompt")) params_base.cache_prompt = try (try j.at("cache_prompt")).get(bool);
        if (j.contains("antiprompt")) {
            const xs = try (try j.at("antiprompt")).getSlice(b, []const u8);
            for (xs) |*x| x.* = try b.dupe(u8, x.*);
            params_base.antiprompt = .fromOwnedSlice(xs);
        }
        if (j.contains("sse_ping_interval")) params_base.sse_ping_interval = try (try j.at("sse_ping_interval")).get(i32);
        if (j.contains("verbosity")) params_base.verbosity = try (try j.at("verbosity")).get(i32);
        if (j.contains("reasoning_format")) params_base.reasoning_format = @fromBackingInt(@intCast(try (try j.at("reasoning_format")).get(c_int)));
        if (j.contains("seed")) params_base.sampling.seed = try (try j.at("seed")).get(u32);
        if (j.contains("temp")) params_base.sampling.temp = try (try j.at("temp")).get(f32);
        if (j.contains("dry_base")) params_base.sampling.dry_base = try (try j.at("dry_base")).get(f32);
        if (j.contains("samplers")) params_base.sampling.samplers = try schema.samplerTypesFromChars(b, try (try j.at("samplers")).get([]const u8));
        if (j.contains("spec_types")) {
            params_base.speculative.types = .empty;
            var it = (try j.at("spec_types")).items();
            while (it.next()) |t| try params_base.speculative.types.append(b, @fromBackingInt(@intCast(try t.value.get(c_int))));
        }
        try line("base ok");
    } else if (std.mem.eql(u8, op, "P")) {
        try guarded(opP(j));
    } else if (std.mem.eql(u8, op, "O")) {
        try guarded(opO(j));
    } else if (std.mem.eql(u8, op, "C")) {
        try guarded(opC(j));
    } else if (std.mem.eql(u8, op, "CS")) {
        try guarded(opCS(j));
    } else if (std.mem.eql(u8, op, "RC")) {
        try guarded(opRC(j));
    } else if (std.mem.eql(u8, op, "AC")) {
        try guarded(opAC(j));
    } else if (std.mem.eql(u8, op, "TC")) {
        try guarded(opTC(j));
    } else if (std.mem.eql(u8, op, "T")) {
        try guarded(opT(j));
        try guarded(opMixed(j));
    } else if (std.mem.eql(u8, op, "M")) {
        try guarded(opM(j));
    } else if (std.mem.eql(u8, op, "ST")) {
        try guarded(opST(j));
    } else if (std.mem.eql(u8, op, "U")) {
        var res: std.ArrayList(u8) = .empty;
        var it = j.items();
        while (it.next()) |h| {
            const s = try unhex(try h.value.get([]const u8));
            try res.print(ra(), "{d}{s} ", .{ common.validateUtf8(s), if (common.isValidUtf8(s)) "v" else "x" });
        }
        try line(try std.mem.concat(ra(), u8, &.{ "utf8 ", res.items }));
    } else if (std.mem.eql(u8, op, "E")) {
        try guarded(opE(j));
    } else if (std.mem.eql(u8, op, "GT")) {
        try guarded(opGT(j));
    } else if (std.mem.eql(u8, op, "S")) {
        try opS(j);
    } else if (std.mem.eql(u8, op, "R")) {
        try guarded(runResult(j));
    } else if (std.mem.eql(u8, op, "F")) {
        try guarded(opF(j));
    } else if (std.mem.eql(u8, op, "K")) {
        try guarded(opK(j));
    } else if (std.mem.eql(u8, op, "I")) {
        try guarded(opI(j));
    } else if (std.mem.eql(u8, op, "Q")) {
        try guarded(opQ(j));
    } else if (std.mem.eql(u8, op, "G")) {
        try guarded(opG(j));
    } else if (std.mem.eql(u8, op, "PC")) {
        try guarded(opPC(j));
    } else if (std.mem.eql(u8, op, "PIPE")) {
        try opPipe(j);
    } else if (std.mem.eql(u8, op, "L")) {
        try guarded(opL(j));
        if (j.contains("a")) try opLoraEq(j);
    } else if (std.mem.eql(u8, op, "J")) {
        try guarded(opJ(j));
    } else if (std.mem.eql(u8, op, "SSE")) {
        try guarded(sse(j, 0));
        try guarded(sse(j, 1));
        try guarded(sse(j, 2));
    } else if (std.mem.eql(u8, op, "RND")) {
        const r = try common.randomString(ra());
        var ok = r.len == 32;
        for (r) |ch| ok = ok and std.ascii.isAlphanumeric(ch);
        const cid = try common.genChatcmplid(ra());
        const r2 = try common.randomString(ra());
        try linef("random {d} {s} {d} {d} {d}", .{ @intFromBool(ok), cid[0..9], cid.len, (try common.genToolCallId(ra())).len, @intFromBool(!std.mem.eql(u8, r, r2)) });
        try str("marker", common.getMediaMarker());
    } else if (std.mem.eql(u8, op, "TK")) {
        try opTK(j);
    } else if (std.mem.eql(u8, op, "D")) {
        try guarded(opD(j));
    } else if (std.mem.eql(u8, op, "DR")) {
        try guarded(opDR(j));
    } else if (std.mem.eql(u8, op, "MA")) {
        try guarded(opMA(j));
    } else if (std.mem.eql(u8, op, "OE")) {
        try guarded(opOE(j));
        try guarded(opOESub(j));
    } else {
        try str("bad_op", op);
    }
}

fn opP(j: *Value) !void {
    const p = try evalParams(ra(), j);
    try guarded(toJsonLine(&p, "params", false));
    try guarded(toJsonLine(&p, "metrics", true));
    try str("extra", try dumpV(&(try paramsExtra(&p))));
}

fn toJsonLine(p: *const task.TaskParams, label: []const u8, only_metrics: bool) !void {
    try str(label, try dumpOwned(try p.toJson(gpa, only_metrics)));
}

fn opO(j: *Value) !void {
    try str("oaicmpl", try dumpOwned(try oaicompat.oaicompatCompletionParamsParse(gpa, j)));
}

/// `make_tmpls`: a template file's text, or a name or source given inline,
/// over the record's vocabulary.
fn makeTmpls(j: *const Value) !*oaicompat.Templates {
    const a = ra();
    const src = if (j.contains("tmpl_file"))
        try std.Io.Dir.cwd().readFileAlloc(g_io, try (try j.at("tmpl_file")).get([]const u8), a, .unlimited)
    else
        try j.value("tmpl", []const u8, "");
    var info = try chat.templates.modelInfo(a, @ptrCast(vocab_models.items[try j.value("v", usize, 0)]));
    const tmpls = try a.create(oaicompat.Templates);
    tmpls.* = try chat.templates.init(a, &info, src, "", "");
    return tmpls;
}

/// `make_chat_opt`.
fn makeChatOpt(j: *const Value) !oaicompat.ServerChatParams {
    const a = ra();
    const o = try j.at("opt");
    var opt: oaicompat.ServerChatParams = .{
        .use_jinja = try o.value("jinja", bool, false),
        .prefill_assistant = try o.value("prefill_assistant", bool, true),
        .reasoning_format = @fromBackingInt(@intCast(try o.value("reasoning_format", c_int, 0))),
        .tmpls = undefined,
        .allow_image = false,
        .allow_audio = false,
        .allow_video = false,
    };
    if (o.contains("kwargs")) {
        var it = (try o.at("kwargs")).items();
        while (it.next()) |item| try opt.chat_template_kwargs.put(a, item.key, try item.value.get([]const u8));
    }
    opt.tmpls = try makeTmpls(j);
    opt.allow_image = try o.value("allow_image", bool, false);
    opt.allow_audio = try o.value("allow_audio", bool, false);
    opt.allow_video = try o.value("allow_video", bool, false);
    opt.enable_thinking = try o.value("enable_thinking", bool, true);
    opt.reasoning_budget = try o.value("reasoning_budget", c_int, -1);
    opt.reasoning_budget_message = try o.value("reasoning_budget_message", []const u8, "");
    opt.media_path = try o.value("media_path", []const u8, "");
    opt.force_pure_content = try o.value("force_pure_content", bool, false);
    return opt;
}

fn opC(j: *Value) !void {
    var opt = try makeChatOpt(j);
    _ = &opt;
    var body = try (try j.at("body")).clone(gpa);
    defer body.deinit(gpa);
    var files: std.ArrayList([]u8) = .empty;
    defer {
        for (files.items) |f| gpa.free(f);
        files.deinit(gpa);
    }
    try guarded(chatParse(&body, &opt, &files));
    try str("body", try dumpV(&body));
    for (files.items) |f| {
        try linef("file {d} {d}", .{ f.len, fnv(f) });
    }
}

/// A chat request converted, its parameters evaluated, and a result state
/// made from them, as `CS` does in the reference.
fn opCS(j: *Value) !void {
    const a = ra();
    const opt = try makeChatOpt(j);
    var body = try (try j.at("body")).clone(a);
    var files: std.ArrayList([]u8) = .empty;
    const parsed = try oaicompat.oaicompatChatParamsParse(a, &body, &opt, &files);
    try str("chat", try dumpV(&parsed));
    var pj = Value.emptyObject();
    try pj.set(a, "v", try Value.from(a, try j.value("v", c_int, 0)));
    try pj.set(a, "data", parsed);
    var t = task.Task.initType(.completion);
    defer t.deinit(gpa);
    const ta = try t.arenaAlloc(gpa);
    t.params = try evalParams(ta, &pj);
    try str("extra", try dumpV(&(try paramsExtra(&t.params))));
    try newStateQuiet(&t, try (try j.at("id")).get(i64));
}

fn opRC(j: *Value) !void {
    try str("conv", try dumpV(&(try sd.convert.responsesToChatcmpl(ra(), j))));
}

fn opAC(j: *Value) !void {
    try str("conv", try dumpV(&(try sd.convert.anthropicToOai(ra(), j))));
}

fn opTC(j: *Value) !void {
    const a = ra();
    const tmpls = try makeTmpls(j);
    var file: ?[]const u8 = null;
    if (j.contains("file") and std.mem.eql(u8, try j.value("file_key", []const u8, "file"), "file")) {
        file = try unhex(try (try j.at("file")).get([]const u8));
    }
    var files: std.ArrayList([]u8) = .empty;
    try files.append(a, try a.dupe(u8, &.{ 1, 2, 3 })); // cleared by the conversion
    try guarded(opTCConv(j, tmpls, file, &files));
    for (files.items) |f| {
        try linef("file {d} {d}", .{ f.len, fnv(f) });
    }
}

fn opTCConv(j: *Value, tmpls: *const oaicompat.Templates, file: ?[]const u8, files: *std.ArrayList([]u8)) !void {
    try str("conv", try dumpV(&(try sd.convert.transcriptionsToChatcmpl(ra(), try j.at("body"), tmpls, file, files))));
}

fn chatParse(body: *Value, opt: *const oaicompat.ServerChatParams, files: *std.ArrayList([]u8)) !void {
    try str("chat", try dumpOwned(try oaicompat.oaicompatChatParamsParse(gpa, body, opt, files)));
}

fn opT(j: *Value) !void {
    const vocab = try vocabAt(j);
    var res = try tokens.tokenizeInputPrompts(gpa, vocab, if (try j.value("mtmd", bool, false)) mctx else null, try j.at("prompt"), try j.value("add_special", bool, true), try j.value("parse_special", bool, true), init_opt);
    defer {
        for (res.items) |*t| t.deinit();
        res.deinit(gpa);
    }
    try linef("n_prompts {d}", .{res.items.len});
    for (res.items) |*t| {
        try showTokens("st", t);
        if (j.contains("ops")) try tokensOps(t, try j.at("ops"));
    }
}

fn opMixed(j: *Value) !void {
    const t = try tokens.tokenizeMixed(gpa, try vocabAt(j), try j.at("prompt"), try j.value("add_special", bool, true), try j.value("parse_special", bool, true));
    defer gpa.free(t);
    try line(try std.mem.concat(ra(), u8, &.{ "mixed ", try toksStr(t) }));
}

fn opM(j: *Value) !void {
    var files: std.ArrayList([]const u8) = .empty;
    if (j.contains("files")) {
        var it = (try j.at("files")).items();
        while (it.next()) |f| try files.append(ra(), try unhex(try f.value.get([]const u8)));
    }
    var t = try tokens.processMtmdPrompt(gpa, mctx, try (try j.at("prompt")).get([]const u8), files.items, init_opt, try j.value("ph", bool, false));
    defer t.deinit();
    try showTokens("mtmd", &t);
    if (j.contains("ops")) try tokensOps(&t, try j.at("ops"));
}

fn opST(j: *Value) !void {
    var t = try ServerTokens.fromTokens(gpa, try (try j.at("tokens")).getSlice(ra(), Token), false);
    defer t.deinit();
    try showTokens("st", &t);
    if (j.contains("ops")) try tokensOps(&t, try j.at("ops"));
}

fn opE(j: *Value) !void {
    const t: common.ErrorType = @fromBackingInt(@intCast(try (try j.at("type")).get(c_int)));
    try str("error", try dumpOwned(try common.formatErrorResponse(gpa, try j.value("msg", []const u8, ""), t)));
}

fn opGT(j: *Value) !void {
    const t = try common.GrammarTrigger.fromJson(ra(), j);
    try str("trigger", try dumpOwned(try t.toJson(gpa)));
}

fn opS(j: *Value) !void {
    const p = try j.at("parser");
    var t = task.Task.initType(.completion);
    defer t.deinit(gpa);
    const a = try t.arenaAlloc(gpa);
    t.params.chat_parser_params.format = try p.value("format", c_int, 0);
    t.params.chat_parser_params.generation_prompt = try a.dupe(u8, try p.value("generation_prompt", []const u8, ""));
    t.params.chat_parser_params.is_continuation = try p.value("is_continuation", bool, false);
    t.params.chat_parser_params.echo = try p.value("echo", bool, false);
    t.params.chat_parser_params.reasoning_format = @fromBackingInt(@intCast(try p.value("reasoning_format", c_int, 0)));
    t.params.chat_parser_params.reasoning_in_content = try p.value("reasoning_in_content", bool, false);
    try guarded(newState(&t, try (try j.at("id")).get(i64)));
}

/// `newState` without the ids line, as `CS` writes it.
fn newStateQuiet(t: *const task.Task, id: i64) !void {
    const s = try gpa.create(task.TaskResultState);
    errdefer gpa.destroy(s);
    s.* = try t.createState(gpa);
    if (try states.fetchPut(gpa, id, s)) |old| {
        old.value.deinit(gpa);
        gpa.destroy(old.value);
    }
    try str("state.init", s.chat_msg.content);
    try str("state.role", s.chat_msg.role);
}

fn newState(t: *const task.Task, id: i64) !void {
    const s = try gpa.create(task.TaskResultState);
    errdefer gpa.destroy(s);
    s.* = try t.createState(gpa);
    if (try states.fetchPut(gpa, id, s)) |old| {
        old.value.deinit(gpa);
        gpa.destroy(old.value);
    }
    try str("state.init", s.chat_msg.content);
    try str("state.role", s.chat_msg.role);
    try linef("state.ids {d} {d} {d}", .{ s.oai_resp_id.len, s.oai_resp_reasoning_id.len, s.oai_resp_message_id.len });
}

fn opF(j: *Value) !void {
    var emb = try (try j.at("emb")).clone(ra());
    try str("embd", try dumpOwned(try oaicompat.formatEmbeddingsResponseOaicompat(gpa, try j.at("req"), try j.value("model", []const u8, ""), &emb, try j.value("b64", bool, false))));
}

fn opK(j: *Value) !void {
    const texts = try stringsOf(ra(), j, "texts");
    try str("rerank", try dumpOwned(try oaicompat.formatResponseRerank(gpa, try j.at("req"), try j.value("model", []const u8, ""), try j.at("ranks"), try j.value("tei", bool, false), @constCast(texts), try j.value("top_n", c_int, 0))));
}

fn opI(j: *Value) !void {
    const prompt = if (j.contains("prompt")) try (try j.at("prompt")).getSlice(ra(), Token) else &[_]Token{};
    const toks = try tokens.formatPromptInfill(gpa, try vocabAt(j), try j.at("prefix"), try j.at("suffix"), try j.at("extra"), try j.value("n_batch", c_int, 512), try j.value("n_predict", c_int, -1), try j.value("n_ctx", c_int, 4096), try j.value("spm", bool, false), prompt);
    defer gpa.free(toks);
    try line(try std.mem.concat(ra(), u8, &.{ "infill ", try toksStr(toks) }));
}

fn opQ(j: *Value) !void {
    const m = vocab_models.items[try j.value("v", usize, 0)];
    var t = try tokens.formatPromptRerank(gpa, m, try vocabAt(j), null, try (try j.at("query")).get([]const u8), try (try j.at("doc")).get([]const u8), init_opt);
    defer t.deinit();
    try showTokens("rerank_prompt", &t);
}

fn opG(j: *Value) !void {
    const vocab = c.llama_model_get_vocab(text_model);
    const toks = try common.tokenize(ra(), vocab, try (try j.at("prompt")).get([]const u8), true, true);
    c.llama_memory_clear(c.llama_get_memory(text_ctx), true);
    var batch = tools.common.Batch.init(gpa, text_ctx);
    defer batch.deinit();
    for (toks, 0..) |t, i| _ = batch.add(t, @intCast(i), 0, true);
    if (c.llama_process(text_ctx, c.LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) return common.throw(error.RuntimeError, "decode failed", .{});
    var it = (try j.at("n_top")).items();
    while (it.next()) |nt| {
        const idx = try j.value("idx", c_int, -1);
        const cur = try common.getTokenProbabilities(gpa, text_ctx, idx, try nt.value.get(usize));
        defer gpa.free(cur);
        var s: std.ArrayList(u8) = .empty;
        try s.print(ra(), "probs n={d}", .{cur.len});
        var sum: f32 = 0.0;
        for (cur, 0..) |t, i| {
            sum += t.p;
            var b1: [8]u8 = undefined;
            var b2: [8]u8 = undefined;
            if (i < 12) try s.print(ra(), " {d}:{s}:{s}", .{ t.id, hexBits(&b1, t.logit), hexBits(&b2, t.p) });
        }
        var b3: [8]u8 = undefined;
        try s.print(ra(), " sum={s}", .{hexBits(&b3, sum)});
        try line(s.items);
    }
    // the prompt's own tokens: the fixture vocabulary throws out_of_range on some others
    for (toks) |t| try fmtTok(t);
    try fmtTok(c.LLAMA_TOKEN_NULL);
    const s = try common.tokensToStrCtx(gpa, text_ctx, toks);
    defer gpa.free(s);
    try str("tokens_to_str", s);
}

fn fmtTok(t: Token) !void {
    const s = try common.tokensToOutputFormattedString(gpa, text_ctx, t);
    defer gpa.free(s);
    try str("fmt", s);
}

fn opPC(j: *Value) !void {
    var cache = task.PromptCache.init(gpa, try j.value("limit_mib", i32, 0), try j.value("limit_tokens", usize, 0));
    defer cache.deinit();
    var cur: task.ServerPrompt = .{ .tokens = .{ .gpa = gpa } };
    defer cur.deinit();
    var it = (try j.at("ops")).items();
    while (it.next()) |o| {
        try guarded(pcOp(&cache, &cur, o.value));
    }
}

fn pcOp(cache: *task.PromptCache, cur: *task.ServerPrompt, o: *Value) !void {
    const k = try (try o.atIndex(0)).get([]const u8);
    if (std.mem.eql(u8, k, "alloc")) {
        var p: task.ServerPrompt = .{ .tokens = try ServerTokens.fromTokens(gpa, try (try o.atIndex(1)).getSlice(ra(), Token), false) };
        defer p.deinit();
        if (o.size() > 4) {
            var sit = (try o.atIndex(4)).items();
            while (sit.next()) |sz| {
                const data = try gpa.alloc(u8, try sz.value.get(usize));
                @memset(data, 0);
                try p.checkpoints.append(gpa, .{ .data_tgt = data });
            }
        }
        const st = try cache.alloc(&p, try (try o.atIndex(2)).get(usize), try (try o.atIndex(3)).get(usize));
        try linef("alloc {d}", .{@intFromBool(st != null)});
    } else if (std.mem.eql(u8, k, "update")) {
        cache.update();
        try line("update");
    } else if (std.mem.eql(u8, k, "cur")) {
        cur.deinit();
        cur.* = .{ .tokens = try ServerTokens.fromTokens(gpa, try (try o.atIndex(1)).getSlice(ra(), Token), false) };
        try line("cur");
    } else if (std.mem.eql(u8, k, "load")) {
        var nt = try ServerTokens.fromTokens(gpa, try (try o.atIndex(1)).getSlice(ra(), Token), false);
        defer nt.deinit();
        const ok = try cache.load(cur, &nt, text_ctx, null, 0);
        const tt = try cur.tokens.getTextTokens(ra());
        try linef("load {d} {s}", .{ @intFromBool(ok), try toksStr(tt) });
    }
    var s: std.ArrayList(u8) = .empty;
    try s.print(ra(), "cache {d} {d} {d} {d}", .{ cache.states.items.len, cache.size(), cache.nTokens(), cache.limit_size });
    for (cache.states.items) |st| {
        try s.print(ra(), " [{d},{d},{d},{d}]", .{ st.prompt.tokens.size(), st.data.main.len, st.data.drft.len, st.prompt.checkpoints.items.len });
    }
    try line(s.items);
}

fn alwaysStop(_: *anyopaque) bool {
    return true;
}

fn opPipe(j: *Value) !void {
    var p = common.Pipe(c_int).init(gpa);
    defer p.deinit();
    p.max_size = try j.value("max_size", usize, 0);
    var s: std.ArrayList(u8) = .empty;
    try s.appendSlice(ra(), "pipe");
    var dummy: u8 = 0;
    var it = (try j.at("ops")).items();
    while (it.next()) |o| {
        const k = try (try o.value.atIndex(0)).get([]const u8);
        if (std.mem.eql(u8, k, "w")) {
            try s.print(ra(), " w{d}", .{@intFromBool(try p.write(try (try o.value.atIndex(1)).get(c_int)))});
        } else if (std.mem.eql(u8, k, "r")) {
            var v: c_int = -1;
            const close_on_stop = if (o.value.size() > 1) try (try o.value.atIndex(1)).get(bool) else true;
            const ok = p.read(&v, .{ .ctx = &dummy, .f = alwaysStop }, close_on_stop);
            try s.print(ra(), " r{s}{d}", .{ if (ok) "1:" else "0:", v });
        } else if (std.mem.eql(u8, k, "cw")) {
            p.closeWrite();
            try s.appendSlice(ra(), " cw");
        } else if (std.mem.eql(u8, k, "cr")) {
            p.closeRead();
            try s.appendSlice(ra(), " cr");
        }
    }
    try line(s.items);
}

fn opL(j: *Value) !void {
    const m = try common.parseLoraRequest(ra(), try j.at("req"));
    var s: std.ArrayList(u8) = .empty;
    try s.appendSlice(ra(), "lora");
    for (m.entries.items) |e| {
        var b: [8]u8 = undefined;
        try s.print(ra(), " {d}:{s}", .{ e.id, hexBits(&b, e.scale) });
    }
    try line(s.items);
}

fn opLoraEq(j: *Value) !void {
    var la: std.ArrayList(common.LoraInfo) = .empty;
    var lb: std.ArrayList(common.LoraInfo) = .empty;
    var it = (try j.at("a")).items();
    while (it.next()) |sc| try la.append(ra(), .{ .path = "", .scale = try sc.value.get(f32) });
    var it2 = (try j.at("b")).items();
    while (it2.next()) |sc| try lb.append(ra(), .{ .path = "", .scale = try sc.value.get(f32) });
    var s: std.ArrayList(u8) = .empty;
    try s.print(ra(), "lora_eq {d} ids", .{@intFromBool(common.areLoraEqual(la.items, lb.items))});
    for (try common.loraGetEnabledIds(ra(), la.items)) |id| try s.print(ra(), " {d}", .{id});
    try line(s.items);
}

fn opJ(j: *Value) !void {
    const js = try j.at("js");
    const paths = try (try j.at("paths")).getSlice(ra(), []const u8);
    try str("nested", try dumpOwned(try common.jsonGetNestedValues(gpa, paths, js)));
    try linef("is {d}{d}{d}", .{ @intFromBool(common.jsonIsArrayOfNumbers(js)), @intFromBool(common.jsonIsArrayOfMixedNumbersStrings(js)), @intFromBool(common.jsonIsArrayAndContainsNumbers(js)) });
}

fn sse(j: *Value, which: u8) !void {
    switch (which) {
        0 => try str("sse_oai", try common.formatOaiSse(ra(), j)),
        1 => try str("sse_anthropic", try common.formatAnthropicSse(ra(), j)),
        else => try str("sse_resp", try common.formatOaiRespSse(ra(), j)),
    }
}

fn opTK(j: *Value) !void {
    var t = task.Task.initType(@fromBackingInt(@intCast(try j.value("type", c_int, 0))));
    t.params.sampling.seed = try j.value("seed", u32, c.LLAMA_DEFAULT_SEED);
    t.id = 10;
    const n = try j.value("children", c_int, 0);
    var i: c_int = 0;
    while (i < n) : (i += 1) try t.addChild(gpa, 10, 11 + i);
    var s: std.ArrayList(u8) = .empty;
    try s.print(ra(), "task embd{d} logits{d} sampling{d}", .{ @intFromBool(t.needEmbd()), @intFromBool(t.needLogits()), @intFromBool(t.needSampling()) });
    try s.print(ra(), " parent{d}", .{@intFromBool(t.isParent())});
    for (t.child_tasks.items) |ch| try s.print(ra(), " {d}:{d}:{d}:{s}", .{ ch.id, ch.id_parent, ch.params.sampling.seed, if (ch.isChild()) "c" else "-" });
    const v = [_]task.Task{t};
    var ids = try task.Task.getListId(gpa, &v);
    defer ids.deinit();
    try s.print(ra(), " ids {d}", .{ids.count()});
    t.deinit(gpa);
    try line(s.items);
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 6) {
        std.debug.print("usage: server_task_port <records> <transcript> <text-model> <mmproj> <vocab>...\n", .{});
        return 2;
    }
    log.verbosity_thold = log.level_debug;
    c.llama_log_set(silent, null);
    c.mtmd_helper_log_set(silent, null);
    chat.msg.log_sink = chatLog;
    chat.log.sink = chatLogAll;
    c.llama_backend_init();

    base_arena = .init(gpa);
    params_base = try tools.params.Params.init(base_arena.allocator());
    params_base.n_parallel = 4;
    init_opt = c.mtmd_helper_init_opt_default();

    for (argv[5..]) |p| {
        var mp = c.llama_model_default_params();
        mp.vocab_only = true;
        const m = c.llama_model_load_from_file(p.ptr, mp) orelse {
            std.debug.print("cannot load {s}\n", .{p});
            return 2;
        };
        try vocab_models.append(gpa, m);
    }
    {
        var mp = c.llama_model_default_params();
        mp.n_gpu_layers = 0;
        text_model = c.llama_model_load_from_file(argv[3].ptr, mp) orelse {
            std.debug.print("cannot load {s}\n", .{argv[3]});
            return 2;
        };
        var cp = c.llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_threads = 1;
        cp.n_threads_batch = 1;
        text_ctx = c.llama_init_from_model(text_model, cp);
        var mpar = c.mtmd_context_params_default();
        mpar.use_gpu = false;
        mpar.print_timings = false;
        mpar.n_threads = 1;
        mpar.warmup = false;
        mpar.media_marker = common.getMediaMarker().ptr;
        const no_mmproj = std.mem.eql(u8, argv[4], "-");
        mctx = if (no_mmproj) null else c.mtmd_init_from_file(argv[4].ptr, text_model, mpar);
        if (text_ctx == null or (mctx == null and !no_mmproj)) {
            std.debug.print("cannot make the contexts\n", .{});
            return 2;
        }
    }

    const io = init.io;
    g_io = io;
    const input = try std.Io.Dir.cwd().readFileAlloc(io, argv[1], arena, .unlimited);
    var lines = std.mem.splitScalar(u8, input, '\n');
    var n: usize = 0;
    while (lines.next()) |ln| {
        if (ln.len == 0 and lines.peek() == null) break;
        rec_arena = .init(gpa);
        defer rec_arena.deinit();
        const sp = std.mem.indexOfScalar(u8, ln, ' ');
        const op = if (sp) |i| ln[0..i] else ln;
        const payload = if (sp) |i| ln[i + 1 ..] else "";
        const opz = try ra().dupeSentinel(u8, op, 0);
        log.wrn("@@ %d %s\n", .{ @as(c_int, @intCast(n)), opz.ptr });
        try linef("@@ {d} {s}", .{ n, op });
        if (json.parse(ra(), payload)) |parsed| {
            var j = parsed;
            try runOp(op, &j);
        } else |e| switch (e) {
            error.JsonError => try str("bad_input", json.what()),
            else => return e,
        }
        n += 1;
    }

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = argv[2], .data = out.items });
    out.deinit(gpa);

    var sit = states.iterator();
    while (sit.next()) |e| {
        e.value_ptr.*.deinit(gpa);
        gpa.destroy(e.value_ptr.*);
    }
    states.deinit(gpa);
    if (mctx) |m| c.mtmd_free(m);
    c.llama_free(text_ctx);
    c.llama_model_free(text_model);
    for (vocab_models.items) |m| c.llama_model_free(m);
    vocab_models.deinit(gpa);
    base_arena.deinit();
    if (gpa_state.deinit() == .leak) {
        std.debug.print("leaks\n", .{});
        return 3;
    }
    return 0;
}

fn silent(_: c.ggml_log_level, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {}

fn chatLogAll(level: chat.log.Level, text: []const u8) void {
    const z = common.dupZ(text);
    defer common.freeZ(z);
    switch (level) {
        .debug => log.dbg("%s", .{z}),
        .info => log.inf("%s", .{z}),
        .warn => log.wrn("%s", .{z}),
        .err => log.err("%s", .{z}),
    }
}

fn chatLog(level: chat.msg.Level, text: []const u8) void {
    const z = common.dupZ(text);
    defer common.freeZ(z);
    switch (level) {
        .warn => log.wrn("%s", .{z}),
        .err => log.err("%s", .{z}),
    }
}

// ---------------------------------------------------------------------------
// the typed decision API (v0.6.0)

fn showDecisionTask(t: *const task.Task) !void {
    const d = &t.decision;
    try linef("task {d} labels{s} markers{s} column{d} pointer{d} first{d} order{s} n_scores{d} embd{d} logits{d}", .{
        t.id,         try toksStr(d.labels.items), try toksStr(d.markers.items), d.column,                   d.pointer,
        d.posFirst(), try toksStr(d.order.items),  d.n_scores,                   @intFromBool(t.needEmbd()), @intFromBool(t.needLogits()),
    });
    try showTokens("prompt", &t.tokens);
}

/// `D`: a /v1/systemone request through a decision model's context.
fn opD(j: *Value) !void {
    const a = ra();
    const decision = sd.decision;
    const i = try j.value("v", usize, 0);
    if (i >= vocab_models.items.len) return error.JsonError;
    const model = vocab_models.items[i];
    try linef("decision_type {d}", .{@backingInt(tools.common.getDecisionType(model))});
    var dc: decision.DecisionContext = .{};
    defer dc.deinit();
    try dc.init(gpa, model);
    try linef("flags share{d} joint{d} images{d} type{d}", .{ @intFromBool(dc.canSharePrompt()), @intFromBool(dc.isJoint()), @intFromBool(dc.canUseImages()), @backingInt(dc.type) });
    if (dc.type == .none) return;

    const body = try j.at("body");
    const questions = try dc.parseQuestions(a, body);
    for (questions) |*q| {
        var s: std.ArrayList(u8) = .empty;
        try s.print(a, "q {s} {d} {s}", .{ q.id, @backingInt(q.type), try dumpV(&q.instructions) });
        for (q.options) |*o| try s.print(a, " [{s}|{s}]", .{ o.key, try dumpV(&o.description) });
        try str("question", s.items);
        try linef("variants {d}", .{dc.nVariants(q)});
    }
    var files: std.ArrayList([]u8) = .empty;
    const state = try dc.parseState(a, body, &files);
    try str("state", try dumpV(&state));
    for (files.items) |f| try linef("file {d} {d}", .{ f.len, fnv(f) });
    // the model is vocab-only: a request with files would tokenize them through mtmd
    if (files.items.len != 0) return;

    var tasks: std.ArrayList(task.Task) = .empty;
    defer {
        for (tasks.items) |*t| t.deinit(gpa);
        tasks.deinit(gpa);
    }
    var id: c_int = 100;
    const no_files: []const []const u8 = &.{};
    if (dc.isJoint()) {
        var t = task.Task.initType(.decision);
        errdefer t.deinit(gpa);
        t.id = id;
        id += 1;
        try dc.fillTaskJoint(a, gpa, &state, questions, no_files, null, init_opt, &t);
        try showDecisionTask(&t);
        try tasks.append(gpa, t);
    } else {
        for (questions) |*q| {
            for (0..dc.nVariants(q)) |v| {
                var t = task.Task.initType(.decision);
                errdefer t.deinit(gpa);
                t.id = id;
                id += 1;
                try dc.fillTask(a, gpa, &state, questions, q, v, no_files, null, init_opt, &t);
                try showDecisionTask(&t);
                try tasks.append(gpa, t);
            }
        }
    }
    // the outputs each task reads: its labels, its markers, or for a joint head one per option
    var n_out: std.ArrayList(usize) = .empty;
    for (tasks.items) |*t| {
        const d = &t.decision;
        try n_out.append(a, if (d.labels.items.len != 0) d.labels.items.len else if (d.markers.items.len != 0) d.markers.items.len else @intCast(d.n_scores));
    }
    const pool: []const f32 = if (j.contains("pool")) try (try j.at("pool")).getSlice(a, f32) else &.{};
    var i_pool: usize = 0;
    const Take = struct {
        fn run(al: std.mem.Allocator, pl: []const f32, ip: *usize, n: usize) ![]f32 {
            const v = try al.alloc(f32, n);
            for (v) |*x| {
                x.* = if (pl.len == 0) 0.0 else pl[ip.* % pl.len];
                if (pl.len != 0) ip.* += 1;
            }
            return v;
        }
    };
    const nan_at = try j.value("nan_at", c_int, -1);
    const short = try j.value("short", bool, false);
    var i_task: usize = 0;
    var i_score: usize = 0;
    var n_answer: c_int = 0;
    for (questions) |*q| {
        var scores: std.ArrayList([]f32) = .empty;
        if (dc.isJoint()) {
            const all = try Take.run(a, pool, &i_pool, n_out.items[0]);
            const lo = @min(all.len, i_score);
            const hi = @min(all.len, i_score + q.options.len);
            try scores.append(a, all[lo..hi]);
            i_score += q.options.len;
        } else {
            for (0..dc.nVariants(q)) |_| {
                try scores.append(a, try Take.run(a, pool, &i_pool, n_out.items[i_task]));
                i_task += 1;
            }
        }
        const this = n_answer;
        n_answer += 1;
        if (this == nan_at and scores.items.len != 0 and scores.items[0].len != 0) scores.items[0][0] = std.math.nan(f32);
        if (short and scores.items.len != 0 and scores.items[0].len != 0) scores.items[0] = scores.items[0][0 .. scores.items[0].len - 1];
        const cs = try a.alloc([]const f32, scores.items.len);
        for (scores.items, cs) |x, *y| y.* = x;
        try guarded(answerLine(&dc, q, cs));
    }
    if (dc.canSharePrompt()) {
        var groups = try decision.groupTasks(gpa, tasks.items, try j.value("n_slots", usize, 1));
        defer {
            for (groups.items) |*g| g.deinit(gpa);
            groups.deinit(gpa);
        }
        for (groups.items) |*g| {
            var s: std.ArrayList(u8) = .empty;
            try s.print(a, "group {d} shared{d} n{d}", .{ g.id, g.n_tokens_shared, g.tokens.size() });
            for (g.child_tasks.items) |*ch| try s.print(a, " {d}:{d}:{d}", .{ ch.id, ch.id_parent, ch.tokens.size() });
            try line(s.items);
        }
    }
}

fn answerLine(dc: *const sd.decision.DecisionContext, q: *const sd.decision.Question, scores: []const []const f32) !void {
    try str("answer", try dumpV(&(try dc.formatAnswer(ra(), q, scores))));
}

/// `DR`: `server_task_result_decision::to_json`.
fn opDR(j: *Value) !void {
    const r = try result.Result.create(gpa, .decision);
    defer r.destroy(gpa);
    r.index = try j.value("index", usize, 0);
    r.kind.decision.n_tokens = try j.value("n_tokens", i32, 0);
    r.kind.decision.scores = if (j.contains("scores")) try (try j.at("scores")).getSlice(r.alloc(), f32) else &.{};
    try str("decision_result", try dumpV(&(try r.toJson(ra()))));
}

/// `MA`: the architecture object of GET /models.
fn opMA(j: *Value) !void {
    const ty: tools.common.DecisionType = @fromBackingInt(@intCast(try j.value("type", c_int, 0)));
    const out_mod = common.serverModelOutputModalities(ty);
    var v = try common.serverModelArchitectureJson(ra(), try j.value("image", bool, false), try j.value("audio", bool, false), try j.value("video", bool, false), out_mod);
    try str("arch", try dumpV(&v));
}

/// `OE`: an /embeddings content array, and the same input as a subprompt.
fn opOE(j: *Value) !void {
    const opt = try makeChatOpt(j);
    var t = try oaicompat.tokenizeOaiContentArray(gpa, try vocabAt(j), if (try j.value("mtmd", bool, false)) mctx else null, &opt, try j.at("content"), try j.value("add_special", bool, true), try j.value("parse_special", bool, true), init_opt);
    defer t.deinit();
    try showTokens("oe", &t);
}

fn opOESub(j: *Value) !void {
    var t = try tokens.tokenizeInputSubprompt(gpa, try vocabAt(j), if (try j.value("mtmd", bool, false)) mctx else null, try j.at("content"), try j.value("add_special", bool, true), try j.value("parse_special", bool, true), init_opt);
    defer t.deinit();
    try showTokens("sub", &t);
}
