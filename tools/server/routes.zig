//! Every route's handler, and the response they answer with:
//! `server_routes` and `server_res_generator`.
//!
//! # Provenance
//!
//! Ports the rest of `llama.cpp/tools/server/server-context.cpp` and
//! `llama.cpp/tools/server/server-context.h` (llama.cpp v1.0.0,
//! `b1b770d90`): `server_res_generator`, `server_routes` with every handler
//! of `init_routes`, `handle_completions_impl`, the slot actions, the
//! embeddings and token-count handlers, `update_cached_responses`, and
//! `get_res_model_info`, `get_res_models` and `get_res_props`. Each
//! declaration names what it ports and the line it began at.
//!
//! # The chat APIs
//!
//! The Responses, Anthropic and transcription routes convert their requests
//! into `/chat/completions` bodies with `convert.zig` (`server-chat.cpp`'s
//! conversions), then run them as chat; their token-count routes likewise.
//! Chat itself renders through `oaicompat.chatTemplatesApply`, Jinja or
//! legacy.
//!
//! # Handlers
//!
//! A handler is a method taking `*const http.Req` and returning the
//! response, an error being the exception `ex_wrapper` catches. Every one
//! starts with `createResponse`, as upstream's do, so that a request made
//! during sleep wakes the server first.

const std = @import("std");
const mtmd = @import("mtmd");
const llama = @import("llama");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const json = @import("json");
const common_chat = @import("common_chat");
const tools_common = @import("tools_common");

fn ggmlAssert(ok: bool, comptime line: c_int, comptime cond: [*:0]const u8) void {
    tools_common.ggml.ggmlAssert(ok, "server-context.cpp", line, cond);
}

// -----------------------------------------------------------------------------
// server_res_generator

/// Ports `server_res_generator` (server-context.cpp:5304 @b1b770d90): a
/// generator-like response, its reader on the task queues, and the stream
/// replay tee it derives from (`server_res_spipe`).
pub const ResGenerator = struct {
    base: mod.http.Res,
    spipe: mod.stream.Spipe = .{},
    rd: mod.context.Reader,
    queue_awake: ?*mod.context.TaskQueue = null, // set while this response keeps the server awake

    /// What the streaming lambda of `handle_completions_impl` captures.
    res_type: mod.task.ResponseType = .none,
    sse_ping_interval: i32 = 0,

    /// What the streaming lambda of `handle_tts_impl` captures.
    tts_is_done: bool = false,
    tts_pcm16: bool = false,

    const vtable: mod.http.Res.VTable = .{ .on_complete = onComplete, .destroy = destroyGen };

    /// Ports `server_res_generator`'s constructor (server-context.cpp:5307
    /// @b1b770d90): may bypass sleep if the task does not use the context;
    /// otherwise the server stays awake until the response is destroyed.
    fn create(queue_tasks: *mod.context.TaskQueue, queue_results: *mod.context.ResultQueue, sleep_idle_seconds: c_int, bypass_sleep_in: bool) !*ResGenerator {
        const self = try std.heap.c_allocator.create(ResGenerator);
        errdefer std.heap.c_allocator.destroy(self);
        self.* = .{
            .base = .{ .arena = std.heap.ArenaAllocator.init(std.heap.c_allocator), .vtable = &vtable },
            .rd = try mod.context.Reader.init(std.heap.c_allocator, queue_tasks, queue_results, mod.context.http_polling_seconds),
        };
        // fast path in case sleeping is disabled
        const bypass_sleep = bypass_sleep_in or sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.waitUntilNoSleep();
            self.queue_awake = queue_tasks;
        }
        return self;
    }

    fn onComplete(res: *mod.http.Res) void {
        const self: *ResGenerator = @fieldParentPtr("base", res);
        self.spipe.onComplete();
    }

    /// Ports the destructor of `server_res_generator` (server-context.cpp:5316
    /// @b1b770d90), then its members' destructors.
    fn destroyGen(res: *mod.http.Res) void {
        const self: *ResGenerator = @fieldParentPtr("base", res);
        if (self.queue_awake) |q| {
            q.releaseInflight();
        }
        self.rd.deinit();
        self.spipe.deinit();
        self.base.arena.deinit();
        std.heap.c_allocator.destroy(self);
    }

    pub fn destroy(self: *ResGenerator) void {
        destroyGen(&self.base);
    }

    fn alloc(self: *ResGenerator) std.mem.Allocator {
        return self.base.allocator();
    }

    /// Ports `server_res_generator`'s `ok` (server-context.cpp:5321 @b1b770d90).
    fn ok(self: *ResGenerator, response_data: *const json.Value) !void {
        self.base.status = 200;
        try self.base.setData(try mod.common.safeJsonToStr(self.alloc(), response_data));
    }

    /// Ports `server_res_generator`'s `error` (server-context.cpp:5325 @b1b770d90).
    fn err(self: *ResGenerator, error_data: *const json.Value) !void {
        const a = self.alloc();
        self.base.status = mod.common.jsonValue(error_data, "code", c_int, 500);
        var outer = json.Value.emptyObject();
        try outer.set(a, "error", try error_data.clone(a));
        try self.base.setData(try mod.common.safeJsonToStr(a, &outer));
    }

    /// `res->error(format_error_response(msg, type))`.
    fn errMsg(self: *ResGenerator, msg: []const u8, ty: mod.common.ErrorType) !void {
        var e = try mod.common.formatErrorResponse(self.alloc(), msg, ty);
        try self.err(&e);
    }

    /// `server_res_spipe::should_stop`.
    fn shouldStop(self: *const ResGenerator) bool {
        return self.spipe.shouldStop();
    }
};

/// `req.should_stop`, as `ResponseReader.next` asks it.
const ReqStop = struct {
    req: *const mod.http.Req,
    pub fn shouldStop(self: ReqStop) bool {
        return self.req.should_stop.call();
    }
};

// -----------------------------------------------------------------------------
// server_routes

/// Ports `server_routes` (server-context.h:99 @b1b770d90): every handler,
/// as a field so router mode can replace it, and what they share.
pub const Routes = struct {
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    get_health: mod.http.Handler = undefined,
    get_metrics: mod.http.Handler = undefined,
    get_slots: mod.http.Handler = undefined,
    post_slots: mod.http.Handler = undefined,
    get_props: mod.http.Handler = undefined,
    post_props: mod.http.Handler = undefined,
    post_infill: mod.http.Handler = undefined,
    post_completions: mod.http.Handler = undefined,
    post_completions_oai: mod.http.Handler = undefined,
    post_chat_completions: mod.http.Handler = undefined,
    post_chat_completions_tok: mod.http.Handler = undefined,
    post_control: mod.http.Handler = undefined,
    post_responses_oai: mod.http.Handler = undefined,
    post_responses_tok_oai: mod.http.Handler = undefined,
    post_transcriptions_oai: mod.http.Handler = undefined,
    ws_realtime: mod.http.WsHandler = undefined,
    post_anthropic_messages: mod.http.Handler = undefined,
    post_anthropic_count_tokens: mod.http.Handler = undefined,
    post_apply_template: mod.http.Handler = undefined,
    get_models: mod.http.Handler = undefined,
    post_tokenize: mod.http.Handler = undefined,
    post_detokenize: mod.http.Handler = undefined,
    post_embeddings: mod.http.Handler = undefined,
    post_embeddings_oai: mod.http.Handler = undefined,
    post_rerank: mod.http.Handler = undefined,
    post_tts: mod.http.Handler = undefined,
    post_speech_oai: mod.http.Handler = undefined,
    post_systemone: mod.http.Handler = undefined,
    get_lora_adapters: mod.http.Handler = undefined,
    post_lora_adapters: mod.http.Handler = undefined,

    // using unique_ptr to allow late initialization of const
    meta: ?mod.context.Meta = null,
    meta_arena: std.heap.ArenaAllocator = std.heap.ArenaAllocator.init(std.heap.c_allocator),

    params: *const tools_common.params.Params,
    ctx_server: *mod.context.Impl,

    queue_tasks: *mod.context.TaskQueue,
    queue_results: *mod.context.ResultQueue,

    // cached responses, to be used during sleep
    mutex_cache: stdlib.sync.Mutex = .{},
    cached_models: json.Value = .null,
    cached_props: json.Value = .null,
    cached_metrics: mod.common.Metrics = .{},
    cache_arena: std.heap.ArenaAllocator = std.heap.ArenaAllocator.init(std.heap.c_allocator),
    // set when a scrape during sleep already reported the throughput buckets
    should_reset_buckets: bool = false,

    /// Ports the constructor of `server_routes` (server-context.cpp:5622
    /// @b1b770d90). `self` must not move afterwards: the handlers and the
    /// sleeping callback point at it.
    pub fn init(self: *Routes, params: *const tools_common.params.Params, ctx_server: *mod.context.ServerContext) !void {
        self.* = .{
            .params = params,
            .ctx_server = ctx_server.impl,
            .queue_tasks = &ctx_server.impl.queue_tasks,
            .queue_results = &ctx_server.impl.queue_results,
        };
        self.initRoutes();

        // note: this must be registered before load_model()
        //       so that on sleep phase, the callback is called before ctx is destroyed
        try self.queue_tasks.onSleepingState(self, onSleeping);
    }

    pub fn deinit(self: *Routes) void {
        self.meta_arena.deinit();
        self.cache_arena.deinit();
    }

    fn onSleeping(ctx: *anyopaque, is_sleeping: bool) void {
        const self: *Routes = @ptrCast(@alignCast(ctx));
        self.updateCachedResponses(is_sleeping) catch |err| switch (err) {
            error.OutOfMemory, error.InvalidArgument, error.RuntimeError, error.JsonError => {},
        };
    }

    /// Ports `server_routes`'s `update_meta` (server-context.h:105
    /// @b1b770d90): not thread-safe, only while `is_ready` is false.
    pub fn updateMeta(self: *Routes, ctx_server: *const mod.context.ServerContext) !void {
        _ = self.meta_arena.reset(.free_all);
        self.meta = try ctx_server.getMeta(self.meta_arena.allocator());
    }

    fn metaRef(self: *const Routes) *const mod.context.Meta {
        return &self.meta.?;
    }

    /// Ports `server_routes`'s `create_response` (server-context.cpp:5618 @b1b770d90).
    fn createResponse(self: *Routes, bypass_sleep: bool) !*ResGenerator {
        return ResGenerator.create(self.queue_tasks, self.queue_results, self.params.sleep_idle_seconds, bypass_sleep);
    }

    fn handler(self: *Routes, comptime f: fn (*Routes, *const mod.http.Req) anyerror!*mod.http.Res) mod.http.Handler {
        return .{ .ctx = self, .func = struct {
            fn call(ctx: ?*anyopaque, req: *const mod.http.Req) anyerror!*mod.http.Res {
                return f(@ptrCast(@alignCast(ctx.?)), req);
            }
        }.call };
    }

    /// Ports `server_routes`'s `init_routes` (server-context.cpp:5774
    /// @b1b770d90). IMPORTANT: all handlers must start with
    /// `createResponse`, so that the sleeping case is handled correctly.
    fn initRoutes(self: *Routes) void {
        self.get_health = self.handler(getHealth);
        self.get_metrics = self.handler(getMetrics);
        self.get_slots = self.handler(getSlots);
        self.post_slots = self.handler(postSlots);
        self.get_props = self.handler(getProps);
        self.post_props = self.handler(postProps);
        self.post_infill = self.handler(postInfill);
        self.post_completions = self.handler(postCompletions);
        self.post_completions_oai = self.handler(postCompletionsOai);
        self.post_chat_completions = self.handler(postChatCompletions);
        self.post_chat_completions_tok = self.handler(postChatCompletionsTok);
        self.post_control = self.handler(postControl);
        self.post_responses_oai = self.handler(postResponsesOai);
        self.post_responses_tok_oai = self.handler(postResponsesTokOai);
        self.post_transcriptions_oai = self.handler(postTranscriptionsOai);
        self.ws_realtime = .{ .ctx = self, .func = wsRealtimeHandler };
        self.post_anthropic_messages = self.handler(postAnthropicMessages);
        self.post_anthropic_count_tokens = self.handler(postAnthropicCountTokens);
        self.post_apply_template = self.handler(postApplyTemplate);
        self.get_models = self.handler(getModels);
        self.post_tokenize = self.handler(postTokenize);
        self.post_detokenize = self.handler(postDetokenize);
        self.post_embeddings = self.handler(postEmbeddings);
        self.post_embeddings_oai = self.handler(postEmbeddingsOai);
        self.post_rerank = self.handler(postRerank);
        self.post_tts = self.handler(postTts);
        self.post_speech_oai = self.handler(postSpeechOai);
        self.post_systemone = self.handler(postSystemone);
        self.get_lora_adapters = self.handler(getLoraAdapters);
        self.post_lora_adapters = self.handler(postLoraAdapters);
    }

    // -------------------------------------------------------------------------
    // the handlers

    /// The `get_health` lambda (server-context.cpp:5778 @b1b770d90): error
    /// and loading states are handled by the middleware; reachable during
    /// sleep.
    fn getHealth(self: *Routes, _: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(true);
        errdefer res.destroy();
        const a = res.alloc();
        var o = mod.common.Obj.init(a);
        try o.put("status", "ok");
        const v = o.done();
        try res.ok(&v);
        return &res.base;
    }

    /// The cached-metrics lambda of `get_metrics` (server-context.cpp:5791 @b1b770d90).
    fn useCachedMetrics(self: *Routes, res: *ResGenerator) !void {
        self.mutex_cache.lock();
        defer self.mutex_cache.unlock();
        const a = res.alloc();
        try res.base.setHeader("Process-Start-Time-Unix", try std.fmt.allocPrint(a, "{d}", .{self.cached_metrics.t_start}));
        const tmp = try mod.result.Result.create(std.heap.c_allocator, .metrics);
        defer tmp.destroy(std.heap.c_allocator);
        tmp.kind.metrics.metrics = try self.cached_metrics.clone(tmp.alloc());
        res.base.content_type = "text/plain; version=0.0.4";
        res.base.status = 200;
        try res.base.setData(try tmp.toMetrics(a));
        // the gauges are averaged over the window between two scrapes
        self.cached_metrics.resetBucket();
        self.should_reset_buckets = true;
    }

    /// The `get_metrics` lambda (server-context.cpp:5791 @b1b770d90).
    fn getMetrics(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(true);
        errdefer res.destroy();
        if (!self.params.endpoint_metrics) {
            try res.errMsg("This server does not support metrics endpoint. Start it with `--metrics`", .not_supported);
            return &res.base;
        }

        if (self.queue_tasks.isSleeping()) {
            try self.useCachedMetrics(res);
        } else {
            // request slots data using task queue
            {
                var task = mod.task.Task.initType(.metrics);
                task.id = res.rd.getNewId();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                try res.rd.postTask(task, true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            const Stop = struct {
                req: *const mod.http.Req,
                q: *mod.context.TaskQueue,
                pub fn shouldStop(s: @This()) bool {
                    return s.req.should_stop.call() or s.q.isSleeping();
                }
            };
            const result = (try res.rd.next(Stop{ .req = req, .q = self.queue_tasks })) orelse {
                if (!req.should_stop.call()) {
                    try self.useCachedMetrics(res);
                }
                return &res.base;
            };
            defer result.destroy(std.heap.c_allocator);

            if (result.isError()) {
                var j = try result.toJson(res.alloc());
                try res.err(&j);
                return &res.base;
            }

            ggmlAssert(result.kind == .metrics, 5842, "res_task != nullptr");

            const a = res.alloc();
            try res.base.setHeader("Process-Start-Time-Unix", try std.fmt.allocPrint(a, "{d}", .{result.kind.metrics.metrics.t_start}));
            res.base.content_type = "text/plain; version=0.0.4";
            res.base.status = 200;
            try res.base.setData(try result.toMetrics(a));
        }

        return &res.base;
    }

    /// The `get_slots` lambda (server-context.cpp:5853 @b1b770d90).
    fn getSlots(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        if (!self.params.endpoint_slots) {
            try res.errMsg("This server does not support slots endpoint. Start it with `--slots`", .not_supported);
            return &res.base;
        }

        // request slots data using task queue
        {
            var task = mod.task.Task.initType(.slot_get);
            task.id = res.rd.getNewId();
            try res.rd.postTask(task, true); // high-priority task
        }

        // get the result
        const result = (try res.rd.next(ReqStop{ .req = req })) orelse {
            // connection was closed
            ggmlAssert(req.should_stop.call(), 5871, "req.should_stop()");
            return &res.base;
        };
        defer result.destroy(std.heap.c_allocator);

        if (result.isError()) {
            var j = try result.toJson(res.alloc());
            try res.err(&j);
            return &res.base;
        }

        ggmlAssert(result.kind == .slots, 5881, "res_task != nullptr");

        // optionally return "fail_on_no_slot" error
        if (req.getParam("fail_on_no_slot", "").len != 0) {
            if (result.kind.slots.n_idle_slots == 0) {
                try res.errMsg("no slot available", .unavailable);
                return &res.base;
            }
        }

        var j = try result.toJson(res.alloc());
        try res.ok(&j);
        return &res.base;
    }

    /// The `post_slots` lambda (server-context.cpp:5895 @b1b770d90).
    fn postSlots(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        var res_alive = true;
        errdefer if (res_alive) res.destroy();
        if (self.params.slot_save_path.len == 0) {
            try res.errMsg("This server does not support slots action. Start it with `--slot-save-path`", .not_supported);
            return &res.base;
        }

        const id_slot_str = try res.alloc().dupeSentinel(u8, req.getParam("id_slot", ""), 0);

        const id_slot = tools_common.arg.stoi(id_slot_str) catch {
            try res.errMsg("Invalid slot ID", .invalid_request);
            return &res.base;
        };

        const action = req.getParam("action", "");

        if (std.mem.eql(u8, action, "save")) {
            res.destroy();
            res_alive = false;
            return self.handleSlotsSave(req, id_slot);
        }
        if (std.mem.eql(u8, action, "restore")) {
            res.destroy();
            res_alive = false;
            return self.handleSlotsRestore(req, id_slot);
        }
        if (std.mem.eql(u8, action, "erase")) {
            res.destroy();
            res_alive = false;
            return self.handleSlotsErase(req, id_slot);
        }

        try res.errMsg("Invalid action", .invalid_request);
        return &res.base;
    }

    /// The `get_props` lambda (server-context.cpp:5928 @b1b770d90): must be
    /// reachable during sleep, so it never touches the context.
    fn getProps(self: *Routes, _: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(true);
        errdefer res.destroy();
        if (self.queue_tasks.isSleeping()) {
            self.mutex_cache.lock();
            defer self.mutex_cache.unlock();
            try res.ok(&self.cached_props);
        } else {
            var p = try getResProps(res.alloc(), self.metaRef(), self.params, false);
            try res.ok(&p);
        }
        return &res.base;
    }

    /// The `post_props` lambda (server-context.cpp:5940 @b1b770d90).
    fn postProps(self: *Routes, _: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        if (!self.params.endpoint_props) {
            try res.errMsg("This server does not support changing global properties. Start it with `--props`", .not_supported);
            return &res.base;
        }
        // update any props here

        var o = mod.common.Obj.init(res.alloc());
        try o.put("success", true);
        const v = o.done();
        try res.ok(&v);
        return &res.base;
    }

    /// The `post_infill` lambda (server-context.cpp:5952 @b1b770d90). Its
    /// three "required" checks set an error on a response that is then
    /// dropped, as upstream's do; a missing field throws at `at` instead.
    fn postInfill(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        var keep = false;
        defer if (!keep) res.destroy();
        const a = res.alloc();
        const vocab = self.ctx_server.vocab;
        // check model compatibility
        var err: std.ArrayList(u8) = .empty;
        if (llama.api.vocab.llama_vocab_fim_pre(vocab) == mod.common.token_null) {
            try err.appendSlice(a, "prefix token is missing. ");
        }
        if (llama.api.vocab.llama_vocab_fim_suf(vocab) == mod.common.token_null) {
            try err.appendSlice(a, "suffix token is missing. ");
        }
        if (llama.api.vocab.llama_vocab_fim_mid(vocab) == mod.common.token_null) {
            try err.appendSlice(a, "middle token is missing. ");
        }
        if (err.items.len != 0) {
            try res.errMsg(try std.mem.concat(a, u8, &.{ "Infill is not supported by this model: ", err.items }), .not_supported);
            keep = true;
            return &res.base;
        }

        // validate input
        var data = try json.parse(std.heap.c_allocator, req.body);
        defer data.deinit(std.heap.c_allocator);
        if (data.contains("prompt") and !(try data.at("prompt")).isString()) {
            // prompt is optional
            try res.errMsg("\"prompt\" must be a string", .invalid_request);
        }

        if (!data.contains("input_prefix")) {
            try res.errMsg("\"input_prefix\" is required", .invalid_request);
        }

        if (!data.contains("input_suffix")) {
            try res.errMsg("\"input_suffix\" is required", .invalid_request);
        }

        if (data.contains("input_extra") and !(try data.at("input_extra")).isArray()) {
            // input_extra is optional
            try res.errMsg("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", .invalid_request);
            keep = true;
            return &res.base;
        }

        var input_extra = try mod.common.jsonValueJson(std.heap.c_allocator, &data, "input_extra", json.Value.emptyArray());
        for (input_extra.array.items) |*chunk| {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") or !(try chunk.at("text")).isString()) {
                input_extra.deinit(std.heap.c_allocator);
                try res.errMsg("extra_context chunk must contain a \"text\" field with a string value", .invalid_request);
                keep = true;
                return &res.base;
            }
            // filename is optional
            if (chunk.contains("filename") and !(try chunk.at("filename")).isString()) {
                input_extra.deinit(std.heap.c_allocator);
                try res.errMsg("extra_context chunk's \"filename\" field must be a string", .invalid_request);
                keep = true;
                return &res.base;
            }
        }
        try data.set(std.heap.c_allocator, "input_extra", input_extra); // default to empty array if it's not exist

        const prompt = mod.common.jsonValue(&data, "prompt", []const u8, "");
        var pv = try json.Value.initString(a, prompt);
        var tokenized_prompts = try mod.tokens.tokenizeInputPrompts(std.heap.c_allocator, vocab, self.ctx_server.mctx, &pv, false, true, self.ctx_server.init_opt);
        defer {
            for (tokenized_prompts.items) |*t| t.deinit();
            tokenized_prompts.deinit(std.heap.c_allocator);
        }
        mod.common.srv(.dbg, "operator()", "creating infill tasks, n_prompts = %d\n", .{@as(c_int, @intCast(tokenized_prompts.items.len))});
        const infill = try mod.tokens.formatPromptInfill(
            a,
            vocab,
            try data.at("input_prefix"),
            try data.at("input_suffix"),
            try data.at("input_extra"),
            self.params.n_batch,
            self.params.n_predict,
            self.metaRef().slot_n_ctx,
            self.params.spm_infill,
            tokenized_prompts.items[0].getTokens(), // TODO: this could maybe be multimodal.
        );
        try data.set(std.heap.c_allocator, "prompt", try mod.common.from(std.heap.c_allocator, infill));

        return self.handleCompletionsImpl(req, .infill, &data, &.{}, .none, null); // infill is not OAI compatible
    }

    /// The `post_completions` lambda (server-context.cpp:6030 @b1b770d90).
    fn postCompletions(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        defer res.destroy();
        var body = try json.parse(std.heap.c_allocator, req.body);
        defer body.deinit(std.heap.c_allocator);
        return self.handleCompletionsImpl(req, .completion, &body, &.{}, .none, null);
    }

    /// The `post_completions_oai` lambda (server-context.cpp:6042 @b1b770d90).
    fn postCompletionsOai(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        defer res.destroy();
        var body = try json.parse(std.heap.c_allocator, req.body);
        defer body.deinit(std.heap.c_allocator);
        return self.handleCompletionsImpl(req, .completion, &body, &.{}, .oai_cmpl, null);
    }

    /// `oaicompat_chat_params_parse` into a body the caller frees, with the
    /// media files.
    const Parsed = struct {
        arena: std.heap.ArenaAllocator,
        body: json.Value,
        files: std.ArrayList([]u8),
        session: common_chat.session.Session,
    };

    fn chatParse(self: *Routes, body_text: []const u8) !*Parsed {
        return self.chatParseConverted(body_text, .none, null);
    }

    /// The conversion a route runs on its body before the chat one.
    const Conversion = enum { none, responses, anthropic, transcriptions };

    /// `chatParse` after one of `server-chat.cpp`'s conversions, with the
    /// two `SRV_DBG` lines each route logs after it (the second dumps the
    /// body, which throws on invalid UTF-8 -- only when it prints).
    fn chatParseConverted(self: *Routes, body_text: []const u8, conv: Conversion, req: ?*const mod.http.Req) !*Parsed {
        const p = try std.heap.c_allocator.create(Parsed);
        errdefer std.heap.c_allocator.destroy(p);
        p.* = .{ .arena = std.heap.ArenaAllocator.init(std.heap.c_allocator), .body = .null, .files = .empty, .session = try common_chat.session.Session.initEmpty(std.heap.c_allocator) };
        errdefer p.arena.deinit();
        errdefer p.session.deinit();
        const a = p.arena.allocator();
        var body = try json.parse(a, body_text);
        switch (conv) {
            .none => {},
            .responses => {
                body = try mod.convert.responsesToChatcmpl(a, &body);
                try logConverted(a, "Request converted: OpenAI Responses -> OpenAI Chat Completions", &body);
            },
            .anthropic => {
                body = try mod.convert.anthropicToOai(a, &body);
                try logConverted(a, "Request converted: Anthropic -> OpenAI Chat Completions", &body);
            },
            .transcriptions => {
                const file: ?[]const u8 = if (req.?.files.get("file")) |f| f.data else null;
                body = try mod.convert.transcriptionsToChatcmpl(a, &body, self.metaRef().chat_params.tmpls, file, &p.files);
                try logConverted(a, "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions", &body);
            },
        }
        p.body = try mod.oaicompat.oaicompatChatParamsParse(a, self.ctx_server.vocab, &body, self.metaRef().chat_params, &p.files, &p.session);
        return p;
    }

    /// `SRV_DBG("%s\n", what); SRV_DBG("converted request: %s\n", body.dump().c_str());`
    fn logConverted(a: std.mem.Allocator, comptime what: []const u8, body: *const json.Value) !void {
        mod.common.srv(.dbg, "operator()", "%s\n", .{@as([*:0]const u8, what ++ "")});
        if (tools_common.log.level_debug <= tools_common.log.verbosity_thold) {
            const d = try json.dump(a, body, -1);
            const z = mod.common.dupZ(d);
            defer mod.common.freeZ(z);
            mod.common.srv(.dbg, "operator()", "converted request: %s\n", .{z});
        }
    }

    fn freeParsed(p: *Parsed) void {
        p.session.deinit();
        p.arena.deinit();
        std.heap.c_allocator.destroy(p);
    }

    /// The `post_chat_completions` lambda (server-context.cpp:6054 @b1b770d90).
    fn postChatCompletions(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        defer res.destroy();
        const p = try self.chatParse(req.body);
        defer freeParsed(p);
        return self.handleCompletionsImpl(req, .completion, &p.body, p.files.items, .oai_chat, &p.session);
    }

    /// The `post_chat_completions_tok` lambda (server-context.cpp:6074 @b1b770d90).
    fn postChatCompletionsTok(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        return self.handleCountTokens(req, .oai_chat);
    }

    /// The `post_control` lambda (server-context.cpp:6078 @b1b770d90).
    fn postControl(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        var body = try json.parse(a, req.body);

        const cmpl_id = mod.common.jsonValue(&body, "id", []const u8, "");
        const action = mod.common.jsonValue(&body, "action", []const u8, "");
        if (cmpl_id.len == 0) {
            try res.errMsg("missing completion id", .invalid_request);
            return &res.base;
        }
        if (!std.mem.eql(u8, action, "reasoning_end")) {
            try res.errMsg("unknown control action", .invalid_request);
            return &res.base;
        }

        {
            var task = mod.task.Task.initType(.control);
            task.id = res.rd.getNewId();
            const ta = try task.arenaAlloc(std.heap.c_allocator);
            task.params.control_cmpl_id = try ta.dupe(u8, cmpl_id);
            task.params.control_action = .reasoning_end;
            try res.rd.postTask(task, false);
        }

        const result = (try res.rd.next(ReqStop{ .req = req })) orelse {
            ggmlAssert(req.should_stop.call(), 6104, "req.should_stop()");
            return &res.base;
        };
        defer result.destroy(std.heap.c_allocator);
        var j = try result.toJson(a);
        if (result.isError()) {
            try res.err(&j);
            return &res.base;
        }
        try res.ok(&j);
        return &res.base;
    }

    /// The `post_responses_oai` lambda (server-context.cpp:6115 @b1b770d90):
    /// the request converted from the Responses API, then run as chat.
    fn postResponsesOai(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        defer res.destroy();
        const p = try self.chatParseConverted(req.body, .responses, null);
        defer freeParsed(p);
        return self.handleCompletionsImpl(req, .completion, &p.body, p.files.items, .oai_resp, &p.session);
    }

    /// The `post_responses_tok_oai` lambda (server-context.cpp:6137 @b1b770d90).
    fn postResponsesTokOai(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        return self.handleCountTokens(req, .oai_resp);
    }

    /// The `post_transcriptions_oai` lambda (server-context.cpp:6141
    /// @b1b770d90): refused without audio input; otherwise the upload and
    /// its form converted into chat over the template's ASR preset.
    fn postTranscriptionsOai(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);

        if (!self.metaRef().has_mtmd or !self.metaRef().chat_params.allow_audio) {
            errdefer res.destroy();
            try res.errMsg("The current model does not support audio input.", .not_supported);
            return &res.base;
        }
        defer res.destroy();

        const p = try self.chatParseConverted(req.body, .transcriptions, req);
        defer freeParsed(p);
        return self.handleCompletionsImpl(req, .completion, &p.body, p.files.items, .oai_asr, &p.session);
    }

    /// The `post_anthropic_messages` lambda (server-context.cpp:6412
    /// @b1b770d90): the request converted from the Anthropic Messages API,
    /// then run as chat.
    fn postAnthropicMessages(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        defer res.destroy();
        const p = try self.chatParseConverted(req.body, .anthropic, null);
        defer freeParsed(p);
        return self.handleCompletionsImpl(req, .completion, &p.body, p.files.items, .anthropic, &p.session);
    }

    /// The `post_anthropic_count_tokens` lambda (server-context.cpp:6434 @b1b770d90).
    fn postAnthropicCountTokens(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        return self.handleCountTokens(req, .anthropic);
    }

    /// The `post_apply_template` lambda (server-context.cpp:6439
    /// @b1b770d90): `handle_chat_completions` without the inference.
    fn postApplyTemplate(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const p = try self.chatParse(req.body);
        defer freeParsed(p);
        const a = res.alloc();
        var o = mod.common.Obj.init(a);
        try o.put("prompt", try (try p.body.at("prompt")).clone(a));
        const v = o.done();
        try res.ok(&v);
        return &res.base;
    }

    /// The `get_models` lambda (server-context.cpp:6454 @b1b770d90).
    fn getModels(self: *Routes, _: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(true);
        errdefer res.destroy();
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (self.queue_tasks.isSleeping()) {
            self.mutex_cache.lock();
            defer self.mutex_cache.unlock();
            try res.ok(&self.cached_models);
        } else {
            var m = try getResModels(res.alloc(), self.metaRef());
            try res.ok(&m);
        }
        return &res.base;
    }

    /// The `post_tokenize` lambda (server-context.cpp:6466 @b1b770d90).
    fn postTokenize(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        var body = try json.parse(a, req.body);
        var tokens_response = json.Value.emptyArray();
        if (body.count("content") != 0) {
            const add_special = mod.common.jsonValue(&body, "add_special", bool, false);
            const parse_special = mod.common.jsonValue(&body, "parse_special", bool, true);
            const with_pieces = mod.common.jsonValue(&body, "with_pieces", bool, false);

            const toks = try mod.tokens.tokenizeMixed(a, self.ctx_server.vocab, try body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (toks) |token| {
                    const piece = try tools_common.common.tokenToPiece(a, self.ctx_server.vocab, token, true);
                    var piece_json: json.Value = undefined;

                    // Check if the piece is valid UTF-8
                    if (mod.common.isValidUtf8(piece)) {
                        piece_json = try json.Value.initString(a, piece);
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json.Value.emptyArray();
                        for (piece) |ch| {
                            try piece_json.pushBack(a, .{ .number_integer = ch });
                        }
                    }

                    var o = mod.common.Obj.init(a);
                    try o.put("id", token);
                    try o.put("piece", piece_json);
                    try tokens_response.pushBack(a, o.done());
                }
            } else {
                tokens_response = try mod.common.from(a, toks);
            }
        }

        var o = mod.common.Obj.init(a);
        try o.put("tokens", tokens_response);
        const v = o.done();
        try res.ok(&v);
        return &res.base;
    }

    /// The `post_detokenize` lambda (server-context.cpp:6507 @b1b770d90).
    fn postDetokenize(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        var body = try json.parse(a, req.body);

        var content: []const u8 = "";
        if (body.count("tokens") != 0) {
            const toks = try (try body.at("tokens")).getSlice(a, llama.impl.llama_c.llama_token);
            content = try mod.common.tokensToStr(a, self.ctx_server.vocab, toks);
        }

        var o = mod.common.Obj.init(a);
        try o.put("content", content);
        const v = o.done();
        try res.ok(&v);
        return &res.base;
    }

    /// The `post_embeddings` lambda (server-context.cpp:6521 @b1b770d90).
    fn postEmbeddings(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        return self.handleEmbeddingsImpl(req, .none);
    }

    /// The `post_embeddings_oai` lambda (server-context.cpp:6525 @b1b770d90).
    fn postEmbeddingsOai(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        return self.handleEmbeddingsImpl(req, .oai_embd);
    }

    /// The `post_systemone` lambda (server-context.cpp:6544 @b1b770d90): the
    /// typed decision API. One task per variant of each question, or one
    /// task for all the questions of a joint model.
    fn postSystemone(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        const decision = &self.ctx_server.decision;
        if (decision.type == .none) {
            try res.errMsg("This model is not a decision model", .not_supported);
            return &res.base;
        }

        const body = try json.parse(a, req.body);
        const questions = try decision.parseQuestions(a, &body);

        var files: std.ArrayList([]u8) = .empty;
        const parsed_state = try decision.parseState(a, &body, &files);
        const state = try decision.prepareState(a, &parsed_state);
        if (files.items.len != 0 and (!decision.canUseImages() or !self.metaRef().has_inp_image)) {
            try res.errMsg("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", .not_supported);
            return &res.base;
        }
        const fs: []const []const u8 = @ptrCast(files.items);

        // one task per variant of each question, or one task for all the questions
        {
            var tasks: std.ArrayList(mod.task.Task) = .empty;
            defer {
                for (tasks.items) |*t| t.deinit(std.heap.c_allocator);
                tasks.deinit(std.heap.c_allocator);
            }
            if (decision.isJoint()) {
                var task = mod.task.Task.initType(.decision);
                errdefer task.deinit(std.heap.c_allocator);
                task.params = try mod.task.TaskParams.init(try task.arenaAlloc(std.heap.c_allocator));
                task.id = res.rd.getNewId();
                try decision.fillTaskJoint(a, std.heap.c_allocator, &state, questions, fs, self.ctx_server.mctx, self.ctx_server.init_opt, &task);
                try tasks.append(std.heap.c_allocator, task);
            } else {
                for (questions) |*question| {
                    for (0..decision.nVariants(question)) |variant| {
                        var task = mod.task.Task.initType(.decision);
                        errdefer task.deinit(std.heap.c_allocator);
                        task.params = try mod.task.TaskParams.init(try task.arenaAlloc(std.heap.c_allocator));
                        task.id = res.rd.getNewId();
                        try decision.fillTask(a, std.heap.c_allocator, &state, questions, question, variant, fs, self.ctx_server.mctx, self.ctx_server.init_opt, &task);
                        try tasks.append(std.heap.c_allocator, task);
                    }
                }
            }
            if (decision.canSharePrompt()) {
                var groups = try mod.decision.groupTasks(std.heap.c_allocator, tasks.items, @intCast(self.params.n_parallel));
                defer groups.deinit(std.heap.c_allocator);
                for (tasks.items) |*t| t.deinit(std.heap.c_allocator);
                tasks.clearRetainingCapacity();
                try tasks.appendSlice(std.heap.c_allocator, groups.items);
            }
            try res.rd.postTasks(tasks.items, null, false);
            tasks.clearRetainingCapacity();
        }

        var all_results = try res.rd.waitForAll(ReqStop{ .req = req });
        defer all_results.deinit(std.heap.c_allocator);

        if (all_results.is_terminated) {
            return &res.base; // connection is closed
        } else if (all_results.err) |e| {
            var j = try e.toJson(a);
            try res.err(&j);
            return &res.base;
        }

        var answers = json.Value.emptyObject();
        var n_tokens: i32 = 0;
        var i_result: usize = 0;
        var i_score: usize = 0;
        for (questions) |*question| {
            var scores: std.ArrayList([]const f32) = .empty;
            if (decision.isJoint()) {
                // one result with the scores of all the questions, in order
                const result = all_results.results[0];
                ggmlAssert(result != null and result.?.kind == .decision and i_score + question.options.len <= result.?.kind.decision.scores.len, 6605, "result != nullptr && i_score + question.options.size() <= result->scores.size()");
                const sc = result.?.kind.decision.scores;
                try scores.append(a, sc[i_score .. i_score + question.options.len]);
                i_score += question.options.len;
                n_tokens = result.?.kind.decision.n_tokens;
            }
            var variant: usize = 0;
            while (!decision.isJoint() and variant < decision.nVariants(question)) : (variant += 1) {
                const result = all_results.results[i_result];
                i_result += 1;
                ggmlAssert(result != null and result.?.kind == .decision, 6612, "result != nullptr");
                try scores.append(a, result.?.kind.decision.scores);
                n_tokens += result.?.kind.decision.n_tokens;
            }
            try answers.set(a, question.id, try decision.formatAnswerMedia(a, question, scores.items, files.items.len != 0));
        }

        var o = mod.common.Obj.init(a);
        try o.put("model", self.metaRef().model_name);
        try o.put("answers", answers);
        var usage = mod.common.Obj.init(a);
        try usage.put("input_tokens", n_tokens);
        try usage.put("output_tokens", @as(i32, 0));
        try o.put("usage", usage.done());
        var root = o.done();
        try res.ok(&root);
        return &res.base;
    }

    /// The `post_rerank` lambda (server-context.cpp:6529 @b1b770d90).
    fn postRerank(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        return self.handleRerankImpl(req);
    }

    /// `parse_item` of `handle_rerank_impl`: one rerank input item, a plain
    /// string or `{"text": str|[str], "image": url|data-uri|raw-base64|[urls]}`.
    fn parseRerankItem(self: *Routes, a: std.mem.Allocator, item: *const json.Value, out_text: *std.ArrayList(u8), out_files: *std.ArrayList([]u8)) !void {
        if (item.isString()) {
            out_text.clearRetainingCapacity();
            try out_text.appendSlice(a, item.string);
            return;
        }
        if (!item.isObject()) {
            return common_chat.exception.throw(error.InvalidArgument, "rerank input must be a string or an object of the form {{\"text\": ..., \"image\": ...}}", .{});
        }
        if (item.contains("video") or item.contains("fps") or item.contains("max_frames")) {
            return common_chat.exception.throw(error.InvalidArgument, "video/fps/max_frames are not supported for rerank input", .{});
        }
        const text: json.Value = if (item.contains("text")) (try item.at("text")).* else .null;
        if (text.isString()) {
            try out_text.appendSlice(a, text.string);
        } else if (text.isArray()) {
            for (text.array.items) |*t| {
                if (!t.isString()) {
                    return common_chat.exception.throw(error.InvalidArgument, "\"text\" array elements must be strings", .{});
                }
                try out_text.appendSlice(a, t.string);
            }
        } else if (!text.isNull()) {
            return common_chat.exception.throw(error.InvalidArgument, "\"text\" must be a string or an array of strings", .{});
        }
        const image: json.Value = if (item.contains("image")) (try item.at("image")).* else .null;
        if (image.isString()) {
            // marker is appended by format_prompt_rerank, one per decoded file
            try mod.oaicompat.handleMedia(a, out_files, image.string, self.params.media_path, self.params.media_allow_private_urls);
        } else if (image.isArray()) {
            for (image.array.items) |*u| {
                if (!u.isString()) {
                    return common_chat.exception.throw(error.InvalidArgument, "\"image\" array elements must be strings", .{});
                }
                try mod.oaicompat.handleMedia(a, out_files, u.string, self.params.media_path, self.params.media_allow_private_urls);
            }
        } else if (!image.isNull()) {
            return common_chat.exception.throw(error.InvalidArgument, "\"image\" must be a string or an array of strings", .{});
        }
    }

    /// Ports `server_routes`'s `handle_rerank_impl` (server-context.cpp:6912 @b1b770d90).
    fn handleRerankImpl(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        if (!self.params.embedding or self.params.pooling_type != llama.impl.llama_c.LLAMA_POOLING_TYPE_RANK) {
            try res.errMsg("This server does not support reranking. Start it with `--reranking`", .not_supported);
            return &res.base;
        }

        var body = try json.parse(a, req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        const is_tei_format = body.contains("texts");

        // Parse query
        var query_str: std.ArrayList(u8) = .empty;
        var query_files: std.ArrayList([]u8) = .empty;
        if (body.count("query") == 1) {
            self.parseRerankItem(a, try body.at("query"), &query_str, &query_files) catch |e| switch (e) {
                // handle_media's std::runtime_error is rethrown as an invalid_argument
                error.InvalidArgument, error.RuntimeError => {
                    try res.errMsg(mod.context.whatOf(e), .invalid_request);
                    return &res.base;
                },
                else => return e,
            };
        } else {
            try res.errMsg("\"query\" must be provided", .invalid_request);
            return &res.base;
        }

        // Resolve Jina/TEI format ambiguity
        const documents_json = try mod.common.jsonValueJson(a, &body, "documents", try mod.common.jsonValueJson(a, &body, "texts", json.Value.emptyArray()));
        if (!documents_json.isArray() or documents_json.empty()) {
            try res.errMsg("\"documents\" must be a non-empty array of strings or objects", .invalid_request);
            return &res.base;
        }

        // Parse documents
        const n_documents = documents_json.array.items.len;
        const documents = try a.alloc([]const u8, n_documents);
        const document_files = try a.alloc([]const []const u8, n_documents);
        for (documents_json.array.items, 0..) |*item, i| {
            var text: std.ArrayList(u8) = .empty;
            var files: std.ArrayList([]u8) = .empty;
            self.parseRerankItem(a, item, &text, &files) catch |e| switch (e) {
                error.InvalidArgument, error.RuntimeError => {
                    try res.errMsg(mod.context.whatOf(e), .invalid_request);
                    return &res.base;
                },
                else => return e,
            };
            documents[i] = text.items;
            document_files[i] = files.items;
        }

        const top_n = mod.common.jsonValue(&body, "top_n", c_int, @intCast(documents.len));
        if (top_n < 0) {
            try res.errMsg("\"top_n\" must be >= 0", .invalid_request);
            return &res.base;
        }

        // create and queue the task
        var responses = json.Value.emptyArray();
        {
            var tasks: std.ArrayList(mod.task.Task) = .empty;
            defer tasks.deinit(std.heap.c_allocator);
            errdefer for (tasks.items) |*t| t.deinit(std.heap.c_allocator);
            try tasks.ensureTotalCapacity(std.heap.c_allocator, documents.len);
            for (documents, document_files) |doc, doc_files| {
                var tmp = try mod.tokens.formatPromptRerank(std.heap.c_allocator, self.ctx_server.model_tgt, self.ctx_server.vocab, self.ctx_server.mctx, query_str.items, doc, query_files.items, doc_files, self.ctx_server.init_opt);
                errdefer tmp.deinit();
                var task = mod.task.Task.initType(.rerank);
                task.params = try mod.task.TaskParams.init(try task.arenaAlloc(std.heap.c_allocator));
                task.id = res.rd.getNewId();
                task.tokens = tmp;
                tasks.appendAssumeCapacity(task);
            }
            try res.rd.postTasks(tasks.items, null, false);
            tasks.clearRetainingCapacity();
        }

        // wait for the results
        var all_results = try res.rd.waitForAll(ReqStop{ .req = req });
        defer all_results.deinit(std.heap.c_allocator);

        // collect results
        if (all_results.is_terminated) {
            return &res.base; // connection is closed
        } else if (all_results.err) |e| {
            var j = try e.toJson(a);
            try res.err(&j);
            return &res.base;
        } else {
            for (all_results.results) |r| {
                ggmlAssert(r.?.kind == .rerank, 7042, "dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr");
                try responses.pushBack(a, try r.?.toJson(a));
            }
        }

        // write JSON response
        var root = try mod.oaicompat.formatResponseRerank(a, &body, self.metaRef().model_name, &responses, is_tei_format, documents, top_n);

        try res.ok(&root);
        return &res.base;
    }

    /// Ports `parse_tts_body` (server-context.cpp:5756 @b1b770d90): the body of
    /// `/tts` and `/v1/audio/speech`; multipart form fields arrive as
    /// strings, the numeric and boolean params get their types back.
    ///
    /// Return: owned by `a`.
    fn parseTtsBody(a: std.mem.Allocator, req: *const mod.http.Req) !json.Value {
        var body = try json.parse(a, req.body);
        for ([_][]const u8{ "top_k", "n_predict", "max_tokens", "seed" }) |k| {
            if (body.contains(k) and (try body.at(k)).isString()) {
                try body.set(a, k, .{ .number_integer = try mod.convert.stoll((try body.at(k)).string) });
            }
        }
        for ([_][]const u8{ "top_p", "temperature", "repeat_penalty", "speed" }) |k| {
            if (body.contains(k) and (try body.at(k)).isString()) {
                try body.set(a, k, try json.Value.from(a, try mod.convert.stof((try body.at(k)).string)));
            }
        }
        if (body.contains("stream") and (try body.at("stream")).isString()) {
            try body.set(a, "stream", .{ .boolean = std.mem.eql(u8, (try body.at("stream")).string, "true") });
        }
        return body;
    }

    /// The `post_tts` lambda (server-context.cpp:6533 @b1b770d90).
    fn postTts(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();
        const body = try parseTtsBody(arena.allocator(), req);
        return self.handleTtsImpl(req, &body);
    }

    /// The `post_speech_oai` lambda (server-context.cpp:6537 @b1b770d90): an
    /// OpenAI speech request, converted to `/tts`'s.
    fn postSpeechOai(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer arena.deinit();
        const aa = arena.allocator();
        const parsed = try parseTtsBody(aa, req);
        const body = try mod.convert.speechToTts(aa, &parsed);
        try logConverted(aa, "Request converted: OpenAI Speech -> TTS", &body);
        return self.handleTtsImpl(req, &body);
    }

    /// Ports `handle_tts_impl` (server-context.cpp:7060 @b1b770d90), a member
    /// of `server_routes`: `/tts` and `/v1/audio/speech`. A model with predefined
    /// voices speaks only with one of them: the one asked for, `--tts-voice`,
    /// or the first. `pcm16` is the float32 output converted to 16-bit.
    fn handleTtsImpl(self: *Routes, req: *const mod.http.Req, body: *const json.Value) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        try res.spipe.setReq(req); // will also set spipe if needed

        if (!self.ctx_server.hasCapTts()) {
            try res.errMsg("this server does not support audio generation", .not_supported);
            return &res.base;
        }

        const info = mtmd.mtmd.mtmd_gen_audio_get_info(self.ctx_server.mctx.?);

        const prompt = mod.common.jsonValue(body, "input", []const u8, mod.common.jsonValue(body, "prompt", []const u8, ""));
        if (prompt.len == 0) {
            try res.errMsg("\"input\" must be a non-empty string", .invalid_request);
            return &res.base;
        }

        const response_format = mod.common.jsonValue(body, "response_format", []const u8, "wav");
        const pcm16 = std.mem.eql(u8, response_format, "pcm16");
        const is_pcm = std.mem.eql(u8, response_format, "pcm");

        var task = mod.task.Task.initType(.tts);
        var task_moved = false;
        defer if (!task_moved) task.deinit(std.heap.c_allocator);
        const ta = try task.arenaAlloc(std.heap.c_allocator);
        task.params = try mod.task.TaskParams.init(ta);
        task.tts_inp = mod.task.TtsInp.init();
        try mod.schema.evalTtsSchema(ta, self.params, &task, body);
        const do_stream = task.params.stream;
        // a model with a "tts" chat template (e.g. LFM3-Audio) speaks the prompt rendered from it
        const lang = mod.common.jsonValue(body, "lang", []const u8, self.params.tts_lang);
        // speaker_ref in the request overrides the default voice
        const has_speaker_ref = req.files.get("speaker_ref") != null or mod.common.jsonValue(body, "speaker_ref_b64", []const u8, "").len != 0;
        const voice = mod.common.jsonValue(body, "voice", []const u8, if (has_speaker_ref) "" else self.params.tts_voice);
        const tts_prompt = try common_chat.templates.getTtsPrompt(a, self.metaRef().chat_params.tmpls, prompt, lang, voice);
        // a model with predefined voices only speaks with one of them
        var voice_ref: ?*mtmd.mtmd.Bitmap = null;
        defer if (voice_ref) |bmp| mtmd.mtmd.mtmd_bitmap_free(bmp);
        if (voice.len != 0 and info.n_voices > 0) {
            voice_ref = mtmd.mtmd.mtmd_gen_audio_get_voice(self.ctx_server.mctx.?, (try a.dupeSentinel(u8, voice, 0)).ptr);
            if (voice_ref == null) {
                try res.errMsg(try std.mem.concat(a, u8, &.{ "unknown voice \"", voice, "\", see \"voices\" in GET /props" }), .invalid_request);
                return &res.base;
            }
        }
        task.tts_inp.prompt_str = try ta.dupeSentinel(u8, if (tts_prompt.len == 0) prompt else tts_prompt, 0);
        task.tts_inp.lang_str = try ta.dupeSentinel(u8, lang, 0);
        task.tts_inp.data.stream = do_stream;
        task.tts_inp.data.out_type = if (is_pcm or pcm16) mtmd.helper_gen.OutType.pcm else mtmd.helper_gen.OutType.wav;
        // note: -1 is clamped to 0 by the sampler, it will disable the penalty
        task.params.sampling.penalty_last_n = if (task.params.n_predict > 0) task.params.n_predict else 512;
        task.params.sampling.seed = task.tts_inp.data.seed; // same seed for the codec/vocoder RNG
        if (task.tts_inp.data.top_k > 0) {
            task.params.sampling.top_k = task.tts_inp.data.top_k;
        }
        if (task.tts_inp.data.top_p > 0) {
            task.params.sampling.top_p = task.tts_inp.data.top_p;
        }
        if (task.tts_inp.data.temp >= 0) {
            task.params.sampling.temp = task.tts_inp.data.temp;
        }

        // speaker reference: either an uploaded form file ("speaker_ref") or a base64 JSON field ("speaker_ref_b64")
        var speaker_ref: []const u8 = "";
        if (req.files.get("speaker_ref")) |file| {
            speaker_ref = file.data;
        } else {
            const speaker_ref_b64 = mod.common.jsonValue(body, "speaker_ref_b64", []const u8, "");
            if (speaker_ref_b64.len != 0) {
                speaker_ref = mod.tokens.base64HppDecode(a, speaker_ref_b64) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => {
                        try res.errMsg("\"speaker_ref_b64\" is not valid base64", .invalid_request);
                        return &res.base;
                    },
                };
            }
        }

        if (speaker_ref.len > 0) {
            if (voice_ref != null) {
                try res.errMsg("\"voice\" and \"speaker_ref\" cannot be used together", .invalid_request);
                return &res.base;
            }
            if (!mtmd.mtmd.mtmd_support_audio(self.ctx_server.mctx.?)) {
                try res.errMsg("this model has no voice encoder, \"speaker_ref\" cannot be used", .invalid_request);
                return &res.base;
            }
            const wrapper = mtmd.helper.mtmd_helper_bitmap_init_from_buf(self.ctx_server.mctx.?, speaker_ref.ptr, speaker_ref.len, false, self.ctx_server.init_opt);
            if (wrapper.bitmap == null) {
                try res.errMsg("failed to decode \"speaker_ref\"", .invalid_request);
                return &res.base;
            }
            task.tts_inp.setSpeakerRef(wrapper.bitmap);
        } else if (voice_ref) |bmp| {
            task.tts_inp.setSpeakerRef(bmp);
            voice_ref = null;
        } else if (info.n_voices > 0) {
            // without a voice prompt the model produces near silence: the first voice is the default
            const first = mtmd.mtmd.mtmd_gen_audio_voice_name(self.ctx_server.mctx.?, 0);
            mod.common.srv(.wrn, "handle_tts_impl", "no voice given, speaking with \"%s\"\n", .{first});
            task.tts_inp.setSpeakerRef(mtmd.mtmd.mtmd_gen_audio_get_voice(self.ctx_server.mctx.?, first));
        } else {
            mod.common.srv(.wrn, "handle_tts_impl", "%s", .{@as([*:0]const u8, "no speaker reference provided, the model may behave randomly\n")});
        }

        task.id = res.rd.getNewId();
        task_moved = true;
        try res.rd.postTask(task, false);

        // raw LE mono samples; audio/L16 would imply 16-bit big-endian (RFC 2586)
        const pcm_type = try std.fmt.allocPrint(a, "audio/pcm;rate={d}", .{info.sample_rate});
        const content_type: []const u8 = if (pcm16)
            try std.mem.concat(a, u8, &.{ pcm_type, ";encoding=signed-int;bits=16" })
        else if (is_pcm)
            try std.mem.concat(a, u8, &.{ pcm_type, ";encoding=float;bits=32" })
        else
            "audio/wav";
        res.tts_pcm16 = pcm16;

        if (!do_stream) {
            const result = (try res.rd.next(ReqStop{ .req = req })) orelse {
                ggmlAssert(req.should_stop.call(), 7185, "req.should_stop()");
                return &res.base; // connection is closed
            };
            defer result.destroy(std.heap.c_allocator);
            if (result.isError()) {
                var j = try result.toJson(a);
                try res.err(&j);
                return &res.base;
            }
            ggmlAssert(result.kind == .tts, 7193, "tts_res != nullptr");
            res.base.status = 200;
            res.base.content_type = content_type;
            try res.base.setData(try takeAudio(a, pcm16, result.kind.tts.audio));
            return &res.base;
        }

        const first_result = (try res.rd.next(ReqStop{ .req = req })) orelse {
            ggmlAssert(req.should_stop.call(), 7201, "req.should_stop()");
            return &res.base; // connection is closed
        };
        defer first_result.destroy(std.heap.c_allocator);
        if (first_result.isError()) {
            var j = try first_result.toJson(a);
            try res.err(&j);
            return &res.base;
        }
        ggmlAssert(first_result.kind == .tts, 7209, "first_tts_res != nullptr");

        res.base.status = 200;
        res.base.content_type = content_type;
        try res.base.setData(try takeAudio(a, pcm16, first_result.kind.tts.audio));
        res.tts_is_done = first_result.kind.tts.final;
        res.spipe.setNext(&res.base, .{ .ctx = res, .func = ttsNext });

        return &res.base;
    }

    /// Ports `take_audio` (server-context.cpp:7177 @b1b770d90), a lambda of
    /// handle_tts_impl: the pipeline outputs float32 samples, converted to
    /// 16-bit for `pcm16`.
    ///
    /// Return: `audio` itself, or the converted bytes owned by `a`.
    fn takeAudio(a: std.mem.Allocator, pcm16: bool, audio: []const u8) ![]const u8 {
        if (!pcm16) return audio;
        const n = audio.len / @sizeOf(f32);
        const samples = try a.alloc(f32, n);
        defer a.free(samples);
        @memcpy(std.mem.sliceAsBytes(samples), audio[0 .. n * @sizeOf(f32)]);
        return mod.common.pcmF32ToS16(a, samples);
    }

    /// The `set_next` lambda (server-context.cpp:7216 @b1b770d90) of `handle_tts_impl`.
    fn ttsNext(ctx: ?*anyopaque, output: *std.ArrayList(u8), oa: std.mem.Allocator) anyerror!bool {
        const res_this: *ResGenerator = @ptrCast(@alignCast(ctx.?));
        // flush buffered audio before is_done - the first result can already be final
        if (res_this.base.data.items.len != 0) {
            try output.appendSlice(oa, res_this.base.data.items);
            res_this.base.data.clearRetainingCapacity();
            return true;
        }
        if (res_this.tts_is_done) {
            return false;
        }
        if (res_this.shouldStop()) {
            return false;
        }

        const rd = &res_this.rd;
        if (!rd.hasNext()) {
            return false;
        }
        const Stop = struct {
            res_this: *ResGenerator,
            pub fn shouldStop(s: @This()) bool {
                return s.res_this.shouldStop();
            }
        };
        const result = (try rd.next(Stop{ .res_this = res_this })) orelse return false;
        defer result.destroy(std.heap.c_allocator);
        if (result.isError()) {
            return false;
        }
        ggmlAssert(result.kind == .tts, 7239, "tts_res != nullptr");
        const audio = try takeAudio(std.heap.c_allocator, res_this.tts_pcm16, result.kind.tts.audio);
        defer if (res_this.tts_pcm16) std.heap.c_allocator.free(audio);
        try output.appendSlice(oa, audio);
        res_this.tts_is_done = result.kind.tts.final;
        return true;
    }

    /// The `ws_realtime` handler: runs on its own HTTP thread for the whole
    /// lifetime of the connection.
    fn wsRealtimeHandler(ctx: ?*anyopaque, req: *const mod.http.Req, conn: *mod.http.WsConn) void {
        const self: *Routes = @ptrCast(@alignCast(ctx.?));
        self.wsRealtime(req, conn) catch |e| {
            const z = mod.common.dupZ(mod.context.whatOf(e));
            defer mod.common.freeZ(z);
            mod.common.srv(.err, "operator()", "realtime session failed: %s\n", .{z});
            conn.close();
        };
    }

    /// The `ws_realtime` lambda (server-context.cpp:6176 @b1b770d90):
    /// realtime transcription over websocket, modeled after the OpenAI
    /// realtime API. Audio is PCM16 mono, at the model sample rate unless a
    /// session.update gives another one (resampled), sent as
    /// base64 in input_audio_buffer.append events;
    /// input_audio_buffer.commit with "final": true ends the input, closing
    /// the socket also ends it.
    fn wsRealtime(self: *Routes, req: *const mod.http.Req, conn: *mod.http.WsConn) !void {
        // models that answer after the audio hold a conversation, see server-realtime.h
        // live transcription: Voxtral Realtime (no chat), or an "asr" model in a session opened with ?intent=transcription
        const server = self.ctx_server;
        const live_asr = server.asr_prompt.len != 0 and std.mem.eql(u8, req.getParam("intent", ""), "transcription");
        if (server.mtmd_cap.audio_stream and server.mtmd_cap.chat and !live_asr) {
            mod.realtime.conversation(.{
                .queue_tasks = self.queue_tasks,
                .queue_results = self.queue_results,
                .meta = self.metaRef(),
                .params = self.params,
                .mctx = server.mctx,
                .cap = &server.mtmd_cap,
                .vocab = server.vocab,
            }, conn);
            return;
        }

        var session: LiveAsr = .{ .routes = self, .conn = conn, .arena = std.heap.ArenaAllocator.init(std.heap.c_allocator) };
        defer session.arena.deinit();
        try session.run();
    }

    /// The `get_lora_adapters` lambda (server-context.cpp:6630 @b1b770d90).
    fn getLoraAdapters(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();

        {
            var task = mod.task.Task.initType(.get_lora);
            task.id = res.rd.getNewId();
            try res.rd.postTask(task, false);
        }

        // get the result
        return self.finishSimple(res, req, .get_lora, 6653);
    }

    /// The `post_lora_adapters` lambda (server-context.cpp:6658 @b1b770d90).
    fn postLoraAdapters(self: *Routes, req: *const mod.http.Req) anyerror!*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        var body = try json.parse(a, req.body);
        if (!body.isArray()) {
            try res.errMsg("Request body must be an array", .invalid_request);
            return &res.base;
        }

        {
            var task = mod.task.Task.initType(.set_lora);
            task.id = res.rd.getNewId();
            task.set_lora = try mod.common.parseLoraRequest(try task.arenaAlloc(std.heap.c_allocator), &body);
            try res.rd.postTask(task, false);
        }

        // get the result
        return self.finishSimple(res, req, .apply_lora, 6687);
    }

    /// The common tail of the single-task handlers: wait for the one
    /// result, answer its error or its JSON.
    fn finishSimple(self: *Routes, res: *ResGenerator, req: *const mod.http.Req, comptime kind: std.meta.Tag(mod.result.Result.Kind), comptime line: c_int) !*mod.http.Res {
        _ = self;
        const result = (try res.rd.next(ReqStop{ .req = req })) orelse {
            // connection was closed
            ggmlAssert(req.should_stop.call(), line - 9, "req.should_stop()");
            return &res.base;
        };
        defer result.destroy(std.heap.c_allocator);

        if (result.isError()) {
            var j = try result.toJson(res.alloc());
            try res.err(&j);
            return &res.base;
        }

        ggmlAssert(result.kind == kind, line, "dynamic_cast<...>(result.get()) != nullptr");
        var j = try result.toJson(res.alloc());
        try res.ok(&j);
        return &res.base;
    }

    /// A slot action's file name, validated, and its path under
    /// `--slot-save-path`; null after answering "Invalid filename".
    fn slotFile(self: *Routes, res: *ResGenerator, req: *const mod.http.Req) !?[2][]const u8 {
        const a = res.alloc();
        var request_data = try json.parse(a, req.body);
        const filename = try (try request_data.at("filename")).get([]const u8);
        if (!mod.oaicompat.fsValidateFilename(filename, false)) {
            try res.errMsg("Invalid filename", .invalid_request);
            return null;
        }
        const filepath = try std.mem.concat(a, u8, &.{ self.params.slot_save_path, filename });
        return .{ filename, filepath };
    }

    fn postSlotAction(self: *Routes, req: *const mod.http.Req, ty: mod.task.TaskType, id_slot: c_int, comptime kind: std.meta.Tag(mod.result.Result.Kind), comptime line: c_int) !*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        var filename: []const u8 = "";
        var filepath: []const u8 = "";
        if (ty != .slot_erase) {
            const f = (try self.slotFile(res, req)) orelse return &res.base;
            filename = f[0];
            filepath = f[1];
        }

        {
            var task = mod.task.Task.initType(ty);
            task.id = res.rd.getNewId();
            const ta = try task.arenaAlloc(std.heap.c_allocator);
            task.slot_action.id_slot = id_slot;
            task.slot_action.filename = try ta.dupe(u8, filename);
            task.slot_action.filepath = try ta.dupe(u8, filepath);
            try res.rd.postTask(task, false);
        }

        return self.finishSimple(res, req, kind, line);
    }

    /// Ports `server_routes`'s `handle_slots_save` (server-context.cpp:6693
    /// @b1b770d90). Its result is not type-checked, as upstream's is not.
    fn handleSlotsSave(self: *Routes, req: *const mod.http.Req, id_slot: c_int) !*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const f = (try self.slotFile(res, req)) orelse return &res.base;

        {
            var task = mod.task.Task.initType(.slot_save);
            task.id = res.rd.getNewId();
            const ta = try task.arenaAlloc(std.heap.c_allocator);
            task.slot_action.id_slot = id_slot;
            task.slot_action.filename = try ta.dupe(u8, f[0]);
            task.slot_action.filepath = try ta.dupe(u8, f[1]);
            try res.rd.postTask(task, false);
        }

        const result = (try res.rd.next(ReqStop{ .req = req })) orelse {
            // connection was closed
            ggmlAssert(req.should_stop.call(), 6716, "req.should_stop()");
            return &res.base;
        };
        defer result.destroy(std.heap.c_allocator);

        var j = try result.toJson(res.alloc());
        if (result.isError()) {
            try res.err(&j);
            return &res.base;
        }

        try res.ok(&j);
        return &res.base;
    }

    /// Ports `server_routes`'s `handle_slots_restore` (server-context.cpp:6729 @b1b770d90).
    fn handleSlotsRestore(self: *Routes, req: *const mod.http.Req, id_slot: c_int) !*mod.http.Res {
        return self.postSlotAction(req, .slot_restore, id_slot, .slot_save_load, 6761);
    }

    /// Ports `server_routes`'s `handle_slots_erase` (server-context.cpp:6766 @b1b770d90).
    fn handleSlotsErase(self: *Routes, req: *const mod.http.Req, id_slot: c_int) !*mod.http.Res {
        return self.postSlotAction(req, .slot_erase, id_slot, .slot_erase, 6788);
    }

    /// Ports `server_routes`'s `handle_embeddings_impl` (server-context.cpp:6793 @b1b770d90).
    fn handleEmbeddingsImpl(self: *Routes, req: *const mod.http.Req, res_type_in: mod.task.ResponseType) !*mod.http.Res {
        var res_type = res_type_in;
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        if (!self.params.embedding) {
            try res.errMsg("This server does not support embeddings. Start it with `--embeddings`", .not_supported);
            return &res.base;
        }

        if (res_type != .none and self.metaRef().pooling_type == llama.impl.llama_c.LLAMA_POOLING_TYPE_NONE) {
            try res.errMsg("Pooling type 'none' is not OAI compatible. Please use a different pooling type", .invalid_request);
            return &res.base;
        }

        var body = try json.parse(a, req.body);

        // for the shape of input/content, see tokenize_input_prompts()
        var prompt: *const json.Value = undefined;
        if (body.count("input") != 0) {
            prompt = try body.at("input");
        } else if (body.contains("content")) {
            res_type = .none; // "content" field is not OAI compatible
            prompt = try body.at("content");
        } else {
            try res.errMsg("\"input\" or \"content\" must be provided", .invalid_request);
            return &res.base;
        }

        var use_base64 = false;
        if (body.count("encoding_format") != 0) {
            const format = try (try body.at("encoding_format")).get([]const u8);
            if (std.mem.eql(u8, format, "base64")) {
                use_base64 = true;
            } else if (!std.mem.eql(u8, format, "float")) {
                try res.errMsg("The format to return the embeddings in. Can be either float or base64", .invalid_request);
                return &res.base;
            }
        }

        // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
        const TokenizeEntry = struct {
            fn run(r: *Routes, p: *const json.Value) !mod.tokens.ServerTokens {
                if (p.isObject() and p.contains("content")) {
                    return mod.oaicompat.tokenizeOaiContentArray(std.heap.c_allocator, r.ctx_server.vocab, r.ctx_server.mctx, r.metaRef().chat_params, try p.at("content"), true, true, r.ctx_server.init_opt);
                }
                return mod.tokens.tokenizeInputSubprompt(std.heap.c_allocator, r.ctx_server.vocab, r.ctx_server.mctx, p, true, true, r.ctx_server.init_opt);
            }
        };

        var tokenized_prompts: std.ArrayList(mod.tokens.ServerTokens) = .empty;
        defer {
            for (tokenized_prompts.items) |*t| t.deinit();
            tokenized_prompts.deinit(std.heap.c_allocator);
        }
        if (prompt.isArray() and !mod.common.jsonIsArrayAndContainsNumbers(prompt)) {
            for (prompt.array.items) |*p| {
                var t = try TokenizeEntry.run(self, p);
                tokenized_prompts.append(std.heap.c_allocator, t) catch |e| {
                    t.deinit();
                    return e;
                };
            }
        } else {
            var t = try TokenizeEntry.run(self, prompt);
            tokenized_prompts.append(std.heap.c_allocator, t) catch |e| {
                t.deinit();
                return e;
            };
        }
        if (tokenized_prompts.items.len == 0) {
            try res.errMsg("\"input\" must not be empty", .invalid_request);
            return &res.base;
        }

        for (tokenized_prompts.items) |*t| {
            // this check is necessary for models that do not add BOS token to the input
            if (t.empty()) {
                try res.errMsg("Input content cannot be empty", .invalid_request);
                return &res.base;
            }
        }

        var embd_normalize = self.params.embd_normalize;
        if (body.count("embd_normalize") != 0) {
            embd_normalize = try (try body.at("embd_normalize")).get(c_int);
            if (self.metaRef().pooling_type == llama.impl.llama_c.LLAMA_POOLING_TYPE_NONE) {
                mod.common.srv(.dbg, "handle_embeddings_impl", "embd_normalize is not supported by pooling type %d, ignoring it\n", .{@as(c_int, self.metaRef().pooling_type)});
            }
        }

        // create and queue the task
        var responses = json.Value.emptyArray();
        {
            var tasks: std.ArrayList(mod.task.Task) = .empty;
            defer tasks.deinit(std.heap.c_allocator);
            errdefer for (tasks.items) |*t| t.deinit(std.heap.c_allocator);
            for (tokenized_prompts.items) |*t| {
                var task = mod.task.Task.initType(.embedding);
                task.params = try mod.task.TaskParams.init(try task.arenaAlloc(std.heap.c_allocator));

                task.id = res.rd.getNewId();
                task.tokens = t.*;
                t.* = mod.tokens.ServerTokens.init(std.heap.c_allocator); // std::move

                // OAI-compat
                task.params.res_type = res_type;
                task.params.embd_normalize = embd_normalize;

                try tasks.append(std.heap.c_allocator, task);
            }
            try res.rd.postTasks(tasks.items, null, false);
            tasks.clearRetainingCapacity();
        }

        // wait for the results
        var all_results = try res.rd.waitForAll(ReqStop{ .req = req });
        defer all_results.deinit(std.heap.c_allocator);

        // collect results
        if (all_results.is_terminated) {
            return &res.base; // connection is closed
        } else if (all_results.err) |e| {
            var j = try e.toJson(a);
            try res.err(&j);
            return &res.base;
        } else {
            for (all_results.results) |r| {
                ggmlAssert(r.?.kind == .embd, 6899, "dynamic_cast<server_task_result_embd*>(res.get()) != nullptr");
                try responses.pushBack(a, try r.?.toJson(a));
            }
        }

        // write JSON response
        var root = if (res_type == .oai_embd)
            try mod.oaicompat.formatEmbeddingsResponseOaicompat(a, &body, self.metaRef().model_name, &responses, use_base64)
        else
            responses;
        try res.ok(&root);
        return &res.base;
    }

    /// Ports `server_routes`'s `handle_count_tokens` (server-context.cpp:7249
    /// @b1b770d90): the prompt a chat, Responses or Anthropic request would
    /// render, counted in tokens.
    fn handleCountTokens(self: *Routes, req: *const mod.http.Req, res_type: mod.task.ResponseType) !*mod.http.Res {
        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        var body = try json.parse(a, req.body);
        var is_oai = false;

        switch (res_type) {
            .oai_chat => {
                is_oai = true;
            },
            .oai_resp => {
                is_oai = true;
                body = try mod.convert.responsesToChatcmpl(a, &body);
            },
            .anthropic => {
                body = try mod.convert.anthropicToOai(a, &body);
            },
            else => {
                try res.errMsg("invalid res_type", .invalid_request);
                return &res.base;
            },
        }

        var files: std.ArrayList([]u8) = .empty;
        var session = try common_chat.session.Session.initEmpty(std.heap.c_allocator); // dummy, unused
        defer session.deinit();
        var body_parsed = try mod.oaicompat.oaicompatChatParamsParse(a, self.ctx_server.vocab, &body, self.metaRef().chat_params, &files, &session);
        const prompt = try body_parsed.at("prompt");

        // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
        var n_tokens: usize = undefined;
        if (self.ctx_server.mctx != null) {
            if (!prompt.isString()) {
                return common_chat.exception.throw(error.RuntimeError, "for mtmd, input prompt must be a string.", .{});
            }
            var t = try mod.tokens.processMtmdPrompt(std.heap.c_allocator, self.ctx_server.mctx, prompt.string, files.items, self.ctx_server.init_opt, true);
            defer t.deinit();
            n_tokens = t.size();
        } else {
            n_tokens = (try mod.tokens.tokenizeMixed(a, self.ctx_server.vocab, prompt, true, true)).len;
        }

        var o = mod.common.Obj.init(a);
        try o.put("input_tokens", @as(i64, @intCast(n_tokens)));
        if (is_oai) {
            try o.put("object", "response.input_tokens");
        }
        const v = o.done();
        try res.ok(&v);
        return &res.base;
    }

    /// Ports `server_routes`'s `handle_completions_impl` (server-context.cpp:5339 @b1b770d90).
    /// `chat_session`: the request's, or null for the default one.
    fn handleCompletionsImpl(self: *Routes, req: *const mod.http.Req, ty: mod.task.TaskType, data: *json.Value, files: []const []u8, res_type: mod.task.ResponseType, chat_session: ?*const common_chat.session.Session) !*mod.http.Res {
        ggmlAssert(ty == .completion or ty == .infill, 5346, "type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL");

        const res = try self.createResponse(false);
        errdefer res.destroy();
        const a = res.alloc();
        const completion_id = try mod.common.genChatcmplid(a);
        const params = self.params;

        try res.spipe.setReq(req); // will also set spipe if needed

        if (!self.ctx_server.hasCapChat()) {
            try res.errMsg("this server does not support chat/completions", .not_supported);
            return &res.base;
        }

        var sse_ping_interval = params.sse_ping_interval;

        self.postCompletionTasks(res, ty, data, files, res_type, completion_id, &sse_ping_interval, chat_session) catch |e| {
            if (e == error.OutOfMemory) return e;
            try res.errMsg(mod.context.whatOf(e), .invalid_request);
            return &res.base;
        };

        const do_stream = mod.common.jsonValue(data, "stream", bool, false);

        if (!do_stream) {
            // non-stream, wait for the results
            var all_results = try res.rd.waitForAll(ReqStop{ .req = req });
            defer all_results.deinit(std.heap.c_allocator);
            if (all_results.is_terminated) {
                return &res.base; // connection is closed
            } else if (all_results.err) |e| {
                var j = try e.toJson(a);
                try res.err(&j);
                return &res.base;
            } else {
                var arr = json.Value.emptyArray();
                for (all_results.results) |r| {
                    ggmlAssert(r.?.kind == .cmpl_final, 5444, "dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr");
                    try arr.pushBack(a, try r.?.toJson(a));
                }
                ggmlAssert(arr.array.items.len != 0, 5447, "!arr.empty() && \"empty results\"");
                if (arr.array.items.len == 1) {
                    // if single request, return single object instead of array
                    try res.ok(&arr.array.items[0]);
                } else if (res_type == .oai_chat or res_type == .oai_cmpl) {
                    // if multiple results in OAI format, we need to re-format them
                    const usage = try arr.array.items[0].index(a, "usage");
                    const choices = try arr.array.items[0].index(a, "choices");
                    for (arr.array.items[1..]) |*item| {
                        const ch = try (try item.index(a, "choices")).indexAt(a, 0);
                        try choices.pushBack(a, ch.*);
                        ch.* = .null;
                    }

                    // OAI counts generated tokens across all choices
                    var n_completion_tokens: i64 = 0;
                    for (arr.array.items) |*item| {
                        n_completion_tokens += try (try (try item.index(a, "usage")).at("completion_tokens")).get(i64);
                    }
                    try usage.set(a, "completion_tokens", .{ .number_integer = n_completion_tokens });
                    try usage.set(a, "total_tokens", .{ .number_integer = (try (try usage.at("prompt_tokens")).get(i64)) + n_completion_tokens });

                    try res.ok(&arr.array.items[0]);
                } else {
                    // multi-results, non-OAI compat
                    try res.ok(&arr);
                }
            }
        } else {
            // in streaming mode, the first error must be treated as non-stream response
            // this is to match the OAI API behavior
            // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
            const first_result = (try res.rd.next(ReqStop{ .req = req })) orelse {
                ggmlAssert(req.should_stop.call(), 5479, "req.should_stop()");
                return &res.base; // connection is closed
            };
            defer first_result.destroy(std.heap.c_allocator);

            if (first_result.isError()) {
                var j = try first_result.toJson(a);
                try res.err(&j);
                return &res.base;
            }

            ggmlAssert(first_result.kind == .cmpl_partial or first_result.kind == .cmpl_final, 5488, "dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr || dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr");

            // next responses are streamed
            // to be sent immediately
            var first_result_json = try first_result.toJson(a);
            if (first_result_json == .null) {
                try res.base.setData(""); // simply send HTTP headers and status code
            } else if (res_type == .anthropic) {
                try res.base.setData(try mod.common.formatAnthropicSse(a, &first_result_json));
            } else if (res_type == .oai_resp) {
                try res.base.setData(try mod.common.formatOaiRespSse(a, &first_result_json));
            } else {
                try res.base.setData(try mod.common.formatOaiSse(a, &first_result_json));
            }
            res.base.status = 200;
            res.base.content_type = "text/event-stream";
            res.res_type = res_type;
            res.sse_ping_interval = sse_ping_interval;
            res.spipe.setNext(&res.base, .{ .ctx = res, .func = streamNext });
        }

        return &res.base;
    }

    /// The `try` block of `handle_completions_impl` (server-context.cpp:5339
    /// @b1b770d90): tokenize, build one task per prompt (and its children),
    /// post them.
    fn postCompletionTasks(self: *Routes, res: *ResGenerator, ty: mod.task.TaskType, data: *json.Value, files: []const []u8, res_type: mod.task.ResponseType, completion_id: []const u8, sse_ping_interval: *i32, chat_session: ?*const common_chat.session.Session) !void {
        const params = self.params;
        const ctx_server = self.ctx_server;
        var tasks: std.ArrayList(mod.task.Task) = .empty;
        defer tasks.deinit(std.heap.c_allocator);
        errdefer for (tasks.items) |*t| t.deinit(std.heap.c_allocator);

        const prompt = try data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format

        if (params.path_prompts_log_dir.len != 0) {
            try logPrompt(res.alloc(), params.path_prompts_log_dir, prompt);
        }

        // process prompt
        var inputs: std.ArrayList(mod.tokens.ServerTokens) = .empty;
        defer {
            for (inputs.items) |*t| t.deinit();
            inputs.deinit(std.heap.c_allocator);
        }

        if (res_type != .none and ctx_server.mctx != null) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            try inputs.append(std.heap.c_allocator, try mod.tokens.processMtmdPrompt(std.heap.c_allocator, ctx_server.mctx, try prompt.get([]const u8), files, ctx_server.init_opt, false));
        } else {
            // Everything else, including multimodal completions.
            inputs = try mod.tokens.tokenizeInputPrompts(std.heap.c_allocator, ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        for (inputs.items) |*input| {
            var task = mod.task.Task.initType(ty);
            errdefer task.deinit(std.heap.c_allocator);

            task.id = res.rd.getNewId();

            task.tokens = input.*;
            input.* = mod.tokens.ServerTokens.init(std.heap.c_allocator); // std::move
            const ta = try task.arenaAlloc(std.heap.c_allocator);
            task.params = try mod.schema.evalLlamaCmplSchema(ta, ctx_server.vocab, params, self.metaRef().logit_bias_eog, data);

            if (chat_session) |cs| try task.applyChatSession(std.heap.c_allocator, cs);

            task.id_slot = mod.common.jsonValue(data, "id_slot", c_int, -1);
            sse_ping_interval.* = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type = res_type;
            task.params.oaicompat_cmpl_id = try ta.dupe(u8, completion_id);
            task.params.oaicompat_model = try ta.dupe(u8, self.metaRef().model_name);

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                const n_children = task.params.n_cmpl - 1;
                var j: i32 = 0;
                while (j < n_children) : (j += 1) {
                    try task.addChild(std.heap.c_allocator, task.id, res.rd.getNewId());
                }
            }

            try tasks.append(std.heap.c_allocator, task);
        }

        try res.rd.postTasks(tasks.items, chat_session, false);
        tasks.clearRetainingCapacity();
    }

    /// The `path_prompts_log_dir` block of `handle_completions_impl`
    /// (server-context.cpp:5339 @b1b770d90): the prompt, written to a file
    /// named for the time.
    fn logPrompt(a: std.mem.Allocator, dir: []const u8, prompt: *const json.Value) !void {
        const name = try std.fmt.allocPrint(a, "{d:0>12}.txt", .{llama.impl.llama_c.ggml_time_ms()});
        const file_path = try std.fs.path.joinZ(a, &.{ dir, name });
        const text = if (prompt.isString()) prompt.string else try json.dump(a, prompt, 2);
        const io = stdlib.file.defaultIo();
        if (stdlib.file.Out.create(io, file_path)) |out| {
            var f = out;
            defer f.close(io);
            // `f << s.c_str()`: up to the first NUL
            f.writeText(std.mem.sliceTo(text, 0)) catch |err| switch (err) {
                else => {}, // the stream's failure unchecked
            };
        } else |_| {
            mod.common.srv(.err, "handle_completions_impl", "failed to create %s\n", .{file_path.ptr});
        }
    }

    /// The `set_next` lambda of `handle_completions_impl`
    /// (server-context.cpp:5339 @b1b770d90): one result per call as SSE, a
    /// ping when the stream stays silent, `[DONE]` at the end where the
    /// format has one.
    fn streamNext(ctx: ?*anyopaque, output: *std.ArrayList(u8), oa: std.mem.Allocator) anyerror!bool {
        const res_this: *ResGenerator = @ptrCast(@alignCast(ctx.?));
        const res_type = res_this.res_type;
        const sse_ping_interval = res_this.sse_ping_interval;

        return streamNextImpl(res_this, output, oa, res_type, sse_ping_interval) catch |e| {
            if (e == error.OutOfMemory) return e;
            var tmp = std.heap.ArenaAllocator.init(std.heap.c_allocator);
            defer tmp.deinit();
            var error_json = try mod.common.formatErrorResponse(tmp.allocator(), mod.context.whatOf(e), .server);
            try output.appendSlice(oa, try formatError(tmp.allocator(), res_type, &error_json));

            // terminate on exception
            return false;
        };
    }

    /// The `format_error` lambda of the streaming lambda (server-context.cpp:5508 @b1b770d90).
    fn formatError(a: std.mem.Allocator, res_type: mod.task.ResponseType, res_json: *const json.Value) ![]u8 {
        if (res_type == .anthropic) {
            var o = mod.common.Obj.init(a);
            try o.put("event", "error");
            try o.put("data", try res_json.clone(a));
            var v = o.done();
            return mod.common.formatAnthropicSse(a, &v);
        } else {
            var o = mod.common.Obj.init(a);
            try o.put("error", try res_json.clone(a));
            var v = o.done();
            return mod.common.formatOaiSse(a, &v);
        }
    }

    fn streamNextImpl(res_this: *ResGenerator, output: *std.ArrayList(u8), oa: std.mem.Allocator, res_type: mod.task.ResponseType, sse_ping_interval: i32) !bool {
        const func = "operator()";
        if (res_this.shouldStop()) {
            mod.common.srv(.dbg, func, "%s", .{@as([*:0]const u8, "stopping streaming due to should_stop condition\n")});
            return false; // should_stop condition met
        }

        if (res_this.base.data.items.len != 0) {
            // flush the first chunk
            try output.appendSlice(oa, res_this.base.data.items);
            res_this.base.data.clearRetainingCapacity();
            return true;
        }

        const rd = &res_this.rd;

        // check if there is more data
        if (!rd.hasNext()) {
            switch (res_type) {
                .none, .oai_resp, .anthropic => {},
                else => try output.appendSlice(oa, "data: [DONE]\n\n"),
            }
            mod.common.srv(.dbg, func, "%s", .{@as([*:0]const u8, "all results received, terminating stream\n")});
            return false; // no more data, terminate
        }

        // receive subsequent results
        const Stop = struct {
            res_this: *ResGenerator,
            timeout: *bool,
            start_time: i64,
            sse_ping_interval: i32,
            pub fn shouldStop(s: @This()) bool {
                if (s.res_this.shouldStop()) {
                    return true; // should_stop condition met
                } else if (s.sse_ping_interval > 0 and llama.impl.llama_c.ggml_time_ms() - s.start_time > @as(i64, s.sse_ping_interval) * 1000) {
                    s.timeout.* = true;
                    return true; // timeout
                }
                return false;
            }
        };
        var timeout = false;
        const result_opt = try rd.next(Stop{ .res_this = res_this, .timeout = &timeout, .start_time = llama.impl.llama_c.ggml_time_ms(), .sse_ping_interval = sse_ping_interval });

        if (timeout) {
            if (result_opt) |r| r.destroy(std.heap.c_allocator);
            // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
            mod.common.srv(.dbg, func, "%s", .{@as([*:0]const u8, "sending SSE ping\n")});
            try output.appendSlice(oa, ":\n\n");
            return true;
        }

        const result = result_opt orelse {
            mod.common.srv(.dbg, func, "%s", .{@as([*:0]const u8, "stopping streaming due to should_stop condition\n")});
            ggmlAssert(res_this.shouldStop(), 5577, "effective_should_stop()");
            return false; // should_stop condition met
        };
        defer result.destroy(std.heap.c_allocator);

        var tmp = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer tmp.deinit();
        const a = tmp.allocator();

        // send the results
        if (result.isError()) {
            var res_json = try result.toJson(a);
            try output.appendSlice(oa, try formatError(a, res_type, &res_json));
            mod.common.srv(.dbg, func, "%s", .{@as([*:0]const u8, "error received during streaming, terminating stream\n")});
            return false; // terminate on error
        } else {
            ggmlAssert(result.kind == .cmpl_partial or result.kind == .cmpl_final, 5588, "dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr");
            var res_json = try result.toJson(a);
            if (res_type == .anthropic) {
                try output.appendSlice(oa, try mod.common.formatAnthropicSse(a, &res_json));
            } else if (res_type == .oai_resp) {
                try output.appendSlice(oa, try mod.common.formatOaiRespSse(a, &res_json));
            } else {
                try output.appendSlice(oa, try mod.common.formatOaiSse(a, &res_json));
            }
        }

        // has next data, continue
        return true;
    }

    /// Ports `server_routes`'s `get_model_info` (server-context.cpp:5751
    /// @b1b770d90): for router mode.
    pub fn getModelInfo(self: *const Routes, a: std.mem.Allocator) !json.Value {
        return getResModelInfo(a, self.metaRef());
    }

    /// Ports `server_routes`'s `update_cached_responses` (server-context.cpp:7303
    /// @b1b770d90): the caller is the task queue, so the context can be
    /// read without holding locks.
    fn updateCachedResponses(self: *Routes, is_sleeping: bool) !void {
        self.mutex_cache.lock();
        defer self.mutex_cache.unlock();

        if (is_sleeping) {
            _ = self.cache_arena.reset(.free_all);
            const a = self.cache_arena.allocator();
            self.cached_models = try getResModels(a, self.metaRef());
            self.cached_props = try getResProps(a, self.metaRef(), self.params, true);
            self.cached_metrics = try self.ctx_server.getMetrics(a);

            self.should_reset_buckets = false;

            mod.common.srv(.dbg, "update_cached_responses", "%s\n", .{@as([*:0]const u8, "cached responses updated")});
        } else if (self.should_reset_buckets) {
            // a scrape during sleep already reported these buckets
            self.ctx_server.resetMetricsBucket();

            self.should_reset_buckets = false;
        }
    }
};

/// `json_value(body, key, std::vector<std::string>(default))`: the member's
/// strings, or `default_value` when it is absent or null, or -- with a
/// warning -- when it does not convert.
///
/// Return: owned by `a`.
fn jsonValueStringsOr(a: std.mem.Allocator, body: *const json.Value, key: []const u8, default_value: []const []const u8) ![][]const u8 {
    if (body.contains(key) and !(try body.at(key)).isNull()) {
        const v = try body.at(key);
        return v.getSlice(a, []const u8) catch |e| switch (e) {
            error.JsonError => {
                const k = try a.dupeSentinel(u8, key, 0);
                const w = try a.dupeSentinel(u8, json.what(), 0);
                tools_common.log.wrn("Wrong type supplied for parameter '%s', using default value: %s\n", .{ k.ptr, w.ptr });
                return a.dupe([]const u8, default_value);
            },
            else => |x| x,
        };
    }
    return a.dupe([]const u8, default_value);
}

/// Ports `get_res_model_info` (server-context.cpp:5636 @b1b770d90): do NOT
/// use the context here, so it works during sleep.
fn getResModelInfo(a: std.mem.Allocator, meta: *const mod.context.Meta) !json.Value {
    var o = mod.common.Obj.init(a);
    try o.put("id", meta.model_name);
    try o.put("aliases", meta.model_aliases);
    try o.put("tags", meta.model_tags);
    try o.put("object", "model");
    try o.put("architecture", try mod.common.serverModelArchitectureJson(
        a,
        meta.has_inp_image,
        meta.has_inp_audio,
        meta.has_inp_video,
        meta.model_output_modalities,
    ));
    try o.put("created", stdlib.clock.timeSeconds());
    try o.put("owned_by", "llamacpp");
    var m = mod.common.Obj.init(a);
    // an enum is its underlying type in a `common_json_value`
    // (json.h:86 @b1b770d90)
    try m.put("vocab_type", @as(c_uint, @intCast(meta.model_vocab_type)));
    try m.put("n_vocab", meta.model_vocab_n_tokens);
    try m.put("n_ctx", meta.slot_n_ctx);
    try m.put("n_ctx_train", meta.model_n_ctx_train);
    try m.put("n_embd", meta.model_n_embd_inp);
    try m.put("n_params", meta.model_n_params);
    try m.put("size", meta.model_size);
    try m.put("ftype", meta.model_ftype);
    try o.put("meta", m.done());
    return o.done();
}

/// Ports `get_res_models` (server-context.cpp:5664 @b1b770d90).
fn getResModels(a: std.mem.Allocator, meta: *const mod.context.Meta) !json.Value {
    var model = mod.common.Obj.init(a);
    try model.put("name", meta.model_name);
    try model.put("model", meta.model_name);
    try model.put("modified_at", "");
    try model.put("size", "");
    try model.put("digest", ""); // dummy value, llama.cpp does not support managing model file's hash
    try model.put("type", "model");
    try model.put("description", "");
    try model.put("tags", &[_][]const u8{""});
    if (meta.has_mtmd) {
        try model.put("capabilities", &[_][]const u8{ "completion", "multimodal" });
    } else {
        try model.put("capabilities", &[_][]const u8{"completion"});
    }
    try model.put("parameters", "");
    var details = mod.common.Obj.init(a);
    try details.put("parent_model", "");
    try details.put("format", "gguf");
    try details.put("family", "");
    try details.put("families", &[_][]const u8{""});
    try details.put("parameter_size", "");
    try details.put("quantization_level", "");
    try model.put("details", details.done());

    var models = json.Value.emptyArray();
    try models.pushBack(a, model.done());
    var data = json.Value.emptyArray();
    try data.pushBack(a, try getResModelInfo(a, meta));

    var o = mod.common.Obj.init(a);
    try o.put("models", models);
    try o.put("object", "list");
    try o.put("data", data);
    return o.done();
}

/// Ports `get_res_props` (server-context.cpp:5697 @b1b770d90): do NOT use
/// the context here, so it works during sleep.
fn getResProps(a: std.mem.Allocator, meta: *const mod.context.Meta, params: *const tools_common.params.Params, is_sleeping: bool) !json.Value {
    var tparams = try mod.task.TaskParams.init(a);
    tparams.sampling = params.sampling;
    var dgs = mod.common.Obj.init(a);
    try dgs.put("params", try tparams.toJson(a, true));
    try dgs.put("n_ctx", meta.slot_n_ctx);

    const tmpl_default = meta.chat_template;
    const tmpl_tools = meta.chat_template_tool_use;

    var props = mod.common.Obj.init(a);
    try props.put("default_generation_settings", dgs.done());
    try props.put("total_slots", params.n_parallel);
    try props.put("model_alias", meta.model_name);
    try props.put("model_ftype", meta.model_ftype);
    try props.put("model_path", meta.model_path);
    var modalities = mod.common.Obj.init(a);
    try modalities.put("vision", meta.has_inp_image);
    try modalities.put("video", meta.has_inp_video);
    try modalities.put("audio", meta.has_inp_audio);
    try props.put("modalities", modalities.done());
    try props.put("media_marker", mod.common.getMediaMarker());
    try props.put("endpoint_slots", params.endpoint_slots);
    try props.put("endpoint_props", params.endpoint_props);
    try props.put("endpoint_metrics", params.endpoint_metrics);
    try props.put("ui", params.ui);
    try props.put("ui_settings", try meta.json_ui_settings.clone(a));
    try props.put("chat_template", tmpl_default);
    try props.put("chat_template_caps", try meta.chat_template_caps.clone(a));
    try props.put("bos_token", meta.bos_token_str);
    try props.put("eos_token", meta.eos_token_str);
    try props.put("build_info", meta.build_info);
    try props.put("is_sleeping", is_sleeping);
    try props.put("cors_proxy_enabled", params.ui_mcp_proxy);

    if (params.server_connect_code.len != 0) {
        try props.put("connect_code", params.server_connect_code);
    }
    if (meta.tts_voices.len != 0) {
        try props.put("voices", meta.tts_voices);
    }
    if (params.use_jinja) {
        if (tmpl_tools.len != 0) {
            try props.put("chat_template_tool_use", tmpl_tools);
        }
    }

    return props.done();
}

// -----------------------------------------------------------------------------
// live transcription

/// `conn.is_open()`'s negation, as `next` asks it.
const ConnStop = struct {
    conn: *mod.http.WsConn,
    pub fn shouldStop(self: ConnStop) bool {
        return !self.conn.isOpen();
    }
};

/// What the `ws_realtime` lambda (server-context.cpp:6176 @b1b770d90) keeps
/// on its stack for a live transcription session.
/// `j.value(key, json::object())` on an object: a copy of the member, of
/// whatever type, or an empty object.
fn valueOr(a: std.mem.Allocator, j: *const json.Value, key: []const u8) !json.Value {
    if (j.contains(key)) return (try j.at(key)).clone(a);
    return json.Value.emptyObject();
}

const LiveAsr = struct {
    routes: *Routes,
    conn: *mod.http.WsConn,
    arena: std.heap.ArenaAllocator,

    session_id: []const u8 = "",
    session_config: json.Value = .null,
    model_rate: c_int = 0,

    // one item per committed audio buffer, deltas are attributed to the current item
    mutex_item: stdlib.sync.Mutex = .{},
    item_id: []const u8 = "",
    item_prev: []const u8 = "", // the committed item before item_id, empty for the first one
    item_text: std.ArrayList(u8) = .empty,

    input_ended: std.atomic.Value(bool) = .init(false),

    fn lock(self: *LiveAsr) void {
        self.mutex_item.lock();
    }

    fn unlock(self: *LiveAsr) void {
        self.mutex_item.unlock();
    }

    fn randomItem(self: *LiveAsr) ![]const u8 {
        const a = self.arena.allocator();
        return std.mem.concat(a, u8, &.{ "item_", try mod.common.randomString(a) });
    }

    /// The `send_event` lambda: `ev` is `ea`'s.
    fn sendEvent(self: *LiveAsr, ea: std.mem.Allocator, ev: json.Value) !void {
        var e = ev;
        const id = try std.mem.concat(ea, u8, &.{ "event_", try mod.common.randomString(ea) });
        try e.set(ea, "event_id", try json.Value.initString(ea, id));
        _ = self.conn.send(try mod.common.safeJsonToStr(ea, &e));
    }

    /// The `send_error_event` lambda.
    fn sendErrorEvent(self: *LiveAsr, ea: std.mem.Allocator, message: []const u8, ty: []const u8) !void {
        var err = mod.common.Obj.init(ea);
        try err.put("type", ty);
        try err.put("code", @as(json.Value, .null));
        try err.put("message", message);
        try err.put("param", @as(json.Value, .null));
        var o = mod.common.Obj.init(ea);
        try o.put("type", "error");
        try o.put("error", err.done());
        try self.sendEvent(ea, o.done());
    }

    fn typed(self: *LiveAsr, ea: std.mem.Allocator, ty: []const u8) !void {
        var o = mod.common.Obj.init(ea);
        try o.put("type", ty);
        try self.sendEvent(ea, o.done());
    }

    fn sessionEvent(self: *LiveAsr, ea: std.mem.Allocator, ty: []const u8) !void {
        var o = mod.common.Obj.init(ea);
        try o.put("type", ty);
        try o.put("session", try self.session_config.clone(ea));
        try self.sendEvent(ea, o.done());
    }

    /// The `send_completed` lambda, called with `mutex_item` held.
    fn sendCompleted(self: *LiveAsr, ea: std.mem.Allocator) !void {
        var o = mod.common.Obj.init(ea);
        try o.put("type", "conversation.item.input_audio_transcription.completed");
        try o.put("item_id", self.item_id);
        try o.put("content_index", @as(i32, 0));
        try o.put("transcript", self.item_text.items);
        try self.sendEvent(ea, o.done());
        self.item_text.clearRetainingCapacity();
    }

    /// The `post_control` lambda.
    fn postControl(self: *LiveAsr, action: mod.task.ControlAction, audio: []const f32) !bool {
        var ctl = try mod.context.Reader.init(std.heap.c_allocator, self.routes.queue_tasks, self.routes.queue_results, mod.context.http_polling_seconds);
        defer ctl.deinit();
        var task = mod.task.Task.initType(.control);
        task.id = ctl.getNewId();
        const ta = try task.arenaAlloc(std.heap.c_allocator);
        task.params.control_action = action;
        task.params.control_cmpl_id = try ta.dupe(u8, self.session_id);
        task.params.control_audio = try ta.dupe(f32, audio);
        try ctl.postTask(task, false);
        const result = (try ctl.next(ConnStop{ .conn = self.conn })) orelse return false;
        defer result.destroy(std.heap.c_allocator);
        return switch (result.kind) {
            .control => |r| r.success,
            else => false,
        };
    }

    /// The `reader` thread of the lambda: client events.
    fn readerLoop(self: *LiveAsr) void {
        self.readEvents() catch |e| {
            const z = mod.common.dupZ(mod.context.whatOf(e));
            defer mod.common.freeZ(z);
            mod.common.srv(.err, "operator()", "realtime session failed: %s\n", .{z});
        };
        // the result loop waits on the socket, nobody reads it from here on
        self.conn.close();
    }

    fn readEvents(self: *LiveAsr) !void {
        var input_rate: c_int = self.model_rate;
        var resampler: ?*mtmd.vendor.miniaudio.HelperResampler = null; // input_rate -> model_rate
        defer if (resampler) |r| mtmd.helper.mtmd_helper_resampler_free(r);
        var msg: std.ArrayList(u8) = .empty;
        defer msg.deinit(std.heap.c_allocator);
        var ev_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer ev_arena.deinit();
        while (try self.conn.read(std.heap.c_allocator, &msg) != 0) {
            _ = ev_arena.reset(.retain_capacity);
            const ea = ev_arena.allocator();
            var ev = json.parse(ea, msg.items) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => {
                    try self.sendErrorEvent(ea, "invalid JSON", "invalid_request_error");
                    continue;
                },
            };
            const ty = mod.common.jsonValue(&ev, "type", []const u8, "");
            if (std.mem.eql(u8, ty, "session.update") or std.mem.eql(u8, ty, "transcription_session.update")) {
                const session = try mod.common.jsonValueJson(ea, &ev, "session", json.Value.emptyObject());
                const audio = if (session.isObject()) try valueOr(ea, &session, "audio") else json.Value.emptyObject();
                const in = if (audio.isObject()) try valueOr(ea, &audio, "input") else json.Value.emptyObject();
                if (in.isObject() and in.contains("format") and (try in.at("format")).isObject()) {
                    const rate = mod.common.jsonValue(try in.at("format"), "rate", c_int, input_rate);
                    if (rate < 8000 or rate > 192000) {
                        try self.sendErrorEvent(ea, "session.update: the input audio rate must be between 8000 and 192000 Hz", "invalid_request_error");
                        continue;
                    }
                    if (rate != input_rate) {
                        input_rate = rate;
                        if (resampler) |r| mtmd.helper.mtmd_helper_resampler_free(r);
                        resampler = null;
                        self.lock();
                        defer self.unlock();
                        const sa = self.arena.allocator();
                        (try (try (try (try self.session_config.index(sa, "audio")).index(sa, "input")).index(sa, "format")).index(sa, "rate")).* = .{ .number_integer = rate };
                    }
                }
                try self.sessionEvent(ea, "session.updated");
            } else if (std.mem.eql(u8, ty, "input_audio_buffer.append")) {
                const pcm = mod.tokens.base64HppDecode(ea, mod.common.jsonValue(&ev, "audio", []const u8, "")) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => {
                        try self.sendErrorEvent(ea, "invalid base64 audio", "invalid_request_error");
                        continue;
                    },
                };
                var samples = try mod.realtime.pcm16ToF32(ea, pcm);
                if (resampler == null) {
                    resampler = mtmd.helper.mtmd_helper_resampler_init(input_rate, self.model_rate);
                    if (resampler == null) {
                        try self.sendErrorEvent(ea, "failed to initialize the audio resampler", "server_error");
                        break;
                    }
                }
                var out: ?[*]const f32 = null;
                const n = mtmd.helper.mtmd_helper_resampler_process(resampler.?, samples.ptr, samples.len, &out);
                if (n < 0) {
                    try self.sendErrorEvent(ea, "failed to resample the audio", "server_error");
                    break;
                }
                if (n == 0) {
                    continue;
                }
                if (out != samples.ptr) {
                    samples = try ea.dupe(f32, out.?[0..@intCast(n)]);
                }
                if (!try self.postControl(.audio_append, samples)) {
                    break;
                }
            } else if (std.mem.eql(u8, ty, "input_audio_buffer.commit")) {
                // "final": true ends the input, the last item is completed when the stream drains
                if (mod.common.jsonValue(&ev, "final", bool, false)) {
                    self.input_ended.store(true, .seq_cst);
                    _ = try self.postControl(.audio_end, &.{});
                    continue;
                }
                if (!try self.postControl(.audio_commit, &.{})) {
                    break;
                }
                // completed is sent once the rows of the committed audio are decoded (audio_commit_drained below)
                self.lock();
                defer self.unlock();
                var o = mod.common.Obj.init(ea);
                try o.put("type", "input_audio_buffer.committed");
                try o.put("previous_item_id", if (self.item_prev.len == 0) @as(json.Value, .null) else try json.Value.initString(ea, self.item_prev));
                try o.put("item_id", self.item_id);
                try self.sendEvent(ea, o.done());
            } else if (std.mem.eql(u8, ty, "input_audio_buffer.clear")) {
                try self.typed(ea, "input_audio_buffer.cleared");
            } else {
                try self.sendErrorEvent(ea, try std.mem.concat(ea, u8, &.{ "unknown event type: ", ty }), "invalid_request_error");
            }
        }
        if (!self.input_ended.swap(true, .seq_cst)) {
            _ = try self.postControl(.audio_end, &.{});
        }
    }

    fn run(self: *LiveAsr) !void {
        const routes = self.routes;
        const server = routes.ctx_server;
        const meta = routes.metaRef();
        const a = self.arena.allocator();

        var ev_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer ev_arena.deinit();
        const ea = ev_arena.allocator();

        if (!meta.has_audio_stream) {
            try self.sendErrorEvent(ea, "The current model does not support realtime transcription.", "invalid_request_error");
            self.conn.close();
            return;
        }

        self.session_id = try std.mem.concat(a, u8, &.{ "sess_", try mod.common.randomString(a) });
        self.model_rate = mtmd.mtmd.mtmd_get_audio_sample_rate(server.mctx.?);
        {
            var format = mod.common.Obj.init(a);
            try format.put("type", "audio/pcm");
            try format.put("rate", self.model_rate);
            var transcription = mod.common.Obj.init(a);
            try transcription.put("model", meta.model_name);
            var input = mod.common.Obj.init(a);
            try input.put("format", format.done());
            try input.put("transcription", transcription.done());
            try input.put("turn_detection", @as(json.Value, .null));
            try input.put("noise_reduction", @as(json.Value, .null));
            var audio = mod.common.Obj.init(a);
            try audio.put("input", input.done());
            var o = mod.common.Obj.init(a);
            try o.put("id", self.session_id);
            try o.put("object", "realtime.transcription_session");
            try o.put("type", "transcription");
            try o.put("audio", audio.done());
            self.session_config = o.done();
        }

        const res = try routes.createResponse(false);
        defer res.destroy();
        const rd = &res.rd;
        {
            var task = mod.task.Task.initType(.transcribe_rt);
            task.id = rd.getNewId();
            const ta = try task.arenaAlloc(std.heap.c_allocator);
            task.params = try mod.task.TaskParams.init(ta);
            task.params.sampling = try mod.task.deepClone(ta, routes.params.sampling);
            task.params.stream = true;
            task.params.oaicompat_cmpl_id = try ta.dupe(u8, self.session_id);
            task.params.oaicompat_model = try ta.dupe(u8, meta.model_name);
            try rd.postTask(task, false);
        }

        // the slot answers once its audio stream is live, the session starts there
        {
            const result = (try rd.next(ConnStop{ .conn = self.conn })) orelse {
                self.conn.close();
                return;
            };
            defer result.destroy(std.heap.c_allocator);
            if (result.isError()) {
                var j = try result.toJson(ea);
                try self.sendErrorEvent(ea, mod.common.jsonValue(&j, "message", []const u8, "server error"), "server_error");
                self.conn.close();
                return;
            }
        }
        try self.sessionEvent(ea, "session.created");

        self.item_id = try self.randomItem();

        // client events are read on this thread, the transcript is sent from the thread below
        const reader = try std.Thread.spawn(.{}, readerLoop, .{self});

        while (true) {
            _ = ev_arena.reset(.retain_capacity);
            const result = (try rd.next(ConnStop{ .conn = self.conn })) orelse break; // connection closed
            defer result.destroy(std.heap.c_allocator);
            if (result.isError()) {
                var j = try result.toJson(ea);
                try self.sendErrorEvent(ea, mod.common.jsonValue(&j, "message", []const u8, "server error"), "server_error");
                break;
            }
            switch (result.kind) {
                .cmpl_partial => |*partial| {
                    if (partial.audio_commit_drained) {
                        // complete the committed item and open the next one
                        self.lock();
                        defer self.unlock();
                        try self.sendCompleted(ea);
                        self.item_prev = self.item_id;
                        self.item_id = try self.randomItem();
                    } else if (partial.content.text.items.len != 0) {
                        self.lock();
                        defer self.unlock();
                        try self.item_text.appendSlice(a, partial.content.text.items);
                        var o = mod.common.Obj.init(ea);
                        try o.put("type", "conversation.item.input_audio_transcription.delta");
                        try o.put("item_id", self.item_id);
                        try o.put("content_index", @as(i32, 0));
                        try o.put("delta", partial.content.text.items);
                        try self.sendEvent(ea, o.done());
                    }
                },
                else => if (result.isStop()) {
                    self.lock();
                    defer self.unlock();
                    try self.sendCompleted(ea);
                    break;
                },
            }
        }

        self.input_ended.store(true, .seq_cst);
        self.conn.close();
        reader.join();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
