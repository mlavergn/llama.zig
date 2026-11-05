//! `llama-server`: the HTTP server over libllama (Stage 10, Decision 65).
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server.cpp` at v0.7.0 (`620d2225c`):
//! `llama_server(argc, argv)`, `llama_server(params, argc, argv)` -- the
//! model, the HTTP context, the route table, the 403 handlers, shutdown and
//! the signal handler -- `ex_wrapper` and `llama_server_terminate`. Each
//! declaration names the C++ it replaces and the line it began at.
//!
//! # Temporary gaps
//!
//! Router mode (Stage 10 step 7) is `models.zig`: without a model (`-m`,
//! `-hf`, `-dr`) the server routes to child servers it spawns, and a
//! process with `LLAMA_SERVER_ROUTER_PORT` set is such a child. Model
//! downloads are `download.zig`, over plain HTTP only, as the reference
//! (built without SSL) has them.
//!
//! Tools, MCP and the CORS proxy (Stage 10 step 6) are `tools.zig`,
//! `mcp.zig` and `cors_proxy.zig`; with them off, `/tools` and
//! `/cors-proxy` answer 403 as upstream's do.
//!
//! # The web UI
//!
//! The reference compiles `llama.cpp.zmake`'s `ui-stub`: no assets, so
//! `http.zig` serves none and neither does this.

const std = @import("std");
const builtin = @import("builtin");
const tc = @import("tools_common");
const json = @import("json");
const c = tc.c;
const io = tc.io;
const arg = tc.arg;

const http = @import("http.zig");
const common = @import("common.zig");
const context = @import("context.zig");
const routes_mod = @import("routes.zig");
const stream = @import("stream.zig");
const tools_mod = @import("tools.zig");
const mcp_mod = @import("mcp.zig");
const cors_proxy = @import("cors_proxy.zig");
const models_mod = @import("models.zig");
const download = @import("download.zig");

const Params = tc.params.Params;
const gpa = context.gpa;
const srv = common.srv;

extern fn setlocale(cat: c_int, locale: [*:0]const u8) ?[*:0]const u8;

/// `LC_NUMERIC`, as `<locale.h>` numbers it on each target.
const lc_numeric: c_int = if (builtin.os.tag.isDarwin()) 4 else if (builtin.os.tag == .windows) 4 else 1;

// -----------------------------------------------------------------------------
// shutdown

/// Ports `shutdown_handler` (server.cpp:25 @620d2225c): what a signal does,
/// set once the server is up.
var shutdown_handler: ?*const fn (signal: c_int) void = null;
/// What `shutdown_handler` acts on.
var shutdown_ctx_server: ?*context.ServerContext = null;
/// The `mcp_mgr` `shutdown_handler` and `clean_up` shut down.
var shutdown_mcp: ?*mcp_mod.Mcp = null;

/// Ports `is_terminating` (server.cpp:26 @620d2225c).
var is_terminating: std.atomic.Value(bool) = .init(false);

extern fn exit(code: c_int) noreturn;

/// Ports `signal_handler` (server.cpp:28 @620d2225c).
fn signalHandler(sig: std.posix.SIG) callconv(.c) void {
    const signal: c_int = @intCast(@backingInt(sig));
    if (is_terminating.swap(true, .seq_cst)) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        io.eprintf("Received second interrupt, terminating immediately.\n", .{});
        exit(1);
    }

    if (shutdown_handler) |h| h(signal);
}

/// The single-model `shutdown_handler` lambda (server.cpp:500 @620d2225c):
/// unblocks `start_loop()`.
fn shutdownSingleModel(_: c_int) void {
    if (shutdown_mcp) |m| m.shutdown();
    // this will unblock start_loop()
    if (shutdown_ctx_server) |s| s.terminate();
}

/// What the router's `shutdown_handler` acts on (Stage 10 step 7).
var shutdown_models_routes: ?*models_mod.ModelsRoutes = null;
var shutdown_ctx_http: ?*http.Context = null;

/// The router's `shutdown_handler` lambda (server.cpp:443 @620d2225c):
/// disconnects the SSE clients and stops the HTTP server, which ends the
/// router's wait on its thread.
fn shutdownRouter(_: c_int) void {
    if (shutdown_models_routes) |m| {
        // important to disconnect any SSE clients
        m.stopping.store(true, .seq_cst);
    }
    if (shutdown_mcp) |m| m.shutdown();
    if (shutdown_ctx_http) |h| h.stop();
}

/// Ports `llama_server_terminate` (server.cpp:46 @620d2225c): for the CLI.
pub fn terminate() void {
    if (shutdown_handler) |h| h(0);
}

// -----------------------------------------------------------------------------
// ex_wrapper

/// Ports `ex_wrapper` (server.cpp:55 @620d2225c): a route that never
/// throws -- an exception becomes an error response, 400 for
/// `std::invalid_argument` and `common_json_error`, 500 for anything else.
const ExWrapper = struct {
    func: http.Handler,

    fn wrap(a: std.mem.Allocator, func: http.Handler) !http.Handler {
        const self = try a.create(ExWrapper);
        self.* = .{ .func = func };
        return .{ .ctx = self, .func = call };
    }

    fn call(ctx: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
        const self: *ExWrapper = @ptrCast(@alignCast(ctx.?));
        var err_type: common.ErrorType = undefined;
        var buf: [4200]u8 = undefined;
        var message: []const u8 = undefined;
        if (self.func.call(req)) |r| {
            return r;
        } else |e| switch (e) {
            // treat invalid_argument as invalid request (400)
            error.InvalidArgument => {
                err_type = .invalid_request;
                message = copyWhat(&buf, e);
            },
            // JSON parse and type errors are invalid requests (400)
            error.JsonError => {
                err_type = .invalid_request;
                message = copyWhat(&buf, e);
            },
            // treat other exceptions as server error (500)
            error.RuntimeError, error.OutOfMemory, error.Runtime, error.Throw => {
                err_type = .server;
                message = copyWhat(&buf, e);
            },
            else => {
                err_type = .server;
                message = "unknown error";
            },
        }

        const res = try http.Res.create(gpa);
        errdefer res.destroy();
        const a = res.allocator();
        res.status = 500;
        build: {
            var error_data = common.formatErrorResponse(a, message, err_type) catch |e2| {
                srvErrAnother(e2, message);
                try res.setData("Internal Server Error");
                break :build;
            };
            res.status = common.jsonValue(&error_data, "code", c_int, 500);
            var outer = json.Value.emptyObject();
            try outer.set(a, "error", error_data);
            try res.setData(try common.safeJsonToStr(a, &outer));
            const z = try a.dupeSentinel(u8, res.data.items, 0);
            srv(.wrn, "operator()", "got exception: %s\n", .{z.ptr});
        }
        return res;
    }

    fn copyWhat(buf: []u8, e: anyerror) []const u8 {
        const w = context.whatOf(e);
        const n = @min(w.len, buf.len);
        @memcpy(buf[0..n], w[0..n]);
        return buf[0..n];
    }

    fn srvErrAnother(e: anyerror, message: []const u8) void {
        const w = common.dupZ(context.whatOf(e));
        defer common.freeZ(w);
        const m = common.dupZ(message);
        defer common.freeZ(m);
        srv(.err, "operator()", "got another exception: %s | while handling exception: %s\n", .{ w, m });
    }
};

/// The `res_403` handler (server.cpp:328 @620d2225c): for disabled features.
fn res403(_: ?*anyopaque, _: *const http.Req) anyerror!*http.Res {
    const res = try http.Res.create(gpa);
    errdefer res.destroy();
    const a = res.allocator();
    res.status = 403;
    var err = common.Obj.init(a);
    try err.put("message", "this feature is disabled");
    try err.put("type", "feature_disabled");
    var outer = common.Obj.init(a);
    try outer.put("error", err.done());
    var v = outer.done();
    try res.setData(try common.safeJsonToStr(a, &v));
    return res;
}

// -----------------------------------------------------------------------------
// llama_server

/// Ports `llama_server` (server.cpp:93 @620d2225c), the overload taking
/// `argc` and `argv`.
///
/// Parameters:
/// - `a`: the process's arena; `Params` keeps its strings there.
/// - `argv`: the command line, `argv[0]` included.
///
/// Return: the exit status.
pub fn run(a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    // in a child, keeps stdout for the commands to the router, before anything is written
    var child = models_mod.Child.init();
    defer child.deinit();

    _ = setlocale(lc_numeric, "C");

    if (comptime builtin.os.tag != .windows) {
        // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
        var ignore: std.posix.Sigaction = .{
            .handler = .{ .handler = std.posix.SIG.IGN },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.PIPE, &ignore, null);
    }

    // own arguments required by this example
    var params = try Params.init(a);

    tc.common.init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    try stream.sessionManagerStart();

    // `common/chat.cpp`'s log and its template check, which the parse runs
    context.installChat();
    if (!try arg.parse(argv, &params, .server, null)) return 1;

    srv(.inf, "llama_server", "%s", .{@as([*:0]const u8, "initializing ...\n")});

    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    // `common_log_flush(common_log_main())` after this returns has nothing
    // to wait for: the port's log writes each message as it is made
    return runParams(a, &params, argv, &child);
}

/// `SRV_ERR(fmt, e.what())` for a caught exception.
fn srvErrWhat(comptime func: []const u8, comptime fmt: []const u8, e: anyerror) void {
    const w = common.dupZ(context.whatOf(e));
    defer common.freeZ(w);
    srv(.err, func, fmt, .{w});
}

/// The server's `common_params` as `server_http_context::init` reads them.
fn httpParams(p: *const Params) http.Params {
    return .{
        .api_prefix = p.api_prefix,
        .port = p.port,
        .hostnames = p.hostnames.items,
        .ssl_file_key = p.ssl_file_key,
        .ssl_file_cert = p.ssl_file_cert,
        .timeout_read = p.timeout_read,
        .timeout_write = p.timeout_write,
        .reuse_port = p.reuse_port,
        .api_keys = p.api_keys.items,
        .cors_origins = p.cors_origins,
        .cors_methods = p.cors_methods,
        .cors_headers = p.cors_headers,
        .cors_credentials = p.cors_credentials,
        .n_threads_http = p.n_threads_http,
        .n_parallel = p.n_parallel,
        .ui = p.ui,
        .public_path = p.public_path,
    };
}

/// Ports `llama_server` (server.cpp:131 @620d2225c), the overload taking
/// the parsed parameters and the child state. `argv` is null when the CLI
/// runs it.
fn runParams(a: std.mem.Allocator, params: *Params, argv: ?[]const [:0]const u8, child: *models_mod.Child) !u8 {
    const func = "llama_server";
    const is_run_by_cli = argv == null;

    var models_handler: download.ModelsHandler = .{};

    // note: router mode also accepts -hf remote-preset, so we need to check that first
    if (!is_run_by_cli and params.model.hf_repo.len != 0) {
        fetch: {
            models_handler = download.modelsHandlerInit(a, params, .server) catch |e| {
                srvErrWhat(func, "failed to fetch model metadata: %s\n", e);
                return 1;
            };
            if (download.modelsHandlerIsPresetRepo(&models_handler)) {
                // apply the preset and start the server in router mode
                download.modelsHandlerApply(a, &models_handler, params, null) catch |e| {
                    srvErrWhat(func, "failed to fetch model metadata: %s\n", e);
                    return 1;
                };
            }
            break :fetch;
        }
    }

    // router server never loads a model and must not touch the GPU
    const is_router_server = params.model.path.len == 0 and
        params.model.hf_repo.len == 0 and
        params.model.docker_repo.len == 0;

    // skip device enumeration so the CUDA primary context stays uncreated
    try tc.common.paramsPrintInfo(a, params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding and params.n_batch > params.n_ubatch) {
            srv(.wrn, func, "embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", .{ params.n_batch, params.n_ubatch });
            srv(.wrn, func, "setting n_batch = n_ubatch = %d to avoid assertion failure\n", .{params.n_ubatch});
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            srv(.trc, func, "%s", .{@as([*:0]const u8, "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n")});

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const ctx_pool_auto_sized = params.kv_unified_per_slot > 0 and
        params.n_ctx == 0 and
        @as(u32, @bitCast(params.fit_params_min_ctx)) != std.math.maxInt(u32);

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        srv(.inf, func, "--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n", .{ params.n_parallel, params.kv_unified_per_slot, params.n_ctx });
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    const model_name = params.model.getName();
    if (params.model_alias.isEmpty() and model_name.len != 0) {
        try params.model_alias.insert(a, model_name);
    }

    // note: this is guaranteed to out-live ctx_http and tools
    var mcp_mgr: mcp_mod.Mcp = .{};
    defer mcp_mgr.deinit();
    shutdown_mcp = &mcp_mgr;

    // struct that contains llama context and inference
    var ctx_server = try context.ServerContext.init();
    defer ctx_server.deinit();

    //
    // Router
    //

    // register API routes
    var routes: routes_mod.Routes = undefined;
    try routes.init(params, &ctx_server);
    defer routes.deinit();
    var tools = try tools_mod.Tools.init();
    defer tools.deinit();

    const W = struct {
        fn w(ar: std.mem.Allocator, h: http.Handler) !http.Handler {
            return ExWrapper.wrap(ar, h);
        }
    };

    // std::optional<server_models_routes> models_routes
    var models_routes: ?*models_mod.ModelsRoutes = null;
    defer if (models_routes) |m| {
        m.deinit();
        gpa.destroy(m);
    };

    // declared after everything its listeners' handlers reach, so it is
    // stopped and joined before they are destroyed
    var http_params = httpParams(params);
    var ctx_http = try http.Context.init(gpa);
    defer ctx_http.deinit();
    if (!try ctx_http.setup(&http_params)) {
        srv(.err, func, "%s", .{@as([*:0]const u8, "failed to initialize HTTP server\n")});
        return 1;
    }
    if (is_router_server) {
        // setup server instances manager
        const mr = try gpa.create(models_mod.ModelsRoutes);
        mr.init(params, argv orelse &.{}) catch |e| {
            gpa.destroy(mr);
            srvErrWhat(func, "failed to initialize router models: %s\n", e);
            return 1;
        };
        models_routes = mr;

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics = mr.proxy_get;
        routes.post_props = mr.proxy_post;
        routes.post_completions = mr.proxy_post;
        routes.post_completions_oai = mr.proxy_post;
        routes.post_chat_completions = mr.proxy_post;
        routes.post_control = mr.proxy_post;
        routes.post_responses_oai = mr.proxy_post;
        routes.post_transcriptions_oai = mr.proxy_post;
        routes.post_anthropic_messages = mr.proxy_post;
        routes.post_anthropic_count_tokens = mr.proxy_post;
        routes.post_infill = mr.proxy_post;
        routes.post_embeddings = mr.proxy_post;
        routes.post_embeddings_oai = mr.proxy_post;
        routes.post_rerank = mr.proxy_post;
        routes.post_systemone = mr.proxy_post;
        routes.post_tokenize = mr.proxy_post;
        routes.post_detokenize = mr.proxy_post;
        routes.post_apply_template = mr.proxy_post;
        routes.post_chat_completions_tok = mr.proxy_post;
        routes.post_responses_tok_oai = mr.proxy_post;
        routes.get_lora_adapters = mr.proxy_get;
        routes.post_lora_adapters = mr.proxy_post;
        routes.get_slots = mr.proxy_get;
        routes.post_slots = mr.proxy_post;

        // custom routes for router
        routes.get_props = mr.get_router_props;
        routes.get_models = mr.get_router_models;

        try ctx_http.post("/models", try W.w(a, mr.post_router_models));
        try ctx_http.post("/models/load", try W.w(a, mr.post_router_models_load));
        try ctx_http.post("/models/unload", try W.w(a, mr.post_router_models_unload));
        try ctx_http.get("/models/sse", try W.w(a, mr.get_router_models_sse));
        try ctx_http.del("/models", try W.w(a, mr.del_router_models));
    }

    try ctx_http.get("/health", try W.w(a, routes.get_health)); // public endpoint (no API key check)
    try ctx_http.get("/v1/health", try W.w(a, routes.get_health)); // public endpoint (no API key check)
    try ctx_http.get("/metrics", try W.w(a, routes.get_metrics));
    try ctx_http.get("/props", try W.w(a, routes.get_props));
    try ctx_http.post("/props", try W.w(a, routes.post_props));
    try ctx_http.get("/models", try W.w(a, routes.get_models));
    try ctx_http.get("/v1/models", try W.w(a, routes.get_models));
    try ctx_http.post("/completion", try W.w(a, routes.post_completions)); // legacy
    try ctx_http.post("/completions", try W.w(a, routes.post_completions));
    try ctx_http.post("/v1/completions", try W.w(a, routes.post_completions_oai));
    try ctx_http.post("/chat/completions", try W.w(a, routes.post_chat_completions));
    try ctx_http.post("/v1/chat/completions", try W.w(a, routes.post_chat_completions));
    try ctx_http.post("/v1/chat/completions/control", try W.w(a, routes.post_control));
    try ctx_http.post("/v1/responses", try W.w(a, routes.post_responses_oai));
    try ctx_http.post("/responses", try W.w(a, routes.post_responses_oai));
    try ctx_http.post("/v1/audio/transcriptions", try W.w(a, routes.post_transcriptions_oai));
    try ctx_http.post("/audio/transcriptions", try W.w(a, routes.post_transcriptions_oai));
    try ctx_http.post("/v1/messages", try W.w(a, routes.post_anthropic_messages)); // anthropic messages API
    try ctx_http.post("/infill", try W.w(a, routes.post_infill));
    try ctx_http.post("/embedding", try W.w(a, routes.post_embeddings)); // legacy
    try ctx_http.post("/embeddings", try W.w(a, routes.post_embeddings));
    try ctx_http.post("/v1/embeddings", try W.w(a, routes.post_embeddings_oai));
    try ctx_http.post("/rerank", try W.w(a, routes.post_rerank));
    try ctx_http.post("/reranking", try W.w(a, routes.post_rerank));
    try ctx_http.post("/v1/rerank", try W.w(a, routes.post_rerank));
    try ctx_http.post("/v1/reranking", try W.w(a, routes.post_rerank));
    try ctx_http.post("/v1/systemone", try W.w(a, routes.post_systemone));
    try ctx_http.post("/tokenize", try W.w(a, routes.post_tokenize));
    try ctx_http.post("/detokenize", try W.w(a, routes.post_detokenize));
    try ctx_http.post("/apply-template", try W.w(a, routes.post_apply_template));
    // token counting
    try ctx_http.post("/chat/completions/input_tokens", try W.w(a, routes.post_chat_completions_tok));
    try ctx_http.post("/v1/chat/completions/input_tokens", try W.w(a, routes.post_chat_completions_tok));
    try ctx_http.post("/responses/input_tokens", try W.w(a, routes.post_responses_tok_oai));
    try ctx_http.post("/v1/responses/input_tokens", try W.w(a, routes.post_responses_tok_oai));
    try ctx_http.post("/v1/messages/count_tokens", try W.w(a, routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    try ctx_http.get("/lora-adapters", try W.w(a, routes.get_lora_adapters));
    try ctx_http.post("/lora-adapters", try W.w(a, routes.post_lora_adapters));
    // Save & load slots
    try ctx_http.get("/slots", try W.w(a, routes.get_slots));
    try ctx_http.post("/slots/:id_slot", try W.w(a, routes.post_slots));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child, see server-stream.h
    const stream_get_h = if (models_routes) |mr| mr.router_stream_get else stream.makeGetHandler();
    const streams_lookup_h = if (models_routes) |mr| mr.router_streams_lookup else stream.makeLookupHandler();
    const stream_delete_h = if (models_routes) |mr| mr.router_stream_delete else stream.makeDeleteHandler();
    try ctx_http.get("/v1/stream", try W.w(a, stream_get_h));
    try ctx_http.post("/v1/streams/lookup", try W.w(a, streams_lookup_h));
    try ctx_http.del("/v1/stream", try W.w(a, stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    try ctx_http.registerGcpCompat();

    // return 403 for disabled features
    const res_403: http.Handler = .{ .ctx = null, .func = res403 };

    if (std.mem.eql(u8, params.cors_origins, "*") and params.api_keys.items.len == 0) {
        srv(.wrn, func, "%s", .{@as([*:0]const u8, "security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)\n")});
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    var warn_names: std.ArrayList([]const u8) = .empty;
    if (is_router_server) {
        try warn_names.append(a, "router mode");
    }

    if (params.ui_mcp_proxy) {
        try ctx_http.get("/cors-proxy", try W.w(a, cors_proxy.handler_get));
        try ctx_http.post("/cors-proxy", try W.w(a, cors_proxy.handler_post));
        try warn_names.append(a, "MCP proxy (experimental)");
    } else {
        try ctx_http.get("/cors-proxy", try W.w(a, res_403));
        try ctx_http.post("/cors-proxy", try W.w(a, res_403));
    }

    mcp_mgr.start(params.mcp_servers_config, params.mcp_servers_json) catch |e| {
        const w = common.dupZ(context.whatOf(e));
        defer common.freeZ(w);
        srv(.err, func, "MCP starting failed: %s\n", .{w});
        return 1;
    };

    if (params.server_tools.items.len != 0 or !mcp_mgr.empty()) {
        tools.setup(params.server_tools.items, &mcp_mgr, params.server_tools_runtime) catch |e| {
            const w = common.dupZ(context.whatOf(e));
            defer common.freeZ(w);
            srv(.err, func, "tools setup failed: %s\n", .{w});
            return 1;
        };
        try ctx_http.get("/tools", try W.w(a, tools.handleGet()));
        try ctx_http.post("/tools", try W.w(a, tools.handlePost()));
        if (params.server_tools.items.len != 0) try warn_names.append(a, "server tools (experimental)");
        if (params.server_tools_runtime.len != 0) try warn_names.append(a, "tools runtime (experimental)");
        if (!mcp_mgr.empty()) try warn_names.append(a, "MCP servers (experimental)");
    } else {
        try ctx_http.get("/tools", try W.w(a, res_403));
        try ctx_http.post("/tools", try W.w(a, res_403));
    }

    if (warn_names.items.len != 0) {
        var features: std.ArrayList(u8) = .empty;
        for (warn_names.items) |name| {
            if (features.items.len != 0) try features.appendSlice(a, ", ");
            try features.appendSlice(a, name);
        }
        const fz = try a.dupeSentinel(u8, features.items, 0);
        srv(.wrn, func, "security: %s enabled - do not expose to untrusted environments\n", .{fz.ptr});
    }

    //
    // Handle downloading model
    //

    if (child.isChild() and child.getMode() == .download) {
        return @intCast(child.runDownload(params));
    } else if (!is_router_server and !is_run_by_cli) {
        // single-model mode (NOT spawned by router)
        // if this is invoked by CLI, model downloading should be already handled
        download.modelsHandlerApply(a, &models_handler, params, null) catch |e| {
            srvErrWhat(func, "failed to download model: %s\n", e);
            return 1;
        };
    }

    //
    // Start the server
    //

    // setup clean up function, to be called before exit
    const CleanUp = struct {
        fn run(h: *http.Context, s: *context.ServerContext) void {
            srv(.inf, "operator()", "%s: cleaning up before exit...\n", .{@as([*:0]const u8, "operator()")});
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            stream.sessionManagerStop();
            h.stop();
            s.terminate();
            if (shutdown_mcp) |m| m.shutdown();
            c.llama_backend_free();
        }
    };

    if (models_routes) |mr| {
        srv(.inf, func, "%s", .{@as([*:0]const u8, "starting server in router mode. models will be automatically loaded on-demand\n")});

        // the router's `clean_up` lambda (server.cpp:424 @620d2225c)
        const RouterCleanUp = struct {
            fn run(m: *models_mod.ModelsRoutes) void {
                srv(.inf, "operator()", "%s: cleaning up before exit...\n", .{@as([*:0]const u8, "operator()")});
                // stop the session GC first, it finalizes live sessions and wakes pending readers
                stream.sessionManagerStop();
                m.stopping.store(true, .seq_cst); // maybe redundant, but just to be safe
                m.models.unloadAll();
                if (shutdown_mcp) |mm| mm.shutdown();
                c.llama_backend_free();
            }
        };

        if (!try ctx_http.start()) {
            RouterCleanUp.run(mr);
            srv(.err, func, "%s", .{@as([*:0]const u8, "exiting due to HTTP server error\n")});
            return 1;
        }
        ctx_http.is_ready.store(true, .seq_cst);

        shutdown_models_routes = mr;
        shutdown_ctx_http = &ctx_http;
        shutdown_handler = shutdownRouter;

        mr.models.loadStartupModels() catch |e| {
            srvErrWhat(func, "failed to load models on startup: %s\n", e);
            ctx_http.stop();
            ctx_http.join();
            RouterCleanUp.run(mr);
            return 1;
        };

        registerSignals(is_run_by_cli);
        try logListening(a, &ctx_http);

        if (params.models_preset_hf.len != 0) {
            const z = try a.dupeSentinel(u8, params.models_preset_hf, 0);
            srv(.wrn, func, "NOTE: using preset.ini from HF repo '%s'\n", .{z.ptr});
            srv(.wrn, func, "%s", .{@as([*:0]const u8, "      please only use presets that you can trust! Unknown presets may be unsafe\n")});
        }

        ctx_http.join(); // keep the main thread alive

        // when the HTTP server stops, clean up and exit
        RouterCleanUp.run(mr);
        return 0;
    }

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!try ctx_http.start()) {
        CleanUp.run(&ctx_http, &ctx_server);
        srv(.err, func, "%s", .{@as([*:0]const u8, "exiting due to HTTP server error\n")});
        return 1;
    }

    // setup communication child --> router if necessary
    if (child.isChild()) {
        ctx_server.setStateCallback(child.stateCallback());
    }

    if (!try ctx_server.loadModel(params)) {
        CleanUp.run(&ctx_http, &ctx_server);
        ctx_http.join();
        srv(.err, func, "%s", .{@as([*:0]const u8, "exiting due to model loading error\n")});
        return 1;
    }

    // `routes.params` and `http_params` read `params`, which loading updated
    http_params = httpParams(params);
    try routes.updateMeta(&ctx_server);
    ctx_http.is_ready.store(true, .seq_cst);

    srv(.inf, func, "%s", .{@as([*:0]const u8, "model loaded\n")});

    shutdown_ctx_server = &ctx_server;
    shutdown_handler = shutdownSingleModel;

    registerSignals(is_run_by_cli);
    try logListening(a, &ctx_http);

    // optionally, notify router server that this instance is ready
    var monitor_thread: ?std.Thread = null;
    if (child.isChild()) {
        monitor_thread = child.setup(.{ .ctx = null, .func = struct {
            fn f(_: ?*anyopaque, sig: c_int) void {
                if (shutdown_handler) |h| h(sig);
            }
        }.f });
        var info_arena = std.heap.ArenaAllocator.init(gpa);
        defer info_arena.deinit();
        const info = try routes.getModelInfo(info_arena.allocator());
        child.notifyToRouter(context.serverStateToStr(.ready), &info);
    }

    // this call blocks the main thread until queue_tasks.terminate() is called
    try ctx_server.startLoop();

    CleanUp.run(&ctx_http, &ctx_server);
    ctx_http.join();
    if (monitor_thread) |t| {
        t.join();
    }

    if (ctx_server.getLlamaContext()) |ll_ctx| {
        try @import("cli_fit").memoryBreakdownPrint(gpa, .{ .verbosity = std.math.maxInt(i32), .sink = tc.common.fitLogSink, .json = tc.common.fitLogJson }, @ptrCast(ll_ctx));
    }

    return 0;
}

/// Registers `signal_handler` (server.cpp:511 @620d2225c) for SIGINT and
/// SIGTERM, unless the CLI runs the server.
fn registerSignals(is_run_by_cli: bool) void {
    if (!is_run_by_cli) {
        if (comptime builtin.os.tag != .windows) {
            var sigint_action: std.posix.Sigaction = .{
                .handler = .{ .handler = signalHandler },
                .mask = std.posix.sigemptyset(),
                .flags = 0,
            };
            std.posix.sigaction(std.posix.SIG.INT, &sigint_action, null);
            std.posix.sigaction(std.posix.SIG.TERM, &sigint_action, null);
        }
    }
}

/// Part of `llama_server` (server.cpp:131 @620d2225c): the "listening on"
/// loop, a line per address.
fn logListening(a: std.mem.Allocator, ctx_http: *const http.Context) !void {
    const func = "llama_server";
    for (ctx_http.listening_addresses.items) |address| {
        const z = try a.dupeSentinel(u8, address, 0);
        srv(.inf, func, "listening on %s\n", .{z.ptr});
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
