//! `llama-server`: the HTTP server over libllama (Stage 10, Decision 65).
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server.cpp` at v0.9.0 (`cd959bae9`):
//! `llama_server(argc, argv)` and what it runs by role -- the router, an
//! instance, the download and measure children -- the route table, the 403
//! handlers, shutdown and the signal handler, `ex_wrapper` and
//! `llama_server_terminate`. `llama_server(params, argc, argv)`, llama-cli's
//! way in, is not ported: the CLI does not embed the server. Each declaration
//! names the C++ it replaces and the line it began at.
//!
//! # Roles
//!
//! `server_run` chooses by `server_role_type`: no model (`-m`, `-hf`, `-dr`)
//! is a router over child servers (`models.zig`), and a process with
//! `LLAMA_SERVER_ROUTER_PORT` set is such a child: an instance, or the
//! download or measure worker `LLAMA_SERVER_CHILD_ROLE` names. Model
//! downloads are `download.zig`, over plain HTTP only, as the reference
//! (built without SSL) has them. `--connect` is `connect.zig`.
//!
//! Tools, MCP and the CORS proxy are `tools.zig`, `mcp.zig` and
//! `cors_proxy.zig`; with them off, `/tools` and `/cors-proxy` answer 403 as
//! upstream's do.
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
const models_sched = @import("models_sched.zig");
const connect = @import("connect.zig");
const fit = @import("cli_fit");

const Params = tc.params.Params;
const gpa = context.gpa;
const srv = common.srv;

extern fn setlocale(cat: c_int, locale: [*:0]const u8) ?[*:0]const u8;

/// `LC_NUMERIC`, as `<locale.h>` numbers it on each target.
const lc_numeric: c_int = if (builtin.os.tag.isDarwin()) 4 else if (builtin.os.tag == .windows) 4 else 1;

// -----------------------------------------------------------------------------
// shutdown

/// Ports `shutdown_handler` (server.cpp:26 @cd959bae9): what a signal does,
/// set once the server is up.
var shutdown_handler: ?*const fn (signal: c_int) void = null;
/// What `shutdown_handler` acts on.
var shutdown_ctx_server: ?*context.ServerContext = null;
/// The `mcp_mgr` `shutdown_handler` and `clean_up` shut down.
var shutdown_mcp: ?*mcp_mod.Mcp = null;

/// Ports `is_terminating` (server.cpp:27 @cd959bae9).
var is_terminating: std.atomic.Value(bool) = .init(false);

extern fn exit(code: c_int) noreturn;

/// Ports `signal_handler` (server.cpp:29 @cd959bae9).
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

/// The single-model `shutdown_handler` lambda (server.cpp:684 @cd959bae9):
/// unblocks `start_loop()`.
fn shutdownSingleModel(_: c_int) void {
    if (shutdown_mcp) |m| m.shutdown();
    // this will unblock start_loop()
    if (shutdown_ctx_server) |s| s.terminate();
}

/// What the router's `shutdown_handler` acts on (Stage 10 step 7).
var shutdown_models_routes: ?*models_mod.ModelsRoutes = null;
var shutdown_ctx_http: ?*http.Context = null;

/// The router's `shutdown_handler` lambda (server.cpp:564 @cd959bae9):
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

/// Ports `llama_server_terminate` (server.cpp:52 @cd959bae9): for the CLI.
pub fn terminate() void {
    if (shutdown_handler) |h| h(0);
}

// -----------------------------------------------------------------------------
// ex_wrapper

/// Ports `ex_wrapper` (server.cpp:61 @cd959bae9): a route that never
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

/// The `res_403` handler (server.cpp:241 @cd959bae9): for disabled features.
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

/// Ports `llama_server` (server.cpp:772 @cd959bae9), the overload taking
/// `argc` and `argv`.
///
/// Parameters:
/// - `a`: the process's arena; `Params` keeps its strings there.
/// - `argv`: the command line, `argv[0]` included.
///
/// Return: the exit status.
pub fn run(a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    // in a child, keeps stdout for the commands to the router, before anything is written
    var role = models_mod.Role.init();
    defer role.deinit();

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
    return serverRun(a, &params, argv, &role, .{});
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

/// Ports `server_run_opts` (server.cpp:41 @cd959bae9): how the server is run;
/// everything is on when it is the llama-server binary.
const RunOpts = struct {
    install_signal_handlers: bool = true, // off when embedded in llama-cli, which handles them itself
    resolve_model: bool = true, // download / resolve -hf and docker models; off when the caller did it already
};

//
// shared by every role
//

/// Ports `server_role_detect` (server.cpp:103 @cd959bae9): a server given no
/// model is a router.
fn serverRoleDetect(params: *const Params, role: *const models_mod.Role) models_mod.RoleType {
    if (role.isManagedByRouter()) {
        return role.getRole();
    }
    const has_model = params.model.path.len != 0 or
        params.model.hf_repo.len != 0 or
        params.model.docker_repo.len != 0;
    return if (has_model) .instance else .router;
}

/// Ports `server_prepare_params` (server.cpp:115 @cd959bae9): adjust the
/// params for the role.
///
/// Return: false on a configuration error.
fn serverPrepareParams(a: std.mem.Allocator, params: *Params, role_type: models_mod.RoleType) !bool {
    const func = "server_prepare_params";
    if (role_type != .router) {
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

    // check early, so a missing llama-connect fails-fast
    if (params.server_connect) {
        const reason = try connect.ServerConnect.unavailableReason(a, params);
        if (reason.len != 0) {
            srv(.err, func, "--connect is not available: %s\n", .{(try a.dupeSentinel(u8, reason, 0)).ptr});
            return false;
        }
        params.server_connect_code = try connect.ServerConnect.resolveCode(a, params.server_connect_code);
    } else if (params.server_connect_code.len != 0) {
        // the code is only meaningful while the tunnel runs, and /props keys off it being set
        srv(.wrn, func, "%s", .{@as([*:0]const u8, "--connect-code is ignored without --connect\n")});
        params.server_connect_code = "";
    }

    return true;
}

/// `ex_wrapper(h)`.
fn exw(a: std.mem.Allocator, h: http.Handler) !http.Handler {
    return ExWrapper.wrap(a, h);
}

/// Ports `server_register_routes` (server.cpp:170 @cd959bae9): the API of
/// every server; a router points the handlers in `routes` at its proxies
/// before calling this.
fn serverRegisterRoutes(a: std.mem.Allocator, ctx_http: *http.Context, routes: *routes_mod.Routes) !void {
    try ctx_http.get("/health", try exw(a, routes.get_health)); // public endpoint (no API key check)
    try ctx_http.get("/v1/health", try exw(a, routes.get_health)); // public endpoint (no API key check)
    try ctx_http.get("/metrics", try exw(a, routes.get_metrics));
    try ctx_http.get("/props", try exw(a, routes.get_props));
    try ctx_http.post("/props", try exw(a, routes.post_props));
    try ctx_http.get("/models", try exw(a, routes.get_models));
    try ctx_http.get("/v1/models", try exw(a, routes.get_models));
    try ctx_http.post("/completion", try exw(a, routes.post_completions)); // legacy
    try ctx_http.post("/completions", try exw(a, routes.post_completions));
    try ctx_http.post("/v1/completions", try exw(a, routes.post_completions_oai));
    try ctx_http.post("/chat/completions", try exw(a, routes.post_chat_completions));
    try ctx_http.post("/v1/chat/completions", try exw(a, routes.post_chat_completions));
    try ctx_http.post("/v1/chat/completions/control", try exw(a, routes.post_control));
    try ctx_http.post("/v1/responses", try exw(a, routes.post_responses_oai));
    try ctx_http.post("/responses", try exw(a, routes.post_responses_oai));
    try ctx_http.post("/v1/audio/transcriptions", try exw(a, routes.post_transcriptions_oai));
    try ctx_http.post("/audio/transcriptions", try exw(a, routes.post_transcriptions_oai));
    try ctx_http.post("/tts", try exw(a, routes.post_tts));
    try ctx_http.ws("/v1/realtime", routes.ws_realtime);
    try ctx_http.post("/v1/messages", try exw(a, routes.post_anthropic_messages)); // anthropic messages API
    try ctx_http.post("/infill", try exw(a, routes.post_infill));
    try ctx_http.post("/embedding", try exw(a, routes.post_embeddings)); // legacy
    try ctx_http.post("/embeddings", try exw(a, routes.post_embeddings));
    try ctx_http.post("/v1/embeddings", try exw(a, routes.post_embeddings_oai));
    try ctx_http.post("/rerank", try exw(a, routes.post_rerank));
    try ctx_http.post("/reranking", try exw(a, routes.post_rerank));
    try ctx_http.post("/v1/rerank", try exw(a, routes.post_rerank));
    try ctx_http.post("/v1/reranking", try exw(a, routes.post_rerank));
    try ctx_http.post("/v1/systemone", try exw(a, routes.post_systemone));
    try ctx_http.post("/tokenize", try exw(a, routes.post_tokenize));
    try ctx_http.post("/detokenize", try exw(a, routes.post_detokenize));
    try ctx_http.post("/apply-template", try exw(a, routes.post_apply_template));
    // token counting
    try ctx_http.post("/chat/completions/input_tokens", try exw(a, routes.post_chat_completions_tok));
    try ctx_http.post("/v1/chat/completions/input_tokens", try exw(a, routes.post_chat_completions_tok));
    try ctx_http.post("/responses/input_tokens", try exw(a, routes.post_responses_tok_oai));
    try ctx_http.post("/v1/responses/input_tokens", try exw(a, routes.post_responses_tok_oai));
    try ctx_http.post("/v1/messages/count_tokens", try exw(a, routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    try ctx_http.get("/lora-adapters", try exw(a, routes.get_lora_adapters));
    try ctx_http.post("/lora-adapters", try exw(a, routes.post_lora_adapters));
    // Save & load slots
    try ctx_http.get("/slots", try exw(a, routes.get_slots));
    try ctx_http.post("/slots/:id_slot", try exw(a, routes.post_slots));

    // Google Cloud Platform (Vertex AI) compat
    try ctx_http.registerGcpCompat();
}

/// Ports `server_register_stream_routes` (server.cpp:222 @cd959bae9):
/// resumable streaming: an instance binds the local session factories, a
/// router binds proxies that resolve the owning child.
fn serverRegisterStreamRoutes(a: std.mem.Allocator, ctx_http: *http.Context, stream_get: http.Handler, streams_lookup: http.Handler, stream_delete: http.Handler) !void {
    try ctx_http.get("/v1/stream", try exw(a, stream_get));
    try ctx_http.post("/v1/streams/lookup", try exw(a, streams_lookup));
    try ctx_http.del("/v1/stream", try exw(a, stream_delete));
}

/// Ports `server_setup_features` (server.cpp:234 @cd959bae9): the features
/// behind a flag, and the warning about the ones that must not face untrusted
/// clients.
///
/// Return: false when one of them fails to start.
fn serverSetupFeatures(a: std.mem.Allocator, ctx_http: *http.Context, params: *const Params, mcp_mgr: *mcp_mod.Mcp, tools: *tools_mod.Tools, role_type: models_mod.RoleType) !bool {
    const func = "server_setup_features";
    // return 403 for disabled features
    const res_403: http.Handler = .{ .ctx = null, .func = res403 };

    if (std.mem.eql(u8, params.cors_origins, "*") and params.api_keys.items.len == 0) {
        srv(.wrn, func, "%s", .{@as([*:0]const u8, "security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)\n")});
    }

    var warn_names: std.ArrayList([]const u8) = .empty;
    if (role_type == .router) {
        try warn_names.append(a, "router mode");
    }

    if (params.server_connect) {
        try warn_names.append(a, "peer-to-peer tunnel (llama-connect, experimental)");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.ui_mcp_proxy) {
        try ctx_http.get("/cors-proxy", try exw(a, cors_proxy.handler_get));
        try ctx_http.post("/cors-proxy", try exw(a, cors_proxy.handler_post));
        try warn_names.append(a, "MCP proxy (experimental)");
    } else {
        try ctx_http.get("/cors-proxy", try exw(a, res_403));
        try ctx_http.post("/cors-proxy", try exw(a, res_403));
    }

    mcp_mgr.start(params.mcp_servers_config, params.mcp_servers_json) catch |e| {
        const wz = common.dupZ(context.whatOf(e));
        defer common.freeZ(wz);
        srv(.err, func, "MCP starting failed: %s\n", .{wz});
        return false;
    };

    if (params.server_tools.items.len != 0 or !mcp_mgr.empty()) {
        tools.setup(params.server_tools.items, mcp_mgr, params.server_tools_runtime) catch |e| {
            const wz = common.dupZ(context.whatOf(e));
            defer common.freeZ(wz);
            srv(.err, func, "tools setup failed: %s\n", .{wz});
            return false;
        };
        try ctx_http.get("/tools", try exw(a, tools.handleGet()));
        try ctx_http.post("/tools", try exw(a, tools.handlePost()));
        if (params.server_tools.items.len != 0) try warn_names.append(a, "server tools (experimental)");
        if (params.server_tools_runtime.len != 0) try warn_names.append(a, "tools runtime (experimental)");
        if (!mcp_mgr.empty()) try warn_names.append(a, "MCP servers (experimental)");
    } else {
        try ctx_http.get("/tools", try exw(a, res_403));
        try ctx_http.post("/tools", try exw(a, res_403));
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

    return true;
}

/// Ports `server_install_signal_handlers` (server.cpp:318 @cd959bae9), its
/// POSIX arm.
fn serverInstallSignalHandlers() void {
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

/// Ports `server_on_listening` (server.cpp:336 @cd959bae9): once the HTTP
/// server listens: log the addresses and spawn llama-connect if asked, so the
/// port is bound and the child startup health check reaches us. The
/// destructor of `connect_proc` stops it.
fn serverOnListening(a: std.mem.Allocator, ctx_http: *const http.Context, params: *const Params, connect_proc: *connect.ServerConnect) !bool {
    const func = "server_on_listening";
    for (ctx_http.listening_addresses.items) |address| {
        const z = try a.dupeSentinel(u8, address, 0);
        srv(.inf, func, "listening on %s\n", .{z.ptr});
    }

    if (params.server_connect and !try connect_proc.start(a, params, ctx_http.port)) {
        srv(.err, func, "%s", .{@as([*:0]const u8, "exiting due to llama-connect error\n")});
        return false;
    }
    return true;
}

//
// download worker: a child of the router that pulls a model into the cache and reports the progress
//

/// Ports `server_download_state` (server.cpp:352 @cd959bae9): a download
/// child's progress, reported to the router.
const DownloadState = struct {
    role: *models_mod.Role,
    cancelled: *std.atomic.Value(bool),
    last_progress_time: std.atomic.Value(i64) = .init(0), // multiple files downloading in different threads
    is_ok: bool = false,

    fn callback(dl: *DownloadState) download.Callback {
        return .{ .ctx = dl, .on_start = onStart, .on_update = onUpdate, .on_done = onDone, .is_cancelled = isCancelled };
    }

    /// Ports `run` (server.cpp:360 @cd959bae9), a member of
    /// `server_download_state`.
    fn run(dl: *DownloadState, params: *Params) bool {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cb = dl.callback();
        blk: {
            var handler = download.modelsHandlerInit(a, params, .server) catch |e| {
                dl.fail(params, e);
                break :blk;
            };
            download.modelsHandlerApply(a, &handler, params, &cb) catch |e| {
                dl.fail(params, e);
                break :blk;
            };
            dl.is_ok = true;
        }
        return dl.is_ok;
    }

    fn fail(dl: *DownloadState, params: *const Params, e: anyerror) void {
        const model_name = common.dupZ(params.model.getName());
        defer common.freeZ(model_name);
        const wz = common.dupZ(context.whatOf(e));
        defer common.freeZ(wz);
        srv(.err, "run", "download failed for model name=%s: %s\n", .{ model_name, wz });
        dl.is_ok = false;
    }

    /// Ports `on_progress` (server.cpp:372 @cd959bae9), a member of
    /// `server_download_state`.
    fn onProgress(dl: *DownloadState, p: *const download.Progress) void {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var o = common.Obj.init(arena.allocator());
        o.put("url", p.url) catch return;
        o.put("downloaded", p.downloaded) catch return;
        o.put("total", p.total) catch return;
        const v = o.done();
        dl.role.notifyToRouter(context.serverStateToStr(.downloading), &v);
    }

    fn onStart(ctx: *anyopaque, p: *const download.Progress) void {
        const dl: *DownloadState = @ptrCast(@alignCast(ctx));
        dl.onProgress(p);
    }

    fn onUpdate(ctx: *anyopaque, p: *const download.Progress) void {
        const dl: *DownloadState = @ptrCast(@alignCast(ctx));
        const now = c.ggml_time_ms();
        // throttle progress updates to avoid flooding logs
        if (now - dl.last_progress_time.load(.monotonic) >= 100) {
            dl.onProgress(p);
            dl.last_progress_time.store(now, .monotonic);
        }
    }

    fn onDone(ctx: *anyopaque, p: *const download.Progress, _: bool) void {
        const dl: *DownloadState = @ptrCast(@alignCast(ctx));
        dl.onProgress(p);
    }

    fn isCancelled(ctx: *anyopaque) bool {
        const dl: *DownloadState = @ptrCast(@alignCast(ctx));
        return dl.cancelled.load(.monotonic);
    }
};

/// Ports `server_run_download` (server.cpp:399 @cd959bae9): download the
/// model, report the result, and wait for the router's exit command.
fn serverRunDownload(params: *Params, role: *models_mod.Role) !u8 {
    const cancelled = try gpa.create(std.atomic.Value(bool));
    cancelled.* = .init(false);

    // monitor stdin for cancellation command from the router
    const signal_thread = role.setup(.{ .ctx = cancelled, .func = struct {
        fn f(ctx: ?*anyopaque, _: c_int) void {
            const flag: *std.atomic.Value(bool) = @ptrCast(@alignCast(ctx.?));
            flag.store(true, .monotonic);
        }
    }.f });

    var dl: DownloadState = .{ .role = role, .cancelled = cancelled };

    const ok = dl.run(params);

    {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var o = common.Obj.init(arena.allocator());
        try o.put("result", if (ok) "download_finished" else "download_failed");
        const v = o.done();
        role.notifyToRouter(context.serverStateToStr(.downloading), &v);
    }

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread) |t| {
        t.join();
    }

    srv(.inf, "server_run_download", "download completed %s\n", .{@as([*:0]const u8, if (ok) "successfully" else "with errors")});
    return 0;
}

//
// measure worker: a child of the router, reports the memory a model needs without loading it
//

/// Ports `server_run_measure` (server.cpp:431 @cd959bae9).
fn serverRunMeasure(a: std.mem.Allocator, params: *Params, role: *models_mod.Role, models_handler: *download.ModelsHandler) !u8 {
    const func = "server_run_measure";
    // the router sets --offline: only resolve the local path, never download
    download.modelsHandlerApply(a, models_handler, params, null) catch |e| {
        srvErrWhat(func, "failed to resolve model: %s\n", e);
        return 1;
    };

    const model_name = try a.dupeSentinel(u8, params.model.getName(), 0);

    // TODO: mmproj is not in llama_get_memory_breakdown, a multimodal model is underestimated
    var o = common.Obj.init(a);
    var what_out: []const u8 = "";
    if (tc.init.measureMemory(gpa, params, &what_out)) |m| {
        const min = try models_sched.ServerModelMemory.fromRows(a, m.n_ctx_min, m.info_min);
        const max = try models_sched.ServerModelMemory.fromRows(a, m.n_ctx_max, m.info_max);
        try o.put("result", "measured");
        try o.put("n_ctx_train", m.n_ctx_train);
        try o.put("min", try min.toJson(a));
        try o.put("max", try max.toJson(a));
        srv(.inf, func, "memory measured for model name=%s at n_ctx=%u (min) and n_ctx=%u (max)\n", .{ model_name.ptr, m.n_ctx_min, m.n_ctx_max });
    } else |e| {
        if (e == error.OutOfMemory) return e;
        const msg = if (what_out.len != 0) what_out else context.whatOf(e);
        srv(.err, func, "failed to measure memory for model name=%s: %s\n", .{ model_name.ptr, (try a.dupeSentinel(u8, msg, 0)).ptr });
        try o.put("result", "failed");
        try o.put("error", msg);
    }

    const v = o.done();
    role.notifyToRouter(context.serverStateToStr(.measuring), &v);
    return 0;
}

//
// router
//

/// The instance's `clean_up` lambda (server.cpp:648 @cd959bae9).
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

/// The router's `clean_up` lambda (server.cpp:545 @cd959bae9).
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

/// Ports `server_run_router` (server.cpp:470 @cd959bae9).
fn serverRunRouter(a: std.mem.Allocator, params: *Params, argv: []const [:0]const u8, opts: RunOpts) !u8 {
    const func = "server_run_router";
    srv(.inf, func, "%s", .{@as([*:0]const u8, "starting server in router mode. models will be automatically loaded on-demand\n")});

    // note: this is guaranteed to out-live ctx_http and tools
    var mcp_mgr: mcp_mod.Mcp = .{};
    defer mcp_mgr.deinit();
    shutdown_mcp = &mcp_mgr;

    // the router loads no model, its context stays empty; the routes it does not proxy still need one
    var ctx_server = try context.ServerContext.init();
    defer ctx_server.deinit();
    var routes: routes_mod.Routes = undefined;
    try routes.init(params, &ctx_server);
    defer routes.deinit();
    var tools = try tools_mod.Tools.init();
    defer tools.deinit();

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

    // setup server instances manager
    const mr = try gpa.create(models_mod.ModelsRoutes);
    mr.init(params, argv) catch |e| {
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
    routes.post_tts = mr.proxy_post;
    routes.ws_realtime = mr.proxy_ws;
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

    try ctx_http.post("/models", try exw(a, mr.post_router_models));
    try ctx_http.post("/models/load", try exw(a, mr.post_router_models_load));
    try ctx_http.post("/models/unload", try exw(a, mr.post_router_models_unload));
    try ctx_http.get("/models/sse", try exw(a, mr.get_router_models_sse));
    try ctx_http.del("/models", try exw(a, mr.del_router_models));

    try serverRegisterRoutes(a, &ctx_http, &routes);
    try serverRegisterStreamRoutes(a, &ctx_http, mr.router_stream_get, mr.router_streams_lookup, mr.router_stream_delete);

    if (!try serverSetupFeatures(a, &ctx_http, params, &mcp_mgr, &tools, .router)) {
        return 1;
    }

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

    if (opts.install_signal_handlers) {
        serverInstallSignalHandlers();
    }

    var connect_proc: connect.ServerConnect = .{};
    defer connect_proc.deinit();
    if (!try serverOnListening(a, &ctx_http, params, &connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        RouterCleanUp.run(mr);
        return 1;
    }

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

//
// instance
//

/// Ports `server_run_instance` (server.cpp:612 @cd959bae9).
fn serverRunInstance(a: std.mem.Allocator, params: *Params, role: *models_mod.Role, models_handler: *download.ModelsHandler, opts: RunOpts) !u8 {
    const func = "server_run_instance";
    // -hf / docker: download if needed and resolve the local path. a child resolves what the router downloaded
    if (opts.resolve_model) {
        download.modelsHandlerApply(a, models_handler, params, null) catch |e| {
            srvErrWhat(func, "failed to download model: %s\n", e);
            return 1;
        };
    }

    // note: this is guaranteed to out-live ctx_http and tools
    var mcp_mgr: mcp_mod.Mcp = .{};
    defer mcp_mgr.deinit();
    shutdown_mcp = &mcp_mgr;

    // struct that contains llama context and inference
    var ctx_server = try context.ServerContext.init();
    defer ctx_server.deinit();
    var routes: routes_mod.Routes = undefined;
    try routes.init(params, &ctx_server);
    defer routes.deinit();
    var tools = try tools_mod.Tools.init();
    defer tools.deinit();

    // declared after everything its listeners' handlers reach, so it is
    // stopped and joined before they are destroyed
    var http_params = httpParams(params);
    var ctx_http = try http.Context.init(gpa);
    defer ctx_http.deinit();
    if (!try ctx_http.setup(&http_params)) {
        srv(.err, func, "%s", .{@as([*:0]const u8, "failed to initialize HTTP server\n")});
        return 1;
    }

    try serverRegisterRoutes(a, &ctx_http, &routes);
    try serverRegisterStreamRoutes(a, &ctx_http, stream.makeGetHandler(), stream.makeLookupHandler(), stream.makeDeleteHandler());

    if (!try serverSetupFeatures(a, &ctx_http, params, &mcp_mgr, &tools, .instance)) {
        return 1;
    }

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!try ctx_http.start()) {
        CleanUp.run(&ctx_http, &ctx_server);
        srv(.err, func, "%s", .{@as([*:0]const u8, "exiting due to HTTP server error\n")});
        return 1;
    }

    // setup communication child --> router if necessary
    if (role.isManagedByRouter()) {
        ctx_server.setStateCallback(role.stateCallback());
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

    if (opts.install_signal_handlers) {
        serverInstallSignalHandlers();
    }

    var connect_proc: connect.ServerConnect = .{};
    defer connect_proc.deinit();
    if (!try serverOnListening(a, &ctx_http, params, &connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        CleanUp.run(&ctx_http, &ctx_server);
        return 1;
    }

    // optionally, notify router server that this instance is ready
    var monitor_thread: ?std.Thread = null;
    if (role.isManagedByRouter()) {
        monitor_thread = role.setup(.{ .ctx = null, .func = struct {
            fn f(_: ?*anyopaque, sig: c_int) void {
                if (shutdown_handler) |h| h(sig);
            }
        }.f });
        var info_arena = std.heap.ArenaAllocator.init(gpa);
        defer info_arena.deinit();
        const ia = info_arena.allocator();
        var info = try routes.getModelInfo(ia);
        // the router tracks device memory from these reports
        if (ctx_server.getLlamaContext()) |ll_ctx| {
            const rows = try fit.memoryBreakdownRows(ia, @ptrCast(ll_ctx));
            const mem = try models_sched.ServerModelMemory.fromRows(ia, c.llama_n_ctx(ll_ctx), rows);
            try info.set(ia, "memory", try mem.toJson(ia));
        }
        role.notifyToRouter(context.serverStateToStr(.ready), &info);
    }

    // this call blocks the main thread until queue_tasks.terminate() is called
    try ctx_server.startLoop();

    CleanUp.run(&ctx_http, &ctx_server);
    ctx_http.join();
    if (monitor_thread) |t| {
        t.join();
    }

    if (ctx_server.getLlamaContext()) |ll_ctx| {
        try fit.memoryBreakdownPrint(gpa, .{ .verbosity = std.math.maxInt(i32), .sink = tc.common.fitLogSink, .json = tc.common.fitLogJson }, @ptrCast(ll_ctx));
    }

    return 0;
}

//
// entry points
//

/// Ports `server_run` (server.cpp:735 @cd959bae9).
fn serverRun(a: std.mem.Allocator, params: *Params, argv: []const [:0]const u8, role: *models_mod.Role, opts: RunOpts) !u8 {
    const func = "server_run";
    var models_handler: download.ModelsHandler = .{};

    // -hf may name a preset repo, which makes this a router: settle that before deciding the role
    if (opts.resolve_model and params.model.hf_repo.len != 0) {
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
    }

    const role_type = serverRoleDetect(params, role);

    // only an instance loads a model; the others must not touch the GPU, so skip the
    // device enumeration for them and the CUDA primary context stays uncreated
    try tc.common.paramsPrintInfo(a, params, role_type == .instance);

    if (!try serverPrepareParams(a, params, role_type)) {
        return 1;
    }

    return switch (role_type) {
        .download => serverRunDownload(params, role),
        .measure => serverRunMeasure(a, params, role, &models_handler),
        .router => serverRunRouter(a, params, argv, opts),
        .instance => serverRunInstance(a, params, role, &models_handler, opts),
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
