//! http-diff's port side: `tools/server/http.zig`'s `Context` serving the
//! same routes `harness/http_ref.cpp` gives upstream's
//! `server_http_context`, with the same configurations and the same log
//! lines. `scripts/http-diff` drives both with the same raw bytes and
//! compares what comes back.
//!
//! **Not a port.** Test scaffolding; it mirrors `harness/http_ref.cpp`
//! line for line, and `exWrapper` mirrors upstream's `ex_wrapper`
//! (tools/server/server.cpp:54 @7fe450e19), which the server's port will
//! own.
//!
//! Usage: http_port <config> <port-file>

const std = @import("std");
const cli_fit = @import("cli_fit");
const stdlib = @import("stdlib");
const http = @import("http");
const tools_common = @import("tools_common");

var g_quit: std.atomic.Value(bool) = .init(false);
var g_wait_complete: std.atomic.Value(bool) = .init(false);

fn sleepMs(ms: u32) void {
    std.Io.sleep(std.Io.Threaded.global_single_threaded.io(), .fromMilliseconds(ms), .awake) catch {};
}

// -----------------------------------------------------------------------------
// ex_wrapper

const ExWrapper = struct {
    inner: http.Handler,

    fn call(ctx: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
        const self: *ExWrapper = @ptrCast(@alignCast(ctx.?));
        return self.inner.call(req) catch |err| {
            var ty: http.ErrorType = .server;
            var message: []const u8 = "unknown error";
            if (err == error.InvalidArgument) {
                ty = .invalid_request;
                message = http.what(err).?;
            } else if (http.what(err)) |w| {
                message = w;
            }
            const message_copy = try std.heap.c_allocator.dupe(u8, message);
            defer std.heap.c_allocator.free(message_copy);
            const res = try http.Res.create(std.heap.c_allocator);
            const a = res.allocator();
            res.status = 500;
            var error_data = try http.formatErrorResponse(a, message_copy, ty);
            res.status = @intCast(try (try error_data.at("code")).get(i64));
            var outer = cli_fit.json.Value.emptyObject();
            try outer.set(a, "error", error_data);
            try res.setData(try http.safeJsonToStr(a, &outer));
            const z = try a.dupeSentinel(u8, res.data.items, 0);
            tools_common.log.wrn("srv  %12.*s: got exception: %s\n", .{ @as(c_int, 12), @as([*:0]const u8, "operator()"), z.ptr });
            return res;
        };
    }
};

fn exWrapper(inner: http.Handler) http.Handler {
    const w = std.heap.c_allocator.create(ExWrapper) catch @panic("OOM");
    w.* = .{ .inner = inner };
    return .{ .ctx = w, .func = ExWrapper.call };
}

fn handler(comptime f: fn (req: *const http.Req) anyerror!*http.Res) http.Handler {
    return .{ .ctx = null, .func = struct {
        fn call(_: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
            return f(req);
        }
    }.call };
}

// -----------------------------------------------------------------------------
// Routes

fn jsonRes(v: *cli_fit.json.Value, va: std.mem.Allocator) !*http.Res {
    defer v.deinit(va);
    const res = try http.Res.create(std.heap.c_allocator);
    try res.setData(try http.safeJsonToStr(res.allocator(), v));
    return res;
}

fn str(a: std.mem.Allocator, s: []const u8) !cli_fit.json.Value {
    return cli_fit.json.Value.initString(a, s);
}

fn echoJson(a: std.mem.Allocator, tag: []const u8, req: *const http.Req) !cli_fit.json.Value {
    var j = cli_fit.json.Value.emptyObject();
    try j.set(a, "tag", try str(a, tag));
    try j.set(a, "path", try str(a, req.path));
    try j.set(a, "query_string", try str(a, req.query_string));
    var p = cli_fit.json.Value.emptyObject();
    for (req.params.entries.items) |e| try p.set(a, e.key, try str(a, e.value));
    try j.set(a, "params", p);
    var h = cli_fit.json.Value.emptyObject();
    for (req.headers.entries.items) |e| try h.set(a, e.key, try str(a, e.value));
    try j.set(a, "headers", h);
    try j.set(a, "body", try str(a, req.body));
    var f = cli_fit.json.Value.emptyObject();
    for (req.files.entries.items) |e| {
        var o = cli_fit.json.Value.emptyObject();
        try o.set(a, "filename", try str(a, e.value.filename));
        try o.set(a, "content_type", try str(a, e.value.content_type));
        try o.set(a, "size", try cli_fit.json.Value.from(a, @as(i64, @intCast(e.value.data.len))));
        try o.set(a, "data", try str(a, e.value.data));
        try f.set(a, e.key, o);
    }
    try j.set(a, "files", f);
    return j;
}

fn echo(comptime tag: []const u8) http.Handler {
    return handler(struct {
        fn f(req: *const http.Req) anyerror!*http.Res {
            var j = try echoJson(std.heap.c_allocator, tag, req);
            return jsonRes(&j, std.heap.c_allocator);
        }
    }.f);
}

/// The websocket route of `harness/http_ref.cpp`: the same greeting, the
/// same echo and the same log lines.
fn wsRoute(_: ?*anyopaque, req: *const http.Req, conn: *http.WsConn) void {
    wsRouteImpl(req, conn) catch {};
}

fn wsRouteImpl(req: *const http.Req, conn: *http.WsConn) !void {
    {
        var j = try echoJson(std.heap.c_allocator, "ws", req);
        defer j.deinit(std.heap.c_allocator);
        try j.set(std.heap.c_allocator, "stop", try cli_fit.json.Value.from(std.heap.c_allocator, req.should_stop.call()));
        const s = try http.safeJsonToStr(std.heap.c_allocator, &j);
        defer std.heap.c_allocator.free(s);
        _ = conn.send(s);
    }
    var msg: std.ArrayList(u8) = .empty;
    defer msg.deinit(std.heap.c_allocator);
    while (true) {
        const ty = try conn.read(std.heap.c_allocator, &msg);
        if (ty == 0) {
            tools_common.log.inf("harness: ws closed, open=%d stop=%d\n", .{ @as(c_int, @intFromBool(conn.isOpen())), @as(c_int, @intFromBool(req.should_stop.call())) });
            return;
        }
        tools_common.log.inf("harness: ws read type=%d len=%zu\n", .{ ty, msg.items.len });
        if (std.mem.eql(u8, msg.items, "close")) {
            conn.close();
            continue;
        }
        if (std.mem.eql(u8, msg.items, "open")) {
            _ = conn.send(if (conn.isOpen()) "open=1" else "open=0");
            continue;
        }
        if (std.mem.startsWith(u8, msg.items, "binary:")) {
            _ = conn.sendBinary(msg.items[7..]);
            continue;
        }
        if (ty == 1) {
            _ = conn.send(msg.items);
        } else {
            _ = conn.sendBinary(msg.items);
        }
    }
}

fn health(_: *const http.Req) anyerror!*http.Res {
    var j = cli_fit.json.Value.emptyObject();
    try j.set(std.heap.c_allocator, "status", try str(std.heap.c_allocator, "ok"));
    return jsonRes(&j, std.heap.c_allocator);
}

fn empty(_: *const http.Req) anyerror!*http.Res {
    return http.Res.create(std.heap.c_allocator);
}

/// `std::stoi`, its two exceptions told apart as `ex_wrapper` tells them.
fn stoi(s: []const u8) !c_int {
    const z = try std.heap.c_allocator.dupeSentinel(u8, s, 0);
    defer std.heap.c_allocator.free(z);
    return tools_common.arg.stoi(z) catch {
        const msg = tools_common.arg.what();
        if (std.mem.eql(u8, msg, "stoi: no conversion")) return http.throwInvalidArgument(msg);
        return http.throwRuntimeError(msg);
    };
}

fn status(req: *const http.Req) anyerror!*http.Res {
    const code = try stoi(req.getParam("code", "200"));
    const res = try http.Res.create(std.heap.c_allocator);
    errdefer res.destroy();
    res.status = code;
    try res.setData(req.getParam("body", ""));
    if (req.params.get("ct")) |ct| res.content_type = try res.allocator().dupe(u8, ct.*);
    for (req.params.entries.items) |e| {
        if (std.mem.startsWith(u8, e.key, "h_")) try res.setHeader(e.key[2..], e.value);
    }
    return res;
}

const StreamState = struct {
    step: i32 = 0,
    stop: http.ShouldStop,
};

fn stream(req: *const http.Req) anyerror!*http.Res {
    const res = try http.Res.create(std.heap.c_allocator);
    res.content_type = "text/event-stream";
    try res.setHeader("Cache-Control", "no-cache");
    const st = try res.allocator().create(StreamState);
    st.* = .{ .stop = req.should_stop };
    res.next = .{ .ctx = st, .func = struct {
        fn next(ctx: ?*anyopaque, out: *std.ArrayList(u8), a: std.mem.Allocator) anyerror!bool {
            const s: *StreamState = @ptrCast(@alignCast(ctx.?));
            const step = s.step;
            s.step += 1;
            switch (step) {
                0 => {
                    try out.appendSlice(a, "data: a\n\n");
                    return true;
                },
                1 => return true,
                2 => {
                    try out.appendSlice(a, "data: bc\n\n");
                    return true;
                },
                3 => return true,
                else => {
                    try out.appendSlice(a, if (s.stop.call()) "data: stop=1\n\n" else "data: stop=0\n\n");
                    return false;
                },
            }
        }
    }.next };
    return res;
}

const wait_vtable: http.Res.VTable = .{ .on_complete = waitComplete, .destroy = http.Res.destroyBase };

fn waitComplete(_: *http.Res) void {
    tools_common.log.inf("harness: stream-wait complete\n", .{});
    g_wait_complete.store(true, .seq_cst);
}

fn streamWait(req: *const http.Req) anyerror!*http.Res {
    const res = try http.Res.create(std.heap.c_allocator);
    res.vtable = &wait_vtable;
    res.content_type = "text/plain";
    const st = try res.allocator().create(StreamState);
    st.* = .{ .stop = req.should_stop };
    res.next = .{ .ctx = st, .func = struct {
        fn next(ctx: ?*anyopaque, out: *std.ArrayList(u8), a: std.mem.Allocator) anyerror!bool {
            const s: *StreamState = @ptrCast(@alignCast(ctx.?));
            const step = s.step;
            s.step += 1;
            if (step == 0) {
                try out.appendSlice(a, "first\n");
                return true;
            }
            for (0..300) |_| {
                if (s.stop.call()) {
                    tools_common.log.inf("harness: should_stop fired\n", .{});
                    return false;
                }
                sleepMs(10);
            }
            tools_common.log.inf("harness: should_stop timed out\n", .{});
            try out.appendSlice(a, "late\n");
            return false;
        }
    }.next };
    return res;
}

fn waitCompleteRoute(_: *const http.Req) anyerror!*http.Res {
    var complete = false;
    var i: usize = 0;
    while (i < 300 and !complete) : (i += 1) {
        complete = g_wait_complete.swap(false, .seq_cst);
        if (!complete) sleepMs(10);
    }
    var j = cli_fit.json.Value.emptyObject();
    try j.set(std.heap.c_allocator, "complete", try cli_fit.json.Value.from(std.heap.c_allocator, complete));
    return jsonRes(&j, std.heap.c_allocator);
}

fn throwInvalid(req: *const http.Req) anyerror!*http.Res {
    const msg = try std.mem.concat(std.heap.c_allocator, u8, &.{ "bad input: ", req.body });
    defer std.heap.c_allocator.free(msg);
    return http.throwInvalidArgument(msg);
}

fn throwRuntime(_: *const http.Req) anyerror!*http.Res {
    return http.throwRuntimeError("boom");
}

fn throwUnknown(_: *const http.Req) anyerror!*http.Res {
    // `throw 42`: not a std::exception
    return error.NotAStdException;
}

fn throwRaw(_: *const http.Req) anyerror!*http.Res {
    return http.throwRuntimeError("raw boom");
}

fn quit(_: *const http.Req) anyerror!*http.Res {
    g_quit.store(true, .seq_cst);
    var j = cli_fit.json.Value.emptyObject();
    try j.set(std.heap.c_allocator, "bye", try cli_fit.json.Value.from(std.heap.c_allocator, true));
    return jsonRes(&j, std.heap.c_allocator);
}

// -----------------------------------------------------------------------------
// Configurations

fn configure(name: []const u8, params: *http.Params, ready: *bool) void {
    params.hostnames = &.{"127.0.0.1"};
    params.port = 0;
    params.timeout_read = 3;
    params.timeout_write = 3;
    params.n_threads_http = 2;
    ready.* = true;
    if (std.mem.eql(u8, name, "apikey")) {
        params.api_keys = &.{ "sk-test-1", "k2" };
        params.n_threads_http = 1;
        params.cors_origins = "localhost";
        params.cors_credentials = false;
    } else if (std.mem.eql(u8, name, "prefix")) {
        params.api_prefix = "/api";
        params.api_keys = &.{"abc"};
        params.n_threads_http = 4;
        params.cors_origins = "https://example.com";
        params.cors_credentials = false;
        params.cors_methods = "GET";
        params.cors_headers = "Authorization";
    } else if (std.mem.eql(u8, name, "auto")) {
        params.n_threads_http = -1;
        params.n_parallel = 3;
        params.ui = false;
    } else if (std.mem.eql(u8, name, "notready")) {
        ready.* = false;
    } else if (std.mem.eql(u8, name, "unix")) {
        params.hostnames = std.heap.c_allocator.dupe([]const u8, &.{sockPath()}) catch unreachable;
    } else if (std.mem.eql(u8, name, "ssl")) {
        params.ssl_file_key = "key.pem";
        params.ssl_file_cert = "cert.pem";
    } else if (std.mem.eql(u8, name, "badhost")) {
        params.hostnames = &.{"256.1.1.1"};
    } else if (std.mem.eql(u8, name, "reuse")) {
        params.reuse_port = true;
        params.api_keys = &.{"abc"};
    } else if (std.mem.eql(u8, name, "multi")) {
        // A listener per address, a Unix socket first and one TCP
        // address, so port 0 is allowed
        params.hostnames = std.heap.c_allocator.dupe([]const u8, &.{ sockPath(), "127.0.0.1" }) catch unreachable;
        params.api_keys = &.{"abc"};
    } else if (std.mem.eql(u8, name, "multitcp")) {
        // two TCP addresses on one fixed port, each IPv6-only where it can be
        params.hostnames = &.{ "127.0.0.1", "::1" };
        params.port = envPort();
    } else if (std.mem.eql(u8, name, "multiport0")) {
        // port 0 with two TCP addresses is refused
        params.hostnames = &.{ "127.0.0.1", "::1" };
    } else if (std.mem.eql(u8, name, "corslocal") or std.mem.eql(u8, name, "child")) {
        // Same-origin only, and the DNS rebinding guard on Host;
        // "child" also has LLAMA_SERVER_ROUTER_PORT set by the driver
        params.cors_origins = "localhost";
        params.cors_credentials = false;
    } else if (std.mem.eql(u8, name, "corslist")) {
        params.cors_origins = "http://good.example:8080, HTTPS://Other.Example ,bad";
        params.cors_credentials = false;
    } else if (std.mem.eql(u8, name, "lan")) {
        // every address, so a request can reach a non-loopback one
        params.hostnames = &.{"0.0.0.0"};
        params.port = envPort();
        params.cors_origins = "localhost";
        params.cors_credentials = false;
    } else if (std.mem.eql(u8, name, "multibad")) {
        // the second address cannot be bound: the first is stopped again
        params.hostnames = &.{ "127.0.0.1", "256.1.1.1" };
        params.port = envPort();
    }
}

fn sockPath() []const u8 {
    return if (std.c.getenv("HTTP_DIFF_SOCK")) |s| std.mem.span(s) else "http-diff.sock";
}

/// `std::atoi(getenv("HTTP_DIFF_PORT"))`, 0 when unset.
fn envPort() i32 {
    const s = std.c.getenv("HTTP_DIFF_PORT") orelse return 0;
    return std.fmt.parseInt(i32, std.mem.span(s), 10) catch 0;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("usage: http_port <config> <port-file>\n", .{});
        return 2;
    }
    const config = args[1];
    tools_common.log.verbosity_thold = tools_common.log.level_debug;

    var params: http.Params = .{};
    var ready = true;
    configure(config, &params, &ready);

    var ctx_http = try http.Context.init(std.heap.c_allocator);
    if (!try ctx_http.setup(&params)) {
        tools_common.log.inf("harness: init failed\n", .{});
        return 1;
    }

    try ctx_http.get("/health", exWrapper(handler(health)));
    try ctx_http.get("/echo", exWrapper(echo("get")));
    try ctx_http.post("/echo", exWrapper(echo("post")));
    try ctx_http.del("/echo", exWrapper(echo("delete")));
    try ctx_http.post("/slots/:id_slot", exWrapper(echo("slots")));
    try ctx_http.get("/two/:a/x/:b", exWrapper(echo("two")));
    try ctx_http.get("/empty", exWrapper(handler(empty)));
    try ctx_http.get("/status", exWrapper(handler(status)));
    try ctx_http.get("/stream", exWrapper(handler(stream)));
    try ctx_http.get("/stream-wait", exWrapper(handler(streamWait)));
    try ctx_http.get("/wait-complete", exWrapper(handler(waitCompleteRoute)));
    try ctx_http.post("/throw-invalid", exWrapper(handler(throwInvalid)));
    try ctx_http.post("/throw-runtime", exWrapper(handler(throwRuntime)));
    try ctx_http.get("/throw-unknown", exWrapper(handler(throwUnknown)));
    // not wrapped: the error reaches cpp-httplib's exception handler
    try ctx_http.get("/throw-raw", handler(throwRaw));
    try ctx_http.get("/quit", exWrapper(handler(quit)));

    try ctx_http.ws("/ws/:name", .{ .ctx = null, .func = wsRoute });

    try ctx_http.registerGcpCompat();

    if (!try ctx_http.start()) {
        tools_common.log.inf("harness: start failed\n", .{});
        return 1;
    }
    if (ready) ctx_http.is_ready.store(true, .seq_cst);
    for (ctx_http.listening_addresses.items) |address| {
        const addr = try std.heap.c_allocator.dupeSentinel(u8, address, 0);
        tools_common.log.inf("harness: listening on %s\n", .{addr.ptr});
    }

    {
        const port_str = try std.fmt.allocPrint(std.heap.c_allocator, "{d}\n", .{ctx_http.port});
        const f = std.c.fopen(args[2], "w") orelse return 1;
        _ = std.c.fwrite(port_str.ptr, 1, port_str.len, f);
        _ = std.c.fclose(f);
    }

    // GET /quit, or <port-file>.quit appearing: a server that is not ready
    // answers /quit with 503
    const quit_file = try std.mem.concat(std.heap.c_allocator, u8, &.{ args[2], ".quit\x00" });
    while (!g_quit.load(.seq_cst)) {
        if (std.c.fopen(@ptrCast(quit_file.ptr), "r")) |q| {
            _ = std.c.fclose(q);
            break;
        }
        sleepMs(5);
    }
    ctx_http.stop();
    ctx_http.join();
    tools_common.log.inf("harness: stopped\n", .{});
    _ = stdlib.libc.fflush(null);
    return 0;
}
