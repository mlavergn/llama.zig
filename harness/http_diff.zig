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
const http = @import("http");
const tools_common = @import("tools_common");
const json = @import("cli_fit").json;

const log = tools_common.log;
const Req = http.Req;
const Res = http.Res;
const Handler = http.Handler;

const gpa = std.heap.c_allocator;

var g_quit: std.atomic.Value(bool) = .init(false);
var g_wait_complete: std.atomic.Value(bool) = .init(false);

fn sleepMs(ms: u32) void {
    std.Io.sleep(std.Io.Threaded.global_single_threaded.io(), .fromMilliseconds(ms), .awake) catch {};
}

// -----------------------------------------------------------------------------
// ex_wrapper

const ExWrapper = struct {
    inner: Handler,

    fn call(ctx: ?*anyopaque, req: *const Req) anyerror!*Res {
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
            const message_copy = try gpa.dupe(u8, message);
            defer gpa.free(message_copy);
            const res = try Res.create(gpa);
            const a = res.allocator();
            res.status = 500;
            var error_data = try http.formatErrorResponse(a, message_copy, ty);
            res.status = @intCast(try (try error_data.at("code")).get(i64));
            var outer = json.Value.emptyObject();
            try outer.set(a, "error", error_data);
            try res.setData(try http.safeJsonToStr(a, &outer));
            const z = try a.dupeSentinel(u8, res.data.items, 0);
            log.wrn("srv  %12.*s: got exception: %s\n", .{ @as(c_int, 12), @as([*:0]const u8, "operator()"), z.ptr });
            return res;
        };
    }
};

fn exWrapper(inner: Handler) Handler {
    const w = gpa.create(ExWrapper) catch @panic("OOM");
    w.* = .{ .inner = inner };
    return .{ .ctx = w, .func = ExWrapper.call };
}

fn handler(comptime f: fn (req: *const Req) anyerror!*Res) Handler {
    return .{ .ctx = null, .func = struct {
        fn call(_: ?*anyopaque, req: *const Req) anyerror!*Res {
            return f(req);
        }
    }.call };
}

// -----------------------------------------------------------------------------
// Routes

fn jsonRes(v: *json.Value, va: std.mem.Allocator) !*Res {
    defer v.deinit(va);
    const res = try Res.create(gpa);
    try res.setData(try http.safeJsonToStr(res.allocator(), v));
    return res;
}

fn str(a: std.mem.Allocator, s: []const u8) !json.Value {
    return json.Value.initString(a, s);
}

fn echoJson(a: std.mem.Allocator, tag: []const u8, req: *const Req) !json.Value {
    var j = json.Value.emptyObject();
    try j.set(a, "tag", try str(a, tag));
    try j.set(a, "path", try str(a, req.path));
    try j.set(a, "query_string", try str(a, req.query_string));
    var p = json.Value.emptyObject();
    for (req.params.entries.items) |e| try p.set(a, e.key, try str(a, e.value));
    try j.set(a, "params", p);
    var h = json.Value.emptyObject();
    for (req.headers.entries.items) |e| try h.set(a, e.key, try str(a, e.value));
    try j.set(a, "headers", h);
    try j.set(a, "body", try str(a, req.body));
    var f = json.Value.emptyObject();
    for (req.files.entries.items) |e| {
        var o = json.Value.emptyObject();
        try o.set(a, "filename", try str(a, e.value.filename));
        try o.set(a, "content_type", try str(a, e.value.content_type));
        try o.set(a, "size", try json.Value.from(a, @as(i64, @intCast(e.value.data.len))));
        try o.set(a, "data", try str(a, e.value.data));
        try f.set(a, e.key, o);
    }
    try j.set(a, "files", f);
    return j;
}

fn echo(comptime tag: []const u8) Handler {
    return handler(struct {
        fn f(req: *const Req) anyerror!*Res {
            var j = try echoJson(gpa, tag, req);
            return jsonRes(&j, gpa);
        }
    }.f);
}

fn health(_: *const Req) anyerror!*Res {
    var j = json.Value.emptyObject();
    try j.set(gpa, "status", try str(gpa, "ok"));
    return jsonRes(&j, gpa);
}

fn empty(_: *const Req) anyerror!*Res {
    return Res.create(gpa);
}

/// `std::stoi`, its two exceptions told apart as `ex_wrapper` tells them.
fn stoi(s: []const u8) !c_int {
    const z = try gpa.dupeSentinel(u8, s, 0);
    defer gpa.free(z);
    return tools_common.arg.stoi(z) catch {
        const msg = tools_common.arg.what();
        if (std.mem.eql(u8, msg, "stoi: no conversion")) return http.throwInvalidArgument(msg);
        return http.throwRuntimeError(msg);
    };
}

fn status(req: *const Req) anyerror!*Res {
    const code = try stoi(req.getParam("code", "200"));
    const res = try Res.create(gpa);
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

fn stream(req: *const Req) anyerror!*Res {
    const res = try Res.create(gpa);
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

const wait_vtable: Res.VTable = .{ .on_complete = waitComplete, .destroy = Res.destroyBase };

fn waitComplete(_: *Res) void {
    log.inf("harness: stream-wait complete\n", .{});
    g_wait_complete.store(true, .seq_cst);
}

fn streamWait(req: *const Req) anyerror!*Res {
    const res = try Res.create(gpa);
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
                    log.inf("harness: should_stop fired\n", .{});
                    return false;
                }
                sleepMs(10);
            }
            log.inf("harness: should_stop timed out\n", .{});
            try out.appendSlice(a, "late\n");
            return false;
        }
    }.next };
    return res;
}

fn waitCompleteRoute(_: *const Req) anyerror!*Res {
    var complete = false;
    var i: usize = 0;
    while (i < 300 and !complete) : (i += 1) {
        complete = g_wait_complete.swap(false, .seq_cst);
        if (!complete) sleepMs(10);
    }
    var j = json.Value.emptyObject();
    try j.set(gpa, "complete", try json.Value.from(gpa, complete));
    return jsonRes(&j, gpa);
}

fn throwInvalid(req: *const Req) anyerror!*Res {
    const msg = try std.mem.concat(gpa, u8, &.{ "bad input: ", req.body });
    defer gpa.free(msg);
    return http.throwInvalidArgument(msg);
}

fn throwRuntime(_: *const Req) anyerror!*Res {
    return http.throwRuntimeError("boom");
}

fn throwUnknown(_: *const Req) anyerror!*Res {
    // `throw 42`: not a std::exception
    return error.NotAStdException;
}

fn throwRaw(_: *const Req) anyerror!*Res {
    return http.throwRuntimeError("raw boom");
}

fn quit(_: *const Req) anyerror!*Res {
    g_quit.store(true, .seq_cst);
    var j = json.Value.emptyObject();
    try j.set(gpa, "bye", try json.Value.from(gpa, true));
    return jsonRes(&j, gpa);
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
        params.hostnames = gpa.dupe([]const u8, &.{sockPath()}) catch unreachable;
    } else if (std.mem.eql(u8, name, "ssl")) {
        params.ssl_file_key = "key.pem";
        params.ssl_file_cert = "cert.pem";
    } else if (std.mem.eql(u8, name, "badhost")) {
        params.hostnames = &.{"256.1.1.1"};
    } else if (std.mem.eql(u8, name, "reuse")) {
        params.reuse_port = true;
        params.api_keys = &.{"abc"};
    } else if (std.mem.eql(u8, name, "multi")) {
        // v0.5.0: a listener per address, a Unix socket first and one TCP
        // address, so port 0 is allowed
        params.hostnames = gpa.dupe([]const u8, &.{ sockPath(), "127.0.0.1" }) catch unreachable;
        params.api_keys = &.{"abc"};
    } else if (std.mem.eql(u8, name, "multitcp")) {
        // two TCP addresses on one fixed port, each IPv6-only where it can be
        params.hostnames = &.{ "127.0.0.1", "::1" };
        params.port = envPort();
    } else if (std.mem.eql(u8, name, "multiport0")) {
        // port 0 with two TCP addresses is refused
        params.hostnames = &.{ "127.0.0.1", "::1" };
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

extern "c" fn fflush(stream: ?*anyopaque) c_int;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("usage: http_port <config> <port-file>\n", .{});
        return 2;
    }
    const config = args[1];
    log.verbosity_thold = log.level_debug;

    var params: http.Params = .{};
    var ready = true;
    configure(config, &params, &ready);

    var ctx_http = try http.Context.init(gpa);
    if (!try ctx_http.setup(&params)) {
        log.inf("harness: init failed\n", .{});
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

    try ctx_http.registerGcpCompat();

    if (!try ctx_http.start()) {
        log.inf("harness: start failed\n", .{});
        return 1;
    }
    if (ready) ctx_http.is_ready.store(true, .seq_cst);
    for (ctx_http.listening_addresses.items) |address| {
        const addr = try gpa.dupeSentinel(u8, address, 0);
        log.inf("harness: listening on %s\n", .{addr.ptr});
    }

    {
        const port_str = try std.fmt.allocPrint(gpa, "{d}\n", .{ctx_http.port});
        const f = std.c.fopen(args[2], "w") orelse return 1;
        _ = std.c.fwrite(port_str.ptr, 1, port_str.len, f);
        _ = std.c.fclose(f);
    }

    // GET /quit, or <port-file>.quit appearing: a server that is not ready
    // answers /quit with 503
    const quit_file = try std.mem.concat(gpa, u8, &.{ args[2], ".quit\x00" });
    while (!g_quit.load(.seq_cst)) {
        if (std.c.fopen(@ptrCast(quit_file.ptr), "r")) |q| {
            _ = std.c.fclose(q);
            break;
        }
        sleepMs(5);
    }
    ctx_http.stop();
    ctx_http.join();
    log.inf("harness: stopped\n", .{});
    _ = fflush(null);
    return 0;
}
