//! `server_http_context`: llama-server's HTTP front end over cpp-httplib --
//! the request and response types every route handler sees, the API-key
//! and CORS middleware, the error and exception handlers, streaming, and
//! the Vertex AI (`AIP_*`) compatibility routes.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-http.cpp` and
//! `llama.cpp/tools/server/server-http.h` (v0.8.0, `8642541b2`), with
//! `common_http_parse_url` and `common_http_format_host` from
//! `llama.cpp/common/http.h`, and `format_error_response` and
//! `safe_json_to_str` from `llama.cpp/tools/server/server-common.cpp`, which
//! this file is the first to need. Each declaration names what it ports and
//! the line it began at.
//!
//! # The web UI
//!
//! Both arms of `#if defined(LLAMA_UI_HAS_ASSETS)` are ported, chosen at
//! comptime by `ui.has_assets` (`tools/server/ui.zig`), which is true when
//! the build embedded a bundle (`-Dui`, `build/ui.zig`). With assets, the
//! frontend paths are `/` and `/` plus every asset's name, and each asset
//! has a route: `index.html` at `/` and `/index.html`, revalidated, with the
//! cross-origin isolation headers; the PWA files uncached; everything else
//! immutable; ETag and `If-None-Match`; a gzip build refuses a client that
//! does not accept gzip with 415. Without, the frontend paths are `{"/"}`
//! and no asset route is registered. `--path` (`public_path`), which mounts
//! a directory through cpp-httplib's file handler, is not ported: `init`
//! refuses it. With the UI off, `/sw.js` answers with a service worker that
//! removes itself (v0.6.0), so a browser that cached the built-in UI drops
//! it.
//!
//! # Websockets
//!
//! `ws` registers a route whose handler runs on its own HTTP thread for the
//! life of the connection, over `WsConn`; `/v1/realtime` is the one route.
//! The frames are `httplib/ws.zig`.
//!
//! # Handlers and exceptions
//!
//! `handler_t`, a `std::function` returning a `server_http_res_ptr`, is
//! `Handler`: a context and a function returning `*Res`. `server_http_res`
//! is a base class the server derives from to override `on_complete`;
//! `Res` carries a `vtable` with `on_complete` and `destroy` for that, and
//! a derived response embeds a `Res` and recovers itself with
//! `@fieldParentPtr`.
//!
//! A C++ exception is an error here, and its `what()` is `what(err)`:
//! `error.InvalidArgument` is `std::invalid_argument`, `error.RuntimeError`
//! any other `std::exception` thrown by message (`throwInvalidArgument`,
//! `throwRuntimeError` set the text), `error.JsonError` a
//! `common_json_error` (`json.what()`), `error.OutOfMemory`
//! `std::bad_alloc`. Any other error is an exception that is not a
//! `std::exception` -- the C's `catch (...)`.
//!
//! # Params
//!
//! `server_http_context::init` reads seventeen members of `common_params`.
//! `Params` is exactly those, under the same names, so this file does not
//! depend on `tools/common/params.zig`'s server members while they are
//! being written; the server builds one from its `Params` (a field-by-field
//! copy). It is kept by `init` and must outlive the context, as the C's
//! pre-routing lambda keeps a reference to `common_params`.

const std = @import("std");
const httplib = @import("httplib/module.zig");
const tools_common = @import("tools_common");
const json = @import("cli_fit").json;
const libcxx = @import("llama").libcxx;
const ui = @import("ui.zig");

const log = tools_common.log;

/// `__func__` inside a C++ lambda.
const lambda = "operator()";

// -----------------------------------------------------------------------------
// Exceptions

/// A C++ exception, as an error: see the file's doc comment.
pub const Exception = error{ InvalidArgument, RuntimeError, JsonError, OutOfMemory };

threadlocal var what_buf: [4096]u8 = undefined;
threadlocal var what_len: usize = 0;

fn setWhat(msg: []const u8) void {
    what_len = @min(msg.len, what_buf.len);
    @memcpy(what_buf[0..what_len], msg[0..what_len]);
}

/// `throw std::invalid_argument(msg)`.
pub fn throwInvalidArgument(msg: []const u8) error{InvalidArgument} {
    setWhat(msg);
    return error.InvalidArgument;
}

/// `throw std::runtime_error(msg)`.
pub fn throwRuntimeError(msg: []const u8) error{RuntimeError} {
    setWhat(msg);
    return error.RuntimeError;
}

/// `e.what()` of the exception `err` stands for.
///
/// Return: borrowed until the next throw on this thread; null for an
/// exception that is not a `std::exception`.
pub fn what(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.InvalidArgument, error.RuntimeError => what_buf[0..what_len],
        error.JsonError => json.what(),
        error.OutOfMemory => "std::bad_alloc",
        else => null,
    };
}

// -----------------------------------------------------------------------------
// server-common.cpp

/// `error_type` (server-common.h:60 @8642541b2).
pub const ErrorType = enum {
    invalid_request,
    authentication,
    server,
    not_found,
    permission,
    unavailable,
    not_supported,
    exceed_context_size,
};

/// Ports `format_error_response` (server-common.cpp:93 @8642541b2):
/// `{"code", "message", "type"}`.
///
/// Return: the object, owned by the caller.
pub fn formatErrorResponse(a: std.mem.Allocator, msg: []const u8, ty: ErrorType) json.Error!json.Value {
    const type_str: []const u8, const code: i64 = switch (ty) {
        .invalid_request => .{ "invalid_request_error", 400 },
        .authentication => .{ "authentication_error", 401 },
        .not_found => .{ "not_found_error", 404 },
        .server => .{ "server_error", 500 },
        .permission => .{ "permission_error", 403 },
        .not_supported => .{ "not_supported_error", 501 },
        .unavailable => .{ "unavailable_error", 503 },
        .exceed_context_size => .{ "exceed_context_size_error", 400 },
    };
    var v = json.Value.emptyObject();
    errdefer v.deinit(a);
    try v.set(a, "code", try json.Value.from(a, code));
    try v.set(a, "message", try json.Value.initString(a, msg));
    try v.set(a, "type", try json.Value.initString(a, type_str));
    return v;
}

/// Ports `safe_json_to_str` (server-common.cpp:1716 @8642541b2): compact,
/// bad UTF-8 replaced.
///
/// Return: allocated by `a`.
pub fn safeJsonToStr(a: std.mem.Allocator, v: *const json.Value) error{OutOfMemory}![]u8 {
    return json.dumpSafe(a, v, -1);
}

/// `{"error": {"message": msg, "type": type_str, "code": code}}`, the
/// literal `server-http.cpp` writes for its own errors, as text.
fn errorBody(a: std.mem.Allocator, msg: []const u8, type_str: []const u8, code: i64) json.Error![]u8 {
    var inner = json.Value.emptyObject();
    errdefer inner.deinit(a);
    try inner.set(a, "message", try json.Value.initString(a, msg));
    try inner.set(a, "type", try json.Value.initString(a, type_str));
    try inner.set(a, "code", try json.Value.from(a, code));
    var outer = json.Value.emptyObject();
    defer outer.deinit(a);
    try outer.set(a, "error", inner);
    return safeJsonToStr(a, &outer);
}

// -----------------------------------------------------------------------------
// common/http.h

/// Ports `common_http_format_host` (http.h:27 @8642541b2): an IPv6 literal
/// bracketed.
///
/// Return: allocated by `a`.
pub fn commonHttpFormatHost(a: std.mem.Allocator, host: []const u8) error{OutOfMemory}![]u8 {
    if (std.mem.findScalar(u8, host, ':') != null) return std.mem.concat(a, u8, &.{ "[", host, "]" });
    return a.dupe(u8, host);
}

/// Ports `common_http_parse_url` (http.h:31 @8642541b2) as far as
/// `origin_is_localhost` reads it: the host. A malformed URL throws in the
/// C; this returns null.
fn commonHttpParseUrlHost(a: std.mem.Allocator, url: []const u8) error{OutOfMemory}!?[]const u8 {
    const scheme_end = std.mem.find(u8, url, "://") orelse return null;
    const scheme = url[0..scheme_end];
    if (!std.mem.eql(u8, scheme, "http") and !std.mem.eql(u8, scheme, "https")) return null;
    var rest = url[scheme_end + 3 ..];
    if (std.mem.findScalar(u8, rest, '@')) |at_pos| rest = rest[at_pos + 1 ..];
    var host = if (std.mem.findScalar(u8, rest, '/')) |slash| rest[0..slash] else rest;
    var port_str: []const u8 = "";
    if (host.len != 0 and host[0] == '[') {
        const close = std.mem.findScalar(u8, host, ']') orelse return null;
        const after = host[close + 1 ..];
        if (after.len != 0 and after[0] == ':') port_str = after[1..];
        host = host[1..close];
    } else if (std.mem.findScalar(u8, host, ':')) |colon| {
        port_str = host[colon + 1 ..];
        host = host[0..colon];
    }
    // std::stoi on the port, which throws on no digits or out of range
    if (port_str.len != 0) {
        const z = try a.dupeSentinel(u8, port_str, 0);
        defer a.free(z);
        _ = tools_common.arg.stoi(z) catch return null;
    }
    return host;
}

/// Ports `origin_is_localhost` (server-http.cpp:75 @8642541b2).
fn originIsLocalhost(a: std.mem.Allocator, origin: []const u8) error{OutOfMemory}!bool {
    const host = try commonHttpParseUrlHost(a, origin) orelse return false;
    return std.mem.eql(u8, host, "localhost") or std.mem.eql(u8, host, "127.0.0.1") or std.mem.eql(u8, host, "::1");
}

// -----------------------------------------------------------------------------
// Logging

fn cz(a: std.mem.Allocator, s: []const u8) [*:0]const u8 {
    return (a.dupeSentinel(u8, s, 0) catch return "").ptr;
}

fn srvDbg(comptime func: []const u8, comptime fmt: []const u8, args: anytype) void {
    log.dbg("srv  %12.*s: " ++ fmt, .{ @as(c_int, 12), @as([*:0]const u8, func ++ "") } ++ args);
}

fn srvTrc(comptime func: []const u8, comptime fmt: []const u8, args: anytype) void {
    log.trc("srv  %12.*s: " ++ fmt, .{ @as(c_int, 12), @as([*:0]const u8, func ++ "") } ++ args);
}

fn srvInf(comptime func: []const u8, comptime fmt: []const u8, args: anytype) void {
    log.inf("srv  %12.*s: " ++ fmt, .{ @as(c_int, 12), @as([*:0]const u8, func ++ "") } ++ args);
}

fn srvWrn(comptime func: []const u8, comptime fmt: []const u8, args: anytype) void {
    log.wrn("srv  %12.*s: " ++ fmt, .{ @as(c_int, 12), @as([*:0]const u8, func ++ "") } ++ args);
}

fn srvErr(comptime func: []const u8, comptime fmt: []const u8, args: anytype) void {
    log.err("srv  %12.*s: " ++ fmt, .{ @as(c_int, 12), @as([*:0]const u8, func ++ "") } ++ args);
}

// -----------------------------------------------------------------------------
// Requests and responses

/// A `std::map<std::string, V>`: unique keys in byte order.
pub fn SortedMap(comptime V: type) type {
    return struct {
        const Self = @This();
        pub const Entry = struct { key: []const u8, value: V };
        entries: std.ArrayList(Entry) = .empty,

        fn lowerBound(self: *const Self, key: []const u8) usize {
            var lo: usize = 0;
            var hi = self.entries.items.len;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (std.mem.order(u8, self.entries.items[mid].key, key) == .lt) lo = mid + 1 else hi = mid;
            }
            return lo;
        }

        /// `find`.
        pub fn get(self: *const Self, key: []const u8) ?*V {
            const i = self.lowerBound(key);
            if (i < self.entries.items.len and std.mem.eql(u8, self.entries.items[i].key, key)) return &self.entries.items[i].value;
            return null;
        }

        /// `m[key] = value`: inserts or replaces; `key` and `value` are
        /// borrowed.
        pub fn put(self: *Self, a: std.mem.Allocator, key: []const u8, value: V) error{OutOfMemory}!void {
            const i = self.lowerBound(key);
            if (i < self.entries.items.len and std.mem.eql(u8, self.entries.items[i].key, key)) {
                self.entries.items[i].value = value;
                return;
            }
            try self.entries.insert(a, i, .{ .key = key, .value = value });
        }
    };
}

/// `std::map<std::string, std::string>`.
pub const StringMap = SortedMap([]const u8);

/// Ports `uploaded_file` (server-http.h:43 @8642541b2).
pub const UploadedFile = struct {
    data: []const u8,
    filename: []const u8,
    content_type: []const u8,
};

/// `const std::function<bool()> & should_stop`: whether the client went
/// away, by asking cpp-httplib's request, or for a websocket its connection.
pub const ShouldStop = struct {
    req: ?*const httplib.Request,
    /// `[]() { return false; }`: what router mode's `proxy_request` passes
    /// when `detached` (server-models.cpp:1720 @8642541b2) is set.
    never: bool = false,
    /// The `should_stop` lambda of a websocket route (server-http.cpp:779
    /// @8642541b2): `!sock.is_open()`.
    ws: ?*const httplib.ws.WebSocket = null,

    pub fn call(self: ShouldStop) bool {
        if (self.never) return false;
        if (self.ws) |w| return !w.isOpen();
        const r = self.req orelse return true;
        return r.isConnectionClosed();
    }
};

/// Ports `server_http_req` (server-http.h:49 @8642541b2): what a handler is
/// given. Every string is borrowed for the request's lifetime.
pub const Req = struct {
    /// path_params + query_params
    params: StringMap = .{},
    /// used by MCP proxy
    headers: StringMap = .{},
    path: []const u8 = "",
    /// query parameters string (e.g. "action=save")
    query_string: []const u8 = "",
    body: []const u8 = "",
    /// used for file uploads (form data)
    files: SortedMap(UploadedFile) = .{},
    should_stop: ShouldStop = .{ .req = null },

    /// Ports `get_param` (server-http.h:58 @8642541b2), a member of
    /// `server_http_req`.
    pub fn getParam(self: *const Req, key: []const u8, def: []const u8) []const u8 {
        if (self.params.get(key)) |v| return v.*;
        return def;
    }
};

/// Ports `server_http_res` (server-http.h:21 @8642541b2): one of two
/// modes -- a whole body in `data`, or, with `next` set, a stream each
/// call to which writes the next chunk (possibly empty) and says whether
/// more follow. Owns an arena every string in it lives in.
pub const Res = struct {
    arena: std.heap.ArenaAllocator,
    content_type: []const u8 = "application/json; charset=utf-8",
    status: c_int = 200,
    data: std.ArrayList(u8) = .empty,
    headers: StringMap = .{},
    next: ?Next = null,
    vtable: *const VTable = &base_vtable,

    /// `std::function<bool(std::string &)> next`: appends the chunk to
    /// `out`, which is empty on entry.
    pub const Next = struct {
        ctx: ?*anyopaque,
        func: *const fn (ctx: ?*anyopaque, out: *std.ArrayList(u8), a: std.mem.Allocator) anyerror!bool,
    };

    /// The class's virtual members: `on_complete`, fired before the request
    /// and response go away, and the destructor.
    pub const VTable = struct {
        on_complete: *const fn (self: *Res) void,
        destroy: *const fn (self: *Res) void,
    };

    const base_vtable: VTable = .{ .on_complete = noComplete, .destroy = destroyBase };

    /// The base class's `on_complete`: nothing.
    pub fn noComplete(_: *Res) void {}

    /// The base class's destructor: the arena, then the response. A
    /// derived response embedding `Res` frees itself instead.
    pub fn destroyBase(self: *Res) void {
        const backing = self.arena.child_allocator;
        self.arena.deinit();
        backing.destroy(self);
    }

    /// `std::make_unique<server_http_res>()`.
    ///
    /// Parameters:
    /// - `backing`: allocates the response and backs its arena; must be
    ///   thread-safe, as a stream is finished on cpp-httplib's thread.
    ///
    /// Return: the response; `destroy` frees it.
    pub fn create(backing: std.mem.Allocator) error{OutOfMemory}!*Res {
        const self = try backing.create(Res);
        self.* = .{ .arena = std.heap.ArenaAllocator.init(backing) };
        return self;
    }

    /// The arena this response's strings live in.
    pub fn allocator(self: *Res) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Ports `is_stream` (server-http.h:28 @8642541b2), a member of
    /// `server_http_res`.
    pub fn isStream(self: *const Res) bool {
        return self.next != null;
    }

    /// `headers[key] = value`, both copied into the arena.
    pub fn setHeader(self: *Res, key: []const u8, value: []const u8) error{OutOfMemory}!void {
        const a = self.allocator();
        try self.headers.put(a, try a.dupe(u8, key), try a.dupe(u8, value));
    }

    /// `data = s`.
    pub fn setData(self: *Res, s: []const u8) error{OutOfMemory}!void {
        self.data.clearRetainingCapacity();
        try self.data.appendSlice(self.allocator(), s);
    }

    pub fn onComplete(self: *Res) void {
        self.vtable.on_complete(self);
    }

    pub fn destroy(self: *Res) void {
        self.vtable.destroy(self);
    }
};

/// `handler_t` (server-http.h:74 @8642541b2): a route. An error is an
/// exception it threw; the server wraps every route in `ex_wrapper` so
/// none escapes.
pub const Handler = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, req: *const Req) anyerror!*Res,

    pub fn call(self: Handler, req: *const Req) anyerror!*Res {
        return self.func(self.ctx, req);
    }
};

/// Ports `ws_conn` (server-http.h:78 @8642541b2): a websocket connection, as
/// a handler sees it. `read` blocks; `send` and `sendBinary` may be called
/// while another thread is in `read`.
pub const WsConn = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        read: *const fn (ctx: *anyopaque, a: std.mem.Allocator, msg: *std.ArrayList(u8)) error{OutOfMemory}!c_int,
        send: *const fn (ctx: *anyopaque, msg: []const u8) bool,
        send_binary: *const fn (ctx: *anyopaque, msg: []const u8) bool,
        is_open: *const fn (ctx: *anyopaque) bool,
        close: *const fn (ctx: *anyopaque) void,
    };

    /// `read`: 1 for a text message, 2 for a binary message, 0 when the
    /// connection is closed. `msg` holds the message, allocated by `a`.
    pub fn read(self: WsConn, a: std.mem.Allocator, msg: *std.ArrayList(u8)) error{OutOfMemory}!c_int {
        return self.vtable.read(self.ctx, a, msg);
    }

    /// `send`: a text message; false when it could not be written.
    pub fn send(self: WsConn, msg: []const u8) bool {
        return self.vtable.send(self.ctx, msg);
    }

    /// `send_binary`: a binary message.
    pub fn sendBinary(self: WsConn, msg: []const u8) bool {
        return self.vtable.send_binary(self.ctx, msg);
    }

    /// `is_open`.
    pub fn isOpen(self: WsConn) bool {
        return self.vtable.is_open(self.ctx);
    }

    /// `close`.
    pub fn close(self: WsConn) void {
        self.vtable.close(self.ctx);
    }
};

/// `ws_handler_t` (server-http.h:89 @8642541b2): runs on its own HTTP thread
/// for the whole lifetime of the connection.
pub const WsHandler = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, req: *const Req, conn: *WsConn) void,

    pub fn call(self: WsHandler, req: *const Req, conn: *WsConn) void {
        self.func(self.ctx, req, conn);
    }
};

// -----------------------------------------------------------------------------
// Params

/// The members of `common_params` that `server_http_context::init` reads,
/// with their defaults: `n_parallel`, and the server block from `port` to
/// `ui`, all in `llama.cpp/common/common.h`.
pub const Params = struct {
    api_prefix: []const u8 = "",
    port: i32 = 9931,
    hostnames: []const []const u8 = &.{"127.0.0.1"},
    ssl_file_key: []const u8 = "",
    ssl_file_cert: []const u8 = "",
    timeout_read: i32 = 3600,
    timeout_write: i32 = 3600,
    reuse_port: bool = false,
    api_keys: []const []const u8 = &.{},
    cors_origins: []const u8 = "*",
    cors_methods: []const u8 = "GET, POST, DELETE, OPTIONS",
    cors_headers: []const u8 = "*",
    cors_credentials: bool = true,
    n_threads_http: i32 = -1,
    n_parallel: i32 = 1,
    ui: bool = true,
    public_path: []const u8 = "",
};

// -----------------------------------------------------------------------------
// GCP

/// Ports `gcp_params` (server-http.cpp:85 @8642541b2): the Vertex AI
/// container's `AIP_*` variables.
const GcpParams = struct {
    enabled: bool,
    path_health: []const u8,
    path_predict: []const u8,
    port: c_int,

    /// Ports `gcp_params` (server-http.cpp:92 @8642541b2), the constructor.
    /// `std::stoi` on a bad `AIP_HTTP_PORT` throws out of it, uncaught, and
    /// terminates; so does this.
    fn init(a: std.mem.Allocator) error{OutOfMemory}!GcpParams {
        const port_str = try getenvOr(a, "AIP_HTTP_PORT", "8080", false);
        const port = tools_common.arg.stoi(try a.dupeSentinel(u8, port_str, 0)) catch {
            const msg = tools_common.arg.what();
            std.debug.print("libc++abi: terminating due to uncaught exception of type std::invalid_argument: {s}\n", .{msg});
            std.process.abort();
        };
        return .{
            .enabled = std.mem.eql(u8, try getenvOr(a, "AIP_MODE", "", false), "PREDICTION"),
            .path_health = try getenvOr(a, "AIP_HEALTH_ROUTE", "", true),
            .path_predict = try getenvOr(a, "AIP_PREDICT_ROUTE", "/predict", true),
            .port = port,
        };
    }

    /// Ports `getenv` (server-http.cpp:99 @8642541b2), a member of
    /// `gcp_params`.
    fn getenvOr(a: std.mem.Allocator, name: [*:0]const u8, default_value: []const u8, ensure_leading_slash: bool) error{OutOfMemory}![]const u8 {
        const value = std.c.getenv(name) orelse return default_value;
        const val = std.mem.span(value);
        if (val.len == 0) return default_value;
        if (ensure_leading_slash and val[0] != '/') return std.mem.concat(a, u8, &.{ "/", val });
        return a.dupe(u8, val);
    }
};

/// Ports `path_to_gcp_format` (server-http.cpp:826 @8642541b2): a route as
/// its camelCase `@requestFormat` alias.
fn pathToGcpFormat(a: std.mem.Allocator, path: []const u8) error{OutOfMemory}![]u8 {
    var s = path;
    if (s.len > 3 and s[0] == '/' and s[1] == 'v' and s[2] == '1') s = s[3..];
    if (s.len != 0 and s[0] == '/') s = s[1..];
    var result: std.ArrayList(u8) = .empty;
    var cap = false;
    for (s) |c| {
        if (c == ':') break;
        if (c == '/' or c == '-' or c == '_') {
            cap = true;
        } else {
            // std::toupper in the "C" locale
            try result.append(a, if (cap) std.ascii.toUpper(c) else c);
            cap = false;
        }
    }
    return result.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// The context

/// Ports `server_http_context` (server-http.h:67 @8642541b2), with its
/// `Impl` (server-http.cpp:20 @8642541b2): one cpp-httplib server per
/// listening address, each on a thread of its own, all handing their
/// connections to one shared pool.
pub const Context = struct {
    allocator: std.mem.Allocator,
    /// one listener per host
    servers: std.ArrayList(*httplib.Server) = .empty,
    hosts: []const []const u8 = &.{},
    /// one thread per listener
    threads: std.ArrayList(std.Thread) = .empty,
    /// single pool shared among all listeners
    pool: ?*httplib.ThreadPool = null,
    is_ready: std.atomic.Value(bool) = .init(false),
    /// Every route registered, by path without the prefix: libc++'s
    /// `unordered_map`, whose order `register_gcp_compat` reads.
    handlers: libcxx.StringMap(Handler),
    path_prefix: []const u8 = "",
    port: c_int = 9931,
    is_ssl: bool = false,
    /// for debugging
    listening_addresses: std.ArrayList([]const u8) = .empty,
    params: *const Params = &default_params,
    n_threads_http: i32 = 0,
    /// The strings `init` and the route registrations keep.
    arena: std.heap.ArenaAllocator,

    const default_params: Params = .{};

    /// Ports `server_http_context` (server-http.cpp:38 @8642541b2), the
    /// constructor.
    ///
    /// Parameters:
    /// - `allocator`: thread-safe; backs the server, its connections and
    ///   the context's own strings.
    ///
    /// Return: the context; `deinit` frees it once its thread has been
    /// joined.
    pub fn init(allocator: std.mem.Allocator) error{OutOfMemory}!Context {
        return .{
            .allocator = allocator,
            .handlers = try libcxx.StringMap(Handler).init(allocator),
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    /// Ports the destructor (`~server_http_context`, server-http.cpp:42
    /// @8642541b2): stops and joins, in case an exit path forgot to, then
    /// frees the servers.
    pub fn deinit(self: *Context) void {
        self.stop();
        self.join();
        for (self.servers.items) |s| s.destroy();
        self.servers.deinit(self.allocator);
        self.threads.deinit(self.allocator);
        self.listening_addresses.deinit(self.allocator);
        self.handlers.deinit();
        self.arena.deinit();
    }

    fn arenaAlloc(self: *Context) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Ports `init` (server-http.cpp:112 @8642541b2), a member of
    /// `server_http_context`: a listener per host (`init_listener`), each
    /// IPv6-only when several TCP addresses are given, and the size of the
    /// pool they share.
    ///
    /// Parameters:
    /// - `params`: kept; must outlive the context.
    ///
    /// Return: false where the C returns false.
    pub fn setup(self: *Context, params: *const Params) !bool {
        const a = self.arenaAlloc();
        const gcp = try GcpParams.init(a);

        self.params = params;
        self.path_prefix = params.api_prefix;
        self.port = params.port;

        if (gcp.enabled) {
            srvTrc("init", "Google Cloud Platform compat: health route = %s, predict route = %s, port = %d\n", .{ cz(a, gcp.path_health), cz(a, gcp.path_predict), gcp.port });
            if (self.port != gcp.port) {
                srvWrn("init", "Google Cloud Platform compat: overriding server port %d with AIP_HTTP_PORT %d\n", .{ self.port, gcp.port });
            }
            self.port = gcp.port;
        }

        self.hosts = params.hostnames;
        var n_tcp_hosts: usize = 0;
        for (self.hosts) |host| {
            if (!std.mem.endsWith(u8, host, ".sock")) n_tcp_hosts += 1;
        }
        if (self.port == 0 and n_tcp_hosts > 1) {
            srvErr("init", "%s", .{@as([*:0]const u8, "--port 0 is not supported with multiple TCP addresses\n")});
            return false;
        }
        for (self.hosts) |_| {
            if (!try self.initListener(params)) return false;
            // with multiple TCP addresses, [::] must not also claim 0.0.0.0
            if (n_tcp_hosts > 1) self.servers.items[self.servers.items.len - 1].setIpv6V6only(true);
        }

        var n_threads_http = params.n_threads_http;
        if (n_threads_http < 1) {
            // +4 threads for monitoring, health and MCP.
            const hw: i32 = @bitCast(httplib.socket.hardwareConcurrency() -% 1);
            n_threads_http = @max(params.n_parallel + 4, hw);
        }
        self.n_threads_http = n_threads_http;
        srvTrc("init", "using %d threads for HTTP server\n", .{n_threads_http});
        return true;
    }

    /// Ports `init_listener` (server-http.cpp:159 @8642541b2), a member of
    /// `server_http_context`: builds one listener -- default headers, the
    /// exception and error handlers, timeouts, socket options, the
    /// middleware, the task queue over the shared pool, the UI routes, the
    /// self-removing service worker when the built-in UI is not served --
    /// from `params`.
    ///
    /// Return: false where the C returns false.
    fn initListener(self: *Context, params: *const Params) !bool {
        const a = self.arenaAlloc();
        if (params.ssl_file_key.len != 0 and params.ssl_file_cert.len != 0) {
            srvErr("init_listener", "%s", .{@as([*:0]const u8, "the server is built without SSL support\n")});
            return false;
        }
        const srv = try httplib.Server.create(self.allocator);
        self.servers.append(self.allocator, srv) catch |e| {
            srv.destroy();
            return e;
        };

        try srv.setDefaultHeaders(&.{.{ .key = "Server", .value = "llama.cpp" }});
        srv.setExceptionHandler(.{ .ctx = null, .func = exceptionHandler });
        try srv.setErrorHandler(.{ .ctx = null, .func = errorHandler });

        // set timeouts and change hostname and port
        srv.setReadTimeout(params.timeout_read, 0);
        srv.setWriteTimeout(params.timeout_write, 0);
        srv.setSocketOptions(.{ .ctx = @constCast(params), .func = socketOptions });

        if (params.api_keys.len == 1) {
            const key = params.api_keys[0];
            // key.length() - 4 as a size_t cast to int, then max'd with 0
            const tail: usize = if (key.len >= 4) key.len - 4 else 0;
            srvTrc("init_listener", "api_keys: ****%s\n", .{cz(a, key[tail..])});
        } else if (params.api_keys.len > 1) {
            srvTrc("init_listener", "api_keys: %zu keys loaded\n", .{params.api_keys.len});
        }

        srv.setPreRoutingHandler(.{ .ctx = self, .func = preRouting });
        srv.new_task_queue = .{ .ctx = self, .func = newTaskQueue };

        // Web UI setup
        if (!params.ui) {
            srvInf("init_listener", "%s", .{@as([*:0]const u8, "The UI is disabled\n")});
            srvInf("init_listener", "%s", .{@as([*:0]const u8, "Use --ui/--no-ui (or deprecated --webui/--no-webui) to enable/disable\n")});
        } else if (params.public_path.len != 0) {
            // set_mount_point and cpp-httplib's file handler are not ported
            srvErr("init_listener", "static assets path not supported by this port: %s\n", .{cz(a, params.public_path)});
            return false;
        } else if (comptime ui.has_assets) {
            try self.registerUiRoutes(srv, params);
        }

        // a browser that used the built-in UI keeps its service worker, so it shows the old UI even after the UI is replaced or disabled
        // answer the worker's update check with a worker that removes itself; a sw.js in public_path is served first
        // (`public_path` is refused above, so only a disabled UI reaches this)
        if (!params.ui or params.public_path.len != 0) {
            try srv.get(try std.mem.concat(a, u8, &.{ params.api_prefix, "/sw.js" }), .{ .ctx = null, .func = serveSwRemove });
        }
        return true;
    }

    /// `sw_remove_js` (server-http.cpp:479 @8642541b2): a service worker
    /// that unregisters itself, empties its caches and reloads its windows.
    const sw_remove_js =
        \\
        \\self.addEventListener('install', () => self.skipWaiting());
        \\self.addEventListener('activate', (e) => e.waitUntil((async () => {
        \\    await self.registration.unregister();
        \\    for (const key of await caches.keys()) await caches.delete(key);
        \\    for (const c of await self.clients.matchAll({ type: 'window' })) c.navigate(c.url);
        \\})()));
        \\
    ;

    /// The `/sw.js` route `init_listener` registers when the built-in UI is
    /// not served: the lambda passed to `Get` (server-http.cpp:478
    /// @8642541b2).
    fn serveSwRemove(_: ?*anyopaque, _: *const httplib.Request, res: *httplib.Response) anyerror!void {
        try res.setHeader("Cache-Control", "no-cache");
        try res.setContent(sw_remove_js, "application/javascript");
    }

    /// One embedded asset's route: what `serve_asset_cached` or
    /// `serve_asset_nocache` captured.
    const AssetRoute = struct {
        name: []const u8,
        /// Null for `serve_asset_nocache`.
        cached: ?struct { isolation: bool, cache_control: []const u8 },
    };

    /// Hashed assets never change under a given name, so they can be cached forever.
    const cache_immutable = "public, max-age=31536000, immutable";
    /// `index.html`: its name is stable while its contents change on every build.
    const cache_revalidate = "no-cache";

    /// `no_cache_names` (server-http.cpp:454 @8642541b2): the PWA
    /// revalidation files.
    const no_cache_names = [_][]const u8{ "sw.js", "manifest.webmanifest", "_app/version.json", "build.json" };

    /// The `LLAMA_UI_HAS_ASSETS` arm of the UI setup in `init_listener`
    /// (server-http.cpp:159 @8642541b2), the block at its line 384: a route
    /// per embedded asset, the index at `/` and `/index.html`.
    fn registerUiRoutes(self: *Context, srv: *httplib.Server, params: *const Params) !void {
        const a = self.arenaAlloc();
        const prefix = params.api_prefix;
        // main index file -- revalidated, so a new build is picked up on the next load
        const index = try a.create(AssetRoute);
        index.* = .{ .name = "index.html", .cached = .{ .isolation = true, .cache_control = cache_revalidate } };
        try srv.get(try std.mem.concat(a, u8, &.{ prefix, "/" }), .{ .ctx = index, .func = serveAsset });
        try srv.get(try std.mem.concat(a, u8, &.{ prefix, "/index.html" }), .{ .ctx = index, .func = serveAsset });

        // All remaining assets registered directly from the embedded asset table.
        // PWA revalidation files (sw.js, manifest, version.json) use no-cache;
        // everything else is immutable.
        for (ui.getAssets()) |*asset| {
            if (std.mem.eql(u8, asset.name, "index.html")) continue; // served at "/" and "/index.html" above
            const route = try a.create(AssetRoute);
            const no_cache = for (no_cache_names) |n| {
                if (std.mem.eql(u8, n, asset.name)) break true;
            } else false;
            if (no_cache) {
                srvDbg("init_listener", "serve nocache for %s\n", .{cz(a, asset.name)});
                route.* = .{ .name = asset.name, .cached = null };
            } else {
                route.* = .{ .name = asset.name, .cached = .{ .isolation = false, .cache_control = cache_immutable } };
            }
            try srv.get(try std.mem.concat(a, u8, &.{ prefix, "/", asset.name }), .{ .ctx = route, .func = serveAsset });
        }
    }

    /// The `handle_gzip_header` lambda (server-http.cpp:386 @8642541b2): a
    /// gzip build needs a client that accepts gzip.
    ///
    /// Return: false, with a 415 and its message set, when it does not.
    fn handleGzipHeader(req: *const httplib.Request, res: *httplib.Response) !bool {
        if (!ui.useGzip()) {
            // no gzip build, skip
            return true;
        }
        if (std.mem.indexOf(u8, req.getHeaderValue("Accept-Encoding"), "gzip") == null) {
            res.status = 415; // unsupported media type
            try res.setContent("Error: gzip is not supported by this browser", "text/plain");
            return false;
        }
        try res.setHeader("Content-Encoding", "gzip");
        return true;
    }

    /// The handlers `serve_asset_cached` (server-http.cpp:408 @8642541b2)
    /// and `serve_asset_nocache` (server-http.cpp:431 @8642541b2) return: the
    /// asset with its caching policy, and for a cached one its ETag and a 304
    /// when `If-None-Match` names it, plain or weak.
    fn serveAsset(ctx: ?*anyopaque, req: *const httplib.Request, res: *httplib.Response) anyerror!void {
        const route: *const AssetRoute = @ptrCast(@alignCast(ctx.?));
        if (!try handleGzipHeader(req, res)) return; // returns error message
        const asset = ui.findAsset(route.name) orelse {
            res.status = 404;
            return;
        };
        const cached = route.cached orelse {
            try res.setHeader("Cache-Control", "no-cache");
            try res.setContent(asset.data, asset.type);
            return;
        };
        try res.setHeader("ETag", asset.etag);
        const inm = req.getHeaderValue("If-None-Match");
        if (inm.len != 0 and (std.mem.eql(u8, inm, asset.etag) or
            (std.mem.startsWith(u8, inm, "W/") and std.mem.eql(u8, inm[2..], asset.etag))))
        {
            res.status = 304;
            return;
        }
        if (cached.isolation) {
            try res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
            try res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        }
        try res.setHeader("Cache-Control", cached.cache_control);
        try res.setContent(asset.data, asset.type);
    }

    /// The `new_task_queue` lambda: a `server_http_task_queue`
    /// (server-http.cpp:29 @8642541b2) over the pool `start` made, whose
    /// `shutdown` and delete leave the pool alone -- `join` drains it.
    fn newTaskQueue(ctx: ?*anyopaque, _: std.mem.Allocator) anyerror!httplib.TaskQueue {
        const self: *Context = @ptrCast(@alignCast(ctx.?));
        const V = struct {
            fn enqueue(c: *anyopaque, fn_: httplib.thread_pool.Task) bool {
                const pool: *httplib.ThreadPool = @ptrCast(@alignCast(c));
                return pool.enqueue(fn_);
            }
            // note: must call join() to drain the pool
            fn shutdown(_: *anyopaque) void {}
            fn destroy(_: *anyopaque) void {}
            const vtable: httplib.TaskQueue.VTable = .{ .enqueue = enqueue, .shutdown = shutdown, .destroy = destroy };
        };
        return .{ .ctx = self.pool.?, .vtable = &V.vtable };
    }

    /// The `set_socket_options` lambda: `SO_REUSEADDR`, and `SO_REUSEPORT`
    /// with `--reuse-port`.
    fn socketOptions(ctx: ?*anyopaque, sock: httplib.socket.Socket) void {
        const params: *const Params = @ptrCast(@alignCast(ctx.?));
        _ = httplib.socket.setSocketOpt(sock, httplib.socket.k.SOL_SOCKET, httplib.socket.k.SO_REUSEADDR, 1);
        if (params.reuse_port) _ = httplib.socket.setSocketOpt(sock, httplib.socket.k.SOL_SOCKET, httplib.socket.k.SO_REUSEPORT, 1);
    }

    /// The `set_exception_handler` lambda: a fail-safe -- `ex_wrapper`
    /// should have caught it -- answering 500 with the message as text.
    fn exceptionHandler(_: ?*anyopaque, _: *const httplib.Request, res: *httplib.Response, err: anyerror) anyerror!void {
        const message = what(err) orelse "Unknown Exception";
        res.status = 500;
        try res.setContent(message, "text/plain");
        srvErr(lambda, "got exception: %s\n", .{cz(res.a, message)});
    }

    /// The `set_error_handler` lambda: a 404 gets its JSON body; other
    /// errors already have theirs.
    fn errorHandler(_: ?*anyopaque, _: *const httplib.Request, res: *httplib.Response) anyerror!void {
        if (res.status == 404) {
            try res.setContent(try errorBody(res.a, "File Not Found", "not_found_error", 404), "application/json; charset=utf-8");
        }
    }

    /// `frontend_paths` (server-http.cpp:243 @8642541b2): "/", and "/"
    /// plus every embedded asset's name.
    fn isFrontendPath(path: []const u8) bool {
        if (std.mem.eql(u8, path, "/")) return true;
        return path.len > 1 and path[0] == '/' and ui.findAsset(path[1..]) != null;
    }

    /// `get_public_endpoints`: the health routes and the frontend paths,
    /// unprefixed.
    fn isPublicEndpoint(path: []const u8) bool {
        return std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/v1/health") or isFrontendPath(path);
    }

    /// The `middleware_validate_api_key` lambda: with keys set, a non-public
    /// path needs one of them, in `Authorization` (a `Bearer ` prefix
    /// stripped) or `X-Api-Key`.
    fn validateApiKey(self: *Context, req: *const httplib.Request, res: *httplib.Response) !bool {
        const api_keys = self.params.api_keys;
        if (api_keys.len == 0) return true;
        if (isPublicEndpoint(req.path)) return true;
        var req_api_key = req.getHeaderValue("Authorization");
        if (req_api_key.len == 0) req_api_key = req.getHeaderValue("X-Api-Key");
        const prefix = "Bearer ";
        if (std.mem.startsWith(u8, req_api_key, prefix)) req_api_key = req_api_key[prefix.len..];
        for (api_keys) |k| if (std.mem.eql(u8, k, req_api_key)) return true;
        res.status = 401;
        try res.setContent(try errorBody(res.a, "Invalid API Key", "authentication_error", 401), "application/json; charset=utf-8");
        srvWrn(lambda, "%s", .{@as([*:0]const u8, "unauthorized: Invalid API Key\n")});
        return false;
    }

    /// The `middleware_server_state` lambda: until ready, only the frontend
    /// may load; everything else is 503.
    fn serverState(self: *Context, req: *const httplib.Request, res: *httplib.Response) !bool {
        if (!self.is_ready.load(.seq_cst)) {
            if (isFrontendPath(req.path)) return true;
            res.status = 503;
            try res.setContent(try errorBody(res.a, "Loading model", "unavailable_error", 503), "application/json; charset=utf-8");
            return false;
        }
        return true;
    }

    /// The `set_pre_routing_handler` lambda: CORS headers on everything, a
    /// preflight answered here, then readiness and the API key.
    fn preRouting(ctx: ?*anyopaque, req: *const httplib.Request, res: *httplib.Response) anyerror!httplib.HandlerResponse {
        const self: *Context = @ptrCast(@alignCast(ctx.?));
        const params = self.params;
        if (params.cors_credentials and std.mem.eql(u8, params.cors_origins, "*")) {
            // special case: echo back the Origin header to allow any origin to access the server with credentials
            try res.setHeader("Access-Control-Allow-Origin", req.getHeaderValue("Origin"));
        } else if (std.mem.eql(u8, params.cors_origins, "localhost")) {
            // special case: only reflect the Origin header if it is a localhost origin
            const origin = req.getHeaderValue("Origin");
            if (origin.len != 0 and try originIsLocalhost(res.a, origin)) {
                try res.setHeader("Access-Control-Allow-Origin", origin);
            } else if (origin.len != 0) {
                srvWrn(lambda, "(CORS) skip non-localhost origin: %s\n", .{cz(res.a, origin)});
            }
        } else {
            try res.setHeader("Access-Control-Allow-Origin", params.cors_origins);
        }
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (std.mem.eql(u8, req.method, "OPTIONS")) {
            try res.setHeader("Access-Control-Allow-Credentials", if (params.cors_credentials) "true" else "false");
            try res.setHeader("Access-Control-Allow-Methods", params.cors_methods);
            try res.setHeader("Access-Control-Allow-Headers", params.cors_headers);
            try res.setContent("", "text/html");
            return .handled;
        }
        if (!try self.serverState(req, res)) return .handled;
        if (!try self.validateApiKey(req, res)) return .handled;
        return .unhandled;
    }

    /// Ports `start` (server-http.cpp:494 @8642541b2), a member of
    /// `server_http_context`: binds each listener -- a Unix socket for a
    /// host ending in `.sock`, any free port for port 0 -- then makes the
    /// shared pool and runs each accept loop on a thread of its own.
    pub fn start(self: *Context) !bool {
        const a = self.arenaAlloc();
        self.listening_addresses.clearRetainingCapacity();
        for (self.servers.items, self.hosts) |srv, host| {
            const is_sock = std.mem.endsWith(u8, host, ".sock");
            const host_z = try a.dupeSentinel(u8, host, 0);
            var was_bound: bool = undefined;
            if (is_sock) {
                srvTrc("start", "%s", .{@as([*:0]const u8, "setting address family to AF_UNIX\n")});
                srv.setAddressFamily(httplib.socket.k.AF_UNIX);
                // AF_UNIX ignores the port, but bind_to_port requires a nonzero value.
                was_bound = srv.bindToPort(host_z, 8080);
            } else if (self.port == 0) {
                const bound_port = srv.bindToAnyPort(host_z);
                was_bound = bound_port >= 0;
                if (was_bound) self.port = bound_port;
            } else {
                was_bound = srv.bindToPort(host_z, self.port);
            }
            if (!was_bound) {
                srvErr("start", "couldn't bind HTTP server socket, hostname: %s, port: %d\n", .{ host_z.ptr, self.port });
                self.stop();
                self.listening_addresses.clearRetainingCapacity();
                return false;
            }
            const address = if (is_sock)
                try std.fmt.allocPrint(a, "unix://{s}", .{host})
            else
                try std.fmt.allocPrint(a, "{s}://{s}:{d}", .{ if (self.is_ssl) "https" else "http", try commonHttpFormatHost(a, host), self.port });
            try self.listening_addresses.append(self.allocator, address);
        }

        // n_threads_http fixed threads (always alive), plus up to 1024 dynamic threads destroyed after each request
        // ref: https://github.com/yhirose/cpp-httplib/pull/2368
        const n: usize = @intCast(self.n_threads_http);
        self.pool = try httplib.ThreadPool.create(self.allocator, n, n + 1024);
        for (self.servers.items, 0..) |srv, i| {
            const addr = self.listening_addresses.items[i];
            try self.threads.ensureUnusedCapacity(self.allocator, 1);
            self.threads.appendAssumeCapacity(try std.Thread.spawn(.{}, listenThread, .{ srv, addr }));
            srv.waitUntilReady();
            if (!srv.is_running.load(.seq_cst)) {
                srvErr("start", "couldn't start HTTP listener on %s\n", .{cz(a, addr)});
                self.stop();
                self.join();
                self.listening_addresses.clearRetainingCapacity();
                return false;
            }
        }
        return true;
    }

    fn listenThread(srv: *httplib.Server, addr: []const u8) void {
        if (!srv.listenAfterBind()) {
            const z = std.heap.c_allocator.dupeSentinel(u8, addr, 0) catch return;
            defer std.heap.c_allocator.free(z);
            srvErr("operator()", "listener on %s stopped unexpectedly\n", .{z.ptr});
        }
    }

    /// Ports `stop` (server-http.cpp:549 @8642541b2), a member of
    /// `server_http_context`: every listener.
    pub fn stop(self: *const Context) void {
        for (self.servers.items) |s| s.stop();
    }

    /// Ports `join` (server-http.cpp:557 @8642541b2), a member of
    /// `server_http_context`: waits for every accept loop, then drains and
    /// frees the shared pool -- queued requests still refer to their servers
    /// until the workers finish.
    pub fn join(self: *Context) void {
        for (self.threads.items) |t| t.join();
        self.threads.clearRetainingCapacity();
        if (self.pool) |pool| {
            pool.shutdown();
            pool.destroy();
            self.pool = null;
        }
    }

    /// The route bound into cpp-httplib for one `handler_t`.
    const Route = struct {
        ctx: *Context,
        handler: Handler,
        kind: enum { get, post, del },
    };

    fn registerRoute(self: *Context, path: []const u8, handler: Handler, kind: @FieldType(Route, "kind")) !void {
        const a = self.arenaAlloc();
        _ = try self.handlers.getOrPut(path, handler);
        const route = try a.create(Route);
        route.* = .{ .ctx = self, .handler = handler, .kind = kind };
        const pattern = try std.mem.concat(a, u8, &.{ self.path_prefix, path });
        const h: httplib.Handler = .{ .ctx = route, .func = routeHandler };
        for (self.servers.items) |srv| switch (kind) {
            .get => try srv.get(pattern, h),
            .post => try srv.post(pattern, h),
            .del => try srv.delete(pattern, h),
        };
    }

    /// Ports `get` (server-http.cpp:676 @8642541b2), a member of
    /// `server_http_context`.
    pub fn get(self: *Context, path: []const u8, handler: Handler) !void {
        try self.registerRoute(path, handler, .get);
    }

    /// Ports `post` (server-http.cpp:697 @8642541b2), a member of
    /// `server_http_context`: a `multipart/form-data` body becomes a JSON
    /// object of its text fields -- a repeated name an array -- and its
    /// files.
    pub fn post(self: *Context, path: []const u8, handler: Handler) !void {
        try self.registerRoute(path, handler, .post);
    }

    /// Ports `del` (server-http.cpp:798 @8642541b2), a member of
    /// `server_http_context`.
    pub fn del(self: *Context, path: []const u8, handler: Handler) !void {
        try self.registerRoute(path, handler, .del);
    }

    /// Ports `ws` (server-http.cpp:748 @8642541b2), a member of
    /// `server_http_context`: a route on every listener, with the prefix.
    /// Unlike `get`, it is not kept in `handlers`.
    pub fn ws(self: *Context, path: []const u8, handler: WsHandler) !void {
        const a = self.arenaAlloc();
        const full_path = try std.mem.concat(a, u8, &.{ self.path_prefix, path });
        const route = try a.create(WsRoute);
        route.* = .{ .handler = handler };
        for (self.servers.items) |srv| {
            try srv.webSocket(full_path, .{ .ctx = route, .func = WsRoute.callback });
        }
    }

    /// The handler `ws`'s lambda holds.
    const WsRoute = struct {
        handler: WsHandler,

        /// The `conn_impl` of `ws`: `ws_conn` over cpp-httplib's socket.
        const ConnImpl = struct {
            sock: *httplib.ws.WebSocket,
            mutex_send: std.c.pthread_mutex_t = .{},

            fn selfOf(ctx: *anyopaque) *ConnImpl {
                return @ptrCast(@alignCast(ctx));
            }
            fn read(ctx: *anyopaque, a: std.mem.Allocator, msg: *std.ArrayList(u8)) error{OutOfMemory}!c_int {
                return switch (try selfOf(ctx).sock.read(a, msg)) {
                    .text => 1,
                    .binary => 2,
                    .fail => 0,
                };
            }
            fn send(ctx: *anyopaque, msg: []const u8) bool {
                const c = selfOf(ctx);
                _ = std.c.pthread_mutex_lock(&c.mutex_send);
                defer _ = std.c.pthread_mutex_unlock(&c.mutex_send);
                return c.sock.send(msg);
            }
            fn sendBinary(ctx: *anyopaque, msg: []const u8) bool {
                const c = selfOf(ctx);
                _ = std.c.pthread_mutex_lock(&c.mutex_send);
                defer _ = std.c.pthread_mutex_unlock(&c.mutex_send);
                return c.sock.sendBinary(msg);
            }
            fn isOpen(ctx: *anyopaque) bool {
                return selfOf(ctx).sock.isOpen();
            }
            fn close(ctx: *anyopaque) void {
                selfOf(ctx).sock.close(.normal, "");
            }
            const vtable: WsConn.VTable = .{ .read = read, .send = send, .send_binary = sendBinary, .is_open = isOpen, .close = close };
        };

        /// The lambda passed to `WebSocket` (server-http.cpp:751
        /// @8642541b2): the request translated, the handler run to the end
        /// of the connection. Out of memory here drops the connection, as an
        /// exception out of the C's handler ends its thread's work.
        fn callback(ctx: ?*anyopaque, req: *const httplib.Request, sock: *httplib.ws.WebSocket) void {
            const route: *WsRoute = @ptrCast(@alignCast(ctx.?));
            const a = req.a;
            var impl: ConnImpl = .{ .sock = sock };
            var conn: WsConn = .{ .ctx = &impl, .vtable = &ConnImpl.vtable };
            const request: Req = .{
                .params = getParams(a, req) catch return,
                .headers = getHeaders(a, req) catch return,
                .path = req.path,
                .query_string = buildQueryString(a, req) catch return,
                .body = req.body.items,
                .should_stop = .{ .req = null, .ws = sock },
            };
            route.handler.call(&request, &conn);
        }
    };

    /// The lambda `get`, `post` and `del` bind: the request translated,
    /// the handler called, its response handed to cpp-httplib.
    fn routeHandler(ctx: ?*anyopaque, req: *const httplib.Request, res: *httplib.Response) anyerror!void {
        const route: *Route = @ptrCast(@alignCast(ctx.?));
        const a = req.a;
        const request = try a.create(Req);
        request.* = .{
            .params = try getParams(a, req),
            .headers = try getHeaders(a, req),
            .path = req.path,
            .query_string = try buildQueryString(a, req),
            .body = req.body.items,
            .should_stop = .{ .req = req },
        };
        if (route.kind == .post and try req.isMultipartFormData()) {
            // translate text fields to a JSON object and use it as the body
            var form_json = json.Value.emptyObject();
            defer form_json.deinit(a);
            for (req.form.fields.entries.items) |e| {
                const content = e.value.content.items;
                if (form_json.contains(e.key)) {
                    // if the key already exists, convert it to an array
                    const slot = try form_json.index(a, e.key);
                    if (!slot.isArray()) {
                        const existing_value = try slot.clone(a);
                        var arr = json.Value.emptyArray();
                        try arr.pushBack(a, existing_value);
                        slot.assign(a, arr);
                    }
                    try slot.pushBack(a, try json.Value.initString(a, content));
                } else {
                    try form_json.set(a, e.key, try json.Value.initString(a, content));
                }
            }
            request.body = try json.dump(a, &form_json, -1);

            // populate files from multipart form
            for (req.form.files.entries.items) |e| {
                try request.files.put(a, e.key, .{ .data = e.value.content.items, .filename = e.value.filename, .content_type = e.value.content_type });
            }
        }
        const response = try route.handler.call(request);
        try processHandlerResponse(request, response, res);
    }

    /// Ports `register_gcp_compat` (server-http.cpp:865 @8642541b2), a
    /// member of `server_http_context`: with `AIP_MODE=PREDICTION`, the
    /// health route's alias and a predict route that runs each instance's
    /// `@requestFormat` route on a thread of its own. Call it after every
    /// other route.
    pub fn registerGcpCompat(self: *Context) !void {
        const a = self.arenaAlloc();
        const gcp = try GcpParams.init(a);
        if (!gcp.enabled) return;

        if (self.handlers.get(gcp.path_predict) != null) {
            srvErr("register_gcp_compat", "AIP_PREDICT_ROUTE=%s conflicts with an existing llama-server route\n", .{cz(a, gcp.path_predict)});
            tools_common.cExit(1);
        }

        // camelCase alias -> canonical path (first registration wins on collision)
        const predict = try a.create(Predict);
        predict.* = .{ .ctx = self, .alias_to_path = try libcxx.StringMap([]const u8).init(a) };
        var it = self.handlers.iterator();
        while (it.next()) |e| {
            const path = try a.dupe(u8, e.key);
            _ = try predict.alias_to_path.getOrPut(try pathToGcpFormat(a, path), path);
        }

        if (gcp.path_health.len != 0) {
            const health_handler = self.handlers.get("/health") orelse {
                std.debug.print("GGML_ASSERT(health_handler != handlers.end()) failed\n", .{});
                std.process.abort();
            };
            try self.get(gcp.path_health, health_handler.*);
        }
        try self.post(gcp.path_predict, .{ .ctx = predict, .func = Predict.handle });
    }
};

/// Ports `get_params` (server-http.cpp:603 @8642541b2): the query
/// parameters, the last of a repeated name winning, then the path
/// parameters, decoded, over them.
fn getParams(a: std.mem.Allocator, req: *const httplib.Request) error{OutOfMemory}!StringMap {
    var params: StringMap = .{};
    for (req.params.entries.items) |e| try params.put(a, e.key, e.value);
    for (req.path_params.items) |e| try params.put(a, e.key, try decodePathComponent(a, e.value));
    return params;
}

/// Ports `get_headers` (server-http.cpp:614 @8642541b2): the last of a
/// repeated name winning, names compared exactly.
fn getHeaders(a: std.mem.Allocator, req: *const httplib.Request) error{OutOfMemory}!StringMap {
    var headers: StringMap = .{};
    for (req.headers.entries.items) |e| try headers.put(a, e.key, e.value);
    return headers;
}

/// Ports `build_query_string` (server-http.cpp:622 @8642541b2): the query
/// parameters, re-encoded, in the order sent.
fn buildQueryString(a: std.mem.Allocator, req: *const httplib.Request) error{OutOfMemory}![]u8 {
    var qs: std.ArrayList(u8) = .empty;
    for (req.params.entries.items) |e| {
        if (qs.items.len != 0) try qs.append(a, '&');
        try qs.appendSlice(a, try httplib.detail.encodeQueryComponent(a, e.key));
        try qs.append(a, '=');
        try qs.appendSlice(a, try httplib.detail.encodeQueryComponent(a, e.value));
    }
    return qs.toOwnedSlice(a);
}

/// Ports `decode_path_component` (server-http.cpp:579 @8642541b2): `%XX`
/// only -- path parameters arrive undecoded from cpp-httplib.
fn decodePathComponent(a: std.mem.Allocator, in: []const u8) error{OutOfMemory}![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(a, in.len);
    var i: usize = 0;
    while (i < in.len) : (i += 1) {
        if (in[i] == '%' and i + 2 < in.len) {
            const hex = struct {
                fn f(c: u8) i32 {
                    if (c >= '0' and c <= '9') return c - '0';
                    if (c >= 'a' and c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' and c <= 'F') return c - 'A' + 10;
                    return -1;
                }
            }.f;
            const hi = hex(in[i + 1]);
            const lo = hex(in[i + 2]);
            if (hi >= 0 and lo >= 0) {
                try out.append(a, @intCast((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        try out.append(a, in[i]);
    }
    return out.toOwnedSlice(a);
}

/// Ports `set_headers` (server-http.cpp:570 @8642541b2): in the map's order.
fn setHeaders(res: *httplib.Response, headers: *const StringMap) error{OutOfMemory}!void {
    for (headers.entries.items) |e| try res.setHeader(e.key, e.value);
}

/// What a streamed response's provider and releaser share: the C's two
/// `shared_ptr`s.
const Stream = struct {
    request: *Req,
    response: *Res,

    /// The `chunked_content_provider` lambda: one `next()`, its chunk
    /// written if any, `done` after the last.
    fn provide(ctx: ?*anyopaque, _: usize, sink: *httplib.DataSink) anyerror!bool {
        const self: *Stream = @ptrCast(@alignCast(ctx.?));
        const response = self.response;
        var chunk: std.ArrayList(u8) = .empty;
        const ca = std.heap.c_allocator;
        defer chunk.deinit(ca);
        const has_next = try response.next.?.func(response.next.?.ctx, &chunk, ca);
        if (chunk.items.len != 0) {
            if (!sink.write(chunk.items)) return false;
            const z = try ca.dupeSentinel(u8, chunk.items, 0);
            defer ca.free(z);
            srvDbg(lambda, "http: streamed chunk: %s\n", .{z.ptr});
        }
        if (!has_next) {
            sink.done();
            srvDbg(lambda, "%s", .{@as([*:0]const u8, "http: stream ended\n")});
        }
        return has_next;
    }

    /// The `on_complete` lambda, as the response's releaser: fires
    /// `on_complete`, then lets both go.
    fn release(ctx: ?*anyopaque, _: bool) void {
        const self: *Stream = @ptrCast(@alignCast(ctx.?));
        self.response.onComplete();
        self.response.destroy();
    }
};

/// Ports `process_handler_response` (server-http.cpp:636 @8642541b2): a
/// stream becomes a chunked provider, `X-Accel-Buffering: no` added; a
/// body is set whole and `on_complete` fired at once.
fn processHandlerResponse(request: *Req, response: *Res, res: *httplib.Response) !void {
    if (response.isStream()) {
        res.status = response.status;
        // Tell Nginx to not buffer any streamed response
        try response.setHeader("X-Accel-Buffering", "no");
        try setHeaders(res, &response.headers);
        const content_type = try res.a.dupe(u8, response.content_type);
        const stream = try res.a.create(Stream);
        stream.* = .{ .request = request, .response = response };
        try res.setChunkedContentProvider(content_type, .{ .ctx = stream, .func = Stream.provide }, .{ .ctx = stream, .func = Stream.release });
    } else {
        defer response.destroy();
        res.status = response.status;
        try setHeaders(res, &response.headers);
        try res.setContent(response.data.items, response.content_type);
        response.onComplete();
    }
}

/// The predict route of `register_gcp_compat`: the `post` lambda and what
/// it captures.
const Predict = struct {
    ctx: *Context,
    alias_to_path: libcxx.StringMap([]const u8),

    const max_instances: usize = 128;

    fn badRequest(msg: []const u8) !*Res {
        const res = try Res.create(std.heap.c_allocator);
        errdefer res.destroy();
        const a = res.allocator();
        res.status = 400;
        var err = try formatErrorResponse(a, msg, .invalid_request);
        errdefer err.deinit(a);
        var outer = json.Value.emptyObject();
        defer outer.deinit(a);
        try outer.set(a, "error", err);
        try res.setData(try safeJsonToStr(a, &outer));
        return res;
    }

    /// `build_error`: `{"error": format_error_response(...)}`.
    fn buildError(a: std.mem.Allocator, msg: []const u8, ty: ErrorType) json.Error!json.Value {
        var err = try formatErrorResponse(a, msg, ty);
        errdefer err.deinit(a);
        var outer = json.Value.emptyObject();
        try outer.set(a, "error", err);
        return outer;
    }

    /// Ports `parse_gcp_predict_response` (server-http.cpp:848 @8642541b2):
    /// a route's body as JSON, as a string when it is not JSON, null when
    /// empty; a stream is refused.
    fn parseGcpPredictResponse(a: std.mem.Allocator, res: ?*Res) !json.Value {
        const r = res orelse return throwRuntimeError("empty response from internal handler");
        if (r.isStream()) return throwInvalidArgument("predict route does not support streaming responses");
        if (r.data.items.len == 0) return .null;
        return json.parse(a, r.data.items) catch |e| switch (e) {
            error.JsonError => json.Value.initString(a, r.data.items),
            else => e,
        };
    }

    /// One instance: the `std::async` lambda.
    const Job = struct {
        predict: *const Predict,
        req: *const Req,
        instance: *const json.Value,
        result: json.Value = .null,
        oom: bool = false,

        fn run(job: *Job) void {
            const a = std.heap.c_allocator;
            job.result = job.compute(a) catch {
                job.oom = true;
                return;
            };
        }

        fn compute(job: *Job, a: std.mem.Allocator) error{OutOfMemory}!json.Value {
            const instance = job.instance;
            if (!instance.isObject()) return buildError(a, "each instance must be a JSON object", .invalid_request) catch |e| return toOom(e);
            const fmt_val = instance.at("@requestFormat") catch null;
            if (fmt_val == null or !fmt_val.?.isString()) return buildError(a, "each instance must include a string @requestFormat", .invalid_request) catch |e| return toOom(e);
            return job.dispatch(a) catch |e| switch (e) {
                error.InvalidArgument => buildError(a, what(e).?, .invalid_request) catch |e2| toOom(e2),
                error.OutOfMemory => error.OutOfMemory,
                else => buildError(a, what(e) orelse "unknown error", .server) catch |e2| toOom(e2),
            };
        }

        /// A `json.Error` building an error object can only be memory.
        fn toOom(e: json.Error) error{OutOfMemory} {
            return switch (e) {
                error.OutOfMemory, error.JsonError => error.OutOfMemory,
            };
        }

        fn dispatch(job: *Job, a: std.mem.Allocator) !json.Value {
            const self = job.predict;
            var payload = try job.instance.clone(a);
            defer payload.deinit(a);
            const format = try a.dupe(u8, try (try payload.at("@requestFormat")).get([]const u8));
            try payload.erase(a, "@requestFormat");
            if (payload.contains("stream")) {
                srvWrn(lambda, "%s", .{@as([*:0]const u8, "ignoring client-provided stream field in instance, streaming is not supported in predict route\n")});
                try payload.set(a, "stream", try json.Value.from(a, false));
            }
            // accept both camelCase aliases (e.g. "chatCompletions") and direct paths
            var dispatch_path: []const u8 = undefined;
            if (self.alias_to_path.get(format)) |p| {
                dispatch_path = p.*;
            } else if (self.ctx.handlers.get(format) != null) {
                dispatch_path = format;
            } else {
                const msg = try std.mem.concat(a, u8, &.{ "no handler registered for @requestFormat: ", format });
                return buildError(a, msg, .invalid_request);
            }
            const internal_req: Req = .{
                .params = job.req.params,
                .headers = job.req.headers,
                .path = try std.mem.concat(a, u8, &.{ self.ctx.path_prefix, dispatch_path }),
                .query_string = job.req.query_string,
                .body = try json.dump(a, &payload, -1),
                .should_stop = job.req.should_stop,
            };
            const handler = self.ctx.handlers.get(dispatch_path).?.*;
            const internal_res = try handler.call(&internal_req);
            defer internal_res.destroy();
            return parseGcpPredictResponse(a, internal_res);
        }
    };

    fn handle(ctx: ?*anyopaque, req: *const Req) anyerror!*Res {
        const self: *Predict = @ptrCast(@alignCast(ctx.?));
        const a = std.heap.c_allocator;
        var data = json.parse(a, req.body) catch |e| switch (e) {
            error.JsonError => return badRequest(json.what()),
            else => return e,
        };
        defer data.deinit(a);
        if (!data.isObject()) return badRequest("request body must be a JSON object");
        const instances_v = data.at("instances") catch null;
        if (instances_v == null or !instances_v.?.isArray()) return badRequest("request body must include an array field named instances");
        const instances = instances_v.?;
        if (instances.size() > max_instances) {
            var buf: [96]u8 = undefined;
            return badRequest(std.fmt.bufPrint(&buf, "instances array exceeds maximum size of {d}", .{max_instances}) catch |err| switch (err) {
                error.NoSpaceLeft => unreachable,
            });
        }

        const n = instances.size();
        const jobs = try a.alloc(Job, n);
        defer a.free(jobs);
        const threads = try a.alloc(?std.Thread, n);
        defer a.free(threads);
        for (0..n) |i| {
            jobs[i] = .{ .predict = self, .req = req, .instance = try instances.atIndex(i) };
            threads[i] = std.Thread.spawn(.{}, Job.run, .{&jobs[i]}) catch null;
            if (threads[i] == null) jobs[i].run();
        }
        var predictions = json.Value.emptyArray();
        defer predictions.deinit(a);
        var oom = false;
        for (0..n) |i| {
            if (threads[i]) |t| t.join();
            if (jobs[i].oom) oom = true;
        }
        if (oom) {
            for (jobs) |*j| j.result.deinit(a);
            return error.OutOfMemory;
        }
        for (jobs) |*j| try predictions.pushBack(a, j.result);

        const res = try Res.create(a);
        errdefer res.destroy();
        var outer = json.Value.emptyObject();
        defer outer.deinit(res.allocator());
        try outer.set(res.allocator(), "predictions", try predictions.clone(res.allocator()));
        try res.setData(try safeJsonToStr(res.allocator(), &outer));
        return res;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(httplib);
}

test "a gcp alias drops the version prefix and camel-cases at separators" {
    const a = std.testing.allocator;
    const s = try pathToGcpFormat(a, "/v1/chat/completions");
    defer a.free(s);
    try std.testing.expectEqualStrings("chatCompletions", s);
    const t = try pathToGcpFormat(a, "/slots/:id_slot");
    defer a.free(t);
    try std.testing.expectEqualStrings("slots", t);
}

test "only a localhost origin is localhost" {
    const a = std.testing.allocator;
    try std.testing.expect(try originIsLocalhost(a, "http://localhost:3000"));
    try std.testing.expect(try originIsLocalhost(a, "http://[::1]:80/x"));
    try std.testing.expect(!try originIsLocalhost(a, "http://example.com"));
    try std.testing.expect(!try originIsLocalhost(a, "ftp://localhost"));
    try std.testing.expect(!try originIsLocalhost(a, "http://localhost:abc"));
}
