//! cpp-httplib's `Server`: binding, the accept loop, one connection's
//! keep-alive loop, reading a request, routing it, and writing the
//! response.
//!
//! # Provenance
//!
//! Ports `Server` from `llama.cpp/vendor/cpp-httplib/httplib.cpp` and its
//! declaration in `llama.cpp/vendor/cpp-httplib/httplib.h` (cpp-httplib
//! 0.59.0, vendored at v0.6.0, `d81235049`), as far as
//! `llama.cpp/tools/server/server-http.cpp` reaches it. Each declaration
//! names what it ports and the line it began at.
//!
//! # What is not here
//!
//! `server-http.cpp` registers plain handlers for GET, POST and DELETE and
//! nothing else, so the content-reader handlers, `Put`, `Patch`, `Options`,
//! `CustomRoute` and `WebSocket` routes are empty lists in the C and absent
//! here; the routing code that consults them still runs, finds nothing and
//! falls through as the C's does. It sets no logger, error logger,
//! pre-request, post-routing, `Expect: 100-continue` or start handler, so
//! `output_log` and `output_error_log` do nothing and are not called. It
//! never mounts a directory without `--path`: `set_mount_point` and the
//! file handler are not ported (`Context.init` refuses a `public_path`).
//! A route is a `PathParamsMatcher` or, when its pattern holds a regex
//! metacharacter and no `/:`, a `RegexMatcher` over `src/llama/regex.zig`,
//! the port of the libc++ `std::regex` the C compiles. The web UI's asset
//! routes are the server's regex routes: every asset name holds a `.`, which
//! matches any character, as upstream's does.
//!
//! # Exceptions
//!
//! A handler's thrown exception is an error here. Inside `routing` it goes
//! to the exception handler with the error, which reads its message where
//! the C reads `e.what()`; anywhere else it unwinds to
//! `process_and_close_socket` as the C's reaches `serve_guarded`, and the
//! connection closes.

const std = @import("std");
const detail = @import("detail.zig");
const socket = @import("socket.zig");
const message = @import("message.zig");
const thread_pool = @import("thread_pool.zig");
const llama_regex = @import("llama").regex;
const Regex = llama_regex.Regex;

const Headers = detail.Headers;
const Request = message.Request;
const Response = message.Response;
const SocketStream = socket.SocketStream;
const Socket = socket.Socket;
const ThreadPool = thread_pool.ThreadPool;
const TaskQueue = thread_pool.TaskQueue;
const StatusCode = detail.StatusCode;

/// `Handler`: a route's callback. An error is an exception it threw.
pub const Handler = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, req: *const Request, res: *Response) anyerror!void,
};

/// `HandlerResponse` (httplib.h:2192 @d81235049).
pub const HandlerResponse = enum { handled, unhandled };

/// `HandlerWithResponse`: the pre-routing and error handlers.
pub const HandlerWithResponse = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, req: *const Request, res: *Response) anyerror!HandlerResponse,
};

/// `ExceptionHandler`: called with the error a route raised, where the C
/// passes its `std::exception_ptr`.
pub const ExceptionHandler = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, req: *const Request, res: *Response, err: anyerror) anyerror!void,
};

/// `new_task_queue`: makes the queue `listen_internal` hands connections to.
pub const TaskQueueFactory = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, a: std.mem.Allocator) anyerror!TaskQueue,
};

/// Ports `PathParamsMatcher` (httplib.h:2096 @d81235049): a pattern's
/// literal fragments, and the names of the `:param` segments between them.
const PathParamsMatcher = struct {
    pattern: []const u8,
    static_fragments: std.ArrayList([]const u8) = .empty,
    param_names: std.ArrayList([]const u8) = .empty,

    /// Ports `PathParamsMatcher` (httplib.cpp:7599 @d81235049), the
    /// constructor. A repeated name throws `std::invalid_argument` in the
    /// C; this returns `error.InvalidArgument`.
    fn init(a: std.mem.Allocator, pattern: []const u8) !PathParamsMatcher {
        var self: PathParamsMatcher = .{ .pattern = pattern };
        const marker = "/:";
        var last_param_end: usize = 0;
        while (true) {
            const from = if (last_param_end == 0) last_param_end else last_param_end - 1;
            const marker_pos = std.mem.findPos(u8, pattern, from, marker) orelse break;
            try self.static_fragments.append(a, pattern[last_param_end .. marker_pos + 1]);
            const param_name_start = marker_pos + marker.len;
            const sep_pos = std.mem.findScalarPos(u8, pattern, param_name_start, '/') orelse pattern.len;
            const param_name = pattern[param_name_start..sep_pos];
            for (self.param_names.items) |n| if (std.mem.eql(u8, n, param_name)) return error.InvalidArgument;
            try self.param_names.append(a, param_name);
            last_param_end = sep_pos + 1;
        }
        if (last_param_end < pattern.len) try self.static_fragments.append(a, pattern[last_param_end..]);
        return self;
    }

    /// Ports `match` (httplib.cpp:7648 @d81235049), a member of
    /// `PathParamsMatcher`: fills `path_params` as it goes, so a failed
    /// match can leave some behind, as the C's does.
    fn match(self: *const PathParamsMatcher, req: *Request) error{OutOfMemory}!bool {
        req.path_params.clearRetainingCapacity();
        if (self.param_names.items.len == 0) return std.mem.eql(u8, req.path, self.pattern);
        const path = req.path;
        var starting_pos: usize = 0;
        for (self.static_fragments.items, 0..) |fragment, i| {
            if (starting_pos + fragment.len > path.len) return false;
            // strncmp: a NUL in the path ends the comparison as a mismatch,
            // no fragment holding one
            if (!std.mem.eql(u8, path[starting_pos..][0..fragment.len], fragment)) return false;
            starting_pos += fragment.len;
            if (i >= self.param_names.items.len) continue;
            const sep_pos = std.mem.findScalarPos(u8, path, starting_pos, '/') orelse path.len;
            try req.path_params.append(req.a, .{ .key = self.param_names.items[i], .value = path[starting_pos..sep_pos] });
            starting_pos = sep_pos + 1;
        }
        return starting_pos >= path.len;
    }
};

/// Ports `RegexMatcher` (httplib.h:2125 @d81235049): the whole path
/// against a `std::regex` of the pattern.
const RegexMatcher = struct {
    pattern: []const u8,
    regex: Regex,

    /// `CPPHTTPLIB_REGEX_ROUTE_PATH_MAX_LENGTH` (httplib.h:160 @d81235049).
    const path_max_length = 256;

    /// Ports `match` (httplib.cpp:7695 @d81235049), a member of
    /// `RegexMatcher`: an overlong path is a non-match rather than a risk of
    /// overflowing the stack in `std::regex_match`. The C keeps the
    /// sub-matches in `request.matches`, which no handler of the server's
    /// reads; this drops them.
    fn match(self: *const RegexMatcher, req: *Request) bool {
        req.path_params.clearRetainingCapacity();
        if (req.path.len > path_max_length) return false;
        var m: llama_regex.Match = .init();
        defer m.deinit();
        return self.regex.match(req.path, &m);
    }
};

/// `MatcherBase`'s two implementations.
const Matcher = union(enum) {
    params: PathParamsMatcher,
    regex: RegexMatcher,

    fn pattern(self: *const Matcher) []const u8 {
        return switch (self.*) {
            inline else => |*m| m.pattern,
        };
    }

    fn match(self: *const Matcher, req: *Request) error{OutOfMemory}!bool {
        return switch (self.*) {
            .params => |*m| m.match(req),
            .regex => |*m| m.match(req),
        };
    }

    fn deinit(self: *Matcher, a: std.mem.Allocator) void {
        switch (self.*) {
            .params => |*m| {
                m.static_fragments.deinit(a);
                m.param_names.deinit(a);
            },
            .regex => |*m| m.regex.deinit(),
        }
        a.free(self.pattern());
    }
};

/// One route: its matcher and its handler.
const Route = struct { matcher: Matcher, handler: Handler };

/// Ports `Server` (httplib.h:2185 @d81235049).
pub const Server = struct {
    allocator: std.mem.Allocator,
    new_task_queue: ?TaskQueueFactory = null,

    svr_sock: std.atomic.Value(Socket) = .init(socket.invalid_socket),
    keep_alive_max_count: usize = detail.keepalive_max_count,
    keep_alive_timeout_sec: i64 = detail.keepalive_timeout_second,
    read_timeout_sec: i64 = detail.server_read_timeout_second,
    read_timeout_usec: i64 = 0,
    write_timeout_sec: i64 = detail.server_write_timeout_second,
    write_timeout_usec: i64 = 0,
    idle_interval_sec: i64 = 0,
    idle_interval_usec: i64 = 0,
    payload_max_length: usize = detail.payload_max_length,

    is_running: std.atomic.Value(bool) = .init(false),
    is_decommissioned: std.atomic.Value(bool) = .init(false),
    has_invalid_registration: bool = false,

    get_handlers: std.ArrayList(Route) = .empty,
    post_handlers: std.ArrayList(Route) = .empty,
    delete_handlers: std.ArrayList(Route) = .empty,

    error_handler: ?HandlerWithResponse = null,
    exception_handler: ?ExceptionHandler = null,
    pre_routing_handler: ?HandlerWithResponse = null,

    address_family: c_int = socket.k.AF_UNSPEC,
    tcp_nodelay: bool = detail.tcp_nodelay,
    ipv6_v6only: bool = false,
    socket_options: socket.SocketOptions = .{ .func = socket.defaultSocketOptions },
    default_headers: Headers = .{},

    /// Ports `Server` (httplib.cpp:8233 @d81235049), the constructor:
    /// ignores `SIGPIPE`, as the C does.
    ///
    /// Parameters:
    /// - `allocator`: the server's routes and strings; must be thread-safe,
    ///   since each connection's arena draws on it.
    ///
    /// Return: the server, heap-allocated since its workers point at it.
    pub fn create(allocator: std.mem.Allocator) error{OutOfMemory}!*Server {
        const self = try allocator.create(Server);
        self.* = .{ .allocator = allocator };
        socket.ignoreSigpipe();
        return self;
    }

    /// Frees the server; it must have stopped listening.
    pub fn destroy(self: *Server) void {
        const a = self.allocator;
        for ([_]*std.ArrayList(Route){ &self.get_handlers, &self.post_handlers, &self.delete_handlers }) |list| {
            for (list.items) |*r| r.matcher.deinit(a);
            list.deinit(a);
        }
        for (self.default_headers.entries.items) |e| {
            a.free(e.key);
            a.free(e.value);
        }
        self.default_headers.entries.deinit(a);
        a.destroy(self);
    }

    /// Ports `make_matcher` (httplib.cpp:8246 @d81235049), a member of
    /// `Server`, with `add_handler`: `/:` or no regex metacharacter makes a
    /// `PathParamsMatcher`, anything else a `RegexMatcher`, whose
    /// `std::regex_error` on a malformed pattern is `error.RegexError`.
    fn addHandler(self: *Server, list: *std.ArrayList(Route), pattern: []const u8, handler: Handler) !void {
        const a = self.allocator;
        const owned = try a.dupe(u8, pattern);
        errdefer a.free(owned);
        const matcher: Matcher = if (std.mem.find(u8, pattern, "/:") == null and std.mem.findAny(u8, pattern, ".^$|()[]{}*+?\\") != null)
            .{ .regex = .{ .pattern = owned, .regex = Regex.compile(owned) catch return error.RegexError } }
        else
            .{ .params = try PathParamsMatcher.init(a, owned) };
        try list.append(a, .{ .matcher = matcher, .handler = handler });
    }

    /// Ports `Get` (httplib.cpp:8264 @d81235049), a member of `Server`.
    pub fn get(self: *Server, pattern: []const u8, handler: Handler) !void {
        try self.addHandler(&self.get_handlers, pattern, handler);
    }

    /// Ports `Post` (httplib.cpp:8268 @d81235049), a member of `Server`.
    pub fn post(self: *Server, pattern: []const u8, handler: Handler) !void {
        try self.addHandler(&self.post_handlers, pattern, handler);
    }

    /// Ports `Delete` (httplib.cpp:8298 @d81235049), a member of `Server`.
    pub fn delete(self: *Server, pattern: []const u8, handler: Handler) !void {
        try self.addHandler(&self.delete_handlers, pattern, handler);
    }

    /// Ports `set_error_handler_core` (httplib.cpp:8440 @d81235049), a
    /// member of `Server`: a handler without an answer, which counts as
    /// `Handled` whatever it does.
    pub fn setErrorHandler(self: *Server, handler: Handler) error{OutOfMemory}!void {
        const Wrap = struct {
            h: Handler,
            fn call(ctx: ?*anyopaque, req: *const Request, res: *Response) anyerror!HandlerResponse {
                const w: *@This() = @ptrCast(@alignCast(ctx.?));
                try w.h.func(w.h.ctx, req, res);
                return .handled;
            }
        };
        const w = try self.allocator.create(Wrap);
        w.* = .{ .h = handler };
        self.error_handler = .{ .ctx = w, .func = Wrap.call };
    }

    /// Ports `set_exception_handler` (httplib.cpp:8449 @d81235049), a member
    /// of `Server`.
    pub fn setExceptionHandler(self: *Server, handler: ExceptionHandler) void {
        self.exception_handler = handler;
    }

    /// Ports `set_pre_routing_handler` (httplib.cpp:8454 @d81235049), a
    /// member of `Server`.
    pub fn setPreRoutingHandler(self: *Server, handler: HandlerWithResponse) void {
        self.pre_routing_handler = handler;
    }

    /// Ports `set_address_family` (httplib.cpp:8495 @d81235049), a member of
    /// `Server`.
    pub fn setAddressFamily(self: *Server, family: c_int) void {
        self.address_family = family;
    }

    /// Ports `set_ipv6_v6only` (httplib.cpp:8505 @d81235049), a member of
    /// `Server`: `IPV6_V6ONLY` on an IPv6 listening socket.
    pub fn setIpv6V6only(self: *Server, on: bool) void {
        self.ipv6_v6only = on;
    }

    /// Ports `set_socket_options` (httplib.cpp:8510 @d81235049), a member of
    /// `Server`.
    pub fn setSocketOptions(self: *Server, socket_options: socket.SocketOptions) void {
        self.socket_options = socket_options;
    }

    /// Ports `set_default_headers` (httplib.cpp:8515 @d81235049), a member
    /// of `Server`: every response starts with these. The strings are
    /// copied.
    pub fn setDefaultHeaders(self: *Server, headers: []const Headers.Entry) error{OutOfMemory}!void {
        const a = self.allocator;
        for (headers) |h| _ = try self.default_headers.emplace(a, try a.dupe(u8, h.key), try a.dupe(u8, h.value));
    }

    /// Ports `set_read_timeout` (httplib.cpp:8551 @d81235049), a member of
    /// `Server`.
    pub fn setReadTimeout(self: *Server, sec: i64, usec: i64) void {
        self.read_timeout_sec = sec;
        self.read_timeout_usec = usec;
    }

    /// Ports `set_write_timeout` (httplib.cpp:8557 @d81235049), a member of
    /// `Server`.
    pub fn setWriteTimeout(self: *Server, sec: i64, usec: i64) void {
        self.write_timeout_sec = sec;
        self.write_timeout_usec = usec;
    }

    /// Ports `bind_to_port` (httplib.cpp:8608 @d81235049), a member of
    /// `Server`.
    pub fn bindToPort(self: *Server, host: [:0]const u8, port: c_int) bool {
        const ret = self.bindInternal(host, port, 0);
        if (ret == -1) self.is_decommissioned.store(true, .seq_cst);
        return ret >= 0;
    }

    /// Ports `bind_to_any_port` (httplib.cpp:8614 @d81235049), a member of
    /// `Server`.
    ///
    /// Return: the port bound, or -1.
    pub fn bindToAnyPort(self: *Server, host: [:0]const u8) c_int {
        const ret = self.bindInternal(host, 0, 0);
        if (ret == -1) self.is_decommissioned.store(true, .seq_cst);
        return ret;
    }

    /// Ports `listen_after_bind` (httplib.cpp:8620 @d81235049), a member of
    /// `Server`: the accept loop, until `stop`.
    pub fn listenAfterBind(self: *Server) bool {
        return self.listenInternal();
    }

    /// Ports `wait_until_ready` (httplib.cpp:8629 @d81235049), a member of
    /// `Server`: polls every millisecond.
    pub fn waitUntilReady(self: *const Server) void {
        while (!self.is_running.load(.seq_cst) and !self.is_decommissioned.load(.seq_cst)) socket.sleepUs(1000);
    }

    /// Ports `stop` (httplib.cpp:8635 @d81235049), a member of `Server`:
    /// closes the listening socket, which ends the accept loop.
    pub fn stop(self: *Server) void {
        const sock = self.svr_sock.swap(socket.invalid_socket, .seq_cst);
        if (sock != socket.invalid_socket) {
            _ = socket.shutdownSocket(sock);
            _ = socket.closeSocket(sock);
        }
        self.is_decommissioned.store(false, .seq_cst);
    }

    /// Ports `create_server_socket` (httplib.cpp:9217 @d81235049), a member
    /// of `Server`.
    fn createServerSocket(self: *Server, host: [:0]const u8, port: c_int, socket_flags: c_int) Socket {
        const S = struct {
            fn bindListen(_: ?*anyopaque, sock: Socket, addr: ?*const anyopaque, addrlen: u32, _: *bool) bool {
                socket.bindAndListen(sock, addr, addrlen, detail.listen_backlog) catch return false;
                return true;
            }
        };
        return socket.createSocket(host, port, self.address_family, socket_flags, self.tcp_nodelay, self.ipv6_v6only, self.socket_options, .{ .ctx = null, .func = S.bindListen });
    }

    /// Ports `bind_internal` (httplib.cpp:9236 @d81235049), a member of
    /// `Server`.
    fn bindInternal(self: *Server, host: [:0]const u8, port: c_int, socket_flags: c_int) c_int {
        if (self.is_decommissioned.load(.seq_cst)) return -1;
        if (self.has_invalid_registration) return -1;
        const sock = self.createServerSocket(host, port, socket_flags);
        self.svr_sock.store(sock, .seq_cst);
        if (sock == socket.invalid_socket) return -1;
        if (port == 0) return socket.boundPort(sock) catch -1;
        return port;
    }

    /// Ports `listen_internal` (httplib.cpp:9266 @d81235049), a member of
    /// `Server`: accepts until the listening socket closes, handing each
    /// connection to the task queue with the read and write timeouts and
    /// `TCP_NODELAY` set.
    fn listenInternal(self: *Server) bool {
        if (self.is_decommissioned.load(.seq_cst) or self.svr_sock.load(.seq_cst) == socket.invalid_socket) {
            self.is_decommissioned.store(true, .seq_cst);
            return false;
        }
        var ret = true;
        self.is_running.store(true, .seq_cst);
        defer self.is_running.store(false, .seq_cst);
        {
            const queue = self.makeTaskQueue() catch {
                // the C's `new_task_queue()` throws out of `listen_after_bind`
                self.is_decommissioned.store(true, .seq_cst);
                return false;
            };
            defer queue.destroy();
            while (self.svr_sock.load(.seq_cst) != socket.invalid_socket) {
                if (self.idle_interval_sec > 0 or self.idle_interval_usec > 0) {
                    const val = socket.selectRead(self.svr_sock.load(.seq_cst), self.idle_interval_sec, self.idle_interval_usec);
                    if (val == 0) continue;
                }
                const sock = socket.acceptSocket(self.svr_sock.load(.seq_cst));
                if (sock == socket.invalid_socket) {
                    if (socket.isAcceptResourceError()) {
                        socket.sleepUs(1);
                        continue;
                    } else if (socket.isAcceptTransientError()) {
                        continue;
                    }
                    const listen_sock = self.svr_sock.swap(socket.invalid_socket, .seq_cst);
                    if (listen_sock != socket.invalid_socket) {
                        _ = socket.closeSocket(listen_sock);
                        ret = false;
                    }
                    break;
                }
                _ = socket.setSocketOptTime(sock, socket.k.SOL_SOCKET, socket.k.SO_RCVTIMEO, self.read_timeout_sec, self.read_timeout_usec);
                _ = socket.setSocketOptTime(sock, socket.k.SOL_SOCKET, socket.k.SO_SNDTIMEO, self.write_timeout_sec, self.write_timeout_usec);
                if (self.tcp_nodelay) _ = socket.setSocketOpt(sock, socket.k.IPPROTO_TCP, socket.k.TCP_NODELAY, 1);
                if (!queue.enqueue(.{ .ctx = self, .arg = @intCast(sock), .run = processAndCloseSocketTask })) {
                    _ = socket.shutdownSocket(sock);
                    _ = socket.closeSocket(sock);
                }
            }
            queue.shutdown();
        }
        self.is_decommissioned.store(!ret, .seq_cst);
        return ret;
    }

    /// `new_task_queue()`, or the C's default pool:
    /// `CPPHTTPLIB_THREAD_POOL_COUNT` threads, four times that at most.
    fn makeTaskQueue(self: *Server) !TaskQueue {
        if (self.new_task_queue) |f| return f.func(f.ctx, self.allocator);
        const hw = socket.hardwareConcurrency();
        const n: usize = @max(8, if (hw > 0) hw - 1 else 0);
        return (try ThreadPool.create(self.allocator, n, n * 4)).taskQueue();
    }

    fn processAndCloseSocketTask(ctx: *anyopaque, arg: usize) void {
        const self: *Server = @ptrCast(@alignCast(ctx));
        _ = self.processAndCloseSocket(@intCast(arg));
    }

    /// Ports `process_and_close_socket` (httplib.cpp:10098 @d81235049), a
    /// member of `Server`, with `serve_guarded` (httplib.h:2351
    /// @d81235049): an error escaping a request ends the connection, as an
    /// exception escaping one does.
    fn processAndCloseSocket(self: *Server, sock: Socket) bool {
        var addrs = std.heap.ArenaAllocator.init(self.allocator);
        defer addrs.deinit();
        var remote_addr: []const u8 = "";
        var remote_port: c_int = 0;
        socket.getRemoteIpAndPort(addrs.allocator(), sock, &remote_addr, &remote_port) catch |err| switch (err) {
            error.OutOfMemory => {},
        };
        var local_addr: []const u8 = "";
        var local_port: c_int = 0;
        socket.getLocalIpAndPort(addrs.allocator(), sock, &local_addr, &local_port) catch |err| switch (err) {
            error.OutOfMemory => {},
        };
        const ret = self.processServerSocket(sock, remote_addr, remote_port, local_addr, local_port) catch false;
        socket.drainAndCloseSocket(sock);
        return ret;
    }

    /// Ports `process_server_socket` (httplib.cpp:1941 @d81235049) and
    /// `process_server_socket_core` (httplib.cpp:1916 @d81235049): up to
    /// `keep_alive_max_count` requests on one `SocketStream`, while the peer
    /// keeps sending; the last is answered with `Connection: close`. A
    /// pipelined request already in the stream's buffer is served without
    /// waiting for the socket, whose bytes it no longer holds.
    fn processServerSocket(self: *Server, sock: Socket, remote_addr: []const u8, remote_port: c_int, local_addr: []const u8, local_port: c_int) !bool {
        // One stream per connection: its read buffer can already hold the start of
        // the next, pipelined request.
        var strm: SocketStream = .{
            .sock = sock,
            .read_timeout_sec = self.read_timeout_sec,
            .read_timeout_usec = self.read_timeout_usec,
            .write_timeout_sec = self.write_timeout_sec,
            .write_timeout_usec = self.write_timeout_usec,
        };
        var ret = false;
        var count = self.keep_alive_max_count;
        while (count > 0) {
            var socket_readable = false;
            if (!strm.isReadable()) {
                if (!socket.keepAlive(&self.svr_sock, sock, self.keep_alive_timeout_sec)) break;
                socket_readable = true;
            }
            const close_connection = count == 1;
            var connection_closed = false;
            if (socket_readable) strm.readable_hint = true;
            ret = try self.processRequest(&strm, remote_addr, remote_port, local_addr, local_port, close_connection, &connection_closed);
            if (!ret or connection_closed) break;
            count -= 1;
        }
        return ret;
    }

    /// Ports `builtin_methods` (httplib.cpp:8312 @d81235049), a member of
    /// `Server`.
    fn isBuiltinMethod(m: []const u8) bool {
        inline for (.{ "GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH", "PRI" }) |b| {
            if (std.mem.eql(u8, m, b)) return true;
        }
        return false;
    }

    /// Ports `parse_request_line` (httplib.cpp:8649 @d81235049), a member of
    /// `Server`: three space-separated parts ending in CRLF, a built-in
    /// method, HTTP/1.0 or 1.1, a target of visible characters, the
    /// fragment dropped, the path decoded and the query parsed.
    fn parseRequestLine(s: []const u8, req: *Request) !bool {
        const a = req.a;
        if (s.len < 2 or s[s.len - 2] != '\r' or s[s.len - 1] != '\n') return false;
        const line = s[0 .. s.len - 2];
        const Ctx = struct {
            req: *Request,
            count: usize = 0,
            pub fn call(self: *@This(), part: []const u8) error{}!void {
                switch (self.count) {
                    0 => self.req.method = part,
                    1 => self.req.target = part,
                    2 => self.req.version = part,
                    else => {},
                }
                self.count += 1;
            }
        };
        var ctx: Ctx = .{ .req = req };
        detail.split(line, ' ', std.math.maxInt(usize), &ctx) catch |err| switch (err) {};
        if (ctx.count != 3) return false;
        req.method = try a.dupe(u8, req.method);
        req.target = try a.dupe(u8, req.target);
        req.version = try a.dupe(u8, req.version);
        if (!isBuiltinMethod(req.method)) return false;
        if (!std.mem.eql(u8, req.version, "HTTP/1.1") and !std.mem.eql(u8, req.version, "HTTP/1.0")) return false;
        if (!detail.isRequestTarget(req.target)) return false;
        if (std.mem.findScalar(u8, req.target, '#')) |i| req.target = req.target[0..i];
        const d = detail.divide(req.target, '?');
        req.path = try detail.decodePathComponent(a, d.lhs);
        try detail.parseQueryText(a, d.rhs, &req.params);
        return true;
    }

    /// Ports `process_request` (httplib.cpp:9736 @d81235049), a member of
    /// `Server`: one request off `strm` and its response back. `100
    /// Continue` is owed here and sent only when the body is read
    /// (`read_content_core`). A response that says `Connection: close`
    /// ends the connection, whichever path wrote it.
    ///
    /// Return: whether the connection may carry another request; an error
    /// is an exception escaping outside `routing`.
    fn processRequest(self: *Server, strm: *SocketStream, remote_addr: []const u8, remote_port: c_int, local_addr: []const u8, local_port: c_int, close_connection: bool, connection_closed: *bool) !bool {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var buf: [2048]u8 = undefined;
        var line_reader = socket.LineReader.init(a, strm, &buf);
        if (!try line_reader.getline()) return false;

        // RFC 9112 2.2: ignore an empty line received before the request-line. Some
        // clients send an extra CRLF after a request body, which would otherwise be
        // parsed as the next request on a persistent connection.
        if (std.mem.eql(u8, line_reader.cstr(), "\r\n") and !try line_reader.getline()) return false;

        var req: Request = .{ .a = a, .remote_addr = remote_addr, .remote_port = remote_port, .local_addr = local_addr, .local_port = local_port };
        var res: Response = .{ .a = a, .headers = try self.default_headers.clone(a) };
        defer res.deinit();

        // RFC 9112 §9.6: a server that sends the "close" connection option must
        // close the connection after that response, whichever path wrote it (an
        // error status, a handler, or a rejected request).
        defer if (detail.hasHeaderToken(&res.headers, "Connection", "close")) {
            connection_closed.* = true;
        };

        // A rejected message leaves the rest of it (and any body) unread, so the
        // connection cannot be reused.
        if (!try parseRequestLine(line_reader.cstr(), &req)) {
            connection_closed.* = true;
            res.status = StatusCode.bad_request_400;
            return self.writeResponse(strm, close_connection, &req, &res);
        }
        if (!try message.readHeaders(a, strm, &req.headers)) {
            connection_closed.* = true;
            res.status = StatusCode.bad_request_400;
            return self.writeResponse(strm, close_connection, &req, &res);
        }
        // a Content-Length that is not a valid decimal length ("42, 42", "+42",
        // empty) would otherwise be read as "no body"
        var is_invalid_content_length = false;
        _ = detail.getHeaderValueU64(&req.headers, "Content-Length", 0, &is_invalid_content_length);
        if (is_invalid_content_length or
            detail.hasConflictingContentLength(&req.headers) or
            (req.hasHeader("Transfer-Encoding") and !detail.isChunkedTransferEncoding(&req.headers)))
        {
            connection_closed.* = true;
            res.status = StatusCode.bad_request_400;
            return self.writeResponse(strm, close_connection, &req, &res);
        }
        if (req.target.len > detail.request_uri_max_length) {
            connection_closed.* = true;
            res.status = StatusCode.uri_too_long_414;
            return self.writeResponse(strm, close_connection, &req, &res);
        }
        if (detail.hasHeaderToken(&req.headers, "Connection", "close")) connection_closed.* = true;
        if (std.mem.eql(u8, req.version, "HTTP/1.0") and !detail.hasHeaderToken(&req.headers, "Connection", "keep-alive")) connection_closed.* = true;

        // no trusted proxies are set, so X-Forwarded-For is never honoured
        req.remote_addr = remote_addr;
        req.remote_port = remote_port;
        req.local_addr = local_addr;
        req.local_port = local_port;

        if (req.hasHeader("Accept")) {
            const accept_header = try detail.getCombinedHeaderValue(a, &req.headers, "Accept");
            if (!try detail.parseAcceptHeader(a, accept_header)) {
                connection_closed.* = true;
                res.status = StatusCode.bad_request_400;
                return self.writeResponse(strm, close_connection, &req, &res);
            }
        }
        if (req.hasHeader("Range")) {
            const range_header_value = req.getHeaderValue("Range");
            if (!try detail.parseRangeHeader(a, range_header_value, &req.ranges)) {
                connection_closed.* = true;
                res.status = StatusCode.range_not_satisfiable_416;
                return self.writeResponse(strm, close_connection, &req, &res);
            }
        }

        // `100 Continue` itself is deferred until the body is actually read (see
        // read_content_core), so a request rejected by a later handler never
        // invites the client to send a body nobody will read.
        if (!std.mem.eql(u8, req.version, "HTTP/1.0") and detail.hasHeaderToken(&req.headers, "Expect", "100-continue")) {
            // no Expect: 100-continue handler is set, so the status is 100
            req.expect_100_continue_pending = true;
        }

        req.sock = strm.sock;

        if (detail.isWebsocketUpgrade(req.method, &req.headers)) {
            if (self.pre_routing_handler) |h| {
                if (try h.func(h.ctx, &req, &res) == .handled) {
                    if (res.status == -1) res.status = StatusCode.ok_200;
                    return self.writeResponseCore(strm, close_connection, &req, &res, true);
                }
            }
            // no WebSocket handler is registered: on to routing and its 404
        }

        var routed = false;
        if (self.routing(&req, &res, strm)) |r| {
            routed = r;
        } else |err| {
            if (self.exception_handler) |h| {
                try h.func(h.ctx, &req, &res, err);
                routed = true;
            } else {
                res.status = StatusCode.internal_server_error_500;
            }
        }

        var ret = false;
        if (routed) {
            if (res.status == -1) res.status = if (req.ranges.items.len == 0) StatusCode.ok_200 else StatusCode.partial_content_206;
            if (try message.rangeError(&req, &res)) {
                res.body.clearRetainingCapacity();
                res.content_length = 0;
                res.content_provider = null;
                res.status = StatusCode.range_not_satisfiable_416;
                ret = try self.writeResponse(strm, close_connection, &req, &res);
            } else {
                ret = try self.writeResponseCore(strm, close_connection, &req, &res, true);
            }
        } else {
            if (res.status == -1) res.status = StatusCode.not_found_404;
            ret = try self.writeResponse(strm, close_connection, &req, &res);
        }

        if (!req.body_consumed and message.hasFramedBody(&req) and
            !detail.hasHeaderToken(&res.headers, "Connection", "close"))
        {
            const Discard = struct {
                pub fn receive(_: @This(), _: []const u8) !bool {
                    return true;
                }
            };
            var dummy_status: c_int = 0;
            if (!try message.readContent(strm, &req, self.payload_max_length, &dummy_status, Discard{}, false)) connection_closed.* = true;
        }
        return ret;
    }

    /// Ports `routing` (httplib.cpp:9360 @d81235049), a member of `Server`:
    /// the pre-routing handler, then the route table for the method.
    fn routing(self: *Server, req: *Request, res: *Response, strm: *SocketStream) !bool {
        if (self.pre_routing_handler) |h| {
            if (try h.func(h.ctx, req, res) == .handled) return true;
        }
        // no mount point, so no file is served; no content-reader route is
        // registered, so expect_content's branch dispatches nothing
        const m = req.method;
        if (std.mem.eql(u8, m, "GET") or std.mem.eql(u8, m, "HEAD")) return self.dispatchRequest(req, res, self.get_handlers.items, strm);
        if (std.mem.eql(u8, m, "POST")) return self.dispatchRequest(req, res, self.post_handlers.items, strm);
        if (std.mem.eql(u8, m, "PUT")) return false;
        if (std.mem.eql(u8, m, "DELETE")) return self.dispatchRequest(req, res, self.delete_handlers.items, strm);
        if (std.mem.eql(u8, m, "OPTIONS")) return false;
        if (std.mem.eql(u8, m, "PATCH")) return false;
        res.status = StatusCode.bad_request_400;
        return false;
    }

    /// Ports `dispatch_request` (httplib.cpp:9473 @d81235049), a member of
    /// `Server`: the first route that matches, its body read only then.
    fn dispatchRequest(self: *Server, req: *Request, res: *Response, handlers: []const Route, strm: *SocketStream) !bool {
        for (handlers) |*x| {
            if (try x.matcher.match(req)) {
                req.matched_route = x.matcher.pattern();
                if (message.expectContent(req) and !try self.readContent(strm, req, res)) return false;
                try x.handler.func(x.handler.ctx, req, res);
                return true;
            }
        }
        return false;
    }

    /// Ports `read_content` (httplib.cpp:8872 @d81235049), a member of
    /// `Server`: the body into `req.body` -- or, for `multipart/form-data`,
    /// into `req.form` -- and a form-urlencoded body's pairs into
    /// `req.params`.
    fn readContent(self: *Server, strm: *SocketStream, req: *Request, res: *Response) !bool {
        const Sink = struct {
            server: *Server,
            req: *Request,
            count: usize = 0,
            is_text_field: bool = false,
            cur: usize = 0,

            pub fn receive(s: *@This(), d: []const u8) !bool {
                const body = &s.req.body;
                const pml = s.server.payload_max_length;
                if (pml > 0 and (body.items.len >= pml or d.len > pml - body.items.len)) return false;
                try body.appendSlice(s.req.a, d);
                return true;
            }

            pub fn header(s: *@This(), file: *const message.FormData) !bool {
                const n = s.count;
                s.count += 1;
                if (n == detail.multipart_form_data_file_max_count) return false;
                const a = s.req.a;
                if (file.filename.len == 0) {
                    s.cur = try s.req.form.fields.emplace(a, file.name, .{ .name = file.name, .headers = try file.headers.clone(a) });
                    s.is_text_field = true;
                } else {
                    var copy = file.*;
                    copy.headers = try file.headers.clone(a);
                    copy.content = .empty;
                    s.cur = try s.req.form.files.emplace(a, file.name, copy);
                    s.is_text_field = false;
                }
                return true;
            }

            pub fn content(s: *@This(), d: []const u8) !bool {
                const a = s.req.a;
                if (s.is_text_field) {
                    try s.req.form.fields.entries.items[s.cur].value.content.appendSlice(a, d);
                } else {
                    try s.req.form.files.entries.items[s.cur].value.content.appendSlice(a, d);
                }
                return true;
            }
        };
        var sink: Sink = .{ .server = self, .req = req };
        if (try self.readContentCore(strm, req, res, &sink)) {
            const content_type = req.getHeaderValue("Content-Type");
            if (std.mem.eql(u8, try detail.extractMediaType(req.a, content_type, null), "application/x-www-form-urlencoded")) {
                if (req.body.items.len > detail.form_url_encoded_payload_max_length) {
                    res.status = StatusCode.payload_too_large_413;
                    return false;
                }
                try detail.parseQueryText(req.a, req.body.items, &req.params);
            }
            return true;
        }
        return false;
    }

    /// Ports `read_content_core` (httplib.cpp:8956 @d81235049), a member of
    /// `Server`: a body framed by chunks or `Content-Length`, after the owed
    /// `100 Continue`; with neither, none -- unless the connection will not
    /// persist and bytes are already waiting, which are then read to the end
    /// of the stream so the payload limit still applies.
    fn readContentCore(self: *Server, strm: *SocketStream, req: *Request, res: *Response, sink: anytype) !bool {
        var parser: message.FormDataParser = .{ .a = req.a };
        const multipart = try req.isMultipartFormData();
        if (multipart) {
            const boundary = try detail.parseMultipartBoundary(req.a, req.getHeaderValue("Content-Type")) orelse {
                res.status = StatusCode.bad_request_400;
                return false;
            };
            try parser.setBoundary(boundary);
        }
        const Out = struct {
            multipart: bool,
            parser: *message.FormDataParser,
            sink: @TypeOf(sink),
            pub fn receive(o: @This(), d: []const u8) !bool {
                if (o.multipart) return o.parser.parse(d, o.sink);
                return o.sink.receive(d);
            }
        };
        const out: Out = .{ .multipart = multipart, .parser = &parser, .sink = sink };

        if (!req.hasHeader("Content-Length") and !detail.isChunkedTransferEncoding(&req.headers)) {
            if (!message.isConnectionPersistent(req) and self.payload_max_length > 0 and self.payload_max_length < std.math.maxInt(usize)) {
                var has_data = strm.isReadable();
                if (!has_data) has_data = socket.selectRead(strm.sock, 0, 0) > 0;
                if (has_data) return message.readContent(strm, req, self.payload_max_length, &res.status, out, true);
            }
            return true;
        }

        // The client is waiting for this before it sends the body.
        if (req.expect_100_continue_pending) {
            req.expect_100_continue_pending = false;
            _ = message.writeResponseLine(strm, StatusCode.continue_100);
            _ = strm.write("\r\n");
        }

        if (!try message.readContent(strm, req, self.payload_max_length, &res.status, out, true)) return false;
        req.body_consumed = true;
        if (multipart and !parser.is_valid) {
            res.status = StatusCode.bad_request_400;
            return false;
        }
        return true;
    }

    /// Ports `write_response` (httplib.cpp:8708 @d81235049), a member of
    /// `Server`: a response whose ranges, if any, do not apply.
    fn writeResponse(self: *Server, strm: *SocketStream, close_connection: bool, req: *Request, res: *Response) !bool {
        req.ranges.clearRetainingCapacity();
        return self.writeResponseCore(strm, close_connection, req, res, false);
    }

    /// Ports `apply_ranges` (httplib.cpp:9575 @d81235049), a member of
    /// `Server`: a 206's range or ranges cut from the body, and the framing
    /// headers. No content coding is compiled in, so none is applied.
    fn applyRanges(req: *const Request, res: *Response) !void {
        const a = res.a;
        var content_type: []const u8 = "";
        var boundary: []const u8 = "";
        if (req.ranges.items.len > 1 and res.status == StatusCode.partial_content_206) {
            if (res.headers.find("Content-Type")) |i| {
                content_type = res.headers.entries.items[i].value;
                _ = res.headers.entries.orderedRemove(i);
            }
            boundary = try message.makeMultipartDataBoundary(a);
            try res.setHeader("Content-Type", try std.mem.concat(a, u8, &.{ "multipart/byteranges; boundary=", boundary }));
        }
        if (res.body.items.len == 0) {
            if (res.content_provider != null and res.is_chunked_content_provider) try res.setHeader("Transfer-Encoding", "chunked");
        } else {
            if (req.ranges.items.len == 0 or res.status != StatusCode.partial_content_206) {
                // the whole body
            } else if (req.ranges.items.len == 1) {
                const ol = message.getRangeOffsetAndLength(req.ranges.items[0]);
                try res.setHeader("Content-Range", try message.makeContentRangeHeaderField(a, ol.offset, ol.length, res.body.items.len));
                const sliced = try a.dupe(u8, res.body.items[ol.offset..][0..ol.length]);
                res.body.clearRetainingCapacity();
                try res.body.appendSlice(a, sliced);
            } else {
                const data = try message.makeMultipartRangesData(a, req.ranges.items, boundary, content_type, res.body.items);
                res.body.clearRetainingCapacity();
                try res.body.appendSlice(a, data);
            }
            res.content_length = res.body.items.len;
            try res.setHeader("Content-Length", try std.fmt.allocPrint(a, "{d}", .{res.content_length}));
        }
    }

    /// Ports `write_response_core` (httplib.cpp:8723 @d81235049), a member
    /// of `Server`: the error handler for a status from 400, the ranges, the
    /// connection and framing headers, the head and a body under
    /// `CPPHTTPLIB_SEND_BUFSIZ` in one write, a larger one in a second, then
    /// a streamed body. The small-file arm is not live: only
    /// `set_file_content_provider` sets `is_file_content_provider_`, and no
    /// mount point or file content is ever set here. A body the client still withholds for a
    /// `100 Continue` never sent closes the connection.
    fn writeResponseCore(self: *Server, strm: *SocketStream, close_connection: bool, req: *const Request, res: *Response, need_apply_ranges_in: bool) !bool {
        const a = res.a;
        var need_apply_ranges = need_apply_ranges_in;
        if (400 <= res.status) {
            if (self.error_handler) |h| {
                if (try h.func(h.ctx, req, res) == .handled) need_apply_ranges = true;
            }
        }
        if (need_apply_ranges) try applyRanges(req, res);

        if (close_connection or detail.hasHeaderToken(&req.headers, "Connection", "close") or
            400 <= res.status or // Don't leave connections open after errors
            // The client withholds the body until `100 Continue`, which was never
            // sent, so whether and when the body follows is unknown.
            (req.expect_100_continue_pending and message.hasFramedBody(req)))
        {
            try res.setHeader("Connection", "close");
        } else {
            try res.setHeader("Keep-Alive", try std.fmt.allocPrint(a, "timeout={d}, max={d}", .{ self.keep_alive_timeout_sec, self.keep_alive_max_count }));
        }
        if ((res.body.items.len != 0 or res.content_length > 0 or res.content_provider != null) and !res.hasHeader("Content-Type")) {
            try res.setHeader("Content-Type", "text/plain");
        }
        if (res.body.items.len == 0 and res.content_length == 0 and res.content_provider == null and !res.hasHeader("Content-Length")) {
            try res.setHeader("Content-Length", "0");
        }
        const is_head = std.mem.eql(u8, req.method, "HEAD");
        if (is_head and !res.hasHeader("Accept-Ranges")) try res.setHeader("Accept-Ranges", "bytes");

        var bstrm: socket.BufferStream = .{ .a = a };
        if (message.writeResponseLine(&bstrm, res.status) < 0) return false;
        if (try message.writeHeaders(a, &bstrm, &res.headers) <= 0) return false;
        // Combine a small body with the headers so the whole response leaves in a
        // single write. A large body is written on its own instead.
        const send_body = !is_head;
        var body_is_separate = false;
        if (send_body and res.body.items.len != 0 and res.content_provider == null) {
            if (res.body.items.len < detail.send_bufsiz) {
                _ = bstrm.write(res.body.items);
            } else {
                body_is_separate = true;
            }
        }
        if (bstrm.oom) return error.OutOfMemory;

        if (!message.writeData(strm, bstrm.buffer.items)) return false;

        if (body_is_separate) return message.writeData(strm, res.body.items);

        var ret = true;
        if (send_body) {
            if (res.content_provider) |provider| {
                // Ports `write_content_with_provider` (httplib.cpp:8822
                // @d81235049). Every provider here is chunked; it has a
                // length only when the error handler gave the response a
                // body, which no 206 has, so the ranged arms stay out
                const ok = if (res.content_length > 0)
                    try message.writeContent(strm, provider, 0, res.content_length, &self.svr_sock)
                else
                    try message.writeContentChunked(a, strm, provider, &self.svr_sock);
                if (ok) {
                    res.content_provider_success = true;
                } else {
                    ret = false;
                }
            }
        }
        return ret;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a path-params pattern captures each segment by name" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const m = try PathParamsMatcher.init(aa, "/slots/:id_slot/x");
    var req: Request = .{ .a = aa, .path = "/slots/3/x" };
    try std.testing.expect(try m.match(&req));
    try std.testing.expectEqualStrings("id_slot", req.path_params.items[0].key);
    try std.testing.expectEqualStrings("3", req.path_params.items[0].value);
    req.path = "/slots/3/y";
    try std.testing.expect(!try m.match(&req));
}
