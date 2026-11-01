//! A response that forwards a request to another HTTP server and relays
//! its answer as it arrives: the CORS proxy's (`cors_proxy.zig`), and
//! router mode's once it is ported.
//!
//! # Provenance
//!
//! Ports `server_http_proxy` from `llama.cpp/tools/server/server-models.cpp`
//! and `llama.cpp/tools/server/server-models.h` (llama.cpp v0.4.1,
//! `b29c606e2`), with the helpers it calls: `to_lower_copy`,
//! `should_strip_proxy_header`, `generate_multipart_boundary` and
//! `build_multipart_body`. Each declaration names what it ports and the
//! line it began at.
//!
//! # Its shape
//!
//! The C++ shares the client and the pipe between the response and a
//! detached thread through `shared_ptr`s; here they live in a `Shared`
//! with a reference count, so whichever of the two ends last frees them.
//! A `msg_t` owns its strings. The client is `httplib/client.zig`'s.

const std = @import("std");
const json = @import("json");
const http = @import("http.zig");
const common = @import("common.zig");
const httplib_client = @import("httplib/client.zig");
const libcxx = @import("llama").libcxx;

const Value = json.Value;
const Allocator = std.mem.Allocator;
const srv = common.srv;
const throw = common.throw;

const gpa = std.heap.c_allocator;

/// Ports `server_http_proxy`'s `msg_t` (server-models.h:406 @b29c606e2): the
/// headers (first message only) or a chunk of the body.
const Msg = struct {
    headers: http.StringMap = .{}, // a std::map: sorted, unique keys
    status: c_int = 0,
    data: []u8 = &.{},
    content_type: []u8 = &.{},

    fn deinit(self: *Msg) void {
        for (self.headers.entries.items) |e| {
            gpa.free(e.key);
            gpa.free(e.value);
        }
        self.headers.entries.deinit(gpa);
        gpa.free(self.data);
        gpa.free(self.content_type);
    }
};

/// What the response and the client thread share.
const Shared = struct {
    pipe: common.Pipe(Msg),
    refs: std.atomic.Value(u32) = .init(2),
    // true once response_handler has already forwarded the headers
    headers_sent: std.atomic.Value(bool) = .init(false),

    fn release(self: *Shared) void {
        if (self.refs.fetchSub(1, .seq_cst) != 1) return;
        self.pipe.closeWrite();
        var m: Msg = undefined;
        while (self.pipe.read(&m, null, false)) m.deinit();
        self.pipe.deinit();
        gpa.destroy(self);
    }
};

/// Ports `to_lower_copy` (server-models.cpp:2333 @b29c606e2).
fn toLowerCopy(a: Allocator, value: []const u8) ![]u8 {
    return std.ascii.allocLowerString(a, value);
}

/// Ports `should_strip_proxy_header` (server-models.cpp:2339 @b29c606e2): headers
/// the router would duplicate.
fn shouldStripProxyHeader(header_name: []const u8) bool {
    // Headers that get duplicated when router forwards child responses
    if (std.mem.eql(u8, header_name, "server") or
        std.mem.eql(u8, header_name, "transfer-encoding") or
        std.mem.eql(u8, header_name, "content-length") or // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        std.mem.eql(u8, header_name, "keep-alive"))
    {
        return true;
    }
    // Router injects CORS, child also sends them: duplicate
    return std.mem.startsWith(u8, header_name, "access-control-");
}

/// Ports `generate_multipart_boundary` (server-models.cpp:2356 @b29c606e2): 16
/// characters from a `std::mt19937` seeded by `std::random_device`.
fn generateMultipartBoundary(a: Allocator) ![]u8 {
    const chars = "0123456789abcdefghijklmnopqrstuvwxyz";
    var gen = libcxx.Mt19937.init(common.randomDevice());
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "----llama-cpp-proxy-");
    // the C++ draws through std::uniform_int_distribution; the seed is the
    // random device's, so no draw is reproducible either way
    for (0..16) |_| try out.append(a, chars[gen.next() % chars.len]);
    return out.items;
}

/// `sanitize_field`: a form field's name, file name or type without CR,
/// LF or double quotes.
fn sanitizeField(a: Allocator, text: []const u8) ![]u8 {
    var result: std.ArrayList(u8) = .empty;
    for (text) |c| {
        if (c != '\n' and c != '\r' and c != '"') try result.append(a, c);
    }
    return result.items;
}

/// Ports `build_multipart_body` (server-models.cpp:2367 @b29c606e2): the form
/// fields (from the JSON the HTTP layer made of them) and the files,
/// re-encoded as `multipart/form-data`.
fn buildMultipartBody(a: Allocator, form_fields: *Value, files: *const http.SortedMap(http.UploadedFile), boundary: []const u8) ![]u8 {
    var body: std.ArrayList(u8) = .empty;
    var it = form_fields.items();
    while (it.next()) |item| {
        const key = try sanitizeField(a, item.key);
        const value = item.value;
        if (value.isArray()) {
            for (value.array.items) |*v| {
                try body.print(a, "--{s}\r\nContent-Disposition: form-data; name=\"{s}\"\r\n\r\n", .{ boundary, key });
                if (!v.isString()) return throw(error.InvalidArgument, "expected string", .{});
                try body.print(a, "{s}\r\n", .{v.string});
            }
        } else {
            try body.print(a, "--{s}\r\nContent-Disposition: form-data; name=\"{s}\"\r\n\r\n", .{ boundary, key });
            if (!value.isString()) return throw(error.InvalidArgument, "expected string", .{});
            try body.print(a, "{s}\r\n", .{value.string});
        }
    }
    for (files.entries.items) |e| {
        try body.print(a, "--{s}\r\nContent-Disposition: form-data; name=\"{s}\"", .{ boundary, try sanitizeField(a, e.key) });
        if (e.value.filename.len != 0) try body.print(a, "; filename=\"{s}\"", .{try sanitizeField(a, e.value.filename)});
        try body.appendSlice(a, "\r\n");
        if (e.value.content_type.len != 0) {
            try body.print(a, "Content-Type: {s}\r\n", .{try sanitizeField(a, e.value.content_type)});
        } else {
            try body.appendSlice(a, "Content-Type: application/octet-stream\r\n");
        }
        try body.appendSlice(a, "\r\n");
        try body.appendSlice(a, e.value.data);
        try body.appendSlice(a, "\r\n");
    }
    try body.print(a, "--{s}--\r\n", .{boundary});
    return body.items;
}

/// `common_http_format_host` (common/http.h:25 @b29c606e2).
const formatHost = http.commonHttpFormatHost;

/// Ports `server_http_proxy` (server-models.h:381 @b29c606e2): the response
/// whose `next` reads the pipe the client thread fills.
pub const Proxy = struct {
    base: http.Res,
    shared: *Shared,
    should_stop: http.ShouldStop,
    /// `std::function<void()> cleanup`, set by its owner after construction.
    cleanup: ?Cleanup = null,

    /// `std::function<void()>`.
    pub const Cleanup = struct { ctx: ?*anyopaque, func: *const fn (?*anyopaque) void };

    const vtable: http.Res.VTable = .{ .on_complete = http.Res.noComplete, .destroy = destroyProxy };

    /// Ports `server_http_proxy`'s destructor (server-models.h:395 @b29c606e2): the pipe
    /// closed both ways, then `cleanup`.
    fn destroyProxy(res: *http.Res) void {
        const self: *Proxy = @fieldParentPtr("base", res);
        // cleanup_pipes
        self.shared.pipe.closeRead();
        self.shared.pipe.closeWrite();
        if (self.cleanup) |c| c.func(c.ctx);
        self.shared.release();
        self.base.arena.deinit();
        gpa.destroy(self);
    }

    fn stopFn(ctx: *anyopaque) bool {
        const s: *const http.ShouldStop = @ptrCast(@alignCast(ctx));
        return s.call();
    }

    fn stopper(self: *Proxy) ?common.Pipe(Msg).ShouldStop {
        return .{ .ctx = &self.should_stop, .f = stopFn };
    }

    /// The `next` lambda: one message's data; false at EOF or a broken
    /// pipe.
    fn next(ctx: ?*anyopaque, out: *std.ArrayList(u8), a: Allocator) anyerror!bool {
        const self: *Proxy = @ptrCast(@alignCast(ctx.?));
        var msg: Msg = .{};
        const has_next = self.shared.pipe.read(&msg, self.stopper(), true);
        defer if (has_next) msg.deinit();
        if (has_next and msg.data.len != 0) try out.appendSlice(a, msg.data);
        return has_next; // false if EOF or pipe broken
    }

    /// The thread's request and client, in an arena of their own.
    const Job = struct {
        arena: std.heap.ArenaAllocator,
        cli: httplib_client.ClientImpl,
        req: httplib_client.Request,
        shared: *Shared,

        /// `make_header_msg`: the status and headers, stripped of the
        /// router's own, `Content-Type` apart.
        fn makeHeaderMsg(response: *const httplib_client.Response) !Msg {
            var msg: Msg = .{ .status = response.status };
            var tmp = std.heap.ArenaAllocator.init(gpa);
            defer tmp.deinit();
            for (response.headers.entries.items) |h| {
                const lowered = try toLowerCopy(tmp.allocator(), h.key);
                if (shouldStripProxyHeader(lowered)) continue;
                if (std.mem.eql(u8, lowered, "content-type")) {
                    gpa.free(msg.content_type);
                    msg.content_type = try gpa.dupe(u8, h.value);
                    continue;
                }
                if (msg.headers.get(h.key)) |v| {
                    gpa.free(v.*);
                    v.* = try gpa.dupe(u8, h.value);
                } else {
                    try msg.headers.put(gpa, try gpa.dupe(u8, h.key), try gpa.dupe(u8, h.value));
                }
            }
            return msg;
        }

        fn responseHandler(ctx: ?*anyopaque, response: *const httplib_client.Response) bool {
            const job: *Job = @ptrCast(@alignCast(ctx.?));
            job.shared.headers_sent.store(true, .seq_cst);
            const msg = makeHeaderMsg(response) catch return false;
            return job.shared.pipe.write(msg) catch false; // send headers first
        }

        fn contentReceiver(ctx: ?*anyopaque, data: []const u8) bool {
            const job: *Job = @ptrCast(@alignCast(ctx.?));
            // send data chunks; false if pipe is closed / broken (signal to stop receiving)
            const d = gpa.dupe(u8, data) catch return false;
            const ok = job.shared.pipe.write(.{ .data = d }) catch false;
            if (!ok) gpa.free(d);
            return ok;
        }

        /// The thread's lambda: the request sent, an error turned into a
        /// 500 and a body, headers sent for a bodyless status.
        fn run(job: *Job) void {
            defer {
                job.shared.release();
                job.cli.deinit();
                job.arena.deinit();
                gpa.destroy(job);
            }
            const result = job.cli.send(&job.req) catch httplib_client.Result{ .res = null, .err = .unknown };
            if (result.err != .success) {
                const err_str = result.err.toString();
                const ez = common.dupZ(err_str);
                defer common.freeZ(ez);
                srv(.err, "operator()", "http client error: %s\n", .{ez});
                _ = job.shared.pipe.write(.{ .status = 500 }) catch false; // header
                const body: []u8 = std.mem.concat(gpa, u8, &.{ "proxy error: ", err_str }) catch @constCast(&[_]u8{});
                _ = job.shared.pipe.write(.{ .data = body }) catch false; // body
            } else if (!job.shared.headers_sent.load(.seq_cst)) {
                // httplib skips response_handler for bodyless statuses like 204, send headers here instead
                if (makeHeaderMsg(result.res.?)) |msg| {
                    _ = job.shared.pipe.write(msg) catch false;
                } else |_| {}
            }
            job.shared.pipe.closeWrite(); // signal EOF to reader
            srv(.dbg, "operator()", "%s", .{@as([*:0]const u8, "client request thread ended\n")});
        }
    };

    /// Ports the constructor `server_http_proxy` (server-models.cpp:2427 @b29c606e2):
    /// starts the client thread and waits for the target's status and
    /// headers.
    ///
    /// Return: the response; `destroy` frees it.
    pub fn create(
        method: []const u8,
        scheme: []const u8,
        host: []const u8,
        port: c_int,
        path: []const u8,
        headers: *const http.StringMap,
        body: []const u8,
        files: *const http.SortedMap(http.UploadedFile),
        should_stop: http.ShouldStop,
        timeout_read: i32,
        timeout_write: i32,
    ) !*http.Res {
        // shared between reader and writer threads
        const job = try gpa.create(Job);
        job.* = .{ .arena = std.heap.ArenaAllocator.init(gpa), .cli = undefined, .req = undefined, .shared = undefined };
        const ja = job.arena.allocator();
        const shared = try gpa.create(Shared);
        shared.* = .{ .pipe = common.Pipe(Msg).init(gpa) };
        job.shared = shared;
        job.cli = httplib_client.ClientImpl.init(ja, try ja.dupe(u8, host), port);
        var started = false;
        errdefer if (!started) {
            shared.release();
            shared.release();
            job.cli.deinit();
            job.arena.deinit();
            gpa.destroy(job);
        };

        if (std.mem.eql(u8, scheme, "https")) {
            return throw(error.RuntimeError, "HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined", .{});
        }

        // setup Client
        job.cli.setFollowLocation(true);
        job.cli.setConnectionTimeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
        job.cli.setWriteTimeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
        job.cli.setReadTimeout(timeout_write, 0);

        const self = try gpa.create(Proxy);
        errdefer if (!started) gpa.destroy(self);
        self.* = .{ .base = .{ .arena = std.heap.ArenaAllocator.init(gpa), .vtable = &vtable }, .shared = shared, .should_stop = should_stop };
        errdefer if (!started) self.base.arena.deinit();
        self.base.status = 500; // to be overwritten upon response

        // wire up the receive end of the pipe
        self.base.next = .{ .ctx = self, .func = next };

        // when files are present, the body was converted from multipart form data to JSON
        // we need to reconstruct the multipart body for the downstream server
        var effective_body: []const u8 = try ja.dupe(u8, body);
        var override_content_type: []const u8 = "";
        const has_files = files.entries.items.len != 0;
        if (has_files) {
            var form_fields = try json.parseNoThrow(ja, body);
            if (!form_fields.isDiscarded()) {
                const boundary = try generateMultipartBoundary(ja);
                effective_body = try buildMultipartBody(ja, &form_fields, files, boundary);
                override_content_type = try std.mem.concat(ja, u8, &.{ "multipart/form-data; boundary=", boundary });
            } else {
                return throw(error.RuntimeError, "failed to parse multipart form fields JSON", .{});
            }
        }

        // prepare the request to destination server
        var req: httplib_client.Request = .{ .a = ja, .method = try ja.dupe(u8, method), .path = try ja.dupe(u8, path) };
        for (headers.entries.items) |h| {
            const lowered = try toLowerCopy(ja, h.key);
            // disable Accept-Encoding to avoid compressed responses
            if (std.mem.eql(u8, lowered, "accept-encoding")) continue;
            // the body is already decoded
            if (std.mem.eql(u8, lowered, "transfer-encoding")) continue;
            // let httplib calculate Content-Length from the actual body
            if (std.mem.eql(u8, lowered, "content-length")) continue;
            if (std.mem.eql(u8, lowered, "content-type")) {
                // we set our own Content-Type with the new boundary
                if (has_files) continue;
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (std.mem.find(u8, h.value, "multipart/form-data") != null) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (std.mem.eql(u8, lowered, "host")) {
                const is_default_port = (std.mem.eql(u8, scheme, "https") and port == 443) or (std.mem.eql(u8, scheme, "http") and port == 80);
                const url_host = try formatHost(ja, host);
                try req.setHeader(h.key, if (is_default_port) url_host else try std.fmt.allocPrint(ja, "{s}:{d}", .{ url_host, port }));
            } else {
                try req.setHeader(h.key, h.value);
            }
        }
        req.body = effective_body;
        if (override_content_type.len != 0) try req.setHeader("Content-Type", override_content_type);
        req.response_handler = .{ .ctx = job, .func = Job.responseHandler };
        req.content_receiver = .{ .ctx = job, .func = Job.contentReceiver };
        job.req = req;

        // start the proxy thread
        {
            const mz = common.dupZ(req.method);
            defer common.freeZ(mz);
            const pz = common.dupZ(req.path);
            defer common.freeZ(pz);
            srv(.dbg, "server_http_proxy", "start proxy thread %s %s\n", .{ mz, pz });
        }
        const thread = try std.Thread.spawn(.{}, Job.run, .{job});
        started = true;
        thread.detach();

        // wait for the first chunk (headers)
        {
            var header: Msg = .{};
            if (shared.pipe.read(&header, self.stopper(), true)) {
                defer header.deinit();
                srv(.dbg, "server_http_proxy", "%s", .{@as([*:0]const u8, "received response headers\n")});
                self.base.status = header.status;
                for (header.headers.entries.items) |e| try self.base.setHeader(e.key, e.value);
                if (header.content_type.len != 0) self.base.content_type = try self.base.allocator().dupe(u8, header.content_type);
            } else {
                srv(.dbg, "server_http_proxy", "%s", .{@as([*:0]const u8, "no response headers received (request cancelled?)\n")});
            }
        }
        return &self.base;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the router's own headers are stripped" {
    try std.testing.expect(shouldStripProxyHeader("content-length"));
    try std.testing.expect(shouldStripProxyHeader("access-control-allow-origin"));
    try std.testing.expect(!shouldStripProxyHeader("content-type"));
}
