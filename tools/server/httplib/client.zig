//! cpp-httplib's plain-HTTP client, as `server_http_proxy` drives it: one
//! request sent with a response handler and a content receiver, redirects
//! followed, the connection closed after each request.
//!
//! # Provenance
//!
//! Ports parts of `llama.cpp/vendor/cpp-httplib/httplib.cpp` and
//! `llama.cpp/vendor/cpp-httplib/httplib.h` (cpp-httplib 0.59.0, vendored at
//! llama.cpp v0.6.0, `d81235049`): `ClientImpl` -- `send`, `send_`,
//! `handle_request`, `process_request`, `write_request`, the redirect path
//! -- and the `detail` functions under it: `create_client_socket`,
//! `wait_until_socket_is_ready`, `parse_url`, `parse_port`,
//! `resolve_relative_location`,
//! `encode_request_target` with `encode_path` and
//! `normalize_query_string`, `parse_status_line`, `redirect`, the host
//! strings, and `to_string(Error)`. Each declaration names what it ports
//! and the line it began at.
//!
//! # What is not here
//!
//! What the proxy never sets is left at its default and its arm not
//! ported: HTTP proxies and `no_proxy`, basic/bearer authentication,
//! digest authentication (SSL-only), the address map, interface binding,
//! compression, content providers, upload and download progress, the
//! logger, and keep-alive (`keep_alive_` stays false, so every request
//! closes its connection). SSL is not built by the reference: an `https`
//! redirect fails with `SSLConnection`, as there. Only one thread ever
//! drives a client, so the request mutex and the in-flight bookkeeping
//! are not ported.

const std = @import("std");
const builtin = @import("builtin");
const detail = @import("detail.zig");
const socket = @import("socket.zig");
const message = @import("message.zig");
const regex = @import("llama").regex;

const Headers = detail.Headers;

/// What a client call can fail with besides an `Error`: only allocation.
const Oom = error{OutOfMemory};
const Socket = socket.Socket;
const k = socket.k;

extern "c" fn connect(sock: c_int, addr: ?*const anyopaque, len: u32) c_int;
extern "c" fn getsockopt(sock: c_int, level: c_int, optname: c_int, optval: ?*anyopaque, optlen: *u32) c_int;
extern "c" fn fcntl(fd: c_int, cmd: c_int, ...) c_int;

const is_darwin = builtin.os.tag.isDarwin();
const F_GETFL: c_int = 3;
const F_SETFL: c_int = 4;
const O_NONBLOCK: c_int = if (is_darwin) 0x0004 else 0o4000;
const EINPROGRESS: c_int = if (is_darwin) 36 else 115;
const SO_ERROR: c_int = if (is_darwin) 0x1007 else 4;

/// Ports `CPPHTTPLIB_VERSION` (httplib.h:11 @d81235049).
pub const cpphttplib_version = "0.59.0";
/// Ports `CPPHTTPLIB_CONNECTION_TIMEOUT_SECOND` (httplib.h:37 @d81235049).
const connection_timeout_second: i64 = 300;
/// Ports `CPPHTTPLIB_CLIENT_READ_TIMEOUT_SECOND` (httplib.h:61 @d81235049).
const client_read_timeout_second: i64 = 300;
/// Ports `CPPHTTPLIB_CLIENT_WRITE_TIMEOUT_SECOND` (httplib.h:69 @d81235049).
const client_write_timeout_second: i64 = 5;
/// Ports `CPPHTTPLIB_EXPECT_100_THRESHOLD` (httplib.h:81 @d81235049).
const expect_100_threshold: usize = 1024;
/// Ports `CPPHTTPLIB_EXPECT_100_TIMEOUT_MSECOND` (httplib.h:85 @d81235049).
const expect_100_timeout_msecond: i64 = 1000;
/// Ports `CPPHTTPLIB_REDIRECT_MAX_COUNT` (httplib.h:121 @d81235049).
const redirect_max_count: usize = 20;

/// Ports `Error` (httplib.h:1907 @d81235049), the members a client reaches.
pub const Error = enum {
    success,
    unknown,
    connection,
    bind_ip_address,
    read,
    write,
    exceed_redirect_count,
    canceled,
    ssl_connection,
    compression,
    connection_timeout,
    invalid_headers,
    unsupported_content_encoding,

    /// Ports `to_string` (httplib.cpp:6461 @d81235049).
    pub fn toString(self: Error) []const u8 {
        return switch (self) {
            .success => "Success (no error)",
            .unknown => "Unknown",
            .connection => "Could not establish connection",
            .bind_ip_address => "Failed to bind IP address",
            .read => "Failed to read connection",
            .write => "Failed to write connection",
            .exceed_redirect_count => "Maximum redirect count exceeded",
            .canceled => "Connection handling canceled",
            .ssl_connection => "SSL connection failed",
            .compression => "Compression failed",
            .connection_timeout => "Connection timed out",
            .invalid_headers => "Invalid headers",
            .unsupported_content_encoding => "Unsupported Content-Encoding",
        };
    }
};

/// Ports `Response` (httplib.h:1832 @d81235049), a client's: what the server
/// answered. Every string lives in `a`.
pub const Response = struct {
    a: std.mem.Allocator,
    version: []const u8 = "",
    status: c_int = -1,
    reason: []const u8 = "",
    headers: Headers = .{},
    trailers: Headers = .{},
    body: std.ArrayList(u8) = .empty,
    location: []const u8 = "", // Redirect location

    /// `get_header_value(key)`.
    pub fn getHeaderValue(self: *const Response, key: []const u8) []const u8 {
        return self.headers.get(key, 0) orelse "";
    }

    /// `has_header(key)`.
    pub fn hasHeader(self: *const Response, key: []const u8) bool {
        return self.headers.find(key) != null;
    }
};

/// `ResponseHandler`: false cancels the request.
pub const ResponseHandler = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, res: *const Response) bool,
};

/// `ContentReceiverWithProgress`, without the offsets the proxy ignores:
/// false cancels the request.
pub const ContentReceiver = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, data: []const u8) bool,
};

/// Ports `Request` (httplib.h:1757 @d81235049), a client's. Strings live in
/// `a`; a copy (`clone`) shares the handlers.
pub const Request = struct {
    a: std.mem.Allocator,
    method: []const u8 = "",
    path: []const u8 = "",
    headers: Headers = .{},
    params: detail.Params = .{},
    body: []const u8 = "",
    response_handler: ?ResponseHandler = null,
    content_receiver: ?ContentReceiver = null,
    redirect_count: usize = redirect_max_count,

    /// `Request(const Request &)`.
    pub fn clone(self: *const Request) Oom!Request {
        var r = self.*;
        r.headers = try self.headers.clone(self.a);
        r.params = try self.params.clone(self.a);
        return r;
    }

    /// `has_header(key)`.
    pub fn hasHeader(self: *const Request, key: []const u8) bool {
        return detail.hasHeader(&self.headers, key);
    }

    /// `set_header(key, val)`: appended when valid.
    pub fn setHeader(self: *Request, key: []const u8, val: []const u8) Oom!void {
        try detail.setHeader(self.a, &self.headers, try self.a.dupe(u8, key), try self.a.dupe(u8, val));
    }
};

/// What `send` returns: the response, or the error that ended the request.
pub const Result = struct {
    res: ?*Response,
    err: Error,
};

/// Ports `parse_port` (httplib.h:845 @d81235049): over
/// `parse_int_in_range`, so the whole string must be the number.
fn parsePort(s: []const u8, port: *c_int) bool {
    return detail.parseIntInRange(s, 1, 65535, port);
}

/// Ports `UrlComponents` (httplib.h:853 @d81235049).
const UrlComponents = struct {
    scheme: []const u8 = "",
    host: []const u8 = "",
    port: []const u8 = "",
    path: []const u8 = "",
    query: []const u8 = "",
};

/// `url.find_first_of(set, pos)`.
fn findFirstOf(s: []const u8, set: []const u8, pos: usize) ?usize {
    var i = pos;
    while (i < s.len) : (i += 1) {
        if (std.mem.findScalar(u8, set, s[i]) != null) return i;
    }
    return null;
}

/// Ports `parse_url` (httplib.h:861 @d81235049): the components a redirect's
/// `Location` names. Slices of `url`.
fn parseUrl(url: []const u8, uc: *UrlComponents) bool {
    uc.* = .{};
    var pos: usize = 0;

    if (std.mem.find(u8, url, "://")) |sep| {
        uc.scheme = url[0..sep];
        // Scheme must be [a-z]+ only
        if (uc.scheme.len == 0) return false;
        for (uc.scheme) |c| {
            if (c < 'a' or c > 'z') return false;
        }
        pos = sep + 3;
    } else if (std.mem.startsWith(u8, url, "//")) {
        pos = 2;
    }

    const has_authority_prefix = pos > 0;
    const has_authority = has_authority_prefix or (url.len != 0 and url[0] != '/' and url[0] != '?' and url[0] != '#');
    if (has_authority) {
        if (pos < url.len and url[pos] == '[') {
            const close = std.mem.findScalarPos(u8, url, pos, ']') orelse return false;
            uc.host = url[pos + 1 .. close];
            // IPv6 host must be [a-fA-F0-9:]+ only
            if (uc.host.len == 0) return false;
            for (uc.host) |c| {
                if (!(detail.isAsciiDigit(c) or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F') or c == ':')) return false;
            }
            pos = close + 1;
            if (pos < url.len) {
                const c = url[pos];
                if (c != ':' and c != '/' and c != '?' and c != '#') return false;
            }
        } else {
            const end = findFirstOf(url, ":/?#", pos) orelse url.len;
            uc.host = url[pos..end];
            pos = end;
        }

        if (pos < url.len and url[pos] == ':') {
            pos += 1;
            const end = findFirstOf(url, "/?#", pos) orelse url.len;
            uc.port = url[pos..end];
            pos = end;
        }

        // Without :// or //, the entire input must be consumed as host[:port].
        if (!has_authority_prefix and pos < url.len) {
            uc.host = "";
            uc.port = "";
            pos = 0;
        }
    }

    if (pos < url.len and url[pos] != '?' and url[pos] != '#') {
        const end = findFirstOf(url, "?#", pos) orelse url.len;
        uc.path = url[pos..end];
        pos = end;
    }

    if (pos < url.len and url[pos] == '?') {
        const end = std.mem.findScalarPos(u8, url, pos, '#') orelse url.len;
        uc.query = url[pos..end];
    }

    return true;
}

/// `s.find_first_of(set)` from `pos`, or `s.len` (`npos` clamped, as every
/// caller below takes a `min` with the size).
fn firstOfOrLen(s: []const u8, set: []const u8, pos: usize) usize {
    return findFirstOf(s, set, pos) orelse s.len;
}

/// Ports `resolve_relative_location` (httplib.h:952 @d81235049): a
/// relative-path or query-only `Location` resolved against the path of the
/// request being redirected (RFC 3986 section 5.2). An absolute URI and a
/// reference starting with '/' come back unchanged.
fn resolveRelativeLocation(a: std.mem.Allocator, location: []const u8, base: []const u8) Oom![]const u8 {
    if (location.len == 0 or location[0] == '/') return location;

    // A ':' in the first segment means the value has a scheme.
    // (`find(':') < find_first_of("/?#")`, with npos the largest value)
    if (std.mem.findScalar(u8, location, ':')) |colon| {
        if (findFirstOf(location, "/?#", 0)) |sep| {
            if (colon < sep) return location;
        } else return location;
    }

    if (location[0] == '#') {
        const end = std.mem.findScalar(u8, base, '#') orelse base.len;
        return std.mem.concat(a, u8, &.{ base[0..end], location });
    }

    var base_path = base[0..firstOfOrLen(base, "?#", 0)];
    if (location[0] == '?') return std.mem.concat(a, u8, &.{ base_path, location });

    if (base_path.len == 0 or base_path[0] != '/') base_path = "/";
    const dir_end = (std.mem.findScalarLast(u8, base_path, '/') orelse 0) + 1;
    const merged = try std.mem.concat(a, u8, &.{ base_path[0..dir_end], location });

    // Remove "." and ".." segments from the merged path.
    const path_end = firstOfOrLen(merged, "?#", 0);
    var path: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < path_end) {
        const next = @min(std.mem.findScalarPos(u8, merged, i + 1, '/') orelse merged.len, path_end);
        const segment = merged[i + 1 .. next];
        const is_last = next == path_end;
        if (std.mem.eql(u8, segment, ".") or std.mem.eql(u8, segment, "..")) {
            if (std.mem.eql(u8, segment, "..")) {
                // path.erase(min(path.rfind('/'), path.size()))
                if (std.mem.findScalarLast(u8, path.items, '/')) |slash| path.shrinkRetainingCapacity(slash);
            }
            if (is_last) try path.append(a, '/');
        } else {
            try path.append(a, '/');
            try path.appendSlice(a, segment);
        }
        i = next;
    }
    if (path.items.len == 0) try path.append(a, '/');

    try path.appendSlice(a, merged[path_end..]);
    return path.items;
}

/// Ports `encode_path` (httplib.cpp:1120 @d81235049): over the C string, so the
/// first NUL ends it.
fn encodePath(a: std.mem.Allocator, s: []const u8) Oom![]u8 {
    var result: std.ArrayList(u8) = .empty;
    for (s) |c| {
        if (c == 0) break;
        switch (c) {
            ' ' => try result.appendSlice(a, "%20"),
            '+' => try result.appendSlice(a, "%2B"),
            '\'' => try result.appendSlice(a, "%27"),
            ',' => try result.appendSlice(a, "%2C"),
            ';' => try result.appendSlice(a, "%3B"),
            else => {
                // Control characters (incl. CR/LF) and non-ASCII bytes are not allowed
                // in a request-target as-is.
                if (c < 0x20 or c == 0x7f or c >= 0x80) {
                    try result.print(a, "%{X:0>2}", .{c});
                } else {
                    try result.append(a, c);
                }
            },
        }
    }
    return result.items;
}

/// Ports `normalize_query_string` (httplib.cpp:4325 @d81235049): each pair decoded
/// and re-encoded, an empty key dropped.
fn normalizeQueryString(a: std.mem.Allocator, query: []const u8) Oom![]u8 {
    const Ctx = struct {
        a: std.mem.Allocator,
        result: std.ArrayList(u8) = .empty,
        pub fn call(self: *@This(), seg: []const u8) Oom!void {
            const pair = detail.divideQueryPair(seg);
            if (pair.key.len == 0) return;
            const dec_key = try detail.decodeQueryComponent(self.a, pair.key);
            const dec_val = try detail.decodeQueryComponent(self.a, pair.val);
            if (self.result.items.len != 0) try self.result.append(self.a, '&');
            try self.result.appendSlice(self.a, try detail.encodeQueryComponent(self.a, dec_key));
            if (pair.val.len != 0 or std.mem.findScalar(u8, seg, '=') != null) {
                try self.result.append(self.a, '=');
                try self.result.appendSlice(self.a, try detail.encodeQueryComponent(self.a, dec_val));
            }
        }
    };
    var ctx: Ctx = .{ .a = a };
    try detail.split(query, '&', std.math.maxInt(usize), &ctx);
    return ctx.result.items;
}

/// Ports `encode_request_target` (httplib.cpp:4351 @d81235049): the path encoded,
/// the query normalized.
fn encodeRequestTarget(a: std.mem.Allocator, target: []const u8, path_encode: bool) Oom![]u8 {
    const query_pos = std.mem.findScalar(u8, target, '?');
    const path_part = target[0 .. query_pos orelse target.len];
    const query_part = if (query_pos) |q| target[q + 1 ..] else "";

    var result: std.ArrayList(u8) = .empty;
    try result.appendSlice(a, if (path_encode) try encodePath(a, path_part) else path_part);

    if (query_part.len != 0) {
        if (path_encode) {
            const normalized = try normalizeQueryString(a, query_part);
            if (normalized.len != 0) {
                try result.append(a, '?');
                try result.appendSlice(a, normalized);
            }
        } else {
            try result.append(a, '?');
            try result.appendSlice(a, query_part);
        }
    }
    return result.items;
}

/// Ports `prepare_host_string` (httplib.cpp:7706 @d81235049): an IPv6 address
/// bracketed, unless it already is.
fn prepareHostString(a: std.mem.Allocator, host: []const u8) Oom![]const u8 {
    if (std.mem.findScalar(u8, host, ':') == null or (host.len != 0 and host[0] == '[')) return host;
    return std.mem.concat(a, u8, &.{ "[", host, "]" });
}

/// Ports `make_host_and_port_string` (httplib.cpp:7718 @d81235049): the port
/// appended unless it is the scheme's default.
fn makeHostAndPortString(a: std.mem.Allocator, host: []const u8, port: c_int, is_ssl: bool) Oom![]const u8 {
    const result = try prepareHostString(a, host);
    if ((!is_ssl and port == 80) or (is_ssl and port == 443)) return result;
    return std.fmt.allocPrint(a, "{s}:{d}", .{ result, port });
}

/// Ports `parse_status_line` (httplib.cpp:3528 @d81235049): `regex_match` against
/// `(HTTP/1\.[01]) (\d{3})(?: (.*?))?\r\n`.
fn parseStatusLine(a: std.mem.Allocator, line: []const u8, version: *[]const u8, status: *c_int, reason: *[]const u8) Oom!bool {
    var re = regex.Regex.compile("(HTTP/1\\.[01]) (\\d{3})(?: (.*?))?\r\n") catch |err| switch (err) {
        error.Range, error.Collate, error.Ctype, error.Escape, error.Backref, error.Brack, error.Paren, error.Brace, error.BadBrace, error.BadRepeat, error.Complexity, error.Parse => unreachable,
    };
    defer re.deinit();
    var m = regex.Match.init();
    defer m.deinit();
    if (!re.match(line, &m)) return false;
    version.* = try a.dupe(u8, line[m.position(1)..][0..m.length(1)]);
    status.* = std.fmt.parseInt(c_int, line[m.position(2)..][0..m.length(2)], 10) catch |err| switch (err) {
        error.Overflow, error.InvalidCharacter => unreachable,
    }; // three digits
    reason.* = if (m.matched(3)) try a.dupe(u8, line[m.position(3)..][0..m.length(3)]) else "";
    return true;
}

/// Ports `has_conflicting_content_length` (httplib.cpp:3754 @d81235049): both a
/// `Transfer-Encoding` and a non-zero `Content-Length`.
fn hasConflictingContentLength(headers: *const Headers) bool {
    var invalid = false;
    return detail.hasHeader(headers, "Transfer-Encoding") and detail.getHeaderValueU64(headers, "Content-Length", 0, &invalid) > 0;
}

/// Ports `set_nonblocking` (httplib.cpp:2507 @d81235049).
fn setNonblocking(sock: Socket, nonblocking: bool) void {
    const flags = fcntl(sock, F_GETFL, @as(c_int, 0));
    _ = fcntl(sock, F_SETFL, if (nonblocking) flags | O_NONBLOCK else flags & ~O_NONBLOCK);
}

/// Ports `wait_until_socket_is_ready` (httplib.cpp:1789 @d81235049): a non-blocking
/// connect's outcome, within the connection timeout.
fn waitUntilSocketIsReady(sock: Socket, sec: i64, usec: i64) Error {
    var pfd = [_]std.posix.pollfd{.{ .fd = sock, .events = k.POLLIN | k.POLLOUT, .revents = 0 }};
    const timeout: c_int = @intCast(sec * 1000 + @divTrunc(usec, 1000));
    var poll_res: c_int = undefined;
    while (true) {
        poll_res = std.c.poll(&pfd, 1, timeout);
        if (poll_res < 0 and socket.errno() == k.EINTR) continue;
        break;
    }
    if (poll_res == 0) return .connection_timeout;
    if (poll_res > 0 and (pfd[0].revents & (k.POLLIN | k.POLLOUT)) != 0) {
        var err: c_int = 0;
        var len: u32 = @sizeOf(c_int);
        const res = getsockopt(sock, k.SOL_SOCKET, SO_ERROR, &err, &len);
        return if (res >= 0 and err == 0) .success else .connection;
    }
    return .connection;
}

/// Ports `ClientImpl` (httplib.h:2690 @d81235049): one host and port, the
/// settings the proxy sets, and the socket of the request in flight.
pub const ClientImpl = struct {
    a: std.mem.Allocator,
    host: []const u8,
    port: c_int,

    connection_timeout_sec: i64 = connection_timeout_second,
    connection_timeout_usec: i64 = 0,
    read_timeout_sec: i64 = client_read_timeout_second,
    read_timeout_usec: i64 = 0,
    write_timeout_sec: i64 = client_write_timeout_second,
    write_timeout_usec: i64 = 0,

    keep_alive: bool = false,
    follow_location: bool = false,
    path_encode: bool = true,
    address_family: c_int = k.AF_UNSPEC,
    tcp_nodelay: bool = detail.tcp_nodelay,
    ipv6_v6only: bool = false,
    decompress: bool = true,
    payload_max_length: usize = detail.payload_max_length,

    basic_auth_username: []const u8 = "",
    basic_auth_password: []const u8 = "",
    default_headers: Headers = .{},

    sock: Socket = socket.invalid_socket,

    /// Ports `ClientImpl`'s constructor `ClientImpl(host, port)` (httplib.cpp:10158 @d81235049).
    /// A host naming Linux's abstract socket namespace (a leading NUL,
    /// `escape_abstract_namespace_unix_domain`) is not supported: the proxy
    /// only connects over TCP.
    pub fn init(a: std.mem.Allocator, host: []const u8, port: c_int) ClientImpl {
        return .{ .a = a, .host = host, .port = port };
    }

    /// Ports `ClientImpl`'s destructor (httplib.cpp:10164 @d81235049): the socket shut down and
    /// closed.
    pub fn deinit(self: *ClientImpl) void {
        if (self.sock != socket.invalid_socket) _ = socket.shutdownSocket(self.sock);
        self.closeSocket();
    }

    /// Ports `set_connection_timeout` (httplib.cpp:12634 @d81235049).
    pub fn setConnectionTimeout(self: *ClientImpl, sec: i64, usec: i64) void {
        self.connection_timeout_sec = sec;
        self.connection_timeout_usec = usec;
    }

    /// Ports `set_read_timeout` (httplib.cpp:12639 @d81235049).
    pub fn setReadTimeout(self: *ClientImpl, sec: i64, usec: i64) void {
        self.read_timeout_sec = sec;
        self.read_timeout_usec = usec;
    }

    /// Ports `set_write_timeout` (httplib.cpp:12644 @d81235049).
    pub fn setWriteTimeout(self: *ClientImpl, sec: i64, usec: i64) void {
        self.write_timeout_sec = sec;
        self.write_timeout_usec = usec;
    }

    /// Ports `set_follow_location` (httplib.cpp:12665 @d81235049).
    pub fn setFollowLocation(self: *ClientImpl, on: bool) void {
        self.follow_location = on;
    }

    /// Ports `set_basic_auth` (httplib.cpp:12653 @d81235049).
    pub fn setBasicAuth(self: *ClientImpl, username: []const u8, password: []const u8) void {
        self.basic_auth_username = username;
        self.basic_auth_password = password;
    }

    /// Ports `set_default_headers` (httplib.cpp:12674 @d81235049).
    pub fn setDefaultHeaders(self: *ClientImpl, headers: Headers) void {
        self.default_headers = headers;
    }

    fn closeSocket(self: *ClientImpl) void {
        if (self.sock == socket.invalid_socket) return;
        _ = socket.closeSocket(self.sock);
        self.sock = socket.invalid_socket;
    }

    /// Ports `disconnect` (httplib.cpp:10318 @d81235049).
    fn disconnect(self: *ClientImpl) void {
        if (self.sock != socket.invalid_socket) _ = socket.shutdownSocket(self.sock);
        self.closeSocket();
    }

    /// Ports `ClientImpl`'s `create_client_socket` (httplib.cpp:10245 @d81235049) over
    /// `detail`'s `create_client_socket` (httplib.cpp:2625 @d81235049): resolved,
    /// connected without blocking past the connection timeout, then given
    /// the read and write timeouts.
    fn createClientSocket(self: *ClientImpl, err: *Error) Oom!Socket {
        const Ctx = struct {
            cli: *ClientImpl,
            err: *Error,
            fn bindOrConnect(ctx: ?*anyopaque, sock2: Socket, addr: ?*const anyopaque, addrlen: u32, quit: *bool) bool {
                const c: *@This() = @ptrCast(@alignCast(ctx.?));
                setNonblocking(sock2, true);
                const ret = connect(sock2, addr, addrlen);
                if (ret < 0) {
                    // is_connection_error
                    if (socket.errno() != EINPROGRESS) {
                        c.err.* = .connection;
                        return false;
                    }
                    c.err.* = waitUntilSocketIsReady(sock2, c.cli.connection_timeout_sec, c.cli.connection_timeout_usec);
                    if (c.err.* != .success) {
                        if (c.err.* == .connection_timeout) quit.* = true;
                        return false;
                    }
                }
                setNonblocking(sock2, false);
                _ = socket.setSocketOptTime(sock2, k.SOL_SOCKET, k.SO_RCVTIMEO, c.cli.read_timeout_sec, c.cli.read_timeout_usec);
                _ = socket.setSocketOptTime(sock2, k.SOL_SOCKET, k.SO_SNDTIMEO, c.cli.write_timeout_sec, c.cli.write_timeout_usec);
                c.err.* = .success;
                return true;
            }
        };
        var ctx: Ctx = .{ .cli = self, .err = err };
        const hz = try self.a.dupeSentinel(u8, self.host, 0);
        const sock = socket.createSocket(hz, self.port, self.address_family, 0, self.tcp_nodelay, self.ipv6_v6only, null, .{ .ctx = &ctx, .func = Ctx.bindOrConnect });
        if (sock != socket.invalid_socket) {
            err.* = .success;
        } else if (err.* == .success) {
            err.* = .connection;
        }
        return sock;
    }

    /// Ports `read_response_line` (httplib.cpp:10324 @d81235049): the status line,
    /// skipping `100 Continue` when asked.
    fn readResponseLine(self: *ClientImpl, strm: *socket.SocketStream, res: *Response, skip_100_continue: bool) Oom!bool {
        _ = self;
        var buf: [2048]u8 = undefined;
        var lr = socket.LineReader.init(res.a, strm, &buf);
        defer lr.deinit();

        if (!try lr.getline()) return false;
        if (!try parseStatusLine(res.a, lr.cstr(), &res.version, &res.status, &res.reason)) {
            return false; // req.method == "CONNECT" never
        }

        // Ignore '100 Continue' (only when not using Expect: 100-continue explicitly)
        while (skip_100_continue and res.status == 100) {
            if (!try lr.getline()) return false; // CRLF
            if (!try lr.getline()) return false; // next response line
            if (!try parseStatusLine(res.a, lr.cstr(), &res.version, &res.status, &res.reason)) return false;
        }
        return true;
    }

    /// Ports `ClientImpl`'s `send` (httplib.cpp:10352 @d81235049), the overload taking a
    /// response and an error: one request, redirects followed.
    pub fn sendInto(self: *ClientImpl, req: *Request, res: *Response, err: *Error) Oom!bool {
        return self.sendImpl(req, res, err);
    }

    /// Ports `ClientImpl`'s `send_` (httplib.cpp:10364 @d81235049): connects unless
    /// the open socket is still alive, runs the request, and closes the
    /// connection after it (no keep-alive here).
    fn sendImpl(self: *ClientImpl, req: *Request, res: *Response, err: *Error) Oom!bool {
        var is_alive = false;
        if (self.sock != socket.invalid_socket) {
            is_alive = socket.isSocketAlive(self.sock);
            // Peer seems gone — non-graceful shutdown to avoid SIGPIPE.
            if (!is_alive) self.disconnect();
        }
        if (!is_alive) {
            const sock = try self.createClientSocket(err);
            if (sock == socket.invalid_socket) return false;
            self.sock = sock;
        }

        for (self.default_headers.entries.items) |h| {
            if (req.headers.find(h.key) == null) _ = try req.headers.emplace(req.a, h.key, h.value);
        }

        var ret = false;
        const close_connection = !self.keep_alive;
        defer if (close_connection or !ret) self.disconnect();

        // `process_socket`, then `process_client_socket` (httplib.cpp:1960 @d81235049)
        var strm: socket.SocketStream = .{
            .sock = self.sock,
            .read_timeout_sec = self.read_timeout_sec,
            .read_timeout_usec = self.read_timeout_usec,
            .write_timeout_sec = self.write_timeout_sec,
            .write_timeout_usec = self.write_timeout_usec,
        };
        ret = try self.handleRequest(&strm, req, res, close_connection, err);

        if (!ret and err.* == .success) err.* = .unknown;
        return ret;
    }

    /// Ports `ClientImpl`'s `send` (httplib.cpp:10455 @d81235049), the overload taking a
    /// request, over `send_` (httplib.cpp:10460 @d81235049): a copy of the request
    /// sent; the response only when it succeeded.
    pub fn send(self: *ClientImpl, req_in: *const Request) Oom!Result {
        var req = try req_in.clone();
        const res = try req.a.create(Response);
        res.* = .{ .a = req.a };
        var err: Error = .success;
        const ret = try self.sendInto(&req, res, &err);
        return .{ .res = if (ret) res else null, .err = err };
    }

    /// Ports `prepare_default_headers` (httplib.cpp:10472 @d81235049): `Host` first,
    /// `Accept`, and -- with no content receiver -- the encoding and agent;
    /// the body's type and length.
    fn prepareDefaultHeaders(self: *ClientImpl, r: *Request, ct: []const u8) Oom!void {
        // Default headers are meant for the origin and may carry its credentials
        if (!std.mem.eql(u8, r.method, "CONNECT")) {
            for (self.default_headers.entries.items) |h| {
                if (!r.hasHeader(h.key)) _ = try r.headers.emplace(r.a, h.key, h.value);
            }
        }
        // RFC 9110 5.3 recommends sending control data such as Host first
        if (!r.hasHeader("Host")) {
            const v = if (self.address_family == k.AF_UNIX) "localhost" else try makeHostAndPortString(r.a, self.host, self.port, false);
            try r.headers.entries.insert(r.a, 0, .{ .key = "Host", .value = v });
        }
        if (!r.hasHeader("Accept")) _ = try r.headers.emplace(r.a, "Accept", "*/*");
        if (r.content_receiver == null) {
            if (!r.hasHeader("Accept-Encoding")) try r.setHeader("Accept-Encoding", "");
            // add_default_user_agent_header
            if (!r.hasHeader("User-Agent")) try r.setHeader("User-Agent", "cpp-httplib/" ++ cpphttplib_version);
        }
        if (r.body.len != 0) {
            if (ct.len != 0 and !r.hasHeader("Content-Type")) _ = try r.headers.emplace(r.a, "Content-Type", ct);
            if (!r.hasHeader("Content-Length")) _ = try r.headers.emplace(r.a, "Content-Length", try std.fmt.allocPrint(r.a, "{d}", .{r.body.len}));
        }
    }

    /// Ports `handle_request` (httplib.cpp:10893 @d81235049): the request processed,
    /// the connection dropped if the server closes it, and a 3xx followed.
    fn handleRequest(self: *ClientImpl, strm: *socket.SocketStream, req: *Request, res: *Response, close_connection: bool, err: *Error) Oom!bool {
        if (req.path.len == 0) {
            err.* = .connection;
            return false;
        }

        const req_save = try req.clone();

        var ret = try self.processRequest(strm, req, res, close_connection, err);
        if (!ret) return false;

        if (detail.hasHeaderToken(&res.headers, "Connection", "close") or
            (std.mem.eql(u8, res.version, "HTTP/1.0") and !std.mem.eql(u8, res.reason, "Connection established")))
        {
            self.disconnect();
        }

        if (300 < res.status and res.status < 400 and self.follow_location) {
            req.* = req_save;
            ret = try self.redirect(req, res, err);
        }
        return ret;
    }

    /// Ports `ClientImpl`'s `redirect` (httplib.cpp:10979 @d81235049): a same-host
    /// `Location` through this client, another host through a new one; a
    /// relative one resolved against the request's path first.
    fn redirect(self: *ClientImpl, req: *Request, res: *Response, err: *Error) Oom!bool {
        if (req.redirect_count == 0) {
            err.* = .exceed_redirect_count;
            return false;
        }

        const location = res.getHeaderValue("location");
        if (location.len == 0) return false;

        var uc: UrlComponents = undefined;
        if (!parseUrl(try resolveRelativeLocation(req.a, location, req.path), &uc)) return false;

        // Only follow http/https redirects
        if (uc.scheme.len != 0 and !std.mem.eql(u8, uc.scheme, "http") and !std.mem.eql(u8, uc.scheme, "https")) return false;

        const scheme = "http";
        var next_scheme = uc.scheme;
        var next_host = uc.host;
        const port_str = uc.port;
        var next_path = uc.path;
        const next_query = uc.query;

        var next_port = self.port;
        if (port_str.len != 0) {
            if (!parsePort(port_str, &next_port)) return false;
        } else if (next_scheme.len != 0) {
            next_port = if (std.mem.eql(u8, next_scheme, "https")) 443 else 80;
        }

        if (next_scheme.len == 0) next_scheme = scheme;
        if (next_host.len == 0) next_host = self.host;
        if (next_path.len == 0) next_path = "/";

        const path = try std.mem.concat(req.a, u8, &.{ try detail.decodePathComponent(req.a, next_path), next_query });

        // Same host redirect - use current client
        if (std.mem.eql(u8, next_scheme, scheme) and std.mem.eql(u8, next_host, self.host) and next_port == self.port) {
            return redirectWith(self, req, res, path, location, err);
        }

        // Cross-host/scheme redirect - create new client with robust setup
        return self.createRedirectClient(next_scheme, next_host, next_port, req, res, path, location, err);
    }

    /// Ports `create_redirect_client` (httplib.cpp:11032 @d81235049), with
    /// `setup_redirect_client` (httplib.cpp:11104 @d81235049): the credentials and
    /// `Host` left behind; `https` is an SSL error in this build.
    fn createRedirectClient(self: *ClientImpl, scheme: []const u8, host: []const u8, port: c_int, req: *Request, res: *Response, path: []const u8, location: []const u8, err: *Error) Oom!bool {
        // Remove headers that should not be carried over to new host
        for ([_][]const u8{ "Host", "Proxy-Authorization", "Authorization", "Cookie", "Cookie2" }) |name| {
            _ = req.headers.eraseKey(name);
        }

        if (std.mem.eql(u8, scheme, "https")) {
            // SSL not supported - set appropriate error
            err.* = .ssl_connection;
            return false;
        }

        var redirect_client = ClientImpl.init(self.a, host, port);
        defer redirect_client.deinit();
        redirect_client.setConnectionTimeout(self.connection_timeout_sec, 0);
        redirect_client.setReadTimeout(self.read_timeout_sec, self.read_timeout_usec);
        redirect_client.setWriteTimeout(self.write_timeout_sec, self.write_timeout_usec);
        redirect_client.keep_alive = self.keep_alive;
        redirect_client.setFollowLocation(true); // Enable redirects to handle multi-step redirects
        redirect_client.path_encode = self.path_encode;
        redirect_client.decompress = self.decompress;
        redirect_client.address_family = self.address_family;
        redirect_client.tcp_nodelay = self.tcp_nodelay;
        redirect_client.ipv6_v6only = self.ipv6_v6only;
        return redirectWith(&redirect_client, req, res, path, location, err);
    }

    /// Ports `write_request` (httplib.cpp:11180 @d81235049): the head -- a
    /// `Connection: close`, the default headers, a zero length for an empty
    /// write -- and, unless waiting on `100 Continue`, the body.
    /// `rejected_locally` says the request line or a header was refused
    /// before any byte reached the socket.
    fn writeRequest(self: *ClientImpl, strm: *socket.SocketStream, req: *Request, close_connection: bool, err: *Error, skip_body: bool, rejected_locally: *bool) Oom!bool {
        rejected_locally.* = false;

        if (close_connection) {
            if (!req.hasHeader("Connection")) try req.setHeader("Connection", "close");
        }

        const ct_for_defaults: []const u8 = if (!req.hasHeader("Content-Type") and req.body.len != 0) "text/plain" else "";
        try self.prepareDefaultHeaders(req, ct_for_defaults);

        if (req.body.len == 0) {
            if (std.mem.eql(u8, req.method, "POST") or std.mem.eql(u8, req.method, "PUT") or std.mem.eql(u8, req.method, "PATCH")) {
                try req.setHeader("Content-Length", "0");
            }
        }

        // `make_basic_authentication_header` (httplib.cpp:6847 @d81235049)
        if (!std.mem.eql(u8, req.method, "CONNECT") and !req.hasHeader("Authorization")) {
            if (self.basic_auth_password.len != 0 or self.basic_auth_username.len != 0) {
                const cred = try std.mem.concat(req.a, u8, &.{ self.basic_auth_username, ":", self.basic_auth_password });
                const enc = std.base64.standard.Encoder;
                const b64 = try req.a.alloc(u8, enc.calcSize(cred.len));
                _ = enc.encode(b64, cred);
                _ = try req.headers.emplace(req.a, "Authorization", try std.mem.concat(req.a, u8, &.{ "Basic ", b64 }));
            }
        }

        // Request line and headers
        {
            var bstrm: socket.BufferStream = .{ .a = req.a };
            const query_pos = std.mem.findScalar(u8, req.path, '?');
            const query_part = if (query_pos) |q| req.path[q + 1 ..] else "";
            const path_with_query = try encodeRequestTarget(req.a, req.path, self.path_encode);
            if (query_part.len != 0) try detail.parseQueryText(req.a, query_part, &req.params);

            // `write_request_line` (httplib.cpp:3880 @d81235049): the method
            // must be a token, the target a request-target
            if (!detail.isToken(req.method) or !detail.isRequestTarget(path_with_query)) {
                err.* = .write;
                rejected_locally.* = true;
                return false;
            }
            _ = bstrm.write(try std.mem.concat(req.a, u8, &.{ req.method, " ", path_with_query, " HTTP/1.1\r\n" }));

            // `check_and_write_headers` (httplib.cpp:7921 @d81235049)
            for (req.headers.entries.items) |h| {
                if (!detail.isFieldValid(h.key, h.value)) {
                    err.* = .invalid_headers;
                    rejected_locally.* = true;
                    return false;
                }
            }
            if (try message.writeHeaders(req.a, &bstrm, &req.headers) <= 0) {
                err.* = .write;
                return false;
            }

            if (!message.writeData(strm, bstrm.buffer.items)) {
                err.* = .write;
                return false;
            }
        }

        // Body
        if (skip_body) return true;
        return self.writeRequestBody(strm, req, err);
    }

    /// Ports `write_request_body` (httplib.cpp:11349 @d81235049).
    fn writeRequestBody(self: *ClientImpl, strm: *socket.SocketStream, req: *Request, err: *Error) Oom!bool {
        _ = self;
        if (req.body.len == 0) return true; // no content provider here
        if (!message.writeData(strm, req.body)) {
            err.* = .write;
            return false;
        }
        return true;
    }

    /// The `out` content receiver of `process_request`: a redirect's body is
    /// swallowed; otherwise the request's receiver, a false from which
    /// cancels; or, without one, the response's body.
    const Out = struct {
        cli: *ClientImpl,
        req: *Request,
        res: *Response,
        redirect: bool,
        err: *Error,

        pub fn receive(self: *Out, buf: []const u8) Oom!bool {
            if (self.req.content_receiver) |cr| {
                if (self.redirect) return true;
                const ret = cr.func(cr.ctx, buf);
                if (!ret) self.err.* = .canceled;
                return ret;
            }
            if (self.cli.payload_max_length > 0 and (self.res.body.items.len >= self.cli.payload_max_length or buf.len > self.cli.payload_max_length - self.res.body.items.len)) return false;
            try self.res.body.appendSlice(self.res.a, buf);
            return true;
        }
    };

    /// Ports `process_request` (httplib.cpp:11537 @d81235049): the request written
    /// (with `Expect: 100-continue` past the threshold), the status line and
    /// headers read, the response handler told, the body read into the
    /// content receiver.
    fn processRequest(self: *ClientImpl, strm: *socket.SocketStream, req: *Request, res: *Response, close_connection: bool, err: *Error) Oom!bool {
        // Auto-add Expect: 100-continue for large bodies
        if (expect_100_threshold > 0 and !req.hasHeader("Expect")) {
            if (req.body.len >= expect_100_threshold) try req.setHeader("Expect", "100-continue");
        }

        // Check for Expect: 100-continue
        var expect_100_continue = detail.hasHeaderToken(&req.headers, "Expect", "100-continue");

        // Send request (skip body if using Expect: 100-continue)
        var rejected_locally = false;
        const write_request_success = try self.writeRequest(strm, req, close_connection, err, expect_100_continue, &rejected_locally);

        // A failed write normally still reads the response below, since the server
        // may have answered early (e.g. 413/414) and closed while the body was being
        // sent. A request rejected before any byte reached the socket gets no such
        // response, and waiting for one would block until the read timeout.
        if (rejected_locally) return false;

        // Handle Expect: 100-continue.
        var status_line_read = false;
        if (expect_100_continue and write_request_success) {
            if (expect_100_timeout_msecond > 0) {
                strm.read_timeout_sec = @divTrunc(expect_100_timeout_msecond, 1000);
                strm.read_timeout_usec = @mod(expect_100_timeout_msecond, 1000) * 1000;
                status_line_read = try self.readResponseLine(strm, res, false);
                strm.read_timeout_sec = self.read_timeout_sec;
                strm.read_timeout_usec = self.read_timeout_usec;
            }
            if (!status_line_read) {
                // No interim response within the timeout: send the body and handle the
                // response as usual.
                if (!try self.writeRequestBody(strm, req, err)) return false;
                expect_100_continue = false; // Switch to normal response handling
            }
        }

        // Receive response and headers
        // When using Expect: 100-continue, don't auto-skip `100 Continue` response
        if ((!status_line_read and !try self.readResponseLine(strm, res, !expect_100_continue)) or
            !try message.readHeaders(res.a, strm, &res.headers))
        {
            if (write_request_success) err.* = .read;
            return false;
        }

        if (!write_request_success) return false;

        // Handle Expect: 100-continue response
        if (expect_100_continue) {
            if (res.status == 100) {
                // Server accepted, send the body
                if (!try self.writeRequestBody(strm, req, err)) return false;
                // Read the actual response
                res.headers = .{};
                res.body.clearRetainingCapacity();
                if (!try self.readResponseLine(strm, res, true) or !try message.readHeaders(res.a, strm, &res.headers)) {
                    err.* = .read;
                    return false;
                }
            }
            // If not 100 Continue, server returned an error; proceed with that response
        }

        // Body
        if (res.status != 204 and !std.mem.eql(u8, req.method, "HEAD") and !std.mem.eql(u8, req.method, "CONNECT")) {
            // Reject ambiguous framing (RFC 9112 §6.3).
            if (res.status != 304 and hasConflictingContentLength(&res.headers)) {
                err.* = .read;
                return false;
            }

            const is_redirect = 300 < res.status and res.status < 400 and res.status != 304 and self.follow_location;

            if (req.response_handler) |rh| {
                if (!is_redirect) {
                    if (!rh.func(rh.ctx, res)) {
                        err.* = .canceled;
                        return false;
                    }
                }
            }

            var out: Out = .{ .cli = self, .req = req, .res = res, .redirect = is_redirect, .err = err };

            if (res.status != 304) {
                var content_status: c_int = 0;
                const max_length = if (req.content_receiver != null) std.math.maxInt(usize) else self.payload_max_length;
                if (!try message.readContent(strm, res, max_length, &content_status, &out, self.decompress)) {
                    if (err.* != .canceled) {
                        err.* = switch (content_status) {
                            415 => .unsupported_content_encoding,
                            500 => .compression,
                            else => .read,
                        };
                    }
                    return false;
                }
            }
        }
        return true;
    }
};

/// `Error`, under the name the downloader uses.
pub const ErrorCode = Error;

/// Ports `to_string` (httplib.cpp:6461 @d81235049), as a free function.
pub fn errorToString(e: Error) []const u8 {
    return e.toString();
}

/// `Headers`.
pub const HeadersMap = Headers;

/// Ports `Client` (httplib.h:3104 @d81235049) for a plain-HTTP host: the
/// verbs over a `ClientImpl`. A failure to allocate is reported as
/// `Error.unknown`, where the C++ throws `std::bad_alloc`.
pub const Client = struct {
    impl: ClientImpl,

    /// `Client(host, port)`.
    pub fn init(a: std.mem.Allocator, host: []const u8, port: c_int) !Client {
        return .{ .impl = ClientImpl.init(a, try a.dupe(u8, host), port) };
    }

    pub fn deinit(self: *Client) void {
        self.impl.deinit();
    }

    pub fn setBasicAuth(self: *Client, username: []const u8, password: []const u8) void {
        self.impl.setBasicAuth(username, password);
    }

    pub fn setFollowLocation(self: *Client, on: bool) void {
        self.impl.setFollowLocation(on);
    }

    pub fn setDefaultHeaders(self: *Client, headers: Headers) void {
        self.impl.setDefaultHeaders(headers);
    }

    pub fn setConnectionTimeout(self: *Client, sec: i64, usec: i64) void {
        self.impl.setConnectionTimeout(sec, usec);
    }

    pub fn setReadTimeout(self: *Client, sec: i64, usec: i64) void {
        self.impl.setReadTimeout(sec, usec);
    }

    pub fn setWriteTimeout(self: *Client, sec: i64, usec: i64) void {
        self.impl.setWriteTimeout(sec, usec);
    }

    fn run(self: *Client, req: *const Request) Result {
        return self.impl.send(req) catch .{ .res = null, .err = .unknown };
    }

    /// Ports `ClientImpl`'s `Get` (httplib.cpp:11826 @d81235049), with headers.
    pub fn get(self: *Client, a: std.mem.Allocator, path: []const u8, headers: *const Headers) Result {
        const req: Request = .{ .a = a, .method = "GET", .path = path, .headers = headers.clone(a) catch return .{ .res = null, .err = .unknown } };
        return self.run(&req);
    }

    /// Ports `ClientImpl`'s `Head` (httplib.cpp:11911 @d81235049).
    pub fn head(self: *Client, a: std.mem.Allocator, path: []const u8) Result {
        const req: Request = .{ .a = a, .method = "HEAD", .path = path };
        return self.run(&req);
    }

    /// Ports `ClientImpl`'s `Get` (httplib.cpp:11862 @d81235049), with a response handler
    /// and a content receiver.
    pub fn getStream(
        self: *Client,
        a: std.mem.Allocator,
        path: []const u8,
        headers: *const Headers,
        ctx: *anyopaque,
        on_response: ?*const fn (ctx: *anyopaque, r: *const Response) bool,
        on_data: *const fn (ctx: *anyopaque, data: []const u8) bool,
    ) Result {
        const Bind = struct {
            ctx: *anyopaque,
            on_response: ?*const fn (ctx: *anyopaque, r: *const Response) bool,
            on_data: *const fn (ctx: *anyopaque, data: []const u8) bool,
            fn rh(c: ?*anyopaque, r: *const Response) bool {
                const b: *@This() = @ptrCast(@alignCast(c.?));
                return b.on_response.?(b.ctx, r);
            }
            fn cr(c: ?*anyopaque, d: []const u8) bool {
                const b: *@This() = @ptrCast(@alignCast(c.?));
                return b.on_data(b.ctx, d);
            }
        };
        var bind: Bind = .{ .ctx = ctx, .on_response = on_response, .on_data = on_data };
        const req: Request = .{
            .a = a,
            .method = "GET",
            .path = path,
            .headers = headers.clone(a) catch return .{ .res = null, .err = .unknown },
            .response_handler = if (on_response != null) .{ .ctx = &bind, .func = Bind.rh } else null,
            .content_receiver = .{ .ctx = &bind, .func = Bind.cr },
        };
        return self.run(&req);
    }

    /// Ports `ClientImpl`'s `Post` (httplib.cpp:11940 @d81235049), a body and its
    /// type, through `send_with_content_provider_and_receiver`
    /// (httplib.cpp:11491 @d81235049).
    pub fn post(self: *Client, a: std.mem.Allocator, path: []const u8, body: []const u8, content_type: []const u8) Result {
        var req: Request = .{ .a = a, .method = "POST", .path = path, .body = body };
        if (content_type.len != 0) req.setHeader("Content-Type", content_type) catch return .{ .res = null, .err = .unknown };
        return self.run(&req);
    }

    /// Ports `ClientImpl`'s `Delete` (httplib.cpp:12566 @d81235049), with no body.
    pub fn delete(self: *Client, a: std.mem.Allocator, path: []const u8) Result {
        const req: Request = .{ .a = a, .method = "DELETE", .path = path };
        return self.run(&req);
    }
};

/// Ports `detail`'s `redirect` (httplib.cpp:4245 @d81235049): the request again at
/// `path`, through `cli`; a 303 turns anything but `GET`/`HEAD` into a
/// bare `GET`.
fn redirectWith(cli: *ClientImpl, req: *Request, res: *Response, path: []const u8, location: []const u8, err: *Error) Oom!bool {
    var new_req = try req.clone();
    new_req.path = path;
    new_req.redirect_count -= 1;

    if (res.status == 303 and (!std.mem.eql(u8, req.method, "GET") and !std.mem.eql(u8, req.method, "HEAD"))) {
        new_req.method = "GET";
        new_req.body = "";
        new_req.headers = .{};
    }

    var new_res: Response = .{ .a = res.a };

    const ret = try cli.sendInto(&new_req, &new_res, err);
    if (ret) {
        req.* = new_req;
        res.* = new_res;
        if (res.location.len == 0) res.location = location;
    }
    return ret;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "encode_request_target and parse_url" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("/a%20b?x=1&y", try encodeRequestTarget(a, "/a b?x=1&y", true));
    var uc: UrlComponents = undefined;
    try std.testing.expect(parseUrl("http://h:81/p?q#f", &uc));
    try std.testing.expectEqualStrings("h", uc.host);
    try std.testing.expectEqualStrings("81", uc.port);
    try std.testing.expectEqualStrings("/p", uc.path);
    try std.testing.expectEqualStrings("?q", uc.query);
}

test "resolve_relative_location and parse_port answer as cpp-httplib 0.59.0" {
    // expected values printed by `detail::resolve_relative_location` and
    // `detail::parse_port` from llama.cpp/vendor/cpp-httplib/httplib.h
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][3][]const u8{
        .{ "../g", "/b/c/d;p?q", "/b/g" },
        .{ "g?y#s", "/b/c/d;p?q", "/b/c/g?y#s" },
        .{ "?y", "/b/c/d;p?q#f", "/b/c/d;p?y" },
        .{ "#s", "/a/b?x#y", "/a/b?x#s" },
        .{ "../../../../g", "/b/c/d", "/g" },
        .{ "g/..", "/b/c", "/b/" },
        .{ "a:b/c", "/x", "a:b/c" },
        .{ "a/b:c", "/x/y", "/x/a/b:c" },
        .{ "g", "noslash", "/g" },
    };
    for (cases) |cs| try std.testing.expectEqualStrings(cs[2], try resolveRelativeLocation(a, cs[0], cs[1]));
    var port: c_int = -7;
    try std.testing.expect(!parsePort("8080x", &port));
    try std.testing.expect(!parsePort("+80", &port));
    try std.testing.expectEqual(@as(c_int, -7), port);
    try std.testing.expect(parsePort("65535", &port));
    try std.testing.expectEqual(@as(c_int, 65535), port);
    // 0.59.0: every control octet and DEL is escaped, CR and LF among them
    try std.testing.expectEqualStrings("/a%01b%7F%0D%0A%20%C3%A9%09", try encodePath(a, "/a\x01b\x7f\r\n \xc3\xa9\t"));
}
