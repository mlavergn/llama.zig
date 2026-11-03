//! `--ui-mcp-proxy`: `/cors-proxy?url=...` forwards a request to another
//! server for the web UI, which a browser's CORS rules would otherwise
//! keep from reaching an MCP server.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-cors-proxy.h` (llama.cpp v0.6.0,
//! `d81235049`) -- `proxy_header_to_lower`, `proxy_request` and the two
//! handlers -- and `common_http_parse_url` from `llama.cpp/common/http.h`.
//! Each declaration names what it ports and the line it began at. The
//! forwarding itself is `proxy.zig`'s `server_http_proxy`.

const std = @import("std");
const tc = @import("tools_common");
const http = @import("http.zig");
const common = @import("common.zig");
const proxy = @import("proxy.zig");

const Allocator = std.mem.Allocator;
const srv = common.srv;
const throw = common.throw;

/// Ports `common_http_url` (common/http.h:15 @d81235049).
pub const HttpUrl = struct {
    scheme: []const u8 = "",
    user: []const u8 = "",
    password: []const u8 = "",
    host: []const u8 = "",
    port: c_int = 0,
    path: []const u8 = "",
};

/// `std::stoi`: `std::invalid_argument` for no digits, `std::out_of_range`
/// past an `int`.
fn stoi(a: Allocator, s: []const u8) !c_int {
    const zs = try a.dupeSentinel(u8, s, 0);
    return tc.arg.stoi(zs) catch {
        const w = tc.arg.what();
        if (std.mem.eql(u8, w, "stoi: no conversion")) return throw(error.InvalidArgument, "{s}", .{w});
        return throw(error.RuntimeError, "{s}", .{w});
    };
}

/// Ports `common_http_parse_url` (common/http.h:29 @d81235049): scheme, user and
/// password, host (an IPv6 literal unbracketed), port and path.
pub fn parseUrl(a: Allocator, url: []const u8) !HttpUrl {
    var parts: HttpUrl = .{};
    const scheme_end = std.mem.find(u8, url, "://") orelse return throw(error.RuntimeError, "invalid URL: no scheme", .{});
    parts.scheme = url[0..scheme_end];

    if (!std.mem.eql(u8, parts.scheme, "http") and !std.mem.eql(u8, parts.scheme, "https")) {
        return throw(error.RuntimeError, "unsupported URL scheme: {s}", .{parts.scheme});
    }

    var rest = url[scheme_end + 3 ..];
    if (std.mem.findScalar(u8, rest, '@')) |at_pos| {
        const auth = rest[0..at_pos];
        if (std.mem.findScalar(u8, auth, ':')) |colon_pos| {
            parts.user = auth[0..colon_pos];
            parts.password = auth[colon_pos + 1 ..];
        } else {
            parts.user = auth;
        }
        rest = rest[at_pos + 1 ..];
    }

    if (std.mem.findScalar(u8, rest, '/')) |slash_pos| {
        parts.host = rest[0..slash_pos];
        parts.path = rest[slash_pos..];
    } else {
        parts.host = rest;
        parts.path = "/";
    }

    // split the authority into host and optional port, a bracketed IPv6 literal keeps its inner colons (RFC 3986)
    var port_str: []const u8 = "";
    if (parts.host.len != 0 and parts.host[0] == '[') {
        const close = std.mem.findScalar(u8, parts.host, ']') orelse
            return throw(error.RuntimeError, "invalid IPv6 URL authority: {s}", .{parts.host});
        const after = parts.host[close + 1 ..];
        if (after.len != 0 and after[0] == ':') port_str = after[1..];
        parts.host = parts.host[1..close];
    } else if (std.mem.findScalar(u8, parts.host, ':')) |colon_pos| {
        port_str = parts.host[colon_pos + 1 ..];
        parts.host = parts.host[0..colon_pos];
    }

    if (port_str.len != 0) {
        parts.port = try stoi(a, port_str);
    } else if (std.mem.eql(u8, parts.scheme, "http")) {
        parts.port = 80;
    } else {
        parts.port = 443;
    }
    return parts;
}

/// Ports `proxy_header_to_lower` (server-cors-proxy.h:15 @d81235049).
fn proxyHeaderToLower(a: Allocator, header: []const u8) ![]u8 {
    return std.ascii.allocLowerString(a, header);
}

/// Ports `proxy_request` (server-cors-proxy.h:22 @d81235049): the `url`
/// parameter's target, and the request's `x-llama-server-proxy-header-*`
/// headers with that prefix removed, forwarded.
fn proxyRequest(req: *const http.Req, method: []const u8) !*http.Res {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const target_url = req.getParam("url", "");
    var parsed_url = try parseUrl(a, target_url);

    if (parsed_url.host.len == 0) return throw(error.RuntimeError, "invalid target URL: missing host", .{});
    if (parsed_url.path.len == 0) parsed_url.path = "/";
    if (parsed_url.password.len != 0) return throw(error.RuntimeError, "authentication in target URL is not supported", .{});
    if (!std.mem.eql(u8, parsed_url.scheme, "http") and !std.mem.eql(u8, parsed_url.scheme, "https")) {
        return throw(error.RuntimeError, "unsupported URL scheme in target URL: {s}", .{parsed_url.scheme});
    }

    {
        const m = try a.dupeSentinel(u8, method, 0);
        const sc = try a.dupeSentinel(u8, parsed_url.scheme, 0);
        const h = try a.dupeSentinel(u8, try http.commonHttpFormatHost(a, parsed_url.host), 0);
        const p = try a.dupeSentinel(u8, parsed_url.path, 0);
        srv(.inf, "proxy_request", "proxying %s request to %s://%s:%i%s\n", .{ m.ptr, sc.ptr, h.ptr, parsed_url.port, p.ptr });
    }

    var headers: http.StringMap = .{};
    const proxy_header_prefix = "x-llama-server-proxy-header-";
    for (req.headers.entries.items) |e| {
        const lowered_key = try proxyHeaderToLower(a, e.key);
        if (!std.mem.startsWith(u8, lowered_key, proxy_header_prefix)) continue;
        const new_key = e.key[proxy_header_prefix.len..];
        if (new_key.len == 0) continue;
        try headers.put(a, new_key, e.value);
    }

    return proxy.Proxy.create(
        method,
        parsed_url.scheme,
        parsed_url.host,
        parsed_url.port,
        parsed_url.path,
        &headers,
        req.body,
        &req.files,
        req.should_stop,
        600, // timeout_read (default to 10 minutes)
        600, // timeout_write (default to 10 minutes)
    );
}

/// Ports `proxy_handler_post` (server-cors-proxy.h:77 @d81235049).
pub const handler_post: http.Handler = .{ .ctx = null, .func = post };
/// Ports `proxy_handler_get` (server-cors-proxy.h:81 @d81235049).
pub const handler_get: http.Handler = .{ .ctx = null, .func = get };

fn post(_: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
    return proxyRequest(req, "POST");
}

fn get(_: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
    return proxyRequest(req, "GET");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "common_http_parse_url splits authority, port and path" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const u = try parseUrl(arena.allocator(), "http://u:p@[::1]:81/x?y");
    try std.testing.expectEqualStrings("::1", u.host);
    try std.testing.expectEqual(@as(c_int, 81), u.port);
    try std.testing.expectEqualStrings("/x?y", u.path);
    try std.testing.expectEqualStrings("p", u.password);
}
