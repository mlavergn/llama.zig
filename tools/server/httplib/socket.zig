//! cpp-httplib's sockets as the server half reaches them: the BSD calls,
//! readiness polling, `SocketStream` and `BufferStream`, the line reader,
//! the keep-alive loop, binding a listening socket and closing one.
//!
//! # Provenance
//!
//! Ports parts of `llama.cpp/vendor/cpp-httplib/httplib.cpp` (cpp-httplib
//! 0.66.0, vendored at v1.0.0, `b1b770d90`): the POSIX arm of each `#if
//! _WIN32`. Each declaration names what it ports and the line it began at.
//!
//! # The C library, called directly
//!
//! The BSD socket calls are declared here against libc with the C's types,
//! and their constants written out per OS, rather than taken from `std.c`:
//! the port has to make exactly the calls the C makes, in the C's order,
//! with the C's `errno` afterwards. Darwin and Linux are covered; Windows
//! (Winsock) is not yet.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const builtin = @import("builtin");

const is_darwin = builtin.os.tag.isDarwin();

/// `socket_t`.
pub const Socket = c_int;
/// `INVALID_SOCKET`.
pub const invalid_socket: Socket = -1;

/// The `<sys/socket.h>`, `<netinet/*.h>`, `<poll.h>`, `<errno.h>` and
/// `<netdb.h>` constants the C passes, by OS.
pub const k = if (is_darwin) struct {
    pub const SOL_SOCKET: c_int = 0xffff;
    pub const SO_REUSEADDR: c_int = 0x4;
    pub const SO_REUSEPORT: c_int = 0x200;
    pub const SO_SNDTIMEO: c_int = 0x1005;
    pub const SO_RCVTIMEO: c_int = 0x1006;
    pub const IPPROTO_TCP: c_int = 6;
    pub const TCP_NODELAY: c_int = 1;
    pub const IPPROTO_IPV6: c_int = 41;
    pub const IPV6_V6ONLY: c_int = 27;
    pub const AF_UNSPEC: c_int = 0;
    pub const AF_UNIX: c_int = 1;
    pub const AF_INET: c_int = 2;
    pub const AF_INET6: c_int = 30;
    pub const SOCK_STREAM: c_int = 1;
    pub const SOCK_CLOEXEC: ?c_int = null;
    pub const POLLIN: c_short = 0x1;
    pub const POLLOUT: c_short = 0x4;
    pub const MSG_PEEK: c_int = 0x2;
    pub const SHUT_WR: c_int = 1;
    pub const SHUT_RDWR: c_int = 2;
    pub const EINTR: c_int = 4;
    pub const EBADF: c_int = 9;
    pub const ENOMEM: c_int = 12;
    pub const ENFILE: c_int = 23;
    pub const EMFILE: c_int = 24;
    pub const EAGAIN: c_int = 35;
    pub const EWOULDBLOCK: c_int = 35;
    pub const ECONNABORTED: c_int = 53;
    pub const ENOBUFS: c_int = 55;
    pub const NI_NUMERICHOST: c_int = 0x2;
    pub const AI_NUMERICHOST: c_int = 0x4;
    pub const NI_MAXHOST: usize = 1025;
    pub const F_SETFD: c_int = 2;
    pub const FD_CLOEXEC: c_int = 1;
    pub const SIGPIPE: c_int = 13;
    pub const sun_path_len: usize = 104;
} else struct {
    pub const SOL_SOCKET: c_int = 1;
    pub const SO_REUSEADDR: c_int = 2;
    pub const SO_REUSEPORT: c_int = 15;
    pub const SO_SNDTIMEO: c_int = 21;
    pub const SO_RCVTIMEO: c_int = 20;
    pub const IPPROTO_TCP: c_int = 6;
    pub const TCP_NODELAY: c_int = 1;
    pub const IPPROTO_IPV6: c_int = 41;
    pub const IPV6_V6ONLY: c_int = 26;
    pub const AF_UNSPEC: c_int = 0;
    pub const AF_UNIX: c_int = 1;
    pub const AF_INET: c_int = 2;
    pub const AF_INET6: c_int = 10;
    pub const SOCK_STREAM: c_int = 1;
    pub const SOCK_CLOEXEC: ?c_int = 0o2000000;
    pub const POLLIN: c_short = 0x1;
    pub const POLLOUT: c_short = 0x4;
    pub const MSG_PEEK: c_int = 0x2;
    pub const SHUT_WR: c_int = 1;
    pub const SHUT_RDWR: c_int = 2;
    pub const EINTR: c_int = 4;
    pub const EBADF: c_int = 9;
    pub const ENOMEM: c_int = 12;
    pub const ENFILE: c_int = 23;
    pub const EMFILE: c_int = 24;
    pub const EAGAIN: c_int = 11;
    pub const EWOULDBLOCK: c_int = 11;
    pub const ECONNABORTED: c_int = 103;
    pub const ENOBUFS: c_int = 105;
    pub const NI_NUMERICHOST: c_int = 1;
    pub const AI_NUMERICHOST: c_int = 4;
    pub const NI_MAXHOST: usize = 1025;
    pub const F_SETFD: c_int = 2;
    pub const FD_CLOEXEC: c_int = 1;
    pub const SIGPIPE: c_int = 13;
    pub const sun_path_len: usize = 108;
};

/// `struct addrinfo`, std's per-OS layout.
pub const AddrInfo = std.c.addrinfo;

/// `struct sockaddr_storage`: 128 bytes; the family is a byte after the
/// length on Darwin and a `u16` on Linux.
const SockaddrStorage = extern struct {
    bytes: [128]u8 align(8) = @splat(0),

    fn family(self: *const SockaddrStorage) c_int {
        return if (is_darwin) self.bytes[1] else std.mem.readInt(u16, self.bytes[0..2], builtin.cpu.arch.endian());
    }

    /// `sin_port` and `sin6_port` both sit at offset 2, in network order.
    fn port(self: *const SockaddrStorage) c_int {
        return std.mem.readInt(u16, self.bytes[2..4], .big);
    }
};

/// The calling thread's `errno`.
pub fn errno() c_int {
    return std.c._errno().*;
}

/// `signal(SIGPIPE, SIG_IGN)`, as `Server`'s constructor does
/// (httplib.cpp:8726 @b1b770d90).
pub fn ignoreSigpipe() void {
    _ = stdlib.libc.signal(k.SIGPIPE, 1); // SIG_IGN
}

/// Ports `set_socket_opt_impl` (httplib.cpp:515 @b1b770d90).
fn setSocketOptImpl(sock: Socket, level: c_int, optname: c_int, optval: *const anyopaque, optlen: u32) bool {
    return stdlib.libc.setsockopt(sock, level, @intCast(optname), optval, optlen) == 0;
}

/// Ports `set_socket_opt_time` (httplib.cpp:526 @b1b770d90).
pub fn setSocketOptTime(sock: Socket, level: c_int, optname: c_int, sec: i64, usec: i64) bool {
    const timeout: std.c.timeval = .{ .sec = @intCast(sec), .usec = @intCast(usec) };
    return setSocketOptImpl(sock, level, optname, &timeout, @sizeOf(std.c.timeval));
}

/// Ports `set_socket_opt` (httplib.cpp:6653 @b1b770d90).
pub fn setSocketOpt(sock: Socket, level: c_int, optname: c_int, optval: c_int) bool {
    return setSocketOptImpl(sock, level, optname, &optval, @sizeOf(c_int));
}

/// Ports `close_socket` (httplib.cpp:1795 @b1b770d90).
pub fn closeSocket(sock: Socket) c_int {
    return stdlib.libc.close(sock);
}

/// Ports `shutdown_socket` (httplib.cpp:2118 @b1b770d90).
pub fn shutdownSocket(sock: Socket) c_int {
    return stdlib.libc.shutdown(sock, k.SHUT_RDWR);
}

/// Ports `read_socket` (httplib.cpp:1830 @b1b770d90), with `handle_EINTR`
/// (httplib.cpp:1803 @b1b770d90): retried after a microsecond while
/// interrupted.
pub fn readSocket(sock: Socket, ptr: [*]u8, size: usize, flags: c_int) isize {
    while (true) {
        const res = stdlib.libc.recv(sock, ptr, size, flags);
        if (res < 0 and errno() == k.EINTR) {
            stdlib.clock.sleepUs(1);
            continue;
        }
        return res;
    }
}

/// Ports `send_socket` (httplib.cpp:1842 @b1b770d90), under `handle_EINTR`.
pub fn sendSocket(sock: Socket, ptr: [*]const u8, size: usize, flags: c_int) isize {
    while (true) {
        const res = stdlib.libc.send(sock, ptr, size, @bitCast(flags));
        if (res < 0 and errno() == k.EINTR) {
            stdlib.clock.sleepUs(1);
            continue;
        }
        return res;
    }
}

/// Ports `select_impl` (httplib.cpp:1863 @b1b770d90): `poll` on one socket;
/// a negative `sec` waits forever.
fn selectImpl(sock: Socket, events: c_short, sec: i64, usec: i64) isize {
    var pfd: std.c.pollfd = .{ .fd = sock, .events = events, .revents = 0 };
    const timeout: c_int = if (sec < 0) -1 else @intCast(sec * 1000 + @divTrunc(usec, 1000));
    while (true) {
        const res = stdlib.libc.poll((&pfd)[0..1], 1, timeout);
        if (res < 0 and errno() == k.EINTR) {
            stdlib.clock.sleepUs(1);
            continue;
        }
        return res;
    }
}

/// Ports `select_read` (httplib.cpp:1877 @b1b770d90).
pub fn selectRead(sock: Socket, sec: i64, usec: i64) isize {
    return selectImpl(sock, k.POLLIN, sec, usec);
}

/// Ports `select_write` (httplib.cpp:1881 @b1b770d90).
pub fn selectWrite(sock: Socket, sec: i64, usec: i64) isize {
    return selectImpl(sock, k.POLLOUT, sec, usec);
}

/// `std::function<bool()>`: `set_read_interrupt`'s callback.
pub const ReadInterrupt = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque) bool,
};

/// Ports `select_read_interruptible` (httplib.cpp:1912 @b1b770d90):
/// `select_read`, but waits in slices and gives up once `interrupted`
/// returns true. The last wait is rounded up to a whole millisecond, as
/// `poll` takes milliseconds and would otherwise end before the deadline.
/// A `sec` below zero never comes (`deadline_after`, httplib.cpp:1887
/// @b1b770d90), as it waits forever in `select_impl`.
pub fn selectReadInterruptible(sock: Socket, sec: i64, usec: i64, interrupted: ?ReadInterrupt) isize {
    const it = interrupted orelse return selectRead(sock, sec, usec);
    const slice: i64 = 100000;
    const deadline: i64 = if (sec < 0) std.math.maxInt(i64) else stdlib.clock.steadyUs() + sec * 1000000 + usec;
    while (!it.func(it.ctx)) {
        const left = deadline -| stdlib.clock.steadyUs();
        const last = left < slice;
        // C++ integer division truncates toward zero.
        const wait_usec = if (last) @max(@divTrunc(left + 999, 1000) * 1000, 0) else slice;
        const ret = selectImpl(sock, k.POLLIN, 0, wait_usec);
        if (ret != 0 or last) return ret;
    }
    return 0;
}

/// Ports `is_socket_alive` (httplib.cpp:1959 @b1b770d90): nothing to read,
/// or a byte to peek at, means the peer is still there.
pub fn isSocketAlive(sock: Socket) bool {
    const val = selectRead(sock, 0, 0);
    if (val == 0) return true;
    if (val < 0 and errno() == k.EBADF) return false;
    var buf: [1]u8 = undefined;
    return readSocket(sock, &buf, 1, k.MSG_PEEK) > 0;
}

/// Ports `is_accept_resource_error` (httplib.cpp:2677 @b1b770d90).
pub fn isAcceptResourceError() bool {
    const err = errno();
    return err == k.EMFILE or err == k.ENFILE or err == k.ENOBUFS or err == k.ENOMEM;
}

/// Ports `is_accept_transient_error` (httplib.cpp:2692 @b1b770d90).
pub fn isAcceptTransientError() bool {
    const err = errno();
    return err == k.EINTR or err == k.EAGAIN or err == k.EWOULDBLOCK or err == k.ECONNABORTED;
}

/// `accept`, as `listen_internal` calls it on a target without
/// `SOCK_CLOEXEC`, or `accept4` with it.
pub fn acceptSocket(sock: Socket) Socket {
    if (k.SOCK_CLOEXEC) |cloexec| return std.c.accept4(sock, null, null, @intCast(cloexec));
    return stdlib.libc.accept(sock, null, null);
}

/// Ports `drain_and_close_socket` (httplib.cpp:2131 @b1b770d90): half-close,
/// read off what the peer still sends for up to 100 ms or 1 MiB, then shut
/// down and close, so the peer sees a FIN rather than a reset.
pub fn drainAndCloseSocket(sock: Socket) void {
    _ = stdlib.libc.shutdown(sock, k.SHUT_WR);
    var buf: [mod.detail.recv_bufsiz]u8 = undefined;
    var total: usize = 0;
    const deadline = stdlib.clock.steadyUs() + 100 * 1000;
    while (total < 1024 * 1024) {
        const remaining = deadline - stdlib.clock.steadyUs();
        if (remaining <= 0) break;
        if (selectRead(sock, 0, remaining) <= 0) break;
        const n = readSocket(sock, &buf, buf.len, 0);
        if (n <= 0) break;
        total += @intCast(n);
    }
    _ = shutdownSocket(sock);
    _ = closeSocket(sock);
}

/// Ports `get_ip_and_port` (httplib.cpp:2833 @b1b770d90).
fn getIpAndPort(a: std.mem.Allocator, addr: *const SockaddrStorage, addr_len: u32, ip: *[]const u8, port: *c_int) error{OutOfMemory}!bool {
    const fam = addr.family();
    if (fam != k.AF_INET and fam != k.AF_INET6) return false;
    port.* = addr.port();
    var ipstr: [k.NI_MAXHOST]u8 = @splat(0);
    if (@backingInt(stdlib.libc.getnameinfo(@ptrCast(addr), addr_len, &ipstr, ipstr.len, null, 0, .{ .NUMERICHOST = true })) != 0) return false;
    ip.* = try a.dupe(u8, std.mem.sliceTo(&ipstr, 0));
    return true;
}

/// Ports `is_ip_address` (httplib.cpp:5804 @b1b770d90): an IPv4 or IPv6
/// literal, as `inet_pton` reads one.
pub fn isIpAddress(a: std.mem.Allocator, host: []const u8) error{OutOfMemory}!bool {
    const hz = try a.dupeSentinel(u8, host, 0);
    defer a.free(hz);
    var addr4: [4]u8 = undefined;
    var addr6: [16]u8 = undefined;
    return stdlib.libc.inet_pton(k.AF_INET, hz.ptr, &addr4) == 1 or stdlib.libc.inet_pton(k.AF_INET6, hz.ptr, &addr6) == 1;
}

/// Ports `hosted_at` (httplib.cpp:6809 @b1b770d90), the overload filling a
/// vector: every address `hostname` resolves to over `SOCK_STREAM`, as
/// `get_ip_and_port` spells it, appended to `addrs`. A failed lookup
/// appends nothing and, on Linux other than Android, calls `res_init` so
/// that a changed `resolv.conf` is read next time. `getaddrinfo_with_timeout`
/// with no timeout is `getaddrinfo`.
///
/// Parameters:
///   a: allocates `addrs`' strings, which the caller owns.
///   hostname: the name or literal to resolve.
///   addrs: appended to.
pub fn hostedAt(a: std.mem.Allocator, hostname: []const u8, addrs: *std.ArrayList([]const u8)) error{OutOfMemory}!void {
    var hints: AddrInfo = std.mem.zeroes(AddrInfo);
    hints.family = k.AF_UNSPEC;
    hints.socktype = k.SOCK_STREAM;
    hints.protocol = 0;

    const hz = try a.dupeSentinel(u8, hostname, 0);
    defer a.free(hz);
    var result: ?*AddrInfo = null;
    if (@backingInt(stdlib.libc.getaddrinfo(hz.ptr, null, &hints, &result)) != 0) {
        if (builtin.os.tag == .linux and !builtin.abi.isAndroid()) _ = stdlib.libc.__res_init();
        return;
    }
    defer if (result) |r| stdlib.libc.freeaddrinfo(r);

    var rp = result;
    while (rp) |ai| : (rp = ai.next) {
        const addr: *const SockaddrStorage = @ptrCast(@alignCast(ai.addr.?));
        var ip: []const u8 = "";
        var dummy: c_int = -1;
        if (try getIpAndPort(a, addr, @sizeOf(SockaddrStorage), &ip, &dummy)) {
            try addrs.append(a, ip);
        }
    }
}

/// Ports `get_local_ip_and_port` (httplib.cpp:2855 @b1b770d90).
pub fn getLocalIpAndPort(a: std.mem.Allocator, sock: Socket, ip: *[]const u8, port: *c_int) error{OutOfMemory}!void {
    var addr: SockaddrStorage = .{};
    var len: u32 = @sizeOf(SockaddrStorage);
    if (stdlib.libc.getsockname(sock, @ptrCast(&addr), &len) == 0) _ = try getIpAndPort(a, &addr, len, ip, port);
}

/// Ports `get_remote_ip_and_port` (httplib.cpp:2864 @b1b770d90). A Unix
/// socket's peer is left alone: Linux's `SO_PEERCRED` pid is not taken, and
/// Darwin defines no `SO_PEERPID`, so the C takes nothing there either.
pub fn getRemoteIpAndPort(a: std.mem.Allocator, sock: Socket, ip: *[]const u8, port: *c_int) error{OutOfMemory}!void {
    var addr: SockaddrStorage = .{};
    var len: u32 = @sizeOf(SockaddrStorage);
    if (stdlib.libc.getpeername(sock, @ptrCast(&addr), &len) == 0) {
        if (addr.family() == k.AF_UNIX) return;
        _ = try getIpAndPort(a, &addr, len, ip, port);
    }
}

/// `SocketOptions`: a callback run on each socket `create_socket` makes.
pub const SocketOptions = struct {
    ctx: ?*anyopaque = null,
    func: *const fn (ctx: ?*anyopaque, sock: Socket) void,

    pub fn call(self: SocketOptions, sock: Socket) void {
        self.func(self.ctx, sock);
    }
};

/// Ports `default_socket_options` (httplib.cpp:6643 @b1b770d90).
pub fn defaultSocketOptions(_: ?*anyopaque, sock: Socket) void {
    _ = setSocketOpt(sock, k.SOL_SOCKET, k.SO_REUSEPORT, 1);
}

/// Ports `unescape_abstract_namespace_unix_domain` (httplib.cpp:2169
/// @b1b770d90): a leading `@` names Linux's abstract namespace.
fn unescapeAbstractNamespaceUnixDomain(dst: []u8, s: []const u8) void {
    @memcpy(dst[0..s.len], s);
    if (s.len > 1 and s[0] == '@') dst[0] = 0;
}

/// What `create_socket` does with each candidate address: `bind` and
/// `listen` it, for the server.
pub const BindOrConnect = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, sock: Socket, addr: ?*const anyopaque, addrlen: u32, quit: *bool) bool,
};

/// Ports `create_socket` (httplib.cpp:2511 @b1b770d90): with an empty
/// `ip`, `host` goes to `getaddrinfo` with `address_family` and
/// `socket_flags`, or -- for `AF_UNIX` -- straight into a `sockaddr_un`; a
/// non-empty `ip` (a client's address map naming an IP literal) is
/// converted with `AI_NUMERICHOST` over `AF_UNSPEC` instead. No DNS timeout
/// is passed: `CPPHTTPLIB_USE_NON_BLOCKING_GETADDRINFO` is not defined, so
/// `getaddrinfo_with_timeout` is `getaddrinfo`.
///
/// Return: the bound socket, or `invalid_socket`.
pub fn createSocket(
    host: [:0]const u8,
    ip: [:0]const u8,
    port: c_int,
    address_family: c_int,
    socket_flags: c_int,
    tcp_nodelay: bool,
    ipv6_v6only: bool,
    socket_options: ?SocketOptions,
    bind_or_connect: BindOrConnect,
) Socket {
    var hints: AddrInfo = std.mem.zeroes(AddrInfo);
    hints.socktype = k.SOCK_STREAM;
    hints.protocol = 0; // IPPROTO_IP
    var node: ?[*:0]const u8 = null;
    if (ip.len != 0) {
        node = ip.ptr;
        // Ask getaddrinfo to convert IP in c-string to address
        hints.family = k.AF_UNSPEC;
        hints.flags = @bitCast(@as(u32, @intCast(k.AI_NUMERICHOST)));
    } else {
        if (host.len != 0) node = host.ptr;
        hints.family = address_family;
        hints.flags = @bitCast(@as(u32, @bitCast(socket_flags)));
    }

    if (hints.family == k.AF_UNIX) {
        const addrlen = host.len;
        if (addrlen > k.sun_path_len) return invalid_socket;
        var sock = stdlib.libc.socket(@intCast(hints.family), @intCast(hints.socktype | (k.SOCK_CLOEXEC orelse 0)), @intCast(hints.protocol));
        if (sock != invalid_socket) {
            // sockaddr_un: Darwin leads with a length byte
            var addr: [2 + k.sun_path_len]u8 align(8) = @splat(0);
            if (is_darwin) {
                addr[1] = @intCast(k.AF_UNIX);
            } else {
                std.mem.writeInt(u16, addr[0..2], @intCast(k.AF_UNIX), builtin.cpu.arch.endian());
            }
            unescapeAbstractNamespaceUnixDomain(addr[2..], host);
            const len: u32 = @intCast(2 + addrlen);
            if (k.SOCK_CLOEXEC == null) _ = stdlib.libc.fcntl(sock, k.F_SETFD, k.FD_CLOEXEC);
            if (socket_options) |so| so.call(sock);
            var dummy = false;
            if (!bind_or_connect.func(bind_or_connect.ctx, sock, &addr, len, &dummy)) {
                _ = closeSocket(sock);
                sock = invalid_socket;
            }
        }
        return sock;
    }

    var service_buf: [16]u8 = @splat(0);
    const service_len = (std.fmt.bufPrint(service_buf[0..15], "{d}", .{port}) catch |err| switch (err) {
        error.NoSpaceLeft => unreachable,
    }).len;
    const service: [:0]const u8 = service_buf[0..service_len :0];

    var result: ?*AddrInfo = null;
    if (@backingInt(stdlib.libc.getaddrinfo(node, service.ptr, &hints, &result)) != 0) return invalid_socket;
    defer if (result) |r| stdlib.libc.freeaddrinfo(r);

    var rp = result;
    while (rp) |ai| : (rp = ai.next) {
        const sock = stdlib.libc.socket(@intCast(ai.family), @intCast(ai.socktype | (k.SOCK_CLOEXEC orelse 0)), @intCast(ai.protocol));
        if (sock == invalid_socket) continue;
        if (k.SOCK_CLOEXEC == null) {
            if (stdlib.libc.fcntl(sock, k.F_SETFD, k.FD_CLOEXEC) == -1) {
                _ = closeSocket(sock);
                continue;
            }
        }
        if (tcp_nodelay) _ = setSocketOpt(sock, k.IPPROTO_TCP, k.TCP_NODELAY, 1);
        if (ai.family == k.AF_INET6) _ = setSocketOpt(sock, k.IPPROTO_IPV6, k.IPV6_V6ONLY, if (ipv6_v6only) 1 else 0);
        if (socket_options) |so| so.call(sock);
        var quit = false;
        if (bind_or_connect.func(bind_or_connect.ctx, sock, ai.addr, ai.addrlen, &quit)) return sock;
        _ = closeSocket(sock);
        if (quit) break;
    }
    return invalid_socket;
}

/// `::bind` then `::listen`, as `create_server_socket`'s lambda does.
pub fn bindAndListen(sock: Socket, addr: ?*const anyopaque, addrlen: u32, backlog: c_uint) error{ Bind, Listen }!void {
    if (stdlib.libc.bind(sock, @ptrCast(@alignCast(addr)), addrlen) != 0) return error.Bind;
    if (stdlib.libc.listen(sock, backlog) != 0) return error.Listen;
}

/// The port a bound socket listens on, as `bind_internal` reads it.
///
/// Return: the port, or an error naming which step failed.
pub fn boundPort(sock: Socket) error{ GetSockName, UnsupportedAddressFamily }!c_int {
    var addr: SockaddrStorage = .{};
    var len: u32 = @sizeOf(SockaddrStorage);
    if (stdlib.libc.getsockname(sock, @ptrCast(&addr), &len) == -1) return error.GetSockName;
    const fam = addr.family();
    if (fam == k.AF_INET or fam == k.AF_INET6) return addr.port();
    return error.UnsupportedAddressFamily;
}

// -----------------------------------------------------------------------------
// Streams

/// What a `Stream` remembers of its last failure; only `get_error`'s
/// readers -- `BodyReader` -- look.
pub const StreamError = enum { success, timeout, connection_closed, read };

/// Ports `SocketStream` (httplib.cpp:7830 @b1b770d90), the server's stream:
/// reads through a 4 KiB buffer of its own, writes straight to the socket
/// once it polls writable. One is made per request, so bytes buffered past
/// the end of one request are dropped with it, as the C drops them.
pub const SocketStream = struct {
    sock: Socket,
    read_timeout_sec: i64,
    read_timeout_usec: i64,
    write_timeout_sec: i64,
    write_timeout_usec: i64,
    read_buff: [read_buff_size]u8 = undefined,
    read_buff_off: usize = 0,
    read_buff_content_size: usize = 0,
    readable_hint: bool = false,
    err: StreamError = .success,
    /// `max_timeout_msec_`; `set_max_timeout` restarts `start_time_us`.
    max_timeout_msec: i64 = 0,
    start_time_us: i64 = 0,
    /// `read_interrupted_`.
    read_interrupted: ?ReadInterrupt = null,

    const read_buff_size: usize = 1024 * 4;

    /// Ports `is_readable` (httplib.cpp:7844 @b1b770d90), a member of
    /// `SocketStream`: buffered bytes remain.
    pub fn isReadable(self: *const SocketStream) bool {
        return self.read_buff_off < self.read_buff_content_size;
    }

    /// Ports `wait_readable` (httplib.cpp:7848 @b1b770d90), a member of
    /// `SocketStream`, under `calc_actual_timeout` (httplib.cpp:7814
    /// @b1b770d90).
    fn waitReadable(self: *const SocketStream) bool {
        if (self.max_timeout_msec <= 0) {
            return selectReadInterruptible(self.sock, self.read_timeout_sec, self.read_timeout_usec, self.read_interrupted) > 0;
        }
        const duration_msec = @divTrunc(stdlib.clock.steadyUs() - self.start_time_us, 1000);
        const timeout_msec = self.read_timeout_sec * 1000 + @divTrunc(self.read_timeout_usec, 1000);
        var actual = @min(self.max_timeout_msec - duration_msec, timeout_msec);
        if (actual < 0) actual = 0;
        return selectReadInterruptible(self.sock, @divTrunc(actual, 1000), @rem(actual, 1000) * 1000, self.read_interrupted) > 0;
    }

    /// Ports `set_read_timeout` (httplib.cpp:7982 @b1b770d90), a member of
    /// `SocketStream`.
    pub fn setReadTimeout(self: *SocketStream, sec: i64, usec: i64) void {
        self.read_timeout_sec = sec;
        self.read_timeout_usec = usec;
    }

    /// Ports `set_max_timeout` (httplib.cpp:7987 @b1b770d90), a member of
    /// `SocketStream`: bounds the reads from now on to `msec` in total, on
    /// top of the timeout of each one. 0 lifts the bound.
    pub fn setMaxTimeout(self: *SocketStream, msec: i64) void {
        self.max_timeout_msec = msec;
        self.start_time_us = stdlib.clock.steadyUs();
    }

    /// Ports `set_read_interrupt` (httplib.cpp:7993 @b1b770d90), a member
    /// of `SocketStream`.
    pub fn setReadInterrupt(self: *SocketStream, interrupted: ?ReadInterrupt) void {
        self.read_interrupted = interrupted;
    }

    /// Ports `wait_writable` (httplib.cpp:7863 @b1b770d90), a member of
    /// `SocketStream`.
    pub fn waitWritable(self: *const SocketStream) bool {
        return selectWrite(self.sock, self.write_timeout_sec, self.write_timeout_usec) > 0;
    }

    /// Ports `ensure_readable` (httplib.cpp:7867 @b1b770d90), a member of
    /// `SocketStream`: the keep-alive loop's readiness, once.
    fn ensureReadable(self: *SocketStream) bool {
        if (self.readable_hint) {
            self.readable_hint = false;
            return true;
        }
        return self.waitReadable();
    }

    /// Ports `buffered_data` (httplib.cpp:7875 @b1b770d90), a member of
    /// `SocketStream`.
    pub fn bufferedData(self: *const SocketStream) []const u8 {
        return self.read_buff[self.read_buff_off..self.read_buff_content_size];
    }

    /// Ports `consume_buffered` (httplib.cpp:7880 @b1b770d90), a member of
    /// `SocketStream`.
    pub fn consumeBuffered(self: *SocketStream, size: usize) void {
        self.read_buff_off += size;
    }

    /// Ports `is_peer_alive` (httplib.cpp:7885 @b1b770d90), a member of
    /// `SocketStream`.
    pub fn isPeerAlive(self: *const SocketStream) bool {
        return isSocketAlive(self.sock);
    }

    /// Ports `read` (httplib.cpp:7889 @b1b770d90), a member of
    /// `SocketStream`: buffered bytes first; otherwise, for a small read, a
    /// 4 KiB `recv` into the buffer, and for a large one straight into
    /// `ptr`.
    ///
    /// Return: bytes read, 0 at end of stream, -1 on a timeout or error.
    pub fn read(self: *SocketStream, ptr: []u8) isize {
        const size = @min(ptr.len, @as(usize, std.math.maxInt(isize)));
        if (self.read_buff_off < self.read_buff_content_size) {
            const remaining_size = self.read_buff_content_size - self.read_buff_off;
            const n = @min(size, remaining_size);
            @memcpy(ptr[0..n], self.read_buff[self.read_buff_off..][0..n]);
            self.read_buff_off += n;
            return @intCast(n);
        }
        if (!self.ensureReadable()) {
            self.err = .timeout;
            return -1;
        }
        self.read_buff_off = 0;
        self.read_buff_content_size = 0;
        if (size < read_buff_size) {
            const n = readSocket(self.sock, &self.read_buff, read_buff_size, 0);
            if (n <= 0) {
                self.err = if (n == 0) .connection_closed else .read;
                return n;
            } else if (n <= @as(isize, @intCast(size))) {
                const un: usize = @intCast(n);
                @memcpy(ptr[0..un], self.read_buff[0..un]);
                return n;
            } else {
                @memcpy(ptr[0..size], self.read_buff[0..size]);
                self.read_buff_off = size;
                self.read_buff_content_size = @intCast(n);
                return @intCast(size);
            }
        } else {
            const n = readSocket(self.sock, ptr.ptr, size, 0);
            if (n <= 0) self.err = if (n == 0) .connection_closed else .read;
            return n;
        }
    }

    /// Ports `write` (httplib.cpp:7951 @b1b770d90), a member of
    /// `SocketStream`: one `send` once the socket polls writable. The
    /// `_WIN32` clamp to `INT_MAX` is not here: this file is the POSIX arm,
    /// and llama-server is not built for Windows.
    ///
    /// Return: bytes sent, or -1.
    pub fn write(self: *SocketStream, data: []const u8) isize {
        if (!self.waitWritable()) return -1;
        return sendSocket(self.sock, data.ptr, data.len, 0);
    }
};

/// Ports `BufferStream` (httplib.h:3912 @b1b770d90), the write side the
/// server stages a response's head in: `write` (httplib.cpp:8014
/// @b1b770d90) appends, `get_buffer` (httplib.cpp:8029 @b1b770d90) is
/// `buffer.items`.
pub const BufferStream = struct {
    a: std.mem.Allocator,
    buffer: std.ArrayList(u8) = .empty,
    oom: bool = false,

    pub fn write(self: *BufferStream, data: []const u8) isize {
        self.buffer.appendSlice(self.a, data) catch {
            self.oom = true;
            return -1;
        };
        return @intCast(data.len);
    }
};

/// Ports `stream_line_reader` (httplib.h:4051 @b1b770d90) over a
/// `SocketStream`: one line at a time into a fixed buffer, spilling into a
/// growable one past it.
pub const LineReader = struct {
    strm: *SocketStream,
    fixed: []u8,
    fixed_used: usize = 0,
    growable: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, strm: *SocketStream, fixed: []u8) LineReader {
        return .{ .strm = strm, .fixed = fixed, .a = a };
    }

    pub fn deinit(self: *LineReader) void {
        self.growable.deinit(self.a);
    }

    /// Ports `ptr` (httplib.cpp:1571 @b1b770d90), a member of
    /// `stream_line_reader`, with `size` (httplib.cpp:1579 @b1b770d90):
    /// the line read so far.
    pub fn line(self: *const LineReader) []const u8 {
        if (self.growable.items.len == 0) return self.fixed[0..self.fixed_used];
        return self.growable.items;
    }

    /// The line as the C's `ptr()` hands it to a `const char *` reader:
    /// up to its first NUL.
    pub fn cstr(self: *const LineReader) []const u8 {
        return std.mem.sliceTo(self.line(), 0);
    }

    /// Ports `end_with_crlf` (httplib.cpp:1587 @b1b770d90), a member of
    /// `stream_line_reader`.
    pub fn endWithCrlf(self: *const LineReader) bool {
        const l = self.line();
        return l.len >= 2 and l[l.len - 2] == '\r' and l[l.len - 1] == '\n';
    }

    /// Ports `append` (httplib.cpp:1674 @b1b770d90), a member of
    /// `stream_line_reader`.
    fn append(self: *LineReader, data: []const u8) error{OutOfMemory}!void {
        if (self.growable.items.len == 0 and self.fixed_used + data.len < self.fixed.len) {
            @memcpy(self.fixed[self.fixed_used..][0..data.len], data);
            self.fixed_used += data.len;
            self.fixed[self.fixed_used] = 0;
        } else {
            if (self.growable.items.len == 0) try self.growable.appendSlice(self.a, self.fixed[0..self.fixed_used]);
            try self.growable.appendSlice(self.a, data);
        }
    }

    /// Ports `getline` (httplib.cpp:1592 @b1b770d90), a member of
    /// `stream_line_reader`: through the first CRLF, scanning what the
    /// stream has buffered in one pass and reading a byte at a time
    /// otherwise. A bare LF does not end a line.
    ///
    /// Return: false at end of stream before any byte, on an error, or past
    /// `CPPHTTPLIB_MAX_LINE_LENGTH`.
    pub fn getline(self: *LineReader) error{OutOfMemory}!bool {
        self.fixed_used = 0;
        self.growable.clearRetainingCapacity();
        var prev_byte: u8 = 0;
        var i: usize = 0;
        while (true) : (i += 1) {
            const buffered = self.strm.bufferedData();
            if (buffered.len != 0) {
                var take = buffered.len;
                var terminated = false;
                var at: usize = 0;
                while (at < buffered.len) {
                    const pos = std.mem.findScalarPos(u8, buffered, at, '\n') orelse break;
                    if ((if (pos > 0) buffered[pos - 1] else prev_byte) == '\r') {
                        take = pos + 1;
                        terminated = true;
                        break;
                    }
                    at = pos + 1;
                }
                if (self.line().len + take > mod.detail.max_line_length) return false;
                prev_byte = buffered[take - 1];
                try self.append(buffered[0..take]);
                self.strm.consumeBuffered(take);
                i += take;
                if (terminated) return true;
                continue;
            }
            if (self.line().len >= mod.detail.max_line_length) return false;
            var byte: [1]u8 = undefined;
            const n = self.strm.read(&byte);
            if (n < 0) return false;
            if (n == 0) {
                if (i == 0) return false;
                break;
            }
            try self.append(&byte);
            if (prev_byte == '\r' and byte[0] == '\n') break;
            prev_byte = byte[0];
        }
        return true;
    }
};

/// Ports `keep_alive` (httplib.cpp:2023 @b1b770d90): waits, polling every
/// 10 ms, for the next request's first byte, until the server socket
/// closes or `keep_alive_timeout_sec` passes.
pub fn keepAlive(svr_sock: *const std.atomic.Value(Socket), sock: Socket, keep_alive_timeout_sec: i64) bool {
    const interval_usec = mod.detail.keepalive_timeout_check_interval_usecond;
    if (selectRead(sock, 0, interval_usec) > 0) return true;
    const start = stdlib.clock.steadyUs() - interval_usec;
    const timeout = keep_alive_timeout_sec * 1000000;
    while (true) {
        if (svr_sock.load(.seq_cst) == invalid_socket) break;
        const val = selectRead(sock, 0, interval_usec);
        if (val < 0) {
            break;
        } else if (val == 0) {
            if (stdlib.clock.steadyUs() - start > timeout) break;
        } else {
            return true;
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
