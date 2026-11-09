//! cpp-httplib's WebSocket: frames, the `WebSocket` connection, and the
//! client `server_models::proxy_ws` opens to a child.
//!
//! # Provenance
//!
//! Ports `ws::WebSocket`, `ws::WebSocketClient`, `ws::impl`,
//! `detail::write_websocket_frame` and the handshake helpers from
//! `llama.cpp/vendor/cpp-httplib/httplib.cpp` and
//! `llama.cpp/vendor/cpp-httplib/httplib.h` (cpp-httplib 0.62.0, vendored at
//! v0.8.0, `8642541b2`). Each declaration names what it ports and the line it
//! began at.
//!
//! # What is not here
//!
//! `server-http.cpp` and `server-models.cpp` set nothing on either side: no
//! subprotocol selector, proxy, TLS, read timeout or ping setting. So the
//! ping interval and missed-pong count stay at their compile-time defaults
//! (30 seconds, and 0, which makes no pong check and sets no read
//! interrupt), the client's read timeout stays 0 (wait forever), and the
//! `set_proxy*`, TLS and `set_*timeout` setters are not ported.

const std = @import("std");
const builtin = @import("builtin");
const detail = @import("detail.zig");
const socket = @import("socket.zig");
const message = @import("message.zig");

const SocketStream = socket.SocketStream;
const Headers = detail.Headers;

/// `CPPHTTPLIB_WEBSOCKET_MAX_PAYLOAD_LENGTH` (httplib.h:247 @cd959bae9).
const max_payload_length: usize = 16777216;
/// `CPPHTTPLIB_WEBSOCKET_SERVER_READ_TIMEOUT_SECOND` (httplib.h:279 @cd959bae9).
pub const server_read_timeout_second: i64 = 300;
/// `CPPHTTPLIB_WEBSOCKET_CLOSE_TIMEOUT_SECOND` (httplib.h:283 @cd959bae9).
const close_timeout_second: i64 = 5;
/// `CPPHTTPLIB_WEBSOCKET_PING_INTERVAL_SECOND` (httplib.h:287 @cd959bae9).
const ping_interval_second: i64 = 30;

/// Ports `Opcode` (httplib.h:4572 @cd959bae9). Any four bits are read off
/// the wire, as the C's cast takes them.
pub const Opcode = enum(u8) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
    _,
};

/// Ports `CloseStatus` (httplib.h:4581 @cd959bae9), the codes used here.
pub const CloseStatus = enum(u16) {
    normal = 1000,
    going_away = 1001,
    protocol_error = 1002,
    invalid_payload = 1007,
};

/// Ports `ReadResult` (httplib.h:4599 @cd959bae9). `Timeout` is out of reach:
/// nothing here asks for a read timeout.
pub const ReadResult = enum(c_int) { fail = 0, text = 1, binary = 2 };

/// Ports `FrameRead` (httplib.h:4914 @cd959bae9).
const FrameRead = enum { ok, fail, timeout };

/// Ports `fill_random_bytes` (httplib.cpp:915 @cd959bae9): unpredictable
/// bytes for masking keys (RFC 6455 Section 5.3) and nonces.
pub fn fillRandomBytes(out: []u8) void {
    if (comptime builtin.os.tag.isDarwin()) {
        std.c.arc4random_buf(out.ptr, out.len);
    } else {
        var filled: usize = 0;
        while (filled < out.len) {
            const n: isize = @bitCast(std.os.linux.getrandom(out[filled..].ptr, out.len - filled, 0));
            if (n > 0) filled += @intCast(n);
        }
    }
}

/// Ports `is_valid_utf8` (httplib.cpp:709 @cd959bae9).
fn isValidUtf8(s: []const u8) bool {
    var i: usize = 0;
    const n = s.len;
    while (i < n) {
        const c = s[i];
        var len: usize = undefined;
        var cp: u32 = undefined;
        if (c < 0x80) {
            i += 1;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (i + len > n) return false;
        for (1..len) |j| {
            const b = s[i + j];
            if ((b & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (b & 0x3F);
        }
        // Overlong encoding check
        if (len == 2 and cp < 0x80) return false;
        if (len == 3 and cp < 0x800) return false;
        if (len == 4 and cp < 0x10000) return false;
        // Surrogate halves (U+D800..U+DFFF) and beyond U+10FFFF are invalid
        if (cp >= 0xD800 and cp <= 0xDFFF) return false;
        if (cp > 0x10FFFF) return false;
        i += len;
    }
    return true;
}

/// Ports `is_valid_close_payload` (httplib.cpp:1105 @cd959bae9): RFC 6455
/// §5.5.1, §7.4: a Close payload is empty, or a status code that may be sent
/// followed by a UTF-8 reason.
fn isValidClosePayload(payload: []const u8) bool {
    if (payload.len == 0) return true;
    if (payload.len < 2) return false;
    const code = (@as(u32, payload[0]) << 8) | payload[1];
    const may_be_sent = (1000 <= code and code <= 1003) or
        (1007 <= code and code <= 1014) or
        (3000 <= code and code <= 4999);
    return may_be_sent and isValidUtf8(payload[2..]);
}

/// Ports `read_exact` (httplib.cpp:1007 @cd959bae9): exactly `buf.len`
/// bytes. A timeout is reported only when nothing at all was consumed: once a
/// byte has been taken the stream sits mid-field and cannot be resumed.
fn readExact(strm: *SocketStream, buf: []u8) FrameRead {
    var total: usize = 0;
    while (total < buf.len) {
        const n = strm.read(buf[total..]);
        if (n <= 0) {
            const timed_out = total == 0 and strm.err == .timeout;
            return if (timed_out) .timeout else .fail;
        }
        total += @intCast(n);
    }
    return .ok;
}

/// Ports `read_websocket_frame` (httplib.cpp:1021 @cd959bae9): one frame into
/// `payload`, cleared first and left unmasked.
fn readWebsocketFrame(a: std.mem.Allocator, strm: *SocketStream, opcode: *Opcode, payload: *std.ArrayList(u8), fin: *bool, expect_masked: bool, max_len: usize) error{OutOfMemory}!FrameRead {
    // Read first 2 bytes. This is the only read that may report a timeout: it
    // sits on a frame boundary, where nothing has been consumed yet.
    var header: [2]u8 = undefined;
    const first = readExact(strm, &header);
    if (first != .ok) return first;

    fin.* = (header[0] & 0x80) != 0;

    // RSV1, RSV2, RSV3 must be 0 when no extension is negotiated
    if ((header[0] & 0x70) != 0) return .fail;

    opcode.* = @fromBackingInt(@intCast(header[0] & 0x0F));
    const masked = (header[1] & 0x80) != 0;
    var payload_len: u64 = header[1] & 0x7F;

    // RFC 6455 Section 5.5: control frames MUST NOT be fragmented and
    // MUST have a payload length of 125 bytes or less
    const is_control = (header[0] & 0x08) != 0;
    if (is_control) {
        if (!fin.*) return .fail;
        if (payload_len > 125) return .fail;
    }

    if (masked != expect_masked) return .fail;

    // Extended payload length
    if (payload_len == 126) {
        var ext: [2]u8 = undefined;
        if (readExact(strm, &ext) != .ok) return .fail;
        payload_len = (@as(u64, ext[0]) << 8) | ext[1];
    } else if (payload_len == 127) {
        var ext: [8]u8 = undefined;
        if (readExact(strm, &ext) != .ok) return .fail;
        // RFC 6455 Section 5.2: the most significant bit MUST be 0
        if ((ext[0] & 0x80) != 0) return .fail;
        payload_len = 0;
        for (ext) |b| payload_len = (payload_len << 8) | b;
    }

    // A max_len of 0 means no limit, as payload_max_length does; every length
    // that survives the check above fits a 64-bit size_t.
    if (max_len > 0 and payload_len > max_len) return .fail;

    // Read mask key if present
    var mask_key: [4]u8 = @splat(0);
    if (masked) {
        if (readExact(strm, &mask_key) != .ok) return .fail;
    }

    // Grow the payload only as it arrives, since a header alone can announce
    // max_len bytes that never come.
    payload.clearRetainingCapacity();
    const len: usize = @intCast(payload_len);
    while (payload.items.len < len) {
        const offset = payload.items.len;
        const n = @min(len - offset, detail.recv_bufsiz);
        try payload.resize(a, offset + n);
        if (readExact(strm, payload.items[offset..][0..n]) != .ok) return .fail;
    }

    // Unmask if needed
    if (masked) {
        for (payload.items, 0..) |*b, i| b.* ^= mask_key[i % 4];
    }

    return .ok;
}

/// Ports `write_websocket_frame` (httplib.cpp:920 @8642541b2): the header,
/// and the payload, masked in chunks when `mask` is set.
fn writeWebsocketFrame(strm: *SocketStream, opcode: Opcode, data: []const u8, fin: bool, mask: bool) bool {
    // First byte: FIN + opcode
    var header: [2]u8 = undefined;
    header[0] = (if (fin) @as(u8, 0x80) else 0) | (@backingInt(opcode) & 0x0F);
    const len = data.len;

    // Second byte: MASK + payload length
    if (len < 126) {
        header[1] = @intCast(len);
        if (mask) header[1] |= 0x80;
        if (strm.write(&header) < 0) return false;
    } else if (len <= 0xFFFF) {
        header[1] = 126;
        if (mask) header[1] |= 0x80;
        if (strm.write(&header) < 0) return false;
        const ext: [2]u8 = .{ @intCast((len >> 8) & 0xFF), @intCast(len & 0xFF) };
        if (strm.write(&ext) < 0) return false;
    } else {
        header[1] = 127;
        if (mask) header[1] |= 0x80;
        if (strm.write(&header) < 0) return false;
        var ext: [8]u8 = undefined;
        std.mem.writeInt(u64, &ext, len, .big);
        if (strm.write(&ext) < 0) return false;
    }

    if (mask) {
        // Generate random mask key
        var mask_key: [4]u8 = undefined;
        fillRandomBytes(&mask_key);
        if (strm.write(&mask_key) < 0) return false;

        // Write masked payload in chunks
        const chunk_size = 4096;
        var buf: [chunk_size]u8 = undefined;
        var offset: usize = 0;
        while (offset < len) : (offset += chunk_size) {
            const n = @min(chunk_size, len - offset);
            for (0..n) |i| buf[i] = data[offset + i] ^ mask_key[(offset + i) % 4];
            if (strm.write(buf[0..n]) < 0) return false;
        }
    } else if (len > 0) {
        if (strm.write(data) < 0) return false;
    }

    return true;
}

/// Ports `websocket_accept_key` (httplib.cpp:878 @cd959bae9).
///
/// Return: the Base64 of the SHA-1, in `out`.
pub fn websocketAcceptKey(client_key: []const u8, out: *[28]u8) []const u8 {
    var h = std.crypto.hash.Sha1.init(.{});
    h.update(client_key);
    h.update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    var digest: [20]u8 = undefined;
    h.final(&digest);
    return std.base64.standard.Encoder.encode(out, &digest);
}

/// `pthread_mutex_trylock`, which `std.c` does not name: 0 when taken.
extern "c" fn pthread_mutex_trylock(mutex: *std.c.pthread_mutex_t) c_int;

/// Ports `WebSocket` (httplib.h:4646 @cd959bae9).
pub const WebSocket = struct {
    a: std.mem.Allocator,
    strm: *SocketStream,
    is_server: bool,
    ping_interval_sec: i64 = ping_interval_second,
    closed: std.atomic.Value(bool) = .init(false),
    write_mutex: std.c.pthread_mutex_t = .{},
    /// Owned by whichever thread is parsing frames off `strm`. Only one
    /// thread may do so: a second parser stealing bytes silently corrupts
    /// the message the first one is assembling.
    read_mutex: std.c.pthread_mutex_t = .{},
    ping_thread: ?std.Thread = null,
    ping_mutex: std.c.pthread_mutex_t = .{},
    ping_cv: std.c.pthread_cond_t = .{},

    /// Ports `WebSocket` (httplib.h:4676 @cd959bae9), the constructor,
    /// into `self`, which its ping thread points at.
    pub fn init(self: *WebSocket, a: std.mem.Allocator, strm: *SocketStream, is_server: bool) void {
        self.* = .{ .a = a, .strm = strm, .is_server = is_server };
        self.startHeartbeat();
    }

    /// Ports the destructor of `WebSocket` (httplib.cpp:18843 @cd959bae9).
    pub fn deinit(self: *WebSocket) void {
        {
            _ = std.c.pthread_mutex_lock(&self.ping_mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.ping_mutex);
            self.closed.store(true, .seq_cst);
        }
        _ = std.c.pthread_cond_broadcast(&self.ping_cv);
        if (self.ping_thread) |t| t.join();
        self.ping_thread = null;
        // A server's stream outlives this object
        self.strm.setReadInterrupt(null);
    }

    /// Ports `send_frame` (httplib.cpp:18663 @cd959bae9), a member of
    /// `WebSocket`.
    fn sendFrame(self: *WebSocket, op: Opcode, data: []const u8) bool {
        _ = std.c.pthread_mutex_lock(&self.write_mutex);
        defer _ = std.c.pthread_mutex_unlock(&self.write_mutex);
        if (self.closed.load(.seq_cst)) return false;
        return writeWebsocketFrame(self.strm, op, data, true, !self.is_server);
    }

    /// The lambda `handle_control` in `WebSocket::read`: answers a control
    /// frame; any other opcode here is a protocol error.
    ///
    /// Return: false when the connection is done.
    fn handleControl(self: *WebSocket, op: Opcode, payload: []const u8) bool {
        switch (op) {
            .ping => {
                _ = std.c.pthread_mutex_lock(&self.write_mutex);
                defer _ = std.c.pthread_mutex_unlock(&self.write_mutex);
                _ = writeWebsocketFrame(self.strm, .pong, payload, true, !self.is_server);
                return true;
            },
            .pong => return true,
            .close => {
                // RFC 6455 §5.5.1: a Close is echoed back, unless its payload is
                // malformed, which is answered with a protocol error instead.
                if (self.closed.swap(true, .seq_cst)) return false;
                var reply = payload;
                const code: u16 = @backingInt(CloseStatus.protocol_error);
                const protocol_error: [2]u8 = .{ @intCast(code >> 8), @intCast(code & 0xff) };
                if (!isValidClosePayload(payload)) reply = &protocol_error;
                _ = std.c.pthread_mutex_lock(&self.write_mutex);
                defer _ = std.c.pthread_mutex_unlock(&self.write_mutex);
                _ = writeWebsocketFrame(self.strm, .close, reply, true, !self.is_server);
                return false;
            },
            else => {
                self.closed.store(true, .seq_cst);
                return false;
            },
        }
    }

    /// Ports `read` (httplib.cpp:18670 @cd959bae9), a member of `WebSocket`:
    /// the next whole message into `msg`, answering the control frames in
    /// between.
    pub fn read(self: *WebSocket, a: std.mem.Allocator, msg: *std.ArrayList(u8)) error{OutOfMemory}!ReadResult {
        var payload: std.ArrayList(u8) = .empty;
        defer payload.deinit(a);

        _ = std.c.pthread_mutex_lock(&self.read_mutex);
        var read_locked = true;
        defer if (read_locked) {
            _ = std.c.pthread_mutex_unlock(&self.read_mutex);
        };

        while (!self.closed.load(.seq_cst)) {
            var opcode: Opcode = undefined;
            var fin: bool = undefined;

            const r = try readWebsocketFrame(a, self.strm, &opcode, &payload, &fin, self.is_server, max_payload_length);
            // No read timeout was asked for, so the compile-time default is a
            // backstop, and elapsing it closes the connection.
            if (r != .ok) {
                self.closed.store(true, .seq_cst);
                return .fail;
            }

            switch (opcode) {
                .text, .binary => {
                    const result: ReadResult = if (opcode == .text) .text else .binary;
                    msg.clearRetainingCapacity();
                    try msg.appendSlice(a, payload.items);

                    // Handle fragmentation
                    if (!fin) {
                        var cont_payload: std.ArrayList(u8) = .empty;
                        defer cont_payload.deinit(a);
                        while (true) {
                            var cont_opcode: Opcode = undefined;
                            var cont_fin: bool = undefined;
                            // Only what is left of the limit, but room for a control frame,
                            // which may come between the fragments.
                            const max_len = @max(@as(usize, 125), max_payload_length -% msg.items.len);
                            // A timeout is not reportable here: half of a fragmented message is
                            // already in `msg` and read() has no way to resume it.
                            if (try readWebsocketFrame(a, self.strm, &cont_opcode, &cont_payload, &cont_fin, self.is_server, max_len) != .ok) {
                                self.closed.store(true, .seq_cst);
                                return .fail;
                            }
                            // RFC 6455 5.4: the fragments are continuation frames, with only
                            // control frames between them.
                            if (cont_opcode != .continuation) {
                                if (self.handleControl(cont_opcode, cont_payload.items)) continue;
                                return .fail;
                            }
                            try msg.appendSlice(a, cont_payload.items);
                            if (msg.items.len > max_payload_length) {
                                self.closed.store(true, .seq_cst);
                                return .fail;
                            }
                            if (cont_fin) break;
                        }
                    }
                    // RFC 6455 Section 5.6: text frames must contain valid UTF-8
                    if (result == .text and !isValidUtf8(msg.items)) {
                        // close() takes the read lock to wait for the peer's Close reply, so
                        // it must not run while this thread still holds it.
                        read_locked = false;
                        _ = std.c.pthread_mutex_unlock(&self.read_mutex);
                        self.close(.invalid_payload, "invalid UTF-8");
                        return .fail;
                    }
                    return result;
                },
                else => {
                    if (self.handleControl(opcode, payload.items)) continue;
                    return .fail;
                },
            }
        }
        return .fail;
    }

    /// Ports `send` (httplib.cpp:18790 @cd959bae9), a member of `WebSocket`:
    /// a text frame.
    pub fn send(self: *WebSocket, data: []const u8) bool {
        return self.sendFrame(.text, data);
    }

    /// Ports `send` (httplib.cpp:18794 @cd959bae9), the overload taking a
    /// pointer and a length, a member of `WebSocket`: a binary frame.
    pub fn sendBinary(self: *WebSocket, data: []const u8) bool {
        return self.sendFrame(.binary, data);
    }

    /// Ports `close` (httplib.cpp:18798 @cd959bae9), a member of `WebSocket`.
    pub fn close(self: *WebSocket, status: CloseStatus, reason: []const u8) void {
        if (self.closed.swap(true, .seq_cst)) return;
        _ = std.c.pthread_cond_broadcast(&self.ping_cv);
        var payload_buf: [125]u8 = undefined;
        const code: u16 = @backingInt(status);
        payload_buf[0] = @intCast((code >> 8) & 0xFF);
        payload_buf[1] = @intCast(code & 0xFF);
        // RFC 6455 Section 5.5: control frame payload must not exceed 125 bytes
        // Close frame has 2-byte status code, so reason is limited to 123 bytes,
        // cut back to a whole UTF-8 character
        var short_reason = reason[0..@min(reason.len, 123)];
        while (short_reason.len != 0 and !isValidUtf8(short_reason)) short_reason = short_reason[0 .. short_reason.len - 1];
        @memcpy(payload_buf[2..][0..short_reason.len], short_reason);
        var payload: []const u8 = payload_buf[0 .. 2 + short_reason.len];
        // A code that may not be sent, such as NoStatus, goes without a payload
        if (!isValidClosePayload(payload)) payload = "";
        {
            _ = std.c.pthread_mutex_lock(&self.write_mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.write_mutex);
            _ = writeWebsocketFrame(self.strm, .close, payload, true, !self.is_server);
        }

        // RFC 6455 Section 7.1.1: after sending a Close frame, wait for the peer's
        // Close response before closing the TCP connection.
        //
        // Wait only when no other thread is parsing frames. When one is, it is the
        // thread positioned to see the peer's reply, and reading here would take
        // bytes out of the message it is assembling. Bailing out also leaves the
        // stream, including its read timeout, entirely to that thread.
        if (pthread_mutex_trylock(&self.read_mutex) != 0) return;
        defer _ = std.c.pthread_mutex_unlock(&self.read_mutex);

        // Use a short timeout to avoid hanging if the peer doesn't respond.
        self.strm.setReadTimeout(close_timeout_second, 0);

        var op: Opcode = undefined;
        var resp: std.ArrayList(u8) = .empty;
        defer resp.deinit(self.a);
        var fin: bool = undefined;
        while ((readWebsocketFrame(self.a, self.strm, &op, &resp, &fin, self.is_server, 125) catch .fail) == .ok) {
            if (op == .close) break;
        }
    }

    /// Ports `is_open` (httplib.cpp:18888 @cd959bae9), a member of
    /// `WebSocket`.
    pub fn isOpen(self: *const WebSocket) bool {
        return !self.closed.load(.seq_cst);
    }

    /// Ports `start_heartbeat` (httplib.cpp:18854 @cd959bae9), a member of
    /// `WebSocket`: a ping every interval until the connection closes.
    fn startHeartbeat(self: *WebSocket) void {
        if (self.ping_interval_sec <= 0) return;
        self.ping_thread = std.Thread.spawn(.{}, pingLoop, .{self}) catch null;
    }

    fn pingLoop(self: *WebSocket) void {
        _ = std.c.pthread_mutex_lock(&self.ping_mutex);
        defer _ = std.c.pthread_mutex_unlock(&self.ping_mutex);
        while (true) {
            var deadline: std.c.timespec = undefined;
            _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &deadline);
            deadline.sec += @intCast(self.ping_interval_sec);
            // The predicate keeps a spurious wakeup from sending a ping early
            while (!self.closed.load(.seq_cst)) {
                if (std.c.pthread_cond_timedwait(&self.ping_cv, &self.ping_mutex, &deadline) == .TIMEDOUT) break;
            }
            if (self.closed.load(.seq_cst)) return;
            _ = std.c.pthread_mutex_unlock(&self.ping_mutex);
            const ok = self.sendFrame(.ping, "");
            _ = std.c.pthread_mutex_lock(&self.ping_mutex);
            if (!ok) {
                self.closed.store(true, .seq_cst);
                return;
            }
        }
    }
};

/// Ports `WebSocketClient` (httplib.h:4671 @8642541b2) over a `ws://` URL.
pub const WebSocketClient = struct {
    a: std.mem.Allocator,
    host: []const u8 = "",
    port: c_int = 0,
    path: []const u8 = "",
    headers: Headers,
    is_valid: bool = false,
    sock: socket.Socket = socket.invalid_socket,
    strm: ?*SocketStream = null,
    ws: ?*WebSocket = null,

    const client = @import("client.zig");

    /// Ports `WebSocketClient` (httplib.cpp:18685 @8642541b2), the
    /// constructor: a scheme other than `ws` throws in the C, which no
    /// caller here reaches; this leaves the client invalid.
    pub fn init(a: std.mem.Allocator, scheme_host_port_path: []const u8, headers: Headers) WebSocketClient {
        var self: WebSocketClient = .{ .a = a, .headers = headers };
        var uc: client.UrlComponents = undefined;
        if (client.parseUrl(scheme_host_port_path, &uc) and uc.scheme.len != 0 and uc.host.len != 0 and uc.path.len != 0) {
            if (!std.mem.eql(u8, uc.scheme, "ws")) return self;
            self.host = uc.host;
            self.port = 80;
            if (uc.port.len != 0 and !client.parsePort(uc.port, &self.port)) return self;
            self.path = uc.path;
            if (uc.query.len != 0) {
                self.path = std.mem.concat(a, u8, &.{ uc.path, uc.query }) catch return self;
            }
            self.is_valid = true;
        }
        return self;
    }

    /// Ports the destructor of `WebSocketClient` (httplib.cpp:18968
    /// @cd959bae9).
    pub fn deinit(self: *WebSocketClient) void {
        self.shutdownAndClose();
    }

    /// Ports `shutdown_and_close` (httplib.cpp:18980 @cd959bae9), a member
    /// of `WebSocketClient`.
    fn shutdownAndClose(self: *WebSocketClient) void {
        if (self.ws) |ws| {
            if (ws.isOpen()) ws.close(.normal, "");
            ws.deinit();
            self.a.destroy(ws);
            self.ws = null;
        }
        if (self.strm) |s| {
            self.a.destroy(s);
            self.strm = null;
        }
        if (self.sock != socket.invalid_socket) {
            _ = socket.shutdownSocket(self.sock);
            _ = socket.closeSocket(self.sock);
            self.sock = socket.invalid_socket;
        }
    }

    /// Ports `connect` (httplib.cpp:19083 @cd959bae9), a member of
    /// `WebSocketClient`: a socket, the upgrade request, and its answer
    /// checked. `create_stream`'s stream waits without bound, the read
    /// timeout being unset.
    ///
    /// Return: whether the handshake succeeded, the C's `Result` read as a
    /// bool.
    pub fn connect(self: *WebSocketClient) error{OutOfMemory}!bool {
        if (!self.is_valid) return false;
        self.shutdownAndClose();

        var arena = std.heap.ArenaAllocator.init(self.a);
        defer arena.deinit();
        var impl = client.ClientImpl.init(arena.allocator(), self.host, self.port);
        impl.read_timeout_sec = 0;
        var err: client.Error = .success;
        self.sock = try impl.createClientSocket(&err);
        if (self.sock == socket.invalid_socket) return false;

        const strm = try self.a.create(SocketStream);
        errdefer self.a.destroy(strm);
        strm.* = .{
            .sock = self.sock,
            .read_timeout_sec = -1,
            .read_timeout_usec = 0,
            .write_timeout_sec = impl.write_timeout_sec,
            .write_timeout_usec = impl.write_timeout_usec,
        };
        self.strm = strm;

        // The handshakes belong to establishing the connection, so an unset read
        // timeout leaves them bounded by the connection timeout instead of forever
        // (`handshake_timeout`, httplib.cpp:18790 @8642541b2): the C's stream
        // waits unbounded for these reads all the same.
        if (!try self.performHandshake(strm)) {
            self.shutdownAndClose();
            return false;
        }

        const ws = try self.a.create(WebSocket);
        ws.init(self.a, strm, false);
        self.ws = ws;
        return true;
    }

    /// Ports `perform_websocket_handshake` (httplib.cpp:5772 @cd959bae9) and
    /// `read_websocket_upgrade_response` (httplib.cpp:3705 @cd959bae9), with
    /// `prepare_default_headers` (httplib.cpp:18850 @8642541b2) before them.
    fn performHandshake(self: *WebSocketClient, strm: *SocketStream) error{OutOfMemory}!bool {
        const a = self.a;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const ar = arena.allocator();

        var headers = try self.headers.clone(ar);
        if (headers.find("Host") == null) {
            _ = try headers.emplace(ar, "Host", try client.makeHostAndPortString(ar, self.host, self.port, false));
        }
        if (headers.find("User-Agent") == null) {
            _ = try headers.emplace(ar, "User-Agent", "cpp-httplib/" ++ client.cpphttplib_version);
        }

        // Generate a random 16-byte Sec-WebSocket-Key (RFC 6455 Section 4.1)
        var key_bytes: [16]u8 = undefined;
        fillRandomBytes(&key_bytes);
        var key_buf: [24]u8 = undefined;
        const client_key = std.base64.standard.Encoder.encode(&key_buf, &key_bytes);

        for ([_][]const u8{ "Upgrade", "Connection", "Sec-WebSocket-Key", "Sec-WebSocket-Version" }) |n| _ = headers.eraseKey(n);
        _ = try headers.emplace(ar, "Upgrade", "websocket");
        _ = try headers.emplace(ar, "Connection", "Upgrade");
        _ = try headers.emplace(ar, "Sec-WebSocket-Key", client_key);
        _ = try headers.emplace(ar, "Sec-WebSocket-Version", "13");

        // `write_request_head` (httplib.cpp:5755 @cd959bae9): built in memory
        // first, so a rejected header never leaves a partial request on the wire
        var bstrm: socket.BufferStream = .{ .a = ar };
        if (!detail.isRequestTarget(self.path)) return false;
        _ = bstrm.write(try std.mem.concat(ar, u8, &.{ "GET ", self.path, " HTTP/1.1\r\n" }));
        for (headers.entries.items) |h| {
            if (!detail.isFieldValid(h.key, h.value)) return false;
        }
        if (try message.writeHeaders(ar, &bstrm, &headers) <= 0) return false;
        if (!message.writeData(strm, bstrm.buffer.items)) return false;

        // Verify 101 response and Sec-WebSocket-Accept header
        var accept_buf: [28]u8 = undefined;
        const expected_accept = websocketAcceptKey(client_key, &accept_buf);

        // `read_response_head` (httplib.cpp:3690 @cd959bae9)
        var buf: [2048]u8 = undefined;
        var lr = socket.LineReader.init(ar, strm, &buf);
        if (!try lr.getline()) return false;
        var version: []const u8 = "";
        var reason: []const u8 = "";
        var status: c_int = -1;
        var res_headers: Headers = .{};
        if (try client.parseStatusLine(ar, lr.cstr(), &version, &status, &reason)) {
            if (!try message.readHeaders(ar, strm, &res_headers)) return false;
        }

        if (status != 101) return false;
        if (!detail.hasHeaderToken(&res_headers, "Upgrade", "websocket")) return false;
        if (!detail.hasHeaderToken(&res_headers, "Connection", "upgrade")) return false;
        const accept = res_headers.get("Sec-WebSocket-Accept", 0) orelse return false;
        if (!std.mem.eql(u8, accept, expected_accept)) return false;
        // no subprotocol is offered, so none may come back
        if (res_headers.get("Sec-WebSocket-Protocol", 0)) |proto| {
            if (proto.len != 0) return false;
        }
        return true;
    }

    /// Ports `read` (httplib.cpp:19135 @cd959bae9), a member of
    /// `WebSocketClient`.
    pub fn read(self: *WebSocketClient, a: std.mem.Allocator, msg: *std.ArrayList(u8)) error{OutOfMemory}!ReadResult {
        const ws = self.ws orelse return .fail;
        return ws.read(a, msg);
    }

    /// Ports `send` (httplib.cpp:19140 @cd959bae9), a member of
    /// `WebSocketClient`.
    pub fn send(self: *WebSocketClient, data: []const u8) bool {
        const ws = self.ws orelse return false;
        return ws.send(data);
    }

    /// Ports `send` (httplib.cpp:19145 @cd959bae9), the overload taking a
    /// pointer and a length, a member of `WebSocketClient`.
    pub fn sendBinary(self: *WebSocketClient, data: []const u8) bool {
        const ws = self.ws orelse return false;
        return ws.sendBinary(data);
    }

    /// Ports `close` (httplib.cpp:19150 @cd959bae9), a member of
    /// `WebSocketClient`.
    pub fn close(self: *WebSocketClient) void {
        if (self.ws) |ws| ws.close(.normal, "");
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the accept key is RFC 6455's example" {
    var buf: [28]u8 = undefined;
    try std.testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", websocketAcceptKey("dGhlIHNhbXBsZSBub25jZQ==", &buf));
}

test "a Close payload is empty, or a code that may be sent and a UTF-8 reason" {
    try std.testing.expect(isValidClosePayload(""));
    try std.testing.expect(!isValidClosePayload("\x03"));
    try std.testing.expect(isValidClosePayload("\x03\xe8bye"));
    try std.testing.expect(!isValidClosePayload("\x03\xedbye")); // 1005
    try std.testing.expect(!isValidClosePayload("\x03\xe8\xff"));
}

test "UTF-8 validation refuses overlong forms and surrogates" {
    try std.testing.expect(isValidUtf8("a\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"));
    try std.testing.expect(!isValidUtf8("\xc0\x80"));
    try std.testing.expect(!isValidUtf8("\xed\xa0\x80"));
    try std.testing.expect(!isValidUtf8("\xf4\x90\x80\x80"));
    try std.testing.expect(!isValidUtf8("\xe2\x82"));
}
