//! `--connect`: spawns `llama-connect`, which exposes this server to a remote
//! browser over WebRTC. The binary is a separate project
//! (https://github.com/ggml-org/llama-connect), found next to `llama-server`
//! or in `PATH`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-connect.cpp` and
//! `llama.cpp/tools/server/server-connect.h` (v1.0.0, `b1b770d90`), their
//! POSIX arm: `llama-server` is not a Windows target. Each declaration names
//! what it ports and the line it began at.

const std = @import("std");
const common_chat = @import("common_chat");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const tools_common = @import("tools_common");

/// Ports `CONNECT_ROOM_CODE_LEN` and `CONNECT_PASS_CODE_LEN`
/// (server-connect.cpp:18, 19 @b1b770d90): share code = room code + pass
/// code, must match llama-connect and the Web UI.
const connect_room_code_len = 8;
const connect_pass_code_len = 32;

/// Ports `CONNECT_CODE_CHARS` (server-connect.cpp:21 @b1b770d90).
const connect_code_chars = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789";

/// Ports `CONNECT_EXE_NAME` (server-connect.cpp:27 @b1b770d90).
const connect_exe_name = "llama-connect";

/// Ports `CONNECT_STOP_TIMEOUT_MS` (server-connect.cpp:32 @b1b770d90): how
/// long to wait for the child to notice the closed stdin before killing it.
const connect_stop_timeout_ms = 3000;

/// Ports `gen_share_code` (server-connect.cpp:35 @b1b770d90): the pass code
/// guards the tunnel, so it is drawn from the system's random device,
/// uniformly, by rejection.
fn genShareCode(a: std.mem.Allocator) std.mem.Allocator.Error![]u8 {
    const code = try a.alloc(u8, connect_room_code_len + connect_pass_code_len);
    const n = connect_code_chars.len;
    const limit = 256 - (256 % n);
    for (code) |*ch| {
        var b: [1]u8 = undefined;
        while (true) {
            mod.httplib.ws.fillRandomBytes(&b);
            if (b[0] < limit) break;
        }
        ch.* = connect_code_chars[b[0] % n];
    }
    return code;
}

/// Ports `format_share_code` (server-connect.cpp:47 @b1b770d90): blocks of 8.
fn formatShareCode(a: std.mem.Allocator, code: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < code.len) : (i += 8) {
        if (i > 0) {
            try out.append(a, ' ');
        }
        try out.appendSlice(a, code[i..@min(i + 8, code.len)]);
    }
    return out.items;
}

/// Ports `normalize_share_code` (server-connect.cpp:60 @b1b770d90):
/// whitespace is tolerated, the Web UI shows the code in blocks and users
/// paste it back.
///
/// Return: an empty string if it is not a valid share code.
fn normalizeShareCode(a: std.mem.Allocator, key: []const u8) std.mem.Allocator.Error![]const u8 {
    var code: std.ArrayList(u8) = .empty;
    for (key) |ch| {
        if (!std.ascii.isWhitespace(ch)) {
            try code.append(a, ch);
        }
    }

    if (code.items.len != connect_room_code_len + connect_pass_code_len) {
        return "";
    }

    for (code.items) |ch| {
        if (std.mem.indexOfScalar(u8, connect_code_chars, ch) == null) {
            return "";
        }
    }

    return code.items;
}

/// Ports `forward_child_log` (server-connect.cpp:80 @b1b770d90): llama-connect
/// logs as `[LEVEL] text`, forwarded at the same level so our verbosity
/// filter applies. Untagged lines are the startup banner, shown as info.
fn forwardChildLog(line: [*:0]const u8) void {
    const Level = enum { err, wrn, inf, dbg };
    const tags = [_]struct { []const u8, Level }{
        .{ "[ERROR] ", .err },
        .{ "[WARN] ", .wrn },
        .{ "[INFO] ", .inf },
        .{ "[DEBUG] ", .dbg },
        .{ "[TRACE] ", .dbg },
    };

    var level: Level = .inf;
    var text = line;
    const s = std.mem.span(line);

    for (tags) |t| {
        if (std.mem.startsWith(u8, s, t[0])) {
            level = t[1];
            text = line + t[0].len;
            break;
        }
    }

    switch (level) {
        .err => tools_common.log.err("connect | %s", .{text}),
        .wrn => tools_common.log.wrn("connect | %s", .{text}),
        .dbg => tools_common.log.dbg("connect | %s", .{text}),
        .inf => tools_common.log.inf("connect | %s", .{text}),
    }
}

/// Ports `upstream_host` (server-connect.cpp:111 @b1b770d90): the child dials
/// over TCP: a wildcard bind is reached on loopback, a specific address on
/// itself.
///
/// Return: empty when every listener is a unix socket.
fn upstreamHost(a: std.mem.Allocator, hosts: []const []const u8) std.mem.Allocator.Error![]const u8 {
    for (hosts) |host| {
        if (std.mem.endsWith(u8, host, ".sock")) {
            continue;
        }
        if (std.mem.eql(u8, host, "0.0.0.0")) {
            return "127.0.0.1";
        }
        if (std.mem.eql(u8, host, "::")) {
            return "[::1]";
        }
        // llama-connect builds http://host:port, so an IPv6 literal needs brackets
        if (std.mem.indexOfScalar(u8, host, ':') != null and host[0] != '[') {
            return std.mem.concat(a, u8, &.{ "[", host, "]" });
        }
        return host;
    }
    return "";
}

/// Ports `server_connect` (server-connect.h:14 @b1b770d90): the
/// `llama-connect` child, its log thread, and whether its exit is expected.
pub const ServerConnect = struct {
    proc: ?*stdlib.subproc.Subproc = null,
    log_thread: ?std.Thread = null,
    stopping: std.atomic.Value(bool) = .init(false), // tells the log thread the exit is expected

    /// Ports `find_binary` (server-connect.cpp:136 @b1b770d90): the path of
    /// the llama-connect binary, preferring the copy shipped next to
    /// llama-server over an unrelated one in `PATH`.
    ///
    /// Return: empty if not found.
    pub fn findBinary(a: std.mem.Allocator) std.mem.Allocator.Error![]const u8 {
        if (mod.models.getServerExecPath(a)) |exe| {
            const sibling = try stdlib.path.join(a, stdlib.path.parent(exe), connect_exe_name);
            if (stdlib.fs.isRegularFile(sibling)) {
                return sibling;
            }
        } else |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            mod.common.srv(.wrn, "find_binary", "could not resolve the llama-server path (%s), looking for llama-connect in PATH only\n", .{try mod.common.zs(a, common_chat.exception.what(e))});
        }

        const path_env: []const u8 = stdlib.env.get("PATH") orelse "";
        var pos: usize = 0;
        while (pos <= path_env.len) {
            const end = std.mem.indexOfScalarPos(u8, path_env, pos, ':') orelse path_env.len;
            const dir = path_env[pos..end];
            if (dir.len != 0) {
                const candidate = try stdlib.path.join(a, dir, connect_exe_name);
                if (stdlib.fs.isRegularFile(candidate)) {
                    return candidate;
                }
            }
            pos = end + 1;
        }

        return "";
    }

    /// Ports `unavailable_reason` (server-connect.cpp:167 @b1b770d90): why
    /// `--connect` cannot work here.
    ///
    /// Return: empty if it can.
    pub fn unavailableReason(a: std.mem.Allocator, params: *const tools_common.params.Params) std.mem.Allocator.Error![]const u8 {
        if (!stdlib.subproc.Subproc.isSupported()) {
            return "this build has subprocess support disabled, rebuild with -DLLAMA_SUBPROCESS=ON";
        }

        if (params.server_connect_code.len != 0 and (try normalizeShareCode(a, params.server_connect_code)).len == 0) {
            return std.fmt.allocPrint(a, "--connect-code must be {d} characters from '{s}'", .{ connect_room_code_len + connect_pass_code_len, connect_code_chars });
        }

        if ((try upstreamHost(a, params.hostnames.items)).len == 0) {
            return "llama-connect reaches the server over TCP, add a TCP address to --host";
        }

        if ((try findBinary(a)).len == 0) {
            return "could not find '" ++ connect_exe_name ++ "' next to llama-server or in PATH.\n" ++
                "    it is a separate binary: download it from https://github.com/ggml-org/llama-connect/releases\n" ++
                "    or build llama.cpp with -DLLAMA_CONNECT=ON to have it fetched automatically";
        }

        return "";
    }

    /// Ports `resolve_code` (server-connect.cpp:190 @b1b770d90).
    pub fn resolveCode(a: std.mem.Allocator, code: []const u8) std.mem.Allocator.Error![]const u8 {
        return if (code.len == 0) try genShareCode(a) else try normalizeShareCode(a, code);
    }

    /// `fgets(buf, sizeof(buf), stream)`: up to `buf.len - 1` bytes, through
    /// the first newline, NUL-terminated.
    ///
    /// Return: the line in `buf`, or null at end of file or on an error with
    /// nothing read.
    fn fgets(r: *std.Io.Reader, buf: []u8) ?[:0]u8 {
        var n: usize = 0;
        while (n + 1 < buf.len) {
            const b = r.takeByte() catch break;
            buf[n] = b;
            n += 1;
            if (b == '\n') break;
        }
        if (n == 0) return null;
        buf[n] = 0;
        return buf[0..n :0];
    }

    /// The `log_thread` lambda of `start`: the child's output forwarded until
    /// EOF, which means the child is gone.
    fn logLoop(self: *ServerConnect) void {
        const out = self.proc.?.stdoutFile() orelse {
            mod.common.srv(.err, "operator()", "%s", .{@as([*:0]const u8, "failed to get stdout of the llama-connect process\n")});
            return;
        };
        var read_buf: [4096]u8 = undefined;
        var reader = out.readerStreaming(stdlib.file.defaultIo(), &read_buf);
        var buf: [4096]u8 = undefined;
        while (fgets(&reader.interface, &buf)) |line| {
            forwardChildLog(line.ptr);
        }
        // EOF means the child is gone
        if (!self.stopping.load(.acquire)) {
            mod.common.srv(.err, "operator()", "%s", .{@as([*:0]const u8, "llama-connect exited on its own, remote access is no longer available\n")});
        }
    }

    /// Ports `start` (server-connect.cpp:194 @b1b770d90), a member of
    /// `server_connect`.
    ///
    /// Parameters:
    /// - `port`: the bound one, which differs from `params.port` with
    ///   `--port 0` or the GCP override.
    pub fn start(self: *ServerConnect, a: std.mem.Allocator, params: *const tools_common.params.Params, port: c_int) !bool {
        const func = "start";
        const bin = try findBinary(a);
        if (bin.len == 0) {
            mod.common.srv(.err, func, "%s", .{@as([*:0]const u8, "llama-connect binary not found\n")});
            return false;
        }

        // main() resolves this before anything can read it from /props
        const code = params.server_connect_code;
        tools_common.ggml.ggmlAssert(code.len != 0, "tools/server/server-connect.cpp", 203, "!code.empty()");

        const args = [_][]const u8{
            bin,
            "--host",
            try upstreamHost(a, params.hostnames.items),
            "--port",
            try std.fmt.allocPrint(a, "{d}", .{port}),
            "--code",
            code,
            // the kernel closes our end of this pipe even if we are killed without cleanup
            // this is what stops the child from outliving us
            "--exit-on-stdin-eof",
        };

        const proc = try std.heap.c_allocator.create(stdlib.subproc.Subproc);
        proc.* = .{};

        const options = stdlib.subproc.option_no_window |
            stdlib.subproc.option_combined_stdout_stderr |
            stdlib.subproc.option_inherit_environment;

        if (!proc.create(&args, options, &.{}, null)) {
            mod.common.srv(.err, func, "failed to spawn '%s'\n", .{try mod.common.zs(a, bin)});
            proc.deinit();
            std.heap.c_allocator.destroy(proc);
            return false;
        }
        self.proc = proc;

        self.log_thread = try std.Thread.spawn(.{}, logLoop, .{self});

        mod.common.srv(.inf, func, "%s", .{@as([*:0]const u8, "-----------------\n")});
        mod.common.srv(.inf, func, "%s", .{@as([*:0]const u8, "remote access is enabled via llama-connect\n")});
        mod.common.srv(.inf, func, "share code (enter it in the Web UI under Settings -> Remote Access): %s\n", .{try mod.common.zs(a, try formatShareCode(a, code))});
        mod.common.srv(.wrn, func, "%s", .{@as([*:0]const u8, "anyone with this code can use this server, do not share it publicly\n")});
        mod.common.srv(.inf, func, "%s", .{@as([*:0]const u8, "-----------------\n")});

        return true;
    }

    /// Ports `stop` (server-connect.cpp:253 @b1b770d90), a member of
    /// `server_connect`: idempotent, also called by the destructor.
    pub fn stop(self: *ServerConnect) void {
        const proc = self.proc orelse return;

        mod.common.srv(.inf, "stop", "%s", .{@as([*:0]const u8, "stopping llama-connect...\n")});

        self.stopping.store(true, .release);

        proc.closeStdin();

        var elapsed: c_int = 0;
        while (elapsed < connect_stop_timeout_ms and proc.alive()) : (elapsed += 100) {
            stdlib.clock.sleepMs(100);
        }

        // no-op if the child already exited. the log thread leaves fgets() on EOF, which the kernel
        // raises once every writer of the stdout pipe is gone, so llama-connect must not spawn
        // children of its own: one of them holding that pipe blocks the join below forever
        proc.terminate();

        if (self.log_thread) |t| {
            t.join();
            self.log_thread = null;
        }

        _ = proc.join(); // reap the zombie
        proc.deinit();
        std.heap.c_allocator.destroy(proc);
        self.proc = null;
    }

    /// Ports the destructor of `server_connect` (server-connect.cpp:289
    /// @b1b770d90).
    pub fn deinit(self: *ServerConnect) void {
        self.stop();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a share code is 40 characters of the alphabet, whitespace aside" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const code = try genShareCode(a);
    try std.testing.expectEqual(@as(usize, 40), code.len);
    try std.testing.expectEqualStrings(code, try normalizeShareCode(a, try formatShareCode(a, code)));
    try std.testing.expectEqualStrings("", try normalizeShareCode(a, "short"));
    try std.testing.expectEqualStrings("", try normalizeShareCode(a, "0000000000000000000000000000000000000000"));
}

test "upstream_host brackets IPv6 and maps wildcards to loopback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("127.0.0.1", try upstreamHost(a, &.{"0.0.0.0"}));
    try std.testing.expectEqualStrings("[::1]", try upstreamHost(a, &.{"::"}));
    try std.testing.expectEqualStrings("[fe80::1]", try upstreamHost(a, &.{"fe80::1"}));
    try std.testing.expectEqualStrings("", try upstreamHost(a, &.{"/tmp/x.sock"}));
    try std.testing.expectEqualStrings("h", try upstreamHost(a, &.{ "/tmp/x.sock", "h" }));
}
