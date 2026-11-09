//! `--connect`: spawns `llama-connect`, which exposes this server to a remote
//! browser over WebRTC. The binary is a separate project
//! (https://github.com/ggml-org/llama-connect), found next to `llama-server`
//! or in `PATH`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-connect.cpp` and
//! `llama.cpp/tools/server/server-connect.h` (v0.8.0, `8642541b2`), their
//! POSIX arm: `llama-server` is not a Windows target. Each declaration names
//! what it ports and the line it began at.

const std = @import("std");
const tc = @import("tools_common");
const common = @import("common.zig");
const context = @import("context.zig");
const hf_cache = @import("hf_cache.zig");
const models_mod = @import("models.zig");
const ws_mod = @import("httplib/ws.zig");

const log = tc.log;
const Params = tc.params.Params;
const Allocator = std.mem.Allocator;
const srv = common.srv;
const gpa = context.gpa;

extern "c" fn fgets(buf: [*]u8, n: c_int, stream: *anyopaque) ?[*:0]u8;

/// Ports `CONNECT_ROOM_CODE_LEN` and `CONNECT_PASS_CODE_LEN`
/// (server-connect.cpp:18, 19 @8642541b2): share code = room code + pass
/// code, must match llama-connect and the Web UI.
const connect_room_code_len = 8;
const connect_pass_code_len = 32;

/// Ports `CONNECT_CODE_CHARS` (server-connect.cpp:21 @8642541b2).
const connect_code_chars = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789";

/// Ports `CONNECT_EXE_NAME` (server-connect.cpp:27 @8642541b2).
const connect_exe_name = "llama-connect";

/// Ports `CONNECT_STOP_TIMEOUT_MS` (server-connect.cpp:32 @8642541b2): how
/// long to wait for the child to notice the closed stdin before killing it.
const connect_stop_timeout_ms = 3000;

fn zs(a: Allocator, s: []const u8) Allocator.Error![*:0]const u8 {
    return (try a.dupeSentinel(u8, s, 0)).ptr;
}

/// Ports `gen_share_code` (server-connect.cpp:35 @8642541b2): the pass code
/// guards the tunnel, so it is drawn from the system's random device,
/// uniformly, by rejection.
fn genShareCode(a: Allocator) Allocator.Error![]u8 {
    const code = try a.alloc(u8, connect_room_code_len + connect_pass_code_len);
    const n = connect_code_chars.len;
    const limit = 256 - (256 % n);
    for (code) |*ch| {
        var b: [1]u8 = undefined;
        while (true) {
            ws_mod.fillRandomBytes(&b);
            if (b[0] < limit) break;
        }
        ch.* = connect_code_chars[b[0] % n];
    }
    return code;
}

/// Ports `format_share_code` (server-connect.cpp:47 @8642541b2): blocks of 8.
fn formatShareCode(a: Allocator, code: []const u8) Allocator.Error![]u8 {
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

/// `std::isspace` in the "C" locale.
fn isSpace(ch: u8) bool {
    return ch == ' ' or (ch >= '\t' and ch <= '\r');
}

/// Ports `normalize_share_code` (server-connect.cpp:60 @8642541b2):
/// whitespace is tolerated, the Web UI shows the code in blocks and users
/// paste it back.
///
/// Return: an empty string if it is not a valid share code.
fn normalizeShareCode(a: Allocator, key: []const u8) Allocator.Error![]const u8 {
    var code: std.ArrayList(u8) = .empty;
    for (key) |ch| {
        if (!isSpace(ch)) {
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

/// Ports `forward_child_log` (server-connect.cpp:80 @8642541b2): llama-connect
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
        .err => log.err("connect | %s", .{text}),
        .wrn => log.wrn("connect | %s", .{text}),
        .dbg => log.dbg("connect | %s", .{text}),
        .inf => log.inf("connect | %s", .{text}),
    }
}

/// Ports `upstream_host` (server-connect.cpp:111 @8642541b2): the child dials
/// over TCP: a wildcard bind is reached on loopback, a specific address on
/// itself.
///
/// Return: empty when every listener is a unix socket.
fn upstreamHost(a: Allocator, hosts: []const []const u8) Allocator.Error![]const u8 {
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

/// Ports `server_connect` (server-connect.h:14 @8642541b2): the
/// `llama-connect` child, its log thread, and whether its exit is expected.
pub const ServerConnect = struct {
    proc: ?*tc.subproc.Subproc = null,
    log_thread: ?std.Thread = null,
    stopping: std.atomic.Value(bool) = .init(false), // tells the log thread the exit is expected

    /// Ports `find_binary` (server-connect.cpp:136 @8642541b2): the path of
    /// the llama-connect binary, preferring the copy shipped next to
    /// llama-server over an unrelated one in `PATH`.
    ///
    /// Return: empty if not found.
    pub fn findBinary(a: Allocator) Allocator.Error![]const u8 {
        if (models_mod.getServerExecPath(a)) |exe| {
            const sibling = try hf_cache.pathJoin(a, hf_cache.pathParent(exe), connect_exe_name);
            if (hf_cache.isRegularFile(a, sibling)) {
                return sibling;
            }
        } else |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            srv(.wrn, "find_binary", "could not resolve the llama-server path (%s), looking for llama-connect in PATH only\n", .{try zs(a, common.what(e))});
        }

        const path_env: []const u8 = if (std.c.getenv("PATH")) |v| std.mem.span(v) else "";
        var pos: usize = 0;
        while (pos <= path_env.len) {
            const end = std.mem.indexOfScalarPos(u8, path_env, pos, ':') orelse path_env.len;
            const dir = path_env[pos..end];
            if (dir.len != 0) {
                const candidate = try hf_cache.pathJoin(a, dir, connect_exe_name);
                if (hf_cache.isRegularFile(a, candidate)) {
                    return candidate;
                }
            }
            pos = end + 1;
        }

        return "";
    }

    /// Ports `unavailable_reason` (server-connect.cpp:167 @8642541b2): why
    /// `--connect` cannot work here.
    ///
    /// Return: empty if it can.
    pub fn unavailableReason(a: Allocator, params: *const Params) Allocator.Error![]const u8 {
        if (!tc.subproc.Subproc.isSupported()) {
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

    /// Ports `resolve_code` (server-connect.cpp:190 @8642541b2).
    pub fn resolveCode(a: Allocator, code: []const u8) Allocator.Error![]const u8 {
        return if (code.len == 0) try genShareCode(a) else try normalizeShareCode(a, code);
    }

    /// The `log_thread` lambda of `start`: the child's output forwarded until
    /// EOF, which means the child is gone.
    fn logLoop(self: *ServerConnect) void {
        const out = self.proc.?.stdoutFile() orelse {
            srv(.err, "operator()", "%s", .{@as([*:0]const u8, "failed to get stdout of the llama-connect process\n")});
            return;
        };
        var buf: [4096]u8 = undefined;
        while (fgets(&buf, buf.len, out) != null) {
            forwardChildLog(@ptrCast(&buf));
        }
        // EOF means the child is gone
        if (!self.stopping.load(.acquire)) {
            srv(.err, "operator()", "%s", .{@as([*:0]const u8, "llama-connect exited on its own, remote access is no longer available\n")});
        }
    }

    /// Ports `start` (server-connect.cpp:194 @8642541b2), a member of
    /// `server_connect`.
    ///
    /// Parameters:
    /// - `port`: the bound one, which differs from `params.port` with
    ///   `--port 0` or the GCP override.
    pub fn start(self: *ServerConnect, a: Allocator, params: *const Params, port: c_int) !bool {
        const func = "start";
        const bin = try findBinary(a);
        if (bin.len == 0) {
            srv(.err, func, "%s", .{@as([*:0]const u8, "llama-connect binary not found\n")});
            return false;
        }

        // main() resolves this before anything can read it from /props
        const code = params.server_connect_code;
        common.ggmlAssert(code.len != 0, "tools/server/server-connect.cpp", 203, "!code.empty()");

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

        const proc = try gpa.create(tc.subproc.Subproc);
        proc.* = .{};

        const options = tc.subproc.option_no_window |
            tc.subproc.option_combined_stdout_stderr |
            tc.subproc.option_inherit_environment;

        if (!proc.create(&args, options, &.{}, null)) {
            srv(.err, func, "failed to spawn '%s'\n", .{try zs(a, bin)});
            proc.deinit();
            gpa.destroy(proc);
            return false;
        }
        self.proc = proc;

        self.log_thread = try std.Thread.spawn(.{}, logLoop, .{self});

        srv(.inf, func, "%s", .{@as([*:0]const u8, "-----------------\n")});
        srv(.inf, func, "%s", .{@as([*:0]const u8, "remote access is enabled via llama-connect\n")});
        srv(.inf, func, "share code (enter it in the Web UI under Settings -> Remote Access): %s\n", .{try zs(a, try formatShareCode(a, code))});
        srv(.wrn, func, "%s", .{@as([*:0]const u8, "anyone with this code can use this server, do not share it publicly\n")});
        srv(.inf, func, "%s", .{@as([*:0]const u8, "-----------------\n")});

        return true;
    }

    /// Ports `stop` (server-connect.cpp:253 @8642541b2), a member of
    /// `server_connect`: idempotent, also called by the destructor.
    pub fn stop(self: *ServerConnect) void {
        const proc = self.proc orelse return;

        srv(.inf, "stop", "%s", .{@as([*:0]const u8, "stopping llama-connect...\n")});

        self.stopping.store(true, .release);

        proc.closeStdin();

        var elapsed: c_int = 0;
        while (elapsed < connect_stop_timeout_ms and proc.alive()) : (elapsed += 100) {
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            while (std.c.nanosleep(&ts, &ts) != 0) {}
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
        gpa.destroy(proc);
        self.proc = null;
    }

    /// Ports the destructor of `server_connect` (server-connect.cpp:289
    /// @8642541b2).
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
