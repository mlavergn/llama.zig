//! The server's MCP client: servers from a Cursor-style `mcpServers`
//! configuration, each a child process speaking newline-delimited JSON-RPC
//! over its standard streams, whose tools `tools.zig` exposes on `/tools`.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-mcp.cpp` and
//! `llama.cpp/tools/server/server-mcp.h` (llama.cpp v0.9.0, `cd959bae9`):
//! `server_mcp_server_config`, `server_mcp_tool_def`,
//! `server_mcp_transport`, `server_mcp_stdio` and `server_mcp`, with the
//! file's static helpers. Each declaration names what it ports and the line
//! it began at.
//!
//! # Its shape
//!
//! `server_mcp_transport` and its one subclass are one `Transport`: the
//! JSON-RPC session and the stdio pumps. The C++ hands transports around as
//! `shared_ptr`s so a concurrent evict cannot destroy one mid-call; here a
//! `Transport` is reference-counted (`retain`/`release`), and the last
//! release closes it, as the last `shared_ptr` runs its destructor.
//! `std::function<bool()>` predicates are `StopFn`s. A C++ throw is an
//! error with its `what()` in `common.what` (`error.JsonError` in
//! `cli/json`'s buffer).
//!
//! # POSIX only
//!
//! The pumps' Windows arms (`PeekNamedPipe`, `PIPE_NOWAIT`) and
//! `mcp_resolve_command`'s `SearchPathW` are not ported: on POSIX the
//! command is spawned as given, through `PATH`.

const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const tc = @import("tools_common");
const common = @import("common.zig");

const Value = json.Value;
const Allocator = std.mem.Allocator;
const Subproc = tc.subproc.Subproc;
const srv = common.srv;

const gpa = std.heap.c_allocator;

extern fn fileno(f: *anyopaque) c_int;
extern fn fcntl(fd: c_int, cmd: c_int, ...) c_int;
extern fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

const F_GETFL: c_int = 3;
const F_SETFL: c_int = 4;
const O_NONBLOCK: c_int = if (builtin.os.tag.isDarwin()) 0x0004 else 0o4000;

fn errno() c_int {
    return std.c._errno().*;
}

fn isErr(e: c_int, comptime which: std.posix.E) bool {
    return e == @backingInt(which);
}

/// `std::chrono::steady_clock::now()`, in nanoseconds.
pub fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

/// `std::this_thread::sleep_for`.
pub fn sleepMs(ms: u64) void {
    var ts: std.c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    while (std.c.nanosleep(&ts, &ts) != 0) {}
}

/// A NUL-terminated copy for a `%s` argument, freed with `common.freeZ`.
fn z(s: []const u8) [*:0]const u8 {
    return common.dupZ(s);
}

/// `std::mutex`.
pub const Mutex = struct {
    m: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    pub fn lock(self: *Mutex) void {
        _ = std.c.pthread_mutex_lock(&self.m);
    }
    pub fn unlock(self: *Mutex) void {
        _ = std.c.pthread_mutex_unlock(&self.m);
    }
};

/// `const std::function<bool()> &`: a context and a predicate; a null
/// `StopFn` never stops.
pub const StopFn = struct {
    ctx: ?*anyopaque,
    f: *const fn (?*anyopaque) bool,

    pub fn call(self: StopFn) bool {
        return self.f(self.ctx);
    }
};

/// The pipe of serialized messages each way.
const StringPipe = common.Pipe([]u8);

// -----------------------------------------------------------------------------
// the NDJSON pump

/// Ports `mcp_pump_ndjson` (server-mcp.cpp:27 @cd959bae9): reads NDJSON lines
/// from a child pipe, calling `on_line` per line until `running` clears, EOF
/// or an error, or `on_line` returns false. Polled, not blocking: a
/// grandchild can inherit the pipe's write end and hold it open.
fn pumpNdjson(f: ?*anyopaque, running: *std.atomic.Value(bool), ctx: anytype, comptime on_line: fn (@TypeOf(ctx), []u8) bool) void {
    const file = f orelse return;
    const poll_ms: c_int = 50;
    const max_line: usize = 8 * 1024 * 1024; // drop any single NDJSON line larger than this, so a child that never emits '\n' can't grow buf without bound
    const fd = fileno(file);
    const fl = fcntl(fd, F_GETFL, @as(c_int, 0));
    if (fl >= 0) _ = fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var skipping = false; // discarding an over-long line until its terminating newline
    var chunk: [4096]u8 = undefined;
    while (running.load(.seq_cst)) {
        var pfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const pr = std.c.poll(&pfd, 1, poll_ms);
        if (pr < 0) {
            if (isErr(errno(), .INTR)) continue;
            break;
        }
        if (pr == 0) continue; // timeout -> re-check running
        if ((pfd[0].revents & (std.posix.POLL.ERR | std.posix.POLL.NVAL)) != 0) break;
        const r = read(fd, &chunk, chunk.len);
        if (r < 0) {
            const e = errno();
            if (isErr(e, .INTR) or isErr(e, .AGAIN)) continue;
            break;
        }
        if (r == 0) break; // EOF: child (and any pipe writers) closed the stream
        buf.appendSlice(gpa, chunk[0..@intCast(r)]) catch break;

        // resync after an over-long, unterminated line: discard bytes until the next newline
        if (skipping) {
            const nl = std.mem.findScalar(u8, buf.items, '\n') orelse {
                if (buf.items.len > max_line) buf.clearRetainingCapacity(); // stay bounded while waiting for a terminator
                continue;
            };
            buf.replaceRangeAssumeCapacity(0, nl + 1, &.{});
            skipping = false;
        }

        while (std.mem.findScalar(u8, buf.items, '\n')) |pos| {
            var line = buf.items[0..pos];
            if (line.len != 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
            const owned = if (line.len != 0) gpa.dupe(u8, line) catch null else null;
            buf.replaceRangeAssumeCapacity(0, pos + 1, &.{});
            if (owned) |o| {
                if (!on_line(ctx, o)) return;
            }
        }

        // a partial line already larger than the cap and still no newline: drop it to avoid unbounded growth
        if (buf.items.len > max_line) {
            srv(.wrn, "mcp_pump_ndjson", "MCP: dropping oversized line (> %zu bytes) from child pipe\n", .{max_line});
            buf.clearRetainingCapacity();
            skipping = true;
        }
    }
}

/// Ports `mcp_write_all` (server-mcp.cpp:517 @cd959bae9): writes all of
/// `data` to the child's stdin, non-blocking and polled so teardown never
/// hangs.
///
/// Return: false on an error, a close or shutdown.
fn writeAll(f: ?*anyopaque, data: []const u8, running: *std.atomic.Value(bool)) bool {
    const file = f orelse return false;
    var total: usize = 0;
    const fd = fileno(file);
    const fl = fcntl(fd, F_GETFL, @as(c_int, 0));
    if (fl >= 0) _ = fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    while (total < data.len and running.load(.seq_cst)) {
        const n = write(fd, data[total..].ptr, data.len - total);
        if (n > 0) {
            total += @intCast(n);
            continue;
        }
        if (n == 0) return false;
        const e = errno();
        if (isErr(e, .INTR)) continue;
        if (!isErr(e, .AGAIN)) return false;
        var pfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.OUT, .revents = 0 }};
        const pr = std.c.poll(&pfd, 1, 50);
        if (pr < 0) {
            if (isErr(errno(), .INTR)) continue;
            return false;
        }
        if ((pfd[0].revents & (std.posix.POLL.ERR | std.posix.POLL.NVAL | std.posix.POLL.HUP)) != 0) return false;
    }
    return total == data.len;
}

// -----------------------------------------------------------------------------
// configuration

/// Ports `server_mcp_server_config` (server-mcp.h:19 @cd959bae9): one entry
/// of `mcpServers`. Strings are owned by `gpa` and live as long as the
/// manager.
pub const ServerConfig = struct {
    name: []const u8, // config key, e.g. "filesystem"
    command: []const u8 = "",
    args: std.ArrayList([]const u8) = .empty,
    /// merged over the parent env; a `std::map`, so sorted by key
    env: std.ArrayList(EnvEntry) = .empty,
    cwd: []const u8 = "",
    timeout_ms: c_int = 30000, // per-tool-call timeout

    pub const EnvEntry = struct { key: []const u8, value: []const u8 };

    fn putEnv(self: *ServerConfig, key: []const u8, value: []const u8) !void {
        var lo: usize = 0;
        var hi = self.env.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (std.mem.order(u8, self.env.items[mid].key, key) == .lt) lo = mid + 1 else hi = mid;
        }
        if (lo < self.env.items.len and std.mem.eql(u8, self.env.items[lo].key, key)) {
            self.env.items[lo].value = try gpa.dupe(u8, value);
            return;
        }
        try self.env.insert(gpa, lo, .{ .key = try gpa.dupe(u8, key), .value = try gpa.dupe(u8, value) });
    }

    fn findEnv(self: *const ServerConfig, key: []const u8) bool {
        for (self.env.items) |e| {
            if (std.mem.eql(u8, e.key, key)) return true;
        }
        return false;
    }

    /// Ports `server_mcp_server_config`'s `parse_from_json` (server-mcp.cpp:136 @cd959bae9).
    pub fn parseFromJson(out: *std.ArrayList(ServerConfig), json_str: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var j = try json.parse(arena.allocator(), json_str);
        return parseCursorFormat(out, &j);
    }

    /// Ports `server_mcp_server_config`'s `parse_cursor_format` (server-mcp.cpp:140 @cd959bae9):
    /// a missing `mcpServers` yields nothing; an entry without a `command`
    /// is skipped with a warning.
    pub fn parseCursorFormat(out: *std.ArrayList(ServerConfig), j: *const Value) !void {
        if (!j.contains("mcpServers") or !(try j.at("mcpServers")).isObject()) return;
        var it = (try j.at("mcpServers")).items();
        while (it.next()) |item| {
            const cfg = item.value;
            var sc: ServerConfig = .{ .name = try gpa.dupe(u8, item.key) };
            sc.command = try gpa.dupe(u8, try cfg.value("command", []const u8, ""));
            sc.cwd = try gpa.dupe(u8, try cfg.value("cwd", []const u8, ""));
            sc.timeout_ms = try cfg.value("timeout_ms", c_int, sc.timeout_ms);

            if (cfg.contains("args") and (try cfg.at("args")).isArray()) {
                for ((try cfg.at("args")).array.items) |*a| try sc.args.append(gpa, try gpa.dupe(u8, try a.get([]const u8)));
            }
            if (cfg.contains("env") and (try cfg.at("env")).isObject()) {
                var eit = (try cfg.at("env")).items();
                while (eit.next()) |e| try sc.putEnv(e.key, try e.value.get([]const u8));
            }

            if (sc.command.len == 0) {
                const n = z(sc.name);
                defer common.freeZ(n);
                srv(.wrn, "parse_cursor_format", "MCP server '%s' has no command, skipping\n", .{n});
                continue;
            }
            try out.append(gpa, sc);
        }
    }
};

/// Ports `server_mcp_tool_def` (server-mcp.h:33 @cd959bae9): a tool an MCP
/// server advertises. Owned by `gpa`.
pub const ToolDef = struct {
    server_name: []const u8,
    name: []const u8, // bare tool name, no "<server>_" prefix
    description: []const u8,
    input_schema: Value = .null, // JSON Schema for the arguments, or null

    /// A deep copy.
    pub fn clone(self: *const ToolDef) !ToolDef {
        return .{
            .server_name = try gpa.dupe(u8, self.server_name),
            .name = try gpa.dupe(u8, self.name),
            .description = try gpa.dupe(u8, self.description),
            .input_schema = try self.input_schema.clone(gpa),
        };
    }
};

// -----------------------------------------------------------------------------
// the transport

/// Ports `MCP_PROTOCOL_VERSION` (server-mcp.cpp:180 @cd959bae9).
const protocol_version = "2024-11-05";

/// Ports `rpc_error_message` (server-mcp.cpp:182 @cd959bae9).
///
/// Return: borrowed from `resp`, or a literal.
fn rpcErrorMessage(resp: *const Value) ![]const u8 {
    if (resp.contains("error")) {
        const e = try resp.at("error");
        if (e.isObject()) return e.value("message", []const u8, "unknown error");
        if (e.isString()) return e.get([]const u8);
    }
    return "unknown error";
}

/// Ports `mcp_result_to_response` (server-mcp.cpp:197 @cd959bae9): the text
/// parts of `result.content[]` joined, and an `isError` result surfaced.
fn resultToResponse(a: Allocator, result: *const Value) !Value {
    var text: std.ArrayList(u8) = .empty;
    if (result.contains("content") and (try result.at("content")).isArray()) {
        for ((try result.at("content")).array.items) |*part| {
            if (part.isObject() and std.mem.eql(u8, try part.value("type", []const u8, ""), "text")) {
                if (text.items.len != 0) try text.append(a, '\n');
                try text.appendSlice(a, try part.value("text", []const u8, ""));
            }
        }
    }
    var o = common.Obj.init(a);
    if (result.isObject() and try result.value("isError", bool, false)) {
        try o.put("error", if (text.items.len == 0) "MCP tool returned an error" else text.items);
        return o.done();
    }
    try o.put("plain_text_response", text.items);
    return o.done();
}

/// `{"error": {"code": -32603, "message": msg}}`.
fn rpcError(a: Allocator, msg: []const u8) !Value {
    var e = common.Obj.init(a);
    try e.put("code", @as(i64, -32603));
    try e.put("message", msg);
    var o = common.Obj.init(a);
    try o.put("error", e.done());
    return o.done();
}

/// Ports `server_mcp_transport` (server-mcp.h:50 @cd959bae9) and
/// `server_mcp_stdio` (server-mcp.h:94 @cd959bae9): one MCP server session
/// over a child's standard streams.
///
///   caller --send_rpc--> to_server   --[writer]--> framing --> server
///   caller <--send_rpc-- from_server <--[reader]-- framing <-- server
pub const Transport = struct {
    name: []const u8,
    timeout_ms: c_int = 30000,

    to_server: StringPipe, // serialized messages we send to the server
    from_server: StringPipe, // serialized messages read from the server

    // per-transport: send_rpc() holds it across the reply wait. guards the members below.
    rpc_mutex: Mutex = .{},
    next_id: u64 = 1, // reset to 1 per (re)spawn
    initialized: bool = false,
    last_error: std.ArrayList(u8) = .empty,
    tools: std.ArrayList(ToolDef) = .empty,

    config: *const ServerConfig,
    proc: ?*Subproc = null,
    reader: ?std.Thread = null, // child stdout -> NDJSON de-framing -> from_server
    writer: ?std.Thread = null, // to_server -> NDJSON framing -> child stdin
    errlog: ?std.Thread = null, // child stderr -> debug log (must be drained or the child blocks)

    // cleared by close() or by the reader on stdout EOF; read without rpc_mutex
    running: std.atomic.Value(bool) = .init(false),

    // bounded tail of the child's stderr, for diagnostics when it dies
    err_mu: Mutex = .{},
    err_tail: std.ArrayList(u8) = .empty,

    /// The `shared_ptr`'s count.
    refs: std.atomic.Value(u32) = .init(1),

    /// Ports the constructor of `server_mcp_stdio` (server-mcp.cpp:441 @cd959bae9).
    ///
    /// Return: a transport with one reference; `release` frees it.
    pub fn create(config: *const ServerConfig) !*Transport {
        const self = try gpa.create(Transport);
        self.* = .{
            .name = config.name,
            .timeout_ms = config.timeout_ms,
            .to_server = StringPipe.init(gpa),
            .from_server = StringPipe.init(gpa),
            .config = config,
        };
        // bound the reply queue: send_rpc only drains during a call, so unsolicited notifications would otherwise grow it without limit
        self.from_server.max_size = 65536;
        return self;
    }

    pub fn retain(self: *Transport) *Transport {
        _ = self.refs.fetchAdd(1, .seq_cst);
        return self;
    }

    /// Drops a reference; the last runs the destructor
    /// (`~server_mcp_stdio`, server-mcp.cpp:448 @cd959bae9): the pumps
    /// joined, the child reaped.
    pub fn release(self: *Transport) void {
        if (self.refs.fetchSub(1, .seq_cst) != 1) return;
        self.joinPumps();
        drain(&self.to_server);
        drain(&self.from_server);
        self.to_server.deinit();
        self.from_server.deinit();
        self.last_error.deinit(gpa);
        self.err_tail.deinit(gpa);
        for (self.tools.items) |*t| freeDef(t);
        self.tools.deinit(gpa);
        gpa.destroy(self);
    }

    fn drain(p: *StringPipe) void {
        p.closeWrite();
        var s: []u8 = undefined;
        while (p.read(&s, null, false)) gpa.free(s);
    }

    fn setLastError(self: *Transport, comptime fmt: []const u8, args: anytype) void {
        self.last_error.clearRetainingCapacity();
        self.last_error.print(gpa, fmt, args) catch {};
    }

    /// Ports `server_mcp_transport`'s `send_rpc` (server-mcp.cpp:215 @cd959bae9): the
    /// reply, or an `{"error": ...}`. Assumes `rpc_mutex` is held.
    fn sendRpc(self: *Transport, a: Allocator, request: *const Value, should_stop: ?StopFn) !Value {
        const dumped = try json.dump(gpa, request, -1);
        if (!(try self.to_server.write(dumped))) {
            gpa.free(dumped);
            return rpcError(a, "transport closed");
        }

        const has_id = request.contains("id");
        const deadline = nowNs() + @as(i128, self.timeout_ms) * std.time.ns_per_ms;
        const Stop = struct {
            should_stop: ?StopFn,
            deadline: i128,
            fn f(ctx: *anyopaque) bool {
                const s: *@This() = @ptrCast(@alignCast(ctx));
                return (if (s.should_stop) |p| p.call() else false) or nowNs() >= s.deadline;
            }
        };
        var stop: Stop = .{ .should_stop = should_stop, .deadline = deadline };

        var frame: []u8 = undefined;
        while (self.from_server.read(&frame, .{ .ctx = &stop, .f = Stop.f }, false)) {
            defer gpa.free(frame);
            const reply = json.parse(a, frame) catch |e| {
                if (e == error.OutOfMemory) return e;
                if (nowNs() >= deadline) break;
                continue; // skip malformed frame
            };
            // no id: a notification. mismatched id: a stale reply from a timed-out request (ids are monotonic, never a future one)
            if (!has_id or (reply.contains("id") and Value.eql(try reply.at("id"), try request.at("id")))) {
                return reply;
            }
            if (nowNs() >= deadline) break; // a flood of notifications must not outrun the deadline
        }

        if (should_stop) |s| {
            if (s.call()) return rpcError(a, "cancelled");
        }
        if (nowNs() >= deadline) return rpcError(a, "request timed out");
        return rpcError(a, "transport closed");
    }

    /// Ports `server_mcp_transport`'s `ensure_init` (server-mcp.cpp:255 @cd959bae9): the
    /// `initialize` handshake, once. Assumes `rpc_mutex` is held.
    fn ensureInit(self: *Transport, a: Allocator, should_stop: ?StopFn) !bool {
        if (self.initialized) return true;

        var client_info = common.Obj.init(a);
        try client_info.put("name", "llama.cpp");
        try client_info.put("version", "1.0");
        var params = common.Obj.init(a);
        try params.put("protocolVersion", protocol_version);
        try params.put("capabilities", Value.emptyObject());
        try params.put("clientInfo", client_info.done());
        var init_req = common.Obj.init(a);
        try init_req.put("jsonrpc", "2.0");
        try init_req.put("id", self.next_id);
        self.next_id += 1;
        try init_req.put("method", "initialize");
        try init_req.put("params", params.done());
        const req = init_req.done();
        const resp = try self.sendRpc(a, &req, should_stop);
        if (!resp.contains("result")) {
            const m = try rpcErrorMessage(&resp);
            self.setLastError("initialize failed: {s}", .{m});
            return false;
        }

        // notifications/initialized: no id, no reply expected
        var notif = common.Obj.init(a);
        try notif.put("jsonrpc", "2.0");
        try notif.put("method", "notifications/initialized");
        const nv = notif.done();
        const nd = try json.dump(gpa, &nv, -1);
        if (!(try self.to_server.write(nd))) gpa.free(nd);

        self.initialized = true;
        return true;
    }

    /// Ports `server_mcp_transport`'s `list_tools` (server-mcp.cpp:284 @cd959bae9): the
    /// server's tools, asked once and cached.
    ///
    /// Return: copies, owned by the caller (`freeDefs`).
    pub fn listTools(self: *Transport, should_stop: ?StopFn) ![]ToolDef {
        self.rpc_mutex.lock();
        defer self.rpc_mutex.unlock();
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        if (!try self.ensureInit(a, should_stop)) return &.{};
        if (self.tools.items.len != 0) return cloneDefs(self.tools.items);

        var req = common.Obj.init(a);
        try req.put("jsonrpc", "2.0");
        try req.put("id", self.next_id);
        self.next_id += 1;
        try req.put("method", "tools/list");
        const rv = req.done();
        const resp = try self.sendRpc(a, &rv, should_stop);
        if (!resp.contains("result")) {
            const m = try rpcErrorMessage(&resp);
            self.setLastError("tools/list failed: {s}", .{m});
            return &.{};
        }

        const result = try resp.at("result");
        if (result.contains("tools") and (try result.at("tools")).isArray()) {
            for ((try result.at("tools")).array.items) |*t| {
                var def: ToolDef = .{
                    .server_name = try gpa.dupe(u8, self.name),
                    .name = try gpa.dupe(u8, try t.value("name", []const u8, "")),
                    .description = try gpa.dupe(u8, try t.value("description", []const u8, "")),
                };
                if (t.contains("inputSchema")) def.input_schema = try (try t.at("inputSchema")).clone(gpa);
                try self.tools.append(gpa, def);
            }
        }
        return cloneDefs(self.tools.items);
    }

    /// Ports `server_mcp_transport`'s `call_tool` (server-mcp.cpp:316 @cd959bae9): the
    /// `tools/call` result in the `/tools` shape, or an `{"error": ...}`.
    ///
    /// Return: allocated by `a`.
    pub fn callTool(self: *Transport, a: Allocator, tool_name: []const u8, arguments: *const Value, should_stop: ?StopFn) !Value {
        self.rpc_mutex.lock();
        defer self.rpc_mutex.unlock();
        if (!try self.ensureInit(a, should_stop)) {
            var o = common.Obj.init(a);
            try o.put("error", self.last_error.items);
            return o.done();
        }

        var params = common.Obj.init(a);
        try params.put("name", tool_name);
        try params.put("arguments", try arguments.clone(a));
        var req = common.Obj.init(a);
        try req.put("jsonrpc", "2.0");
        try req.put("id", self.next_id);
        self.next_id += 1;
        try req.put("method", "tools/call");
        try req.put("params", params.done());
        const rv = req.done();
        const resp = try self.sendRpc(a, &rv, should_stop);
        if (resp.contains("error")) {
            var o = common.Obj.init(a);
            try o.put("error", try rpcErrorMessage(&resp));
            return o.done();
        }
        if (resp.contains("result")) return resultToResponse(a, try resp.at("result"));
        var o = common.Obj.init(a);
        try o.put("error", "invalid response from MCP server");
        return o.done();
    }

    /// Ports `server_mcp_stdio`'s `start` (server-mcp.cpp:452 @cd959bae9): spawns the
    /// child and its three pumps.
    pub fn start(self: *Transport) bool {
        const config = self.config;
        var argv_s: std.ArrayList([]const u8) = .empty;
        defer argv_s.deinit(gpa);
        argv_s.append(gpa, config.command) catch return false; // mcp_resolve_command: the command as given on POSIX
        argv_s.appendSlice(gpa, config.args.items) catch return false;

        var options = tc.subproc.option_no_window | tc.subproc.option_search_user_path;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var envp_s: []const []const u8 = &.{};
        if (config.env.items.len == 0) {
            options |= tc.subproc.option_inherit_environment;
        } else {
            envp_s = buildEnv(arena.allocator(), config) catch return false;
        }

        const handle = gpa.create(Subproc) catch return false;
        handle.* = .{};
        const ok = handle.create(argv_s.items, options, envp_s, if (config.cwd.len == 0) null else config.cwd);
        if (!ok) {
            gpa.destroy(handle);
            const n = z(config.name);
            defer common.freeZ(n);
            const cmd = z(config.command);
            defer common.freeZ(cmd);
            srv(.wrn, "start", "MCP '%s': failed to spawn '%s'\n", .{ n, cmd });
            return false;
        }

        self.proc = handle;
        self.running.store(true, .seq_cst);
        self.reader = std.Thread.spawn(.{}, readerLoop, .{self}) catch null;
        self.writer = std.Thread.spawn(.{}, writerLoop, .{self}) catch null;
        self.errlog = std.Thread.spawn(.{}, errlogLoop, .{self}) catch null;
        return true;
    }

    /// Ports `server_mcp_stdio`'s `close` (server-mcp.cpp:483 @cd959bae9): blocking and
    /// idempotent.
    pub fn close(self: *Transport) void {
        self.joinPumps();
    }

    /// Ports `server_mcp_stdio`'s `is_alive` (server-mcp.cpp:487 @cd959bae9): never
    /// blocks behind an in-flight `send_rpc()`.
    pub fn isAlive(self: *Transport) bool {
        return self.running.load(.seq_cst);
    }

    /// Ports `server_mcp_stdio`'s `diagnostics` (server-mcp.cpp:491 @cd959bae9): the last
    /// RPC error and the tail of the child's stderr.
    ///
    /// Return: owned by `a`.
    pub fn diagnostics(self: *Transport, a: Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        {
            self.rpc_mutex.lock(); // last_error is written by send_rpc's callers
            defer self.rpc_mutex.unlock();
            try out.appendSlice(a, self.last_error.items);
        }
        self.err_mu.lock();
        defer self.err_mu.unlock();
        if (self.err_tail.items.len != 0) {
            if (out.items.len != 0) try out.appendSlice(a, "; ");
            try out.appendSlice(a, "last stderr: ");
            try out.appendSlice(a, self.err_tail.items);
        }
        return out.items;
    }

    /// Ports `server_mcp_stdio`'s `reader_loop` (server-mcp.cpp:507 @cd959bae9).
    fn readerLoop(self: *Transport) void {
        const P = struct {
            fn onLine(t: *Transport, line: []u8) bool {
                const ok = t.from_server.write(line) catch false; // false => consumer gone, stop
                if (!ok) gpa.free(line);
                return ok;
            }
        };
        pumpNdjson(self.proc.?.stdoutFile(), &self.running, self, P.onLine);
        self.running.store(false, .seq_cst);
        self.to_server.closeWrite(); // stop the writer
        self.from_server.closeWrite(); // EOF to any waiting caller
    }

    /// Ports `server_mcp_stdio`'s `writer_loop` (server-mcp.cpp:583 @cd959bae9).
    fn writerLoop(self: *Transport) void {
        const Stop = struct {
            fn f(ctx: *anyopaque) bool {
                const t: *Transport = @ptrCast(@alignCast(ctx));
                return !t.running.load(.seq_cst);
            }
        };
        var msg: []u8 = undefined;
        while (self.to_server.read(&msg, .{ .ctx = self, .f = Stop.f }, true)) {
            const framed = std.mem.concat(gpa, u8, &.{ msg, "\n" }) catch {
                gpa.free(msg);
                break;
            };
            gpa.free(msg);
            defer gpa.free(framed);
            if (!writeAll(self.proc.?.stdinFile(), framed, &self.running)) break; // child gone or shutting down
        }
        self.running.store(false, .seq_cst);
        self.to_server.closeRead(); // fail fast on any further send_rpc write
        self.from_server.closeWrite(); // wake any caller waiting for a reply
    }

    /// Ports `server_mcp_stdio`'s `errlog_loop` (server-mcp.cpp:597 @cd959bae9): drains
    /// stderr (an undrained pipe blocks the child), logging it and keeping a
    /// bounded tail for when the server dies.
    fn errlogLoop(self: *Transport) void {
        const err_tail_max: usize = 4096;
        const P = struct {
            fn onLine(t: *Transport, line: []u8) bool {
                defer gpa.free(line);
                const n = z(t.name);
                defer common.freeZ(n);
                const l = z(line);
                defer common.freeZ(l);
                srv(.dbg, "operator()", "MCP '%s' stderr: %s\n", .{ n, l });
                t.err_mu.lock();
                defer t.err_mu.unlock();
                t.err_tail.appendSlice(gpa, line) catch |err| switch (err) {
                    error.OutOfMemory => {},
                };
                t.err_tail.append(gpa, '\n') catch |err| switch (err) {
                    error.OutOfMemory => {},
                };
                if (t.err_tail.items.len > err_tail_max) {
                    t.err_tail.replaceRangeAssumeCapacity(0, t.err_tail.items.len - err_tail_max, &.{});
                }
                return true;
            }
        };
        pumpNdjson(self.proc.?.stderrFile(), &self.running, self, P.onLine);
    }

    /// Ports `server_mcp_stdio`'s `join_pumps` (server-mcp.cpp:613 @cd959bae9).
    fn joinPumps(self: *Transport) void {
        const proc = self.proc orelse return;
        self.running.store(false, .seq_cst);
        self.to_server.closeWrite(); // wake the writer if it waits for a message
        self.from_server.closeWrite(); // wake any caller waiting for a reply

        proc.terminate(); // child death unblocks the blocked fread/fwrite

        if (self.writer) |t| t.join();
        if (self.reader) |t| t.join();
        if (self.errlog) |t| t.join();
        self.writer = null;
        self.reader = null;
        self.errlog = null;

        _ = proc.join(); // reap the child: never waiting would leave the pid a zombie for the process lifetime
        proc.deinit();
        gpa.destroy(proc);
        self.proc = null;
    }
};

/// Ports `mcp_build_env` (server-mcp.cpp:426 @cd959bae9) over
/// `mcp_parent_env` (server-mcp.cpp:405 @cd959bae9): the parent's
/// environment with the configured overrides applied, `KEY=VALUE`.
fn buildEnv(a: Allocator, config: *const ServerConfig) ![]const []const u8 {
    var env: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (std.c.environ[i]) |e_ptr| : (i += 1) {
        const e = std.mem.span(e_ptr);
        const key = if (std.mem.findScalar(u8, e, '=')) |eq| e[0..eq] else e;
        if (!config.findEnv(key)) try env.append(a, e);
    }
    for (config.env.items) |kv| try env.append(a, try std.mem.concat(a, u8, &.{ kv.key, "=", kv.value }));
    return env.items;
}

fn cloneDefs(defs: []const ToolDef) ![]ToolDef {
    const out = try gpa.alloc(ToolDef, defs.len);
    for (defs, 0..) |*d, i| out[i] = try d.clone();
    return out;
}

/// Frees one definition's strings and schema.
pub fn freeDef(d: *ToolDef) void {
    gpa.free(d.server_name);
    gpa.free(d.name);
    gpa.free(d.description);
    d.input_schema.deinit(gpa);
}

/// Frees what `listTools` returned.
pub fn freeDefs(defs: []ToolDef) void {
    for (defs) |*d| freeDef(d);
    if (defs.len != 0) gpa.free(defs);
}

// -----------------------------------------------------------------------------
// server_mcp

/// Ports `MCP_COOLDOWN_SECONDS` (server-mcp.cpp:636 @cd959bae9).
const cooldown_seconds = 5;
/// Ports `MCP_WARMUP_TIMEOUT_SECONDS` (server-mcp.cpp:637 @cd959bae9): caps
/// per-server tool discovery at startup.
const warmup_timeout_seconds = 10;

/// Ports `server_mcp` (server-mcp.h:132 @cd959bae9): the configured servers,
/// a live transport per server spawned on first use, and the tools
/// discovered at startup. Declare it before the HTTP context, so it
/// outlives every `/tools` handler.
pub const Mcp = struct {
    configs: std.ArrayList(ServerConfig) = .empty,

    mutex: Mutex = .{}, // guards transports, dead_servers, registry

    transports: std.ArrayList(Named(*Transport)) = .empty,
    dead_servers: std.ArrayList(Named(i128)) = .empty, // spawn-failure cooldown
    registry: std.ArrayList(ToolDef) = .empty,

    stopping: std.atomic.Value(bool) = .init(false),

    fn Named(comptime V: type) type {
        return struct { name: []const u8, value: V };
    }

    /// Ports `server_mcp`'s destructor (server-mcp.cpp:639 @cd959bae9): shuts
    /// down, then closes every transport.
    pub fn deinit(self: *Mcp) void {
        self.shutdown();
        var to_close: std.ArrayList(*Transport) = .empty;
        defer to_close.deinit(gpa);
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            for (self.transports.items) |t| to_close.append(gpa, t.value) catch |err| switch (err) {
                error.OutOfMemory => {},
            };
            self.transports.clearRetainingCapacity();
        }
        for (to_close.items) |t| {
            t.close();
            t.release();
        }
        self.transports.deinit(gpa);
        self.dead_servers.deinit(gpa);
        for (self.registry.items) |*d| freeDef(d);
        self.registry.deinit(gpa);
    }

    /// Ports `server_mcp`'s `create_transport` (server-mcp.cpp:655 @cd959bae9): the
    /// only place that names a concrete transport.
    fn createTransport(cfg: *const ServerConfig) !*Transport {
        return Transport.create(cfg);
    }

    /// Ports `server_mcp`'s `shutdown` (server-mcp.cpp:659 @cd959bae9): flips the cancel
    /// flag so in-flight calls return. Idempotent.
    pub fn shutdown(self: *Mcp) void {
        self.stopping.store(true, .seq_cst);
    }

    /// Ports `server_mcp`'s `find_config` (server-mcp.cpp:663 @cd959bae9).
    fn findConfig(self: *const Mcp, name: []const u8) ?*const ServerConfig {
        for (self.configs.items) |*cfg| {
            if (std.mem.eql(u8, cfg.name, name)) return cfg;
        }
        return null;
    }

    /// Ports `server_mcp`'s `empty` (server-mcp.h:143 @cd959bae9): true until
    /// `start` has parsed a server.
    pub fn empty(self: *const Mcp) bool {
        return self.configs.items.len == 0;
    }

    /// The `append` lambda of `start`: one JSON configuration's servers,
    /// names unique across both sources.
    fn append(self: *Mcp, json_str: []const u8) !void {
        var parsed: std.ArrayList(ServerConfig) = .empty;
        defer parsed.deinit(gpa);
        ServerConfig.parseFromJson(&parsed, json_str) catch |e| {
            if (e == error.OutOfMemory) return e;
            return common.exception.rethrow(error.RuntimeError, e, "failed to parse MCP config JSON: ", "");
        };
        if (parsed.items.len == 0) {
            srv(.wrn, "operator()", "%s", .{@as([*:0]const u8, "MCP config: no servers found in JSON\n")});
        }
        for (parsed.items) |p| {
            // names must be unique across both config sources: get_or_create / find_config key on the name
            if (self.findConfig(p.name) != null) {
                const n = z(p.name);
                defer common.freeZ(n);
                srv(.wrn, "operator()", "MCP config: duplicate server name '%s', skipping\n", .{n});
                continue;
            }
            try self.configs.append(gpa, p);
        }
    }

    /// Ports `server_mcp`'s `start` (server-mcp.cpp:672 @cd959bae9): parses the
    /// configuration (file and inline JSON), then spawns each server once,
    /// lists its tools and shuts it down. Throws on a parse error; a spawn
    /// failure is logged.
    ///
    /// Parameters:
    /// - `config_path`: `--mcp-servers-config`, or empty.
    /// - `config_json`: `--mcp-servers-json`, or empty.
    pub fn start(self: *Mcp, config_path: []const u8, config_json: []const u8) !void {
        if (config_path.len != 0) {
            const text = readWholeFile(config_path) orelse
                return common.throw(error.RuntimeError, "failed to open MCP config file: {s}", .{config_path});
            defer gpa.free(text);
            try self.append(text);
        }
        if (config_json.len != 0) try self.append(config_json);

        if (self.configs.items.len == 0) return;

        var discovered: std.ArrayList(ToolDef) = .empty;
        for (self.configs.items) |*cfg| {
            const t = try createTransport(cfg);
            defer t.release();
            if (!t.start()) {
                var arena = std.heap.ArenaAllocator.init(gpa);
                defer arena.deinit();
                const n = z(cfg.name);
                defer common.freeZ(n);
                const d = z(try t.diagnostics(arena.allocator()));
                defer common.freeZ(d);
                srv(.wrn, "start", "MCP warmup: failed to spawn '%s': %s\n", .{ n, d });
                continue;
            }
            // bound warmup per server so an unresponsive one can't stall startup for the full per-call timeout
            const Stop = struct {
                mgr: *Mcp,
                deadline: i128,
                fn f(ctx: ?*anyopaque) bool {
                    const s: *@This() = @ptrCast(@alignCast(ctx.?));
                    return s.mgr.stopping.load(.seq_cst) or nowNs() >= s.deadline;
                }
            };
            var stop: Stop = .{ .mgr = self, .deadline = nowNs() + warmup_timeout_seconds * std.time.ns_per_s };
            const tools = try t.listTools(.{ .ctx = &stop, .f = Stop.f });
            defer if (tools.len != 0) gpa.free(tools);
            const n = z(cfg.name);
            defer common.freeZ(n);
            srv(.inf, "start", "MCP warmup: '%s' discovered %zu tools\n", .{ n, tools.len });
            try discovered.appendSlice(gpa, tools);
            t.close();
        }

        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.registry.items) |*d| freeDef(d);
        self.registry.deinit(gpa);
        self.registry = discovered;
    }

    /// Ports `server_mcp`'s `list_tools` (server-mcp.cpp:730 @cd959bae9).
    ///
    /// Return: copies, owned by the caller (`freeDefs`).
    pub fn listTools(self: *Mcp) ![]ToolDef {
        self.mutex.lock();
        defer self.mutex.unlock();
        return cloneDefs(self.registry.items);
    }

    /// Ports `server_mcp`'s `call_tool` (server-mcp.cpp:735 @cd959bae9): lazily
    /// (re)spawns the transport; the MCP result or an `{"error": ...}`.
    /// `should_stop` is or-ed with the manager's cancel flag.
    ///
    /// Return: allocated by `a`.
    pub fn callTool(self: *Mcp, a: Allocator, server_name: []const u8, tool_name: []const u8, arguments: *const Value, should_stop: ?StopFn) !Value {
        const transport = (try self.getOrCreate(server_name)) orelse {
            var o = common.Obj.init(a);
            try o.put("error", try std.mem.concat(a, u8, &.{ "MCP server unavailable: ", server_name }));
            return o.done();
        };
        defer transport.release();

        const Stop = struct {
            mgr: *Mcp,
            should_stop: ?StopFn,
            fn f(ctx: ?*anyopaque) bool {
                const s: *@This() = @ptrCast(@alignCast(ctx.?));
                return s.mgr.stopping.load(.seq_cst) or (if (s.should_stop) |p| p.call() else false);
            }
        };
        var stop: Stop = .{ .mgr = self, .should_stop = should_stop };
        return transport.callTool(a, tool_name, arguments, .{ .ctx = &stop, .f = Stop.f });
    }

    /// Ports `server_mcp`'s `get_or_create` (server-mcp.cpp:750 @cd959bae9): the live
    /// transport, respawned when its child died; null during the cooldown
    /// after a failed spawn, or during shutdown.
    ///
    /// Return: a reference the caller releases.
    fn getOrCreate(self: *Mcp, name: []const u8) !?*Transport {
        var to_close: std.ArrayList(*Transport) = .empty; // closed after unlock
        defer to_close.deinit(gpa);
        var result: ?*Transport = null;

        {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.stopping.load(.seq_cst)) return null;

            const now = nowNs();
            for (self.dead_servers.items, 0..) |d, i| {
                if (std.mem.eql(u8, d.name, name)) {
                    if (now < d.value) return null;
                    _ = self.dead_servers.orderedRemove(i);
                    break;
                }
            }

            for (self.transports.items, 0..) |t, i| {
                if (!std.mem.eql(u8, t.name, name)) continue;
                if (t.value.isAlive()) return t.value.retain();
                var arena = std.heap.ArenaAllocator.init(gpa);
                defer arena.deinit();
                const n = z(name);
                defer common.freeZ(n);
                const d = z(try t.value.diagnostics(arena.allocator()));
                defer common.freeZ(d);
                srv(.wrn, "get_or_create", "MCP '%s' is no longer alive: %s\n", .{ n, d });
                try to_close.append(gpa, t.value);
                _ = self.transports.orderedRemove(i);
                break;
            }

            if (self.findConfig(name)) |cfg| {
                const fresh = try createTransport(cfg);
                if (fresh.start() and fresh.isAlive()) {
                    try self.transports.append(gpa, .{ .name = cfg.name, .value = fresh });
                    result = fresh.retain();
                } else {
                    var arena = std.heap.ArenaAllocator.init(gpa);
                    defer arena.deinit();
                    const n = z(name);
                    defer common.freeZ(n);
                    const d = z(try fresh.diagnostics(arena.allocator()));
                    defer common.freeZ(d);
                    srv(.wrn, "get_or_create", "MCP '%s': failed to start: %s\n", .{ n, d });
                    try to_close.append(gpa, fresh);
                    try self.setDead(cfg.name, now + cooldown_seconds * std.time.ns_per_s);
                }
            }
        }

        for (to_close.items) |t| {
            t.close(); // blocking call, no leaks
            t.release();
        }
        return result;
    }

    fn setDead(self: *Mcp, name: []const u8, until: i128) !void {
        for (self.dead_servers.items) |*d| {
            if (std.mem.eql(u8, d.name, name)) {
                d.value = until;
                return;
            }
        }
        try self.dead_servers.append(gpa, .{ .name = name, .value = until });
    }
};

/// `fs_open_ifstream` and `ss << f.rdbuf()`: the whole file, or null when
/// it cannot be opened.
fn readWholeFile(path: []const u8) ?[]u8 {
    const pz = gpa.dupeSentinel(u8, path, 0) catch return null;
    defer gpa.free(pz);
    const fd = std.c.open(pz.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = read(fd, &buf, buf.len);
        if (n <= 0) break;
        out.appendSlice(gpa, buf[0..@intCast(n)]) catch return null;
    }
    return out.toOwnedSlice(gpa) catch null;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
