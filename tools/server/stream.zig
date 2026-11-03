//! Resumable stream sessions: a generation's SSE bytes kept per
//! conversation, so a client that reconnects can replay them (`/v1/stream`),
//! and the tee that feeds them from a streamed response.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-stream.cpp` and
//! `llama.cpp/tools/server/server-stream.h` (llama.cpp v0.6.0,
//! `d81235049`). Each declaration names what it ports and the line it began
//! at.
//!
//! # Shapes
//!
//! A `std::shared_ptr<stream_session>` is a `*Session` with an atomic
//! reference count: the manager's map holds one, each pipe end one. The
//! map is libc++'s `unordered_map<std::string, ...>` (`libcxx.StringMap`),
//! because `/v1/streams/lookup` and the "live" list in a warning are in its
//! order. It has no erase; an evicted entry is a tombstone (null), which
//! iteration skips and a later session for the same id reuses -- libc++
//! would relink the node, possibly elsewhere, and its bucket count counts
//! live entries only. Both differ only after an eviction, and only in the
//! order of a later lookup's answer.
//!
//! The `std::shared_mutex` is a plain mutex: the readers' sharing was never
//! observable.

const std = @import("std");
const json = @import("json");
const libcxx = @import("llama").libcxx;

const http = @import("http.zig");
const common = @import("common.zig");

const Value = json.Value;
const Allocator = std.mem.Allocator;

const gpa = std.heap.c_allocator;
const srv = common.srv;

/// Ports `stream_read_status` (server-stream.cpp:11 @d81235049).
const ReadStatus = enum { ok, offset_lost };

/// Ports `STREAM_SESSION_TTL_SECONDS`, `STREAM_SESSION_MAX_BYTES`,
/// `STREAM_SESSION_GC_INTERVAL_SECONDS` and `STREAM_READ_WAKE_INTERVAL_MS`
/// (server-stream.cpp:17, 18, 19, 20 @d81235049).
const session_ttl_seconds: i64 = 300;
const session_max_bytes: usize = 4 * 1024 * 1024;
const session_gc_interval_seconds: i64 = 60;
const read_wake_interval_ms: i64 = 200;

/// Ports `now_seconds` (server-stream.cpp:22 @d81235049).
fn nowSeconds() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @intCast(ts.sec);
}

/// A pthread mutex, as `queue.zig` takes them.
const Mutex = struct {
    m: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    fn lock(self: *Mutex) void {
        _ = std.c.pthread_mutex_lock(&self.m);
    }
    fn unlock(self: *Mutex) void {
        _ = std.c.pthread_mutex_unlock(&self.m);
    }
};

/// A pthread condition variable.
const Cond = struct {
    c: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
    fn waitFor(self: *Cond, m: *Mutex, ns: u64) void {
        var now: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &now);
        var sec: i64 = @as(i64, @intCast(now.sec)) + @as(i64, @intCast(ns / std.time.ns_per_s));
        var nsec: i64 = @as(i64, @intCast(now.nsec)) + @as(i64, @intCast(ns % std.time.ns_per_s));
        if (nsec >= std.time.ns_per_s) {
            sec += 1;
            nsec -= std.time.ns_per_s;
        }
        const deadline: std.c.timespec = .{ .sec = @intCast(sec), .nsec = @intCast(nsec) };
        _ = std.c.pthread_cond_timedwait(&self.c, &m.m, &deadline);
    }
    fn notifyAll(self: *Cond) void {
        _ = std.c.pthread_cond_broadcast(&self.c);
    }
};

// -----------------------------------------------------------------------------
// stream_session

/// Ports `stream_session` (server-stream.cpp:75 @d81235049): the streaming
/// buffer for one generation, which survives an HTTP disconnect. The
/// producer appends SSE bytes; readers drain from any offset.
pub const Session = struct {
    conversation_id: []u8,
    started_ts: i64, // unix seconds at construction

    refs: std.atomic.Value(u32) = .init(1),

    mu: Mutex = .{},
    cv: Cond = .{},
    buffer: std.ArrayList(u8) = .empty,
    prefix_dropped: usize = 0,
    cap_bytes: usize,
    done: bool = false,
    cancelled: std.atomic.Value(bool) = .init(false), // polled lock-free by the should_stop closure, no mu
    completed_ts: i64 = 0,

    /// Ports `stream_session`'s constructor (server-stream.cpp:111 @d81235049).
    fn create(conversation_id: []const u8, max_bytes: usize) !*Session {
        const self = try gpa.create(Session);
        errdefer gpa.destroy(self);
        self.* = .{
            .conversation_id = try gpa.dupe(u8, conversation_id),
            .started_ts = nowSeconds(),
            .cap_bytes = max_bytes,
        };
        try self.buffer.ensureTotalCapacity(gpa, 64 * 1024);
        return self;
    }

    /// The `shared_ptr` copy.
    pub fn acquire(self: *Session) *Session {
        _ = self.refs.fetchAdd(1, .acq_rel);
        return self;
    }

    /// The `shared_ptr` release: the last one frees.
    pub fn release(self: *Session) void {
        if (self.refs.fetchSub(1, .acq_rel) == 1) {
            self.buffer.deinit(gpa);
            gpa.free(self.conversation_id);
            gpa.destroy(self);
        }
    }

    /// Ports `stream_session`'s `append` (server-stream.cpp:122 @d81235049).
    fn append(self: *Session, data: []const u8) !bool {
        if (data.len == 0) {
            return true;
        }
        {
            self.mu.lock();
            defer self.mu.unlock();
            if (self.done) {
                return false;
            }
            if (data.len >= self.cap_bytes) {
                // single chunk bigger than the cap, keep only the tail that fits
                const skip = data.len - self.cap_bytes;
                self.prefix_dropped += self.buffer.items.len + skip;
                self.buffer.clearRetainingCapacity();
                try self.buffer.appendSlice(gpa, data[skip..]);
            } else {
                const needed = self.buffer.items.len + data.len;
                if (needed > self.cap_bytes) {
                    const to_drop = needed - self.cap_bytes;
                    std.mem.copyForwards(u8, self.buffer.items[0 .. self.buffer.items.len - to_drop], self.buffer.items[to_drop..]);
                    self.buffer.shrinkRetainingCapacity(self.buffer.items.len - to_drop);
                    self.prefix_dropped += to_drop;
                }
                try self.buffer.appendSlice(gpa, data);
            }
        }
        self.cv.notifyAll();
        return true;
    }

    /// Ports `stream_session`'s `finalize` (server-stream.cpp:151 @d81235049).
    fn finalize(self: *Session) void {
        {
            self.mu.lock();
            defer self.mu.unlock();
            if (self.done) {
                return;
            }
            self.done = true;
            self.completed_ts = nowSeconds();
        }
        self.cv.notifyAll();
    }

    /// Ports `stream_session`'s `read_from` (server-stream.cpp:163
    /// @d81235049): drain from `offset` into `sink`, blocking for more bytes
    /// or the end. `offset_lost` if the offset fell below the dropped prefix.
    fn readFrom(self: *Session, offset_in: usize, sink: anytype, should_stop: anytype) !ReadStatus {
        var offset = offset_in;
        self.mu.lock();
        while (true) {
            if (should_stop.call()) {
                self.mu.unlock();
                return .ok;
            }
            if (offset < self.prefix_dropped) {
                self.mu.unlock();
                return .offset_lost;
            }
            const logical_end = self.prefix_dropped + self.buffer.items.len;
            if (offset < logical_end) {
                const local_off = offset - self.prefix_dropped;
                // copy the available chunk under the lock, release before calling the sink
                const chunk = gpa.dupe(u8, self.buffer.items[local_off..]) catch |e| {
                    self.mu.unlock();
                    return e;
                };
                defer gpa.free(chunk);
                offset += chunk.len;
                self.mu.unlock();
                const keep_going = try sink.call(chunk);
                if (!keep_going) {
                    return .ok;
                }
                self.mu.lock();
                continue;
            }
            if (self.done) {
                self.mu.unlock();
                return .ok;
            }
            // wait for new bytes, finalize, or a periodic wake to re check should_stop
            self.cv.waitFor(&self.mu, @intCast(read_wake_interval_ms * std.time.ns_per_ms));
        }
    }

    /// Ports `stream_session`'s `is_done` (server-stream.cpp:197 @d81235049).
    fn isDone(self: *Session) bool {
        self.mu.lock();
        defer self.mu.unlock();
        return self.done;
    }

    /// Ports `stream_session`'s `total_size` (server-stream.cpp:202 @d81235049).
    fn totalSize(self: *Session) usize {
        self.mu.lock();
        defer self.mu.unlock();
        return self.prefix_dropped + self.buffer.items.len;
    }

    /// Ports `stream_session`'s `dropped_prefix` (server-stream.cpp:207 @d81235049).
    fn droppedPrefix(self: *Session) usize {
        self.mu.lock();
        defer self.mu.unlock();
        return self.prefix_dropped;
    }

    /// Ports `stream_session`'s `completed_at` (server-stream.cpp:212 @d81235049).
    fn completedAt(self: *Session) i64 {
        self.mu.lock();
        defer self.mu.unlock();
        return self.completed_ts;
    }

    /// Ports `stream_session`'s `cancel` (server-stream.cpp:217 @d81235049):
    /// the should_stop closure on both the producer and any HTTP reader
    /// polls is_cancelled(), so flipping this is the only signal needed to
    /// unwind both sides.
    fn cancel(self: *Session) void {
        self.cancelled.store(true, .release);
    }

    /// Ports `stream_session`'s `is_cancelled` (server-stream.cpp:223 @d81235049).
    fn isCancelled(self: *Session) bool {
        return self.cancelled.load(.acquire);
    }
};

// -----------------------------------------------------------------------------
// stream_session_manager

/// Ports `stream_session_manager` (server-stream.cpp:31 @d81235049): every
/// live session, keyed by conversation id; one conversation, at most one
/// live session; a periodic GC evicts expired ones.
const Manager = struct {
    map_mu: Mutex = .{},
    sessions: ?libcxx.StringMap(?*Session) = null, // key: conversation_id
    gc_thread: ?std.Thread = null,
    running: bool = false,
    gc_wake_mu: Mutex = .{},
    gc_wake_cv: Cond = .{},

    fn map(self: *Manager) !*libcxx.StringMap(?*Session) {
        if (self.sessions == null) self.sessions = try libcxx.StringMap(?*Session).init(gpa);
        return &self.sessions.?;
    }

    /// Ports `stream_session_manager`'s `create_or_replace` (server-stream.cpp:235
    /// @d81235049): install a new session, evicting and cancelling any
    /// previous one.
    fn createOrReplace(self: *Manager, conversation_id: []const u8) !*Session {
        // evict any previous session on the same conv, this guarantees the invariant
        // "one conv = at most one live session" and propagates cancel to its producer
        var previous: ?*Session = null;
        const fresh = try Session.create(conversation_id, session_max_bytes);
        {
            self.map_mu.lock();
            defer self.map_mu.unlock();
            const r = (try self.map()).getOrPut(conversation_id, null) catch |e| {
                fresh.release();
                return e;
            };
            previous = r.value.*;
            r.value.* = fresh.acquire();
        }
        if (previous) |p| {
            p.cancel();
            p.finalize();
            p.release();
        }
        return fresh;
    }

    /// Ports `stream_session_manager`'s `get` (server-stream.cpp:257 @d81235049).
    ///
    /// Return: a reference the caller releases.
    fn get(self: *Manager, conversation_id: []const u8) ?*Session {
        self.map_mu.lock();
        defer self.map_mu.unlock();
        const m = self.map() catch return null;
        const v = m.get(conversation_id) orelse return null;
        const s = v.* orelse return null;
        return s.acquire();
    }

    /// Ports `stream_session_manager`'s `list_all` (server-stream.cpp:266
    /// @d81235049), in the map's order.
    ///
    /// Return: references the caller releases.
    fn listAll(self: *Manager, a: Allocator) ![]*Session {
        var out: std.ArrayList(*Session) = .empty;
        self.map_mu.lock();
        defer self.map_mu.unlock();
        var it = (try self.map()).iterator();
        while (it.next()) |e| {
            if (e.value.*) |s| try out.append(a, s.acquire());
        }
        return out.toOwnedSlice(a);
    }

    /// Ports `stream_session_manager`'s `evict` (server-stream.cpp:276 @d81235049).
    fn evict(self: *Manager, conversation_id: []const u8) void {
        var s: *Session = undefined;
        {
            self.map_mu.lock();
            defer self.map_mu.unlock();
            const m = self.map() catch return;
            const v = m.get(conversation_id) orelse return;
            s = v.* orelse return;
            v.* = null;
        }
        // finalize outside the map lock so any pending readers wake up and exit
        s.finalize();
        s.release();
    }

    /// Ports `stream_session_manager`'s `evict_and_cancel` (server-stream.cpp:291 @d81235049).
    fn evictAndCancel(self: *Manager, conversation_id: []const u8) !void {
        var s: *Session = undefined;
        {
            self.map_mu.lock();
            defer self.map_mu.unlock();
            const m = try self.map();
            const found: ?*?*Session = if (m.get(conversation_id)) |v| (if (v.* != null) v else null) else null;
            const v = found orelse {
                var live: std.ArrayList(u8) = .empty;
                defer live.deinit(gpa);
                var n: usize = 0;
                var it = m.iterator();
                while (it.next()) |e| {
                    if (e.value.* == null) continue;
                    if (live.items.len != 0) try live.appendSlice(gpa, ", ");
                    try live.appendSlice(gpa, e.key);
                    n += 1;
                }
                const id_z = try gpa.dupeSentinel(u8, conversation_id, 0);
                defer gpa.free(id_z);
                const live_z = try gpa.dupeSentinel(u8, live.items, 0);
                defer gpa.free(live_z);
                srv(.wrn, "evict_and_cancel", "stop on unknown stream session, conv_id=%s matched nothing, %zu live: [%s]\n", .{ id_z.ptr, n, live_z.ptr });
                return;
            };
            s = v.*.?;
            v.* = null;
        }
        // cancel first so the producer's on_complete() drain loop and any pending HTTP reader
        // observe is_cancelled() and stop pulling further output, then finalize to wake readers
        // blocked in read_from(). note: this does not interrupt the underlying generation itself,
        // which keeps running to its own natural stop condition (EOS/max_tokens)
        s.cancel();
        s.finalize();
        s.release();
    }

    /// Ports `stream_session_manager`'s `start_gc` (server-stream.cpp:317 @d81235049).
    fn startGc(self: *Manager) !void {
        {
            self.gc_wake_mu.lock();
            defer self.gc_wake_mu.unlock();
            if (self.running) {
                return;
            }
            self.running = true;
        }
        self.gc_thread = try std.Thread.spawn(.{}, gcLoop, .{self});
    }

    /// Ports `stream_session_manager`'s `stop_gc` (server-stream.cpp:328 @d81235049).
    fn stopGc(self: *Manager) void {
        var was_running: bool = undefined;
        {
            self.gc_wake_mu.lock();
            defer self.gc_wake_mu.unlock();
            was_running = self.running;
            self.running = false;
        }
        if (was_running) {
            self.gc_wake_cv.notifyAll();
            if (self.gc_thread) |t| {
                t.join();
                self.gc_thread = null;
            }
        }
        // finalize all live sessions so no reader ever hangs
        var snapshot: std.ArrayList(*Session) = .empty;
        defer snapshot.deinit(gpa);
        {
            self.map_mu.lock();
            defer self.map_mu.unlock();
            if (self.sessions) |*m| {
                var it = m.iterator();
                while (it.next()) |e| {
                    if (e.value.*) |s| snapshot.append(gpa, s) catch s.release();
                }
                m.deinit();
                self.sessions = null;
            }
        }
        for (snapshot.items) |s| {
            s.finalize();
            s.release();
        }
    }

    /// Ports `stream_session_manager`'s `gc_loop` (server-stream.cpp:356 @d81235049).
    fn gcLoop(self: *Manager) void {
        while (true) {
            {
                self.gc_wake_mu.lock();
                defer self.gc_wake_mu.unlock();
                const deadline = nowSeconds() + session_gc_interval_seconds;
                while (self.running and nowSeconds() < deadline) {
                    self.gc_wake_cv.waitFor(&self.gc_wake_mu, @intCast(session_gc_interval_seconds * std.time.ns_per_s));
                }
                if (!self.running) {
                    return;
                }
            }
            const cutoff = nowSeconds() - session_ttl_seconds;
            var to_drop: std.ArrayList(*Session) = .empty;
            defer to_drop.deinit(gpa);
            {
                self.map_mu.lock();
                defer self.map_mu.unlock();
                if (self.sessions) |*m| {
                    var it = m.iterator();
                    while (it.next()) |e| {
                        const s = e.value.* orelse continue;
                        const completed = s.completedAt();
                        if (completed != 0 and completed <= cutoff) {
                            to_drop.append(gpa, s) catch continue;
                            e.value.* = null;
                        }
                    }
                }
            }
            // finalize outside the map lock, idempotent if the session was already done
            for (to_drop.items) |s| {
                s.finalize();
                s.release();
            }
        }
    }
};

/// Ports `g_stream_sessions` (server-stream.cpp:65 @d81235049): process
/// wide, its lifecycle controlled by `llama_server` via start/stop.
var g_stream_sessions: Manager = .{};

/// Ports `server_stream_session_manager_start` (server-stream.cpp:67 @d81235049).
pub fn sessionManagerStart() !void {
    try g_stream_sessions.startGc();
}

/// Ports `server_stream_session_manager_stop` (server-stream.cpp:71 @d81235049).
pub fn sessionManagerStop() void {
    g_stream_sessions.stopGc();
}

// -----------------------------------------------------------------------------
// stream_pipe

/// Ports `stream_pipe_producer` (server-stream.h:33 @d81235049): writes
/// chunks into the ring buffer and owns the session's lifetime, finalizing
/// it on destruction.
pub const Producer = struct {
    session: *Session,

    /// Ports `stream_pipe_producer`'s `create` (server-stream.cpp:424 @d81235049).
    fn create(session: *Session) !*Producer {
        const self = try gpa.create(Producer);
        self.* = .{ .session = session };
        return self;
    }

    /// Ports the destructor of `stream_pipe_producer` (server-stream.cpp:416 @d81235049).
    pub fn destroy(self: *Producer) void {
        self.session.finalize();
        self.session.release();
        gpa.destroy(self);
    }

    /// Ports `stream_pipe_producer`'s `write` (server-stream.cpp:420 @d81235049).
    pub fn write(self: *Producer, data: []const u8) !bool {
        return self.session.append(data);
    }

    /// Ports `stream_pipe`'s `is_cancelled` (server-stream.cpp:406 @d81235049).
    pub fn isCancelled(self: *const Producer) bool {
        return self.session.isCancelled();
    }
};

/// Ports `make_error_response` (server-stream.cpp:445 @d81235049): the
/// standard error response in a new `server_http_res`.
fn makeErrorResponse(status: c_int, message: []const u8, ty: common.ErrorType) !*http.Res {
    const res = try http.Res.create(gpa);
    errdefer res.destroy();
    const a = res.allocator();
    var err = try common.formatErrorResponse(a, message, ty);
    res.status = common.jsonValue(&err, "code", c_int, status);
    res.content_type = "application/json; charset=utf-8";
    var outer = Value.emptyObject();
    try outer.set(a, "error", err);
    try res.setData(try common.safeJsonToStr(a, &outer));
    return res;
}

/// The response the GET handler returns: the base `server_http_res` and
/// what its `next` lambda captures -- the consumer pipe (a session
/// reference), the offset, and the request's `should_stop`.
const ReplayRes = struct {
    base: http.Res,
    session: *Session,
    offset: usize,
    should_stop: http.ShouldStop,

    const vtable: http.Res.VTable = .{ .on_complete = http.Res.noComplete, .destroy = destroyReplay };

    fn destroyReplay(res: *http.Res) void {
        const self: *ReplayRes = @fieldParentPtr("base", res);
        self.session.release();
        self.base.arena.deinit();
        gpa.destroy(self);
    }

    /// The `next` lambda of `server_stream_make_get_handler`
    /// (server-stream.cpp:454 @d81235049): exit each call after draining the
    /// available chunk so the chunked provider gets a chance to flush.
    fn next(ctx: ?*anyopaque, out: *std.ArrayList(u8), a: Allocator) anyerror!bool {
        const self: *ReplayRes = @ptrCast(@alignCast(ctx.?));
        const Sink = struct {
            out: *std.ArrayList(u8),
            a: Allocator,
            offset: *usize,
            got_any: bool = false,
            fn call(s: *@This(), d: []const u8) !bool {
                try s.out.appendSlice(s.a, d);
                s.offset.* += d.len;
                s.got_any = true;
                return false;
            }
        };
        var sink: Sink = .{ .out = out, .a = a, .offset = &self.offset };
        _ = try self.session.readFrom(self.offset, &sink, self.should_stop);
        return sink.got_any;
    }
};

/// The handler `server_stream_make_get_handler` (server-stream.cpp:454
/// @d81235049) makes: `GET /v1/stream?conv_id=<id>&from=N` replays the
/// buffered SSE bytes, then blocks for live ones until the session ends.
fn getHandler(_: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
    const conv_id = req.getParam("conv_id", "");
    if (conv_id.len == 0) {
        return makeErrorResponse(400, "Missing conversation id in path", .invalid_request);
    }
    const session = g_stream_sessions.get(conv_id) orelse {
        return makeErrorResponse(404, "Stream not found or expired", .not_found);
    };
    var keep = false;
    defer if (!keep) session.release();
    var from: usize = 0;
    const from_str = req.getParam("from", "");
    if (from_str.len != 0) {
        from = stoull(from_str) orelse {
            return makeErrorResponse(400, "Invalid 'from' offset", .invalid_request);
        };
    }
    if (from < session.droppedPrefix()) {
        return makeErrorResponse(400, "Stream offset lost, please restart", .invalid_request);
    }
    const self = try gpa.create(ReplayRes);
    self.* = .{
        .base = .{ .arena = std.heap.ArenaAllocator.init(gpa), .vtable = &ReplayRes.vtable },
        .session = session,
        .offset = from,
        .should_stop = req.should_stop,
    };
    keep = true;
    self.base.status = 200;
    self.base.content_type = "text/event-stream";
    self.base.next = .{ .ctx = self, .func = ReplayRes.next };
    return &self.base;
}

/// `std::stoull`: leading whitespace, an optional sign, digits in base 10;
/// null where it throws (no digits, or out of range).
fn stoull(s: []const u8) ?usize {
    var i: usize = 0;
    while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
    var neg = false;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) {
        neg = s[i] == '-';
        i += 1;
    }
    const start = i;
    var v: u64 = 0;
    while (i < s.len and std.ascii.isDigit(s[i])) : (i += 1) {
        v = std.math.mul(u64, v, 10) catch return null;
        v = std.math.add(u64, v, s[i] - '0') catch return null;
    }
    if (i == start) return null;
    // strtoull negates in unsigned arithmetic
    return if (neg) @as(usize, 0) -% v else v;
}

/// Ports `server_stream_make_get_handler` (server-stream.cpp:454 @d81235049).
pub fn makeGetHandler() http.Handler {
    return .{ .ctx = null, .func = getHandler };
}

/// The handler `server_stream_make_lookup_handler` (server-stream.cpp:503
/// @d81235049) makes: `POST /v1/streams/lookup` answers only for the ids
/// asked about, each matching its exact key and any `<id>::<model>`.
fn lookupHandler(_: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
    const res = try http.Res.create(gpa);
    errdefer res.destroy();
    const a = res.allocator();
    var requested: std.ArrayList([]const u8) = .empty;
    parse: {
        var body = json.parse(a, req.body) catch |e| {
            if (e == error.OutOfMemory) return e;
            res.status = 400;
            res.content_type = "application/json; charset=utf-8";
            var err = common.Obj.init(a);
            const msg = try std.mem.concat(a, u8, &.{ "invalid body: ", json.what() });
            try err.put("message", msg);
            try err.put("type", "invalid_request_error");
            var outer = common.Obj.init(a);
            try outer.put("error", err.done());
            var v = outer.done();
            try res.setData(try common.safeJsonToStr(a, &v));
            return res;
        };
        if (body.contains("conversation_ids") and (try body.index(a, "conversation_ids")).isArray()) {
            for ((try body.index(a, "conversation_ids")).array.items) |*v| {
                if (v.isString()) {
                    const id = v.string;
                    if (id.len != 0) {
                        try requested.append(a, id);
                    }
                }
            }
        }
        break :parse;
    }

    var sessions: std.ArrayList(*Session) = .empty;
    var all: []*Session = &.{};
    defer for (all) |s| s.release();
    if (requested.items.len != 0) {
        all = try g_stream_sessions.listAll(gpa);
        for (requested.items) |rid| {
            const with_sep = try std.mem.concat(a, u8, &.{ rid, "::" });
            for (all) |s| {
                if (std.mem.eql(u8, s.conversation_id, rid) or std.mem.startsWith(u8, s.conversation_id, with_sep)) {
                    try sessions.append(a, s);
                }
            }
        }
    }

    var arr = Value.emptyArray();
    for (sessions.items) |s| {
        var o = common.Obj.init(a);
        try o.put("conversation_id", s.conversation_id);
        try o.put("is_done", s.isDone());
        try o.put("total_bytes", s.totalSize());
        try o.put("started_at", s.started_ts);
        try o.put("completed_at", s.completedAt());
        try arr.pushBack(a, o.done());
    }
    res.status = 200;
    res.content_type = "application/json; charset=utf-8";
    try res.setData(try common.safeJsonToStr(a, &arr));
    return res;
}

/// Ports `server_stream_make_lookup_handler` (server-stream.cpp:503 @d81235049).
pub fn makeLookupHandler() http.Handler {
    return .{ .ctx = null, .func = lookupHandler };
}

/// The handler `server_stream_make_delete_handler` (server-stream.cpp:561
/// @d81235049) makes: `DELETE /v1/stream?conv_id=<id>` is the explicit
/// user Stop; idempotent, 204 even if the session was already gone.
fn deleteHandler(_: ?*anyopaque, req: *const http.Req) anyerror!*http.Res {
    const conv_id = req.getParam("conv_id", "");
    if (conv_id.len == 0) {
        return makeErrorResponse(400, "Missing conversation id in path", .invalid_request);
    }
    {
        const z = try gpa.dupeSentinel(u8, conv_id, 0);
        defer gpa.free(z);
        srv(.trc, "operator()", "DELETE /v1/stream conv_id=%s -> evict_and_cancel\n", .{z.ptr});
    }
    try g_stream_sessions.evictAndCancel(conv_id);
    const res = try http.Res.create(gpa);
    res.status = 204;
    res.content_type = "application/json";
    return res;
}

/// Ports `server_stream_make_delete_handler` (server-stream.cpp:561 @d81235049).
pub fn makeDeleteHandler() http.Handler {
    return .{ .ctx = null, .func = deleteHandler };
}

/// Ports `server_stream_conv_id_from_headers` (server-stream.cpp:578
/// @d81235049): the `X-Conversation-Id` header's value, its name compared
/// without case; "" when absent.
pub fn convIdFromHeaders(headers: *const http.StringMap) []const u8 {
    // case-insensitive scan for x-conversation-id
    const target = "x-conversation-id";
    for (headers.entries.items) |e| {
        if (e.key.len != target.len) continue;
        if (std.ascii.eqlIgnoreCase(e.key, target)) {
            return e.value;
        }
    }
    return "";
}

/// Ports `server_stream_create_spipe` (server-stream.cpp:597 @d81235049).
fn createSpipe(headers: *const http.StringMap) !?*Producer {
    const conversation_id = convIdFromHeaders(headers);
    {
        const z = try gpa.dupeSentinel(u8, conversation_id, 0);
        defer gpa.free(z);
        srv(.trc, "server_stream_create_spipe", "conv_id=%s (empty=%d)\n", .{ z.ptr, @as(c_int, @intFromBool(conversation_id.len == 0)) });
    }
    if (conversation_id.len == 0) {
        return null;
    }
    const session = try g_stream_sessions.createOrReplace(conversation_id);
    errdefer session.release();
    return try Producer.create(session);
}

// -----------------------------------------------------------------------------
// server_res_spipe

/// Ports `server_res_spipe` (server-stream.h:62 @d81235049): the tee-style
/// pipe for stream replay, as the members a response embeds beside its
/// `http.Res`. With a producer, the stream survives a client disconnect:
/// its output is forwarded there for a later reader.
pub const Spipe = struct {
    spipe: ?*Producer = null,
    // if spipe is set, use this next_orig to implement tee-style pipe
    next_orig: ?http.Res.Next = null,
    req: ?*const http.Req = null,
    // set once next_orig reports no more data, so on_complete() doesn't re-drain a finished stream
    next_finished: bool = false,

    /// The destructor's part: the producer finalizes the session.
    pub fn deinit(self: *Spipe) void {
        if (self.spipe) |p| p.destroy();
        self.spipe = null;
    }

    /// Ports `server_res_spipe`'s `set_req` (server-stream.cpp:611
    /// @d81235049): optionally attach a producer when `X-Conversation-Id`
    /// is present.
    pub fn setReq(self: *Spipe, req: *const http.Req) !void {
        self.req = req;
        if (self.spipe) |p| p.destroy();
        self.spipe = null;
        self.spipe = try createSpipe(&req.headers);
    }

    /// Ports `server_res_spipe`'s `conn_alive` (server-stream.cpp:617 @d81235049).
    pub fn connAlive(self: *const Spipe) bool {
        common.ggmlAssert(self.req != null, "server-stream.cpp", 618, "req != nullptr");
        return !self.req.?.should_stop.call();
    }

    /// Ports `server_res_spipe`'s `should_stop` (server-stream.cpp:622 @d81235049).
    pub fn shouldStop(self: *const Spipe) bool {
        if (self.spipe) |p| {
            // note: if DELETE /v1/stream is called for this conv, is_cancelled() will be true
            return p.isCancelled();
        } else {
            return !self.connAlive();
        }
    }

    /// Ports `server_res_spipe`'s `on_complete` (server-stream.cpp:631 @d81235049).
    pub fn onComplete(self: *Spipe) void {
        const p = self.spipe orelse return;
        if (self.next_finished) {
            return;
        }
        // an empty next_orig means set_next() never ran: the request failed before streaming
        // started, typically a params validation throw. evict the session installed by set_req()
        // so the failed request leaves nothing behind for discovery or replay
        const next_orig = self.next_orig orelse {
            g_stream_sessions.evict(convIdFromHeaders(&self.req.?.headers));
            return;
        };
        var chunk: std.ArrayList(u8) = .empty;
        defer chunk.deinit(gpa);
        while (!p.isCancelled()) {
            chunk.clearRetainingCapacity();
            const has_next = next_orig.func(next_orig.ctx, &chunk, gpa) catch false;
            if (chunk.items.len != 0) {
                _ = p.write(chunk.items) catch {};
            }
            if (!has_next) {
                break;
            }
        }
    }

    /// Ports `server_res_spipe`'s `set_next` (server-stream.cpp:655
    /// @d81235049): `res.next` becomes the tee over `next_fn`.
    pub fn setNext(self: *Spipe, res: *http.Res, next_fn: http.Res.Next) void {
        self.next_orig = next_fn;
        res.next = .{ .ctx = self, .func = tee };
    }

    fn tee(ctx: ?*anyopaque, out: *std.ArrayList(u8), a: Allocator) anyerror!bool {
        const self: *Spipe = @ptrCast(@alignCast(ctx.?));
        const next_orig = self.next_orig.?;
        const has_next = try next_orig.func(next_orig.ctx, out, a);
        if (self.spipe) |p| {
            // if spipe is set, tee-style pipe input to both HTTP and spipe
            _ = try p.write(out.items);
        }
        if (!has_next) {
            self.next_finished = true;
        }
        return has_next;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "stoull reads a leading number and refuses none" {
    try std.testing.expectEqual(@as(?usize, 12), stoull("12abc"));
    try std.testing.expectEqual(@as(?usize, 7), stoull("  +7"));
    try std.testing.expectEqual(@as(?usize, null), stoull("x"));
    try std.testing.expectEqual(@as(?usize, null), stoull(""));
}
