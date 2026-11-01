//! cpp-httplib's `ThreadPool`: the task queue a server hands each accepted
//! connection to.
//!
//! # Provenance
//!
//! Ports `ThreadPool` from `llama.cpp/vendor/cpp-httplib/httplib.cpp` and
//! its declaration in `llama.cpp/vendor/cpp-httplib/httplib.h` (cpp-httplib
//! 0.56.0, vendored at v0.4.1, `b29c606e2`). Each declaration names what it
//! ports and the line it began at.
//!
//! # pthreads, as the C's are
//!
//! The mutex and condition variable are pthreads', as `CLAUDE.md` asks of
//! ported threading code: `std::mutex` and `std::condition_variable` are
//! those on every POSIX libc++, and Zig's own take an `Io` this layer has
//! none of. A dynamic worker's idle wait is `pthread_cond_timedwait`
//! against the realtime clock, which is what libc++'s `wait_for` reaches on
//! Darwin and Linux alike.

const std = @import("std");
const detail = @import("detail.zig");

extern "c" fn pthread_equal(t1: std.c.pthread_t, t2: std.c.pthread_t) c_int;

/// One `std::function<void()>` the pool runs: a function and its argument.
pub const Task = struct {
    ctx: *anyopaque,
    arg: usize,
    run: *const fn (ctx: *anyopaque, arg: usize) void,
};

/// A worker's thread and its handle, for `move_to_finished` to find by
/// `pthread_self`.
const Worker = struct {
    thread: std.Thread,
    id: std.c.pthread_t,
};

/// Ports `ThreadPool` (httplib.h:1957 @b29c606e2): `base` threads always
/// alive, and up to `max - base` more spawned when a job arrives with none
/// idle, each exiting after `CPPHTTPLIB_THREAD_POOL_IDLE_TIMEOUT` seconds
/// without work.
pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    base_thread_count: usize,
    max_thread_count: usize,
    max_queued_requests: usize,
    idle_timeout_sec: i64,
    idle_thread_count: usize = 0,
    shutdown_: bool = false,
    jobs: std.ArrayList(Task) = .empty,
    jobs_head: usize = 0,
    threads: std.ArrayList(Worker) = .empty,
    dynamic_threads: std.ArrayList(Worker) = .empty,
    finished_threads: std.ArrayList(Worker) = .empty,
    cond: std.c.pthread_cond_t = .{},
    mutex: std.c.pthread_mutex_t = .{},
    /// Held by a dynamic worker from before its spawn until it has
    /// recorded its own handle, so `move_to_finished` always finds it.
    started: std.c.pthread_mutex_t = .{},

    /// Ports `ThreadPool` (httplib.cpp:7129 @b29c606e2), the constructor:
    /// starts the `n` base threads. The C throws when `max_n` is below `n`;
    /// this returns `error.InvalidArgument`.
    ///
    /// Parameters:
    /// - `allocator`: the pool's lists; kept.
    /// - `n`: base threads.
    /// - `max_n`: the most threads at once, 0 for `n`.
    ///
    /// Return: the pool, heap-allocated since its workers point at it.
    pub fn create(allocator: std.mem.Allocator, n: usize, max_n: usize) !*ThreadPool {
        if (max_n != 0 and max_n < n) return error.InvalidArgument;
        const self = try allocator.create(ThreadPool);
        self.* = .{
            .allocator = allocator,
            .base_thread_count = n,
            .max_thread_count = if (max_n == 0) n else max_n,
            .max_queued_requests = 0,
            .idle_timeout_sec = detail.thread_pool_idle_timeout,
        };
        try self.threads.ensureTotalCapacity(allocator, n);
        for (0..n) |_| {
            const t = std.Thread.spawn(.{}, worker, .{ self, false }) catch |e| {
                _ = std.c.pthread_mutex_lock(&self.mutex);
                self.shutdown_ = true;
                _ = std.c.pthread_mutex_unlock(&self.mutex);
                _ = std.c.pthread_cond_broadcast(&self.cond);
                for (self.threads.items) |w| w.thread.join();
                return e;
            };
            self.threads.appendAssumeCapacity(.{ .thread = t, .id = t.getHandle() });
        }
        return self;
    }

    /// Frees the pool after `shutdown`.
    pub fn destroy(self: *ThreadPool) void {
        const a = self.allocator;
        self.jobs.deinit(a);
        self.threads.deinit(a);
        self.dynamic_threads.deinit(a);
        self.finished_threads.deinit(a);
        a.destroy(self);
    }

    /// Ports `enqueue` (httplib.cpp:7167 @b29c606e2), a member of
    /// `ThreadPool`: queues `fn_`, spawning a dynamic thread when none is
    /// idle and the pool is under its maximum.
    ///
    /// Return: false once shut down.
    pub fn enqueue(self: *ThreadPool, fn_: Task) bool {
        {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.mutex);
            if (self.shutdown_) return false;
            if (self.max_queued_requests > 0 and self.jobs.items.len - self.jobs_head >= self.max_queued_requests) return false;
            self.jobs.append(self.allocator, fn_) catch return false;
            if (self.idle_thread_count == 0 and self.threads.items.len + self.dynamic_threads.items.len < self.max_thread_count) {
                self.cleanupFinishedThreads();
                self.dynamic_threads.ensureUnusedCapacity(self.allocator, 1) catch return true;
                _ = std.c.pthread_mutex_lock(&self.started);
                if (std.Thread.spawn(.{}, worker, .{ self, true })) |t| {
                    self.dynamic_threads.appendAssumeCapacity(.{ .thread = t, .id = t.getHandle() });
                } else |_| {}
                _ = std.c.pthread_mutex_unlock(&self.started);
            }
        }
        _ = std.c.pthread_cond_signal(&self.cond);
        return true;
    }

    /// Ports `shutdown` (httplib.cpp:7188 @b29c606e2), a member of
    /// `ThreadPool`: lets the queue drain, then joins every thread.
    pub fn shutdown(self: *ThreadPool) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        self.shutdown_ = true;
        _ = std.c.pthread_mutex_unlock(&self.mutex);
        _ = std.c.pthread_cond_broadcast(&self.cond);
        for (self.threads.items) |w| w.thread.join();
        self.threads.clearRetainingCapacity();

        _ = std.c.pthread_mutex_lock(&self.mutex);
        var remaining = self.dynamic_threads;
        self.dynamic_threads = .empty;
        _ = std.c.pthread_mutex_unlock(&self.mutex);
        for (remaining.items) |w| w.thread.join();
        remaining.deinit(self.allocator);

        _ = std.c.pthread_mutex_lock(&self.mutex);
        self.cleanupFinishedThreads();
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    /// Ports `move_to_finished` (httplib.cpp:7215 @b29c606e2), a member of
    /// `ThreadPool`; `mutex` is held.
    fn moveToFinished(self: *ThreadPool, id: std.c.pthread_t) void {
        for (self.dynamic_threads.items, 0..) |w, i| {
            if (pthread_equal(w.id, id) != 0) {
                self.finished_threads.append(self.allocator, w) catch {
                    // the C's push_back throws here; detach rather than leak a joinable
                    w.thread.detach();
                    _ = self.dynamic_threads.orderedRemove(i);
                    return;
                };
                _ = self.dynamic_threads.orderedRemove(i);
                return;
            }
        }
    }

    /// Ports `cleanup_finished_threads` (httplib.cpp:7226 @b29c606e2), a
    /// member of `ThreadPool`; `mutex` is held.
    fn cleanupFinishedThreads(self: *ThreadPool) void {
        for (self.finished_threads.items) |w| w.thread.join();
        self.finished_threads.clearRetainingCapacity();
    }

    /// `jobs_.front()` and `pop_front()`.
    fn popJob(self: *ThreadPool) Task {
        const t = self.jobs.items[self.jobs_head];
        self.jobs_head += 1;
        if (self.jobs_head == self.jobs.items.len) {
            self.jobs.clearRetainingCapacity();
            self.jobs_head = 0;
        }
        return t;
    }

    fn hasJobs(self: *const ThreadPool) bool {
        return self.jobs_head < self.jobs.items.len;
    }

    /// Ports `worker` (httplib.cpp:7234 @b29c606e2), a member of
    /// `ThreadPool`: runs jobs until shut down with none left; a dynamic
    /// worker also leaves after its idle timeout.
    fn worker(self: *ThreadPool, is_dynamic: bool) void {
        if (is_dynamic) {
            // wait for `enqueue` to have recorded this thread
            _ = std.c.pthread_mutex_lock(&self.started);
            _ = std.c.pthread_mutex_unlock(&self.started);
        }
        while (true) {
            var fn_: Task = undefined;
            {
                _ = std.c.pthread_mutex_lock(&self.mutex);
                defer _ = std.c.pthread_mutex_unlock(&self.mutex);
                self.idle_thread_count += 1;
                if (is_dynamic) {
                    var deadline: std.c.timespec = undefined;
                    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &deadline);
                    deadline.sec += @intCast(self.idle_timeout_sec);
                    var has_work = self.hasJobs() or self.shutdown_;
                    while (!has_work) {
                        const rc = std.c.pthread_cond_timedwait(&self.cond, &self.mutex, &deadline);
                        has_work = self.hasJobs() or self.shutdown_;
                        if (rc == .TIMEDOUT) break;
                    }
                    if (!has_work) {
                        self.idle_thread_count -= 1;
                        self.moveToFinished(std.c.pthread_self());
                        break;
                    }
                } else {
                    while (!(self.hasJobs() or self.shutdown_)) _ = std.c.pthread_cond_wait(&self.cond, &self.mutex);
                }
                self.idle_thread_count -= 1;
                if (self.shutdown_ and !self.hasJobs()) break;
                fn_ = self.popJob();
            }
            fn_.run(fn_.ctx, fn_.arg);
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the pool runs every job it accepted before shutting down" {
    const Counter = struct {
        n: std.atomic.Value(usize) = .init(0),
        fn run(ctx: *anyopaque, arg: usize) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = self.n.fetchAdd(arg, .seq_cst);
        }
    };
    var c: Counter = .{};
    const pool = try ThreadPool.create(std.testing.allocator, 2, 6);
    for (0..20) |_| try std.testing.expect(pool.enqueue(.{ .ctx = &c, .arg = 1, .run = Counter.run }));
    pool.shutdown();
    pool.destroy();
    try std.testing.expectEqual(@as(usize, 20), c.n.load(.seq_cst));
}
