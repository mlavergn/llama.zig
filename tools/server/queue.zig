//! The server's two queues: tasks from the HTTP threads to the decode loop,
//! and results back; and the reader an HTTP handler waits on.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-queue.cpp` and
//! `llama.cpp/tools/server/server-queue.h` (llama.cpp v0.9.0, `cd959bae9`):
//! `server_queue`, `server_response` and `server_response_reader`. Each
//! declaration names what it ports and the line it began at.
//!
//! # Generic over the task and result
//!
//! The queues are written against the parts of `server_task` and
//! `server_task_result` they touch, so they are generic over both: `Task`
//! has `id`, `id_target`, `id_slot`, `index`, `type` (with `.cancel` and
//! `.metrics`), `child_tasks`, `isParent()` and `initType(type)`; the
//! state has `Session` and `init(gpa, ?*const Session)`; a result is a pointer to a type with `id`, `index`,
//! `isError()`, `isStop()`, `update(*State)`, `clone(gpa)` and `destroy(gpa)`
//! -- the C++'s `std::unique_ptr<server_task_result>` and its virtuals.
//!
//! # Threads
//!
//! As `ggml-cpu.c`'s port does, the mutexes and condition variables are
//! pthreads': a handler reaches these from httplib's worker threads, which
//! carry no `std.Io`. A C++ exception thrown by a callback is an error
//! here, carried across `yield_to_queue` the way the C++ carries an
//! `std::exception_ptr`.
//!
//! # Hash-set order
//!
//! `std::unordered_set<int>` is walked by `server_response_reader::stop`,
//! which posts one cancel task per id and logs each -- so the cancel tasks'
//! order and the log's are libc++'s bucket order. `IdSet` reproduces it
//! over `libcxx.I32Map`.

const std = @import("std");
const llama = @import("llama");
const tools = @import("tools_common");

const c = @import("tools_h");
const log = tools.log;
const I32Map = llama.libcxx.I32Map;

extern fn abort() noreturn;

fn timeMs() i64 {
    return c.ggml_time_ms();
}

/// `GGML_ASSERT` with the C's message.
fn ggmlAssert(ok: bool, comptime line: c_int, comptime cond: [*:0]const u8) void {
    if (!ok) {
        log.err("%s:%d: GGML_ASSERT(%s) failed\n", .{ "server-queue.cpp", line, cond });
        abort();
    }
}

/// `std::unordered_set<int>` as libc++ lays it out (see the file header).
pub const IdSet = struct {
    map: I32Map(void),

    /// `std::unordered_set<int>()`.
    pub fn init(gpa: std.mem.Allocator) !IdSet {
        return .{ .map = try I32Map(void).init(gpa) };
    }

    /// `std::unordered_set<int>(n)`: an empty set with `n` buckets asked for.
    pub fn initCapacity(gpa: std.mem.Allocator, n: usize) !IdSet {
        var s = try init(gpa);
        errdefer s.deinit();
        try s.map.rehashTo(n);
        return s;
    }

    pub fn deinit(self: *IdSet) void {
        self.map.deinit();
    }

    /// `insert(id)`.
    pub fn insert(self: *IdSet, id: i32) !void {
        try self.map.put(id, {});
    }

    /// `find(id) != end()`.
    pub fn contains(self: *const IdSet, id: i32) bool {
        return self.map.get(id) != null;
    }

    /// `size()`.
    pub fn count(self: *const IdSet) usize {
        return self.map.count();
    }

    /// Return: the ids in libc++'s iteration order, owned by the caller.
    pub fn ordered(self: *const IdSet, gpa: std.mem.Allocator) ![]i32 {
        const out = try gpa.alloc(i32, self.count());
        var it = self.map.iterator();
        var i: usize = 0;
        while (it.next()) |e| : (i += 1) out[i] = e.key;
        return out;
    }
};

/// A pthread mutex with the C++'s `std::mutex` semantics.
const Mutex = struct {
    m: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    fn lock(self: *Mutex) void {
        _ = std.c.pthread_mutex_lock(&self.m);
    }
    fn unlock(self: *Mutex) void {
        _ = std.c.pthread_mutex_unlock(&self.m);
    }
};

/// A pthread condition variable with the C++'s `std::condition_variable`
/// operations the queues use.
const Cond = struct {
    c: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,

    fn wait(self: *Cond, m: *Mutex) void {
        _ = std.c.pthread_cond_wait(&self.c, &m.m);
    }

    /// `wait_for(lock, seconds)`. Return: false on timeout.
    fn waitFor(self: *Cond, m: *Mutex, ns: u64) bool {
        return self.waitUntil(m, deadlineIn(ns));
    }

    /// `wait_until(lock, deadline)`.
    ///
    /// Return: false on a timeout.
    fn waitUntil(self: *Cond, m: *Mutex, deadline: std.c.timespec) bool {
        const rc = std.c.pthread_cond_timedwait(&self.c, &m.m, &deadline);
        return rc != .TIMEDOUT;
    }

    /// The moment `ns` nanoseconds from now, as `pthread_cond_timedwait` reads it.
    fn deadlineIn(ns: u64) std.c.timespec {
        var now: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &now);
        const total: u64 = @as(u64, @intCast(now.nsec)) + ns % std.time.ns_per_s;
        return .{
            .sec = now.sec + @as(@TypeOf(now.sec), @intCast(ns / std.time.ns_per_s + total / std.time.ns_per_s)),
            .nsec = @intCast(total % std.time.ns_per_s),
        };
    }

    fn notifyOne(self: *Cond) void {
        _ = std.c.pthread_cond_signal(&self.c);
    }
    fn notifyAll(self: *Cond) void {
        _ = std.c.pthread_cond_broadcast(&self.c);
    }
};

/// Ports `task_resets_idle_timer` (server-queue.cpp:24 @cd959bae9).
fn taskResetsIdleTimer(t: anytype) bool {
    return t != .metrics;
}

/// Ports `server_queue` (server-queue.h:15 @cd959bae9).
pub fn Queue(comptime Task: type) type {
    return struct {
        const Self = @This();

        /// `callback_new_task`: takes the task when it returns true, and
        /// leaves it with the caller when it declines it (false).
        pub const NewTaskFn = *const fn (ctx: *anyopaque, task: *Task, is_yielding: bool) anyerror!bool;
        pub const UpdateSlotsFn = *const fn (ctx: *anyopaque) anyerror!void;
        pub const SleepingFn = *const fn (ctx: *anyopaque, sleeping: bool) void;

        gpa: std.mem.Allocator,

        id: i32 = 0,
        running: bool = false,
        sleeping: bool = false,
        req_stop_sleeping: bool = false,
        time_last_task: i64 = 0,

        queue_tasks: std.ArrayList(Task) = .empty,
        queue_tasks_deferred: std.ArrayList(Task) = .empty,
        queue_tasks_unhandled: std.ArrayList(Task) = .empty,

        mutex_tasks: Mutex = .{},
        condition_tasks: Cond = .{},

        worker: struct {
            thread: ?std.Thread = null,
            cv: Cond = .{},
            err: ?anyerror = null,
            stop: bool = false,
            busy: bool = false,
            yielding: bool = false,
        } = .{},

        cb_ctx: *anyopaque = undefined,
        callback_new_task: ?NewTaskFn = null,
        callback_update_slots: ?UpdateSlotsFn = null,
        callback_sleeping_state: std.ArrayList(struct { ctx: *anyopaque, f: SleepingFn }) = .empty,

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        /// Ports the destructor of `server_queue` (server-queue.h:50 @cd959bae9), and frees the queued tasks.
        pub fn deinit(self: *Self) void {
            self.workerStop();
            for ([_]*std.ArrayList(Task){ &self.queue_tasks, &self.queue_tasks_deferred, &self.queue_tasks_unhandled }) |q| {
                for (q.items) |*t| t.deinit(self.gpa);
                q.deinit(self.gpa);
            }
            self.callback_sleeping_state.deinit(self.gpa);
        }

        fn push(self: *Self, q: *std.ArrayList(Task), task: Task, front: bool) !void {
            if (front) try q.insert(self.gpa, 0, task) else try q.append(self.gpa, task);
        }

        /// Ports `server_queue`'s `post` (server-queue.cpp:28 @cd959bae9): one task,
        /// whose id is already set.
        ///
        /// Return: the task's id.
        pub fn post(self: *Self, task: Task, front: bool) !i32 {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            ggmlAssert(task.id != -1, 30, "task.id != -1");
            // if this is cancel task make sure to clean up pending tasks
            if (task.type == .cancel) self.cleanupPendingTask(task.id_target);
            const task_id = task.id;
            const reset_timer = taskResetsIdleTimer(task.type);
            log.dbg("que  %12.*s: new task, id = %d, front = %d\n", .{ @as(c_int, 12), "post", task_id, @as(c_int, @intFromBool(front)) });
            try self.push(&self.queue_tasks, task, front);
            if (reset_timer) self.time_last_task = timeMs();
            self.condition_tasks.notifyOne();
            return task_id;
        }

        /// Ports `server_queue`'s vector `post` (server-queue.cpp:50 @cd959bae9):
        /// gives each task without an id the next one, in order.
        pub fn postMany(self: *Self, tasks: []Task, front: bool) !void {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            var reset_timer = false;
            for (tasks) |*task| {
                if (task.id == -1) {
                    task.id = self.id;
                    self.id += 1;
                }
                // if this is cancel task make sure to clean up pending tasks
                if (task.type == .cancel) self.cleanupPendingTask(task.id_target);
                reset_timer = reset_timer or taskResetsIdleTimer(task.type);
                log.dbg("que  %12.*s: new task, id = %d/%d, front = %d\n", .{ @as(c_int, 12), "post", task.id, @as(c_int, @intCast(tasks.len)), @as(c_int, @intFromBool(front)) });
                try self.push(&self.queue_tasks, task.*, front);
            }
            if (reset_timer) self.time_last_task = timeMs();
            self.condition_tasks.notifyOne();
        }

        /// Ports `server_queue`'s `defer` (server-queue.cpp:76 @cd959bae9).
        pub fn deferTask(self: *Self, task: Task) !void {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            log.dbg("que  %12.*s: defer task, id = %d\n", .{ @as(c_int, 12), "defer", task.id });
            try self.queue_tasks_deferred.append(self.gpa, task);
            self.time_last_task = timeMs();
            self.condition_tasks.notifyOne();
        }

        /// Ports `server_queue`'s `get_new_id` (server-queue.cpp:84 @cd959bae9).
        pub fn getNewId(self: *Self) i32 {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            const new_id = self.id;
            self.id += 1;
            return new_id;
        }

        /// Ports `server_queue`'s `pop_deferred_task` (server-queue.cpp:90 @cd959bae9).
        pub fn popDeferredTask(self: *Self, id_slot: i32) !void {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            if (self.queue_tasks_deferred.items.len != 0) {
                // try to find a task that uses the specified slot
                var found = false;
                for (self.queue_tasks_deferred.items, 0..) |t, i| {
                    if (t.id_slot == id_slot) {
                        log.dbg("que  %12.*s: pop deferred task (use slot %d), id_task = %d\n", .{ @as(c_int, 12), "pop_deferred_task", id_slot, t.id });
                        try self.queue_tasks.insert(self.gpa, 0, t);
                        _ = self.queue_tasks_deferred.orderedRemove(i);
                        found = true;
                        break;
                    }
                }
                // if not tasks found using the slot, just pop the first deferred task (default behavior)
                if (!found) {
                    log.dbg("que  %12.*s: pop deferred task, id_task = %d\n", .{ @as(c_int, 12), "pop_deferred_task", self.queue_tasks_deferred.items[0].id });
                    try self.queue_tasks.insert(self.gpa, 0, self.queue_tasks_deferred.items[0]);
                    _ = self.queue_tasks_deferred.orderedRemove(0);
                }
            }
            self.time_last_task = timeMs();
            self.condition_tasks.notifyOne();
        }

        /// Ports `server_queue`'s `wait_until_no_sleep` (server-queue.cpp:115 @cd959bae9).
        pub fn waitUntilNoSleep(self: *Self) void {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            if (!self.sleeping) return;
            if (!self.req_stop_sleeping) {
                log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "wait_until_no_sleep", "requesting to stop sleeping\n" });
                self.req_stop_sleeping = true;
                self.condition_tasks.notifyOne(); // only main thread is waiting on this
            }
            log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "wait_until_no_sleep", "waiting until no sleep\n" });
            while (self.sleeping) self.condition_tasks.wait(&self.mutex_tasks);
        }

        /// Ports `server_queue`'s `is_sleeping` (server-queue.h:72 @cd959bae9).
        pub fn isSleeping(self: *Self) bool {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            return self.sleeping;
        }

        /// Ports `server_queue`'s `terminate` (server-queue.cpp:132 @cd959bae9).
        pub fn terminate(self: *Self) void {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            self.running = false;
            self.condition_tasks.notifyAll();
        }

        /// Ports `server_queue`'s `queue_tasks_deferred_size` (server-queue.h:107 @cd959bae9).
        pub fn queueTasksDeferredSize(self: *Self) usize {
            self.mutex_tasks.lock();
            defer self.mutex_tasks.unlock();
            return self.queue_tasks_deferred.items.len;
        }

        /// Ports `on_new_task`, `on_update_slots` and
        /// `on_sleeping_state` (server-queue.h:121, 126, 135 @cd959bae9):
        /// one context for the first two, which the server binds to the
        /// same `server_context`.
        pub fn onCallbacks(self: *Self, ctx: *anyopaque, new_task: NewTaskFn, update_slots: UpdateSlotsFn) void {
            self.cb_ctx = ctx;
            self.callback_new_task = new_task;
            self.callback_update_slots = update_slots;
        }

        pub fn onSleepingState(self: *Self, ctx: *anyopaque, f: SleepingFn) !void {
            try self.callback_sleeping_state.append(self.gpa, .{ .ctx = ctx, .f = f });
        }

        /// Ports `server_queue`'s `process_new_tasks` (server-queue.cpp:138 @cd959bae9).
        ///
        /// Return: true when the queue was terminated.
        fn processNewTasks(self: *Self, is_yielding: bool) !bool {
            while (true) {
                self.mutex_tasks.lock();
                if (!self.running) {
                    log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "process_new_tasks", "terminate\n" });
                    self.mutex_tasks.unlock();
                    return true;
                }
                if (self.queue_tasks.items.len == 0) {
                    self.mutex_tasks.unlock();
                    return false;
                }
                var task = self.queue_tasks.orderedRemove(0);
                self.mutex_tasks.unlock();

                log.dbg("que  %12.*s: processing task, id = %d\n", .{ @as(c_int, 12), "process_new_tasks", task.id });
                if (!try self.callback_new_task.?(self.cb_ctx, &task, is_yielding)) {
                    // set it aside, do not put it back in the queue, else we offer it again in a loop
                    ggmlAssert(is_yielding, 155, "is_yielding && \"a task can only be declined while yielding\"");
                    log.dbg("que  %12.*s: task declined, id = %d\n", .{ @as(c_int, 12), "process_new_tasks", task.id });
                    self.mutex_tasks.lock();
                    defer self.mutex_tasks.unlock();
                    try self.queue_tasks_unhandled.append(self.gpa, task);
                }
            }
        }

        /// Ports `server_queue`'s `worker_loop` (server-queue.cpp:163 @cd959bae9).
        fn workerLoop(self: *Self) void {
            while (true) {
                {
                    self.mutex_tasks.lock();
                    defer self.mutex_tasks.unlock();
                    // wait on busy instead of yielding - busy stays set even when the yield already ended
                    while (!(self.worker.stop or self.worker.busy)) self.worker.cv.wait(&self.mutex_tasks);
                    if (self.worker.stop) return;
                }

                // process tasks while the yield is active
                while (true) {
                    // note: do not hold any lock here, the callback may post new tasks
                    const terminated = self.processNewTasks(true) catch |e| {
                        self.mutex_tasks.lock();
                        self.worker.err = e;
                        self.mutex_tasks.unlock();
                        break;
                    };

                    self.mutex_tasks.lock();
                    if (terminated or self.worker.stop or !self.worker.yielding) {
                        self.mutex_tasks.unlock();
                        break;
                    }
                    if (self.queue_tasks.items.len != 0) {
                        self.mutex_tasks.unlock();
                        continue; // a new task arrived in the meantime
                    }
                    while (!(self.worker.stop or !self.running or !self.worker.yielding or self.queue_tasks.items.len != 0)) {
                        self.condition_tasks.wait(&self.mutex_tasks);
                    }
                    self.mutex_tasks.unlock();
                }

                // signal to yield_to_queue() that no more tasks will be processed
                self.mutex_tasks.lock();
                self.worker.busy = false;
                self.mutex_tasks.unlock();
                self.condition_tasks.notifyAll();
            }
        }

        /// Ports `server_queue`'s `worker_stop` (server-queue.cpp:209 @cd959bae9).
        fn workerStop(self: *Self) void {
            const t = self.worker.thread orelse return;
            self.mutex_tasks.lock();
            self.worker.stop = true;
            self.mutex_tasks.unlock();
            self.worker.cv.notifyOne();
            self.condition_tasks.notifyAll();
            t.join();
            self.worker.thread = null;
        }

        /// Ports `server_queue`'s `yield_to_queue` (server-queue.cpp:222 @cd959bae9):
        /// runs `work` here while the worker thread keeps taking tasks.
        pub fn yieldToQueue(self: *Self, work_ctx: anytype, comptime work: fn (@TypeOf(work_ctx)) anyerror!void) anyerror!void {
            ggmlAssert(self.worker.thread != null, 223, "worker.thread.joinable() && \"yield_to_queue() requires start_loop() to be running\"");

            log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "yield_to_queue", "yielding to queue\n" });

            {
                self.mutex_tasks.lock();
                defer self.mutex_tasks.unlock();
                ggmlAssert(!self.worker.busy, 229, "!worker.busy && \"yield_to_queue() cannot be nested\"");
                self.worker.busy = true;
                self.worker.yielding = true;
            }
            self.worker.cv.notifyOne();

            // run the work on the current thread, so that all ggml compute stays on the same thread
            var exception: ?anyerror = null;
            work(work_ctx) catch |e| {
                exception = e;
            };

            {
                self.mutex_tasks.lock();
                defer self.mutex_tasks.unlock();

                // the yield is over, wait for the worker to finish its current task
                self.worker.yielding = false;
                self.condition_tasks.notifyAll();
                while (self.worker.busy) self.condition_tasks.wait(&self.mutex_tasks);

                // put the declined tasks back, keeping their order
                while (self.queue_tasks_unhandled.pop()) |t| {
                    try self.queue_tasks.insert(self.gpa, 0, t);
                }

                // make sure to avoid idle timeout here
                self.time_last_task = timeMs();

                // an exception from work() takes precedence over the one from the worker
                if (exception == null) {
                    exception = self.worker.err;
                    self.worker.err = null;
                } else {
                    self.worker.err = null;
                }
            }

            log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "yield_to_queue", "done yielding to queue\n" });

            // note: rethrow only after the declined tasks are back in the queue, so they are not lost
            if (exception) |e| return e;
        }

        /// Ports `server_queue`'s `start_loop` (server-queue.cpp:278 @cd959bae9):
        /// the decode loop, until `terminate`.
        pub fn startLoop(self: *Self, idle_sleep_ms: i64) !void {
            self.running = true;
            self.time_last_task = timeMs();

            // spawn the worker thread used by yield_to_queue()
            ggmlAssert(self.worker.thread == null, 283, "!worker.thread.joinable() && \"start_loop() is already running\"");
            self.worker.stop = false;
            self.worker.busy = false;
            self.worker.yielding = false;
            self.worker.thread = try std.Thread.spawn(.{}, workerLoop, .{self});
            defer self.workerStop();

            const max_wait_time = std.time.ns_per_s;

            outer: while (true) {
                log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "start_loop", "processing new tasks\n" });
                if (try self.processNewTasks(false)) break; // terminate

                // all tasks in the current loop is processed, slots data is now ready
                log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "start_loop", "update slots\n" });

                // this will run the main inference process for all slots
                const t_update_slots = timeMs();
                try self.callback_update_slots.?(self.cb_ctx);
                {
                    // update_slots() may take a while to finish, we need to make sure it's not counted as idle
                    // shift instead of reset, so that non-task_resets_idle_timer tasks do not delay the sleep
                    self.mutex_tasks.lock();
                    defer self.mutex_tasks.unlock();
                    const now = timeMs();
                    self.time_last_task = @min(now, self.time_last_task + (now - t_update_slots));
                }

                log.dbg("que  %12.*s: %s", .{ @as(c_int, 12), "start_loop", "waiting for new tasks\n" });
                while (true) {
                    self.mutex_tasks.lock();
                    if (!self.running or self.queue_tasks.items.len != 0) {
                        self.mutex_tasks.unlock();
                        break; // go back to process new tasks or terminate
                    }

                    // no tasks, check for sleeping state
                    const should_sleep = idle_sleep_ms >= 0 and (timeMs() - self.time_last_task) >= idle_sleep_ms;
                    if (should_sleep) {
                        log.inf("que  %12.*s: %s", .{ @as(c_int, 12), "start_loop", "entering sleeping state\n" });
                        self.sleeping = true;
                        // Call order cb0 -> cb1 -> cb{N}
                        for (self.callback_sleeping_state.items) |cb| cb.f(cb.ctx, true);
                        self.req_stop_sleeping = false;
                        // wait until we are requested to exit sleeping state
                        while (!(!self.running or self.req_stop_sleeping)) self.condition_tasks.wait(&self.mutex_tasks);
                        if (!self.running) { // may changed during sleep
                            self.mutex_tasks.unlock();
                            break :outer; // terminate
                        }
                        log.inf("que  %12.*s: %s", .{ @as(c_int, 12), "start_loop", "exiting sleeping state\n" });
                        self.req_stop_sleeping = false;
                        // Call order cb{N} -> cb1 -> cb0
                        var i = self.callback_sleeping_state.items.len;
                        while (i > 0) : (i -= 1) {
                            const cb = self.callback_sleeping_state.items[i - 1];
                            cb.f(cb.ctx, false);
                        }
                        self.sleeping = false;
                        self.time_last_task = timeMs();
                        self.condition_tasks.notifyAll(); // notify wait_until_no_sleep()
                        self.mutex_tasks.unlock();
                        break; // process new tasks
                    } else {
                        // wait for new tasks or timeout for checking sleeping condition
                        var res = self.queue_tasks.items.len != 0 or !self.running;
                        if (!res) {
                            _ = self.condition_tasks.waitFor(&self.mutex_tasks, max_wait_time);
                            res = self.queue_tasks.items.len != 0 or !self.running;
                        }
                        self.mutex_tasks.unlock();
                        if (res) break; // new task arrived or terminate
                        // otherwise, loop again to check sleeping condition
                    }
                }
            }
        }

        /// Ports `server_queue`'s `cleanup_pending_task` (server-queue.cpp:368 @cd959bae9).
        /// Called with the lock held, by `post` alone.
        fn cleanupPendingTask(self: *Self, id_target: i32) void {
            for ([_]*std.ArrayList(Task){ &self.queue_tasks, &self.queue_tasks_deferred, &self.queue_tasks_unhandled }) |q| {
                var w: usize = 0;
                for (q.items) |*t| {
                    if (t.id == id_target) {
                        t.deinit(self.gpa);
                    } else {
                        q.items[w] = t.*;
                        w += 1;
                    }
                }
                q.shrinkRetainingCapacity(w);
            }
        }
    };
}

/// Ports `server_response` (server-queue.h:154 @cd959bae9). `ResultPtr` is a
/// pointer to a result; the queue owns the results in it.
pub fn Response(comptime ResultPtr: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        running: bool = true,
        waiting_task_ids: IdSet,
        queue_results: std.ArrayList(ResultPtr) = .empty,
        mutex_results: Mutex = .{},
        condition_results: Cond = .{},

        pub fn init(gpa: std.mem.Allocator) !Self {
            return .{ .gpa = gpa, .waiting_task_ids = try IdSet.init(gpa) };
        }

        pub fn deinit(self: *Self) void {
            for (self.queue_results.items) |r| r.destroy(self.gpa);
            self.queue_results.deinit(self.gpa);
            self.waiting_task_ids.deinit();
        }

        /// Ports `server_response`'s `add_waiting_task_id` (server-queue.cpp:389 @cd959bae9).
        pub fn addWaitingTaskId(self: *Self, id_task: i32) !void {
            log.dbg("res  %12.*s: add task %d to waiting list. current waiting = %d (before add)\n", .{ @as(c_int, 12), "add_waiting_task_id", id_task, @as(c_int, @intCast(self.waiting_task_ids.count())) });
            self.mutex_results.lock();
            defer self.mutex_results.unlock();
            try self.waiting_task_ids.insert(id_task);
        }

        /// Ports `server_response`'s `add_waiting_task_ids` (server-queue.cpp:396 @cd959bae9).
        pub fn addWaitingTaskIds(self: *Self, id_tasks: *const IdSet) !void {
            self.mutex_results.lock();
            defer self.mutex_results.unlock();
            var it = id_tasks.map.iterator();
            while (it.next()) |e| {
                log.dbg("res  %12.*s: add task %d to waiting list. current waiting = %d (before add)\n", .{ @as(c_int, 12), "add_waiting_task_ids", e.key, @as(c_int, @intCast(self.waiting_task_ids.count())) });
                try self.waiting_task_ids.insert(e.key);
            }
        }

        /// Ports `server_response`'s `remove_waiting_task_id` (server-queue.cpp:405 @cd959bae9).
        pub fn removeWaitingTaskId(self: *Self, id_task: i32) void {
            log.dbg("res  %12.*s: remove task %d from waiting list. current waiting = %d (before remove)\n", .{ @as(c_int, 12), "remove_waiting_task_id", id_task, @as(c_int, @intCast(self.waiting_task_ids.count())) });
            self.mutex_results.lock();
            defer self.mutex_results.unlock();
            self.eraseId(id_task);
            // make sure to clean up all pending results
            var w: usize = 0;
            for (self.queue_results.items) |r| {
                if (r.id == id_task) {
                    r.destroy(self.gpa);
                } else {
                    self.queue_results.items[w] = r;
                    w += 1;
                }
            }
            self.queue_results.shrinkRetainingCapacity(w);
        }

        /// Ports `server_response`'s `remove_waiting_task_ids` (server-queue.cpp:418 @cd959bae9).
        pub fn removeWaitingTaskIds(self: *Self, id_tasks: *const IdSet) void {
            self.mutex_results.lock();
            defer self.mutex_results.unlock();
            var it = id_tasks.map.iterator();
            while (it.next()) |e| {
                log.dbg("res  %12.*s: remove task %d from waiting list. current waiting = %d (before remove)\n", .{ @as(c_int, 12), "remove_waiting_task_ids", e.key, @as(c_int, @intCast(self.waiting_task_ids.count())) });
                self.eraseId(e.key);
            }
        }

        /// `waiting_task_ids.erase(id)`. libc++'s erase unlinks one node and
        /// leaves the others' order alone; the set is only ever searched,
        /// never walked in an order anything observes, so it is rebuilt.
        fn eraseId(self: *Self, id: i32) void {
            if (!self.waiting_task_ids.contains(id)) return;
            var next = IdSet.init(self.gpa) catch return;
            var it = self.waiting_task_ids.map.iterator();
            while (it.next()) |e| {
                if (e.key != id) next.insert(e.key) catch |err| switch (err) {
                    error.OutOfMemory => {},
                };
            }
            self.waiting_task_ids.deinit();
            self.waiting_task_ids = next;
        }

        fn takeFirst(self: *Self, id_tasks: *const IdSet) ?ResultPtr {
            for (self.queue_results.items, 0..) |r, i| {
                if (id_tasks.contains(r.id)) return self.queue_results.orderedRemove(i);
            }
            return null;
        }

        /// Ports `server_response`'s `recv` (server-queue.cpp:427 @cd959bae9).
        pub fn recv(self: *Self, id_tasks: *const IdSet) ResultPtr {
            while (true) {
                self.mutex_results.lock();
                defer self.mutex_results.unlock();
                while (true) {
                    if (!self.running) {
                        log.dbg("res  %12.*s: %s : queue result stop\n", .{ @as(c_int, 12), "operator()", "recv" });
                        abort(); // we cannot return here since the caller is HTTP code
                    }
                    if (self.queue_results.items.len != 0) break;
                    self.condition_results.wait(&self.mutex_results);
                }
                if (self.takeFirst(id_tasks)) |r| return r;
            }
        }

        /// Ports `server_response`'s `recv_with_timeout` (server-queue.cpp:450 @cd959bae9):
        /// one deadline for the whole wait, so results of other tasks waking
        /// the condition do not extend it.
        ///
        /// Return: null on a timeout.
        pub fn recvWithTimeout(self: *Self, id_tasks: *const IdSet, timeout: c_int) ?ResultPtr {
            const deadline = Cond.deadlineIn(@as(u64, @intCast(@max(timeout, 0))) * std.time.ns_per_s);
            self.mutex_results.lock();
            defer self.mutex_results.unlock();
            while (true) {
                if (self.takeFirst(id_tasks)) |r| return r;
                const ok = self.condition_results.waitUntil(&self.mutex_results, deadline);
                if (!self.running) {
                    log.dbg("res  %12.*s: %s : queue result stop\n", .{ @as(c_int, 12), "recv_with_timeout", "recv_with_timeout" });
                    abort(); // we cannot return here since the caller is HTTP code
                }
                if (!ok) return null;
            }
        }

        /// Ports `server_response`'s `send` (server-queue.cpp:481 @cd959bae9).
        /// Takes the result; drops it when nobody waits for its task.
        pub fn send(self: *Self, result: ResultPtr) !void {
            log.dbg("res  %12.*s: sending result for task id = %d\n", .{ @as(c_int, 12), "send", result.id });
            self.mutex_results.lock();
            defer self.mutex_results.unlock();
            if (self.waiting_task_ids.contains(result.id)) {
                log.dbg("res  %12.*s: task id = %d pushed to result queue\n", .{ @as(c_int, 12), "send", result.id });
                try self.queue_results.append(self.gpa, result);
                self.condition_results.notifyAll();
                return;
            }
            result.destroy(self.gpa);
        }

        /// Ports `server_response`'s `broadcast` (server-queue.cpp:496 @cd959bae9).
        pub fn broadcast(self: *Self, result: ResultPtr) !void {
            defer result.destroy(self.gpa);
            self.mutex_results.lock();
            defer self.mutex_results.unlock();
            var it = self.waiting_task_ids.map.iterator();
            while (it.next()) |e| {
                log.dbg("res  %12.*s: task id = %d pushed to result queue\n", .{ @as(c_int, 12), "broadcast", e.key });
                const res_copy = try result.clone(self.gpa);
                res_copy.id = e.key; // override id with target task id
                try self.queue_results.append(self.gpa, res_copy);
            }
            self.condition_results.notifyAll();
        }

        /// Ports `server_response`'s `terminate` (server-queue.cpp:507 @cd959bae9).
        pub fn terminate(self: *Self) void {
            self.running = false;
            self.condition_results.notifyAll();
        }
    };
}

/// Ports `server_response_reader` (server-queue.h:203 @cd959bae9). `State`
/// is `task_result_state`: `State.init(gpa, session)` makes one, from a
/// chat session or (null) the default one.
pub fn ResponseReader(comptime Task: type, comptime ResultPtr: type, comptime State: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        id_tasks: IdSet,
        queue_tasks: *Queue(Task),
        queue_results: *Response(ResultPtr),
        received_count: usize = 0,
        cancelled: bool = false,
        polling_interval_seconds: c_int,
        states: std.ArrayList(State) = .empty,

        pub fn init(gpa: std.mem.Allocator, queue_tasks: *Queue(Task), queue_results: *Response(ResultPtr), polling_interval_seconds: c_int) !Self {
            return .{
                .gpa = gpa,
                .id_tasks = try IdSet.init(gpa),
                .queue_tasks = queue_tasks,
                .queue_results = queue_results,
                .polling_interval_seconds = polling_interval_seconds,
            };
        }

        /// Ports the destructor of `server_response_reader` (server-queue.h:218 @cd959bae9).
        pub fn deinit(self: *Self) void {
            self.stop() catch |err| switch (err) {
                error.OutOfMemory => {},
            };
            for (self.states.items) |*s| s.deinit(self.gpa);
            self.states.deinit(self.gpa);
            self.id_tasks.deinit();
        }

        /// Ports `server_response_reader`'s `get_new_id` (server-queue.h:222 @cd959bae9).
        pub fn getNewId(self: *Self) i32 {
            return self.queue_tasks.getNewId();
        }

        /// Ports `server_response_reader`'s `post_task` (server-queue.cpp:516 @cd959bae9).
        pub fn postTask(self: *Self, task_in: Task, front: bool) !void {
            var task = task_in;
            ggmlAssert(self.id_tasks.count() == 0, 517, "id_tasks.empty() && \"post_task() can only be called once per reader\"");
            ggmlAssert(!task.isParent(), 518, "!task.is_parent() && \"not supported, use post_tasks() instead\"");
            task.index = 0;
            try self.id_tasks.insert(task.id);
            try self.states.append(self.gpa, try State.init(self.gpa, null));
            try self.queue_results.addWaitingTaskId(task.id);
            _ = try self.queue_tasks.post(task, front);
        }

        /// Ports `server_response_reader`'s `post_tasks` (server-queue.cpp:526 @cd959bae9):
        /// every task's state, and its children's, a copy of `session` (null: the
        /// default session).
        pub fn postTasks(self: *Self, tasks: []Task, session: ?*const State.Session, front: bool) !void {
            ggmlAssert(self.id_tasks.count() == 0, 527, "id_tasks.empty() && \"post_tasks() can only be called once per reader\"");
            self.id_tasks.deinit();
            self.id_tasks = try Task.getListId(self.gpa, tasks);
            try self.states.ensureTotalCapacity(self.gpa, tasks.len);
            var index: usize = 0;
            for (tasks) |*task| {
                task.index = index;
                index += 1;
                try self.states.append(self.gpa, try State.init(self.gpa, session));
                // for child tasks
                for (task.child_tasks.items) |*child_task| {
                    child_task.index = index;
                    index += 1;
                    try self.states.append(self.gpa, try State.init(self.gpa, session));
                }
            }
            ggmlAssert(self.states.items.len == self.id_tasks.count(), 540, "states.size() == id_tasks.size()");
            try self.queue_results.addWaitingTaskIds(&self.id_tasks);
            try self.queue_tasks.postMany(tasks, front);
        }

        /// Ports `server_response_reader`'s `has_next` (server-queue.cpp:545 @cd959bae9).
        pub fn hasNext(self: *const Self) bool {
            return !self.cancelled and self.received_count < self.id_tasks.count();
        }

        /// Ports `server_response_reader`'s `next` (server-queue.cpp:551 @cd959bae9).
        ///
        /// Return: null when `should_stop` said so before a result came.
        pub fn next(self: *Self, should_stop: anytype) !?ResultPtr {
            while (true) {
                const result = self.queue_results.recvWithTimeout(&self.id_tasks, self.polling_interval_seconds) orelse {
                    // timeout, check stop condition
                    if (should_stop.shouldStop()) return null;
                    continue;
                };
                if (result.isError()) {
                    try self.stop(); // cancel remaining tasks
                    log.dbg("srv  %12.*s: %s", .{ @as(c_int, 12), "next", "received error result, stopping further processing\n" });
                    return result;
                }
                if (self.states.items.len != 0) {
                    // update the generation state if needed
                    const idx = result.index;
                    ggmlAssert(idx < self.states.items.len, 568, "idx < states.size()");
                    try result.update(&self.states.items[idx]);
                }
                if (result.isStop()) self.received_count += 1;
                return result;
            }
        }

        /// Ports `server_response_reader`'s `batch_response` (server-queue.h:235 @cd959bae9).
        pub const BatchResponse = struct {
            is_terminated: bool = false, // if true, indicates that processing was stopped before all results were received
            results: []?ResultPtr = &.{},
            err: ?ResultPtr = null, // nullptr if no error

            pub fn deinit(self: *BatchResponse, gpa: std.mem.Allocator) void {
                for (self.results) |r| if (r) |x| x.destroy(gpa);
                gpa.free(self.results);
                if (self.err) |e| e.destroy(gpa);
            }
        };

        /// Ports `server_response_reader`'s `wait_for_all` (server-queue.cpp:581 @cd959bae9).
        pub fn waitForAll(self: *Self, should_stop: anytype) !BatchResponse {
            var batch_res: BatchResponse = .{};
            errdefer batch_res.deinit(self.gpa);
            batch_res.results = try self.gpa.alloc(?ResultPtr, self.id_tasks.count());
            @memset(batch_res.results, null);
            while (self.hasNext()) {
                const res = try self.next(should_stop) orelse {
                    batch_res.is_terminated = true;
                    return batch_res;
                };
                if (res.isError()) {
                    batch_res.err = res;
                    return batch_res;
                }
                const idx = res.index;
                ggmlAssert(idx < batch_res.results.len, 596, "idx < batch_res.results.size() && \"index out of range\"");
                ggmlAssert(batch_res.results[idx] == null, 597, "batch_res.results[idx] == nullptr && \"duplicate result received\"");
                batch_res.results[idx] = res;
            }
            return batch_res;
        }

        /// Ports `server_response_reader`'s `stop` (server-queue.cpp:603 @cd959bae9).
        pub fn stop(self: *Self) !void {
            self.queue_results.removeWaitingTaskIds(&self.id_tasks);
            if (self.hasNext() and !self.cancelled) {
                // if tasks is not finished yet, cancel them
                self.cancelled = true;
                var cancel_tasks: std.ArrayList(Task) = .empty;
                defer cancel_tasks.deinit(self.gpa);
                try cancel_tasks.ensureTotalCapacity(self.gpa, self.id_tasks.count());
                var it = self.id_tasks.map.iterator();
                while (it.next()) |e| {
                    log.wrn("srv  %12.*s: cancel task, id_task = %d\n", .{ @as(c_int, 12), "stop", e.key });
                    var task = Task.initType(.cancel);
                    task.id_target = e.key;
                    self.queue_results.removeWaitingTaskId(e.key);
                    cancel_tasks.appendAssumeCapacity(task);
                }
                // push to beginning of the queue, so it has highest priority
                try self.queue_tasks.postMany(cancel_tasks.items, true);
            } else {
                log.dbg("srv  %12.*s: %s", .{ @as(c_int, 12), "stop", "all tasks already finished, no need to cancel\n" });
            }
        }
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

/// A stand-in `server_task` for the tests: the members the queues touch.
const TestTaskType = enum { completion, cancel, metrics };
const TestState = struct {
    n: usize = 0,
    pub const Session = void;
    pub fn init(_: std.mem.Allocator, _: ?*const void) !TestState {
        return .{};
    }
    pub fn deinit(_: *TestState, _: std.mem.Allocator) void {}
};
const TestTask = struct {
    id: i32 = -1,
    id_target: i32 = -1,
    id_slot: i32 = -1,
    index: usize = 0,
    type: TestTaskType = .completion,
    child_tasks: std.ArrayList(TestTask) = .empty,
    pub fn initType(t: TestTaskType) TestTask {
        return .{ .type = t };
    }
    pub fn isParent(self: *const TestTask) bool {
        return self.child_tasks.items.len > 0;
    }
    pub fn deinit(self: *TestTask, gpa: std.mem.Allocator) void {
        self.child_tasks.deinit(gpa);
    }
    pub fn getListId(gpa: std.mem.Allocator, tasks: []const TestTask) !IdSet {
        var ids = try IdSet.initCapacity(gpa, tasks.len);
        for (tasks) |t| {
            try ids.insert(t.id);
            for (t.child_tasks.items) |ch| try ids.insert(ch.id);
        }
        return ids;
    }
};
const TestResult = struct {
    id: i32,
    index: usize = 0,
    err: bool = false,
    stop: bool = true,
    pub fn isError(self: *TestResult) bool {
        return self.err;
    }
    pub fn isStop(self: *TestResult) bool {
        return self.stop;
    }
    pub fn update(_: *TestResult, s: *TestState) !void {
        s.n += 1;
    }
    pub fn clone(self: *TestResult, gpa: std.mem.Allocator) !*TestResult {
        const r = try gpa.create(TestResult);
        r.* = self.*;
        return r;
    }
    pub fn destroy(self: *TestResult, gpa: std.mem.Allocator) void {
        gpa.destroy(self);
    }
};

const Never = struct {
    pub fn shouldStop(_: Never) bool {
        return false;
    }
};

test "ids, ordering and a reader round trip" {
    const gpa = std.testing.allocator;
    var queue = Queue(TestTask).init(gpa);
    defer queue.deinit();
    var resp = try Response(*TestResult).init(gpa);
    defer resp.deinit();

    var tasks = [_]TestTask{ .{}, .{}, .{} };
    for (&tasks) |*t| t.id = queue.getNewId();
    var reader = try ResponseReader(TestTask, *TestResult, TestState).init(gpa, &queue, &resp, 1);
    defer reader.deinit();
    try reader.postTasks(&tasks, null, false);
    try std.testing.expectEqual(@as(usize, 3), queue.queue_tasks.items.len);
    try std.testing.expectEqual(@as(i32, 0), queue.queue_tasks.items[0].id);
    try std.testing.expectEqual(@as(i32, 2), queue.queue_tasks.items[2].id);
    // results arrive out of order
    for ([_]i32{ 2, 0, 1 }) |id| {
        const r = try gpa.create(TestResult);
        r.* = .{ .id = id, .index = @intCast(id) };
        try resp.send(r);
    }
    // a result nobody waits for is dropped
    const stray = try gpa.create(TestResult);
    stray.* = .{ .id = 99 };
    try resp.send(stray);
    var batch = try reader.waitForAll(Never{});
    defer batch.deinit(gpa);
    try std.testing.expect(!batch.is_terminated);
    for (batch.results, 0..) |r, i| try std.testing.expectEqual(@as(i32, @intCast(i)), r.?.id);
}

test "front posts reverse, cancel cleans up" {
    const gpa = std.testing.allocator;
    var queue = Queue(TestTask).init(gpa);
    defer queue.deinit();
    var ts = [_]TestTask{ .{}, .{}, .{} };
    try queue.postMany(&ts, true);
    try std.testing.expectEqual(@as(i32, 2), queue.queue_tasks.items[0].id);
    var cancel = TestTask.initType(.cancel);
    cancel.id = queue.getNewId();
    cancel.id_target = 1;
    _ = try queue.post(cancel, false);
    try std.testing.expectEqual(@as(usize, 3), queue.queue_tasks.items.len);
    for (queue.queue_tasks.items) |t| try std.testing.expect(t.id != 1);
}

test "the loop runs, yields and terminates" {
    const gpa = std.testing.allocator;
    const Ctx = struct {
        queue: Queue(TestTask),
        seen: std.ArrayList(i32) = .empty,
        updates: usize = 0,
        fn onTask(p: *anyopaque, t: *TestTask, yielding: bool) anyerror!bool {
            const self: *@This() = @ptrCast(@alignCast(p));
            if (yielding and t.type == .metrics) return false;
            try self.seen.append(std.testing.allocator, t.id);
            if (t.id == 3) self.queue.terminate();
            return true;
        }
        fn onUpdate(p: *anyopaque) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(p));
            self.updates += 1;
        }
    };
    var ctx: Ctx = .{ .queue = Queue(TestTask).init(gpa) };
    defer ctx.queue.deinit();
    defer ctx.seen.deinit(gpa);
    ctx.queue.onCallbacks(&ctx, Ctx.onTask, Ctx.onUpdate);
    var ts = [_]TestTask{ .{}, .{}, .{}, .{} };
    try ctx.queue.postMany(&ts, false);
    try ctx.queue.startLoop(-1);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, ctx.seen.items);
}

test "IdSet walks in libc++'s order" {
    // `std::unordered_set<int> s(5)` with 10, 3, 7, 21, 4 inserted walks
    // 4 21 7 3 10 over 5 buckets, measured against the SDK's libc++.
    const gpa = std.testing.allocator;
    var s = try IdSet.initCapacity(gpa, 5);
    defer s.deinit();
    for ([_]i32{ 10, 3, 7, 21, 4 }) |x| try s.insert(x);
    const o = try s.ordered(gpa);
    defer gpa.free(o);
    try std.testing.expectEqualSlices(i32, &.{ 4, 21, 7, 3, 10 }, o);
    try std.testing.expectEqual(@as(usize, 5), s.map.bucketCount());
}
