//! A condition variable on one futex word, paired with `Mutex`.
//!
//! # Provenance
//!
//! **Not a port.** It replaces the `pthread_cond_t` and Windows
//! `CONDITION_VARIABLE` under upstream's `std::condition_variable` and
//! `ggml_cond_t`. The word is a sequence number: a waiter reads it before
//! releasing the mutex and sleeps only while it is unchanged, so a signal
//! between the unlock and the sleep is never lost. A count of waiters lets a
//! signal with nobody waiting skip the wake, as `std.Io.Condition` does: the
//! CPU thread pool broadcasts on every graph, usually to workers still
//! spinning.

const std = @import("std");
const futex = @import("futex.zig");
const Mutex = @import("mutex.zig").Mutex;

/// A condition variable. `.{}` is ready to use and needs no teardown.
///
/// A wait may return without a signal, as `pthread_cond_wait` may: every
/// caller waits in a loop on its own condition.
pub const Condition = extern struct {
    seq: std.atomic.Value(u32) = .init(0),
    waiters: std.atomic.Value(u32) = .init(0),

    /// Releases `m`, sleeps until signalled, and re-acquires `m`.
    ///
    /// Parameters:
    /// - `m`: the mutex the caller holds.
    ///
    /// Return: nothing.
    pub fn wait(self: *Condition, m: *Mutex) void {
        _ = self.timedWait(m, null);
    }

    /// `wait`, giving up after `timeout_ns`.
    ///
    /// Parameters:
    /// - `m`: the mutex the caller holds.
    /// - `timeout_ns`: the longest sleep, relative; null waits indefinitely.
    ///
    /// Return: false when the timeout passed, true otherwise -- which includes
    /// a spurious wakeup.
    pub fn timedWait(self: *Condition, m: *Mutex, timeout_ns: ?u64) bool {
        // Counted in before the sequence is read, both `seq_cst`, against
        // `wake`'s bump-then-read: a signaller that sees no waiter bumped the
        // sequence before this read it, so the sleep below returns at once.
        _ = self.waiters.fetchAdd(1, .seq_cst);
        const seq = self.seq.load(.seq_cst);
        m.unlock();
        const result = futex.wait(&self.seq, seq, timeout_ns);
        m.lock();
        _ = self.waiters.fetchSub(1, .monotonic);
        return result != .timed_out;
    }

    /// Wakes one waiter.
    ///
    /// Return: nothing.
    pub fn signal(self: *Condition) void {
        self.wake(1);
    }

    /// Wakes every waiter.
    ///
    /// Return: nothing.
    pub fn broadcast(self: *Condition) void {
        self.wake(std.math.maxInt(u32));
    }

    fn wake(self: *Condition, max_waiters: u32) void {
        _ = self.seq.fetchAdd(1, .seq_cst);
        if (self.waiters.load(.seq_cst) != 0) futex.wake(&self.seq, max_waiters);
    }
};

// ---------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a timed wait with no signal times out holding the mutex" {
    var m: Mutex = .{};
    var c: Condition = .{};
    m.lock();
    var spurious: usize = 0;
    while (c.timedWait(&m, 1 * std.time.ns_per_ms)) {
        spurious += 1;
        try std.testing.expect(spurious < 100);
    }
    try std.testing.expect(!m.tryLock());
    m.unlock();
}

test "broadcast releases every waiter" {
    const threads = 6;
    const Shared = struct {
        m: Mutex = .{},
        c: Condition = .{},
        go: bool = false,
        done: std.atomic.Value(u32) = .init(0),
        fn run(s: *@This()) void {
            s.m.lock();
            while (!s.go) s.c.wait(&s.m);
            s.m.unlock();
            _ = s.done.fetchAdd(1, .release);
        }
    };
    var s: Shared = .{};
    var ts: [threads]std.Thread = undefined;
    for (&ts) |*t| t.* = try std.Thread.spawn(.{}, Shared.run, .{&s});
    s.m.lock();
    s.go = true;
    s.c.broadcast();
    s.m.unlock();
    for (ts) |t| t.join();
    try std.testing.expectEqual(@as(u32, threads), s.done.load(.acquire));
}

test "a producer and consumer hand over every item through signal" {
    const items = 50_000;
    const Shared = struct {
        m: Mutex = .{},
        c: Condition = .{},
        queued: u32 = 0,
        taken: u32 = 0,
        fn consume(s: *@This()) void {
            s.m.lock();
            defer s.m.unlock();
            while (s.taken < items) {
                while (s.queued == 0) s.c.wait(&s.m);
                s.queued -= 1;
                s.taken += 1;
            }
        }
    };
    var s: Shared = .{};
    const t = try std.Thread.spawn(.{}, Shared.consume, .{&s});
    for (0..items) |_| {
        s.m.lock();
        s.queued += 1;
        s.m.unlock();
        s.c.signal();
    }
    t.join();
    try std.testing.expectEqual(@as(u32, items), s.taken);
}

test "a signal before anyone waits is not kept for a later waiter" {
    var m: Mutex = .{};
    var c: Condition = .{};
    c.signal();
    c.broadcast();
    m.lock();
    defer m.unlock();
    var spurious: usize = 0;
    while (c.timedWait(&m, 1 * std.time.ns_per_ms)) {
        spurious += 1;
        try std.testing.expect(spurious < 100);
    }
}
