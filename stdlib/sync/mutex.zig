//! A mutex on one futex word, for every lock in the port.
//!
//! # Provenance
//!
//! **Not a port.** It replaces the `pthread_mutex_t` and Windows `SRWLOCK`
//! that upstream's `std::mutex` and `ggml_mutex_t` are built on, with no C
//! library under it: `futex.zig` reaches the OS directly. The algorithm is the
//! three-state mutex of Drepper's "Futexes Are Tricky" (mutex2): only an
//! unlock that saw contention pays for a wake.

const std = @import("std");
const futex = @import("futex.zig");

/// A non-recursive mutex. `.{}` is unlocked and needs no setup or teardown,
/// as `PTHREAD_MUTEX_INITIALIZER` and `std::mutex`'s `constexpr` constructor
/// do.
///
/// `extern` so it can stand in a C-ABI global (`ggml_critical_section_mutex`).
pub const Mutex = extern struct {
    state: std.atomic.Value(u32) = .init(unlocked),

    const unlocked: u32 = 0;
    const locked: u32 = 1;
    /// Locked, and someone may be sleeping on it.
    const contended: u32 = 2;

    /// Acquires the lock, sleeping while another thread holds it.
    ///
    /// Return: nothing.
    pub fn lock(self: *Mutex) void {
        if (self.state.cmpxchgWeak(unlocked, locked, .acquire, .monotonic) == null) return;
        self.lockSlow();
    }

    fn lockSlow(self: *Mutex) void {
        @branchHint(.cold);
        // From here on the lock is marked contended, so whoever releases it
        // wakes a sleeper; a thread that takes it this way cannot know
        // there is no one else waiting.
        if (self.state.load(.monotonic) == contended) _ = futex.wait(&self.state, contended, null);
        while (self.state.swap(contended, .acquire) != unlocked) {
            _ = futex.wait(&self.state, contended, null);
        }
    }

    /// Acquires the lock only if it is free.
    ///
    /// Return: whether the lock is now held.
    pub fn tryLock(self: *Mutex) bool {
        return self.state.cmpxchgStrong(unlocked, locked, .acquire, .monotonic) == null;
    }

    /// Releases the lock, waking one sleeper if any may be waiting.
    ///
    /// Return: nothing.
    pub fn unlock(self: *Mutex) void {
        if (self.state.swap(unlocked, .release) == contended) futex.wake(&self.state, 1);
    }
};

// ---------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a held mutex refuses tryLock and frees on unlock" {
    var m: Mutex = .{};
    try std.testing.expect(m.tryLock());
    try std.testing.expect(!m.tryLock());
    m.unlock();
    try std.testing.expect(m.tryLock());
    m.unlock();
}

test "eight threads incrementing under the lock lose no increment" {
    const threads = 8;
    const per_thread = 100_000;
    const Shared = struct {
        m: Mutex = .{},
        n: u64 = 0,
        fn run(s: *@This()) void {
            for (0..per_thread) |_| {
                s.m.lock();
                s.n += 1;
                s.m.unlock();
            }
        }
    };
    var s: Shared = .{};
    var ts: [threads]std.Thread = undefined;
    for (&ts) |*t| t.* = try std.Thread.spawn(.{}, Shared.run, .{&s});
    for (ts) |t| t.join();
    try std.testing.expectEqual(@as(u64, threads * per_thread), s.n);
}
