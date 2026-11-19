//! Waiting on and waking a 32-bit word: each OS's futex, called directly.
//!
//! # Provenance
//!
//! **Not a port.** The port's own primitive, under `Mutex` and `Condition`.
//! The per-OS calls are the ones `std.Io.Threaded` makes (Zig 0.17's
//! `lib/std/Io/Threaded.zig`, `futexWaitInner` and `futexWake`), without the
//! `Io` and its cancelation, which a function entered through a C ABI cannot
//! supply:
//!
//! - Linux: the `futex` system call, private, with no C library in between.
//! - Darwin: `__ulock_wait2` and `__ulock_wake`. Darwin has no stable system
//!   call ABI, so these are libSystem's, as everything there is.
//! - Windows: `RtlWaitOnAddress` and `RtlWakeAddress*`, from ntdll.

const std = @import("std");
const builtin = @import("builtin");

const native_os = builtin.os.tag;

/// How a wait ended.
pub const Wait = enum {
    /// Woken, spuriously, or `ptr.*` was not `expect` to begin with.
    woken,
    /// The timeout passed.
    timed_out,
};

/// Sleeps while `ptr.*` is `expect`, until woken or `timeout_ns` passes.
///
/// May return early for no reason; callers re-check their condition.
///
/// Parameters:
/// - `ptr`: the word to wait on.
/// - `expect`: the value it must still hold for the wait to sleep.
/// - `timeout_ns`: the longest sleep, relative; null waits indefinitely.
///
/// Return: whether the timeout passed.
pub fn wait(ptr: *const std.atomic.Value(u32), expect: u32, timeout_ns: ?u64) Wait {
    const addr: *const u32 = &ptr.raw;
    switch (native_os) {
        .linux => {
            const linux = std.os.linux;
            var ts: linux.timespec = undefined;
            const tsp: ?*const linux.timespec = if (timeout_ns) |ns| blk: {
                ts = .{
                    .sec = @intCast(ns / std.time.ns_per_s),
                    .nsec = @intCast(ns % std.time.ns_per_s),
                };
                break :blk &ts;
            } else null;
            const rc = linux.futex_4arg(addr, .{ .cmd = .WAIT, .private = true }, expect, tsp);
            return switch (linux.errno(rc)) {
                .TIMEDOUT => .timed_out,
                else => .woken, // woken, AGAIN (changed), INTR
            };
        },
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
            const c = std.c;
            const flags: c.UL = .{ .op = .COMPARE_AND_WAIT, .NO_ERRNO = true };
            // 0 means no timeout to `__ulock_wait2`; the shortest real one is 1.
            const ns: u64 = if (timeout_ns) |t| @max(t, 1) else 0;
            const status = c.__ulock_wait2(flags, addr, expect, ns, 0);
            if (status >= 0) return .woken;
            return switch (@as(c.E, @fromBackingInt(@intCast(-status)))) {
                .TIMEDOUT => .timed_out,
                else => .woken, // INTR, FAULT (paged out): the caller retries
            };
        },
        .windows => {
            const windows = std.os.windows;
            // Relative timeouts are negative, in 100 ns units.
            var t: windows.LARGE_INTEGER = undefined;
            const tp: ?*const windows.LARGE_INTEGER = if (timeout_ns) |ns| blk: {
                t = -@as(windows.LARGE_INTEGER, @intCast(@max(ns / 100, 1)));
                break :blk &t;
            } else null;
            const status = windows.ntdll.RtlWaitOnAddress(addr, &expect, @sizeOf(u32), tp);
            return if (status == .TIMEOUT) .timed_out else .woken;
        },
        else => @compileError("no futex for " ++ @tagName(native_os)),
    }
}

/// Wakes threads sleeping in `wait` on `ptr`.
///
/// Parameters:
/// - `ptr`: the word they wait on.
/// - `max_waiters`: how many to wake at most; 1 wakes one.
///
/// Return: nothing.
pub fn wake(ptr: *const std.atomic.Value(u32), max_waiters: u32) void {
    const addr: *const u32 = &ptr.raw;
    switch (native_os) {
        .linux => {
            const linux = std.os.linux;
            _ = linux.futex_3arg(addr, .{ .cmd = .WAKE, .private = true }, @min(max_waiters, std.math.maxInt(i32)));
        },
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
            const c = std.c;
            const flags: c.UL = .{ .op = .COMPARE_AND_WAIT, .NO_ERRNO = true, .WAKE_ALL = max_waiters > 1 };
            while (true) {
                const status = c.__ulock_wake(flags, addr, 0);
                if (status >= 0) return;
                switch (@as(c.E, @fromBackingInt(@intCast(-status)))) {
                    .INTR, .CANCELED => continue,
                    else => return, // NOENT: nobody was waiting
                }
            }
        },
        .windows => {
            const ntdll = std.os.windows.ntdll;
            if (max_waiters > 1) ntdll.RtlWakeAddressAll(addr) else ntdll.RtlWakeAddressSingle(addr);
        },
        else => @compileError("no futex for " ++ @tagName(native_os)),
    }
}

// ---------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a wait on a word that has already changed returns at once" {
    var word: std.atomic.Value(u32) = .init(1);
    try std.testing.expectEqual(Wait.woken, wait(&word, 0, null));
}

test "a timed wait that nobody wakes times out" {
    var word: std.atomic.Value(u32) = .init(0);
    var spurious: usize = 0;
    while (wait(&word, 0, 1 * std.time.ns_per_ms) == .woken) {
        spurious += 1;
        try std.testing.expect(spurious < 100);
    }
}

test "a wake reaches a sleeping thread" {
    var word: std.atomic.Value(u32) = .init(0);
    const T = struct {
        fn run(w: *std.atomic.Value(u32)) void {
            while (w.load(.acquire) == 0) _ = wait(w, 0, null);
        }
    };
    const t = try std.Thread.spawn(.{}, T.run, .{&word});
    word.store(1, .release);
    wake(&word, 1);
    t.join();
}
