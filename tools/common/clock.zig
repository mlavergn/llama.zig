//! The two clocks the ported tools read, on every target.
//!
//! # Provenance
//!
//! **Not a port of a file.** Upstream reads `std::chrono`; the port read
//! `clock_gettime` through `std.c`, whose declaration Zig 0.17 refuses on
//! Windows (`clockid_t` is `void` there), so nothing that named it compiled
//! for `x86_64-windows-gnu`. This keeps that call elsewhere and asks Win32
//! there, which is what libc++'s `std::chrono` does underneath.

const std = @import("std");
const builtin = @import("builtin");

const is_windows = builtin.os.tag == .windows;

extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.winapi) c_int;
extern "kernel32" fn QueryPerformanceFrequency(frequency: *i64) callconv(.winapi) c_int;
extern "kernel32" fn GetSystemTimePreciseAsFileTime(time: *u64) callconv(.winapi) void;

/// 100-nanosecond intervals from 1601, a `FILETIME`'s epoch, to 1970.
const filetime_unix_epoch: u64 = 116444736000000000;

/// The system clock.
///
/// Return: microseconds since the Unix epoch.
pub fn realtimeUs() i64 {
    if (comptime is_windows) {
        var ft: u64 = undefined;
        GetSystemTimePreciseAsFileTime(&ft);
        return @intCast((ft - filetime_unix_epoch) / 10);
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i64, ts.sec) * 1000000 + @divTrunc(@as(i64, ts.nsec), 1000);
}

/// A monotonic clock.
///
/// Return: nanoseconds from an arbitrary origin. Whole seconds and the
/// remainder scale apart: an `i128` division here would pull in `__divti3`,
/// and `compiler_rt` with it, which shadows libm's `expf` (`CLAUDE.md`).
pub fn monotonicNs() u64 {
    if (comptime is_windows) {
        var count: i64 = undefined;
        var frequency: i64 = undefined;
        _ = QueryPerformanceCounter(&count);
        _ = QueryPerformanceFrequency(&frequency);
        const c: u64 = @intCast(count);
        const f: u64 = @intCast(frequency);
        return (c / f) * 1_000_000_000 + (c % f) * 1_000_000_000 / f;
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the clocks read sensible values" {
    const a = monotonicNs();
    const b = monotonicNs();
    try std.testing.expect(b >= a);
    // After 2020-01-01, in microseconds.
    try std.testing.expect(realtimeUs() > 1_577_836_800_000_000);
}
