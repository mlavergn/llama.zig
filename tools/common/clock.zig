//! The two clocks the ported tools read, on `std.Io`.
//!
//! # Provenance
//!
//! **Not a port of a file.** Upstream reads `std::chrono`. Its
//! `system_clock` is `CLOCK_REALTIME` (Windows: the precise system time),
//! std's `.real`. Its `steady_clock`, as libc++ reads it
//! (`src/chrono.cpp`), is `CLOCK_MONOTONIC_RAW` on Apple, `CLOCK_MONOTONIC`
//! elsewhere and `QueryPerformanceCounter` on Windows: std's `.boot` on
//! Apple and `.awake` on the rest.
//!
//! std's timestamps are `i96` nanoseconds; each is narrowed to `i64` before
//! any division, since dividing a wider integer pulls in `__divti3`, and
//! `compiler_rt` with it, which shadows libm's `expf`.

const std = @import("std");
const builtin = @import("builtin");
const file = @import("file.zig");

/// `steady_clock`'s clock, as libc++ reads it on this target.
const steady: std.Io.Clock = if (builtin.os.tag.isDarwin()) .boot else .awake;

fn nowNs(clock: std.Io.Clock) i64 {
    return @intCast(clock.now(file.defaultIo()).nanoseconds);
}

/// The system clock.
///
/// Return: microseconds since the Unix epoch.
pub fn realtimeUs() i64 {
    return @divTrunc(nowNs(.real), 1000);
}

/// A monotonic clock: `steady_clock`.
///
/// Return: nanoseconds from an arbitrary origin.
pub fn monotonicNs() u64 {
    return @intCast(nowNs(steady));
}

/// `time(NULL)`.
///
/// Return: seconds since the Unix epoch.
pub fn timeSeconds() i64 {
    return @divFloor(nowNs(.real), std.time.ns_per_s);
}

/// `strftime(buf, n, "%FT%TZ", gmtime(&t))`: `t` in UTC, ISO 8601.
///
/// Parameters:
/// - `buf`: where the text goes.
/// - `t`: seconds since the Unix epoch, not before it.
///
/// Return: the text, in `buf`; `error.NoSpaceLeft` never, as the 20 bytes
/// it writes fit.
pub fn isoUtc(buf: *[32]u8, t: i64) error{NoSpaceLeft}![]const u8 {
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(t, 0)) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year,              md.month.numeric(),      @as(u32, md.day_index) + 1,
        ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    });
}

/// `std::this_thread::sleep_for(std::chrono::milliseconds(ms))`.
pub fn sleepMs(ms: u64) void {
    const d: std.Io.Clock.Duration = .{ .raw = .fromMilliseconds(@intCast(ms)), .clock = .awake };
    d.sleep(file.defaultIo()) catch |err| switch (err) {
        error.Canceled => {}, // the global Io cannot be canceled
    };
}

/// `sleep(seconds)`.
///
/// Return: nothing; `error.Canceled` never, as the global `Io` cannot be
/// canceled.
pub fn sleepSeconds(seconds: u32) std.Io.Cancelable!void {
    const d: std.Io.Clock.Duration = .{ .raw = .fromSeconds(seconds), .clock = .awake };
    try d.sleep(file.defaultIo());
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

test "isoUtc writes what strftime %FT%TZ over gmtime writes" {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("1970-01-01T00:00:00Z", try isoUtc(&buf, 0));
    try std.testing.expectEqualStrings("2000-02-29T23:59:59Z", try isoUtc(&buf, 951868799));
    try std.testing.expectEqualStrings("2026-11-19T21:05:07Z", try isoUtc(&buf, 1795122307));
}
