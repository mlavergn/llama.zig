//! Where `common/chat.cpp`, the auto-parser and the per-model parsers log.
//!
//! # Provenance
//!
//! **Not a port.** Upstream logs through `common/log.h`'s `LOG_DBG`,
//! `LOG_INF`, `LOG_WRN` and `LOG_ERR`, each a `printf`-style call into
//! `common_log_main()`. The port formats with Zig's `std.fmt` and hands the
//! finished line to whatever the host installs here -- the server its
//! `common_log`, a gate its transcript -- so a module built without a
//! logger stays silent, as an unconfigured one would.
//!
//! `msg.zig`'s older `log_sink` takes the two levels it uses; a line at
//! either goes there too when no sink is installed here, so a host that set
//! only that one keeps seeing warnings and errors.

const std = @import("std");
const msg = @import("msg.zig");

/// `common_log`'s four levels, as the macros name them.
pub const Level = enum { debug, info, warn, err };

/// The host's logger, or null to drop every line but warnings and errors
/// (see the file's header). The text is the whole line, `\n` included where
/// upstream's format has one.
pub var sink: ?*const fn (level: Level, text: []const u8) void = null;

/// `LOG_DBG(...)` and friends.
///
/// Parameters:
/// - `level`: the macro's level.
/// - `fmt`, `args`: the line, in Zig's `std.fmt` syntax.
pub fn print(level: Level, comptime fmt: []const u8, args: anytype) void {
    if (sink == null and msg.log_sink == null) return;
    if (level == .debug and !debugEnabled()) return;
    if (sink == null and (level == .debug or level == .info)) return;
    var stack: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&stack);
    const text = std.fmt.allocPrint(fba.allocator(), fmt, args) catch
        std.fmt.allocPrint(std.heap.smp_allocator, fmt, args) catch return;
    defer if (!fba.ownsSlice(text)) std.heap.smp_allocator.free(text);
    if (sink) |s| return s(level, text);
    msg.log_sink.?(if (level == .warn) .warn else .err, text);
}

/// `LOG_DBG`.
pub fn dbg(comptime fmt: []const u8, args: anytype) void {
    print(.debug, fmt, args);
}

/// `LOG_INF`.
pub fn inf(comptime fmt: []const u8, args: anytype) void {
    print(.info, fmt, args);
}

/// `LOG_WRN`.
pub fn wrn(comptime fmt: []const u8, args: anytype) void {
    print(.warn, fmt, args);
}

/// `LOG_ERR`.
pub fn err(comptime fmt: []const u8, args: anytype) void {
    print(.err, fmt, args);
}

/// The host's verbosity check for `LOG_DBG`, which evaluates its arguments
/// only when the line prints; null lets every debug line reach the sink.
pub var debug_on: ?*const fn () bool = null;

/// Whether a debug line would reach anyone, so a caller can skip building
/// an expensive one (a parser's dump) as upstream's verbosity check does.
pub fn debugEnabled() bool {
    if (sink == null) return false;
    return if (debug_on) |f| f() else true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

var test_seen: [64]u8 = undefined;
var test_len: usize = 0;

fn testSink(level: Level, text: []const u8) void {
    _ = level;
    @memcpy(test_seen[0..text.len], text);
    test_len = text.len;
}

test "a line reaches the installed sink" {
    sink = testSink;
    defer sink = null;
    wrn("{s}: {d}\n", .{ "x", 3 });
    try std.testing.expectEqualStrings("x: 3\n", test_seen[0..test_len]);
}
