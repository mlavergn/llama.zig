//! `common_log`: the `LOG_*` macros the tools write through.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/log.cpp` and the macros of `common/log.h` (v0.3.0,
//! `c1d0e7a00`), and `tty_can_use_colors` from `common/common.cpp`. Each
//! declaration names what it ports and the line it began at.
//!
//! # Synchronous, where upstream is not
//!
//! Upstream queues each message for a worker thread, so a `LOG_ERR` and a
//! direct `fprintf(stderr, ...)` after it can reach `stderr` in either order
//! -- upstream is not deterministic there. Here every message is written when
//! it is logged, in the order the C issues them, which is one of the orders
//! upstream can produce and the one a gate can compare.
//!
//! Each message is formatted with `snprintf` and the C's own format string,
//! as `cli/fit.zig`'s logger is: Zig's formatting differs from `printf` in
//! the last digit of a rounded float.

const std = @import("std");
const builtin = @import("builtin");
const io = @import("io.zig");

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
extern fn fputs(s: [*:0]const u8, stream: *anyopaque) c_int;
extern fn fflush(stream: ?*anyopaque) c_int;
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern fn isatty(fd: c_int) c_int;
extern fn _isatty(fd: c_int) c_int;

/// `ggml_log_level`.
pub const Level = enum(c_int) { none = 0, debug = 1, info = 2, warn = 3, err = 4, cont = 5 };

/// `LOG_LEVEL_DEBUG`, `LOG_LEVEL_TRACE`, `LOG_LEVEL_INFO`, `LOG_LEVEL_WARN`,
/// `LOG_LEVEL_ERROR`, `LOG_LEVEL_OUTPUT` (log.h:24, 25, 26, 27, 28, 29 @c1d0e7a00).
pub const level_debug: c_int = 5;
pub const level_trace: c_int = 4;
pub const level_info: c_int = 3;
pub const level_warn: c_int = 2;
pub const level_error: c_int = 1;
pub const level_output: c_int = 0;

/// Ports `common_log_verbosity_thold` (log.cpp:29 @c1d0e7a00):
/// `LOG_DEFAULT_LLAMA`.
pub var verbosity_thold: c_int = level_info;

/// The logger's settings, `common_log`'s members.
var colors: ?bool = null;
var paused: bool = false;
var file: ?*anyopaque = null;
var prefix: bool = false;
var timestamps: bool = false;
var t_start: i64 = 0;

/// Ports `g_col` and `set_colors` (log.cpp:57, 330 @c1d0e7a00), a member of `common_log`:
/// default, bold, red, green, yellow, blue, magenta, cyan, white.
const col_on = [_][*:0]const u8{ "\x1b[0m", "\x1b[1m", "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[34m", "\x1b[35m", "\x1b[36m", "\x1b[37m" };
const col_default = 0;
const col_red = 2;
const col_green = 3;
const col_yellow = 4;
const col_blue = 5;
const col_magenta = 6;

fn col(i: usize) [*:0]const u8 {
    return if (useColors()) col_on[i] else "";
}

/// `common_log_main`'s first call sets the colours from the terminal
/// (log.cpp:373 @c1d0e7a00).
fn useColors() bool {
    if (colors == null) colors = ttyCanUseColors();
    return colors.?;
}

/// Ports `tty_can_use_colors` (common/common.cpp:1185 @c1d0e7a00).
pub fn ttyCanUseColors() bool {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (getenv("NO_COLOR")) |no_color| {
        if (no_color[0] != 0) return false;
    }
    // Check TERM environment variable
    if (getenv("TERM")) |term| {
        if (std.mem.eql(u8, std.mem.span(term), "dumb")) return false;
    }
    // Check if stdout and stderr are connected to a terminal
    const tty = if (comptime builtin.os.tag == .windows) _isatty else isatty;
    return tty(1) != 0 or tty(2) != 0;
}

/// Ports `common_log_set_colors` (log.cpp:413 @c1d0e7a00) for an explicit
/// choice; `null` is `LOG_COLORS_AUTO`.
pub fn setColors(on: ?bool) void {
    main();
    colors = on orelse ttyCanUseColors();
}

/// Ports `common_log_pause` (log.cpp:390 @c1d0e7a00): messages logged while
/// paused are discarded.
pub fn pause() void {
    main();
    paused = true;
}

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: *anyopaque) c_int;

/// Ports `common_log_set_file` (log.cpp:409 @c1d0e7a00): every message is
/// also written to `path`.
pub fn setFile(path: ?[*:0]const u8) void {
    main();
    if (file) |f| _ = fclose(f);
    file = if (path) |p| fopen(p, "w") else null;
}

/// Ports `common_log_set_prefix` (log.cpp:428 @c1d0e7a00).
pub fn setPrefix(on: bool) void {
    main();
    prefix = on;
}

/// Ports `common_log_set_timestamps` (log.cpp:432 @c1d0e7a00).
pub fn setTimestamps(on: bool) void {
    main();
    timestamps = on;
}

/// `t_us`, the system clock in microseconds (log.cpp:39 @c1d0e7a00).
fn tUs() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i64, ts.sec) * 1000000 + @divTrunc(@as(i64, ts.nsec), 1000);
}

/// `common_log_main`'s first call constructs the logger, which starts its
/// clock (log.cpp:373 @c1d0e7a00); every entry point here is such a call.
fn main() void {
    if (t_start == 0) t_start = tUs();
}

/// Ports `common_log_add` and `print` (log.cpp:402, 80
/// @c1d0e7a00), a member of `common_log_entry`, one message, written at once.
pub fn add(level: Level, comptime fmt: [*:0]const u8, args: anytype) void {
    // discard messages while the worker thread is paused
    main();
    if (paused) return;
    var stack: [256]u8 = undefined;
    const n: usize = @intCast(@call(.auto, snprintf, .{ &stack, stack.len, fmt } ++ args));
    var heap: ?[]u8 = null;
    defer if (heap) |h| std.heap.c_allocator.free(h);
    const msg: [*:0]const u8 = if (n < stack.len) @ptrCast(&stack) else blk: {
        const h = std.heap.c_allocator.alloc(u8, n + 1) catch return;
        heap = h;
        _ = @call(.auto, snprintf, .{ h.ptr, n + 1, fmt } ++ args);
        break :blk @ptrCast(h.ptr);
    };
    const timestamp: i64 = if (timestamps) tUs() - t_start else 0;

    print(level, msg, timestamp, null);
    if (file) |f| print(level, msg, timestamp, f);
}

/// Ports `print` (log.cpp:80 @c1d0e7a00), a member of `common_log_entry`: to `to`, or to
/// `stdout` or `stderr` by level.
fn print(level: Level, msg: [*:0]const u8, timestamp: i64, to: ?*anyopaque) void {
    var fcur: *anyopaque = undefined;
    if (to) |t| {
        fcur = t;
    } else {
        // stderr displays DBG messages only when their verbosity level is not
        // higher than the threshold
        if (level == .debug and verbosity_thold < level_debug) return;
        fcur = if (level == .none) io.cStdout() else io.cStderr();
    }

    if (level != .none and level != .cont and prefix) {
        if (timestamp != 0) {
            fprintfTo(fcur, "%s%d.%02d.%03d.%03d%s ", .{
                col(col_blue),
                @as(c_int, @intCast(@divTrunc(timestamp, 1000000 * 60))),
                @as(c_int, @intCast(@mod(@divTrunc(timestamp, 1000000), 60))),
                @as(c_int, @intCast(@mod(@divTrunc(timestamp, 1000), 1000))),
                @as(c_int, @intCast(@mod(timestamp, 1000))),
                col(col_default),
            });
        }
        switch (level) {
            .info => fprintfTo(fcur, "%sI %s", .{ col(col_green), col(col_default) }),
            .warn => fprintfTo(fcur, "%sW %s", .{ col(col_magenta), @as([*:0]const u8, "") }),
            .err => fprintfTo(fcur, "%sE %s", .{ col(col_red), @as([*:0]const u8, "") }),
            .debug => fprintfTo(fcur, "%sD %s", .{ col(col_yellow), @as([*:0]const u8, "") }),
            else => {},
        }
    }

    _ = fputs(msg, fcur);

    if (level == .warn or level == .err or level == .debug) _ = fputs(col(col_default), fcur);

    _ = fflush(fcur);
}

extern fn fprintf(stream: *anyopaque, format: [*:0]const u8, ...) c_int;

fn fprintfTo(stream: *anyopaque, comptime fmt: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, fprintf, .{ stream, fmt } ++ args);
}

/// Ports `LOG_TMPL` (log.h:104 @c1d0e7a00): `LOGV`, `LOG_DBGV` and the other
/// `*V` macros are this with an explicit verbosity.
pub fn tmpl(level: Level, verbosity: c_int, comptime fmt: [*:0]const u8, args: anytype) void {
    if (verbosity <= verbosity_thold) add(level, fmt, args);
}

/// Ports `LOG` (log.h:111 @c1d0e7a00): output, on stdout.
pub fn out(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.none, level_output, fmt, args);
}

/// Ports `LOG_DBG` (log.h:114 @c1d0e7a00).
pub fn dbg(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.debug, level_debug, fmt, args);
}

/// Ports `LOG_TRC` (log.h:115 @c1d0e7a00).
pub fn trc(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.info, level_trace, fmt, args);
}

/// Ports `LOG_INF` (log.h:116 @c1d0e7a00).
pub fn inf(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.info, level_info, fmt, args);
}

/// Ports `LOG_WRN` (log.h:117 @c1d0e7a00).
pub fn wrn(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.warn, level_warn, fmt, args);
}

/// Ports `LOG_ERR` (log.h:118 @c1d0e7a00).
pub fn err(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.err, level_error, fmt, args);
}

/// Ports `LOG_CNT` (log.h:119 @c1d0e7a00).
pub fn cnt(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.cont, level_info, fmt, args);
}

/// Ports `COM_ERR` (common/common.h:34 @c1d0e7a00): `LOG_ERR` with the calling
/// function's name, as `cmn  %12.*s: `.
pub fn comErr(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    err("cmn  %12.*s: " ++ fmt, .{ @as(c_int, 12), func } ++ args);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
