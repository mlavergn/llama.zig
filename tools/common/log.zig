//! `common_log`: the `LOG_*` macros the tools write through.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/log.cpp` and the macros of `common/log.h` (v0.8.0,
//! `8642541b2`), and `tty_can_use_colors`, `tty_enable_ansi` and
//! `common_is_tty` from `common/common.cpp`. Each declaration names what it
//! ports and the line it began at.
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
const json = @import("cli_fit").json;

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
extern fn fputs(s: [*:0]const u8, stream: *anyopaque) c_int;
extern fn fflush(stream: ?*anyopaque) c_int;
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern fn isatty(fd: c_int) c_int;
extern fn _isatty(fd: c_int) c_int;

/// `ggml_log_level`.
pub const Level = enum(c_int) { none = 0, debug = 1, info = 2, warn = 3, err = 4, cont = 5 };

/// `LOG_LEVEL_DEBUG`, `LOG_LEVEL_TRACE`, `LOG_LEVEL_INFO`, `LOG_LEVEL_WARN`,
/// `LOG_LEVEL_ERROR`, `LOG_LEVEL_OUTPUT` (common/log.h:24, 25, 26, 27, 28, 29 @cd959bae9).
pub const level_debug: c_int = 5;
pub const level_trace: c_int = 4;
pub const level_info: c_int = 3;
pub const level_warn: c_int = 2;
pub const level_error: c_int = 1;
pub const level_output: c_int = 0;

/// Ports `common_log_verbosity_thold` (common/log.cpp:17 @cd959bae9):
/// `LOG_DEFAULT_LLAMA`.
pub var verbosity_thold: c_int = level_info;

/// Ports `common_log_jsonl` (common/log.cpp:27 @cd959bae9): `--log-jsonl`.
var jsonl: bool = false;

/// Ports `common_log_get_jsonl` (common/log.cpp:29 @cd959bae9).
pub fn getJsonl() bool {
    return jsonl;
}

/// Ports `common_log_set_jsonl` (common/log.cpp:33 @cd959bae9): every message is
/// then one JSON object on stdout.
pub fn setJsonl(on: bool) void {
    jsonl = on;
}

/// The logger's settings, `common_log`'s members.
var colors: ?bool = null;
var paused: bool = false;
var file: ?*anyopaque = null;
/// Ports `stdo` (common/log.cpp:194 @cd959bae9), a member of `common_log`:
/// whether messages go to stdout and stderr; set to true by its constructor
/// (common/log.cpp:163 @cd959bae9).
var stdo: bool = true;
var prefix: bool = false;
var timestamps: bool = false;
var t_start: i64 = 0;

/// Ports `g_col` (common/log.cpp:55 @cd959bae9): default, bold, red, green,
/// yellow, blue, magenta, cyan, white, when colours are on.
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
/// (common/log.cpp:464 @cd959bae9).
fn useColors() bool {
    if (colors == null) setColorsImpl(ttyCanUseColors());
    return colors.?;
}

/// Ports `set_colors` (common/log.cpp:419 @cd959bae9), a member of `common_log`:
/// colours only where the console renders them.
fn setColorsImpl(on: bool) void {
    colors = on and ttyEnableAnsi();
}

/// Ports `get_colors` (common/log.cpp:415 @cd959bae9), a member of `common_log`,
/// through `common_log_get_colors` (common/log.cpp:531 @cd959bae9).
pub fn getColors() bool {
    main();
    return useColors();
}

extern fn fileno(stream: *anyopaque) c_int;
extern fn _fileno(stream: *anyopaque) c_int;

/// Ports `common_is_tty` (common/common.cpp:1063 @cd959bae9): whether
/// `file` is attached to a terminal.
pub fn isTty(stream: *anyopaque) bool {
    if (comptime builtin.os.tag == .windows) return _isatty(_fileno(stream)) != 0;
    return isatty(fileno(stream)) != 0;
}

const win = struct {
    const HANDLE = ?*anyopaque;
    extern "kernel32" fn GetStdHandle(id: u32) callconv(.winapi) HANDLE;
    extern "kernel32" fn GetConsoleMode(h: HANDLE, mode: *u32) callconv(.winapi) c_int;
    extern "kernel32" fn SetConsoleMode(h: HANDLE, mode: u32) callconv(.winapi) c_int;
    const STD_OUTPUT_HANDLE: u32 = @bitCast(@as(i32, -11));
    const STD_ERROR_HANDLE: u32 = @bitCast(@as(i32, -12));
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
};

/// Ports `tty_enable_ansi` (common/common.cpp:1091 @cd959bae9): false when
/// stdout or stderr is a console that cannot render ANSI sequences.
pub fn ttyEnableAnsi() bool {
    if (comptime builtin.os.tag == .windows) {
        // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
        for ([_]u32{ win.STD_OUTPUT_HANDLE, win.STD_ERROR_HANDLE }) |id| {
            const h = win.GetStdHandle(id);
            var mode: u32 = 0;
            if (win.GetConsoleMode(h, &mode) != 0 and win.SetConsoleMode(h, mode | win.ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0) {
                return false;
            }
        }
    }
    return true;
}

/// Ports `tty_can_use_colors` (common/common.cpp:1071 @cd959bae9).
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
    // We check both because log messages can go to either
    return isTty(io.cStdout()) or isTty(io.cStderr());
}

/// Ports `common_log_set_colors` (common/log.cpp:516 @cd959bae9) for an explicit
/// choice; `null` is `LOG_COLORS_AUTO`.
pub fn setColors(on: ?bool) void {
    main();
    setColorsImpl(on orelse ttyCanUseColors());
}

/// Ports `common_log_pause` (common/log.cpp:481 @cd959bae9): messages logged while
/// paused are discarded.
pub fn pause() void {
    main();
    paused = true;
}

/// Ports `common_log_resume` (common/log.cpp:485 @cd959bae9): messages are written
/// again. (`resume` is a Zig keyword.)
pub fn unpause() void {
    main();
    paused = false;
}

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: *anyopaque) c_int;

/// Ports `common_log_set_file` (common/log.cpp:508 @cd959bae9): every message is
/// also written to `path`.
pub fn setFile(path: ?[*:0]const u8) void {
    main();
    if (file) |f| _ = fclose(f);
    file = if (path) |p| fopen(p, "w") else null;
}

/// Ports `common_log_set_stdo` and `set_stdo` (common/log.cpp:512, 407
/// @cd959bae9), a member of `common_log`: whether messages go to stdout
/// and stderr. The C++ pauses and resumes the worker around the change, so
/// the logger runs afterwards.
pub fn setStdo(enabled: bool) void {
    main();
    stdo = enabled;
    paused = false; // `resume()`
}

/// Ports `common_log_set_prefix` (common/log.cpp:535 @cd959bae9).
pub fn setPrefix(on: bool) void {
    main();
    prefix = on;
}

/// Ports `common_log_set_timestamps` (common/log.cpp:539 @cd959bae9).
pub fn setTimestamps(on: bool) void {
    main();
    timestamps = on;
}

/// `t_us`, the system clock in microseconds (common/log.cpp:37 @cd959bae9).
fn tUs() i64 {
    return @import("clock.zig").realtimeUs();
}

/// `common_log_main`'s first call constructs the logger, which starts its
/// clock (common/log.cpp:464 @cd959bae9); every entry point here is such a call.
fn main() void {
    if (t_start == 0) t_start = tUs();
}

/// Ports `common_log_add` and `add` (common/log.cpp:493, 239 @cd959bae9), a member
/// of `common_log`, one message, written at once. Whether it is written as
/// JSON is decided here, when it is logged, as the C records it in the
/// entry.
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

    const entry: Entry = .{ .level = level, .msg = msg, .timestamp = timestamp, .jsonl = jsonl };
    printEntry(entry);
}

/// Ports `common_log_add_json` and `add_json` (common/log.cpp:500, 299
/// @cd959bae9), a member of `common_log`: one `LOG_JSON` record, already
/// serialized, wrapped as `{"type":...,"data":...}`, on stdout.
///
/// Parameters:
/// - `kind`: the record's `type`.
/// - `data`: the record's `data`, JSON text.
pub fn addJson(kind: [*:0]const u8, data: []const u8) void {
    // `LOG_JSON` checks the flag before `common_log_main()`, and
    // `common_log_add_json` again after it
    if (!jsonl) return;
    main();
    // discard messages while the worker thread is paused
    if (paused) return;
    const a = std.heap.c_allocator;
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    var full = json.Object.begin(&text, a) catch return;
    full.string("type", std.mem.span(kind)) catch return;
    full.raw("data", data) catch return;
    full.end() catch return;
    text.append(a, 0) catch return;

    const entry: Entry = .{ .level = .none, .msg = @ptrCast(text.items.ptr), .timestamp = 0, .jsonl = true, .is_json = true };
    printEntry(entry);
}

/// Ports `print_entry` (common/log.cpp:207 @cd959bae9), a member of `common_log`.
fn printEntry(entry: Entry) void {
    if (stdo) {
        print(entry, null);
    }
    if (file) |f| print(entry, f);
}

/// Ports `level_str` (common/log.cpp:67 @cd959bae9).
fn levelStr(level: Level) []const u8 {
    return switch (level) {
        .debug => "debug",
        .info => "info",
        .warn => "warn",
        .err => "error",
        .cont => "cont",
        else => "none",
    };
}

/// Ports `common_log_entry` (common/log.cpp:78 @cd959bae9), the members `print`
/// reads.
const Entry = struct {
    level: Level,
    msg: [*:0]const u8,
    timestamp: i64,
    jsonl: bool = false,
    /// msg already holds a serialized JSON object
    is_json: bool = false,
};

/// Ports `print` (common/log.cpp:91 @cd959bae9), a member of `common_log_entry`: to `to`, or to
/// `stdout` or `stderr` by level -- or, as JSONL, to `stdout` whatever the
/// level. A warning's, error's or debug message's colour reset goes before
/// its trailing newlines, so that every line carries its own colours.
fn print(entry: Entry, to: ?*anyopaque) void {
    const level = entry.level;
    const msg = entry.msg;
    const timestamp = entry.timestamp;
    var fcur: *anyopaque = undefined;
    if (to) |t| {
        fcur = t;
    } else {
        // stderr displays DBG messages only when their verbosity level is not
        // higher than the threshold
        if (level == .debug and verbosity_thold < level_debug) return;
        fcur = if (level != .none and !entry.jsonl) io.cStderr() else io.cStdout();
    }

    if (entry.jsonl) {
        if (entry.is_json) {
            fprintfTo(fcur, "%s\n", .{msg});
            _ = fflush(fcur);
            return;
        }

        const a = std.heap.c_allocator;
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(a);
        (struct {
            fn build(t: *std.ArrayList(u8), al: std.mem.Allocator, e: Entry) !void {
                var obj = try json.Object.begin(t, al);
                try obj.string("type", "log");
                try obj.int("time", e.timestamp);
                try obj.string("level", levelStr(e.level));
                try obj.string("msg", std.mem.span(e.msg));
                try obj.end();
                try t.append(al, 0);
            }
        }).build(&text, a, entry) catch return;
        fprintfTo(fcur, "%s\n", .{@as([*:0]const u8, @ptrCast(text.items.ptr))});
        _ = fflush(fcur);
        return;
    }

    // One message is one `fputs`: upstream's single worker thread writes a
    // whole entry before the next, and here every thread writes its own, so
    // a prefix written by one call could be split from its text by another
    // thread's line (a server's HTTP thread against its decode loop). Each
    // stdio call holds the stream's lock; one call per line keeps lines whole.
    var head: [96]u8 = undefined;
    var n_head: usize = 0;
    if (level != .none and level != .cont and prefix) {
        if (timestamp != 0) {
            n_head = @min(head.len - 1, @as(usize, @intCast(@max(0, snprintf(&head, head.len, "%s%d.%02d.%03d.%03d%s ", col(col_blue), @as(c_int, @intCast(@divTrunc(timestamp, 1000000 * 60))), @as(c_int, @intCast(@mod(@divTrunc(timestamp, 1000000), 60))), @as(c_int, @intCast(@mod(@divTrunc(timestamp, 1000), 1000))), @as(c_int, @intCast(@mod(timestamp, 1000))), col(col_default))))));
        }
        const rest = head[n_head..];
        const n: c_int = switch (level) {
            .info => snprintf(rest.ptr, rest.len, "%sI %s", col(col_green), col(col_default)),
            .warn => snprintf(rest.ptr, rest.len, "%sW %s", col(col_magenta), @as([*:0]const u8, "")),
            .err => snprintf(rest.ptr, rest.len, "%sE %s", col(col_red), @as([*:0]const u8, "")),
            .debug => snprintf(rest.ptr, rest.len, "%sD %s", col(col_yellow), @as([*:0]const u8, "")),
            else => 0,
        };
        n_head = @min(head.len - 1, n_head + @as(usize, @intCast(@max(0, n))));
    }
    // the reset goes before the trailing newlines, so that every line carries its own colors
    const reset = level == .warn or level == .err or level == .debug;
    const reset_str: []const u8 = if (reset) std.mem.span(col(col_default)) else "";
    const whole = std.mem.span(msg);
    var end = whole.len;
    while (end > 0 and whole[end - 1] == '\n') end -= 1;

    const a = std.heap.c_allocator;
    if (std.mem.concatWithSentinel(a, u8, &.{ head[0..n_head], whole[0..end], reset_str, whole[end..] }, 0)) |line| {
        defer a.free(line);
        _ = fputs(line.ptr, fcur);
    } else |_| {
        head[n_head] = 0;
        _ = fputs(@ptrCast(&head), fcur);
        fprintfTo(fcur, "%.*s%s%s", .{ @as(c_int, @intCast(end)), msg, if (reset) col(col_default) else @as([*:0]const u8, ""), msg + end });
    }

    _ = fflush(fcur);
}

extern fn fprintf(stream: *anyopaque, format: [*:0]const u8, ...) c_int;

fn fprintfTo(stream: *anyopaque, comptime fmt: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, fprintf, .{ stream, fmt } ++ args);
}

/// Ports `LOG_TMPL` (common/log.h:112 @cd959bae9): `LOGV`, `LOG_DBGV` and the other
/// `*V` macros are this with an explicit verbosity.
pub fn tmpl(level: Level, verbosity: c_int, comptime fmt: [*:0]const u8, args: anytype) void {
    if (verbosity <= verbosity_thold) add(level, fmt, args);
}

/// Ports `LOG` (common/log.h:119 @cd959bae9): output, on stdout.
pub fn out(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.none, level_output, fmt, args);
}

/// Ports `LOG_DBG` (common/log.h:122 @cd959bae9).
pub fn dbg(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.debug, level_debug, fmt, args);
}

/// Ports `LOG_TRC` (common/log.h:123 @cd959bae9).
pub fn trc(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.info, level_trace, fmt, args);
}

/// Ports `LOG_INF` (common/log.h:124 @cd959bae9).
pub fn inf(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.info, level_info, fmt, args);
}

/// Ports `LOG_WRN` (common/log.h:125 @cd959bae9).
pub fn wrn(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.warn, level_warn, fmt, args);
}

/// Ports `LOG_ERR` (common/log.h:126 @cd959bae9).
pub fn err(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.err, level_error, fmt, args);
}

/// Ports `LOG_CNT` (common/log.h:127 @cd959bae9).
pub fn cnt(comptime fmt: [*:0]const u8, args: anytype) void {
    tmpl(.cont, level_info, fmt, args);
}

/// Ports `COM_ERR` (common/common.h:40 @cd959bae9): `LOG_ERR` with the calling
/// function's name, as `cmn  %12.*s: `.
pub fn comErr(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    err("cmn  %12.*s: " ++ fmt, .{ @as(c_int, 12), func } ++ args);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
