//! libllama's logger: the callback `llama_log_set` installs, and the
//! formatting every `LLAMA_LOG_*` goes through.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-impl.cpp` at v0.4.1 (`b29c606e2`):
//! `llama_logger_state`, `g_logger_state`, `llama_log_internal_v`,
//! `llama_log_internal` and `llama_log_callback_default`. The rest of that
//! file is ported elsewhere: `time_meas` in `impl.zig`, `replace_all` in
//! `vocab/vocab.zig` and `loader.zig`, `llama_format_tensor_shape` and
//! `gguf_kv_to_str` in `loader.zig`; `format` has no counterpart, Zig
//! formats natively. `llama_log_get` and `llama_log_set` are in
//! `api/impl.zig`.
//!
//! `impl.zig`'s `logInfo` and friends call `logInternal`. An in-process
//! harness, which links the C++ beside the port, sets this state through
//! `api.impl.llama_log_set` as well as the C++'s through `llama_log_set`.

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;

/// Ports `llama_logger_state` (llama-impl.cpp:13 @b29c606e2).
pub const LoggerState = struct {
    log_callback: c.ggml_log_callback = callbackDefault,
    log_callback_user_data: ?*anyopaque = null,
};

/// Ports `g_logger_state` (llama-impl.cpp:18 @b29c606e2).
pub var g_logger_state: LoggerState = .{};

extern "c" fn snprintf(buf: [*]u8, size: usize, format: [*:0]const u8, ...) c_int;

/// Ports `llama_log_internal` (llama-impl.cpp:55 @b29c606e2) and
/// `llama_log_internal_v` (llama-impl.cpp:38 @b29c606e2): formats into 128
/// bytes on the stack, and again on the heap when that was too short, then
/// hands the text to the callback.
///
/// The C++ takes a `va_list`; this takes the arguments as a tuple and formats
/// with two `snprintf` calls where the C++ makes two `vsnprintf` calls on a
/// copied list. Zig can call a C variadic on every target but cannot define
/// one on aarch64 Linux, so nothing here is variadic.
///
/// Parameters:
/// - `level`: passed to the callback.
/// - `format`: a printf format.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logInternal(level: c.enum_ggml_log_level, comptime format: [:0]const u8, args: anytype) void {
    var buffer: [128]u8 = undefined;
    const len: c_int = @call(.auto, snprintf, .{ &buffer, @as(usize, 128), format.ptr } ++ args);
    const cb = g_logger_state.log_callback.?;
    if (len < 128) {
        cb(level, &buffer, g_logger_state.log_callback_user_data);
    } else {
        const n: usize = @intCast(len);
        const buffer2 = impl.allocator.alloc(u8, n + 1) catch impl.abort("llama-impl.cpp", 45, "out of memory");
        defer impl.allocator.free(buffer2);
        _ = @call(.auto, snprintf, .{ buffer2.ptr, n + 1, format.ptr } ++ args);
        buffer2[n] = 0;
        cb(level, buffer2.ptr, g_logger_state.log_callback_user_data);
    }
}

/// Ports `llama_log_callback_default` (llama-impl.cpp:62 @b29c606e2): the
/// text to `stderr`, flushed.
///
/// Parameters:
/// - `level`: ignored.
/// - `text`: the message.
/// - `user_data`: ignored.
///
/// Return: nothing.
pub fn callbackDefault(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = user_data;
    _ = c.fputs(text, impl.cStderr());
    _ = c.fflush(impl.cStderr());
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(LoggerState);
}

var test_seen: std.ArrayList(u8) = .empty;
var test_level: c.enum_ggml_log_level = c.GGML_LOG_LEVEL_NONE;

fn testCapture(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    test_level = level;
    const tag: *const u8 = @ptrCast(user_data.?);
    std.debug.assert(tag.* == 'x');
    test_seen.appendSlice(std.testing.allocator, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
}

test "a message past 128 bytes takes the heap path whole" {
    defer test_seen.deinit(std.testing.allocator);
    const saved = g_logger_state;
    defer g_logger_state = saved;
    var tag: u8 = 'x';
    g_logger_state = .{ .log_callback = testCapture, .log_callback_user_data = &tag };

    logInternal(c.GGML_LOG_LEVEL_WARN, "%s=%d\n", .{ @as([*:0]const u8, "short"), @as(c_int, 7) });
    try std.testing.expectEqualStrings("short=7\n", test_seen.items);
    try std.testing.expectEqual(@as(c.enum_ggml_log_level, c.GGML_LOG_LEVEL_WARN), test_level);

    test_seen.clearRetainingCapacity();
    const long: [200:0]u8 = @splat('a');
    logInternal(c.GGML_LOG_LEVEL_INFO, "%s|%d", .{ @as([*:0]const u8, &long), @as(c_int, 12345) });
    try std.testing.expectEqualStrings(&long, test_seen.items[0..200]);
    try std.testing.expectEqualStrings("|12345", test_seen.items[200..]);
}
