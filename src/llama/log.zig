//! libllama's logger: the callback `llama_log_set` installs, and the
//! formatting every `LLAMA_LOG_*` goes through.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-impl.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_logger_state`, `g_logger_state`, `llama_log_internal_v`,
//! `llama_log_internal` and `llama_log_callback_default`. The rest of that
//! file is ported elsewhere: `time_meas` in `impl.zig`, `replace_all` in
//! `vocab/vocab.zig` and `loader.zig`, `llama_format_tensor_shape` and
//! `gguf_kv_to_str` in `loader.zig`; `format` has no counterpart, Zig
//! formats natively. `llama_log_get` and `llama_log_set` are in
//! `api/impl.zig`.
//!
//! # Not yet the logger the port uses
//!
//! `impl.zig`'s `logInfo` and friends still call the C++'s mangled
//! `llama_log_internal`, so that the in-process harnesses see the port's
//! messages through the callback the C++ holds. At the swap they route here
//! instead, and this state becomes the only one.

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;

/// Ports `llama_logger_state` (llama-impl.cpp:13 @c1d0e7a00).
pub const LoggerState = struct {
    log_callback: c.ggml_log_callback = callbackDefault,
    log_callback_user_data: ?*anyopaque = null,
};

/// Ports `g_logger_state` (llama-impl.cpp:18 @c1d0e7a00).
pub var g_logger_state: LoggerState = .{};

/// Ports `llama_log_internal_v` (llama-impl.cpp:38 @c1d0e7a00): formats into
/// 128 bytes on the stack, and again on the heap when that was too short.
///
/// Parameters:
/// - `level`: passed to the callback.
/// - `format`: a printf format.
/// - `args`: its arguments; consumed.
///
/// Return: nothing.
pub fn logInternalV(level: c.enum_ggml_log_level, format: [*:0]const u8, args: *std.builtin.VaList) void {
    var args_copy = @cVaCopy(args);
    defer @cVaEnd(&args_copy);
    var buffer: [128]u8 = undefined;
    const len = c.vsnprintf(&buffer, 128, format, args.*);
    const cb = g_logger_state.log_callback.?;
    if (len < 128) {
        cb(level, &buffer, g_logger_state.log_callback_user_data);
    } else {
        const n: usize = @intCast(len);
        const buffer2 = impl.allocator.alloc(u8, n + 1) catch impl.abort("llama-impl.cpp", 45, "out of memory");
        defer impl.allocator.free(buffer2);
        _ = c.vsnprintf(buffer2.ptr, n + 1, format, args_copy);
        buffer2[n] = 0;
        cb(level, buffer2.ptr, g_logger_state.log_callback_user_data);
    }
}

/// Ports `llama_log_internal` (llama-impl.cpp:55 @c1d0e7a00).
///
/// The C++ gives it C++ linkage; it is `pub` with the C calling convention
/// here so the swap can export it under that mangled name.
///
/// Parameters:
/// - `level`: passed to the callback.
/// - `format`: a printf format, followed by its arguments.
///
/// Return: nothing.
pub fn logInternal(level: c.enum_ggml_log_level, format: [*:0]const u8, ...) callconv(.c) void {
    var args = @cVaStart();
    defer @cVaEnd(&args);
    logInternalV(level, format, &args);
}

/// Ports `llama_log_callback_default` (llama-impl.cpp:62 @c1d0e7a00): the
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
    _ = c.fputs(text, c.__stderrp);
    _ = c.fflush(c.__stderrp);
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

    logInternal(c.GGML_LOG_LEVEL_WARN, "%s=%d\n", "short", @as(c_int, 7));
    try std.testing.expectEqualStrings("short=7\n", test_seen.items);
    try std.testing.expectEqual(@as(c.enum_ggml_log_level, c.GGML_LOG_LEVEL_WARN), test_level);

    test_seen.clearRetainingCapacity();
    const long: [200:0]u8 = @splat('a');
    logInternal(c.GGML_LOG_LEVEL_INFO, "%s|%d", @as([*:0]const u8, &long), @as(c_int, 12345));
    try std.testing.expectEqualStrings(&long, test_seen.items[0..200]);
    try std.testing.expectEqualStrings("|12345", test_seen.items[200..]);
}
