//! `llama.h`'s logging entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-impl.cpp` (v0.4.1, `b29c606e2`):
//! `llama_log_get` and `llama_log_set`, its two unmangled exports. The
//! logger they configure is `log.zig`.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const log = @import("../log.zig");

const c = impl.c;

/// Ports `llama_log_get` (llama-impl.cpp:28 @b29c606e2): what ggml's logger
/// holds, which `llama_log_set` keeps equal to libllama's own.
pub fn llama_log_get(log_callback: [*c]c.ggml_log_callback, user_data: [*c]?*anyopaque) callconv(.c) void {
    c.ggml_log_get(log_callback, user_data);
}

/// Ports `llama_log_set` (llama-impl.cpp:32 @b29c606e2): ggml's logger and
/// libllama's both, a null callback restoring the default.
pub fn llama_log_set(log_callback: c.ggml_log_callback, user_data: ?*anyopaque) callconv(.c) void {
    c.ggml_log_set(log_callback, user_data);
    log.g_logger_state.log_callback = log_callback orelse log.callbackDefault;
    log.g_logger_state.log_callback_user_data = user_data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = log;
}
