//! `llama.h`'s logging entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-impl.cpp` (v1.0.0, `b1b770d90`):
//! `llama_log_get` and `llama_log_set`, its two unmangled exports. The
//! logger they configure is `log.zig`.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_log_get` (llama-impl.cpp:50 @b1b770d90): what ggml's logger
/// holds, which `llama_log_set` keeps equal to libllama's own.
pub fn llama_log_get(log_callback: [*c]llama.impl.llama_c.ggml_log_callback, user_data: [*c]?*anyopaque) callconv(.c) void {
    llama.impl.llama_c.ggml_log_get(log_callback, user_data);
}

/// Ports `llama_log_set` (llama-impl.cpp:54 @b1b770d90): ggml's logger and
/// libllama's both, a null callback restoring the default.
pub fn llama_log_set(log_callback: llama.impl.llama_c.ggml_log_callback, user_data: ?*anyopaque) callconv(.c) void {
    llama.impl.llama_c.ggml_log_set(log_callback, user_data);
    llama.log.g_logger_state.log_callback = log_callback orelse llama.log.callbackDefault;
    llama.log.g_logger_state.log_callback_user_data = user_data;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = llama.log;
}
