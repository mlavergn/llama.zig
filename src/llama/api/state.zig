//! `llama.h`'s state API: saving and restoring a context, whole or one
//! sequence, to a buffer or a file.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-context.cpp` (v0.4.1, `b29c606e2`):
//! the C API from `llama_get_state_size` through
//! `llama_state_seq_load_file`. Each function is `pub`, not `export`; see
//! `api/context.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const context_state = @import("../context_state.zig");
const api_context = @import("context.zig");

const c = impl.c;
const Exception = impl.Exception;
const ctxOf = api_context.ctxOf;

/// Ports `llama_get_state_size` (llama-context.cpp:4122 @b29c606e2).
pub fn llama_get_state_size(ctx: ?*c.struct_llama_context) callconv(.c) usize {
    return llama_state_get_size(ctx);
}

/// Ports `llama_copy_state_data` (llama-context.cpp:4127 @b29c606e2).
pub fn llama_copy_state_data(ctx: ?*c.struct_llama_context, dst: [*]u8) callconv(.c) usize {
    return llama_state_get_data(ctx, dst, std.math.maxInt(usize));
}

/// Ports `llama_set_state_data` (llama-context.cpp:4132 @b29c606e2).
pub fn llama_set_state_data(ctx: ?*c.struct_llama_context, src: [*]const u8) callconv(.c) usize {
    return llama_state_set_data(ctx, src, std.math.maxInt(usize));
}

/// Ports `llama_load_session_file` (llama-context.cpp:4137 @b29c606e2).
pub fn llama_load_session_file(ctx: ?*c.struct_llama_context, path_session: [*:0]const u8, tokens_out: [*]c.llama_token, n_token_capacity: usize, n_token_count_out: *usize) callconv(.c) bool {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

/// Ports `llama_save_session_file` (llama-context.cpp:4142 @b29c606e2).
pub fn llama_save_session_file(ctx: ?*c.struct_llama_context, path_session: [*:0]const u8, tokens: [*]const c.llama_token, n_token_count: usize) callconv(.c) bool {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

/// Ports `llama_state_get_size` (llama-context.cpp:4148 @b29c606e2).
pub fn llama_state_get_size(ctx: ?*c.struct_llama_context) callconv(.c) usize {
    return context_state.stateGetSize(ctxOf(ctx));
}

/// Ports `llama_state_get_data` (llama-context.cpp:4152 @b29c606e2).
pub fn llama_state_get_data(ctx: ?*c.struct_llama_context, dst: [*]u8, size: usize) callconv(.c) usize {
    const self = ctxOf(ctx);
    self.synchronize();
    return context_state.stateGetData(self, dst, size);
}

/// Ports `llama_state_set_data` (llama-context.cpp:4159 @b29c606e2).
pub fn llama_state_set_data(ctx: ?*c.struct_llama_context, src: [*]const u8, size: usize) callconv(.c) usize {
    const self = ctxOf(ctx);
    self.synchronize();
    return context_state.stateSetData(self, src, size);
}

/// Ports `llama_state_load_file` (llama-context.cpp:4165 @b29c606e2).
pub fn llama_state_load_file(ctx: ?*c.struct_llama_context, path_session: [*:0]const u8, tokens_out: [*]c.llama_token, n_token_capacity: usize, n_token_count_out: *usize) callconv(.c) bool {
    const self = ctxOf(ctx);
    self.synchronize();
    var exc: Exception = .{};
    return context_state.stateLoadFile(self, std.mem.span(path_session), tokens_out, n_token_capacity, n_token_count_out, &exc) catch {
        impl.logError("%s: error loading session file: %s\n", .{ "llama_state_load_file", exc.whatZ() });
        return false;
    };
}

/// Ports `llama_state_save_file` (llama-context.cpp:4176 @b29c606e2).
pub fn llama_state_save_file(ctx: ?*c.struct_llama_context, path_session: [*:0]const u8, tokens: [*]const c.llama_token, n_token_count: usize) callconv(.c) bool {
    const self = ctxOf(ctx);
    self.synchronize();
    var exc: Exception = .{};
    return context_state.stateSaveFile(self, std.mem.span(path_session), tokens, n_token_count, &exc) catch {
        impl.logError("%s: error saving session file: %s\n", .{ "llama_state_save_file", exc.whatZ() });
        return false;
    };
}

/// Ports `llama_state_seq_get_size` (llama-context.cpp:4187 @b29c606e2).
pub fn llama_state_seq_get_size(ctx: ?*c.struct_llama_context, seq_id: c.llama_seq_id) callconv(.c) usize {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

/// Ports `llama_state_seq_get_data` (llama-context.cpp:4191 @b29c606e2).
pub fn llama_state_seq_get_data(ctx: ?*c.struct_llama_context, dst: [*]u8, size: usize, seq_id: c.llama_seq_id) callconv(.c) usize {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

/// Ports `llama_state_seq_set_data` (llama-context.cpp:4195 @b29c606e2).
pub fn llama_state_seq_set_data(ctx: ?*c.struct_llama_context, src: [*]const u8, size: usize, seq_id: c.llama_seq_id) callconv(.c) usize {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

/// Ports `llama_state_seq_get_size_ext` (llama-context.cpp:4199 @b29c606e2).
pub fn llama_state_seq_get_size_ext(ctx: ?*c.struct_llama_context, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) callconv(.c) usize {
    return context_state.stateSeqGetSize(ctxOf(ctx), seq_id, flags);
}

/// Ports `llama_state_seq_get_data_ext` (llama-context.cpp:4203 @b29c606e2).
pub fn llama_state_seq_get_data_ext(ctx: ?*c.struct_llama_context, dst: [*]u8, size: usize, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) callconv(.c) usize {
    const self = ctxOf(ctx);
    self.synchronize();
    return context_state.stateSeqGetData(self, seq_id, dst, size, flags);
}

/// Ports `llama_state_seq_set_data_ext` (llama-context.cpp:4208 @b29c606e2).
pub fn llama_state_seq_set_data_ext(ctx: ?*c.struct_llama_context, src: [*]const u8, size: usize, seq_id: c.llama_seq_id, flags: c.llama_state_seq_flags) callconv(.c) usize {
    const self = ctxOf(ctx);
    self.synchronize();
    return context_state.stateSeqSetData(self, seq_id, src, size, flags);
}

/// Ports `llama_state_seq_save_file` (llama-context.cpp:4214 @b29c606e2).
pub fn llama_state_seq_save_file(ctx: ?*c.struct_llama_context, filepath: [*:0]const u8, seq_id: c.llama_seq_id, tokens: [*]const c.llama_token, n_token_count: usize) callconv(.c) usize {
    const self = ctxOf(ctx);
    self.synchronize();
    var exc: Exception = .{};
    return context_state.stateSeqSaveFile(self, seq_id, std.mem.span(filepath), tokens, n_token_count, &exc) catch {
        impl.logError("%s: error saving sequence state file: %s\n", .{ "llama_state_seq_save_file", exc.whatZ() });
        return 0;
    };
}

/// Ports `llama_state_seq_load_file` (llama-context.cpp:4225 @b29c606e2).
pub fn llama_state_seq_load_file(ctx: ?*c.struct_llama_context, filepath: [*:0]const u8, dest_seq_id: c.llama_seq_id, tokens_out: ?[*]c.llama_token, n_token_capacity: usize, n_token_count_out: *usize) callconv(.c) usize {
    const self = ctxOf(ctx);
    self.synchronize();
    var exc: Exception = .{};
    return context_state.stateSeqLoadFile(self, dest_seq_id, std.mem.span(filepath), tokens_out, n_token_capacity, n_token_count_out, &exc) catch {
        impl.logError("%s: error loading sequence state file: %s\n", .{ "llama_state_seq_load_file", exc.whatZ() });
        return 0;
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
