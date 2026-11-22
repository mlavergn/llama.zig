//! `llama.h`'s state API: saving and restoring a context, whole or one
//! sequence, to a buffer or a file.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-context.cpp` (v1.0.0, `b1b770d90`):
//! the C API from `llama_state_get_size` through
//! `llama_state_seq_load_file`. Each function is `pub`, not `export`; see
//! `api/context.zig`.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_state_get_size` (llama-context.cpp:4977 @b1b770d90).
pub fn llama_state_get_size(ctx: ?*llama.impl.c.struct_llama_context) callconv(.c) usize {
    return llama.context_state.stateGetSize(llama.api.context.ctxOf(ctx));
}

/// Ports `llama_state_get_data` (llama-context.cpp:4981 @b1b770d90).
pub fn llama_state_get_data(ctx: ?*llama.impl.c.struct_llama_context, dst: [*]u8, size: usize) callconv(.c) usize {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    return llama.context_state.stateGetData(self, dst, size);
}

/// Ports `llama_state_set_data` (llama-context.cpp:4988 @b1b770d90).
pub fn llama_state_set_data(ctx: ?*llama.impl.c.struct_llama_context, src: [*]const u8, size: usize) callconv(.c) usize {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    return llama.context_state.stateSetData(self, src, size);
}

/// Ports `llama_state_load_file` (llama-context.cpp:4994 @b1b770d90).
pub fn llama_state_load_file(ctx: ?*llama.impl.c.struct_llama_context, path_session: [*:0]const u8, tokens_out: [*]llama.impl.c.llama_token, n_token_capacity: usize, n_token_count_out: *usize) callconv(.c) bool {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    var exc: llama.impl.Exception = .{};
    return llama.context_state.stateLoadFile(self, std.mem.span(path_session), tokens_out, n_token_capacity, n_token_count_out, &exc) catch {
        llama.impl.logError("%s: error loading session file: %s\n", .{ "llama_state_load_file", exc.whatZ() });
        return false;
    };
}

/// Ports `llama_state_save_file` (llama-context.cpp:5005 @b1b770d90).
pub fn llama_state_save_file(ctx: ?*llama.impl.c.struct_llama_context, path_session: [*:0]const u8, tokens: [*]const llama.impl.c.llama_token, n_token_count: usize) callconv(.c) bool {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    var exc: llama.impl.Exception = .{};
    return llama.context_state.stateSaveFile(self, std.mem.span(path_session), tokens, n_token_count, &exc) catch {
        llama.impl.logError("%s: error saving session file: %s\n", .{ "llama_state_save_file", exc.whatZ() });
        return false;
    };
}

/// Ports `llama_state_seq_get_size` (llama-context.cpp:5016 @b1b770d90).
pub fn llama_state_seq_get_size(ctx: ?*llama.impl.c.struct_llama_context, seq_id: llama.impl.c.llama_seq_id) callconv(.c) usize {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

/// Ports `llama_state_seq_get_data` (llama-context.cpp:5020 @b1b770d90).
pub fn llama_state_seq_get_data(ctx: ?*llama.impl.c.struct_llama_context, dst: [*]u8, size: usize, seq_id: llama.impl.c.llama_seq_id) callconv(.c) usize {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

/// Ports `llama_state_seq_set_data` (llama-context.cpp:5024 @b1b770d90).
pub fn llama_state_seq_set_data(ctx: ?*llama.impl.c.struct_llama_context, src: [*]const u8, size: usize, seq_id: llama.impl.c.llama_seq_id) callconv(.c) usize {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

/// Ports `llama_state_seq_get_size_ext` (llama-context.cpp:5028 @b1b770d90).
pub fn llama_state_seq_get_size_ext(ctx: ?*llama.impl.c.struct_llama_context, seq_id: llama.impl.c.llama_seq_id, flags: llama.impl.c.llama_state_seq_flags) callconv(.c) usize {
    return llama.context_state.stateSeqGetSize(llama.api.context.ctxOf(ctx), seq_id, flags);
}

/// Ports `llama_state_seq_get_data_ext` (llama-context.cpp:5032 @b1b770d90).
pub fn llama_state_seq_get_data_ext(ctx: ?*llama.impl.c.struct_llama_context, dst: [*]u8, size: usize, seq_id: llama.impl.c.llama_seq_id, flags: llama.impl.c.llama_state_seq_flags) callconv(.c) usize {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    return llama.context_state.stateSeqGetData(self, seq_id, dst, size, flags);
}

/// Ports `llama_state_seq_set_data_ext` (llama-context.cpp:5037 @b1b770d90).
pub fn llama_state_seq_set_data_ext(ctx: ?*llama.impl.c.struct_llama_context, src: [*]const u8, size: usize, seq_id: llama.impl.c.llama_seq_id, flags: llama.impl.c.llama_state_seq_flags) callconv(.c) usize {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    return llama.context_state.stateSeqSetData(self, seq_id, src, size, flags);
}

/// Ports `llama_state_seq_save_file` (llama-context.cpp:5043 @b1b770d90).
pub fn llama_state_seq_save_file(ctx: ?*llama.impl.c.struct_llama_context, filepath: [*:0]const u8, seq_id: llama.impl.c.llama_seq_id, tokens: [*]const llama.impl.c.llama_token, n_token_count: usize) callconv(.c) usize {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    var exc: llama.impl.Exception = .{};
    return llama.context_state.stateSeqSaveFile(self, seq_id, std.mem.span(filepath), tokens, n_token_count, &exc) catch {
        llama.impl.logError("%s: error saving sequence state file: %s\n", .{ "llama_state_seq_save_file", exc.whatZ() });
        return 0;
    };
}

/// Ports `llama_state_seq_load_file` (llama-context.cpp:5054 @b1b770d90).
pub fn llama_state_seq_load_file(ctx: ?*llama.impl.c.struct_llama_context, filepath: [*:0]const u8, dest_seq_id: llama.impl.c.llama_seq_id, tokens_out: ?[*]llama.impl.c.llama_token, n_token_capacity: usize, n_token_count_out: *usize) callconv(.c) usize {
    const self = llama.api.context.ctxOf(ctx);
    self.synchronize();
    var exc: llama.impl.Exception = .{};
    return llama.context_state.stateSeqLoadFile(self, dest_seq_id, std.mem.span(filepath), tokens_out, n_token_capacity, n_token_count_out, &exc) catch {
        llama.impl.logError("%s: error loading sequence state file: %s\n", .{ "llama_state_seq_load_file", exc.whatZ() });
        return 0;
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
