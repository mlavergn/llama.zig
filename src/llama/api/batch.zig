//! `llama.h`'s batch entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-batch.cpp` (v0.7.0, `620d2225c`):
//! `llama_batch_get_one`, `llama_batch_init` and `llama_batch_free`, the
//! twelve `llama_batch_ext_*` functions of `llama.h`'s extended batch API,
//! and `llama_batch_ext_set_decision_order`, which `llama-ext.h` declares
//! with C++ linkage and `exports.zig` therefore leaves unexported. The
//! bodies are `batch.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const batch = @import("../batch.zig");
const context = @import("context.zig");

const c = impl.c;
const BatchExt = batch.BatchExt;

/// The port's batch behind a `llama_batch_ext *`.
pub fn batchOf(b: ?*const c.struct_llama_batch_ext) *BatchExt {
    return @ptrCast(@alignCast(@constCast(b.?)));
}

/// Ports `llama_batch_get_one` (llama-batch.cpp:1077 @620d2225c).
pub fn llama_batch_get_one(tokens: [*c]c.llama_token, n_tokens: i32) callconv(.c) c.struct_llama_batch {
    return batch.batchGetOne(tokens, n_tokens);
}

/// Ports `llama_batch_init` (llama-batch.cpp:1091 @620d2225c).
pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) callconv(.c) c.struct_llama_batch {
    return batch.batchInit(n_tokens, embd, n_seq_max);
}

/// Ports `llama_batch_free` (llama-batch.cpp:1121 @620d2225c).
pub fn llama_batch_free(b: c.struct_llama_batch) callconv(.c) void {
    batch.batchFree(b);
}

/// Ports `llama_batch_ext_init` (llama-batch.cpp:1342 @620d2225c).
pub fn llama_batch_ext_init(ctx: ?*c.struct_llama_context) callconv(.c) ?*c.struct_llama_batch_ext {
    return @ptrCast(context.ctxOf(ctx).batchExtCreate());
}

/// Ports `llama_batch_ext_free` (llama-batch.cpp:1346 @620d2225c).
pub fn llama_batch_ext_free(b: ?*c.struct_llama_batch_ext) callconv(.c) void {
    if (b != null) batchOf(b).destroy();
}

/// Ports `llama_batch_ext_clear` (llama-batch.cpp:1350 @620d2225c).
pub fn llama_batch_ext_clear(b: ?*c.struct_llama_batch_ext) callconv(.c) void {
    batchOf(b).clear();
}

/// Ports `llama_batch_ext_add` (llama-batch.cpp:1354 @620d2225c).
pub fn llama_batch_ext_add(b: ?*c.struct_llama_batch_ext, seq_id: c.llama_seq_id) callconv(.c) i32 {
    return batchOf(b).addToken(seq_id);
}

/// Ports `llama_batch_ext_add_token` (llama-batch.cpp:1358 @620d2225c).
pub fn llama_batch_ext_add_token(b: ?*c.struct_llama_batch_ext, seq_id: c.llama_seq_id, id: c.llama_token) callconv(.c) i32 {
    const self = batchOf(b);
    const idx = self.addToken(seq_id);
    if (idx < 0) {
        return idx;
    }
    if (!self.setTokenId(idx, id)) {
        return -2;
    }
    return idx;
}

/// Ports `llama_batch_ext_add_embd` (llama-batch.cpp:1369 @620d2225c).
pub fn llama_batch_ext_add_embd(b: ?*c.struct_llama_batch_ext, seq_id: c.llama_seq_id, embd: c.struct_llama_embd) callconv(.c) i32 {
    const self = batchOf(b);
    const idx = self.addToken(seq_id);
    if (idx < 0) {
        return idx;
    }
    if (!self.setTokenEmbd(idx, embd)) {
        return -2;
    }
    return idx;
}

/// Ports `llama_batch_ext_add_seq` (llama-batch.cpp:1380 @620d2225c).
pub fn llama_batch_ext_add_seq(b: ?*c.struct_llama_batch_ext, idx: i32, seq_id: c.llama_seq_id) callconv(.c) bool {
    return batchOf(b).addSeq(idx, seq_id);
}

/// Ports `llama_batch_ext_set_pos` (llama-batch.cpp:1384 @620d2225c).
pub fn llama_batch_ext_set_pos(b: ?*c.struct_llama_batch_ext, idx: i32, pos: [*c]const c.llama_pos) callconv(.c) bool {
    return batchOf(b).setTokenPos(idx, pos);
}

/// Ports `llama_batch_ext_set_embd_token` (llama-batch.cpp:1388 @620d2225c).
pub fn llama_batch_ext_set_embd_token(b: ?*c.struct_llama_batch_ext, idx: i32, embd: c.struct_llama_embd) callconv(.c) bool {
    return batchOf(b).setTokenEmbd(idx, embd);
}

/// Ports `llama_batch_ext_set_embd_state` (llama-batch.cpp:1392 @620d2225c):
/// false when the context takes no state embedding (only MTP and LiliCorr
/// contexts do).
pub fn llama_batch_ext_set_embd_state(b: ?*c.struct_llama_batch_ext, idx: i32, embd: c.struct_llama_embd) callconv(.c) bool {
    return batchOf(b).setTokenState(idx, embd);
}

/// Ports `llama_batch_ext_set_output_embd` (llama-batch.cpp:1396 @620d2225c).
pub fn llama_batch_ext_set_output_embd(b: ?*c.struct_llama_batch_ext, idx: i32, value: bool) callconv(.c) bool {
    return batchOf(b).setOutput(idx, value);
}

/// Ports `llama_batch_ext_set_output_logits` (llama-batch.cpp:1400 @620d2225c).
pub fn llama_batch_ext_set_output_logits(b: ?*c.struct_llama_batch_ext, idx: i32, value: bool) callconv(.c) bool {
    return batchOf(b).setOutput(idx, value);
}

/// `llama-ext.h`'s `enum llama_decision_order` (llama-ext.h:115
/// @620d2225c): which entries a joint decision head (clef) reads. A run of
/// entries with one value is a span, spans are separated by `none`, and an
/// option belongs to the last question before it.
pub const llama_decision_order = enum(c_uint) {
    none = 0,
    question_noul = 1,
    question_choice = 2,
    question_score = 3,
    option = 4,
    _,
};

/// Ports `llama_batch_ext_set_decision_order` (llama-batch.cpp:1404
/// @620d2225c), from `llama-ext.h`: C++ linkage, so not exported.
pub fn llama_batch_ext_set_decision_order(b: ?*c.struct_llama_batch_ext, idx: i32, order: llama_decision_order) callconv(.c) bool {
    return batchOf(b).setDecisionOrder(idx, @bitCast(@backingInt(order)));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
