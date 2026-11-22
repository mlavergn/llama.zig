//! `llama.h`'s batch entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-batch.cpp` (v1.0.0, `b1b770d90`):
//! `llama_batch_get_one`, `llama_batch_init` and `llama_batch_free`, the
//! `llama_batch_ext_*` functions of `llama.h`'s extended batch API
//! (`llama_batch_ext_add_embd_row` among them), and
//! `llama_batch_ext_set_decision_order`, which `llama-ext.h` declares with
//! C++ linkage and `exports.zig` therefore leaves unexported. The bodies
//! are `batch.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const llama = @import("../module.zig");

/// The port's batch behind a `llama_batch_ext *`.
pub fn batchOf(b: ?*const llama.impl.llama_c.struct_llama_batch_ext) *llama.batch.BatchExt {
    return @ptrCast(@alignCast(@constCast(b.?)));
}

/// Ports `llama_batch_get_one` (llama-batch.cpp:1133 @b1b770d90).
pub fn llama_batch_get_one(tokens: [*c]llama.impl.llama_c.llama_token, n_tokens: i32) callconv(.c) llama.impl.llama_c.struct_llama_batch {
    return llama.batch.batchGetOne(tokens, n_tokens);
}

/// Ports `llama_batch_init` (llama-batch.cpp:1147 @b1b770d90).
pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) callconv(.c) llama.impl.llama_c.struct_llama_batch {
    return llama.batch.batchInit(n_tokens, embd, n_seq_max);
}

/// Ports `llama_batch_free` (llama-batch.cpp:1177 @b1b770d90).
pub fn llama_batch_free(b: llama.impl.llama_c.struct_llama_batch) callconv(.c) void {
    llama.batch.batchFree(b);
}

/// Ports `llama_batch_ext_init` (llama-batch.cpp:1435 @b1b770d90).
pub fn llama_batch_ext_init(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) ?*llama.impl.llama_c.struct_llama_batch_ext {
    return @ptrCast(llama.api.context.ctxOf(ctx).batchExtCreate());
}

/// Ports `llama_batch_ext_free` (llama-batch.cpp:1439 @b1b770d90).
pub fn llama_batch_ext_free(b: ?*llama.impl.llama_c.struct_llama_batch_ext) callconv(.c) void {
    if (b != null) batchOf(b).destroy();
}

/// Ports `llama_batch_ext_clear` (llama-batch.cpp:1443 @b1b770d90).
pub fn llama_batch_ext_clear(b: ?*llama.impl.llama_c.struct_llama_batch_ext) callconv(.c) void {
    batchOf(b).clear();
}

/// Ports `llama_batch_ext_add` (llama-batch.cpp:1447 @b1b770d90).
pub fn llama_batch_ext_add(b: ?*llama.impl.llama_c.struct_llama_batch_ext, seq_id: llama.impl.llama_c.llama_seq_id) callconv(.c) i32 {
    return batchOf(b).addToken(seq_id);
}

/// Ports `llama_batch_ext_add_token` (llama-batch.cpp:1451 @b1b770d90).
pub fn llama_batch_ext_add_token(b: ?*llama.impl.llama_c.struct_llama_batch_ext, seq_id: llama.impl.llama_c.llama_seq_id, id: llama.impl.llama_c.llama_token) callconv(.c) i32 {
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

/// Ports `llama_batch_ext_add_embd` (llama-batch.cpp:1462 @b1b770d90).
pub fn llama_batch_ext_add_embd(b: ?*llama.impl.llama_c.struct_llama_batch_ext, seq_id: llama.impl.llama_c.llama_seq_id, embd: llama.impl.llama_c.struct_llama_embd) callconv(.c) i32 {
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

/// Ports `llama_batch_ext_add_embd_row` (llama-batch.cpp:1473 @b1b770d90),
/// which `llama-ext.h` declares with C++ linkage.
pub fn llama_batch_ext_add_embd_row(b: ?*llama.impl.llama_c.struct_llama_batch_ext, seq_id: llama.impl.llama_c.llama_seq_id, row: i32) callconv(.c) i32 {
    const self = batchOf(b);
    const idx = self.addToken(seq_id);
    if (idx < 0) {
        return idx;
    }
    if (!self.setTokenEmbdRow(idx, row)) {
        return -2;
    }
    return idx;
}

/// Ports `llama_batch_ext_add_seq` (llama-batch.cpp:1484 @b1b770d90).
pub fn llama_batch_ext_add_seq(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, seq_id: llama.impl.llama_c.llama_seq_id) callconv(.c) bool {
    return batchOf(b).addSeq(idx, seq_id);
}

/// Ports `llama_batch_ext_set_pos` (llama-batch.cpp:1488 @b1b770d90).
pub fn llama_batch_ext_set_pos(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, pos: [*c]const llama.impl.llama_c.llama_pos) callconv(.c) bool {
    return batchOf(b).setTokenPos(idx, pos);
}

/// Ports `llama_batch_ext_set_embd_token` (llama-batch.cpp:1492 @b1b770d90).
pub fn llama_batch_ext_set_embd_token(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, embd: llama.impl.llama_c.struct_llama_embd) callconv(.c) bool {
    return batchOf(b).setTokenEmbd(idx, embd);
}

/// Ports `llama_batch_ext_set_embd_state` (llama-batch.cpp:1496 @b1b770d90):
/// false when the context takes no state embedding (only MTP and LiliCorr
/// contexts do).
pub fn llama_batch_ext_set_embd_state(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, embd: llama.impl.llama_c.struct_llama_embd) callconv(.c) bool {
    return batchOf(b).setTokenState(idx, embd);
}

/// Ports `llama_batch_ext_set_embd_state_row` (llama-batch.cpp:1500
/// @b1b770d90): the state of entry `idx` is row `row` of the `h_nextn`
/// device buffer of the context set with `llama_set_ctx_state_src`.
pub fn llama_batch_ext_set_embd_state_row(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, row: i32) callconv(.c) bool {
    return batchOf(b).setTokenStateRow(idx, row);
}

/// Ports `llama_batch_ext_set_output_embd` (llama-batch.cpp:1504 @b1b770d90).
pub fn llama_batch_ext_set_output_embd(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, value: bool) callconv(.c) bool {
    return batchOf(b).setOutput(idx, value);
}

/// Ports `llama_batch_ext_set_output_logits` (llama-batch.cpp:1508 @b1b770d90).
pub fn llama_batch_ext_set_output_logits(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, value: bool) callconv(.c) bool {
    return batchOf(b).setOutput(idx, value);
}

/// `llama-ext.h`'s `enum llama_decision_order` (llama-ext.h:130
/// @b1b770d90): which entries a joint decision head (clef) reads. A run of
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

/// Ports `llama_batch_ext_set_decision_order` (llama-batch.cpp:1512
/// @b1b770d90), from `llama-ext.h`: C++ linkage, so not exported.
pub fn llama_batch_ext_set_decision_order(b: ?*llama.impl.llama_c.struct_llama_batch_ext, idx: i32, order: llama_decision_order) callconv(.c) bool {
    return batchOf(b).setDecisionOrder(idx, @bitCast(@backingInt(order)));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
