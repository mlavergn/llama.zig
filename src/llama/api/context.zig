//! `llama.h`'s context, memory, decode and perf entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-context.cpp` (v1.0.0, `b1b770d90`):
//! the C API at the end of the file, from `llama_context_default_params`
//! through `llama_get_ctx_other`, except the state API
//! (`api/state.zig`), the adapters (`api/adapter.zig`) and training
//! (`api/opt.zig`).
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const llama = @import("../module.zig");

/// The port's context behind a `llama_context *`.
pub fn ctxOf(ctx: ?*const llama.impl.llama_c.struct_llama_context) *llama.context.Context {
    return @ptrCast(@alignCast(@constCast(ctx.?)));
}

/// The port's model behind a `llama_model *`.
pub fn modelOf(model: ?*const llama.impl.llama_c.struct_llama_model) *llama.model.Model {
    return @ptrCast(@alignCast(@constCast(model.?)));
}

/// The port's memory behind a `llama_memory_t`, a pointer to the
/// interface the context holds.
fn memOf(mem: llama.impl.llama_c.llama_memory_t) ?*llama.memory.Memory {
    return @ptrCast(@alignCast(mem));
}

/// Ports `llama_context_default_params` (llama-context.cpp:4415 @b1b770d90).
pub fn llama_context_default_params() callconv(.c) llama.impl.llama_c.llama_context_params {
    return llama.context.defaultParams();
}

/// Ports `llama_init_from_model` (llama-context.cpp:4460 @b1b770d90).
pub fn llama_init_from_model(model: ?*llama.impl.llama_c.struct_llama_model, params: llama.impl.llama_c.llama_context_params) callconv(.c) ?*llama.impl.llama_c.struct_llama_context {
    const m: ?*llama.model.Model = if (model) |p| modelOf(p) else null;
    return @ptrCast(llama.context.initFromModel(m, params));
}

/// Ports `llama_free` (llama-context.cpp:4606 @b1b770d90).
pub fn llama_free(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) void {
    if (ctx != null) ctxOf(ctx).destroy();
}

/// Ports `llama_n_ctx` (llama-context.cpp:4610 @b1b770d90).
pub fn llama_n_ctx(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nCtx();
}

/// Ports `llama_n_ctx_seq` (llama-context.cpp:4614 @b1b770d90).
pub fn llama_n_ctx_seq(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nCtxSeq();
}

/// Ports `llama_n_batch` (llama-context.cpp:4618 @b1b770d90).
pub fn llama_n_batch(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nBatch();
}

/// Ports `llama_n_ubatch` (llama-context.cpp:4622 @b1b770d90).
pub fn llama_n_ubatch(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nUbatch();
}

/// Ports `llama_n_seq_max` (llama-context.cpp:4626 @b1b770d90).
pub fn llama_n_seq_max(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nSeqMax();
}

/// Ports `llama_n_rs_seq` (llama-context.cpp:4630 @b1b770d90).
pub fn llama_n_rs_seq(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).cparams.n_rs_seq;
}

/// Ports `llama_get_model` (llama-context.cpp:4634 @b1b770d90).
pub fn llama_get_model(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) ?*const llama.impl.llama_c.struct_llama_model {
    return @ptrCast(ctxOf(ctx).getModel());
}

/// Ports `llama_pooling_type` (llama-context.cpp:4638 @b1b770d90).
pub fn llama_pooling_type(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) llama.impl.llama_c.enum_llama_pooling_type {
    return ctxOf(ctx).poolingType();
}

/// Ports `llama_attach_threadpool` (llama-context.cpp:4642 @b1b770d90).
pub fn llama_attach_threadpool(ctx: ?*llama.impl.llama_c.struct_llama_context, threadpool: llama.impl.llama_c.ggml_threadpool_t, threadpool_batch: llama.impl.llama_c.ggml_threadpool_t) callconv(.c) void {
    ctxOf(ctx).attachThreadpool(threadpool, threadpool_batch);
}

/// Ports `llama_detach_threadpool` (llama-context.cpp:4649 @b1b770d90).
pub fn llama_detach_threadpool(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) void {
    ctxOf(ctx).detachThreadpool();
}

/// Ports `llama_set_n_threads` (llama-context.cpp:4653 @b1b770d90).
pub fn llama_set_n_threads(ctx: ?*llama.impl.llama_c.struct_llama_context, n_threads: i32, n_threads_batch: i32) callconv(.c) void {
    ctxOf(ctx).setNThreads(n_threads, n_threads_batch);
}

/// Ports `llama_n_threads` (llama-context.cpp:4657 @b1b770d90).
pub fn llama_n_threads(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) i32 {
    return @bitCast(ctxOf(ctx).nThreads());
}

/// Ports `llama_n_threads_batch` (llama-context.cpp:4661 @b1b770d90).
pub fn llama_n_threads_batch(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) i32 {
    return @bitCast(ctxOf(ctx).nThreadsBatch());
}

/// Ports `llama_set_abort_callback` (llama-context.cpp:4665 @b1b770d90).
pub fn llama_set_abort_callback(ctx: ?*llama.impl.llama_c.struct_llama_context, abort_callback: llama.impl.llama_c.ggml_abort_callback, abort_callback_data: ?*anyopaque) callconv(.c) void {
    ctxOf(ctx).setAbortCallback(abort_callback, abort_callback_data);
}

/// Ports `llama_set_embeddings` (llama-context.cpp:4669 @b1b770d90).
pub fn llama_set_embeddings(ctx: ?*llama.impl.llama_c.struct_llama_context, embeddings: bool) callconv(.c) void {
    ctxOf(ctx).setEmbeddings(embeddings);
}

/// Ports `llama_set_causal_attn` (llama-context.cpp:4673 @b1b770d90).
pub fn llama_set_causal_attn(ctx: ?*llama.impl.llama_c.struct_llama_context, causal_attn: bool) callconv(.c) void {
    ctxOf(ctx).setCausalAttn(causal_attn);
}

/// Ports `llama_get_causal_attn` (llama-context.cpp:4677 @b1b770d90).
pub fn llama_get_causal_attn(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) bool {
    return ctxOf(ctx).getCausalAttn();
}

/// Ports `llama_synchronize` (llama-context.cpp:4681 @b1b770d90).
pub fn llama_synchronize(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) void {
    ctxOf(ctx).synchronize();
}

/// Ports `llama_get_logits` (llama-context.cpp:4685 @b1b770d90).
pub fn llama_get_logits(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getLogits();
}

/// Ports `llama_get_logits_ith` (llama-context.cpp:4691 @b1b770d90): the
/// backend-sampled logits where there are some.
pub fn llama_get_logits_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledLogitsIth(i) orelse self.getLogitsIth(i);
}

/// Ports `llama_get_embeddings` (llama-context.cpp:4705 @b1b770d90).
pub fn llama_get_embeddings(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddings();
}

/// Ports `llama_get_embeddings_ith` (llama-context.cpp:4711 @b1b770d90).
pub fn llama_get_embeddings_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsIth(i);
}

/// Ports `llama_get_embeddings_seq` (llama-context.cpp:4717 @b1b770d90).
pub fn llama_get_embeddings_seq(ctx: ?*llama.impl.llama_c.struct_llama_context, seq_id: llama.impl.llama_c.llama_seq_id) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsSeq(seq_id);
}

/// Ports `llama_set_embeddings_nextn` (llama-context.cpp:4723 @b1b770d90).
pub fn llama_set_embeddings_nextn(ctx: ?*llama.impl.llama_c.struct_llama_context, value: bool, masked: bool) callconv(.c) void {
    ctxOf(ctx).setEmbeddingsNextn(value, masked);
}

/// Ports `llama_set_embeddings_nextn_dev` (llama-context.cpp:4727 @b1b770d90).
pub fn llama_set_embeddings_nextn_dev(ctx: ?*llama.impl.llama_c.struct_llama_context, value: bool) callconv(.c) bool {
    return ctxOf(ctx).setEmbeddingsNextnDev(value);
}

/// Ports `llama_get_embeddings_nextn_rows` (llama-context.cpp:4731 @b1b770d90).
pub fn llama_get_embeddings_nextn_rows(ctx: ?*llama.impl.llama_c.struct_llama_context, row0: i64, n_rows: i64, out: [*c]f32) callconv(.c) bool {
    return ctxOf(ctx).getHNextnRows(row0, n_rows, out);
}

/// Ports `llama_set_embeddings_nextn_rows` (llama-context.cpp:4735 @b1b770d90).
pub fn llama_set_embeddings_nextn_rows(ctx: ?*llama.impl.llama_c.struct_llama_context, row0: i64, n_rows: i64, data: [*c]const f32) callconv(.c) bool {
    return ctxOf(ctx).setHNextnRows(row0, n_rows, data);
}

/// Ports `llama_set_ctx_state_src` (llama-context.cpp:4739 @b1b770d90).
pub fn llama_set_ctx_state_src(ctx: ?*llama.impl.llama_c.struct_llama_context, ctx_src: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) bool {
    return ctxOf(ctx).setCtxStateSrc(if (ctx_src) |p| ctxOf(p) else null);
}

/// Ports `llama_set_embeddings_layer_inp_dev` (llama-context.cpp:4743
/// @b1b770d90).
pub fn llama_set_embeddings_layer_inp_dev(ctx: ?*llama.impl.llama_c.struct_llama_context, value: bool) callconv(.c) void {
    ctxOf(ctx).setEmbeddingsLayerInpDev(value);
}

/// Ports `llama_supports_embd_row` (llama-context.cpp:4747 @b1b770d90).
pub fn llama_supports_embd_row(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) bool {
    return ctxOf(ctx).supportsEmbdRow();
}

/// Ports `llama_set_embeddings_layer_inp` (llama-context.cpp:4751 @b1b770d90).
pub fn llama_set_embeddings_layer_inp(ctx: ?*llama.impl.llama_c.struct_llama_context, lid: u32, value: bool) callconv(.c) void {
    ctxOf(ctx).setEmbeddingsLayerInp(lid, value);
}

/// Ports `llama_set_nextn_layer_offset` (llama-context.cpp:4755 @b1b770d90).
pub fn llama_set_nextn_layer_offset(ctx: ?*llama.impl.llama_c.struct_llama_context, offset: i32) callconv(.c) void {
    ctxOf(ctx).setNextnLayerOffset(offset);
}

/// Ports `llama_get_memory` (llama-context.cpp:4759 @b1b770d90).
pub fn llama_get_memory(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) llama.impl.llama_c.llama_memory_t {
    if (ctx == null) return null;
    const self = ctxOf(ctx);
    if (self.mem) |*m| return @ptrCast(m);
    return null;
}

/// Ports `llama_get_embeddings_nextn` (llama-context.cpp:4767 @b1b770d90).
pub fn llama_get_embeddings_nextn(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsNextn();
}

/// Ports `llama_get_embeddings_nextn_ith` (llama-context.cpp:4773 @b1b770d90).
pub fn llama_get_embeddings_nextn_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsNextnIth(i);
}

/// Ports `llama_get_embeddings_layer_inp` (llama-context.cpp:4779 @b1b770d90).
pub fn llama_get_embeddings_layer_inp(ctx: ?*llama.impl.llama_c.struct_llama_context, lid: u32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsLayerInp(lid);
}

/// Ports `llama_set_sampler` (llama-context.cpp:4785 @b1b770d90).
pub fn llama_set_sampler(ctx: ?*llama.impl.llama_c.struct_llama_context, seq_id: llama.impl.llama_c.llama_seq_id, smpl: ?*llama.impl.llama_c.struct_llama_sampler) callconv(.c) bool {
    return ctxOf(ctx).setSampler(seq_id, smpl);
}

/// Ports `llama_get_sampled_token_ith` (llama-context.cpp:4789 @b1b770d90).
pub fn llama_get_sampled_token_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) llama.impl.llama_c.llama_token {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledTokenIth(i);
}

/// Ports `llama_get_sampled_probs_ith` (llama-context.cpp:4795 @b1b770d90).
pub fn llama_get_sampled_probs_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledProbsIth(i);
}

/// Ports `llama_get_sampled_logits_ith` (llama-context.cpp:4801 @b1b770d90).
pub fn llama_get_sampled_logits_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledLogitsIth(i);
}

/// Ports `llama_get_sampled_candidates_ith` (llama-context.cpp:4807 @b1b770d90).
pub fn llama_get_sampled_candidates_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) [*]llama.impl.llama_c.llama_token {
    const self = ctxOf(ctx);
    self.synchronize();
    return @constCast(self.getSampledCandidatesIth(i));
}

/// Ports `llama_get_sampled_candidates_count_ith` (llama-context.cpp:4813 @b1b770d90).
pub fn llama_get_sampled_candidates_count_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) u32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return @truncate(self.getSampledCandidatesCount(i));
}

/// Ports `llama_get_sampled_logits_count_ith` (llama-context.cpp:4819 @b1b770d90).
pub fn llama_get_sampled_logits_count_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) u32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return @truncate(self.getSampledLogitsCount(i));
}

/// Ports `llama_get_sampled_probs_count_ith` (llama-context.cpp:4825 @b1b770d90).
pub fn llama_get_sampled_probs_count_ith(ctx: ?*llama.impl.llama_c.struct_llama_context, i: i32) callconv(.c) u32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return @truncate(self.getSampledProbsCount(i));
}

/// Ports `llama_memory_clear` (llama-context.cpp:4876 @b1b770d90).
pub fn llama_memory_clear(mem: llama.impl.llama_c.llama_memory_t, data: bool) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.clear(data);
}

/// Ports `llama_memory_seq_rm` (llama-context.cpp:4884 @b1b770d90).
pub fn llama_memory_seq_rm(mem: llama.impl.llama_c.llama_memory_t, seq_id: llama.impl.llama_c.llama_seq_id, p0: llama.impl.llama_c.llama_pos, p1: llama.impl.llama_c.llama_pos) callconv(.c) bool {
    const m = memOf(mem) orelse return true;
    return m.seqRm(seq_id, p0, p1);
}

/// Ports `llama_memory_seq_cp` (llama-context.cpp:4896 @b1b770d90).
pub fn llama_memory_seq_cp(mem: llama.impl.llama_c.llama_memory_t, seq_id_src: llama.impl.llama_c.llama_seq_id, seq_id_dst: llama.impl.llama_c.llama_seq_id, p0: llama.impl.llama_c.llama_pos, p1: llama.impl.llama_c.llama_pos) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqCp(seq_id_src, seq_id_dst, p0, p1);
}

/// Ports `llama_memory_seq_keep` (llama-context.cpp:4909 @b1b770d90).
pub fn llama_memory_seq_keep(mem: llama.impl.llama_c.llama_memory_t, seq_id: llama.impl.llama_c.llama_seq_id) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqKeep(seq_id);
}

/// Ports `llama_memory_seq_add` (llama-context.cpp:4919 @b1b770d90).
pub fn llama_memory_seq_add(mem: llama.impl.llama_c.llama_memory_t, seq_id: llama.impl.llama_c.llama_seq_id, p0: llama.impl.llama_c.llama_pos, p1: llama.impl.llama_c.llama_pos, delta: llama.impl.llama_c.llama_pos) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqAdd(seq_id, p0, p1, delta);
}

/// Ports `llama_memory_seq_div` (llama-context.cpp:4932 @b1b770d90).
pub fn llama_memory_seq_div(mem: llama.impl.llama_c.llama_memory_t, seq_id: llama.impl.llama_c.llama_seq_id, p0: llama.impl.llama_c.llama_pos, p1: llama.impl.llama_c.llama_pos, d: c_int) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqDiv(seq_id, p0, p1, d);
}

/// Ports `llama_memory_seq_pos_min` (llama-context.cpp:4945 @b1b770d90).
pub fn llama_memory_seq_pos_min(mem: llama.impl.llama_c.llama_memory_t, seq_id: llama.impl.llama_c.llama_seq_id) callconv(.c) llama.impl.llama_c.llama_pos {
    const m = memOf(mem) orelse return -1;
    return m.seqPosMin(seq_id);
}

/// Ports `llama_memory_seq_pos_max` (llama-context.cpp:4955 @b1b770d90).
pub fn llama_memory_seq_pos_max(mem: llama.impl.llama_c.llama_memory_t, seq_id: llama.impl.llama_c.llama_seq_id) callconv(.c) llama.impl.llama_c.llama_pos {
    const m = memOf(mem) orelse return -1;
    return m.seqPosMax(seq_id);
}

/// Ports `llama_memory_can_shift` (llama-context.cpp:4965 @b1b770d90).
pub fn llama_memory_can_shift(mem: llama.impl.llama_c.llama_memory_t) callconv(.c) bool {
    const m = memOf(mem) orelse return false;
    return m.getCanShift();
}

/// Ports `llama_encode` (llama-context.cpp:5079 @b1b770d90).
pub fn llama_encode(ctx: ?*llama.impl.llama_c.struct_llama_context, batch: llama.impl.llama_c.llama_batch) callconv(.c) i32 {
    const ret = ctxOf(ctx).encode(batch);
    if (ret != 0) llama.impl.logError("%s: failed to encode, ret = %d\n", .{ "llama_encode", ret });
    return ret;
}

/// Ports `llama_decode` (llama-context.cpp:5090 @b1b770d90).
pub fn llama_decode(ctx: ?*llama.impl.llama_c.struct_llama_context, batch: llama.impl.llama_c.llama_batch) callconv(.c) i32 {
    const ret = ctxOf(ctx).decode(batch);
    if (ret != 0 and ret != 1) llama.impl.logError("%s: failed to decode, ret = %d\n", .{ "llama_decode", ret });
    return ret;
}

/// Ports `llama_process` (llama-context.cpp:5167 @b1b770d90): an extended
/// batch through the context's encoder or decoder. Unlike `llama_encode`
/// and `llama_decode` it logs nothing of its own on failure.
///
/// Return: as `llama_decode`, and -1 for an unknown `type`.
pub fn llama_process(ctx: ?*llama.impl.llama_c.struct_llama_context, @"type": llama.impl.llama_c.enum_llama_process_type, b: ?*llama.impl.llama_c.struct_llama_batch_ext) callconv(.c) i32 {
    const self = ctxOf(ctx);
    const bx: *const llama.batch.BatchExt = @ptrCast(@alignCast(b.?));
    return switch (@"type") {
        llama.impl.llama_c.LLAMA_PROCESS_TYPE_ENCODE => self.encodeExt(bx),
        llama.impl.llama_c.LLAMA_PROCESS_TYPE_DECODE => self.decodeExt(bx),
        else => -1,
    };
}

/// Ports `llama_perf_context` (llama-context.cpp:5105 @b1b770d90).
pub fn llama_perf_context(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) llama.impl.llama_c.llama_perf_context_data {
    if (ctx == null) return std.mem.zeroes(llama.impl.llama_c.llama_perf_context_data);
    return ctxOf(ctx).perfGetData();
}

/// Ports `llama_perf_context_print` (llama-context.cpp:5117 @b1b770d90).
pub fn llama_perf_context_print(ctx: ?*const llama.impl.llama_c.struct_llama_context) callconv(.c) void {
    const func = "llama_perf_context_print";
    const data = llama_perf_context(ctx);
    const t_end_ms = 1e-3 * @as(f64, @floatFromInt(llama.impl.llama_c.ggml_time_us()));
    const n_p_eval: f64 = @floatFromInt(data.n_p_eval);
    const n_eval: f64 = @floatFromInt(data.n_eval);
    llama.impl.logInfo("%s:        load time = %10.2f ms\n", .{ func, data.t_load_ms });
    llama.impl.logInfo("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", .{ func, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / n_p_eval, 1e3 / data.t_p_eval_ms * n_p_eval });
    llama.impl.logInfo("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n", .{ func, data.t_eval_ms, data.n_eval, data.t_eval_ms / n_eval, 1e3 / data.t_eval_ms * n_eval });
    llama.impl.logInfo("%s:       total time = %10.2f ms / %5d tokens\n", .{ func, t_end_ms - data.t_start_ms, data.n_p_eval + data.n_eval });
    llama.impl.logInfo("%s:    graphs reused = %10d\n", .{ func, data.n_reused });
    llama.impl.logInfo("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", .{ func, data.n_reallocs_same, data.n_reallocs_diff });
}

/// Ports `llama_perf_context_reset` (llama-context.cpp:5132 @b1b770d90).
pub fn llama_perf_context_reset(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) void {
    ctxOf(ctx).perfReset();
}

/// Ports `llama_get_memory_breakdown` (llama-context.cpp:5179 @b1b770d90),
/// from `llama-ext.h`. The C++ returns a `std::map` keyed by buffer type,
/// so this has no C signature and is never exported; `cli/fit.zig`, the
/// port of its one caller, calls it as Zig.
///
/// Parameters:
/// - `ctx`: the context.
/// - `gpa`: allocates the returned entries.
///
/// Return: one entry per buffer type, in the map's buffer-type pointer
/// order; the caller frees it with `gpa`.
pub fn llama_get_memory_breakdown(ctx: ?*const llama.impl.llama_c.struct_llama_context, gpa: std.mem.Allocator) ![]llama.context.MemoryBreakdownEntry {
    const mb = ctxOf(ctx).memoryBreakdown();
    defer std.heap.c_allocator.free(mb);
    return gpa.dupe(llama.context.MemoryBreakdownEntry, mb);
}

/// Ports `llama_memory_get_buffers` (llama-memory.cpp:134 @b1b770d90), from
/// `llama-ext.h`. The C++ returns a `std::vector`, so this has no C
/// signature and is never exported; it is called as Zig.
///
/// Parameters:
/// - `mem`: the memory, or null for none.
/// - `gpa`: allocates the returned list.
///
/// Return: the backend buffers a state write of `mem` would touch, in
/// address order -- an empty cache reports none; the caller frees it with
/// `gpa`. A state write that throws escapes as the C++'s exception does.
pub fn llama_memory_get_buffers(mem: llama.impl.llama_c.llama_memory_t, gpa: std.mem.Allocator) ![]llama.impl.llama_c.ggml_backend_buffer_t {
    const m = memOf(mem) orelse return gpa.alloc(llama.impl.llama_c.ggml_backend_buffer_t, 0);
    var exc: llama.impl.Exception = .{};
    const bufs = llama.memory.getBuffers(m.*, &exc) catch exc.terminate("llama-memory.cpp", 0);
    defer std.heap.c_allocator.free(bufs);
    return gpa.dupe(llama.impl.llama_c.ggml_backend_buffer_t, bufs);
}

/// Ports `llama_get_ctx_other` (llama-context.cpp:5183 @b1b770d90): the context whose memory this one shares, if any.
/// `common/speculative.cpp`'s MTP draft compares it with the target.
pub fn llama_get_ctx_other(ctx: ?*llama.impl.llama_c.struct_llama_context) callconv(.c) ?*llama.impl.llama_c.struct_llama_context {
    return @ptrCast(ctxOf(ctx).getCparams().ctx_other);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
