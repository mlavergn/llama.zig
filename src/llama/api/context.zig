//! `llama.h`'s context, memory, decode and perf entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-context.cpp` (v0.4.0, `5266f24da`):
//! the C API at the end of the file, from `llama_new_context_with_model`
//! through `llama_get_ctx_other`, except the state API
//! (`api/state.zig`), the adapters (`api/adapter.zig`) and training
//! (`api/opt.zig`).
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const context = @import("../context.zig");
const memory = @import("../memory.zig");
const model_mod = @import("../model.zig");

const c = impl.c;
const Context = context.Context;

/// The port's context behind a `llama_context *`.
pub fn ctxOf(ctx: ?*const c.struct_llama_context) *Context {
    return @ptrCast(@alignCast(@constCast(ctx.?)));
}

/// The port's model behind a `llama_model *`.
pub fn modelOf(model: ?*const c.struct_llama_model) *model_mod.Model {
    return @ptrCast(@alignCast(@constCast(model.?)));
}

/// The port's memory behind a `llama_memory_t`, a pointer to the
/// interface the context holds.
fn memOf(mem: c.llama_memory_t) ?*memory.Memory {
    return @ptrCast(@alignCast(mem));
}

/// Ports `llama_context_default_params` (llama-context.cpp:3615 @5266f24da).
pub fn llama_context_default_params() callconv(.c) c.llama_context_params {
    return context.defaultParams();
}

/// Ports `llama_init_from_model` (llama-context.cpp:3659 @5266f24da).
pub fn llama_init_from_model(model: ?*c.struct_llama_model, params: c.llama_context_params) callconv(.c) ?*c.struct_llama_context {
    const m: ?*model_mod.Model = if (model) |p| modelOf(p) else null;
    return @ptrCast(context.initFromModel(m, params));
}

/// Ports `llama_new_context_with_model` (llama-context.cpp:3764 @5266f24da).
pub fn llama_new_context_with_model(model: ?*c.struct_llama_model, params: c.llama_context_params) callconv(.c) ?*c.struct_llama_context {
    return llama_init_from_model(model, params);
}

/// Ports `llama_free` (llama-context.cpp:3770 @5266f24da).
pub fn llama_free(ctx: ?*c.struct_llama_context) callconv(.c) void {
    if (ctx != null) ctxOf(ctx).destroy();
}

/// Ports `llama_n_ctx` (llama-context.cpp:3774 @5266f24da).
pub fn llama_n_ctx(ctx: ?*const c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nCtx();
}

/// Ports `llama_n_ctx_seq` (llama-context.cpp:3778 @5266f24da).
pub fn llama_n_ctx_seq(ctx: ?*const c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nCtxSeq();
}

/// Ports `llama_n_batch` (llama-context.cpp:3782 @5266f24da).
pub fn llama_n_batch(ctx: ?*const c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nBatch();
}

/// Ports `llama_n_ubatch` (llama-context.cpp:3786 @5266f24da).
pub fn llama_n_ubatch(ctx: ?*const c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nUbatch();
}

/// Ports `llama_n_seq_max` (llama-context.cpp:3790 @5266f24da).
pub fn llama_n_seq_max(ctx: ?*const c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).nSeqMax();
}

/// Ports `llama_n_rs_seq` (llama-context.cpp:3794 @5266f24da).
pub fn llama_n_rs_seq(ctx: ?*const c.struct_llama_context) callconv(.c) u32 {
    return ctxOf(ctx).cparams.n_rs_seq;
}

/// Ports `llama_get_model` (llama-context.cpp:3798 @5266f24da).
pub fn llama_get_model(ctx: ?*const c.struct_llama_context) callconv(.c) ?*const c.struct_llama_model {
    return @ptrCast(ctxOf(ctx).getModel());
}

/// Ports `llama_pooling_type` (llama-context.cpp:3802 @5266f24da).
pub fn llama_pooling_type(ctx: ?*const c.struct_llama_context) callconv(.c) c.enum_llama_pooling_type {
    return ctxOf(ctx).poolingType();
}

/// Ports `llama_attach_threadpool` (llama-context.cpp:3806 @5266f24da).
pub fn llama_attach_threadpool(ctx: ?*c.struct_llama_context, threadpool: c.ggml_threadpool_t, threadpool_batch: c.ggml_threadpool_t) callconv(.c) void {
    ctxOf(ctx).attachThreadpool(threadpool, threadpool_batch);
}

/// Ports `llama_detach_threadpool` (llama-context.cpp:3813 @5266f24da).
pub fn llama_detach_threadpool(ctx: ?*c.struct_llama_context) callconv(.c) void {
    ctxOf(ctx).detachThreadpool();
}

/// Ports `llama_set_n_threads` (llama-context.cpp:3817 @5266f24da).
pub fn llama_set_n_threads(ctx: ?*c.struct_llama_context, n_threads: i32, n_threads_batch: i32) callconv(.c) void {
    ctxOf(ctx).setNThreads(n_threads, n_threads_batch);
}

/// Ports `llama_n_threads` (llama-context.cpp:3821 @5266f24da).
pub fn llama_n_threads(ctx: ?*c.struct_llama_context) callconv(.c) i32 {
    return @bitCast(ctxOf(ctx).nThreads());
}

/// Ports `llama_n_threads_batch` (llama-context.cpp:3825 @5266f24da).
pub fn llama_n_threads_batch(ctx: ?*c.struct_llama_context) callconv(.c) i32 {
    return @bitCast(ctxOf(ctx).nThreadsBatch());
}

/// Ports `llama_set_abort_callback` (llama-context.cpp:3829 @5266f24da).
pub fn llama_set_abort_callback(ctx: ?*c.struct_llama_context, abort_callback: c.ggml_abort_callback, abort_callback_data: ?*anyopaque) callconv(.c) void {
    ctxOf(ctx).setAbortCallback(abort_callback, abort_callback_data);
}

/// Ports `llama_set_embeddings` (llama-context.cpp:3833 @5266f24da).
pub fn llama_set_embeddings(ctx: ?*c.struct_llama_context, embeddings: bool) callconv(.c) void {
    ctxOf(ctx).setEmbeddings(embeddings);
}

/// Ports `llama_set_causal_attn` (llama-context.cpp:3837 @5266f24da).
pub fn llama_set_causal_attn(ctx: ?*c.struct_llama_context, causal_attn: bool) callconv(.c) void {
    ctxOf(ctx).setCausalAttn(causal_attn);
}

/// Ports `llama_set_warmup` (llama-context.cpp:3841 @5266f24da).
pub fn llama_set_warmup(ctx: ?*c.struct_llama_context, warmup: bool) callconv(.c) void {
    ctxOf(ctx).setWarmup(warmup);
}

/// Ports `llama_synchronize` (llama-context.cpp:3845 @5266f24da).
pub fn llama_synchronize(ctx: ?*c.struct_llama_context) callconv(.c) void {
    ctxOf(ctx).synchronize();
}

/// Ports `llama_get_logits` (llama-context.cpp:3849 @5266f24da).
pub fn llama_get_logits(ctx: ?*c.struct_llama_context) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getLogits();
}

/// Ports `llama_get_logits_ith` (llama-context.cpp:3855 @5266f24da): the
/// backend-sampled logits where there are some.
pub fn llama_get_logits_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledLogitsIth(i) orelse self.getLogitsIth(i);
}

/// Ports `llama_get_embeddings` (llama-context.cpp:3869 @5266f24da).
pub fn llama_get_embeddings(ctx: ?*c.struct_llama_context) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddings();
}

/// Ports `llama_get_embeddings_ith` (llama-context.cpp:3875 @5266f24da).
pub fn llama_get_embeddings_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsIth(i);
}

/// Ports `llama_get_embeddings_seq` (llama-context.cpp:3881 @5266f24da).
pub fn llama_get_embeddings_seq(ctx: ?*c.struct_llama_context, seq_id: c.llama_seq_id) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsSeq(seq_id);
}

/// Ports `llama_set_embeddings_nextn` (llama-context.cpp:3887 @5266f24da).
pub fn llama_set_embeddings_nextn(ctx: ?*c.struct_llama_context, value: bool, masked: bool) callconv(.c) void {
    ctxOf(ctx).setEmbeddingsNextn(value, masked);
}

/// Ports `llama_set_embeddings_layer_inp` (llama-context.cpp:3891 @5266f24da).
pub fn llama_set_embeddings_layer_inp(ctx: ?*c.struct_llama_context, lid: u32, value: bool) callconv(.c) void {
    ctxOf(ctx).setEmbeddingsLayerInp(lid, value);
}

/// Ports `llama_set_nextn_layer_offset` (llama-context.cpp:3895 @5266f24da).
pub fn llama_set_nextn_layer_offset(ctx: ?*c.struct_llama_context, offset: i32) callconv(.c) void {
    ctxOf(ctx).setNextnLayerOffset(offset);
}

/// Ports `llama_get_memory` (llama-context.cpp:3899 @5266f24da).
pub fn llama_get_memory(ctx: ?*const c.struct_llama_context) callconv(.c) c.llama_memory_t {
    if (ctx == null) return null;
    const self = ctxOf(ctx);
    if (self.mem) |*m| return @ptrCast(m);
    return null;
}

/// Ports `llama_get_embeddings_nextn` (llama-context.cpp:3907 @5266f24da).
pub fn llama_get_embeddings_nextn(ctx: ?*c.struct_llama_context) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsNextn();
}

/// Ports `llama_get_embeddings_nextn_ith` (llama-context.cpp:3913 @5266f24da).
pub fn llama_get_embeddings_nextn_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsNextnIth(i);
}

/// Ports `llama_get_embeddings_layer_inp` (llama-context.cpp:3919 @5266f24da).
pub fn llama_get_embeddings_layer_inp(ctx: ?*c.struct_llama_context, lid: u32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getEmbeddingsLayerInp(lid);
}

/// Ports `llama_set_sampler` (llama-context.cpp:3925 @5266f24da).
pub fn llama_set_sampler(ctx: ?*c.struct_llama_context, seq_id: c.llama_seq_id, smpl: ?*c.struct_llama_sampler) callconv(.c) bool {
    return ctxOf(ctx).setSampler(seq_id, smpl);
}

/// Ports `llama_get_sampled_token_ith` (llama-context.cpp:3929 @5266f24da).
pub fn llama_get_sampled_token_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) c.llama_token {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledTokenIth(i);
}

/// Ports `llama_get_sampled_probs_ith` (llama-context.cpp:3935 @5266f24da).
pub fn llama_get_sampled_probs_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledProbsIth(i);
}

/// Ports `llama_get_sampled_logits_ith` (llama-context.cpp:3941 @5266f24da).
pub fn llama_get_sampled_logits_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) ?[*]f32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return self.getSampledLogitsIth(i);
}

/// Ports `llama_get_sampled_candidates_ith` (llama-context.cpp:3947 @5266f24da).
pub fn llama_get_sampled_candidates_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) [*]c.llama_token {
    const self = ctxOf(ctx);
    self.synchronize();
    return @constCast(self.getSampledCandidatesIth(i));
}

/// Ports `llama_get_sampled_candidates_count_ith` (llama-context.cpp:3953 @5266f24da).
pub fn llama_get_sampled_candidates_count_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) u32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return @truncate(self.getSampledCandidatesCount(i));
}

/// Ports `llama_get_sampled_logits_count_ith` (llama-context.cpp:3959 @5266f24da).
pub fn llama_get_sampled_logits_count_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) u32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return @truncate(self.getSampledLogitsCount(i));
}

/// Ports `llama_get_sampled_probs_count_ith` (llama-context.cpp:3965 @5266f24da).
pub fn llama_get_sampled_probs_count_ith(ctx: ?*c.struct_llama_context, i: i32) callconv(.c) u32 {
    const self = ctxOf(ctx);
    self.synchronize();
    return @truncate(self.getSampledProbsCount(i));
}

/// Ports `llama_memory_clear` (llama-context.cpp:4016 @5266f24da).
pub fn llama_memory_clear(mem: c.llama_memory_t, data: bool) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.clear(data);
}

/// Ports `llama_memory_seq_rm` (llama-context.cpp:4024 @5266f24da).
pub fn llama_memory_seq_rm(mem: c.llama_memory_t, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) callconv(.c) bool {
    const m = memOf(mem) orelse return true;
    return m.seqRm(seq_id, p0, p1);
}

/// Ports `llama_memory_seq_cp` (llama-context.cpp:4036 @5266f24da).
pub fn llama_memory_seq_cp(mem: c.llama_memory_t, seq_id_src: c.llama_seq_id, seq_id_dst: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqCp(seq_id_src, seq_id_dst, p0, p1);
}

/// Ports `llama_memory_seq_keep` (llama-context.cpp:4049 @5266f24da).
pub fn llama_memory_seq_keep(mem: c.llama_memory_t, seq_id: c.llama_seq_id) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqKeep(seq_id);
}

/// Ports `llama_memory_seq_add` (llama-context.cpp:4059 @5266f24da).
pub fn llama_memory_seq_add(mem: c.llama_memory_t, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos, delta: c.llama_pos) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqAdd(seq_id, p0, p1, delta);
}

/// Ports `llama_memory_seq_div` (llama-context.cpp:4072 @5266f24da).
pub fn llama_memory_seq_div(mem: c.llama_memory_t, seq_id: c.llama_seq_id, p0: c.llama_pos, p1: c.llama_pos, d: c_int) callconv(.c) void {
    const m = memOf(mem) orelse return;
    m.seqDiv(seq_id, p0, p1, d);
}

/// Ports `llama_memory_seq_pos_min` (llama-context.cpp:4085 @5266f24da).
pub fn llama_memory_seq_pos_min(mem: c.llama_memory_t, seq_id: c.llama_seq_id) callconv(.c) c.llama_pos {
    const m = memOf(mem) orelse return -1;
    return m.seqPosMin(seq_id);
}

/// Ports `llama_memory_seq_pos_max` (llama-context.cpp:4095 @5266f24da).
pub fn llama_memory_seq_pos_max(mem: c.llama_memory_t, seq_id: c.llama_seq_id) callconv(.c) c.llama_pos {
    const m = memOf(mem) orelse return -1;
    return m.seqPosMax(seq_id);
}

/// Ports `llama_memory_can_shift` (llama-context.cpp:4105 @5266f24da).
pub fn llama_memory_can_shift(mem: c.llama_memory_t) callconv(.c) bool {
    const m = memOf(mem) orelse return false;
    return m.getCanShift();
}

/// Ports `llama_encode` (llama-context.cpp:4232 @5266f24da).
pub fn llama_encode(ctx: ?*c.struct_llama_context, batch: c.llama_batch) callconv(.c) i32 {
    const ret = ctxOf(ctx).encode(batch);
    if (ret != 0) impl.logError("%s: failed to encode, ret = %d\n", .{ "llama_encode", ret });
    return ret;
}

/// Ports `llama_decode` (llama-context.cpp:4243 @5266f24da).
pub fn llama_decode(ctx: ?*c.struct_llama_context, batch: c.llama_batch) callconv(.c) i32 {
    const ret = ctxOf(ctx).decode(batch);
    if (ret != 0 and ret != 1) impl.logError("%s: failed to decode, ret = %d\n", .{ "llama_decode", ret });
    return ret;
}

/// Ports `llama_perf_context` (llama-context.cpp:4258 @5266f24da).
pub fn llama_perf_context(ctx: ?*const c.struct_llama_context) callconv(.c) c.llama_perf_context_data {
    if (ctx == null) return std.mem.zeroes(c.llama_perf_context_data);
    return ctxOf(ctx).perfGetData();
}

/// Ports `llama_perf_context_print` (llama-context.cpp:4270 @5266f24da).
pub fn llama_perf_context_print(ctx: ?*const c.struct_llama_context) callconv(.c) void {
    const func = "llama_perf_context_print";
    const data = llama_perf_context(ctx);
    const t_end_ms = 1e-3 * @as(f64, @floatFromInt(c.ggml_time_us()));
    const n_p_eval: f64 = @floatFromInt(data.n_p_eval);
    const n_eval: f64 = @floatFromInt(data.n_eval);
    impl.logInfo("%s:        load time = %10.2f ms\n", .{ func, data.t_load_ms });
    impl.logInfo("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", .{ func, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / n_p_eval, 1e3 / data.t_p_eval_ms * n_p_eval });
    impl.logInfo("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n", .{ func, data.t_eval_ms, data.n_eval, data.t_eval_ms / n_eval, 1e3 / data.t_eval_ms * n_eval });
    impl.logInfo("%s:       total time = %10.2f ms / %5d tokens\n", .{ func, t_end_ms - data.t_start_ms, data.n_p_eval + data.n_eval });
    impl.logInfo("%s:    graphs reused = %10d\n", .{ func, data.n_reused });
}

/// Ports `llama_perf_context_reset` (llama-context.cpp:4284 @5266f24da).
pub fn llama_perf_context_reset(ctx: ?*c.struct_llama_context) callconv(.c) void {
    ctxOf(ctx).perfReset();
}

/// Ports `llama_get_memory_breakdown` (llama-context.cpp:4323 @5266f24da),
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
pub fn llama_get_memory_breakdown(ctx: ?*const c.struct_llama_context, gpa: std.mem.Allocator) ![]context.MemoryBreakdownEntry {
    const mb = ctxOf(ctx).memoryBreakdown();
    defer impl.allocator.free(mb);
    return gpa.dupe(context.MemoryBreakdownEntry, mb);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
