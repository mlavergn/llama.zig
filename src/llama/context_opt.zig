//! Training a model through its inference context: marking its weights as
//! parameters, and running an epoch of a dataset through `ggml-opt`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-context.cpp` (v0.3.0, `c1d0e7a00`):
//! `llama_set_param` and `llama_context`'s `opt_init`, `opt_epoch_iter` and
//! `opt_epoch`. They are methods of `llama_context` in the C++; here they
//! are free functions over `*Context`, the same split as
//! `context_state.zig`. The C API over them is `api/opt.zig`.
//!
//! # What reaches ggml
//!
//! Everything here drives `ggml-opt.h` through its C ABI — the optimizer
//! context lives in `Context.opt_ctx` — so these functions are the same
//! whether `ggml-opt` is the C++ or `src/ggml/opt.zig`.
//!
//! # `llama_layer` is walked in the C++'s member order
//!
//! `opt_init` hands every F32 tensor of every layer to the caller's
//! `param_filter`, walking `llama_layer` as an array of pointers, so the
//! order of those calls is the order of the C++ struct's members. The
//! port's `Layer` keeps only the members the ported architectures read, in
//! its own order; `layer_param_order` below lists them in the C++'s, and a
//! comptime check fails the build if `Layer` gains a member it does not
//! list. A member the port lacks is null on every model the port can load,
//! and the C++ skips a null one before calling the filter.
//!
//! The same holds for the model-level tensors: of the fourteen the C++
//! offers, the port's `Model` has the eight its architectures use, and the
//! other six are null wherever the port can load a model.

const std = @import("std");
const impl = @import("impl.zig");
const batch_mod = @import("batch.zig");
const memory = @import("memory.zig");
const graph = @import("graph.zig");
const context_mod = @import("context.zig");
const cparams_mod = @import("cparams.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Context = context_mod.Context;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;

/// Mirrors `ctx_type_to_graph_type` (llama-context.cpp:27 @c1d0e7a00),
/// which `context.zig` keeps file-private.
fn ctxTypeToGraphType(ctx_type: c.enum_llama_context_type) graph.GraphType {
    return switch (ctx_type) {
        c.LLAMA_CONTEXT_TYPE_DEFAULT => .default,
        c.LLAMA_CONTEXT_TYPE_MTP => .decoder_mtp,
        else => impl.throwRuntime("llama-context.cpp", 32, "Unsupported ctx type"),
    };
}

/// Ports `llama_set_param` (llama-context.cpp:3291 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: a weight, or null.
/// - `param_filter`: decides whether an F32 tensor is trainable.
/// - `userdata`: the filter's.
///
/// Return: nothing; flags `tensor` as a parameter when it qualifies.
fn setParam(tensor: ?*Tensor, param_filter: c.llama_opt_param_filter, userdata: ?*anyopaque) void {
    const t = tensor orelse return;
    if (t.type != c.GGML_TYPE_F32) return;
    if (!param_filter.?(t, userdata)) return;
    const name = std.mem.sliceTo(&t.name, 0);
    if (std.mem.eql(u8, name, "token_embd.weight")) return; // FIXME
    if (std.mem.eql(u8, name, "rope_freqs.weight")) return; // FIXME
    c.ggml_set_param(t);
}

/// The port's `Layer` members in the order of `llama_layer`'s
/// (llama-model.h:249 @c1d0e7a00), `nextn` standing for the nested
/// `llama_layer_nextn`, walked in its own member order.
const layer_param_order = [_][]const u8{
    "attn_norm",          "attn_norm_b",      "attn_q_norm",         "attn_k_norm",       "attn_post_norm",
    "ssm_norm",           "wq",               "wk",                  "wv",                "wo",
    "wqkv",               "wq_b",             "wk_b",                "wv_b",              "wqkv_b",
    "wo_b",               "wqkv_gate",        "ffn_gate",            "ffn_down",          "ffn_up",
    "ffn_gate_inp",       "ffn_gate_exps",    "ffn_down_exps",       "ffn_up_exps",       "ffn_gate_exps_s",
    "ffn_down_exps_s",    "ffn_up_exps_s",    "ffn_gate_shexp",      "ffn_down_shexp",    "ffn_up_shexp",
    "ssm_in",             "ssm_dt",           "ssm_out",             "ssm_conv1d",        "ssm_a",
    "ssm_alpha",          "rope_long",        "rope_short",          "rope_freqs",        "wq_s",
    "wk_s",               "wv_s",             "wo_s",                "wqkv_s",            "wqkv_gate_s",
    "ffn_gate_s",         "ffn_up_s",         "ffn_down_s",          "ffn_gate_shexp_s",  "ffn_up_shexp_s",
    "ffn_down_shexp_s",   "ssm_in_s",         "ssm_out_s",           "ssm_alpha_s",       "ssm_beta_s",
    "wq_in_s",            "wk_in_s",          "wv_in_s",             "wo_in_s",           "wqkv_in_s",
    "wqkv_gate_in_s",     "ffn_gate_in_s",    "ffn_up_in_s",         "ffn_down_in_s",     "ffn_gate_exps_in_s",
    "ffn_down_exps_in_s", "ffn_up_exps_in_s", "ffn_gate_shexp_in_s", "ffn_up_shexp_in_s", "ffn_down_shexp_in_s",
    "ssm_in_in_s",        "ssm_out_in_s",     "ssm_alpha_in_s",      "ssm_beta_in_s",     "ssm_beta",
    "nextn",
};

comptime {
    @setEvalBranchQuota(100_000);
    // Every member of `Layer` is listed exactly once, or a tensor the C++
    // offers to the filter would be skipped here.
    const names = @typeInfo(model_mod.Layer).@"struct".field_names;
    if (names.len != layer_param_order.len) @compileError("layer_param_order does not cover Layer");
    for (names) |n| {
        var found = false;
        for (layer_param_order) |o| {
            if (std.mem.eql(u8, n, o)) found = true;
        }
        if (!found) @compileError("Layer." ++ n ++ " is missing from layer_param_order");
    }
}

/// Ports `opt_init` (llama-context.cpp:3307 @c1d0e7a00), a method of
/// `llama_context`.
///
/// Parameters:
/// - `self`: the context; it has no optimizer context yet.
/// - `model`: the model whose weights become parameters; its
///   `n_ctx_train` is overwritten.
/// - `lopt_params`: the training parameters.
///
/// Return: nothing.
pub fn optInit(self: *Context, model: *Model, lopt_params: c.struct_llama_opt_params) void {
    impl.assert(self.opt_ctx == null, "llama-context.cpp", 3308, "!opt_ctx");
    model.hparams.n_ctx_train = if (lopt_params.n_ctx_train > 0) lopt_params.n_ctx_train else self.nCtx();
    const n_batch = @min(self.nBatch(), model.hparams.n_ctx_train);
    const n_ubatch = @min(self.nUbatch(), n_batch);
    impl.assert(model.hparams.n_ctx_train % n_batch == 0, "llama-context.cpp", 3312, "model->hparams.n_ctx_train % n_batch == 0");
    impl.assert(n_batch % n_ubatch == 0, "llama-context.cpp", 3313, "n_batch % n_ubatch == 0");

    var opt_params = c.ggml_opt_default_params(self.sched, c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period = @intCast(n_batch / n_ubatch);
    opt_params.get_opt_pars = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer = lopt_params.optimizer_type;
    self.opt_ctx = c.ggml_opt_init(opt_params);

    const param_filter = lopt_params.param_filter;
    const param_filter_ud = lopt_params.param_filter_ud;

    // The C++ offers type_embd, pos_embd, tok_norm, tok_norm_b,
    // output_norm_b, output_b and output_norm_enc too, in this order around
    // the ones below; no ported architecture creates them.
    //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    setParam(model.output_norm, param_filter, param_filter_ud);
    setParam(model.output, param_filter, param_filter_ud);
    setParam(model.cls, param_filter, param_filter_ud);
    setParam(model.cls_b, param_filter, param_filter_ud);
    setParam(model.cls_out, param_filter, param_filter_ud);
    setParam(model.cls_out_b, param_filter, param_filter_ud);
    setParam(model.cls_norm, param_filter, param_filter_ud);

    for (model.layers) |*layer| {
        inline for (layer_param_order) |name| {
            if (comptime std.mem.eql(u8, name, "nextn")) {
                inline for (@typeInfo(model_mod.LayerNextn).@"struct".field_names) |n| {
                    setParam(@field(layer.nextn, n), param_filter, param_filter_ud);
                }
            } else {
                setParam(@field(layer, name), param_filter, param_filter_ud);
            }
        }
    }
}

/// Ports `opt_epoch_iter` (llama-context.cpp:3348 @c1d0e7a00), a method of
/// `llama_context`.
///
/// Runs one datapoint — a context's worth of tokens — through the model in
/// `n_batch` batches of `n_ubatch` ubatches, one optimizer evaluation per
/// ubatch, with one-hot labels built from `labels_sparse`.
///
/// Parameters:
/// - `self`: the context, after `optInit`.
/// - `dataset`: passed through to `callback`.
/// - `result`: accumulates each evaluation.
/// - `tokens`: the datapoint's `n_ctx` input tokens.
/// - `labels_sparse`: the token each position should predict.
/// - `batch`: a batch of `n_batch` tokens to fill.
/// - `callback`: called after each evaluation; may be null.
/// - `train`: backward passes and optimizer steps, or forward only.
/// - `idata_in_loop`: ubatches of this section before this datapoint.
/// - `ndata_in_loop`: ubatches in this section.
/// - `t_loop_start`: when this section started.
///
/// Return: nothing.
fn optEpochIter(
    self: *Context,
    dataset: c.ggml_opt_dataset_t,
    result: c.ggml_opt_result_t,
    tokens: []const c.llama_token,
    labels_sparse: []const c.llama_token,
    batch: *c.llama_batch,
    callback: c.ggml_opt_epoch_callback,
    train: bool,
    idata_in_loop: i64,
    ndata_in_loop: i64,
    t_loop_start: i64,
) void {
    const func = "opt_epoch_iter";
    impl.assert(self.opt_ctx != null, "llama-context.cpp", 3359, "opt_ctx");
    const n_ctx: u32 = self.model.hparams.n_ctx_train;
    const n_batch = @min(self.nBatch(), n_ctx);
    const n_ubatch = @min(self.nUbatch(), n_batch);

    const mem = self.mem.?;
    mem.clear(true);

    var pos_ctx: u32 = 0;
    while (pos_ctx < n_ctx) : (pos_ctx += n_batch) {
        batch.n_tokens = @intCast(n_batch);
        for (0..n_batch) |pos_batch| {
            batch.token[pos_batch] = tokens[pos_ctx + pos_batch];
            batch.pos[pos_batch] = @intCast(pos_ctx + pos_batch);
            batch.n_seq_id[pos_batch] = 1;
            batch.seq_id[pos_batch][0] = 0;
            batch.logits[pos_batch] = 1;
        }

        const n_seq_max: u32 = if (self.cparams.kv_unified) cparams_mod.max_seq else self.cparams.n_seq_max;
        if (!self.balloc.initBatch(batch.*, &self.model.vocab, null, self.model.hparams.nEmbdInp(), n_seq_max, true)) {
            impl.logError("%s: failed to initialize batch\n", .{func});
            return;
        }

        const n_tokens_all = self.balloc.getNTokens();

        self.n_queued_tokens += n_tokens_all;

        for (self.embd_seq.items) |*e| e.v.deinit(allocator);
        self.embd_seq.clearRetainingCapacity();

        const n_outputs_all: u32 = n_tokens_all;

        const mctx = mem.initBatch(self.balloc, self.cparams.n_ubatch, true);
        defer mctx.destroy();
        if (mctx.getStatus() != .success) {
            impl.logError("%s: could not initialize batch\n", .{func});
            break;
        }

        // reserve output buffer
        if (self.outputReserve(@intCast(n_outputs_all)) < n_outputs_all) {
            impl.logError("%s: could not reserve space for batch with %d outputs\n", .{ func, n_outputs_all });
            impl.abort("llama-context.cpp", 3398, "TODO: handle this error");
        }

        var pos_batch: u32 = 0;
        while (true) {
            const ubatch = mctx.getUbatch();

            self.n_outputs = ubatch.n_tokens;

            if (!mctx.apply()) {
                impl.logError("%s: failed to update the memory context\n", .{func});
                break;
            }

            const res = self.gf_res_prev.?;

            var gparams = self.graphParams(res, ubatch, mctx, ctxTypeToGraphType(self.cparams.ctx_type));
            defer gparams.samplers.deinit();

            res.reset();

            const gf = self.model.buildGraph(&gparams);

            var ctx_compute_opt: ?*c.ggml_context = undefined;
            {
                const size_gf: usize = @intCast(c.ggml_graph_size(gf));
                const size_meta = 4 * size_gf * c.ggml_tensor_overhead() + 2 * c.ggml_graph_overhead_custom(size_gf, true);
                const params: c.ggml_init_params = .{
                    .mem_size = size_meta,
                    .mem_buffer = null,
                    .no_alloc = true,
                };
                ctx_compute_opt = c.ggml_init(params);
            }
            c.ggml_opt_prepare_alloc(self.opt_ctx, ctx_compute_opt, gf, res.getInpTokens(), res.getLogits());
            c.ggml_opt_alloc(self.opt_ctx, train);

            res.setInputs(ubatch);
            {
                const labels: *Tensor = c.ggml_opt_labels(self.opt_ctx);
                impl.assert(labels.ne[1] == n_ubatch, "llama-context.cpp", 3437, "labels->ne[1] == n_ubatch");
                _ = c.ggml_set_zero(labels);
                const onef: f32 = 1.0;
                for (0..n_ubatch) |pos_ubatch| {
                    const ilabel = pos_ctx + pos_batch + pos_ubatch;
                    impl.assert(labels_sparse[ilabel] < labels.ne[0], "llama-context.cpp", 3442, "labels_sparse[ilabel] < labels->ne[0]");
                    const off: usize = @intCast(@as(i64, @intCast(pos_ubatch)) * labels.ne[0] + labels_sparse[ilabel]);
                    c.ggml_backend_tensor_set(labels, &onef, off * @sizeOf(f32), @sizeOf(f32));
                }
            }
            c.ggml_opt_eval(self.opt_ctx, result);
            if (callback) |cb| {
                cb(train, self.opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch) / n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            c.ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
            if (!mctx.next()) break;
        }
    }
}

/// Ports `opt_epoch` (llama-context.cpp:3457 @c1d0e7a00), a method of
/// `llama_context`.
///
/// Parameters:
/// - `self`: the context, after `optInit`.
/// - `dataset`: datapoints of `n_ctx` tokens, labels the same.
/// - `result_train`: accumulates the training datapoints.
/// - `result_eval`: accumulates the evaluation datapoints.
/// - `idata_split`: datapoints before this train, the rest evaluate.
/// - `callback_train`: called after each training evaluation; may be null.
/// - `callback_eval`: called after each evaluation-only one; may be null.
///
/// Return: nothing.
pub fn optEpoch(
    self: *Context,
    dataset: c.ggml_opt_dataset_t,
    result_train: c.ggml_opt_result_t,
    result_eval: c.ggml_opt_result_t,
    idata_split: i64,
    callback_train: c.ggml_opt_epoch_callback,
    callback_eval: c.ggml_opt_epoch_callback,
) void {
    const n_ctx = self.nCtx();
    const n_batch = @min(self.cparams.n_batch, n_ctx);
    const n_ubatch = @min(self.cparams.n_ubatch, n_batch);
    const ndata = c.ggml_opt_dataset_ndata(dataset);

    impl.assert(idata_split >= 0, "llama-context.cpp", 3469, "idata_split >= 0");
    impl.assert(idata_split <= ndata, "llama-context.cpp", 3470, "idata_split <= ndata");

    const ubatch_per_ctx: u32 = n_ctx / n_ubatch;

    var batch = batch_mod.batchInit(@intCast(n_batch), 0, 1);
    defer batch_mod.batchFree(batch);
    const tokens = allocator.alloc(c.llama_token, n_ctx) catch oom();
    defer allocator.free(tokens);
    const labels_sparse = allocator.alloc(c.llama_token, n_ctx) catch oom();
    defer allocator.free(labels_sparse);
    @memset(tokens, 0);
    @memset(labels_sparse, 0);

    var idata: i64 = 0;

    var t_loop_start = c.ggml_time_us();
    var ndata_in_loop: i64 = idata_split * ubatch_per_ctx;
    while (idata < idata_split) : (idata += 1) {
        const train = true;
        const idata_in_loop = idata * ubatch_per_ctx;

        c.ggml_opt_dataset_get_batch_host(dataset, tokens.ptr, n_ctx * @sizeOf(c.llama_token), labels_sparse.ptr, idata);
        optEpochIter(self, dataset, result_train, tokens, labels_sparse, &batch, callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = c.ggml_time_us();
    ndata_in_loop = (ndata - idata_split) * ubatch_per_ctx;
    while (idata < ndata) : (idata += 1) {
        const train = false;
        const idata_in_loop = (idata - idata_split) * ubatch_per_ctx;

        c.ggml_opt_dataset_get_batch_host(dataset, tokens.ptr, n_ctx * @sizeOf(c.llama_token), labels_sparse.ptr, idata);
        optEpochIter(self, dataset, result_eval, tokens, labels_sparse, &batch, callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }
}

fn oom() noreturn {
    impl.abort("llama-context.cpp", 0, "out of memory");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "layer_param_order names every Layer member once" {
    for (layer_param_order, 0..) |a, i| {
        for (layer_param_order[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a, b));
    }
}
