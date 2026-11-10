//! A model's inference context: the scheduler, the memory, the outputs,
//! and decoding a batch through them.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-context.h` -- `llama_context`
//! - `llama.cpp/src/llama-context.cpp` -- its constructor and destructor,
//!   `ctx_type_to_graph_type`, `llm_fused_op_probe` and its probes,
//!   `resolve_fused_ops`, `sched_reserve`, `synchronize`, the getters and
//!   setters, `memory_update`, the output accessors, `process_ubatch`,
//!   `encode`, `decode`, `copy_tensor_async_rows`, `needs_raw_logits`,
//!   `output_reserve`, `extract_layer_inputs`, `output_reorder`,
//!   `graph_max_nodes`, `ubatch_prepare_reserve`, `graph_reserve`,
//!   `graph_params`, `graph_compute`, `graph_get_cb`, the perf counters,
//!   `llama_context_default_params` and `llama_init_from_model`
//!
//! All at v0.9.0 (`cd959bae9`). Saving and restoring state is
//! `context_state.zig`'s; training is `context_opt.zig`'s.
//!
//! # What the reference compiles out
//!
//! The output accessors abort on a bad index under `#ifndef NDEBUG` and
//! return null otherwise. The reference is built with `NDEBUG`, so here
//! they return null.

const std = @import("std");
const impl = @import("impl.zig");
const arch_mod = @import("arch.zig");
const cparams_mod = @import("cparams.zig");
const hparams_mod = @import("hparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const graph = @import("graph.zig");
const adapter = @import("adapter.zig");
const model_mod = @import("model.zig");
const context_state = @import("context_state.zig");
const libcxx = @import("libcxx.zig");
const moe_cache_mod = @import("moe_cache.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Cparams = cparams_mod.Cparams;
const Model = model_mod.Model;
const Ubatch = batch.Ubatch;
const Tensor = c.ggml_tensor;
const max_seq = cparams_mod.max_seq;

extern fn _Z27llama_sampler_backend_beginP13llama_sampler(smpl: *c.llama_sampler) void;
extern fn _Z29llama_sampler_backend_n_nodesPK13llama_sampler(smpl: *const c.llama_sampler) u32;

fn oom() noreturn {
    impl.abort("llama-context.cpp", 0, "out of memory");
}

/// Ports `llama_verify_hadamard_graph` (llama-context.cpp:32 @cd959bae9):
/// checks that each folded weight in the graph gets its Hadamard transform,
/// and each latent lookup the inverse. Without it an architecture that
/// skips the transform helpers loads and computes wrong results.
///
/// The C++'s `lookups` is a `std::map` keyed by the node's address, which
/// the last loop walks in that order; so does this one.
///
/// Return: nothing, or `error.Thrown` naming the first offending tensor.
fn verifyHadamardGraph(
    gf: *c.ggml_cgraph,
    rotations: *const model_mod.HadamardRotations,
    inverses: *const model_mod.HadamardRotations,
    moe_cache: ?*const moe_cache_mod.MoeCache,
    exc: *Exception,
) Thrown!void {
    const func = "llama_verify_hadamard_graph";

    // the MoE cache gives the matmul a copy of the expert weights, so the
    // copy needs the transform of its source too
    var forward = rotations.clone(allocator) catch oom();
    defer forward.deinit(allocator);
    if (moe_cache) |mc| {
        var it = rotations.iterator();
        while (it.next()) |e| {
            if (mc.getExperts(e.key_ptr.*)) |cached| {
                const slot = forward.getOrPut(allocator, cached) catch oom();
                if (!slot.found_existing) slot.value_ptr.* = e.value_ptr.*;
            }
        }
    }

    const unwrap = struct {
        fn f(t_: ?*const Tensor) ?*const Tensor {
            var t = t_;
            while (t) |x| {
                if (x.op != c.GGML_OP_RESHAPE and x.op != c.GGML_OP_VIEW) break;
                t = x.src[0];
            }
            return t;
        }
    }.f;
    const name = struct {
        fn f(t: *const Tensor) []const u8 {
            return std.mem.sliceTo(&t.name, 0);
        }
    }.f;

    // the get_rows results of the latent tables
    const Lookup = struct { node: *const Tensor, ok: bool };
    var lookups: std.ArrayList(Lookup) = .empty;
    defer lookups.deinit(allocator);

    const n_nodes = c.ggml_graph_n_nodes(gf);
    var i: c_int = 0;
    while (i < n_nodes) : (i += 1) {
        const node: *const Tensor = c.ggml_graph_node(gf, i);

        if (node.op == c.GGML_OP_GET_ROWS and node.src[0] != null and inverses.contains(node.src[0])) {
            lookups.append(allocator, .{ .node = node, .ok = false }) catch oom();
            continue;
        }

        if (node.op != c.GGML_OP_MUL_MAT and node.op != c.GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node.op == c.GGML_OP_MUL_MAT and node.op_params[1] == c.GGML_HINT_SRC0_IS_HADAMARD) {
            const src1 = unwrap(node.src[1]);
            for (lookups.items) |*lk| {
                if (@as(?*const Tensor, lk.node) == src1) lk.ok = true;
            }
            continue;
        }

        const src0: *const Tensor = node.src[0];
        const t = forward.get(src0) orelse {
            if (inverses.contains(src0)) {
                return exc.throw(.runtime_error, "Hadamard-latent table '{s}' is used as a head without a forward transform", .{name(src0)});
            }
            continue;
        };
        const src = unwrap(node.src[1]);
        const transformed = if (src) |x| x.op == c.GGML_OP_MUL_MAT and
            x.op_params[1] == c.GGML_HINT_SRC0_IS_HADAMARD and
            x.src[0] == t.rot else false;
        if (!transformed) {
            return exc.throw(.runtime_error, "Hadamard-folded weight '{s}' is consumed without its activation transform; this graph's matmul path does not support prism.hadamard folding", .{name(src0)});
        }
    }

    std.mem.sort(Lookup, lookups.items, {}, struct {
        fn lt(_: void, a: Lookup, b: Lookup) bool {
            return @intFromPtr(a.node) < @intFromPtr(b.node);
        }
    }.lt);
    for (lookups.items) |lk| {
        if (lk.ok) continue;
        i = 0;
        while (i < n_nodes) : (i += 1) {
            const n2: *const Tensor = c.ggml_graph_node(gf, i);
            for (n2.src, 0..) |src_, sidx| {
                const src = src_ orelse break;
                if (unwrap(src) == lk.node) {
                    impl.logWarn("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n", .{
                        func,
                        c.ggml_get_name(lk.node),
                        c.ggml_op_name(n2.op),
                        c.ggml_get_name(n2),
                        @as(c_int, @intCast(sidx)),
                        n2.op_params[1],
                    });
                }
            }
        }
        return exc.throw(.runtime_error, "Hadamard-latent table '{s}' is read without the inverse transform", .{name(lk.node.src[0])});
    }
}

/// Ports `ctx_type_to_graph_type` (llama-context.cpp:113 @cd959bae9).
fn ctxTypeToGraphType(ctx_type: c.enum_llama_context_type) graph.GraphType {
    return switch (ctx_type) {
        c.LLAMA_CONTEXT_TYPE_DEFAULT => .default,
        c.LLAMA_CONTEXT_TYPE_MTP => .decoder_mtp,
        else => impl.throwRuntime("llama-context.cpp", 118, "Unsupported ctx type"),
    };
}

/// Mirrors `llm_fused_op_probe` (llama-context.cpp:121 @cd959bae9): a fused
/// op to check the device support of, and the ubatch to check it with.
const FusedOpProbe = struct {
    op: graph.FusedOp,
    name: [:0]const u8,
    n_tokens_per_seq: u32,
};

/// The probes, `llm_fused_op_flash_attn_probe`,
/// `llm_fused_op_flash_attn_kv_rows_probe`, `llm_fused_op_gdn_ar_probe`,
/// `llm_fused_op_gdn_ch_probe`, `llm_fused_op_lid_probe`,
/// `llm_fused_op_dsv4_hc_pre_probe`, `llm_fused_op_dsv4_hc_comb_probe` and
/// `llm_fused_op_dsv4_hc_post_probe`, `llm_fused_op_top_k_mask_probe` (llama-context.cpp:127, 133, 139, 145, 151, 157, 163, 169, 175 @cd959bae9).
const probe_flash_attn: FusedOpProbe = .{ .op = .flash_attn, .name = "Flash Attention", .n_tokens_per_seq = 1 };
const probe_flash_attn_kv_rows: FusedOpProbe = .{ .op = .flash_attn_kv_rows, .name = "Flash Attention over K/V rows", .n_tokens_per_seq = 1 };
const probe_gdn_ar: FusedOpProbe = .{ .op = .gdn_ar, .name = "fused Gated Delta Net (autoregressive)", .n_tokens_per_seq = 1 };
const probe_gdn_ch: FusedOpProbe = .{ .op = .gdn_ch, .name = "fused Gated Delta Net (chunked)", .n_tokens_per_seq = 16 };
const probe_lid: FusedOpProbe = .{ .op = .lightning_indexer, .name = "Lightning Indexer", .n_tokens_per_seq = 1 };
const probe_dsv4_hc_pre: FusedOpProbe = .{ .op = .dsv4_hc_pre, .name = "fused DeepSeek V4 HC pre", .n_tokens_per_seq = 1 };
const probe_dsv4_hc_comb: FusedOpProbe = .{ .op = .dsv4_hc_comb, .name = "fused DeepSeek V4 HC comb", .n_tokens_per_seq = 1 };
const probe_dsv4_hc_post: FusedOpProbe = .{ .op = .dsv4_hc_post, .name = "fused DeepSeek V4 HC post", .n_tokens_per_seq = 1 };
const probe_top_k_mask: FusedOpProbe = .{ .op = .top_k_mask, .name = "fused top-k mask", .n_tokens_per_seq = 1 };

/// `buffer_view` (llama-impl.h:90 @cd959bae9).
pub fn BufferView(comptime T: type) type {
    return struct {
        data: ?[*]T = null,
        size: usize = 0,

        pub fn hasData(self: @This()) bool {
            return self.data != null and self.size > 0;
        }
    };
}

/// One entry of the context's memory breakdown, `llama_memory_breakdown`'s
/// value with its key.
pub const MemoryBreakdownEntry = struct {
    buft: c.ggml_backend_buffer_type_t,
    model: usize = 0,
    context: usize = 0,
    compute: usize = 0,
};

/// Ports `llama_context` (llama-context.h:45 @cd959bae9).
pub const Context = struct {
    model: *const Model,
    cparams: Cparams = .{},
    /// The `std::vector<bool>` behind `cparams.embeddings_layer_inp`.
    embeddings_layer_inp: []bool = &.{},

    cvec: *adapter.Cvec,
    loras: *adapter.Loras,
    cross: graph.Cross = .{},
    mem: ?memory.Memory = null,
    moe_cache: ?*moe_cache_mod.MoeCache = null,

    logits: BufferView(f32) = .{},
    embd: BufferView(f32) = .{},
    embd_nextn: BufferView(f32) = .{},
    embd_layer_inp: []BufferView(f32) = &.{},

    sampling: Sampling = .{},

    /// `embd_seq`: each sequence's pooled embedding, by sequence.
    embd_seq: std.ArrayList(struct { seq_id: i32, v: std.ArrayList(f32) }) = .empty,

    /// Heap-allocated: the sanitized batch points into it.
    balloc: *batch.BatchAllocr,

    /// The number of tensors marked as input during the last graph reserve.
    n_input_tensors: u32 = 0,
    /// The outputs in the current ubatch, or the last logical batch.
    n_outputs: u32 = 0,
    /// A batch token's position to its row in the output buffers.
    output_ids: std.ArrayList(i32) = .empty,
    output_swaps: std.ArrayList(struct { i0: u32, i1: u32 }) = .empty,
    /// The extracted rows' original batch indices, for the layer inputs
    /// and unmasked NextN embeddings, which hold every token row.
    embd_batch_idxs: std.ArrayList(i32) = .empty,

    sched: c.ggml_backend_sched_t = null,
    sched_need_reserve: bool = true,

    /// Whether the reserve graph has a logits tensor, set in `schedReserve`;
    /// true until then, because the first `outputReserve` runs before any
    /// graph is reserved.
    graph_has_logits: bool = true,

    /// The state of `schedCopyExperts`, reset before each graph compute.
    copy_experts: CopyExpertsInfo = .{},

    /// Graph reallocations during inference where the backend assignments
    /// did not change, split by whether the graph size is the same as the
    /// previously scheduled one (unexpected) or different.
    n_reallocs_same: i32 = 0,
    n_reallocs_diff: i32 = 0,

    backend_cpu: c.ggml_backend_t = null,
    backends: std.ArrayList(c.ggml_backend_t) = .empty,

    opt_ctx: c.ggml_opt_context_t = null,

    threadpool: c.ggml_threadpool_t = null,
    threadpool_batch: c.ggml_threadpool_t = null,

    abort_callback: c.ggml_abort_callback = null,
    abort_callback_data: ?*anyopaque = null,

    set_n_threads_fns: std.ArrayList(struct { backend: c.ggml_backend_t, f: SetNThreadsFn }) = .empty,

    backend_ptrs: std.ArrayList(c.ggml_backend_t) = .empty,
    backend_buft: std.ArrayList(c.ggml_backend_buffer_type_t) = .empty,
    /// The expected buffer sizes.
    backend_buf_exp_size: std.ArrayList(usize) = .empty,

    /// Each sequence's state copied on the device, for
    /// `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`.
    mem_storage: context_state.MemStorage = .{},

    /// Separate arenas give batches with and without outputs distinct CUDA
    /// graph cache keys: index 1 for a ubatch with outputs, 0 without.
    gf_res_prev: [2]?*graph.Result = .{ null, null },
    gf_res_reserve: ?*graph.Result = null,

    /// The `gf_res_prev` arena whose graph is built and allocated in the
    /// scheduler now, so the only one a ubatch may reuse.
    gf_res_prev_active: ?*graph.Result = null,

    buf_output: c.ggml_backend_buffer_t = null,

    /// The device buffer with the enabled layer inputs,
    /// `[n_layer_inp*n_embd, n_batch]`, row = batch index.
    ctx_layer_inp_dev: ?*c.ggml_context = null,
    buf_layer_inp_dev: c.ggml_backend_buffer_t = null,
    layer_inp_dev: ?*Tensor = null,
    layer_inp_dev_dev: c.ggml_backend_dev_t = null,
    layer_inp_backend: c.ggml_backend_t = null,

    /// Ordering with the reader context, with no host sync: `written` is
    /// recorded by this context after the batch and waited on by the
    /// reader; `read` is recorded by the reader after it consumed the rows
    /// and waited on by this context.
    ev_layer_inp_written: c.ggml_backend_event_t = null,
    ev_layer_inp_read: c.ggml_backend_event_t = null,
    ev_layer_inp_read_recorded: bool = false,

    has_evaluated_once: bool = false,
    graph_reuse_disable: bool = false,

    /// True after the `prism.hadamard` coverage check passed on a reserved
    /// graph.
    hadamard_verified: bool = false,

    t_start_us: i64 = 0,
    t_load_us: i64 = 0,
    t_p_eval_us: i64 = 0,
    t_eval_us: i64 = 0,
    t_compute_start_us: i64 = 0,
    n_queued_tokens: i64 = 0,
    /// Tokens in prompt eval calls, those with a batch size over 1.
    n_p_eval: i32 = 0,
    n_eval: i32 = 0,
    /// How often the previous graph was reused.
    n_reused: i32 = 0,

    const SetNThreadsFn = *const fn (c.ggml_backend_t, c_int) callconv(.c) void;

    /// `copy_experts_info` (llama-context.h:387 @cd959bae9).
    pub const CopyExpertsInfo = struct {
        ids: ?*const Tensor = null,
        ids_data: std.ArrayList(i32) = .empty,
        used: std.ArrayList(bool) = .empty,

        pub fn reset(self: *CopyExpertsInfo) void {
            self.ids = null;
            self.ids_data.clearRetainingCapacity();
            self.used.clearRetainingCapacity();
        }

        fn deinit(self: *CopyExpertsInfo) void {
            self.ids_data.deinit(allocator);
            self.used.deinit(allocator);
        }
    };

    /// `sampling_info` (llama-context.h:340 @cd959bae9).
    const Sampling = struct {
        samplers: graph.SamplerMap = .{},
        logits: BufferView(f32) = .{},
        sampled: BufferView(c.llama_token) = .{},
        probs: BufferView(f32) = .{},
        candidates: BufferView(c.llama_token) = .{},
        logits_count: std.ArrayList(u32) = .empty,
        probs_count: std.ArrayList(u32) = .empty,
        candidates_count: std.ArrayList(u32) = .empty,
        token_ids_full_vocab: std.ArrayList(c.llama_token) = .empty,
    };

    /// Ports the constructor of `llama_context` (llama-context.cpp:181 @cd959bae9).
    ///
    /// Return: the context, the caller's to `destroy`, or `error.Thrown`.
    pub fn create(model: *const Model, params: c.llama_context_params, exc: *Exception) Thrown!*Context {
        const func = "llama_context";
        const self = allocator.create(Context) catch oom();
        const cvec = allocator.create(adapter.Cvec) catch oom();
        cvec.* = .{};
        const loras = allocator.create(adapter.Loras) catch oom();
        loras.* = .{};
        const balloc = allocator.create(batch.BatchAllocr) catch oom();
        // MTP takes the hidden state through the embedding input
        balloc.* = batch.BatchAllocr.init(model.hparams.nPosPerEmbd(), arch_mod.supportsMixedBatch(model.arch) and params.ctx_type == c.LLAMA_CONTEXT_TYPE_DEFAULT);
        self.* = .{ .model = model, .cvec = cvec, .loras = loras, .balloc = balloc };
        errdefer self.destroy();

        impl.logInfo("%s: constructing llama_context\n", .{func});

        self.t_start_us = model.t_start_us;
        self.t_load_us = model.t_load_us;

        const hparams = &model.hparams;
        const cp = &self.cparams;

        cp.n_seq_max = @max(1, params.n_seq_max);
        if (cp.n_seq_max > max_seq) return exc.throw(.runtime_error, "n_seq_max must be <= {d}", .{max_seq});

        cp.n_rs_seq = params.n_rs_seq;
        if (cp.n_rs_seq > 0 and !arch_mod.supportsRsRollback(model.arch)) {
            impl.logDebug("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n", .{ func, cp.n_rs_seq });
            cp.n_rs_seq = 0;
        }

        cp.n_threads = params.n_threads;
        cp.n_threads_batch = params.n_threads_batch;
        cp.yarn_ext_factor = if (params.yarn_ext_factor >= 0.0) params.yarn_ext_factor else hparams.yarn_ext_factor;
        cp.yarn_attn_factor = if (params.yarn_attn_factor >= 0.0) params.yarn_attn_factor else hparams.yarn_attn_factor;
        cp.yarn_beta_fast = if (params.yarn_beta_fast >= 0.0) params.yarn_beta_fast else hparams.yarn_beta_fast;
        cp.yarn_beta_slow = if (params.yarn_beta_slow >= 0.0) params.yarn_beta_slow else hparams.yarn_beta_slow;
        cp.embeddings = params.embeddings;
        cp.embeddings_nextn = false;
        cp.embeddings_nextn_masked = false;
        cp.offload_kqv = params.offload_kqv;
        cp.no_perf = params.no_perf;
        cp.warmup = false;

        // +1: id n_layer() taps the last layer's output, the head's input.
        self.embeddings_layer_inp = allocator.alloc(bool, hparams.nLayer() + 1) catch oom();
        @memset(self.embeddings_layer_inp, false);
        cp.embeddings_layer_inp = self.embeddings_layer_inp;
        cp.embeddings_layer_inp_dev = false;
        self.embd_layer_inp = allocator.alloc(BufferView(f32), hparams.nLayer() + 1) catch oom();
        @memset(self.embd_layer_inp, .{});

        cp.ctx_type = params.ctx_type;
        cp.rope_scaling_type = params.rope_scaling_type;
        cp.pooling_type = params.pooling_type;

        cp.n_ctx = if (params.n_ctx == 0) hparams.n_ctx_train else params.n_ctx;
        cp.rope_freq_base = if (params.rope_freq_base == 0.0) hparams.rope_freq_base_train else params.rope_freq_base;
        cp.rope_freq_scale = if (params.rope_freq_scale == 0.0) hparams.rope_freq_scale_train else params.rope_freq_scale;

        cp.n_ctx_orig_yarn = if (params.yarn_orig_ctx != 0) params.yarn_orig_ctx else if (hparams.n_ctx_orig_yarn != 0) hparams.n_ctx_orig_yarn else hparams.n_ctx_train;

        cp.cb_eval = params.cb_eval;
        cp.cb_eval_user_data = params.cb_eval_user_data;

        cp.ctx_other = null;

        if (model.arch == .gemma4_assistant) {
            if (params.ctx_other == null) {
                return exc.throw(.runtime_error, "Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)", .{});
            }
            cp.ctx_other = params.ctx_other;
        }
        if (model.arch == .eagle3 or model.arch == .dflash) {
            if (model.tok_embd == null or model.output == null) {
                if (params.ctx_other == null) {
                    return exc.throw(.runtime_error, "{s} requires ctx_other to be set (this warning is normal during memory fitting)", .{model.archName()});
                }
                cp.ctx_other = params.ctx_other;
            } else if (model.arch == .dflash) {
                // to read the target's layer input device buffer
                cp.ctx_other = params.ctx_other;
            }
        }

        if (cp.rope_scaling_type == c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) cp.rope_scaling_type = hparams.rope_scaling_type_train;

        // Never scale when the scaling type is none.
        if (cp.rope_scaling_type == c.LLAMA_ROPE_SCALING_TYPE_NONE) cp.rope_freq_scale = 1.0;

        // Negative means not set.
        if (cp.yarn_ext_factor < 0.0) cp.yarn_ext_factor = if (cp.rope_scaling_type == c.LLAMA_ROPE_SCALING_TYPE_YARN) 1.0 else 0.0;

        if (cp.yarn_ext_factor != 0) {
            const factor: f32 = 1.0 / cp.rope_freq_scale;
            if (hparams.rope_yarn_log_mul != 0.0) {
                // Assumes mscale == 1.
                var mscale: f32 = 1.0;
                const mscale_all_dims = hparams.rope_yarn_log_mul;
                // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
                if (model.arch == .deepseek2 and mscale_all_dims != 1.0) mscale = mscale_all_dims;
                cp.yarn_attn_factor = getMscale(factor, mscale) / getMscale(factor, mscale_all_dims);
                impl.logWarn("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n", .{ func, @as(f64, cp.yarn_attn_factor), @as(f64, mscale), @as(f64, mscale_all_dims) });
            } else {
                cp.yarn_attn_factor = getMscale(factor, 1.0);
            }
            // YaRN with ext_factor != 0 cancels this factor in the kernel.
            cp.yarn_attn_factor *= 1.0 / (1.0 + 0.1 * impl.logf(factor));
        }

        cp.yarn_attn_factor *= hparams.rope_attn_factor;

        if (cp.pooling_type == c.LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cp.pooling_type = if (hparams.pooling_type == c.LLAMA_POOLING_TYPE_UNSPECIFIED) c.LLAMA_POOLING_TYPE_NONE else hparams.pooling_type;
        }

        if (params.attention_type == c.LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
            cp.causal_attn = hparams.causal_attn;
        } else {
            cp.causal_attn = params.attention_type == c.LLAMA_ATTENTION_TYPE_CAUSAL;
        }

        cp.flash_attn = params.flash_attn_type != c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.auto_fa = params.flash_attn_type == c.LLAMA_FLASH_ATTN_TYPE_AUTO;

        // The KV cache is shared with ctx_other, so follow its flash
        // attention setting instead of resolving it here: the probe disables
        // it when the layers and the shared KV are on different devices,
        // which breaks a quantized V cache.
        if (model.arch == .gemma4_assistant) {
            const other: *const Context = @ptrCast(@alignCast(cp.ctx_other.?));
            const flash_attn_other = other.cparams.flash_attn;
            if (!cp.auto_fa and cp.flash_attn != flash_attn_other) {
                impl.logWarn("%s: flash_attn = %d does not match the context the KV cache is shared with, using %d\n", .{ func, @as(c_int, @intFromBool(cp.flash_attn)), @as(c_int, @intFromBool(flash_attn_other)) });
            }
            cp.flash_attn = flash_attn_other;
            cp.auto_fa = false;
        }

        cp.fused_gdn_ar = true;
        cp.fused_gdn_ch = true;
        cp.auto_fgdn = false;

        cp.fused_lid = true;
        cp.auto_flid = false;

        cp.fused_dsv4_hc_pre = true;
        cp.fused_dsv4_hc_comb = true;
        cp.fused_dsv4_hc_post = true;
        cp.auto_fhc = true;

        // only the indexer models build this mask, skip the probe for the others
        cp.fused_top_k_mask = true;
        cp.auto_ftkm = hparams.indexer_top_k > 0;

        // With causal attention, the batch is limited by the context.
        cp.n_batch = if (cp.causal_attn) @min(cp.n_ctx, params.n_batch) else params.n_batch;
        cp.n_ubatch = @min(cp.n_batch, if (params.n_ubatch == 0) params.n_batch else params.n_ubatch);

        cp.n_outputs_max = if (params.n_outputs_max == 0 or model.hasEncoder()) cp.n_batch else params.n_outputs_max;
        cp.n_outputs_max_per_seq = if (params.n_outputs_max_per_seq == 0) cp.n_outputs_max else @min(params.n_outputs_max_per_seq, cp.n_outputs_max);

        // The backend samplers join the sampling graph before the reserve
        // passes below, which avoids a re-reserve later.
        if (params.samplers != null and params.n_samplers > 0) {
            for (params.samplers[0..params.n_samplers]) |config| {
                if (c.llama_sampler_chain_get(config.sampler, -1) == null) {
                    return exc.throw(.runtime_error, "the backend samplers must be of type llama_sampler_chain", .{});
                }
                if (self.setSampler(config.seq_id, config.sampler)) {
                    const n_samplers = c.llama_sampler_chain_n(config.sampler);
                    impl.logInfo("%s: setting backend sampler for seq_id %d (n = %d)\n", .{ func, config.seq_id, n_samplers });
                }
            }
        }

        cp.op_offload = params.op_offload;
        cp.kv_unified = params.kv_unified;
        cp.moe_cache_size = params.moe_cache_size;

        // Resolved after flash attention, in `resolveFusedOps`.
        {
            cp.fused_kv_rows = false;

            if (std.c.getenv("LLAMA_KV_ROWS_DISABLE") == null) {
                cp.auto_fkvr = cp.kv_unified;
            } else {
                impl.logInfo("%s: LLAMA_KV_ROWS_DISABLE set, disabling kv_rows\n", .{func});
                cp.auto_fkvr = false;
            }
        }

        // Set later.
        cp.pipeline_parallel = false;

        if (std.c.getenv("LLAMA_GRAPH_REUSE_DISABLE")) |v| self.graph_reuse_disable = impl.atoi(v) != 0;
        if (self.graph_reuse_disable) impl.logWarn("%s: graph reuse disabled\n", .{func});

        cp.n_ctx = padTo(cp.n_ctx, 256);

        if (cp.kv_unified) {
            cp.n_ctx_seq = cp.n_ctx;
        } else {
            cp.n_ctx_seq = cp.n_ctx / cp.n_seq_max;
            cp.n_ctx_seq = padTo(cp.n_ctx_seq, 256);
            if (cp.n_ctx_seq == 0) return exc.throw(.runtime_error, "n_ctx_seq == 0", .{});
            if (cp.n_ctx != cp.n_ctx_seq * cp.n_seq_max) {
                cp.n_ctx = cp.n_ctx_seq * cp.n_seq_max;
                impl.logWarn("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", .{ func, cp.n_ctx });
            }
        }

        impl.logInfo("%s: n_seq_max             = %u\n", .{ func, cp.n_seq_max });
        impl.logInfo("%s: n_ctx                 = %u\n", .{ func, cp.n_ctx });
        impl.logInfo("%s: n_ctx_seq             = %u\n", .{ func, cp.n_ctx_seq });
        impl.logInfo("%s: n_batch               = %u\n", .{ func, cp.n_batch });
        impl.logInfo("%s: n_ubatch              = %u\n", .{ func, cp.n_ubatch });
        impl.logInfo("%s: causal_attn           = %d\n", .{ func, @as(c_int, @intFromBool(cp.causal_attn)) });
        impl.logInfo("%s: flash_attn            = %s\n", .{ func, flashAttnTypeName(params.flash_attn_type).ptr });
        impl.logInfo("%s: kv_unified            = %s\n", .{ func, @as([*:0]const u8, if (cp.kv_unified) "true" else "false") });
        impl.logInfo("%s: freq_base             = %.1f\n", .{ func, @as(f64, cp.rope_freq_base) });
        impl.logInfo("%s: freq_scale            = %g\n", .{ func, @as(f64, cp.rope_freq_scale) });
        impl.logInfo("%s: n_rs_seq              = %u\n", .{ func, cp.n_rs_seq });
        impl.logInfo("%s: n_outputs_max         = %u\n", .{ func, cp.n_outputs_max });
        impl.logInfo("%s: n_outputs_max_per_seq = %u\n", .{ func, cp.n_outputs_max_per_seq });

        if (cp.n_ctx_seq < hparams.n_ctx_train) {
            impl.logInfo("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n", .{ func, cp.n_ctx_seq, hparams.n_ctx_train });
        }
        if (cp.n_ctx_seq > hparams.n_ctx_train) {
            impl.logWarn("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n", .{ func, cp.n_ctx_seq, hparams.n_ctx_train });
        }

        if (!hparams.vocab_only) {
            // GPU backends.
            for (model.devices.items) |dev| {
                const backend = c.ggml_backend_dev_init(dev.dev, null) orelse
                    return exc.throw(.runtime_error, "failed to initialize {s} backend", .{std.mem.span(c.ggml_backend_dev_name(dev.dev))});
                self.backends.append(allocator, backend) catch oom();
            }
            // The tensors of ctx_other (a shared KV cache, say) can be on
            // devices this model does not use.
            if (cp.ctx_other) |o| {
                const other: *const Context = @ptrCast(@alignCast(o));
                for (other.model.devices.items) |dev_other| {
                    var found = false;
                    for (model.devices.items) |dev| {
                        if (dev.dev == dev_other.dev) found = true;
                    }
                    if (found) {
                        continue;
                    }
                    const backend = c.ggml_backend_dev_init(dev_other.dev, null) orelse
                        return exc.throw(.runtime_error, "failed to initialize {s} backend", .{std.mem.span(c.ggml_backend_dev_name(dev_other.dev))});
                    self.backends.append(allocator, backend) catch oom();
                }
            }
            // ACCEL backends, such as BLAS.
            var i: usize = 0;
            while (i < c.ggml_backend_dev_count()) : (i += 1) {
                const dev = c.ggml_backend_dev_get(i);
                if (c.ggml_backend_dev_type(dev) == c.GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                    const backend = c.ggml_backend_dev_init(dev, null) orelse
                        return exc.throw(.runtime_error, "failed to initialize {s} backend", .{std.mem.span(c.ggml_backend_dev_name(dev))});
                    self.backends.append(allocator, backend) catch oom();
                }
            }
            // The CPU backend.
            self.backend_cpu = c.ggml_backend_init_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU, null) orelse
                return exc.throw(.runtime_error, "failed to initialize CPU backend", .{});
            self.backends.append(allocator, self.backend_cpu) catch oom();

            // The backends' set_n_threads functions.
            for (self.backends.items) |backend| {
                const dev = c.ggml_backend_get_device(backend);
                const reg = if (dev != null) c.ggml_backend_dev_backend_reg(dev) else null;
                if (reg != null) {
                    const f: ?SetNThreadsFn = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")));
                    if (f) |fn_| self.set_n_threads_fns.append(allocator, .{ .backend = backend, .f = fn_ }) catch oom();
                }
            }

            self.setAbortCallback(params.abort_callback, params.abort_callback_data);

            // The graph outputs buffer.
            if (self.outputReserve(@intCast(params.n_seq_max)) < params.n_seq_max) {
                return exc.throw(.runtime_error, "failed to reserve initial output buffer", .{});
            }
            impl.logInfo("%s: %10s  output buffer size = %8.2f MiB\n", .{ func, c.ggml_backend_buffer_name(self.buf_output), @as(f64, @floatFromInt(c.ggml_backend_buffer_get_size(self.buf_output))) / 1024.0 / 1024.0 });
        }

        // The memory module.
        if (!hparams.vocab_only) {
            const other: ?*Context = @ptrCast(@alignCast(cp.ctx_other));
            const params_mem: memory.Params = .{
                .type_k = params.type_k,
                .type_v = params.type_v,
                .swa_full = params.swa_full,
                .ctx_type = cp.ctx_type,
                .mem_other = if (other) |o| o.mem else null,
                .no_alloc = model.noAlloc(),
            };
            self.mem = try model.createMemory(&params_mem, cp, exc);
        }

        // The backends.
        if (!hparams.vocab_only) {
            impl.logDebug("%s: enumerating backends\n", .{func});
            self.backend_buft.clearRetainingCapacity();
            self.backend_ptrs.clearRetainingCapacity();
            self.backend_buf_exp_size.clearRetainingCapacity();

            for (self.backends.items) |backend| {
                var buft = c.ggml_backend_get_default_buffer_type(backend);
                const backend_type = c.ggml_backend_dev_type(c.ggml_backend_get_device(backend));
                if (backend_type == c.GGML_BACKEND_DEVICE_TYPE_CPU and model.devices.items.len != 0) {
                    // The first device's host buffer, for faster transfer
                    // of the intermediate state.
                    if (c.ggml_backend_dev_host_buffer_type(model.devices.items[0].dev)) |host_buft| buft = host_buft;
                }
                self.backend_buft.append(allocator, buft) catch oom();
                self.backend_ptrs.append(allocator, backend) catch oom();
                self.backend_buf_exp_size.append(allocator, 0) catch oom();
            }

            impl.logDebug("%s: backend_ptrs.size() = %zu\n", .{ func, self.backend_ptrs.items.len });

            // Pipeline parallelism costs memory; only when necessary.
            var pipeline_parallel = model.nDevices() > 1 and
                model.nGpuLayers() > model.hparams.n_layer_all and
                model.params.split_mode == c.LLAMA_SPLIT_MODE_LAYER and
                cp.offload_kqv and
                !model.hasTensorOverrides() and
                cp.moe_cache_size == 0; // not supported by the MoE cache

            // It needs async compute and events on every device.
            if (pipeline_parallel) {
                for (self.backends.items) |backend| {
                    const dev = c.ggml_backend_get_device(backend);
                    if (c.ggml_backend_dev_type(dev) == c.GGML_BACKEND_DEVICE_TYPE_CPU) continue;
                    var props: c.ggml_backend_dev_props = undefined;
                    c.ggml_backend_dev_get_props(dev, &props);
                    if (!props.caps.async or !props.caps.events) {
                        pipeline_parallel = false;
                        break;
                    }
                }
            }

            cp.pipeline_parallel = pipeline_parallel;
            if (cp.pipeline_parallel) impl.logInfo("%s: pipeline parallelism enabled\n", .{func});

            if (cp.moe_cache_size > 0) {
                self.moe_cache = try moe_cache_mod.MoeCache.create(model, model.noAlloc(), self.backend_ptrs.items, self.backend_buft.items, cp.moe_cache_size, exc);
            }

            try self.schedReserve(exc);

            if (!cp.flash_attn and c.ggml_is_quantized(params.type_v)) {
                return exc.throw(.runtime_error, "quantized V cache was requested, but this requires Flash Attention", .{});
            }
        }

        // The full vocabulary's token ids, for backend samplers.
        const n_vocab = model.vocab.nTokens();
        self.sampling.token_ids_full_vocab.resize(allocator, n_vocab) catch oom();
        for (self.sampling.token_ids_full_vocab.items, 0..) |*t, i| t.* = @intCast(i);

        return self;
    }

    fn getMscale(scale: f32, mscale: f32) f32 {
        return if (scale <= 1.0) 1.0 else (0.1 * mscale * impl.logf(scale) + 1.0);
    }

    /// Ports the destructor of `llama_context` (llama-context.cpp:638 @cd959bae9).
    ///
    /// Return: nothing.
    pub fn destroy(self: *Context) void {
        // Wait for pending asynchronous copies into the output buffers.
        self.synchronize();

        // Training allocates extra buffers through the scheduler, so the
        // sizes no longer match the expectation.
        if (!self.model.noAlloc() and self.opt_ctx == null and self.sched != null) {
            for (self.backend_ptrs.items, self.backend_buft.items, self.backend_buf_exp_size.items) |backend, buft, size_exp| {
                const size_act = c.ggml_backend_sched_get_buffer_size(self.sched, backend);
                if (size_exp == size_act) {
                    impl.logDebug("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n", .{ "~llama_context", c.ggml_backend_buft_name(buft), @as(f64, @floatFromInt(size_act)) / (1024.0 * 1024.0), @as(f64, @floatFromInt(size_exp)) / (1024.0 * 1024.0) });
                } else {
                    impl.logWarn("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n", .{ "~llama_context", c.ggml_backend_buft_name(buft), @as(f64, @floatFromInt(size_act)) / (1024.0 * 1024.0), @as(f64, @floatFromInt(size_exp)) / (1024.0 * 1024.0) });
                }
            }
        }
        c.ggml_opt_free(self.opt_ctx);

        // The members, in the C++'s reverse declaration order.
        if (self.buf_output != null) c.ggml_backend_buffer_free(self.buf_output);
        if (self.ev_layer_inp_read != null) c.ggml_backend_event_free(self.ev_layer_inp_read);
        if (self.ev_layer_inp_written != null) c.ggml_backend_event_free(self.ev_layer_inp_written);
        if (self.buf_layer_inp_dev != null) c.ggml_backend_buffer_free(self.buf_layer_inp_dev);
        if (self.ctx_layer_inp_dev) |ctx| c.ggml_free(ctx);
        if (self.gf_res_reserve) |r| r.destroy();
        for (self.gf_res_prev) |res| if (res) |r| r.destroy();
        self.mem_storage.deinit();
        self.backend_buf_exp_size.deinit(allocator);
        self.backend_buft.deinit(allocator);
        self.backend_ptrs.deinit(allocator);
        self.set_n_threads_fns.deinit(allocator);
        if (self.sched != null) c.ggml_backend_sched_free(self.sched);
        self.output_swaps.deinit(allocator);
        self.embd_batch_idxs.deinit(allocator);
        self.output_ids.deinit(allocator);
        self.balloc.deinit();
        allocator.destroy(self.balloc);
        for (self.embd_seq.items) |*e| e.v.deinit(allocator);
        self.embd_seq.deinit(allocator);
        self.sampling.samplers.deinit();
        self.sampling.logits_count.deinit(allocator);
        self.sampling.probs_count.deinit(allocator);
        self.sampling.candidates_count.deinit(allocator);
        self.sampling.token_ids_full_vocab.deinit(allocator);
        allocator.free(self.embd_layer_inp);
        self.copy_experts.deinit();
        if (self.mem) |m| m.destroy();
        self.cross.deinit();
        self.loras.deinit();
        allocator.destroy(self.loras);
        self.cvec.deinit();
        allocator.destroy(self.cvec);
        allocator.free(self.embeddings_layer_inp);
        // The backends after the scheduler that used them.
        for (self.backends.items) |b| c.ggml_backend_free(b);
        self.backends.deinit(allocator);
        // The MoE cache after the backends, as the C++ declares it before
        // them: its destructor logs the hit statistics after the backends'
        // own messages.
        if (self.moe_cache) |mc| mc.destroy();
        allocator.destroy(self);
    }

    /// Ports `llama_context`'s `resolve_fused_ops` (llama-context.cpp:662
    /// @cd959bae9): enable a fused op only where its node lands on its
    /// layer's device.
    fn resolveFusedOps(self: *Context, mctx: ?memory.Context, n_seqs: u32, exc: *Exception) Thrown!void {
        const func = "resolve_fused_ops";
        const cp = &self.cparams;
        if (cp.auto_fa) {
            try self.resolveProbe(&probe_flash_attn, &cp.flash_attn, mctx, n_seqs, exc);
            cp.auto_fa = false;
        }
        // The fallback attends to the whole unified cache with a dense mask.
        if (cp.auto_fkvr) {
            cp.fused_kv_rows = cp.flash_attn;
            try self.resolveProbe(&probe_flash_attn_kv_rows, &cp.fused_kv_rows, mctx, n_seqs, exc);
            cp.auto_fkvr = false;
        }
        if (cp.auto_fgdn) {
            impl.logInfo("%s: resolving fused Gated Delta Net support:\n", .{func});
            try self.resolveProbe(&probe_gdn_ar, &cp.fused_gdn_ar, mctx, n_seqs, exc);
            try self.resolveProbe(&probe_gdn_ch, &cp.fused_gdn_ch, mctx, n_seqs, exc);
            cp.auto_fgdn = false;
        }
        if (cp.auto_flid) {
            impl.logInfo("%s: resolving fused Lightning Indexer support:\n", .{func});
            try self.resolveProbe(&probe_lid, &cp.fused_lid, mctx, n_seqs, exc);
            cp.auto_flid = false;
        }
        if (cp.auto_fhc) {
            impl.logInfo("%s: resolving fused DeepSeek V4 HC support:\n", .{func});
            try self.resolveProbe(&probe_dsv4_hc_pre, &cp.fused_dsv4_hc_pre, mctx, n_seqs, exc);
            try self.resolveProbe(&probe_dsv4_hc_comb, &cp.fused_dsv4_hc_comb, mctx, n_seqs, exc);
            try self.resolveProbe(&probe_dsv4_hc_post, &cp.fused_dsv4_hc_post, mctx, n_seqs, exc);
            cp.auto_fhc = false;
        }
        if (cp.auto_ftkm) {
            impl.logInfo("%s: resolving fused top-k mask support:\n", .{func});
            try self.resolveProbe(&probe_top_k_mask, &cp.fused_top_k_mask, mctx, n_seqs, exc);
            cp.auto_ftkm = false;
        }
    }

    /// The `resolve` lambda of `resolve_fused_ops` (llama-context.cpp:662 @cd959bae9).
    fn resolveProbe(self: *Context, probe: *const FusedOpProbe, enabled: *bool, mctx: ?memory.Context, n_seqs: u32, exc: *Exception) Thrown!void {
        const func = "resolve_fused_ops";
        if (!enabled.*) return;
        const n_tokens_probe = probe.n_tokens_per_seq * n_seqs;
        _ = try self.graphReserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true, null, exc) orelse
            return exc.throw(.runtime_error, "failed to reserve graph for {s} check", .{probe.name});

        var device_mismatch = false;
        for (self.getGfResReserve().fused_nodes.items) |node| {
            if (node.op != probe.op) continue;
            impl.assert(node.il >= 0, "llama-context.cpp", 682, "node.il >= 0");
            const backend_fused = c.ggml_backend_sched_get_tensor_backend(self.sched, node.tensor);
            const device_fused = if (backend_fused != null) c.ggml_backend_get_device(backend_fused) else null;
            // model.dev_layer() keeps the current behavior, though it is
            // wrong for cases like --no-kv-offload.
            const device_layer = self.model.devLayer(@intCast(node.il));
            if (device_fused != device_layer) {
                impl.logWarn("%s: layer %d is assigned to device %s but %s is assigned to device %s (usually due to missing support)\n", .{
                    func,
                    node.il,
                    if (device_layer != null) c.ggml_backend_dev_name(device_layer) else @as([*c]const u8, "none"),
                    probe.name.ptr,
                    if (device_fused != null) c.ggml_backend_dev_name(device_fused) else @as([*c]const u8, "none"),
                });
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled.* = false;
            impl.logWarn("%s: %s not supported, set to disabled\n", .{ func, probe.name.ptr });
        } else {
            enabled.* = true;
            impl.logInfo("%s: %s enabled\n", .{ func, probe.name.ptr });
        }
    }

    /// Ports `llama_graph_n_input_tensors` (llama-context.cpp:752 @cd959bae9):
    /// the distinct tensors the graph marks as input, each logged with the
    /// nodes that read it; one that is not a plain `GGML_OP_NONE` leaf
    /// aborts.
    ///
    /// The C++ keys an `std::unordered_map` on the tensor pointer and logs
    /// in its iteration order, which `libcxx.StringMap` reproduces over the
    /// pointer's bytes, as libc++'s `hash<T *>` hashes them. The order is
    /// therefore the addresses', and differs between any two processes.
    ///
    /// Return: the number of distinct input tensors.
    fn graphNInputTensors(gf: *c.ggml_cgraph) c_int {
        const func = "llama_graph_n_input_tensors";
        const Users = std.ArrayList(*c.ggml_tensor);
        var users = libcxx.StringMap(Users).init(allocator) catch oom();
        defer {
            var it = users.iterator();
            while (it.next()) |e| e.value.deinit(allocator);
            users.deinit();
        }
        const Add = struct {
            fn run(u: *libcxx.StringMap(Users), tensor: *c.ggml_tensor, user: *c.ggml_tensor) void {
                const key: ?*const c.ggml_tensor = tensor;
                const slot = u.getOrPut(std.mem.asBytes(&key), .empty) catch oom();
                slot.value.append(allocator, user) catch oom();
            }
        };

        var i: c_int = 0;
        while (i < c.ggml_graph_n_nodes(gf)) : (i += 1) {
            const node: *c.ggml_tensor = c.ggml_graph_node(gf, i);
            if ((node.flags & c.GGML_TENSOR_FLAG_INPUT) != 0) Add.run(&users, node, node);
            for (node.src) |s_| {
                const src: *c.ggml_tensor = s_ orelse break;
                if ((src.flags & c.GGML_TENSOR_FLAG_INPUT) != 0) Add.run(&users, src, node);
            }
        }

        var it = users.iterator();
        while (it.next()) |e| {
            const p: usize = std.mem.readInt(usize, e.key[0..@sizeOf(usize)], .little);
            const tensor: *const c.ggml_tensor = @ptrFromInt(p);
            impl.assert(tensor.op == c.GGML_OP_NONE, "llama-context.cpp", 771, "tensor->op == GGML_OP_NONE");
            for (e.value.items) |node| {
                impl.logDebug("%s: input tensor '%32s' [%s, ne = { %5lld, %5lld, %5lld, %5lld }] is used by node '%s' (%s)\n", .{
                    func,                          c.ggml_get_name(tensor),       c.ggml_type_name(tensor.type),
                    @as(c_longlong, tensor.ne[0]), @as(c_longlong, tensor.ne[1]), @as(c_longlong, tensor.ne[2]),
                    @as(c_longlong, tensor.ne[3]), c.ggml_get_name(node),         c.ggml_op_name(node.op),
                });
            }
        }

        return @intCast(users.count());
    }

    /// Ports `llama_context`'s `sched_new` (llama-context.cpp:783 @cd959bae9):
    /// a new backend scheduler for a graph of `max_nodes`, with the copy and
    /// realloc callbacks registered.
    ///
    /// Return: nothing.
    fn schedNew(self: *Context, max_nodes: usize, parallel: bool) void {
        if (self.sched != null) c.ggml_backend_sched_free(self.sched);
        self.sched = c.ggml_backend_sched_new(self.backend_ptrs.items.ptr, self.backend_buft.items.ptr, @intCast(self.backend_ptrs.items.len), max_nodes, parallel, self.cparams.op_offload);

        c.ggml_backend_sched_set_copy_callback(self.sched, schedCopyExperts, self);
        c.ggml_backend_sched_set_realloc_callback(self.sched, schedGraphRealloc, self);
    }

    /// Ports `llama_context`'s `sched_reserve` (llama-context.cpp:791 @cd959bae9):
    /// the worst-case graphs, so the compute buffers are allocated once.
    pub fn schedReserve(self: *Context, exc: *Exception) Thrown!void {
        const func = "sched_reserve";
        if (!self.sched_need_reserve) return;
        self.sched_need_reserve = false;

        impl.logInfo("%s: reserving ...\n", .{func});

        self.synchronize();

        const t_start_us = c.ggml_time_us();

        const n_seqs = self.cparams.n_seq_max;
        const n_tokens = @min(self.cparams.n_ctx, self.cparams.n_ubatch);

        const max_nodes = self.graphMaxNodes(n_tokens);

        impl.logDebug("%s: max_nodes = %zu\n", .{ func, @as(usize, max_nodes) });

        for (&self.gf_res_prev) |*res| {
            if (res.*) |r| r.destroy();
            res.* = null;
        }
        if (self.gf_res_reserve) |r| r.destroy();
        self.gf_res_reserve = graph.Result.create(max_nodes);
        self.gf_res_prev_active = null;

        self.schedNew(max_nodes, self.cparams.pipeline_parallel);

        var mctx: ?memory.Context = null;
        defer if (mctx) |m| m.destroy();
        if (self.mem) |m| {
            impl.logDebug("%s: reserving full memory module\n", .{func});
            mctx = m.initFull();
        }

        // No graph with zero outputs: one output per sequence.
        const n_outputs = n_seqs;

        impl.logDebug("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", .{ func, n_tokens, n_seqs, n_outputs });

        try self.resolveFusedOps(mctx, n_seqs, exc);

        const no_alloc = self.model.noAlloc();
        const n_outputs_pp = @min(n_tokens, self.cparams.n_outputs_max);

        // The prompt-processing graph first, so buffers are allocated once.
        var n_splits_pp: c_int = -1;
        var n_nodes_pp: c_int = -1;
        var n_inputs_pp: c_int = -1;
        var n_input_tensors_pp: c_int = -1;
        {
            var gf = try self.graphReserve(n_tokens, n_seqs, n_outputs_pp, mctx, no_alloc, if (no_alloc) self.backend_buf_exp_size.items.ptr else null, exc);
            if (gf == null) {
                if (self.cparams.pipeline_parallel) {
                    impl.logWarn("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", .{func});
                    self.cparams.pipeline_parallel = false;
                    self.schedNew(max_nodes, false);
                    gf = try self.graphReserve(n_tokens, n_seqs, n_outputs_pp, mctx, false, null, exc);
                }
                if (gf == null) return exc.throw(.runtime_error, "failed to allocate compute pp buffers", .{});
            }
            n_splits_pp = c.ggml_backend_sched_get_n_splits(self.sched);
            n_nodes_pp = c.ggml_graph_n_nodes(gf);
            n_inputs_pp = @truncate(@as(i64, @intCast(self.getGfResReserve().inputs.items.len)));
            n_input_tensors_pp = @bitCast(self.n_input_tensors);

            self.graph_has_logits = self.getGfResReserve().getLogits() != null;
        }

        // The token-generation graph, for its splits and nodes; none for
        // training.
        var n_splits_tg: c_int = -1;
        var n_nodes_tg: c_int = -1;
        var n_inputs_tg: c_int = -1;
        var n_input_tensors_tg: c_int = -1;
        if (self.cparams.training) {
            n_splits_tg = n_splits_pp;
            n_nodes_tg = n_nodes_pp;
        } else {
            const gf = try self.graphReserve(n_seqs, n_seqs, n_seqs, mctx, no_alloc, null, exc) orelse
                return exc.throw(.runtime_error, "failed to allocate compute tg buffers", .{});
            n_splits_tg = c.ggml_backend_sched_get_n_splits(self.sched);
            n_nodes_tg = c.ggml_graph_n_nodes(gf);
            n_inputs_tg = @truncate(@as(i64, @intCast(self.getGfResReserve().inputs.items.len)));
            n_input_tensors_tg = @bitCast(self.n_input_tensors);
        }

        // The prompt-processing graph again, to avoid reallocations during
        // inference. The worst case is not always reached with n_seqs > 1:
        // Kimi Linear's and MiniMax-01's `inp_diag_decay` grows with
        // n_seq_tokens^2, so one sequence needs more memory than several.
        const n_seqs_pp: u32 = switch (self.model.arch) {
            .kimi_linear, .minimax_01 => 1,
            else => n_seqs,
        };
        _ = try self.graphReserve(n_tokens, n_seqs_pp, n_outputs_pp, mctx, no_alloc, null, exc) orelse
            return exc.throw(.runtime_error, "failed to allocate compute pp buffers", .{});

        for (self.backend_ptrs.items, self.backend_buft.items, 0..) |backend, buft, i| {
            if (!no_alloc) self.backend_buf_exp_size.items[i] = c.ggml_backend_sched_get_buffer_size(self.sched, backend);
            if (self.backend_buf_exp_size.items[i] > 1) {
                impl.logInfo("%s: %10s compute buffer size = %8.2f MiB\n", .{ func, c.ggml_backend_buft_name(buft), @as(f64, @floatFromInt(self.backend_buf_exp_size.items[i])) / 1024.0 / 1024.0 });
            }
        }

        {
            const diff = n_nodes_pp != n_nodes_tg or n_splits_pp != n_splits_tg or
                n_inputs_pp != n_inputs_tg or n_input_tensors_pp != n_input_tensors_tg;

            const Val = struct {
                buf: [32]u8 = undefined,
                fn of(v: *@This(), d: bool, v_pp: c_int, v_tg: c_int) [*:0]const u8 {
                    const r = if (d)
                        std.fmt.bufPrintSentinel(&v.buf, "{d} / {d}", .{ v_pp, v_tg }, 0)
                    else
                        std.fmt.bufPrintSentinel(&v.buf, "{d}", .{v_pp}, 0);
                    return (r catch unreachable).ptr;
                }
            };
            var hdr_buf: [64]u8 = undefined;
            const hdr: [*:0]const u8 = if (diff)
                (std.fmt.bufPrintSentinel(&hdr_buf, " (pp bs={d}, tg bs={d})", .{ @as(c_int, @bitCast(n_tokens)), @as(c_int, @bitCast(n_seqs)) }, 0) catch unreachable).ptr
            else
                "";
            var v_nodes: Val = .{};
            var v_splits: Val = .{};
            var v_inputs: Val = .{};
            var v_input_tensors: Val = .{};

            impl.logInfo("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n", .{
                func,
                hdr,
                v_nodes.of(diff, n_nodes_pp, n_nodes_tg),
                v_splits.of(diff, n_splits_pp, n_splits_tg),
                v_inputs.of(diff, n_inputs_pp, n_inputs_tg),
                v_input_tensors.of(diff, n_input_tensors_pp, n_input_tensors_tg),
            });
        }

        const t_end_us = c.ggml_time_us();
        impl.logInfo("%s: reserve took %.2f ms, sched copies = %d\n", .{ func, @as(f64, @floatFromInt(t_end_us - t_start_us)) / 1000.0, c.ggml_backend_sched_get_n_copies(self.sched) });
    }

    /// Ports `llama_context`'s `synchronize` (llama-context.cpp:947 @cd959bae9):
    /// wait for the backends, then account the time.
    pub fn synchronize(self: *Context) void {
        if (self.sched == null) return;
        c.ggml_backend_sched_synchronize(self.sched);

        // Single tokens evaluated without a synchronization between them
        // count as prompt evaluation.
        if (self.n_queued_tokens == 1) {
            if (!self.cparams.no_perf) self.t_eval_us += c.ggml_time_us() - self.t_compute_start_us;
            self.n_eval += 1;
        } else if (self.n_queued_tokens > 1) {
            if (!self.cparams.no_perf) self.t_p_eval_us += c.ggml_time_us() - self.t_compute_start_us;
            self.n_p_eval +%= @truncate(self.n_queued_tokens);
        }

        // A more accurate load time, on the first eval.
        if (self.n_queued_tokens > 0 and !self.has_evaluated_once) {
            self.t_load_us = c.ggml_time_us() - self.t_start_us;
            self.has_evaluated_once = true;
        }

        self.n_queued_tokens = 0;
        self.t_compute_start_us = 0;
    }

    /// Ports `llama_context`'s `get_model` (llama-context.cpp:981 @cd959bae9).
    pub fn getModel(self: *const Context) *const Model {
        return self.model;
    }

    /// Ports `llama_context`'s `get_cparams` (llama-context.cpp:985 @cd959bae9).
    pub fn getCparams(self: *const Context) *const Cparams {
        return &self.cparams;
    }

    /// Ports `llama_context`'s `get_sched` (llama-context.cpp:989 @cd959bae9).
    pub fn getSched(self: *const Context) c.ggml_backend_sched_t {
        return self.sched;
    }

    pub fn nCtx(self: *const Context) u32 {
        return self.cparams.n_ctx;
    }
    pub fn nCtxSeq(self: *const Context) u32 {
        return self.cparams.n_ctx_seq;
    }
    pub fn nBatch(self: *const Context) u32 {
        return self.cparams.n_batch;
    }
    pub fn nUbatch(self: *const Context) u32 {
        return self.cparams.n_ubatch;
    }
    pub fn nSeqMax(self: *const Context) u32 {
        return self.cparams.n_seq_max;
    }
    pub fn nThreads(self: *const Context) u32 {
        return @bitCast(self.cparams.n_threads);
    }
    pub fn nThreadsBatch(self: *const Context) u32 {
        return @bitCast(self.cparams.n_threads_batch);
    }

    /// Ports `llama_context`'s `get_memory` (llama-context.cpp:1021 @cd959bae9).
    pub fn getMemory(self: *const Context) ?memory.Memory {
        return self.mem;
    }

    /// Ports `llama_context`'s `memory_update` (llama-context.cpp:1025 @cd959bae9):
    /// apply pending shifts and copies, then reserve a worst-case graph.
    ///
    /// Return: whether the memory updated.
    pub fn memoryUpdate(self: *Context, optimize: bool) bool {
        const func = "memory_update";
        const m = self.mem orelse return false;
        {
            const mctx = m.initUpdate(self, optimize);
            defer mctx.destroy();
            switch (mctx.getStatus()) {
                .success => {},
                .no_update => return false,
                .failed_prepare, .failed_compute => {
                    impl.logError("%s: failed to prepare memory update\n", .{func});
                    return false;
                },
            }
            // The previous graphs cannot be reused.
            for (self.gf_res_prev) |res| if (res) |r| r.reset();
            self.gf_res_prev_active = null;
            if (!mctx.apply()) impl.logError("%s: failed to apply memory update\n", .{func});
        }

        // Any computation the memory did needs a new worst-case graph.
        {
            const mctx = m.initFull();
            defer mctx.destroy();
            const n_seqs = self.cparams.n_seq_max;
            const n_tokens = @min(self.cparams.n_ctx, self.cparams.n_ubatch);
            const n_outputs_max = @min(n_tokens, self.cparams.n_outputs_max);
            // the Hadamard check only throws on the first reserve, in the constructor
            var exc: Exception = .{};
            const gf = self.graphReserve(n_tokens, n_seqs, n_outputs_max, mctx, false, null, &exc) catch exc.terminate("llama-context.cpp", 0);
            if (gf == null) {
                impl.logError("%s: failed to reserve graph after the memory update\n", .{func});
            }
        }
        return true;
    }

    /// Ports `llama_context`'s `pooling_type` (llama-context.cpp:1085 @cd959bae9).
    pub fn poolingType(self: *const Context) c.enum_llama_pooling_type {
        return self.cparams.pooling_type;
    }

    /// Ports `llama_context`'s `get_logits` (llama-context.cpp:1089 @cd959bae9).
    pub fn getLogits(self: *Context) ?[*]f32 {
        self.outputReorder();
        return self.logits.data;
    }

    /// Ports `llama_context`'s `output_resolve_row` (llama-context.cpp:1095
    /// @cd959bae9): batch token `i` to its output row; a negative `i`
    /// counts from the last output.
    ///
    /// Return: the row, or `error.Thrown` with the C++'s message.
    fn outputResolveRow(self: *const Context, i: i32, exc: *Exception) Thrown!i64 {
        var j: i64 = -1;
        if (i < 0) {
            // `n_outputs + i` is `uint32_t` arithmetic, so it wraps rather
            // than going negative: this check never fires, and an index
            // before the first output reports a corrupt buffer below.
            j = self.n_outputs +% @as(u32, @bitCast(i));
            if (j < 0) return exc.throw(.runtime_error, "negative index out of range [0, {d})", .{self.n_outputs});
        } else if (@as(usize, @intCast(i)) >= self.output_ids.items.len) {
            return exc.throw(.runtime_error, "out of range [0, {d})", .{self.output_ids.items.len});
        } else {
            j = self.output_ids.items[@intCast(i)];
        }
        if (j < 0) return exc.throw(.runtime_error, "batch.logits[{d}] != true", .{i});
        if (j >= self.n_outputs) return exc.throw(.runtime_error, "corrupt output buffer (j={d}, n_outputs={d})", .{ j, self.n_outputs });
        return j;
    }

    fn logBadIndex(comptime func: [:0]const u8, comptime what: [:0]const u8, i: i32, exc: *Exception) void {
        exc.buf[exc.len] = 0;
        impl.logError("%s: invalid " ++ what ++ " id %d, reason: %s\n", .{ func.ptr, i, @as([*:0]const u8, @ptrCast(&exc.buf)) });
    }

    /// Ports `llama_context`'s `get_logits_ith` (llama-context.cpp:1124 @cd959bae9).
    pub fn getLogitsIth(self: *Context, i: i32) ?[*]f32 {
        self.outputReorder();
        var exc: Exception = .{};
        const data = self.logits.data orelse {
            exc.set(.runtime_error, "no logits", .{});
            logBadIndex("get_logits_ith", "logits", i, &exc);
            return null;
        };
        const j = self.outputResolveRow(i, &exc) catch {
            logBadIndex("get_logits_ith", "logits", i, &exc);
            return null;
        };
        return data + @as(usize, @intCast(j)) * self.model.vocab.nTokens();
    }

    /// Ports `llama_context`'s `get_embeddings` (llama-context.cpp:1144 @cd959bae9).
    pub fn getEmbeddings(self: *Context) ?[*]f32 {
        self.outputReorder();
        return self.embd.data;
    }

    /// Ports `llama_context`'s `get_sampled_tokens` (llama-context.cpp:1150 @cd959bae9).
    pub fn getSampledTokens(self: *const Context) ?[*]c.llama_token {
        return self.sampling.sampled.data;
    }

    /// Ports `llama_context`'s `get_embeddings_ith` (llama-context.cpp:1154 @cd959bae9).
    pub fn getEmbeddingsIth(self: *Context, i: i32) ?[*]f32 {
        self.outputReorder();
        var exc: Exception = .{};
        const data = self.embd.data orelse {
            exc.set(.runtime_error, "no embeddings", .{});
            logBadIndex("get_embeddings_ith", "embeddings", i, &exc);
            return null;
        };
        const j = self.outputResolveRow(i, &exc) catch {
            logBadIndex("get_embeddings_ith", "embeddings", i, &exc);
            return null;
        };
        return data + @as(usize, @intCast(j)) * self.model.hparams.nEmbdOut();
    }

    /// Ports `llama_context`'s `get_embeddings_seq` (llama-context.cpp:1175 @cd959bae9).
    pub fn getEmbeddingsSeq(self: *Context, seq_id: i32) ?[*]f32 {
        for (self.embd_seq.items) |*e| if (e.seq_id == seq_id) return e.v.items.ptr;
        return null;
    }

    /// Ports `llama_context`'s `get_embeddings_nextn` (llama-context.cpp:1184 @cd959bae9).
    pub fn getEmbeddingsNextn(self: *Context) ?[*]f32 {
        self.outputReorder();
        return self.embd_nextn.data;
    }

    /// Ports `llama_context`'s `get_embeddings_nextn_ith` (llama-context.cpp:1190 @cd959bae9).
    pub fn getEmbeddingsNextnIth(self: *Context, i: i32) ?[*]f32 {
        self.outputReorder();
        var exc: Exception = .{};
        const data = self.embd_nextn.data orelse {
            exc.set(.runtime_error, "no nextn embeddings", .{});
            logBadIndex("get_embeddings_nextn_ith", "nextn embeddings", i, &exc);
            return null;
        };
        const n_embd: usize = self.model.hparams.nEmbdOut();
        if (!self.cparams.embeddings_nextn_masked) {
            // Unmasked rows are dense, by raw token position.
            if (i < 0 or (@as(usize, @intCast(i)) + 1) * n_embd > self.embd_nextn.size) {
                exc.set(.runtime_error, "out of range [0, {d})", .{self.embd_nextn.size / n_embd});
                logBadIndex("get_embeddings_nextn_ith", "nextn embeddings", i, &exc);
                return null;
            }
            return data + @as(usize, @intCast(i)) * n_embd;
        }
        const j = self.outputResolveRow(i, &exc) catch {
            logBadIndex("get_embeddings_nextn_ith", "nextn embeddings", i, &exc);
            return null;
        };
        return data + @as(usize, @intCast(j)) * n_embd;
    }

    /// Ports `llama_context`'s `get_embeddings_layer_inp` (llama-context.cpp:1220 @cd959bae9).
    pub fn getEmbeddingsLayerInp(self: *Context, lid: u32) ?[*]f32 {
        self.outputReorder();
        impl.assert(lid < self.embd_layer_inp.len and self.embd_layer_inp[lid].hasData(), "llama-context.cpp", 1223, "lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data()");
        return self.embd_layer_inp[lid].data;
    }

    /// Ports `llama_context`'s `get_sampled_token_ith` (llama-context.cpp:1228 @cd959bae9).
    pub fn getSampledTokenIth(self: *Context, idx: i32) c.llama_token {
        self.outputReorder();
        if (!self.sampling.sampled.hasData()) return c.LLAMA_TOKEN_NULL;
        var exc: Exception = .{};
        const row = self.outputResolveRow(idx, &exc) catch {
            logBadIndex("get_sampled_token_ith", "backend sampled token", idx, &exc);
            return c.LLAMA_TOKEN_NULL;
        };
        impl.assert(row < @as(i64, @intCast(self.sampling.sampled.size)), "llama-context.cpp", 1237, "row < (int64_t) sampling.sampled.size");
        return self.sampling.sampled.data.?[@intCast(row)];
    }

    fn sampledRows(self: *Context, comptime what: [:0]const u8, comptime func: [:0]const u8, view: BufferView(f32), counts: []const u32, idx: i32) ?[*]f32 {
        self.outputReorder();
        if (!view.hasData()) return null;
        var exc: Exception = .{};
        const row = self.outputResolveRow(idx, &exc) catch {
            logBadIndex(func, what, idx, &exc);
            return null;
        };
        if (@as(usize, @intCast(row)) >= counts.len or counts[@intCast(row)] == 0) return null;
        return view.data.? + @as(usize, @intCast(row)) * self.model.vocab.nTokens();
    }

    /// Ports `llama_context`'s `get_sampled_probs_ith` (llama-context.cpp:1245 @cd959bae9).
    pub fn getSampledProbsIth(self: *Context, idx: i32) ?[*]f32 {
        return self.sampledRows("backend sampled probs", "get_sampled_probs_ith", self.sampling.probs, self.sampling.probs_count.items, idx);
    }

    /// Ports `llama_context`'s `get_sampled_logits_ith` (llama-context.cpp:1264 @cd959bae9).
    pub fn getSampledLogitsIth(self: *Context, idx: i32) ?[*]f32 {
        return self.sampledRows("backend sampled logits", "get_sampled_logits_ith", self.sampling.logits, self.sampling.logits_count.items, idx);
    }

    /// Ports `llama_context`'s `get_sampled_candidates_ith` (llama-context.cpp:1283
    /// @cd959bae9): the full vocabulary when the row has no candidates.
    pub fn getSampledCandidatesIth(self: *Context, idx: i32) [*]const c.llama_token {
        self.outputReorder();
        var exc: Exception = .{};
        if (self.outputResolveRow(idx, &exc)) |row| {
            const r: usize = @intCast(row);
            if (self.sampling.candidates.hasData() and r < self.sampling.candidates_count.items.len and self.sampling.candidates_count.items[r] > 0) {
                return self.sampling.candidates.data.? + r * self.model.vocab.nTokens();
            }
        } else |_| {}
        return self.sampling.token_ids_full_vocab.items.ptr;
    }

    fn sampledCount(self: *Context, comptime what: [:0]const u8, comptime func: [:0]const u8, has_data: bool, counts: []const u32, idx: i32, empty: usize) usize {
        self.outputReorder();
        if (!has_data) return empty;
        var exc: Exception = .{};
        const row = self.outputResolveRow(idx, &exc) catch {
            logBadIndex(func, what, idx, &exc);
            return 0;
        };
        if (@as(usize, @intCast(row)) >= counts.len) return 0;
        return counts[@intCast(row)];
    }

    /// Ports `llama_context`'s `get_sampled_candidates_count` (llama-context.cpp:1301 @cd959bae9).
    pub fn getSampledCandidatesCount(self: *Context, idx: i32) usize {
        return self.sampledCount("backend sampled candidates count", "get_sampled_candidates_count", self.sampling.candidates.hasData(), self.sampling.candidates_count.items, idx, 0);
    }

    /// Ports `llama_context`'s `get_sampled_logits_count` (llama-context.cpp:1320 @cd959bae9).
    pub fn getSampledLogitsCount(self: *Context, idx: i32) usize {
        return self.sampledCount("backend sampled logits count", "get_sampled_logits_count", self.sampling.logits.hasData(), self.sampling.logits_count.items, idx, self.model.vocab.nTokens());
    }

    /// Ports `llama_context`'s `get_sampled_probs_count` (llama-context.cpp:1339 @cd959bae9).
    pub fn getSampledProbsCount(self: *Context, idx: i32) usize {
        return self.sampledCount("backend sampled probs count", "get_sampled_probs_count", self.sampling.probs.hasData(), self.sampling.probs_count.items, idx, 0);
    }

    /// Ports `llama_context`'s `attach_threadpool` (llama-context.cpp:1359 @cd959bae9).
    pub fn attachThreadpool(self: *Context, threadpool: c.ggml_threadpool_t, threadpool_batch: c.ggml_threadpool_t) void {
        impl.logDebug("%s: call\n", .{"attach_threadpool"});
        self.threadpool = threadpool;
        self.threadpool_batch = if (threadpool_batch != null) threadpool_batch else threadpool;
    }

    /// Ports `llama_context`'s `detach_threadpool` (llama-context.cpp:1368 @cd959bae9).
    pub fn detachThreadpool(self: *Context) void {
        impl.logDebug("%s: call\n", .{"detach_threadpool"});
        self.threadpool = null;
        self.threadpool_batch = null;
    }

    /// Ports `llama_context`'s `set_n_threads` (llama-context.cpp:1375 @cd959bae9).
    pub fn setNThreads(self: *Context, n_threads: i32, n_threads_batch: i32) void {
        impl.logDebug("%s: n_threads = %d, n_threads_batch = %d\n", .{ "set_n_threads", n_threads, n_threads_batch });
        self.cparams.n_threads = n_threads;
        self.cparams.n_threads_batch = n_threads_batch;
    }

    /// Ports `llama_context`'s `set_abort_callback` (llama-context.cpp:1382 @cd959bae9).
    pub fn setAbortCallback(self: *Context, abort_callback: c.ggml_abort_callback, abort_callback_data: ?*anyopaque) void {
        impl.logDebug("%s: call\n", .{"set_abort_callback"});
        self.abort_callback = abort_callback;
        self.abort_callback_data = abort_callback_data;
        for (self.backends.items) |backend| {
            const reg = c.ggml_backend_dev_backend_reg(c.ggml_backend_get_device(backend));
            if (reg != null) {
                const f: ?*const fn (c.ggml_backend_t, c.ggml_abort_callback, ?*anyopaque) callconv(.c) void = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback")));
                if (f) |set| set(backend, self.abort_callback, self.abort_callback_data);
            }
        }
    }

    /// Ports `llama_context`'s `set_embeddings` (llama-context.cpp:1399 @cd959bae9).
    pub fn setEmbeddings(self: *Context, value: bool) void {
        impl.logDebug("%s: value = %d\n", .{ "set_embeddings", @as(c_int, @intFromBool(value)) });

        // the toggle changes the graph shape (pooled: logits only; pooling
        // NONE: the reserve graph outputs all tokens, which crashed
        // ggml_gallocr on the first /completion), so re-reserve on every toggle
        if (self.cparams.embeddings != value) {
            self.sched_need_reserve = true;
        }

        self.cparams.embeddings = value;
    }

    /// Ports `llama_context`'s `set_embeddings_nextn` (llama-context.cpp:1411 @cd959bae9).
    pub fn setEmbeddingsNextn(self: *Context, value: bool, masked: bool) void {
        impl.logDebug("%s: value = %d, masked = %d\n", .{ "set_embeddings_nextn", @as(c_int, @intFromBool(value)), @as(c_int, @intFromBool(masked)) });

        if (self.cparams.embeddings_nextn != value or self.cparams.embeddings_nextn_masked != masked) {
            // these flags change the graph shape
            self.sched_need_reserve = true;
        }

        self.cparams.embeddings_nextn = value;
        self.cparams.embeddings_nextn_masked = masked;
    }

    /// Ports `llama_context`'s `set_embeddings_layer_inp` (llama-context.cpp:1423 @cd959bae9).
    pub fn setEmbeddingsLayerInp(self: *Context, lid: u32, enable: bool) void {
        impl.logDebug("%s: lid = %d, enable = %d\n", .{ "set_embeddings_layer_inp", lid, @as(c_int, @intFromBool(enable)) });
        impl.assert(lid <= self.model.hparams.nLayer(), "llama-context.cpp", 1426, "lid <= model.hparams.n_layer()");
        self.embeddings_layer_inp[lid] = enable;
        // Without this reserve the draft acceptance drops to zero.
        self.sched_need_reserve = true;

        if (self.cparams.embeddings_layer_inp_dev) {
            self.layerInpDevAlloc();
        }
    }

    /// Ports `llama_context`'s `set_embeddings_layer_inp_dev`
    /// (llama-context.cpp:1438 @cd959bae9): write the enabled layer inputs
    /// to a device buffer instead of host memory.
    pub fn setEmbeddingsLayerInpDev(self: *Context, value: bool) void {
        impl.logDebug("%s: value = %d\n", .{ "set_embeddings_layer_inp_dev", @as(c_int, @intFromBool(value)) });

        if (self.cparams.embeddings_layer_inp_dev != value) {
            self.sched_need_reserve = true;
        }

        self.cparams.embeddings_layer_inp_dev = value;

        self.layerInpDevAlloc();
    }

    /// Ports `llama_context`'s `get_layer_inp_dev` (llama-context.h:96 @cd959bae9).
    pub fn getLayerInpDev(self: *const Context) ?*Tensor {
        return self.layer_inp_dev;
    }

    /// Ports `llama_context`'s `supports_embd_row` (llama-context.cpp:1450
    /// @cd959bae9): whether decode accepts entries whose embedding is a
    /// row of `ctx_other`'s layer input device buffer -- a DFlash/DSpark
    /// decoder whose backends can read that buffer.
    pub fn supportsEmbdRow(self: *const Context) bool {
        // the DFlash/DSpark decoder graphs read the device buffer
        if (self.model.arch != .dflash) {
            return false;
        }

        const other: ?*const Context = @ptrCast(@alignCast(self.cparams.ctx_other));
        const src = if (other) |o| o.getLayerInpDev() else null;
        if (src == null) {
            return false;
        }

        // One of our backends must read the buffer: not a draft on another
        // GPU than the target's output, say.
        const buft = c.ggml_backend_buffer_get_type(src.?.buffer);
        for (self.backends.items) |backend| {
            if (c.ggml_backend_supports_buft(backend, buft)) {
                return true;
            }
        }

        return false;
    }

    /// Ports `llama_context`'s `backend_from_dev` (llama-context.cpp:1472 @cd959bae9).
    fn backendFromDev(self: *const Context, dev: c.ggml_backend_dev_t) c.ggml_backend_t {
        for (self.backends.items) |backend| {
            if (c.ggml_backend_get_device(backend) == dev) {
                return backend;
            }
        }
        return null;
    }

    /// Ports `llama_context`'s `layer_inp_dev_alloc` (llama-context.cpp:1481
    /// @cd959bae9): (re)allocate the device buffer for
    /// `cparams.embeddings_layer_inp_dev`, on the model's output device.
    fn layerInpDevAlloc(self: *Context) void {
        const func = "layer_inp_dev_alloc";
        self.synchronize();

        self.layer_inp_dev = null;
        self.layer_inp_dev_dev = null;
        self.layer_inp_backend = null;
        if (self.buf_layer_inp_dev != null) c.ggml_backend_buffer_free(self.buf_layer_inp_dev);
        self.buf_layer_inp_dev = null;
        if (self.ctx_layer_inp_dev) |ctx| c.ggml_free(ctx);
        self.ctx_layer_inp_dev = null;

        if (!self.cparams.embeddings_layer_inp_dev) {
            return;
        }

        var n_layer_inp: i64 = 0;
        for (self.cparams.embeddings_layer_inp) |enabled| {
            n_layer_inp += @intFromBool(enabled);
        }
        if (n_layer_inp == 0) {
            return;
        }

        self.ctx_layer_inp_dev = c.ggml_init(.{
            .mem_size = c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        });

        const t = c.ggml_new_tensor_2d(self.ctx_layer_inp_dev, c.GGML_TYPE_F32, n_layer_inp * self.model.hparams.n_embd, self.cparams.n_batch);
        _ = c.ggml_set_name(t, "layer_inp_dev");
        self.layer_inp_dev = t;

        self.layer_inp_dev_dev = self.model.devOutput();
        const buft = if (self.layer_inp_dev_dev != null) c.ggml_backend_dev_buffer_type(self.layer_inp_dev_dev) else c.ggml_backend_cpu_buffer_type();

        self.buf_layer_inp_dev = c.ggml_backend_alloc_ctx_tensors_from_buft(self.ctx_layer_inp_dev, buft);
        if (self.buf_layer_inp_dev == null) {
            // a large n_batch that does not fit next to the model, say: keep the host copies
            impl.logWarn("%s: failed to allocate the layer input device buffer, using host copies\n", .{func});
            self.cparams.embeddings_layer_inp_dev = false;
            self.layer_inp_dev = null;
            self.layer_inp_dev_dev = null;
            if (self.ctx_layer_inp_dev) |ctx| c.ggml_free(ctx);
            self.ctx_layer_inp_dev = null;
            return;
        }

        impl.logInfo("%s: %10s layer input buffer size = %8.2f MiB (%d layers)\n", .{ func, c.ggml_backend_buffer_name(self.buf_layer_inp_dev), @as(f64, @floatFromInt(c.ggml_backend_buffer_get_size(self.buf_layer_inp_dev))) / 1024.0 / 1024.0, @as(c_int, @intCast(n_layer_inp)) });

        self.layer_inp_backend = if (self.layer_inp_dev_dev != null) self.backendFromDev(self.layer_inp_dev_dev) else null;
        if (self.layer_inp_backend != null and self.ev_layer_inp_written == null) {
            self.ev_layer_inp_written = c.ggml_backend_event_new(self.layer_inp_dev_dev);
            self.ev_layer_inp_read = c.ggml_backend_event_new(self.layer_inp_dev_dev);
            self.ev_layer_inp_read_recorded = false;
        }
    }

    /// Ports `llama_context`'s `set_nextn_layer_offset` (llama-context.cpp:1537 @cd959bae9).
    pub fn setNextnLayerOffset(self: *Context, offset: i32) void {
        self.cparams.nextn_layer_offset = offset;
    }

    /// Ports `llama_context`'s `set_causal_attn` (llama-context.cpp:1541 @cd959bae9).
    /// No scheduler reserve: graph shapes must not depend on `causal_attn`,
    /// so a flip only rebuilds the graph.
    pub fn setCausalAttn(self: *Context, value: bool) void {
        impl.logDebug("%s: value = %d\n", .{ "set_causal_attn", @as(c_int, @intFromBool(value)) });
        self.cparams.causal_attn = value;
    }

    /// Ports `llama_context`'s `get_causal_attn` (llama-context.cpp:1549 @cd959bae9).
    ///
    /// Return: whether the context uses causal attention now.
    pub fn getCausalAttn(self: *const Context) bool {
        return self.cparams.causal_attn;
    }

    /// Ports `llama_context`'s `set_warmup` (llama-context.cpp:1553 @cd959bae9):
    /// no scheduler reserve, warmups being usually with small batches.
    pub fn setWarmup(self: *Context, value: bool) void {
        impl.logDebug("%s: value = %d\n", .{ "set_warmup", @as(c_int, @intFromBool(value)) });
        self.cparams.warmup = value;
    }

    var warned_sm_tensor = false;

    /// Ports `llama_context`'s `set_sampler` (llama-context.cpp:1561 @cd959bae9):
    /// offload a sampler chain to the backend, or remove one.
    ///
    /// Return: whether the sampler is now offloaded (or removed).
    pub fn setSampler(self: *Context, seq_id: i32, sampler: ?*c.llama_sampler) bool {
        const func = "set_sampler";
        const present = self.sampling.samplers.get(seq_id) != null;
        if (sampler == null and !present) return true;

        impl.logDebug("%s: seq_id = %d, sampler = %p\n", .{ func, @as(c_int, seq_id), @as(?*anyopaque, @ptrCast(sampler)) });

        if (sampler != null and self.model.params.split_mode == c.LLAMA_SPLIT_MODE_TENSOR) {
            if (!warned_sm_tensor) {
                impl.logWarn("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", .{func});
                warned_sm_tensor = true;
            }
            if (present) self.sched_need_reserve = true;
            _ = self.sampling.samplers.remove(seq_id);
            return false;
        }

        const can_offload = if (sampler) |s| s.*.iface.*.backend_init != null and s.*.iface.*.backend_apply != null and c.llama_sampler_chain_n(s) > 0 else false;

        if (sampler != null and can_offload) {
            const s = sampler.?;
            const buft = c.ggml_backend_dev_buffer_type(self.model.devOutput());
            _ = s.*.iface.*.backend_init.?(s, buft, @intCast(self.cparams.n_outputs_max_per_seq));
            self.sampling.samplers.put(seq_id, s);
            self.sched_need_reserve = true;
            return true;
        }

        if (sampler != null and !can_offload) {
            impl.logWarn("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", .{ func, c.llama_sampler_name(sampler), seq_id });
            if (present) self.sched_need_reserve = true;
            _ = self.sampling.samplers.remove(seq_id);
            return false;
        }

        _ = self.sampling.samplers.remove(seq_id);
        self.sched_need_reserve = true;
        return true;
    }

    /// Ports `llama_context`'s `set_adapters_lora` (llama-context.cpp:1618
    /// @cd959bae9): replace the active LoRA adapters, those with a zero
    /// scale left out, unless they are already the active ones.
    ///
    /// Parameters:
    /// - `adapters`, `scales`: `n_adapters` of each; borrowed for the call.
    ///
    /// Return: nothing.
    pub fn setAdaptersLora(self: *Context, adapters: ?[*]const ?*adapter.Lora, n_adapters: usize, scales: ?[*]const f32) void {
        impl.logDebug("%s: adapters = %p\n", .{ "set_adapters_lora", @as(?*const anyopaque, @ptrCast(adapters)) });

        if (self.adaptersLoraAreSame(adapters, n_adapters, scales)) return;

        // `loras.reset(new llama_adapter_loras())`: a new map, at a new
        // address, which the previous graph's parameters no longer match.
        const fresh = allocator.create(adapter.Loras) catch oom();
        fresh.* = .{};
        self.loras.deinit();
        allocator.destroy(self.loras);
        self.loras = fresh;

        for (0..n_adapters) |i| {
            if (scales.?[i] != 0.0) self.loras.insert(adapters.?[i].?, scales.?[i]);
        }

        self.sched_need_reserve = true;
    }

    /// Ports `llama_context`'s `adapters_lora_are_same` (llama-context.cpp:1636
    /// @cd959bae9).
    ///
    /// Return: whether the adapters with a non-zero scale are exactly the
    /// active ones, at the same scales.
    pub fn adaptersLoraAreSame(self: *const Context, adapters: ?[*]const ?*adapter.Lora, n_adapters: usize, scales: ?[*]const f32) bool {
        impl.logDebug("%s: adapters = %p\n", .{ "adapters_lora_are_same", @as(?*const anyopaque, @ptrCast(adapters)) });

        // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
        var n_non_zero: usize = 0;
        for (0..n_adapters) |i| {
            if (scales.?[i] == 0.0) continue;
            n_non_zero += 1;
            const scale = self.loras.find(adapters.?[i]) orelse return false;
            if (scale != scales.?[i]) return false;
        }
        return n_non_zero == self.loras.count();
    }

    /// Ports `llama_context`'s `set_adapter_cvec` (llama-context.cpp:1662
    /// @cd959bae9).
    ///
    /// Parameters: as `adapter.Cvec.apply`.
    ///
    /// Return: whether the control vector was applied.
    pub fn setAdapterCvec(self: *Context, data: ?[*]const f32, len: usize, n_embd: i32, il_start: i32, il_end: i32) bool {
        impl.logDebug("%s: il_start = %d, il_end = %d\n", .{ "set_adapter_cvec", il_start, il_end });
        const res = self.cvec.apply(self.model, data, len, n_embd, il_start, il_end);
        self.sched_need_reserve = true;
        return res;
    }

    /// Ports `llama_context`'s `process_ubatch` (llama-context.cpp:1677
    /// @cd959bae9): build or reuse the graph, feed it, compute it.
    ///
    /// Return: the graph's result, or null with `ret` saying why.
    pub fn processUbatch(self: *Context, ubatch: *const Ubatch, gtype: graph.GraphType, mctx: ?memory.Context, ret: *c.enum_ggml_status) ?*graph.Result {
        const func = "process_ubatch";
        if (mctx) |m| {
            if (!m.apply()) {
                impl.logError("%s: failed to apply memory context\n", .{func});
                ret.* = c.GGML_STATUS_FAILED;
                return null;
            }
        }

        const res = self.getGfResPrev();
        var gf = res.getGf();

        // A graph is reused only if these parameters determine its whole
        // topology.
        var gparams = self.graphParams(res, ubatch, mctx, gtype);
        defer gparams.samplers.deinit();

        if (!self.graph_reuse_disable and self.gf_res_prev_active == res and res.canReuse(&gparams)) {
            // With pipeline parallelism the previous compute may still be
            // reading the inputs.
            if (self.cparams.pipeline_parallel) c.ggml_backend_sched_synchronize(self.sched);
            self.n_reused += 1;
        } else {
            self.gf_res_prev_active = null;
            res.reset();
            c.ggml_backend_sched_reset(self.sched);
            c.ggml_backend_sched_set_eval_callback(self.sched, self.cparams.cb_eval, self.cparams.cb_eval_user_data);

            gf = self.model.buildGraph(&gparams);

            if (!c.ggml_backend_sched_alloc_graph(self.sched, gf)) {
                impl.logError("%s: failed to allocate graph\n", .{func});
                ret.* = c.GGML_STATUS_ALLOC_FAILED;
                return null;
            }

            self.gf_res_prev_active = res;
        }

        // The input data for the input tensors.
        res.setInputs(ubatch);

        const status = self.graphCompute(res.getGf(), ubatch.n_tokens > 1);
        if (status != c.GGML_STATUS_SUCCESS) {
            impl.logError("%s: failed to compute graph, compute status: %d\n", .{ func, status });
            ret.* = status;
            return null;
        }
        ret.* = c.GGML_STATUS_SUCCESS;
        return res;
    }

    fn embdSeqSlot(self: *Context, seq_id: i32) *std.ArrayList(f32) {
        var pos: usize = 0;
        while (pos < self.embd_seq.items.len) : (pos += 1) {
            if (self.embd_seq.items[pos].seq_id == seq_id) return &self.embd_seq.items[pos].v;
            if (self.embd_seq.items[pos].seq_id > seq_id) break;
        }
        self.embd_seq.insert(allocator, pos, .{ .seq_id = seq_id, .v = .empty }) catch oom();
        return &self.embd_seq.items[pos].v;
    }

    fn embdSeqClear(self: *Context) void {
        for (self.embd_seq.items) |*e| e.v.deinit(allocator);
        self.embd_seq.clearRetainingCapacity();
    }

    /// The pooled embeddings of `ubatch`'s sequences, `n` floats each,
    /// copied out of `t_embd` -- shared by encode and decode.
    fn extractSeqEmbeddings(self: *Context, ubatch: *const Ubatch, backend_embd: c.ggml_backend_t, t_embd: *Tensor, n: usize) void {
        for (0..ubatch.n_seqs_unq) |s| {
            const seq_id = ubatch.seq_id_unq[s];
            const seq_idx: usize = @intCast(ubatch.seq_idx[@intCast(seq_id)]);
            const out = self.embdSeqSlot(seq_id);
            out.resize(allocator, n) catch oom();
            c.ggml_backend_tensor_get_async(backend_embd, t_embd, out.items.ptr, n * seq_idx * @sizeOf(f32), n * @sizeOf(f32));
        }
    }

    /// Ports the constructor of `llama_batch_ext` from a context
    /// (llama-batch.cpp:1188 @cd959bae9): an empty extended batch sized by
    /// the context and its model.
    ///
    /// Return: the batch; free it with `destroy`.
    pub fn batchExtCreate(self: *const Context) *batch.BatchExt {
        const model = self.model;
        const b = batch.BatchExt.create(
            self.nBatch(),
            batch.selectNEmbdInp(self.cparams.ctx_type, model.arch, &model.hparams),
            model.hparams.nEmbdInpEnc(),
            @bitCast(self.nSeqMax()),
            if (self.mem) |*m| m.posView() else null,
            @intCast(model.vocab.nTokens()),
            model.hparams.nPosPerEmbd(),
        );
        b.n_embd_state = batch.selectNEmbdState(self.cparams.ctx_type, &model.hparams);
        return b;
    }

    /// Ports `llama_context`'s compat `encode` (llama-context.cpp:4877
    /// @cd959bae9): a `llama_batch` through `llama_batch_compat`, its
    /// embedding rows at the encoder's input width.
    ///
    /// Return: as `encodeExt`.
    pub fn encode(self: *Context, batch_inp: c.llama_batch) c_int {
        const b = self.batchExtCreate();
        defer b.destroy();
        batch.compatInit(b, batch_inp, self.model.hparams.nEmbdInpEnc());
        return self.encodeExt(b);
    }

    /// Ports `llama_context`'s compat `decode` (llama-context.cpp:4882
    /// @cd959bae9).
    ///
    /// Return: as `decodeExt`.
    pub fn decode(self: *Context, batch_inp: c.llama_batch) c_int {
        const b = self.batchExtCreate();
        defer b.destroy();
        batch.compatInit(b, batch_inp, 0);
        return self.decodeExt(b);
    }

    /// Ports `llama_context`'s `encode` (llama-context.cpp:1752 @cd959bae9).
    ///
    /// Return: 0 on success, as `llama_encode` returns.
    pub fn encodeExt(self: *Context, batch_inp: *const batch.BatchExt) c_int {
        const func = "encode";

        if (batch_inp.tokens.items.len == 0) {
            impl.logError("%s: n_tokens == 0\n", .{func});
            return -1;
        }

        const hparams = &self.model.hparams;

        if (batch_inp.n_embd > 0 and batch_inp.n_embd != hparams.nEmbdInpEnc()) {
            impl.logError("%s: embd row width %zu does not match the encoder input %u\n", .{ func, batch_inp.n_embd, hparams.nEmbdInpEnc() });
            return -1;
        }

        if (batch_inp.tokens.items.len != 0 and batch_inp.tokens.items[0].embd_row >= 0) {
            impl.logError("%s: embd row batches are not supported by encode\n", .{func});
            return -1;
        }

        // eagle3/DFlash: features as the encoder's input.
        const n_vocab: usize = self.model.vocab.nTokens();

        // An encode skips position continuity checks and outputs all tokens;
        // a decoder without memory (see decode()) outputs the marked tokens,
        // unless it extracts embeddings.
        if (!self.balloc.initBatch(batch_inp, &self.model.vocab, self.mem != null or self.cparams.embeddings)) {
            impl.logError("%s: failed to initialize batch\n", .{func});
            return -1;
        }

        const n_tokens = self.balloc.getNTokens();

        // [TAG_NO_CACHE_PAD]
        var ubatch = self.balloc.splitSimple(n_tokens);
        defer ubatch.release();

        // Non-causal encoding cannot be micro-batched.
        impl.assert(self.cparams.n_ubatch >= n_tokens, "llama-context.cpp", 1788, "cparams.n_ubatch >= n_tokens && \"encoder requires n_ubatch >= n_tokens\"");

        // Wait for in-flight copies into embd_seq before freeing it.
        if (self.embd_seq.items.len != 0) self.synchronize();
        self.embdSeqClear();

        if (self.t_compute_start_us == 0) self.t_compute_start_us = c.ggml_time_us();

        var exc: Exception = .{};
        self.schedReserve(&exc) catch exc.terminate("llama-context.cpp", 1801);

        self.n_queued_tokens += n_tokens;

        const n_outputs_all = self.balloc.getNOutputs();

        if (self.outputReserve(@intCast(n_outputs_all)) < n_outputs_all) {
            impl.logError("%s: could not reserve space for batch with %u outputs\n", .{ func, n_outputs_all });
            return -2;
        }
        const out_ids = self.balloc.getOutIds().items;
        for (0..n_outputs_all) |i| self.output_ids.items[@intCast(out_ids[i])] = @intCast(i);
        self.n_outputs = n_outputs_all;

        // Encoder graphs are always non-causal, for now.
        const causal_attn_org = self.cparams.causal_attn;
        self.cparams.causal_attn = false;

        var status: c.enum_ggml_status = c.GGML_STATUS_SUCCESS;
        const res_opt = self.processUbatch(&ubatch, .encoder, null, &status);

        self.cparams.causal_attn = causal_attn_org;

        const res = res_opt orelse return statusCode(status);

        const t_logits = res.getLogits();
        const t_embd = if (res.getEmbdPooled()) |t| t else res.getEmbd();
        const t_h_nextn = if (self.cparams.embeddings_nextn) res.getHNextn() else null;

        // The logits.
        if (self.logits.data != null and t_logits != null) {
            const backend_res = c.ggml_backend_sched_get_tensor_backend(self.sched, t_logits);
            impl.assert(backend_res != null, "llama-context.cpp", 1848, "backend_res != nullptr");
            c.ggml_backend_tensor_get_async(backend_res, t_logits, self.logits.data, 0, self.n_outputs * n_vocab * @sizeOf(f32));
        }

        // The embeddings.
        if (self.embd.data != null and t_embd != null) {
            const backend_embd = c.ggml_backend_sched_get_tensor_backend(self.sched, t_embd);
            impl.assert(backend_embd != null, "llama-context.cpp", 1857, "backend_embd != nullptr");
            switch (self.cparams.pooling_type) {
                c.LLAMA_POOLING_TYPE_NONE => {
                    const n_embd_out: usize = hparams.nEmbdOut();
                    impl.assert(self.n_outputs * n_embd_out <= self.embd.size, "llama-context.cpp", 1866, "n_outputs*n_embd_out <= (int64_t) embd.size");
                    c.ggml_backend_tensor_get_async(backend_embd, t_embd, self.embd.data, 0, self.n_outputs * n_embd_out * @sizeOf(f32));
                },
                c.LLAMA_POOLING_TYPE_MEAN, c.LLAMA_POOLING_TYPE_CLS, c.LLAMA_POOLING_TYPE_LAST => {
                    // n_embd_out, not n_embd_inp: the pooled embedding has the
                    // model's output dimension.
                    self.extractSeqEmbeddings(&ubatch, backend_embd, t_embd.?, hparams.nEmbdOut());
                },
                c.LLAMA_POOLING_TYPE_RANK => self.extractSeqEmbeddings(&ubatch, backend_embd, t_embd.?, hparams.n_cls_out),
                else => impl.abort("llama-context.cpp", 1904, "unknown pooling type"),
            }
        }

        // The nextn embeddings, the hidden state before the output norm.
        if (self.embd_nextn.data != null and t_h_nextn != null and self.cparams.pooling_type == c.LLAMA_POOLING_TYPE_NONE) {
            const backend_h = c.ggml_backend_sched_get_tensor_backend(self.sched, t_h_nextn);
            impl.assert(backend_h != null, "llama-context.cpp", 1912, "backend_h != nullptr");
            const n_embd_out: usize = hparams.nEmbdOut();
            impl.assert(self.n_outputs * n_embd_out <= self.embd_nextn.size, "llama-context.cpp", 1915, "n_outputs*n_embd <= (int64_t) embd_nextn.size");
            c.ggml_backend_tensor_get_async(backend_h, t_h_nextn, self.embd_nextn.data, 0, self.n_outputs * n_embd_out * @sizeOf(f32));
        }

        // T5 keeps the encoder output for cross-attention; the C++'s TODO
        // calls this a hacky solution.
        if (self.model.arch == .t5) {
            if (t_embd) |te| {
                self.synchronize();

                self.cross.n_embd = te.ne[0];
                self.cross.n_enc = te.ne[1];
                const n: usize = @intCast(self.cross.n_embd * self.cross.n_enc);
                self.cross.v_embd.resize(allocator, n) catch oom();
                @memcpy(std.mem.sliceAsBytes(self.cross.v_embd.items), @as([*]const u8, @ptrCast(self.embd.data.?))[0..c.ggml_nbytes(te)]);

                const b = self.balloc.getBatch();

                // The sequences each encoder token belongs to, for the
                // decoder's cross-attention.
                const old = self.cross.seq_ids_enc.items.len;
                if (n_tokens < old) {
                    for (self.cross.seq_ids_enc.items[n_tokens..]) |*s| s.deinit(allocator);
                }
                self.cross.seq_ids_enc.resize(allocator, n_tokens) catch oom();
                for (self.cross.seq_ids_enc.items[@min(old, n_tokens)..]) |*s| s.* = .empty;
                for (0..n_tokens) |i| {
                    self.cross.seq_ids_enc.items[i].clearRetainingCapacity();
                    var s: usize = 0;
                    while (s < b.n_seq_id[i]) : (s += 1) self.cross.insertSeq(i, b.seq_id[i][s]);
                }
            }
        }

        return 0;
    }

    fn statusCode(status: c.enum_ggml_status) c_int {
        return switch (status) {
            c.GGML_STATUS_ABORTED => 2,
            c.GGML_STATUS_ALLOC_FAILED => -2,
            c.GGML_STATUS_FAILED => -3,
            c.GGML_STATUS_SUCCESS => impl.abort("llama-context.cpp", 1837, "should not happen"),
            else => impl.abort("llama-context.cpp", 1835, "unknown status"),
        };
    }

    /// Ports `llama_context`'s `decode` (llama-context.cpp:2000 @cd959bae9).
    ///
    /// Return: 0 on success, as `llama_decode` returns.
    pub fn decodeExt(self: *Context, batch_inp: *const batch.BatchExt) c_int {
        const func = "decode";

        const mem = self.mem orelse {
            impl.logDebug("%s: cannot decode batches with this context (calling encode() instead)\n", .{func});
            return self.encodeExt(batch_inp);
        };

        if (batch_inp.tokens.items.len == 0) {
            impl.logError("%s: n_tokens == 0\n", .{func});
            return -1;
        }

        if (batch_inp.n_embd > 0 and batch_inp.n_embd != batch_inp.n_embd_inp) {
            impl.logError("%s: embd row width %zu does not match the decoder input %zu\n", .{ func, batch_inp.n_embd, batch_inp.n_embd_inp });
            return -1;
        }

        const has_embd_row = batch_inp.tokens.items.len != 0 and batch_inp.tokens.items[0].embd_row >= 0;
        if (has_embd_row and !self.supportsEmbdRow()) {
            impl.logError("%s: this context cannot read embd rows (needs a DFlash/DSpark decoder and a layer input device buffer in ctx_other)\n", .{func});
            return -1;
        }

        const vocab = &self.model.vocab;
        const hparams = &self.model.hparams;

        const n_vocab: usize = vocab.nTokens();

        // Computing embeddings outputs every token.
        const output_all = self.cparams.embeddings;
        // Non-causal attention needs the whole sequence in one ubatch.
        const embd_all = output_all or !self.cparams.causal_attn;
        const has_samplers = self.sampling.samplers.items.items.len != 0;

        const n_seq_max: u32 = if (self.cparams.kv_unified) max_seq else self.cparams.n_seq_max;

        // Embedding contexts output every token even when none is marked.
        if (has_samplers) {
            var seq_output_count: [max_seq]i32 = @splat(0);
            for (batch_inp.tokens.items) |*tok| {
                if (!output_all and !tok.output) continue;
                for (tok.seq_ids.items.items) |seq_id| {
                    if (seq_id < 0 or @as(u32, @bitCast(seq_id)) >= n_seq_max) continue;
                    seq_output_count[@intCast(seq_id)] += 1;
                    if (self.sampling.samplers.get(seq_id) != null and seq_output_count[@intCast(seq_id)] > @as(i32, @intCast(self.cparams.n_outputs_max_per_seq))) {
                        impl.logError("%s: backend sampling supports at most %u outputs per sequence (seq_id %d had %d)\n", .{ func, self.cparams.n_outputs_max_per_seq, seq_id, seq_output_count[@intCast(seq_id)] });
                        return -1;
                    }
                }
            }
        }

        if (!self.balloc.initBatch(batch_inp, vocab, output_all)) {
            impl.logError("%s: failed to initialize batch\n", .{func});
            return -1;
        }

        const n_tokens_all = self.balloc.getNTokens();
        const n_outputs_all = self.balloc.getNOutputs();

        // Pooling needs every token output.
        if (output_all and n_outputs_all != n_tokens_all) {
            impl.logError("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n", .{ func, n_outputs_all, n_tokens_all });
            return -1;
        }

        impl.assert(n_tokens_all <= self.cparams.n_batch, "llama-context.cpp", 2081, "n_tokens_all <= cparams.n_batch");
        impl.assert(self.cparams.causal_attn or self.cparams.n_ubatch >= n_tokens_all, "llama-context.cpp", 2083, "(cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && \"non-causal attention requires n_ubatch >= n_tokens\"");

        // Wait for in-flight copies into embd_seq before freeing it.
        if (self.embd_seq.items.len != 0) self.synchronize();
        self.embdSeqClear();

        if (self.t_compute_start_us == 0) self.t_compute_start_us = c.ggml_time_us();
        self.n_queued_tokens += n_tokens_all;

        self.output_swaps.clearRetainingCapacity();
        self.embd_batch_idxs.clearRetainingCapacity();

        var exc: Exception = .{};
        self.schedReserve(&exc) catch exc.terminate("llama-context.cpp", 2100);

        var did_optimize = false;

        // Pending shifts and copies.
        _ = self.memoryUpdate(false);

        var mctx: memory.Context = undefined;
        while (true) {
            mctx = mem.initBatch(self.balloc, self.cparams.n_ubatch, embd_all);
            switch (mctx.getStatus()) {
                .success => {},
                .no_update => {
                    impl.logError("%s: unexpected memory context status: %d\n", .{ func, @backingInt(mctx.getStatus()) });
                    mctx.destroy();
                    return -2;
                },
                .failed_prepare => {
                    if (!did_optimize) {
                        did_optimize = true;
                        if (self.memoryUpdate(true)) {
                            impl.logDebug("%s: retrying batch size %d after cache optimization\n", .{ func, self.balloc.getNTokens() });
                            mctx.destroy();
                            continue;
                        }
                    }
                    impl.logWarn("%s: failed to find a memory slot for batch of size %d\n", .{ func, self.balloc.getNTokens() });
                    mctx.destroy();
                    return 1;
                },
                .failed_compute => {
                    impl.logError("%s: compute failed while preparing batch of size %d\n", .{ func, self.balloc.getNTokens() });
                    mctx.destroy();
                    return -2;
                },
            }
            break;
        }
        defer mctx.destroy();

        if (self.outputReserve(@intCast(n_outputs_all)) < n_outputs_all) {
            impl.logError("%s: could not reserve space for batch with %d outputs\n", .{ func, n_outputs_all });
            return -2;
        }

        // A new sampling transaction for this logical batch.
        for (self.sampling.samplers.items.items) |e| _Z27llama_sampler_backend_beginP13llama_sampler(e.sampler);

        var n_outputs_prev: usize = 0;
        var n_tokens_prev: usize = 0;

        // The ordering for the layer input device buffer.
        const inject_src: ?*Context = if (has_embd_row) @ptrCast(@alignCast(self.cparams.ctx_other)) else null;
        var inject_backend: c.ggml_backend_t = null;
        if (inject_src) |src| {
            inject_backend = if (src.ev_layer_inp_written != null) self.backendFromDev(src.layer_inp_dev_dev) else null;
            if (inject_backend != null) {
                // Wait on the host: a device-side wait would stall our queue
                // for the whole target batch, which trips the Metal command
                // buffer timeout on long prompts.
                c.ggml_backend_event_synchronize(src.ev_layer_inp_written);
            } else {
                src.synchronize();
            }
        }
        if (self.layer_inp_dev != null and self.ev_layer_inp_read != null and self.ev_layer_inp_read_recorded) {
            // Do not overwrite rows the reader has not consumed yet.
            c.ggml_backend_event_wait(self.layer_inp_backend, self.ev_layer_inp_read);
        }

        while (true) {
            const ubatch = mctx.getUbatch();

            // The outputs in this ubatch, before the graph is built.
            {
                var n_outputs_new: u32 = 0;
                if (n_outputs_all == n_tokens_all) {
                    n_outputs_new = ubatch.n_tokens;
                } else {
                    for (0..ubatch.n_tokens) |i| n_outputs_new += @intFromBool(ubatch.output[i] != 0);
                }
                self.n_outputs = n_outputs_new;
            }

            var status: c.enum_ggml_status = c.GGML_STATUS_SUCCESS;
            const res = self.processUbatch(ubatch, ctxTypeToGraphType(self.cparams.ctx_type), mctx, &status) orelse {
                // The ubatch failed or was aborted: remove its positions
                // from the memory.
                var pos_min: [max_seq]i32 = @splat(std.math.maxInt(i32));
                for (0..ubatch.n_tokens) |i| {
                    const seq_id: usize = @intCast(ubatch.seq_id[i][0]);
                    pos_min[seq_id] = @min(pos_min[seq_id], ubatch.pos[i]);
                }
                for (0..max_seq) |s| {
                    if (pos_min[s] == std.math.maxInt(i32)) continue;
                    impl.logWarn("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", .{ func, @as(c_int, @intCast(s)), pos_min[s] });
                    _ = mem.seqRm(@intCast(s), pos_min[s], -1);
                }
                return statusCode(status);
            };

            const t_logits = res.getLogits();
            var t_embd = if (self.cparams.embeddings) res.getEmbd() else null;
            const t_h_nextn = if (self.cparams.embeddings_nextn) res.getHNextn() else null;

            if (t_embd != null and res.getEmbdPooled() != null) t_embd = res.getEmbdPooled();

            const n_outputs: usize = self.n_outputs;

            // The logits.
            if (self.logits.data != null and t_logits != null and n_outputs > 0 and needsRawLogits(ubatch, &self.sampling.samplers)) {
                const backend_res = c.ggml_backend_sched_get_tensor_backend(self.sched, t_logits);
                impl.assert(backend_res != null, "llama-context.cpp", 2254, "backend_res != nullptr");
                impl.assert(self.logits.data != null, "llama-context.cpp", 2255, "logits.data != nullptr");
                const logits_out = self.logits.data.? + n_outputs_prev * n_vocab;
                if (n_outputs != 0) {
                    impl.assert(n_outputs_prev + n_outputs <= n_outputs_all, "llama-context.cpp", 2260, "n_outputs_prev + n_outputs <= n_outputs_all");
                    impl.assert((n_outputs_prev + n_outputs) * n_vocab <= self.logits.size, "llama-context.cpp", 2261, "(n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size");
                    c.ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs * n_vocab * @sizeOf(f32));
                }
            }

            // The embeddings.
            if (self.embd.data != null and t_embd != null and n_outputs > 0) {
                const backend_embd = c.ggml_backend_sched_get_tensor_backend(self.sched, t_embd);
                impl.assert(backend_embd != null, "llama-context.cpp", 2269, "backend_embd != nullptr");
                switch (self.cparams.pooling_type) {
                    c.LLAMA_POOLING_TYPE_NONE => {
                        const n_embd_out: usize = hparams.nEmbdOut();
                        const embd_out = self.embd.data.? + n_outputs_prev * n_embd_out;
                        impl.assert(n_outputs_prev + n_outputs <= n_outputs_all, "llama-context.cpp", 2260, "n_outputs_prev + n_outputs <= n_outputs_all");
                        impl.assert((n_outputs_prev + n_outputs) * n_embd_out <= self.embd.size, "llama-context.cpp", 2281, "(n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size");
                        c.ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs * n_embd_out * @sizeOf(f32));
                    },
                    c.LLAMA_POOLING_TYPE_MEAN, c.LLAMA_POOLING_TYPE_CLS, c.LLAMA_POOLING_TYPE_LAST => self.extractSeqEmbeddings(ubatch, backend_embd, t_embd.?, hparams.nEmbdOut()),
                    c.LLAMA_POOLING_TYPE_RANK => self.extractSeqEmbeddings(ubatch, backend_embd, t_embd.?, hparams.n_cls_out),
                    else => impl.abort("llama-context.cpp", 2321, "unknown pooling type"),
                }
            }

            // [TAG_EXTRACT_TARGET_EMBEDDINGS]
            var extract_all_idxs = self.extractLayerInputs(res, n_tokens_prev, ubatch.n_tokens);

            // The nextn embeddings; meaningful only without pooling.
            {
                const masked = self.cparams.embeddings_nextn_masked;
                const n_rows: usize = if (masked) n_outputs else ubatch.n_tokens;
                const offset: usize = if (masked) n_outputs_prev else n_tokens_prev;
                if (self.embd_nextn.data != null and t_h_nextn != null and n_rows > 0 and self.cparams.pooling_type == c.LLAMA_POOLING_TYPE_NONE) {
                    const backend_h = c.ggml_backend_sched_get_tensor_backend(self.sched, t_h_nextn);
                    impl.assert(backend_h != null, "llama-context.cpp", 2338, "backend_h != nullptr");
                    const n_embd_out: usize = hparams.nEmbdOut();
                    const embd_nextn_out = self.embd_nextn.data.? + offset * n_embd_out;
                    impl.assert((offset + n_rows) * n_embd_out <= self.embd_nextn.size, "llama-context.cpp", 2343, "(offset + n_rows)*n_embd <= (int64_t) embd_nextn.size");
                    c.ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows * n_embd_out * @sizeOf(f32));
                    extract_all_idxs = extract_all_idxs or !masked;
                }
            }

            if (extract_all_idxs) {
                impl.assert(ubatch.data != null and ubatch.data.?.batch_idxs.items.len == ubatch.n_tokens, "llama-context.cpp", 2350, "ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens");
                impl.assert(self.embd_batch_idxs.items.len == n_tokens_prev, "llama-context.cpp", 2351, "embd_batch_idxs.size() == (size_t) n_tokens_prev");
                self.embd_batch_idxs.appendSlice(allocator, ubatch.data.?.batch_idxs.items) catch oom();
            }

            if (has_samplers) {
                // The sampling data, copied from the backend asynchronously.
                const off: u32 = @intCast(n_outputs_prev);
                copyTensorAsyncRows(c.llama_token, res.t_sampled.items, self.sampling.sampled, 1, off, self.sched, null);
                copyTensorAsyncRows(f32, res.t_sampled_logits.items, self.sampling.logits, n_vocab, off, self.sched, &self.sampling.logits_count);
                copyTensorAsyncRows(f32, res.t_sampled_probs.items, self.sampling.probs, n_vocab, off, self.sched, &self.sampling.probs_count);
                copyTensorAsyncRows(c.llama_token, res.t_candidates.items, self.sampling.candidates, n_vocab, off, self.sched, &self.sampling.candidates_count);
            }

            n_outputs_prev += n_outputs;
            n_tokens_prev += ubatch.n_tokens;

            if (!mctx.next()) break;
        }

        if (self.layer_inp_dev != null and self.ev_layer_inp_written != null) {
            c.ggml_backend_event_record(self.ev_layer_inp_written, self.layer_inp_backend);
        }
        if (inject_src) |src| {
            if (inject_backend != null) {
                c.ggml_backend_event_record(src.ev_layer_inp_read, inject_backend);
                src.ev_layer_inp_read_recorded = true;
            } else {
                self.synchronize();
            }
        }

        // The batch's total, for llama_get_logits_ith.
        self.n_outputs = n_outputs_all;

        // The output mappings.
        if (self.n_outputs > 0) {
            var sorted_output = true;
            const out_ids = self.balloc.getOutIds();
            impl.assert(out_ids.items.len == self.n_outputs, "llama-context.cpp", 2391, "out_ids.size() == (size_t) n_outputs");

            for (0..self.n_outputs) |i| {
                const out_id = out_ids.items[i];
                self.output_ids.items[@intCast(out_id)] = @intCast(i);
                if (out_id != @as(i32, @intCast(i))) sorted_output = false;
            }

            // The outputs in the order of the caller's batch; this matters
            // mostly for recurrent models.
            if (!sorted_output and self.n_outputs > 1) {
                impl.assert(self.n_outputs == out_ids.items.len, "llama-context.cpp", 2404, "(size_t) n_outputs == out_ids.size()");
                // A selection sort, which minimizes swaps.
                var i: u32 = 0;
                while (i < self.n_outputs - 1) : (i += 1) {
                    var j_min = i;
                    var j = i + 1;
                    while (j < self.n_outputs) : (j += 1) {
                        if (out_ids.items[j] < out_ids.items[j_min]) j_min = j;
                    }
                    if (j_min == i) continue;
                    std.mem.swap(i32, &out_ids.items[i], &out_ids.items[j_min]);
                    // Applied lazily, when the outputs are read.
                    self.output_swaps.append(allocator, .{ .i0 = i, .i1 = j_min }) catch oom();
                }
                @memset(self.output_ids.items, -1);
                for (0..self.n_outputs) |k| self.output_ids.items[@intCast(out_ids.items[k])] = @intCast(k);
            }
        }
        return 0;
    }

    /// Ports `llama_context`'s `output_reserve` (llama-context.cpp:2442
    /// @cd959bae9): the host buffer every output is copied into.
    ///
    /// Return: the outputs reserved; 0 when allocation failed.
    pub fn outputReserve(self: *Context, n_outputs: i32) u32 {
        const func = "output_reserve";
        const hparams = &self.model.hparams;
        const n_outputs_max: usize = @intCast(@max(@as(i64, n_outputs), @as(i64, self.nSeqMax())));

        const n_batch: usize = self.cparams.n_batch;
        const n_vocab: usize = self.model.vocab.nTokens();
        const n_embd: usize = hparams.n_embd;
        const n_embd_out: usize = hparams.nEmbdOut();

        var has_logits = !self.cparams.embeddings or self.graph_has_logits;
        var has_embd = self.cparams.embeddings;
        const has_embd_nextn = self.cparams.embeddings_nextn;

        // Hacky encoder-decoder support.
        if (self.model.arch == .t5) {
            has_logits = true;
            has_embd = true;
        }

        var backend_float_count: usize = 0;
        var backend_token_count: usize = 0;
        var embd_layer_inp_float_count: usize = 0;

        self.logits.size = if (has_logits) n_vocab * n_outputs_max else 0;
        self.embd.size = if (has_embd) n_embd_out * n_outputs_max else 0;
        self.embd_nextn.size = if (has_embd_nextn) n_embd_out * n_outputs_max else 0;

        // Unmasked: a nextn row for every token in the batch.
        if (has_embd_nextn and !self.cparams.embeddings_nextn_masked) self.embd_nextn.size = n_embd_out * n_batch;

        for (self.embeddings_layer_inp) |enabled| {
            if (enabled and !self.cparams.embeddings_layer_inp_dev) embd_layer_inp_float_count += n_embd * n_batch;
        }

        // Backend sampling output buffers.
        const has_sampling = self.sampling.samplers.items.items.len != 0;
        if (has_sampling) {
            backend_float_count = 2 * n_vocab * n_outputs_max; // logits + probs
            backend_token_count = (1 + n_vocab) * n_outputs_max; // sampled + candidates
        }

        // Set once, never resized afterwards.
        if (self.output_ids.items.len == 0) self.output_ids.resize(allocator, n_batch) catch oom();

        const prev_size: usize = if (self.buf_output != null) c.ggml_backend_buffer_get_size(self.buf_output) else 0;
        const new_size = (self.logits.size + self.embd.size + self.embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * @sizeOf(f32) + backend_token_count * @sizeOf(c.llama_token);

        // Allocate only when the capacity is exceeded.
        if (self.buf_output == null or prev_size < new_size) {
            if (self.buf_output != null) {
                self.synchronize();
                c.ggml_backend_buffer_free(self.buf_output);
                self.buf_output = null;
                self.logits.data = null;
                self.embd.data = null;
                self.embd_nextn.data = null;
                @memset(self.embd_layer_inp, .{});
            }
            var buft = c.ggml_backend_cpu_buffer_type();
            // The output device's host buffer, for faster transfer.
            const output_dev = self.model.devOutput();
            const output_dev_host_buft = if (output_dev != null) c.ggml_backend_dev_host_buffer_type(output_dev) else null;
            if (output_dev_host_buft != null) buft = output_dev_host_buft;
            self.buf_output = c.ggml_backend_buft_alloc_buffer(buft, new_size);
            if (self.buf_output == null) {
                impl.logError("%s: failed to allocate output buffer of size %.2f MiB\n", .{ func, @as(f64, @floatFromInt(new_size)) / (1024.0 * 1024.0) });
                return 0;
            }
            c.ggml_backend_buffer_clear(self.buf_output, 0);
        }

        const base: [*]u8 = @ptrCast(c.ggml_backend_buffer_get_base(self.buf_output));
        var offset: usize = 0;

        self.logits = if (has_logits) .{ .data = @ptrCast(@alignCast(base)), .size = self.logits.size } else .{};
        offset += self.logits.size * @sizeOf(f32);

        self.embd = if (has_embd) .{ .data = @ptrCast(@alignCast(base + offset)), .size = self.embd.size } else .{};
        offset += self.embd.size * @sizeOf(f32);

        self.embd_nextn = if (has_embd_nextn) .{ .data = @ptrCast(@alignCast(base + offset)), .size = self.embd_nextn.size } else .{};
        offset += self.embd_nextn.size * @sizeOf(f32);

        for (self.embd_layer_inp, 0..) |*layer_inp, il| {
            if (self.embeddings_layer_inp[il] and !self.cparams.embeddings_layer_inp_dev) {
                layer_inp.* = .{ .data = @ptrCast(@alignCast(base + offset)), .size = n_embd * n_batch };
                offset += layer_inp.size * @sizeOf(f32);
            } else {
                layer_inp.* = .{};
            }
        }

        const s = &self.sampling;
        if (has_sampling) {
            s.logits = .{ .data = @ptrCast(@alignCast(base + offset)), .size = n_vocab * n_outputs_max };
            offset += s.logits.size * @sizeOf(f32);
            s.probs = .{ .data = @ptrCast(@alignCast(base + offset)), .size = n_vocab * n_outputs_max };
            offset += s.probs.size * @sizeOf(f32);
            s.sampled = .{ .data = @ptrCast(@alignCast(base + offset)), .size = n_outputs_max };
            offset += s.sampled.size * @sizeOf(c.llama_token);
            s.candidates = .{ .data = @ptrCast(@alignCast(base + offset)), .size = n_vocab * n_outputs_max };
            offset += s.candidates.size * @sizeOf(c.llama_token);

            // The counts actually copied per output row.
            s.logits_count.resize(allocator, n_outputs_max) catch oom();
            s.probs_count.resize(allocator, n_outputs_max) catch oom();
            s.candidates_count.resize(allocator, n_outputs_max) catch oom();
            @memset(s.logits_count.items, 0);
            @memset(s.probs_count.items, 0);
            @memset(s.candidates_count.items, 0);
            @memset(s.sampled.data.?[0..s.sampled.size], c.LLAMA_TOKEN_NULL);
        } else {
            s.logits = .{};
            s.probs = .{};
            s.sampled = .{};
            s.candidates = .{};
            s.logits_count.clearRetainingCapacity();
            s.probs_count.clearRetainingCapacity();
            s.candidates_count.clearRetainingCapacity();
        }

        // Every id invalid.
        @memset(self.output_ids.items, -1);
        self.n_outputs = 0;

        impl.assert(n_outputs_max <= self.cparams.n_outputs_max, "llama-context.cpp", 2599, "n_outputs_max <= cparams.n_outputs_max");
        return @intCast(n_outputs_max);
    }

    /// Ports `llama_context`'s `extract_layer_inputs` (llama-context.cpp:2604
    /// @cd959bae9): copy the enabled layers' inputs out asynchronously,
    /// unless they go to the device buffer.
    ///
    /// Return: whether any layer was copied.
    fn extractLayerInputs(self: *Context, res: *const graph.Result, token_offset: usize, n_tokens: usize) bool {
        var extracted = false;
        if (self.cparams.embeddings_layer_inp_dev) {
            return extracted;
        }
        for (self.embeddings_layer_inp, 0..) |enabled, il| {
            if (!enabled) continue;
            if (!self.embd_layer_inp[il].hasData()) impl.abort("llama-context.cpp", 2614, "output layer input buffer not allocated");
            const t = res.t_layer_inp[il] orelse impl.abort("llama-context.cpp", 2618, "layer input tensor not found");
            const nbytes = c.ggml_nbytes(t);
            const nfloats = nbytes / @sizeOf(f32);
            impl.assert(n_tokens > 0, "llama-context.cpp", 2623, "n_tokens > 0");
            impl.assert(nfloats % n_tokens == 0, "llama-context.cpp", 2624, "nfloats % n_tokens == 0");
            const row_floats = nfloats / n_tokens;
            impl.assert(row_floats == self.model.hparams.n_embd, "llama-context.cpp", 2627, "row_floats == model.hparams.n_embd");
            const dst_offset = token_offset * row_floats;
            impl.assert(dst_offset + nfloats <= self.embd_layer_inp[il].size, "llama-context.cpp", 2629, "dst_offset + nfloats <= embd_layer_inp[il].size");
            const backend = c.ggml_backend_sched_get_tensor_backend(self.sched, t);
            impl.assert(backend != null, "llama-context.cpp", 2632, "backend != nullptr");
            // A tensor-split backend needs a zero source offset.
            c.ggml_backend_tensor_get_async(backend, t, self.embd_layer_inp[il].data.? + dst_offset, 0, nbytes);
            extracted = true;
        }
        return extracted;
    }

    /// Ports `llama_context`'s `output_reorder` (llama-context.cpp:2640
    /// @cd959bae9): apply the swaps decode deferred, then put the rows that
    /// hold every token -- layer inputs, unmasked NextN embeddings -- back
    /// in batch order, whatever the logits selection was.
    pub fn outputReorder(self: *Context) void {
        const n_vocab: usize = self.model.vocab.nTokens();
        const n_embd: usize = self.model.hparams.n_embd;
        const n_embd_out: usize = self.model.hparams.nEmbdOut();
        for (self.output_swaps.items) |sw| {
            const i_0: usize = sw.i0;
            const i_1: usize = sw.i1;
            if (self.logits.size > 0) swapRows(f32, self.logits.data.?, i_0, i_1, n_vocab);
            if (self.embd.size > 0) swapRows(f32, self.embd.data.?, i_0, i_1, n_embd_out);
            if (self.embd_nextn.size > 0 and self.cparams.embeddings_nextn_masked) swapRows(f32, self.embd_nextn.data.?, i_0, i_1, n_embd_out);
            if (self.sampling.samplers.items.items.len != 0) {
                const s = &self.sampling;
                swapRows(f32, s.logits.data.?, i_0, i_1, n_vocab);
                swapRows(f32, s.probs.data.?, i_0, i_1, n_vocab);
                swapRows(c.llama_token, s.candidates.data.?, i_0, i_1, n_vocab);
                std.mem.swap(c.llama_token, &s.sampled.data.?[i_0], &s.sampled.data.?[i_1]);
                std.mem.swap(u32, &s.logits_count.items[i_0], &s.logits_count.items[i_1]);
                std.mem.swap(u32, &s.probs_count.items[i_0], &s.probs_count.items[i_1]);
                std.mem.swap(u32, &s.candidates_count.items[i_0], &s.candidates_count.items[i_1]);
            }
        }
        self.output_swaps.clearRetainingCapacity();

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        const idxs = self.embd_batch_idxs.items;
        for (0..idxs.len) |i| {
            while (idxs[i] != @as(i32, @intCast(i))) {
                const j = idxs[i];
                impl.assert(j >= 0 and @as(usize, @intCast(j)) < idxs.len, "llama-context.cpp", 2702, "j >= 0 && (size_t) j < embd_batch_idxs.size()");
                const ju: usize = @intCast(j);
                if (self.embd_nextn.hasData() and !self.cparams.embeddings_nextn_masked) swapRows(f32, self.embd_nextn.data.?, i, ju, n_embd_out);
                for (self.embd_layer_inp) |li| {
                    if (li.hasData()) swapRows(f32, li.data.?, i, ju, n_embd);
                }
                std.mem.swap(i32, &idxs[i], &idxs[ju]);
            }
        }
        self.embd_batch_idxs.clearRetainingCapacity();
    }

    fn swapRows(comptime T: type, data: [*]T, i_0: usize, i_1: usize, n: usize) void {
        for (0..n) |k| std.mem.swap(T, &data[i_0 * n + k], &data[i_1 * n + k]);
    }

    /// Ports `llama_context`'s `graph_max_nodes` (llama-context.cpp:2725 @cd959bae9).
    pub fn graphMaxNodes(self: *const Context, n_tokens: u32) u32 {
        const m = self.model;
        const n_tensors: u32 = @intCast(m.nTensors());
        var res: u32 = undefined;
        if (m.arch == .kimi_k3 or m.arch == .glm5_next) {
            // n_tokens*40 runs out at ubatch 3840.
            res = @max(n_tokens *% 160, 64 *% n_tensors);
        } else if (m.arch == .hrm_text) {
            // the 128-slot looped graph needs roughly one stack per token budget
            res = @max(n_tokens *% 80, 64 *% n_tensors);
        } else if (m.arch == .qwen3next or m.arch == .kimi_linear or m.arch == .metis16 or m.arch == .bailingmoe3 or m.arch == .qwen35 or
            m.arch == .qwen35moe or m.arch == .clef or m.arch == .clover_moe or m.arch == .qwen4exp or m.arch == .deepseek4 or m.arch == .deepseek41 or (m.arch == .dflash and m.hparams.dsv4_hc_mult > 0) or
            m.arch == .nanbeige or m.arch == .minimax_01 or m.arch == .minimax_m3 or m.arch == .hy_v4)
        {
            res = @max(n_tokens *% 40, 32 *% n_tensors);
        } else if (m.arch == .dflash and m.hparams.dflash_selector_rank > 0) {
            // DFlash2's convolutions and selector are shape work rather than
            // matmuls: ~8.6 nodes a tensor against ~5.9 for a plain draft.
            res = @max(1024, 12 *% n_tensors);
        } else {
            res = @max(1024, 8 *% n_tensors);
            // `model.loras`: the LoRA adapters loaded against the model.
            for (m.loras.items) |lora| res +%= lora.getNNodes();
        }

        // every sampler builds n_outputs_max_per_seq chains, see
        // `Builder.buildSampling`
        for (self.sampling.samplers.items.items) |e| {
            res +%= _Z29llama_sampler_backend_n_nodesPK13llama_sampler(e.sampler) *% self.cparams.n_outputs_max_per_seq;
        }

        if (self.cparams.training) res *%= 4;

        return res;
    }

    /// Ports `llama_context`'s `get_gf_res_reserve` (llama-context.cpp:2773 @cd959bae9).
    pub fn getGfResReserve(self: *const Context) *graph.Result {
        return self.gf_res_reserve.?;
    }

    /// Ports `llama_context`'s `get_gf_res_prev` (llama-context.cpp:2777
    /// @cd959bae9): the arena for a ubatch with outputs or for one without,
    /// created on first use with the reserve's node budget.
    pub fn getGfResPrev(self: *Context) *graph.Result {
        const res = &self.gf_res_prev[@intFromBool(self.n_outputs > 0)];
        if (res.* == null) res.* = graph.Result.create(self.gf_res_reserve.?.getMaxNodes());
        return res.*.?;
    }

    /// Ports `llama_context`'s `graph_reserve` (llama-context.cpp:2842
    /// @cd959bae9): build the worst-case graph for a ubatch shape and size
    /// the scheduler's buffers for it.
    ///
    /// Return: the graph, null when the buffers could not be allocated, or
    /// `error.Thrown` when a `prism.hadamard` model's graph skips a
    /// transform.
    pub fn graphReserve(self: *Context, n_tokens_: u32, n_seqs: u32, n_outputs: u32, mctx: ?memory.Context, split_only: bool, sizes: ?[*]usize, exc: *Exception) Thrown!?*c.ggml_cgraph {
        const func = "graph_reserve";
        var n_tokens = n_tokens_;
        impl.logDebug("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", .{ func, n_tokens, n_seqs, n_outputs });
        impl.assert(n_outputs >= 1, "llama-context.cpp", 2845, "n_outputs >= 1");

        if (n_tokens % n_seqs != 0) {
            // Round up to a multiple of n_seqs.
            n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs;
            impl.logDebug("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", .{ func, n_tokens, n_seqs, n_outputs });
        }

        c.ggml_backend_sched_reset(self.sched);

        // A reset scheduler cannot reuse old graphs.
        for (self.gf_res_prev) |res| if (res) |r| r.reset();
        self.gf_res_prev_active = null;

        // Keep n_outputs and restore it after.
        const save_n_outputs = self.n_outputs;
        self.n_outputs = n_outputs;

        var balloc = batch.BatchAllocr.init(self.model.hparams.nPosPerEmbd(), false);
        defer balloc.deinit();
        var ubatch = balloc.ubatchReserve(n_tokens / n_seqs, n_seqs);
        defer ubatch.release();

        ubatchPrepareReserve(&ubatch, n_outputs, &self.sampling.samplers, self.cparams.n_outputs_max_per_seq);

        const res = self.gf_res_reserve.?;
        var gparams = self.graphParams(res, &ubatch, mctx, ctxTypeToGraphType(self.cparams.ctx_type));
        defer gparams.samplers.deinit();

        res.reset();

        // `llama_model_deepseek41`'s `build_arch_graph` throws for an MTP
        // graph (models/deepseek41.cpp:220 @cd959bae9). The architectures'
        // builders have no error path, and the first reservation reaches it.
        if (self.model.arch == .deepseek41 and gparams.gtype == .decoder_mtp) {
            return exc.throw(.runtime_error, "DeepSeek-V4.1 has no MTP graph: its DSpark draft is a separate DFLASH GGUF (convert_hf_to_gguf.py --dspark), used with --spec-type draft-dspark", .{});
        }

        const gf = self.model.buildGraph(&gparams);

        // check the graph before scheduling: cross-backend copies break the
        // producer chain the check follows
        const hdmd = &self.model.hdmd;
        if (!self.hadamard_verified and (hdmd.rot.count() != 0 or hdmd.inv.count() != 0)) {
            self.n_outputs = save_n_outputs;
            try verifyHadamardGraph(gf, &hdmd.rot, &hdmd.inv, self.moe_cache, exc);
            self.hadamard_verified = true;
        }

        self.n_input_tensors = @intCast(graphNInputTensors(gf));
        self.n_outputs = save_n_outputs;

        // Initialize the scheduler with the graph.
        if (split_only) {
            if (sizes) |s| {
                c.ggml_backend_sched_reserve_size(self.sched, gf, s);
            } else {
                _ = c.ggml_backend_sched_split_graph(self.sched, gf);
            }
        } else if (!c.ggml_backend_sched_reserve(self.sched, gf)) {
            impl.assert(sizes == null, "llama-context.cpp", 2898, "!sizes");
            impl.logError("%s: failed to allocate compute buffers\n", .{func});
            return null;
        }
        return gf;
    }

    /// Ports `llama_context`'s `graph_params` (llama-context.cpp:2906 @cd959bae9).
    ///
    /// Return: the parameters; the caller deinits their `samplers`, the
    /// copy the C++ makes of the map.
    pub fn graphParams(self: *Context, res: *graph.Result, ubatch: *const Ubatch, mctx: ?memory.Context, gtype: graph.GraphType) graph.Params {
        return .{
            .arch = self.model.arch,
            .hparams = &self.model.hparams,
            .cparams = self.cparams,
            .ubatch = ubatch.*,
            .gtype = gtype,
            .sched = self.sched,
            .backend_cpu = self.backend_cpu,
            .cvec = self.cvec,
            .loras = self.loras,
            .mctx = mctx,
            .cross = &self.cross,
            .moe_cache = self.moe_cache,
            .prec_policy = &self.model.prec_policy,
            .hdmd = if (self.model.hdmd.rot.count() == 0) null else &self.model.hdmd,
            .layer_inp_dev = self.layer_inp_dev,
            .layer_inp_src = if (self.supportsEmbdRow()) @as(*const Context, @ptrCast(@alignCast(self.cparams.ctx_other.?))).getLayerInpDev() else null,
            .samplers = self.sampling.samplers.clone(),
            .n_outputs = self.n_outputs,
            .cb = .{ .ctx = self, .func = graphCb },
            .res = res,
        };
    }

    /// Ports `llama_context`'s `sched_graph_realloc` (llama-context.cpp:2935
    /// @cd959bae9): the scheduler's realloc callback, which counts and traces
    /// this context's graph reallocations. Reallocating after the backend
    /// assignments changed is expected, so only the others are traced.
    fn schedGraphRealloc(sched: c.ggml_backend_sched_t, gf: ?*c.ggml_cgraph, backend_ids_changed: bool, graph_size_changed: bool, user_data: ?*anyopaque) callconv(.c) void {
        if (backend_ids_changed) {
            return;
        }

        const self: *Context = @ptrCast(@alignCast(user_data.?));

        if (graph_size_changed) {
            self.n_reallocs_diff += 1;
        } else {
            self.n_reallocs_same += 1;
        }

        impl.logInfo("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n", .{
            "sched_graph_realloc",
            c.ggml_graph_n_nodes(gf),
            @as(c_int, @intFromBool(graph_size_changed)),
            c.ggml_backend_sched_get_n_splits(sched),
            self.n_reallocs_same,
            self.n_reallocs_diff,
        });
    }

    /// Ports `llama_context`'s `sched_copy_experts` (llama-context.cpp:2985
    /// @cd959bae9): the scheduler's copy callback. A slot map of the MoE
    /// cache goes to the cache; a host expert tensor read by the split's
    /// first `MUL_MAT_ID` copies only the experts its ids use, from the
    /// cache where they are in it.
    ///
    /// Return: false when the copy is not one the callback makes, for the
    /// scheduler to make in full.
    fn schedCopyExperts(backend: c.ggml_backend_t, src_: [*c]const Tensor, dst_: [*c]Tensor, graph_: ?*c.ggml_cgraph, user_data: ?*anyopaque) callconv(.c) bool {
        const self: *Context = @ptrCast(@alignCast(user_data.?));
        const src: *const Tensor = @ptrCast(src_);
        const dst: *Tensor = @ptrCast(dst_);
        const gr = graph_.?;

        // the slot maps of the MoE cache
        if (self.moe_cache) |mc| {
            if (mc.copy(backend, src, dst, gr)) return true;
        }

        const st = &self.copy_experts;

        // the ids must be computed before the split starts, so only the
        // first node of the split is considered
        if (c.ggml_graph_n_nodes(gr) == 0) {
            return false;
        }
        const node: *const Tensor = c.ggml_graph_node(gr, 0);
        if (node.op != c.GGML_OP_MUL_MAT_ID or node.src[0] != dst) {
            return false;
        }

        const ids: *const Tensor = node.src[2];
        if (c.ggml_nelements(ids) == 0) {
            return true;
        }

        const n_expert: i64 = src.ne[2];
        const expert_size: usize = src.nb[2];

        if (ids != @as(?*const Tensor, st.ids) or @as(i64, @intCast(st.used.items.len)) != n_expert) {
            st.ids_data.resize(allocator, c.ggml_nbytes(ids) / @sizeOf(i32)) catch oom();
            c.ggml_backend_tensor_get_async(backend, ids, if (st.ids_data.items.len == 0) null else st.ids_data.items.ptr, 0, c.ggml_nbytes(ids));
            c.ggml_backend_synchronize(backend);

            st.used.resize(allocator, @intCast(n_expert)) catch oom();
            @memset(st.used.items, false);
            var i_1: i64 = 0;
            while (i_1 < ids.ne[1]) : (i_1 += 1) {
                var i_0: i64 = 0;
                while (i_0 < ids.ne[0]) : (i_0 += 1) {
                    const at = @as(usize, @intCast(i_1)) * ids.nb[1] / @sizeOf(i32) + @as(usize, @intCast(i_0)) * ids.nb[0] / @sizeOf(i32);
                    const id = st.ids_data.items[at];
                    impl.assert(id >= 0 and id < n_expert, "llama-context.cpp", 3021, "id >= 0 && id < n_expert");
                    st.used.items[@intCast(id)] = true;
                }
            }

            st.ids = ids;
        }

        const src_data: [*]const u8 = @ptrCast(src.data.?);

        // group consecutive experts and copy them together
        var first: i64 = 0;
        while (first < n_expert) {
            if (!st.used.items[@intCast(first)]) {
                first += 1;
                continue;
            }
            var last = first;
            while (last + 1 < n_expert and st.used.items[@intCast(last + 1)]) {
                last += 1;
            }

            // the experts in the MoE cache are copied from device memory,
            // the others are uploaded
            var next = first;
            if (self.moe_cache) |mc| {
                var e = first;
                while (e <= last) {
                    const n = mc.copyExperts(backend, src, dst, e, last);
                    if (n == 0) {
                        e += 1;
                        continue;
                    }
                    if (next < e) {
                        const off = @as(usize, @intCast(next)) * expert_size;
                        c.ggml_backend_tensor_set_async(backend, dst, src_data + off, off, @as(usize, @intCast(e - next)) * expert_size);
                    }
                    e += n;
                    next = e;
                }
            }

            // copy a bit extra to ensure there are no NaNs in the padding of
            // the last expert, which CUDA's MMQ needs
            const offset = @as(usize, @intCast(next)) * expert_size;
            const padding: usize = if (last < n_expert - 1) @min(expert_size, 512) else 0;
            const size = @as(usize, @intCast(last + 1 - next)) * expert_size + padding;
            if (size > 0) {
                c.ggml_backend_tensor_set_async(backend, dst, src_data + offset, offset, size);
            }

            first = last + 1;
        }

        return true;
    }

    /// Ports `llama_context`'s `graph_compute` (llama-context.cpp:2954 @cd959bae9).
    pub fn graphCompute(self: *Context, gf: *c.ggml_cgraph, batched: bool) c.enum_ggml_status {
        const n_threads: c_int = if (batched) self.cparams.n_threads_batch else self.cparams.n_threads;
        const tp = if (batched) self.threadpool_batch else self.threadpool;

        if (self.backend_cpu != null) {
            const reg = c.ggml_backend_dev_backend_reg(c.ggml_backend_get_device(self.backend_cpu));
            const set_threadpool_fn: ?*const fn (c.ggml_backend_t, c.ggml_threadpool_t) callconv(.c) void = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool")));
            if (set_threadpool_fn) |f| f(self.backend_cpu, tp);
        }

        // The thread count for every backend.
        for (self.set_n_threads_fns.items) |e| e.f(e.backend, n_threads);

        self.copy_experts.reset();

        const status = c.ggml_backend_sched_graph_compute_async(self.sched, gf);
        if (status != c.GGML_STATUS_SUCCESS) {
            impl.logError("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", .{ "graph_compute", status });
        }
        return status;
    }

    /// Ports the lambda `llama_context`'s `graph_get_cb` returns
    /// (llama-context.cpp:3069 @cd959bae9): name each tensor, and pin a
    /// layer's norm to its device.
    fn graphCb(ctx: ?*anyopaque, ubatch: *const Ubatch, cur: *Tensor, name: [*:0]const u8, il: c_int) void {
        const self: *Context = @ptrCast(@alignCast(ctx.?));
        if (il >= 0) {
            _ = c.ggml_format_name(cur, "%s-%d", name, il);
        } else {
            _ = c.ggml_set_name(cur, name);
        }

        // The norm may be assigned to the previous layer's backend, and a
        // layer's last op to the next layer's; pin them.
        const full_offload = self.model.nGpuLayers() > self.model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 or full_offload) {
            const n = std.mem.span(name);
            if (il != -1 and (std.mem.eql(u8, n, "norm") or std.mem.eql(u8, n, "l_last"))) {
                const dev_layer = self.model.devLayer(@intCast(il));
                for (self.backends.items) |backend| {
                    if (c.ggml_backend_get_device(backend) == dev_layer) {
                        if (c.ggml_backend_supports_op(backend, cur)) c.ggml_backend_sched_set_tensor_backend(self.sched, cur, backend);
                    }
                }
            }
        }
    }

    /// Ports `llama_context`'s `perf_get_data` (llama-context.cpp:3942 @cd959bae9).
    pub fn perfGetData(self: *const Context) c.llama_perf_context_data {
        return .{
            .t_start_ms = 1e-3 * @as(f64, @floatFromInt(self.t_start_us)),
            .t_load_ms = 1e-3 * @as(f64, @floatFromInt(self.t_load_us)),
            .t_p_eval_ms = 1e-3 * @as(f64, @floatFromInt(self.t_p_eval_us)),
            .t_eval_ms = 1e-3 * @as(f64, @floatFromInt(self.t_eval_us)),
            .n_p_eval = @max(1, self.n_p_eval),
            .n_eval = @max(1, self.n_eval),
            .n_reused = @max(0, self.n_reused),
            .n_reallocs_same = self.n_reallocs_same,
            .n_reallocs_diff = self.n_reallocs_diff,
        };
    }

    /// Ports `llama_context`'s `perf_reset` (llama-context.cpp:3959 @cd959bae9).
    pub fn perfReset(self: *Context) void {
        self.t_start_us = c.ggml_time_us();
        self.t_eval_us = 0;
        self.n_eval = 0;
        self.t_p_eval_us = 0;
        self.n_p_eval = 0;
        self.n_reused = 0;

        // the reallocation counters are not reset: they track the lifetime
        // totals for this context
    }

    /// Ports `llama_context`'s `memory_breakdown` (llama-context.cpp:3968
    /// @cd959bae9): bytes per buffer type, for the model, the context and
    /// the compute buffers, in buffer-type pointer order.
    ///
    /// Return: the entries, the caller's to free with `impl.allocator`.
    pub fn memoryBreakdown(self: *const Context) []MemoryBreakdownEntry {
        var ret: std.ArrayList(MemoryBreakdownEntry) = .empty;
        const mb_model = self.model.memoryBreakdown();
        defer allocator.free(mb_model);
        for (mb_model) |e| breakdownAt(&ret, e.buft).model += e.size;
        if (self.mem) |m| {
            var list: std.ArrayList(memory.BuftSize) = .empty;
            defer list.deinit(allocator);
            m.memoryBreakdown(&list);
            for (list.items) |e| breakdownAt(&ret, e.buft).context += e.size;
        }
        if (self.moe_cache) |mc| {
            const list = mc.memoryBreakdown();
            defer allocator.free(list);
            for (list) |e| breakdownAt(&ret, e.buft).context += e.size;
        }
        for (self.backends.items, 0..) |backend, i| {
            const buft = c.ggml_backend_sched_get_buffer_type(self.sched, backend);
            breakdownAt(&ret, buft).compute += if (self.model.noAlloc()) self.backend_buf_exp_size.items[i] else c.ggml_backend_sched_get_buffer_size(self.sched, backend);
        }
        return ret.toOwnedSlice(allocator) catch oom();
    }

    fn breakdownAt(list: *std.ArrayList(MemoryBreakdownEntry), buft: c.ggml_backend_buffer_type_t) *MemoryBreakdownEntry {
        var pos: usize = 0;
        while (pos < list.items.len) : (pos += 1) {
            if (list.items[pos].buft == buft) return &list.items[pos];
            if (@intFromPtr(list.items[pos].buft) > @intFromPtr(buft)) break;
        }
        list.insert(allocator, pos, .{ .buft = buft }) catch oom();
        return &list.items[pos];
    }
};

/// `GGML_PAD`.
fn padTo(x: u32, n: u32) u32 {
    return (x + n - 1) / n * n;
}

/// Ports `llama_flash_attn_type_name` (src/llama.cpp:38 @cd959bae9).
pub fn flashAttnTypeName(t: c.enum_llama_flash_attn_type) [:0]const u8 {
    return switch (t) {
        c.LLAMA_FLASH_ATTN_TYPE_AUTO => "auto",
        c.LLAMA_FLASH_ATTN_TYPE_DISABLED => "disabled",
        c.LLAMA_FLASH_ATTN_TYPE_ENABLED => "enabled",
        else => impl.abort("llama.cpp", 47, "fatal error"),
    };
}

/// Ports `copy_tensor_async_rows` (llama-context.cpp:1949 @cd959bae9):
/// each non-null tensor's data into its row of `dst`, counting what was
/// copied.
fn copyTensorAsyncRows(comptime T: type, tensors: []const ?*Tensor, dst: BufferView(T), stride: usize, row_offset: u32, sched: c.ggml_backend_sched_t, counts: ?*std.ArrayList(u32)) void {
    if (!dst.hasData()) return;
    for (tensors, 0..) |t, i| {
        const tensor = t orelse continue;
        const row: usize = row_offset + @as(u32, @intCast(i));
        const n_elements: usize = @intCast(c.ggml_nelements(tensor));
        impl.assert(c.ggml_is_contiguous(tensor), "llama-context.cpp", 1968, "ggml_is_contiguous(tensor) && \"sampling tensor must be contiguous for async copy\"");
        impl.assert(n_elements <= stride, "llama-context.cpp", 1969, "n_elements <= stride");
        impl.assert(row * stride + n_elements <= dst.size, "llama-context.cpp", 1970, "(size_t) row * stride + n_elements <= dst.size");
        const backend = c.ggml_backend_sched_get_tensor_backend(sched, tensor);
        const row_ptr = dst.data.? + row * stride;
        c.ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, c.ggml_nbytes(tensor));
        if (counts) |cn| {
            impl.assert(row < cn.items.len, "llama-context.cpp", 1977, "row < counts->size()");
            cn.items[row] = @intCast(n_elements);
        }
    }
}

/// Ports `needs_raw_logits` (llama-context.cpp:1983 @cd959bae9): whether
/// some output has a sequence without a backend sampler.
fn needsRawLogits(ubatch: *const Ubatch, samplers: *const graph.SamplerMap) bool {
    for (0..ubatch.n_tokens) |i| {
        if (ubatch.output[i] == 0) continue;
        for (0..@intCast(ubatch.n_seq_id[i])) |j| {
            if (samplers.get(ubatch.seq_id[i][j]) == null) return true;
        }
    }
    // Every sequence uses backend sampling.
    return false;
}

/// Ports `ubatch_prepare_reserve` (llama-context.cpp:2786 @cd959bae9):
/// pack the outputs into the sequences with samplers first.
fn ubatchPrepareReserve(ubatch: *Ubatch, n_outputs: u32, samplers: *const graph.SamplerMap, n_outputs_max_per_seq: u32) void {
    const n_seqs = ubatch.n_seqs;
    const n_seq_tokens = ubatch.n_seq_tokens;

    for (0..n_seqs) |s| {
        for (0..n_seq_tokens) |t| {
            const i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // The sequences with a sampler that fit in this ubatch.
    var sampler_seqs: std.ArrayList(u32) = .empty;
    defer sampler_seqs.deinit(allocator);
    const has_sampler = allocator.alloc(bool, n_seqs) catch oom();
    defer allocator.free(has_sampler);
    @memset(has_sampler, false);
    for (samplers.items.items) |e| {
        if (e.seq_id < 0 or @as(u32, @bitCast(e.seq_id)) >= n_seqs) continue;
        sampler_seqs.append(allocator, @intCast(e.seq_id)) catch oom();
        has_sampler[@intCast(e.seq_id)] = true;
    }

    var n_outputs_set: u32 = 0;
    const n_outputs_per_seq = @min(n_seq_tokens, n_outputs_max_per_seq);
    for (sampler_seqs.items) |s| {
        if (n_outputs_set >= n_outputs) break;
        var t: u32 = 0;
        while (t < n_outputs_per_seq and n_outputs_set < n_outputs) : (t += 1) {
            ubatch.output[s * n_seq_tokens + t] = 1;
            n_outputs_set += 1;
        }
    }

    // The sequences without samplers take the rest.
    var t: u32 = 0;
    while (t < n_seq_tokens and n_outputs_set < n_outputs) : (t += 1) {
        var s: u32 = 0;
        while (s < n_seqs and n_outputs_set < n_outputs) : (s += 1) {
            if (has_sampler[s]) continue;
            ubatch.output[s * n_seq_tokens + t] = 1;
            n_outputs_set += 1;
        }
    }
}

/// Ports `llama_context_default_params` (llama-context.cpp:4244 @cd959bae9).
pub fn defaultParams() c.llama_context_params {
    return .{
        .n_ctx = 512,
        .n_batch = 2048,
        .n_ubatch = 512,
        .n_seq_max = 1,
        .n_rs_seq = 0,
        .n_outputs_max = 0,
        .n_outputs_max_per_seq = 1,
        .n_threads = c.GGML_DEFAULT_N_THREADS,
        .n_threads_batch = c.GGML_DEFAULT_N_THREADS,
        .ctx_type = c.LLAMA_CONTEXT_TYPE_DEFAULT,
        .rope_scaling_type = c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        .pooling_type = c.LLAMA_POOLING_TYPE_UNSPECIFIED,
        .attention_type = c.LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        .flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_AUTO,
        .rope_freq_base = 0.0,
        .rope_freq_scale = 0.0,
        .yarn_ext_factor = -1.0,
        .yarn_attn_factor = -1.0,
        .yarn_beta_fast = -1.0,
        .yarn_beta_slow = -1.0,
        .yarn_orig_ctx = 0,
        .defrag_thold = -1.0,
        .cb_eval = null,
        .cb_eval_user_data = null,
        .type_k = c.GGML_TYPE_F16,
        .type_v = c.GGML_TYPE_F16,
        .moe_cache_size = 0,
        .abort_callback = null,
        .abort_callback_data = null,
        .embeddings = false,
        .offload_kqv = true,
        .no_perf = true,
        .op_offload = true,
        .swa_full = true,
        .kv_unified = false,
        .samplers = null,
        .n_samplers = 0,
        .ctx_other = null,
    };
}

/// Ports `llama_init_from_model` (llama-context.cpp:4289 @cd959bae9). The
/// model is mutable, as the C++'s is: a custom YaRN scale re-adjusts its
/// `n_ctx_train`.
///
/// Return: the context, the caller's to `destroy`, or null.
pub fn initFromModel(model_: ?*Model, params_: c.llama_context_params) ?*Context {
    const func = "llama_init_from_model";
    var params = params_;
    const model = model_ orelse {
        impl.logError("%s: model cannot be NULL\n", .{func});
        return null;
    };
    const hp = &model.hparams;

    if (params.n_batch == 0 and params.n_ubatch == 0) {
        impl.logError("%s: n_batch and n_ubatch cannot both be zero\n", .{func});
        return null;
    }
    if (params.n_ctx == 0 and hp.n_ctx_train == 0) {
        impl.logError("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", .{func});
        return null;
    }
    if (params.flash_attn_type != c.LLAMA_FLASH_ATTN_TYPE_DISABLED and model.arch == .grok) {
        impl.logWarn("%s: flash_attn is not compatible with Grok - forcing off\n", .{func});
        params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (model.params.split_mode == c.LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == c.LLAMA_FLASH_ATTN_TYPE_AUTO) {
            impl.logInfo("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", .{func});
            params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != c.LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            impl.logError("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", .{func});
            return null;
        }
        if (model.get_split_state_ud.n_devices == 1) {
            impl.logWarn("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", .{func});
        }
    }
    if ((hp.isMla() or model.arch == .deepseek4 or model.arch == .deepseek41) and params.type_k != params.type_v) {
        impl.logError("%s: model does not support different K (%s) and V (%s) cache types\n", .{ func, c.ggml_type_name(params.type_k), c.ggml_type_name(params.type_v) });
        return null;
    }
    if (c.ggml_is_quantized(params.type_v) and params.flash_attn_type != c.LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == c.LLAMA_FLASH_ATTN_TYPE_AUTO) {
            impl.logInfo("%s: enabling flash_attn since it is required for quantized V cache\n", .{func});
            params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == c.LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            impl.logError("%s: quantized V cache requires flash_attn to be enabled\n", .{func});
            return null;
        }
    }
    if (params.flash_attn_type != c.LLAMA_FLASH_ATTN_TYPE_DISABLED and c.ggml_is_quantized(params.type_k)) {
        const blck_size: u32 = @intCast(c.ggml_blck_size(params.type_k));
        var il: u32 = 0;
        while (il < hp.nLayer()) : (il += 1) {
            if (hp.nEmbdHeadK(il) % blck_size != 0) {
                impl.logError("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n", .{ func, c.ggml_type_name(params.type_k), blck_size, hp.nEmbdHeadK(il) });
                return null;
            }
        }
    }
    if (params.flash_attn_type != c.LLAMA_FLASH_ATTN_TYPE_DISABLED and c.ggml_is_quantized(params.type_v)) {
        const blck_size: u32 = @intCast(c.ggml_blck_size(params.type_v));
        var il: u32 = 0;
        while (il < hp.nLayer()) : (il += 1) {
            if (hp.nEmbdHeadV(il) % blck_size != 0) {
                impl.logError("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n", .{ func, c.ggml_type_name(params.type_v), blck_size, hp.nEmbdHeadV(il) });
                return null;
            }
        }
    }
    if (params.pooling_type != c.LLAMA_POOLING_TYPE_UNSPECIFIED and params.pooling_type != hp.pooling_type) {
        // The user's pooling type differs from the model's default.
        impl.logWarn("%s: model default pooling_type is [%d], but [%d] was specified\n", .{ func, hp.pooling_type, params.pooling_type });
    }
    // router_layer >= 0 repurposes n_layer_nextn for a router layer.
    if (params.ctx_type == c.LLAMA_CONTEXT_TYPE_MTP and (hp.n_layer_nextn == 0 or hp.router_layer >= 0)) {
        impl.logWarn("%s: context type MTP requested but model doesn't contain MTP layers\n", .{func});
        return null;
    }

    var exc: Exception = .{};
    const ctx = Context.create(model, params, &exc) catch {
        exc.buf[exc.len] = 0;
        impl.logError("%s: failed to initialize the context: %s\n", .{ func, @as([*:0]const u8, @ptrCast(&exc.buf)) });
        return null;
    };
    const cparams = &ctx.cparams;
    if (cparams.rope_scaling_type == c.LLAMA_ROPE_SCALING_TYPE_YARN and cparams.rope_freq_scale != model.hparams.rope_freq_scale_train) {
        impl.logInfo("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", .{ func, model.hparams.n_ctx_train });
        // `uint32_t / float` is a float division, truncated on assignment;
        // aarch64's `fcvtzu` saturates, as `lossyCast` does.
        model.hparams.n_ctx_train = std.math.lossyCast(u32, @as(f32, @floatFromInt(cparams.n_ctx_orig_yarn)) / cparams.rope_freq_scale);
        impl.logInfo("%s: n_ctx_train adjusted to %u\n", .{ func, model.hparams.n_ctx_train });
    }
    return ctx;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Context);
}

test "padding rounds up to the multiple" {
    try std.testing.expectEqual(@as(u32, 512), padTo(300, 256));
    try std.testing.expectEqual(@as(u32, 256), padTo(256, 256));
}
