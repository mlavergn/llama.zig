//! vLLM Decision 2.0: a decision model. A Qwen3 backbone, then a head that
//! scores each option the batch marks against the batch's one question --
//! one score per option, not logits.
//!
//! The batch says what the head reads through
//! `llama_batch_ext_set_decision_order`, as for CLEF (`clef.zig`): runs of
//! entries marked as the text of a question or of an option. The head reads
//! the hidden state of the last entry of each run, so those entries must be
//! outputs. A batch with no usable order still runs the head, over one
//! option at output row 0, so that the graph has the same nodes for every
//! batch.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/vllm-decision2.cpp` at v0.7.0
//! (`620d2225c`): `llama_model_vllm_decision2`'s `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph`, `decision2_rows` and
//! `decision2_get_rows`, the graph input `input_decision`, and its `graph`
//! with `build_head`; and from `llama.cpp/src/models/models.h` at v0.7.0,
//! `llama_model_vllm_decision2`'s members.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const batch = @import("../batch.zig");
const qwen3 = @import("qwen3.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const Ubatch = batch.Ubatch;
const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("models/vllm-decision2.cpp", 0, "out of memory");
}

/// Ports `llama_decision_order` (llama-ext.h:115 @620d2225c): what the
/// decision head reads an entry as.
const decision_order_none: i32 = 0;
const decision_order_question_noul: i32 = 1;
const decision_order_question_choice: i32 = 2;
const decision_order_question_score: i32 = 3;
const decision_order_option: i32 = 4;

/// The members `llama_model_vllm_decision2` adds to its base
/// (src/models/models.h:612 @620d2225c), kept on the `Model` as
/// `vllm_decision2`, which only this file reads or writes. The tensors are
/// the model's; the state owns nothing.
pub const State = struct {
    option_norm: ?*Tensor = null,
    option_norm_b: ?*Tensor = null,
    query_norm: ?*Tensor = null,
    query_norm_b: ?*Tensor = null,
    key: ?*Tensor = null,
    query: ?*Tensor = null,
    option_mlp: ?*Tensor = null,
    option_mlp_b: ?*Tensor = null,
    query_mlp: ?*Tensor = null,
    scorer_out: ?*Tensor = null,
};

/// Ports `llama_model_vllm_decision2`'s `load_arch_hparams`
/// (models/vllm-decision2.cpp:7 @620d2225c).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    try qwen3.loadArchHparams(self, ml, exc);

    const hp = &self.hparams;

    // Used by the head.
    hp.f_norm_eps = (try ml.meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    // The output is one score per token: see llama_batch_ext_set_decision_order().
    hp.n_embd_out_impl = 1;
}

/// Ports `llama_model_vllm_decision2`'s `load_arch_tensors`
/// (models/vllm-decision2.cpp:17 @620d2225c).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    try qwen3.loadArchTensors(self, ml, exc);

    const l = self.loadLocals();
    const n_embd = l.n_embd;
    const st = &self.vllm_decision2;

    const name_key = (loader.Tn{ .tensor = .decision_key, .suffix = "weight" }).str();
    const w_key = ml.getWeight(name_key.slice()) orelse {
        return exc.throw(.runtime_error, "the decision head is missing", .{});
    };
    const n_embd_h: i64 = w_key.tensor.ne[1];

    st.option_norm = try self.createTensor(.{ .tensor = .decision_option_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    st.option_norm_b = try self.createTensor(.{ .tensor = .decision_option_norm, .suffix = "bias" }, &.{n_embd}, 0, exc);
    st.query_norm = try self.createTensor(.{ .tensor = .decision_query_norm, .suffix = "weight" }, &.{n_embd}, 0, exc);
    st.query_norm_b = try self.createTensor(.{ .tensor = .decision_query_norm, .suffix = "bias" }, &.{n_embd}, 0, exc);

    st.key = try self.createTensor(.{ .tensor = .decision_key, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.query = try self.createTensor(.{ .tensor = .decision_query, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.option_mlp = try self.createTensor(.{ .tensor = .decision_option_mlp, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.option_mlp_b = try self.createTensor(.{ .tensor = .decision_option_mlp, .suffix = "bias" }, &.{n_embd_h}, 0, exc);
    st.query_mlp = try self.createTensor(.{ .tensor = .decision_query_mlp, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.scorer_out = try self.createTensor(.{ .tensor = .decision_scorer_out, .suffix = "weight" }, &.{ n_embd_h, 1 }, 0, exc);
}

/// Ports `llama_model_vllm_decision2`'s `build_arch_graph`
/// (models/vllm-decision2.cpp:41 @620d2225c).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// The rows the head reads

/// Ports `decision2_rows` (models/vllm-decision2.cpp:46 @620d2225c): the
/// output rows the head reads, the last token of each option span and the
/// last token of the question span.
const Rows = struct {
    options: std.ArrayList(i32) = .empty,
    query: i32 = 0,
    valid: bool = false,

    fn deinit(self: *Rows) void {
        self.options.deinit(allocator);
    }
};

/// Ports `decision2_get_rows` (models/vllm-decision2.cpp:54 @620d2225c): the
/// rows of the batch's decision order (see
/// `llama_batch_ext_set_decision_order`). A batch with no usable order gets
/// one option at row 0.
///
/// Return: the rows, which the caller deinits.
fn getRows(ubatch: *const Ubatch) Rows {
    var res: Rows = .{};

    const n_tokens: i32 = @intCast(ubatch.n_tokens);
    var ok = ubatch.decision_order != null and ubatch.n_seqs_unq == 1;

    // The output row of each token, -1 if the token has no output.
    const rows = allocator.alloc(i32, @intCast(n_tokens)) catch oom();
    defer allocator.free(rows);
    @memset(rows, -1);
    {
        var n: i32 = 0;
        for (rows, 0..) |*r, i| {
            if (ubatch.output[i] != 0) {
                r.* = n;
                n += 1;
            }
        }
    }

    var n_questions: i32 = 0;
    var i: i32 = 0;
    while (ok and i < n_tokens) {
        const order = ubatch.decision_order[@intCast(i)];
        var end = i + 1;
        while (end < n_tokens and ubatch.decision_order[@intCast(end)] == order) end += 1;
        const last = rows[@intCast(end - 1)];
        switch (order) {
            decision_order_none => {},
            decision_order_question_noul, decision_order_question_choice, decision_order_question_score => {
                res.query = last;
                n_questions += 1;
            },
            decision_order_option => res.options.append(allocator, last) catch oom(),
            else => ok = false,
        }
        ok = ok and (order == decision_order_none or last >= 0);
        i = end;
    }

    res.valid = ok and n_questions == 1 and res.options.items.len != 0;
    if (!res.valid) {
        res.options.clearRetainingCapacity();
        res.options.append(allocator, 0) catch oom();
        res.query = 0;
    }
    return res;
}

/// Ports `input_decision` (models/vllm-decision2.cpp:102 @620d2225c), a
/// class of `llama_model_vllm_decision2`: the rows the head reads.
const InpDecision = struct {
    /// I32 `[n_options]`.
    options: ?*Tensor = null,
    /// I32 `[1]`.
    query: ?*Tensor = null,
    /// F32 `[1]`, added to the scores: 0, or NaN on invalid input.
    status: ?*Tensor = null,

    n_options: usize,

    /// Ports `input_decision`'s constructor (models/vllm-decision2.cpp:104
    /// @620d2225c).
    fn init(ubatch: *const Ubatch) InpDecision {
        var rows = getRows(ubatch);
        defer rows.deinit();
        return .{ .n_options = rows.options.items.len };
    }

    /// Ports `input_decision`'s `set_input` (models/vllm-decision2.cpp:106
    /// @620d2225c).
    pub fn setInput(self: *InpDecision, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        var rows = getRows(ubatch);
        defer rows.deinit();
        impl.assert(rows.options.items.len == self.n_options, "models/vllm-decision2.cpp", 108, "rows.options.size() == n_options");

        // The scores are NaN if the batch has a decision order that cannot be used.
        const status_data: f32 = if (rows.valid or ubatch.decision_order == null) 0.0 else std.math.nan(f32);

        c.ggml_backend_tensor_set(self.options, rows.options.items.ptr, 0, c.ggml_nbytes(self.options));
        c.ggml_backend_tensor_set(self.query, &rows.query, 0, c.ggml_nbytes(self.query));
        c.ggml_backend_tensor_set(self.status, &status_data, 0, c.ggml_nbytes(self.status));
    }

    /// Ports `input_decision`'s `can_reuse` (models/vllm-decision2.cpp:118
    /// @620d2225c): `set_input` computes the values again, so only the
    /// shapes must match.
    pub fn canReuse(self: *InpDecision, params: *const graph.Params) bool {
        var rows = getRows(&params.ubatch);
        defer rows.deinit();
        return rows.options.items.len == self.n_options;
    }
};

// -----------------------------------------------------------------------------
// The graph

/// Ports the constructor of `llama_model_vllm_decision2`'s `graph`
/// (models/vllm-decision2.cpp:130 @620d2225c): Qwen3's backbone, then the
/// head, which is always evaluated so that the graph has the same nodes for
/// every batch.
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;

    const inp_decision = allocator.create(InpDecision) catch oom();
    inp_decision.* = InpDecision.init(b.ubatch);
    _ = b.res.addInput(InpDecision, inp_decision);

    var cur = qwen3.buildBackbone(model, b);

    cur = buildHead(model, b, cur, inp_decision);

    // Row i of the output is the score of option i.
    cur = c.ggml_pad(ctx, cur, 0, @intCast(b.n_outputs - cur.ne[1]), 0, 0);
    cur = c.ggml_add(ctx, cur, inp_decision.status);
    b.cb(cur, "result_decision", -1);

    b.res.t_embd = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_vllm_decision2::graph`'s `build_head`
/// (models/vllm-decision2.cpp:151 @620d2225c).
///
/// Parameters:
/// - `hidden`: `[n_embd, n_outputs]`.
///
/// Return: one score per option, `[1, n_options]`.
fn buildHead(model: *const Model, b: *const graph.Builder, hidden: *Tensor, inp: *InpDecision) *Tensor {
    const ctx = b.ctx0;
    const st = &model.vllm_decision2;

    inp.options = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, @intCast(inp.n_options));
    inp.query = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, 1);
    inp.status = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 1);
    for ([_]?*Tensor{ inp.options, inp.query, inp.status }) |t| {
        c.ggml_set_input(t);
    }

    var options: *Tensor = c.ggml_get_rows(ctx, hidden, inp.options); // [n_embd, n_options]
    var query: *Tensor = c.ggml_get_rows(ctx, hidden, inp.query); // [n_embd, 1]

    options = b.buildNorm(options, st.option_norm, st.option_norm_b, .norm, -1);
    query = b.buildNorm(query, st.query_norm, st.query_norm_b, .norm, -1);

    // The scaled dot product of the option key and the query.
    const k: *Tensor = c.ggml_mul_mat(ctx, st.key, options); // [n_embd_h, n_options]
    const q: *Tensor = c.ggml_mul_mat(ctx, st.query, query); // [n_embd_h, 1]
    var bilinear: *Tensor = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, k, q));
    bilinear = c.ggml_scale(ctx, bilinear, 1.0 / impl.sqrtf(@floatFromInt(k.ne[0]))); // [1, n_options]

    var mlp: *Tensor = c.ggml_add(ctx, c.ggml_mul_mat(ctx, st.option_mlp, options), st.option_mlp_b);
    mlp = c.ggml_add(ctx, mlp, c.ggml_mul_mat(ctx, st.query_mlp, query));
    mlp = c.ggml_gelu_erf(ctx, mlp);
    const nonlinear: *Tensor = c.ggml_mul_mat(ctx, st.scorer_out, mlp); // [1, n_options]

    const scores: *Tensor = c.ggml_add(ctx, bilinear, nonlinear);
    b.cb(scores, "decision_scores", -1);
    return scores;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
