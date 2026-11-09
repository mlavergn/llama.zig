//! CLEF: a decision model. A Qwen3.5 backbone run over the whole batch
//! without a memory, then a joint decision head that scores the options
//! the batch marks against their questions -- one score per option, not
//! logits.
//!
//! The batch says what the head reads through
//! `llama_batch_ext_set_decision_order`: runs of entries marked as the text
//! of a question (of one of three types) or of an option, an option
//! belonging to the last question before it. A batch with no usable order
//! still runs the head, over one empty question with one empty option, so
//! that the graph has the same nodes for every batch.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/clef.cpp` at v0.8.0 (`8642541b2`):
//! `llama_model_clef`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, `clef_spans` and `clef_get_spans`, the graph inputs
//! `input_decision` and `llm_graph_input_attn_clef`, and its `graph` with
//! `build_attn_inp_causal`, `build_layer_attn`, `build_layer_attn_linear`,
//! `build_head`, `build_head_norm`, `build_head_attn` and `build_head_ffn`;
//! and from `llama.cpp/src/models/models.h` at v0.8.0, `llama_model_clef`'s
//! members; and from `llama.cpp/src/llama-ext.h` at v0.8.0,
//! `llama_decision_order`.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const batch = @import("../batch.zig");
const arch = @import("../arch.zig");
const qwen35 = @import("qwen35.zig");
const delta_net = @import("delta_net.zig");

const c = impl.c;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const Ubatch = batch.Ubatch;
const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("models/clef.cpp", 0, "out of memory");
}

/// Ports `llama_decision_order` (llama-ext.h:140 @8642541b2): what the
/// decision head reads an entry as.
const decision_order_none: i32 = 0;
const decision_order_question_noul: i32 = 1;
const decision_order_question_choice: i32 = 2;
const decision_order_question_score: i32 = 3;
const decision_order_option: i32 = 4;

/// `LLAMA_MAX_LAYERS` (llama-hparams.h:11 @8642541b2).
const max_layers: u32 = 512;

/// Mirrors `llama_model_clef`'s `attn` (src/models/models.h:2668 @8642541b2):
/// one attention of the head, every projection with a bias.
pub const Attn = struct {
    wq: ?*Tensor = null,
    bq: ?*Tensor = null,
    wk: ?*Tensor = null,
    bk: ?*Tensor = null,
    wv: ?*Tensor = null,
    bv: ?*Tensor = null,
    wo: ?*Tensor = null,
    bo: ?*Tensor = null,
};

/// Mirrors `llama_model_clef`'s `norm` (src/models/models.h:2679 @8642541b2): a
/// LayerNorm with a bias.
pub const Norm = struct {
    w: ?*Tensor = null,
    b: ?*Tensor = null,
};

/// Mirrors `llama_model_clef`'s `head_layer` (src/models/models.h:2685
/// @8642541b2): one block of the head. The first `n_layer_routing` blocks
/// have no self-attention, and only they have `cross_norm_kv`.
pub const HeadLayer = struct {
    self_norm: Norm = .{},
    self_attn: Attn = .{},
    cross_norm: Norm = .{},
    cross_norm_kv: Norm = .{},
    cross_attn: Attn = .{},
    ffn_norm: Norm = .{},
    ffn_up: ?*Tensor = null,
    ffn_up_b: ?*Tensor = null,
    ffn_down: ?*Tensor = null,
    ffn_down_b: ?*Tensor = null,
};

/// The members `llama_model_clef` adds to its base (src/models/models.h:2663
/// @8642541b2), kept on the `Model` as `clef`, which only this file reads
/// or writes.
pub const State = struct {
    n_layer_routing: u32 = 0,
    n_layer_joint: u32 = 0,
    n_head_decision: u32 = 0,

    /// The routing blocks, then the joint blocks; owned.
    head_layers: []HeadLayer = &.{},

    hidden_norm: Norm = .{},
    option_summary_norm: Norm = .{},
    field_norm: Norm = .{},
    option_norm: Norm = .{},

    proj_memory: ?*Tensor = null,
    proj_question: ?*Tensor = null,
    proj_option_question: ?*Tensor = null,
    proj_global: ?*Tensor = null,
    proj_option_context: ?*Tensor = null,
    proj_option_lexical: ?*Tensor = null,
    /// The prior's scale, the joint scale and the residual gate.
    scales: ?*Tensor = null,
    scorer: ?*Tensor = null,
    scorer_b: ?*Tensor = null,
    scorer_out: ?*Tensor = null,
    scorer_out_b: ?*Tensor = null,

    /// Frees `head_layers`.
    ///
    /// Return: nothing.
    pub fn deinit(self: *State) void {
        if (self.head_layers.len != 0) allocator.free(self.head_layers);
        self.* = .{};
    }
};

/// Ports `llama_model_clef`'s `load_arch_hparams` (models/clef.cpp:8 @8642541b2).
pub fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
    try qwen35.loadArchHparams(self, ml, exc);

    const meta = &ml.meta;
    const hp = &self.hparams;
    const st = &self.clef;

    st.n_layer_routing = (try meta.getKey(u32, .decision_routing_block_count, true, exc)).?;
    st.n_layer_joint = (try meta.getKey(u32, .decision_block_count, true, exc)).?;
    st.n_head_decision = (try meta.getKey(u32, .decision_head_count, true, exc)).?;

    if (st.n_head_decision == 0 or st.n_layer_routing > max_layers or st.n_layer_joint > max_layers) {
        return exc.throw(.runtime_error, "invalid size of the decision head", .{});
    }

    // Used by the head.
    hp.f_norm_eps = (try meta.getKey(f32, .attention_layernorm_eps, true, exc)).?;

    // The output is one score per token: see llama_batch_ext_set_decision_order().
    hp.n_embd_out_impl = 1;
}

/// The `load_norm` lambda of `load_arch_tensors`.
fn loadNorm(self: *Model, n: *Norm, ty: arch.Tensor, size: i64, il: i32, exc: *Exception) Thrown!void {
    n.w = try self.createTensor(.{ .tensor = ty, .suffix = "weight", .bid = il }, &.{size}, 0, exc);
    n.b = try self.createTensor(.{ .tensor = ty, .suffix = "bias", .bid = il }, &.{size}, 0, exc);
}

/// The `load_attn` lambda of `load_arch_tensors`.
fn loadAttn(self: *Model, a: *Attn, q: arch.Tensor, k: arch.Tensor, v: arch.Tensor, o: arch.Tensor, n_embd_h: i64, il: i32, exc: *Exception) Thrown!void {
    a.wq = try self.createTensor(.{ .tensor = q, .suffix = "weight", .bid = il }, &.{ n_embd_h, n_embd_h }, 0, exc);
    a.bq = try self.createTensor(.{ .tensor = q, .suffix = "bias", .bid = il }, &.{n_embd_h}, 0, exc);
    a.wk = try self.createTensor(.{ .tensor = k, .suffix = "weight", .bid = il }, &.{ n_embd_h, n_embd_h }, 0, exc);
    a.bk = try self.createTensor(.{ .tensor = k, .suffix = "bias", .bid = il }, &.{n_embd_h}, 0, exc);
    a.wv = try self.createTensor(.{ .tensor = v, .suffix = "weight", .bid = il }, &.{ n_embd_h, n_embd_h }, 0, exc);
    a.bv = try self.createTensor(.{ .tensor = v, .suffix = "bias", .bid = il }, &.{n_embd_h}, 0, exc);
    a.wo = try self.createTensor(.{ .tensor = o, .suffix = "weight", .bid = il }, &.{ n_embd_h, n_embd_h }, 0, exc);
    a.bo = try self.createTensor(.{ .tensor = o, .suffix = "bias", .bid = il }, &.{n_embd_h}, 0, exc);
}

/// Ports `llama_model_clef`'s `load_arch_tensors` (models/clef.cpp:26 @8642541b2).
pub fn loadArchTensors(self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void {
    try qwen35.loadArchTensors(self, ml, exc);

    const l = self.loadLocals();
    const n_embd = l.n_embd;
    const st = &self.clef;

    const name_memory = (loader.Tn{ .tensor = .decision_proj_memory, .suffix = "weight" }).str();
    const name_ffn = (loader.Tn{ .tensor = .dec_ffn_up, .suffix = "weight", .bid = 0 }).str();
    const w_memory = ml.getWeight(name_memory.slice());
    const w_ffn = ml.getWeight(name_ffn.slice());
    if (w_memory == null or w_ffn == null) {
        return exc.throw(.runtime_error, "the decision head is missing", .{});
    }
    const n_embd_h: i64 = w_memory.?.tensor.ne[1];
    const n_ff_h: i64 = w_ffn.?.tensor.ne[1];

    if (@rem(n_embd_h, @as(i64, st.n_head_decision)) != 0) {
        return exc.throw(.runtime_error, "invalid width of the decision head", .{});
    }

    const n_head_layers = st.n_layer_routing + st.n_layer_joint;
    if (st.head_layers.len != 0) allocator.free(st.head_layers);
    st.head_layers = allocator.alloc(HeadLayer, n_head_layers) catch oom();
    @memset(st.head_layers, .{});

    for (st.head_layers, 0..) |*layer, i| {
        const il: i32 = @intCast(i);

        if (il < st.n_layer_routing) {
            try loadNorm(self, &layer.cross_norm_kv, .dec_cross_attn_norm_kv, n_embd_h, il, exc);
        } else {
            try loadNorm(self, &layer.self_norm, .dec_attn_norm, n_embd_h, il, exc);
            try loadAttn(self, &layer.self_attn, .dec_attn_q, .dec_attn_k, .dec_attn_v, .dec_attn_out, n_embd_h, il, exc);
        }

        try loadNorm(self, &layer.cross_norm, .dec_cross_attn_norm, n_embd_h, il, exc);
        try loadAttn(self, &layer.cross_attn, .dec_cross_attn_q, .dec_cross_attn_k, .dec_cross_attn_v, .dec_cross_attn_out, n_embd_h, il, exc);

        try loadNorm(self, &layer.ffn_norm, .dec_ffn_norm, n_embd_h, il, exc);
        layer.ffn_up = try self.createTensor(.{ .tensor = .dec_ffn_up, .suffix = "weight", .bid = il }, &.{ n_embd_h, n_ff_h }, 0, exc);
        layer.ffn_up_b = try self.createTensor(.{ .tensor = .dec_ffn_up, .suffix = "bias", .bid = il }, &.{n_ff_h}, 0, exc);
        layer.ffn_down = try self.createTensor(.{ .tensor = .dec_ffn_down, .suffix = "weight", .bid = il }, &.{ n_ff_h, n_embd_h }, 0, exc);
        layer.ffn_down_b = try self.createTensor(.{ .tensor = .dec_ffn_down, .suffix = "bias", .bid = il }, &.{n_embd_h}, 0, exc);
    }

    try loadNorm(self, &st.hidden_norm, .decision_hidden_norm, n_embd, -1, exc);
    try loadNorm(self, &st.option_summary_norm, .decision_option_summary_norm, n_embd_h, -1, exc);
    try loadNorm(self, &st.field_norm, .decision_field_norm, n_embd_h, -1, exc);
    try loadNorm(self, &st.option_norm, .decision_option_norm, n_embd_h, -1, exc);

    st.proj_memory = try self.createTensor(.{ .tensor = .decision_proj_memory, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.proj_question = try self.createTensor(.{ .tensor = .decision_proj_question, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.proj_option_question = try self.createTensor(.{ .tensor = .decision_proj_option_question, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.proj_global = try self.createTensor(.{ .tensor = .decision_proj_global, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.proj_option_context = try self.createTensor(.{ .tensor = .decision_proj_option_context, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);
    st.proj_option_lexical = try self.createTensor(.{ .tensor = .decision_proj_option_lexical, .suffix = "weight" }, &.{ n_embd, n_embd_h }, 0, exc);

    st.scales = try self.createTensor(.{ .tensor = .decision_scales }, &.{3}, 0, exc);
    self.type_embd = try self.createTensor(.{ .tensor = .token_types, .suffix = "weight" }, &.{ n_embd_h, 3 }, 0, exc);

    st.scorer = try self.createTensor(.{ .tensor = .decision_scorer, .suffix = "weight" }, &.{ 4 * n_embd_h, n_embd_h }, 0, exc);
    st.scorer_b = try self.createTensor(.{ .tensor = .decision_scorer, .suffix = "bias" }, &.{n_embd_h}, 0, exc);
    st.scorer_out = try self.createTensor(.{ .tensor = .decision_scorer_out, .suffix = "weight" }, &.{ n_embd_h, 1 }, 0, exc);
    st.scorer_out_b = try self.createTensor(.{ .tensor = .decision_scorer_out, .suffix = "bias" }, &.{1}, 0, exc);
}

/// Ports `llama_model_clef`'s `build_arch_graph` (models/clef.cpp:101 @8642541b2).
pub fn buildArchGraph(model: *const Model, b: *const graph.Builder, _: *const graph.Params) void {
    buildGraph(model, b);
}

// -----------------------------------------------------------------------------
// The spans the head reads

/// Ports `clef_spans` (models/clef.cpp:106 @8642541b2): the spans the head
/// reads, `[start, end)` in ubatch token indices.
const Spans = struct {
    const Question = struct {
        /// noul, choice or score.
        type: i32,
        start: i32,
        end: i32,
    };
    const Option = struct {
        question: i32,
        start: i32,
        end: i32,
    };

    questions: std.ArrayList(Question) = .empty,
    options: std.ArrayList(Option) = .empty,
    valid: bool = false,

    fn deinit(self: *Spans) void {
        self.questions.deinit(allocator);
        self.options.deinit(allocator);
    }
};

/// Ports `clef_get_spans` (models/clef.cpp:124 @8642541b2): the spans of the
/// batch's decision order (see `llama_batch_ext_set_decision_order`). A
/// batch with no usable order gets one empty question with one empty
/// option.
///
/// Return: the spans, which the caller deinits.
fn getSpans(ubatch: *const Ubatch) Spans {
    var res: Spans = .{};
    var has_option: std.ArrayList(bool) = .empty;
    defer has_option.deinit(allocator);

    // The C++'s TODO: support multiple sequences.
    var ok = ubatch.decision_order != null and ubatch.token != null and ubatch.n_seqs_unq == 1;

    const n_tokens: i32 = @intCast(ubatch.n_tokens);
    var i: i32 = 0;
    while (ok and i < n_tokens) {
        const order = ubatch.decision_order[@intCast(i)];
        var end = i + 1;
        while (end < n_tokens and ubatch.decision_order[@intCast(end)] == order) end += 1;
        switch (order) {
            decision_order_none => {},
            decision_order_question_noul, decision_order_question_choice, decision_order_question_score => {
                res.questions.append(allocator, .{ .type = order - decision_order_question_noul, .start = i, .end = end }) catch oom();
                has_option.append(allocator, false) catch oom();
            },
            decision_order_option => {
                ok = res.questions.items.len != 0;
                if (ok) {
                    res.options.append(allocator, .{ .question = @as(i32, @intCast(res.questions.items.len)) - 1, .start = i, .end = end }) catch oom();
                    has_option.items[has_option.items.len - 1] = true;
                }
            },
            else => ok = false,
        }
        i = end;
    }

    // The head reads the spans' token ids: they cannot be embeddings.
    i = 0;
    while (ok and ubatch.isMixed() and i < n_tokens) : (i += 1) {
        ok = ubatch.type[@intCast(i)] == 0 or ubatch.decision_order[@intCast(i)] == decision_order_none;
    }

    // Each question needs an option.
    res.valid = ok and res.questions.items.len != 0 and std.mem.indexOfScalar(bool, has_option.items, false) == null;
    if (!res.valid) {
        res.questions.clearRetainingCapacity();
        res.options.clearRetainingCapacity();
        res.questions.append(allocator, .{ .type = 0, .start = 0, .end = 0 }) catch oom();
        res.options.append(allocator, .{ .question = 0, .start = 0, .end = 0 }) catch oom();
    }
    return res;
}

/// Ports `input_decision` (models/clef.cpp:177
/// @8642541b2): the pooling matrices and indices computed from the spans.
const InpDecision = struct {
    /// I32 `[n_tokens]`.
    tokens: ?*Tensor = null,
    /// F32 `[n_tokens, n_questions]`: the mean over each question's span.
    pool_q: ?*Tensor = null,
    /// F32 `[n_tokens, n_options]`: the mean over each option's span.
    pool_o: ?*Tensor = null,
    /// I32 `[n_questions]`.
    question_type: ?*Tensor = null,
    /// I32 `[n_options]`.
    option_question: ?*Tensor = null,
    /// F32 `[n_options, n_questions]`: 0 where the option belongs to the
    /// question, -inf elsewhere.
    option_mask: ?*Tensor = null,
    /// F32 `[1]`, added to the scores: 0, or NaN on unusable input.
    status: ?*Tensor = null,

    n_tokens: i64,
    n_questions: usize,
    n_options: usize,

    /// Ports `input_decision`'s constructor (models/clef.cpp:179 @8642541b2).
    fn init(ubatch: *const Ubatch) InpDecision {
        var spans = getSpans(ubatch);
        defer spans.deinit();
        return .{ .n_tokens = ubatch.n_tokens, .n_questions = spans.questions.items.len, .n_options = spans.options.items.len };
    }

    /// Ports `input_decision`'s `set_input` (models/clef.cpp:185 @8642541b2).
    pub fn setInput(self: *InpDecision, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const n_tokens: usize = @intCast(self.n_tokens);
        const nq = self.n_questions;
        const no = self.n_options;

        // A batch of embeddings has no token ids, and no usable spans.
        const no_tokens = allocator.alloc(i32, if (ubatch.token != null) 0 else n_tokens) catch oom();
        defer allocator.free(no_tokens);
        @memset(no_tokens, 0);
        const tok_src: *const anyopaque = if (ubatch.token != null) @ptrCast(ubatch.token) else @ptrCast(no_tokens.ptr);
        c.ggml_backend_tensor_set(self.tokens, tok_src, 0, n_tokens * @sizeOf(i32));

        var spans = getSpans(ubatch);
        defer spans.deinit();
        impl.assert(spans.questions.items.len == nq and spans.options.items.len == no, "models/clef.cpp", 191, "spans.questions.size() == n_questions && spans.options.size() == n_options");

        // The scores are NaN if the batch has a decision order that cannot be used.
        const status_data: f32 = if (spans.valid or ubatch.decision_order == null) 0.0 else std.math.nan(f32);
        c.ggml_backend_tensor_set(self.status, &status_data, 0, c.ggml_nbytes(self.status));

        const pool_q_data = allocator.alloc(f32, n_tokens * nq) catch oom();
        defer allocator.free(pool_q_data);
        @memset(pool_q_data, 0.0);
        const types = allocator.alloc(i32, nq) catch oom();
        defer allocator.free(types);
        for (spans.questions.items, 0..) |q, i| {
            var j = q.start;
            while (j < q.end) : (j += 1) {
                pool_q_data[i * n_tokens + @as(usize, @intCast(j))] = 1.0 / @as(f32, @floatFromInt(q.end - q.start));
            }
            types[i] = q.type;
        }

        const pool_o_data = allocator.alloc(f32, n_tokens * no) catch oom();
        defer allocator.free(pool_o_data);
        @memset(pool_o_data, 0.0);
        const owner = allocator.alloc(i32, no) catch oom();
        defer allocator.free(owner);
        const mask = allocator.alloc(f32, no * nq) catch oom();
        defer allocator.free(mask);
        @memset(mask, -std.math.inf(f32));
        for (spans.options.items, 0..) |o, i| {
            var j = o.start;
            while (j < o.end) : (j += 1) {
                pool_o_data[i * n_tokens + @as(usize, @intCast(j))] = 1.0 / @as(f32, @floatFromInt(o.end - o.start));
            }
            owner[i] = o.question;
            mask[@as(usize, @intCast(o.question)) * no + i] = 0.0;
        }

        c.ggml_backend_tensor_set(self.pool_q, pool_q_data.ptr, 0, c.ggml_nbytes(self.pool_q));
        c.ggml_backend_tensor_set(self.pool_o, pool_o_data.ptr, 0, c.ggml_nbytes(self.pool_o));
        c.ggml_backend_tensor_set(self.question_type, types.ptr, 0, c.ggml_nbytes(self.question_type));
        c.ggml_backend_tensor_set(self.option_question, owner.ptr, 0, c.ggml_nbytes(self.option_question));
        c.ggml_backend_tensor_set(self.option_mask, mask.ptr, 0, c.ggml_nbytes(self.option_mask));
    }

    /// Ports `input_decision`'s `can_reuse` (models/clef.cpp:226 @8642541b2):
    /// `set_input` computes the values again, so only the shapes must match.
    pub fn canReuse(self: *InpDecision, params: *const graph.Params) bool {
        var spans = getSpans(&params.ubatch);
        defer spans.deinit();
        return spans.questions.items.len == self.n_questions and spans.options.items.len == self.n_options;
    }
};

/// Ports `llm_graph_input_attn_clef` (models/clef.cpp:319 @8642541b2): a
/// causal mask by batch order, since the tokens of an image share one
/// position. The C++ derives from `llm_graph_input_attn_no_cache` and
/// overrides `set_input`; here the base is a member the attention reads.
const InpAttnClef = struct {
    base: graph.InpAttnNoCache,

    fn fillMask(comptime T: type, data: [*]T, ubatch: *const Ubatch, n_kv: usize, zero: T, ninf: T) void {
        const n_tokens: usize = ubatch.n_tokens;
        for (0..n_tokens) |j1| {
            for (0..n_kv) |j0| {
                const visible = j0 <= j1 and ubatch.seq_id[j0][0] == ubatch.seq_id[j1][0];
                data[j1 * n_kv + j0] = if (visible) zero else ninf;
            }
        }
    }

    /// Ports `llm_graph_input_attn_clef`'s `set_input` (models/clef.cpp:323
    /// @8642541b2): the mask is `n_kv` wide, padded past the batch.
    pub fn setInput(self: *InpAttnClef, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const m = self.base.self_kq_mask.?;
        const n_kv: usize = @intCast(m.ne[0]);
        impl.assert(c.ggml_backend_buffer_is_host(m.buffer), "models/clef.cpp", 336, "ggml_backend_buffer_is_host(self_kq_mask->buffer)");
        if (m.type == c.GGML_TYPE_F16) {
            fillMask(u16, @ptrCast(@alignCast(m.data)), ubatch, n_kv, c.ggml_fp32_to_fp16(0.0), c.ggml_fp32_to_fp16(-std.math.inf(f32)));
        } else {
            fillMask(f32, @ptrCast(@alignCast(m.data)), ubatch, n_kv, 0.0, -std.math.inf(f32));
        }
    }
};

// -----------------------------------------------------------------------------
// The graph

/// Ports `llama_model_clef::graph`'s `build_attn_inp_causal`
/// (models/clef.cpp:346 @8642541b2): `build_attn_inp_no_cache` with a
/// causal mask even when the batch takes the encoder path.
fn buildAttnInpCausal(b: *const graph.Builder) *graph.InpAttnNoCache {
    const inp = allocator.create(InpAttnClef) catch oom();
    inp.* = .{ .base = .{ .hparams = b.hparams, .cparams = b.cparams.*, .debug = 0 } };

    const type_mask: c.enum_ggml_type = if (b.cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

    // `GGML_PAD(n_tokens, 256)`.
    const n_kv: i64 = @divTrunc(b.n_tokens + 255, 256) * 256;
    inp.base.self_kq_mask = c.ggml_new_tensor_4d(b.ctx0, type_mask, n_kv, b.n_tokens, 1, 1);
    c.ggml_set_input(inp.base.self_kq_mask);
    b.cb(inp.base.self_kq_mask.?, "self_kq_mask", -1);

    inp.base.self_kq_mask_cnv = inp.base.self_kq_mask;

    return &b.res.addInput(InpAttnClef, inp).base;
}

/// Ports the constructor of `llama_model_clef`'s `graph`
/// (models/clef.cpp:246 @8642541b2): the backbone is
/// `llama_model_qwen35::graph`'s without the memory module, and the head is
/// always evaluated.
fn buildGraph(model: *const Model, b: *const graph.Builder) void {
    const ctx = b.ctx0;
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);

    impl.assert(n_embd_head == hp.nEmbdHeadK(0), "models/clef.cpp", 250, "n_embd_head == hparams.n_embd_head_k()");

    var sections: [4]c_int = undefined;
    @memcpy(&sections, hp.rope_sections[0..4]);

    var inpL = b.buildInpEmbd(model.tok_embd.?);

    const inp_pos = b.buildInpPos();

    const inp_attn = buildAttnInpCausal(b);

    const inp_decision = allocator.create(InpDecision) catch oom();
    inp_decision.* = InpDecision.init(b.ubatch);
    _ = b.res.addInput(InpDecision, inp_decision);

    var il: c_int = 0;
    while (il < b.n_layer) : (il += 1) {
        const layer = &model.layers[@intCast(il)];
        const inpSA = inpL;

        var cur = b.buildNorm(inpL, layer.attn_norm, null, .rms, il);
        b.cb(cur, "attn_norm", il);

        if (hp.isRecr(@intCast(il))) {
            cur = buildLayerAttnLinear(model, b, cur, il);
        } else {
            cur = buildLayerAttn(model, b, inp_attn, cur, inp_pos, &sections, il);
        }

        cur = c.ggml_add(ctx, cur, inpSA);
        b.cb(cur, "attn_residual", il);

        const ffn_residual = cur;

        cur = b.buildNorm(cur, layer.attn_post_norm, null, .rms, il);
        b.cb(cur, "attn_post_norm", il);

        cur = b.buildFfn(cur, layer.ffn_up, null, layer.ffn_up_s, layer.ffn_gate, null, layer.ffn_gate_s, layer.ffn_down, null, layer.ffn_down_s, null, .silu, .par, il);
        b.cb(cur, "ffn_out", il);

        cur = c.ggml_add(ctx, cur, ffn_residual);
        b.cb(cur, "l_out", il);

        inpL = cur;
    }

    var cur = b.buildNorm(inpL, model.output_norm, null, .rms, -1);
    b.cb(cur, "result_norm", -1);

    // The head is always evaluated, so that the graph has the same nodes for every batch.
    cur = buildHead(model, b, cur, inp_decision);

    // Row i of the output is the score of option i.
    cur = c.ggml_pad(ctx, cur, 0, @intCast(b.n_tokens - cur.ne[1]), 0, 0);
    cur = c.ggml_add(ctx, cur, inp_decision.status);
    b.cb(cur, "result_decision", -1);

    b.res.t_embd = cur;

    c.ggml_build_forward_expand(b.gf, cur);
}

/// Ports `llama_model_clef::graph`'s `build_layer_attn` (models/clef.cpp:360
/// @8642541b2): Qwen3.5's gated full attention, over the batch alone.
fn buildLayerAttn(model: *const Model, b: *const graph.Builder, inp: *graph.InpAttnNoCache, cur_: *Tensor, inp_pos: *Tensor, sections: *[4]c_int, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    const n_embd_head: i64 = hp.nEmbdHeadV(0);
    const n_head = b.n_head;
    const n_head_kv = b.n_head_kv;
    const n_tokens = b.n_tokens;

    // The Q projection outputs the query and the gate.
    const qkv = b.buildQkvHeads(layer, cur_, n_embd_head * 2, n_head, n_embd_head, n_head_kv, n_embd_head, n_head_kv, il, false);
    const Qcur_full = qkv.q;
    var Kcur = qkv.k;
    var Vcur = qkv.v;

    const es = c.ggml_element_size(Qcur_full);
    const head_sz: usize = @intCast(n_embd_head);
    var Qcur = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), 0);

    Qcur = b.buildNorm(Qcur, layer.attn_q_norm, null, .rms, il);
    b.cb(Qcur, "Qcur_normed", il);

    Kcur = c.ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = b.buildNorm(Kcur, layer.attn_k_norm, null, .rms, il);
    b.cb(Kcur, "Kcur_normed", il);

    var gate = c.ggml_view_3d(ctx, Qcur_full, n_embd_head, n_head, n_tokens, es * head_sz * 2, es * head_sz * 2 * @as(usize, @intCast(n_head)), es * head_sz);
    gate = c.ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);

    Vcur = c.ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = c.ggml_rope_multi(ctx, Qcur, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);
    Kcur = c.ggml_rope_multi(ctx, Kcur, inp_pos, null, @intCast(b.n_rot), sections, b.rope_type, b.n_ctx_orig, b.freq_base, b.freq_scale, b.ext_factor, b.attn_factor, b.beta_fast, b.beta_slow);

    const kq_scale: f32 = if (hp.f_attention_scale == 0.0) 1.0 / impl.sqrtf(@floatFromInt(n_embd_head)) else hp.f_attention_scale;

    var cur = b.buildAttn(inp, null, null, null, Qcur, Kcur, Vcur, null, null, null, kq_scale, il);
    b.cb(cur, "attn_pregate", il);

    cur = c.ggml_mul(ctx, cur, c.ggml_sigmoid(ctx, gate));

    cur = b.buildLoraMm(layer.wo.?, cur, layer.wo_s);
    b.cb(cur, "attn_output", il);

    return cur;
}

/// Ports `llama_model_clef::graph`'s `build_layer_attn_linear`
/// (models/clef.cpp:422 @8642541b2): the gated delta net over the whole
/// batch, its conv and recurrent states starting from zero and not kept.
fn buildLayerAttnLinear(model: *const Model, b: *const graph.Builder, cur_: *Tensor, il: c_int) *Tensor {
    const ctx = b.ctx0;
    const layer = &model.layers[@intCast(il)];
    const hp = b.hparams;
    const n_tokens = b.n_tokens;

    const d_inner: i64 = hp.ssm_d_inner;
    const head_k_dim: i64 = hp.ssm_d_state;
    const num_k_heads: i64 = hp.ssm_n_group;
    const num_v_heads: i64 = hp.ssm_dt_rank;
    const head_v_dim: i64 = @divTrunc(d_inner, num_v_heads);
    const n_seqs: i64 = 1;

    var qkv_mixed = b.buildLoraMm(layer.wqkv.?, cur_, layer.wqkv_s);
    qkv_mixed = c.ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed.ne[0], n_tokens, n_seqs);

    const z = b.buildLoraMm(layer.wqkv_gate.?, cur_, layer.wqkv_gate_s);

    var beta = b.buildLoraMm(layer.ssm_beta.?, cur_, layer.ssm_beta_s);
    beta = c.ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_tokens, n_seqs);
    beta = c.ggml_sigmoid(ctx, beta);

    var alpha = b.buildLoraMm(layer.ssm_alpha.?, cur_, layer.ssm_alpha_s);
    alpha = c.ggml_reshape_3d(ctx, alpha, num_v_heads, n_tokens, n_seqs);

    var gate = c.ggml_softplus(ctx, c.ggml_add(ctx, alpha, layer.ssm_dt));
    gate = c.ggml_mul(ctx, gate, layer.ssm_a);
    gate = c.ggml_reshape_4d(ctx, gate, 1, num_v_heads, n_tokens, n_seqs);

    const conv_kernel_size = layer.ssm_conv1d.?.ne[0];
    const conv_channels = d_inner + 2 * num_k_heads * head_k_dim;

    const conv_states = c.ggml_fill(ctx, c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, conv_kernel_size - 1, conv_channels, n_seqs), 0.0);
    const conv_input = c.ggml_concat(ctx, conv_states, c.ggml_transpose(ctx, qkv_mixed), 0);

    const conv_out = c.ggml_silu(ctx, c.ggml_ssm_conv(ctx, conv_input, layer.ssm_conv1d));
    b.cb(conv_out, "conv_output_silu", il);

    const qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const ty = conv_out.*.type;
    const nb1_qkv = c.ggml_row_size(ty, qkv_dim);
    const nb2_qkv = nb1_qkv * @as(usize, @intCast(n_tokens));

    var q_conv = c.ggml_view_4d(ctx, conv_out, head_k_dim, num_k_heads, n_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb2_qkv, 0);
    var k_conv = c.ggml_view_4d(ctx, conv_out, head_k_dim, num_k_heads, n_tokens, n_seqs, c.ggml_row_size(ty, head_k_dim), nb1_qkv, nb2_qkv, @as(usize, @intCast(head_k_dim * num_k_heads)) * c.ggml_element_size(conv_out));
    const v_conv = c.ggml_view_4d(ctx, conv_out, head_v_dim, num_v_heads, n_tokens, n_seqs, c.ggml_row_size(ty, head_v_dim), nb1_qkv, nb2_qkv, c.ggml_row_size(ty, 2 * head_k_dim * num_k_heads));

    q_conv = delta_net.buildGdnL2Norm(ctx, q_conv, hp.f_norm_rms_eps);
    k_conv = delta_net.buildGdnL2Norm(ctx, k_conv, hp.f_norm_rms_eps);

    // An explicit repeat only without the fused GDN.
    if (num_k_heads != num_v_heads and (!b.cparams.fused_gdn_ar or !b.cparams.fused_gdn_ch)) {
        impl.assert(@rem(num_v_heads, num_k_heads) == 0, "models/clef.cpp", 477, "num_v_heads % num_k_heads == 0");
        q_conv = c.ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
        k_conv = c.ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
    }

    const state = c.ggml_fill(ctx, c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, head_v_dim, head_v_dim, num_v_heads, n_seqs), 0.0);

    var output = delta_net.buildDeltaNet(b, q_conv, k_conv, v_conv, gate, beta, state, il).out;
    b.cb(output, "attn_output", il);

    // The gated normalization.
    const z_4d = c.ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_tokens, n_seqs);
    output = b.buildNorm(output, layer.ssm_norm, null, .rms, il);
    output = c.ggml_mul(ctx, output, c.ggml_silu(ctx, z_4d));

    output = c.ggml_reshape_3d(ctx, output, head_v_dim * num_v_heads, n_tokens, n_seqs);

    const cur = b.buildLoraMm(layer.ssm_out.?, output, layer.ssm_out_s);
    b.cb(cur, "linear_attn_out", il);

    return c.ggml_reshape_2d(ctx, cur, b.n_embd, n_tokens);
}

/// Ports `llama_model_clef::graph`'s `build_head_norm` (models/clef.cpp:500
/// @8642541b2).
fn buildHeadNorm(b: *const graph.Builder, cur: *Tensor, n: *const Norm) *Tensor {
    return b.buildNorm(cur, n.w, n.b, .norm, -1);
}

/// Ports `llama_model_clef::graph`'s `build_head_attn` (models/clef.cpp:505
/// @8642541b2): multi-head attention of `q` `[n_embd_h, n_q]` over `kv`
/// `[n_embd_h, n_kv]`, without a mask.
fn buildHeadAttn(model: *const Model, b: *const graph.Builder, q: *Tensor, kv: *Tensor, a: *const Attn) *Tensor {
    const ctx = b.ctx0;
    const n_embd_h = q.ne[0];
    const n_head_h: i64 = model.clef.n_head_decision;
    const d_head = @divTrunc(n_embd_h, n_head_h);
    const n_q = q.ne[1];
    const n_kv = kv.ne[1];

    var Qcur = c.ggml_add(ctx, c.ggml_mul_mat(ctx, a.wq, q), a.bq);
    var Kcur = c.ggml_add(ctx, c.ggml_mul_mat(ctx, a.wk, kv), a.bk);
    var Vcur = c.ggml_add(ctx, c.ggml_mul_mat(ctx, a.wv, kv), a.bv);

    Qcur = c.ggml_permute(ctx, c.ggml_reshape_3d(ctx, Qcur, d_head, n_head_h, n_q), 0, 2, 1, 3); // [d_head, n_q,  n_head]
    Kcur = c.ggml_permute(ctx, c.ggml_reshape_3d(ctx, Kcur, d_head, n_head_h, n_kv), 0, 2, 1, 3); // [d_head, n_kv, n_head]
    Vcur = c.ggml_permute(ctx, c.ggml_reshape_3d(ctx, Vcur, d_head, n_head_h, n_kv), 1, 2, 0, 3); // [n_kv, d_head, n_head]
    Vcur = c.ggml_cont(ctx, Vcur);

    var kq = c.ggml_mul_mat(ctx, Kcur, Qcur); // [n_kv, n_q, n_head]
    kq = c.ggml_soft_max_ext(ctx, kq, null, 1.0 / impl.sqrtf(@floatFromInt(d_head)), 0.0);

    var cur = c.ggml_mul_mat(ctx, Vcur, kq); // [d_head, n_q, n_head]
    cur = c.ggml_cont_2d(ctx, c.ggml_permute(ctx, cur, 0, 2, 1, 3), n_embd_h, n_q);

    return c.ggml_add(ctx, c.ggml_mul_mat(ctx, a.wo, cur), a.bo);
}

/// Ports `llama_model_clef::graph`'s `build_head_ffn` (models/clef.cpp:530
/// @8642541b2).
fn buildHeadFfn(b: *const graph.Builder, cur_: *Tensor, layer: *const HeadLayer) *Tensor {
    const ctx = b.ctx0;
    var cur = buildHeadNorm(b, cur_, &layer.ffn_norm);
    cur = c.ggml_add(ctx, c.ggml_mul_mat(ctx, layer.ffn_up, cur), layer.ffn_up_b);
    cur = c.ggml_gelu_erf(ctx, cur);
    return c.ggml_add(ctx, c.ggml_mul_mat(ctx, layer.ffn_down, cur), layer.ffn_down_b);
}

/// Ports `llama_model_clef::graph`'s `build_head` (models/clef.cpp:538
/// @8642541b2), `JointSchemaHead` in the model repository's
/// `joint_schema_model.py`: `hidden` `[n_embd, n_tokens]` in, one score
/// per option `[1, n_options]` out.
fn buildHead(model: *const Model, b: *const graph.Builder, hidden_: *Tensor, inp: *InpDecision) *Tensor {
    const ctx = b.ctx0;
    const st = &model.clef;
    const n_tokens = b.n_tokens;
    const n_embd = b.n_embd;
    const n_questions: i64 = @intCast(inp.n_questions);
    const n_options: i64 = @intCast(inp.n_options);

    inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_tokens);
    inp.pool_q = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_tokens, n_questions);
    inp.pool_o = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_tokens, n_options);
    inp.question_type = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_questions);
    inp.option_question = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_options);
    inp.option_mask = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_options, n_questions);
    inp.status = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 1);
    for ([_]?*Tensor{ inp.tokens, inp.pool_q, inp.pool_o, inp.question_type, inp.option_question, inp.option_mask, inp.status }) |t| {
        c.ggml_set_input(t);
    }

    const hidden = buildHeadNorm(b, hidden_, &st.hidden_norm);

    const memory = c.ggml_mul_mat(ctx, st.proj_memory, hidden); // [n_embd_h, n_tokens]

    const n_embd_h = memory.*.ne[0];

    // The hidden state of the last token.
    const global = c.ggml_view_2d(ctx, hidden, n_embd, 1, hidden.nb[1], @as(usize, @intCast(n_tokens - 1)) * hidden.nb[1]);

    // The mean of the hidden states over each span.
    const hidden_t = c.ggml_cont(ctx, c.ggml_transpose(ctx, hidden));
    const question = c.ggml_mul_mat(ctx, hidden_t, inp.pool_q); // [n_embd, n_questions]
    const opt_ctx = c.ggml_mul_mat(ctx, hidden_t, inp.pool_o); // [n_embd, n_options]

    // The mean of the output embeddings of each option's tokens.
    var lexical = c.ggml_get_rows(ctx, model.output, inp.tokens);
    lexical = c.ggml_mul_mat(ctx, c.ggml_cont(ctx, c.ggml_transpose(ctx, lexical)), inp.pool_o); // [n_embd, n_options]

    // One query per option.
    var options = c.ggml_mul_mat(ctx, st.proj_option_context, opt_ctx);
    options = c.ggml_add(ctx, options, c.ggml_mul_mat(ctx, st.proj_option_lexical, lexical));
    options = c.ggml_add(ctx, options, c.ggml_get_rows(ctx, c.ggml_mul_mat(ctx, st.proj_option_question, question), inp.option_question));

    // The options read the prompt.
    var il: u32 = 0;
    while (il < st.n_layer_routing) : (il += 1) {
        const layer = &st.head_layers[il];

        const cur = buildHeadAttn(model, b, buildHeadNorm(b, options, &layer.cross_norm), buildHeadNorm(b, memory, &layer.cross_norm_kv), &layer.cross_attn);
        options = c.ggml_add(ctx, options, cur);
        options = c.ggml_add(ctx, options, buildHeadFfn(b, options, layer));
    }
    b.cb(options, "decision_options", -1);

    // One vector per question: its text, a summary of its options, the end
    // of the prompt and its type.
    var fields = c.ggml_mul_mat(ctx, st.proj_question, question); // [n_embd_h, n_questions]
    {
        // Each question weights its own options.
        var weights = c.ggml_mul_mat(ctx, options, fields); // [n_options, n_questions]
        weights = c.ggml_scale(ctx, weights, 1.0 / impl.sqrtf(@floatFromInt(n_embd_h)));
        weights = c.ggml_soft_max(ctx, c.ggml_add(ctx, weights, inp.option_mask));

        const summary = c.ggml_mul_mat(ctx, c.ggml_cont(ctx, c.ggml_transpose(ctx, options)), weights);

        fields = c.ggml_add(ctx, fields, buildHeadNorm(b, summary, &st.option_summary_norm));
        fields = c.ggml_add(ctx, fields, c.ggml_mul_mat(ctx, st.proj_global, global));
        fields = c.ggml_add(ctx, fields, c.ggml_get_rows(ctx, model.type_embd, inp.question_type));
    }

    // The questions read each other and the prompt.
    il = st.n_layer_routing;
    while (il < st.head_layers.len) : (il += 1) {
        const layer = &st.head_layers[il];

        var cur = buildHeadNorm(b, fields, &layer.self_norm);
        fields = c.ggml_add(ctx, fields, buildHeadAttn(model, b, cur, cur, &layer.self_attn));

        cur = buildHeadNorm(b, fields, &layer.cross_norm);
        fields = c.ggml_add(ctx, fields, buildHeadAttn(model, b, cur, memory, &layer.cross_attn));

        fields = c.ggml_add(ctx, fields, buildHeadFfn(b, fields, layer));
    }
    fields = buildHeadNorm(b, fields, &st.field_norm);
    b.cb(fields, "decision_fields", -1);

    const eps: f32 = 1e-12;

    // The prior: each option's output embeddings against its question and
    // the end of the prompt.
    var anchor = c.ggml_l2_norm(ctx, c.ggml_add(ctx, question, global), eps);
    anchor = c.ggml_get_rows(ctx, anchor, inp.option_question);
    const prior = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, c.ggml_l2_norm(ctx, lexical, eps), anchor)); // [1, n_options]

    // The joint score: each option against its question's vector.
    options = buildHeadNorm(b, options, &st.option_norm);
    const field = c.ggml_get_rows(ctx, fields, inp.option_question); // [n_embd_h, n_options]

    const cosine = c.ggml_sum_rows(ctx, c.ggml_mul(ctx, c.ggml_l2_norm(ctx, field, eps), c.ggml_l2_norm(ctx, options, eps)));

    var features = c.ggml_concat(ctx, field, options, 0);
    features = c.ggml_concat(ctx, features, c.ggml_mul(ctx, field, options), 0);
    features = c.ggml_concat(ctx, features, c.ggml_abs(ctx, c.ggml_sub(ctx, field, options)), 0);

    var residual = c.ggml_add(ctx, c.ggml_mul_mat(ctx, st.scorer, features), st.scorer_b);
    residual = c.ggml_gelu_erf(ctx, residual);
    residual = c.ggml_add(ctx, c.ggml_mul_mat(ctx, st.scorer_out, residual), st.scorer_out_b); // [1, n_options]

    const joint = c.ggml_add(ctx, c.ggml_mul(ctx, cosine, scaleAt(b, st, 1)), residual);

    return c.ggml_add(ctx, c.ggml_mul(ctx, prior, scaleAt(b, st, 0)), c.ggml_mul(ctx, joint, scaleAt(b, st, 2)));
}

/// The `scale` lambda of `build_head`: element `i` of `scales`.
fn scaleAt(b: *const graph.Builder, st: *const State, i: usize) *Tensor {
    return c.ggml_view_1d(b.ctx0, st.scales, 1, i * c.ggml_element_size(st.scales));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a batch without a decision order falls back to one empty question and option" {
    var ub: Ubatch = .{};
    var spans = getSpans(&ub);
    defer spans.deinit();
    try std.testing.expect(!spans.valid);
    try std.testing.expectEqual(@as(usize, 1), spans.questions.items.len);
    try std.testing.expectEqual(@as(usize, 1), spans.options.items.len);
}

test "a question with its options makes valid spans" {
    var tok = [_]i32{ 5, 6, 7, 8, 9, 10, 11 };
    var order = [_]i32{ 0, 2, 2, 0, 4, 0, 4 };
    var ub: Ubatch = .{ .n_tokens = tok.len, .n_seqs_unq = 1, .token = &tok, .decision_order = &order };
    var spans = getSpans(&ub);
    defer spans.deinit();
    try std.testing.expect(spans.valid);
    try std.testing.expectEqual(@as(usize, 1), spans.questions.items.len);
    try std.testing.expectEqual(@as(i32, 1), spans.questions.items[0].type);
    try std.testing.expectEqual(@as(i32, 1), spans.questions.items[0].start);
    try std.testing.expectEqual(@as(i32, 3), spans.questions.items[0].end);
    try std.testing.expectEqual(@as(usize, 2), spans.options.items.len);
    try std.testing.expectEqual(@as(i32, 6), spans.options.items[1].start);
}

test "an option before any question, or a question without one, is unusable" {
    var tok = [_]i32{ 1, 2, 3 };
    var order1 = [_]i32{ 4, 0, 1 };
    var ub: Ubatch = .{ .n_tokens = tok.len, .n_seqs_unq = 1, .token = &tok, .decision_order = &order1 };
    var spans = getSpans(&ub);
    try std.testing.expect(!spans.valid);
    spans.deinit();
    var order2 = [_]i32{ 1, 0, 3 };
    ub.decision_order = &order2;
    spans = getSpans(&ub);
    defer spans.deinit();
    try std.testing.expect(!spans.valid);
}
