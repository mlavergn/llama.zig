//! Compares `src/llama/context_opt.zig` and `src/llama/api/opt.zig` with
//! llama.cpp's training entry points in `llama-context.cpp`, in one process.
//!
//! **Not a port.** The harness behind `scripts/opt-llama-diff`.
//!
//! # The model
//!
//! Training needs F32 weights, and Qwen3.5 is the only architecture the
//! port builds graphs for, so the harness writes one: a four-layer Qwen3.5
//! (three Gated Delta Net layers, one full-attention layer) with a 64-wide
//! embedding, every tensor F32 and filled from a fixed seed, and the 2B
//! model's metadata and vocabulary copied from the GGUF given on the command
//! line with the shapes edited. Untied: `output.weight` is its own tensor,
//! because a tied output is `token_embd.weight`, which `llama_set_param`
//! never trains.
//!
//! # What is compared
//!
//! The C++ and the port each load the file, make a context and train it,
//! the same script through `llama.h` and through the port's `api`
//! functions of the same names. Each side writes a transcript:
//!
//! - every tensor the parameter filter is offered, in order — the order of
//!   `llama_layer`'s members, which the port has to reproduce;
//! - every weight's `PARAM` flag after `llama_opt_init`, and `n_ctx_train`;
//! - **every node of every training graph**, through `cb_eval`: name, op,
//!   shape, sources and a hash of its data — the forward pass, the
//!   backward pass, the optimizer steps, whose outputs are the updated
//!   weights;
//! - every epoch callback's arguments and the result it is handed (loss,
//!   uncertainty, accuracy and predictions, on bits);
//! - after every epoch, a hash of every weight's data;
//! - everything logged while training.
//!
//! The two transcripts must be identical. `ggml-opt` itself is the same
//! object on both sides — the one in `libggml.a` — so only libllama differs.
//!
//! # What cannot run
//!
//! With every weight a parameter, `ggml_build_backward_expand` reaches the
//! Gated Delta Net layers and aborts on an op with no backward pass, on
//! both sides. That is a ggml limitation, not a port difference: the script
//! runs it as a separate process per side (`--abort`) and compares what
//! each printed. The in-process cases train what can be trained.
//!
//! Usage: opt_llama_diff <base-qwen35.gguf> <tmp-dir> [--abort <case> <ref|port>]

const std = @import("std");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;
const M = harness_root.model;
const A = harness_root.api;

const gpa = std.heap.c_allocator;

extern fn _ZNK11llama_model10get_tensorEPKc(model: *const c.struct_llama_model, name: [*:0]const u8) ?*c.ggml_tensor;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

// -----------------------------------------------------------------------------
// The transcript

var rec: ?*std.ArrayList(u8) = null;

fn note(comptime fmt: []const u8, args: anytype) void {
    const r = rec orelse return;
    r.print(gpa, fmt ++ "\n", args) catch unreachable;
}

var log_on = false;

fn logCb(level: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    _ = level;
    if (!log_on) return;
    const r = rec orelse return;
    const s = std.mem.span(@as([*:0]const u8, @ptrCast(text)));
    // Metal compiles a pipeline the first time anything in the process
    // uses it, which is whichever side ran first.
    if (std.mem.startsWith(u8, s, "ggml_metal_library_compile_pipeline:")) return;
    r.appendSlice(gpa, s) catch unreachable;
}

fn hashBytes(p: [*]const u8, n: usize) u64 {
    return std.hash.Wyhash.hash(0, p[0..n]);
}

var scratch: std.ArrayList(u8) = .empty;

fn tensorHash(t: *const c.ggml_tensor) u64 {
    const n = c.ggml_nbytes(t);
    scratch.resize(gpa, n) catch unreachable;
    c.ggml_backend_tensor_get(t, scratch.items.ptr, 0, n);
    return hashBytes(scratch.items.ptr, n);
}

var n_nodes_seen: usize = 0;

fn nodeCb(t: [*c]c.ggml_tensor, ask: bool, _: ?*anyopaque) callconv(.c) bool {
    if (ask) return true;
    const node: *c.ggml_tensor = t;
    n_nodes_seen += 1;
    const r = rec orelse return true;
    r.print(gpa, "node {s} {s} {s} [{d},{d},{d},{d}] flags {x} src", .{
        std.mem.sliceTo(&node.name, 0), c.ggml_op_desc(node), c.ggml_type_name(node.type),
        node.ne[0],                     node.ne[1],           node.ne[2],
        node.ne[3],                     node.flags,
    }) catch unreachable;
    for (node.src) |s| {
        if (s == null) continue;
        const src: *c.ggml_tensor = s;
        r.print(gpa, " {s}:{s}:{x}", .{ std.mem.sliceTo(&src.name, 0), c.ggml_op_desc(src), src.flags }) catch unreachable;
    }
    r.print(gpa, " h {x}\n", .{tensorHash(node)}) catch unreachable;
    return true;
}

// -----------------------------------------------------------------------------
// The model

const n_embd: i64 = 64;
const n_ff: i64 = 128;
const n_layer: u32 = 4;
const n_head: i64 = 2;
const n_head_kv: i64 = 1;
const head_dim: i64 = 32;
const ssm_d_state: i64 = 16; // head_k_dim
const ssm_n_group: i64 = 2; // num_k_heads
const ssm_dt_rank: i64 = 2; // num_v_heads
const ssm_d_inner: i64 = 32; // num_v_heads * head_v_dim
const ssm_d_conv: i64 = 4;
const conv_dim: i64 = 2 * ssm_n_group * ssm_d_state + ssm_d_inner;

const TensorSpec = struct { name: []const u8, ne: [2]i64, kind: enum { norm, matrix, ssm_a, dt, conv, embd } };

/// Writes the four-layer F32 Qwen3.5 to `path`, its metadata and vocabulary
/// taken from `base`.
fn writeModel(base: [:0]const u8, path: [:0]const u8) void {
    const src = c.gguf_init_from_file(base.ptr, .{ .no_alloc = true, .ctx = null }) orelse {
        std.debug.print("FAIL: cannot read {s}\n", .{base});
        std.process.exit(2);
    };
    defer c.gguf_free(src);
    const n_vocab: i64 = @intCast(c.gguf_get_arr_n(src, c.gguf_find_key(src, "tokenizer.ggml.tokens")));

    const g = c.gguf_init_empty();
    defer c.gguf_free(g);
    c.gguf_set_kv(g, src);
    for ([_][*:0]const u8{ "quantize.imatrix.file", "quantize.imatrix.dataset", "quantize.imatrix.entries_count", "quantize.imatrix.chunks_count" }) |k| _ = c.gguf_remove_key(g, k);
    c.gguf_set_val_str(g, "general.name", "opt-diff-tiny");
    c.gguf_set_val_u32(g, "general.file_type", c.LLAMA_FTYPE_ALL_F32);
    c.gguf_set_val_u32(g, "qwen35.block_count", n_layer);
    c.gguf_set_val_u32(g, "qwen35.context_length", 256);
    c.gguf_set_val_u32(g, "qwen35.embedding_length", n_embd);
    c.gguf_set_val_u32(g, "qwen35.feed_forward_length", n_ff);
    c.gguf_set_val_u32(g, "qwen35.attention.head_count", n_head);
    c.gguf_set_val_u32(g, "qwen35.attention.head_count_kv", n_head_kv);
    c.gguf_set_val_u32(g, "qwen35.attention.key_length", head_dim);
    c.gguf_set_val_u32(g, "qwen35.attention.value_length", head_dim);
    c.gguf_set_val_u32(g, "qwen35.rope.dimension_count", 16);
    const sections = [_]i32{ 3, 3, 2, 0 };
    c.gguf_set_arr_data(g, "qwen35.rope.dimension_sections", c.GGUF_TYPE_INT32, &sections, sections.len);
    c.gguf_set_val_u32(g, "qwen35.ssm.conv_kernel", ssm_d_conv);
    c.gguf_set_val_u32(g, "qwen35.ssm.state_size", ssm_d_state);
    c.gguf_set_val_u32(g, "qwen35.ssm.group_count", ssm_n_group);
    c.gguf_set_val_u32(g, "qwen35.ssm.time_step_rank", ssm_dt_rank);
    c.gguf_set_val_u32(g, "qwen35.ssm.inner_size", ssm_d_inner);
    c.gguf_set_val_u32(g, "qwen35.full_attention_interval", 4);

    var specs: std.ArrayList(TensorSpec) = .empty;
    defer specs.deinit(gpa);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    const add = struct {
        fn f(list: *std.ArrayList(TensorSpec), nl: *std.ArrayList([]u8), comptime fmt: []const u8, args: anytype, ne: [2]i64, kind: @FieldType(TensorSpec, "kind")) void {
            const n = std.fmt.allocPrint(gpa, fmt, args) catch unreachable;
            nl.append(gpa, n) catch unreachable;
            list.append(gpa, .{ .name = n, .ne = ne, .kind = kind }) catch unreachable;
        }
    }.f;
    add(&specs, &names, "token_embd.weight", .{}, .{ n_embd, n_vocab }, .embd);
    add(&specs, &names, "output_norm.weight", .{}, .{ n_embd, 1 }, .norm);
    add(&specs, &names, "output.weight", .{}, .{ n_embd, n_vocab }, .matrix);
    for (0..n_layer) |il| {
        add(&specs, &names, "blk.{d}.attn_norm.weight", .{il}, .{ n_embd, 1 }, .norm);
        add(&specs, &names, "blk.{d}.post_attention_norm.weight", .{il}, .{ n_embd, 1 }, .norm);
        add(&specs, &names, "blk.{d}.ffn_gate.weight", .{il}, .{ n_embd, n_ff }, .matrix);
        add(&specs, &names, "blk.{d}.ffn_up.weight", .{il}, .{ n_embd, n_ff }, .matrix);
        add(&specs, &names, "blk.{d}.ffn_down.weight", .{il}, .{ n_ff, n_embd }, .matrix);
        if ((il + 1) % 4 != 0) {
            add(&specs, &names, "blk.{d}.attn_qkv.weight", .{il}, .{ n_embd, conv_dim }, .matrix);
            add(&specs, &names, "blk.{d}.attn_gate.weight", .{il}, .{ n_embd, ssm_d_inner }, .matrix);
            add(&specs, &names, "blk.{d}.ssm_conv1d.weight", .{il}, .{ ssm_d_conv, conv_dim }, .conv);
            add(&specs, &names, "blk.{d}.ssm_dt.bias", .{il}, .{ ssm_dt_rank, 1 }, .dt);
            add(&specs, &names, "blk.{d}.ssm_a", .{il}, .{ ssm_dt_rank, 1 }, .ssm_a);
            add(&specs, &names, "blk.{d}.ssm_alpha.weight", .{il}, .{ n_embd, ssm_dt_rank }, .matrix);
            add(&specs, &names, "blk.{d}.ssm_beta.weight", .{il}, .{ n_embd, ssm_dt_rank }, .matrix);
            add(&specs, &names, "blk.{d}.ssm_norm.weight", .{il}, .{ @divExact(ssm_d_inner, ssm_dt_rank), 1 }, .norm);
            add(&specs, &names, "blk.{d}.ssm_out.weight", .{il}, .{ ssm_d_inner, n_embd }, .matrix);
        } else {
            add(&specs, &names, "blk.{d}.attn_q.weight", .{il}, .{ n_embd, 2 * n_head * head_dim }, .matrix);
            add(&specs, &names, "blk.{d}.attn_k.weight", .{il}, .{ n_embd, n_head_kv * head_dim }, .matrix);
            add(&specs, &names, "blk.{d}.attn_v.weight", .{il}, .{ n_embd, n_head_kv * head_dim }, .matrix);
            add(&specs, &names, "blk.{d}.attn_output.weight", .{il}, .{ n_head * head_dim, n_embd }, .matrix);
            add(&specs, &names, "blk.{d}.attn_q_norm.weight", .{il}, .{ head_dim, 1 }, .norm);
            add(&specs, &names, "blk.{d}.attn_k_norm.weight", .{il}, .{ head_dim, 1 }, .norm);
        }
    }

    var total: usize = 0;
    for (specs.items) |s| total += @intCast(s.ne[0] * s.ne[1] * 4);
    const ctx = c.ggml_init(.{ .mem_size = total + (specs.items.len + 1) * (c.ggml_tensor_overhead() + 64), .mem_buffer = null, .no_alloc = false }).?;
    defer c.ggml_free(ctx);
    var prng = std.Random.DefaultPrng.init(0x0b7d1ff);
    const rand = prng.random();
    for (specs.items) |s| {
        const t = if (s.ne[1] == 1) c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, s.ne[0]) else c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, s.ne[0], s.ne[1]);
        var nb: [128]u8 = undefined;
        _ = c.ggml_set_name(t, (std.fmt.bufPrintSentinel(&nb, "{s}", .{s.name}, 0) catch unreachable).ptr);
        const data: [*]f32 = @ptrCast(@alignCast(t.*.data));
        const n: usize = @intCast(s.ne[0] * s.ne[1]);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(s.ne[0])));
        for (data[0..n]) |*v| {
            const u = rand.float(f32) * 2 - 1;
            v.* = switch (s.kind) {
                .norm => 1.0 + 0.1 * u,
                .matrix => u * scale,
                .embd => 0.1 * u,
                .ssm_a => -(1.0 + u) * 0.5 - 0.1,
                .dt => 0.5 * u,
                .conv => 0.3 * u,
            };
        }
        c.gguf_add_tensor(g, t);
    }
    if (!c.gguf_write_to_file(g, path.ptr, false)) {
        std.debug.print("FAIL: cannot write {s}\n", .{path});
        std.process.exit(2);
    }
}

// -----------------------------------------------------------------------------
// The script

/// Which weights a case trains.
const Filter = enum {
    /// `llama_opt_param_filter_all`: every F32 weight.
    all,
    /// The output layer and its norm.
    output,
    /// Everything ggml can differentiate in this model: the output layer
    /// and its norm, and in the last layer the feed-forward block, its
    /// norm, the attention output and the query norm -- the query norm's
    /// gradient runs back through the attention, the softmax and M-RoPE.
    /// Measured, one weight at a time: every other weight either reaches a
    /// cache write -- `ggml_set_rows` into the KV cache, or the chunked
    /// Gated Delta Net's `ggml_set_inplace` -- and `ggml_build_backward_expand`
    /// asserts "inplace operations are currently not supported"
    /// (ggml.c:7318 @c1d0e7a00), or reaches the attention output gate's
    /// `ggml_sigmoid`, a unary op with no backward pass, and aborts with
    /// "fatal error" after "unsupported unary op for backward pass" (ggml.c:7093 @c1d0e7a00).
    head,
    /// One weight per abort, below.
    attn_q,
    ffn0,
    /// The names in `OPT_LLAMA_FILTER`, comma-separated: for probing what
    /// ggml can differentiate.
    custom,

    fn accepts(self: Filter, name: []const u8) bool {
        return switch (self) {
            .all => true,
            .output => std.mem.startsWith(u8, name, "output"),
            .head => std.mem.startsWith(u8, name, "output") or
                std.mem.startsWith(u8, name, "blk.3.ffn_") or
                std.mem.eql(u8, name, "blk.3.post_attention_norm.weight") or
                std.mem.eql(u8, name, "blk.3.attn_output.weight") or
                std.mem.eql(u8, name, "blk.3.attn_q_norm.weight"),
            .attn_q => std.mem.eql(u8, name, "blk.3.attn_q.weight"),
            .ffn0 => std.mem.eql(u8, name, "blk.0.ffn_up.weight"),
            .custom => blk: {
                const list = std.mem.span(std.c.getenv("OPT_LLAMA_FILTER") orelse break :blk false);
                var it = std.mem.splitScalar(u8, list, ',');
                while (it.next()) |n| if (std.mem.eql(u8, n, name)) break :blk true;
                break :blk false;
            },
        };
    }
};

var current_filter: Filter = .output;

/// Records every tensor it is offered, then decides by `current_filter`.
fn recordingFilter(tensor: [*c]const c.struct_ggml_tensor, userdata: ?*anyopaque) callconv(.c) bool {
    const t: *const c.ggml_tensor = @ptrCast(tensor);
    const name = std.mem.sliceTo(&t.name, 0);
    const accept = current_filter.accepts(name);
    note("filter {s} ud={} -> {}", .{ name, userdata == @as(?*anyopaque, @ptrCast(&current_filter)), accept });
    return accept;
}

var g_opt_pars: c.struct_ggml_opt_optimizer_params = undefined;
var n_opt_pars_calls: usize = 0;

fn optPars(userdata: ?*anyopaque) callconv(.c) c.struct_ggml_opt_optimizer_params {
    n_opt_pars_calls += 1;
    note("get_opt_pars ud={}", .{userdata == @as(?*anyopaque, @ptrCast(&g_opt_pars))});
    return g_opt_pars;
}

fn epochCb(train: bool, opt_ctx: c.ggml_opt_context_t, dataset: c.ggml_opt_dataset_t, result: c.ggml_opt_result_t, ibatch: i64, ibatch_max: i64, t_start_us: i64) callconv(.c) void {
    note("callback train={} opt_ctx={} dataset={} ibatch={d}/{d} t_start>0={}", .{ train, opt_ctx != null, dataset == g_dataset, ibatch, ibatch_max, t_start_us > 0 });
    recResult("callback", result);
}

var g_dataset: c.ggml_opt_dataset_t = null;

fn recResult(tag: []const u8, res: c.ggml_opt_result_t) void {
    var nd: i64 = 0;
    c.ggml_opt_result_ndata(res, &nd);
    var loss: f64 = 0;
    var unc: f64 = 0;
    c.ggml_opt_result_loss(res, &loss, &unc);
    var acc: f64 = 0;
    var acc_unc: f64 = 0;
    c.ggml_opt_result_accuracy(res, &acc, &acc_unc);
    var pred_h: u64 = 0;
    if (nd > 0) {
        const pred = gpa.alloc(i32, @intCast(nd)) catch unreachable;
        defer gpa.free(pred);
        c.ggml_opt_result_pred(res, pred.ptr);
        pred_h = hashBytes(@ptrCast(pred.ptr), pred.len * 4);
    }
    note("{s}: ndata={d} loss={x} unc={x} acc={x} acc_unc={x} pred={x}", .{ tag, nd, @as(u64, @bitCast(loss)), @as(u64, @bitCast(unc)), @as(u64, @bitCast(acc)), @as(u64, @bitCast(acc_unc)), pred_h });
}

const Case = struct {
    name: []const u8,
    filter: Filter,
    optimizer: c.enum_ggml_opt_optimizer_type,
    n_gpu_layers: i32,
    n_batch: u32 = 128,
    n_ubatch: u32 = 64,
    n_ctx_train: u32 = 0,
    epochs: usize = 2,
    /// Datapoints trained on before evaluation starts.
    idata_split: i64 = 2,
};

const cases = [_]Case{
    .{ .name = "gpu adamw output", .filter = .output, .optimizer = c.GGML_OPT_OPTIMIZER_TYPE_ADAMW, .n_gpu_layers = -1 },
    .{ .name = "gpu sgd head", .filter = .head, .optimizer = c.GGML_OPT_OPTIMIZER_TYPE_SGD, .n_gpu_layers = -1 },
    .{ .name = "gpu adamw head, one ubatch per batch", .filter = .head, .optimizer = c.GGML_OPT_OPTIMIZER_TYPE_ADAMW, .n_gpu_layers = -1, .n_batch = 64, .n_ubatch = 64 },
    .{ .name = "cpu adamw head", .filter = .head, .optimizer = c.GGML_OPT_OPTIMIZER_TYPE_ADAMW, .n_gpu_layers = 0 },
    .{ .name = "cpu sgd output, n_ctx_train 128", .filter = .output, .optimizer = c.GGML_OPT_OPTIMIZER_TYPE_SGD, .n_gpu_layers = 0, .n_ctx_train = 128, .idata_split = 3 },
};

/// The weights' names, in the port's `tensors_by_name` order, to read both
/// models' tensors by.
var weight_names: std.ArrayList([:0]u8) = .empty;

fn script(comptime side: enum { ref, port }, model: *c.struct_llama_model, cs: Case, tokens: []const i32) void {
    const Api = if (side == .ref) c else struct {
        const llama_init_from_model = A.context.llama_init_from_model;
        const llama_free = A.context.llama_free;
        const llama_decode = A.context.llama_decode;
        const llama_n_ctx = A.context.llama_n_ctx;
        const llama_opt_init = A.opt.llama_opt_init;
        const llama_opt_epoch = A.opt.llama_opt_epoch;
        const llama_opt_param_filter_all = A.opt.llama_opt_param_filter_all;
        const llama_model_n_ctx_train = A.model.llama_model_n_ctx_train;
    };
    const getTensor = struct {
        fn f(m: *c.struct_llama_model, name: [:0]const u8) ?*c.ggml_tensor {
            if (side == .ref) return _ZNK11llama_model10get_tensorEPKc(m, name.ptr);
            const pm: *M.Model = @ptrCast(@alignCast(m));
            for (pm.tensors_by_name.items) |t| {
                if (std.mem.eql(u8, std.mem.sliceTo(&t.name, 0), name)) return t;
            }
            return null;
        }
    }.f;

    var params = c.llama_context_default_params();
    params.cb_eval = nodeCb;
    params.cb_eval_user_data = null;
    params.no_perf = true;
    params.n_ctx = 256;
    params.n_batch = cs.n_batch;
    params.n_ubatch = cs.n_ubatch;
    params.type_k = c.GGML_TYPE_F32;
    params.type_v = c.GGML_TYPE_F32;
    params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.n_threads = 4;
    params.n_threads_batch = 4;

    const ctx = Api.llama_init_from_model(model, params) orelse {
        note("init failed", .{});
        return;
    };
    defer Api.llama_free(ctx);
    const n_ctx: i64 = Api.llama_n_ctx(ctx);
    note("n_ctx {d} n_ctx_train before {d}", .{ n_ctx, Api.llama_model_n_ctx_train(model) });

    // One decode first: it installs `cb_eval` on the scheduler, which the
    // training graphs then run under.
    {
        var b = c.llama_batch_init(8, 0, 1);
        defer c.llama_batch_free(b);
        for (0..4) |k| {
            b.token[k] = tokens[k];
            b.pos[k] = @intCast(k);
            b.n_seq_id[k] = 1;
            b.seq_id[k][0] = 0;
            b.logits[k] = @intFromBool(k == 3);
        }
        b.n_tokens = 4;
        note("warm decode: {d}", .{Api.llama_decode(ctx, b)});
    }

    current_filter = cs.filter;
    g_opt_pars = c.ggml_opt_get_default_optimizer_params(null);
    g_opt_pars.adamw.alpha = 0.01;
    g_opt_pars.adamw.wd = 0.01;
    g_opt_pars.sgd.alpha = 0.05;
    g_opt_pars.sgd.wd = 0.001;
    const lopt: c.struct_llama_opt_params = .{
        .n_ctx_train = cs.n_ctx_train,
        .param_filter = if (cs.filter == .all) Api.llama_opt_param_filter_all else recordingFilter,
        .param_filter_ud = &current_filter,
        .get_opt_pars = optPars,
        .get_opt_pars_ud = &g_opt_pars,
        .optimizer_type = cs.optimizer,
    };
    log_on = true;
    Api.llama_opt_init(ctx, model, lopt);
    if (twice) Api.llama_opt_init(ctx, model, lopt);
    note("n_ctx_train after {d}", .{Api.llama_model_n_ctx_train(model)});
    for (weight_names.items) |name| {
        const t = getTensor(model, name) orelse {
            note("weight {s}: missing", .{name});
            continue;
        };
        note("weight {s}: flags {x}", .{ name, t.flags });
    }

    // Datapoints of n_ctx tokens, each labelled with the next token, as
    // `common_opt_dataset_init` builds them, with a stride of half a
    // context.
    const ndata: i64 = 3;
    const stride: i64 = @divExact(n_ctx, 2);
    const dataset = c.ggml_opt_dataset_init(c.GGML_TYPE_I32, c.GGML_TYPE_I32, n_ctx, n_ctx, ndata, 1);
    defer c.ggml_opt_dataset_free(dataset);
    g_dataset = dataset;
    {
        const data: [*]i32 = @ptrCast(@alignCast(c.ggml_opt_dataset_data(dataset).*.data));
        const labels: [*]i32 = @ptrCast(@alignCast(c.ggml_opt_dataset_labels(dataset).*.data));
        for (0..@intCast(ndata)) |idata| {
            const at: usize = @intCast(@as(i64, @intCast(idata)) * stride);
            @memcpy(data[idata * @as(usize, @intCast(n_ctx)) ..][0..@intCast(n_ctx)], tokens[at..][0..@intCast(n_ctx)]);
            @memcpy(labels[idata * @as(usize, @intCast(n_ctx)) ..][0..@intCast(n_ctx)], tokens[at + 1 ..][0..@intCast(n_ctx)]);
        }
    }

    const result_train = c.ggml_opt_result_init();
    defer c.ggml_opt_result_free(result_train);
    const result_eval = c.ggml_opt_result_init();
    defer c.ggml_opt_result_free(result_eval);

    for (0..cs.epochs) |epoch| {
        note("epoch {d}", .{epoch});
        Api.llama_opt_epoch(ctx, dataset, result_train, result_eval, cs.idata_split, epochCb, epochCb);
        recResult("train", result_train);
        recResult("eval", result_eval);
        for (weight_names.items) |name| {
            const t = getTensor(model, name) orelse continue;
            note("weight {s}: h {x}", .{ name, tensorHash(t) });
        }
        c.ggml_opt_result_reset(result_train);
        c.ggml_opt_result_reset(result_eval);
    }
    note("get_opt_pars calls {d}", .{n_opt_pars_calls});
    n_opt_pars_calls = 0;
    log_on = false;
}

fn firstDiff(name: []const u8, a: []const u8, b: []const u8) usize {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return line - 1;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            std.debug.print("FAIL {s}: transcript line {d}:\n    ref  '{s}'\n    port '{s}'\n", .{ name, line, la orelse "<end>", lb orelse "<end>" });
            std.process.exit(1);
        }
    }
}

fn loadBoth(path: [:0]const u8, n_gpu_layers: i32, rm: **c.struct_llama_model, pm: **M.Model, splits: *std.ArrayList([:0]const u8)) void {
    var mp = c.llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    // Writable weights: training updates them in place.
    mp.load_mode = c.LLAMA_LOAD_MODE_NONE;
    rm.* = c.llama_model_load_from_file(path.ptr, mp) orelse {
        std.debug.print("FAIL: the C++ cannot load {s}\n", .{path});
        std.process.exit(1);
    };
    pm.* = M.loadFromFileImpl(.{ .path = path }, splits, null, null, mp) orelse {
        std.debug.print("FAIL: the port cannot load {s}\n", .{path});
        std.process.exit(1);
    };
}

fn randomTokens(n_vocab: i32) []i32 {
    const tokens = gpa.alloc(i32, 1024) catch unreachable;
    var prng = std.Random.DefaultPrng.init(0x70c3);
    const rand = prng.random();
    // A small alphabet, so the network has something learnable to fit.
    for (tokens) |*t| t.* = rand.intRangeLessThan(i32, 0, @min(n_vocab, 500));
    return tokens;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: opt_llama_diff <base-qwen35.gguf> <tmp-dir> [--abort <case> <ref|port>]\n", .{});
        std.process.exit(2);
    }
    _ = setenv("GGML_NO_BACKTRACE", "1", 1);
    const base = try gpa.dupeSentinel(u8, args[1], 0);
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/tiny-qwen35-f32.gguf", .{args[2]}, 0);

    c.llama_log_set(logCb, null);

    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    if (args.len >= 6 and std.mem.eql(u8, args[3], "--abort")) {
        // The main run writes the model; a lone abort run writes its own.
        if (std.c.access(path.ptr, 0) != 0) writeModel(base, path);
        return abortCase(path, args[4], args[5]);
    }

    writeModel(base, path);

    var checks: usize = 0;
    for (cases) |cs| {
        var rm: *c.struct_llama_model = undefined;
        var pm: *M.Model = undefined;
        var splits: std.ArrayList([:0]const u8) = .empty;
        defer {
            for (splits.items) |s| gpa.free(s);
            splits.deinit(gpa);
        }
        loadBoth(path, cs.n_gpu_layers, &rm, &pm, &splits);
        defer c.llama_model_free(rm);
        defer M.destroy(pm);

        weight_names.clearRetainingCapacity();
        for (pm.tensors_by_name.items) |t| weight_names.append(gpa, gpa.dupeSentinel(u8, std.mem.sliceTo(&t.name, 0), 0) catch unreachable) catch unreachable;
        const tokens = randomTokens(c.llama_vocab_n_tokens(c.llama_model_get_vocab(rm)));
        defer gpa.free(tokens);

        var ref_t: std.ArrayList(u8) = .empty;
        defer ref_t.deinit(gpa);
        var port_t: std.ArrayList(u8) = .empty;
        defer port_t.deinit(gpa);
        const nodes_before = n_nodes_seen;
        rec = &ref_t;
        script(.ref, rm, cs, tokens);
        const ref_nodes = n_nodes_seen - nodes_before;
        rec = &port_t;
        script(.port, @ptrCast(pm), cs, tokens);
        rec = null;
        if (ref_nodes == 0) {
            std.debug.print("FAIL {s}: the training graphs computed no nodes, so the node trace is empty\n", .{cs.name});
            std.process.exit(1);
        }
        if (std.c.getenv("OPT_LLAMA_DIFF_OUT")) |dir| dump(std.mem.span(dir), cs.name, ref_t.items, port_t.items);
        const n = firstDiff(cs.name, ref_t.items, port_t.items);
        std.debug.print("  {s:<40} ok {d:>7} lines, {d} nodes\n", .{ cs.name, n, ref_nodes });
        checks += n;
    }
    std.debug.print("opt-llama-diff: {d} transcript lines identical\n", .{checks});
}

fn dump(dir: []const u8, name: []const u8, a: []const u8, b: []const u8) void {
    for ([_][]const u8{ "ref", "port" }, [_][]const u8{ a, b }) |side, text| {
        var pb: [512]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&pb, "{s}/{s}.{s}.txt", .{ dir, name, side }, 0) catch unreachable;
        const f = std.c.fopen(p.ptr, "wb") orelse continue;
        _ = std.c.fwrite(text.ptr, 1, text.len, f);
        _ = std.c.fclose(f);
    }
}

// -----------------------------------------------------------------------------
// Cases that abort, one side per process

fn abortCase(path: [:0]const u8, which: []const u8, side: []const u8) void {
    var rm: *c.struct_llama_model = undefined;
    var pm: *M.Model = undefined;
    var splits: std.ArrayList([:0]const u8) = .empty;
    loadBoth(path, -1, &rm, &pm, &splits);
    const tokens = randomTokens(c.llama_vocab_n_tokens(c.llama_model_get_vocab(rm)));
    var cs: Case = .{ .name = "abort", .filter = .output, .optimizer = c.GGML_OPT_OPTIMIZER_TYPE_ADAMW, .n_gpu_layers = -1, .epochs = 1 };
    if (std.mem.eql(u8, which, "filter-all")) {
        cs.filter = .all;
    } else if (std.mem.eql(u8, which, "n_ctx_train")) {
        cs.n_ctx_train = 100;
    } else if (std.mem.eql(u8, which, "idata_split")) {
        cs.idata_split = 4;
    } else if (std.mem.eql(u8, which, "attn-q")) {
        cs.filter = .attn_q;
    } else if (std.mem.eql(u8, which, "ffn0")) {
        cs.filter = .ffn0;
    } else if (std.mem.eql(u8, which, "custom")) {
        cs.filter = .custom;
    } else if (std.mem.eql(u8, which, "init-twice")) {
        cs.epochs = 0;
        twice = true;
    } else {
        std.debug.print("unknown abort case {s}\n", .{which});
        std.process.exit(2);
    }
    var t: std.ArrayList(u8) = .empty;
    rec = &t;
    if (std.mem.eql(u8, side, "ref")) script(.ref, rm, cs, tokens) else script(.port, @ptrCast(pm), cs, tokens);
    std.debug.print("returned\n", .{});
}

var twice = false;
