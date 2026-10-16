//! Compares the port's Gemma 4 assistant with the C++'s, over a Gemma 4
//! target whose KV cache it shares.
//!
//! **Not a port.** `scripts/arch-model-diff` cannot reach
//! `gemma4-assistant`: upstream's fixture recipe trips the architecture's
//! own `n_layer_nextn == n_layer_all` assertion, and an assistant context
//! needs a target context (`ctx_other`) the harness never creates. This
//! builds both models from metadata alone through
//! `llama_model_init_from_user`, with the same deterministic tensor data on
//! both sides, and runs one script through the C++ (`libllama-ref.a`) and
//! through the port's `api` functions of the same names:
//!
//! - a Gemma 4 target of six layers -- SWA pattern `[1,1,1,0,1,0]`, two
//!   shared-KV layers (so layers 4 and 5 reuse layers 2 and 3), head sizes
//!   32 on the SWA layers and 64 on the full ones, per-layer embeddings --
//!   decodes a prompt with unmasked next-token embeddings;
//! - a two-layer assistant (`[1,0]`, n_embd 64 against the target's 128)
//!   shares the target's cache through `create_memory`'s `share` and drafts:
//!   single tokens at one position, as the speculative decoder does, then a
//!   batch with every output, then again after the target has moved on.
//!
//! Everything logged, every graph node (op, shape, strides, `op_params`,
//! sources, data hash), every return value and every output row's bits must
//! agree. Pointers in log lines are masked.
//!
//! Usage: gemma4_assistant_diff gpu|cpu [moe], as `scripts/gemma4-assistant-diff`
//! runs it.

const std = @import("std");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;

const gpa = std.heap.c_allocator;

/// The port, under the C names.
const P = struct {
    pub const llama_model_init_from_user = harness_root.api.llama.llama_model_init_from_user;
    pub const llama_model_free = harness_root.api.model.llama_model_free;
    pub const llama_model_load_from_file = harness_root.api.llama.llama_model_load_from_file;
    pub const llama_model_n_layer = harness_root.api.model.llama_model_n_layer;
    pub const llama_init_from_model = harness_root.api.context.llama_init_from_model;
    pub const llama_free = harness_root.api.context.llama_free;
    pub const llama_n_ctx = harness_root.api.context.llama_n_ctx;
    pub const llama_decode = harness_root.api.context.llama_decode;
    pub const llama_get_logits_ith = harness_root.api.context.llama_get_logits_ith;
    pub const llama_set_embeddings_nextn = harness_root.api.context.llama_set_embeddings_nextn;
    pub const llama_get_embeddings_nextn_ith = harness_root.api.context.llama_get_embeddings_nextn_ith;
    pub const llama_get_memory = harness_root.api.context.llama_get_memory;
    pub const llama_memory_seq_pos_min = harness_root.api.context.llama_memory_seq_pos_min;
    pub const llama_memory_seq_pos_max = harness_root.api.context.llama_memory_seq_pos_max;
};

/// The C++, under the C names. The next-token embedding calls are in
/// , which has C++ linkage, so they are reached by their
/// mangled names, taken from `nm` on `libllama-ref.a`.
const R = struct {
    pub const llama_model_init_from_user = c.llama_model_init_from_user;
    pub const llama_model_free = c.llama_model_free;
    pub const llama_model_load_from_file = c.llama_model_load_from_file;
    pub const llama_model_n_layer = c.llama_model_n_layer;
    pub const llama_init_from_model = c.llama_init_from_model;
    pub const llama_free = c.llama_free;
    pub const llama_n_ctx = c.llama_n_ctx;
    pub const llama_decode = c.llama_decode;
    pub const llama_get_logits_ith = c.llama_get_logits_ith;
    pub const llama_get_memory = c.llama_get_memory;
    pub const llama_memory_seq_pos_min = c.llama_memory_seq_pos_min;
    pub const llama_memory_seq_pos_max = c.llama_memory_seq_pos_max;
    pub const llama_set_embeddings_nextn = @extern(*const fn (?*c.struct_llama_context, bool, bool) callconv(.c) void, .{ .name = "_Z26llama_set_embeddings_nextnP13llama_contextbb" });
    pub const llama_get_embeddings_nextn_ith = @extern(*const fn (?*c.struct_llama_context, i32) callconv(.c) ?[*]f32, .{ .name = "_Z30llama_get_embeddings_nextn_ithP13llama_contexti" });
};

// -----------------------------------------------------------------------------
// The transcript

var rec: ?*std.ArrayList(u8) = null;

fn note(comptime fmt: []const u8, args: anytype) void {
    const r = rec orelse return;
    r.print(gpa, fmt ++ "\n", args) catch unreachable;
}

fn logCb(_: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const r = rec orelse return;
    r.appendSlice(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
}

fn hashBytes(p: [*]const u8, n: usize) u64 {
    return std.hash.Wyhash.hash(0, p[0..n]);
}

var scratch: std.ArrayList(u8) = .empty;
var n_nodes: usize = 0;

fn nodeCb(t: [*c]c.ggml_tensor, ask: bool, _: ?*anyopaque) callconv(.c) bool {
    if (ask) return true;
    const node: *c.ggml_tensor = t;
    n_nodes += 1;
    const r = rec orelse return true;
    r.print(gpa, "node {s} {s} {s} [{d},{d},{d},{d}] [{d},{d},{d},{d}] p", .{
        std.mem.sliceTo(&node.name, 0), c.ggml_op_desc(node), c.ggml_type_name(node.type),
        node.ne[0],                     node.ne[1],           node.ne[2],
        node.ne[3],                     node.nb[0],           node.nb[1],
        node.nb[2],                     node.nb[3],
    }) catch unreachable;
    for (node.op_params) |w| r.print(gpa, " {x}", .{@as(u32, @bitCast(w))}) catch unreachable;
    r.appendSlice(gpa, " src") catch unreachable;
    for (node.src) |s| {
        if (s == null) continue;
        const src: *c.ggml_tensor = s;
        r.print(gpa, " {s}:{s}", .{ std.mem.sliceTo(&src.name, 0), c.ggml_op_desc(src) }) catch unreachable;
    }
    if (node.view_src != null) {
        const v: *c.ggml_tensor = node.view_src;
        r.print(gpa, " view {s}+{d}", .{ std.mem.sliceTo(&v.name, 0), node.view_offs }) catch unreachable;
    }
    const n = c.ggml_nbytes(node);
    scratch.resize(gpa, n) catch unreachable;
    c.ggml_backend_tensor_get(node, scratch.items.ptr, 0, n);
    r.print(gpa, " flags {x} h {x}\n", .{ node.flags, hashBytes(scratch.items.ptr, n) }) catch unreachable;
    return true;
}

/// Lines whose figures move with time, with what the process holds, or
/// with where an allocator put something.
fn normalize(log: []const u8) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, log, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "ggml_metal_library_compile_pipeline:")) continue;
        if (!first) out.append(gpa, '\n') catch unreachable;
        first = false;
        if (std.mem.find(u8, line, "reserve took ")) |at| {
            out.appendSlice(gpa, line[0 .. at + "reserve took ".len]) catch unreachable;
            out.appendSlice(gpa, "N ms") catch unreachable;
            if (std.mem.find(u8, line, ", sched copies")) |rest| out.appendSlice(gpa, line[rest..]) catch unreachable;
            continue;
        }
        if (std.mem.startsWith(u8, line, "ggml_metal_log_allocated_size:")) {
            if (std.mem.findScalar(u8, line, '(')) |open| {
                out.appendSlice(gpa, line[0..open]) catch unreachable;
                out.appendSlice(gpa, "(N)") catch unreachable;
                continue;
            }
        }
        if (std.mem.find(u8, line, "available_memory_kb: ")) |at| {
            // CUDA on a unified-memory device reads the system's free
            // memory from /proc/meminfo, which moves between any two reads.
            out.appendSlice(gpa, line[0 .. at + "available_memory_kb: ".len]) catch unreachable;
            out.appendSlice(gpa, "N") catch unreachable;
            continue;
        }
        if (std.mem.find(u8, line, " MiB free")) |end| {
            // A device's free memory moves with what the process holds.
            var start = end;
            while (start > 0 and std.ascii.isDigit(line[start - 1])) start -= 1;
            out.appendSlice(gpa, line[0..start]) catch unreachable;
            out.appendSlice(gpa, "N") catch unreachable;
            out.appendSlice(gpa, line[end..]) catch unreachable;
            continue;
        }
        // Pointers.
        var i: usize = 0;
        while (i < line.len) {
            if (i + 1 < line.len and line[i] == '0' and line[i + 1] == 'x') {
                out.appendSlice(gpa, "0xP") catch unreachable;
                i += 2;
                while (i < line.len and std.ascii.isHex(line[i])) i += 1;
                continue;
            }
            out.append(gpa, line[i]) catch unreachable;
            i += 1;
        }
    }
    return out;
}

fn firstDiff(a: []const u8, b: []const u8) bool {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return true;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            std.debug.print("FAIL: transcript line {d}:\n    ref  '{s}'\n    port '{s}'\n", .{ line, la orelse "<end>", lb orelse "<end>" });
            return false;
        }
    }
}

// -----------------------------------------------------------------------------
// The models

const n_vocab = 128;
const n_ctx_train = 256;

fn setU32(g: *c.gguf_context, comptime arch: []const u8, comptime key: []const u8, v: u32) void {
    c.gguf_set_val_u32(g, arch ++ "." ++ key, v);
}
fn setF32(g: *c.gguf_context, comptime arch: []const u8, comptime key: []const u8, v: f32) void {
    c.gguf_set_val_f32(g, arch ++ "." ++ key, v);
}
fn setPattern(g: *c.gguf_context, comptime arch: []const u8, pattern: []const bool) void {
    c.gguf_set_arr_data(g, arch ++ ".attention.sliding_window_pattern", c.GGUF_TYPE_BOOL, pattern.ptr, pattern.len);
}

/// A Gemma 4 target: six layers, two of them reading earlier layers' KV.
fn targetMeta() *c.gguf_context {
    const g: *c.gguf_context = c.gguf_init_empty().?;
    const a = "gemma4";
    c.gguf_set_val_str(g, "general.architecture", a);
    c.gguf_set_val_str(g, "tokenizer.ggml.model", "no_vocab");
    setU32(g, a, "vocab_size", n_vocab);
    setU32(g, a, "context_length", n_ctx_train);
    setU32(g, a, "embedding_length", 128);
    setU32(g, a, "feed_forward_length", 192);
    setU32(g, a, "block_count", 6);
    setU32(g, a, "attention.head_count", 2);
    setU32(g, a, "attention.head_count_kv", 1);
    setU32(g, a, "attention.key_length", 64);
    setU32(g, a, "attention.value_length", 64);
    setU32(g, a, "attention.key_length_swa", 32);
    setU32(g, a, "attention.value_length_swa", 32);
    setU32(g, a, "rope.dimension_count", 64);
    setU32(g, a, "rope.dimension_count_swa", 32);
    setF32(g, a, "rope.freq_base", 1000000.0);
    setF32(g, a, "rope.freq_base_swa", 10000.0);
    setF32(g, a, "attention.layer_norm_rms_epsilon", 1e-6);
    setU32(g, a, "attention.sliding_window", 16);
    setU32(g, a, "attention.shared_kv_layers", 2);
    setU32(g, a, "embedding_length_per_layer_input", 32);
    setF32(g, a, "final_logit_softcapping", 30.0);
    setPattern(g, a, &.{ true, true, true, false, true, false });
    return g;
}

/// A Gemma 4 assistant for that target: two next-token layers.
fn assistantMeta() *c.gguf_context {
    const g: *c.gguf_context = c.gguf_init_empty().?;
    const a = "gemma4-assistant";
    c.gguf_set_val_str(g, "general.architecture", a);
    c.gguf_set_val_str(g, "tokenizer.ggml.model", "no_vocab");
    setU32(g, a, "vocab_size", n_vocab);
    setU32(g, a, "context_length", n_ctx_train);
    setU32(g, a, "embedding_length", 64);
    setU32(g, a, "embedding_length_out", 128);
    setU32(g, a, "feed_forward_length", 96);
    setU32(g, a, "block_count", 2);
    setU32(g, a, "nextn_predict_layers", 2);
    setU32(g, a, "attention.head_count", 2);
    setU32(g, a, "attention.head_count_kv", 1);
    setU32(g, a, "attention.key_length", 64);
    setU32(g, a, "attention.value_length", 64);
    setU32(g, a, "attention.key_length_swa", 32);
    setU32(g, a, "attention.value_length_swa", 32);
    setU32(g, a, "rope.dimension_count", 64);
    setU32(g, a, "rope.dimension_count_swa", 32);
    setF32(g, a, "rope.freq_base", 1000000.0);
    setF32(g, a, "rope.freq_base_swa", 10000.0);
    setF32(g, a, "attention.layer_norm_rms_epsilon", 1e-6);
    setU32(g, a, "attention.sliding_window", 16);
    setPattern(g, a, &.{ true, false });
    return g;
}

/// Deterministic data from the tensor's name: uniform in [-0.25, 0.25),
/// norms near one so activations keep their scale.
fn setTensorData(t_: [*c]c.ggml_tensor, ud: ?*anyopaque) callconv(.c) void {
    const t: *c.ggml_tensor = t_;
    const seed: *const u64 = @ptrCast(@alignCast(ud.?));
    const name = std.mem.sliceTo(&t.name, 0);
    var rng = std.Random.DefaultPrng.init(seed.* ^ std.hash.Wyhash.hash(0, name));
    const r = rng.random();
    const is_norm = std.mem.find(u8, name, "norm") != null or std.mem.find(u8, name, "out_scale") != null or std.mem.find(u8, name, "rope_freqs") != null;
    const n: usize = @intCast(c.ggml_nelements(t));
    const vals = gpa.alloc(f32, n) catch unreachable;
    defer gpa.free(vals);
    for (vals) |*v| v.* = if (is_norm) 1.0 + (r.float(f32) - 0.5) * 0.2 else (r.float(f32) - 0.5) * 0.5;
    switch (t.type) {
        c.GGML_TYPE_F32 => c.ggml_backend_tensor_set(t, vals.ptr, 0, c.ggml_nbytes(t)),
        c.GGML_TYPE_F16 => {
            const h = gpa.alloc(c.ggml_fp16_t, n) catch unreachable;
            defer gpa.free(h);
            for (vals, h) |v, *o| o.* = c.ggml_fp32_to_fp16(v);
            c.ggml_backend_tensor_set(t, h.ptr, 0, c.ggml_nbytes(t));
        },
        else => @panic("tensor type"),
    }
}

const arch_mod = harness_root.arch;

/// Fills `t`'s host data as `setTensorData` fills a device tensor.
fn fillHost(t: *c.ggml_tensor, seed: u64) void {
    const name = std.mem.sliceTo(&t.name, 0);
    var rng = std.Random.DefaultPrng.init(seed ^ std.hash.Wyhash.hash(0, name));
    const r = rng.random();
    const is_norm = std.mem.find(u8, name, "norm") != null or std.mem.find(u8, name, "out_scale") != null or std.mem.find(u8, name, "rope_freqs") != null;
    const n: usize = @intCast(c.ggml_nelements(t));
    const data: [*]f32 = @ptrCast(@alignCast(t.data.?));
    for (data[0..n]) |*v| v.* = if (is_norm) 1.0 + (r.float(f32) - 0.5) * 0.2 else (r.float(f32) - 0.5) * 0.5;
}

/// Writes the assistant as a file: `llama_model_init_from_user` cannot
/// build it, since it creates the optional `masked_embd_*` tensors with no
/// shape and the C++ aborts in `ggml_new_tensor`.
fn writeAssistant(path: [:0]const u8, seed: u64) void {
    const meta = assistantMeta();
    defer c.gguf_free(meta);
    const gctx: *c.ggml_context = c.ggml_init(.{ .mem_size = 8 << 20, .mem_buffer = null, .no_alloc = false }).?;
    defer c.ggml_free(gctx);
    const n_embd = 64;
    const n_ff = 96;
    Add.t(gctx, meta, seed, .token_embd, "weight", -1, &.{ n_embd, n_vocab });
    Add.t(gctx, meta, seed, .output_norm, "weight", -1, &.{n_embd});
    Add.t(gctx, meta, seed, .nextn_proj_post, "weight", -1, &.{ n_embd, n_embd_tgt });
    Add.t(gctx, meta, seed, .nextn_proj_pre, "weight", 0, &.{ 2 * n_embd_tgt, n_embd });
    for (0..2) |il| {
        const i: i32 = @intCast(il);
        const hd: i64 = if (il == 0) 32 else 64; // [1, 0]: SWA, then full
        Add.t(gctx, meta, seed, .attn_norm, "weight", i, &.{n_embd});
        Add.t(gctx, meta, seed, .attn_q, "weight", i, &.{ n_embd, hd * 2 });
        Add.t(gctx, meta, seed, .attn_out, "weight", i, &.{ hd * 2, n_embd });
        Add.t(gctx, meta, seed, .attn_q_norm, "weight", i, &.{hd});
        Add.t(gctx, meta, seed, .attn_post_norm, "weight", i, &.{n_embd});
        Add.t(gctx, meta, seed, .layer_out_scale, "weight", i, &.{1});
        if (il == 1) Add.t(gctx, meta, seed, .rope_freqs, "weight", i, &.{@divTrunc(hd, 2)});
        Add.t(gctx, meta, seed, .ffn_norm, "weight", i, &.{n_embd});
        Add.t(gctx, meta, seed, .ffn_gate, "weight", i, &.{ n_embd, n_ff });
        Add.t(gctx, meta, seed, .ffn_up, "weight", i, &.{ n_embd, n_ff });
        Add.t(gctx, meta, seed, .ffn_down, "weight", i, &.{ n_ff, n_embd });
        Add.t(gctx, meta, seed, .ffn_post_norm, "weight", i, &.{n_embd});
    }
    if (!c.gguf_write_to_file(meta, path.ptr, false)) @panic("gguf_write_to_file");
}

const Add = struct {
    fn t(g: *c.ggml_context, m: *c.gguf_context, seed_: u64, tensor: arch_mod.Tensor, suffix: ?[]const u8, bid: i32, ne: []const i64) void {
        const nm = arch_mod.tensorName(tensor, suffix, bid, -1);
        const x: *c.ggml_tensor = c.ggml_new_tensor(g, c.GGML_TYPE_F32, @intCast(ne.len), ne.ptr);
        var buf: [128]u8 = undefined;
        const z = std.fmt.bufPrintSentinel(&buf, "{s}", .{nm.slice()}, 0) catch unreachable;
        _ = c.ggml_set_name(x, z.ptr);
        fillHost(x, seed_);
        c.gguf_add_tensor(m, x);
    }
};

/// The target as a file, with a mixture of experts on layers 1, 3 and 5 --
/// fused gate-up experts on layer 1, separate ones on 3 and 5 -- a V
/// projection on layers 0 and 3 only, and an output scale on layers 1 and
/// 4. `llama_model_init_from_user` creates none of a layer's optional
/// tensors, so this variant cannot come from metadata alone.
fn writeTarget(path: [:0]const u8, seed: u64) void {
    const meta = targetMeta();
    defer c.gguf_free(meta);
    setU32(meta, "gemma4", "expert_count", 4);
    setU32(meta, "gemma4", "expert_used_count", 2);
    setU32(meta, "gemma4", "expert_feed_forward_length", 64);
    const gctx: *c.ggml_context = c.ggml_init(.{ .mem_size = 32 << 20, .mem_buffer = null, .no_alloc = false }).?;
    defer c.ggml_free(gctx);
    const n_embd = 128;
    const n_ff = 192;
    const n_pl = 32;
    const n_exp = 4;
    const n_ff_exp = 64;
    Add.t(gctx, meta, seed, .token_embd, "weight", -1, &.{ n_embd, n_vocab });
    Add.t(gctx, meta, seed, .output_norm, "weight", -1, &.{n_embd});
    Add.t(gctx, meta, seed, .per_layer_token_embd, "weight", -1, &.{ n_pl * 6, n_vocab });
    Add.t(gctx, meta, seed, .per_layer_model_proj, "weight", 0, &.{ n_embd, n_pl * 6 });
    Add.t(gctx, meta, seed, .per_layer_proj_norm, "weight", 0, &.{n_pl});
    const swa = [_]bool{ true, true, true, false, true, false };
    var rope_freqs_done = false;
    for (0..6) |il| {
        const i: i32 = @intCast(il);
        const hd: i64 = if (swa[il]) 32 else 64;
        const has_kv = il < 4;
        Add.t(gctx, meta, seed, .attn_norm, "weight", i, &.{n_embd});
        Add.t(gctx, meta, seed, .attn_q, "weight", i, &.{ n_embd, hd * 2 });
        if (has_kv) Add.t(gctx, meta, seed, .attn_k, "weight", i, &.{ n_embd, hd });
        if (il == 0 or il == 3) Add.t(gctx, meta, seed, .attn_v, "weight", i, &.{ n_embd, hd });
        Add.t(gctx, meta, seed, .attn_out, "weight", i, &.{ hd * 2, n_embd });
        Add.t(gctx, meta, seed, .attn_q_norm, "weight", i, &.{hd});
        if (has_kv) Add.t(gctx, meta, seed, .attn_k_norm, "weight", i, &.{hd});
        Add.t(gctx, meta, seed, .attn_post_norm, "weight", i, &.{n_embd});
        if (il == 1 or il == 4) Add.t(gctx, meta, seed, .layer_out_scale, "weight", i, &.{1});
        if (!swa[il] and !rope_freqs_done) {
            Add.t(gctx, meta, seed, .rope_freqs, "weight", i, &.{@divTrunc(hd, 2)});
            rope_freqs_done = true;
        }
        Add.t(gctx, meta, seed, .ffn_norm, "weight", i, &.{n_embd});
        Add.t(gctx, meta, seed, .ffn_gate, "weight", i, &.{ n_embd, n_ff });
        Add.t(gctx, meta, seed, .ffn_up, "weight", i, &.{ n_embd, n_ff });
        Add.t(gctx, meta, seed, .ffn_down, "weight", i, &.{ n_ff, n_embd });
        Add.t(gctx, meta, seed, .ffn_post_norm, "weight", i, &.{n_embd});
        if (il % 2 == 1) {
            Add.t(gctx, meta, seed, .ffn_gate_inp, "weight", i, &.{ n_embd, n_exp });
            Add.t(gctx, meta, seed, .ffn_gate_inp, "scale", i, &.{n_embd});
            Add.t(gctx, meta, seed, .ffn_pre_norm_2, "weight", i, &.{n_embd});
            Add.t(gctx, meta, seed, .ffn_post_norm_1, "weight", i, &.{n_embd});
            Add.t(gctx, meta, seed, .ffn_post_norm_2, "weight", i, &.{n_embd});
            if (il == 1) {
                Add.t(gctx, meta, seed, .ffn_gate_up_exps, "weight", i, &.{ n_embd, n_ff_exp * 2, n_exp });
            } else {
                Add.t(gctx, meta, seed, .ffn_gate_exps, "weight", i, &.{ n_embd, n_ff_exp, n_exp });
                Add.t(gctx, meta, seed, .ffn_up_exps, "weight", i, &.{ n_embd, n_ff_exp, n_exp });
            }
            Add.t(gctx, meta, seed, .ffn_down_exps, "weight", i, &.{ n_ff_exp, n_embd, n_exp });
        }
        Add.t(gctx, meta, seed, .per_layer_inp_gate, "weight", i, &.{ n_embd, n_pl });
        Add.t(gctx, meta, seed, .per_layer_proj, "weight", i, &.{ n_pl, n_embd });
        Add.t(gctx, meta, seed, .per_layer_post_norm, "weight", i, &.{n_embd});
    }
    if (!c.gguf_write_to_file(meta, path.ptr, false)) @panic("gguf_write_to_file");
}

var assistant_path: [:0]const u8 = "";
var target_path: ?[:0]const u8 = null;

// -----------------------------------------------------------------------------
// The script

fn argmax(r: [*]const f32) i32 {
    var best: usize = 0;
    for (1..n_vocab) |i| if (r[i] > r[best]) {
        best = i;
    };
    return @intCast(best);
}

fn row(comptime what: []const u8, p: ?[*]f32, n: usize, i: i32) void {
    if (p) |r| {
        note(what ++ " {d}: {x}", .{ i, hashBytes(@ptrCast(r), n * @sizeOf(f32)) });
    } else {
        note(what ++ " {d}: null", .{i});
    }
}

const n_embd_tgt = 128;
const n_prompt = 20;

fn script(comptime A: type, gpu: bool) void {
    var seed: u64 = 42;
    var mp = c.llama_model_default_params();
    if (!gpu) mp.n_gpu_layers = 0;

    const meta_t = targetMeta();
    defer c.gguf_free(meta_t);
    const target_ = if (target_path) |tp| A.llama_model_load_from_file(tp.ptr, mp) else A.llama_model_init_from_user(meta_t, setTensorData, &seed, mp);
    const target = target_ orelse {
        note("target: init failed", .{});
        return;
    };
    defer A.llama_model_free(target);
    note("target n_layer {d}", .{A.llama_model_n_layer(target)});

    var cp = c.llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_ubatch = 64;
    cp.cb_eval = nodeCb;
    cp.no_perf = true;
    const tctx = A.llama_init_from_model(target, cp) orelse {
        note("target: context failed", .{});
        return;
    };
    defer A.llama_free(tctx);
    A.llama_set_embeddings_nextn(tctx, true, false);

    var tb = c.llama_batch_init(64, 0, 1);
    defer c.llama_batch_free(tb);
    var tokens: [n_prompt]i32 = undefined;
    for (&tokens, 0..) |*t, k| t.* = @intCast((k * 37 + 11) % n_vocab);
    for (tokens, 0..) |t, k| {
        tb.token[k] = t;
        tb.pos[k] = @intCast(k);
        tb.n_seq_id[k] = 1;
        tb.seq_id[k][0] = 0;
        tb.logits[k] = @intFromBool(k + 1 == n_prompt);
    }
    tb.n_tokens = n_prompt;
    note("target decode prompt: {d}", .{A.llama_decode(tctx, tb)});
    const last = A.llama_get_logits_ith(tctx, -1);
    row("target logits", last, n_vocab, -1);
    var id: i32 = if (last) |r| argmax(r) else 0;
    var h: [n_embd_tgt]f32 = @splat(0);
    for (0..n_prompt) |k| row("target h", A.llama_get_embeddings_nextn_ith(tctx, @intCast(k)), n_embd_tgt, @intCast(k));
    if (A.llama_get_embeddings_nextn_ith(tctx, n_prompt - 1)) |r| @memcpy(&h, r[0..n_embd_tgt]);
    const tmem = A.llama_get_memory(tctx);
    note("target seq 0: [{d}, {d}]", .{ A.llama_memory_seq_pos_min(tmem, 0), A.llama_memory_seq_pos_max(tmem, 0) });

    // The assistant, over the target's context.
    const assistant = A.llama_model_load_from_file(assistant_path.ptr, mp) orelse {
        note("assistant: init failed", .{});
        return;
    };
    defer A.llama_model_free(assistant);

    // Without a target the C++ refuses the context.
    var dcp = c.llama_context_default_params();
    dcp.cb_eval = nodeCb;
    dcp.no_perf = true;
    dcp.ctx_type = c.LLAMA_CONTEXT_TYPE_MTP;
    dcp.n_ctx = A.llama_n_ctx(tctx);
    dcp.n_ubatch = 64;
    dcp.n_rs_seq = 0;
    if (A.llama_init_from_model(assistant, dcp)) |bad| {
        note("assistant without target: created", .{});
        A.llama_free(bad);
    } else note("assistant without target: refused", .{});

    dcp.ctx_other = tctx;
    const dctx = A.llama_init_from_model(assistant, dcp) orelse {
        note("assistant: context failed", .{});
        return;
    };
    defer A.llama_free(dctx);
    A.llama_set_embeddings_nextn(dctx, true, true);
    const dmem = A.llama_get_memory(dctx);
    note("assistant memory {} seq 0: [{d}, {d}]", .{ dmem != null, A.llama_memory_seq_pos_min(dmem, 0), A.llama_memory_seq_pos_max(dmem, 0) });

    // llama_batch_init allocates only one of token and embd; the assistant
    // needs both.
    var db = c.llama_batch_init(8, n_embd_tgt, 1);
    defer {
        std.c.free(db.token);
        db.token = null;
        c.llama_batch_free(db);
    }
    db.token = @ptrCast(@alignCast(std.c.malloc(8 * @sizeOf(i32)).?));

    const Draft = struct {
        fn steps(dctx_: anytype, db_: *c.llama_batch, id_: *i32, h_: *[n_embd_tgt]f32, pos: i32, n: usize) void {
            for (0..n) |s| {
                db_.n_tokens = 0;
                db_.token[0] = id_.*;
                db_.pos[0] = pos;
                db_.n_seq_id[0] = 1;
                db_.seq_id[0][0] = 0;
                db_.logits[0] = 1;
                @memcpy(db_.embd[0..n_embd_tgt], h_);
                db_.n_tokens = 1;
                note("draft {d} at {d}: {d}", .{ s, pos, A.llama_decode(dctx_, db_.*) });
                const l = A.llama_get_logits_ith(dctx_, 0);
                row("draft logits", l, n_vocab, 0);
                if (l) |r| id_.* = argmax(r);
                note("draft token {d}", .{id_.*});
                const hn = A.llama_get_embeddings_nextn_ith(dctx_, 0);
                row("draft h", hn, n_embd_tgt, 0);
                if (hn) |r| @memcpy(h_, r[0..n_embd_tgt]);
            }
        }
    };

    // Single tokens at one position, as the speculative decoder drafts
    // over a shared cache.
    Draft.steps(dctx, &db, &id, &h, n_prompt, 3);

    // A batch with every output.
    db.n_tokens = 0;
    for (0..4) |k| {
        db.token[k] = tokens[k + 3];
        db.pos[k] = @intCast(n_prompt - 4 + k);
        db.n_seq_id[k] = 1;
        db.seq_id[k][0] = 0;
        db.logits[k] = 1;
        for (0..n_embd_tgt) |e| db.embd[k * n_embd_tgt + e] = h[(e + k * 7) % n_embd_tgt];
    }
    db.n_tokens = 4;
    note("draft batch: {d}", .{A.llama_decode(dctx, db)});
    for (0..4) |k| {
        row("draft logits", A.llama_get_logits_ith(dctx, @intCast(k)), n_vocab, @intCast(k));
        row("draft h", A.llama_get_embeddings_nextn_ith(dctx, @intCast(k)), n_embd_tgt, @intCast(k));
    }

    // The target moves on; the assistant sees the new cells.
    tb.n_tokens = 0;
    for (0..2) |k| {
        tb.token[k] = id;
        tb.pos[k] = @intCast(n_prompt + k);
        tb.n_seq_id[k] = 1;
        tb.seq_id[k][0] = 0;
        tb.logits[k] = 1;
    }
    tb.n_tokens = 2;
    note("target decode step: {d}", .{A.llama_decode(tctx, tb)});
    row("target logits", A.llama_get_logits_ith(tctx, 1), n_vocab, 1);
    if (A.llama_get_embeddings_nextn_ith(tctx, 1)) |r| @memcpy(&h, r[0..n_embd_tgt]);
    note("target seq 0: [{d}, {d}]", .{ A.llama_memory_seq_pos_min(tmem, 0), A.llama_memory_seq_pos_max(tmem, 0) });
    note("assistant seq 0: [{d}, {d}]", .{ A.llama_memory_seq_pos_min(dmem, 0), A.llama_memory_seq_pos_max(dmem, 0) });
    Draft.steps(dctx, &db, &id, &h, n_prompt + 2, 2);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const gpu = !(args.len > 1 and std.mem.eql(u8, args[1], "cpu"));

    c.llama_log_set(logCb, null);
    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    const dir = if (std.c.getenv("ASSISTANT_DIFF_TMP")) |d| std.mem.span(d) else "/tmp";
    var path_buf: [512]u8 = undefined;
    assistant_path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/gemma4-assistant.gguf", .{dir}, 0);
    writeAssistant(assistant_path, 42);
    var tpath_buf: [512]u8 = undefined;
    if (args.len > 2 and std.mem.eql(u8, args[2], "moe")) {
        const tp = try std.fmt.bufPrintSentinel(&tpath_buf, "{s}/gemma4-moe.gguf", .{dir}, 0);
        writeTarget(tp, 42);
        target_path = tp;
    }

    var ref: std.ArrayList(u8) = .empty;
    var port: std.ArrayList(u8) = .empty;
    const side = if (std.c.getenv("ASSISTANT_DIFF_SIDE")) |s| std.mem.span(s) else "";

    if (!std.mem.eql(u8, side, "port")) {
        rec = &ref;
        n_nodes = 0;
        script(R, gpu);
        rec = null;
        std.debug.print("ref: {d} nodes\n", .{n_nodes});
    }
    if (!std.mem.eql(u8, side, "ref")) {
        rec = &port;
        n_nodes = 0;
        script(P, gpu);
        rec = null;
        std.debug.print("port: {d} nodes\n", .{n_nodes});
    }
    var rn = normalize(ref.items);
    var pn = normalize(port.items);
    if (std.c.getenv("ASSISTANT_DIFF_OUT")) |d| {
        var pb: [512]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/{s}.ref", .{ std.mem.span(d), if (gpu) "gpu" else "cpu" }), .data = rn.items });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/{s}.port", .{ std.mem.span(d), if (gpu) "gpu" else "cpu" }), .data = pn.items });
    }
    if (side.len != 0) return;
    const ok = firstDiff(rn.items, pn.items);
    std.debug.print("{s}: {s}, {d} lines\n", .{ if (gpu) "gpu" else "cpu", if (ok) "PASS" else "FAIL", std.mem.count(u8, rn.items, "\n") });
    rn.deinit(gpa);
    pn.deinit(gpa);
    if (!ok) std.process.exit(1);
}
