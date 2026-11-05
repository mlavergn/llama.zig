//! Compares `src/llama/context.zig` and everything a decode reaches --
//! the batch splitter, the memory modules, the graph builder and Qwen3.5's
//! graph -- with llama.cpp's `llama-context.cpp` and its callees.
//!
//! **Not a port.** The harness behind `scripts/context-diff`. It loads one
//! model twice in one process, through the C++ in `libllama.a` and through
//! the port, and for every case runs the same script twice: once through
//! `llama.h`, once through the port's `api` functions of the same names.
//! Each side writes a transcript, and the two must be identical:
//!
//! - **everything logged**, from `llama_init_from_model` to `llama_free`
//!   -- the context's parameters, the fused-op resolution, the reserve's
//!   node and split counts, the compute buffer sizes, every error;
//! - **every graph node computed**, through `cb_eval`: its name, op, type,
//!   shape, strides, `op_params`, sources by name and op, and a hash of
//!   its data -- which also fixes the node order;
//! - every return value, every logit and embedding row's bits, the memory's
//!   positions per sequence after each step, and the perf counters that
//!   are not times.
//!
//! The script decodes a prompt, then greedy steps from its own logits, a
//! batch with every output, interleaved sequences, the memory API on them,
//! errors the C++ reports, and the same again after a clear, so the graph
//! is reused where the C++ reuses it.
//!
//! Usage: context_diff <model.gguf> [--quick] [--model <config>] [--case <substring>]
//!   [--case-exact <name>] [--skip-case <name>] [--list-cases]
//!   [--hadamard | --hadamard-tied] [--section-order <tyxz|zyxt>] [--target <model.gguf>]
//!
//! v0.7.0 additions: the `moe-cache` and `host-experts` configurations run
//! only when named with `--model` (they need a MoE model; deepseek2-moe's
//! 64 experts exercise the LRU, `MOE_CACHE_SIZE` sets the budget);
//! `--hadamard` derives `prism.hadamard` metadata onto the model (version
//! 1, explicit signs, latent token embedding), `--hadamard-tied` the tied
//! version 2 without `output.weight`; `--section-order` sets the M-RoPE
//! section order; `--target` gives every context a target's context as
//! `ctx_other`.

const std = @import("std");
const input_tensor_order = @import("input_tensor_order.zig");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;
const M = harness_root.model;
const api = harness_root.api.context;

/// `llama_batch_ext_set_decision_order` in `libllama-ref.a`, which
/// `llama-ext.h` declares with C++ linkage: the name is `nm`'s (the enum is
/// passed as its `unsigned int`).
extern fn _Z34llama_batch_ext_set_decision_orderP15llama_batch_exti20llama_decision_order(b: ?*c.struct_llama_batch_ext, idx: i32, order: c_uint) bool;

fn ref_ext_set_decision_order(b: ?*c.struct_llama_batch_ext, idx: i32, order: i32) c_int {
    return @intFromBool(_Z34llama_batch_ext_set_decision_orderP15llama_batch_exti20llama_decision_order(b, idx, @bitCast(order)));
}

/// `llama_model_set_attn_rot` and `llama_memory_get_buffers` in
/// `libllama-ref.a`, both `llama-ext.h`'s, so C++ linkage: the names are
/// `nm`'s. The second returns a `std::vector<ggml_backend_buffer_t>`, which
/// the aarch64 and x86_64 ABIs return through a hidden pointer as they
/// return any 24-byte struct; its storage is freed with `operator delete`.
extern fn _Z24llama_model_set_attn_rotP11llama_modelb(model: ?*c.struct_llama_model, value: bool) void;
const StdVector = extern struct { begin: [*c]c.ggml_backend_buffer_t, end: [*c]c.ggml_backend_buffer_t, cap: [*c]c.ggml_backend_buffer_t };
extern fn _Z24llama_memory_get_buffersP14llama_memory_i(mem: c.llama_memory_t) StdVector;
extern fn _ZdlPv(p: ?*anyopaque) void;
extern fn _Z26llama_set_embeddings_nextnP13llama_contextbb(ctx: ?*c.struct_llama_context, value: bool, masked: bool) void;
extern fn _Z26llama_get_embeddings_nextnP13llama_context(ctx: ?*c.struct_llama_context) [*c]f32;

/// Both sides' `llama_memory_get_buffers`, as the buffer names and sizes
/// in the order returned -- which is the buffers' address order, so only
/// a count and a sorted multiset are compared.
fn noteMemoryBuffers(comptime A: type, mem: c.llama_memory_t) void {
    if (mem == null) return;
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    if (A == c) {
        const v = _Z24llama_memory_get_buffersP14llama_memory_i(mem);
        const n: usize = (@intFromPtr(v.end) - @intFromPtr(v.begin)) / @sizeOf(c.ggml_backend_buffer_t);
        for (0..n) |i| names.append(gpa, std.fmt.allocPrint(gpa, "{s}:{d}", .{ std.mem.span(c.ggml_backend_buffer_name(v.begin[i])), c.ggml_backend_buffer_get_size(v.begin[i]) }) catch unreachable) catch unreachable;
        if (v.begin != null) _ZdlPv(@ptrCast(v.begin));
    } else {
        const bufs = harness_root.api.context.llama_memory_get_buffers(mem, gpa) catch unreachable;
        defer gpa.free(bufs);
        for (bufs) |buf| names.append(gpa, std.fmt.allocPrint(gpa, "{s}:{d}", .{ std.mem.span(c.ggml_backend_buffer_name(buf)), c.ggml_backend_buffer_get_size(buf) }) catch unreachable) catch unreachable;
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    note("memory buffers {d}", .{names.items.len});
    for (names.items) |n| note("  buffer {s}", .{n});
}

const gpa = std.heap.c_allocator;

var checks: usize = 0;
var failures: usize = 0;

// -----------------------------------------------------------------------------
// The transcript

var rec: ?*std.ArrayList(u8) = null;
var hash_nodes = true;
var n_nodes_seen: usize = 0;

/// CONTEXT_DIFF_LIVE=1 echoes the port's transcript to stderr as it is
/// written, to find where a crashing run stopped.
var live = false;
var stream_out = false;

fn flushStream() void {
    if (!stream_out) return;
    const r = rec orelse return;
    _ = std.c.write(1, r.items.ptr, r.items.len);
    r.clearRetainingCapacity();
}

fn note(comptime fmt: []const u8, args: anytype) void {
    if (live and std.mem.eql(u8, side, "port")) std.debug.print(fmt ++ "\n", args);
    const r = rec orelse return;
    r.print(gpa, fmt ++ "\n", args) catch unreachable;
    flushStream();
}

fn logCb(level: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const r = rec orelse return;
    _ = level;
    r.appendSlice(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
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

fn nodeCb(t: [*c]c.ggml_tensor, ask: bool, _: ?*anyopaque) callconv(.c) bool {
    // Without hashing, each node's structure is recorded when the scheduler
    // asks, and declining keeps the graph whole.
    if (ask and hash_nodes) return true;
    const node: *c.ggml_tensor = t;
    n_nodes_seen += 1;
    const r = rec orelse return hash_nodes;
    r.print(gpa, "node {s} {s} {s} [{d},{d},{d},{d}] [{d},{d},{d},{d}] p", .{
        std.mem.sliceTo(&node.name, 0),
        c.ggml_op_desc(node),
        c.ggml_type_name(node.type),
        node.ne[0],
        node.ne[1],
        node.ne[2],
        node.ne[3],
        node.nb[0],
        node.nb[1],
        node.nb[2],
        node.nb[3],
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
    if (ask) {
        r.print(gpa, " flags {x}\n", .{node.flags}) catch unreachable;
        flushStream();
        return false;
    }
    // A node without GGML_TENSOR_FLAG_COMPUTE -- an unselected branch of
    // `ggml_build_forward_select`, such as v0.6.0's mixed-batch input when
    // the ubatch is not mixed -- is skipped by every backend (ggml-cpu.c:3137,
    // ggml-metal-ops.cpp:218 @620d2225c): its bytes are the allocator's
    // leftovers on either side. The flags above are compared, so a node one
    // side computes and the other does not still differs.
    if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) {
        r.print(gpa, " flags {x} h -\n", .{node.flags}) catch unreachable;
        return true;
    }
    r.print(gpa, " flags {x} h {x}\n", .{ node.flags, tensorHash(node) }) catch unreachable;
    return true;
}

/// Lines whose figures move with time or with what the process has
/// allocated, and not with what either side computed.
fn normalize(log: []const u8) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, log, '\n');
    var first = true;
    while (lines.next()) |line| {
        // Metal compiles a pipeline the first time anything in the
        // process uses it, which is whichever side ran first.
        if (std.mem.startsWith(u8, line, "ggml_metal_library_compile_pipeline:")) continue;
        if (!first) out.append(gpa, '\n') catch unreachable;
        first = false;
        if (std.mem.find(u8, line, "reserve took ")) |at| {
            out.appendSlice(gpa, line[0 .. at + "reserve took ".len]) catch unreachable;
            out.appendSlice(gpa, "N ms") catch unreachable;
            if (std.mem.find(u8, line, ", sched copies")) |rest| out.appendSlice(gpa, line[rest..]) catch unreachable;
            continue;
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
        if (std.mem.startsWith(u8, line, "ggml_metal_log_allocated_size:")) {
            if (std.mem.findScalar(u8, line, '(')) |open| {
                out.appendSlice(gpa, line[0..open]) catch unreachable;
                out.appendSlice(gpa, "(N)") catch unreachable;
                continue;
            }
        }
        out.appendSlice(gpa, line) catch unreachable;
    }
    const sorted = input_tensor_order.sortRuns(gpa, out.items);
    out.deinit(gpa);
    return sorted;
}

fn firstDiff(name: []const u8, a: []const u8, b: []const u8) bool {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return true;
        checks += 1;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            failures += 1;
            std.debug.print("FAIL {s}: transcript line {d}:\n    ref  '{s}'\n    port '{s}'\n", .{ name, line, la orelse "<end>", lb orelse "<end>" });
            return false;
        }
    }
}

// -----------------------------------------------------------------------------
// The script

const Case = struct {
    name: []const u8,
    edit: *const fn (*c.llama_context_params) void,
    /// The memory API and the interleaved sequences need n_seq_max >= 2.
    multi_seq: bool = false,
    /// Whether the C++ itself runs it under `LLAMA_SPLIT_MODE_TENSOR`.
    /// With the KV cache on the host the meta backend asserts on the
    /// attention's split axes (ggml-backend-meta.cpp:826).
    tensor_ok: bool = true,
    /// Whether the C++ itself runs it on a model without memory -- an
    /// encoder, an embedding or a diffusion model -- whose decode is an
    /// encode: that asserts the batch fits one ubatch
    /// (llama-context.cpp:1426) and that no output limit is below it
    /// (llama-context.cpp:2189).
    memoryless_ok: bool = true,
    /// Turns the KV cache's attention rotation off on the model before the
    /// context is made, through `llama_model_set_attn_rot` (v0.7.0), and
    /// back on after.
    attn_rot_off: bool = false,
};

const prompt_text = "The quick brown fox jumps over the lazy dog. In 1492, Columbus sailed the ocean blue; the capital of France is";

fn argmax(row: [*]const f32, n: usize) i32 {
    var best: usize = 0;
    for (1..n) |i| {
        if (row[i] > row[best]) best = i;
    }
    return @intCast(best);
}

fn batchAdd(b: *c.llama_batch, tok: i32, pos: i32, seqs: []const i32, logits: bool) void {
    const i: usize = @intCast(b.n_tokens);
    b.token[i] = tok;
    b.pos[i] = pos;
    b.n_seq_id[i] = @intCast(seqs.len);
    for (seqs, 0..) |s, k| b.seq_id[i][k] = s;
    b.logits[i] = @intFromBool(logits);
    b.n_tokens += 1;
}

// The speculative drafts, eagle3 and DFlash. `has_encoder` is true for
// them, but their encoder reads the target model's features as embeddings,
// not tokens, so the T5 encode below would hand both sides a token batch for
// an embedding input and crash both -- which compared nothing, and read as
// matching crashes. They are driven the way `common/speculative.cpp` drives
// them instead: an embedding batch of n_embd_inp_enc features encoded first;
// eagle3's decoder batches carry the g embeddings beside the tokens; and a
// DFlash round ends by injecting target features into the cache -- since
// v0.4.0 the raw n_embd_inp_enc-wide ones, which the decoder fuses -- and
// decoding a token over them.
var model_path: [:0]const u8 = "";
const DraftKind = enum { none, eagle3, dflash };
const Draft = struct { kind: DraftKind = .none, n_enc: usize = 0, n_embd: usize = 0 };

fn metaStr(comptime MA: type, model: ?*c.struct_llama_model, key: [*:0]const u8, buf: []u8) []const u8 {
    const n = MA.llama_model_meta_val_str(@ptrCast(model), key, buf.ptr, buf.len);
    if (n < 0) return "";
    return buf[0..@min(@as(usize, @intCast(n)), buf.len - 1)];
}

fn draftOf(comptime MA: type, model: ?*c.struct_llama_model) Draft {
    var buf: [256]u8 = undefined;
    const arch = metaStr(MA, model, "general.architecture", &buf);
    const n_embd: usize = @intCast(MA.llama_model_n_embd(@ptrCast(model)));
    if (std.mem.eql(u8, arch, "eagle3")) {
        const v = metaStr(MA, model, "eagle3.target_hidden_size", &buf);
        const h = std.fmt.parseInt(usize, v, 10) catch 0;
        return .{ .kind = .eagle3, .n_enc = 3 * h, .n_embd = n_embd };
    }
    if (std.mem.eql(u8, arch, "dflash")) {
        // Arrays are not in the model's string metadata, so the target
        // layer count is read from the file.
        const g = c.gguf_init_from_file(model_path.ptr, .{ .no_alloc = true, .ctx = null }) orelse return .{};
        defer c.gguf_free(g);
        const k = c.gguf_find_key(g, "dflash.target_layers");
        if (k < 0) return .{};
        const n: usize = c.gguf_get_arr_n(g, k);
        return .{ .kind = .dflash, .n_enc = n * n_embd, .n_embd = n_embd };
    }
    return .{};
}

/// Deterministic features: small, varied, the same on both sides.
fn fillFeatures(dst: []f32, salt: usize) void {
    for (dst, 0..) |*v, i| {
        const x: f32 = @floatFromInt((i * 2654435761 + salt * 40503) % 2003);
        v.* = (x - 1001.0) / 1001.0;
    }
}

fn draftBatch(n: usize, width: usize, pos0: i32, logits: bool, salt: usize) c.llama_batch {
    var eb = c.llama_batch_init(@intCast(n), @intCast(width), 1);
    eb.n_tokens = @intCast(n);
    fillFeatures(eb.embd[0 .. n * width], salt);
    for (0..n) |i| {
        eb.pos[i] = pos0 + @as(i32, @intCast(i));
        eb.n_seq_id[i] = 1;
        eb.seq_id[i][0] = 0;
        eb.logits[i] = @intFromBool(logits);
    }
    return eb;
}

/// Run the script through `A`, which is `c` (the C++) or `api` (the port).
fn script(comptime A: type, comptime S: type, model: ?*c.struct_llama_model, cs: Case, tokens: []const i32, n_vocab: usize, n_embd: usize) void {
    var params = c.llama_context_default_params();
    params.cb_eval = nodeCb;
    params.cb_eval_user_data = null;
    params.no_perf = true;
    cs.edit(&params);
    if (model_ctx_edit) |e| e(&params);

    // `--target`: a target model's context as `ctx_other`, which a DFlash
    // draft without its own output head reads its head from (#30111).
    const target_model: ?*c.struct_llama_model = if (A == c) target_ref else @ptrCast(target_port);
    var target_ctx: ?*c.struct_llama_context = null;
    if (target_model) |tm| {
        var tp = c.llama_context_default_params();
        tp.n_ctx = 512;
        target_ctx = A.llama_init_from_model(tm, tp);
        note("target context: {}", .{target_ctx != null});
        params.ctx_other = target_ctx;
    }
    defer if (target_ctx) |tc| A.llama_free(tc);

    if (cs.attn_rot_off) {
        if (A == c) _Z24llama_model_set_attn_rotP11llama_modelb(model, false) else harness_root.api.model.llama_model_set_attn_rot(model, false);
    }
    defer if (cs.attn_rot_off) {
        if (A == c) _Z24llama_model_set_attn_rotP11llama_modelb(model, true) else harness_root.api.model.llama_model_set_attn_rot(model, true);
    };

    const ctx = A.llama_init_from_model(model, params) orelse {
        note("init failed", .{});
        return;
    };
    defer A.llama_free(ctx);

    note("n_ctx {d} n_ctx_seq {d} n_batch {d} n_ubatch {d} n_seq_max {d} n_rs_seq {d} pooling {d} threads {d}/{d}", .{
        A.llama_n_ctx(ctx),           A.llama_n_ctx_seq(ctx), A.llama_n_batch(ctx),      A.llama_n_ubatch(ctx),
        A.llama_n_seq_max(ctx),       A.llama_n_rs_seq(ctx),  A.llama_pooling_type(ctx), A.llama_n_threads(ctx),
        A.llama_n_threads_batch(ctx),
    });
    const mem = A.llama_get_memory(ctx);
    note("memory {} can_shift {}", .{ mem != null, A.llama_memory_can_shift(mem) });

    const embeddings = params.embeddings;
    const pooled = embeddings and A.llama_pooling_type(ctx) != c.LLAMA_POOLING_TYPE_NONE;
    const n_batch: usize = A.llama_n_batch(ctx);
    var b = c.llama_batch_init(@intCast(@max(n_batch, tokens.len + 8)), 0, 4);
    defer c.llama_batch_free(b);

    const MAd = if (A == c) c else harness_root.api.model;
    const draft = draftOf(MAd, model);
    var g_embd: []f32 = &.{};
    if (draft.kind == .eagle3) {
        g_embd = gpa.alloc(f32, @as(usize, @intCast(@max(n_batch, tokens.len + 8))) * draft.n_embd) catch unreachable;
        fillFeatures(g_embd, 7);
        b.embd = g_embd.ptr;
    }
    defer {
        b.embd = null;
        if (g_embd.len > 0) gpa.free(g_embd);
    }

    const Out = struct {
        fn row(comptime what: []const u8, p: ?[*]f32, n: usize, i: i32) void {
            if (p) |r| {
                note(what ++ " {d}: {x}", .{ i, hashBytes(@ptrCast(r), n * @sizeOf(f32)) });
            } else {
                note(what ++ " {d}: null", .{i});
            }
        }
    };

    const State = struct {
        fn positions(m: c.llama_memory_t, n_seq: i32) void {
            var s: i32 = 0;
            while (s < n_seq) : (s += 1) note("seq {d}: [{d}, {d}]", .{ s, A.llama_memory_seq_pos_min(m, s), A.llama_memory_seq_pos_max(m, s) });
        }
    };

    const n_seq: i32 = @intCast(A.llama_n_seq_max(ctx));

    // An encoder-decoder model (T5) decodes against what the encoder made
    // of the prompt, and a decode before any encode asserts "llama_encode
    // must be called first" on both sides -- which compared nothing of
    // either graph. So each round encodes the prompt first, as upstream's
    // own test-llama-archs does, and the decoder then reads it.
    const MA = if (A == c) c else harness_root.api.model;
    const enc_dec = draft.kind == .none and MA.llama_model_has_encoder(@ptrCast(model)) and MA.llama_model_has_decoder(@ptrCast(model));

    for (0..2) |round| {
        note("round {d}", .{round});
        // A pooled encode asserts on both sides: the encoder graph has no
        // t_embd to pool.
        if (draft.kind != .none and !pooled) {
            const n_enc_tok = @min(tokens.len, @as(usize, A.llama_n_ubatch(ctx)));
            const eb = draftBatch(n_enc_tok, draft.n_enc, 0, true, round);
            defer c.llama_batch_free(eb);
            note("draft encode {d}x{d}: {d}", .{ n_enc_tok, draft.n_enc, A.llama_encode(ctx, eb) });
        }
        if (enc_dec) {
            b.n_tokens = 0;
            for (tokens, 0..) |t, i| batchAdd(&b, t, @intCast(i), &.{0}, true);
            const er = A.llama_encode(ctx, b);
            note("encode: {d}", .{er});
            if (er != 0) return;
            if (A.llama_get_embeddings(ctx)) |e| note("encoder embd: {x}", .{hashBytes(@ptrCast(e), tokens.len * n_embd * @sizeOf(f32))});
        }
        // The prompt, logits on the last token only (every token when
        // computing embeddings).
        b.n_tokens = 0;
        for (tokens, 0..) |t, i| batchAdd(&b, t, @intCast(i), &.{0}, i + 1 == tokens.len);
        var ret = A.llama_decode(ctx, b);
        note("decode prompt: {d}", .{ret});
        if (ret != 0) return;
        // v0.7.0: a pooled embeddings context has no LM head, so no
        // logits buffer (#30255, #30445)
        note("logits buffer: {}", .{A.llama_get_logits(ctx) != null});
        // v0.7.0's `llama_memory_get_buffers`, on a memory holding the
        // prompt: an empty one reports none.
        if (round == 0) noteMemoryBuffers(A, mem);

        var next: i32 = 0;
        if (embeddings) {
            if (pooled) {
                Out.row("embd_seq", A.llama_get_embeddings_seq(ctx, 0), n_embd, 0);
            } else {
                var i: i32 = 0;
                while (i < tokens.len) : (i += 1) Out.row("embd", A.llama_get_embeddings_ith(ctx, i), n_embd, i);
            }
            next = tokens[tokens.len - 1];
        } else {
            const last = A.llama_get_logits_ith(ctx, -1);
            Out.row("logits", last, n_vocab, -1);
            if (last) |r| next = argmax(r, n_vocab);
            Out.row("logits", A.llama_get_logits_ith(ctx, @intCast(tokens.len - 1)), n_vocab, @intCast(tokens.len - 1));
            // Rows that were not output, and past the batch.
            Out.row("logits", A.llama_get_logits_ith(ctx, 0), n_vocab, 0);
            Out.row("logits", A.llama_get_logits_ith(ctx, @intCast(tokens.len + 5)), n_vocab, @intCast(tokens.len + 5));
            Out.row("logits", A.llama_get_logits_ith(ctx, -2), n_vocab, -2);
        }
        State.positions(mem, n_seq);

        // The state after the prompt, every way it can be taken.
        var saved: std.ArrayList(u8) = .empty;
        defer saved.deinit(gpa);
        const first_next = next;
        if (round == 0) saveState(S, ctx, tokens, &saved);

        // Greedy steps, one token at a time.
        var pos: i32 = @intCast(tokens.len);
        for (0..3) |_| {
            b.n_tokens = 0;
            batchAdd(&b, next, pos, &.{0}, true);
            ret = A.llama_decode(ctx, b);
            note("decode step {d}: {d}", .{ pos, ret });
            if (ret != 0) return;
            if (embeddings) {
                Out.row("embd", A.llama_get_embeddings_ith(ctx, 0), n_embd, 0);
            } else {
                const r = A.llama_get_logits_ith(ctx, 0);
                Out.row("logits", r, n_vocab, 0);
                if (r) |row| next = argmax(row, n_vocab);
                note("token {d}", .{next});
            }
            pos += 1;
        }

        // A batch with every output.
        b.n_tokens = 0;
        for (0..5) |k| batchAdd(&b, tokens[k], pos + @as(i32, @intCast(k)), &.{0}, true);
        ret = A.llama_decode(ctx, b);
        note("decode all-out: {d}", .{ret});
        if (ret != 0) return;
        pos += 5;
        if (!embeddings) {
            for (0..5) |k| Out.row("logits", A.llama_get_logits_ith(ctx, @intCast(k)), n_vocab, @intCast(k));
            if (A.llama_get_logits(ctx)) |all| note("logits all: {x}", .{hashBytes(@ptrCast(all), 5 * n_vocab * @sizeOf(f32))});
        }
        State.positions(mem, n_seq);

        // Since v0.6.0: the extended batch through `llama_process` -- tokens
        // with decision-order spans, then tokens and embeddings mixed, which
        // an architecture that picks weights by the input kind refuses.
        if (draft.kind == .none) {
            const AB = if (A == c) c else harness_root.api.batch;
            note("causal_attn {}", .{A.llama_get_causal_attn(ctx)});
            const eb = AB.llama_batch_ext_init(ctx) orelse {
                note("batch_ext_init failed", .{});
                return;
            };
            defer AB.llama_batch_ext_free(eb);
            const n_pos: usize = if (MA.llama_model_rope_type(@ptrCast(model)) == c.LLAMA_ROPE_TYPE_MROPE or MA.llama_model_rope_type(@ptrCast(model)) == c.LLAMA_ROPE_TYPE_IMROPE) 4 else 1;
            const orders = [_]i32{ 1, 1, 0, 4, 4, 0, 4 };
            for (0..7) |k| {
                const idx = AB.llama_batch_ext_add_token(eb, 0, tokens[(k + 2) % tokens.len]);
                const p = [_]i32{ pos + @as(i32, @intCast(k)), 0, 0, 0 };
                _ = AB.llama_batch_ext_set_pos(eb, idx, &p);
                // The question's last token and every option token are
                // outputs: vllm-decision2's head (v0.7.0) reads the output
                // row of each span's last token, so the question's must be
                // one; the separator at k = 2 is not, which keeps the batch
                // at five outputs for "n_outputs_max 5".
                _ = AB.llama_batch_ext_set_output_logits(eb, idx, k != 0 and k != 2);
                if (A == c) {
                    _ = ref_ext_set_decision_order(eb, idx, orders[k]);
                } else {
                    _ = harness_root.api.batch.llama_batch_ext_set_decision_order(eb, idx, @fromBackingInt(@intCast(@as(c_uint, @intCast(orders[k])))));
                }
            }
            ret = A.llama_process(ctx, c.LLAMA_PROCESS_TYPE_DECODE, eb);
            note("process ext: {d}", .{ret});
            if (ret == 0) {
                pos += 7;
                if (embeddings) {
                    if (!pooled) for (0..7) |k| Out.row("embd", A.llama_get_embeddings_ith(ctx, @intCast(k)), n_embd, @intCast(k));
                } else {
                    for (0..7) |k| Out.row("logits", A.llama_get_logits_ith(ctx, @intCast(k)), n_vocab, @intCast(k));
                }
            }
            State.positions(mem, n_seq);

            AB.llama_batch_ext_clear(eb);
            const n_embd_inp: usize = @intCast(MA.llama_model_n_embd_inp(@ptrCast(model)));
            const rows = gpa.alloc(f32, 2 * n_embd_inp) catch unreachable;
            defer gpa.free(rows);
            fillFeatures(rows, 31 + round);
            for (0..5) |k| {
                const as_embd = k == 1 or k == 2;
                const idx = if (as_embd)
                    AB.llama_batch_ext_add_embd(eb, 0, .{ .data = rows.ptr + (k - 1) * n_embd_inp, .n_rows = 1, .n_embd = n_embd_inp })
                else
                    AB.llama_batch_ext_add_token(eb, 0, tokens[k]);
                const pk = pos + @as(i32, @intCast(k));
                // distinct y and x, so a section order (v0.7.0) that swaps
                // them shows
                const p = if (n_pos > 1 and as_embd) [_]i32{ pk, pk + 1, pk + 2, 0 } else [_]i32{ pk, 0, 0, 0 };
                _ = AB.llama_batch_ext_set_pos(eb, idx, &p);
                _ = AB.llama_batch_ext_set_output_embd(eb, idx, true);
            }
            ret = A.llama_process(ctx, c.LLAMA_PROCESS_TYPE_DECODE, eb);
            note("process mixed: {d}", .{ret});
            if (ret == 0) {
                pos += 5;
                if (embeddings) {
                    if (!pooled) for (0..5) |k| Out.row("embd", A.llama_get_embeddings_ith(ctx, @intCast(k)), n_embd, @intCast(k));
                } else {
                    for (0..5) |k| Out.row("logits", A.llama_get_logits_ith(ctx, @intCast(k)), n_vocab, @intCast(k));
                }
            }
            note("process type 7: {d}", .{A.llama_process(ctx, 7, eb)});
            State.positions(mem, n_seq);
            // The cells the mixed batch wrote -- a row's token id among them,
            // which an embedding row must not carry.
            {
                const ssz = S.llama_state_seq_get_size(ctx, 0);
                const sbuf = gpa.alloc(u8, @max(ssz, 1)) catch unreachable;
                defer gpa.free(sbuf);
                const sn = S.llama_state_seq_get_data(ctx, sbuf.ptr, ssz, 0);
                note("state after mixed: {d} {x}", .{ sn, hashBytes(sbuf.ptr, sn) });
            }
        }

        if (round == 0) {
            restoreState(S, ctx, tokens, saved.items);
            State.positions(mem, n_seq);
            // The first step again, on the restored state.
            b.n_tokens = 0;
            batchAdd(&b, first_next, @intCast(tokens.len), &.{0}, true);
            note("decode after restore: {d}", .{A.llama_decode(ctx, b)});
            if (!embeddings) Out.row("logits", A.llama_get_logits_ith(ctx, 0), n_vocab, 0);
            State.positions(mem, n_seq);
            pos = @intCast(tokens.len + 1);
        }

        // Rolling back the last tokens: refused by a recurrent memory
        // without n_rs_seq, which must leave it unchanged.
        const rm_tail = A.llama_memory_seq_rm(mem, 0, pos - 2, -1);
        note("seq_rm tail: {}", .{rm_tail});
        State.positions(mem, n_seq);
        // A memory that removed the tail continues after what it kept, or
        // every later batch on sequence 0 is refused for a position gap
        // (a plain KV cache did, and the interleaved and copy steps below
        // had never run on one).
        if (rm_tail) pos = A.llama_memory_seq_pos_max(mem, 0) + 1;

        if (cs.multi_seq and n_seq >= 2) {
            // Two sequences in one batch, interleaved, and a token shared
            // by both.
            // Sequence 1 starts fresh beside sequence 0 (or 2, where there
            // is one), each at its own positions.
            const other: i32 = if (n_seq >= 3) 2 else 0;
            var other_pos: i32 = if (other == 0) pos else 0;
            b.n_tokens = 0;
            // The lower sequence id first: `split_equal` takes only
            // increasing ids into one ubatch, so with sequence 1 first it
            // splits by sequence anyway and cannot be told from
            // `split_seq` (v0.7.0's `embd_all` split; an injection that
            // dropped it escaped).
            const one_first = other > 1;
            for (0..6) |k| {
                const first = (k % 2 == 0) == one_first;
                // Outputs from the second token: the memory splits the
                // batch by sequence, so they come back out of order.
                if (first) {
                    batchAdd(&b, tokens[k], @intCast(k / 2), &.{1}, k >= 1);
                } else {
                    batchAdd(&b, tokens[k], other_pos, &.{other}, k >= 1);
                    other_pos += 1;
                }
            }
            ret = A.llama_decode(ctx, b);
            note("decode interleaved: {d}", .{ret});
            if (ret == 0) {
                for (0..6) |k| Out.row("logits", A.llama_get_logits_ith(ctx, @intCast(k)), n_vocab, @intCast(k));
            }
            if (other == 0) pos = other_pos;
            State.positions(mem, n_seq);

            seqStateOnDevice(A, S, ctx, mem, 1, tokens);
            State.positions(mem, n_seq);
            // A token on the restored sequence.
            b.n_tokens = 0;
            batchAdd(&b, tokens[6], 3, &.{1}, true);
            ret = A.llama_decode(ctx, b);
            note("decode on restored seq: {d}", .{ret});
            if (ret == 0) Out.row("logits", A.llama_get_logits_ith(ctx, 0), n_vocab, 0);

            A.llama_memory_seq_cp(mem, 0, 1, -1, -1);
            note("seq_cp 0 -> 1", .{});
            State.positions(mem, n_seq);
            // v0.7.0: a recurrent cell shared by several sequences cannot
            // roll back for one of them (with n_rs_seq it otherwise could).
            note("seq_rm 0 tail while shared: {}", .{A.llama_memory_seq_rm(mem, 0, pos - 1, -1)});
            State.positions(mem, n_seq);
            b.n_tokens = 0;
            batchAdd(&b, tokens[7], pos, &.{1}, true);
            ret = A.llama_decode(ctx, b);
            note("decode after cp: {d}", .{ret});
            if (ret == 0) Out.row("logits", A.llama_get_logits_ith(ctx, 0), n_vocab, 0);

            note("seq_rm 1 all: {}", .{A.llama_memory_seq_rm(mem, 1, -1, -1)});
            A.llama_memory_seq_keep(mem, 0);
            note("seq_keep 0", .{});
            State.positions(mem, n_seq);
            // Shifting asserts on a model that cannot shift, M-RoPE included.
            if (A.llama_memory_can_shift(mem)) {
                A.llama_memory_seq_add(mem, 0, 0, -1, 0);
                State.positions(mem, n_seq);
            }

            // A sequence past n_seq_max, and a bad token: rejected.
            b.n_tokens = 0;
            batchAdd(&b, tokens[0], 0, &.{n_seq + 3}, true);
            note("decode bad seq: {d}", .{A.llama_decode(ctx, b)});
        }

        b.n_tokens = 0;
        batchAdd(&b, @intCast(n_vocab + 10), pos, &.{0}, true);
        note("decode bad token: {d}", .{A.llama_decode(ctx, b)});
        b.n_tokens = 0;
        batchAdd(&b, tokens[0], 1, &.{0}, true);
        note("decode position gap: {d}", .{A.llama_decode(ctx, b)});

        if (draft.kind == .dflash) {
            const p0: i32 = A.llama_memory_seq_pos_max(mem, 0) + 1;
            // Since v0.4.0 the injection carries the raw target features,
            // n_embd_inp_enc wide, and the decoder fuses them itself
            // (common/speculative.cpp, `batch_inject`).
            const ib = draftBatch(4, draft.n_enc, p0, false, 11 + round);
            defer c.llama_batch_free(ib);
            note("draft inject: {d}", .{A.llama_decode(ctx, ib)});
            State.positions(mem, n_seq);
            b.n_tokens = 0;
            batchAdd(&b, tokens[3], p0 + 4, &.{0}, true);
            const ret_i = A.llama_decode(ctx, b);
            note("decode over injected: {d}", .{ret_i});
            if (ret_i == 0) Out.row("logits", A.llama_get_logits_ith(ctx, 0), n_vocab, 0);

            // v0.7.0's LiliCorr head builds only for a token ubatch with
            // state embeddings of `block_size` tokens per sequence: one
            // drafting block as common/speculative.cpp:1297 makes it, every
            // entry with a state row, NextN embeddings on, outputs off.
            var vb: [64]u8 = undefined;
            if (metaStr(MA, model, "lilicorr.candidate_topk", &vb).len != 0 and ret_i == 0) {
                const block_size = std.fmt.parseInt(usize, metaStr(MA, model, "dflash.block_size", &vb), 10) catch 0;
                const n_state: usize = @intCast(MA.llama_model_n_embd_inp(@ptrCast(model)));
                const AB = if (A == c) c else harness_root.api.batch;
                if (A == c) _Z26llama_set_embeddings_nextnP13llama_contextbb(ctx, true, false) else harness_root.api.context.llama_set_embeddings_nextn(ctx, true, false);
                const state = gpa.alloc(f32, @max(block_size, 1) * n_state) catch unreachable;
                defer gpa.free(state);
                fillFeatures(state, 17 + round);
                if (AB.llama_batch_ext_init(ctx)) |lb| {
                    defer AB.llama_batch_ext_free(lb);
                    const q0: i32 = A.llama_memory_seq_pos_max(mem, 0) + 1;
                    var ok = true;
                    for (0..block_size) |k| {
                        const idx = AB.llama_batch_ext_add_token(lb, 0, if (k == 0) tokens[3] else tokens[(k + 4) % tokens.len]);
                        const p = [_]i32{ q0 + @as(i32, @intCast(k)), 0, 0, 0 };
                        _ = AB.llama_batch_ext_set_pos(lb, idx, &p);
                        _ = AB.llama_batch_ext_set_output_logits(lb, idx, false);
                        ok = ok and AB.llama_batch_ext_set_embd_state(lb, idx, .{ .data = state.ptr + k * n_state, .n_rows = 1, .n_embd = n_state });
                    }
                    note("lilicorr block {d}: state set {}", .{ block_size, ok });
                    const rc = A.llama_process(ctx, c.LLAMA_PROCESS_TYPE_DECODE, lb);
                    note("lilicorr process: {d}", .{rc});
                    if (rc == 0) {
                        const e = if (A == c) _Z26llama_get_embeddings_nextnP13llama_context(ctx) else harness_root.api.context.llama_get_embeddings_nextn(ctx);
                        if (e) |r| note("lilicorr lattice: {x}", .{hashBytes(@ptrCast(r), block_size * n_embd * @sizeOf(f32))}) else note("lilicorr lattice: null", .{});
                    }
                }
                if (A == c) _Z26llama_set_embeddings_nextnP13llama_contextbb(ctx, false, true) else harness_root.api.context.llama_set_embeddings_nextn(ctx, false, true);
                State.positions(mem, n_seq);
            }
        }

        // `check_causal_attn_toggle` (tests/test-llama-archs.cpp:620
        // @620d2225c): non-causal decodes of n_ubatch/2 and n_ubatch tokens,
        // a flip that must not need a scheduler reserve.
        if (round == 1 and draft.kind == .none) {
            const AB = if (A == c) c else harness_root.api.batch;
            var cpos: i32 = A.llama_memory_seq_pos_max(mem, 0) + 1;
            // n_ubatch, or what the context still has room for
            const room = @as(i32, @intCast(A.llama_n_ctx(ctx))) - cpos;
            const n_ub: i32 = @min(@as(i32, @intCast(A.llama_n_ubatch(ctx))), @divTrunc(room * 2, 3));
            if (n_ub >= 2) {
                A.llama_set_causal_attn(ctx, false);
                note("causal_attn {}", .{A.llama_get_causal_attn(ctx)});
                for ([_]i32{ @divTrunc(n_ub, 2), n_ub }) |nt| {
                    const tb = AB.llama_batch_ext_init(ctx) orelse break;
                    defer AB.llama_batch_ext_free(tb);
                    for (0..@intCast(nt)) |k| {
                        const idx = AB.llama_batch_ext_add_token(tb, 0, tokens[k % tokens.len]);
                        const p = [_]i32{ cpos, 0, 0, 0 };
                        cpos += 1;
                        _ = AB.llama_batch_ext_set_pos(tb, idx, &p);
                        // the last token only: "n_outputs_max 5" caps them
                        _ = AB.llama_batch_ext_set_output_logits(tb, idx, @as(i32, @intCast(k)) + 1 == nt);
                    }
                    const rc = A.llama_process(ctx, c.LLAMA_PROCESS_TYPE_DECODE, tb);
                    note("process non-causal {d}: {d}", .{ nt, rc });
                    if (rc != 0) break;
                    if (embeddings) {
                        if (!pooled) Out.row("embd", A.llama_get_embeddings_ith(ctx, nt - 1), n_embd, nt - 1);
                    } else {
                        Out.row("logits", A.llama_get_logits_ith(ctx, nt - 1), n_vocab, nt - 1);
                    }
                }
                A.llama_set_causal_attn(ctx, true);
                State.positions(mem, n_seq);
            }
        }

        // Since v0.6.0 a failed sequence restore zeroes the K/V and
        // recurrent rows it wrote -- attention still reads free cells -- and
        // a hybrid memory undoes its attention half when the recurrent half
        // fails: a restore cut in the middle and one cut at its tail, each
        // followed by a decode over the cleared sequence.
        if (draft.kind == .none and !enc_dec) {
            const ssz = S.llama_state_seq_get_size(ctx, 0);
            const sbuf = gpa.alloc(u8, @max(ssz, 1)) catch unreachable;
            defer gpa.free(sbuf);
            const sn = S.llama_state_seq_get_data(ctx, sbuf.ptr, ssz, 0);
            for ([_]usize{ sn / 2, sn -| 16 }) |cut| {
                note("seq set cut at {d}/{d}: {d}", .{ cut, sn, S.llama_state_seq_set_data(ctx, sbuf.ptr, cut, 0) });
                State.positions(mem, n_seq);
                b.n_tokens = 0;
                batchAdd(&b, tokens[1], A.llama_memory_seq_pos_max(mem, 0) + 1, &.{0}, true);
                const rc = A.llama_decode(ctx, b);
                note("decode after cut restore: {d}", .{rc});
                if (rc == 0 and !embeddings) Out.row("logits", A.llama_get_logits_ith(ctx, 0), n_vocab, 0);
                note("seq set full: {d}", .{S.llama_state_seq_set_data(ctx, sbuf.ptr, sn, 0)});
            }
        }

        A.llama_memory_clear(mem, true);
        note("cleared", .{});
        State.positions(mem, n_seq);
    }

    // Since v0.7.0 a pooled context has no LM head while it computes
    // embeddings, so toggling them off changes the graph and must reserve
    // again and grow the output buffer for logits.
    if (pooled and draft.kind == .none and !enc_dec) {
        A.llama_set_embeddings(ctx, false);
        b.n_tokens = 0;
        batchAdd(&b, tokens[2], A.llama_memory_seq_pos_max(mem, 0) + 1, &.{0}, true);
        var rc = A.llama_decode(ctx, b);
        note("decode, embeddings off: {d}", .{rc});
        if (rc == 0) Out.row("logits", A.llama_get_logits_ith(ctx, -1), n_vocab, -1);
        A.llama_set_embeddings(ctx, true);
        b.n_tokens = 0;
        batchAdd(&b, tokens[3], A.llama_memory_seq_pos_max(mem, 0) + 1, &.{0}, true);
        rc = A.llama_decode(ctx, b);
        note("decode, embeddings on: {d}", .{rc});
        if (rc == 0) Out.row("embd_seq", A.llama_get_embeddings_seq(ctx, 0), n_embd, 0);
    }

    // `llama_memory_get_buffers` (v0.7.0): the buffers a state write of the
    // memory touches.
    noteMemoryBuffers(A, mem);

    const perf = A.llama_perf_context(ctx);
    note("perf n_p_eval {d} n_eval {d} n_reused {d} n_reallocs_same {d} n_reallocs_diff {d}", .{ perf.n_p_eval, perf.n_eval, perf.n_reused, perf.n_reallocs_same, perf.n_reallocs_diff });
}

/// `--target`'s model on each side.
var target_ref: ?*c.struct_llama_model = null;
var target_port: ?*M.Model = null;

/// The model configuration's own context edit, applied after the case's:
/// the MoE cache needs both an override and a context parameter.
var model_ctx_edit: ?*const fn (*c.llama_context_params) void = null;

// -----------------------------------------------------------------------------
// State

/// Which side is running, for its files.
var side: []const u8 = "ref";
var tmp_dir: []const u8 = ".";

fn sidePath(buf: []u8, what: []const u8) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{s}.{s}", .{ tmp_dir, side, what }, 0) catch unreachable;
}

fn fileHash(path: [:0]const u8) u64 {
    const f = std.c.fopen(path.ptr, "rb") orelse return 0;
    defer _ = std.c.fclose(f);
    var h = std.hash.Wyhash.init(0);
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.fread(&buf, 1, buf.len, f);
        if (n == 0) break;
        h.update(buf[0..n]);
    }
    return h.final();
}

fn saveState(comptime S: type, ctx: anytype, tokens: []const i32, saved: *std.ArrayList(u8)) void {
    const sz = S.llama_state_get_size(ctx);
    note("state size {d}", .{sz});
    saved.resize(gpa, sz) catch unreachable;
    const n = S.llama_state_get_data(ctx, saved.items.ptr, sz);
    note("state get {d} h {x}", .{ n, hashBytes(saved.items.ptr, n) });
    var small: [64]u8 = undefined;
    note("state get into 64 bytes: {d}", .{S.llama_state_get_data(ctx, &small, small.len)});

    var seq: std.ArrayList(u8) = .empty;
    defer seq.deinit(gpa);
    const ssz = S.llama_state_seq_get_size(ctx, 0);
    seq.resize(gpa, ssz) catch unreachable;
    const sn = S.llama_state_seq_get_data(ctx, seq.items.ptr, ssz, 0);
    note("seq 0 state size {d} get {d} h {x}", .{ ssz, sn, hashBytes(seq.items.ptr, sn) });
    note("seq 3 state size {d}", .{S.llama_state_seq_get_size(ctx, 3)});
    const psz = S.llama_state_seq_get_size_ext(ctx, 0, c.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    note("seq 0 partial state size {d}", .{psz});

    var pb: [512]u8 = undefined;
    const session = sidePath(&pb, "session");
    note("save file: {}", .{S.llama_state_save_file(ctx, session.ptr, tokens.ptr, tokens.len)});
    note("session file h {x}", .{fileHash(session)});
    var pb2: [512]u8 = undefined;
    const seqfile = sidePath(&pb2, "seq");
    note("seq save file: {d}", .{S.llama_state_seq_save_file(ctx, seqfile.ptr, 0, tokens.ptr, tokens.len)});
    note("seq file h {x}", .{fileHash(seqfile)});
    var pb3: [512]u8 = undefined;
    note("save to a missing directory: {}", .{S.llama_state_save_file(ctx, std.fmt.bufPrintSentinel(&pb3, "{s}/no/such/dir/session", .{tmp_dir}, 0) catch unreachable, tokens.ptr, tokens.len)});
}

fn restoreState(comptime S: type, ctx: anytype, tokens: []const i32, saved: []const u8) void {
    // A truncated buffer reads what fits, and the tensors it queued are
    // still written.
    note("state set truncated: {d}", .{S.llama_state_set_data(ctx, saved.ptr, saved.len / 2)});
    var bad: std.ArrayList(u8) = .empty;
    defer bad.deinit(gpa);
    bad.appendSlice(gpa, saved) catch unreachable;
    bad.items[4] ^= 0x20;
    note("state set, wrong arch: {d}", .{S.llama_state_set_data(ctx, bad.items.ptr, bad.items.len)});

    var pb: [512]u8 = undefined;
    const session = sidePath(&pb, "session");
    var out: [256]i32 = undefined;
    var n_out: usize = 0;
    note("load file, small capacity: {}", .{S.llama_state_load_file(ctx, session.ptr, &out, 3, &n_out)});
    var pbm: [512]u8 = undefined;
    note("load missing file: {}", .{S.llama_state_load_file(ctx, std.fmt.bufPrintSentinel(&pbm, "{s}/missing", .{tmp_dir}, 0) catch unreachable, &out, out.len, &n_out)});
    n_out = 0;
    note("load file: {} n {d} tokens {x}", .{ S.llama_state_load_file(ctx, session.ptr, &out, out.len, &n_out), n_out, hashBytes(@ptrCast(&out), n_out * 4) });
    _ = tokens;

    var pb2: [512]u8 = undefined;
    const seqfile = sidePath(&pb2, "seq");
    n_out = 0;
    note("seq load file, count only: {d} n {d}", .{ S.llama_state_seq_load_file(ctx, seqfile.ptr, 0, null, 0, &n_out), n_out });
    note("seq load file into a session: {d}", .{S.llama_state_seq_load_file(ctx, session.ptr, 0, &out, out.len, &n_out)});
    note("seq load file: {d} n {d}", .{ S.llama_state_seq_load_file(ctx, seqfile.ptr, 0, &out, out.len, &n_out), n_out });

    note("state set: {d}", .{S.llama_state_set_data(ctx, saved.ptr, saved.len)});
}

/// One sequence copied on the device, then restored from it.
fn seqStateOnDevice(comptime A: type, comptime S: type, ctx: anytype, mem: c.llama_memory_t, seq_id: i32, tokens: []const i32) void {
    const flags = c.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    const sz = S.llama_state_seq_get_size_ext(ctx, seq_id, flags);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    buf.resize(gpa, sz) catch unreachable;
    const n = S.llama_state_seq_get_data_ext(ctx, buf.items.ptr, sz, seq_id, flags);
    note("seq {d} on-device size {d} get {d} h {x}", .{ seq_id, sz, n, hashBytes(buf.items.ptr, n) });
    // Twice: the second reuses the buffers the first allocated.
    const n2 = S.llama_state_seq_get_data_ext(ctx, buf.items.ptr, sz, seq_id, flags);
    note("seq {d} on-device again {d}", .{ seq_id, n2 });
    // Removed, overwritten by another decode, removed again, then
    // restored from the device copy: a restore that copies nothing shows.
    note("seq {d} rm: {}", .{ seq_id, A.llama_memory_seq_rm(mem, seq_id, -1, -1) });
    {
        var ob = c.llama_batch_init(8, 0, 1);
        defer c.llama_batch_free(ob);
        for (0..4) |k| batchAdd(&ob, tokens[tokens.len - 1 - k], @intCast(k), &.{seq_id}, k == 3);
        note("seq {d} overwrite: {d}", .{ seq_id, A.llama_decode(ctx, ob) });
        note("seq {d} rm again: {}", .{ seq_id, A.llama_memory_seq_rm(mem, seq_id, -1, -1) });
    }
    note("seq {d} after rm: [{d}, {d}]", .{ seq_id, A.llama_memory_seq_pos_min(mem, seq_id), A.llama_memory_seq_pos_max(mem, seq_id) });
    note("seq {d} on-device set {d}", .{ seq_id, S.llama_state_seq_set_data_ext(ctx, buf.items.ptr, n, seq_id, flags) });
    note("seq {d} restored: [{d}, {d}]", .{ seq_id, A.llama_memory_seq_pos_min(mem, seq_id), A.llama_memory_seq_pos_max(mem, seq_id) });
    // Host flags on the same metadata.
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(gpa);
    const hsz = S.llama_state_seq_get_size(ctx, seq_id);
    host.resize(gpa, hsz) catch unreachable;
    const hn = S.llama_state_seq_get_data(ctx, host.items.ptr, hsz, seq_id);
    note("seq {d} host get {d} h {x}", .{ seq_id, hn, hashBytes(host.items.ptr, hn) });
    host.items[0] ^= 1;
    note("seq {d} set, wrong magic: {d}", .{ seq_id, S.llama_state_seq_set_data(ctx, host.items.ptr, hn, seq_id) });
    host.items[0] ^= 1;
    _ = A.llama_memory_seq_rm(mem, seq_id, -1, -1);
    note("seq {d} set: {d}", .{ seq_id, S.llama_state_seq_set_data(ctx, host.items.ptr, hn, seq_id) });
}

// -----------------------------------------------------------------------------
// The cases

const Edits = struct {
    fn none(_: *c.llama_context_params) void {}
    fn faOff(p: *c.llama_context_params) void {
        p.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    fn faOn(p: *c.llama_context_params) void {
        p.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    fn smallUbatch(p: *c.llama_context_params) void {
        p.n_batch = 64;
        p.n_ubatch = 7;
    }
    fn twoSeqs(p: *c.llama_context_params) void {
        p.n_seq_max = 2;
        p.n_ctx = 1024;
    }
    /// v0.7.0: every token an output with a stream per sequence, which
    /// splits by sequence (`init_batch`'s `embd_all`).
    fn twoSeqsEmbd(p: *c.llama_context_params) void {
        twoSeqs(p);
        p.embeddings = true;
        p.pooling_type = c.LLAMA_POOLING_TYPE_NONE;
    }
    fn threeSeqsUnified(p: *c.llama_context_params) void {
        p.n_seq_max = 3;
        p.kv_unified = true;
    }
    fn rsSeq(p: *c.llama_context_params) void {
        p.n_rs_seq = 4;
        p.n_seq_max = 2;
    }
    fn embd(p: *c.llama_context_params) void {
        p.embeddings = true;
        p.pooling_type = c.LLAMA_POOLING_TYPE_NONE;
    }
    fn embdMean(p: *c.llama_context_params) void {
        p.embeddings = true;
        p.pooling_type = c.LLAMA_POOLING_TYPE_MEAN;
    }
    fn embdLast(p: *c.llama_context_params) void {
        p.embeddings = true;
        p.pooling_type = c.LLAMA_POOLING_TYPE_LAST;
    }
    fn quantKv(p: *c.llama_context_params) void {
        p.type_k = c.GGML_TYPE_Q8_0;
        p.type_v = c.GGML_TYPE_Q8_0;
    }
    fn quantVNoFa(p: *c.llama_context_params) void {
        p.type_v = c.GGML_TYPE_Q4_0;
        p.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    fn noOffloadKqv(p: *c.llama_context_params) void {
        p.offload_kqv = false;
    }
    fn noOpOffload(p: *c.llama_context_params) void {
        p.op_offload = false;
    }
    fn oneThread(p: *c.llama_context_params) void {
        p.n_threads = 1;
        p.n_threads_batch = 1;
    }
    /// n_batch is clamped to n_ctx before n_ctx pads up to 256.
    fn tinyCtx(p: *c.llama_context_params) void {
        p.n_ctx = 40;
        p.n_batch = 4096;
        p.n_ubatch = 32;
    }
    fn ropeOverride(p: *c.llama_context_params) void {
        p.rope_freq_base = 20000.0;
        p.rope_freq_scale = 0.5;
        p.rope_scaling_type = c.LLAMA_ROPE_SCALING_TYPE_YARN;
        p.yarn_orig_ctx = 2048;
    }
    fn nonCausal(p: *c.llama_context_params) void {
        p.attention_type = c.LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    }
    fn outputsMax(p: *c.llama_context_params) void {
        // The script's widest batch outputs five.
        p.n_outputs_max = 5;
    }
    fn badBatch(p: *c.llama_context_params) void {
        p.n_batch = 0;
        p.n_ubatch = 0;
    }
    fn tooManySeqs(p: *c.llama_context_params) void {
        p.n_seq_max = 300;
    }
};

const cases = [_]Case{
    .{ .name = "default", .edit = Edits.none },
    .{ .name = "flash attention off", .edit = Edits.faOff },
    .{ .name = "flash attention on", .edit = Edits.faOn },
    .{ .name = "n_ubatch 7", .edit = Edits.smallUbatch, .memoryless_ok = false },
    .{ .name = "two sequences", .edit = Edits.twoSeqs, .multi_seq = true },
    .{ .name = "two sequences, embeddings", .edit = Edits.twoSeqsEmbd, .multi_seq = true },
    .{ .name = "three sequences, unified", .edit = Edits.threeSeqsUnified, .multi_seq = true },
    .{ .name = "n_rs_seq 4", .edit = Edits.rsSeq, .multi_seq = true },
    .{ .name = "embeddings", .edit = Edits.embd },
    .{ .name = "embeddings, mean", .edit = Edits.embdMean },
    .{ .name = "embeddings, last", .edit = Edits.embdLast },
    .{ .name = "q8_0 cache", .edit = Edits.quantKv },
    .{ .name = "q8_0 cache, attn rot off", .edit = Edits.quantKv, .attn_rot_off = true },
    .{ .name = "q4_0 V without FA", .edit = Edits.quantVNoFa },
    .{ .name = "no KQV offload", .edit = Edits.noOffloadKqv, .tensor_ok = false },
    .{ .name = "no op offload", .edit = Edits.noOpOffload },
    .{ .name = "one thread", .edit = Edits.oneThread },
    .{ .name = "n_ctx 40", .edit = Edits.tinyCtx },
    .{ .name = "rope override, yarn", .edit = Edits.ropeOverride },
    .{ .name = "non-causal", .edit = Edits.nonCausal },
    .{ .name = "n_outputs_max 5", .edit = Edits.outputsMax, .memoryless_ok = false },
    .{ .name = "n_batch 0", .edit = Edits.badBatch },
    .{ .name = "n_seq_max 300", .edit = Edits.tooManySeqs },
};

const ModelCfg = struct {
    name: []const u8,
    edit: *const fn (*c.llama_model_params) void,
    /// Context parameters every case of this configuration gets, after the
    /// case's own.
    ctx_edit: ?*const fn (*c.llama_context_params) void = null,
    /// Only for an explicit `--model`: a configuration the model at hand
    /// may not support (the MoE cache needs a MoE model).
    explicit: bool = false,
    /// Whether `cb_eval` asks for every node. The meta device cannot
    /// compute a graph the scheduler has cut at every node -- the C++
    /// asserts `i_start == cgraph->n_nodes` -- so under tensor
    /// parallelism only what the graph outputs is compared.
    nodes: bool = true,
};

const ModelEdits = struct {
    fn gpu(_: *c.llama_model_params) void {}
    fn cpu(p: *c.llama_model_params) void {
        p.n_gpu_layers = 0;
    }
    fn partial(p: *c.llama_model_params) void {
        p.n_gpu_layers = 10;
    }
    var overrides: [2]c.llama_model_tensor_buft_override = undefined;
    /// The norm weights in host memory: the case where `graph_get_cb`'s
    /// pinning of a layer's norm to the layer's device decides where it
    /// runs.
    fn normOnCpu(p: *c.llama_model_params) void {
        overrides[0] = .{ .pattern = "attn_norm|post_attention_norm", .buft = c.ggml_backend_cpu_buffer_type() };
        overrides[1] = .{ .pattern = null, .buft = null };
        p.tensor_buft_overrides = &overrides;
    }
    var tensor_devices: [3]c.ggml_backend_dev_t = .{ null, null, null };
    /// `LLAMA_SPLIT_MODE_TENSOR` over the GPU and the CPU: a meta device,
    /// with every weight and cache tensor sliced by
    /// `llama_meta_device_get_split_state`. The multi-GPU path, on the two
    /// devices this machine has.
    /// The GB10's CUDA device is integrated (`IGPU`), so a `GPU` lookup
    /// alone found nothing there and left an empty list.
    fn tensorSplit(p: *c.llama_model_params) void {
        tensor_devices[0] = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_GPU) orelse
            c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_IGPU);
        tensor_devices[1] = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU);
        p.devices = &tensor_devices;
        p.split_mode = c.LLAMA_SPLIT_MODE_TENSOR;
    }
    var split_ratio: [16]f32 = .{ 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    /// The same, uneven: a quarter on the CPU.
    fn tensorSplitUneven(p: *c.llama_model_params) void {
        tensorSplit(p);
        p.tensor_split = &split_ratio;
    }
    var expert_overrides: [2]c.llama_model_tensor_buft_override = undefined;
    /// The experts in host memory, as `--cpu-moe` puts them: the layers
    /// the MoE cache (v0.7.0) caches on the GPU.
    fn hostExperts(p: *c.llama_model_params) void {
        expert_overrides[0] = .{ .pattern = "\\.ffn_(up|down|gate|gate_up)_exps", .buft = c.ggml_backend_cpu_buffer_type() };
        expert_overrides[1] = .{ .pattern = null, .buft = null };
        p.tensor_buft_overrides = &expert_overrides;
    }
    /// A MoE cache too small for every expert, so the LRU evicts, and
    /// ubatches of two tokens, so the cache is used: it reads the cache
    /// only for 32 tokens or fewer. `upstream`'s own test uses the same
    /// n_ubatch (tests/test-llama-archs.cpp:526 @620d2225c).
    fn moeCacheCtx(p: *c.llama_context_params) void {
        p.moe_cache_size = moe_cache_size;
        // non-causal attention needs the whole batch in one ubatch (the
        // C++ asserts it, llama-context.cpp:1898), so that case keeps its
        // ubatch, and its large batches read the cache through
        // `copy_experts` instead
        if (p.attention_type != c.LLAMA_ATTENTION_TYPE_NON_CAUSAL and p.n_batch != 0) p.n_ubatch = 2;
    }
    /// The budget: `MOE_CACHE_SIZE` bytes, or 1 MiB.
    var moe_cache_size: usize = 1 << 20;

    var proxy_devices: [2]c.ggml_backend_dev_t = .{ null, null };
    /// Every layer on harness/proxy_dev.c's device, which declines the
    /// fused Gated Delta Net.
    fn proxy(p: *c.llama_model_params) void {
        proxy_devices[0] = harness_proxy_device();
        p.devices = &proxy_devices;
    }
};

extern fn harness_proxy_device() c.ggml_backend_dev_t;

extern fn _Z33llama_meta_device_get_split_statePK11ggml_tensorPv(tensor: ?*const c.ggml_tensor, ud: ?*anyopaque) c.struct_ggml_backend_meta_split_state;

/// Every weight's split state and the memory tensors', for 2 to 4 devices,
/// from `llama_meta_device_get_split_state` and the port's: the slicing a
/// multi-GPU machine would use, compared without one.
fn splitStates(rm: *c.struct_llama_model, pm: *M.Model) void {
    const before = failures;
    var n: usize = 0;
    const hp = &pm.hparams;
    // The memory tensors, named as the caches name them.
    const ctx = c.ggml_init(.{ .mem_size = 4 * hp.n_layer_all * c.ggml_tensor_overhead(), .mem_buffer = null, .no_alloc = true }).?;
    defer c.ggml_free(ctx);
    var extra: std.ArrayList(*c.ggml_tensor) = .empty;
    defer extra.deinit(gpa);
    for (0..hp.n_layer_all) |il| {
        const l: u32 = @intCast(il);
        if (hp.isRecr(l)) {
            const r = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, hp.nEmbdR());
            _ = c.ggml_format_name(r, "cache_r_l%d", @as(c_int, @intCast(il)));
            const sx = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, hp.nEmbdS());
            _ = c.ggml_format_name(sx, "cache_s_l%d", @as(c_int, @intCast(il)));
            extra.appendSlice(gpa, &.{ r, sx }) catch unreachable;
        } else {
            const k = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F16, hp.nEmbdKGqa(l), 256);
            _ = c.ggml_format_name(k, "cache_k_l%d", @as(c_int, @intCast(il)));
            const v = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F16, hp.nEmbdVGqa(l), 256);
            _ = c.ggml_format_name(v, "cache_v_l%d", @as(c_int, @intCast(il)));
            extra.appendSlice(gpa, &.{ k, v }) catch unreachable;
        }
    }
    var n_devices: usize = 2;
    while (n_devices <= 4) : (n_devices += 1) {
        var ref_ud: harness_root.meta_split.Userdata = .{ .n_devices = n_devices, .model = @ptrCast(@alignCast(rm)) };
        var port_ud: harness_root.meta_split.Userdata = .{ .n_devices = n_devices, .model = pm };
        const lists = [_][]const *c.ggml_tensor{ pm.tensors_by_name.items, extra.items };
        for (lists) |list| for (list) |t| {
            const a = _Z33llama_meta_device_get_split_statePK11ggml_tensorPv(t, &ref_ud);
            const b = harness_root.meta_split.getSplitState(t, &port_ud);
            n += 1;
            checks += 1;
            if (!std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b))) {
                failures += 1;
                if (failures - before <= 5) {
                    std.debug.print("FAIL split state {s}, {d} devices: ref axis {d} n_segments {d} ne {any} nr {any}\n                                   port axis {d} n_segments {d} ne {any} nr {any}\n", .{
                        std.mem.sliceTo(&t.name, 0), n_devices, a.axis, a.n_segments, a.ne[0..8], a.nr[0..4], b.axis, b.n_segments, b.ne[0..8], b.nr[0..4],
                    });
                }
            }
        };
    }
    std.debug.print("  {s:<40} {s:<4} {d:>7} tensors\n", .{ "split states", if (failures == before) "ok" else "FAIL", n });
}

// -----------------------------------------------------------------------------
// prism.hadamard (v0.7.0)

/// The weight kinds `load_hparams_hadamard` accepts as folded
/// (llama-model.cpp:2030 @620d2225c).
const foldable_kinds = [_][]const u8{
    "attn_q",           "attn_k",         "attn_v",       "attn_qkv",       "attn_gate",   "attn_output",
    "ffn_gate",         "ffn_up",         "ffn_down",     "ffn_gate_exps",  "ffn_up_exps", "ffn_down_exps",
    "ffn_gate_up_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ssm_out",
};

fn foldable(name: []const u8) bool {
    if (std.mem.eql(u8, name, "output.weight")) return true;
    if (!std.mem.startsWith(u8, name, "blk.")) return false;
    var pos: usize = 4;
    while (pos < name.len and std.ascii.isDigit(name[pos])) pos += 1;
    if (pos == 4 or pos >= name.len or name[pos] != '.') return false;
    const rest = name[pos + 1 ..];
    for (foldable_kinds) |k| {
        if (rest.len == k.len + 7 and std.mem.startsWith(u8, rest, k) and std.mem.endsWith(u8, rest, ".weight")) return true;
    }
    return false;
}

/// `--hadamard`: `src` with `prism.hadamard` metadata added, written to
/// `dst` -- every foldable weight whose input width the block of 8 divides
/// marked folded, explicit signs per width, the token embedding as the
/// latent table, and for a gated delta net the grouped value heads. The
/// weights are not really folded, so the model computes nonsense; both
/// sides compute the same nonsense, through the transforms the graph must
/// apply, which is what is compared.
///
/// With `drop_output`, `output.weight` is left out, so the head is the
/// token embedding: version 2's tied latent output.
fn deriveHadamard(src: [:0]const u8, dst: [:0]const u8, drop_output: bool) void {
    var data_ctx: ?*c.ggml_context = null;
    const g = c.gguf_init_from_file(src.ptr, .{ .no_alloc = false, .ctx = &data_ctx }) orelse @panic("cannot read the model for --hadamard");
    defer c.gguf_free(g);
    defer c.ggml_free(data_ctx);
    const out = c.gguf_init_empty().?;
    defer c.gguf_free(out);
    c.gguf_set_kv(out, g);

    const block: u32 = 8;
    var names: std.ArrayList([*:0]const u8) = .empty;
    defer names.deinit(gpa);
    var widths: std.ArrayList(i32) = .empty;
    defer widths.deinit(gpa);
    var has_output = false;
    var inverse = false;
    // The file's tensors in its order, not the context's: that also holds
    // the data blob.
    var ti: i64 = 0;
    while (ti < c.gguf_get_n_tensors(g)) : (ti += 1) {
        const t = c.ggml_get_tensor(data_ctx, c.gguf_get_tensor_name(g, ti)).?;
        const name = std.mem.sliceTo(&t.*.name, 0);
        if (drop_output and std.mem.eql(u8, name, "output.weight")) continue;
        c.gguf_add_tensor(out, t);
        const w: i32 = @intCast(t.*.ne[0]);
        if (std.mem.eql(u8, name, "output.weight")) has_output = true;
        const as_inverse = std.mem.eql(u8, name, "token_embd.weight");
        if (!(foldable(name) or as_inverse) or @rem(w, @as(i32, block)) != 0) continue;
        if (as_inverse) inverse = true else names.append(gpa, @ptrCast(&t.*.name)) catch unreachable;
        if (std.mem.indexOfScalar(i32, widths.items, w) == null) widths.append(gpa, w) catch unreachable;
    }
    // A tied output (no output.weight) needs version 2 and tied_output.
    const tied = inverse and !has_output;
    c.gguf_set_val_u32(out, "prism.hadamard.version", if (tied) 2 else 1);
    c.gguf_set_val_bool(out, "prism.hadamard.tied_output", tied);
    c.gguf_set_val_u32(out, "prism.hadamard.block_size", block);
    c.gguf_set_val_str(out, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    c.gguf_set_val_str(out, "prism.hadamard.axis", "input-last-dimension");
    c.gguf_set_val_str(out, "prism.hadamard.sign_mode", "explicit");
    c.gguf_set_arr_data(out, "prism.hadamard.sign_widths", c.GGUF_TYPE_INT32, widths.items.ptr, widths.items.len);
    var values: std.ArrayList(i32) = .empty;
    defer values.deinit(gpa);
    for (widths.items) |w| {
        for (0..@intCast(w)) |i| values.append(gpa, if ((i * 7 + @as(usize, @intCast(w))) % 3 == 0) -1 else 1) catch unreachable;
    }
    c.gguf_set_arr_data(out, "prism.hadamard.sign_values", c.GGUF_TYPE_INT32, values.items.ptr, values.items.len);
    c.gguf_set_arr_str(out, "prism.hadamard.weight_names", @ptrCast(names.items.ptr), names.items.len);
    if (inverse) {
        const inv = [_][*:0]const u8{"token_embd.weight"};
        c.gguf_set_arr_str(out, "prism.hadamard.inverse_weight_names", @ptrCast(@constCast(&inv)), inv.len);
    }
    // The gated delta net's value heads in grouped order, where the
    // geometry allows it.
    const arch_k = c.gguf_find_key(g, "general.architecture");
    const arch = if (arch_k >= 0) std.mem.span(c.gguf_get_val_str(g, arch_k)) else "";
    var kb: [96]u8 = undefined;
    const dt_k = c.gguf_find_key(g, (std.fmt.bufPrintSentinel(&kb, "{s}.ssm.time_step_rank", .{arch}, 0) catch unreachable).ptr);
    const ng_k = c.gguf_find_key(g, (std.fmt.bufPrintSentinel(&kb, "{s}.ssm.group_count", .{arch}, 0) catch unreachable).ptr);
    if (dt_k >= 0 and ng_k >= 0) {
        const n_v = c.gguf_get_val_u32(g, dt_k);
        const n_k = c.gguf_get_val_u32(g, ng_k);
        if (n_k > 0 and n_v > 0 and n_v % n_k == 0) c.gguf_set_val_bool(out, "prism.hadamard.gdn_v_grouped", true);
    }
    if (!c.gguf_write_to_file(out, dst.ptr, false)) @panic("cannot write the --hadamard model");
    std.debug.print("  --hadamard: {d} folded weights, {d} widths, inverse {}, tied {}, written to {s}\n", .{ names.items.len, widths.items.len, inverse, tied, dst });
}

/// `--section-order <order>`: `src` with `<arch>.rope.section_order` set
/// (v0.7.0, MiniCPM-V 4.7's `zyxt`), written to `dst`, for an M-RoPE model:
/// the embedding rows of the mixed batches then have their positions
/// reordered into the model's section order.
fn deriveSectionOrder(src: [:0]const u8, dst: [:0]const u8, order: []const u8) void {
    var data_ctx: ?*c.ggml_context = null;
    const g = c.gguf_init_from_file(src.ptr, .{ .no_alloc = false, .ctx = &data_ctx }) orelse @panic("cannot read the model for --section-order");
    defer c.gguf_free(g);
    defer c.ggml_free(data_ctx);
    const out = c.gguf_init_empty().?;
    defer c.gguf_free(out);
    c.gguf_set_kv(out, g);
    var ti: i64 = 0;
    while (ti < c.gguf_get_n_tensors(g)) : (ti += 1) {
        c.gguf_add_tensor(out, c.ggml_get_tensor(data_ctx, c.gguf_get_tensor_name(g, ti)).?);
    }
    const arch_k = c.gguf_find_key(g, "general.architecture");
    var kb: [96]u8 = undefined;
    var vb: [16]u8 = undefined;
    const key = std.fmt.bufPrintSentinel(&kb, "{s}.rope.section_order", .{std.mem.span(c.gguf_get_val_str(g, arch_k))}, 0) catch unreachable;
    const val = std.fmt.bufPrintSentinel(&vb, "{s}", .{order}, 0) catch unreachable;
    c.gguf_set_val_str(out, key.ptr, val.ptr);
    if (!c.gguf_write_to_file(out, dst.ptr, false)) @panic("cannot write the --section-order model");
    std.debug.print("  --section-order: {s} = {s}, written to {s}\n", .{ key, val, dst });
}

const model_cfgs = [_]ModelCfg{
    .{ .name = "gpu", .edit = ModelEdits.gpu },
    .{ .name = "cpu", .edit = ModelEdits.cpu },
    .{ .name = "partial", .edit = ModelEdits.partial },
    .{ .name = "proxy", .edit = ModelEdits.proxy },
    .{ .name = "norm-on-cpu", .edit = ModelEdits.normOnCpu },
    .{ .name = "tensor", .edit = ModelEdits.tensorSplit, .nodes = false },
    .{ .name = "tensor-uneven", .edit = ModelEdits.tensorSplitUneven, .nodes = false },
    .{ .name = "moe-cache", .edit = ModelEdits.hostExperts, .ctx_edit = ModelEdits.moeCacheCtx, .explicit = true },
    .{ .name = "host-experts", .edit = ModelEdits.hostExperts, .explicit = true },
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: context_diff <model.gguf> [--quick] [--model gpu|cpu|partial|proxy] [--case substring]\n", .{});
        std.process.exit(2);
    }
    var path = try gpa.dupeSentinel(u8, args[1], 0);
    var hadamard = false;
    var hadamard_tied = false;
    var section_order: ?[]const u8 = null;
    var target_path: ?[:0]const u8 = null;
    var quick = false;
    var only: ?[]const u8 = null;
    var only_exact = false;
    var skip_case: ?[]const u8 = null;
    var only_model: ?[]const u8 = null;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--quick")) quick = true;
        if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            only_model = args[i + 1];
            i += 1;
        }
        if (std.mem.eql(u8, args[i], "--case") and i + 1 < args.len) {
            only = args[i + 1];
            i += 1;
        }
        // The exact name, for running one case alone.
        if (std.mem.eql(u8, args[i], "--case-exact") and i + 1 < args.len) {
            only = args[i + 1];
            only_exact = true;
            i += 1;
        }
        // One case left out, by exact name: the C++ itself is
        // nondeterministic there (scripts/arch-model-diff's list says why).
        if (std.mem.eql(u8, args[i], "--skip-case") and i + 1 < args.len) {
            skip_case = args[i + 1];
            i += 1;
        }
        // The model with prism.hadamard metadata added, see deriveHadamard.
        if (std.mem.eql(u8, args[i], "--hadamard")) hadamard = true;
        // A target model for a draft that needs one (see `script`).
        if (std.mem.eql(u8, args[i], "--target") and i + 1 < args.len) {
            target_path = try gpa.dupeSentinel(u8, args[i + 1], 0);
            i += 1;
        }
        if (std.mem.eql(u8, args[i], "--section-order") and i + 1 < args.len) {
            section_order = args[i + 1];
            i += 1;
        }
        if (std.mem.eql(u8, args[i], "--hadamard-tied")) {
            hadamard = true;
            hadamard_tied = true;
        }
        // The case names, one a line, for a script that runs them one by one.
        if (std.mem.eql(u8, args[i], "--list-cases")) {
            for (cases) |cs| std.debug.print("{s}\n", .{cs.name});
            std.process.exit(0);
        }
    }

    // CONTEXT_DIFF_OUT=<dir> keeps both transcripts of every case.
    const out_dir: ?[]const u8 = if (std.c.getenv("CONTEXT_DIFF_OUT")) |d| std.mem.span(d) else null;
    // CONTEXT_DIFF_TMP=<dir> holds the state files each side writes.
    if (std.c.getenv("CONTEXT_DIFF_TMP")) |d| tmp_dir = std.mem.span(d);
    if (hadamard) {
        const dst = try std.fmt.allocPrintSentinel(gpa, "{s}/hadamard.gguf", .{tmp_dir}, 0);
        deriveHadamard(path, dst, hadamard_tied);
        path = dst;
    }
    if (section_order) |so| {
        const dst = try std.fmt.allocPrintSentinel(gpa, "{s}/section-order.gguf", .{tmp_dir}, 0);
        deriveSectionOrder(path, dst, so);
        path = dst;
    }
    model_path = path;
    live = std.c.getenv("CONTEXT_DIFF_LIVE") != null;
    const port_first = std.c.getenv("CONTEXT_DIFF_PORT_FIRST") != null;

    if (std.c.getenv("MOE_CACHE_SIZE")) |v| ModelEdits.moe_cache_size = std.fmt.parseInt(usize, std.mem.span(v), 10) catch ModelEdits.moe_cache_size;

    c.llama_log_set(logCb, null);

    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    // CONTEXT_DIFF_SIDE=ref|port loads and runs that side alone: a side
    // that aborts while loading must not take the other's run with it,
    // or a port load abort reads as the C++'s.
    const side_only: ?[]const u8 = if (std.c.getenv("CONTEXT_DIFF_SIDE")) |one| std.mem.span(one) else null;
    const load_ref = side_only == null or !std.mem.eql(u8, side_only.?, "port");
    const load_port = side_only == null or std.mem.eql(u8, side_only.?, "port");

    for (model_cfgs) |mc| {
        if (only_model) |o| {
            if (!std.mem.eql(u8, mc.name, o)) continue;
        } else if (mc.explicit or (quick and !std.mem.eql(u8, mc.name, "gpu"))) continue;
        model_ctx_edit = mc.ctx_edit;
        var mp = c.llama_model_default_params();
        mc.edit(&mp);
        // What each side logs while loading, compared below.
        var ref_load: std.ArrayList(u8) = .empty;
        defer ref_load.deinit(gpa);
        var port_load: std.ArrayList(u8) = .empty;
        defer port_load.deinit(gpa);
        rec = &ref_load;
        const rm_opt: ?*c.struct_llama_model = if (!load_ref) null else c.llama_model_load_from_file(path.ptr, mp) orelse {
            // The last lines say why, as for the port below.
            var lines = std.mem.splitBackwardsScalar(u8, std.mem.trimEnd(u8, ref_load.items, "\n"), '\n');
            var tail: [3][]const u8 = undefined;
            var n: usize = 0;
            while (n < tail.len) : (n += 1) tail[n] = lines.next() orelse break;
            std.debug.print("FAIL: the C++ cannot load {s}\n", .{path});
            while (n > 0) : (n -= 1) std.debug.print("    ref: {s}\n", .{tail[n - 1]});
            std.process.exit(1);
        };
        defer if (rm_opt) |m| c.llama_model_free(m);
        var splits: std.ArrayList([:0]const u8) = .empty;
        defer {
            for (splits.items) |s| gpa.free(s);
            splits.deinit(gpa);
        }
        rec = &port_load;
        const pm_opt: ?*M.Model = if (!load_port) null else M.loadFromFileImpl(.{ .path = path }, &splits, null, null, mp) orelse {
            // The last lines say why: an architecture without a ported
            // `load_arch_hparams` stops with a message naming it, which is
            // how `scripts/arch-model-diff` tells unported from broken.
            var lines = std.mem.splitBackwardsScalar(u8, std.mem.trimEnd(u8, port_load.items, "\n"), '\n');
            var tail: [3][]const u8 = undefined;
            var n: usize = 0;
            while (n < tail.len) : (n += 1) tail[n] = lines.next() orelse break;
            std.debug.print("FAIL: the port cannot load {s}\n", .{path});
            while (n > 0) : (n -= 1) std.debug.print("    port: {s}\n", .{tail[n - 1]});
            std.process.exit(1);
        };
        defer if (pm_opt) |m| M.destroy(m);
        if (target_path) |tpath| {
            var tsplits: std.ArrayList([:0]const u8) = .empty;
            defer {
                for (tsplits.items) |ts| gpa.free(ts);
                tsplits.deinit(gpa);
            }
            var tlog: std.ArrayList(u8) = .empty;
            defer tlog.deinit(gpa);
            rec = &tlog;
            if (load_ref) target_ref = c.llama_model_load_from_file(tpath.ptr, mp);
            if (load_port) target_port = M.loadFromFileImpl(.{ .path = tpath }, &tsplits, null, null, mp);
            rec = null;
            if ((load_ref and target_ref == null) or (load_port and target_port == null)) {
                std.debug.print("FAIL: cannot load the target {s}\n", .{tpath});
                std.process.exit(1);
            }
        }
        defer {
            if (target_ref) |m| c.llama_model_free(m);
            if (target_port) |m| M.destroy(m);
            target_ref = null;
            target_port = null;
        }
        rec = null;
        if (load_ref and load_port) {
            var rn = normalize(ref_load.items);
            defer rn.deinit(gpa);
            var pn = normalize(port_load.items);
            defer pn.deinit(gpa);
            var nb: [64]u8 = undefined;
            const before = failures;
            _ = firstDiff(try std.fmt.bufPrint(&nb, "{s}: load", .{mc.name}), rn.items, pn.items);
            std.debug.print("  {s:<40} {s:<4} {d:>7} lines\n", .{ try std.fmt.bufPrint(&nb, "{s}: load", .{mc.name}), if (failures == before) "ok" else "FAIL", std.mem.count(u8, rn.items, "\n") });
        }

        if (load_ref and load_port and std.mem.startsWith(u8, mc.name, "tensor")) splitStates(rm_opt.?, pm_opt.?);

        // Whichever side is loaded answers for the vocabulary.
        const VA = struct {
            fn vocabOf(r: ?*c.struct_llama_model, p_: ?*M.Model) ?*const c.struct_llama_vocab {
                return if (r) |m| c.llama_model_get_vocab(m) else harness_root.api.model.llama_model_get_vocab(@ptrCast(p_.?));
            }
            fn nTokens(r: bool, v: ?*const c.struct_llama_vocab) i32 {
                return if (r) c.llama_vocab_n_tokens(v) else harness_root.api.vocab.llama_vocab_n_tokens(v);
            }
            fn vtype(r: bool, v: ?*const c.struct_llama_vocab) c.enum_llama_vocab_type {
                return if (r) c.llama_vocab_type(v) else harness_root.api.vocab.llama_vocab_type(v);
            }
        };
        const rm: *c.struct_llama_model = if (rm_opt) |m| m else @ptrCast(pm_opt.?);
        const pm: *M.Model = if (pm_opt) |m| m else undefined;
        const vocab = VA.vocabOf(rm_opt, pm_opt);
        const n_vocab: usize = @intCast(VA.nTokens(load_ref, vocab));
        var tokens: [256]i32 = undefined;
        // `scripts/arch-model-diff`'s fixtures have no tokenizer
        // (`no_vocab`), so they get a fixed sequence of ids instead, spread
        // over the whole vocabulary.
        const nt: c_int = if (VA.vtype(load_ref, vocab) == c.LLAMA_VOCAB_TYPE_NONE) blk: {
            const n = 24;
            for (0..n) |k| tokens[k] = @intCast((k * 37 + 11) % n_vocab);
            break :blk n;
        } else if (load_ref)
            c.llama_tokenize(vocab, prompt_text, prompt_text.len, &tokens, tokens.len, true, false)
        else
            harness_root.api.vocab.llama_tokenize(vocab, prompt_text, prompt_text.len, &tokens, tokens.len, true, false);
        if (nt <= 8) {
            std.debug.print("FAIL: tokenizing the prompt gave {d}\n", .{nt});
            std.process.exit(1);
        }
        const n_embd: usize = @intCast(if (load_ref) c.llama_model_n_embd_out(rm) else harness_root.api.model.llama_model_n_embd_out(rm));

        // A first context initializes what each backend initializes once
        // per process, which logs.
        var memoryless = false;
        {
            var p = c.llama_context_default_params();
            p.n_ctx = 256;
            // A port-only run must not run the C++ at all, or a fixture the
            // C++ aborts on would abort here and read as the port's abort.
            if (!load_ref) {
                if (api.llama_init_from_model(@ptrCast(pm), p)) |ctx| {
                    memoryless = api.llama_get_memory(ctx) == null;
                    api.llama_free(ctx);
                }
            } else if (c.llama_init_from_model(rm, p)) |ctx| {
                memoryless = c.llama_get_memory(ctx) == null;
                c.llama_free(ctx);
            }
        }
        var skipped: usize = 0;

        for (cases) |cs| {
            if (only) |o| {
                if (only_exact and !std.mem.eql(u8, cs.name, o)) continue;
                if (!only_exact and std.mem.find(u8, cs.name, o) == null) continue;
            }
            if (quick and !std.mem.eql(u8, cs.name, "default") and !cs.multi_seq) continue;
            if (!cs.tensor_ok and std.mem.startsWith(u8, mc.name, "tensor")) continue;
            if (skip_case) |sk| if (std.mem.eql(u8, cs.name, sk)) {
                skipped += 1;
                continue;
            };
            if (!cs.memoryless_ok and memoryless) {
                skipped += 1;
                continue;
            }
            var ref: std.ArrayList(u8) = .empty;
            defer ref.deinit(gpa);
            var port: std.ArrayList(u8) = .empty;
            defer port.deinit(gpa);

            hash_nodes = mc.nodes;
            n_nodes_seen = 0;
            rec = &ref;
            // CONTEXT_DIFF_PORT_FIRST=1 runs the port's side first, to tell
            // a fault from an order dependence.
            var ref_nodes: usize = 0;
            // CONTEXT_DIFF_SIDE=ref|port runs one side and writes its
            // transcript to stdout as it goes, for a side that crashes.
            if (std.c.getenv("CONTEXT_DIFF_SIDE")) |one| {
                const which = std.mem.span(one);
                var buf: std.ArrayList(u8) = .empty;
                defer buf.deinit(gpa);
                rec = &buf;
                side = which;
                stream_out = true;
                if (std.mem.eql(u8, which, "port")) {
                    script(api, harness_root.api.state, @ptrCast(pm), cs, tokens[0..@intCast(nt)], n_vocab, n_embd);
                } else {
                    script(c, c, rm, cs, tokens[0..@intCast(nt)], n_vocab, n_embd);
                }
                rec = null;
                continue;
            }
            if (port_first) {
                rec = &port;
                side = "port";
                script(api, harness_root.api.state, @ptrCast(pm), cs, tokens[0..@intCast(nt)], n_vocab, n_embd);
                n_nodes_seen = 0;
            }
            rec = &ref;
            side = "ref";
            script(c, c, rm, cs, tokens[0..@intCast(nt)], n_vocab, n_embd);
            ref_nodes = n_nodes_seen;
            if (!port_first) {
                n_nodes_seen = 0;
                rec = &port;
                side = "port";
                script(api, harness_root.api.state, @ptrCast(pm), cs, tokens[0..@intCast(nt)], n_vocab, n_embd);
            }
            rec = null;

            var rn = normalize(ref.items);
            defer rn.deinit(gpa);
            var pn = normalize(port.items);
            defer pn.deinit(gpa);

            if (out_dir) |d| {
                var pb: [512]u8 = undefined;
                const ref_path = try std.fmt.bufPrint(&pb, "{s}/{s}-{s}.ref", .{ d, mc.name, cs.name });
                try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = ref_path, .data = rn.items });
                const port_path = try std.fmt.bufPrint(&pb, "{s}/{s}-{s}.port", .{ d, mc.name, cs.name });
                try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = port_path, .data = pn.items });
            }

            var name_buf: [128]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s}: {s}", .{ mc.name, cs.name });
            const before = failures;
            checks += 1;
            if (ref_nodes == 0 and std.mem.find(u8, ref.items, "init failed") == null) {
                failures += 1;
                std.debug.print("FAIL {s}: the C++ computed no node\n", .{name});
            }
            _ = firstDiff(name, rn.items, pn.items);
            const lines = std.mem.count(u8, rn.items, "\n");
            std.debug.print("  {s:<40} {s:<4} {d:>7} lines {d:>7} nodes\n", .{ name, if (failures == before) "ok" else "FAIL", lines, ref_nodes });
        }
        if (skipped != 0) std.debug.print("  {s}: {d} cases skipped: the C++ asserts on them for a model without memory, or is nondeterministic there\n", .{ mc.name, skipped });
    }

    std.debug.print("context-diff: {d} checks, {d} failures\n", .{ checks, failures });
    if (failures != 0) std.process.exit(1);
}
