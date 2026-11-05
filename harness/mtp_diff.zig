//! Compares the port's MTP (NextN) draft head with the C++'s, over a fixture
//! with a NextN block.
//!
//! **Not a port.** `scripts/arch-model-diff` runs every fixture through
//! `harness/context_diff.zig`, which creates only target contexts: an MTP
//! graph (`LLM_GRAPH_TYPE_DECODER_MTP`) is built only in a context of type
//! `LLAMA_CONTEXT_TYPE_MTP`, fed the target's next-token hidden state. This
//! loads one fixture with `load_mtp` through the C++ (`libllama-ref.a`) and
//! through the port's `api` functions of the same names, and runs one script
//! on each, as `common/speculative.cpp`'s MTP drafter drives the two
//! contexts:
//!
//! - the target decodes a prompt with unmasked next-token embeddings (the
//!   trunk's `crop_after_nextn` arm), then two more tokens with masked ones
//!   (`crop_before_nextn` with the hidden state on);
//! - an MTP context catches up on the prompt, each token paired with the
//!   target's hidden state one position back, outputs off;
//! - it drafts three single tokens, chaining its own hidden state, with
//!   masked embeddings; then a batch with every output and unmasked ones;
//!   then a batch of embedding inputs (multimodal tokens, which carry an
//!   embedding instead of an id) with their states.
//!
//! Everything logged, every graph node of both contexts (op, shape, strides,
//! `op_params`, sources, data hash), every return value and every output
//! row's bits must agree. Pointers in log lines are masked.
//!
//! Usage: mtp_diff <fixture.gguf> gpu|cpu, as `scripts/mtp-diff` runs it.
//! `MTP_DIFF_SIDE=ref|port` runs one side; `MTP_DIFF_OUT=<dir>` keeps both
//! transcripts.

const std = @import("std");
const input_tensor_order = @import("input_tensor_order.zig");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;

const gpa = std.heap.c_allocator;

const api = harness_root.api;

/// The port, under the C names.
const P = struct {
    pub const llama_model_load_from_file = api.llama.llama_model_load_from_file;
    pub const llama_model_free = api.model.llama_model_free;
    pub const llama_model_n_embd_inp = api.model.llama_model_n_embd_inp;
    pub const llama_model_n_embd_out = api.model.llama_model_n_embd_out;
    pub const llama_model_get_vocab = api.model.llama_model_get_vocab;
    pub const llama_vocab_n_tokens = api.vocab.llama_vocab_n_tokens;
    pub const llama_init_from_model = api.context.llama_init_from_model;
    pub const llama_free = api.context.llama_free;
    pub const llama_decode = api.context.llama_decode;
    pub const llama_process = api.context.llama_process;
    pub const llama_get_logits_ith = api.context.llama_get_logits_ith;
    pub const llama_set_embeddings_nextn = api.context.llama_set_embeddings_nextn;
    pub const llama_get_embeddings_nextn_ith = api.context.llama_get_embeddings_nextn_ith;
    pub const llama_get_memory = api.context.llama_get_memory;
    pub const llama_memory_seq_pos_min = api.context.llama_memory_seq_pos_min;
    pub const llama_memory_seq_pos_max = api.context.llama_memory_seq_pos_max;
    pub const llama_batch_ext_init = api.batch.llama_batch_ext_init;
    pub const llama_batch_ext_free = api.batch.llama_batch_ext_free;
    pub const llama_batch_ext_clear = api.batch.llama_batch_ext_clear;
    pub const llama_batch_ext_add_token = api.batch.llama_batch_ext_add_token;
    pub const llama_batch_ext_add_embd = api.batch.llama_batch_ext_add_embd;
    pub const llama_batch_ext_set_pos = api.batch.llama_batch_ext_set_pos;
    pub const llama_batch_ext_set_embd_state = api.batch.llama_batch_ext_set_embd_state;
    pub const llama_batch_ext_set_output_logits = api.batch.llama_batch_ext_set_output_logits;
};

/// The C++, under the C names. The next-token embedding calls are in
/// `llama-ext.h`, which has C++ linkage, so they are reached by their
/// mangled names, taken from `nm` on `libllama-ref.a`.
const R = struct {
    pub const llama_model_load_from_file = c.llama_model_load_from_file;
    pub const llama_model_free = c.llama_model_free;
    pub const llama_model_n_embd_inp = c.llama_model_n_embd_inp;
    pub const llama_model_n_embd_out = c.llama_model_n_embd_out;
    pub const llama_model_get_vocab = c.llama_model_get_vocab;
    pub const llama_vocab_n_tokens = c.llama_vocab_n_tokens;
    pub const llama_init_from_model = c.llama_init_from_model;
    pub const llama_free = c.llama_free;
    pub const llama_decode = c.llama_decode;
    pub const llama_process = c.llama_process;
    pub const llama_get_logits_ith = c.llama_get_logits_ith;
    pub const llama_get_memory = c.llama_get_memory;
    pub const llama_memory_seq_pos_min = c.llama_memory_seq_pos_min;
    pub const llama_memory_seq_pos_max = c.llama_memory_seq_pos_max;
    pub const llama_batch_ext_init = c.llama_batch_ext_init;
    pub const llama_batch_ext_free = c.llama_batch_ext_free;
    pub const llama_batch_ext_clear = c.llama_batch_ext_clear;
    pub const llama_batch_ext_add_token = c.llama_batch_ext_add_token;
    pub const llama_batch_ext_add_embd = c.llama_batch_ext_add_embd;
    pub const llama_batch_ext_set_pos = c.llama_batch_ext_set_pos;
    pub const llama_batch_ext_set_embd_state = c.llama_batch_ext_set_embd_state;
    pub const llama_batch_ext_set_output_logits = c.llama_batch_ext_set_output_logits;
    pub const llama_set_embeddings_nextn = @extern(*const fn (?*c.struct_llama_context, bool, bool) callconv(.c) void, .{ .name = "_Z26llama_set_embeddings_nextnP13llama_contextbb" });
    pub const llama_get_embeddings_nextn_ith = @extern(*const fn (?*c.struct_llama_context, i32) callconv(.c) ?[*]f32, .{ .name = "_Z30llama_get_embeddings_nextn_ithP13llama_contexti" });
};

// -----------------------------------------------------------------------------
// The transcript

var rec: ?*std.ArrayList(u8) = null;

/// One side alone (`MTP_DIFF_SIDE`) also writes its transcript to stderr as
/// it goes, so a run that aborts leaves the lines before the abort for
/// `scripts/mtp-diff` to compare.
var stream = false;

fn note(comptime fmt: []const u8, args: anytype) void {
    const r = rec orelse return;
    r.print(gpa, fmt ++ "\n", args) catch unreachable;
    if (stream) std.debug.print(fmt ++ "\n", args);
}

fn logCb(_: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const r = rec orelse return;
    const t = std.mem.span(@as([*:0]const u8, @ptrCast(text)));
    r.appendSlice(gpa, t) catch unreachable;
    if (stream) std.debug.print("{s}", .{t});
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
            out.appendSlice(gpa, line[0 .. at + "available_memory_kb: ".len]) catch unreachable;
            out.appendSlice(gpa, "N") catch unreachable;
            continue;
        }
        if (std.mem.find(u8, line, " MiB free")) |end| {
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
    const sorted = input_tensor_order.sortRuns(gpa, out.items);
    out.deinit(gpa);
    return sorted;
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
// The script

fn argmax(r: [*]const f32, n: usize) i32 {
    var best: usize = 0;
    for (1..n) |i| if (r[i] > r[best]) {
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

const n_prompt = 24;

fn embd(p: [*]const f32, n: usize) c.struct_llama_embd {
    return .{ .data = p, .n_rows = 1, .n_embd = n };
}

fn script(comptime A: type, path: [:0]const u8, gpu: bool) void {
    var mp = c.llama_model_default_params();
    if (!gpu) mp.n_gpu_layers = 0;
    mp.load_mtp = true;

    const model = A.llama_model_load_from_file(path.ptr, mp) orelse {
        note("model: load failed", .{});
        return;
    };
    defer A.llama_model_free(model);
    const n_vocab: usize = @intCast(A.llama_vocab_n_tokens(A.llama_model_get_vocab(model)));
    const n_embd_inp: usize = @intCast(A.llama_model_n_embd_inp(model));
    const n_embd_out: usize = @intCast(A.llama_model_n_embd_out(model));
    note("n_vocab {d} n_embd_inp {d} n_embd_out {d}", .{ n_vocab, n_embd_inp, n_embd_out });

    var cp = c.llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_ubatch = 64;
    cp.cb_eval = nodeCb;
    cp.no_perf = true;
    const tctx = A.llama_init_from_model(model, cp) orelse {
        note("target: context failed", .{});
        return;
    };
    defer A.llama_free(tctx);
    // Unmasked: every row's hidden state, the trunk's crop_after_nextn.
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
    var id: i32 = if (last) |r| argmax(r, n_vocab) else 0;

    // The target's hidden state per prompt row.
    const h_tgt = gpa.alloc(f32, n_prompt * n_embd_out) catch unreachable;
    defer gpa.free(h_tgt);
    @memset(h_tgt, 0);
    for (0..n_prompt) |k| {
        const r = A.llama_get_embeddings_nextn_ith(tctx, @intCast(k));
        row("target h", r, n_embd_out, @intCast(k));
        if (r) |p| @memcpy(h_tgt[k * n_embd_out ..][0..n_embd_out], p[0..n_embd_out]);
    }

    var dcp = c.llama_context_default_params();
    dcp.cb_eval = nodeCb;
    dcp.no_perf = true;
    dcp.ctx_type = c.LLAMA_CONTEXT_TYPE_MTP;
    dcp.n_ctx = 128;
    dcp.n_ubatch = 64;
    const dctx = A.llama_init_from_model(model, dcp) orelse {
        note("mtp: context failed", .{});
        return;
    };
    defer A.llama_free(dctx);
    A.llama_set_embeddings_nextn(dctx, true, true);
    const dmem = A.llama_get_memory(dctx);

    const eb = A.llama_batch_ext_init(dctx) orelse {
        note("mtp: batch failed", .{});
        return;
    };
    defer A.llama_batch_ext_free(eb);

    const zeros = gpa.alloc(f32, n_embd_out) catch unreachable;
    defer gpa.free(zeros);
    @memset(zeros, 0);

    // The catch-up: each prompt token with the target's state one row back,
    // the first with the pending (here zero) state.
    for (tokens, 0..) |t, k| {
        const idx = A.llama_batch_ext_add_token(eb, 0, t);
        const pos: i32 = @intCast(k);
        _ = A.llama_batch_ext_set_pos(eb, idx, &pos);
        const h_row: [*]const f32 = if (k == 0) zeros.ptr else h_tgt[(k - 1) * n_embd_out ..].ptr;
        note("catch-up {d}: idx {d} state {}", .{ k, idx, A.llama_batch_ext_set_embd_state(eb, idx, embd(h_row, n_embd_out)) });
    }
    note("mtp catch-up: {d}", .{A.llama_process(dctx, c.LLAMA_PROCESS_TYPE_DECODE, eb)});
    note("mtp seq 0: [{d}, {d}]", .{ A.llama_memory_seq_pos_min(dmem, 0), A.llama_memory_seq_pos_max(dmem, 0) });

    // Three single-token drafts, each chaining the head's own state.
    const h = gpa.alloc(f32, n_embd_out) catch unreachable;
    defer gpa.free(h);
    @memcpy(h, h_tgt[(n_prompt - 1) * n_embd_out ..][0..n_embd_out]);
    var pos: i32 = n_prompt;
    for (0..3) |s| {
        A.llama_batch_ext_clear(eb);
        const idx = A.llama_batch_ext_add_token(eb, 0, id);
        _ = A.llama_batch_ext_set_pos(eb, idx, &pos);
        _ = A.llama_batch_ext_set_embd_state(eb, idx, embd(h.ptr, n_embd_out));
        _ = A.llama_batch_ext_set_output_logits(eb, idx, true);
        note("draft {d} at {d}: {d}", .{ s, pos, A.llama_process(dctx, c.LLAMA_PROCESS_TYPE_DECODE, eb) });
        const l = A.llama_get_logits_ith(dctx, 0);
        row("draft logits", l, n_vocab, 0);
        if (l) |r| id = argmax(r, n_vocab);
        note("draft token {d}", .{id});
        const hn = A.llama_get_embeddings_nextn_ith(dctx, 0);
        row("draft h", hn, n_embd_out, 0);
        if (hn) |r| @memcpy(h, r[0..n_embd_out]);
        pos += 1;
    }

    // A batch with every output, unmasked: the MTP graph's
    // crop_after_nextn.
    A.llama_set_embeddings_nextn(dctx, true, false);
    A.llama_batch_ext_clear(eb);
    for (0..4) |k| {
        const idx = A.llama_batch_ext_add_token(eb, 0, tokens[k + 3]);
        _ = A.llama_batch_ext_set_pos(eb, idx, &pos);
        _ = A.llama_batch_ext_set_embd_state(eb, idx, embd(h_tgt[(k + 5) * n_embd_out ..].ptr, n_embd_out));
        _ = A.llama_batch_ext_set_output_logits(eb, idx, true);
        pos += 1;
    }
    note("draft batch: {d}", .{A.llama_process(dctx, c.LLAMA_PROCESS_TYPE_DECODE, eb)});
    for (0..4) |k| {
        row("draft logits", A.llama_get_logits_ith(dctx, @intCast(k)), n_vocab, @intCast(k));
        row("draft h", A.llama_get_embeddings_nextn_ith(dctx, @intCast(k)), n_embd_out, @intCast(k));
    }

    // Embedding inputs, as a vision token carries: the head reads the
    // embedding in place of a token row. Masked again, the last row out.
    A.llama_set_embeddings_nextn(dctx, true, true);
    A.llama_batch_ext_clear(eb);
    const e_in = gpa.alloc(f32, 3 * n_embd_inp) catch unreachable;
    defer gpa.free(e_in);
    for (e_in, 0..) |*v, k| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((k * 29) % 61)) - 30)) / 64.0;
    for (0..3) |k| {
        const idx = A.llama_batch_ext_add_embd(eb, 0, embd(e_in[k * n_embd_inp ..].ptr, n_embd_inp));
        // An embedding takes n_pos_per_embd positions, [t, y, x, z] under
        // M-RoPE; four cover every model.
        const pos4 = [4]i32{ pos, pos + 1, pos + 2, 0 };
        _ = A.llama_batch_ext_set_pos(eb, idx, &pos4);
        _ = A.llama_batch_ext_set_embd_state(eb, idx, embd(h_tgt[(k + 11) * n_embd_out ..].ptr, n_embd_out));
        _ = A.llama_batch_ext_set_output_logits(eb, idx, k == 2);
        pos += 1;
    }
    note("draft embd batch: {d}", .{A.llama_process(dctx, c.LLAMA_PROCESS_TYPE_DECODE, eb)});
    row("draft logits", A.llama_get_logits_ith(dctx, -1), n_vocab, -1);
    row("draft h", A.llama_get_embeddings_nextn_ith(dctx, -1), n_embd_out, -1);
    note("mtp seq 0: [{d}, {d}]", .{ A.llama_memory_seq_pos_min(dmem, 0), A.llama_memory_seq_pos_max(dmem, 0) });

    // The target moves on with masked embeddings: crop_before_nextn with
    // the hidden state on.
    A.llama_set_embeddings_nextn(tctx, true, true);
    tb.n_tokens = 0;
    for (0..2) |k| {
        tb.token[k] = id;
        tb.pos[k] = @intCast(n_prompt + k);
        tb.n_seq_id[k] = 1;
        tb.seq_id[k][0] = 0;
        tb.logits[k] = @intFromBool(k == 1);
    }
    tb.n_tokens = 2;
    note("target decode step: {d}", .{A.llama_decode(tctx, tb)});
    row("target logits", A.llama_get_logits_ith(tctx, -1), n_vocab, -1);
    row("target h", A.llama_get_embeddings_nextn_ith(tctx, -1), n_embd_out, -1);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: mtp_diff <fixture.gguf> [gpu|cpu]\n", .{});
        std.process.exit(2);
    }
    const path: [:0]const u8 = args[1];
    const gpu = !(args.len > 2 and std.mem.eql(u8, args[2], "cpu"));

    c.llama_log_set(logCb, null);
    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    var ref: std.ArrayList(u8) = .empty;
    var port: std.ArrayList(u8) = .empty;
    const side = if (std.c.getenv("MTP_DIFF_SIDE")) |s| std.mem.span(s) else "";
    stream = side.len != 0;

    var ref_nodes: usize = 0;
    if (!std.mem.eql(u8, side, "port")) {
        rec = &ref;
        n_nodes = 0;
        script(R, path, gpu);
        rec = null;
        ref_nodes = n_nodes;
        std.debug.print("ref: {d} nodes\n", .{n_nodes});
    }
    if (!std.mem.eql(u8, side, "ref")) {
        rec = &port;
        n_nodes = 0;
        script(P, path, gpu);
        rec = null;
        std.debug.print("port: {d} nodes\n", .{n_nodes});
    }
    var rn = normalize(ref.items);
    var pn = normalize(port.items);
    if (std.c.getenv("MTP_DIFF_OUT")) |d| {
        var pb: [512]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/{s}.ref", .{ std.mem.span(d), if (gpu) "gpu" else "cpu" }), .data = rn.items });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = try std.fmt.bufPrint(&pb, "{s}/{s}.port", .{ std.mem.span(d), if (gpu) "gpu" else "cpu" }), .data = pn.items });
    }
    if (side.len != 0) return;
    var ok = firstDiff(rn.items, pn.items);
    // A script that never reached the MTP graph compares nothing about it.
    if (ok and std.mem.count(u8, rn.items, "node mtp_eh_proj") == 0) {
        std.debug.print("FAIL: no MTP graph ran (no mtp_eh_proj node)\n", .{});
        ok = false;
    }
    std.debug.print("{s}: {s}, {d} lines, {d} nodes\n", .{ if (gpu) "gpu" else "cpu", if (ok) "PASS" else "FAIL", std.mem.count(u8, rn.items, "\n"), ref_nodes });
    rn.deinit(gpa);
    pn.deinit(gpa);
    if (!ok) std.process.exit(1);
}
