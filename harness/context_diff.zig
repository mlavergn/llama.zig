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
//! Usage: context_diff <model.gguf> [--quick]

const std = @import("std");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;
const M = harness_root.model;
const api = harness_root.api.context;

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
    return out;
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

/// Run the script through `A`, which is `c` (the C++) or `api` (the port).
fn script(comptime A: type, comptime S: type, model: ?*c.struct_llama_model, cs: Case, tokens: []const i32, n_vocab: usize, n_embd: usize) void {
    var params = c.llama_context_default_params();
    params.cb_eval = nodeCb;
    params.cb_eval_user_data = null;
    params.no_perf = true;
    cs.edit(&params);

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

    for (0..2) |round| {
        note("round {d}", .{round});
        // The prompt, logits on the last token only (every token when
        // computing embeddings).
        b.n_tokens = 0;
        for (tokens, 0..) |t, i| batchAdd(&b, t, @intCast(i), &.{0}, i + 1 == tokens.len);
        var ret = A.llama_decode(ctx, b);
        note("decode prompt: {d}", .{ret});
        if (ret != 0) return;

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
        note("seq_rm tail: {}", .{A.llama_memory_seq_rm(mem, 0, pos - 2, -1)});
        State.positions(mem, n_seq);

        if (cs.multi_seq and n_seq >= 2) {
            // Two sequences in one batch, interleaved, and a token shared
            // by both.
            // Sequence 1 starts fresh beside sequence 0 (or 2, where there
            // is one), each at its own positions.
            const other: i32 = if (n_seq >= 3) 2 else 0;
            var other_pos: i32 = if (other == 0) pos else 0;
            b.n_tokens = 0;
            for (0..6) |k| {
                const first = k % 2 == 0;
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

        A.llama_memory_clear(mem, true);
        note("cleared", .{});
        State.positions(mem, n_seq);
    }

    const perf = A.llama_perf_context(ctx);
    note("perf n_p_eval {d} n_eval {d} n_reused {d}", .{ perf.n_p_eval, perf.n_eval, perf.n_reused });
}

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
    .{ .name = "n_ubatch 7", .edit = Edits.smallUbatch },
    .{ .name = "two sequences", .edit = Edits.twoSeqs, .multi_seq = true },
    .{ .name = "three sequences, unified", .edit = Edits.threeSeqsUnified, .multi_seq = true },
    .{ .name = "n_rs_seq 4", .edit = Edits.rsSeq, .multi_seq = true },
    .{ .name = "embeddings", .edit = Edits.embd },
    .{ .name = "embeddings, mean", .edit = Edits.embdMean },
    .{ .name = "embeddings, last", .edit = Edits.embdLast },
    .{ .name = "q8_0 cache", .edit = Edits.quantKv },
    .{ .name = "q4_0 V without FA", .edit = Edits.quantVNoFa },
    .{ .name = "no KQV offload", .edit = Edits.noOffloadKqv, .tensor_ok = false },
    .{ .name = "no op offload", .edit = Edits.noOpOffload },
    .{ .name = "one thread", .edit = Edits.oneThread },
    .{ .name = "n_ctx 40", .edit = Edits.tinyCtx },
    .{ .name = "rope override, yarn", .edit = Edits.ropeOverride },
    .{ .name = "non-causal", .edit = Edits.nonCausal },
    .{ .name = "n_outputs_max 5", .edit = Edits.outputsMax },
    .{ .name = "n_batch 0", .edit = Edits.badBatch },
    .{ .name = "n_seq_max 300", .edit = Edits.tooManySeqs },
};

const ModelCfg = struct {
    name: []const u8,
    edit: *const fn (*c.llama_model_params) void,
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
    fn tensorSplit(p: *c.llama_model_params) void {
        tensor_devices[0] = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_GPU);
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

const model_cfgs = [_]ModelCfg{
    .{ .name = "gpu", .edit = ModelEdits.gpu },
    .{ .name = "cpu", .edit = ModelEdits.cpu },
    .{ .name = "partial", .edit = ModelEdits.partial },
    .{ .name = "proxy", .edit = ModelEdits.proxy },
    .{ .name = "norm-on-cpu", .edit = ModelEdits.normOnCpu },
    .{ .name = "tensor", .edit = ModelEdits.tensorSplit, .nodes = false },
    .{ .name = "tensor-uneven", .edit = ModelEdits.tensorSplitUneven, .nodes = false },
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: context_diff <model.gguf> [--quick] [--model gpu|cpu|partial|proxy] [--case substring]\n", .{});
        std.process.exit(2);
    }
    const path = try gpa.dupeSentinel(u8, args[1], 0);
    var quick = false;
    var only: ?[]const u8 = null;
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
    }

    // CONTEXT_DIFF_OUT=<dir> keeps both transcripts of every case.
    const out_dir: ?[]const u8 = if (std.c.getenv("CONTEXT_DIFF_OUT")) |d| std.mem.span(d) else null;
    // CONTEXT_DIFF_TMP=<dir> holds the state files each side writes.
    if (std.c.getenv("CONTEXT_DIFF_TMP")) |d| tmp_dir = std.mem.span(d);
    live = std.c.getenv("CONTEXT_DIFF_LIVE") != null;
    const port_first = std.c.getenv("CONTEXT_DIFF_PORT_FIRST") != null;

    c.llama_log_set(logCb, null);

    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    for (model_cfgs) |mc| {
        if (only_model) |o| {
            if (!std.mem.eql(u8, mc.name, o)) continue;
        } else if (quick and !std.mem.eql(u8, mc.name, "gpu")) continue;
        var mp = c.llama_model_default_params();
        mc.edit(&mp);
        // What each side logs while loading, compared below.
        var ref_load: std.ArrayList(u8) = .empty;
        defer ref_load.deinit(gpa);
        var port_load: std.ArrayList(u8) = .empty;
        defer port_load.deinit(gpa);
        rec = &ref_load;
        const rm = c.llama_model_load_from_file(path.ptr, mp) orelse {
            std.debug.print("FAIL: the C++ cannot load {s}\n", .{path});
            std.process.exit(1);
        };
        defer c.llama_model_free(rm);
        var splits: std.ArrayList([:0]const u8) = .empty;
        defer {
            for (splits.items) |s| gpa.free(s);
            splits.deinit(gpa);
        }
        rec = &port_load;
        const pm = M.loadFromFileImpl(.{ .path = path }, &splits, null, null, mp) orelse {
            std.debug.print("FAIL: the port cannot load {s}\n", .{path});
            std.process.exit(1);
        };
        defer M.destroy(pm);
        rec = null;
        {
            var rn = normalize(ref_load.items);
            defer rn.deinit(gpa);
            var pn = normalize(port_load.items);
            defer pn.deinit(gpa);
            var nb: [64]u8 = undefined;
            const before = failures;
            _ = firstDiff(try std.fmt.bufPrint(&nb, "{s}: load", .{mc.name}), rn.items, pn.items);
            std.debug.print("  {s:<40} {s:<4} {d:>7} lines\n", .{ try std.fmt.bufPrint(&nb, "{s}: load", .{mc.name}), if (failures == before) "ok" else "FAIL", std.mem.count(u8, rn.items, "\n") });
        }

        if (std.mem.startsWith(u8, mc.name, "tensor")) splitStates(rm, pm);

        const vocab = c.llama_model_get_vocab(rm);
        var tokens: [256]i32 = undefined;
        const nt = c.llama_tokenize(vocab, prompt_text, prompt_text.len, &tokens, tokens.len, true, false);
        if (nt <= 8) {
            std.debug.print("FAIL: tokenizing the prompt gave {d}\n", .{nt});
            std.process.exit(1);
        }
        const n_vocab: usize = @intCast(c.llama_vocab_n_tokens(vocab));
        const n_embd: usize = @intCast(c.llama_model_n_embd_out(rm));

        // A first context initializes what each backend initializes once
        // per process, which logs.
        {
            var p = c.llama_context_default_params();
            p.n_ctx = 256;
            if (c.llama_init_from_model(rm, p)) |ctx| c.llama_free(ctx);
        }

        for (cases) |cs| {
            if (only) |o| if (std.mem.find(u8, cs.name, o) == null) continue;
            if (quick and !std.mem.eql(u8, cs.name, "default") and !cs.multi_seq) continue;
            if (!cs.tensor_ok and std.mem.startsWith(u8, mc.name, "tensor")) continue;
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
    }

    std.debug.print("context-diff: {d} checks, {d} failures\n", .{ checks, failures });
    if (failures != 0) std.process.exit(1);
}
