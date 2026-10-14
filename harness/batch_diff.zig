//! Compares `src/llama/batch.zig` with llama.cpp's `llama-batch.cpp`.
//!
//! **Not a port.** The harness behind `scripts/batch-diff`. It drives the
//! C++ `llama_batch_allocr` in `libllama.a` (through `harness/batch_ref.cpp`)
//! and the port's `BatchAllocr` side by side, one allocator of each per
//! session, on the same input batches, the same vocabulary and the same
//! memory -- a table both sides' `seq_pos_min`/`seq_pos_max` read -- and
//! compares:
//!
//! - **everything logged**, level included, with `%p` values masked to
//!   null or non-null: the errors and warning `init` raises, and under
//!   `LLAMA_BATCH_DEBUG` the whole batch and ubatch dumps;
//! - `init`'s answer, `get_n_tokens`, `get_n_outputs`, and `seq_pos_min` and
//!   `seq_pos_max` of all 256 sequences, after every `init`, failed or not;
//! - the sanitized batch: which arrays are still the caller's, and the
//!   contents of the positions, sequence ids and output flags filled in;
//! - every ubatch of `split_simple`, `split_equal` (sequential and not, with
//!   and without `n_keep_tail`) and `split_seq` until the batch is consumed,
//!   and `ubatch_reserve`'s: every scalar, which pointers are null, and every
//!   array element -- compared only after the whole split has run, so a
//!   ubatch whose data did not outlive the next split fails;
//! - `get_n_used` after every split and `get_out_ids` after each mode.
//!
//! Sessions run several batches through one allocator, so state one batch
//! leaves behind is seen by the next. Inputs are hand-built cases for each
//! error path and random batches: one sequence or many, independent,
//! coupled or sharing a prefix; positions given or generated, from memory or
//! not; tokens or embeddings; one position per token or M-RoPE's four; and
//! a mutation -- a gap, a decrease, an invalid id, a diverged memory -- in a
//! third of them. The `GGML_ASSERT`s run in forked children.
//!
//! Usage: batch_diff <vocab.gguf> [seed]

const std = @import("std");
const staged = @import("staged");

const c = staged.impl.c;
const B = staged.batch;
const M = staged.model;

const gpa = std.heap.c_allocator;

const RefUbatch = extern struct {
    b_equal_seqs: u32 = 0,
    n_tokens: u32 = 0,
    n_seq_tokens: u32 = 0,
    n_seqs: u32 = 0,
    n_seqs_unq: u32 = 0,
    n_pos: u32 = 0,
    token: [*c]c.llama_token = null,
    embd: [*c]f32 = null,
    pos: [*c]c.llama_pos = null,
    n_seq_id: [*c]i32 = null,
    seq_id: [*c][*c]c.llama_seq_id = null,
    seq_id_unq: [*c]c.llama_seq_id = null,
    seq_idx: [*c]i32 = null,
    output: [*c]i8 = null,
    holder: ?*anyopaque = null,
};

extern fn ref_mem_set(s: c_int, p_min: c.llama_pos, p_max: c.llama_pos) void;
extern fn ref_mem_min(s: c_int) c.llama_pos;
extern fn ref_mem_max(s: c_int) c.llama_pos;
extern fn ref_new(n_pos_per_embd: u32) *anyopaque;
extern fn ref_delete(a: *anyopaque) void;
extern fn ref_init(a: *anyopaque, batch: *const c.llama_batch, vocab: *const c.llama_vocab, use_memory: c_int, n_embd: u32, n_seq_max: u32, output_all: c_int) c_int;
extern fn ref_get_batch(a: *anyopaque, out: *c.llama_batch) void;
extern fn ref_n_tokens(a: *anyopaque) u32;
extern fn ref_n_outputs(a: *anyopaque) u32;
extern fn ref_n_used(a: *anyopaque) u32;
extern fn ref_out_ids(a: *anyopaque, ids: *[*c]const i32) usize;
extern fn ref_seq_pos_min(a: *anyopaque, s: c.llama_seq_id) c.llama_pos;
extern fn ref_seq_pos_max(a: *anyopaque, s: c.llama_seq_id) c.llama_pos;
extern fn ref_split_reset(a: *anyopaque) void;
extern fn ref_split_simple(a: *anyopaque, n_ubatch: u32, out: *RefUbatch) void;
extern fn ref_split_equal(a: *anyopaque, n_ubatch: u32, sequential: c_int, n_keep_tail: u32, out: *RefUbatch) void;
extern fn ref_split_seq(a: *anyopaque, n_ubatch: u32, out: *RefUbatch) void;
extern fn ref_ubatch_reserve(a: *anyopaque, n_seq_tokens: u32, n_seqs: u32, out: *RefUbatch) void;
extern fn ref_ubatch_free(u: *RefUbatch) void;
extern fn ref_ubatch_owned(u: *const RefUbatch) c_int;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;
extern "c" fn fork() c_int;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn dup2(old: c_int, new: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;

const sigabrt = 6;

var checks: usize = 0;
var results: usize = 0;
var failures: usize = 0;
var session_failures: usize = 0;
var session_name: []const u8 = "";

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    session_failures += 1;
    if (session_failures <= 4 and failures <= 40) std.debug.print("FAIL [{s}] " ++ fmt ++ "\n", .{session_name} ++ args);
}

fn expectInt(what: []const u8, ref: i128, port: i128) bool {
    checks += 1;
    if (ref != port) {
        fail("{s}: ref {d} port {d}", .{ what, ref, port });
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Logs

var log_target: ?*std.ArrayList(u8) = null;

fn logCb(level: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const t = log_target orelse return;
    t.print(gpa, "<{d}>", .{level}) catch unreachable;
    // `%p` prints a heap address, which differs between the sides; keep
    // only whether it is null.
    const s = std.mem.span(@as([*:0]const u8, @ptrCast(text)));
    var i: usize = 0;
    while (i < s.len) {
        if (i + 2 < s.len and s[i] == '0' and s[i + 1] == 'x' and std.ascii.isHex(s[i + 2])) {
            var j = i + 2;
            while (j < s.len and std.ascii.isHex(s[j])) j += 1;
            const zero = std.mem.allEqual(u8, s[i + 2 .. j], '0');
            t.appendSlice(gpa, if (zero) "NULL" else "PTR") catch unreachable;
            i = j;
        } else {
            t.append(gpa, s[i]) catch unreachable;
            i += 1;
        }
    }
}

fn firstDiff(a: []const u8, b: []const u8) void {
    checks += 1;
    if (std.mem.eql(u8, a, b)) return;
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            fail("log line {d}:\n    ref  '{s}'\n    port '{s}'", .{ line, la orelse "<end>", lb orelse "<end>" });
            return;
        }
    }
}

/// The distinct error and warning messages the reference raised, by their
/// first words, to show which paths the inputs reached.
var reached: std.StringArrayHashMapUnmanaged(usize) = .empty;

fn noteReached(log: []const u8) void {
    var it = std.mem.splitScalar(u8, log, '\n');
    while (it.next()) |line| {
        const warn = std.mem.startsWith(u8, line, "<3>");
        const err = std.mem.startsWith(u8, line, "<4>");
        if (!warn and !err) continue;
        // `<4>init: the tokens of sequence 3 in ...`: strip the numbers so
        // one path is one key.
        var key: std.ArrayList(u8) = .empty;
        for (line[0..@min(line.len, 90)]) |ch| {
            if (!std.ascii.isDigit(ch)) key.append(gpa, ch) catch unreachable;
        }
        const gop = reached.getOrPut(gpa, key.items) catch unreachable;
        if (gop.found_existing) key.deinit(gpa);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
}

// -----------------------------------------------------------------------------
// The memory both sides read

fn memMin(_: *anyopaque, s: c.llama_seq_id) c.llama_pos {
    return ref_mem_min(s);
}

fn memMax(_: *anyopaque, s: c.llama_seq_id) c.llama_pos {
    return ref_mem_max(s);
}

var mem_ctx: u8 = 0;
const port_memory: B.MemoryPos = .{ .ctx = &mem_ctx, .seq_pos_min = memMin, .seq_pos_max = memMax };

fn memClear() void {
    for (0..B.max_seq) |s| ref_mem_set(@intCast(s), -1, -1);
}

// -----------------------------------------------------------------------------
// Inputs

const Input = struct {
    batch: c.llama_batch,
    n_embd: u32,
    n_seq_max: u32,
    output_all: bool,
    use_memory: bool,
};

/// A batch under construction, its arrays in an arena.
const Builder = struct {
    ar: std.mem.Allocator,
    token: ?[]c.llama_token = null,
    embd: ?[]f32 = null,
    pos: ?[]c.llama_pos = null,
    n_seq_id: ?[]i32 = null,
    seq_id: ?[][*c]c.llama_seq_id = null,
    logits: ?[]i8 = null,
    n: usize,

    fn cbatch(self: *const Builder) c.llama_batch {
        return .{
            .n_tokens = @intCast(self.n),
            .token = if (self.token) |t| t.ptr else null,
            .embd = if (self.embd) |t| t.ptr else null,
            .pos = if (self.pos) |t| t.ptr else null,
            .n_seq_id = if (self.n_seq_id) |t| t.ptr else null,
            .seq_id = if (self.seq_id) |t| t.ptr else null,
            .logits = if (self.logits) |t| t.ptr else null,
        };
    }

    /// Sets token `i`'s sequences; there is always room for at least one,
    /// so `seq_id[i][0]` is readable even when `n_seq_id[i]` is 0.
    fn setSeqs(self: *Builder, i: usize, ids: []const c.llama_seq_id) void {
        const store = self.ar.alloc(c.llama_seq_id, @max(ids.len, 1) + 2) catch unreachable;
        @memset(store, 0);
        @memcpy(store[0..ids.len], ids);
        self.seq_id.?[i] = store.ptr;
        self.n_seq_id.?[i] = @intCast(ids.len);
    }

    fn withSeqArrays(self: *Builder) void {
        self.n_seq_id = self.ar.alloc(i32, self.n) catch unreachable;
        const s = self.ar.alloc([*c]c.llama_seq_id, self.n + 1) catch unreachable;
        @memset(s, null);
        self.seq_id = s;
    }
};

const Gen = struct {
    r: std.Random,
    ar: std.mem.Allocator,
    n_vocab: u32,

    fn below(self: Gen, n: usize) usize {
        return self.r.uintLessThan(usize, n);
    }

    fn chance(self: Gen, pct: usize) bool {
        return self.below(100) < pct;
    }

    fn pick(self: Gen, comptime T: type, items: []const T) T {
        return items[self.below(items.len)];
    }

    /// One random input batch for an allocator with `n_pos_per_embd`
    /// positions per token. Sets the memory table as a side effect.
    fn input(self: Gen, n_pos_per_embd: u32) Input {
        memClear();

        const embd_input = self.chance(25);
        const n_embd: u32 = if (embd_input) @intCast(1 + self.below(6)) else @intCast(self.below(8));
        var n_seq_max: u32 = self.pick(u32, &.{ 1, 1, 2, 3, 4, 4, 5, 8, 16, 64, 256 });
        const output_all = self.chance(20);
        const use_memory = self.chance(70);

        // the sequences in play
        const k = 1 + self.below(@min(n_seq_max, 6));
        var seqs: [6]c.llama_seq_id = undefined;
        if (self.chance(50)) {
            for (0..k) |i| seqs[i] = @intCast(i);
        } else {
            var n: usize = 0;
            while (n < k) {
                const s: c.llama_seq_id = @intCast(self.below(n_seq_max));
                if (std.mem.findScalar(c.llama_seq_id, seqs[0..n], s) == null) {
                    seqs[n] = s;
                    n += 1;
                }
            }
        }
        const chosen = seqs[0..k];

        // the memory, and each sequence's next position
        var next: [B.max_seq]c.llama_pos = @splat(0);
        const base: c.llama_pos = if (self.chance(50)) 0 else @intCast(self.below(10));
        for (0..B.max_seq) |s| next[s] = base;
        if (use_memory) {
            for (chosen) |s| {
                if (self.chance(40)) continue;
                const lo: c.llama_pos = @intCast(self.below(20));
                const hi = lo + @as(c.llama_pos, @intCast(self.below(30)));
                ref_mem_set(s, lo, hi);
                next[@intCast(s)] = hi + 1;
            }
            // an unrelated sequence with a history
            if (self.chance(30)) {
                const s: c_int = @intCast(self.below(n_seq_max));
                if (std.mem.findScalar(c.llama_seq_id, chosen, s) == null) ref_mem_set(s, 3, 7);
            }
        } else {
            for (chosen) |s| next[@intCast(s)] = base;
        }

        // the token stream: a list of (sequence set, position) by stream
        const Tok = struct { set: []c.llama_seq_id, pos: c.llama_pos };
        var streams: std.ArrayList(std.ArrayList(Tok)) = .empty;
        var prefix: std.ArrayList(Tok) = .empty;

        const scale: usize = if (self.chance(5)) 40 else if (self.chance(20)) 12 else 6;
        const jumps = n_pos_per_embd > 1 and self.chance(40);

        const style = if (k == 1) 3 else self.below(3);
        const sync_memory = !self.chance(15);

        // groups: sets of sequences that move together
        var groups: std.ArrayList([]c.llama_seq_id) = .empty;
        switch (style) {
            1 => {
                var i: usize = 0;
                while (i < k) {
                    const g = @min(k - i, 1 + self.below(3));
                    groups.append(self.ar, chosen[i .. i + g]) catch unreachable;
                    i += g;
                }
            },
            else => for (0..k) |i| groups.append(self.ar, chosen[i .. i + 1]) catch unreachable,
        }

        // coupled sequences must agree on their memory to be valid
        const syncGroup = struct {
            fn f(g: []const c.llama_seq_id, nx: *[B.max_seq]c.llama_pos) void {
                const lo = ref_mem_min(g[0]);
                const hi = ref_mem_max(g[0]);
                for (g[1..]) |s| {
                    ref_mem_set(s, lo, hi);
                    nx[@intCast(s)] = nx[@intCast(g[0])];
                }
            }
        }.f;

        if (style == 2) {
            // a prefix shared by a subset of two or more, then each alone
            const np = 2 + self.below(k - 1);
            const members = chosen[0..np];
            if (sync_memory) syncGroup(members, &next);
            const len = 1 + self.below(scale);
            for (0..len) |_| {
                const set = self.ar.dupe(c.llama_seq_id, members) catch unreachable;
                if (self.chance(30)) self.r.shuffle(c.llama_seq_id, set);
                const p = next[@intCast(members[0])];
                prefix.append(self.ar, .{ .set = set, .pos = p }) catch unreachable;
                for (members) |s| next[@intCast(s)] = p + 1;
            }
        }

        for (groups.items) |g| {
            if (g.len > 1 and sync_memory) syncGroup(g, &next);
            var list: std.ArrayList(Tok) = .empty;
            const len = (if (style == 2) self.below(scale) else 1 + self.below(scale)) * (if (style == 3) @as(usize, 4) else 1);
            for (0..len) |_| {
                const set = self.ar.dupe(c.llama_seq_id, g) catch unreachable;
                if (set.len > 1 and self.chance(30)) self.r.shuffle(c.llama_seq_id, set);
                const p = next[@intCast(g[0])];
                list.append(self.ar, .{ .set = set, .pos = p }) catch unreachable;
                const step: c.llama_pos = if (jumps and self.chance(30)) @intCast(1 + self.below(3)) else 1;
                for (g) |s| next[@intCast(s)] = p + step;
            }
            streams.append(self.ar, list) catch unreachable;
        }

        // interleave the streams, each in order, the prefix first
        var toks: std.ArrayList(Tok) = .empty;
        toks.appendSlice(self.ar, prefix.items) catch unreachable;
        const blocked = self.chance(30);
        var heads = self.ar.alloc(usize, streams.items.len) catch unreachable;
        @memset(heads, 0);
        while (true) {
            var live: usize = 0;
            for (streams.items, 0..) |st, i| live += @intFromBool(heads[i] < st.items.len);
            if (live == 0) break;
            var which = self.below(live);
            for (streams.items, 0..) |st, i| {
                if (heads[i] >= st.items.len) continue;
                if (blocked or which == 0) {
                    toks.append(self.ar, st.items[heads[i]]) catch unreachable;
                    heads[i] += 1;
                    break;
                }
                which -= 1;
            }
        }
        if (toks.items.len == 0) {
            const set = self.ar.dupe(c.llama_seq_id, chosen[0..1]) catch unreachable;
            toks.append(self.ar, .{ .set = set, .pos = next[@intCast(chosen[0])] }) catch unreachable;
        }

        const n = toks.items.len;
        var b: Builder = .{ .ar = self.ar, .n = n };

        if (embd_input) {
            const e = self.ar.alloc(f32, n * n_embd) catch unreachable;
            for (e) |*x| x.* = self.r.float(f32) * 2 - 1;
            b.embd = e;
        } else {
            const t = self.ar.alloc(c.llama_token, n) catch unreachable;
            for (t) |*x| x.* = @intCast(self.below(self.n_vocab));
            b.token = t;
        }

        // positions: given or generated
        const must_pos = embd_input and n_pos_per_embd > 1;
        if (must_pos or !self.chance(35)) {
            const npos = if (embd_input) n * n_pos_per_embd else n;
            const p = self.ar.alloc(c.llama_pos, npos) catch unreachable;
            for (toks.items, 0..) |t, i| p[i] = t.pos;
            for (p[n..]) |*x| x.* = @intCast(self.below(100));
            b.pos = p;
        }

        // sequence ids: given, or (one sequence, 0) left to the allocator
        const only_zero = k == 1 and chosen[0] == 0;
        if (!(only_zero and self.chance(50))) {
            b.withSeqArrays();
            for (toks.items, 0..) |t, i| b.setSeqs(i, t.set);
        } else if (self.chance(30)) {
            const ns = self.ar.alloc(i32, n) catch unreachable;
            for (ns) |*x| x.* = if (self.chance(10)) 0 else 1;
            b.n_seq_id = ns;
        }

        // output flags
        switch (self.below(10)) {
            0, 1, 2 => {},
            3, 4, 5 => {
                const l = self.ar.alloc(i8, n) catch unreachable;
                for (l) |*x| x.* = if (self.chance(30)) 1 else 0;
                b.logits = l;
            },
            6, 7 => {
                const l = self.ar.alloc(i8, n) catch unreachable;
                @memset(l, 1);
                b.logits = l;
            },
            else => {
                const l = self.ar.alloc(i8, n) catch unreachable;
                @memset(l, 0);
                b.logits = l;
            },
        }

        // mutations
        if (self.chance(35)) {
            const n_mut = 1 + self.below(2);
            for (0..n_mut) |_| {
                const i = self.below(n);
                switch (self.below(11)) {
                    0 => if (b.pos) |p| {
                        const j = self.below(n);
                        std.mem.swap(c.llama_pos, &p[i], &p[j]);
                    },
                    1 => if (b.pos) |p| {
                        p[i] += @intCast(1 + self.below(3));
                    },
                    2 => if (b.pos) |p| {
                        p[i] -= 1;
                    },
                    3 => if (b.seq_id) |s| {
                        if (b.n_seq_id.?[i] > 0) {
                            const at = self.below(@intCast(b.n_seq_id.?[i]));
                            s[i][at] = if (self.chance(30)) -1 else @as(c.llama_seq_id, @intCast(n_seq_max + self.below(3)));
                        }
                    },
                    4 => if (b.token) |t| {
                        t[i] = self.pick(c.llama_token, &.{ -1, @intCast(self.n_vocab), @intCast(self.n_vocab + 5), std.math.minInt(i32) });
                    },
                    5 => {
                        const s = chosen[self.below(k)];
                        const d: c.llama_pos = @as(c.llama_pos, @intCast(self.below(5))) - 2;
                        ref_mem_set(s, ref_mem_min(s), ref_mem_max(s) + d);
                    },
                    6 => if (b.seq_id != null) {
                        // keeps seq_id[i][0], which the C++ still reads
                        b.n_seq_id.?[i] = 0;
                    },
                    7 => n_seq_max = self.pick(u32, &.{ 257, 1000 }),
                    8 => if (b.seq_id != null) {
                        // join another sequence to this token
                        const old = b.seq_id.?[i][0..@intCast(b.n_seq_id.?[i])];
                        const extra: c.llama_seq_id = @intCast(self.below(n_seq_max));
                        const set = self.ar.alloc(c.llama_seq_id, old.len + 1) catch unreachable;
                        @memcpy(set[0..old.len], old);
                        set[old.len] = extra;
                        b.setSeqs(i, set);
                    },
                    9 => if (b.seq_id != null and b.n_seq_id.?[i] > 0) {
                        // a sequence named twice in one token
                        const old = b.seq_id.?[i][0..@intCast(b.n_seq_id.?[i])];
                        const set = self.ar.alloc(c.llama_seq_id, old.len + 1) catch unreachable;
                        @memcpy(set[0..old.len], old);
                        set[old.len] = old[0];
                        b.setSeqs(i, set);
                    },
                    else => if (b.pos != null and b.seq_id == null and b.n_seq_id == null) {
                        // no sequences allowed at all, with sequence 0 implied
                        n_seq_max = 0;
                    },
                }
            }
        }

        return .{ .batch = b.cbatch(), .n_embd = n_embd, .n_seq_max = n_seq_max, .output_all = output_all, .use_memory = use_memory };
    }
};

// -----------------------------------------------------------------------------
// Comparing

fn sameNull(what: []const u8, a: anytype, b: anytype) bool {
    checks += 1;
    if ((a == null) != (b == null)) {
        fail("{s}: ref {s} port {s}", .{ what, if (a == null) "null" else "set", if (b == null) "null" else "set" });
        return false;
    }
    return a != null;
}

fn sameArray(comptime T: type, what: []const u8, a: [*c]const T, b: [*c]const T, n: usize) void {
    checks += 1;
    for (0..n) |i| {
        const x = std.mem.asBytes(&a[i]);
        const y = std.mem.asBytes(&b[i]);
        if (!std.mem.eql(u8, x, y)) {
            fail("{s}[{d}]: ref {any} port {any}", .{ what, i, a[i], b[i] });
            return;
        }
    }
}

fn compareUbatch(what: []const u8, r: *const RefUbatch, p: *const B.Ubatch, n_embd: u32) void {
    results += 1;
    const before = failures;
    _ = expectInt("b_equal_seqs", r.b_equal_seqs, p.b_equal_seqs);
    _ = expectInt("n_tokens", r.n_tokens, p.n_tokens);
    _ = expectInt("n_seq_tokens", r.n_seq_tokens, p.n_seq_tokens);
    _ = expectInt("n_seqs", r.n_seqs, p.n_seqs);
    _ = expectInt("n_seqs_unq", r.n_seqs_unq, p.n_seqs_unq);
    _ = expectInt("n_pos", r.n_pos, p.n_pos);
    _ = expectInt("owns its data", ref_ubatch_owned(r), @intFromBool(p.data != null));
    if (failures != before) {
        fail("  in {s}", .{what});
        return;
    }
    const n = r.n_tokens;
    if (sameNull("token", r.token, p.token)) sameArray(c.llama_token, "token", r.token, p.token, n);
    if (sameNull("embd", r.embd, p.embd)) sameArray(f32, "embd", r.embd, p.embd, @as(usize, n) * n_embd);
    if (sameNull("pos", r.pos, p.pos)) sameArray(c.llama_pos, "pos", r.pos, p.pos, @as(usize, n) * r.n_pos);
    const have_nsid = sameNull("n_seq_id", r.n_seq_id, p.n_seq_id);
    if (have_nsid) sameArray(i32, "n_seq_id", r.n_seq_id, p.n_seq_id, n);
    if (sameNull("seq_id", r.seq_id, p.seq_id) and have_nsid) {
        for (0..n) |i| {
            if (!sameNull("seq_id[i]", r.seq_id[i], p.seq_id[i])) continue;
            sameArray(c.llama_seq_id, "seq_id[i]", r.seq_id[i], p.seq_id[i], @intCast(@max(0, r.n_seq_id[i])));
        }
    }
    if (sameNull("seq_id_unq", r.seq_id_unq, p.seq_id_unq)) sameArray(c.llama_seq_id, "seq_id_unq", r.seq_id_unq, p.seq_id_unq, r.n_seqs_unq);
    if (sameNull("seq_idx", r.seq_idx, p.seq_idx)) sameArray(i32, "seq_idx", r.seq_idx, p.seq_idx, B.max_seq);
    if (sameNull("output", r.output, p.output)) sameArray(i8, "output", r.output, p.output, n);
    if (failures != before) fail("  in {s}", .{what});
}

fn compareState(ra: *anyopaque, pa: *B.BatchAllocr) void {
    _ = expectInt("get_n_tokens", ref_n_tokens(ra), pa.getNTokens());
    _ = expectInt("get_n_outputs", ref_n_outputs(ra), pa.getNOutputs());
    for (0..B.max_seq) |s| {
        const sid: c.llama_seq_id = @intCast(s);
        if (!expectInt("seq_pos_min", ref_seq_pos_min(ra, sid), pa.seqPosMin(sid))) return;
        if (!expectInt("seq_pos_max", ref_seq_pos_max(ra, sid), pa.seqPosMax(sid))) return;
    }
}

fn compareSanitized(ra: *anyopaque, pa: *B.BatchAllocr, in: c.llama_batch) void {
    results += 1;
    var r: c.llama_batch = undefined;
    ref_get_batch(ra, &r);
    const p = pa.getBatch().*;
    const n: usize = @intCast(r.n_tokens);
    _ = expectInt("batch.n_tokens", r.n_tokens, p.n_tokens);
    checks += 1;
    if (r.token != in.token or p.token != in.token) fail("batch.token is not the caller's", .{});
    if (r.embd != in.embd or p.embd != in.embd) fail("batch.embd is not the caller's", .{});
    // a given array stays the caller's on both sides; a generated one is
    // compared by content
    inline for (.{ "pos", "n_seq_id", "seq_id", "logits" }) |f| {
        checks += 1;
        const given = @field(in, f) != null;
        if (given and (@field(r, f) != @field(in, f) or @field(p, f) != @field(in, f))) {
            // `logits` is replaced when output_all overrides the flags
            if ((@field(r, f) == @field(in, f)) != (@field(p, f) == @field(in, f))) fail("batch.{s}: one side kept the caller's array", .{f});
        }
    }
    sameArray(c.llama_pos, "batch.pos", r.pos, p.pos, n);
    sameArray(i32, "batch.n_seq_id", r.n_seq_id, p.n_seq_id, n);
    for (0..n) |i| sameArray(c.llama_seq_id, "batch.seq_id[i]", r.seq_id[i], p.seq_id[i], @intCast(@max(0, r.n_seq_id[i])));
    if (in.seq_id == null) {
        checks += 1;
        if (r.seq_id[n] != null or p.seq_id[n] != null) fail("batch.seq_id is not null-terminated", .{});
    }
    sameArray(i8, "batch.logits", r.logits, p.logits, n);
}

fn compareOutIds(what: []const u8, ra: *anyopaque, pa: *B.BatchAllocr) void {
    var ids: [*c]const i32 = null;
    const n = ref_out_ids(ra, &ids);
    const port = pa.getOutIds().items;
    if (!expectInt("out_ids size", @intCast(n), @intCast(port.len))) {
        fail("  after {s}", .{what});
        return;
    }
    sameArray(i32, "out_ids", ids, port.ptr, n);
}

const Split = union(enum) {
    simple: u32,
    equal: struct { n_ubatch: u32, sequential: bool, n_keep_tail: u32 },
    seq: u32,
};

/// Runs one split mode to exhaustion on both sides, in lockstep, then
/// compares every ubatch -- after the last, so each must have kept its data.
fn runSplit(ra: *anyopaque, pa: *B.BatchAllocr, sp: Split, n_embd: u32, ref_log: *std.ArrayList(u8), port_log: *std.ArrayList(u8), reset: bool) void {
    if (reset) {
        ref_split_reset(ra);
        pa.splitReset();
    }

    var rl: std.ArrayList(RefUbatch) = .empty;
    defer rl.deinit(gpa);
    var pl: std.ArrayList(B.Ubatch) = .empty;
    defer pl.deinit(gpa);

    const cap = @as(usize, ref_n_tokens(ra)) * 2 + 4;
    var it: usize = 0;
    while (it < cap) : (it += 1) {
        var r: RefUbatch = .{};
        var p: B.Ubatch = undefined;
        log_target = ref_log;
        switch (sp) {
            .simple => |n| ref_split_simple(ra, n, &r),
            .equal => |e| ref_split_equal(ra, e.n_ubatch, @intFromBool(e.sequential), e.n_keep_tail, &r),
            .seq => |n| ref_split_seq(ra, n, &r),
        }
        log_target = port_log;
        p = switch (sp) {
            .simple => |n| pa.splitSimple(n),
            .equal => |e| pa.splitEqual(e.n_ubatch, e.sequential, e.n_keep_tail),
            .seq => |n| pa.splitSeq(n),
        };
        log_target = null;
        rl.append(gpa, r) catch unreachable;
        pl.append(gpa, p) catch unreachable;
        _ = expectInt("get_n_used", ref_n_used(ra), pa.getNUsed());
        if (r.n_tokens == 0 and p.n_tokens == 0) break;
        if ((r.n_tokens == 0) != (p.n_tokens == 0)) {
            fail("{s}: one side is exhausted (ref {d} tokens, port {d})", .{ @tagName(sp), r.n_tokens, p.n_tokens });
            break;
        }
    } else {
        // A token in no sequence passes `init`, and its empty set overlaps
        // nothing, so `split_equal` takes it once per copy of that set and
        // the `n_keep_tail` truncation hands it back: upstream never
        // finishes. Agreeing on that for `cap` ubatches is agreement.
        const b = pa.getBatch().*;
        var seqless = false;
        for (0..@intCast(b.n_tokens)) |i| seqless = seqless or b.n_seq_id[i] <= 0;
        if (seqless and sp == .equal) {
            tally.endless += 1;
        } else {
            fail("{s}: not exhausted after {d} ubatches", .{ @tagName(sp), cap });
        }
    }

    var name_buf: [96]u8 = undefined;
    for (rl.items, pl.items, 0..) |*r, *p, i| {
        const name = switch (sp) {
            .simple => |n| std.fmt.bufPrint(&name_buf, "split_simple({d}) #{d}", .{ n, i }),
            .equal => |e| std.fmt.bufPrint(&name_buf, "split_equal({d}, {}, {d}) #{d}", .{ e.n_ubatch, e.sequential, e.n_keep_tail, i }),
            .seq => |n| std.fmt.bufPrint(&name_buf, "split_seq({d}) #{d}", .{ n, i }),
        } catch "split";
        compareUbatch(name, r, p, n_embd);
        ref_ubatch_free(r);
        p.release();
    }
    compareOutIds(@tagName(sp), ra, pa);
}

fn runReserve(ra: *anyopaque, pa: *B.BatchAllocr, n_seq_tokens: u32, n_seqs: u32, ref_log: *std.ArrayList(u8), port_log: *std.ArrayList(u8)) void {
    var r: RefUbatch = .{};
    log_target = ref_log;
    ref_ubatch_reserve(ra, n_seq_tokens, n_seqs, &r);
    log_target = port_log;
    var p = pa.ubatchReserve(n_seq_tokens, n_seqs);
    log_target = null;
    var name_buf: [64]u8 = undefined;
    compareUbatch(std.fmt.bufPrint(&name_buf, "ubatch_reserve({d}, {d})", .{ n_seq_tokens, n_seqs }) catch "ubatch_reserve", &r, &p, 0);
    ref_ubatch_free(&r);
    p.release();
    compareState(ra, pa);
    _ = expectInt("get_n_used", ref_n_used(ra), pa.getNUsed());
    compareOutIds("ubatch_reserve", ra, pa);
}

/// The vocabulary, on both sides.
const Vocabs = struct {
    ref: *const c.llama_vocab,
    port: *const staged.vocab.Vocab,
};

/// Runs `init` on both allocators and compares; on success, runs every
/// split mode.
fn runBatch(g: ?Gen, ra: *anyopaque, pa: *B.BatchAllocr, v: Vocabs, in: Input, ref_log: *std.ArrayList(u8), port_log: *std.ArrayList(u8)) void {
    results += 1;
    log_target = ref_log;
    const rok = ref_init(ra, &in.batch, v.ref, @intFromBool(in.use_memory), in.n_embd, in.n_seq_max, @intFromBool(in.output_all));
    log_target = port_log;
    const pok = pa.initBatch(in.batch, v.port, if (in.use_memory) port_memory else null, in.n_embd, in.n_seq_max, in.output_all);
    log_target = null;
    if (!expectInt("init", rok, @intFromBool(pok))) return;
    compareState(ra, pa);
    if (rok == 0) {
        tally.rejected += 1;
        return;
    }
    tally.accepted += 1;
    compareSanitized(ra, pa, in.batch);

    const sizes = [_]u32{ 1, 2, 3, 4, 5, 7, 8, 16, 64, 1000 };
    const r = g;
    const pickN = struct {
        fn f(rr: ?Gen, i: usize) u32 {
            return if (rr) |x| x.pick(u32, &sizes) else sizes[i % sizes.len];
        }
    }.f;

    runSplit(ra, pa, .{ .simple = pickN(r, 2) }, in.n_embd, ref_log, port_log, true);
    runSplit(ra, pa, .{ .equal = .{ .n_ubatch = pickN(r, 3), .sequential = false, .n_keep_tail = 0 } }, in.n_embd, ref_log, port_log, true);
    runSplit(ra, pa, .{ .equal = .{ .n_ubatch = pickN(r, 4), .sequential = true, .n_keep_tail = 0 } }, in.n_embd, ref_log, port_log, true);
    {
        const keep: u32 = if (r) |x| @intCast(1 + x.below(3)) else 2;
        var nu = pickN(r, 6);
        if (nu <= keep) nu = keep + 1 + (if (r) |x| @as(u32, @intCast(x.below(4))) else 0);
        const seq = if (r) |x| x.chance(50) else false;
        runSplit(ra, pa, .{ .equal = .{ .n_ubatch = nu, .sequential = seq, .n_keep_tail = keep } }, in.n_embd, ref_log, port_log, true);
    }
    runSplit(ra, pa, .{ .seq = pickN(r, 5) }, in.n_embd, ref_log, port_log, true);
    // two modes on one reset: whatever the first left unused
    if (r) |x| {
        if (x.chance(30)) {
            ref_split_reset(ra);
            pa.splitReset();
            var rr: RefUbatch = .{};
            ref_split_simple(ra, 2, &rr);
            var pp = pa.splitSimple(2);
            compareUbatch("split_simple(2) before split_equal", &rr, &pp, in.n_embd);
            ref_ubatch_free(&rr);
            pp.release();
            runSplit(ra, pa, .{ .equal = .{ .n_ubatch = 3, .sequential = false, .n_keep_tail = 0 } }, in.n_embd, ref_log, port_log, false);
        }
    }
}

const Tally = struct { accepted: usize = 0, rejected: usize = 0, sessions: usize = 0, endless: usize = 0 };
var tally: Tally = .{};

/// One allocator per side, several batches through it, then a reserve and
/// splits on the cleared allocator.
fn session(name: []const u8, n_pos_per_embd: u32, v: Vocabs, g: ?Gen, inputs: []const Input, n_random: usize) void {
    session_name = name;
    session_failures = 0;
    tally.sessions += 1;

    var ref_log: std.ArrayList(u8) = .empty;
    defer ref_log.deinit(gpa);
    var port_log: std.ArrayList(u8) = .empty;
    defer port_log.deinit(gpa);

    const ra = ref_new(n_pos_per_embd);
    defer ref_delete(ra);
    var pa = B.BatchAllocr.init(n_pos_per_embd);
    defer pa.deinit();

    for (inputs) |in| runBatch(g, ra, &pa, v, in, &ref_log, &port_log);
    // random inputs are generated one at a time, each setting the memory
    // table it runs against
    for (0..n_random) |_| runBatch(g, ra, &pa, v, g.?.input(n_pos_per_embd), &ref_log, &port_log);

    const nst: u32 = if (g) |x| @intCast(x.below(9)) else 3;
    const nsq: u32 = if (g) |x| @intCast(1 + x.below(if (x.chance(10)) 256 else 8)) else 2;
    runReserve(ra, &pa, nst, nsq, &ref_log, &port_log);
    // the allocator after a reserve: nothing to split, though a coupled
    // batch before it still refuses a sequential split
    runSplit(ra, &pa, .{ .equal = .{ .n_ubatch = 4, .sequential = true, .n_keep_tail = 0 } }, 0, &ref_log, &port_log, false);
    runSplit(ra, &pa, .{ .simple = 4 }, 0, &ref_log, &port_log, false);

    noteReached(ref_log.items);
    firstDiff(ref_log.items, port_log.items);
}

// -----------------------------------------------------------------------------
// Hand-built cases

fn hand(ar: std.mem.Allocator, v: Vocabs) void {
    const H = struct {
        fn tokens(a: std.mem.Allocator, n: usize) Builder {
            var b: Builder = .{ .ar = a, .n = n };
            const t = a.alloc(c.llama_token, n) catch unreachable;
            for (t, 0..) |*x, i| x.* = @intCast((i * 7919) % 1000);
            b.token = t;
            return b;
        }
        fn embds(a: std.mem.Allocator, n: usize, n_embd: usize, n_pos: usize) Builder {
            var b: Builder = .{ .ar = a, .n = n };
            const e = a.alloc(f32, n * n_embd) catch unreachable;
            for (e, 0..) |*x, i| x.* = @floatFromInt(i);
            b.embd = e;
            const p = a.alloc(c.llama_pos, n * n_pos) catch unreachable;
            for (p, 0..) |*x, i| x.* = @intCast(i % n + 10 * (i / n));
            b.pos = p;
            return b;
        }
        fn pos(b: *Builder, ps: []const c.llama_pos) void {
            b.pos = b.ar.dupe(c.llama_pos, ps) catch unreachable;
        }
        fn seqs(b: *Builder, sets: []const []const c.llama_seq_id) void {
            b.withSeqArrays();
            for (sets, 0..) |s, i| b.setSeqs(i, s);
        }
        fn in(b: Builder, n_seq_max: u32, n_embd: u32, output_all: bool, mem: bool) Input {
            return .{ .batch = b.cbatch(), .n_embd = n_embd, .n_seq_max = n_seq_max, .output_all = output_all, .use_memory = mem };
        }
    };

    // llama_batch_get_one: nothing but tokens
    memClear();
    {
        const b = H.tokens(ar, 9);
        session("get_one", 1, v, null, &.{ H.in(b, 1, 0, false, false), H.in(b, 1, 0, true, false), H.in(b, 4, 0, false, true) }, 0);
    }
    // continuing a sequence the memory holds, and contradicting it
    memClear();
    ref_mem_set(0, 0, 9);
    {
        var ok = H.tokens(ar, 4);
        H.pos(&ok, &.{ 10, 11, 12, 13 });
        var bad = H.tokens(ar, 4);
        H.pos(&bad, &.{ 5, 6, 7, 8 });
        var gap = H.tokens(ar, 3);
        H.pos(&gap, &.{ 10, 11, 13 });
        var dec = H.tokens(ar, 4);
        H.pos(&dec, &.{ 10, 11, 12, 11 });
        session("memory", 1, v, null, &.{ H.in(H.tokens(ar, 4), 1, 0, false, true), H.in(ok, 1, 0, false, true), H.in(bad, 1, 0, false, true), H.in(gap, 1, 0, false, true), H.in(dec, 1, 0, false, true) }, 0);
    }
    // coupled sequences: agreeing memory, diverged memory, and a sequential split
    memClear();
    {
        var b = H.tokens(ar, 6);
        H.seqs(&b, &.{ &.{ 0, 1 }, &.{ 1, 0 }, &.{0}, &.{1}, &.{0}, &.{1} });
        H.pos(&b, &.{ 0, 1, 2, 2, 3, 3 });
        session("coupled", 1, v, null, &.{H.in(b, 2, 0, false, true)}, 0);
        ref_mem_set(1, 0, 3);
        var b2 = H.tokens(ar, 2);
        H.seqs(&b2, &.{ &.{ 0, 1 }, &.{ 0, 1 } });
        session("coupled, diverged", 1, v, null, &.{H.in(b2, 2, 0, false, true)}, 0);
    }
    // the n_seq_max bound, a sequence id past it, and n_seq_max = 0
    memClear();
    {
        var b = H.tokens(ar, 2);
        H.seqs(&b, &.{ &.{0}, &.{3} });
        var z = H.tokens(ar, 2);
        H.pos(&z, &.{ 0, 1 });
        session("n_seq_max", 1, v, null, &.{ H.in(b, 257, 0, false, false), H.in(b, 3, 0, false, false), H.in(b, 4, 0, false, false), H.in(z, 0, 0, false, false) }, 0);
    }
    // M-RoPE: jumps, X < Y for tokens and X <= Y for embeddings
    memClear();
    ref_mem_set(0, 0, 10);
    {
        var t = H.tokens(ar, 3);
        H.pos(&t, &.{ 11, 15, 30 });
        var teq = H.tokens(ar, 2);
        H.pos(&teq, &.{ 10, 12 });
        var e = H.embds(ar, 3, 2, 4);
        for (0..3) |i| e.pos.?[i] = @intCast(10 + i);
        var elt = H.embds(ar, 3, 2, 4);
        for (0..3) |i| elt.pos.?[i] = @intCast(9 + i);
        session("m-rope", 4, v, null, &.{ H.in(t, 1, 0, false, true), H.in(teq, 1, 0, false, true), H.in(e, 1, 2, false, true), H.in(elt, 1, 2, false, true) }, 0);
    }
    // embeddings with output_all over flags that leave some tokens out
    memClear();
    {
        var e = H.embds(ar, 4, 3, 1);
        e.logits = ar.dupe(i8, &.{ 1, 0, 1, 0 }) catch unreachable;
        session("embeddings", 1, v, null, &.{ H.in(e, 1, 3, true, false), H.in(e, 1, 3, false, false) }, 0);
    }
    // all 256 sequences, one token each, and 255 coupled to 0
    memClear();
    {
        var b = H.tokens(ar, 256);
        b.withSeqArrays();
        for (0..256) |i| b.setSeqs(i, &.{@intCast(i)});
        var cpl = H.tokens(ar, 256);
        cpl.withSeqArrays();
        for (0..256) |i| cpl.setSeqs(i, &.{@intCast(i)});
        cpl.setSeqs(0, &.{ 0, 255 });
        session("256 sequences", 1, v, null, &.{ H.in(b, 256, 0, false, false), H.in(cpl, 256, 0, false, false), H.in(b, 256, 0, true, false) }, 0);
    }
    // invalid tokens
    memClear();
    {
        var b = H.tokens(ar, 3);
        b.token.?[1] = -1;
        var b2 = H.tokens(ar, 3);
        b2.token.?[2] = @intCast(v.port.nTokens());
        session("invalid token", 1, v, null, &.{ H.in(b, 1, 0, false, false), H.in(b2, 1, 0, false, false) }, 0);
    }
}

// -----------------------------------------------------------------------------
// Aborts and the C interface

var abort_vocab: Vocabs = undefined;

fn abortsInChild(body: *const fn (side: c_int) void, side: c_int) bool {
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) return false;
    const pid = fork();
    if (pid == 0) {
        _ = dup2(fds[1], 2);
        _ = close(fds[0]);
        body(side);
        _exit(0);
    }
    _ = close(fds[1]);
    var buf: [4096]u8 = undefined;
    while (read(fds[0], &buf, buf.len) > 0) {}
    _ = close(fds[0]);
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    return (status & 0x7f) == sigabrt;
}

fn emptyBatch(side: c_int) void {
    var tok = [_]c.llama_token{1};
    const b = B.batchGetOne(&tok, 0);
    if (side == 0) {
        _ = ref_init(ref_new(1), &b, abort_vocab.ref, 0, 0, 1, 0);
    } else {
        var pa = B.BatchAllocr.init(1);
        _ = pa.initBatch(b, abort_vocab.port, null, 0, 1, false);
    }
}

fn keepTailTooLong(side: c_int) void {
    var tok = [_]c.llama_token{ 1, 2, 3 };
    const b = B.batchGetOne(&tok, 3);
    if (side == 0) {
        const ra = ref_new(1);
        _ = ref_init(ra, &b, abort_vocab.ref, 0, 0, 1, 0);
        var r: RefUbatch = .{};
        ref_split_equal(ra, 2, 0, 2, &r);
    } else {
        var pa = B.BatchAllocr.init(1);
        _ = pa.initBatch(b, abort_vocab.port, null, 0, 1, false);
        _ = pa.splitEqual(2, false, 2);
    }
}

fn checkAborts(v: Vocabs) void {
    session_name = "aborts";
    session_failures = 0;
    abort_vocab = v;
    inline for (.{ .{ "n_tokens = 0", emptyBatch }, .{ "n_keep_tail >= n_ubatch", keepTailTooLong } }) |t| {
        checks += 1;
        results += 1;
        const r = abortsInChild(t[1], 0);
        const p = abortsInChild(t[1], 1);
        if (!r or !p) fail("{s}: ref aborted {} port aborted {}", .{ t[0], r, p });
    }
}

fn checkCInterface() void {
    session_name = "C interface";
    session_failures = 0;
    var tok = [_]c.llama_token{ 5, 6, 7 };
    const r1 = c.llama_batch_get_one(&tok, 3);
    const p1 = B.batchGetOne(&tok, 3);
    results += 1;
    checks += 1;
    inline for (@typeInfo(c.llama_batch).@"struct".field_names) |f| {
        if (@field(r1, f) != @field(p1, f)) fail("llama_batch_get_one.{s} differs", .{f});
    }

    for ([_][3]i32{ .{ 4, 0, 1 }, .{ 4, 8, 2 }, .{ 0, 0, 1 }, .{ 1, 3, 256 } }) |a| {
        results += 1;
        const r = c.llama_batch_init(a[0], a[1], a[2]);
        const p = B.batchInit(a[0], a[1], a[2]);
        _ = expectInt("n_tokens", r.n_tokens, p.n_tokens);
        _ = sameNull("batch_init token", r.token, p.token);
        _ = sameNull("batch_init embd", r.embd, p.embd);
        _ = sameNull("batch_init pos", r.pos, p.pos);
        _ = sameNull("batch_init n_seq_id", r.n_seq_id, p.n_seq_id);
        _ = sameNull("batch_init seq_id", r.seq_id, p.seq_id);
        _ = sameNull("batch_init logits", r.logits, p.logits);
        const n: usize = @intCast(a[0]);
        for (0..n) |i| _ = sameNull("batch_init seq_id[i]", r.seq_id[i], p.seq_id[i]);
        _ = sameNull("batch_init seq_id[n]", r.seq_id[n], p.seq_id[n]);
        // both allocate with malloc, so each frees the other's
        c.llama_batch_free(p);
        B.batchFree(r);
    }
}

// -----------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: batch_diff <vocab.gguf> [seed]\n", .{});
        std.process.exit(2);
    }
    const path = try gpa.dupeSentinel(u8, args[1], 0);
    const seed: u64 = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 20261013;

    c.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    log_target = null;

    // the vocabulary, once through each side
    var mp = c.llama_model_default_params();
    mp.vocab_only = true;
    const ref_model = c.llama_model_load_from_file(path.ptr, mp) orelse {
        std.debug.print("FAIL: the C++ cannot load {s}\n", .{path});
        std.process.exit(1);
    };
    defer c.llama_model_free(ref_model);
    var exc: staged.impl.Exception = .{};
    var splits: std.ArrayList([:0]const u8) = .empty;
    var params = M.defaultParams();
    params.vocab_only = true;
    var loaded = M.loadMetadata(.{ .path = path }, &splits, params, null, &exc);
    if (loaded != .model) {
        std.debug.print("FAIL: the port cannot load {s}\n", .{path});
        std.process.exit(1);
    }
    defer loaded.model.deinit();
    const v: Vocabs = .{ .ref = c.llama_model_get_vocab(ref_model).?, .port = &loaded.model.vocab };
    if (c.llama_vocab_n_tokens(v.ref) != @as(i32, @intCast(v.port.nTokens()))) {
        std.debug.print("FAIL: the vocabularies differ in size\n", .{});
        std.process.exit(1);
    }

    checkCInterface();
    checkAborts(v);

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var prng: std.Random.DefaultPrng = .init(seed);
    const g: Gen = .{ .r = prng.random(), .ar = arena.allocator(), .n_vocab = v.port.nTokens() };

    const modes = [_]struct { env: ?[:0]const u8, sessions: usize }{
        .{ .env = null, .sessions = 2500 },
        .{ .env = "2", .sessions = 800 },
        .{ .env = "1", .sessions = 100 },
        .{ .env = "x", .sessions = 50 },
    };
    for (modes) |md| {
        if (md.env) |e| _ = setenv("LLAMA_BATCH_DEBUG", e, 1) else _ = unsetenv("LLAMA_BATCH_DEBUG");
        std.debug.print("  LLAMA_BATCH_DEBUG={s}\n", .{md.env orelse "(unset)"});

        _ = arena.reset(.retain_capacity);
        hand(arena.allocator(), v);

        var name_buf: [64]u8 = undefined;
        for (0..md.sessions) |i| {
            _ = arena.reset(.retain_capacity);
            const n_pos: u32 = g.pick(u32, &.{ 1, 1, 1, 1, 1, 1, 1, 4, 4, 3 });
            session(std.fmt.bufPrint(&name_buf, "random {d} (seed {d})", .{ i, seed }) catch "random", n_pos, v, g, &.{}, 1 + g.below(3));
        }
    }
    _ = unsetenv("LLAMA_BATCH_DEBUG");

    std.debug.print("  {d} sessions, {d} batches accepted, {d} rejected; {d} endless split_equal runs agreed to their cap\n", .{ tally.sessions, tally.accepted, tally.rejected, tally.endless });
    std.debug.print("  errors and warnings the C++ raised:\n", .{});
    for (reached.keys(), reached.values()) |k, n| std.debug.print("    {d:>7}  {s}\n", .{ n, k });

    if (failures != 0) {
        std.debug.print("FAIL: {d} of {d} checks differ\n", .{ failures, checks });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} batch results identical, Zig vs llama-batch.cpp ({d} checks)\n", .{ results, checks });
}
