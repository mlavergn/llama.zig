//! Batches: the extended batch, sanitizing one and splitting it into
//! ubatches.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-batch.h`   -- `llama_ubatch`, `llama_batch_ext`,
//!   `llama_batch_allocr`, `llama_batch_compat`
//! - `llama.cpp/src/llama-batch.cpp` -- their bodies, `llama_batch_get_one`,
//!   `llama_batch_init` and `llama_batch_free`, and the `llama_batch_ext_*`
//!   C API's bodies
//! - `llama.cpp/src/llama-cparams.h` -- `LLAMA_MAX_SEQ`
//!
//! All at v0.6.0 (`d81235049`). Each declaration below names the source file
//! and line it came from. The C entry points are `api/batch.zig`'s.
//!
//! # The extended batch's sequence sets
//!
//! A `llama_batch_ext` entry keeps its sequence ids in a
//! `std::unordered_set<llama_seq_id>`, and `llama_batch_allocr::init` copies
//! them out in its iteration order -- the order `seq_id[i]` then has, whose
//! first element decides coupling, `split_equal`'s sequential test and
//! `llama_batch_compat`'s generated positions. `SeqIdSet` keeps the ids in
//! insertion order and replays that order through `libcxx.I32Map`, which
//! lays the table out as libc++ does (`std::hash<int>` is the value
//! sign-extended). libc++ 22's copy constructor keeps a table's layout, so
//! the C++'s copies of an entry change nothing.
//!
//! # What the C++ leaves to libc++, and why none of it is reproduced
//!
//! Three containers here have an iteration order or layout the standard
//! leaves to the library, and each was checked for whether anything reads it:
//!
//! - **`seq_set_map`, a `std::unordered_map<std::bitset<256>, ...>`, is never
//!   iterated.** It is written at one site and otherwise only looked up by
//!   key (`split_equal`'s five lookups) or cleared, so libc++'s bucket order
//!   and `std::hash<std::bitset>` are unobservable. A Zig hash map keyed on
//!   the 256-bit mask stands in for it.
//! - **`seq_pos[s]`, a `std::set<llama_pos>`, is read only through `empty`,
//!   `size`, `begin` and `rbegin`** -- a distinct count, a minimum and a
//!   maximum. A sorted, de-duplicated list gives all four.
//! - **`used` and `seq_cpl` are `std::vector<bool>`**, read element by
//!   element. Plain `bool`s and bit masks.
//!
//! # Sharing a ubatch's data
//!
//! `llama_ubatch` is copied by value all over libllama, and its arrays live in
//! a `data_t` held by `std`'s `shared_ptr`. `Ubatch` keeps the C++'s fields and
//! makes the reference count explicit: `retain` and `release` stand where the
//! copy constructor and destructor of the `shared_ptr` would run.

const std = @import("std");
const impl = @import("impl.zig");
const vm = @import("vocab/module.zig");
const libcxx = @import("libcxx.zig");
const arch_mod = @import("arch.zig");
const hparams_mod = @import("hparams.zig");

const c = impl.c;
const allocator = impl.allocator;
const Vocab = vm.Vocab;

const Token = c.llama_token;
const Pos = c.llama_pos;
const SeqId = c.llama_seq_id;

extern "c" fn atoi(s: [*:0]const u8) c_int;

fn oom() noreturn {
    impl.abort("llama-batch.cpp", 0, "out of memory");
}

/// Ports `LLAMA_MAX_SEQ` (llama-cparams.h:8 @d81235049).
pub const max_seq: u32 = 256;

/// Ports `llama_batch_allocr`'s `seq_set_t`, `std::bitset<LLAMA_MAX_SEQ>`:
/// bit `s` is sequence `s`.
pub const SeqSet = u256;

fn bit(s: SeqId) SeqSet {
    return @as(SeqSet, 1) << @intCast(s);
}

/// A `std`'s `vector`'s `data()` for a vector that has only ever been resized:
/// null while it is empty, as libc++'s is before its first allocation. The
/// debug log prints these pointers, so null is observable.
fn dataPtr(comptime T: type, list: *std.ArrayList(T)) [*c]T {
    return if (list.items.len == 0) null else list.items.ptr;
}

fn cInt(v: u32) c_int {
    return @bitCast(v);
}

// -----------------------------------------------------------------------------
// llama_ubatch

/// Ports `llama_ubatch`'s `data_t` (llama-batch.h:63 @d81235049): the arrays a
/// `Ubatch` built by the allocator points into, with the reference count
/// `std`'s `shared_ptr` keeps beside them.
pub const UbatchData = struct {
    /// The `shared_ptr`'s use count.
    refs: std.atomic.Value(u32) = .init(1),

    token: std.ArrayList(Token) = .empty,
    embd: std.ArrayList(f32) = .empty,
    pos: std.ArrayList(Pos) = .empty,
    n_seq_id: std.ArrayList(i32) = .empty,
    /// These point into `seq_id_data`.
    seq_id: std.ArrayList([*c]SeqId) = .empty,
    seq_id_unq: std.ArrayList(SeqId) = .empty,
    seq_idx: std.ArrayList(i32) = .empty,
    output: std.ArrayList(i8) = .empty,
    /// Mixed ubatches only: 1 for an embedding row, 0 for a token.
    type: std.ArrayList(i8) = .empty,
    /// Each token's index in the original batch.
    batch_idxs: std.ArrayList(i32) = .empty,
    decision_order: std.ArrayList(i32) = .empty,

    seq_id_data: std.ArrayList(SeqId) = .empty,

    /// `std::make_shared<llama_ubatch::data_t>()`.
    ///
    /// Return: empty data holding one reference.
    fn create() *UbatchData {
        const d = allocator.create(UbatchData) catch oom();
        d.* = .{};
        return d;
    }

    fn destroy(self: *UbatchData) void {
        self.token.deinit(allocator);
        self.embd.deinit(allocator);
        self.pos.deinit(allocator);
        self.n_seq_id.deinit(allocator);
        self.seq_id.deinit(allocator);
        self.seq_id_unq.deinit(allocator);
        self.seq_idx.deinit(allocator);
        self.output.deinit(allocator);
        self.type.deinit(allocator);
        self.batch_idxs.deinit(allocator);
        self.decision_order.deinit(allocator);
        self.seq_id_data.deinit(allocator);
        allocator.destroy(self);
    }
};

/// Ports `llama_ubatch` (llama-batch.h:17 @d81235049).
///
/// A value type, as in the C++, and default-initializable to the empty
/// ubatch the splitters return when the batch is consumed. Its pointers
/// point into `data` when it is set, and otherwise into memory someone else
/// owns -- the input batch, for the ubatch `initBatch` prints.
///
/// **A plain copy is a borrow.** A copy that outlives the ubatch it was taken
/// from must call `retain` on the source and store the copy, and
/// `release` it when done -- what the C++'s `shared_ptr` copy constructor and
/// destructor do.
pub const Ubatch = struct {
    /// A boolean, kept as `uint32_t` as the C++ does for alignment.
    b_equal_seqs: u32 = 0,

    /// Total tokens, `n_seq_tokens * n_seqs`.
    n_tokens: u32 = 0,
    /// Tokens per sequence set.
    n_seq_tokens: u32 = 0,
    /// Sequence sets in the ubatch.
    n_seqs: u32 = 0,
    /// Unique sequence ids in the ubatch.
    n_seqs_unq: u32 = 0,
    /// Position inputs for each token or embedding.
    n_pos: u32 = 0,

    /// `[n_tokens]`, or null for an embeddings batch.
    token: [*c]Token = null,
    /// `[n_embd * n_tokens]`, or null for a token batch.
    embd: [*c]f32 = null,
    /// `[n_tokens * n_pos]`.
    pos: [*c]Pos = null,
    /// `[n_tokens]`.
    n_seq_id: [*c]i32 = null,
    /// `[n_tokens]`, each `[n_seq_id[i]]`.
    seq_id: [*c][*c]SeqId = null,
    /// `[n_seqs_unq]`: the unique sequence ids.
    seq_id_unq: [*c]SeqId = null,
    /// `[LLAMA_MAX_SEQ]`: each sequence's index in `seq_id_unq`, or -1.
    seq_idx: [*c]i32 = null,
    /// `[n_tokens]`.
    output: [*c]i8 = null,
    /// `[n_tokens]`, mixed ubatches only: 0 for a token, 1 for an embedding.
    type: [*c]i8 = null,
    /// `[n_tokens]`, or null when no entry has one: see
    /// `llama_batch_ext_set_decision_order`.
    decision_order: [*c]i32 = null,

    /// The arrays above, when this ubatch owns them.
    data: ?*UbatchData = null,

    /// Ports `llama_ubatch`'s `equal_seqs` (llama-batch.h:18 @d81235049).
    ///
    /// Return: whether every sequence set has `n_seq_tokens` tokens.
    pub fn equalSeqs(self: *const Ubatch) bool {
        return self.b_equal_seqs != 0;
    }

    /// Ports `llama_ubatch`'s `is_pos_2d` (llama-batch.h:27 @d81235049).
    ///
    /// Return: whether positions are M-RoPE's, three or more per token.
    pub fn isPos2d(self: *const Ubatch) bool {
        return self.n_pos >= 3;
    }

    /// Ports `llama_ubatch`'s `is_mixed` (llama-batch.h:33 @d81235049).
    ///
    /// Return: whether `type` picks a token or an embedding per row, with
    /// `n_pos` position sections for every row.
    pub fn isMixed(self: *const Ubatch) bool {
        return self.type != null;
    }

    /// Ports `ASSERT_EMBD_OR_TOKEN` (llama-batch.h:84 @d81235049): aborts
    /// when a mixed ubatch reaches code that takes only tokens or only
    /// embeddings.
    ///
    /// Parameters:
    /// - `file`, `line`: the call site, for the abort message.
    ///
    /// Return: nothing.
    pub fn assertEmbdOrToken(self: *const Ubatch, comptime file: [:0]const u8, comptime line: c_int) void {
        impl.assert(!self.isMixed(), file, line, "!(ubatch).is_mixed() && \"mixed token/embd ubatch is not supported here\"");
    }

    /// Takes a reference to `data`, as copying the C++'s `shared_ptr` does.
    /// The caller then stores a by-value copy and `release`s that copy.
    ///
    /// Return: nothing.
    pub fn retain(self: *const Ubatch) void {
        if (self.data) |d| _ = d.refs.fetchAdd(1, .monotonic);
    }

    /// Drops one reference to `data`, freeing it with the last one, as the
    /// `shared_ptr`'s destructor does; `data` is null afterwards. The
    /// pointers are left as they were and dangle once the data is freed.
    ///
    /// Return: nothing.
    pub fn release(self: *Ubatch) void {
        const d = self.data orelse return;
        self.data = null;
        if (d.refs.fetchSub(1, .acq_rel) == 1) d.destroy();
    }
};

// -----------------------------------------------------------------------------
// The memory module, as far as the allocator asks it

/// What `llama_batch_allocr`'s `init` asks of `llama_memory_i`: the first and
/// last position each sequence holds, -1 for an empty sequence.
///
/// The memory modules are not ported yet; this is the two-method slice of
/// the interface the allocator reads, so it can be gated now.
pub const MemoryPos = struct {
    ctx: *anyopaque,
    seq_pos_min: *const fn (ctx: *anyopaque, seq_id: SeqId) Pos,
    seq_pos_max: *const fn (ctx: *anyopaque, seq_id: SeqId) Pos,

    fn posMin(self: MemoryPos, s: SeqId) Pos {
        return self.seq_pos_min(self.ctx, s);
    }

    fn posMax(self: MemoryPos, s: SeqId) Pos {
        return self.seq_pos_max(self.ctx, s);
    }
};

/// `llama_batch_allocr`'s `pos_set_t`, `std::set<llama_pos>`, read only
/// through `empty`, `size`, `begin` and `rbegin`: a sorted list without
/// duplicates.
const PosSet = struct {
    items: std.ArrayList(Pos) = .empty,

    fn insert(self: *PosSet, p: Pos) void {
        var lo: usize = 0;
        var hi: usize = self.items.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.items.items[mid] < p) lo = mid + 1 else hi = mid;
        }
        if (lo < self.items.items.len and self.items.items[lo] == p) return;
        self.items.insert(allocator, lo, p) catch oom();
    }
};

// -----------------------------------------------------------------------------
// llama_batch_allocr

/// Ports `llama_batch_allocr` (llama-batch.h:145 @d81235049): converts a
/// `llama_batch_ext` into the flat `llama_batch` it keeps, sanitizes it,
/// and splits it into ubatches.
pub const BatchAllocr = struct {
    batch: c.llama_batch = .{},

    /// Only for debugging.
    vocab: ?*const Vocab = null,

    n_pos_per_embd: u32,

    /// Whether a batch may mix token and embedding entries.
    allow_mixed: bool = false,

    n_embd: u32 = 0,
    n_seq_max: u32 = 0,
    n_outputs: u32 = 0,

    /// The token ids built from the extended batch.
    token_vec: std.ArrayList(Token) = .empty,
    /// The embeddings built from the extended batch.
    embd_vec: std.ArrayList(f32) = .empty,
    /// Mixed batches only: 1 for an embedding entry, 0 for a token.
    is_embd_vec: std.ArrayList(i8) = .empty,
    /// Flat storage the `seq_id` pointers point into.
    seq_id_data: std.ArrayList(SeqId) = .empty,

    pos: std.ArrayList(Pos) = .empty,
    n_seq_id: std.ArrayList(i32) = .empty,
    seq_id: std.ArrayList([*c]SeqId) = .empty,
    seq_id_unq: std.ArrayList(SeqId) = .empty,
    seq_idx: std.ArrayList(i32) = .empty,
    output: std.ArrayList(i8) = .empty,
    /// Empty when no entry has one.
    decision_order: std.ArrayList(i32) = .empty,

    /// Whether any token in the batch belongs to more than one sequence.
    has_cpl: bool = false,

    /// `seq_pos[s]`: the positions in sequence `s`.
    seq_pos: [max_seq]PosSet = @splat(.{}),
    /// `seq_cpl[s0]` bit `s1`: sequence `s0` is coupled to `s1`.
    seq_cpl: [max_seq]SeqSet = @splat(0),

    /// `seq_set[i]`: the sequence set of token `i`.
    seq_set: std.ArrayList(SeqSet) = .empty,
    /// The indices at which each sequence set appears.
    seq_set_map: std.AutoHashMapUnmanaged(SeqSet, std.ArrayList(i32)) = .empty,

    /// Batch indices of the outputs, in the order the splits met them.
    out_ids: std.ArrayList(i32) = .empty,

    n_used: u32 = 0,

    /// `used[i]`: token `i` went into an earlier ubatch.
    used: std.ArrayList(bool) = .empty,

    debug: c_int = 0,

    /// Ports the constructor of `llama_batch_allocr` (llama-batch.cpp:15
    /// @d81235049), which reads `LLAMA_BATCH_DEBUG` once, here.
    ///
    /// Parameters:
    /// - `n_pos_per_embd`: positions per token, 4 for M-RoPE.
    /// - `allow_mixed`: whether a batch may mix token and embedding entries.
    ///
    /// Return: an allocator with no batch; free it with `deinit`.
    pub fn init(n_pos_per_embd: u32, allow_mixed: bool) BatchAllocr {
        var self: BatchAllocr = .{ .n_pos_per_embd = n_pos_per_embd, .allow_mixed = allow_mixed };
        self.debug = if (std.c.getenv("LLAMA_BATCH_DEBUG")) |v| atoi(v) else 0;
        self.seq_idx.appendNTimes(allocator, -1, max_seq) catch oom();
        return self;
    }

    /// Return: nothing.
    pub fn deinit(self: *BatchAllocr) void {
        self.token_vec.deinit(allocator);
        self.embd_vec.deinit(allocator);
        self.is_embd_vec.deinit(allocator);
        self.seq_id_data.deinit(allocator);
        self.decision_order.deinit(allocator);
        self.pos.deinit(allocator);
        self.n_seq_id.deinit(allocator);
        self.seq_id.deinit(allocator);
        self.seq_id_unq.deinit(allocator);
        self.seq_idx.deinit(allocator);
        self.output.deinit(allocator);
        for (&self.seq_pos) |*p| p.items.deinit(allocator);
        self.seq_set.deinit(allocator);
        self.clearSeqSetMap();
        self.seq_set_map.deinit(allocator);
        self.out_ids.deinit(allocator);
        self.used.deinit(allocator);
    }

    fn clearSeqSetMap(self: *BatchAllocr) void {
        var it = self.seq_set_map.valueIterator();
        while (it.next()) |v| v.deinit(allocator);
        self.seq_set_map.clearRetainingCapacity();
    }

    /// Ports `llama_batch_allocr`'s `init` (llama-batch.cpp:28 @d81235049):
    /// converts the extended batch into the flat batch this allocator keeps,
    /// validates it, and checks it against the memory the extended batch
    /// names.
    ///
    /// Parameters:
    /// - `batch_inp`: the batch; read only during the call.
    /// - `vocab`: the vocabulary, for the debug log; borrowed until the next
    ///   `initBatch` or `deinit`.
    /// - `output_all`: every token is an output.
    ///
    /// Return: whether the batch is valid; what is wrong is logged.
    pub fn initBatch(self: *BatchAllocr, batch_inp: *const BatchExt, vocab: *const Vocab, output_all: bool) bool {
        const f = "init";
        self.clear();

        self.vocab = vocab;
        self.n_embd = @truncate(if (batch_inp.n_embd > 0) batch_inp.n_embd else batch_inp.n_embd_inp);
        self.n_seq_max = @bitCast(batch_inp.n_seq_max);

        const toks = batch_inp.tokens.items;
        const n_tok: i32 = @truncate(@as(isize, @bitCast(toks.len)));

        impl.assert(n_tok > 0, "llama-batch.cpp", 40, "n_tok > 0");
        const n_tokens: usize = @intCast(n_tok);

        if (self.n_seq_max > max_seq) {
            impl.logError("%s: n_seq_max = %d > %d\n", .{ f, cInt(self.n_seq_max), cInt(max_seq) });
            return false;
        }

        const mem = batch_inp.mem;
        const n_seq_max = self.n_seq_max;

        //
        // the content types: an entry carries a token id, an embedding, or
        // both (MTP hook batches); all carry the same, or tokens and
        // embeddings mix
        //

        var n_tok_only: i32 = 0;
        var n_embd_only: i32 = 0;
        var n_both: i32 = 0;

        for (toks, 0..) |*t, i| {
            const is_tok = t.id != null_token;
            const is_emb = t.has_embd;

            if (!is_tok and !is_emb) {
                impl.logError("%s: entry %d has neither a token id nor an embedding\n", .{ f, @as(i32, @intCast(i)) });
                return false;
            }

            n_tok_only += @intFromBool(is_tok and !is_emb);
            n_embd_only += @intFromBool(is_emb and !is_tok);
            n_both += @intFromBool(is_tok and is_emb);
        }

        if (n_both > 0 and n_both != n_tok) {
            impl.logError("%s: entries with both a token id and an embedding cannot be mixed with other entries\n", .{f});
            return false;
        }

        const mixed = n_tok_only > 0 and n_embd_only > 0;

        if (mixed and !self.allow_mixed) {
            impl.logError("%s: this model or context does not support batches mixing token and embedding entries\n", .{f});
            return false;
        }

        const has_token = n_tok_only > 0 or n_both > 0;
        const has_embd = n_embd_only > 0 or n_both > 0;

        if (mixed) {
            self.is_embd_vec.resize(allocator, n_tokens) catch oom();
            for (toks, 0..) |*t, i| self.is_embd_vec.items[i] = @intFromBool(t.has_embd);
        }

        //
        // the flat token and embedding arrays
        //

        if (has_token) {
            self.token_vec.resize(allocator, n_tokens) catch oom();
            for (toks, 0..) |*t, i| {
                if (mixed and self.is_embd_vec.items[i] != 0) {
                    self.token_vec.items[i] = 0; // a placeholder
                    continue;
                }
                const id = t.id;
                if (id < 0 or id >= batch_inp.n_vocab) {
                    impl.logError("%s: invalid token[%d] = %d\n", .{ f, @as(i32, @intCast(i)), id });
                    return false;
                }
                self.token_vec.items[i] = id;
            }
        }

        const n_embd: usize = self.n_embd;
        if (mixed) {
            self.embd_vec.appendNTimes(allocator, 0, n_tokens * n_embd) catch oom();
            for (toks, 0..) |*t, i| {
                if (self.is_embd_vec.items[i] != 0) {
                    @memcpy(self.embd_vec.items[i * n_embd ..][0..n_embd], batch_inp.embd.items[t.embd_off..][0..n_embd]);
                }
            }
        } else if (has_embd) {
            // as the entries filled it, which is the entries' order only
            // when each one's embedding was set in turn, as the C++ copies
            self.embd_vec.appendSlice(allocator, batch_inp.embd.items) catch oom();
        }

        //
        // the flat positions, section-major: pos[j*n_tok + i] is section j of
        // entry i; a token entry is [p, p, p, 0] (M-RoPE's text position),
        // an embedding entry its positions as they are
        //

        const n_pos: usize = self.n_pos_per_embd;
        self.pos.resize(allocator, n_tokens * n_pos) catch oom();
        for (toks, 0..) |*tok, i| {
            const expand = tok.id != null_token;
            for (0..n_pos) |j| {
                var p = tok.pos[j];
                if (expand) {
                    p = if (j < 3) tok.pos[0] else 0;
                }
                self.pos.items[j * n_tokens + i] = p;
            }
        }

        //
        // n_seq_id and seq_id
        //

        self.n_seq_id.resize(allocator, n_tokens) catch oom();
        self.seq_id.resize(allocator, n_tokens + 1) catch oom();
        self.seq_id.items[n_tokens] = null;

        {
            var total: usize = 0;
            for (toks) |*t| total += t.seq_ids.count();
            self.seq_id_data.ensureTotalCapacity(allocator, total) catch oom();

            for (toks) |*t| {
                const n = t.seq_ids.count();
                const dst = self.seq_id_data.addManyAsSliceAssumeCapacity(n);
                t.seq_ids.ordered(dst);
            }

            var off: usize = 0;
            for (toks, 0..) |*t, i| {
                self.n_seq_id.items[i] = @intCast(t.seq_ids.count());
                self.seq_id.items[i] = self.seq_id_data.items.ptr + off;
                off += @intCast(self.n_seq_id.items[i]);

                var s: i32 = 0;
                while (s < self.n_seq_id.items[i]) : (s += 1) {
                    const id = self.seq_id.items[i][@intCast(s)];
                    if (id < 0 or id >= @as(SeqId, @bitCast(n_seq_max))) {
                        impl.logError("%s: invalid seq_id[%d][%d] = %d >= %d\n", .{ f, @as(i32, @intCast(i)), s, id, cInt(n_seq_max) });
                        return false;
                    }
                }
            }
        }

        //
        // the output flags
        //

        {
            self.output.resize(allocator, n_tokens) catch oom();
            for (toks, 0..) |*t, i| self.output.items[i] = @intFromBool(t.output);

            if (output_all) {
                var warn = false;
                for (self.output.items) |o| {
                    if (o == 0) {
                        warn = true;
                        break;
                    }
                }
                if (warn) {
                    impl.logWarn("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n", .{f});
                    @memset(self.output.items, 1);
                }
            }
        }

        // kept empty when no entry has one
        for (toks, 0..) |*t, i| {
            if (t.decision_order != 0) {
                if (self.decision_order.items.len < n_tokens) {
                    self.decision_order.appendNTimes(allocator, 0, n_tokens - self.decision_order.items.len) catch oom();
                }
                self.decision_order.items[i] = t.decision_order;
            }
        }

        //
        // the flat batch, over the arrays above
        //

        self.batch.n_tokens = n_tok;
        self.batch.token = if (has_token) dataPtr(Token, &self.token_vec) else null;
        self.batch.embd = if (has_embd) dataPtr(f32, &self.embd_vec) else null;
        self.batch.pos = dataPtr(Pos, &self.pos);
        self.batch.n_seq_id = dataPtr(i32, &self.n_seq_id);
        self.batch.seq_id = dataPtr([*c]SeqId, &self.seq_id);
        self.batch.logits = dataPtr(i8, &self.output);

        //
        // compute stats
        //

        for (0..n_tokens) |i| {
            self.n_outputs += @intFromBool(self.batch.logits[i] != 0);
        }

        self.has_cpl = false;

        // coupled sequences: pairs that share at least one token
        for (0..n_tokens) |i| {
            var s: usize = 0;
            while (s < self.batch.n_seq_id[i]) : (s += 1) {
                const s1 = self.batch.seq_id[i][s];

                self.seq_pos[@intCast(s1)].insert(self.batch.pos[i]);

                if (s > 0) {
                    // s1 is coupled to s0; the other direction is not tracked
                    const s0 = self.batch.seq_id[i][0];
                    self.seq_cpl[@intCast(s1)] |= bit(s0);

                    self.has_cpl = true;
                }
            }
        }

        // each token's sequence set, and the unique sequence ids
        {
            var seq_set_unq: SeqSet = 0;

            for (0..n_tokens) |i| {
                var cur: SeqSet = 0;
                var s: usize = 0;
                while (s < self.batch.n_seq_id[i]) : (s += 1) {
                    const seq_id = self.batch.seq_id[i][s];
                    cur |= bit(seq_id);
                    seq_set_unq |= bit(seq_id);
                }

                self.seq_set.append(allocator, cur) catch oom();
                const gop = self.seq_set_map.getOrPut(allocator, cur) catch oom();
                if (!gop.found_existing) gop.value_ptr.* = .empty;
                gop.value_ptr.append(allocator, @intCast(i)) catch oom();
            }

            for (0..n_seq_max) |s| {
                if (seq_set_unq & bit(@intCast(s)) != 0) {
                    self.seq_idx.items[s] = @intCast(self.seq_id_unq.items.len);
                    self.seq_id_unq.append(allocator, @intCast(s)) catch oom();
                }
            }
        }

        if (self.debug > 0) {
            impl.logDebug("%s: input batch info:\n", .{f});

            const ubatch: Ubatch = .{
                .b_equal_seqs = 0,
                .n_tokens = @intCast(self.batch.n_tokens),
                .n_seq_tokens = 1,
                .n_seqs = @intCast(self.batch.n_tokens),
                .n_seqs_unq = @intCast(self.seq_id_unq.items.len),
                .n_pos = self.n_pos_per_embd,
                .token = self.batch.token,
                .embd = self.batch.embd,
                .pos = self.batch.pos,
                .n_seq_id = self.batch.n_seq_id,
                .seq_id = self.batch.seq_id,
                .seq_id_unq = self.seq_id_unq.items.ptr,
                .seq_idx = self.seq_idx.items.ptr,
                .output = self.batch.logits,
                .type = dataPtr(i8, &self.is_embd_vec),
                .decision_order = dataPtr(i32, &self.decision_order),
                .data = null,
            };

            self.ubatchPrint(&ubatch, self.debug);

            impl.logDebug("%s:   seq       = [\n", .{f});
            var ss: std.ArrayList(u8) = .empty;
            defer ss.deinit(allocator);
            for (0..max_seq) |s0| {
                if (self.seq_pos[s0].items.items.len == 0) continue;

                ss.clearRetainingCapacity();
                for (0..max_seq) |s1| {
                    if (self.seq_cpl[s0] & bit(@intCast(s1)) != 0) {
                        ss.print(allocator, "{d} ", .{s1}) catch oom();
                    }
                }
                if (ss.items.len == 0) ss.append(allocator, '-') catch oom();
                ss.append(allocator, 0) catch oom();

                const s: SeqId = @intCast(s0);
                impl.logDebug("%s:  %4d: pos = [%4d, %4d], cpl = %s\n", .{ f, s, self.seqPosMin(s), self.seqPosMax(s), @as([*:0]const u8, @ptrCast(ss.items.ptr)) });
            }
            impl.logDebug("%s:   ]\n", .{f});
        }

        //
        // consistency checks
        //

        if (self.n_pos_per_embd > 1) {
            // in a mixed batch, the first entry of each sequence picks the rule
            var seq_first_embd: [max_seq]i8 = @splat(if (self.batch.token != null) 0 else 1);
            if (mixed) {
                var seen: [max_seq]bool = @splat(false);
                for (0..n_tokens) |i| {
                    var s: usize = 0;
                    while (s < self.batch.n_seq_id[i]) : (s += 1) {
                        const sid: usize = @intCast(self.batch.seq_id[i][s]);
                        if (!seen[sid]) {
                            seen[sid] = true;
                            seq_first_embd[sid] = self.is_embd_vec.items[i];
                        }
                    }
                }
            }

            // M-RoPE: positions may jump forward, and need not be continuous
            for (0..n_seq_max) |su| {
                if (self.seq_pos[su].items.items.len == 0) continue;
                const s: SeqId = @intCast(su);

                const p0: Pos = if (mem) |m| m.posMax(s) else -1;

                if (seq_first_embd[su] == 0) {
                    if (p0 >= 0 and p0 >= self.seqPosMin(s)) {
                        impl.logError(
                            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n" ++
                                " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n" ++
                                " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n" ++
                                " for M-RoPE, it is required that the position satisfies: X < Y\n",
                            .{ f, s, s, p0, s, self.seqPosMin(s) },
                        );
                        return false;
                    }
                } else {
                    // embedding inputs can have overlapping positions
                    if (p0 >= 0 and p0 > self.seqPosMin(s)) {
                        impl.logError(
                            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n" ++
                                " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n" ++
                                " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n" ++
                                " for M-RoPE, it is required that the position satisfies: X <= Y\n",
                            .{ f, s, s, p0, s, self.seqPosMin(s) },
                        );
                        return false;
                    }
                }
            }
        } else {
            for (0..n_seq_max) |su| {
                if (self.seq_pos[su].items.items.len == 0) continue;
                const s: SeqId = @intCast(su);

                const p0: Pos = if (mem) |m| m.posMax(s) else -1;

                if (p0 >= 0) {
                    if (self.seqPosMin(s) != p0 + 1) {
                        impl.logError(
                            "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n" ++
                                " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n" ++
                                " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n" ++
                                " it is required that the sequence positions remain consecutive: Y = X + 1\n",
                            .{ f, s, s, p0, s, self.seqPosMin(s) },
                        );
                        return false;
                    }
                }

                if (self.seqPosMax(s) - self.seqPosMin(s) + 1 > @as(c_int, @intCast(self.seq_pos[su].items.items.len))) {
                    impl.logError("%s: sequence %d positions are not continuous\n", .{ f, s });
                    return false;
                }
            }
        }

        if (mem) |m| {
            for (0..n_seq_max) |s0| {
                for (0..n_seq_max) |s1| {
                    if (self.seq_cpl[s0] & bit(@intCast(s1)) != 0) {
                        const a: SeqId = @intCast(s0);
                        const b: SeqId = @intCast(s1);
                        if (m.posMin(a) != m.posMin(b) or m.posMax(a) != m.posMax(b)) {
                            impl.logError("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", .{ f, a, b });
                            return false;
                        }
                    }
                }
            }
        }

        // disallow partial sequence sub-sets, and decreasing positions within
        // a sequence (the C++'s comment has a diagram of each)
        {
            var cur_seq_set: [max_seq]SeqSet = @splat(0);
            for (0..n_seq_max) |s| cur_seq_set[s] = ~@as(SeqSet, 0);

            var cur_seq_pos: [max_seq]Pos = @splat(-1);

            for (0..n_tokens) |i| {
                const pos = self.batch.pos[i];

                var s: usize = 0;
                while (s < self.batch.n_seq_id[i]) : (s += 1) {
                    const seq_id = self.batch.seq_id[i][s];
                    const sid: usize = @intCast(seq_id);

                    cur_seq_set[sid] &= self.seq_set.items[i];

                    if (cur_seq_set[sid] == 0) {
                        impl.logError("%s: sequence %d belongs to incompatible sequence sets (not allowed)\n", .{ f, seq_id });
                        return false;
                    }

                    if (pos < cur_seq_pos[sid]) {
                        impl.logError("%s: sequence %d positions are decreasing (not allowed)\n", .{ f, seq_id });
                        return false;
                    }

                    cur_seq_pos[sid] = pos;
                }
            }
        }

        self.splitReset();

        return true;
    }

    /// Ports `llama_batch_allocr`'s `ubatch_reserve` (llama-batch.cpp:484
    /// @d81235049): a well-defined ubatch of `n_seqs` sequences of
    /// `n_seq_tokens` tokens, all zero, for sizing compute buffers. It clears
    /// the allocator's batch.
    ///
    /// Parameters:
    /// - `n_seq_tokens`: tokens per sequence.
    /// - `n_seqs`: sequences, at most `LLAMA_MAX_SEQ`.
    ///
    /// Return: the ubatch, holding one reference the caller must `release`.
    pub fn ubatchReserve(self: *BatchAllocr, n_seq_tokens: u32, n_seqs: u32) Ubatch {
        const n_tokens = n_seq_tokens *% n_seqs;

        self.clear();
        self.splitReset();

        const n_pos_all: usize = @as(usize, n_tokens) * self.n_pos_per_embd;

        const udata = UbatchData.create();

        udata.token.appendNTimes(allocator, 0, n_tokens) catch oom();
        udata.pos.appendNTimes(allocator, 0, n_pos_all) catch oom();
        udata.n_seq_id.appendNTimes(allocator, 0, n_tokens) catch oom();
        udata.seq_id.appendNTimes(allocator, null, n_tokens) catch oom();
        udata.seq_idx.appendNTimes(allocator, -1, max_seq) catch oom();
        udata.output.appendNTimes(allocator, 0, n_tokens) catch oom();

        for (0..n_seqs) |s| {
            udata.seq_idx.items[s] = @intCast(s);
            udata.seq_id_unq.append(allocator, @intCast(s)) catch oom();
        }

        return .{
            .b_equal_seqs = 1,
            .n_tokens = n_tokens,
            .n_seq_tokens = n_seq_tokens,
            .n_seqs = n_seqs,
            .n_seqs_unq = n_seqs,
            .n_pos = self.n_pos_per_embd,
            .token = dataPtr(Token, &udata.token),
            .embd = null,
            .pos = dataPtr(Pos, &udata.pos),
            .n_seq_id = dataPtr(i32, &udata.n_seq_id),
            .seq_id = dataPtr([*c]SeqId, &udata.seq_id),
            .seq_id_unq = dataPtr(SeqId, &udata.seq_id_unq),
            .seq_idx = dataPtr(i32, &udata.seq_idx),
            .output = dataPtr(i8, &udata.output),
            .type = null,
            .decision_order = null,
            .data = udata,
        };
    }

    /// Ports `llama_batch_allocr`'s `get_batch` (llama-batch.cpp:532 @d81235049).
    ///
    /// Return: the sanitized batch, borrowed from the allocator.
    pub fn getBatch(self: *const BatchAllocr) *const c.llama_batch {
        return &self.batch;
    }

    /// Ports `llama_batch_allocr`'s `get_n_tokens` (llama-batch.cpp:536 @d81235049).
    ///
    /// Return: the batch's token count.
    pub fn getNTokens(self: *const BatchAllocr) u32 {
        return @bitCast(self.batch.n_tokens);
    }

    /// Ports `llama_batch_allocr`'s `get_n_outputs` (llama-batch.cpp:540 @d81235049).
    ///
    /// Return: how many tokens are outputs.
    pub fn getNOutputs(self: *const BatchAllocr) u32 {
        return self.n_outputs;
    }

    /// Ports `llama_batch_allocr`'s `get_n_used` (llama-batch.cpp:544 @d81235049).
    ///
    /// Return: how many tokens the splits since `splitReset` took.
    pub fn getNUsed(self: *const BatchAllocr) u32 {
        return self.n_used;
    }

    /// Ports `llama_batch_allocr`'s `get_out_ids` (llama-batch.cpp:548 @d81235049).
    ///
    /// Return: the output indices in the order the splits met them; the
    /// list is the allocator's, and the caller may change it, as the C++'s
    /// non-const reference allows.
    pub fn getOutIds(self: *BatchAllocr) *std.ArrayList(i32) {
        return &self.out_ids;
    }

    /// Ports `llama_batch_allocr`'s `seq_pos_min` (llama-batch.cpp:552 @d81235049).
    ///
    /// Parameters:
    /// - `seq_id`: the sequence, in `[0, LLAMA_MAX_SEQ)`.
    ///
    /// Return: its smallest position in the batch, or -1 when it has none.
    pub fn seqPosMin(self: *const BatchAllocr, seq_id: SeqId) Pos {
        const p = self.seq_pos[@intCast(seq_id)].items.items;
        return if (p.len == 0) -1 else p[0];
    }

    /// Ports `llama_batch_allocr`'s `seq_pos_max` (llama-batch.cpp:556 @d81235049).
    ///
    /// Parameters:
    /// - `seq_id`: the sequence, in `[0, LLAMA_MAX_SEQ)`.
    ///
    /// Return: its largest position in the batch, or -1 when it has none.
    pub fn seqPosMax(self: *const BatchAllocr, seq_id: SeqId) Pos {
        const p = self.seq_pos[@intCast(seq_id)].items.items;
        return if (p.len == 0) -1 else p[p.len - 1];
    }

    /// Ports `llama_batch_allocr`'s `split_reset` (llama-batch.cpp:560
    /// @d81235049): call once before splitting.
    ///
    /// Return: nothing.
    pub fn splitReset(self: *BatchAllocr) void {
        self.out_ids.clearRetainingCapacity();

        self.n_used = 0;

        self.used.clearRetainingCapacity();
        self.used.appendNTimes(allocator, false, self.getNTokens()) catch oom();
    }

    /// Ports `llama_batch_allocr`'s `split_simple` (llama-batch.cpp:569
    /// @d81235049): the next `n_ubatch` unused tokens, in batch order, as
    /// sequence sets of one token each.
    ///
    /// Parameters:
    /// - `n_ubatch`: the most tokens to take.
    ///
    /// Return: the ubatch, holding one reference the caller must `release`;
    /// empty when the batch is consumed.
    pub fn splitSimple(self: *BatchAllocr, n_ubatch: u32) Ubatch {
        const used = self.used.items;

        // the first unused token
        var cur_idx: usize = 0;
        while (cur_idx < used.len and used[cur_idx]) cur_idx += 1;

        if (cur_idx >= used.len) return .{};

        var idxs: std.ArrayList(i32) = .empty;
        defer idxs.deinit(allocator);

        while (true) {
            idxs.append(allocator, @intCast(cur_idx)) catch oom();

            used[cur_idx] = true;
            self.n_used += 1;

            cur_idx += 1;

            if (cur_idx >= used.len) break;

            if (idxs.items.len >= n_ubatch) break;
        }

        return self.ubatchAdd(idxs.items, @intCast(idxs.items.len), false);
    }

    fn seqSetIdxs(self: *BatchAllocr, set: SeqSet) []i32 {
        return self.seq_set_map.getPtr(set).?.items;
    }

    /// Ports `llama_batch_allocr`'s `split_equal` (llama-batch.cpp:603
    /// @d81235049): ubatches of non-overlapping sequence sets with the same
    /// number of tokens each.
    ///
    /// Parameters:
    /// - `n_ubatch`: the most tokens to take -- though when more than
    ///   `n_ubatch` sequence sets qualify, `n_ubatch + 1` sets of one token
    ///   are taken, as in the C++.
    /// - `sequential`: the sets' first sequence ids must increase by one; not
    ///   allowed with coupled sequences, which logs an error.
    /// - `n_keep_tail`: the fewest trailing tokens of a sequence that must
    ///   land in one ubatch; 0 for no constraint, else below `n_ubatch`.
    ///
    /// Return: the ubatch, holding one reference the caller must `release`;
    /// empty when the batch is consumed or the split is refused.
    pub fn splitEqual(self: *BatchAllocr, n_ubatch: u32, sequential: bool, n_keep_tail: u32) Ubatch {
        if (sequential and self.has_cpl) {
            impl.logError("%s: sequential split is not supported when there are coupled sequences in the input batch (you may need to use the -kvu flag)\n", .{"split_equal"});
            return .{};
        }

        const used = self.used.items;

        var cur_seq_set: std.ArrayList(SeqSet) = .empty;
        defer cur_seq_set.deinit(allocator);

        var last_seq_id: SeqId = -1;

        // the non-overlapping sequence sets in this ubatch
        for (0..@as(usize, @intCast(self.batch.n_tokens))) |i| {
            if (used[i]) continue;

            var add = true;

            for (cur_seq_set.items) |s| {
                if (s & self.seq_set.items[i] != 0) {
                    add = false;
                    break;
                }
            }

            // accept only increasing sequence ids
            if (sequential) {
                add = add and (cur_seq_set.items.len == 0 or self.batch.seq_id[i][0] == last_seq_id + 1);
            }

            if (add) {
                cur_seq_set.append(allocator, self.seq_set.items[i]) catch oom();

                last_seq_id = self.batch.seq_id[i][0];

                if (cur_seq_set.items.len > n_ubatch) break;
            }
        }

        var n_seqs: u32 = @intCast(cur_seq_set.items.len);

        if (n_seqs == 0) return .{};

        // each sequence set's current batch index
        const cur_idx = allocator.alloc(i32, n_seqs) catch oom();
        defer allocator.free(cur_idx);
        @memset(cur_idx, 0);

        for (0..n_seqs) |s| {
            const list = self.seqSetIdxs(cur_seq_set.items[s]);
            while (used[@intCast(list[@intCast(cur_idx[s])])]) cur_idx[s] += 1;
        }

        // each sequence set's batch indices, concatenated at the end
        const idxs_per_seq = allocator.alloc(std.ArrayList(i32), n_seqs) catch oom();
        @memset(idxs_per_seq, .empty);
        defer {
            for (idxs_per_seq) |*l| l.deinit(allocator);
            allocator.free(idxs_per_seq);
        }

        while (true) {
            // expand only while every set has another unused token
            var can_expand = true;

            for (0..n_seqs) |s| {
                if (cur_idx[s] >= @as(i32, @intCast(self.seqSetIdxs(cur_seq_set.items[s]).len))) {
                    can_expand = false;
                    break;
                }
            }

            if (!can_expand) break;

            for (0..n_seqs) |s| {
                const idx = self.seqSetIdxs(cur_seq_set.items[s])[@intCast(cur_idx[s])];

                idxs_per_seq[s].append(allocator, idx) catch oom();

                used[@intCast(idx)] = true;
                self.n_used += 1;

                cur_idx[s] += 1;
            }

            if ((idxs_per_seq[0].items.len + 1) * n_seqs > n_ubatch) break;
        }

        // with n_keep_tail, keep only the sets that finish here or leave at
        // least n_keep_tail tokens for later, so no set's trailing
        // n_keep_tail tokens are split across ubatches
        if (n_keep_tail > 0) {
            impl.assert(n_ubatch > n_keep_tail, "llama-batch.cpp", 702, "n_ubatch > n_keep_tail");

            const Remaining = struct {
                fn of(a: *BatchAllocr, sets: []const SeqSet, idx: []const i32, s: usize) u32 {
                    return @truncate(a.seqSetIdxs(sets[s]).len -% @as(usize, @intCast(idx[s])));
                }
            };

            // the longest prefix of sets that satisfy it, keeping the
            // sequence ids sequential
            var n_keep: u32 = 0;
            while (n_keep < n_seqs) {
                const remaining = Remaining.of(self, cur_seq_set.items, cur_idx, n_keep);

                if (remaining != 0 and remaining < n_keep_tail) break;

                n_keep += 1;
            }

            // every set violates it: resolve the first and emit it alone
            if (n_keep == 0) {
                const idxs = &idxs_per_seq[0];

                const seq_idxs = self.seqSetIdxs(cur_seq_set.items[0]);

                if (idxs.items.len + Remaining.of(self, cur_seq_set.items, cur_idx, 0) <= n_ubatch) {
                    // extend the set to completion
                    while (Remaining.of(self, cur_seq_set.items, cur_idx, 0) > 0) {
                        const idx = seq_idxs[@intCast(cur_idx[0])];

                        idxs.append(allocator, idx) catch oom();

                        used[@intCast(idx)] = true;
                        self.n_used += 1;

                        cur_idx[0] += 1;
                    }
                } else {
                    // truncate it so that at least n_keep_tail tokens remain
                    while (Remaining.of(self, cur_seq_set.items, cur_idx, 0) < n_keep_tail) {
                        used[@intCast(idxs.items[idxs.items.len - 1])] = false;
                        self.n_used -%= 1;

                        _ = idxs.pop();

                        cur_idx[0] -= 1;
                    }
                }

                n_keep = 1;
            }

            // return the deferred sets' tokens to the pool
            for (n_keep..n_seqs) |s| {
                for (idxs_per_seq[s].items) |idx| {
                    used[@intCast(idx)] = false;
                    self.n_used -%= 1;
                }
            }

            n_seqs = n_keep;
        }

        // concatenate the per-set lists
        var idxs: std.ArrayList(i32) = .empty;
        defer idxs.deinit(allocator);

        for (0..n_seqs) |s| {
            idxs.appendSlice(allocator, idxs_per_seq[s].items) catch oom();
        }

        return self.ubatchAdd(idxs.items, n_seqs, true);
    }

    /// Ports `llama_batch_allocr`'s `split_seq` (llama-batch.cpp:774
    /// @d81235049): one sequence set per ubatch, taking later tokens whose
    /// set is a subset of the current one.
    ///
    /// Parameters:
    /// - `n_ubatch`: the most tokens to take.
    ///
    /// Return: the ubatch, holding one reference the caller must `release`;
    /// empty when the batch is consumed.
    pub fn splitSeq(self: *BatchAllocr, n_ubatch: u32) Ubatch {
        const used = self.used.items;

        // the first unused token
        var cur_idx: u32 = 0;
        while (cur_idx < used.len and used[cur_idx]) cur_idx += 1;

        if (cur_idx >= used.len) return .{};

        // the starting sequence set; later tokens join only if theirs is a
        // subset of the current one
        var cur_seq_set = self.seq_set.items[cur_idx];

        var idxs: std.ArrayList(i32) = .empty;
        defer idxs.deinit(allocator);

        while (true) {
            idxs.append(allocator, @intCast(cur_idx)) catch oom();

            used[cur_idx] = true;
            self.n_used += 1;

            if (idxs.items.len >= n_ubatch) break;

            while (true) {
                cur_idx += 1;
                if (!(cur_idx < self.getNTokens() and (used[cur_idx] or ((cur_seq_set & self.seq_set.items[cur_idx]) != self.seq_set.items[cur_idx])))) break;
            }

            if (cur_idx == self.getNTokens()) break;

            cur_seq_set = self.seq_set.items[cur_idx];
        }

        return self.ubatchAdd(idxs.items, 1, true);
    }

    /// Ports `llama_batch_allocr`'s `clear` (llama-batch.cpp:816 @d81235049).
    /// `has_cpl`, `used`, `out_ids` and `n_used` survive it, as in the C++.
    fn clear(self: *BatchAllocr) void {
        self.n_outputs = 0;

        self.batch = .{};

        self.token_vec.clearRetainingCapacity();
        self.embd_vec.clearRetainingCapacity();
        self.is_embd_vec.clearRetainingCapacity();
        self.seq_id_data.clearRetainingCapacity();
        self.pos.clearRetainingCapacity();
        self.n_seq_id.clearRetainingCapacity();
        self.seq_id.clearRetainingCapacity();
        self.seq_id_unq.clearRetainingCapacity();
        self.output.clearRetainingCapacity();
        self.decision_order.clearRetainingCapacity();

        for (&self.seq_pos) |*p| p.items.clearRetainingCapacity();

        self.seq_cpl = @splat(0);

        self.seq_set.clearRetainingCapacity();

        self.clearSeqSetMap();

        @memset(self.seq_idx.items, -1);
    }

    /// Ports `llama_batch_allocr`'s `ubatch_add` (llama-batch.cpp:847
    /// @d81235049): copies the tokens at `idxs` into a new ubatch and
    /// records its outputs in `out_ids`. From a mixed batch, a ubatch whose
    /// rows are all of one kind is a plain token or embedding ubatch.
    fn ubatchAdd(self: *BatchAllocr, idxs: []const i32, n_seqs: u32, equal_seqs: bool) Ubatch {
        const n_tokens: u32 = @intCast(idxs.len);

        const udata = UbatchData.create();

        const mixed_batch = self.is_embd_vec.items.len != 0;

        var n_embd_rows: u32 = 0;
        if (mixed_batch) {
            for (idxs) |idx| n_embd_rows += @intCast(self.is_embd_vec.items[@intCast(idx)]);
        }
        const mixed = mixed_batch and n_embd_rows > 0 and n_embd_rows < n_tokens;
        const use_token = self.batch.token != null and !(mixed_batch and n_embd_rows == n_tokens);
        const use_embd = self.batch.embd != null and !(mixed_batch and n_embd_rows == 0);

        const n_embd_all: usize = if (use_embd) @as(usize, n_tokens) * self.n_embd else 0;
        const n_pos_all: usize = @as(usize, n_tokens) * self.n_pos_per_embd;

        udata.token.appendNTimes(allocator, 0, n_tokens) catch oom();
        udata.embd.appendNTimes(allocator, 0, n_embd_all) catch oom();
        udata.pos.appendNTimes(allocator, 0, n_pos_all) catch oom();
        udata.n_seq_id.appendNTimes(allocator, 0, n_tokens) catch oom();
        udata.seq_id.appendNTimes(allocator, null, n_tokens) catch oom();
        udata.seq_idx.appendNTimes(allocator, -1, max_seq) catch oom();
        udata.output.appendNTimes(allocator, 0, n_tokens) catch oom();
        udata.type.appendNTimes(allocator, 0, if (mixed) n_tokens else 0) catch oom();
        udata.decision_order.appendNTimes(allocator, 0, if (self.decision_order.items.len == 0) 0 else n_tokens) catch oom();

        udata.batch_idxs.appendSlice(allocator, idxs) catch oom();
        udata.seq_id_data.ensureTotalCapacity(allocator, n_tokens) catch oom();

        var seq_set_unq: SeqSet = 0;

        const n_embd: usize = self.n_embd;
        const batch_n_tokens: usize = @intCast(self.batch.n_tokens);

        for (idxs, 0..) |idx_i, i| {
            const idx: usize = @intCast(idx_i);

            if (use_token) {
                udata.token.items[i] = self.batch.token[idx];
            }

            if (use_embd) {
                @memcpy(udata.embd.items[i * n_embd ..][0..n_embd], self.batch.embd[idx * n_embd ..][0..n_embd]);
            }

            if (mixed) {
                udata.type.items[i] = self.is_embd_vec.items[idx];
            }

            for (0..self.n_pos_per_embd) |j| {
                udata.pos.items[j * n_tokens + i] = self.batch.pos[j * batch_n_tokens + idx];
            }

            udata.n_seq_id.items[i] = self.batch.n_seq_id[idx];
            udata.output.items[i] = self.batch.logits[idx];

            if (self.decision_order.items.len != 0) {
                udata.decision_order.items[i] = self.decision_order.items[idx];
            }

            var s: usize = 0;
            while (s < udata.n_seq_id.items[i]) : (s += 1) {
                const seq_id = self.batch.seq_id[idx][s];

                udata.seq_id_data.append(allocator, seq_id) catch oom();
                seq_set_unq |= bit(seq_id);
            }

            if (udata.output.items[i] != 0) {
                self.out_ids.append(allocator, idx_i) catch oom();
            }
        }

        var seq_id_ptr: [*c]SeqId = udata.seq_id_data.items.ptr;
        for (0..idxs.len) |i| {
            udata.seq_id.items[i] = seq_id_ptr;
            seq_id_ptr += @as(usize, @intCast(udata.n_seq_id.items[i]));
        }

        for (0..self.n_seq_max) |s| {
            if (seq_set_unq & bit(@intCast(s)) != 0) {
                udata.seq_idx.items[s] = @intCast(udata.seq_id_unq.items.len);
                udata.seq_id_unq.append(allocator, @intCast(s)) catch oom();
            }
        }

        const res: Ubatch = .{
            .b_equal_seqs = @intFromBool(equal_seqs),
            .n_tokens = n_tokens,
            .n_seq_tokens = n_tokens / n_seqs,
            .n_seqs = n_seqs,
            .n_seqs_unq = @intCast(udata.seq_id_unq.items.len),
            .n_pos = self.n_pos_per_embd,
            .token = if (use_token) dataPtr(Token, &udata.token) else null,
            .embd = if (use_embd) dataPtr(f32, &udata.embd) else null,
            .pos = dataPtr(Pos, &udata.pos),
            .n_seq_id = dataPtr(i32, &udata.n_seq_id),
            .seq_id = dataPtr([*c]SeqId, &udata.seq_id),
            .seq_id_unq = dataPtr(SeqId, &udata.seq_id_unq),
            .seq_idx = dataPtr(i32, &udata.seq_idx),
            .output = dataPtr(i8, &udata.output),
            .type = if (mixed) dataPtr(i8, &udata.type) else null,
            .decision_order = dataPtr(i32, &udata.decision_order),
            .data = udata,
        };

        if (self.debug > 0) {
            impl.logDebug("%s: added ubatch to split:\n", .{"ubatch_add"});

            self.ubatchPrint(&res, self.debug);
        }

        return res;
    }

    /// Ports `llama_batch_allocr`'s `ubatch_print` (llama-batch.cpp:965
    /// @d81235049), for `LLAMA_BATCH_DEBUG`.
    fn ubatchPrint(self: *const BatchAllocr, ubatch: *const Ubatch, debug: c_int) void {
        if (debug <= 0) return;
        const f = "ubatch_print";

        impl.logDebug("%s:   equal_seqs   = %d\n", .{ f, @as(c_int, @intFromBool(ubatch.equalSeqs())) });
        impl.logDebug("%s:   n_tokens     = %d\n", .{ f, cInt(ubatch.n_tokens) });
        impl.logDebug("%s:   n_seq_tokens = %d\n", .{ f, cInt(ubatch.n_seq_tokens) });
        impl.logDebug("%s:   n_seqs       = %d\n", .{ f, cInt(ubatch.n_seqs) });
        impl.logDebug("%s:   n_seqs_unq   = %d\n", .{ f, cInt(ubatch.n_seqs_unq) });

        var ss_seq_id_unq: std.ArrayList(u8) = .empty;
        defer ss_seq_id_unq.deinit(allocator);
        var ss_seq_idx: std.ArrayList(u8) = .empty;
        defer ss_seq_idx.deinit(allocator);

        ss_seq_id_unq.appendSlice(allocator, "[ ") catch oom();
        ss_seq_idx.append(allocator, '[') catch oom();

        for (0..ubatch.n_seqs_unq) |s| {
            ss_seq_id_unq.print(allocator, "{d} ", .{ubatch.seq_id_unq[s]}) catch oom();
        }

        for (0..max_seq) |s| {
            if (ubatch.seq_idx[s] >= 0) {
                ss_seq_idx.print(allocator, "{d}", .{@rem(ubatch.seq_idx[s], 10)}) catch oom();
            } else {
                ss_seq_idx.append(allocator, '.') catch oom();
            }
        }

        ss_seq_id_unq.appendSlice(allocator, "]\x00") catch oom();
        ss_seq_idx.appendSlice(allocator, "]\x00") catch oom();

        impl.logDebug("%s:   token      = %p\n", .{ f, @as(?*anyopaque, @ptrCast(ubatch.token)) });
        impl.logDebug("%s:   embd       = %p\n", .{ f, @as(?*anyopaque, @ptrCast(ubatch.embd)) });
        impl.logDebug("%s:   pos        = %p\n", .{ f, @as(?*anyopaque, @ptrCast(ubatch.pos)) });
        impl.logDebug("%s:   n_seq_id   = %p\n", .{ f, @as(?*anyopaque, @ptrCast(ubatch.n_seq_id)) });
        impl.logDebug("%s:   seq_id     = %p\n", .{ f, @as(?*anyopaque, @ptrCast(ubatch.seq_id)) });
        impl.logDebug("%s:   seq_id_unq = %s\n", .{ f, @as([*:0]const u8, @ptrCast(ss_seq_id_unq.items.ptr)) });
        impl.logDebug("%s:   seq_idx    = %s\n", .{ f, @as([*:0]const u8, @ptrCast(ss_seq_idx.items.ptr)) });
        impl.logDebug("%s:   output     = %p\n", .{ f, @as(?*anyopaque, @ptrCast(ubatch.output)) });
        impl.logDebug("%s:   type       = %p\n", .{ f, @as(?*anyopaque, @ptrCast(ubatch.type)) });
        impl.logDebug("%s:   n_outputs  = %d\n", .{ f, cInt(self.n_outputs) });

        var seq_id_max: SeqId = 0;
        for (0..ubatch.n_tokens) |i| {
            var s: usize = 0;
            while (s < ubatch.n_seq_id[i]) : (s += 1) {
                seq_id_max = @max(seq_id_max, ubatch.seq_id[i][s]);
            }
        }
        seq_id_max += 1;

        const marks = allocator.alloc(i8, @intCast(seq_id_max)) catch oom();
        defer allocator.free(marks);
        var ss: std.ArrayList(u8) = .empty;
        defer ss.deinit(allocator);
        var piece: std.ArrayList(u8) = .empty;
        defer piece.deinit(allocator);

        impl.logDebug("%s:   token     = [\n", .{f});
        for (0..ubatch.n_tokens) |i| {
            @memset(marks, 0);

            var s: usize = 0;
            while (s < ubatch.n_seq_id[i]) : (s += 1) {
                marks[@intCast(ubatch.seq_id[i][s])] = 1;
            }

            ss.clearRetainingCapacity();
            for (marks, 0..) |m, k| {
                if (m != 0) {
                    ss.print(allocator, "{d}", .{k % 10}) catch oom();
                } else {
                    ss.append(allocator, '.') catch oom();
                }
            }
            ss.append(allocator, 0) catch oom();
            const ss_z: [*:0]const u8 = @ptrCast(ss.items.ptr);

            const ii: c_int = @intCast(i);
            if (ubatch.token != null and !(ubatch.isMixed() and ubatch.type[i] != 0)) {
                var exc: impl.Exception = .{};
                const text = self.vocab.?.cachedPiece(ubatch.token[i], &exc) catch exc.terminate("llama-batch.cpp", 1035);
                piece.clearRetainingCapacity();
                piece.appendSlice(allocator, text) catch oom();
                piece.append(allocator, 0) catch oom();

                impl.logDebug("%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n", .{
                    f,               ii,
                    ubatch.token[i], @as([*:0]const u8, @ptrCast(piece.items.ptr)),
                    ubatch.pos[i],   ubatch.n_seq_id[i],
                    ss_z,            @as(c_int, ubatch.output[i]),
                });
            } else {
                impl.logDebug("%s:  %4d: [embd], pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n", .{
                    f, ii, ubatch.pos[i], ubatch.n_seq_id[i], ss_z, @as(c_int, ubatch.output[i]),
                });
            }
        }
        impl.logDebug("%s:   ]\n", .{f});
    }
};

// -----------------------------------------------------------------------------
// llama_batch_ext

/// `LLAMA_TOKEN_NULL`.
pub const null_token: Token = -1;

/// `GGML_MROPE_SECTIONS`: the positions an entry holds.
pub const mrope_sections = 4;

/// A `llama_batch_ext` entry's `std::unordered_set<llama_seq_id>`: the ids
/// in insertion order, without duplicates, from which `ordered` replays
/// libc++'s iteration order.
pub const SeqIdSet = struct {
    items: std.ArrayList(SeqId) = .empty,

    fn deinit(self: *SeqIdSet) void {
        self.items.deinit(allocator);
    }

    /// `insert(s)`: a no-op when `s` is already present.
    fn insert(self: *SeqIdSet, s: SeqId) void {
        for (self.items.items) |x| if (x == s) return;
        self.items.append(allocator, s) catch oom();
    }

    /// `size()`.
    pub fn count(self: *const SeqIdSet) usize {
        return self.items.items.len;
    }

    /// The set in libc++'s iteration order.
    ///
    /// Parameters:
    /// - `out`: `count()` slots to fill.
    ///
    /// Return: nothing.
    pub fn ordered(self: *const SeqIdSet, out: []SeqId) void {
        const items = self.items.items;
        if (items.len <= 1) {
            @memcpy(out, items);
            return;
        }
        var m = libcxx.I32Map(void).init(allocator) catch oom();
        defer m.deinit();
        for (items) |s| m.put(s, {}) catch oom();
        var it = m.iterator();
        var k: usize = 0;
        while (it.next()) |e| : (k += 1) out[k] = e.key;
    }
};

/// Ports `llama_batch_ext_select_n_embd_inp` (llama-batch.cpp:1112
/// @d81235049): the width of a batch's embedding rows. MTP hook batches carry
/// the target model's hidden state, DFlash batches the fused target features
/// at the encoder's input width, and the rest token embeddings.
///
/// Parameters:
/// - `ctx_type`: the context's type.
/// - `arch`: the model's architecture.
/// - `hparams`: the model's hyperparameters.
///
/// Return: the row width, in floats.
pub fn selectNEmbdInp(ctx_type: c.enum_llama_context_type, arch: arch_mod.Arch, hparams: *const hparams_mod.Hparams) usize {
    if (ctx_type == c.LLAMA_CONTEXT_TYPE_MTP) {
        return hparams.nEmbdOut();
    }
    if (arch == .dflash) {
        return hparams.nEmbdInpEnc();
    }
    return hparams.nEmbdInp();
}

/// Ports `llama_batch_ext` (llama-batch.h:93 @d81235049): the extended
/// batch, a list of entries each carrying a token id, an embedding or both,
/// its sequences, positions and output flag. The handle behind
/// `struct llama_batch_ext *`.
pub const BatchExt = struct {
    /// The most entries the batch holds.
    n_tokens_max: usize,
    /// The decoder's embedding row width.
    n_embd_inp: usize,
    /// The encoder's embedding row width (eagle3 and DFlash features).
    n_embd_inp_enc: usize,
    /// The most sequences.
    n_seq_max: SeqId,
    /// The memory, for position inference; null for none.
    mem: ?MemoryPos,
    /// One past the largest token id accepted.
    n_vocab: Token,
    n_pos_per_embd: usize,

    /// This batch's embedding row width, set by the first `setTokenEmbd`:
    /// `n_embd_inp` or `n_embd_inp_enc`, checked against the graph's input
    /// by encode and decode.
    n_embd: usize = 0,

    tokens: std.ArrayList(Entry) = .empty,
    embd: std.ArrayList(f32) = .empty,

    /// Ports `llama_batch_ext`'s `token` (llama-batch.h:106 @d81235049).
    pub const Entry = struct {
        id: Token = null_token,
        /// Whether `embd_off` is set.
        has_embd: bool = false,
        /// The offset of the entry's embedding in `embd`.
        embd_off: usize = 0,
        output: bool = false,
        decision_order: i32 = 0,
        seq_ids: SeqIdSet = .{},
        pos: [mrope_sections]Pos = .{ 0, 0, 0, 0 },
    };

    /// Ports the test constructor of `llama_batch_ext` (llama-batch.cpp:1133
    /// @d81235049), which the context constructor calls with the context's
    /// values: `Context.batchExtCreate` ports that one (llama-batch.cpp:1122
    /// @d81235049).
    ///
    /// Return: an empty batch on the heap; free it with `destroy`.
    pub fn create(
        n_tokens_max: usize,
        n_embd_inp: usize,
        n_embd_inp_enc: usize,
        n_seq_max: SeqId,
        mem: ?MemoryPos,
        n_vocab: Token,
        n_pos_per_embd: usize,
    ) *BatchExt {
        const self = allocator.create(BatchExt) catch oom();
        self.* = .{
            .n_tokens_max = n_tokens_max,
            .n_embd_inp = n_embd_inp,
            .n_embd_inp_enc = n_embd_inp_enc,
            .n_seq_max = n_seq_max,
            .mem = mem,
            .n_vocab = n_vocab,
            .n_pos_per_embd = n_pos_per_embd,
        };
        return self;
    }

    /// Frees the batch, as `delete` does.
    ///
    /// Return: nothing.
    pub fn destroy(self: *BatchExt) void {
        self.clear();
        self.tokens.deinit(allocator);
        self.embd.deinit(allocator);
        allocator.destroy(self);
    }

    /// Ports `llama_batch_ext`'s `clear` (llama-batch.cpp:1151 @d81235049).
    ///
    /// Return: nothing.
    pub fn clear(self: *BatchExt) void {
        for (self.tokens.items) |*t| t.seq_ids.deinit();
        self.tokens.clearRetainingCapacity();
        self.embd.clearRetainingCapacity();
        self.n_embd = 0;
    }

    fn validIdx(self: *const BatchExt, idx: i32) bool {
        return idx >= 0 and idx < @as(i32, @truncate(@as(isize, @bitCast(self.tokens.items.len))));
    }

    /// Ports `llama_batch_ext`'s `add_token` (llama-batch.cpp:1157
    /// @d81235049): an entry in one sequence, with no position yet.
    ///
    /// Return: its index, -1 when the batch is full, -3 for an invalid
    /// sequence id.
    pub fn addToken(self: *BatchExt, seq_id: SeqId) i32 {
        if (self.tokens.items.len >= self.n_tokens_max) {
            return -1; // size limit reached
        }
        if (seq_id < 0 or seq_id >= self.n_seq_max) {
            return -3; // invalid sequence id
        }

        var t: Entry = .{};
        t.seq_ids.insert(seq_id);

        self.tokens.append(allocator, t) catch oom();

        return @truncate(@as(isize, @bitCast(self.tokens.items.len - 1)));
    }

    /// Ports `llama_batch_ext`'s `add_seq` (llama-batch.cpp:1174 @d81235049).
    ///
    /// Return: false for an invalid index or sequence id.
    pub fn addSeq(self: *BatchExt, idx: i32, seq_id: SeqId) bool {
        if (!self.validIdx(idx)) {
            return false;
        }
        if (seq_id < 0 or seq_id >= self.n_seq_max) {
            return false;
        }

        self.tokens.items[@intCast(idx)].seq_ids.insert(seq_id);

        return true;
    }

    /// Ports `llama_batch_ext`'s `set_token_id` (llama-batch.cpp:1189
    /// @d81235049).
    ///
    /// Return: false for an invalid index or token id.
    pub fn setTokenId(self: *BatchExt, idx: i32, id: Token) bool {
        if (!self.validIdx(idx)) {
            return false;
        }
        if (id < 0 or id >= self.n_vocab) {
            return false;
        }
        self.tokens.items[@intCast(idx)].id = id;
        return true;
    }

    /// Ports `llama_batch_ext`'s `set_token_embd` (llama-batch.cpp:1200
    /// @d81235049): the entry's embedding, `n_rows * n_embd` floats, which
    /// must be the batch's row width -- the first one sets it.
    ///
    /// Return: false for an invalid index, a null or mis-sized embedding, or
    /// an entry that has one already.
    pub fn setTokenEmbd(self: *BatchExt, idx: i32, embd_in: c.struct_llama_embd) bool {
        const f = "set_token_embd";
        if (!self.validIdx(idx)) {
            return false;
        }
        if (embd_in.data == null) {
            return false;
        }

        const n_total = embd_in.n_rows *% embd_in.n_embd;
        if (self.n_embd == 0) {
            if (n_total != self.n_embd_inp and n_total != self.n_embd_inp_enc) {
                impl.logError("%s: embedding size mismatch, got %zu rows x %zu = %zu, expected %zu or %zu\n", .{ f, embd_in.n_rows, embd_in.n_embd, n_total, self.n_embd_inp, self.n_embd_inp_enc });
                return false;
            }
            self.n_embd = n_total;
        } else if (n_total != self.n_embd) {
            impl.logError("%s: embedding size mismatch, got %zu rows x %zu = %zu, expected %zu\n", .{ f, embd_in.n_rows, embd_in.n_embd, n_total, self.n_embd });
            return false;
        }

        const t = &self.tokens.items[@intCast(idx)];

        if (t.has_embd) {
            impl.logError("%s: embedding for token %d is already set\n", .{ f, idx });
            return false;
        }

        t.has_embd = true;
        t.embd_off = self.embd.items.len;
        self.embd.appendSlice(allocator, embd_in.data[0..n_total]) catch oom();

        return true;
    }

    /// Ports `llama_batch_ext`'s `set_token_pos` (llama-batch.cpp:1236
    /// @d81235049): one position for an entry with a token id, and
    /// `n_pos_per_embd` for an embedding.
    ///
    /// Return: false for an invalid index or a null `pos_in`.
    pub fn setTokenPos(self: *BatchExt, idx: i32, pos_in: [*c]const Pos) bool {
        if (!self.validIdx(idx)) {
            return false;
        }
        if (pos_in == null) {
            return false;
        }

        const t = &self.tokens.items[@intCast(idx)];

        const n_pos: usize = if (t.id != null_token) 1 else self.n_pos_per_embd;
        for (0..n_pos) |i| {
            t.pos[i] = pos_in[i];
        }

        return true;
    }

    /// Ports `llama_batch_ext`'s `set_output` (llama-batch.cpp:1254
    /// @d81235049).
    ///
    /// Return: false for an invalid index.
    pub fn setOutput(self: *BatchExt, idx: i32, output_last: bool) bool {
        if (!self.validIdx(idx)) {
            return false;
        }
        self.tokens.items[@intCast(idx)].output = output_last;
        return true;
    }

    /// Ports `llama_batch_ext`'s `set_decision_order` (llama-batch.cpp:1262
    /// @d81235049).
    ///
    /// Return: false for an invalid index.
    pub fn setDecisionOrder(self: *BatchExt, idx: i32, order: i32) bool {
        if (!self.validIdx(idx)) {
            return false;
        }
        self.tokens.items[@intCast(idx)].decision_order = order;
        return true;
    }
};

/// Ports `llama_batch_compat`'s `init` (llama-batch.cpp:1344 @d81235049):
/// fills an extended batch from a `llama_batch`, generating what the old API
/// leaves out -- one sequence 0, positions continuing from the memory along
/// each entry's first sequence, and the last entry as the only output.
///
/// Parameters:
/// - `dst`: the batch to append to.
/// - `batch_inp`: the old batch; read only during the call.
/// - `n_embd_row_in`: the width of its embedding rows, 0 for the decoder's.
///
/// Return: nothing.
pub fn compatInit(dst: *BatchExt, batch_inp: c.llama_batch, n_embd_row_in: usize) void {
    const n_embd_row = if (n_embd_row_in == 0) dst.n_embd_inp else n_embd_row_in;

    // a batch can carry both, the MTP hook batches for one
    const has_token = batch_inp.token != null;
    const has_embd = batch_inp.embd != null;

    const default_seq_id: SeqId = 0;

    // positions continue from the memory when batch_inp.pos is null
    const n_seq_max: usize = @intCast(@max(dst.n_seq_max, 0));
    const pos_next = allocator.alloc(Pos, n_seq_max) catch oom();
    defer allocator.free(pos_next);
    for (0..n_seq_max) |s| {
        const p_max: Pos = if (dst.mem) |m| m.posMax(@intCast(s)) else -1;
        pos_next[s] = p_max + 1;
    }

    var i: i32 = 0;
    while (i < batch_inp.n_tokens) : (i += 1) {
        const iu: usize = @intCast(i);
        const n_sid: i32 = if (batch_inp.n_seq_id != null) batch_inp.n_seq_id[iu] else 1;
        const sids: [*c]const SeqId = if (batch_inp.seq_id != null) batch_inp.seq_id[iu] else &default_seq_id;

        var t: BatchExt.Entry = .{};

        var s: i32 = 0;
        while (s < n_sid) : (s += 1) {
            t.seq_ids.insert(sids[@intCast(s)]);
        }

        if (batch_inp.pos != null) {
            if (has_token) {
                // a token batch: one position per token
                t.pos[0] = batch_inp.pos[iu];
            } else {
                // an embedding batch (M-RoPE): section-major, pos[j*n_tokens + i]
                for (0..dst.n_pos_per_embd) |j| {
                    t.pos[j] = batch_inp.pos[@as(usize, @intCast(@as(i32, @intCast(j)) * batch_inp.n_tokens + i))];
                }
            }
        } else {
            // from the first sequence id; one out of range makes the C++
            // index past its vector, and the batch fails validation anyway
            const sid = sids[0];
            if (sid >= 0 and @as(usize, @intCast(sid)) < n_seq_max) {
                t.pos[0] = pos_next[@intCast(sid)];
                pos_next[@intCast(sid)] += 1;
            }
        }

        if (has_token) {
            t.id = batch_inp.token[iu];
        }

        if (has_embd) {
            t.has_embd = true;
            t.embd_off = dst.embd.items.len;
            const src = batch_inp.embd + iu * n_embd_row;
            dst.embd.appendSlice(allocator, src[0..n_embd_row]) catch oom();
            dst.n_embd = n_embd_row;
        }

        // without a logits array, only the last token is an output
        t.output = if (batch_inp.logits != null) batch_inp.logits[iu] != 0 else i == batch_inp.n_tokens - 1;

        dst.tokens.append(allocator, t) catch oom();
    }
}

// -----------------------------------------------------------------------------
// The C interface

/// Ports `llama_batch_get_one` (llama-batch.cpp:1051 @d81235049): a batch of
/// one sequence over `tokens`, everything else left for `initBatch` to fill.
///
/// Parameters:
/// - `tokens`: the tokens; borrowed by the batch.
/// - `n_tokens`: how many.
///
/// Return: the batch; it owns nothing.
pub fn batchGetOne(tokens: [*c]Token, n_tokens: i32) c.llama_batch {
    return .{
        .n_tokens = n_tokens,
        .token = tokens,
        .embd = null,
        .pos = null,
        .n_seq_id = null,
        .seq_id = null,
        .logits = null,
    };
}

fn usz(n: i32) usize {
    return @bitCast(@as(isize, n));
}

fn cMalloc(comptime T: type, n: usize) [*c]T {
    return @ptrCast(@alignCast(std.c.malloc(n)));
}

/// Ports `llama_batch_init` (llama-batch.cpp:1065 @d81235049): a batch whose
/// arrays are `malloc`ed for `n_tokens_alloc` tokens, with `n_tokens` 0.
///
/// Parameters:
/// - `n_tokens_alloc`: the capacity, in tokens.
/// - `embd`: non-zero for an embeddings batch of this width, and then no
///   token array.
/// - `n_seq_max`: the sequence ids each token can hold.
///
/// Return: the batch; free it with `batchFree`, or the C's
/// `llama_batch_free`, since both use `malloc`.
pub fn batchInit(n_tokens_alloc: i32, embd: i32, n_seq_max: i32) c.llama_batch {
    var batch: c.llama_batch = .{};

    const n = usz(n_tokens_alloc);
    if (embd != 0) {
        batch.embd = cMalloc(f32, @sizeOf(f32) *% n *% usz(embd));
    } else {
        batch.token = cMalloc(Token, @sizeOf(Token) *% n);
    }

    batch.pos = cMalloc(Pos, @sizeOf(Pos) *% n);
    batch.n_seq_id = cMalloc(i32, @sizeOf(i32) *% n);
    batch.seq_id = cMalloc([*c]SeqId, @sizeOf([*c]SeqId) *% (n +% 1));
    var i: i32 = 0;
    while (i < n_tokens_alloc) : (i += 1) {
        batch.seq_id[usz(i)] = cMalloc(SeqId, @sizeOf(SeqId) *% usz(n_seq_max));
    }
    batch.seq_id[n] = null;

    batch.logits = cMalloc(i8, @sizeOf(i8) *% n);

    return batch;
}

/// Ports `llama_batch_free` (llama-batch.cpp:1095 @d81235049).
///
/// Parameters:
/// - `batch`: a batch from `batchInit`, or the C's `llama_batch_init`.
///
/// Return: nothing.
pub fn batchFree(batch: c.llama_batch) void {
    if (batch.token != null) std.c.free(@ptrCast(batch.token));
    if (batch.embd != null) std.c.free(@ptrCast(batch.embd));
    if (batch.pos != null) std.c.free(@ptrCast(batch.pos));
    if (batch.n_seq_id != null) std.c.free(@ptrCast(batch.n_seq_id));
    if (batch.seq_id != null) {
        var i: usize = 0;
        while (batch.seq_id[i] != null) : (i += 1) {
            std.c.free(@ptrCast(batch.seq_id[i]));
        }
        std.c.free(@ptrCast(batch.seq_id));
    }
    if (batch.logits != null) std.c.free(@ptrCast(batch.logits));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(UbatchData);
    std.testing.refAllDecls(Ubatch);
    std.testing.refAllDecls(MemoryPos);
    std.testing.refAllDecls(BatchAllocr);
}

test "the empty ubatch is all zeros and null" {
    const u: Ubatch = .{};
    try std.testing.expectEqual(@as(u32, 0), u.n_tokens);
    try std.testing.expect(u.token == null and u.data == null);
    try std.testing.expect(!u.equalSeqs());
    try std.testing.expect(!u.isPos2d());
}

test "a retained copy outlives the release of its source" {
    var a = BatchAllocr.init(1, false);
    defer a.deinit();

    var u = a.ubatchReserve(3, 2);
    try std.testing.expectEqual(@as(u32, 6), u.n_tokens);
    try std.testing.expectEqual(@as(u32, 2), u.n_seqs_unq);
    try std.testing.expectEqual(@as(i32, 1), u.seq_idx[1]);
    try std.testing.expectEqual(@as(i32, -1), u.seq_idx[2]);

    u.retain();
    var copy = u;
    u.release();
    try std.testing.expect(u.data == null);
    try std.testing.expectEqual(@as(c.llama_seq_id, 1), copy.seq_id_unq[1]);
    copy.release();
}

test "the position set keeps a sorted distinct list" {
    var p: PosSet = .{};
    defer p.items.deinit(allocator);
    for ([_]Pos{ 5, 1, 3, 5, 1, 9 }) |v| p.insert(v);
    try std.testing.expectEqualSlices(Pos, &.{ 1, 3, 5, 9 }, p.items.items);
}

test "batchInit and batchFree pair up" {
    const b = batchInit(4, 0, 2);
    defer batchFree(b);
    try std.testing.expect(b.token != null and b.embd == null);
    try std.testing.expect(b.seq_id[4] == null and b.seq_id[3] != null);

    const e = batchGetOne(b.token, 4);
    try std.testing.expectEqual(@as(i32, 4), e.n_tokens);
    try std.testing.expect(e.pos == null and e.logits == null);
}
