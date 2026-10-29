//! Batches: sanitizing a `llama_batch` and splitting it into ubatches.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-batch.h`   -- `llama_ubatch`, `llama_batch_allocr`
//! - `llama.cpp/src/llama-batch.cpp` -- their bodies, and `llama_batch_get_one`,
//!   `llama_batch_init` and `llama_batch_free`
//! - `llama.cpp/src/llama-cparams.h` -- `LLAMA_MAX_SEQ`
//!
//! All at v0.4.1 (`b29c606e2`). Each declaration below names the source file
//! and line it came from.
//!
//! # Not exported yet
//!
//! libllama swaps once, at `llama.h` (`PLAN.md`, Decision 42), and until then
//! the C++ still defines `llama_batch_get_one`, `llama_batch_init` and
//! `llama_batch_free`. Here they are plain Zig functions; the shim that
//! exports them comes with the swap.
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

/// Ports `LLAMA_MAX_SEQ` (llama-cparams.h:8 @b29c606e2).
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

/// Ports `llama_ubatch`'s `data_t` (llama-batch.h:54 @b29c606e2): the arrays a
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
        self.seq_id_data.deinit(allocator);
        allocator.destroy(self);
    }
};

/// Ports `llama_ubatch` (llama-batch.h:15 @b29c606e2).
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

    /// The arrays above, when this ubatch owns them.
    data: ?*UbatchData = null,

    /// Ports `llama_ubatch`'s `equal_seqs` (llama-batch.h:16 @b29c606e2).
    ///
    /// Return: whether every sequence set has `n_seq_tokens` tokens.
    pub fn equalSeqs(self: *const Ubatch) bool {
        return self.b_equal_seqs != 0;
    }

    /// Ports `llama_ubatch`'s `is_pos_2d` (llama-batch.h:25 @b29c606e2).
    ///
    /// Return: whether positions are M-RoPE's, three or more per token.
    pub fn isPos2d(self: *const Ubatch) bool {
        return self.n_pos >= 3;
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

/// Ports `llama_batch_allocr` (llama-batch.h:72 @b29c606e2): sanitizes an
/// input batch, fills in what it omits, and splits it into ubatches.
///
/// **It must not move once `initBatch` has run**: the sanitized batch points
/// at `seq_id_0`, a field, as the C++'s does.
pub const BatchAllocr = struct {
    batch: c.llama_batch = .{},

    /// Only for debugging.
    vocab: ?*const Vocab = null,

    n_pos_per_embd: u32,

    n_embd: u32 = 0,
    n_seq_max: u32 = 0,
    n_outputs: u32 = 0,

    /// The default sequence id.
    seq_id_0: [1]SeqId = .{0},

    pos: std.ArrayList(Pos) = .empty,
    n_seq_id: std.ArrayList(i32) = .empty,
    seq_id: std.ArrayList([*c]SeqId) = .empty,
    seq_id_unq: std.ArrayList(SeqId) = .empty,
    seq_idx: std.ArrayList(i32) = .empty,
    output: std.ArrayList(i8) = .empty,

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

    /// Ports the constructor of `llama_batch_allocr` (llama-batch.cpp:12
    /// @b29c606e2), which reads `LLAMA_BATCH_DEBUG` once, here.
    ///
    /// Parameters:
    /// - `n_pos_per_embd`: positions per token, 4 for M-RoPE.
    ///
    /// Return: an allocator with no batch; free it with `deinit`.
    pub fn init(n_pos_per_embd: u32) BatchAllocr {
        var self: BatchAllocr = .{ .n_pos_per_embd = n_pos_per_embd };
        self.debug = if (std.c.getenv("LLAMA_BATCH_DEBUG")) |v| atoi(v) else 0;
        self.seq_idx.appendNTimes(allocator, -1, max_seq) catch oom();
        return self;
    }

    /// Return: nothing.
    pub fn deinit(self: *BatchAllocr) void {
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

    /// Ports `llama_batch_allocr`'s `init` (llama-batch.cpp:25 @b29c606e2):
    /// validates the batch, generates what it omits -- sequence ids,
    /// positions, output flags -- and checks it against the memory.
    ///
    /// Parameters:
    /// - `batch_inp`: the batch; its arrays are borrowed until the next
    ///   `initBatch`, `ubatchReserve` or `deinit`.
    /// - `vocab`: the vocabulary, for the token-id check and the debug log;
    ///   borrowed for as long as the batch.
    /// - `memory`: the memory module, for sequence continuity and the
    ///   starting positions; null for none.
    /// - `n_embd`: the embedding width, for an embeddings batch.
    /// - `n_seq_max`: the number of sequences the context has.
    /// - `output_all`: every token is an output.
    ///
    /// Return: whether the batch is valid; what is wrong is logged.
    pub fn initBatch(
        self: *BatchAllocr,
        batch_inp: c.llama_batch,
        vocab: *const Vocab,
        memory: ?MemoryPos,
        n_embd: u32,
        n_seq_max: u32,
        output_all: bool,
    ) bool {
        const f = "init";
        self.clear();

        self.batch = batch_inp;
        self.vocab = vocab;

        impl.assert(self.batch.n_tokens > 0, "llama-batch.cpp", 38, "batch.n_tokens > 0");
        const n_tokens: usize = @intCast(self.batch.n_tokens);

        //
        // validate input batch
        //

        if (n_seq_max > max_seq) {
            impl.logError("%s: n_seq_max = %d > %d\n", .{ f, cInt(n_seq_max), cInt(max_seq) });
            return false;
        }

        if (self.batch.token != null) {
            for (0..n_tokens) |i| {
                const t = self.batch.token[i];
                if (t < 0 or @as(u32, @bitCast(t)) >= vocab.nTokens()) {
                    impl.logError("%s: invalid token[%d] = %d\n", .{ f, @as(i32, @intCast(i)), t });
                    return false;
                }
            }
        }

        if (self.batch.seq_id != null) {
            for (0..n_tokens) |i| {
                var s: i32 = 0;
                while (s < self.batch.n_seq_id[i]) : (s += 1) {
                    const id = self.batch.seq_id[i][@intCast(s)];
                    if (id < 0 or id >= @as(SeqId, @bitCast(n_seq_max))) {
                        impl.logError("%s: invalid seq_id[%d][%d] = %d >= %d\n", .{ f, @as(i32, @intCast(i)), s, id, cInt(n_seq_max) });
                        return false;
                    }
                }
            }
        }

        //
        // auto-generate missing fields
        //

        if (self.batch.n_seq_id == null) {
            self.n_seq_id.appendNTimes(allocator, self.seq_id_0.len, n_tokens) catch oom();
            self.batch.n_seq_id = self.n_seq_id.items.ptr;
        }

        if (self.batch.seq_id == null) {
            self.seq_id.appendNTimes(allocator, &self.seq_id_0, n_tokens) catch oom();
            self.seq_id.append(allocator, null) catch oom();
            self.batch.seq_id = self.seq_id.items.ptr;
        }

        if (self.batch.pos == null) {
            self.pos.resize(allocator, n_tokens) catch oom();

            // the starting position of each sequence, from the memory
            var p0: [max_seq]Pos = undefined;
            for (0..n_seq_max) |s| {
                p0[s] = if (memory) |m| m.posMax(@intCast(s)) + 1 else 0;
            }

            for (0..n_tokens) |i| {
                const seq_id = self.batch.seq_id[i][0];

                self.pos.items[i] = p0[@intCast(seq_id)];

                // update the starting position of every sequence this token is in
                var s: usize = 0;
                while (s < self.batch.n_seq_id[i]) : (s += 1) {
                    p0[@intCast(self.batch.seq_id[i][s])] = self.pos.items[i] + 1;
                }
            }

            self.batch.pos = self.pos.items.ptr;
        }

        if (self.batch.logits == null) {
            if (output_all) {
                self.output.appendNTimes(allocator, 1, n_tokens) catch oom();
            } else {
                // only the last token
                self.output.appendNTimes(allocator, 0, n_tokens) catch oom();
                self.output.items[n_tokens - 1] = 1;
            }
            self.batch.logits = self.output.items.ptr;
        } else if (output_all) {
            var warn = false;
            for (0..n_tokens) |i| {
                if (self.batch.logits[i] == 0) warn = true;
            }

            if (warn) {
                impl.logWarn("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n", .{f});

                // `output` is empty here: `clear` emptied it and the arm
                // above did not run, so the C++'s resize fills every slot
                self.output.appendNTimes(allocator, 1, n_tokens) catch oom();
                self.batch.logits = self.output.items.ptr;
            }
        }

        //
        // compute stats
        //

        self.n_embd = n_embd;
        self.n_seq_max = n_seq_max;

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
            // M-RoPE: positions may jump forward, and need not be continuous
            for (0..n_seq_max) |su| {
                if (self.seq_pos[su].items.items.len == 0) continue;
                const s: SeqId = @intCast(su);

                const p0: Pos = if (memory) |m| m.posMax(s) else -1;

                if (self.batch.token != null) {
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

                const p0: Pos = if (memory) |m| m.posMax(s) else -1;

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

        if (memory) |m| {
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

    /// Ports `llama_batch_allocr`'s `ubatch_reserve` (llama-batch.cpp:393
    /// @b29c606e2): a well-defined ubatch of `n_seqs` sequences of
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
            .data = udata,
        };
    }

    /// Ports `llama_batch_allocr`'s `get_batch` (llama-batch.cpp:439 @b29c606e2).
    ///
    /// Return: the sanitized batch, borrowed from the allocator.
    pub fn getBatch(self: *const BatchAllocr) *const c.llama_batch {
        return &self.batch;
    }

    /// Ports `llama_batch_allocr`'s `get_n_tokens` (llama-batch.cpp:443 @b29c606e2).
    ///
    /// Return: the batch's token count.
    pub fn getNTokens(self: *const BatchAllocr) u32 {
        return @bitCast(self.batch.n_tokens);
    }

    /// Ports `llama_batch_allocr`'s `get_n_outputs` (llama-batch.cpp:447 @b29c606e2).
    ///
    /// Return: how many tokens are outputs.
    pub fn getNOutputs(self: *const BatchAllocr) u32 {
        return self.n_outputs;
    }

    /// Ports `llama_batch_allocr`'s `get_n_used` (llama-batch.cpp:451 @b29c606e2).
    ///
    /// Return: how many tokens the splits since `splitReset` took.
    pub fn getNUsed(self: *const BatchAllocr) u32 {
        return self.n_used;
    }

    /// Ports `llama_batch_allocr`'s `get_out_ids` (llama-batch.cpp:455 @b29c606e2).
    ///
    /// Return: the output indices in the order the splits met them; the
    /// list is the allocator's, and the caller may change it, as the C++'s
    /// non-const reference allows.
    pub fn getOutIds(self: *BatchAllocr) *std.ArrayList(i32) {
        return &self.out_ids;
    }

    /// Ports `llama_batch_allocr`'s `seq_pos_min` (llama-batch.cpp:459 @b29c606e2).
    ///
    /// Parameters:
    /// - `seq_id`: the sequence, in `[0, LLAMA_MAX_SEQ)`.
    ///
    /// Return: its smallest position in the batch, or -1 when it has none.
    pub fn seqPosMin(self: *const BatchAllocr, seq_id: SeqId) Pos {
        const p = self.seq_pos[@intCast(seq_id)].items.items;
        return if (p.len == 0) -1 else p[0];
    }

    /// Ports `llama_batch_allocr`'s `seq_pos_max` (llama-batch.cpp:463 @b29c606e2).
    ///
    /// Parameters:
    /// - `seq_id`: the sequence, in `[0, LLAMA_MAX_SEQ)`.
    ///
    /// Return: its largest position in the batch, or -1 when it has none.
    pub fn seqPosMax(self: *const BatchAllocr, seq_id: SeqId) Pos {
        const p = self.seq_pos[@intCast(seq_id)].items.items;
        return if (p.len == 0) -1 else p[p.len - 1];
    }

    /// Ports `llama_batch_allocr`'s `split_reset` (llama-batch.cpp:467
    /// @b29c606e2): call once before splitting.
    ///
    /// Return: nothing.
    pub fn splitReset(self: *BatchAllocr) void {
        self.out_ids.clearRetainingCapacity();

        self.n_used = 0;

        self.used.clearRetainingCapacity();
        self.used.appendNTimes(allocator, false, self.getNTokens()) catch oom();
    }

    /// Ports `llama_batch_allocr`'s `split_simple` (llama-batch.cpp:476
    /// @b29c606e2): the next `n_ubatch` unused tokens, in batch order, as
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

    /// Ports `llama_batch_allocr`'s `split_equal` (llama-batch.cpp:510
    /// @b29c606e2): ubatches of non-overlapping sequence sets with the same
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
            impl.assert(n_ubatch > n_keep_tail, "llama-batch.cpp", 609, "n_ubatch > n_keep_tail");

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

    /// Ports `llama_batch_allocr`'s `split_seq` (llama-batch.cpp:681
    /// @b29c606e2): one sequence set per ubatch, taking later tokens whose
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

    /// Ports `llama_batch_allocr`'s `clear` (llama-batch.cpp:723 @b29c606e2).
    /// `has_cpl`, `used`, `out_ids` and `n_used` survive it, as in the C++.
    fn clear(self: *BatchAllocr) void {
        self.n_outputs = 0;

        self.batch = .{};

        self.pos.clearRetainingCapacity();
        self.n_seq_id.clearRetainingCapacity();
        self.seq_id.clearRetainingCapacity();
        self.seq_id_unq.clearRetainingCapacity();
        self.output.clearRetainingCapacity();

        for (&self.seq_pos) |*p| p.items.clearRetainingCapacity();

        self.seq_cpl = @splat(0);

        self.seq_set.clearRetainingCapacity();

        self.clearSeqSetMap();

        @memset(self.seq_idx.items, -1);
    }

    /// Ports `llama_batch_allocr`'s `ubatch_add` (llama-batch.cpp:749
    /// @b29c606e2): copies the tokens at `idxs` into a new ubatch and
    /// records its outputs in `out_ids`.
    fn ubatchAdd(self: *BatchAllocr, idxs: []const i32, n_seqs: u32, equal_seqs: bool) Ubatch {
        const n_tokens: u32 = @intCast(idxs.len);

        std.debug.assert(n_tokens % n_seqs == 0);

        const udata = UbatchData.create();

        const n_embd_all: usize = if (self.batch.embd != null) @as(usize, n_tokens) * self.n_embd else 0;
        const n_pos_all: usize = @as(usize, n_tokens) * self.n_pos_per_embd;

        udata.token.appendNTimes(allocator, 0, n_tokens) catch oom();
        udata.embd.appendNTimes(allocator, 0, n_embd_all) catch oom();
        udata.pos.appendNTimes(allocator, 0, n_pos_all) catch oom();
        udata.n_seq_id.appendNTimes(allocator, 0, n_tokens) catch oom();
        udata.seq_id.appendNTimes(allocator, null, n_tokens) catch oom();
        udata.seq_idx.appendNTimes(allocator, -1, max_seq) catch oom();
        udata.output.appendNTimes(allocator, 0, n_tokens) catch oom();

        udata.seq_id_data.ensureTotalCapacity(allocator, n_tokens) catch oom();

        var seq_set_unq: SeqSet = 0;

        const n_embd: usize = self.n_embd;
        const batch_n_tokens: usize = @intCast(self.batch.n_tokens);

        for (idxs, 0..) |idx_i, i| {
            const idx: usize = @intCast(idx_i);

            if (self.batch.token != null) {
                udata.token.items[i] = self.batch.token[idx];
            }

            if (self.batch.embd != null) {
                @memcpy(udata.embd.items[i * n_embd ..][0..n_embd], self.batch.embd[idx * n_embd ..][0..n_embd]);
            }

            for (0..self.n_pos_per_embd) |j| {
                // M-RoPE: a text batch broadcasts its one position to every
                // section; an embeddings batch carries all of them
                const src_off: usize = if (self.batch.token != null) 0 else j * batch_n_tokens;
                udata.pos.items[j * n_tokens + i] = self.batch.pos[src_off + idx];
            }

            udata.n_seq_id.items[i] = self.batch.n_seq_id[idx];
            udata.output.items[i] = self.batch.logits[idx];

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
            .token = if (self.batch.token != null) dataPtr(Token, &udata.token) else null,
            .embd = if (self.batch.embd != null) dataPtr(f32, &udata.embd) else null,
            .pos = dataPtr(Pos, &udata.pos),
            .n_seq_id = dataPtr(i32, &udata.n_seq_id),
            .seq_id = dataPtr([*c]SeqId, &udata.seq_id),
            .seq_id_unq = dataPtr(SeqId, &udata.seq_id_unq),
            .seq_idx = dataPtr(i32, &udata.seq_idx),
            .output = dataPtr(i8, &udata.output),
            .data = udata,
        };

        if (self.debug > 0) {
            impl.logDebug("%s: added ubatch to split:\n", .{"ubatch_add"});

            self.ubatchPrint(&res, self.debug);
        }

        return res;
    }

    /// Ports `llama_batch_allocr`'s `ubatch_print` (llama-batch.cpp:846
    /// @b29c606e2), for `LLAMA_BATCH_DEBUG`.
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
            if (ubatch.token != null) {
                var exc: impl.Exception = .{};
                const text = self.vocab.?.cachedPiece(ubatch.token[i], &exc) catch exc.terminate("llama-batch.cpp", 915);
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
// The C interface

/// Ports `llama_batch_get_one` (llama-batch.cpp:931 @b29c606e2): a batch of
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

/// Ports `llama_batch_init` (llama-batch.cpp:945 @b29c606e2): a batch whose
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

/// Ports `llama_batch_free` (llama-batch.cpp:975 @b29c606e2).
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
    var a = BatchAllocr.init(1);
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
