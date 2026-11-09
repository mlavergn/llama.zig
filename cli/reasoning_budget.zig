//! `common_reasoning_budget` for `llama-cli`: a sampler that limits how many
//! tokens a model may spend inside a reasoning block, then forces the block
//! closed -- what `--reasoning-budget` sets up.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/reasoning-budget.cpp` and
//! `llama.cpp/common/reasoning-budget.h`, with `common_utf8_is_complete` from
//! `llama.cpp/common/unicode.cpp` and the `common_aho_corasick` constructor
//! from `llama.cpp/common/trie.cpp` (v0.8.0, `8642541b2`). Each declaration
//! names what it ports and the line it began at.
//!
//! # A second copy, on purpose
//!
//! `tools/common/reasoning_budget.zig` ports the same file for the upstream
//! tools and the server, held to the C++ by `make spec-diff`. This is that
//! port with its two module dependencies cut: the tools' copy reaches
//! `tools/common/`'s logger and `common_token_to_piece` by relative import,
//! which would pull the whole `tools_common` module -- and its own
//! `@cImport` of `llama.h` -- into the CLI. Here the trace lines are dropped
//! (`llama-cli` logs at ERROR, where `LOG_TRC` never prints) and the piece is
//! read through `c.zig`. Everything else is the tools' text; a change to one
//! belongs in both. `make cnv-diff`'s budget scenarios hold this one to
//! upstream's `common_reasoning_budget` in `libllama-common-ref.a`.
//!
//! # The sampler
//!
//! It is a `llama_sampler` with its own interface, as upstream's is, so a
//! chain or `llama_sampler_clone`/`llama_sampler_copy` treats it as any
//! other: `copy_state` is null in upstream's interface, so a copy clones and
//! transplants, and this one leaves it null too.
//!
//! The token matchers run an Aho-Corasick automaton over token ids, built
//! from `cli/json/trie.zig`'s `Trie` with the ids as code points, exactly as
//! upstream builds a `common_trie` from `std::vector<uint32_t>`.

const std = @import("std");
const c = @import("c.zig").api;
const trie = @import("cli_fit").json.trie;

const gpa = std.heap.c_allocator;

/// What upstream's `new` does when it fails: `std::bad_alloc`, uncaught.
fn oom() noreturn {
    _ = c.ggml_abort("reasoning-budget.cpp", 0, "out of memory");
    unreachable;
}

/// Ports `common_reasoning_budget_state` (reasoning-budget.h:10 @8642541b2).
pub const State = enum(c_int) {
    idle, // waiting for start sequence
    counting, // counting down tokens
    forcing, // forcing budget message + end sequence
    waiting_utf8, // budget exhausted, waiting for UTF-8 completion
    done, // passthrough forever
};

/// `COM_TRC`: `LOG_TRC`, which `llama-cli`'s ERROR verbosity never prints;
/// kept so the state machine reads as the C++'s does.
fn comTrc(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    _ = func;
    _ = fmt;
    _ = args;
}

/// The piece `common_token_to_piece(vocab, token, false)` returns, in a
/// caller's buffer. Only its UTF-8 completeness is read.
///
/// Return: the piece, or null when it does not fit in `buf` -- which no
/// vocabulary's single token reaches at 256 bytes.
fn tokenPiece(vocab: ?*const c.llama_vocab, token: c.llama_token, buf: []u8) []const u8 {
    const n = c.llama_token_to_piece(vocab, token, buf.ptr, @intCast(buf.len), 0, false);
    if (n < 0) oom();
    return buf[0..@intCast(n)];
}

/// Ports `common_utf8_is_complete` (common/unicode.cpp:77 @8642541b2): whether `s`
/// does not end inside a UTF-8 sequence.
pub fn utf8IsComplete(s: []const u8) bool {
    if (s.len == 0) return true;
    var i: usize = 1;
    while (i <= @min(4, s.len)) : (i += 1) {
        const ch = s[s.len - i];
        if ((ch & 0xC0) != 0x80) {
            const expected: usize = if (ch >= 0xF0) 4 else if (ch >= 0xE0) 3 else if (ch >= 0xC0) 2 else 1;
            return i >= expected;
        }
    }
    return false;
}

/// Ports the `common_aho_corasick` constructor (trie.cpp:77 @8642541b2) over a
/// trie already built: failure links in breadth-first order, the longest
/// pattern ending at each state, the alphabet. `cli/json/trie.zig`'s
/// `AhoCorasick.init` takes strings; the matchers here insert token ids.
fn ahoFromTrie(t: trie.Trie) !trie.AhoCorasick {
    const nodes = t.nodes.items;
    const n = nodes.len;

    const fail = try gpa.alloc(usize, n);
    @memset(fail, 0);
    var order: std.ArrayList(usize) = .empty;
    try order.ensureTotalCapacityPrecise(gpa, n);

    var queue: std.ArrayList(usize) = .empty;
    defer queue.deinit(gpa);
    try queue.append(gpa, 0);
    var head: usize = 0;
    while (head < queue.items.len) {
        const u = queue.items[head];
        head += 1;
        order.appendAssumeCapacity(u);
        for (nodes[u].children.items) |e| {
            const v = e.child;
            if (u != 0) {
                var f = fail[u];
                while (f != 0 and findChild(nodes[f], e.cpt) == null) f = fail[f];
                const it = findChild(nodes[f], e.cpt);
                fail[v] = if (it != null and it.? != v) it.? else 0;
            }
            try queue.append(gpa, v);
        }
    }

    const match = try gpa.alloc(i32, n);
    @memset(match, -1);
    for (order.items) |u| {
        match[u] = if (nodes[u].pattern >= 0) nodes[u].pattern else if (u != 0) match[fail[u]] else -1;
    }

    var alphabet: std.ArrayList(u32) = .empty;
    for (nodes) |node| {
        for (node.children.items) |e| {
            var at: usize = 0;
            while (at < alphabet.items.len and alphabet.items[at] < e.cpt) at += 1;
            if (at < alphabet.items.len and alphabet.items[at] == e.cpt) continue;
            try alphabet.insert(gpa, at, e.cpt);
        }
    }

    return .{ .t = t, .fail = fail, .order = try order.toOwnedSlice(gpa), .match = match, .alphabet = try alphabet.toOwnedSlice(gpa) };
}

fn findChild(node: trie.Trie.Node, cpt: u32) ?usize {
    for (node.children.items) |e| {
        if (e.cpt == cpt) return e.child;
    }
    return null;
}

/// Ports `token_matcher` (reasoning-budget.cpp:14 @8642541b2): every
/// sequence at once, through one automaton.
const TokenMatcher = struct {
    seqs: [][]c.llama_token,
    ac: trie.AhoCorasick,
    state: usize = 0,

    /// Ports `token_matcher` (reasoning-budget.cpp:19 @8642541b2), the
    /// constructor: `collect` then `build_trie`.
    fn init(seqs_in: []const []const c.llama_token) !TokenMatcher {
        const seqs = try collect(seqs_in);
        var t = try trie.Trie.init(gpa, &.{});
        for (seqs) |seq| {
            const syms = try gpa.alloc(u32, seq.len);
            defer gpa.free(syms);
            for (seq, syms) |tok, *s| s.* = @bitCast(tok);
            _ = try t.insertSymbols(syms);
        }
        return .{ .seqs = seqs, .ac = try ahoFromTrie(t) };
    }

    fn deinit(self: *TokenMatcher) void {
        for (self.seqs) |s| gpa.free(s);
        gpa.free(self.seqs);
        self.ac.deinit();
    }

    /// Ports `collect` (reasoning-budget.cpp:21 @8642541b2): the non-empty
    /// sequences, first occurrence of each.
    fn collect(seqs: []const []const c.llama_token) ![][]c.llama_token {
        var res: std.ArrayList([]c.llama_token) = .empty;
        for (seqs) |seq| {
            if (seq.len == 0) continue;
            const seen = for (res.items) |r| {
                if (std.mem.eql(c.llama_token, r, seq)) break true;
            } else false;
            if (!seen) try res.append(gpa, try gpa.dupe(c.llama_token, seq));
        }
        return res.toOwnedSlice(gpa);
    }

    /// Ports `advance` (reasoning-budget.cpp:40 @8642541b2).
    ///
    /// Return: the index into `seqs` of the longest sequence ending at this
    /// token, or -1.
    fn advance(self: *TokenMatcher, token: c.llama_token) i32 {
        self.state = self.ac.next(self.state, @bitCast(token));
        const p = self.ac.matchPattern(self.state);
        if (p >= 0) self.state = 0;
        return p;
    }

    /// Ports `reset` (reasoning-budget.cpp:49 @8642541b2).
    fn reset(self: *TokenMatcher) void {
        self.state = 0;
    }
};

/// Ports `common_reasoning_budget_ctx` (reasoning-budget.cpp:52 @8642541b2).
const Ctx = struct {
    vocab: ?*const c.llama_vocab,

    start_matcher: TokenMatcher,
    end_matcher: TokenMatcher,
    forced_tokens: []c.llama_token,

    budget: i32, // maximum tokens in reasoning block
    remaining: i32, // tokens remaining in budget

    state: State,

    // for forcing
    force_pos: usize, // next position in forced_tokens to force

    end_match: i32, // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    fn destroy(self: *Ctx) void {
        self.start_matcher.deinit();
        self.end_matcher.deinit();
        gpa.free(self.forced_tokens);
        gpa.destroy(self);
    }
};

fn ctxOf(smpl: ?*const c.llama_sampler) *Ctx {
    return @ptrCast(@alignCast(smpl.?.ctx.?));
}

/// Ports `common_reasoning_budget_name` (reasoning-budget.cpp:70 @8642541b2).
fn name(_: ?*const c.llama_sampler) callconv(.c) [*c]const u8 {
    return "reasoning-budget";
}

/// Ports `common_reasoning_budget_accept` (reasoning-budget.cpp:74 @8642541b2).
fn accept(smpl: ?*c.llama_sampler, token: c.llama_token) callconv(.c) void {
    const ctx = ctxOf(smpl);
    const func = "common_reasoning_budget_accept";

    switch (ctx.state) {
        .idle => {
            if (ctx.start_matcher.advance(token) >= 0) {
                ctx.state = .counting;
                ctx.remaining = ctx.budget;
                comTrc(func, "activated, budget=%d tokens\n", .{ctx.budget});

                if (ctx.remaining <= 0) {
                    ctx.state = .forcing;
                    ctx.force_pos = 0;
                    comTrc(func, "%s", .{@as([*:0]const u8, "budget=0, forcing immediately\n")});
                }
            }
        },
        .counting, .waiting_utf8 => {
            const match = ctx.end_matcher.advance(token);
            if (match >= 0) {
                ctx.state = .done;
                ctx.end_match = match;
                comTrc(func, "%s", .{@as([*:0]const u8, "deactivated (natural end)\n")});
                return;
            }

            var utf8_complete = true;
            if (ctx.vocab != null) {
                var buf: [256]u8 = undefined;
                utf8_complete = utf8IsComplete(tokenPiece(ctx.vocab, token, &buf));
            }

            if (ctx.state == .waiting_utf8) {
                if (utf8_complete) {
                    ctx.state = .forcing;
                    ctx.force_pos = 0;
                    ctx.end_matcher.reset();
                    comTrc(func, "%s", .{@as([*:0]const u8, "UTF-8 complete, now forcing end sequence\n")});
                }
            } else if (ctx.state == .counting) {
                ctx.remaining -%= 1;
                if (ctx.remaining <= 0) {
                    if (utf8_complete) {
                        ctx.state = .forcing;
                        ctx.force_pos = 0;
                        ctx.end_matcher.reset();
                        comTrc(func, "%s", .{@as([*:0]const u8, "budget exhausted, forcing end sequence\n")});
                    } else {
                        ctx.state = .waiting_utf8;
                        ctx.end_matcher.reset();
                        comTrc(func, "%s", .{@as([*:0]const u8, "budget exhausted, waiting for UTF-8 completion\n")});
                    }
                }
            }
        },
        .forcing => {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const match = ctx.end_matcher.advance(token);
            ctx.force_pos += 1;
            if (ctx.force_pos >= ctx.forced_tokens.len) {
                ctx.state = .done;
                ctx.end_match = match;
                comTrc(func, "%s", .{@as([*:0]const u8, "forced sequence complete, done\n")});
            }
        },
        .done => {
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx.start_matcher.advance(token) >= 0) {
                ctx.state = .counting;
                ctx.remaining = ctx.budget;
                ctx.end_matcher.reset();
                ctx.end_match = -1;
                comTrc(func, "re-activated on new start tag, budget=%d tokens\n", .{ctx.budget});

                if (ctx.remaining <= 0) {
                    ctx.state = .forcing;
                    ctx.force_pos = 0;
                    comTrc(func, "%s", .{@as([*:0]const u8, "budget=0, forcing immediately\n")});
                }
            }
        },
    }
}

/// Ports `common_reasoning_budget_apply` (reasoning-budget.cpp:166 @8642541b2):
/// while forcing, every logit but the forced token's goes to -inf.
fn apply(smpl: ?*c.llama_sampler, cur_p: ?*c.llama_token_data_array) callconv(.c) void {
    const ctx = ctxOf(smpl);

    if (ctx.state != .forcing) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx.force_pos >= ctx.forced_tokens.len) return;

    const forced = ctx.forced_tokens[ctx.force_pos];

    // set all logits to -inf except the forced token
    const p = cur_p.?;
    for (0..p.size) |i| {
        if (p.data[i].id != forced) p.data[i].logit = -std.math.inf(f32);
    }
}

/// Ports `common_reasoning_budget_reset` (reasoning-budget.cpp:188 @8642541b2).
fn reset(smpl: ?*c.llama_sampler) callconv(.c) void {
    const ctx = ctxOf(smpl);
    ctx.state = .idle;
    ctx.remaining = ctx.budget;
    ctx.start_matcher.reset();
    ctx.end_matcher.reset();
    ctx.force_pos = 0;
    ctx.end_match = -1;
}

/// Ports `common_reasoning_budget_free` (reasoning-budget.cpp:205 @8642541b2).
fn free(smpl: ?*c.llama_sampler) callconv(.c) void {
    ctxOf(smpl).destroy();
}

/// Ports `common_reasoning_budget_i` (reasoning-budget.cpp:209 @8642541b2).
var iface: c.struct_llama_sampler_i = blk: {
    var i = std.mem.zeroes(c.struct_llama_sampler_i);
    i.name = name;
    i.accept = accept;
    i.apply = apply;
    i.reset = reset;
    i.clone = clone;
    i.free = free;
    break :blk i;
};

/// Ports `common_reasoning_budget_clone` (reasoning-budget.cpp:224 @8642541b2):
/// a copy of the context. The matchers are rebuilt from their sequences,
/// which `collect` already left unique and non-empty, so the automaton is
/// the same one; the walk state is copied.
fn clone(smpl: ?*const c.llama_sampler) callconv(.c) ?*c.llama_sampler {
    const ctx = ctxOf(smpl);
    const res = initState(ctx.vocab, @ptrCast(ctx.start_matcher.seqs), @ptrCast(ctx.end_matcher.seqs), ctx.forced_tokens, ctx.budget, ctx.state) catch oom();
    const nctx = ctxOf(res);
    nctx.remaining = ctx.remaining;
    nctx.state = ctx.state;
    nctx.force_pos = ctx.force_pos;
    nctx.end_match = ctx.end_match;
    nctx.start_matcher.state = ctx.start_matcher.state;
    nctx.end_matcher.state = ctx.end_matcher.state;
    return res;
}

/// Ports `common_reasoning_budget_init_state` (reasoning-budget.cpp:233 @8642541b2).
fn initState(
    vocab: ?*const c.llama_vocab,
    start_seqs: []const []const c.llama_token,
    end_seqs: []const []const c.llama_token,
    forced_tokens: []const c.llama_token,
    budget: i32,
    initial_state_in: State,
) !*c.llama_sampler {
    var initial_state = initial_state_in;
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == .counting and budget <= 0) initial_state = .forcing;

    const ctx = try gpa.create(Ctx);
    ctx.* = .{
        .vocab = vocab,
        .start_matcher = try TokenMatcher.init(start_seqs),
        .end_matcher = try TokenMatcher.init(end_seqs),
        .forced_tokens = try gpa.dupe(c.llama_token, forced_tokens),
        .budget = budget,
        .remaining = budget,
        .state = initial_state,
        .force_pos = 0,
        .end_match = -1,
    };
    return c.llama_sampler_init(&iface, ctx).?;
}

/// Ports `common_reasoning_budget_init` (reasoning-budget.cpp:261 @8642541b2).
///
/// Parameters:
/// - `vocab`: for UTF-8 boundary detection; can be null.
/// - `start_seqs`: token sequences, any of which activates counting.
/// - `end_seqs`: token sequences, any of which naturally deactivates.
/// - `forced_tokens`: the sequence forced when the budget expires.
/// - `budget`: the most tokens allowed in the reasoning block.
/// - `initial_state`: where the state machine starts.
///
/// Return: the sampler, freed with `llama_sampler_free`.
pub fn init(
    vocab: ?*const c.llama_vocab,
    start_seqs: []const []const c.llama_token,
    end_seqs: []const []const c.llama_token,
    forced_tokens: []const c.llama_token,
    budget: i32,
    initial_state: State,
) !*c.llama_sampler {
    return initState(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state);
}

/// Ports `common_reasoning_budget_get_state` (reasoning-budget.cpp:271 @8642541b2).
pub fn getState(smpl: ?*const c.llama_sampler) State {
    if (smpl == null) return .idle;
    return ctxOf(smpl).state;
}

/// Ports `common_reasoning_budget_get_end_match` (reasoning-budget.cpp:278 @8642541b2).
///
/// Return: the end sequence that moved the sampler to DONE, or null if none
/// was recorded; borrowed from the sampler.
pub fn getEndMatch(smpl: ?*const c.llama_sampler) ?[]const c.llama_token {
    if (smpl == null) return null;
    const ctx = ctxOf(smpl);
    if (ctx.end_match < 0) return null;
    return ctx.end_matcher.seqs[@intCast(ctx.end_match)];
}

/// Ports `common_reasoning_budget_force` (reasoning-budget.cpp:291 @8642541b2):
/// moves a counting sampler into FORCING.
///
/// Return: whether the transition happened.
pub fn force(smpl: ?*c.llama_sampler) bool {
    if (smpl == null) return false;

    const ctx = ctxOf(smpl);

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx.state != .counting) return false;

    ctx.state = .forcing;
    ctx.force_pos = 0;
    ctx.end_matcher.reset();
    comTrc("common_reasoning_budget_force", "%s", .{@as([*:0]const u8, "forced into forcing state (manual transition)\n")});

    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "utf8IsComplete looks back at most four bytes" {
    try std.testing.expect(utf8IsComplete(""));
    try std.testing.expect(utf8IsComplete("a"));
    try std.testing.expect(!utf8IsComplete("\xc3"));
    try std.testing.expect(utf8IsComplete("\xc3\xa9"));
    try std.testing.expect(!utf8IsComplete("\xe6\x97"));
    try std.testing.expect(!utf8IsComplete("\x80\x80\x80\x80"));
}
