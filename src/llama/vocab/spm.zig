//! The SentencePiece (SPM) tokenizer: merge the highest-scoring adjacent
//! pair until none is a token, then fall back to bytes.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v0.3.0 (`c1d0e7a00`).
//!
//! The work queue is a `std::priority_queue` with a float comparator, so
//! among equal scores -- and around a NaN score, where the comparator is no
//! longer an order -- which pair merges first is decided by libc++'s
//! `push_heap` and `pop_heap`. `libcxx.PriorityQueue` is those two.

const std = @import("std");
const impl = @import("../impl.zig");
const libcxx = @import("../libcxx.zig");
const unicode = @import("../unicode.zig");
const vm = @import("module.zig");

const allocator = impl.allocator;
const Vocab = vm.Vocab;
const Token = vm.Token;
const token_null = vm.vocab.token_null;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

/// Ports `llm_symbol` (llama-vocab.cpp:80 @c1d0e7a00), with the text as an
/// offset into the string being tokenized.
pub const Symbol = struct {
    prev: i32,
    next: i32,
    off: usize,
    n: usize,
};

/// Ports `llm_bigram_spm` (llama-vocab.cpp:96 @c1d0e7a00).
const Bigram = struct {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
};

/// `llm_bigram_spm`'s `comparator`.
fn bigramLess(_: void, l: Bigram, r: Bigram) bool {
    return (l.score < r.score) or (l.score == r.score and l.left > r.left);
}

/// Ports `llm_tokenizer_spm_session` (llama-vocab.cpp:114 @c1d0e7a00).
const Session = struct {
    vocab: *const Vocab,
    text: []const u8,
    symbols: std.ArrayList(Symbol) = .empty,
    work_queue: libcxx.PriorityQueue(Bigram, void, bigramLess) = .{ .ctx = {} },
    /// `rev_merge`: each pair's text to the symbols it last merged.
    rev_merge: std.StringHashMapUnmanaged([2]i32) = .empty,
    keys: std.ArrayList([]u8) = .empty,

    fn deinit(self: *Session) void {
        self.symbols.deinit(allocator);
        self.work_queue.deinit(allocator);
        self.rev_merge.deinit(allocator);
        for (self.keys.items) |k| allocator.free(k);
        self.keys.deinit(allocator);
    }

    fn symText(self: *const Session, s: Symbol) []const u8 {
        return self.text[s.off..][0..s.n];
    }

    /// Ports `llm_tokenizer_spm_session`'s `try_add_bigram`.
    fn tryAddBigram(self: *Session, left: i32, right: i32, exc: *impl.Exception) impl.Thrown!void {
        if (left == -1 or right == -1) return;
        const l = self.symbols.items[@intCast(left)];
        const r = self.symbols.items[@intCast(right)];
        const t = self.text[l.off..][0 .. l.n + r.n];
        const token = self.vocab.textToToken(t);
        if (token == token_null) return;
        if (@as(u32, @bitCast(token)) >= self.vocab.nTokens()) return;

        const tok_data = try self.vocab.getTokenData(token, exc);
        self.work_queue.push(allocator, .{ .left = left, .right = right, .score = tok_data.score, .size = t.len }) catch oom();

        const g = self.rev_merge.getOrPut(allocator, t) catch oom();
        if (!g.found_existing) {
            const key = allocator.dupe(u8, t) catch oom();
            self.keys.append(allocator, key) catch oom();
            g.key_ptr.* = key;
        }
        g.value_ptr.* = .{ left, right };
    }

    /// Ports `llm_tokenizer_spm_session`'s `resegment`.
    fn resegment(self: *Session, symbol: Symbol, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
        const t = self.symText(symbol);
        const token = self.vocab.textToToken(t);
        if (token != token_null) {
            output.append(allocator, token) catch oom();
            return;
        }
        const p = self.rev_merge.get(t) orelse {
            // output any symbols that did not form tokens as bytes.
            output.ensureUnusedCapacity(allocator, symbol.n) catch oom();
            for (t) |ch| output.append(allocator, try self.vocab.byteToToken(ch, exc)) catch oom();
            return;
        };
        try self.resegment(self.symbols.items[@intCast(p[0])], output, exc);
        try self.resegment(self.symbols.items[@intCast(p[1])], output, exc);
    }
};

/// Ports `llm_tokenizer_spm_session`'s `tokenize` (llama-vocab.cpp:117
/// @c1d0e7a00).
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `text`: the escaped text; not empty.
/// - `output`: tokens are appended here.
/// - `exc`: receives what is thrown.
///
/// Return: nothing, or `error.Thrown` -- a byte with no token.
pub fn tokenize(vocab: *const Vocab, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
    var s: Session = .{ .vocab = vocab, .text = text };
    defer s.deinit();

    // split string into utf8 chars
    var index: i32 = 0;
    var offs: usize = 0;
    while (offs < text.len) {
        const len = unicode.lenUtf8(text[offs]);
        const n = @min(len, text.len - offs);
        const sym: Symbol = .{ .off = offs, .n = n, .prev = index - 1, .next = if (offs + n == text.len) -1 else index + 1 };
        offs += n;
        index += 1;
        s.symbols.append(allocator, sym) catch oom();
    }

    // seed the work queue with all possible 2-character tokens.
    var i: i32 = 1;
    while (i < @as(i32, @intCast(s.symbols.items.len))) : (i += 1) try s.tryAddBigram(i - 1, i, exc);

    // keep substituting the highest frequency pairs for as long as we can.
    while (!s.work_queue.empty()) {
        const bigram = s.work_queue.pop();
        const left_sym = &s.symbols.items[@intCast(bigram.left)];
        const right_sym = &s.symbols.items[@intCast(bigram.right)];

        // if one of the symbols already got merged, skip it.
        if (left_sym.n == 0 or right_sym.n == 0 or left_sym.n + right_sym.n != bigram.size) continue;

        // merge the right sym into the left one
        left_sym.n += right_sym.n;
        right_sym.n = 0;

        // remove the right sym from the chain
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) s.symbols.items[@intCast(right_sym.next)].prev = bigram.left;

        // find more substitutions
        const prev = left_sym.prev;
        const next = left_sym.next;
        try s.tryAddBigram(prev, bigram.left, exc);
        try s.tryAddBigram(bigram.left, next, exc);
    }

    var j: i32 = 0;
    while (j != -1) : (j = s.symbols.items[@intCast(j)].next) {
        try s.resegment(s.symbols.items[@intCast(j)], output, exc);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
