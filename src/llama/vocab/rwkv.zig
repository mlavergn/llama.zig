//! The RWKV tokenizer: greedy longest match over a trie of the unescaped
//! token texts.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v1.0.0 (`b1b770d90`).

const std = @import("std");
const llama = @import("../module.zig");

fn oom() noreturn {
    llama.impl.abort("llama-vocab.cpp", 0, "out of memory");
}

/// Ports `llama_unescape_rwkv_token` (llama-vocab.cpp:1356 @b1b770d90):
/// `\t`, `\n`, `\r`, `\xHH` and `\` before anything else. A hex digit is
/// read as lowercase-or-digit with no check, as there.
pub fn unescapeToken(escaped: []const u8, output: *std.ArrayList(u8)) void {
    output.ensureUnusedCapacity(std.heap.c_allocator, escaped.len) catch oom();

    var escaping = false;
    var hex_remaining: u8 = 0;
    var hex_acc: u8 = 0;

    for (escaped) |ch| {
        // If we're parsing a hex code, interpret the next character
        if (hex_remaining != 0) {
            // `c >= 'a'` on a plain `char`, whose signedness is the target's:
            // signed on Darwin and x86, where a byte above 0x7f is negative,
            // unsigned on aarch64 Linux. `c_char` is the same choice.
            const plain: c_char = @bitCast(ch);
            const value: u8 = if (plain >= 'a') ch -% 'a' +% 10 else ch -% '0';
            hex_acc = (hex_acc << 4) +% value;
            hex_remaining -= 1;
            if (hex_remaining == 0) {
                output.append(std.heap.c_allocator, hex_acc) catch oom();
                hex_acc = 0;
            }
            continue;
        }

        // If we got an escape character, interpret it
        if (escaping) {
            switch (ch) {
                't' => output.append(std.heap.c_allocator, '\t') catch oom(),
                'n' => output.append(std.heap.c_allocator, '\n') catch oom(),
                'r' => output.append(std.heap.c_allocator, '\r') catch oom(),
                'x' => hex_remaining = 2,
                else => output.append(std.heap.c_allocator, ch) catch oom(),
            }
            escaping = false;
            continue;
        }

        if (ch == '\\') {
            escaping = true;
            continue;
        }

        output.append(std.heap.c_allocator, ch) catch oom();
    }
}

/// Ports `llm_tokenizer_rwkv` (llama-vocab.cpp:1410 @b1b770d90).
pub const Tokenizer = struct {
    token_matcher: llama.vocab.trie.Trie,

    /// Return: the tokenizer, every token's unescaped text in the trie.
    pub fn init(vocab: *const llama.vocab.Vocab) Tokenizer {
        var t: Tokenizer = .{ .token_matcher = llama.vocab.trie.Trie.init() };
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(std.heap.c_allocator);
        const n = vocab.nTokens();
        var id: u32 = 0;
        while (id < n) : (id += 1) {
            buf.clearRetainingCapacity();
            unescapeToken(vocab.text(@intCast(id)), &buf);
            t.token_matcher.insert(buf.items, @intCast(id));
        }
        return t;
    }

    /// Return: nothing.
    pub fn deinit(self: *Tokenizer) void {
        self.token_matcher.deinit();
    }
};

/// Ports `llm_tokenizer_rwkv_session`'s `tokenize` (llama-vocab.cpp:1429
/// @b1b770d90). A byte with no token becomes UNK; otherwise the longest
/// valued prefix is taken. A path with no valued node yields token 0 and
/// restarts at 0, as the C++ does.
pub fn tokenize(vocab: *const llama.vocab.Vocab, tokenizer: *const Tokenizer, text: []const u8, output: *std.ArrayList(llama.vocab.Token)) void {
    var position: u32 = 0;
    while (position < text.len) {
        var node = tokenizer.token_matcher.traverse(llama.vocab.trie.Trie.root, text[position]) orelse {
            // no matching token found, add unknown token
            output.append(std.heap.c_allocator, vocab.special_unk_id) catch oom();
            position += 1;
            continue;
        };

        // traverse the trie to find the longest matching token
        var token_id: u32 = 0;
        var token_length: u32 = 0;
        while (true) {
            const v = tokenizer.token_matcher.get(node);
            if (v.has_value) {
                token_id = @bitCast(v.value);
                token_length = position + 1;
            }
            if (position + 1 >= text.len) break;
            position += 1;
            node = tokenizer.token_matcher.traverse(node, text[position]) orelse break;
        }

        // add the longest matching token
        output.append(std.heap.c_allocator, @bitCast(token_id)) catch oom();
        position = token_length;
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "unescape reads the RWKV escapes" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.heap.c_allocator);
    unescapeToken("a\\n\\x41\\\\b", &out);
    try std.testing.expectEqualStrings("a\nA\\b", out.items);
}
