//! The RWKV tokenizer: greedy longest match over a trie of the unescaped
//! token texts.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v0.3.0 (`c1d0e7a00`).

const std = @import("std");
const impl = @import("../impl.zig");
const vm = @import("module.zig");

const allocator = impl.allocator;
const Vocab = vm.Vocab;
const Token = vm.Token;
const Trie = vm.trie.Trie;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

/// Ports `llama_unescape_rwkv_token` (llama-vocab.cpp:1242 @c1d0e7a00):
/// `\t`, `\n`, `\r`, `\xHH` and `\` before anything else. A hex digit is
/// read as lowercase-or-digit with no check, as there.
pub fn unescapeToken(escaped: []const u8, output: *std.ArrayList(u8)) void {
    output.ensureUnusedCapacity(allocator, escaped.len) catch oom();

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
                output.append(allocator, hex_acc) catch oom();
                hex_acc = 0;
            }
            continue;
        }

        // If we got an escape character, interpret it
        if (escaping) {
            switch (ch) {
                't' => output.append(allocator, '\t') catch oom(),
                'n' => output.append(allocator, '\n') catch oom(),
                'r' => output.append(allocator, '\r') catch oom(),
                'x' => hex_remaining = 2,
                else => output.append(allocator, ch) catch oom(),
            }
            escaping = false;
            continue;
        }

        if (ch == '\\') {
            escaping = true;
            continue;
        }

        output.append(allocator, ch) catch oom();
    }
}

/// Ports `llm_tokenizer_rwkv` (llama-vocab.cpp:1296 @c1d0e7a00).
pub const Tokenizer = struct {
    token_matcher: Trie,

    /// Return: the tokenizer, every token's unescaped text in the trie.
    pub fn init(vocab: *const Vocab) Tokenizer {
        var t: Tokenizer = .{ .token_matcher = Trie.init() };
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);
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

/// Ports `llm_tokenizer_rwkv_session`'s `tokenize` (llama-vocab.cpp:1315
/// @c1d0e7a00). A byte with no token becomes UNK; otherwise the longest
/// valued prefix is taken. A path with no valued node yields token 0 and
/// restarts at 0, as the C++ does.
pub fn tokenize(vocab: *const Vocab, tokenizer: *const Tokenizer, text: []const u8, output: *std.ArrayList(Token)) void {
    var position: u32 = 0;
    while (position < text.len) {
        var node = tokenizer.token_matcher.traverse(Trie.root, text[position]) orelse {
            // no matching token found, add unknown token
            output.append(allocator, vocab.special_unk_id) catch oom();
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
        output.append(allocator, @bitCast(token_id)) catch oom();
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
    defer out.deinit(allocator);
    unescapeToken("a\\n\\x41\\\\b", &out);
    try std.testing.expectEqualStrings("a\nA\\b", out.items);
}
