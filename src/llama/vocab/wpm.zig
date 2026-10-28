//! The WordPiece (WPM) tokenizer BERT models use: normalize, split on
//! whitespace and punctuation, then take the longest known prefix of each
//! word, repeatedly.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v0.4.0 (`5266f24da`).

const std = @import("std");
const impl = @import("../impl.zig");
const unicode = @import("../unicode.zig");
const vm = @import("module.zig");

const allocator = impl.allocator;
const Vocab = vm.Vocab;
const Token = vm.Token;
const token_null = vm.vocab.token_null;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

/// Ports `llm_tokenizer_wpm_session`'s `tokenize` (llama-vocab.cpp:771
/// @5266f24da).
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `text`: one fragment.
/// - `output`: tokens are appended here.
/// - `exc`: receives what is thrown.
///
/// Return: nothing, or `error.Thrown` from `unicode_cpt_to_utf8`.
pub fn tokenize(vocab: *const Vocab, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
    var words = try preprocess(text, vocab.normalizer_opts, exc);
    defer {
        for (words.items) |*w| w.deinit(allocator);
        words.deinit(allocator);
    }

    var word1: std.ArrayList(u8) = .empty;
    defer word1.deinit(allocator);

    for (words.items) |word| {
        // skip empty words
        if (word.items.len == 0) continue;

        // prepend phantom space
        word1.clearRetainingCapacity();
        word1.appendSlice(allocator, "\xe2\x96\x81") catch oom();
        word1.appendSlice(allocator, word.items) catch oom();
        const n: i32 = @intCast(word1.items.len);

        const current_tokens = output.items.len;

        // move through character position in word
        var i: i32 = 0;
        while (i < n) : (i += 1) {
            // loop through possible match length
            var match = false;
            var j: i32 = @min(n, i + vocab.max_token_len + 1);
            while (j > i) : (j -= 1) {
                const id = vocab.textToToken(word1.items[@intCast(i)..@intCast(j)]);
                if (id != token_null) {
                    output.append(allocator, id) catch oom();
                    match = true;
                    i = j - 1;
                    break;
                }
            }

            if (!match) { // discard all
                output.shrinkRetainingCapacity(current_tokens);
                break; // and discard next tokens
            }
        }

        // we didn't find any matches for this word
        if (current_tokens == output.items.len) output.append(allocator, vocab.special_unk_id) catch oom();
    }
}

/// Ports `llm_tokenizer_wpm_session`'s `preprocess` (llama-vocab.cpp:818
/// @5266f24da): NFD and accent stripping if asked, lowercasing if asked,
/// whitespace dropped, controls dropped, and punctuation, ASCII symbols and
/// Chinese characters each a word of their own.
///
/// Return: the words, the last never empty, owned by the caller.
pub fn preprocess(text: []const u8, opts: vm.vocab.NormalizerOptions, exc: *impl.Exception) impl.Thrown!std.ArrayList(std.ArrayList(u8)) {
    var cpts = unicode.cptsFromUtf8(text);
    defer allocator.free(cpts);
    if (opts.strip_accents) {
        const nfd = unicode.cptsNormalizeNfd(cpts);
        allocator.free(cpts);
        cpts = nfd;
    }

    var words: std.ArrayList(std.ArrayList(u8)) = .empty;
    errdefer {
        for (words.items) |*w| w.deinit(allocator);
        words.deinit(allocator);
    }
    words.append(allocator, .empty) catch oom();

    for (cpts) |cpt| {
        const flags = unicode.cptFlagsFromCpt(cpt);

        if (flags.is_whitespace) {
            if (last(&words).items.len != 0) words.append(allocator, .empty) catch oom(); // finish previous word if any
            continue;
        }

        if (cpt == 0 or cpt == 0xFFFD or flags.is_control) continue;

        if (opts.strip_accents and flags.is_accent_mark) continue;

        var b: [4]u8 = undefined;
        const s = unicode.cptToUtf8(if (opts.lowercase) unicode.tolower(cpt) else cpt, &b) catch
            return exc.throw(.invalid_argument, "invalid codepoint", .{});
        if (flags.is_punctuation or (cpt < 0x7F and flags.is_symbol) or isChineseChar(cpt)) {
            if (last(&words).items.len != 0) words.append(allocator, .empty) catch oom(); // finish previous word if any
            const w = last(&words);
            w.clearRetainingCapacity();
            w.appendSlice(allocator, s) catch oom(); // single char word
            words.append(allocator, .empty) catch oom(); // start a new word
        } else {
            last(&words).appendSlice(allocator, s) catch oom(); // append char to word
        }
    }

    if (last(&words).items.len == 0) {
        var w = words.pop().?;
        w.deinit(allocator);
    }

    return words;
}

fn last(words: *std.ArrayList(std.ArrayList(u8))) *std.ArrayList(u8) {
    return &words.items[words.items.len - 1];
}

/// Ports `llm_tokenizer_wpm_session`'s `is_chinese_char` (llama-vocab.cpp:863
/// @5266f24da). Extension E starts at 0x2B920 here, as in the HF Rust code
/// it follows, not at 0x2B820.
pub fn isChineseChar(cpt: u32) bool {
    return (cpt >= 0x04E00 and cpt <= 0x09FFF) or
        (cpt >= 0x03400 and cpt <= 0x04DBF) or
        (cpt >= 0x20000 and cpt <= 0x2A6DF) or
        (cpt >= 0x2A700 and cpt <= 0x2B73F) or
        (cpt >= 0x2B740 and cpt <= 0x2B81F) or
        (cpt >= 0x2B920 and cpt <= 0x2CEAF) or
        (cpt >= 0x0F900 and cpt <= 0x0FAFF) or
        (cpt >= 0x2F800 and cpt <= 0x2FA1F);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "preprocess splits on spaces and isolates punctuation" {
    var exc: impl.Exception = .{};
    var words = try preprocess("Hello, World", .{}, &exc);
    defer {
        for (words.items) |*w| w.deinit(allocator);
        words.deinit(allocator);
    }
    try std.testing.expectEqual(@as(usize, 3), words.items.len);
    try std.testing.expectEqualStrings("hello", words.items[0].items);
    try std.testing.expectEqualStrings(",", words.items[1].items);
    try std.testing.expectEqualStrings("world", words.items[2].items);
}
