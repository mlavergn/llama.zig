//! The Unigram (UGM) tokenizer T5 models use: normalize through the
//! model's precompiled character map, then a Viterbi search for the
//! highest-scoring segmentation.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` at v0.4.1 (`b29c606e2`).
//!
//! Scores are summed in `double`, as there -- "to make tokenization results
//! exactly the same as in the HF tokenizer". Nothing is multiplied, so no
//! contraction is in play.

const std = @import("std");
const impl = @import("../impl.zig");
const libcxx = @import("../libcxx.zig");
const unicode = @import("../unicode.zig");
const vm = @import("module.zig");

const allocator = impl.allocator;
const Vocab = vm.Vocab;
const Token = vm.Token;
const Trie = vm.trie.Trie;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

/// Ports `llm_tokenizer_ugm` (llama-vocab.cpp:897 @b29c606e2).
pub const Tokenizer = struct {
    /// U+2581, the escaped space.
    pub const escaped_space = "\xE2\x96\x81";

    /// The replacement strings the XCDA indexes into, borrowed from the
    /// vocabulary's `precompiled_charsmap`.
    prefix_replacements: []const u8 = &.{},
    /// The XCDA entries, also borrowed from it.
    xcda: []const u8 = &.{},

    user_defined_token_matcher: Trie,
    token_matcher: Trie,

    min_score: f32 = std.math.floatMax(f32),
    max_score: f32 = -std.math.floatMax(f32),

    unknown_token_score_penalty: f32 = 10.0,
    unknown_token_score: f32 = 0,

    /// Ports the constructor of `llm_tokenizer_ugm`.
    ///
    /// Return: the tokenizer; it borrows `vocab`'s charsmap.
    pub fn init(vocab: *const Vocab) Tokenizer {
        var t: Tokenizer = .{ .user_defined_token_matcher = Trie.init(), .token_matcher = Trie.init() };
        const charsmap = vocab.precompiled_charsmap.items;
        if (charsmap.len > 0) {
            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            const xcda_blob_size = std.mem.readInt(u32, charsmap[0..4], .little);
            // `load` checked that the blob ends inside the map.
            t.xcda = charsmap[4..][0 .. xcda_blob_size / 4 * 4];
            t.prefix_replacements = charsmap[4 + xcda_blob_size ..];
        }

        const n: u32 = vocab.nTokens();
        var id: u32 = 0;
        while (id < n) : (id += 1) {
            const tid: Token = @intCast(id);
            const d = vocab.id_to_token.items[id];
            if (vocab.isNormal(tid)) {
                t.min_score = libcxx.min(f32, t.min_score, d.score);
                t.max_score = libcxx.max(f32, t.max_score, d.score);
            }
            if (vocab.isNormal(tid) or vocab.isUserDefined(tid) or vocab.isUnused(tid)) {
                t.token_matcher.insert(vocab.text(tid), tid);
            }
            if (vocab.isUserDefined(tid)) {
                t.user_defined_token_matcher.insert(vocab.text(tid), 0);
            }
        }

        t.unknown_token_score = t.min_score - t.unknown_token_score_penalty;
        return t;
    }

    /// Return: nothing.
    pub fn deinit(self: *Tokenizer) void {
        self.user_defined_token_matcher.deinit();
        self.token_matcher.deinit();
    }

    fn xcdaSize(self: *const Tokenizer) usize {
        return self.xcda.len / 4;
    }
};

/// Ports `xcda_array_view` (llama-vocab.cpp:1127 @b29c606e2).
const XcdaView = struct {
    xcda: []const u8,

    fn getNode(self: XcdaView, index: usize, exc: *impl.Exception) impl.Thrown!u32 {
        if (index >= self.xcda.len / 4) return exc.throw(.runtime_error, "Index out of array bounds in XCDA array!", .{});
        return std.mem.readInt(u32, self.xcda[index * 4 ..][0..4], .little);
    }

    fn getBase(self: XcdaView, index: usize, exc: *impl.Exception) impl.Thrown!u32 {
        const packed_node = try self.getNode(index, exc);
        return (packed_node >> 10) << @intCast((packed_node & (1 << 9)) >> 6);
    }

    fn getLcheck(self: XcdaView, index: usize, exc: *impl.Exception) impl.Thrown!u32 {
        const packed_node = try self.getNode(index, exc);
        return packed_node & ((1 << 31) | 0xff);
    }

    fn getLeaf(self: XcdaView, index: usize, exc: *impl.Exception) impl.Thrown!bool {
        const packed_node = try self.getNode(index, exc);
        return ((packed_node >> 8) & 1) != 0;
    }

    fn getValue(self: XcdaView, index: usize, exc: *impl.Exception) impl.Thrown!u32 {
        const packed_node = try self.getNode(index, exc);
        return packed_node & ((1 << 31) - 1);
    }
};

/// Ports `best_tokenization`.
const BestTokenization = struct {
    token_id: Token,
    input_offset: usize,
    score_sum: f64,
};

/// Ports `normalization_result`: a replacement, as a slice of the input or
/// of the charsmap.
const NormalizationResult = struct {
    normalized: []const u8,
    consumed_input: usize,
};

/// Ports `llm_tokenizer_ugm_session`'s `tokenize` (llama-vocab.cpp:977
/// @b29c606e2).
///
/// Parameters:
/// - `vocab`: the vocabulary.
/// - `tokenizer`: its UGM tokenizer.
/// - `text`: one fragment.
/// - `output`: tokens are appended here.
/// - `exc`: receives what is thrown.
///
/// Return: nothing, or `error.Thrown` from a malformed charsmap.
pub fn tokenize(vocab: *const Vocab, tokenizer: *const Tokenizer, text: []const u8, output: *std.ArrayList(Token), exc: *impl.Exception) impl.Thrown!void {
    // get current size of output (for reversal later)
    const output_size = output.items.len;

    // normalize the input first
    var normalized: std.ArrayList(u8) = .empty;
    defer normalized.deinit(allocator);
    try normalize(vocab, tokenizer, text, &normalized, exc);
    const norm = normalized.items;
    const input_len = norm.len;
    if (input_len == 0) return;

    // `normalized[i]` on a `std::string` answers NUL at `size()`.
    const at = struct {
        fn f(s: []const u8, i: usize) u8 {
            return if (i < s.len) s[i] else 0;
        }
    }.f;

    const unk = vocab.special_unk_id;
    const results = allocator.alloc(BestTokenization, input_len + 1) catch oom();
    defer allocator.free(results);
    @memset(results, .{ .token_id = unk, .input_offset = 0, .score_sum = -std.math.floatMax(f64) });
    // at the beginning tokenization score is zero
    results[0] = .{ .token_id = unk, .input_offset = 0, .score_sum = 0 };

    var input_offset: usize = 0;
    while (input_offset < input_len) {
        var prefix_offset = input_offset;
        // how many code units are in the currently processed UTF code point
        const n_utf8_code_units = @min(unicode.lenUtf8(norm[input_offset]), input_len - input_offset);

        // traverse the token matcher trie to find a matching token
        var single_codepoint_token_found = false;
        const current_best = results[input_offset];
        var node = tokenizer.token_matcher.traverse(Trie.root, at(norm, prefix_offset));
        prefix_offset += 1;

        while (prefix_offset <= input_len) {
            const nd = node orelse break;
            const value = tokenizer.token_matcher.get(nd);
            // check if we found valid token in prefix
            if (value.has_value) {
                // check if it corresponds to the whole UTF code point
                if (prefix_offset - input_offset == n_utf8_code_units) single_codepoint_token_found = true;
                const token_id: Token = value.value;
                const token_data = try vocab.getTokenData(token_id, exc);

                // user-defined tokens score 0, to make them more likely to be
                // selected; normal token scores are log probabilities.
                const token_score: f64 = if (vocab.isUserDefined(token_id)) 0.0 else token_data.score;
                const challenger_score = current_best.score_sum + token_score;
                const current_champ = &results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    current_champ.* = .{ .token_id = token_id, .input_offset = input_offset, .score_sum = challenger_score };
                }
            }
            node = tokenizer.token_matcher.traverse(nd, at(norm, prefix_offset));
            prefix_offset += 1;
        }

        // if we didn't find a valid token corresponding to the whole UTF code point
        // then use unknown token as the tokenization of this UTF code point
        if (!single_codepoint_token_found) {
            const challenger_score = current_best.score_sum + @as(f64, tokenizer.unknown_token_score);
            prefix_offset = input_offset + n_utf8_code_units;
            const current_champ = &results[prefix_offset];
            if (challenger_score > current_champ.score_sum) {
                current_champ.* = .{ .token_id = unk, .input_offset = input_offset, .score_sum = challenger_score };
            }
        }

        // move to the next UTF code point
        input_offset += n_utf8_code_units;
    }

    // now backtrack from the end to gather token ids of the best tokenization
    // merge sequences of consecutive unknown tokens into single unknown tokens
    var is_prev_unknown = false;
    var tokenization = results[input_len];
    while (true) {
        const is_unknown = tokenization.token_id == unk;
        if (!(is_prev_unknown and is_unknown)) output.append(allocator, tokenization.token_id) catch oom();
        if (tokenization.input_offset == 0) break;
        is_prev_unknown = is_unknown;
        tokenization = results[tokenization.input_offset];
    }

    // reverse the output since we added tokens starting from the end of the input
    std.mem.reverse(Token, output.items[output_size..]);
}

/// Ports `llm_tokenizer_ugm_session`'s `normalize` (llama-vocab.cpp:1072
/// @b29c606e2).
fn normalize(vocab: *const Vocab, tokenizer: *const Tokenizer, input: []const u8, normalized: *std.ArrayList(u8), exc: *impl.Exception) impl.Thrown!void {
    normalized.clearRetainingCapacity();
    normalized.ensureTotalCapacity(allocator, input.len * 3) catch oom();

    const space: []const u8 = if (vocab.escape_whitespaces) Tokenizer.escaped_space else " ";

    const shall_prepend_space = !vocab.treat_whitespace_as_suffix and vocab.add_space_prefix;
    const shall_append_space = vocab.treat_whitespace_as_suffix and vocab.add_space_prefix;
    const shall_merge_spaces = vocab.remove_extra_whitespaces;

    var is_space_prepended = false;
    var processing_non_ws = false;

    var input_offset: usize = 0;
    while (input_offset < input.len) {
        const norm_res = try normalizePrefix(tokenizer, input, input_offset, exc);
        for (norm_res.normalized) |ch| {
            if (ch != ' ') {
                if (!processing_non_ws) {
                    processing_non_ws = true;
                    if ((shall_prepend_space and !is_space_prepended) or shall_merge_spaces) {
                        normalized.appendSlice(allocator, space) catch oom();
                        is_space_prepended = true;
                    }
                }
                normalized.append(allocator, ch) catch oom();
            } else {
                if (processing_non_ws) processing_non_ws = false;
                if (!shall_merge_spaces) normalized.appendSlice(allocator, space) catch oom();
            }
        }
        input_offset += norm_res.consumed_input;
    }

    if (shall_append_space) normalized.appendSlice(allocator, space) catch oom();
}

/// Ports `llm_tokenizer_ugm_session`'s `normalize_prefix` (llama-vocab.cpp:1165
/// @b29c606e2): a user-defined token passes through whole; else the
/// longest XCDA match is replaced; else one code point passes through, or
/// an invalid byte becomes U+FFFD.
fn normalizePrefix(tokenizer: *const Tokenizer, input: []const u8, input_offset: usize, exc: *impl.Exception) impl.Thrown!NormalizationResult {
    if (input_offset == input.len) return .{ .normalized = input[input_offset..input_offset], .consumed_input = 0 };

    // if input prefix matches some user-defined token return this token as normalization result
    const user_defined_len = tokenizer.user_defined_token_matcher.longestPrefix(input[input_offset..]);
    if (user_defined_len > 0) {
        return .{ .normalized = input[input_offset..][0..user_defined_len], .consumed_input = user_defined_len };
    }

    var longest_prefix_length: usize = 0;
    var longest_prefix_offset: usize = 0;

    if (tokenizer.xcdaSize() > 0) {
        const view: XcdaView = .{ .xcda = tokenizer.xcda };

        // Walk the XCDA from the root: the next node is BASE[s] ^ c.
        var node_index: u32 = 0;
        node_index = try view.getBase(node_index, exc);
        var prefix_offset = input_offset;
        while (prefix_offset < input.len) : (prefix_offset += 1) {
            const ch = input[prefix_offset];
            if (ch == 0) break;
            node_index ^= ch;
            // if LCHECK is not c, this is not a child of the previous node
            if ((try view.getLcheck(node_index, exc)) != ch) break;
            const is_leaf = try view.getLeaf(node_index, exc);
            node_index ^= try view.getBase(node_index, exc);
            // a leaf's BASE points at the node holding the replacement's index
            if (is_leaf) {
                longest_prefix_length = prefix_offset - input_offset + 1;
                longest_prefix_offset = try view.getValue(node_index, exc);
            }
        }
    }

    if (longest_prefix_length > 0) {
        // we have a match, so return the replacement sequence
        if (longest_prefix_offset >= tokenizer.prefix_replacements.len) {
            return exc.throw(.runtime_error, "Index out of array bounds in precompiled charsmap!", .{});
        }
        const replacement = tokenizer.prefix_replacements[longest_prefix_offset..];
        const repl_len = std.mem.findScalar(u8, replacement, 0) orelse
            return exc.throw(.runtime_error, "Unterminated string in precompiled charsmap!", .{});
        return .{ .normalized = replacement[0..repl_len], .consumed_input = longest_prefix_length };
    }

    // a valid UTF-8 sequence passes through unmodified
    var prefix_offset = input_offset;
    if (unicode.cptFromUtf8(input, &prefix_offset)) |_| {
        return .{ .normalized = input[input_offset..prefix_offset], .consumed_input = prefix_offset - input_offset };
    } else |_| {
        // consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
        return .{ .normalized = "\xEF\xBF\xBD", .consumed_input = 1 };
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
