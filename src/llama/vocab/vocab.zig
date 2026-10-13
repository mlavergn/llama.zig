//! The vocabulary: loading it from GGUF metadata, and tokenizing and
//! detokenizing through it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-vocab.cpp` and
//! `llama.cpp/src/llama-vocab.h` at v0.3.0 (`c1d0e7a00`), with `replace_all`
//! from `llama.cpp/src/llama-impl.cpp`. `struct llama_vocab` and its pimpl
//! `llama_vocab::impl` are one struct here; the C++'s forwarding layer adds
//! nothing.
//!
//! # Contract
//!
//! Nothing is exported yet. libllama swaps once, at `llama.h` (`PLAN.md`,
//! Decision 42), so until then this is reached only through
//! `scripts/vocab-diff`, which loads the same files through the C++ and
//! compares. The `llama_vocab_*` and `llama_tokenize` entry points move
//! with the swap.
//!
//! # What is libc++'s and not the C++'s
//!
//! Two orders here are the standard library's: `token_to_id` is a
//! `std::unordered_map`, iterated four times while loading -- and the first
//! token whose text matches decides which id becomes EOT and which attributes
//! change -- so it is `libcxx.StringMap`, which lays nodes out as libc++
//! does; and `cache_special_tokens` is sorted by length with an unstable
//! `std::sort`, whose order among equal lengths decides which overlapping
//! special token wins, so it is `libcxx.sort`. Both are gated by
//! `libcxx-diff`.

const std = @import("std");
const impl = @import("../impl.zig");
const libcxx = @import("../libcxx.zig");
const unicode = @import("../unicode.zig");
const arch = @import("../arch.zig");
const gguf_meta = @import("../gguf_meta.zig");
const vm = @import("module.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

fn oom() noreturn {
    impl.abort("llama-vocab.cpp", 0, "out of memory");
}

pub const Token = c.llama_token;
pub const token_null: Token = c.LLAMA_TOKEN_NULL;
pub const Attr = c.enum_llama_token_attr;

const attr_unknown: Attr = c.LLAMA_TOKEN_ATTR_UNKNOWN;
const attr_unused: Attr = c.LLAMA_TOKEN_ATTR_UNUSED;
const attr_normal: Attr = c.LLAMA_TOKEN_ATTR_NORMAL;
const attr_control: Attr = c.LLAMA_TOKEN_ATTR_CONTROL;
const attr_user_defined: Attr = c.LLAMA_TOKEN_ATTR_USER_DEFINED;
const attr_byte: Attr = c.LLAMA_TOKEN_ATTR_BYTE;
const attr_lstrip: Attr = c.LLAMA_TOKEN_ATTR_LSTRIP;
const attr_rstrip: Attr = c.LLAMA_TOKEN_ATTR_RSTRIP;
const attr_undefined: Attr = c.LLAMA_TOKEN_ATTR_UNDEFINED;

pub const Type = c.enum_llama_vocab_type;
pub const type_none: Type = c.LLAMA_VOCAB_TYPE_NONE;
pub const type_spm: Type = c.LLAMA_VOCAB_TYPE_SPM;
pub const type_bpe: Type = c.LLAMA_VOCAB_TYPE_BPE;
pub const type_wpm: Type = c.LLAMA_VOCAB_TYPE_WPM;
pub const type_ugm: Type = c.LLAMA_VOCAB_TYPE_UGM;
pub const type_rwkv: Type = c.LLAMA_VOCAB_TYPE_RWKV;
pub const type_plamo2: Type = c.LLAMA_VOCAB_TYPE_PLAMO2;

/// Ports `enum llama_vocab_pre_type` (llama-vocab.h:10 @c1d0e7a00).
pub const PreType = enum(c_int) {
    default = 0,
    llama3 = 1,
    deepseek_llm = 2,
    deepseek_coder = 3,
    falcon = 4,
    mpt = 5,
    starcoder = 6,
    gpt2 = 7,
    refact = 8,
    command_r = 9,
    stablelm2 = 10,
    qwen2 = 11,
    olmo = 12,
    dbrx = 13,
    smaug = 14,
    poro = 15,
    chatglm3 = 16,
    chatglm4 = 17,
    viking = 18,
    jais = 19,
    tekken = 20,
    smollm = 21,
    codeshell = 22,
    bloom = 23,
    gpt3_finnish = 24,
    exaone = 25,
    chameleon = 26,
    minerva = 27,
    deepseek3_llm = 28,
    gpt4o = 29,
    superbpe = 30,
    trillion = 31,
    bailingmoe = 32,
    llama4 = 33,
    pixtral = 34,
    seed_coder = 35,
    hunyuan = 36,
    kimi_k2 = 37,
    hunyuan_dense = 38,
    grok_2 = 39,
    granite_docling = 40,
    minimax_m2 = 41,
    afmoe = 42,
    solar_open = 43,
    youtu = 44,
    exaone_moe = 45,
    qwen35 = 46,
    tiny_aya = 47,
    joyai_llm = 48,
    jais2 = 49,
    gemma4 = 50,
    sarvam_moe = 51,
    minicpm5 = 52,
    whitespace = 53,
    granite_emb_multi = 54,
    mellum2 = 55,
    laguna = 56,
};

/// Ports `llama_vocab`'s `token_data` (llama-vocab.h:74 @c1d0e7a00). The text
/// lives in `Vocab.text_buf`, NUL-terminated, as `c_str()` exposes it.
pub const TokenData = struct {
    off: u32,
    len: u32,
    score: f32,
    attr: Attr,
};

/// Ports `llama_vocab`'s `normalizer_options` (llama-vocab.h:80 @c1d0e7a00).
pub const NormalizerOptions = struct {
    lowercase: bool = true,
    strip_accents: bool = true,
};

/// The merge table, `std::unordered_map<std::pair<std::string,
/// std::string>, int, pair_hash>`. Only looked up, so any hash will do;
/// the pair's halves live in `buf`.
const BpeRanks = struct {
    const Key = struct { l_off: u32, l_len: u32, r_off: u32, r_len: u32 };
    const Pair = struct { left: []const u8, right: []const u8 };

    buf: std.ArrayList(u8) = .empty,
    map: std.HashMapUnmanaged(Key, i32, Ctx, 80) = .empty,

    const Ctx = struct {
        buf: []const u8,
        pub fn hash(self: Ctx, k: Key) u64 {
            return hashPair(.{ .left = self.buf[k.l_off..][0..k.l_len], .right = self.buf[k.r_off..][0..k.r_len] });
        }
        pub fn eql(self: Ctx, a: Key, b: Key) bool {
            return std.mem.eql(u8, self.buf[a.l_off..][0..a.l_len], self.buf[b.l_off..][0..b.l_len]) and
                std.mem.eql(u8, self.buf[a.r_off..][0..a.r_len], self.buf[b.r_off..][0..b.r_len]);
        }
    };

    const Adapted = struct {
        buf: []const u8,
        pub fn hash(_: Adapted, p: Pair) u64 {
            return hashPair(p);
        }
        pub fn eql(self: Adapted, p: Pair, k: Key) bool {
            return std.mem.eql(u8, p.left, self.buf[k.l_off..][0..k.l_len]) and std.mem.eql(u8, p.right, self.buf[k.r_off..][0..k.r_len]);
        }
    };

    fn hashPair(p: Pair) u64 {
        return std.hash.Wyhash.hash(0, p.left) ^ (std.hash.Wyhash.hash(1, p.right) << 1);
    }

    fn deinit(self: *BpeRanks) void {
        self.buf.deinit(allocator);
        self.map.deinit(allocator);
    }

    fn count(self: *const BpeRanks) usize {
        return self.map.count();
    }

    /// `bpe_ranks.emplace(pair, rank)`: the first rank for a pair stays.
    fn emplace(self: *BpeRanks, left: []const u8, right: []const u8, rank: i32) void {
        const ctx: Adapted = .{ .buf = self.buf.items };
        if (self.map.containsAdapted(Pair{ .left = left, .right = right }, ctx)) return;
        const l_off: u32 = @intCast(self.buf.items.len);
        self.buf.appendSlice(allocator, left) catch oom();
        const r_off: u32 = @intCast(self.buf.items.len);
        self.buf.appendSlice(allocator, right) catch oom();
        const key: Key = .{ .l_off = l_off, .l_len = @intCast(left.len), .r_off = r_off, .r_len = @intCast(right.len) };
        self.map.putContext(allocator, key, rank, .{ .buf = self.buf.items }) catch oom();
    }

    fn find(self: *const BpeRanks, left: []const u8, right: []const u8) ?i32 {
        return self.map.getAdapted(Pair{ .left = left, .right = right }, Adapted{ .buf = self.buf.items });
    }
};

/// The tokenizer `init_tokenizer` builds, one per vocabulary type.
pub const Tokenizer = union(enum) {
    none,
    spm,
    bpe: vm.bpe.Tokenizer,
    wpm,
    ugm: vm.ugm.Tokenizer,
    rwkv: vm.rwkv.Tokenizer,
    plamo2: vm.plamo2.Tokenizer,

    fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            .bpe => |*t| t.deinit(),
            .ugm => |*t| t.deinit(),
            .rwkv => |*t| t.deinit(),
            .plamo2 => |*t| t.deinit(),
            else => {},
        }
    }
};

/// A piece of the text being tokenized: a token already decided, or a
/// span of the raw text still to tokenize. Ports `fragment_buffer_variant`
/// (llama-vocab.cpp:1752 @c1d0e7a00).
pub const Fragment = union(enum) {
    token: Token,
    raw: struct { offset: usize, length: usize },
};

/// Ports `llama_vocab` and `llama_vocab`'s `impl` (llama-vocab.cpp:1781
/// @c1d0e7a00).
pub const Vocab = struct {
    n_token_types: u32 = 0,

    tokenizer_model: std.ArrayList(u8) = .empty,
    tokenizer_pre: std.ArrayList(u8) = .empty,

    type: Type = type_spm,
    pre_type: PreType = .default,

    max_token_len: i32 = 0,

    special_bos_id: Token = 1,
    special_eos_id: Token = 2,
    special_eot_id: Token = token_null,
    special_eom_id: Token = token_null,
    special_unk_id: Token = 0,
    special_sep_id: Token = token_null,
    special_pad_id: Token = token_null,
    special_mask_id: Token = token_null,

    linefeed_id: Token = 13,

    special_fim_pre_id: Token = token_null,
    special_fim_suf_id: Token = token_null,
    special_fim_mid_id: Token = token_null,
    special_fim_pad_id: Token = token_null,
    special_fim_rep_id: Token = token_null,
    special_fim_sep_id: Token = token_null,

    add_space_prefix: bool = false,
    add_bos: bool = false,
    add_eos: bool = false,
    add_sep: bool = false,
    ignore_merges: bool = false,
    clean_spaces: bool = false,
    remove_extra_whitespaces: bool = false,
    escape_whitespaces: bool = true,
    treat_whitespace_as_suffix: bool = false,

    normalizer_opts: NormalizerOptions = .{},

    token_to_id: libcxx.StringMap(Token),
    id_to_token: std.ArrayList(TokenData) = .empty,
    /// Every token's text, each followed by a NUL.
    text_buf: std.ArrayList(u8) = .empty,

    cache_special_tokens: std.ArrayList(Token) = .empty,
    /// `cache_token_to_piece`: piece `i` is `piece_buf[piece_off[i]..piece_off[i + 1]]`.
    piece_buf: std.ArrayList(u8) = .empty,
    piece_off: std.ArrayList(u32) = .empty,

    bpe_ranks: BpeRanks = .{},

    /// `std::set<llama_token>`: kept sorted, as the set iterates.
    special_eog_ids: std.ArrayList(Token) = .empty,

    suppress_tokens: std.ArrayList(Token) = .empty,

    tokenizer: Tokenizer = .none,

    precompiled_charsmap: std.ArrayList(u8) = .empty,

    /// Ports `llama_vocab`'s `llama_vocab` (llama-vocab.cpp:3844 @c1d0e7a00).
    ///
    /// Return: an empty vocabulary, reserved for the lifecycle convention as
    /// fallible.
    pub fn init() error{OutOfMemory}!Vocab {
        return .{ .token_to_id = try libcxx.StringMap(Token).init(allocator) };
    }

    /// Return: nothing.
    pub fn deinit(self: *Vocab) void {
        self.tokenizer.deinit();
        self.tokenizer_model.deinit(allocator);
        self.tokenizer_pre.deinit(allocator);
        self.token_to_id.deinit();
        self.id_to_token.deinit(allocator);
        self.text_buf.deinit(allocator);
        self.cache_special_tokens.deinit(allocator);
        self.piece_buf.deinit(allocator);
        self.piece_off.deinit(allocator);
        self.bpe_ranks.deinit();
        self.special_eog_ids.deinit(allocator);
        self.suppress_tokens.deinit(allocator);
        self.precompiled_charsmap.deinit(allocator);
    }

    // -------------------------------------------------------------------------
    // Accessors

    /// Ports `llama_vocab`'s `n_tokens` (llama-vocab.cpp:3869 @c1d0e7a00).
    pub fn nTokens(self: *const Vocab) u32 {
        return @intCast(self.id_to_token.items.len);
    }

    /// Ports `llama_vocab::impl`'s `type_name` (llama-vocab.cpp:3100 @c1d0e7a00).
    pub fn typeName(self: *const Vocab) [:0]const u8 {
        return switch (self.type) {
            type_none => "no vocab",
            type_spm => "SPM",
            type_bpe => "BPE",
            type_wpm => "WPM",
            type_ugm => "UGM",
            type_rwkv => "RWKV",
            type_plamo2 => "PLaMo2",
            else => "unknown",
        };
    }

    /// The text of token `id`, with no bounds check, as `id_to_token[id]`.
    pub fn text(self: *const Vocab, id: Token) [:0]const u8 {
        const d = self.id_to_token.items[@intCast(id)];
        return self.text_buf.items[d.off..][0..d.len :0];
    }

    fn assertNotNone(self: *const Vocab, comptime line: c_int) void {
        impl.assert(self.type != type_none, "llama-vocab.cpp", line, "type != LLAMA_VOCAB_TYPE_NONE");
    }

    /// Ports `llama_vocab::impl`'s `is_normal` (llama-vocab.cpp:3113 @c1d0e7a00).
    pub fn isNormal(self: *const Vocab, id: Token) bool {
        self.assertNotNone(3114);
        return (self.id_to_token.items[@intCast(id)].attr & attr_normal) != 0;
    }

    /// Ports `llama_vocab::impl`'s `is_unknown` (llama-vocab.cpp:3118 @c1d0e7a00).
    pub fn isUnknown(self: *const Vocab, id: Token) bool {
        self.assertNotNone(3119);
        return (self.id_to_token.items[@intCast(id)].attr & attr_unknown) != 0;
    }

    /// Ports `llama_vocab::impl`'s `is_control` (llama-vocab.cpp:3123 @c1d0e7a00).
    pub fn isControl(self: *const Vocab, id: Token) bool {
        self.assertNotNone(3124);
        return (self.id_to_token.items[@intCast(id)].attr & attr_control) != 0;
    }

    /// Ports `llama_vocab::impl`'s `is_byte` (llama-vocab.cpp:3128 @c1d0e7a00).
    pub fn isByte(self: *const Vocab, id: Token) bool {
        self.assertNotNone(3129);
        return (self.id_to_token.items[@intCast(id)].attr & attr_byte) != 0;
    }

    /// Ports `llama_vocab::impl`'s `is_user_defined` (llama-vocab.cpp:3133 @c1d0e7a00).
    pub fn isUserDefined(self: *const Vocab, id: Token) bool {
        self.assertNotNone(3134);
        return (self.id_to_token.items[@intCast(id)].attr & attr_user_defined) != 0;
    }

    /// Ports `llama_vocab::impl`'s `is_unused` (llama-vocab.cpp:3138 @c1d0e7a00).
    pub fn isUnused(self: *const Vocab, id: Token) bool {
        self.assertNotNone(3139);
        return (self.id_to_token.items[@intCast(id)].attr & attr_unused) != 0;
    }

    /// Ports `llama_vocab::impl`'s `is_eog` (llama-vocab.cpp:3143 @c1d0e7a00).
    pub fn isEog(self: *const Vocab, id: Token) bool {
        return id != token_null and eogIndex(self.special_eog_ids.items, id) != null;
    }

    /// Ports `llama_vocab`'s `text_to_token` (llama-vocab.cpp:3943 @c1d0e7a00).
    ///
    /// Return: the token whose text is `t`, or `token_null`.
    pub fn textToToken(self: *const Vocab, t: []const u8) Token {
        impl.assert(self.type != type_none, "llama-vocab.cpp", 3944, "pimpl->type != LLAMA_VOCAB_TYPE_NONE");
        return if (self.token_to_id.get(t)) |v| v.* else token_null;
    }

    /// `id_to_token.at(id)`.
    ///
    /// Return: the token's data, or `error.Thrown` with
    /// `std::out_of_range("vector")`.
    pub fn tokenDataAt(self: *const Vocab, id: Token, exc: *Exception) Thrown!TokenData {
        if (id < 0 or id >= self.id_to_token.items.len) return exc.throw(.out_of_range, "vector", .{});
        return self.id_to_token.items[@intCast(id)];
    }

    /// Ports `llama_vocab`'s `get_token_data` (llama-vocab.cpp:3952 @c1d0e7a00).
    pub fn getTokenData(self: *const Vocab, id: Token, exc: *Exception) Thrown!TokenData {
        impl.assert(self.type != type_none, "llama-vocab.cpp", 3953, "pimpl->type != LLAMA_VOCAB_TYPE_NONE");
        return self.tokenDataAt(id, exc);
    }

    /// Ports `llama_vocab::impl`'s `token_get_attr` (llama-vocab.cpp:3170 @c1d0e7a00).
    pub fn tokenGetAttr(self: *const Vocab, id: Token, exc: *Exception) Thrown!Attr {
        self.assertNotNone(3171);
        return (try self.tokenDataAt(id, exc)).attr;
    }

    /// Ports `llama_vocab`'s `find_bpe_rank` (llama-vocab.cpp:4091 @c1d0e7a00).
    ///
    /// Return: the merge's rank, or -1.
    pub fn findBpeRank(self: *const Vocab, left: []const u8, right: []const u8) i32 {
        impl.assert(std.mem.findScalar(u8, left, ' ') == null, "llama-vocab.cpp", 4092, "token_left.find(' ')   == std::string::npos");
        impl.assert(std.mem.findScalar(u8, right, ' ') == null, "llama-vocab.cpp", 4093, "token_right.find(' ')  == std::string::npos");
        return self.bpe_ranks.find(left, right) orelse -1;
    }

    /// The number of merges, `bpe_ranks.size()`.
    pub fn nMerges(self: *const Vocab) usize {
        return self.bpe_ranks.count();
    }

    /// Ports `llama_vocab::impl`'s `token_to_byte` (llama-vocab.cpp:3147 @c1d0e7a00).
    ///
    /// Return: the byte a `<0xXX>` token stands for, or `error.Thrown`.
    pub fn tokenToByte(self: *const Vocab, id: Token, exc: *Exception) Thrown!u8 {
        impl.assert(self.type != type_none, "llama-vocab.cpp", 3148, "get_type() != LLAMA_VOCAB_TYPE_NONE");
        impl.assert(self.isByte(id), "llama-vocab.cpp", 3149, "is_byte(id)");
        const d = try self.tokenDataAt(id, exc);
        switch (self.type) {
            type_spm, type_ugm, type_bpe => {
                const t = self.text_buf.items[d.off..][0..d.len];
                // `token_data.text.substr(3, 2)`.
                if (t.len < 3) return exc.throw(.out_of_range, "basic_string", .{});
                return @truncate(@as(u64, @bitCast(strtolHex(t[3..@min(t.len, 5)]))));
            },
            type_wpm => impl.abort("llama-vocab.cpp", 3163, "fatal error"),
            else => impl.abort("llama-vocab.cpp", 3166, "fatal error"),
        }
    }

    /// Ports `llama_vocab`'s `byte_to_token` (llama-vocab.cpp:3913 @c1d0e7a00).
    ///
    /// Return: the token for byte `ch`, or `error.Thrown` with
    /// `unordered_map`'s `at`'s `std`'s `out_of_range`.
    pub fn byteToToken(self: *const Vocab, ch: u8, exc: *Exception) Thrown!Token {
        impl.assert(self.type != type_none, "llama-vocab.cpp", 3914, "get_type() != LLAMA_VOCAB_TYPE_NONE");
        const hex = "0123456789ABCDEF";
        switch (self.type) {
            type_spm, type_ugm => {
                const buf = [_]u8{ '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>' };
                if (self.token_to_id.get(&buf)) |v| return v.*;
                // `buf2` is a C string: a NUL byte is the empty key.
                const buf2: []const u8 = if (ch == 0) "" else &[_]u8{ch};
                return self.atKey(buf2, exc);
            },
            type_wpm, type_bpe => return self.atKey(unicode.byteToUtf8(ch), exc),
            type_plamo2 => {
                var buf: [8]u8 = undefined;
                const s = std.mem.print(&buf, "<0x{X:0>2}>", .{ch}) catch unreachable;
                return self.atKey(s, exc);
            },
            else => impl.abort("llama-vocab.cpp", 3939, "fatal error"),
        }
    }

    /// `token_to_id.at(key)`.
    fn atKey(self: *const Vocab, key: []const u8, exc: *Exception) Thrown!Token {
        if (self.token_to_id.get(key)) |v| return v.*;
        return exc.throw(.out_of_range, "unordered_map::at: key not found", .{});
    }

    // -------------------------------------------------------------------------
    // Loading

    /// Ports `llama_vocab::impl`'s `load` (llama-vocab.cpp:1923 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `ml`: the metadata, as `llama_model_loader` reads it.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: nothing, or `error.Thrown` -- every `std`'s `exception` the C++
    /// lets escape, which `llama_model_load` catches and reports.
    pub fn load(self: *Vocab, ml: *const gguf_meta.Meta, exc: *Exception) Thrown!void {
        const ctx = ml.ctx;

        // determine vocab type
        {
            const model = (try ml.getKey([]const u8, .tokenizer_model, true, exc)).?;
            self.tokenizer_model.appendSlice(allocator, model) catch oom();
            if (try ml.getKey([]const u8, .tokenizer_pre, false, exc)) |pre| {
                self.tokenizer_pre.clearRetainingCapacity();
                self.tokenizer_pre.appendSlice(allocator, pre) catch oom();
            }
            if (try ml.getKey(u32, .tokenizer_token_type_count, false, exc)) |n| self.n_token_types = n;

            const tm = self.tokenizer_model.items;
            if (eql(tm, "no_vocab") or eql(tm, "none")) {
                self.type = type_none;

                self.special_bos_id = token_null;
                self.special_eos_id = token_null;
                self.special_unk_id = token_null;
                self.special_sep_id = token_null;
                self.special_pad_id = token_null;
                self.special_mask_id = token_null;
                self.linefeed_id = token_null;

                if (try ml.getKey(u32, .vocab_size, false, exc)) |n_tokens| {
                    impl.logWarn("%s: adding %u dummy tokens\n", .{ "load", n_tokens });
                    self.id_to_token.ensureTotalCapacity(allocator, n_tokens) catch oom();
                    for (0..n_tokens) |_| {
                        const off: u32 = @intCast(self.text_buf.items.len);
                        self.text_buf.append(allocator, 0) catch oom();
                        self.id_to_token.append(allocator, .{ .off = off, .len = 0, .score = 0, .attr = attr_undefined }) catch oom();
                    }
                }
                return;
            }

            if (eql(tm, "llama")) {
                self.type = type_spm;

                self.special_bos_id = 1;
                self.special_eos_id = 2;
                self.special_unk_id = 0;
                self.special_sep_id = token_null;
                self.special_pad_id = token_null;
                self.special_mask_id = token_null;
            } else if (eql(tm, "bert")) {
                self.type = type_wpm;

                self.special_bos_id = 101;
                self.special_eos_id = token_null;
                self.special_unk_id = 100;
                self.special_sep_id = 102;
                self.special_pad_id = 0;
                self.special_mask_id = 103;

                self.add_sep = true;
            } else if (eql(tm, "gpt2") or eql(tm, "hybriddna") or eql(tm, "whitespace")) {
                self.type = type_bpe;

                const merges_keyidx = ml.find(.tokenizer_merges);
                const is_kimi_k2 = eql(self.tokenizer_pre.items, "kimi-k2");

                if (merges_keyidx == -1) {
                    if (!is_kimi_k2) return exc.throw(.runtime_error, "cannot find tokenizer merges in model file\n", .{});
                    impl.logInfo("%s: Kimi-K2 tokenizer detected, skipping BPE merges\n", .{"load"});
                } else {
                    try self.readMerges(ml, merges_keyidx, exc);
                }

                self.special_bos_id = 11;
                self.special_eos_id = 11;
                self.special_unk_id = token_null;
                self.special_sep_id = token_null;
                self.special_pad_id = token_null;
                self.special_mask_id = token_null;
            } else if (eql(tm, "t5")) {
                self.type = type_ugm;

                self.special_bos_id = token_null;
                self.special_eos_id = 1;
                self.special_unk_id = 2;
                self.special_sep_id = token_null;
                self.special_pad_id = 0;
                self.special_mask_id = token_null;

                const idx = ml.find(.tokenizer_precompiled_charsmap);
                if (idx != -1) {
                    if (c.gguf_get_kv_type(ctx, idx) != c.GGUF_TYPE_ARRAY) {
                        const kn = ml.name(.tokenizer_precompiled_charsmap);
                        return exc.throw(.runtime_error, "invalid gguf type for {s}", .{kn.slice()});
                    }
                    const pc_type = c.gguf_get_arr_type(ctx, idx);
                    if (pc_type != c.GGUF_TYPE_INT8 and pc_type != c.GGUF_TYPE_UINT8) {
                        const kn = ml.name(.tokenizer_precompiled_charsmap);
                        return exc.throw(.runtime_error, "invalid gguf type for {s}", .{kn.slice()});
                    }
                    const n = c.gguf_get_arr_n(ctx, idx);
                    const pc: [*]const u8 = @ptrCast(c.gguf_get_arr_data(ctx, idx).?);
                    self.precompiled_charsmap.appendSlice(allocator, pc[0..n]) catch oom();
                    if (self.precompiled_charsmap.items.len < 4) {
                        return exc.throw(.runtime_error, "precompiled_charsmap too small for xcda_blob_size header!", .{});
                    }
                    const xcda_blob_size = std.mem.readInt(u32, self.precompiled_charsmap.items[0..4], .little);
                    if (@as(u64, xcda_blob_size) + 4 >= self.precompiled_charsmap.items.len) {
                        return exc.throw(.runtime_error, "Index out of array bounds in precompiled charsmap!", .{});
                    }
                }
            } else if (eql(tm, "rwkv")) {
                self.type = type_rwkv;

                self.special_bos_id = token_null;
                self.special_eos_id = token_null;
                self.special_unk_id = token_null;
                self.special_sep_id = token_null;
                self.special_pad_id = token_null;
            } else if (eql(tm, "plamo2")) {
                self.type = type_plamo2;

                self.special_bos_id = 1;
                self.special_eos_id = 2;
                self.special_unk_id = 0;
                self.special_sep_id = token_null;
                self.special_pad_id = 3;
                self.special_mask_id = token_null;
            } else if (eql(tm, "gemma4")) {
                self.type = type_bpe;

                const merges_keyidx = ml.find(.tokenizer_merges);
                if (merges_keyidx == -1) return exc.throw(.runtime_error, "cannot find tokenizer merges in model file\n", .{});
                try self.readMerges(ml, merges_keyidx, exc);

                self.special_bos_id = token_null;
                self.special_eos_id = token_null;
                self.special_unk_id = token_null;
                self.special_sep_id = token_null;
                self.special_pad_id = token_null;
                self.special_mask_id = token_null;

                self.tokenizer_pre.clearRetainingCapacity();
                self.tokenizer_pre.appendSlice(allocator, "gemma4") catch oom();
            } else {
                return exc.throw(.runtime_error, "unknown tokenizer: '{s}'", .{tm});
            }

            // for now, only BPE models have pre-tokenizers
            if (self.type == type_bpe) {
                self.add_space_prefix = false;
                self.escape_whitespaces = false;
                self.clean_spaces = true;
                try self.setPreType(exc);
            } else if (self.type == type_spm) {
                self.pre_type = .default;
                self.add_space_prefix = true;
                self.clean_spaces = false;
                self.add_bos = true;
                self.add_eos = false;
            } else if (self.type == type_wpm) {
                self.pre_type = .default;
                self.add_space_prefix = false;
                self.clean_spaces = true;
                self.add_bos = true;
                self.add_eos = false;
                self.add_sep = true;
            } else if (self.type == type_ugm) {
                self.pre_type = .default;
                self.add_bos = false;
                self.add_eos = true;
            } else if (self.type == type_rwkv) {
                self.pre_type = .default;
                self.add_space_prefix = false;
                self.clean_spaces = false;
                self.add_bos = false;
                self.add_eos = false;
            } else {
                self.pre_type = .default;
            }

            if (try ml.getKey(bool, .tokenizer_add_prefix, false, exc)) |v| self.add_space_prefix = v;
            if (try ml.getKey(bool, .tokenizer_remove_extra_ws, false, exc)) |v| self.remove_extra_whitespaces = v;
        }

        try self.readTokens(ml, exc);

        // hybriddna: the marker suffix kept k-mer ids distinct in
        // token_to_id; erase it from id_to_token so the k-mers detokenize to
        // the bare DNA sequence.
        if (eql(self.tokenizer_model.items, "hybriddna")) {
            if (self.token_to_id.get("<oov>")) |oov| {
                const marker = vm.bpe.dna_kmer_marker;
                var i: usize = @intCast(oov.* + 1);
                while (i < self.id_to_token.items.len) : (i += 1) {
                    const d = &self.id_to_token.items[i];
                    const t = self.text_buf.items[d.off..][0..d.len];
                    if (t.len > marker.len and std.mem.eql(u8, t[t.len - marker.len ..], marker)) {
                        d.len -= marker.len;
                        self.text_buf.items[d.off + d.len] = 0;
                    }
                }
            }
        }

        try self.initTokenizer(exc);

        // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
        if (self.type == type_spm) {
            var e2: Exception = .{};
            if (self.byteToToken('\n', &e2)) |id| {
                self.linefeed_id = id;
            } else |_| {
                var wbuf: [1025]u8 = undefined;
                const w = std.mem.printSentinel(&wbuf, "{s}", .{e2.what()}, 0) catch unreachable;
                impl.logWarn("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", .{ "load", w.ptr });
                self.linefeed_id = self.special_pad_id;
            }
        } else if (self.type == type_wpm) {
            self.linefeed_id = self.special_pad_id;
        } else if (self.type == type_rwkv) {
            var ids = try self.tokenize("\n", false, false, exc);
            defer ids.deinit(allocator);
            impl.assert(ids.items.len != 0, "llama-vocab.cpp", 2534, "!ids.empty() && \"model vocab missing newline token\"");
            self.linefeed_id = ids.items[0];
        } else {
            var ids = try self.tokenize("\n", false, false, exc);
            defer ids.deinit(allocator);
            if (ids.items.len == 0) {
                impl.logWarn("%s: model vocab missing newline token, using special_pad_id instead\n", .{"load"});
                self.linefeed_id = self.special_pad_id;
            } else {
                self.linefeed_id = ids.items[0];
            }
        }

        try self.loadSpecialTokens(ml, exc);
        self.buildSpecialTokensCache();
        try self.buildPieceCache(exc);
        try self.setPerTokenAttributes(ml, exc);
    }

    fn readMerges(self: *Vocab, ml: *const gguf_meta.Meta, idx: i64, exc: *Exception) Thrown!void {
        const ctx = ml.ctx;
        if (c.gguf_get_kv_type(ctx, idx) != c.GGUF_TYPE_ARRAY or c.gguf_get_arr_type(ctx, idx) != c.GGUF_TYPE_STRING) {
            const kn = ml.name(.tokenizer_merges);
            return exc.throw(.runtime_error, "invalid gguf type for {s}", .{kn.slice()});
        }
        const n_merges: c_int = @truncate(@as(i64, @intCast(c.gguf_get_arr_n(ctx, idx))));
        var i: c_int = 0;
        while (i < n_merges) : (i += 1) {
            const word = gguf_meta.cString(c.gguf_get_arr_str(ctx, idx, @intCast(i)));
            var first: []const u8 = "";
            var second: []const u8 = "";
            // `word.find(' ', 1)`
            if (word.len > 1) {
                if (std.mem.findScalarPos(u8, word, 1, ' ')) |pos| {
                    first = word[0..pos];
                    second = word[pos + 1 ..];
                }
            }
            self.bpe_ranks.emplace(first, second, i);
        }
    }

    /// The pre-tokenizer names: the `if` chain at llama-vocab.cpp:2133.
    fn setPreType(self: *Vocab, exc: *Exception) Thrown!void {
        const p = self.tokenizer_pre.items;
        if (p.len == 0) {
            impl.logWarn("%s: missing pre-tokenizer type, using: 'default'\n", .{"load"});
            impl.logWarn("%s:                                             \n", .{"load"});
            impl.logWarn("%s: ************************************        \n", .{"load"});
            impl.logWarn("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", .{"load"});
            impl.logWarn("%s: CONSIDER REGENERATING THE MODEL             \n", .{"load"});
            impl.logWarn("%s: ************************************        \n", .{"load"});
            impl.logWarn("%s:                                             \n", .{"load"});
            self.pre_type = .default;
        } else if (eql(p, "default")) {
            self.pre_type = .default;
        } else if (eql(p, "minicpm5")) {
            self.pre_type = .minicpm5;
            self.ignore_merges = true;
        } else if (oneOf(p, &.{ "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano" })) {
            self.pre_type = .llama3;
            self.ignore_merges = true;
            self.add_bos = true;
        } else if (eql(p, "deepseek-llm")) {
            self.pre_type = .deepseek_llm;
            self.clean_spaces = false;
        } else if (eql(p, "deepseek-coder")) {
            self.pre_type = .deepseek_coder;
            self.clean_spaces = false;
        } else if (eql(p, "deepseek-v3")) {
            self.pre_type = .deepseek3_llm;
            self.clean_spaces = false;
        } else if (eql(p, "youtu")) {
            self.pre_type = .youtu;
            self.clean_spaces = false;
            self.ignore_merges = true;
        } else if (eql(p, "falcon")) {
            self.pre_type = .falcon;
        } else if (eql(p, "mpt")) {
            self.pre_type = .mpt;
        } else if (eql(p, "starcoder")) {
            self.pre_type = .starcoder;
        } else if (oneOf(p, &.{ "gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "a.x-4.0", "mellum", "modern-bert" })) {
            self.pre_type = .gpt2;
        } else if (eql(p, "jais-2")) {
            self.pre_type = .jais2;
        } else if (oneOf(p, &.{ "gemma4", "granite-embed-multi-311m" })) {
            self.pre_type = .gemma4;
            self.escape_whitespaces = true;
        } else if (eql(p, "sarvam-moe")) {
            self.pre_type = .sarvam_moe;
            self.escape_whitespaces = true;
            self.clean_spaces = false;
        } else if (oneOf(p, &.{ "jina-v1-en", "jina-v2-code", "roberta-bpe" })) {
            self.pre_type = .gpt2;
            self.add_sep = true;
        } else if (eql(p, "whitespace")) {
            self.pre_type = .whitespace;
            self.normalizer_opts.lowercase = false;
        } else if (eql(p, "refact")) {
            self.pre_type = .refact;
        } else if (eql(p, "command-r")) {
            self.pre_type = .command_r;
            self.clean_spaces = false;
        } else if (oneOf(p, &.{ "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2" })) {
            self.pre_type = .qwen2;
            self.clean_spaces = false;
        } else if (eql(p, "qwen35")) {
            self.pre_type = .qwen35;
            self.clean_spaces = false;
        } else if (eql(p, "stablelm2")) {
            self.pre_type = .stablelm2;
        } else if (eql(p, "olmo")) {
            self.pre_type = .olmo;
        } else if (eql(p, "dbrx")) {
            self.pre_type = .dbrx;
        } else if (eql(p, "smaug-bpe")) {
            self.pre_type = .smaug;
        } else if (eql(p, "poro-chat")) {
            self.pre_type = .poro;
            self.clean_spaces = false;
        } else if (oneOf(p, &.{ "glm4", "chatglm-bpe" })) {
            self.pre_type = .chatglm4;
            self.special_bos_id = token_null;
        } else if (eql(p, "viking")) {
            self.pre_type = .viking;
            self.clean_spaces = false;
        } else if (eql(p, "jais")) {
            self.pre_type = .jais;
        } else if (eql(p, "tekken")) {
            self.pre_type = .tekken;
            self.clean_spaces = false;
            self.ignore_merges = true;
            self.add_bos = true;
        } else if (eql(p, "smollm")) {
            self.pre_type = .smollm;
            self.clean_spaces = false;
        } else if (eql(p, "codeshell")) {
            self.pre_type = .codeshell;
        } else if (eql(p, "bloom")) {
            self.pre_type = .bloom;
        } else if (eql(p, "gpt3-finnish")) {
            self.pre_type = .gpt3_finnish;
        } else if (eql(p, "exaone")) {
            self.pre_type = .exaone;
        } else if (eql(p, "exaone4")) {
            self.pre_type = .gpt2;
        } else if (eql(p, "exaone-moe")) {
            self.pre_type = .exaone_moe;
        } else if (eql(p, "chameleon")) {
            self.pre_type = .chameleon;
            self.add_bos = true;
            self.clean_spaces = false;
        } else if (eql(p, "minerva-7b")) {
            self.pre_type = .minerva;
        } else if (eql(p, "megrez")) {
            self.pre_type = .qwen2;
        } else if (oneOf(p, &.{ "gpt-4o", "llama4", "kanana2", "talkie" })) {
            self.pre_type = .gpt4o;
            self.clean_spaces = false;
        } else if (eql(p, "granite-embed-multi-97m")) {
            self.pre_type = .granite_emb_multi;
            self.clean_spaces = false;
            self.ignore_merges = true;
        } else if (oneOf(p, &.{ "tiny_aya", "cohere2moe" })) {
            self.pre_type = .tiny_aya;
            self.clean_spaces = false;
        } else if (eql(p, "superbpe")) {
            self.pre_type = .superbpe;
            self.clean_spaces = false;
        } else if (eql(p, "trillion")) {
            self.pre_type = .trillion;
            self.clean_spaces = false;
        } else if (eql(p, "granite-docling")) {
            self.pre_type = .granite_docling;
            self.clean_spaces = false;
        } else if (oneOf(p, &.{ "bailingmoe", "bailingmoe2", "llada-moe" })) {
            self.pre_type = .bailingmoe;
            self.clean_spaces = false;
        } else if (eql(p, "seed-coder")) {
            self.pre_type = .seed_coder;
            self.clean_spaces = false;
        } else if (eql(p, "hunyuan")) {
            self.pre_type = .hunyuan;
            self.clean_spaces = false;
        } else if (eql(p, "hunyuan-dense")) {
            self.pre_type = .hunyuan_dense;
            self.clean_spaces = false;
        } else if (eql(p, "joyai-llm")) {
            self.pre_type = .joyai_llm;
            self.clean_spaces = false;
        } else if (eql(p, "kimi-k2")) {
            self.pre_type = .kimi_k2;
            self.clean_spaces = false;
        } else if (eql(p, "grok-2")) {
            self.pre_type = .grok_2;
            self.clean_spaces = false;
        } else if (eql(p, "afmoe")) {
            self.pre_type = .afmoe;
            self.clean_spaces = false;
        } else if (eql(p, "laguna")) {
            self.pre_type = .laguna;
            self.clean_spaces = false;
        } else if (eql(p, "minimax-m2")) {
            self.pre_type = .minimax_m2;
            self.clean_spaces = false;
        } else if (eql(p, "solar-open")) {
            self.pre_type = .solar_open;
            self.clean_spaces = false;
        } else if (eql(p, "mellum2")) {
            self.pre_type = .mellum2;
        } else {
            return exc.throw(.runtime_error, "unknown pre-tokenizer type: '{s}'", .{p});
        }
    }

    /// The token list, scores and types: llama-vocab.cpp:2418-2501.
    fn readTokens(self: *Vocab, ml: *const gguf_meta.Meta, exc: *Exception) Thrown!void {
        const ctx = ml.ctx;

        const token_idx = ml.find(.tokenizer_list);
        if (token_idx == -1) return exc.throw(.runtime_error, "cannot find tokenizer vocab in model file\n", .{});
        if (c.gguf_get_kv_type(ctx, token_idx) != c.GGUF_TYPE_ARRAY or c.gguf_get_arr_type(ctx, token_idx) != c.GGUF_TYPE_STRING) {
            const kn = ml.name(.tokenizer_list);
            return exc.throw(.runtime_error, "invalid gguf type for {s}", .{kn.slice()});
        }

        const n_tokens: u32 = @truncate(c.gguf_get_arr_n(ctx, token_idx));

        var scores: ?[*]const f32 = null;
        var iscores: ?[*]const i32 = null;
        const score_idx = ml.find(.tokenizer_scores);
        if (score_idx != -1) {
            const kv_type = c.gguf_get_kv_type(ctx, score_idx);
            const arr_type = if (kv_type == c.GGUF_TYPE_ARRAY) c.gguf_get_arr_type(ctx, score_idx) else c.GGUF_TYPE_COUNT;
            if (arr_type != c.GGUF_TYPE_INT32 and arr_type != c.GGUF_TYPE_FLOAT32) {
                const kn = ml.name(.tokenizer_scores);
                return exc.throw(.runtime_error, "invalid gguf type for {s}", .{kn.slice()});
            }
            const n_scores: u32 = @truncate(c.gguf_get_arr_n(ctx, score_idx));
            if (n_scores < n_tokens) {
                return exc.throw(.runtime_error, "Index out of array bounds for scores ({d} < {d})\n", .{ n_scores, n_tokens });
            }
            if (arr_type == c.GGUF_TYPE_INT32) {
                iscores = @ptrCast(@alignCast(c.gguf_get_arr_data(ctx, score_idx).?));
            } else {
                scores = @ptrCast(@alignCast(c.gguf_get_arr_data(ctx, score_idx).?));
            }
        }

        var toktypes: ?[*]const i32 = null;
        const toktype_idx = ml.find(.tokenizer_token_type);
        if (toktype_idx != -1) {
            if (c.gguf_get_kv_type(ctx, toktype_idx) != c.GGUF_TYPE_ARRAY or c.gguf_get_arr_type(ctx, toktype_idx) != c.GGUF_TYPE_INT32) {
                const kn = ml.name(.tokenizer_token_type);
                return exc.throw(.runtime_error, "invalid gguf type for {s}", .{kn.slice()});
            }
            const n_toktypes: u32 = @truncate(c.gguf_get_arr_n(ctx, toktype_idx));
            if (n_toktypes < n_tokens) {
                return exc.throw(.runtime_error, "Index out of array bounds for toktypes ({d} < {d})\n", .{ n_toktypes, n_tokens });
            }
            toktypes = @ptrCast(@alignCast(c.gguf_get_arr_data(ctx, toktype_idx).?));
        }

        self.id_to_token.ensureTotalCapacity(allocator, n_tokens) catch oom();

        var i: u32 = 0;
        while (i < n_tokens) : (i += 1) {
            var word = gguf_meta.cString(c.gguf_get_arr_str(ctx, token_idx, i));
            var empty_buf: [32]u8 = undefined;
            if (word.len == 0) {
                impl.logWarn("%s: empty token at index %u\n", .{ "load", i });
                word = std.mem.print(&empty_buf, "[EMPTY_{d}]", .{i}) catch unreachable;
            }

            const slot = self.token_to_id.getOrPut(word, 0) catch oom();
            slot.value.* = @intCast(i);
            self.max_token_len = @max(self.max_token_len, @as(i32, @intCast(word.len)));

            const off: u32 = @intCast(self.text_buf.items.len);
            self.text_buf.appendSlice(allocator, word) catch oom();
            self.text_buf.append(allocator, 0) catch oom();

            var d: TokenData = .{ .off = off, .len = @intCast(word.len), .score = 0.0, .attr = attr_normal };
            if (scores) |s| {
                d.score = s[i];
            } else if (iscores) |s| {
                d.score = @floatFromInt(s[i]);
            }
            if (toktypes) |tt| {
                d.attr = switch (tt[i]) {
                    c.LLAMA_TOKEN_TYPE_UNKNOWN => attr_unknown,
                    c.LLAMA_TOKEN_TYPE_UNUSED => attr_unused,
                    c.LLAMA_TOKEN_TYPE_NORMAL => attr_normal,
                    c.LLAMA_TOKEN_TYPE_CONTROL => attr_control,
                    c.LLAMA_TOKEN_TYPE_USER_DEFINED => attr_user_defined,
                    c.LLAMA_TOKEN_TYPE_BYTE => attr_byte,
                    else => attr_undefined,
                };
            }
            self.id_to_token.append(allocator, d) catch oom();
        }
        impl.assert(self.id_to_token.items.len == self.token_to_id.count(), "llama-vocab.cpp", 2501, "id_to_token.size() == token_to_id.size()");
    }

    /// Ports `llama_vocab::impl`'s `init_tokenizer` (llama-vocab.cpp:3175
    /// @c1d0e7a00).
    fn initTokenizer(self: *Vocab, exc: *Exception) Thrown!void {
        impl.logDebug("%s: initializing tokenizer for type %d\n", .{ "init_tokenizer", @as(c_int, @intCast(self.type)) });
        self.tokenizer = switch (self.type) {
            type_spm => .spm,
            type_bpe => .{ .bpe = vm.bpe.Tokenizer.init(self) },
            type_wpm => .wpm,
            type_ugm => .{ .ugm = vm.ugm.Tokenizer.init(self) },
            type_rwkv => .{ .rwkv = vm.rwkv.Tokenizer.init(self) },
            type_plamo2 => .{ .plamo2 = try vm.plamo2.Tokenizer.init(self, exc) },
            else => impl.abort("llama-vocab.cpp", 3198, "unsupported vocab type"),
        };
    }

    fn eogInsert(self: *Vocab, id: Token) void {
        const items = self.special_eog_ids.items;
        var lo: usize = 0;
        var hi: usize = items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (items[mid] < id) lo = mid + 1 else hi = mid;
        }
        if (lo < items.len and items[lo] == id) return;
        self.special_eog_ids.insert(allocator, lo, id) catch oom();
    }

    fn eogErase(self: *Vocab, id: Token) void {
        if (eogIndex(self.special_eog_ids.items, id)) |i| _ = self.special_eog_ids.orderedRemove(i);
    }

    fn controlLooking(self: *Vocab, id: Token, t: []const u8) void {
        const attr = &self.id_to_token.items[@intCast(id)].attr;
        if ((attr.* & attr_control) == 0) {
            warnControlLooking(id, t);
            attr.* = attr.* | attr_control;
        }
    }

    /// Special tokens, EOG detection and their workarounds:
    /// llama-vocab.cpp:2549-2990.
    fn loadSpecialTokens(self: *Vocab, ml: *const gguf_meta.Meta, exc: *Exception) Thrown!void {
        const n_vocab = self.id_to_token.items.len;
        const Special = struct { kv: gguf_meta.Kv, id: *Token };
        const special_token_types = [_]Special{
            .{ .kv = .tokenizer_bos_id, .id = &self.special_bos_id },
            .{ .kv = .tokenizer_eos_id, .id = &self.special_eos_id },
            .{ .kv = .tokenizer_eot_id, .id = &self.special_eot_id },
            .{ .kv = .tokenizer_eom_id, .id = &self.special_eom_id },
            .{ .kv = .tokenizer_unk_id, .id = &self.special_unk_id },
            .{ .kv = .tokenizer_sep_id, .id = &self.special_sep_id },
            .{ .kv = .tokenizer_pad_id, .id = &self.special_pad_id },
            .{ .kv = .tokenizer_mask_id, .id = &self.special_mask_id },
            .{ .kv = .tokenizer_fim_pre_id, .id = &self.special_fim_pre_id },
            .{ .kv = .tokenizer_fim_suf_id, .id = &self.special_fim_suf_id },
            .{ .kv = .tokenizer_fim_mid_id, .id = &self.special_fim_mid_id },
            .{ .kv = .tokenizer_fim_pad_id, .id = &self.special_fim_pad_id },
            .{ .kv = .tokenizer_fim_rep_id, .id = &self.special_fim_rep_id },
            .{ .kv = .tokenizer_fim_sep_id, .id = &self.special_fim_sep_id },
            // deprecated
            .{ .kv = .tokenizer_prefix_id, .id = &self.special_fim_pre_id },
            .{ .kv = .tokenizer_suffix_id, .id = &self.special_fim_suf_id },
            .{ .kv = .tokenizer_middle_id, .id = &self.special_fim_mid_id },
        };

        for (special_token_types) |it| {
            var kb: [arch.max_name + 1]u8 = undefined;
            const key_name = ml.name(it.kv);
            const key = key_name.slice();
            @memcpy(kb[0..key.len], key);
            kb[key.len] = 0;
            const key_z: [*:0]const u8 = @ptrCast(&kb);
            if (it.id.* >= 0 and @as(usize, @intCast(it.id.*)) >= n_vocab) {
                impl.logWarn("%s: default special token '%s' = %d out of vocab range, disabling\n", .{ "load", key_z, it.id.* });
                it.id.* = token_null;
            }
            const new_id = (try ml.getKey(u32, it.kv, false, exc)) orelse continue;
            if (new_id >= n_vocab) {
                impl.logWarn("%s: bad special token: '%s' = %u, using default id %d\n", .{ "load", key_z, new_id, it.id.* });
            } else {
                it.id.* = @intCast(new_id);
            }
        }

        // Handle add_bos, add_eos and add_sep
        {
            if (try ml.getKey(bool, .tokenizer_add_bos, false, exc)) |v| self.add_bos = v;
            if (try ml.getKey(bool, .tokenizer_add_eos, false, exc)) |v| self.add_eos = v;
            if (try ml.getKey(bool, .tokenizer_add_sep, false, exc)) |v| self.add_sep = v;

            // workaround for Gemma 4
            if (self.pre_type == .gemma4 and !self.add_bos) {
                self.add_bos = true;
                impl.logWarn("%s: override '%s' to 'true' for Gemma4\n", .{ "load", @as([*:0]const u8, "tokenizer.ggml.add_bos_token") });
            }
        }

        // BertNormalizer options
        if (try ml.getKey(bool, .tokenizer_normalizer_lowercase, false, exc)) |v| self.normalizer_opts.lowercase = v;
        self.normalizer_opts.strip_accents = self.normalizer_opts.lowercase;
        if (try ml.getKey(bool, .tokenizer_normalizer_strip_accents, false, exc)) |v| self.normalizer_opts.strip_accents = v;

        // suppress tokens
        {
            const ctx = ml.ctx;
            const idx = ml.find(.tokenizer_suppress_tokens);
            if (idx != -1) {
                if (c.gguf_get_kv_type(ctx, idx) != c.GGUF_TYPE_ARRAY or c.gguf_get_arr_type(ctx, idx) != c.GGUF_TYPE_INT32) {
                    const kn = ml.name(.tokenizer_suppress_tokens);
                    return exc.throw(.runtime_error, "invalid gguf type for {s}", .{kn.slice()});
                }
                const n: c_int = @truncate(@as(i64, @intCast(c.gguf_get_arr_n(ctx, idx))));
                if (n > 0) {
                    const data: [*]const i32 = @ptrCast(@alignCast(c.gguf_get_arr_data(ctx, idx).?));
                    for (0..@intCast(n)) |i| {
                        const id = data[i];
                        if (id >= 0 and id < @as(c_int, @intCast(n_vocab))) self.suppress_tokens.append(allocator, id) catch oom();
                    }
                }
            }
        }

        // auto-detect special tokens by text, in token_to_id's order
        {
            var it = self.token_to_id.iterator();
            while (it.next()) |t| {
                const id = t.value.*;
                const k = t.key;
                if (self.special_eot_id == token_null and oneOf(k, &.{
                    "<|eot_id|>",         "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
                    "<|end_of_text|>",    "<EOT>",      "_<EOT>",  "[EOT]",         "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
                    "<end_of_utterance>",
                })) {
                    self.special_eot_id = id;
                    self.controlLooking(id, k);
                }
                if (self.special_eom_id == token_null and eql(k, "<|eom_id|>")) {
                    self.special_eom_id = id;
                    self.controlLooking(id, k);
                }
                if (self.special_fim_pre_id == token_null and oneOf(k, &.{
                    "<|fim_prefix|>", "<fim-prefix>",  "<fim_prefix>",    "<\u{ff5c}fim\u{2581}begin\u{ff5c}>",
                    "<PRE>",          "\u{2581}<PRE>", "<|code_prefix|>", "<|prefix|>",
                })) {
                    self.special_fim_pre_id = id;
                    self.controlLooking(id, k);
                }
                if (self.special_fim_suf_id == token_null and oneOf(k, &.{
                    "<|fim_suffix|>", "<fim-suffix>",  "<fim_suffix>",    "<\u{ff5c}fim\u{2581}hole\u{ff5c}>",
                    "<SUF>",          "\u{2581}<SUF>", "<|code_suffix|>", "<|suffix|>",
                })) {
                    self.special_fim_suf_id = id;
                    self.controlLooking(id, k);
                }
                if (self.special_fim_mid_id == token_null and oneOf(k, &.{
                    "<|fim_middle|>", "<fim-middle>",  "<fim_middle>",    "<\u{ff5c}fim\u{2581}end\u{ff5c}>",
                    "<MID>",          "\u{2581}<MID>", "<|code_middle|>", "<|middle|>",
                })) {
                    self.special_fim_mid_id = id;
                    self.controlLooking(id, k);
                }
                if (self.special_fim_pad_id == token_null and oneOf(k, &.{ "<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]" })) {
                    self.special_fim_pad_id = id;
                    self.controlLooking(id, k);
                }
                if (self.special_fim_rep_id == token_null and oneOf(k, &.{ "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>" })) {
                    self.special_fim_rep_id = id;
                    self.controlLooking(id, k);
                }
                if (self.special_fim_sep_id == token_null and eql(k, "<|file_sep|>")) {
                    self.special_fim_sep_id = id;
                    self.controlLooking(id, k);
                }
            }
        }

        // auto-detect unused tokens: control tokens with the word "unused"
        {
            var n_unused: u32 = 0;
            var it = self.token_to_id.iterator();
            while (it.next()) |t| {
                const attr = &self.id_to_token.items[@intCast(t.value.*)].attr;
                if ((attr.* & attr_control) == 0) continue;
                if ((attr.* & attr_unused) == 0) {
                    // `strstr` on `c_str()`: the key up to its first NUL.
                    const k = t.key[0 .. std.mem.findScalar(u8, t.key, 0) orelse t.key.len];
                    if (std.mem.find(u8, k, "unused") != null) attr.* = attr.* | attr_unused;
                }
                if ((attr.* & attr_unused) != 0) n_unused += 1;
            }
            impl.logInfo("%s: %u unused tokens\n", .{ "load", n_unused });
        }

        // the tokens that end generation
        self.special_eog_ids.clearRetainingCapacity();

        if (self.special_fim_pad_id != token_null) self.eogInsert(self.special_fim_pad_id);
        if (self.special_fim_rep_id != token_null) self.eogInsert(self.special_fim_rep_id);
        if (self.special_fim_sep_id != token_null) self.eogInsert(self.special_fim_sep_id);

        {
            var it = self.token_to_id.iterator();
            while (it.next()) |t| {
                const id = t.value.*;
                const k = t.key;
                if (oneOf(k, &.{
                    "<|eot_id|>",                                      "<|im_end|>",         "<|end|>",       "<|return|>",    "<|call|>",
                    "<|flush|>",                                       "<|calls|>",          "<end_of_turn>", "<|endoftext|>", "</s>",
                    "<|eom_id|>",                                      "<EOT>",              "_<EOT>",        "[EOT]",         "[EOS]",
                    "<|end_of_text|>",                                 "<end_of_utterance>", "<eos>",         "<turn|>",       "<|tool_response>",
                    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>", "[e~[",
                })) {
                    self.eogInsert(id);
                    self.controlLooking(id, k);
                } else {
                    const attr = self.id_to_token.items[@intCast(id)].attr;
                    if ((attr & attr_control) != 0 and (attr & attr_unused) == 0) {
                        if (eogIndex(self.special_eog_ids.items, id) == null) {
                            var tb: [512]u8 = undefined;
                            impl.logDebug("%s: control token: %6d '%s' is not marked as EOG\n", .{ "load", id, cz(&tb, k) });
                        }
                    }
                }
            }
        }

        // quick hack for gpt-oss, always render these tokens
        {
            var it = self.token_to_id.iterator();
            while (it.next()) |t| {
                const k = t.key;
                if (oneOf(k, &.{ "<|channel|>", "<|message|>", "<|start|>", "<|constrain|>" })) {
                    const attr = &self.id_to_token.items[@intCast(t.value.*)].attr;
                    var tb: [512]u8 = undefined;
                    impl.logWarn("%s: setting token '%s' (%d) attribute to USER_DEFINED (%u), old attributes: %u\n", .{
                        "load", cz(&tb, k), t.value.*, @as(c_uint, attr_user_defined), @as(c_uint, attr.*),
                    });
                    attr.* = attr_user_defined;
                }
            }
        }

        // sanity checks
        if (self.special_eos_id != token_null and eogIndex(self.special_eog_ids.items, self.special_eos_id) == null) {
            self.eogInsert(self.special_eos_id);
            impl.logWarn("%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", .{"load"});
        }
        if (self.special_eot_id != token_null and eogIndex(self.special_eog_ids.items, self.special_eot_id) == null) {
            self.eogInsert(self.special_eot_id);
            impl.logWarn("%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", .{"load"});
        }
        if (self.special_eom_id != token_null and eogIndex(self.special_eog_ids.items, self.special_eom_id) == null) {
            self.eogInsert(self.special_eom_id);
            impl.logWarn("%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", .{"load"});
        }

        // o200k_harmony and solar-open: "<|end|>" is not EOG when the
        // tokenizer has "<|return|>" and "<|call|>", or "<|calls|>" and
        // "<|flush|>"
        {
            var has_return = false;
            var has_call = false;
            var has_end = false;
            var has_flush = false;
            var end_id: Token = token_null;

            impl.logInfo("%s: printing all EOG tokens:\n", .{"load"});
            for (self.special_eog_ids.items) |tid| {
                if (tid < 0 or tid >= @as(Token, @intCast(n_vocab))) {
                    impl.logWarn("%s: EOG token id %d is out of range (vocab size %zu), skipping\n", .{ "load", tid, n_vocab });
                    continue;
                }
                const t = self.text(tid);
                impl.logInfo("%s:   - %d ('%s')\n", .{ "load", tid, t.ptr });
                if (eql(t, "<|return|>")) {
                    has_return = true;
                } else if (eql(t, "<|call|>") or eql(t, "<|calls|>")) {
                    has_call = true;
                } else if (eql(t, "<|flush|>")) {
                    has_flush = true;
                } else if (eql(t, "<|end|>")) {
                    has_end = true;
                    end_id = tid;
                }
            }

            if ((has_return and has_call and has_end) or (has_call and has_flush and has_end)) {
                self.eogErase(end_id);
                self.id_to_token.items[@intCast(end_id)].attr = attr_user_defined;
                impl.logWarn("%s: special_eog_ids contains both '<|return|>' and '<|call|>', or '<|calls|>' and '<|flush|>' tokens, removing '<|end|>' token from EOG list\n", .{"load"});
            }
        }

        // gemma4 and paddleocr: "</s>" is not EOG beside "<|tool_response>"
        {
            var has_tool_response = false;
            var has_s = false;
            var s_id: Token = token_null;
            for (self.special_eog_ids.items) |tid| {
                if (tid < 0 or tid >= @as(Token, @intCast(n_vocab))) continue;
                const t = self.text(tid);
                if (eql(t, "<|tool_response>")) {
                    has_tool_response = true;
                } else if (eql(t, "</s>")) {
                    has_s = true;
                    s_id = tid;
                }
            }
            if (has_tool_response and has_s) {
                self.eogErase(s_id);
                self.id_to_token.items[@intCast(s_id)].attr = attr_normal;
                impl.logWarn("%s: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list\n", .{"load"});
            }
        }
    }

    const SpecialCtx = struct { v: *const Vocab };

    fn longerText(ctx: SpecialCtx, a: Token, b: Token) bool {
        return ctx.v.id_to_token.items[@intCast(a)].len > ctx.v.id_to_token.items[@intCast(b)].len;
    }

    /// The special-token cache: llama-vocab.cpp:2993-3007.
    fn buildSpecialTokensCache(self: *Vocab) void {
        const n: Token = @intCast(self.id_to_token.items.len);
        var id: Token = 0;
        while (id < n) : (id += 1) {
            if ((self.id_to_token.items[@intCast(id)].attr & (attr_control | attr_user_defined | attr_unknown)) != 0) {
                self.cache_special_tokens.append(allocator, id) catch oom();
            }
        }
        libcxx.sort(Token, self.cache_special_tokens.items, SpecialCtx{ .v = self }, longerText);
        impl.logInfo("%s: special tokens cache size = %u\n", .{ "load", @as(u32, @intCast(self.cache_special_tokens.items.len)) });
    }

    /// The piece cache: llama-vocab.cpp:3010-3024, through
    /// `token_to_piece_for_cache` (llama-vocab.cpp:3327 @c1d0e7a00). The C++
    /// starts with a 22-byte buffer and retries at the size reported; the
    /// piece is the same either way, so here it is written once at full
    /// size.
    fn buildPieceCache(self: *Vocab, exc: *Exception) Thrown!void {
        const n_tokens = self.id_to_token.items.len;
        var piece_buf: std.ArrayList(u8) = .empty;
        var piece_off: std.ArrayList(u32) = .empty;
        piece_off.append(allocator, 0) catch oom();
        var scratch: std.ArrayList(u8) = .empty;
        defer scratch.deinit(allocator);
        var id: usize = 0;
        while (id < n_tokens) : (id += 1) {
            scratch.clearRetainingCapacity();
            scratch.resize(allocator, 64) catch oom();
            var n = try self.tokenToPiece(@intCast(id), scratch.items, 0, true, exc);
            if (n < 0) {
                scratch.resize(allocator, @intCast(-n)) catch oom();
                const check = try self.tokenToPiece(@intCast(id), scratch.items, 0, true, exc);
                impl.assert(check == -n, "llama-vocab.cpp", 3334, "check == -n_chars");
                n = check;
            }
            piece_buf.appendSlice(allocator, scratch.items[0..@intCast(n)]) catch oom();
            piece_off.append(allocator, @intCast(piece_buf.items.len)) catch oom();
        }
        self.piece_buf = piece_buf;
        self.piece_off = piece_off;
        const mb = @as(f64, @floatFromInt(piece_buf.items.len)) / 1024.0 / 1024.0;
        impl.logInfo("%s: token to piece cache size = %.4f MB\n", .{ "load", mb });
    }

    /// Per-token attributes by model name: llama-vocab.cpp:3030-3093.
    fn setPerTokenAttributes(self: *Vocab, ml: *const gguf_meta.Meta, exc: *Exception) Thrown!void {
        var model_name_buf: std.ArrayList(u8) = .empty;
        defer model_name_buf.deinit(allocator);
        if (try ml.getKey([]const u8, .general_name, false, exc)) |v| model_name_buf.appendSlice(allocator, v) catch oom();
        const tokenizer_pre: []const u8 = (try ml.getKey([]const u8, .tokenizer_pre, false, exc)) orelse "";
        const general_arch: []const u8 = (try ml.getKey([]const u8, .general_architecture, false, exc)) orelse "";

        // model name to lowercase, `std::tolower` in the "C" locale
        for (model_name_buf.items) |*ch| ch.* = std.ascii.toLower(ch.*);
        const model_name = model_name_buf.items;

        if (containsAny(tokenizer_pre, &.{ "jina-v2-de", "jina-v2-es", "jina-v2-code" }) or
            containsAny(general_arch, &.{ "nomic-bert-moe", "jina-bert-v3" }))
        {
            if (self.token_to_id.get("<mask>") == null) {
                impl.logWarn("%s: Mask token is missing in vocab, please reconvert model!\n", .{"load"});
            } else {
                try self.setTokenAttr("<mask>", attr_lstrip, true, exc);
            }
        } else if (containsAny(model_name, &.{ "phi-3", "phi3" })) {
            for (self.cache_special_tokens.items) |id| try self.setTokenIdAttr(id, attr_rstrip, true, exc);
            try self.setTokenAttr("</s>", attr_rstrip, true, exc);
            for ([_][]const u8{ "<unk>", "<s>", "<|endoftext|>" }) |t| try self.setTokenAttr(t, attr_rstrip, false, exc);
        } else if (containsAny(model_name, &.{"modern-bert"})) {
            if (self.token_to_id.get("[MASK]") == null) {
                impl.logWarn("%s: Mask token missing in vocab!\n", .{"load"});
            } else {
                try self.setTokenAttr("[MASK]", attr_lstrip, true, exc);
            }
        }
    }

    /// `_set_tokenid_attr`.
    fn setTokenIdAttr(self: *Vocab, id: Token, attr: Attr, value: bool, exc: *Exception) Thrown!void {
        var current: u32 = (try self.tokenDataAt(id, exc)).attr;
        current = if (value) current | attr else current & ~@as(u32, attr);
        self.id_to_token.items[@intCast(id)].attr = current;
    }

    /// `_set_token_attr`.
    fn setTokenAttr(self: *Vocab, token: []const u8, attr: Attr, value: bool, exc: *Exception) Thrown!void {
        try self.setTokenIdAttr(try self.atKey(token, exc), attr, value, exc);
    }

    // -------------------------------------------------------------------------
    // Tokenizing

    /// Ports `llama_vocab::impl`'s `tokenizer_st_partition` (llama-vocab.cpp:3208
    /// @c1d0e7a00): every special token, longest first, cut out of every
    /// raw-text fragment.
    ///
    /// The C++ edits a `std`'s `forward_list` in place; the effect is the
    /// per-fragment loop below. A match's left remainder (lstripped when the
    /// token says so) and the token replace the fragment, and the search
    /// continues in the right remainder (rstripped likewise), which stays a
    /// fragment of its own if nothing more matches. No empty fragment is
    /// ever kept. `isspace` sees a signed `char`, so only the six ASCII
    /// spaces count.
    fn stPartition(self: *const Vocab, raw_text: []const u8, buffer: *std.ArrayList(Fragment), parse_special: bool) void {
        var next: std.ArrayList(Fragment) = .empty;
        defer next.deinit(allocator);
        for (self.cache_special_tokens.items) |special_id| {
            const d = self.id_to_token.items[@intCast(special_id)];
            const t = self.text_buf.items[d.off..][0..d.len];

            if (!parse_special and (d.attr & (attr_control | attr_unknown)) != 0) continue;

            next.clearRetainingCapacity();
            for (buffer.items) |frag| {
                switch (frag) {
                    .token => next.append(allocator, frag) catch oom(),
                    .raw => |r| {
                        var base_offset = r.offset;
                        var base_length = r.length;
                        while (true) {
                            const match = std.mem.findPos(u8, raw_text[0 .. base_offset + base_length], base_offset, t) orelse {
                                if (base_length > 0) next.append(allocator, .{ .raw = .{ .offset = base_offset, .length = base_length } }) catch oom();
                                break;
                            };

                            if (match > base_offset) {
                                const left_offset = base_offset;
                                var left_length = match - base_offset;
                                if ((d.attr & attr_lstrip) != 0) {
                                    while (left_length > 0 and isSpace(raw_text[left_offset + left_length - 1])) left_length -= 1;
                                }
                                if (left_length > 0) next.append(allocator, .{ .raw = .{ .offset = left_offset, .length = left_length } }) catch oom();
                            }

                            next.append(allocator, .{ .token = special_id }) catch oom();

                            if (match + t.len < base_offset + base_length) {
                                var right_offset = match + t.len;
                                var right_length = base_length - ((match - base_offset) + t.len);
                                if ((d.attr & attr_rstrip) != 0) {
                                    while (right_length > 0 and isSpace(raw_text[right_offset])) {
                                        right_offset += 1;
                                        right_length -= 1;
                                    }
                                }
                                base_offset = right_offset;
                                base_length = right_length;
                            } else {
                                break;
                            }
                        }
                    },
                }
            }
            std.mem.swap(std.ArrayList(Fragment), buffer, &next);
        }
    }

    /// Ports `llama_vocab::impl`'s `tokenize` (llama-vocab.cpp:3371 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `raw_text`: the text.
    /// - `add_special`: whether to add BOS, EOS and SEP as the model asks.
    /// - `parse_special`: whether control and unknown tokens' texts become
    ///   those tokens.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the tokens, owned by the caller, or `error.Thrown`.
    pub fn tokenize(self: *const Vocab, raw_text: []const u8, add_special: bool, parse_special: bool, exc: *Exception) Thrown!std.ArrayList(Token) {
        impl.assert(self.tokenizer != .none, "llama-vocab.cpp", 3375, "tokenizer && \"Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.\"");

        var output: std.ArrayList(Token) = .empty;
        errdefer output.deinit(allocator);
        var fragments: std.ArrayList(Fragment) = .empty;
        defer fragments.deinit(allocator);

        if (raw_text.len != 0) {
            fragments.append(allocator, .{ .raw = .{ .offset = 0, .length = raw_text.len } }) catch oom();
            self.stPartition(raw_text, &fragments, parse_special);
        }

        var text_buf: std.ArrayList(u8) = .empty;
        defer text_buf.deinit(allocator);

        switch (self.type) {
            type_spm => {
                // prefix with space if first token
                var is_prev_special = true;

                if (add_special and self.add_bos) {
                    impl.assert(self.special_bos_id != token_null, "llama-vocab.cpp", 3396, "special_bos_id != LLAMA_TOKEN_NULL");
                    output.append(allocator, self.special_bos_id) catch oom();
                    is_prev_special = true;
                }

                for (fragments.items) |frag| switch (frag) {
                    .raw => |r| {
                        text_buf.clearRetainingCapacity();
                        if (self.add_space_prefix and is_prev_special) text_buf.append(allocator, ' ') catch oom();
                        text_buf.appendSlice(allocator, raw_text[r.offset..][0..r.length]) catch oom();
                        escapeWhitespace(&text_buf);
                        try vm.spm.tokenize(self, text_buf.items, &output, exc);
                        is_prev_special = false;
                    },
                    .token => |tok| {
                        output.append(allocator, tok) catch oom();
                        is_prev_special = true;
                    },
                };

                if (add_special and self.add_bos and output.items.len >= 2 and output.items[1] == self.special_bos_id) {
                    warnDoubleBos("tokenize");
                }

                if (add_special and self.add_eos) {
                    impl.assert(self.special_eos_id != token_null, "llama-vocab.cpp", 3433, "special_eos_id != LLAMA_TOKEN_NULL");
                    output.append(allocator, self.special_eos_id) catch oom();
                }
            },
            type_bpe => {
                var session = vm.bpe.Session.init(self, &self.tokenizer.bpe);
                defer session.deinit();

                if (add_special) session.appendBos(&output);
                for (fragments.items) |frag| switch (frag) {
                    .raw => |r| {
                        text_buf.clearRetainingCapacity();
                        text_buf.appendSlice(allocator, raw_text[r.offset..][0..r.length]) catch oom();
                        if (self.escape_whitespaces) escapeWhitespace(&text_buf);
                        try session.tokenize(text_buf.items, &output, exc);
                    },
                    .token => |tok| output.append(allocator, tok) catch oom(),
                };

                if (add_special) {
                    session.appendEos(&output);
                    session.checkDoubleBosEos(output.items);
                }
            },
            type_wpm => {
                if (add_special) {
                    impl.assert(self.special_bos_id != token_null, "llama-vocab.cpp", 3480, "special_bos_id != LLAMA_TOKEN_NULL");
                    output.append(allocator, self.special_bos_id) catch oom();
                }
                for (fragments.items) |frag| switch (frag) {
                    .raw => |r| try vm.wpm.tokenize(self, raw_text[r.offset..][0..r.length], &output, exc),
                    .token => |tok| output.append(allocator, tok) catch oom(),
                };
                if (add_special) {
                    impl.assert(self.special_sep_id != token_null, "llama-vocab.cpp", 3500, "special_sep_id != LLAMA_TOKEN_NULL");
                    output.append(allocator, self.special_sep_id) catch oom();
                }
            },
            type_ugm => {
                if (add_special and self.add_bos) {
                    impl.assert(self.special_bos_id != token_null, "llama-vocab.cpp", 3507, "special_bos_id != LLAMA_TOKEN_NULL");
                    output.append(allocator, self.special_bos_id) catch oom();
                }
                for (fragments.items) |frag| switch (frag) {
                    .raw => |r| try vm.ugm.tokenize(self, &self.tokenizer.ugm, raw_text[r.offset..][0..r.length], &output, exc),
                    .token => |tok| output.append(allocator, tok) catch oom(),
                };
                if (add_special and self.add_bos and output.items.len >= 2 and output.items[1] == self.special_bos_id) {
                    warnDoubleBos("tokenize");
                }
                if (add_special and self.add_eos) {
                    impl.assert(self.special_eos_id != token_null, "llama-vocab.cpp", 3533, "special_eos_id != LLAMA_TOKEN_NULL");
                    output.append(allocator, self.special_eos_id) catch oom();
                }
            },
            type_rwkv => {
                for (fragments.items) |frag| switch (frag) {
                    .raw => |r| vm.rwkv.tokenize(self, &self.tokenizer.rwkv, raw_text[r.offset..][0..r.length], &output),
                    .token => |tok| output.append(allocator, tok) catch oom(),
                };
            },
            type_plamo2 => {
                for (fragments.items) |frag| switch (frag) {
                    .raw => |r| try self.tokenizer.plamo2.encode(raw_text[r.offset..][0..r.length], &output, exc),
                    .token => |tok| output.append(allocator, tok) catch oom(),
                };
            },
            else => impl.abort("llama-vocab.cpp", 3571, "fatal error"),
        }

        return output;
    }

    /// Ports `llama_vocab`'s `tokenize` (llama-vocab.cpp:4121 @c1d0e7a00), the
    /// C-buffer form.
    ///
    /// Return: the number of tokens written; minus the number needed when
    /// `tokens` is too small; `INT32_MIN` past `int32_t`; or `error.Thrown`.
    pub fn tokenizeInto(self: *const Vocab, text_in: []const u8, tokens: []Token, add_special: bool, parse_special: bool, exc: *Exception) Thrown!i32 {
        var res = try self.tokenize(text_in, add_special, parse_special, exc);
        defer res.deinit(allocator);
        if (res.items.len >= std.math.maxInt(i32)) {
            impl.logError("%s: tokenization result size %zu exceeds int32_t limit\n", .{ "tokenize", res.items.len });
            return std.math.minInt(i32);
        }
        if (tokens.len < res.items.len) return -@as(i32, @intCast(res.items.len));
        @memcpy(tokens[0..res.items.len], res.items);
        return @intCast(res.items.len);
    }

    // -------------------------------------------------------------------------
    // Detokenizing

    /// `_try_copy`: the piece into `buf`, after up to `lstrip` leading
    /// spaces.
    fn tryCopy(piece_in: []const u8, buf: []u8, lstrip: i32) i32 {
        var piece = piece_in;
        if (piece.len >= std.math.maxInt(i32)) impl.abort("llama-vocab.cpp", 3589, "invalid token size: exceeds int32_t limit");
        var i: i32 = 0;
        while (i < lstrip and piece.len != 0 and piece[0] == ' ') : (i += 1) piece = piece[1..];
        if (buf.len < piece.len) return -@as(i32, @intCast(piece.len));
        @memcpy(buf[0..piece.len], piece);
        return @intCast(piece.len);
    }

    /// Ports `llama_vocab::impl`'s `token_to_piece` (llama-vocab.cpp:3577
    /// @c1d0e7a00). Writes no NUL.
    ///
    /// Parameters:
    /// - `token`: the token.
    /// - `buf`: where the piece goes; its length is the C++'s `length`.
    /// - `lstrip`: how many leading spaces to drop.
    /// - `special`: whether control and unknown tokens render.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the bytes written, minus the bytes needed, or `error.Thrown`.
    pub fn tokenToPiece(self: *const Vocab, token: Token, buf: []u8, lstrip: i32, special: bool, exc: *Exception) Thrown!i32 {
        const attr_special = attr_unknown | attr_control;
        const attr = try self.tokenGetAttr(token, exc);
        if (!special and (attr & attr_special) != 0) return 0;

        // if we have a cache - use it
        if (self.piece_off.items.len != 0) {
            if (token < 0 or token >= self.piece_off.items.len - 1) return exc.throw(.out_of_range, "vector", .{});
            const i: usize = @intCast(token);
            return tryCopy(self.piece_buf.items[self.piece_off.items[i]..self.piece_off.items[i + 1]], buf, lstrip);
        }

        if (0 <= token and token < @as(i32, @intCast(self.id_to_token.items.len))) {
            const token_text = self.text(token);
            switch (self.type) {
                type_wpm, type_spm, type_ugm => {
                    if ((attr & (attr_special | attr_user_defined)) != 0) return tryCopy(token_text, buf, lstrip);
                    if ((attr & attr_normal) != 0) {
                        var result: std.ArrayList(u8) = .empty;
                        defer result.deinit(allocator);
                        result.appendSlice(allocator, token_text) catch oom();
                        unescapeWhitespace(&result);
                        return tryCopy(result.items, buf, lstrip);
                    }
                    if ((attr & attr_byte) != 0) {
                        const byte = try self.tokenToByte(token, exc);
                        return tryCopy(&[_]u8{byte}, buf, lstrip);
                    }
                },
                type_bpe => {
                    if ((attr & (attr_special | attr_user_defined)) != 0) return tryCopy(token_text, buf, lstrip);
                    if ((attr & attr_normal) != 0) {
                        var result: std.ArrayList(u8) = .empty;
                        defer result.deinit(allocator);
                        if (self.escape_whitespaces) {
                            result.appendSlice(allocator, token_text) catch oom();
                            unescapeWhitespace(&result);
                        } else {
                            try decodeText(token_text, &result, exc);
                        }
                        return tryCopy(result.items, buf, lstrip);
                    }
                    if ((attr & attr_byte) != 0) {
                        const byte = try self.tokenToByte(token, exc);
                        return tryCopy(&[_]u8{byte}, buf, lstrip);
                    }
                },
                type_rwkv => {
                    var result: std.ArrayList(u8) = .empty;
                    defer result.deinit(allocator);
                    vm.rwkv.unescapeToken(token_text, &result);
                    if (result.items.len > buf.len) return -@as(i32, @intCast(result.items.len));
                    @memcpy(buf[0..result.items.len], result.items);
                    return @intCast(result.items.len);
                },
                type_plamo2 => {
                    if (self.isByte(token)) {
                        if (token_text.len == 6 and std.mem.eql(u8, token_text[0..3], "<0x") and token_text[5] == '>') {
                            if (std.ascii.isHex(token_text[3]) and std.ascii.isHex(token_text[4])) {
                                const hex_val = std.fmt.parseInt(u8, token_text[3..5], 16) catch unreachable;
                                if (buf.len < 1) return -1;
                                buf[0] = hex_val;
                                return 1;
                            }
                        }
                    }
                    return tryCopy(token_text, buf, lstrip);
                },
                else => impl.abort("llama-vocab.cpp", 3690, "fatal error"),
            }
        }
        return 0;
    }

    /// Ports `llama_vocab::impl`'s one-argument `token_to_piece`
    /// (llama-vocab.cpp:3697 @c1d0e7a00): the cached piece.
    ///
    /// Return: the piece, borrowed until `deinit`, or `error.Thrown`.
    pub fn cachedPiece(self: *const Vocab, token: Token, exc: *Exception) Thrown![]const u8 {
        if (token < 0 or self.piece_off.items.len == 0 or token >= self.piece_off.items.len - 1) return exc.throw(.out_of_range, "vector", .{});
        const i: usize = @intCast(token);
        return self.piece_buf.items[self.piece_off.items[i]..self.piece_off.items[i + 1]];
    }

    /// Ports `llama_vocab::impl`'s `detokenize` (llama-vocab.cpp:3701
    /// @c1d0e7a00).
    ///
    /// Parameters:
    /// - `tokens_in`: the tokens.
    /// - `text_out`: the output buffer; its length is `text_len_max`.
    /// - `remove_special`: whether to drop a leading BOS and trailing EOS the
    ///   model adds.
    /// - `unparse_special`: whether control and unknown tokens render.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the bytes written, minus the bytes needed, or `error.Thrown`.
    pub fn detokenize(self: *const Vocab, tokens_in: []const Token, text_out: []u8, remove_special: bool, unparse_special: bool, exc: *Exception) Thrown!i32 {
        if (self.type == type_none) return 0;
        impl.assert(self.tokenizer != .none, "llama-vocab.cpp", 3712, "tokenizer && \"Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.\"");

        const text_len_max: i32 = @intCast(text_out.len);
        var avail: i32 = text_len_max;
        var total: i32 = 0;
        var tokens = tokens_in;

        // remove the leading space
        var remove_space = self.add_space_prefix;

        if (remove_special and self.add_bos) {
            if (tokens.len > 0 and tokens[0] == self.special_bos_id) {
                remove_space = false;
                tokens = tokens[1..];
            }
        }
        if (remove_special and self.add_eos) {
            if (tokens.len > 0 and tokens[tokens.len - 1] == self.special_eos_id) tokens = tokens[0 .. tokens.len - 1];
        }

        var at: usize = 0;
        for (tokens) |tok| {
            impl.assert(avail >= 0, "llama-vocab.cpp", 3735, "avail >= 0");
            const n_chars = try self.tokenToPiece(tok, text_out[at..][0..@intCast(avail)], @intFromBool(remove_space), unparse_special, exc);
            remove_space = false;
            if (n_chars < 0) {
                avail = 0;
                total -= n_chars;
            } else if (n_chars > 0) {
                avail -= n_chars;
                at += @intCast(n_chars);
                total += n_chars;
            }
        }

        if (total > text_len_max) return -total;

        if (self.clean_spaces) {
            const t = text_out;

            // first pass: characters ?!.,
            const total1 = total;
            total = if (total != 0) 1 else 0;
            var i: i32 = 1;
            while (i < total1) : (i += 1) {
                const x = t[@intCast(i)];
                if (t[@intCast(i - 1)] == ' ') {
                    if (x == '?' or x == '!' or x == '.' or x == ',') total -= 1;
                }
                t[@intCast(total)] = x;
                total += 1;
            }

            // second pass: strip single apostrophe between spaces
            const total2 = total;
            total = if (total != 0) 1 else 0;
            i = 1;
            while (i < total2) : (i += 1) {
                const x = t[@intCast(i)];
                if (x == '\'' and i + 1 < total2 and t[@intCast(i - 1)] == ' ' and t[@intCast(i + 1)] == ' ') {
                    total -= 1;
                    i += 1;
                    t[@intCast(i)] = 0;
                }
                t[@intCast(total)] = x;
                total += 1;
            }

            // third pass: apostrophe contractions
            const total3 = total;
            total = if (total != 0) 1 else 0;
            i = 1;
            while (i < total3) : (i += 1) {
                const x = t[@intCast(i)];
                if (t[@intCast(i - 1)] == ' ') {
                    if (x == '\'' and i + 1 < total3) {
                        const x1 = t[@intCast(i + 1)];
                        if (x1 == 't' or x1 == 'd') {
                            // " 't", " 'd": kept
                        } else if (x1 == 's' or x1 == 'm') {
                            total -= 1;
                        } else if (i + 2 < total3) {
                            const x2 = t[@intCast(i + 2)];
                            if (x1 == 'l' and x2 == 'l') {
                                // " 'll": kept
                            } else if ((x1 == 'r' and x2 == 'e') or (x1 == 'v' and x2 == 'e')) {
                                total -= 1;
                            }
                        }
                    }
                }
                t[@intCast(total)] = x;
                total += 1;
            }
        }

        return if (total <= text_len_max) total else -total;
    }

    /// Ports `llama_vocab::impl`'s `print_info` (llama-vocab.cpp:3813 @c1d0e7a00).
    ///
    /// Return: nothing.
    pub fn printInfo(self: *const Vocab) void {
        const f = "print_info";
        impl.logInfo("%s: vocab type            = %s\n", .{ f, self.typeName().ptr });
        impl.logInfo("%s: n_vocab               = %u\n", .{ f, self.nTokens() });
        impl.logInfo("%s: n_merges              = %u\n", .{ f, @as(u32, @intCast(self.bpe_ranks.count())) });

        const fmts = [_][:0]const u8{
            "%s: BOS token             = %d '%s'\n",
            "%s: EOS token             = %d '%s'\n",
            "%s: EOT token             = %d '%s'\n",
            "%s: EOM token             = %d '%s'\n",
            "%s: UNK token             = %d '%s'\n",
            "%s: SEP token             = %d '%s'\n",
            "%s: PAD token             = %d '%s'\n",
            "%s: MASK token            = %d '%s'\n",
            "%s: LF token              = %d '%s'\n",
            "%s: FIM PRE token         = %d '%s'\n",
            "%s: FIM SUF token         = %d '%s'\n",
            "%s: FIM MID token         = %d '%s'\n",
            "%s: FIM PAD token         = %d '%s'\n",
            "%s: FIM REP token         = %d '%s'\n",
            "%s: FIM SEP token         = %d '%s'\n",
        };
        const ids = [_]Token{
            self.special_bos_id,     self.special_eos_id,     self.special_eot_id,     self.special_eom_id,
            self.special_unk_id,     self.special_sep_id,     self.special_pad_id,     self.special_mask_id,
            self.linefeed_id,        self.special_fim_pre_id, self.special_fim_suf_id, self.special_fim_mid_id,
            self.special_fim_pad_id, self.special_fim_rep_id, self.special_fim_sep_id,
        };
        inline for (fmts, 0..) |fmt, k| {
            const id = ids[k];
            if (id != token_null) impl.logInfo(fmt, .{ f, id, self.textAt(id).ptr });
        }
        for (self.special_eog_ids.items) |id| {
            impl.logInfo("%s: EOG token             = %d '%s'\n", .{ f, id, self.textAt(id).ptr });
        }
        impl.logInfo("%s: max token length      = %d\n", .{ f, self.max_token_len });
    }

    /// `id_to_token.at(id).text` where the C++ would terminate on a bad id.
    fn textAt(self: *const Vocab, id: Token) [:0]const u8 {
        if (id < 0 or id >= self.id_to_token.items.len) {
            var e: Exception = .{};
            e.set(.out_of_range, "vector", .{});
            e.terminate("llama-vocab.cpp", 3819);
        }
        return self.text(id);
    }
};

// -----------------------------------------------------------------------------
// Helpers

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn oneOf(s: []const u8, comptime list: []const []const u8) bool {
    inline for (list) |x| {
        if (std.mem.eql(u8, s, x)) return true;
    }
    return false;
}

/// `_contains_any`: some needle occurs in `s`.
fn containsAny(s: []const u8, comptime needles: []const []const u8) bool {
    inline for (needles) |n| {
        if (std.mem.find(u8, s, n) != null) return true;
    }
    return false;
}

fn eogIndex(items: []const Token, id: Token) ?usize {
    var lo: usize = 0;
    var hi: usize = items.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (items[mid] < id) lo = mid + 1 else hi = mid;
    }
    return if (lo < items.len and items[lo] == id) lo else null;
}

/// A slice as a C string for a log line, truncated to fit `buf`; a key
/// with an embedded NUL prints up to it, as its `c_str()` does.
fn cz(buf: []u8, s: []const u8) [*:0]const u8 {
    const n = @min(s.len, buf.len - 1);
    @memcpy(buf[0..n], s[0..n]);
    buf[n] = 0;
    return @ptrCast(buf.ptr);
}

fn warnControlLooking(id: Token, t: []const u8) void {
    var tb: [512]u8 = undefined;
    impl.logWarn("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n", .{ "load", id, cz(&tb, t) });
}

/// The double-BOS warning SPM and UGM share.
pub fn warnDoubleBos(comptime func: [:0]const u8) void {
    impl.logWarn("%s: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?\n", .{func.ptr});
}

/// `isspace` on a signed `char` in the "C" locale.
fn isSpace(ch: u8) bool {
    return ch == ' ' or (ch >= '\t' and ch <= '\r');
}

/// `strtol(s, NULL, 16)` over a short string: leading spaces, a sign, an
/// optional `0x`, then hex digits up to the first that is not one.
fn strtolHex(s: []const u8) i64 {
    var i: usize = 0;
    while (i < s.len and isSpace(s[i])) i += 1;
    var neg = false;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) {
        neg = s[i] == '-';
        i += 1;
    }
    if (i + 2 < s.len + 1 and i + 1 < s.len and s[i] == '0' and (s[i + 1] == 'x' or s[i + 1] == 'X') and i + 2 < s.len and std.ascii.isHex(s[i + 2])) i += 2;
    var v: i64 = 0;
    while (i < s.len) : (i += 1) {
        const d = std.fmt.charToDigit(s[i], 16) catch break;
        v = v * 16 + d;
    }
    return if (neg) -v else v;
}

/// Ports `replace_all` (llama-impl.cpp:69 @c1d0e7a00).
pub fn replaceAll(s: *std.ArrayList(u8), search: []const u8, replace: []const u8) void {
    if (search.len == 0) return;
    if (std.mem.find(u8, s.items, search) == null) return;
    var builder: std.ArrayList(u8) = .empty;
    builder.ensureTotalCapacity(allocator, s.items.len) catch oom();
    var last_pos: usize = 0;
    while (std.mem.findPos(u8, s.items, last_pos, search)) |pos| {
        builder.appendSlice(allocator, s.items[last_pos..pos]) catch oom();
        builder.appendSlice(allocator, replace) catch oom();
        last_pos = pos + search.len;
    }
    builder.appendSlice(allocator, s.items[last_pos..]) catch oom();
    s.deinit(allocator);
    s.* = builder;
}

/// Ports `llama_escape_whitespace` (llama-vocab.cpp:3343 @c1d0e7a00).
pub fn escapeWhitespace(s: *std.ArrayList(u8)) void {
    replaceAll(s, " ", "\xe2\x96\x81");
}

/// Ports `llama_unescape_whitespace` (llama-vocab.cpp:3347 @c1d0e7a00).
pub fn unescapeWhitespace(s: *std.ArrayList(u8)) void {
    replaceAll(s, "\xe2\x96\x81", " ");
}

/// Ports `llama_decode_text` (llama-vocab.cpp:3351 @c1d0e7a00): each code
/// point of a byte-encoded token back to its byte, or an `[UNK_BYTE_0x..]`
/// marker that also repeats the whole text.
fn decodeText(t: []const u8, out: *std.ArrayList(u8), exc: *Exception) Thrown!void {
    const cpts = unicode.cptsFromUtf8(t);
    defer allocator.free(cpts);
    for (cpts) |cpt| {
        var b: [4]u8 = undefined;
        const utf8 = unicode.cptToUtf8(cpt, &b) catch return exc.throw(.invalid_argument, "invalid codepoint", .{});
        if (unicode.utf8ToByte(utf8)) |byte| {
            out.append(allocator, byte) catch oom();
        } else |_| {
            out.appendSlice(allocator, "[UNK_BYTE_0x") catch oom();
            for (utf8) |ch| {
                var hb: [2]u8 = undefined;
                out.appendSlice(allocator, std.mem.print(&hb, "{x:0>2}", .{ch}) catch unreachable) catch oom();
            }
            out.appendSlice(allocator, t) catch oom();
            out.append(allocator, ']') catch oom();
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "replace_all replaces every occurrence, left to right" {
    var s: std.ArrayList(u8) = .empty;
    defer s.deinit(allocator);
    try s.appendSlice(allocator, " a  b ");
    escapeWhitespace(&s);
    try std.testing.expectEqualStrings("\xe2\x96\x81a\xe2\x96\x81\xe2\x96\x81b\xe2\x96\x81", s.items);
    unescapeWhitespace(&s);
    try std.testing.expectEqualStrings(" a  b ", s.items);
}

test "strtol reads a byte token's hex digits" {
    try std.testing.expectEqual(@as(i64, 0x0A), strtolHex("0A"));
    try std.testing.expectEqual(@as(i64, 0xff), strtolHex("ff"));
    try std.testing.expectEqual(@as(i64, 0), strtolHex("0x"));
    try std.testing.expectEqual(@as(i64, 0), strtolHex("zz"));
}
