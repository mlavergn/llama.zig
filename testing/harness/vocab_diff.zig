//! Compares `src/llama/vocab/` with llama.cpp's `llama-vocab.cpp`.
//!
//! **Not a port.** The harness behind `testing/scripts/vocab-diff`. It links the
//! C++ vocabulary, inside `libllama.a`, and the Zig one, through
//! `src/llama/module.zig`, into one process; loads every GGUF file both
//! ways -- the C++ through `llama_model_load_from_file` with `vocab_only`,
//! the Zig through the same metadata -- and compares:
//!
//! - **what loading logged**: every `load:` line and the vocabulary's
//!   `print_info` block, which between them carry the special tokens, the
//!   EOG list, the cache sizes and every warning; or, for a file that must
//!   fail, the error message;
//! - **every scalar** `llama_vocab_*` returns, and every token's text,
//!   score bits, attributes, EOG and control flags;
//! - **every token's piece**, with and without `special`, with and without
//!   `lstrip`, and into a buffer too small for it;
//! - **tokenization** of upstream's vocab test strings and random text --
//!   special-token texts spliced in, spaces around them for the strip
//!   attributes -- under all four `add_special` x `parse_special`, and into
//!   a buffer too small;
//! - **detokenization** of every result and of random id sequences, under
//!   all four `remove_special` x `unparse_special`, with `clean_spaces`
//!   reached wherever the model sets it.
//!
//! The vocabulary types no real file covers -- RWKV, PLaMo-2, PLaMo-3,
//! `hybriddna`, `test`, `whitespace`, `none` -- the pre-tokenizers and
//! normalizer options no real file sets (K2 Horizon, Unigram NFC, Flint,
//! Domyn Small, Nemotron 3.5 Embed, Soprano, the Metaspace prepend scheme,
//! `clean_spaces` as a key, tail="pad_n" and `dna_auto_tags`), and the
//! load-time error paths are synthesized from upstream's files with the gguf
//! writer, into the directory given.
//!
//! Usage: vocab_diff <tmpdir> <file.gguf | corpus.inp>...

const std = @import("std");
const llama = @import("llama");

var checks: usize = 0;
var failures: usize = 0;
var case_failures: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    case_failures += 1;
    if (case_failures <= 6) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

// -----------------------------------------------------------------------------
// Logs

var log_target: ?*std.ArrayList(u8) = null;

fn logCb(_: llama.impl.llama_c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const t = log_target orelse return;
    t.appendSlice(std.heap.c_allocator, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
}

/// The vocabulary's own log: its `load:` lines, then its `print_info`
/// block, one string.
fn vocabLog(log: []const u8, out: *std.ArrayList(u8)) void {
    out.clearRetainingCapacity();
    var in_info = false;
    var it = std.mem.splitScalar(u8, log, '\n');
    while (it.next()) |line| {
        const take = blk: {
            if (std.mem.startsWith(u8, line, "load: ")) break :blk true;
            if (std.mem.startsWith(u8, line, "validate_override: ")) break :blk true;
            if (std.mem.startsWith(u8, line, "print_info: vocab type")) in_info = true;
            if (in_info) {
                if (std.mem.startsWith(u8, line, "print_info: max token length")) {
                    in_info = false;
                    break :blk true;
                }
                break :blk std.mem.startsWith(u8, line, "print_info: ");
            }
            break :blk false;
        };
        if (take) {
            out.appendSlice(std.heap.c_allocator, line) catch unreachable;
            out.append(std.heap.c_allocator, '\n') catch unreachable;
        }
    }
}

// -----------------------------------------------------------------------------
// Cases

const Case = struct {
    name: []const u8,
    path: [:0]const u8,
    overrides: []const llama.impl.llama_c.llama_model_kv_override = &.{},
};

fn kvBool(key: []const u8, v: bool) llama.impl.llama_c.llama_model_kv_override {
    var o = std.mem.zeroes(llama.impl.llama_c.llama_model_kv_override);
    o.tag = llama.impl.llama_c.LLAMA_KV_OVERRIDE_TYPE_BOOL;
    @memcpy(o.key[0..key.len], key);
    o.unnamed_0.val_bool = v;
    return o;
}

fn kvInt(key: []const u8, v: i64) llama.impl.llama_c.llama_model_kv_override {
    var o = std.mem.zeroes(llama.impl.llama_c.llama_model_kv_override);
    o.tag = llama.impl.llama_c.LLAMA_KV_OVERRIDE_TYPE_INT;
    @memcpy(o.key[0..key.len], key);
    o.unnamed_0.val_i64 = v;
    return o;
}

const kv_end = std.mem.zeroes(llama.impl.llama_c.llama_model_kv_override);

/// Every text a token list should be tokenized from: upstream's test
/// strings, random text, and both with special-token texts spliced in.
const Corpus = struct {
    texts: std.ArrayList([]const u8) = .empty,
};

const pool = [_][]const u8{
    // Texts where the overlap cases' equal-length tokens meet.
    "xyzwab",              "zwabxy",         "abxyzw",                                                                          "wabqxyzw",         "QQRRSS",                   "RRSSQQRR",
    // DNA spans whose tail pads to a k-mer the vocabulary has.
    "<dna>ACGTACGT</dna>", "x<dna>acgtacgt", "<dna>gt</dna>tail",
    // PLaMo-2 pieces whose scores tie only after rounding.
                                                                  "pq",               "xpqx",
    // UGM text where a known pair competes with a lone known char plus an
    // unknown one.
                        "ac",
    "zacz",
    // An lstrip token after nothing but spaces.
                   " <mask>",        "  <mask> x",
    // An RWKV escape with a high byte after it.
                                                                         "\\x\xc3\xa9",      "a",                        "Z",
    "q",                   "7",              "0",                                                                               " ",                "  ",                       "   ",
    "\t",                  "\n",             "\n\n",                                                                            "\r\n",             "'",                        "'s",
    "'S",                  "'re",            "'LL",                                                                             "'d",               ".",                        ",",
    "!",                   "?",              "-",                                                                               "_",                "$",                        "+",
    "<",                   ">",              "|",                                                                               "~",                "`",                        "\\",
    "/",                   "@",              "#",                                                                               "(",                ")",                        "[",
    "]",                   "{",              "}",
    "é",
    "ß",
    "İ",
    "Σ",
    "ж",
    "\u{0301}",
    "中",
    "一",
    "龥",
    "가",
    "の",
    "ア",
    "ก",
    "ع",
    "٣",
    "１",
    "！",
    "。",
    "\u{00A0}",            "\u{3000}",       "\u{0085}",
    "😀",
    "€",
    "©",
    "\u{FFFD}",
    "ﬁ",
    "123",                 "4567",           "Hello",                                                                           "world",            "the",                      " the",
    "▁",
    "<0x41>",              "\xff",           "\xc3",                                                                            "\x00",             "ACGT",                     "<dna>",
    "</dna>",              "acgtn",
    // PLaMo-3's pre-segmentation -- `<|plamo:...|>` markers (empty,
    // with whitespace, U+001C, too long), runs of repeats and spaces, the
    // U+EE00 boundary, newline runs it leaves alone.
             "<|plamo:x|>",                                                                     "<|plamo:|>",       "<|plamo:a b|>",            "<|plamo:\x1c|>",
    "<|plamo:",            "|>",             "<|plamo:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|>", "aaaa",             "aaaaaaa",                  "    ",
    "\n\n\n\n\n",          "\u{EE00}",       "\u{FEFF}",
    // NFC -- decomposed and reordered marks, Hangul jamo, singletons.
                                                                           "e\u{0301}",        "A\u{030A}",                "a\u{0323}\u{0302}",
    "a\u{0302}\u{0323}",   "\u{212B}",       "\u{0344}",                                                                        "\u{1100}\u{1161}", "\u{1100}\u{1161}\u{11A8}", "\u{AC00}\u{11A8}",
    "\u{0F73}",
    // K2 Horizon's letter runs and long s.
               "\u{200C}",       "\u{200D}",                                                                        "\u{017F}",         "'\u{017F}",
    // Nemotron 3.5 Embed's zero-width non-joiner after `[` and `<`,
    // Soprano's full lowercase of U+0130 and its whitespace runs, and the
    // spaces clean_spaces eats.
                   "[\u{200C}",
    "<\u{200C}",           "\u{200C}[",      "[\u{200C}\u{200C}",                                                               "\u{0130}x",        "X\u{0130}",                " \t\n ",
    "\u{3000} \u{00A0}",   " ,",             " .",                                                                              " !",               " ?",                       " 's",
    " 'm",                 " 're",           " 've",                                                                            " ' ",              " 't",
};

/// The hybriddna corpus: spaced displays either side of each threshold, runs
/// either side of the auto-tagging's, unmatched and empty tags, whitespace
/// and non-ASCII inside a span.
const dna_corpus = [_][]const u8{
    "ACGTA CGTAC GTACG TACGT ACGTA CGTAC",
    "AC GT AC GT AC GT",
    "AC GT AC GT AC G",
    "A C G T A C",
    "A C G T A C G T A C G T",
    "xACGTA CGTAC GTACG TACGT ACGTA CGTAC",
    "ACGTA CGTAC GTACG TACGT ACGTA CGTACx",
    "ACGTA CGTAC GTACG TACGT ACGTA CGTAC GTACGx",
    "ACGTA\tCGTAC\t  GTACG TACGT ACGTA CGTAC",
    "ACGTAC GTACGT ACGTAC GTACGT ACGTAC GTACGT",
    "1ACGTA CGTAC GTACG TACGT ACGTA CGTAC 2",
    "see ACGTA CGTAC GTACG TACGT ACGTA CGTAC ok",
    "ACGTA",
    "ACGTAC",
    "acgtac",
    "AcGtAc",
    "ACGTACG",
    "ACGTACGTACGTACG",
    "ACGTNACGTNAC",
    "ACGTNACGTNA",
    "ACGTNNACGTNN",
    "ACGTNACGTNACGTN",
    "ACGTNNACGTNACGT",
    "ACGTNNNACGTNACGT",
    "acgtnacgtnac",
    "NNNNNNNNNNNN",
    "RYSWKMBDHVNA",
    "RYSWKM",
    "the ACGTACGT sequence",
    "ACGTACGTxyz",
    "xyzACGTACGT",
    "ACGT ACGTAC, ACGTAC.",
    "<dna>  ACGT \n</dna>",
    "<dna>AC\u{00A0}GT</dna>",
    "<dna>\u{3000}ACGT\u{3000}</dna>",
    "<dna>\u{00E9}CGTAC</dna>",
    "<dna>ACGTA\u{00E9}</dna>",
    "</dna>",
    "ACGT</dna>",
    "ACGT</dna>tail",
    "a</dna><dna>C",
    "<dna>ACGT",
    "<dna>",
    "<dna></dna>",
    "<dna> </dna>",
    "<dna>CT</dna>",
    "<dna>CTNNNN</dna>",
    "<dna>NNNNNN</dna>",
    "<dna>A</dna>",
    "<dna>acgtac</dna>",
    "<dna>ACGTACGTAC</dna>x<dna>G</dna>",
    "ACGTAC<dna>ACGTAC</dna>ACGTAC",
    "ACGTAC</dna>ACGTAC<dna>ACGTAC",
};

/// What a random DNA text is built from.
const dna_pieces = [_][]const u8{ "A", "C", "G", "T", "a", "c", "g", "t", "N", "R", "Y", "n", " ", " ", "\t", "\n", "ACGTA", "CGTAC", "<dna>", "</dna>", "\u{00E9}", "x", "1", "\u{3000}" };

fn randomText(r: std.Random, buf: *std.ArrayList(u8), specials: []const []const u8) []const u8 {
    buf.clearRetainingCapacity();
    const n = r.uintLessThan(usize, 24);
    for (0..n) |_| {
        if (specials.len != 0 and r.uintLessThan(u8, 6) == 0) {
            if (r.boolean()) buf.append(std.heap.c_allocator, ' ') catch unreachable;
            buf.appendSlice(std.heap.c_allocator, specials[r.uintLessThan(usize, specials.len)]) catch unreachable;
            if (r.boolean()) buf.append(std.heap.c_allocator, ' ') catch unreachable;
        } else {
            buf.appendSlice(std.heap.c_allocator, pool[r.uintLessThan(usize, pool.len)]) catch unreachable;
        }
    }
    return buf.items;
}

// -----------------------------------------------------------------------------
// Comparison

fn compareCase(cs: Case, upstream: []const []const u8, r: std.Random) void {
    case_failures = 0;
    var ref_log: std.ArrayList(u8) = .empty;
    defer ref_log.deinit(std.heap.c_allocator);
    var port_log: std.ArrayList(u8) = .empty;
    defer port_log.deinit(std.heap.c_allocator);

    // The overrides as both sides take them: the C++ list ends with an
    // empty key.
    var ovr: std.ArrayList(llama.impl.llama_c.llama_model_kv_override) = .empty;
    defer ovr.deinit(std.heap.c_allocator);
    ovr.appendSlice(std.heap.c_allocator, cs.overrides) catch unreachable;
    ovr.append(std.heap.c_allocator, kv_end) catch unreachable;

    // Reference.
    log_target = &ref_log;
    var mp = llama.impl.llama_c.llama_model_default_params();
    mp.vocab_only = true;
    if (cs.overrides.len != 0) mp.kv_overrides = ovr.items.ptr;
    const model = llama.impl.llama_c.llama_model_load_from_file(cs.path.ptr, mp);
    defer if (model) |m| llama.impl.llama_c.llama_model_free(m);

    // Port.
    log_target = &port_log;
    const gctx = llama.impl.llama_c.gguf_init_from_file(cs.path.ptr, .{ .no_alloc = true, .ctx = null }) orelse {
        log_target = null;
        fail("{s}: gguf_init_from_file failed", .{cs.name});
        return;
    };
    defer llama.impl.llama_c.gguf_free(gctx);
    const arch_k = llama.impl.llama_c.gguf_find_key(gctx, "general.architecture");
    const arch_name = if (arch_k >= 0) llama.gguf_meta.cString(llama.impl.llama_c.gguf_get_val_str(gctx, arch_k)) else "";
    var meta = llama.gguf_meta.Meta.init(gctx, llama.arch.archFromString(arch_name), if (cs.overrides.len != 0) ovr.items.ptr else null);
    defer meta.deinit();
    var vocab = llama.vocab.vocab.Vocab.init() catch unreachable;
    defer vocab.deinit();
    var exc: llama.impl.Exception = .{};
    var loaded = if (vocab.load(&meta, &exc)) |_| true else |_| false;
    // A throw out of `print_info` is caught further out than the vocabulary
    // load's, and says so differently.
    var info_failed = false;
    if (loaded) {
        vocab.printInfo(&exc) catch {
            loaded = false;
            info_failed = true;
        };
    }
    log_target = null;

    checks += 1;
    if (model == null) {
        if (loaded) {
            fail("{s}: the reference failed to load and the port loaded", .{cs.name});
            return;
        }
        var want: std.ArrayList(u8) = .empty;
        defer want.deinit(std.heap.c_allocator);
        want.print(std.heap.c_allocator, "{s}{s}\n", .{ if (info_failed) "error loading model: " else "llama_model_load: error loading model: error loading model vocabulary: ", exc.what() }) catch unreachable;
        if (std.mem.find(u8, ref_log.items, want.items) == null) {
            fail("{s}: error differs: port \"{f}\"", .{ cs.name, std.zig.fmtString(exc.what()) });
        }
        return;
    }
    if (!loaded) {
        fail("{s}: the port failed to load: {s}", .{ cs.name, exc.what() });
        return;
    }

    {
        var a: std.ArrayList(u8) = .empty;
        defer a.deinit(std.heap.c_allocator);
        var b: std.ArrayList(u8) = .empty;
        defer b.deinit(std.heap.c_allocator);
        vocabLog(ref_log.items, &a);
        vocabLog(port_log.items, &b);
        checks += 1;
        if (!std.mem.eql(u8, a.items, b.items)) {
            fail("{s}: load log differs", .{cs.name});
            var ai = std.mem.splitScalar(u8, a.items, '\n');
            var bi = std.mem.splitScalar(u8, b.items, '\n');
            while (true) {
                const la = ai.next();
                const lb = bi.next();
                if (la == null and lb == null) break;
                if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
                    std.debug.print("  ref:  {s}\n  port: {s}\n", .{ la orelse "<end>", lb orelse "<end>" });
                    break;
                }
            }
        }
    }

    const rv = llama.impl.llama_c.llama_model_get_vocab(model).?;
    compareScalars(cs, rv, &vocab);
    if (vocab.type == llama.vocab.vocab.type_none) return;
    comparePieces(cs, rv, &vocab);

    // Special-token texts to splice into random text.
    var specials: std.ArrayList([]const u8) = .empty;
    defer specials.deinit(std.heap.c_allocator);
    for (vocab.cache_special_tokens.items) |id| {
        if (specials.items.len >= 40) break;
        specials.append(std.heap.c_allocator, vocab.text(id)) catch unreachable;
    }

    var rbuf: std.ArrayList(u8) = .empty;
    defer rbuf.deinit(std.heap.c_allocator);
    var texts: std.ArrayList([]const u8) = .empty;
    defer {
        for (texts.items[upstream.len..]) |t| std.heap.c_allocator.free(t);
        texts.deinit(std.heap.c_allocator);
    }
    texts.appendSlice(std.heap.c_allocator, upstream) catch unreachable;
    for (0..400) |_| texts.append(std.heap.c_allocator, std.heap.c_allocator.dupe(u8, randomText(r, &rbuf, specials.items)) catch unreachable) catch unreachable;
    // Chains over four letters, where the many-overlap case's tokens meet
    // at every position.
    for (0..60) |_| {
        rbuf.clearRetainingCapacity();
        for (0..r.uintLessThan(usize, 40) + 4) |_| rbuf.append(std.heap.c_allocator, "klmn"[r.uintLessThan(usize, 4)]) catch unreachable;
        texts.append(std.heap.c_allocator, std.heap.c_allocator.dupe(u8, rbuf.items) catch unreachable) catch unreachable;
    }

    if (std.mem.eql(u8, vocab.tokenizer_model.items, "hybriddna")) {
        for (0..600) |_| {
            rbuf.clearRetainingCapacity();
            for (0..r.uintLessThan(usize, 24) + 1) |_| rbuf.appendSlice(std.heap.c_allocator, dna_pieces[r.uintLessThan(usize, dna_pieces.len)]) catch unreachable;
            texts.append(std.heap.c_allocator, std.heap.c_allocator.dupe(u8, rbuf.items) catch unreachable) catch unreachable;
        }
    }

    for (texts.items) |t| compareText(cs, rv, &vocab, t);

    // Random id sequences through detokenize.
    const n_tokens: i32 = @intCast(vocab.nTokens());
    var ids: [24]llama.vocab.Token = undefined;
    for (0..300) |_| {
        const n = r.uintLessThan(usize, ids.len + 1);
        for (ids[0..n]) |*id| id.* = if (r.uintLessThan(u8, 4) == 0 and vocab.cache_special_tokens.items.len != 0)
            vocab.cache_special_tokens.items[r.uintLessThan(usize, vocab.cache_special_tokens.items.len)]
        else
            r.intRangeLessThan(i32, 0, n_tokens);
        compareDetokenize(cs, rv, &vocab, ids[0..n]);
    }
}

fn compareScalars(cs: Case, rv: *const llama.impl.llama_c.llama_vocab, v: *const llama.vocab.Vocab) void {
    const Pair = struct { name: []const u8, ref: i64, port: i64 };
    const pairs = [_]Pair{
        .{ .name = "n_tokens", .ref = llama.impl.llama_c.llama_vocab_n_tokens(rv), .port = v.nTokens() },
        .{ .name = "type", .ref = llama.impl.llama_c.llama_vocab_type(rv), .port = v.type },
        .{ .name = "bos", .ref = llama.impl.llama_c.llama_vocab_bos(rv), .port = v.special_bos_id },
        .{ .name = "eos", .ref = llama.impl.llama_c.llama_vocab_eos(rv), .port = v.special_eos_id },
        .{ .name = "eot", .ref = llama.impl.llama_c.llama_vocab_eot(rv), .port = v.special_eot_id },
        .{ .name = "sep", .ref = llama.impl.llama_c.llama_vocab_sep(rv), .port = v.special_sep_id },
        .{ .name = "nl", .ref = llama.impl.llama_c.llama_vocab_nl(rv), .port = v.linefeed_id },
        .{ .name = "pad", .ref = llama.impl.llama_c.llama_vocab_pad(rv), .port = v.special_pad_id },
        .{ .name = "mask", .ref = llama.impl.llama_c.llama_vocab_mask(rv), .port = v.special_mask_id },
        .{ .name = "fim_pre", .ref = llama.impl.llama_c.llama_vocab_fim_pre(rv), .port = v.special_fim_pre_id },
        .{ .name = "fim_suf", .ref = llama.impl.llama_c.llama_vocab_fim_suf(rv), .port = v.special_fim_suf_id },
        .{ .name = "fim_mid", .ref = llama.impl.llama_c.llama_vocab_fim_mid(rv), .port = v.special_fim_mid_id },
        .{ .name = "fim_pad", .ref = llama.impl.llama_c.llama_vocab_fim_pad(rv), .port = v.special_fim_pad_id },
        .{ .name = "fim_rep", .ref = llama.impl.llama_c.llama_vocab_fim_rep(rv), .port = v.special_fim_rep_id },
        .{ .name = "fim_sep", .ref = llama.impl.llama_c.llama_vocab_fim_sep(rv), .port = v.special_fim_sep_id },
        .{ .name = "add_bos", .ref = @intFromBool(llama.impl.llama_c.llama_vocab_get_add_bos(rv)), .port = @intFromBool(v.add_bos) },
        .{ .name = "add_eos", .ref = @intFromBool(llama.impl.llama_c.llama_vocab_get_add_eos(rv)), .port = @intFromBool(v.add_eos) },
        .{ .name = "add_sep", .ref = @intFromBool(llama.impl.llama_c.llama_vocab_get_add_sep(rv)), .port = @intFromBool(v.add_sep) },
    };
    for (pairs) |p| {
        checks += 1;
        if (p.ref != p.port) fail("{s}: {s}: ref {d} port {d}", .{ cs.name, p.name, p.ref, p.port });
    }

    var n_sup: i32 = 0;
    const sup = llama.impl.llama_c.llama_vocab_get_suppress_tokens(rv, &n_sup);
    checks += 1;
    if (n_sup != v.suppress_tokens.items.len or (n_sup > 0 and !std.mem.eql(llama.vocab.Token, sup[0..@intCast(n_sup)], v.suppress_tokens.items))) {
        fail("{s}: suppress tokens differ", .{cs.name});
    }

    if (v.type == llama.vocab.vocab.type_none) return;
    const n: i32 = @intCast(v.nTokens());
    var id: i32 = 0;
    while (id < n) : (id += 1) {
        checks += 1;
        const rt = std.mem.span(llama.impl.llama_c.llama_vocab_get_text(rv, id));
        const d = v.id_to_token.items[@intCast(id)];
        const rs: u32 = @bitCast(llama.impl.llama_c.llama_vocab_get_score(rv, id));
        const ra = llama.impl.llama_c.llama_vocab_get_attr(rv, id);
        if (!std.mem.eql(u8, rt, v.text(id)) or rs != @as(u32, @bitCast(d.score)) or ra != d.attr or
            llama.impl.llama_c.llama_vocab_is_eog(rv, id) != v.isEog(id) or llama.impl.llama_c.llama_vocab_is_control(rv, id) != v.isControl(id))
        {
            fail("{s}: token {d} differs: text \"{f}\"/\"{f}\" attr {d}/{d} eog {}/{}", .{
                cs.name, id, std.zig.fmtString(rt), std.zig.fmtString(v.text(id)), ra, d.attr, llama.impl.llama_c.llama_vocab_is_eog(rv, id), v.isEog(id),
            });
        }
    }
    checks += 2;
    if (llama.impl.llama_c.llama_vocab_is_eog(rv, llama.impl.llama_c.LLAMA_TOKEN_NULL) != v.isEog(llama.impl.llama_c.LLAMA_TOKEN_NULL)) fail("{s}: is_eog(NULL)", .{cs.name});
}

fn comparePieces(cs: Case, rv: *const llama.impl.llama_c.llama_vocab, v: *const llama.vocab.Vocab) void {
    const n: i32 = @intCast(v.nTokens());
    var rb: [512]u8 = undefined;
    var zb: [512]u8 = undefined;
    var exc: llama.impl.Exception = .{};
    var id: i32 = 0;
    while (id < n) : (id += 1) {
        for ([_]bool{ false, true }) |special| {
            for ([_]i32{ 0, 1, 3 }) |lstrip| {
                for ([_]usize{ rb.len, 1, 0 }) |len| {
                    checks += 1;
                    const rn = llama.impl.llama_c.llama_token_to_piece(rv, id, &rb, @intCast(len), lstrip, special);
                    const zn = v.tokenToPiece(id, zb[0..len], lstrip, special, &exc) catch exc.terminate("vocab_diff", 0);
                    if (rn != zn or (rn > 0 and !std.mem.eql(u8, rb[0..@intCast(rn)], zb[0..@intCast(zn)]))) {
                        fail("{s}: piece {d} special={} lstrip={d} len={d}: ref {d} port {d}", .{ cs.name, id, special, lstrip, len, rn, zn });
                    }
                }
            }
        }
    }
}

fn compareText(cs: Case, rv: *const llama.impl.llama_c.llama_vocab, v: *const llama.vocab.Vocab, t: []const u8) void {
    var rt: [4096]llama.vocab.Token = undefined;
    var zt: [4096]llama.vocab.Token = undefined;
    var exc: llama.impl.Exception = .{};
    for ([_]bool{ false, true }) |add_special| {
        for ([_]bool{ false, true }) |parse_special| {
            checks += 1;
            const rn = llama.impl.llama_c.llama_tokenize(rv, t.ptr, @intCast(t.len), &rt, rt.len, add_special, parse_special);
            const zn = v.tokenizeInto(t, &zt, add_special, parse_special, &exc) catch exc.terminate("vocab_diff", 0);
            if (rn != zn or (rn > 0 and !std.mem.eql(llama.vocab.Token, rt[0..@intCast(rn)], zt[0..@intCast(zn)]))) {
                fail("{s}: tokenize \"{f}\" add={} parse={}: ref {d} port {d}", .{ cs.name, std.zig.fmtString(t), add_special, parse_special, rn, zn });
                continue;
            }
            if (rn > 1) {
                // Too small a buffer reports the size needed.
                checks += 1;
                const rs = llama.impl.llama_c.llama_tokenize(rv, t.ptr, @intCast(t.len), &rt, 1, add_special, parse_special);
                const zs = v.tokenizeInto(t, zt[0..1], add_special, parse_special, &exc) catch exc.terminate("vocab_diff", 0);
                if (rs != zs) fail("{s}: short tokenize \"{f}\": ref {d} port {d}", .{ cs.name, std.zig.fmtString(t), rs, zs });
            }
            if (rn > 0) compareDetokenize(cs, rv, v, rt[0..@intCast(rn)]);
        }
    }
}

fn compareDetokenize(cs: Case, rv: *const llama.impl.llama_c.llama_vocab, v: *const llama.vocab.Vocab, ids: []const llama.vocab.Token) void {
    var rb: [16384]u8 = undefined;
    var zb: [16384]u8 = undefined;
    var exc: llama.impl.Exception = .{};
    for ([_]bool{ false, true }) |remove_special| {
        for ([_]bool{ false, true }) |unparse_special| {
            for ([_]usize{ rb.len, 3 }) |len| {
                checks += 1;
                const rn = llama.impl.llama_c.llama_detokenize(rv, ids.ptr, @intCast(ids.len), &rb, @intCast(len), remove_special, unparse_special);
                const zn = v.detokenize(ids, zb[0..len], remove_special, unparse_special, &exc) catch exc.terminate("vocab_diff", 0);
                if (rn != zn or (rn > 0 and !std.mem.eql(u8, rb[0..@intCast(rn)], zb[0..@intCast(zn)]))) {
                    fail("{s}: detokenize {d} ids remove={} unparse={} len={d}: ref {d} port {d}", .{ cs.name, ids.len, remove_special, unparse_special, len, rn, zn });
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Synthetic files

fn setStrs(ctx: *llama.impl.llama_c.gguf_context, key: [:0]const u8, strs: []const []const u8) void {
    var ptrs: std.ArrayList([*c]const u8) = .empty;
    defer ptrs.deinit(std.heap.c_allocator);
    var owned: std.ArrayList([:0]u8) = .empty;
    defer {
        for (owned.items) |o| std.heap.c_allocator.free(o);
        owned.deinit(std.heap.c_allocator);
    }
    for (strs) |s| {
        const z = std.heap.c_allocator.dupeSentinel(u8, s, 0) catch unreachable;
        owned.append(std.heap.c_allocator, z) catch unreachable;
        ptrs.append(std.heap.c_allocator, z.ptr) catch unreachable;
    }
    llama.impl.llama_c.gguf_set_arr_str(ctx, key.ptr, @ptrCast(ptrs.items.ptr), ptrs.items.len);
}

fn setI32s(ctx: *llama.impl.llama_c.gguf_context, key: [:0]const u8, v: []const i32) void {
    llama.impl.llama_c.gguf_set_arr_data(ctx, key.ptr, llama.impl.llama_c.GGUF_TYPE_INT32, v.ptr, v.len);
}

fn setF32s(ctx: *llama.impl.llama_c.gguf_context, key: [:0]const u8, v: []const f32) void {
    llama.impl.llama_c.gguf_set_arr_data(ctx, key.ptr, llama.impl.llama_c.GGUF_TYPE_FLOAT32, v.ptr, v.len);
}

/// A copy of `base` with `edit` applied, written to `dir/name.gguf`.
fn derive(dir: []const u8, base: [:0]const u8, name: []const u8, comptime edit: fn (*llama.impl.llama_c.gguf_context) void) [:0]const u8 {
    const ctx = llama.impl.llama_c.gguf_init_from_file(base.ptr, .{ .no_alloc = true, .ctx = null }).?;
    defer llama.impl.llama_c.gguf_free(ctx);
    edit(ctx);
    const path = std.heap.c_allocator.printSentinel("{s}/{s}.gguf", .{ dir, name }, 0) catch unreachable;
    if (!llama.impl.llama_c.gguf_write_to_file(ctx, path.ptr, true)) @panic("cannot write a synthetic file");
    return path;
}

fn findStrs(ctx: *llama.impl.llama_c.gguf_context, key: [:0]const u8, out: *std.ArrayList([]const u8)) void {
    const k = llama.impl.llama_c.gguf_find_key(ctx, key.ptr);
    const n = llama.impl.llama_c.gguf_get_arr_n(ctx, k);
    for (0..n) |i| out.append(std.heap.c_allocator, std.heap.c_allocator.dupe(u8, llama.gguf_meta.cString(llama.impl.llama_c.gguf_get_arr_str(ctx, k, i))) catch unreachable) catch unreachable;
}

const edits = struct {
    fn rwkv(ctx: *llama.impl.llama_c.gguf_context) void {
        var toks: std.ArrayList([]const u8) = .empty;
        for (0..256) |b| toks.append(std.heap.c_allocator, std.heap.c_allocator.print("\\x{x:0>2}", .{b}) catch unreachable) catch unreachable;
        for ([_][]const u8{ "the", " the", "hello", " world", "\\n\\n", "ab\\\\c", "\\t", "  ", "中文", "\\x41BC", "\\x\xc3\xa9", "\\x\xffz" }) |w| toks.append(std.heap.c_allocator, w) catch unreachable;
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "rwkv");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.token_type");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.pre");
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
    }

    fn plamo2(ctx: *llama.impl.llama_c.gguf_context) void {
        var toks: std.ArrayList([]const u8) = .empty;
        var types: std.ArrayList(i32) = .empty;
        var scores: std.ArrayList(f32) = .empty;
        for ([_][]const u8{ "<|plamo:unk|>", "<|plamo:bos|>", "<|plamo:eos|>", "<|plamo:pad|>" }) |w| {
            toks.append(std.heap.c_allocator, w) catch unreachable;
            types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_CONTROL) catch unreachable;
            scores.append(std.heap.c_allocator, 0) catch unreachable;
        }
        for (0..256) |b| {
            toks.append(std.heap.c_allocator, std.heap.c_allocator.print("<0x{X:0>2}>", .{b}) catch unreachable) catch unreachable;
            types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_BYTE) catch unreachable;
            scores.append(std.heap.c_allocator, 0) catch unreachable;
        }
        const words = [_][]const u8{ "the", "he", "e", "hello", "lo", "world", " the", " ", "日本", "本", "語", "日本語", "ab", "abc", "bc", "c", "Hello", "wor", "ld", "\u{FEFF}x" };
        // "pq" costs 20000.3 before rounding and "p" + "q" exactly 20000: a
        // tie once rounded, which the longer piece wins, and not before.
        for ([_][]const u8{ "pq", "p", "q" }, [_]f32{ -2.00003, -1.0, -1.0 }) |w, sc| {
            toks.append(std.heap.c_allocator, w) catch unreachable;
            types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL) catch unreachable;
            scores.append(std.heap.c_allocator, sc) catch unreachable;
        }
        for (words, 0..) |w, i| {
            toks.append(std.heap.c_allocator, w) catch unreachable;
            types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL) catch unreachable;
            scores.append(std.heap.c_allocator, -@as(f32, @floatFromInt(i)) * 0.37 - 1.0) catch unreachable;
        }
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "plamo2");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.pre");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.bos_token_id");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.eos_token_id");
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
        setF32s(ctx, "tokenizer.ggml.scores", scores.items);
    }

    /// Tokens appended with their types and scores, all three arrays kept
    /// the same length.
    fn appendTyped(ctx: *llama.impl.llama_c.gguf_context, words: []const []const u8, ty: i32, score: f32) void {
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        toks.appendSlice(std.heap.c_allocator, words) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        const tk = llama.impl.llama_c.gguf_find_key(ctx, "tokenizer.ggml.token_type");
        var types: std.ArrayList(i32) = .empty;
        const tp: [*]const i32 = @ptrCast(@alignCast(llama.impl.llama_c.gguf_get_arr_data(ctx, tk)));
        types.appendSlice(std.heap.c_allocator, tp[0..llama.impl.llama_c.gguf_get_arr_n(ctx, tk)]) catch unreachable;
        types.appendNTimes(std.heap.c_allocator, ty, words.len) catch unreachable;
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
        const sk = llama.impl.llama_c.gguf_find_key(ctx, "tokenizer.ggml.scores");
        var scores: std.ArrayList(f32) = .empty;
        const sp: [*]const f32 = @ptrCast(@alignCast(llama.impl.llama_c.gguf_get_arr_data(ctx, sk)));
        scores.appendSlice(std.heap.c_allocator, sp[0..llama.impl.llama_c.gguf_get_arr_n(ctx, sk)]) catch unreachable;
        scores.appendNTimes(std.heap.c_allocator, score, words.len) catch unreachable;
        setF32s(ctx, "tokenizer.ggml.scores", scores.items);
    }

    /// The PLaMo-3 tokenizer: PLaMo-2's vocabulary with pieces that
    /// span what the pre-segmentation cuts -- so cutting or not changes the
    /// tokens -- and the PLaMo FIM and file-separator tokens.
    fn plamo3(ctx: *llama.impl.llama_c.gguf_context) void {
        plamo2(ctx);
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "plamo3");
        appendTyped(ctx, &.{ "<|plamo:fim_prefix|>", "<|plamo:fim_suffix|>", "<|plamo:fim_middle|>", "<|plamo:file_separator|>" }, llama.impl.llama_c.LLAMA_TOKEN_TYPE_CONTROL, 0);
        appendTyped(ctx, &.{
            "aa",       "aaa",      "aaaa",      "aaaaa",     "  ",   "   ",      " a", "a ", "x<|",   "|>x", "<|plamo:x|>", "<|plamo:", "x|>", "|>",
            "\u{FEFF}", "\u{EE00}", "\u{EE00}a", "a\u{EE00}", "\n\n", "\n\n\n\n", "<|", ":",  "plamo", "x\n", "\nx",
        }, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL, -1.5);
    }

    /// The Unigram NFC option on a vocabulary with no charsmap, so
    /// nothing but NFC composes: precomposed pieces beside their parts.
    fn ugmNfc(ctx: *llama.impl.llama_c.gguf_context) void {
        ugmScores(ctx);
        appendTyped(ctx, &.{ "\u{00E9}", "e", "\u{0301}", "\u{00C5}", "A", "\u{030A}", "\u{AC00}", "\u{AC01}", "\u{1100}", "\u{1161}", "\u{11A8}", "\u{1EAD}", "\u{0323}", "\u{0302}", "\u{0308}" }, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL, -2.0);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.normalizer.nfc", true);
    }

    /// ... and on a real Unigram vocabulary with its charsmap.
    fn ugmNfcCharsmap(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.normalizer.nfc", true);
    }

    /// The K2 Horizon pre-tokenizer on the GPT-2 vocabulary, with its
    /// end-of-turn token.
    fn preK2Horizon(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "k2-horizon");
        appendNormal(ctx, &.{"<|ifm|im_end|>"});
    }

    fn appendTokens(ctx: *llama.impl.llama_c.gguf_context, extra: []const []const u8) void {
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        for (extra) |e| toks.append(std.heap.c_allocator, e) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.token_type");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.scores");
    }

    fn hybriddna(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "hybriddna");
        appendTokens(ctx, &.{ "<dna>", "</dna>", "<oov>", "AAAAAA\xee\x80\x80", "ACGTAC\xee\x80\x80", "GTAAAA\xee\x80\x80", "CCCCCC\xee\x80\x80" });
    }

    /// The model name alone: no DNA pieces, so it falls back to plain BPE.
    fn hybriddnaBare(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "hybriddna");
    }

    /// Every 6-mer, as a real tail="pad_a" vocabulary has.
    fn dnaKmers(toks: *std.ArrayList([]const u8)) void {
        for (0..4096) |v| {
            var w: [9]u8 = undefined;
            for (w[0..6], 0..) |*ch, i| ch.* = "ACGT"[(v >> @intCast(2 * i)) & 3];
            w[6..9].* = "\xee\x80\x80".*;
            toks.append(std.heap.c_allocator, std.heap.c_allocator.dupe(u8, &w) catch unreachable) catch unreachable;
        }
    }

    fn hybriddnaFull(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "hybriddna");
        var toks: std.ArrayList([]const u8) = .empty;
        toks.appendSlice(std.heap.c_allocator, &.{ "<dna>", "</dna>", "<oov>" }) catch unreachable;
        dnaKmers(&toks);
        appendTokens(ctx, toks.items);
    }

    /// ... and tail="pad_n": every shorter k-mer padded with N, the k-mer of
    /// N alone, which has no base to strip back to, and a token past the
    /// k-mers ending in N whose text is not DNA.
    fn hybriddnaPadN(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "hybriddna");
        var toks: std.ArrayList([]const u8) = .empty;
        toks.appendSlice(std.heap.c_allocator, &.{ "<dna>", "</dna>", "<oov>" }) catch unreachable;
        dnaKmers(&toks);
        for (1..6) |len| {
            for (0..@as(usize, 1) << @intCast(2 * len)) |v| {
                var w: [9]u8 = @splat('N');
                for (w[0..len], 0..) |*ch, i| ch.* = "ACGT"[(v >> @intCast(2 * i)) & 3];
                w[6..9].* = "\xee\x80\x80".*;
                toks.append(std.heap.c_allocator, std.heap.c_allocator.dupe(u8, &w) catch unreachable) catch unreachable;
            }
        }
        toks.appendSlice(std.heap.c_allocator, &.{ "NNNNNN\xee\x80\x80", "xNNN\xee\x80\x80", "\xee\x80\x80" }) catch unreachable;
        appendTokens(ctx, toks.items);
    }

    fn hybriddnaAuto(ctx: *llama.impl.llama_c.gguf_context) void {
        hybriddnaFull(ctx);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.dna_auto_tags", true);
    }

    fn hybriddnaPadNAuto(ctx: *llama.impl.llama_c.gguf_context) void {
        hybriddnaPadN(ctx);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.dna_auto_tags", true);
    }

    fn hybriddnaAutoOff(ctx: *llama.impl.llama_c.gguf_context) void {
        hybriddnaFull(ctx);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.dna_auto_tags", false);
    }

    fn dnaAutoWrongType(ctx: *llama.impl.llama_c.gguf_context) void {
        hybriddna(ctx);
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.dna_auto_tags", 1);
    }

    /// One pre-tokenizer's name on the GPT-2 vocabulary: its flags --
    /// `clean_spaces`, which only detokenizing shows, `add_bos`,
    /// `ignore_merges` -- and its split patterns.
    fn pre(comptime name: [:0]const u8) fn (*llama.impl.llama_c.gguf_context) void {
        return struct {
            fn f(ctx: *llama.impl.llama_c.gguf_context) void {
                llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", name);
            }
        }.f;
    }

    /// ... and with whole-word tokens no merge reaches, which only
    /// `ignore_merges` emits.
    fn preWhole(comptime name: [:0]const u8) fn (*llama.impl.llama_c.gguf_context) void {
        return struct {
            fn f(ctx: *llama.impl.llama_c.gguf_context) void {
                llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", name);
                appendNormal(ctx, &.{ "zqzq", "\u{120}zqzq" });
            }
        }.f;
    }

    /// Flint's turn tokens end generation, and only under its own
    /// pre-tokenizer.
    fn preFlintTurns(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "flint");
        appendNormal(ctx, &.{ "<|user|>", "<|observation|>" });
    }

    fn flintTurnsOtherPre(ctx: *llama.impl.llama_c.gguf_context) void {
        appendNormal(ctx, &.{ "<|user|>", "<|observation|>" });
    }

    /// Domyn Small is Sarvam's SPM-style BPE, on a vocabulary whose
    /// tokens carry U+2581.
    fn preDomyn(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "domyn-small");
    }

    /// Soprano lowercases and folds whitespace, keeps raw UTF-8,
    /// emits UNK for what is not a token, and its EOS always ends generation.
    /// A real UNK, because a text with no vocabulary token puts its id in the
    /// output.
    fn preSoprano(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "soprano");
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.unknown_token_id", 0);
        appendNormal(ctx, &.{ " ", "i\xcc\x87", "\xcf\x83", "zq zq", "\xe2\x96\x81x" });
    }

    /// ... with an EOS out of range, so the id it inserts is the null token.
    fn preSopranoNoEos(ctx: *llama.impl.llama_c.gguf_context) void {
        tinyBpe(ctx);
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "soprano");
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.unknown_token_id", 0);
    }

    /// The end-of-turn tokens: muse-spark's ...
    fn eotMuse(ctx: *llama.impl.llama_c.gguf_context) void {
        appendNormal(ctx, &.{"<|eot|>"});
    }

    /// ... and beam's tool-call marker, which makes `</s>` a normal token
    /// from outside the EOG list.
    fn beamMarker(ctx: *llama.impl.llama_c.gguf_context) void {
        appendNormal(ctx, &.{ "</s>", "<rfl:parameter name=\"" });
    }

    fn sNoMarker(ctx: *llama.impl.llama_c.gguf_context) void {
        appendNormal(ctx, &.{"</s>"});
    }

    /// `tokenizer.ggml.clean_spaces`, which overrides whatever the
    /// pre-tokenizer set.
    fn cleanOn(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.clean_spaces", true);
    }

    fn cleanOff(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.clean_spaces", false);
    }

    fn cleanOnPre(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "qwen2");
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.clean_spaces", true);
    }

    fn cleanWrongType(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.clean_spaces", 1);
    }

    /// The Metaspace prepend scheme.
    fn scheme(comptime name: [:0]const u8) fn (*llama.impl.llama_c.gguf_context) void {
        return struct {
            fn f(ctx: *llama.impl.llama_c.gguf_context) void {
                llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.metaspace.prepend_scheme", name);
            }
        }.f;
    }

    /// "first" where it matters: tokens carry U+2581 and a BOS is added, so
    /// a skipped BOS must not take the space off the first rendered piece.
    fn schemeFirstBos(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.metaspace.prepend_scheme", "first");
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", true);
    }

    fn schemeFirstMmbert(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "mmbert");
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.metaspace.prepend_scheme", "first");
    }

    fn schemeWrongType(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.metaspace.prepend_scheme", 1);
    }

    /// Every text `eot` auto-detection takes, among tokens enough to fill
    /// `n`. Which one becomes EOT is the first in `token_to_id`'s order, and
    /// `reserve` sets that order: a power of two stays one, where
    /// growing only ever reached primes.
    fn eotOrder(comptime n: usize) fn (*llama.impl.llama_c.gguf_context) void {
        return struct {
            fn f(ctx: *llama.impl.llama_c.gguf_context) void {
                var toks: std.ArrayList([]const u8) = .empty;
                toks.appendSlice(std.heap.c_allocator, &.{
                    "<|eot_id|>",     "<|im_end|>",   "<|end|>",      "<end_of_turn>", "<|endoftext|>",   "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]", "<end_of_utterance>",
                    "<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<PRE>",         "<|code_prefix|>", "<|prefix|>",
                }) catch unreachable;
                while (toks.items.len < n) toks.append(std.heap.c_allocator, std.heap.c_allocator.print("tok{d}", .{toks.items.len}) catch unreachable) catch unreachable;
                setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
                _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.token_type");
                _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.bos_token_id");
                _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.eos_token_id");
                _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.unknown_token_id");
            }
        }.f;
    }

    /// User-defined tokens of equal length that overlap in text, so which
    /// one the special-token cache tries first decides the result.
    fn overlap(ctx: *llama.impl.llama_c.gguf_context) void {
        const extra = [_][]const u8{ "xyzw", "zwab", "abxy", "wabq", "QQRR", "RRSS" };
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        const k = llama.impl.llama_c.gguf_find_key(ctx, "tokenizer.ggml.token_type");
        var types: std.ArrayList(i32) = .empty;
        if (k >= 0) {
            const tt: [*]const i32 = @ptrCast(@alignCast(llama.impl.llama_c.gguf_get_arr_data(ctx, k).?));
            types.appendSlice(std.heap.c_allocator, tt[0..toks.items.len]) catch unreachable;
        } else {
            types.appendNTimes(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL, toks.items.len) catch unreachable;
        }
        for (extra) |e| {
            toks.append(std.heap.c_allocator, e) catch unreachable;
            types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_USER_DEFINED) catch unreachable;
        }
        // The scores, where there are any, must still cover every token.
        const sk = llama.impl.llama_c.gguf_find_key(ctx, "tokenizer.ggml.scores");
        if (sk >= 0) {
            const sc: [*]const f32 = @ptrCast(@alignCast(llama.impl.llama_c.gguf_get_arr_data(ctx, sk).?));
            var scores: std.ArrayList(f32) = .empty;
            scores.appendSlice(std.heap.c_allocator, sc[0 .. toks.items.len - extra.len]) catch unreachable;
            scores.appendNTimes(std.heap.c_allocator, 0, extra.len) catch unreachable;
            setF32s(ctx, "tokenizer.ggml.scores", scores.items);
        }
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
    }

    /// Sixty-four user-defined tokens of length four over four letters, so
    /// the special-token cache holds many equal lengths -- enough for
    /// libc++'s introsort to reorder them, where a short array takes its
    /// stable insertion sort -- and they overlap everywhere in a text over
    /// the same letters.
    fn overlapMany(ctx: *llama.impl.llama_c.gguf_context) void {
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        const base = toks.items.len;
        var prng = std.Random.DefaultPrng.init(4);
        var picked: [256]bool = @splat(false);
        var n: usize = 0;
        while (n < 64) {
            const v = prng.random().int(u8);
            if (picked[v]) continue;
            picked[v] = true;
            n += 1;
            var w: [4]u8 = undefined;
            for (&w, 0..) |*ch, i| ch.* = "klmn"[(v >> @intCast(2 * i)) & 3];
            toks.append(std.heap.c_allocator, std.heap.c_allocator.dupe(u8, &w) catch unreachable) catch unreachable;
        }
        var types: std.ArrayList(i32) = .empty;
        types.appendNTimes(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL, base) catch unreachable;
        types.appendNTimes(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_USER_DEFINED, 64) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
    }

    /// A BPE vocabulary with the `<mask>` lstrip jina's pre-tokenizer name
    /// sets, so a space before it is stripped -- where a lone space would
    /// otherwise become a token.
    fn lstripBpe(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "jina-v2-de");
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        var types: std.ArrayList(i32) = .empty;
        types.appendNTimes(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL, toks.items.len) catch unreachable;
        toks.append(std.heap.c_allocator, "<mask>") catch unreachable;
        types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_CONTROL) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
    }

    /// A small T5 vocabulary whose scores make the unknown-token path
    /// compete: "a" + unknown "c" against the pair "ac".
    fn ugmScores(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "t5");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.pre");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.bos_token_id");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.eos_token_id");
        setStrs(ctx, "tokenizer.ggml.tokens", &.{ "<pad>", "</s>", "<unk>", "\xe2\x96\x81", "a", "ac", "z", "\xe2\x96\x81z", "x" });
        setI32s(ctx, "tokenizer.ggml.token_type", &.{ llama.impl.llama_c.LLAMA_TOKEN_TYPE_CONTROL, llama.impl.llama_c.LLAMA_TOKEN_TYPE_CONTROL, llama.impl.llama_c.LLAMA_TOKEN_TYPE_UNKNOWN, 1, 1, 1, 1, 1, 1 });
        // min score -1 makes UNK -11; "a" at 9.5 then puts a + UNK at -1.5,
        // just below "ac" at -1, and a penalty of 9 would put it above.
        setF32s(ctx, "tokenizer.ggml.scores", &.{ 0, 0, 0, -0.5, 9.5, -1.0, -0.25, -0.75, -0.6 });
    }

    /// A BPE vocabulary of exactly eleven tokens, so the default BOS and
    /// EOS id 11 is one past the end.
    fn tinyBpe(ctx: *llama.impl.llama_c.gguf_context) void {
        setStrs(ctx, "tokenizer.ggml.tokens", &.{ "a", "b", "c", "d", "e", "f", "g", "h", "ab", "\xc4\x8a", "\xc4\xa0" });
        setStrs(ctx, "tokenizer.ggml.merges", &.{"a b"});
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.token_type");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.bos_token_id");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.eos_token_id");
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.unknown_token_id");
    }

    fn whitespace(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "whitespace");
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "whitespace");
    }

    fn none(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "no_vocab");
        llama.impl.llama_c.gguf_set_val_u32(ctx, "gpt2.vocab_size", 77);
    }

    fn spmFlags(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", false);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_eos_token", true);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_space_prefix", false);
    }

    fn ugmFlags(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.remove_extra_whitespaces", true);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_space_prefix", true);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", true);
    }

    fn badModel(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "no-such-tokenizer");
    }

    /// The pre-tokenizers on the GPT-2 vocabulary: their names,
    /// `clean_spaces` and their split patterns.
    fn preHyV4(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "hy_v4");
    }

    fn preSpark25(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "spark2_5");
    }

    /// The pre-tokenizer: Qwen2's pattern without the English
    /// contraction group.
    fn preUfakzeka(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "ufakzeka");
    }

    /// The `LLAMA_VOCAB_TYPE_TEST`: a rolling hash of five-byte
    /// chunks into the vocabulary, and each token's id in hex back out.
    fn testType(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "test");
    }

    /// The pre-tokenizers on the GPT-2 vocabulary: mmBERT's
    /// Metaspace split (escaped spaces, a word per escaped space) and its
    /// `<mask>` attribute, and GLM-5's ChatGLM-4 pattern with
    /// `ignore_merges`.
    fn preMmbert(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "mmbert");
    }

    fn preGlm5(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "glm5");
        appendNormal(ctx, &.{ "zqzq", "\u{120}zqzq" });
    }

    fn preGlm4(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "glm4");
        appendNormal(ctx, &.{ "zqzq", "\u{120}zqzq" });
    }

    /// Whole-word tokens no merge reaches, which only `ignore_merges` emits.
    fn appendNormal(ctx: *llama.impl.llama_c.gguf_context, words: []const []const u8) void {
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        toks.appendSlice(std.heap.c_allocator, words) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        const tk = llama.impl.llama_c.gguf_find_key(ctx, "tokenizer.ggml.token_type");
        if (tk >= 0) {
            var types: std.ArrayList(i32) = .empty;
            const tp: [*]const i32 = @ptrCast(@alignCast(llama.impl.llama_c.gguf_get_arr_data(ctx, tk)));
            types.appendSlice(std.heap.c_allocator, tp[0..llama.impl.llama_c.gguf_get_arr_n(ctx, tk)]) catch unreachable;
            for (words) |_| types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_NORMAL) catch unreachable;
            setI32s(ctx, "tokenizer.ggml.token_type", types.items);
        }
    }

    /// The PLaMo-2 tokenize: BOS and EOS added as asked, and a
    /// `</s>` that is a normal token beside `<|plamo:eos|>`.
    fn plamo2Special(ctx: *llama.impl.llama_c.gguf_context) void {
        plamo2(ctx);
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        toks.append(std.heap.c_allocator, "</s>") catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        const tk = llama.impl.llama_c.gguf_find_key(ctx, "tokenizer.ggml.token_type");
        var types: std.ArrayList(i32) = .empty;
        const tp: [*]const i32 = @ptrCast(@alignCast(llama.impl.llama_c.gguf_get_arr_data(ctx, tk)));
        types.appendSlice(std.heap.c_allocator, tp[0..llama.impl.llama_c.gguf_get_arr_n(ctx, tk)]) catch unreachable;
        types.append(std.heap.c_allocator, llama.impl.llama_c.LLAMA_TOKEN_TYPE_CONTROL) catch unreachable;
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
        const sk = llama.impl.llama_c.gguf_find_key(ctx, "tokenizer.ggml.scores");
        var scores: std.ArrayList(f32) = .empty;
        const sp: [*]const f32 = @ptrCast(@alignCast(llama.impl.llama_c.gguf_get_arr_data(ctx, sk)));
        scores.appendSlice(std.heap.c_allocator, sp[0..llama.impl.llama_c.gguf_get_arr_n(ctx, sk)]) catch unreachable;
        scores.append(std.heap.c_allocator, 0) catch unreachable;
        setF32s(ctx, "tokenizer.ggml.scores", scores.items);
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", 1);
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 2);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", true);
        llama.impl.llama_c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_eos_token", true);
    }

    fn badPre(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "no-such-pre");
    }

    fn noMerges(ctx: *llama.impl.llama_c.gguf_context) void {
        _ = llama.impl.llama_c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
    }

    fn shortScores(ctx: *llama.impl.llama_c.gguf_context) void {
        setF32s(ctx, "tokenizer.ggml.scores", &.{ 1.0, 2.0 });
    }

    fn wrongType(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.add_bos_token", 1);
    }

    fn badSpecial(ctx: *llama.impl.llama_c.gguf_context) void {
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 99999999);
        llama.impl.llama_c.gguf_set_val_u32(ctx, "tokenizer.ggml.eot_token_id", 5);
        setI32s(ctx, "tokenizer.ggml.suppress_tokens", &.{ 3, -1, 99999999, 7 });
    }
};

// -----------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: vocab_diff <tmpdir> <file.gguf | corpus.inp>...\n", .{});
        std.process.exit(2);
    }
    const dir = args[1];

    llama.impl.llama_c.llama_log_set(logCb, null);

    llama.api.impl.llama_log_set(logCb, null);
    llama.impl.llama_c.llama_backend_init();

    var corpus: std.ArrayList([]const u8) = .empty;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var cases: std.ArrayList(Case) = .empty;
    var by_name: std.StringHashMapUnmanaged([:0]const u8) = .empty;

    for (args[2..]) |a| {
        if (std.mem.endsWith(u8, a, ".inp")) {
            const f = std.c.fopen(a.ptr, "rb") orelse return error.CannotOpen;
            var buf: std.ArrayList(u8) = .empty;
            var chunk: [65536]u8 = undefined;
            while (true) {
                const n = std.c.fread(&chunk, 1, chunk.len, f);
                if (n == 0) break;
                try buf.appendSlice(std.heap.c_allocator, chunk[0..n]);
            }
            _ = std.c.fclose(f);
            var it = std.mem.splitSequence(u8, buf.items, "\n__ggml_vocab_test__\n");
            while (it.next()) |s| {
                if ((try seen.getOrPut(std.heap.c_allocator, s)).found_existing) continue;
                try corpus.append(std.heap.c_allocator, s);
            }
        } else {
            const base = std.Io.Dir.path.basename(a);
            try cases.append(std.heap.c_allocator, .{ .name = base, .path = a });
            try by_name.put(std.heap.c_allocator, base, a);
        }
    }

    // Synthetic cases, derived from upstream's files.
    const gpt2 = by_name.get("ggml-vocab-gpt-2.gguf") orelse return error.MissingGpt2Vocab;
    const spm = by_name.get("ggml-vocab-llama-spm.gguf") orelse return error.MissingSpmVocab;
    const ugm = by_name.get("ggml-vocab-nomic-bert-moe.gguf") orelse return error.MissingUgmVocab;
    const gemma4 = by_name.get("ggml-vocab-gemma-4.gguf") orelse return error.MissingGemma4Vocab;
    // words only `ignore_merges` (GLM-4/5) keeps whole
    for ([_][]const u8{ "zqzq zqzq\nzqzq", "a zqzq, zqzqzq" }) |t| try corpus.append(std.heap.c_allocator, t);
    // a leading BOM, which PLaMo-2 drops and PLaMo-3 keeps
    for ([_][]const u8{ "\u{FEFF}hello", "\u{FEFF}", "\u{FEFF}\u{FEFF}aaaa" }) |t| try corpus.append(std.heap.c_allocator, t);
    // PLaMo-3's marker body at its 64-code-point limit and one past it, and
    // newline runs, each between pieces that span the cut
    for ([_][]const u8{
        "x<|plamo:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|>x",
        "x<|plamo:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|>x",
        "x\n\n\n\nx",
        "x\n\n\n\n\nx",
    }) |t| try corpus.append(std.heap.c_allocator, t);
    for (dna_corpus) |t| try corpus.append(std.heap.c_allocator, t);
    // each ambiguity code, and a letter that is not one, joining two runs
    for ("RYSWKMBDHVNrysnvEFIJLOPQUXZ") |ch| {
        try corpus.append(std.heap.c_allocator, try std.heap.c_allocator.print("ACGTACGT{c}ACGTACGT", .{ch}));
    }
    // what `clean_spaces` rewrites, in a sentence
    try corpus.append(std.heap.c_allocator, "a , b . c ! d ? e 's f 'm g 're h 've i ' j 't k 'd l 'll");
    const synthetic = [_]Case{
        .{ .name = "synthetic-rwkv", .path = derive(dir, gpt2, "rwkv", edits.rwkv) },
        .{ .name = "synthetic-plamo2", .path = derive(dir, gpt2, "plamo2", edits.plamo2) },
        .{ .name = "synthetic-hybriddna", .path = derive(dir, gpt2, "hybriddna", edits.hybriddna) },
        .{ .name = "synthetic-whitespace", .path = derive(dir, gpt2, "whitespace", edits.whitespace) },
        .{ .name = "synthetic-overlap-bpe", .path = derive(dir, gpt2, "overlap-bpe", edits.overlap) },
        .{ .name = "synthetic-overlap-spm", .path = derive(dir, spm, "overlap-spm", edits.overlap) },
        .{ .name = "synthetic-overlap-many", .path = derive(dir, gpt2, "overlap-many", edits.overlapMany) },
        .{ .name = "synthetic-lstrip-bpe", .path = derive(dir, gpt2, "lstrip-bpe", edits.lstripBpe) },
        .{ .name = "synthetic-ugm-scores", .path = derive(dir, gpt2, "ugm-scores", edits.ugmScores) },
        .{ .name = "synthetic-tiny-bpe", .path = derive(dir, gpt2, "tiny-bpe", edits.tinyBpe) },
        .{ .name = "synthetic-none", .path = derive(dir, gpt2, "none", edits.none) },
        .{ .name = "synthetic-spm-flags", .path = derive(dir, spm, "spm-flags", edits.spmFlags) },
        .{ .name = "synthetic-ugm-flags", .path = derive(dir, ugm, "ugm-flags", edits.ugmFlags) },
        .{ .name = "synthetic-bad-special", .path = derive(dir, gpt2, "bad-special", edits.badSpecial) },
        .{ .name = "synthetic-pre-hy-v4", .path = derive(dir, gpt2, "pre-hy-v4", edits.preHyV4) },
        .{ .name = "synthetic-pre-spark2-5", .path = derive(dir, gpt2, "pre-spark2-5", edits.preSpark25) },
        .{ .name = "synthetic-pre-ufakzeka", .path = derive(dir, gpt2, "pre-ufakzeka", edits.preUfakzeka) },
        .{ .name = "synthetic-pre-mmbert", .path = derive(dir, gpt2, "pre-mmbert", edits.preMmbert) },
        .{ .name = "synthetic-pre-mmbert-gemma4", .path = derive(dir, gemma4, "pre-mmbert-gemma4", edits.preMmbert) },
        .{ .name = "synthetic-pre-glm5", .path = derive(dir, gpt2, "pre-glm5", edits.preGlm5) },
        .{ .name = "synthetic-pre-glm4", .path = derive(dir, gpt2, "pre-glm4", edits.preGlm4) },
        .{ .name = "synthetic-plamo2-special", .path = derive(dir, gpt2, "plamo2-special", edits.plamo2Special) },
        .{ .name = "synthetic-plamo3", .path = derive(dir, gpt2, "plamo3", edits.plamo3) },
        .{ .name = "synthetic-ugm-nfc", .path = derive(dir, gpt2, "ugm-nfc", edits.ugmNfc) },
        .{ .name = "synthetic-ugm-nfc-charsmap", .path = derive(dir, ugm, "ugm-nfc-charsmap", edits.ugmNfcCharsmap) },
        .{ .name = "synthetic-pre-k2-horizon", .path = derive(dir, gpt2, "pre-k2-horizon", edits.preK2Horizon) },
        .{ .name = "synthetic-test-bpe", .path = derive(dir, gpt2, "test-bpe", edits.testType) },
        .{ .name = "synthetic-test-spm", .path = derive(dir, spm, "test-spm", edits.testType) },
        .{ .name = "synthetic-dna-bare", .path = derive(dir, gpt2, "dna-bare", edits.hybriddnaBare) },
        .{ .name = "synthetic-dna-full", .path = derive(dir, gpt2, "dna-full", edits.hybriddnaFull) },
        .{ .name = "synthetic-dna-padn", .path = derive(dir, gpt2, "dna-padn", edits.hybriddnaPadN) },
        .{ .name = "synthetic-dna-auto", .path = derive(dir, gpt2, "dna-auto", edits.hybriddnaAuto) },
        .{ .name = "synthetic-dna-padn-auto", .path = derive(dir, gpt2, "dna-padn-auto", edits.hybriddnaPadNAuto) },
        .{ .name = "synthetic-dna-auto-off", .path = derive(dir, gpt2, "dna-auto-off", edits.hybriddnaAutoOff) },
        .{ .name = "synthetic-pre-flint", .path = derive(dir, gpt2, "pre-flint", edits.pre("flint")) },
        .{ .name = "synthetic-pre-flint-turns", .path = derive(dir, gpt2, "pre-flint-turns", edits.preFlintTurns) },
        .{ .name = "synthetic-flint-turns-other", .path = derive(dir, gpt2, "flint-turns-other", edits.flintTurnsOtherPre) },
        .{ .name = "synthetic-pre-domyn", .path = derive(dir, gemma4, "pre-domyn", edits.preDomyn) },
        .{ .name = "synthetic-pre-domyn-gpt2", .path = derive(dir, gpt2, "pre-domyn-gpt2", edits.pre("domyn-small")) },
        .{ .name = "synthetic-pre-nemotron", .path = derive(dir, gpt2, "pre-nemotron", edits.preWhole("nemotron-3.5-embed")) },
        .{ .name = "synthetic-pre-soprano", .path = derive(dir, gpt2, "pre-soprano", edits.preSoprano) },
        .{ .name = "synthetic-pre-soprano-no-eos", .path = derive(dir, gpt2, "pre-soprano-no-eos", edits.preSopranoNoEos) },
        .{ .name = "synthetic-pre-tekken", .path = derive(dir, gpt2, "pre-tekken", edits.pre("tekken")) },
        .{ .name = "synthetic-pre-tekken-num3", .path = derive(dir, gpt2, "pre-tekken-num3", edits.pre("tekken-num3")) },
        .{ .name = "synthetic-pre-pixtral", .path = derive(dir, gpt2, "pre-pixtral", edits.preWhole("pixtral")) },
        .{ .name = "synthetic-pre-gpt-4o", .path = derive(dir, gpt2, "pre-gpt-4o", edits.pre("gpt-4o")) },
        .{ .name = "synthetic-pre-minimax-m2", .path = derive(dir, gpt2, "pre-minimax-m2", edits.pre("minimax-m2")) },
        .{ .name = "synthetic-pre-llama3", .path = derive(dir, gpt2, "pre-llama3", edits.pre("llama3")) },
        .{ .name = "synthetic-pre-lfm2", .path = derive(dir, gpt2, "pre-lfm2", edits.preWhole("lfm2")) },
        .{ .name = "synthetic-pre-minicpm5", .path = derive(dir, gpt2, "pre-minicpm5", edits.pre("minicpm5")) },
        .{ .name = "synthetic-pre-gemma4-name", .path = derive(dir, gpt2, "pre-gemma4-name", edits.pre("gemma4")) },
        .{ .name = "synthetic-pre-granite-311m", .path = derive(dir, gpt2, "pre-granite-311m", edits.pre("granite-embed-multi-311m")) },
        .{ .name = "synthetic-pre-dbrx", .path = derive(dir, gpt2, "pre-dbrx", edits.pre("dbrx")) },
        .{ .name = "synthetic-pre-chatglm-bpe", .path = derive(dir, gpt2, "pre-chatglm-bpe", edits.pre("chatglm-bpe")) },
        .{ .name = "synthetic-pre-exaone4", .path = derive(dir, gpt2, "pre-exaone4", edits.pre("exaone4")) },
        .{ .name = "synthetic-pre-mellum2", .path = derive(dir, gpt2, "pre-mellum2", edits.pre("mellum2")) },
        .{ .name = "synthetic-eot-muse", .path = derive(dir, gpt2, "eot-muse", edits.eotMuse) },
        .{ .name = "synthetic-beam-marker", .path = derive(dir, gpt2, "beam-marker", edits.beamMarker) },
        .{ .name = "synthetic-s-no-marker", .path = derive(dir, gpt2, "s-no-marker", edits.sNoMarker) },
        .{ .name = "synthetic-clean-on-spm", .path = derive(dir, spm, "clean-on-spm", edits.cleanOn) },
        .{ .name = "synthetic-clean-off-bpe", .path = derive(dir, gpt2, "clean-off-bpe", edits.cleanOff) },
        .{ .name = "synthetic-clean-on-pre", .path = derive(dir, gpt2, "clean-on-pre", edits.cleanOnPre) },
        .{ .name = "synthetic-scheme-first-gemma4", .path = derive(dir, gemma4, "scheme-first-gemma4", edits.scheme("first")) },
        .{ .name = "synthetic-scheme-first-mmbert", .path = derive(dir, gemma4, "scheme-first-mmbert", edits.schemeFirstMmbert) },
        .{ .name = "synthetic-scheme-first-gpt2", .path = derive(dir, gpt2, "scheme-first-gpt2", edits.scheme("first")) },
        .{ .name = "synthetic-scheme-first-bos", .path = derive(dir, gpt2, "scheme-first-bos", edits.schemeFirstBos) },
        .{ .name = "synthetic-scheme-first-spm", .path = derive(dir, spm, "scheme-first-spm", edits.scheme("first")) },
        .{ .name = "synthetic-scheme-always-spm", .path = derive(dir, spm, "scheme-always-spm", edits.scheme("always")) },
        .{ .name = "synthetic-scheme-never", .path = derive(dir, gemma4, "scheme-never", edits.scheme("never")) },
        .{ .name = "synthetic-eot-order-pow2", .path = derive(dir, gpt2, "eot-order-pow2", edits.eotOrder(1024)) },
        .{ .name = "synthetic-eot-order-pow2-small", .path = derive(dir, gpt2, "eot-order-pow2-small", edits.eotOrder(64)) },
        .{ .name = "synthetic-eot-order", .path = derive(dir, gpt2, "eot-order", edits.eotOrder(1000)) },
        .{ .name = "error-scheme-always-bpe", .path = derive(dir, gpt2, "scheme-always-bpe", edits.scheme("always")) },
        .{ .name = "error-scheme-unknown", .path = derive(dir, gpt2, "scheme-unknown", edits.scheme("sometimes")) },
        .{ .name = "error-scheme-type", .path = derive(dir, gpt2, "scheme-type", edits.schemeWrongType) },
        .{ .name = "error-clean-type", .path = derive(dir, gpt2, "clean-type", edits.cleanWrongType) },
        .{ .name = "error-dna-auto-type", .path = derive(dir, gpt2, "dna-auto-type", edits.dnaAutoWrongType) },
        .{ .name = "error-model", .path = derive(dir, gpt2, "bad-model", edits.badModel) },
        .{ .name = "error-pre", .path = derive(dir, gpt2, "bad-pre", edits.badPre) },
        .{ .name = "error-merges", .path = derive(dir, gpt2, "no-merges", edits.noMerges) },
        .{ .name = "error-scores", .path = derive(dir, spm, "short-scores", edits.shortScores) },
        .{ .name = "error-type", .path = derive(dir, gpt2, "wrong-type", edits.wrongType) },
        .{ .name = "override-bool", .path = gpt2, .overrides = &.{ kvBool("tokenizer.ggml.add_bos_token", true), kvBool("tokenizer.ggml.add_eos_token", true) } },
        .{ .name = "override-mistyped", .path = spm, .overrides = &.{ kvInt("tokenizer.ggml.add_bos_token", 0), kvInt("tokenizer.ggml.eos_token_id", 5) } },
    };
    try cases.appendSlice(std.heap.c_allocator, &synthetic);

    var prng = std.Random.DefaultPrng.init(0x70c);
    for (cases.items) |cs| {
        const before = checks;
        compareCase(cs, corpus.items, prng.random());
        std.debug.print("  {s: <40} {d: >9} checks{s}\n", .{ cs.name, checks - before, if (case_failures != 0) "  FAIL" else "" });
    }

    if (failures == 0) {
        std.debug.print("PASS: {d} vocabulary results identical, Zig vs llama-vocab.cpp, over {d} files\n", .{ checks, cases.items.len });
        return;
    }
    std.debug.print("FAIL: {d} of {d} checks differ\n", .{ failures, checks });
    std.process.exit(1);
}
