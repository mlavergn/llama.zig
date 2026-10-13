//! Compares `src/llama/vocab/` with llama.cpp's `llama-vocab.cpp`.
//!
//! **Not a port.** The harness behind `scripts/vocab-diff`. It links the
//! C++ vocabulary, inside `libllama.a`, and the Zig one, through
//! `src/llama/staged.zig`, into one process; loads every GGUF file both
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
//! The vocabulary types no real file covers -- RWKV, PLaMo-2, `hybriddna`,
//! `whitespace`, `none` -- and the load-time error paths are synthesized
//! from upstream's files with the gguf writer, into the directory given.
//!
//! Usage: vocab_diff <tmpdir> <file.gguf | corpus.inp>...

const std = @import("std");
const staged = @import("staged");

const c = staged.impl.c;
const V = staged.vocab;
const Vocab = V.Vocab;
const Token = V.Token;

const gpa = std.heap.c_allocator;

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

fn logCb(_: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const t = log_target orelse return;
    t.appendSlice(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
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
            out.appendSlice(gpa, line) catch unreachable;
            out.append(gpa, '\n') catch unreachable;
        }
    }
}

// -----------------------------------------------------------------------------
// Cases

const Case = struct {
    name: []const u8,
    path: [:0]const u8,
    overrides: []const c.llama_model_kv_override = &.{},
};

fn kvBool(key: []const u8, v: bool) c.llama_model_kv_override {
    var o = std.mem.zeroes(c.llama_model_kv_override);
    o.tag = c.LLAMA_KV_OVERRIDE_TYPE_BOOL;
    @memcpy(o.key[0..key.len], key);
    o.unnamed_0.val_bool = v;
    return o;
}

fn kvInt(key: []const u8, v: i64) c.llama_model_kv_override {
    var o = std.mem.zeroes(c.llama_model_kv_override);
    o.tag = c.LLAMA_KV_OVERRIDE_TYPE_INT;
    @memcpy(o.key[0..key.len], key);
    o.unnamed_0.val_i64 = v;
    return o;
}

const kv_end = std.mem.zeroes(c.llama_model_kv_override);

/// Every text a token list should be tokenized from: upstream's test
/// strings, random text, and both with special-token texts spliced in.
const Corpus = struct {
    texts: std.ArrayList([]const u8) = .empty,
};

const pool = [_][]const u8{
    // Texts where the overlap cases' equal-length tokens meet.
    "xyzwab",              "zwabxy",         "abxyzw",            "wabqxyzw",    "QQRRSS", "RRSSQQRR",
    // DNA spans whose tail pads to a k-mer the vocabulary has.
    "<dna>ACGTACGT</dna>", "x<dna>acgtacgt", "<dna>gt</dna>tail",
    // PLaMo-2 pieces whose scores tie only after rounding.
    "pq",          "xpqx",
    // UGM text where a known pair competes with a lone known char plus an
    // unknown one.
      "ac",
    "zacz",
    // An lstrip token after nothing but spaces.
                   " <mask>",        "  <mask> x",
    // An RWKV escape with a high byte after it.
           "\\x\xc3\xa9", "a",      "Z",
    "q",                   "7",              "0",                 " ",           "  ",     "   ",
    "\t",                  "\n",             "\n\n",              "\r\n",        "'",      "'s",
    "'S",                  "'re",            "'LL",               "'d",          ".",      ",",
    "!",                   "?",              "-",                 "_",           "$",      "+",
    "<",                   ">",              "|",                 "~",           "`",      "\\",
    "/",                   "@",              "#",                 "(",           ")",      "[",
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
    "123",                 "4567",           "Hello",             "world",       "the",    " the",
    "▁",
    "<0x41>",              "\xff",           "\xc3",              "\x00",        "ACGT",   "<dna>",
    "</dna>",              "acgtn",
};

fn randomText(r: std.Random, buf: *std.ArrayList(u8), specials: []const []const u8) []const u8 {
    buf.clearRetainingCapacity();
    const n = r.uintLessThan(usize, 24);
    for (0..n) |_| {
        if (specials.len != 0 and r.uintLessThan(u8, 6) == 0) {
            if (r.boolean()) buf.append(gpa, ' ') catch unreachable;
            buf.appendSlice(gpa, specials[r.uintLessThan(usize, specials.len)]) catch unreachable;
            if (r.boolean()) buf.append(gpa, ' ') catch unreachable;
        } else {
            buf.appendSlice(gpa, pool[r.uintLessThan(usize, pool.len)]) catch unreachable;
        }
    }
    return buf.items;
}

// -----------------------------------------------------------------------------
// Comparison

fn compareCase(cs: Case, upstream: []const []const u8, r: std.Random) void {
    case_failures = 0;
    var ref_log: std.ArrayList(u8) = .empty;
    defer ref_log.deinit(gpa);
    var port_log: std.ArrayList(u8) = .empty;
    defer port_log.deinit(gpa);

    // The overrides as both sides take them: the C++ list ends with an
    // empty key.
    var ovr: std.ArrayList(c.llama_model_kv_override) = .empty;
    defer ovr.deinit(gpa);
    ovr.appendSlice(gpa, cs.overrides) catch unreachable;
    ovr.append(gpa, kv_end) catch unreachable;

    // Reference.
    log_target = &ref_log;
    var mp = c.llama_model_default_params();
    mp.vocab_only = true;
    if (cs.overrides.len != 0) mp.kv_overrides = ovr.items.ptr;
    const model = c.llama_model_load_from_file(cs.path.ptr, mp);
    defer if (model) |m| c.llama_model_free(m);

    // Port.
    log_target = &port_log;
    const gctx = c.gguf_init_from_file(cs.path.ptr, .{ .no_alloc = true, .ctx = null }) orelse {
        log_target = null;
        fail("{s}: gguf_init_from_file failed", .{cs.name});
        return;
    };
    defer c.gguf_free(gctx);
    const arch_k = c.gguf_find_key(gctx, "general.architecture");
    const arch = if (arch_k >= 0) staged.gguf_meta.cString(c.gguf_get_val_str(gctx, arch_k)) else "(unknown)";
    var meta = staged.gguf_meta.Meta.init(gctx, arch, if (cs.overrides.len != 0) ovr.items.ptr else null);
    defer meta.deinit();
    var vocab = Vocab.init() catch unreachable;
    defer vocab.deinit();
    var exc: staged.impl.Exception = .{};
    const loaded = if (vocab.load(&meta, &exc)) |_| true else |_| false;
    if (loaded) vocab.printInfo();
    log_target = null;

    checks += 1;
    if (model == null) {
        if (loaded) {
            fail("{s}: the reference failed to load and the port loaded", .{cs.name});
            return;
        }
        var want: std.ArrayList(u8) = .empty;
        defer want.deinit(gpa);
        want.print(gpa, "llama_model_load: error loading model: error loading model vocabulary: {s}\n", .{exc.what()}) catch unreachable;
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
        defer a.deinit(gpa);
        var b: std.ArrayList(u8) = .empty;
        defer b.deinit(gpa);
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

    const rv = c.llama_model_get_vocab(model).?;
    compareScalars(cs, rv, &vocab);
    if (vocab.type == V.vocab.type_none) return;
    comparePieces(cs, rv, &vocab);

    // Special-token texts to splice into random text.
    var specials: std.ArrayList([]const u8) = .empty;
    defer specials.deinit(gpa);
    for (vocab.cache_special_tokens.items) |id| {
        if (specials.items.len >= 40) break;
        specials.append(gpa, vocab.text(id)) catch unreachable;
    }

    var rbuf: std.ArrayList(u8) = .empty;
    defer rbuf.deinit(gpa);
    var texts: std.ArrayList([]const u8) = .empty;
    defer {
        for (texts.items[upstream.len..]) |t| gpa.free(t);
        texts.deinit(gpa);
    }
    texts.appendSlice(gpa, upstream) catch unreachable;
    for (0..400) |_| texts.append(gpa, gpa.dupe(u8, randomText(r, &rbuf, specials.items)) catch unreachable) catch unreachable;
    // Chains over four letters, where the many-overlap case's tokens meet
    // at every position.
    for (0..60) |_| {
        rbuf.clearRetainingCapacity();
        for (0..r.uintLessThan(usize, 40) + 4) |_| rbuf.append(gpa, "klmn"[r.uintLessThan(usize, 4)]) catch unreachable;
        texts.append(gpa, gpa.dupe(u8, rbuf.items) catch unreachable) catch unreachable;
    }

    for (texts.items) |t| compareText(cs, rv, &vocab, t);

    // Random id sequences through detokenize.
    const n_tokens: i32 = @intCast(vocab.nTokens());
    var ids: [24]Token = undefined;
    for (0..300) |_| {
        const n = r.uintLessThan(usize, ids.len + 1);
        for (ids[0..n]) |*id| id.* = if (r.uintLessThan(u8, 4) == 0 and vocab.cache_special_tokens.items.len != 0)
            vocab.cache_special_tokens.items[r.uintLessThan(usize, vocab.cache_special_tokens.items.len)]
        else
            r.intRangeLessThan(i32, 0, n_tokens);
        compareDetokenize(cs, rv, &vocab, ids[0..n]);
    }
}

fn compareScalars(cs: Case, rv: *const c.llama_vocab, v: *const Vocab) void {
    const Pair = struct { name: []const u8, ref: i64, port: i64 };
    const pairs = [_]Pair{
        .{ .name = "n_tokens", .ref = c.llama_vocab_n_tokens(rv), .port = v.nTokens() },
        .{ .name = "type", .ref = c.llama_vocab_type(rv), .port = v.type },
        .{ .name = "bos", .ref = c.llama_vocab_bos(rv), .port = v.special_bos_id },
        .{ .name = "eos", .ref = c.llama_vocab_eos(rv), .port = v.special_eos_id },
        .{ .name = "eot", .ref = c.llama_vocab_eot(rv), .port = v.special_eot_id },
        .{ .name = "sep", .ref = c.llama_vocab_sep(rv), .port = v.special_sep_id },
        .{ .name = "nl", .ref = c.llama_vocab_nl(rv), .port = v.linefeed_id },
        .{ .name = "pad", .ref = c.llama_vocab_pad(rv), .port = v.special_pad_id },
        .{ .name = "mask", .ref = c.llama_vocab_mask(rv), .port = v.special_mask_id },
        .{ .name = "fim_pre", .ref = c.llama_vocab_fim_pre(rv), .port = v.special_fim_pre_id },
        .{ .name = "fim_suf", .ref = c.llama_vocab_fim_suf(rv), .port = v.special_fim_suf_id },
        .{ .name = "fim_mid", .ref = c.llama_vocab_fim_mid(rv), .port = v.special_fim_mid_id },
        .{ .name = "fim_pad", .ref = c.llama_vocab_fim_pad(rv), .port = v.special_fim_pad_id },
        .{ .name = "fim_rep", .ref = c.llama_vocab_fim_rep(rv), .port = v.special_fim_rep_id },
        .{ .name = "fim_sep", .ref = c.llama_vocab_fim_sep(rv), .port = v.special_fim_sep_id },
        .{ .name = "add_bos", .ref = @intFromBool(c.llama_vocab_get_add_bos(rv)), .port = @intFromBool(v.add_bos) },
        .{ .name = "add_eos", .ref = @intFromBool(c.llama_vocab_get_add_eos(rv)), .port = @intFromBool(v.add_eos) },
        .{ .name = "add_sep", .ref = @intFromBool(c.llama_vocab_get_add_sep(rv)), .port = @intFromBool(v.add_sep) },
    };
    for (pairs) |p| {
        checks += 1;
        if (p.ref != p.port) fail("{s}: {s}: ref {d} port {d}", .{ cs.name, p.name, p.ref, p.port });
    }

    var n_sup: i32 = 0;
    const sup = c.llama_vocab_get_suppress_tokens(rv, &n_sup);
    checks += 1;
    if (n_sup != v.suppress_tokens.items.len or (n_sup > 0 and !std.mem.eql(Token, sup[0..@intCast(n_sup)], v.suppress_tokens.items))) {
        fail("{s}: suppress tokens differ", .{cs.name});
    }

    if (v.type == V.vocab.type_none) return;
    const n: i32 = @intCast(v.nTokens());
    var id: i32 = 0;
    while (id < n) : (id += 1) {
        checks += 1;
        const rt = std.mem.span(c.llama_vocab_get_text(rv, id));
        const d = v.id_to_token.items[@intCast(id)];
        const rs: u32 = @bitCast(c.llama_vocab_get_score(rv, id));
        const ra = c.llama_vocab_get_attr(rv, id);
        if (!std.mem.eql(u8, rt, v.text(id)) or rs != @as(u32, @bitCast(d.score)) or ra != d.attr or
            c.llama_vocab_is_eog(rv, id) != v.isEog(id) or c.llama_vocab_is_control(rv, id) != v.isControl(id))
        {
            fail("{s}: token {d} differs: text \"{f}\"/\"{f}\" attr {d}/{d} eog {}/{}", .{
                cs.name, id, std.zig.fmtString(rt), std.zig.fmtString(v.text(id)), ra, d.attr, c.llama_vocab_is_eog(rv, id), v.isEog(id),
            });
        }
    }
    checks += 2;
    if (c.llama_vocab_is_eog(rv, c.LLAMA_TOKEN_NULL) != v.isEog(c.LLAMA_TOKEN_NULL)) fail("{s}: is_eog(NULL)", .{cs.name});
}

fn comparePieces(cs: Case, rv: *const c.llama_vocab, v: *const Vocab) void {
    const n: i32 = @intCast(v.nTokens());
    var rb: [512]u8 = undefined;
    var zb: [512]u8 = undefined;
    var exc: staged.impl.Exception = .{};
    var id: i32 = 0;
    while (id < n) : (id += 1) {
        for ([_]bool{ false, true }) |special| {
            for ([_]i32{ 0, 1, 3 }) |lstrip| {
                for ([_]usize{ rb.len, 1, 0 }) |len| {
                    checks += 1;
                    const rn = c.llama_token_to_piece(rv, id, &rb, @intCast(len), lstrip, special);
                    const zn = v.tokenToPiece(id, zb[0..len], lstrip, special, &exc) catch exc.terminate("vocab_diff", 0);
                    if (rn != zn or (rn > 0 and !std.mem.eql(u8, rb[0..@intCast(rn)], zb[0..@intCast(zn)]))) {
                        fail("{s}: piece {d} special={} lstrip={d} len={d}: ref {d} port {d}", .{ cs.name, id, special, lstrip, len, rn, zn });
                    }
                }
            }
        }
    }
}

fn compareText(cs: Case, rv: *const c.llama_vocab, v: *const Vocab, t: []const u8) void {
    var rt: [4096]Token = undefined;
    var zt: [4096]Token = undefined;
    var exc: staged.impl.Exception = .{};
    for ([_]bool{ false, true }) |add_special| {
        for ([_]bool{ false, true }) |parse_special| {
            checks += 1;
            const rn = c.llama_tokenize(rv, t.ptr, @intCast(t.len), &rt, rt.len, add_special, parse_special);
            const zn = v.tokenizeInto(t, &zt, add_special, parse_special, &exc) catch exc.terminate("vocab_diff", 0);
            if (rn != zn or (rn > 0 and !std.mem.eql(Token, rt[0..@intCast(rn)], zt[0..@intCast(zn)]))) {
                fail("{s}: tokenize \"{f}\" add={} parse={}: ref {d} port {d}", .{ cs.name, std.zig.fmtString(t), add_special, parse_special, rn, zn });
                continue;
            }
            if (rn > 1) {
                // Too small a buffer reports the size needed.
                checks += 1;
                const rs = c.llama_tokenize(rv, t.ptr, @intCast(t.len), &rt, 1, add_special, parse_special);
                const zs = v.tokenizeInto(t, zt[0..1], add_special, parse_special, &exc) catch exc.terminate("vocab_diff", 0);
                if (rs != zs) fail("{s}: short tokenize \"{f}\": ref {d} port {d}", .{ cs.name, std.zig.fmtString(t), rs, zs });
            }
            if (rn > 0) compareDetokenize(cs, rv, v, rt[0..@intCast(rn)]);
        }
    }
}

fn compareDetokenize(cs: Case, rv: *const c.llama_vocab, v: *const Vocab, ids: []const Token) void {
    var rb: [16384]u8 = undefined;
    var zb: [16384]u8 = undefined;
    var exc: staged.impl.Exception = .{};
    for ([_]bool{ false, true }) |remove_special| {
        for ([_]bool{ false, true }) |unparse_special| {
            for ([_]usize{ rb.len, 3 }) |len| {
                checks += 1;
                const rn = c.llama_detokenize(rv, ids.ptr, @intCast(ids.len), &rb, @intCast(len), remove_special, unparse_special);
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

fn setStrs(ctx: *c.gguf_context, key: [:0]const u8, strs: []const []const u8) void {
    var ptrs: std.ArrayList([*c]const u8) = .empty;
    defer ptrs.deinit(gpa);
    var owned: std.ArrayList([:0]u8) = .empty;
    defer {
        for (owned.items) |o| gpa.free(o);
        owned.deinit(gpa);
    }
    for (strs) |s| {
        const z = gpa.dupeSentinel(u8, s, 0) catch unreachable;
        owned.append(gpa, z) catch unreachable;
        ptrs.append(gpa, z.ptr) catch unreachable;
    }
    c.gguf_set_arr_str(ctx, key.ptr, @ptrCast(ptrs.items.ptr), ptrs.items.len);
}

fn setI32s(ctx: *c.gguf_context, key: [:0]const u8, v: []const i32) void {
    c.gguf_set_arr_data(ctx, key.ptr, c.GGUF_TYPE_INT32, v.ptr, v.len);
}

fn setF32s(ctx: *c.gguf_context, key: [:0]const u8, v: []const f32) void {
    c.gguf_set_arr_data(ctx, key.ptr, c.GGUF_TYPE_FLOAT32, v.ptr, v.len);
}

/// A copy of `base` with `edit` applied, written to `dir/name.gguf`.
fn derive(dir: []const u8, base: [:0]const u8, name: []const u8, comptime edit: fn (*c.gguf_context) void) [:0]const u8 {
    const ctx = c.gguf_init_from_file(base.ptr, .{ .no_alloc = true, .ctx = null }).?;
    defer c.gguf_free(ctx);
    edit(ctx);
    const path = gpa.printSentinel("{s}/{s}.gguf", .{ dir, name }, 0) catch unreachable;
    if (!c.gguf_write_to_file(ctx, path.ptr, true)) @panic("cannot write a synthetic file");
    return path;
}

fn findStrs(ctx: *c.gguf_context, key: [:0]const u8, out: *std.ArrayList([]const u8)) void {
    const k = c.gguf_find_key(ctx, key.ptr);
    const n = c.gguf_get_arr_n(ctx, k);
    for (0..n) |i| out.append(gpa, gpa.dupe(u8, staged.gguf_meta.cString(c.gguf_get_arr_str(ctx, k, i))) catch unreachable) catch unreachable;
}

const edits = struct {
    fn rwkv(ctx: *c.gguf_context) void {
        var toks: std.ArrayList([]const u8) = .empty;
        for (0..256) |b| toks.append(gpa, gpa.print("\\x{x:0>2}", .{b}) catch unreachable) catch unreachable;
        for ([_][]const u8{ "the", " the", "hello", " world", "\\n\\n", "ab\\\\c", "\\t", "  ", "中文", "\\x41BC", "\\x\xc3\xa9", "\\x\xffz" }) |w| toks.append(gpa, w) catch unreachable;
        c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "rwkv");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.token_type");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.pre");
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
    }

    fn plamo2(ctx: *c.gguf_context) void {
        var toks: std.ArrayList([]const u8) = .empty;
        var types: std.ArrayList(i32) = .empty;
        var scores: std.ArrayList(f32) = .empty;
        for ([_][]const u8{ "<|plamo:unk|>", "<|plamo:bos|>", "<|plamo:eos|>", "<|plamo:pad|>" }) |w| {
            toks.append(gpa, w) catch unreachable;
            types.append(gpa, c.LLAMA_TOKEN_TYPE_CONTROL) catch unreachable;
            scores.append(gpa, 0) catch unreachable;
        }
        for (0..256) |b| {
            toks.append(gpa, gpa.print("<0x{X:0>2}>", .{b}) catch unreachable) catch unreachable;
            types.append(gpa, c.LLAMA_TOKEN_TYPE_BYTE) catch unreachable;
            scores.append(gpa, 0) catch unreachable;
        }
        const words = [_][]const u8{ "the", "he", "e", "hello", "lo", "world", " the", " ", "日本", "本", "語", "日本語", "ab", "abc", "bc", "c", "Hello", "wor", "ld", "\u{FEFF}x" };
        // "pq" costs 20000.3 before rounding and "p" + "q" exactly 20000: a
        // tie once rounded, which the longer piece wins, and not before.
        for ([_][]const u8{ "pq", "p", "q" }, [_]f32{ -2.00003, -1.0, -1.0 }) |w, sc| {
            toks.append(gpa, w) catch unreachable;
            types.append(gpa, c.LLAMA_TOKEN_TYPE_NORMAL) catch unreachable;
            scores.append(gpa, sc) catch unreachable;
        }
        for (words, 0..) |w, i| {
            toks.append(gpa, w) catch unreachable;
            types.append(gpa, c.LLAMA_TOKEN_TYPE_NORMAL) catch unreachable;
            scores.append(gpa, -@as(f32, @floatFromInt(i)) * 0.37 - 1.0) catch unreachable;
        }
        c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "plamo2");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.pre");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.bos_token_id");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.eos_token_id");
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
        setF32s(ctx, "tokenizer.ggml.scores", scores.items);
    }

    fn appendTokens(ctx: *c.gguf_context, extra: []const []const u8) void {
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        for (extra) |e| toks.append(gpa, e) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.token_type");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.scores");
    }

    fn hybriddna(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "hybriddna");
        appendTokens(ctx, &.{ "<dna>", "</dna>", "<oov>", "AAAAAA\xee\x80\x80", "ACGTAC\xee\x80\x80", "GTAAAA\xee\x80\x80", "CCCCCC\xee\x80\x80" });
    }

    /// User-defined tokens of equal length that overlap in text, so which
    /// one the special-token cache tries first decides the result.
    fn overlap(ctx: *c.gguf_context) void {
        const extra = [_][]const u8{ "xyzw", "zwab", "abxy", "wabq", "QQRR", "RRSS" };
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        const k = c.gguf_find_key(ctx, "tokenizer.ggml.token_type");
        var types: std.ArrayList(i32) = .empty;
        if (k >= 0) {
            const tt: [*]const i32 = @ptrCast(@alignCast(c.gguf_get_arr_data(ctx, k).?));
            types.appendSlice(gpa, tt[0..toks.items.len]) catch unreachable;
        } else {
            types.appendNTimes(gpa, c.LLAMA_TOKEN_TYPE_NORMAL, toks.items.len) catch unreachable;
        }
        for (extra) |e| {
            toks.append(gpa, e) catch unreachable;
            types.append(gpa, c.LLAMA_TOKEN_TYPE_USER_DEFINED) catch unreachable;
        }
        // The scores, where there are any, must still cover every token.
        const sk = c.gguf_find_key(ctx, "tokenizer.ggml.scores");
        if (sk >= 0) {
            const sc: [*]const f32 = @ptrCast(@alignCast(c.gguf_get_arr_data(ctx, sk).?));
            var scores: std.ArrayList(f32) = .empty;
            scores.appendSlice(gpa, sc[0 .. toks.items.len - extra.len]) catch unreachable;
            scores.appendNTimes(gpa, 0, extra.len) catch unreachable;
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
    fn overlapMany(ctx: *c.gguf_context) void {
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
            toks.append(gpa, gpa.dupe(u8, &w) catch unreachable) catch unreachable;
        }
        var types: std.ArrayList(i32) = .empty;
        types.appendNTimes(gpa, c.LLAMA_TOKEN_TYPE_NORMAL, base) catch unreachable;
        types.appendNTimes(gpa, c.LLAMA_TOKEN_TYPE_USER_DEFINED, 64) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
    }

    /// A BPE vocabulary with the `<mask>` lstrip jina's pre-tokenizer name
    /// sets, so a space before it is stripped -- where a lone space would
    /// otherwise become a token.
    fn lstripBpe(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "jina-v2-de");
        var toks: std.ArrayList([]const u8) = .empty;
        findStrs(ctx, "tokenizer.ggml.tokens", &toks);
        var types: std.ArrayList(i32) = .empty;
        types.appendNTimes(gpa, c.LLAMA_TOKEN_TYPE_NORMAL, toks.items.len) catch unreachable;
        toks.append(gpa, "<mask>") catch unreachable;
        types.append(gpa, c.LLAMA_TOKEN_TYPE_CONTROL) catch unreachable;
        setStrs(ctx, "tokenizer.ggml.tokens", toks.items);
        setI32s(ctx, "tokenizer.ggml.token_type", types.items);
    }

    /// A small T5 vocabulary whose scores make the unknown-token path
    /// compete: "a" + unknown "c" against the pair "ac".
    fn ugmScores(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "t5");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.pre");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.bos_token_id");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.eos_token_id");
        setStrs(ctx, "tokenizer.ggml.tokens", &.{ "<pad>", "</s>", "<unk>", "\xe2\x96\x81", "a", "ac", "z", "\xe2\x96\x81z", "x" });
        setI32s(ctx, "tokenizer.ggml.token_type", &.{ c.LLAMA_TOKEN_TYPE_CONTROL, c.LLAMA_TOKEN_TYPE_CONTROL, c.LLAMA_TOKEN_TYPE_UNKNOWN, 1, 1, 1, 1, 1, 1 });
        // min score -1 makes UNK -11; "a" at 9.5 then puts a + UNK at -1.5,
        // just below "ac" at -1, and a penalty of 9 would put it above.
        setF32s(ctx, "tokenizer.ggml.scores", &.{ 0, 0, 0, -0.5, 9.5, -1.0, -0.25, -0.75, -0.6 });
    }

    /// A BPE vocabulary of exactly eleven tokens, so the default BOS and
    /// EOS id 11 is one past the end.
    fn tinyBpe(ctx: *c.gguf_context) void {
        setStrs(ctx, "tokenizer.ggml.tokens", &.{ "a", "b", "c", "d", "e", "f", "g", "h", "ab", "\xc4\x8a", "\xc4\xa0" });
        setStrs(ctx, "tokenizer.ggml.merges", &.{"a b"});
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.token_type");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.bos_token_id");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.eos_token_id");
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.unknown_token_id");
    }

    fn whitespace(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "whitespace");
        c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "whitespace");
    }

    fn none(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "no_vocab");
        c.gguf_set_val_u32(ctx, "gpt2.vocab_size", 77);
    }

    fn spmFlags(ctx: *c.gguf_context) void {
        c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", false);
        c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_eos_token", true);
        c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_space_prefix", false);
    }

    fn ugmFlags(ctx: *c.gguf_context) void {
        c.gguf_set_val_bool(ctx, "tokenizer.ggml.remove_extra_whitespaces", true);
        c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_space_prefix", true);
        c.gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", true);
    }

    fn badModel(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "tokenizer.ggml.model", "no-such-tokenizer");
    }

    fn badPre(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "tokenizer.ggml.pre", "no-such-pre");
    }

    fn noMerges(ctx: *c.gguf_context) void {
        _ = c.gguf_remove_key(ctx, "tokenizer.ggml.merges");
    }

    fn shortScores(ctx: *c.gguf_context) void {
        setF32s(ctx, "tokenizer.ggml.scores", &.{ 1.0, 2.0 });
    }

    fn wrongType(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "tokenizer.ggml.add_bos_token", 1);
    }

    fn badSpecial(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", 99999999);
        c.gguf_set_val_u32(ctx, "tokenizer.ggml.eot_token_id", 5);
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

    c.llama_log_set(logCb, null);
    c.llama_backend_init();

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
                try buf.appendSlice(gpa, chunk[0..n]);
            }
            _ = std.c.fclose(f);
            var it = std.mem.splitSequence(u8, buf.items, "\n__ggml_vocab_test__\n");
            while (it.next()) |s| {
                if ((try seen.getOrPut(gpa, s)).found_existing) continue;
                try corpus.append(gpa, s);
            }
        } else {
            const base = std.Io.Dir.path.basename(a);
            try cases.append(gpa, .{ .name = base, .path = a });
            try by_name.put(gpa, base, a);
        }
    }

    // Synthetic cases, derived from upstream's files.
    const gpt2 = by_name.get("ggml-vocab-gpt-2.gguf") orelse return error.MissingGpt2Vocab;
    const spm = by_name.get("ggml-vocab-llama-spm.gguf") orelse return error.MissingSpmVocab;
    const ugm = by_name.get("ggml-vocab-nomic-bert-moe.gguf") orelse return error.MissingUgmVocab;
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
        .{ .name = "error-model", .path = derive(dir, gpt2, "bad-model", edits.badModel) },
        .{ .name = "error-pre", .path = derive(dir, gpt2, "bad-pre", edits.badPre) },
        .{ .name = "error-merges", .path = derive(dir, gpt2, "no-merges", edits.noMerges) },
        .{ .name = "error-scores", .path = derive(dir, spm, "short-scores", edits.shortScores) },
        .{ .name = "error-type", .path = derive(dir, gpt2, "wrong-type", edits.wrongType) },
        .{ .name = "override-bool", .path = gpt2, .overrides = &.{ kvBool("tokenizer.ggml.add_bos_token", true), kvBool("tokenizer.ggml.add_eos_token", true) } },
        .{ .name = "override-mistyped", .path = spm, .overrides = &.{ kvInt("tokenizer.ggml.add_bos_token", 0), kvInt("tokenizer.ggml.eos_token_id", 5) } },
    };
    try cases.appendSlice(gpa, &synthetic);

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
