//! Compares `src/llama/unicode.zig` with llama.cpp's `unicode.cpp`.
//!
//! **Not a port.** The harness behind `scripts/unicode-diff`, linked with
//! `harness/unicode_ref.cpp` and the unmodified `unicode.cpp` and
//! `unicode-data.cpp`, so both sides run in one process on the same input:
//!
//! - **Every code point**, 0 to 0x10FFFF and a few beyond: the flags,
//!   `tolower`, `is_han`, NFD, and the UTF-8 encoding or its refusal.
//! - **UTF-8 decoding**: every one- and two-byte string, three- and
//!   four-byte strings over the lead bytes and a set of trailing bytes
//!   chosen to sit on each boundary, and random byte strings -- through the
//!   single-sequence decoder, the whole-string one and the flags-of-string
//!   one.
//! - **The byte map**, both directions, over every byte and every one- and
//!   two-byte string.
//! - **`unicode_regex_split`**, for every pre-tokenizer pattern set in
//!   `llama-vocab.cpp` (extracted by the script, so the list cannot drift)
//!   and a set of synthetic ones that reach the collapse and error paths,
//!   over upstream's vocab test strings and random strings, with and
//!   without the byte encoding.
//!
//! Usage: unicode_diff <sets-file> <corpus.inp>...

const std = @import("std");
const unicode = @import("unicode");

extern fn ref_uni_flags(cpt: u32) u16;
extern fn ref_uni_tolower(cpt: u32) u32;
extern fn ref_uni_is_han(cpt: u32) c_int;
extern fn ref_uni_len_utf8(c: u8) usize;
extern fn ref_uni_nfd(in: [*]const u32, n: usize, out: [*]u32) void;
extern fn ref_uni_cpt_to_utf8(cpt: u32, out: [*]u8) c_int;
extern fn ref_uni_cpts_from_utf8(s: [*]const u8, n: usize, out: [*]u32) usize;
extern fn ref_uni_cpt_from_utf8(s: [*]const u8, n: usize, offset: *usize, cpt: *u32) c_int;
extern fn ref_uni_flags_from_utf8(s: [*]const u8, n: usize) c_int;
extern fn ref_uni_byte_to_utf8(b: u8, out: [*]u8) c_int;
extern fn ref_uni_utf8_to_byte(s: [*]const u8, n: usize) c_int;
extern fn ref_uni_regex_split(text: [*]const u8, text_len: usize, exprs: [*]const [*]const u8, expr_lens: [*]const usize, n_exprs: usize, byte_encode: c_int, out: *?[*]u8, out_len: *usize) c_long;
extern fn ref_uni_free(p: ?[*]u8) void;

const gpa = std.heap.c_allocator;

var checks: usize = 0;
var failures: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    if (failures <= 20) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

// -----------------------------------------------------------------------------
// Code points

fn sweepCodepoints() void {
    const extra = [_]u32{ 0x110000, 0x110001, 0x1FFFFF, 0x200000, 0xFFFFFFFF };
    const n: usize = 0x110000 + extra.len;
    const all = gpa.alloc(u32, n) catch unreachable;
    defer gpa.free(all);
    for (0..0x110000) |c| all[c] = @intCast(c);
    @memcpy(all[0x110000..], &extra);

    for (all) |c| {
        checks += 4;
        const rf = ref_uni_flags(c);
        const zf = unicode.cptFlagsFromCpt(c).asUint();
        if (rf != zf) fail("flags U+{X:0>4}: ref 0x{X:0>4} zig 0x{X:0>4}", .{ c, rf, zf });
        const rl = ref_uni_tolower(c);
        const zl = unicode.tolower(c);
        if (rl != zl) fail("tolower U+{X:0>4}: ref U+{X:0>4} zig U+{X:0>4}", .{ c, rl, zl });
        const rh = ref_uni_is_han(c) != 0;
        const zh = unicode.cptIsHan(c);
        if (rh != zh) fail("is_han U+{X:0>4}: ref {} zig {}", .{ c, rh, zh });
        var rb: [4]u8 = undefined;
        var zb: [4]u8 = undefined;
        const rn = ref_uni_cpt_to_utf8(c, &rb);
        if (unicode.cptToUtf8(c, &zb)) |zs| {
            if (rn < 0 or !std.mem.eql(u8, rb[0..@intCast(rn)], zs)) fail("cpt_to_utf8 U+{X:0>4}: ref {d} zig {x}", .{ c, rn, zs });
        } else |_| {
            if (rn >= 0) fail("cpt_to_utf8 U+{X:0>4}: ref {d} bytes, zig refused", .{ c, rn });
        }
    }

    // NFD, the whole range as one vector.
    const rn = gpa.alloc(u32, n) catch unreachable;
    defer gpa.free(rn);
    ref_uni_nfd(all.ptr, n, rn.ptr);
    const zn = unicode.cptsNormalizeNfd(all);
    defer gpa.free(zn);
    for (all, rn, zn) |c, r, z| {
        checks += 1;
        if (r != z) fail("nfd U+{X:0>4}: ref U+{X:0>4} zig U+{X:0>4}", .{ c, r, z });
    }

    for (0..256) |b| {
        checks += 1;
        const r = ref_uni_len_utf8(@intCast(b));
        const z = unicode.lenUtf8(@intCast(b));
        if (r != z) fail("len_utf8 0x{X:0>2}: ref {d} zig {d}", .{ b, r, z });
    }
}

// -----------------------------------------------------------------------------
// UTF-8 decoding

fn compareDecode(s: []const u8) void {
    // The single-sequence decoder at offset 0, when there is one.
    if (s.len > 0) {
        checks += 2;
        var roff: usize = 0;
        var rcpt: u32 = 0;
        const rr = ref_uni_cpt_from_utf8(s.ptr, s.len, &roff, &rcpt);
        var zoff: usize = 0;
        if (unicode.cptFromUtf8(s, &zoff)) |zcpt| {
            if (rr != 0 or rcpt != zcpt or roff != zoff) fail("cpt_from_utf8 {x}: ref rc {d} U+{X} off {d}, zig U+{X} off {d}", .{ s, rr, rcpt, roff, zcpt, zoff });
        } else |_| {
            if (rr == 0 or roff != zoff) fail("cpt_from_utf8 {x}: ref rc {d} off {d}, zig refused off {d}", .{ s, rr, roff, zoff });
        }
        const rf = ref_uni_flags_from_utf8(s.ptr, s.len);
        const zf: c_int = if (unicode.cptFlagsFromUtf8(s)) |f| f.asUint() else |_| -1;
        if (rf != zf) fail("flags_from_utf8 {x}: ref {d} zig {d}", .{ s, rf, zf });
    }
    checks += 1;
    var rbuf: [64]u32 = undefined;
    const rn = ref_uni_cpts_from_utf8(s.ptr, s.len, &rbuf);
    const z = unicode.cptsFromUtf8(s);
    defer gpa.free(z);
    if (!std.mem.eql(u32, rbuf[0..rn], z)) fail("cpts_from_utf8 {x}: ref {any} zig {any}", .{ s, rbuf[0..rn], z });
}

fn sweepDecode(r: std.Random) void {
    var buf: [8]u8 = undefined;
    compareDecode("");
    for (0..256) |a| {
        buf[0] = @intCast(a);
        compareDecode(buf[0..1]);
        for (0..256) |b| {
            buf[1] = @intCast(b);
            compareDecode(buf[0..2]);
        }
    }
    // Trailing bytes on every boundary a decoder tests.
    const trail = [_]u8{ 0x00, 0x41, 0x7F, 0x80, 0x8F, 0x90, 0xA0, 0xBF, 0xC0, 0xC2, 0xE0, 0xF0, 0xFF };
    for (0x80..0x100) |a| {
        buf[0] = @intCast(a);
        for (trail) |b| for (trail) |c| {
            buf[1] = b;
            buf[2] = c;
            compareDecode(buf[0..3]);
            if (a >= 0xE0) for (trail) |d| {
                buf[3] = d;
                compareDecode(buf[0..4]);
            };
        };
    }
    for (0..300000) |_| {
        const n = r.uintLessThan(usize, buf.len + 1);
        for (buf[0..n]) |*ch| ch.* = if (r.boolean()) r.int(u8) else trail[r.uintLessThan(usize, trail.len)];
        compareDecode(buf[0..n]);
    }
}

// -----------------------------------------------------------------------------
// The byte map

fn sweepByteMap() void {
    for (0..256) |b| {
        checks += 1;
        var rb: [4]u8 = undefined;
        const rn = ref_uni_byte_to_utf8(@intCast(b), &rb);
        const z = unicode.byteToUtf8(@intCast(b));
        if (!std.mem.eql(u8, rb[0..@intCast(rn)], z)) fail("byte_to_utf8 0x{X:0>2}: ref {x} zig {x}", .{ b, rb[0..@intCast(rn)], z });
    }
    var buf: [3]u8 = undefined;
    const lens = [_]usize{ 0, 1, 2 };
    for (lens) |len| {
        const count: usize = if (len == 0) 1 else if (len == 1) 256 else 65536;
        for (0..count) |i| {
            if (len >= 1) buf[0] = @truncate(i);
            if (len == 2) buf[1] = @truncate(i >> 8);
            const s = buf[0..len];
            checks += 1;
            const r = ref_uni_utf8_to_byte(s.ptr, s.len);
            const z: c_int = if (unicode.utf8ToByte(s)) |b| b else |_| -1;
            if (r != z) fail("utf8_to_byte {x}: ref {d} zig {d}", .{ s, r, z });
        }
    }
    // Three bytes is never a key; check a few that start like one.
    for ([_][3]u8{ .{ 0xC4, 0x80, 0x00 }, .{ 'a', 'b', 'c' }, .{ 0xE2, 0x82, 0xAC } }) |t| {
        buf = t;
        checks += 1;
        const r = ref_uni_utf8_to_byte(&buf, 3);
        const z: c_int = if (unicode.utf8ToByte(&buf)) |b| b else |_| -1;
        if (r != z) fail("utf8_to_byte {x}: ref {d} zig {d}", .{ buf, r, z });
    }
}

// -----------------------------------------------------------------------------
// unicode_regex_split

const Set = []const []const u8;

/// Reads the sets the script extracted: a `SET` line opens each, and each
/// pattern follows as a line of hex.
fn readSets(path: []const u8) []Set {
    const text = readFile(path);
    var sets: std.ArrayList(Set) = .empty;
    var cur: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.eql(u8, line, "SET")) {
            if (cur.items.len != 0) sets.append(gpa, cur.toOwnedSlice(gpa) catch unreachable) catch unreachable;
            continue;
        }
        if (line.len == 0) continue;
        const bytes = gpa.alloc(u8, line.len / 2) catch unreachable;
        _ = std.fmt.hexToBytes(bytes, line) catch @panic("bad hex in sets file");
        cur.append(gpa, bytes) catch unreachable;
    }
    if (cur.items.len != 0) sets.append(gpa, cur.toOwnedSlice(gpa) catch unreachable) catch unreachable;
    return sets.toOwnedSlice(gpa) catch unreachable;
}

fn readFile(path: []const u8) []u8 {
    const f = std.c.fopen(@ptrCast(gpa.dupeZ(u8, path) catch unreachable), "rb") orelse {
        std.debug.print("cannot open {s}\n", .{path});
        std.process.exit(2);
    };
    defer _ = std.c.fclose(f);
    var out: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.fread(&buf, 1, buf.len, f);
        if (n == 0) break;
        out.appendSlice(gpa, buf[0..n]) catch unreachable;
    }
    return out.toOwnedSlice(gpa) catch unreachable;
}

/// Patterns no vocabulary uses, for the paths the real sets do not reach:
/// a category inside and outside a bracket, a property name that is not a
/// category, a category beside non-ASCII text (an error), malformed
/// patterns (an error), a pattern that can match empty, and the custom
/// splitters' strings run one at a time.
const synthetic = [_]Set{
    &.{"\\p{N}+|[\\p{L}\\p{M}]+|\\p{P}|\\p{S}"},
    &.{"[^\\p{L}\\p{N}\\s]+|\\p{Lu}\\p{Ll}*"},
    &.{"\\p{X}+|\\p{N}"},
    &.{"\\p{Lo}{2}|\\p{Lt}|\\p{Lm}"},
    &.{"é\\p{L}"},
    &.{"(\\p{L}"},
    &.{"[a-"},
    &.{"x*"},
    &.{"(?=a)|b*"},
    &.{ "\\s+", "\\S" },
    &.{"[\\u4e00-\\u9fa5]+|."},
    &.{"\\bé+\\b|\\w+"},
    &.{"^.|.$"},
    &.{ "\\p{AFMoE_digits}", "[^\\n]+|[\\n]+" },
    &.{"\\d{1,3}(?=(?:\\d{3})*\\b)"},
    &.{"\\p{Han}+"},
};

/// Code points and short runs random text is built from: one or more for
/// every branch of every custom splitter and every collapse category.
const pool = [_][]const u8{
    "a",        "Z",        "q",        "7",            "0",        " ",   " ",    "  ",    "\t",    "\n", "\r", "\r\n", "\x0B", "'", "'s", "'S", "'re", "'LL", "'d", "'x",
    ".",        ",",        "!",        "?",            "-",        "_",   "$",    "+",     "<",     "|",  "~",  "`",    "\\",   "/", "@",  "#",  "(",   ")",   "[",  "]",
    "é",
    "ß",
    "İ",
    "Σ",
    "ж",
    "ǅ",
    "ʰ",
    "\u{0301}", "\u{0308}",
    "中",
    "一",
    "龥",
    "𠀀",
    "가",
    "の",
    "ア",
    "ก",
    "ع",
    "٣",
    "１",
    "！",
    "。",
    "\u{00A0}", "\u{2028}", "\u{3000}", "\u{0085}",     "\u{200B}",
    "😀",
    "\u{200D}",
    "€",
    "©",
    "\u{FFFD}",
    "ﬁ",
    "Ⅻ",
    "½",
    "\x00",     "\xff",     "\xc3",     "\xed\xa0\x80", "\xc0\x80", "123", "4567", "Hello", "world",
};

/// Rare: a code point above U+10FFFF, which `unicode_cpt_to_utf8` refuses.
const pool_rare = [_][]const u8{ "\xf4\x90\x80\x80", "\xf7\xbf\xbf\xbf" };

fn randomText(r: std.Random, buf: *std.ArrayList(u8)) []const u8 {
    buf.clearRetainingCapacity();
    const n = r.uintLessThan(usize, 40);
    for (0..n) |_| {
        const piece = if (r.uintLessThan(u16, 400) == 0) pool_rare[r.uintLessThan(usize, pool_rare.len)] else pool[r.uintLessThan(usize, pool.len)];
        buf.appendSlice(gpa, piece) catch unreachable;
    }
    return buf.items;
}

fn compareSplit(text: []const u8, set: Set, byte_encode: bool) void {
    checks += 1;
    var ptrs: [16][*]const u8 = undefined;
    var lens: [16]usize = undefined;
    for (set, 0..) |e, i| {
        ptrs[i] = e.ptr;
        lens[i] = e.len;
    }
    var rout: ?[*]u8 = null;
    var rlen: usize = 0;
    const rn = ref_uni_regex_split(text.ptr, text.len, &ptrs, &lens, set.len, @intFromBool(byte_encode), &rout, &rlen);
    defer ref_uni_free(rout);

    var zwords = unicode.regexSplit(text, set, byte_encode) catch |err| {
        const zn: c_long = switch (err) {
            error.RegexFailed => -1,
            error.CategoryWithNonAscii => -2,
            error.InvalidCodepoint => -3,
        };
        if (rn != zn) fail("split set[0]=\"{f}\" text=\"{f}\" encode={}: ref {d} zig {d}", .{ std.zig.fmtString(set[0]), std.zig.fmtString(text), byte_encode, rn, zn });
        return;
    };
    defer zwords.deinit();
    if (rn < 0) {
        fail("split set[0]=\"{f}\" text=\"{f}\" encode={}: ref {d}, zig {d} pieces", .{ std.zig.fmtString(set[0]), std.zig.fmtString(text), byte_encode, rn, zwords.len() });
        return;
    }
    var same = @as(usize, @intCast(rn)) == zwords.len();
    var at: usize = 0;
    var i: usize = 0;
    while (same and i < zwords.len()) : (i += 1) {
        const len = std.mem.readInt(u32, rout.?[at..][0..4], .little);
        const piece = rout.?[at + 4 .. at + 4 + len];
        at += 4 + len;
        if (!std.mem.eql(u8, piece, zwords.get(i))) same = false;
    }
    if (!same) {
        fail("split set[0]=\"{f}\" text=\"{f}\" encode={}: pieces differ at {d} (ref {d} pieces, zig {d})", .{ std.zig.fmtString(set[0]), std.zig.fmtString(text), byte_encode, i, rn, zwords.len() });
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: unicode_diff <sets-file> <corpus.inp>...\n", .{});
        std.process.exit(2);
    }

    var prng = std.Random.DefaultPrng.init(0x0c0de);
    const r = prng.random();

    sweepCodepoints();
    const c0 = checks;
    sweepDecode(r);
    const c1 = checks;
    sweepByteMap();
    const c2 = checks;

    const sets = readSets(args[1]);
    var corpus: std.ArrayList([]const u8) = .empty;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    for (args[2..]) |path| {
        const text = readFile(path);
        var it = std.mem.splitSequence(u8, text, "\n__ggml_vocab_test__\n");
        while (it.next()) |s| {
            if ((try seen.getOrPut(gpa, s)).found_existing) continue;
            try corpus.append(gpa, s);
        }
    }
    const n_upstream = corpus.items.len;
    var rbuf: std.ArrayList(u8) = .empty;
    for (0..1500) |_| try corpus.append(gpa, try gpa.dupe(u8, randomText(r, &rbuf)));

    for (sets) |set| {
        for (corpus.items) |text| {
            compareSplit(text, set, true);
            compareSplit(text, set, false);
        }
    }
    for (synthetic) |set| {
        for (corpus.items) |text| compareSplit(text, set, false);
    }

    std.debug.print("  code points  {d} checks\n  decoding     {d}\n  byte map     {d}\n  regex split  {d} ({d} sets + {d} synthetic, {d} upstream + 1500 random texts)\n", .{
        c0, c1 - c0, c2 - c1, checks - c2, sets.len, synthetic.len, n_upstream,
    });
    if (failures == 0) {
        std.debug.print("PASS: {d} unicode results identical, Zig vs unicode.cpp\n", .{checks});
        return;
    }
    std.debug.print("FAIL: {d} of {d} checks differ\n", .{ failures, checks });
    std.process.exit(1);
}
