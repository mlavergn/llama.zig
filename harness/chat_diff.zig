//! Compares `src/llama/chat.zig` with llama.cpp's `llama-chat.cpp`.
//!
//! **Not a port.** The harness behind `scripts/chat-diff`. It links the
//! C++ in `libllama.a` -- through `harness/chat_ref.cpp` for the
//! C++-linkage functions, directly for `llama_chat_apply_template` and
//! `llama_chat_builtin_templates` -- beside the port, in one process, and
//! compares on bytes:
//!
//! - **`llama_chat_builtin_templates`** at length 0 (with a null array), a
//!   short length, exactly the count and past it: the count, every name in
//!   order, and that nothing past `len` was written;
//! - **`llm_chat_template_from_str` and `llm_chat_detect_template`** on
//!   every built-in name and variants of it; on every marker string
//!   `llm_chat_detect_template` tests, **extracted from `llama-chat.cpp` at
//!   run time** so the list cannot drift, alone, in pairs, in random
//!   combinations and with single bytes dropped; and on every chat template
//!   in the files given -- the `tokenizer.chat_template*` keys of GGUF files
//!   and `.jinja` sources -- whole, cut just before and just after every
//!   marker occurrence, with occurrences deleted one at a time and one
//!   marker at a time, peeled marker by marker down the cascade, and cut
//!   every 97 bytes;
//! - **`llm_chat_apply_template`** for every `llm_chat_template` value,
//!   `UNKNOWN` included, over fixed and random conversations, both values
//!   of `add_ass`: the return value and the whole output, with the output
//!   pre-loaded so a -1 that leaves it alone is told apart from one that
//!   clears it. The conversations sweep every byte from 1 to 255 at both
//!   edges of a message, for `trim`'s `isspace`, and as a role's first byte
//!   and a whole role, for `toupper` -- in the `"C"` locale and again under
//!   UTF-8, Latin-1 and PT154 locales, where those two stop being
//!   ASCII-only;
//! - **`llama_chat_apply_template`**, the public entry point, for every
//!   built-in name and every template source, against `detectTemplate`
//!   followed by `applyTemplate`.
//!
//! Usage: chat_diff <llama-chat.cpp> <file.gguf|file.jinja>...

const std = @import("std");
const staged = @import("staged");

const c = staged.impl.c;
const chat = staged.chat;
const Message = chat.Message;

const gpa = std.heap.c_allocator;

extern fn ref_detect(p: [*]const u8, n: usize) c_int;
extern fn ref_from_str(p: [*]const u8, n: usize) c_int;
extern fn ref_apply(tmpl: c_int, msgs: [*]const Message, n: usize, add_ass: bool, init: [*:0]const u8, out: *[*]u8, out_len: *usize) i32;
extern fn ref_free_buf(p: [*]u8) void;

extern "c" fn setlocale(category: c_int, locale: ?[*:0]const u8) ?[*:0]const u8;
/// `LC_ALL` in `<locale.h>` on this target.
const LC_ALL: c_int = 0;
/// UTF-8 and Latin-1, where `isspace` and `toupper` reach past ASCII, and
/// `kk_KZ.PT154`, whose `toupper` table is wrong for ASCII itself (`'a'`
/// and `'A'` both map to `'Z'`) -- the one installed locale, of 288
/// measured, where upper-casing a *signed* `char` differs from ASCII.
const locales = [_][*:0]const u8{ "en_US.UTF-8", "en_US.ISO8859-1", "tr_TR.ISO8859-9", "kk_KZ.PT154" };

var checks: usize = 0;
var failures: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    if (failures <= 25) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

/// `s` with control and non-ASCII bytes escaped, at most 160 bytes of it.
fn esc(s: []const u8) []const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s[0..@min(s.len, 160)]) |b| {
        if (b >= 0x20 and b < 0x7f and b != '\\') {
            out.append(gpa, b) catch unreachable;
        } else {
            const e = std.fmt.allocPrint(gpa, "\\x{x:0>2}", .{b}) catch unreachable;
            out.appendSlice(gpa, e) catch unreachable;
        }
    }
    if (s.len > 160) out.appendSlice(gpa, "...") catch unreachable;
    return out.items;
}

fn readFile(path: []const u8) []u8 {
    const f = std.c.fopen(@ptrCast(gpa.dupeSentinel(u8, path, 0) catch unreachable), "rb") orelse {
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

// -----------------------------------------------------------------------------
// The marker strings

/// Every string literal in `llm_chat_detect_template`'s body, unescaped:
/// the markers it tests for, plus `"mistral"`, which it tests as a prefix.
fn extractNeedles(src: []const u8) [][]const u8 {
    const start = std.mem.find(u8, src, "llm_chat_template llm_chat_detect_template(") orelse {
        std.debug.print("FAIL: llm_chat_detect_template not found in llama-chat.cpp\n", .{});
        std.process.exit(1);
    };
    const end = start + (std.mem.find(u8, src[start..], "return LLM_CHAT_TEMPLATE_UNKNOWN;") orelse {
        std.debug.print("FAIL: the end of llm_chat_detect_template not found\n", .{});
        std.process.exit(1);
    });
    var needles: std.ArrayList([]const u8) = .empty;
    var i = start;
    while (i < end) : (i += 1) {
        // Skip `//` comments, which hold URLs and prose, not markers.
        if (src[i] == '/' and src[i + 1] == '/') {
            while (src[i] != '\n') i += 1;
            continue;
        }
        if (src[i] != '"') continue;
        var lit: std.ArrayList(u8) = .empty;
        i += 1;
        while (src[i] != '"') : (i += 1) {
            if (src[i] == '\\') {
                i += 1;
                lit.append(gpa, switch (src[i]) {
                    'n' => '\n',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    else => {
                        std.debug.print("FAIL: unhandled escape \\{c} in llama-chat.cpp\n", .{src[i]});
                        std.process.exit(1);
                    },
                }) catch unreachable;
            } else {
                lit.append(gpa, src[i]) catch unreachable;
            }
        }
        needles.append(gpa, lit.items) catch unreachable;
    }
    return needles.items;
}

// -----------------------------------------------------------------------------
// Checks

/// Which `llm_chat_template` values the reference's detection returned, for
/// the coverage line.
var detected: [64]usize = @splat(0);

fn checkDetect(what: []const u8, s: []const u8) void {
    checks += 1;
    const r = ref_detect(s.ptr, s.len);
    const p = @backingInt(chat.detectTemplate(s));
    if (r >= 0 and r < detected.len) detected[@intCast(r)] += 1;
    if (r != p) fail("detect [{s}] '{s}' (len {d}): ref {d} port {d}", .{ what, esc(s), s.len, r, p });
}

fn checkFromStr(s: []const u8) void {
    checks += 1;
    const r = ref_from_str(s.ptr, s.len);
    const p: c_int = if (chat.templateFromStr(s)) |t| @backingInt(t) else -1;
    if (r != p) fail("from_str '{s}': ref {d} port {d}", .{ esc(s), r, p });
}

fn describe(conv: []const Message) []const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (conv) |m| {
        const s = std.fmt.allocPrint(gpa, "[{s}: {s}] ", .{ esc(std.mem.span(m.role)), esc(std.mem.span(m.content)) }) catch unreachable;
        out.appendSlice(gpa, s) catch unreachable;
    }
    return out.items;
}

fn compareOutput(what: []const u8, conv: []const Message, add_ass: bool, r: i32, ref_out: []const u8, p: i32, port_out: []const u8) void {
    checks += 1;
    if (r != p or !std.mem.eql(u8, ref_out, port_out)) {
        const at = std.mem.findDiff(u8, ref_out, port_out) orelse @min(ref_out.len, port_out.len);
        const lo = at -| 24;
        fail("{s} add_ass={} on {s}\n    returned ref {d} port {d}; output differs at byte {d}:\n    ref  '{s}'\n    port '{s}'", .{
            what,                                  add_ass,                                 describe(conv), r, p, at,
            esc(ref_out[@min(lo, ref_out.len)..]), esc(port_out[@min(lo, port_out.len)..]),
        });
    }
}

var out_list: std.ArrayList(u8) = .empty;

fn checkApply(t: c_int, conv: []const Message, add_ass: bool) void {
    var ref_p: [*]u8 = undefined;
    var ref_len: usize = 0;
    const r = ref_apply(t, conv.ptr, conv.len, add_ass, "sentinel", &ref_p, &ref_len);
    defer ref_free_buf(ref_p);
    out_list.clearRetainingCapacity();
    out_list.appendSlice(staged.impl.allocator, "sentinel") catch unreachable;
    const p = chat.applyTemplate(@fromBackingInt(@intCast(t)), conv, &out_list, add_ass);
    var nb: [32]u8 = undefined;
    const what = std.fmt.bufPrint(&nb, "apply tmpl {d}", .{t}) catch unreachable;
    compareOutput(what, conv, add_ass, r, ref_p[0..ref_len], p, out_list.items);
}

/// `llama_chat_apply_template` against what its shim will do: detect, then
/// apply, -1 for an unknown template.
fn checkPublic(tmpl: [:0]const u8, conv: []const Message, add_ass: bool) void {
    const r = c.llama_chat_apply_template(tmpl.ptr, conv.ptr, conv.len, add_ass, null, 0);
    var ref_out: []u8 = &.{};
    if (r > 0) {
        ref_out = gpa.alloc(u8, @intCast(r + 1)) catch unreachable;
        const r2 = c.llama_chat_apply_template(tmpl.ptr, conv.ptr, conv.len, add_ass, ref_out.ptr, r + 1);
        std.debug.assert(r2 == r);
        ref_out = ref_out[0..@intCast(r)];
    }
    defer if (ref_out.len != 0) gpa.free(ref_out);
    out_list.clearRetainingCapacity();
    const t = chat.detectTemplate(tmpl);
    const p = if (t == .unknown) -1 else chat.applyTemplate(t, conv, &out_list, add_ass);
    const port_out: []const u8 = if (p > 0) out_list.items else &.{};
    var nb: [200]u8 = undefined;
    const what = std.fmt.bufPrint(&nb, "llama_chat_apply_template '{s}'", .{tmpl[0..@min(tmpl.len, 60)]}) catch unreachable;
    compareOutput(what, conv, add_ass, r, ref_out, p, port_out);
}

// -----------------------------------------------------------------------------
// Builtin names

fn checkBuiltins() [][:0]const u8 {
    const ref_n = c.llama_chat_builtin_templates(null, 0);
    const port_n = chat.builtinTemplates(null, 0);
    checks += 1;
    if (ref_n != port_n) fail("builtin count: ref {d} port {d}", .{ ref_n, port_n });
    const n: usize = @intCast(ref_n);

    const sentinel: [*:0]const u8 = "<untouched>";
    for ([_]usize{ 1, 5, n - 1, n, n + 1, 200 }) |len| {
        var ref_arr: [256][*c]const u8 = @splat(sentinel);
        var port_arr: [256]?[*:0]const u8 = @splat(sentinel);
        const r = c.llama_chat_builtin_templates(&ref_arr, len);
        const p = chat.builtinTemplates(&port_arr, len);
        checks += 1;
        if (r != p) fail("builtin len {d}: ref {d} port {d}", .{ len, r, p });
        for (0..ref_arr.len) |i| {
            checks += 1;
            const a = std.mem.span(@as([*:0]const u8, @ptrCast(ref_arr[i])));
            const b = std.mem.span(port_arr[i].?);
            if (!std.mem.eql(u8, a, b)) fail("builtin len {d} [{d}]: ref '{s}' port '{s}'", .{ len, i, a, b });
        }
    }

    var names: std.ArrayList([:0]const u8) = .empty;
    var arr: [256][*c]const u8 = undefined;
    _ = c.llama_chat_builtin_templates(&arr, n);
    for (arr[0..n]) |p| names.append(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(p)))) catch unreachable;
    return names.items;
}

fn checkNames(names: []const [:0]const u8) void {
    const extra = [_][]const u8{ "dots1", "unknown", "", "chatml\x00", " chatml", "ChatML", "mistral", "mistral-v2", "llama2-sys-strip " };
    for (extra) |s| {
        checkFromStr(s);
        checkDetect("name", s);
    }
    for (names) |name| {
        checkFromStr(name);
        checkDetect("name", name);
        for (0..name.len) |k| {
            checkFromStr(name[0..k]);
            checkDetect("name prefix", name[0..k]);
            checkFromStr(name[k + 1 ..]);
            checkDetect("name suffix", name[k + 1 ..]);
        }
        const upper = gpa.dupe(u8, name) catch unreachable;
        for (upper) |*b| b.* = std.ascii.toUpper(b.*);
        checkFromStr(upper);
        checkDetect("name upper", upper);
        const padded = std.fmt.allocPrint(gpa, "{s} ", .{name}) catch unreachable;
        checkFromStr(padded);
        checkDetect("name padded", padded);
    }
}

// -----------------------------------------------------------------------------
// Detection

fn checkNeedles(needles: []const []const u8, rand: std.Random) void {
    for (needles) |a| {
        checkDetect("needle", a);
        // Each byte dropped and each byte changed: a mistyped marker in the
        // port matches one of these where the C++'s does not.
        // Alone, and after every other marker, since most markers decide
        // only beside a companion (`<|assistant|>` with `<|end|>`).
        for (0..a.len) |k| {
            const dropped = std.mem.concat(gpa, u8, &.{ a[0..k], a[k + 1 ..] }) catch unreachable;
            const changed = gpa.dupe(u8, a) catch unreachable;
            changed[k] ^= 0x20;
            checkDetect("needle dropped byte", dropped);
            checkDetect("needle changed byte", changed);
            for (needles) |b| {
                const d2 = std.mem.concat(gpa, u8, &.{ b, " ", dropped }) catch unreachable;
                checkDetect("marker, needle dropped byte", d2);
                gpa.free(d2);
                const c2 = std.mem.concat(gpa, u8, &.{ b, " ", changed }) catch unreachable;
                checkDetect("marker, needle changed byte", c2);
                gpa.free(c2);
            }
        }
        for (needles) |b| {
            checkDetect("pair", std.mem.concat(gpa, u8, &.{ a, b }) catch unreachable);
            checkDetect("pair spaced", std.mem.concat(gpa, u8, &.{ a, " x ", b }) catch unreachable);
        }
    }
    const fillers = [_][]const u8{ "", " ", "x", "{{ ", " }}", "\n", "mistral", "'", "\"" };
    for (0..60_000) |_| {
        var s: std.ArrayList(u8) = .empty;
        const k = rand.intRangeAtMost(usize, 1, 7);
        for (0..k) |_| {
            s.appendSlice(gpa, fillers[rand.uintLessThan(usize, fillers.len)]) catch unreachable;
            s.appendSlice(gpa, needles[rand.uintLessThan(usize, needles.len)]) catch unreachable;
        }
        checkDetect("random markers", s.items);
        s.deinit(gpa);
    }
}

fn removeAll(s: []const u8, needle: []const u8) []const u8 {
    if (needle.len == 0) return s;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (std.mem.findPos(u8, s, i, needle)) |p| {
        out.appendSlice(gpa, s[i..p]) catch unreachable;
        i = p + needle.len;
    }
    out.appendSlice(gpa, s[i..]) catch unreachable;
    return out.items;
}

fn checkTemplateSource(name: []const u8, t: []const u8, needles: []const []const u8) void {
    checkDetect(name, t);
    for (needles) |nd| {
        var from: usize = 0;
        var seen: usize = 0;
        while (std.mem.findPos(u8, t, from, nd)) |p| : (from = p + 1) {
            seen += 1;
            if (seen > 40) break;
            const e = p + nd.len;
            checkDetect(name, t[0..e]);
            checkDetect(name, t[0 .. e - 1]);
            checkDetect(name, t[p..]);
            checkDetect(name, std.mem.concat(gpa, u8, &.{ t[0..p], t[e..] }) catch unreachable);
            if (nd.len == 0) break;
        }
        checkDetect(name, removeAll(t, nd));
    }
    // Peel the cascade: remove each marker in turn, in the C++'s order.
    var s = t;
    for (needles) |nd| {
        s = removeAll(s, nd);
        checkDetect(name, s);
    }
    var k: usize = 0;
    while (k < t.len) : (k += 97) checkDetect(name, t[0..k]);
}

fn templatesFromGguf(path: []const u8, out: *std.ArrayList([:0]const u8)) void {
    const pz = gpa.dupeSentinel(u8, path, 0) catch unreachable;
    const ctx = c.gguf_init_from_file(pz.ptr, .{ .no_alloc = true, .ctx = null }) orelse {
        fail("{s}: gguf_init_from_file failed", .{path});
        return;
    };
    defer c.gguf_free(ctx);
    const n: usize = @intCast(c.gguf_get_n_kv(ctx));
    for (0..n) |i| {
        const key = std.mem.span(c.gguf_get_key(ctx, @intCast(i)));
        if (!std.mem.startsWith(u8, key, "tokenizer.chat_template")) continue;
        if (c.gguf_get_kv_type(ctx, @intCast(i)) != c.GGUF_TYPE_STRING) continue;
        const v = std.mem.span(c.gguf_get_val_str(ctx, @intCast(i)));
        out.append(gpa, gpa.dupeSentinel(u8, v, 0) catch unreachable) catch unreachable;
    }
}

// -----------------------------------------------------------------------------
// Conversations

const roles = [_][:0]const u8{
    "system", "user",  "assistant", "tool",   "function", "assistant_tool_call",
    "model",  "",      "System",    "USER",   "bot",      "\xc3\xbcn\xc3\xaf",
    "\xe9t",  "a b c", "human",     "Mixed1",
};

const contents = [_][:0]const u8{
    "",
    "Hello",
    "  padded  ",
    "\t\n\x0b\x0c\r x \r\n",
    " ",
    "\n\n",
    "<|im_start|>user\nhi<|im_end|>",
    "[INST] x [/INST]</s>",
    "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88",
    "<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>",
    "\xa0nbsp\xa0",
    "\x85nel\x85",
    "line1\nline2",
    "%s %d %n",
    "{{ messages }}",
    "<s>assistant\n",
};

fn conversation(spec: []const [2][:0]const u8) []Message {
    const out = gpa.alloc(Message, spec.len) catch unreachable;
    for (spec, out) |s, *m| m.* = .{ .role = s[0].ptr, .content = s[1].ptr };
    return out;
}

/// Every byte at both edges of a message, for `trim`'s `isspace`, and as a
/// role's first and last byte and as a one-byte role, for `toupper`.
fn byteSweep() [][]Message {
    var convs: std.ArrayList([]Message) = .empty;
    for (1..256) |b| {
        const by: u8 = @intCast(b);
        const edged = gpa.dupeSentinel(u8, &.{ by, 'x', by }, 0) catch unreachable;
        const role1 = gpa.dupeSentinel(u8, &.{ by, 'o', 'l', 'e', by }, 0) catch unreachable;
        const role0 = gpa.dupeSentinel(u8, &.{by}, 0) catch unreachable;
        convs.append(gpa, conversation(&.{ .{ "system", edged }, .{ "user", edged }, .{ "assistant", edged }, .{ "tool", edged } })) catch unreachable;
        convs.append(gpa, conversation(&.{ .{ role1, "c" }, .{ role0, edged } })) catch unreachable;
    }
    return convs.items;
}

fn buildConversations(rand: std.Random) [][]Message {
    var convs: std.ArrayList([]Message) = .empty;
    const fixed = [_][]const [2][:0]const u8{
        &.{},
        &.{.{ "user", "Hello" }},
        &.{.{ "system", "You are helpful." }},
        &.{.{ "assistant", " starts " }},
        &.{ .{ "system", "  sys  " }, .{ "user", "hi" } },
        &.{ .{ "user", "hi" }, .{ "assistant", "  hello  " } },
        &.{ .{ "system", "s" }, .{ "user", "u1" }, .{ "assistant", "a1" }, .{ "user", "u2" } },
        &.{ .{ "system", "s1" }, .{ "system", "s2" }, .{ "user", "u" } },
        &.{ .{ "user", "u1" }, .{ "user", "u2" }, .{ "assistant", "a" } },
        &.{ .{ "system", "s" }, .{ "assistant", "a" }, .{ "user", "u" } },
        &.{ .{ "tool", "t" }, .{ "user", "u" }, .{ "function", "f" }, .{ "assistant_tool_call", "c" } },
        &.{ .{ "user", "u" }, .{ "system", "late system" }, .{ "user", "u2" } },
        &.{ .{ "", "" }, .{ "", "" } },
        &.{ .{ "model", "m" }, .{ "user", "u" } },
        &.{ .{ "system", "" }, .{ "user", "" }, .{ "assistant", "" } },
    };
    for (fixed) |f| convs.append(gpa, conversation(f)) catch unreachable;

    convs.appendSlice(gpa, byteSweep()) catch unreachable;

    for (0..2500) |_| {
        const n = rand.uintLessThan(usize, 8);
        const conv = gpa.alloc(Message, n) catch unreachable;
        for (conv) |*m| {
            const r = if (rand.boolean()) roles[rand.uintLessThan(usize, 4)] else roles[rand.uintLessThan(usize, roles.len)];
            m.* = .{ .role = r.ptr, .content = contents[rand.uintLessThan(usize, contents.len)].ptr };
        }
        convs.append(gpa, conv) catch unreachable;
    }
    return convs.items;
}

// -----------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: chat_diff <llama-chat.cpp> <file.gguf|file.jinja>...\n", .{});
        std.process.exit(2);
    }
    var prng = std.Random.DefaultPrng.init(0xc4a7);
    const rand = prng.random();

    std.debug.print("  builtin template names\n", .{});
    const names = checkBuiltins();
    checkNames(names);

    const needles = extractNeedles(readFile(args[1]));
    std.debug.print("  {d} markers from llm_chat_detect_template\n", .{needles.len});
    if (needles.len < 70) {
        std.debug.print("FAIL: only {d} markers extracted; the extraction is broken\n", .{needles.len});
        std.process.exit(1);
    }
    checkNeedles(needles, rand);

    var sources: std.ArrayList([:0]const u8) = .empty;
    for (args[2..]) |path| {
        if (std.mem.endsWith(u8, path, ".gguf")) {
            templatesFromGguf(path, &sources);
        } else {
            sources.append(gpa, gpa.dupeSentinel(u8, readFile(path), 0) catch unreachable) catch unreachable;
        }
    }
    std.debug.print("  {d} template sources from {d} files\n", .{ sources.items.len, args.len - 2 });
    for (sources.items, 0..) |t, i| {
        var nb: [32]u8 = undefined;
        checkTemplateSource(std.fmt.bufPrint(&nb, "source {d}", .{i}) catch unreachable, t, needles);
    }

    var reached: usize = 0;
    for (detected[0..56]) |n| reached += @intFromBool(n != 0);
    std.debug.print("  detection reached {d} of 56 llm_chat_template values\n", .{reached});
    checks += 1;
    if (reached != 56) {
        var missing: std.ArrayList(u8) = .empty;
        for (detected[0..56], 0..) |n, i| if (n == 0) missing.appendSlice(gpa, std.fmt.allocPrint(gpa, " {d}", .{i}) catch unreachable) catch unreachable;
        fail("detection never returned:{s}", .{missing.items});
    }

    const convs = buildConversations(rand);
    std.debug.print("  {d} conversations through every template\n", .{convs.len});
    for (convs) |conv| {
        for (0..56) |t| {
            checkApply(@intCast(t), conv, false);
            checkApply(@intCast(t), conv, true);
        }
    }

    // `isspace` and `toupper` depend on the locale, and the C locale is
    // ASCII-only, so it cannot tell libc's answer from a hand-written ASCII
    // one. These can: under UTF-8, `isspace(0xa0)` holds and `toupper`
    // maps 0xe0-0xfe -- which the C++ never reaches for a role, because it
    // passes a signed `char`. A missing locale is reported, not failed.
    for (locales) |loc| {
        if (setlocale(LC_ALL, loc) == null) {
            std.debug.print("  locale {s} not installed; its sweep is skipped\n", .{loc});
            continue;
        }
        std.debug.print("  byte sweep under {s}\n", .{loc});
        for (byteSweep()) |conv| {
            for (0..56) |t| {
                checkApply(@intCast(t), conv, false);
                checkApply(@intCast(t), conv, true);
            }
        }
    }
    _ = setlocale(LC_ALL, "C");

    std.debug.print("  llama_chat_apply_template\n", .{});
    for (convs[0..300]) |conv| {
        for (names) |name| {
            checkPublic(name, conv, false);
            checkPublic(name, conv, true);
        }
        for (sources.items) |t| checkPublic(t, conv, true);
    }

    if (failures != 0) {
        std.debug.print("FAIL: {d} of {d} chat results differ\n", .{ failures, checks });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} chat results identical, Zig vs llama-chat.cpp\n", .{checks});
}
