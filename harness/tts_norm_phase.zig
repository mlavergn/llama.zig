//! `make jinja-diff`'s TTS norm phase: `cli/common_chat/tts_norm.zig` against
//! upstream's `common/chat-tts-norm.cpp`, compiled unmodified beside the
//! engine. Three parts, compared on bytes:
//!
//! - `common_tts_norm_sub` over random patterns, texts and replacements --
//!   the result, or the `std::runtime_error` text;
//! - `common_tts_norm_scan` over one to three random patterns -- every
//!   segment's rule, byte offsets and groups, or the error text;
//! - task templates rendered as `common_chat_template_render` renders one,
//!   with the two builtins: upstream's own `test-jinja.cpp` cases, and random
//!   rule tables as a string, a dict or a list, with arguments of the wrong
//!   count and type -- the text and each byte's input mark, with the variables
//!   bound unmarked (the task-template path) and marked; a render that raises
//!   must raise on both sides, the engine's message being its own.
//!
//! **Not a port.** The generators aim at every arm of the pattern parser and
//! matcher: every class and its negation, `%b`, anchors, sets with ranges and
//! classes, every group kind, lookaround, `(?i)`, each quantifier form, and
//! the malformed spellings each error names.

const std = @import("std");
const common_chat = @import("common_chat");
const json_mod = @import("json");

extern fn tts_norm_ref_sub(text: [*]const u8, text_len: usize, pat: [*]const u8, pat_len: usize, repl: [*]const u8, repl_len: usize, out: *?[*]u8, out_len: *usize) c_int;
extern fn tts_norm_ref_scan(text: [*]const u8, text_len: usize, pats: [*]const [*]const u8, pat_lens: [*]const usize, n: usize, out: *?[*]u8, out_len: *usize) c_int;
extern fn tts_norm_ref_render(tmpl: [*]const u8, tmpl_len: usize, vars: [*]const u8, vars_len: usize, mark_input: c_int, out: *?[*]u8, out_len: *usize, parts: *?[*]u8) c_int;
extern fn jinja_ref_free(p: ?*anyopaque) void;

const tts_norm = common_chat.tts_norm;
const gpa = std.heap.c_allocator;

/// What the phase found.
pub const Result = struct {
    checks: usize = 0,
    differ: usize = 0,
    /// matches where upstream threw, so the error arms are known to be reached
    ref_errors: usize = 0,
    /// matches where some segment matched a rule
    matched: usize = 0,
};

var shown: usize = 0;
const show_max = 12;

fn report(kind: []const u8, what: []const u8, ref: []const u8, port: []const u8) void {
    if (shown >= show_max) return;
    shown += 1;
    std.debug.print("  tts-norm {s}: {s}\n      ref  {s}\n      port {s}\n", .{ kind, what, ref, port });
}

// -----------------------------------------------------------------------------
// generators

const text_pool = [_][]const u8{
    "a",        "b",        "Z",        "x",         "0",        "1",  "5",  "9",   " ",  " ",  "\t", "\n",   "-",     ":", ".", ",", "%",    "$",
    "€",
    "£",
    "é",
    "ß",
    "你",
    "好",
    "\u{a0}",   "\u{200b}", "\u{3000}", "\u{1f600}", "\u{fffd}", "(",  ")",  "[",   "]",  "{",  "}",  "?",    "*",     "+", "|", "'", "\xff", "\xc3",
    "\xe4\xbd", "Dr",       "dr",       "st",        "nd",       "rd", "th", "GPT", "AB", "ab", "12", "3.50", "10:45",
};

fn genText(r: std.Random, out: *std.ArrayList(u8)) !void {
    const n = r.uintLessThan(usize, 24);
    for (0..n) |_| try out.appendSlice(gpa, text_pool[r.uintLessThan(usize, text_pool.len)]);
}

const atoms = [_][]const u8{
    "a",         "b",       "x",     "1",        "0",     " ",       "-",       ":",
    "é",
    "你",
    "€",
    "%.",        "%%",      "%-",    "%(",       "%)",    "%[",      "%]",      "%{",
    "%}",        "%$",      "%d",    "%s",       "%p",    "%a",      "%w",      "%u",
    "%l",        "%h",      "%c",    "%D",       "%S",    "%P",      "%A",      "%W",
    "%U",        "%L",      "%H",    "%C",       "%b",    ".",       "^",       "$",
    "[abc]",     "[^%d]",   "[a-z]", "[0-9a-f]",
    "[$€£]",
    "[%s%p]",    "[^%w%s]", "[-a]",  "[a-]",     "[%%x]", "(?<=%d)", "(?<!%w)", "(?<=.)",
    "(?<![ab])",
};

const quants = [_][]const u8{ "", "", "", "?", "*", "+", "{2}", "{1,}", "{0,3}", "{2,4}" };

const bad = [_][]const u8{ ")", "]", "}", "%q", "%B", "%", "(?<n>a)", "{2,1}", "++", "a{", "[", "[]", "(?", "(?i)", "*", "(?<=ab)", "(?<=)", "[z-a]", "[%d-z]", "{1001}", "%b+", "^*" };

fn genSeq(r: std.Random, out: *std.ArrayList(u8), depth: usize) std.mem.Allocator.Error!void {
    const n = 1 + r.uintLessThan(usize, 4);
    for (0..n) |_| {
        const roll = r.uintLessThan(usize, 10);
        if (roll < 2 and depth < 3) {
            const opens = [_][]const u8{ "(", "(?:", "(?=", "(?!" };
            const open = opens[r.uintLessThan(usize, opens.len)];
            try out.appendSlice(gpa, open);
            try genAlts(r, out, depth + 1);
            try out.append(gpa, ')');
            // lookahead cannot be repeated
            if (open.len < 3 or open[2] == ':') try out.appendSlice(gpa, quants[r.uintLessThan(usize, quants.len)]);
        } else {
            const atom = atoms[r.uintLessThan(usize, atoms.len)];
            try out.appendSlice(gpa, atom);
            const zero_width = std.mem.eql(u8, atom, "%b") or std.mem.eql(u8, atom, "^") or std.mem.eql(u8, atom, "$") or std.mem.startsWith(u8, atom, "(?<");
            if (!zero_width) try out.appendSlice(gpa, quants[r.uintLessThan(usize, quants.len)]);
        }
    }
}

fn genAlts(r: std.Random, out: *std.ArrayList(u8), depth: usize) std.mem.Allocator.Error!void {
    try genSeq(r, out, depth);
    while (r.uintLessThan(usize, 4) == 0) {
        try out.append(gpa, '|');
        try genSeq(r, out, depth);
    }
}

fn genPattern(r: std.Random, out: *std.ArrayList(u8)) !void {
    if (r.uintLessThan(usize, 6) == 0) try out.appendSlice(gpa, "(?i)");
    try genAlts(r, out, 0);
    // a malformed one now and then, spliced in at a random place
    if (r.uintLessThan(usize, 5) == 0) {
        const at = r.uintLessThan(usize, out.items.len + 1);
        try out.insertSlice(gpa, at, bad[r.uintLessThan(usize, bad.len)]);
    }
}

fn genRepl(r: std.Random, out: *std.ArrayList(u8)) !void {
    const pieces = [_][]const u8{ "X", "<", ">", " ", "%0", "%1", "%2", "%3", "%9", "%%", "é", "minus " };
    const n = r.uintLessThan(usize, 5);
    for (0..n) |_| try out.appendSlice(gpa, pieces[r.uintLessThan(usize, pieces.len)]);
    if (r.uintLessThan(usize, 12) == 0) try out.appendSlice(gpa, if (r.boolean()) "%" else "%x");
}

// -----------------------------------------------------------------------------
// sub and scan

fn checkSub(res: *Result, text: []const u8, pat: []const u8, repl: []const u8) !void {
    var out: ?[*]u8 = null;
    var out_len: usize = 0;
    const rc = tts_norm_ref_sub(text.ptr, text.len, pat.ptr, pat.len, repl.ptr, repl.len, &out, &out_len);
    defer jinja_ref_free(out);
    const ref = out.?[0..out_len];

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var port_rc: c_int = 0;
    const port: []const u8 = blk: {
        const p = tts_norm.Pattern.init(a, pat) catch |e| {
            port_rc = 1;
            break :blk try a.dupe(u8, common_chat.exception.what(e));
        };
        break :blk tts_norm.sub(a, text, &p, repl) catch |e| {
            port_rc = 1;
            break :blk try a.dupe(u8, common_chat.exception.what(e));
        };
    };
    res.checks += 1;
    if (rc != 0) res.ref_errors += 1;
    if (rc != port_rc or !std.mem.eql(u8, ref, port)) {
        res.differ += 1;
        const what = try std.fmt.allocPrint(a, "sub({f}, {f}, {f})", .{ std.zig.fmtString(text), std.zig.fmtString(pat), std.zig.fmtString(repl) });
        report("sub", what, try std.fmt.allocPrint(a, "{d} {f}", .{ rc, std.zig.fmtString(ref) }), try std.fmt.allocPrint(a, "{d} {f}", .{ port_rc, std.zig.fmtString(port) }));
    }
}

fn checkScan(res: *Result, text: []const u8, pats: []const []const u8) !void {
    var ptrs: [4][*]const u8 = undefined;
    var lens: [4]usize = undefined;
    for (pats, 0..) |p, i| {
        ptrs[i] = p.ptr;
        lens[i] = p.len;
    }
    var out: ?[*]u8 = null;
    var out_len: usize = 0;
    const rc = tts_norm_ref_scan(text.ptr, text.len, &ptrs, &lens, pats.len, &out, &out_len);
    defer jinja_ref_free(out);
    const ref = out.?[0..out_len];

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var port_rc: c_int = 0;
    var port: std.ArrayList(u8) = .empty;
    blk: {
        const patterns = try a.alloc(tts_norm.Pattern, pats.len);
        const pp = try a.alloc(*const tts_norm.Pattern, pats.len);
        for (pats, 0..) |p, i| {
            patterns[i] = tts_norm.Pattern.init(a, p) catch |e| {
                port_rc = 1;
                try port.appendSlice(a, common_chat.exception.what(e));
                break :blk;
            };
            pp[i] = &patterns[i];
        }
        const segs = tts_norm.scan(a, text, pp) catch |e| {
            port_rc = 1;
            try port.appendSlice(a, common_chat.exception.what(e));
            break :blk;
        };
        for (segs) |seg| {
            if (seg.rule >= 0) res.matched += 1;
            try port.print(a, "{d} {d} {d}", .{ seg.rule, seg.start, seg.end });
            for (seg.groups) |g| {
                if (g) |x| try port.print(a, " {d}:{d}", .{ x[0], x[1] }) else try port.appendSlice(a, " -");
            }
            try port.append(a, '\n');
        }
    }
    res.checks += 1;
    if (rc != 0) res.ref_errors += 1;
    if (rc != port_rc or !std.mem.eql(u8, ref, port.items)) {
        res.differ += 1;
        var what: std.ArrayList(u8) = .empty;
        try what.print(a, "scan({f},", .{std.zig.fmtString(text)});
        for (pats) |p| try what.print(a, " {f}", .{std.zig.fmtString(p)});
        try what.append(a, ')');
        report("scan", what.items, try std.fmt.allocPrint(a, "{d} {f}", .{ rc, std.zig.fmtString(ref) }), try std.fmt.allocPrint(a, "{d} {f}", .{ port_rc, std.zig.fmtString(port.items) }));
    }
}

// -----------------------------------------------------------------------------
// task templates

const RenderOutcome = union(enum) {
    /// `mask`: per byte, 1 where it is input
    ok: struct { text: []const u8, mask: []const u8 },
    failed: []const u8,
};

fn refRender(a: std.mem.Allocator, source: []const u8, vars: []const u8, mark: bool) !RenderOutcome {
    var out: ?[*]u8 = null;
    var out_len: usize = 0;
    var parts: ?[*]u8 = null;
    const rc = tts_norm_ref_render(source.ptr, source.len, vars.ptr, vars.len, @intFromBool(mark), &out, &out_len, &parts);
    defer jinja_ref_free(out);
    defer jinja_ref_free(parts);
    if (rc != 0) return .{ .failed = try a.dupe(u8, std.mem.span(@as([*:0]u8, @ptrCast(out.?)))) };
    var mask: std.ArrayList(u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(@as([*:0]u8, @ptrCast(parts.?))), ',');
    while (it.next()) |p| try mask.appendNTimes(a, @intFromBool(p[0] == 'I'), try std.fmt.parseInt(usize, p[1..], 10));
    return .{ .ok = .{ .text = try a.dupe(u8, out.?[0..out_len]), .mask = mask.items } };
}

fn portRender(a: std.mem.Allocator, source: []const u8, vars: []const u8, mark: bool) !RenderOutcome {
    var tmpl = common_chat.template.Template.init(gpa, source, "", "") catch |e| {
        if (e == error.OutOfMemory) return e;
        return .{ .failed = try a.dupe(u8, common_chat.exception.what(e)) };
    };
    defer tmpl.deinit();
    var v = try json_mod.parse(a, vars);
    var r: common_chat.template.Render = undefined;
    r.init(gpa, false);
    defer r.deinit();
    try tts_norm.addBuiltins(r.alloc(), &r, common_chat.template.Render.set);
    try r.bindJson(&v, mark);
    const prompt = r.runPrompt(&tmpl, a) catch |e| {
        if (e == error.OutOfMemory) return e;
        return .{ .failed = try a.dupe(u8, common_chat.exception.what(e)) };
    };
    var text: std.ArrayList(u8) = .empty;
    var mask: std.ArrayList(u8) = .empty;
    for (prompt.parts.items) |p| {
        try text.appendSlice(a, p.text);
        try mask.appendNTimes(a, @intFromBool(p.is_input), p.text.len);
    }
    return .{ .ok = .{ .text = text.items, .mask = mask.items } };
}

fn checkRender(res: *Result, source: []const u8, vars: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for ([_]bool{ false, true }) |mark| {
        const ref = try refRender(a, source, vars, mark);
        const port = try portRender(a, source, vars, mark);
        res.checks += 1;
        const same = switch (ref) {
            .failed => blk: {
                res.ref_errors += 1;
                break :blk port == .failed;
            },
            .ok => |rr| port == .ok and std.mem.eql(u8, rr.text, port.ok.text) and std.mem.eql(u8, rr.mask, port.ok.mask),
        };
        if (!same) {
            res.differ += 1;
            const what = try std.fmt.allocPrint(a, "render{s} {f} over {s}", .{ if (mark) " (marked)" else "", std.zig.fmtString(source), vars });
            const show = struct {
                fn f(al: std.mem.Allocator, o: RenderOutcome) ![]const u8 {
                    return switch (o) {
                        .failed => |m| std.fmt.allocPrint(al, "raises: {s}", .{m}),
                        .ok => |x| std.fmt.allocPrint(al, "{f} mask {any}", .{ std.zig.fmtString(x.text), x.mask }),
                    };
                }
            }.f;
            report("render", what, try show(a, ref), try show(a, port));
        }
    }
}

/// tests/test-jinja.cpp's `test_tts_norm_patterns`, each with its variables.
const upstream_cases = [_][2][]const u8{
    .{ "{{ tts_norm_scan is defined }}", "{}" },
    .{ "{%- for s in tts_norm_scan(text, [['num', '%d+'], ['word', '%a+']]) -%}[{{ s.rule or '-' }}:{{ s.text }}]{%- endfor -%}", "{\"text\":\"abc 123, x9\"}" },
    .{ "{%- for s in tts_norm_scan(text, [['money', '%$(%d+)(?:%.(%d%d))?']]) -%}{%- if s.rule -%}{%- set whole, cents = s.groups -%}{{ whole }}|{{ cents }}|{{ s.groups | length }};{%- endif -%}{%- endfor -%}", "{\"text\":\"$3.50 and $7\"}" },
    .{ "{%- for s in tts_norm_scan(text, {'ordinal': '%d+(?:st|nd|rd|th)%b', 'number': '%d+'}) -%}{%- if s.rule -%}<{{ s.rule }}:{{ s.text }}>{%- else -%}{{ s.text }}{%- endif -%}{%- endfor -%}", "{\"text\":\"the 3rd of 10 or 4three\"}" },
    .{ "{%- for s in tts_norm_scan(text, '(?:a|ab)c') -%}[{{ s.rule or '-' }}:{{ s.text }}]{%- endfor -%}", "{\"text\":\"abc ac\"}" },
    .{ "{{ tts_norm_sub(text, '%d+0', 'X') }}|{{ tts_norm_sub(text, '%d+', 'X') }}", "{\"text\":\"100 20\"}" },
    .{ "{{ tts_norm_sub(text, '%d{3}-%d{2,}', '#') }}", "{\"text\":\"123-4 123-45 123-456789\"}" },
    .{ "{{ tts_norm_sub(text, '(?<!%w)-(%d+)', 'minus %1') }}", "{\"text\":\"GPT-5 is -5, (-3)\"}" },
    .{ "{{ tts_norm_sub(text, '(%d%d?):(%d%d)(?!%d)', '%1h%2') }}", "{\"text\":\"at 10:30 or 1:234\"}" },
    .{ "{{ tts_norm_sub(text, '(%d+)%%', '%1 percent (%0) 100%%') }}", "{\"text\":\"up 50%\"}" },
    .{ "{{ tts_norm_sub(tts_norm_sub(text, '(%a)(%d)', '%1 %2'), '(%d)(%a)', '%1 %2') }}", "{\"text\":\"R2D2 café2 4K\"}" },
    .{ "{{ tts_norm_sub(text, '(?i)%bdr%.', 'doctor') }}", "{\"text\":\"Dr. Who and dr. No, not Mdr.\"}" },
    .{ "{{ tts_norm_sub(text, '[$€£]%s*(%d+)', '<%1>') }}|{{ tts_norm_sub(text, '[^%w%s]', '') }}", "{\"text\":\"€ 5 & $6!\"}" },
    .{ "{{ tts_norm_sub(text, '(%h) +(%h)', '%1%2') }}", "{\"text\":\"你 好 world 世界\"}" },
    .{ "{{ tts_norm_scan('x', '(a') }}", "{}" },
    .{ "{{ tts_norm_scan('x', 'a)') }}", "{}" },
    .{ "{{ tts_norm_scan('x', 1) }}", "{}" },
    .{ "{{ tts_norm_sub('x', '(a)', '%2') }}", "{}" },
};

/// Rule tables and calls the random templates draw from: valid and broken.
const scan_rules = [_][]const u8{
    "'%d+'",                           "[['n', '%d+'], ['w', '%a+']]", "{'n': '%d+', 'w': '%a+', 's': '%s+'}", "[['t', '(%d%d?):(%d%d)(?!%d)']]",
    "[['m', '%$(%d+)(?:%.(%d%d))?']]", "{'p': '(%d+)%%'}",             "[['n', 1]]",                           "[['n']]",
    "{'n': 2}",                        "[]",                           "{}",                                   "none",
    "['%d+']",                         "[['bad', '(a']]",              "rules",                                "[['h', '(%h) +(%h)'], ['neg', '(?<!%w)-(%d+)']]",
};

fn genRenderTemplate(r: std.Random, a: std.mem.Allocator) ![]const u8 {
    const kind = r.uintLessThan(usize, 6);
    const rules = scan_rules[r.uintLessThan(usize, scan_rules.len)];
    return switch (kind) {
        0, 1 => std.fmt.allocPrint(a, "{{%- for s in tts_norm_scan(text, {s}) -%}}[{{{{ s.rule }}}}|{{{{ s.text }}}}|{{{{ s.groups | join(',') }}}}]{{%- endfor -%}}", .{rules}),
        2 => std.fmt.allocPrint(a, "{{{{ tts_norm_sub(text, pattern, '<%1>') }}}}", .{}),
        3 => std.fmt.allocPrint(a, "{{{{ tts_norm_sub(text, '%d+') }}}}", .{}),
        4 => std.fmt.allocPrint(a, "{{{{ tts_norm_scan(text) | length }}}}|{{{{ tts_norm_sub(1, '%d', 'x') }}}}", .{}),
        else => std.fmt.allocPrint(a, "{{%- for s in tts_norm_scan(text, {s}) -%}}{{{{ tts_norm_sub(s.text, '(%a)', '[%1]') }}}}{{%- endfor -%}}", .{rules}),
    };
}

/// Runs the phase.
pub fn run(seed: u64, n_random: usize) !Result {
    var res: Result = .{};
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    var pat: std.ArrayList(u8) = .empty;
    defer pat.deinit(gpa);
    var repl: std.ArrayList(u8) = .empty;
    defer repl.deinit(gpa);
    var pats: [3]std.ArrayList(u8) = .{ .empty, .empty, .empty };
    defer for (&pats) |*p| p.deinit(gpa);

    for (0..n_random) |_| {
        text.clearRetainingCapacity();
        pat.clearRetainingCapacity();
        repl.clearRetainingCapacity();
        try genText(r, &text);
        try genPattern(r, &pat);
        try genRepl(r, &repl);
        try checkSub(&res, text.items, pat.items, repl.items);

        const n = 1 + r.uintLessThan(usize, 3);
        var slices: [3][]const u8 = undefined;
        for (0..n) |i| {
            pats[i].clearRetainingCapacity();
            try genPattern(r, &pats[i]);
            slices[i] = pats[i].items;
        }
        try checkScan(&res, text.items, slices[0..n]);
    }
    // a pattern past the length limit
    {
        const long = try gpa.alloc(u8, 4097);
        defer gpa.free(long);
        @memset(long, 'a');
        try checkSub(&res, "aaa", long, "x");
        try checkSub(&res, "aaa", long[0..4096], "x");
    }

    for (upstream_cases) |cs| try checkRender(&res, cs[0], cs[1]);
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for (0..n_random / 8) |_| {
        text.clearRetainingCapacity();
        pat.clearRetainingCapacity();
        try genText(r, &text);
        try genPattern(r, &pat);
        if (!std.unicode.utf8ValidateSlice(text.items)) continue; // the variables travel as JSON
        var vars: std.Io.Writer.Allocating = .init(a);
        var js: std.json.Stringify = .{ .writer = &vars.writer };
        try js.beginObject();
        try js.objectField("text");
        try js.write(text.items);
        try js.objectField("pattern");
        try js.write(if (std.unicode.utf8ValidateSlice(pat.items)) pat.items else "%d");
        try js.objectField("rules");
        try js.write("%a+");
        try js.endObject();
        try checkRender(&res, try genRenderTemplate(r, a), vars.written());
    }
    return res;
}
