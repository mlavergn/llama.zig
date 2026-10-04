//! Check that every `Ports X (file:line)` reference in the port still points
//! at the C it names, and that every file carrying one declares the commit it
//! was read at.
//!
//! # Provenance
//!
//! **Not a port.** This is our own code.
//!
//! # What it enforces
//!
//! Every ported declaration carries a line like
//!
//! ```
//! /// Ports `ggml_hash_set_new` (ggml.c:6516).
//! ```
//!
//! naming the C function it replaces, the file it came from, and the line that
//! definition starts on. Those references are the map from this port back to
//! upstream: when the pin moves, they are what tells you which of our
//! functions a given upstream change touches. **A stale line number is worse
//! than no line number**, because it sends you to the wrong function with no
//! indication that it did.
//!
//! So this exists. It resolves every reference and requires the named symbol
//! to appear on exactly that line. 216 of the 790 were wrong when it was first
//! run -- drifted by a handful of lines each, two pointing at the wrong file
//! entirely -- which is what a convention does when nothing checks it.
//!
//! # The commit, on every citation
//!
//! A line number means nothing without the revision it was read at, so each
//! citation carries its own:
//!
//! ```
//! /// Ports `ggml_hash_set_new` (ggml.c:6516 @c1d0e7a00).
//! ```
//!
//! **Per citation, not per file.** An upstream sync does not move a whole file
//! at once -- one function gets re-read against a newer commit while its forty
//! neighbours stay where they were, which is exactly what a diff produces. A
//! single commit declared in the module header could not express that, and
//! would be lying about the other forty the moment anyone did it.
//!
//! So the checker reads each cited file **as it was at that citation's own
//! commit**, via `git show <sha>:<path>`, and a file may cite as many commits
//! as it needs to. The working tree is used as a fast path only when the
//! citation is at the checkout's `HEAD`.
//!
//! # The rule the references follow
//!
//! **A reference points at the first line of the symbol's definition.** For a
//! function or a macro the name is on that line and the check is direct. For
//! an anonymous `typedef struct { ... } name;` the name appears only on the
//! closing line, so a citation of the opening `typedef struct {` is accepted
//! when the brace it opens closes on a line naming the symbol -- which keeps
//! citations pointing at where a definition *starts*, where a diff lands.
//!
//! Where a symbol has several definitions behind `#if` arms -- the four
//! `ggml_thread_apply_affinity`s in `ggml-cpu.c`, say -- the reference points
//! at the arm this target compiles, and the surrounding doc comment says which
//! arm that is.
//!
//! A citation may name several symbols at once, `(ggml-impl.h:173, 179, 185,
//! 191)`, in which case the symbols and the line numbers pair up in order.
//!
//! # Usage
//!
//! Driven by `scripts/port-links`, which supplies the checkout's `HEAD`:
//!
//! ```
//! zig run harness/port_links.zig -- <port-dir>... -- <reference-root> [sha]
//! ```

const std = @import("std");

/// One `Ports X (file:line)` reference, as found in our source.
const Reference = struct {
    /// The file it was found in, for the error message.
    from: []const u8,
    /// The line it was found on, for the error message.
    from_line: usize,
    /// The C symbol named, with any `struct `/`enum ` prefix stripped, or an
    /// empty slice when the citation names none and only the line is checked.
    symbol: []const u8,
    /// The reference file as written, e.g. `ggml.c` or `arch/arm/quants.c`.
    file: []const u8,
    /// The line claimed. Zero means the citation was malformed.
    line: usize,
    /// The upstream commit the line was read at, empty for a citation of our
    /// own `harness/`.
    sha: []const u8,
};

/// Extensions that can hold a definition worth citing.
fn isSource(name: []const u8) bool {
    for ([_][]const u8{ ".c", ".h", ".cpp", ".hpp" }) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}

/// Resolves a reference like `arch/arm/quants.c` to a path under the reference
/// tree.
///
/// **The match must be unique.** Two files named `common.h` exist -- one under
/// `common/`, one under `ggml-cpu/` -- and `quants.c` exists eight times, once
/// per architecture. Picking the shortest path silently sent a
/// `GGML_FA_TILE_Q` reference into the wrong `common.h`, so the rule is that
/// an ambiguous reference is an error and the comment has to carry enough path
/// to disambiguate.
///
/// Parameters:
/// - `paths`: every source file under the reference tree.
/// - `want`: the reference as written.
///
/// Return: the resolved path, or null when nothing or more than one matches.
fn resolve(paths: []const []const u8, want: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    for (paths) |p| {
        if (p.len <= want.len) continue;
        if (!std.mem.endsWith(u8, p, want)) continue;
        // The character before the suffix must be a separator, so `quants.c`
        // does not match `arm-quants.c`.
        if (p[p.len - want.len - 1] != '/') continue;
        if (found != null) return null;
        found = p;
    }
    return found;
}

/// Whether `s` could be a C identifier, so prose in backticks is not mistaken
/// for a symbol.
fn isIdentifier(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.ascii.isDigit(s[0])) return false;
    for (s) |ch| if (!isIdent(ch)) return false;
    return true;
}

/// Appends every backticked identifier in `text` to `out`, in order.
///
/// `struct foo` and `enum foo` cite the tag, not the keyword. Backticked prose
/// and wildcard families like `ggml_vec_set_*` are skipped: they name no one
/// symbol, so there is nothing to look for.
fn collectSymbols(
    arena: std.mem.Allocator,
    out: *std.ArrayList([]const u8),
    text: []const u8,
) !void {
    var rest = text;
    while (std.mem.indexOfScalar(u8, rest, '`')) |open| {
        rest = rest[open + 1 ..];
        const close = std.mem.indexOfScalar(u8, rest, '`') orelse return;
        var sym = rest[0..close];
        rest = rest[close + 1 ..];

        inline for ([_][]const u8{ "struct ", "enum ", "union " }) |kw| {
            if (std.mem.startsWith(u8, sym, kw)) sym = sym[kw.len..];
        }
        if (!isIdentifier(sym)) continue;
        try out.append(arena, sym);
    }
}

/// One parsed citation: `(ggml.c:6516 @c1d0e7a00)`.
const Citation = struct {
    /// Offset of the opening paren, so the symbols before it can be read.
    at: usize,
    /// The reference file as written, e.g. `ggml.c`.
    file: []const u8,
    /// The comma-separated line numbers, unparsed.
    nums: []const u8,
    /// The commit, empty when the citation names none.
    sha: []const u8,
};

/// Finds the `(path:line[, line...] [@sha])` citation on `line`, if any.
///
/// Text that does not look like a citation -- a path with no dot, or one with
/// a space in it -- is prose and is left alone.
fn findCitation(line: []const u8) ?Citation {
    var from: usize = 0;
    while (std.mem.indexOfScalarPos(u8, line, from, '(')) |open| {
        from = open + 1;
        const close = std.mem.indexOfScalarPos(u8, line, open, ')') orelse continue;
        const inside = line[open + 1 .. close];

        // The commit trails the line numbers: `ggml.c:6516 @c1d0e7a00`.
        var body = inside;
        var sha: []const u8 = "";
        if (std.mem.lastIndexOfScalar(u8, inside, '@')) |at| {
            sha = std.mem.trim(u8, inside[at + 1 ..], " \t");
            body = std.mem.trimEnd(u8, inside[0..at], " \t");
        }

        const colon = std.mem.lastIndexOfScalar(u8, body, ':') orelse continue;
        const file = std.mem.trim(u8, body[0..colon], " \t");
        if (file.len == 0) continue;
        if (std.mem.indexOfScalar(u8, file, ' ') != null) continue;
        if (std.mem.indexOfScalar(u8, file, '.') == null) continue;

        return .{ .at = open, .file = file, .nums = body[colon + 1 ..], .sha = sha };
    }
    return null;
}

/// Whether `line` is part of a doc comment block.
fn isComment(line: []const u8) bool {
    const t = std.mem.trimStart(u8, line, " \t");
    return std.mem.startsWith(u8, t, "//");
}

/// Whether `line` opens a citation it does not close -- `(ggml.c:6516`
/// with the `)` on the next comment line.
///
/// **This was a silent hole.** `findCitation` parses one line, and an
/// unclosed `(` simply fell through its `orelse continue`: no citation, no
/// error, no check. 36 citations across nine files were wrapped that way
/// and had never been looked at, in the same class as the `Mirrors …` ones
/// `CLAUDE.md` records. Wrapping at 80 columns is normal here, so the fix
/// is to join the continuation rather than forbid it.
fn opensUnclosedCitation(line: []const u8) bool {
    const open = std.mem.lastIndexOfScalar(u8, line, '(') orelse return false;
    if (std.mem.indexOfScalarPos(u8, line, open, ')') != null) return false;
    const tail = line[open + 1 ..];
    // A citation's head is `<path with a dot>:<digit>`; prose is not.
    const colon = std.mem.indexOfScalar(u8, tail, ':') orelse return false;
    if (std.mem.indexOfScalar(u8, tail[0..colon], '.') == null) return false;
    if (std.mem.indexOfScalar(u8, tail[0..colon], ' ') != null) return false;
    return colon + 1 < tail.len and std.ascii.isDigit(tail[colon + 1]);
}

/// A comment line's text, with its `//`, `///` or `//!` marker removed.
fn commentPayload(line: []const u8) []const u8 {
    var t = std.mem.trimStart(u8, line, " \t");
    if (!std.mem.startsWith(u8, t, "//")) return "";
    t = t[2..];
    if (t.len > 0 and (t[0] == '/' or t[0] == '!')) t = t[1..];
    return std.mem.trimStart(u8, t, " \t");
}

/// `lines[idx]`, with following comment lines joined on while the citation
/// it opens stays unclosed. Two continuations is more than any real
/// citation needs and stops a runaway.
fn logicalLine(
    arena: std.mem.Allocator,
    lines: []const []const u8,
    idx: usize,
) ![]const u8 {
    if (!opensUnclosedCitation(lines[idx])) return lines[idx];

    var joined: std.ArrayList(u8) = .empty;
    try joined.appendSlice(arena, lines[idx]);
    var j = idx + 1;
    while (j < lines.len and j <= idx + 2) : (j += 1) {
        if (!isComment(lines[j])) break;
        try joined.append(arena, ' ');
        try joined.appendSlice(arena, commentPayload(lines[j]));
        if (!opensUnclosedCitation(joined.items)) break;
    }
    return joined.items;
}

/// Pulls every citation out of one of our source files.
///
/// A citation names as many symbols as it gives line numbers, and they pair up
/// in order. The symbols are taken from the citation's own line where it has
/// enough of them, and otherwise from the doc comment block it closes -- which
/// is how a citation wrapped across two lines still finds its names.
fn scanFile(
    arena: std.mem.Allocator,
    out: *std.ArrayList(Reference),
    path: []const u8,
    text: []const u8,
) !void {
    var lines: std.ArrayList([]const u8) = .empty;
    var split = std.mem.splitScalar(u8, text, '\n');
    while (split.next()) |l| try lines.append(arena, l);

    for (0..lines.items.len) |idx| {
        const line = try logicalLine(arena, lines.items, idx);
        const cite = findCitation(line) orelse continue;
        const file = cite.file;

        // Line numbers: one, or several separated by commas.
        var numbers: std.ArrayList(usize) = .empty;
        var malformed = false;
        var nums = std.mem.splitScalar(u8, cite.nums, ',');
        while (nums.next()) |tok| {
            const t = std.mem.trim(u8, tok, " \t");
            // A bad number is a malformed citation, not prose. An earlier
            // version skipped it, and a citation this file had mangled to
            // `common/common.cpp:` passed unnoticed.
            const n = std.fmt.parseInt(usize, t, 10) catch {
                malformed = true;
                break;
            };
            try numbers.append(arena, n);
        }
        if (malformed or numbers.items.len == 0) {
            try out.append(arena, .{
                .from = path,
                .from_line = idx + 1,
                .symbol = "",
                .file = try arena.dupe(u8, file),
                .line = 0,
                .sha = try arena.dupe(u8, cite.sha),
            });
            continue;
        }

        // Symbols: this line first, widening to the doc block if it has fewer
        // names than the citation has numbers.
        var symbols: std.ArrayList([]const u8) = .empty;
        try collectSymbols(arena, &symbols, line[0..cite.at]);
        if (symbols.items.len < numbers.items.len) {
            var start = idx;
            while (start > 0 and isComment(lines.items[start - 1])) start -= 1;
            symbols.clearRetainingCapacity();
            for (lines.items[start..idx]) |prev| try collectSymbols(arena, &symbols, prev);
            try collectSymbols(arena, &symbols, line[0..cite.at]);
        }

        // Pair from the end: the names nearest the citation are its own.
        const take = @min(symbols.items.len, numbers.items.len);
        const tail = symbols.items[symbols.items.len - take ..];

        for (numbers.items, 0..) |n, k| {
            const sym: []const u8 = if (numbers.items.len == 1)
                // One number: any name on the line will do, since a citation
                // like ``Ports `a` and `b` (f:n)`` has them at one line.
                (if (tail.len > 0) tail[tail.len - 1] else "")
            else if (k < tail.len and tail.len == numbers.items.len)
                tail[k]
            else
                "";

            try out.append(arena, .{
                .from = path,
                .from_line = idx + 1,
                .symbol = try arena.dupe(u8, sym),
                .file = try arena.dupe(u8, file),
                .line = n,
                .sha = try arena.dupe(u8, cite.sha),
            });
        }
    }
}

/// Collects every source file under `root`, recursively.
fn collect(
    arena: std.mem.Allocator,
    io: std.Io,
    out: *std.ArrayList([]const u8),
    root: []const u8,
    comptime want_zig: bool,
) !void {
    var dir = std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch return;
    defer dir.close(io);

    var walker = try dir.walk(arena);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        const ok = if (want_zig)
            std.mem.endsWith(u8, entry.basename, ".zig")
        else
            isSource(entry.basename);
        if (!ok) continue;
        try out.append(arena, try std.fmt.allocPrint(arena, "{s}/{s}", .{ root, entry.path }));
    }
}

/// The contents of `rel` in the reference repo as of `sha`.
///
/// **This is what makes a per-citation commit worth writing down.** A citation
/// re-synced against a newer upstream commit is checked against *that* commit,
/// not against whatever the checkout happens to be on, so one function can be
/// updated without touching the other forty in its file.
///
/// Parameters:
/// - `ref_root`: the reference checkout.
/// - `sha`: the commit named by the citation.
/// - `rel`: the path within the repository.
///
/// Return: the file's bytes at that commit, or null when git cannot produce
/// them -- an unknown commit, or a path that did not exist yet.
fn blobAt(
    arena: std.mem.Allocator,
    io: std.Io,
    ref_root: []const u8,
    sha: []const u8,
    rel: []const u8,
) !?[]u8 {
    const spec = try std.fmt.allocPrint(arena, "{s}:{s}", .{ sha, rel });
    const res = std.process.run(arena, io, .{
        .argv = &.{ "git", "-C", ref_root, "show", spec },
        .stdout_limit = .limited(64 << 20),
    }) catch return null;
    switch (res.term) {
        .exited => |code| if (code != 0) return null,
        else => return null,
    }
    return res.stdout;
}

fn readAll(arena: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const limit: std.Io.Limit = .limited(64 << 20);
    return try std.Io.Dir.cwd().readFileAlloc(io, path, arena, limit);
}

/// The `n`th line of `text`, 1-indexed, or null when past the end.
fn nthLine(text: []const u8, n: usize) ?[]const u8 {
    if (n == 0) return null;
    var i: usize = 1;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| : (i += 1) {
        if (i == n) return line;
    }
    return null;
}

/// Whether the definition beginning at `n` names `symbol`.
///
/// Either the name is on that line, or the line opens a brace block that
/// closes on a line naming it -- the anonymous `typedef struct { ... } name;`
/// that `ggml-common.h` uses for every block type.
///
/// Parameters:
/// - `text`: the whole reference file.
/// - `n`: the cited line, 1-indexed.
/// - `symbol`: the name to find.
///
/// Return: whether the citation holds.
fn definesAt(text: []const u8, n: usize, symbol: []const u8) bool {
    const first = nthLine(text, n) orelse return false;
    if (hasWord(first, symbol)) return true;
    if (std.mem.indexOfScalar(u8, first, '{') == null) return false;

    var depth: isize = 0;
    var i: usize = 1;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| : (i += 1) {
        if (i < n) continue;
        for (line) |ch| {
            if (ch == '{') depth += 1;
            if (ch == '}') depth -= 1;
        }
        if (depth <= 0) return hasWord(line, symbol);
        // A block that runs on for pages is not a struct; give up rather than
        // scan a whole translation unit for a name that could be anywhere.
        if (i > n + 200) return false;
    }
    return false;
}

/// Whether `haystack` contains `needle` bounded by non-identifier characters,
/// so `d` does not match inside `dmin`.
fn hasWord(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return false;
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, from, needle)) |at| {
        const before_ok = at == 0 or !isIdent(haystack[at - 1]);
        const end = at + needle.len;
        const after_ok = end >= haystack.len or !isIdent(haystack[end]);
        if (before_ok and after_ok) return true;
        from = at + 1;
    }
    return false;
}

fn isIdent(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_';
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    // <port-dir>... -- <reference-root> [sha]
    var port_dirs: std.ArrayList([]const u8) = .empty;
    var ref_root: []const u8 = "llama.cpp";
    var want_sha: []const u8 = "";
    var after: usize = 0;
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "--")) {
            after = 1;
            continue;
        }
        switch (after) {
            0 => try port_dirs.append(arena, a),
            1 => {
                ref_root = a;
                after = 2;
            },
            else => want_sha = a,
        }
    }
    if (port_dirs.items.len == 0) {
        try port_dirs.append(arena, "src");
        try port_dirs.append(arena, "cli");
    }

    var stderr_buf: [4096]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(io, &stderr_buf);
    const err = &stderr.interface;

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    const out = &stdout.interface;

    // Every C file the references could point at. `common/` and `tools/` are
    // in scope because `cli/` cites them: our CLI is not a port, but its flag
    // surface and sampler chain are modelled on upstream's and the references
    // have to be as accurate as the port's.
    var ref_paths: std.ArrayList([]const u8) = .empty;
    for ([_][]const u8{ "/ggml/include", "/ggml/src", "/include", "/src", "/common", "/tools" }) |sub| {
        const p = try std.fmt.allocPrint(arena, "{s}{s}", .{ ref_root, sub });
        try collect(arena, io, &ref_paths, p, false);
    }

    // Our own C, cited the same way: `testing.zig` points at the harness that
    // captured its goldens. These carry no upstream commit, so a file citing
    // only these is not asked to name one.
    var own_paths: std.ArrayList([]const u8) = .empty;
    try collect(arena, io, &own_paths, "harness", false);
    if (ref_paths.items.len == 0) {
        try err.print("missing reference tree at {s} -- run 'make clone'\n", .{ref_root});
        try err.flush();
        std.process.exit(2);
    }

    // Every citation in our source.
    var refs: std.ArrayList(Reference) = .empty;
    var zig_paths: std.ArrayList([]const u8) = .empty;
    for (port_dirs.items) |d| try collect(arena, io, &zig_paths, d, true);

    var bad: usize = 0;
    var files_with_refs: std.StringHashMapUnmanaged(void) = .empty;

    for (zig_paths.items) |p| {
        const text = try readAll(arena, io, p);
        const before = refs.items.len;
        try scanFile(arena, &refs, p, text);
        if (refs.items.len == before) continue;
        try files_with_refs.put(arena, p, {});
    }

    // Check each citation, caching file contents: ggml.c alone is cited 373
    // times.
    var cache: std.StringHashMapUnmanaged([]const u8) = .empty;
    const short_head: []const u8 = if (want_sha.len >= 9) want_sha[0..9] else "HEAD";

    for (refs.items) |r| {
        if (r.line == 0) {
            try err.print(
                "{s}:{d}: citation of {s} has no usable line number\n",
                .{ r.from, r.from_line, r.file },
            );
            bad += 1;
            continue;
        }

        const into_reference = resolve(ref_paths.items, r.file) != null;
        const target = resolve(ref_paths.items, r.file) orelse resolve(own_paths.items, r.file) orelse {
            try err.print(
                "{s}:{d}: `{s}` -> {s}:{d}: no such file under {s} (or ambiguous)\n",
                .{ r.from, r.from_line, r.symbol, r.file, r.line, ref_root },
            );
            bad += 1;
            continue;
        };

        // A line number is meaningless without the revision it was read at.
        if (into_reference and r.sha.len < 7) {
            try err.print(
                "{s}:{d}: `{s}` -> {s}:{d}: names no commit -- write `({s}:{d} @{s})`\n",
                .{ r.from, r.from_line, r.symbol, r.file, r.line, r.file, r.line, short_head },
            );
            bad += 1;
            continue;
        }

        // Read the file as it was at the citation's own commit. The working
        // tree is the fast path for the overwhelmingly common case of a
        // citation at the checkout's HEAD.
        const key = try std.fmt.allocPrint(arena, "{s}:{s}", .{ r.sha, target });
        const text = cache.get(key) orelse blk: {
            const at_head = !into_reference or
                (want_sha.len >= r.sha.len and std.mem.startsWith(u8, want_sha, r.sha));
            const t = if (at_head)
                try readAll(arena, io, target)
            else t: {
                const rel = target[ref_root.len + 1 ..];
                break :t (try blobAt(arena, io, ref_root, r.sha, rel)) orelse {
                    try err.print(
                        "{s}:{d}: `{s}` -> {s}:{d}: cannot read {s} at commit {s}\n",
                        .{ r.from, r.from_line, r.symbol, r.file, r.line, rel, r.sha },
                    );
                    bad += 1;
                    continue;
                };
            };
            try cache.put(arena, key, t);
            break :blk t;
        };

        const line = nthLine(text, r.line) orelse {
            try err.print(
                "{s}:{d}: `{s}` -> {s}:{d}: past end of file\n",
                .{ r.from, r.from_line, r.symbol, r.file, r.line },
            );
            bad += 1;
            continue;
        };

        if (r.symbol.len == 0) {
            try err.print(
                "{s}:{d}: citation of {s}:{d} names no symbol to check\n",
                .{ r.from, r.from_line, r.file, r.line },
            );
            bad += 1;
            continue;
        }

        if (!definesAt(text, r.line, r.symbol)) {
            try err.print(
                "{s}:{d}: `{s}` -> {s}:{d} @{s}: that line is `{s}`\n",
                .{ r.from, r.from_line, r.symbol, r.file, r.line, r.sha, std.mem.trim(u8, line, " \t") },
            );
            bad += 1;
        }
    }

    if (bad > 0) {
        try err.print(
            "\n{d} problems across {d} citations.\n" ++
                "If the pin moved, these are exactly the functions upstream changed.\n",
            .{ bad, refs.items.len },
        );
        try err.flush();
        std.process.exit(1);
    }

    var commits: std.StringHashMapUnmanaged(void) = .empty;
    for (refs.items) |r| if (r.sha.len > 0) try commits.put(arena, r.sha, {});

    try out.print(
        "PASS: {d} citations in {d} files resolve, across {d} upstream commit(s); HEAD is {s}\n",
        .{ refs.items.len, files_with_refs.count(), commits.count(), short_head },
    );
    try out.flush();
}
