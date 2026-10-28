//! Turn an upstream diff into the list of our declarations it lands in, and
//! repin the citations it does not touch.
//!
//! # Provenance
//!
//! **Not a port.** This is our own code (`MAINTENANCE.md`, Phase 3).
//!
//! # What it does
//!
//! Every citation at commit `A` names a definition by its first line. This
//! reads each cited file at `A`, takes the definition's extent from that line
//! -- to the brace that closes it, the `;` that ends a declaration, or the
//! last continued line of a `#define` -- and intersects the extents with the
//! hunks of `git diff -U0 A B`. That yields three lists:
//!
//! - **touched**: citations whose definition a hunk changed. Each is a Zig
//!   declaration to re-read at `B`.
//! - **unmapped**: hunks no cited extent covers -- new definitions, changes to
//!   code nobody cites, headers and tables. Each must be ported, deferred with
//!   a reason, or recorded as out of scope.
//! - **untouched**: citations whose definition no hunk touched. Their new line
//!   is the old one shifted by every hunk above it, and `--repin` rewrites
//!   them to `B` in place, after checking the definition really is at that
//!   line at `B`.
//!
//! A citation naming several lines repins only when every one of them is
//! untouched, since it carries a single commit.
//!
//! # Usage
//!
//! ```
//! zig run harness/upstream_map.zig -- [--repin] <A> <B> <port-dir>... -- <reference-root> <path>...
//! ```
//!
//! The trailing paths limit the diff to upstream's ported scope.

const std = @import("std");
const links = @import("port_links.zig");

/// One `-U0` hunk: lines `old_start .. old_start + old_count` at `A` became
/// `new_count` lines at `B`. A pure insertion has `old_count == 0` and sits
/// after `old_start`.
const Hunk = struct {
    old_start: usize,
    old_count: usize,
    new_start: usize,
    new_count: usize,
};

/// What one upstream file's diff and citations come to.
const FileMap = struct {
    hunks: []Hunk,
    /// Whether the hunk at the same index fell inside some cited extent.
    mapped: []bool,
};

/// The last line of the definition starting at `n`, 1-indexed.
///
/// A heuristic over C and C++ text, not a parser: it skips comments and
/// string and character literals, ends a `#define` at its last continued
/// line, a declaration at a `;` before any `{`, and a block at the brace that
/// returns the depth to zero. A `{` in an unbalanced macro arm can fool it;
/// the extent then runs long, which only over-reports touched definitions.
///
/// Parameters:
/// - `lines`: the file, split on newlines.
/// - `n`: the definition's first line.
///
/// Return: the definition's last line.
fn extentEnd(lines: []const []const u8, n: usize) usize {
    const first = std.mem.trimStart(u8, lines[n - 1], " \t");
    if (std.mem.startsWith(u8, first, "#")) {
        var i = n;
        while (i < lines.len and std.mem.endsWith(u8, std.mem.trimEnd(u8, lines[i - 1], " \t\r"), "\\")) i += 1;
        return i;
    }

    var depth: isize = 0;
    // A `;` inside parentheses -- a `for` header -- ends nothing.
    var parens: isize = 0;
    var opened = false;
    var in_block_comment = false;
    var i = n;
    while (i <= lines.len) : (i += 1) {
        const line = lines[i - 1];
        var k: usize = 0;
        var quote: u8 = 0;
        while (k < line.len) : (k += 1) {
            const ch = line[k];
            if (in_block_comment) {
                if (ch == '*' and k + 1 < line.len and line[k + 1] == '/') {
                    in_block_comment = false;
                    k += 1;
                }
                continue;
            }
            if (quote != 0) {
                if (ch == '\\') {
                    k += 1;
                } else if (ch == quote) quote = 0;
                continue;
            }
            switch (ch) {
                '"', '\'' => quote = ch,
                '/' => if (k + 1 < line.len) {
                    if (line[k + 1] == '/') break;
                    if (line[k + 1] == '*') {
                        in_block_comment = true;
                        k += 1;
                    }
                },
                '{' => {
                    depth += 1;
                    opened = true;
                },
                '}' => {
                    depth -= 1;
                    if (opened and depth <= 0) return i;
                },
                '(' => parens += 1,
                ')' => parens -= 1,
                ';' => if (!opened and depth == 0 and parens <= 0) return i,
                else => {},
            }
        }
        // A runaway block is not one definition; cap it rather than claim the
        // rest of the file.
        if (i > n + 5000) return i;
    }
    return lines.len;
}

/// Parses `git diff -U0` output for one file into hunks.
fn parseHunks(arena: std.mem.Allocator, diff: []const u8) ![]Hunk {
    var out: std.ArrayList(Hunk) = .empty;
    var it = std.mem.splitScalar(u8, diff, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "@@ -")) continue;
        // @@ -a[,b] +c[,d] @@
        const rest = line[4..];
        const sp = std.mem.findScalar(u8, rest, ' ') orelse continue;
        const old = rest[0..sp];
        const plus = rest[sp + 2 ..];
        const sp2 = std.mem.findScalar(u8, plus, ' ') orelse continue;
        const new = plus[0..sp2];
        const o = try parseRange(old);
        const n = try parseRange(new);
        try out.append(arena, .{ .old_start = o[0], .old_count = o[1], .new_start = n[0], .new_count = n[1] });
    }
    return out.items;
}

fn parseRange(s: []const u8) ![2]usize {
    if (std.mem.findScalar(u8, s, ',')) |c| {
        return .{ try std.fmt.parseInt(usize, s[0..c], 10), try std.fmt.parseInt(usize, s[c + 1 ..], 10) };
    }
    return .{ try std.fmt.parseInt(usize, s, 10), 1 };
}

/// Whether hunk `h` changes anything in lines `start ..= end` at `A`.
///
/// An insertion after line `s` touches a definition when it lands strictly
/// inside it: after its first line and before its last.
fn touches(h: Hunk, start: usize, end: usize) bool {
    if (h.old_count == 0) return h.old_start >= start and h.old_start < end;
    const h_end = h.old_start + h.old_count - 1;
    return h.old_start <= end and h_end >= start;
}

/// The line `n` at `A` moves to at `B`, given that no hunk touches it.
fn shifted(hunks: []const Hunk, n: usize) usize {
    var delta: isize = 0;
    for (hunks) |h| {
        const last_old = if (h.old_count == 0) h.old_start else h.old_start + h.old_count - 1;
        if (last_old < n) delta += @as(isize, @intCast(h.new_count)) - @as(isize, @intCast(h.old_count));
    }
    return @intCast(@as(isize, @intCast(n)) + delta);
}

fn git(arena: std.mem.Allocator, io: std.Io, argv: []const []const u8) ![]u8 {
    const res = try std.process.run(arena, io, .{ .argv = argv, .stdout_limit = .limited(256 << 20) });
    switch (res.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("git failed: {s}\n{s}", .{ argv[3], res.stderr });
            return error.GitFailed;
        },
        else => return error.GitFailed,
    }
    return res.stdout;
}

fn splitLines(arena: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| try lines.append(arena, l);
    return lines.items;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var repin = false;
    var positional: std.ArrayList([]const u8) = .empty;
    var port_dirs: std.ArrayList([]const u8) = .empty;
    var scope: std.ArrayList([]const u8) = .empty;
    var ref_root: ?[]const u8 = null;
    var after_sep = false;
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "--repin")) {
            repin = true;
        } else if (std.mem.eql(u8, a, "--")) {
            after_sep = true;
        } else if (after_sep) {
            if (ref_root == null) ref_root = a else try scope.append(arena, a);
        } else if (positional.items.len < 2) {
            try positional.append(arena, a);
        } else try port_dirs.append(arena, a);
    }
    if (positional.items.len < 2 or port_dirs.items.len == 0 or ref_root == null) {
        std.debug.print("usage: upstream_map [--repin] <A> <B> <port-dir>... -- <reference-root> <path>...\n", .{});
        std.process.exit(2);
    }
    const root = ref_root.?;
    // `^{commit}`: a release tag is an annotated tag object, and a citation
    // names the commit it points at.
    const sha_a = try git(arena, io, &.{ "git", "-C", root, "rev-parse", try arena.print("{s}^{{commit}}", .{positional.items[0]}) });
    const sha_b = try git(arena, io, &.{ "git", "-C", root, "rev-parse", try arena.print("{s}^{{commit}}", .{positional.items[1]}) });
    const a_full = std.mem.trim(u8, sha_a, " \n");
    const b_full = std.mem.trim(u8, sha_b, " \n");
    const b_short = b_full[0..9];

    var stdout_buf: [64 << 10]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    const out = &stdout.interface;

    // Every upstream source, at A, so a citation resolves as port-links
    // resolves it.
    var paths_argv: std.ArrayList([]const u8) = .empty;
    try paths_argv.appendSlice(arena, &.{ "git", "-C", root, "ls-tree", "-r", "--name-only", a_full, "--" });
    for ([_][]const u8{ "ggml/include", "ggml/src", "include", "src", "common", "tools", "vendor" }) |sub| try paths_argv.append(arena, sub);
    const ls = try git(arena, io, paths_argv.items);
    var ref_paths: std.ArrayList([]const u8) = .empty;
    for (try splitLines(arena, ls)) |p| {
        if (p.len == 0 or !links.isSource(p)) continue;
        try ref_paths.append(arena, try arena.print("{s}/{s}", .{ root, p }));
    }

    // Every citation in the port.
    var zig_paths: std.ArrayList([]const u8) = .empty;
    for (port_dirs.items) |d| try links.collect(arena, io, &zig_paths, d, true);
    var refs: std.ArrayList(links.Reference) = .empty;
    for (zig_paths.items) |p| try links.scanFile(arena, &refs, p, try links.readAll(arena, io, p));

    // Group the citations at A by the upstream file they resolve to.
    var by_file: std.StringArrayHashMapUnmanaged(std.ArrayList(usize)) = .empty;
    var not_at_a: usize = 0;
    for (refs.items, 0..) |r, idx| {
        if (r.sha.len < 7 or !std.mem.startsWith(u8, a_full, r.sha)) {
            if (r.sha.len >= 7) not_at_a += 1;
            continue;
        }
        const target = links.resolve(ref_paths.items, r.file) orelse continue;
        const gop = try by_file.getOrPut(arena, target[root.len + 1 ..]);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(arena, idx);
    }

    // The files that changed in scope.
    var diff_argv: std.ArrayList([]const u8) = .empty;
    try diff_argv.appendSlice(arena, &.{ "git", "-C", root, "diff", "--name-status", "--no-renames", a_full, b_full, "--" });
    try diff_argv.appendSlice(arena, scope.items);
    const changed = try git(arena, io, diff_argv.items);

    var touched_count: usize = 0;
    var untouched: std.ArrayList(struct { ref: usize, new_line: usize }) = .empty;
    var blocked_repin = std.AutoHashMapUnmanaged(usize, void).empty; // ref idx of touched
    var unmapped_lines: std.ArrayList(u8) = .empty;
    var touched_lines: std.ArrayList(u8) = .empty;
    var uncited_files: std.ArrayList(u8) = .empty;
    var unmapped_hunk_count: usize = 0;
    var changed_cited: std.StringHashMapUnmanaged(void) = .empty;

    for (try splitLines(arena, changed)) |row| {
        if (row.len < 3) continue;
        const status = row[0];
        const path = std.mem.trim(u8, row[1..], " \t");
        if (!links.isSource(path)) continue;
        const cites = by_file.get(path);

        if (cites == null) {
            const d = try git(arena, io, &.{ "git", "-C", root, "diff", "--shortstat", a_full, b_full, "--", path });
            try uncited_files.print(arena, "  {c} {s} {s}\n", .{ status, path, std.mem.trim(u8, d, " \n") });
            continue;
        }
        try changed_cited.put(arena, path, {});

        const text_a = try git(arena, io, &.{ "git", "-C", root, "show", try arena.print("{s}:{s}", .{ a_full, path }) });
        const lines_a = try splitLines(arena, text_a);
        const diff = if (status == 'D') "" else try git(arena, io, &.{ "git", "-C", root, "diff", "-U0", a_full, b_full, "--", path });
        const hunks = try parseHunks(arena, diff);
        const mapped = try arena.alloc(bool, hunks.len);
        @memset(mapped, false);

        for (cites.?.items) |ri| {
            const r = refs.items[ri];
            if (status == 'D') {
                try blocked_repin.put(arena, ri, {});
                touched_count += 1;
                try touched_lines.print(arena, "  {s}:{d}  `{s}` -> {s}:{d}  (file deleted at B)\n", .{ r.from, r.from_line, r.symbol, path, r.line });
                continue;
            }
            if (r.line == 0 or r.line > lines_a.len) continue;
            const end = extentEnd(lines_a, r.line);
            var hit = false;
            for (hunks, 0..) |h, hi| if (touches(h, r.line, end)) {
                hit = true;
                mapped[hi] = true;
            };
            if (hit) {
                try blocked_repin.put(arena, ri, {});
                touched_count += 1;
                try touched_lines.print(arena, "  {s}:{d}  `{s}` -> {s}:{d}-{d}\n", .{ r.from, r.from_line, r.symbol, path, r.line, end });
            } else {
                try untouched.append(arena, .{ .ref = ri, .new_line = shifted(hunks, r.line) });
            }
        }

        for (hunks, 0..) |h, hi| {
            if (mapped[hi]) continue;
            unmapped_hunk_count += 1;
            try unmapped_lines.print(arena, "  {s}:{d} (-{d} +{d} at B:{d})\n", .{ path, h.old_start, h.old_count, h.new_count, h.new_start });
        }
    }

    // A citation into a file that did not change is untouched at the same line.
    var it = by_file.iterator();
    while (it.next()) |e| {
        if (changed_cited.contains(e.key_ptr.*)) continue;
        // Out of scope paths are not in `changed`; diff them on their own.
        const d = try git(arena, io, &.{ "git", "-C", root, "diff", "-U0", a_full, b_full, "--", e.key_ptr.* });
        const hunks = try parseHunks(arena, d);
        if (hunks.len > 0) {
            const text_a = try git(arena, io, &.{ "git", "-C", root, "show", try arena.print("{s}:{s}", .{ a_full, e.key_ptr.* }) });
            const lines_a = try splitLines(arena, text_a);
            for (e.value_ptr.items) |ri| {
                const r = refs.items[ri];
                if (r.line == 0 or r.line > lines_a.len) continue;
                const end = extentEnd(lines_a, r.line);
                var hit = false;
                for (hunks) |h| if (touches(h, r.line, end)) {
                    hit = true;
                };
                if (hit) {
                    try blocked_repin.put(arena, ri, {});
                    touched_count += 1;
                    try touched_lines.print(arena, "  {s}:{d}  `{s}` -> {s}:{d}-{d}  (outside the scope paths)\n", .{ r.from, r.from_line, r.symbol, e.key_ptr.*, r.line, end });
                } else try untouched.append(arena, .{ .ref = ri, .new_line = shifted(hunks, r.line) });
            }
            continue;
        }
        for (e.value_ptr.items) |ri| try untouched.append(arena, .{ .ref = ri, .new_line = refs.items[ri].line });
    }

    try out.print("upstream-map {s} -> {s}\n\n", .{ a_full[0..9], b_short });
    try out.print("TOUCHED ({d} citations): re-read at B and cite @{s}\n{s}\n", .{ touched_count, b_short, touched_lines.items });
    try out.print("UNMAPPED ({d} hunks in cited files): new or uncited code\n{s}\n", .{ unmapped_hunk_count, unmapped_lines.items });
    try out.print("UNCITED FILES (changed, no citation at A)\n{s}\n", .{uncited_files.items});

    // Repin: group the untouched by Zig file and source line, so a citation
    // naming several lines moves only when all of them are untouched.
    const Key = struct { from: []const u8, from_line: usize };
    var groups: std.AutoArrayHashMapUnmanaged(usize, void) = .empty;
    _ = &groups;
    var move: std.StringArrayHashMapUnmanaged(std.ArrayList(struct { line: usize, old: usize, new: usize, file: []const u8, symbol: []const u8 })) = .empty;
    var verified_bad: usize = 0;
    // One-line extents that do not look like one-line declarations: where the
    // heuristic may have stopped short, which the text check cannot see.
    var suspect: usize = 0;
    var cache: std.StringHashMapUnmanaged([]const []const u8) = .empty;
    var cache_a: std.StringHashMapUnmanaged([]const []const u8) = .empty;
    var blocked_sites: std.StringHashMapUnmanaged(void) = .empty;
    for (refs.items, 0..) |r, ri| if (blocked_repin.contains(ri)) {
        try blocked_sites.put(arena, try arena.print("{s}:{d}", .{ r.from, r.from_line }), {});
    };
    _ = Key;
    for (untouched.items) |u| {
        const r = refs.items[u.ref];
        const site = try arena.print("{s}:{d}", .{ r.from, r.from_line });
        if (blocked_sites.contains(site)) continue;
        // Verify at B: the symbol must be at the shifted line.
        const target = links.resolve(ref_paths.items, r.file).?;
        const rel = target[root.len + 1 ..];
        const lines_b = cache.get(rel) orelse blk: {
            const t = (try links.blobAt(arena, io, root, b_full, rel)) orelse "";
            const l = try splitLines(arena, t);
            try cache.put(arena, rel, l);
            break :blk l;
        };
        const text_b = try std.mem.join(arena, "\n", lines_b);
        // The definition's text must be identical at both ends, not merely
        // start at the shifted line: an extent the heuristic cut short would
        // otherwise class a changed definition as untouched and repin it
        // without anyone reading it.
        const lines_a = cache_a.get(rel) orelse blk: {
            const t = (try links.blobAt(arena, io, root, a_full, rel)) orelse "";
            const l = try splitLines(arena, t);
            try cache_a.put(arena, rel, l);
            break :blk l;
        };
        if (r.line <= lines_a.len and extentEnd(lines_a, r.line) == r.line) {
            const raw = lines_a[r.line - 1];
            const code = if (std.mem.find(u8, raw, "//")) |c| raw[0..c] else raw;
            const t = std.mem.trim(u8, code, " \t\r");
            const one_line = std.mem.endsWith(u8, t, ";") or std.mem.startsWith(u8, t, "#") or
                (std.mem.findScalar(u8, t, '{') != null and std.mem.endsWith(u8, t, "}"));
            if (!one_line) {
                suspect += 1;
                try out.print("  SUSPECT EXTENT {s}:{d} `{s}` {s}:{d}: `{s}`\n", .{ r.from, r.from_line, r.symbol, r.file, r.line, t });
            }
        }
        const same = u.new_line <= lines_b.len and r.line <= lines_a.len and blk: {
            const end_a = extentEnd(lines_a, r.line);
            const end_b = extentEnd(lines_b, u.new_line);
            if (end_a - r.line != end_b - u.new_line) break :blk false;
            for (lines_a[r.line - 1 .. end_a], lines_b[u.new_line - 1 .. end_b]) |la, lb| {
                if (!std.mem.eql(u8, la, lb)) break :blk false;
            }
            break :blk true;
        };
        if (!same) {
            verified_bad += 1;
            try out.print("  REPIN CHECK FAILED (text differs) {s}:{d} `{s}` {s}:{d} -> {d}\n", .{ r.from, r.from_line, r.symbol, r.file, r.line, u.new_line });
            try blocked_sites.put(arena, site, {});
            continue;
        }
        if (r.symbol.len == 0 or !links.definesAt(text_b, u.new_line, r.symbol)) {
            verified_bad += 1;
            try out.print("  REPIN CHECK FAILED {s}:{d} `{s}` {s}:{d} -> {d}\n", .{ r.from, r.from_line, r.symbol, r.file, r.line, u.new_line });
            try blocked_sites.put(arena, site, {});
            continue;
        }
        const gop = try move.getOrPut(arena, r.from);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(arena, .{ .line = r.from_line, .old = r.line, .new = u.new_line, .file = r.file, .symbol = r.symbol });
    }

    var repinned: usize = 0;
    var mit = move.iterator();
    while (mit.next()) |e| {
        var moved_here: usize = 0;
        for (e.value_ptr.items) |m| {
            const site = try arena.print("{s}:{d}", .{ e.key_ptr.*, m.line });
            if (!blocked_sites.contains(site)) moved_here += 1;
        }
        repinned += moved_here;
    }
    try out.print("UNTOUCHED: {d} citations repinnable to @{s}, {d} failed the check at B, {d} cite another commit, {d} suspect extents\n", .{ repinned, b_short, verified_bad, not_at_a, suspect });

    if (repin) {
        // Every reference at each citation site, in the order its numbers
        // appear: a site moves only when all of them do, and its numbers are
        // rewritten by position, never by value -- `545, 546` both moving by
        // one must not become `547, 547`.
        var site_refs: std.StringArrayHashMapUnmanaged(std.ArrayList(usize)) = .empty;
        for (refs.items, 0..) |r, ri| {
            const gop = try site_refs.getOrPut(arena, try arena.print("{s}:{d}", .{ r.from, r.from_line }));
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(arena, ri);
        }
        var new_line_of: std.AutoHashMapUnmanaged(usize, usize) = .empty;
        var mit2 = move.iterator();
        while (mit2.next()) |e| for (e.value_ptr.items) |m| {
            const site = try arena.print("{s}:{d}", .{ e.key_ptr.*, m.line });
            if (blocked_sites.contains(site)) continue;
            for (site_refs.get(site).?.items) |ri| {
                if (refs.items[ri].line == m.old and std.mem.eql(u8, refs.items[ri].symbol, m.symbol) and !new_line_of.contains(ri)) {
                    try new_line_of.put(arena, ri, m.new);
                    break;
                }
            }
        };

        var files_written: usize = 0;
        var sites_moved: usize = 0;
        var rit = move.iterator();
        while (rit.next()) |e| {
            const path = e.key_ptr.*;
            const lines = try splitLines(arena, try links.readAll(arena, io, path));
            const edited = try arena.alloc([]const u8, lines.len);
            @memcpy(edited, lines);
            var changed_any = false;

            var done_lines: std.AutoHashMapUnmanaged(usize, void) = .empty;
            for (e.value_ptr.items) |m| {
                if (done_lines.contains(m.line)) continue;
                try done_lines.put(arena, m.line, {});
                const site = try arena.print("{s}:{d}", .{ path, m.line });
                if (blocked_sites.contains(site)) continue;
                const members = site_refs.get(site).?.items;
                var news: std.ArrayList(usize) = .empty;
                for (members) |ri| {
                    const n = new_line_of.get(ri) orelse break;
                    try news.append(arena, n);
                }
                if (news.items.len != members.len) continue;

                const ln = m.line;
                const span = @min(3, edited.len - (ln - 1));
                const joined = try std.mem.join(arena, "\n", edited[ln - 1 .. ln - 1 + span]);
                const rewritten = (try rewriteCitation(arena, joined, edited[ln - 1].len, news.items, b_short)) orelse {
                    try out.print("  REPIN SKIPPED (citation not found) {s}\n", .{site});
                    continue;
                };
                var parts = std.mem.splitScalar(u8, rewritten, '\n');
                var k: usize = 0;
                while (parts.next()) |part| : (k += 1) edited[ln - 1 + k] = part;
                if (k != span) return error.LineCountChanged;
                changed_any = true;
                sites_moved += 1;
            }
            if (!changed_any) continue;
            const new_text = try std.mem.join(arena, "\n", edited);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = new_text });
            files_written += 1;
        }
        try out.print("repinned {d} citation sites in {d} files\n", .{ sites_moved, files_written });
    }
    try out.flush();
}

/// Rewrites the first citation that opens on the first line of `text`:
/// its line numbers become `news`, in order, and its commit becomes `sha`.
///
/// It finds the citation as `port_links.zig`'s `findCitation` does -- the
/// first `(` whose contents read `file:numbers @commit`, with a dot and no
/// space in the file -- but over the raw lines, so a citation wrapped onto
/// the next comment line keeps its line breaks. Nothing outside its
/// parentheses changes, so a second citation sharing the line keeps its own
/// commit.
///
/// Parameters:
/// - `text`: up to three source lines joined by `\n`.
/// - `first_len`: the first line's length; the citation must open in it.
/// - `news`: the new line numbers, one per number in the citation.
/// - `sha`: the new commit.
///
/// Return: the rewritten text, or null when no citation opens on the first
/// line or its numbers do not match `news` in count.
fn rewriteCitation(arena: std.mem.Allocator, text: []const u8, first_len: usize, news: []const usize, sha: []const u8) !?[]const u8 {
    var from: usize = 0;
    while (std.mem.findScalarPos(u8, text, from, '(')) |open| {
        if (open >= first_len) return null;
        from = open + 1;
        const close = std.mem.findScalarPos(u8, text, open, ')') orelse continue;
        const inside = text[open + 1 .. close];
        const at = std.mem.findScalarLast(u8, inside, '@') orelse continue;
        const body = inside[0..at];
        const colon = std.mem.findScalarLast(u8, body, ':') orelse continue;
        const file = std.mem.trim(u8, body[0..colon], " \t");
        if (file.len == 0) continue;
        if (std.mem.findScalar(u8, file, ' ') != null) continue;
        if (std.mem.findScalar(u8, file, '.') == null) continue;

        // Rewrite the digit runs of `body[colon + 1 ..]` by position.
        const nums = body[colon + 1 ..];
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(arena, text[0 .. open + 1 + colon + 1]);
        var k: usize = 0;
        var i: usize = 0;
        while (i < nums.len) {
            if (std.ascii.isDigit(nums[i])) {
                var j = i;
                while (j < nums.len and std.ascii.isDigit(nums[j])) j += 1;
                if (k >= news.len) return null;
                try out.print(arena, "{d}", .{news[k]});
                k += 1;
                i = j;
                continue;
            }
            try out.append(arena, nums[i]);
            i += 1;
        }
        if (k != news.len) return null;
        // `@` and the commit, with whatever whitespace followed it kept.
        const after_at = inside[at + 1 ..];
        const lead = after_at.len - std.mem.trimStart(u8, after_at, " \t").len;
        const trail = after_at.len - std.mem.trimEnd(u8, after_at, " \t").len;
        try out.append(arena, '@');
        try out.appendSlice(arena, after_at[0..lead]);
        try out.appendSlice(arena, sha);
        try out.appendSlice(arena, after_at[after_at.len - trail ..]);
        try out.appendSlice(arena, text[close..]);
        return out.items;
    }
    return null;
}
