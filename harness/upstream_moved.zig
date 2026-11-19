//! Follow citations into upstream files that a sync deleted or renamed.
//!
//! # Provenance
//!
//! **Not a port.** This is our own code, a companion to `upstream_map.zig`.
//!
//! # Why it exists
//!
//! `upstream_map.zig` diffs `A..B` with `--no-renames` and repins a citation
//! only within its own file. Upstream deleted `tools/mtmd/clip.cpp`
//! and renamed `clip-impl.h`, `clip-model.h` and `clip-graph.h` to `mtmd-*`,
//! moving their definitions across seven new files. Every citation into them
//! then fails `port-links` -- the file no longer exists at `B` -- and
//! `upstream-map` reports each one as touched, though most definitions moved
//! byte for byte.
//!
//! # What it does
//!
//! For every citation at `A` whose file exists at `A` but not at `B`, it
//! takes the definition's extent at `A` (`upstream_map.extentEnd`) and looks
//! for the same lines, compared with trailing whitespace trimmed, in the
//! destination files at `B`. Exactly one match makes the citation **moved**:
//! with `--write` its file, line and commit are rewritten in place. A site
//! naming several lines moves only when all of them land in one file.
//! Anything else is **hand work**, printed with the lines where the symbol is
//! defined at `B`, if any: the definition changed on the way, or it is gone.
//!
//! The new file is written as the shortest path suffix that resolves
//! uniquely at `B`, the rule `port_links.resolve` enforces.
//!
//! # Usage
//!
//! ```
//! zig run harness/upstream_moved.zig -- [--write] <A> <B> <port-dir>... -- <reference-root> <dest-dir>...
//! ```

const std = @import("std");
const links = @import("port_links.zig");
const map = @import("upstream_map.zig");

/// Where one citation's definition went.
const Move = struct {
    ref: usize,
    /// The destination's path relative to the reference root.
    dest: []const u8,
    line: usize,
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var write = false;
    var positional: std.ArrayList([]const u8) = .empty;
    var port_dirs: std.ArrayList([]const u8) = .empty;
    var dest_dirs: std.ArrayList([]const u8) = .empty;
    var ref_root: ?[]const u8 = null;
    var after_sep = false;
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "--write")) {
            write = true;
        } else if (std.mem.eql(u8, a, "--")) {
            after_sep = true;
        } else if (after_sep) {
            if (ref_root == null) ref_root = a else try dest_dirs.append(arena, a);
        } else if (positional.items.len < 2) {
            try positional.append(arena, a);
        } else try port_dirs.append(arena, a);
    }
    if (positional.items.len < 2 or port_dirs.items.len == 0 or ref_root == null or dest_dirs.items.len == 0) {
        std.debug.print("usage: upstream_moved [--write] <A> <B> <port-dir>... -- <reference-root> <dest-dir>...\n", .{});
        std.process.exit(2);
    }
    const root = ref_root.?;
    const a_full = std.mem.trim(u8, try map.git(arena, io, &.{ "git", "-C", root, "rev-parse", try arena.print("{s}^{{commit}}", .{positional.items[0]}) }), " \n");
    const b_full = std.mem.trim(u8, try map.git(arena, io, &.{ "git", "-C", root, "rev-parse", try arena.print("{s}^{{commit}}", .{positional.items[1]}) }), " \n");
    const b_short = b_full[0..9];

    var stdout_buf: [64 << 10]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    const out = &stdout.interface;

    const scope = [_][]const u8{ "ggml/include", "ggml/src", "include", "src", "common", "tools", "vendor" };
    const paths_a = try sourcePaths(arena, io, root, a_full, &scope);
    const paths_b = try sourcePaths(arena, io, root, b_full, &scope);
    var at_b: std.StringHashMapUnmanaged(void) = .empty;
    for (paths_b) |p| try at_b.put(arena, p[root.len + 1 ..], {});

    // The destination files at B, read once.
    var dests: std.ArrayList(struct { rel: []const u8, lines: []const []const u8 }) = .empty;
    for (paths_b) |p| {
        const rel = p[root.len + 1 ..];
        for (dest_dirs.items) |d| {
            if (!std.mem.startsWith(u8, rel, d)) continue;
            const blob = (try links.blobAt(arena, io, root, b_full, rel)) orelse continue;
            try dests.append(arena, .{ .rel = rel, .lines = try map.splitLines(arena, blob) });
            break;
        }
    }

    var zig_paths: std.ArrayList([]const u8) = .empty;
    for (port_dirs.items) |d| try links.collect(arena, io, &zig_paths, d, true);
    var refs: std.ArrayList(links.Reference) = .empty;
    for (zig_paths.items) |p| try links.scanFile(arena, &refs, p, try links.readAll(arena, io, p));

    var blob_cache: std.StringHashMapUnmanaged([]const []const u8) = .empty;
    var moves: std.AutoHashMapUnmanaged(usize, Move) = .empty;
    var gone_count: usize = 0;
    var hand: std.ArrayList(u8) = .empty;

    for (refs.items, 0..) |r, ri| {
        if (r.sha.len < 7 or !std.mem.startsWith(u8, a_full, r.sha) or r.line == 0) continue;
        const target = links.resolve(paths_a, r.file) orelse continue;
        const rel = target[root.len + 1 ..];
        if (at_b.contains(rel)) continue;
        gone_count += 1;

        const lines = blob_cache.get(rel) orelse blk: {
            const blob = (try links.blobAt(arena, io, root, a_full, rel)) orelse return error.MissingBlob;
            const l = try map.splitLines(arena, blob);
            try blob_cache.put(arena, rel, l);
            break :blk l;
        };
        if (r.line > lines.len) continue;
        const end = map.extentEnd(lines, r.line);
        const body = lines[r.line - 1 .. end];

        var found: ?Move = null;
        var count: usize = 0;
        for (dests.items) |d| {
            if (d.lines.len < body.len) continue;
            var m: usize = 0;
            while (m + body.len <= d.lines.len) : (m += 1) {
                if (!sameLines(d.lines[m .. m + body.len], body)) continue;
                count += 1;
                found = .{ .ref = ri, .dest = d.rel, .line = m + 1 };
            }
        }
        if (count == 1) {
            try moves.put(arena, ri, found.?);
            continue;
        }

        // Hand work: say where the symbol is defined now, if anywhere.
        const hw = &hand;
        try hw.print(arena, "  {s}:{d}  `{s}` {s}:{d}-{d} ({d} lines) -> {s}", .{
            r.from,                                             r.from_line, r.symbol, r.file, r.line, end, body.len,
            if (count == 0) "changed or gone" else "ambiguous",
        });
        if (r.symbol.len > 0) {
            var shown: usize = 0;
            for (dests.items) |d| {
                const text = try std.mem.join(arena, "\n", d.lines);
                for (d.lines, 0..) |l, k| {
                    if (!links.hasWord(l, r.symbol)) continue;
                    if (!links.definesAt(text, k + 1, r.symbol)) continue;
                    if (!looksLikeDefinition(l, r.symbol)) continue;
                    if (shown < 3) try hw.print(arena, " | {s}:{d}", .{ d.rel, k + 1 });
                    shown += 1;
                }
            }
            if (shown > 3) try hw.print(arena, " (+{d})", .{shown - 3});
        }
        try hw.append(arena, '\n');
    }

    // Group by site: a site moves only when every reference on it moved, to
    // one file.
    var site_refs: std.StringArrayHashMapUnmanaged(std.ArrayList(usize)) = .empty;
    for (refs.items, 0..) |r, ri| {
        const gop = try site_refs.getOrPut(arena, try arena.print("{s}:{d}", .{ r.from, r.from_line }));
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(arena, ri);
    }

    var by_path: std.StringArrayHashMapUnmanaged(std.ArrayList(struct { line: usize, dest: []const u8, news: []usize })) = .empty;
    var split_sites: usize = 0;
    var moved_sites: usize = 0;
    var sit = site_refs.iterator();
    while (sit.next()) |e| {
        const members = e.value_ptr.items;
        const first = moves.get(members[0]) orelse continue;
        var news = try arena.alloc(usize, members.len);
        var ok = true;
        for (members, 0..) |ri, k| {
            const m = moves.get(ri) orelse {
                ok = false;
                break;
            };
            if (!std.mem.eql(u8, m.dest, first.dest)) {
                ok = false;
                break;
            }
            news[k] = m.line;
        }
        if (!ok) {
            split_sites += 1;
            try out.print("  SITE NOT MOVED (lines went to different files, or one did not move) {s}\n", .{e.key_ptr.*});
            continue;
        }
        moved_sites += 1;
        const r0 = refs.items[members[0]];
        const gop = try by_path.getOrPut(arena, r0.from);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(arena, .{ .line = r0.from_line, .dest = try shortestUnique(arena, paths_b, root, first.dest), .news = news });
    }

    try out.print("HAND WORK ({d} citations into files gone at B whose definition did not move verbatim)\n{s}", .{ gone_count - moves.count(), hand.items });
    try out.print("MOVED: {d} of {d} citations into files gone at B; {d} sites movable, {d} not\n", .{ moves.count(), gone_count, moved_sites, split_sites });

    if (write) {
        var files_written: usize = 0;
        var sites_written: usize = 0;
        var pit = by_path.iterator();
        while (pit.next()) |e| {
            const path = e.key_ptr.*;
            const lines = try map.splitLines(arena, try links.readAll(arena, io, path));
            const edited = try arena.alloc([]const u8, lines.len);
            @memcpy(edited, lines);
            for (e.value_ptr.items) |s| {
                const ln = s.line;
                const span = @min(3, edited.len - (ln - 1));
                const joined = try std.mem.join(arena, "\n", edited[ln - 1 .. ln - 1 + span]);
                const rewritten = (try rewriteMoved(arena, joined, edited[ln - 1].len, s.dest, s.news, b_short)) orelse {
                    try out.print("  REWRITE SKIPPED (citation not found) {s}:{d}\n", .{ path, ln });
                    continue;
                };
                var parts = std.mem.splitScalar(u8, rewritten, '\n');
                var k: usize = 0;
                while (parts.next()) |part| : (k += 1) edited[ln - 1 + k] = part;
                if (k != span) return error.LineCountChanged;
                sites_written += 1;
            }
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = try std.mem.join(arena, "\n", edited) });
            files_written += 1;
        }
        try out.print("rewrote {d} citation sites in {d} files\n", .{ sites_written, files_written });
    }
    try out.flush();
}

/// Every source path under `scope` at commit `sha`, prefixed with `root/`.
fn sourcePaths(arena: std.mem.Allocator, io: std.Io, root: []const u8, sha: []const u8, scope: []const []const u8) ![]const []const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(arena, &.{ "git", "-C", root, "ls-tree", "-r", "--name-only", sha, "--" });
    try argv.appendSlice(arena, scope);
    const ls = try map.git(arena, io, argv.items);
    var paths: std.ArrayList([]const u8) = .empty;
    for (try map.splitLines(arena, ls)) |p| {
        if (p.len == 0 or !links.isSource(p)) continue;
        try paths.append(arena, try arena.print("{s}/{s}", .{ root, p }));
    }
    return paths.items;
}

/// Whether two line runs are equal with trailing whitespace ignored.
fn sameLines(a: []const []const u8, b: []const []const u8) bool {
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, std.mem.trimEnd(u8, x, " \t\r"), std.mem.trimEnd(u8, y, " \t\r"))) return false;
    }
    return true;
}

/// A loose filter for the hand-work hints: the symbol is followed by `(`,
/// `{`, `=` or is a `#define`, `struct`, `enum` or `class` name.
fn looksLikeDefinition(line: []const u8, symbol: []const u8) bool {
    const t = std.mem.trimStart(u8, line, " \t");
    if (std.mem.startsWith(u8, t, "//") or std.mem.startsWith(u8, t, "*")) return false;
    if (std.mem.startsWith(u8, t, "return ")) return false;
    const at = std.mem.find(u8, line, symbol) orelse return false;
    const rest = std.mem.trimStart(u8, line[at + symbol.len ..], " \t");
    if (std.mem.startsWith(u8, t, "#define")) return true;
    if (std.mem.startsWith(u8, t, "struct ") or std.mem.startsWith(u8, t, "enum ") or std.mem.startsWith(u8, t, "class ")) return true;
    return rest.len == 0 or rest[0] == '(' or rest[0] == '{' or rest[0] == '=' or rest[0] == ';' or rest[0] == ',' or rest[0] == '[';
}

/// The shortest suffix of `rel` that `port_links.resolve` resolves uniquely
/// among `paths`.
fn shortestUnique(arena: std.mem.Allocator, paths: []const []const u8, root: []const u8, rel: []const u8) ![]const u8 {
    _ = arena;
    _ = root;
    var cut: usize = rel.len;
    while (true) {
        const slash = std.mem.findScalarLast(u8, rel[0..cut], '/') orelse return rel;
        const want = rel[slash + 1 ..];
        if (links.resolve(paths, want)) |_| return want;
        cut = slash;
    }
}

/// Rewrites the first citation opening on the first line of `text`: its file
/// becomes `file`, its numbers `news` by position, its commit `sha`.
///
/// Parameters:
/// - `text`: up to three source lines joined by `\n`.
/// - `first_len`: the first line's length; the citation must open in it.
/// - `file`: the new file name.
/// - `news`: the new line numbers.
/// - `sha`: the new commit.
///
/// Return: the rewritten text, or null when no citation opens on the first
/// line or its numbers do not match `news` in count.
fn rewriteMoved(arena: std.mem.Allocator, text: []const u8, first_len: usize, file: []const u8, news: []const usize, sha: []const u8) !?[]const u8 {
    var from: usize = 0;
    while (std.mem.findScalarPos(u8, text, from, '(')) |open| {
        if (open >= first_len) return null;
        from = open + 1;
        const close = std.mem.findScalarPos(u8, text, open, ')') orelse continue;
        const inside = text[open + 1 .. close];
        const at = std.mem.findScalarLast(u8, inside, '@') orelse continue;
        const body = inside[0..at];
        const colon = std.mem.findScalarLast(u8, body, ':') orelse continue;
        const old_file = std.mem.trim(u8, body[0..colon], " \t");
        if (old_file.len == 0) continue;
        if (std.mem.findScalar(u8, old_file, ' ') != null) continue;
        if (std.mem.findScalar(u8, old_file, '.') == null) continue;
        const lead_ws = body[0..colon].len - std.mem.trimStart(u8, body[0..colon], " \t").len;

        var o: std.ArrayList(u8) = .empty;
        try o.appendSlice(arena, text[0 .. open + 1 + lead_ws]);
        try o.appendSlice(arena, file);
        try o.append(arena, ':');
        const nums = body[colon + 1 ..];
        var k: usize = 0;
        var i: usize = 0;
        while (i < nums.len) {
            if (std.ascii.isDigit(nums[i])) {
                var j = i;
                while (j < nums.len and std.ascii.isDigit(nums[j])) j += 1;
                if (k >= news.len) return null;
                try o.print(arena, "{d}", .{news[k]});
                k += 1;
                i = j;
                continue;
            }
            try o.append(arena, nums[i]);
            i += 1;
        }
        if (k != news.len) return null;
        const after_at = inside[at + 1 ..];
        const lead = after_at.len - std.mem.trimStart(u8, after_at, " \t").len;
        const trail = after_at.len - std.mem.trimEnd(u8, after_at, " \t").len;
        try o.append(arena, '@');
        try o.appendSlice(arena, after_at[0..lead]);
        try o.appendSlice(arena, sha);
        try o.appendSlice(arena, after_at[after_at.len - trail ..]);
        try o.appendSlice(arena, text[close..]);
        return o.items;
    }
    return null;
}
