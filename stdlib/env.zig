//! The process environment, as upstream's tools read it with `getenv` and
//! hand it to a child through `environ`.
//!
//! # Provenance
//!
//! **Not a port.** On POSIX std hands the environment to `main` alone, in
//! `std.process.Init`; every tool's `main`, and the server's, passes it to
//! `set` before anything reads it. The tools never change their own
//! environment, so what `main` was given is what `getenv` and `environ`
//! would read.
//!
//! `llama-server-core`'s consumer calls `set` the same way. One that does
//! not gets the C library's `environ`, copied on first use: std exposes no
//! POSIX environment outside `main`, so that fallback is the C library's
//! until it does.

const std = @import("std");
const builtin = @import("builtin");
const sync = @import("sync/module.zig");

/// The environment, each value allocated with a terminating NUL after its
/// slice so `get` can hand out a C string. An `Environ.Map`, so a Windows
/// lookup ignores case as `getenv` does there; its order is `environ`'s.
var copy: ?std.process.Environ.Map = null;
/// `copy` as a POSIX block, for the `Io` that spawns children.
var copy_block: std.process.Environ = .{ .block = .empty };
/// Set once `copy` is final; read without the lock.
var ready: std.atomic.Value(bool) = .init(false);
/// Guards the fallback copy of `environ`; `set` runs before any thread.
var fallback_mutex: sync.Mutex = .{};
var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);

/// Takes the environment `main` was given.
///
/// Parameters:
/// - `source`: `std.process.Init.environ_map`; read here, not kept.
pub fn set(source: *const std.process.Environ.Map) void {
    const a = arena_state.allocator();
    var m: std.process.Environ.Map = .init(a);
    var it = source.array_hash_map.iterator();
    while (it.next()) |e| add(&m, e.key_ptr.*, e.value_ptr.*) catch return;
    publish(m) catch return;
}

/// Adds one variable to `m` unless it already holds the name: a duplicate
/// keeps its first value, the one `getenv` returns.
fn add(m: *std.process.Environ.Map, key: []const u8, value: []const u8) std.mem.Allocator.Error!void {
    if (m.contains(key)) return;
    const a = arena_state.allocator();
    try m.putMove(try a.dupe(u8, key), try a.dupeSentinel(u8, value, 0));
}

/// Makes `m` the environment, with its POSIX block.
fn publish(m: std.process.Environ.Map) std.mem.Allocator.Error!void {
    copy_block = if (comptime builtin.os.tag == .windows)
        .{ .block = .global }
    else
        .{ .block = try m.createPosixBlock(arena_state.allocator(), .{}) };
    copy = m;
    ready.store(true, .release);
}

/// Copies the C library's `environ` once, for a consumer that never called
/// `set`.
fn ensure() void {
    if (ready.load(.acquire)) return;
    fallback_mutex.lock();
    defer fallback_mutex.unlock();
    if (ready.load(.acquire)) return;
    if (comptime builtin.link_libc and builtin.os.tag != .windows) {
        var m: std.process.Environ.Map = .init(arena_state.allocator());
        var i: usize = 0;
        while (std.c.environ[i]) |p| : (i += 1) {
            const entry = std.mem.span(p);
            const eq = std.mem.findScalar(u8, entry, '=') orelse continue; // getenv never matches one
            if (eq == 0) continue;
            add(&m, entry[0..eq], entry[eq + 1 ..]) catch return;
        }
        publish(m) catch return;
    }
}

/// `getenv(name)`, the C library's live environment: what the libraries
/// read, since a host -- or a gate -- may `setenv` between calls, which a
/// copy taken once would miss. It stays the C library's until std reads
/// the environment outside `main`.
///
/// Return: the value, valid until the environment changes; null when
/// `name` is not set.
pub fn getenv(name: [*:0]const u8) ?[*:0]u8 {
    return std.c.getenv(name);
}

/// `getenv(name)`, from the environment `main` was given (`set`).
///
/// Return: the value, valid for the process's life, or null when `name` is
/// not set.
pub fn get(name: []const u8) ?[:0]const u8 {
    ensure();
    const m = &(copy orelse return null);
    const v = m.get(name) orelse return null;
    return v.ptr[0..v.len :0];
}

/// `getpwuid(getuid())->pw_dir`: the password database's home for this
/// user, where upstream falls back to it when `HOME` is unset. The C
/// library's, as std reads no password database.
///
/// Return: the C library's string, valid until its next password lookup;
/// null on Windows, or when there is no entry or no directory.
pub fn passwdHome() ?[:0]const u8 {
    if (comptime builtin.os.tag == .windows) return null;
    const pw = std.c.getpwuid(std.c.getuid()) orelse return null;
    const dir = pw.dir orelse return null;
    return std.mem.span(dir);
}

/// The whole environment, in `environ`'s order: what a child inherits.
///
/// Return: valid for the process's life, never changed once read; null
/// when there is none (a consumer on Windows that never called `set`).
pub fn map() ?*const std.process.Environ.Map {
    ensure();
    return if (copy) |*m| m else null;
}

/// The environment as std's `Io` takes it (`std.Io.Threaded.InitOptions`'s
/// `environ`), for spawning children: the PATH a program name is searched
/// in, and what a child inherits.
///
/// Return: valid for the process's life.
pub fn processEnviron() std.process.Environ {
    ensure();
    return copy_block;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "set copies the environment and get answers from it, as C strings" {
    var source: std.process.Environ.Map = .init(std.testing.allocator);
    defer source.deinit();
    try source.put("LLAMA_TEST_A", "one");
    try source.put("LLAMA_TEST_EMPTY", "");
    const saved = .{ copy, copy_block, ready.load(.acquire) };
    defer {
        copy = saved[0];
        copy_block = saved[1];
        ready.store(saved[2], .release);
    }
    set(&source);
    const a = get("LLAMA_TEST_A") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("one", a);
    try std.testing.expectEqual(@as(u8, 0), a.ptr[a.len]);
    try std.testing.expectEqualStrings("", get("LLAMA_TEST_EMPTY").?);
    try std.testing.expect(get("LLAMA_TEST_MISSING") == null);
    // no NUL inside a stored value, so a child's block reads "KEY=value"
    try std.testing.expectEqualStrings("one", map().?.get("LLAMA_TEST_A").?);
    if (comptime builtin.os.tag != .windows) {
        var found = false;
        for (processEnviron().block.slice) |p| found = found or std.mem.eql(u8, std.mem.span(p.?), "LLAMA_TEST_A=one");
        try std.testing.expect(found);
    }
}

test "a duplicate name keeps its first value" {
    var m: std.process.Environ.Map = .init(arena_state.allocator());
    try add(&m, "LLAMA_TEST_DUP", "first");
    try add(&m, "LLAMA_TEST_DUP", "second");
    try std.testing.expectEqualStrings("first", m.get("LLAMA_TEST_DUP").?);
}
