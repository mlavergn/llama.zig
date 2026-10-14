//! Compares `src/llama/arch.zig` with llama.cpp's `llama-arch.cpp`.
//!
//! **Not a port.** The harness behind `scripts/arch-diff`. It links the C++
//! `llama-arch.o`, inside `libllama.a` and reached through the C wrappers in
//! `harness/arch_ref.cpp`, beside the Zig through `src/llama/harness_root.zig`, and
//! asks both the same questions:
//!
//! - every architecture's name and its five predicates, and `llm_arch_all`;
//! - `llm_arch_from_string` on every name and on near misses;
//! - every metadata key's name for every architecture, bare and with a
//!   suffix;
//! - every tensor's name over block and expert indices at the edges of
//!   `int`, bare and with a suffix;
//! - every tensor's info, or the exception `map::at` throws for it.
//!
//! The tables are indexed by enum ordinal, so a name compared at an ordinal
//! also checks the generated enum's order. Ordinals past the end of each
//! Zig enum are put to the C++ alone and must miss its maps -- that is what
//! says the generated enums are not short. A tensor with no name makes the
//! C++ abort rather than throw, so those run in a forked child on each side
//! and must both die of `SIGABRT` with the same message.
//!
//! Usage: arch_diff

const std = @import("std");
const harness_root = @import("harness_root");

const A = harness_root.arch;
const c = harness_root.impl.c;

extern fn ref_arch_unknown() c_int;
extern fn ref_arch_name(arch: c_int) [*:0]const u8;
extern fn ref_arch_from_string(name: [*:0]const u8) c_int;
extern fn ref_arch_all(out: [*]c_int, cap: c_int) c_int;
extern fn ref_arch_flags(arch: c_int) c_int;
extern fn ref_kv_name(arch: c_int, kv: c_int, suffix: ?[*:0]const u8, buf: [*]u8, n: usize) c_int;
extern fn ref_tensor_name(arch: c_int, tensor: c_int, suffix: ?[*:0]const u8, bid: c_int, xid: c_int, buf: [*]u8, n: usize) c_int;
extern fn ref_tensor_info(tensor: c_int, layer: *c_int, op: *c_int, buf: [*]u8, n: usize) c_int;

extern "c" fn fork() c_int;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn dup2(old: c_int, new: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;

const sigabrt = 6;

var checks: usize = 0;
var failures: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    if (failures <= 20) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

fn expectStr(what: []const u8, ref: []const u8, port: []const u8) void {
    checks += 1;
    if (!std.mem.eql(u8, ref, port)) fail("{s}: ref '{s}' port '{s}'", .{ what, ref, port });
}

fn expectInt(what: []const u8, ref: i64, port: i64) void {
    checks += 1;
    if (ref != port) fail("{s}: ref {d} port {d}", .{ what, ref, port });
}

fn count(comptime E: type) usize {
    return std.enums.values(E).len;
}

fn flags(a: A.Arch) c_int {
    return @as(c_int, @intFromBool(A.isRecurrent(a))) |
        @as(c_int, @intFromBool(A.isHybrid(a))) << 1 |
        @as(c_int, @intFromBool(A.isDiffusion(a))) << 2 |
        @as(c_int, @intFromBool(A.supportsRsRollback(a))) << 3 |
        @as(c_int, @intFromBool(A.supportsSmTensor(a))) << 4;
}

// -----------------------------------------------------------------------------
// Architectures

fn checkArchs() void {
    const n = count(A.Arch);
    expectInt("LLM_ARCH_UNKNOWN", ref_arch_unknown(), @backingInt(A.Arch.unknown));
    expectInt("llm_arch count", ref_arch_unknown() + 1, @intCast(n));

    var all: [512]c_int = undefined;
    const n_all = ref_arch_all(&all, all.len);
    const port_all = A.archAll();
    expectInt("llm_arch_all length", n_all, @intCast(port_all.len));
    for (port_all, 0..) |a, i| {
        if (i < n_all) expectInt("llm_arch_all entry", all[i], @backingInt(a));
    }

    for (std.enums.values(A.Arch)) |a| {
        const i = @backingInt(a);
        var what: [64]u8 = undefined;
        const w = std.mem.print(&what, "arch {d}", .{i}) catch unreachable;
        expectStr(w, std.mem.span(ref_arch_name(i)), A.archName(a));
        expectInt(w, ref_arch_flags(i), flags(a));
    }
    // Past the end of the enum, the map must miss.
    for (n..n + 4) |i| expectStr("arch past the enum", std.mem.span(ref_arch_name(@intCast(i))), "unknown");

    // llm_arch_from_string: every name, and near misses of each.
    var probes: std.ArrayList([:0]const u8) = .empty;
    const extra = [_][:0]const u8{ "", "unknown", "(unknown)", "LLAMA", "llama ", " llama", "qwen3", "qwen35", "qwen35moe", "qwen", "llama4" };
    for (extra) |e| probes.append(std.heap.c_allocator, e) catch unreachable;
    for (std.enums.values(A.Arch)) |a| {
        const name = A.archName(a);
        probes.append(std.heap.c_allocator, name) catch unreachable;
        if (name.len > 1) probes.append(std.heap.c_allocator, std.heap.c_allocator.dupeSentinel(u8, name[0 .. name.len - 1], 0) catch unreachable) catch unreachable;
        const upper = std.heap.c_allocator.dupeSentinel(u8, name, 0) catch unreachable;
        for (upper) |*ch| ch.* = std.ascii.toUpper(ch.*);
        probes.append(std.heap.c_allocator, upper) catch unreachable;
    }
    for (probes.items) |p| {
        expectInt(p, ref_arch_from_string(p.ptr), @backingInt(A.archFromString(p)));
    }
}

// -----------------------------------------------------------------------------
// Metadata keys

fn checkKeys() void {
    const suffixes = [_]?[:0]const u8{ null, "weight", "" };
    var buf: [512]u8 = undefined;
    for (std.enums.values(A.Arch)) |a| {
        for (std.enums.values(A.Kv)) |kv| {
            for (suffixes) |sfx| {
                const r = ref_kv_name(@backingInt(a), @backingInt(kv), if (sfx) |s| s.ptr else null, &buf, buf.len);
                if (r < 0) {
                    checks += 1;
                    fail("LLM_KV({s})({s}) threw '{s}'", .{ @tagName(a), @tagName(kv), std.mem.sliceTo(&buf, 0) });
                    continue;
                }
                const port = A.kvName(a, kv, sfx);
                expectStr(@tagName(kv), buf[0..@intCast(r)], port.slice());
            }
        }
    }
    // Past the end of the enum, LLM_KV_NAMES.at must throw.
    const n = count(A.Kv);
    for (n..n + 4) |i| {
        const r = ref_kv_name(@backingInt(A.Arch.llama), @intCast(i), null, &buf, buf.len);
        expectInt("kv past the enum throws", r, -1);
        if (r < 0) expectStr("kv past the enum", std.mem.sliceTo(&buf, 0), "map::at:  key not found");
    }
}

// -----------------------------------------------------------------------------
// Tensors

/// Runs `body` in a child with its stderr captured, and reports whether it
/// died of SIGABRT and the first line of what `ggml_abort` printed after
/// its `file:line: ` prefix.
fn abortsWith(body: *const fn (ctx: c_int) void, ctx: c_int, out: []u8) ?[]const u8 {
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) return null;
    const pid = fork();
    if (pid == 0) {
        _ = dup2(fds[1], 2);
        _ = close(fds[0]);
        body(ctx);
        _exit(0);
    }
    _ = close(fds[1]);
    var len: usize = 0;
    while (len < out.len) {
        const got = read(fds[0], out[len..].ptr, out.len - len);
        if (got <= 0) break;
        len += @intCast(got);
    }
    _ = close(fds[0]);
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    const sig = status & 0x7f;
    if (sig != sigabrt) return null;
    const text = out[0..len];
    const line = text[0 .. std.mem.findScalar(u8, text, '\n') orelse text.len];
    // Strip `file:line: `, which names the C++ source on one side and the
    // port's citation on the other.
    var rest = line;
    for (0..2) |_| rest = rest[(std.mem.findScalar(u8, rest, ':') orelse return rest) + 1 ..];
    return std.mem.trim(u8, rest, " ");
}

fn refTensorAbort(t: c_int) void {
    var buf: [256]u8 = undefined;
    _ = ref_tensor_name(@backingInt(A.Arch.llama), t, null, -1, -1, &buf, buf.len);
}

fn portTensorAbort(t: c_int) void {
    _ = A.tensorName(@fromBackingInt(@intCast(t)), null, -1, -1);
}

fn checkTensors() void {
    const suffixes = [_]?[:0]const u8{ null, "weight", "bias" };
    const ids = [_]i32{ -1, 0, 1, 9, 10, 123, std.math.maxInt(i32), std.math.minInt(i32) };
    var buf: [512]u8 = undefined;
    for (std.enums.values(A.Tensor)) |t| {
        const ti: c_int = @backingInt(t);
        if (harness_root.arch_table.tensor_names[@intCast(ti)] == null) {
            var rb: [4096]u8 = undefined;
            var pb: [4096]u8 = undefined;
            const r = abortsWith(&refTensorAbort, ti, &rb);
            const p = abortsWith(&portTensorAbort, ti, &pb);
            checks += 1;
            if (r == null or p == null) {
                fail("tensor {s}: aborts ref={} port={}", .{ @tagName(t), r != null, p != null });
            } else expectStr(@tagName(t), r.?, p.?);
            continue;
        }
        for (ids) |bid| for (ids[0..4]) |xid| for (suffixes) |sfx| {
            const r = ref_tensor_name(@backingInt(A.Arch.llama), ti, if (sfx) |s| s.ptr else null, bid, xid, &buf, buf.len);
            const port = A.tensorName(t, sfx, bid, xid);
            expectStr(@tagName(t), buf[0..@intCast(r)], port.slice());
        };
    }
    // Past the end of the enum, LLM_TENSOR_NAMES must miss: the C++ aborts.
    const n = count(A.Tensor);
    for (n..n + 2) |i| {
        var rb: [4096]u8 = undefined;
        checks += 1;
        if (abortsWith(&refTensorAbort, @intCast(i), &rb) == null) fail("tensor {d} past the enum has a name", .{i});
    }

    for (0..n + 4) |i| {
        var layer: c_int = 0;
        var op: c_int = 0;
        const r = ref_tensor_info(@intCast(i), &layer, &op, &buf, buf.len);
        if (i >= n) {
            expectInt("tensor info past the enum throws", r, -1);
            continue;
        }
        var exc: harness_root.impl.Exception = .{};
        if (A.tensorInfoFor(@fromBackingInt(@intCast(i)), &exc)) |info| {
            expectInt("tensor info throws", r, 0);
            expectInt("tensor info layer", layer, @backingInt(info.layer));
            expectInt("tensor info op", op, @intCast(info.op));
        } else |_| {
            expectInt("tensor info throws", r, -1);
            if (r < 0) expectStr("tensor info what()", std.mem.sliceTo(&buf, 0), exc.what());
        }
    }
}

pub fn main() void {
    checkArchs();
    const after_archs = checks;
    checkKeys();
    const after_keys = checks;
    checkTensors();
    std.debug.print("  architectures {d:>10} checks\n", .{after_archs});
    std.debug.print("  metadata keys {d:>10} checks\n", .{after_keys - after_archs});
    std.debug.print("  tensors       {d:>10} checks\n", .{checks - after_keys});
    if (failures != 0) {
        std.debug.print("FAIL: {d} of {d} arch results differ, Zig vs llama-arch.cpp\n", .{ failures, checks });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} arch results identical, Zig vs llama-arch.cpp\n", .{checks});
}
