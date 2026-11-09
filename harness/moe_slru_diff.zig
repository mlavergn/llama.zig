//! Compares `src/llama/moe_cache.zig`'s `Slru` with llama.cpp's
//! `moe_cache_slru`.
//!
//! **Not a port.** The harness behind `scripts/moe-slru-diff`. `context-diff`
//! cannot see the eviction policy: an expert read from a different slot
//! computes the same bits, so three injected faults in `Slru` passed
//! `--model moe-cache`. This drives both with the same random requests --
//! layer counts, expert counts, slot budgets from one slot to more than the
//! experts, skewed ids with repeats, requests past the budget -- and compares
//! every result, every fill, the hit count and every slot map after every
//! call.
//!
//! Usage: moe_slru_diff [--quick]

const std = @import("std");
const harness_root = @import("harness_root");

const Slru = harness_root.moe_cache.Slru;
const gpa = std.heap.c_allocator;

extern fn slru_ref_new(n_layer: c_int, n_expert: c_int, n_slots: c_int) ?*anyopaque;
extern fn slru_ref_free(p: ?*anyopaque) void;
extern fn slru_ref_set_map(p: ?*anyopaque, il: c_int, data: [*]i32) void;
extern fn slru_ref_plan(p: ?*anyopaque, il: c_int, ids: [*]const i32, n_ids: usize, fills: [*]i32, n_fills: *usize, n_hit: *usize) c_int;

var checks: usize = 0;
var failures: usize = 0;
var evictions: usize = 0;
var refused: usize = 0;
var hits: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    if (failures <= 20) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

fn expect(what: []const u8, seed: u64, step: usize, ref: i64, port: i64) void {
    checks += 1;
    if (ref != port) fail("seed {d} step {d}: {s}: ref {d} port {d}", .{ seed, step, what, ref, port });
}

fn run(seed: u64, n_layer: usize, n_expert: i32, n_slots: i32, n_steps: usize) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const ne: usize = @intCast(n_expert);
    const ref_maps = gpa.alloc(i32, n_layer * ne) catch unreachable;
    defer gpa.free(ref_maps);
    const port_maps = gpa.alloc(i32, n_layer * ne) catch unreachable;
    defer gpa.free(port_maps);
    @memset(ref_maps, -1);
    @memset(port_maps, -1);

    const ref = slru_ref_new(@intCast(n_layer), n_expert, n_slots);
    defer slru_ref_free(ref);
    var port: Slru = .{};
    defer port.deinit();
    port.init(n_layer, n_expert, n_slots);
    for (0..n_layer) |il| {
        slru_ref_set_map(ref, @intCast(il), ref_maps[il * ne ..].ptr);
        port.slot_map[il] = port_maps[il * ne ..].ptr;
    }

    var ids: [256]i32 = undefined;
    var ref_fills: [1024]i32 = undefined;
    var port_fills: std.ArrayList(Slru.Fill) = .empty;
    defer port_fills.deinit(gpa);

    for (0..n_steps) |step| {
        const il = r.uintLessThan(usize, n_layer);
        // Mostly a few experts, as a router's skew gives; now and then a
        // request larger than the budget.
        const budget: usize = @intCast(n_slots);
        const n_ids: usize = if (r.uintLessThan(u32, 16) == 0) ids.len else 1 + r.uintLessThan(usize, @min(ids.len, budget + 1));
        const hot = 1 + r.uintLessThan(usize, ne);
        for (ids[0..n_ids]) |*id| {
            const span: usize = if (r.boolean()) hot else ne;
            id.* = @intCast(r.uintLessThan(usize, span));
        }

        var ref_n_fills: usize = 0;
        var ref_hit: usize = 0;
        const ref_ok = slru_ref_plan(ref, @intCast(il), &ids, n_ids, &ref_fills, &ref_n_fills, &ref_hit);
        var port_hit: usize = 0;
        const port_ok = port.plan(@intCast(il), ids[0..n_ids], &port_fills, &port_hit);

        expect("plan", seed, step, ref_ok, @intFromBool(port_ok));
        if (ref_ok == 0) refused += 1;
        expect("n_hit", seed, step, @intCast(ref_hit), @intCast(port_hit));
        hits += ref_hit;
        expect("n_fills", seed, step, @intCast(ref_n_fills), @intCast(port_fills.items.len));
        for (port_fills.items, 0..) |f, i| {
            if (i >= ref_n_fills) break;
            expect("fill expert", seed, step, ref_fills[2 * i], f.expert);
            expect("fill slot", seed, step, ref_fills[2 * i + 1], f.slot);
        }
        checks += 1;
        if (!std.mem.eql(i32, ref_maps, port_maps)) fail("seed {d} step {d}: the slot maps differ", .{ seed, step });
        if (failures > 20) return;
        evictions += ref_n_fills;
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const quick = args.len > 1 and std.mem.eql(u8, args[1], "--quick");
    const n_steps: usize = if (quick) 300 else 3000;

    var seed: u64 = 1;
    for ([_]usize{ 1, 2, 3, 7 }) |n_layer| {
        for ([_]i32{ 2, 4, 8, 64, 257 }) |n_expert| {
            for ([_]i32{ 1, 2, 3, 4, 5, 8, 16, 31, 64, 300 }) |n_slots| {
                if (n_slots > n_expert * @as(i32, @intCast(n_layer)) + 3) continue;
                run(seed, n_layer, n_expert, n_slots, n_steps);
                seed += 1;
            }
        }
    }

    // A gate that never evicted or refused would be saying nothing.
    if (hits == 0 or evictions == 0 or refused == 0) fail("hits {d} fills {d} refused {d}", .{ hits, evictions, refused });
    std.debug.print("  {d} runs, {d} hits, {d} fills, {d} refused\n", .{ seed - 1, hits, evictions, refused });
    if (failures != 0) {
        std.debug.print("FAIL: {d} of {d} SLRU results differ, Zig vs llama-moe-cache.cpp\n", .{ failures, checks });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} SLRU results identical, Zig vs llama-moe-cache.cpp\n", .{checks});
}
