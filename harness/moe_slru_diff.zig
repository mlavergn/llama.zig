//! Compares `src/llama/moe_cache.zig`'s `Lru` with llama.cpp's
//! `moe_cache_lru` (the ClassLRU).
//!
//! **Not a port.** The harness behind `scripts/moe-slru-diff`. `context-diff`
//! cannot see the eviction policy: an expert read from a different slot
//! computes the same bits, so three injected faults in the SLRU passed
//! `--model moe-cache`. This drives both with the same random requests --
//! layer counts, expert counts, slot budgets from one slot to more than the
//! experts, skewed ids with repeats, requests past the budget, CPU shares
//! with their carry and zero-slot counts -- and compares every result, every
//! fill, every CPU miss, the hit count, the filled count, the carry and
//! every slot map after every call.
//!
//! Usage: moe_slru_diff [--quick]

const std = @import("std");
const harness_root = @import("harness_root");

const Lru = harness_root.moe_cache.Lru;
const CpuShare = harness_root.moe_cache.CpuShare;
const gpa = std.heap.c_allocator;

extern fn slru_ref_new(n_layer: c_int, n_expert: c_int, n_slots: c_int) ?*anyopaque;
extern fn slru_ref_free(p: ?*anyopaque) void;
extern fn slru_ref_set_map(p: ?*anyopaque, il: c_int, data: [*]i32) void;
extern fn slru_ref_empty(p: ?*anyopaque) c_int;
extern fn slru_ref_n_filled(p: ?*anyopaque) c_int;
extern fn slru_ref_plan(p: ?*anyopaque, il: c_int, ids: [*]const i32, n_ids: usize, n_tokens: i64, n_used: i32, n_zero: i32, share_q: u32, acc: *u32, fills: [*]i32, n_fills: *usize, cpu: [*]i32, n_cpu: *usize, n_hit: *usize) c_int;

var checks: usize = 0;
var failures: usize = 0;
var evictions: usize = 0;
var refused: usize = 0;
var hits: usize = 0;
var cpu_misses: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    if (failures <= 20) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

fn expect(what: []const u8, seed: u64, step: usize, ref: i64, port: i64) void {
    checks += 1;
    if (ref != port) fail("seed {d} step {d}: {s}: ref {d} port {d}", .{ seed, step, what, ref, port });
}

fn run(seed: u64, n_layer: usize, n_expert: i32, n_slots: i32, n_steps: usize, sticky: bool) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const ne: usize = @intCast(n_expert);
    const ref_maps = gpa.alloc(i32, n_layer * ne) catch unreachable;
    defer gpa.free(ref_maps);
    const port_maps = gpa.alloc(i32, n_layer * ne) catch unreachable;
    defer gpa.free(port_maps);

    const ref = slru_ref_new(@intCast(n_layer), n_expert, n_slots);
    defer slru_ref_free(ref);
    var port: Lru = .{};
    defer port.deinit();
    port.init(n_layer, n_expert, n_slots);
    // the slot maps start at `empty`, as the cache fills them
    @memset(ref_maps, slru_ref_empty(ref));
    @memset(port_maps, port.empty);
    for (0..n_layer) |il| {
        slru_ref_set_map(ref, @intCast(il), ref_maps[il * ne ..].ptr);
        port.slot_map[il] = port_maps[il * ne ..].ptr;
    }

    // a share for the run, and the carries of each layer, as the cache keeps them
    const shares = [_]u32{ 0, 16384, 32768, 52429, 65536 };
    const share_q: u32 = if (r.uintLessThan(u32, 4) == 0) r.uintLessThan(u32, 65537) else shares[r.uintLessThan(usize, shares.len)];
    var ref_acc: [8]u32 = @splat(0);
    var port_acc: [8]u32 = @splat(0);

    var ids: [256]i32 = undefined;
    var ref_fills: [1024]i32 = undefined;
    var ref_cpu: [1024]i32 = undefined;
    var port_fills: std.ArrayList(Lru.Fill) = .empty;
    defer port_fills.deinit(gpa);
    var port_cpu: std.ArrayList(Lru.Fill) = .empty;
    defer port_cpu.deinit(gpa);

    for (0..n_steps) |step| {
        const il = r.uintLessThan(usize, n_layer);
        // Mostly a few experts, as a router's skew gives; now and then a
        // request larger than the budget. The ids are n_tokens rows of
        // n_used, distinct within a row as a router's top-k mostly are.
        const budget: usize = @intCast(n_slots);
        const n_used: usize = 1 + r.uintLessThan(usize, @min(ne, 8));
        const max_tokens = ids.len / n_used;
        // a sticky run's layer 0 decodes one token at a time, so the decay
        // between uses stays small
        const n_tokens: usize = if (sticky and il == 0) 1 else if (r.uintLessThan(u32, 16) == 0) max_tokens else 1 + r.uintLessThan(usize, @min(max_tokens, budget / n_used + 1));
        const n_ids = n_tokens * n_used;
        const hot = 1 + r.uintLessThan(usize, ne);
        const distinct = r.uintLessThan(u32, 4) != 0;
        for (0..n_tokens) |t| {
            for (0..n_used) |k| {
                while (true) {
                    const span: usize = if (r.boolean()) hot else ne;
                    const id: i32 = @intCast(r.uintLessThan(usize, span));
                    var dup = false;
                    if (distinct) {
                        for (ids[t * n_used ..][0..k]) |x| dup = dup or x == id;
                    }
                    if (!dup) {
                        ids[t * n_used + k] = id;
                        break;
                    }
                }
            }
        }
        // a sticky run selects expert 0 in every row of layer 0, so its CRF
        // climbs past 128 into the top class, which skewed random ids never
        // reach in a run's length
        if (sticky and il == 0) {
            for (0..n_tokens) |t| {
                var has0 = false;
                for (ids[t * n_used ..][0..n_used]) |x| has0 = has0 or x == 0;
                if (!has0) ids[t * n_used] = 0;
            }
        }
        const n_zero: i32 = if (share_q == 0) 0 else @intCast(@min(n_used, 64) - r.uintLessThan(usize, @min(n_used, 3)));
        // copy() asks for a CPU part only once the cache is full
        const sq: u32 = if (slru_ref_n_filled(ref) == n_slots) share_q else 0;

        var ref_n_fills: usize = 0;
        var ref_n_cpu: usize = 0;
        var ref_hit: usize = 0;
        const ref_ok = slru_ref_plan(ref, @intCast(il), &ids, n_ids, @intCast(n_tokens), @intCast(n_used), n_zero, sq, &ref_acc[il], &ref_fills, &ref_n_fills, &ref_cpu, &ref_n_cpu, &ref_hit);
        var port_hit: usize = 0;
        const port_ok = port.plan(@intCast(il), ids[0..n_ids], @intCast(n_tokens), @intCast(n_used), n_zero, CpuShare{ .share_q = sq, .acc = &port_acc[il] }, &port_fills, &port_cpu, &port_hit);

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
        expect("n_cpu", seed, step, @intCast(ref_n_cpu), @intCast(port_cpu.items.len));
        for (port_cpu.items, 0..) |f, i| {
            if (i >= ref_n_cpu) break;
            expect("cpu expert", seed, step, ref_cpu[2 * i], f.expert);
            expect("cpu slot", seed, step, ref_cpu[2 * i + 1], f.slot);
        }
        expect("carry", seed, step, ref_acc[il], port_acc[il]);
        expect("n_filled", seed, step, slru_ref_n_filled(ref), port.n_filled);
        checks += 1;
        if (!std.mem.eql(i32, ref_maps, port_maps)) fail("seed {d} step {d}: the slot maps differ", .{ seed, step });
        if (failures > 20) return;
        evictions += ref_n_fills;
        cpu_misses += ref_n_cpu;
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
                run(seed, n_layer, n_expert, n_slots, n_steps, false);
                seed += 1;
            }
        }
    }

    for ([_]usize{ 1, 2 }) |n_layer| {
        for ([_]i32{ 4, 64 }) |n_expert| {
            for ([_]i32{ 3, 8, 16 }) |n_slots| {
                run(seed, n_layer, n_expert, n_slots, n_steps, true);
                seed += 1;
            }
        }
    }

    // A gate that never evicted or refused would be saying nothing.
    if (hits == 0 or evictions == 0 or refused == 0 or cpu_misses == 0) fail("hits {d} fills {d} refused {d} cpu {d}", .{ hits, evictions, refused, cpu_misses });
    std.debug.print("  {d} runs, {d} hits, {d} fills, {d} on the CPU, {d} refused\n", .{ seed - 1, hits, evictions, cpu_misses, refused });
    if (failures != 0) {
        std.debug.print("FAIL: {d} of {d} MoE cache LRU results differ, Zig vs llama-moe-cache.cpp\n", .{ failures, checks });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} MoE cache LRU results identical, Zig vs llama-moe-cache.cpp\n", .{checks});
}
