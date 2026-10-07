//! Compares `src/llama/libcxx.zig` against the libc++ templates it reproduces.
//!
//! **Not a port.** The harness behind `scripts/libcxx-diff`: it calls the
//! reference instantiations in `harness/libcxx_ref.cpp` and the Zig on the
//! same inputs, in one process, and requires every result to be identical --
//! the whole permuted array for the sorts, the bits for the floats.
//!
//! A sort is only distinguishable from another correct sort by the order it
//! leaves equal elements in, so the inputs are built to tie: few distinct
//! logits, all equal, runs, and the organ-pipe and already-sorted shapes that
//! steer introsort into its partition and insertion-sort exits. Lengths
//! straddle every threshold in `sort.h` -- the sorting networks below 6,
//! insertion sort below 24, the ninther above 128 -- and the 32000 the
//! sampler sees on a real vocabulary.

const std = @import("std");
const libcxx = @import("libcxx");

const TokenData = extern struct {
    id: i32,
    logit: f32,
    p: f32,
};

extern fn ref_sort_td(d: [*]TokenData, n: usize) void;
extern fn ref_partial_sort_td(d: [*]TokenData, n: usize, k: usize) void;
extern fn ref_sort_idx(idx: [*]usize, n: usize, scores: [*]const f32) void;
extern fn ref_partial_sort_idx(idx: [*]usize, n: usize, k: usize, scores: [*]const f32) void;
extern fn ref_mt(seed: u32, out: [*]u32, n: usize) void;
extern fn ref_uniform_f64(seed: u32, out: [*]f64, n: usize) void;
extern fn ref_uniform_f32(seed: u32, out: [*]f32, n: usize) void;
extern fn ref_discrete(d: [*]const TokenData, n: usize, seed: u32, draws: c_int, out: [*]c_int) void;
extern fn ref_hash_string(s: [*]const u8, n: usize) usize;
extern fn ref_next_prime(n: usize) usize;
extern fn ref_umap(blob: [*]const u8, offs: [*]const usize, n: usize, out: [*]c_int, bc: *usize) usize;

const PqItem = extern struct { score: f32, left: c_int };
extern fn ref_pq(items: [*]const PqItem, ops: [*]const c_int, n_ops: usize, out: [*]PqItem) usize;

/// llama-vocab.cpp's `llm_bigram_spm::comparator`, as `ref_pq` has it.
fn pqLess(_: void, l: PqItem, r: PqItem) bool {
    return (l.score < r.score) or (l.score == r.score and l.left > r.left);
}

/// Strings for the hash and the map: every length to 300, so CityHash's
/// four length classes and its 64-byte loop are all reached, over bytes
/// that set the high bits a 32-bit shift would drop.
fn randomString(r: std.Random, buf: []u8, len: usize) []const u8 {
    for (buf[0..len]) |*b| b.* = switch (r.uintLessThan(u8, 4)) {
        0 => r.int(u8),
        1 => 0xFF,
        2 => 'a' + r.uintLessThan(u8, 3),
        else => 0x80 | r.uintLessThan(u8, 0x80),
    };
    return buf[0..len];
}

fn greaterLogit(_: void, a: TokenData, b: TokenData) bool {
    return a.logit > b.logit;
}

const IdxCtx = struct { scores: []const f32 };

fn lessScore(ctx: IdxCtx, a: usize, b: usize) bool {
    return ctx.scores[a] < ctx.scores[b];
}

const Shape = enum { random, ties2, ties8, equal, ascending, descending, organ, runs };

fn fillTokens(r: std.Random, d: []TokenData, shape: Shape) void {
    for (d, 0..) |*t, i| {
        const fi: f32 = @floatFromInt(i);
        const n: f32 = @floatFromInt(d.len);
        t.id = @intCast(i);
        t.p = r.float(f32);
        t.logit = switch (shape) {
            .random => r.float(f32) * 2.0 - 1.0,
            .ties2 => @floatFromInt(r.intRangeLessThan(u8, 0, 2)),
            .ties8 => @floatFromInt(r.intRangeLessThan(u8, 0, 8)),
            .equal => 0.5,
            .ascending => fi,
            .descending => -fi,
            .organ => if (fi < n / 2) fi else n - fi,
            .runs => @floatFromInt(i / 7),
        };
    }
}

var checks: usize = 0;
var failures: usize = 0;

fn expectTokens(what: []const u8, n: usize, shape: Shape, want: []const TokenData, got: []const TokenData) void {
    checks += 1;
    if (std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) return;
    failures += 1;
    if (failures <= 10) {
        var first: usize = 0;
        while (first < want.len and want[first].id == got[first].id and
            @as(u32, @bitCast(want[first].logit)) == @as(u32, @bitCast(got[first].logit))) first += 1;
        std.debug.print("FAIL {s} n={d} shape={s}: first difference at {d} (ref id {d}, zig id {d})\n", .{
            what, n, @tagName(shape), first, want[first].id, got[first].id,
        });
    }
}

pub fn main() !void {
    const gpa = std.heap.c_allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();

    const lengths = [_]usize{
        0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   10,  11,  12,  13,  15,  16,  17,   20,   23,   24,   25,   26,    31, 32, 33,
        40, 50, 63, 64, 65, 100, 127, 128, 129, 130, 200, 255, 256, 257, 500, 777, 1000, 1024, 1025, 2048, 4096, 32000,
    };

    const ref_buf = try gpa.alloc(TokenData, 32000);
    defer gpa.free(ref_buf);
    const zig_buf = try gpa.alloc(TokenData, 32000);
    defer gpa.free(zig_buf);

    // std::sort and std::partial_sort over the sampler's element type.
    for (lengths) |n| {
        const reps: usize = if (n <= 256) 40 else if (n <= 4096) 8 else 2;
        for (std.enums.values(Shape)) |shape| {
            for (0..reps) |_| {
                fillTokens(r, ref_buf[0..n], shape);
                // Shuffle the layout so a shape is not also an id order.
                r.shuffle(TokenData, ref_buf[0..n]);
                @memcpy(zig_buf[0..n], ref_buf[0..n]);
                ref_sort_td(ref_buf.ptr, n);
                libcxx.sort(TokenData, zig_buf[0..n], {}, greaterLogit);
                expectTokens("sort", n, shape, ref_buf[0..n], zig_buf[0..n]);

                const ks = [_]usize{ 0, 1, 2, 5, 40, 128, n / 2, n };
                for (ks) |k_raw| {
                    const k = @min(k_raw, n);
                    fillTokens(r, ref_buf[0..n], shape);
                    r.shuffle(TokenData, ref_buf[0..n]);
                    @memcpy(zig_buf[0..n], ref_buf[0..n]);
                    ref_partial_sort_td(ref_buf.ptr, n, k);
                    libcxx.partialSort(TokenData, zig_buf[0..n], k, {}, greaterLogit);
                    // The whole array, not just the head: the tail's order is
                    // what the next sampler in a chain reads.
                    expectTokens("partial_sort", n, shape, ref_buf[0..n], zig_buf[0..n]);
                }
            }
        }
    }

    // typical's index sort: sorting indices by a captured score array.
    //
    // Two score shapes: few levels, so most compare equal, and continuous
    // values with a sprinkling of exact ties -- what typical's shifted
    // scores look like, where two candidates either side of the entropy tie.
    // The second shape at 32000 is what found a difference the first missed.
    {
        const scores = try gpa.alloc(f32, 32000);
        defer gpa.free(scores);
        const a = try gpa.alloc(usize, 32000);
        defer gpa.free(a);
        const b = try gpa.alloc(usize, 32000);
        defer gpa.free(b);
        for (lengths) |n| {
            for (0..10) |rep| {
                if (rep % 2 == 0) {
                    for (scores[0..n]) |*s| s.* = @floatFromInt(r.intRangeLessThan(u8, 0, 6));
                } else {
                    for (scores[0..n]) |*s| s.* = r.float(f32) * 10.0;
                    // Make a few exact ties.
                    var t: usize = 0;
                    while (n > 1 and t < n / 50 + 1) : (t += 1) {
                        scores[r.uintLessThan(usize, n)] = scores[r.uintLessThan(usize, n)];
                    }
                }
                for (0..n) |i| a[i] = i;
                @memcpy(b[0..n], a[0..n]);
                ref_sort_idx(a.ptr, n, scores.ptr);
                libcxx.sort(usize, b[0..n], IdxCtx{ .scores = scores[0..n] }, lessScore);
                checks += 1;
                if (!std.mem.eql(usize, a[0..n], b[0..n])) {
                    failures += 1;
                    std.debug.print("FAIL index sort n={d}\n", .{n});
                }

                // The heap path directly, which introsort reaches only when
                // its depth runs out -- rarely on random input, and exactly
                // what typical's 32000-candidate sort hit. An arithmetic
                // element type heapifies differently in libc++.
                for ([_]usize{ 1, 7, 128, n / 2, n }) |k_raw| {
                    const k = @min(k_raw, n);
                    for (0..n) |i| a[i] = i;
                    r.shuffle(usize, a[0..n]);
                    @memcpy(b[0..n], a[0..n]);
                    ref_partial_sort_idx(a.ptr, n, k, scores.ptr);
                    libcxx.partialSort(usize, b[0..n], k, IdxCtx{ .scores = scores[0..n] }, lessScore);
                    checks += 1;
                    if (!std.mem.eql(usize, a[0..n], b[0..n])) {
                        failures += 1;
                        std.debug.print("FAIL index partial_sort n={d} k={d}\n", .{ n, k });
                    }
                }
            }
        }
    }

    // The engine and the two canonical draws, on bits.
    {
        var ru: [3000]u32 = undefined;
        var rd: [3000]f64 = undefined;
        var rf: [3000]f32 = undefined;
        for ([_]u32{ 0, 1, 5, 42, 1234, 5489, 0xffffffff, 0x80000000 }) |seed| {
            ref_mt(seed, &ru, ru.len);
            var g = libcxx.Mt19937.init(seed);
            for (ru) |want| {
                checks += 1;
                if (g.next() != want) {
                    failures += 1;
                    std.debug.print("FAIL mt19937 seed={d}\n", .{seed});
                    break;
                }
            }

            ref_uniform_f64(seed, &rd, rd.len);
            g = libcxx.Mt19937.init(seed);
            for (rd) |want| {
                checks += 1;
                const got = libcxx.uniformF64(&g);
                if (@as(u64, @bitCast(got)) != @as(u64, @bitCast(want))) {
                    failures += 1;
                    std.debug.print("FAIL uniform<double> seed={d}\n", .{seed});
                    break;
                }
            }

            ref_uniform_f32(seed, &rf, rf.len);
            g = libcxx.Mt19937.init(seed);
            for (rf) |want| {
                checks += 1;
                const got = libcxx.uniformF32(&g);
                if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(want))) {
                    failures += 1;
                    std.debug.print("FAIL uniform<float> seed={d}\n", .{seed});
                    break;
                }
            }
        }
    }

    // discrete_distribution, a fresh one per draw over the same engine.
    {
        const Weights = struct {
            d: []const TokenData,
            len: usize,
            pub fn get(self: @This(), i: usize) f64 {
                return self.d[i].p;
            }
        };
        var out: [32]c_int = undefined;
        for ([_]usize{ 0, 1, 2, 3, 7, 40, 1000 }) |n| {
            for ([_]u32{ 1, 9, 77 }) |seed| {
                fillTokens(r, ref_buf[0..n], .random);
                // Zeros and ties among the weights.
                for (ref_buf[0..n], 0..) |*t, i| {
                    if (i % 5 == 0) t.p = 0.0;
                    if (i % 7 == 0) t.p = 0.25;
                }
                ref_discrete(ref_buf.ptr, n, seed, out.len, &out);
                var g = libcxx.Mt19937.init(seed);
                for (out) |want| {
                    checks += 1;
                    const got = try libcxx.discreteDraw(gpa, Weights{ .d = ref_buf[0..n], .len = n }, &g);
                    if (got != @as(usize, @intCast(want))) {
                        failures += 1;
                        std.debug.print("FAIL discrete n={d} seed={d}: ref {d} zig {d}\n", .{ n, seed, want, got });
                        break;
                    }
                }
            }
        }
    }

    // std::hash<std::string>: every length 0..300, many strings each.
    {
        var buf: [300]u8 = undefined;
        for (0..301) |len| {
            for (0..200) |_| {
                const s = randomString(r, &buf, len);
                checks += 1;
                const want = ref_hash_string(s.ptr, s.len);
                const got = libcxx.hashString(s);
                if (want != got) {
                    failures += 1;
                    if (failures <= 10) std.debug.print("FAIL hash len={d}: ref {x} zig {x}\n", .{ len, want, got });
                }
            }
        }
    }

    // __next_prime against the dylib.
    {
        var n: usize = 0;
        while (n < 200000) : (n += 1) {
            checks += 1;
            if (ref_next_prime(n) != libcxx.nextPrime(n)) {
                failures += 1;
                if (failures <= 10) std.debug.print("FAIL next_prime({d}): ref {d} zig {d}\n", .{ n, ref_next_prime(n), libcxx.nextPrime(n) });
            }
        }
        for ([_]usize{ 1 << 20, 1 << 22, 3_000_017, 16_777_216, 4294967291, 4294967292 }) |big| {
            checks += 1;
            if (ref_next_prime(big) != libcxx.nextPrime(big)) {
                failures += 1;
                std.debug.print("FAIL next_prime({d})\n", .{big});
            }
        }
    }

    // unordered_map<string, int>: iteration order and bucket count, built
    // `m[key] = i` as token_to_id is, at sizes through several rehashes,
    // with duplicate keys so assignment over an existing key is reached.
    {
        var blob: std.ArrayList(u8) = .empty;
        defer blob.deinit(gpa);
        var offs: std.ArrayList(usize) = .empty;
        defer offs.deinit(gpa);
        var buf: [80]u8 = undefined;
        const out = try gpa.alloc(c_int, 300000);
        defer gpa.free(out);
        for ([_]usize{ 0, 1, 2, 3, 5, 12, 13, 29, 30, 97, 1000, 32000, 151936, 248320 }) |n| {
            blob.clearRetainingCapacity();
            offs.clearRetainingCapacity();
            try offs.append(gpa, 0);
            for (0..n) |i| {
                const len = if (i % 9 == 0) r.uintLessThan(usize, 3) else r.uintLessThan(usize, 80);
                try blob.appendSlice(gpa, randomString(r, &buf, len));
                try offs.append(gpa, blob.items.len);
            }
            var rbc: usize = 0;
            const rn = ref_umap(blob.items.ptr, offs.items.ptr, n, out.ptr, &rbc);

            var m = try libcxx.StringMap(c_int).init(gpa);
            defer m.deinit();
            for (0..n) |i| {
                const slot = try m.getOrPut(blob.items[offs.items[i]..offs.items[i + 1]], 0);
                slot.value.* = @intCast(i);
            }
            checks += 1;
            var same = rn == m.count() and rbc == m.bucketCount();
            var it = m.iterator();
            var k: usize = 0;
            while (it.next()) |e| : (k += 1) {
                if (k >= rn or out[k] != e.value.*) same = false;
            }
            if (k != rn) same = false;
            if (!same) {
                failures += 1;
                std.debug.print("FAIL unordered_map n={d}: ref size {d} buckets {d}, zig size {d} buckets {d}\n", .{ n, rn, rbc, m.count(), m.bucketCount() });
            }
        }
    }

    // priority_queue with the SPM bigram comparator: ties on score, equal
    // `left`, and NaN scores, under interleaved pushes and pops.
    {
        var items: [64]PqItem = undefined;
        var ops: [400]c_int = undefined;
        var rout: [400]PqItem = undefined;
        for (0..3000) |_| {
            for (&items) |*it| {
                it.score = switch (r.uintLessThan(u8, 6)) {
                    0 => std.math.nan(f32),
                    1, 2 => @floatFromInt(r.uintLessThan(u8, 3)),
                    else => r.float(f32),
                };
                it.left = r.intRangeLessThan(c_int, 0, 6);
            }
            const n_ops = r.uintLessThan(usize, ops.len + 1);
            for (ops[0..n_ops]) |*o| o.* = if (r.uintLessThan(u8, 3) == 0) -1 else r.intRangeLessThan(c_int, 0, items.len);
            const rn = ref_pq(&items, &ops, n_ops, &rout);

            var q: libcxx.PriorityQueue(PqItem, void, pqLess) = .{ .ctx = {} };
            defer q.deinit(gpa);
            var zout: [400]PqItem = undefined;
            var zn: usize = 0;
            for (ops[0..n_ops]) |o| {
                if (o >= 0) {
                    try q.push(gpa, items[@intCast(o)]);
                } else if (!q.empty()) {
                    zout[zn] = q.pop();
                    zn += 1;
                }
            }
            while (!q.empty()) {
                zout[zn] = q.pop();
                zn += 1;
            }
            checks += 1;
            if (rn != zn or !std.mem.eql(u8, std.mem.sliceAsBytes(rout[0..rn]), std.mem.sliceAsBytes(zout[0..zn]))) {
                failures += 1;
                if (failures <= 10) std.debug.print("FAIL priority_queue: {d} ops\n", .{n_ops});
            }
        }
    }

    if (failures == 0) {
        std.debug.print("PASS: {d} libc++ results identical, Zig vs the SDK templates\n", .{checks});
        return;
    }
    std.debug.print("FAIL: {d} of {d} checks differ\n", .{ failures, checks });
    std.process.exit(1);
}
