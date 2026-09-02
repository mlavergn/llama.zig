//! Building the i-quant codebooks and their nearest-neighbour maps.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`):
//! `iq2_data_index` (2833), `iq2_grid_size` (2841), `iq2_compare_func`,
//! `iq2xs_init_impl` (2838), `iq2xs_free_impl`, and the `iq3` equivalents at
//! 3679-3900. The tables they embed are in `grids.zig`, extracted separately.
//!
//! # What is being built
//!
//! Quantizing to an i-quant means finding, for each group of 8 weights, the
//! closest entry in a codebook of a few hundred 8-element vectors. Searching
//! all of them per group would be far too slow, so this precomputes:
//!
//! - **`grid`** -- the codebook itself, unpacked from a table of 2-bit digit
//!   strings into eight `int8` values per entry.
//! - **`map`** -- indexed by a *quantized* 8-element pattern. A non-negative
//!   entry is an exact codebook hit. A negative entry `-(n+1)` points into...
//! - **`neighbours`** -- for patterns with no exact entry, a short list of the
//!   nearest codebook entries to try. Length-prefixed, so `neighbours[n]` is a
//!   count followed by that many indices.
//!
//! So the quantizer's inner loop is a table lookup that either hits or yields
//! a handful of candidates, rather than a scan of the whole codebook.
//!
//! # Process-global, and built once
//!
//! These live in process-global state, built lazily on first use and freed by
//! `ggml_quantize_free`. That is the C's design and the reason
//! `ggml_quantize_init` exists at all. `quantize.zig` calls into here.
//!
//! # No OpenMP
//!
//! The C parallelises the neighbour search with OpenMP, in three passes so the
//! output is order-independent: count per entry, prefix-sum the offsets, then
//! fill. Our build has OpenMP off (see `CLAUDE.md`), so the `#pragma`s compile
//! away and the C runs exactly the serial loop written here. The three-pass
//! structure is kept regardless -- it is what makes the result deterministic,
//! and it is the shape a future `std.Thread` version would need.

const std = @import("std");
const impl = @import("../impl.zig");
const grids = @import("grids.zig");
const c = impl.c;

/// Ports `NGRID_IQ1S` (ggml-common.h:1131 @c1d0e7a00).
const ngrid_iq1s = 2048;

/// Ports `iq2_entry_t` (ggml-quants.c:2824 @c1d0e7a00).
const Iq2Entry = struct {
    grid: ?[*]u64 = null,
    map: ?[*]i32 = null,
    neighbours: ?[*]u16 = null,
};

/// Ports `iq2_data` (ggml-quants.c:2826 @c1d0e7a00). Four slots: xxs, xs, 1-bit, s.
var iq2_data = [_]Iq2Entry{.{}} ** 4;

/// Ports `iq2_data_index` (ggml-quants.c:2833 @c1d0e7a00).
///
/// `iq1_s` and `iq1_m` share slot 2: they index the same 2048-entry codebook.
fn iq2DataIndex(t: c.enum_ggml_type) usize {
    return switch (t) {
        c.GGML_TYPE_IQ2_XXS => 0,
        c.GGML_TYPE_IQ2_XS => 1,
        c.GGML_TYPE_IQ1_S, c.GGML_TYPE_IQ1_M => 2,
        c.GGML_TYPE_IQ2_S => 3,
        else => impl.abort("iq2_data_index: unsupported type"),
    };
}

/// Ports `iq2_grid_size` (ggml-quants.c:2840 @c1d0e7a00).
fn iq2GridSize(t: c.enum_ggml_type) usize {
    return switch (t) {
        c.GGML_TYPE_IQ2_XXS => 256,
        c.GGML_TYPE_IQ2_XS => 512,
        c.GGML_TYPE_IQ1_S, c.GGML_TYPE_IQ1_M => ngrid_iq1s,
        c.GGML_TYPE_IQ2_S => 1024,
        else => impl.abort("iq2_grid_size: unsupported type"),
    };
}

/// One `(distance, index)` pair in the neighbour sort.
///
/// The C sorts a flat `int[2*grid_size]` with `qsort` and a comparator that
/// falls back to the index on a distance tie. A pair struct says the same
/// thing, and the tie-break is what makes the result deterministic rather than
/// dependent on the sort's stability.
const Dist = struct {
    d2: i32,
    index: i32,

    /// Ports `iq2_compare_func` (ggml-quants.c:2847 @c1d0e7a00).
    fn lessThan(_: void, a: Dist, b: Dist) bool {
        if (a.d2 != b.d2) return a.d2 < b.d2;
        return a.index < b.index;
    }
};

/// The 8-position vector a 2-bit digit string decodes to.
///
/// Each 2-bit digit `l` becomes `2l + 1`, so the values are the odd numbers
/// 1, 3, 5, 7 -- centred on zero once the sign byte is applied, and never
/// zero, which is what lets a sign bit carry information for every element.
inline fn unpackPositions(bits: u16, pos: *[8]i8) void {
    for (0..8) |i| {
        const l: i32 = @intCast((bits >> @intCast(2 * i)) & 0x3);
        pos[i] = @intCast(2 * l + 1);
    }
}

/// Ports `iq2xs_init_impl` (ggml-quants.c:2853 @c1d0e7a00).
///
/// Idempotent: returns immediately if this type's codebook is already built.
pub export fn iq2xs_init_impl(t: c.enum_ggml_type) void {
    const gindex = iq2DataIndex(t);
    const grid_size = iq2GridSize(t);
    if (iq2_data[gindex].grid != null) return;

    // How many *distance tiers* of neighbours to keep, not how many
    // neighbours: the loop below stops after `nwant` distinct distances, so
    // the list length varies with how many entries tie at each distance.
    const nwant: i32 = switch (t) {
        c.GGML_TYPE_IQ1_S, c.GGML_TYPE_IQ1_M => 3,
        c.GGML_TYPE_IQ2_S => 1,
        else => 2,
    };

    const kgrid: []const u16 = switch (t) {
        c.GGML_TYPE_IQ2_XXS => &grids.kgrid_2bit_256,
        c.GGML_TYPE_IQ2_XS => &grids.kgrid_2bit_512,
        c.GGML_TYPE_IQ1_S, c.GGML_TYPE_IQ1_M => &grids.kgrid_1bit_2048,
        else => &grids.kgrid_2bit_1024,
    };

    // 43692 = the number of distinct 8-digit base-4 patterns the map indexes.
    const kmap_size: usize = 43692;

    const the_grid: [*]u64 = @ptrCast(@alignCast(std.c.malloc(grid_size * @sizeOf(u64)).?));
    for (0..grid_size) |k| {
        const pos: *[8]i8 = @ptrCast(&the_grid[k]);
        unpackPositions(kgrid[k], pos);
    }
    iq2_data[gindex].grid = the_grid;

    const kmap: [*]i32 = @ptrCast(@alignCast(std.c.malloc(kmap_size * @sizeOf(i32)).?));
    iq2_data[gindex].map = kmap;
    for (0..kmap_size) |i| kmap[i] = -1;

    // Re-encode each codebook entry back to its pattern index, so an exact
    // match can be found by lookup.
    for (0..grid_size) |i| {
        const aux8: *const [8]u8 = @ptrCast(&the_grid[i]);
        var index: u16 = 0;
        for (0..8) |k| {
            const q: u16 = (aux8[k] - 1) / 2;
            index |= q << @intCast(2 * k);
        }
        kmap[index] = @intCast(i);
    }

    const n_per_i: [*]i32 = @ptrCast(@alignCast(std.c.malloc(kmap_size * @sizeOf(i32)).?));
    defer std.c.free(n_per_i);

    const dist_mem = std.c.malloc(grid_size * @sizeOf(Dist)).?;
    defer std.c.free(dist_mem);
    const dists: [*]Dist = @ptrCast(@alignCast(dist_mem));

    // Pass 1: count each pattern's neighbours.
    var num_neighbors: usize = 0;
    var num_not_in_map: usize = 0;
    for (0..kmap_size) |i| {
        if (kmap[i] >= 0) {
            n_per_i[i] = 0;
            continue;
        }
        num_not_in_map += 1;
        const n = countNeighbours(i, the_grid, grid_size, dists, nwant);
        n_per_i[i] = @intCast(n);
        num_neighbors += n;
    }

    // Pass 2: prefix-sum into offsets, so pass 3 can write independently.
    // The `1 +` is the length prefix each list carries.
    const kneighbors: [*]u16 = @ptrCast(@alignCast(std.c.malloc((num_neighbors + num_not_in_map) * @sizeOf(u16)).?));
    iq2_data[gindex].neighbours = kneighbors;

    const offsets: [*]i32 = @ptrCast(@alignCast(std.c.malloc(kmap_size * @sizeOf(i32)).?));
    defer std.c.free(offsets);

    var counter: i32 = 0;
    for (0..kmap_size) |i| {
        if (kmap[i] >= 0) {
            offsets[i] = -1;
            continue;
        }
        offsets[i] = counter;
        counter += 1 + n_per_i[i];
    }

    // Pass 3: write each list, and point the map at it.
    for (0..kmap_size) |i| {
        if (kmap[i] >= 0) continue;
        var local: usize = @intCast(offsets[i]);
        // Negative map entries are `-(offset + 1)`, so that offset zero is
        // still distinguishable from an exact hit at index zero.
        kmap[i] = -(offsets[i] + 1);
        const start = local;
        local += 1;

        const n = fillNeighbours(i, the_grid, grid_size, dists, nwant, kneighbors, &local);
        kneighbors[start] = @intCast(n);
    }
}

/// The distance sort both neighbour passes begin with.
fn sortByDistance(pattern: usize, grid: [*]u64, grid_size: usize, dists: [*]Dist) void {
    var pos: [8]i8 = undefined;
    unpackPositions(@intCast(pattern & 0xFFFF), &pos);

    for (0..grid_size) |j| {
        const pg: *const [8]i8 = @ptrCast(&grid[j]);
        var d2: i32 = 0;
        for (0..8) |k| {
            const diff: i32 = @as(i32, pg[k]) - @as(i32, pos[k]);
            d2 += diff * diff;
        }
        dists[j] = .{ .d2 = d2, .index = @intCast(j) };
    }
    std.sort.pdq(Dist, dists[0..grid_size], {}, Dist.lessThan);
}

/// Pass 1's body: how many entries fall within `nwant` distinct distances.
fn countNeighbours(pattern: usize, grid: [*]u64, grid_size: usize, dists: [*]Dist, nwant: i32) usize {
    sortByDistance(pattern, grid, grid_size, dists);

    var n: usize = 0;
    var d2 = dists[0].d2;
    var nhave: i32 = 1;
    for (0..grid_size) |j| {
        if (dists[j].d2 > d2) {
            if (nhave == nwant) break;
            d2 = dists[j].d2;
            nhave += 1;
        }
        n += 1;
    }
    return n;
}

/// Pass 3's body: the same walk, writing the indices out.
fn fillNeighbours(pattern: usize, grid: [*]u64, grid_size: usize, dists: [*]Dist, nwant: i32, out: [*]u16, local: *usize) usize {
    sortByDistance(pattern, grid, grid_size, dists);

    var n: usize = 0;
    var d2 = dists[0].d2;
    var nhave: i32 = 1;
    for (0..grid_size) |j| {
        if (dists[j].d2 > d2) {
            if (nhave == nwant) break;
            d2 = dists[j].d2;
            nhave += 1;
        }
        out[local.*] = @intCast(dists[j].index);
        local.* += 1;
        n += 1;
    }
    return n;
}

/// Ports `iq2xs_free_impl` (ggml-quants.c:3260 @c1d0e7a00).
pub export fn iq2xs_free_impl(t: c.enum_ggml_type) void {
    const gindex = iq2DataIndex(t);
    if (iq2_data[gindex].grid) |g| {
        std.c.free(g);
        iq2_data[gindex].grid = null;
        if (iq2_data[gindex].map) |m| std.c.free(m);
        iq2_data[gindex].map = null;
        if (iq2_data[gindex].neighbours) |n| std.c.free(n);
        iq2_data[gindex].neighbours = null;
    }
}

/// The built codebook for one type, for the quantizers.
pub fn iq2Grid(t: c.enum_ggml_type) [*]const u64 {
    return iq2_data[iq2DataIndex(t)].grid.?;
}

pub fn iq2Map(t: c.enum_ggml_type) [*]const i32 {
    return iq2_data[iq2DataIndex(t)].map.?;
}

pub fn iq2Neighbours(t: c.enum_ggml_type) [*]const u16 {
    return iq2_data[iq2DataIndex(t)].neighbours.?;
}

// -----------------------------------------------------------------------------
// The 3-bit codebooks
//
// Same three-pass construction as above, differing only in shape: four
// positions of three bits each rather than eight of two, so the codebook entry
// is a `u32` and the pattern space is 4096 rather than 43692.
//
// The C duplicates the whole routine for this. The two are kept separate here
// too: unifying them would mean a generic over the position count and the
// entry width, and the resulting code would be harder to check against either
// original than two straightforward copies are.

/// Ports `iq3_entry_t` (ggml-quants.c:3684 @c1d0e7a00).
const Iq3Entry = struct {
    grid: ?[*]u32 = null,
    map: ?[*]i32 = null,
    neighbours: ?[*]u16 = null,
};

/// Ports `iq3_data` (ggml-quants.c:3686 @c1d0e7a00). Two slots: grid size 256 and 512.
var iq3_data = [_]Iq3Entry{.{}} ** 2;

/// Ports `iq3_data_index` (ggml-quants.c:3691 @c1d0e7a00).
fn iq3DataIndex(grid_size: c_int) usize {
    impl.assert(grid_size == 256 or grid_size == 512, "grid_size == 256 || grid_size == 512");
    return if (grid_size == 256) 0 else 1;
}

/// The 4-position vector a 3-bit digit string decodes to. Odd values again,
/// here spanning 1..15.
inline fn unpackPositions3(bits: u16, pos: *[4]i8) void {
    for (0..4) |i| {
        const l: i32 = @intCast((bits >> @intCast(3 * i)) & 0x7);
        pos[i] = @intCast(2 * l + 1);
    }
}

/// Ports `iq3xs_init_impl` (ggml-quants.c:3703 @c1d0e7a00).
pub export fn iq3xs_init_impl(grid_size_in: c_int) void {
    const gindex = iq3DataIndex(grid_size_in);
    if (iq3_data[gindex].grid != null) return;

    const grid_size: usize = @intCast(grid_size_in);
    const kmap_size: usize = 4096;
    const nwant: i32 = if (grid_size == 256) 2 else 3;
    const kgrid: []const u16 = if (grid_size == 256) &grids.kgrid_256 else &grids.kgrid_512;

    const the_grid: [*]u32 = @ptrCast(@alignCast(std.c.malloc(grid_size * @sizeOf(u32)).?));
    for (0..grid_size) |k| {
        const pos: *[4]i8 = @ptrCast(&the_grid[k]);
        unpackPositions3(kgrid[k], pos);
    }
    iq3_data[gindex].grid = the_grid;

    const kmap: [*]i32 = @ptrCast(@alignCast(std.c.malloc(kmap_size * @sizeOf(i32)).?));
    iq3_data[gindex].map = kmap;
    for (0..kmap_size) |i| kmap[i] = -1;

    for (0..grid_size) |i| {
        const aux8: *const [4]u8 = @ptrCast(&the_grid[i]);
        var index: u16 = 0;
        for (0..4) |k| {
            const q: u16 = (aux8[k] - 1) / 2;
            index |= q << @intCast(3 * k);
        }
        kmap[index] = @intCast(i);
    }

    const n_per_i: [*]i32 = @ptrCast(@alignCast(std.c.malloc(kmap_size * @sizeOf(i32)).?));
    defer std.c.free(n_per_i);

    const dist_mem = std.c.malloc(grid_size * @sizeOf(Dist)).?;
    defer std.c.free(dist_mem);
    const dists: [*]Dist = @ptrCast(@alignCast(dist_mem));

    var num_neighbors: usize = 0;
    var num_not_in_map: usize = 0;
    for (0..kmap_size) |i| {
        if (kmap[i] >= 0) {
            n_per_i[i] = 0;
            continue;
        }
        num_not_in_map += 1;
        sortByDistance3(i, the_grid, grid_size, dists);
        const n = walkNeighbours(dists, grid_size, nwant, null, null);
        n_per_i[i] = @intCast(n);
        num_neighbors += n;
    }

    const kneighbors: [*]u16 = @ptrCast(@alignCast(std.c.malloc((num_neighbors + num_not_in_map) * @sizeOf(u16)).?));
    iq3_data[gindex].neighbours = kneighbors;

    const offsets: [*]i32 = @ptrCast(@alignCast(std.c.malloc(kmap_size * @sizeOf(i32)).?));
    defer std.c.free(offsets);

    var counter: i32 = 0;
    for (0..kmap_size) |i| {
        if (kmap[i] >= 0) {
            offsets[i] = -1;
            continue;
        }
        offsets[i] = counter;
        counter += 1 + n_per_i[i];
    }

    for (0..kmap_size) |i| {
        if (kmap[i] >= 0) continue;
        var local: usize = @intCast(offsets[i]);
        kmap[i] = -(offsets[i] + 1);
        const start = local;
        local += 1;

        sortByDistance3(i, the_grid, grid_size, dists);
        const n = walkNeighbours(dists, grid_size, nwant, kneighbors, &local);
        kneighbors[start] = @intCast(n);
    }
}

/// `sortByDistance` for the 4-position codebooks.
fn sortByDistance3(pattern: usize, grid: [*]u32, grid_size: usize, dists: [*]Dist) void {
    var pos: [4]i8 = undefined;
    unpackPositions3(@intCast(pattern & 0xFFFF), &pos);

    for (0..grid_size) |j| {
        const pg: *const [4]i8 = @ptrCast(&grid[j]);
        var d2: i32 = 0;
        for (0..4) |k| {
            const diff: i32 = @as(i32, pg[k]) - @as(i32, pos[k]);
            d2 += diff * diff;
        }
        dists[j] = .{ .d2 = d2, .index = @intCast(j) };
    }
    std.sort.pdq(Dist, dists[0..grid_size], {}, Dist.lessThan);
}

/// The distance-tier walk both passes perform, counting when `out` is null and
/// writing when it is not.
///
/// Factored out here because the 3-bit version needs it twice and the two
/// copies in the C differ only in whether they store.
fn walkNeighbours(dists: [*]Dist, grid_size: usize, nwant: i32, out: ?[*]u16, local: ?*usize) usize {
    var n: usize = 0;
    var d2 = dists[0].d2;
    var nhave: i32 = 1;
    for (0..grid_size) |j| {
        if (dists[j].d2 > d2) {
            if (nhave == nwant) break;
            d2 = dists[j].d2;
            nhave += 1;
        }
        if (out) |o| {
            o[local.?.*] = @intCast(dists[j].index);
            local.?.* += 1;
        }
        n += 1;
    }
    return n;
}

/// Ports `iq3xs_free_impl` (ggml-quants.c:3904 @c1d0e7a00).
pub export fn iq3xs_free_impl(grid_size: c_int) void {
    const gindex = iq3DataIndex(grid_size);
    if (iq3_data[gindex].grid) |g| {
        std.c.free(g);
        iq3_data[gindex].grid = null;
        if (iq3_data[gindex].map) |m| std.c.free(m);
        iq3_data[gindex].map = null;
        if (iq3_data[gindex].neighbours) |n| std.c.free(n);
        iq3_data[gindex].neighbours = null;
    }
}

/// The built 3-bit codebook, for the quantizers.
pub fn iq3Grid(grid_size: c_int) [*]const u32 {
    return iq3_data[iq3DataIndex(grid_size)].grid.?;
}

pub fn iq3Map(grid_size: c_int) [*]const i32 {
    return iq3_data[iq3DataIndex(grid_size)].map.?;
}

pub fn iq3Neighbours(grid_size: c_int) [*]const u16 {
    return iq3_data[iq3DataIndex(grid_size)].neighbours.?;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the codebooks build, are idempotent, and free cleanly" {
    // Building twice must not leak or rebuild: `ggml_quantize_init` is called
    // on every quantize, so the early-out is load-bearing rather than tidy.
    iq2xs_init_impl(c.GGML_TYPE_IQ2_XXS);
    const first = iq2Grid(c.GGML_TYPE_IQ2_XXS);
    iq2xs_init_impl(c.GGML_TYPE_IQ2_XXS);
    try std.testing.expectEqual(first, iq2Grid(c.GGML_TYPE_IQ2_XXS));

    // Every codebook entry decodes to odd values in 1..7, and the map finds
    // each one exactly.
    const grid = iq2Grid(c.GGML_TYPE_IQ2_XXS);
    const map = iq2Map(c.GGML_TYPE_IQ2_XXS);
    for (0..256) |i| {
        const pos: *const [8]i8 = @ptrCast(&grid[i]);
        var index: u16 = 0;
        for (pos, 0..) |v, k| {
            try std.testing.expect(v == 1 or v == 3 or v == 5 or v == 7);
            index |= @as(u16, @intCast(@divExact(v - 1, 2))) << @intCast(2 * k);
        }
        try std.testing.expectEqual(@as(i32, @intCast(i)), map[index]);
    }

    iq2xs_free_impl(c.GGML_TYPE_IQ2_XXS);
    // Freeing twice is safe, which `ggml_quantize_free` relies on: it frees
    // every codebook unconditionally, built or not.
    iq2xs_free_impl(c.GGML_TYPE_IQ2_XXS);
}

test "a pattern with no exact entry gets a non-empty neighbour list" {
    iq2xs_init_impl(c.GGML_TYPE_IQ2_XXS);
    defer iq2xs_free_impl(c.GGML_TYPE_IQ2_XXS);

    const map = iq2Map(c.GGML_TYPE_IQ2_XXS);
    const neighbours = iq2Neighbours(c.GGML_TYPE_IQ2_XXS);

    var checked: usize = 0;
    for (0..43692) |i| {
        if (map[i] >= 0) continue;
        // Negative entries encode -(offset + 1).
        const offset: usize = @intCast(-map[i] - 1);
        const n = neighbours[offset];
        try std.testing.expect(n > 0);
        checked += 1;
        if (checked > 64) break;
    }
    try std.testing.expect(checked > 0);
}

test "the 3-bit codebooks build for both grid sizes" {
    for ([_]c_int{ 256, 512 }) |size| {
        iq3xs_init_impl(size);
        const grid = iq3Grid(size);
        const map = iq3Map(size);
        for (0..@intCast(size)) |i| {
            const pos: *const [4]i8 = @ptrCast(&grid[i]);
            var index: u16 = 0;
            for (pos, 0..) |v, k| {
                // Three bits, so 1..15 rather than 1..7.
                try std.testing.expect(v >= 1 and v <= 15 and @rem(v, 2) == 1);
                index |= @as(u16, @intCast(@divExact(v - 1, 2))) << @intCast(3 * k);
            }
            try std.testing.expectEqual(@as(i32, @intCast(i)), map[index]);
        }
        iq3xs_free_impl(size);
    }
}
