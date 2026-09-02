//! Shared fixtures for the quantization tests.
//!
//! # Provenance
//!
//! **Not a port.** This reproduces, in Zig, the input generation and hashing
//! that `harness/quants_golden.c` used to capture `golden.zig`. The two must
//! agree bit for bit or every comparison is meaningless, so both are written
//! against the same fixed algorithms rather than a library's, and the patterns
//! below mirror `pattern_t` in that file one for one.

const std = @import("std");
const golden = @import("golden.zig");

pub const n_per_row = golden.n_per_row;
pub const n_rows = golden.n_rows;
pub const n_elem = n_per_row * n_rows;

/// FNV-1a, matching `fnv` in `harness/quants_golden.c`.
pub fn fnv(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        h ^= b;
        h = h *% 0x100000001b3;
    }
    return h;
}

/// The generator's LCG, matching `next_float` in `harness/quants_golden.c`.
///
/// A fixed LCG rather than a std PRNG: the values must not depend on the
/// implementation on either side, or the goldens stop being reproducible.
const Rng = struct {
    state: u32,

    fn next(self: *Rng) f32 {
        self.state = self.state *% 1664525 +% 1013904223;
        const v = @as(f32, @floatFromInt(self.state >> 8)) / @as(f32, 1 << 24);
        return v * 4.0 - 2.0;
    }
};

/// Mirrors `pattern_t` in `harness/quants_golden.c`.
pub const Pattern = enum {
    random,
    zeros,
    constant,
    outlier,
    alternating,

    pub fn name(self: Pattern) []const u8 {
        return @tagName(self);
    }
};

pub const all_patterns = [_]Pattern{ .random, .zeros, .constant, .outlier, .alternating };

/// Fills `src` with one pattern, matching `fill_src`.
pub fn fillSrc(pattern: Pattern, src: []f32) void {
    switch (pattern) {
        .zeros => @memset(src, 0.0),
        .constant => @memset(src, 0.5),
        .outlier => {
            @memset(src, 0.001);
            // One per row, off the block boundary so it is not always the
            // first element of a block.
            for (0..n_rows) |r| src[r * n_per_row + 7] = 100.0;
        },
        .alternating => {
            for (src, 0..) |*v, i| v.* = if (i & 1 != 0) -1.25 else 1.25;
        },
        .random => {
            var rng: Rng = .{ .state = 12345 };
            for (src) |*v| v.* = rng.next();
        },
    }
}

/// Fills the importance matrix, matching `fill_imatrix`.
///
/// Its own stream, seeded separately, so it is identical for every pattern --
/// sharing one RNG with `fillSrc` would make it depend on how many values the
/// pattern happened to draw.
pub fn fillImatrix(imatrix: []f32) void {
    var rng: Rng = .{ .state = 999 };
    for (imatrix) |*v| v.* = (rng.next() + 2.0) * 0.5 + 0.01;
}

/// Looks up one type's golden record for one pattern.
pub fn find(name: []const u8, pattern: Pattern) golden.Golden {
    for (golden.all) |g| {
        if (std.mem.eql(u8, g.name, name) and std.mem.eql(u8, g.pattern, pattern.name())) return g;
    }
    @panic("no golden record for that type and pattern");
}
