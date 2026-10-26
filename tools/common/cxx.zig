//! The C++ standard library calls the tools' parsing leans on, as libc++
//! answers them.
//!
//! # Provenance
//!
//! **Not a port of a file.** `std::stoi` is libc++'s `string.cpp`:
//! `strtol` in base 10, then `invalid_argument` if nothing converted and
//! `out_of_range` if `errno` says so or the value leaves `int` -- with the
//! messages libc++ throws, which upstream prints through `e.what()`.

const std = @import("std");

extern fn strtol(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_long;

/// What `std::stoi` returns or throws.
pub const Stoi = union(enum) {
    ok: c_int,
    /// `std::invalid_argument`, `what()` "stoi: no conversion".
    no_conversion,
    /// `std::out_of_range`, `what()` "stoi: out of range".
    out_of_range,

    /// `e.what()` for the two throws.
    pub fn what(self: Stoi) [*:0]const u8 {
        return switch (self) {
            .ok => "",
            .no_conversion => "stoi: no conversion",
            .out_of_range => "stoi: out of range",
        };
    }
};

/// `std::stoi(s)`.
pub fn stoi(s: [:0]const u8) Stoi {
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtol(s.ptr, &end, 10);
    if (end == s.ptr) return .no_conversion;
    if (std.c._errno().* == @backingInt(std.c.E.RANGE) or v < std.math.minInt(c_int) or v > std.math.maxInt(c_int)) return .out_of_range;
    return .{ .ok = @intCast(v) };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "stoi converts a prefix and throws as libc++ does" {
    try std.testing.expectEqual(Stoi{ .ok = 12 }, stoi(" 12abc"));
    try std.testing.expectEqual(Stoi.no_conversion, stoi("abc"));
    try std.testing.expectEqual(Stoi.out_of_range, stoi("99999999999"));
}
