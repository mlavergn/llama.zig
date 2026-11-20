//! What the tools take from `ggml.h`'s macros, which `translate-c` cannot
//! give them: `GGML_ASSERT` and `GGML_PAD`, and the abort the port takes
//! where upstream's allocation would throw `std::bad_alloc`.
//!
//! # Provenance
//!
//! Ports `GGML_PAD` and `GGML_ASSERT` from `llama.cpp/ggml/include/ggml.h`
//! (v1.0.0, `b1b770d90`). Each tool binds them to its own upstream file
//! name, the `__FILE__` the C's message carries.

const c = @import("tools_h");

/// Ports `GGML_ASSERT` (ggml.h:115 @b1b770d90): `ggml_abort` with the
/// failed condition, in every build mode.
///
/// Parameters:
/// - `file`, `line`: where upstream's assertion is, for its message.
/// - `expr`: the condition's text.
pub fn ggmlAssert(ok: bool, comptime file: [*:0]const u8, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort(file, line, "GGML_ASSERT(%s) failed", expr);
}

/// An allocation failed where upstream's would throw `std::bad_alloc` and
/// terminate: `ggml_abort` instead, naming the file.
///
/// Parameters:
/// - `file`: the upstream file the allocation is in.
pub fn oom(comptime file: [*:0]const u8) noreturn {
    _ = c.ggml_abort(file, 0, "out of memory");
    unreachable;
}

/// Ports `GGML_PAD` (ggml.h:94 @b1b770d90): `x` rounded up to a multiple of
/// `n`, a power of two.
///
/// Return: the padded size.
pub fn pad(x: usize, n: usize) usize {
    return (x + n - 1) & ~(n - 1);
}

// -----------------------------------------------------------------------------
// Unit Tests

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}

test "pad rounds up to the alignment" {
    try std.testing.expectEqual(@as(usize, 0), pad(0, 32));
    try std.testing.expectEqual(@as(usize, 32), pad(1, 32));
    try std.testing.expectEqual(@as(usize, 32), pad(32, 32));
    try std.testing.expectEqual(@as(usize, 64), pad(33, 32));
}
