//! What the tools take from `ggml.h`'s macros, which `translate-c` cannot
//! give them: `GGML_ASSERT` and `GGML_PAD`, and the abort the port takes
//! where upstream's allocation would throw `std::bad_alloc`.
//!
//! # Provenance
//!
//! Ports `GGML_PAD` and `GGML_ASSERT` from `llama.cpp/ggml/include/ggml.h`
//! (v1.0.0, `b1b770d90`). Each tool binds them to its own upstream file
//! name, the `__FILE__` the C's message carries.

/// Ports `GGML_ASSERT` (ggml.h:115 @b1b770d90): `ggml_abort` with the
/// failed condition, in every build mode.
///
/// Parameters:
/// - `file`, `line`: where upstream's assertion is, for its message.
/// - `expr`: the condition's text.
pub fn ggmlAssert(ok: bool, comptime file: [*:0]const u8, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = llama.impl.llama_c.ggml_abort(file, line, "GGML_ASSERT(%s) failed", expr);
}

/// An allocation failed where upstream's would throw `std::bad_alloc` and
/// terminate: `ggml_abort` instead, naming the file.
///
/// Parameters:
/// - `file`: the upstream file the allocation is in.
pub fn oom(comptime file: [*:0]const u8) noreturn {
    _ = llama.impl.llama_c.ggml_abort(file, 0, "out of memory");
    unreachable;
}

// -----------------------------------------------------------------------------
// Unit Tests

const std = @import("std");
const llama = @import("llama");

test {
    std.testing.refAllDecls(@This());
}
