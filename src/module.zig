//! Library barrel for llamazig.
//!
//! Re-exports every public type in `src/` so sibling files and consumers
//! import through this module rather than reaching for each other directly.
//!
//! The greeting placeholders this repository started from are gone, so little
//! is left here yet. The ported ggml is a separate module -- see the note
//! below -- and this barrel grows as the port takes ownership of code that is
//! ours rather than a translation unit standing in for a C one.

const std = @import("std");
const builtin = @import("builtin");

pub const is_debug = builtin.mode == .Debug;

// `src/ggml/` is deliberately *not* re-exported here. It is built by
// `build/llamacpp.zig` as its own module, with the include paths and build
// options ggml needs; importing it through this barrel would instantiate a
// second copy without them. Its barrel is `src/ggml/module.zig`, and the
// library it produces is `libggml.a`, not `libllamazig.a`.

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
