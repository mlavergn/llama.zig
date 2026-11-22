//! libmtmd's C API: every file that exports an `mtmd.h` or `mtmd-helper.h`
//! symbol, forced into the compilation.
//!
//! # Provenance
//!
//! **Not a port.** Linking this is what defines the `mtmd_*` symbols in a
//! binary. `module.zig` forces it when libmtmd itself is the compilation
//! root, and under test; an executable that compiles the `mtmd` module in and
//! calls the C API forces it too (`_ = mtmd.lib`). A harness never does: it
//! links upstream's libmtmd beside the port, and two definitions of a symbol
//! pick one silently.

const std = @import("std");

comptime {
    _ = @import("exports.zig");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
