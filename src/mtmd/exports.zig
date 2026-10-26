//! Exports `mtmd.h` and `mtmd-helper.h`: the step that makes this library
//! libmtmd.
//!
//! **Not a port.** Every `mtmd_*` function in `mtmd.zig`, `helper.zig` and
//! `helper_gen.zig` is a `pub` Zig function with the C signature and no
//! export, so the in-process harnesses can call it beside upstream's
//! libmtmd, which defines the same names. Importing this file exports all
//! of them by name, as `src/llama/exports.zig` does for `llama.h`. Only
//! `module.zig`, the library root, imports it; `harness_root.zig` must not.
//!
//! The reference archive also exports stb_image's, xxHash's and the
//! SHA-256's C entry points, which upstream compiles into `mtmd-helper.cpp`
//! and `vendor/hash`. No caller of libmtmd links against them, and the port
//! keeps its decoders internal (`vendor/`, `std.crypto`).

const std = @import("std");

comptime {
    @setEvalBranchQuota(100_000);
    for (.{ @import("mtmd.zig"), @import("helper.zig"), @import("helper_gen.zig") }) |file| {
        for (@typeInfo(file).@"struct".decl_names) |name| {
            if (!std.mem.startsWith(u8, name, "mtmd_")) continue;
            const f = @field(file, name);
            if (@typeInfo(@TypeOf(f)) != .@"fn") continue;
            @export(&f, .{ .name = name });
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
