//! The root the in-process mtmd harnesses compile.
//!
//! **Not a port.** The harnesses link the port beside upstream's libmtmd
//! (`libmtmd-ref.a`) and call both under the same names. They cannot root
//! at `module.zig`: it imports `exports.zig`, and two definitions of a
//! symbol in one link pick one silently. This root re-exports the port's
//! files and nothing that exports a symbol; compile it with its `llama`
//! import pointing at `src/llama/harness_root.zig` for the same reason.
//! Never import `exports.zig` from here.

const std = @import("std");

pub const names = @import("names.zig");
pub const impl = @import("impl.zig");
pub const model = @import("model.zig");
pub const graph = @import("graph.zig");
pub const clip = @import("clip.zig");
pub const models = @import("models/module.zig");
pub const image = @import("image.zig");
pub const audio = @import("audio.zig");
pub const vendor = @import("vendor/module.zig");
pub const mtmd = @import("mtmd.zig");
pub const helper = @import("helper.zig");
pub const helper_gen = @import("helper_gen.zig");

test {
    std.testing.refAllDecls(@This());
}
