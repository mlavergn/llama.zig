//! Documentation root.
//!
//! Zig's autodoc cannot root a module at `module.zig`, so this sibling exists
//! solely for the `docs` build step. It re-exports the barrel; the rest of the
//! module comes along with it.

pub const mod = @import("module.zig");
