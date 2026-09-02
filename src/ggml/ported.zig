//! Test root for the ported ggml, holding only what has been ported so far.
//!
//! # Provenance
//!
//! **Not a port.** Nothing here corresponds to a file in the reference
//! checkout; it is scaffolding this project owns. The one exported symbol is a
//! stub, marked as such where it is defined.
//!
//! # Why this exists
//!
//! Porting a translation unit is all-or-nothing: a partially ported `ggml.c`
//! cannot be linked next to the original, because the linker would pull
//! `ggml.c.o` in for the symbols still missing and then find duplicates for the
//! ones already done.
//!
//! So the ported files cannot be tested inside the real library until the whole
//! translation unit is finished. This root exists to test them before then. It
//! imports only ported code, and `zig build test-port` links it against the two
//! C translation units the ported code genuinely depends on -- `ggml-quants.c`
//! for the reference conversion routines the traits table points at, and
//! `ggml-threading.c` for the critical section -- neither of which overlaps
//! with `ggml.c`.
//!
//! Once `ggml.c` is fully ported, this becomes redundant: the files move into
//! `module.zig` and are tested as part of the library.
//!
//! # The backend is real here, not stubbed
//!
//! This root used to define its own `ggml_backend_tensor_memset` because
//! `ggml-backend.cpp` could not be linked: it needs the graph symbols, and the
//! graph section of `ggml.c` was not ported. It is now, so the real file is
//! linked and the stub is gone. `graph.zig` and `ops.zig` reach the genuine
//! backend, which is the only way the paths through it are worth testing.

const std = @import("std");

comptime {
    _ = @import("alloc.zig");
    _ = @import("impl.zig");
    _ = @import("types.zig");
    _ = @import("context.zig");
    _ = @import("runtime.zig");
    _ = @import("ops.zig");
    _ = @import("graph.zig");
    _ = @import("quantize.zig");
    _ = @import("quants/module.zig");
    _ = @import("cpu/module.zig");
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("alloc.zig");
    _ = @import("impl.zig");
    _ = @import("types.zig");
    _ = @import("context.zig");
    _ = @import("runtime.zig");
    _ = @import("ops.zig");
    _ = @import("graph.zig");
    _ = @import("quantize.zig");
    _ = @import("quants/module.zig");
    _ = @import("cpu/module.zig");
}
