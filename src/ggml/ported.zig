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
//! imports only ported code, and `zig build test-port` links it against the
//! C++ translation units the ported code genuinely depends on -- the backend,
//! and the CPU op kernels the ported dispatch calls into. None of them
//! overlaps with `ggml.c`. Nothing C is linked at all any more, and since
//! `ggml-threading.cpp` was ported the critical section comes from here too.
//!
//! That moment has arrived: `ggml.c` is fully ported, and so is everything
//! else under `ggml/src/`. What is left of this root is the *name* --
//! `zig build test-port` and `scripts/port-coverage` both point at it --
//! so it now simply re-exports `module.zig`.
//!
//! **It used to repeat `module.zig`'s import list, and that was a hole.**
//! `port-coverage` compiles *this* root and reads its symbols, while the
//! library is built from `module.zig`. Adding a newly ported file to one
//! and not the other made the coverage report read
//! `6 / 6 symbols (100%), complete and swapped into the build` for a
//! library that contained none of them. The linker caught it only because
//! something happened to reference those symbols; a new export that
//! nothing calls yet would have passed. One list cannot diverge from
//! itself.
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
    _ = @import("module.zig");
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("module.zig");
}
