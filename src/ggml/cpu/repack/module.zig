//! Barrel for the `repack.cpp` / `arch/arm/repack.cpp` port.
//!
//! **Not a port.** It re-exports the pieces so siblings reach each other
//! through one import, per the project's barrel convention.
//!
//! # The cluster moves together
//!
//! `repack.cpp` derives from the two base classes `traits.cpp` declares,
//! and `ggml-cpu.cpp` registers the buffer type this directory builds, so
//! `traits.cpp`, both repack files and `ggml-cpu.cpp` are swapped in one
//! step or not at all. Two definitions of a symbol in one static archive
//! is not a link error -- the linker picks one silently -- so a partial
//! swap reads as success.

const impl = @import("../../impl.zig");
const c = impl.c;

/// Whether the C++ dispatch layer of `repack.cpp` has been ported.
///
/// **This exists because the symbol count lies for this file.** `repack.cpp`
/// is one of the four translation units `CLAUDE.md` names as exceptions to
/// the unmangled-exports contract. Its 36 unmangled symbols are the
/// gemv/gemm and quantize kernels; the buffer type,
/// `extra_buffer_type::{supports_op, get_tensor_traits}` and the
/// `tensor_traits` instantiations that decide when to call them all have
/// C++ linkage and never appear in `nm -gU | grep -v _Z`.
///
/// Measured: with the dispatch stubbed, `port-coverage` read
/// `36 / 36 symbols (100%)` and `node-diff` failed on the first `MUL_MAT`
/// because the repack buffer type was never registered and the graph was
/// built differently. `scripts/cluster-check` asserts this flag at
/// comptime so that cannot read as complete again.
///
/// `dispatch.zig` is what makes it true.
pub const dispatch_implemented = true;

pub const arm = @import("arm/module.zig");
pub const blocks = @import("blocks.zig");
pub const convert = @import("convert.zig");
pub const dispatch = @import("dispatch.zig");
pub const epilogue = @import("epilogue.zig");
pub const nibble4 = @import("nibble4.zig");
pub const q2_k = @import("q2_k.zig");
pub const q4_0 = @import("q4_0.zig");
pub const q4_k = @import("q4_k.zig");
pub const q5_k = @import("q5_k.zig");
pub const q6_k = @import("q6_k.zig");
pub const q8_0 = @import("q8_0.zig");
pub const quantize = @import("quantize.zig");

/// Ports `ggml_backend_cpu_repack_buffer_type` (ggml-cpu/repack.cpp:4821
/// @c1d0e7a00). Re-exported from `dispatch.zig` so `cpu_backend.zig` keeps
/// reaching this directory through the barrel.
pub const ggml_backend_cpu_repack_buffer_type = dispatch.ggml_backend_cpu_repack_buffer_type;

comptime {
    _ = arm;
    _ = blocks;
    _ = convert;
    _ = dispatch;
    _ = epilogue;
    _ = nibble4;
    _ = q2_k;
    _ = q4_0;
    _ = q4_k;
    _ = q5_k;
    _ = q6_k;
    _ = q8_0;
    _ = quantize;
}

// -----------------------------------------------------------------------------
// Unit Tests

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}

// **This test is the reason `cluster-check` cannot report a false 100%.**
//
// `repack.cpp` is one of the four translation units whose contract is *not*
// its unmangled exports. Its 36 unmangled symbols are the gemv/gemm and
// quantize kernels; everything that decides when to call them — the buffer
// type, `extra_buffer_type::{supports_op, get_tensor_traits}`, and the
// `tensor_traits` template instantiations — has C++ linkage and is
// invisible to `scripts/port-coverage`.
//
// Measured the hard way: with those stubbed, `port-coverage` read
// `36 / 36 symbols (100%)` and `node-diff` failed on the first `MUL_MAT`
// because the graph was built differently. A symbol count cannot see that;
// this can.
test "the repack buffer type is registered, not stubbed" {
    const buft = ggml_backend_cpu_repack_buffer_type();
    try std.testing.expect(buft != null);

    // A registered buffer type carries the `extra_buffer_type` context the
    // dispatch reaches through, and a name.
    try std.testing.expect(buft.?.*.context != null);
    try std.testing.expect(buft.?.*.iface.get_name != null);
    try std.testing.expect(buft.?.*.iface.alloc_buffer != null);
}
