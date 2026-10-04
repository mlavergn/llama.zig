//! The 6-bit scale/min decoder the NEON K-quant repack kernels share.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`).
//!
//! # The same shuffle as `repack/q4_k.zig`, cut differently
//!
//! `q4_K` and `q5_K` pack eight 6-bit scales and eight 6-bit mins into
//! twelve bytes. The generic kernels unpack all eight sub-blocks into a
//! 32-word `utmp` and index it; this one decodes **one** sub-block at a
//! time into a vector of mins and eight scale bytes, because the NEON
//! kernels consume them that way.
//!
//! The bit arithmetic is identical and has the same hazard: `scales_u32[1]`
//! reads `sm[0]` after `scales_u32[0]` has already masked a copy of it, so
//! the two lines are order-dependent in the C only because they read the
//! *original* `sm[0]`. Writing to `sm` in between would break it.

const std = @import("std");
const impl = @import("../../../impl.zig");
const neon = @import("../../quants/arm/neon.zig");

const i16x8 = neon.i16x8;

const kmask1: u32 = 0x3f3f3f3f;
const kmask2: u32 = 0x0f0f0f0f;
const kmask3: u32 = 0x03030303;
const scales_size: usize = 12;

/// Ports `decode_q_Kx8_6bit_scales` (arch/arm/repack.cpp:29 @c1d0e7a00).
///
/// Parameters:
/// - `scales_in`: the twelve packed bytes for one sub-block.
/// - `out_mins`: receives the eight mins, widened to `i16`.
/// - `out_scales`: receives the eight scales as bytes.
pub inline fn decodeQKx86BitScales(scales_in: [*]const u8, out_mins: *i16x8, out_scales: *[8]i8) void {
    var sm: [3]u32 = undefined;
    @memcpy(@as([*]u8, @ptrCast(&sm))[0..scales_size], scales_in[0..scales_size]);

    const mins_0_3 = sm[1] & kmask1;
    const mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);
    const mins_u32: @Vector(2, u32) = .{ mins_0_3, mins_4_7 };

    const mins_u8: @Vector(8, u8) = @bitCast(mins_u32);
    out_mins.* = @bitCast(neon.movl_u8(mins_u8));

    var scales_u32: [2]u32 = undefined;
    scales_u32[0] = sm[0] & kmask1;
    scales_u32[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
    @memcpy(@as([*]u8, @ptrCast(out_scales))[0..8], @as([*]const u8, @ptrCast(&scales_u32))[0..8]);
}

comptime {
    _ = std;
    _ = impl;
}
