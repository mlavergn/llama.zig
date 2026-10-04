//! Diff every interleaved `repack` kernel against the reference C++, on bits.
//!
//! **Not a port.** This is a gate, driven by `scripts/repack-diff`.
//!
//! # Why it exists
//!
//! The 36 kernels of `repack.cpp` and `arch/arm/repack.cpp` had no oracle.
//! They are unreachable from `test-backend-ops`, which never allocates a
//! `CPU_REPACK` buffer; from `make ops-diff`, which builds its tensors in
//! a plain CPU buffer; and from `make port`, which runs on Metal. The only
//! gate that saw them at all was `make node-diff`, and node-diff reports
//! the *first* divergent node and stops — one kernel per run, with a model
//! load and two full decodes between each.
//!
//! Measured: swapping the cluster in, `node-diff` named `MUL_MAT ffn_out-0`
//! and nothing else. This harness named `ggml_gemm_q6_K_8x4_q8_K`, and the
//! fault was `half * 1024` where the C has `half * 512`.
//!
//! # How the reference is reached
//!
//! `scripts/repack-diff` compiles the three reference translation units
//! with every one of their 66 unmangled exports renamed to `ref_*` on the
//! command line, so both implementations link into one process and can be
//! called back to back on identical bytes. That is the same `-D` rename
//! `NOTES.md` records as unusable for *swapping* a translation unit — it
//! renames internal callers too — which is exactly the property wanted
//! here: the renamed object is self-consistent and calls its own kernels.
//!
//! # The inputs are random bytes, and that is enough
//!
//! These kernels are integer dot products with a float epilogue; every
//! input bit pattern is a valid quantized block. Only the `d`/`dmin`/`e`
//! scale fields are fixed up, to keep the epilogue off NaN and Inf, where
//! a bit comparison would stop meaning anything.

const std = @import("std");

/// The public ggml headers, for the converter half: it drives the real
/// `CPU_REPACK` buffer type rather than calling a kernel, so it needs
/// `ggml_tensor`, the context API and the backend-buffer API.
const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-backend.h");
    @cInclude("ggml-cpu.h");
});

/// The reference's `CPU_REPACK` buffer type, by its **mangled** name.
///
/// `repack.h` declares `ggml_backend_cpu_repack_buffer_type` outside any
/// `extern "C"`, so its only exported name is `_Z35…v`. It is therefore
/// *not* one of `repack.cpp`'s 36 unmangled exports and `-D` cannot rename
/// it — which is the same fact, seen from a third direction, that
/// `CLAUDE.md` records as "the symbol count is not the contract for this
/// file".
const refRepackBuft = @extern(
    *const fn () callconv(.c) c.ggml_backend_buffer_type_t,
    .{ .name = "_Z35ggml_backend_cpu_repack_buffer_typev" },
);

/// Our own, reached the way `llama.cpp` reaches it: through the CPU
/// registry's `ggml_backend_dev_get_extra_bufts` proc address. That is a
/// pure C ABI, and it puts `cpu_backend.zig`'s extra-buffer list under the
/// same check.
fn portedRepackBuft() c.ggml_backend_buffer_type_t {
    const reg = c.ggml_backend_cpu_reg();
    const proc = c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts") orelse return null;
    const get: *const fn (c.ggml_backend_dev_t) callconv(.c) [*c]c.ggml_backend_buffer_type_t = @ptrCast(@alignCast(proc));
    const list = get(c.ggml_backend_reg_dev_get(reg, 0));
    if (list == null) return null;
    return list[0];
}

/// Every interleaved kernel takes this shape.
const Kernel = fn (c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) callconv(.c) void;
const QuantizeMat = fn ([*]const f32, *anyopaque, i64) callconv(.c) void;

const ours = struct {
    extern fn ggml_gemv_q4_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q4_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q4_0_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q2_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q4_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q4_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q5_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q5_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q6_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q6_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_iq4_nl_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_iq4_nl_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_mxfp4_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_mxfp4_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q8_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemv_q8_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;

    extern fn ggml_gemm_q4_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q4_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q4_0_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q2_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q4_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q4_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q5_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q5_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q6_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q6_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_iq4_nl_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_iq4_nl_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_mxfp4_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_mxfp4_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q8_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ggml_gemm_q8_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;

    extern fn ggml_quantize_mat_q8_0_4x4([*]const f32, *anyopaque, i64) void;
    extern fn ggml_quantize_mat_q8_0_4x8([*]const f32, *anyopaque, i64) void;
    extern fn ggml_quantize_mat_q8_K_4x4([*]const f32, *anyopaque, i64) void;
    extern fn ggml_quantize_mat_q8_K_4x8([*]const f32, *anyopaque, i64) void;
};

const reference = struct {
    extern fn ref_ggml_gemv_q4_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q4_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q4_0_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q2_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q4_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q4_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q5_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q5_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q6_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q6_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_iq4_nl_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_iq4_nl_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_mxfp4_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_mxfp4_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q8_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemv_q8_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;

    extern fn ref_ggml_gemm_q4_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q4_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q4_0_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q2_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q4_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q4_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q5_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q5_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q6_K_8x4_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q6_K_8x8_q8_K(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_iq4_nl_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_iq4_nl_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_mxfp4_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_mxfp4_8x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q8_0_4x4_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;
    extern fn ref_ggml_gemm_q8_0_4x8_q8_0(c_int, [*]f32, usize, *const anyopaque, *const anyopaque, c_int, c_int) void;

    extern fn ref_ggml_quantize_mat_q8_0_4x4([*]const f32, *anyopaque, i64) void;
    extern fn ref_ggml_quantize_mat_q8_0_4x8([*]const f32, *anyopaque, i64) void;
    extern fn ref_ggml_quantize_mat_q8_K_4x4([*]const f32, *anyopaque, i64) void;
    extern fn ref_ggml_quantize_mat_q8_K_4x8([*]const f32, *anyopaque, i64) void;
};

/// The block layouts, written out from `ggml-cpu/repack.h` and
/// `ggml-common.h` rather than imported from `src/`.
///
/// **That is deliberate.** This harness exists to check the port against
/// the reference; taking the layouts from the port would mean a mistyped
/// array bound produced two wrong buffers that agreed with each other.
/// `blocks.zig` carries the C's own `static_assert`s for the same reason.
const QK4_0 = 32;
const QK8_0 = 32;
const QK_K = 256;
const QK4_NL = 32;
const QK_MXFP4 = 32;
const half = u16;

/// Mirrors `block<K, N>` (ggml-cpu/repack.h:23 @c1d0e7a00).
fn Block(comptime K: comptime_int, comptime N: comptime_int) type {
    return extern struct {
        d: [N]half,
        qs: [(32 * N * K) / 8]i8,
    };
}

const Q4_0x4 = Block(4, 4);
const Q4_0x8 = Block(4, 8);
const Q8_0x4 = Block(8, 4);

/// Mirrors `block_q4_Kx8` (ggml-cpu/repack.h:43 @c1d0e7a00).
const Q4_Kx8 = extern struct { d: [8]half, dmin: [8]half, scales: [96]u8, qs: [1024]u8 };
/// Mirrors `block_q2_Kx8` (ggml-cpu/repack.h:59 @c1d0e7a00).
const Q2_Kx8 = extern struct { d: [8]half, dmin: [8]half, scales: [128]u8, qs: [512]u8 };
/// Mirrors `block_q5_Kx8` (ggml-cpu/repack.h:75 @c1d0e7a00).
const Q5_Kx8 = extern struct { d: [8]half, dmin: [8]half, scales: [96]u8, qh: [QK_K * 8 / 8]u8, qs: [QK_K * 8 / 2]u8 };
/// Mirrors `block_q6_Kx8` (ggml-cpu/repack.h:86 @c1d0e7a00).
const Q6_Kx8 = extern struct { d: [8]half, scales: [QK_K / 16 * 8]i8, ql: [QK_K / 2 * 8]u8, qh: [QK_K / 4 * 8]u8 };
/// Mirrors `block_q8_Kx4` (ggml-cpu/repack.h:96 @c1d0e7a00).
const Q8Kx4 = extern struct { d: [4]f32, qs: [QK_K * 4]i8, bsums: [QK_K / 4]i16 };
/// Mirrors `block_iq4_nlx4` (ggml-cpu/repack.h:104 @c1d0e7a00).
const IQ4_NLx4 = extern struct { d: [4]half, qs: [QK4_NL * 2]u8 };
/// Mirrors `block_iq4_nlx8` (ggml-cpu/repack.h:111 @c1d0e7a00).
const IQ4_NLx8 = extern struct { d: [8]half, qs: [QK4_NL * 4]u8 };
/// Mirrors `block_mxfp4x4` (ggml-cpu/repack.h:124 @c1d0e7a00).
const MXFP4x4 = extern struct { e: [4]u8, qs: [QK_MXFP4 * 2]u8 };
/// Mirrors `block_mxfp4x8` (ggml-cpu/repack.h:130 @c1d0e7a00).
const MXFP4x8 = extern struct { e: [8]u8, qs: [QK_MXFP4 * 4]u8 };

/// Mirrors `block_q8_0` (ggml-common.h:239 @c1d0e7a00): the Q8_0
/// activation row a gemv reads.
const Q8_0 = extern struct { d: half, qs: [QK8_0]i8 };
/// Mirrors `block_q8_K` (ggml-common.h:371 @c1d0e7a00): the Q8_K one.
const Q8K = extern struct { d: f32, qs: [QK_K]i8, bsums: [QK_K / 16]i16 };

comptime {
    // The C's own static_asserts, which are what catch a mistyped bound.
    const h = @sizeOf(half);
    std.debug.assert(@sizeOf(Q4_0x4) == 4 * h + QK8_0 * 2);
    std.debug.assert(@sizeOf(Q4_0x8) == 8 * h + QK8_0 * 4);
    std.debug.assert(@sizeOf(Q8_0x4) == 4 * h + QK8_0 * 4);
    std.debug.assert(@sizeOf(Q4_Kx8) == h * 16 + 12 * 8 + QK_K * 4);
    std.debug.assert(@sizeOf(Q2_Kx8) == h * 16 + QK_K / 2 + QK_K * 2);
    std.debug.assert(@sizeOf(Q5_Kx8) == h * 16 + 12 * 8 + QK_K * 5);
    std.debug.assert(@sizeOf(Q6_Kx8) == h * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8);
    std.debug.assert(@sizeOf(Q8Kx4) == 4 * @sizeOf(f32) + QK_K * 4 + QK_K / 4 * @sizeOf(i16));
    std.debug.assert(@sizeOf(IQ4_NLx4) == 4 * h + QK4_NL * 2);
    std.debug.assert(@sizeOf(IQ4_NLx8) == 8 * h + QK4_NL * 4);
    std.debug.assert(@sizeOf(MXFP4x4) == 4 + QK_MXFP4 * 2);
    std.debug.assert(@sizeOf(MXFP4x8) == 8 + QK_MXFP4 * 4);
}

var prng = std.Random.DefaultPrng.init(0x9e3779b97f4a7c15);

/// Random bytes, with the float scale fields pulled back to a finite range.
///
/// Everything else is a quant: every bit pattern is legal and none of them
/// can reach a NaN, so leaving them random is what makes this a wide test
/// rather than a narrow one. The scales are the exception -- an f16 of
/// 0x7c01 is a NaN, and a comparison on bits stops meaning anything once
/// one appears.
fn fillBlocks(comptime T: type, items: []T) void {
    prng.random().bytes(std.mem.sliceAsBytes(items));
    for (items) |*b| {
        inline for (.{ "d", "dmin" }) |name| {
            if (@hasField(T, name)) {
                const f = &@field(b, name);
                switch (@typeInfo(@TypeOf(f.*))) {
                    .array => |a| switch (a.child) {
                        // ggml_half: keep the sign and mantissa, force the
                        // exponent to 2^-3.
                        u16 => for (f) |*h| {
                            h.* = (h.* & 0x83ff) | 0x3000;
                        },
                        f32 => for (f) |*v| {
                            v.* = prng.random().float(f32) * 0.25;
                        },
                        else => @compileError("unexpected scale element"),
                    },
                    .int => f.* = (f.* & 0x83ff) | 0x3000,
                    .float => f.* = prng.random().float(f32) * 0.25,
                    else => @compileError("unexpected scale field"),
                }
            }
        }
        // mxfp4's shared exponent is an E8M0 byte: 0xff is its NaN.
        if (@hasField(T, "e")) {
            const f = &@field(b, "e");
            switch (@typeInfo(@TypeOf(f.*))) {
                .array => for (f) |*v| {
                    v.* = 0x70 +% (v.* & 0x0f);
                },
                else => f.* = 0x70 +% (f.* & 0x0f),
            }
        }
    }
}

var failures: usize = 0;
var checks: usize = 0;

fn report(name: []const u8, a: []const f32, b: []const f32) void {
    checks += 1;
    var bad: usize = 0;
    var first: usize = 0;
    for (a, b, 0..) |x, y, i| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            if (bad == 0) first = i;
            bad += 1;
        }
    }
    if (bad == 0) {
        std.debug.print("  {s:<30} {d:>4} lanes identical\n", .{ name, a.len });
        return;
    }
    failures += 1;
    std.debug.print("  {s:<30} DIFFERS: {d} of {d} lanes; first at {d}, ours {d} ref {d}\n", .{ name, bad, a.len, first, a[first], b[first] });
}

/// One kernel pair: identical random weights and activations into both,
/// compared on bits.
///
/// Parameters:
/// - `W`: the interleaved weight block type.
/// - `Y`: the activation block a gemv reads (a plain quantized row).
/// - `Y4`: the activation block a gemm reads (four rows interleaved).
/// - `qk`: elements per weight block.
/// - `cols`: `NB_COLS`, weight rows per interleaved block.
fn runCase(
    comptime name: []const u8,
    comptime W: type,
    comptime Y: type,
    comptime Y4: type,
    comptime qk: i64,
    comptime cols: i64,
    comptime gemv_ours: *const Kernel,
    comptime gemv_ref: *const Kernel,
    comptime gemm_ours: *const Kernel,
    comptime gemm_ref: *const Kernel,
    alloc: std.mem.Allocator,
) !void {
    const nb: i64 = 2;
    const n: c_int = @intCast(qk * nb);
    const nc: c_int = 16;

    const w = try alloc.alloc(W, @intCast(@divExact(nc, cols) * nb));
    defer alloc.free(w);
    fillBlocks(W, w);

    // gemv: a single activation row.
    {
        const y = try alloc.alloc(Y, @intCast(nb));
        defer alloc.free(y);
        fillBlocks(Y, y);
        const a = try alloc.alloc(f32, @intCast(nc));
        defer alloc.free(a);
        const b = try alloc.alloc(f32, @intCast(nc));
        defer alloc.free(b);
        @memset(a, 0);
        @memset(b, 0);
        gemv_ours(n, a.ptr, @intCast(nc), w.ptr, y.ptr, 1, nc);
        gemv_ref(n, b.ptr, @intCast(nc), w.ptr, y.ptr, 1, nc);
        report("gemv_" ++ name, a, b);
    }

    // gemm: four activation rows.
    {
        const y = try alloc.alloc(Y4, @intCast(nb));
        defer alloc.free(y);
        fillBlocks(Y4, y);
        const a = try alloc.alloc(f32, @intCast(4 * nc));
        defer alloc.free(a);
        const b = try alloc.alloc(f32, @intCast(4 * nc));
        defer alloc.free(b);
        @memset(a, 0);
        @memset(b, 0);
        gemm_ours(n, a.ptr, @intCast(nc), w.ptr, y.ptr, 4, nc);
        gemm_ref(n, b.ptr, @intCast(nc), w.ptr, y.ptr, 4, nc);
        report("gemm_" ++ name, a, b);
    }
}

/// The four `ggml_quantize_mat_*` kernels: four rows of f32 in, one
/// interleaved activation block out, compared on bytes.
fn runQuantizeMat(
    comptime name: []const u8,
    comptime Out: type,
    comptime qk: i64,
    comptime f_ours: *const QuantizeMat,
    comptime f_ref: *const QuantizeMat,
    alloc: std.mem.Allocator,
) !void {
    const nb: i64 = 2;
    const n: i64 = qk * nb;

    const x = try alloc.alloc(f32, @intCast(4 * n));
    defer alloc.free(x);
    // Amplitude 1 and 30, the two the ops-diff harness runs: a cutoff that
    // only one of them reaches is a fault neither alone would show.
    for (x, 0..) |*v, i| v.* = (prng.random().float(f32) * 2.0 - 1.0) * (if (i % 2 == 0) @as(f32, 1.0) else 30.0);

    const a = try alloc.alloc(Out, @intCast(nb));
    defer alloc.free(a);
    const b = try alloc.alloc(Out, @intCast(nb));
    defer alloc.free(b);
    @memset(std.mem.sliceAsBytes(a), 0);
    @memset(std.mem.sliceAsBytes(b), 0);
    f_ours(x.ptr, a.ptr, n);
    f_ref(x.ptr, b.ptr, n);

    checks += 1;
    const ab = std.mem.sliceAsBytes(a);
    const bb = std.mem.sliceAsBytes(b);
    var bad: usize = 0;
    var first: usize = 0;
    for (ab, bb, 0..) |p, q, i| if (p != q) {
        if (bad == 0) first = i;
        bad += 1;
    };
    if (bad == 0) {
        std.debug.print("  {s:<30} {d:>4} bytes identical\n", .{ name, ab.len });
    } else {
        failures += 1;
        std.debug.print("  {s:<30} DIFFERS: {d} of {d} bytes; first at {d}\n", .{ name, bad, ab.len, first });
    }
}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("=== repack kernels: ported vs reference, on bits ===\n", .{});

    try runCase("q4_0_4x4_q8_0", Q4_0x4, Q8_0, Q8_0x4, 32, 4, ours.ggml_gemv_q4_0_4x4_q8_0, reference.ref_ggml_gemv_q4_0_4x4_q8_0, ours.ggml_gemm_q4_0_4x4_q8_0, reference.ref_ggml_gemm_q4_0_4x4_q8_0, alloc);
    try runCase("q4_0_4x8_q8_0", Q4_0x4, Q8_0, Q8_0x4, 32, 4, ours.ggml_gemv_q4_0_4x8_q8_0, reference.ref_ggml_gemv_q4_0_4x8_q8_0, ours.ggml_gemm_q4_0_4x8_q8_0, reference.ref_ggml_gemm_q4_0_4x8_q8_0, alloc);
    try runCase("q4_0_8x8_q8_0", Q4_0x8, Q8_0, Q8_0x4, 32, 8, ours.ggml_gemv_q4_0_8x8_q8_0, reference.ref_ggml_gemv_q4_0_8x8_q8_0, ours.ggml_gemm_q4_0_8x8_q8_0, reference.ref_ggml_gemm_q4_0_8x8_q8_0, alloc);
    try runCase("q8_0_4x4_q8_0", Q8_0x4, Q8_0, Q8_0x4, 32, 4, ours.ggml_gemv_q8_0_4x4_q8_0, reference.ref_ggml_gemv_q8_0_4x4_q8_0, ours.ggml_gemm_q8_0_4x4_q8_0, reference.ref_ggml_gemm_q8_0_4x4_q8_0, alloc);
    try runCase("q8_0_4x8_q8_0", Q8_0x4, Q8_0, Q8_0x4, 32, 4, ours.ggml_gemv_q8_0_4x8_q8_0, reference.ref_ggml_gemv_q8_0_4x8_q8_0, ours.ggml_gemm_q8_0_4x8_q8_0, reference.ref_ggml_gemm_q8_0_4x8_q8_0, alloc);
    try runCase("iq4_nl_4x4_q8_0", IQ4_NLx4, Q8_0, Q8_0x4, 32, 4, ours.ggml_gemv_iq4_nl_4x4_q8_0, reference.ref_ggml_gemv_iq4_nl_4x4_q8_0, ours.ggml_gemm_iq4_nl_4x4_q8_0, reference.ref_ggml_gemm_iq4_nl_4x4_q8_0, alloc);
    try runCase("iq4_nl_8x8_q8_0", IQ4_NLx8, Q8_0, Q8_0x4, 32, 8, ours.ggml_gemv_iq4_nl_8x8_q8_0, reference.ref_ggml_gemv_iq4_nl_8x8_q8_0, ours.ggml_gemm_iq4_nl_8x8_q8_0, reference.ref_ggml_gemm_iq4_nl_8x8_q8_0, alloc);
    try runCase("mxfp4_4x4_q8_0", MXFP4x4, Q8_0, Q8_0x4, 32, 4, ours.ggml_gemv_mxfp4_4x4_q8_0, reference.ref_ggml_gemv_mxfp4_4x4_q8_0, ours.ggml_gemm_mxfp4_4x4_q8_0, reference.ref_ggml_gemm_mxfp4_4x4_q8_0, alloc);
    try runCase("mxfp4_8x8_q8_0", MXFP4x8, Q8_0, Q8_0x4, 32, 8, ours.ggml_gemv_mxfp4_8x8_q8_0, reference.ref_ggml_gemv_mxfp4_8x8_q8_0, ours.ggml_gemm_mxfp4_8x8_q8_0, reference.ref_ggml_gemm_mxfp4_8x8_q8_0, alloc);

    try runCase("q2_K_8x8_q8_K", Q2_Kx8, Q8K, Q8Kx4, 256, 8, ours.ggml_gemv_q2_K_8x8_q8_K, reference.ref_ggml_gemv_q2_K_8x8_q8_K, ours.ggml_gemm_q2_K_8x8_q8_K, reference.ref_ggml_gemm_q2_K_8x8_q8_K, alloc);
    try runCase("q4_K_8x4_q8_K", Q4_Kx8, Q8K, Q8Kx4, 256, 8, ours.ggml_gemv_q4_K_8x4_q8_K, reference.ref_ggml_gemv_q4_K_8x4_q8_K, ours.ggml_gemm_q4_K_8x4_q8_K, reference.ref_ggml_gemm_q4_K_8x4_q8_K, alloc);
    try runCase("q4_K_8x8_q8_K", Q4_Kx8, Q8K, Q8Kx4, 256, 8, ours.ggml_gemv_q4_K_8x8_q8_K, reference.ref_ggml_gemv_q4_K_8x8_q8_K, ours.ggml_gemm_q4_K_8x8_q8_K, reference.ref_ggml_gemm_q4_K_8x8_q8_K, alloc);
    try runCase("q5_K_8x4_q8_K", Q5_Kx8, Q8K, Q8Kx4, 256, 8, ours.ggml_gemv_q5_K_8x4_q8_K, reference.ref_ggml_gemv_q5_K_8x4_q8_K, ours.ggml_gemm_q5_K_8x4_q8_K, reference.ref_ggml_gemm_q5_K_8x4_q8_K, alloc);
    try runCase("q5_K_8x8_q8_K", Q5_Kx8, Q8K, Q8Kx4, 256, 8, ours.ggml_gemv_q5_K_8x8_q8_K, reference.ref_ggml_gemv_q5_K_8x8_q8_K, ours.ggml_gemm_q5_K_8x8_q8_K, reference.ref_ggml_gemm_q5_K_8x8_q8_K, alloc);
    try runCase("q6_K_8x4_q8_K", Q6_Kx8, Q8K, Q8Kx4, 256, 8, ours.ggml_gemv_q6_K_8x4_q8_K, reference.ref_ggml_gemv_q6_K_8x4_q8_K, ours.ggml_gemm_q6_K_8x4_q8_K, reference.ref_ggml_gemm_q6_K_8x4_q8_K, alloc);
    try runCase("q6_K_8x8_q8_K", Q6_Kx8, Q8K, Q8Kx4, 256, 8, ours.ggml_gemv_q6_K_8x8_q8_K, reference.ref_ggml_gemv_q6_K_8x8_q8_K, ours.ggml_gemm_q6_K_8x8_q8_K, reference.ref_ggml_gemm_q6_K_8x8_q8_K, alloc);

    try runQuantizeMat("quantize_mat_q8_0_4x4", Q8_0x4, 32, ours.ggml_quantize_mat_q8_0_4x4, reference.ref_ggml_quantize_mat_q8_0_4x4, alloc);
    try runQuantizeMat("quantize_mat_q8_0_4x8", Q8_0x4, 32, ours.ggml_quantize_mat_q8_0_4x8, reference.ref_ggml_quantize_mat_q8_0_4x8, alloc);
    try runQuantizeMat("quantize_mat_q8_K_4x4", Q8Kx4, 256, ours.ggml_quantize_mat_q8_K_4x4, reference.ref_ggml_quantize_mat_q8_K_4x4, alloc);
    try runQuantizeMat("quantize_mat_q8_K_4x8", Q8Kx4, 256, ours.ggml_quantize_mat_q8_K_4x8, reference.ref_ggml_quantize_mat_q8_K_4x8, alloc);

    // Both `tensor_traits::repack` implementations log at debug level; the
    // gate compares bytes, so the log is noise here.
    c.ggml_log_set(silence, null);

    std.debug.print("\n=== repack converters, through the real buffer type ===\n", .{});
    for ([_]ConverterCase{
        .{ .name = "q4_0", .ty = c.GGML_TYPE_Q4_0 },
        .{ .name = "q4_K", .ty = c.GGML_TYPE_Q4_K },
        .{ .name = "q5_K", .ty = c.GGML_TYPE_Q5_K },
        .{ .name = "q6_K", .ty = c.GGML_TYPE_Q6_K },
        .{ .name = "q8_0", .ty = c.GGML_TYPE_Q8_0 },
        .{ .name = "iq4_nl", .ty = c.GGML_TYPE_IQ4_NL },
        .{ .name = "mxfp4", .ty = c.GGML_TYPE_MXFP4 },
        .{ .name = "q2_K", .ty = c.GGML_TYPE_Q2_K },
    }) |case| try runConverter(case, alloc);

    std.debug.print("\n", .{});
    if (failures == 0) {
        std.debug.print("PASS: {d} repack checks identical, ported vs reference\n", .{checks});
        return;
    }
    std.debug.print("FAIL: {d} of {d} repack checks differ\n", .{ failures, checks });
    std.process.exit(1);
}

fn silence(level: c.ggml_log_level, text: [*c]const u8, user: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user;
}

const ConverterCase = struct { name: []const u8, ty: c_uint };

/// One block converter, driven the way the model loader drives it.
///
/// The kernel half above calls exported functions directly. The eleven
/// `repack_*_to_*_bl` converters are `static` in the C and have no
/// exported name, so the only way to reach the reference's is through its
/// `CPU_REPACK` buffer type — which is the right way anyway: it also puts
/// `ggml_repack_get_optimal_repack_type`, `init_tensor` and `set_tensor`
/// under the same comparison.
///
/// A type the dispatch does not select here repacks on neither side, and
/// *that* is the thing compared: a port that selected a different
/// instantiation would show up as one side repacking and the other not.
///
/// Parameters:
/// - `case`: the stored block type to round-trip.
/// - `alloc`: for the source bytes.
fn runConverter(case: ConverterCase, alloc: std.mem.Allocator) !void {
    // ne0 = 256 is a whole number of blocks for every type here and is
    // divisible by 8, which nine of the eleven converters require; ne1 = 8
    // satisfies both the `% 4` and `% 8` row guards.
    const ne0: i64 = 256;
    const ne1: i64 = 8;

    var out: [2][]u8 = undefined;
    var repacked: [2]bool = undefined;
    var src: []u8 = &.{};
    defer if (src.len != 0) alloc.free(src);

    for ([_]c.ggml_backend_buffer_type_t{
        portedRepackBuft(),
        refRepackBuft(),
    }, 0..) |buft, side| {
        var params = std.mem.zeroes(c.struct_ggml_init_params);
        params.mem_size = 16 * 1024 * 1024;
        params.no_alloc = true;
        const ctx = c.ggml_init(params).?;
        defer c.ggml_free(ctx);

        const t = c.ggml_new_tensor_2d(ctx, case.ty, ne0, ne1).?;
        const nbytes = c.ggml_nbytes(t);
        if (src.len == 0) {
            src = try alloc.alloc(u8, nbytes);
            prng.random().bytes(src);
        }

        const buffer = c.ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft).?;
        defer c.ggml_backend_buffer_free(buffer);

        // `init_tensor` has run by now and hung the traits off `extra`;
        // null means this type and shape have no interleaved kernel here.
        repacked[side] = t.*.extra != null;
        if (!repacked[side]) continue;

        // The write path is the repack: `set_tensor` interleaves in place.
        c.ggml_backend_tensor_set(t, src.ptr, 0, nbytes);

        const base: [*]const u8 = @ptrCast(c.ggml_backend_buffer_get_base(buffer).?);
        out[side] = try alloc.dupe(u8, base[0..nbytes]);
    }
    defer for (0..2) |i| if (repacked[i]) alloc.free(out[i]);

    checks += 1;
    if (repacked[0] != repacked[1]) {
        failures += 1;
        std.debug.print("  {s:<30} DIFFERS: ported repacked={}, reference repacked={}\n", .{ case.name, repacked[0], repacked[1] });
        return;
    }
    if (!repacked[0]) {
        std.debug.print("  {s:<30} not selected on this target, both sides\n", .{case.name});
        return;
    }
    var bad: usize = 0;
    var first: usize = 0;
    for (out[0], out[1], 0..) |a, b, i| if (a != b) {
        if (bad == 0) first = i;
        bad += 1;
    };
    if (bad == 0) {
        std.debug.print("  {s:<30} {d:>6} bytes identical\n", .{ case.name, out[0].len });
    } else {
        failures += 1;
        std.debug.print("  {s:<30} DIFFERS: {d} of {d} bytes; first at {d}\n", .{ case.name, bad, out[0].len, first });
    }
}
