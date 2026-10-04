//! The NEON `q4_0` × `q8_0` interleaved gemm: hand-written aarch64.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` at v0.3.0
//! (`c1d0e7a00`).
//!
//! # Why this one is assembly
//!
//! `ggml_gemm_q4_0_4x4_q8_0` is **445 lines of hand-written aarch64** in
//! the C, not intrinsics — the only live `__asm__` block in the file. The
//! other three asm blocks there sit behind `__ARM_FEATURE_MATMUL_INT8` or
//! SVE, which this build does not select, so those functions fall through
//! to their `_generic` form and are ported that way in `fallback.zig`.
//!
//! The instruction sequence is reproduced exactly. Rewriting it as
//! intrinsics would be a different kernel with its own scheduling and its
//! own last bit, and there is no reason to believe a rewrite would match.
//!
//! # Two mechanical differences from the C, and nothing else
//!
//! - **`%x[name]` becomes `%[name]`.** GCC's `x` operand modifier forces
//!   the 64-bit spelling of a register holding a 32-bit value. Zig has no
//!   operand modifiers, so `nr`, `nb` and `nc` are widened to `usize` here
//!   and the plain `%[name]` prints the same `xN`. Verified against the
//!   disassembly: `mov x10, x2`, not `mov x10, w2`.
//! - **Clobbers are a typed struct.** Zig 0.16 spells them as fields;
//!   `cc` is `nzcv`, and the `v0`-`v31` of the C are `z0`-`z31`, which
//!   alias the same registers.
//!
//! The `//` annotations inside the template are the C's own label
//! comments. They are line comments to the aarch64 assembler too, so they
//! travel with the code rather than being dropped.

const std = @import("std");
const impl = @import("../../../impl.zig");

const c = impl.c;

/// Ports `ggml_gemm_q4_0_4x4_q8_0` (arch/arm/repack.cpp:1826 @c1d0e7a00).
///
/// Parameters:
/// - `n`: the reduction length; `nb = n / 32` blocks.
/// - `s`, `bs`: the result and its row stride in elements.
/// - `vx`, `vy`: the interleaved weights and the quantized activations.
/// - `nr`, `nc`: rows and columns of the product.
pub export fn ggml_gemm_q4_0_4x4_q8_0(
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) callconv(.c) void {
    const nb: usize = @intCast(@divTrunc(n, 32));

    const b_ptr: [*]const u8 = @ptrCast(vx);
    var a_ptr: [*]const u8 = @ptrCast(vy);
    var res_ptr: [*]f32 = s;
    const res_stride: usize = bs * @sizeOf(f32);
    const nr_u: usize = @intCast(nr);
    const nc_u: usize = @intCast(nc);

    asm volatile (
        \\mov x10, %[nr]
        \\mov x9, #0x88
        \\cmp x10, #0x10
        \\mul x9, %[nb], x9
        \\blt 4f
        \\1:  // Row loop
        \\add x28, %[b_ptr], #0x8
        \\mov x27, %[nc]
        \\add x26, %[res_ptr], %[res_stride], LSL #4
        \\2:  // Column loop
        \\add x25, %[a_ptr], #0x8
        \\movi v15.16b, #0x0
        \\movi v19.16b, #0x0
        \\mov x24, %[nb]
        \\add x23, x25, x9
        \\movi v18.16b, #0x0
        \\movi v14.16b, #0x0
        \\add x22, x23, x9
        \\movi v11.16b, #0x0
        \\movi v13.16b, #0x0
        \\add x21, x22, x9
        \\movi v23.16b, #0x0
        \\movi v16.16b, #0x0
        \\movi v25.16b, #0x0
        \\movi v7.16b, #0x0
        \\movi v0.16b, #0x0
        \\movi v4.16b, #0x0
        \\movi v5.16b, #0x0
        \\movi v21.16b, #0x0
        \\movi v8.16b, #0x0
        \\movi v1.16b, #0x0
        \\3:  // Block loop
        \\ldr q3, [x28, #0x0]
        \\ldr q31, [x25, #0x0]
        \\movi v28.16b, #0x4
        \\movi v10.4s, #0x0
        \\ldr q22, [x28, #0x10]
        \\ldr q6, [x25, #0x10]
        \\movi v29.4s, #0x0
        \\movi v9.4s, #0x0
        \\ldr q27, [x28, #0x20]
        \\ldr q30, [x28, #0x30]
        \\movi v20.4s, #0x0
        \\movi v24.16b, #0xf0
        \\ldr d2, [x25, #-0x8]
        \\ldr d26, [x23, #-0x8]
        \\sshl v12.16b, v3.16b, v28.16b
        \\sub x20, x28, #0x8
        \\ldr d17, [x20, #0x0]
        \\and v3.16b, v3.16b, v24.16b
        \\subs x24, x24, #0x1
        \\add x28, x28, #0x48
        \\.inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]
        \\.inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]
        \\.inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]
        \\.inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]
        \\sshl v31.16b, v22.16b, v28.16b
        \\and v22.16b, v22.16b, v24.16b
        \\fcvtl v17.4s, v17.4h
        \\fcvtl v2.4s, v2.4h
        \\fcvtl v26.4s, v26.4h
        \\.inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]
        \\.inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]
        \\.inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]
        \\.inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]
        \\sshl v6.16b, v27.16b, v28.16b
        \\sshl v28.16b, v30.16b, v28.16b
        \\and v27.16b, v27.16b, v24.16b
        \\and v30.16b, v30.16b, v24.16b
        \\ldr q24, [x25, #0x20]
        \\.inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]
        \\.inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]
        \\.inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]
        \\.inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]
        \\ldr q24, [x25, #0x30]
        \\.inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]
        \\.inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]
        \\.inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]
        \\.inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]
        \\ldr q24, [x25, #0x40]
        \\.inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]
        \\.inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]
        \\.inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]
        \\.inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]
        \\ldr q24, [x25, #0x50]
        \\.inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]
        \\.inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]
        \\.inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]
        \\.inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]
        \\ldr q24, [x25, #0x60]
        \\.inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]
        \\.inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]
        \\.inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]
        \\.inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]
        \\ldr q24, [x25, #0x70]
        \\add x25, x25, #0x88
        \\.inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]
        \\.inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]
        \\.inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]
        \\.inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]
        \\fmul v24.4s, v17.4s, v2.s[0]
        \\scvtf v10.4s, v10.4s, #0x4
        \\scvtf v29.4s, v29.4s, #0x4
        \\scvtf v9.4s, v9.4s, #0x4
        \\scvtf v20.4s, v20.4s, #0x4
        \\fmla v15.4s, v10.4s, v24.4s
        \\ldr q24, [x23, #0x0]
        \\fmul v10.4s, v17.4s, v2.s[1]
        \\fmla v19.4s, v29.4s, v10.4s
        \\ldr q10, [x23, #0x10]
        \\fmul v29.4s, v17.4s, v2.s[2]
        \\fmul v2.4s, v17.4s, v2.s[3]
        \\fmla v18.4s, v9.4s, v29.4s
        \\movi v9.4s, #0x0
        \\movi v29.4s, #0x0
        \\.inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]
        \\.inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]
        \\fmla v14.4s, v20.4s, v2.4s
        \\movi v20.4s, #0x0
        \\movi v2.4s, #0x0
        \\.inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]
        \\.inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]
        \\ldr q24, [x23, #0x20]
        \\.inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]
        \\.inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]
        \\.inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]
        \\.inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]
        \\ldr q10, [x23, #0x30]
        \\.inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]
        \\.inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]
        \\.inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]
        \\.inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]
        \\ldr q24, [x23, #0x40]
        \\.inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]
        \\.inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]
        \\.inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]
        \\.inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]
        \\ldr q10, [x23, #0x50]
        \\.inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]
        \\.inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]
        \\.inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]
        \\.inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]
        \\ldr q24, [x23, #0x60]
        \\.inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]
        \\.inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]
        \\.inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]
        \\.inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]
        \\ldr q10, [x23, #0x70]
        \\add x23, x23, #0x88
        \\.inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]
        \\.inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]
        \\.inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]
        \\.inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]
        \\ldr q24, [x22, #0x0]
        \\.inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]
        \\.inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]
        \\.inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]
        \\.inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]
        \\fmul v10.4s, v17.4s, v26.s[0]
        \\scvtf v9.4s, v9.4s, #0x4
        \\scvtf v29.4s, v29.4s, #0x4
        \\scvtf v20.4s, v20.4s, #0x4
        \\scvtf v2.4s, v2.4s, #0x4
        \\fmla v11.4s, v9.4s, v10.4s
        \\ldr q9, [x22, #0x10]
        \\fmul v10.4s, v17.4s, v26.s[1]
        \\fmla v13.4s, v29.4s, v10.4s
        \\ldr d29, [x22, #-0x8]
        \\fmul v10.4s, v17.4s, v26.s[2]
        \\fmul v26.4s, v17.4s, v26.s[3]
        \\fcvtl v29.4s, v29.4h
        \\fmla v23.4s, v20.4s, v10.4s
        \\movi v20.4s, #0x0
        \\movi v10.4s, #0x0
        \\fmla v16.4s, v2.4s, v26.4s
        \\movi v26.4s, #0x0
        \\movi v2.4s, #0x0
        \\.inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]
        \\.inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]
        \\.inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]
        \\.inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]
        \\ldr q24, [x22, #0x20]
        \\.inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]
        \\.inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]
        \\.inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]
        \\.inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]
        \\ldr q9, [x22, #0x30]
        \\.inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]
        \\.inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]
        \\.inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]
        \\.inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]
        \\ldr q24, [x22, #0x40]
        \\.inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]
        \\.inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]
        \\.inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]
        \\.inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]
        \\ldr q9, [x22, #0x50]
        \\.inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]
        \\.inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]
        \\.inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]
        \\.inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]
        \\ldr q24, [x22, #0x60]
        \\.inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]
        \\.inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]
        \\.inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]
        \\.inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]
        \\ldr q9, [x22, #0x70]
        \\add x22, x22, #0x88
        \\.inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]
        \\.inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]
        \\.inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]
        \\.inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]
        \\ldr q24, [x21, #0x0]
        \\.inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]
        \\.inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]
        \\.inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]
        \\.inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]
        \\fmul v9.4s, v17.4s, v29.s[0]
        \\scvtf v20.4s, v20.4s, #0x4
        \\scvtf v10.4s, v10.4s, #0x4
        \\scvtf v26.4s, v26.4s, #0x4
        \\scvtf v2.4s, v2.4s, #0x4
        \\fmla v25.4s, v20.4s, v9.4s
        \\ldr q9, [x21, #0x10]
        \\fmul v20.4s, v17.4s, v29.s[1]
        \\fmla v7.4s, v10.4s, v20.4s
        \\ldr d20, [x21, #-0x8]
        \\fmul v10.4s, v17.4s, v29.s[2]
        \\fmul v29.4s, v17.4s, v29.s[3]
        \\fcvtl v20.4s, v20.4h
        \\fmla v0.4s, v26.4s, v10.4s
        \\movi v26.4s, #0x0
        \\movi v10.4s, #0x0
        \\fmla v4.4s, v2.4s, v29.4s
        \\movi v2.4s, #0x0
        \\movi v29.4s, #0x0
        \\.inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]
        \\.inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]
        \\.inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]
        \\.inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]
        \\ldr q12, [x21, #0x20]
        \\fmul v24.4s, v17.4s, v20.s[0]
        \\.inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]
        \\.inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]
        \\.inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]
        \\.inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]
        \\ldr q9, [x21, #0x30]
        \\fmul v31.4s, v17.4s, v20.s[1]
        \\.inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]
        \\.inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]
        \\.inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]
        \\.inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]
        \\ldr q12, [x21, #0x40]
        \\fmul v6.4s, v17.4s, v20.s[2]
        \\fmul v20.4s, v17.4s, v20.s[3]
        \\.inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]
        \\.inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]
        \\.inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]
        \\.inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]
        \\ldr q9, [x21, #0x50]
        \\.inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]
        \\.inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]
        \\.inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]
        \\.inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]
        \\ldr q12, [x21, #0x60]
        \\.inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]
        \\.inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]
        \\.inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]
        \\.inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]
        \\ldr q17, [x21, #0x70]
        \\add x21, x21, #0x88
        \\.inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]
        \\.inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]
        \\.inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]
        \\.inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]
        \\.inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]
        \\.inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]
        \\.inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]
        \\.inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]
        \\scvtf v26.4s, v26.4s, #0x4
        \\scvtf v10.4s, v10.4s, #0x4
        \\fmla v5.4s, v26.4s, v24.4s
        \\scvtf v2.4s, v2.4s, #0x4
        \\scvtf v29.4s, v29.4s, #0x4
        \\fmla v21.4s, v10.4s, v31.4s
        \\fmla v8.4s, v2.4s, v6.4s
        \\fmla v1.4s, v29.4s, v20.4s
        \\bgt 3b
        \\mov x20, %[res_ptr]
        \\subs x27, x27, #0x4
        \\add %[res_ptr], %[res_ptr], #0x10
        \\str q15, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q19, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q18, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q14, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q11, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q13, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q23, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q16, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q25, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q7, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q0, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q4, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q5, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q21, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q8, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\str q1, [x20, #0x0]
        \\bne 2b
        \\mov x20, #0x4
        \\sub x10, x10, #0x10
        \\cmp x10, #0x10
        \\mov %[res_ptr], x26
        \\madd %[a_ptr], x20, x9, %[a_ptr]
        \\bge 1b
        \\4:  // Row loop skip
        \\cbz x10, 9f
        \\5:  // Row tail: Row loop
        \\add x24, %[b_ptr], #0x8
        \\mov x23, %[nc]
        \\add x22, %[res_ptr], %[res_stride], LSL #2
        \\6:  // Row tail: Column loop
        \\movi v15.16b, #0x0
        \\movi v19.16b, #0x0
        \\add x25, %[a_ptr], #0x8
        \\mov x21, %[nb]
        \\movi v18.16b, #0x0
        \\movi v14.16b, #0x0
        \\7:  // Row tail: Block loop
        \\ldr q7, [x24, #0x0]
        \\ldr q5, [x25, #0x0]
        \\movi v9.16b, #0x4
        \\movi v4.4s, #0x0
        \\ldr q3, [x24, #0x10]
        \\ldr q2, [x25, #0x10]
        \\movi v1.4s, #0x0
        \\movi v0.4s, #0x0
        \\ldr q13, [x24, #0x20]
        \\ldr q31, [x25, #0x20]
        \\movi v30.4s, #0x0
        \\movi v29.16b, #0xf0
        \\ldr q28, [x24, #0x30]
        \\ldr q27, [x25, #0x30]
        \\sshl v20.16b, v7.16b, v9.16b
        \\sub x20, x24, #0x8
        \\ldr q26, [x25, #0x40]
        \\ldr q25, [x25, #0x50]
        \\sshl v17.16b, v3.16b, v9.16b
        \\and v7.16b, v7.16b, v29.16b
        \\ldr q24, [x25, #0x60]
        \\ldr q16, [x25, #0x70]
        \\sshl v22.16b, v13.16b, v9.16b
        \\and v3.16b, v3.16b, v29.16b
        \\ldr d21, [x20, #0x0]
        \\ldr d12, [x25, #-0x8]
        \\.inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]
        \\.inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]
        \\.inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]
        \\.inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]
        \\sshl v9.16b, v28.16b, v9.16b
        \\subs x21, x21, #0x1
        \\and v13.16b, v13.16b, v29.16b
        \\and v28.16b, v28.16b, v29.16b
        \\add x25, x25, #0x88
        \\add x24, x24, #0x48
        \\fcvtl v21.4s, v21.4h
        \\fcvtl v12.4s, v12.4h
        \\.inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]
        \\.inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]
        \\.inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]
        \\.inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]
        \\fmul v11.4s, v21.4s, v12.s[0]
        \\fmul v23.4s, v21.4s, v12.s[1]
        \\fmul v17.4s, v21.4s, v12.s[2]
        \\.inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]
        \\fmul v6.4s, v21.4s, v12.s[3]
        \\.inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]
        \\.inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]
        \\.inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]
        \\.inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]
        \\.inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]
        \\.inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]
        \\.inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]
        \\.inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]
        \\.inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]
        \\.inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]
        \\.inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]
        \\.inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]
        \\.inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]
        \\.inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]
        \\.inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]
        \\.inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]
        \\.inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]
        \\.inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]
        \\.inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]
        \\.inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]
        \\.inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]
        \\.inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]
        \\.inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]
        \\scvtf v4.4s, v4.4s, #0x4
        \\scvtf v1.4s, v1.4s, #0x4
        \\scvtf v0.4s, v0.4s, #0x4
        \\fmla v15.4s, v4.4s, v11.4s
        \\scvtf v30.4s, v30.4s, #0x4
        \\fmla v19.4s, v1.4s, v23.4s
        \\fmla v18.4s, v0.4s, v17.4s
        \\fmla v14.4s, v30.4s, v6.4s
        \\bgt 7b
        \\mov x20, %[res_ptr]
        \\cmp x10, #0x1
        \\str q15, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\ble 8f
        \\cmp x10, #0x2
        \\str q19, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\ble 8f
        \\cmp x10, #0x3
        \\str q18, [x20, #0x0]
        \\add x20, x20, %[res_stride]
        \\ble 8f
        \\str q14, [x20, #0x0]
        \\8:  // Row tail: Accumulator store skip
        \\subs x23, x23, #0x4
        \\add %[res_ptr], %[res_ptr], #0x10
        \\bne 6b
        \\subs x10, x10, #0x4
        \\add %[a_ptr], %[a_ptr], x9
        \\mov %[res_ptr], x22
        \\bgt 5b
        \\9:  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr),
          [res_ptr] "+&r" (res_ptr),
        : [b_ptr] "r" (b_ptr),
          [nr] "r" (nr_u),
          [nb] "r" (nb),
          [res_stride] "r" (res_stride),
          [nc] "r" (nc_u),
        : .{
          .nzcv = true,
          .memory = true,
          .z0 = true,
          .z1 = true,
          .z2 = true,
          .z3 = true,
          .z4 = true,
          .z5 = true,
          .z6 = true,
          .z7 = true,
          .z8 = true,
          .z9 = true,
          .z10 = true,
          .z11 = true,
          .z12 = true,
          .z13 = true,
          .z14 = true,
          .z15 = true,
          .z16 = true,
          .z17 = true,
          .z18 = true,
          .z19 = true,
          .z20 = true,
          .z21 = true,
          .z22 = true,
          .z23 = true,
          .z24 = true,
          .z25 = true,
          .z26 = true,
          .z27 = true,
          .z28 = true,
          .z29 = true,
          .z30 = true,
          .z31 = true,
          .x9 = true,
          .x10 = true,
          .x20 = true,
          .x21 = true,
          .x22 = true,
          .x23 = true,
          .x24 = true,
          .x25 = true,
          .x26 = true,
          .x27 = true,
          .x28 = true,
        });
}

comptime {
    _ = std;
    _ = c;
}
