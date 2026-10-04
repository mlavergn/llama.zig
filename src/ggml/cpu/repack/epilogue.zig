//! The one float expression every generic `repack` kernel ends on, and the
//! fusion clang puts in it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Not a single function there — twenty copies of one
//! statement, written out in each generic gemv and gemm:
//!
//! ```c
//! sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
//! ```
//!
//! # The fusion is named, because there is an oracle
//!
//! `CLAUDE.md`'s rule is to name a fusion only when something can tell you
//! you got it wrong. `scripts/repack-diff` is that something: it runs
//! every one of these kernels against the reference C++ in one process and
//! compares on bits. Before this helper existed it reported 12 of 36
//! kernels differing, every one of them by one ULP.
//!
//! **Which** multiply clang fuses was read off the reference's
//! disassembly, not reasoned about. `ref_ggml_gemv_iq4_nl_8x8_q8_0` in
//! `repack.cpp`'s object, at `-O2` with the default `-ffp-contract=on`:
//!
//! ```text
//! scvtf  s2, w8        ; sumi -> f32
//! fcvt   s3, h3        ; the weight's d
//! fmul   s2, s2, s3    ; sumi * d_weight  -- rounded
//! fmadd  s2, s2, s1, s4 ; (that) * d_act + sumf  -- fused
//! ```
//!
//! So the *inner* product rounds and the *outer* one fuses with the add.
//! That is the same shape `CLAUDE.md` records for `xielu`: the left
//! operand of the `+`.

const std = @import("std");

/// `acc += sumi * d_weight * d_act`, with clang's contraction.
///
/// Parameters:
/// - `acc`: the running `sumf` or `sum_minf` for this column.
/// - `sumi`: the integer dot product, or the integer min product.
/// - `d_weight`: the weight block's scale for this column, already f32.
/// - `d_act`: the activation block's scale for this row.
///
/// Return: the new accumulator.
pub inline fn accumulate(acc: f32, sumi: i32, d_weight: f32, d_act: f32) f32 {
    return @mulAdd(f32, @as(f32, @floatFromInt(sumi)) * d_weight, d_act, acc);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the outer multiply is fused" {
    // `3 * (1/3)` is `1 + 2^-26` exactly, which rounds to 1.0 in f32. Added
    // to -1.0 the strict form cancels to zero; the fused one keeps the
    // 2^-26, because the product never gets rounded.
    const sumi: i32 = 3;
    const d_weight: f32 = 1.0;
    const d_act: f32 = 1.0 / 3.0;
    const acc: f32 = -1.0;

    const fused = accumulate(acc, sumi, d_weight, d_act);
    const strict = acc + (@as(f32, @floatFromInt(sumi)) * d_weight) * d_act;
    try std.testing.expectEqual(@as(f32, 0.0), strict);
    try std.testing.expect(fused != 0.0);
}
