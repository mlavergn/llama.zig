//! `ssm_conv` and `ssm_scan`: the state-space-model kernels Mamba and the
//! hybrid architectures run.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Two different state updates, on purpose
//!
//! `ssm_scan`'s Mamba-2 arm runs its `d_state` loop through `GGML_SIMD` for
//! the first `nc & ~15` elements and a scalar loop for the rest. They do not
//! compute the same thing:
//!
//! - the **vector body** is `vmulq`, `vmulq`, `vaddq`, then `vfmaq` into one
//!   of four accumulators. Separate intrinsics are separate IR instructions;
//!   `-ffp-contract=on` fuses within one expression only, so the state is
//!   **two roundings and an add**.
//! - the **scalar tail** is `(s0[i] * dA) + (B[ig] * x_dt)`, one expression,
//!   which clang fuses on its left product.
//!
//! Both are reproduced as the C has them. The four accumulators are reduced
//! by the NEON `GGML_F32x4_REDUCE` tree and the pairwise `vaddvq_f32`, and the
//! tail then adds on to that `sumf`.
//!
//! # Loop index names
//!
//! `i0`, `i1`, `i2` and `i3` are Zig integer type names. Renamed `j0`…`j3`,
//! digit for digit.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const neon = @import("../quants/arm/neon.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;
const f32x4 = neon.f32x4;

extern fn expf(x: f32) f32;
extern fn logf(x: f32) f32;

/// Byte offset `i*nb`; see `common.byteOff`.
const off = common.byteOff;

/// Ports `ggml_compute_softplus_f32` (ggml-impl.h:107 @c1d0e7a00) through
/// libm.
///
/// `impl.softplus` is the same expression on Zig's `@log` and `@exp`, which
/// lower to LLVM intrinsics and need not match libm in the last bit. A private
/// copy keeps this file on the functions the C calls.
inline fn softplus(input: f32) f32 {
    return if (input > 20.0) input else logf(1 + expf(input));
}

// -----------------------------------------------------------------------------
// ssm_conv

/// Ports `ggml_compute_forward_ssm_conv_f32` (ops.cpp:9564 @c1d0e7a00).
///
/// `sumf += s[...] * c[...]` is one expression, but the loop is a
/// contiguous reduction and the reference compiler vectorizes it: products in
/// groups of four are rounded, only the remainder is fused (`vec.strictDot`).
/// Fusing every step was 1 ULP out on 61 of 160 outputs under `make
/// ops-diff`. The C says why it does not call `ggml_vec_dot_f32`: that one
/// sums in `double`.
fn ssmConvF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // conv_x
    const src1 = impl.one(Tensor, dst.src[1]); // conv1d.weight

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nc = src1.ne[0]; // d_conv
    const ncs = src0.ne[0]; // d_conv - 1 + n_t
    const nr = src0.ne[1]; // d_inner
    const n_t = dst.ne[1]; // tokens per sequence
    const n_s = dst.ne[2]; // number of sequences in the batch

    impl.assert(dst.ne[0] == nr, "dst->ne[0] == nr");
    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "src1->nb[0] == sizeof(float)");
    impl.assert(src0.nb[1] == @as(usize, @intCast(src0.ne[0])) * @sizeOf(f32), "src0->nb[1] == src0->ne[0]*sizeof(float)");

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);
    const ir = ir1 - ir0;

    const s_base: [*]const u8 = @ptrCast(src0.data.?);
    const c_base: [*]const u8 = @ptrCast(src1.data.?);
    const x_base: [*]u8 = @ptrCast(dst.data.?);

    var j3: i64 = 0;
    while (j3 < n_s) : (j3 += 1) {
        var j2: i64 = 0;
        while (j2 < n_t) : (j2 += 1) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            const s: [*]const f32 = @ptrCast(@alignCast(s_base + off(ir0, src0.nb[1]) + off(j2, src0.nb[0]) + off(j3, src0.nb[2]))); // {d_conv, d_inner, n_s}
            const cw: [*]const f32 = @ptrCast(@alignCast(c_base + off(ir0, src1.nb[1]))); // {d_conv, d_inner}
            const x: [*]f32 = @ptrCast(@alignCast(x_base + off(ir0, dst.nb[0]) + off(j2, dst.nb[1]) + off(j3, dst.nb[2]))); // {d_inner, n_t, n_s}

            // d_inner
            var j1: i64 = 0;
            while (j1 < ir) : (j1 += 1) {
                // rowwise dot product
                // d_conv
                const sumf = vec.strictDot(s + @as(usize, @intCast(j1 * ncs)), cw + @as(usize, @intCast(j1 * nc)), 0, @intCast(nc), 0.0);
                x[@intCast(j1)] = sumf;
            }
        }
    }
}

/// Ports `ggml_compute_forward_ssm_conv` (ops.cpp:9617 @c1d0e7a00).
pub export fn ggml_compute_forward_ssm_conv(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    switch (impl.one(Tensor, dst.src[0]).type) {
        c.GGML_TYPE_F32 => ssmConvF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// ssm_scan

/// `GGML_F32_STEP` and `GGML_F32_EPR` (simd-mappings.h:337, 338 @c1d0e7a00),
/// the NEON arm.
const f32_step: i64 = 16;
const f32_epr: usize = 4;

/// Ports `ggml_compute_forward_ssm_scan_f32` (ops.cpp:9634 @c1d0e7a00). The
/// Mamba-2 arm is the `GGML_SIMD` NEON one; the Mamba-1 arm is the
/// non-SVE scalar loop.
fn ssmScanF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // s  {d_state, dim, n_head, n_seqs+}
    const src1 = impl.one(Tensor, dst.src[1]); // x  {dim, n_head, n_seq_tokens, n_seqs}
    const src2 = impl.one(Tensor, dst.src[2]); // dt {n_head, n_seq_tokens, n_seqs}
    const src3 = impl.one(Tensor, dst.src[3]); // A  {d_state, n_head} or {1, n_head}
    const src4 = impl.one(Tensor, dst.src[4]); // B  {d_state, n_group, n_seq_tokens, n_seqs}
    const src5 = impl.one(Tensor, dst.src[5]); // C  {d_state, n_group, n_seq_tokens, n_seqs}
    const src6 = impl.one(Tensor, dst.src[6]); // ids {n_seqs}

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nc = src0.ne[0]; // d_state
    const nr = src0.ne[1]; // dim
    const nh = src1.ne[1]; // n_head
    const ng = src4.ne[1];
    const nt = src1.ne[2]; // number of tokens per sequence
    const ns = src1.ne[3]; // number of sequences in the batch
    const K: i64 = impl.getOpParamsI32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    const s_off: usize = @as(usize, @intCast(c.ggml_nelements(src1))) * c.ggml_element_size(src1);

    impl.assert(K >= 1, "K >= 1");
    impl.assert(c.ggml_nelements(src1) + K * nc * nr * nh * ns == c.ggml_nelements(dst), "ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst)");
    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "src1->nb[0] == sizeof(float)");
    impl.assert(src2.nb[0] == @sizeOf(f32), "src2->nb[0] == sizeof(float)");
    impl.assert(src3.nb[0] == @sizeOf(f32), "src3->nb[0] == sizeof(float)");
    impl.assert(src4.nb[0] == @sizeOf(f32), "src4->nb[0] == sizeof(float)");
    impl.assert(src5.nb[0] == @sizeOf(f32), "src5->nb[0] == sizeof(float)");
    impl.assert(src6.nb[0] == @sizeOf(i32), "src6->nb[0] == sizeof(int32_t)");
    impl.assert(@rem(nh, ng) == 0, "nh % ng == 0");
    impl.assert(src3.ne[0] == 1 or K == 1, "src3->ne[0] == 1 || K == 1");

    // heads per thread
    const dh = @divTrunc(nh + nth - 1, nth);

    // head range for this thread
    const ih0 = dh * ith;
    const ih1 = @min(ih0 + dh, nh);

    const ids: [*]const i32 = @ptrCast(@alignCast(src6.data.?));

    const b0: [*]const u8 = @ptrCast(src0.data.?);
    const b1: [*]const u8 = @ptrCast(src1.data.?);
    const b2: [*]const u8 = @ptrCast(src2.data.?);
    const b4: [*]const u8 = @ptrCast(src4.data.?);
    const b5: [*]const u8 = @ptrCast(src5.data.?);
    const bd: [*]u8 = @ptrCast(dst.data.?);

    const A: [*]const f32 = @ptrCast(@alignCast(src3.data.?)); // {d_state, nh} or {1, nh}

    var j3: i64 = 0;
    while (j3 < ns) : (j3 += 1) {
        var s0: [*]const f32 = @ptrCast(@alignCast(b0 + off(ids[@intCast(j3)], src0.nb[3]))); // {d_state, dim, nh, ns}
        const s: [*]f32 = @ptrCast(@alignCast(bd + off(j3, src0.nb[3]) + s_off)); // {d_state, dim, nh, ns}

        var j2: i64 = 0;
        while (j2 < nt) : (j2 += 1) {
            const x: [*]const f32 = @ptrCast(@alignCast(b1 + off(j2, src1.nb[2]) + off(j3, src1.nb[3]))); // {dim, nh, nt, ns}
            const dt: [*]const f32 = @ptrCast(@alignCast(b2 + off(j2, src2.nb[1]) + off(j3, src2.nb[2]))); // {nh, nt, ns}
            const B: [*]const f32 = @ptrCast(@alignCast(b4 + off(j2, src4.nb[2]) + off(j3, src4.nb[3]))); // {d_state, ng, nt, ns}
            const C: [*]const f32 = @ptrCast(@alignCast(b5 + off(j2, src5.nb[2]) + off(j3, src5.nb[3]))); // {d_state, ng, nt, ns}
            const y: [*]f32 = @ptrCast(@alignCast(bd + off(j2, @as(usize, @intCast(nh * nr)) * @sizeOf(f32)) +
                off(j3, @as(usize, @intCast(nt * nh * nr)) * @sizeOf(f32)))); // {dim, nh, nt, ns}

            if (src3.ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                var h: i64 = ih0;
                while (h < ih1) : (h += 1) {
                    const hu: usize = @intCast(h);
                    const dt_soft_plus = softplus(dt[hu]);
                    const dA = expf(dt_soft_plus * A[hu]);
                    const g = @divTrunc(h, @divTrunc(nh, ng)); // repeat_interleave

                    // dim
                    var j1: i64 = 0;
                    while (j1 < nr) : (j1 += 1) {
                        const ii = j1 + h * nr;
                        const iiu: usize = @intCast(ii);
                        const x_dt = x[iiu] * dt_soft_plus;
                        var sumf: f32 = 0.0;

                        const np = nc & ~(f32_step - 1);

                        var sum: [4]f32x4 = .{@as(f32x4, @splat(0.0))} ** 4;

                        const adA: f32x4 = @splat(dA);
                        const axdt: f32x4 = @splat(x_dt);

                        var i: i64 = 0;
                        while (i < np) : (i += f32_step) {
                            for (0..4) |j| {
                                const so: usize = @as(usize, @intCast(i + ii * nc)) + j * f32_epr;
                                const go: usize = @as(usize, @intCast(i + g * nc)) + j * f32_epr;
                                var ax: f32x4 = s0[so..][0..4].*;
                                var ay: f32x4 = B[go..][0..4].*;
                                const az: f32x4 = C[go..][0..4].*;

                                ax = ax * adA;
                                ay = ay * axdt;

                                ax = ax + ay;

                                sum[j] = neon.fma_f32(sum[j], ax, az);

                                s[so..][0..4].* = ax;
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        sumf = common.f32VecReduce(&sum);

                        // d_state
                        //
                        // The compiler vectorizes this tail when its guards
                        // pass, rounding `state * C` in groups of four; see
                        // `common.strictLoopVectorized`. `state` itself is
                        // elementwise and fused either way.
                        const t0: usize = @intCast(np + ii * nc);
                        const tg: usize = @intCast(np + g * nc);
                        const split: i64 = if (common.strictLoopVectorized(nc - np, s + t0, &.{ s0 + t0, B + tg, C + tg }))
                            np + (nc - np) - @rem(nc - np, 4)
                        else
                            np;
                        var j0: i64 = np;
                        while (j0 < nc) : (j0 += 1) {
                            const iu: usize = @intCast(j0 + ii * nc);
                            const ig: usize = @intCast(j0 + g * nc);
                            // state = prev_state * dA + dB * x
                            // One expression: clang fuses the left product.
                            const state = @mulAdd(f32, s0[iu], dA, B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            if (j0 < split) sumf += state * C[ig] else sumf = @mulAdd(f32, state, C[ig], sumf);
                            s[iu] = state;
                        }
                        y[iiu] = sumf;
                    }
                }
            } else {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                var h: i64 = ih0;
                while (h < ih1) : (h += 1) {
                    const dt_soft_plus = softplus(dt[@intCast(h)]);
                    const g = @divTrunc(h, @divTrunc(nh, ng)); // repeat_interleave

                    // dim
                    var j1: i64 = 0;
                    while (j1 < nr) : (j1 += 1) {
                        const ii = j1 + h * nr;
                        const iiu: usize = @intCast(ii);
                        const x_dt = x[iiu] * dt_soft_plus;
                        var sumf: f32 = 0.0;
                        // d_state
                        //
                        // Vectorized by the compiler despite the `expf` --
                        // each lane's call is scalarized -- when its guards
                        // pass: then `state * C` is rounded in groups of
                        // four and only the remainder fused. Measured on the
                        // reference `ops.o`; see `common.strictLoopVectorized`.
                        const r0: usize = @intCast(ii * nc);
                        const rg: usize = @intCast(g * nc);
                        const split: i64 = if (common.strictLoopVectorized(nc, s + r0, &.{ s0 + r0, A + @as(usize, @intCast(h * nc)), B + rg, C + rg }))
                            nc - @rem(nc, 4)
                        else
                            0;
                        var j0: i64 = 0;
                        while (j0 < nc) : (j0 += 1) {
                            const iu: usize = @intCast(j0 + ii * nc);
                            const ig: usize = @intCast(j0 + g * nc);
                            // state = prev_state * dA + dB * x
                            // One expression: clang fuses the left product,
                            // whose right operand is the `expf` call.
                            const state = @mulAdd(f32, s0[iu], expf(dt_soft_plus * A[@intCast(j0 + h * nc)]), B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            if (j0 < split) sumf += state * C[ig] else sumf = @mulAdd(f32, state, C[ig], sumf);
                            s[iu] = state;
                        }
                        y[iiu] = sumf;
                    }
                }
            }
            const slot = nt - 1 - j2;
            if (K > 1 and slot > 0 and slot < K) {
                const s_snapshot: [*]u8 = bd + s_off + off(slot * ns + j3, src0.nb[3]);
                const sb: [*]const u8 = @ptrCast(s);
                var h: i64 = ih0;
                while (h < ih1) : (h += 1) {
                    const o = off(h, src0.nb[2]);
                    @memcpy(s_snapshot[o..][0..src0.nb[2]], sb[o..][0..src0.nb[2]]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

/// Ports `ggml_compute_forward_ssm_scan` (ops.cpp:9857 @c1d0e7a00).
pub export fn ggml_compute_forward_ssm_scan(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    switch (impl.one(Tensor, dst.src[0]).type) {
        c.GGML_TYPE_F32 => ssmScanF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
