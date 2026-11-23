//! `conv_transpose_1d`, `conv_transpose_2d`, `im2col`, `im2col_back_f32`,
//! `im2col_3d`, `col2im_1d`, `conv_2d`, `conv_3d` and `conv_2d_dw`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v1.0.0 (`b1b770d90`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Two kernels reach the matmul
//!
//! `conv_2d` and `conv_3d` lower to an im2col into the work buffer followed by
//! a call to `ggml_compute_forward_mul_mat` on stack-built tensors. That
//! function came with `ggml-cpu.c` and is `cpu/mulmat.zig`; it is declared
//! `extern` here, as the C++ reaches it across a translation-unit boundary.
//!
//! # `conv_2d_dw` fuses on both arms
//!
//! The channels-last kernel has a live NEON arm (`GGML_SIMD`, no SVE) that
//! accumulates four channels at a time with `vfmaq_f32`, and a scalar tail for
//! the rest. The tail's `sum += k * s` is one expression at
//! `-ffp-contract=on`, so it fuses too. Lanes never meet, and both arms walk
//! the kernel window in the same order, so every channel is the same chain of
//! fused multiply-adds whichever arm takes it — they are one loop here, as
//! `vecinline.zig` does for `mad_f32`.
//!
//! # Integer widths
//!
//! The C narrows several products to `int` (`nk`, `nr`, `ofs0`, `i1n`) and
//! indexes with `int` loop counters. Those only differ from the 64-bit
//! arithmetic used here when the C's own result would overflow, which is
//! undefined; the widths are not reproduced.
//!
//! # Loop index names
//!
//! `i00`, `i01`, `i10`, `i11` and the rest are Zig integer type names. Renamed
//! `j00`, `j01`, `j10`, `j11`, digit for digit.

const std = @import("std");
const ggml = @import("../../module.zig");

/// Ports `ggml_compute_forward_mul_mat` (ggml-cpu.c:1264 @b1b770d90), which
/// `cpu/mulmat.zig` provides, by its C name.
inline fn uz(x: i64) usize {
    return @intCast(x);
}

/// `ggml_barrier(params->threadpool)`; see `common.barrier`.
/// Zeroes the whole work buffer, the C's `memset(params->wdata, 0, params->wsize)`.
inline fn zeroWork(params: *const ggml.cpu.defs.ComputeParams) void {
    const w: [*]u8 = @ptrCast(params.wdata.?);
    @memset(w[0..params.wsize], 0);
}

/// Zeroes `dst`, the C's `memset(dst->data, 0, ggml_nbytes(dst))`.
inline fn zeroDst(dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const d: [*]u8 = @ptrCast(dst.data.?);
    @memset(d[0..ggml.types.ggml_nbytes(dst)], 0);
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_conv_transpose_1d

/// Ports `ggml_compute_forward_conv_transpose_1d_f16_f32` and
/// `ggml_compute_forward_conv_transpose_1d_f32` (ops.cpp:6413, 6499 @b1b770d90).
///
/// The two differ in the kernel element type and, per #31854, in
/// the work buffer's layout: the `f16` variant keeps the permuted source in
/// `f32` **first** and the `f16` kernel after it, and dots them with
/// `ggml_vec_dot_f16_f32` (kernel, then source); the `f32` variant keeps
/// the kernel first and the source after it, and dots source against kernel.
/// One function over a comptime `T`, with the layout chosen on it.
fn convTranspose1d(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    if (T == ggml.impl.ggml_c.ggml_fp16_t) {
        ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F16");
    } else {
        ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    }
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nk = l.ne00 * l.ne01 * l.ne02;

    ggml.impl.assert(l.nb00 == @sizeOf(T), "nb00 == sizeof(T)");
    ggml.impl.assert(l.nb10 == @sizeOf(f32), "nb10 == sizeof(float)");

    const wbytes: [*]u8 = @ptrCast(params.wdata.?);
    // `f16`: source (f32) at 0, kernel (f16) after `ne10*ne11` floats.
    // `f32`: kernel at 0, source after `nk` floats.
    const wkernel: [*]T = if (T == ggml.impl.ggml_c.ggml_fp16_t)
        @ptrCast(@alignCast(wbytes + uz(l.ne10 * l.ne11) * @sizeOf(f32)))
    else
        @ptrCast(@alignCast(wbytes));
    const wsrc: [*]f32 = if (T == ggml.impl.ggml_c.ggml_fp16_t)
        @ptrCast(@alignCast(wbytes))
    else
        @as([*]f32, @ptrCast(@alignCast(wbytes))) + uz(nk);

    if (ith == 0) {
        zeroWork(params);

        if (T == ggml.impl.ggml_c.ggml_fp16_t) permuteSrc1d(&l, src1, wsrc);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            const s0: [*]const u8 = @ptrCast(src0.data.?);

            var j02: i64 = 0;
            while (j02 < l.ne02) : (j02 += 1) {
                var j01: i64 = 0;
                while (j01 < l.ne01) : (j01 += 1) {
                    const src: [*]const T = @ptrCast(@alignCast(s0 + uz(j02) * l.nb02 + uz(j01) * l.nb01));
                    const dst_data = wkernel + uz(j01 * l.ne00 * l.ne02);
                    var j00: i64 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        dst_data[uz(j00 * l.ne02 + j02)] = src[uz(j00)];
                    }
                }
            }
        }

        if (T != ggml.impl.ggml_c.ggml_fp16_t) permuteSrc1d(&l, src1, wsrc);

        // need to zero dst since we are accumulating into it
        zeroDst(dst);
    }
    ggml.cpu.ops.common.barrier(params);

    const s0: i64 = dst.op_params[0];

    // total rows in dst
    const nr = l.ne1;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const dd: [*]u8 = @ptrCast(dst.data.?);

    var j1: i64 = ir0;
    while (j1 < ir1) : (j1 += 1) {
        const dst_data: [*]f32 = @ptrCast(@alignCast(dd + uz(j1) * l.nb1));
        const wdata_kernel = wkernel + uz(j1 * l.ne02 * l.ne00);
        var j10: i64 = 0;
        while (j10 < l.ne10) : (j10 += 1) {
            const j1n = j10 * l.ne11;
            var j00: i64 = 0;
            while (j00 < l.ne00) : (j00 += 1) {
                var v: f32 = 0;
                if (T == ggml.impl.ggml_c.ggml_fp16_t) {
                    ggml.cpu.vec.ggml_vec_dot_f16_f32(@intCast(l.ne02), &v, 0, wdata_kernel + uz(j00 * l.ne02), 0, wsrc + uz(j1n), 0, 1);
                } else {
                    ggml.cpu.vec.ggml_vec_dot_f32(@intCast(l.ne02), &v, 0, wsrc + uz(j1n), 0, wdata_kernel + uz(j00 * l.ne02), 0, 1);
                }
                dst_data[uz(j10 * s0 + j00)] += v;
            }
        }
    }
}

/// The source permutation both `conv_transpose_1d` variants share: (L x Cin)
/// to (Cin x L), in `f32`.
fn permuteSrc1d(l: *const ggml.cpu.defs.BinaryLocals, src1: *ggml.impl.ggml_c.ggml_tensor, dst_data: [*]f32) void {
    const s1: [*]const u8 = @ptrCast(src1.data.?);

    var j11: i64 = 0;
    while (j11 < l.ne11) : (j11 += 1) {
        const src: [*]const f32 = @ptrCast(@alignCast(s1 + uz(j11) * l.nb11));
        var j10: i64 = 0;
        while (j10 < l.ne10) : (j10 += 1) {
            dst_data[uz(j10 * l.ne11 + j11)] = src[uz(j10)];
        }
    }
}

/// Ports `ggml_compute_forward_conv_transpose_1d` (ops.cpp:6587 @b1b770d90).
pub fn ggml_compute_forward_conv_transpose_1d(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F16 => convTranspose1d(ggml.impl.ggml_c.ggml_fp16_t, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F32 => convTranspose1d(f32, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_im2col

/// The geometry `im2col_f32` and `im2col_f16` both unpack from `op_params`
/// and the tensor shapes. The C spells it out in each; gathered here so the
/// two kernels below differ only in what they write.
const Im2col = struct {
    s0: i64,
    s1: i64,
    p0: i64,
    p1: i64,
    d0: i64,
    d1: i64,
    n: i64,
    ic: i64,
    ih: i64,
    iw: i64,
    kh: i64,
    kw: i64,
    oh: i64,
    ow: i64,
    ofs0: usize,
    ofs1: usize,

    fn of(dst: *const ggml.impl.ggml_c.ggml_tensor, l: ggml.cpu.defs.BinaryLocals) Im2col {
        const is_2D = dst.op_params[6] == 1;
        return .{
            .s0 = dst.op_params[0],
            .s1 = dst.op_params[1],
            .p0 = dst.op_params[2],
            .p1 = dst.op_params[3],
            .d0 = dst.op_params[4],
            .d1 = dst.op_params[5],
            .n = if (is_2D) l.ne13 else l.ne12,
            .ic = if (is_2D) l.ne12 else l.ne11,
            .ih = if (is_2D) l.ne11 else 1,
            .iw = l.ne10,
            .kh = if (is_2D) l.ne01 else 1,
            .kw = l.ne00,
            .oh = if (is_2D) l.ne2 else 1,
            .ow = l.ne1,
            .ofs0 = if (is_2D) l.nb13 else l.nb12,
            .ofs1 = if (is_2D) l.nb12 else l.nb11,
        };
    }
};

/// Ports `ggml_compute_forward_im2col_f32` (ops.cpp:6613 @b1b770d90).
///
/// src0: kernel [OC, IC, KH, KW]; src1: image [N, IC, IH, IW];
/// dst: result [N, OH, OW, IC*KH*KW].
fn im2colF32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);
    const g = Im2col.of(dst, l);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    ggml.impl.assert(l.nb10 == @sizeOf(f32), "nb10 == sizeof(float)");

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    const wdata: [*]f32 = @ptrCast(@alignCast(dst.data.?));
    const s1: [*]const u8 = @ptrCast(src1.data.?);

    var in: i64 = 0;
    while (in < g.n) : (in += 1) {
        var ioh: i64 = 0;
        while (ioh < g.oh) : (ioh += 1) {
            var iow: i64 = 0;
            while (iow < g.ow) : (iow += 1) {
                var iic: i64 = ith;
                while (iic < g.ic) : (iic += nth) {
                    // micro kernel
                    const dst_data = wdata + uz((in * g.oh * g.ow + ioh * g.ow + iow) * (g.ic * g.kh * g.kw)); // [IC, KH, KW]
                    const src_data: [*]const f32 = @ptrCast(@alignCast(s1 + uz(in) * g.ofs0 + uz(iic) * g.ofs1)); // [IH, IW]

                    var ikh: i64 = 0;
                    while (ikh < g.kh) : (ikh += 1) {
                        var ikw: i64 = 0;
                        while (ikw < g.kw) : (ikw += 1) {
                            const iiw = iow * g.s0 + ikw * g.d0 - g.p0;
                            const iih = ioh * g.s1 + ikh * g.d1 - g.p1;

                            const di = uz(iic * (g.kh * g.kw) + ikh * g.kw + ikw);
                            if (iih < 0 or iih >= g.ih or iiw < 0 or iiw >= g.iw) {
                                dst_data[di] = 0;
                            } else {
                                dst_data[di] = src_data[uz(iih * g.iw + iiw)];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_im2col_f16` (ops.cpp:6689 @b1b770d90).
///
/// The same walk as `im2colF32`, writing `f16`, from an `f16` or `f32` image.
fn im2colF16(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F16, "dst->type == GGML_TYPE_F16");

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);
    const g = Im2col.of(dst, l);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    ggml.impl.assert(l.nb10 == ggml.types.ggml_type_size(src1.type), "nb10 == ggml_type_size(src1->type)");

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    const wdata: [*]ggml.impl.ggml_c.ggml_fp16_t = @ptrCast(@alignCast(dst.data.?));
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const is_f32 = src1.type == ggml.impl.ggml_c.GGML_TYPE_F32;

    var in: i64 = 0;
    while (in < g.n) : (in += 1) {
        var ioh: i64 = 0;
        while (ioh < g.oh) : (ioh += 1) {
            var iow: i64 = 0;
            while (iow < g.ow) : (iow += 1) {
                var iic: i64 = ith;
                while (iic < g.ic) : (iic += nth) {
                    // micro kernel
                    const dst_data = wdata + uz((in * g.oh * g.ow + ioh * g.ow + iow) * (g.ic * g.kh * g.kw)); // [IC, KH, KW]
                    const src_row = s1 + uz(in) * g.ofs0 + uz(iic) * g.ofs1; // [IH, IW]
                    const src_data_f32: [*]const f32 = @ptrCast(@alignCast(src_row));
                    const src_data_f16: [*]const ggml.impl.ggml_c.ggml_fp16_t = @ptrCast(@alignCast(src_row));

                    var ikh: i64 = 0;
                    while (ikh < g.kh) : (ikh += 1) {
                        var ikw: i64 = 0;
                        while (ikw < g.kw) : (ikw += 1) {
                            const iiw = iow * g.s0 + ikw * g.d0 - g.p0;
                            const iih = ioh * g.s1 + ikh * g.d1 - g.p1;

                            const di = uz(iic * (g.kh * g.kw) + ikh * g.kw + ikw);
                            if (iih < 0 or iih >= g.ih or iiw < 0 or iiw >= g.iw) {
                                dst_data[di] = 0;
                            } else if (is_f32) {
                                dst_data[di] = ggml.cpu.convert.cpuFp32ToFp16(src_data_f32[uz(iih * g.iw + iiw)]);
                            } else {
                                dst_data[di] = src_data_f16[uz(iih * g.iw + iiw)];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_im2col` (ops.cpp:6769 @b1b770d90).
pub fn ggml_compute_forward_im2col(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    switch (dst.type) {
        ggml.impl.ggml_c.GGML_TYPE_F16 => im2colF16(params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F32 => im2colF32(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_im2col_back_f32` (ops.cpp:6790 @b1b770d90).
///
/// Note the roles swap against `im2col`: the image geometry comes from `dst`
/// and the kernel's from `src1`. `%` and `/` on a negative offset truncate
/// toward zero in the C, hence `@rem` and `@divTrunc`.
pub fn ggml_compute_forward_im2col_back_f32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]); // gradients of forward pass output
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]); // convolution kernel

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src1.type == ggml.impl.ggml_c.GGML_TYPE_F16, "src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const s0: i64 = dst.op_params[0];
    const s1: i64 = dst.op_params[1];
    const p0: i64 = dst.op_params[2];
    const p1: i64 = dst.op_params[3];
    const d0: i64 = dst.op_params[4];
    const d1: i64 = dst.op_params[5];
    const is_2D = dst.op_params[6] == 1;

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const N = if (is_2D) l.ne3 else l.ne2;
    const IC = if (is_2D) l.ne2 else l.ne1;
    const IH = if (is_2D) l.ne1 else 1;
    const IW = l.ne0;

    const KH = if (is_2D) l.ne11 else 1;
    const KW = l.ne10;

    const OH = if (is_2D) l.ne02 else 1;
    const OW = l.ne01;

    const ofs0 = if (is_2D) l.nb3 else l.nb2;
    const ofs1 = if (is_2D) l.nb2 else l.nb1;

    ggml.impl.assert(l.nb0 == @sizeOf(f32), "nb0  == sizeof(float)");

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    const wdata: [*]u8 = @ptrCast(dst.data.?);
    const grad_base: [*]const f32 = @ptrCast(@alignCast(src0.data.?));

    var in: i64 = 0;
    while (in < N) : (in += 1) {
        var iic: i64 = ith;
        while (iic < IC) : (iic += nth) {
            var iih: i64 = 0;
            while (iih < IH) : (iih += 1) {
                var iiw: i64 = 0;
                while (iiw < IW) : (iiw += 1) {
                    // micro kernel
                    var grad: f32 = 0.0;
                    var ikh: i64 = 0;
                    while (ikh < KH) : (ikh += 1) {
                        var ikw: i64 = 0;
                        while (ikw < KW) : (ikw += 1) {
                            // For s0 > 1 some values were skipped over in the forward pass.
                            // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                            const tmpw = iiw + p0 - ikw * d0;
                            if (@rem(tmpw, s0) != 0) continue;
                            const iow = @divTrunc(tmpw, s0);

                            // Equivalent logic as above except for s1.
                            var ioh: i64 = undefined;
                            if (is_2D) {
                                const tmph = iih + p1 - ikh * d1;
                                if (@rem(tmph, s1) != 0) continue;
                                ioh = @divTrunc(tmph, s1);
                            } else {
                                ioh = 0;
                            }

                            if (iow < 0 or iow >= OW or ioh < 0 or ioh >= OH) continue;

                            const grad_in = grad_base + uz((in * OH * OW + ioh * OW + iow) * (IC * KH * KW)); // [IC, KH, KW]
                            grad += grad_in[uz(iic * (KH * KW) + ikh * KW + ikw)];
                        }
                    }
                    const dst_data: [*]f32 = @ptrCast(@alignCast(wdata + (uz(in) * ofs0 + uz(iic) * ofs1))); // [IH, IW]
                    dst_data[uz(iih * IW + iiw)] = grad;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_im2col_3d

/// Ports `ggml_compute_forward_im2col_3d_f16` and
/// `ggml_compute_forward_im2col_3d_f32` (ops.cpp:6888, 6978 @b1b770d90).
///
/// src0: kernel [OC*IC, KD, KH, KW]; src1: image [N*IC, ID, IH, IW];
/// dst: result [N*OD, OH, OW, IC * KD * KH * KW].
///
/// Identical but for the element they write, so one function over `T`. The
/// `f32` variant's bounds test repeats `iid < 0 || iid >= ID` at its end; it
/// is redundant and has no effect.
fn im2col3d(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    if (T == ggml.impl.ggml_c.ggml_fp16_t) {
        ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F16, "dst->type == GGML_TYPE_F16");
    } else {
        ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");
    }

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const s0: i64 = dst.op_params[0];
    const s1: i64 = dst.op_params[1];
    const s2: i64 = dst.op_params[2];
    const p0: i64 = dst.op_params[3];
    const p1: i64 = dst.op_params[4];
    const p2: i64 = dst.op_params[5];
    const d0: i64 = dst.op_params[6];
    const d1: i64 = dst.op_params[7];
    const d2: i64 = dst.op_params[8];
    const IC: i64 = dst.op_params[9];

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const N = @divTrunc(l.ne13, IC);
    const ID = l.ne12;
    const IH = l.ne11;
    const IW = l.ne10;

    const KD = l.ne02;
    const KH = l.ne01;
    const KW = l.ne00;

    const OD = @divTrunc(l.ne3, N);
    const OH = l.ne2;
    const OW = l.ne1;
    const OH_OW = OH * OW;
    const KD_KH_KW = KD * KH * KW;
    const KH_KW = KH * KW;
    const IC_KD_KH_KW = IC * KD * KH * KW;

    ggml.impl.assert(l.nb10 == @sizeOf(f32), "nb10 == sizeof(float)");

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    const wdata: [*]T = @ptrCast(@alignCast(dst.data.?));
    const sb: [*]const u8 = @ptrCast(src1.data.?);

    var in: i64 = 0;
    while (in < N) : (in += 1) {
        var iod: i64 = 0;
        while (iod < OD) : (iod += 1) {
            var ioh: i64 = 0;
            while (ioh < OH) : (ioh += 1) {
                var iow: i64 = 0;
                while (iow < OW) : (iow += 1) {
                    var iic: i64 = ith;
                    while (iic < IC) : (iic += nth) {
                        // micro kernel
                        const dst_data = wdata + uz((in * OD * OH_OW + iod * OH_OW + ioh * OW + iow) * IC_KD_KH_KW); // [IC, KD, KH, KW]
                        const src_data = sb + uz(in * IC + iic) * l.nb13; // [ID, IH, IW]

                        var ikd: i64 = 0;
                        while (ikd < KD) : (ikd += 1) {
                            var ikh: i64 = 0;
                            while (ikh < KH) : (ikh += 1) {
                                var ikw: i64 = 0;
                                while (ikw < KW) : (ikw += 1) {
                                    const iiw = iow * s0 + ikw * d0 - p0;
                                    const iih = ioh * s1 + ikh * d1 - p1;
                                    const iid = iod * s2 + ikd * d2 - p2;

                                    const di = uz(iic * KD_KH_KW + ikd * KH_KW + ikh * KW + ikw);
                                    if (iid < 0 or iid >= ID or iih < 0 or iih >= IH or iiw < 0 or iiw >= IW) {
                                        dst_data[di] = ggml.cpu.ops.common.fromF32(T, 0);
                                    } else {
                                        const s: *const f32 = @ptrCast(@alignCast(src_data + uz(iid) * l.nb12 + uz(iih) * l.nb11 + uz(iiw) * l.nb10)); // [ID, IH, IW]
                                        dst_data[di] = ggml.cpu.ops.common.fromF32(T, s.*);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_im2col_3d` (ops.cpp:7066 @b1b770d90).
pub fn ggml_compute_forward_im2col_3d(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    switch (dst.type) {
        ggml.impl.ggml_c.GGML_TYPE_F16 => im2col3d(ggml.impl.ggml_c.ggml_fp16_t, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F32 => im2col3d(f32, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// The matmul the convolutions lower to

/// Ports `ggml_call_mul_mat` (ops.cpp:7085 @b1b770d90).
///
/// Builds three contiguous 2-D tensors on the stack around caller-owned
/// buffers and hands them to `ggml_compute_forward_mul_mat`: `a` is
/// `[m, k]` of `src1_type`, `b` is `[n, k]` of `src0_type`, and `out`
/// receives `[m, n]` as `f32`. Every field not set is zero, as the C's `= {}`
/// leaves it. The two types are separate (#31854): the
/// convolutions keep their patches in `f32` whatever the kernel is.
fn callMulMat(src0_type: ggml.impl.ggml_c.ggml_type, src1_type: ggml.impl.ggml_c.ggml_type, params: *const ggml.cpu.defs.ComputeParams, m: i64, n: i64, k: i64, a: *anyopaque, b: *anyopaque, out: [*]f32) void {
    const ts0 = ggml.types.ggml_get_type_traits(src0_type).*.type_size;
    const ts1 = ggml.types.ggml_get_type_traits(src1_type).*.type_size;

    var src1 = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    src1.type = src1_type;
    src1.ne = .{ k, m, 1, 1 };
    src1.nb[0] = ts1;
    src1.nb[1] = uz(k) * ts1;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.data = a;

    var src0 = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    src0.type = src0_type;
    src0.ne = .{ k, n, 1, 1 };
    src0.nb[0] = ts0;
    src0.nb[1] = uz(k) * ts0;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.data = b;

    var dst = std.mem.zeroes(ggml.impl.ggml_c.ggml_tensor);
    dst.ne = .{ n, m, 1, 1 };
    dst.nb[0] = @sizeOf(f32);
    dst.nb[1] = uz(n) * @sizeOf(f32);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data = out;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml.cpu.mulmat.ggml_compute_forward_mul_mat(params, &dst);
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_col2im_1d

/// Ports `ggml_compute_forward_col2im_1d_impl` (ops.cpp:7141 @b1b770d90).
///
/// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC], where
/// T_out = (T_in - 1)*s + K - 2*p, done as a gather: each output reads
/// ceil(K/s) inputs. Parallelised over the time axis so the split stays
/// balanced whatever OC is. Accumulates in `f32` whatever `T` is.
fn col2im1d(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]); // [K*OC, T_in]

    ggml.impl.assert(ggml.types.ggml_is_contiguous(src), "ggml_is_contiguous(src)");
    ggml.impl.assert(ggml.types.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    const s0: i64 = dst.op_params[0];
    const OC: i64 = dst.op_params[1];
    const p0: i64 = dst.op_params[2];

    const K_OC = src.ne[0];
    const T_in = src.ne[1];
    const K = @divTrunc(K_OC, OC);
    const T_out = dst.ne[0];

    const col_data: [*]const T = @ptrCast(@alignCast(src.data.?));
    const dst_data: [*]T = @ptrCast(@alignCast(dst.data.?));

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    const dr = @divTrunc(T_out + nth - 1, nth);
    const it0 = dr * ith;
    const it1 = if (it0 + dr < T_out) it0 + dr else T_out;

    var oc: i64 = 0;
    while (oc < OC) : (oc += 1) {
        var t_out = it0;
        while (t_out < it1) : (t_out += 1) {
            const t_abs = t_out + p0; // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            var t_in_min = @divTrunc(t_abs - K + 1 + s0 - 1, s0); // ceil((t_abs-K+1)/s)
            if (t_in_min < 0) t_in_min = 0;
            var t_in_max = @divTrunc(t_abs, s0);
            if (t_in_max >= T_in) t_in_max = T_in - 1;

            var sum: f32 = 0.0;
            var t_in = t_in_min;
            while (t_in <= t_in_max) : (t_in += 1) {
                const k = t_abs - t_in * s0;
                if (k >= 0 and k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += ggml.cpu.ops.common.toF32(T, col_data[uz((oc * K + k) + t_in * K_OC)]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[uz(t_out + oc * T_out)] = ggml.cpu.ops.common.fromF32(T, sum);
        }
    }
}

/// Ports `ggml_compute_forward_col2im_1d` (ops.cpp:7194 @b1b770d90).
///
/// The C's abort message carries the type number through a format string;
/// `impl.abort` takes a literal, so the number is dropped.
pub fn ggml_compute_forward_col2im_1d(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    switch (ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]).type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => col2im1d(f32, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F16 => col2im1d(ggml.impl.ggml_c.ggml_fp16_t, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_BF16 => col2im1d(ggml.impl.ggml_c.ggml_bf16_t, params, dst),
        else => ggml.impl.abort("col2im_1d: unsupported type"),
    }
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_conv_2d

/// Ports `ggml_compute_forward_conv_2d_impl` (ops.cpp:7208 @b1b770d90).
///
/// kernel [KW, KH, IC, OC], src [W, H, C, N], dst [OW, OH, OC, N]. Patches
/// are im2col'd into the work buffer in batches sized to fit it, multiplied
/// against the kernel, and permuted back into `dst`.
fn conv2dImpl(params: *const ggml.cpu.defs.ComputeParams, kernel: *const ggml.impl.ggml_c.ggml_tensor, src: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, kernel_type: ggml.impl.ggml_c.ggml_type) void {
    ggml.impl.assert(ggml.types.ggml_is_contiguous(kernel), "ggml_is_contiguous(kernel)");
    ggml.impl.assert(kernel_type == ggml.impl.ggml_c.GGML_TYPE_F16 or kernel_type == ggml.impl.ggml_c.GGML_TYPE_F32, "kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32");
    ggml.impl.assert(kernel.type == kernel_type, "kernel->type == kernel_type");

    const stride_x: i64 = dst.op_params[0];
    const stride_y: i64 = dst.op_params[1];
    const pad_x: i64 = dst.op_params[2];
    const pad_y: i64 = dst.op_params[3];
    const dilation_x: i64 = dst.op_params[4];
    const dilation_y: i64 = dst.op_params[5];

    const c_in = src.ne[2];
    const c_out = kernel.ne[3];
    ggml.impl.assert(c_in == kernel.ne[2], "c_in == kernel->ne[2]");

    const src_w = src.ne[0];
    const src_h = src.ne[1];
    const knl_w = kernel.ne[0];
    const knl_h = kernel.ne[1];
    const dst_w = dst.ne[0];
    const dst_h = dst.ne[1];

    const src_data: [*]const u8 = @ptrCast(src.data.?);
    const knl_data = kernel.data.?;
    const dst_data: [*]u8 = @ptrCast(dst.data.?);

    const knl_n = knl_w * knl_h * c_in;
    const patch_total = dst.ne[3] * dst_w * dst_h;

    const space_per_patch: usize = uz(knl_n) * @sizeOf(f32) + uz(c_out) * @sizeOf(f32);
    const batch_size: i64 = @intCast(params.wsize / space_per_patch);
    const patches_per_batch = if (batch_size > 8) @divTrunc(batch_size, 8) * 8 else batch_size;
    const batch_n = @divTrunc(patch_total + patches_per_batch - 1, patches_per_batch);

    ggml.impl.assert(patches_per_batch > 0 and batch_size >= 1, "patches_per_batch > 0 && batch_size >= 1");

    const tmp: [*]u8 = @ptrCast(params.wdata.?);
    const nth: i64 = params.nth;
    const ith: i64 = params.ith;

    var batch_i: i64 = 0;
    while (batch_i < batch_n) : (batch_i += 1) {
        const patch_start_batch = batch_i * patches_per_batch;
        const patch_end_batch = @min(patch_start_batch + patches_per_batch, patch_total);
        const patch_n = patch_end_batch - patch_start_batch;

        const patch_per_thread = @divTrunc(patch_n + nth - 1, nth);
        const patch_start = patch_start_batch + ith * patch_per_thread;
        const patch_end = @min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        var p = patch_start;
        while (p < patch_end) : (p += 1) {
            const batch_idx = @divTrunc(p, dst_w * dst_h);
            const src_x = @rem(@divTrunc(p, dst_w), dst_h);
            const src_y = @rem(p, dst_w);

            const src_base = src_data + uz(batch_idx) * src.nb[3];
            const dst_row: [*]f32 = @as([*]f32, @ptrCast(@alignCast(tmp))) + uz(@rem(p, patches_per_batch) * knl_n);

            var ic: i64 = 0;
            while (ic < c_in) : (ic += 1) {
                var ky: i64 = 0;
                while (ky < knl_h) : (ky += 1) {
                    var kx: i64 = 0;
                    while (kx < knl_w) : (kx += 1) {
                        const sy = src_x * stride_y + ky * dilation_y - pad_y;
                        const sx = src_y * stride_x + kx * dilation_x - pad_x;

                        const dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        var src_val: f32 = undefined;
                        if (sy < 0 or sy >= src_h or sx < 0 or sx >= src_w) {
                            src_val = 0.0;
                        } else {
                            const src_ptr: *const f32 = @ptrCast(@alignCast(src_base + uz(sx) * src.nb[0] + uz(sy) * src.nb[1] + uz(ic) * src.nb[2]));
                            src_val = src_ptr.*;
                        }

                        dst_row[uz(dst_idx)] = src_val;
                    }
                }
            }
        } // patches handled by this thread

        ggml.cpu.ops.common.barrier(params);

        const gemm_output: [*]f32 = @ptrCast(@alignCast(tmp + uz(patches_per_batch * knl_n) * @sizeOf(f32)));

        // The C compares against `(float*)tmp + params->wsize` -- the work
        // size in *floats*, four times the buffer -- so this bound is loose
        // in the C as well. Reproduced, not tightened.
        ggml.impl.assert(@intFromPtr(gemm_output + uz(patch_n * c_out)) <= @intFromPtr(@as([*]f32, @ptrCast(@alignCast(tmp))) + params.wsize), "gemm_output + patch_n * c_out <= (float*)tmp + params->wsize");

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        callMulMat(kernel_type, ggml.impl.ggml_c.GGML_TYPE_F32, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml.cpu.ops.common.barrier(params);

        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        const permute_per_thread = @divTrunc(patch_n + nth - 1, nth);
        const permute_start = ith * permute_per_thread;
        const permute_end = @min(permute_start + permute_per_thread, patch_n);

        var i = permute_start;
        while (i < permute_end) : (i += 1) {
            const pp = patch_start_batch + i;
            const batch_idx = @divTrunc(pp, dst_w * dst_h);
            const dst_y = @rem(@divTrunc(pp, dst_w), dst_h);
            const dst_x = @rem(pp, dst_w);

            var oc: i64 = 0;
            while (oc < c_out) : (oc += 1) {
                const value = gemm_output[uz(i * c_out + oc)];
                const dst_ptr: *f32 = @ptrCast(@alignCast(dst_data + uz(dst_x) * dst.nb[0] + uz(dst_y) * dst.nb[1] + uz(oc) * dst.nb[2] + uz(batch_idx) * dst.nb[3]));
                dst_ptr.* = value;
            }
        }
    }
}

/// Ports `ggml_compute_forward_conv_2d` (ops.cpp:7326 @b1b770d90).
pub fn ggml_compute_forward_conv_2d(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    conv2dImpl(params, src0, src1, dst, src0.type);
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_conv_3d

/// Ports `ggml_compute_forward_conv_3d_impl` (ops.cpp:7338 @b1b770d90).
///
/// The `conv_2d` scheme with a depth axis. Channels and batch arrive in
/// `op_params` rather than the shapes, since both are folded into `ne[3]`.
fn conv3dImpl(params: *const ggml.cpu.defs.ComputeParams, kernel: *const ggml.impl.ggml_c.ggml_tensor, src: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, kernel_type: ggml.impl.ggml_c.ggml_type) void {
    ggml.impl.assert(ggml.types.ggml_is_contiguous(kernel), "ggml_is_contiguous(kernel)");
    ggml.impl.assert(kernel_type == ggml.impl.ggml_c.GGML_TYPE_F16 or kernel_type == ggml.impl.ggml_c.GGML_TYPE_F32, "kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32");
    ggml.impl.assert(kernel.type == kernel_type, "kernel->type == kernel_type");

    const s0: i64 = dst.op_params[0];
    const s1: i64 = dst.op_params[1];
    const s2: i64 = dst.op_params[2];
    const p0: i64 = dst.op_params[3];
    const p1: i64 = dst.op_params[4];
    const p2: i64 = dst.op_params[5];
    const d0: i64 = dst.op_params[6];
    const d1: i64 = dst.op_params[7];
    const d2: i64 = dst.op_params[8];
    const cc: i64 = dst.op_params[9];
    const n: i64 = dst.op_params[10];
    const oc: i64 = dst.op_params[11];

    const src_w = src.ne[0];
    const src_h = src.ne[1];
    const src_d = src.ne[2];
    const knl_w = kernel.ne[0];
    const knl_h = kernel.ne[1];
    const knl_d = kernel.ne[2];
    const dst_w = dst.ne[0];
    const dst_h = dst.ne[1];
    const dst_d = dst.ne[2];

    const src_data: [*]const u8 = @ptrCast(src.data.?);
    const knl_data = kernel.data.?;
    const dst_data: [*]u8 = @ptrCast(dst.data.?);

    const knl_n_per_channel = knl_w * knl_h * knl_d;
    const knl_n_total = knl_n_per_channel * cc;
    const patch_total = n * dst_w * dst_h * dst_d;

    const space_per_patch: usize = uz(knl_n_total) * @sizeOf(f32) + uz(oc) * @sizeOf(f32);
    const batch_size: i64 = @intCast(params.wsize / space_per_patch);
    const patches_per_batch = if (batch_size > 8) @divTrunc(batch_size, 8) * 8 else batch_size;
    const batch_n = @divTrunc(patch_total + patches_per_batch - 1, patches_per_batch);

    ggml.impl.assert(patches_per_batch > 0 and batch_size >= 1, "patches_per_batch > 0 && batch_size >= 1");

    const tmp: [*]u8 = @ptrCast(params.wdata.?);
    const nth: i64 = params.nth;
    const ith: i64 = params.ith;

    var batch_i: i64 = 0;
    while (batch_i < batch_n) : (batch_i += 1) {
        const patch_start_batch = batch_i * patches_per_batch;
        const patch_end_batch = @min(patch_start_batch + patches_per_batch, patch_total);
        const patch_n_in_batch = patch_end_batch - patch_start_batch;

        const patch_per_thread = @divTrunc(patch_n_in_batch + nth - 1, nth);
        const patch_start = patch_start_batch + ith * patch_per_thread;
        const patch_end = @min(patch_start + patch_per_thread, patch_end_batch);

        var p = patch_start;
        while (p < patch_end) : (p += 1) {
            const p_in_batch = @rem(p, dst_w * dst_h * dst_d);
            const p_in_depth = @rem(p_in_batch, dst_w * dst_h);
            const batch_idx = @divTrunc(p, dst_w * dst_h * dst_d);
            const dst_z = @divTrunc(p_in_batch, dst_w * dst_h);
            const dst_y = @divTrunc(p_in_depth, dst_w);
            const dst_x = @rem(p_in_depth, dst_w);

            const dst_row: [*]f32 = @as([*]f32, @ptrCast(@alignCast(tmp))) + uz(@rem(p, patches_per_batch) * knl_n_total);

            var ic: i64 = 0;
            while (ic < cc) : (ic += 1) {
                var kz: i64 = 0;
                while (kz < knl_d) : (kz += 1) {
                    var ky: i64 = 0;
                    while (ky < knl_h) : (ky += 1) {
                        var kx: i64 = 0;
                        while (kx < knl_w) : (kx += 1) {
                            const sz = dst_z * s2 + kz * d2 - p2;
                            const sy = dst_y * s1 + ky * d1 - p1;
                            const sx = dst_x * s0 + kx * d0 - p0;

                            const dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            var src_val: f32 = undefined;
                            if (sz < 0 or sz >= src_d or sy < 0 or sy >= src_h or sx < 0 or sx >= src_w) {
                                src_val = 0.0;
                            } else {
                                const cn_idx = batch_idx * cc + ic;
                                const src_ptr: *const f32 = @ptrCast(@alignCast(src_data + uz(sx) * src.nb[0] + uz(sy) * src.nb[1] + uz(sz) * src.nb[2] + uz(cn_idx) * src.nb[3]));
                                src_val = src_ptr.*;
                            }

                            dst_row[uz(dst_idx)] = src_val;
                        }
                    }
                }
            }
        }

        ggml.cpu.ops.common.barrier(params);

        const gemm_output: [*]f32 = @ptrCast(@alignCast(tmp + uz(patches_per_batch * knl_n_total) * @sizeOf(f32)));
        callMulMat(kernel_type, ggml.impl.ggml_c.GGML_TYPE_F32, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml.cpu.ops.common.barrier(params);

        const permute_per_thread = @divTrunc(patch_n_in_batch + nth - 1, nth);
        const permute_start = ith * permute_per_thread;
        const permute_end = @min(permute_start + permute_per_thread, patch_n_in_batch);

        var i = permute_start;
        while (i < permute_end) : (i += 1) {
            const pp = patch_start_batch + i;
            const p_in_batch = @rem(pp, dst_w * dst_h * dst_d);
            const p_in_depth = @rem(p_in_batch, dst_w * dst_h);
            const batch_idx = @divTrunc(pp, dst_w * dst_h * dst_d);
            const dst_z = @divTrunc(p_in_batch, dst_w * dst_h);
            const dst_y = @divTrunc(p_in_depth, dst_w);
            const dst_x = @rem(p_in_depth, dst_w);

            var ioc: i64 = 0;
            while (ioc < oc) : (ioc += 1) {
                const value = gemm_output[uz(i * oc + ioc)];
                const ocn_idx = batch_idx * oc + ioc;
                const dst_ptr: *f32 = @ptrCast(@alignCast(dst_data + uz(dst_x) * dst.nb[0] + uz(dst_y) * dst.nb[1] + uz(dst_z) * dst.nb[2] + uz(ocn_idx) * dst.nb[3]));
                dst_ptr.* = value;
            }
        }
    }
}

/// Ports `ggml_compute_forward_conv_3d` (ops.cpp:7463 @b1b770d90).
pub fn ggml_compute_forward_conv_3d(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);
    conv3dImpl(params, src0, src1, dst, src0.type);
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_conv_transpose_2d

/// Ports `ggml_compute_forward_conv_transpose_2d_impl` (ops.cpp:7472 @b1b770d90).
///
/// Batched over `src1`'s `ne[3]`: each image is permuted into its own slice
/// of the work buffer, and every thread's output channels are computed for
/// every image.
fn convTranspose2d(comptime T: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F16 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32");
    ggml.impl.assert(src1.type == ggml.impl.ggml_c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    ggml.impl.assert(dst.type == ggml.impl.ggml_c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nsrc = l.ne10 * l.ne11 * l.ne12 * l.ne13;

    ggml.impl.assert(l.nb00 == ggml.types.ggml_type_size(src0.type), "nb00 == ggml_type_size(src0->type)");
    ggml.impl.assert(l.nb10 == @sizeOf(f32), "nb10 == sizeof(float)");

    // The permuted source comes first (#31854), in `f32`, and
    // the kernel follows it in its own type.
    const wbytes: [*]u8 = @ptrCast(params.wdata.?);
    const wsrc: [*]f32 = @ptrCast(@alignCast(wbytes));
    const wkernel: [*]T = @ptrCast(@alignCast(wbytes + uz(nsrc) * @sizeOf(f32)));

    if (ith == 0) {
        zeroWork(params);

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            const s1: [*]const u8 = @ptrCast(src1.data.?);

            var j13: i64 = 0;
            while (j13 < l.ne13) : (j13 += 1) {
                const wdata_b = wsrc + uz(j13 * l.ne10 * l.ne11 * l.ne12);
                var j12: i64 = 0;
                while (j12 < l.ne12) : (j12 += 1) {
                    var j11: i64 = 0;
                    while (j11 < l.ne11) : (j11 += 1) {
                        const src: [*]const f32 = @ptrCast(@alignCast(s1 + uz(j13) * l.nb13 + uz(j12) * l.nb12 + uz(j11) * l.nb11));
                        const dst_data = wdata_b + uz(j11 * l.ne10 * l.ne12);
                        var j10: i64 = 0;
                        while (j10 < l.ne10) : (j10 += 1) {
                            dst_data[uz(j10 * l.ne12 + j12)] = src[uz(j10)];
                        }
                    }
                }
            }
        }

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            const wdata = wkernel;
            const s0: [*]const u8 = @ptrCast(src0.data.?);

            var j03: i64 = 0;
            while (j03 < l.ne03) : (j03 += 1) {
                var j02: i64 = 0;
                while (j02 < l.ne02) : (j02 += 1) {
                    const src: [*]const T = @ptrCast(@alignCast(s0 + uz(j03) * l.nb03 + uz(j02) * l.nb02));
                    const dst_data = wdata + uz(j02 * l.ne01 * l.ne00 * l.ne03);
                    var j01: i64 = 0;
                    while (j01 < l.ne01) : (j01 += 1) {
                        var j00: i64 = 0;
                        while (j00 < l.ne00) : (j00 += 1) {
                            dst_data[uz(j01 * l.ne00 * l.ne03 + j00 * l.ne03 + j03)] = src[uz(j01 * l.ne00 + j00)];
                        }
                    }
                }
            }
        }

        zeroDst(dst);
    }
    ggml.cpu.ops.common.barrier(params);

    const stride: i64 = ggml.impl.getOpParamsI32(dst, 0);

    // total patches in dst
    const np = l.ne2;

    // patches per thread
    const dp = @divTrunc(np + nth - 1, nth);

    // patch range for this thread
    const ip0 = dp * ith;
    const ip1 = @min(ip0 + dp, np);

    const dd: [*]u8 = @ptrCast(dst.data.?);

    var j2: i64 = ip0;
    while (j2 < ip1) : (j2 += 1) { // Cout
        const wdata_kernel_c = wkernel + uz(j2 * l.ne01 * l.ne00 * l.ne03);
        var j3: i64 = 0;
        while (j3 < l.ne3) : (j3 += 1) { // batch
            const dst_data: [*]f32 = @ptrCast(@alignCast(dd + uz(j3) * l.nb3 + uz(j2) * l.nb2));
            const wdata_src_b = wsrc + uz(j3 * l.ne10 * l.ne11 * l.ne12);
            var j11: i64 = 0;
            while (j11 < l.ne11) : (j11 += 1) {
                var j10: i64 = 0;
                while (j10 < l.ne10) : (j10 += 1) {
                    const j1n = j11 * l.ne10 * l.ne12 + j10 * l.ne12;
                    var j01: i64 = 0;
                    while (j01 < l.ne01) : (j01 += 1) {
                        var j00: i64 = 0;
                        while (j00 < l.ne00) : (j00 += 1) {
                            var v: f32 = 0;
                            const kp = wdata_kernel_c + uz(j01 * l.ne00 * l.ne03 + j00 * l.ne03);
                            if (T == ggml.impl.ggml_c.ggml_fp16_t) {
                                ggml.cpu.vec.ggml_vec_dot_f16_f32(@intCast(l.ne03), &v, 0, kp, 0, wdata_src_b + uz(j1n), 0, 1);
                            } else {
                                ggml.cpu.vec.ggml_vec_dot_f32(@intCast(l.ne03), &v, 0, kp, 0, wdata_src_b + uz(j1n), 0, 1);
                            }
                            dst_data[uz((j11 * stride + j01) * l.ne0 + j10 * stride + j00)] += v;
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_conv_transpose_2d` (ops.cpp:7578 @b1b770d90).
pub fn ggml_compute_forward_conv_transpose_2d(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F16 => convTranspose2d(ggml.impl.ggml_c.ggml_fp16_t, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F32 => convTranspose2d(f32, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_conv_2d_dw

/// Ports `struct ggml_conv_2d_dw_params` (ops.cpp:7602 @b1b770d90).
///
/// The strides, paddings and dilations are `int` in the C; widened here,
/// since every use multiplies them into an `int64_t` anyway.
const Conv2dDwParams = struct {
    channels: i64,
    batch: i64,
    src_w: i64,
    src_h: i64,
    dst_w: i64,
    dst_h: i64,
    knl_w: i64,
    knl_h: i64,
    stride_x: i64,
    stride_y: i64,
    pad_x: i64,
    pad_y: i64,
    dilation_x: i64,
    dilation_y: i64,
};

/// Ports `ggml_conv_2d_dw_knl_f32` (ops.cpp:7619 @b1b770d90): one kernel
/// element, widened if the kernel is `f16`.
inline fn conv2dDwKnlF32(data: [*]const u8, i: i64, @"type": ggml.impl.ggml_c.ggml_type) f32 {
    if (@"type" == ggml.impl.ggml_c.GGML_TYPE_F16) {
        return ggml.impl.fp16ToFp32(@as([*]const ggml.impl.ggml_c.ggml_fp16_t, @ptrCast(@alignCast(data)))[uz(i)]);
    }
    return @as([*]const f32, @ptrCast(@alignCast(data)))[uz(i)];
}

/// Ports `ggml_compute_forward_conv_2d_dw_cwhn` (ops.cpp:7626 @b1b770d90),
/// the `GGML_SIMD` NEON arm.
///
/// Channels-last. With an `f32` kernel the C takes four channels at a time
/// through `vfmaq_f32(sum, k, s)` and the remainder through `sum += k * s`;
/// both are a fused multiply-add per channel over the same window order, so
/// one fused loop covers every channel. See the file header.
fn conv2dDwCwhn(params: *const ggml.cpu.defs.ComputeParams, src: *const ggml.impl.ggml_c.ggml_tensor, kernel: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, p: Conv2dDwParams) void {
    const cc = p.channels;
    const knl_data: [*]const u8 = @ptrCast(kernel.data.?);
    const knl_type = kernel.type;

    const nth: i64 = params.nth;
    const ith: i64 = params.ith;
    const rows_total = p.dst_h * p.batch;
    const rows_per_thread = @divTrunc(rows_total + nth - 1, nth);
    const row_start = ith * rows_per_thread;
    const row_end = @min(row_start + rows_per_thread, rows_total);

    const sbase: [*]const f32 = @ptrCast(@alignCast(src.data.?));
    const dbase: [*]f32 = @ptrCast(@alignCast(dst.data.?));

    var row = row_start;
    while (row < row_end) : (row += 1) {
        const dst_y = @rem(row, p.dst_h);
        const src_data = sbase + uz(@divTrunc(row, p.dst_h) * p.src_w * p.src_h * cc);
        var dst_x: i64 = 0;
        while (dst_x < p.dst_w) : (dst_x += 1) {
            const dst_data = dbase + uz((row * p.dst_w + dst_x) * cc);
            const src_y_base = dst_y * p.stride_y - p.pad_y;
            const src_x_base = dst_x * p.stride_x - p.pad_x;

            var c_i: i64 = 0;
            while (c_i < cc) : (c_i += 1) {
                var sum: f32 = 0.0;
                var knl_y: i64 = 0;
                while (knl_y < p.knl_h) : (knl_y += 1) {
                    const src_y = src_y_base + knl_y * p.dilation_y;
                    if (src_y < 0 or src_y >= p.src_h) continue;
                    var knl_x: i64 = 0;
                    while (knl_x < p.knl_w) : (knl_x += 1) {
                        const src_x = src_x_base + knl_x * p.dilation_x;
                        if (src_x < 0 or src_x >= p.src_w) continue;
                        // `sum += k * s`, fused -- `vfmaq_f32` in the body,
                        // clang's contraction in the tail.
                        sum = @mulAdd(
                            f32,
                            conv2dDwKnlF32(knl_data, (knl_y * p.knl_w + knl_x) * cc + c_i, knl_type),
                            src_data[uz((src_y * p.src_w + src_x) * cc + c_i)],
                            sum,
                        );
                    }
                }
                dst_data[uz(c_i)] = sum;
            }
        }
    }
}

/// Ports `ggml_compute_forward_conv_2d_dw_whcn` (ops.cpp:7709 @b1b770d90).
///
/// Channels-first, scalar. `sum += k * s` is one expression, so it fuses.
fn conv2dDwWhcn(params: *const ggml.cpu.defs.ComputeParams, src: *const ggml.impl.ggml_c.ggml_tensor, kernel: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor, p: Conv2dDwParams) void {
    const n = p.channels * p.batch;
    const nth: i64 = params.nth;
    const ith: i64 = params.ith;
    const per_thread = @divTrunc(n + nth - 1, nth);
    const start = ith * per_thread;
    const end = @min(start + per_thread, n);
    const knl_base: [*]const u8 = @ptrCast(kernel.data.?);
    const knl_type = kernel.type;

    const sbase: [*]const f32 = @ptrCast(@alignCast(src.data.?));
    const dbase: [*]f32 = @ptrCast(@alignCast(dst.data.?));

    var i = start;
    while (i < end) : (i += 1) {
        const knl_offset = @rem(i, p.channels) * p.knl_w * p.knl_h;
        const src_data = sbase + uz(i * p.src_w * p.src_h);
        const dst_data = dbase + uz(i * p.dst_w * p.dst_h);

        var dst_y: i64 = 0;
        while (dst_y < p.dst_h) : (dst_y += 1) {
            var dst_x: i64 = 0;
            while (dst_x < p.dst_w) : (dst_x += 1) {
                var sum: f32 = 0.0;
                var knl_y: i64 = 0;
                while (knl_y < p.knl_h) : (knl_y += 1) {
                    const src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    if (src_y < 0 or src_y >= p.src_h) continue;
                    var knl_x: i64 = 0;
                    while (knl_x < p.knl_w) : (knl_x += 1) {
                        const src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        if (src_x < 0 or src_x >= p.src_w) continue;
                        // `sum += k * s` is one expression; clang fuses it.
                        sum = @mulAdd(
                            f32,
                            conv2dDwKnlF32(knl_base, knl_offset + knl_y * p.knl_w + knl_x, knl_type),
                            src_data[uz(src_y * p.src_w + src_x)],
                            sum,
                        );
                    }
                }
                dst_data[uz(dst_y * p.dst_w + dst_x)] = sum;
            }
        }
    }
}

/// Ports `ggml_compute_forward_conv_2d_dw` (ops.cpp:7752 @b1b770d90).
pub fn ggml_compute_forward_conv_2d_dw(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const kernel = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);
    const p: Conv2dDwParams = .{
        .channels = src.ne[2],
        .batch = src.ne[3],
        .src_w = src.ne[0],
        .src_h = src.ne[1],
        .dst_w = dst.ne[0],
        .dst_h = dst.ne[1],
        .knl_w = kernel.ne[0],
        .knl_h = kernel.ne[1],
        .stride_x = dst.op_params[0],
        .stride_y = dst.op_params[1],
        .pad_x = dst.op_params[2],
        .pad_y = dst.op_params[3],
        .dilation_x = dst.op_params[4],
        .dilation_y = dst.op_params[5],
    };

    ggml.impl.assert(kernel.type == ggml.impl.ggml_c.GGML_TYPE_F32 or kernel.type == ggml.impl.ggml_c.GGML_TYPE_F16, "kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16");
    ggml.impl.assert(kernel.ne[3] == p.channels, "kernel->ne[3] == p.channels");
    ggml.impl.assert(dst.ne[3] == p.batch, "dst->ne[3] == p.batch");

    if (ggml.types.ggml_is_contiguous(src)) {
        conv2dDwWhcn(params, src, kernel, dst, p);
    } else if (ggml.types.ggml_is_contiguous_channels(src)) {
        ggml.impl.assert(kernel.nb[0] >= kernel.nb[2] and kernel.nb[1] >= kernel.nb[0], "kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]");
        conv2dDwCwhn(params, src, kernel, dst, p);
    } else {
        ggml.impl.abort("non-contiguous memory layout not supported");
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "conv_2d_dw reads an f16 kernel element widened, and an f32 one as is" {
    const k32 = [_]f32{ 1.5, -2.25 };
    try std.testing.expectEqual(@as(f32, -2.25), conv2dDwKnlF32(@ptrCast(&k32), 1, ggml.impl.ggml_c.GGML_TYPE_F32));

    const k16 = [_]ggml.impl.ggml_c.ggml_fp16_t{ ggml.impl.fp32ToFp16(0.5), ggml.impl.fp32ToFp16(3.0) };
    try std.testing.expectEqual(@as(f32, 3.0), conv2dDwKnlF32(@ptrCast(&k16), 1, ggml.impl.ggml_c.GGML_TYPE_F16));
}
