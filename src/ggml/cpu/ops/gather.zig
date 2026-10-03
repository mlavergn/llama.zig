//! `get_rows`, `get_rows_back`, `set_rows`, `diag`, `diag_mask_inf` and
//! `diag_mask_zero`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # `assert` versus `GGML_ASSERT`
//!
//! The C mixes the two. `GGML_ASSERT` stays live in every build and is
//! `impl.assert` here; plain `assert` is compiled out under `NDEBUG`, which the
//! release build defines, so it is `std.debug.assert` — the same split
//! `cpu/mulmat.zig` makes.
//!
//! # Loop index names
//!
//! `i1`, `i2`, `i3`, `i01`, `i10`, `i11`, `i12` and the rest are Zig integer
//! type names. Renamed `j1`, `j2`, `j3`, `j01`, `j10`, `j11`, `j12`, digit for
//! digit — the `cpu/mulmat.zig` convention. Do not renumber them.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const threading = @import("../threading.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// Typed pointer at a byte offset; see `common.ptr`.
const at = common.ptr;

/// Stride as `i64`; see `common.sz`.
const s = common.sz;

// -----------------------------------------------------------------------------
// get_rows

/// How `getRows` turns one source row into `f32`.
const RowKind = enum { q, f16, bf16, f32 };

/// Ports `ggml_compute_forward_get_rows_q`, `ggml_compute_forward_get_rows_f16`,
/// `ggml_compute_forward_get_rows_bf16` and `ggml_compute_forward_get_rows_f32`
/// (ops.cpp:4846, 4890, 4931, 4972 @c1d0e7a00).
///
/// The four are the same loop with a different row conversion, so they are
/// one function over a comptime kind. The `f32` variant also serves `I32`
/// tables: it copies four-byte elements without interpreting them.
fn getRows(comptime kind: RowKind, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = common.BinaryLocals.of(src0, src1, dst);

    const nc = l.ne00;
    const nr: i64 = c.ggml_nelements(src1);

    std.debug.assert(l.ne0 == nc);
    std.debug.assert(l.ne02 == l.ne11);
    std.debug.assert(l.nb00 == switch (kind) {
        .q => c.ggml_type_size(src0.type),
        .f16 => @sizeOf(c.ggml_fp16_t),
        .bf16 => @sizeOf(c.ggml_bf16_t),
        .f32 => @sizeOf(f32),
    });
    std.debug.assert(c.ggml_nrows(dst) == nr);

    const dequantize_row_q = if (kind == .q) c.ggml_get_type_traits(src0.type).*.to_float.? else {};

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    var i: i64 = ir0;
    while (i < ir1) : (i += 1) {
        const j12 = @divTrunc(i, l.ne11 * l.ne10);
        const j11 = @divTrunc(i - j12 * l.ne11 * l.ne10, l.ne10);
        const j10 = i - j12 * l.ne11 * l.ne10 - j11 * l.ne10;
        const j01: i64 = at(i32, src1.data, j10 * s(l.nb10) + j11 * s(l.nb11) + j12 * s(l.nb12))[0];

        impl.assert(j01 >= 0 and j01 < l.ne01, "i01 >= 0 && i01 < ne01");

        const src_off = j01 * s(l.nb01) + j11 * s(l.nb02) + j12 * s(l.nb03);
        const dst_row = at(f32, dst.data, j10 * s(l.nb1) + j11 * s(l.nb2) + j12 * s(l.nb3));

        switch (kind) {
            .q => dequantize_row_q(at(u8, src0.data, src_off), dst_row, nc),
            .f16 => c.ggml_cpu_fp16_to_fp32(at(c.ggml_fp16_t, src0.data, src_off), dst_row, nc),
            .bf16 => c.ggml_cpu_bf16_to_fp32(at(c.ggml_bf16_t, src0.data, src_off), dst_row, nc),
            .f32 => vec.cpy_f32(nc, dst_row, at(f32, src0.data, src_off)),
        }
    }
}

/// Ports `ggml_compute_forward_get_rows` (ops.cpp:5013 @c1d0e7a00).
pub export fn ggml_compute_forward_get_rows(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_Q1_0,
        c.GGML_TYPE_Q2_0,
        c.GGML_TYPE_Q4_0,
        c.GGML_TYPE_Q4_1,
        c.GGML_TYPE_Q5_0,
        c.GGML_TYPE_Q5_1,
        c.GGML_TYPE_Q8_0,
        c.GGML_TYPE_Q8_1,
        c.GGML_TYPE_MXFP4,
        c.GGML_TYPE_NVFP4,
        c.GGML_TYPE_Q2_K,
        c.GGML_TYPE_Q3_K,
        c.GGML_TYPE_Q4_K,
        c.GGML_TYPE_Q5_K,
        c.GGML_TYPE_Q6_K,
        c.GGML_TYPE_TQ1_0,
        c.GGML_TYPE_TQ2_0,
        c.GGML_TYPE_IQ2_XXS,
        c.GGML_TYPE_IQ2_XS,
        c.GGML_TYPE_IQ3_XXS,
        c.GGML_TYPE_IQ1_S,
        c.GGML_TYPE_IQ1_M,
        c.GGML_TYPE_IQ4_NL,
        c.GGML_TYPE_IQ4_XS,
        c.GGML_TYPE_IQ3_S,
        c.GGML_TYPE_IQ2_S,
        => getRows(.q, params, dst),
        c.GGML_TYPE_F16 => getRows(.f16, params, dst),
        c.GGML_TYPE_BF16 => getRows(.bf16, params, dst),
        c.GGML_TYPE_F32, c.GGML_TYPE_I32 => getRows(.f32, params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// set_rows

/// Ports `ggml_compute_forward_set_rows_impl` (ops.cpp:5088 @c1d0e7a00).
///
/// Parameters:
/// - `SrcT`: `f32` or `ggml_fp16_t`, the source row type.
/// - `IdxT`: `i64` or `i32`, the index type in `src1`.
fn setRowsImpl(comptime SrcT: type, comptime IdxT: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = common.BinaryLocals.of(src0, src1, dst);

    const nc = l.ne00;
    const nr = l.ne01;

    std.debug.assert(l.ne0 == nc);
    std.debug.assert(l.ne2 == l.ne02);
    std.debug.assert(l.ne3 == l.ne03);
    impl.assert(src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    std.debug.assert(@rem(l.ne02, l.ne11) == 0);
    std.debug.assert(@rem(l.ne03, l.ne12) == 0);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const rs = c.ggml_row_size(src0.type, nc);

    const from_float = c.ggml_get_type_traits_cpu(dst.type).*.from_float;

    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var i: i64 = ir0;
            while (i < ir1) : (i += 1) {
                const j12 = @rem(j03, l.ne12);
                const j11 = @rem(j02, l.ne11);
                const j10 = i;

                const j1: i64 = at(IdxT, src1.data, j10 * s(l.nb10) + j11 * s(l.nb11) + j12 * s(l.nb12))[0];

                impl.assert(j1 >= 0 and j1 < l.ne1, "i1 >= 0 && i1 < ne1");

                const src_off = i * s(l.nb01) + j02 * s(l.nb02) + j03 * s(l.nb03);
                const dst_row = at(u8, dst.data, j1 * s(l.nb1) + j02 * s(l.nb2) + j03 * s(l.nb3));

                if (SrcT == f32) {
                    from_float.?(at(f32, src0.data, src_off), dst_row, nc);
                } else if (SrcT == c.ggml_fp16_t) {
                    if (dst.type == c.GGML_TYPE_F16) {
                        @memcpy(dst_row[0..rs], at(u8, src0.data, src_off)[0..rs]);
                    } else {
                        const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
                        const wdata = wbase + (@as(usize, @intCast(nc)) + common.cache_line_size_f32) * @as(usize, @intCast(ith));
                        c.ggml_fp16_to_fp32_row(at(c.ggml_fp16_t, src0.data, src_off), wdata, nc);
                        from_float.?(wdata, dst_row, nc);
                    }
                } else {
                    @compileError("src0 type not supported");
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_set_rows` (ops.cpp:5158 @c1d0e7a00).
pub export fn ggml_compute_forward_set_rows(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => {
            if (src1.type == c.GGML_TYPE_I64) {
                setRowsImpl(f32, i64, params, dst);
            } else if (src1.type == c.GGML_TYPE_I32) {
                setRowsImpl(f32, i32, params, dst);
            } else {
                impl.abort("src1->type not supported");
            }
        },
        c.GGML_TYPE_F16 => {
            if (src1.type == c.GGML_TYPE_I64) {
                setRowsImpl(c.ggml_fp16_t, i64, params, dst);
            } else if (src1.type == c.GGML_TYPE_I32) {
                setRowsImpl(c.ggml_fp16_t, i32, params, dst);
            } else {
                impl.abort("src1->type not supported");
            }
        },
        else => impl.abort("src0->type not supported"),
    }
}

// -----------------------------------------------------------------------------
// get_rows_back

/// Ports `ggml_compute_forward_get_rows_back_f32_f16` (ops.cpp:5195 @c1d0e7a00).
fn getRowsBackF32F16(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    @memset(at(u8, dst.data, 0)[0..c.ggml_nbytes(dst)], 0);

    const nc: i64 = src0.ne[0];
    const nr: i64 = c.ggml_nelements(src1);

    impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
    impl.assert(src0.nb[0] == @sizeOf(c.ggml_fp16_t), "src0->nb[0] == sizeof(ggml_fp16_t)");

    const idx = at(i32, src1.data, 0);

    var i: i64 = 0;
    while (i < nr) : (i += 1) {
        const r: i64 = idx[@intCast(i)];

        const srow = at(c.ggml_fp16_t, src0.data, i * s(src0.nb[1]));
        const drow = at(f32, dst.data, r * s(dst.nb[1]));

        var j: usize = 0;
        while (j < @as(usize, @intCast(nc))) : (j += 1) {
            drow[j] += impl.fp16ToFp32(srow[j]);
        }
    }
}

/// Ports `ggml_compute_forward_get_rows_back_f32` (ops.cpp:5228 @c1d0e7a00).
fn getRowsBackF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    @memset(at(u8, dst.data, 0)[0..c.ggml_nbytes(dst)], 0);

    const nc: i64 = src0.ne[0];
    const nr: i64 = c.ggml_nelements(src1);

    impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    const idx = at(i32, src1.data, 0);

    var i: i64 = 0;
    while (i < nr) : (i += 1) {
        const r: i64 = idx[@intCast(i)];

        const drow = at(f32, dst.data, r * s(dst.nb[1]));
        vec.add_f32(nc, drow, drow, at(f32, src0.data, i * s(src0.nb[1])));
    }
}

/// Ports `ggml_compute_forward_get_rows_back` (ops.cpp:5261 @c1d0e7a00).
pub export fn ggml_compute_forward_get_rows_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F16 => getRowsBackF32F16(params, dst),
        c.GGML_TYPE_F32 => getRowsBackF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// diag

/// Ports `ggml_compute_forward_diag_f32` (ops.cpp:5303 @c1d0e7a00).
fn diagF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    // TODO: handle transposed/permuted matrices

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.ne00 == l.ne0, "ne00 == ne0");
    impl.assert(l.ne00 == l.ne1, "ne00 == ne1");
    impl.assert(l.ne01 == 1, "ne01 == 1");
    impl.assert(l.ne02 == l.ne2, "ne02 == ne2");
    impl.assert(l.ne03 == l.ne3, "ne03 == ne3");

    impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");
    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

    var j3: i64 = 0;
    while (j3 < l.ne3) : (j3 += 1) {
        var j2: i64 = 0;
        while (j2 < l.ne2) : (j2 += 1) {
            var j1: i64 = 0;
            while (j1 < l.ne1) : (j1 += 1) {
                const d = at(f32, dst.data, j3 * s(l.nb3) + j2 * s(l.nb2) + j1 * s(l.nb1));
                const sp = at(f32, src0.data, j3 * s(l.nb03) + j2 * s(l.nb02));
                const k: usize = @intCast(j1);
                var j0: usize = 0;
                while (j0 < k) : (j0 += 1) d[j0] = 0;
                d[k] = sp[k];
                j0 = k + 1;
                while (j0 < @as(usize, @intCast(l.ne0))) : (j0 += 1) d[j0] = 0;
            }
        }
    }
}

/// Ports `ggml_compute_forward_diag` (ops.cpp:5343 @c1d0e7a00).
pub export fn ggml_compute_forward_diag(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => diagF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// diag_mask

/// Ports `ggml_compute_forward_diag_mask_f32` (ops.cpp:5363 @c1d0e7a00).
///
/// Parameters:
/// - `value`: what the masked entries are set to — `-inf` or `0`.
fn diagMaskF32(params: *const ComputeParams, dst: *Tensor, value: f32) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const n_past: i64 = impl.getOpParamsI32(dst, 0);
    const inplace = src0.data == dst.data;

    impl.assert(n_past >= 0, "n_past >= 0");

    if (!inplace) {
        if (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            impl.assert(c.ggml_nelements(dst) == c.ggml_nelements(src0), "ggml_nelements(dst) == ggml_nelements(src0)");
            impl.assert(c.ggml_is_contiguous(dst) and c.ggml_is_contiguous(src0), "ggml_is_contiguous(dst) && ggml_is_contiguous(src0)");
            const nbytes = c.ggml_nbytes(dst);
            @memcpy(at(u8, dst.data, 0)[0..nbytes], at(u8, src0.data, 0)[0..nbytes]);
        }
        threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
    }

    // TODO: handle transposed/permuted matrices

    const n: i64 = c.ggml_nrows(src0);
    const nc: i64 = src0.ne[0];
    const nr: i64 = src0.ne[1];
    const nz = @divTrunc(n, nr);

    impl.assert(dst.nb[0] == @sizeOf(f32), "dst->nb[0] == sizeof(float)");
    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    var k: i64 = 0;
    while (k < nz) : (k += 1) {
        var j: i64 = ith;
        while (j < nr) : (j += nth) {
            var i: i64 = n_past;
            while (i < nc) : (i += 1) {
                if (i > n_past + j) {
                    at(f32, dst.data, k * s(dst.nb[2]) + j * s(dst.nb[1]) + i * s(dst.nb[0]))[0] = value;
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_diag_mask_inf` (ops.cpp:5413 @c1d0e7a00).
pub export fn ggml_compute_forward_diag_mask_inf(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => diagMaskF32(params, dst, -std.math.inf(f32)),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_diag_mask_zero` (ops.cpp:5431 @c1d0e7a00).
pub export fn ggml_compute_forward_diag_mask_zero(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => diagMaskF32(params, dst, 0),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
