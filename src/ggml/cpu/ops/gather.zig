//! `get_rows`, `get_rows_back`, `set_rows` and `diag`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v1.0.0 (`b1b770d90`).
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
const ggml = @import("../../module.zig");

/// Typed pointer at a byte offset; see `common.ptr`.
/// Stride as `i64`; see `common.sz`.

// -----------------------------------------------------------------------------
// get_rows

/// How `getRows` turns one source row into `f32`.
const RowKind = enum { q, f16, bf16, f32 };

/// Ports `ggml_compute_forward_get_rows_q`, `ggml_compute_forward_get_rows_f16`,
/// `ggml_compute_forward_get_rows_bf16` and `ggml_compute_forward_get_rows_f32`
/// (ops.cpp:5187, 5231, 5272, 5313 @b1b770d90).
///
/// The four are the same loop with a different row conversion, so they are
/// one function over a comptime kind. The `f32` variant also serves `I32`
/// tables: it copies four-byte elements without interpreting them.
fn getRows(comptime kind: RowKind, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const nc = l.ne00;
    const nr: i64 = ggml.impl.ggml_c.ggml_nelements(src1);

    std.debug.assert(l.ne0 == nc);
    std.debug.assert(l.ne02 == l.ne11);
    std.debug.assert(l.nb00 == switch (kind) {
        .q => ggml.impl.ggml_c.ggml_type_size(src0.type),
        .f16 => @sizeOf(ggml.impl.ggml_c.ggml_fp16_t),
        .bf16 => @sizeOf(ggml.impl.ggml_c.ggml_bf16_t),
        .f32 => @sizeOf(f32),
    });
    std.debug.assert(ggml.impl.ggml_c.ggml_nrows(dst) == nr);

    const dequantize_row_q = if (kind == .q) ggml.impl.ggml_c.ggml_get_type_traits(src0.type).*.to_float.? else {};

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
        const j01: i64 = ggml.cpu.ops.common.ptr(i32, src1.data, j10 * ggml.cpu.ops.common.sz(l.nb10) + j11 * ggml.cpu.ops.common.sz(l.nb11) + j12 * ggml.cpu.ops.common.sz(l.nb12))[0];

        ggml.impl.assert(j01 >= 0 and j01 < l.ne01, "i01 >= 0 && i01 < ne01");

        const src_off = j01 * ggml.cpu.ops.common.sz(l.nb01) + j11 * ggml.cpu.ops.common.sz(l.nb02) + j12 * ggml.cpu.ops.common.sz(l.nb03);
        const dst_row = ggml.cpu.ops.common.ptr(f32, dst.data, j10 * ggml.cpu.ops.common.sz(l.nb1) + j11 * ggml.cpu.ops.common.sz(l.nb2) + j12 * ggml.cpu.ops.common.sz(l.nb3));

        switch (kind) {
            .q => dequantize_row_q(ggml.cpu.ops.common.ptr(u8, src0.data, src_off), dst_row, nc),
            .f16 => ggml.impl.ggml_c.ggml_cpu_fp16_to_fp32(ggml.cpu.ops.common.ptr(ggml.impl.ggml_c.ggml_fp16_t, src0.data, src_off), dst_row, nc),
            .bf16 => ggml.impl.ggml_c.ggml_cpu_bf16_to_fp32(ggml.cpu.ops.common.ptr(ggml.impl.ggml_c.ggml_bf16_t, src0.data, src_off), dst_row, nc),
            .f32 => ggml.cpu.ops.vecinline.cpy_f32(nc, dst_row, ggml.cpu.ops.common.ptr(f32, src0.data, src_off)),
        }
    }
}

/// Ports `ggml_compute_forward_get_rows` (ops.cpp:5354 @b1b770d90).
pub fn ggml_compute_forward_get_rows(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_Q1_0,
        ggml.impl.ggml_c.GGML_TYPE_Q2_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_0,
        ggml.impl.ggml_c.GGML_TYPE_Q4_1,
        ggml.impl.ggml_c.GGML_TYPE_Q5_0,
        ggml.impl.ggml_c.GGML_TYPE_Q5_1,
        ggml.impl.ggml_c.GGML_TYPE_Q8_0,
        ggml.impl.ggml_c.GGML_TYPE_Q8_1,
        ggml.impl.ggml_c.GGML_TYPE_MXFP4,
        ggml.impl.ggml_c.GGML_TYPE_NVFP4,
        ggml.impl.ggml_c.GGML_TYPE_F8_E4M3,
        ggml.impl.ggml_c.GGML_TYPE_Q2_K,
        ggml.impl.ggml_c.GGML_TYPE_Q3_K,
        ggml.impl.ggml_c.GGML_TYPE_Q4_K,
        ggml.impl.ggml_c.GGML_TYPE_Q5_K,
        ggml.impl.ggml_c.GGML_TYPE_Q6_K,
        ggml.impl.ggml_c.GGML_TYPE_TQ1_0,
        ggml.impl.ggml_c.GGML_TYPE_TQ2_0,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_XXS,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ1_M,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_NL,
        ggml.impl.ggml_c.GGML_TYPE_IQ4_XS,
        ggml.impl.ggml_c.GGML_TYPE_IQ3_S,
        ggml.impl.ggml_c.GGML_TYPE_IQ2_S,
        => getRows(.q, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F16 => getRows(.f16, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_BF16 => getRows(.bf16, params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F32, ggml.impl.ggml_c.GGML_TYPE_I32 => getRows(.f32, params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// set_rows

/// Ports `ggml_compute_forward_set_rows_impl` (ops.cpp:5430 @b1b770d90).
///
/// Parameters:
/// - `SrcT`: `f32` or `ggml_fp16_t`, the source row type.
/// - `IdxT`: `i64` or `i32`, the index type in `src1`.
fn setRowsImpl(comptime SrcT: type, comptime IdxT: type, params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    const l = ggml.cpu.defs.BinaryLocals.of(src0, src1, dst);

    const nc = l.ne00;
    const nr = l.ne01;

    std.debug.assert(l.ne0 == nc);
    std.debug.assert(l.ne2 == l.ne02);
    std.debug.assert(l.ne3 == l.ne03);
    ggml.impl.assert(src0.type == ggml.impl.ggml_c.GGML_TYPE_F32 or src0.type == ggml.impl.ggml_c.GGML_TYPE_F16, "src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16");
    std.debug.assert(@rem(l.ne02, l.ne11) == 0);
    std.debug.assert(@rem(l.ne03, l.ne12) == 0);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const rs = ggml.impl.ggml_c.ggml_row_size(src0.type, nc);

    const from_float = ggml.impl.ggml_c.ggml_get_type_traits_cpu(dst.type).*.from_float;

    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            var i: i64 = ir0;
            while (i < ir1) : (i += 1) {
                const j12 = @rem(j03, l.ne12);
                const j11 = @rem(j02, l.ne11);
                const j10 = i;

                const j1: i64 = ggml.cpu.ops.common.ptr(IdxT, src1.data, j10 * ggml.cpu.ops.common.sz(l.nb10) + j11 * ggml.cpu.ops.common.sz(l.nb11) + j12 * ggml.cpu.ops.common.sz(l.nb12))[0];

                ggml.impl.assert(j1 >= 0 and j1 < l.ne1, "i1 >= 0 && i1 < ne1");

                const src_off = i * ggml.cpu.ops.common.sz(l.nb01) + j02 * ggml.cpu.ops.common.sz(l.nb02) + j03 * ggml.cpu.ops.common.sz(l.nb03);
                const dst_row = ggml.cpu.ops.common.ptr(u8, dst.data, j1 * ggml.cpu.ops.common.sz(l.nb1) + j02 * ggml.cpu.ops.common.sz(l.nb2) + j03 * ggml.cpu.ops.common.sz(l.nb3));

                if (SrcT == f32) {
                    from_float.?(ggml.cpu.ops.common.ptr(f32, src0.data, src_off), dst_row, nc);
                } else if (SrcT == ggml.impl.ggml_c.ggml_fp16_t) {
                    if (dst.type == ggml.impl.ggml_c.GGML_TYPE_F16) {
                        @memcpy(dst_row[0..rs], ggml.cpu.ops.common.ptr(u8, src0.data, src_off)[0..rs]);
                    } else {
                        const wbase: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
                        const wdata = wbase + (@as(usize, @intCast(nc)) + ggml.cpu.ops.common.cache_line_size_f32) * @as(usize, @intCast(ith));
                        ggml.impl.ggml_c.ggml_fp16_to_fp32_row(ggml.cpu.ops.common.ptr(ggml.impl.ggml_c.ggml_fp16_t, src0.data, src_off), wdata, nc);
                        from_float.?(wdata, dst_row, nc);
                    }
                } else {
                    @compileError("src0 type not supported");
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_set_rows` (ops.cpp:5500 @b1b770d90).
pub fn ggml_compute_forward_set_rows(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => {
            if (src1.type == ggml.impl.ggml_c.GGML_TYPE_I64) {
                setRowsImpl(f32, i64, params, dst);
            } else if (src1.type == ggml.impl.ggml_c.GGML_TYPE_I32) {
                setRowsImpl(f32, i32, params, dst);
            } else {
                ggml.impl.abort("src1->type not supported");
            }
        },
        ggml.impl.ggml_c.GGML_TYPE_F16 => {
            if (src1.type == ggml.impl.ggml_c.GGML_TYPE_I64) {
                setRowsImpl(ggml.impl.ggml_c.ggml_fp16_t, i64, params, dst);
            } else if (src1.type == ggml.impl.ggml_c.GGML_TYPE_I32) {
                setRowsImpl(ggml.impl.ggml_c.ggml_fp16_t, i32, params, dst);
            } else {
                ggml.impl.abort("src1->type not supported");
            }
        },
        else => ggml.impl.abort("src0->type not supported"),
    }
}

// -----------------------------------------------------------------------------
// get_rows_back

/// Ports `ggml_compute_forward_get_rows_back_f32_f16` (ops.cpp:5537 @b1b770d90).
fn getRowsBackF32F16(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    if (params.ith != 0) return;

    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    @memset(ggml.cpu.ops.common.ptr(u8, dst.data, 0)[0..ggml.impl.ggml_c.ggml_nbytes(dst)], 0);

    const nc: i64 = src0.ne[0];
    const nr: i64 = ggml.impl.ggml_c.ggml_nelements(src1);

    ggml.impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
    ggml.impl.assert(src0.nb[0] == @sizeOf(ggml.impl.ggml_c.ggml_fp16_t), "src0->nb[0] == sizeof(ggml_fp16_t)");

    const idx = ggml.cpu.ops.common.ptr(i32, src1.data, 0);

    var i: i64 = 0;
    while (i < nr) : (i += 1) {
        const r: i64 = idx[@intCast(i)];

        ggml.impl.assert(r >= 0 and r < dst.ne[1], "r >= 0 && r < dst->ne[1]");

        const srow = ggml.cpu.ops.common.ptr(ggml.impl.ggml_c.ggml_fp16_t, src0.data, i * ggml.cpu.ops.common.sz(src0.nb[1]));
        const drow = ggml.cpu.ops.common.ptr(f32, dst.data, r * ggml.cpu.ops.common.sz(dst.nb[1]));

        var j: usize = 0;
        while (j < @as(usize, @intCast(nc))) : (j += 1) {
            drow[j] += ggml.impl.fp16ToFp32(srow[j]);
        }
    }
}

/// Ports `ggml_compute_forward_get_rows_back_f32` (ops.cpp:5572 @b1b770d90).
fn getRowsBackF32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);
    const src1 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[1]);

    if (params.ith != 0) return;

    ggml.impl.assert(ggml.impl.ggml_c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    @memset(ggml.cpu.ops.common.ptr(u8, dst.data, 0)[0..ggml.impl.ggml_c.ggml_nbytes(dst)], 0);

    const nc: i64 = src0.ne[0];
    const nr: i64 = ggml.impl.ggml_c.ggml_nelements(src1);

    ggml.impl.assert(dst.ne[0] == nc, "dst->ne[0] == nc");
    ggml.impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    const idx = ggml.cpu.ops.common.ptr(i32, src1.data, 0);

    var i: i64 = 0;
    while (i < nr) : (i += 1) {
        const r: i64 = idx[@intCast(i)];

        ggml.impl.assert(r >= 0 and r < dst.ne[1], "r >= 0 && r < dst->ne[1]");

        const drow = ggml.cpu.ops.common.ptr(f32, dst.data, r * ggml.cpu.ops.common.sz(dst.nb[1]));
        ggml.cpu.ops.vecinline.add_f32(nc, drow, drow, ggml.cpu.ops.common.ptr(f32, src0.data, i * ggml.cpu.ops.common.sz(src0.nb[1])));
    }
}

/// Ports `ggml_compute_forward_get_rows_back` (ops.cpp:5607 @b1b770d90).
pub fn ggml_compute_forward_get_rows_back(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F16 => getRowsBackF32F16(params, dst),
        ggml.impl.ggml_c.GGML_TYPE_F32 => getRowsBackF32(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// diag

/// Ports `ggml_compute_forward_diag_f32` (ops.cpp:5649 @b1b770d90).
fn diagF32(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    if (params.ith != 0) return;

    // TODO: handle transposed/permuted matrices

    const l = ggml.cpu.defs.UnaryLocals.of(src0, dst);

    ggml.impl.assert(l.ne00 == l.ne0, "ne00 == ne0");
    ggml.impl.assert(l.ne00 == l.ne1, "ne00 == ne1");
    ggml.impl.assert(l.ne01 == 1, "ne01 == 1");
    ggml.impl.assert(l.ne02 == l.ne2, "ne02 == ne2");
    ggml.impl.assert(l.ne03 == l.ne3, "ne03 == ne3");

    ggml.impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");
    ggml.impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

    var j3: i64 = 0;
    while (j3 < l.ne3) : (j3 += 1) {
        var j2: i64 = 0;
        while (j2 < l.ne2) : (j2 += 1) {
            var j1: i64 = 0;
            while (j1 < l.ne1) : (j1 += 1) {
                const d = ggml.cpu.ops.common.ptr(f32, dst.data, j3 * ggml.cpu.ops.common.sz(l.nb3) + j2 * ggml.cpu.ops.common.sz(l.nb2) + j1 * ggml.cpu.ops.common.sz(l.nb1));
                const sp = ggml.cpu.ops.common.ptr(f32, src0.data, j3 * ggml.cpu.ops.common.sz(l.nb03) + j2 * ggml.cpu.ops.common.sz(l.nb02));
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

/// Ports `ggml_compute_forward_diag` (ops.cpp:5689 @b1b770d90).
pub fn ggml_compute_forward_diag(params: *const ggml.cpu.defs.ComputeParams, dst: *ggml.impl.ggml_c.ggml_tensor) callconv(.c) void {
    const src0 = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst.src[0]);

    switch (src0.type) {
        ggml.impl.ggml_c.GGML_TYPE_F32 => diagF32(params, dst),
        else => ggml.impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
