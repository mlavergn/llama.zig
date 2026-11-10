//! `repeat`, `repeat_back` and `concat`: tiling a tensor, summing the tiles
//! back, and joining two tensors along one dimension.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.9.0 (`cd959bae9`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # These are byte moves, typed only by width
//!
//! `repeat` sends F16, BF16 and I16 through its `_f16` kernel and F32 and I32
//! through its `_f32` one. Nothing is converted, so the element types below
//! are chosen by width alone — `u16` where the C says `ggml_fp16_t`, `f32`
//! where it says `float` — and a value is only ever loaded and stored.
//! `concat` has had one body for every type since v0.8.0, which moves rows,
//! or elements of a strided source, by `ggml_type_size`.
//!
//! # Loop index names
//!
//! `i0`…`i3`, `i00`…`i13` are Zig integer type names. Renamed `j0`…`j3`,
//! `j00`…`j13`, digit for digit — the `cpu/mulmat.zig` convention. Do not
//! renumber them.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// Byte offset `i*nb`; see `common.byteOff`.
const at = common.byteOff;

/// Ports `ggml_compute_forward_repeat_f32` (ops.cpp:1728 @cd959bae9) and
/// `ggml_compute_forward_repeat_f16` (ops.cpp:1772 @cd959bae9).
///
/// The two are the same loop nest over a different element width; the `f16`
/// one inlines the row copy where the `f32` one calls `ggml_vec_cpy_f32`. Both
/// are a plain element copy, so one comptime function covers them.
fn repeatT(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_can_repeat(src0, dst), "ggml_can_repeat(src0, dst)");

    const l = common.UnaryLocals.of(src0, dst);

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const nr0: i64 = @divTrunc(l.ne0, l.ne00);
    const nr1: i64 = @divTrunc(l.ne1, l.ne01);
    const nr2: i64 = @divTrunc(l.ne2, l.ne02);
    const nr3: i64 = @divTrunc(l.ne3, l.ne03);

    // TODO: support for transposed / permuted tensors
    impl.assert(l.nb0 == @sizeOf(T), "nb0 == sizeof(T)");
    impl.assert(l.nb00 == @sizeOf(T), "nb00 == sizeof(T)");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    // TODO: maybe this is not optimal?
    var j3: i64 = 0;
    while (j3 < nr3) : (j3 += 1) {
        var k3: i64 = 0;
        while (k3 < l.ne03) : (k3 += 1) {
            var j2: i64 = 0;
            while (j2 < nr2) : (j2 += 1) {
                var k2: i64 = 0;
                while (k2 < l.ne02) : (k2 += 1) {
                    var j1: i64 = 0;
                    while (j1 < nr1) : (j1 += 1) {
                        var k1: i64 = 0;
                        while (k1 < l.ne01) : (k1 += 1) {
                            var j0: i64 = 0;
                            while (j0 < nr0) : (j0 += 1) {
                                const y: [*]T = @ptrCast(@alignCast(dd + at(j3 * l.ne03 + k3, l.nb3) + at(j2 * l.ne02 + k2, l.nb2) + at(j1 * l.ne01 + k1, l.nb1) + at(j0 * l.ne00, l.nb0)));
                                const x: [*]const T = @ptrCast(@alignCast(s0 + at(k3, l.nb03) + at(k2, l.nb02) + at(k1, l.nb01)));
                                if (T == f32) {
                                    vec.cpy_f32(l.ne00, y, x);
                                } else {
                                    // ggml_vec_cpy_f16(ne00, y, x)
                                    var i: usize = 0;
                                    while (i < @as(usize, @intCast(l.ne00))) : (i += 1) y[i] = x[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_repeat` (ops.cpp:1819 @cd959bae9).
pub export fn ggml_compute_forward_repeat(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F16, c.GGML_TYPE_BF16, c.GGML_TYPE_I16 => repeatT(c.ggml_fp16_t, params, dst),
        c.GGML_TYPE_F32, c.GGML_TYPE_I32 => repeatT(f32, params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_repeat_back_f32` (ops.cpp:1852 @cd959bae9).
fn repeatBackF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (params.ith != 0) return;

    impl.assert(c.ggml_can_repeat(dst, src0), "ggml_can_repeat(dst, src0)");

    const l = common.UnaryLocals.of(src0, dst);

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const nr0: i64 = @divTrunc(l.ne00, l.ne0);
    const nr1: i64 = @divTrunc(l.ne01, l.ne1);
    const nr2: i64 = @divTrunc(l.ne02, l.ne2);
    const nr3: i64 = @divTrunc(l.ne03, l.ne3);

    // TODO: support for transposed / permuted tensors
    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    if (c.ggml_is_contiguous(dst)) {
        vec.set_f32(l.ne0 * l.ne1 * l.ne2 * l.ne3, @ptrCast(@alignCast(dd)), 0);
    } else {
        var k3: i64 = 0;
        while (k3 < l.ne3) : (k3 += 1) {
            var k2: i64 = 0;
            while (k2 < l.ne2) : (k2 += 1) {
                var k1: i64 = 0;
                while (k1 < l.ne1) : (k1 += 1) {
                    vec.set_f32(l.ne0, @ptrCast(@alignCast(dd + at(k1, l.nb1) + at(k2, l.nb2) + at(k3, l.nb3))), 0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    var j3: i64 = 0;
    while (j3 < nr3) : (j3 += 1) {
        var k3: i64 = 0;
        while (k3 < l.ne3) : (k3 += 1) {
            var j2: i64 = 0;
            while (j2 < nr2) : (j2 += 1) {
                var k2: i64 = 0;
                while (k2 < l.ne2) : (k2 += 1) {
                    var j1: i64 = 0;
                    while (j1 < nr1) : (j1 += 1) {
                        var k1: i64 = 0;
                        while (k1 < l.ne1) : (k1 += 1) {
                            var j0: i64 = 0;
                            while (j0 < nr0) : (j0 += 1) {
                                vec.acc_f32(
                                    l.ne0,
                                    @ptrCast(@alignCast(dd + at(k3, l.nb3) + at(k2, l.nb2) + at(k1, l.nb1))),
                                    @ptrCast(@alignCast(s0 + at(j3 * l.ne3 + k3, l.nb03) + at(j2 * l.ne2 + k2, l.nb02) + at(j1 * l.ne1 + k1, l.nb01) + at(j0 * l.ne0, l.nb00))),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_repeat_back` (ops.cpp:1910 @cd959bae9).
pub export fn ggml_compute_forward_repeat_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => repeatBackF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `GGML_CONCAT_TILE` (ops.cpp:1931 @cd959bae9): rows of `dst` copied
/// together, so that a strided source is read along neighboring rows.
const concat_tile: i64 = 32;

/// Ports `ggml_concat_copy_strided` (ops.cpp:1935 @cd959bae9): `n_rows` rows
/// of `n` elements of `es` bytes from a source with element stride `nb0`,
/// element by element.
fn concatCopyStrided(comptime es: usize, y: [*]u8, nby: usize, x: [*]const u8, nbx: usize, nb0: usize, n: i64, n_rows: i64) void {
    var j0: i64 = 0;
    while (j0 < n) : (j0 += 1) {
        var r: i64 = 0;
        while (r < n_rows) : (r += 1) {
            const d = y + at(r, nby) + at(j0, es);
            const s = x + at(r, nbx) + at(j0, nb0);
            @memcpy(d[0..es], s[0..es]);
        }
    }
}

/// Ports `ggml_concat_copy_rows` (ops.cpp:1944 @cd959bae9): `n_rows` rows of
/// `n` elements of `src` into `dst`, rows `nbx` bytes apart in `src` and
/// `nby` bytes apart in `dst`.
fn concatCopyRows(y: [*]u8, nby: usize, src: *const Tensor, x: [*]const u8, nbx: usize, n: i64, n_rows: i64) void {
    const es = c.ggml_type_size(src.type);

    if (src.nb[0] == es) {
        const row_size = c.ggml_row_size(src.type, n);
        var r: i64 = 0;
        while (r < n_rows) : (r += 1) {
            const d = y + at(r, nby);
            const s = x + at(r, nbx);
            @memcpy(d[0..row_size], s[0..row_size]);
        }
        return;
    }

    switch (es) {
        1 => concatCopyStrided(1, y, nby, x, nbx, src.nb[0], n, n_rows),
        2 => concatCopyStrided(2, y, nby, x, nbx, src.nb[0], n, n_rows),
        4 => concatCopyStrided(4, y, nby, x, nbx, src.nb[0], n, n_rows),
        8 => concatCopyStrided(8, y, nby, x, nbx, src.nb[0], n, n_rows),
        else => impl.abort("unsupported element size"),
    }
}

/// Ports `ggml_compute_forward_concat` (ops.cpp:1965 @cd959bae9).
///
/// Each `dst` row comes from `src0` or `src1`, or for dim 0 from the row of
/// `src0` followed by the row of `src1`. The rows of `dst` are split evenly
/// across threads and walked in tiles of consecutive rows that share `i2`,
/// `i3` and their source.
pub export fn ggml_compute_forward_concat(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = common.BinaryLocals.of(src0, src1, dst);

    const dim = impl.getOpParamsI32(dst, 0);

    impl.assert(dim >= 0 and dim < 4, "dim >= 0 && dim < 4");
    impl.assert(l.nb0 == c.ggml_type_size(dst.type), "nb0 == ggml_type_size(dst->type)");
    impl.assert(!c.ggml_is_quantized(dst.type) or (l.nb00 == l.nb0 and l.nb10 == l.nb0), "!ggml_is_quantized(dst->type) || (nb00 == nb0 && nb10 == nb0)");
    impl.assert(dim != 0 or (@rem(l.ne00, c.ggml_blck_size(dst.type)) == 0 and @rem(l.ne10, c.ggml_blck_size(dst.type)) == 0), "dim != 0 || (ne00 % ggml_blck_size(dst->type) == 0 && ne10 % ggml_blck_size(dst->type) == 0)");

    const d: usize = @intCast(dim);

    const y1: usize = if (dim == 0) c.ggml_row_size(dst.type, l.ne00) else 0;

    const nr = l.ne1 * l.ne2 * l.ne3;
    const dr = @divTrunc(nr + params.nth - 1, params.nth);

    const ir0 = dr * params.ith;
    const ir1 = @min(ir0 + dr, nr);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir = ir0;
    while (ir < ir1) {
        var i = [4]i64{ 0, @rem(ir, l.ne1), @rem(@divTrunc(ir, l.ne1), l.ne2), @divTrunc(ir, l.ne1 * l.ne2) };

        // a tile of consecutive rows sharing i2, i3 and their source
        var nt = @min(@min(ir1 - ir, l.ne1 - i[1]), concat_tile);
        if (dim == 1 and i[1] < l.ne01) {
            nt = @min(nt, l.ne01 - i[1]);
        }

        const y = dd + at(i[1], l.nb1) + at(i[2], l.nb2) + at(i[3], l.nb3);

        if (dim == 0 or i[d] < src0.ne[d]) {
            concatCopyRows(y, l.nb1, src0, s0 + at(i[1], l.nb01) + at(i[2], l.nb02) + at(i[3], l.nb03), l.nb01, l.ne00, nt);
        }

        if (dim == 0 or i[d] >= src0.ne[d]) {
            i[d] -= src0.ne[d]; // i[0] does not address rows
            concatCopyRows(y + y1, l.nb1, src1, s1 + at(i[1], l.nb11) + at(i[2], l.nb12) + at(i[3], l.nb13), l.nb11, l.ne10, nt);
        }

        ir += nt;
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
