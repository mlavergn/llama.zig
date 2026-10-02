//! `pool_1d`, `pool_2d` and `pool_2d_back`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Single-threaded, and `std::max` is not `@max`
//!
//! All three kernels return immediately on any thread but the first. The max
//! pools call `std::max(v, res)`, which is `(v < res) ? res : v`: it returns
//! the *first* argument when either is NaN. Zig's `@max` drops a NaN instead,
//! so the C's expression is written out as `stdMax`.
//!
//! # The `f16` average gradient adds bit patterns
//!
//! `pool_2d_back`'s average branch does `((ggml_fp16_t *) drow)[j] +=
//! GGML_CPU_FP32_TO_FP16(grad)`. `ggml_fp16_t` is `uint16_t`, so that is an
//! *integer* add of two half-precision encodings, not a float add. It is
//! reproduced as written, wrapping as the C's narrowing store does; it is a
//! bug upstream, not here.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;
const fp16 = c.ggml_fp16_t;

/// `FLT_MAX`, which the max pools start from negated.
const flt_max = std.math.floatMax(f32);

/// `std::max(a, b)`: `(a < b) ? b : a`. See the file header for why this is
/// not `@max`.
inline fn stdMax(a: f32, b: f32) f32 {
    return if (a < b) b else a;
}

/// Reads element `j` of a row as `f32`, from `f32` or `f16` storage.
inline fn loadF32(is_f32: bool, row: [*]const u8, j: i64) f32 {
    const i: usize = @intCast(j);
    if (is_f32) return @as([*]const f32, @ptrCast(@alignCast(row)))[i];
    return impl.fp16ToFp32(@as([*]const fp16, @ptrCast(@alignCast(row)))[i]);
}

/// Ports `ggml_compute_forward_pool_1d_ksp` (ops.cpp:7544 @c1d0e7a00).
///
/// Parameters:
/// - `op`: `GGML_OP_POOL_AVG` or `GGML_OP_POOL_MAX`.
/// - `k`, `s`, `p`: kernel size, stride and padding.
fn pool1dKsp(params: *const ComputeParams, op: c.enum_ggml_op_pool, k: i32, s: i32, p: i32, dst: *Tensor) void {
    const src = impl.one(Tensor, dst.src[0]);

    std.debug.assert(src.type == c.GGML_TYPE_F32 or src.type == c.GGML_TYPE_F16);

    if (params.ith != 0) return;

    const IW = src.ne[0];
    const OW = dst.ne[0];

    const nr: i64 = @intCast(c.ggml_nrows(src));
    const is_f32 = src.type == c.GGML_TYPE_F32;

    const sd: [*]const u8 = @ptrCast(src.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = 0;
    while (ir < nr) : (ir += 1) {
        const srow_bytes = sd + @as(usize, @intCast(ir)) * src.nb[1];
        const drow: [*]f32 = @ptrCast(@alignCast(dd + @as(usize, @intCast(ir)) * dst.nb[1]));

        var ow: i64 = 0;
        while (ow < OW) : (ow += 1) {
            var res: f32 = 0;
            switch (op) {
                c.GGML_OP_POOL_AVG => res = 0.0,
                c.GGML_OP_POOL_MAX => res = -flt_max,
                c.GGML_OP_POOL_COUNT => impl.abort("fatal error"),
                else => {},
            }

            var count: i32 = 0;
            const base: i32 = @as(i32, @intCast(ow)) * s - p;

            var ki: i32 = 0;
            while (ki < k) : (ki += 1) {
                const j = base + ki;
                if (j < 0 or j >= @as(i32, @intCast(IW))) continue;

                const v = loadF32(is_f32, srow_bytes, j);

                switch (op) {
                    c.GGML_OP_POOL_AVG => res += v,
                    c.GGML_OP_POOL_MAX => res = stdMax(v, res),
                    c.GGML_OP_POOL_COUNT => impl.abort("fatal error"),
                    else => {},
                }

                count += 1;
            }

            switch (op) {
                c.GGML_OP_POOL_AVG => res = if (count > 0) res / @as(f32, @floatFromInt(count)) else 0.0,
                c.GGML_OP_POOL_MAX => {},
                c.GGML_OP_POOL_COUNT => impl.abort("fatal error"),
                else => {},
            }

            drow[@intCast(ow)] = res;
        }
    }
}

/// Ports `ggml_compute_forward_pool_1d` (ops.cpp:7615 @c1d0e7a00).
pub export fn ggml_compute_forward_pool_1d(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const opts = dst.op_params;
    const op: c.enum_ggml_op_pool = @bitCast(opts[0]);
    const k0 = opts[1];
    const s0 = opts[2];
    const p0 = opts[3];

    pool1dKsp(params, op, k0, s0, p0, dst);
}

/// Ports `ggml_compute_forward_pool_2d` (ops.cpp:7630 @c1d0e7a00).
///
/// Walks `src` one `nb[2]` plane at a time until its bytes run out, so the
/// third and fourth dimensions are flattened together.
pub export fn ggml_compute_forward_pool_2d(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src = impl.one(Tensor, dst.src[0]);

    std.debug.assert(src.type == c.GGML_TYPE_F32 or src.type == c.GGML_TYPE_F16);

    if (params.ith != 0) return;

    const opts = dst.op_params;

    const op: c.enum_ggml_op_pool = @bitCast(opts[0]);
    const k0 = opts[1];
    const k1 = opts[2];
    const s0 = opts[3];
    const s1 = opts[4];
    const p0 = opts[5];
    const p1 = opts[6];
    var cdata: [*]const u8 = @ptrCast(src.data.?);
    const data_end = cdata + c.ggml_nbytes(src);

    const px = dst.ne[0];
    const py = dst.ne[1];
    const pa = px * py;

    var dplane: [*]f32 = @ptrCast(@alignCast(dst.data.?));

    const ka = k0 * k1;
    const offset0 = -p0;
    const offset1 = -p1;

    const is_f32 = src.type == c.GGML_TYPE_F32;

    while (@intFromPtr(cdata) < @intFromPtr(data_end)) {
        var oy: i32 = 0;
        while (oy < py) : (oy += 1) {
            const drow = dplane + @as(usize, @intCast(oy * px));
            const out = drow;

            var ox: i32 = 0;
            while (ox < px) : (ox += 1) {
                var res: f32 = 0;
                switch (op) {
                    c.GGML_OP_POOL_AVG => res = 0,
                    c.GGML_OP_POOL_MAX => res = -flt_max,
                    c.GGML_OP_POOL_COUNT => impl.abort("fatal error"),
                    else => {},
                }

                const ix = offset0 + ox * s0;
                const iy = offset1 + oy * s1;

                var ky: i32 = 0;
                while (ky < k1) : (ky += 1) {
                    if (iy + ky < 0 or iy + ky >= src.ne[1]) continue;

                    const srow = cdata + src.nb[1] * @as(usize, @intCast(iy + ky));
                    var kx: i32 = 0;
                    while (kx < k0) : (kx += 1) {
                        const j = ix + kx;
                        if (j < 0 or j >= src.ne[0]) continue;

                        const srow_j = loadF32(is_f32, srow, j);
                        switch (op) {
                            c.GGML_OP_POOL_AVG => res += srow_j,
                            c.GGML_OP_POOL_MAX => res = stdMax(srow_j, res),
                            c.GGML_OP_POOL_COUNT => impl.abort("fatal error"),
                            else => {},
                        }
                    }
                }
                switch (op) {
                    c.GGML_OP_POOL_AVG => res /= @as(f32, @floatFromInt(ka)),
                    c.GGML_OP_POOL_MAX => {},
                    c.GGML_OP_POOL_COUNT => impl.abort("fatal error"),
                    else => {},
                }

                out[@intCast(ox)] = res;
            }
        }

        cdata += src.nb[2];
        dplane += @as(usize, @intCast(pa));
    }
}

/// Ports `ggml_compute_forward_pool_2d_back` (ops.cpp:7717 @c1d0e7a00).
///
/// `src` is the gradient of the pooled output and `src[1]` the forward input.
/// Note the forward tensor is read through `dst`'s type and row stride, as in
/// the C — the two are the same shape and type by construction.
pub export fn ggml_compute_forward_pool_2d_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src = impl.one(Tensor, dst.src[0]);
    const dstf = impl.one(Tensor, dst.src[1]); // forward tensor of dst

    std.debug.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16);

    if (params.ith != 0) return;

    const opts = dst.op_params;
    const op: c.enum_ggml_op_pool = @bitCast(opts[0]);
    const k0 = opts[1];
    const k1 = opts[2];
    const s0 = opts[3];
    const s1 = opts[4];
    const p0 = opts[5];
    const p1 = opts[6];

    var cdata: [*]u8 = @ptrCast(dst.data.?);
    var cdataf: [*]const u8 = @ptrCast(dstf.data.?);
    const nbytes = c.ggml_nbytes(dst);
    const data_end = cdata + nbytes;

    impl.assert(params.ith == 0, "params->ith == 0");
    @memset(cdata[0..nbytes], 0);

    const px = src.ne[0];
    const py = src.ne[1];
    const pa = px * py;

    var splane: [*]const f32 = @ptrCast(@alignCast(src.data.?));

    const ka = k0 * k1;
    const offset0 = -p0;
    const offset1 = -p1;

    const is_f32 = dst.type == c.GGML_TYPE_F32;

    while (@intFromPtr(cdata) < @intFromPtr(data_end)) {
        var oy: i32 = 0;
        while (oy < py) : (oy += 1) {
            const srow = splane + @as(usize, @intCast(oy * px));
            var ox: i32 = 0;
            while (ox < px) : (ox += 1) {
                const grad0 = srow[@intCast(ox)];

                const ix = offset0 + ox * s0;
                const iy = offset1 + oy * s1;

                if (op == c.GGML_OP_POOL_MAX) {
                    var maxval: f32 = -flt_max;
                    var kxmax: i32 = -1;
                    var kymax: i32 = -1;

                    var ky: i32 = 0;
                    while (ky < k1) : (ky += 1) {
                        if (iy + ky < 0 or iy + ky >= dst.ne[1]) continue;
                        const drowf = cdataf + dst.nb[1] * @as(usize, @intCast(iy + ky));
                        var kx: i32 = 0;
                        while (kx < k0) : (kx += 1) {
                            const j = ix + kx;
                            if (j < 0 or j >= dst.ne[0]) continue;

                            const val = loadF32(is_f32, drowf, j);
                            if (val <= maxval) continue;

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    if (kxmax == -1 or kymax == -1) continue;

                    const drow = cdata + dst.nb[1] * @as(usize, @intCast(iy + kymax));
                    const j: usize = @intCast(ix + kxmax);
                    if (is_f32) {
                        @as([*]f32, @ptrCast(@alignCast(drow)))[j] += grad0;
                    } else {
                        const d16: [*]fp16 = @ptrCast(@alignCast(drow));
                        d16[j] = impl.fp32ToFp16(grad0 + impl.fp16ToFp32(d16[j]));
                    }
                } else if (op == c.GGML_OP_POOL_AVG) {
                    const grad = grad0 / @as(f32, @floatFromInt(ka));

                    var ky: i32 = 0;
                    while (ky < k1) : (ky += 1) {
                        if (iy + ky < 0 or iy + ky >= dst.ne[1]) continue;
                        const drow = cdata + dst.nb[1] * @as(usize, @intCast(iy + ky));
                        var kx: i32 = 0;
                        while (kx < k0) : (kx += 1) {
                            const j = ix + kx;
                            if (j < 0 or j >= dst.ne[0]) continue;

                            const jj: usize = @intCast(j);
                            if (is_f32) {
                                @as([*]f32, @ptrCast(@alignCast(drow)))[jj] += grad;
                            } else {
                                // Integer add of two f16 encodings; see the
                                // file header.
                                const d16: [*]fp16 = @ptrCast(@alignCast(drow));
                                d16[jj] +%= impl.fp32ToFp16(grad);
                            }
                        }
                    }
                } else {
                    impl.assert(false, "false");
                }
            }
        }

        cdata += dst.nb[2];
        cdataf += dst.nb[2];
        splane += @as(usize, @intCast(pa));
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "stdMax returns its first argument when either is NaN, as std::max does" {
    const nan = std.math.nan(f32);
    try std.testing.expect(std.math.isNan(stdMax(nan, 1.0)));
    try std.testing.expectEqual(@as(f32, 1.0), stdMax(1.0, nan));
    try std.testing.expectEqual(@as(f32, 2.0), stdMax(1.0, 2.0));
    try std.testing.expectEqual(@as(f32, 2.0), stdMax(2.0, 1.0));
}
