//! `upscale`, `pad`, `pad_reflect_1d`, `roll`, `arange` and
//! `timestep_embedding`: the kernels that resample, pad or generate a tensor
//! rather than combine two.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # Contraction is named, site by site
//!
//! `ops.cpp` compiles at `-ffp-contract=on`, so clang fuses `a*b + c` when the
//! multiply and the add sit in one expression. Where it has two products to
//! choose from it takes the **left** operand of the `+` — clang's
//! `tryEmitFMulAdd` looks at the left side first. The bilinear and bicubic
//! upscalers and `arange` are full of these, and each is written as the
//! `@mulAdd` clang emits, with the reasoning at the site. `make ops-diff` runs
//! `upscale` in both nearest and bilinear modes and `arange`, so it can tell
//! the port when one of these is wrong.
//!
//! # `std::min` and `std::max` are not `@min` and `@max`
//!
//! `std::max(a, b)` is `(a < b) ? b : a`, which returns `a` when either is
//! NaN; Zig's `@max` returns the non-NaN operand. The two agree on every
//! ordered input, but the C's version is reproduced so a NaN coordinate takes
//! the C's path. See `stdMin` and `stdMax`.
//!
//! # Plain `assert`, not `GGML_ASSERT`
//!
//! `pad_f32` checks `dst->nb[0]` with the C library's `assert`, which `NDEBUG`
//! compiles out of a release build. It is kept as a comment at the site, so
//! the port does not check what the shipped C does not.
//!
//! # Loop index names
//!
//! `i0`…`i3`, `i00`…`i03` and `i01` are Zig integer type names. Renamed
//! `j0`…`j3`, `j00`…`j03`, digit for digit.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

extern fn expf(x: f32) f32;
extern fn logf(x: f32) f32;

/// A stride as the signed integer the C's index arithmetic promotes it to.
inline fn s(x: usize) i64 {
    return @intCast(x);
}

/// The `f32` at byte offset `off` from `base`.
inline fn at(base: ?*anyopaque, off: i64) *f32 {
    const p: [*]u8 = @ptrCast(base.?);
    return @ptrCast(@alignCast(p + @as(usize, @intCast(off))));
}

/// `std::min(a, b)` on floats: `(b < a) ? b : a`.
inline fn stdMin(a: f32, b: f32) f32 {
    return if (b < a) b else a;
}

/// `std::max(a, b)` on floats: `(a < b) ? b : a`.
inline fn stdMax(a: f32, b: f32) f32 {
    return if (a < b) b else a;
}

/// `(float) i`, the C's implicit promotion of an `int64_t` operand.
inline fn f(i: i64) f32 {
    return @floatFromInt(i);
}

/// The C's implicit `float` to `int64_t` conversion, which truncates.
inline fn trunc64(x: f32) i64 {
    return @intFromFloat(x);
}

// -----------------------------------------------------------------------------
// upscale

/// Ports the `triangle_filter` lambda in `ggml_compute_forward_upscale_f32`
/// (ops.cpp:7839 @c1d0e7a00): `std::max(1.0f - fabsf(x), 0.0f)`.
inline fn triangleFilter(x: f32) f32 {
    return stdMax(1.0 - @abs(x), 0.0);
}

/// `a` in the bicubic lambdas of `ggml_compute_forward_upscale_f32`: the
/// PyTorch alpha.
const bicubic_a: f32 = -0.75;

/// Ports the `weight1` lambda in `ggml_compute_forward_upscale_f32`
/// (ops.cpp:7839 @c1d0e7a00): `((a + 2) * x - (a + 3)) * x * x + 1`.
///
/// Two fusions. The inner `(a + 2) * x - (a + 3)` has its product on the
/// left; the outer `... * x * x + 1` fuses its last multiply, leaving
/// `t * x` plain.
inline fn weight1(x: f32) f32 {
    const t = @mulAdd(f32, bicubic_a + 2, x, -(bicubic_a + 3));
    return @mulAdd(f32, t * x, x, 1);
}

/// Ports the `weight2` lambda in `ggml_compute_forward_upscale_f32`
/// (ops.cpp:7839 @c1d0e7a00): `((a * x - 5 * a) * x + 8 * a) * x - 4 * a`.
///
/// Three fusions, each taking the left product, which is always the one with
/// `x` in it. `5 * a`, `8 * a` and `4 * a` are exact.
inline fn weight2(x: f32) f32 {
    const a = bicubic_a;
    const t1 = @mulAdd(f32, a, x, -(5 * a));
    const t2 = @mulAdd(f32, t1, x, 8 * a);
    return @mulAdd(f32, t2, x, -(4 * a));
}

/// Ports the `bicubic` lambda in `ggml_compute_forward_upscale_f32`
/// (ops.cpp:7839 @c1d0e7a00): `p0*w0 + p1*w1 + p2*w2 + p3*w3`.
///
/// Left to right: the first `+` fuses its left product `p0*w0` over a plain
/// `p1*w1`; the next two have a fused call on the left and so fuse their
/// right product.
inline fn bicubic(p0: f32, p1: f32, p2: f32, p3: f32, x: f32) f32 {
    const w0 = weight2(x + 1);
    const w1 = weight1(x + 0);
    const w2 = weight1(1 - x);
    const w3 = weight2(2 - x);
    const s01 = @mulAdd(f32, p0, w0, p1 * w1);
    const s012 = @mulAdd(f32, p2, w2, s01);
    return @mulAdd(f32, p3, w3, s012);
}

/// Ports `ggml_compute_forward_upscale_f32` (ops.cpp:7839 @c1d0e7a00).
fn upscaleF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.UnaryLocals.of(src0, dst);
    const nb00 = s(l.nb00);
    const nb01 = s(l.nb01);
    const nb02 = s(l.nb02);
    const nb03 = s(l.nb03);
    const nb0 = s(l.nb0);
    const nb1 = s(l.nb1);
    const nb2 = s(l.nb2);
    const nb3 = s(l.nb3);

    var sf0 = f(l.ne0) / f(src0.ne[0]);
    var sf1 = f(l.ne1) / f(src0.ne[1]);
    const sf2 = f(l.ne2) / f(src0.ne[2]);
    const sf3 = f(l.ne3) / f(src0.ne[3]);
    var pixel_offset: f32 = 0.5;

    const mode_flags = impl.getOpParamsI32(dst, 0);
    const mode = mode_flags & 0xFF;

    if ((mode_flags & c.GGML_SCALE_FLAG_ALIGN_CORNERS) != 0) {
        pixel_offset = 0.0;
        sf0 = if (l.ne0 > 1 and l.ne00 > 1) f(l.ne0 - 1) / f(l.ne00 - 1) else sf0;
        sf1 = if (l.ne1 > 1 and l.ne01 > 1) f(l.ne1 - 1) / f(l.ne01 - 1) else sf1;
    }

    if (mode == c.GGML_SCALE_MODE_NEAREST) {
        var j3: i64 = 0;
        while (j3 < l.ne3) : (j3 += 1) {
            const j03 = trunc64(f(j3) / sf3);
            var j2: i64 = ith;
            while (j2 < l.ne2) : (j2 += nth) {
                const j02 = trunc64(f(j2) / sf2);
                var j1: i64 = 0;
                while (j1 < l.ne1) : (j1 += 1) {
                    const j01 = trunc64(f(j1) / sf1);
                    var j0: i64 = 0;
                    while (j0 < l.ne0) : (j0 += 1) {
                        const j00 = trunc64(f(j0) / sf0);

                        const x = at(src0.data, j00 * nb00 + j01 * nb01 + j02 * nb02 + j03 * nb03);
                        const y = at(dst.data, j0 * nb0 + j1 * nb1 + j2 * nb2 + j3 * nb3);

                        y.* = x.*;
                    }
                }
            }
        }
    } else if (mode == c.GGML_SCALE_MODE_BILINEAR and (mode_flags & c.GGML_SCALE_FLAG_ANTIALIAS) != 0) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp

        // support and invscale, minimum 1 pixel for bilinear
        const support1 = stdMax(1.0, 1.0 / sf1);
        const invscale1 = 1.0 / support1;
        const support0 = stdMax(1.0, 1.0 / sf0);
        const invscale0 = 1.0 / support0;

        var j3: i64 = 0;
        while (j3 < l.ne3) : (j3 += 1) {
            const j03 = trunc64(f(j3) / sf3);
            var j2: i64 = ith;
            while (j2 < l.ne2) : (j2 += nth) {
                const j02 = trunc64(f(j2) / sf2);
                var j1: i64 = 0;
                while (j1 < l.ne1) : (j1 += 1) {
                    const y = (f(j1) + pixel_offset) / sf1;
                    var j0: i64 = 0;
                    while (j0 < l.ne0) : (j0 += 1) {
                        const x = (f(j0) + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        const x_min = @max(trunc64(x - support0 + pixel_offset), 0);
                        const x_max = @min(trunc64(x + support0 + pixel_offset), l.ne00);
                        const y_min = @max(trunc64(y - support1 + pixel_offset), 0);
                        const y_max = @min(trunc64(y + support1 + pixel_offset), l.ne01);

                        // bilinear filter with antialiasing
                        var val: f32 = 0.0;
                        var total_weight: f32 = 0.0;

                        var sy = y_min;
                        while (sy < y_max) : (sy += 1) {
                            const weight_y = triangleFilter((f(sy) - y + pixel_offset) * invscale1);

                            var sx = x_min;
                            while (sx < x_max) : (sx += 1) {
                                const weight_x = triangleFilter((f(sx) - x + pixel_offset) * invscale0);
                                const weight = weight_x * weight_y;

                                if (weight <= 0.0) continue;

                                const pixel = at(src0.data, sx * nb00 + sy * nb01 + j02 * nb02 + j03 * nb03).*;
                                // `val += pixel * weight` is one expression: fused.
                                val = @mulAdd(f32, pixel, weight, val);
                                total_weight += weight;
                            }
                        }

                        if (total_weight > 0.0) {
                            val /= total_weight;
                        }

                        at(dst.data, j0 * nb0 + j1 * nb1 + j2 * nb2 + j3 * nb3).* = val;
                    }
                }
            }
        }
    } else if (mode == c.GGML_SCALE_MODE_BILINEAR) {
        var j3: i64 = 0;
        while (j3 < l.ne3) : (j3 += 1) {
            const j03 = trunc64(f(j3) / sf3);
            var j2: i64 = ith;
            while (j2 < l.ne2) : (j2 += nth) {
                const j02 = trunc64(f(j2) / sf2);
                var j1: i64 = 0;
                while (j1 < l.ne1) : (j1 += 1) {
                    const y = (f(j1) + pixel_offset) / sf1 - pixel_offset;
                    var y0 = trunc64(@floor(y));
                    var y1 = y0 + 1;

                    y0 = @max(0, @min(y0, l.ne01 - 1));
                    y1 = @max(0, @min(y1, l.ne01 - 1));

                    var dy = y - f(y0);
                    dy = stdMax(0.0, stdMin(dy, 1.0));

                    var j0: i64 = 0;
                    while (j0 < l.ne0) : (j0 += 1) {
                        const x = (f(j0) + pixel_offset) / sf0 - pixel_offset;
                        var x0 = trunc64(@floor(x));
                        var x1 = x0 + 1;

                        x0 = @max(0, @min(x0, l.ne00 - 1));
                        x1 = @max(0, @min(x1, l.ne00 - 1));

                        var dx = x - f(x0);
                        dx = stdMax(0.0, stdMin(dx, 1.0));

                        // fetch the four surrounding pixel values and interpolate
                        // (`c` is renamed `cc`: `c` is the C import here.)
                        const a = at(src0.data, x0 * nb00 + y0 * nb01 + j02 * nb02 + j03 * nb03).*;
                        const b = at(src0.data, x1 * nb00 + y0 * nb01 + j02 * nb02 + j03 * nb03).*;
                        const cc = at(src0.data, x0 * nb00 + y1 * nb01 + j02 * nb02 + j03 * nb03).*;
                        const d = at(src0.data, x1 * nb00 + y1 * nb01 + j02 * nb02 + j03 * nb03).*;

                        // `a*(1-dx)*(1-dy) + b*dx*(1-dy) + c*(1-dx)*dy + d*dx*dy`
                        // groups left to right. The first `+` fuses the left
                        // term's outer multiply over a plain `b*dx*(1-dy)`;
                        // the next two have a fused call on their left and so
                        // fuse the right term's outer multiply.
                        const ab = @mulAdd(f32, a * (1 - dx), 1 - dy, b * dx * (1 - dy));
                        const abc = @mulAdd(f32, cc * (1 - dx), dy, ab);
                        const val = @mulAdd(f32, d * dx, dy, abc);

                        at(dst.data, j0 * nb0 + j1 * nb1 + j2 * nb2 + j3 * nb3).* = val;
                    }
                }
            }
        }
    } else if (mode == c.GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        var j3: i64 = 0;
        while (j3 < l.ne3) : (j3 += 1) {
            const j03 = trunc64(f(j3) / sf3);
            var j2: i64 = ith;
            while (j2 < l.ne2) : (j2 += nth) {
                const j02 = trunc64(f(j2) / sf2);
                var j1: i64 = 0;
                while (j1 < l.ne1) : (j1 += 1) {
                    const y = (f(j1) + pixel_offset) / sf1 - pixel_offset;
                    const y0 = trunc64(@floor(y));
                    const dy = y - f(y0);

                    var j0: i64 = 0;
                    while (j0 < l.ne0) : (j0 += 1) {
                        const x = (f(j0) + pixel_offset) / sf0 - pixel_offset;
                        const x0 = trunc64(@floor(x));
                        const dx = x - f(x0);

                        const p: BicubicTap = .{
                            .data = src0.data,
                            .x0 = x0,
                            .y0 = y0,
                            .ne00 = l.ne00,
                            .ne01 = l.ne01,
                            .base = j02 * nb02 + j03 * nb03,
                            .nb00 = nb00,
                            .nb01 = nb01,
                        };

                        const val = bicubic(
                            bicubic(p.tap(-1, -1), p.tap(0, -1), p.tap(1, -1), p.tap(2, -1), dx),
                            bicubic(p.tap(-1, 0), p.tap(0, 0), p.tap(1, 0), p.tap(2, 0), dx),
                            bicubic(p.tap(-1, 1), p.tap(0, 1), p.tap(1, 1), p.tap(2, 1), dx),
                            bicubic(p.tap(-1, 2), p.tap(0, 2), p.tap(1, 2), p.tap(2, 2), dx),
                            dy,
                        );

                        at(dst.data, j0 * nb0 + j1 * nb1 + j2 * nb2 + j3 * nb3).* = val;
                    }
                }
            }
        }
    } else {
        impl.abort("unsupported upscale mode");
    }
}

/// Ports the `p` lambda in `ggml_compute_forward_upscale_f32`
/// (ops.cpp:7839 @c1d0e7a00): one source pixel, clamped to the edge.
///
/// The lambda captures by copy; this struct is that capture.
const BicubicTap = struct {
    data: ?*anyopaque,
    x0: i64,
    y0: i64,
    ne00: i64,
    ne01: i64,
    /// `i02*nb02 + i03*nb03`, which the lambda adds last.
    base: i64,
    nb00: i64,
    nb01: i64,

    inline fn tap(self: BicubicTap, x_off: i64, y_off: i64) f32 {
        const j00 = @max(0, @min(self.x0 + x_off, self.ne00 - 1));
        const j01 = @max(0, @min(self.y0 + y_off, self.ne01 - 1));
        return at(self.data, j00 * self.nb00 + j01 * self.nb01 + self.base).*;
    }
};

/// Ports `ggml_compute_forward_upscale` (ops.cpp:8035 @c1d0e7a00).
pub export fn ggml_compute_forward_upscale(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => upscaleF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// pad

/// Ports `ggml_wrap_around` (ops.cpp:6872 @c1d0e7a00).
///
/// `static inline` in the conv section of `ops.cpp`, which the pad kernel
/// calls from further down. A private copy: the conv kernels are another
/// file's.
inline fn wrapAround(coord: i64, size: i64) i64 {
    return @rem(coord + size, size); // adding size avoids negative number weirdness
}

/// Ports `ggml_compute_forward_pad_f32` (ops.cpp:8057 @c1d0e7a00).
///
/// `template<bool circular_t>` in the C; a comptime `bool` here, so the
/// `if constexpr` folds the same way.
fn padF32(comptime circular: bool, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    // assert(dst->nb[0] == sizeof(float)); -- compiled out under NDEBUG

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.UnaryLocals.of(src0, dst);
    const nb00 = s(l.nb00);
    const nb01 = s(l.nb01);
    const nb02 = s(l.nb02);
    const nb03 = s(l.nb03);

    const dst_ptr: [*]f32 = @ptrCast(@alignCast(dst.data.?));
    const lp0: i64 = impl.getOpParamsI32(dst, 0);
    const rp0: i64 = impl.getOpParamsI32(dst, 1);
    const lp1: i64 = impl.getOpParamsI32(dst, 2);
    const rp1: i64 = impl.getOpParamsI32(dst, 3);
    const lp2: i64 = impl.getOpParamsI32(dst, 4);
    const rp2: i64 = impl.getOpParamsI32(dst, 5);
    const lp3: i64 = impl.getOpParamsI32(dst, 6);
    const rp3: i64 = impl.getOpParamsI32(dst, 7);

    // TODO: optimize

    var j2: i64 = 0;
    while (j2 < l.ne2) : (j2 += 1) {
        var j1: i64 = ith;
        while (j1 < l.ne1) : (j1 += nth) {
            var j0: i64 = 0;
            while (j0 < l.ne0) : (j0 += 1) {
                var j3: i64 = 0;
                while (j3 < l.ne3) : (j3 += 1) {
                    const dst_idx: usize = @intCast(j3 * (l.ne0 * l.ne1 * l.ne2) + j2 * (l.ne0 * l.ne1) + j1 * l.ne0 + j0);
                    // circular means wrap around on a torus, so x and y loop around
                    if (circular) {
                        const src_i0 = wrapAround(j0 - lp0, l.ne00);
                        const src_i1 = wrapAround(j1 - lp1, l.ne01);
                        const src_i2 = wrapAround(j2 - lp2, l.ne02);
                        const src_i3 = wrapAround(j3 - lp3, l.ne03);

                        const src_idx =
                            src_i3 * nb03 +
                            src_i2 * nb02 +
                            src_i1 * nb01 +
                            src_i0 * nb00;

                        dst_ptr[dst_idx] = at(src0.data, src_idx).*;
                    } else {
                        if ((j0 >= lp0 and j0 < l.ne0 - rp0) and
                            (j1 >= lp1 and j1 < l.ne1 - rp1) and
                            (j2 >= lp2 and j2 < l.ne2 - rp2) and
                            (j3 >= lp3 and j3 < l.ne3 - rp3))
                        {
                            const src_idx = (j3 - lp3) * nb03 + (j2 - lp2) * nb02 + (j1 - lp1) * nb01 + (j0 - lp0) * nb00;
                            dst_ptr[dst_idx] = at(src0.data, src_idx).*;
                        } else {
                            dst_ptr[dst_idx] = 0;
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_pad` (ops.cpp:8122 @c1d0e7a00).
pub export fn ggml_compute_forward_pad(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const circular = impl.getOpParamsI32(dst, 8) != 0;
    switch (src0.type) {
        c.GGML_TYPE_F32 => if (circular) padF32(true, params, dst) else padF32(false, params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// pad_reflect_1d

/// Ports `ggml_compute_forward_pad_reflect_1d` (ops.cpp:8145 @c1d0e7a00).
pub export fn ggml_compute_forward_pad_reflect_1d(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const p0: i64 = impl.getOpParamsI32(dst, 0);
    const p1: i64 = impl.getOpParamsI32(dst, 1);

    const l = common.UnaryLocals.of(src0, dst);
    const nb01 = s(l.nb01);
    const nb02 = s(l.nb02);
    const nb03 = s(l.nb03);
    const nb0 = s(l.nb0);
    const nb1 = s(l.nb1);
    const nb2 = s(l.nb2);
    const nb3 = s(l.nb3);

    var j3: i64 = 0;
    while (j3 < l.ne3) : (j3 += 1) {
        var j2: i64 = 0;
        while (j2 < l.ne2) : (j2 += 1) {
            var j1: i64 = ith;
            while (j1 < l.ne1) : (j1 += nth) {
                const left: [*]f32 = @ptrCast(at(dst.data, j3 * nb3 + j2 * nb2 + j1 * nb1 + p0 * nb0));
                const right: [*]f32 = @ptrCast(at(dst.data, j3 * nb3 + j2 * nb2 + j1 * nb1 + (l.ne0 - p1 - 1) * nb0));

                vec.cpy_f32(l.ne00, left, @ptrCast(at(src0.data, j3 * nb03 + j2 * nb02 + j1 * nb01)));

                // `left[-i0]` and `right[-i0]`: pointer arithmetic below the
                // row start, inside the padded row.
                var k: i64 = 1;
                while (k <= p0) : (k += 1) (left - @as(usize, @intCast(k)))[0] = left[@intCast(k)];
                k = 1;
                while (k <= p1) : (k += 1) right[@intCast(k)] = (right - @as(usize, @intCast(k)))[0];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// roll

/// Ports `ggml_wrap_index` (ops.cpp:8180 @c1d0e7a00).
fn wrapIndex(i: i64, ne: i64) i64 {
    if (i < 0) {
        return i + ne;
    } else if (i >= ne) {
        return i - ne;
    }
    return i;
}

/// Ports `ggml_compute_forward_roll_f32` (ops.cpp:8189 @c1d0e7a00).
fn rollF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src_data: [*]const f32 = @ptrCast(@alignCast(src0.data.?));
    const dst_data: [*]f32 = @ptrCast(@alignCast(dst.data.?));

    const l = common.UnaryLocals.of(src0, dst);

    const s0: i64 = impl.getOpParamsI32(dst, 0);
    const s1: i64 = impl.getOpParamsI32(dst, 1);
    const s2: i64 = impl.getOpParamsI32(dst, 2);
    const s3: i64 = impl.getOpParamsI32(dst, 3);

    const total = l.ne1 * l.ne2 * l.ne3;
    // Note `+ nth`, not `+ nth - 1`: the C over-allocates by one row per
    // thread, and the `min` below absorbs it.
    const per_thread = @divTrunc(total + params.nth, params.nth);
    const start = params.ith * per_thread;
    const end = @min(start + per_thread, total);

    var i = start;
    while (i < end) : (i += 1) {
        const j1 = @rem(i, l.ne1);
        const j2 = @rem(@divTrunc(i, l.ne1), l.ne2);
        const j3 = @divTrunc(i, l.ne2 * l.ne1);
        const dst_row = dst_data + (@as(usize, @intCast(j3)) * l.nb3 + @as(usize, @intCast(j2)) * l.nb2 + @as(usize, @intCast(j1)) * l.nb1) / @sizeOf(f32);

        const j01 = wrapIndex(j1 - s1, l.ne01);
        const j02 = wrapIndex(j2 - s2, l.ne02);
        const j03 = wrapIndex(j3 - s3, l.ne03);
        const src_row = src_data + (@as(usize, @intCast(j03)) * l.nb03 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j01)) * l.nb01) / @sizeOf(f32);

        const sh = wrapIndex(-s0, l.ne00);
        const n = l.ne00 - sh;
        vec.cpy_f32(n, dst_row, src_row + @as(usize, @intCast(sh)));
        vec.cpy_f32(sh, dst_row + @as(usize, @intCast(n)), src_row);
    }
}

/// Ports `ggml_compute_forward_roll` (ops.cpp:8227 @c1d0e7a00).
pub export fn ggml_compute_forward_roll(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => rollF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// arange

/// Ports `ggml_compute_forward_arange_f32` (ops.cpp:8247 @c1d0e7a00).
fn arangeF32(params: *const ComputeParams, dst: *Tensor) void {
    impl.assert(dst.nb[0] == @sizeOf(f32), "dst->nb[0] == sizeof(float)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const start = impl.getOpParamsF32(dst, 0);
    const stop = impl.getOpParamsF32(dst, 1);
    const step = impl.getOpParamsF32(dst, 2);

    const steps: i64 = @intFromFloat(@ceil((stop - start) / step));

    impl.assert(c.ggml_nelements(dst) == steps, "ggml_nelements(dst) == steps");

    const d: [*]f32 = @ptrCast(@alignCast(dst.data.?));
    var i: i64 = ith;
    while (i < steps) : (i += nth) {
        // `start + step * i` is one expression: fused, with `i` promoted to
        // `float` first.
        const value = @mulAdd(f32, step, f(i), start);
        d[@intCast(i)] = value;
    }
}

/// Ports `ggml_compute_forward_arange` (ops.cpp:8270 @c1d0e7a00).
pub export fn ggml_compute_forward_arange(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    switch (dst.type) {
        c.GGML_TYPE_F32 => arangeF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// timestep_embedding

/// Ports `ggml_compute_forward_timestep_embedding_f32` (ops.cpp:8285 @c1d0e7a00).
fn timestepEmbeddingF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const l = common.UnaryLocals.of(src0, dst);

    const dim: i64 = impl.getOpParamsI32(dst, 0);
    const max_period = impl.getOpParamsI32(dst, 1);

    const half = @divTrunc(dim, 2);

    const src: [*]const f32 = @ptrCast(@alignCast(src0.data.?));

    var i: i64 = 0;
    while (i < l.ne00) : (i += 1) {
        const embed_data: [*]f32 = @ptrCast(at(dst.data, i * s(l.nb1)));
        var j: i64 = ith;
        while (j < half) : (j += nth) {
            const timestep = src[@intCast(i)];
            // `-logf(max_period) * j / half`, each `int` promoted to `float`;
            // no add, so nothing to fuse.
            const freq = expf(-logf(@floatFromInt(max_period)) * f(j) / f(half));
            const arg = timestep * freq;
            // clang folds this `cosf`/`sinf` pair into one `__sincosf_stret`
            // call, which rounds differently -- see `common.sinCos`.
            const sc = common.sinCos(arg);
            embed_data[@intCast(j)] = sc.cos;
            embed_data[@intCast(j + half)] = sc.sin;
        }
        if (@rem(dim, 2) != 0 and ith == 0) {
            embed_data[@intCast(2 * half)] = 0.0;
        }
    }
}

/// Ports `ggml_compute_forward_timestep_embedding` (ops.cpp:8318 @c1d0e7a00).
pub export fn ggml_compute_forward_timestep_embedding(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => timestepEmbeddingF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "std::min and std::max keep the C's NaN behaviour" {
    const nan = std.math.nan(f32);
    // `std::min(NaN, 1)` is `(1 < NaN) ? 1 : NaN`, so NaN; `@min` would say 1.
    try std.testing.expect(std.math.isNan(stdMin(nan, 1.0)));
    // `std::max(0, NaN)` is `(0 < NaN) ? NaN : 0`, so 0.
    try std.testing.expectEqual(@as(f32, 0.0), stdMax(0.0, nan));
    try std.testing.expectEqual(@as(f32, 2.0), stdMax(1.0, 2.0));
    try std.testing.expectEqual(@as(f32, 1.0), stdMin(1.0, 2.0));
}

test "the bicubic weights sum to one" {
    // A partition of unity holds for any alpha; at x = 0 the kernel picks
    // the centre tap exactly.
    try std.testing.expectEqual(@as(f32, 1.0), weight1(0));
    try std.testing.expectEqual(@as(f32, 0.0), weight2(1));
    try std.testing.expectEqual(@as(f32, 5.0), bicubic(3, 5, 7, 9, 0));
    const x: f32 = 0.25;
    const sum = weight2(x + 1) + weight1(x) + weight1(1 - x) + weight2(2 - x);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
}

test "wrap helpers" {
    try std.testing.expectEqual(@as(i64, 4), wrapAround(-1, 5));
    try std.testing.expectEqual(@as(i64, 0), wrapAround(5, 5));
    try std.testing.expectEqual(@as(i64, 4), wrapIndex(-1, 5));
    try std.testing.expectEqual(@as(i64, 0), wrapIndex(5, 5));
    try std.testing.expectEqual(@as(i64, 3), wrapIndex(3, 5));
}
