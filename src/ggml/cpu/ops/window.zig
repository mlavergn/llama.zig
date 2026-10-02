//! `win_part`, `win_unpart`, `get_rel_pos` and `add_rel_pos` — the windowed
//! attention helpers of the SAM image encoder.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! The C places `unary` and `glu` between `win_unpart` and `get_rel_pos`; those
//! belong to the activation family and live in their own file.
//!
//! # Plain `assert`, not `GGML_ASSERT`
//!
//! `win_part` and `win_unpart` check their shapes with the C library's
//! `assert`, which `NDEBUG` compiles out of a release build. They are kept as
//! comments at the site, so the port does not check what the shipped C does
//! not.
//!
//! # Loop index names
//!
//! `i0`…`i3`, `i00`…`i02`, `i10`…`i13` are Zig integer type names. Renamed
//! `j0`…`j3`, `j00`…`j02`, `j10`…`j13`, digit for digit. The C's own `j` in
//! `add_rel_pos` is renamed `jj` to keep it apart from them.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const threading = @import("../threading.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// Ports `ggml_compute_forward_win_part_f32` (ops.cpp:9874 @c1d0e7a00).
fn winPartF32(params: *const ComputeParams, dst: *Tensor) void {
    _ = params;

    const src0 = impl.one(Tensor, dst.src[0]);

    const l = common.UnaryLocals.of(src0, dst);

    const nep0: i64 = impl.getOpParamsI32(dst, 0);
    const nep1: i64 = impl.getOpParamsI32(dst, 1);
    const w: i64 = impl.getOpParamsI32(dst, 2);

    // assert(ne00 == ne0);
    // assert(ne3  == nep0*nep1);

    const d: [*]f32 = @ptrCast(@alignCast(dst.data.?));
    const s: [*]const f32 = @ptrCast(@alignCast(src0.data.?));

    // TODO: optimize / multi-thread
    var py: i64 = 0;
    while (py < nep1) : (py += 1) {
        var px: i64 = 0;
        while (px < nep0) : (px += 1) {
            const j3 = py * nep0 + px;
            var j2: i64 = 0;
            while (j2 < l.ne2) : (j2 += 1) {
                var j1: i64 = 0;
                while (j1 < l.ne1) : (j1 += 1) {
                    var j0: i64 = 0;
                    while (j0 < l.ne0) : (j0 += 1) {
                        const j02 = py * w + j2;
                        const j01 = px * w + j1;
                        const j00 = j0;

                        const i: usize = @intCast(j3 * l.ne2 * l.ne1 * l.ne0 + j2 * l.ne1 * l.ne0 + j1 * l.ne0 + j0);

                        if (py * w + j2 >= l.ne02 or px * w + j1 >= l.ne01) {
                            d[i] = 0.0;
                        } else {
                            const j: usize = @intCast(j02 * l.ne01 * l.ne00 + j01 * l.ne00 + j00);
                            d[i] = s[j];
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_win_part` (ops.cpp:9917 @c1d0e7a00).
pub export fn ggml_compute_forward_win_part(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => winPartF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_win_unpart_f32` (ops.cpp:9937 @c1d0e7a00).
fn winUnpartF32(params: *const ComputeParams, dst: *Tensor) void {
    _ = params;

    const src0 = impl.one(Tensor, dst.src[0]);

    const l = common.UnaryLocals.of(src0, dst);

    const w: i64 = impl.getOpParamsI32(dst, 0);

    // padding
    const px = @rem(w - @rem(l.ne1, w), w);
    //const int py = (w - ne2%w)%w;

    const npx = @divTrunc(px + l.ne1, w);
    //const int npy = (py + ne2)/w;

    // assert(ne0 == ne00);

    const d: [*]f32 = @ptrCast(@alignCast(dst.data.?));
    const s: [*]const f32 = @ptrCast(@alignCast(src0.data.?));

    // TODO: optimize / multi-thread
    var j2: i64 = 0;
    while (j2 < l.ne2) : (j2 += 1) {
        var j1: i64 = 0;
        while (j1 < l.ne1) : (j1 += 1) {
            var j0: i64 = 0;
            while (j0 < l.ne0) : (j0 += 1) {
                const ip2 = @divTrunc(j2, w);
                const ip1 = @divTrunc(j1, w);

                const j02 = @rem(j2, w);
                const j01 = @rem(j1, w);
                const j00 = j0;

                const i: usize = @intCast((ip2 * npx + ip1) * l.ne02 * l.ne01 * l.ne00 + j02 * l.ne01 * l.ne00 + j01 * l.ne00 + j00);
                const j: usize = @intCast(j2 * l.ne1 * l.ne0 + j1 * l.ne0 + j0);

                d[j] = s[i];
            }
        }
    }
}

/// Ports `ggml_compute_forward_win_unpart` (ops.cpp:9978 @c1d0e7a00).
pub export fn ggml_compute_forward_win_unpart(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => winUnpartF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_get_rel_pos_f16` (ops.cpp:10142 @c1d0e7a00).
///
/// A pure copy of 16-bit elements, which is why the dispatcher sends `BF16`
/// here too: the bits move unchanged whatever they encode.
fn getRelPosF16(params: *const ComputeParams, dst: *Tensor) void {
    _ = params;

    const src0 = impl.one(Tensor, dst.src[0]);

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    const l = common.UnaryLocals.of(src0, dst);

    const w = l.ne1;

    const src0_data: [*]const c.ggml_fp16_t = @ptrCast(@alignCast(src0.data.?));
    const dst_data: [*]c.ggml_fp16_t = @ptrCast(@alignCast(dst.data.?));

    var j2: i64 = 0;
    while (j2 < l.ne2) : (j2 += 1) {
        var j1: i64 = 0;
        while (j1 < l.ne1) : (j1 += 1) {
            const pos = (w - j1 - 1) + j2;
            var j0: i64 = 0;
            while (j0 < l.ne0) : (j0 += 1) {
                dst_data[@intCast(j2 * l.ne1 * l.ne0 + j1 * l.ne0 + j0)] = src0_data[@intCast(pos * l.ne00 + j0)];
            }
        }
    }
}

/// Ports `ggml_compute_forward_get_rel_pos` (ops.cpp:10168 @c1d0e7a00).
pub export fn ggml_compute_forward_get_rel_pos(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F16, c.GGML_TYPE_BF16 => getRelPosF16(params, dst),
        else => impl.abort("fatal error"),
    }
}

/// Ports `ggml_compute_forward_add_rel_pos_f32` (ops.cpp:10189 @c1d0e7a00).
fn addRelPosF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const src2 = impl.one(Tensor, dst.src[2]);

    const inplace = impl.getOpParamsI32(dst, 0) != 0;
    if (!inplace) {
        if (params.ith == 0) {
            const n = c.ggml_nbytes(dst);
            const dd: [*]u8 = @ptrCast(dst.data.?);
            const ss: [*]const u8 = @ptrCast(src0.data.?);
            @memcpy(dd[0..n], ss[0..n]);
        }
        threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    const src1_data: [*]const f32 = @ptrCast(@alignCast(src1.data.?));
    const src2_data: [*]const f32 = @ptrCast(@alignCast(src2.data.?));
    const dst_data: [*]f32 = @ptrCast(@alignCast(dst.data.?));

    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // total patches in dst
    const np = ne13;

    // patches per thread
    const dp = @divTrunc(np + nth - 1, nth);

    // patch range for this thread
    const ip0 = dp * ith;
    const ip1 = @min(ip0 + dp, np);

    var j13: i64 = ip0;
    while (j13 < ip1) : (j13 += 1) {
        var j12: i64 = 0;
        while (j12 < ne12) : (j12 += 1) {
            var j11: i64 = 0;
            while (j11 < ne11) : (j11 += 1) {
                const jp1 = j13 * ne12 * ne11 * ne10 + j12 * ne11 * ne10 + j11 * ne10;
                var j10: i64 = 0;
                while (j10 < ne10) : (j10 += 1) {
                    const jp0 = jp1 + j10;
                    const src1_e = src1_data[@intCast(jp0)];
                    const src2_e = src2_data[@intCast(jp0)];

                    const jdh = jp0 * ne10;
                    const jdw = jdh - (ne10 - 1) * j10;

                    var jj: i64 = 0;
                    while (jj < ne10) : (jj += 1) {
                        dst_data[@intCast(jdh + jj)] += src2_e;
                        dst_data[@intCast(jdw + jj * ne10)] += src1_e;
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_add_rel_pos` (ops.cpp:10250 @c1d0e7a00).
pub export fn ggml_compute_forward_add_rel_pos(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => addRelPosF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
