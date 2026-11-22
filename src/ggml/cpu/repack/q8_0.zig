//! The `q8_0` × `q8_0` interleaved gemv and gemm, generic forms.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # The same skeleton as `q4_0.zig`, without the nibble split
//!
//! `q8_0` quants are whole bytes, so the inner loop reads one value rather
//! than two packed ones and there is no `>> 4` on the sum. The step is
//! `qk / blocklen` where `q4_0`'s is `qk / (2 * blocklen)`, for the same
//! reason: half as many values per byte.
//!
//! Only the `4x4` and `4x8` shapes exist here; there is no `8x8`.

const ggml = @import("../../module.zig");

const QK8_0 = 32;

/// Ports `ggml_gemv_q8_0_4x4_q8_0_generic` and
/// `ggml_gemv_q8_0_4x8_q8_0_generic`
/// (ggml-cpu/repack.cpp:1274, 1321 @b1b770d90).
fn gemv(
    comptime ncols_interleaved: i32,
    comptime blocklen: i32,
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) void {
    _ = bs;
    const qk: i32 = QK8_0;
    const nb: i32 = @divTrunc(n, qk);

    ggml.impl.assert(nr == 1, "nr == 1");
    ggml.impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    ggml.impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [ncols_interleaved]f32 = undefined;

    const a_ptr: [*]const ggml.impl.ggml_c.block_q8_0 = @ptrCast(@alignCast(vy));
    var x: i32 = 0;
    while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
        const b_ptr: [*]const ggml.cpu.repack.blocks.block_q8_0x4 =
            @as([*]const ggml.cpu.repack.blocks.block_q8_0x4, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));

        for (0..ncols_interleaved) |j| sumf[j] = 0.0;
        var l: i32 = 0;
        while (l < nb) : (l += 1) {
            const bl = &b_ptr[@intCast(l)];
            const al = &a_ptr[@intCast(l)];
            var k: i32 = 0;
            while (k < @divTrunc(qk, blocklen)) : (k += 1) {
                for (0..ncols_interleaved) |j| {
                    var sumi: i32 = 0;
                    var i: i32 = 0;
                    while (i < blocklen) : (i += 1) {
                        const v0: i32 = bl.qs[@intCast(k * ncols_interleaved * blocklen + @as(i32, @intCast(j)) * blocklen + i)];
                        sumi += v0 * al.qs[@intCast(k * blocklen + i)];
                    }
                    sumf[j] = ggml.cpu.repack.epilogue.accumulate(sumf[j], sumi, ggml.cpu.convert.cpuFp16ToFp32(bl.d[j]), ggml.cpu.convert.cpuFp16ToFp32(al.d));
                }
            }
        }
        for (0..ncols_interleaved) |j| s[@intCast(x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[j];
    }
}

/// Ports `ggml_gemm_q8_0_4x4_q8_0_generic` and
/// `ggml_gemm_q8_0_4x8_q8_0_generic`
/// (ggml-cpu/repack.cpp:2407, 2461 @b1b770d90).
fn gemm(
    comptime ncols_interleaved: i32,
    comptime blocklen: i32,
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) void {
    const qk: i32 = QK8_0;
    const nb: i32 = @divTrunc(n, qk);

    ggml.impl.assert(@rem(n, qk) == 0, "n % qk == 0");
    ggml.impl.assert(@rem(nr, 4) == 0, "nr % 4 == 0");
    ggml.impl.assert(@rem(nc, ncols_interleaved) == 0, "nc % ncols_interleaved == 0");

    var sumf: [4][ncols_interleaved]f32 = undefined;

    var y: i32 = 0;
    while (y < @divTrunc(nr, 4)) : (y += 1) {
        const a_ptr: [*]const ggml.cpu.repack.blocks.block_q8_0x4 =
            @as([*]const ggml.cpu.repack.blocks.block_q8_0x4, @ptrCast(@alignCast(vy))) + @as(usize, @intCast(y * nb));
        var x: i32 = 0;
        while (x < @divTrunc(nc, ncols_interleaved)) : (x += 1) {
            const b_ptr: [*]const ggml.cpu.repack.blocks.block_q8_0x4 =
                @as([*]const ggml.cpu.repack.blocks.block_q8_0x4, @ptrCast(@alignCast(vx))) + @as(usize, @intCast(x * nb));
            for (0..4) |m| {
                for (0..ncols_interleaved) |j| sumf[m][j] = 0.0;
            }
            var l: i32 = 0;
            while (l < nb) : (l += 1) {
                const bl = &b_ptr[@intCast(l)];
                const al = &a_ptr[@intCast(l)];
                var k: i32 = 0;
                while (k < @divTrunc(qk, blocklen)) : (k += 1) {
                    for (0..4) |m| {
                        for (0..ncols_interleaved) |j| {
                            var sumi: i32 = 0;
                            var i: i32 = 0;
                            while (i < blocklen) : (i += 1) {
                                const v0: i32 = bl.qs[@intCast(k * ncols_interleaved * blocklen + @as(i32, @intCast(j)) * blocklen + i)];
                                sumi += v0 * al.qs[@intCast(k * 4 * blocklen + @as(i32, @intCast(m)) * blocklen + i)];
                            }
                            sumf[m][j] = ggml.cpu.repack.epilogue.accumulate(sumf[m][j], sumi, ggml.cpu.convert.cpuFp16ToFp32(bl.d[j]), ggml.cpu.convert.cpuFp16ToFp32(al.d[m]));
                        }
                    }
                }
            }
            for (0..4) |m| {
                for (0..ncols_interleaved) |j| {
                    s[@intCast((y * 4 + @as(i32, @intCast(m))) * @as(i32, @intCast(bs)) + x * ncols_interleaved + @as(i32, @intCast(j)))] = sumf[m][j];
                }
            }
        }
    }
}

pub fn ggml_gemv_q8_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(4, 4, n, s, bs, vx, vy, nr, nc);
}
pub fn ggml_gemv_q8_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemv(4, 8, n, s, bs, vx, vy, nr, nc);
}
pub fn ggml_gemm_q8_0_4x4_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(4, 4, n, s, bs, vx, vy, nr, nc);
}
pub fn ggml_gemm_q8_0_4x8_q8_0_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    gemm(4, 8, n, s, bs, vx, vy, nr, nc);
}
