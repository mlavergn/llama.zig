//! `argsort`, `top_k`, `tri`, `fill` and `fwht`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! The grouping is ours. `argsort` and `top_k` sit together in the C; `fill`
//! and `tri` sit among the activations, and `fwht` near the end of the file.
//! They are gathered here because none of them computes anything but an
//! ordering, a mask, a constant or a butterfly.
//!
//! # Where the C's own answer is unspecified
//!
//! `argsort` calls `std::sort` and `top_k` calls `std::partial_sort`. Neither
//! is stable, so **the order the C produces among equal keys is whatever
//! libc++'s introsort happens to do**, and libstdc++ would do something else.
//! That is the category `CLAUDE.md` records under "Where the C's own answer is
//! unspecified", and it gets the same treatment: the comparator is reproduced
//! exactly, the algorithm is a stable sort, and ties therefore resolve by
//! index. Output depends only on input. On keys without ties — which is what
//! `harness/ops_dump.c` feeds it — the two agree exactly.
//!
//! # Loop index names
//!
//! `i01`, `i02`, `i03`, `i11`, `i12`, `i13` are Zig integer type names. Renamed
//! `j01`, `j02`, `j03`, `j11`, `j12`, `j13`, digit for digit.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

extern fn sqrtf(x: f32) f32;

inline fn off(i: i64, nb: usize) usize {
    return @as(usize, @intCast(i)) * nb;
}

// -----------------------------------------------------------------------------
// fill

/// Ports `ggml_compute_forward_fill_f32` and `ggml_compute_forward_fill_f16`
/// (ops.cpp:2230, 2249 @c1d0e7a00).
///
/// The two differ only in the element type and which `ggml_vec_set_*` they
/// call. The `f16` constant is narrowed once, before the loop, as the C does.
fn fillTyped(comptime T: type, params: *const ComputeParams, dst: *Tensor) void {
    const value = impl.getOpParamsF32(dst, 0);
    const cval: T = if (T == f32) value else impl.fp32ToFp16(value);

    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const nb1 = dst.nb[1];
    const nb2 = dst.nb[2];
    const nb3 = dst.nb[3];

    const ir0, const ir1 = common.getThreadRange(params, dst);

    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, ne2 * ne1);
        const j02 = @divTrunc(ir - j03 * ne2 * ne1, ne1);
        const j01 = ir - j03 * ne2 * ne1 - j02 * ne1;

        const dst_ptr: [*]T = @ptrCast(@alignCast(dd + off(j03, nb3) + off(j02, nb2) + off(j01, nb1)));

        if (T == f32) vec.set_f32(ne0, dst_ptr, cval) else vec.set_f16(ne0, dst_ptr, cval);
    }
}

/// Ports `ggml_compute_forward_fill` (ops.cpp:2268 @c1d0e7a00).
pub export fn ggml_compute_forward_fill(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => fillTyped(f32, params, dst),
        c.GGML_TYPE_F16 => fillTyped(c.ggml_fp16_t, params, dst),
        else => impl.abort("unsupported type for ggml_compute_forward_fill"),
    }
}

// -----------------------------------------------------------------------------
// tri

/// Ports `ggml_compute_forward_tri_f32` (ops.cpp:2289 @c1d0e7a00).
///
/// The C picks one of four lambdas into a function pointer before the loop;
/// here the switch is on the row/column comparison itself, which is the same
/// predicate per element.
fn triF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    const ttype = impl.getOpParamsI32(dst, 0);

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(src0)");

    const l = common.UnaryLocals.of(src0, dst);

    const ir0, const ir1 = common.getThreadRange(params, src0);

    switch (ttype) {
        c.GGML_TRI_TYPE_LOWER, c.GGML_TRI_TYPE_LOWER_DIAG, c.GGML_TRI_TYPE_UPPER, c.GGML_TRI_TYPE_UPPER_DIAG => {},
        else => impl.abort("invalid tri type"),
    }

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        const src_ptr: [*]const f32 = @ptrCast(@alignCast(s0 + off(j03, l.nb03) + off(j02, l.nb02) + off(j01, l.nb01)));
        const dst_ptr: [*]f32 = @ptrCast(@alignCast(dd + off(j03, l.nb3) + off(j02, l.nb2) + off(j01, l.nb1)));

        // The C's lambdas take `int`; the comparison is the same in `i64`.
        var j0: i64 = 0;
        while (j0 < l.ne0) : (j0 += 1) {
            const keep = switch (ttype) {
                c.GGML_TRI_TYPE_LOWER => j0 < j01,
                c.GGML_TRI_TYPE_LOWER_DIAG => j0 <= j01,
                c.GGML_TRI_TYPE_UPPER => j0 > j01,
                else => j0 >= j01,
            };
            const k: usize = @intCast(j0);
            dst_ptr[k] = if (keep) src_ptr[k] else 0.0;
        }
    }
}

/// Ports `ggml_compute_forward_tri` (ops.cpp:2324 @c1d0e7a00).
pub export fn ggml_compute_forward_tri(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => triF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// argsort

/// Ports `cmp_argsort` (ops.cpp:8339 @c1d0e7a00), the comparator.
///
/// `template<enum ggml_sort_order order>` becomes a comptime `bool`. A strict
/// `<` or `>` on the referenced floats, exactly as the C compares.
fn CmpArgsort(comptime ascending: bool) type {
    return struct {
        data: [*]const f32,

        fn lessThan(self: @This(), a: i32, b: i32) bool {
            const va = self.data[@intCast(a)];
            const vb = self.data[@intCast(b)];
            return if (ascending) va < vb else va > vb;
        }
    };
}

/// Ports `ggml_compute_forward_argsort_f32` (ops.cpp:8350 @c1d0e7a00).
///
/// `std::sort` becomes a stable sort; see the file header for why that is the
/// right answer to an unstable sort's ties.
fn argsortF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const order = impl.getOpParamsI32(dst, 0);

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);
    const n: usize = @intCast(l.ne0);

    var i: i64 = ith;
    while (i < nr) : (i += nth) {
        const src_data: [*]const f32 = @ptrCast(@alignCast(s0 + off(i, l.nb01)));
        const dst_data: [*]i32 = @ptrCast(@alignCast(dd + off(i, l.nb1)));

        for (0..n) |j| dst_data[j] = @intCast(j);

        switch (order) {
            c.GGML_SORT_ORDER_ASC => {
                const Cmp = CmpArgsort(true);
                std.mem.sort(i32, dst_data[0..n], Cmp{ .data = src_data }, Cmp.lessThan);
            },
            c.GGML_SORT_ORDER_DESC => {
                const Cmp = CmpArgsort(false);
                std.mem.sort(i32, dst_data[0..n], Cmp{ .data = src_data }, Cmp.lessThan);
            },
            else => impl.abort("invalid sort order"),
        }
    }
}

/// Ports `ggml_compute_forward_argsort` (ops.cpp:8391 @c1d0e7a00).
pub export fn ggml_compute_forward_argsort(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => argsortF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// top_k

/// Ports `ggml_compute_forward_top_k_f32` (ops.cpp:8418 @c1d0e7a00).
///
/// `std::partial_sort` with `cmp_top_k` (ops.cpp:8411 @c1d0e7a00), a strict `>`, leaves
/// the `top_k` largest keys at the front in descending order. A stable sort of
/// the whole row leaves exactly the same prefix whenever the keys are
/// distinct; among equal keys it keeps index order, where libc++'s heap-based
/// `partial_sort` keeps something unspecified.
///
/// The final swap of the first two is the C's, there to make a caller that
/// relies on the order fail loudly.
fn topKF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    const l = common.UnaryLocals.of(src0, dst);

    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const top_k: usize = @intCast(l.ne0);
    const n: usize = @intCast(l.ne00);

    const wbase: [*]i32 = @ptrCast(@alignCast(params.wdata.?));
    const tmp = wbase + (n + common.cache_line_size_f32) * @as(usize, @intCast(ith));

    const s0: [*]const u8 = @ptrCast(src0.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);

    const Cmp = CmpArgsort(false);

    var i: i64 = ith;
    while (i < nr) : (i += nth) {
        const src_data: [*]const f32 = @ptrCast(@alignCast(s0 + off(i, l.nb01)));

        for (0..n) |j| tmp[j] = @intCast(j);

        std.mem.sort(i32, tmp[0..n], Cmp{ .data = src_data }, Cmp.lessThan);

        const dst_data: [*]i32 = @ptrCast(@alignCast(dd + off(i, l.nb1)));

        @memcpy(dst_data[0..top_k], tmp[0..top_k]);

        // emphasize that the order is not important
        if (top_k > 1) {
            std.mem.swap(i32, &dst_data[0], &dst_data[1]);
        }
    }
}

/// Ports `ggml_compute_forward_top_k` (ops.cpp:8457 @c1d0e7a00).
pub export fn ggml_compute_forward_top_k(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => topKF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// fwht

/// `GGML_F32_EPR` (simd-mappings.h:338 @c1d0e7a00), the NEON arm.
const f32_epr: i64 = 4;

/// Ports `ggml_compute_forward_fwht_f32` (ops.cpp:11847 @c1d0e7a00), the
/// `GGML_SIMD` arm without SVE.
///
/// The C runs the butterflies narrower than `GGML_F32_EPR` as scalars and the
/// rest four lanes at a time. The vector `u - v` is
/// `GGML_F32_VEC_FMA(u, v, -1)`, a fused `u + v * -1` — and `v * -1` is exact,
/// so it is bit-for-bit `u - v`. Both phases are therefore the same
/// arithmetic, and the split is kept only so the loop reads like the C.
fn fwhtF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    const l = common.BinaryLocals.of(src0, src1, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const n = l.ne10;
    impl.assert((n & (n - 1)) == 0, "(n & (n - 1)) == 0");

    const nr = l.ne11 * l.ne12 * l.ne13;
    const rows_per_thread = @divTrunc(nr + nth - 1, nth);
    const start_row = ith * rows_per_thread;
    const end_row = @min(start_row + rows_per_thread, nr);

    const scale: f32 = 1.0 / sqrtf(@floatFromInt(n));

    const s1: [*]const u8 = @ptrCast(src1.data.?);
    const dd: [*]u8 = @ptrCast(dst.data.?);
    const nn: usize = @intCast(n);

    var r: i64 = start_row;
    while (r < end_row) : (r += 1) {
        const j13 = @divTrunc(r, l.ne11 * l.ne12);
        const j12 = @divTrunc(r - j13 * l.ne11 * l.ne12, l.ne11);
        const j11 = r - j13 * l.ne11 * l.ne12 - j12 * l.ne11;

        const src_row: [*]const f32 = @ptrCast(@alignCast(s1 + off(j11, l.nb11) + off(j12, l.nb12) + off(j13, l.nb13)));
        const dst_row: [*]f32 = @ptrCast(@alignCast(dd + off(j11, l.nb1) + off(j12, l.nb2) + off(j13, l.nb3)));

        for (0..nn) |j| dst_row[j] = src_row[j] * scale;

        // Scalar passes
        var len: usize = 1;
        while (len < f32_epr and len < nn) : (len <<= 1) butterfly(dst_row, nn, len);

        // SIMD passes
        while (len < nn) : (len <<= 1) butterfly(dst_row, nn, len);
    }
}

/// One radix-2 pass of `fwhtF32` at span `len`.
inline fn butterfly(row: [*]f32, n: usize, len: usize) void {
    var i: usize = 0;
    while (i < n) : (i += 2 * len) {
        for (0..len) |j| {
            const u = row[i + j];
            const v = row[i + len + j];
            row[i + j] = u + v;
            row[i + len + j] = u - v;
        }
    }
}

/// Ports `ggml_compute_forward_fwht` (ops.cpp:11923 @c1d0e7a00).
pub export fn ggml_compute_forward_fwht(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src1 = impl.one(Tensor, dst.src[1]);

    switch (src1.type) {
        c.GGML_TYPE_F32 => fwhtF32(params, dst),
        else => impl.abort("fatal error - fwht is F32 only"),
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the argsort comparator is strict, so ties resolve by index under a stable sort" {
    const data = [_]f32{ 1, 3, 3, 2 };
    var idx = [_]i32{ 0, 1, 2, 3 };
    const Desc = CmpArgsort(false);
    std.mem.sort(i32, &idx, Desc{ .data = &data }, Desc.lessThan);
    try std.testing.expectEqual([4]i32{ 1, 2, 3, 0 }, idx);

    idx = .{ 0, 1, 2, 3 };
    const Asc = CmpArgsort(true);
    std.mem.sort(i32, &idx, Asc{ .data = &data }, Asc.lessThan);
    try std.testing.expectEqual([4]i32{ 0, 3, 1, 2 }, idx);
}

test "the butterfly is an unnormalised Walsh-Hadamard transform" {
    var row = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    var len: usize = 1;
    while (len < 8) : (len <<= 1) butterfly(&row, 8, len);
    try std.testing.expectEqual([_]f32{1} ** 8, row);
}
