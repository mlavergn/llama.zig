//! Matrix multiplication on the CPU, plain and expert-routed.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (v0.3.0, `c1d0e7a00`),
//! the section beginning at line 1162. Each function names the C it replaces
//! and the line it began at.
//!
//! # What this file does and does not compute
//!
//! Neither function multiplies anything itself. Both quantise the right-hand
//! operand into the type the kernel wants, carve the result into chunks, hand
//! each chunk to the `vec_dot` from `type_traits_cpu`, and coordinate the
//! threads doing it. The arithmetic is in `quants.c` and `vec.cpp`.
//!
//! So the risk here is not rounding, it is indexing: a stride computed from
//! the wrong `nb`, or a chunk boundary off by one, silently reads the wrong
//! row. `test-backend-ops` exercises this against Metal across 21,093
//! configurations, which is the gate that catches it.
//!
//! # The C's index names are one letter off here
//!
//! Zig reserves `i11`, `i12`, `i13`, `i1`, `i2` and `i3` as integer type
//! names, and the C uses every one of them as a loop index. They are spelled
//! `j11`, `j12`, `j13` and so on below, digit for digit, so the arithmetic can
//! still be read against the C line by line.
//!
//! # The llamafile fast path
//!
//! Two blocks try `llamafile_sgemm` before the generic loop, and both use a
//! `goto` to fall through when it declines a shape. The `goto` becomes a
//! labelled loop with an `ok` flag: the C returns only when *every* call
//! succeeded, and a partial success still falls through to recompute the lot.

const std = @import("std");
const impl = @import("../impl.zig");
const types = @import("../types.zig");
const defs = @import("defs.zig");
const traits = @import("traits.zig");
const features = @import("features.zig");
const threading = @import("threading.zig");
const c = impl.c;

const Tensor = defs.Tensor;
const ComputeParams = defs.ComputeParams;

/// The op kernels this file falls back to, still C++ in `ggml-cpu/ops.cpp`.
extern fn ggml_compute_forward_fwht(params: *const ComputeParams, dst: *Tensor) void;

/// `ggml-cpu/llamafile/sgemm.cpp`, still C++.
///
/// Returns false when it has no kernel for the shape or type combination, in
/// which case the caller must compute the product itself.
extern fn llamafile_sgemm(
    params: *const ComputeParams,
    m: i64,
    n: i64,
    k: i64,
    a: ?*const anyopaque,
    lda: i64,
    b: ?*const anyopaque,
    ldb: i64,
    cc: ?*anyopaque,
    ldc: i64,
    atype: c_int,
    btype: c_int,
    ctype: c_int,
) bool;

inline fn min(a: i64, b: i64) i64 {
    return if (a < b) a else b;
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_mul_mat

/// Ports `ggml_compute_forward_mul_mat_one_chunk` (ggml-cpu.c:1164 @c1d0e7a00).
///
/// Computes the rectangle `[ir0_start, ir0_end) x [ir1_start, ir1_end)` of the
/// result, tiling 16x16 so a tile of `dst` stays in cache across the inner
/// loop.
///
/// Parameters:
/// - `params`: the thread's slice of the work buffer.
/// - `dst`: the result tensor; its `src[0]` and `src[1]` are the operands.
/// - `@"type"`: `src0`'s type, which selects the kernel.
/// - `num_rows_per_vec_dot`: 1, or 2 when an i8mm kernel takes two rows.
/// - `ir0_start`, `ir0_end`: row range within `dst`'s first dimension.
/// - `ir1_start`, `ir1_end`: flattened range over `dst`'s other dimensions.
fn oneChunk(
    params: *const ComputeParams,
    dst: *Tensor,
    @"type": c.enum_ggml_type,
    num_rows_per_vec_dot: i64,
    ir0_start: i64,
    ir0_end: i64,
    ir1_start: i64,
    ir1_end: i64,
) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = defs.BinaryLocals.of(src0, src1, dst);

    const src1_cont = types.ggml_is_contiguous(src1);

    const vec_dot = traits.table[@intCast(@"type")].vec_dot.?;
    const vec_dot_type = traits.table[@intCast(@"type")].vec_dot_type;

    // Broadcast factors.
    const r2 = @divTrunc(l.ne12, l.ne02);
    const r3 = @divTrunc(l.ne13, l.ne03);

    // Threads with no work simply yield.
    if (ir0_start >= ir0_end or ir1_start >= ir1_end) return;

    const wdata: [*]const u8 = if (src1.type == vec_dot_type)
        @ptrCast(src1.data.?)
    else
        @ptrCast(params.wdata.?);
    const row_size = types.ggml_row_size(vec_dot_type, l.ne10);

    std.debug.assert(@rem(l.ne12, l.ne02) == 0);
    std.debug.assert(@rem(l.ne13, l.ne03) == 0);

    // Block-tiling attempt.
    const blck_0: i64 = 16;
    const blck_1: i64 = 16;

    const src1_col_stride = if (src1_cont or src1.type != vec_dot_type) row_size else l.nb11;

    // 16 * 2, accounting for mmla kernels. The C notes this is an attempt to
    // reduce false sharing that did not measurably help; kept as written.
    var tmp: [32]f32 = undefined;

    var iir1 = ir1_start;
    while (iir1 < ir1_end) : (iir1 += blck_1) {
        var iir0 = ir0_start;
        while (iir0 < ir0_end) : (iir0 += blck_0) {
            var ir1 = iir1;
            while (ir1 < iir1 + blck_1 and ir1 < ir1_end) : (ir1 += num_rows_per_vec_dot) {
                const j13 = @divTrunc(ir1, l.ne12 * l.ne1);
                const j12 = @divTrunc(ir1 - j13 * l.ne12 * l.ne1, l.ne1);
                const j11 = ir1 - j13 * l.ne12 * l.ne1 - j12 * l.ne1;

                // Broadcast src0 into src1.
                const j03 = @divTrunc(j13, r3);
                const j02 = @divTrunc(j12, r2);

                const src0_row: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) +
                    @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;

                // When src1 is not one contiguous block the offset has to come
                // from the strides. When it is, the data has either been made
                // contiguous in `wdata` or is being read in place, so the
                // indices can be used directly.
                const src1_col: [*]const u8 = wdata + if (src1_cont or src1.type != vec_dot_type)
                    @as(usize, @intCast(j11 + j12 * l.ne11 + j13 * l.ne12 * l.ne11)) * row_size
                else
                    @as(usize, @intCast(j11)) * l.nb11 + @as(usize, @intCast(j12)) * l.nb12 +
                        @as(usize, @intCast(j13)) * l.nb13;

                const dst_col: [*]f32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(dst.data.?)) +
                    @as(usize, @intCast(j11)) * l.nb1 + @as(usize, @intCast(j12)) * l.nb2 +
                    @as(usize, @intCast(j13)) * l.nb3));

                const wide = num_rows_per_vec_dot > 1;

                var ir0 = iir0;
                while (ir0 < iir0 + blck_0 and ir0 < ir0_end) : (ir0 += num_rows_per_vec_dot) {
                    vec_dot(
                        @intCast(l.ne00),
                        &tmp[@intCast(ir0 - iir0)],
                        if (wide) 16 else 0,
                        src0_row + @as(usize, @intCast(ir0)) * l.nb01,
                        if (wide) l.nb01 else 0,
                        src1_col,
                        if (wide) src1_col_stride else 0,
                        @intCast(num_rows_per_vec_dot),
                    );
                }

                const count: usize = @intCast(min(iir0 + blck_0, ir0_end) - iir0);
                var cn: i64 = 0;
                while (cn < num_rows_per_vec_dot) : (cn += 1) {
                    const at = @as(usize, @intCast(iir0)) + @as(usize, @intCast(cn)) * l.nb1 / l.nb0;
                    @memcpy(
                        dst_col[at .. at + count],
                        tmp[@intCast(cn * 16)..][0..count],
                    );
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_mul_mat` (ggml-cpu.c:1254 @c1d0e7a00).
///
/// Parameters:
/// - `params`: this thread's index, count, and work buffer.
/// - `dst`: the result; `src[0]` is the weight matrix and `src[1]` the
///   activations.
pub export fn ggml_compute_forward_mul_mat(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    // A Hadamard-structured src0 has a fast transform that beats the general
    // product outright, unless the caller asked for the reference path.
    const hint = impl.getOpParamsI32(dst, 1);
    if (hint == c.GGML_HINT_SRC0_IS_HADAMARD and !params.use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    const l = defs.BinaryLocals.of(src0, src1, dst);

    const ith = params.ith;
    const nth = params.nth;

    const vec_dot_type = traits.table[@intCast(src0.type)].vec_dot_type;
    const from_float = traits.table[@intCast(vec_dot_type)].from_float;
    const vec_dot_num_rows = traits.table[@intCast(src0.type)].nrows;

    impl.assert(l.ne0 == l.ne01, "ne0 == ne01");
    impl.assert(l.ne1 == l.ne11, "ne1 == ne11");
    impl.assert(l.ne2 == l.ne12, "ne2 == ne12");
    impl.assert(l.ne3 == l.ne13, "ne3 == ne13");

    // Permuted src0 or src1 is not supported.
    impl.assert(l.nb00 == types.ggml_type_size(src0.type), "nb00 == ggml_type_size(src0->type)");
    impl.assert(l.nb10 == types.ggml_type_size(src1.type), "nb10 == ggml_type_size(src1->type)");

    // dst cannot be transposed or permuted.
    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(l.nb0 <= l.nb1, "nb0 <= nb1");
    impl.assert(l.nb1 <= l.nb2, "nb1 <= nb2");
    impl.assert(l.nb2 <= l.nb3, "nb2 <= nb3");

    const r2 = @divTrunc(l.ne12, l.ne02);
    const r3 = @divTrunc(l.ne13, l.ne03);
    const src1_cont = types.ggml_is_contiguous(src1);

    // First llamafile attempt: src1 is already contiguous and already the
    // type the kernel wants, so nothing needs staging.
    if (features.use_llamafile and src1_cont) {
        var ok = true;
        outer: for (0..@intCast(l.ne13)) |j13| {
            for (0..@intCast(l.ne12)) |j12| {
                if (!llamafile_sgemm(
                    params,
                    l.ne01,
                    l.ne11,
                    @divTrunc(l.ne00, types.ggml_blck_size(src0.type)),
                    @as([*]const u8, @ptrCast(src0.data.?)) +
                        @as(usize, j12) / @as(usize, @intCast(r2)) * l.nb02 +
                        @as(usize, j13) / @as(usize, @intCast(r3)) * l.nb03,
                    @intCast(l.nb01 / types.ggml_type_size(src0.type)),
                    @as([*]const u8, @ptrCast(src1.data.?)) + j12 * l.nb12 + j13 * l.nb13,
                    @intCast(l.nb11 / types.ggml_type_size(src1.type)),
                    @as([*]u8, @ptrCast(dst.data.?)) + j12 * l.nb2 + j13 * l.nb3,
                    @intCast(l.nb1 / types.ggml_type_size(dst.type)),
                    @intCast(src0.type),
                    @intCast(src1.type),
                    @intCast(dst.type),
                )) {
                    ok = false;
                    break :outer;
                }
            }
        }
        if (ok) return;
    }

    if (src1.type != vec_dot_type) {
        const wdata: [*]u8 = @ptrCast(params.wdata.?);

        const nbw0 = types.ggml_type_size(vec_dot_type);
        const nbw1 = types.ggml_row_size(vec_dot_type, l.ne10);
        const nbw2 = nbw1 * @as(usize, @intCast(l.ne11));
        const nbw3 = nbw2 * @as(usize, @intCast(l.ne12));

        std.debug.assert(params.wsize >= @as(usize, @intCast(l.ne13)) * nbw3);
        impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

        // Every thread walks every row and converts its own slice of the
        // columns, rather than taking whole rows round-robin. Block-quantised
        // types have to split on a block boundary, which the row-per-thread
        // form cannot express.
        const bs = types.ggml_blck_size(vec_dot_type);
        const block_start = @divTrunc(@as(i64, ith) * @divTrunc(l.ne10, bs), nth);
        const block_end = @divTrunc((@as(i64, ith) + 1) * @divTrunc(l.ne10, bs), nth);

        for (0..@intCast(l.ne13)) |j13| {
            for (0..@intCast(l.ne12)) |j12| {
                for (0..@intCast(l.ne11)) |j11| {
                    const src: [*]const u8 = @as([*]const u8, @ptrCast(src1.data.?)) +
                        j13 * l.nb13 + j12 * l.nb12 + j11 * l.nb11 +
                        @as(usize, @intCast(block_start * bs)) * l.nb10;
                    const dstw: [*]u8 = wdata + j13 * nbw3 + j12 * nbw2 + j11 * nbw1 +
                        @as(usize, @intCast(block_start)) * nbw0;
                    from_float.?(@ptrCast(@alignCast(src)), dstw, (block_end - block_start) * bs);
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at its own index, so the first unclaimed chunk
        // is `nth`. Saves a round of coordination at the start.
        params.threadpool.?.current_chunk.store(nth, .monotonic);
    }

    threading.ggml_barrier(params.threadpool.?);

    // Second llamafile attempt, now that src1 has been staged into `wdata` in
    // the kernel's own type.
    if (features.use_llamafile and src1.type != vec_dot_type) {
        const wdata: [*]const u8 = @ptrCast(params.wdata.?);
        const row_size = types.ggml_row_size(vec_dot_type, l.ne10);

        var ok = true;
        outer: for (0..@intCast(l.ne13)) |j13| {
            for (0..@intCast(l.ne12)) |j12| {
                if (!llamafile_sgemm(
                    params,
                    l.ne01,
                    l.ne11,
                    @divTrunc(l.ne00, types.ggml_blck_size(src0.type)),
                    @as([*]const u8, @ptrCast(src0.data.?)) +
                        @as(usize, j12) / @as(usize, @intCast(r2)) * l.nb02 +
                        @as(usize, j13) / @as(usize, @intCast(r3)) * l.nb03,
                    @intCast(l.nb01 / types.ggml_type_size(src0.type)),
                    wdata + (j12 * @as(usize, @intCast(l.ne11)) +
                        j13 * @as(usize, @intCast(l.ne12 * l.ne11))) * row_size,
                    @intCast(row_size / types.ggml_type_size(vec_dot_type)),
                    @as([*]u8, @ptrCast(dst.data.?)) + j12 * l.nb2 + j13 * l.nb3,
                    @intCast(l.nb1 / types.ggml_type_size(dst.type)),
                    @intCast(src0.type),
                    @intCast(vec_dot_type),
                    @intCast(dst.type),
                )) {
                    ok = false;
                    break :outer;
                }
            }
        }
        if (ok) return;
    }

    // The size of the result's first dimension, which the asserts above have
    // already tied to ne01.
    const nr0 = l.ne0;

    // The size of the rest of the result's dimensions.
    const nr1 = l.ne1 * l.ne2 * l.ne3;

    // A reasonable chunk size, stepped up when one dimension is degenerate.
    var chunk_size: i64 = 16;
    if (nr0 == 1 or nr1 == 1) chunk_size = 64;

    var nchunk0 = @divTrunc(nr0 + chunk_size - 1, chunk_size);
    var nchunk1 = @divTrunc(nr1 + chunk_size - 1, chunk_size);

    // If that chunking is poor for the thread count, scrap it and chunk by
    // thread instead. Chunking by thread also measured better on NUMA -- see
    // ggml-org/llama.cpp#6915 -- which is why NUMA forces this branch.
    if (nchunk0 * nchunk1 < @as(i64, nth) * 4 or threading.ggml_is_numa()) {
        nchunk0 = if (nr0 > nr1) nth else 1; // parallelise by src0 rows
        nchunk1 = if (nr0 > nr1) 1 else nth; // parallelise by src1 rows
    }

    const dr0 = @divTrunc(nr0 + nchunk0 - 1, nchunk0);
    const dr1 = @divTrunc(nr1 + nchunk1 - 1, nchunk1);

    // The first chunk is this thread's index; the rest are claimed as they
    // come free.
    var current_chunk: i64 = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const ith0 = @rem(current_chunk, nchunk0);
        const ith1 = @divTrunc(current_chunk, nchunk0);

        const ir0_start = dr0 * ith0;
        const ir0_end = min(ir0_start + dr0, nr0);

        const ir1_start = dr1 * ith1;
        const ir1_end = min(ir1_start + dr1, nr1);

        // Dot kernels take one row and column at a time; mmla kernels take
        // two. These checks keep a pair from straddling a dim-1 boundary.
        var num_rows_per_vec_dot = vec_dot_num_rows;
        if (@rem(nr0, 2) != 0 or @rem(l.ne11, 2) != 0 or
            @rem(ir0_end - ir0_start, 2) != 0 or @rem(ir1_end - ir1_start, 2) != 0)
        {
            num_rows_per_vec_dot = 1;
        }

        oneChunk(params, dst, src0.type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (@as(i64, nth) >= nchunk0 * nchunk1) break;

        current_chunk = params.threadpool.?.current_chunk.fetchAdd(1, .monotonic);
    }
}

// -----------------------------------------------------------------------------
// ggml_compute_forward_mul_mat_id

/// Ports `struct mmid_row_mapping` (ggml-cpu.c:1458 @c1d0e7a00).
pub const RowMapping = extern struct {
    /// The selected expert's index within the row.
    i1: i32,
    /// The row of `src1` this mapping came from.
    i2: i32,
};

/// Ports the `MMID_MATRIX_ROW` macro (ggml-cpu.c:1456 @c1d0e7a00).
inline fn matrixRow(matrix_rows: [*]RowMapping, ids: *const Tensor, row_id: i64, idx1: i64) *RowMapping {
    return &matrix_rows[@intCast(row_id * ids.ne[0] * ids.ne[1] + idx1)];
}

/// Ports `incr_ptr_aligned` (ggml-cpu.c:1526 @c1d0e7a00).
///
/// Carves a sub-allocation out of the work buffer and advances the cursor.
/// Distinct from the identically named helper in `ggml.c`, which the graph
/// allocator uses.
///
/// Parameters:
/// - `p`: cursor into the work buffer, advanced past the returned block.
/// - `size`: bytes to reserve.
/// - `alignment`: alignment the block must start on.
///
/// Return: the aligned start of the block.
fn incrPtrAligned(p: *[*]u8, size: usize, alignment: usize) [*]u8 {
    const ptr: [*]u8 = @ptrFromInt(impl.pad(@intFromPtr(p.*), alignment));
    p.* = ptr + size;
    return ptr;
}

/// Ports `ggml_compute_forward_mul_mat_id_one_chunk` (ggml-cpu.c:1463 @c1d0e7a00).
///
/// The mul_mat_id counterpart of `oneChunk`. It takes more arguments because
/// the expert grouping is computed once by thread zero and shared, rather than
/// being rederivable from `dst`.
///
/// Parameters:
/// - `dst`, `src0`, `src1`, `ids`: the tensors, `ids` naming the expert per row.
/// - `cur_a`: which expert this chunk belongs to.
/// - `ir0_start`, `ir0_end`: row range within the expert's matrix.
/// - `ir1_start`, `ir1_end`: range within the rows routed to this expert.
/// - `src0_cur`: the expert's slice of `src0`.
/// - `matrix_rows`: the routing table thread zero built.
/// - `row_size`: bytes per staged row of `src1`.
/// - `src1_cont`: whether `src1` is one contiguous block.
/// - `wdata`: where the staged `src1` lives.
fn idOneChunk(
    dst: *Tensor,
    src0: *const Tensor,
    src1: *const Tensor,
    ids: *const Tensor,
    cur_a: i64,
    ir0_start: i64,
    ir0_end: i64,
    ir1_start: i64,
    ir1_end: i64,
    src0_cur: [*]const u8,
    matrix_rows: [*]RowMapping,
    row_size: usize,
    src1_cont: bool,
    wdata: [*]const u8,
) void {
    const l = defs.BinaryLocals.of(src0, src1, dst);

    const @"type" = src0.type;

    const vec_dot = traits.table[@intCast(@"type")].vec_dot.?;
    const vec_dot_type = traits.table[@intCast(@"type")].vec_dot_type;

    const blck_0: i64 = 16;
    const blck_1: i64 = 16;

    var tmp: [16]f32 = undefined;

    var iir1 = ir1_start;
    while (iir1 < ir1_end) : (iir1 += blck_1) {
        var iir0 = ir0_start;
        while (iir0 < ir0_end) : (iir0 += blck_0) {
            var ir1 = iir1;
            while (ir1 < iir1 + blck_1 and ir1 < ir1_end) : (ir1 += 1) {
                // The logical row index for this expert.
                const row_mapping = matrixRow(matrix_rows, ids, cur_a, ir1).*;
                const id: i64 = row_mapping.i1; // selected expert index

                const j11 = @rem(id, l.ne11);
                const j12: i64 = row_mapping.i2; // row index in src1

                const src1_col: [*]const u8 = wdata + if (src1_cont or src1.type != vec_dot_type)
                    @as(usize, @intCast(j11 + j12 * l.ne11)) * row_size
                else
                    @as(usize, @intCast(j11)) * l.nb11 + @as(usize, @intCast(j12)) * l.nb12;

                const dst_col: [*]f32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(dst.data.?)) +
                    @as(usize, @intCast(id)) * l.nb1 + @as(usize, @intCast(j12)) * l.nb2));

                var ir0 = iir0;
                while (ir0 < iir0 + blck_0 and ir0 < ir0_end) : (ir0 += 1) {
                    vec_dot(
                        @intCast(l.ne00),
                        &tmp[@intCast(ir0 - iir0)],
                        0,
                        src0_cur + @as(usize, @intCast(ir0)) * l.nb01,
                        0,
                        src1_col,
                        0,
                        1,
                    );
                }

                const count: usize = @intCast(min(iir0 + blck_0, ir0_end) - iir0);
                const at: usize = @intCast(iir0);
                @memcpy(dst_col[at .. at + count], tmp[0..count]);
            }
        }
    }
}

/// Ports `ggml_compute_forward_mul_mat_id` (ggml-cpu.c:1534 @c1d0e7a00).
///
/// A mixture-of-experts product: `src[2]` names, per row of `src1`, which
/// matrix in `src0`'s third dimension that row should be multiplied by.
///
/// Parameters:
/// - `params`: this thread's index, count, and work buffer.
/// - `dst`: the result; `src[0]` is the stack of experts, `src[1]` the
///   activations, `src[2]` the routing table.
pub fn computeForwardMulMatId(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);
    const ids = impl.one(Tensor, dst.src[2]);

    const l = defs.BinaryLocals.of(src0, src1, dst);

    const ith = params.ith;
    const nth = params.nth;

    const @"type" = src0.type;

    const src1_cont = types.ggml_is_contiguous(src1);

    const vec_dot_type = traits.table[@intCast(@"type")].vec_dot_type;
    const from_float = traits.table[@intCast(vec_dot_type)].from_float;

    // Permuted src0 or src1 is not supported.
    impl.assert(l.nb00 == types.ggml_type_size(@"type"), "nb00 == ggml_type_size(type)");
    impl.assert(l.nb10 == types.ggml_type_size(src1.type), "nb10 == ggml_type_size(src1->type)");

    // dst cannot be transposed or permuted.
    impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(l.nb0 <= l.nb1, "nb0 <= nb1");
    impl.assert(l.nb1 <= l.nb2, "nb1 <= nb2");
    impl.assert(l.nb2 <= l.nb3, "nb2 <= nb3");

    const n_ids = ids.ne[0]; // experts used per token
    const n_as = l.ne02; // experts available

    // The work buffer holds four things back to back: the staged src1, a
    // per-expert row count, the routing table, and one cache line per expert
    // for its chunk counter. `ggml_graph_plan` sizes it to match, so the
    // layout here and the arithmetic there have to stay in step.
    var wdata_cur: [*]u8 = @ptrCast(params.wdata.?);

    if (src1.type != vec_dot_type) {
        _ = incrPtrAligned(
            &wdata_cur,
            types.ggml_row_size(vec_dot_type, types.ggml_nelements(src1)),
            @sizeOf(i64),
        );
    }

    const matrix_row_counts: [*]i64 = @ptrCast(@alignCast(incrPtrAligned(
        &wdata_cur,
        @as(usize, @intCast(n_as)) * @sizeOf(i64),
        @sizeOf(i64),
    )));

    const matrix_rows: [*]RowMapping = @ptrCast(@alignCast(incrPtrAligned(
        &wdata_cur,
        @as(usize, @intCast(n_as * ids.ne[0] * ids.ne[1])) * @sizeOf(RowMapping),
        @sizeOf(i64),
    )));

    const atomic_current_chunk = incrPtrAligned(
        &wdata_cur,
        defs.cache_line_size * @as(usize, @intCast(n_as)),
        defs.cache_line_size,
    );

    impl.assert(
        params.wsize >= @intFromPtr(wdata_cur) - @intFromPtr(params.wdata.?),
        "params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata)",
    );

    if (src1.type != vec_dot_type) {
        const wdata: [*]u8 = @ptrCast(params.wdata.?);

        const nbw0 = types.ggml_type_size(vec_dot_type);
        const nbw1 = types.ggml_row_size(vec_dot_type, l.ne10);
        const nbw2 = nbw1 * @as(usize, @intCast(l.ne11));
        const nbw3 = nbw2 * @as(usize, @intCast(l.ne12));

        std.debug.assert(params.wsize >= @as(usize, @intCast(l.ne13)) * nbw3);
        impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");

        const bs = types.ggml_blck_size(vec_dot_type);
        const block_start = @divTrunc(@as(i64, ith) * @divTrunc(l.ne10, bs), nth);
        const block_end = @divTrunc((@as(i64, ith) + 1) * @divTrunc(l.ne10, bs), nth);

        for (0..@intCast(l.ne13)) |j13| {
            for (0..@intCast(l.ne12)) |j12| {
                for (0..@intCast(l.ne11)) |j11| {
                    const src: [*]const u8 = @as([*]const u8, @ptrCast(src1.data.?)) +
                        j13 * l.nb13 + j12 * l.nb12 + j11 * l.nb11 +
                        @as(usize, @intCast(block_start * bs)) * l.nb10;
                    const dstw: [*]u8 = wdata + j13 * nbw3 + j12 * nbw2 + j11 * nbw1 +
                        @as(usize, @intCast(block_start)) * nbw0;
                    from_float.?(@ptrCast(@alignCast(src)), dstw, (block_end - block_start) * bs);
                }
            }
        }
    }

    if (ith == 0) {
        @memset(matrix_row_counts[0..@intCast(n_as)], 0);

        // Group the rows by which expert they were routed to.
        var iid1: i64 = 0;
        while (iid1 < ids.ne[1]) : (iid1 += 1) {
            var id: i64 = 0;
            while (id < n_ids) : (id += 1) {
                const p: *align(1) const i32 = @ptrCast(@as([*]const u8, @ptrCast(ids.data.?)) +
                    @as(usize, @intCast(iid1)) * ids.nb[1] + @as(usize, @intCast(id)) * ids.nb[0]);
                const j02 = p.*;

                std.debug.assert(j02 >= 0 and j02 < n_as);

                matrixRow(matrix_rows, ids, j02, matrix_row_counts[@intCast(j02)]).* = .{
                    .i1 = @intCast(id),
                    .i2 = @intCast(iid1),
                };
                matrix_row_counts[@intCast(j02)] += 1;
            }
        }
    }

    // Reset each expert's chunk counter. One cache line apiece, so threads
    // claiming chunks for different experts do not contend.
    {
        var cur_a: i64 = ith;
        while (cur_a < n_as) : (cur_a += nth) {
            chunkCounter(atomic_current_chunk, cur_a).store(nth, .monotonic);
        }
    }

    threading.ggml_barrier(params.threadpool.?);

    var cur_a: i64 = 0;
    while (cur_a < n_as) : (cur_a += 1) {
        const cne1 = matrix_row_counts[@intCast(cur_a)];
        if (cne1 == 0) continue;

        const src0_cur: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) +
            @as(usize, @intCast(cur_a)) * l.nb02;
        const wdata: [*]const u8 = if (src1.type == vec_dot_type)
            @ptrCast(src1.data.?)
        else
            @ptrCast(params.wdata.?);
        const row_size = types.ggml_row_size(vec_dot_type, l.ne10);

        const nr0 = l.ne01;
        const nr1 = cne1;

        var chunk_size: i64 = 16;
        if (nr0 == 1 or nr1 == 1) chunk_size = 64;

        const disable_chunking = threading.ggml_is_numa();

        var nchunk0 = @divTrunc(nr0 + chunk_size - 1, chunk_size);
        var nchunk1 = @divTrunc(nr1 + chunk_size - 1, chunk_size);

        if (nchunk0 * nchunk1 < @as(i64, nth) * 4 or disable_chunking) {
            nchunk0 = if (nr0 > nr1) nth else 1;
            nchunk1 = if (nr0 > nr1) 1 else nth;
        }

        const dr0 = @divTrunc(nr0 + nchunk0 - 1, nchunk0);
        const dr1 = @divTrunc(nr1 + nchunk1 - 1, nchunk1);

        var current_chunk: i64 = ith;
        const counter = chunkCounter(atomic_current_chunk, cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const ith0 = @rem(current_chunk, nchunk0);
            const ith1 = @divTrunc(current_chunk, nchunk0);

            const ir0_start = dr0 * ith0;
            const ir0_end = min(ir0_start + dr0, nr0);

            const ir1_start = dr1 * ith1;
            const ir1_end = min(ir1_start + dr1, nr1);

            idOneChunk(
                dst,
                src0,
                src1,
                ids,
                cur_a,
                ir0_start,
                ir0_end,
                ir1_start,
                ir1_end,
                src0_cur,
                matrix_rows,
                row_size,
                src1_cont,
                wdata,
            );

            if (@as(i64, nth) >= nchunk0 * nchunk1) break;

            current_chunk = counter.fetchAdd(1, .monotonic);
        }
    }
}

/// The chunk counter for one expert, one cache line into the block reserved
/// for them.
inline fn chunkCounter(base: [*]u8, cur_a: i64) *std.atomic.Value(c_int) {
    return @ptrCast(@alignCast(base + @as(usize, @intCast(cur_a)) * defs.cache_line_size));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the routing table is addressed as the C's macro addresses it" {
    // `MMID_MATRIX_ROW(row_id, i1)` indexes by `ids->ne[0]*ids->ne[1]`, not by
    // the number of rows actually routed. Getting that wrong overlaps two
    // experts' rows, which shows up as a wrong answer rather than a crash.
    var ids: Tensor = std.mem.zeroes(Tensor);
    ids.ne = .{ 2, 5, 1, 1 };

    var rows: [64]RowMapping = @splat(.{ .i1 = 0, .i2 = 0 });

    try std.testing.expectEqual(&rows[0], matrixRow(&rows, &ids, 0, 0));
    try std.testing.expectEqual(&rows[3], matrixRow(&rows, &ids, 0, 3));
    try std.testing.expectEqual(&rows[10], matrixRow(&rows, &ids, 1, 0));
    try std.testing.expectEqual(&rows[23], matrixRow(&rows, &ids, 2, 3));
}

test "incrPtrAligned advances past each block and honours alignment" {
    var buf: [512]u8 align(64) = undefined;
    var p: [*]u8 = &buf;

    const a = incrPtrAligned(&p, 10, 8);
    try std.testing.expectEqual(@intFromPtr(&buf), @intFromPtr(a));

    const b = incrPtrAligned(&p, 4, 8);
    try std.testing.expectEqual(@intFromPtr(&buf) + 16, @intFromPtr(b));

    const line = incrPtrAligned(&p, 64, defs.cache_line_size);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(line) % defs.cache_line_size);
    try std.testing.expectEqual(@intFromPtr(line) + 64, @intFromPtr(p));
}

test "each expert's chunk counter gets its own cache line" {
    var buf: [4 * defs.cache_line_size]u8 align(64) = @splat(0);

    const a = chunkCounter(&buf, 0);
    const b = chunkCounter(&buf, 1);

    try std.testing.expectEqual(defs.cache_line_size, @intFromPtr(b) - @intFromPtr(a));

    a.store(7, .monotonic);
    b.store(9, .monotonic);
    try std.testing.expectEqual(@as(c_int, 7), a.load(.monotonic));
    try std.testing.expectEqual(@as(c_int, 9), b.load(.monotonic));
}
