//! `map_custom1`, `map_custom2`, `map_custom3`, `custom`,
//! `cross_entropy_loss`, `cross_entropy_loss_back`, `opt_step_adamw`,
//! `opt_step_sgd` and `solve_tri`: the user-callback ops, the training ops,
//! and the triangular solve.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! `solve_tri` sits in the C between the recurrent ops, not at the end with
//! the rest of these; it is here because the split's `recurrent.zig` is about
//! state-space kernels and this is linear algebra. `fwht` and
//! `lightning_indexer` sit in the C's last stretch too and belong to other
//! files of the split.
//!
//! # Contraction is named, site by site
//!
//! `ops.cpp` compiles at `-ffp-contract=on`. The optimizers are where it
//! bites: `m*beta1 + g*(1 - beta1)` is fused on the **left** product, and
//! `w*keep - alpha*mh/vh` fuses `w*keep` over a quotient. Each site says
//! which multiply clang takes and why.
//!
//! # Debug-only checks
//!
//! The C's `#ifndef NDEBUG` NaN scans and its plain `assert`s are compiled
//! out of a release build. They are kept as comments at the site, so the
//! port does not check what the shipped C does not.
//!
//! # Loop index names
//!
//! `i00`…`i03` and `i1` are Zig integer type names. Renamed `j00`…`j03` and
//! `j1`, digit for digit.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const vec = @import("vecinline.zig");
const threading = @import("../threading.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

extern fn sqrtf(x: f32) f32;

/// `memcpy(&p, dst->op_params, sizeof(p))`: the four custom-op parameter
/// structs share one layout, so one read serves them all.
fn customParams(dst: *const Tensor) impl.CustomOpParams {
    var p: impl.CustomOpParams = undefined;
    @memcpy(std.mem.asBytes(&p), std.mem.asBytes(&dst.op_params)[0..@sizeOf(impl.CustomOpParams)]);
    return p;
}

// -----------------------------------------------------------------------------
// The user-callback ops

/// Ports `ggml_compute_forward_map_custom1` (ops.cpp:11468 @c1d0e7a00).
pub export fn ggml_compute_forward_map_custom1(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const a = dst.src[0];

    const p = customParams(dst);
    const fun: c.ggml_custom1_op_t = @ptrCast(@alignCast(p.fun));

    fun.?(dst, a, params.ith, params.nth, p.userdata);
}

/// Ports `ggml_compute_forward_map_custom2` (ops.cpp:11482 @c1d0e7a00).
pub export fn ggml_compute_forward_map_custom2(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const a = dst.src[0];
    const b = dst.src[1];

    const p = customParams(dst);
    const fun: c.ggml_custom2_op_t = @ptrCast(@alignCast(p.fun));

    fun.?(dst, a, b, params.ith, params.nth, p.userdata);
}

/// Ports `ggml_compute_forward_map_custom3` (ops.cpp:11497 @c1d0e7a00).
pub export fn ggml_compute_forward_map_custom3(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const a = dst.src[0];
    const b = dst.src[1];
    const cc = dst.src[2];

    const p = customParams(dst);
    const fun: c.ggml_custom3_op_t = @ptrCast(@alignCast(p.fun));

    fun.?(dst, a, b, cc, params.ith, params.nth, p.userdata);
}

/// Ports `ggml_compute_forward_custom` (ops.cpp:11513 @c1d0e7a00).
pub export fn ggml_compute_forward_custom(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const p = customParams(dst);
    const fun: c.ggml_custom_op_t = @ptrCast(@alignCast(p.fun));

    fun.?(dst, params.ith, params.nth, p.userdata);
}

// -----------------------------------------------------------------------------
// cross_entropy_loss

/// Ports `ggml_compute_forward_cross_entropy_loss_f32` (ops.cpp:11525 @c1d0e7a00).
fn crossEntropyLossF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(src0.nb[0] == c.ggml_type_size(src0.type), "src0->nb[0] == ggml_type_size(src0->type)");
    impl.assert(src1.nb[0] == c.ggml_type_size(src1.type), "src1->nb[0] == ggml_type_size(src1->type)");
    impl.assert(c.ggml_are_same_shape(src0, src1), "ggml_are_same_shape(src0, src1)");
    impl.assert(c.ggml_is_scalar(dst), "ggml_is_scalar(dst)");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    // TODO: handle transposed/permuted matrices
    const nc = src0.ne[0];
    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const sums: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
    const st = sums + @as(usize, @intCast(nth + ith * nc));
    var sum_thread: f32 = 0.0;

    impl.assert(params.wsize >= @sizeOf(f32) * @as(usize, @intCast(nth + nth * nc)), "params->wsize >= sizeof(float) * (nth + nth * nc)");

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const d0: [*]const u8 = @ptrCast(src0.data.?);
    const d1: [*]const u8 = @ptrCast(src1.data.?);

    var j1: i64 = ir0;
    while (j1 < ir1) : (j1 += 1) {
        const s0: [*]const f32 = @ptrCast(@alignCast(d0 + @as(usize, @intCast(j1)) * src0.nb[1]));
        const s1: [*]const f32 = @ptrCast(@alignCast(d1 + @as(usize, @intCast(j1)) * src1.nb[1]));

        // #ifndef NDEBUG: assert(!isnan(s0[i])), assert(!isnan(s1[i]))

        var max: f32 = -std.math.inf(f32);
        vec.max_f32(nc, &max, s0);
        const sum_softmax = vec.log_soft_max_f32(@intCast(nc), st, s0, max);
        // assert(sum_softmax >= 0.0); -- compiled out under NDEBUG

        // `-sum_softmax` is a `ggml_float` narrowed at the call.
        vec.add1_f32(nc, st, st, @floatCast(-sum_softmax));
        vec.mul_f32(nc, st, st, s1);

        var sum_st: f32 = 0.0;
        vec.sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

        // #ifndef NDEBUG: assert(!isnan(st[i])), assert(!isinf(st[i]))
    }
    sums[@intCast(ith)] = sum_thread;
    threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));

    if (ith == 0) {
        const dp: [*]f32 = @ptrCast(@alignCast(dst.data.?));
        vec.sum_f32(nth, &dp[0], sums);
        dp[0] *= -1.0 / @as(f32, @floatFromInt(nr));
    }
}

/// Ports `ggml_compute_forward_cross_entropy_loss` (ops.cpp:11601 @c1d0e7a00).
pub export fn ggml_compute_forward_cross_entropy_loss(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => crossEntropyLossF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// cross_entropy_loss_back

/// Ports `ggml_compute_forward_cross_entropy_loss_back_f32`
/// (ops.cpp:11621 @c1d0e7a00).
fn crossEntropyLossBackF32(params: *const ComputeParams, dst: *Tensor) void {
    const grad = impl.one(Tensor, dst.src[0]); // gradient of forward pass output
    const src0f = impl.one(Tensor, dst.src[1]); // src0 of forward pass
    const src1f = impl.one(Tensor, dst.src[2]); // src1 of forward pass

    impl.assert(c.ggml_is_contiguous(dst), "ggml_is_contiguous(dst)");
    impl.assert(c.ggml_is_contiguous(src0f), "ggml_is_contiguous(src0f)");
    impl.assert(c.ggml_is_contiguous(src1f), "ggml_is_contiguous(src1f)");
    impl.assert(c.ggml_is_contiguous(grad), "ggml_is_contiguous(grad)");
    impl.assert(c.ggml_are_same_shape(src0f, src1f) and c.ggml_are_same_shape(src0f, dst), "ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // TODO: handle transposed/permuted matrices
    const nc = src0f.ne[0];
    const nr: i64 = @intCast(c.ggml_nrows(src0f));

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const d_by_nr = @as([*]const f32, @ptrCast(@alignCast(grad.data.?)))[0] / @as(f32, @floatFromInt(nr));

    const dd: [*]u8 = @ptrCast(dst.data.?);
    const d0: [*]const u8 = @ptrCast(src0f.data.?);
    const d1: [*]const u8 = @ptrCast(src1f.data.?);

    var j1: i64 = ir0;
    while (j1 < ir1) : (j1 += 1) {
        const row: usize = @intCast(j1);
        const ds0: [*]f32 = @ptrCast(@alignCast(dd + row * dst.nb[1]));
        const s0: [*]const f32 = @ptrCast(@alignCast(d0 + row * src0f.nb[1]));
        const s1: [*]const f32 = @ptrCast(@alignCast(d1 + row * src1f.nb[1]));

        // #ifndef NDEBUG: assert(!isnan(s0[i])), assert(!isnan(s1[i]))

        // soft_max
        var max: f32 = -std.math.inf(f32);
        vec.max_f32(nc, &max, s0);
        const sum = vec.soft_max_f32(@intCast(nc), ds0, s0, max);
        // assert(sum > 0.0); -- compiled out under NDEBUG
        // `1.0/sum` is a `double` division, narrowed at the call.
        vec.scale_f32(nc, ds0, @floatCast(1.0 / sum));

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        vec.sub_f32(nc, ds0, ds0, s1);
        vec.scale_f32(nc, ds0, d_by_nr);

        // #ifndef NDEBUG: assert(!isnan(ds0[i])), assert(!isinf(ds0[i]))
    }
}

/// Ports `ggml_compute_forward_cross_entropy_loss_back`
/// (ops.cpp:11684 @c1d0e7a00).
pub export fn ggml_compute_forward_cross_entropy_loss_back(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => crossEntropyLossBackF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// opt_step_adamw

/// Ports `ggml_compute_forward_opt_step_adamw_f32` (ops.cpp:11702 @c1d0e7a00).
///
/// The C counts rows in `int`; `i64` here holds the same values.
fn optStepAdamwF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_grad = impl.one(Tensor, dst.src[1]);
    const src0_grad_m = impl.one(Tensor, dst.src[2]);
    const src0_grad_v = impl.one(Tensor, dst.src[3]);
    const adamw_params = impl.one(Tensor, dst.src[4]);

    impl.assert(c.ggml_are_same_shape(src0, src0_grad), "ggml_are_same_shape(src0, src0_grad)");
    impl.assert(c.ggml_are_same_shape(src0, src0_grad_m), "ggml_are_same_shape(src0, src0_grad_m)");
    impl.assert(c.ggml_are_same_shape(src0, src0_grad_v), "ggml_are_same_shape(src0, src0_grad_v)");
    impl.assert(c.ggml_nelements(adamw_params) == 7, "ggml_nelements(adamw_params) == 7");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const l = common.UnaryLocals.of(src0, dst);
    impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const adamw_params_ptr: [*]const f32 = c.ggml_get_data_f32(adamw_params);

    const alpha = adamw_params_ptr[0];
    const beta1 = adamw_params_ptr[1];
    const beta2 = adamw_params_ptr[2];
    const eps = adamw_params_ptr[3];
    const wd = adamw_params_ptr[4];
    const beta1h = adamw_params_ptr[5];
    const beta2h = adamw_params_ptr[6];
    // `1.f - alpha * wd`: the product is the right operand of a `-`, so
    // clang fuses it as `fma(-alpha, wd, 1)`.
    const keep = @mulAdd(f32, -alpha, wd, 1.0);

    const w_base: [*]u8 = @ptrCast(src0.data.?);
    const g_base: [*]const u8 = @ptrCast(src0_grad.data.?);
    const m_base: [*]u8 = @ptrCast(src0_grad_m.data.?);
    const v_base: [*]u8 = @ptrCast(src0_grad_v.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        const offset = @as(usize, @intCast(j03)) * l.nb03 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j01)) * l.nb01;

        const w: [*]f32 = @ptrCast(@alignCast(w_base + offset)); // weight
        const g: [*]const f32 = @ptrCast(@alignCast(g_base + offset)); // grad
        const m: [*]f32 = @ptrCast(@alignCast(m_base + offset));
        const v: [*]f32 = @ptrCast(@alignCast(v_base + offset));

        var j00: usize = 0;
        while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
            // Both updates have a product on each side of the `+`; clang
            // fuses the left one and leaves the right one rounded.
            m[j00] = @mulAdd(f32, m[j00], beta1, g[j00] * (1.0 - beta1));
            v[j00] = @mulAdd(f32, v[j00], beta2, g[j00] * g[j00] * (1.0 - beta2));

            const mh = m[j00] * beta1h;
            // The left operand of this `+` is a call, not a multiply: plain.
            const vh = sqrtf(v[j00] * beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            //
            // `w*keep - alpha*mh/vh`: the right operand is a quotient, so
            // only `w*keep` can fuse.
            w[j00] = @mulAdd(f32, w[j00], keep, -(alpha * mh / vh));
        }
    }
}

/// Ports `ggml_compute_forward_opt_step_adamw` (ops.cpp:11769 @c1d0e7a00).
pub export fn ggml_compute_forward_opt_step_adamw(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => optStepAdamwF32(params, dst),
        else => impl.abort("fatal error"),
    }
}

// -----------------------------------------------------------------------------
// opt_step_sgd

/// Ports `ggml_compute_forward_opt_step_sgd_f32` (ops.cpp:11787 @c1d0e7a00).
fn optStepSgdF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src0_grad = impl.one(Tensor, dst.src[1]);
    const sgd_params = impl.one(Tensor, dst.src[2]);

    impl.assert(c.ggml_are_same_shape(src0, src0_grad), "ggml_are_same_shape(src0, src0_grad)");
    impl.assert(c.ggml_nelements(sgd_params) == 2, "ggml_nelements(sgd_params) == 2");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const nr: i64 = @intCast(c.ggml_nrows(src0));

    const l = common.UnaryLocals.of(src0, dst);
    impl.assert(l.nb00 == @sizeOf(f32), "nb00 == sizeof(float)");

    // rows per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // row range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    const sgd_params_ptr: [*]const f32 = c.ggml_get_data_f32(sgd_params);
    const alpha = sgd_params_ptr[0];
    // Fused as in adamw's `keep`.
    const keep = @mulAdd(f32, -alpha, sgd_params_ptr[1], 1.0);

    const w_base: [*]u8 = @ptrCast(src0.data.?);
    const g_base: [*]const u8 = @ptrCast(src0_grad.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * l.ne01);
        const j02 = @divTrunc(ir - j03 * l.ne02 * l.ne01, l.ne01);
        const j01 = ir - j03 * l.ne02 * l.ne01 - j02 * l.ne01;

        const offset = @as(usize, @intCast(j03)) * l.nb03 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j01)) * l.nb01;

        const w: [*]f32 = @ptrCast(@alignCast(w_base + offset)); // weight
        const g: [*]const f32 = @ptrCast(@alignCast(g_base + offset)); // grad

        var j00: usize = 0;
        while (j00 < @as(usize, @intCast(l.ne00))) : (j00 += 1) {
            // `w*keep - alpha*g`: two products; the left one fuses.
            w[j00] = @mulAdd(f32, w[j00], keep, -(alpha * g[j00]));
        }
    }
}

/// Ports `ggml_compute_forward_opt_step_sgd` (ops.cpp:11831 @c1d0e7a00).
pub export fn ggml_compute_forward_opt_step_sgd(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    switch (src0.type) {
        c.GGML_TYPE_F32 => optStepSgdF32(params, dst),
        else => impl.abort("fatal error - sgd is F32 only"),
    }
}

// -----------------------------------------------------------------------------
// solve_tri

/// Ports `ggml_compute_forward_solve_tri_f32` (ops.cpp:10681 @c1d0e7a00):
/// forward substitution, one right-hand-side column per work unit.
fn solveTriF32(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // A (lower triangular)
    const src1 = impl.one(Tensor, dst.src[1]); // B (RHS)

    const l = common.BinaryLocals.of(src0, src1, dst);

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "src1->type == GGML_TYPE_F32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type  == GGML_TYPE_F32");

    impl.assert(l.ne00 == l.ne01, "ne00 == ne01"); // A must be square
    impl.assert(l.ne0 == l.ne10, "ne0  == ne10"); // solution cols == B cols
    impl.assert(l.ne1 == l.ne11, "ne1  == ne11"); // solution rows == B rows

    impl.assert(l.ne02 == l.ne12 and l.ne12 == l.ne2, "ne02 == ne12 && ne12 == ne2");
    impl.assert(l.ne03 == l.ne13 and l.ne13 == l.ne3, "ne03 == ne13 && ne13 == ne3");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const k = l.ne10; // number of RHS columns
    const n = l.ne11; // A is n×n
    const nr = l.ne02 * l.ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    const dr = @divTrunc(nr + nth - 1, nth);

    // chunk range for this thread
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const A: [*]const f32 = @ptrCast(@alignCast(src0.data.?)); // [n, n, B1, B2]
    const B: [*]const f32 = @ptrCast(@alignCast(src1.data.?)); // [n, k, B1, B2]
    const X: [*]f32 = @ptrCast(@alignCast(dst.data.?)); // [n, k, B1, B2]

    const ku: usize = @intCast(k);
    const nu: usize = @intCast(n);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        const j03 = @divTrunc(ir, l.ne02 * k);
        const j02 = @divTrunc(ir - j03 * l.ne02 * k, k);
        const j01: usize = @intCast(ir - j03 * l.ne02 * k - j02 * k);

        const k02: usize = @intCast(j02);
        const k03: usize = @intCast(j03);
        const A_batch = A + k02 * l.nb02 / @sizeOf(f32) + k03 * l.nb03 / @sizeOf(f32);
        const B_batch = B + k02 * l.nb12 / @sizeOf(f32) + k03 * l.nb13 / @sizeOf(f32);

        const X_batch = X + k02 * l.nb2 / @sizeOf(f32) + k03 * l.nb3 / @sizeOf(f32);

        var j00: usize = 0;
        while (j00 < nu) : (j00 += 1) {
            var sum: f32 = 0.0;
            var t: usize = 0;
            while (t < j00) : (t += 1) {
                // `sum += A*X` is one expression: fused.
                sum = @mulAdd(f32, A_batch[j00 * nu + t], X_batch[t * ku + j01], sum);
            }

            const diag = A_batch[j00 * nu + j00];
            // assert(diag != 0.0f && "Zero diagonal in triangular matrix"); -- compiled out under NDEBUG

            X_batch[j00 * ku + j01] = (B_batch[j00 * ku + j01] - sum) / diag;
        }
    }
}

/// Ports `ggml_compute_forward_solve_tri` (ops.cpp:10740 @c1d0e7a00).
pub export fn ggml_compute_forward_solve_tri(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    if (src0.type == c.GGML_TYPE_F32 and src1.type == c.GGML_TYPE_F32) {
        solveTriF32(params, dst);
    } else {
        impl.abort("fatal error");
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the custom-op parameters read back the layout ggml.c writes" {
    var t: Tensor = std.mem.zeroes(Tensor);
    var userdata: u8 = 0;
    impl.setOpParamsValue(&t, impl.CustomOpParams{
        .fun = @ptrFromInt(0x1000),
        .n_tasks = 3,
        .userdata = &userdata,
    });
    const p = customParams(&t);
    try std.testing.expectEqual(@as(usize, 0x1000), @intFromPtr(p.fun));
    try std.testing.expectEqual(@as(c_int, 3), p.n_tasks);
    try std.testing.expectEqual(@as(?*anyopaque, &userdata), p.userdata);
}
