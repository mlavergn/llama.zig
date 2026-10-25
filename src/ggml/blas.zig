//! ggml's BLAS backend, over Apple's Accelerate.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-blas/ggml-blas.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Accelerate only
//!
//! The C++ picks a BLAS by `#if`: Accelerate, MKL, BLIS, NVPL, or a plain
//! `cblas.h` (OpenBLAS and the rest). This port compiles the Accelerate arm
//! alone (PLAN.md, Decision 60): every other arm links a vendor library, which
//! the library may not gain, and the port already links Accelerate on every
//! Darwin target for the CPU backend's vDSP calls. `build.zig` refuses
//! `-Dblas` elsewhere. Of the vendor thread-count calls the C++ makes before
//! each multiply, Accelerate's arm has none, and `GGML_USE_OPENMP` is off in
//! this build, so the dequantization runs on the `std::async` arm.
//!
//! # Loop indices
//!
//! Zig reserves `j01`, `j03` and their kin as integer type names, so the
//! C++'s `j01`, `j02`, `j03`, `j12` and `j13` are `j01` ... `j13`, digit for
//! digit, as in `cpu/mulmat.zig`.
//!
//! # The multiply
//!
//! `ACCELERATE_NEW_LAPACK` and `ACCELERATE_LAPACK_ILP64`, which upstream's
//! CMake defines for Accelerate, bind `cblas_sgemm` to the new interface with
//! 64-bit integers, a different symbol from the classic one:
//! `cblas_sgemm$NEWLAPACK$ILP64`, read from the reference object with `nm`.
//! The port calls that symbol, so both sides run the same routine.

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// The C++ `new`s the context and the backend and `delete`s both in
/// `ggml_backend_blas_free`; libc's allocator is what a C entry point can
/// reach, as in `cpu/cpu_backend.zig`.
const allocator = std.heap.c_allocator;

/// `enum CBLAS_ORDER` and `enum CBLAS_TRANSPOSE` (vecLib's `cblas_new.h`).
const cblas_row_major: c_int = 101;
const cblas_no_trans: c_int = 111;
const cblas_trans: c_int = 112;

/// `cblas_sgemm` as `ACCELERATE_NEW_LAPACK` and `ACCELERATE_LAPACK_ILP64` bind
/// it: `__LAPACK_int` is `long`.
const cblas_sgemm = @extern(*const fn (
    order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_long,
    n: c_long,
    k: c_long,
    alpha: f32,
    a: [*c]const f32,
    lda: c_long,
    b: [*c]const f32,
    ldb: c_long,
    beta: f32,
    cc: [*c]f32,
    ldc: c_long,
) callconv(.c) void, .{ .name = "cblas_sgemm$NEWLAPACK$ILP64" });

/// Ports `struct ggml_backend_blas_context` (ggml-blas.cpp:22 @c1d0e7a00).
///
/// `tasks` is not kept here: the C++ empties its vector after every
/// multiply, so the port holds the spawned threads for the length of one
/// call instead.
const Context = struct {
    n_threads: c_int = c.GGML_DEFAULT_N_THREADS,
    work_data: ?[*]u8 = null,
    work_size: usize = 0,
};

/// One `std::async` task of `ggml_backend_blas_mul_mat`: rows `start..end`
/// of one plane converted to `f32`.
fn convertRows(to_float: c.ggml_to_float_t, x: [*]const u8, nb01: usize, wplane: [*]f32, ne00: i64, start: i64, end: i64) void {
    var j01 = start;
    while (j01 < end) : (j01 += 1) {
        to_float.?(x + @as(usize, @intCast(j01)) * nb01, wplane + @as(usize, @intCast(j01 * ne00)), ne00);
    }
}

/// Ports `ggml_backend_blas_mul_mat` (ggml-blas.cpp:31 @c1d0e7a00).
fn mulMat(ctx: *Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];
    const ne02 = src0.ne[2];
    const ne03 = src0.ne[3];
    const nb00 = src0.nb[0];
    const nb01 = src0.nb[1];
    const nb02 = src0.nb[2];
    const nb03 = src0.nb[3];
    const ne10 = src1.ne[0];
    const ne11 = src1.ne[1];
    const ne12 = src1.ne[2];
    const ne13 = src1.ne[3];
    const nb10 = src1.nb[0];
    const nb12 = src1.nb[2];
    const nb13 = src1.nb[3];
    const ne0 = dst.ne[0];
    const ne1 = dst.ne[1];
    const ne2 = dst.ne[2];
    const ne3 = dst.ne[3];
    const nb0 = dst.nb[0];
    const nb1 = dst.nb[1];
    const nb2 = dst.nb[2];
    const nb3 = dst.nb[3];

    const @"type" = src0.type;

    impl.assert(ne0 == ne01, "ne0 == ne01");
    impl.assert(ne1 == ne11, "ne1 == ne11");
    impl.assert(ne2 == ne12, "ne2 == ne12");
    impl.assert(ne3 == ne13, "ne3 == ne13");

    // we don't support permuted src0 or src1
    impl.assert(nb00 == c.ggml_type_size(@"type"), "nb00 == ggml_type_size(type)");
    impl.assert(nb10 == c.ggml_type_size(src1.type), "nb10 == ggml_type_size(src1->type)");

    // dst cannot be transposed or permuted
    impl.assert(nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
    impl.assert(nb0 <= nb1, "nb0 <= nb1");
    impl.assert(nb1 <= nb2, "nb1 <= nb2");
    impl.assert(nb2 <= nb3, "nb2 <= nb3");

    // broadcast factors
    const r2 = @divTrunc(ne12, ne02);
    const r3 = @divTrunc(ne13, ne03);

    const ne_plane = ne01 * ne00;
    const desired_wsize: usize = if (@"type" == c.GGML_TYPE_F32) 0 else @as(usize, @intCast(ne03 * ne02 * ne_plane)) * @sizeOf(f32);

    if (ctx.work_size < desired_wsize) {
        if (ctx.work_data) |w| allocator.free(w[0..ctx.work_size]);
        ctx.work_data = (allocator.alloc(u8, desired_wsize) catch impl.abort("std::bad_alloc")).ptr;
        ctx.work_size = desired_wsize;
    }
    // Unread unless src0 needs converting, when it has just been sized.
    const wdata: [*]f32 = if (ctx.work_data) |w| @ptrCast(@alignCast(w)) else undefined;

    // convert src0 to float
    if (@"type" != c.GGML_TYPE_F32) {
        const type_traits = c.ggml_get_type_traits(@"type");
        const to_float = type_traits.*.to_float;

        var tasks: std.ArrayList(std.Thread) = .empty;
        defer tasks.deinit(allocator);

        var j03: i64 = 0;
        while (j03 < ne03) : (j03 += 1) {
            var j02: i64 = 0;
            while (j02 < ne02) : (j02 += 1) {
                const x: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(j02)) * nb02 + @as(usize, @intCast(j03)) * nb03;
                const wplane: [*]f32 = wdata + @as(usize, @intCast(j02 * ne_plane + j03 * ne02 * ne_plane));

                const min_cols_per_thread: c_int = 4096;
                const min_rows_per_thread: c_int = @max(@as(c_int, @truncate(@divTrunc(min_cols_per_thread, ne00))), 1);
                const n_threads: c_int = @max(@min(ctx.n_threads, @as(c_int, @truncate(@divTrunc(ne01, min_rows_per_thread)))), 1);

                var i: c_int = 1;
                while (i < n_threads) : (i += 1) {
                    const start = @divTrunc(i * ne01, n_threads);
                    const end = @divTrunc((i + 1) * ne01, n_threads);
                    if (start < end) {
                        const t = std.Thread.spawn(.{}, convertRows, .{ to_float, x, nb01, wplane, ne00, start, end }) catch
                            impl.abort("std::system_error");
                        tasks.append(allocator, t) catch impl.abort("std::bad_alloc");
                    }
                }
                {
                    // reuse the current thread for the first task
                    convertRows(to_float, x, nb01, wplane, ne00, 0, @divTrunc(ne01, n_threads));
                }
            }
        }

        // wait for all tasks to finish
        for (tasks.items) |t| t.join();
    }

    var j13: i64 = 0;
    while (j13 < ne13) : (j13 += 1) {
        var j12: i64 = 0;
        while (j12 < ne12) : (j12 += 1) {
            const j03 = @divTrunc(j13, r3);
            const j02 = @divTrunc(j12, r2);

            var x: [*]const f32 = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(j02)) * nb02 + @as(usize, @intCast(j03)) * nb03));
            const y: [*]const f32 = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(src1.data.?)) + @as(usize, @intCast(j12)) * nb12 + @as(usize, @intCast(j13)) * nb13));
            const d: [*]f32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(dst.data.?)) + @as(usize, @intCast(j12)) * nb2 + @as(usize, @intCast(j13)) * nb3));

            if (@"type" != c.GGML_TYPE_F32) {
                x = wdata + @as(usize, @intCast(j02 * ne_plane + j03 * ne02 * ne_plane));
            }

            cblas_sgemm(cblas_row_major, cblas_no_trans, cblas_trans, ne1, ne01, ne10, 1.0, y, ne10, x, ne00, 0.0, d, ne01);
        }
    }
}

/// Ports `ggml_backend_blas_out_prod` (ggml-blas.cpp:151 @c1d0e7a00).
fn outProd(ctx: *Context, dst: *Tensor) void {
    _ = ctx;
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    impl.assert(dst.ne[0] == src0.ne[0], "ne0  == ne00");
    impl.assert(dst.ne[1] == src1.ne[0], "ne1  == ne10");
    impl.assert(dst.ne[2] == src0.ne[2], "ne2  == ne02");
    impl.assert(src0.ne[2] == src1.ne[2], "ne02 == ne12");
    impl.assert(dst.ne[3] == src1.ne[3], "ne3  == ne13");
    impl.assert(src0.ne[3] == src1.ne[3], "ne03 == ne13");

    // we don't support permuted src0 or src1
    impl.assert(src0.nb[0] == @sizeOf(f32), "nb00 == sizeof(float)");

    // dst cannot be transposed or permuted
    impl.assert(dst.nb[0] == @sizeOf(f32), "nb0 == sizeof(float)");

    // src0: (k,n), src1: (k,m), dst: (m,n) as (major,minor); sgemm's a is
    // src1 transposed, b is src0 -- unless src1 is already transposed, when
    // sgemm must not transpose it again. The C++ narrows the sizes to `int`.
    const n: c_int = @truncate(src0.ne[0]);
    const k: c_int = @truncate(src0.ne[1]);
    const m: c_int = @truncate(src1.ne[0]);

    var transpose_a: c_int = undefined;
    var lda: c_int = undefined;

    if (!c.ggml_is_transposed(src1)) {
        transpose_a = cblas_trans;
        lda = m;
    } else {
        transpose_a = cblas_no_trans;
        lda = k;
    }

    const a: [*c]const f32 = @ptrCast(@alignCast(src1.data));
    const b: [*c]const f32 = @ptrCast(@alignCast(src0.data));
    const cc: [*c]f32 = @ptrCast(@alignCast(dst.data));

    cblas_sgemm(cblas_row_major, transpose_a, cblas_no_trans, m, n, k, 1.0, a, lda, b, n, 0.0, cc, n);
}

// -----------------------------------------------------------------------------
// The backend

/// Ports `ggml_backend_blas_get_name` (ggml-blas.cpp:214 @c1d0e7a00).
fn getName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    _ = backend;
    return "BLAS";
}

/// Ports `ggml_backend_blas_free` (ggml-blas.cpp:220 @c1d0e7a00).
fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    const b = impl.one(c.struct_ggml_backend, backend);
    const ctx: *Context = @ptrCast(@alignCast(b.context.?));
    if (ctx.work_data) |w| allocator.free(w[0..ctx.work_size]);
    allocator.destroy(ctx);
    allocator.destroy(b);
}

/// Ports `ggml_backend_blas_graph_compute` (ggml-blas.cpp:226 @c1d0e7a00).
fn graphCompute(backend: c.ggml_backend_t, cgraph: *impl.CGraph) callconv(.c) c.enum_ggml_status {
    const ctx: *Context = @ptrCast(@alignCast(impl.one(c.struct_ggml_backend, backend).context.?));

    var i: usize = 0;
    while (i < @as(usize, @intCast(cgraph.n_nodes))) : (i += 1) {
        const node = cgraph.nodes[i].?;

        if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) continue;

        switch (node.op) {
            c.GGML_OP_MUL_MAT => mulMat(ctx, node),
            c.GGML_OP_OUT_PROD => outProd(ctx, node),
            c.GGML_OP_NONE, c.GGML_OP_RESHAPE, c.GGML_OP_VIEW, c.GGML_OP_PERMUTE, c.GGML_OP_TRANSPOSE => {},
            else => c.ggml_abort(@src().file.ptr, @intCast(@src().line), "%s: unsupported op %s\n", "ggml_backend_blas_graph_compute", c.ggml_op_desc(node)),
        }
    }

    return c.GGML_STATUS_SUCCESS;
}

/// Ports `blas_backend_i` (ggml-blas.cpp:262 @c1d0e7a00).
const backend_i = c.struct_ggml_backend_i{
    .get_name = getName,
    .free = backendFree,
    .set_tensor_async = null,
    .get_tensor_async = null,
    .set_tensor_2d_async = null,
    .get_tensor_2d_async = null,
    .cpy_tensor_async = null,
    .synchronize = null,
    .graph_plan_create = null,
    .graph_plan_free = null,
    .graph_plan_update = null,
    .graph_plan_compute = null,
    .graph_compute = @ptrCast(&graphCompute),
    .event_record = null,
    .event_wait = null,
    .graph_optimize = null,
};

/// Ports `ggml_backend_blas_guid` (ggml-blas.cpp:281 @c1d0e7a00).
var blas_guid: c.ggml_guid = .{ 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };

fn guid() c.ggml_guid_t {
    return &blas_guid;
}

/// Ports `ggml_backend_blas_init` (ggml-blas.cpp:286 @c1d0e7a00).
///
/// Return: a new BLAS backend, owned by the caller and freed with
/// `ggml_backend_free`; null if allocation failed, where the C++'s `new`
/// would throw.
pub export fn ggml_backend_blas_init() callconv(.c) c.ggml_backend_t {
    const ctx = allocator.create(Context) catch return null;
    ctx.* = .{};

    const backend = allocator.create(c.struct_ggml_backend) catch {
        allocator.destroy(ctx);
        return null;
    };
    backend.* = .{
        .guid = guid(),
        .iface = backend_i,
        .device = c.ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        .context = ctx,
    };
    return backend;
}

/// Ports `ggml_backend_is_blas` (ggml-blas.cpp:309 @c1d0e7a00).
pub export fn ggml_backend_is_blas(backend: c.ggml_backend_t) callconv(.c) bool {
    return backend != null and c.ggml_guid_matches(impl.one(c.struct_ggml_backend, backend).guid, guid());
}

/// Ports `ggml_backend_blas_set_n_threads` (ggml-blas.cpp:313 @c1d0e7a00).
pub export fn ggml_backend_blas_set_n_threads(backend_blas: c.ggml_backend_t, n_threads: c_int) callconv(.c) void {
    impl.assert(ggml_backend_is_blas(backend_blas), "ggml_backend_is_blas(backend_blas)");

    const ctx: *Context = @ptrCast(@alignCast(impl.one(c.struct_ggml_backend, backend_blas).context.?));
    ctx.n_threads = n_threads;
}

// -----------------------------------------------------------------------------
// The device

/// Ports `ggml_backend_blas_device_get_name` (ggml-blas.cpp:322 @c1d0e7a00).
fn deviceGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    _ = dev;
    return "BLAS";
}

/// Ports `ggml_backend_blas_device_get_description` (ggml-blas.cpp:328
/// @c1d0e7a00): the `GGML_BLAS_USE_ACCELERATE` arm.
fn deviceGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    _ = dev;
    return "Accelerate";
}

/// Ports `ggml_backend_blas_device_get_memory` (ggml-blas.cpp:346 @c1d0e7a00).
fn deviceGetMemory(dev: c.ggml_backend_dev_t, free: *usize, total: *usize) callconv(.c) void {
    _ = dev;
    // no memory to report
    free.* = 0;
    total.* = 0;
}

/// Ports `ggml_backend_blas_device_get_type` (ggml-blas.cpp:354 @c1d0e7a00).
fn deviceGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    _ = dev;
    return c.GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

/// Ports `ggml_backend_blas_device_get_props` (ggml-blas.cpp:360 @c1d0e7a00).
fn deviceGetProps(dev: c.ggml_backend_dev_t, props: *c.struct_ggml_backend_dev_props) callconv(.c) void {
    props.name = deviceGetName(dev);
    props.description = deviceGetDescription(dev);
    props.type = deviceGetType(dev);
    deviceGetMemory(dev, &props.memory_free, &props.memory_total);
    props.caps = .{
        .async = false,
        .host_buffer = false,
        .buffer_from_host_ptr = true,
        .events = false,
        .mmap_support = true,
    };
}

/// Ports `ggml_backend_blas_device_init_backend` (ggml-blas.cpp:374
/// @c1d0e7a00).
fn deviceInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = dev;
    _ = params;
    return ggml_backend_blas_init();
}

/// Ports `ggml_backend_blas_device_get_buffer_type` (ggml-blas.cpp:381
/// @c1d0e7a00).
fn deviceGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    _ = dev;
    return c.ggml_backend_cpu_buffer_type();
}

/// Ports `ggml_backend_blas_device_buffer_from_host_ptr` (ggml-blas.cpp:387
/// @c1d0e7a00).
fn deviceBufferFromHostPtr(dev: c.ggml_backend_dev_t, ptr: ?*anyopaque, size: usize, max_tensor_size: usize) callconv(.c) c.ggml_backend_buffer_t {
    _ = dev;
    _ = max_tensor_size;
    return c.ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

/// Ports `ggml_backend_blas_device_supports_op` (ggml-blas.cpp:394
/// @c1d0e7a00).
fn deviceSupportsOp(dev: c.ggml_backend_dev_t, op: *const Tensor) callconv(.c) bool {
    _ = dev;
    switch (op.op) {
        c.GGML_OP_NONE, c.GGML_OP_RESHAPE, c.GGML_OP_VIEW, c.GGML_OP_PERMUTE, c.GGML_OP_TRANSPOSE => return true,

        c.GGML_OP_MUL_MAT => {
            // BLAS usually is only faster for large matrices
            const src0 = impl.one(Tensor, op.src[0]);
            const src1 = impl.one(Tensor, op.src[1]);

            const ne10 = src1.ne[0];

            const ne0 = op.ne[0];
            const ne1 = op.ne[1];

            // TODO: find the optimal value
            const min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            if (opParamI32(op, 1) == c.GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            return c.ggml_is_contiguous(src0) and
                c.ggml_is_contiguous(src1) and
                src1.type == c.GGML_TYPE_F32 and
                (ne0 >= min_batch and ne1 >= min_batch and ne10 >= min_batch) and
                (src0.type == c.GGML_TYPE_F32 or c.ggml_get_type_traits(src0.type).*.to_float != null);
        },

        c.GGML_OP_OUT_PROD => {
            const src0 = op.src[0];
            const src1 = op.src[1];
            return impl.one(Tensor, src0).type == c.GGML_TYPE_F32 and
                impl.one(Tensor, src1).type == c.GGML_TYPE_F32 and
                c.ggml_is_matrix(src0) and
                c.ggml_is_matrix(src1) and
                c.ggml_is_contiguous(src0) and
                (c.ggml_is_contiguous(src1) or c.ggml_is_transposed(src1)) and
                (impl.one(Tensor, src0).type == c.GGML_TYPE_F32 or c.ggml_get_type_traits(impl.one(Tensor, src0).type).*.to_float != null);
        },

        else => return false,
    }
}

/// `ggml_get_op_params_i32` (ggml-impl.h), which `@cImport` cannot reach.
fn opParamI32(op: *const Tensor, i: usize) i32 {
    return op.op_params[i];
}

/// Ports `ggml_backend_blas_device_supports_buft` (ggml-blas.cpp:450
/// @c1d0e7a00).
fn deviceSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = dev;
    return c.ggml_backend_buft_is_host(buft);
}

/// Ports `ggml_backend_blas_device_i` (ggml-blas.cpp:456 @c1d0e7a00).
const device_i = c.struct_ggml_backend_device_i{
    .get_name = deviceGetName,
    .get_description = deviceGetDescription,
    .get_memory = @ptrCast(&deviceGetMemory),
    .get_type = deviceGetType,
    .get_props = @ptrCast(&deviceGetProps),
    .init_backend = deviceInitBackend,
    .get_buffer_type = deviceGetBufferType,
    .get_host_buffer_type = null,
    .buffer_from_host_ptr = deviceBufferFromHostPtr,
    .supports_op = @ptrCast(&deviceSupportsOp),
    .supports_buft = deviceSupportsBuft,
    .offload_op = null,
    .event_new = null,
    .event_free = null,
    .event_synchronize = null,
};

// -----------------------------------------------------------------------------
// The registration

/// Ports `ggml_backend_blas_reg_get_name` (ggml-blas.cpp:476 @c1d0e7a00).
fn regGetName(reg: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = reg;
    return "BLAS";
}

/// Ports `ggml_backend_blas_reg_get_device_count` (ggml-blas.cpp:482
/// @c1d0e7a00).
fn regGetDeviceCount(reg: c.ggml_backend_reg_t) callconv(.c) usize {
    _ = reg;
    return 1;
}

/// The C++'s function-local `static ggml_backend_device`, whose `reg` is the
/// registration it is first asked through -- always `blas_reg`, the only
/// one -- so it is initialised statically here.
var blas_device: c.struct_ggml_backend_device = .{ .iface = device_i, .reg = &blas_reg, .context = null };

/// Ports `ggml_backend_blas_reg_get_device` (ggml-blas.cpp:488 @c1d0e7a00).
fn regGetDevice(reg: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    _ = reg;
    impl.assert(index == 0, "index == 0");
    return &blas_device;
}

/// Ports `ggml_backend_blas_get_proc_address` (ggml-blas.cpp:503 @c1d0e7a00).
fn regGetProcAddress(reg: c.ggml_backend_reg_t, name: [*c]const u8) callconv(.c) ?*anyopaque {
    _ = reg;
    if (std.mem.eql(u8, std.mem.span(@as([*:0]const u8, @ptrCast(name))), "ggml_backend_set_n_threads")) {
        return @ptrCast(@constCast(&ggml_backend_blas_set_n_threads));
    }
    return null;
}

/// Ports `ggml_backend_blas_reg_i` (ggml-blas.cpp:513 @c1d0e7a00).
const reg_i = c.struct_ggml_backend_reg_i{
    .get_name = regGetName,
    .get_device_count = regGetDeviceCount,
    .get_device = regGetDevice,
    .get_proc_address = regGetProcAddress,
};

var blas_reg = c.struct_ggml_backend_reg{
    .api_version = c.GGML_BACKEND_API_VERSION,
    .iface = reg_i,
    .context = null,
};

/// Ports `ggml_backend_blas_reg` (ggml-blas.cpp:520 @c1d0e7a00).
///
/// Return: the BLAS registration, a process-lifetime singleton.
pub export fn ggml_backend_blas_reg() callconv(.c) c.ggml_backend_reg_t {
    return &blas_reg;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
