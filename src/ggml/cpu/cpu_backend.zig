//! The CPU backend's `ggml_backend` and `ggml_backend_device` interfaces.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Not to be confused with `ggml-cpu.c`
//!
//! That file is already ported, across `cpu/{forward,plan,threading,…}.zig`.
//! This is the C++ one beside it: the backend registration and device
//! interface, where that one is the compute engine.
//!
//! # The extra-buffer list
//!
//! The C's `ggml_backend_cpu_get_extra_buffer_types()` returns
//! `std::vector<ggml_backend_buffer_type_t> &`, built once by a lambda. It
//! is the only C++-linkage symbol crossing `ggml-cpu/`, and it disappears
//! here: the list is a static array filled on first use, handed out as a
//! slice. `cpu/extra.zig` consumes it.
//!
//! # `new` without `delete`
//!
//! Several objects the C `new`s are never freed — the device context, the
//! registration. Those are process-lifetime singletons and the C leaks them
//! deliberately; the port allocates them the same way rather than inventing
//! an ownership model the C does not have.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../impl.zig");
const defs = @import("defs.zig");
const extra = @import("extra.zig");
const repack = @import("repack/module.zig");

const c = impl.c;
const Tensor = defs.Tensor;

/// The C `new`s these and mostly never frees them. libc's allocator is what
/// a C entry point can reach without being handed one — the same reasoning
/// as `backend_reg.zig` and `backend.zig`.
const allocator = std.heap.c_allocator;

// -----------------------------------------------------------------------------
// The extra buffer types

/// Storage for `ggml_backend_cpu_get_extra_buffer_types` (ggml-cpu.cpp:42
/// @c1d0e7a00).
///
/// Only the repack entry compiles on this target: the AMX, SpaceMIT and
/// KleidiAI arms are behind `__AMX_INT8__`, `GGML_USE_CPU_RISCV64_SPACEMIT`
/// and `GGML_USE_CPU_KLEIDIAI`, none of which this build defines.
var extra_bufts: [2]c.ggml_backend_buffer_type_t = .{ null, null };
var extra_bufts_n: usize = 0;

/// The C++ magic-static guard, written out — the same shape
/// `backend_reg.zig` uses for `get_reg`. Every call after the first is a
/// single acquire load.
var extra_bufts_ready = std.atomic.Value(bool).init(false);
var extra_bufts_mutex: std.c.pthread_mutex_t = .{};

fn ensureExtraBufts() void {
    if (extra_bufts_ready.load(.acquire)) return;
    _ = std.c.pthread_mutex_lock(&extra_bufts_mutex);
    defer _ = std.c.pthread_mutex_unlock(&extra_bufts_mutex);
    if (extra_bufts_ready.load(.acquire)) return;

    const buft = repack.ggml_backend_cpu_repack_buffer_type();
    if (buft != null) {
        extra_bufts[extra_bufts_n] = buft;
        extra_bufts_n += 1;
    }
    extra_bufts_ready.store(true, .release);
}

/// Ports `ggml_backend_cpu_get_extra_buffer_types` (ggml-cpu.cpp:42
/// @c1d0e7a00).
///
/// Return: the extra buffer types, borrowed for the life of the process.
pub fn getExtraBufferTypes() []const c.ggml_backend_buffer_type_t {
    ensureExtraBufts();
    return extra_bufts[0..extra_bufts_n];
}

/// Ports `ggml_backend_cpu_device_get_extra_buffers_type` (ggml-cpu.cpp:76
/// @c1d0e7a00): the same list, null-terminated, as the proc-address export
/// hands to `llama.cpp`.
fn deviceGetExtraBuffersType(device: c.ggml_backend_dev_t) callconv(.c) [*c]c.ggml_backend_buffer_type_t {
    _ = device;
    ensureExtraBufts();
    // `extra_bufts` has a trailing null slot that `extra_bufts_n` excludes,
    // which is the `bufts.push_back(nullptr)` the C does into a second
    // vector.
    return &extra_bufts;
}

/// Ports `ggml_backend_cpu_is_extra_buffer_type` (ggml-cpu.cpp:88
/// @c1d0e7a00).
fn isExtraBufferType(buft: c.ggml_backend_buffer_type_t) bool {
    for (getExtraBufferTypes()) |e| {
        if (e == buft) return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// The backend (stream)

/// Ports `struct ggml_backend_cpu_context` (ggml-cpu.cpp:99 @c1d0e7a00).
const Context = struct {
    n_threads: c_int,
    threadpool: c.ggml_threadpool_t,
    work_data: ?[*]u8,
    work_size: usize,
    abort_callback: c.ggml_abort_callback,
    abort_callback_data: ?*anyopaque,
    /// Take the reference path, skipping fused and vectorised kernels.
    use_ref: bool,
};

/// Ports `ggml_backend_cpu_get_name` (ggml-cpu.cpp:112 @c1d0e7a00).
fn getName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    _ = backend;
    return "CPU";
}

/// Ports `ggml_backend_cpu_free` (ggml-cpu.cpp:118 @c1d0e7a00).
fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    const b = impl.one(c.struct_ggml_backend, backend);
    const ctx: *Context = @ptrCast(@alignCast(b.context.?));
    if (ctx.work_data) |w| allocator.free(w[0..ctx.work_size]);
    allocator.destroy(ctx);
    allocator.destroy(b);
}

/// Ports `struct ggml_backend_plan_cpu` (ggml-cpu.cpp:125 @c1d0e7a00).
const PlanCpu = struct {
    cplan: c.struct_ggml_cplan,
    cgraph: impl.CGraph,
};

/// Ports `ggml_backend_cpu_graph_plan_create` (ggml-cpu.cpp:130 @c1d0e7a00).
fn graphPlanCreate(backend: c.ggml_backend_t, cgraph: *const impl.CGraph) callconv(.c) c.ggml_backend_graph_plan_t {
    const b = impl.one(c.struct_ggml_backend, backend);
    const ctx: *Context = @ptrCast(@alignCast(b.context.?));

    const plan = allocator.create(PlanCpu) catch return null;

    plan.cplan = c.ggml_graph_plan(@ptrCast(cgraph), ctx.n_threads, ctx.threadpool);
    // The C's own comment: "FIXME: deep copy".
    plan.cgraph = cgraph.*;

    if (plan.cplan.work_size > 0) {
        const buf = allocator.alloc(u8, plan.cplan.work_size) catch {
            allocator.destroy(plan);
            return null;
        };
        plan.cplan.work_data = buf.ptr;
    }

    plan.cplan.abort_callback = ctx.abort_callback;
    plan.cplan.abort_callback_data = ctx.abort_callback_data;
    plan.cplan.use_ref = ctx.use_ref;

    return plan;
}

/// Ports `ggml_backend_cpu_graph_plan_free` (ggml-cpu.cpp:153 @c1d0e7a00).
fn graphPlanFree(backend: c.ggml_backend_t, plan_in: c.ggml_backend_graph_plan_t) callconv(.c) void {
    _ = backend;
    const plan: *PlanCpu = @ptrCast(@alignCast(plan_in.?));
    if (plan.cplan.work_data) |w| allocator.free(w[0..plan.cplan.work_size]);
    allocator.destroy(plan);
}

/// Ports `ggml_backend_cpu_graph_plan_compute` (ggml-cpu.cpp:162 @c1d0e7a00).
fn graphPlanCompute(backend: c.ggml_backend_t, plan_in: c.ggml_backend_graph_plan_t) callconv(.c) c.enum_ggml_status {
    _ = backend;
    const plan: *PlanCpu = @ptrCast(@alignCast(plan_in.?));
    return c.ggml_graph_compute(@ptrCast(&plan.cgraph), &plan.cplan);
}

/// Ports `ggml_backend_cpu_graph_compute` (ggml-cpu.cpp:170 @c1d0e7a00).
fn graphCompute(backend: c.ggml_backend_t, cgraph: *impl.CGraph) callconv(.c) c.enum_ggml_status {
    const b = impl.one(c.struct_ggml_backend, backend);
    const ctx: *Context = @ptrCast(@alignCast(b.context.?));

    var cplan = c.ggml_graph_plan(@ptrCast(cgraph), ctx.n_threads, ctx.threadpool);

    if (ctx.work_size < cplan.work_size) {
        if (ctx.work_data) |w| allocator.free(w[0..ctx.work_size]);
        const buf = allocator.alloc(u8, cplan.work_size) catch {
            ctx.work_data = null;
            ctx.work_size = 0;
            return c.GGML_STATUS_ALLOC_FAILED;
        };
        ctx.work_data = buf.ptr;
        ctx.work_size = cplan.work_size;
    }
    cplan.work_data = ctx.work_data;

    cplan.abort_callback = ctx.abort_callback;
    cplan.abort_callback_data = ctx.abort_callback_data;
    cplan.use_ref = ctx.use_ref;

    return c.ggml_graph_compute(@ptrCast(cgraph), &cplan);
}

/// Ports `ggml_backend_cpu_i` (ggml-cpu.cpp:193 @c1d0e7a00).
const backend_i = c.struct_ggml_backend_i{
    .get_name = getName,
    .free = backendFree,
    .set_tensor_async = null,
    .get_tensor_async = null,
    .set_tensor_2d_async = null,
    .get_tensor_2d_async = null,
    .cpy_tensor_async = null,
    .synchronize = null,
    .graph_plan_create = @ptrCast(&graphPlanCreate),
    .graph_plan_free = graphPlanFree,
    .graph_plan_update = null,
    .graph_plan_compute = graphPlanCompute,
    .graph_compute = @ptrCast(&graphCompute),
    .event_record = null,
    .event_wait = null,
    .graph_optimize = null,
};

/// Ports `ggml_backend_cpu_guid` (ggml-cpu.cpp:212 @c1d0e7a00).
var cpu_guid: c.ggml_guid = .{ 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };

fn guid() c.ggml_guid_t {
    return &cpu_guid;
}

extern fn ggml_cpu_init() void;

/// Ports `ggml_backend_cpu_init` (ggml-cpu.cpp:217 @c1d0e7a00).
///
/// Return: a new CPU backend, or null if allocation failed. The caller owns
/// it and frees it with `ggml_backend_free`.
pub export fn ggml_backend_cpu_init() callconv(.c) c.ggml_backend_t {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    const ctx = allocator.create(Context) catch return null;
    ctx.* = .{
        .n_threads = c.GGML_DEFAULT_N_THREADS,
        .threadpool = null,
        .work_data = null,
        .work_size = 0,
        .abort_callback = null,
        .abort_callback_data = null,
        .use_ref = false,
    };

    const backend = allocator.create(c.struct_ggml_backend) catch {
        allocator.destroy(ctx);
        return null;
    };
    backend.* = .{
        .guid = guid(),
        .iface = backend_i,
        .device = c.ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        .context = ctx,
    };
    return backend;
}

/// Ports `ggml_backend_is_cpu` (ggml-cpu.cpp:249 @c1d0e7a00).
pub export fn ggml_backend_is_cpu(backend: c.ggml_backend_t) callconv(.c) bool {
    return backend != null and c.ggml_guid_matches(impl.one(c.struct_ggml_backend, backend).guid, guid());
}

inline fn contextOf(backend_cpu: c.ggml_backend_t) *Context {
    impl.assert(ggml_backend_is_cpu(backend_cpu), "ggml_backend_is_cpu(backend_cpu)");
    return @ptrCast(@alignCast(impl.one(c.struct_ggml_backend, backend_cpu).context.?));
}

/// Ports `ggml_backend_cpu_set_n_threads` (ggml-cpu.cpp:253 @c1d0e7a00).
pub export fn ggml_backend_cpu_set_n_threads(backend_cpu: c.ggml_backend_t, n_threads: c_int) callconv(.c) void {
    contextOf(backend_cpu).n_threads = n_threads;
}

/// Ports `ggml_backend_cpu_set_threadpool` (ggml-cpu.cpp:260 @c1d0e7a00).
pub export fn ggml_backend_cpu_set_threadpool(backend_cpu: c.ggml_backend_t, threadpool: c.ggml_threadpool_t) callconv(.c) void {
    const ctx = contextOf(backend_cpu);
    if (ctx.threadpool != null and ctx.threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        c.ggml_threadpool_pause(ctx.threadpool);
    }
    ctx.threadpool = threadpool;
}

/// Ports `ggml_backend_cpu_set_abort_callback` (ggml-cpu.cpp:272 @c1d0e7a00).
pub export fn ggml_backend_cpu_set_abort_callback(backend_cpu: c.ggml_backend_t, abort_callback: c.ggml_abort_callback, abort_callback_data: ?*anyopaque) callconv(.c) void {
    const ctx = contextOf(backend_cpu);
    ctx.abort_callback = abort_callback;
    ctx.abort_callback_data = abort_callback_data;
}

/// Ports `ggml_backend_cpu_set_use_ref` (ggml-cpu.cpp:280 @c1d0e7a00).
pub export fn ggml_backend_cpu_set_use_ref(backend_cpu: c.ggml_backend_t, use_ref: bool) callconv(.c) void {
    contextOf(backend_cpu).use_ref = use_ref;
}

comptime {
    _ = isExtraBufferType;
    _ = deviceGetExtraBuffersType;
    _ = extra;
}

// -----------------------------------------------------------------------------
// The device

/// Ports `struct ggml_backend_cpu_device_context` (ggml-cpu.cpp:289
/// @c1d0e7a00), the `__APPLE__` arm of its constructor.
///
/// The C holds a `std::string` and resizes it around `sysctlbyname`; here it
/// is a fixed buffer, because the brand string is a short, bounded value and
/// the C never frees its string either. The Linux and Windows arms are not
/// ported.
const DeviceContext = struct {
    description: [256]u8 = undefined,
    description_len: usize = 0,

    fn init(self: *DeviceContext) void {
        const fallback = "CPU";
        @memcpy(self.description[0..fallback.len], fallback);
        self.description[fallback.len] = 0;
        self.description_len = fallback.len;

        if (builtin.os.tag != .macos) return;
        var len: usize = self.description.len;
        if (sysctlbyname("machdep.cpu.brand_string", &self.description, &len, null, 0) == 0 and len > 0) {
            // `len` counts the NUL the call wrote.
            self.description_len = len - 1;
            self.description[self.description_len] = 0;
        }
    }
};

extern fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;
extern fn sysconf(name: c_int) c_long;

/// `_SC_PHYS_PAGES` and `_SC_PAGE_SIZE` on Darwin.
const sc_phys_pages: c_int = 200;
const sc_page_size: c_int = 29;

fn deviceGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    _ = dev;
    return "CPU";
}

/// Ports `ggml_backend_cpu_device_get_description` (ggml-cpu.cpp:359
/// @c1d0e7a00).
fn deviceGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    const d = impl.one(c.struct_ggml_backend_device, dev);
    const ctx: *DeviceContext = @ptrCast(@alignCast(d.context.?));
    return &ctx.description;
}

/// Ports `ggml_backend_cpu_device_get_memory` (ggml-cpu.cpp:365 @c1d0e7a00),
/// the non-Windows arm.
fn deviceGetMemory(dev: c.ggml_backend_dev_t, free: *usize, total: *usize) callconv(.c) void {
    _ = dev;
    const pages = sysconf(sc_phys_pages);
    const page_size = sysconf(sc_page_size);
    total.* = @intCast(pages * page_size);
    // "free" system memory is ill-defined, for practical purposes assume
    // that all of it is free:
    free.* = total.*;
}

fn deviceGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    _ = dev;
    return c.GGML_BACKEND_DEVICE_TYPE_CPU;
}

/// Ports `ggml_backend_cpu_device_get_props` (ggml-cpu.cpp:390 @c1d0e7a00).
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

fn deviceInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = dev;
    _ = params;
    return ggml_backend_cpu_init();
}

fn deviceGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    _ = dev;
    return c.ggml_backend_cpu_buffer_type();
}

fn deviceBufferFromHostPtr(dev: c.ggml_backend_dev_t, ptr: ?*anyopaque, size: usize, max_tensor_size: usize) callconv(.c) c.ggml_backend_buffer_t {
    _ = dev;
    _ = max_tensor_size;
    return c.ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

/// Ports `ggml_backend_cpu_device_supports_op` (ggml-cpu.cpp:424 @c1d0e7a00).
/// `op->src[i]` narrowed, or null.
///
/// Ports nothing by itself — the C writes `const struct ggml_tensor * src0
/// = op->src[0];`, which is a plain copy that may be null. This is that,
/// plus the `[*c]` narrowing Zig 0.16 needs before any `.ne[i]` on it.
inline fn srcOf(p: [*c]Tensor) ?*const Tensor {
    return if (p == null) null else impl.one(Tensor, p);
}

fn deviceSupportsOp(dev: c.ggml_backend_dev_t, op: *const Tensor) callconv(.c) bool {
    // The C copies the two raw pointers up front and lets each arm below
    // assume its own are non-null. Both must stay **optional** here: a
    // leaf weight reaches this with `op == GGML_OP_NONE` and no sources at
    // all, and narrowing a null `[*c]` to `*T` traps in a safe build.
    // Measured — eagerly narrowing them aborted `llama_init_from_model` in
    // Debug while passing every ReleaseFast gate, where the cast is
    // unchecked. `make parity-cli` is what caught it.
    //
    // Narrowed rather than left as `[*c]` because Zig 0.16 types
    // `p.*.ne[i]` on a C pointer as the whole array and indexes in steps of
    // its size. See `CLAUDE.md`, "Threads in ported code".
    const src0 = srcOf(op.src[0]);
    const src1 = srcOf(op.src[1]);

    if (op.op == c.GGML_OP_NONE or op.op == c.GGML_OP_RESHAPE or op.op == c.GGML_OP_VIEW or
        op.op == c.GGML_OP_PERMUTE or op.op == c.GGML_OP_TRANSPOSE)
    {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to
    // reduce overhead, increase if necessary
    for (0..4) |i| {
        const s = op.src[i] orelse continue;
        const buffer = s.*.buffer orelse continue;
        if (isExtraBufferType(buffer.*.buft)) {
            const buf_extra: *const extra.ExtraBufferType = @ptrCast(@alignCast(buffer.*.buft.*.context.?));
            return buf_extra.vtable.supports_op(buf_extra, dev, op);
        }
    }

    switch (op.op) {
        c.GGML_OP_CPY, c.GGML_OP_SET_ROWS => {
            // missing type_traits.from_float
            return op.type != c.GGML_TYPE_IQ3_XXS and op.type != c.GGML_TYPE_IQ3_S and
                op.type != c.GGML_TYPE_IQ2_XXS and op.type != c.GGML_TYPE_IQ2_XS and
                op.type != c.GGML_TYPE_IQ2_S and op.type != c.GGML_TYPE_IQ1_S and
                op.type != c.GGML_TYPE_IQ1_M;
        },
        c.GGML_OP_MUL_MAT => {
            return src1.?.type == c.GGML_TYPE_F32 or
                src1.?.type == c.ggml_get_type_traits_cpu(src0.?.type).*.vec_dot_type;
        },
        c.GGML_OP_SOFT_MAX_BACK => {
            if (src0.?.type != c.GGML_TYPE_F32 or src1.?.type != c.GGML_TYPE_F32) return false;
            var max_bias: f32 = 0.0;
            const params: [*]const f32 = @ptrCast(&op.op_params);
            @memcpy(@as([*]u8, @ptrCast(&max_bias))[0..4], @as([*]const u8, @ptrCast(params + 1))[0..4]);
            return max_bias == 0.0;
        },
        c.GGML_OP_IM2COL_BACK => {
            return src0.?.type == c.GGML_TYPE_F32 and
                (src1.?.type == c.GGML_TYPE_F32 or src1.?.type == c.GGML_TYPE_F16);
        },
        c.GGML_OP_GET_ROWS_BACK => {
            return src0.?.type == c.GGML_TYPE_F32 or src0.?.type == c.GGML_TYPE_F16;
        },
        c.GGML_OP_OUT_PROD => {
            return (src0.?.type == c.GGML_TYPE_F32 or
                ((src0.?.type == c.GGML_TYPE_F16 or c.ggml_is_quantized(src0.?.type)) and
                    src0.?.ne[2] == src1.?.ne[2] and src0.?.ne[3] == src1.?.ne[3])) and
                src1.?.type == c.GGML_TYPE_F32 and op.type == c.GGML_TYPE_F32;
        },
        c.GGML_OP_CONV_2D => return c.ggml_is_contiguous(src0.?),
        c.GGML_OP_SSM_SCAN => {
            return impl.getOpParamsI32(op, 0) == 1 or impl.one(Tensor, op.src[3]).ne[0] == 1;
        },
        else => return true,
    }
}

/// Ports `ggml_backend_cpu_device_supports_buft` (ggml-cpu.cpp:482
/// @c1d0e7a00).
fn deviceSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = dev;
    return c.ggml_backend_buft_is_host(buft) or isExtraBufferType(buft);
}

/// Ports `ggml_backend_cpu_device_i` (ggml-cpu.cpp:487 @c1d0e7a00).
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

fn regGetName(reg: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = reg;
    return "CPU";
}

fn regGetDeviceCount(reg: c.ggml_backend_reg_t) callconv(.c) usize {
    _ = reg;
    return 1;
}

var device_ctx: DeviceContext = .{};
var cpu_device: c.struct_ggml_backend_device = undefined;
var device_ready = std.atomic.Value(bool).init(false);
var device_mutex: std.c.pthread_mutex_t = .{};

fn ensureDevice(reg: c.ggml_backend_reg_t) void {
    if (device_ready.load(.acquire)) return;
    _ = std.c.pthread_mutex_lock(&device_mutex);
    defer _ = std.c.pthread_mutex_unlock(&device_mutex);
    if (device_ready.load(.acquire)) return;
    device_ctx.init();
    cpu_device = .{ .iface = device_i, .reg = reg, .context = &device_ctx };
    device_ready.store(true, .release);
}

/// Ports `ggml_backend_cpu_reg_get_device` (ggml-cpu.cpp:519 @c1d0e7a00).
///
/// The C's two `static` locals are initialised on first call; `std.once`
/// is the same guarantee, and the device's `reg` back-pointer is captured
/// the same way.
fn regGetDevice(reg: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    impl.assert(index == 0, "index == 0");
    ensureDevice(reg);
    return &cpu_device;
}

/// Ports `ggml_backend_cpu_get_features` (ggml-cpu.cpp:534 @c1d0e7a00).
///
/// Only the entries whose predicate can be true on this target are kept:
/// the x86, RISC-V, POWER, s390x and WASM arms are all compiled out. SVE,
/// SME and MATMUL_INT8 stay because they are runtime checks on ARM even
/// though this build does not select them.
var features: [10]c.ggml_backend_feature = undefined;
var features_n: usize = 0;
var features_ready = std.atomic.Value(bool).init(false);
var features_mutex: std.c.pthread_mutex_t = .{};

fn pushFeature(name: [*c]const u8, value: [*c]const u8) void {
    features[features_n] = .{ .name = name, .value = value };
    features_n += 1;
}

fn ensureFeatures() void {
    if (features_ready.load(.acquire)) return;
    _ = std.c.pthread_mutex_lock(&features_mutex);
    defer _ = std.c.pthread_mutex_unlock(&features_mutex);
    if (features_ready.load(.acquire)) return;
    ggml_cpu_init();
    if (c.ggml_cpu_has_neon() != 0) pushFeature("NEON", "1");
    if (c.ggml_cpu_has_arm_fma() != 0) pushFeature("ARM_FMA", "1");
    if (c.ggml_cpu_has_fp16_va() != 0) pushFeature("FP16_VA", "1");
    if (c.ggml_cpu_has_matmul_int8() != 0) pushFeature("MATMUL_INT8", "1");
    if (c.ggml_cpu_has_sve() != 0) pushFeature("SVE", "1");
    if (c.ggml_cpu_has_dotprod() != 0) pushFeature("DOTPROD", "1");
    if (c.ggml_cpu_has_sme() != 0) pushFeature("SME", "1");
    if (c.ggml_cpu_has_llamafile() != 0) pushFeature("LLAMAFILE", "1");
    pushFeature("ACCELERATE", "1");
    pushFeature("REPACK", "1");
    pushFeature(null, null);
    features_ready.store(true, .release);
}

fn regGetFeatures(reg: c.ggml_backend_reg_t) callconv(.c) [*c]c.ggml_backend_feature {
    _ = reg;
    ensureFeatures();
    return &features;
}

/// Ports `ggml_backend_cpu_get_proc_address` (ggml-cpu.cpp:651 @c1d0e7a00).
fn regGetProcAddress(reg: c.ggml_backend_reg_t, name: [*c]const u8) callconv(.c) ?*anyopaque {
    _ = reg;
    const n = std.mem.span(@as([*:0]const u8, @ptrCast(name)));
    const table = .{
        .{ "ggml_backend_set_n_threads", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_n_threads))) },
        .{ "ggml_backend_dev_get_extra_bufts", @as(?*anyopaque, @ptrCast(@constCast(&deviceGetExtraBuffersType))) },
        .{ "ggml_backend_get_features", @as(?*anyopaque, @ptrCast(@constCast(&regGetFeatures))) },
        .{ "ggml_backend_set_abort_callback", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_abort_callback))) },
        .{ "ggml_backend_cpu_numa_init", @as(?*anyopaque, @ptrCast(@constCast(&c.ggml_numa_init))) },
        .{ "ggml_backend_cpu_is_numa", @as(?*anyopaque, @ptrCast(@constCast(&c.ggml_is_numa))) },
        .{ "ggml_backend_cpu_set_use_ref", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_use_ref))) },
        // threadpool - the C's own TODO: move to ggml-base
        .{ "ggml_threadpool_new", @as(?*anyopaque, @ptrCast(@constCast(&c.ggml_threadpool_new))) },
        .{ "ggml_threadpool_free", @as(?*anyopaque, @ptrCast(@constCast(&c.ggml_threadpool_free))) },
        .{ "ggml_backend_cpu_set_threadpool", @as(?*anyopaque, @ptrCast(@constCast(&ggml_backend_cpu_set_threadpool))) },
    };
    inline for (table) |entry| {
        if (std.mem.eql(u8, n, entry[0])) return entry[1];
    }
    return null;
}

/// Ports `ggml_backend_cpu_reg_i` (ggml-cpu.cpp:692 @c1d0e7a00).
const reg_i = c.struct_ggml_backend_reg_i{
    .get_name = regGetName,
    .get_device_count = regGetDeviceCount,
    .get_device = regGetDevice,
    .get_proc_address = regGetProcAddress,
};

var cpu_reg = c.struct_ggml_backend_reg{
    .api_version = c.GGML_BACKEND_API_VERSION,
    .iface = reg_i,
    .context = null,
};

/// Ports `ggml_backend_cpu_reg` (ggml-cpu.cpp:699 @c1d0e7a00).
///
/// Return: the CPU registration, a process-lifetime singleton.
pub export fn ggml_backend_cpu_reg() callconv(.c) c.ggml_backend_reg_t {
    // init CPU feature detection
    ggml_cpu_init();
    return &cpu_reg;
}
