//! The "meta" backend: one device, buffer type, buffer and backend standing in
//! for several "simple" ones, so a model can be split tensor-parallel across
//! GPUs (`LLAMA_SPLIT_MODE_TENSOR`).
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-backend-meta.cpp` (v0.3.0,
//! `c1d0e7a00`), with the declarations it implements from
//! `llama.cpp/ggml/include/ggml-backend.h` (the split axis, the split state and
//! `ggml_backend_meta_device`) and `llama.cpp/ggml/src/ggml-backend-impl.h`
//! (the `is_meta` predicates and the three backend accessors). Each
//! declaration below names the C++ it replaces and the line it began at. This
//! file exports the same eight C symbols with the same signatures, so the C++
//! that calls into it (`llama.cpp`, `llama-model.cpp`) links unchanged.
//!
//! # What it does
//!
//! Every statically allocated tensor gets a *split state* from a callback the
//! caller supplies — libllama's is `llama_meta_device_get_split_state` — that
//! says which axis it is cut along and how many elements each device holds,
//! possibly in several repeated segments. Every compute tensor's split state
//! is then *derived* from its sources, op by op, in `Propagate.calculate`.
//! Each meta tensor is backed by one "simple" tensor per device, built by
//! `bufferInitTensorImpl`; `set_tensor` and `get_tensor` scatter and gather
//! through them. A graph is computed by cutting it into subgraphs wherever a
//! node holds a `PARTIAL` sum, running each subgraph on every device, and
//! all-reducing the partial sums in between — through the backend's own
//! collective if it registers `ggml_backend_comm_allreduce_tensor` (NCCL for
//! CUDA), and otherwise through a butterfly of copies and `ADD`s.
//!
//! # Ungated on this machine (one GPU)
//!
//! The reference accepts the same device more than once, which is what lets
//! `scripts/meta-diff` run this with two, three and four Metal backends on one
//! GPU, and with Metal beside the CPU. That reaches the butterfly all-reduce,
//! the non-power-of-two fold, the segmented Qwen3.5 weights and the zero-slice
//! paths. What it cannot reach is marked "ungated on this machine (one GPU)"
//! below: chiefly the `comm_allreduce` collective, which no backend on macOS
//! registers. Those paths are ported in full regardless, because they are what
//! a CUDA build will run.
//!
//! # Two quirks that are reproduced, not fixed
//!
//! - **The device and buffer type are named `Meta()`.** Both context
//!   constructors move their `std::vector` argument into the member and then
//!   build the name by looping over the *parameter*, which shadows the member
//!   and is by then empty. The backend's name is built from a different loop
//!   and is `Meta(MTL0,MTL0)`. Measured in the reference's own log:
//!   `load_tensors: layer 0 assigned to device Meta()`.
//! - **`ggml_backend_meta_device` and the buffer-type map are process-wide
//!   statics** that are never freed, and the C++ notes they are not
//!   thread-safe. Neither is this.
//!
//! # An upstream use-after-free, reproduced
//!
//! When a graph raises `max_nnodes`, `backendGraphCompute` frees the context
//! holding every subgraph and re-creates only the first `n_subgraphs` of
//! them, at line 2223 of the C++. A later graph with more
//! subgraphs, but no more nodes, then runs on a `cgraph_main` in freed
//! memory. Measured with `scripts/meta-diff`'s harness: a 19-node graph with one
//! subgraph followed by a 15-node graph with two crashed the port and not the
//! reference, which was heap luck, not a difference. Graphs re-created when
//! only the subgraph count grows are also sized for the graph at hand rather
//! than for `max_nnodes`, so a later, larger graph can overrun them; that one
//! is reasoned from the source, not observed. libllama reserves its largest
//! graph first, which keeps it clear of both. The port keeps the C++'s
//! behaviour, since fixing it here would make the two disagree.

const std = @import("std");
const impl = @import("impl.zig");
const c = impl.c;

/// The heap the rest of ggml uses. A function entered through a C ABI has no
/// allocator to be handed one, the same reasoning as `backend.zig`.
const allocator = std.heap.c_allocator;

/// Aborts on a failed allocation, standing in for the `std::bad_alloc` the C++
/// would let escape to `std::terminate`.
fn oom() noreturn {
    impl.abort("ggml-backend-meta: out of memory");
}

/// Ports `GGML_ABORT` with format arguments (ggml.h:287 @c1d0e7a00).
///
/// `impl.abort` takes only a literal; the meta backend's aborts name tensors.
///
/// Parameters:
/// - `line`: the C++ line of the abort, passed through as ggml's `__LINE__`.
/// - `fmt`: printf format.
/// - `args`: its arguments.
fn abortFmt(comptime line: c_int, comptime fmt: [*:0]const u8, args: anytype) noreturn {
    @call(.auto, c.ggml_abort, .{ "ggml-backend-meta.cpp", line, fmt } ++ args);
}

extern fn atoi(s: [*:0]const u8) c_int;

const SplitState = c.struct_ggml_backend_meta_split_state;
const Axis = c.enum_ggml_backend_meta_split_axis;

const axis_0: Axis = c.GGML_BACKEND_SPLIT_AXIS_0;
const axis_1: Axis = c.GGML_BACKEND_SPLIT_AXIS_1;
const axis_2: Axis = c.GGML_BACKEND_SPLIT_AXIS_2;
const axis_3: Axis = c.GGML_BACKEND_SPLIT_AXIS_3;
const axis_mirrored: Axis = c.GGML_BACKEND_SPLIT_AXIS_MIRRORED;
const axis_partial: Axis = c.GGML_BACKEND_SPLIT_AXIS_PARTIAL;
const axis_none: Axis = c.GGML_BACKEND_SPLIT_AXIS_NONE;
const axis_unknown: Axis = c.GGML_BACKEND_SPLIT_AXIS_UNKNOWN;

const max_dims: usize = c.GGML_MAX_DIMS;
const max_src: usize = c.GGML_MAX_SRC;
/// Ports `GGML_BACKEND_META_MAX_DEVICES` (ggml-backend.h:360 @c1d0e7a00).
const max_devices: usize = c.GGML_BACKEND_META_MAX_DEVICES;

/// True for an axis that names a tensor dimension — the C's recurring
/// `axis >= 0 && axis < GGML_MAX_DIMS`. The enum promotes to `int` there, and
/// every value it ever holds is non-negative, so the lower bound is vacuous.
inline fn isDim(axis: Axis) bool {
    return axis < max_dims;
}

/// The C's brace initialiser `{axis, {0}, {1}, 1}`: every `ne` zero, `nr[0]`
/// one and the rest zero, one segment.
fn ssMake(axis: Axis) SplitState {
    var ss = std.mem.zeroes(SplitState);
    ss.axis = axis;
    ss.nr[0] = 1;
    ss.n_segments = 1;
    return ss;
}

/// The axis as the C's `int` — what `int split_dim = split_state.axis` holds.
inline fn axisInt(axis: Axis) c_int {
    return @intCast(axis);
}

/// Source `i` of `t`, narrowed: `[*c]` array fields must not be indexed
/// through (the `p.*.arr[i]` miscompile in `CLAUDE.md`), so every source is
/// read through this before its `ne` or `nb` is touched.
inline fn srcOf(t: *const c.ggml_tensor, i: usize) *c.ggml_tensor {
    return impl.one(c.ggml_tensor, t.src[i]);
}

/// Source `i` of `t`, or null.
inline fn srcOpt(t: *const c.ggml_tensor, i: usize) ?*c.ggml_tensor {
    return t.src[i];
}

/// True for the C's `tensor->src[i] == nullptr || tensor->src[i] == tensor`.
inline fn srcAbsentOrSelf(t: *const c.ggml_tensor, i: usize) bool {
    const s = srcOpt(t, i) orelse return true;
    return s == t;
}

/// Builds `"Meta(" + names joined by "," + ")"`, the shape of all three name
/// loops in the C++.
///
/// Parameters:
/// - `names`: the parts, in order.
///
/// Return: a NUL-terminated string owned by the caller.
fn metaName(names: []const [*c]const u8) [:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    out.appendSlice(allocator, "Meta(") catch oom();
    for (names, 0..) |n, i| {
        if (i > 0) out.append(allocator, ',') catch oom();
        out.appendSlice(allocator, std.mem.span(n)) catch oom();
    }
    out.append(allocator, ')') catch oom();
    return out.toOwnedSliceSentinel(allocator, 0) catch oom();
}

/// Ports `ggml_backend_meta_split_axis_name` (ggml-backend-meta.cpp:27 @c1d0e7a00).
///
/// Return: a static string; aborts on a value outside the enum.
pub export fn ggml_backend_meta_split_axis_name(split_axis: c.enum_ggml_backend_meta_split_axis) callconv(.c) [*c]const u8 {
    return switch (split_axis) {
        axis_0 => "0",
        axis_1 => "1",
        axis_2 => "2",
        axis_3 => "3",
        axis_mirrored => "MIRRORED",
        axis_partial => "PARTIAL",
        axis_none => "NONE",
        axis_unknown => "UNKNOWN",
        else => impl.abort("fatal error"),
    };
}

// -----------------------------------------------------------------------------
// Meta backend device

/// Ports `struct ggml_backend_meta_device_context` (ggml-backend-meta.cpp:54 @c1d0e7a00).
const DeviceContext = struct {
    simple_devs: []c.ggml_backend_dev_t,
    get_split_state: c.ggml_backend_meta_get_split_state_t,
    get_split_state_ud: ?*anyopaque,

    name: [:0]u8,
    description: [:0]u8,

    /// Ports the `ggml_backend_meta_device_context` constructor (ggml-backend-meta.cpp:62 @c1d0e7a00).
    ///
    /// **The name loop runs zero times.** The C++ initialises the member
    /// `simple_devs` by moving from the parameter of the same name, and the
    /// body's loop then reads `simple_devs.size()` — which names the
    /// *parameter*, empty after the move (`std::vector`'s move constructor
    /// guarantees it). So the name and description are `Meta()`; the loop is
    /// kept with an empty list so it reads against the C.
    ///
    /// Parameters:
    /// - `simple_devs`: the devices, owned by the result from here on.
    /// - `get_split_state`, `get_split_state_ud`: the caller's callback.
    fn init(simple_devs: []c.ggml_backend_dev_t, get_split_state: c.ggml_backend_meta_get_split_state_t, get_split_state_ud: ?*anyopaque) DeviceContext {
        const moved_from: []const [*c]const u8 = &.{};
        return .{
            .simple_devs = simple_devs,
            .get_split_state = get_split_state,
            .get_split_state_ud = get_split_state_ud,
            .name = metaName(moved_from),
            .description = metaName(moved_from),
        };
    }

    fn deinit(self: *DeviceContext) void {
        allocator.free(self.description);
        allocator.free(self.name);
        allocator.free(self.simple_devs);
    }

    /// Equivalence under the operator< of `ggml_backend_meta_device_context`
    /// (ggml-backend-meta.cpp:79 @c1d0e7a00), which compares the tuple
    /// `(simple_devs, get_split_state, get_split_state_ud)`.
    fn eql(self: *const DeviceContext, other: *const DeviceContext) bool {
        if (self.simple_devs.len != other.simple_devs.len) return false;
        for (self.simple_devs, other.simple_devs) |a, b| {
            if (a != b) return false;
        }
        return self.get_split_state == other.get_split_state and self.get_split_state_ud == other.get_split_state_ud;
    }
};

inline fn deviceCtx(dev: c.ggml_backend_dev_t) *DeviceContext {
    return @ptrCast(@alignCast(dev.*.context));
}

/// Ports `ggml_backend_meta_device_get_name` (ggml-backend-meta.cpp:87 @c1d0e7a00).
fn deviceGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    return deviceCtx(dev).name.ptr;
}

/// Ports `ggml_backend_meta_device_get_description` (ggml-backend-meta.cpp:93 @c1d0e7a00).
fn deviceGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    return deviceCtx(dev).description.ptr;
}

/// Ports `ggml_backend_meta_device_get_memory` (ggml-backend-meta.cpp:99 @c1d0e7a00).
///
/// The sum over the simple devices — of the same physical device, when one is
/// listed twice, which counts its memory twice. The C++ does the same.
fn deviceGetMemory(dev: c.ggml_backend_dev_t, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    const meta_dev_ctx = deviceCtx(dev);
    free.* = 0;
    total.* = 0;
    for (meta_dev_ctx.simple_devs) |simple_dev| {
        var tmp_free: usize = undefined;
        var tmp_total: usize = undefined;
        c.ggml_backend_dev_memory(simple_dev, &tmp_free, &tmp_total);
        free.* += tmp_free;
        total.* += tmp_total;
    }
}

/// Ports `ggml_backend_meta_device_get_type` (ggml-backend-meta.cpp:112 @c1d0e7a00).
fn deviceGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    _ = dev;
    return c.GGML_BACKEND_DEVICE_TYPE_META;
}

/// Ports `ggml_backend_meta_device_get_props` (ggml-backend-meta.cpp:118 @c1d0e7a00).
///
/// The capabilities start from the C++'s placeholders — async and mmap on,
/// host buffers, host pointers and events off ("Not implemented") — and are
/// AND-ed with every simple device's.
fn deviceGetProps(dev: c.ggml_backend_dev_t, props_: [*c]c.struct_ggml_backend_dev_props) callconv(.c) void {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    const meta_dev_ctx = deviceCtx(dev);
    const props = impl.one(c.struct_ggml_backend_dev_props, props_);

    // TODO replace placeholders
    props.name = deviceGetName(dev);
    props.description = deviceGetDescription(dev);
    props.type = deviceGetType(dev);
    props.device_id = null;

    deviceGetMemory(dev, &props.memory_free, &props.memory_total);

    props.caps = .{
        .async = true,
        .host_buffer = false, // Not implemented.
        .buffer_from_host_ptr = false, // Not implemented.
        .events = false, // Not implemented.
        .mmap_support = true,
    };
    for (meta_dev_ctx.simple_devs) |simple_dev| {
        var tmp_props: c.struct_ggml_backend_dev_props = undefined;
        c.ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props.caps.async = props.caps.async and tmp_props.caps.async;
        props.caps.host_buffer = props.caps.host_buffer and tmp_props.caps.host_buffer;
        props.caps.buffer_from_host_ptr = props.caps.buffer_from_host_ptr and tmp_props.caps.buffer_from_host_ptr;
        props.caps.events = props.caps.events and tmp_props.caps.events;
        props.caps.mmap_support = props.caps.mmap_support and tmp_props.caps.mmap_support;
    }
}

/// Ports `ggml_backend_meta_device_supports_op` (ggml-backend-meta.cpp:154 @c1d0e7a00).
///
/// Return: true when every simple device supports `op` (short-circuiting, as
/// `std::all_of` does).
fn deviceSupportsOp(dev: c.ggml_backend_dev_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    for (deviceCtx(dev).simple_devs) |simple_dev| {
        if (!c.ggml_backend_dev_supports_op(simple_dev, op)) return false;
    }
    return true;
}

/// Ports `ggml_backend_meta_device_supports_buft` (ggml-backend-meta.cpp:161 @c1d0e7a00).
///
/// Return: true only for a meta buffer type over the same simple devices, in
/// the same order.
fn deviceSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    const dev_buft = c.ggml_backend_buft_get_device(buft);
    if (!devIsMeta(dev_buft)) {
        return false;
    }
    const meta_dev_ctx = deviceCtx(dev);
    const meta_buft_dev_ctx = deviceCtx(dev_buft);
    if (meta_dev_ctx.simple_devs.len != meta_buft_dev_ctx.simple_devs.len) {
        return false;
    }
    for (meta_dev_ctx.simple_devs, meta_buft_dev_ctx.simple_devs) |a, b| {
        if (a != b) return false;
    }
    return true;
}

/// Ports `ggml_backend_meta_device_iface` (ggml-backend-meta.cpp:180 @c1d0e7a00).
const device_iface: c.struct_ggml_backend_device_i = .{
    .get_name = &deviceGetName,
    .get_description = &deviceGetDescription,
    .get_memory = &deviceGetMemory,
    .get_type = &deviceGetType,
    .get_props = &deviceGetProps,
    .init_backend = &deviceInitBackend,
    .get_buffer_type = &deviceGetBufferType,
    .get_host_buffer_type = &deviceGetHostBufferType,
    .buffer_from_host_ptr = null,
    .supports_op = &deviceSupportsOp,
    .supports_buft = &deviceSupportsBuft,
    .offload_op = null,
    .event_new = null,
    .event_free = null,
    .event_synchronize = null,
};

/// Ports `ggml_backend_dev_is_meta` (ggml-backend-meta.cpp:198 @c1d0e7a00).
///
/// Identity is the `get_name` function's address, so a device made by another
/// implementation of this file (the reference, in `scripts/meta-diff`) is not
/// one of ours.
fn devIsMeta(dev: c.ggml_backend_dev_t) bool {
    return dev != null and dev.*.iface.get_name == device_iface.get_name;
}

/// Ports `ggml_backend_meta_dev_n_devs` (ggml-backend-meta.cpp:202 @c1d0e7a00).
fn devNDevs(meta_dev: c.ggml_backend_dev_t) usize {
    impl.assert(devIsMeta(meta_dev), "ggml_backend_dev_is_meta(meta_dev)");
    return deviceCtx(meta_dev).simple_devs.len;
}

/// Ports `ggml_backend_meta_dev_simple_dev` (ggml-backend-meta.cpp:208 @c1d0e7a00).
fn devSimpleDev(meta_dev: c.ggml_backend_dev_t, index: usize) c.ggml_backend_dev_t {
    impl.assert(devIsMeta(meta_dev), "ggml_backend_dev_is_meta(meta_dev)");
    const meta_dev_ctx = deviceCtx(meta_dev);
    impl.assert(index < meta_dev_ctx.simple_devs.len, "index < meta_dev_ctx->simple_devs.size()");
    return meta_dev_ctx.simple_devs[index];
}

/// One registered meta device: the C++ keeps the context in `ctxs` and the
/// device in `meta_devs`; here they share an allocation, so the device's
/// `context` pointer and the device itself stay put for the process.
const DeviceEntry = struct {
    ctx: DeviceContext,
    dev: c.struct_ggml_backend_device,
};

/// The C++'s two function-local statics in `ggml_backend_meta_device`. Only
/// ever searched for an equal key, never iterated in order, so a list stands
/// in for the `std::map`.
var meta_devs: std.ArrayList(*DeviceEntry) = .empty;

/// Ports `ggml_backend_meta_device` (ggml-backend-meta.cpp:215 @c1d0e7a00).
///
/// Interned: the same devices, callback and user data give back the same
/// device pointer, for the life of the process.
///
/// Parameters:
/// - `devs`: `n_devs` simple devices; the same device may appear twice.
/// - `get_split_state`, `get_split_state_ud`: assigns split states to
///   statically allocated tensors.
///
/// Return: the meta device; never freed.
pub export fn ggml_backend_meta_device(
    devs: [*c]c.ggml_backend_dev_t,
    n_devs: usize,
    get_split_state: c.ggml_backend_meta_get_split_state_t,
    get_split_state_ud: ?*anyopaque,
) callconv(.c) c.ggml_backend_dev_t {
    impl.assert(n_devs <= max_devices, "n_devs <= GGML_BACKEND_META_MAX_DEVICES");
    // TODO: this is not thread-safe - needs to be fixed

    const simple_devs = allocator.alloc(c.ggml_backend_dev_t, n_devs) catch oom();
    for (0..n_devs) |i| {
        simple_devs[i] = devs[i];
    }
    var ctx = DeviceContext.init(simple_devs, get_split_state, get_split_state_ud);

    for (meta_devs.items) |entry| {
        if (entry.ctx.eql(&ctx)) {
            ctx.deinit();
            return &entry.dev;
        }
    }

    const entry = allocator.create(DeviceEntry) catch oom();
    entry.ctx = ctx;
    entry.dev = .{
        .iface = device_iface,
        .reg = null,
        .context = &entry.ctx,
    };
    meta_devs.append(allocator, entry) catch oom();
    return &entry.dev;
}

// -----------------------------------------------------------------------------
// Meta backend buffer type

/// Ports `struct ggml_backend_meta_buffer_type_context` (ggml-backend-meta.cpp:251 @c1d0e7a00).
const BufferTypeContext = struct {
    simple_bufts: []c.ggml_backend_buffer_type_t,

    name: [:0]u8,

    /// Ports the `ggml_backend_meta_buffer_type_context` constructor (ggml-backend-meta.cpp:256 @c1d0e7a00).
    ///
    /// The same shadowed-parameter quirk as `DeviceContext.init`: the loop
    /// reads the moved-from argument, so the name is `Meta()`.
    fn init(simple_bufts: []c.ggml_backend_buffer_type_t) BufferTypeContext {
        const moved_from: []const [*c]const u8 = &.{};
        return .{ .simple_bufts = simple_bufts, .name = metaName(moved_from) };
    }
};

inline fn buftCtx(buft: c.ggml_backend_buffer_type_t) *BufferTypeContext {
    return @ptrCast(@alignCast(buft.*.context));
}

/// Ports `ggml_backend_meta_buft_n_bufts` (ggml-backend-meta.cpp:272 @c1d0e7a00).
fn buftNBufts(meta_buft: c.ggml_backend_buffer_type_t) usize {
    impl.assert(ggml_backend_buft_is_meta(meta_buft), "ggml_backend_buft_is_meta(meta_buft)");
    return buftCtx(meta_buft).simple_bufts.len;
}

/// Ports `ggml_backend_meta_buffer_type_get_name` (ggml-backend-meta.cpp:278 @c1d0e7a00).
fn buftGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    impl.assert(ggml_backend_buft_is_meta(buft), "ggml_backend_buft_is_meta(buft)");
    return buftCtx(buft).name.ptr;
}

/// Ports `ggml_backend_meta_buft_simple_buft` (ggml-backend-meta.cpp:284 @c1d0e7a00).
fn buftSimpleBuft(meta_buft: c.ggml_backend_buffer_type_t, index: usize) c.ggml_backend_buffer_type_t {
    impl.assert(ggml_backend_buft_is_meta(meta_buft), "ggml_backend_buft_is_meta(meta_buft)");
    const meta_buft_ctx = buftCtx(meta_buft);
    impl.assert(index < meta_buft_ctx.simple_bufts.len, "index < meta_buft_ctx->simple_bufts.size()");
    return meta_buft_ctx.simple_bufts[index];
}

/// Ports `ggml_backend_meta_buffer_type_get_alignment` (ggml-backend-meta.cpp:293 @c1d0e7a00).
///
/// Return: the largest simple alignment, which must be a multiple of every
/// other one.
fn buftGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    const n_simple_bufts = buftNBufts(buft);
    var max_alignment: usize = 1;
    for (0..n_simple_bufts) |i| {
        const alignment = c.ggml_backend_buft_get_alignment(buftSimpleBuft(buft, i));
        max_alignment = @max(max_alignment, alignment);
        impl.assert(max_alignment % alignment == 0, "max_alignment % alignment == 0");
    }
    return max_alignment;
}

/// Ports `ggml_backend_meta_buffer_type_get_max_size` (ggml-backend-meta.cpp:304 @c1d0e7a00).
fn buftGetMaxSize(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    const n_simple_bufts = buftNBufts(buft);
    var max_size: usize = std.math.maxInt(usize);
    for (0..n_simple_bufts) |i| {
        max_size = @min(max_size, c.ggml_backend_buft_get_max_size(buftSimpleBuft(buft, i)));
    }
    return max_size;
}

/// Ports `ggml_backend_meta_buffer_type_get_alloc_size` (ggml-backend-meta.cpp:313 @c1d0e7a00).
///
/// Asks each simple buffer type about the *whole* meta tensor and takes the
/// largest answer — an over-estimate of what a slice needs, as in the C++.
fn buftGetAllocSize(buft: c.ggml_backend_buffer_type_t, tensor: [*c]const c.ggml_tensor) callconv(.c) usize {
    const n_simple_bufts = buftNBufts(buft);
    var max_alloc_size: usize = 0;
    for (0..n_simple_bufts) |i| {
        const alloc_size = c.ggml_backend_buft_get_alloc_size(buftSimpleBuft(buft, i), tensor);
        max_alloc_size = @max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

/// Ports `ggml_backend_meta_buffer_type_is_host` (ggml-backend-meta.cpp:323 @c1d0e7a00).
fn buftIsHost(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    const n_simple_bufts = buftNBufts(buft);
    for (0..n_simple_bufts) |i| {
        if (!c.ggml_backend_buft_is_host(buftSimpleBuft(buft, i))) {
            return false;
        }
    }
    return true;
}

/// Ports `ggml_backend_meta_buffer_type_iface` (ggml-backend-meta.cpp:333 @c1d0e7a00).
const buffer_type_iface: c.struct_ggml_backend_buffer_type_i = .{
    .get_name = &buftGetName,
    .alloc_buffer = &buftAllocBuffer,
    .get_alignment = &buftGetAlignment,
    .get_max_size = &buftGetMaxSize,
    .get_alloc_size = &buftGetAllocSize,
    .is_host = &buftIsHost,
};

/// Ports `ggml_backend_buft_is_meta` (ggml-backend-meta.cpp:342 @c1d0e7a00).
pub export fn ggml_backend_buft_is_meta(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    return buft != null and buft.*.iface.get_name == buffer_type_iface.get_name;
}

/// One meta buffer type per meta device, at a stable address.
const BuftEntry = struct {
    dev: c.ggml_backend_dev_t,
    buft: c.struct_ggml_backend_buffer_type,
};

/// The C++'s function-local `std::map<dev, buffer_type>`; only searched by key.
var meta_bufts: std.ArrayList(*BuftEntry) = .empty;

/// Ports `ggml_backend_meta_device_get_buffer_type` (ggml-backend-meta.cpp:346 @c1d0e7a00).
///
/// Return: the device's meta buffer type, created on first request and kept
/// for the process.
fn deviceGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    for (meta_bufts.items) |entry| {
        if (entry.dev == dev) return &entry.buft;
    }

    const n_devs = devNDevs(dev);
    const simple_bufts = allocator.alloc(c.ggml_backend_buffer_type_t, n_devs) catch oom();
    for (0..n_devs) |i| {
        simple_bufts[i] = c.ggml_backend_dev_buffer_type(devSimpleDev(dev, i));
    }
    const buft_ctx = allocator.create(BufferTypeContext) catch oom();
    buft_ctx.* = BufferTypeContext.init(simple_bufts);

    const entry = allocator.create(BuftEntry) catch oom();
    entry.* = .{
        .dev = dev,
        .buft = .{
            .iface = buffer_type_iface,
            .device = dev,
            .context = buft_ctx,
        },
    };
    meta_bufts.append(allocator, entry) catch oom();
    return &entry.buft;
}

/// Ports `ggml_backend_meta_device_get_host_buffer_type` (ggml-backend-meta.cpp:373 @c1d0e7a00).
///
/// Return: the simple devices' shared host buffer type, or null when any of
/// them has none or they disagree.
///
/// Every non-null arm is **ungated on this machine (one GPU)**: neither the
/// Metal nor the CPU device has a host buffer type, so the loop always
/// returns at its first null. CUDA's pinned host buffer type is what would
/// reach the rest.
fn deviceGetHostBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    impl.assert(devIsMeta(dev), "ggml_backend_dev_is_meta(dev)");
    var host_buft: c.ggml_backend_buffer_type_t = null;
    for (deviceCtx(dev).simple_devs) |simple_dev| {
        const simple_host_buft = c.ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == null) {
            return null;
        }
        if (host_buft == null) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return null;
        }
    }
    return host_buft;
}

// -----------------------------------------------------------------------------
// Meta backend buffer

/// One meta tensor's simple tensors, indexed by device. The C++ holds a
/// `std::vector` of `n_bufs`; at most `GGML_BACKEND_META_MAX_DEVICES`, so an
/// array does.
const SimpleTensors = [max_devices]?*c.ggml_tensor;

/// Ports `struct ggml_backend_meta_simple_tensor_container` (ggml-backend-meta.cpp:399 @c1d0e7a00).
///
/// The `std::map` is only ever looked up, assigned and cleared — never
/// iterated — so a hash map stands in for it.
const SimpleTensorContainer = struct {
    ctxs: std.ArrayList(?*c.ggml_context) = .empty,
    simple_tensors: std.AutoHashMapUnmanaged(*const c.ggml_tensor, SimpleTensors) = .empty,

    /// Ports the sized `ggml_backend_meta_simple_tensor_container` constructor (ggml-backend-meta.cpp:403 @c1d0e7a00): one
    /// context per simple buffer. The default constructor (line 409), which
    /// makes none, is `.{}`.
    fn initCtxs(params: c.struct_ggml_init_params, n_simple: usize) SimpleTensorContainer {
        var self: SimpleTensorContainer = .{};
        self.ctxs.ensureTotalCapacity(allocator, n_simple) catch oom();
        for (0..n_simple) |_| {
            self.ctxs.appendAssumeCapacity(c.ggml_init(params));
        }
        return self;
    }

    /// The destructor: members in reverse declaration order, and the
    /// contexts last to first, as `~vector` destroys them.
    fn deinit(self: *SimpleTensorContainer) void {
        self.simple_tensors.deinit(allocator);
        var i = self.ctxs.items.len;
        while (i > 0) {
            i -= 1;
            if (self.ctxs.items[i]) |ctx| c.ggml_free(ctx);
        }
        self.ctxs.deinit(allocator);
    }
};

/// In `ggml_backend_meta_buffer_context`, ports `nbtc` (ggml-backend-meta.cpp:429 @c1d0e7a00):
/// the bytes of a tensor the split-state cache compares, everything but the
/// trailing padding.
const nbtc = @sizeOf(c.ggml_tensor) - @sizeOf(@FieldType(c.ggml_tensor, "padding"));

const CacheKey = struct {
    tensor: *const c.ggml_tensor,
    assume_sync: bool,
};

const CacheEntry = struct {
    ss: SplitState,
    bytes: [nbtc]u8,
};

/// Ports `struct ggml_backend_meta_buffer_context` (ggml-backend-meta.cpp:412 @c1d0e7a00).
const BufferContext = struct {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    stc_static: SimpleTensorContainer,
    stc_compute: [2]SimpleTensorContainer,
    stc_compute_index: c_int = 0,
    stc_compute_index_next: c_int = 0,
    /// Owning: `ggml_backend_buffer_ptr` in the C++. May hold nulls.
    bufs: std.ArrayList(c.ggml_backend_buffer_t) = .empty,

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    split_state_cache: std.AutoHashMapUnmanaged(CacheKey, CacheEntry) = .empty,

    debug: c_int,

    /// Ports the `ggml_backend_meta_buffer_context` constructor (ggml-backend-meta.cpp:434 @c1d0e7a00).
    ///
    /// Parameters:
    /// - the three containers, moved into the result.
    /// - `bufs`: simple buffers, owned by the result from here on.
    ///
    /// Return: a heap context, released by `destroy`.
    fn create(
        stc_static: SimpleTensorContainer,
        stc_compute_0: SimpleTensorContainer,
        stc_compute_1: SimpleTensorContainer,
        bufs: []const c.ggml_backend_buffer_t,
    ) *BufferContext {
        const self = allocator.create(BufferContext) catch oom();
        self.* = .{
            .stc_static = stc_static,
            .stc_compute = .{ stc_compute_0, stc_compute_1 },
            .debug = 0,
        };
        self.bufs.ensureTotalCapacity(allocator, bufs.len) catch oom();
        for (bufs) |buf| self.bufs.appendAssumeCapacity(buf);
        const GGML_META_DEBUG = std.c.getenv("GGML_META_DEBUG");
        self.debug = if (GGML_META_DEBUG) |s| atoi(s) else 0;
        return self;
    }

    /// The destructor: members in reverse declaration order — the cache, then
    /// the buffers (last to first, as `~vector` runs), then the containers.
    fn destroy(self: *BufferContext) void {
        self.split_state_cache.deinit(allocator);
        var i = self.bufs.items.len;
        while (i > 0) {
            i -= 1;
            if (self.bufs.items[i] != null) c.ggml_backend_buffer_free(self.bufs.items[i]);
        }
        self.bufs.deinit(allocator);
        self.stc_compute[1].deinit();
        self.stc_compute[0].deinit();
        self.stc_static.deinit();
        allocator.destroy(self);
    }

    /// Ports `get_simple_tensor_container` (ggml-backend-meta.cpp:448 @c1d0e7a00).
    fn getSimpleTensorContainer(self: *BufferContext, tensor: *const c.ggml_tensor) *SimpleTensorContainer {
        if (self.stc_static.simple_tensors.contains(tensor)) {
            return &self.stc_static;
        }
        return &self.stc_compute[@intCast(self.stc_compute_index)];
    }

    /// `split_state_cache[key]`: the entry, value-initialised (all zero) if it
    /// was absent, as `std::map::operator[]` does. The pointer is valid only
    /// until the cache is next changed.
    fn cacheIndex(self: *BufferContext, key: CacheKey) *CacheEntry {
        const gop = self.split_state_cache.getOrPut(allocator, key) catch oom();
        if (!gop.found_existing) gop.value_ptr.* = std.mem.zeroes(CacheEntry);
        return gop.value_ptr;
    }
};

inline fn bufferCtx(buffer: c.ggml_backend_buffer_t) *BufferContext {
    return @ptrCast(@alignCast(buffer.*.context));
}

/// The first `nbtc` bytes of a tensor, which the split-state cache compares.
inline fn tensorBytes(tensor: *const c.ggml_tensor) *const [nbtc]u8 {
    return @ptrCast(tensor);
}

/// Ports `ggml_backend_meta_buffer_free_buffer` (ggml-backend-meta.cpp:456 @c1d0e7a00).
fn bufferFreeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    impl.assert(ggml_backend_buffer_is_meta(buffer), "ggml_backend_buffer_is_meta(buffer)");
    bufferCtx(buffer).destroy();
}

/// Ports `ggml_backend_meta_buffer_n_bufs` (ggml-backend-meta.cpp:462 @c1d0e7a00).
fn bufferNBufs(meta_buf: c.ggml_backend_buffer_t) usize {
    impl.assert(ggml_backend_buffer_is_meta(meta_buf), "ggml_backend_buffer_is_meta(meta_buf)");
    return bufferCtx(meta_buf).bufs.items.len;
}

/// Ports `ggml_backend_meta_buffer_simple_buffer` (ggml-backend-meta.cpp:468 @c1d0e7a00).
fn bufferSimpleBuffer(meta_buf: c.ggml_backend_buffer_t, index: usize) c.ggml_backend_buffer_t {
    impl.assert(ggml_backend_buffer_is_meta(meta_buf), "ggml_backend_buffer_is_meta(meta_buf)");
    const buf_ctx = bufferCtx(meta_buf);
    impl.assert(index < buf_ctx.bufs.items.len, "index < buf_ctx->bufs.size()");
    return buf_ctx.bufs.items[index];
}

/// Ports `ggml_backend_meta_buffer_simple_tensor` (ggml-backend-meta.cpp:475 @c1d0e7a00).
///
/// Return: the slice of `tensor` on device `index`, or null when the tensor
/// has not been initialised in this buffer's current container.
fn bufferSimpleTensor(tensor: *const c.ggml_tensor, index: usize) ?*c.ggml_tensor {
    impl.assert(ggml_backend_buffer_is_meta(tensor.buffer), "ggml_backend_buffer_is_meta(tensor->buffer)");
    const buf_ctx = bufferCtx(tensor.buffer);
    impl.assert(index < buf_ctx.bufs.items.len, "index < buf_ctx->bufs.size()");

    const stc = buf_ctx.getSimpleTensorContainer(tensor);
    const simple = stc.simple_tensors.get(tensor) orelse return null;
    return simple[index];
}

/// The lambdas of `ggml_backend_meta_get_split_state` (ggml-backend-meta.cpp:490 @c1d0e7a00),
/// which all capture the same four values.
const Propagate = struct {
    stc: *SimpleTensorContainer,
    tensor: *const c.ggml_tensor,
    n_bufs: usize,
    assume_sync: bool,

    /// Ports the `split_states_equal` lambda (ggml-backend-meta.cpp:498 @c1d0e7a00):
    /// same axis and the same total per device, segments and repeats folded.
    fn equal(p: *const Propagate, a: *const SplitState, b: *const SplitState) bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (0..p.n_bufs) |j| {
            var sum_a: i64 = 0;
            for (0..a.n_segments) |s| {
                sum_a += a.ne[s * p.n_bufs + j] * @as(i64, a.nr[s]);
            }
            var sum_b: i64 = 0;
            for (0..b.n_segments) |s| {
                sum_b += b.ne[s * p.n_bufs + j] * @as(i64, b.nr[s]);
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    }

    /// Ports the `handle_generic` lambda (ggml-backend-meta.cpp:518 @c1d0e7a00):
    /// every source must agree, and a `scalar_only` op cannot take a dimension
    /// split.
    fn generic(p: *const Propagate, src_ss: []const SplitState, scalar_only: bool) SplitState {
        const tensor = p.tensor;
        var ret = ssMake(axis_none);
        for (0..max_src) |i| {
            if (srcAbsentOrSelf(tensor, i)) {
                continue;
            }
            if (ret.axis == axis_none) {
                ret = src_ss[i];
            } else if (!p.equal(&src_ss[i], &ret)) {
                ret = ssMake(axis_unknown);
                break;
            }
        }
        if (ret.axis == axis_none) {
            ret = ssMake(axis_unknown);
        }
        if (scalar_only and isDim(ret.axis)) {
            ret = ssMake(axis_unknown);
        }
        impl.assert(ret.axis != axis_unknown, "ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN");
        return ret;
    }

    /// Ports the `handle_per_row` lambda (ggml-backend-meta.cpp:542 @c1d0e7a00).
    fn perRow(p: *const Propagate, src_ss: []const SplitState) SplitState {
        _ = p;
        impl.assert(src_ss[0].axis != axis_0, "src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0");
        return src_ss[0];
    }

    /// Ports the `handle_bin_bcast` lambda (ggml-backend-meta.cpp:548 @c1d0e7a00).
    fn binBcast(p: *const Propagate, src_ss: []const SplitState) SplitState {
        const tensor = p.tensor;
        if (isDim(src_ss[0].axis) and
            srcOf(tensor, 1).ne[src_ss[0].axis] == 1 and src_ss[1].axis == axis_mirrored)
        {
            return src_ss[0];
        }
        if (src_ss[2].axis == axis_mirrored and (src_ss[0].axis == src_ss[1].axis or
            (src_ss[0].axis == axis_mirrored and (src_ss[1].axis == axis_partial))))
        {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        impl.assert(srcOpt(tensor, 2) == null or src_ss[2].axis == axis_mirrored, "tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");
        return p.generic(src_ss, false);
    }

    /// Ports the `handle_concat` lambda (ggml-backend-meta.cpp:561 @c1d0e7a00).
    fn concat(p: *const Propagate, src_ss: []const SplitState) SplitState {
        const concat_axis: Axis = @bitCast(impl.getOpParamsI32(p.tensor, 0));
        if (src_ss[0].axis == axis_mirrored and isDim(src_ss[1].axis)) {
            impl.assert(concat_axis != src_ss[1].axis, "concat_axis != src_ss[1].axis");
            return src_ss[1];
        }
        if (src_ss[1].axis == axis_mirrored and isDim(src_ss[0].axis)) {
            impl.assert(concat_axis != src_ss[0].axis, "concat_axis != src_ss[0].axis");
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis and src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return p.generic(src_ss, true);
    }

    /// Ports the `handle_mul_mat` lambda (ggml-backend-meta.cpp:577 @c1d0e7a00).
    ///
    /// A weight split along its rows (axis 1) against a mirrored activation
    /// gives an output split along axis 0; split along its columns (axis 0)
    /// against an activation split the same way gives a partial sum, which
    /// is `MIRRORED` once the caller assumes it has been all-reduced.
    fn mulMat(p: *const Propagate, src_ss: []const SplitState) SplitState {
        const tensor = p.tensor;
        if (src_ss[0].axis == axis_mirrored and src_ss[1].axis == axis_mirrored) {
            return ssMake(axis_mirrored);
        }
        if (src_ss[0].axis == axis_1 and src_ss[1].axis == axis_mirrored) {
            var ret = src_ss[0];
            ret.axis = axis_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == axis_1 and src_ss[0].axis == axis_mirrored) {
            return src_ss[1];
        }
        if (src_ss[0].axis == axis_0 and src_ss[1].axis == axis_0) {
            impl.assert(p.equal(&src_ss[0], &src_ss[1]), "split_states_equal(src_ss[0], src_ss[1])");
            return ssMake(if (p.assume_sync) axis_mirrored else axis_partial);
        }
        if (src_ss[0].axis == src_ss[1].axis and src_ss[0].axis >= axis_2 and isDim(src_ss[0].axis)) {
            impl.assert(p.equal(&src_ss[0], &src_ss[1]), "split_states_equal(src_ss[0], src_ss[1])");
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= axis_2 and isDim(src_ss[0].axis) and src_ss[1].axis == axis_mirrored) {
            return src_ss[0];
        }
        abortFmt(605, "unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d", .{
            @as([*c]const u8, &tensor.name),
            @as([*c]const u8, &srcOf(tensor, 0).name),
            axisInt(src_ss[0].axis),
            @as([*c]const u8, &srcOf(tensor, 1).name),
            axisInt(src_ss[1].axis),
        });
    }

    /// Ports the `handle_reshape` lambda (ggml-backend-meta.cpp:610 @c1d0e7a00).
    ///
    /// A dimension split becomes a split along the first output dimension
    /// whose running element count is a multiple of the input's count up to
    /// and including the split axis — the outer dimensions that survive the
    /// reshape unchanged.
    fn reshape(p: *const Propagate, src_ss: []const SplitState) SplitState {
        const tensor = p.tensor;
        switch (src_ss[0].axis) {
            axis_0, axis_1, axis_2, axis_3 => {
                const src0 = srcOf(tensor, 0);
                var base_ne_in: i64 = 1;
                var dim: usize = 0;
                while (dim <= src_ss[0].axis) : (dim += 1) {
                    base_ne_in *= src0.ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in = @divTrunc(base_ne_in, @as(i64, src_ss[0].nr[0]));
                    if (axisInt(src_ss[0].axis) == c.ggml_n_dims(src0) - 1 and src_ss[0].nr[0] == 1) {
                        return ssMake(@intCast(c.ggml_n_dims(tensor) - 1));
                    }
                    if (src_ss[0].axis == axis_0 and tensor.ne[0] == src0.ne[0] and
                        tensor.ne[1] == 1 and src_ss[0].nr[0] == 1)
                    {
                        var complete_rows = true;
                        for (0..p.n_bufs) |j| {
                            const ne = src_ss[0].ne[j];
                            complete_rows = complete_rows and (ne == 0 or ne == src0.ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return ssMake(axis_1);
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                var base_ne_out: i64 = 1;
                for (0..max_dims) |d| {
                    base_ne_out *= tensor.ne[d];
                    if (@rem(base_ne_out, base_ne_in) == 0) {
                        var ret = ssMake(@intCast(d));
                        // uint32_t(base_ne_out/base_ne_in): modulo 2^32.
                        ret.nr[0] = @truncate(@as(u64, @bitCast(@divTrunc(base_ne_out, base_ne_in))));
                        return ret;
                    }
                    if (base_ne_out > base_ne_in) {
                        impl.assert(src_ss[0].n_segments == 1, "src_ss[0].n_segments == 1");
                        impl.assert(src_ss[0].nr[0] == 1, "src_ss[0].nr[0] == 1");
                        return ssMake(@intCast(d));
                    }
                }
                abortFmt(651, "shape mismatch for %s", .{c.ggml_op_name(tensor.op)});
            },
            axis_mirrored, axis_partial => {
                return src_ss[0];
            },
            else => {
                impl.abort("fatal error");
            },
        }
    }

    /// Ports the `handle_cpy` lambda (ggml-backend-meta.cpp:664 @c1d0e7a00).
    fn cpy(p: *const Propagate, src_ss: []const SplitState) SplitState {
        if (isDim(src_ss[0].axis)) {
            return p.reshape(src_ss);
        }
        return p.generic(src_ss, false);
    }

    /// Ports the `handle_view` lambda (ggml-backend-meta.cpp:671 @c1d0e7a00).
    fn view(p: *const Propagate, src_ss: []const SplitState) SplitState {
        const tensor = p.tensor;
        const src0 = srcOf(tensor, 0);
        if (c.ggml_is_contiguous(tensor) and c.ggml_is_contiguous(src0)) {
            return p.reshape(src_ss);
        }
        const axis: c_int = axisInt(src_ss[0].axis);
        {
            var all_strides_the_same = true;
            for (0..max_dims) |dim| {
                if (tensor.ne[dim] == 1 and src0.ne[dim] == 1) {
                    continue;
                }
                if (tensor.nb[dim] != src0.nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!c.ggml_is_permuted(tensor) and !c.ggml_is_permuted(src0) and axis >= 0 and axis < max_dims - 1) {
            for (0..max_dims - 1) |dim| {
                if (tensor.nb[dim + 1] == src0.nb[@as(usize, @intCast(axis)) + 1]) {
                    return ssMake(@intCast(dim));
                }
            }
            impl.abort("fatal error");
        }
        if (src_ss[0].axis == axis_mirrored or src_ss[0].axis == axis_partial) {
            return src_ss[0];
        }
        impl.abort("view of permuted tensor not implemented");
    }

    /// Ports the `handle_permute` lambda (ggml-backend-meta.cpp:706 @c1d0e7a00).
    fn permute(p: *const Propagate, src_ss: []const SplitState) SplitState {
        switch (src_ss[0].axis) {
            axis_0, axis_1, axis_2, axis_3 => {
                impl.assert(src_ss[0].n_segments == 1 or src_ss[0].nr[0] == 1, "src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1");
                var ret = ssMake(@bitCast(p.tensor.op_params[src_ss[0].axis]));
                ret.nr[0] = src_ss[0].nr[0];
                return ret;
            },
            axis_mirrored, axis_partial => {
                return src_ss[0];
            },
            else => {
                impl.abort("fatal error");
            },
        }
    }

    /// Ports the `handle_transpose` lambda (ggml-backend-meta.cpp:726 @c1d0e7a00).
    fn transpose(p: *const Propagate, src_ss: []const SplitState) SplitState {
        _ = p;
        switch (src_ss[0].axis) {
            axis_0, axis_1 => {
                impl.assert(src_ss[0].n_segments == 1 or src_ss[0].nr[0] == 1, "src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1");
                var ret = ssMake(src_ss[0].axis ^ 1);
                ret.nr[0] = src_ss[0].nr[0];
                return ret;
            },
            axis_2, axis_3, axis_mirrored, axis_partial => {
                return src_ss[0];
            },
            else => {
                impl.abort("fatal error");
            },
        }
    }

    /// Ports the `handle_get_rows` lambda (ggml-backend-meta.cpp:746 @c1d0e7a00).
    fn getRows(p: *const Propagate, src_ss: []const SplitState) SplitState {
        if (src_ss[0].axis == axis_0 and src_ss[1].axis == axis_mirrored) {
            return src_ss[0];
        }
        return p.generic(src_ss, true);
    }

    /// Ports the `handle_set_rows` lambda (ggml-backend-meta.cpp:753 @c1d0e7a00).
    fn setRows(p: *const Propagate, src_ss: []const SplitState) SplitState {
        impl.assert(src_ss[0].axis != axis_1, "src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1");
        impl.assert(src_ss[1].axis == axis_mirrored, "src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");
        impl.assert(p.equal(&src_ss[0], &src_ss[2]), "split_states_equal(src_ss[0], src_ss[2])");
        return src_ss[0];
    }

    /// Ports the `handle_rope` lambda (ggml-backend-meta.cpp:760 @c1d0e7a00).
    fn rope(p: *const Propagate, src_ss: []const SplitState) SplitState {
        _ = p;
        impl.assert(src_ss[1].axis == axis_mirrored, "src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");
        return src_ss[0];
    }

    /// Ports the `handle_pad` lambda (ggml-backend-meta.cpp:765 @c1d0e7a00).
    fn pad(p: *const Propagate, src_ss: []const SplitState) SplitState {
        if (isDim(src_ss[0].axis)) {
            impl.assert(p.tensor.op_params[2 * src_ss[0].axis + 0] == 0, "tensor->op_params[2*src_ss[0].axis + 0] == 0");
            impl.assert(p.tensor.op_params[2 * src_ss[0].axis + 1] == 0, "tensor->op_params[2*src_ss[0].axis + 1] == 0");
        }
        return src_ss[0];
    }

    /// Ports the `handle_flash_attn_ext` lambda (ggml-backend-meta.cpp:773 @c1d0e7a00).
    fn flashAttnExt(p: *const Propagate, src_ss: []const SplitState) SplitState {
        const tensor = p.tensor;
        impl.assert(srcOpt(tensor, 3) == null or src_ss[3].axis == axis_mirrored, "tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");

        if (src_ss[0].axis == axis_mirrored) {
            impl.assert(src_ss[1].axis == axis_mirrored, "src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");
            impl.assert(src_ss[2].axis == axis_mirrored, "src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");
            impl.assert(srcOpt(tensor, 4) == null or src_ss[4].axis == axis_mirrored, "tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");
            return ssMake(axis_mirrored);
        }

        impl.assert(src_ss[0].axis == axis_2, "src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2");
        const kv_split = src_ss[1].axis == axis_2 and src_ss[2].axis == axis_2;
        const kv_mirrored = src_ss[1].axis == axis_mirrored and src_ss[2].axis == axis_mirrored;
        impl.assert(kv_split or kv_mirrored, "kv_split || kv_mirrored");
        impl.assert(srcOpt(tensor, 4) == null or src_ss[4].axis == axis_0, "tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0");
        return ssMake(axis_1);
    }

    /// Ports the `handle_lightning_indexer` lambda (ggml-backend-meta.cpp:793 @c1d0e7a00).
    fn lightningIndexer(p: *const Propagate, src_ss: []const SplitState) SplitState {
        _ = p;
        for (0..4) |i| {
            impl.assert(src_ss[i].axis == axis_mirrored, "src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");
        }
        return ssMake(axis_mirrored);
    }

    /// Ports the `handle_ssm_conv` lambda (ggml-backend-meta.cpp:801 @c1d0e7a00).
    fn ssmConv(p: *const Propagate, src_ss: []const SplitState) SplitState {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == axis_0) {
                return ssMake(axis_1);
            }
            if (src_ss[0].axis == axis_1) {
                return ssMake(axis_0);
            }
        }
        return p.generic(src_ss, false);
    }

    /// Ports the `handle_gated_delta_net` lambda (ggml-backend-meta.cpp:813 @c1d0e7a00).
    fn gatedDeltaNet(p: *const Propagate, src_ss: []const SplitState) SplitState {
        _ = p;
        if (src_ss[0].axis == axis_mirrored and src_ss[1].axis == axis_mirrored and
            src_ss[2].axis == axis_mirrored and src_ss[3].axis == axis_mirrored and
            src_ss[4].axis == axis_mirrored and src_ss[5].axis == axis_mirrored)
        {
            return src_ss[0];
        }
        impl.assert(src_ss[0].axis == axis_1, "src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1");
        impl.assert(src_ss[1].axis == axis_1, "src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1");
        impl.assert(src_ss[2].axis == axis_1, "src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1");
        impl.assert(src_ss[3].axis == axis_1, "src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1");
        impl.assert(src_ss[4].axis == axis_1, "src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1");
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        impl.assert(src_ss[5].axis == axis_2 or src_ss[5].axis == axis_1 or src_ss[5].axis == axis_0, "src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0");
        return ssMake(axis_0);
    }

    /// Ports the `calculate_split_state` lambda (ggml-backend-meta.cpp:830 @c1d0e7a00).
    ///
    /// A statically allocated tensor asks the caller's callback and has its
    /// answer checked; anything else derives its state from its sources' by
    /// op, then takes over the first dimension-split source's per-device
    /// ratio, rescaled to its own extent along the split axis.
    fn calculate(p: *const Propagate) SplitState {
        const tensor = p.tensor;
        const n_bufs = p.n_bufs;
        if (c.ggml_nelements(tensor) == 0) {
            return ssMake(axis_unknown);
        }
        if (c.ggml_backend_buffer_get_usage(tensor.buffer) != c.GGML_BACKEND_BUFFER_USAGE_COMPUTE and tensor.view_src == null) {
            const dev = c.ggml_backend_buft_get_device(c.ggml_backend_buffer_get_type(tensor.buffer));
            const dev_ctx = deviceCtx(dev);
            const ret = dev_ctx.get_split_state.?(tensor, dev_ctx.get_split_state_ud);
            if (isDim(ret.axis)) {
                const granularity: i64 = if (ret.axis == axis_0) c.ggml_blck_size(tensor.type) else 1;
                var ne_sum: i64 = 0;
                for (0..ret.n_segments) |s| {
                    for (0..n_bufs) |j| {
                        impl.assert(@rem(ret.ne[s * n_bufs + j], granularity) == 0, "ret.ne[s*n_bufs + j] % granularity == 0");
                        ne_sum += ret.ne[s * n_bufs + j] * @as(i64, ret.nr[s]);
                    }
                }
                impl.assert(ne_sum == tensor.ne[ret.axis], "ne_sum == tensor->ne[ret.axis]");
            } else if (ret.axis == axis_partial) {
                impl.assert(ret.n_segments == 1, "ret.n_segments == 1");
                impl.assert(ret.nr[0] == 1, "ret.nr[0] == 1");
            }
            return ret;
        }

        // A std::vector in the C++, so on the heap: this recurses once per
        // uncached source, and ten 2 KiB states per frame on the stack would
        // not survive a deep graph.
        const src_ss = allocator.alloc(SplitState, max_src) catch oom();
        defer allocator.free(src_ss);
        for (src_ss) |*ss| ss.* = ssMake(axis_none);
        for (0..max_src) |i| {
            if (srcAbsentOrSelf(tensor, i)) {
                src_ss[i] = ssMake(axis_unknown);
                continue;
            }
            src_ss[i] = getSplitState(p.stc, srcOf(tensor, i), true);
            impl.assert(src_ss[i].axis != axis_unknown, "src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN");
        }

        var split_state: SplitState = switch (tensor.op) {
            c.GGML_OP_NONE => ssMake(axis_mirrored),
            c.GGML_OP_DUP => p.generic(src_ss, true),
            c.GGML_OP_ADD, c.GGML_OP_ADD_ID => p.binBcast(src_ss),
            c.GGML_OP_ADD1, c.GGML_OP_ACC => p.generic(src_ss, true),
            c.GGML_OP_SUB, c.GGML_OP_MUL, c.GGML_OP_DIV => p.binBcast(src_ss),
            c.GGML_OP_SQR, c.GGML_OP_SQRT, c.GGML_OP_LOG, c.GGML_OP_SIN, c.GGML_OP_COS => p.generic(src_ss, false),
            c.GGML_OP_SUM => p.generic(src_ss, true),
            c.GGML_OP_SUM_ROWS, c.GGML_OP_CUMSUM, c.GGML_OP_MEAN, c.GGML_OP_ARGMAX, c.GGML_OP_COUNT_EQUAL => p.perRow(src_ss),
            c.GGML_OP_REPEAT, c.GGML_OP_REPEAT_BACK => p.generic(src_ss, false),
            c.GGML_OP_CONCAT => p.concat(src_ss),
            c.GGML_OP_SILU_BACK => p.generic(src_ss, false),
            c.GGML_OP_NORM, c.GGML_OP_RMS_NORM, c.GGML_OP_RMS_NORM_BACK, c.GGML_OP_GROUP_NORM, c.GGML_OP_L2_NORM => p.perRow(src_ss),
            c.GGML_OP_MUL_MAT, c.GGML_OP_MUL_MAT_ID => p.mulMat(src_ss),
            c.GGML_OP_OUT_PROD => p.generic(src_ss, true),
            c.GGML_OP_SCALE => p.generic(src_ss, false),
            c.GGML_OP_SET => p.generic(src_ss, true),
            c.GGML_OP_CPY => p.cpy(src_ss),
            c.GGML_OP_CONT, c.GGML_OP_RESHAPE => p.reshape(src_ss),
            c.GGML_OP_VIEW => p.view(src_ss),
            c.GGML_OP_PERMUTE => p.permute(src_ss),
            c.GGML_OP_TRANSPOSE => p.transpose(src_ss),
            c.GGML_OP_GET_ROWS => p.getRows(src_ss),
            c.GGML_OP_GET_ROWS_BACK => p.generic(src_ss, true),
            c.GGML_OP_SET_ROWS => p.setRows(src_ss),
            c.GGML_OP_DIAG, c.GGML_OP_DIAG_MASK_INF, c.GGML_OP_DIAG_MASK_ZERO => p.generic(src_ss, true),
            c.GGML_OP_SOFT_MAX, c.GGML_OP_SOFT_MAX_BACK => p.generic(src_ss, false),
            c.GGML_OP_ROPE => p.rope(src_ss),
            c.GGML_OP_ROPE_BACK => p.rope(src_ss),
            c.GGML_OP_CLAMP => p.generic(src_ss, false),
            c.GGML_OP_CONV_TRANSPOSE_1D,
            c.GGML_OP_IM2COL,
            c.GGML_OP_IM2COL_BACK,
            c.GGML_OP_IM2COL_3D,
            c.GGML_OP_CONV_2D,
            c.GGML_OP_CONV_3D,
            c.GGML_OP_CONV_2D_DW,
            c.GGML_OP_CONV_TRANSPOSE_2D,
            c.GGML_OP_POOL_1D,
            c.GGML_OP_POOL_2D,
            c.GGML_OP_POOL_2D_BACK,
            c.GGML_OP_UPSCALE,
            => p.generic(src_ss, true),
            c.GGML_OP_PAD => p.pad(src_ss),
            c.GGML_OP_PAD_REFLECT_1D, c.GGML_OP_ROLL, c.GGML_OP_ARANGE, c.GGML_OP_TIMESTEP_EMBEDDING => p.generic(src_ss, true),
            c.GGML_OP_ARGSORT, c.GGML_OP_TOP_K => p.perRow(src_ss),
            c.GGML_OP_LEAKY_RELU => p.generic(src_ss, false),
            c.GGML_OP_TRI => p.generic(src_ss, true),
            c.GGML_OP_FILL => p.generic(src_ss, false),
            c.GGML_OP_FLASH_ATTN_EXT => p.flashAttnExt(src_ss),
            c.GGML_OP_FLASH_ATTN_BACK => p.generic(src_ss, true),
            c.GGML_OP_SSM_CONV => p.ssmConv(src_ss),
            c.GGML_OP_SSM_SCAN,
            c.GGML_OP_WIN_PART,
            c.GGML_OP_WIN_UNPART,
            c.GGML_OP_GET_REL_POS,
            c.GGML_OP_ADD_REL_POS,
            c.GGML_OP_RWKV_WKV6,
            c.GGML_OP_GATED_LINEAR_ATTN,
            c.GGML_OP_RWKV_WKV7,
            c.GGML_OP_SOLVE_TRI,
            => p.generic(src_ss, true),
            c.GGML_OP_GATED_DELTA_NET => p.gatedDeltaNet(src_ss),
            c.GGML_OP_LIGHTNING_INDEXER => p.lightningIndexer(src_ss),
            c.GGML_OP_DSV4_HC_COMB, c.GGML_OP_DSV4_HC_PRE, c.GGML_OP_DSV4_HC_POST => p.generic(src_ss, true),
            c.GGML_OP_UNARY => p.generic(src_ss, false),
            c.GGML_OP_MAP_CUSTOM1, c.GGML_OP_MAP_CUSTOM2, c.GGML_OP_MAP_CUSTOM3, c.GGML_OP_CUSTOM => p.generic(src_ss, true),
            c.GGML_OP_CROSS_ENTROPY_LOSS, c.GGML_OP_CROSS_ENTROPY_LOSS_BACK => p.perRow(src_ss),
            c.GGML_OP_OPT_STEP_ADAMW, c.GGML_OP_OPT_STEP_SGD, c.GGML_OP_GLU => p.generic(src_ss, false),
            else => abortFmt(1062, "ggml op not implemented: %s", .{c.ggml_op_name(tensor.op)}),
        };
        if (isDim(split_state.axis)) {
            var first_src_split_by_axis = true;
            const n_bufs_t = bufferNBufs(tensor.buffer);

            for (0..max_src) |i| {
                if (srcOpt(tensor, i) == null or !isDim(src_ss[i].axis)) {
                    continue;
                }
                const src = srcOf(tensor, i);
                if (first_src_split_by_axis) {
                    for (0..n_bufs_t) |j| {
                        // Take over ratio from src:
                        for (0..src_ss[i].n_segments) |s| {
                            split_state.ne[s * n_bufs_t + j] = 0;
                        }
                        for (0..src_ss[i].n_segments) |s| {
                            split_state.ne[j] += src_ss[i].ne[s * n_bufs_t + j] * @as(i64, src_ss[i].nr[s]);
                        }
                        split_state.ne[j] *= tensor.ne[split_state.axis];
                        if (split_state.ne[j] != 0 or src.ne[src_ss[i].axis] != 0) {
                            const div: i64 = src.ne[src_ss[i].axis] * @as(i64, split_state.nr[0]);
                            impl.assert(@rem(split_state.ne[j], div) == 0, "split_state.ne[j] % div == 0");
                            split_state.ne[j] = @divTrunc(split_state.ne[j], div);
                        }
                    }
                } else {
                    impl.assert(split_state.n_segments == 1, "split_state.n_segments == 1");
                    for (0..n_bufs_t) |j| {
                        // Assert that ratio is consistent:
                        var sum: i64 = 0;
                        for (0..src_ss[i].n_segments) |s| {
                            sum += src_ss[i].ne[s * n_bufs_t + j] * @as(i64, src_ss[i].nr[s]);
                        }
                        impl.assert(split_state.ne[j] * @as(i64, split_state.nr[0]) * src.ne[src_ss[i].axis] ==
                            sum * tensor.ne[split_state.axis], "ratio is consistent");
                    }
                }
                first_src_split_by_axis = false;
            }
            impl.assert(!first_src_split_by_axis, "!first_src_split_by_axis");
        }
        return split_state;
    }
};

/// Appends `ne[j]x nr[0]` for every buffer, comma-separated — the C++'s
/// `std::to_string(ne) + "x" + std::to_string(nr)` loop.
fn appendNeInfo(out: *std.ArrayList(u8), ss: *const SplitState, n_bufs: usize) void {
    for (0..n_bufs) |j| {
        if (out.items.len != 0) out.appendSlice(allocator, ", ") catch oom();
        out.print(allocator, "{d}x{d}", .{ ss.ne[j], ss.nr[0] }) catch oom();
    }
}

/// Ports `ggml_backend_meta_get_split_state` (ggml-backend-meta.cpp:490 @c1d0e7a00),
/// the three-argument overload.
///
/// Memoised per `(tensor, assume_sync)` in the tensor's buffer. An entry is
/// trusted only while the tensor's bytes — everything but its padding — are
/// what they were when it was computed; on any mismatch the *whole* cache is
/// dropped, since a changed tensor can change everything derived from it.
///
/// Parameters:
/// - `stc`: the container being initialised, passed down the recursion.
/// - `tensor`: a tensor in a meta buffer.
/// - `assume_sync`: whether a partial sum counts as already all-reduced.
///
/// Return: the split state; never `NONE`.
fn getSplitState(stc: *SimpleTensorContainer, tensor: *const c.ggml_tensor, assume_sync: bool) SplitState {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const n_bufs = bufferNBufs(tensor.buffer);
    const buf_ctx = bufferCtx(tensor.buffer);
    const p: Propagate = .{ .stc = stc, .tensor = tensor, .n_bufs = n_bufs, .assume_sync = assume_sync };

    const key: CacheKey = .{ .tensor = tensor, .assume_sync = assume_sync };
    var hit = false;
    if (buf_ctx.split_state_cache.getPtr(key)) |entry| {
        hit = true;
        if (!std.mem.eql(u8, &entry.bytes, tensorBytes(tensor))) {
            buf_ctx.split_state_cache.clearRetainingCapacity();
            hit = false;
        }
    }

    if (!hit) {
        // `cache[key].first = calculate_split_state();` -- C++17 sequences the
        // right operand first, so the recursion runs before the entry exists.
        const calculated = p.calculate();
        buf_ctx.cacheIndex(key).ss = calculated;
        @memcpy(&buf_ctx.cacheIndex(key).bytes, tensorBytes(tensor));
        if (buf_ctx.debug > 0) {
            var srcs_info: std.ArrayList(u8) = .empty;
            defer srcs_info.deinit(allocator);
            for (0..max_src) |i| {
                if (srcAbsentOrSelf(tensor, i)) {
                    continue;
                }
                if (srcs_info.items.len != 0) {
                    srcs_info.appendSlice(allocator, ", ") catch oom();
                }
                const src = srcOf(tensor, i);
                const split_state = getSplitStateAuto(src, true);
                impl.assert(split_state.n_segments == 1, "split_state.n_segments == 1");
                const axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                var ne_info: std.ArrayList(u8) = .empty;
                defer ne_info.deinit(allocator);
                appendNeInfo(&ne_info, &split_state, n_bufs);
                srcs_info.print(allocator, "{s}[{s}, {s}, {{{s}}}]", .{
                    std.mem.sliceTo(&src.name, 0),
                    std.mem.span(c.ggml_op_name(src.op)),
                    std.mem.span(axis_name),
                    ne_info.items,
                }) catch oom();
            }
            var ne_info: std.ArrayList(u8) = .empty;
            defer ne_info.deinit(allocator);
            for (0..n_bufs) |j| {
                if (ne_info.items.len != 0) {
                    ne_info.appendSlice(allocator, ", ") catch oom();
                }
                const ss = buf_ctx.cacheIndex(key).ss;
                ne_info.print(allocator, "{d}x{d}", .{ ss.ne[j], ss.nr[0] }) catch oom();
            }
            srcs_info.append(allocator, 0) catch oom();
            ne_info.append(allocator, 0) catch oom();
            impl.logDebug("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", .{
                @as([*c]const u8, srcs_info.items.ptr),
                @as([*c]const u8, &tensor.name),
                c.ggml_op_name(tensor.op),
                ggml_backend_meta_split_axis_name(buf_ctx.cacheIndex(key).ss.axis),
                @as([*c]const u8, ne_info.items.ptr),
            });
        }
    }

    const ret = buf_ctx.cacheIndex(key).ss;
    impl.assert(ret.axis != axis_none, "ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE");
    // The `#ifndef NDEBUG` re-check of the slice sum that follows in the C++
    // is compiled out: the reference builds with NDEBUG.
    return ret;
}

/// Ports `ggml_backend_meta_get_split_state` (ggml-backend-meta.cpp:1170 @c1d0e7a00),
/// the two-argument overload: picks the tensor's own container.
fn getSplitStateAuto(tensor: *const c.ggml_tensor, assume_sync: bool) SplitState {
    impl.assert(ggml_backend_buffer_is_meta(tensor.buffer), "ggml_backend_buffer_is_meta(tensor->buffer)");
    const buf_ctx = bufferCtx(tensor.buffer);
    return getSplitState(buf_ctx.getSimpleTensorContainer(tensor), tensor, assume_sync);
}

/// Ports `ggml_backend_meta_buffer_get_base` (ggml-backend-meta.cpp:1176 @c1d0e7a00).
///
/// A fixed, never-dereferenced address: the allocator only needs offsets from
/// it, which `bufferInitTensorImpl` carries over to each simple buffer.
fn bufferGetBase(buffer: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    _ = buffer;
    return @ptrFromInt(0x1000000000000000); // FIXME
}

/// Ports `ggml_backend_meta_buffer_init_tensor_impl` (ggml-backend-meta.cpp:1181 @c1d0e7a00).
///
/// Builds the per-device simple tensors for `tensor`: its shape with the
/// split axis cut to this device's share, strides rescaled to match, sources
/// and view source swapped for their own slices, and data at the same offset
/// into the device's buffer. A device whose source slice is empty has the
/// node's compute flag cleared.
fn bufferInitTensorImpl(stc: *SimpleTensorContainer, tensor: *c.ggml_tensor) c.enum_ggml_status {
    impl.assert(ggml_backend_buffer_is_meta(tensor.buffer), "ggml_backend_buffer_is_meta(tensor->buffer)");
    const buf_ctx = bufferCtx(tensor.buffer);
    const n_simple_bufs = bufferNBufs(tensor.buffer);

    const split_state = getSplitState(stc, tensor, true);
    impl.assert(c.ggml_nelements(tensor) == 0 or split_state.axis != axis_unknown, "ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN");
    impl.assert(split_state.n_segments <= 16, "split_state.n_segments <= 16");

    const split_dim: c_int = axisInt(split_state.axis);
    const split_is_dim = split_dim >= 0 and split_dim < max_dims;
    const sd: usize = if (split_is_dim) @intCast(split_dim) else 0;
    var ne: [max_dims]i64 = undefined;
    var nb: [max_dims]usize = undefined;
    for (0..max_dims) |k| {
        ne[k] = tensor.ne[k];
        nb[k] = tensor.nb[k];
    }

    var simple_tensors: SimpleTensors = @splat(null);
    for (0..n_simple_bufs) |j| {
        const simple_ctx = stc.ctxs.items[j];
        const simple_buf = buf_ctx.bufs.items[j];

        if ((simple_buf != null) and c.ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            impl.abort("multi buffers are not supported by the meta backend");
        }

        if (split_is_dim) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[sd] = 0;
            for (0..split_state.n_segments) |s| {
                ne[sd] += split_state.ne[s * n_simple_bufs + j] * @as(i64, split_state.nr[s]);
            }
            for (0..max_dims) |i| {
                if (tensor.nb[i] > tensor.nb[sd]) {
                    // size_t * int64_t / int64_t: the usual conversions make it unsigned.
                    nb[i] = tensor.nb[i] *% @as(usize, @bitCast(ne[sd])) / @as(usize, @bitCast(tensor.ne[sd]));
                }
            }
        }

        const t_ij = impl.one(c.ggml_tensor, c.ggml_new_tensor(simple_ctx, tensor.type, max_dims, &ne));
        t_ij.op = tensor.op;
        for (0..max_dims) |i| {
            t_ij.nb[i] = nb[i];
        }
        t_ij.flags = tensor.flags;
        t_ij.op_params = tensor.op_params;
        _ = c.ggml_set_name(t_ij, &tensor.name);
        t_ij.buffer = simple_buf;
        t_ij.view_src = tensor.view_src;
        t_ij.view_offs = tensor.view_offs;
        if (t_ij.view_src != null and ggml_backend_buffer_is_meta(t_ij.view_src.*.buffer)) {
            const view_src = impl.one(c.ggml_tensor, tensor.view_src);
            t_ij.view_src = bufferSimpleTensor(view_src, j);
            if (t_ij.view_offs > 0 and split_is_dim) {
                impl.assert(tensor.ne[sd] != 0, "tensor->ne[split_dim] != 0");
                const split_dim_view_src: c_int = axisInt(getSplitStateAuto(view_src, true).axis);
                impl.assert(split_dim_view_src >= 0 and split_dim_view_src < max_dims, "split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS");

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                var split_internal_offset = t_ij.view_offs <= view_src.nb[@intCast(split_dim_view_src)];
                for (0..max_dims) |i| {
                    const dim_size: usize = @as(usize, @bitCast(tensor.ne[i])) *% tensor.nb[i];
                    if (tensor.view_offs <= dim_size and dim_size < tensor.nb[sd]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij.view_offs = t_ij.view_offs *% @as(usize, @bitCast(ne[sd])) / @as(usize, @bitCast(tensor.ne[sd]));
                }
            }
        }
        if (t_ij.view_src != null) {
            t_ij.data = @ptrFromInt(@intFromPtr(t_ij.view_src.*.data) +% t_ij.view_offs);
        } else if (simple_buf != null) {
            t_ij.data = @ptrFromInt(@intFromPtr(c.ggml_backend_buffer_get_base(simple_buf)) +%
                @intFromPtr(tensor.data) -% @intFromPtr(c.ggml_backend_buffer_get_base(tensor.buffer)));
        }
        t_ij.extra = tensor.extra;
        for (0..max_src) |i| {
            t_ij.src[i] = tensor.src[i];
            if (srcOpt(tensor, i) == tensor) {
                t_ij.src[i] = t_ij;
            } else if (t_ij.src[i] != null and ggml_backend_buffer_is_meta(t_ij.src[i].*.buffer)) {
                t_ij.src[i] = bufferSimpleTensor(srcOf(tensor, i), j);
            }
        }

        simple_tensors[j] = t_ij;
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (0..max_src) |i| {
        const src = srcOpt(tensor, i) orelse continue;
        if (!ggml_backend_buffer_is_meta(src.buffer)) {
            continue;
        }

        const split_state_src = getSplitStateAuto(src, true);
        if (!isDim(split_state_src.axis)) {
            continue;
        }
        for (0..n_simple_bufs) |j| {
            var ne_sum: i64 = 0;
            for (0..split_state_src.n_segments) |s| {
                ne_sum += split_state_src.ne[s * n_simple_bufs + j] * @as(i64, split_state_src.nr[s]);
            }
            if (ne_sum == 0) {
                simple_tensors[j].?.flags &= ~@as(i32, c.GGML_TENSOR_FLAG_COMPUTE);
            }
        }
    }

    stc.simple_tensors.put(allocator, tensor, simple_tensors) catch oom();

    return c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_meta_buffer_init_tensor` (ggml-backend-meta.cpp:1301 @c1d0e7a00).
///
/// Makes the next compute container current first: `graph_compute` flips
/// `stc_compute_index_next` when it rebuilds, and the flip takes effect on
/// the first tensor initialised after it.
fn bufferInitTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor) callconv(.c) c.enum_ggml_status {
    impl.assert(ggml_backend_buffer_is_meta(buffer), "ggml_backend_buffer_is_meta(buffer)");
    const buf_ctx = bufferCtx(buffer);
    const t = impl.one(c.ggml_tensor, tensor);
    buf_ctx.stc_compute_index = buf_ctx.stc_compute_index_next;
    return bufferInitTensorImpl(buf_ctx.getSimpleTensorContainer(t), t);
}

/// The segmented layout's row bounds, shared by the three segmented arms of
/// memset, set and get: `offset` and `size` as whole rows of `row_stride`.
const RowRange = struct {
    start: i64,
    count: i64,

    fn of(offset: usize, size: usize, row_stride: usize, n_rows: i64) RowRange {
        impl.assert(offset % row_stride == 0, "offset % row_stride == 0");
        impl.assert(size % row_stride == 0, "size   % row_stride == 0");
        const r: RowRange = .{ .start = @intCast(offset / row_stride), .count = @intCast(size / row_stride) };
        impl.assert(r.start + r.count <= n_rows, "row_start + row_count <= tensor->ne[...]");
        return r;
    }
};

/// The chunk bounds of the contiguous dimension-split arms of memset, set and
/// get, and of the async pair: `offset` and `size` as whole chunks of one
/// step along the axis above the split.
const ChunkRange = struct {
    size_full: usize,
    start: i64,
    stop: i64,

    fn of(tensor: *const c.ggml_tensor, axis: Axis, offset: usize, size: usize) ChunkRange {
        const chunk_size_full = tensor.nb[axis + 1];
        impl.assert(offset % chunk_size_full == 0, "offset % chunk_size_full == 0");
        impl.assert(size % chunk_size_full == 0, "size   % chunk_size_full == 0");
        return .{
            .size_full = chunk_size_full,
            .start = @intCast(offset / chunk_size_full),
            .stop = @intCast((offset + size) / chunk_size_full),
        };
    }
};

/// The common preamble of the segmented arms: `n_segments != 1 || nr[0] != 1`.
fn checkSegmented(tensor: *const c.ggml_tensor, split_state: *const SplitState) void {
    impl.assert(isDim(split_state.axis), "split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS");
    impl.assert(split_state.nr[0] != 0, "split_state.nr[0] != 0");
    impl.assert(tensor.ne[3] == 1, "tensor->ne[3] == 1");
}

/// Ports `ggml_backend_meta_buffer_memset_tensor` (ggml-backend-meta.cpp:1308 @c1d0e7a00).
fn bufferMemsetTensor(buffer: c.ggml_backend_buffer_t, tensor_: [*c]c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    const tensor = impl.one(c.ggml_tensor, tensor_);
    const n_bufs = bufferNBufs(buffer);
    const split_state = getSplitStateAuto(tensor, false);
    impl.assert(c.ggml_is_contiguous(tensor) or split_state.axis == axis_mirrored, "ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");

    if (split_state.n_segments != 1 or split_state.nr[0] != 1) {
        checkSegmented(tensor, &split_state);

        var simple_offsets: [max_devices]usize = @splat(0);
        if (split_state.axis == axis_0) {
            impl.assert(tensor.ne[2] == 1, "tensor->ne[2] == 1");

            const rows = RowRange.of(offset, size, tensor.nb[1], tensor.ne[1]);

            const blck_size: i64 = c.ggml_blck_size(tensor.type);
            for (0..split_state.n_segments) |s| {
                for (0..split_state.nr[s]) |_| {
                    for (0..n_bufs) |j| {
                        const simple_tensor = bufferSimpleTensor(tensor, j).?;
                        impl.assert(@rem(split_state.ne[s * n_bufs + j], blck_size) == 0, "split_state.ne[s*n_bufs + j] % blck_size == 0");
                        const nbytes: usize = @as(usize, @intCast(@divTrunc(split_state.ne[s * n_bufs + j], blck_size))) * tensor.nb[0];
                        var row: i64 = 0;
                        while (row < rows.count) : (row += 1) {
                            c.ggml_backend_tensor_memset(simple_tensor, value, simple_offsets[j] + @as(usize, @intCast(rows.start + row)) * simple_tensor.nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        impl.assert(split_state.axis == axis_1, "split_state.axis == GGML_BACKEND_SPLIT_AXIS_1");

        const rows = RowRange.of(offset, size, tensor.nb[2], tensor.ne[2]);

        for (0..split_state.n_segments) |s| {
            for (0..split_state.nr[s]) |_| {
                for (0..n_bufs) |j| {
                    const simple_tensor = bufferSimpleTensor(tensor, j).?;
                    const nbytes: usize = @as(usize, @intCast(split_state.ne[s * n_bufs + j])) * tensor.nb[1];
                    var row: i64 = 0;
                    while (row < rows.count) : (row += 1) {
                        c.ggml_backend_tensor_memset(simple_tensor, value, simple_offsets[j] + @as(usize, @intCast(rows.start + row)) * simple_tensor.nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        axis_0, axis_1, axis_2 => {
            const chunks = ChunkRange.of(tensor, split_state.axis, offset, size);
            for (0..n_bufs) |j| {
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                const chunk_size = simple_tensor.nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                var i = chunks.start;
                while (i < chunks.stop) : (i += 1) {
                    c.ggml_backend_tensor_memset(simple_tensor, value, @as(usize, @intCast(i)) * chunk_size, chunk_size);
                }
            }
        },
        axis_partial, axis_mirrored => {
            if (split_state.axis == axis_partial) {
                impl.assert(value == 0, "value == 0");
            }
            // [[fallthrough]] from PARTIAL
            for (0..n_bufs) |j| {
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                c.ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        },
        else => {
            impl.abort("fatal error");
        },
    }
}

/// Ports `ggml_backend_meta_buffer_set_tensor` (ggml-backend-meta.cpp:1410 @c1d0e7a00).
///
/// Scatters host data over the simple tensors. A segmented split walks the
/// segments and their repeats device by device; a plain dimension split
/// takes one 2-D copy per device; a mirrored tensor is copied whole to every
/// device; a partial one is divided evenly between the devices that
/// contribute to it, and zeroed on the rest.
fn bufferSetTensor(buffer: c.ggml_backend_buffer_t, tensor_: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    const tensor = impl.one(c.ggml_tensor, tensor_);
    const bytes: [*]const u8 = @ptrCast(data);
    const n_bufs = bufferNBufs(buffer);
    const split_state = getSplitStateAuto(tensor, false);
    impl.assert(c.ggml_is_contiguous(tensor) or split_state.axis == axis_mirrored, "ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");

    if (split_state.n_segments != 1 or split_state.nr[0] != 1) {
        checkSegmented(tensor, &split_state);

        var offset_data: usize = 0;
        var simple_offsets: [max_devices]usize = @splat(0);
        if (split_state.axis == axis_0) {
            impl.assert(tensor.ne[2] == 1, "tensor->ne[2] == 1");

            const rows = RowRange.of(offset, size, tensor.nb[1], tensor.ne[1]);

            const blck_size: i64 = c.ggml_blck_size(tensor.type);
            for (0..split_state.n_segments) |s| {
                for (0..split_state.nr[s]) |_| {
                    for (0..n_bufs) |j| {
                        const simple_tensor = bufferSimpleTensor(tensor, j).?;
                        impl.assert(@rem(split_state.ne[s * n_bufs + j], blck_size) == 0, "split_state.ne[s*n_bufs + j] % blck_size == 0");
                        const nbytes: usize = @as(usize, @intCast(@divTrunc(split_state.ne[s * n_bufs + j], blck_size))) * tensor.nb[0];
                        c.ggml_backend_tensor_set_2d(simple_tensor, bytes + offset_data, simple_offsets[j] + @as(usize, @intCast(rows.start)) * simple_tensor.nb[1], nbytes, @intCast(rows.count), simple_tensor.nb[1], tensor.nb[1]);
                        offset_data += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            impl.assert(offset_data * @as(usize, @intCast(rows.count)) == size, "offset_data*row_count == size");
            return;
        }
        impl.assert(split_state.axis == axis_1, "split_state.axis == GGML_BACKEND_SPLIT_AXIS_1");

        const rows = RowRange.of(offset, size, tensor.nb[2], tensor.ne[2]);

        for (0..split_state.n_segments) |s| {
            for (0..split_state.nr[s]) |_| {
                for (0..n_bufs) |j| {
                    const simple_tensor = bufferSimpleTensor(tensor, j).?;
                    const nbytes: usize = @as(usize, @intCast(split_state.ne[s * n_bufs + j])) * tensor.nb[1];
                    c.ggml_backend_tensor_set_2d(simple_tensor, bytes + offset_data, simple_offsets[j] + @as(usize, @intCast(rows.start)) * simple_tensor.nb[2], nbytes, @intCast(rows.count), simple_tensor.nb[2], tensor.nb[2]);
                    offset_data += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        impl.assert(offset_data * @as(usize, @intCast(rows.count)) == size, "offset_data*row_count == size");
        return;
    }

    switch (split_state.axis) {
        axis_0, axis_1, axis_2 => {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const chunks = ChunkRange.of(tensor, split_state.axis, offset, size);
            var offset_j: usize = 0;
            for (0..n_bufs) |j| {
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                const chunk_size_j = simple_tensor.nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const simple_offset = @as(usize, @intCast(chunks.start)) * chunk_size_j;
                c.ggml_backend_tensor_set_2d(simple_tensor, bytes + offset_j, simple_offset, chunk_size_j, @intCast(chunks.stop - chunks.start), chunk_size_j, chunks.size_full);
                offset_j += chunk_size_j;
            }
            impl.assert(offset_j == chunks.size_full, "offset_j == chunk_size_full");
        },
        axis_mirrored => {
            for (0..n_bufs) |j| {
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                c.ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        },
        axis_partial => {
            impl.assert(tensor.type == c.GGML_TYPE_F32, "tensor->type == GGML_TYPE_F32");
            impl.assert(offset % @sizeOf(f32) == 0, "offset % sizeof(float) == 0");
            impl.assert(size % @sizeOf(f32) == 0, "size   % sizeof(float) == 0");
            const n_values = size / @sizeOf(f32);
            var n_contributors: usize = 0;
            for (0..n_bufs) |j| {
                n_contributors += @intFromBool(split_state.ne[j] != 0);
            }
            const has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            const values: [*]align(1) const f32 = @ptrCast(data);
            const tmp = allocator.alloc(f32, n_values) catch oom();
            defer allocator.free(tmp);
            for (0..n_values) |i| {
                tmp[i] = values[i] / @as(f32, @floatFromInt(n_contributors));
            }
            var zero: []f32 = &.{};
            defer if (zero.len != 0) allocator.free(zero);
            if (has_contributor_mask) {
                zero = allocator.alloc(f32, n_values) catch oom();
                @memset(zero, 0.0);
            }
            for (0..n_bufs) |j| {
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                const partial: [*]const f32 = if (has_contributor_mask and split_state.ne[j] == 0) zero.ptr else tmp.ptr;
                c.ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        },
        else => {
            impl.abort("fatal error");
        },
    }
}

/// Ports `ggml_backend_meta_buffer_get_tensor` (ggml-backend-meta.cpp:1538 @c1d0e7a00).
///
/// The gather that mirrors `bufferSetTensor`. A mirrored tensor is read from
/// device 0; a partial one cannot be read at all and aborts.
fn bufferGetTensor(buffer: c.ggml_backend_buffer_t, tensor_: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    const tensor: *const c.ggml_tensor = @ptrCast(tensor_);
    const bytes: [*]u8 = @ptrCast(data);
    const n_bufs = bufferNBufs(buffer);
    const split_state = getSplitStateAuto(tensor, false);
    impl.assert(c.ggml_is_contiguous(tensor) or split_state.axis == axis_mirrored, "ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED");

    if (split_state.n_segments != 1 or split_state.nr[0] != 1) {
        checkSegmented(tensor, &split_state);

        var offset_data: usize = 0;
        var simple_offsets: [max_devices]usize = @splat(0);
        if (split_state.axis == axis_0) {
            impl.assert(tensor.ne[2] == 1, "tensor->ne[2] == 1");

            const rows = RowRange.of(offset, size, tensor.nb[1], tensor.ne[1]);

            const blck_size: i64 = c.ggml_blck_size(tensor.type);
            for (0..split_state.n_segments) |s| {
                for (0..split_state.nr[s]) |_| {
                    for (0..n_bufs) |j| {
                        const simple_tensor = bufferSimpleTensor(tensor, j).?;
                        impl.assert(@rem(split_state.ne[s * n_bufs + j], blck_size) == 0, "split_state.ne[s*n_bufs + j] % blck_size == 0");
                        const nbytes: usize = @as(usize, @intCast(@divTrunc(split_state.ne[s * n_bufs + j], blck_size))) * tensor.nb[0];
                        c.ggml_backend_tensor_get_2d(simple_tensor, bytes + offset_data, simple_offsets[j] + @as(usize, @intCast(rows.start)) * simple_tensor.nb[1], nbytes, @intCast(rows.count), simple_tensor.nb[1], tensor.nb[1]);
                        offset_data += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            impl.assert(offset_data * @as(usize, @intCast(rows.count)) == size, "offset_data*row_count == size");
            return;
        }
        impl.assert(split_state.axis == axis_1, "split_state.axis == GGML_BACKEND_SPLIT_AXIS_1");

        const rows = RowRange.of(offset, size, tensor.nb[2], tensor.ne[2]);

        for (0..split_state.n_segments) |s| {
            for (0..split_state.nr[s]) |_| {
                for (0..n_bufs) |j| {
                    const simple_tensor = bufferSimpleTensor(tensor, j).?;
                    const nbytes: usize = @as(usize, @intCast(split_state.ne[s * n_bufs + j])) * tensor.nb[1];
                    c.ggml_backend_tensor_get_2d(simple_tensor, bytes + offset_data, simple_offsets[j] + @as(usize, @intCast(rows.start)) * simple_tensor.nb[2], nbytes, @intCast(rows.count), simple_tensor.nb[2], tensor.nb[2]);
                    offset_data += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        impl.assert(offset_data * @as(usize, @intCast(rows.count)) == size, "offset_data*row_count == size");
        return;
    }

    switch (split_state.axis) {
        axis_0, axis_1, axis_2 => {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const chunks = ChunkRange.of(tensor, split_state.axis, offset, size);
            var offset_j: usize = 0;
            for (0..n_bufs) |j| {
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                const chunk_size_j = simple_tensor.nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const simple_offset = @as(usize, @intCast(chunks.start)) * chunk_size_j;
                c.ggml_backend_tensor_get_2d(simple_tensor, bytes + offset_j, simple_offset, chunk_size_j, @intCast(chunks.stop - chunks.start), chunk_size_j, chunks.size_full);
                offset_j += chunk_size_j;
            }
            impl.assert(offset_j == chunks.size_full, "offset_j == chunk_size_full");
        },
        axis_mirrored => {
            // TODO other simple backend may be better
            const simple_tensor = bufferSimpleTensor(tensor, 0).?;
            c.ggml_backend_tensor_get(simple_tensor, data, offset, size);
        },
        else => {
            impl.abort("fatal error");
        },
    }
}

/// Ports `ggml_backend_meta_buffer_clear` (ggml-backend-meta.cpp:1638 @c1d0e7a00).
fn bufferClear(buffer: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    const n_buffers = bufferNBufs(buffer);
    for (0..n_buffers) |i| {
        c.ggml_backend_buffer_clear(bufferSimpleBuffer(buffer, i), value);
    }
}

/// Ports `ggml_backend_meta_buffer_reset` (ggml-backend-meta.cpp:1645 @c1d0e7a00).
fn bufferReset(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    impl.assert(ggml_backend_buffer_is_meta(buffer), "ggml_backend_buffer_is_meta(buffer)");
    const buf_ctx = bufferCtx(buffer);
    for (0..buf_ctx.bufs.items.len) |i| {
        c.ggml_backend_buffer_reset(bufferSimpleBuffer(buffer, i));
    }
}

/// Ports `ggml_backend_meta_buffer_iface` (ggml-backend-meta.cpp:1653 @c1d0e7a00).
const buffer_iface: c.struct_ggml_backend_buffer_i = .{
    .free_buffer = &bufferFreeBuffer,
    .get_base = &bufferGetBase,
    .init_tensor = &bufferInitTensor,
    .memset_tensor = &bufferMemsetTensor,
    .set_tensor = &bufferSetTensor,
    .get_tensor = &bufferGetTensor,
    .set_tensor_2d = null,
    .get_tensor_2d = null,
    .cpy_tensor = null,
    .clear = &bufferClear,
    .reset = &bufferReset,
};

/// Ports `ggml_backend_buffer_is_meta` (ggml-backend-meta.cpp:1667 @c1d0e7a00).
pub export fn ggml_backend_buffer_is_meta(buf: c.ggml_backend_buffer_t) callconv(.c) bool {
    return buf != null and buf.*.iface.free_buffer == buffer_iface.free_buffer;
}

/// Ports `ggml_backend_meta_buffer_type_alloc_buffer` (ggml-backend-meta.cpp:1671 @c1d0e7a00).
///
/// One simple buffer of `size` bytes per device — the full size on each, not
/// a share — plus two compute containers sized for a million tensors. A
/// buffer made this way has no static container, so every tensor in it is a
/// compute tensor.
fn buftAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const n_simple_bufts = buftNBufts(buft);

    const params: c.struct_ggml_init_params = .{
        .mem_size = 1024 * 1024 * c.ggml_tensor_overhead(), // FIXME
        .mem_buffer = null,
        .no_alloc = true,
    };
    const stc_static: SimpleTensorContainer = .{};
    const stc_compute_0 = SimpleTensorContainer.initCtxs(params, n_simple_bufts);
    const stc_compute_1 = SimpleTensorContainer.initCtxs(params, n_simple_bufts);

    var max_size: usize = 0;
    var bufs: [max_devices]c.ggml_backend_buffer_t = @splat(null);
    for (0..n_simple_bufts) |i| {
        bufs[i] = c.ggml_backend_buft_alloc_buffer(buftSimpleBuft(buft, i), size);
        impl.assert(bufs[i] != null, "bufs.back() != nullptr");
        max_size = @max(max_size, c.ggml_backend_buffer_get_size(bufs[i]));
    }
    const buf_ctx = BufferContext.create(stc_static, stc_compute_0, stc_compute_1, bufs[0..n_simple_bufts]);

    return c.ggml_backend_buffer_init(buft, buffer_iface, buf_ctx, max_size);
}

/// Ports `ggml_backend_meta_alloc_ctx_tensors_from_buft` (ggml-backend-meta.cpp:1696 @c1d0e7a00).
///
/// The meta half of `ggml_backend_alloc_ctx_tensors_from_buft`, which hands a
/// meta buffer type straight here. Every tensor in `ctx` becomes a static
/// meta tensor whose slices go into one simple context per device; each of
/// those is then allocated with the device's own buffer type. A device whose
/// slices are all empty gets a dummy zero-sized buffer instead, since the
/// allocator returns null for that case.
///
/// Return: the meta buffer, owned by the caller; its size is the largest
/// simple buffer's.
pub export fn ggml_backend_meta_alloc_ctx_tensors_from_buft(ctx: ?*c.struct_ggml_context, buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]c.struct_ggml_backend_buffer {
    const n_simple_bufts = buftNBufts(buft);

    const compute_headroom: usize = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const params_static: c.struct_ggml_init_params = .{
        .mem_size = c.ggml_get_mem_size(ctx),
        .mem_buffer = null,
        .no_alloc = true,
    };
    const params_compute: c.struct_ggml_init_params = .{
        .mem_size = compute_headroom * c.ggml_get_mem_size(ctx),
        .mem_buffer = null,
        .no_alloc = true,
    };
    const stc_static = SimpleTensorContainer.initCtxs(params_static, n_simple_bufts);
    const stc_compute_0 = SimpleTensorContainer.initCtxs(params_compute, n_simple_bufts);
    const stc_compute_1 = SimpleTensorContainer.initCtxs(params_compute, n_simple_bufts);

    const bufs: [max_devices]c.ggml_backend_buffer_t = @splat(null);
    const meta_buf_ctx = BufferContext.create(stc_static, stc_compute_0, stc_compute_1, bufs[0..n_simple_bufts]);

    const meta_buf = c.ggml_backend_buffer_init(buft, buffer_iface, meta_buf_ctx, 0);
    var t = c.ggml_get_first_tensor(ctx);
    while (t != null) : (t = c.ggml_get_next_tensor(ctx, t)) {
        const tensor = impl.one(c.ggml_tensor, t);
        tensor.buffer = meta_buf;
        _ = bufferInitTensorImpl(&meta_buf_ctx.stc_static, tensor);
        tensor.data = @ptrFromInt(0x2000000000000000); // FIXME
    }
    for (0..n_simple_bufts) |i| {
        const simple_ctx = meta_buf_ctx.stc_static.ctxs.items[i];
        const simple_buft = buftSimpleBuft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        var any_nonzero_slice = false;
        var st = c.ggml_get_first_tensor(simple_ctx);
        while (st != null) : (st = c.ggml_get_next_tensor(simple_ctx, st)) {
            if (c.ggml_nelements(st) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        // `bufs[i].reset(new)`: the new buffer exists before the old (always
        // null here) would be released.
        if (any_nonzero_slice) {
            replaceBuffer(&meta_buf_ctx.bufs.items[i], c.ggml_backend_alloc_ctx_tensors_from_buft(simple_ctx, simple_buft));
        } else {
            replaceBuffer(&meta_buf_ctx.bufs.items[i], c.ggml_backend_buft_alloc_buffer(simple_buft, 0));
            st = c.ggml_get_first_tensor(simple_ctx);
            while (st != null) : (st = c.ggml_get_next_tensor(simple_ctx, st)) {
                st.*.buffer = meta_buf_ctx.bufs.items[i];
            }
        }
        impl.assert(meta_buf_ctx.bufs.items[i] != null, "meta_buf_ctx->bufs[i]");
        meta_buf.*.size = @max(meta_buf.*.size, c.ggml_backend_buffer_get_size(meta_buf_ctx.bufs.items[i]));
    }
    return meta_buf;
}

/// `ggml_backend_buffer_ptr::reset`: install the new buffer, then free the old.
fn replaceBuffer(slot: *c.ggml_backend_buffer_t, new: c.ggml_backend_buffer_t) void {
    const old = slot.*;
    slot.* = new;
    if (old != null) c.ggml_backend_buffer_free(old);
}

// -----------------------------------------------------------------------------
// Meta backend

/// Ports the `guid` in `ggml_backend_meta_guid` (ggml-backend-meta.cpp:1754 @c1d0e7a00).
var meta_guid: c.ggml_guid = .{ 0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda };

/// Ports `ggml_backend_meta_guid` (ggml-backend-meta.cpp:1754 @c1d0e7a00).
fn backendGuid() c.ggml_guid_t {
    return &meta_guid;
}

/// In `ggml_backend_meta_context`, ports `cgraph_config` (ggml-backend-meta.cpp:1760 @c1d0e7a00).
///
/// The C++'s per-config `cgraphs_aux` vector is never used; it is dropped.
const CgraphConfig = struct {
    cgraph_main: ?*impl.CGraph = null,
    offset: c_int = 0, // Node offset vs. original graph
};

/// In `ggml_backend_meta_context`, ports `backend_config` (ggml-backend-meta.cpp:1766 @c1d0e7a00).
const BackendConfig = struct {
    backend: c.ggml_backend_t,

    cgraphs: std.ArrayList(CgraphConfig) = .empty,
    nodes: std.ArrayList(?*c.ggml_tensor) = .empty,
    /// Owning, one per reduction step; null until first allocated.
    bufs: std.ArrayList(c.ggml_backend_buffer_t) = .empty,

    fn init(backend: c.ggml_backend_t, n_reduce_steps: usize) BackendConfig {
        var self: BackendConfig = .{ .backend = backend };
        self.bufs.appendNTimes(allocator, null, n_reduce_steps) catch oom();
        return self;
    }

    /// The members' destructors, in reverse declaration order.
    fn deinit(self: *BackendConfig) void {
        var i = self.bufs.items.len;
        while (i > 0) {
            i -= 1;
            if (self.bufs.items[i] != null) c.ggml_backend_buffer_free(self.bufs.items[i]);
        }
        self.bufs.deinit(allocator);
        self.nodes.deinit(allocator);
        self.cgraphs.deinit(allocator);
    }
};

/// Ports `struct ggml_backend_meta_context` (ggml-backend-meta.cpp:1759 @c1d0e7a00).
const BackendContext = struct {
    name: [:0]u8,
    backend_configs: std.ArrayList(BackendConfig) = .empty,
    /// Owning (`ggml_context_ptr`): the graphs and auxiliary nodes below.
    ctx: ?*c.ggml_context = null,
    cgraphs_aux: std.ArrayList(?*impl.CGraph) = .empty,
    nodes_aux: std.ArrayList(?*c.ggml_tensor) = .empty,
    n_reduce_steps: usize,
    max_nnodes: c_int = 0,
    max_tmp_size: usize = 0,
    max_subgraphs: usize = 0,
    n_subgraphs: usize = 0,
    uid: u64 = 0,

    comm_ctx: ?*anyopaque = null,
    comm_allreduce: c.ggml_backend_comm_allreduce_tensor_t = null,

    /// Ports the `ggml_backend_meta_context` constructor (ggml-backend-meta.cpp:1792 @c1d0e7a00).
    ///
    /// Initialises one simple backend per device, and asks the first one's
    /// registry for a communicator. **Ungated on this machine (one GPU)**: no
    /// backend on macOS exports `ggml_backend_comm_init`, so `comm_ctx` stays
    /// null here and the butterfly fallback carries every all-reduce.
    fn create(meta_dev: c.ggml_backend_dev_t, params: [*c]const u8) *BackendContext {
        const n_devs = devNDevs(meta_dev);
        const self = allocator.create(BackendContext) catch oom();
        self.* = .{
            .name = undefined,
            // std::ceil(std::log2(n_devs)), through double as in the C++.
            .n_reduce_steps = @intFromFloat(@ceil(@log2(@as(f64, @floatFromInt(n_devs))))),
        };
        var names: [max_devices][*c]const u8 = undefined;
        var simple_backends: [max_devices]c.ggml_backend_t = undefined;
        self.backend_configs.ensureTotalCapacity(allocator, n_devs) catch oom();
        for (0..n_devs) |i| {
            const simple_dev = devSimpleDev(meta_dev, i);
            names[i] = c.ggml_backend_dev_name(simple_dev);
            simple_backends[i] = c.ggml_backend_dev_init(simple_dev, params);
            self.backend_configs.appendAssumeCapacity(BackendConfig.init(simple_backends[i], self.n_reduce_steps));
        }
        self.name = metaName(names[0..n_devs]);

        if (n_devs > 1) {
            const comm_init: c.ggml_backend_comm_init_t = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(
                c.ggml_backend_dev_backend_reg(c.ggml_backend_get_device(simple_backends[0])),
                "ggml_backend_comm_init",
            )));
            if (comm_init) |init_fn| {
                self.comm_ctx = init_fn(&simple_backends, n_devs);
            }
        }
        if (self.comm_ctx != null) {
            self.comm_allreduce = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(
                c.ggml_backend_dev_backend_reg(c.ggml_backend_get_device(simple_backends[0])),
                "ggml_backend_comm_allreduce_tensor",
            )));
            impl.assert(self.comm_allreduce != null, "comm_allreduce != nullptr");
        }
        return self;
    }

    /// Ports the `ggml_backend_meta_context` destructor (ggml-backend-meta.cpp:1825 @c1d0e7a00).
    ///
    /// The body frees the communicator and the simple backends; only then do
    /// the members go, so each config's scratch buffers outlive its backend.
    /// Freeing the communicator is **ungated on this machine (one GPU)**, as
    /// creating it is.
    fn destroy(self: *BackendContext) void {
        if (self.comm_ctx != null) {
            const comm_free: c.ggml_backend_comm_free_t = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(
                c.ggml_backend_dev_backend_reg(c.ggml_backend_get_device(self.backend_configs.items[0].backend)),
                "ggml_backend_comm_free",
            )));
            impl.assert(comm_free != null, "comm_free != nullptr");
            comm_free.?(self.comm_ctx);
        }
        for (self.backend_configs.items) |bc| {
            c.ggml_backend_free(bc.backend);
        }

        self.nodes_aux.deinit(allocator);
        self.cgraphs_aux.deinit(allocator);
        if (self.ctx) |ctx| c.ggml_free(ctx);
        var i = self.backend_configs.items.len;
        while (i > 0) {
            i -= 1;
            self.backend_configs.items[i].deinit();
        }
        self.backend_configs.deinit(allocator);
        allocator.free(self.name);
        allocator.destroy(self);
    }
};

inline fn backendCtx(backend: c.ggml_backend_t) *BackendContext {
    return @ptrCast(@alignCast(backend.*.context));
}

/// Ports `ggml_backend_meta_get_name` (ggml-backend-meta.cpp:1838 @c1d0e7a00).
fn backendGetName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    impl.assert(ggml_backend_is_meta(backend), "ggml_backend_is_meta(backend)");
    return backendCtx(backend).name.ptr;
}

/// Ports `ggml_backend_meta_free` (ggml-backend-meta.cpp:1844 @c1d0e7a00).
fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    impl.assert(ggml_backend_is_meta(backend), "ggml_backend_is_meta(backend)");
    backendCtx(backend).destroy();
    allocator.destroy(impl.one(c.struct_ggml_backend, backend));
}

/// Ports `ggml_backend_meta_set_tensor_async` (ggml-backend-meta.cpp:1851 @c1d0e7a00).
///
/// Whole contiguous tensors only (`offset == 0`), one segment. Each device's
/// copy is written at `offset` — always 0 — rather than at the chunk start
/// the synchronous path computes, which is the same thing when `offset` is 0.
fn backendSetTensorAsync(backend: c.ggml_backend_t, tensor_: [*c]c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    const tensor = impl.one(c.ggml_tensor, tensor_);
    const bytes: [*]const u8 = @ptrCast(data);
    const n_backends = ggml_backend_meta_n_backends(backend);
    impl.assert(offset == 0, "offset == 0");
    impl.assert(c.ggml_is_contiguous(tensor), "ggml_is_contiguous(tensor)");

    const split_state = getSplitStateAuto(tensor, false);
    impl.assert(split_state.n_segments == 1, "split_state.n_segments == 1");
    impl.assert(split_state.nr[0] == 1, "split_state.nr[0]      == 1");

    switch (split_state.axis) {
        axis_0, axis_1, axis_2 => {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const chunks = ChunkRange.of(tensor, split_state.axis, offset, size);
            var offset_j: usize = 0;
            for (0..n_backends) |j| {
                const simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                const chunk_size_j = simple_tensor.nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                c.ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, bytes + offset_j, offset, chunk_size_j, @intCast(chunks.stop - chunks.start), chunk_size_j, chunks.size_full);
                offset_j += chunk_size_j;
            }
            impl.assert(offset_j == chunks.size_full, "offset_j == chunk_size_full");
        },
        axis_mirrored => {
            for (0..n_backends) |j| {
                c.ggml_backend_tensor_set_async(ggml_backend_meta_simple_backend(backend, j), bufferSimpleTensor(tensor, j), data, offset, size);
            }
        },
        else => {
            impl.abort("fatal error");
        },
    }
}

/// Ports `ggml_backend_meta_get_tensor_async` (ggml-backend-meta.cpp:1896 @c1d0e7a00).
fn backendGetTensorAsync(backend: c.ggml_backend_t, tensor_: [*c]const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    const tensor: *const c.ggml_tensor = @ptrCast(tensor_);
    const bytes: [*]u8 = @ptrCast(data);
    const n_backends = ggml_backend_meta_n_backends(backend);
    impl.assert(offset == 0, "offset == 0");
    impl.assert(c.ggml_is_contiguous(tensor), "ggml_is_contiguous(tensor)");

    const split_state = getSplitStateAuto(tensor, false);
    impl.assert(split_state.n_segments == 1, "split_state.n_segments == 1");
    impl.assert(split_state.nr[0] == 1, "split_state.nr[0]      == 1");

    switch (split_state.axis) {
        axis_0, axis_1, axis_2 => {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const chunks = ChunkRange.of(tensor, split_state.axis, offset, size);
            var offset_j: usize = 0;
            for (0..n_backends) |j| {
                const simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const simple_tensor = bufferSimpleTensor(tensor, j).?;
                const chunk_size_j = simple_tensor.nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                c.ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, bytes + offset_j, offset, chunk_size_j, @intCast(chunks.stop - chunks.start), chunk_size_j, chunks.size_full);
                offset_j += chunk_size_j;
            }
            impl.assert(offset_j == chunks.size_full, "offset_j == chunk_size_full");
        },
        axis_mirrored => {
            // TODO other simple backend may be better
            const simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const simple_tensor = bufferSimpleTensor(tensor, 0);
            c.ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        },
        else => {
            impl.abort("fatal error");
        },
    }
}

/// Ports `ggml_backend_meta_synchronize` (ggml-backend-meta.cpp:1941 @c1d0e7a00).
fn backendSynchronize(backend: c.ggml_backend_t) callconv(.c) void {
    const n_backends = ggml_backend_meta_n_backends(backend);
    for (0..n_backends) |i| {
        c.ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

/// Ports `ggml_node_get_use_count` (ggml-impl.h:629 @c1d0e7a00).
fn nodeGetUseCount(cgraph: *const impl.CGraph, node_idx: c_int) i32 {
    const node = cgraph.nodes[@intCast(node_idx)].?;
    const hash_pos = impl.hashFind(&cgraph.visited_hash_set, node);
    if (!impl.bitsetGet(cgraph.visited_hash_set.used, hash_pos)) {
        return 0;
    }
    return cgraph.use_counts[hash_pos];
}

/// The lambdas that decide how far an all-reduce can be put off,
/// `get_i_delayed_branch` (ggml-backend-meta.cpp:2010 @c1d0e7a00) and
/// `get_i_delayed` (ggml-backend-meta.cpp:2111 @c1d0e7a00), with what they
/// capture.
const Delay = struct {
    cgraph: *impl.CGraph,
    backend_ctx: *BackendContext,
    n_backends: usize,

    inline fn node(d: *const Delay, i: c_int) *c.ggml_tensor {
        return d.cgraph.nodes[@intCast(i)].?;
    }

    /// Ports the `get_i_delayed_branch` lambda (ggml-backend-meta.cpp:2010 @c1d0e7a00).
    ///
    /// For MoE models it may make sense to delay the AllReduce in order to
    /// reduce I/O: follows an `ADD_ID` bias, then a chain of `MUL`s by
    /// mirrored scales, then the expert views and the `ADD` tree that sums
    /// them, skipping mirrored nodes that do not consume the partial sum.
    ///
    /// Return: the index of the last node the reduction can wait for.
    fn branch(d: *const Delay, i: c_int) c_int {
        const cgraph = d.cgraph;
        var id: c_int = i; // i_delayed
        var idr: c_int = i; // i_delayed return, last safe return value

        var cur = d.node(id);
        var n_used: i32 = nodeGetUseCount(cgraph, id);

        // Skip MIRRORED nodes that don't consume node
        const skipUnrelated = struct {
            fn f(dd: *const Delay, idp: *c_int, nd: *c.ggml_tensor) void {
                while (idp.* + 1 < dd.cgraph.n_nodes) {
                    const next = dd.node(idp.* + 1);
                    if (getSplitStateAuto(next, false).axis != axis_mirrored) {
                        break;
                    }
                    var safe = true;
                    for (0..max_src) |s| {
                        const src = srcOpt(next, s) orelse continue;
                        if (src == nd) {
                            safe = false;
                            break;
                        }
                        if (getSplitStateAuto(src, false).axis != axis_mirrored) {
                            safe = false;
                            break;
                        }
                    }
                    if (!safe) {
                        break;
                    }
                    idp.* += 1;
                }
            }
        }.f;

        skipUnrelated(d, &id, cur);
        if (id + 1 >= cgraph.n_nodes) {
            return idr;
        }
        {
            const next = d.node(id + 1);
            if (next.op == c.GGML_OP_ADD_ID and srcOpt(next, 0) == cur and
                getSplitStateAuto(srcOf(next, 1), false).axis == axis_partial and
                getSplitStateAuto(srcOf(next, 2), false).axis == axis_mirrored)
            {
                cur = next;
                id += 1;
                idr = id;
                n_used = nodeGetUseCount(cgraph, id);
            }
        }
        // Chain of MULs with MIRRORED src[1]
        while (true) {
            skipUnrelated(d, &id, cur);
            if (id + 1 >= cgraph.n_nodes) {
                return idr;
            }
            const next = d.node(id + 1);
            if (next.op == c.GGML_OP_MUL and srcOpt(next, 0) == cur and
                getSplitStateAuto(srcOf(next, 1), false).axis == axis_mirrored)
            {
                cur = next;
                id += 1;
                idr = id;
                n_used = nodeGetUseCount(cgraph, id);
            } else {
                break;
            }
        }

        if (n_used != cur.ne[1] or id + 2 * n_used - 1 >= cgraph.n_nodes) {
            return idr;
        }
        var k: i32 = 0;
        while (k < n_used) : (k += 1) {
            const next = d.node(id + 1);
            if (next.op != c.GGML_OP_VIEW or next.view_src != @as([*c]c.ggml_tensor, cur) or
                next.view_offs != @as(usize, @intCast(k)) *% cur.nb[1] or
                next.ne[0] != cur.ne[0] or next.ne[1] != cur.ne[2] or next.nb[1] != cur.nb[2] or
                nodeGetUseCount(cgraph, id + 1) != 1)
            {
                return idr;
            }
            id += 1;
        }
        {
            const next = d.node(id + 1);
            if (next.op != c.GGML_OP_ADD or srcOpt(next, 0) != d.node(id - (n_used - 1)) or
                srcOpt(next, 1) != d.node(id - (n_used - 2)) or nodeGetUseCount(cgraph, id + 1) != 1)
            {
                return idr;
            }
            id += 1;
        }
        k = 0;
        while (k < n_used - 2) : (k += 1) {
            const next = d.node(id + 1);
            if (next.op != c.GGML_OP_ADD or srcOpt(next, 0) != d.node(id) or
                srcOpt(next, 1) != d.node(id - (n_used - 2)) or nodeGetUseCount(cgraph, id + 1) != 1)
            {
                return idr;
            }
            id += 1;
        }
        idr = id;
        return idr;
    }

    /// Ports the `get_i_delayed` lambda (ggml-backend-meta.cpp:2111 @c1d0e7a00).
    ///
    /// AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial
    /// branches: when the next partial node's branch ends in an `ADD` of the
    /// two, one reduction after the sum serves both.
    fn delayed(d: *const Delay, i: c_int) c_int {
        const cgraph = d.cgraph;
        const i_delayed = d.branch(i);
        const cur = d.node(i_delayed);

        if (nodeGetUseCount(cgraph, i_delayed) != 1) {
            return i_delayed;
        }

        var id: c_int = i_delayed + 1;
        while (id < cgraph.n_nodes) : (id += 1) {
            const next = d.node(id);
            if (next.view_src == @as([*c]c.ggml_tensor, cur)) {
                return i_delayed;
            }
            for (0..max_src) |s| {
                if (srcOpt(next, s) == cur) {
                    return i_delayed;
                }
            }

            if (next.view_src != null and next.view_src.*.op == c.GGML_OP_NONE and c.ggml_backend_buffer_is_host(next.view_src.*.buffer)) {
                continue;
            }
            if (getSplitStateAuto(next, false).axis != axis_partial) {
                continue;
            }

            const i_other = id;
            const i_other_delayed = d.branch(i_other);
            const other = d.node(i_other_delayed);
            if (nodeGetUseCount(cgraph, i_other_delayed) != 1 or i_other_delayed + 1 >= cgraph.n_nodes) {
                return i_delayed;
            }

            const sum = d.node(i_other_delayed + 1);
            if (sum.op != c.GGML_OP_ADD or
                !c.ggml_are_same_shape(cur, other) or cur.type != other.type or sum.type != cur.type or
                !((srcOpt(sum, 0) == cur and srcOpt(sum, 1) == other) or
                    (srcOpt(sum, 0) == other and srcOpt(sum, 1) == cur)) or
                getSplitStateAuto(sum, false).axis != axis_mirrored)
            {
                return i_delayed;
            }

            for (d.backend_ctx.backend_configs.items[0..d.n_backends]) |*bcj| {
                const compute = (bcj.nodes.items[@intCast(i)].?.flags & c.GGML_TENSOR_FLAG_COMPUTE) != 0;
                const compute_other = (bcj.nodes.items[@intCast(i_other)].?.flags & c.GGML_TENSOR_FLAG_COMPUTE) != 0;
                if (compute != compute_other) {
                    return i_delayed;
                }
            }
            return i_other_delayed + 1;
        }
        return i_delayed;
    }
};

/// True for the C++'s `s_copy_main` FIXME test: a view of a CPU-resident
/// leaf, which the graph carries but which has no meta slices.
///
/// Its true arm is **ungated on this machine (one GPU)**: it needs a meta
/// device whose buffers are host memory, and with only CPU devices the
/// reference crashes in `graph_compute` (the FIXME skips views it should
/// slice), so no gate runs that configuration.
inline fn isHostLeafView(node: *const c.ggml_tensor) bool {
    return node.view_src != null and node.view_src.*.op == c.GGML_OP_NONE and c.ggml_backend_buffer_is_host(node.view_src.*.buffer);
}

/// Grows `list` to `n` items, filling with `value`, as `std::vector::resize`
/// does; never shrinks below the current length in the one place it is used.
fn resizeFill(comptime T: type, list: *std.ArrayList(T), n: usize, value: T) void {
    if (n > list.items.len) {
        list.appendNTimes(allocator, value, n - list.items.len) catch oom();
    } else {
        list.shrinkRetainingCapacity(n);
    }
}

/// The per-call state of the helper lambdas `get_node_aux`
/// (ggml-backend-meta.cpp:2275 @c1d0e7a00), `set_tmp_data` and
/// `get_cgraph_aux` in `ggml_backend_meta_graph_compute`: the auxiliary node
/// and graph cursors.
const Aux = struct {
    backend_ctx: *BackendContext,
    iga: usize = 0, // i graph aux
    ina: usize = 0, // i node aux

    /// Ports the `get_node_aux` lambda (ggml-backend-meta.cpp:2275 @c1d0e7a00):
    /// the next preallocated scratch tensor, cleared and shaped like `t`.
    fn getNodeAux(a: *Aux, t: *const c.ggml_tensor) *c.ggml_tensor {
        const ret = a.backend_ctx.nodes_aux.items[a.ina].?;
        a.ina += 1;
        ret.* = std.mem.zeroes(c.ggml_tensor);
        ret.op = c.GGML_OP_NONE;
        ret.type = t.type;
        for (0..max_dims) |k| {
            ret.ne[k] = t.ne[k];
            ret.nb[k] = t.nb[k];
        }
        return ret;
    }

    /// Ports the `set_tmp_data` lambda (ggml-backend-meta.cpp:2286 @c1d0e7a00):
    /// points `tensor` at device `j`'s scratch buffer for step `i_buf`,
    /// growing it to `max_tmp_size` first if it is missing or too small.
    fn setTmpData(a: *Aux, tensor: *c.ggml_tensor, j: usize, i_buf: usize) void {
        const bcj = &a.backend_ctx.backend_configs.items[j];
        const buf_ptr = &bcj.bufs.items[i_buf];
        if (buf_ptr.* == null or c.ggml_backend_buffer_get_size(buf_ptr.*) < a.backend_ctx.max_tmp_size) {
            replaceBuffer(buf_ptr, c.ggml_backend_alloc_buffer(bcj.backend, a.backend_ctx.max_tmp_size));
        }
        tensor.buffer = buf_ptr.*;
        tensor.data = c.ggml_backend_buffer_get_base(buf_ptr.*);
    }

    // FIXME usage_counts
    /// Ports the `get_cgraph_aux` lambda (ggml-backend-meta.cpp:2296 @c1d0e7a00).
    fn getCgraphAux(a: *Aux) *impl.CGraph {
        const ret = a.backend_ctx.cgraphs_aux.items[a.iga].?;
        a.iga += 1;
        return ret;
    }
};

/// The last node of device `j`'s subgraph `i` — the partial sum to reduce.
inline fn lastNode(bcj: *const BackendConfig, i: usize) *c.ggml_tensor {
    const cg = bcj.cgraphs.items[i].cgraph_main.?;
    return cg.nodes[@intCast(cg.n_nodes - 1)].?;
}

inline fn computeAsync(backend: c.ggml_backend_t, cgraph: *impl.CGraph) c.enum_ggml_status {
    return c.ggml_backend_graph_compute_async(backend, @ptrCast(cgraph));
}

/// Ports the `allreduce_fallback` lambda (ggml-backend-meta.cpp:2302 @c1d0e7a00).
///
/// The generic all-reduce of subgraph `i`'s last node, for backends with no
/// collective of their own: zero the slices of devices whose compute flag
/// was cleared, fold any devices beyond the largest power of two into the
/// first ones, run a butterfly of copy-and-`ADD` steps through per-step
/// scratch buffers, and copy the result back out to the folded devices.
///
/// The early returns on a failed `graph_compute_async` are ungated: no
/// backend here fails a compute.
///
/// Return: the first failing status, or success.
fn allreduceFallback(aux: *Aux, n_backends: usize, i: usize) c.enum_ggml_status {
    const backend_ctx = aux.backend_ctx;
    var step_cgraphs: [max_devices]?*impl.CGraph = @splat(null);

    // Zero out nodes that were disabled due to having a zero-sized slice:
    for (0..n_backends) |j| {
        const bcj = &backend_ctx.backend_configs.items[j];
        const node = lastNode(bcj, i);
        if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) != 0) {
            continue;
        }
        const node_zero = aux.getNodeAux(node);
        node_zero.op = c.GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
        node_zero.src[0] = node;
        impl.setOpParamsF32(node_zero, 0, 0.0);
        node_zero.data = node.data;
        node_zero.buffer = node.buffer;
        node_zero.flags |= c.GGML_TENSOR_FLAG_COMPUTE;

        const cg = aux.getCgraphAux();
        step_cgraphs[j] = cg;
        cg.nodes[0] = node_zero;
        cg.n_nodes = 1;
        const status = computeAsync(bcj.backend, cg);
        if (status != c.GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    @memset(&step_cgraphs, null);

    const pushData = struct {
        /// Ports the `push_data` lambda (ggml-backend-meta.cpp:2330 @c1d0e7a00):
        /// copy `j_src`'s partial into `j_dst`'s scratch, and queue an `ADD`
        /// of it into `j_dst`'s partial in place.
        fn f(ax: *Aux, steps: *[max_devices]?*impl.CGraph, ii: usize, j_src: usize, j_dst: usize, i_buf: usize) void {
            // assert(step_cgraphs[j_dst] == nullptr) is a plain assert, compiled out under NDEBUG.
            const bcj_src = &ax.backend_ctx.backend_configs.items[j_src];
            const bcj_dst = &ax.backend_ctx.backend_configs.items[j_dst];

            const node_src = lastNode(bcj_src, ii);
            const node_dst = lastNode(bcj_dst, ii);
            impl.assert(c.ggml_is_contiguous(node_src), "ggml_is_contiguous(node_src)");
            impl.assert(c.ggml_is_contiguous(node_dst), "ggml_is_contiguous(node_dst)");

            const node_tmp = ax.getNodeAux(node_dst);
            ax.setTmpData(node_tmp, j_dst, i_buf);

            c.ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            const node_red = ax.getNodeAux(node_dst);
            node_red.view_src = if (node_dst.view_src == null) node_dst else node_dst.view_src;
            node_red.view_offs = node_dst.view_offs;
            node_red.op = c.GGML_OP_ADD;
            node_red.src[0] = node_dst;
            node_red.src[1] = node_tmp;
            node_red.flags |= c.GGML_TENSOR_FLAG_COMPUTE;
            _ = c.ggml_backend_view_init(node_red);

            const cgraph_aux = ax.getCgraphAux();
            cgraph_aux.nodes[0] = node_red;
            cgraph_aux.n_nodes = 1;
            steps[j_dst] = cgraph_aux;
        }
    }.f;

    var offset_j: usize = n_backends / 2;
    while ((offset_j & (offset_j -% 1)) != 0) {
        offset_j -= 1;
    }
    const offset_j_max = offset_j;
    var i_buf: usize = 0;

    // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
    var j_src: usize = 2 * offset_j_max;
    while (j_src < n_backends) : (j_src += 1) {
        const j_dst = j_src - 2 * offset_j_max;
        pushData(aux, &step_cgraphs, i, j_src, j_dst, i_buf);
        const status = computeAsync(backend_ctx.backend_configs.items[j_dst].backend, step_cgraphs[j_dst].?);
        if (status != c.GGML_STATUS_SUCCESS) {
            return status;
        }
        i_buf = 1;
    }

    // Butterfly reduction:
    while (offset_j >= 1) : (offset_j /= 2) {
        @memset(&step_cgraphs, null);

        for (0..2 * offset_j_max) |j| {
            const j_other = j ^ offset_j;
            if (j_other >= n_backends) {
                continue;
            }
            pushData(aux, &step_cgraphs, i, j, j_other, i_buf);
        }

        for (0..2 * offset_j_max) |j| {
            const cg = step_cgraphs[j] orelse continue;
            const bcj = &backend_ctx.backend_configs.items[j];
            const status = computeAsync(bcj.backend, cg);
            if (status != c.GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        i_buf += 1;
    }
    // assert(i_buf == backend_ctx->n_reduce_steps) is compiled out under NDEBUG.

    // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
    var j: usize = 2 * offset_j_max;
    while (j < n_backends) : (j += 1) {
        const bcj_src = &backend_ctx.backend_configs.items[j - 2 * offset_j_max];
        const bcj_dst = &backend_ctx.backend_configs.items[j];

        const node_src = lastNode(bcj_src, i);
        const node_dst = lastNode(bcj_dst, i);
        c.ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
    }

    return c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_meta_graph_compute` (ggml-backend-meta.cpp:1948 @c1d0e7a00).
///
/// Unless the graph's `uid` matches the last one, rebuilds: flips every used
/// buffer to its other compute container, maps each node to its per-device
/// slice, cuts the graph after every `PARTIAL` node (pushed later where the
/// reduction can be delayed), grows the scratch buffers and graph storage,
/// and fills one subgraph per cut per device. Then runs each subgraph on
/// every device and all-reduces between them.
fn backendGraphCompute(backend: c.ggml_backend_t, cgraph_: ?*c.struct_ggml_cgraph) callconv(.c) c.enum_ggml_status {
    const cgraph: *impl.CGraph = @ptrCast(@alignCast(cgraph_.?));
    impl.assert(cgraph.grads == null, "cgraph->grads == nullptr");
    const n_backends = ggml_backend_meta_n_backends(backend);
    const backend_ctx = backendCtx(backend);
    const configs = backend_ctx.backend_configs.items[0..n_backends];

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const needs_rebuild = (cgraph.uid == 0) or (cgraph.uid != backend_ctx.uid);

    var max_nnodes_raised = false;
    if (cgraph.n_nodes > backend_ctx.max_nnodes) {
        for (configs) |*bcj| {
            resizeFill(?*c.ggml_tensor, &bcj.nodes, @intCast(cgraph.n_nodes), null);
            resizeFill(CgraphConfig, &bcj.cgraphs, @intCast(cgraph.n_nodes), .{});
        }
        backend_ctx.max_nnodes = cgraph.n_nodes;
        max_nnodes_raised = true;
        // assert(needs_rebuild) is a plain assert, compiled out under NDEBUG.
    }

    if (needs_rebuild) {
        // std::set<ggml_backend_buffer_t>: deduplicated and visited in address
        // order. The work per buffer is independent, but the order is kept.
        var used_buffers: std.ArrayList(usize) = .empty;
        defer used_buffers.deinit(allocator);
        for (0..@intCast(cgraph.n_leafs)) |i| {
            const leaf = cgraph.leafs[i].?;
            if (ggml_backend_buffer_is_meta(leaf.buffer)) {
                used_buffers.append(allocator, @intFromPtr(leaf.buffer)) catch oom();
            }
        }
        for (0..@intCast(cgraph.n_nodes)) |i| {
            const node = cgraph.nodes[i].?;
            if (ggml_backend_buffer_is_meta(node.buffer)) {
                used_buffers.append(allocator, @intFromPtr(node.buffer)) catch oom();
            }
        }
        std.mem.sort(usize, used_buffers.items, {}, std.sort.asc(usize));
        var prev: usize = 0;
        for (used_buffers.items) |addr| {
            if (addr == prev) continue;
            prev = addr;
            const buf: c.ggml_backend_buffer_t = @ptrFromInt(addr);
            const buf_ctx = bufferCtx(buf);
            buf_ctx.stc_compute_index_next = buf_ctx.stc_compute_index ^ 1;
            const stc = &buf_ctx.stc_compute[@intCast(buf_ctx.stc_compute_index_next)];
            for (stc.ctxs.items) |ctx| {
                c.ggml_reset(ctx);
            }
            stc.simple_tensors.clearRetainingCapacity();
        }
        var n_subgraphs: usize = 0;
        var max_tmp_size: usize = 0;

        for (configs, 0..) |*bcj, j| {
            for (0..@intCast(cgraph.n_nodes)) |i| {
                const node = cgraph.nodes[i].?;
                if (isHostLeafView(node)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes.items[i] = node;
                    continue;
                }
                bcj.nodes.items[i] = bufferSimpleTensor(node, j);
                impl.assert(bcj.nodes.items[i] != null, "bcj.nodes[i]");
            }
        }

        {
            const delay: Delay = .{ .cgraph = cgraph, .backend_ctx = backend_ctx, .n_backends = n_backends };

            var i_start: c_int = 0;
            var i: c_int = 0;
            while (i < cgraph.n_nodes) : (i += 1) {
                const node = delay.node(i);
                if (isHostLeafView(node)) {
                    continue;
                }
                const split_state = getSplitStateAuto(node, false);
                if (split_state.axis == axis_partial) {
                    max_tmp_size = @max(max_tmp_size, c.ggml_nbytes(node));
                }
                const new_subgraph = i + 1 == cgraph.n_nodes or split_state.axis == axis_partial;
                if (!new_subgraph) {
                    continue;
                }

                const i_delayed = delay.delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (configs) |*bcj| {
                        if ((bcj.nodes.items[@intCast(i)].?.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            var ii: c_int = i + 1;
                            while (ii <= i_delayed) : (ii += 1) {
                                bcj.nodes.items[@intCast(ii)].?.flags &= ~@as(i32, c.GGML_TENSOR_FLAG_COMPUTE);
                            }
                        }
                    }
                }

                i = i_delayed;

                for (configs) |*bcj| {
                    bcj.cgraphs.items[n_subgraphs].offset = i_start;
                }
                n_subgraphs += 1;
                i_start = i + 1;
            }
            impl.assert(i_start == cgraph.n_nodes, "i_start == cgraph->n_nodes");
        }

        backend_ctx.uid = cgraph.uid;
        backend_ctx.n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx.max_tmp_size) {
            for (configs) |*bcj| {
                for (0..backend_ctx.n_reduce_steps) |i| {
                    replaceBuffer(&bcj.bufs.items[i], c.ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx.max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised or n_subgraphs > backend_ctx.max_subgraphs) {
            backend_ctx.max_subgraphs = @max(backend_ctx.max_subgraphs, n_subgraphs);
            const n_nodes_per_device = 3 * backend_ctx.n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const n_cgraphs_per_device = 2 * backend_ctx.n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const grads = cgraph.grads != null;
            const mem_per_device_graphs_main = backend_ctx.max_subgraphs * c.ggml_graph_overhead_custom(@intCast(backend_ctx.max_nnodes), grads);
            const mem_per_device_graphs_aux = n_cgraphs_per_device * backend_ctx.max_subgraphs * c.ggml_graph_overhead_custom(1, grads);
            const mem_per_device_nodes_aux = n_nodes_per_device * backend_ctx.max_subgraphs * c.ggml_tensor_overhead();
            const params: c.struct_ggml_init_params = .{
                .mem_size = n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                .mem_buffer = null,
                .no_alloc = true,
            };
            // ctx.reset(ggml_init(params)): the new context exists before the old is freed.
            const new_ctx = c.ggml_init(params);
            const old_ctx = backend_ctx.ctx;
            backend_ctx.ctx = new_ctx;
            if (old_ctx) |old| c.ggml_free(old);
            for (configs) |*bcj| {
                for (0..n_subgraphs) |i| {
                    bcj.cgraphs.items[i].cgraph_main = @ptrCast(@alignCast(c.ggml_new_graph_custom(backend_ctx.ctx, @intCast(cgraph.n_nodes), false)));
                }
            }
            resizeFill(?*impl.CGraph, &backend_ctx.cgraphs_aux, n_backends * n_cgraphs_per_device * backend_ctx.max_subgraphs, null);
            for (backend_ctx.cgraphs_aux.items) |*cg| {
                cg.* = @ptrCast(@alignCast(c.ggml_new_graph_custom(backend_ctx.ctx, 1, grads)));
            }
            resizeFill(?*c.ggml_tensor, &backend_ctx.nodes_aux, n_backends * n_nodes_per_device * backend_ctx.max_subgraphs, null);
            for (backend_ctx.nodes_aux.items) |*t| {
                t.* = c.ggml_new_tensor_1d(backend_ctx.ctx, c.GGML_TYPE_F32, 1);
            }
        }

        for (configs) |*bcj| {
            for (0..n_subgraphs) |i_graph| {
                const cgraph_ij = bcj.cgraphs.items[i_graph].cgraph_main.?;
                const i_node_start: usize = @intCast(bcj.cgraphs.items[i_graph].offset);
                const i_node_stop: usize = if (i_graph + 1 < n_subgraphs) @intCast(bcj.cgraphs.items[i_graph + 1].offset) else @intCast(cgraph.n_nodes);
                cgraph_ij.n_nodes = @intCast(i_node_stop - i_node_start);
                ggml_hash_set_reset(&cgraph_ij.visited_hash_set);
                for (i_node_start..i_node_stop) |i_node| {
                    const node_ij = bcj.nodes.items[i_node].?;
                    cgraph_ij.nodes[i_node - i_node_start] = node_ij;
                    const hash_pos_orig = impl.hashFind(&cgraph.visited_hash_set, cgraph.nodes[i_node].?);
                    const hash_pos_ij = impl.hashInsert(&cgraph_ij.visited_hash_set, node_ij);
                    cgraph_ij.use_counts[hash_pos_ij] = cgraph.use_counts[hash_pos_orig];
                }
                cgraph_ij.uid = ggml_graph_next_uid();
            }
        }
    }

    var aux: Aux = .{ .backend_ctx = backend_ctx };

    for (0..backend_ctx.n_subgraphs) |i| {
        for (configs) |*bcj| {
            const status = computeAsync(bcj.backend, bcj.cgraphs.items[i].cgraph_main.?);
            if (status != c.GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (n_backends > 1 and i < backend_ctx.n_subgraphs - 1) {
            var backend_allreduce_success = false;
            if (backend_ctx.comm_ctx != null) {
                // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA).
                // Ungated on this machine (one GPU): no macOS backend provides one.
                var nodes: [max_devices][*c]c.ggml_tensor = undefined;
                for (configs, 0..) |*bcj, j| {
                    nodes[j] = lastNode(bcj, i);
                }
                backend_allreduce_success = backend_ctx.comm_allreduce.?(backend_ctx.comm_ctx, &nodes);
            }

            if (!backend_allreduce_success) {
                // use a generic fallback if unavailable:
                const status = allreduceFallback(&aux, n_backends, i);
                if (status != c.GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_meta_i` (ggml-backend-meta.cpp:2451 @c1d0e7a00).
const backend_iface: c.struct_ggml_backend_i = .{
    .get_name = &backendGetName,
    .free = &backendFree,
    .set_tensor_async = &backendSetTensorAsync,
    .get_tensor_async = &backendGetTensorAsync,
    .set_tensor_2d_async = null,
    .get_tensor_2d_async = null,
    .cpy_tensor_async = null,
    .synchronize = &backendSynchronize,
    .graph_plan_create = null,
    .graph_plan_free = null,
    .graph_plan_update = null,
    .graph_plan_compute = null,
    .graph_compute = &backendGraphCompute,
    .event_record = null,
    .event_wait = null,
    .graph_optimize = null,
};

/// Ports `ggml_backend_is_meta` (ggml-backend-meta.cpp:2470 @c1d0e7a00).
pub export fn ggml_backend_is_meta(backend: c.ggml_backend_t) callconv(.c) bool {
    return backend != null and backend.*.iface.get_name == backend_iface.get_name;
}

/// Ports `ggml_backend_meta_device_init_backend` (ggml-backend-meta.cpp:2474 @c1d0e7a00).
///
/// Return: a meta backend owning one simple backend per device, released
/// with `ggml_backend_free`.
fn deviceInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    const backend_ctx = BackendContext.create(dev, params);

    const backend = allocator.create(c.struct_ggml_backend) catch oom();
    backend.guid = backendGuid();
    backend.iface = backend_iface;
    backend.device = dev;
    backend.context = backend_ctx;
    return backend;
}

/// Ports `ggml_backend_meta_n_backends` (ggml-backend-meta.cpp:2485 @c1d0e7a00).
pub export fn ggml_backend_meta_n_backends(meta_backend: c.ggml_backend_t) callconv(.c) usize {
    impl.assert(ggml_backend_is_meta(meta_backend), "ggml_backend_is_meta(meta_backend)");
    return backendCtx(meta_backend).backend_configs.items.len;
}

/// Ports `ggml_backend_meta_simple_backend` (ggml-backend-meta.cpp:2491 @c1d0e7a00).
///
/// `index` is unchecked, as `std::vector::operator[]` is.
pub export fn ggml_backend_meta_simple_backend(meta_backend: c.ggml_backend_t, index: usize) callconv(.c) c.ggml_backend_t {
    impl.assert(ggml_backend_is_meta(meta_backend), "ggml_backend_is_meta(meta_backend)");
    return backendCtx(meta_backend).backend_configs.items.ptr[index].backend;
}

// -----------------------------------------------------------------------------
// Imported from the sibling ported files
//
// `ggml-impl.h` declares these but cannot be imported, so they are redeclared
// here as `backend_sched.zig` does. Importing `graph.zig` instead would drag
// its exports into every standalone build of this file.

extern fn ggml_hash_set_reset(hash_set: *impl.HashSet) void;
extern fn ggml_graph_next_uid() u64;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(DeviceContext);
    std.testing.refAllDecls(BufferTypeContext);
    std.testing.refAllDecls(SimpleTensorContainer);
    std.testing.refAllDecls(BufferContext);
    std.testing.refAllDecls(Propagate);
    std.testing.refAllDecls(BackendConfig);
    std.testing.refAllDecls(BackendContext);
    std.testing.refAllDecls(Delay);
    std.testing.refAllDecls(Aux);
}

test "split axis names match the C++ switch" {
    try std.testing.expectEqualStrings("0", std.mem.span(ggml_backend_meta_split_axis_name(axis_0)));
    try std.testing.expectEqualStrings("3", std.mem.span(ggml_backend_meta_split_axis_name(axis_3)));
    try std.testing.expectEqualStrings("MIRRORED", std.mem.span(ggml_backend_meta_split_axis_name(axis_mirrored)));
    try std.testing.expectEqualStrings("PARTIAL", std.mem.span(ggml_backend_meta_split_axis_name(axis_partial)));
    try std.testing.expectEqualStrings("NONE", std.mem.span(ggml_backend_meta_split_axis_name(axis_none)));
    try std.testing.expectEqualStrings("UNKNOWN", std.mem.span(ggml_backend_meta_split_axis_name(axis_unknown)));
}

test "ssMake is the C's {axis, {0}, {1}, 1}" {
    const ss = ssMake(axis_2);
    try std.testing.expectEqual(axis_2, ss.axis);
    try std.testing.expectEqual(@as(u32, 1), ss.nr[0]);
    try std.testing.expectEqual(@as(u32, 0), ss.nr[1]);
    try std.testing.expectEqual(@as(u32, 1), ss.n_segments);
    for (ss.ne) |v| try std.testing.expectEqual(@as(i64, 0), v);
}

test "the device and buffer-type names are Meta(), as the shadowed loop makes them" {
    // The reference logs `assigned to device Meta()`; see the module doc.
    var ctx = DeviceContext.init(try allocator.alloc(c.ggml_backend_dev_t, 0), null, null);
    defer ctx.deinit();
    try std.testing.expectEqualStrings("Meta()", ctx.name);
    try std.testing.expectEqualStrings("Meta()", ctx.description);
    const parts = [_][*c]const u8{ "MTL0", "CPU" };
    const name = metaName(&parts);
    defer allocator.free(name);
    try std.testing.expectEqualStrings("Meta(MTL0,CPU)", name);
}

test "null is never meta" {
    try std.testing.expect(!ggml_backend_is_meta(null));
    try std.testing.expect(!ggml_backend_buffer_is_meta(null));
    try std.testing.expect(!ggml_backend_buft_is_meta(null));
    try std.testing.expect(!devIsMeta(null));
}

test "the cache compares every tensor byte but the padding" {
    try std.testing.expectEqual(@sizeOf(c.ggml_tensor) - 8, nbtc);
    try std.testing.expectEqual(@offsetOf(c.ggml_tensor, "padding"), nbtc);
}
