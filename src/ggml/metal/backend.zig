//! The Metal backend: three buffer types, the backend, the device and the
//! registry.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # What it does and does not do
//!
//! Nothing here talks to Metal. Every Metal call goes through
//! `device_c.zig` to `ggml-metal-device.m` and `ggml-metal-context.m`,
//! which stay Objective-C permanently (Decision 13). This file is the
//! `ggml_backend` vtable plumbing around them, and its shape is the one
//! `cpu/cpu_backend.zig` already established: iface structs of function
//! pointers, magic statics for the per-device singletons, and a registry.
//!
//! # Three buffer types, two ifaces, one body each
//!
//! The C writes out three buffer types — shared, private, mapped — and two
//! buffer ifaces, shared and private. Measured with `diff`: the two ifaces
//! are identical but for the polarity of one assertion
//! (`ggml_metal_buffer_is_shared` versus its negation), and the three
//! buffer types differ only in a name suffix and whether
//! `alloc_buffer` asks for shared memory. So each becomes one
//! comptime-parameterised body, which is the same move `repack/dispatch.zig`
//! made for `tensor_traits`.
//!
//! **`supports_buft` compares `get_name` function pointers**, so the three
//! buffer types need three distinguishable ones. Collapsing them would be
//! harmless even if the compiler folded the bodies — all three comparisons
//! would then match the one address, giving the same answer as the C's
//! three-way `or` — but the test at the bottom checks the behaviour rather
//! than the addresses, so the question does not arise.
//!
//! # `GGML_BACKEND_DL_IMPL` expands to nothing
//!
//! The C's last line is `GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)`,
//! which `ggml-backend-impl.h:264` defines as empty unless
//! `GGML_BACKEND_DL` is set. This build is static and does not set it,
//! which is why `ggml_backend_init` is not among the exports.

const std = @import("std");
const impl = @import("../impl.zig");
const mc = @import("device_c.zig");
const metal_common = @import("common.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;

/// The C `malloc`s the backend and `new`s the devices, and nothing hands
/// it an allocator. libc's is what a C entry point can reach — the same
/// reasoning as `backend_reg.zig`, `backend.zig` and `cpu_backend.zig`.
const allocator = std.heap.c_allocator;

/// Zig 0.16's `std.c` does not declare `setenv`, only `getenv`. The C's
/// macOS workaround below needs it, so it is declared here.
extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Ports `GGML_METAL_NAME` (ggml-metal.cpp:14 @c1d0e7a00).
const metal_name = "MTL";

/// Ports `GGML_METAL_MAX_DEVICES` (ggml-metal.cpp:15 @c1d0e7a00).
///
/// The C's `std::vector`s are sized by `g_devices` at run time; a fixed
/// array of the maximum is the move `cpu_backend.zig` already made, and it
/// is what lets the per-device singletons be plain statics.
const max_devices = 16;

/// Ports `g_devices` (ggml-metal.cpp:19 @c1d0e7a00): the device count,
/// overridable by `GGML_METAL_DEVICES` to simulate virtual devices.
var g_devices: c_int = 1;

// -----------------------------------------------------------------------------
// Buffers

/// Ports `ggml_backend_buffer_is_metal` (ggml-metal.cpp:180 @c1d0e7a00).
fn bufferIsMetal(buffer: c.ggml_backend_buffer_t) bool {
    if (buffer == null) return false;
    const name = c.ggml_backend_buffer_name(buffer);
    if (name == null) return false;
    return std.mem.startsWith(u8, std.mem.span(name), metal_name);
}

/// Ports the shared and private buffer ifaces, from
/// `ggml_backend_metal_buffer_shared_free_buffer` and
/// `ggml_backend_metal_buffer_private_free_buffer`
/// (ggml-metal.cpp:30, 106 @c1d0e7a00) down.
///
/// Parameters:
/// - `shared`: which of the two the C writes out. The bodies are
///   identical; `diff` says the only difference is that the private set
///   asserts `!ggml_metal_buffer_is_shared`.
fn BufferIface(comptime shared: bool) type {
    return struct {
        fn ctxOf(buffer: c.ggml_backend_buffer_t) *mc.Buffer {
            const b = impl.one(c.struct_ggml_backend_buffer, buffer);
            const ctx: *mc.Buffer = @ptrCast(b.context.?);
            // The C's assertion, in both polarities.
            if (shared) {
                impl.assert(mc.ggml_metal_buffer_is_shared(ctx), "ggml_metal_buffer_is_shared(ctx)");
            } else {
                impl.assert(!mc.ggml_metal_buffer_is_shared(ctx), "!ggml_metal_buffer_is_shared(ctx)");
            }
            return ctx;
        }

        fn freeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
            mc.ggml_metal_buffer_free(ctxOf(buffer));
        }

        fn getBase(buffer: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
            return mc.ggml_metal_buffer_get_base(ctxOf(buffer));
        }

        fn memsetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]Tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
            mc.ggml_metal_buffer_memset_tensor(ctxOf(buffer), impl.one(Tensor, tensor), value, offset, size);
        }

        fn setTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]Tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
            mc.ggml_metal_buffer_set_tensor(ctxOf(buffer), impl.one(Tensor, tensor), data, offset, size);
        }

        fn getTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]const Tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
            mc.ggml_metal_buffer_get_tensor(ctxOf(buffer), @ptrCast(tensor), data, offset, size);
        }

        fn cpyTensor(buffer: c.ggml_backend_buffer_t, src: [*c]const Tensor, dst: [*c]Tensor) callconv(.c) bool {
            const ctx = ctxOf(buffer);
            const s = impl.one(Tensor, @constCast(src));
            if (!bufferIsMetal(s.buffer)) return false;
            return mc.ggml_metal_buffer_cpy_tensor(ctx, s, impl.one(Tensor, dst));
        }

        fn clear(buffer: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
            mc.ggml_metal_buffer_clear(ctxOf(buffer), value);
        }

        /// Ports `ggml_backend_metal_buffer_shared_i` and
        /// `ggml_backend_metal_buffer_private_i`
        /// (ggml-metal.cpp:90, 166 @c1d0e7a00).
        pub const iface = c.ggml_backend_buffer_i{
            .free_buffer = freeBuffer,
            .get_base = getBase,
            .init_tensor = null,
            .memset_tensor = memsetTensor,
            .set_tensor = setTensor,
            .get_tensor = getTensor,
            .set_tensor_2d = null,
            .get_tensor_2d = null,
            .cpy_tensor = cpyTensor,
            .clear = clear,
            .reset = null,
        };
    };
}

const shared_buffer = BufferIface(true);
const private_buffer = BufferIface(false);

// -----------------------------------------------------------------------------
// Buffer types

/// Ports `struct ggml_backend_metal_buffer_type` (ggml-metal.cpp:189
/// @c1d0e7a00).
///
/// The C's `name` is a `std::string`; here it is a fixed buffer, since the
/// longest it holds is `"MTL15_Private"`. `get_name` returns a pointer
/// into it, so it has to outlive every caller — which it does, being part
/// of a static.
const BufferTypeCtx = extern struct {
    device: c_int,
    name: [32]u8,
};

/// Ports `ggml_backend_metal_buffer_type_alloc_buffer` (ggml-metal.cpp:203
/// @c1d0e7a00): the body the three buffer types share.
///
/// Parameters:
/// - `buft`: the buffer type being allocated from.
/// - `size`: bytes.
/// - `shared`: whether to ask Metal for shared memory. The C asks, then
///   picks the iface from what it actually got.
fn bufferTypeAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize, shared: bool) c.ggml_backend_buffer_t {
    const bt = impl.one(c.struct_ggml_backend_buffer_type, buft);
    const ctx_dev: *mc.Device = @ptrCast(impl.one(c.struct_ggml_backend_device, bt.device).context.?);
    const res = mc.ggml_metal_buffer_init(ctx_dev, size, shared) orelse return null;

    // Note the C reads the result rather than the request: a shared
    // allocation can come back private.
    const buf_i = if (mc.ggml_metal_buffer_is_shared(res))
        shared_buffer.iface
    else
        private_buffer.iface;

    return c.ggml_backend_buffer_init(buft, buf_i, res, size);
}

/// Ports `ggml_backend_metal_buffer_type_get_alloc_size` (ggml-metal.cpp:214
/// @c1d0e7a00): `ggml_nbytes`, plus the scratch some ops need alongside
/// their output.
fn bufferTypeGetAllocSize(buft: c.ggml_backend_buffer_type_t, tensor: [*c]const Tensor) callconv(.c) usize {
    _ = buft;
    const t = impl.one(Tensor, @constCast(tensor));
    var res = c.ggml_nbytes(t);

    // some operations require additional memory for fleeting data:
    switch (t.op) {
        c.GGML_OP_MUL_MAT_ID => {
            res += mc.ggml_metal_op_mul_mat_id_extra_tpe(t);
            res += mc.ggml_metal_op_mul_mat_id_extra_ids(t);
        },
        c.GGML_OP_FLASH_ATTN_EXT => {
            res += mc.ggml_metal_op_flash_attn_ext_extra_pad(t);
            res += mc.ggml_metal_op_flash_attn_ext_extra_blk(t);
            res += mc.ggml_metal_op_flash_attn_ext_extra_tmp(t);
            res += mc.ggml_metal_op_flash_attn_ext_extra_kv_f16(t);
        },
        c.GGML_OP_CUMSUM, c.GGML_OP_ARGSORT => {
            res *= 2;
        },
        c.GGML_OP_TOP_K => {
            res = 2 * @sizeOf(i32) * @as(usize, @intCast(c.ggml_nelements(t.src[0])));
        },
        else => {},
    }

    return res;
}

/// Which of the three buffer types an instantiation is.
const Kind = enum {
    shared,
    private,
    mapped,

    /// The suffix the C appends to `"MTL<i>"`: none, `_Private`,
    /// `_Mapped`, set where each context is built inside
    /// `ggml_backend_metal_buffer_type_shared`,
    /// `ggml_backend_metal_buffer_type_private` and
    /// `ggml_backend_metal_buffer_type_mapped`
    /// (ggml-metal.cpp:283, 359, 435 @c1d0e7a00).
    fn suffix(self: Kind) []const u8 {
        return switch (self) {
            .shared => "",
            .private => "_Private",
            .mapped => "_Mapped",
        };
    }

    /// Whether `alloc_buffer` asks for shared memory. The mapped type
    /// prefers it, which the C notes in a comment.
    fn wantsShared(self: Kind) bool {
        return self != .private;
    }
};

/// Ports the three buffer-type groups, from
/// `ggml_backend_metal_buffer_type_shared_get_name`,
/// `ggml_backend_metal_buffer_type_private_get_name` and
/// `ggml_backend_metal_buffer_type_mapped_get_name`
/// (ggml-metal.cpp:251, 327, 402 @c1d0e7a00) down, and the factories
/// `ggml_backend_metal_buffer_type_shared`,
/// `ggml_backend_metal_buffer_type_private` and
/// `ggml_backend_metal_buffer_type_mapped`
/// (ggml-metal.cpp:283, 359, 435 @c1d0e7a00).
fn BufferType(comptime kind: Kind) type {
    return struct {
        fn getName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
            const bt = impl.one(c.struct_ggml_backend_buffer_type, buft);
            const ctx: *const BufferTypeCtx = @ptrCast(@alignCast(bt.context.?));
            return @ptrCast(&ctx.name);
        }

        fn allocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
            return bufferTypeAllocBuffer(buft, size, kind.wantsShared());
        }

        fn getAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
            _ = buft;
            return 32;
        }

        fn getMaxSize(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
            const bt = impl.one(c.struct_ggml_backend_buffer_type, buft);
            const ctx_dev: *mc.Device = @ptrCast(impl.one(c.struct_ggml_backend_device, bt.device).context.?);
            return mc.ggml_metal_device_get_props(ctx_dev).max_buffer_size;
        }

        fn isHost(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
            _ = buft;
            return false;
        }

        const iface = c.struct_ggml_backend_buffer_type_i{
            .get_name = getName,
            .alloc_buffer = allocBuffer,
            .get_alignment = getAlignment,
            .get_max_size = getMaxSize,
            .get_alloc_size = bufferTypeGetAllocSize,
            .is_host = isHost,
        };

        var bufts: [max_devices]c.struct_ggml_backend_buffer_type = undefined;
        var ctxs: [max_devices]BufferTypeCtx = undefined;
        var ready = std.atomic.Value(bool).init(false);
        var mutex: std.c.pthread_mutex_t = .{};

        /// The C's function-local `static std::mutex` plus `static bool
        /// initialized`, written out. Zig 0.16 has no `std.once`; this is
        /// the shape `cpu_backend.zig` and `repack/dispatch.zig` use.
        ///
        /// Parameters:
        /// - `device`: which device's buffer type to return.
        ///
        /// Return: the buffer type, borrowed for the life of the process.
        fn get(device: c_int) c.ggml_backend_buffer_type_t {
            if (!ready.load(.acquire)) {
                _ = std.c.pthread_mutex_lock(&mutex);
                defer _ = std.c.pthread_mutex_unlock(&mutex);
                if (!ready.load(.acquire)) {
                    var i: c_int = 0;
                    while (i < g_devices) : (i += 1) {
                        const u: usize = @intCast(i);
                        ctxs[u] = .{ .device = i, .name = undefined };
                        // `GGML_METAL_NAME + std::to_string(i) + suffix`.
                        const written = std.fmt.bufPrint(
                            &ctxs[u].name,
                            "{s}{d}{s}",
                            .{ metal_name, i, kind.suffix() },
                        ) catch impl.abort("metal: buffer type name does not fit");
                        ctxs[u].name[written.len] = 0;

                        bufts[u] = .{
                            .iface = iface,
                            .device = c.ggml_backend_reg_dev_get(ggml_backend_metal_reg(), @intCast(i)),
                            .context = &ctxs[u],
                        };
                    }
                    ready.store(true, .release);
                }
            }
            return &bufts[@intCast(device)];
        }
    };
}

const shared_buft = BufferType(.shared);
const private_buft = BufferType(.private);
const mapped_buft = BufferType(.mapped);

// -----------------------------------------------------------------------------
// The backend

fn ctxOfBackend(backend: c.ggml_backend_t) *mc.Context {
    const b = impl.one(c.struct_ggml_backend, backend);
    return @ptrCast(b.context.?);
}

/// Ports `ggml_backend_metal_name` (ggml-metal.cpp:480 @c1d0e7a00).
fn backendGetName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return mc.ggml_metal_get_name(ctxOfBackend(backend));
}

/// Ports `ggml_backend_metal_free` (ggml-metal.cpp:486 @c1d0e7a00).
fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    const ctx = ctxOfBackend(backend);

    // wait for any ongoing async operations to finish
    mc.ggml_metal_synchronize(ctx);
    mc.ggml_metal_free(ctx);

    // The C `malloc`s the backend in `init` and `free`s it here.
    std.c.free(backend);
}

/// Ports `ggml_backend_metal_synchronize` (ggml-metal.cpp:497 @c1d0e7a00).
fn backendSynchronize(backend: c.ggml_backend_t) callconv(.c) void {
    mc.ggml_metal_synchronize(ctxOfBackend(backend));
}

/// Ports `ggml_backend_metal_set_tensor_async` (ggml-metal.cpp:503
/// @c1d0e7a00).
fn backendSetTensorAsync(backend: c.ggml_backend_t, tensor: [*c]Tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    mc.ggml_metal_set_tensor_async(ctxOfBackend(backend), impl.one(Tensor, tensor), data, offset, size);
}

/// Ports `ggml_backend_metal_get_tensor_async` (ggml-metal.cpp:509
/// @c1d0e7a00).
fn backendGetTensorAsync(backend: c.ggml_backend_t, tensor: [*c]const Tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    mc.ggml_metal_get_tensor_async(ctxOfBackend(backend), @ptrCast(tensor), data, offset, size);
}

/// Ports `ggml_backend_metal_cpy_tensor_async` (ggml-metal.cpp:515
/// @c1d0e7a00): only reached in multi-GPU setups.
fn backendCpyTensorAsync(backend_src: c.ggml_backend_t, backend_dst: c.ggml_backend_t, src: [*c]const Tensor, dst: [*c]Tensor) callconv(.c) bool {
    if (!ggml_backend_is_metal(backend_src) or !ggml_backend_is_metal(backend_dst)) return false;

    const s = impl.one(Tensor, @constCast(src));
    const d = impl.one(Tensor, dst);
    if (!bufferIsMetal(s.buffer) or !bufferIsMetal(d.buffer)) return false;

    return mc.ggml_metal_cpy_tensor_async(ctxOfBackend(backend_src), ctxOfBackend(backend_dst), s, d);
}

/// Ports `ggml_backend_metal_graph_compute` (ggml-metal.cpp:536
/// @c1d0e7a00).
fn backendGraphCompute(backend: c.ggml_backend_t, cgraph: *c.ggml_cgraph) callconv(.c) c.enum_ggml_status {
    return mc.ggml_metal_graph_compute(ctxOfBackend(backend), cgraph);
}

/// Ports `ggml_backend_metal_event_record` (ggml-metal.cpp:542 @c1d0e7a00).
fn backendEventRecord(backend: c.ggml_backend_t, event: c.ggml_backend_event_t) callconv(.c) void {
    const ev: *mc.Event = @ptrCast(impl.one(c.struct_ggml_backend_event, event).context.?);
    mc.ggml_metal_event_record(ctxOfBackend(backend), ev);
}

/// Ports `ggml_backend_metal_event_wait` (ggml-metal.cpp:549 @c1d0e7a00).
fn backendEventWait(backend: c.ggml_backend_t, event: c.ggml_backend_event_t) callconv(.c) void {
    const ev: *mc.Event = @ptrCast(impl.one(c.struct_ggml_backend_event, event).context.?);
    mc.ggml_metal_event_wait(ctxOfBackend(backend), ev);
}

/// Ports `ggml_backend_metal_graph_optimize` (ggml-metal.cpp:556
/// @c1d0e7a00).
///
/// The context's own optimize calls `ggml_graph_optimize`, which is
/// `metal/common.zig`. Referenced here so the dependency is visible.
fn backendGraphOptimize(backend: c.ggml_backend_t, cgraph: *c.ggml_cgraph) callconv(.c) void {
    mc.ggml_metal_graph_optimize(ctxOfBackend(backend), cgraph);
}

/// Ports `ggml_backend_metal_set_n_cb` (ggml-metal.cpp:562 @c1d0e7a00).
fn backendSetNCb(backend: c.ggml_backend_t, n_cb: c_int) void {
    impl.assert(ggml_backend_is_metal(backend), "ggml_backend_is_metal(backend)");
    mc.ggml_metal_set_n_cb(ctxOfBackend(backend), n_cb);
}

/// Ports `ggml_backend_metal_i` (ggml-metal.cpp:570 @c1d0e7a00).
const backend_i = c.struct_ggml_backend_i{
    .get_name = backendGetName,
    .free = backendFree,
    .set_tensor_async = backendSetTensorAsync,
    .get_tensor_async = backendGetTensorAsync,
    .set_tensor_2d_async = null,
    .get_tensor_2d_async = null,
    // only needed for multi-GPU setups
    .cpy_tensor_async = backendCpyTensorAsync,
    .synchronize = backendSynchronize,
    .graph_plan_create = null,
    .graph_plan_free = null,
    .graph_plan_update = null,
    .graph_plan_compute = null,
    .graph_compute = @ptrCast(&backendGraphCompute),
    .event_record = backendEventRecord,
    .event_wait = backendEventWait,
    .graph_optimize = @ptrCast(&backendGraphOptimize),
};

/// Ports `ggml_backend_metal_guid` (ggml-metal.cpp:589 @c1d0e7a00).
fn guid() c.ggml_guid_t {
    const S = struct {
        var value: c.ggml_guid = .{ 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    };
    return &S.value;
}

/// The body `ggml_backend_metal_init` and
/// `ggml_backend_metal_device_init_backend` share
/// (ggml-metal.cpp:594, 690 @c1d0e7a00) — the C writes it out twice,
/// identically but for the unused `params`.
fn backendFromDevice(dev: c.ggml_backend_dev_t) c.ggml_backend_t {
    const d = impl.one(c.struct_ggml_backend_device, dev);
    const ctx_dev: *mc.Device = @ptrCast(d.context.?);

    const ctx = mc.ggml_metal_init(ctx_dev) orelse {
        impl.logError("%s: error: failed to allocate context\n", .{"ggml_backend_metal_init"});
        return null;
    };

    // `malloc`, not the Zig allocator: `backendFree` calls `free`.
    const backend: c.ggml_backend_t = @ptrCast(@alignCast(std.c.malloc(@sizeOf(c.struct_ggml_backend)) orelse return null));
    backend.* = .{
        .guid = guid(),
        .iface = backend_i,
        .device = dev,
        .context = ctx,
    };

    backendSetNCb(backend, 1);

    return backend;
}

/// Ports `ggml_backend_metal_init` (ggml-metal.cpp:594 @c1d0e7a00).
///
/// Return: a new Metal backend on device 0, or null if the context could
/// not be allocated. Owned by the caller, freed with `ggml_backend_free`.
pub export fn ggml_backend_metal_init() callconv(.c) c.ggml_backend_t {
    return backendFromDevice(c.ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0));
}

/// Ports `ggml_backend_is_metal` (ggml-metal.cpp:618 @c1d0e7a00).
pub export fn ggml_backend_is_metal(backend: c.ggml_backend_t) callconv(.c) bool {
    return backend != null and
        c.ggml_guid_matches(impl.one(c.struct_ggml_backend, backend).guid, guid());
}

/// Ports `ggml_backend_metal_set_abort_callback` (ggml-metal.cpp:622
/// @c1d0e7a00).
pub export fn ggml_backend_metal_set_abort_callback(
    backend: c.ggml_backend_t,
    abort_callback: c.ggml_abort_callback,
    user_data: ?*anyopaque,
) callconv(.c) void {
    impl.assert(ggml_backend_is_metal(backend), "ggml_backend_is_metal(backend)");
    mc.ggml_metal_set_abort_callback(ctxOfBackend(backend), abort_callback, user_data);
}

/// Ports `ggml_backend_metal_supports_family` (ggml-metal.cpp:630
/// @c1d0e7a00).
pub export fn ggml_backend_metal_supports_family(backend: c.ggml_backend_t, family: c_int) callconv(.c) bool {
    impl.assert(ggml_backend_is_metal(backend), "ggml_backend_is_metal(backend)");
    return mc.ggml_metal_supports_family(ctxOfBackend(backend), family);
}

/// Ports `ggml_backend_metal_capture_next_compute` (ggml-metal.cpp:638
/// @c1d0e7a00).
pub export fn ggml_backend_metal_capture_next_compute(backend: c.ggml_backend_t) callconv(.c) void {
    impl.assert(ggml_backend_is_metal(backend), "ggml_backend_is_metal(backend)");
    mc.ggml_metal_capture_next_compute(ctxOfBackend(backend));
}

// -----------------------------------------------------------------------------
// The device

fn devOf(dev: c.ggml_backend_dev_t) *mc.Device {
    return @ptrCast(impl.one(c.struct_ggml_backend_device, dev).context.?);
}

/// Ports `ggml_backend_metal_device_get_name` (ggml-metal.cpp:648
/// @c1d0e7a00).
fn deviceGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return @ptrCast(&mc.ggml_metal_device_get_props(devOf(dev)).name);
}

/// Ports `ggml_backend_metal_device_get_description` (ggml-metal.cpp:656
/// @c1d0e7a00).
fn deviceGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return @ptrCast(&mc.ggml_metal_device_get_props(devOf(dev)).desc);
}

/// Ports `ggml_backend_metal_device_get_memory` (ggml-metal.cpp:662
/// @c1d0e7a00).
fn deviceGetMemory(dev: c.ggml_backend_dev_t, free: *usize, total: *usize) callconv(.c) void {
    mc.ggml_metal_device_get_memory(devOf(dev), free, total);
}

/// Ports `ggml_backend_metal_device_get_type` (ggml-metal.cpp:668
/// @c1d0e7a00).
fn deviceGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    _ = dev;
    return c.GGML_BACKEND_DEVICE_TYPE_GPU;
}

/// Ports `ggml_backend_metal_device_get_props` (ggml-metal.cpp:674
/// @c1d0e7a00).
fn deviceGetProps(dev: c.ggml_backend_dev_t, props: *c.struct_ggml_backend_dev_props) callconv(.c) void {
    props.name = deviceGetName(dev);
    props.description = deviceGetDescription(dev);
    props.type = deviceGetType(dev);

    deviceGetMemory(dev, &props.memory_free, &props.memory_total);

    props.caps = .{
        .async = true,
        .host_buffer = false,
        .buffer_from_host_ptr = true,
        .events = true,
        .mmap_support = true,
    };
}

/// Ports `ggml_backend_metal_device_init_backend` (ggml-metal.cpp:690
/// @c1d0e7a00).
fn deviceInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = params;
    return backendFromDevice(dev);
}

/// Ports `ggml_backend_metal_device_get_buffer_type` (ggml-metal.cpp:715
/// @c1d0e7a00).
fn deviceGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    const props = mc.ggml_metal_device_get_props(devOf(dev));
    return if (props.use_shared_buffers)
        shared_buft.get(props.device)
    else
        private_buft.get(props.device);
}

/// Ports `ggml_backend_metal_device_buffer_mapped` (ggml-metal.cpp:723
/// @c1d0e7a00).
///
/// The mapped buffer gets the *shared* iface, not a mapped one: the C has
/// only two ifaces.
fn deviceBufferMapped(dev: c.ggml_backend_dev_t, ptr: ?*anyopaque, size: usize, max_tensor_size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const ctx_dev = devOf(dev);
    const res = mc.ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size) orelse return null;
    const props = mc.ggml_metal_device_get_props(ctx_dev);
    return c.ggml_backend_buffer_init(mapped_buft.get(props.device), shared_buffer.iface, res, size);
}

/// Ports `ggml_backend_metal_device_supports_op` (ggml-metal.cpp:733
/// @c1d0e7a00).
fn deviceSupportsOp(dev: c.ggml_backend_dev_t, op: *const Tensor) callconv(.c) bool {
    return mc.ggml_metal_device_supports_op(devOf(dev), op);
}

/// Ports `ggml_backend_metal_device_supports_buft` (ggml-metal.cpp:739
/// @c1d0e7a00): the buffer type must be one of ours, on this device.
fn deviceSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    if (buft == null) return false;
    const bt = impl.one(c.struct_ggml_backend_buffer_type, buft);
    if (bt.device != dev) return false;
    const name = bt.iface.get_name;
    return name == shared_buft.getName or
        name == private_buft.getName or
        name == mapped_buft.getName;
}

/// Ports `get_op_batch_size` (ggml-metal.cpp:749 @c1d0e7a00).
fn opBatchSize(op: *const Tensor) i64 {
    return switch (op.op) {
        c.GGML_OP_MUL_MAT => op.ne[1],
        c.GGML_OP_MUL_MAT_ID => op.ne[2],
        else => c.ggml_nrows(op),
    };
}

/// Ports `ggml_backend_metal_device_offload_op` (ggml-metal.cpp:760
/// @c1d0e7a00).
fn deviceOffloadOp(dev: c.ggml_backend_dev_t, op: *const Tensor) callconv(.c) bool {
    const props = mc.ggml_metal_device_get_props(devOf(dev));
    return (op.op == c.GGML_OP_MUL_MAT or op.op == c.GGML_OP_MUL_MAT_ID) and
        opBatchSize(op) >= props.op_offload_min_batch_size;
}

/// Ports `ggml_backend_metal_device_event_new` (ggml-metal.cpp:768
/// @c1d0e7a00).
fn deviceEventNew(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_event_t {
    const event = mc.ggml_metal_device_event_init(devOf(dev));
    impl.assert(event != null, "event");

    const ev = allocator.create(c.struct_ggml_backend_event) catch impl.abort("metal: out of memory allocating an event");
    ev.* = .{ .device = dev, .context = event };
    return ev;
}

/// Ports `ggml_backend_metal_device_event_free` (ggml-metal.cpp:782
/// @c1d0e7a00).
fn deviceEventFree(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    const e = impl.one(c.struct_ggml_backend_event, event);
    const ev: *mc.Event = @ptrCast(e.context.?);
    mc.ggml_metal_device_event_free(devOf(dev), ev);
    allocator.destroy(e);
}

/// Ports `ggml_backend_metal_device_event_synchronize` (ggml-metal.cpp:792
/// @c1d0e7a00).
fn deviceEventSynchronize(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    const ev: *mc.Event = @ptrCast(impl.one(c.struct_ggml_backend_event, event).context.?);
    mc.ggml_metal_device_event_synchronize(devOf(dev), ev);
}

/// Ports `ggml_backend_metal_device_i` (ggml-metal.cpp:800 @c1d0e7a00).
const device_i = c.struct_ggml_backend_device_i{
    .get_name = deviceGetName,
    .get_description = deviceGetDescription,
    .get_memory = @ptrCast(&deviceGetMemory),
    .get_type = deviceGetType,
    .get_props = @ptrCast(&deviceGetProps),
    .init_backend = deviceInitBackend,
    .get_buffer_type = deviceGetBufferType,
    .get_host_buffer_type = null,
    .buffer_from_host_ptr = deviceBufferMapped,
    .supports_op = @ptrCast(&deviceSupportsOp),
    .supports_buft = deviceSupportsBuft,
    .offload_op = @ptrCast(&deviceOffloadOp),
    .event_new = deviceEventNew,
    .event_free = deviceEventFree,
    .event_synchronize = deviceEventSynchronize,
};

// -----------------------------------------------------------------------------
// The registry

/// Ports `struct ggml_backend_metal_reg` (ggml-metal.cpp:820 @c1d0e7a00).
///
/// The C's `std::vector<ggml_backend_dev_t>` becomes a fixed array of the
/// maximum plus a count, as `cpu_backend.zig` did for its extra buffer
/// types.
const RegCtx = struct {
    devices: [max_devices]c.ggml_backend_dev_t = @splat(null),
    n_devices: usize = 0,
};

/// Ports `ggml_backend_metal_reg_get_name` (ggml-metal.cpp:844
/// @c1d0e7a00).
fn regGetName(reg: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = reg;
    return metal_name;
}

/// Ports `ggml_backend_metal_reg_device_count` (ggml-metal.cpp:850
/// @c1d0e7a00).
fn regDeviceCount(reg: c.ggml_backend_reg_t) callconv(.c) usize {
    const r = impl.one(c.struct_ggml_backend_reg, reg);
    const ctx: *const RegCtx = @ptrCast(@alignCast(r.context.?));
    return ctx.n_devices;
}

/// Ports `ggml_backend_metal_reg_device_get` (ggml-metal.cpp:855
/// @c1d0e7a00).
fn regDeviceGet(reg: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    const r = impl.one(c.struct_ggml_backend_reg, reg);
    const ctx: *const RegCtx = @ptrCast(@alignCast(r.context.?));
    impl.assert(index < ctx.n_devices, "index < ctx->devices.size()");
    return ctx.devices[index];
}

/// Ports `g_ggml_backend_metal_features` (ggml-metal.cpp:861 @c1d0e7a00).
///
/// `EMBED_LIBRARY` is present because this build defines
/// `GGML_METAL_EMBED_LIBRARY`, which it must: the non-embed path needs
/// `xcrun metal`, which Zig cannot replace.
var features = [_]c.struct_ggml_backend_feature{
    .{ .name = "EMBED_LIBRARY", .value = "1" },
    .{ .name = null, .value = null },
};

/// Ports `ggml_backend_metal_get_features` (ggml-metal.cpp:868
/// @c1d0e7a00).
fn regGetFeatures(reg: c.ggml_backend_reg_t) callconv(.c) [*c]c.struct_ggml_backend_feature {
    _ = reg;
    return &features;
}

// The six tuning hooks the C exposes through `get_proc_address`
// (ggml-metal.cpp:933 @c1d0e7a00). `test-backend-ops` uses them to sweep
// the FA vector configurations. They reach `metal/tuning.zig` directly
// here rather than through its mangled C++ names.

const tuning = @import("tuning.zig");

/// Ports `ggml_backend_metal_tuning_set_fa_vec_override`
/// (ggml-metal.cpp:875 @c1d0e7a00).
fn tuningSetFaVecOverride(Q: c_int, NE: c_int) callconv(.c) void {
    tuning.setOverride(.{ .Q = @intCast(Q), .NE = @intCast(NE) });
}

/// Ports `ggml_backend_metal_tuning_clear_fa_vec_override`
/// (ggml-metal.cpp:879 @c1d0e7a00).
fn tuningClearFaVecOverride() callconv(.c) void {
    tuning.clearOverride();
}

/// Ports `ggml_backend_metal_tuning_fa_vec_ne11_bucket`
/// (ggml-metal.cpp:883 @c1d0e7a00).
fn tuningNe11Bucket(ne11: i64) callconv(.c) c_int {
    return tuning.ne11Bucket(ne11);
}

/// Ports `ggml_backend_metal_tuning_fa_vec_ne01_bucket`
/// (ggml-metal.cpp:887 @c1d0e7a00).
fn tuningNe01Bucket(ne01: i64) callconv(.c) c_int {
    return tuning.ne01Bucket(ne01);
}

/// Ports `ggml_backend_metal_tuning_fa_vec_baseline_ne`
/// (ggml-metal.cpp:891 @c1d0e7a00).
fn tuningBaselineNe(dk: c_int, dv: c_int) callconv(.c) c_int {
    return tuning.baselineNe(dk, dv);
}

/// Ports `ggml_backend_metal_tuning_device_token` (ggml-metal.cpp:895
/// @c1d0e7a00).
fn tuningDeviceToken(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    const props = mc.ggml_metal_device_get_props(devOf(dev));
    return mc.ggml_metal_device_id_token(props.device_id);
}

/// Ports `ggml_backend_metal_get_proc_address` (ggml-metal.cpp:901
/// @c1d0e7a00).
fn regGetProcAddress(reg: c.ggml_backend_reg_t, name: [*c]const u8) callconv(.c) ?*anyopaque {
    _ = reg;
    const want = std.mem.span(name);
    const table = .{
        .{ "ggml_backend_get_features", @as(?*anyopaque, @ptrCast(@constCast(&regGetFeatures))) },
        .{ "ggml_backend_metal_tuning_set_fa_vec_override", @as(?*anyopaque, @ptrCast(@constCast(&tuningSetFaVecOverride))) },
        .{ "ggml_backend_metal_tuning_clear_fa_vec_override", @as(?*anyopaque, @ptrCast(@constCast(&tuningClearFaVecOverride))) },
        .{ "ggml_backend_metal_tuning_fa_vec_ne11_bucket", @as(?*anyopaque, @ptrCast(@constCast(&tuningNe11Bucket))) },
        .{ "ggml_backend_metal_tuning_fa_vec_ne01_bucket", @as(?*anyopaque, @ptrCast(@constCast(&tuningNe01Bucket))) },
        .{ "ggml_backend_metal_tuning_fa_vec_baseline_ne", @as(?*anyopaque, @ptrCast(@constCast(&tuningBaselineNe))) },
        .{ "ggml_backend_metal_tuning_device_token", @as(?*anyopaque, @ptrCast(@constCast(&tuningDeviceToken))) },
    };
    inline for (table) |entry| {
        if (std.mem.eql(u8, want, entry[0])) return entry[1];
    }
    return null;
}

/// Ports `ggml_backend_metal_reg_i` (ggml-metal.cpp:929 @c1d0e7a00).
const reg_i = c.struct_ggml_backend_reg_i{
    .get_name = regGetName,
    .get_device_count = regDeviceCount,
    .get_device = regDeviceGet,
    .get_proc_address = regGetProcAddress,
};

var reg_storage: c.struct_ggml_backend_reg = undefined;
var reg_ctx: RegCtx = .{};
var reg_devices: [max_devices]c.struct_ggml_backend_device = undefined;
var reg_ready = std.atomic.Value(bool).init(false);
var reg_mutex: std.c.pthread_mutex_t = .{};

/// Ports `ggml_backend_metal_reg` (ggml-metal.cpp:820 @c1d0e7a00).
///
/// Return: the Metal registry, borrowed for the life of the process.
pub export fn ggml_backend_metal_reg() callconv(.c) c.ggml_backend_reg_t {
    if (!reg_ready.load(.acquire)) {
        _ = std.c.pthread_mutex_lock(&reg_mutex);
        defer _ = std.c.pthread_mutex_unlock(&reg_mutex);
        if (!reg_ready.load(.acquire)) {
            // The C reads this on every call, outside the `initialized`
            // guard, so a later change still takes effect for a later
            // buffer type. Kept where it is.
            if (std.c.getenv("GGML_METAL_DEVICES")) |env| {
                g_devices = std.fmt.parseInt(c_int, std.mem.span(env), 10) catch 0;
            }

            // Workaround for a macOS limitation
            // (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
            // until a proper fix becomes possible.
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            _ = setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 1);

            var i: c_int = 0;
            while (i < g_devices) : (i += 1) {
                const u: usize = @intCast(i);
                reg_devices[u] = .{
                    .iface = device_i,
                    .reg = &reg_storage,
                    .context = mc.ggml_metal_device_get(i, g_devices),
                };
                reg_ctx.devices[u] = &reg_devices[u];
                reg_ctx.n_devices = u + 1;
            }

            reg_storage = .{
                .api_version = c.GGML_BACKEND_API_VERSION,
                .iface = reg_i,
                .context = &reg_ctx,
            };

            reg_ready.store(true, .release);
        }
    }
    return &reg_storage;
}

comptime {
    // `metal/common.zig` supplies `ggml_graph_optimize`, which the Metal
    // context calls. Referenced so the dependency is explicit rather than
    // incidental to both being in the same module.
    _ = metal_common;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the three buffer types get distinct names per device" {
    // `GGML_METAL_NAME + to_string(i) + suffix`. Checked by formatting
    // into the same fixed buffer the port uses, which is also what proves
    // "MTL15_Private" fits.
    var buf: [32]u8 = undefined;
    for ([_]struct { Kind, []const u8 }{
        .{ .shared, "MTL0" },
        .{ .private, "MTL0_Private" },
        .{ .mapped, "MTL0_Mapped" },
    }) |case| {
        const got = try std.fmt.bufPrint(&buf, "{s}{d}{s}", .{ metal_name, 0, case[0].suffix() });
        try std.testing.expectEqualStrings(case[1], got);
    }

    const longest = try std.fmt.bufPrint(&buf, "{s}{d}{s}", .{ metal_name, max_devices - 1, Kind.private.suffix() });
    try std.testing.expectEqualStrings("MTL15_Private", longest);
    // plus the NUL the port writes after it
    try std.testing.expect(longest.len + 1 <= buf.len);
}

test "only the private buffer type asks for private memory" {
    try std.testing.expect(Kind.shared.wantsShared());
    try std.testing.expect(Kind.mapped.wantsShared());
    try std.testing.expect(!Kind.private.wantsShared());
}

test "the three get_name functions are distinguishable" {
    // `supports_buft` compares `buft->iface.get_name` against all three.
    // If the compiler folded them to one address the comparison would
    // still give the C's answer, but the structure would not be what the
    // port claims, so it is checked.
    const a = @intFromPtr(&shared_buft.getName);
    const b = @intFromPtr(&private_buft.getName);
    const d = @intFromPtr(&mapped_buft.getName);
    try std.testing.expect(a != b or b != d or a != d);
}

test "op batch size reads the dimension the op indexes by" {
    var t = std.mem.zeroes(Tensor);
    t.ne = .{ 7, 11, 13, 1 };

    t.op = c.GGML_OP_MUL_MAT;
    try std.testing.expectEqual(@as(i64, 11), opBatchSize(&t));

    t.op = c.GGML_OP_MUL_MAT_ID;
    try std.testing.expectEqual(@as(i64, 13), opBatchSize(&t));

    // anything else falls back to the row count
    t.op = c.GGML_OP_ADD;
    try std.testing.expectEqual(c.ggml_nrows(&t), opBatchSize(&t));
}

test "the guid is stable and matches itself" {
    try std.testing.expect(c.ggml_guid_matches(guid(), guid()));
    try std.testing.expect(!ggml_backend_is_metal(null));
}
