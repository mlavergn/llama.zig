//! The Vulkan backend's registry and device, and the C functions of
//! `ggml-vulkan.h`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` at v0.4.1
//! (`b29c606e2`). Each declaration below names the C++ it replaces and the
//! line it began at. The instance and the device queries are in
//! `instance.zig`.
//!
//! # The whole host layer
//!
//! Every vtable entry is ported, and since the swap (PLAN.md, "Stage 7 —
//! Vulkan") this is the Vulkan backend `-Dvulkan` builds; `supports_op` is
//! the reference's predicate alone, as Stage 6 left CUDA's. The buffer
//! types are in `buffer.zig`, the backend iface, events and
//! `buffer_from_host_ptr` in `iface.zig`.
//!
//! # `GGML_BACKEND_DL_IMPL` expands to nothing
//!
//! This build is static, as in `metal/backend.zig`.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const m = @import("module.zig");
const instance = m.instance;

const vk = common.vk;
const c = impl.c;
const allocator = common.allocator;

extern fn atoi(s: [*:0]const u8) c_int;

/// Ports `vk_conv_shapes` (ggml-vulkan.cpp:550 @b29c606e2).
pub const ConvShape = enum(u32) { @"128x128", @"64x32", @"32x256", @"64x128" };

/// Ports `vk_conv_block_size` (ggml-vulkan.cpp:558 @b29c606e2).
pub const ConvBlockSize = extern struct { K: u32, NPQ: u32, CRS: u32 };

/// Ports `vk_conv_block_sizes` (ggml-vulkan.cpp:564 @b29c606e2). Exported,
/// because the C++ leaves out its `static` and the reference's contract
/// therefore includes it.
pub var conv_block_sizes = [_]ConvBlockSize{
    .{ .K = 128, .NPQ = 128, .CRS = 16 }, // CONV_SHAPE_128x128
    .{ .K = 64, .NPQ = 32, .CRS = 32 }, // CONV_SHAPE_64x32
    .{ .K = 32, .NPQ = 256, .CRS = 16 }, // CONV_SHAPE_32x256
    .{ .K = 64, .NPQ = 128, .CRS = 16 }, // CONV_SHAPE_64x128
};

// -----------------------------------------------------------------------------
// The backend

/// Ports `ggml_backend_vk_guid` (ggml-vulkan.cpp:18956 @b29c606e2).
var guid: c.ggml_guid = .{ 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

/// Ports `ggml_backend_vk_init` (ggml-vulkan.cpp:18961 @b29c606e2).
fn init(dev_num: usize) callconv(.c) c.ggml_backend_t {
    return m.iface.backendInit(dev_num, &guid);
}

/// Ports `ggml_backend_is_vk` (ggml-vulkan.cpp:18981 @b29c606e2).
fn isVk(backend: c.ggml_backend_t) callconv(.c) bool {
    return backend != null and c.ggml_guid_matches(impl.one(c.struct_ggml_backend, backend).guid, &guid);
}

/// Ports `ggml_backend_vk_get_device_count` (ggml-vulkan.cpp:18985 @b29c606e2).
fn getDeviceCount() callconv(.c) c_int {
    return instance.getDeviceCount();
}

/// Ports `ggml_backend_vk_get_device_description` (ggml-vulkan.cpp:18989 @b29c606e2).
fn getDeviceDescription(device: c_int, description: [*c]u8, description_size: usize) callconv(.c) void {
    impl.assert(device < @as(c_int, @intCast(instance.instance.device_indices.items.len)), "device < (int) vk_instance.device_indices.size()");
    const dev_idx = instance.instance.device_indices.items[@intCast(device)];
    instance.getPhysicalDeviceDescription(dev_idx, description, description_size);
}

/// Ports `ggml_backend_vk_get_device_memory` (ggml-vulkan.cpp:18995 @b29c606e2).
fn getDeviceMemory(device: c_int, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    instance.getDeviceMemory(device, impl.one(usize, free), impl.one(usize, total));
}

/// `ggml_backend_vk_buffer_type` and `ggml_backend_vk_host_buffer_type`
/// live with the buffers they allocate.
const bufferType = m.buffer.bufferType;
const hostBufferType = m.buffer.hostBufferType;

// -----------------------------------------------------------------------------
// The device

/// Ports `ggml_backend_vk_device_context` (ggml-vulkan.cpp:19079 @b29c606e2).
const DeviceCtx = struct {
    device: usize,
    name: [:0]const u8,
    description: [:0]const u8,
    is_integrated_gpu: bool,
    pci_bus_id: [:0]const u8,
    op_offload_min_batch_size: c_int,
};

fn deviceCtx(dev: c.ggml_backend_dev_t) *DeviceCtx {
    return @ptrCast(@alignCast(impl.one(c.struct_ggml_backend_device, dev).context.?));
}

/// Ports `ggml_backend_vk_device_get_name` (ggml-vulkan.cpp:19088 @b29c606e2).
fn deviceGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return deviceCtx(dev).name.ptr;
}

/// Ports `ggml_backend_vk_device_get_description` (ggml-vulkan.cpp:19093 @b29c606e2).
fn deviceGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return deviceCtx(dev).description.ptr;
}

/// Ports `ggml_backend_vk_device_get_memory` (ggml-vulkan.cpp:19098 @b29c606e2).
fn deviceGetMemory(dev: c.ggml_backend_dev_t, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    getDeviceMemory(@intCast(deviceCtx(dev).device), free, total);
}

/// Ports `ggml_backend_vk_device_get_buffer_type` (ggml-vulkan.cpp:19103 @b29c606e2).
fn deviceGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return bufferType(deviceCtx(dev).device);
}

/// Ports `ggml_backend_vk_device_get_host_buffer_type` (ggml-vulkan.cpp:19108 @b29c606e2).
fn deviceGetHostBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    _ = dev;
    return hostBufferType();
}

/// Ports `ggml_backend_vk_device_get_type` (ggml-vulkan.cpp:19113 @b29c606e2).
fn deviceGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    return if (deviceCtx(dev).is_integrated_gpu) c.GGML_BACKEND_DEVICE_TYPE_IGPU else c.GGML_BACKEND_DEVICE_TYPE_GPU;
}

/// Ports `ggml_backend_vk_device_get_props` (ggml-vulkan.cpp:19119 @b29c606e2).
fn deviceGetProps(dev: c.ggml_backend_dev_t, props_c: [*c]c.struct_ggml_backend_dev_props) callconv(.c) void {
    const ctx = deviceCtx(dev);
    const props = impl.one(c.struct_ggml_backend_dev_props, props_c);
    props.name = deviceGetName(dev);
    props.description = deviceGetDescription(dev);
    props.type = deviceGetType(dev);
    props.device_id = if (ctx.pci_bus_id.len == 0) null else ctx.pci_bus_id.ptr;
    deviceGetMemory(dev, &props.memory_free, &props.memory_total);
    props.caps = .{
        .async = true,
        .host_buffer = true,
        .buffer_from_host_ptr = false,
        .events = true,
        .mmap_support = !ctx.is_integrated_gpu,
    };
}

/// Ports `ggml_backend_vk_device_init` (ggml-vulkan.cpp:19136 @b29c606e2).
fn deviceInit(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = params;
    return init(deviceCtx(dev).device);
}

/// Ports `ggml_backend_vk_device_supports_op` (ggml-vulkan.cpp:19142 @b29c606e2);
/// the predicate itself is `supports.zig`.
fn deviceSupportsOp(dev: c.ggml_backend_dev_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    const t = impl.one(c.ggml_tensor, @constCast(op));
    return m.supports.supportsOp(deviceCtx(dev).device, t);
}

/// Ports `ggml_backend_vk_device_supports_buft` (ggml-vulkan.cpp:19797 @b29c606e2).
fn deviceSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    if (!m.buffer.isVkBufferType(buft)) return false;
    const buft_ctx: *m.buffer.BufferTypeCtx = @ptrCast(@alignCast(buft.*.context));
    return buft_ctx.device.idx == deviceCtx(dev).device;
}

/// Ports `ggml_vk_get_op_batch_size` (ggml-vulkan.cpp:19808 @b29c606e2).
fn opBatchSize(op: *const c.ggml_tensor) i64 {
    return switch (op.op) {
        c.GGML_OP_GET_ROWS => 0,
        c.GGML_OP_MUL_MAT => op.ne[1],
        c.GGML_OP_MUL_MAT_ID, c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => op.ne[2],
        else => c.ggml_nrows(op),
    };
}

/// Ports `ggml_backend_vk_device_offload_op` (ggml-vulkan.cpp:19823 @b29c606e2).
fn deviceOffloadOp(dev: c.ggml_backend_dev_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    return opBatchSize(impl.one(c.ggml_tensor, @constCast(op))) >= deviceCtx(dev).op_offload_min_batch_size;
}

/// Ports `ggml_backend_vk_device_event_new` (ggml-vulkan.cpp:19829 @b29c606e2).
fn deviceEventNew(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_event_t {
    return m.iface.deviceEventNew(dev, deviceCtx(dev).device);
}

/// Ports `ggml_backend_vk_device_event_free` (ggml-vulkan.cpp:19852 @b29c606e2).
fn deviceEventFree(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    m.iface.deviceEventFree(deviceCtx(dev).device, event);
}

/// Ports `ggml_backend_vk_device_event_synchronize` (ggml-vulkan.cpp:19872 @b29c606e2).
fn deviceEventSynchronize(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    m.iface.deviceEventSynchronize(deviceCtx(dev).device, event);
}

/// Ports `ggml_backend_vk_device_buffer_from_host_ptr` (ggml-vulkan.cpp:19929 @b29c606e2).
fn deviceBufferFromHostPtr(dev: c.ggml_backend_dev_t, ptr: ?*anyopaque, size: usize, max_tensor_size: usize) callconv(.c) c.ggml_backend_buffer_t {
    _ = max_tensor_size;
    return m.iface.deviceBufferFromHostPtr(dev, deviceCtx(dev).device, ptr, size);
}

/// Ports `ggml_backend_vk_device_i` (ggml-vulkan.cpp:19949 @b29c606e2).
const device_iface: c.struct_ggml_backend_device_i = .{
    .get_name = deviceGetName,
    .get_description = deviceGetDescription,
    .get_memory = deviceGetMemory,
    .get_type = deviceGetType,
    .get_props = deviceGetProps,
    .init_backend = deviceInit,
    .get_buffer_type = deviceGetBufferType,
    .get_host_buffer_type = deviceGetHostBufferType,
    .buffer_from_host_ptr = deviceBufferFromHostPtr,
    .supports_op = deviceSupportsOp,
    .supports_buft = deviceSupportsBuft,
    .offload_op = deviceOffloadOp,
    .event_new = deviceEventNew,
    .event_free = deviceEventFree,
    .event_synchronize = deviceEventSynchronize,
};

// -----------------------------------------------------------------------------
// The registry

/// Ports `ggml_backend_vk_reg_get_name` (ggml-vulkan.cpp:19967 @b29c606e2).
fn regGetName(r: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = r;
    return common.name;
}

/// Ports `ggml_backend_vk_reg_get_device_count` (ggml-vulkan.cpp:19972 @b29c606e2).
fn regGetDeviceCount(r: c.ggml_backend_reg_t) callconv(.c) usize {
    _ = r;
    return @intCast(getDeviceCount());
}

/// `ggml_backend_vk_reg_get_device`'s function-local statics.
const Devices = struct {
    var list: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    var initialized = false;
    var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
};

/// Ports `ggml_backend_vk_reg_get_device` (ggml-vulkan.cpp:19977 @b29c606e2).
fn regGetDevice(r: c.ggml_backend_reg_t, device: usize) callconv(.c) c.ggml_backend_dev_t {
    {
        _ = std.c.pthread_mutex_lock(&Devices.mutex);
        defer _ = std.c.pthread_mutex_unlock(&Devices.mutex);
        if (!Devices.initialized) {
            const min_batch_size: c_int = if (std.c.getenv("GGML_OP_OFFLOAD_MIN_BATCH")) |e| atoi(e) else 32;
            const count: usize = @intCast(getDeviceCount());
            for (0..count) |i| {
                var desc: [256]u8 = undefined;
                getDeviceDescription(@intCast(i), &desc, desc.len);
                var pci: [16]u8 = undefined;
                const ctx = allocator.create(DeviceCtx) catch impl.abort("ggml_vulkan: out of memory");
                ctx.* = .{
                    .device = i,
                    .name = std.fmt.allocPrintSentinel(allocator, "{s}{d}", .{ common.name, i }, 0) catch impl.abort("ggml_vulkan: out of memory"),
                    .description = allocator.dupeSentinel(u8, std.mem.sliceTo(&desc, 0), 0) catch impl.abort("ggml_vulkan: out of memory"),
                    .is_integrated_gpu = instance.getDeviceType(i) == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                    .pci_bus_id = allocator.dupeSentinel(u8, instance.getDevicePciId(i, &pci), 0) catch impl.abort("ggml_vulkan: out of memory"),
                    .op_offload_min_batch_size = min_batch_size,
                };
                const dev = allocator.create(c.struct_ggml_backend_device) catch impl.abort("ggml_vulkan: out of memory");
                dev.* = .{ .iface = device_iface, .reg = r, .context = ctx };
                Devices.list.append(allocator, dev) catch impl.abort("ggml_vulkan: out of memory");
            }
            Devices.initialized = true;
        }
    }
    impl.assert(device < Devices.list.items.len, "device < devices.size()");
    return Devices.list.items[device];
}

/// Ports `ggml_backend_vk_reg_i` (ggml-vulkan.cpp:20011 @b29c606e2).
const reg_iface: c.struct_ggml_backend_reg_i = .{
    .get_name = regGetName,
    .get_device_count = regGetDeviceCount,
    .get_device = regGetDevice,
    .get_proc_address = null,
};

var the_reg: c.struct_ggml_backend_reg = .{
    .api_version = c.GGML_BACKEND_API_VERSION,
    .iface = reg_iface,
    .context = null,
};

/// Ports `ggml_backend_vk_reg` (ggml-vulkan.cpp:20018 @b29c606e2).
///
/// The C++ catches every exception from `ggml_vk_instance_init` and returns
/// null. Its messages go through `VK_LOG_DEBUG` (ggml-vulkan.cpp:219 @b29c606e2),
/// which is empty without `GGML_VULKAN_DEBUG`, so nothing is printed.
pub fn reg() callconv(.c) c.ggml_backend_reg_t {
    instance.instanceInit() catch return null;
    return &the_reg;
}

// -----------------------------------------------------------------------------
// The C ABI: `ggml-vulkan.h`, and the table the C++ leaves unstatic

comptime {
    @export(&init, .{ .name = "ggml_backend_vk_init" });
    @export(&isVk, .{ .name = "ggml_backend_is_vk" });
    @export(&getDeviceCount, .{ .name = "ggml_backend_vk_get_device_count" });
    @export(&getDeviceDescription, .{ .name = "ggml_backend_vk_get_device_description" });
    @export(&getDeviceMemory, .{ .name = "ggml_backend_vk_get_device_memory" });
    @export(&bufferType, .{ .name = "ggml_backend_vk_buffer_type" });
    @export(&hostBufferType, .{ .name = "ggml_backend_vk_host_buffer_type" });
    @export(&reg, .{ .name = "ggml_backend_vk_reg" });
    @export(&conv_block_sizes, .{ .name = "vk_conv_block_sizes" });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
