//! The Vulkan backend's registry and device, and the C functions of
//! `ggml-vulkan.h`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-debug.cpp`, and
//! `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan-types.h` at v1.0.0
//! (`b1b770d90`). Each declaration below names the C++ it replaces and the line it
//! began at. The instance and the device queries are in `instance.zig`.
//!
//! # The whole host layer
//!
//! Every vtable entry is ported, and since the swap
//! this is the Vulkan backend `-Dvulkan` builds; `supports_op` is
//! the reference's predicate alone, as on CUDA. The buffer
//! types are in `buffer.zig`, the backend iface, events and
//! `buffer_from_host_ptr` in `iface.zig`.
//!
//! # `GGML_BACKEND_DL_IMPL` expands to nothing
//!
//! This build is static, as in `metal/backend.zig`.

const std = @import("std");
const sync = @import("sync");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const m = @import("module.zig");
const instance = m.instance;

const vk = common.vk;
const c = impl.c;
const allocator = common.allocator;

extern fn atoi(s: [*:0]const u8) c_int;

/// Ports `vk_conv_shapes` (ggml-vulkan-types.h:380 @b1b770d90).
pub const ConvShape = enum(u32) { @"128x128", @"64x32", @"32x256", @"64x128", @"256x128" };

/// Ports `vk_conv_block_size` (ggml-vulkan-types.h:389 @b1b770d90).
pub const ConvBlockSize = extern struct { K: u32, NPQ: u32, CRS: u32 };

/// Ports `vk_conv_block_sizes` (ggml-vulkan-types.h:395 @b1b770d90). Exported,
/// because the C++ leaves out its `static` and the reference's contract
/// therefore includes it -- weakly, as an `inline`
/// variable in a header.
pub var conv_block_sizes = [_]ConvBlockSize{
    .{ .K = 128, .NPQ = 128, .CRS = 16 }, // CONV_SHAPE_128x128
    .{ .K = 64, .NPQ = 32, .CRS = 32 }, // CONV_SHAPE_64x32
    .{ .K = 32, .NPQ = 256, .CRS = 16 }, // CONV_SHAPE_32x256
    .{ .K = 64, .NPQ = 128, .CRS = 16 }, // CONV_SHAPE_64x128
    .{ .K = 256, .NPQ = 128, .CRS = 16 }, // CONV_SHAPE_256x128
};

// -----------------------------------------------------------------------------
// The backend

/// Ports `ggml_backend_vk_guid` (ggml-vulkan.cpp:15717 @b1b770d90).
var guid: c.ggml_guid = .{ 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

/// Ports `ggml_backend_vk_init` (ggml-vulkan.cpp:15722 @b1b770d90).
fn init(dev_num: usize) callconv(.c) c.ggml_backend_t {
    return m.iface.backendInit(dev_num, &guid);
}

/// Ports `ggml_backend_is_vk` (ggml-vulkan.cpp:15742 @b1b770d90).
pub fn isVk(backend: c.ggml_backend_t) callconv(.c) bool {
    return backend != null and c.ggml_guid_matches(impl.one(c.struct_ggml_backend, backend).guid, &guid);
}

/// Ports `ggml_backend_vk_get_device_count` (ggml-vulkan.cpp:15746 @b1b770d90).
fn getDeviceCount() callconv(.c) c_int {
    return instance.getDeviceCount();
}

/// Ports `ggml_backend_vk_get_device_description` (ggml-vulkan.cpp:15750 @b1b770d90).
fn getDeviceDescription(device: c_int, description: [*c]u8, description_size: usize) callconv(.c) void {
    impl.assert(device < @as(c_int, @intCast(instance.instance.device_indices.items.len)), "device < (int) vk_instance.device_indices.size()");
    const dev_idx = instance.instance.device_indices.items[@intCast(device)];
    instance.getPhysicalDeviceDescription(dev_idx, description, description_size);
}

/// Ports `ggml_backend_vk_get_device_memory` (ggml-vulkan.cpp:15756 @b1b770d90).
fn getDeviceMemory(device: c_int, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    instance.getDeviceMemory(device, impl.one(usize, free), impl.one(usize, total));
}

/// `ggml_backend_vk_buffer_type` and `ggml_backend_vk_host_buffer_type`
/// live with the buffers they allocate.
const bufferType = m.buffer.bufferType;
const hostBufferType = m.buffer.hostBufferType;

// -----------------------------------------------------------------------------
// The device

/// Ports `ggml_backend_vk_device_context` (ggml-vulkan-types.h:1525 @b1b770d90).
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

/// Ports `ggml_backend_vk_device_get_name` (ggml-vulkan.cpp:15841 @b1b770d90).
fn deviceGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return deviceCtx(dev).name.ptr;
}

/// Ports `ggml_backend_vk_device_get_description` (ggml-vulkan.cpp:15846 @b1b770d90).
fn deviceGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return deviceCtx(dev).description.ptr;
}

/// Ports `ggml_backend_vk_device_get_memory` (ggml-vulkan.cpp:15851 @b1b770d90).
fn deviceGetMemory(dev: c.ggml_backend_dev_t, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    getDeviceMemory(@intCast(deviceCtx(dev).device), free, total);
}

/// Ports `ggml_backend_vk_device_get_buffer_type` (ggml-vulkan.cpp:15856 @b1b770d90).
fn deviceGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return bufferType(deviceCtx(dev).device);
}

/// Ports `ggml_backend_vk_device_get_host_buffer_type` (ggml-vulkan.cpp:15861 @b1b770d90).
fn deviceGetHostBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    _ = dev;
    return hostBufferType();
}

/// Ports `ggml_backend_vk_device_get_type` (ggml-vulkan.cpp:15866 @b1b770d90).
fn deviceGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    return if (deviceCtx(dev).is_integrated_gpu) c.GGML_BACKEND_DEVICE_TYPE_IGPU else c.GGML_BACKEND_DEVICE_TYPE_GPU;
}

/// Ports `ggml_backend_vk_device_get_props` (ggml-vulkan.cpp:15872 @b1b770d90).
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

/// Ports `ggml_backend_vk_device_init` (ggml-vulkan.cpp:15889 @b1b770d90).
fn deviceInit(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = params;
    return init(deviceCtx(dev).device);
}

/// Ports `ggml_backend_vk_device_supports_op` (ggml-vulkan.cpp:15895 @b1b770d90);
/// the predicate itself is `supports.zig`.
fn deviceSupportsOp(dev: c.ggml_backend_dev_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    const t = impl.one(c.ggml_tensor, @constCast(op));
    return m.supports.supportsOp(deviceCtx(dev).device, t);
}

/// Ports `ggml_backend_vk_device_supports_buft` (ggml-vulkan.cpp:16556 @b1b770d90).
fn deviceSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    if (!m.buffer.isVkBufferType(buft)) return false;
    const buft_ctx: *m.buffer.BufferTypeCtx = @ptrCast(@alignCast(buft.*.context));
    return buft_ctx.device.idx == deviceCtx(dev).device;
}

/// Ports `ggml_vk_get_op_batch_size` (ggml-vulkan-debug.cpp:821 @b1b770d90).
fn opBatchSize(op: *const c.ggml_tensor) i64 {
    return switch (op.op) {
        c.GGML_OP_GET_ROWS => 0,
        c.GGML_OP_MUL_MAT => op.ne[1],
        c.GGML_OP_MUL_MAT_ID, c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => op.ne[2],
        else => c.ggml_nrows(op),
    };
}

/// Ports `ggml_backend_vk_device_offload_op` (ggml-vulkan.cpp:16567 @b1b770d90).
fn deviceOffloadOp(dev: c.ggml_backend_dev_t, op: [*c]const c.ggml_tensor) callconv(.c) bool {
    return opBatchSize(impl.one(c.ggml_tensor, @constCast(op))) >= deviceCtx(dev).op_offload_min_batch_size;
}

/// Ports `ggml_backend_vk_device_event_new` (ggml-vulkan.cpp:16573 @b1b770d90).
fn deviceEventNew(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_event_t {
    return m.iface.deviceEventNew(dev, deviceCtx(dev).device);
}

/// Ports `ggml_backend_vk_device_event_free` (ggml-vulkan.cpp:16596 @b1b770d90).
fn deviceEventFree(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    m.iface.deviceEventFree(deviceCtx(dev).device, event);
}

/// Ports `ggml_backend_vk_device_event_synchronize` (ggml-vulkan.cpp:16616 @b1b770d90).
fn deviceEventSynchronize(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    m.iface.deviceEventSynchronize(deviceCtx(dev).device, event);
}

/// Ports `ggml_backend_vk_device_buffer_from_host_ptr` (ggml-vulkan.cpp:16648 @b1b770d90).
pub fn deviceBufferFromHostPtr(dev: c.ggml_backend_dev_t, ptr: ?*anyopaque, size: usize, max_tensor_size: usize) callconv(.c) c.ggml_backend_buffer_t {
    _ = max_tensor_size;
    return m.iface.deviceBufferFromHostPtr(dev, deviceCtx(dev).device, ptr, size);
}

/// Ports `ggml_backend_vk_device_i` (ggml-vulkan.cpp:16668 @b1b770d90).
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

/// Ports `ggml_backend_vk_reg_get_name` (ggml-vulkan.cpp:16686 @b1b770d90).
fn regGetName(r: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = r;
    return common.name;
}

/// Ports `ggml_backend_vk_reg_get_device_count` (ggml-vulkan.cpp:16691 @b1b770d90).
fn regGetDeviceCount(r: c.ggml_backend_reg_t) callconv(.c) usize {
    _ = r;
    return @intCast(getDeviceCount());
}

/// `ggml_backend_vk_reg_get_device`'s function-local statics.
const Devices = struct {
    var list: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    var initialized = false;
    var mutex: sync.Mutex = .{};
};

/// Ports `ggml_backend_vk_reg_get_device` (ggml-vulkan.cpp:16696 @b1b770d90).
fn regGetDevice(r: c.ggml_backend_reg_t, device: usize) callconv(.c) c.ggml_backend_dev_t {
    {
        Devices.mutex.lock();
        defer Devices.mutex.unlock();
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

/// Ports `ggml_backend_vk_fusion_get` (ggml-vulkan-debug.cpp:23 @b1b770d90):
/// the device a backend device's fusion stats live on.
fn fusionGet(dev: c.ggml_backend_dev_t) callconv(.c) ?*anyopaque {
    return @ptrCast(m.device.getDevice(deviceCtx(dev).device));
}

/// Ports `ggml_backend_vk_reg_get_proc_address` (ggml-vulkan.cpp:16730 @b1b770d90):
/// the all-reduce of `comm.zig`, everywhere but Windows, and the
/// test-fusion interface.
fn regGetProcAddress(r: c.ggml_backend_reg_t, name: [*c]const u8) callconv(.c) ?*anyopaque {
    _ = r;
    if (m.comm.supported) {
        const n = std.mem.span(name);
        if (std.mem.eql(u8, n, "ggml_backend_comm_init")) return @ptrCast(@constCast(&m.comm.commInit));
        if (std.mem.eql(u8, n, "ggml_backend_comm_free")) return @ptrCast(@constCast(&m.comm.commFree));
        if (std.mem.eql(u8, n, "ggml_backend_comm_allreduce_tensor")) return @ptrCast(@constCast(&m.comm.commAllreduceTensor));
    }
    const n = std.mem.span(name);
    if (std.mem.eql(u8, n, "ggml_backend_fusion_get")) return @ptrCast(@constCast(&fusionGet));
    if (std.mem.eql(u8, n, "ggml_backend_fusion_stats_init")) return @ptrCast(@constCast(&m.perf.fusionStatsInit));
    if (std.mem.eql(u8, n, "ggml_backend_fusion_stats_reset")) return @ptrCast(@constCast(&m.perf.fusionStatsReset));
    if (std.mem.eql(u8, n, "ggml_backend_fusion_stats_get")) return @ptrCast(@constCast(&m.perf.fusionStatsGet));
    if (std.mem.eql(u8, n, "ggml_backend_fusion_set_enabled")) return @ptrCast(@constCast(&m.perf.fusionSetEnabled));
    return null;
}

/// Ports `ggml_backend_vk_reg_i` (ggml-vulkan.cpp:16761 @b1b770d90).
const reg_iface: c.struct_ggml_backend_reg_i = .{
    .get_name = regGetName,
    .get_device_count = regGetDeviceCount,
    .get_device = regGetDevice,
    .get_proc_address = regGetProcAddress,
};

var the_reg: c.struct_ggml_backend_reg = .{
    .api_version = c.GGML_BACKEND_API_VERSION,
    .iface = reg_iface,
    .context = null,
};

/// Ports `ggml_backend_vk_reg` (ggml-vulkan.cpp:16768 @b1b770d90).
///
/// The C++ catches every exception from `ggml_vk_instance_init` and returns
/// null. Its messages go through `VK_LOG_DEBUG` (ggml-vulkan-types.h:217 @b1b770d90),
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
    @export(&conv_block_sizes, .{ .name = "vk_conv_block_sizes", .linkage = .weak });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
