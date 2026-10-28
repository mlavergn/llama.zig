//! The CUDA backend: the device buffer type, the pinned host buffer type,
//! the backend, the device, the registry, and the ten C functions of
//! `ggml-cuda.h`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v0.4.0
//! (`5266f24da`). Each declaration below names the C++ it replaces and the
//! line it began at. The compute loop is in `graph.zig` and `supports_op`
//! in `supports.zig`.
//!
//! # Multi-GPU
//!
//! The communication context for tensor-parallel all-reduce
//! (`ggml_backend_cuda_comm_*`) is ported here, but its `internal` arm
//! calls `allreduce.cu`'s pipeline, which is not ported yet: until it is,
//! `arPipelineInit` reports failure and the chain falls back to the meta
//! backend's butterfly, logging exactly the warning the C logs when its own
//! pipeline fails to start. NCCL is not compiled in either build.
//!
//! # `GGML_BACKEND_DL_IMPL` expands to nothing
//!
//! As in `metal/backend.zig`: this build is static.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const graph = @import("graph.zig");
const supports = @import("supports.zig");
const fattn = @import("fattn.zig");

const cu = common.cu;
const c = impl.c;
const Tensor = c.ggml_tensor;
const allocator = common.allocator;
const cudaCheck = common.cudaCheck;

// -----------------------------------------------------------------------------
// Device buffers

/// Ports `ggml_backend_cuda_buffer_context` (ggml-cuda.cu:729 @5266f24da).
pub const BufferCtx = struct {
    device: c_int,
    dev_ptr: ?*anyopaque,
    name: [16:0]u8 = @splat(0),
};

fn bufferCtx(buffer: c.ggml_backend_buffer_t) *BufferCtx {
    return @ptrCast(@alignCast(impl.one(c.struct_ggml_backend_buffer, buffer).context.?));
}

/// Ports `ggml_backend_cuda_buffer_free_buffer` (ggml-cuda.cu:744 @5266f24da).
fn bufferFreeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    const ctx = bufferCtx(buffer);
    // `~ggml_backend_cuda_buffer_context`.
    cudaCheck(cu.cudaFree(ctx.dev_ptr), "cudaFree(dev_ptr)", @src());
    allocator.destroy(ctx);
}

/// Ports `ggml_backend_buffer_is_cuda` (ggml-cuda.cu:749 @5266f24da).
pub fn bufferIsCuda(buffer: c.ggml_backend_buffer_t) bool {
    return impl.one(c.struct_ggml_backend_buffer, buffer).iface.free_buffer == &bufferFreeBuffer;
}

/// Ports `ggml_backend_cuda_buffer_get_base` (ggml-cuda.cu:753 @5266f24da).
fn bufferGetBase(buffer: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return bufferCtx(buffer).dev_ptr;
}

/// Ports `ggml_backend_cuda_buffer_init_tensor` (ggml-cuda.cu:758 @5266f24da).
fn bufferInitTensor(buffer: c.ggml_backend_buffer_t, tensor_c: [*c]Tensor) callconv(.c) c.enum_ggml_status {
    const ctx = bufferCtx(buffer);
    const tensor = impl.one(Tensor, tensor_c);

    if (tensor.view_src != null) return c.GGML_STATUS_SUCCESS;

    if (c.ggml_is_quantized(tensor.type) and c.ggml_backend_buffer_get_usage(buffer) != c.GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const original_size = c.ggml_nbytes(tensor);
        const padded_size = c.ggml_backend_buft_get_alloc_size(impl.one(c.struct_ggml_backend_buffer, buffer).buft, tensor);

        if (padded_size > original_size) {
            common.setDevice(ctx.device);
            cudaCheck(cu.cudaMemset(@as([*]u8, @ptrCast(tensor.data.?)) + original_size, 0, padded_size - original_size), "cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size)", @src());
        }
    }
    return c.GGML_STATUS_SUCCESS;
}

fn at(p: ?*anyopaque, offset: usize) *anyopaque {
    return @ptrCast(@as([*]u8, @ptrCast(p.?)) + offset);
}

const stream_per_thread: cu.cudaStream_t = @ptrFromInt(0x2);

/// Ports `ggml_backend_cuda_buffer_memset_tensor` (ggml-cuda.cu:779 @5266f24da).
fn bufferMemsetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]Tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    common.setDevice(bufferCtx(buffer).device);
    cudaCheck(cu.cudaMemsetAsync(at(tensor.*.data, offset), value, size, stream_per_thread), "cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread)", @src());
    cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_set_tensor` (ggml-cuda.cu:787 @5266f24da).
fn bufferSetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]Tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    common.setDevice(bufferCtx(buffer).device);
    cudaCheck(cu.cudaMemcpyAsync(at(tensor.*.data, offset), data, size, cu.cudaMemcpyHostToDevice, stream_per_thread), "cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread)", @src());
    cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_get_tensor` (ggml-cuda.cu:795 @5266f24da).
fn bufferGetTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]const Tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    common.setDevice(bufferCtx(buffer).device);
    cudaCheck(cu.cudaMemcpyAsync(data, at(tensor.*.data, offset), size, cu.cudaMemcpyDeviceToHost, stream_per_thread), "cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread)", @src());
    cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_set_tensor_2d` (ggml-cuda.cu:803 @5266f24da).
fn bufferSetTensor2d(buffer: c.ggml_backend_buffer_t, tensor: [*c]Tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    common.setDevice(bufferCtx(buffer).device);
    cudaCheck(cu.cudaMemcpy2DAsync(at(tensor.*.data, offset), stride_tensor, data, stride_data, size, n_copies, cu.cudaMemcpyHostToDevice, stream_per_thread), "cudaMemcpy2DAsync(...)", @src());
    cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_get_tensor_2d` (ggml-cuda.cu:813 @5266f24da).
fn bufferGetTensor2d(buffer: c.ggml_backend_buffer_t, tensor: [*c]const Tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    common.setDevice(bufferCtx(buffer).device);
    cudaCheck(cu.cudaMemcpy2DAsync(data, stride_data, at(tensor.*.data, offset), stride_tensor, size, n_copies, cu.cudaMemcpyDeviceToHost, stream_per_thread), "cudaMemcpy2DAsync(...)", @src());
    cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_cpy_tensor` (ggml-cuda.cu:823 @5266f24da).
fn bufferCpyTensor(buffer: c.ggml_backend_buffer_t, src_c: [*c]const Tensor, dst_c: [*c]Tensor) callconv(.c) bool {
    _ = buffer;
    const src = impl.one(Tensor, @constCast(src_c));
    const dst = impl.one(Tensor, dst_c);
    if (!bufferIsCuda(src.buffer)) return false;
    const src_ctx = bufferCtx(src.buffer);
    const dst_ctx = bufferCtx(dst.buffer);
    // compare the backing physical devices: distinct virtual devices may
    // share one physical GPU, in which case a same-device copy is required
    const src_physical = common.physicalDevice(src_ctx.device);
    const dst_physical = common.physicalDevice(dst_ctx.device);
    if (src_physical == dst_physical) {
        cudaCheck(cu.cudaMemcpyAsync(dst.data, src.data, c.ggml_nbytes(src), cu.cudaMemcpyDeviceToDevice, stream_per_thread), "cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread)", @src());
    } else {
        cudaCheck(cu.cudaMemcpyPeerAsync(dst.data, dst_physical, src.data, src_physical, c.ggml_nbytes(src), stream_per_thread), "cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread)", @src());
    }
    cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
    return true;
}

/// Ports `ggml_backend_cuda_buffer_clear` (ggml-cuda.cu:848 @5266f24da).
fn bufferClear(buffer: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    const ctx = bufferCtx(buffer);
    common.setDevice(ctx.device);
    cudaCheck(cu.cudaMemsetAsync(ctx.dev_ptr, value, impl.one(c.struct_ggml_backend_buffer, buffer).size, stream_per_thread), "cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread)", @src());
    cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_interface` (ggml-cuda.cu:856 @5266f24da).
const buffer_interface = c.ggml_backend_buffer_i{
    .free_buffer = bufferFreeBuffer,
    .get_base = bufferGetBase,
    .init_tensor = bufferInitTensor,
    .memset_tensor = bufferMemsetTensor,
    .set_tensor = bufferSetTensor,
    .get_tensor = bufferGetTensor,
    .set_tensor_2d = bufferSetTensor2d,
    .get_tensor_2d = bufferGetTensor2d,
    .cpy_tensor = bufferCpyTensor,
    .clear = bufferClear,
    .reset = null,
};

// -----------------------------------------------------------------------------
// Device buffer type

/// Ports `ggml_backend_cuda_buffer_type_context` (ggml-cuda.cu:871 @5266f24da).
pub const BufferTypeCtx = struct {
    device: c_int,
    name: [16:0]u8 = @splat(0),
};

/// Ports `ggml_backend_cuda_buffer_type_get_name` (ggml-cuda.cu:876 @5266f24da).
fn bufferTypeGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    const ctx: *const BufferTypeCtx = @ptrCast(@alignCast(impl.one(c.struct_ggml_backend_buffer_type, buft).context.?));
    return &ctx.name;
}

/// Ports `ggml_backend_buft_is_cuda` (ggml-cuda.cu:882 @5266f24da).
pub fn buftIsCuda(buft: c.ggml_backend_buffer_type_t) bool {
    return impl.one(c.struct_ggml_backend_buffer_type, buft).iface.get_name == &bufferTypeGetName;
}

/// The device a CUDA buffer type allocates on.
pub fn buftDevice(buft: c.ggml_backend_buffer_type_t) c_int {
    const ctx: *const BufferTypeCtx = @ptrCast(@alignCast(impl.one(c.struct_ggml_backend_buffer_type, buft).context.?));
    return ctx.device;
}

/// Ports `ggml_backend_cuda_buffer_type_alloc_buffer` (ggml-cuda.cu:886 @5266f24da).
fn bufferTypeAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const device = buftDevice(buft);
    common.setDevice(device);

    var dev_ptr: ?*anyopaque = null;
    const err = common.deviceMalloc(&dev_ptr, size, device);
    if (err != cu.cudaSuccess) {
        // clear the error
        _ = cu.cudaGetLastError();
        impl.logError("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", .{ "ggml_backend_cuda_buffer_type_alloc_buffer", @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0, device, cu.cudaGetErrorString(err) });
        return null;
    }

    const ctx = allocator.create(BufferCtx) catch impl.abort("ggml-cuda: out of memory");
    ctx.* = .{ .device = device, .dev_ptr = dev_ptr };
    _ = std.fmt.bufPrintSentinel(&ctx.name, common.name ++ "{d}", .{device}, 0) catch unreachable;

    return c.ggml_backend_buffer_init(buft, buffer_interface, ctx, size);
}

/// Ports `ggml_backend_cuda_buffer_type_get_alignment` (ggml-cuda.cu:905 @5266f24da).
fn bufferTypeGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return 128;
}

/// Ports `ggml_backend_cuda_buffer_type_get_alloc_size` (ggml-cuda.cu:911 @5266f24da).
fn bufferTypeGetAllocSize(buft: c.ggml_backend_buffer_type_t, tensor_c: [*c]const Tensor) callconv(.c) usize {
    const tensor = impl.one(Tensor, @constCast(tensor_c));
    var size = if (tensor.op == c.GGML_OP_FLASH_ATTN_EXT)
        fattn.getAllocSize(buftDevice(buft), tensor)
    else
        c.ggml_nbytes(tensor);
    const ne0 = tensor.ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (c.ggml_is_quantized(tensor.type)) {
        if (@mod(ne0, common.matrix_row_padding) != 0) {
            impl.assert(tensor.nb[0] == c.ggml_element_size(tensor), "tensor->nb[0] == ggml_element_size(tensor)");
            size += c.ggml_row_size(tensor.type, common.matrix_row_padding - @mod(ne0, common.matrix_row_padding));
        }
    }

    return size;
}

/// Ports `ggml_backend_cuda_buffer_type_interface` (ggml-cuda.cu:930 @5266f24da).
const buffer_type_interface = c.struct_ggml_backend_buffer_type_i{
    .get_name = bufferTypeGetName,
    .alloc_buffer = bufferTypeAllocBuffer,
    .get_alignment = bufferTypeGetAlignment,
    .get_max_size = null,
    .get_alloc_size = bufferTypeGetAllocSize,
    .is_host = null,
};

const BufferTypes = struct {
    var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
    var types: [common.max_devices]c.struct_ggml_backend_buffer_type = undefined;
    var ctxs: [common.max_devices]BufferTypeCtx = undefined;
    var initialized = false;
};

/// Ports `ggml_backend_cuda_buffer_type` (ggml-cuda.cu:939 @5266f24da).
pub fn bufferType(device: c_int) callconv(.c) c.ggml_backend_buffer_type_t {
    _ = std.c.pthread_mutex_lock(&BufferTypes.mutex);
    defer _ = std.c.pthread_mutex_unlock(&BufferTypes.mutex);

    if (device >= getDeviceCount()) return null;

    if (!BufferTypes.initialized) {
        var i: c_int = 0;
        while (i < getDeviceCount()) : (i += 1) {
            const u: usize = @intCast(i);
            BufferTypes.ctxs[u] = .{ .device = i };
            _ = std.fmt.bufPrintSentinel(&BufferTypes.ctxs[u].name, common.name ++ "{d}", .{i}, 0) catch unreachable;
            BufferTypes.types[u] = .{
                .iface = buffer_type_interface,
                .device = c.ggml_backend_reg_dev_get(reg(), u),
                .context = &BufferTypes.ctxs[u],
            };
        }
        BufferTypes.initialized = true;
    }

    return &BufferTypes.types[@intCast(device)];
}

// -----------------------------------------------------------------------------
// Tensor-parallel communication

/// Ports `ggml_backend_cuda_comm_context` (ggml-cuda.cu:971 @5266f24da),
/// without the NCCL member: `GGML_USE_NCCL` is not defined.
const CommContext = struct {
    const TryAllreduce = *const fn (*CommContext, [*c][*c]Tensor) bool;

    backends: []c.ggml_backend_t,
    dev_ids: []c_int,
    try_allreduce: ?TryAllreduce = null,
    ar_pipeline: ?*anyopaque = null,

    fn destroy(self: *CommContext) void {
        arPipelineFree(self.ar_pipeline);
        allocator.free(self.backends);
        allocator.free(self.dev_ids);
        allocator.destroy(self);
    }
};

/// `ggml_cuda_ar_pipeline_init` (allreduce.cu), not ported yet: reports
/// failure, so the init chain falls back to the butterfly as the C's does
/// when the pipeline cannot start.
fn arPipelineInit(dev_ids: []const c_int) ?*anyopaque {
    _ = dev_ids;
    return null;
}

/// `ggml_cuda_ar_pipeline_free` (allreduce.cu); null-tolerant, as the C's is.
fn arPipelineFree(p: ?*anyopaque) void {
    std.debug.assert(p == null);
}

/// Ports `ggml_backend_cuda_comm_try_allreduce_butterfly` (ggml-cuda.cu:1144 @5266f24da).
fn commTryAllreduceButterfly(_: *CommContext, _: [*c][*c]Tensor) bool {
    return false;
}

/// Ports `ggml_backend_cuda_comm_free` (ggml-cuda.cu:1149 @5266f24da).
fn commFree(comm_ctx_v: ?*anyopaque) callconv(.c) void {
    const ctx: *CommContext = @ptrCast(@alignCast(comm_ctx_v orelse return));
    ctx.destroy();
}

/// Ports `ggml_backend_cuda_comm_init_none` (ggml-cuda.cu:1160 @5266f24da).
fn commInitNone(ret: *CommContext) void {
    ret.try_allreduce = commTryAllreduceButterfly;
}

/// Ports `ggml_backend_cuda_comm_init_internal` (ggml-cuda.cu:1164 @5266f24da).
fn commInitInternal(ret: *CommContext) void {
    ret.ar_pipeline = arPipelineInit(ret.dev_ids);
    if (ret.ar_pipeline != null) {
        unreachable; // the internal pipeline is not ported; see the file comment
    }

    // Clear sticky CUDA error from the failed init.
    _ = cu.cudaGetLastError();
    impl.logWarn("internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly\n", .{});
    commInitNone(ret);
}

/// Ports `ggml_backend_cuda_comm_init_nccl` (ggml-cuda.cu:1178 @5266f24da),
/// the arm without `GGML_USE_NCCL`.
fn commInitNccl(ret: *CommContext) void {
    impl.logWarn("NCCL not compiled in; falling back to internal AllReduce.  Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n", .{});
    commInitInternal(ret);
}

/// Ports `ggml_backend_cuda_comm_init` (ggml-cuda.cu:1214 @5266f24da).
fn commInit(backends: [*c]c.ggml_backend_t, n_backends: usize) callconv(.c) ?*anyopaque {
    for (0..n_backends) |i| {
        if (!isCuda(backends[i])) return null;
    }

    const ret = allocator.create(CommContext) catch impl.abort("ggml-cuda: out of memory");
    ret.* = .{
        .backends = allocator.dupe(c.ggml_backend_t, backends[0..n_backends]) catch impl.abort("ggml-cuda: out of memory"),
        .dev_ids = allocator.alloc(c_int, n_backends) catch impl.abort("ggml-cuda: out of memory"),
    };
    for (0..n_backends) |i| ret.dev_ids[i] = backendCtx(backends[i]).device;

    if (std.c.getenv("GGML_CUDA_ALLREDUCE")) |env_c| {
        const env = std.mem.span(env_c);
        if (std.mem.eql(u8, env, "nccl")) {
            commInitNccl(ret);
        } else if (std.mem.eql(u8, env, "internal")) {
            commInitInternal(ret);
        } else if (std.mem.eql(u8, env, "none")) {
            commInitNone(ret);
        } else {
            impl.logWarn("unknown GGML_CUDA_ALLREDUCE value: %s\n", .{env_c});
            commInitNone(ret);
        }
    } else {
        // Platform default: Linux uses NCCL, otherwise internal.
        if (@import("builtin").os.tag == .linux) commInitNccl(ret) else commInitInternal(ret);
    }

    return ret;
}

/// Ports `ggml_backend_cuda_comm_allreduce_tensor` (ggml-cuda.cu:1255 @5266f24da).
fn commAllreduceTensor(comm_ctx_v: ?*anyopaque, tensors: [*c][*c]Tensor) callconv(.c) bool {
    const ctx: *CommContext = @ptrCast(@alignCast(comm_ctx_v orelse return false));
    return ctx.try_allreduce.?(ctx, tensors);
}

// -----------------------------------------------------------------------------
// Pinned host buffer type

/// Ports `ggml_backend_cuda_host_buffer_type_name` (ggml-cuda.cu:1265 @5266f24da).
fn hostBufferTypeName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    _ = buft;
    return common.name ++ "_Host";
}

/// Ports `ggml_backend_buft_is_cuda_host` (ggml-cuda.cu:1271 @5266f24da).
pub fn buftIsCudaHost(buft: c.ggml_backend_buffer_type_t) bool {
    return impl.one(c.struct_ggml_backend_buffer_type, buft).iface.get_name == &hostBufferTypeName;
}

/// Ports `ggml_backend_cuda_host_buffer_free_buffer` (ggml-cuda.cu:1275 @5266f24da).
fn hostBufferFreeBuffer(buffer: c.ggml_backend_buffer_t) callconv(.c) void {
    cudaCheck(cu.cudaFreeHost(impl.one(c.struct_ggml_backend_buffer, buffer).context), "cudaFreeHost(buffer->context)", @src());
}

/// Ports `ggml_cuda_host_malloc` (ggml-cuda.cu:1279 @5266f24da).
fn hostMalloc(size: usize) ?*anyopaque {
    if (std.c.getenv("GGML_CUDA_NO_PINNED") != null) return null;

    var ptr: ?*anyopaque = null;
    const err = cu.cudaMallocHost(&ptr, size);
    if (err != cu.cudaSuccess) {
        // clear the error
        _ = cu.cudaGetLastError();
        impl.logDebug("%s: failed to allocate %.2f MiB of pinned memory: %s\n", .{ "ggml_cuda_host_malloc", @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0, cu.cudaGetErrorString(err) });
        return null;
    }
    return ptr;
}

/// Ports `ggml_backend_cuda_host_buffer_type_alloc_buffer` (ggml-cuda.cu:1297 @5266f24da).
fn hostBufferTypeAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const ptr = hostMalloc(size) orelse {
        // fallback to cpu buffer
        return c.ggml_backend_buft_alloc_buffer(c.ggml_backend_cpu_buffer_type(), size);
    };

    const buffer = c.ggml_backend_cpu_buffer_from_ptr(ptr, size);
    const b = impl.one(c.struct_ggml_backend_buffer, buffer);
    b.buft = buft;
    b.iface.free_buffer = hostBufferFreeBuffer;
    return buffer;
}

const HostBufferType = struct {
    var once_value: c.struct_ggml_backend_buffer_type = undefined;
    fn run() void {
        const cpu = impl.one(c.struct_ggml_backend_buffer_type, c.ggml_backend_cpu_buffer_type());
        once_value = .{
            .iface = .{
                .get_name = hostBufferTypeName,
                .alloc_buffer = hostBufferTypeAllocBuffer,
                .get_alignment = cpu.iface.get_alignment,
                .get_max_size = null,
                .get_alloc_size = cpu.iface.get_alloc_size,
                .is_host = cpu.iface.is_host,
            },
            .device = c.ggml_backend_reg_dev_get(reg(), 0),
            .context = null,
        };
    }
    const once = common.Once(run);
};

/// Ports `ggml_backend_cuda_host_buffer_type` (ggml-cuda.cu:1312 @5266f24da).
pub fn hostBufferType() callconv(.c) c.ggml_backend_buffer_type_t {
    HostBufferType.once.call();
    return &HostBufferType.once_value;
}

// -----------------------------------------------------------------------------
// Backend

pub fn backendCtx(backend: c.ggml_backend_t) *common.Context {
    return @ptrCast(@alignCast(impl.one(c.struct_ggml_backend, backend).context.?));
}

/// Ports `ggml_backend_cuda_get_name` (ggml-cuda.cu:2427 @5266f24da).
fn backendGetName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return backendCtx(backend).getName();
}

/// Ports `ggml_backend_cuda_free` (ggml-cuda.cu:2433 @5266f24da).
fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    backendCtx(backend).destroy();
    allocator.destroy(impl.one(c.struct_ggml_backend, backend));
}

fn bufferOf(tensor: *const Tensor) c.ggml_backend_buffer_t {
    return if (tensor.view_src) |v| v.*.buffer else tensor.buffer;
}

fn assertOwnBuffer(cuda_ctx: *common.Context, tensor: *const Tensor) void {
    const buf = impl.one(c.struct_ggml_backend_buffer, bufferOf(tensor));
    impl.assert(buf.buft == bufferType(cuda_ctx.device), "buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && \"unsupported buffer type\"");
}

/// Ports `ggml_backend_cuda_set_tensor_async` (ggml-cuda.cu:2440 @5266f24da).
fn backendSetTensorAsync(backend: c.ggml_backend_t, tensor: [*c]Tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, impl.one(Tensor, tensor));
    cudaCheck(cu.cudaMemcpyAsync(at(tensor.*.data, offset), data, size, cu.cudaMemcpyHostToDevice, cuda_ctx.stream()), "cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_get_tensor_async` (ggml-cuda.cu:2449 @5266f24da).
fn backendGetTensorAsync(backend: c.ggml_backend_t, tensor: [*c]const Tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, impl.one(Tensor, @constCast(tensor)));
    cudaCheck(cu.cudaMemcpyAsync(data, at(tensor.*.data, offset), size, cu.cudaMemcpyDeviceToHost, cuda_ctx.stream()), "cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_set_tensor_2d_async` (ggml-cuda.cu:2458 @5266f24da).
fn backendSetTensor2dAsync(backend: c.ggml_backend_t, tensor: [*c]Tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, impl.one(Tensor, tensor));
    cudaCheck(cu.cudaMemcpy2DAsync(at(tensor.*.data, offset), stride_tensor, data, stride_data, size, n_copies, cu.cudaMemcpyHostToDevice, cuda_ctx.stream()), "cudaMemcpy2DAsync(...)", @src());
}

/// Ports `ggml_backend_cuda_get_tensor_2d_async` (ggml-cuda.cu:2469 @5266f24da).
fn backendGetTensor2dAsync(backend: c.ggml_backend_t, tensor: [*c]const Tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, impl.one(Tensor, @constCast(tensor)));
    cudaCheck(cu.cudaMemcpy2DAsync(data, stride_data, at(tensor.*.data, offset), stride_tensor, size, n_copies, cu.cudaMemcpyDeviceToHost, cuda_ctx.stream()), "cudaMemcpy2DAsync(...)", @src());
}

/// Ports `ggml_backend_cuda_cpy_tensor_async` (ggml-cuda.cu:2480 @5266f24da).
fn backendCpyTensorAsync(backend_src: c.ggml_backend_t, backend_dst: c.ggml_backend_t, src_c: [*c]const Tensor, dst_c: [*c]Tensor) callconv(.c) bool {
    const src = impl.one(Tensor, @constCast(src_c));
    const dst = impl.one(Tensor, dst_c);
    const buf_src = bufferOf(src);
    const buf_dst = bufferOf(dst);

    if (!isCuda(backend_src) or !isCuda(backend_dst)) return false;
    if (!bufferIsCuda(buf_src) or !bufferIsCuda(buf_dst)) return false;

    // device -> device copy
    const cuda_ctx_src = backendCtx(backend_src);
    const cuda_ctx_dst = backendCtx(backend_dst);
    const buf_ctx_src = bufferCtx(buf_src);
    const buf_ctx_dst = bufferCtx(buf_dst);

    if (cuda_ctx_src.device != buf_ctx_src.device or cuda_ctx_dst.device != buf_ctx_dst.device) {
        // `#ifndef NDEBUG` logging: NDEBUG is defined.
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        const src_physical = common.physicalDevice(cuda_ctx_src.device);
        const dst_physical = common.physicalDevice(cuda_ctx_dst.device);
        if (src_physical == dst_physical) {
            cudaCheck(cu.cudaMemcpyAsync(dst.data, src.data, c.ggml_nbytes(dst), cu.cudaMemcpyDeviceToDevice, cuda_ctx_src.stream()), "cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream())", @src());
        } else {
            cudaCheck(cu.cudaMemcpyPeerAsync(dst.data, dst_physical, src.data, src_physical, c.ggml_nbytes(dst), cuda_ctx_src.stream()), "cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream())", @src());
        }

        // record event on src stream after the copy
        if (cuda_ctx_src.copy_event == null) {
            common.setDevice(cuda_ctx_src.device);
            cudaCheck(cu.cudaEventCreateWithFlags(&cuda_ctx_src.copy_event, cu.cudaEventDisableTiming), "cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming)", @src());
        }

        cudaCheck(cu.cudaEventRecord(cuda_ctx_src.copy_event, cuda_ctx_src.stream()), "cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream())", @src());

        // wait on dst stream for the copy to complete
        cudaCheck(cu.cudaStreamWaitEvent(cuda_ctx_dst.stream(), cuda_ctx_src.copy_event, 0), "cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0)", @src());
    } else {
        // src and dst are on the same backend
        cudaCheck(cu.cudaMemcpyAsync(dst.data, src.data, c.ggml_nbytes(dst), cu.cudaMemcpyDeviceToDevice, cuda_ctx_src.stream()), "cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream())", @src());
    }
    return true;
}

/// Ports `ggml_backend_cuda_synchronize` (ggml-cuda.cu:2539 @5266f24da).
fn backendSynchronize(backend: c.ggml_backend_t) callconv(.c) void {
    cudaCheck(cu.cudaStreamSynchronize(backendCtx(backend).stream()), "cudaStreamSynchronize(cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_event_record` (ggml-cuda.cu:4474 @5266f24da).
fn backendEventRecord(backend: c.ggml_backend_t, event: c.ggml_backend_event_t) callconv(.c) void {
    cudaCheck(cu.cudaEventRecord(@ptrCast(impl.one(c.struct_ggml_backend_event, event).context), backendCtx(backend).stream()), "cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_event_wait` (ggml-cuda.cu:4480 @5266f24da).
fn backendEventWait(backend: c.ggml_backend_t, event: c.ggml_backend_event_t) callconv(.c) void {
    if (isCuda(backend)) {
        cudaCheck(cu.cudaStreamWaitEvent(backendCtx(backend).stream(), @ptrCast(impl.one(c.struct_ggml_backend_event, event).context), 0), "cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0)", @src());
    } else {
        impl.abort("fatal error");
    }
}

/// Ports `ggml_backend_cuda_interface` (ggml-cuda.cu:4767 @5266f24da).
const backend_interface = c.struct_ggml_backend_i{
    .get_name = backendGetName,
    .free = backendFree,
    .set_tensor_async = backendSetTensorAsync,
    .get_tensor_async = backendGetTensorAsync,
    .set_tensor_2d_async = backendSetTensor2dAsync,
    .get_tensor_2d_async = backendGetTensor2dAsync,
    .cpy_tensor_async = backendCpyTensorAsync,
    .synchronize = backendSynchronize,
    .graph_plan_create = null,
    .graph_plan_free = null,
    .graph_plan_update = null,
    .graph_plan_compute = null,
    .graph_compute = graph.graphCompute,
    .event_record = backendEventRecord,
    .event_wait = backendEventWait,
    .graph_optimize = graph.graphOptimize,
};

/// Ports `ggml_backend_cuda_guid` (ggml-cuda.cu:4786 @5266f24da).
var guid: c.ggml_guid = .{ 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };

/// Ports `ggml_backend_is_cuda` (ggml-cuda.cu:4791 @5266f24da).
pub fn isCuda(backend: c.ggml_backend_t) callconv(.c) bool {
    return backend != null and c.ggml_guid_matches(impl.one(c.struct_ggml_backend, backend).guid, &guid);
}

/// Ports `ggml_backend_cuda_get_device_count` (ggml-cuda.cu:4795 @5266f24da).
pub fn getDeviceCount() callconv(.c) c_int {
    return common.info().device_count;
}

/// Ports `ggml_cuda_device_description` (ggml-cuda.cu:4799 @5266f24da).
fn deviceDescription(device: c_int, out: []u8) [:0]u8 {
    var prop: cu.cudaDeviceProp = undefined;
    cudaCheck(cu.cudaGetDeviceProperties(&prop, common.physicalDevice(device)), "cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device))", @src());

    const inf = common.info();
    const prop_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&prop.name)), 0);
    if (inf.device_count > inf.physical_device_count) {
        const d = inf.devices[@intCast(device)];
        return std.fmt.bufPrintSentinel(out, "{s} (dev p{d}/v{d})", .{ prop_name, d.physical_device, d.virtual_index }, 0) catch unreachable;
    }
    return std.fmt.bufPrintSentinel(out, "{s}", .{prop_name}, 0) catch unreachable;
}

/// Ports `ggml_backend_cuda_get_device_description` (ggml-cuda.cu:4812 @5266f24da).
pub fn getDeviceDescription(device: c_int, description: [*c]u8, description_size: usize) callconv(.c) void {
    var buf: [320]u8 = undefined;
    const d = deviceDescription(device, &buf);
    _ = snprintf(description, description_size, "%s", d.ptr);
}

extern fn snprintf(buf: [*c]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

/// Ports `ggml_cuda_physical_device_share_count` (ggml-cuda.cu:4816 @5266f24da).
fn physicalDeviceShareCount(device: c_int) c_int {
    const inf = common.info();
    impl.assert(device >= 0 and device < inf.device_count, "device >= 0 && device < info.device_count");
    return inf.devices[@intCast(device)].physical_share_count;
}

/// Ports `ggml_backend_cuda_get_device_memory` (ggml-cuda.cu:4822 @5266f24da).
pub fn getDeviceMemory(device: c_int, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    common.setDevice(device);
    cudaCheck(cu.cudaMemGetInfo(free, total), "cudaMemGetInfo(free, total)", @src());
    // virtual devices sharing one physical GPU share its memory pool
    const share_count: usize = @intCast(physicalDeviceShareCount(device));
    free.* /= share_count;
    total.* /= share_count;
}

/// Ports `ggml_backend_cuda_register_host_buffer` (ggml-cuda.cu:4833 @5266f24da).
pub fn registerHostBuffer(buffer: ?*anyopaque, size: usize) callconv(.c) bool {
    if (std.c.getenv("GGML_CUDA_REGISTER_HOST") == null) return false;

    const err = cu.cudaHostRegister(buffer, size, cu.cudaHostRegisterPortable | cu.cudaHostRegisterReadOnly);
    if (err != cu.cudaSuccess) {
        // clear the error
        _ = cu.cudaGetLastError();
        impl.logDebug("%s: failed to register %.2f MiB of pinned memory: %s\n", .{ "ggml_backend_cuda_register_host_buffer", @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0, cu.cudaGetErrorString(err) });
        return false;
    }
    return true;
}

/// Ports `ggml_backend_cuda_unregister_host_buffer` (ggml-cuda.cu:4856 @5266f24da).
pub fn unregisterHostBuffer(buffer: ?*anyopaque) callconv(.c) void {
    if (std.c.getenv("GGML_CUDA_REGISTER_HOST") == null) return;
    const err = cu.cudaHostUnregister(buffer);
    if (err != cu.cudaSuccess) {
        // clear the error
        _ = cu.cudaGetLastError();
    }
}

// -----------------------------------------------------------------------------
// Device

/// Ports `ggml_backend_cuda_device_context` (ggml-cuda.cu:4871 @5266f24da).
pub const DeviceCtx = struct {
    device: c_int,
    name: [16:0]u8 = @splat(0),
    description: [320:0]u8 = @splat(0),
    pci_bus_id: [48:0]u8 = @splat(0),
    op_offload_min_batch_size: c_int,
};

pub fn deviceCtx(dev: c.ggml_backend_dev_t) *DeviceCtx {
    return @ptrCast(@alignCast(impl.one(c.struct_ggml_backend_device, dev).context.?));
}

/// Ports `ggml_backend_cuda_device_get_name` (ggml-cuda.cu:4879 @5266f24da).
fn deviceGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return &deviceCtx(dev).name;
}

/// Ports `ggml_backend_cuda_device_get_description` (ggml-cuda.cu:4884 @5266f24da).
fn deviceGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return &deviceCtx(dev).description;
}

/// Ports `ggml_backend_cuda_get_available_uma_memory` (ggml-cuda.cu:4891 @5266f24da).
fn getAvailableUmaMemory(available_memory_kb: *c_long, free_swap_kb: *c_long) bool {
    const buffer_size = 2048;
    var file_buffer: [buffer_size]u8 = undefined;
    var huge_tlb_total_pages: c_long = -1;
    var huge_tlb_free_pages: c_long = -1;
    var huge_tlb_page_size: c_long = -1;

    const meminfo_file = std.c.fopen("/proc/meminfo", "r") orelse {
        impl.logError("%s: failed to open /proc/meminfo\n", .{"ggml_backend_cuda_get_available_uma_memory"});
        return false;
    };

    // Read file into buffer
    const bytes_read = std.c.fread(&file_buffer, 1, buffer_size - 1, meminfo_file);
    _ = std.c.fclose(meminfo_file);

    if (bytes_read == 0) {
        impl.logError("%s: failed to read from /proc/meminfo\n", .{"ggml_backend_cuda_get_available_uma_memory"});
        return false;
    }
    file_buffer[bytes_read] = 0;

    available_memory_kb.* = -1;
    free_swap_kb.* = -1;

    // Parse the file buffer line by line; each `sscanf` pattern matches a
    // fixed prefix, optional whitespace and a decimal.
    var lines = std.mem.splitScalar(u8, file_buffer[0..bytes_read], '\n');
    while (lines.next()) |line| {
        if (scanField(line, "MemAvailable:", " kB")) |v| {
            available_memory_kb.* = v;
        } else if (scanField(line, "SwapFree:", " kB")) |v| {
            free_swap_kb.* = v;
        } else if (scanField(line, "HugePages_Total:", "")) |v| {
            huge_tlb_total_pages = v;
        } else if (scanField(line, "HugePages_Free:", "")) |v| {
            huge_tlb_free_pages = v;
        } else if (scanField(line, "Hugepagesize:", " kB")) |v| {
            huge_tlb_page_size = v;
        }
    }

    if (huge_tlb_total_pages != 0 and huge_tlb_total_pages != -1) {
        available_memory_kb.* = huge_tlb_free_pages * huge_tlb_page_size;
        // Hugetlbfs pages are not swappable.
        free_swap_kb.* = 0;
    }

    impl.logDebug("%s: final available_memory_kb: %ld\n", .{ "ggml_backend_cuda_get_available_uma_memory", available_memory_kb.* });
    return true;
}

/// `sscanf(line, "<prefix> %ld<suffix>", &value) == 1`: the conversion is
/// what counts, so a trailing suffix need not match.
fn scanField(line: []const u8, prefix: []const u8, suffix: []const u8) ?c_long {
    _ = suffix;
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    var rest = std.mem.trimStart(u8, line[prefix.len..], " \t");
    var end: usize = 0;
    if (end < rest.len and (rest[end] == '-' or rest[end] == '+')) end += 1;
    const digits_start = end;
    while (end < rest.len and std.ascii.isDigit(rest[end])) end += 1;
    if (end == digits_start) return null;
    rest = rest[0..end];
    return std.fmt.parseInt(c_long, rest, 10) catch null;
}

/// Ports `ggml_backend_cuda_device_get_memory` (ggml-cuda.cu:4965 @5266f24da).
fn deviceGetMemory(dev: c.ggml_backend_dev_t, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    const ctx = deviceCtx(dev);
    common.setDevice(ctx.device);
    const err = cu.cudaMemGetInfo(free, total);
    if (err != cu.cudaSuccess) {
        _ = cu.cudaGetLastError();
        impl.logWarn("%s: cudaMemGetInfo failed (%s), returning 0/0\n", .{ "ggml_backend_cuda_device_get_memory", cu.cudaGetErrorString(err) });
        free.* = 0;
        total.* = 0;
        return;
    }

    // Check if this is a UMA (Unified Memory Architecture) system
    var prop: cu.cudaDeviceProp = undefined;
    cudaCheck(cu.cudaGetDeviceProperties(&prop, common.physicalDevice(ctx.device)), "cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device))", @src());

    const uma_env = std.c.getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != null;
    const is_uma = prop.integrated > 0 or uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        var available_memory_kb: c_long = 0;
        var free_swap_kb: c_long = 0;
        if (getAvailableUmaMemory(&available_memory_kb, &free_swap_kb) and available_memory_kb > 0) {
            free.* = @as(usize, @intCast(available_memory_kb)) * 1024;
        } else {
            impl.logError("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", .{"ggml_backend_cuda_device_get_memory"});
        }
    }

    // virtual devices sharing one physical GPU share its memory pool
    const share_count: usize = @intCast(physicalDeviceShareCount(ctx.device));
    free.* /= share_count;
    total.* /= share_count;
}

/// Ports `ggml_backend_cuda_device_get_type` (ggml-cuda.cu:5006 @5266f24da).
fn deviceGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    var prop: cu.cudaDeviceProp = undefined;
    cudaCheck(cu.cudaGetDeviceProperties(&prop, common.physicalDevice(deviceCtx(dev).device)), "cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device))", @src());
    return if (prop.integrated != 0) c.GGML_BACKEND_DEVICE_TYPE_IGPU else c.GGML_BACKEND_DEVICE_TYPE_GPU;
}

/// Ports `ggml_backend_cuda_device_get_props` (ggml-cuda.cu:5017 @5266f24da).
fn deviceGetProps(dev: c.ggml_backend_dev_t, props_c: [*c]c.struct_ggml_backend_dev_props) callconv(.c) void {
    const ctx = deviceCtx(dev);
    const props = impl.one(c.struct_ggml_backend_dev_props, props_c);

    props.name = deviceGetName(dev);
    props.description = deviceGetDescription(dev);
    props.type = deviceGetType(dev);
    props.device_id = if (ctx.pci_bus_id[0] == 0) null else &ctx.pci_bus_id;
    deviceGetMemory(dev, &props.memory_free, &props.memory_total);

    const host_buffer = std.c.getenv("GGML_CUDA_NO_PINNED") == null;
    props.caps = .{
        .async = true,
        .host_buffer = host_buffer,
        .buffer_from_host_ptr = false,
        .events = true,
        .mmap_support = props.type != c.GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

/// Ports `ggml_backend_cuda_device_init_backend` (ggml-cuda.cu:5042 @5266f24da).
fn deviceInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = params;
    return init(deviceCtx(dev).device);
}

/// Ports `ggml_backend_cuda_device_get_buffer_type` (ggml-cuda.cu:5048 @5266f24da).
fn deviceGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return bufferType(deviceCtx(dev).device);
}

/// Ports `ggml_backend_cuda_device_get_host_buffer_type` (ggml-cuda.cu:5053 @5266f24da).
fn deviceGetHostBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    _ = dev;
    return hostBufferType();
}

/// Ports `ggml_backend_cuda_device_supports_buft` (ggml-cuda.cu:5518 @5266f24da).
fn deviceSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    const integrated = common.info().devices[@intCast(deviceCtx(dev).device)].integrated;
    return (buftIsCuda(buft) and impl.one(c.struct_ggml_backend_buffer_type, buft).device == dev) or (integrated and buftIsCudaHost(buft));
}

/// Ports `get_op_batch_size` (ggml-cuda.cu:5524 @5266f24da).
fn getOpBatchSize(op: *const Tensor) i64 {
    return switch (op.op) {
        c.GGML_OP_GET_ROWS => 0,
        c.GGML_OP_MUL_MAT => op.ne[1],
        c.GGML_OP_MUL_MAT_ID, c.GGML_OP_ROPE, c.GGML_OP_ROPE_BACK => op.ne[2],
        else => c.ggml_nrows(op),
    };
}

/// Ports `ggml_backend_cuda_device_offload_op` (ggml-cuda.cu:5539 @5266f24da).
fn deviceOffloadOp(dev: c.ggml_backend_dev_t, op: [*c]const Tensor) callconv(.c) bool {
    return getOpBatchSize(impl.one(Tensor, @constCast(op))) >= deviceCtx(dev).op_offload_min_batch_size;
}

/// Ports `ggml_backend_cuda_device_event_new` (ggml-cuda.cu:5545 @5266f24da).
fn deviceEventNew(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_event_t {
    common.setDevice(deviceCtx(dev).device);
    var event: cu.cudaEvent_t = null;
    cudaCheck(cu.cudaEventCreateWithFlags(&event, cu.cudaEventDisableTiming), "cudaEventCreateWithFlags(&event, cudaEventDisableTiming)", @src());
    const e = allocator.create(c.struct_ggml_backend_event) catch impl.abort("ggml-cuda: out of memory");
    e.* = .{ .device = dev, .context = event };
    return e;
}

/// Ports `ggml_backend_cuda_device_event_free` (ggml-cuda.cu:5564 @5266f24da).
fn deviceEventFree(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    _ = dev;
    const e = impl.one(c.struct_ggml_backend_event, event);
    cudaCheck(cu.cudaEventDestroy(@ptrCast(e.context)), "cudaEventDestroy((cudaEvent_t)event->context)", @src());
    allocator.destroy(e);
}

/// Ports `ggml_backend_cuda_device_event_synchronize` (ggml-cuda.cu:5571 @5266f24da).
fn deviceEventSynchronize(dev: c.ggml_backend_dev_t, event: c.ggml_backend_event_t) callconv(.c) void {
    _ = dev;
    cudaCheck(cu.cudaEventSynchronize(@ptrCast(impl.one(c.struct_ggml_backend_event, event).context)), "cudaEventSynchronize((cudaEvent_t)event->context)", @src());
}

/// Ports `ggml_backend_cuda_device_interface` (ggml-cuda.cu:5576 @5266f24da).
const device_interface = c.struct_ggml_backend_device_i{
    .get_name = deviceGetName,
    .get_description = deviceGetDescription,
    .get_memory = deviceGetMemory,
    .get_type = deviceGetType,
    .get_props = deviceGetProps,
    .init_backend = deviceInitBackend,
    .get_buffer_type = deviceGetBufferType,
    .get_host_buffer_type = deviceGetHostBufferType,
    .buffer_from_host_ptr = null,
    .supports_op = supports.supportsOp,
    .supports_buft = deviceSupportsBuft,
    .offload_op = deviceOffloadOp,
    .event_new = deviceEventNew,
    .event_free = deviceEventFree,
    .event_synchronize = deviceEventSynchronize,
};

// -----------------------------------------------------------------------------
// Registry

/// Ports `ggml_backend_cuda_reg_context` (ggml-cuda.cu:5596 @5266f24da).
const RegCtx = struct {
    devices: [common.max_devices]c.ggml_backend_dev_t = @splat(null),
    count: usize = 0,
};

/// Ports `ggml_backend_cuda_reg_get_name` (ggml-cuda.cu:5600 @5266f24da).
fn regGetName(r: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = r;
    return common.name;
}

fn regCtx(r: c.ggml_backend_reg_t) *RegCtx {
    return @ptrCast(@alignCast(impl.one(c.struct_ggml_backend_reg, r).context.?));
}

/// Ports `ggml_backend_cuda_reg_get_device_count` (ggml-cuda.cu:5605 @5266f24da).
fn regGetDeviceCount(r: c.ggml_backend_reg_t) callconv(.c) usize {
    return regCtx(r).count;
}

/// Ports `ggml_backend_cuda_reg_get_device` (ggml-cuda.cu:5610 @5266f24da).
fn regGetDevice(r: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    const ctx = regCtx(r);
    impl.assert(index < ctx.count, "index < ctx->devices.size()");
    return ctx.devices[index];
}

const Features = struct {
    var list: [10]c.struct_ggml_backend_feature = undefined;
    var arch_list: [8:0]u8 = @splat(0);
    fn run() void {
        var n: usize = 0;
        // `STRINGIFY(__CUDA_ARCH_LIST__)`.
        _ = std.fmt.bufPrintSentinel(&arch_list, "{d}", .{common.highestCompiledArch(std.math.maxInt(c_int))}, 0) catch unreachable;
        list[n] = .{ .name = "ARCHS", .value = &arch_list };
        n += 1;
        list[n] = .{ .name = "USE_GRAPHS", .value = "1" };
        n += 1;
        list[n] = .{ .name = "FA_ALL_QUANTS", .value = "1" };
        n += 1;
        const inf = common.info();
        for (0..@intCast(inf.device_count)) |id| {
            if (common.blackwellMmaAvailable(inf.devices[id].cc)) {
                list[n] = .{ .name = "BLACKWELL_NATIVE_FP4", .value = "1" };
                n += 1;
                break;
            }
        }
        list[n] = .{ .name = null, .value = null };
    }
    const once = common.Once(run);
};

/// Ports `ggml_backend_cuda_get_features` (ggml-cuda.cu:5616 @5266f24da).
fn getFeatures(r: c.ggml_backend_reg_t) callconv(.c) [*c]c.struct_ggml_backend_feature {
    _ = r;
    Features.once.call();
    return &Features.list;
}

/// Ports `ggml_backend_cuda_reg_get_proc_address` (ggml-cuda.cu:5673 @5266f24da).
fn regGetProcAddress(r: c.ggml_backend_reg_t, name_c: [*c]const u8) callconv(.c) ?*anyopaque {
    _ = r;
    const name = std.mem.span(@as([*:0]const u8, @ptrCast(name_c)));
    if (std.mem.eql(u8, name, "ggml_backend_comm_init")) return @ptrCast(@constCast(&commInit));
    if (std.mem.eql(u8, name, "ggml_backend_comm_free")) return @ptrCast(@constCast(&commFree));
    if (std.mem.eql(u8, name, "ggml_backend_comm_allreduce_tensor")) return @ptrCast(@constCast(&commAllreduceTensor));
    if (std.mem.eql(u8, name, "ggml_backend_register_host_buffer")) return @ptrCast(@constCast(&registerHostBuffer));
    if (std.mem.eql(u8, name, "ggml_backend_unregister_host_buffer")) return @ptrCast(@constCast(&unregisterHostBuffer));
    if (std.mem.eql(u8, name, "ggml_backend_get_features")) return @ptrCast(@constCast(&getFeatures));
    return null;
}

/// Ports `ggml_backend_cuda_reg_interface` (ggml-cuda.cu:5696 @5266f24da).
const reg_interface = c.struct_ggml_backend_reg_i{
    .get_name = regGetName,
    .get_device_count = regGetDeviceCount,
    .get_device = regGetDevice,
    .get_proc_address = regGetProcAddress,
};

const Registry = struct {
    var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
    var value: c.struct_ggml_backend_reg = undefined;
    var initialized = false;
    var ctx: RegCtx = .{};
    var devices: [common.max_devices]c.struct_ggml_backend_device = undefined;
    var device_ctxs: [common.max_devices]DeviceCtx = undefined;
};

/// Ports `ggml_backend_cuda_reg` (ggml-cuda.cu:5704 @5266f24da).
pub fn reg() callconv(.c) c.ggml_backend_reg_t {
    _ = std.c.pthread_mutex_lock(&Registry.mutex);
    defer _ = std.c.pthread_mutex_unlock(&Registry.mutex);
    if (!Registry.initialized) {
        const min_batch_size: c_int = if (std.c.getenv("GGML_OP_OFFLOAD_MIN_BATCH")) |e| common.atoi(e) else 32;

        const inf = common.info();
        const virtual_devices = inf.device_count > inf.physical_device_count;

        for (0..@intCast(inf.device_count)) |i| {
            const physical_id = inf.devices[i].physical_device;

            const dev_ctx = &Registry.device_ctxs[i];
            dev_ctx.* = .{ .device = @intCast(i), .op_offload_min_batch_size = min_batch_size };
            _ = std.fmt.bufPrintSentinel(&dev_ctx.name, common.name ++ "{d}", .{i}, 0) catch unreachable;
            _ = deviceDescription(@intCast(i), &dev_ctx.description);

            var pci_bus_id: [32]u8 = @splat(0);
            cudaCheck(cu.cudaDeviceGetPCIBusId(&pci_bus_id, pci_bus_id.len, physical_id), "cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id)", @src());
            const bus = std.mem.sliceTo(&pci_bus_id, 0);
            const written = if (virtual_devices)
                // make the pci bus id unique for virtual devices
                std.fmt.bufPrintSentinel(&dev_ctx.pci_bus_id, "{s}-v{d}", .{ bus, i }, 0) catch unreachable
            else
                std.fmt.bufPrintSentinel(&dev_ctx.pci_bus_id, "{s}", .{bus}, 0) catch unreachable;
            for (written) |*ch| ch.* = std.ascii.toLower(ch.*);

            Registry.devices[i] = .{
                .iface = device_interface,
                .reg = &Registry.value,
                .context = dev_ctx,
            };
            Registry.ctx.devices[i] = &Registry.devices[i];
            Registry.ctx.count = i + 1;
        }

        Registry.value = .{
            .api_version = c.GGML_BACKEND_API_VERSION,
            .iface = reg_interface,
            .context = &Registry.ctx,
        };
    }
    Registry.initialized = true;
    return &Registry.value;
}

/// Ports `ggml_backend_cuda_init` (ggml-cuda.cu:5759 @5266f24da).
pub fn init(device: c_int) callconv(.c) c.ggml_backend_t {
    if (device < 0 or device >= getDeviceCount()) {
        impl.logError("%s: invalid device %d\n", .{ "ggml_backend_cuda_init", device });
        return null;
    }

    const ctx = common.Context.create(device);
    const backend = allocator.create(c.struct_ggml_backend) catch impl.abort("ggml-cuda: out of memory");
    backend.* = .{
        .guid = &guid,
        .iface = backend_interface,
        .device = c.ggml_backend_reg_dev_get(reg(), @intCast(device)),
        .context = ctx,
    };
    return backend;
}

// -----------------------------------------------------------------------------
// The C ABI: `ggml-cuda.h`

comptime {
    @export(&init, .{ .name = "ggml_backend_cuda_init" });
    @export(&isCuda, .{ .name = "ggml_backend_is_cuda" });
    @export(&bufferType, .{ .name = "ggml_backend_cuda_buffer_type" });
    @export(&hostBufferType, .{ .name = "ggml_backend_cuda_host_buffer_type" });
    @export(&getDeviceCount, .{ .name = "ggml_backend_cuda_get_device_count" });
    @export(&getDeviceDescription, .{ .name = "ggml_backend_cuda_get_device_description" });
    @export(&getDeviceMemory, .{ .name = "ggml_backend_cuda_get_device_memory" });
    @export(&registerHostBuffer, .{ .name = "ggml_backend_cuda_register_host_buffer" });
    @export(&unregisterHostBuffer, .{ .name = "ggml_backend_cuda_unregister_host_buffer" });
    @export(&reg, .{ .name = "ggml_backend_cuda_reg" });
}
