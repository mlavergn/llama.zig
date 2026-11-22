//! The CUDA backend: the device buffer type, the pinned host buffer type,
//! the backend, the device, the registry, and the ten C functions of
//! `ggml-cuda.h`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` at v1.0.0
//! (`b1b770d90`). Each declaration below names the C++ it replaces and the
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
const ggml = @import("../module.zig");
const stdlib = @import("stdlib");
const config = @import("config");

// -----------------------------------------------------------------------------
// Device buffers

/// Ports `ggml_backend_cuda_buffer_context` (ggml-cuda.cu:944 @b1b770d90).
pub const BufferCtx = struct {
    device: c_int,
    dev_ptr: ?*anyopaque,
    name: [16:0]u8 = @splat(0),
};

fn bufferCtx(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t) *BufferCtx {
    return @ptrCast(@alignCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, buffer).context.?));
}

/// Ports `ggml_backend_cuda_buffer_free_buffer` (ggml-cuda.cu:959 @b1b770d90).
fn bufferFreeBuffer(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) void {
    const ctx = bufferCtx(buffer);
    // `~ggml_backend_cuda_buffer_context`.
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaFree(ctx.dev_ptr), "cudaFree(dev_ptr)", @src());
    std.heap.c_allocator.destroy(ctx);
}

/// Ports `ggml_backend_buffer_is_cuda` (ggml-cuda.cu:964 @b1b770d90).
pub fn bufferIsCuda(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t) bool {
    return ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, buffer).iface.free_buffer == &bufferFreeBuffer;
}

/// Ports `ggml_backend_cuda_buffer_get_base` (ggml-cuda.cu:968 @b1b770d90).
fn bufferGetBase(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return bufferCtx(buffer).dev_ptr;
}

/// Ports `ggml_backend_cuda_buffer_init_tensor` (ggml-cuda.cu:973 @b1b770d90).
/// The padding memset runs on the per-thread stream and is
/// waited for, not on the legacy default stream.
fn bufferInitTensor(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensor_c: [*c]ggml.impl.ggml_c.ggml_tensor) callconv(.c) ggml.impl.ggml_c.enum_ggml_status {
    const ctx = bufferCtx(buffer);
    const tensor = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor_c);

    if (tensor.view_src != null) return ggml.impl.ggml_c.GGML_STATUS_SUCCESS;

    if (ggml.impl.ggml_c.ggml_is_quantized(tensor.type) and ggml.impl.ggml_c.ggml_backend_buffer_get_usage(buffer) != ggml.impl.ggml_c.GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const original_size = ggml.impl.ggml_c.ggml_nbytes(tensor);
        const padded_size = ggml.impl.ggml_c.ggml_backend_buft_get_alloc_size(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, buffer).buft, tensor);

        if (padded_size > original_size) {
            ggml.cuda.common.setDevice(ctx.device);
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemsetAsync(@as([*]u8, @ptrCast(tensor.data.?)) + original_size, 0, padded_size - original_size, stream_per_thread), "cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread)", @src());
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
        }
    }
    return ggml.impl.ggml_c.GGML_STATUS_SUCCESS;
}

fn at(p: ?*anyopaque, offset: usize) *anyopaque {
    return @ptrCast(@as([*]u8, @ptrCast(p.?)) + offset);
}

const stream_per_thread: ggml.cuda.common.cu.cudaStream_t = @ptrFromInt(0x2);

/// Ports `ggml_backend_cuda_buffer_memset_tensor` (ggml-cuda.cu:995 @b1b770d90).
fn bufferMemsetTensor(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]ggml.impl.ggml_c.ggml_tensor, value: u8, offset: usize, size: usize) callconv(.c) void {
    ggml.cuda.common.setDevice(bufferCtx(buffer).device);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemsetAsync(at(tensor.*.data, offset), value, size, stream_per_thread), "cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_set_tensor` (ggml-cuda.cu:1003 @b1b770d90).
fn bufferSetTensor(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    ggml.cuda.common.setDevice(bufferCtx(buffer).device);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(at(tensor.*.data, offset), data, size, ggml.cuda.common.cu.cudaMemcpyHostToDevice, stream_per_thread), "cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_get_tensor` (ggml-cuda.cu:1011 @b1b770d90).
fn bufferGetTensor(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    ggml.cuda.common.setDevice(bufferCtx(buffer).device);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(data, at(tensor.*.data, offset), size, ggml.cuda.common.cu.cudaMemcpyDeviceToHost, stream_per_thread), "cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_set_tensor_2d` (ggml-cuda.cu:1019 @b1b770d90).
fn bufferSetTensor2d(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    ggml.cuda.common.setDevice(bufferCtx(buffer).device);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpy2DAsync(at(tensor.*.data, offset), stride_tensor, data, stride_data, size, n_copies, ggml.cuda.common.cu.cudaMemcpyHostToDevice, stream_per_thread), "cudaMemcpy2DAsync(...)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_get_tensor_2d` (ggml-cuda.cu:1029 @b1b770d90).
fn bufferGetTensor2d(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensor: [*c]const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    ggml.cuda.common.setDevice(bufferCtx(buffer).device);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpy2DAsync(data, stride_data, at(tensor.*.data, offset), stride_tensor, size, n_copies, ggml.cuda.common.cu.cudaMemcpyDeviceToHost, stream_per_thread), "cudaMemcpy2DAsync(...)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_set_tensor_n` (ggml-cuda.cu:1039 @b1b770d90),
/// Every non-empty copy on the per-thread stream, then one
/// wait.
fn bufferSetTensorN(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensors: [*c]const [*c]ggml.impl.ggml_c.ggml_tensor, data: [*c]const ?*const anyopaque, offsets: [*c]const usize, sizes: [*c]const usize, n_tensors: usize) callconv(.c) void {
    const ctx = bufferCtx(buffer);

    ggml.cuda.common.setDevice(ctx.device);

    for (0..n_tensors) |i| {
        if (sizes[i] == 0) {
            continue;
        }
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(at(tensors[i].*.data, offsets[i]), data[i], sizes[i], ggml.cuda.common.cu.cudaMemcpyHostToDevice, stream_per_thread), "cudaMemcpyAsync( (char *) tensors[i]->data + offsets[i], data[i], sizes[i], cudaMemcpyHostToDevice, cudaStreamPerThread)", @src());
    }

    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_get_tensor_n` (ggml-cuda.cu:1059 @b1b770d90),
/// `batch_get.getTensorN`'s gather when pinned memory is
/// allowed and it accepts the request, else one copy per read.
fn bufferGetTensorN(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, tensors: [*c]const [*c]const ggml.impl.ggml_c.ggml_tensor, data: [*c]const ?*anyopaque, offsets: [*c]const usize, sizes: [*c]const usize, n_tensors: usize) callconv(.c) void {
    const ctx = bufferCtx(buffer);

    ggml.cuda.common.setDevice(ctx.device);

    if (!ggml.cuda.common.pinnedMemoryDisabled() and
        ggml.cuda.batch_get.getTensorN(ctx.device, tensors, data, offsets, sizes, n_tensors))
    {
        return;
    }

    for (0..n_tensors) |i| {
        if (sizes[i] == 0) {
            continue;
        }
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(data[i], at(tensors[i].*.data, offsets[i]), sizes[i], ggml.cuda.common.cu.cudaMemcpyDeviceToHost, stream_per_thread), "cudaMemcpyAsync( data[i], (const char *) tensors[i]->data + offsets[i], sizes[i], cudaMemcpyDeviceToHost, cudaStreamPerThread)", @src());
    }

    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_cpy_tensor` (ggml-cuda.cu:1084 @b1b770d90).
fn bufferCpyTensor(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, src_c: [*c]const ggml.impl.ggml_c.ggml_tensor, dst_c: [*c]ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    _ = buffer;
    const src = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(src_c));
    const dst = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst_c);
    if (!bufferIsCuda(src.buffer)) return false;
    const src_ctx = bufferCtx(src.buffer);
    const dst_ctx = bufferCtx(dst.buffer);
    // compare the backing physical devices: distinct virtual devices may
    // share one physical GPU, in which case a same-device copy is required
    const src_physical = ggml.cuda.common.physicalDevice(src_ctx.device);
    const dst_physical = ggml.cuda.common.physicalDevice(dst_ctx.device);
    if (src_physical == dst_physical) {
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(dst.data, src.data, ggml.impl.ggml_c.ggml_nbytes(src), ggml.cuda.common.cu.cudaMemcpyDeviceToDevice, stream_per_thread), "cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread)", @src());
    } else {
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyPeerAsync(dst.data, dst_physical, src.data, src_physical, ggml.impl.ggml_c.ggml_nbytes(src), stream_per_thread), "cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread)", @src());
    }
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
    return true;
}

/// The buffer a tensor's bytes live in: its view source's, for a view.
fn backingBuffer(t: *const ggml.impl.ggml_c.ggml_tensor) ggml.impl.ggml_c.ggml_backend_buffer_t {
    return if (t.view_src) |v| v.*.buffer else t.buffer;
}

/// Ports `ggml_backend_cuda_buffer_cpy_tensor_n` (ggml-cuda.cu:1109 @b1b770d90),
/// False, having copied nothing, unless every source is in a
/// CUDA buffer; then each copy same-device or peer on the per-thread stream,
/// and one wait. `GGML_CUDA_NO_PEER_COPY` is not defined, so its two arms
/// are not ported.
fn bufferCpyTensorN(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, srcs: [*c]const [*c]const ggml.impl.ggml_c.ggml_tensor, dsts: [*c]const [*c]ggml.impl.ggml_c.ggml_tensor, n_tensors: usize) callconv(.c) bool {
    const dst_ctx = bufferCtx(buffer);

    for (0..n_tensors) |i| {
        const src_buf = backingBuffer(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(srcs[i])));
        if (!bufferIsCuda(src_buf)) {
            return false;
        }
    }

    ggml.cuda.common.setDevice(dst_ctx.device);

    for (0..n_tensors) |i| {
        const src = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(srcs[i]));
        const dst = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dsts[i]);
        const src_ctx = bufferCtx(backingBuffer(src));
        const dst_ctx_i = bufferCtx(backingBuffer(dst));
        const src_physical = ggml.cuda.common.physicalDevice(src_ctx.device);
        const dst_physical = ggml.cuda.common.physicalDevice(dst_ctx_i.device);
        if (src_physical == dst_physical) {
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(dst.data, src.data, ggml.impl.ggml_c.ggml_nbytes(src), ggml.cuda.common.cu.cudaMemcpyDeviceToDevice, stream_per_thread), "cudaMemcpyAsync(dsts[i]->data, srcs[i]->data, ggml_nbytes(srcs[i]), cudaMemcpyDeviceToDevice, cudaStreamPerThread)", @src());
        } else {
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyPeerAsync(dst.data, dst_physical, src.data, src_physical, ggml.impl.ggml_c.ggml_nbytes(src), stream_per_thread), "cudaMemcpyPeerAsync(dsts[i]->data, dst_physical, srcs[i]->data, src_physical, ggml_nbytes(srcs[i]), cudaStreamPerThread)", @src());
        }
    }

    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
    return true;
}

/// Ports `ggml_backend_cuda_buffer_clear` (ggml-cuda.cu:1157 @b1b770d90).
fn bufferClear(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    const ctx = bufferCtx(buffer);
    ggml.cuda.common.setDevice(ctx.device);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemsetAsync(ctx.dev_ptr, value, ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, buffer).size, stream_per_thread), "cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread)", @src());
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
}

/// Ports `ggml_backend_cuda_buffer_interface` (ggml-cuda.cu:1165 @b1b770d90).
/// The CUDA buffer implements the three batched slots.
const buffer_interface = ggml.impl.ggml_c.ggml_backend_buffer_i{
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
    .set_tensor_n = bufferSetTensorN,
    .get_tensor_n = bufferGetTensorN,
    .cpy_tensor_n = bufferCpyTensorN,
};

// -----------------------------------------------------------------------------
// Device buffer type

/// Ports `ggml_backend_cuda_buffer_type_context` (ggml-cuda.cu:1183 @b1b770d90).
pub const BufferTypeCtx = struct {
    device: c_int,
    name: [16:0]u8 = @splat(0),
};

/// Ports `ggml_backend_cuda_buffer_type_get_name` (ggml-cuda.cu:1188 @b1b770d90).
fn bufferTypeGetName(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    const ctx: *const BufferTypeCtx = @ptrCast(@alignCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer_type, buft).context.?));
    return &ctx.name;
}

/// Ports `ggml_backend_buft_is_cuda` (ggml-cuda.cu:1194 @b1b770d90).
pub fn buftIsCuda(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) bool {
    return ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer_type, buft).iface.get_name == &bufferTypeGetName;
}

/// The device a CUDA buffer type allocates on.
pub fn buftDevice(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) c_int {
    const ctx: *const BufferTypeCtx = @ptrCast(@alignCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer_type, buft).context.?));
    return ctx.device;
}

/// Ports `ggml_backend_cuda_buffer_type_alloc_buffer` (ggml-cuda.cu:1198 @b1b770d90).
fn bufferTypeAllocBuffer(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t, size: usize) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_t {
    const device = buftDevice(buft);
    ggml.cuda.common.setDevice(device);

    var dev_ptr: ?*anyopaque = null;
    const err = ggml.cuda.common.deviceMalloc(&dev_ptr, size, device);
    if (err != ggml.cuda.common.cu.cudaSuccess) {
        // clear the error
        _ = ggml.cuda.common.cu.cudaGetLastError();
        ggml.impl.logError("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", .{ "ggml_backend_cuda_buffer_type_alloc_buffer", @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0, device, ggml.cuda.common.cu.cudaGetErrorString(err) });
        return null;
    }

    const ctx = std.heap.c_allocator.create(BufferCtx) catch ggml.impl.abort("ggml-cuda: out of memory");
    ctx.* = .{ .device = device, .dev_ptr = dev_ptr };
    _ = std.fmt.bufPrintSentinel(&ctx.name, ggml.cuda.common.name ++ "{d}", .{device}, 0) catch unreachable;

    return ggml.impl.ggml_c.ggml_backend_buffer_init(buft, buffer_interface, ctx, size);
}

/// Ports `ggml_backend_cuda_buffer_type_get_alignment` (ggml-cuda.cu:1217 @b1b770d90).
fn bufferTypeGetAlignment(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return 128;
}

/// Ports `ggml_backend_cuda_buffer_type_get_alloc_size` (ggml-cuda.cu:1223 @b1b770d90).
fn bufferTypeGetAllocSize(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t, tensor_c: [*c]const ggml.impl.ggml_c.ggml_tensor) callconv(.c) usize {
    const tensor = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(tensor_c));
    var size = if (tensor.op == ggml.impl.ggml_c.GGML_OP_FLASH_ATTN_EXT)
        ggml.cuda.fattn.getAllocSize(buftDevice(buft), tensor)
    else
        ggml.impl.ggml_c.ggml_nbytes(tensor);
    const ne0 = tensor.ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml.impl.ggml_c.ggml_is_quantized(tensor.type)) {
        if (@mod(ne0, ggml.cuda.common.matrix_row_padding) != 0) {
            ggml.impl.assert(tensor.nb[0] == ggml.impl.ggml_c.ggml_element_size(tensor), "tensor->nb[0] == ggml_element_size(tensor)");
            size += ggml.impl.ggml_c.ggml_row_size(tensor.type, ggml.cuda.common.matrix_row_padding - @mod(ne0, ggml.cuda.common.matrix_row_padding));
        }
    }

    return size;
}

/// Ports `ggml_backend_cuda_buffer_type_interface` (ggml-cuda.cu:1242 @b1b770d90).
/// `alloc_buffer_n` and `get_alloc_size_n` are null: the
/// generic planner in `ggml-backend.cpp` serves them.
const buffer_type_interface = ggml.impl.ggml_c.struct_ggml_backend_buffer_type_i{
    .get_name = bufferTypeGetName,
    .alloc_buffer = bufferTypeAllocBuffer,
    .alloc_buffer_n = null,
    .get_alignment = bufferTypeGetAlignment,
    .get_max_size = null,
    .get_alloc_size = bufferTypeGetAllocSize,
    .get_alloc_size_n = null,
    .is_host = null,
};

const BufferTypes = struct {
    var mutex: stdlib.sync.Mutex = .{};
    var types: [ggml.cuda.common.max_devices]ggml.impl.ggml_c.struct_ggml_backend_buffer_type = undefined;
    var ctxs: [ggml.cuda.common.max_devices]BufferTypeCtx = undefined;
    var initialized = false;
};

/// Ports `ggml_backend_cuda_buffer_type` (ggml-cuda.cu:1253 @b1b770d90).
pub fn bufferType(device: c_int) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    BufferTypes.mutex.lock();
    defer BufferTypes.mutex.unlock();

    if (device >= getDeviceCount()) return null;

    if (!BufferTypes.initialized) {
        var i: c_int = 0;
        while (i < getDeviceCount()) : (i += 1) {
            const u: usize = @intCast(i);
            BufferTypes.ctxs[u] = .{ .device = i };
            _ = std.fmt.bufPrintSentinel(&BufferTypes.ctxs[u].name, ggml.cuda.common.name ++ "{d}", .{i}, 0) catch unreachable;
            BufferTypes.types[u] = .{
                .iface = buffer_type_interface,
                .device = ggml.impl.ggml_c.ggml_backend_reg_dev_get(reg(), u),
                .context = &BufferTypes.ctxs[u],
            };
        }
        BufferTypes.initialized = true;
    }

    return &BufferTypes.types[@intCast(device)];
}

// -----------------------------------------------------------------------------
// Tensor-parallel communication

/// Ports `ggml_backend_cuda_comm_context` (ggml-cuda.cu:1285 @b1b770d90),
/// without the NCCL member: `GGML_USE_NCCL` is not defined. So neither is
/// `ggml_backend_cuda_comm_init_rank` (one rank per process,
/// `ncclCommInitRank`), the only thing that sets `world_size`.
const CommContext = struct {
    const TryAllreduce = *const fn (*CommContext, [*c][*c]ggml.impl.ggml_c.ggml_tensor) bool;

    backends: []ggml.impl.ggml_c.ggml_backend_t,
    dev_ids: []c_int,
    try_allreduce: ?TryAllreduce = null,
    ar_pipeline: ?*anyopaque = null,

    /// Number of ranks in the communicator when each rank is a separate
    /// process. Zero unless comm_init_rank created this context.
    world_size: c_int = 0,

    /// Ports `ggml_backend_cuda_comm_context`'s `n_ranks`: the ranks taking
    /// part in the reduction. Only the NCCL reduction reads it.
    fn nRanks(self: *const CommContext) usize {
        return if (self.world_size > 0) @intCast(self.world_size) else self.backends.len;
    }

    fn destroy(self: *CommContext) void {
        arPipelineFree(self.ar_pipeline);
        std.heap.c_allocator.free(self.backends);
        std.heap.c_allocator.free(self.dev_ids);
        std.heap.c_allocator.destroy(self);
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

/// Ports `ggml_backend_cuda_comm_try_allreduce_butterfly` (ggml-cuda.cu:1471 @b1b770d90).
fn commTryAllreduceButterfly(_: *CommContext, _: [*c][*c]ggml.impl.ggml_c.ggml_tensor) bool {
    return false;
}

/// Ports `ggml_backend_cuda_comm_free` (ggml-cuda.cu:1476 @b1b770d90).
fn commFree(comm_ctx_v: ?*anyopaque) callconv(.c) void {
    const ctx: *CommContext = @ptrCast(@alignCast(comm_ctx_v orelse return));
    ctx.destroy();
}

/// Ports `ggml_backend_cuda_comm_init_none` (ggml-cuda.cu:1487 @b1b770d90).
fn commInitNone(ret: *CommContext) void {
    ret.try_allreduce = commTryAllreduceButterfly;
}

/// Ports `ggml_backend_cuda_comm_init_internal` (ggml-cuda.cu:1491 @b1b770d90).
fn commInitInternal(ret: *CommContext) void {
    ret.ar_pipeline = arPipelineInit(ret.dev_ids);
    if (ret.ar_pipeline != null) {
        unreachable; // the internal pipeline is not ported; see the file comment
    }

    // Clear sticky CUDA error from the failed init.
    _ = ggml.cuda.common.cu.cudaGetLastError();
    ggml.impl.logWarn("internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly\n", .{});
    commInitNone(ret);
}

/// Ports `ggml_backend_cuda_comm_init_nccl` (ggml-cuda.cu:1505 @b1b770d90),
/// the arm without `GGML_USE_NCCL`.
fn commInitNccl(ret: *CommContext) void {
    ggml.impl.logWarn("NCCL not compiled in; falling back to internal AllReduce.  Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n", .{});
    commInitInternal(ret);
}

/// Ports `ggml_backend_cuda_comm_init` (ggml-cuda.cu:1541 @b1b770d90).
fn commInit(backends: [*c]ggml.impl.ggml_c.ggml_backend_t, n_backends: usize) callconv(.c) ?*anyopaque {
    for (0..n_backends) |i| {
        if (!isCuda(backends[i])) return null;
    }

    const ret = std.heap.c_allocator.create(CommContext) catch ggml.impl.abort("ggml-cuda: out of memory");
    ret.* = .{
        .backends = std.heap.c_allocator.dupe(ggml.impl.ggml_c.ggml_backend_t, backends[0..n_backends]) catch ggml.impl.abort("ggml-cuda: out of memory"),
        .dev_ids = std.heap.c_allocator.alloc(c_int, n_backends) catch ggml.impl.abort("ggml-cuda: out of memory"),
    };
    for (0..n_backends) |i| ret.dev_ids[i] = backendCtx(backends[i]).device;

    if (stdlib.env.getenv("GGML_CUDA_ALLREDUCE")) |env_c| {
        const env = std.mem.span(env_c);
        if (std.mem.eql(u8, env, "nccl")) {
            commInitNccl(ret);
        } else if (std.mem.eql(u8, env, "internal")) {
            commInitInternal(ret);
        } else if (std.mem.eql(u8, env, "none")) {
            commInitNone(ret);
        } else {
            ggml.impl.logWarn("unknown GGML_CUDA_ALLREDUCE value: %s\n", .{env_c});
            commInitNone(ret);
        }
    } else {
        // Platform default: Linux uses NCCL, otherwise internal.
        if (@import("builtin").os.tag == .linux) commInitNccl(ret) else commInitInternal(ret);
    }

    return ret;
}

/// Ports `ggml_backend_cuda_comm_allreduce_tensor` (ggml-cuda.cu:1666 @b1b770d90).
fn commAllreduceTensor(comm_ctx_v: ?*anyopaque, tensors: [*c][*c]ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    const ctx: *CommContext = @ptrCast(@alignCast(comm_ctx_v orelse return false));
    return ctx.try_allreduce.?(ctx, tensors);
}

// -----------------------------------------------------------------------------
// Pinned host buffer type

/// Ports `ggml_backend_cuda_host_buffer_type_name` (ggml-cuda.cu:1676 @b1b770d90).
fn hostBufferTypeName(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    _ = buft;
    return ggml.cuda.common.name ++ "_Host";
}

/// Ports `ggml_backend_buft_is_cuda_host` (ggml-cuda.cu:1682 @b1b770d90).
pub fn buftIsCudaHost(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) bool {
    return ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer_type, buft).iface.get_name == &hostBufferTypeName;
}

/// Ports `ggml_backend_cuda_host_buffer_free_buffer` (ggml-cuda.cu:1686 @b1b770d90).
fn hostBufferFreeBuffer(buffer: ggml.impl.ggml_c.ggml_backend_buffer_t) callconv(.c) void {
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaFreeHost(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, buffer).context), "cudaFreeHost(buffer->context)", @src());
}

/// Ports `ggml_cuda_host_malloc` (ggml-cuda.cu:1690 @b1b770d90).
fn hostMalloc(size: usize) ?*anyopaque {
    if (ggml.cuda.common.pinnedMemoryDisabled()) return null;

    var ptr: ?*anyopaque = null;
    const err = ggml.cuda.common.cu.cudaMallocHost(&ptr, size);
    if (err != ggml.cuda.common.cu.cudaSuccess) {
        // clear the error
        _ = ggml.cuda.common.cu.cudaGetLastError();
        ggml.impl.logDebug("%s: failed to allocate %.2f MiB of pinned memory: %s\n", .{ "ggml_cuda_host_malloc", @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0, ggml.cuda.common.cu.cudaGetErrorString(err) });
        return null;
    }
    return ptr;
}

/// Ports `ggml_backend_cuda_host_buffer_type_alloc_buffer` (ggml-cuda.cu:1708 @b1b770d90).
fn hostBufferTypeAllocBuffer(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t, size: usize) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_t {
    const ptr = hostMalloc(size) orelse {
        // fallback to cpu buffer
        return ggml.impl.ggml_c.ggml_backend_buft_alloc_buffer(ggml.impl.ggml_c.ggml_backend_cpu_buffer_type(), size);
    };

    const buffer = ggml.impl.ggml_c.ggml_backend_cpu_buffer_from_ptr(ptr, size);
    const b = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, buffer);
    b.buft = buft;
    b.iface.free_buffer = hostBufferFreeBuffer;
    return buffer;
}

/// Ports `ggml_backend_cuda_host_buffer_type_get_max_size` (ggml-cuda.cu:1723 @b1b770d90):
/// very large pinned allocations can fail (e.g. on Windows) and fall back to
/// pageable memory, smaller ones succeed.
fn hostBufferTypeGetMaxSize(buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return 4 * 1024 * 1024 * 1024;
}

const HostBufferType = struct {
    var once_value: ggml.impl.ggml_c.struct_ggml_backend_buffer_type = undefined;
    fn run() void {
        const cpu = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer_type, ggml.impl.ggml_c.ggml_backend_cpu_buffer_type());
        once_value = .{
            .iface = .{
                .get_name = hostBufferTypeName,
                .alloc_buffer = hostBufferTypeAllocBuffer,
                .alloc_buffer_n = null,
                .get_alignment = cpu.iface.get_alignment,
                .get_max_size = hostBufferTypeGetMaxSize,
                .get_alloc_size = cpu.iface.get_alloc_size,
                .get_alloc_size_n = null,
                .is_host = cpu.iface.is_host,
            },
            .device = ggml.impl.ggml_c.ggml_backend_reg_dev_get(reg(), 0),
            .context = null,
        };
    }
    const once = ggml.cuda.common.Once(run);
};

/// Ports `ggml_backend_cuda_host_buffer_type` (ggml-cuda.cu:1729 @b1b770d90).
pub fn hostBufferType() callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    HostBufferType.once.call();
    return &HostBufferType.once_value;
}

// -----------------------------------------------------------------------------
// Backend

pub fn backendCtx(backend: ggml.impl.ggml_c.ggml_backend_t) *ggml.cuda.common.Context {
    return @ptrCast(@alignCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend).context.?));
}

/// Ports `ggml_backend_cuda_get_name` (ggml-cuda.cu:3075 @b1b770d90).
fn backendGetName(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return backendCtx(backend).getName();
}

/// Ports `ggml_backend_cuda_free` (ggml-cuda.cu:3081 @b1b770d90).
fn backendFree(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    backendCtx(backend).destroy();
    std.heap.c_allocator.destroy(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend));
}

fn bufferOf(tensor: *const ggml.impl.ggml_c.ggml_tensor) ggml.impl.ggml_c.ggml_backend_buffer_t {
    return if (tensor.view_src) |v| v.*.buffer else tensor.buffer;
}

fn assertOwnBuffer(cuda_ctx: *ggml.cuda.common.Context, tensor: *const ggml.impl.ggml_c.ggml_tensor) void {
    const buf = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer, bufferOf(tensor));
    ggml.impl.assert(buf.buft == bufferType(cuda_ctx.device), "buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && \"unsupported buffer type\"");
}

/// Ports `ggml_backend_cuda_set_tensor_async` (ggml-cuda.cu:3088 @b1b770d90).
fn backendSetTensorAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor: [*c]ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor));
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(at(tensor.*.data, offset), data, size, ggml.cuda.common.cu.cudaMemcpyHostToDevice, cuda_ctx.stream()), "cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_get_tensor_async` (ggml-cuda.cu:3097 @b1b770d90).
fn backendGetTensorAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor: [*c]const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(tensor)));
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(data, at(tensor.*.data, offset), size, ggml.cuda.common.cu.cudaMemcpyDeviceToHost, cuda_ctx.stream()), "cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_set_tensor_2d_async` (ggml-cuda.cu:3106 @b1b770d90).
fn backendSetTensor2dAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor: [*c]ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, tensor));
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpy2DAsync(at(tensor.*.data, offset), stride_tensor, data, stride_data, size, n_copies, ggml.cuda.common.cu.cudaMemcpyHostToDevice, cuda_ctx.stream()), "cudaMemcpy2DAsync(...)", @src());
}

/// Ports `ggml_backend_cuda_get_tensor_2d_async` (ggml-cuda.cu:3117 @b1b770d90).
fn backendGetTensor2dAsync(backend: ggml.impl.ggml_c.ggml_backend_t, tensor: [*c]const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize, n_copies: usize, stride_tensor: usize, stride_data: usize) callconv(.c) void {
    const cuda_ctx = backendCtx(backend);
    assertOwnBuffer(cuda_ctx, ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(tensor)));
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpy2DAsync(data, stride_data, at(tensor.*.data, offset), stride_tensor, size, n_copies, ggml.cuda.common.cu.cudaMemcpyDeviceToHost, cuda_ctx.stream()), "cudaMemcpy2DAsync(...)", @src());
}

/// Ports `ggml_backend_cuda_cpy_tensor_async` (ggml-cuda.cu:3128 @b1b770d90).
fn backendCpyTensorAsync(backend_src: ggml.impl.ggml_c.ggml_backend_t, backend_dst: ggml.impl.ggml_c.ggml_backend_t, src_c: [*c]const ggml.impl.ggml_c.ggml_tensor, dst_c: [*c]ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    const src = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(src_c));
    const dst = ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, dst_c);
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
        const src_physical = ggml.cuda.common.physicalDevice(cuda_ctx_src.device);
        const dst_physical = ggml.cuda.common.physicalDevice(cuda_ctx_dst.device);
        if (src_physical == dst_physical) {
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(dst.data, src.data, ggml.impl.ggml_c.ggml_nbytes(dst), ggml.cuda.common.cu.cudaMemcpyDeviceToDevice, cuda_ctx_src.stream()), "cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream())", @src());
        } else {
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyPeerAsync(dst.data, dst_physical, src.data, src_physical, ggml.impl.ggml_c.ggml_nbytes(dst), cuda_ctx_src.stream()), "cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream())", @src());
        }

        // record event on src stream after the copy
        if (cuda_ctx_src.copy_event == null) {
            ggml.cuda.common.setDevice(cuda_ctx_src.device);
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventCreateWithFlags(&cuda_ctx_src.copy_event, ggml.cuda.common.cu.cudaEventDisableTiming), "cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming)", @src());
        }

        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventRecord(cuda_ctx_src.copy_event, cuda_ctx_src.stream()), "cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream())", @src());

        // wait on dst stream for the copy to complete
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamWaitEvent(cuda_ctx_dst.stream(), cuda_ctx_src.copy_event, 0), "cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0)", @src());
    } else {
        // src and dst are on the same backend
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemcpyAsync(dst.data, src.data, ggml.impl.ggml_c.ggml_nbytes(dst), ggml.cuda.common.cu.cudaMemcpyDeviceToDevice, cuda_ctx_src.stream()), "cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream())", @src());
    }
    return true;
}

/// Ports `ggml_backend_cuda_synchronize` (ggml-cuda.cu:3187 @b1b770d90).
fn backendSynchronize(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) void {
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamSynchronize(backendCtx(backend).stream()), "cudaStreamSynchronize(cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_event_record` (ggml-cuda.cu:5483 @b1b770d90).
fn backendEventRecord(backend: ggml.impl.ggml_c.ggml_backend_t, event: ggml.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventRecord(@ptrCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_event, event).context), backendCtx(backend).stream()), "cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream())", @src());
}

/// Ports `ggml_backend_cuda_event_wait` (ggml-cuda.cu:5489 @b1b770d90).
fn backendEventWait(backend: ggml.impl.ggml_c.ggml_backend_t, event: ggml.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    if (isCuda(backend)) {
        ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaStreamWaitEvent(backendCtx(backend).stream(), @ptrCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_event, event).context), 0), "cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0)", @src());
    } else {
        ggml.impl.abort("fatal error");
    }
}

/// Ports `ggml_backend_cuda_interface` (ggml-cuda.cu:5904 @b1b770d90).
const backend_interface = ggml.impl.ggml_c.struct_ggml_backend_i{
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
    .graph_compute = ggml.cuda.graph.graphCompute,
    .event_record = backendEventRecord,
    .event_wait = backendEventWait,
    .graph_optimize = ggml.cuda.graph.graphOptimize,
};

/// Ports `ggml_backend_cuda_guid` (ggml-cuda.cu:5923 @b1b770d90).
var guid: ggml.impl.ggml_c.ggml_guid = .{ 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };

/// Ports `ggml_backend_is_cuda` (ggml-cuda.cu:5928 @b1b770d90).
pub fn isCuda(backend: ggml.impl.ggml_c.ggml_backend_t) callconv(.c) bool {
    return backend != null and ggml.impl.ggml_c.ggml_guid_matches(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend, backend).guid, &guid);
}

/// Ports `ggml_backend_cuda_get_device_count` (ggml-cuda.cu:5932 @b1b770d90).
pub fn getDeviceCount() callconv(.c) c_int {
    return ggml.cuda.common.info().device_count;
}

/// Ports `ggml_cuda_device_description` (ggml-cuda.cu:5936 @b1b770d90).
fn deviceDescription(device: c_int, out: []u8) [:0]u8 {
    var prop: ggml.cuda.common.cu.cudaDeviceProp = undefined;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetDeviceProperties(&prop, ggml.cuda.common.physicalDevice(device)), "cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device))", @src());

    const inf = ggml.cuda.common.info();
    const prop_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&prop.name)), 0);
    if (inf.device_count > inf.physical_device_count) {
        const d = inf.devices[@intCast(device)];
        return std.fmt.bufPrintSentinel(out, "{s} (dev p{d}/v{d})", .{ prop_name, d.physical_device, d.virtual_index }, 0) catch unreachable;
    }
    return std.fmt.bufPrintSentinel(out, "{s}", .{prop_name}, 0) catch unreachable;
}

/// Ports `ggml_backend_cuda_get_device_description` (ggml-cuda.cu:5949 @b1b770d90).
pub fn getDeviceDescription(device: c_int, description: [*c]u8, description_size: usize) callconv(.c) void {
    var buf: [320]u8 = undefined;
    const d = deviceDescription(device, &buf);
    _ = stdlib.libc.snprintf(description, description_size, "%s", d.ptr);
}

/// Ports `ggml_cuda_physical_device_share_count` (ggml-cuda.cu:5953 @b1b770d90).
fn physicalDeviceShareCount(device: c_int) c_int {
    const inf = ggml.cuda.common.info();
    ggml.impl.assert(device >= 0 and device < inf.device_count, "device >= 0 && device < info.device_count");
    return inf.devices[@intCast(device)].physical_share_count;
}

/// Ports `ggml_backend_cuda_get_device_memory` (ggml-cuda.cu:5959 @b1b770d90).
pub fn getDeviceMemory(device: c_int, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    ggml.cuda.common.setDevice(device);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaMemGetInfo(free, total), "cudaMemGetInfo(free, total)", @src());
    // virtual devices sharing one physical GPU share its memory pool
    const share_count: usize = @intCast(physicalDeviceShareCount(device));
    free.* /= share_count;
    total.* /= share_count;
}

/// Ports `ggml_backend_cuda_register_host_buffer` (ggml-cuda.cu:5970 @b1b770d90).
pub fn registerHostBuffer(buffer: ?*anyopaque, size: usize) callconv(.c) bool {
    if (!ggml.cuda.common.registerHostEnabled()) return false;

    const err = ggml.cuda.common.cu.cudaHostRegister(buffer, size, ggml.cuda.common.cu.cudaHostRegisterPortable | ggml.cuda.common.cu.cudaHostRegisterReadOnly);
    if (err != ggml.cuda.common.cu.cudaSuccess) {
        // clear the error
        _ = ggml.cuda.common.cu.cudaGetLastError();
        ggml.impl.logDebug("%s: failed to register %.2f MiB of pinned memory: %s\n", .{ "ggml_backend_cuda_register_host_buffer", @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0, ggml.cuda.common.cu.cudaGetErrorString(err) });
        return false;
    }
    return true;
}

/// Ports `ggml_backend_cuda_unregister_host_buffer` (ggml-cuda.cu:5993 @b1b770d90).
pub fn unregisterHostBuffer(buffer: ?*anyopaque) callconv(.c) void {
    if (!ggml.cuda.common.registerHostEnabled()) return;
    const err = ggml.cuda.common.cu.cudaHostUnregister(buffer);
    if (err != ggml.cuda.common.cu.cudaSuccess) {
        // clear the error
        _ = ggml.cuda.common.cu.cudaGetLastError();
    }
}

// -----------------------------------------------------------------------------
// Device

/// Ports `ggml_backend_cuda_device_context` (ggml-cuda.cu:6008 @b1b770d90).
pub const DeviceCtx = struct {
    device: c_int,
    name: [16:0]u8 = @splat(0),
    description: [320:0]u8 = @splat(0),
    pci_bus_id: [48:0]u8 = @splat(0),
    op_offload_min_batch_size: c_int,
};

pub fn deviceCtx(dev: ggml.impl.ggml_c.ggml_backend_dev_t) *DeviceCtx {
    return @ptrCast(@alignCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_device, dev).context.?));
}

/// Ports `ggml_backend_cuda_device_get_name` (ggml-cuda.cu:6016 @b1b770d90).
fn deviceGetName(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return &deviceCtx(dev).name;
}

/// Ports `ggml_backend_cuda_device_get_description` (ggml-cuda.cu:6021 @b1b770d90).
fn deviceGetDescription(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return &deviceCtx(dev).description;
}

/// Ports `ggml_backend_cuda_get_available_uma_memory` (ggml-cuda.cu:6028 @b1b770d90).
fn getAvailableUmaMemory(available_memory_kb: *c_long, free_swap_kb: *c_long) bool {
    const buffer_size = 2048;
    var file_buffer: [buffer_size]u8 = undefined;
    var huge_tlb_total_pages: c_long = -1;
    var huge_tlb_free_pages: c_long = -1;
    var huge_tlb_page_size: c_long = -1;

    const meminfo_file = std.c.fopen("/proc/meminfo", "r") orelse {
        ggml.impl.logError("%s: failed to open /proc/meminfo\n", .{"ggml_backend_cuda_get_available_uma_memory"});
        return false;
    };

    // Read file into buffer
    const bytes_read = std.c.fread(&file_buffer, 1, buffer_size - 1, meminfo_file);
    _ = std.c.fclose(meminfo_file);

    if (bytes_read == 0) {
        ggml.impl.logError("%s: failed to read from /proc/meminfo\n", .{"ggml_backend_cuda_get_available_uma_memory"});
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

    ggml.impl.logDebug("%s: final available_memory_kb: %ld\n", .{ "ggml_backend_cuda_get_available_uma_memory", available_memory_kb.* });
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

/// Ports `ggml_backend_cuda_device_get_memory` (ggml-cuda.cu:6102 @b1b770d90).
fn deviceGetMemory(dev: ggml.impl.ggml_c.ggml_backend_dev_t, free: [*c]usize, total: [*c]usize) callconv(.c) void {
    const ctx = deviceCtx(dev);
    ggml.cuda.common.setDevice(ctx.device);
    const err = ggml.cuda.common.cu.cudaMemGetInfo(free, total);
    if (err != ggml.cuda.common.cu.cudaSuccess) {
        _ = ggml.cuda.common.cu.cudaGetLastError();
        ggml.impl.logWarn("%s: cudaMemGetInfo failed (%s), returning 0/0\n", .{ "ggml_backend_cuda_device_get_memory", ggml.cuda.common.cu.cudaGetErrorString(err) });
        free.* = 0;
        total.* = 0;
        return;
    }

    // Check if this is a UMA (Unified Memory Architecture) system
    var prop: ggml.cuda.common.cu.cudaDeviceProp = undefined;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetDeviceProperties(&prop, ggml.cuda.common.physicalDevice(ctx.device)), "cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device))", @src());

    // Check if UMA is explicitly enabled via environment variable
    const is_uma = prop.integrated > 0 or ggml.cuda.common.unifiedMemoryEnabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        var available_memory_kb: c_long = 0;
        var free_swap_kb: c_long = 0;
        if (getAvailableUmaMemory(&available_memory_kb, &free_swap_kb) and available_memory_kb > 0) {
            free.* = @as(usize, @intCast(available_memory_kb)) * 1024;
        } else {
            ggml.impl.logError("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", .{"ggml_backend_cuda_device_get_memory"});
        }
    }

    // virtual devices sharing one physical GPU share its memory pool
    const share_count: usize = @intCast(physicalDeviceShareCount(ctx.device));
    free.* /= share_count;
    total.* /= share_count;
}

/// Ports `ggml_backend_cuda_device_get_type` (ggml-cuda.cu:6142 @b1b770d90).
fn deviceGetType(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) ggml.impl.ggml_c.enum_ggml_backend_dev_type {
    var prop: ggml.cuda.common.cu.cudaDeviceProp = undefined;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaGetDeviceProperties(&prop, ggml.cuda.common.physicalDevice(deviceCtx(dev).device)), "cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device))", @src());
    return if (prop.integrated != 0) ggml.impl.ggml_c.GGML_BACKEND_DEVICE_TYPE_IGPU else ggml.impl.ggml_c.GGML_BACKEND_DEVICE_TYPE_GPU;
}

/// Ports `ggml_backend_cuda_device_get_props` (ggml-cuda.cu:6153 @b1b770d90).
fn deviceGetProps(dev: ggml.impl.ggml_c.ggml_backend_dev_t, props_c: [*c]ggml.impl.ggml_c.struct_ggml_backend_dev_props) callconv(.c) void {
    const ctx = deviceCtx(dev);
    const props = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_dev_props, props_c);

    props.name = deviceGetName(dev);
    props.description = deviceGetDescription(dev);
    props.type = deviceGetType(dev);
    props.device_id = if (ctx.pci_bus_id[0] == 0) null else &ctx.pci_bus_id;
    deviceGetMemory(dev, &props.memory_free, &props.memory_total);

    const host_buffer = !ggml.cuda.common.pinnedMemoryDisabled();
    props.caps = .{
        .async = true,
        .host_buffer = host_buffer,
        .buffer_from_host_ptr = false,
        .events = true,
        .mmap_support = props.type != ggml.impl.ggml_c.GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

/// Ports `ggml_backend_cuda_device_init_backend` (ggml-cuda.cu:6178 @b1b770d90).
fn deviceInitBackend(dev: ggml.impl.ggml_c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) ggml.impl.ggml_c.ggml_backend_t {
    _ = params;
    return init(deviceCtx(dev).device);
}

/// Ports `ggml_backend_cuda_device_get_buffer_type` (ggml-cuda.cu:6184 @b1b770d90).
fn deviceGetBufferType(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    return bufferType(deviceCtx(dev).device);
}

/// Ports `ggml_backend_cuda_device_get_host_buffer_type` (ggml-cuda.cu:6189 @b1b770d90).
fn deviceGetHostBufferType(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) ggml.impl.ggml_c.ggml_backend_buffer_type_t {
    _ = dev;
    return hostBufferType();
}

/// Ports `ggml_backend_cuda_device_supports_buft` (ggml-cuda.cu:6687 @b1b770d90).
fn deviceSupportsBuft(dev: ggml.impl.ggml_c.ggml_backend_dev_t, buft: ggml.impl.ggml_c.ggml_backend_buffer_type_t) callconv(.c) bool {
    const integrated = ggml.cuda.common.info().devices[@intCast(deviceCtx(dev).device)].integrated;
    return (buftIsCuda(buft) and ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_buffer_type, buft).device == dev) or (integrated and buftIsCudaHost(buft));
}

/// Ports `get_op_batch_size` (ggml-cuda.cu:6693 @b1b770d90).
fn getOpBatchSize(op: *const ggml.impl.ggml_c.ggml_tensor) i64 {
    return switch (op.op) {
        ggml.impl.ggml_c.GGML_OP_GET_ROWS => 0,
        ggml.impl.ggml_c.GGML_OP_MUL_MAT => op.ne[1],
        ggml.impl.ggml_c.GGML_OP_MUL_MAT_ID, ggml.impl.ggml_c.GGML_OP_ROPE, ggml.impl.ggml_c.GGML_OP_ROPE_BACK => op.ne[2],
        else => ggml.impl.ggml_c.ggml_nrows(op),
    };
}

/// Ports `ggml_backend_cuda_device_offload_op` (ggml-cuda.cu:6708 @b1b770d90).
fn deviceOffloadOp(dev: ggml.impl.ggml_c.ggml_backend_dev_t, op: [*c]const ggml.impl.ggml_c.ggml_tensor) callconv(.c) bool {
    return getOpBatchSize(ggml.impl.one(ggml.impl.ggml_c.ggml_tensor, @constCast(op))) >= deviceCtx(dev).op_offload_min_batch_size;
}

/// Ports `ggml_backend_cuda_device_event_new` (ggml-cuda.cu:6714 @b1b770d90).
fn deviceEventNew(dev: ggml.impl.ggml_c.ggml_backend_dev_t) callconv(.c) ggml.impl.ggml_c.ggml_backend_event_t {
    ggml.cuda.common.setDevice(deviceCtx(dev).device);
    var event: ggml.cuda.common.cu.cudaEvent_t = null;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventCreateWithFlags(&event, ggml.cuda.common.cu.cudaEventDisableTiming), "cudaEventCreateWithFlags(&event, cudaEventDisableTiming)", @src());
    const e = std.heap.c_allocator.create(ggml.impl.ggml_c.struct_ggml_backend_event) catch ggml.impl.abort("ggml-cuda: out of memory");
    e.* = .{ .device = dev, .context = event };
    return e;
}

/// Ports `ggml_backend_cuda_device_event_free` (ggml-cuda.cu:6733 @b1b770d90).
fn deviceEventFree(dev: ggml.impl.ggml_c.ggml_backend_dev_t, event: ggml.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    _ = dev;
    const e = ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_event, event);
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventDestroy(@ptrCast(e.context)), "cudaEventDestroy((cudaEvent_t)event->context)", @src());
    std.heap.c_allocator.destroy(e);
}

/// Ports `ggml_backend_cuda_device_event_synchronize` (ggml-cuda.cu:6740 @b1b770d90).
fn deviceEventSynchronize(dev: ggml.impl.ggml_c.ggml_backend_dev_t, event: ggml.impl.ggml_c.ggml_backend_event_t) callconv(.c) void {
    _ = dev;
    ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaEventSynchronize(@ptrCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_event, event).context)), "cudaEventSynchronize((cudaEvent_t)event->context)", @src());
}

/// Ports `ggml_backend_cuda_device_interface` (ggml-cuda.cu:6745 @b1b770d90).
const device_interface = ggml.impl.ggml_c.struct_ggml_backend_device_i{
    .get_name = deviceGetName,
    .get_description = deviceGetDescription,
    .get_memory = deviceGetMemory,
    .get_type = deviceGetType,
    .get_props = deviceGetProps,
    .init_backend = deviceInitBackend,
    .get_buffer_type = deviceGetBufferType,
    .get_host_buffer_type = deviceGetHostBufferType,
    .buffer_from_host_ptr = null,
    .supports_op = ggml.cuda.supports.supportsOp,
    .supports_buft = deviceSupportsBuft,
    .offload_op = deviceOffloadOp,
    .event_new = deviceEventNew,
    .event_free = deviceEventFree,
    .event_synchronize = deviceEventSynchronize,
};

// -----------------------------------------------------------------------------
// Registry

/// Ports `ggml_backend_cuda_reg_context` (ggml-cuda.cu:6765 @b1b770d90).
const RegCtx = struct {
    devices: [ggml.cuda.common.max_devices]ggml.impl.ggml_c.ggml_backend_dev_t = @splat(null),
    count: usize = 0,
};

/// Ports `ggml_backend_cuda_reg_get_name` (ggml-cuda.cu:6769 @b1b770d90).
fn regGetName(r: ggml.impl.ggml_c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    _ = r;
    return ggml.cuda.common.name;
}

fn regCtx(r: ggml.impl.ggml_c.ggml_backend_reg_t) *RegCtx {
    return @ptrCast(@alignCast(ggml.impl.one(ggml.impl.ggml_c.struct_ggml_backend_reg, r).context.?));
}

/// Ports `ggml_backend_cuda_reg_get_device_count` (ggml-cuda.cu:6774 @b1b770d90).
fn regGetDeviceCount(r: ggml.impl.ggml_c.ggml_backend_reg_t) callconv(.c) usize {
    return regCtx(r).count;
}

/// Ports `ggml_backend_cuda_reg_get_device` (ggml-cuda.cu:6779 @b1b770d90).
fn regGetDevice(r: ggml.impl.ggml_c.ggml_backend_reg_t, index: usize) callconv(.c) ggml.impl.ggml_c.ggml_backend_dev_t {
    const ctx = regCtx(r);
    ggml.impl.assert(index < ctx.count, "index < ctx->devices.size()");
    return ctx.devices[index];
}

const Features = struct {
    const fa_quants = std.fmt.comptimePrint("{s}", .{config.cuda_fa_quants});
    var list: [10]ggml.impl.ggml_c.struct_ggml_backend_feature = undefined;
    var arch_list: [8:0]u8 = @splat(0);
    fn run() void {
        var n: usize = 0;
        // `STRINGIFY(__CUDA_ARCH_LIST__)`.
        _ = std.fmt.bufPrintSentinel(&arch_list, "{d}", .{ggml.cuda.common.highestCompiledArch(std.math.maxInt(c_int))}, 0) catch unreachable;
        list[n] = .{ .name = "ARCHS", .value = &arch_list };
        n += 1;
        list[n] = .{ .name = "USE_GRAPHS", .value = "1" };
        n += 1;
        // `GGML_CUDA_FA_QUANTS`, as `ggml_cuda_fattn_vec_instances` defines it.
        list[n] = .{ .name = "FA_QUANTS", .value = fa_quants };
        n += 1;
        const inf = ggml.cuda.common.info();
        var has_fp8 = false;
        var has_fp4 = false;
        for (0..@intCast(inf.device_count)) |id| {
            has_fp8 = has_fp8 or ggml.cuda.common.fp8MmaHardwareAvailable(inf.devices[id].cc);
            has_fp4 = has_fp4 or ggml.cuda.common.blackwellMmaAvailable(inf.devices[id].cc);
            if (has_fp8 and has_fp4) break;
        }
        if (has_fp8) {
            list[n] = .{ .name = "NATIVE_FP8", .value = "1" };
            n += 1;
        }
        if (has_fp4) {
            list[n] = .{ .name = "BLACKWELL_NATIVE_FP4", .value = "1" };
            n += 1;
        }
        list[n] = .{ .name = null, .value = null };
    }
    const once = ggml.cuda.common.Once(run);
};

/// Ports `ggml_backend_cuda_get_features` (ggml-cuda.cu:6785 @b1b770d90).
/// `NATIVE_FP8` comes ahead of `BLACKWELL_NATIVE_FP4`.
fn getFeatures(r: ggml.impl.ggml_c.ggml_backend_reg_t) callconv(.c) [*c]ggml.impl.ggml_c.struct_ggml_backend_feature {
    _ = r;
    Features.once.call();
    return &Features.list;
}

/// Ports `ggml_backend_cuda_reg_get_proc_address` (ggml-cuda.cu:6851 @b1b770d90).
/// `ggml_backend_comm_init_rank` is answered only under
/// `GGML_USE_NCCL`, which is not defined.
fn regGetProcAddress(r: ggml.impl.ggml_c.ggml_backend_reg_t, name_c: [*c]const u8) callconv(.c) ?*anyopaque {
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

/// Ports `ggml_backend_cuda_reg_interface` (ggml-cuda.cu:6879 @b1b770d90).
const reg_interface = ggml.impl.ggml_c.struct_ggml_backend_reg_i{
    .get_name = regGetName,
    .get_device_count = regGetDeviceCount,
    .get_device = regGetDevice,
    .get_proc_address = regGetProcAddress,
};

const Registry = struct {
    var mutex: stdlib.sync.Mutex = .{};
    var value: ggml.impl.ggml_c.struct_ggml_backend_reg = undefined;
    var initialized = false;
    var ctx: RegCtx = .{};
    var devices: [ggml.cuda.common.max_devices]ggml.impl.ggml_c.struct_ggml_backend_device = undefined;
    var device_ctxs: [ggml.cuda.common.max_devices]DeviceCtx = undefined;
};

/// Ports `ggml_backend_cuda_reg` (ggml-cuda.cu:6887 @b1b770d90).
pub fn reg() callconv(.c) ggml.impl.ggml_c.ggml_backend_reg_t {
    Registry.mutex.lock();
    defer Registry.mutex.unlock();
    if (!Registry.initialized) {
        const min_batch_size: c_int = ggml.cuda.common.opOffloadMinBatch();

        const inf = ggml.cuda.common.info();
        const virtual_devices = inf.device_count > inf.physical_device_count;

        for (0..@intCast(inf.device_count)) |i| {
            const physical_id = inf.devices[i].physical_device;

            const dev_ctx = &Registry.device_ctxs[i];
            dev_ctx.* = .{ .device = @intCast(i), .op_offload_min_batch_size = min_batch_size };
            _ = std.fmt.bufPrintSentinel(&dev_ctx.name, ggml.cuda.common.name ++ "{d}", .{i}, 0) catch unreachable;
            _ = deviceDescription(@intCast(i), &dev_ctx.description);

            var pci_bus_id: [32]u8 = @splat(0);
            ggml.cuda.common.cudaCheck(ggml.cuda.common.cu.cudaDeviceGetPCIBusId(&pci_bus_id, pci_bus_id.len, physical_id), "cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id)", @src());
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
            .api_version = ggml.impl.ggml_c.GGML_BACKEND_API_VERSION,
            .iface = reg_interface,
            .context = &Registry.ctx,
        };
    }
    Registry.initialized = true;
    return &Registry.value;
}

/// Ports `ggml_backend_cuda_init` (ggml-cuda.cu:6942 @b1b770d90).
pub fn init(device: c_int) callconv(.c) ggml.impl.ggml_c.ggml_backend_t {
    if (device < 0 or device >= getDeviceCount()) {
        ggml.impl.logError("%s: invalid device %d\n", .{ "ggml_backend_cuda_init", device });
        return null;
    }

    const ctx = ggml.cuda.common.Context.create(device);
    const backend = std.heap.c_allocator.create(ggml.impl.ggml_c.struct_ggml_backend) catch ggml.impl.abort("ggml-cuda: out of memory");
    backend.* = .{
        .guid = &guid,
        .iface = backend_interface,
        .device = ggml.impl.ggml_c.ggml_backend_reg_dev_get(reg(), @intCast(device)),
        .context = ctx,
    };
    return backend;
}

// -----------------------------------------------------------------------------
// The C ABI: `ggml-cuda.h`
