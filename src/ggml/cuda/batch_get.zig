//! Batched device-to-host reads: many small `get_tensor` copies in one
//! gather kernel writing straight into registered host memory.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/batch-get.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/batch-get.cuh` at v0.9.0 (`cd959bae9`),
//! new at v0.9.0, the non-HIP, non-MUSA arm. The kernel,
//! `ggml_cuda_batch_get_kernel`, stays CUDA, in a new top-level `.cu`.
//! The caller is `backend.zig`'s `get_tensor_n`.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const backend = @import("backend.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;

/// `cudaStreamPerThread`.
const stream_per_thread: cu.cudaStream_t = @ptrFromInt(0x2);

/// Ports `ggml_cuda_batch_get_descriptor` (batch-get.cu:8 @cd959bae9): one
/// copy, as the kernel reads it.
const Descriptor = extern struct {
    src: ?[*]const u8,
    dst: ?[*]u8,
    size: usize,
};

comptime {
    // The kernel reads an array of them; three 8-byte fields on aarch64 Linux.
    std.debug.assert(@sizeOf(Descriptor) == 24);
}

/// Ports `ggml_cuda_batch_get_mapping` (batch-get.cu:53 @cd959bae9): the
/// host span registered and mapped for the kernel, unregistered on `deinit`
/// only if this object registered it.
const Mapping = struct {
    host: ?*anyopaque = null,
    device: ?[*]u8 = null,

    /// The C's constructor.
    fn init(host: ?*anyopaque, size: usize) Mapping {
        var self: Mapping = .{};
        const err = cu.cudaHostRegister(host, size, cu.cudaHostRegisterMapped | cu.cudaHostRegisterPortable);
        if (err != cu.cudaSuccess) {
            _ = cu.cudaGetLastError();
            impl.logDebug("%s: failed to register %.2f MiB of pinned memory: %s\n", .{
                "ggml_cuda_batch_get_mapping",
                @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0,
                cu.cudaGetErrorString(err),
            });
            return self;
        }
        self.host = host;
        var dev: ?*anyopaque = null;
        if (cu.cudaHostGetDevicePointer(&dev, host, 0) != cu.cudaSuccess) {
            _ = cu.cudaGetLastError();
            self.device = null;
            return self;
        }
        self.device = @ptrCast(dev);
        return self;
    }

    /// The C's destructor.
    fn deinit(self: *Mapping) void {
        if (self.host) |h| {
            // Release only the registration acquired by this object.
            common.cudaCheck(cu.cudaHostUnregister(h), "cudaHostUnregister(host)", @src());
        }
    }
};

/// A `ggml_backend_buffer_ptr`: frees its buffer at the end of its scope.
fn freeBuffer(buf: c.ggml_backend_buffer_t) void {
    if (buf != null) c.ggml_backend_buffer_free(buf);
}

const Range = struct { first: usize, second: usize };

/// `std::pair`'s `operator<`.
fn rangeLess(_: void, a: Range, b: Range) bool {
    return a.first < b.first or (a.first == b.first and a.second < b.second);
}

/// Ports `ggml_cuda_get_tensor_n` (batch-get.cu:85 @cd959bae9).
///
/// Parameters:
/// - `device`: the buffer's device, already current.
/// - `tensors`, `data`, `offsets`, `sizes`, `n_tensors`: the
///   `get_tensor_n` request.
///
/// Return: true when every read was done here (gathered or by DMA into
/// the registered span), false when the caller must do them one at a time.
pub fn getTensorN(device: c_int, tensors: [*c]const [*c]const Tensor, data: [*c]const ?*anyopaque, offsets: [*c]const usize, sizes: [*c]const usize, n_tensors: usize) bool {
    // Provisional dispatch values based on experiments.
    const min_batch_reads: usize = 128;
    const max_gather_read_bytes: usize = 128 * 1024;
    // Collect non-empty CPU destination ranges.
    var ranges: std.ArrayList(Range) = .empty;
    defer ranges.deinit(common.allocator);
    ranges.ensureTotalCapacity(common.allocator, n_tensors) catch impl.abort("ggml-cuda: out of memory");
    var payload_bytes: usize = 0;
    // Number of non-empty reads whose size is at most max_gather_read_bytes.
    var n_reads_small: usize = 0;
    for (0..n_tensors) |i| {
        if (sizes[i] == 0) {
            continue;
        }
        const address: usize = @intFromPtr(data[i]);
        impl.assert(address <= std.math.maxInt(usize) - sizes[i], "address <= UINTPTR_MAX - sizes[i]");
        impl.assert(payload_bytes <= std.math.maxInt(usize) - sizes[i], "payload_bytes <= SIZE_MAX - sizes[i]");
        payload_bytes += sizes[i];
        n_reads_small += @intFromBool(sizes[i] <= max_gather_read_bytes);
        ranges.appendAssumeCapacity(.{ .first = address, .second = address + sizes[i] });
    }
    if (ranges.items.len == 0) {
        return true;
    }
    // Total number of non-empty reads.
    const n_reads = ranges.items.len;
    if (n_reads > std.math.maxInt(usize) / @sizeOf(Descriptor) or n_reads > std.math.maxInt(c_int)) {
        return false;
    }
    if (n_reads < 2 or (n_reads < min_batch_reads and n_reads_small == n_reads)) {
        return false;
    }
    std.mem.sort(Range, ranges.items, {}, rangeLess);
    for (1..ranges.items.len) |i| {
        // Preserve the ordered copy path when CPU destinations overlap.
        if (ranges.items[i].first < ranges.items[i - 1].second) {
            return false;
        }
    }
    // Find the CPU span to register, including gaps between destinations.
    const begin = ranges.items[0].first;
    const end = ranges.items[ranges.items.len - 1].second;
    if (end -% begin -% payload_bytes > payload_bytes) {
        return false;
    }
    // Register and map the existing CPU destination so the kernel can write to it directly.
    var mapping = Mapping.init(@ptrFromInt(begin), end - begin);
    defer mapping.deinit();
    if (mapping.host == null) {
        return false;
    }
    // Reads selected for gather; zero means all reads use DMA.
    var n_reads_gather: usize = if (mapping.device != null and n_reads_small >= min_batch_reads) n_reads_small else 0;
    // Keep descriptor storage alive until the final stream synchronization.
    const max_batch_descriptors: usize = 128 * 1024;
    const bytes: usize = @as(usize, @min(n_reads_gather, max_batch_descriptors)) * @sizeOf(Descriptor);
    const descriptors: c.ggml_backend_buffer_t = if (n_reads_gather != 0) c.ggml_backend_buft_alloc_buffer(backend.bufferType(device), bytes) else null;
    defer freeBuffer(descriptors);
    if (descriptors == null) {
        n_reads_gather = 0;
    }
    const host_type = backend.hostBufferType();
    const host_descriptors: c.ggml_backend_buffer_t = if (n_reads_gather != 0) c.ggml_backend_buft_alloc_buffer(host_type, bytes) else null;
    defer freeBuffer(host_descriptors);
    if (host_descriptors == null or c.ggml_backend_buffer_get_type(host_descriptors) != host_type) {
        n_reads_gather = 0;
    }
    if (n_reads_gather != 0) {
        const copies: [*]Descriptor = @ptrCast(@alignCast(c.ggml_backend_buffer_get_base(host_descriptors)));
        const device_copies: [*]Descriptor = @ptrCast(@alignCast(c.ggml_backend_buffer_get_base(descriptors)));
        const threads = 256;
        const nsm: usize = @intCast(common.info().devices[@intCast(device)].nsm);
        var i: usize = 0;
        var first: usize = 0;
        while (first < n_reads_gather) : (first += max_batch_descriptors) {
            const n_batch: usize = @min(n_reads_gather - first, max_batch_descriptors);
            var count: usize = 0;
            while (i < n_tensors and count < n_batch) : (i += 1) {
                if (sizes[i] == 0 or sizes[i] > max_gather_read_bytes) {
                    continue;
                }
                copies[count] = .{
                    .src = @as([*]const u8, @ptrCast(impl.one(Tensor, @constCast(tensors[i])).data.?)) + offsets[i],
                    .dst = mapping.device.? + (@intFromPtr(data[i]) - begin),
                    .size = sizes[i],
                };
                count += 1;
            }
            const blocks_per_copy: usize = @min(@as(usize, std.math.maxInt(u16)), @max(@as(usize, 1), (nsm + n_batch - 1) / n_batch));
            const blocks: kernels.Dim3 = .{ .x = @truncate(n_batch), .y = @truncate(blocks_per_copy), .z = 1 };
            // Stream ordering keeps each kernel ahead of the next descriptor overwrite.
            common.cudaCheck(cu.cudaMemcpyAsync(device_copies, copies, n_batch * @sizeOf(Descriptor), cu.cudaMemcpyHostToDevice, stream_per_thread), "cudaMemcpyAsync(device_copies, copies, n_batch * sizeof(copies[0]), cudaMemcpyHostToDevice, cudaStreamPerThread)", @src());
            kernels.launch(kernels.get("batch-get", "ggml_cuda_batch_get_kernel"), .{
                .grid = blocks,
                .block = .{ .x = threads },
                .shmem = 0,
                .stream = stream_per_thread,
            }, .{ @as(?[*]const Descriptor, device_copies), n_batch });
            common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
            if (first + n_batch < n_reads_gather) {
                // Wait before refilling the pinned host descriptor buffer.
                common.cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
            }
        }
    }
    // Keep the destination pinned for DMA too; overlapping destinations already declined above.
    for (0..n_tensors) |i| {
        if (sizes[i] != 0 and (n_reads_gather == 0 or sizes[i] > max_gather_read_bytes)) {
            const src: [*]const u8 = @as([*]const u8, @ptrCast(impl.one(Tensor, @constCast(tensors[i])).data.?)) + offsets[i];
            common.cudaCheck(cu.cudaMemcpyAsync(data[i], src, sizes[i], cu.cudaMemcpyDeviceToHost, stream_per_thread), "cudaMemcpyAsync(data[i], (const char *) tensors[i]->data + offsets[i], sizes[i], cudaMemcpyDeviceToHost, cudaStreamPerThread)", @src());
        }
    }
    // Wait for GPU writes before releasing the CPU mapping and descriptor buffers.
    common.cudaCheck(cu.cudaStreamSynchronize(stream_per_thread), "cudaStreamSynchronize(cudaStreamPerThread)", @src());
    return true;
}
