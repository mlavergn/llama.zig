//! The CUDA host layer's shared state: error checks, device discovery, the
//! two memory pools and the backend context.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/common.cuh` (the host half:
//! the compute-capability predicates, `init_fastdiv_values`,
//! `ggml_cuda_device_info`, `ggml_cuda_pool`, `ggml_cuda_pool_alloc`,
//! `ggml_cuda_graph`, `ggml_cuda_concurrent_event`,
//! `ggml_backend_cuda_context`) and `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`
//! (`ggml_cuda_error`, device selection, `ggml_cuda_init`, both pools and the
//! context destructor), at v0.4.1 (`b29c606e2`).
//!
//! # What is not here
//!
//! The device half of `common.cuh` -- warp reductions, `fastdiv` itself,
//! the type traits' `qk`/`qr`/`qi` -- is device code and compiles with the
//! kernels. HIP and MUSA arms are not ported: this build is CUDA only, and
//! their `#if` arms are cited nowhere below.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const impl = @import("../impl.zig");

/// The translated CUDA headers; see `src/ggml/cuda/c_import.h`.
pub const cu = @import("cuda_c");
const c = impl.c;
const Tensor = c.ggml_tensor;

/// The C `new`s every host-side object and nothing hands it an allocator;
/// libc's is what a C entry point can reach, as in `metal/backend.zig`.
pub const allocator = std.heap.c_allocator;

/// Ports `GGML_CUDA_NAME` (ggml-cuda.h:11 @b29c606e2).
pub const name = "CUDA";

/// Ports `GGML_CUDA_MAX_DEVICES` (ggml-cuda.h:20 @b29c606e2).
pub const max_devices = 16;

/// Ports `GGML_CUDA_MAX_STREAMS` (common.cuh:188 @b29c606e2).
pub const max_streams = 8;

/// Ports `WARP_SIZE` (common.cuh:46 @b29c606e2).
pub const warp_size = 32;

/// Ports `MATRIX_ROW_PADDING` (common.cuh:186 @b29c606e2): the last row of
/// a quantized matrix is padded to a multiple of this, so kernels can read
/// past it.
pub const matrix_row_padding = 512;

// -----------------------------------------------------------------------------
// Compute capabilities

/// Ports the `GGML_CUDA_CC_*` constants this build's NVIDIA arms compare
/// against, from `GGML_CUDA_CC_PASCAL` (common.cuh:50 @b29c606e2).
pub const cc = struct {
    pub const pascal = 600;
    pub const dp4a = 610;
    pub const volta = 700;
    pub const turing = 750;
    pub const ampere = 800;
    pub const orin = 870;
    pub const ada_lovelace = 890;
    pub const hopper = 900;
    pub const blackwell = 1200;
    pub const dgx_spark = 1210;
    pub const rubin = 1300;
    pub const offset_amd = 0x1000000;
    pub const offset_mthreads = 0x0100000;

    /// Ports `GGML_CUDA_CC_IS_NVIDIA` (common.cuh:65 @b29c606e2).
    pub fn isNvidia(x: c_int) bool {
        return x < offset_mthreads;
    }

    /// Ports `GGML_CUDA_CC_IS_AMD` (common.cuh:86 @b29c606e2). Always false
    /// on a CUDA device, but the C's predicates test it, so it is kept for
    /// transliteration.
    pub fn isAmd(x: c_int) bool {
        return x >= offset_amd;
    }

    /// Ports `GGML_CUDA_CC_IS_MTHREADS` (common.cuh:109 @b29c606e2).
    pub fn isMthreads(x: c_int) bool {
        return x >= offset_mthreads and x < offset_amd;
    }
};

/// The architectures the fatbins were built for, as `__CUDA_ARCH_LIST__`
/// spells them: `121a` becomes 1210.
const compiled_archs = blk: {
    const arch = config.cuda_arch;
    var digits: usize = 0;
    while (digits < arch.len and std.ascii.isDigit(arch[digits])) digits += 1;
    const n = std.fmt.parseInt(c_int, arch[0..digits], 10) catch @compileError("bad cuda_arch " ++ arch);
    break :blk [_]c_int{n * 10};
};

/// Ports `ggml_cuda_highest_compiled_arch` (common.cuh:175 @b29c606e2), the
/// `__CUDA_ARCH_LIST__` arm: the highest compiled architecture not above
/// `arch`, or -1.
pub fn highestCompiledArch(arch: c_int) c_int {
    var cur: c_int = 0;
    for (compiled_archs) |first| {
        if (first <= arch and first > cur) cur = first;
    }
    return if (cur == 0) -1 else cur;
}

/// Ports `fp16_available` (common.cuh:308 @b29c606e2).
pub fn fp16Available(x: c_int) bool {
    return highestCompiledArch(x) >= cc.pascal or (cc.isMthreads(x) and x >= cc.offset_mthreads + 0x310);
}

/// Ports `fast_fp16_available` (common.cuh:313 @b29c606e2).
pub fn fastFp16Available(x: c_int) bool {
    return cc.isAmd(x) or (cc.isNvidia(x) and fp16Available(x) and highestCompiledArch(x) != 610) or
        (cc.isMthreads(x) and fp16Available(x));
}

/// Ports `fast_fp16_hardware_available` (common.cuh:320 @b29c606e2).
pub fn fastFp16HardwareAvailable(x: c_int) bool {
    return (cc.isNvidia(x) and x >= cc.pascal and x != 610) or cc.isAmd(x) or
        (cc.isMthreads(x) and x >= cc.offset_mthreads + 0x220);
}

/// Ports `fp16_mma_hardware_available` (common.cuh:326 @b29c606e2), NVIDIA
/// arm; the CDNA/RDNA tests are false on a CUDA device.
pub fn fp16MmaHardwareAvailable(x: c_int) bool {
    return (cc.isNvidia(x) and x >= cc.volta) or (cc.isMthreads(x) and x >= cc.offset_mthreads + 0x220);
}

/// Ports `fast_bf16_hardware_available` (common.cuh:333 @b29c606e2), NVIDIA
/// arm; the AMD test is false on a CUDA device. For feature selection of
/// external libraries, e.g. cuBLAS.
pub fn fastBf16HardwareAvailable(x: c_int) bool {
    return cc.isNvidia(x) and x >= cc.ampere;
}

/// Ports `bf16_mma_hardware_available` (common.cuh:338 @b29c606e2), NVIDIA
/// arm.
pub fn bf16MmaHardwareAvailable(x: c_int) bool {
    return (cc.isNvidia(x) and x >= cc.ampere) or x >= cc.offset_amd + 0x1100 or
        (cc.isMthreads(x) and x >= cc.offset_mthreads + 0x310);
}

/// Ports `volta_mma_available` (common.cuh:360 @b29c606e2).
pub fn voltaMmaAvailable(x: c_int) bool {
    return cc.isNvidia(x) and highestCompiledArch(x) == cc.volta;
}

/// Ports `turing_mma_available` (common.cuh:364 @b29c606e2).
pub fn turingMmaAvailable(x: c_int) bool {
    return cc.isNvidia(x) and highestCompiledArch(x) >= cc.turing;
}

/// Ports `ampere_mma_available` (common.cuh:368 @b29c606e2).
pub fn ampereMmaAvailable(x: c_int) bool {
    return cc.isNvidia(x) and highestCompiledArch(x) >= cc.ampere;
}

/// Ports `cp_async_available` (common.cuh:372 @b29c606e2).
pub fn cpAsyncAvailable(x: c_int) bool {
    return cc.isNvidia(x) and highestCompiledArch(x) >= cc.ampere;
}

/// Ports `blackwell_mma_available` (common.cuh:376 @b29c606e2).
pub fn blackwellMmaAvailable(x: c_int) bool {
    return cc.isNvidia(x) and highestCompiledArch(x) >= cc.blackwell and highestCompiledArch(x) < cc.rubin;
}

/// Ports `ggml_cuda_is_aligned` (common.cuh:382 @b29c606e2).
pub fn isAligned(t: *const Tensor, alignment: usize) bool {
    return @intFromPtr(t.data) % alignment == 0 and
        t.nb[1] % alignment == 0 and t.nb[2] % alignment == 0 and t.nb[3] % alignment == 0;
}

/// Ports `init_fastdiv_values` (common.cuh:925 @b29c606e2): `<mp, L, d>`
/// such that `n / d == (mulhi(n, mp) + n) >> L`.
pub fn initFastdivValues(d_64: u64) cu.uint3 {
    impl.assert(d_64 != 0, "d_64 != 0");
    impl.assert(d_64 <= std.math.maxInt(u32), "d_64 <= std::numeric_limits<uint32_t>::max()");
    const d: u32 = @intCast(d_64);
    var l: u32 = 0;
    while (l < 32 and (@as(u64, 1) << @intCast(l)) < d) l += 1;
    const mp: u32 = @truncate((@as(u64, 1) << 32) * ((@as(u64, 1) << @intCast(l)) - d) / d + 1);
    return .{ .x = mp, .y = l, .z = d };
}

// -----------------------------------------------------------------------------
// Errors

/// Ports `ggml_cuda_error` (ggml-cuda.cu:100 @b29c606e2).
///
/// Parameters:
/// - `stmt`: the call that failed, as the C's `#err` stringifies it.
/// - `src`: where; the C's `__func__`, `__FILE__`, `__LINE__`.
/// - `msg`: the library's description of the error.
pub fn cudaError(stmt: [*:0]const u8, src: std.builtin.SourceLocation, msg: [*:0]const u8) noreturn {
    var id: c_int = -1;
    _ = cu.cudaGetDevice(&id);
    impl.logError(name ++ " error: %s\n", .{msg});
    impl.logError("  current device: %d, in function %s at %s:%d\n", .{ id, src.fn_name.ptr, src.file.ptr, @as(c_int, @intCast(src.line)) });
    impl.logError("  %s\n", .{stmt});
    impl.abort(name ++ " error");
}

/// Ports `CUDA_CHECK` (common.cuh:201 @b29c606e2).
pub fn cudaCheck(err: cu.cudaError_t, stmt: [*:0]const u8, src: std.builtin.SourceLocation) void {
    if (err != cu.cudaSuccess) cudaError(stmt, src, cu.cudaGetErrorString(err));
}

/// Ports `CUBLAS_CHECK` (common.cuh:225 @b29c606e2), with
/// `cublas_get_error_str`'s CUDA 12 arm.
pub fn cublasCheck(err: cu.cublasStatus_t, stmt: [*:0]const u8, src: std.builtin.SourceLocation) void {
    if (err != cu.CUBLAS_STATUS_SUCCESS) cudaError(stmt, src, cu.cublasGetStatusString(err));
}

/// Ports `CU_CHECK` (common.cuh:237 @b29c606e2), with `cu_get_error_str`.
pub fn cuCheck(err: cu.CUresult, stmt: [*:0]const u8, src: std.builtin.SourceLocation) void {
    if (err != cu.CUDA_SUCCESS) {
        var str: [*c]const u8 = null;
        _ = cu.cuGetErrorString(err, &str);
        cudaError(stmt, src, if (str) |s| s else "unknown error");
    }
}

/// A C++ function-local `static` initialiser: runs `f` exactly once,
/// whichever thread gets there first, and makes its effects visible to
/// every caller. Zig 0.17's std has no `once`.
pub fn Once(comptime f: fn () void) type {
    return struct {
        var done: std.atomic.Value(bool) = .init(false);
        var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;

        pub fn call() void {
            if (done.load(.acquire)) return;
            _ = std.c.pthread_mutex_lock(&mutex);
            defer _ = std.c.pthread_mutex_unlock(&mutex);
            if (done.load(.monotonic)) return;
            f();
            done.store(true, .release);
        }
    };
}

/// `atoi` over a C string.
pub fn atoi(s: [*:0]const u8) c_int {
    return std.c.atoi(@ptrCast(s));
}

/// `GGML_ABORT` with a runtime message, for the few places the C formats
/// one.
pub fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    var buf: [512]u8 = undefined;
    const msg = std.fmt.bufPrintSentinel(&buf, fmt, args, 0) catch buf[0 .. buf.len - 1 :0];
    impl.logError("%s\n", .{msg.ptr});
    impl.abort("fatal error");
}

// -----------------------------------------------------------------------------
// Devices

/// Ports `ggml_cuda_device_info`'s `cuda_device_info` (common.cuh:1181 @b29c606e2).
pub const DeviceInfo = struct {
    /// Compute capability.
    cc: c_int = 0,
    /// Streaming multiprocessors.
    nsm: c_int = 0,
    /// Max. shared memory per block.
    smpb: usize = 0,
    /// Max. shared memory per block with opt-in.
    smpbo: usize = 0,
    integrated: bool = false,
    vmm: bool = false,
    vmm_granularity: usize = 0,
    total_vram: usize = 0,
    warp_size: c_int = 0,
    supports_cooperative_launch: bool = false,
    physical_device: c_int = 0,
    physical_share_count: c_int = 0,
    virtual_index: c_int = 0,
};

/// Ports `ggml_cuda_device_info` (common.cuh:1177 @b29c606e2).
pub const Info = struct {
    /// Number of (possibly virtual) devices exposed to the rest of ggml.
    device_count: c_int = 0,
    /// Number of physical CUDA devices actually present.
    physical_device_count: c_int = 0,
    devices: [max_devices]DeviceInfo = @splat(.{}),
    default_tensor_split: [max_devices]f32 = @splat(0),
};

/// Ports `ggml_cuda_info` (ggml-cuda.cu:410 @b29c606e2): the device table,
/// built on first use.
pub fn info() *const Info {
    const S = struct {
        var value: Info = .{};
        fn run() void {
            value = init();
        }
        const once = Once(run);
    };
    S.once.call();
    return &S.value;
}

/// Ports `ggml_cuda_init` (ggml-cuda.cu:220 @b29c606e2), the NVIDIA arms.
fn init() Info {
    var inf: Info = .{};

    const err = cu.cudaGetDeviceCount(&inf.physical_device_count);
    if (err != cu.cudaSuccess) {
        impl.logError("%s: failed to initialize " ++ name ++ ": %s\n", .{ "ggml_cuda_init", cu.cudaGetErrorString(err) });
        return inf;
    }

    impl.assert(inf.physical_device_count <= max_devices, "info.physical_device_count <= GGML_CUDA_MAX_DEVICES");

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can
    // request a different number of (virtual) devices to emulate multi-GPU
    // systems on a machine with fewer GPUs
    inf.device_count = inf.physical_device_count;

    if (std.c.getenv("GGML_CUDA_DEVICES")) |devices_env| {
        if (inf.physical_device_count > 0) {
            const requested = atoi(devices_env);
            if (requested > 0) {
                inf.device_count = requested;
            } else {
                impl.logWarn("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", .{ "ggml_cuda_init", devices_env });
            }
        }
    }

    if (inf.device_count > max_devices) {
        impl.logWarn("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n", .{ "ggml_cuda_init", inf.device_count, @as(c_int, max_devices) });
        inf.device_count = max_devices;
    }

    // map each (virtual) device to a backing physical device (round-robin)
    var physical_share_count: [max_devices]c_int = @splat(0);
    impl.assert(inf.device_count == 0 or inf.physical_device_count > 0, "info.device_count == 0 || info.physical_device_count > 0");
    for (0..@intCast(inf.device_count)) |id| {
        const phys: usize = @intCast(@mod(@as(c_int, @intCast(id)), inf.physical_device_count));
        inf.devices[id].physical_device = @intCast(phys);
        inf.devices[id].virtual_index = physical_share_count[phys];
        physical_share_count[phys] += 1;
    }

    var total_vram: i64 = 0;
    for (0..@intCast(inf.physical_device_count)) |id| {
        var prop: cu.cudaDeviceProp = undefined;
        cudaCheck(cu.cudaGetDeviceProperties(&prop, @intCast(id)), "cudaGetDeviceProperties(&prop, id)", @src());
        total_vram += @intCast(prop.totalGlobalMem);
    }
    impl.logInfo("%s: found %d " ++ name ++ " devices (Total VRAM: %zu MiB):\n", .{ "ggml_cuda_init", inf.physical_device_count, @as(usize, @intCast(@divTrunc(total_vram, 1024 * 1024))) });
    if (inf.device_count != inf.physical_device_count) {
        impl.logInfo("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n", .{ "ggml_cuda_init", inf.device_count, inf.physical_device_count });
    }
    total_vram = 0;

    // The C keeps a vector of Turing devices without tensor cores and logs
    // them after the loop, behind `ggml_cuda_highest_compiled_arch(TURING)
    // >= TURING`; up to max_devices of them fit here.
    var turing_ids: [max_devices]c_int = undefined;
    var turing_names: [max_devices][256]u8 = undefined;
    var turing_count: usize = 0;

    var default_split: [max_devices]f64 = @splat(0);
    for (0..@intCast(inf.device_count)) |id| {
        const physical_id = inf.devices[id].physical_device;

        var device_vmm: c_int = 0;
        var device: cu.CUdevice = 0;
        cuCheck(cu.cuDeviceGet(&device, physical_id), "cuDeviceGet(&device, physical_id)", @src());
        cuCheck(cu.cuDeviceGetAttribute(&device_vmm, cu.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device), "cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device)", @src());

        if (device_vmm != 0) {
            var alloc_prop: cu.CUmemAllocationProp = std.mem.zeroes(cu.CUmemAllocationProp);
            alloc_prop.type = cu.CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = cu.CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            cuCheck(cu.cuMemGetAllocationGranularity(&inf.devices[id].vmm_granularity, &alloc_prop, cu.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)", @src());
        }
        inf.devices[id].vmm = device_vmm != 0;

        var prop: cu.cudaDeviceProp = undefined;
        cudaCheck(cu.cudaGetDeviceProperties(&prop, physical_id), "cudaGetDeviceProperties(&prop, physical_id)", @src());

        // a virtual device owns only a share of its physical GPU's memory
        const share = physical_share_count[@intCast(physical_id)];
        impl.assert(share > 0, "physical_share_count[physical_id] > 0");
        inf.devices[id].physical_share_count = share;
        const device_vram: usize = prop.totalGlobalMem / @as(usize, @intCast(share));
        const device_vram_mib: usize = device_vram / (1024 * 1024);

        // `default_tensor_split` is a `std::array<float>` the C fills with
        // an `int64_t` running total and later divides in place, so each
        // entry is that total rounded to float, then the float quotient.
        default_split[id] = @as(f64, @as(f32, @floatFromInt(total_vram)));
        total_vram += @intCast(device_vram);
        // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        inf.devices[id].integrated = false;
        inf.devices[id].nsm = prop.multiProcessorCount;
        inf.devices[id].smpb = prop.sharedMemPerBlock;
        inf.devices[id].warp_size = prop.warpSize;

        var supports_coop_launch: c_int = 0;
        cudaCheck(cu.cudaDeviceGetAttribute(&supports_coop_launch, cu.cudaDevAttrCooperativeLaunch, physical_id), "cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id)", @src());
        inf.devices[id].supports_cooperative_launch = supports_coop_launch != 0;

        inf.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        inf.devices[id].cc = 100 * prop.major + 10 * prop.minor;
        impl.logInfo("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n", .{ @as(c_int, @intCast(id)), @as([*:0]const u8, @ptrCast(&prop.name)), prop.major, prop.minor, @as([*:0]const u8, if (device_vmm != 0) "yes" else "no"), device_vram_mib });

        const device_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&prop.name)), 0);
        if (std.mem.eql(u8, device_name, "NVIDIA GeForce MX450") or
            std.mem.eql(u8, device_name, "NVIDIA GeForce MX550") or
            std.mem.startsWith(u8, device_name, "NVIDIA GeForce GTX 16"))
        {
            turing_ids[turing_count] = @intCast(id);
            const n = @min(device_name.len, 255);
            @memcpy(turing_names[turing_count][0..n], device_name[0..n]);
            turing_names[turing_count][n] = 0;
            turing_count += 1;
        }

        // Temporary performance fix: device scheduling strategy "spinning"
        // for iGPUs with cc121, to avoid delays in cuda synchronize calls.
        if (prop.major == 12 and prop.minor == 1) {
            cudaCheck(cu.cudaSetDevice(physical_id), "cudaSetDevice(physical_id)", @src());
            cudaCheck(cu.cudaSetDeviceFlags(cu.cudaDeviceScheduleSpin), "cudaSetDeviceFlags(cudaDeviceScheduleSpin)", @src());
        }
    }

    if (highestCompiledArch(cc.turing) >= cc.turing and turing_count != 0) {
        impl.logInfo("The following devices will have suboptimal performance due to a lack of tensor cores:\n", .{});
        for (0..turing_count) |i| {
            impl.logInfo("  Device %d: %s\n", .{ turing_ids[i], @as([*:0]const u8, @ptrCast(&turing_names[i])) });
        }
        impl.logInfo("Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n", .{});
    }

    for (0..@intCast(inf.device_count)) |id| {
        // `float /= int64_t` promotes the float to float, then divides.
        const total: f32 = @floatFromInt(total_vram);
        inf.default_tensor_split[id] = @as(f32, @floatCast(default_split[id])) / total;
    }

    if (std.c.getenv("GGML_CUDA_P2P") != null) {
        for (0..@intCast(inf.physical_device_count)) |id| {
            cudaCheck(cu.cudaSetDevice(@intCast(id)), "cudaSetDevice(id)", @src());
            for (0..@intCast(inf.physical_device_count)) |id_other| {
                if (id == id_other) continue;
                var can_access_peer: c_int = 0;
                cudaCheck(cu.cudaDeviceCanAccessPeer(&can_access_peer, @intCast(id), @intCast(id_other)), "cudaDeviceCanAccessPeer(&can_access_peer, id, id_other)", @src());
                if (can_access_peer != 0) {
                    cudaCheck(cu.cudaDeviceEnablePeerAccess(@intCast(id_other), 0), "cudaDeviceEnablePeerAccess(id_other, 0)", @src());
                }
            }
        }
    }

    return inf;
}

/// Ports `ggml_cuda_get_physical_device` (ggml-cuda.cu:112 @b29c606e2).
pub fn physicalDevice(device: c_int) c_int {
    const inf = info();
    impl.assert(device >= 0 and device < inf.device_count, "device >= 0 && device < info.device_count");
    return inf.devices[@intCast(device)].physical_device;
}

/// Ports `ggml_cuda_set_device` (ggml-cuda.cu:120 @b29c606e2).
pub fn setDevice(device: c_int) void {
    const physical_device = physicalDevice(device);
    var current_device: c_int = 0;
    cudaCheck(cu.cudaGetDevice(&current_device), "cudaGetDevice(&current_device)", @src());
    if (physical_device == current_device) return;
    cudaCheck(cu.cudaSetDevice(physical_device), "cudaSetDevice(physical_device)", @src());
}

/// Ports `ggml_cuda_get_device` (ggml-cuda.cu:134 @b29c606e2).
pub fn getDevice() c_int {
    var id: c_int = 0;
    cudaCheck(cu.cudaGetDevice(&id), "cudaGetDevice(&id)", @src());
    return id;
}

/// Ports `ggml_cuda_device_malloc` (ggml-cuda.cu:140 @b29c606e2), CUDA arm.
pub fn deviceMalloc(ptr: *?*anyopaque, size: usize, device: c_int) cu.cudaError_t {
    setDevice(device);
    if (std.c.getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != null) {
        return cu.cudaMallocManaged(ptr, size, cu.cudaMemAttachGlobal);
    }
    return cu.cudaMalloc(ptr, size);
}

// -----------------------------------------------------------------------------
// Pools

/// Ports `ggml_cuda_pool` (common.cuh:1207 @b29c606e2). The C's virtual
/// base becomes a tagged union of its two implementations.
pub const Pool = union(enum) {
    leg: PoolLeg,
    vmm: PoolVmm,

    pub fn alloc(self: *Pool, size: usize, actual_size: *usize) ?*anyopaque {
        return switch (self.*) {
            inline else => |*p| p.alloc(size, actual_size),
        };
    }

    pub fn free(self: *Pool, ptr: ?*anyopaque, size: usize) void {
        switch (self.*) {
            inline else => |*p| p.free(ptr, size),
        }
    }

    pub fn deinit(self: *Pool) void {
        switch (self.*) {
            inline else => |*p| p.deinit(),
        }
    }
};

/// Ports `ggml_cuda_pool_leg` (ggml-cuda.cu:418 @b29c606e2): a best-fit
/// cache of `cudaMalloc`ed buffers.
pub const PoolLeg = struct {
    const max_buffers = 256;
    const Buffer = struct { ptr: ?*anyopaque = null, size: usize = 0 };

    device: c_int,
    buffer_pool: [max_buffers]Buffer = @splat(.{}),
    pool_size: usize = 0,

    fn deinit(self: *PoolLeg) void {
        self.clearPool();
        impl.assert(self.pool_size == 0, "pool_size == 0");
    }

    fn clearPool(self: *PoolLeg) void {
        setDevice(self.device);
        for (&self.buffer_pool) |*b| {
            if (b.ptr != null) {
                cudaCheck(cu.cudaFree(b.ptr), "cudaFree(b.ptr)", @src());
                self.pool_size -= b.size;
                b.* = .{};
            }
        }
    }

    fn alloc(self: *PoolLeg, size: usize, actual_size: *usize) ?*anyopaque {
        var best_diff: usize = 1 << 36;
        var ibest: ?usize = null;
        for (&self.buffer_pool, 0..) |*b, i| {
            if (b.ptr != null and b.size >= size) {
                const diff = b.size - size;
                if (diff < best_diff) {
                    best_diff = diff;
                    ibest = i;
                    if (best_diff == 0) {
                        const ptr = b.ptr;
                        actual_size.* = b.size;
                        b.* = .{};
                        return ptr;
                    }
                }
            }
        }
        if (ibest) |i| {
            const b = &self.buffer_pool[i];
            const ptr = b.ptr;
            actual_size.* = b.size;
            b.* = .{};
            return ptr;
        }
        var ptr: ?*anyopaque = null;
        // `(size_t) (1.05 * size)`: the product is a double.
        var look_ahead_size: usize = @intFromFloat(1.05 * @as(f64, @floatFromInt(size)));
        look_ahead_size = 256 * ((look_ahead_size + 255) / 256);
        setDevice(self.device);
        var err = deviceMalloc(&ptr, look_ahead_size, self.device);
        if (err == cu.cudaErrorMemoryAllocation) {
            _ = cu.cudaGetLastError();
            const cached_bytes = self.pool_size;
            impl.logDebug(name ++ " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n", .{ self.device, @as(f64, @floatFromInt(look_ahead_size)) / 1024.0 / 1024.0, @as(f64, @floatFromInt(cached_bytes)) / 1024.0 / 1024.0 });
            cudaCheck(cu.cudaDeviceSynchronize(), "cudaDeviceSynchronize()", @src());
            self.clearPool();
            err = deviceMalloc(&ptr, look_ahead_size, self.device);
            if (err == cu.cudaSuccess) {
                impl.logDebug(name ++ " pool[%d]: retry succeeded\n", .{self.device});
            }
        }
        cudaCheck(err, "err", @src());
        actual_size.* = look_ahead_size;
        self.pool_size += look_ahead_size;
        return ptr;
    }

    fn free(self: *PoolLeg, ptr: ?*anyopaque, size: usize) void {
        for (&self.buffer_pool) |*b| {
            if (b.ptr == null) {
                b.* = .{ .ptr = ptr, .size = size };
                return;
            }
        }
        impl.logDebug(name ++ " buffer pool full, increase MAX_CUDA_BUFFERS\n", .{});
        setDevice(self.device);
        cudaCheck(cu.cudaFree(ptr), "cudaFree(ptr)", @src());
        self.pool_size -= size;
    }
};

/// Ports `ggml_cuda_pool_vmm` (ggml-cuda.cu:535 @b29c606e2): a stack
/// allocator over one reserved virtual range, growing its physical backing
/// on demand. Frees must come in reverse order.
pub const PoolVmm = struct {
    /// 32 GB.
    const max_size: usize = 1 << 35;

    device: c_int,
    physical_device: c_int,
    pool_addr: cu.CUdeviceptr = 0,
    pool_used: usize = 0,
    pool_size: usize = 0,
    granularity: usize,

    fn initFor(device: c_int) PoolVmm {
        return .{
            .device = device,
            .physical_device = physicalDevice(device),
            .granularity = info().devices[@intCast(device)].vmm_granularity,
        };
    }

    fn deinit(self: *PoolVmm) void {
        if (self.pool_addr != 0) {
            cuCheck(cu.cuMemUnmap(self.pool_addr, self.pool_size), "cuMemUnmap(pool_addr, pool_size)", @src());
            cuCheck(cu.cuMemAddressFree(self.pool_addr, max_size), "cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE)", @src());
        }
    }

    fn alloc(self: *PoolVmm, size_in: usize, actual_size: *usize) ?*anyopaque {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        const alignment = 128;
        const size = alignment * ((size_in + alignment - 1) / alignment);

        const avail = self.pool_size - self.pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            var reserve_size = size - avail;
            reserve_size = self.granularity * ((reserve_size + self.granularity - 1) / self.granularity);

            impl.assert(self.pool_size + reserve_size <= max_size, "pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE");

            // allocate more physical memory
            var prop: cu.CUmemAllocationProp = std.mem.zeroes(cu.CUmemAllocationProp);
            prop.type = cu.CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = cu.CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = self.physical_device;
            var h: cu.CUmemGenericAllocationHandle = 0;
            cuCheck(cu.cuMemCreate(&h, reserve_size, &prop, 0), "cuMemCreate(&handle, reserve_size, &prop, 0)", @src());

            // reserve virtual address space (if not already reserved)
            if (self.pool_addr == 0) {
                cuCheck(cu.cuMemAddressReserve(&self.pool_addr, max_size, 0, 0, 0), "cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0)", @src());
            }

            // map at the end of the pool
            const start_ptr: cu.CUdeviceptr = self.pool_addr + self.pool_size;
            cuCheck(cu.cuMemMap(start_ptr, reserve_size, 0, h, 0), "cuMemMap(start_ptr, reserve_size, 0, handle, 0)", @src());

            // the memory allocation handle is no longer needed after mapping
            cuCheck(cu.cuMemRelease(h), "cuMemRelease(handle)", @src());

            const use_peer_access = std.c.getenv("GGML_CUDA_P2P") != null;
            if (use_peer_access) {
                // VMM allocations do not become peer-accessible from
                // cudaDeviceEnablePeerAccess, so access is granted here, on
                // the backing physical devices, deduplicated.
                var access_descs: [max_devices]cu.CUmemAccessDesc = undefined;
                var n_descs: usize = 0;
                var physical_seen: [max_devices]bool = @splat(false);
                const device_count = info().device_count;
                for (0..@intCast(device_count)) |id| {
                    const id_physical = physicalDevice(@intCast(id));
                    if (id_physical != self.physical_device) {
                        var can_access_peer: c_int = 0;
                        cudaCheck(cu.cudaDeviceCanAccessPeer(&can_access_peer, id_physical, self.physical_device), "cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device)", @src());
                        if (can_access_peer == 0) continue;
                    }
                    if (physical_seen[@intCast(id_physical)]) continue;
                    physical_seen[@intCast(id_physical)] = true;
                    var access: cu.CUmemAccessDesc = std.mem.zeroes(cu.CUmemAccessDesc);
                    access.location.type = cu.CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = cu.CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs[n_descs] = access;
                    n_descs += 1;
                }
                cuCheck(cu.cuMemSetAccess(start_ptr, reserve_size, &access_descs, n_descs), "cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size())", @src());
            } else {
                // set access for non P2P
                var access: cu.CUmemAccessDesc = std.mem.zeroes(cu.CUmemAccessDesc);
                access.location.type = cu.CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = self.physical_device;
                access.flags = cu.CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                cuCheck(cu.cuMemSetAccess(start_ptr, reserve_size, &access, 1), "cuMemSetAccess(start_ptr, reserve_size, &access, 1)", @src());
            }

            // add to the pool
            self.pool_size += reserve_size;
        }

        impl.assert(self.pool_addr != 0, "pool_addr != 0");

        const ptr: ?*anyopaque = @ptrFromInt(self.pool_addr + self.pool_used);
        actual_size.* = size;
        self.pool_used += size;
        return ptr;
    }

    fn free(self: *PoolVmm, ptr: ?*anyopaque, size: usize) void {
        self.pool_used -= size;
        // all deallocations must be in reverse order of the allocations
        impl.assert(@intFromPtr(ptr) == self.pool_addr + self.pool_used, "ptr == (void *) ((char *)(pool_addr) + pool_used)");
    }
};

/// Ports `ggml_backend_cuda_context`'s `new_pool_for_device`
/// (ggml-cuda.cu:684 @b29c606e2).
fn newPoolForDevice(device: c_int) *Pool {
    const p = allocator.create(Pool) catch impl.abort("ggml-cuda: out of memory");
    if (info().devices[@intCast(device)].vmm) {
        p.* = .{ .vmm = PoolVmm.initFor(device) };
    } else {
        p.* = .{ .leg = .{ .device = device } };
    }
    return p;
}

/// Ports `ggml_cuda_pool_alloc` (common.cuh:1215 @b29c606e2), for each `T`. The C's
/// destructor is `deinit`, which the caller `defer`s.
pub fn PoolAlloc(comptime T: type) type {
    return struct {
        const Self = @This();

        pool: ?*Pool = null,
        ptr: ?[*]T = null,
        actual_size: usize = 0,

        pub fn init(pool: *Pool) Self {
            return .{ .pool = pool };
        }

        /// The C's sized constructor.
        pub fn initAlloc(pool: *Pool, size: usize) Self {
            var self: Self = .{ .pool = pool };
            _ = self.alloc(size);
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.ptr) |p| self.pool.?.free(@ptrCast(p), self.actual_size);
        }

        /// `size` is in elements.
        pub fn alloc(self: *Self, size: usize) [*]T {
            impl.assert(self.pool != null, "pool != nullptr");
            impl.assert(self.ptr == null, "ptr == nullptr");
            self.ptr = @ptrCast(@alignCast(self.pool.?.alloc(size * @sizeOf(T), &self.actual_size)));
            return self.ptr.?;
        }

        pub fn allocFrom(self: *Self, pool: *Pool, size: usize) [*]T {
            self.pool = pool;
            return self.alloc(size);
        }

        pub fn get(self: *const Self) ?[*]T {
            return self.ptr;
        }
    };
}

// -----------------------------------------------------------------------------
// CUDA graphs

/// Ports `ggml_cuda_graph`'s `node_properties` (common.cuh:1289 @b29c606e2).
pub const NodeProperties = struct {
    node: Tensor,
    node_src_data_ptrs: [c.GGML_MAX_SRC]?*anyopaque,
    node_src_ne: [c.GGML_MAX_SRC][c.GGML_MAX_DIMS]i64,
    node_src_nb: [c.GGML_MAX_SRC][c.GGML_MAX_DIMS]usize,
};

/// Ports `ggml_cuda_graph` (common.cuh:1271 @b29c606e2), with
/// `USE_CUDA_GRAPH` defined: `GGML_CUDA_USE_GRAPHS` is in the reference's
/// flags.
pub const Graph = struct {
    graph: cu.cudaGraph_t = null,
    instance: cu.cudaGraphExec_t = null,
    num_nodes: usize = 0,
    nodes: std.ArrayList(cu.cudaGraphNode_t) = .empty,
    disable_due_to_gpu_arch: bool = false,
    warmup_complete: bool = false,
    uid: u64 = 0,
    last_used_time: i64 = 0,
    node_props: std.ArrayList(NodeProperties) = .empty,

    pub fn deinit(self: *Graph) void {
        if (self.instance != null) cudaCheck(cu.cudaGraphExecDestroy(self.instance), "cudaGraphExecDestroy(instance)", @src());
        if (self.graph != null) cudaCheck(cu.cudaGraphDestroy(self.graph), "cudaGraphDestroy(graph)", @src());
        self.nodes.deinit(allocator);
        self.node_props.deinit(allocator);
    }

    /// Ports `ggml_cuda_graph`'s `is_enabled` (common.cuh:1297 @b29c606e2).
    pub fn isEnabled(self: *const Graph) bool {
        const S = struct {
            var disabled_by_env = false;
            fn run() void {
                disabled_by_env = std.c.getenv("GGML_CUDA_DISABLE_GRAPHS") != null;
            }
            const once = Once(run);
        };
        S.once.call();
        return !(self.disable_due_to_gpu_arch or S.disabled_by_env);
    }
};

// -----------------------------------------------------------------------------
// Concurrent streams

/// Ports `ggml_cuda_concurrent_event` (common.cuh:1304 @b29c606e2).
pub const ConcurrentEvent = struct {
    join_events: std.ArrayList(cu.cudaEvent_t) = .empty,
    fork_event: cu.cudaEvent_t = null,
    n_streams: c_int = 0,
    stream_mapping: std.AutoHashMapUnmanaged(*const Tensor, c_int) = .empty,
    /// Original order of the nodes in this concurrent region, before
    /// interleaving; used to restore grouping for fusion within streams.
    original_order: std.ArrayList(*const Tensor) = .empty,
    join_node: ?*const Tensor = null,

    pub fn init(n_streams: c_int) ConcurrentEvent {
        var self: ConcurrentEvent = .{ .n_streams = n_streams };
        self.join_events.resize(allocator, @intCast(n_streams)) catch impl.abort("ggml-cuda: out of memory");
        for (self.join_events.items) |*e| {
            cudaCheck(cu.cudaEventCreateWithFlags(e, cu.cudaEventDisableTiming), "cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming)", @src());
        }
        cudaCheck(cu.cudaEventCreateWithFlags(&self.fork_event, cu.cudaEventDisableTiming), "cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming)", @src());
        return self;
    }

    pub fn deinit(self: *ConcurrentEvent) void {
        if (self.fork_event != null) cudaCheck(cu.cudaEventDestroy(self.fork_event), "cudaEventDestroy(fork_event)", @src());
        for (self.join_events.items) |e| {
            if (e != null) cudaCheck(cu.cudaEventDestroy(e), "cudaEventDestroy(e)", @src());
        }
        self.join_events.deinit(allocator);
        self.stream_mapping.deinit(allocator);
        self.original_order.deinit(allocator);
    }

    const Range = struct {
        start: i64,
        end: i64,
        fn lessThan(_: void, a: Range, b: Range) bool {
            return a.start < b.start or (a.start == b.start and a.end < b.end);
        }
    };

    fn rangeOf(tensor: *const Tensor) Range {
        const t: *const Tensor = if (tensor.view_src) |v| v else tensor;
        const start: i64 = @intCast(@intFromPtr(t.data));
        return .{ .start = start, .end = start + @as(i64, @intCast(c.ggml_nbytes(t))) };
    }

    /// Ports `ggml_cuda_concurrent_event`'s `is_valid` (common.cuh:1345 @b29c606e2):
    /// whether the branches write disjoint memory and read only within
    /// their own branch.
    pub fn isValid(self: *const ConcurrentEvent) bool {
        var write_ranges = allocator.alloc(std.ArrayList(Range), @intCast(self.n_streams)) catch impl.abort("ggml-cuda: out of memory");
        defer {
            for (write_ranges) |*w| w.deinit(allocator);
            allocator.free(write_ranges);
        }
        for (write_ranges) |*w| w.* = .empty;

        // join_node's range is excluded from overlap checking: several nodes
        // may use its buffer, and the join synchronises on it.
        const join = rangeOf(self.join_node.?);

        var it = self.stream_mapping.iterator();
        while (it.next()) |kv| {
            const r = rangeOf(kv.key_ptr.*);
            if ((r.start <= join.start and join.start < r.end) or (join.start <= r.start and r.start < join.end)) continue;
            // concurrent streams begin from 1
            write_ranges[@intCast(kv.value_ptr.* - 1)].append(allocator, r) catch impl.abort("ggml-cuda: out of memory");
        }

        for (write_ranges) |w| std.mem.sort(Range, w.items, {}, Range.lessThan);

        var writes_overlap = false;
        var dependent_srcs = false;
        it = self.stream_mapping.iterator();
        while (it.next()) |kv| {
            const tensor = kv.key_ptr.*;
            const stream = kv.value_ptr.*;
            const r = rangeOf(tensor);
            if ((r.start <= join.start and join.start < r.end) or (join.start <= r.start and r.start < join.end)) continue;

            for (0..@intCast(self.n_streams)) |i| {
                if (i == stream - 1) continue;
                // std::lower_bound: the first range not less than r.
                const ranges = write_ranges[i].items;
                const pos = std.sort.lowerBound(Range, ranges, r, struct {
                    fn order(key: Range, item: Range) std.math.Order {
                        if (Range.lessThan({}, item, key)) return .gt;
                        return .lt;
                    }
                }.order);
                if (pos != ranges.len and ranges[pos].start < r.end) {
                    impl.logDebug("Writes overlap for %s", .{@as([*:0]const u8, @ptrCast(&tensor.name))});
                    writes_overlap = true;
                    break;
                }
            }

            for (tensor.src) |src| {
                const s = src orelse continue;
                const other = self.stream_mapping.get(s) orelse continue;
                if (other != stream) {
                    dependent_srcs = true;
                    break;
                }
            }

            if (dependent_srcs or writes_overlap) break;
        }

        return !writes_overlap and !dependent_srcs;
    }
};

/// Ports `ggml_cuda_stream_context` (common.cuh:1447 @b29c606e2).
pub const StreamContext = struct {
    concurrent_events: std.AutoArrayHashMapUnmanaged(*const Tensor, ConcurrentEvent) = .empty,

    pub fn reset(self: *StreamContext) void {
        for (self.concurrent_events.values()) |*e| e.deinit();
        self.concurrent_events.clearRetainingCapacity();
    }

    pub fn deinit(self: *StreamContext) void {
        self.reset();
        self.concurrent_events.deinit(allocator);
    }
};

// -----------------------------------------------------------------------------
// The backend context

/// Ports `ggml_backend_cuda_context` (common.cuh:1455 @b29c606e2).
pub const Context = struct {
    device: c_int,
    name_buf: [16:0]u8 = @splat(0),
    copy_event: cu.cudaEvent_t = null,

    streams: [max_devices][max_streams]cu.cudaStream_t = @splat(@splat(null)),
    cublas_handles: [max_devices][max_streams]cu.cublasHandle_t = @splat(@splat(null)),
    cublas_workspaces: [max_devices][max_streams]?*anyopaque = @splat(@splat(null)),
    cublas_workspace_sizes: [max_devices]usize = @splat(0),

    curr_stream_no: c_int = 0,

    /// Map from first_node_ptr to cuda_graph -- multiple graphs per context
    /// when the computation is split across CPU and GPU.
    cuda_graphs: std.AutoArrayHashMapUnmanaged(?*const anyopaque, *Graph) = .empty,
    last_graph_eviction_sweep: i64 = 0,

    concurrent_stream_context: StreamContext = .{},

    pools: [max_devices][max_streams]?*Pool = @splat(@splat(null)),

    /// The C's constructor: `name(GGML_CUDA_NAME + std::to_string(device))`.
    pub fn create(device: c_int) *Context {
        const self = allocator.create(Context) catch impl.abort("ggml-cuda: out of memory");
        self.* = .{ .device = device };
        _ = std.fmt.bufPrintSentinel(&self.name_buf, name ++ "{d}", .{device}, 0) catch unreachable;
        return self;
    }

    pub fn getName(self: *const Context) [*:0]const u8 {
        return &self.name_buf;
    }

    /// Ports the destructor of `ggml_backend_cuda_context` (ggml-cuda.cu:701 @b29c606e2).
    pub fn destroy(self: *Context) void {
        lock.waitIdle();

        if (self.copy_event != null) cudaCheck(cu.cudaEventDestroy(self.copy_event), "cudaEventDestroy(copy_event)", @src());
        for (0..max_devices) |i| {
            for (0..max_streams) |j| {
                if (self.streams[i][j] != null) cudaCheck(cu.cudaStreamDestroy(self.streams[i][j]), "cudaStreamDestroy(streams[i][j])", @src());
                if (self.cublas_handles[i][j] != null) cublasCheck(cu.cublasDestroy(self.cublas_handles[i][j]), "cublasDestroy(cublas_handles[i][j])", @src());
                if (self.cublas_workspaces[i][j] != null) cudaCheck(cu.cudaFree(self.cublas_workspaces[i][j]), "cudaFree(cublas_workspaces[i][j])", @src());
            }
        }
        // The members the C destroys implicitly, after the body: the
        // concurrent stream context, the pools and the CUDA graphs, in
        // reverse declaration order.
        for (0..max_devices) |i| {
            for (0..max_streams) |j| {
                if (self.pools[i][j]) |p| {
                    p.deinit();
                    allocator.destroy(p);
                }
            }
        }
        self.concurrent_stream_context.deinit();
        for (self.cuda_graphs.values()) |g| {
            g.deinit();
            allocator.destroy(g);
        }
        self.cuda_graphs.deinit(allocator);
        allocator.destroy(self);
    }

    /// Ports `ggml_backend_cuda_context`'s `cuda_graph` (common.cuh:1474 @b29c606e2).
    pub fn cudaGraph(self: *Context, first_node_ptr: ?*const anyopaque) *Graph {
        const time_now = c.ggml_time_us();

        // sweep every 5s, evicting cuda graphs unused for >=10s
        if (time_now - self.last_graph_eviction_sweep >= 5_000_000) {
            self.last_graph_eviction_sweep = time_now;
            var i: usize = 0;
            while (i < self.cuda_graphs.count()) {
                const g = self.cuda_graphs.values()[i];
                if (time_now - g.last_used_time >= 10_000_000) {
                    g.deinit();
                    allocator.destroy(g);
                    self.cuda_graphs.swapRemoveAt(i);
                } else {
                    i += 1;
                }
            }
        }

        const gop = self.cuda_graphs.getOrPut(allocator, first_node_ptr) catch impl.abort("ggml-cuda: out of memory");
        if (!gop.found_existing) {
            gop.value_ptr.* = allocator.create(Graph) catch impl.abort("ggml-cuda: out of memory");
            gop.value_ptr.*.* = .{};
        }
        gop.value_ptr.*.last_used_time = time_now;
        return gop.value_ptr.*;
    }

    /// Ports `ggml_backend_cuda_context`'s `any_cuda_graph_enabled` (common.cuh:1499 @b29c606e2).
    pub fn anyCudaGraphEnabled(self: *const Context) bool {
        for (self.cuda_graphs.values()) |g| if (g.isEnabled()) return true;
        return false;
    }

    /// Ports `ggml_backend_cuda_context`'s `any_cuda_graph_has_instance` (common.cuh:1509 @b29c606e2).
    pub fn anyCudaGraphHasInstance(self: *const Context) bool {
        for (self.cuda_graphs.values()) |g| if (g.instance != null) return true;
        return false;
    }

    /// Ports `ggml_backend_cuda_context`'s two-argument `stream` (common.cuh:1528 @b29c606e2).
    pub fn streamFor(self: *Context, device: c_int, stream_no: c_int) cu.cudaStream_t {
        const s = &self.streams[@intCast(device)][@intCast(stream_no)];
        if (s.* == null) {
            setDevice(device);
            cudaCheck(cu.cudaStreamCreateWithFlags(s, cu.cudaStreamNonBlocking), "cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking)", @src());
        }
        return s.*;
    }

    /// Ports `ggml_backend_cuda_context`'s argument-free `stream` (common.cuh:1536 @b29c606e2).
    pub fn stream(self: *Context) cu.cudaStream_t {
        return self.streamFor(self.device, self.curr_stream_no);
    }

    /// Ports `ggml_backend_cuda_context`'s `cublas_handle` (common.cuh:1540 @b29c606e2).
    pub fn cublasHandle(self: *Context) cu.cublasHandle_t {
        const d: usize = @intCast(self.device);
        const s: usize = @intCast(self.curr_stream_no);
        if (self.cublas_handles[d][s] == null) {
            setDevice(self.device);
            cublasCheck(cu.cublasCreate_v2(&self.cublas_handles[d][s]), "cublasCreate(&cublas_handles[device][curr_stream_no])", @src());
            cublasCheck(cu.cublasSetMathMode(self.cublas_handles[d][s], cu.CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode(cublas_handles[device][curr_stream_no], CUBLAS_TF32_TENSOR_OP_MATH)", @src());
            cublasCheck(cu.cublasSetStream_v2(self.cublas_handles[d][s], self.stream()), "cublasSetStream(cublas_handles[device][curr_stream_no], stream())", @src());
            if (self.cublas_workspace_sizes[d] == 0) {
                const x = info().devices[d].cc;
                self.cublas_workspace_sizes[d] = if (x >= cc.hopper) 32 * 1024 * 1024 else 4 * 1024 * 1024;
            }
            cudaCheck(cu.cudaMalloc(&self.cublas_workspaces[d][s], self.cublas_workspace_sizes[d]), "cudaMalloc(&cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device])", @src());
            cublasCheck(cu.cublasSetWorkspace_v2(self.cublas_handles[d][s], self.cublas_workspaces[d][s], self.cublas_workspace_sizes[d]), "cublasSetWorkspace(cublas_handles[device][curr_stream_no], cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device])", @src());
        }
        return self.cublas_handles[d][s];
    }

    /// Ports `ggml_backend_cuda_context`'s per-device `pool` (common.cuh:1563 @b29c606e2).
    pub fn poolFor(self: *Context, device: c_int) *Pool {
        const p = &self.pools[@intCast(device)][@intCast(self.curr_stream_no)];
        if (p.* == null) p.* = newPoolForDevice(device);
        return p.*.?;
    }

    /// Ports `ggml_backend_cuda_context`'s argument-free `pool` (common.cuh:1570 @b29c606e2).
    pub fn pool(self: *Context) *Pool {
        return self.poolFor(self.device);
    }

    /// The device's compute capability, `ggml_cuda_info().devices[ctx.device].cc`.
    pub fn computeCapability(self: *const Context) c_int {
        return info().devices[@intCast(self.device)].cc;
    }
};

/// Ports `ggml_cuda_lock`, `ggml_cuda_lock_cv` and `ggml_cuda_lock_counter`
/// (ggml-cuda.cu:697, 698, 699 @b29c606e2): destroying a cuBLAS handle while a graph
/// is being captured in another thread can be a CUDA error, so the context
/// destructor waits until no capture is in progress.
pub const lock = struct {
    var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
    var cv: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER;
    pub var counter: std.atomic.Value(c_int) = .init(0);

    fn waitIdle() void {
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        while (counter.load(.monotonic) != 0) _ = std.c.pthread_cond_wait(&cv, &mutex);
    }

    /// Ends a capture: `ggml_cuda_lock_counter.fetch_sub(1)` and, at zero,
    /// `notify_all` under the mutex.
    pub fn release() void {
        if (counter.fetchSub(1, .monotonic) == 1) {
            _ = std.c.pthread_mutex_lock(&mutex);
            defer _ = std.c.pthread_mutex_unlock(&mutex);
            _ = std.c.pthread_cond_broadcast(&cv);
        }
    }

    /// Begins a capture.
    pub fn acquire() void {
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        _ = counter.fetchAdd(1, .monotonic);
    }
};

/// Ports `ggml_cuda_mm_fusion_args_host` (common.cuh:1575 @b29c606e2).
/// `glu_limit` is `GGML_GLU_OP_SWIGLU_CLAMP`'s limit, op_params slot 3.
pub const MmFusionArgsHost = struct {
    x_bias: ?*const Tensor = null,
    gate: ?*const Tensor = null,
    gate_bias: ?*const Tensor = null,
    x_scale: ?*const Tensor = null,
    gate_scale: ?*const Tensor = null,
    glu_op: c.enum_ggml_glu_op = 0,
    glu_limit: f32 = 0.0,
};

/// Ports `ggml_cuda_mm_fusion_args_device` (common.cuh:1584 @b29c606e2).
/// Passed to kernels by value, so its layout is the kernel's: the `float`
/// follows the 4-byte enum, and the struct stays 48 bytes.
pub const MmFusionArgsDevice = extern struct {
    x_bias: ?*const anyopaque = null,
    gate: ?*const anyopaque = null,
    gate_bias: ?*const anyopaque = null,
    x_scale: ?*const anyopaque = null,
    gate_scale: ?*const anyopaque = null,
    glu_op: c.enum_ggml_glu_op = 0,
    glu_limit: f32 = 0.0,
};

// -----------------------------------------------------------------------------
// Type traits

/// Ports the `QR*` and `QI*` constants of `ggml-common.h`'s CUDA arm, from
/// `QR1_0` (ggml-common.h:97 @b29c606e2) to the arm's `#endif`: values per quantized unit (`qr`) and
/// 32-bit integers per block (`qi`). `ggml_c` is translated with
/// `GGML_COMMON_DECL_C`, which leaves them out.
pub const q = struct {
    pub const QR1_0 = 1;
    pub const QI1_0 = c.QK1_0 / 32;
    pub const QR2_0 = 1;
    pub const QI2_0 = c.QK2_0 / 32;
    pub const QR4_0 = 2;
    pub const QI4_0 = c.QK4_0 / (4 * QR4_0);
    pub const QR4_1 = 2;
    pub const QI4_1 = c.QK4_1 / (4 * QR4_1);
    pub const QR_MXFP4 = 2;
    pub const QI_MXFP4 = c.QK_MXFP4 / (4 * QR_MXFP4);
    pub const QR_NVFP4 = 2;
    pub const QI_NVFP4 = c.QK_NVFP4 / (4 * QR_NVFP4);
    pub const QR5_0 = 2;
    pub const QI5_0 = c.QK5_0 / (4 * QR5_0);
    pub const QR5_1 = 2;
    pub const QI5_1 = c.QK5_1 / (4 * QR5_1);
    pub const QR8_0 = 1;
    pub const QI8_0 = c.QK8_0 / (4 * QR8_0);
    pub const QR8_1 = 1;
    pub const QI8_1 = c.QK8_1 / (4 * QR8_1);
    pub const QR2_K = 4;
    pub const QI2_K = c.QK_K / (4 * QR2_K);
    pub const QR3_K = 4;
    pub const QI3_K = c.QK_K / (4 * QR3_K);
    pub const QR4_K = 2;
    pub const QI4_K = c.QK_K / (4 * QR4_K);
    pub const QR5_K = 2;
    pub const QI5_K = c.QK_K / (4 * QR5_K);
    pub const QR6_K = 2;
    pub const QI6_K = c.QK_K / (4 * QR6_K);
    pub const QR2_XXS = 4;
    pub const QI2_XXS = c.QK_K / (4 * QR2_XXS);
    pub const QR2_XS = 4;
    pub const QI2_XS = c.QK_K / (4 * QR2_XS);
    pub const QR2_S = 4;
    pub const QI2_S = c.QK_K / (4 * QR2_S);
    pub const QR3_XXS = 4;
    pub const QI3_XXS = c.QK_K / (4 * QR3_XXS);
    pub const QR3_XS = 4;
    pub const QI3_XS = c.QK_K / (4 * QR3_XS);
    pub const QR1_S = 8;
    pub const QI1_S = c.QK_K / (4 * QR1_S);
    pub const QR1_M = 8;
    pub const QI1_M = c.QK_K / (4 * QR1_M);
    pub const QR4_NL = 2;
    pub const QI4_NL = c.QK4_NL / (4 * QR4_NL);
    pub const QR4_XS = 2;
    pub const QI4_XS = c.QK_K / (4 * QR4_XS);
    pub const QR3_S = 4;
    pub const QI3_S = c.QK_K / (4 * QR3_S);
};

/// Ports `ggml_cuda_type_traits` (common.cuh:982 @b29c606e2): `qk`, `qr`
/// and `qi` for each type that has a specialisation. `F16` has no `qi`.
/// The specialisations' `bs`, the block size in bytes, is left out: only
/// device code reads it (`mul_mat_vec_q`'s DGX Spark prefetch).
pub const TypeTraits = struct { qk: comptime_int, qr: comptime_int, qi: comptime_int };

pub fn typeTraits(comptime t: c.enum_ggml_type) TypeTraits {
    return switch (t) {
        c.GGML_TYPE_F16 => .{ .qk = 1, .qr = 1, .qi = 0 },
        c.GGML_TYPE_Q1_0 => .{ .qk = c.QK1_0, .qr = q.QR1_0, .qi = q.QI1_0 },
        c.GGML_TYPE_Q2_0 => .{ .qk = c.QK2_0, .qr = q.QR2_0, .qi = q.QI2_0 },
        c.GGML_TYPE_Q4_0 => .{ .qk = c.QK4_0, .qr = q.QR4_0, .qi = q.QI4_0 },
        c.GGML_TYPE_Q4_1 => .{ .qk = c.QK4_1, .qr = q.QR4_1, .qi = q.QI4_1 },
        c.GGML_TYPE_Q5_0 => .{ .qk = c.QK5_0, .qr = q.QR5_0, .qi = q.QI5_0 },
        c.GGML_TYPE_Q5_1 => .{ .qk = c.QK5_1, .qr = q.QR5_1, .qi = q.QI5_1 },
        c.GGML_TYPE_Q8_0 => .{ .qk = c.QK8_0, .qr = q.QR8_0, .qi = q.QI8_0 },
        c.GGML_TYPE_MXFP4 => .{ .qk = c.QK_MXFP4, .qr = q.QR_MXFP4, .qi = q.QI_MXFP4 },
        c.GGML_TYPE_NVFP4 => .{ .qk = c.QK_NVFP4, .qr = q.QR_NVFP4, .qi = q.QI_NVFP4 },
        c.GGML_TYPE_Q2_K => .{ .qk = c.QK_K, .qr = q.QR2_K, .qi = q.QI2_K },
        c.GGML_TYPE_Q3_K => .{ .qk = c.QK_K, .qr = q.QR3_K, .qi = q.QI3_K },
        c.GGML_TYPE_Q4_K => .{ .qk = c.QK_K, .qr = q.QR4_K, .qi = q.QI4_K },
        c.GGML_TYPE_Q5_K => .{ .qk = c.QK_K, .qr = q.QR5_K, .qi = q.QI5_K },
        c.GGML_TYPE_Q6_K => .{ .qk = c.QK_K, .qr = q.QR6_K, .qi = q.QI6_K },
        c.GGML_TYPE_IQ2_XXS => .{ .qk = c.QK_K, .qr = q.QR2_XXS, .qi = q.QI2_XXS },
        c.GGML_TYPE_IQ2_XS => .{ .qk = c.QK_K, .qr = q.QR2_XS, .qi = q.QI2_XS },
        c.GGML_TYPE_IQ2_S => .{ .qk = c.QK_K, .qr = q.QR2_S, .qi = q.QI2_S },
        c.GGML_TYPE_IQ3_XXS => .{ .qk = c.QK_K, .qr = q.QR3_XXS, .qi = q.QI3_XXS },
        c.GGML_TYPE_IQ1_S => .{ .qk = c.QK_K, .qr = q.QR1_S, .qi = q.QI1_S },
        c.GGML_TYPE_IQ1_M => .{ .qk = c.QK_K, .qr = q.QR1_M, .qi = q.QI1_M },
        c.GGML_TYPE_IQ4_NL => .{ .qk = c.QK4_NL, .qr = q.QR4_NL, .qi = q.QI4_NL },
        c.GGML_TYPE_IQ4_XS => .{ .qk = c.QK_K, .qr = q.QR4_XS, .qi = q.QI4_XS },
        c.GGML_TYPE_IQ3_S => .{ .qk = c.QK_K, .qr = q.QR3_S, .qi = q.QI3_S },
        else => @compileError("no ggml_cuda_type_traits specialisation"),
    };
}
