//! Finding and launching the CUDA kernels.
//!
//! # Provenance
//!
//! Ports the launch helpers of `llama.cpp/ggml/src/ggml-cuda/common.cuh` at
//! v0.3.0 (`c1d0e7a00`): `ggml_cuda_kernel_launch`, `ggml_cuda_pdl_config`,
//! `ggml_cuda_kernel_can_use_pdl` and `CUDA_SET_SHARED_MEMORY_LIMIT`. The
//! lookup and the loader are not ports; they replace what `nvcc`'s host
//! stubs and cudart's fatbin registration do for the C++ (PLAN.md,
//! Decisions 44–46).
//!
//! # How a kernel is named
//!
//! `get("binbcast", "k_bin_bcast<&op_add, float, float, float>")`: the
//! translation unit whose host code launches it, and the key
//! `build/cuda_kernels.zig` derives from the demangled symbol. Both are
//! comptime, and a key the table lacks is a compile error naming the
//! nearest keys, because upstream's host code only instantiates the
//! kernels it launches and a miss means our dispatch asked for one it
//! does not. The translation unit is part of the name, not a hint: two
//! `.cu` files may each define a `static` kernel of the same name.
//!
//! # Loading
//!
//! `get` is comptime and everything after it is not: a launch is generic
//! only over its argument tuple's type, so the hundreds of kernel
//! instantiations MMQ and MMVQ dispatch to share one copy of the launch
//! code per signature instead of one per kernel.
//!
//! One `CUlibrary` per translation unit, loaded on first use, and one
//! `CUkernel` per kernel, also on first use. Both are context-independent,
//! so neither is per device. `cuLibraryLoadData` follows
//! `CUDA_MODULE_LOADING` as cudart's registration does, so the reference and
//! the port load lazily alike.
//!
//! # The argument check
//!
//! A kernel's parameters are a byte buffer the driver copies; a Zig
//! argument of the wrong width is not a type error, it is a different
//! number. The first launch of every kernel compares each argument's size
//! and the argument count against `cuKernelGetParamInfo` (Decision 46).
//! **In every build, not only safe ones**: the gates build ReleaseFast,
//! and a check they cannot see is a check nothing runs. After the first
//! launch it costs one atomic load. It cannot see field order inside a
//! struct; that is `make cuda-struct-layout`'s job.

const std = @import("std");
const builtin = @import("builtin");
const common = @import("common.zig");
const gen = @import("cuda_kernels");

const cu = common.cu;

/// A kernel, resolved at comptime.
pub const Kernel = struct {
    /// Index into the process-wide kernel cache.
    index: u32,
    /// Index into `gen.libraries`.
    library: u16,
    mangled: [:0]const u8,
    /// `library`'s name and the key, for messages.
    name: []const u8,
    /// The library's fatbin. Carried here, not looked up by `library` at
    /// run time, so that only the libraries some kernel names are embedded.
    fatbin: []const u8,
};

/// Resolves a kernel at comptime.
///
/// Parameters:
/// - `library`: the translation unit's stem, e.g. `"binbcast"` or
///   `"mmq-instance-q4_k"`.
/// - `key`: the demangled template spelling without return type or
///   parameters.
///
/// Return: the kernel; a compile error when either is unknown.
pub fn get(comptime library: []const u8, comptime key: []const u8) Kernel {
    return comptime lookup(library, key);
}

fn lookup(comptime library: []const u8, comptime key: []const u8) Kernel {
    @setEvalBranchQuota(10_000_000);
    const li = for (gen.libraries, 0..) |lib, i| {
        if (std.mem.eql(u8, lib.name, library)) break i;
    } else @compileError("no CUDA translation unit named '" ++ library ++ "'");
    const lib = gen.libraries[li];
    var lo: usize = 0;
    var hi: usize = lib.entries.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        switch (std.mem.order(u8, lib.entries[mid].key, key)) {
            .eq => return .{
                .index = lib.first + mid,
                .library = li,
                .mangled = lib.entries[mid].mangled,
                .name = library ++ ": " ++ key,
                .fatbin = lib.fatbin,
            },
            .lt => lo = mid + 1,
            .gt => hi = mid,
        }
    }
    const near = if (lo < lib.entries.len) lib.entries[lo].key else "(end)";
    const before = if (lo > 0) lib.entries[lo - 1].key else "(start)";
    @compileError("no CUDA kernel '" ++ key ++ "' in '" ++ library ++ "'; nearest: '" ++ before ++ "', '" ++ near ++ "'");
}

// -----------------------------------------------------------------------------
// Loading

var load_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var libraries: [gen.libraries.len]cu.CUlibrary = @splat(null);
var handles: [gen.kernel_count]std.atomic.Value(?*anyopaque) = @splat(.init(null));
var checked: [gen.kernel_count]std.atomic.Value(bool) = @splat(.init(false));

/// The kernel's handle, loading its library on first use.
///
/// Parameters:
/// - `k`: from `get`.
///
/// Return: the context-independent handle; aborts if the driver refuses.
pub fn handle(k: Kernel) cu.CUkernel {
    if (handles[k.index].load(.acquire)) |h| return @ptrCast(h);
    _ = std.c.pthread_mutex_lock(&load_mutex);
    defer _ = std.c.pthread_mutex_unlock(&load_mutex);
    if (handles[k.index].load(.acquire)) |h| return @ptrCast(h);
    if (libraries[k.library] == null) {
        const image = k.fatbin;
        common.cuCheck(cu.cuLibraryLoadData(&libraries[k.library], image.ptr, null, null, 0, null, null, 0), "cuLibraryLoadData", @src());
    }
    var h: cu.CUkernel = null;
    common.cuCheck(cu.cuLibraryGetKernel(&h, libraries[k.library], k.mangled.ptr), "cuLibraryGetKernel", @src());
    handles[k.index].store(@ptrCast(h), .release);
    return h;
}

// -----------------------------------------------------------------------------
// Launching

/// `dim3`.
pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,

    pub fn of(x: anytype, y: anytype, z: anytype) Dim3 {
        return .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) };
    }
};

/// Ports `ggml_cuda_kernel_launch_params` (common.cuh:1552 @c1d0e7a00).
pub const LaunchParams = struct {
    grid: Dim3,
    block: Dim3,
    shmem: usize = 0,
    stream: cu.cudaStream_t,
};

/// The pointer array the driver reads, one per argument.
fn ArgPointers(comptime Args: type) type {
    return [@typeInfo(Args).@"struct".field_types.len]?*anyopaque;
}

/// `Args` with every field a runtime field. A tuple literal's
/// comptime-known elements are comptime fields, which have no address.
fn RuntimeArgs(comptime Args: type) type {
    return @Tuple(@typeInfo(Args).@"struct".field_types);
}

fn toRuntime(args: anytype) RuntimeArgs(@TypeOf(args)) {
    var out: RuntimeArgs(@TypeOf(args)) = undefined;
    inline for (0..@typeInfo(@TypeOf(args)).@"struct".field_types.len) |i| out[i] = args[i];
    return out;
}

/// Packs `args` into the pointer array the driver reads, after the
/// check against the kernel's declared parameters on its first launch.
fn pack(k: Kernel, h: cu.CUkernel, args: anytype) ArgPointers(@TypeOf(args.*)) {
    const types = @typeInfo(@TypeOf(args.*)).@"struct".field_types;
    if (!checked[k.index].load(.acquire)) {
        checkArgs(k, h, types);
        checked[k.index].store(true, .release);
    }
    var ptrs: ArgPointers(@TypeOf(args.*)) = undefined;
    inline for (0..types.len) |i| ptrs[i] = @ptrCast(@constCast(&args[i]));
    return ptrs;
}

/// Decision 46: every argument's size, and the count, against the kernel.
fn checkArgs(k: Kernel, h: cu.CUkernel, comptime types: []const type) void {
    inline for (types, 0..) |T, i| {
        var offset: usize = 0;
        var size: usize = 0;
        const r = cu.cuKernelGetParamInfo(h, i, &offset, &size);
        if (r != cu.CUDA_SUCCESS) {
            common.fatal("CUDA kernel {s}: argument {d} of {d} has no parameter", .{ k.name, i, types.len });
        }
        if (size != @sizeOf(T)) {
            common.fatal("CUDA kernel {s}: argument {d} is {d} bytes ({s}), the parameter {d}", .{ k.name, i, @sizeOf(T), @typeName(T), size });
        }
    }
    var offset: usize = 0;
    var size: usize = 0;
    if (cu.cuKernelGetParamInfo(h, types.len, &offset, &size) == cu.CUDA_SUCCESS) {
        common.fatal("CUDA kernel {s}: {d} arguments given, it takes more", .{ k.name, types.len });
    }
}

/// The current context's function for `h`, for the per-device queries.
fn function(h: cu.CUkernel) cu.CUfunction {
    var f: cu.CUfunction = null;
    common.cuCheck(cu.cuKernelGetFunction(&f, h), "cuKernelGetFunction", @src());
    return f;
}

/// `kernel<<<grid, block, shmem, stream>>>(args...)`, then
/// `CUDA_CHECK(cudaGetLastError())`.
///
/// Parameters:
/// - `k`: from `get`.
/// - `p`: grid, block, dynamic shared memory and stream.
/// - `args`: a tuple of the kernel's arguments, in order, each of exactly
///   the parameter's type.
pub fn launch(k: Kernel, p: LaunchParams, args: anytype) void {
    const h = handle(k);
    var a = toRuntime(args);
    var ptrs = pack(k, h, &a);
    common.cuCheck(cu.cuLaunchKernel(@ptrCast(h), p.grid.x, p.grid.y, p.grid.z, p.block.x, p.block.y, p.block.z, @intCast(p.shmem), @ptrCast(p.stream), &ptrs, null), "kernel launch", @src());
}

/// Ports `ggml_cuda_kernel_launch` (common.cuh:1657 @c1d0e7a00): `launch`
/// with programmatic dependent launch where the kernel's PTX allows it and
/// `GGML_CUDA_PDL` does not forbid it.
pub fn launchPdl(k: Kernel, p: LaunchParams, args: anytype) void {
    const h = handle(k);
    if (pdlEnabled() and canUsePdl(k, h)) {
        var a = toRuntime(args);
        var ptrs = pack(k, h, &a);
        // Ports `ggml_cuda_pdl_config` (common.cuh:1568 @c1d0e7a00).
        var attr: cu.CUlaunchAttribute = std.mem.zeroes(cu.CUlaunchAttribute);
        attr.id = cu.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
        var cfg: cu.CUlaunchConfig = std.mem.zeroes(cu.CUlaunchConfig);
        cfg.gridDimX = p.grid.x;
        cfg.gridDimY = p.grid.y;
        cfg.gridDimZ = p.grid.z;
        cfg.blockDimX = p.block.x;
        cfg.blockDimY = p.block.y;
        cfg.blockDimZ = p.block.z;
        cfg.sharedMemBytes = @intCast(p.shmem);
        cfg.hStream = @ptrCast(p.stream);
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        common.cuCheck(cu.cuLaunchKernelEx(&cfg, @ptrCast(h), &ptrs, null), "cudaLaunchKernelEx", @src());
        return;
    }
    launch(k, p, args);
}

/// `CUDA_CHECK(cudaLaunchCooperativeKernel((void *) kernel, grid, block,
/// args, shmem, stream))`, as `softmax.cu` launches
/// `soft_max_f32_parallelize_cols`.
///
/// The driver's `cuLaunchCooperativeKernel` takes the same grid, block,
/// dynamic shared memory, stream and argument array, and accepts a
/// `CUkernel` as `launch`'s `cuLaunchKernel` does; it is what the runtime
/// call forwards to. `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`
/// would launch the same grid too, but it is the driver's counterpart of
/// `cudaLaunchKernelEx`, which the C does not call here. No PDL attribute:
/// the C passes none.
///
/// Parameters:
/// - `k`: from `get`.
/// - `p`: grid, block, dynamic shared memory and stream.
/// - `args`: a tuple of the kernel's arguments, in order, each of exactly
///   the parameter's type.
pub fn launchCooperative(k: Kernel, p: LaunchParams, args: anytype) void {
    const h = handle(k);
    var a = toRuntime(args);
    var ptrs = pack(k, h, &a);
    common.cuCheck(cu.cuLaunchCooperativeKernel(@ptrCast(h), p.grid.x, p.grid.y, p.grid.z, p.block.x, p.block.y, p.block.z, @intCast(p.shmem), @ptrCast(p.stream), &ptrs), "cudaLaunchCooperativeKernel", @src());
}

/// The C's function-local `env_pdl_enabled` static.
fn pdlEnabled() bool {
    const S = struct {
        var value: bool = true;
        fn init() void {
            const env = std.c.getenv("GGML_CUDA_PDL");
            value = env == null or common.atoi(env.?) != 0;
        }
        const once = common.Once(init);
    };
    S.once.call();
    return S.value;
}

/// Ports `ggml_cuda_kernel_can_use_pdl` (common.cuh:1592 @c1d0e7a00).
///
/// The C caches the answer per (device, kernel) behind a mutex; the PTX
/// version a loaded function reports is a property of the kernel image and
/// the device, so the cache here is per (device, kernel) too, as a
/// tri-state byte.
fn canUsePdl(k: Kernel, h: cu.CUkernel) bool {
    const device: usize = @intCast(common.getDevice());
    const slot = &pdl_cache[device][k.index];
    switch (slot.load(.acquire)) {
        1 => return false,
        2 => return true,
        else => {},
    }
    var ptx: c_int = 0;
    common.cuCheck(cu.cuFuncGetAttribute(&ptx, cu.CU_FUNC_ATTRIBUTE_PTX_VERSION, function(h)), "cudaFuncGetAttributes", @src());
    // PDL device-side primitives are emitted only for PTX versions >= 90.
    const can = ptx >= 90;
    slot.store(if (can) 2 else 1, .release);
    return can;
}

var pdl_cache: [common.max_devices][gen.kernel_count]std.atomic.Value(u8) = @splat(@splat(.init(0)));

/// Ports `CUDA_SET_SHARED_MEMORY_LIMIT` (common.cuh:231 @c1d0e7a00): raise
/// the kernel's dynamic shared memory limit on the current device, once.
pub fn setSharedMemoryLimit(k: Kernel, nbytes: usize) void {
    const device = common.getDevice();
    const slot = &smem_raised[@intCast(device)][k.index];
    if (slot.load(.acquire)) return;
    var dev: cu.CUdevice = 0;
    common.cuCheck(cu.cuDeviceGet(&dev, device), "cuDeviceGet", @src());
    common.cuCheck(cu.cuKernelSetAttribute(cu.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, @intCast(nbytes), handle(k), dev), "cudaFuncSetAttribute", @src());
    slot.store(true, .release);
}

var smem_raised: [common.max_devices][gen.kernel_count]std.atomic.Value(bool) = @splat(@splat(.init(false)));

// -----------------------------------------------------------------------------
// Spelling a key
//
// `cu++filt` writes a template argument the way these produce it: a type by
// its C++ name, a non-type argument as `(int)64`, `(unsigned long)128`,
// `(bool)0` or `(ggml_type)12`, a function as `&op_add`, and arguments
// joined by ", ". Measured from the generated table, not assumed.

/// `half` in a kernel's template arguments. A distinct type, so a host
/// function generic over the element type can name it.
pub const half = enum(u16) { _ };

/// `nv_bfloat16` in a kernel's template arguments.
pub const bf16 = enum(u16) { _ };

/// `half2`.
pub const half2 = enum(u32) { _ };

/// `nv_bfloat162`.
pub const bf162 = enum(u32) { _ };

/// The demangled C++ name of a host type that appears as a template
/// argument.
pub fn typeName(comptime T: type) []const u8 {
    return switch (T) {
        f32 => "float",
        f64 => "double",
        half => "__half",
        bf16 => "__nv_bfloat16",
        half2 => "__half2",
        bf162 => "__nv_bfloat162",
        i8 => "signed char",
        u8 => "unsigned char",
        i16 => "short",
        u16 => "unsigned short",
        i32 => "int",
        u32 => "unsigned int",
        i64 => "long",
        u64 => "unsigned long",
        else => @compileError("no C++ spelling for " ++ @typeName(T)),
    };
}

/// `const T *`.
pub fn constPtr(comptime T: type) []const u8 {
    return "const " ++ typeName(T) ++ " *";
}

/// `(int)v`.
pub fn int(comptime v: comptime_int) []const u8 {
    return std.fmt.comptimePrint("(int){d}", .{v});
}

/// `(long)v`.
pub fn long(comptime v: comptime_int) []const u8 {
    return std.fmt.comptimePrint("(long){d}", .{v});
}

/// `(unsigned long)v`, a `size_t` argument.
pub fn ulong(comptime v: comptime_int) []const u8 {
    return std.fmt.comptimePrint("(unsigned long){d}", .{v});
}

/// `(bool)0` or `(bool)1`.
pub fn boolean(comptime v: bool) []const u8 {
    return if (v) "(bool)1" else "(bool)0";
}

/// `(ggml_type)t`.
pub fn ggmlType(comptime t: comptime_int) []const u8 {
    return std.fmt.comptimePrint("(ggml_type){d}", .{t});
}

/// `name<a, b, c>`.
pub fn tmpl(comptime name: []const u8, comptime args: []const []const u8) []const u8 {
    comptime var s: []const u8 = name ++ "<";
    inline for (args, 0..) |a, i| s = s ++ (if (i == 0) "" else ", ") ++ a;
    return s ++ ">";
}

/// The tuple type `Base` with `n` more elements of type `P`.
pub fn Append(comptime Base: type, comptime n: usize, comptime P: type) type {
    const bt = @typeInfo(Base).@"struct".field_types;
    var types: [bt.len + n]type = undefined;
    for (bt, 0..) |t, i| types[i] = t;
    for (0..n) |i| types[bt.len + i] = P;
    return @Tuple(&types);
}

/// `base` with `extra` appended, for a kernel whose trailing parameters are
/// a C++ parameter pack -- `k_bin_bcast`'s `src1_ptrs...`.
pub fn append(base: anytype, comptime n: usize, comptime P: type, extra: [n]P) Append(@TypeOf(base), n, P) {
    const len = @typeInfo(@TypeOf(base)).@"struct".field_types.len;
    var out: Append(@TypeOf(base), n, P) = undefined;
    inline for (0..len) |i| out[i] = base[i];
    inline for (0..n) |i| out[len + i] = extra[i];
    return out;
}
