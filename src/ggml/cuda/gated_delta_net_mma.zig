//! The chunked gated-delta-net kernel, new at v0.7.0 (#29353): for a
//! long non-KDA prefill with 128-wide heads, `gdn_single` advances the
//! recurrence 16 tokens at a time through BF16 tensor-core products (a
//! high and a residual half per operand), where `gated_delta_net_cuda`
//! steps one token at a time. `gated_delta_net.zig` asks `launch` first and
//! falls back to the recurrent kernel when it declines.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/gated-delta-net-mma.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/gated-delta-net-mma.cuh` at v0.8.0
//! (`8642541b2`): the host side -- the argument struct, the per-device
//! shared-memory opt-in, the path choice and the launch -- on the CUDA arms
//! (`GGML_USE_HIP` and `GGML_USE_MUSA` compile out). The kernel stays CUDA.
//!
//! # The kernel's name
//!
//! `gdn_single` is a `static __global__` template inside a namespace, so
//! its demangled key carries the namespace. Through v0.7.0 that was an
//! anonymous one, which `cu++filt` spells `<unnamed>`; since v0.8.0 it is
//! named: `ggml_cuda_gated_delta_net_mma::gdn_single<(int)16, (int)128,
//! (int)1>`. The first such key in this layer.
//!
//! # The shared-memory size
//!
//! The launch passes `sizeof(shared<C, V>)` as dynamic shared memory and
//! the kernel reinterprets it as that struct; nothing checks the two
//! agree. `Shared` mirrors the struct field for field (BF16 is two bytes,
//! the struct is `alignas(128)`), and its three sizes are pinned below
//! against the arithmetic; the GB10 pickup compares them with `nvcc`'s.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const cu = common.cu;

/// Ports `ggml_cuda_gdn_mma_args` (gated-delta-net-mma.cuh:7 @8642541b2).
/// Passed to the kernel by value, so its size is checked against the
/// kernel's parameter on the first launch (Decision 46).
pub const Args = extern struct {
    q: ?*const f32,
    k: ?*const f32,
    v: ?*const f32,
    g: ?*const f32,
    beta: ?*const f32,
    state: ?*const f32,
    dst: ?*f32,
    state_out: ?*f32,

    H: i64,
    H_k: i64,
    n_tokens: i64,
    n_seqs: i64,
    rq3: i64,
    sq1: i64,
    sq2: i64,
    sq3: i64,
    sv1: i64,
    sv2: i64,
    sv3: i64,
    sb1: i64,
    sb2: i64,
    sb3: i64,
    scale: f32,
    eligible: bool,
};

comptime {
    // 8 pointers, 14 int64_t, a float and a bool, padded to 8. The size is
    // all the first launch's check sees, and two fields swapped can keep it,
    // so the kernel-read offsets are pinned too.
    if (@sizeOf(Args) != 184) @compileError("ggml_cuda_gdn_mma_args is 184 bytes on a 64-bit target");
    if (@offsetOf(Args, "state_out") != 56 or @offsetOf(Args, "H") != 64 or @offsetOf(Args, "sb3") != 168 or
        @offsetOf(Args, "scale") != 176 or @offsetOf(Args, "eligible") != 180)
    {
        @compileError("ggml_cuda_gdn_mma_args: a field is not where the kernel reads it");
    }
}

/// Ports `D` (gated-delta-net-mma.cu:8 @8642541b2): the head size.
const D = 128;

/// `nv_bfloat16`, for sizes only.
const Bf16 = u16;

/// Ports `matrix` (gated-delta-net-mma.cu:11 @8642541b2), `matrix<R, C>`, for
/// its size: a high and a residual BF16 plane.
fn Matrix(comptime R: comptime_int, comptime C: comptime_int) type {
    return extern struct {
        hi: [R * C]Bf16,
        lo: [R * C]Bf16,
    };
}

/// Ports `shared` (gated-delta-net-mma.cu:44 @8642541b2), `shared<C, V>`, for
/// its size, which is the launch's dynamic shared memory.
fn Shared(comptime C: comptime_int, comptime V: comptime_int) type {
    return extern struct {
        q: Matrix(C, D) align(128),
        k: Matrix(C, D),
        p: Matrix(C, C),
        inverse: Matrix(C, C),
        delta: Matrix(C, V),
        scratch: extern union {
            state: Matrix(D, V),
            solved: Matrix(C, V),
        },
        lower: [C * C]f32,
        prefix: [C]f32,
        beta: [C]f32,
    };
}

comptime {
    // 4*C*(2*D + 2*C + V) + 4*D*V + 4*C*C + 8*C, rounded up to 128.
    if (@sizeOf(Shared(16, 32)) != 38016) @compileError("shared<16, 32> is 38016 bytes");
    if (@sizeOf(Shared(16, 64)) != 56448) @compileError("shared<16, 64> is 56448 bytes");
    if (@sizeOf(Shared(16, 128)) != 93312) @compileError("shared<16, 128> is 93312 bytes");
}

fn kernel(comptime C: comptime_int, comptime V: comptime_int, comptime BLOCKS: comptime_int) kernels.Kernel {
    return kernels.get("gated-delta-net-mma", kernels.tmpl("ggml_cuda_gated_delta_net_mma::gdn_single", &.{ kernels.int(C), kernels.int(V), kernels.int(BLOCKS) }));
}

/// Ports `init` (gated-delta-net-mma.cu:366 @8642541b2), `init<C, V, BLOCKS>`:
/// whether the device can give the kernel `sizeof(shared<C, V>)` bytes of
/// shared memory, asked once per device. The C's `std::call_once` per
/// device; a refusal is remembered as well as an acceptance.
fn Init(comptime C: comptime_int, comptime V: comptime_int, comptime BLOCKS: comptime_int) type {
    return struct {
        var mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
        var done: [common.max_devices]bool = @splat(false);
        var available: [common.max_devices]bool = @splat(false);

        fn get(device: c_int) bool {
            const d: usize = @intCast(device);
            _ = std.c.pthread_mutex_lock(&mutex);
            defer _ = std.c.pthread_mutex_unlock(&mutex);
            if (!done[d]) {
                done[d] = true;
                run(device);
            }
            return available[d];
        }

        fn run(device: c_int) void {
            common.setDevice(device);
            const size = @sizeOf(Shared(C, V));
            if (size > common.info().devices[@intCast(device)].smpbo) return;
            // `cudaFuncSetAttribute((const void *) gdn_single<C, V, BLOCKS>,
            // cudaFuncAttributeMaxDynamicSharedMemorySize, size)` on the
            // device just made current.
            var dev: cu.CUdevice = 0;
            common.cuCheck(cu.cuDeviceGet(&dev, common.getDevice()), "cuDeviceGet", @src());
            const status = cu.cuKernelSetAttribute(cu.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size, kernels.handle(kernel(C, V, BLOCKS)), dev);
            available[@intCast(device)] = status == cu.CUDA_SUCCESS;
            if (status != cu.CUDA_SUCCESS) {
                _ = cu.cudaGetLastError();
            }
        }
    };
}

/// Ports `gdn_path` (gated-delta-net-mma.cu:384 @8642541b2).
const Path = enum { ar, slice32_one_block, slice32_two_blocks, slice64, whole_head };

/// Ports `select_gdn_mma_path` (gated-delta-net-mma.cu:386 @8642541b2), the
/// CUDA arm: only the architectures upstream benchmarked, a value-split
/// schedule when whole heads would leave half the SMs idle.
fn selectPath(device: c_int, a: *const Args) Path {
    impl.assert(device >= 0 and device < common.max_devices, "device >= 0 && device < GGML_CUDA_MAX_DEVICES");
    if (!a.eligible or a.H_k != 16 or (a.H != 16 and a.H != 32 and a.H != 48 and a.H != 64) or a.n_tokens < 64 or
        a.n_tokens > std.math.maxInt(i64) - 32 or a.rq3 <= 0 or a.n_seqs <= 0 or a.n_seqs > @divTrunc(std.math.maxInt(c_int), a.H * 4))
    {
        return .ar;
    }
    const inf = &common.info().devices[@intCast(device)];
    // Enable only architectures benchmarked for this schedule. MMA support alone does not imply a speedup.
    if (!common.ampereMmaAvailable(inf.cc) or
        (inf.cc != common.cc.ampere_ga10x and inf.cc != common.cc.ada_lovelace and
            inf.cc != common.cc.blackwell and inf.cc != common.cc.dgx_spark))
    {
        return .ar;
    }
    // Value splitting pays off when whole heads leave at least half the SMs idle.
    if (a.H * a.n_seqs * 2 <= inf.nsm) {
        // A grid that already fits in one wave benefits from a larger register budget.
        if (a.H * a.n_seqs * 4 <= inf.nsm) {
            return if (Init(16, 32, 1).get(device)) .slice32_one_block else .ar;
        }
        return if (Init(16, 32, 2).get(device)) .slice32_two_blocks else .ar;
    }
    return if (Init(16, 128, 1).get(device)) .whole_head else .ar;
}

/// `dim3(n)` from an `int64_t`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// Ports `ggml_cuda_gdn_mma_launch` (gated-delta-net-mma.cu:422
/// @8642541b2), which launches with `<<<>>>`.
///
/// Return: whether the chunked kernel ran; false leaves the op to the
/// recurrent kernel, with nothing launched.
pub fn launch(device: c_int, a: *const Args, stream: cu.cudaStream_t) bool {
    if (a.state_out == null) return false;
    const path = selectPath(device, a);
    if (path == .ar) return false;
    const block: kernels.Dim3 = .{ .x = 32, .y = 8 };
    if (path == .slice32_two_blocks) {
        kernels.launch(kernel(16, 32, 2), .{ .grid = .{ .x = dim(a.H * a.n_seqs * 4) }, .block = block, .shmem = @sizeOf(Shared(16, 32)), .stream = stream }, .{a.*});
    } else if (path == .slice32_one_block) {
        kernels.launch(kernel(16, 32, 1), .{ .grid = .{ .x = dim(a.H * a.n_seqs * 4) }, .block = block, .shmem = @sizeOf(Shared(16, 32)), .stream = stream }, .{a.*});
    } else {
        kernels.launch(kernel(16, 128, 1), .{ .grid = .{ .x = dim(a.H * a.n_seqs) }, .block = block, .shmem = @sizeOf(Shared(16, 128)), .stream = stream }, .{a.*});
    }
    common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
    return true;
}
