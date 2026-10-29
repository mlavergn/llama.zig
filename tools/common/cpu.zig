//! How many threads upstream gives the CPU by default.
//!
//! # Provenance
//!
//! Ports `common_cpu_get_num_physical_cores`, `common_cpu_get_num_math` and
//! the x86 Linux hybrid-core probe from `llama.cpp/common/common.cpp`
//! (v0.4.1, `b29c606e2`). Each declaration names what it ports and the line
//! it began at. The AIX arm is not ported, and Windows takes the MinGW
//! path upstream's own `#if` takes: `hardware_concurrency` halved.

const std = @import("std");
const builtin = @import("builtin");

extern fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*anyopaque, newlen: usize) c_int;
// `std.c` declares no `fgets`.
extern fn fgets(s: [*]u8, n: c_int, stream: *std.c.FILE) ?[*]u8;

/// `std::thread::hardware_concurrency()`.
fn hardwareConcurrency() u32 {
    return @intCast(std.Thread.getCpuCount() catch 0);
}

/// Ports `common_cpu_get_num_physical_cores` (common/common.cpp:78 @b29c606e2).
pub fn numPhysicalCores() i32 {
    if (comptime builtin.os.tag == .linux) {
        // enumerate the set of thread siblings, num entries is num cores
        var seen: [1024][]u8 = undefined;
        var n_seen: usize = 0;
        var store: [1024 * 64]u8 = undefined;
        var used: usize = 0;
        var cpu: u32 = 0;
        while (cpu < std.math.maxInt(u32)) : (cpu += 1) {
            var path_buf: [96]u8 = undefined;
            const path = std.fmt.bufPrintSentinel(&path_buf, "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings", .{cpu}, 0) catch break;
            const f = std.c.fopen(path.ptr, "r") orelse break; // no more cpus
            defer _ = std.c.fclose(f);
            var line: [256]u8 = undefined;
            const got = fgets(&line, line.len, f) orelse continue;
            var s: []u8 = std.mem.span(@as([*:0]u8, @ptrCast(got)));
            if (s.len > 0 and s[s.len - 1] == '\n') s = s[0 .. s.len - 1];
            var dup = false;
            for (seen[0..n_seen]) |x| {
                if (std.mem.eql(u8, x, s)) dup = true;
            }
            if (!dup and n_seen < seen.len and used + s.len <= store.len) {
                @memcpy(store[used..][0..s.len], s);
                seen[n_seen] = store[used..][0..s.len];
                used += s.len;
                n_seen += 1;
            }
        }
        if (n_seen != 0) return @intCast(n_seen);
    } else if (comptime builtin.os.tag.isDarwin()) {
        var num_physical_cores: i32 = 0;
        var len: usize = @sizeOf(i32);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, null, 0) == 0) return num_physical_cores;
        if (sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, null, 0) == 0) return num_physical_cores;
    }
    const n_threads = hardwareConcurrency();
    return @intCast(if (n_threads > 0) (if (n_threads <= 4) n_threads else n_threads / 2) else 4);
}

fn cpuid(leaf: u32, subleaf: u32) [4]u32 {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [eax] "={eax}" (eax),
          [ebx] "={ebx}" (ebx),
          [ecx] "={ecx}" (ecx),
          [edx] "={edx}" (edx),
        : [leaf] "{eax}" (leaf),
          [subleaf] "{ecx}" (subleaf),
    );
    return .{ eax, ebx, ecx, edx };
}

/// Ports `is_hybrid_cpu` (common/common.cpp:168 @b29c606e2).
fn isHybridCpu() bool {
    return cpuid(7, 0)[3] & (1 << 15) != 0;
}

/// Ports `is_running_on_efficiency_core` (common/common.cpp:174 @b29c606e2).
fn isRunningOnEfficiencyCore() bool {
    const intel_atom = 0x20;
    const core_type = (cpuid(0x1a, 0)[0] & 0xff000000) >> 24;
    return core_type == intel_atom;
}

const CpuSet = [128]u8;
extern fn pthread_self() usize;
extern fn pthread_setaffinity_np(thread: usize, size: usize, set: *const CpuSet) c_int;
extern fn pthread_getaffinity_np(thread: usize, size: usize, set: *CpuSet) c_int;

/// Ports `pin_cpu` (common/common.cpp:161 @b29c606e2).
fn pinCpu(cpu: usize) c_int {
    var mask: CpuSet = @splat(0);
    mask[cpu / 8] |= @as(u8, 1) << @intCast(cpu % 8);
    return pthread_setaffinity_np(pthread_self(), @sizeOf(CpuSet), &mask);
}

/// Ports `cpu_count_math_cpus` (common/common.cpp:182 @b29c606e2).
fn countMathCpus(n_cpu: usize) i32 {
    var result: i32 = 0;
    var cpu: usize = 0;
    while (cpu < n_cpu) : (cpu += 1) {
        if (pinCpu(cpu) != 0) return -1;
        if (isRunningOnEfficiencyCore()) continue; // efficiency cores harm lockstep threading
        cpu += 1; // hyperthreading isn't useful for linear algebra
        result += 1;
    }
    return result;
}

/// Ports `common_cpu_get_num_math` (common/common.cpp:202 @b29c606e2): the CPUs
/// useful for math.
pub fn numMath() i32 {
    if (comptime builtin.cpu.arch == .x86_64 and builtin.os.tag == .linux and builtin.abi != .android) {
        const n_cpu = hardwareConcurrency();
        if (n_cpu < 1) return numPhysicalCores();
        if (isHybridCpu()) {
            var affinity: CpuSet = undefined;
            if (pthread_getaffinity_np(pthread_self(), @sizeOf(CpuSet), &affinity) == 0) {
                const result = countMathCpus(n_cpu);
                _ = pthread_setaffinity_np(pthread_self(), @sizeOf(CpuSet), &affinity);
                if (result > 0) return result;
            }
        }
    }
    return numPhysicalCores();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
