//! Every C library function the port still calls, declared once: the
//! ledger of what is left of the libc dependency, each group with the
//! reason it is still the C library's.
//!
//! # Provenance
//!
//! **Not a port.** A file that needs one of these calls `libc.<name>`;
//! nothing else declares a C library function. libm is `libm.zig`. The
//! socket API (`socket`, `getaddrinfo`, `poll`...) stays in
//! `tools/server/httplib/socket.zig`, beside its struct layouts, until the
//! socket pass moves it to `std.Io.net`.
//!
//! # Types
//!
//! `FILE *` is `*anyopaque`: the C library's streams are opaque to the
//! port, and a `*c.FILE` or `*std.c.FILE` a caller holds coerces to it.
//! A parameter C accepts as `NULL` is optional.

const std = @import("std");
const builtin = @import("builtin");

// -----------------------------------------------------------------------------
// stdio: C streams and `printf` formats
//
// The libraries write where the host's C library writes (`stderr` through a
// log callback, files the C code opens), and every message is formatted with
// the C's own format string: Zig's formatting differs from `printf` in the
// last digit of a rounded float, until the C-format formatter lands.

pub extern "c" fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;
pub extern "c" fn vsnprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ap: CVaList) c_int;
pub extern "c" fn fprintf(stream: ?*anyopaque, format: [*:0]const u8, ...) c_int;
pub extern "c" fn fputs(s: [*:0]const u8, stream: ?*anyopaque) c_int;
pub extern "c" fn fputc(char: c_int, stream: ?*anyopaque) c_int;
pub extern "c" fn fflush(stream: ?*anyopaque) c_int;
pub extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
/// Windows' wide-character `fopen`, for a path that is not ASCII.
pub extern "c" fn _wfopen(path: [*:0]const u16, mode: [*:0]const u16) ?*anyopaque;
pub extern "c" fn fclose(stream: ?*anyopaque) c_int;
pub extern "c" fn fread(ptr: ?*anyopaque, size: usize, n: usize, stream: ?*anyopaque) usize;
pub extern "c" fn fwrite(ptr: ?*const anyopaque, size: usize, n: usize, stream: ?*anyopaque) usize;
pub extern "c" fn fseek(stream: ?*anyopaque, offset: c_long, whence: c_int) c_int;
pub extern "c" fn fseeko(stream: ?*anyopaque, offset: i64, whence: c_int) c_int;
pub extern "c" fn ftello(stream: ?*anyopaque) i64;
pub extern "c" fn fgets(buf: [*]u8, n: c_int, stream: ?*anyopaque) ?[*]u8;
pub extern "c" fn getline(line: *?[*]u8, n: *usize, stream: ?*anyopaque) isize;
/// Windows' `fgetwc`: `wint_t` is an `unsigned short` there.
pub extern "c" fn fgetwc(stream: ?*anyopaque) u16;

/// What `va_list` is as a parameter: on x86-64 System V a one-element array
/// of `__va_list_tag`, which decays to a pointer in a parameter list, so the
/// callee takes `*VaList`; on AArch64 -- AAPCS64 and Apple alike -- it is not
/// an array and travels by value. Declaring it by value everywhere worked
/// until the first x86 build, where `ggml_format_name` handed `vsnprintf` the
/// struct's bytes as a pointer and it faulted inside `__vsnprintf`. Windows
/// x64's `va_list` is a `char *` by value, and Zig refuses `VaList` there
/// (`zig_varargs`), so nothing on Windows reaches these.
pub const CVaList = if (va_list_is_array) *std.lang.VaList else std.lang.VaList;

const va_list_is_array = builtin.cpu.arch == .x86_64 and builtin.os.tag != .windows;

/// `ap` as `CVaList` expects it: its address where `va_list` is an array, a
/// copy elsewhere. Either way the callee consumes the list, as in C.
pub inline fn cVaList(ap: *std.lang.VaList) CVaList {
    return if (comptime va_list_is_array) ap else ap.*;
}

// -----------------------------------------------------------------------------
// stdlib: memory, the environment, the process
//
// `free` and `malloc` pair with allocations the C library made (`getline`'s
// buffer, `realpath`'s); `posix_memalign` and Windows' `_aligned_malloc` are
// ggml's `ggml_aligned_malloc`; `setenv` is what a gate does between runs.
// `exit` still flushes the C streams the libraries and the console write.

pub extern "c" fn malloc(size: usize) ?*anyopaque;
pub extern "c" fn free(ptr: ?*anyopaque) void;
pub extern "c" fn posix_memalign(memptr: *?*anyopaque, alignment: usize, size: usize) c_int;
pub extern "c" fn _aligned_malloc(size: usize, alignment: usize) ?*anyopaque;
pub extern "c" fn _aligned_free(ptr: ?*anyopaque) void;
pub extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
pub extern "c" fn abort() noreturn;
pub extern "c" fn exit(code: c_int) noreturn;
pub extern "c" fn _exit(code: c_int) noreturn;
pub extern "c" fn rand() c_int;
pub extern "c" fn arc4random() u32;
pub extern "c" fn rand_s(r: *c_uint) c_int;
pub extern "c" fn strlen(s: [*c]const u8) usize;
pub extern "c" fn strncmp(a: [*]const u8, b: [*:0]const u8, n: usize) c_int;
pub extern "c" fn strerror(errnum: c_int) [*:0]const u8;
pub extern "c" fn setlocale(category: c_int, locale: ?[*:0]const u8) ?[*:0]const u8;
pub extern "c" fn clock() c_long;
/// `signal`, the handler as an address: `SIG_IGN` is 1, `SIG_DFL` 0, a
/// function `@intFromPtr`.
pub extern "c" fn signal(sig: c_int, handler: usize) usize;

// -----------------------------------------------------------------------------
// POSIX: descriptors, processes, memory maps
//
// `fork`, `pipe`, `dup2` and `waitpid` run a gate's reference side in a
// child; `mmap` and the advice calls are `llama-mmap.cpp`'s; `statx` stands
// in for `stat` where `std.c.Stat` is void (Linux); `ioctl` reads the
// entropy count libc++'s `random_device` reports.

pub extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
pub extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
pub extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
pub extern "c" fn close(fd: c_int) c_int;
pub extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;
pub extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
pub extern "c" fn fcntl(fd: c_int, cmd: c_int, ...) c_int;
pub extern "c" fn ioctl(fd: c_int, request: c_ulong, ...) c_int;
pub extern "c" fn fork() c_int;
pub extern "c" fn pipe(fds: *[2]c_int) c_int;
pub extern "c" fn dup2(old: c_int, new: c_int) c_int;
pub extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
pub extern "c" fn sysconf(name: c_int) c_long;
pub extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;
pub extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
pub extern "c" fn munmap(addr: ?*anyopaque, len: usize) c_int;
pub extern "c" fn madvise(addr: ?*anyopaque, len: usize, advice: c_int) c_int;
pub extern "c" fn posix_madvise(addr: ?*anyopaque, len: usize, advice: c_int) c_int;
pub extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;
pub extern "c" fn statx(dirfd: c_int, path: [*:0]const u8, flags: c_int, mask: std.os.linux.STATX, buf: *std.os.linux.Statx) c_int;
pub extern "c" fn backtrace(buffer: [*]?*anyopaque, size: c_int) c_int;
pub extern "c" fn backtrace_symbols_fd(buffer: [*]const ?*anyopaque, size: c_int, fd: c_int) void;
pub extern "c" fn inet_pton(af: c_int, src: [*:0]const u8, dst: *anyopaque) c_int;
/// Windows' `_fullpath`, `realpath`'s nearest.
pub extern "c" fn _fullpath(abs: [*]u8, rel: [*:0]const u8, max: usize) ?[*:0]u8;
/// MinGW's UCRT, behind the `stdin`, `stdout` and `stderr` macros.
pub extern "c" fn __acrt_iob_func(index: c_uint) *anyopaque;

// -----------------------------------------------------------------------------
// pthreads: thread affinity, which std does not expose

pub extern "c" fn pthread_self() std.c.pthread_t;
/// glibc's `getcpu`: the CPU and NUMA node the caller runs on.
pub extern "c" fn getcpu(cpu: *c_uint, node: *c_uint) c_int;
pub extern "c" fn pthread_setaffinity_np(thread: std.c.pthread_t, size: usize, set: *const anyopaque) c_int;
pub extern "c" fn pthread_getaffinity_np(thread: std.c.pthread_t, size: usize, set: *anyopaque) c_int;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "snprintf formats as C does" {
    var buf: [32]u8 = undefined;
    const n = snprintf(&buf, buf.len, "%.3f|%d", @as(f64, 2.5), @as(c_int, -7));
    try std.testing.expectEqualStrings("2.500|-7", buf[0..@intCast(n)]);
}
