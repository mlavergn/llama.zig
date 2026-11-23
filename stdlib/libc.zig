//! Every C library function the port still calls, declared once: the
//! ledger of what is left of the libc dependency, each group with the
//! reason it is still the C library's.
//!
//! # Provenance
//!
//! **Not a port.** A file that needs one of these calls `libc.<name>`;
//! nothing else declares a C library function. libm is `libm.zig`.
//!
//! A function std declares is an alias of std's, which exists only on the
//! targets that have it; an `extern` here is one std lacks, or one whose std
//! type is stricter than C's (`mlock`, `setlocale`).
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
pub extern "c" fn printf(format: [*:0]const u8, ...) c_int;
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
pub extern "c" fn ftell(stream: ?*anyopaque) c_long;
pub extern "c" fn tmpfile() ?*anyopaque;
pub extern "c" fn fileno(stream: ?*anyopaque) c_int;
pub extern "c" fn fscanf(stream: ?*anyopaque, format: [*:0]const u8, ...) c_int;
pub extern "c" fn sscanf(s: [*:0]const u8, format: [*:0]const u8, ...) c_int;
pub extern "c" fn ferror(stream: ?*anyopaque) c_int;
pub extern "c" fn getwchar() u32;
pub extern "c" fn fgets(buf: [*]u8, n: c_int, stream: ?*anyopaque) ?[*]u8;
pub extern "c" fn getline(line: *?[*]u8, n: *usize, stream: ?*anyopaque) isize;
/// Windows' `fgetwc`: `wint_t` is an `unsigned short` there.
pub extern "c" fn fgetwc(stream: ?*anyopaque) u16;
/// The UCRT's descriptor and 64-bit offset calls on a stream.
pub extern "c" fn _fileno(stream: ?*anyopaque) c_int;
pub extern "c" fn _get_osfhandle(fd: c_int) isize;
pub extern "c" fn _ftelli64(stream: ?*anyopaque) i64;
pub extern "c" fn _fseeki64(stream: ?*anyopaque, offset: i64, whence: c_int) c_int;
pub extern "c" fn _setmode(fd: c_int, mode: c_int) c_int;
/// The UCRT's descriptor calls, where std's `open` and `dup2` take `HANDLE`s.
pub extern "c" fn _open(path: [*:0]const u8, flags: c_int, ...) c_int;
pub extern "c" fn _dup2(old: c_int, new: c_int) c_int;

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

pub extern "c" fn _aligned_malloc(size: usize, alignment: usize) ?*anyopaque;
pub extern "c" fn _aligned_free(ptr: ?*anyopaque) void;
pub extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
pub extern "c" fn rand() c_int;
pub extern "c" fn arc4random() u32;
pub extern "c" fn rand_s(r: *c_uint) c_int;
pub extern "c" fn strlen(s: [*c]const u8) usize;
/// The `"C"` locale's ctype, which `llama-chat.cpp` hands a plain `char`
/// (negative above 0x80 on Darwin): called rather than tabulated.
pub extern "c" fn isspace(ch: c_int) c_int;
pub extern "c" fn toupper(ch: c_int) c_int;
pub extern "c" fn wcwidth(wc: u32) c_int;
pub extern "c" fn iswspace(wc: u32) c_int;
pub extern "c" fn atexit(f: *const fn () callconv(.c) void) c_int;
/// C's conversions where a caller walks the end pointer, as a harness's
/// reference-matching parser does (`stdlib.num` parses slices).
pub extern "c" fn strtol(s: [*:0]const u8, end: ?*[*:0]u8, base: c_int) c_long;
pub extern "c" fn strtod(s: [*:0]const u8, end: ?*[*:0]u8) f64;
pub extern "c" fn atoi(s: [*:0]const u8) c_int;
pub extern "c" fn atof(s: [*:0]const u8) f64;
pub extern "c" fn strerror(errnum: c_int) [*:0]const u8;
/// `setlocale`, declared here because std's takes glibc's category numbers on
/// every OS; pass an `LC` constant.
pub extern "c" fn setlocale(category: c_int, locale: ?[*:0]const u8) ?[*:0]const u8;

/// The `LC_*` categories `<locale.h>` defines on this target: glibc's and
/// musl's on Linux, BSD's on Darwin, the UCRT's on Windows.
pub const LC = if (builtin.os.tag == .linux) struct {
    pub const CTYPE: c_int = 0;
    pub const NUMERIC: c_int = 1;
    pub const ALL: c_int = 6;
} else struct {
    pub const ALL: c_int = 0;
    pub const CTYPE: c_int = 2;
    pub const NUMERIC: c_int = 4;
};
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

pub extern "c" fn posix_madvise(addr: ?*anyopaque, len: usize, advice: c_int) c_int;
/// `mlock`, declared here because std's demands a page-aligned address and
/// C's takes any.
pub extern "c" fn mlock(addr: ?*const anyopaque, len: usize) c_int;
pub extern "c" fn execlp(file: [*:0]const u8, arg0: [*:0]const u8, ...) c_int;
/// glibc's resolver reset, which httplib calls before `getaddrinfo`.
pub extern "c" fn __res_init() c_int;
/// Darwin's `vm_allocate`; `vm_deallocate` is std's.
pub extern "c" fn vm_allocate(target: c_uint, address: *usize, size: usize, flags: c_int) c_int;
pub extern "c" fn backtrace(buffer: [*]?*anyopaque, size: c_int) c_int;
pub extern "c" fn backtrace_symbols_fd(buffer: [*]const ?*anyopaque, size: c_int, fd: c_int) void;
pub extern "c" fn inet_pton(af: c_int, src: [*:0]const u8, dst: *anyopaque) c_int;
/// Windows' `_fullpath`, `realpath`'s nearest.
pub extern "c" fn _fullpath(abs: [*]u8, rel: [*:0]const u8, max: usize) ?[*:0]u8;
/// MinGW's UCRT, behind the `stdin`, `stdout` and `stderr` macros.
pub extern "c" fn __acrt_iob_func(index: c_uint) *anyopaque;

// -----------------------------------------------------------------------------
// Sockets: httplib's, until the socket pass moves it to `std.Io.net`

// -----------------------------------------------------------------------------
// pthreads and the scheduler: what std does not expose (affinity is
// `stdlib.thread`'s syscalls)

pub extern "c" fn pthread_equal(a: std.c.pthread_t, b: std.c.pthread_t) c_int;
/// `param` is a `struct sched_param`.
pub extern "c" fn pthread_setschedparam(thread: std.c.pthread_t, policy: c_int, param: *const anyopaque) c_int;
pub extern "c" fn setpriority(which: c_int, who: c_uint, prio: c_int) c_int;

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
