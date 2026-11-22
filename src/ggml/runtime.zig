//! Process-level services for ggml: aborting, logging, timing, allocation.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml.c` (v1.0.0, `b1b770d90`), which is
//! 8,067 lines and split across several files here. This one covers the
//! preamble, roughly lines 36-630 and 8036-8067: the pieces that talk to the
//! operating system rather than to tensors. Each function names the C function
//! it replaces and the line it began at.
//!
//! # Fidelity notes
//!
//! - `ggml_abort` and `ggml_log_internal` are variadic and called from C with
//!   varargs, so they are defined variadic here. Note that a `va_list` must be
//!   handed to `vsnprintf` **by value**: passing a pointer compiles and links
//!   but silently formats garbage on aarch64. Where Zig cannot define a
//!   variadic function (`impl.zig_varargs`), `src/ggml/varargs.c` defines all
//!   three and calls back into `abortWith` and `logEmit` here.
//! - The macOS and Linux paths are ported. The C file also carries Windows
//!   and s390x variants of most of this, which are not.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const builtin = @import("builtin");

// -----------------------------------------------------------------------------
// libc and Mach declarations
//
// Declared rather than imported: these come from headers `@cImport` does not
// see, and the variadic signatures need the by-value `va_list` noted above.

/// `_WIN32`, and with this target's toolchain also `__MINGW32__`: the C
/// tests one or the other, and they coincide here.
const is_windows = builtin.os.tag == .windows;

/// `CLOCKS_PER_SEC` on macOS, and on glibc.
const clocks_per_sec: i64 = 1_000_000;

/// Mach VM interface, used by `ggml_aligned_malloc` on Apple OSes. The C reaches
/// for `vm_allocate` there rather than `posix_memalign` so that large tensor
/// buffers come straight from the kernel, page-aligned and untouched by malloc.
extern var mach_task_self_: c_uint;

const vm_flags_anywhere: c_int = 0x0001;
const kern_success: c_int = 0;
const kern_invalid_address: c_int = 1;
const kern_no_space: c_int = 3;

const einval: c_int = 22;
const enomem: c_int = 12;
const efault: c_int = 14;

// -----------------------------------------------------------------------------
// Graph identity

/// Counter behind `ggml_graph_next_uid`.
var graph_uid = std.atomic.Value(u64).init(0);

/// Ports `ggml_graph_next_uid` (ggml.c:57 @b1b770d90).
///
/// Return: a value unique across the process. Graphs use it to recognise that
/// two of them are the same, so a repeat must never occur.
pub fn ggml_graph_next_uid() callconv(.c) u64 {
    return graph_uid.fetchAdd(1, .monotonic);
}

// -----------------------------------------------------------------------------
// Aborting

/// Ports `g_abort_callback` (ggml.c:244 @b1b770d90), the storage behind
/// `ggml_set_abort_callback`.
var abort_callback: mod.impl.ggml_c.ggml_abort_callback_t = null;

/// Ports `ggml_print_backtrace_symbols` (ggml.c:147 @b1b770d90) -- the `__APPLE__`
/// variant, and the `__linux__ && __GLIBC__` one at ggml.c:139, which is the
/// same body.
///
/// The C file has four variants of this; macOS uses `backtrace` directly
/// rather than forking a debugger, because libdispatch does not survive a fork
/// and lldb attaching to its own parent crashes Terminal.
fn printBacktraceSymbols() void {
    var trace: [100]?*anyopaque = undefined;
    const nptrs = stdlib.libc.backtrace(&trace, trace.len);
    stdlib.libc.backtrace_symbols_fd(&trace, nptrs, std.posix.STDERR_FILENO);
}

/// Ports `ggml_print_backtrace`, `ggml_print_backtrace` (ggml.c:158, 239
/// @b1b770d90): the Unix arm and the `#else` that prints nothing.
///
/// Honours `GGML_NO_BACKTRACE`. The C also honours `GGML_BACKTRACE_LLDB` on
/// macOS, forking to attach a debugger; that path is not ported, because it is
/// documented upstream as crashing Terminal and is opt-in only. Windows takes
/// the `#else`, which does not read the variable either.
pub fn ggml_print_backtrace() callconv(.c) void {
    if (comptime is_windows) return;
    if (stdlib.env.getenv("GGML_NO_BACKTRACE") != null) return;
    if (comptime builtin.os.tag == .linux) return printBacktraceLinux();
    printBacktraceSymbols();
}

/// `PR_SET_PTRACER` from `<linux/prctl.h>`.
const pr_set_ptracer: c_int = 0x59616d61;

/// The `__linux__` arm of `ggml_print_backtrace` (ggml.c:158 @b1b770d90):
/// unless a debugger is already attached, fork and have the child attach
/// `gdb`, then `lldb`, to the parent, falling back to `backtrace` if neither
/// runs. The pipe holds the child back until the parent has allowed it to
/// ptrace, which Yama's `ptrace_scope = 1` requires.
fn printBacktraceLinux() void {
    const f = stdlib.libc.fopen("/proc/self/status", "r") orelse return;
    var size: usize = 0;
    var line: ?[*]u8 = null;
    while (true) {
        const length = stdlib.libc.getline(&line, &size, f);
        if (length <= 0) break;
        const l = line.?[0..@intCast(length)];
        if (std.mem.startsWith(u8, l, "TracerPid:") and
            (l.len != "TracerPid:\t0\n".len or l[l.len - 2] != '0'))
        {
            // Already being debugged, and the breakpoint is the later abort()
            std.c.free(line);
            _ = stdlib.libc.fclose(f);
            return;
        }
    }
    std.c.free(line);
    _ = stdlib.libc.fclose(f);

    var lock: [2]std.c.fd_t = .{ -1, -1 };
    _ = std.c.pipe(&lock); // Don't start gdb until after PR_SET_PTRACER

    const parent_pid = std.c.getpid();
    const child_pid = std.c.fork();
    if (child_pid < 0) {
        _ = std.c.close(lock[1]);
        _ = std.c.close(lock[0]);
        return;
    } else if (child_pid == 0) {
        var attach: [32]u8 = undefined;
        _ = stdlib.libc.snprintf(&attach, attach.len, "attach %d", parent_pid);
        const attach_z: [*:0]const u8 = @ptrCast(&attach);
        _ = std.c.close(lock[1]);
        _ = std.c.read(lock[0], @ptrCast(&lock), 1);
        _ = std.c.close(lock[0]);
        // try gdb
        _ = stdlib.libc.execlp("gdb", "gdb", "--batch", "-ex", "set style enabled on", "-ex", attach_z, "-ex", "bt -frame-info source-and-location", "-ex", "detach", "-ex", "quit", @as(?[*:0]const u8, null));
        // try lldb
        _ = stdlib.libc.execlp("lldb", "lldb", "--batch", "-o", "bt", "-o", "quit", "-p", attach_z + "attach ".len, @as(?[*:0]const u8, null));
        // gdb failed, fallback to backtrace_symbols
        printBacktraceSymbols();
        std.c._Exit(0);
    } else {
        _ = std.c.prctl(pr_set_ptracer, @as(c_ulong, @intCast(child_pid)));
        _ = std.c.close(lock[1]);
        _ = std.c.close(lock[0]);
        _ = std.c.waitpid(child_pid, null, 0);
    }
}

/// Ports `ggml_set_abort_callback` (ggml.c:247 @b1b770d90).
///
/// Return: the previous callback, so a caller can chain or restore it. Passing
/// null restores the default of printing a message and a backtrace.
pub fn ggml_set_abort_callback(callback: mod.impl.ggml_c.ggml_abort_callback_t) callconv(.c) mod.impl.ggml_c.ggml_abort_callback_t {
    const previous = abort_callback;
    abort_callback = callback;
    return previous;
}

/// Ports `ggml_abort` (ggml.c:253 @b1b770d90).
///
/// Formats `file:line: ` followed by the message, hands it to the callback if
/// one is set, and otherwise prints it with a backtrace. Never returns.
pub fn ggmlAbort(file: [*:0]const u8, line: c_int, fmt: [*:0]const u8, ...) callconv(.c) noreturn {
    _ = stdlib.libc.fflush(stdlib.stdio.cStream(.stdout));

    var message: [2048]u8 = undefined;
    const offset = stdlib.libc.snprintf(&message, message.len, "%s:%d: ", file, line);

    var ap = @cVaStart();
    defer @cVaEnd(&ap);
    if (offset > 0 and @as(usize, @intCast(offset)) < message.len) {
        const used: usize = @intCast(offset);
        _ = stdlib.libc.vsnprintf(message[used..].ptr, message.len - used, fmt, mod.impl.cVaList(&ap));
    }

    // `snprintf` and `vsnprintf` both terminate what they write.
    abortWith(@ptrCast(&message));
}

/// The second half of `ggml_abort` (ggml.c:253 @b1b770d90), once the message
/// is formatted: exported as `ggml_zig_abort_message` for `varargs.c` where
/// that file defines `ggml_abort`.
pub fn abortWith(message: [*:0]const u8) callconv(.c) noreturn {
    // `fflush(stdout)`, for what a tool wrote through `stdlib.stdio`; the C
    // library's own stdout was flushed before the message was formatted
    @import("stdlib").stdio.stdout().flush();
    if (abort_callback) |callback| {
        callback(message);
    } else {
        _ = stdlib.libc.fprintf(stdlib.stdio.cStream(.stderr), "%s\n", message);
        ggml_print_backtrace();
    }

    std.c.abort();
}

// -----------------------------------------------------------------------------
// Logging

/// Ports `struct ggml_logger_state` (ggml.c:281 @b1b770d90).
const LoggerState = struct {
    log_callback: mod.impl.ggml_c.ggml_log_callback,
    log_callback_user_data: ?*anyopaque,
};

var logger_state: LoggerState = .{
    .log_callback = ggml_log_callback_default,
    .log_callback_user_data = null,
};

/// Ports `ggml_log_internal_v` (ggml.c:287 @b1b770d90).
///
/// Formats into a small stack buffer first and only allocates when the message
/// does not fit, which is why the argument list has to be copied before the
/// first attempt consumes it.
fn logInternalV(level: mod.impl.ggml_c.enum_ggml_log_level, format: ?[*:0]const u8, ap: std.lang.VaList) void {
    const fmt = format orelse return;

    var ap_first = ap;
    var ap_copy = @cVaCopy(@constCast(&ap));
    defer @cVaEnd(&ap_copy);

    var buffer: [128]u8 = undefined;
    const len = stdlib.libc.vsnprintf(&buffer, buffer.len, fmt, mod.impl.cVaList(&ap_first));
    if (len < buffer.len) {
        logger_state.log_callback.?(level, &buffer, logger_state.log_callback_user_data);
        return;
    }

    const size: usize = @intCast(len + 1);
    const buffer2: [*]u8 = @ptrCast(std.c.calloc(size, 1) orelse return);
    defer std.c.free(buffer2);
    _ = stdlib.libc.vsnprintf(buffer2, size, fmt, mod.impl.cVaList(&ap_copy));
    buffer2[size - 1] = 0;
    logger_state.log_callback.?(level, @ptrCast(buffer2), logger_state.log_callback_user_data);
}

/// Ports `ggml_log_internal` (ggml.c:307 @b1b770d90).
pub fn ggmlLogInternal(level: mod.impl.ggml_c.enum_ggml_log_level, format: ?[*:0]const u8, ...) callconv(.c) void {
    var ap = @cVaStart();
    defer @cVaEnd(&ap);
    logInternalV(level, format, ap);
}

/// Hands a message `varargs.c` has formatted to the current callback: the last
/// line of `ggml_log_internal_v` (ggml.c:287 @b1b770d90). Exported as
/// `ggml_zig_log_emit` where that file defines `ggml_log_internal`.
pub fn logEmit(level: mod.impl.ggml_c.enum_ggml_log_level, text: [*:0]const u8) callconv(.c) void {
    logger_state.log_callback.?(level, text, logger_state.log_callback_user_data);
}

/// Ports `ggml_log_callback_default` (ggml.c:314 @b1b770d90).
pub fn ggml_log_callback_default(
    level: mod.impl.ggml_c.enum_ggml_log_level,
    text: [*c]const u8,
    user_data: ?*anyopaque,
) callconv(.c) void {
    _ = level;
    _ = user_data;
    _ = stdlib.libc.fputs(@ptrCast(text), stdlib.stdio.cStream(.stderr));
    _ = stdlib.libc.fflush(stdlib.stdio.cStream(.stderr));
}

/// Ports `ggml_log_get` (ggml.c:8084 @b1b770d90).
pub fn ggml_log_get(log_callback: *mod.impl.ggml_c.ggml_log_callback, user_data: *?*anyopaque) callconv(.c) void {
    log_callback.* = logger_state.log_callback;
    user_data.* = logger_state.log_callback_user_data;
}

/// Ports `ggml_log_set` (ggml.c:8089 @b1b770d90).
///
/// A null callback restores the default rather than disabling logging, so
/// there is always something to call.
pub fn ggml_log_set(log_callback: mod.impl.ggml_c.ggml_log_callback, user_data: ?*anyopaque) callconv(.c) void {
    logger_state.log_callback = log_callback orelse ggml_log_callback_default;
    logger_state.log_callback_user_data = user_data;
}

// -----------------------------------------------------------------------------
// Aligned allocation

/// Ports `GGML_TSAN_ENABLED` (ggml.c:335 @b1b770d90): the thread sanitizer
/// does not intercept `vm_allocate` and `vm_deallocate`, so under it macOS
/// takes the `posix_memalign` arm. The C asks the compiler's
/// `__has_feature(thread_sanitizer)`; this module's own sanitizer setting is
/// the same question.
const tsan_enabled = builtin.sanitize_thread;

/// True where `ggml_aligned_malloc` and `ggml_aligned_free` take the C's
/// `#elif defined(__APPLE__) && !defined(GGML_TSAN_ENABLED)` arm, the Mach VM.
/// Every Apple OS.
const use_vm_allocate = builtin.os.tag.isDarwin() and !tsan_enabled;

/// Ports `ggml_aligned_malloc` (ggml.c:343 @b1b770d90).
///
/// On every Apple OS this goes to the Mach VM rather than `posix_memalign`,
/// which is what the C does under `__APPLE__` without the thread sanitizer:
/// libmalloc can keep freed pages in the process footprint, and
/// `vm_deallocate` returns them at once. The alignment argument is unused on
/// that path because `vm_allocate` always returns page-aligned memory. Every
/// other target takes the C's `#else` arm, `posix_memalign` at 64 bytes.
/// MinGW takes the first arm, `_aligned_malloc` at 64 bytes, which has
/// neither the zero-size warning nor the failure log.
///
/// Return: the allocation, or null on failure after logging why.
pub fn ggml_aligned_malloc(size: usize) callconv(.c) ?*anyopaque {
    if (comptime is_windows) return stdlib.libc._aligned_malloc(size, 64);

    if (size == 0) {
        mod.impl.logWarn("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n", .{});
        return null;
    }

    if (comptime !use_vm_allocate) {
        var aligned_memory: ?*anyopaque = null;
        const result = std.c.posix_memalign(&aligned_memory, 64, size);
        if (result != 0) {
            logAllocFailure(result, size);
            return null;
        }
        return aligned_memory;
    }

    var address: usize = 0;
    const status = stdlib.libc.vm_allocate(mach_task_self_, &address, size, vm_flags_anywhere);
    const result: c_int = switch (status) {
        kern_success => 0,
        kern_invalid_address => einval,
        kern_no_space => enomem,
        else => efault,
    };

    if (result != 0) {
        logAllocFailure(result, size);
        return null;
    }

    return @ptrFromInt(address);
}

/// The failure branch of `ggml_aligned_malloc` (ggml.c:343 @b1b770d90), shared
/// by both allocation arms. `EINVAL` and `ENOMEM` are 22 and 12 on Darwin and
/// Linux alike.
fn logAllocFailure(result: c_int, size: usize) void {
    const error_desc: [*:0]const u8 = switch (result) {
        einval => "invalid alignment value",
        enomem => "insufficient memory",
        else => "unknown allocation error",
    };
    mod.impl.logError("%s: %s (attempted to allocate %6.2f MB)\n", .{
        "ggml_aligned_malloc",
        error_desc,
        @as(f64, @floatFromInt(size)) / (1024.0 * 1024.0),
    });
}

/// Ports `ggml_aligned_free` (ggml.c:400 @b1b770d90).
///
/// On an Apple OS `size` is required, not decorative: `vm_deallocate` needs the
/// length that was allocated, so this cannot be a plain `free`. Elsewhere it
/// is the C's plain `free`, or on MinGW `_aligned_free`, and `size` is unused.
pub fn ggml_aligned_free(ptr: ?*anyopaque, size: usize) callconv(.c) void {
    if (comptime is_windows) return stdlib.libc._aligned_free(ptr);
    if (comptime !use_vm_allocate) return std.c.free(ptr);
    const p = ptr orelse return;
    _ = std.c.vm_deallocate(mach_task_self_, @intFromPtr(p), size);
}

// -----------------------------------------------------------------------------
// Status and version

/// Ports `ggml_status_to_string` (ggml.c:450 @b1b770d90).
pub fn ggml_status_to_string(status: mod.impl.ggml_c.enum_ggml_status) callconv(.c) [*:0]const u8 {
    return switch (status) {
        mod.impl.ggml_c.GGML_STATUS_ALLOC_FAILED => "GGML status: error (failed to allocate memory)",
        mod.impl.ggml_c.GGML_STATUS_FAILED => "GGML status: error (operation failed)",
        mod.impl.ggml_c.GGML_STATUS_SUCCESS => "GGML status: success",
        mod.impl.ggml_c.GGML_STATUS_ABORTED => "GGML status: warning (operation aborted)",
        else => "GGML status: unknown",
    };
}

/// Ports `ggml_version` (ggml.c:525 @b1b770d90).
pub fn ggml_version() callconv(.c) [*:0]const u8 {
    return build_version;
}

/// Ports `ggml_commit` (ggml.c:529 @b1b770d90).
pub fn ggml_commit() callconv(.c) [*:0]const u8 {
    return build_commit;
}

/// Mirrors the `GGML_VERSION` and `GGML_COMMIT` macros of the
/// `ggml-version.h` CMake configures from its template: ggml's own
/// version, not llama.cpp's, and the pin's short commit. They must equal
/// `build/upstream.zig`'s `ggml_version` and `commit`, which
/// `scripts/check-reference-pin` asserts.
const build_version: [*:0]const u8 = "1.0.0";
const build_commit: [*:0]const u8 = "b1b770d90";

/// Ports `ggml_guid_matches` (ggml.c:521 @b1b770d90).
pub fn ggml_guid_matches(guid_a: mod.impl.ggml_c.ggml_guid_t, guid_b: mod.impl.ggml_c.ggml_guid_t) callconv(.c) bool {
    const a: [*]const u8 = @ptrCast(guid_a);
    const b: [*]const u8 = @ptrCast(guid_b);
    return std.mem.eql(u8, a[0..16], b[0..16]);
}

// -----------------------------------------------------------------------------
// Float conversions
//
// The single-value entry points exist so callers outside ggml can convert
// without the header's macros. The row variants are what the type traits table
// points at, so they are on the hot path for every dequantized tensor.

/// Ports `ggml_fp16_to_fp32` (ggml.c:462 @b1b770d90).
pub fn ggml_fp16_to_fp32(x: mod.impl.ggml_c.ggml_fp16_t) callconv(.c) f32 {
    return mod.impl.fp16ToFp32(x);
}

/// Ports `ggml_fp32_to_fp16` (ggml.c:467 @b1b770d90).
pub fn ggml_fp32_to_fp16(x: f32) callconv(.c) mod.impl.ggml_c.ggml_fp16_t {
    return mod.impl.fp32ToFp16(x);
}

/// Ports `ggml_bf16_to_fp32` (ggml.c:472 @b1b770d90).
///
/// **The parameter is spelled `u16`, not `c.ggml_bf16_t`, because Zig
/// 0.16 miscompiles the struct.** `ggml_bf16_t` is `struct { uint16_t
/// bits; }` — two bytes — and an `extern struct` that size received by
/// value across the C ABI arrives as **zeros** on aarch64-macos. This
/// function returned 0.0 for every input from the day `ggml.c` was
/// ported, and no gate saw it; `harness/abi_structs.zig` is the gate that
/// does now. `u16` has the same size and is passed correctly, and the C
/// caller is unaffected: `ggml_bf16_t` is a one-field struct over exactly
/// those bits.
pub fn ggml_bf16_to_fp32(x: u16) callconv(.c) f32 {
    return mod.impl.bf16ToFp32(x);
}

/// Ports `ggml_fp32_to_bf16` (ggml.c:477 @b1b770d90).
pub fn ggml_fp32_to_bf16(x: f32) callconv(.c) mod.impl.ggml_c.ggml_bf16_t {
    return .{ .bits = mod.impl.fp32ToBf16(x) };
}

/// Ports `ggml_fp16_to_fp32_row` (ggml.c:481 @b1b770d90).
pub fn ggml_fp16_to_fp32_row(x: [*]const mod.impl.ggml_c.ggml_fp16_t, y: [*]f32, n: i64) callconv(.c) void {
    for (0..@intCast(n)) |i| y[i] = mod.impl.fp16ToFp32(x[i]);
}

/// Ports `ggml_fp32_to_fp16_row` (ggml.c:487 @b1b770d90).
pub fn ggml_fp32_to_fp16_row(x: [*]const f32, y: [*]mod.impl.ggml_c.ggml_fp16_t, n: i64) callconv(.c) void {
    for (0..@intCast(n)) |i| y[i] = mod.impl.fp32ToFp16(x[i]);
}

/// Ports `ggml_bf16_to_fp32_row` (ggml.c:493 @b1b770d90).
pub fn ggml_bf16_to_fp32_row(x: [*]const mod.impl.ggml_c.ggml_bf16_t, y: [*]f32, n: i64) callconv(.c) void {
    for (0..@intCast(n)) |i| y[i] = mod.impl.bf16ToFp32(x[i].bits);
}

/// Ports `ggml_fp32_to_bf16_row_ref` (ggml.c:499 @b1b770d90).
pub fn ggml_fp32_to_bf16_row_ref(x: [*]const f32, y: [*]mod.impl.ggml_c.ggml_bf16_t, n: i64) callconv(.c) void {
    for (0..@intCast(n)) |i| y[i] = .{ .bits = mod.impl.fp32ToBf16(x[i]) };
}

/// Ports `ggml_fp32_to_bf16_row` (ggml.c:505 @b1b770d90).
///
/// The C has an AVX512-BF16 fast path here; on ARM it falls through to the
/// same scalar loop as the reference version.
pub fn ggml_fp32_to_bf16_row(x: [*]const f32, y: [*]mod.impl.ggml_c.ggml_bf16_t, n: i64) callconv(.c) void {
    for (0..@intCast(n)) |i| y[i] = .{ .bits = mod.impl.fp32ToBf16(x[i]) };
}

// -----------------------------------------------------------------------------
// Timing

/// Ports `timer_freq`, `timer_start` (ggml.c:538, 538 @b1b770d90): the
/// performance counter's frequency and its reading at `ggml_time_init`.
/// Windows only.
var timer_freq: i64 = 0;
var timer_start: i64 = 0;

/// Ports `ggml_time_init_once` (ggml.c:539 @b1b770d90).
fn timeInitOnce(once: *std.os.windows.INIT_ONCE, param: ?*anyopaque, ctx: ?*?*anyopaque) callconv(.winapi) mod.win32.BOOL {
    _ = once;
    _ = param;
    _ = ctx;
    var t: i64 = undefined;
    _ = mod.win32.QueryPerformanceFrequency(&t);
    timer_freq = t;
    _ = mod.win32.QueryPerformanceCounter(&t);
    timer_start = t;
    return 1;
}

/// Ports `ggml_time_init`, `ggml_time_init` (ggml.c:553, 568 @b1b770d90).
///
/// A no-op outside Windows, where the C initialises a performance counter
/// once per process.
pub fn ggml_time_init() callconv(.c) void {
    if (comptime !is_windows) return;
    const S = struct {
        var once: std.os.windows.INIT_ONCE = std.os.windows.INIT_ONCE_STATIC_INIT;
    };
    _ = mod.win32.InitOnceExecuteOnce(&S.once, timeInitOnce, null, null);
}

/// Ports `ggml_time_ms`, `ggml_time_ms` (ggml.c:557, 569 @b1b770d90).
pub fn ggml_time_ms() callconv(.c) i64 {
    if (comptime is_windows) {
        var t: i64 = undefined;
        _ = mod.win32.QueryPerformanceCounter(&t);
        return @intFromFloat(@as(f64, @floatFromInt(t - timer_start)) / @as(f64, @floatFromInt(timer_freq)) * 1000);
    }
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

/// Ports `ggml_time_us`, `ggml_time_us` (ggml.c:562, 575 @b1b770d90).
pub fn ggml_time_us() callconv(.c) i64 {
    if (comptime is_windows) {
        var t: i64 = undefined;
        _ = mod.win32.QueryPerformanceCounter(&t);
        return @intFromFloat(@as(f64, @floatFromInt(t - timer_start)) / @as(f64, @floatFromInt(timer_freq)) * 1000000);
    }
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, ts.sec) * 1_000_000 + @divTrunc(@as(i64, ts.nsec), 1_000);
}

/// Ports `ggml_cycles` (ggml.c:582 @b1b770d90).
pub fn ggml_cycles() callconv(.c) i64 {
    return @intCast(stdlib.libc.clock());
}

/// Ports `ggml_cycles_per_ms` (ggml.c:586 @b1b770d90).
pub fn ggml_cycles_per_ms() callconv(.c) i64 {
    return @divTrunc(clocks_per_sec, 1000);
}

// -----------------------------------------------------------------------------
// Threadpool parameters

/// Ports `ggml_threadpool_params_init` (ggml.c:8094 @b1b770d90).
pub fn ggml_threadpool_params_init(p: *mod.impl.ggml_c.struct_ggml_threadpool_params, n_threads: c_int) callconv(.c) void {
    p.n_threads = n_threads;
    p.prio = 0; // normal, or inherited
    p.poll = 50; // hybrid polling
    p.strict_cpu = false; // all threads share one cpumask
    p.paused = false;
    // All-zero means "use the default affinity", which is usually inherited.
    @memset(&p.cpumask, false);
}

/// Ports `ggml_threadpool_params_default` (ggml.c:8103 @b1b770d90).
pub fn ggml_threadpool_params_default(n_threads: c_int) callconv(.c) mod.impl.ggml_c.struct_ggml_threadpool_params {
    var p: mod.impl.ggml_c.struct_ggml_threadpool_params = undefined;
    ggml_threadpool_params_init(&p, n_threads);
    return p;
}

/// Ports `ggml_threadpool_params_match` (ggml.c:8109 @b1b770d90).
pub fn ggml_threadpool_params_match(
    p0: *const mod.impl.ggml_c.struct_ggml_threadpool_params,
    p1: *const mod.impl.ggml_c.struct_ggml_threadpool_params,
) callconv(.c) bool {
    if (p0.n_threads != p1.n_threads) return false;
    if (p0.prio != p1.prio) return false;
    if (p0.poll != p1.poll) return false;
    if (p0.strict_cpu != p1.strict_cpu) return false;
    // `paused` is deliberately not compared, matching the C.
    return std.mem.eql(bool, &p0.cpumask, &p1.cpumask);
}

// -----------------------------------------------------------------------------
// Unit Tests

test "graph uids are unique and monotonic" {
    const a = ggml_graph_next_uid();
    const b = ggml_graph_next_uid();
    try std.testing.expect(b > a);
}

test "status strings cover every status" {
    try std.testing.expectEqualStrings("GGML status: success", std.mem.span(ggml_status_to_string(mod.impl.ggml_c.GGML_STATUS_SUCCESS)));
    try std.testing.expectEqualStrings("GGML status: unknown", std.mem.span(ggml_status_to_string(@intCast(999))));
}

test "threadpool params round-trip and compare" {
    const a = ggml_threadpool_params_default(4);
    var b = ggml_threadpool_params_default(4);
    try std.testing.expect(ggml_threadpool_params_match(&a, &b));
    b.n_threads = 8;
    try std.testing.expect(!ggml_threadpool_params_match(&a, &b));
}

test "aligned malloc returns aligned memory and frees it" {
    const size = 4096 * 4;
    const p = ggml_aligned_malloc(size) orelse return error.AllocFailed;
    // `vm_allocate` hands out whole pages; `posix_memalign` promises the 64
    // bytes asked of it.
    const alignment: usize = if (builtin.os.tag == .macos) 4096 else 64;
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(p) % alignment);
    // Writing proves the pages are actually mapped, not just reserved.
    @memset(@as([*]u8, @ptrCast(p))[0..size], 0xAB);
    ggml_aligned_free(p, size);
}

// -----------------------------------------------------------------------------
// File access

/// Ports `ggml_fopen` (ggml.c:614 @b1b770d90), both arms.
///
/// A plain `fopen`, except on Windows, where both arguments are converted from
/// UTF-8 to wide characters and the file is opened with `_wfopen`, so a path
/// outside the ANSI code page still opens.
pub fn ggml_fopen(fname: [*:0]const u8, mode: [*:0]const u8) callconv(.c) ?*std.c.FILE {
    if (comptime !is_windows) return @ptrCast(stdlib.libc.fopen(fname, mode));

    var file: ?*std.c.FILE = null;
    if (mbstowcs(fname)) |wfname| {
        if (mbstowcs(mode)) |wmode| {
            file = @ptrCast(@alignCast(stdlib.libc._wfopen(wfname, wmode)));
            std.c.free(wmode);
        }
        std.c.free(wfname);
    }
    return file;
}

/// Ports `ggml_mbstowcs` (ggml.c:595 @b1b770d90): UTF-8 to a `malloc`ed,
/// NUL-terminated wide string, or null with `errno` set to `EINVAL`.
fn mbstowcs(mbs: [*:0]const u8) ?[*:0]u16 {
    var wlen = mod.win32.MultiByteToWideChar(mod.win32.CP_UTF8, 0, mbs, -1, null, 0);
    if (wlen == 0) {
        std.c._errno().* = einval;
        return null;
    }

    const wbuf: [*]u16 = @ptrCast(@alignCast(mod.impl.ggmlMalloc(@as(usize, @intCast(wlen)) * @sizeOf(u16)).?));
    wlen = mod.win32.MultiByteToWideChar(mod.win32.CP_UTF8, 0, mbs, -1, wbuf, wlen);
    if (wlen == 0) {
        std.c.free(wbuf);
        std.c._errno().* = einval;
        return null;
    }

    return @ptrCast(wbuf);
}
