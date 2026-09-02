//! Process-level services for ggml: aborting, logging, timing, allocation.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml.c` (v0.3.0, `c1d0e7a00`), which is
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
//!   but silently formats garbage on aarch64.
//! - Only the macOS paths are ported. The C file carries Windows, Linux, and
//!   s390x variants of most of this; those are Stage 5's problem, and the
//!   places where a second platform will need code are marked.

const std = @import("std");
const impl = @import("impl.zig");
const c = impl.c;

// -----------------------------------------------------------------------------
// libc and Mach declarations
//
// Declared rather than imported: these come from headers `@cImport` does not
// see, and the variadic signatures need the by-value `va_list` noted above.

extern fn vsnprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ap: std.builtin.VaList) c_int;
extern fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;
extern fn fputs(s: [*:0]const u8, stream: *anyopaque) c_int;
extern fn fflush(stream: ?*anyopaque) c_int;
extern fn fprintf(stream: *anyopaque, fmt: [*:0]const u8, ...) c_int;
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*std.c.FILE;
extern fn abort() noreturn;
extern fn clock() c_long;
extern fn backtrace(buffer: [*]?*anyopaque, size: c_int) c_int;
extern fn backtrace_symbols_fd(buffer: [*]const ?*anyopaque, size: c_int, fd: c_int) void;

extern var __stdoutp: *anyopaque;
extern var __stderrp: *anyopaque;

/// `CLOCKS_PER_SEC` on macOS.
const clocks_per_sec: i64 = 1_000_000;

/// Mach VM interface, used by `ggml_aligned_malloc` on macOS. The C reaches
/// for `vm_allocate` there rather than `posix_memalign` so that large tensor
/// buffers come straight from the kernel, page-aligned and untouched by malloc.
extern var mach_task_self_: c_uint;
extern fn vm_allocate(target: c_uint, address: *usize, size: usize, flags: c_int) c_int;
extern fn vm_deallocate(target: c_uint, address: usize, size: usize) c_int;

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

/// Ports `ggml_graph_next_uid` (ggml.c:56 @c1d0e7a00).
///
/// Return: a value unique across the process. Graphs use it to recognise that
/// two of them are the same, so a repeat must never occur.
export fn ggml_graph_next_uid() u64 {
    return graph_uid.fetchAdd(1, .monotonic);
}

// -----------------------------------------------------------------------------
// Aborting

/// Ports `g_abort_callback` (ggml.c:243 @c1d0e7a00), the storage behind
/// `ggml_set_abort_callback`.
var abort_callback: c.ggml_abort_callback_t = null;

/// Ports `ggml_print_backtrace_symbols` (ggml.c:146 @c1d0e7a00) -- the `__APPLE__`
/// variant, which is the one this target compiles.
///
/// The C file has four variants of this; macOS uses `backtrace` directly
/// rather than forking a debugger, because libdispatch does not survive a fork
/// and lldb attaching to its own parent crashes Terminal.
fn printBacktraceSymbols() void {
    var trace: [100]?*anyopaque = undefined;
    const nptrs = backtrace(&trace, trace.len);
    backtrace_symbols_fd(&trace, nptrs, std.posix.STDERR_FILENO);
}

/// Ports `ggml_print_backtrace` (ggml.c:157 @c1d0e7a00).
///
/// Honours `GGML_NO_BACKTRACE`. The C also honours `GGML_BACKTRACE_LLDB` on
/// macOS, forking to attach a debugger; that path is not ported, because it is
/// documented upstream as crashing Terminal and is opt-in only.
export fn ggml_print_backtrace() void {
    if (getenv("GGML_NO_BACKTRACE") != null) return;
    printBacktraceSymbols();
}

/// Ports `ggml_set_abort_callback` (ggml.c:246 @c1d0e7a00).
///
/// Return: the previous callback, so a caller can chain or restore it. Passing
/// null restores the default of printing a message and a backtrace.
export fn ggml_set_abort_callback(callback: c.ggml_abort_callback_t) c.ggml_abort_callback_t {
    const previous = abort_callback;
    abort_callback = callback;
    return previous;
}

/// Ports `ggml_abort` (ggml.c:252 @c1d0e7a00).
///
/// Formats `file:line: ` followed by the message, hands it to the callback if
/// one is set, and otherwise prints it with a backtrace. Never returns.
export fn ggml_abort(file: [*:0]const u8, line: c_int, fmt: [*:0]const u8, ...) callconv(.c) noreturn {
    _ = fflush(__stdoutp);

    var message: [2048]u8 = undefined;
    const offset = snprintf(&message, message.len, "%s:%d: ", file, line);

    var ap = @cVaStart();
    defer @cVaEnd(&ap);
    if (offset > 0 and @as(usize, @intCast(offset)) < message.len) {
        const used: usize = @intCast(offset);
        _ = vsnprintf(message[used..].ptr, message.len - used, fmt, ap);
    }

    if (abort_callback) |callback| {
        callback(&message);
    } else {
        _ = fprintf(__stderrp, "%s\n", &message);
        ggml_print_backtrace();
    }

    abort();
}

// -----------------------------------------------------------------------------
// Logging

/// Ports `struct ggml_logger_state` (ggml.c:280 @c1d0e7a00).
const LoggerState = struct {
    log_callback: c.ggml_log_callback,
    log_callback_user_data: ?*anyopaque,
};

var logger_state: LoggerState = .{
    .log_callback = ggml_log_callback_default,
    .log_callback_user_data = null,
};

/// Ports `ggml_log_internal_v` (ggml.c:286 @c1d0e7a00).
///
/// Formats into a small stack buffer first and only allocates when the message
/// does not fit, which is why the argument list has to be copied before the
/// first attempt consumes it.
fn logInternalV(level: c.enum_ggml_log_level, format: ?[*:0]const u8, ap: std.builtin.VaList) void {
    const fmt = format orelse return;

    var ap_copy = @cVaCopy(@constCast(&ap));
    defer @cVaEnd(&ap_copy);

    var buffer: [128]u8 = undefined;
    const len = vsnprintf(&buffer, buffer.len, fmt, ap);
    if (len < buffer.len) {
        logger_state.log_callback.?(level, &buffer, logger_state.log_callback_user_data);
        return;
    }

    const size: usize = @intCast(len + 1);
    const buffer2: [*]u8 = @ptrCast(std.c.calloc(size, 1) orelse return);
    defer std.c.free(buffer2);
    _ = vsnprintf(buffer2, size, fmt, ap_copy);
    buffer2[size - 1] = 0;
    logger_state.log_callback.?(level, @ptrCast(buffer2), logger_state.log_callback_user_data);
}

/// Ports `ggml_log_internal` (ggml.c:306 @c1d0e7a00).
export fn ggml_log_internal(level: c.enum_ggml_log_level, format: ?[*:0]const u8, ...) callconv(.c) void {
    var ap = @cVaStart();
    defer @cVaEnd(&ap);
    logInternalV(level, format, ap);
}

/// Ports `ggml_log_callback_default` (ggml.c:313 @c1d0e7a00).
export fn ggml_log_callback_default(
    level: c.enum_ggml_log_level,
    text: [*c]const u8,
    user_data: ?*anyopaque,
) callconv(.c) void {
    _ = level;
    _ = user_data;
    _ = fputs(@ptrCast(text), __stderrp);
    _ = fflush(__stderrp);
}

/// Ports `ggml_log_get` (ggml.c:8036 @c1d0e7a00).
export fn ggml_log_get(log_callback: *c.ggml_log_callback, user_data: *?*anyopaque) void {
    log_callback.* = logger_state.log_callback;
    user_data.* = logger_state.log_callback_user_data;
}

/// Ports `ggml_log_set` (ggml.c:8041 @c1d0e7a00).
///
/// A null callback restores the default rather than disabling logging, so
/// there is always something to call.
export fn ggml_log_set(log_callback: c.ggml_log_callback, user_data: ?*anyopaque) void {
    logger_state.log_callback = log_callback orelse ggml_log_callback_default;
    logger_state.log_callback_user_data = user_data;
}

// -----------------------------------------------------------------------------
// Aligned allocation

/// Ports `ggml_aligned_malloc` (ggml.c:331 @c1d0e7a00).
///
/// On macOS this goes to the Mach VM rather than `posix_memalign`, which is
/// what the C does under `TARGET_OS_OSX`. The alignment argument is unused on
/// that path because `vm_allocate` always returns page-aligned memory.
///
/// Return: the allocation, or null on failure after logging why.
export fn ggml_aligned_malloc(size: usize) ?*anyopaque {
    if (size == 0) {
        impl.logWarn("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n", .{});
        return null;
    }

    var address: usize = 0;
    const status = vm_allocate(mach_task_self_, &address, size, vm_flags_anywhere);
    const result: c_int = switch (status) {
        kern_success => 0,
        kern_invalid_address => einval,
        kern_no_space => enomem,
        else => efault,
    };

    if (result != 0) {
        const error_desc: [*:0]const u8 = switch (result) {
            einval => "invalid alignment value",
            enomem => "insufficient memory",
            else => "unknown allocation error",
        };
        impl.logError("%s: %s (attempted to allocate %6.2f MB)\n", .{
            "ggml_aligned_malloc",
            error_desc,
            @as(f64, @floatFromInt(size)) / (1024.0 * 1024.0),
        });
        return null;
    }

    return @ptrFromInt(address);
}

/// Ports `ggml_aligned_free` (ggml.c:387 @c1d0e7a00).
///
/// `size` is required, not decorative: `vm_deallocate` needs the length that
/// was allocated, so this cannot be a plain `free`.
export fn ggml_aligned_free(ptr: ?*anyopaque, size: usize) void {
    const p = ptr orelse return;
    _ = vm_deallocate(mach_task_self_, @intFromPtr(p), size);
}

// -----------------------------------------------------------------------------
// Status and version

/// Ports `ggml_status_to_string` (ggml.c:437 @c1d0e7a00).
export fn ggml_status_to_string(status: c.enum_ggml_status) [*:0]const u8 {
    return switch (status) {
        c.GGML_STATUS_ALLOC_FAILED => "GGML status: error (failed to allocate memory)",
        c.GGML_STATUS_FAILED => "GGML status: error (operation failed)",
        c.GGML_STATUS_SUCCESS => "GGML status: success",
        c.GGML_STATUS_ABORTED => "GGML status: warning (operation aborted)",
        else => "GGML status: unknown",
    };
}

/// Ports `ggml_version` (ggml.c:514 @c1d0e7a00).
export fn ggml_version() [*:0]const u8 {
    return build_version;
}

/// Ports `ggml_commit` (ggml.c:518 @c1d0e7a00).
export fn ggml_commit() [*:0]const u8 {
    return build_commit;
}

/// Mirrors the `GGML_VERSION` and `GGML_COMMIT` macros that `build/llamacpp.zig`
/// defines for the C sources. Kept in step with the `-D` flags there.
const build_version: [*:0]const u8 = "0.3.0";
const build_commit: [*:0]const u8 = "c1d0e7a00";

/// Ports `ggml_guid_matches` (ggml.c:510 @c1d0e7a00).
export fn ggml_guid_matches(guid_a: c.ggml_guid_t, guid_b: c.ggml_guid_t) bool {
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

/// Ports `ggml_fp16_to_fp32` (ggml.c:449 @c1d0e7a00).
export fn ggml_fp16_to_fp32(x: c.ggml_fp16_t) f32 {
    return impl.fp16ToFp32(x);
}

/// Ports `ggml_fp32_to_fp16` (ggml.c:454 @c1d0e7a00).
export fn ggml_fp32_to_fp16(x: f32) c.ggml_fp16_t {
    return impl.fp32ToFp16(x);
}

/// Ports `ggml_bf16_to_fp32` (ggml.c:459 @c1d0e7a00).
export fn ggml_bf16_to_fp32(x: c.ggml_bf16_t) f32 {
    return impl.bf16ToFp32(x.bits);
}

/// Ports `ggml_fp32_to_bf16` (ggml.c:464 @c1d0e7a00).
export fn ggml_fp32_to_bf16(x: f32) c.ggml_bf16_t {
    return .{ .bits = impl.fp32ToBf16(x) };
}

/// Ports `ggml_fp16_to_fp32_row` (ggml.c:468 @c1d0e7a00).
export fn ggml_fp16_to_fp32_row(x: [*]const c.ggml_fp16_t, y: [*]f32, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = impl.fp16ToFp32(x[i]);
}

/// Ports `ggml_fp32_to_fp16_row` (ggml.c:474 @c1d0e7a00).
export fn ggml_fp32_to_fp16_row(x: [*]const f32, y: [*]c.ggml_fp16_t, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = impl.fp32ToFp16(x[i]);
}

/// Ports `ggml_bf16_to_fp32_row` (ggml.c:481 @c1d0e7a00).
export fn ggml_bf16_to_fp32_row(x: [*]const c.ggml_bf16_t, y: [*]f32, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = impl.bf16ToFp32(x[i].bits);
}

/// Ports `ggml_fp32_to_bf16_row_ref` (ggml.c:488 @c1d0e7a00).
export fn ggml_fp32_to_bf16_row_ref(x: [*]const f32, y: [*]c.ggml_bf16_t, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = .{ .bits = impl.fp32ToBf16(x[i]) };
}

/// Ports `ggml_fp32_to_bf16_row` (ggml.c:494 @c1d0e7a00).
///
/// The C has an AVX512-BF16 fast path here; on ARM it falls through to the
/// same scalar loop as the reference version.
export fn ggml_fp32_to_bf16_row(x: [*]const f32, y: [*]c.ggml_bf16_t, n: i64) void {
    for (0..@intCast(n)) |i| y[i] = .{ .bits = impl.fp32ToBf16(x[i]) };
}

// -----------------------------------------------------------------------------
// Timing

/// Ports `ggml_time_init` (ggml.c:560 @c1d0e7a00).
///
/// A no-op outside Windows, where the C initialises a performance counter.
export fn ggml_time_init() void {}

/// Ports `ggml_time_ms` (ggml.c:561 @c1d0e7a00).
export fn ggml_time_ms() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

/// Ports `ggml_time_us` (ggml.c:567 @c1d0e7a00).
export fn ggml_time_us() i64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, ts.sec) * 1_000_000 + @divTrunc(@as(i64, ts.nsec), 1_000);
}

/// Ports `ggml_cycles` (ggml.c:574 @c1d0e7a00).
export fn ggml_cycles() i64 {
    return @intCast(clock());
}

/// Ports `ggml_cycles_per_ms` (ggml.c:578 @c1d0e7a00).
export fn ggml_cycles_per_ms() i64 {
    return @divTrunc(clocks_per_sec, 1000);
}

// -----------------------------------------------------------------------------
// Threadpool parameters

/// Ports `ggml_threadpool_params_init` (ggml.c:8046 @c1d0e7a00).
export fn ggml_threadpool_params_init(p: *c.struct_ggml_threadpool_params, n_threads: c_int) void {
    p.n_threads = n_threads;
    p.prio = 0; // normal, or inherited
    p.poll = 50; // hybrid polling
    p.strict_cpu = false; // all threads share one cpumask
    p.paused = false;
    // All-zero means "use the default affinity", which is usually inherited.
    @memset(&p.cpumask, false);
}

/// Ports `ggml_threadpool_params_default` (ggml.c:8055 @c1d0e7a00).
export fn ggml_threadpool_params_default(n_threads: c_int) c.struct_ggml_threadpool_params {
    var p: c.struct_ggml_threadpool_params = undefined;
    ggml_threadpool_params_init(&p, n_threads);
    return p;
}

/// Ports `ggml_threadpool_params_match` (ggml.c:8061 @c1d0e7a00).
export fn ggml_threadpool_params_match(
    p0: *const c.struct_ggml_threadpool_params,
    p1: *const c.struct_ggml_threadpool_params,
) bool {
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
    try std.testing.expectEqualStrings("GGML status: success", std.mem.span(ggml_status_to_string(c.GGML_STATUS_SUCCESS)));
    try std.testing.expectEqualStrings("GGML status: unknown", std.mem.span(ggml_status_to_string(@intCast(999))));
}

test "threadpool params round-trip and compare" {
    const a = ggml_threadpool_params_default(4);
    var b = ggml_threadpool_params_default(4);
    try std.testing.expect(ggml_threadpool_params_match(&a, &b));
    b.n_threads = 8;
    try std.testing.expect(!ggml_threadpool_params_match(&a, &b));
}

test "aligned malloc returns page-aligned memory and frees it" {
    const size = 4096 * 4;
    const p = ggml_aligned_malloc(size) orelse return error.AllocFailed;
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(p) % 4096);
    // Writing proves the pages are actually mapped, not just reserved.
    @memset(@as([*]u8, @ptrCast(p))[0..size], 0xAB);
    ggml_aligned_free(p, size);
}

// -----------------------------------------------------------------------------
// File access

/// Ports `ggml_fopen` (ggml.c:606 @c1d0e7a00).
///
/// A plain `fopen`. The C wraps a `_WIN32` branch that converts both arguments
/// from UTF-8 to wide characters and calls `_wfopen`; on every other platform
/// it is the `#else` below, and only that branch is ported.
pub export fn ggml_fopen(fname: [*:0]const u8, mode: [*:0]const u8) ?*std.c.FILE {
    return fopen(fname, mode);
}
