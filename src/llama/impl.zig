//! The internal libllama helpers the ported sampler and grammar call.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-impl.h`   — `time_meas`, the `LLAMA_LOG_*` macros
//! - `llama.cpp/src/llama-impl.cpp` — `time_meas`'s bodies
//! - `llama.cpp/ggml/include/ggml.h` — `GGML_ASSERT` and `GGML_ABORT`
//!
//! All at v0.3.0 (`c1d0e7a00`). Each declaration below names the source file
//! and line it came from.
//!
//! # The C ABI import
//!
//! `c` is the one `@cImport` of `llama.h` under `src/llama/`. Each `@cImport`
//! produces its own types, so a second one would give two incompatible
//! `*llama_vocab` -- the trap `cli/c.zig` records. It is separate from
//! `src/ggml/impl.zig`'s import because libllama is built as its own module
//! and library; nothing here passes a ggml pointer to Zig code in
//! `src/ggml/`, only to ggml's exported C functions.
//!
//! # Logging
//!
//! The `LLAMA_LOG_*` macros go to `log.zig`'s `logInternal`, which formats
//! with libc's `snprintf` and calls the callback `llama_log_set` installed.
//! No C variadic is defined anywhere in libllama -- Zig *calls* one on every
//! target, but cannot define one on every target -- so there is one path
//! for all of them. An in-process harness reaches the port's logger through
//! `api.impl.llama_log_set`, beside the C++'s.

const std = @import("std");

/// `llama.h`, and through it `ggml.h`, `ggml-backend.h`, `ggml-cpu.h` and
/// `ggml-opt.h`; and `gguf.h`, which the vocabulary and the model loader
/// read metadata through. Translated from `c_import.h` by
/// `build/llamacpp.zig` (Zig 0.17 has no `@cImport`).
pub const c = @import("llama_c");

/// `llama_token_data`, the element every sampler works on.
pub const TokenData = c.llama_token_data;
/// `llama_token_data_array`, the candidate list every sampler mutates.
pub const TokenDataArray = c.llama_token_data_array;
/// `llama_token`.
pub const Token = c.llama_token;
/// `struct llama_sampler`.
pub const Sampler = c.struct_llama_sampler;
/// `struct llama_sampler_i`, the C vtable.
pub const SamplerI = c.struct_llama_sampler_i;

/// The allocator for every object the ported code hands across the C ABI.
///
/// `malloc`, like the C++'s `operator new`: a sampler's context outlives any
/// Zig scope, is freed through a C entry point, and must not depend on an
/// allocator instance that only Zig code could reach.
pub const allocator = std.heap.c_allocator;

/// C's `stdout`: `__stdoutp` on Darwin, `stdout` on glibc, `__acrt_iob_func(1)`
/// in MinGW's UCRT. The macro expands to a different variable or call on each,
/// so it is read by name.
///
/// Return: the stream.
pub fn cStdout() *c.FILE {
    if (comptime is_windows) return @ptrCast(@alignCast(__acrt_iob_func(1)));
    return @extern(*const *c.FILE, .{ .name = if (@import("builtin").os.tag.isDarwin()) "__stdoutp" else "stdout" }).*;
}

const is_windows = @import("builtin").os.tag == .windows;

/// What the `stdout` and `stderr` macros expand to (MinGW's `stdio.h`, line 124).
extern "c" fn __acrt_iob_func(index: c_uint) *anyopaque;

/// `std::mutex`, as the libc++ this build links lays it out and drives it:
/// a `pthread_mutex_t` behind `PTHREAD_MUTEX_INITIALIZER` on POSIX, an
/// `SRWLOCK` behind `AcquireSRWLockExclusive` on Windows. Not a port of
/// anything; `src/ggml/impl.zig` has the same type for ggml, which this module
/// cannot import.
pub const CxxMutex = extern struct {
    raw: Raw = .{},

    const win32 = @import("win32.zig");
    const Raw = if (is_windows) win32.SRWLOCK else std.c.pthread_mutex_t;

    /// `std::mutex::lock`.
    pub fn lock(self: *CxxMutex) void {
        if (comptime is_windows) return win32.AcquireSRWLockExclusive(&self.raw);
        _ = std.c.pthread_mutex_lock(&self.raw);
    }

    /// `std::mutex::unlock`.
    pub fn unlock(self: *CxxMutex) void {
        if (comptime is_windows) return win32.ReleaseSRWLockExclusive(&self.raw);
        _ = std.c.pthread_mutex_unlock(&self.raw);
    }
};

/// C's `stderr`; see `cStdout`.
///
/// Return: the stream.
pub fn cStderr() *c.FILE {
    if (comptime is_windows) return @ptrCast(@alignCast(__acrt_iob_func(2)));
    return @extern(*const *c.FILE, .{ .name = if (@import("builtin").os.tag.isDarwin()) "__stderrp" else "stderr" }).*;
}

// -----------------------------------------------------------------------------
// Logging

const log = @import("log.zig");

/// Ports `LLAMA_LOG_WARN` (llama-impl.h:29 @c1d0e7a00).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logWarn(comptime format: [:0]const u8, args: anytype) void {
    log.logInternal(c.GGML_LOG_LEVEL_WARN, format, args);
}

/// Ports `LLAMA_LOG_ERROR` (llama-impl.h:30 @c1d0e7a00).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logError(comptime format: [:0]const u8, args: anytype) void {
    log.logInternal(c.GGML_LOG_LEVEL_ERROR, format, args);
}

/// Ports `LLAMA_LOG_INFO` (llama-impl.h:28 @c1d0e7a00).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logInfo(comptime format: [:0]const u8, args: anytype) void {
    log.logInternal(c.GGML_LOG_LEVEL_INFO, format, args);
}

/// Ports `LLAMA_LOG_DEBUG` (llama-impl.h:31 @c1d0e7a00).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logDebug(comptime format: [:0]const u8, args: anytype) void {
    log.logInternal(c.GGML_LOG_LEVEL_DEBUG, format, args);
}

/// Ports `LLAMA_LOG_CONT` (llama-impl.h:32 @c1d0e7a00): continues the
/// previous line.
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logCont(comptime format: [:0]const u8, args: anytype) void {
    log.logInternal(c.GGML_LOG_LEVEL_CONT, format, args);
}

// -----------------------------------------------------------------------------
// Assertions

/// Mirrors `GGML_ABORT` (ggml.h:287 @c1d0e7a00), through ggml's own
/// `ggml_abort`, so the message and backtrace are the C++'s.
///
/// Parameters:
/// - `file`: the C++ source the abort stands in for.
/// - `line`: its line there.
/// - `msg`: the message, used as a literal format string.
///
/// Return: never.
pub fn abort(comptime file: [:0]const u8, comptime line: c_int, comptime msg: [:0]const u8) noreturn {
    c.ggml_abort(file.ptr, line, "%s", msg.ptr);
}

/// Mirrors `GGML_ASSERT` (ggml.h:288 @c1d0e7a00).
///
/// Parameters:
/// - `ok`: the condition.
/// - `file`: the C++ source the assertion stands in for.
/// - `line`: its line there.
/// - `expr`: the asserted expression's text, for the message.
///
/// Return: nothing when `ok`; never otherwise.
pub inline fn assert(ok: bool, comptime file: [:0]const u8, comptime line: c_int, comptime expr: [:0]const u8) void {
    if (!ok) c.ggml_abort(file.ptr, line, "GGML_ASSERT(%s) failed", expr.ptr);
}

/// Stands in for a C++ exception that escapes through the C ABI.
///
/// Upstream throws `std::runtime_error` from several places a C caller
/// reaches -- an empty grammar stack after accepting a token, a regex that
/// fails to compile -- and nothing catches it, so the process terminates.
/// This terminates too, with the exception's message, through `ggml_abort`.
///
/// Parameters:
/// - `file`: the C++ source the throw stands in for.
/// - `line`: its line there.
/// - `msg`: the exception's `what()`.
///
/// Return: never.
pub fn throwRuntime(comptime file: [:0]const u8, comptime line: c_int, msg: []const u8) noreturn {
    var buf: [1024]u8 = undefined;
    const n = @min(msg.len, buf.len - 1);
    @memcpy(buf[0..n], msg[0..n]);
    buf[n] = 0;
    c.ggml_abort(file.ptr, line, "terminating due to uncaught exception of type std::runtime_error: %s", &buf);
}

/// The one error a function standing in for a throwing C++ function returns.
/// What was thrown is in the `Exception` the caller passed.
pub const Thrown = error{Thrown};

/// A C++ exception in flight: its type and `what()`.
///
/// Where the C++ throws, the port fills one of these and returns
/// `error.Thrown`; whatever stands where the C++ catches decides what
/// happens next. That differs by path, which is why the exception has to
/// travel: a `std::runtime_error` from a tokenizer terminates the process
/// when it escapes `llama_tokenize`, and the same exception raised while the
/// vocabulary loads is caught by `llama_model_load` and reported.
pub const Exception = struct {
    kind: Kind = .runtime_error,
    buf: [1024]u8 = undefined,
    len: usize = 0,

    /// The exception types upstream throws on these paths.
    pub const Kind = enum {
        runtime_error,
        out_of_range,
        invalid_argument,
        length_error,

        fn name(self: Kind) [:0]const u8 {
            return switch (self) {
                .runtime_error => "std::runtime_error",
                .out_of_range => "std::out_of_range",
                .invalid_argument => "std::invalid_argument",
                .length_error => "std::length_error",
            };
        }
    };

    /// Records an exception.
    ///
    /// Parameters:
    /// - `kind`: its type.
    /// - `fmt`, `args`: its `what()`, formatted; truncated at 1023 bytes.
    ///
    /// Return: `error.Thrown`, always, for the caller to return.
    pub fn throw(self: *Exception, kind: Kind, comptime fmt: []const u8, args: anytype) Thrown {
        self.set(kind, fmt, args);
        return error.Thrown;
    }

    /// Records an exception without raising it, for a caller about to
    /// `terminate` with it.
    ///
    /// Return: nothing.
    pub fn set(self: *Exception, kind: Kind, comptime fmt: []const u8, args: anytype) void {
        self.kind = kind;
        const s = std.mem.print(self.buf[0 .. self.buf.len - 1], fmt, args) catch self.buf[0 .. self.buf.len - 1];
        self.len = s.len;
    }

    /// Return: the message, borrowed from the exception.
    pub fn what(self: *const Exception) []const u8 {
        return self.buf[0..self.len];
    }

    /// Return: the message, NUL-terminated for a `%s`, borrowed from the
    /// exception.
    pub fn whatZ(self: *Exception) [*:0]const u8 {
        self.buf[self.len] = 0;
        return @ptrCast(&self.buf);
    }

    /// The exception escaping through the C ABI: the process terminates
    /// with libc++abi's message, through `ggml_abort`.
    ///
    /// Return: never.
    pub fn terminate(self: *Exception, comptime file: [:0]const u8, comptime line: c_int) noreturn {
        self.buf[self.len] = 0;
        c.ggml_abort(file.ptr, line, "terminating due to uncaught exception of type %s: %s", self.kind.name().ptr, &self.buf);
    }
};

// -----------------------------------------------------------------------------
// Timing

/// Ports `time_meas` (llama-impl.h:77 @c1d0e7a00).
///
/// The C++ adds the elapsed time to its accumulator in the destructor; here
/// `deinit` does, called through `defer`.
pub const TimeMeas = struct {
    t_start_us: i64,
    t_acc: *i64,

    /// Ports the constructor of `time_meas` (llama-impl.cpp:20 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `t_acc`: the accumulator; borrowed until `deinit`.
    /// - `disable`: when set, nothing is measured.
    ///
    /// Return: the running measurement.
    pub fn init(t_acc: *i64, disable: bool) TimeMeas {
        return .{ .t_start_us = if (disable) -1 else c.ggml_time_us(), .t_acc = t_acc };
    }

    /// Ports the destructor of `time_meas` (llama-impl.cpp:22 @c1d0e7a00).
    ///
    /// Return: nothing.
    pub fn deinit(self: TimeMeas) void {
        if (self.t_start_us >= 0) {
            self.t_acc.* += c.ggml_time_us() - self.t_start_us;
        }
    }
};

// -----------------------------------------------------------------------------
// libm
//
// Every transcendental goes through the platform libm, as the C++'s calls
// do. Zig's `@exp`, `@log` and friends are Zig's own implementations, free to
// differ from Apple's in the last bit -- `src/ggml/cpu/unary_ops.zig` records
// the same choice.

pub extern fn expf(x: f32) f32;
pub extern fn logf(x: f32) f32;
pub extern fn log2f(x: f32) f32;
pub extern fn powf(x: f32, y: f32) f32;
pub extern fn sqrtf(x: f32) f32;
pub extern fn exp(x: f64) f64;
pub extern fn pow(x: f64, y: f64) f64;
pub extern fn atoi(s: [*:0]const u8) c_int;

/// libc's `arc4random`, which is what libc++'s `std::random_device` draws
/// from on Apple platforms.
pub extern fn arc4random() u32;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
