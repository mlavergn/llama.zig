//! The internal libllama helpers the ported sampler and grammar call.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-impl.h`   — `time_meas`, the `LLAMA_LOG_*` macros
//! - `llama.cpp/src/llama-impl.cpp` — `time_meas`'s bodies,
//!   `llama_prefetch_rows` and `llama_clear_tensor_data`
//! - `llama.cpp/ggml/include/ggml.h` — `GGML_ASSERT` and `GGML_ABORT`
//!
//! All at v1.0.0 (`b1b770d90`). Each declaration below names the source file
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
const mod = @import("module.zig");
const stdlib = @import("stdlib");

/// `llama.h`, and through it `ggml.h`, `ggml-backend.h`, `ggml-cpu.h` and
/// `ggml-opt.h`; and `gguf.h`, which the vocabulary and the model loader
/// read metadata through. Translated from `c_import.h` by
/// `build/llamacpp.zig` (Zig 0.17 has no `@cImport`).
pub const llama_c = @import("llama_c");

/// C's `stdout`: `__stdoutp` on Darwin, `stdout` on glibc, `__acrt_iob_func(1)`
/// in MinGW's UCRT. The macro expands to a different variable or call on each,
/// so it is read by name.
///
/// Return: the stream.
pub fn cStdout() *llama_c.FILE {
    return @ptrCast(@alignCast(stdlib.stdio.cStream(.stdout)));
}

/// C's `stderr`; see `cStdout`.
///
/// Return: the stream.
pub fn cStderr() *llama_c.FILE {
    return @ptrCast(@alignCast(stdlib.stdio.cStream(.stderr)));
}

/// Writes `parts` to C's `stderr` in order -- the bytes and the stream of
/// the C's `fprintf(stderr, ...)` at the two sites that print directly, the
/// grammar parser's error and `unicode_regex_split`'s.
///
/// Library code never uses `std.debug.print`: it reaches `std.Io.Threaded`,
/// which Zig 0.17 cannot compile for iOS (its `NullFile` has no `fd`; found
/// when the xcframework's iOS slice failed to build), and whose dependency
/// on compiler_rt brings in compiler_rt's `expf` and `logf`.
/// A failed write is dropped, as the C ignores `fprintf`'s result.
///
/// Parameters:
/// - `parts`: the text, in pieces.
///
/// Return: nothing.
pub fn writeStderr(parts: []const []const u8) void {
    // unbuffered, as C's `stderr` is, so it lands where `fprintf` would
    for (parts) |part| stdlib.stdio.stderr().write(part);
}

// -----------------------------------------------------------------------------
// Logging

/// Ports `llama_prefetch_rows` (llama-impl.cpp:22 @b1b770d90): prefetch the
/// host pages a gather of these rows of `tensor` will read. A no-op for a
/// tensor that is not in host memory, or for no rows.
///
/// Parameters:
/// - `tensor`: a matrix.
/// - `rows`: the row indices, each in `[0, ne[1])`.
/// - `n_rows`: how many.
///
/// Return: nothing.
pub fn prefetchRows(tensor: ?*const llama_c.ggml_tensor, rows: [*c]const i32, n_rows: usize) void {
    const t = tensor orelse return;
    if (t.data == null or t.buffer == null or !llama_c.ggml_backend_buffer_is_host(t.buffer) or n_rows == 0) {
        return;
    }

    assert(llama_c.ggml_is_matrix(t), "llama-impl.cpp", 27, "ggml_is_matrix(tensor)");

    const row_bytes = llama_c.ggml_row_size(t.type, t.ne[0]);
    const base: [*]const u8 = @ptrCast(t.data.?);

    const mr = std.heap.c_allocator.alloc(mod.mmap.MemoryRange, n_rows) catch abort("llama-impl.cpp", 32, "out of memory");
    defer std.heap.c_allocator.free(mr);
    for (0..n_rows) |i| {
        assert(rows[i] >= 0 and rows[i] < t.ne[1], "llama-impl.cpp", 35, "rows[i] >= 0 && rows[i] < tensor->ne[1]");
        mr[i] = .{ .addr = base + @as(usize, @intCast(rows[i])) * t.nb[1], .size = row_bytes };
    }

    mod.mmap.prefetchRanges(mr);
}

/// Ports `llama_clear_tensor_data` (llama-impl.cpp:91 @b1b770d90): zero
/// `size` bytes of `t` from `offset` by writing zeros, 1 MiB at a time,
/// since not every backend buffer implements `ggml_backend_tensor_memset`.
///
/// Return: nothing.
pub fn clearTensorData(t: *llama_c.ggml_tensor, offset: usize, size: usize) void {
    const zeros = struct {
        var buf: [1024 * 1024]u8 = @splat(0);
    };
    var ofs: usize = 0;
    while (ofs < size) : (ofs += zeros.buf.len) {
        llama_c.ggml_backend_tensor_set(t, &zeros.buf, offset + ofs, @min(size - ofs, zeros.buf.len));
    }
}

/// Ports `LLAMA_LOG_WARN` (llama-impl.h:29 @b1b770d90).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logWarn(comptime format: [:0]const u8, args: anytype) void {
    mod.log.logInternal(llama_c.GGML_LOG_LEVEL_WARN, format, args);
}

/// Ports `LLAMA_LOG_ERROR` (llama-impl.h:30 @b1b770d90).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logError(comptime format: [:0]const u8, args: anytype) void {
    mod.log.logInternal(llama_c.GGML_LOG_LEVEL_ERROR, format, args);
}

/// Ports `LLAMA_LOG_INFO` (llama-impl.h:28 @b1b770d90).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logInfo(comptime format: [:0]const u8, args: anytype) void {
    mod.log.logInternal(llama_c.GGML_LOG_LEVEL_INFO, format, args);
}

/// Ports `LLAMA_LOG_DEBUG` (llama-impl.h:31 @b1b770d90).
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logDebug(comptime format: [:0]const u8, args: anytype) void {
    mod.log.logInternal(llama_c.GGML_LOG_LEVEL_DEBUG, format, args);
}

/// Ports `LLAMA_LOG_CONT` (llama-impl.h:32 @b1b770d90): continues the
/// previous line.
///
/// Parameters:
/// - `format`: a printf format string.
/// - `args`: C-compatible arguments for it.
///
/// Return: nothing.
pub fn logCont(comptime format: [:0]const u8, args: anytype) void {
    mod.log.logInternal(llama_c.GGML_LOG_LEVEL_CONT, format, args);
}

// -----------------------------------------------------------------------------
// Assertions

/// Ports `replace_all` (llama-impl.cpp:101 @b1b770d90).
///
/// Parameters:
/// - `s`: the string, rewritten in place.
/// - `search`: what to replace; nothing happens when it is empty.
/// - `replace`: what replaces it.
///
/// Return: nothing.
pub fn replaceAll(s: *std.ArrayList(u8), search: []const u8, replace: []const u8) void {
    if (search.len == 0) return;
    var builder: std.ArrayList(u8) = .empty;
    builder.ensureTotalCapacity(std.heap.c_allocator, s.items.len) catch abort("llama-impl.cpp", 101, "out of memory");
    var last_pos: usize = 0;
    while (std.mem.findPos(u8, s.items, last_pos, search)) |pos| {
        builder.appendSlice(std.heap.c_allocator, s.items[last_pos..pos]) catch abort("llama-impl.cpp", 101, "out of memory");
        builder.appendSlice(std.heap.c_allocator, replace) catch abort("llama-impl.cpp", 101, "out of memory");
        last_pos = pos + search.len;
    }
    builder.appendSlice(std.heap.c_allocator, s.items[last_pos..]) catch abort("llama-impl.cpp", 101, "out of memory");
    s.deinit(std.heap.c_allocator);
    s.* = builder;
}

/// Mirrors `GGML_ABORT` (ggml.h:114 @b1b770d90), through ggml's own
/// `ggml_abort`, so the message and backtrace are the C++'s.
///
/// Parameters:
/// - `file`: the C++ source the abort stands in for.
/// - `line`: its line there.
/// - `msg`: the message, used as a literal format string.
///
/// Return: never.
pub fn abort(comptime file: [:0]const u8, comptime line: c_int, comptime msg: [:0]const u8) noreturn {
    llama_c.ggml_abort(file.ptr, line, "%s", msg.ptr);
}

/// Mirrors `GGML_ASSERT` (ggml.h:115 @b1b770d90).
///
/// Parameters:
/// - `ok`: the condition.
/// - `file`: the C++ source the assertion stands in for.
/// - `line`: its line there.
/// - `expr`: the asserted expression's text, for the message.
///
/// Return: nothing when `ok`; never otherwise.
pub inline fn assert(ok: bool, comptime file: [:0]const u8, comptime line: c_int, comptime expr: [:0]const u8) void {
    if (!ok) llama_c.ggml_abort(file.ptr, line, "GGML_ASSERT(%s) failed", expr.ptr);
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
    llama_c.ggml_abort(file.ptr, line, "terminating due to uncaught exception of type std::runtime_error: %s", &buf);
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
        llama_c.ggml_abort(file.ptr, line, "terminating due to uncaught exception of type %s: %s", self.kind.name().ptr, &self.buf);
    }
};

// -----------------------------------------------------------------------------
// Timing

/// Ports `time_meas` (llama-impl.h:80 @b1b770d90).
///
/// The C++ adds the elapsed time to its accumulator in the destructor; here
/// `deinit` does, called through `defer`.
pub const TimeMeas = struct {
    t_start_us: i64,
    t_acc: *i64,

    /// Ports the constructor of `time_meas` (llama-impl.cpp:42 @b1b770d90).
    ///
    /// Parameters:
    /// - `t_acc`: the accumulator; borrowed until `deinit`.
    /// - `disable`: when set, nothing is measured.
    ///
    /// Return: the running measurement.
    pub fn init(t_acc: *i64, disable: bool) TimeMeas {
        return .{ .t_start_us = if (disable) -1 else llama_c.ggml_time_us(), .t_acc = t_acc };
    }

    /// Ports the destructor of `time_meas` (llama-impl.cpp:44 @b1b770d90).
    ///
    /// Return: nothing.
    pub fn deinit(self: TimeMeas) void {
        if (self.t_start_us >= 0) {
            self.t_acc.* += llama_c.ggml_time_us() - self.t_start_us;
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

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
