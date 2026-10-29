//! The byte streams a context's state is saved to and restored from.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-io.h` -- `llama_io_write_i` and `llama_io_read_i`
//! - `llama.cpp/src/llama-io.cpp` -- `write_string` and `read_string`
//!
//! All at v0.4.1 (`b29c606e2`).
//!
//! # Exceptions
//!
//! The C++ implementations throw `std`'s `runtime_error` when a buffer runs
//! out, and the `llama_state_*` entry points catch it and report failure.
//! Here every read and write returns `error.Thrown` with the message in
//! the `Exception` the stream carries, so the catching site can log the
//! same text.

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

/// Ports `llama_io_write_i` (llama-io.h:9 @b29c606e2): a sink for state.
pub const Writer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        write: *const fn (ptr: *anyopaque, src: []const u8, exc: *Exception) Thrown!void,
        writeTensor: *const fn (ptr: *anyopaque, tensor: *const c.ggml_tensor, offset: usize, size: usize, exc: *Exception) Thrown!void,
        nBytes: *const fn (ptr: *anyopaque) usize,
    };

    /// Return: nothing, or `error.Thrown` when the sink is full.
    pub fn write(self: Writer, src: []const u8, exc: *Exception) Thrown!void {
        return self.vtable.write(self.ptr, src, exc);
    }

    /// Writes `size` bytes of `tensor`'s data from `offset`.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn writeTensor(self: Writer, tensor: *const c.ggml_tensor, offset: usize, size: usize, exc: *Exception) Thrown!void {
        return self.vtable.writeTensor(self.ptr, tensor, offset, size, exc);
    }

    /// Return: the bytes written so far.
    pub fn nBytes(self: Writer) usize {
        return self.vtable.nBytes(self.ptr);
    }

    /// Writes a plain value's bytes.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn writeValue(self: Writer, value: anytype, exc: *Exception) Thrown!void {
        return self.write(std.mem.asBytes(&value), exc);
    }

    /// Ports `llama_io_write_i`'s `write_string` (llama-io.cpp:5 @b29c606e2):
    /// a `uint32_t` length, then the bytes.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn writeString(self: Writer, str: []const u8, exc: *Exception) Thrown!void {
        const str_size: u32 = @truncate(str.len);
        try self.writeValue(str_size, exc);
        try self.write(str[0..str_size], exc);
    }
};

/// Ports `llama_io_read_i` (llama-io.h:23 @b29c606e2): a source of state.
pub const Reader = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        read: *const fn (ptr: *anyopaque, dst: []u8, exc: *Exception) Thrown!void,
        readTensor: *const fn (ptr: *anyopaque, tensor: *c.ggml_tensor, offset: usize, size: usize, exc: *Exception) Thrown!void,
        nBytes: *const fn (ptr: *anyopaque) usize,
    };

    /// Return: nothing, or `error.Thrown` when the source runs out.
    pub fn read(self: Reader, dst: []u8, exc: *Exception) Thrown!void {
        return self.vtable.read(self.ptr, dst, exc);
    }

    /// Reads `size` bytes into `tensor`'s data at `offset`.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn readTensor(self: Reader, tensor: *c.ggml_tensor, offset: usize, size: usize, exc: *Exception) Thrown!void {
        return self.vtable.readTensor(self.ptr, tensor, offset, size, exc);
    }

    /// Return: the bytes read so far.
    pub fn nBytes(self: Reader) usize {
        return self.vtable.nBytes(self.ptr);
    }

    /// Reads a plain value.
    ///
    /// Return: the value, or `error.Thrown`.
    pub fn readValue(self: Reader, comptime T: type, exc: *Exception) Thrown!T {
        var v: T = undefined;
        try self.read(std.mem.asBytes(&v), exc);
        return v;
    }

    /// Ports `llama_io_read_i`'s `read_string` (llama-io.cpp:12 @b29c606e2).
    ///
    /// Parameters:
    /// - `out`: receives the string, replacing what it held.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn readString(self: Reader, out: *std.ArrayList(u8), exc: *Exception) Thrown!void {
        const str_size = try self.readValue(u32, exc);
        out.resize(allocator, str_size) catch impl.abort("llama-io.cpp", 15, "out of memory");
        try self.read(out.items, exc);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Writer);
    std.testing.refAllDecls(Reader);
}
