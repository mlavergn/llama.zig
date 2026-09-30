//! The GGUF container format: reading a model file's key/value metadata and
//! tensor table, and writing one back out.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/gguf.cpp` (v0.3.0, `c1d0e7a00`), 1,706
//! lines of C++. Each declaration below names the C++ it replaces and the line
//! it began at. This file exports the same C symbols with the same signatures,
//! so the C++ that calls into it links unchanged.
//!
//! **The first STL-heavy translation unit ported.** `ggml-threading.cpp` and
//! `ggml-backend-reg.cpp` came first, but neither is representative: one is a
//! mutex and the other a registry of eight entries. This file is 61 exported
//! functions over `std::vector`, `std::string`, two `std::map` lookup tables,
//! a template-dispatched reader and a virtual writer hierarchy — the shapes
//! the rest of Stage 4 is made of.
//!
//! # The contract is 61 symbols, not 44
//!
//! `PLAN.md` records 44. Measured against `zig c++ -c` at the pinned commit,
//! the unmangled export count is **61**; the 2,002 mangled names are template
//! instantiations and inline functions emitted weakly, and none of them is
//! referenced from another translation unit in this build.
//!
//! Two of those mangled names are real functions rather than template noise:
//! `gguf_type_size` and `gguf_write_to_buf`, declared in `ggml-impl.h:781-782`
//! under the comment "expose GGUF internals for test code" and *inside*
//! `#ifdef __cplusplus`, which gives them C++ linkage. Their only caller
//! anywhere is `tests/test-gguf.cpp`, which this project does not build. They
//! are therefore outside the contract and are not exported here — `gguf_type_size`
//! survives as the file-private `typeSize` below, and `gguf_write_to_buf` as
//! `writeToBuf`.
//!
//! # Exceptions
//!
//! Every `throw` in the C++ is caught inside the C++. On a short `fputc` or
//! `fwrite` the file writer raises `std::runtime_error` from either of its two
//! overrides, `write` and `write` (gguf.cpp:1585, 1594 @c1d0e7a00), and
//! `gguf_write_to_file_ptr` (gguf.cpp:1665 @c1d0e7a00) catches it, turning it
//! into `false` plus a log line. Three more sites catch `std::length_error` and
//! `std::bad_alloc` around vector growth while reading a malformed file.
//! None of that crosses the translation-unit boundary, so the port needs no
//! exception machinery: the file writer returns a Zig error union and
//! `gguf_write_to_file_ptr` is the one place that turns it back into a `bool`.
//!
//! Allocation failure is the one case that is *not* reproduced, and cannot be.
//! The C++ lets `std::bad_alloc` propagate out of `gguf_get_meta_size` and
//! `gguf_write_to_buf`, where nothing catches it and `std::terminate` runs.
//! This port aborts through `ggml_abort` instead, which is how every other
//! allocation failure in ported ggml already behaves (`impl.ggmlMalloc`).
//!
//! # Deliberate differences
//!
//! - **`gguf_tensor_info.t` is zero-initialised.** The C++ declares its
//!   `gguf_tensor_info` without an initialiser (gguf.cpp:632 @c1d0e7a00) and
//!   fills in only `name`, `ne`, `nb`, `type` and `offset`, leaving `op`,
//!   `src`, `buffer`, `data`, `view_src`, `extra`, `flags` and `op_params`
//!   indeterminate before `push_back` copies the whole struct. Nothing on the
//!   read path reads them, but both writers branch on `info.t.buffer` inside
//!   `write_tensor_data` (gguf.cpp:1557 @c1d0e7a00) and again in the file
//!   writer's `write_tensor_data` (gguf.cpp:1602 @c1d0e7a00), so a context read
//!   from a file and then written out would branch on a garbage pointer.
//!   Zeroing costs nothing and makes the output depend only on the input, which
//!   is the same call this project made for `quantize_row_iq4_nl_ref`.
//! - **The writer's virtual dispatch becomes a comptime generic.** The C++
//!   comments its own vtable with "we bet on devirtualization", just above the
//!   pure-virtual `write` (gguf.cpp:1439 @c1d0e7a00), and `gguf_write_out`
//!   (gguf.cpp:1622 @c1d0e7a00) is already a template over the writer type.
//!   `Writer(Impl)` below is that template with the bet settled at compile
//!   time.
//! - **Byte buffers are `u8`, not `int8_t`.** `gguf_kv::data` is
//!   `std::vector<int8_t>` and every use of it is `memcpy`, `push_back` or a
//!   reinterpreting read; no arithmetic depends on the sign. The one place the
//!   element type is visible across the ABI is `gguf_get_arr_data`, which
//!   returns `const void *`.

const std = @import("std");
const impl = @import("impl.zig");
const c = impl.c;

/// The registry's vectors never cross the C ABI, but there is no allocator to
/// be handed one at a C entry point. libc's is the same heap the rest of ggml
/// uses, and the same choice `backend_reg.zig` makes.
const allocator = std.heap.c_allocator;

/// Ports `GGUF_MAX_STRING_LENGTH` (gguf.cpp:18 @c1d0e7a00).
const max_string_length: u64 = 1024 * 1024 * 1024;

/// Ports `GGUF_MAX_ARRAY_ELEMENTS` (gguf.cpp:19 @c1d0e7a00).
const max_array_elements: u64 = 1024 * 1024 * 1024;

/// Aborts on a failed allocation, standing in for the `std::bad_alloc` the C++
/// would let escape to `std::terminate`. See the note in the file header.
fn oom() noreturn {
    impl.abort("gguf: out of memory");
}

// -----------------------------------------------------------------------------
// Type tables
//
// The C++ holds these as two `std::map<gguf_type, ...>` with a
// `static_assert(GGUF_TYPE_COUNT == 13)` under each. A switch is the closer
// translation than a hash map would be: both tables are dense, constant, and
// looked up by an enum that is already an integer.

comptime {
    // Ports the `static_assert` that follows each of the two tables. Both
    // name `GGUF_TYPE_COUNT`, `GGUF_TYPE_COUNT` (gguf.cpp:107, 124 @c1d0e7a00).
    std.debug.assert(c.GGUF_TYPE_COUNT == 13);
}

/// Ports `type_to_gguf_type` (gguf.cpp:30 @c1d0e7a00), the twelve explicit
/// specialisations collapsed into one comptime switch.
///
/// `Str` stands for the `std::string` arm of `type_to_gguf_type`
/// (gguf.cpp:73 @c1d0e7a00).
fn typeToGgufType(comptime T: type) c_uint {
    return switch (T) {
        u8 => c.GGUF_TYPE_UINT8,
        i8 => c.GGUF_TYPE_INT8,
        u16 => c.GGUF_TYPE_UINT16,
        i16 => c.GGUF_TYPE_INT16,
        u32 => c.GGUF_TYPE_UINT32,
        i32 => c.GGUF_TYPE_INT32,
        f32 => c.GGUF_TYPE_FLOAT32,
        bool => c.GGUF_TYPE_BOOL,
        Str => c.GGUF_TYPE_STRING,
        u64 => c.GGUF_TYPE_UINT64,
        i64 => c.GGUF_TYPE_INT64,
        f64 => c.GGUF_TYPE_FLOAT64,
        else => @compileError("no gguf_type for " ++ @typeName(T)),
    };
}

/// An owned, NUL-terminated string: this port's `std::string`.
///
/// The sentinel is what makes `gguf_get_key`, `gguf_get_val_str` and
/// `gguf_get_arr_str` able to return a `const char *` the way `c_str()` does,
/// while `.len` stays the `length()` the writer serialises. A GGUF string may
/// contain embedded NULs and `std::string` keeps those too; both agree that
/// `length()`, not the first NUL, is what gets written back out.
const Str = [:0]u8;

/// Ports `GGUF_TYPE_SIZE` (gguf.cpp:92 @c1d0e7a00) together with the
/// `gguf_type_size` lookup that reads it (gguf.cpp:126 @c1d0e7a00).
///
/// Not exported: the C++ declares `gguf_type_size` in `ggml-impl.h` inside
/// `#ifdef __cplusplus`, so its symbol is mangled and belongs to no C ABI. See
/// the file header.
///
/// Parameters:
/// - `t`: the GGUF type tag.
///
/// Return: the size of one element in bytes; 0 for `STRING` and `ARRAY`, whose
/// entries the C++ map marks "undefined", and 0 for any value outside the enum,
/// which is the `it == end` arm.
fn typeSize(t: c_uint) usize {
    return switch (t) {
        c.GGUF_TYPE_UINT8, c.GGUF_TYPE_INT8 => @sizeOf(u8),
        c.GGUF_TYPE_UINT16, c.GGUF_TYPE_INT16 => @sizeOf(u16),
        c.GGUF_TYPE_UINT32, c.GGUF_TYPE_INT32, c.GGUF_TYPE_FLOAT32 => @sizeOf(u32),
        c.GGUF_TYPE_BOOL => @sizeOf(i8),
        c.GGUF_TYPE_STRING => 0, // undefined
        c.GGUF_TYPE_ARRAY => 0, // undefined
        c.GGUF_TYPE_UINT64, c.GGUF_TYPE_INT64, c.GGUF_TYPE_FLOAT64 => @sizeOf(u64),
        else => 0,
    };
}

/// Ports `GGUF_TYPE_NAME` (gguf.cpp:109 @c1d0e7a00).
///
/// Return: the short type name, or null for a value outside the enum — the
/// `it == end` arm of `gguf_type_name`.
fn typeNameOrNull(t: c_uint) ?[*:0]const u8 {
    return switch (t) {
        c.GGUF_TYPE_UINT8 => "u8",
        c.GGUF_TYPE_INT8 => "i8",
        c.GGUF_TYPE_UINT16 => "u16",
        c.GGUF_TYPE_INT16 => "i16",
        c.GGUF_TYPE_UINT32 => "u32",
        c.GGUF_TYPE_INT32 => "i32",
        c.GGUF_TYPE_FLOAT32 => "f32",
        c.GGUF_TYPE_BOOL => "bool",
        c.GGUF_TYPE_STRING => "str",
        c.GGUF_TYPE_ARRAY => "arr",
        c.GGUF_TYPE_UINT64 => "u64",
        c.GGUF_TYPE_INT64 => "i64",
        c.GGUF_TYPE_FLOAT64 => "f64",
        else => null,
    };
}

/// Duplicates `bytes` into an owned NUL-terminated string. The C++ gets this
/// from `std::string`'s copy constructor.
fn dupeStr(bytes: []const u8) Str {
    return allocator.dupeZ(u8, bytes) catch oom();
}

/// Allocates a zero-filled `Str` of `n` bytes, standing in for
/// `std::string::resize`, which value-initialises to `'\0'`.
fn allocStr(n: usize) Str {
    const s = allocator.allocSentinel(u8, n, 0) catch oom();
    @memset(s, 0);
    return s;
}

// -----------------------------------------------------------------------------
// Key/value pairs

/// Ports `struct gguf_kv` (gguf.cpp:131 @c1d0e7a00).
///
/// Both payload fields exist on every value, as in the C++: `data` carries the
/// raw bytes of a numeric or boolean value, `data_string` the elements of a
/// string value. Exactly one is ever non-empty, decided by `type`.
const Kv = struct {
    key: Str,
    is_array: bool,
    type: c_uint,
    data: std.ArrayList(u8) = .empty,
    data_string: std.ArrayList(Str) = .empty,

    /// Ports the scalar `gguf_kv` constructor (gguf.cpp:141 @c1d0e7a00).
    fn initScalar(key: []const u8, comptime T: type, value: T) Kv {
        impl.assert(key.len != 0, "!key.empty()");
        var kv: Kv = .{ .key = dupeStr(key), .is_array = false, .type = typeToGgufType(T) };
        kv.data.resize(allocator, @sizeOf(T)) catch oom();
        @memcpy(kv.data.items, std.mem.asBytes(&value));
        return kv;
    }

    /// Ports the `std::vector<T>` `gguf_kv` constructor (gguf.cpp:149 @c1d0e7a00).
    fn initArray(key: []const u8, comptime T: type, values: []const T) Kv {
        impl.assert(key.len != 0, "!key.empty()");
        var kv: Kv = .{ .key = dupeStr(key), .is_array = true, .type = typeToGgufType(T) };
        kv.data.resize(allocator, values.len * @sizeOf(T)) catch oom();
        for (values, 0..) |v, i| {
            const tmp = v;
            @memcpy(kv.data.items[i * @sizeOf(T) ..][0..@sizeOf(T)], std.mem.asBytes(&tmp));
        }
        return kv;
    }

    /// Ports the `std::string` `gguf_kv` constructor (gguf.cpp:159 @c1d0e7a00).
    fn initStr(key: []const u8, value: []const u8) Kv {
        impl.assert(key.len != 0, "!key.empty()");
        var kv: Kv = .{ .key = dupeStr(key), .is_array = false, .type = c.GGUF_TYPE_STRING };
        kv.data_string.append(allocator, dupeStr(value)) catch oom();
        return kv;
    }

    /// Ports the string-array `gguf_kv` constructor (gguf.cpp:165 @c1d0e7a00).
    ///
    /// Takes ownership of `values` and of every string in it; the C++ copies,
    /// because `std::vector` assignment does.
    fn initStrArrayOwned(key: []const u8, values: std.ArrayList(Str)) Kv {
        impl.assert(key.len != 0, "!key.empty()");
        return .{
            .key = dupeStr(key),
            .is_array = true,
            .type = c.GGUF_TYPE_STRING,
            .data_string = values,
        };
    }

    /// Stands in for `~gguf_kv`, which the C++ gets from its members.
    fn deinit(self: *Kv) void {
        allocator.free(self.key);
        self.data.deinit(allocator);
        for (self.data_string.items) |s| allocator.free(s);
        self.data_string.deinit(allocator);
    }

    /// Ports `gguf_kv`'s `get_key` (gguf.cpp:171 @c1d0e7a00).
    fn getKey(self: *const Kv) Str {
        return self.key;
    }

    /// Ports `gguf_kv`'s `get_type` (gguf.cpp:175 @c1d0e7a00).
    fn getType(self: *const Kv) c_uint {
        return self.type;
    }

    /// Ports `gguf_kv`'s `get_ne` (gguf.cpp:179 @c1d0e7a00).
    ///
    /// Return: the number of elements this value holds.
    fn getNe(self: *const Kv) usize {
        if (self.type == c.GGUF_TYPE_STRING) {
            const ne = self.data_string.items.len;
            impl.assert(self.is_array or ne == 1, "is_array || ne == 1");
            return ne;
        }
        const type_size = typeSize(self.type);
        impl.assert(self.data.items.len % type_size == 0, "data.size() % type_size == 0");
        const ne = self.data.items.len / type_size;
        impl.assert(self.is_array or ne == 1, "is_array || ne == 1");
        return ne;
    }

    /// Ports `gguf_kv`'s `get_val` (gguf.cpp:193 @c1d0e7a00), the non-string arm.
    fn getVal(self: *const Kv, comptime T: type, i: usize) T {
        impl.assert(typeToGgufType(T) == self.type, "type_to_gguf_type<T>::value == type");
        const type_size = typeSize(self.type);
        impl.assert(self.data.items.len % type_size == 0, "data.size() % type_size == 0");
        impl.assert(self.data.items.len >= (i + 1) * type_size, "data.size() >= (i+1)*type_size");
        var out: T = undefined;
        @memcpy(std.mem.asBytes(&out), self.data.items[i * @sizeOf(T) ..][0..@sizeOf(T)]);
        return out;
    }

    /// Ports `gguf_kv`'s `get_val` (gguf.cpp:193 @c1d0e7a00), the
    /// `std::string` `if constexpr` arm.
    fn getValStr(self: *const Kv, i: usize) Str {
        impl.assert(c.GGUF_TYPE_STRING == self.type, "type_to_gguf_type<T>::value == type");
        impl.assert(self.data_string.items.len >= i + 1, "data_string.size() >= i+1");
        return self.data_string.items[i];
    }

    /// Ports `gguf_kv`'s `cast` (gguf.cpp:205 @c1d0e7a00).
    ///
    /// Reinterprets the bytes already stored as a different element type. Only
    /// `gguf_set_arr_data` uses it, to stamp the caller's type onto a buffer
    /// that was built as raw bytes.
    fn cast(self: *Kv, new_type: c_uint) void {
        const new_type_size = typeSize(new_type);
        impl.assert(self.data.items.len % new_type_size == 0, "data.size() % new_type_size == 0");
        self.type = new_type;
    }
};

// -----------------------------------------------------------------------------
// Tensors and the context

/// Ports `struct gguf_tensor_info` (gguf.cpp:212 @c1d0e7a00).
const TensorInfo = struct {
    /// `struct ggml_tensor t` — held for the shape, type and name it carries,
    /// not as a live tensor. Zero-initialised; see the file header.
    t: c.ggml_tensor,
    /// Offset from the start of `data`, a multiple of `alignment`.
    offset: u64,
};

/// Ports `struct gguf_context` (gguf.cpp:217 @c1d0e7a00).
///
/// The C++ gives every field a default member initialiser and relies on `new
/// gguf_context` to apply them; the defaults here are the same values.
const Context = struct {
    version: u32 = c.GGUF_VERSION,

    kv: std.ArrayList(Kv) = .empty,
    info: std.ArrayList(TensorInfo) = .empty,

    alignment: usize = c.GGUF_DEFAULT_ALIGNMENT,
    /// Offset of `data` from the beginning of the file.
    offset: usize = 0,
    /// Size of `data` in bytes.
    size: usize = 0,

    data: ?*anyopaque = null,

    /// Stands in for `new gguf_context`.
    fn create() *Context {
        const self = allocator.create(Context) catch oom();
        self.* = .{};
        return self;
    }

    /// Stands in for `delete ctx`, which runs `~gguf_context` and with it every
    /// member's destructor.
    fn destroy(self: *Context) void {
        for (self.kv.items) |*kv| kv.deinit();
        self.kv.deinit(allocator);
        self.info.deinit(allocator);
        allocator.destroy(self);
    }
};

/// Narrows the opaque `struct gguf_context *` the C ABI passes to the real type.
inline fn ctxOf(p: ?*c.struct_gguf_context) *Context {
    return @ptrCast(@alignCast(p.?));
}

/// Narrows a `const struct gguf_context *`.
inline fn ctxOfConst(p: ?*const c.struct_gguf_context) *const Context {
    return @ptrCast(@alignCast(p.?));
}

// -----------------------------------------------------------------------------
// Reading
//
// Ports `struct gguf_reader` (gguf.cpp:230 @c1d0e7a00). The C++ passes it as a
// `const &` and marks `data_offset` and `nbytes_remain` `mutable`; this takes a
// `*Reader` instead, which says the same thing without the qualifier dance.
//
// The C++ dispatches `read` by overload and template specialisation. Zig has no
// overloading, so each arm keeps the C++'s parameter type in its name:
// `readInto` is `read(T &)`, `readVec` is `read(std::vector<T> &, n)`, and so
// on. Every one writes straight into the destination, including on a short
// read, because callers depend on that: `gguf_init_from_reader` assigns
// `info.t.ne[j] = 1` before reading over it and then range-checks whatever
// survived -- the `ne` assignment at (gguf.cpp:677 @c1d0e7a00).

/// Ports the `#else` arm of the `gguf_ftell` and `gguf_fseek` macros
/// (gguf.cpp:25, 26 @c1d0e7a00). The `_WIN32` arm names `_ftelli64` and
/// `_fseeki64`; this target compiles the other one.
const Libc = struct {
    extern fn ftello(stream: *c.FILE) i64;
    extern fn fseeko(stream: *c.FILE, offset: i64, whence: c_int) c_int;
    extern fn fread(ptr: *anyopaque, size: usize, nitems: usize, stream: *c.FILE) usize;
    extern fn fwrite(ptr: *const anyopaque, size: usize, nitems: usize, stream: *c.FILE) usize;
    extern fn fputc(char: c_int, stream: *c.FILE) c_int;
    extern fn fclose(stream: *c.FILE) c_int;
    extern fn strerror(errnum: c_int) [*:0]const u8;
    const SEEK_SET: c_int = 0;
    const SEEK_END: c_int = 2;
};

const Reader = struct {
    callback: c.gguf_reader_callback_t,
    userdata: ?*anyopaque,
    max_chunk_read: usize,
    data_offset: u64 = 0,
    nbytes_remain: u64 = 0,

    /// Ports the `gguf_reader` constructor (gguf.cpp:231 @c1d0e7a00).
    fn init(
        callback: c.gguf_reader_callback_t,
        userdata: ?*anyopaque,
        max_chunk_read: usize,
        data_offset: u64,
        nbytes_remain: u64,
    ) Reader {
        impl.assert(max_chunk_read > 0, "max_chunk_read > 0");
        return .{
            .callback = callback,
            .userdata = userdata,
            .max_chunk_read = max_chunk_read,
            .data_offset = data_offset,
            .nbytes_remain = nbytes_remain,
        };
    }

    /// Ports `gguf_reader`'s `file_remain` (gguf.cpp:246 @c1d0e7a00).
    ///
    /// Return: bytes from the current position to end of file, or 0 if any of
    /// the three seeks fails — in which case the original position is restored
    /// first, as the C++ does.
    fn fileRemain(file: *c.FILE) u64 {
        const cur = Libc.ftello(file);
        if (cur < 0) return 0;
        if (Libc.fseeko(file, 0, Libc.SEEK_END) != 0) {
            _ = Libc.fseeko(file, cur, Libc.SEEK_SET);
            return 0;
        }
        const end = Libc.ftello(file);
        if (end < 0) {
            _ = Libc.fseeko(file, cur, Libc.SEEK_SET);
            return 0;
        }
        _ = Libc.fseeko(file, cur, Libc.SEEK_SET);
        return @intCast(end - cur);
    }

    /// Ports `gguf_reader`'s `read` (gguf.cpp:267 @c1d0e7a00), the `T &` overload.
    fn readInto(self: *Reader, comptime T: type, dst: *T) bool {
        const size = @sizeOf(T);
        if (size > self.nbytes_remain) return false;
        return self.readRaw(std.mem.asBytes(dst)) == size;
    }

    /// Ports `gguf_reader`'s `read` (gguf.cpp:313 @c1d0e7a00), the `bool &` overload.
    fn readBool(self: *Reader, dst: *bool) bool {
        var tmp: i8 = -1;
        if (!self.readInto(i8, &tmp)) return false;
        dst.* = tmp != 0;
        return true;
    }

    /// Ports `gguf_reader`'s two enum overloads of `read`, `read`
    /// (gguf.cpp:322, 331 @c1d0e7a00). They differ only in the enum they cast
    /// to and are one function here, because both are `c_uint` after the
    /// import.
    fn readEnum(self: *Reader, dst: *c_uint) bool {
        var tmp: i32 = -1;
        if (!self.readInto(i32, &tmp)) return false;
        dst.* = @bitCast(tmp);
        return true;
    }

    /// Ports `gguf_reader`'s `read` (gguf.cpp:340 @c1d0e7a00), the
    /// `std::string &` overload.
    ///
    /// Treats `dst` as uninitialised and **always leaves it owning a valid
    /// string**, empty if the read failed before a length was known. The C++
    /// gets that for free: `dst` is a live `std::string` on entry and
    /// `resize` keeps it live whatever happens next. Callers depend on it —
    /// `gguf_init_from_reader` calls `ggml_set_name` with the tensor name even
    /// when the read failed (gguf.cpp:651 @c1d0e7a00), and every caller frees
    /// `dst`.
    fn readStr(self: *Reader, dst: *Str) bool {
        dst.* = allocStr(0);

        var size: u64 = 0;
        if (!self.readInto(u64, &size)) return false;
        if (size > max_string_length) {
            impl.logError(
                "%s: string length %llu exceeds maximum %llu\n",
                .{ "read", size, max_string_length },
            );
            return false;
        }
        if (size > self.nbytes_remain) {
            impl.logError(
                "%s: string length %llu exceeds remaining file size %llu bytes\n",
                .{ "read", size, self.nbytes_remain },
            );
            return false;
        }

        allocator.free(dst.*);
        dst.* = allocStr(@intCast(size));
        return self.readRaw(dst.*[0..@intCast(size)]) == size;
    }

    /// Ports the length guards of `gguf_reader`'s vector `read`
    /// (gguf.cpp:276 @c1d0e7a00), shared by the two element kinds below.
    ///
    /// `stride` is `sizeof(T)`, except for strings, where the C++ uses
    /// `sizeof(uint64_t)` because each element is length-prefixed.
    fn vecGuard(self: *const Reader, n: u64, stride: usize) bool {
        if (n > max_array_elements) return false;
        if (n > std.math.maxInt(usize) / stride) return false;
        if (self.nbytes_remain < n * stride) return false;
        return true;
    }

    /// Ports `gguf_reader`'s vector `read` (gguf.cpp:276 @c1d0e7a00), the
    /// non-string arm, including its `bool` special case.
    fn readVec(self: *Reader, comptime T: type, dst: *std.ArrayList(T), n: u64) bool {
        if (!self.vecGuard(n, @sizeOf(T))) return false;
        dst.resize(allocator, @intCast(n)) catch oom();
        // `std::vector::resize` value-initialises; Zig's leaves the new items
        // undefined. Nothing reads them on the failure path, but matching the
        // C++ keeps the port's output a function of its input only.
        @memset(dst.items, std.mem.zeroes(T));
        for (dst.items) |*slot| {
            if (T == bool) {
                var tmp: bool = undefined;
                if (!self.readBool(&tmp)) return false;
                slot.* = tmp;
            } else {
                if (!self.readInto(T, slot)) return false;
            }
        }
        return true;
    }

    /// Ports `gguf_reader`'s vector `read` (gguf.cpp:276 @c1d0e7a00), the
    /// `std::is_same<T, std::string>` arm.
    ///
    /// Every element is allocated, so a failure part-way leaves the prefix
    /// owned by `dst` and the rest empty; the caller frees the whole list.
    fn readVecStr(self: *Reader, dst: *std.ArrayList(Str), n: u64) bool {
        if (!self.vecGuard(n, @sizeOf(u64))) return false;
        dst.resize(allocator, @intCast(n)) catch oom();
        // Give every slot an owned empty string before any read can fail, so a
        // short read leaves the whole list freeable rather than half of it
        // pointing at undefined memory. `readStr` replaces each in turn.
        for (dst.items) |*slot| slot.* = allocStr(0);
        for (dst.items) |*slot| {
            allocator.free(slot.*);
            if (!self.readStr(slot)) return false;
        }
        return true;
    }

    /// Ports `gguf_reader`'s raw `read` (gguf.cpp:357 @c1d0e7a00), the
    /// `void *` overload.
    fn readBytes(self: *Reader, dst: []u8) bool {
        if (dst.len > self.nbytes_remain) return false;
        return self.readRaw(dst) == dst.len;
    }

    /// Ports `gguf_reader`'s `tell` (gguf.cpp:364 @c1d0e7a00).
    fn tell(self: *const Reader) u64 {
        return self.data_offset;
    }

    /// Ports `gguf_reader`'s `seek` (gguf.cpp:368 @c1d0e7a00).
    ///
    /// Return: false if `absolute_offset` is past the end of what remains, in
    /// which case nothing moves.
    fn seek(self: *Reader, absolute_offset: u64) bool {
        const end_offset = self.data_offset + self.nbytes_remain;
        if (absolute_offset > end_offset) return false;
        self.data_offset = absolute_offset;
        self.nbytes_remain = end_offset - absolute_offset;
        return true;
    }

    /// Ports `gguf_reader`'s `read_raw` (gguf.cpp:381 @c1d0e7a00).
    ///
    /// Return: the number of bytes actually read. A short read latches
    /// `nbytes_remain` to zero, so every later read fails its guard rather than
    /// calling the callback again.
    fn readRaw(self: *Reader, dst: []u8) usize {
        if (self.callback == null or dst.len == 0) return 0;

        var total_nread: usize = 0;
        var reached_eof = false;

        while (total_nread < dst.len) {
            const chunk_size = @min(self.max_chunk_read, dst.len - total_nread);
            // The C++ guards an unsigned wrap on a 64-bit offset; `+%` is that
            // addition, and the comparison is the same one.
            if (self.data_offset +% total_nread < self.data_offset) break;
            const nread = self.callback.?(
                self.userdata,
                @ptrCast(dst[total_nread..].ptr),
                self.data_offset + total_nread,
                chunk_size,
            );
            total_nread += nread;
            if (nread != chunk_size) {
                reached_eof = true;
                break;
            }
        }

        self.data_offset += total_nread;
        impl.assert(total_nread <= self.nbytes_remain, "total_nread <= nbytes_remain");
        self.nbytes_remain -= total_nread;

        if (reached_eof) self.nbytes_remain = 0;

        return total_nread;
    }
};

// -----------------------------------------------------------------------------
// Construction

/// Ports `gguf_init_empty` (gguf.cpp:421 @c1d0e7a00).
///
/// Return: a context with no keys and no tensors, owned by the caller and
/// released with `gguf_free`.
export fn gguf_init_empty() callconv(.c) ?*c.struct_gguf_context {
    return @ptrCast(Context.create());
}

/// Ports `gguf_read_emplace_helper` (gguf.cpp:426 @c1d0e7a00), the non-string
/// instantiations.
///
/// The C++ wraps the array read in `catch (std::length_error &)` and
/// `catch (std::bad_alloc &)`, both of which only fire on allocation failure
/// inside `std::vector::resize`. This port aborts there instead; see the file
/// header.
fn readEmplaceHelper(
    comptime T: type,
    gr: *Reader,
    kv: *std.ArrayList(Kv),
    key: []const u8,
    is_array: bool,
    n: u64,
) bool {
    if (is_array) {
        var value: std.ArrayList(T) = .empty;
        defer value.deinit(allocator);
        if (!gr.readVec(T, &value, n)) return false;
        kv.append(allocator, Kv.initArray(key, T, value.items)) catch oom();
    } else {
        var value: T = std.mem.zeroes(T);
        if (!gr.readInto(T, &value)) return false;
        kv.append(allocator, Kv.initScalar(key, T, value)) catch oom();
    }
    return true;
}

/// Ports `gguf_read_emplace_helper` (gguf.cpp:426 @c1d0e7a00), the `bool`
/// instantiation.
///
/// Separate from `readEmplaceHelper` because the scalar arm goes through the
/// `bool &` overload of `read` (gguf.cpp:313 @c1d0e7a00) rather than the
/// byte-sized template, and reads an `int8_t` it then compares against zero.
fn readEmplaceHelperBool(
    gr: *Reader,
    kv: *std.ArrayList(Kv),
    key: []const u8,
    is_array: bool,
    n: u64,
) bool {
    if (is_array) {
        var value: std.ArrayList(bool) = .empty;
        defer value.deinit(allocator);
        if (!gr.readVec(bool, &value, n)) return false;
        kv.append(allocator, Kv.initArray(key, bool, value.items)) catch oom();
    } else {
        var value: bool = false;
        if (!gr.readBool(&value)) return false;
        kv.append(allocator, Kv.initScalar(key, bool, value)) catch oom();
    }
    return true;
}

/// Ports `gguf_read_emplace_helper` (gguf.cpp:426 @c1d0e7a00), the
/// `std::string` instantiation.
fn readEmplaceHelperStr(
    gr: *Reader,
    kv: *std.ArrayList(Kv),
    key: []const u8,
    is_array: bool,
    n: u64,
) bool {
    if (is_array) {
        var value: std.ArrayList(Str) = .empty;
        const ok = gr.readVecStr(&value, n);
        if (!ok) {
            for (value.items) |s| allocator.free(s);
            value.deinit(allocator);
            return false;
        }
        // `initStrArrayOwned` takes the list; the C++ copies it and lets the
        // local go out of scope.
        kv.append(allocator, Kv.initStrArrayOwned(key, value)) catch oom();
    } else {
        var value: Str = undefined;
        const ok = gr.readStr(&value);
        defer allocator.free(value);
        if (!ok) return false;
        kv.append(allocator, Kv.initStr(key, value)) catch oom();
    }
    return true;
}

/// Ports `gguf_init_from_reader` (gguf.cpp:451 @c1d0e7a00).
///
/// Parameters:
/// - `gr`: the reader, positioned at the start of the GGUF header.
/// - `params`: `no_alloc` and the optional `ggml_context` out-parameter.
///
/// Return: the parsed context, or null on any malformed input. Every failure
/// path frees what it built, as the C++ does.
fn initFromReader(gr: *Reader, params: c.gguf_init_params) ?*Context {
    const ctx = Context.create();

    var ok = true;

    // file magic
    {
        var magic: std.ArrayList(u8) = .empty;
        defer magic.deinit(allocator);
        ok = ok and gr.readVec(u8, &magic, 4);

        if (!ok) {
            impl.logError("%s: failed to read magic\n", .{"gguf_init_from_reader"});
            ctx.destroy();
            return null;
        }

        for (magic.items, 0..) |ch, i| {
            if (ch != c.GGUF_MAGIC[i]) {
                const c0 = if (std.ascii.isPrint(magic.items[0])) magic.items[0] else '?';
                const c1 = if (std.ascii.isPrint(magic.items[1])) magic.items[1] else '?';
                const c2 = if (std.ascii.isPrint(magic.items[2])) magic.items[2] else '?';
                const c3 = if (std.ascii.isPrint(magic.items[3])) magic.items[3] else '?';
                impl.logError(
                    "%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n",
                    .{ "gguf_init_from_reader", @as(c_int, c0), @as(c_int, c1), @as(c_int, c2), @as(c_int, c3) },
                );
                ctx.destroy();
                return null;
            }
        }
    }

    // header
    var n_kv: i64 = 0;
    var n_tensors: i64 = 0;

    if (ok and gr.readInto(u32, &ctx.version)) {
        if (ok and ctx.version == 0) {
            impl.logError("%s: bad GGUF version: %u\n", .{ "gguf_init_from_reader", ctx.version });
            ok = false;
        }

        // The C++ comment: a non-native-endian file reads GGUFv3 as 0x30000000,
        // so a zero low half says the file's endianness is not the host's.
        if (ok and (ctx.version & 0x0000FFFF) == 0x00000000) {
            impl.logError(
                "%s: failed to load model: this GGUF file version %u is extremely large, is there a mismatch between the host and model endianness?\n",
                .{ "gguf_init_from_reader", ctx.version },
            );
            ok = false;
        }

        if (ok and ctx.version == 1) {
            impl.logError(
                "%s: GGUFv1 is no longer supported, please use a more up-to-date version\n",
                .{"gguf_init_from_reader"},
            );
            ok = false;
        }
        if (ok and ctx.version > c.GGUF_VERSION) {
            impl.logError(
                "%s: this GGUF file is version %u but this software only supports up to version %d\n",
                .{ "gguf_init_from_reader", ctx.version, @as(c_int, c.GGUF_VERSION) },
            );
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok and gr.readInto(i64, &n_tensors)) {
        // The C++ bounds this by `SIZE_MAX/sizeof(gguf_tensor_info)`. Our
        // `TensorInfo` is a different size from the C++ struct, so the limit
        // differs — both are around 1e17 and neither is reachable by a real
        // file; what the check is for is a negative or absurd count.
        if (n_tensors < 0 or n_tensors > @as(i64, @intCast(std.math.maxInt(usize) / @sizeOf(TensorInfo)))) {
            impl.logError(
                "%s: number of tensors is %lld but must be in [0, %zu]\n",
                .{ "gguf_init_from_reader", n_tensors, @as(usize, std.math.maxInt(usize) / @sizeOf(TensorInfo)) },
            );
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok and gr.readInto(i64, &n_kv)) {
        if (n_kv < 0 or n_kv > @as(i64, @intCast(std.math.maxInt(usize) / @sizeOf(Kv)))) {
            impl.logError(
                "%s: number of key value pairs is %lld but must be in [0, %zu]\n",
                .{ "gguf_init_from_reader", n_kv, @as(usize, std.math.maxInt(usize) / @sizeOf(Kv)) },
            );
            ok = false;
        }
    } else {
        ok = false;
    }

    if (!ok) {
        impl.logError("%s: failed to read header\n", .{"gguf_init_from_reader"});
        ctx.destroy();
        return null;
    }

    // KV pairs
    {
        var i: i64 = 0;
        while (ok and i < n_kv) : (i += 1) {
            var key: Str = undefined;
            const key_ok = gr.readStr(&key);
            defer allocator.free(key);
            ok = ok and key_ok;

            if (ok and key.len == 0) {
                impl.logError("%s: key %lld is empty\n", .{ "gguf_init_from_reader", i });
                ok = false;
            }
            var j: usize = 0;
            while (ok and j < ctx.kv.items.len) : (j += 1) {
                // `std::string::operator==` compares the full length, not up to
                // the first NUL the way `gguf_find_key`'s `strcmp` does.
                if (std.mem.eql(u8, key, ctx.kv.items[j].key)) {
                    impl.logError(
                        "%s: duplicate key '%s' for tensors %zu and %lld \n",
                        .{ "gguf_init_from_reader", key.ptr, j, i },
                    );
                    ok = false;
                }
            }
            if (!ok) break;

            var kv_type: c_uint = @bitCast(@as(i32, -1));
            var is_array = false;
            var n: u64 = 1;

            ok = ok and gr.readEnum(&kv_type);
            if (kv_type == c.GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok and gr.readEnum(&kv_type);
                ok = ok and gr.readInto(u64, &n);
            }
            if (!ok) break;

            ok = ok and switch (kv_type) {
                c.GGUF_TYPE_UINT8 => readEmplaceHelper(u8, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_INT8 => readEmplaceHelper(i8, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_UINT16 => readEmplaceHelper(u16, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_INT16 => readEmplaceHelper(i16, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_UINT32 => readEmplaceHelper(u32, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_INT32 => readEmplaceHelper(i32, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_FLOAT32 => readEmplaceHelper(f32, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_BOOL => readEmplaceHelperBool(gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_STRING => readEmplaceHelperStr(gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_UINT64 => readEmplaceHelper(u64, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_INT64 => readEmplaceHelper(i64, gr, &ctx.kv, key, is_array, n),
                c.GGUF_TYPE_FLOAT64 => readEmplaceHelper(f64, gr, &ctx.kv, key, is_array, n),
                else => blk: {
                    impl.logError(
                        "%s: key '%s' has invalid GGUF type %d\n",
                        .{ "gguf_init_from_reader", key.ptr, @as(c_int, @bitCast(kv_type)) },
                    );
                    break :blk false;
                },
            };
        }

        if (!ok) {
            impl.logError("%s: failed to read key-value pairs\n", .{"gguf_init_from_reader"});
            ctx.destroy();
            return null;
        }
        impl.assert(@as(i64, @intCast(ctx.kv.items.len)) == n_kv, "int64_t(ctx->kv.size()) == n_kv");

        const alignment_idx = gguf_find_key(@ptrCast(ctx), c.GGUF_KEY_GENERAL_ALIGNMENT);
        if (alignment_idx != -1 and gguf_get_kv_type(@ptrCast(ctx), alignment_idx) != c.GGUF_TYPE_UINT32) {
            impl.logError(
                "%s: key '%s' must be of type %s but is %s\n",
                .{
                    "gguf_init_from_reader",
                    @as([*:0]const u8, c.GGUF_KEY_GENERAL_ALIGNMENT),
                    gguf_type_name(c.GGUF_TYPE_UINT32),
                    gguf_type_name(gguf_get_kv_type(@ptrCast(ctx), alignment_idx)),
                },
            );
            ctx.destroy();
            return null;
        }
        ctx.alignment = if (alignment_idx == -1)
            c.GGUF_DEFAULT_ALIGNMENT
        else
            gguf_get_val_u32(@ptrCast(ctx), alignment_idx);

        if (ctx.alignment == 0 or (ctx.alignment & (ctx.alignment - 1)) != 0) {
            impl.logError(
                "%s: alignment %zu is not a power of 2\n",
                .{ "gguf_init_from_reader", ctx.alignment },
            );
            ctx.destroy();
            return null;
        }
    }

    // read the tensor info
    {
        var i: i64 = 0;
        while (ok and i < n_tensors) : (i += 1) {
            // Zeroed rather than left indeterminate; see the file header.
            var info: TensorInfo = .{ .t = std.mem.zeroes(c.ggml_tensor), .offset = 0 };

            // tensor name
            {
                var name: Str = undefined;
                const name_ok = gr.readStr(&name);
                defer allocator.free(name);
                ok = ok and name_ok;

                if (name.len >= c.GGML_MAX_NAME) {
                    impl.logError(
                        "%s: tensor name %lld is too long: %zu >= %d\n",
                        .{ "gguf_init_from_reader", i, name.len, @as(c_int, c.GGML_MAX_NAME) },
                    );
                    ok = false;
                    break;
                }
                _ = c.ggml_set_name(&info.t, name.ptr);

                // make sure there are no duplicate tensor names
                var j: i64 = 0;
                while (ok and j < i) : (j += 1) {
                    if (std.mem.orderZ(u8, @ptrCast(&info.t.name), @ptrCast(&ctx.info.items[@intCast(j)].t.name)) == .eq) {
                        impl.logError(
                            "%s: duplicate tensor name '%s' for tensors %lld and %lld\n",
                            .{ "gguf_init_from_reader", @as([*:0]const u8, @ptrCast(&info.t.name)), j, i },
                        );
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) break;

            // tensor shape
            {
                var n_dims: u32 = 0;
                ok = ok and gr.readInto(u32, &n_dims);
                if (n_dims > c.GGML_MAX_DIMS) {
                    impl.logError(
                        "%s: tensor '%s' has invalid number of dimensions: %u > %u\n",
                        .{ "gguf_init_from_reader", @as([*:0]const u8, @ptrCast(&info.t.name)), n_dims, @as(c_uint, c.GGML_MAX_DIMS) },
                    );
                    ok = false;
                    break;
                }
                var bad_dim = false;
                var j: u32 = 0;
                while (ok and j < c.GGML_MAX_DIMS) : (j += 1) {
                    info.t.ne[j] = 1;
                    if (j < n_dims) {
                        ok = ok and gr.readInto(i64, &info.t.ne[j]);
                    }

                    // check that all ne are non-negative
                    if (info.t.ne[j] < 0) {
                        impl.logError(
                            "%s: tensor '%s' dimension %u has invalid number of elements: %lld < 0\n",
                            .{ "gguf_init_from_reader", @as([*:0]const u8, @ptrCast(&info.t.name)), j, info.t.ne[j] },
                        );
                        ok = false;
                        bad_dim = true;
                        break;
                    }
                }
                if (bad_dim) break;

                // check that the total number of elements is representable
                // (a zero-element tensor is trivially representable; the guard
                // also avoids a division by zero below)
                if (ok and c.ggml_nelements(&info.t) > 0 and
                    ((@divTrunc(std.math.maxInt(i64), info.t.ne[1]) <= info.t.ne[0]) or
                        (@divTrunc(std.math.maxInt(i64), info.t.ne[2]) <= info.t.ne[0] * info.t.ne[1]) or
                        (@divTrunc(std.math.maxInt(i64), info.t.ne[3]) <= info.t.ne[0] * info.t.ne[1] * info.t.ne[2])))
                {
                    impl.logError(
                        "%s: total number of elements in tensor '%s' with shape (%lld, %lld, %lld, %lld) is >= %lld\n",
                        .{
                            "gguf_init_from_reader",
                            @as([*:0]const u8, @ptrCast(&info.t.name)),
                            info.t.ne[0],
                            info.t.ne[1],
                            info.t.ne[2],
                            info.t.ne[3],
                            @as(i64, std.math.maxInt(i64)),
                        },
                    );
                    ok = false;
                    break;
                }
            }
            if (!ok) break;

            // tensor type
            {
                var t_type: c_uint = 0;
                ok = ok and gr.readEnum(&t_type);
                info.t.type = t_type;

                // check that tensor type is within defined range
                if (@as(i32, @bitCast(info.t.type)) < 0 or info.t.type >= c.GGML_TYPE_COUNT) {
                    impl.logError(
                        "%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                        .{
                            "gguf_init_from_reader",
                            @as([*:0]const u8, @ptrCast(&info.t.name)),
                            @as(c_int, @bitCast(info.t.type)),
                            @as(c_int, c.GGML_TYPE_COUNT),
                        },
                    );
                    ok = false;
                    break;
                }
                const type_size = c.ggml_type_size(info.t.type);
                const blck_size = c.ggml_blck_size(info.t.type);

                // check that row size is divisible by block size
                if (blck_size == 0 or @rem(info.t.ne[0], blck_size) != 0) {
                    impl.logError(
                        "%s: tensor '%s' of type %d (%s) has %lld elements per row, not a multiple of block size (%lld)\n",
                        .{
                            "gguf_init_from_reader",
                            @as([*:0]const u8, @ptrCast(&info.t.name)),
                            @as(c_int, @bitCast(info.t.type)),
                            c.ggml_type_name(info.t.type),
                            info.t.ne[0],
                            blck_size,
                        },
                    );
                    ok = false;
                    break;
                }

                // check that the size of the tensor in bytes is representable
                if (ok and @as(u64, @intCast(@divTrunc(c.ggml_nelements(&info.t), c.ggml_blck_size(info.t.type)))) >
                    std.math.maxInt(usize) / c.ggml_type_size(info.t.type))
                {
                    impl.logError(
                        "%s: tensor '%s' with shape (%lld, %lld, %lld, %lld) has a size in bytes > %zu\n",
                        .{
                            "gguf_init_from_reader",
                            @as([*:0]const u8, @ptrCast(&info.t.name)),
                            info.t.ne[0],
                            info.t.ne[1],
                            info.t.ne[2],
                            info.t.ne[3],
                            @as(usize, std.math.maxInt(usize)),
                        },
                    );
                    ok = false;
                    break;
                }

                // calculate byte offsets given the tensor shape and type
                info.t.nb[0] = type_size;
                info.t.nb[1] = info.t.nb[0] * @as(usize, @intCast(@divTrunc(info.t.ne[0], blck_size)));
                for (2..c.GGML_MAX_DIMS) |j| {
                    info.t.nb[j] = info.t.nb[j - 1] * @as(usize, @intCast(info.t.ne[j - 1]));
                }
            }
            if (!ok) break;

            // tensor data offset within buffer
            ok = ok and gr.readInto(u64, &info.offset);

            ctx.info.append(allocator, info) catch oom();
        }
    }

    if (!ok) {
        impl.logError("%s: failed to read tensor info\n", .{"gguf_init_from_reader"});
        ctx.destroy();
        return null;
    }
    impl.assert(@as(i64, @intCast(ctx.info.items.len)) == n_tensors, "int64_t(ctx->info.size()) == n_tensors");

    // we require the data section to be aligned, so take into account any padding
    if (n_tensors > 0 and !gr.seek(impl.pad(gr.tell(), ctx.alignment))) {
        impl.logError("%s: failed to seek to beginning of data section\n", .{"gguf_init_from_reader"});
        ctx.destroy();
        return null;
    }

    // store the current file offset - this is where the data section starts
    ctx.offset = @intCast(gr.tell());

    // compute the total size of the data section, taking into account the alignment
    {
        ctx.size = 0;
        for (ctx.info.items) |*ti| {
            if (ti.offset != ctx.size) {
                impl.logError(
                    "%s: tensor '%s' has offset %llu, expected %zu\n",
                    .{ "gguf_init_from_reader", @as([*:0]const u8, @ptrCast(&ti.t.name)), ti.offset, ctx.size },
                );
                impl.logError("%s: failed to read tensor data\n", .{"gguf_init_from_reader"});
                ctx.destroy();
                return null;
            }
            const padded_size = impl.pad(c.ggml_nbytes(&ti.t), ctx.alignment);
            if (std.math.maxInt(usize) - ctx.size < padded_size) {
                impl.logError(
                    "%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    .{ "gguf_init_from_reader", @as([*:0]const u8, @ptrCast(&ti.t.name)), ctx.size, padded_size },
                );
                ctx.destroy();
                return null;
            }
            ctx.size += padded_size;
        }
    }

    // load the tensor data only if requested
    if (params.ctx != null) {
        // if the provided gguf_context is no_alloc, then we create "empty"
        // tensors and do not read the binary blob; otherwise we load the blob
        // into the created ggml_context as well and point each tensor's `data`
        // at the right place inside it.

        // compute the exact size needed for the new ggml_context
        var mem_size: usize = 0;
        const n_tensors_u: usize = @intCast(n_tensors);
        if (params.no_alloc) {
            if (n_tensors != 0 and std.math.maxInt(usize) / n_tensors_u < c.ggml_tensor_overhead()) {
                impl.logError("%s: memory size overflow while allocating ggml context\n", .{"gguf_init_from_reader"});
                ctx.destroy();
                return null;
            }

            mem_size = n_tensors_u * c.ggml_tensor_overhead();
        } else {
            if ((n_tensors + 1) != 0 and std.math.maxInt(usize) / (n_tensors_u + 1) < c.ggml_tensor_overhead()) {
                impl.logError("%s: memory size overflow while allocating ggml context\n", .{"gguf_init_from_reader"});
                ctx.destroy();
                return null;
            }

            const overhead = (n_tensors_u + 1) * c.ggml_tensor_overhead();

            if (std.math.maxInt(usize) - overhead < ctx.size) {
                impl.logError("%s: memory size overflow while allocating ggml context\n", .{"gguf_init_from_reader"});
                ctx.destroy();
                return null;
            }

            mem_size = overhead + ctx.size;
        }

        const pdata: c.ggml_init_params = .{
            .mem_size = mem_size,
            .mem_buffer = null,
            .no_alloc = params.no_alloc,
        };

        params.ctx.* = c.ggml_init(pdata);
        if (params.ctx.* == null) {
            impl.logError("%s: failed to initialize ggml context for storing tensors\n", .{"gguf_init_from_reader"});
            ctx.destroy();
            return null;
        }

        const ctx_data = params.ctx.*;

        var data: ?*c.ggml_tensor = null;

        if (!params.no_alloc) {
            data = c.ggml_new_tensor_1d(ctx_data, c.GGML_TYPE_I8, @intCast(ctx.size));

            ok = ok and data != null;

            if (ok) {
                _ = c.ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok and gr.readBytes(@as([*]u8, @ptrCast(data.?.data))[0..ctx.size]);

            if (!ok) {
                impl.logError("%s: failed to read tensor data binary blob\n", .{"gguf_init_from_reader"});
                c.ggml_free(ctx_data);
                params.ctx.* = null;
                ctx.destroy();
                return null;
            }

            ctx.data = data.?.data;
        }

        c.ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (ctx.info.items) |*info| {
            const cur = c.ggml_new_tensor(ctx_data, info.t.type, c.GGML_MAX_DIMS, &info.t.ne);

            ok = ok and cur != null;

            if (!ok) break;

            _ = c.ggml_set_name(cur, @ptrCast(&info.t.name));

            // point the data member to the appropriate location in the binary
            // blob using the tensor info
            if (!params.no_alloc) {
                cur.*.data = @as([*]u8, @ptrCast(data.?.data)) + info.offset;
            }
        }

        if (!ok) {
            impl.logError("%s: failed to create tensors\n", .{"gguf_init_from_reader"});
            c.ggml_free(ctx_data);
            params.ctx.* = null;
            ctx.destroy();
            return null;
        }

        c.ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    return ctx;
}

/// Ports `gguf_init_from_callback` (gguf.cpp:909 @c1d0e7a00).
///
/// Parameters:
/// - `callback`: fills a buffer from an arbitrary source; null returns null.
/// - `userdata`: passed back to `callback` untouched.
/// - `max_chunk_read`: largest single request; 0 means no limit.
/// - `max_expected_size`: bytes the source is believed to hold.
/// - `params`: forwarded to the reader.
///
/// Return: the parsed context, or null.
export fn gguf_init_from_callback(
    callback: c.gguf_reader_callback_t,
    userdata: ?*anyopaque,
    max_chunk_read: usize,
    max_expected_size: u64,
    params: c.gguf_init_params,
) callconv(.c) ?*c.struct_gguf_context {
    if (callback == null) return null;

    var gr = Reader.init(
        callback,
        userdata,
        if (max_chunk_read == 0) std.math.maxInt(usize) else max_chunk_read,
        0,
        max_expected_size,
    );
    return @ptrCast(initFromReader(&gr, params));
}

/// Ports `struct gguf_file_reader` (gguf.cpp:918 @c1d0e7a00).
const FileReader = struct {
    file: *c.FILE,
    offset: u64,
};

/// Ports `gguf_file_reader_callback` (gguf.cpp:923 @c1d0e7a00).
///
/// Seeks only when the requested offset is not where the file already sits,
/// which keeps a sequential parse to one `fread` per call.
fn fileReaderCallback(userdata: ?*anyopaque, output: ?*anyopaque, offset: u64, len: usize) callconv(.c) usize {
    impl.assert(len > 0, "len > 0");

    const reader: *FileReader = @ptrCast(@alignCast(userdata.?));

    if (reader.offset != offset) {
        if (offset > std.math.maxInt(i64) or
            Libc.fseeko(reader.file, @intCast(offset), Libc.SEEK_SET) != 0)
        {
            return 0;
        }

        reader.offset = offset;
    }

    const nread = Libc.fread(output.?, 1, len, reader.file);
    reader.offset += nread;
    return nread;
}

/// Ports `gguf_init_from_file_ptr` (gguf.cpp:941 @c1d0e7a00).
///
/// Parameters:
/// - `file`: an open, seekable stream positioned at the start of the GGUF data.
/// - `params`: forwarded to the reader.
///
/// Return: the parsed context, or null. The stream is left where the parse
/// stopped and is not closed.
export fn gguf_init_from_file_ptr(file: ?*c.FILE, params: c.gguf_init_params) callconv(.c) ?*c.struct_gguf_context {
    const f = file orelse return null;

    const cur = Libc.ftello(f);
    if (cur < 0) return null;

    var reader: FileReader = .{ .file = f, .offset = @intCast(cur) };
    var gr = Reader.init(
        fileReaderCallback,
        &reader,
        std.math.maxInt(usize),
        reader.offset,
        Reader.fileRemain(f),
    );
    return @ptrCast(initFromReader(&gr, params));
}

/// Ports `struct gguf_buffer_reader` (gguf.cpp:959 @c1d0e7a00).
const BufferReader = struct {
    data: [*]const u8,
    size: usize,
};

/// Ports `gguf_buffer_reader_callback` (gguf.cpp:964 @c1d0e7a00).
fn bufferReaderCallback(userdata: ?*anyopaque, output: ?*anyopaque, offset: u64, len: usize) callconv(.c) usize {
    impl.assert(len > 0, "len > 0");

    const reader: *const BufferReader = @ptrCast(@alignCast(userdata.?));

    if (offset > reader.size or len > reader.size - offset) return 0;

    const data_offset: usize = @intCast(offset);
    const nread = @min(len, reader.size - data_offset);
    @memcpy(@as([*]u8, @ptrCast(output.?))[0..nread], reader.data[data_offset..][0..nread]);
    return nread;
}

/// Ports `gguf_init_from_buffer` (gguf.cpp:979 @c1d0e7a00).
///
/// Parameters:
/// - `data`: the GGUF bytes; null or a zero `size` returns null.
/// - `size`: how many bytes `data` holds.
/// - `params`: forwarded to the reader.
///
/// Return: the parsed context, or null. `data` is only read during the call.
export fn gguf_init_from_buffer(data: ?*const anyopaque, size: usize, params: c.gguf_init_params) callconv(.c) ?*c.struct_gguf_context {
    if (data == null or size == 0) return null;

    var reader: BufferReader = .{ .data = @ptrCast(data.?), .size = size };
    var gr = Reader.init(bufferReaderCallback, &reader, std.math.maxInt(usize), 0, size);
    return @ptrCast(initFromReader(&gr, params));
}

/// Ports `gguf_init_from_file` (gguf.cpp:992 @c1d0e7a00).
///
/// Parameters:
/// - `fname`: path to the GGUF file.
/// - `params`: forwarded to the reader.
///
/// Return: the parsed context, or null. The file is closed before returning
/// either way.
export fn gguf_init_from_file(fname: [*:0]const u8, params: c.gguf_init_params) callconv(.c) ?*c.struct_gguf_context {
    const file = c.ggml_fopen(fname, "rb") orelse {
        impl.logError(
            "%s: failed to open GGUF file '%s' (%s)\n",
            .{ "gguf_init_from_file", fname, Libc.strerror(std.c._errno().*) },
        );
        return null;
    };

    const result = gguf_init_from_file_ptr(file, params);
    _ = Libc.fclose(file);
    return result;
}

/// Ports `gguf_free` (gguf.cpp:1005 @c1d0e7a00).
///
/// Return: nothing. Null is accepted and ignored, as the C++'s `delete` is.
export fn gguf_free(ctx: ?*c.struct_gguf_context) callconv(.c) void {
    if (ctx == null) return;
    ctxOf(ctx).destroy();
}

/// Ports `gguf_type_name` (gguf.cpp:1012 @c1d0e7a00).
///
/// Return: a static name, or null for a value outside `enum gguf_type`.
export fn gguf_type_name(t: c_uint) callconv(.c) ?[*:0]const u8 {
    return typeNameOrNull(t);
}

// -----------------------------------------------------------------------------
// Accessors
//
// Every one of these asserts its key or tensor index is in range, exactly as
// the C++ does; the header documents them as aborting on a wrong type rather
// than reporting.

/// Ports `gguf_get_version` (gguf.cpp:1017 @c1d0e7a00).
export fn gguf_get_version(ctx: ?*const c.struct_gguf_context) callconv(.c) u32 {
    return ctxOfConst(ctx).version;
}

/// Ports `gguf_get_alignment` (gguf.cpp:1021 @c1d0e7a00).
export fn gguf_get_alignment(ctx: ?*const c.struct_gguf_context) callconv(.c) usize {
    return ctxOfConst(ctx).alignment;
}

/// Ports `gguf_get_data_offset` (gguf.cpp:1025 @c1d0e7a00).
export fn gguf_get_data_offset(ctx: ?*const c.struct_gguf_context) callconv(.c) usize {
    return ctxOfConst(ctx).offset;
}

/// Ports `gguf_get_n_kv` (gguf.cpp:1029 @c1d0e7a00).
export fn gguf_get_n_kv(ctx: ?*const c.struct_gguf_context) callconv(.c) i64 {
    return @intCast(ctxOfConst(ctx).kv.items.len);
}

/// Ports `gguf_find_key` (gguf.cpp:1033 @c1d0e7a00).
///
/// Return: the key's index, or -1 if absent. Matching is `strcmp`, so it stops
/// at the first NUL in either string — unlike the duplicate-key check in
/// `initFromReader`, which compares full lengths.
export fn gguf_find_key(ctx: ?*const c.struct_gguf_context, key: [*:0]const u8) callconv(.c) i64 {
    var keyfound: i64 = -1;

    const n_kv = gguf_get_n_kv(ctx);

    var i: i64 = 0;
    while (i < n_kv) : (i += 1) {
        if (std.mem.orderZ(u8, key, gguf_get_key(ctx, i)) == .eq) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

/// Ports `gguf_get_key` (gguf.cpp:1049 @c1d0e7a00).
///
/// Return: the key, borrowed from the context and valid until it is freed or
/// the key is removed.
export fn gguf_get_key(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) [*:0]const u8 {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    return ctxOfConst(ctx).kv.items[@intCast(key_id)].getKey().ptr;
}

/// Ports `gguf_get_kv_type` (gguf.cpp:1054 @c1d0e7a00).
///
/// Return: `GGUF_TYPE_ARRAY` for any array-valued key, whatever its elements
/// are; `gguf_get_arr_type` reports the element type.
export fn gguf_get_kv_type(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) c_uint {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];
    return if (kv.is_array) c.GGUF_TYPE_ARRAY else kv.getType();
}

/// Ports `gguf_get_arr_type` (gguf.cpp:1059 @c1d0e7a00).
export fn gguf_get_arr_type(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) c_uint {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];
    impl.assert(kv.is_array, "ctx->kv[key_id].is_array");
    return kv.getType();
}

/// Ports `gguf_get_arr_data` (gguf.cpp:1065 @c1d0e7a00).
///
/// Return: the first element of the array, borrowed. Bool arrays are stored
/// one byte per element, as the header notes.
export fn gguf_get_arr_data(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) ?*const anyopaque {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];
    impl.assert(kv.getType() != c.GGUF_TYPE_STRING, "ctx->kv[key_id].get_type() != GGUF_TYPE_STRING");
    return @ptrCast(kv.data.items.ptr);
}

/// Ports `gguf_get_arr_str` (gguf.cpp:1071 @c1d0e7a00).
///
/// Return: the `i`th string, borrowed from the context.
export fn gguf_get_arr_str(ctx: ?*const c.struct_gguf_context, key_id: i64, i: usize) callconv(.c) [*:0]const u8 {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];
    impl.assert(kv.getType() == c.GGUF_TYPE_STRING, "ctx->kv[key_id].get_type() == GGUF_TYPE_STRING");
    return kv.data_string.items[i].ptr;
}

/// Ports `gguf_get_arr_n` (gguf.cpp:1077 @c1d0e7a00).
///
/// Return: the element count. Computed from the byte length rather than by
/// calling `get_ne`, so it does not assert the non-array single-element rule.
export fn gguf_get_arr_n(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) usize {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];

    if (kv.type == c.GGUF_TYPE_STRING) {
        return kv.data_string.items.len;
    }

    const type_size = typeSize(kv.type);
    impl.assert(kv.data.items.len % type_size == 0, "ctx->kv[key_id].data.size() % type_size == 0");
    return kv.data.items.len / type_size;
}

/// Shared body of the twelve `gguf_get_val_*` accessors, each the same three
/// lines over a different type. The family begins at `gguf_get_val_u8`
/// (gguf.cpp:1089 @c1d0e7a00) and ends at `gguf_get_val_data`.
inline fn getVal(comptime T: type, ctx: ?*const c.struct_gguf_context, key_id: i64) T {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];
    impl.assert(kv.getNe() == 1, "ctx->kv[key_id].get_ne() == 1");
    return kv.getVal(T, 0);
}

/// Ports `gguf_get_val_u8` (gguf.cpp:1089 @c1d0e7a00).
export fn gguf_get_val_u8(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) u8 {
    return getVal(u8, ctx, key_id);
}

/// Ports `gguf_get_val_i8` (gguf.cpp:1095 @c1d0e7a00).
export fn gguf_get_val_i8(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) i8 {
    return getVal(i8, ctx, key_id);
}

/// Ports `gguf_get_val_u16` (gguf.cpp:1101 @c1d0e7a00).
export fn gguf_get_val_u16(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) u16 {
    return getVal(u16, ctx, key_id);
}

/// Ports `gguf_get_val_i16` (gguf.cpp:1107 @c1d0e7a00).
export fn gguf_get_val_i16(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) i16 {
    return getVal(i16, ctx, key_id);
}

/// Ports `gguf_get_val_u32` (gguf.cpp:1113 @c1d0e7a00).
export fn gguf_get_val_u32(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) u32 {
    return getVal(u32, ctx, key_id);
}

/// Ports `gguf_get_val_i32` (gguf.cpp:1119 @c1d0e7a00).
export fn gguf_get_val_i32(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) i32 {
    return getVal(i32, ctx, key_id);
}

/// Ports `gguf_get_val_f32` (gguf.cpp:1125 @c1d0e7a00).
export fn gguf_get_val_f32(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) f32 {
    return getVal(f32, ctx, key_id);
}

/// Ports `gguf_get_val_u64` (gguf.cpp:1131 @c1d0e7a00).
export fn gguf_get_val_u64(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) u64 {
    return getVal(u64, ctx, key_id);
}

/// Ports `gguf_get_val_i64` (gguf.cpp:1137 @c1d0e7a00).
export fn gguf_get_val_i64(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) i64 {
    return getVal(i64, ctx, key_id);
}

/// Ports `gguf_get_val_f64` (gguf.cpp:1143 @c1d0e7a00).
export fn gguf_get_val_f64(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) f64 {
    return getVal(f64, ctx, key_id);
}

/// Ports `gguf_get_val_bool` (gguf.cpp:1149 @c1d0e7a00).
export fn gguf_get_val_bool(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) bool {
    return getVal(bool, ctx, key_id);
}

/// Ports `gguf_get_val_str` (gguf.cpp:1155 @c1d0e7a00).
///
/// Return: the value, borrowed from the context.
export fn gguf_get_val_str(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) [*:0]const u8 {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];
    impl.assert(kv.getNe() == 1, "ctx->kv[key_id].get_ne() == 1");
    return kv.getValStr(0).ptr;
}

/// Ports `gguf_get_val_data` (gguf.cpp:1161 @c1d0e7a00).
///
/// Return: the raw bytes of a single non-string value, borrowed.
export fn gguf_get_val_data(ctx: ?*const c.struct_gguf_context, key_id: i64) callconv(.c) ?*const anyopaque {
    impl.assert(key_id >= 0 and key_id < gguf_get_n_kv(ctx), "key_id >= 0 && key_id < gguf_get_n_kv(ctx)");
    const kv = &ctxOfConst(ctx).kv.items[@intCast(key_id)];
    impl.assert(kv.getNe() == 1, "ctx->kv[key_id].get_ne() == 1");
    impl.assert(kv.getType() != c.GGUF_TYPE_STRING, "ctx->kv[key_id].get_type() != GGUF_TYPE_STRING");
    return @ptrCast(kv.data.items.ptr);
}

/// Ports `gguf_get_n_tensors` (gguf.cpp:1168 @c1d0e7a00).
export fn gguf_get_n_tensors(ctx: ?*const c.struct_gguf_context) callconv(.c) i64 {
    return @intCast(ctxOfConst(ctx).info.items.len);
}

/// Ports `gguf_find_tensor` (gguf.cpp:1172 @c1d0e7a00).
///
/// Return: the tensor's index, or -1 if absent.
export fn gguf_find_tensor(ctx: ?*const c.struct_gguf_context, name: [*:0]const u8) callconv(.c) i64 {
    var tensor_id: i64 = -1;

    const n_tensors = gguf_get_n_tensors(ctx);

    var i: i64 = 0;
    while (i < n_tensors) : (i += 1) {
        if (std.mem.orderZ(u8, name, gguf_get_tensor_name(ctx, i)) == .eq) {
            tensor_id = i;
            break;
        }
    }

    return tensor_id;
}

/// Ports `gguf_get_tensor_offset` (gguf.cpp:1188 @c1d0e7a00).
export fn gguf_get_tensor_offset(ctx: ?*const c.struct_gguf_context, tensor_id: i64) callconv(.c) usize {
    impl.assert(tensor_id >= 0 and tensor_id < gguf_get_n_tensors(ctx), "tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx)");
    return @intCast(ctxOfConst(ctx).info.items[@intCast(tensor_id)].offset);
}

/// Ports `gguf_get_tensor_name` (gguf.cpp:1193 @c1d0e7a00).
///
/// Return: the name, borrowed from the context.
export fn gguf_get_tensor_name(ctx: ?*const c.struct_gguf_context, tensor_id: i64) callconv(.c) [*:0]const u8 {
    impl.assert(tensor_id >= 0 and tensor_id < gguf_get_n_tensors(ctx), "tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx)");
    return @ptrCast(&ctxOfConst(ctx).info.items[@intCast(tensor_id)].t.name);
}

/// Ports `gguf_get_tensor_ne` (gguf.cpp:1198 @c1d0e7a00).
///
/// Return: `GGML_MAX_DIMS` extents, borrowed; entries past the tensor's rank
/// are 1.
export fn gguf_get_tensor_ne(ctx: ?*const c.struct_gguf_context, tensor_id: i64) callconv(.c) [*]const i64 {
    impl.assert(tensor_id >= 0 and tensor_id < gguf_get_n_tensors(ctx), "tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx)");
    return &ctxOfConst(ctx).info.items[@intCast(tensor_id)].t.ne;
}

/// Ports `gguf_get_tensor_type` (gguf.cpp:1203 @c1d0e7a00).
export fn gguf_get_tensor_type(ctx: ?*const c.struct_gguf_context, tensor_id: i64) callconv(.c) c_uint {
    impl.assert(tensor_id >= 0 and tensor_id < gguf_get_n_tensors(ctx), "tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx)");
    return ctxOfConst(ctx).info.items[@intCast(tensor_id)].t.type;
}

/// Ports `gguf_get_tensor_size` (gguf.cpp:1208 @c1d0e7a00).
export fn gguf_get_tensor_size(ctx: ?*const c.struct_gguf_context, tensor_id: i64) callconv(.c) usize {
    impl.assert(tensor_id >= 0 and tensor_id < gguf_get_n_tensors(ctx), "tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx)");
    return c.ggml_nbytes(&ctxOfConst(ctx).info.items[@intCast(tensor_id)].t);
}

// -----------------------------------------------------------------------------
// Mutation

/// Ports `gguf_remove_key` (gguf.cpp:1213 @c1d0e7a00).
///
/// Return: the index the key held before removal, or -1 if it was absent.
/// Later keys shift down, which is what `std::vector::erase` does.
export fn gguf_remove_key(ctx: ?*c.struct_gguf_context, key: [*:0]const u8) callconv(.c) i64 {
    const key_id = gguf_find_key(ctx, key);
    if (key_id >= 0) {
        var removed = ctxOf(ctx).kv.orderedRemove(@intCast(key_id));
        // `erase` runs the element's destructor; `orderedRemove` hands it back.
        removed.deinit();
    }
    return key_id;
}

/// Ports `gguf_check_reserved_keys` (gguf.cpp:1222 @c1d0e7a00).
///
/// Only `general.alignment` is reserved: it must be a `u32` power of two,
/// because `gguf_init_from_reader` pads the data section by it.
fn checkReservedKeys(comptime T: type, key: [*:0]const u8, val: T) void {
    if (std.mem.orderZ(u8, key, c.GGUF_KEY_GENERAL_ALIGNMENT) == .eq) {
        if (T == u32) {
            impl.assert(
                val > 0 and (val & (val - 1)) == 0,
                c.GGUF_KEY_GENERAL_ALIGNMENT ++ " must be power of 2",
            );
        } else {
            impl.abort(c.GGUF_KEY_GENERAL_ALIGNMENT ++ " must be type u32");
        }
    }
}

/// Shared body of the eleven scalar `gguf_set_val_*` entry points, beginning at
/// `gguf_set_val_u8` (gguf.cpp:1233 @c1d0e7a00). Each removes any existing key
/// of that name and appends the new pair at the back, so a set is also a
/// move-to-end.
inline fn setVal(comptime T: type, ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: T) void {
    checkReservedKeys(T, key, val);
    _ = gguf_remove_key(ctx, key);
    ctxOf(ctx).kv.append(allocator, Kv.initScalar(std.mem.span(key), T, val)) catch oom();
}

/// Ports `gguf_set_val_u8` (gguf.cpp:1233 @c1d0e7a00).
export fn gguf_set_val_u8(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: u8) callconv(.c) void {
    setVal(u8, ctx, key, val);
}

/// Ports `gguf_set_val_i8` (gguf.cpp:1239 @c1d0e7a00).
export fn gguf_set_val_i8(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: i8) callconv(.c) void {
    setVal(i8, ctx, key, val);
}

/// Ports `gguf_set_val_u16` (gguf.cpp:1245 @c1d0e7a00).
export fn gguf_set_val_u16(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: u16) callconv(.c) void {
    setVal(u16, ctx, key, val);
}

/// Ports `gguf_set_val_i16` (gguf.cpp:1251 @c1d0e7a00).
export fn gguf_set_val_i16(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: i16) callconv(.c) void {
    setVal(i16, ctx, key, val);
}

/// Ports `gguf_set_val_u32` (gguf.cpp:1257 @c1d0e7a00).
export fn gguf_set_val_u32(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: u32) callconv(.c) void {
    setVal(u32, ctx, key, val);
}

/// Ports `gguf_set_val_i32` (gguf.cpp:1263 @c1d0e7a00).
export fn gguf_set_val_i32(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: i32) callconv(.c) void {
    setVal(i32, ctx, key, val);
}

/// Ports `gguf_set_val_f32` (gguf.cpp:1269 @c1d0e7a00).
export fn gguf_set_val_f32(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: f32) callconv(.c) void {
    setVal(f32, ctx, key, val);
}

/// Ports `gguf_set_val_u64` (gguf.cpp:1275 @c1d0e7a00).
export fn gguf_set_val_u64(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: u64) callconv(.c) void {
    setVal(u64, ctx, key, val);
}

/// Ports `gguf_set_val_i64` (gguf.cpp:1281 @c1d0e7a00).
export fn gguf_set_val_i64(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: i64) callconv(.c) void {
    setVal(i64, ctx, key, val);
}

/// Ports `gguf_set_val_f64` (gguf.cpp:1287 @c1d0e7a00).
export fn gguf_set_val_f64(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: f64) callconv(.c) void {
    setVal(f64, ctx, key, val);
}

/// Ports `gguf_set_val_bool` (gguf.cpp:1293 @c1d0e7a00).
export fn gguf_set_val_bool(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: bool) callconv(.c) void {
    setVal(bool, ctx, key, val);
}

/// Ports `gguf_set_val_str` (gguf.cpp:1299 @c1d0e7a00).
///
/// `val` is copied; the caller keeps ownership of what it passed.
export fn gguf_set_val_str(ctx: ?*c.struct_gguf_context, key: [*:0]const u8, val: [*:0]const u8) callconv(.c) void {
    checkReservedKeys([*:0]const u8, key, val);
    _ = gguf_remove_key(ctx, key);
    ctxOf(ctx).kv.append(allocator, Kv.initStr(std.mem.span(key), std.mem.span(val))) catch oom();
}

/// Ports `gguf_set_arr_data` (gguf.cpp:1305 @c1d0e7a00).
///
/// Parameters:
/// - `key`: overwritten if present.
/// - `t`: element type; `GGUF_TYPE_STRING` is not valid here.
/// - `data`: `n * gguf_type_size(t)` bytes, copied.
/// - `n`: element count.
export fn gguf_set_arr_data(
    ctx: ?*c.struct_gguf_context,
    key: [*:0]const u8,
    t: c_uint,
    data: ?*const anyopaque,
    n: usize,
) callconv(.c) void {
    checkReservedKeys(?*const anyopaque, key, data);
    _ = gguf_remove_key(ctx, key);

    const nbytes = n * typeSize(t);
    // The C++ builds a `std::vector<int8_t>`, hands it to the `vector<T>`
    // constructor as a byte array, and then re-labels the element type with
    // `cast`. Same two steps here.
    var tmp: std.ArrayList(u8) = .empty;
    defer tmp.deinit(allocator);
    tmp.resize(allocator, nbytes) catch oom();
    if (tmp.items.len != 0) {
        @memcpy(tmp.items, @as([*]const u8, @ptrCast(data.?))[0..nbytes]);
    }
    ctxOf(ctx).kv.append(allocator, Kv.initArray(std.mem.span(key), u8, tmp.items)) catch oom();
    ctxOf(ctx).kv.items[ctxOf(ctx).kv.items.len - 1].cast(t);
}

/// Ports `gguf_set_arr_str` (gguf.cpp:1318 @c1d0e7a00).
///
/// Parameters:
/// - `key`: overwritten if present.
/// - `data`: `n` NUL-terminated strings, each copied.
/// - `n`: element count.
export fn gguf_set_arr_str(
    ctx: ?*c.struct_gguf_context,
    key: [*:0]const u8,
    data: [*]const [*:0]const u8,
    n: usize,
) callconv(.c) void {
    checkReservedKeys([*]const [*:0]const u8, key, data);
    _ = gguf_remove_key(ctx, key);

    var tmp: std.ArrayList(Str) = .empty;
    tmp.resize(allocator, n) catch oom();
    for (tmp.items, 0..) |*slot, i| slot.* = dupeStr(std.mem.span(data[i]));
    ctxOf(ctx).kv.append(allocator, Kv.initStrArrayOwned(std.mem.span(key), tmp)) catch oom();
}

/// Ports `gguf_set_kv` (gguf.cpp:1330 @c1d0e7a00).
///
/// Copies every key from `src` into `ctx`, overwriting same-named keys.
export fn gguf_set_kv(ctx: ?*c.struct_gguf_context, src: ?*const c.struct_gguf_context) callconv(.c) void {
    const n_kv = gguf_get_n_kv(src);
    var i: i64 = 0;
    while (i < n_kv) : (i += 1) {
        const kv = &ctxOfConst(src).kv.items[@intCast(i)];

        if (!kv.is_array) {
            switch (kv.getType()) {
                c.GGUF_TYPE_UINT8 => gguf_set_val_u8(ctx, kv.getKey().ptr, kv.getVal(u8, 0)),
                c.GGUF_TYPE_INT8 => gguf_set_val_i8(ctx, kv.getKey().ptr, kv.getVal(i8, 0)),
                c.GGUF_TYPE_UINT16 => gguf_set_val_u16(ctx, kv.getKey().ptr, kv.getVal(u16, 0)),
                c.GGUF_TYPE_INT16 => gguf_set_val_i16(ctx, kv.getKey().ptr, kv.getVal(i16, 0)),
                c.GGUF_TYPE_UINT32 => gguf_set_val_u32(ctx, kv.getKey().ptr, kv.getVal(u32, 0)),
                c.GGUF_TYPE_INT32 => gguf_set_val_i32(ctx, kv.getKey().ptr, kv.getVal(i32, 0)),
                c.GGUF_TYPE_FLOAT32 => gguf_set_val_f32(ctx, kv.getKey().ptr, kv.getVal(f32, 0)),
                c.GGUF_TYPE_UINT64 => gguf_set_val_u64(ctx, kv.getKey().ptr, kv.getVal(u64, 0)),
                c.GGUF_TYPE_INT64 => gguf_set_val_i64(ctx, kv.getKey().ptr, kv.getVal(i64, 0)),
                c.GGUF_TYPE_FLOAT64 => gguf_set_val_f64(ctx, kv.getKey().ptr, kv.getVal(f64, 0)),
                c.GGUF_TYPE_BOOL => gguf_set_val_bool(ctx, kv.getKey().ptr, kv.getVal(bool, 0)),
                c.GGUF_TYPE_STRING => gguf_set_val_str(ctx, kv.getKey().ptr, kv.getValStr(0).ptr),
                else => impl.abort("invalid type"),
            }
            continue;
        }

        const ne = kv.getNe();

        switch (kv.getType()) {
            c.GGUF_TYPE_UINT8,
            c.GGUF_TYPE_INT8,
            c.GGUF_TYPE_UINT16,
            c.GGUF_TYPE_INT16,
            c.GGUF_TYPE_UINT32,
            c.GGUF_TYPE_INT32,
            c.GGUF_TYPE_FLOAT32,
            c.GGUF_TYPE_UINT64,
            c.GGUF_TYPE_INT64,
            c.GGUF_TYPE_FLOAT64,
            c.GGUF_TYPE_BOOL,
            => gguf_set_arr_data(ctx, kv.getKey().ptr, kv.getType(), @ptrCast(kv.data.items.ptr), ne),
            c.GGUF_TYPE_STRING => {
                var tmp: std.ArrayList([*:0]const u8) = .empty;
                defer tmp.deinit(allocator);
                tmp.resize(allocator, ne) catch oom();
                for (tmp.items, 0..) |*slot, j| slot.* = kv.data_string.items[j].ptr;
                gguf_set_arr_str(ctx, kv.getKey().ptr, tmp.items.ptr, ne);
            },
            else => impl.abort("invalid type"),
        }
    }
}

/// Ports `gguf_add_tensor` (gguf.cpp:1384 @c1d0e7a00).
///
/// The tensor struct is copied by value; only its shape, type, name and data
/// pointer are used. Aborts on a duplicate name, as the C++ does.
export fn gguf_add_tensor(ctx: ?*c.struct_gguf_context, tensor: ?*const c.ggml_tensor) callconv(.c) void {
    impl.assert(tensor != null, "tensor");
    if (gguf_find_tensor(@ptrCast(ctx), @ptrCast(&tensor.?.name)) != -1) {
        impl.abort("duplicate tensor name");
    }

    const self = ctxOf(ctx);
    var ti: TensorInfo = .{ .t = tensor.?.*, .offset = 0 };
    ti.offset = if (self.info.items.len == 0) 0 else blk: {
        const back = &self.info.items[self.info.items.len - 1];
        break :blk back.offset + impl.pad(c.ggml_nbytes(&back.t), self.alignment);
    };
    self.info.append(allocator, ti) catch oom();
}

/// Ports `gguf_set_tensor_type` (gguf.cpp:1399 @c1d0e7a00).
///
/// Recomputes this tensor's strides and then every later tensor's offset, so
/// the data section stays one contiguous block.
export fn gguf_set_tensor_type(ctx: ?*c.struct_gguf_context, name: [*:0]const u8, t: c_uint) callconv(.c) void {
    const tensor_id = gguf_find_tensor(@ptrCast(ctx), name);
    if (tensor_id < 0) {
        impl.abort("tensor not found");
    }
    const self = ctxOf(ctx);
    const tensor = &self.info.items[@intCast(tensor_id)].t;
    const type_size = c.ggml_type_size(t);
    const blck_size = c.ggml_blck_size(t);

    tensor.type = t;
    impl.assert(@rem(tensor.ne[0], blck_size) == 0, "tensor row size not divisible by block size of new type");

    tensor.nb[0] = type_size;
    tensor.nb[1] = tensor.nb[0] * @as(usize, @intCast(@divTrunc(tensor.ne[0], blck_size)));
    for (2..c.GGML_MAX_DIMS) |i| {
        tensor.nb[i] = tensor.nb[i - 1] * @as(usize, @intCast(tensor.ne[i - 1]));
    }

    // update offsets
    const n_tensors = gguf_get_n_tensors(@ptrCast(ctx));
    var i: i64 = tensor_id + 1;
    while (i < n_tensors) : (i += 1) {
        const prev = &self.info.items[@intCast(i - 1)];
        self.info.items[@intCast(i)].offset = prev.offset + impl.pad(c.ggml_nbytes(&prev.t), self.alignment);
    }
}

/// Ports `gguf_set_tensor_data` (gguf.cpp:1424 @c1d0e7a00).
///
/// Records where the tensor's bytes live; the context does not take ownership
/// and does not copy.
export fn gguf_set_tensor_data(ctx: ?*c.struct_gguf_context, name: [*:0]const u8, data: ?*const anyopaque) callconv(.c) void {
    const tensor_id = gguf_find_tensor(@ptrCast(ctx), name);
    if (tensor_id < 0) {
        impl.abort("tensor not found");
    }

    // The C++ double-casts through `uintptr_t` to shed the `const`.
    ctxOf(ctx).info.items[@intCast(tensor_id)].t.data = @constCast(data);
}

// -----------------------------------------------------------------------------
// Writing
//
// Ports `struct gguf_writer_base` (gguf.cpp:1433 @c1d0e7a00) and its two
// `final` subclasses. The C++ declares three pure-virtual methods and comments
// them "we bet on devirtualization" just above the first, `write`
// (gguf.cpp:1439 @c1d0e7a00). `gguf_write_out` (gguf.cpp:1622 @c1d0e7a00) is
// already a template over the writer type, so every call is monomorphic at the
// one place it matters. `Writer(Impl)` settles that bet at compile time:
// `Impl` supplies the three, `Writer` supplies everything built on them.

/// The failure the file writer can report. The C++ raises it as a
/// `std::runtime_error` carrying a message that `gguf_write_to_file_ptr` logs;
/// the message is built here at the throw site instead, so the error value
/// stays a plain tag.
const WriteError = error{WriteFailed};

fn Writer(comptime Impl: type) type {
    return struct {
        const Self = @This();

        impl_: Impl,
        /// Ports `gguf_writer_base`'s `written_bytes` (gguf.cpp:1434 @c1d0e7a00).
        written_bytes: usize = 0,

        /// Ports `gguf_writer_base`'s pure-virtual `write` (gguf.cpp:1439
        /// @c1d0e7a00), the `int8_t` overload, dispatched to the concrete
        /// writer.
        fn writeByte(self: *Self, val: u8) WriteError!void {
            try self.impl_.writeByte(val);
            self.written_bytes += 1;
        }

        /// Ports `gguf_writer_base`'s pure-virtual `write` (gguf.cpp:1440
        /// @c1d0e7a00), the `std::vector<int8_t>` overload, dispatched to the
        /// concrete writer.
        fn writeBytes(self: *Self, val: []const u8) WriteError!void {
            try self.impl_.writeBytes(val);
            self.written_bytes += val.len;
        }

        /// Ports `gguf_writer_base`'s templated `write` (gguf.cpp:1444 @c1d0e7a00).
        ///
        /// Byte at a time, as the C++ does. That matters for the file writer:
        /// the C++ throws on the first `fputc` that disagrees, so a short write
        /// is detected mid-value rather than after it.
        fn write(self: *Self, val: anytype) WriteError!void {
            const bytes = std.mem.asBytes(&val);
            for (bytes) |b| try self.writeByte(b);
        }

        /// Ports `gguf_writer_base`'s `bool` `write` (gguf.cpp:1450 @c1d0e7a00).
        fn writeBool(self: *Self, val: bool) WriteError!void {
            const val8: u8 = if (val) 1 else 0;
            try self.writeByte(val8);
        }

        /// Ports `gguf_writer_base`'s `std::string` `write`
        /// (gguf.cpp:1455 @c1d0e7a00): a `u64` length then the bytes, with no
        /// terminator.
        fn writeStr(self: *Self, val: []const u8) WriteError!void {
            {
                const n: u64 = val.len;
                try self.write(n);
            }
            for (val) |b| try self.writeByte(b);
        }

        /// Ports `gguf_writer_base`'s two enum overloads of `write`, `write`
        /// (gguf.cpp:1469, 1473 @c1d0e7a00). Both widen to `int32_t`.
        fn writeEnum(self: *Self, val: c_uint) WriteError!void {
            try self.write(@as(i32, @bitCast(val)));
        }

        /// Ports `gguf_writer_base`'s `gguf_kv` overload of `write`
        /// (gguf.cpp:1477 @c1d0e7a00).
        fn writeKv(self: *Self, kv: *const Kv) WriteError!void {
            const ne: u64 = kv.getNe();

            try self.writeStr(kv.getKey());

            if (kv.is_array) {
                try self.writeEnum(c.GGUF_TYPE_ARRAY);
                try self.writeEnum(kv.getType());
                try self.write(ne);
            } else {
                try self.writeEnum(kv.getType());
            }

            switch (kv.getType()) {
                c.GGUF_TYPE_UINT8,
                c.GGUF_TYPE_INT8,
                c.GGUF_TYPE_UINT16,
                c.GGUF_TYPE_INT16,
                c.GGUF_TYPE_UINT32,
                c.GGUF_TYPE_INT32,
                c.GGUF_TYPE_FLOAT32,
                c.GGUF_TYPE_UINT64,
                c.GGUF_TYPE_INT64,
                c.GGUF_TYPE_FLOAT64,
                => try self.writeBytes(kv.data.items),
                c.GGUF_TYPE_BOOL => {
                    for (0..ne) |i| try self.writeBool(kv.getVal(bool, i));
                },
                c.GGUF_TYPE_STRING => {
                    for (0..ne) |i| try self.writeStr(kv.getValStr(i));
                },
                else => impl.abort("invalid type"),
            }
        }

        /// Ports `gguf_writer_base`'s `write_tensor_meta`
        /// (gguf.cpp:1518 @c1d0e7a00).
        ///
        /// Writes `n_dims` as `ggml_n_dims` reports it — the trailing extents
        /// that are 1 are not written, and the reader fills them back in.
        fn writeTensorMeta(self: *Self, info: *const TensorInfo) WriteError!void {
            const name: [*:0]const u8 = @ptrCast(&info.t.name);
            try self.writeStr(std.mem.span(name));

            const n_dims: u32 = @intCast(c.ggml_n_dims(&info.t));
            try self.write(n_dims);

            for (0..n_dims) |j| {
                try self.write(info.t.ne[j]);
            }
            try self.writeEnum(info.t.type);
            try self.write(info.offset);
        }

        /// Ports `gguf_writer_base`'s `pad` (gguf.cpp:1531 @c1d0e7a00).
        fn pad(self: *Self, alignment: usize) WriteError!void {
            while (self.written_bytes % alignment != 0) {
                const zero: u8 = 0;
                try self.writeByte(zero);
            }
        }

        /// Ports the `write_tensor_data`, `write_tensor_data` overrides of the
        /// buffer and file writers (gguf.cpp:1557, 1602 @c1d0e7a00).
        fn writeTensorData(self: *Self, info: *const TensorInfo, offset_data: usize, alignment: usize) WriteError!void {
            try self.impl_.writeTensorData(self, info, offset_data, alignment);
            try self.pad(alignment);
        }
    };
}

/// Ports `struct gguf_writer_buf` (gguf.cpp:1540 @c1d0e7a00).
///
/// Cannot fail: the C++'s `push_back` and `insert` only throw `std::bad_alloc`,
/// which this port turns into an abort. Its methods still carry `WriteError` so
/// the generic above has one signature to call.
const BufWriter = struct {
    buf: *std.ArrayList(u8),

    /// Ports `gguf_writer_buf`'s `int8_t` `write` (gguf.cpp:1547 @c1d0e7a00).
    fn writeByte(self: *BufWriter, val: u8) WriteError!void {
        self.buf.append(allocator, val) catch oom();
    }

    /// Ports `gguf_writer_buf`'s `std::vector<int8_t>` `write`
    /// (gguf.cpp:1552 @c1d0e7a00).
    fn writeBytes(self: *BufWriter, val: []const u8) WriteError!void {
        self.buf.appendSlice(allocator, val) catch oom();
    }

    /// Ports `gguf_writer_buf`'s `write_tensor_data` (gguf.cpp:1557 @c1d0e7a00).
    ///
    /// The `pad` the C++ does at the end lives in the generic caller, so it is
    /// not repeated here.
    fn writeTensorData(
        self: *BufWriter,
        w: *Writer(BufWriter),
        info: *const TensorInfo,
        offset_data: usize,
        alignment: usize,
    ) WriteError!void {
        _ = alignment;
        impl.assert(self.buf.items.len - offset_data == info.offset, "buf.size() - offset_data == info.offset");

        impl.assert(c.ggml_is_contiguous(&info.t), "ggml_is_contiguous(&info.t)");
        const offset = self.buf.items.len;
        const nbytes = c.ggml_nbytes(&info.t);

        self.buf.resize(allocator, offset + nbytes) catch oom();
        if (info.t.buffer != null) {
            c.ggml_backend_tensor_get(&info.t, @ptrCast(self.buf.items[offset..].ptr), 0, nbytes);
        } else {
            impl.assert(info.t.data != null, "info.t.data");
            @memcpy(self.buf.items[offset..][0..nbytes], @as([*]const u8, @ptrCast(info.t.data))[0..nbytes]);
        }
        w.written_bytes += nbytes;
    }
};

/// Ports `struct gguf_writer_file` (gguf.cpp:1578 @c1d0e7a00).
const FileWriter = struct {
    file: *c.FILE,

    /// Ports `gguf_writer_file`'s `int8_t` `write` (gguf.cpp:1585 @c1d0e7a00).
    ///
    /// The C++ throws `std::runtime_error` when `fputc` does not echo the byte
    /// back; the log line that `gguf_write_to_file_ptr` would print from
    /// `ex.what()` is emitted here instead.
    fn writeByte(self: *FileWriter, val: u8) WriteError!void {
        const real_val: c_int = val;
        const ret = Libc.fputc(real_val, self.file);
        if (ret != real_val) {
            impl.logError(
                "%s: failed to write GGUF data: unexpected fputc result '%d' instead of '%d'\n",
                .{ "gguf_write_to_file_ptr", ret, real_val },
            );
            return error.WriteFailed;
        }
    }

    /// Ports `gguf_writer_file`'s `std::vector<int8_t>` `write`
    /// (gguf.cpp:1594 @c1d0e7a00).
    fn writeBytes(self: *FileWriter, val: []const u8) WriteError!void {
        if (val.len == 0) return;
        const ret = Libc.fwrite(@ptrCast(val.ptr), 1, val.len, self.file);
        if (ret != val.len) {
            impl.logError(
                "%s: failed to write GGUF data: unexpected fwrite number of bytes written, '%zu' instead of '%zu'\n",
                .{ "gguf_write_to_file_ptr", ret, val.len },
            );
            return error.WriteFailed;
        }
    }

    /// Ports `gguf_writer_file`'s `write_tensor_data` (gguf.cpp:1602 @c1d0e7a00).
    fn writeTensorData(
        self: *FileWriter,
        w: *Writer(FileWriter),
        info: *const TensorInfo,
        offset_data: usize,
        alignment: usize,
    ) WriteError!void {
        _ = self;
        _ = alignment;
        impl.assert(w.written_bytes - offset_data == info.offset, "written_bytes - offset_data == info.offset");

        impl.assert(c.ggml_is_contiguous(&info.t), "ggml_is_contiguous(&info.t)");
        const nbytes = c.ggml_nbytes(&info.t);

        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);
        buf.resize(allocator, nbytes) catch oom();
        if (info.t.buffer != null) {
            c.ggml_backend_tensor_get(&info.t, @ptrCast(buf.items.ptr), 0, nbytes);
        } else {
            impl.assert(info.t.data != null, "info.t.data");
            @memcpy(buf.items, @as([*]const u8, @ptrCast(info.t.data))[0..nbytes]);
        }
        try w.writeBytes(buf.items);
    }
};

/// Ports `gguf_write_out` (gguf.cpp:1622 @c1d0e7a00).
///
/// Parameters:
/// - `ctx`: the context to serialise.
/// - `gw`: the writer, buffer- or file-backed.
/// - `only_meta`: stop after the header, key/value pairs and tensor table.
fn writeOut(ctx: *const Context, gw: anytype, only_meta: bool) WriteError!void {
    const n_kv = ctx.kv.items.len;
    const n_tensors = ctx.info.items.len;

    // write header
    try gw.write(@as(u8, c.GGUF_MAGIC[0]));
    try gw.write(@as(u8, c.GGUF_MAGIC[1]));
    try gw.write(@as(u8, c.GGUF_MAGIC[2]));
    try gw.write(@as(u8, c.GGUF_MAGIC[3]));
    try gw.write(ctx.version);
    try gw.write(@as(i64, @intCast(n_tensors)));
    try gw.write(@as(i64, @intCast(n_kv)));

    // write key-value pairs
    for (ctx.kv.items) |*kv| {
        try gw.writeKv(kv);
    }

    // write tensor info
    for (ctx.info.items) |*info| {
        try gw.writeTensorMeta(info);
    }

    // we require the data section to be aligned
    try gw.pad(ctx.alignment);

    if (only_meta) return;

    const offset_data = gw.written_bytes;

    // write tensor data
    for (ctx.info.items) |*info| {
        try gw.writeTensorData(info, offset_data, ctx.alignment);
    }
}

/// Ports `gguf_write_to_buf` (gguf.cpp:1660 @c1d0e7a00).
///
/// Not exported: the C++ declares it in `ggml-impl.h` inside `#ifdef
/// __cplusplus` with a `std::vector<int8_t> &` parameter, so its symbol is
/// mangled and no C caller could reach it. See the file header.
fn writeToBuf(ctx: *const Context, buf: *std.ArrayList(u8), only_meta: bool) void {
    const backing: BufWriter = .{ .buf = buf };
    var gw: Writer(BufWriter) = .{ .impl_ = backing };
    // `BufWriter` never returns `WriteFailed`; allocation failure aborts.
    writeOut(ctx, &gw, only_meta) catch unreachable;
}

/// Ports `gguf_write_to_file_ptr` (gguf.cpp:1665 @c1d0e7a00).
///
/// Return: true on success. The C++'s `catch (const std::runtime_error &)` is
/// this `catch`; the message it logged is emitted at the failing write.
export fn gguf_write_to_file_ptr(ctx: ?*const c.struct_gguf_context, file: ?*c.FILE, only_meta: bool) callconv(.c) bool {
    impl.assert(file != null, "file");

    const backing: FileWriter = .{ .file = file.? };
    var gw: Writer(FileWriter) = .{ .impl_ = backing };
    writeOut(ctxOfConst(ctx), &gw, only_meta) catch return false;
    return true;
}

/// Ports `gguf_write_to_file` (gguf.cpp:1678 @c1d0e7a00).
///
/// Return: true on success. The file is closed either way.
export fn gguf_write_to_file(ctx: ?*const c.struct_gguf_context, fname: [*:0]const u8, only_meta: bool) callconv(.c) bool {
    const file = c.ggml_fopen(fname, "wb") orelse {
        impl.logError(
            "%s: failed to open file '%s' for writing GGUF data\n",
            .{ "gguf_write_to_file", fname },
        );
        return false;
    };

    const success = gguf_write_to_file_ptr(ctx, file, only_meta);
    if (!success) {
        impl.logError("%s: failed to write GGUF data into '%s'\n", .{ "gguf_write_to_file", fname });
    }

    _ = Libc.fclose(file);
    return success;
}

/// Ports `gguf_get_meta_size` (gguf.cpp:1695 @c1d0e7a00).
///
/// Return: the byte length of the metadata section, padding included. Computed
/// by serialising it and measuring, as the C++ does.
export fn gguf_get_meta_size(ctx: ?*const c.struct_gguf_context) callconv(.c) usize {
    // only return size
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    writeToBuf(ctxOfConst(ctx), &buf, true);
    return buf.items.len;
}

/// Ports `gguf_get_meta_data` (gguf.cpp:1702 @c1d0e7a00).
///
/// Parameters:
/// - `data`: at least `gguf_get_meta_size` bytes, written in full.
export fn gguf_get_meta_data(ctx: ?*const c.struct_gguf_context, data: ?*anyopaque) callconv(.c) void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    writeToBuf(ctxOfConst(ctx), &buf, true);
    @memcpy(@as([*]u8, @ptrCast(data.?))[0..buf.items.len], buf.items);
}

// -----------------------------------------------------------------------------
// Unit Tests
//
// The round-trip tests are the real check: they drive the buffer writer and the
// reader against each other through the public C ABI, so a constructor, an
// accessor, the serialiser and the parser all have to agree for one to pass.
// No model file is needed — `gguf_get_meta_size` and `gguf_get_meta_data`
// produce a complete GGUF whenever there is no tensor data to follow.

test {
    std.testing.refAllDecls(@This());
}

test "type sizes and names match the C's two tables" {
    try std.testing.expectEqual(@as(usize, 1), typeSize(c.GGUF_TYPE_UINT8));
    try std.testing.expectEqual(@as(usize, 1), typeSize(c.GGUF_TYPE_INT8));
    try std.testing.expectEqual(@as(usize, 2), typeSize(c.GGUF_TYPE_UINT16));
    try std.testing.expectEqual(@as(usize, 4), typeSize(c.GGUF_TYPE_FLOAT32));
    try std.testing.expectEqual(@as(usize, 8), typeSize(c.GGUF_TYPE_FLOAT64));
    // The C++ map gives these entries 0 and comments them "undefined".
    try std.testing.expectEqual(@as(usize, 0), typeSize(c.GGUF_TYPE_STRING));
    try std.testing.expectEqual(@as(usize, 0), typeSize(c.GGUF_TYPE_ARRAY));
    // The `it == end` arm.
    try std.testing.expectEqual(@as(usize, 0), typeSize(999));

    try std.testing.expectEqualStrings("u8", std.mem.span(gguf_type_name(c.GGUF_TYPE_UINT8).?));
    try std.testing.expectEqualStrings("f32", std.mem.span(gguf_type_name(c.GGUF_TYPE_FLOAT32).?));
    try std.testing.expectEqualStrings("arr", std.mem.span(gguf_type_name(c.GGUF_TYPE_ARRAY).?));
    try std.testing.expectEqualStrings("f64", std.mem.span(gguf_type_name(c.GGUF_TYPE_FLOAT64).?));
    try std.testing.expect(gguf_type_name(999) == null);
}

/// Serialises `ctx` the way `gguf_get_meta_data` does and hands back the bytes.
/// The caller frees them.
fn metaBytes(ctx: ?*const c.struct_gguf_context) ![]u8 {
    const n = gguf_get_meta_size(ctx);
    const buf = try std.testing.allocator.alloc(u8, n);
    gguf_get_meta_data(ctx, buf.ptr);
    return buf;
}

test "every scalar type round-trips through write and read" {
    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);

    gguf_set_val_u8(ctx, "k.u8", 0xAB);
    gguf_set_val_i8(ctx, "k.i8", -7);
    gguf_set_val_u16(ctx, "k.u16", 0xBEEF);
    gguf_set_val_i16(ctx, "k.i16", -300);
    gguf_set_val_u32(ctx, "k.u32", 0xDEADBEEF);
    gguf_set_val_i32(ctx, "k.i32", -70000);
    gguf_set_val_f32(ctx, "k.f32", 0.15625);
    gguf_set_val_u64(ctx, "k.u64", 0x0123456789ABCDEF);
    gguf_set_val_i64(ctx, "k.i64", -5_000_000_000);
    gguf_set_val_f64(ctx, "k.f64", 2.718281828459045);
    gguf_set_val_bool(ctx, "k.bool.t", true);
    gguf_set_val_bool(ctx, "k.bool.f", false);
    gguf_set_val_str(ctx, "k.str", "hello gguf");

    const bytes = try metaBytes(ctx);
    defer std.testing.allocator.free(bytes);

    const back = gguf_init_from_buffer(bytes.ptr, bytes.len, .{ .no_alloc = true, .ctx = null }) orelse
        return error.ParseFailed;
    defer gguf_free(back);

    try std.testing.expectEqual(gguf_get_n_kv(ctx), gguf_get_n_kv(back));
    try std.testing.expectEqual(@as(u32, c.GGUF_VERSION), gguf_get_version(back));

    try std.testing.expectEqual(@as(u8, 0xAB), gguf_get_val_u8(back, gguf_find_key(back, "k.u8")));
    try std.testing.expectEqual(@as(i8, -7), gguf_get_val_i8(back, gguf_find_key(back, "k.i8")));
    try std.testing.expectEqual(@as(u16, 0xBEEF), gguf_get_val_u16(back, gguf_find_key(back, "k.u16")));
    try std.testing.expectEqual(@as(i16, -300), gguf_get_val_i16(back, gguf_find_key(back, "k.i16")));
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), gguf_get_val_u32(back, gguf_find_key(back, "k.u32")));
    try std.testing.expectEqual(@as(i32, -70000), gguf_get_val_i32(back, gguf_find_key(back, "k.i32")));
    // Compared on bits: the value is exact in binary, so anything else is a
    // serialisation fault rather than rounding.
    try std.testing.expectEqual(
        @as(u32, @bitCast(@as(f32, 0.15625))),
        @as(u32, @bitCast(gguf_get_val_f32(back, gguf_find_key(back, "k.f32")))),
    );
    try std.testing.expectEqual(@as(u64, 0x0123456789ABCDEF), gguf_get_val_u64(back, gguf_find_key(back, "k.u64")));
    try std.testing.expectEqual(@as(i64, -5_000_000_000), gguf_get_val_i64(back, gguf_find_key(back, "k.i64")));
    try std.testing.expectEqual(
        @as(u64, @bitCast(@as(f64, 2.718281828459045))),
        @as(u64, @bitCast(gguf_get_val_f64(back, gguf_find_key(back, "k.f64")))),
    );
    try std.testing.expect(gguf_get_val_bool(back, gguf_find_key(back, "k.bool.t")));
    try std.testing.expect(!gguf_get_val_bool(back, gguf_find_key(back, "k.bool.f")));
    try std.testing.expectEqualStrings(
        "hello gguf",
        std.mem.span(gguf_get_val_str(back, gguf_find_key(back, "k.str"))),
    );
}

test "arrays round-trip, and a bool array stays one byte per element" {
    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);

    const i32s = [_]i32{ 1, -2, 3, -4, 5 };
    gguf_set_arr_data(ctx, "a.i32", c.GGUF_TYPE_INT32, &i32s, i32s.len);

    const f32s = [_]f32{ 0.5, 0.25, 0.125 };
    gguf_set_arr_data(ctx, "a.f32", c.GGUF_TYPE_FLOAT32, &f32s, f32s.len);

    const bools = [_]i8{ 1, 0, 1, 1 };
    gguf_set_arr_data(ctx, "a.bool", c.GGUF_TYPE_BOOL, &bools, bools.len);

    const strs = [_][*:0]const u8{ "alpha", "beta", "" };
    gguf_set_arr_str(ctx, "a.str", &strs, strs.len);

    const bytes = try metaBytes(ctx);
    defer std.testing.allocator.free(bytes);

    const back = gguf_init_from_buffer(bytes.ptr, bytes.len, .{ .no_alloc = true, .ctx = null }) orelse
        return error.ParseFailed;
    defer gguf_free(back);

    {
        const id = gguf_find_key(back, "a.i32");
        try std.testing.expectEqual(@as(c_uint, c.GGUF_TYPE_ARRAY), gguf_get_kv_type(back, id));
        try std.testing.expectEqual(@as(c_uint, c.GGUF_TYPE_INT32), gguf_get_arr_type(back, id));
        try std.testing.expectEqual(@as(usize, i32s.len), gguf_get_arr_n(back, id));
        const got: [*]const i32 = @ptrCast(@alignCast(gguf_get_arr_data(back, id).?));
        try std.testing.expectEqualSlices(i32, &i32s, got[0..i32s.len]);
    }
    {
        const id = gguf_find_key(back, "a.f32");
        const got: [*]const f32 = @ptrCast(@alignCast(gguf_get_arr_data(back, id).?));
        try std.testing.expectEqualSlices(f32, &f32s, got[0..f32s.len]);
    }
    {
        // The header promises bool arrays are stored as int8 on all platforms.
        const id = gguf_find_key(back, "a.bool");
        try std.testing.expectEqual(@as(c_uint, c.GGUF_TYPE_BOOL), gguf_get_arr_type(back, id));
        try std.testing.expectEqual(@as(usize, bools.len), gguf_get_arr_n(back, id));
        const got: [*]const i8 = @ptrCast(gguf_get_arr_data(back, id).?);
        try std.testing.expectEqualSlices(i8, &bools, got[0..bools.len]);
    }
    {
        const id = gguf_find_key(back, "a.str");
        try std.testing.expectEqual(@as(c_uint, c.GGUF_TYPE_STRING), gguf_get_arr_type(back, id));
        try std.testing.expectEqual(@as(usize, strs.len), gguf_get_arr_n(back, id));
        try std.testing.expectEqualStrings("alpha", std.mem.span(gguf_get_arr_str(back, id, 0)));
        try std.testing.expectEqualStrings("beta", std.mem.span(gguf_get_arr_str(back, id, 1)));
        // The empty string is the one a length-prefixed format can get wrong
        // by writing a terminator it never promised.
        try std.testing.expectEqualStrings("", std.mem.span(gguf_get_arr_str(back, id, 2)));
    }
}

test "tensor metadata round-trips with shape, type and offset" {
    const gctx = c.ggml_init(.{ .mem_size = 16 * 1024 * 1024, .mem_buffer = null, .no_alloc = true });
    defer c.ggml_free(gctx);

    // 5 f32 is 20 bytes, which the default 32-byte alignment has to round up.
    // A shape whose size is already a multiple of the alignment would let an
    // unpadded offset calculation pass -- measured: with 8x4 f32 (128 bytes,
    // pad(128, 32) == 128) dropping the `GGML_PAD` here changes nothing.
    const a = c.ggml_new_tensor_1d(gctx, c.GGML_TYPE_F32, 5);
    _ = c.ggml_set_name(a, "weight.a");
    const b = c.ggml_new_tensor_1d(gctx, c.GGML_TYPE_F16, 16);
    _ = c.ggml_set_name(b, "bias.b");

    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);
    gguf_add_tensor(ctx, a);
    gguf_add_tensor(ctx, b);

    // The second tensor starts after the first, padded to the alignment.
    try std.testing.expectEqual(@as(usize, 0), gguf_get_tensor_offset(ctx, 0));
    try std.testing.expectEqual(@as(usize, 20), c.ggml_nbytes(a));
    try std.testing.expectEqual(@as(usize, 32), gguf_get_tensor_offset(ctx, 1));
    try std.testing.expectEqual(
        impl.pad(c.ggml_nbytes(a), gguf_get_alignment(ctx)),
        gguf_get_tensor_offset(ctx, 1),
    );

    const bytes = try metaBytes(ctx);
    defer std.testing.allocator.free(bytes);

    const back = gguf_init_from_buffer(bytes.ptr, bytes.len, .{ .no_alloc = true, .ctx = null }) orelse
        return error.ParseFailed;
    defer gguf_free(back);

    try std.testing.expectEqual(@as(i64, 2), gguf_get_n_tensors(back));

    const ia = gguf_find_tensor(back, "weight.a");
    try std.testing.expectEqual(@as(i64, 0), ia);
    try std.testing.expectEqual(@as(c_uint, c.GGML_TYPE_F32), gguf_get_tensor_type(back, ia));
    const ne = gguf_get_tensor_ne(back, ia);
    try std.testing.expectEqual(@as(i64, 5), ne[0]);
    // Trailing extents are not written; the reader fills them back in as 1.
    try std.testing.expectEqual(@as(i64, 1), ne[1]);
    try std.testing.expectEqual(@as(i64, 1), ne[2]);
    try std.testing.expectEqual(@as(i64, 1), ne[3]);
    try std.testing.expectEqual(c.ggml_nbytes(a), gguf_get_tensor_size(back, ia));

    const ib = gguf_find_tensor(back, "bias.b");
    try std.testing.expectEqual(@as(c_uint, c.GGML_TYPE_F16), gguf_get_tensor_type(back, ib));
    try std.testing.expectEqual(gguf_get_tensor_offset(ctx, 1), gguf_get_tensor_offset(back, ib));

    try std.testing.expectEqual(@as(i64, -1), gguf_find_tensor(back, "nope"));
}

test "a malformed buffer is rejected rather than parsed" {
    const params: c.gguf_init_params = .{ .no_alloc = true, .ctx = null };

    try std.testing.expect(gguf_init_from_buffer(null, 0, params) == null);

    const empty = [_]u8{};
    try std.testing.expect(gguf_init_from_buffer(&empty, 0, params) == null);

    // Wrong magic, on an otherwise *valid* file. A buffer that is malformed in
    // some later way as well would be rejected even with the magic check
    // disabled, and would prove nothing -- measured: `"XGUF"` followed by
    // zeroes is caught by the version check instead.
    {
        const ctx = gguf_init_empty().?;
        defer gguf_free(ctx);
        gguf_set_val_u32(ctx, "k", 1);

        const good = try metaBytes(ctx);
        defer std.testing.allocator.free(good);
        try std.testing.expect(gguf_init_from_buffer(good.ptr, good.len, params) != null);
        const reparsed = gguf_init_from_buffer(good.ptr, good.len, params).?;
        gguf_free(reparsed);

        good[0] = 'X';
        try std.testing.expect(gguf_init_from_buffer(good.ptr, good.len, params) == null);
    }

    // Right magic, truncated before the header is complete.
    const truncated = "GGUF" ++ [_]u8{ 3, 0, 0, 0 };
    try std.testing.expect(gguf_init_from_buffer(truncated, truncated.len, params) == null);

    // Right magic, version 0.
    const v0 = "GGUF" ++ [_]u8{0} ** 20;
    try std.testing.expect(gguf_init_from_buffer(v0, v0.len, params) == null);
}

test "a set overwrites in place of appending a duplicate" {
    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);

    gguf_set_val_u32(ctx, "a", 1);
    gguf_set_val_u32(ctx, "b", 2);
    gguf_set_val_u32(ctx, "a", 3);

    try std.testing.expectEqual(@as(i64, 2), gguf_get_n_kv(ctx));
    // The C++ erases then pushes to the back, so the rewritten key moves last.
    try std.testing.expectEqualStrings("b", std.mem.span(gguf_get_key(ctx, 0)));
    try std.testing.expectEqualStrings("a", std.mem.span(gguf_get_key(ctx, 1)));
    try std.testing.expectEqual(@as(u32, 3), gguf_get_val_u32(ctx, gguf_find_key(ctx, "a")));

    try std.testing.expectEqual(@as(i64, 0), gguf_remove_key(ctx, "b"));
    try std.testing.expectEqual(@as(i64, 1), gguf_get_n_kv(ctx));
    try std.testing.expectEqual(@as(i64, -1), gguf_remove_key(ctx, "b"));
    try std.testing.expectEqual(@as(i64, -1), gguf_find_key(ctx, "b"));
}

test "removing a key preserves the order of the ones after it" {
    // `std::vector::erase` shifts the tail down; it does not fill the hole with
    // the last element. Four keys with the removal in the middle is the
    // smallest case that can tell the two apart -- measured: with only two
    // keys, a swap-remove passes.
    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);

    gguf_set_val_u32(ctx, "a", 1);
    gguf_set_val_u32(ctx, "b", 2);
    gguf_set_val_u32(ctx, "c", 3);
    gguf_set_val_u32(ctx, "d", 4);

    try std.testing.expectEqual(@as(i64, 1), gguf_remove_key(ctx, "b"));

    try std.testing.expectEqual(@as(i64, 3), gguf_get_n_kv(ctx));
    try std.testing.expectEqualStrings("a", std.mem.span(gguf_get_key(ctx, 0)));
    try std.testing.expectEqualStrings("c", std.mem.span(gguf_get_key(ctx, 1)));
    try std.testing.expectEqualStrings("d", std.mem.span(gguf_get_key(ctx, 2)));

    // The values travel with their keys.
    try std.testing.expectEqual(@as(u32, 3), gguf_get_val_u32(ctx, 1));
    try std.testing.expectEqual(@as(u32, 4), gguf_get_val_u32(ctx, 2));
}

test "gguf_set_kv copies every key from another context" {
    const src = gguf_init_empty().?;
    defer gguf_free(src);
    gguf_set_val_i32(src, "s.i32", -42);
    gguf_set_val_str(src, "s.str", "copied");
    const arr = [_]u16{ 7, 8, 9 };
    gguf_set_arr_data(src, "s.arr", c.GGUF_TYPE_UINT16, &arr, arr.len);
    const strs = [_][*:0]const u8{ "x", "y" };
    gguf_set_arr_str(src, "s.strs", &strs, strs.len);

    const dst = gguf_init_empty().?;
    defer gguf_free(dst);
    gguf_set_kv(dst, src);

    try std.testing.expectEqual(gguf_get_n_kv(src), gguf_get_n_kv(dst));
    try std.testing.expectEqual(@as(i32, -42), gguf_get_val_i32(dst, gguf_find_key(dst, "s.i32")));
    try std.testing.expectEqualStrings("copied", std.mem.span(gguf_get_val_str(dst, gguf_find_key(dst, "s.str"))));

    const id = gguf_find_key(dst, "s.arr");
    try std.testing.expectEqual(@as(usize, arr.len), gguf_get_arr_n(dst, id));
    const got: [*]const u16 = @ptrCast(@alignCast(gguf_get_arr_data(dst, id).?));
    try std.testing.expectEqualSlices(u16, &arr, got[0..arr.len]);

    const sid = gguf_find_key(dst, "s.strs");
    try std.testing.expectEqualStrings("y", std.mem.span(gguf_get_arr_str(dst, sid, 1)));
}

test "general.alignment is honoured and validated" {
    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);

    try std.testing.expectEqual(@as(usize, c.GGUF_DEFAULT_ALIGNMENT), gguf_get_alignment(ctx));

    gguf_set_val_u32(ctx, c.GGUF_KEY_GENERAL_ALIGNMENT, 64);
    const bytes = try metaBytes(ctx);
    defer std.testing.allocator.free(bytes);

    const back = gguf_init_from_buffer(bytes.ptr, bytes.len, .{ .no_alloc = true, .ctx = null }) orelse
        return error.ParseFailed;
    defer gguf_free(back);
    try std.testing.expectEqual(@as(usize, 64), gguf_get_alignment(back));
}

test "the serialised bytes are the ones the format specifies" {
    // Round-trip tests only prove the writer and the reader agree with each
    // other. They pass just as happily if both sides are wrong the same way --
    // measured: writing `2` for a true `bool` instead of `1` round-trips
    // perfectly, because `read(bool &)` compares against zero.
    //
    // So this pins the exact bytes, against the layout documented at the top of
    // `llama.cpp/ggml/include/gguf.h` rather than against our own output.
    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);
    gguf_set_val_bool(ctx, "b", true);

    const bytes = try metaBytes(ctx);
    defer std.testing.allocator.free(bytes);

    const expect =
        "GGUF" ++ // 1. magic
        [_]u8{ 3, 0, 0, 0 } ++ // 2. version, u32 little-endian
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 } ++ // 3. n_tensors, i64
        [_]u8{ 1, 0, 0, 0, 0, 0, 0, 0 } ++ // 4. n_kv, i64
        [_]u8{ 1, 0, 0, 0, 0, 0, 0, 0 } ++ // 5.1 key length, u64
        "b" ++ //                             5.1 key, no terminator
        [_]u8{ 7, 0, 0, 0 } ++ // 5.2 GGUF_TYPE_BOOL, stored as i32
        [_]u8{1} ++ //            5.3b the value, one int8
        [_]u8{0} ** 26; // 7. padded up to the 32-byte default alignment

    try std.testing.expectEqual(@as(usize, 64), expect.len);
    try std.testing.expectEqualSlices(u8, expect, bytes);
}

test "a string keeps its length rather than stopping at an embedded NUL" {
    // `std::string` serialises `length()` bytes, so a key or value holding a
    // NUL survives the round trip whole. Reading it back through the C ABI
    // still yields a `const char *`, which is why `Str` carries a sentinel on
    // top of its length.
    const ctx = gguf_init_empty().?;
    defer gguf_free(ctx);

    var kv = Kv.initStr("embedded", "a\x00b");
    defer kv.deinit();
    try std.testing.expectEqual(@as(usize, 3), kv.getValStr(0).len);
    try std.testing.expectEqual(@as(u8, 0), kv.getValStr(0)[1]);
    try std.testing.expectEqual(@as(u8, 0), kv.getValStr(0).ptr[3]);
}
