//! Model files: reading them, mapping them, and locking them in memory.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-mmap.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_file`, `llama_mmap`, `llama_mlock` and `llama_path_max`, the POSIX
//! arm of each.
//!
//! # What this target compiles
//!
//! Every `#if` here is resolved for macOS on arm64, which is measured, not
//! assumed:
//!
//! - **No direct I/O.** `init_fd` and the unbuffered read path are behind
//!   `__linux__`, so `fd` is always -1, `alignment` always 1, and
//!   `has_direct_io` always false. `read_raw` is therefore always
//!   `read_raw_unsafe`'s buffered arm.
//! - **`_POSIX_MAPPED_FILES` is defined**, so mapping is the `mmap` arm, with
//!   neither `posix_fadvise` nor `MAP_POPULATE`.
//! - **`_POSIX_MEMLOCK_RANGE` is defined -- as -1.** The C++ asks `#ifdef`,
//!   not `#if`, so the `mlock` arm compiles and `SUPPORTED` is true.
//! - **The `mlock` suggestion is never printed.** `TargetConditionals.h`
//!   defines `TARGET_OS_TV` (as 0), and the C++ asks `defined(...)`, so
//!   `suggest` is forced false and `getrlimit` is never called.

const std = @import("std");
const impl = @import("impl.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

extern "c" fn strerror(errnum: c_int) [*:0]const u8;
/// `errno`'s location: `__error` on Darwin, `__errno_location` on glibc.
const errnoPtr = std.c._errno;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(addr: ?*anyopaque, len: usize) c_int;
extern "c" fn posix_madvise(addr: ?*anyopaque, len: usize, advice: c_int) c_int;
extern "c" fn sysconf(name: c_int) c_long;
extern "c" fn mlock(addr: ?*const anyopaque, len: usize) c_int;
extern "c" fn fileno(stream: *c.FILE) c_int;

// <sys/mman.h>, <unistd.h> and <limits.h> on this target.
const PROT_READ: c_int = 1;
const MAP_SHARED: c_int = 1;
const MAP_FAILED: usize = std.math.maxInt(usize);
const POSIX_MADV_RANDOM: c_int = 1;
const POSIX_MADV_WILLNEED: c_int = 3;
const _SC_PAGESIZE: c_int = 29;
const PATH_MAX: usize = 1024;

fn errnoText() []const u8 {
    return std.mem.span(strerror(errnoPtr().*));
}

// -----------------------------------------------------------------------------
// llama_file

/// Ports `llama_file` and its POSIX `impl` (llama-mmap.cpp:68 @c1d0e7a00).
///
/// A buffered `FILE *` and its size. The C++ also carries an unbuffered
/// file descriptor for direct I/O; that path is Linux-only, so here the
/// descriptor would always be -1 and is not carried.
pub const File = struct {
    fp: *c.FILE,
    owns_fp: bool,
    size: usize,
    /// `alignment`: the read alignment, 1 without direct I/O.
    alignment: usize = 1,

    /// Ports `llama_file`'s `impl`'s path constructor (llama-mmap.cpp:183
    /// @c1d0e7a00) and `init_fp` (llama-mmap.cpp:218 @c1d0e7a00): open,
    /// then seek to the end for the size.
    ///
    /// Parameters:
    /// - `fname`: the path.
    /// - `mode`: the `fopen` mode.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the file, or `error.Thrown` when it cannot be opened; free it
    /// with `close`.
    pub fn open(fname: [:0]const u8, mode: [:0]const u8, exc: *Exception) Thrown!File {
        const fp = c.ggml_fopen(fname.ptr, mode.ptr) orelse {
            return exc.throw(.runtime_error, "failed to open {s}: {s}", .{ fname, errnoText() });
        };
        var f: File = .{ .fp = fp, .owns_fp = true, .size = 0 };
        errdefer _ = c.fclose(fp);
        try f.seek(0, c.SEEK_END, exc);
        f.size = try f.tell(exc);
        try f.seek(0, c.SEEK_SET, exc);
        return f;
    }

    /// Ports `llama_file`'s `impl`'s `FILE *` constructor (llama-mmap.cpp:228
    /// @c1d0e7a00): borrowed, measured the same way.
    ///
    /// Parameters:
    /// - `fp`: the open file; borrowed, never closed here.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the file, or `error.Thrown` when it cannot be measured.
    pub fn borrow(fp: *c.FILE, exc: *Exception) Thrown!File {
        var f: File = .{ .fp = fp, .owns_fp = false, .size = 0 };
        try f.seek(0, c.SEEK_END, exc);
        f.size = try f.tell(exc);
        try f.seek(0, c.SEEK_SET, exc);
        return f;
    }

    /// Ports `llama_file`'s `impl`'s destructor (llama-mmap.cpp:376 @c1d0e7a00).
    ///
    /// Return: nothing.
    pub fn close(self: *File) void {
        if (self.owns_fp) _ = c.fclose(self.fp);
    }

    /// Ports `llama_file`'s `impl`'s `tell` (llama-mmap.cpp:235 @c1d0e7a00).
    ///
    /// Return: the position, or `error.Thrown`.
    pub fn tell(self: *const File, exc: *Exception) Thrown!usize {
        const ret = c.ftello(self.fp);
        if (ret == -1) return exc.throw(.runtime_error, "ftell error: {s}", .{errnoText()});
        return @intCast(ret);
    }

    /// Ports `llama_file`'s `impl`'s `seek` (llama-mmap.cpp:252 @c1d0e7a00).
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn seek(self: *const File, offset: usize, whence: c_int, exc: *Exception) Thrown!void {
        if (c.fseeko(self.fp, @bitCast(offset), whence) == -1) {
            return exc.throw(.runtime_error, "seek error: {s}", .{errnoText()});
        }
    }

    /// Ports `llama_file`'s `impl`'s `read_raw_unsafe` (llama-mmap.cpp:264
    /// @c1d0e7a00), its buffered arm: a read is clamped to what is left of
    /// the file, so reading past the end succeeds short.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn readRawUnsafe(self: *const File, ptr: *anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (len == 0) return;
        errnoPtr().* = 0;
        const curr_off = try self.tell(exc);
        const to_read = @min(len, self.size -% curr_off);
        const ret = c.fread(ptr, to_read, 1, self.fp);
        if (c.ferror(self.fp) != 0) return exc.throw(.runtime_error, "read error: {s}", .{errnoText()});
        if (to_read > 0 and ret != 1) return exc.throw(.runtime_error, "unexpectedly reached end of file", .{});
    }

    /// Ports `llama_file`'s `impl`'s `read_raw` (llama-mmap.cpp:343 @c1d0e7a00).
    /// `has_direct_io` is always false here, so this is `read_raw_unsafe`.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn readRaw(self: *const File, ptr: *anyopaque, len: usize, exc: *Exception) Thrown!void {
        return self.readRawUnsafe(ptr, len, exc);
    }

    /// Ports `llama_file`'s `impl`'s `read_u32` (llama-mmap.cpp:351 @c1d0e7a00).
    ///
    /// Return: the value, or `error.Thrown`.
    pub fn readU32(self: *const File, exc: *Exception) Thrown!u32 {
        var v: u32 = undefined;
        try self.readRaw(&v, @sizeOf(u32), exc);
        return v;
    }

    /// Ports `llama_file`'s `impl`'s `write_raw` (llama-mmap.cpp:357 @c1d0e7a00).
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn writeRaw(self: *const File, ptr: *const anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (len == 0) return;
        errnoPtr().* = 0;
        if (c.fwrite(ptr, len, 1, self.fp) != 1) return exc.throw(.runtime_error, "write error: {s}", .{errnoText()});
    }

    /// Ports `llama_file`'s `impl`'s `write_u32` (llama-mmap.cpp:368 @c1d0e7a00).
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn writeU32(self: *const File, val: u32, exc: *Exception) Thrown!void {
        return self.writeRaw(&val, @sizeOf(u32), exc);
    }

    /// Ports `llama_file`'s `file_id` (llama-mmap.cpp:411 @c1d0e7a00).
    ///
    /// Return: the descriptor behind the `FILE *`.
    pub fn fileId(self: *const File) c_int {
        return fileno(self.fp);
    }

    /// Ports `llama_file`'s `impl`'s `has_direct_io` (llama-mmap.cpp:372
    /// @c1d0e7a00): false, since there is no descriptor.
    pub fn hasDirectIo(_: *const File) bool {
        return false;
    }

    /// Return: `llama_file`'s `read_alignment`, always 1 without direct I/O.
    pub fn readAlignment(self: *const File) usize {
        return self.alignment;
    }
};

// -----------------------------------------------------------------------------
// llama_mmap

/// Ports `llama_mmap` and its POSIX `impl` (llama-mmap.cpp:441 @c1d0e7a00).
pub const Mmap = struct {
    addr: *anyopaque,
    size: usize,
    /// `mapped_fragments`: the byte ranges still mapped.
    fragments: std.ArrayList([2]usize) = .empty,

    /// `llama_mmap`'s `SUPPORTED` (llama-mmap.cpp:629 @c1d0e7a00).
    pub const supported = true;

    /// Ports `llama_mmap`'s `impl`'s constructor (llama-mmap.cpp:445 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `file`: the file to map, read-only and shared.
    /// - `prefetch`: how many leading bytes to advise the kernel will be
    ///   needed; 0 for none.
    /// - `numa`: advise random access and do not prefetch.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the mapping, or `error.Thrown`; free it with `deinit`.
    pub fn init(file: *const File, prefetch_: usize, numa: bool, exc: *Exception) Thrown!Mmap {
        var prefetch = prefetch_;
        const size = file.size;
        const fd = file.fileId();
        if (numa) prefetch = 0;
        const addr = mmap(null, file.size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == null or @intFromPtr(addr) == MAP_FAILED) {
            return exc.throw(.runtime_error, "mmap failed: {s}", .{errnoText()});
        }
        if (prefetch > 0) {
            if (posix_madvise(addr, @min(file.size, prefetch), POSIX_MADV_WILLNEED) != 0) {
                impl.logWarn("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n", .{strerror(errnoPtr().*)});
            }
        }
        if (numa) {
            if (posix_madvise(addr, file.size, POSIX_MADV_RANDOM) != 0) {
                impl.logWarn("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n", .{strerror(errnoPtr().*)});
            }
        }
        var m: Mmap = .{ .addr = addr.?, .size = size };
        m.fragments.append(allocator, .{ 0, file.size }) catch impl.abort("llama-mmap.cpp", 475, "out of memory");
        return m;
    }

    /// Ports `llama_mmap`'s `impl`'s `align_range` (llama-mmap.cpp:478 @c1d0e7a00).
    fn alignRange(first: *usize, last: *usize, page_size: usize) void {
        const offset_in_page = first.* & (page_size - 1);
        const offset_to_page = if (offset_in_page == 0) 0 else page_size - offset_in_page;
        first.* += offset_to_page;
        last.* = last.* & ~(page_size - 1);
        if (last.* <= first.*) last.* = first.*;
    }

    /// Ports `llama_mmap`'s `impl`'s `unmap_fragment` (llama-mmap.cpp:490
    /// @c1d0e7a00): release the whole pages inside `[first, last)`.
    ///
    /// Return: nothing.
    pub fn unmapFragment(self: *Mmap, first_: usize, last_: usize) void {
        var first = first_;
        var last = last_;
        const page_size: c_int = @intCast(sysconf(_SC_PAGESIZE));
        const ps: usize = @intCast(page_size);
        alignRange(&first, &last, ps);
        const len = last - first;
        if (len == 0) return;

        impl.assert(first % ps == 0, "llama-mmap.cpp", 499, "first % page_size == 0");
        impl.assert(last % ps == 0, "llama-mmap.cpp", 500, "last % page_size == 0");
        impl.assert(last > first, "llama-mmap.cpp", 501, "last > first");

        const next_page_start: *anyopaque = @ptrFromInt(@intFromPtr(self.addr) + first);
        if (munmap(next_page_start, len) != 0) {
            impl.logWarn("warning: munmap failed: %s\n", .{strerror(errnoPtr().*)});
        }

        var next: std.ArrayList([2]usize) = .empty;
        const oom = struct {
            fn f() noreturn {
                impl.abort("llama-mmap.cpp", 510, "out of memory");
            }
        }.f;
        for (self.fragments.items) |frag| {
            if (frag[0] < first and frag[1] > last) {
                next.append(allocator, .{ frag[0], first }) catch oom();
                next.append(allocator, .{ last, frag[1] }) catch oom();
            } else if (frag[0] < first and frag[1] > first) {
                next.append(allocator, .{ frag[0], first }) catch oom();
            } else if (frag[0] < last and frag[1] > last) {
                next.append(allocator, .{ last, frag[1] }) catch oom();
            } else if (frag[0] >= first and frag[1] <= last) {
                // The fragment is entirely inside the unmapped range.
            } else {
                next.append(allocator, frag) catch oom();
            }
        }
        self.fragments.deinit(allocator);
        self.fragments = next;
    }

    /// Ports `llama_mmap`'s `impl`'s destructor (llama-mmap.cpp:526 @c1d0e7a00):
    /// unmap whatever is still mapped.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Mmap) void {
        for (self.fragments.items) |frag| {
            if (munmap(@ptrFromInt(@intFromPtr(self.addr) + frag[0]), frag[1] - frag[0]) != 0) {
                impl.logWarn("warning: munmap failed: %s\n", .{strerror(errnoPtr().*)});
            }
        }
        self.fragments.deinit(allocator);
    }
};

// -----------------------------------------------------------------------------
// llama_mlock

/// Ports `llama_mlock` and its POSIX `impl` (llama-mmap.cpp:636 @c1d0e7a00).
///
/// The C++ gives the `impl` no destructor, so a lock is never released
/// before the process ends; neither is one here.
pub const Mlock = struct {
    addr: ?*anyopaque = null,
    size: usize = 0,
    failed_already: bool = false,

    /// `llama_mlock`'s `SUPPORTED` (llama-mmap.cpp:772 @c1d0e7a00).
    pub const supported = true;

    /// Ports `llama_mlock`'s `impl`'s `lock_granularity` (llama-mmap.cpp:638 @c1d0e7a00).
    fn lockGranularity() usize {
        return @intCast(sysconf(_SC_PAGESIZE));
    }

    /// Ports `llama_mlock`'s `impl`'s `raw_lock` (llama-mmap.cpp:642 @c1d0e7a00).
    /// The suggestion is never printed on this target; see the file header.
    fn rawLock(self: *const Mlock, addr: *const anyopaque, size: usize) bool {
        if (mlock(addr, size) == 0) return true;
        impl.logWarn("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s", .{
            size, self.size, strerror(errnoPtr().*), @as([*:0]const u8, ""),
        });
        return false;
    }

    /// Ports `llama_mlock`'s `impl`'s `init` (llama-mmap.cpp:738 @c1d0e7a00).
    ///
    /// Return: nothing.
    pub fn init(self: *Mlock, ptr: *anyopaque) void {
        impl.assert(self.addr == null and self.size == 0, "llama-mmap.cpp", 739, "addr == NULL && size == 0");
        self.addr = ptr;
    }

    /// Ports `llama_mlock`'s `impl`'s `grow_to` (llama-mmap.cpp:743 @c1d0e7a00):
    /// lock up to `target_size` bytes, rounded up to a page, and give up for
    /// good after the first failure.
    ///
    /// Return: nothing.
    pub fn growTo(self: *Mlock, target_size_: usize) void {
        impl.assert(self.addr != null, "llama-mmap.cpp", 744, "addr");
        if (self.failed_already) return;
        const granularity = lockGranularity();
        const target_size = (target_size_ + granularity - 1) & ~(granularity - 1);
        if (target_size > self.size) {
            if (self.rawLock(@ptrFromInt(@intFromPtr(self.addr.?) + self.size), target_size - self.size)) {
                self.size = target_size;
            } else {
                self.failed_already = true;
            }
        }
    }
};

/// Ports `llama_path_max` (llama-mmap.cpp:777 @c1d0e7a00).
pub fn pathMax() usize {
    return PATH_MAX;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(File);
    std.testing.refAllDecls(Mmap);
    std.testing.refAllDecls(Mlock);
}

test "align_range rounds the start up and the end down to a page" {
    var first: usize = 1;
    var last: usize = 8191;
    Mmap.alignRange(&first, &last, 4096);
    try std.testing.expectEqual(@as(usize, 4096), first);
    try std.testing.expectEqual(@as(usize, 4096), last);

    first = 4096;
    last = 16384;
    Mmap.alignRange(&first, &last, 4096);
    try std.testing.expectEqual(@as(usize, 4096), first);
    try std.testing.expectEqual(@as(usize, 16384), last);
}
