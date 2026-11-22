//! Model files: reading them, mapping them, and locking them in memory.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-mmap.cpp` -- `llama_file`, `llama_mmap`,
//!   `ranges_complement`, `llama_mlock`, `llama_prefetch` and
//!   `llama_path_max`, the POSIX arm of each, with its `__linux__` and
//!   `__APPLE__` arms, and the `_WIN32` arm of each
//! - `llama.cpp/src/llama-mmap.h` -- `llama_mmap::ranges`,
//!   `llama_memory_range` and `LLAMA_DIRECT_IO_BUFFER_SIZE`
//!
//! At v1.0.0 (`b1b770d90`).
//!
//! # What each target compiles
//!
//! The `#if`s are resolved at comptime on `builtin.os`, as the preprocessor
//! resolves them; on Darwin this was measured, not assumed:
//!
//! - **Direct I/O is Linux and Windows only.** On POSIX `init_fd` and the
//!   unbuffered read path are behind `__linux__`; elsewhere on POSIX `fd` is
//!   always -1, `alignment` always 1, `has_direct_io` always false, and
//!   `read_raw` is always `read_raw_unsafe`'s buffered arm. Windows has its
//!   own (below).
//! - **`_POSIX_MAPPED_FILES` is defined on both**, so mapping is the `mmap`
//!   arm; Linux adds `posix_fadvise` and `MAP_POPULATE`.
//! - **`_POSIX_MEMLOCK_RANGE` is defined on both -- as -1 on Darwin.** The
//!   C++ asks `#ifdef`, not `#if`, so the `mlock` arm compiles and
//!   `SUPPORTED` is true.
//! - **The `mlock` suggestion is never printed on Darwin.**
//!   `TargetConditionals.h` defines `TARGET_OS_TV` (as 0), and the C++ asks
//!   `defined(...)`, so `suggest` is forced false and `getrlimit` is never
//!   called. On Linux it is printed when `mlock` failed for want of the
//!   `RLIMIT_MEMLOCK` headroom.
//! - **Windows is its own arm of all three**, measured with MinGW's headers,
//!   where `_POSIX_MAPPED_FILES` and `_POSIX_MEMLOCK_RANGE` are undefined.
//!   `llama_file` keeps the `FILE *` but seeks, tells, reads and writes
//!   through the Win32 handle behind it, and with direct I/O asked for
//!   reopens it unbuffered (`ReOpenFile`, `FILE_FLAG_NO_BUFFERING`) at the
//!   volume's sector alignment, the `NTDDI_WIN8` arm; mapping is
//!   `CreateFileMappingA` and `MapViewOfFile`, with no fragments to unmap;
//!   locking is `VirtualLock`, growing the working set once on failure.

const std = @import("std");
const libc = @import("stdlib").libc;
const stdlib = @import("stdlib");
const builtin = @import("builtin");
const impl = @import("impl.zig");
const win32 = @import("win32.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

/// `errno`'s location: `__error` on Darwin, `__errno_location` on glibc.
const errnoPtr = std.c._errno;
extern "c" fn mlock(addr: ?*const anyopaque, len: usize) c_int;
extern "c" fn fileno(stream: *c.FILE) c_int;

/// Ports `LLAMA_DIRECT_IO_BUFFER_SIZE` (llama-mmap.h:10 @b1b770d90): the
/// staging buffer for direct I/O reads; 64 MiB works well for NVMe drives.
pub const direct_io_buffer_size: usize = 64 * 1024 * 1024;

/// `__linux__`: the arms of `llama-mmap.cpp` only Linux compiles.
const is_linux = builtin.os.tag == .linux;

/// `_WIN32`: the arms of `llama-mmap.cpp` only Windows compiles.
const is_windows = builtin.os.tag == .windows;

/// Ports `llama_format_win_err` (llama-mmap.cpp:55 @b1b770d90): the system's
/// text for a Win32 error code, `"FormatMessageA failed"` when there is none.
///
/// Return: the text, owned by the caller; free with `allocator.free`.
fn formatWinErr(err: win32.DWORD) []u8 {
    var buf: ?[*]u8 = null;
    const size = win32.FormatMessageA(
        win32.FORMAT_MESSAGE_ALLOCATE_BUFFER | win32.FORMAT_MESSAGE_FROM_SYSTEM | win32.FORMAT_MESSAGE_IGNORE_INSERTS,
        null,
        err,
        win32.LANG_NEUTRAL_SUBLANG_DEFAULT,
        &buf,
        0,
        null,
    );
    const text: []const u8 = if (size == 0) "FormatMessageA failed" else buf.?[0..size];
    defer if (size != 0) {
        _ = win32.LocalFree(buf);
    };
    return allocator.dupe(u8, text) catch impl.abort("llama-mmap.cpp", 55, "out of memory");
}

/// Ports `llama_file`'s `impl`'s `GetErrorMessageWin32` (llama-mmap.cpp:76
/// @b1b770d90). Like `formatWinErr` but with its own fallback, the message
/// copied up to its terminator rather than by length, and trailing `\r` and
/// `\n` trimmed.
///
/// Return: the text, owned by the caller; free with `allocator.free`.
fn errorMessageWin32(error_code: win32.DWORD) []u8 {
    var buf: ?[*]u8 = null;
    const len = win32.FormatMessageA(
        win32.FORMAT_MESSAGE_ALLOCATE_BUFFER | win32.FORMAT_MESSAGE_FROM_SYSTEM | win32.FORMAT_MESSAGE_IGNORE_INSERTS,
        null,
        error_code,
        win32.LANG_NEUTRAL_SUBLANG_DEFAULT,
        &buf,
        0,
        null,
    );
    if (len == 0) {
        return std.fmt.allocPrint(allocator, "Win32 error code: {x}", .{error_code}) catch impl.abort("llama-mmap.cpp", 82, "out of memory");
    }
    defer _ = win32.LocalFree(buf);
    const text = std.mem.trimEnd(u8, std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf.?)), 0), "\r\n");
    return allocator.dupe(u8, text) catch impl.abort("llama-mmap.cpp", 84, "out of memory");
}

/// How `fp_win32` (llama-mmap.cpp:176 @b1b770d90) is set: `(HANDLE)
/// _get_osfhandle(_fileno(fp))`.
fn osHandle(fp: *c.FILE) win32.HANDLE {
    return @ptrFromInt(@as(usize, @bitCast(win32._get_osfhandle(win32._fileno(fp)))));
}

// <sys/mman.h>, <unistd.h> and <limits.h> on this target.
const PROT_READ: c_int = 1;
const MAP_SHARED: c_int = 1;
const MAP_FAILED: usize = std.math.maxInt(usize);
const POSIX_MADV_RANDOM: c_int = 1;
const POSIX_MADV_WILLNEED: c_int = 3;
/// Darwin's value, and glibc's.
const _SC_PAGESIZE: c_int = if (@import("builtin").os.tag.isDarwin()) 29 else 30;
/// `<limits.h>`'s: Darwin's value, and glibc's. MinGW has none, and the C++
/// defines `PATH_MAX` (llama-mmap.cpp:36 @b1b770d90) as `MAX_PATH`, 260.
const PATH_MAX: usize = if (is_windows) 260 else if (@import("builtin").os.tag.isDarwin()) 1024 else 4096;

fn errnoText() []const u8 {
    return std.mem.span(stdlib.errno.strerror(errnoPtr().*));
}

// -----------------------------------------------------------------------------
// llama_file

/// Ports `llama_file` and its `impl` (llama-mmap.cpp:70 @b1b770d90): the
/// `_WIN32` arm and the POSIX one.
///
/// A buffered `FILE *` and its size, or, when direct I/O was asked for and
/// the open succeeded, an unbuffered handle read in aligned chunks: on Linux
/// a descriptor that replaces the `FILE *`, on Windows a second handle
/// (`fp_win32_direct`) beside it. On Windows the `FILE *` only opens and
/// closes the file; every read, write and seek goes through `fp_win32`, the
/// handle behind it, which stays the logical cursor while the direct handle
/// only issues aligned reads.
pub const File = struct {
    /// `fp_win32`: the Win32 handle behind `fp`. Windows only.
    fp_win32: if (is_windows) win32.HANDLE else void = if (is_windows) win32.INVALID_HANDLE_VALUE else {},
    /// `fp_win32_direct`: the unbuffered handle, `INVALID_HANDLE_VALUE`
    /// without direct I/O. Windows only.
    fp_win32_direct: if (is_windows) win32.HANDLE else void = if (is_windows) win32.INVALID_HANDLE_VALUE else {},
    /// `fp`: null while `fd` is open.
    fp: ?*c.FILE,
    owns_fp: bool,
    size: usize,
    /// `alignment`: the read alignment, 1 without direct I/O.
    alignment: usize = 1,
    /// `fd`: the direct-I/O descriptor, -1 when reading through `fp`. Never
    /// anything but -1 off Linux.
    fd: c_int = -1,
    /// `fname`: the path, or `"(file*)"` for a borrowed `FILE *`; owned.
    fname: [:0]u8,

    /// Ports `llama_file`'s `impl`'s path constructors, the Windows one
    /// (llama-mmap.cpp:95 @b1b770d90) and the POSIX one (llama-mmap.cpp:407
    /// @b1b770d90), with `init_fp` (llama-mmap.cpp:444 @b1b770d90) and
    /// `init_buffered_io` (llama-mmap.cpp:171 @b1b770d90): open, then seek
    /// to the end for the size.
    ///
    /// Parameters:
    /// - `fname`: the path.
    /// - `mode`: the `fopen` mode.
    /// - `use_direct_io`: try unbuffered I/O; honoured on Linux and Windows
    ///   for `"rb"` only, as in the C++.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the file, or `error.Thrown` when it cannot be opened; free it
    /// with `close`.
    pub fn open(fname: [:0]const u8, mode: [:0]const u8, use_direct_io: bool, exc: *Exception) Thrown!File {
        var f: File = .{
            .fp = null,
            .owns_fp = true,
            .size = 0,
            .fname = allocator.dupeSentinel(u8, fname, 0) catch impl.abort("llama-mmap.cpp", 407, "out of memory"),
        };
        errdefer allocator.free(f.fname);
        if (comptime is_windows) {
            try f.initFp(mode, exc);
            if (use_direct_io and std.mem.eql(u8, mode, "rb")) {
                var err: win32.DWORD = win32.ERROR_SUCCESS;
                if (!f.initDirectIo(&err)) {
                    const msg = errorMessageWin32(err);
                    defer allocator.free(msg);
                    const msg_z = allocator.dupeSentinel(u8, msg, 0) catch impl.abort("llama-mmap.cpp", 99, "out of memory");
                    defer allocator.free(msg_z);
                    impl.logWarn("Failed to open file '%s' with unbuffered I/O: %s. Falling back to buffered I/O\n", .{ fname.ptr, msg_z.ptr });
                }
            }
            return f;
        }
        if (is_linux and use_direct_io and std.mem.eql(u8, mode, "rb")) {
            if (try f.initFd(exc)) {
                impl.logInfo("%s: file '%s' opened with direct I/O (alignment: %zu bytes)\n", .{ "impl", fname.ptr, f.alignment });
                return f;
            }
            impl.logWarn("Failed to open file '%s' with unbuffered I/O: %s. Falling back to buffered I/O\n", .{ fname.ptr, stdlib.errno.strerror(errnoPtr().*) });
        }
        try f.initFp(mode, exc);
        return f;
    }

    /// Ports `llama_file`'s `impl`'s `init_fd` (llama-mmap.cpp:424
    /// @b1b770d90): open for direct I/O, and take the size and the read
    /// alignment from `fstat`, which is asked
    /// through `statx` here; see `linux_c`.
    ///
    /// Return: whether the descriptor opened, or `error.Thrown`.
    fn initFd(self: *File, exc: *Exception) Thrown!bool {
        if (comptime is_linux) {
            const fd = std.c.open(self.fname.ptr, .{ .ACCMODE = .RDONLY, .DIRECT = true });
            if (fd == -1) return false;
            self.fd = fd;
            var file_stats: std.os.linux.Statx = std.mem.zeroes(std.os.linux.Statx);
            _ = libc.statx(fd, "", std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX.BASIC_STATS, &file_stats);
            self.size = @intCast(file_stats.size);
            self.alignment = @intCast(file_stats.blksize);
            if (std.c.lseek(fd, 0, std.c.SEEK.SET) == -1) {
                return exc.throw(.runtime_error, "seek error: {s}", .{errnoText()});
            }
            return true;
        } else return false;
    }

    /// Ports `llama_file`'s `_WIN32` `impl`'s `init_direct_io`
    /// (llama-mmap.cpp:106 @b1b770d90): reopen the file unbuffered beside
    /// the buffered handle and take the alignment from the volume's sector
    /// sizes. MinGW's headers define `NTDDI_VERSION` as Windows 10, so the
    /// `NTDDI_WIN8` arm is the one compiled -- measured, with `zig cc
    /// -target x86_64-windows-gnu`.
    ///
    /// Parameters:
    /// - `err`: receives the Win32 error on failure, `ERROR_SUCCESS` otherwise.
    ///
    /// Return: whether the direct handle opened.
    fn initDirectIo(self: *File, err: *win32.DWORD) bool {
        if (comptime !is_windows) unreachable;
        err.* = win32.ERROR_SUCCESS;
        self.alignment = 1;

        self.fp_win32_direct = win32.ReOpenFile(
            self.fp_win32,
            win32.GENERIC_READ,
            win32.FILE_SHARE_READ | win32.FILE_SHARE_WRITE | win32.FILE_SHARE_DELETE,
            win32.FILE_FLAG_NO_BUFFERING,
        );

        if (self.fp_win32_direct == win32.INVALID_HANDLE_VALUE) {
            err.* = win32.GetLastError();
            return false;
        }

        const ok = blk: {
            var storage_info: win32.FILE_STORAGE_INFO = std.mem.zeroes(win32.FILE_STORAGE_INFO);
            if (win32.GetFileInformationByHandleEx(self.fp_win32_direct, win32.FileStorageInfo, &storage_info, @sizeOf(win32.FILE_STORAGE_INFO)) == 0) {
                err.* = win32.GetLastError();
                break :blk false;
            }

            self.alignment = @max(
                @as(usize, storage_info.LogicalBytesPerSector),
                @as(usize, storage_info.PhysicalBytesPerSectorForAtomicity),
                @as(usize, storage_info.PhysicalBytesPerSectorForPerformance),
                @as(usize, storage_info.FileSystemEffectivePhysicalBytesPerSectorForAtomicity),
            );
            // Required for bit-mask rounding and aligned allocation schemes.
            if (self.alignment <= 1 or (self.alignment & (self.alignment - 1)) != 0) {
                err.* = win32.ERROR_NOT_SUPPORTED;
                break :blk false;
            }

            // ReOpenFile creates a distinct handle; explicitly start direct
            // I/O at offset 0.
            if (win32.SetFilePointerEx(self.fp_win32_direct, 0, null, win32.FILE_BEGIN) == 0) {
                err.* = win32.GetLastError();
                break :blk false;
            }

            break :blk true;
        };

        if (!ok) {
            _ = win32.CloseHandle(self.fp_win32_direct);
            self.fp_win32_direct = win32.INVALID_HANDLE_VALUE;
            self.alignment = 1;
            return false;
        }

        impl.logInfo("%s: file '%s' opened with unbuffered I/O (alignment: %zu bytes)\n", .{ "init_direct_io", self.fname.ptr, self.alignment });
        return true;
    }

    /// Ports `llama_file`'s `impl`'s `init_fp` (llama-mmap.cpp:444
    /// @b1b770d90) and the `_WIN32` `init_buffered_io` (llama-mmap.cpp:171
    /// @b1b770d90).
    ///
    /// Return: nothing, or `error.Thrown`.
    fn initFp(self: *File, mode: [:0]const u8, exc: *Exception) Thrown!void {
        const fp = c.ggml_fopen(self.fname.ptr, mode.ptr) orelse {
            return exc.throw(.runtime_error, "failed to open {s}: {s}", .{ self.fname, errnoText() });
        };
        self.fp = fp;
        if (comptime is_windows) self.fp_win32 = osHandle(fp);
        errdefer {
            _ = c.fclose(fp);
            self.fp = null;
        }
        try self.seek(0, c.SEEK_END, exc);
        self.size = try self.tell(exc);
        try self.seek(0, c.SEEK_SET, exc);
    }

    /// Ports `llama_file`'s `impl`'s `FILE *` constructors, the Windows one
    /// (llama-mmap.cpp:182 @b1b770d90) and the POSIX one
    /// (llama-mmap.cpp:454 @b1b770d90): borrowed, measured the same way.
    ///
    /// Parameters:
    /// - `fp`: the open file; borrowed, never closed here.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the file, or `error.Thrown` when it cannot be measured.
    pub fn borrow(fp: *c.FILE, exc: *Exception) Thrown!File {
        var f: File = .{
            .fp = fp,
            .owns_fp = false,
            .size = 0,
            .fname = allocator.dupeSentinel(u8, "(file*)", 0) catch impl.abort("llama-mmap.cpp", 454, "out of memory"),
        };
        errdefer allocator.free(f.fname);
        if (comptime is_windows) f.fp_win32 = osHandle(fp);
        try f.seek(0, c.SEEK_END, exc);
        f.size = try f.tell(exc);
        try f.seek(0, c.SEEK_SET, exc);
        return f;
    }

    /// Ports `llama_file`'s `impl`'s destructors, the Windows one
    /// (llama-mmap.cpp:397 @b1b770d90) and the POSIX one
    /// (llama-mmap.cpp:624 @b1b770d90).
    ///
    /// Return: nothing.
    pub fn close(self: *File) void {
        if (comptime is_windows) {
            if (self.fp_win32_direct != win32.INVALID_HANDLE_VALUE) {
                _ = win32.CloseHandle(self.fp_win32_direct);
            }
            if (self.owns_fp) if (self.fp) |fp| {
                _ = c.fclose(fp);
            };
        } else if (self.fd != -1) {
            _ = std.c.close(self.fd);
        } else if (self.owns_fp) {
            if (self.fp) |fp| _ = c.fclose(fp);
        }
        allocator.free(self.fname);
    }

    /// Ports `llama_file`'s `impl`'s `tell`, the Windows one
    /// (llama-mmap.cpp:190 @b1b770d90) and the POSIX one
    /// (llama-mmap.cpp:461 @b1b770d90).
    ///
    /// Return: the position, or `error.Thrown`.
    pub fn tell(self: *const File, exc: *Exception) Thrown!usize {
        if (comptime is_windows) {
            var li: i64 = 0;
            if (win32.SetFilePointerEx(self.fp_win32, 0, &li, win32.FILE_CURRENT) == 0) {
                return throwWin32(exc, "tell error: {s}", win32.GetLastError());
            }
            return @intCast(li);
        }
        if (self.fd == -1) {
            const ret = c.ftello(self.fp.?);
            if (ret == -1) return exc.throw(.runtime_error, "ftell error: {s}", .{errnoText()});
            return @intCast(ret);
        }
        const pos = std.c.lseek(self.fd, 0, std.c.SEEK.CUR);
        if (pos == -1) return exc.throw(.runtime_error, "lseek error: {s}", .{errnoText()});
        return @intCast(pos);
    }

    /// Ports `llama_file`'s `impl`'s `seek`, the Windows one
    /// (llama-mmap.cpp:201 @b1b770d90) and the POSIX one
    /// (llama-mmap.cpp:478 @b1b770d90). On Windows `SEEK_SET`, `SEEK_CUR`
    /// and `SEEK_END` are `FILE_BEGIN`, `FILE_CURRENT` and `FILE_END`.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn seek(self: *const File, offset: usize, whence: c_int, exc: *Exception) Thrown!void {
        if (comptime is_windows) {
            if (win32.SetFilePointerEx(self.fp_win32, @bitCast(offset), null, @intCast(whence)) == 0) {
                return throwWin32(exc, "seek error: {s}", win32.GetLastError());
            }
            return;
        }
        const ret: i64 = if (self.fd == -1)
            c.fseeko(self.fp.?, @bitCast(offset), whence)
        else
            std.c.lseek(self.fd, @bitCast(offset), @intCast(whence));
        if (ret == -1) {
            return exc.throw(.runtime_error, "seek error: {s}", .{errnoText()});
        }
    }

    /// Ports `llama_file`'s `impl`'s `read_raw_unsafe` (llama-mmap.cpp:490
    /// @b1b770d90). Buffered, a read is clamped to what is left of the file,
    /// so reading past the end succeeds short. Unbuffered, it retries on
    /// `EINTR`, falls back to buffered I/O on `EFAULT` or `EINVAL`, and
    /// zero-fills alignment padding read past the end.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn readRawUnsafe(self: *File, ptr: *anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (comptime is_windows) return self.readRawUnsafeWin32(ptr, len, exc);
        if (len == 0) return;
        errnoPtr().* = 0;
        if (self.fd == -1) {
            const read_offset = try self.tell(exc);
            const to_read = @min(len, self.size -% read_offset);
            const ret = c.fread(ptr, to_read, 1, self.fp.?);
            if (c.ferror(self.fp.?) != 0) return exc.throw(.runtime_error, "read error: {s}", .{errnoText()});
            if (to_read > 0 and ret != 1) return exc.throw(.runtime_error, "unexpectedly reached end of file", .{});
            return;
        }
        const bytes: [*]u8 = @ptrCast(ptr);
        var bytes_read: usize = 0;
        while (bytes_read < len) {
            const to_read = len - bytes_read;
            const ret = std.c.read(self.fd, bytes + bytes_read, to_read);
            if (ret == -1) {
                const err: std.c.E = @fromBackingInt(@intCast(errnoPtr().*));
                if (err == .INTR) continue;
                if (err == .FAULT or err == .INVAL) {
                    impl.logWarn("%s: file '%s' falling back to buffered I/O due to %s\n", .{ "read_raw_unsafe", self.fname.ptr, stdlib.errno.strerror(errnoPtr().*) });
                    const read_offset = try self.tell(exc);
                    _ = std.c.close(self.fd);
                    self.fd = -1;
                    self.alignment = 1;
                    try self.initFp("rb", exc);
                    try self.seek(read_offset, c.SEEK_SET, exc);
                    return self.readRawUnsafe(ptr, len, exc);
                }
                return exc.throw(.runtime_error, "read error: {s}", .{errnoText()});
            }
            if (ret == 0) {
                const pos = std.c.lseek(self.fd, 0, std.c.SEEK.CUR);
                if (pos != -1 and @as(usize, @intCast(pos)) == self.size) {
                    @memset(bytes[bytes_read..len], 0);
                    return;
                }
                return exc.throw(.runtime_error, "unexpectedly reached end of file", .{});
            }
            bytes_read += @intCast(ret);
        }
    }

    /// Ports `llama_file`'s `impl`'s `read_aligned_chunk` (llama-mmap.cpp:544
    /// @b1b770d90): read the aligned span around `[offset, offset + len)`
    /// through an aligned staging buffer of at most
    /// `LLAMA_DIRECT_IO_BUFFER_SIZE`, so a large tensor is not held in memory
    /// twice, copy out the part asked for, and leave the position after it.
    /// When the buffer cannot be allocated, fall back to buffered I/O.
    ///
    /// Return: nothing, or `error.Thrown`.
    fn readAlignedChunk(self: *File, dest: *anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (comptime is_windows) return self.readAlignedChunkWin32(dest, len, exc);
        const alignment = self.alignment;
        const offset = try self.tell(exc);
        const aligned_offset = offset & ~(alignment - 1);
        const offset_from_alignment = offset - aligned_offset;
        const bytes_to_read = (offset_from_alignment + len + alignment - 1) & ~(alignment - 1);

        const buffer_size = @min(bytes_to_read, direct_io_buffer_size);

        var raw_buffer: ?*anyopaque = null;
        const ret = libc.posix_memalign(&raw_buffer, alignment, buffer_size);
        if (ret != 0) {
            impl.logWarn("%s: file '%s' falling back to buffered I/O due to %s\n", .{ "read_aligned_chunk", self.fname.ptr, stdlib.errno.strerror(ret) });
            _ = std.c.close(self.fd);
            self.fd = -1;
            self.alignment = 1;
            try self.initFp("rb", exc);
            try self.seek(offset, c.SEEK_SET, exc);
            return self.readRawUnsafe(dest, len, exc);
        }
        defer libc.free(raw_buffer);

        try self.seek(aligned_offset, c.SEEK_SET, exc);

        const buf: [*]const u8 = @ptrCast(raw_buffer.?);
        const out: [*]u8 = @ptrCast(dest);
        var skip = offset_from_alignment;
        var copied: usize = 0;
        var done: usize = 0;
        while (done < bytes_to_read) {
            const n = @min(buffer_size, bytes_to_read - done);
            try self.readRawUnsafe(raw_buffer.?, n, exc);

            const count = @min(n - skip, len - copied);
            @memcpy(out[copied..][0..count], buf[skip..][0..count]);

            copied += count;
            skip = 0;
            done += n;
        }

        try self.seek(offset + len, c.SEEK_SET, exc);
    }

    /// Ports `llama_file`'s `impl`'s `read_raw`, the Windows one
    /// (llama-mmap.cpp:349 @b1b770d90) and the POSIX one
    /// (llama-mmap.cpp:591 @b1b770d90). Only the `_WIN32` one checks the
    /// bounds first.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn readRaw(self: *File, ptr: *anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (comptime is_windows) {
            if (len == 0) return;
            const offset = try self.tell(exc);
            if (offset > self.size or len > self.size - offset) {
                return exc.throw(.runtime_error, "unexpectedly reached end of file", .{});
            }
        }
        if (self.hasDirectIo()) return self.readAlignedChunk(ptr, len, exc);
        return self.readRawUnsafe(ptr, len, exc);
    }

    /// Ports `llama_file`'s `impl`'s `read_u32` (llama-mmap.cpp:599 @b1b770d90).
    ///
    /// Return: the value, or `error.Thrown`.
    pub fn readU32(self: *File, exc: *Exception) Thrown!u32 {
        var v: u32 = undefined;
        try self.readRaw(&v, @sizeOf(u32), exc);
        return v;
    }

    /// Ports `llama_file`'s `impl`'s `write_raw` (llama-mmap.cpp:605 @b1b770d90).
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn writeRaw(self: *const File, ptr: *const anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (comptime is_windows) return self.writeRawWin32(ptr, len, exc);
        if (len == 0) return;
        errnoPtr().* = 0;
        if (c.fwrite(ptr, len, 1, self.fp.?) != 1) return exc.throw(.runtime_error, "write error: {s}", .{errnoText()});
    }

    /// Ports `llama_file`'s `impl`'s `write_u32` (llama-mmap.cpp:616 @b1b770d90).
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn writeU32(self: *const File, val: u32, exc: *Exception) Thrown!void {
        return self.writeRaw(&val, @sizeOf(u32), exc);
    }

    /// Ports `llama_file`'s `file_id` (llama-mmap.cpp:659 @b1b770d90).
    ///
    /// Return: the direct-I/O descriptor, or the one behind the `FILE *`.
    pub fn fileId(self: *const File) c_int {
        if (comptime is_windows) return win32._fileno(self.fp.?);
        if (self.fd != -1) return self.fd;
        return fileno(self.fp.?);
    }

    /// Ports `llama_file`'s `impl`'s `has_direct_io`, the Windows one
    /// (llama-mmap.cpp:393 @b1b770d90) and the POSIX one
    /// (llama-mmap.cpp:620 @b1b770d90).
    pub fn hasDirectIo(self: *const File) bool {
        if (comptime is_windows) return self.fp_win32_direct != win32.INVALID_HANDLE_VALUE and self.alignment > 1;
        return self.fd != -1 and self.alignment > 1;
    }

    /// Throws `fmt` with `GetErrorMessageWin32(error_code)` as its one
    /// argument, as each Win32 failure in `llama_file` does.
    fn throwWin32(exc: *Exception, comptime fmt: []const u8, error_code: win32.DWORD) Thrown {
        const msg = errorMessageWin32(error_code);
        defer allocator.free(msg);
        return exc.throw(.runtime_error, fmt, .{msg});
    }

    /// Logs `fmt` with `__func__`, `fname` and `GetErrorMessageWin32(error_code)`,
    /// as the `_WIN32` arm's falls back to buffered I/O do.
    fn warnWin32Fallback(self: *const File, comptime func: [:0]const u8, error_code: win32.DWORD) void {
        const msg = errorMessageWin32(error_code);
        defer allocator.free(msg);
        const msg_z = allocator.dupeSentinel(u8, msg, 0) catch impl.abort("llama-mmap.cpp", 283, "out of memory");
        defer allocator.free(msg_z);
        impl.logWarn("%s: file '%s' falling back to buffered I/O due to %s\n", .{ func.ptr, self.fname.ptr, msg_z.ptr });
    }

    /// Drops the direct handle and goes on buffered, as each `_WIN32`
    /// fallback does.
    fn dropDirectWin32(self: *File) void {
        _ = win32.CloseHandle(self.fp_win32_direct);
        self.fp_win32_direct = win32.INVALID_HANDLE_VALUE;
        self.alignment = 1;
    }

    /// Ports `llama_file`'s `_WIN32` `impl`'s `read_raw_unsafe`
    /// (llama-mmap.cpp:214 @b1b770d90). Buffered: `ReadFile` in chunks of at
    /// most `MAXDWORD`, clamped to what is left of the file, any short chunk
    /// an error. Unbuffered: aligned `ReadFile`s on the direct handle from the
    /// buffered handle's position, a short read allowed only at the end of
    /// the file; a misaligned request or buffer falls back to buffered I/O,
    /// and the buffered handle's position is moved past what was read.
    ///
    /// Return: nothing, or `error.Thrown`.
    fn readRawUnsafeWin32(self: *File, ptr: *anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (len == 0) return;

        const read_offset = try self.tell(exc);
        if (read_offset > self.size) {
            return exc.throw(.runtime_error, "unexpectedly reached end of file", .{});
        }

        const bytes: [*]u8 = @ptrCast(ptr);
        if (!self.hasDirectIo()) {
            const read_size = @min(len, self.size - read_offset);
            var bytes_read: usize = 0;
            while (bytes_read < read_size) {
                const to_read: win32.DWORD = @intCast(@min(read_size - bytes_read, win32.MAXDWORD));
                var chunk_read: win32.DWORD = 0;
                if (win32.ReadFile(self.fp_win32, bytes + bytes_read, to_read, &chunk_read, null) == 0) {
                    return throwWin32(exc, "read error: {s}", win32.GetLastError());
                }
                if (chunk_read < to_read) {
                    return exc.throw(.runtime_error, "unexpectedly reached end of file", .{});
                }
                bytes_read += chunk_read;
            }
            return;
        }

        const alignment = self.alignment;
        var err: win32.DWORD = win32.ERROR_SUCCESS;
        if ((read_offset & (alignment - 1)) != 0 or
            (len & (alignment - 1)) != 0 or
            (@intFromPtr(ptr) & (alignment - 1)) != 0)
        {
            err = win32.ERROR_INVALID_PARAMETER;
        } else if (win32.SetFilePointerEx(self.fp_win32_direct, @intCast(read_offset), null, win32.FILE_BEGIN) == 0) {
            err = win32.GetLastError();
        }

        var bytes_read: usize = 0;
        const max_read_size: usize = @as(usize, win32.MAXDWORD) & ~(alignment - 1);
        while (err == win32.ERROR_SUCCESS and bytes_read < len) {
            const to_read: win32.DWORD = @intCast(@min(len - bytes_read, max_read_size));
            var chunk_read: win32.DWORD = 0;
            if (win32.ReadFile(self.fp_win32_direct, bytes + bytes_read, to_read, &chunk_read, null) == 0) {
                err = win32.GetLastError();
                if (err == win32.ERROR_HANDLE_EOF and bytes_read == self.size - read_offset) {
                    err = win32.ERROR_SUCCESS;
                    break;
                }
                break;
            }

            bytes_read += chunk_read;
            if (chunk_read < to_read) {
                if (bytes_read == self.size - read_offset) break;
                err = win32.ERROR_HANDLE_EOF;
            }
        }

        if (err == win32.ERROR_INVALID_PARAMETER or err == win32.ERROR_INVALID_USER_BUFFER or err == win32.ERROR_NOACCESS) {
            self.warnWin32Fallback("read_raw_unsafe", err);
            self.dropDirectWin32();
            try self.seek(read_offset, c.SEEK_SET, exc);
            return self.readRawUnsafeWin32(ptr, len, exc);
        }
        if (err != win32.ERROR_SUCCESS) {
            return throwWin32(exc, "read error: {s}", err);
        }

        try self.seek(if (len > self.size - read_offset) self.size else read_offset + len, c.SEEK_SET, exc);
    }

    /// Ports `llama_file`'s `_WIN32` `impl`'s `read_aligned_chunk`
    /// (llama-mmap.cpp:299 @b1b770d90): as the POSIX one, with the bounds
    /// checked first, `_aligned_malloc` for the staging buffer, and a skip
    /// that may span more than one staging chunk.
    ///
    /// Return: nothing, or `error.Thrown`.
    fn readAlignedChunkWin32(self: *File, dest: *anyopaque, len: usize, exc: *Exception) Thrown!void {
        if (len == 0) return;

        const offset = try self.tell(exc);
        if (offset > self.size or len > self.size - offset) {
            return exc.throw(.runtime_error, "unexpectedly reached end of file", .{});
        }

        const alignment = self.alignment;
        const aligned_offset = offset & ~(alignment - 1);
        const offset_from_alignment = offset - aligned_offset;
        const bytes_to_read = (offset_from_alignment + len + alignment - 1) & ~(alignment - 1);
        const buffer_size = @min(bytes_to_read, direct_io_buffer_size);

        const raw_buffer = libc._aligned_malloc(buffer_size, alignment) orelse {
            const msg = errorMessageWin32(win32.ERROR_NOT_ENOUGH_MEMORY);
            defer allocator.free(msg);
            const msg_z = allocator.dupeSentinel(u8, msg, 0) catch impl.abort("llama-mmap.cpp", 317, "out of memory");
            defer allocator.free(msg_z);
            impl.logWarn("%s: Falling back to buffered I/O due to %s\n", .{ "read_aligned_chunk", msg_z.ptr });
            self.dropDirectWin32();
            try self.seek(offset, c.SEEK_SET, exc);
            return self.readRawUnsafeWin32(dest, len, exc);
        };
        defer libc._aligned_free(raw_buffer);

        try self.seek(aligned_offset, c.SEEK_SET, exc);

        const buf: [*]const u8 = @ptrCast(raw_buffer);
        const out: [*]u8 = @ptrCast(dest);
        var skip = offset_from_alignment;
        var copied: usize = 0;
        var done: usize = 0;
        while (done < bytes_to_read) {
            const n = @min(buffer_size, bytes_to_read - done);
            try self.readRawUnsafeWin32(raw_buffer, n, exc);

            const chunk_skip = @min(skip, n);
            const count = @min(n - chunk_skip, len - copied);
            @memcpy(out[copied..][0..count], buf[chunk_skip..][0..count]);

            copied += count;
            skip -= chunk_skip;
            done += n;
        }

        try self.seek(offset + len, c.SEEK_SET, exc);
    }

    /// Ports `llama_file`'s `_WIN32` `impl`'s `write_raw` (llama-mmap.cpp:372
    /// @b1b770d90): `WriteFile` in chunks of at most
    /// `LLAMA_DIRECT_IO_BUFFER_SIZE`.
    ///
    /// Return: nothing, or `error.Thrown`.
    fn writeRawWin32(self: *const File, ptr: *const anyopaque, len: usize, exc: *Exception) Thrown!void {
        const bytes: [*]const u8 = @ptrCast(ptr);
        var bytes_written: usize = 0;
        while (bytes_written < len) {
            const chunk_size = @min(len - bytes_written, direct_io_buffer_size);
            var chunk_written: win32.DWORD = 0;
            if (win32.WriteFile(self.fp_win32, bytes + bytes_written, @intCast(chunk_size), &chunk_written, null) == 0) {
                return throwWin32(exc, "write error: {s}", win32.GetLastError());
            }
            if (chunk_written < chunk_size or chunk_written == 0) {
                return exc.throw(.runtime_error, "unexpectedly failed to write bytes", .{});
            }
            bytes_written += chunk_written;
        }
    }

    /// Return: `llama_file`'s `read_alignment`, 1 without direct I/O.
    pub fn readAlignment(self: *const File) usize {
        return self.alignment;
    }
};

// -----------------------------------------------------------------------------
// llama_mmap

/// Mirrors `llama_mmap`'s `ranges` (llama-mmap.h:49 @b1b770d90): `[first, last)`
/// byte ranges within a file.
pub const Ranges = std.ArrayList([2]usize);

fn rangeLessThan(_: void, a: [2]usize, b: [2]usize) bool {
    // `std::pair`'s `operator<`.
    if (a[0] != b[0]) return a[0] < b[0];
    return a[1] < b[1];
}

/// Ports `ranges_complement` (llama-mmap.cpp:687 @b1b770d90): merge `ranges`
/// and return their complement within `[0, limit)`.
///
/// Parameters:
/// - `ranges`: the ranges, in any order, possibly overlapping. Copied, as
///   the C++ takes them by value.
/// - `limit`: the end of the span complemented.
///
/// Return: the gaps, in order; owned by the caller, free with `deinit`.
pub fn rangesComplement(ranges: []const [2]usize, limit: usize) Ranges {
    var res: Ranges = .empty;
    const sorted = allocator.dupe([2]usize, ranges) catch impl.abort("llama-mmap.cpp", 687, "out of memory");
    defer allocator.free(sorted);
    // `std::sort`: equal pairs are identical, so the order is the same.
    std.sort.pdq([2]usize, sorted, {}, rangeLessThan);

    var pos: usize = 0;
    for (sorted) |range| {
        const beg = @min(range[0], limit);
        const end = @min(range[1], limit);
        if (beg > pos) res.append(allocator, .{ pos, beg }) catch impl.abort("llama-mmap.cpp", 696, "out of memory");
        pos = @max(pos, end);
    }
    if (pos < limit) res.append(allocator, .{ pos, limit }) catch impl.abort("llama-mmap.cpp", 701, "out of memory");

    return res;
}

/// Ports `llama_mmap` and its `impl` (llama-mmap.cpp:708 @b1b770d90): the
/// POSIX arm and the `_WIN32` one.
pub const Mmap = struct {
    addr: *anyopaque,
    size: usize,
    /// `mapped_fragments`: the byte ranges still mapped. POSIX only.
    fragments: std.ArrayList([2]usize) = .empty,
    /// `hMapping`: the file-mapping object. Windows only.
    mapping: if (is_windows) ?win32.HANDLE else void = if (is_windows) null else {},

    /// `llama_mmap`'s `SUPPORTED` (llama-mmap.cpp:919 @b1b770d90):
    /// `_POSIX_MEMLOCK_RANGE` or `_WIN32`, which every target this builds
    /// for defines.
    pub const supported = true;

    /// Ports `llama_mmap`'s `impl`'s constructor (llama-mmap.cpp:712 @b1b770d90).
    ///
    /// Parameters:
    /// - `file`: the file to map, read-only and shared.
    /// - `prefetch`: how many leading bytes to advise the kernel will be
    ///   needed; 0 for none.
    /// - `numa`: advise random access and do not prefetch.
    /// - `lazy_ranges`: the byte ranges read on demand: kept out of the
    ///   prefetch and advised random. Borrowed.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the mapping, or `error.Thrown`; free it with `deinit`.
    pub fn init(file: *const File, prefetch_: usize, numa: bool, lazy_ranges: []const [2]usize, exc: *Exception) Thrown!Mmap {
        if (comptime is_windows) return initWin32(file, prefetch_, lazy_ranges, exc);
        var prefetch = prefetch_;
        const size = file.size;
        const fd = file.fileId();
        var flags: c_int = MAP_SHARED;
        if (numa) prefetch = 0;
        if (is_linux) {
            if (libc.posix_fadvise(fd, 0, 0, std.os.linux.POSIX_FADV.SEQUENTIAL) != 0) {
                impl.logWarn("warning: libc.posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n", .{stdlib.errno.strerror(errnoPtr().*)});
            }
            // MAP_POPULATE would fault in the lazy ranges too.
            if (prefetch != 0 and lazy_ranges.len == 0) flags = @bitCast(std.os.linux.MAP{ .TYPE = .SHARED, .POPULATE = true });
        }
        const addr = libc.mmap(null, file.size, PROT_READ, flags, fd, 0);
        if (addr == null or @intFromPtr(addr) == MAP_FAILED) {
            return exc.throw(.runtime_error, "mmap failed: {s}", .{errnoText()});
        }

        if (prefetch > 0) {
            var gaps = rangesComplement(lazy_ranges, @min(file.size, prefetch));
            defer gaps.deinit(allocator);
            for (gaps.items) |range| advise(addr.?, file.size, range[0], range[1], POSIX_MADV_WILLNEED, "POSIX_MADV_WILLNEED");
        }
        for (lazy_ranges) |range| advise(addr.?, file.size, range[0], range[1], POSIX_MADV_RANDOM, "POSIX_MADV_RANDOM");
        if (numa) {
            if (libc.posix_madvise(addr, file.size, POSIX_MADV_RANDOM) != 0) {
                impl.logWarn("warning: libc.posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n", .{stdlib.errno.strerror(errnoPtr().*)});
            }
        }
        var m: Mmap = .{ .addr = addr.?, .size = size };
        m.fragments.append(allocator, .{ 0, file.size }) catch impl.abort("llama-mmap.cpp", 758, "out of memory");
        return m;
    }

    /// The `advise` lambda of `llama_mmap`'s `impl`'s constructor
    /// (llama-mmap.cpp:712 @b1b770d90): a page-aligned `posix_madvise` over
    /// `[beg, end)`, clamped to the file.
    fn advise(addr: *anyopaque, file_size: usize, beg_: usize, end_: usize, advice: c_int, name: [*:0]const u8) void {
        const page_size: usize = @intCast(libc.sysconf(_SC_PAGESIZE));
        const beg = beg_ & ~(page_size - 1);
        const end = @min((end_ +% page_size -% 1) & ~(page_size - 1), file_size);
        if (beg >= end) return;
        if (libc.posix_madvise(@ptrFromInt(@intFromPtr(addr) + beg), end - beg, advice) != 0) {
            impl.logWarn("warning: libc.posix_madvise(.., %s) failed: %s\n", .{ name, stdlib.errno.strerror(errnoPtr().*) });
        }
    }

    /// Ports `llama_mmap`'s `_WIN32` `impl`'s constructor (llama-mmap.cpp:819
    /// @b1b770d90). `numa` is unused there.
    fn initWin32(file: *const File, prefetch: usize, lazy_ranges: []const [2]usize, exc: *Exception) Thrown!Mmap {
        const size = file.size;

        const h_file: win32.HANDLE = @ptrFromInt(@as(usize, @bitCast(win32._get_osfhandle(file.fileId()))));

        const mapping = win32.CreateFileMappingA(h_file, null, win32.PAGE_READONLY, 0, 0, null) orelse {
            const msg = formatWinErr(win32.GetLastError());
            defer allocator.free(msg);
            return exc.throw(.runtime_error, "CreateFileMappingA failed: {s}", .{msg});
        };

        const addr = win32.MapViewOfFile(mapping, win32.FILE_MAP_READ, 0, 0, 0);
        const err = win32.GetLastError();

        if (addr == null) {
            _ = win32.CloseHandle(mapping);
            const msg = formatWinErr(err);
            defer allocator.free(msg);
            return exc.throw(.runtime_error, "MapViewOfFile failed: {s}", .{msg});
        }

        if (prefetch > 0) {
            // `_WIN32_WINNT` is MinGW's default, 0xa00, so the C++'s
            // `>= 0x602` arm compiles.
            const kernel32 = win32.GetModuleHandleW(std.unicode.utf8ToUtf16LeStringLiteral("kernel32.dll"));
            const sym = if (kernel32) |k| win32.GetProcAddress(k, "PrefetchVirtualMemory") else null;
            if (sym) |p| {
                const prefetch_virtual_memory: win32.PrefetchVirtualMemoryFn = @ptrCast(p);
                var gaps = rangesComplement(lazy_ranges, @min(size, prefetch));
                defer gaps.deinit(allocator);
                var entries: std.ArrayList(win32.WIN32_MEMORY_RANGE_ENTRY) = .empty;
                defer entries.deinit(allocator);
                for (gaps.items) |range| {
                    entries.append(allocator, .{
                        .VirtualAddress = @ptrFromInt(@intFromPtr(addr.?) + range[0]),
                        .NumberOfBytes = range[1] - range[0],
                    }) catch impl.abort("llama-mmap.cpp", 854, "out of memory");
                }
                if (entries.items.len != 0 and
                    prefetch_virtual_memory(win32.GetCurrentProcess(), entries.items.len, entries.items.ptr, 0) == 0)
                {
                    warnWin32("warning: PrefetchVirtualMemory failed: %s\n");
                }
            }
        }

        return .{ .addr = addr.?, .size = size, .mapping = mapping };
    }

    /// Ports `llama_mmap`'s `impl`'s `align_range` (llama-mmap.cpp:761 @b1b770d90).
    fn alignRange(first: *usize, last: *usize, page_size: usize) void {
        const offset_in_page = first.* & (page_size - 1);
        const offset_to_page = if (offset_in_page == 0) 0 else page_size - offset_in_page;
        first.* += offset_to_page;
        last.* = last.* & ~(page_size - 1);
        if (last.* <= first.*) last.* = first.*;
    }

    /// Ports `llama_mmap`'s `impl`'s `unmap_fragment` (llama-mmap.cpp:773
    /// @b1b770d90): release the whole pages inside `[first, last)`.
    ///
    /// Return: nothing.
    pub fn unmapFragment(self: *Mmap, first_: usize, last_: usize) void {
        // `unmap_fragment` (llama-mmap.cpp:868 @b1b770d90) is a no-op on
        // Windows.
        if (comptime is_windows) return;
        var first = first_;
        var last = last_;
        const page_size: c_int = @intCast(libc.sysconf(_SC_PAGESIZE));
        const ps: usize = @intCast(page_size);
        alignRange(&first, &last, ps);
        const len = last - first;
        if (len == 0) return;

        impl.assert(first % ps == 0, "llama-mmap.cpp", 782, "first % page_size == 0");
        impl.assert(last % ps == 0, "llama-mmap.cpp", 783, "last % page_size == 0");
        impl.assert(last > first, "llama-mmap.cpp", 784, "last > first");

        const next_page_start: *anyopaque = @ptrFromInt(@intFromPtr(self.addr) + first);
        if (libc.munmap(next_page_start, len) != 0) {
            impl.logWarn("warning: munmap failed: %s\n", .{stdlib.errno.strerror(errnoPtr().*)});
        }

        var next: std.ArrayList([2]usize) = .empty;
        const oom = struct {
            fn f() noreturn {
                impl.abort("llama-mmap.cpp", 793, "out of memory");
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

    /// Ports `llama_mmap`'s `impl`'s destructor (llama-mmap.cpp:809 @b1b770d90):
    /// unmap whatever is still mapped.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Mmap) void {
        if (comptime is_windows) return self.deinitWin32();
        for (self.fragments.items) |frag| {
            if (libc.munmap(@ptrFromInt(@intFromPtr(self.addr) + frag[0]), frag[1] - frag[0]) != 0) {
                impl.logWarn("warning: munmap failed: %s\n", .{stdlib.errno.strerror(errnoPtr().*)});
            }
        }
        self.fragments.deinit(allocator);
    }

    /// Ports `llama_mmap`'s `_WIN32` `impl`'s destructor (llama-mmap.cpp:873
    /// @b1b770d90).
    fn deinitWin32(self: *Mmap) void {
        const mapping = self.mapping orelse return;
        if (win32.UnmapViewOfFile(self.addr) == 0) {
            warnWin32("warning: UnmapViewOfFile failed: %s\n");
        }
        if (win32.CloseHandle(mapping) == 0) {
            warnWin32("warning: CloseHandle failed: %s\n");
        }
    }
};

/// Logs `fmt` as a warning with `llama_format_win_err(GetLastError())` as its
/// one argument, as each Win32 failure in `llama_mmap` and `llama_mlock` does.
fn warnWin32(comptime fmt: [:0]const u8) void {
    const msg = formatWinErr(win32.GetLastError());
    defer allocator.free(msg);
    const z = allocator.dupeSentinel(u8, msg, 0) catch impl.abort("llama-mmap.cpp", 55, "out of memory");
    defer allocator.free(z);
    impl.logWarn(fmt, .{z.ptr});
}

// -----------------------------------------------------------------------------
// llama_mlock

/// Ports `llama_mlock` and its `impl` (llama-mmap.cpp:926 @b1b770d90): the
/// POSIX arm and the `_WIN32` one.
///
/// The C++ gives the `impl` no destructor, so a lock is never released
/// before the process ends; neither is one here.
pub const Mlock = struct {
    addr: ?*anyopaque = null,
    size: usize = 0,
    failed_already: bool = false,

    /// `llama_mlock`'s `SUPPORTED` (llama-mmap.cpp:1062 @b1b770d90).
    pub const supported = true;

    /// Ports `llama_mlock`'s `impl`'s `lock_granularity` (llama-mmap.cpp:928 @b1b770d90).
    fn lockGranularity() usize {
        if (comptime is_windows) {
            // llama-mmap.cpp:973 @b1b770d90.
            var si: win32.SYSTEM_INFO = undefined;
            win32.GetSystemInfo(&si);
            return si.dwPageSize;
        }
        return @intCast(libc.sysconf(_SC_PAGESIZE));
    }

    /// Ports `llama_mlock`'s `_WIN32` `impl`'s `raw_lock` (llama-mmap.cpp:979
    /// @b1b770d90): on the first failure, grow the working set by the request
    /// plus 1 MiB and try once more.
    fn rawLockWin32(self: *const Mlock, ptr: *anyopaque, len: usize) bool {
        var tries: c_int = 1;
        while (true) : (tries += 1) {
            if (win32.VirtualLock(ptr, len) != 0) return true;
            if (tries == 2) {
                const msg = formatWinErr(win32.GetLastError());
                defer allocator.free(msg);
                const z = allocator.dupeSentinel(u8, msg, 0) catch impl.abort("llama-mmap.cpp", 985, "out of memory");
                defer allocator.free(z);
                impl.logWarn("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n", .{ len, self.size, z.ptr });
                return false;
            }

            var min_ws_size: usize = undefined;
            var max_ws_size: usize = undefined;
            if (win32.GetProcessWorkingSetSize(win32.GetCurrentProcess(), &min_ws_size, &max_ws_size) == 0) {
                warnWin32("warning: GetProcessWorkingSetSize failed: %s\n");
                return false;
            }
            const increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (win32.SetProcessWorkingSetSize(win32.GetCurrentProcess(), min_ws_size, max_ws_size) == 0) {
                warnWin32("warning: SetProcessWorkingSetSize failed: %s\n");
                return false;
            }
        }
    }

    /// Ports `llama_mlock`'s `impl`'s `raw_lock` (llama-mmap.cpp:932 @b1b770d90).
    /// The suggestion is never printed on Darwin; see the file header.
    fn rawLock(self: *const Mlock, addr: *const anyopaque, size: usize) bool {
        if (comptime is_windows) return self.rawLockWin32(@constCast(addr), size);
        if (mlock(addr, size) == 0) return true;

        const errmsg = stdlib.errno.strerror(errnoPtr().*);
        var suggest = errnoPtr().* == @backingInt(std.c.E.NOMEM);
        if (comptime builtin.os.tag.isDarwin()) {
            suggest = false;
        } else {
            var lock_limit: std.c.rlimit = undefined;
            if (suggest and std.c.getrlimit(.MEMLOCK, &lock_limit) != 0) suggest = false;
            if (suggest and @as(u64, lock_limit.max) > @as(u64, lock_limit.cur) +% size) suggest = false;
        }

        impl.logWarn("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s", .{
            size, self.size, errmsg, @as([*:0]const u8, if (suggest) mlock_suggestion else ""),
        });
        return false;
    }

    /// Ports the `MLOCK_SUGGESTION` macro (llama-mmap.cpp:938 @b1b770d90): its
    /// `__APPLE__` arm, and the `#else` four lines below it.
    const mlock_suggestion: [*:0]const u8 = if (builtin.os.tag.isDarwin())
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " ++
            "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
    else
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n";

    /// Ports `llama_mlock`'s `impl`'s `init` (llama-mmap.cpp:1028 @b1b770d90).
    ///
    /// Return: nothing.
    pub fn init(self: *Mlock, ptr: *anyopaque) void {
        impl.assert(self.addr == null and self.size == 0, "llama-mmap.cpp", 1029, "addr == NULL && size == 0");
        self.addr = ptr;
    }

    /// Ports `llama_mlock`'s `impl`'s `grow_to` (llama-mmap.cpp:1033 @b1b770d90):
    /// lock up to `target_size` bytes, rounded up to a page, and give up for
    /// good after the first failure.
    ///
    /// Return: nothing.
    pub fn growTo(self: *Mlock, target_size_: usize) void {
        impl.assert(self.addr != null, "llama-mmap.cpp", 1034, "addr");
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

/// Ports `llama_memory_range` (llama-mmap.h:82 @b1b770d90).
pub const MemoryRange = struct {
    addr: ?*const anyopaque,
    size: usize,
};

/// `MADV_WILLNEED`, the same on Linux's every architecture and on Darwin.
const MADV_WILLNEED: c_int = 3;

fn lessAddr(_: void, a: MemoryRange, b: MemoryRange) bool {
    return @intFromPtr(a.addr) < @intFromPtr(b.addr);
}

/// Ports `llama_prefetch` (llama-mmap.cpp:1067 @b1b770d90): advise the system
/// that the host pages covering these ranges will be needed. The ranges are
/// first merged into runs of pages, written back over the front of `mr`,
/// then prefetched: on Windows in one call, so the reads can be issued
/// concurrently, elsewhere one `madvise` per run. Linux, Darwin and Windows;
/// elsewhere it does nothing, as the C++ compiles it.
///
/// Parameters:
/// - `mr`: the ranges; sorted and merged in place.
///
/// Return: nothing. A failure is logged as a warning.
pub fn prefetchRanges(mr: []MemoryRange) void {
    if (comptime !(is_linux or is_windows or builtin.os.tag.isDarwin())) return;
    if (mr.len == 0) return;

    // Not `?win32.PrefetchVirtualMemoryFn` everywhere: its `callconv(.winapi)`
    // is `aarch64_aapcs_win` on aarch64 Linux, which LLVM cannot lower.
    var pvm: if (is_windows) ?win32.PrefetchVirtualMemoryFn else void = if (is_windows) null else {};
    if (comptime is_windows) {
        const kernel32 = win32.GetModuleHandleW(std.unicode.utf8ToUtf16LeStringLiteral("kernel32.dll"));
        const sym = if (kernel32) |k| win32.GetProcAddress(k, "PrefetchVirtualMemory") else null;
        pvm = @ptrCast(sym orelse return);
    }

    const page_size: c_long = if (comptime is_windows) blk: {
        var info: win32.SYSTEM_INFO = undefined;
        win32.GetSystemInfo(&info);
        break :blk @intCast(info.dwPageSize);
    } else libc.sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;

    const page: usize = @intCast(page_size);
    std.sort.pdq(MemoryRange, mr, {}, lessAddr);

    var begin: usize = 0;
    var end: usize = 0;
    var n_ranges: usize = 0;
    for (mr) |range| {
        if (range.addr == null or range.size == 0) continue;
        const pointer = @intFromPtr(range.addr);
        const first = pointer / page * page;
        const last = (pointer + range.size + page - 1) / page * page;
        if (end != 0 and first > end) {
            mr[n_ranges] = .{ .addr = @ptrFromInt(begin), .size = end - begin };
            n_ranges += 1;
            end = 0;
        }
        if (end == 0) begin = first;
        end = @max(end, last);
    }
    if (end != 0) {
        mr[n_ranges] = .{ .addr = @ptrFromInt(begin), .size = end - begin };
        n_ranges += 1;
    }
    const merged = mr[0..n_ranges];

    if (comptime is_windows) {
        // the merged ranges in one call, so the reads can be issued concurrently
        var entries: std.ArrayList(win32.WIN32_MEMORY_RANGE_ENTRY) = .empty;
        defer entries.deinit(allocator);
        entries.ensureTotalCapacity(allocator, merged.len) catch impl.abort("llama-mmap.cpp", 1121, "out of memory");
        for (merged) |range| {
            entries.appendAssumeCapacity(.{ .VirtualAddress = @constCast(range.addr), .NumberOfBytes = range.size });
        }
        if (entries.items.len != 0 and pvm.?(win32.GetCurrentProcess(), entries.items.len, entries.items.ptr, 0) == 0) {
            warnWin32("llama_prefetch: PrefetchVirtualMemory failed: %s\n");
        }
    } else {
        for (merged) |range| {
            if (libc.madvise(@constCast(range.addr), range.size, MADV_WILLNEED) != 0) {
                impl.logWarn("llama_prefetch: libc.madvise(MADV_WILLNEED) failed: %s\n", .{stdlib.errno.strerror(errnoPtr().*)});
                return;
            }
        }
    }
}

/// Ports `llama_path_max` (llama-mmap.cpp:1144 @b1b770d90).
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

test "ranges_complement merges overlaps and clamps to the limit" {
    var res = rangesComplement(&.{ .{ 30, 40 }, .{ 10, 20 }, .{ 15, 25 }, .{ 90, 200 } }, 100);
    defer res.deinit(allocator);
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 10 }, .{ 25, 30 }, .{ 40, 90 } }, res.items);

    var none = rangesComplement(&.{}, 64);
    defer none.deinit(allocator);
    try std.testing.expectEqualSlices([2]usize, &.{.{ 0, 64 }}, none.items);

    var empty = rangesComplement(&.{.{ 0, 64 }}, 0);
    defer empty.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), empty.items.len);
}
