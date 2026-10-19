//! The Win32 and CRT calls libllama's `_WIN32` arms make.
//!
//! Not a port of anything: declarations, as `src/ggml/win32.zig` is for ggml
//! -- this module cannot import that one. Zig 0.17's `std.os.windows` has the
//! types but almost none of the kernel32 functions, so the ones the C++ calls
//! are declared here with their documented signatures, and the constants are
//! copied from the MinGW headers Zig ships (`lib/libc/include/any-windows-any/`),
//! which is what the reference compiles against. The files that call them cite
//! the C++.
//!
//! Every declaration is referenced only from a `builtin.os.tag == .windows`
//! branch, so none of it is analysed on any other target.

const std = @import("std");
const windows = std.os.windows;

pub const BOOL = c_int;
pub const DWORD = windows.DWORD;
pub const HANDLE = windows.HANDLE;
pub const HMODULE = windows.HMODULE;
pub const SRWLOCK = windows.SRWLOCK;

/// `FORMAT_MESSAGE_ALLOCATE_BUFFER`, `FORMAT_MESSAGE_IGNORE_INSERTS` and
/// `FORMAT_MESSAGE_FROM_SYSTEM` (MinGW's `winbase.h`, lines 1457, 1451 and 1454).
pub const FORMAT_MESSAGE_ALLOCATE_BUFFER: DWORD = 0x100;
pub const FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x200;
pub const FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x1000;
/// `MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT)` (MinGW's `winnt.h`, lines 1387, 959 and 1102).
pub const LANG_NEUTRAL_SUBLANG_DEFAULT: DWORD = (0x01 << 10) | 0x00;
/// `FILE_CURRENT` (MinGW's `winbase.h`, line 70). `FILE_BEGIN` and `FILE_END` equal
/// `SEEK_SET` and `SEEK_END`, which the C++ asserts.
pub const FILE_CURRENT: DWORD = 1;
/// `PAGE_READONLY` (MinGW's `winnt.h`, line 5912) and `FILE_MAP_READ`, which is
/// `SECTION_MAP_READ` (MinGW's `memoryapi.h`, line 44, and `winnt.h`, line 5899).
pub const PAGE_READONLY: DWORD = 0x02;
pub const FILE_MAP_READ: DWORD = 0x0004;

/// `SYSTEM_INFO` (MinGW's `sysinfoapi.h`, line 18), its leading union flattened to the
/// `dwOemId` member.
pub const SYSTEM_INFO = extern struct {
    dwOemId: DWORD,
    dwPageSize: DWORD,
    lpMinimumApplicationAddress: ?*anyopaque,
    lpMaximumApplicationAddress: ?*anyopaque,
    dwActiveProcessorMask: usize,
    dwNumberOfProcessors: DWORD,
    dwProcessorType: DWORD,
    dwAllocationGranularity: DWORD,
    wProcessorLevel: u16,
    wProcessorRevision: u16,
};

/// `WIN32_MEMORY_RANGE_ENTRY`, from MinGW's `memoryapi.h`.
pub const WIN32_MEMORY_RANGE_ENTRY = extern struct {
    VirtualAddress: ?*anyopaque,
    NumberOfBytes: usize,
};

/// `PrefetchVirtualMemory`'s type, which the C++ looks up at run time.
pub const PrefetchVirtualMemoryFn = *const fn (process: HANDLE, count: usize, ranges: [*]WIN32_MEMORY_RANGE_ENTRY, flags: windows.ULONG) callconv(.winapi) BOOL;

// kernel32
pub extern "kernel32" fn GetLastError() callconv(.winapi) DWORD;
pub extern "kernel32" fn FormatMessageA(flags: DWORD, source: ?*const anyopaque, message_id: DWORD, language_id: DWORD, buffer: *?[*]u8, size: DWORD, args: ?*anyopaque) callconv(.winapi) DWORD;
pub extern "kernel32" fn LocalFree(mem: ?*anyopaque) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn SetFilePointerEx(file: HANDLE, distance: i64, new_pointer: ?*i64, method: DWORD) callconv(.winapi) BOOL;
pub extern "kernel32" fn ReadFile(file: HANDLE, buffer: [*]u8, to_read: DWORD, read: ?*DWORD, overlapped: ?*anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn WriteFile(file: HANDLE, buffer: [*]const u8, to_write: DWORD, written: ?*DWORD, overlapped: ?*anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn CreateFileMappingA(file: HANDLE, attributes: ?*anyopaque, protect: DWORD, size_high: DWORD, size_low: DWORD, name: ?[*:0]const u8) callconv(.winapi) ?HANDLE;
pub extern "kernel32" fn MapViewOfFile(mapping: HANDLE, access: DWORD, offset_high: DWORD, offset_low: DWORD, size: usize) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn UnmapViewOfFile(addr: *const anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn CloseHandle(handle: HANDLE) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetModuleHandleW(name: [*:0]const u16) callconv(.winapi) ?HMODULE;
pub extern "kernel32" fn GetProcAddress(module: HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn GetCurrentProcess() callconv(.winapi) HANDLE;
pub extern "kernel32" fn GetSystemInfo(info: *SYSTEM_INFO) callconv(.winapi) void;
pub extern "kernel32" fn VirtualLock(addr: *anyopaque, size: usize) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetProcessWorkingSetSize(process: HANDLE, min: *usize, max: *usize) callconv(.winapi) BOOL;
pub extern "kernel32" fn SetProcessWorkingSetSize(process: HANDLE, min: usize, max: usize) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetSystemTimePreciseAsFileTime(ft: *windows.FILETIME) callconv(.winapi) void;
pub extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.winapi) void;
pub extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.winapi) void;

// The UCRT.
pub extern "c" fn _fileno(stream: *anyopaque) c_int;
pub extern "c" fn _get_osfhandle(fd: c_int) isize;
pub extern "c" fn _ftelli64(stream: *anyopaque) i64;
pub extern "c" fn _fseeki64(stream: *anyopaque, offset: i64, whence: c_int) c_int;
