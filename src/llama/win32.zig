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

pub const BOOL = c_int;

/// `FORMAT_MESSAGE_ALLOCATE_BUFFER`, `FORMAT_MESSAGE_IGNORE_INSERTS` and
/// `FORMAT_MESSAGE_FROM_SYSTEM` (MinGW's `winbase.h`, lines 1457, 1451 and 1454).
pub const FORMAT_MESSAGE_ALLOCATE_BUFFER: std.os.windows.DWORD = 0x100;
pub const FORMAT_MESSAGE_IGNORE_INSERTS: std.os.windows.DWORD = 0x200;
pub const FORMAT_MESSAGE_FROM_SYSTEM: std.os.windows.DWORD = 0x1000;
/// `MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT)` (MinGW's `winnt.h`, lines 1387, 959 and 1102).
pub const LANG_NEUTRAL_SUBLANG_DEFAULT: std.os.windows.DWORD = (0x01 << 10) | 0x00;
/// `FILE_CURRENT` (MinGW's `winbase.h`, line 70). `FILE_BEGIN` and `FILE_END` equal
/// `SEEK_SET` and `SEEK_END`, which the C++ asserts.
pub const FILE_CURRENT: std.os.windows.DWORD = 1;
/// `FILE_BEGIN` (MinGW's `winbase.h`, line 69).
pub const FILE_BEGIN: std.os.windows.DWORD = 0;
/// `INVALID_HANDLE_VALUE` (MinGW's `handleapi.h`, line 16).
pub const INVALID_HANDLE_VALUE: std.os.windows.HANDLE = @ptrFromInt(std.math.maxInt(usize));
/// `GENERIC_READ` (MinGW's `winnt.h`, line 3180).
pub const GENERIC_READ: std.os.windows.DWORD = 0x80000000;
/// `FILE_SHARE_READ`, `FILE_SHARE_WRITE` and `FILE_SHARE_DELETE` (MinGW's
/// `winnt.h`, lines 6181-6183).
pub const FILE_SHARE_READ: std.os.windows.DWORD = 0x1;
pub const FILE_SHARE_WRITE: std.os.windows.DWORD = 0x2;
pub const FILE_SHARE_DELETE: std.os.windows.DWORD = 0x4;
/// `FILE_FLAG_NO_BUFFERING` (MinGW's `winbase.h`, line 86).
pub const FILE_FLAG_NO_BUFFERING: std.os.windows.DWORD = 0x20000000;
/// `MAXDWORD` (MinGW's `winnt.h`, line 693).
pub const MAXDWORD: std.os.windows.DWORD = 0xffffffff;
/// `ERROR_SUCCESS`, `ERROR_NOT_ENOUGH_MEMORY`, `ERROR_HANDLE_EOF`,
/// `ERROR_NOT_SUPPORTED`, `ERROR_INVALID_PARAMETER`, `ERROR_NOACCESS` and
/// `ERROR_INVALID_USER_BUFFER` (MinGW's `winerror.h`, lines 177, 205, 207,
/// 236, 856 and 1401), measured with `zig cc -target x86_64-windows-gnu`.
pub const ERROR_SUCCESS: std.os.windows.DWORD = 0;
pub const ERROR_NOT_ENOUGH_MEMORY: std.os.windows.DWORD = 8;
pub const ERROR_HANDLE_EOF: std.os.windows.DWORD = 38;
pub const ERROR_NOT_SUPPORTED: std.os.windows.DWORD = 50;
pub const ERROR_INVALID_PARAMETER: std.os.windows.DWORD = 87;
pub const ERROR_NOACCESS: std.os.windows.DWORD = 998;
pub const ERROR_INVALID_USER_BUFFER: std.os.windows.DWORD = 1784;
/// `FileStorageInfo`, a `FILE_INFO_BY_HANDLE_CLASS` (MinGW's `minwinbase.h`,
/// line 159): 16, measured.
pub const FileStorageInfo: c_int = 16;

/// `FILE_STORAGE_INFO` (MinGW's `winbase.h`, line 3303): seven `ULONG`s, 28
/// bytes, measured with `zig cc -target x86_64-windows-gnu`.
pub const FILE_STORAGE_INFO = extern struct {
    LogicalBytesPerSector: std.os.windows.ULONG,
    PhysicalBytesPerSectorForAtomicity: std.os.windows.ULONG,
    PhysicalBytesPerSectorForPerformance: std.os.windows.ULONG,
    FileSystemEffectivePhysicalBytesPerSectorForAtomicity: std.os.windows.ULONG,
    Flags: std.os.windows.ULONG,
    ByteOffsetForSectorAlignment: std.os.windows.ULONG,
    ByteOffsetForPartitionAlignment: std.os.windows.ULONG,
};

/// `PAGE_READONLY` (MinGW's `winnt.h`, line 5912) and `FILE_MAP_READ`, which is
/// `SECTION_MAP_READ` (MinGW's `memoryapi.h`, line 44, and `winnt.h`, line 5899).
pub const PAGE_READONLY: std.os.windows.DWORD = 0x02;
pub const FILE_MAP_READ: std.os.windows.DWORD = 0x0004;

/// `SYSTEM_INFO` (MinGW's `sysinfoapi.h`, line 18), its leading union flattened to the
/// `dwOemId` member.
pub const SYSTEM_INFO = extern struct {
    dwOemId: std.os.windows.DWORD,
    dwPageSize: std.os.windows.DWORD,
    lpMinimumApplicationAddress: ?*anyopaque,
    lpMaximumApplicationAddress: ?*anyopaque,
    dwActiveProcessorMask: usize,
    dwNumberOfProcessors: std.os.windows.DWORD,
    dwProcessorType: std.os.windows.DWORD,
    dwAllocationGranularity: std.os.windows.DWORD,
    wProcessorLevel: u16,
    wProcessorRevision: u16,
};

/// `WIN32_MEMORY_RANGE_ENTRY`, from MinGW's `memoryapi.h`.
pub const WIN32_MEMORY_RANGE_ENTRY = extern struct {
    VirtualAddress: ?*anyopaque,
    NumberOfBytes: usize,
};

/// `PrefetchVirtualMemory`'s type, which the C++ looks up at run time.
pub const PrefetchVirtualMemoryFn = *const fn (process: std.os.windows.HANDLE, count: usize, ranges: [*]WIN32_MEMORY_RANGE_ENTRY, flags: std.os.windows.ULONG) callconv(.winapi) BOOL;

// kernel32
pub extern "kernel32" fn GetLastError() callconv(.winapi) std.os.windows.DWORD;
pub extern "kernel32" fn FormatMessageA(flags: std.os.windows.DWORD, source: ?*const anyopaque, message_id: std.os.windows.DWORD, language_id: std.os.windows.DWORD, buffer: *?[*]u8, size: std.os.windows.DWORD, args: ?*anyopaque) callconv(.winapi) std.os.windows.DWORD;
pub extern "kernel32" fn LocalFree(mem: ?*anyopaque) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn SetFilePointerEx(file: std.os.windows.HANDLE, distance: i64, new_pointer: ?*i64, method: std.os.windows.DWORD) callconv(.winapi) BOOL;
pub extern "kernel32" fn ReadFile(file: std.os.windows.HANDLE, buffer: [*]u8, to_read: std.os.windows.DWORD, read: ?*std.os.windows.DWORD, overlapped: ?*anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn WriteFile(file: std.os.windows.HANDLE, buffer: [*]const u8, to_write: std.os.windows.DWORD, written: ?*std.os.windows.DWORD, overlapped: ?*anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn CreateFileMappingA(file: std.os.windows.HANDLE, attributes: ?*anyopaque, protect: std.os.windows.DWORD, size_high: std.os.windows.DWORD, size_low: std.os.windows.DWORD, name: ?[*:0]const u8) callconv(.winapi) ?std.os.windows.HANDLE;
pub extern "kernel32" fn MapViewOfFile(mapping: std.os.windows.HANDLE, access: std.os.windows.DWORD, offset_high: std.os.windows.DWORD, offset_low: std.os.windows.DWORD, size: usize) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn UnmapViewOfFile(addr: *const anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn ReOpenFile(original: std.os.windows.HANDLE, access: std.os.windows.DWORD, share: std.os.windows.DWORD, flags: std.os.windows.DWORD) callconv(.winapi) std.os.windows.HANDLE;
pub extern "kernel32" fn GetFileInformationByHandleEx(file: std.os.windows.HANDLE, class: c_int, info: *anyopaque, size: std.os.windows.DWORD) callconv(.winapi) BOOL;
pub extern "kernel32" fn CloseHandle(handle: std.os.windows.HANDLE) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetModuleHandleW(name: [*:0]const u16) callconv(.winapi) ?std.os.windows.HMODULE;
pub extern "kernel32" fn GetProcAddress(module: std.os.windows.HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn GetCurrentProcess() callconv(.winapi) std.os.windows.HANDLE;
pub extern "kernel32" fn GetSystemInfo(info: *SYSTEM_INFO) callconv(.winapi) void;
pub extern "kernel32" fn VirtualLock(addr: *anyopaque, size: usize) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetProcessWorkingSetSize(process: std.os.windows.HANDLE, min: *usize, max: *usize) callconv(.winapi) BOOL;
pub extern "kernel32" fn SetProcessWorkingSetSize(process: std.os.windows.HANDLE, min: usize, max: usize) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetSystemTimePreciseAsFileTime(ft: *std.os.windows.FILETIME) callconv(.winapi) void;
pub extern "kernel32" fn AcquireSRWLockExclusive(lock: *std.os.windows.SRWLOCK) callconv(.winapi) void;
pub extern "kernel32" fn ReleaseSRWLockExclusive(lock: *std.os.windows.SRWLOCK) callconv(.winapi) void;

// The UCRT.
