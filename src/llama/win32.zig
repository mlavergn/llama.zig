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
/// `FILE_BEGIN` (MinGW's `winbase.h`, line 69).
pub const FILE_BEGIN: DWORD = 0;
/// `INVALID_HANDLE_VALUE` (MinGW's `handleapi.h`, line 16).
pub const INVALID_HANDLE_VALUE: HANDLE = @ptrFromInt(std.math.maxInt(usize));
/// `GENERIC_READ` (MinGW's `winnt.h`, line 3180).
pub const GENERIC_READ: DWORD = 0x80000000;
/// `FILE_SHARE_READ`, `FILE_SHARE_WRITE` and `FILE_SHARE_DELETE` (MinGW's
/// `winnt.h`, lines 6181-6183).
pub const FILE_SHARE_READ: DWORD = 0x1;
pub const FILE_SHARE_WRITE: DWORD = 0x2;
pub const FILE_SHARE_DELETE: DWORD = 0x4;
/// `FILE_FLAG_NO_BUFFERING` (MinGW's `winbase.h`, line 86).
pub const FILE_FLAG_NO_BUFFERING: DWORD = 0x20000000;
/// `MAXDWORD` (MinGW's `winnt.h`, line 693).
pub const MAXDWORD: DWORD = 0xffffffff;
/// `ERROR_SUCCESS`, `ERROR_NOT_ENOUGH_MEMORY`, `ERROR_HANDLE_EOF`,
/// `ERROR_NOT_SUPPORTED`, `ERROR_INVALID_PARAMETER`, `ERROR_NOACCESS` and
/// `ERROR_INVALID_USER_BUFFER` (MinGW's `winerror.h`, lines 177, 205, 207,
/// 236, 856 and 1401), measured with `zig cc -target x86_64-windows-gnu`.
pub const ERROR_SUCCESS: DWORD = 0;
pub const ERROR_NOT_ENOUGH_MEMORY: DWORD = 8;
pub const ERROR_HANDLE_EOF: DWORD = 38;
pub const ERROR_NOT_SUPPORTED: DWORD = 50;
pub const ERROR_INVALID_PARAMETER: DWORD = 87;
pub const ERROR_NOACCESS: DWORD = 998;
pub const ERROR_INVALID_USER_BUFFER: DWORD = 1784;
/// `FileStorageInfo`, a `FILE_INFO_BY_HANDLE_CLASS` (MinGW's `minwinbase.h`,
/// line 159): 16, measured.
pub const FileStorageInfo: c_int = 16;

/// `FILE_STORAGE_INFO` (MinGW's `winbase.h`, line 3303): seven `ULONG`s, 28
/// bytes, measured with `zig cc -target x86_64-windows-gnu`.
pub const FILE_STORAGE_INFO = extern struct {
    LogicalBytesPerSector: windows.ULONG,
    PhysicalBytesPerSectorForAtomicity: windows.ULONG,
    PhysicalBytesPerSectorForPerformance: windows.ULONG,
    FileSystemEffectivePhysicalBytesPerSectorForAtomicity: windows.ULONG,
    Flags: windows.ULONG,
    ByteOffsetForSectorAlignment: windows.ULONG,
    ByteOffsetForPartitionAlignment: windows.ULONG,
};

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
pub extern "kernel32" fn ReOpenFile(original: HANDLE, access: DWORD, share: DWORD, flags: DWORD) callconv(.winapi) HANDLE;
pub extern "kernel32" fn GetFileInformationByHandleEx(file: HANDLE, class: c_int, info: *anyopaque, size: DWORD) callconv(.winapi) BOOL;
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
