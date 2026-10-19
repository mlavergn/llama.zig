//! The Win32 and CRT calls ggml's `_WIN32` arms make.
//!
//! Not a port of anything: declarations. Zig 0.17's `std.os.windows` has the
//! types -- `SRWLOCK`, `CONDITION_VARIABLE`, `INIT_ONCE` -- but almost none of
//! the kernel32 functions, so the ones the C calls are declared here with
//! their documented signatures, and the constants are copied from the MinGW
//! headers Zig ships (`lib/libc/include/any-windows-any/`), which is what the
//! reference compiles against. The files that call them cite the C.
//!
//! Every declaration is referenced only from a `builtin.os.tag == .windows`
//! branch, so none of it is analysed on any other target.

const std = @import("std");
const windows = std.os.windows;

pub const BOOL = c_int;
pub const DWORD = windows.DWORD;
pub const HANDLE = windows.HANDLE;
pub const HMODULE = windows.HMODULE;
pub const HKEY = windows.HKEY;
pub const SRWLOCK = windows.SRWLOCK;
pub const CONDITION_VARIABLE = windows.CONDITION_VARIABLE;
pub const INIT_ONCE = windows.INIT_ONCE;

/// `INFINITE` (MinGW's `winbase.h`, line 518).
pub const INFINITE: DWORD = 0xffffffff;
/// `CONDITION_VARIABLE_LOCKMODE_SHARED` (MinGW's `synchapi.h`, line 29, and `winnt.h`).
pub const CONDITION_VARIABLE_LOCKMODE_SHARED: windows.ULONG = 0x1;
/// `CP_UTF8` (MinGW's `winnls.h`, line 184).
pub const CP_UTF8: c_uint = 65001;
/// `MAX_PATH` (MinGW's `minwindef.h`, line 33).
pub const MAX_PATH = 260;
/// `SEM_FAILCRITICALERRORS` (MinGW's `winbase.h`, line 1409).
pub const SEM_FAILCRITICALERRORS: c_uint = 0x0001;
/// `ERROR_SUCCESS` (MinGW's `winerror.h`, line 167).
pub const ERROR_SUCCESS: c_long = 0;
/// `HKEY_LOCAL_MACHINE` (MinGW's `winreg.h`, line 48): `(HKEY)(ULONG_PTR)(LONG)0x80000002`,
/// sign-extended to the pointer's width.
pub const HKEY_LOCAL_MACHINE: HKEY = @ptrFromInt(@as(usize, @bitCast(@as(isize, @as(i32, @bitCast(@as(u32, 0x80000002)))))));
/// `KEY_READ` (MinGW's `winnt.h`, line 10001): `(STANDARD_RIGHTS_READ | KEY_QUERY_VALUE |
/// KEY_ENUMERATE_SUB_KEYS | KEY_NOTIFY) & ~SYNCHRONIZE`.
pub const KEY_READ: DWORD = (0x00020000 | 0x0001 | 0x0008 | 0x0010) & ~@as(DWORD, 0x00100000);

/// The `THREAD_PRIORITY_*` levels (MinGW's `winbase.h`, lines 448-455, and `winnt.h`, lines 4665-4667).
pub const THREAD_PRIORITY_BELOW_NORMAL: c_int = -1;
pub const THREAD_PRIORITY_NORMAL: c_int = 0;
pub const THREAD_PRIORITY_ABOVE_NORMAL: c_int = 1;
pub const THREAD_PRIORITY_HIGHEST: c_int = 2;
pub const THREAD_PRIORITY_TIME_CRITICAL: c_int = 15;

/// `ThreadPowerThrottling`, the fourth `THREAD_INFORMATION_CLASS`
/// (MinGW's `winbase.h`, line 1222).
pub const ThreadPowerThrottling: c_int = 3;
/// `THREAD_POWER_THROTTLING_CURRENT_VERSION` and
/// `THREAD_POWER_THROTTLING_EXECUTION_SPEED` (MinGW's `processthreadsapi.h`, lines 354 and 355).
pub const THREAD_POWER_THROTTLING_CURRENT_VERSION: windows.ULONG = 1;
pub const THREAD_POWER_THROTTLING_EXECUTION_SPEED: windows.ULONG = 0x1;

/// `THREAD_POWER_THROTTLING_STATE`, from MinGW's `processthreadsapi.h`.
pub const THREAD_POWER_THROTTLING_STATE = extern struct {
    Version: windows.ULONG,
    ControlMask: windows.ULONG,
    StateMask: windows.ULONG,
};

/// `MEMORYSTATUSEX` (MinGW's `sysinfoapi.h`, line 58).
pub const MEMORYSTATUSEX = extern struct {
    dwLength: DWORD,
    dwMemoryLoad: DWORD,
    ullTotalPhys: u64,
    ullAvailPhys: u64,
    ullTotalPageFile: u64,
    ullAvailPageFile: u64,
    ullTotalVirtual: u64,
    ullAvailVirtual: u64,
    ullAvailExtendedVirtual: u64,
};

/// `INVALID_FILE_ATTRIBUTES` (MinGW's `fileapi.h`, line 26), `FILE_ATTRIBUTE_DIRECTORY` and
/// `FILE_ATTRIBUTE_REPARSE_POINT` (MinGW's `winnt.h`, lines 6188 and 6194), and
/// `IO_REPARSE_TAG_SYMLINK` (MinGW's `winnt.h`, line 6437).
pub const INVALID_FILE_ATTRIBUTES: DWORD = 0xffffffff;
pub const FILE_ATTRIBUTE_DIRECTORY: DWORD = 0x10;
pub const FILE_ATTRIBUTE_REPARSE_POINT: DWORD = 0x400;
pub const IO_REPARSE_TAG_SYMLINK: DWORD = 0xA000000C;

/// `WIN32_FIND_DATAW` (MinGW's `minwinbase.h`, line 79), without the `_MAC` fields.
pub const WIN32_FIND_DATAW = extern struct {
    dwFileAttributes: DWORD,
    ftCreationTime: windows.FILETIME,
    ftLastAccessTime: windows.FILETIME,
    ftLastWriteTime: windows.FILETIME,
    nFileSizeHigh: DWORD,
    nFileSizeLow: DWORD,
    dwReserved0: DWORD,
    dwReserved1: DWORD,
    cFileName: [MAX_PATH]u16,
    cAlternateFileName: [14]u16,
};

pub const INIT_ONCE_FN = *const fn (once: *INIT_ONCE, param: ?*anyopaque, ctx: ?*?*anyopaque) callconv(.winapi) BOOL;

// kernel32
pub extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.winapi) BOOL;
pub extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.winapi) BOOL;
pub extern "kernel32" fn InitOnceExecuteOnce(once: *INIT_ONCE, func: INIT_ONCE_FN, param: ?*anyopaque, ctx: ?*?*anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn MultiByteToWideChar(code_page: c_uint, flags: DWORD, mbs: [*]const u8, mbs_len: c_int, wcs: ?[*]u16, wcs_len: c_int) callconv(.winapi) c_int;

pub extern "kernel32" fn InitializeSRWLock(lock: *SRWLOCK) callconv(.winapi) void;
pub extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.winapi) void;
pub extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.winapi) void;
pub extern "kernel32" fn AcquireSRWLockShared(lock: *SRWLOCK) callconv(.winapi) void;
pub extern "kernel32" fn ReleaseSRWLockShared(lock: *SRWLOCK) callconv(.winapi) void;
pub extern "kernel32" fn InitializeConditionVariable(cond: *CONDITION_VARIABLE) callconv(.winapi) void;
pub extern "kernel32" fn SleepConditionVariableSRW(cond: *CONDITION_VARIABLE, lock: *SRWLOCK, ms: DWORD, flags: windows.ULONG) callconv(.winapi) BOOL;
pub extern "kernel32" fn WakeAllConditionVariable(cond: *CONDITION_VARIABLE) callconv(.winapi) void;

pub extern "kernel32" fn GetCurrentThread() callconv(.winapi) HANDLE;
pub extern "kernel32" fn SetThreadAffinityMask(thread: HANDLE, mask: usize) callconv(.winapi) usize;
pub extern "kernel32" fn SetThreadPriority(thread: HANDLE, priority: c_int) callconv(.winapi) BOOL;
pub extern "kernel32" fn SetThreadInformation(thread: HANDLE, class: c_int, info: *anyopaque, size: DWORD) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetLastError() callconv(.winapi) DWORD;
pub extern "kernel32" fn Sleep(ms: DWORD) callconv(.winapi) void;

pub extern "kernel32" fn GlobalMemoryStatusEx(status: *MEMORYSTATUSEX) callconv(.winapi) BOOL;

pub extern "kernel32" fn GetModuleFileNameW(module: ?HMODULE, path: [*]u16, size: DWORD) callconv(.winapi) DWORD;
pub extern "kernel32" fn LoadLibraryW(path: [*:0]const u16) callconv(.winapi) ?HMODULE;
pub extern "kernel32" fn GetProcAddress(module: HMODULE, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn FreeLibrary(module: HMODULE) callconv(.winapi) BOOL;
pub extern "kernel32" fn SetErrorMode(mode: c_uint) callconv(.winapi) c_uint;

pub extern "kernel32" fn FindFirstFileW(pattern: [*:0]const u16, data: *WIN32_FIND_DATAW) callconv(.winapi) HANDLE;
pub extern "kernel32" fn FindNextFileW(find: HANDLE, data: *WIN32_FIND_DATAW) callconv(.winapi) BOOL;
pub extern "kernel32" fn FindClose(find: HANDLE) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetFileAttributesW(path: [*:0]const u16) callconv(.winapi) DWORD;
pub extern "kernel32" fn GetCurrentDirectoryW(size: DWORD, buf: ?[*]u16) callconv(.winapi) DWORD;

// advapi32. `TEXT(...)` is narrow without `UNICODE`, so the C's
// `RegOpenKeyEx` is the `A` form.
pub extern "advapi32" fn RegOpenKeyExA(key: HKEY, sub_key: [*:0]const u8, options: DWORD, sam: DWORD, result: *HKEY) callconv(.winapi) c_long;
pub extern "advapi32" fn RegQueryValueExA(key: HKEY, name: [*:0]const u8, reserved: ?*DWORD, kind: ?*DWORD, data: ?[*]u8, size: ?*DWORD) callconv(.winapi) c_long;
pub extern "advapi32" fn RegCloseKey(key: HKEY) callconv(.winapi) c_long;

// The UCRT.
pub extern "c" fn _aligned_malloc(size: usize, alignment: usize) ?*anyopaque;
pub extern "c" fn _aligned_free(ptr: ?*anyopaque) void;
pub extern "c" fn _wfopen(path: [*:0]const u16, mode: [*:0]const u16) ?*std.c.FILE;
/// What the `stdin`, `stdout` and `stderr` macros expand to (MinGW's `stdio.h`, line 124).
pub extern "c" fn __acrt_iob_func(index: c_uint) *anyopaque;
