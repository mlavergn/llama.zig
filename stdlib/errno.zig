//! `errno` and its text: what upstream prints with `strerror(errno)` and
//! `std::error_code::message()`, for an error std reports by name.
//!
//! # Provenance
//!
//! **Not a port.** std turns `errno` into named errors; `of` turns them
//! back, so a message reads as upstream's. `strerror` is the C library's
//! table, read in this one place: std has no table of its own, and the
//! words differ between C libraries.

const std = @import("std");
const mod = @import("module.zig");

/// `strerror(errnum)`: the C library's text for an `errno`.
///
/// Parameters:
/// - `errnum`: an `errno` value.
///
/// Return: static, or valid until the next call.
pub fn strerror(errnum: c_int) [*:0]const u8 {
    return mod.libc.strerror(errnum);
}

/// The `errno` a file error came from, for upstream's `strerror(errno)`.
/// std keeps most apart; where it folds several into one error, the one a
/// model or data file meets: `EINVAL` for a bad name, `ENXIO` for no device,
/// `EFBIG` for too large, `ENOENT` for not found.
///
/// Return: the `errno`, or null for an error no `errno` stands behind.
pub fn of(err: anyerror) ?c_int {
    const E = std.c.E;
    const e: E = switch (err) {
        error.FileNotFound => .NOENT,
        error.AccessDenied => .ACCES,
        error.PermissionDenied => .PERM,
        error.IsDir => .ISDIR,
        error.NotDir => .NOTDIR,
        error.SymLinkLoop => .LOOP,
        error.NameTooLong => .NAMETOOLONG,
        error.BadPathName => .INVAL,
        error.ProcessFdQuotaExceeded => .MFILE,
        error.SystemFdQuotaExceeded => .NFILE,
        error.SystemResources => .NOMEM,
        error.NoSpaceLeft => .NOSPC,
        error.FileTooBig => .FBIG,
        error.DeviceBusy => .BUSY,
        error.FileBusy => .TXTBSY,
        error.PathAlreadyExists => .EXIST,
        error.NoDevice => .NXIO,
        error.ReadOnlyFileSystem => .ROFS,
        error.WouldBlock => .AGAIN,
        error.FileLocksUnsupported => .OPNOTSUPP,
        error.InputOutput => .IO,
        error.BrokenPipe => .PIPE,
        error.DiskQuota => .DQUOT,
        error.DirNotEmpty => .NOTEMPTY,
        error.CrossDevice => .XDEV,
        error.LinkQuotaExceeded => .MLINK,
        error.NotLink => .INVAL,
        error.FileSystem, error.HardwareFailure => .IO,
        error.OperationUnsupported => .OPNOTSUPP,
        else => return null,
    };
    return @backingInt(e);
}

/// What upstream prints for a file error: `strerror(errno)`, through
/// `of`; the error's name when no `errno` stands behind it.
///
/// Return: the text, static or the C library's; valid until the next call.
pub fn text(err: anyerror) [*:0]const u8 {
    if (of(err)) |n| return strerror(n);
    return @errorName(err);
}

/// The `errno` behind `err`, for code that answers in `errno` as the C
/// did: `ENOMEM` for running out of memory, `EIO` where no `errno` stands
/// behind the error.
///
/// Return: a positive `errno`.
pub fn from(err: anyerror) c_int {
    if (err == error.OutOfMemory) return @backingInt(std.c.E.NOMEM);
    return of(err) orelse @backingInt(std.c.E.IO);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an error maps back to its errno, and to the C library's words" {
    try std.testing.expectEqual(@as(?c_int, @backingInt(std.c.E.NOENT)), of(error.FileNotFound));
    try std.testing.expectEqual(@as(?c_int, null), of(error.Unexpected));
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.NOMEM)), from(error.OutOfMemory));
    try std.testing.expectEqual(@as(c_int, @backingInt(std.c.E.IO)), from(error.Unexpected));
    try std.testing.expectEqualStrings("Unexpected", std.mem.span(text(error.Unexpected)));
}
