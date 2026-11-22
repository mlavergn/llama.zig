//! Whether the standard streams are a terminal that takes colour.
//!
//! # Provenance
//!
//! Ports `common_is_tty`, `tty_can_use_colors` and `tty_enable_ansi` from
//! `llama.cpp/common/common.cpp` (v1.0.0, `b1b770d90`), which the CLI and
//! the tools both call. Each declaration names what it ports and the line
//! it began at. `NO_COLOR` and `TERM` are read from `env`, the environment
//! `main` handed over.

const std = @import("std");
const builtin = @import("builtin");
const file = @import("file.zig");
const env = @import("env.zig");

/// Ports `common_is_tty` (common/common.cpp:1032 @b1b770d90): whether
/// `f` is attached to a terminal.
pub fn isTty(f: std.Io.File) bool {
    return f.isTty(file.defaultIo()) catch false;
}

const win = struct {
    const HANDLE = ?*anyopaque;
    extern "kernel32" fn GetStdHandle(id: u32) callconv(.winapi) HANDLE;
    extern "kernel32" fn GetConsoleMode(h: HANDLE, mode: *u32) callconv(.winapi) c_int;
    extern "kernel32" fn SetConsoleMode(h: HANDLE, mode: u32) callconv(.winapi) c_int;
    const STD_OUTPUT_HANDLE: u32 = @bitCast(@as(i32, -11));
    const STD_ERROR_HANDLE: u32 = @bitCast(@as(i32, -12));
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
};

/// Ports `tty_enable_ansi` (common/common.cpp:1060 @b1b770d90): false when
/// stdout or stderr is a console that cannot render ANSI sequences.
pub fn enableAnsi() bool {
    if (comptime builtin.os.tag == .windows) {
        // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
        for ([_]u32{ win.STD_OUTPUT_HANDLE, win.STD_ERROR_HANDLE }) |id| {
            const h = win.GetStdHandle(id);
            var mode: u32 = 0;
            if (win.GetConsoleMode(h, &mode) != 0 and win.SetConsoleMode(h, mode | win.ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0) {
                return false;
            }
        }
    }
    return true;
}

/// Ports `tty_can_use_colors` (common/common.cpp:1040 @b1b770d90).
pub fn canUseColors() bool {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (env.get("NO_COLOR")) |no_color| {
        if (no_color.len != 0) return false;
    }
    // Check TERM environment variable
    if (env.get("TERM")) |term| {
        if (std.mem.eql(u8, term, "dumb")) return false;
    }
    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    return isTty(std.Io.File.stdout()) or isTty(std.Io.File.stderr());
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
