//! Not a port: the forked reference child of the in-process harnesses,
//! for Windows, which has no `fork`.
//!
//! `schema_diff.zig` and `peg_diff.zig` run upstream's C++ in a child so
//! that a case which kills it -- a recursive `$ref`, a left-recursive PEG,
//! one that backtracks past the time limit -- costs that case and not the
//! run. On POSIX the child is a `fork` with a pipe. Here it is the harness
//! itself, started again as `<exe> --ref-child <mode> <in> <out>`: the parent
//! writes the bytecode to `<in>`, the child runs the reference over it and
//! writes what it returned to `<out>`, and a child that crashes or outlives
//! its time is a child that did not finish, exactly as a fork's is.
//!
//! The files sit in the working directory. The error mode is set so that a
//! crashing child exits with its exception code rather than waiting on a
//! Windows Error Reporting dialog no one will see.

const std = @import("std");

extern "kernel32" fn GetModuleFileNameA(module: ?*anyopaque, buf: [*]u8, size: u32) callconv(.winapi) u32;
extern "kernel32" fn WaitForSingleObject(handle: *anyopaque, ms: u32) callconv(.winapi) u32;
extern "kernel32" fn TerminateProcess(handle: *anyopaque, code: u32) callconv(.winapi) c_int;
extern "kernel32" fn GetExitCodeProcess(handle: *anyopaque, code: *u32) callconv(.winapi) c_int;
extern "kernel32" fn CloseHandle(handle: *anyopaque) callconv(.winapi) c_int;
extern "kernel32" fn SetErrorMode(mode: u32) callconv(.winapi) u32;
extern "c" fn _spawnv(mode: c_int, cmd: [*:0]const u8, argv: [*:null]const ?[*:0]const u8) isize;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fclose(f: *anyopaque) c_int;
extern "c" fn fread(p: [*]u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fwrite(p: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn remove(path: [*:0]const u8) c_int;

const P_NOWAIT: c_int = 1;
const WAIT_TIMEOUT: u32 = 0x102;
const SEM_FAILCRITICALERRORS: u32 = 0x0001;
const SEM_NOGPFAULTERRORBOX: u32 = 0x0002;

const gpa = std.heap.smp_allocator;

var counter: usize = 0;

/// The child's side: `argv` is the harness's own; null unless it was
/// started as a child.
///
/// Return: the mode, the input's bytes and the output path, when this
/// process is a child.
pub fn childArgs(argv: []const [:0]const u8) ?struct { mode: []const u8, input: []u8, out: [:0]const u8 } {
    if (argv.len != 5 or !std.mem.eql(u8, argv[1], "--ref-child")) return null;
    return .{ .mode = argv[2], .input = readFile(argv[3]) orelse std.process.exit(4), .out = argv[4] };
}

/// Writes `bytes` to `path`. Return: whether all of it was written.
pub fn writeFile(path: [*:0]const u8, bytes: []const u8) bool {
    const f = fopen(path, "wb") orelse return false;
    defer _ = fclose(f);
    return fwrite(bytes.ptr, 1, bytes.len, f) == bytes.len;
}

/// Reads all of `path`. Return: its bytes (owned by the caller, `gpa`), or
/// null when it cannot be opened.
pub fn readFile(path: [*:0]const u8) ?[]u8 {
    const f = fopen(path, "rb") orelse return null;
    defer _ = fclose(f);
    var buf: std.ArrayList(u8) = .empty;
    var chunk: [65536]u8 = undefined;
    while (true) {
        const n = fread(&chunk, 1, chunk.len, f);
        if (n == 0) break;
        buf.appendSlice(gpa, chunk[0..n]) catch return null;
    }
    return buf.toOwnedSlice(gpa) catch null;
}

/// Runs this executable as a child in `mode` over `input`, killing it
/// after `seconds` (0: no limit).
///
/// Parameters:
///   mode    - the child's mode, passed back by `childArgs`
///   input   - the bytes the child reads
///   out     - receives what the child wrote, when it finished
///   seconds - the time limit; 0 for none
/// Return: whether the child ran to completion and exited 0.
pub fn run(mode: [:0]const u8, input: []const u8, out: *std.ArrayList(u8), seconds: u32) bool {
    _ = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    var exe: [1024]u8 = undefined;
    const n = GetModuleFileNameA(null, &exe, exe.len - 1);
    if (n == 0 or n >= exe.len - 1) return false;
    exe[n] = 0;
    counter += 1;
    var in_buf: [64]u8 = undefined;
    var out_buf: [64]u8 = undefined;
    const in_path = std.fmt.bufPrintSentinel(&in_buf, "ref-child-{d}.in", .{counter}, 0) catch return false;
    const out_path = std.fmt.bufPrintSentinel(&out_buf, "ref-child-{d}.out", .{counter}, 0) catch return false;
    defer _ = remove(in_path.ptr);
    defer _ = remove(out_path.ptr);
    _ = remove(out_path.ptr);
    if (!writeFile(in_path.ptr, input)) return false;
    // `_spawnv` joins its arguments with spaces and quotes nothing, and the
    // executable's path has a space in it (the user's directory).
    const quoted = std.fmt.allocPrintSentinel(gpa, "\"{s}\"", .{exe[0..n]}, 0) catch return false;
    defer gpa.free(quoted);
    const args = [_:null]?[*:0]const u8{ quoted.ptr, "--ref-child", mode.ptr, in_path.ptr, out_path.ptr };
    const h = _spawnv(P_NOWAIT, @ptrCast(&exe), &args);
    if (h == -1 or h == 0) return false;
    const handle: *anyopaque = @ptrFromInt(@as(usize, @bitCast(h)));
    defer _ = CloseHandle(handle);
    if (WaitForSingleObject(handle, if (seconds == 0) 0xffffffff else seconds * 1000) == WAIT_TIMEOUT) {
        _ = TerminateProcess(handle, 124);
        _ = WaitForSingleObject(handle, 0xffffffff);
        return false;
    }
    var code: u32 = 1;
    if (GetExitCodeProcess(handle, &code) == 0 or code != 0) return false;
    const bytes = readFile(out_path.ptr) orelse return false;
    defer gpa.free(bytes);
    out.appendSlice(gpa, bytes) catch return false;
    return true;
}
