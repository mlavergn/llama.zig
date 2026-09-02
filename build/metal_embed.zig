//! Build-time tool that prepares one Metal kernel source for embedding.
//!
//! Replaces the `cat`/`sed` pipeline that llama.cpp's CMake uses when
//! `GGML_METAL_EMBED_LIBRARY` is on. The Metal driver compiles the shader
//! source at runtime, so nothing here invokes `xcrun metal`; the job is purely
//! to flatten a kernel and its headers into one self-contained MSL file, then
//! emit an assembly stub that pulls that file into the binary.
//!
//! Usage:
//!   metal_embed <kind> <out.metal> <out.s> <ggml-common.h> <ggml-metal-impl.h>
//!               <kernels/common.h> <kernels/dequantize.h> <kernels/quantize.h>
//!               <kernels/{kind}.metal>

const std = @import("std");

/// Header includes that are inlined rather than resolved by a compiler, and so
/// must be stripped from the flattened output.
const stripped_includes = [_][]const u8{
    "#include \"common.h\"",
    "#include \"dequantize.h\"",
    "#include \"quantize.h\"",
};

/// Sentinel in `kernels/dequantize.h` marking where `ggml-common.h` belongs.
const common_sentinel = "__embed_ggml-common.h__";

/// Include directive marking where `ggml-metal-impl.h` belongs.
const impl_include = "#include \"ggml-metal-impl.h\"";

/// Entry point: flattens one kernel and writes the `.metal` and `.s` outputs.
///
/// Parameters:
/// - `init`: process capabilities supplied by the runtime (allocator, IO, args).
///
/// Return: nothing on success; propagates read, write, and allocation failures,
/// and returns `error.BadUsage` when the argument count is wrong.
pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 10) return error.BadUsage;

    const kind = args[1];
    const out_metal = args[2];
    const out_asm = args[3];
    const cwd = std.Io.Dir.cwd();

    const limit: std.Io.Limit = .limited(64 << 20);
    const ggml_common = try cwd.readFileAlloc(io, args[4], gpa, limit);
    defer gpa.free(ggml_common);
    const metal_impl = try cwd.readFileAlloc(io, args[5], gpa, limit);
    defer gpa.free(metal_impl);
    const kernels_common = try cwd.readFileAlloc(io, args[6], gpa, limit);
    defer gpa.free(kernels_common);
    const dequantize = try cwd.readFileAlloc(io, args[7], gpa, limit);
    defer gpa.free(dequantize);
    const quantize = try cwd.readFileAlloc(io, args[8], gpa, limit);
    defer gpa.free(quantize);
    const source = try cwd.readFileAlloc(io, args[9], gpa, limit);
    defer gpa.free(source);

    // Only prepend the headers this kernel actually includes. Prepending all of
    // them would compile, but every kernel would carry the full dequantize and
    // quantize tables, and the driver recompiles this text on every load.
    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(gpa);

    try joined.appendSlice(gpa, kernels_common);
    try joined.append(gpa, '\n');
    if (std.mem.indexOf(u8, source, "#include \"dequantize.h\"") != null) {
        try joined.appendSlice(gpa, dequantize);
        try joined.append(gpa, '\n');
    }
    if (std.mem.indexOf(u8, source, "#include \"quantize.h\"") != null) {
        try joined.appendSlice(gpa, quantize);
        try joined.append(gpa, '\n');
    }
    try joined.appendSlice(gpa, source);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);

    var lines = std.mem.splitScalar(u8, joined.items, '\n');
    while (lines.next()) |line| {
        // `#pragma once` is meaningless once the headers are concatenated, and
        // would make the driver reject the combined source.
        if (std.mem.indexOf(u8, line, "#pragma once") != null) continue;

        if (containsAny(line, &stripped_includes)) continue;

        // The two inlined headers are substituted in place, so a reader of the
        // generated file sees them where the original expected them.
        if (std.mem.indexOf(u8, line, common_sentinel) != null) {
            try out.appendSlice(gpa, ggml_common);
            try out.append(gpa, '\n');
            continue;
        }
        if (std.mem.indexOf(u8, line, impl_include) != null) {
            try out.appendSlice(gpa, metal_impl);
            try out.append(gpa, '\n');
            continue;
        }

        try out.appendSlice(gpa, line);
        try out.append(gpa, '\n');
    }

    try cwd.writeFile(io, .{ .sub_path = out_metal, .data = out.items });

    // `-` is not legal in a C identifier, so the symbol stem replaces it. The
    // Mach-O section name is capped at 16 characters, so all kinds share one
    // section and only the global symbols vary.
    const sym = try gpa.dupe(u8, kind);
    defer gpa.free(sym);
    std.mem.replaceScalar(u8, sym, '-', '_');

    var stub: std.ArrayList(u8) = .empty;
    defer stub.deinit(gpa);
    try stub.print(gpa,
        \\.section __DATA,__ggml_metallib
        \\.globl _ggml_metallib_{s}_start
        \\_ggml_metallib_{s}_start:
        \\.incbin "{s}"
        \\.globl _ggml_metallib_{s}_end
        \\_ggml_metallib_{s}_end:
        \\
    , .{ sym, sym, out_metal, sym, sym });

    try cwd.writeFile(io, .{ .sub_path = out_asm, .data = stub.items });
}

/// Reports whether `line` contains any of `needles`.
///
/// Parameters:
/// - `line`: the text to search; borrowed for the call only.
/// - `needles`: substrings to look for.
///
/// Return: true when at least one needle occurs in `line`.
fn containsAny(line: []const u8, needles: []const []const u8) bool {
    for (needles) |needle| {
        if (std.mem.indexOf(u8, line, needle) != null) return true;
    }
    return false;
}
