//! Stage 8 workstream build file: `mtmd_fixtures`. Owned by that workstream; wired
//! into `build.zig` through `build/mtmd.zig`'s `Gates`.
//!
//! Adds `mmproj-fixtures`: `harness/mmproj_fixtures.cpp`, which writes a small
//! random mmproj for every projector type and the text models they load
//! against, and validates each through upstream's libmtmd. It links
//! `libmtmd-ref`, `libllama-ref` and our `libggml`, the same libraries as the
//! reference tools.
//!
//! **Not a port.** This is build description.

const std = @import("std");
const mtmd = @import("mtmd.zig");
const llamacpp = @import("llamacpp.zig");

/// Adds this workstream's steps.
pub fn add(b: *std.Build, opts: mtmd.GateOptions) void {
    const mod = b.createModule(.{
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // reference code and its headers compile as shipped
        .sanitize_c = .off,
    });
    const root = "llama.cpp";
    mod.addIncludePath(b.path(root ++ "/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/tools/mtmd"));
    mod.addCSourceFiles(.{ .files = &.{"harness/mmproj_fixtures.cpp"}, .flags = &.{ "-std=c++17", "-w" } });
    mod.linkLibrary(opts.mtmd_ref);
    mod.linkLibrary(opts.reference.llama_ref);
    mod.linkLibrary(opts.reference.ggml);
    llamacpp.addSdkPaths(b, mod, opts.sdk);
    llamacpp.addCudaRuntime(b, mod, opts.cuda);
    const exe = b.addExecutable(.{ .name = "mmproj-fixtures", .root_module = mod });
    const step = b.step("mmproj-fixtures", "Build the mmproj fixture generator and validator (harness/mmproj_fixtures.cpp)");
    step.dependOn(&b.addInstallArtifact(exe, .{}).step);
}
