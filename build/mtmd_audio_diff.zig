//! Stage 8 workstream build file: `mtmd_audio_diff`, the gate for
//! `src/mtmd/audio.zig` against upstream's `tools/mtmd/mtmd-audio.cpp`.
//!
//! The reference is `mtmd-audio.cpp` compiled unmodified here with the flags
//! `libmtmd-ref` compiles it with (`build/tools.zig`, `cxx_flags`), beside
//! `harness/mtmd_audio_ref.cpp`, which supplies the two clip.cpp symbols it
//! reaches so the rest of libmtmd stays out of the link.
//!
//! **Not a port.** This is build description.

const std = @import("std");
const mtmd = @import("mtmd.zig");
const llamacpp = @import("llamacpp.zig");

const root = "llama.cpp";

/// `build/tools.zig`'s `cxx_flags`: what `libmtmd-ref` compiles with.
const cxx_flags = [_][]const u8{ "-std=c++17", "-w" };

/// Adds `mtmd-audio-diff`.
pub fn add(b: *std.Build, opts: mtmd.GateOptions) void {
    const mod = b.createModule(.{
        .root_source_file = b.path("harness/mtmd_audio_diff.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // Reference code compiles as shipped; see `addLlamaRef`.
        .sanitize_c = .off,
    });
    // The port, rooted at `src/mtmd/audio.zig` itself rather than the whole
    // `mtmd` module: it reaches only `impl.zig`, `model.zig` and `names.zig`
    // beside it, so this gate builds whatever state the rest of Stage 8 is in.
    const audio_mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/audio.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
    });
    audio_mod.addImport("llama", opts.reference.llama.root_module);
    mod.addImport("mtmd_audio", audio_mod);
    for ([_][]const u8{ "", "/include", "/common", "/ggml/include", "/ggml/src", "/vendor", "/tools/mtmd" }) |sub| {
        mod.addIncludePath(b.path(b.fmt("{s}{s}", .{ root, sub })));
    }
    mod.addCSourceFiles(.{ .root = b.path(root), .files = &.{"tools/mtmd/mtmd-audio.cpp"}, .flags = &cxx_flags });
    mod.addCSourceFiles(.{ .files = &.{"harness/mtmd_audio_ref.cpp"}, .flags = &cxx_flags });
    mod.linkLibrary(opts.reference.ggml);
    llamacpp.addSdkPaths(b, mod, opts.sdk);

    const exe = b.addExecutable(.{ .name = "mtmd-audio-diff", .root_module = mod });
    const install = b.addInstallArtifact(exe, .{});
    const step = b.step("mtmd-audio-diff", "Build the mtmd audio preprocessing gate (scripts/mtmd-audio-diff runs it)");
    step.dependOn(&install.step);
}
