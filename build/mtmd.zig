//! libmtmd's port (Stage 8): the `mtmd` module rooted at
//! `src/mtmd/module.zig`, and its unit tests.
//!
//! **Not a port.** This is build description.

const std = @import("std");
const llamacpp = @import("llamacpp.zig");

pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sdk: ?[]const u8,
    reference: llamacpp.Artifacts,
    /// The CUDA toolkit when ggml has a CUDA backend: the gate executables
    /// link its libraries through `llamacpp.addCudaLibraries`.
    cuda: ?llamacpp.Cuda = null,
};

/// The `mtmd` module, for the tools and harnesses that import it.
pub fn module(b: *std.Build, opts: Options) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/mtmd/module.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
    });
    mod.addImport("llama", opts.reference.llama.root_module);
    return mod;
}

/// What a Stage 8 workstream's build file receives.
pub const GateOptions = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sdk: ?[]const u8,
    reference: llamacpp.Artifacts,
    /// The port, rooted at `src/mtmd/module.zig`.
    mtmd: *std.Build.Module,
    /// Upstream's libmtmd, compiled unmodified over `libllama-ref`.
    mtmd_ref: *std.Build.Step.Compile,
    /// As `Options.cuda`.
    cuda: ?llamacpp.Cuda,
};

/// Adds `test-mtmd`, the module's unit tests, and every workstream's steps.
pub fn add(b: *std.Build, opts: Options, mtmd_ref: *std.Build.Step.Compile) *std.Build.Module {
    const mod = module(b, opts);
    const tests = b.addTest(.{ .root_module = mod });
    llamacpp.addSdkPaths(b, mod, opts.sdk);
    const step = b.step("test-mtmd", "Run libmtmd's unit tests");
    step.dependOn(&b.addRunArtifact(tests).step);

    const gate_opts: GateOptions = .{
        .target = opts.target,
        .optimize = opts.optimize,
        .sdk = opts.sdk,
        .reference = opts.reference,
        .mtmd = mod,
        .mtmd_ref = mtmd_ref,
        .cuda = opts.cuda,
    };
    @import("mtmd_image_diff.zig").add(b, gate_opts);
    @import("mtmd_audio_diff.zig").add(b, gate_opts);
    @import("mtmd_decode_diff.zig").add(b, gate_opts);
    @import("mtmd_fixtures.zig").add(b, gate_opts);
    return mod;
}
