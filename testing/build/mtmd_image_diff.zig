//! Build file for `mtmd_image_diff`, wired
//! into `build.zig` through `build/mtmd.zig`'s `Gates`.
//!
//! **Not a port.** This is build description. It adds `mtmd-image-diff`:
//! `testing/harness/mtmd_image_diff.zig` over the `mtmd` module, with upstream's
//! `mtmd-image.cpp` compiled unmodified beside `testing/harness/mtmd_image_ref.cpp`
//! -- with the same flags `libmtmd-ref` uses (`build/tools.zig`), so the two
//! sides differ only in the code.

const std = @import("std");
const mtmd = @import("../../build/mtmd.zig");
const llamacpp = @import("../../build/llamacpp.zig");

const root = "llama.cpp";

/// Adds this workstream's steps.
pub fn add(b: *std.Build, opts: mtmd.GateOptions) void {
    const mod = b.createModule(.{
        .root_source_file = b.path("testing/harness/mtmd_image_diff.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // reference code compiles as shipped
        .sanitize_c = .off,
    });
    mod.addImport("mtmd", opts.mtmd);
    mod.addImport("llama", opts.mtmd.import_table.get("llama").?);
    mod.addImport("stdlib", llamacpp.stdlibOf(opts.mtmd));
    mod.addIncludePath(b.path(root ++ "/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/tools/mtmd"));
    const flags = &[_][]const u8{ "-std=c++17", "-w" };
    mod.addCSourceFiles(.{ .root = b.path(root), .files = &.{"tools/mtmd/mtmd-image.cpp"}, .flags = flags });
    mod.addCSourceFile(.{ .file = b.path("testing/harness/mtmd_image_ref.cpp"), .flags = flags });
    mod.linkLibrary(opts.reference.ggml);
    llamacpp.addSdkPaths(b, mod, opts.sdk);
    llamacpp.addCudaRuntime(b, mod, opts.cuda);

    const exe = b.addExecutable(.{ .name = "mtmd-image-diff", .root_module = mod });
    const install = b.addInstallArtifact(exe, .{});
    const run = b.addRunArtifact(exe);
    run.has_side_effects = true;
    run.addPassthruArgs();
    run.step.dependOn(&install.step);
    const step = b.step("mtmd-image-diff", "Diff the image preprocessors against upstream's mtmd-image.cpp, on bits");
    step.dependOn(&run.step);
}
