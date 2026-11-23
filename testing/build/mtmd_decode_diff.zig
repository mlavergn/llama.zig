//! Build file for `mtmd_decode_diff`, wired
//! into `build.zig` through `build/mtmd.zig`'s `Gates`.
//!
//! **Not a port.** This is build description. It adds `mtmd-decode-diff`:
//! `testing/harness/mtmd_decode_diff.zig` over `src/mtmd/vendor/`, with upstream's
//! vendored `stb_image.h` and `miniaudio.h` compiled unmodified in
//! `testing/harness/mtmd_decode_ref.cpp` under the defines `mtmd-helper.cpp` uses.

const std = @import("std");
const mtmd = @import("../../build/mtmd.zig");
const llamacpp = @import("../../build/llamacpp.zig");

/// Adds this workstream's steps.
pub fn add(b: *std.Build, opts: mtmd.GateOptions) void {
    const vendor = b.createModule(.{
        .root_source_file = b.path("src/mtmd/vendor/module.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
    });
    const mod = b.createModule(.{
        .root_source_file = b.path("testing/harness/mtmd_decode_diff.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // reference code compiles as shipped
        .sanitize_c = .off,
    });
    vendor.addImport("stdlib", llamacpp.stdlibModule(b, opts.target, opts.optimize));
    mod.addImport("vendor", vendor);
    mod.addImport("stdlib", llamacpp.stdlibModule(b, opts.target, opts.optimize));
    mod.addIncludePath(b.path("llama.cpp/vendor"));
    mod.addCSourceFile(.{ .file = b.path("testing/harness/mtmd_decode_ref.cpp"), .flags = &.{ "-std=c++17", "-w" } });
    llamacpp.addSdkPaths(b, mod, opts.sdk);
    llamacpp.addCudaRuntime(b, mod, opts.cuda);

    const exe = b.addExecutable(.{ .name = "mtmd-decode-diff", .root_module = mod });
    const install = b.addInstallArtifact(exe, .{});
    const run = b.addRunArtifact(exe);
    run.has_side_effects = true;
    run.addPassthruArgs();
    run.step.dependOn(&install.step);
    const build_step = b.step("mtmd-decode-diff-build", "Build mtmd-decode-diff without running it");
    build_step.dependOn(&install.step);
    const step = b.step("mtmd-decode-diff", "Diff the vendored image and audio decoders against stb_image.h and miniaudio.h, on bits");
    step.dependOn(&run.step);

    const tests = b.addTest(.{ .root_module = vendor });
    const test_step = b.step("test-mtmd-vendor", "Run the vendored decoders' unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
