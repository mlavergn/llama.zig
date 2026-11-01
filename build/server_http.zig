//! Stage 10 step 2 (PLAN.md, Decision 65): the `http-diff` gate's two
//! harnesses, `zig build http-diff-bins` into `zig-out/http-diff/`.
//!
//! **Not a port.** This is build description.
//!
//! - `http_port`: `harness/http_diff.zig` over `tools/server/http.zig`, the
//!   port of `server-http.cpp` and cpp-httplib's server half.
//! - `http_ref`: `harness/http_ref.cpp` over upstream's `server-http.cpp`,
//!   `server-common.cpp` and `httplib.cpp`, compiled unmodified with
//!   `llama.cpp.zmake`'s `httplib_defines` and the web UI `-Dui` names
//!   (zmake's `ui-stub` without it; `build/ui.zig`), over upstream's
//!   `llama-common`, `libmtmd-ref` and `libllama-ref` -- what
//!   `server-common.cpp` links against.
//!
//! `build/tools.zig` builds the same `llama-common` for the tool
//! references; its source list is private there, and is exactly
//! `common/*.cpp`, `common/jinja/*.cpp` and `common/parsers/*.cpp` at this
//! pin, so it is globbed here rather than copied.

const std = @import("std");
const builtin = @import("builtin");
const llamacpp = @import("llamacpp.zig");
const upstream = @import("upstream.zig");

const root = "llama.cpp";

/// `llama.cpp.zmake`'s `httplib_defines`, as `build/tools.zig` has them.
const httplib_defines = [_][]const u8{
    "-DCPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576",
    "-DCPPHTTPLIB_LISTEN_BACKLOG=512",
    "-DCPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768",
    "-DCPPHTTPLIB_TCP_NODELAY=1",
};

const cxx_flags = [_][]const u8{ "-std=c++17", "-w" };


pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sdk: ?[]const u8,
    reference: llamacpp.Artifacts,
    /// Upstream's libmtmd, which `server-common.cpp` calls.
    mtmd_ref: *std.Build.Step.Compile,
    cuda: ?llamacpp.Cuda = null,
    /// The web UI both sides embed (`build/ui.zig`).
    ui: @import("ui.zig").Ui,
};

/// The target triple `--version` prints, as `build/tools.zig` formats it.
fn buildTarget(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    return b.fmt("{s}-{s}", .{ @tagName(target.result.cpu.arch), @tagName(target.result.os.tag) });
}

/// Every `.cpp` directly under `llama.cpp/<dir>`, sorted.
fn globCpp(b: *std.Build, dir_path: []const u8) []const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var dir = std.Io.Dir.cwd().openDir(b.graph.io, b.fmt(root ++ "/{s}", .{dir_path}), .{ .iterate = true }) catch @panic("cannot open a common/ directory");
    defer dir.close(b.graph.io);
    var it = dir.iterate();
    while (it.next(b.graph.io) catch null) |e| {
        if (e.kind == .file and std.mem.endsWith(u8, e.name, ".cpp")) out.append(b.allocator, b.fmt("{s}/{s}", .{ dir_path, e.name })) catch @panic("OOM");
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    return out.items;
}

fn refModule(b: *std.Build, opts: Options) *std.Build.Module {
    const mod = b.createModule(.{
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // Reference code compiles as shipped.
        .sanitize_c = .off,
    });
    for ([_][]const u8{ root, root ++ "/include", root ++ "/common", root ++ "/ggml/include", root ++ "/ggml/src", root ++ "/vendor", root ++ "/tools/mtmd" }) |p| {
        mod.addIncludePath(b.path(p));
    }
    return mod;
}

/// Adds `http-diff-bins`.
pub fn add(b: *std.Build, opts: Options) void {
    const step = b.step("http-diff-bins", "Build the http-diff harnesses (Stage 10 step 2) into zig-out/http-diff");
    const dest: std.Build.Step.InstallArtifact.Options = .{ .dest_dir = .{ .override = .{ .custom = "http-diff" } } };

    // The port: a module tree like the tools', with the HTTP layer's
    // own code built ReleaseSafe whatever the libraries under it are.
    const safe: std.builtin.OptimizeMode = if (opts.optimize == .debug) .debug else .safe;

    const tools_h = b.addTranslateC(.{
        .root_source_file = b.path("tools/common/c_import.h"),
        .target = opts.target,
        .optimize = opts.optimize,
    });
    tools_h.addIncludePath(b.path(root ++ "/include"));
    tools_h.addIncludePath(b.path(root ++ "/ggml/include"));
    tools_h.addIncludePath(b.path(root ++ "/tools/mtmd"));
    const tools_h_module = tools_h.createModule();

    const build_info = b.addOptions();
    build_info.addOption(u32, "build_number", upstream.build_number);
    build_info.addOption([]const u8, "commit", upstream.commit);
    build_info.addOption([]const u8, "compiler", "Zig " ++ builtin.zig_version_string ++ " clang");
    build_info.addOption([]const u8, "build_target", buildTarget(b, opts.target));

    const fit_module = b.createModule(.{ .root_source_file = b.path("cli/fit.zig"), .target = opts.target, .optimize = safe, .link_libc = true });
    fit_module.addImport("llama", opts.reference.llama.root_module);

    const common_module = b.createModule(.{ .root_source_file = b.path("tools/common/module.zig"), .target = opts.target, .optimize = safe, .link_libc = true });
    common_module.addImport("tools_h", tools_h_module);
    common_module.addOptions("build_info", build_info);
    common_module.addImport("cli_fit", fit_module);

    const http_module = b.createModule(.{ .root_source_file = b.path("tools/server/http.zig"), .target = opts.target, .optimize = safe, .link_libc = true });
    http_module.addImport("tools_common", common_module);
    http_module.addImport("cli_fit", fit_module);
    http_module.addImport("llama", opts.reference.llama.root_module);
    http_module.addImport("ui_assets", opts.ui.port_module);

    const port_mod = b.createModule(.{ .root_source_file = b.path("harness/http_diff.zig"), .target = opts.target, .optimize = safe, .link_libc = true, .link_libcpp = true });
    port_mod.addImport("http", http_module);
    port_mod.addImport("tools_common", common_module);
    port_mod.addImport("cli_fit", fit_module);
    port_mod.addImport("llama", opts.reference.llama.root_module);
    llamacpp.addSdkPaths(b, port_mod, opts.sdk);
    llamacpp.addLibgcc(b, port_mod);
    llamacpp.addCudaRuntime(b, port_mod, opts.cuda);
    const port_exe = b.addExecutable(.{ .name = "http_port", .root_module = port_mod });
    step.dependOn(&b.addInstallArtifact(port_exe, dest).step);

    const http_tests = b.addTest(.{ .root_module = http_module });
    const test_step = b.step("test-server-http", "Run the HTTP layer's unit tests (tools/server/http.zig, tools/server/httplib/)");
    test_step.dependOn(&b.addRunArtifact(http_tests).step);

    // The reference: upstream's llama-common, as build/tools.zig builds it.
    const build_info_cpp = b.addConfigHeader(.{
        .style = .{ .cmake = b.path(root ++ "/common/build-info.cpp.in") },
        .include_path = "build-info.cpp",
    }, .{
        .LLAMA_BUILD_NUMBER = upstream.build_number,
        .LLAMA_BUILD_COMMIT = upstream.commit,
        .BUILD_COMPILER = "Zig " ++ builtin.zig_version_string ++ " clang",
        .BUILD_TARGET = buildTarget(b, opts.target),
    }).getOutputFile();
    const common_mod = refModule(b, opts);
    common_mod.addCSourceFile(.{ .file = build_info_cpp, .flags = &cxx_flags });
    for ([_][]const u8{ "common", "common/jinja", "common/parsers" }) |d| {
        common_mod.addCSourceFiles(.{ .root = b.path(root), .files = globCpp(b, d), .flags = &cxx_flags });
    }
    common_mod.addIncludePath(b.path(root ++ "/vendor/cpp-httplib"));
    common_mod.addCSourceFiles(.{ .root = b.path(root), .files = &.{"vendor/cpp-httplib/httplib.cpp"}, .flags = &(cxx_flags ++ httplib_defines) });
    if (opts.target.result.os.tag == .windows) common_mod.linkSystemLibrary("ws2_32", .{});
    const common_ref = b.addLibrary(.{ .name = "llama-common-http-ref", .root_module = common_mod, .linkage = .static });

    const ref_mod = refModule(b, opts);
    ref_mod.addIncludePath(b.path(root ++ "/tools/server"));
    ref_mod.addIncludePath(b.path(root ++ "/vendor/cpp-httplib"));
    ref_mod.addIncludePath(opts.ui.ref_include);
    const flags = if (opts.cuda != null) &(cxx_flags ++ .{"-DGGML_USE_CUDA"}) else &cxx_flags;
    ref_mod.addCSourceFiles(.{ .root = b.path(root), .files = &.{ "tools/server/server-http.cpp", "tools/server/server-common.cpp" }, .flags = flags });
    ref_mod.addCSourceFile(.{ .file = opts.ui.ref_cpp, .flags = &cxx_flags });
    ref_mod.addCSourceFiles(.{ .root = b.path("harness"), .files = &.{"http_ref.cpp"}, .flags = &cxx_flags });
    ref_mod.linkLibrary(common_ref);
    ref_mod.linkLibrary(opts.mtmd_ref);
    ref_mod.linkLibrary(opts.reference.llama_ref);
    ref_mod.linkLibrary(opts.reference.ggml);
    llamacpp.addSdkPaths(b, ref_mod, opts.sdk);
    llamacpp.addLibgcc(b, ref_mod);
    llamacpp.addCudaRuntime(b, ref_mod, opts.cuda);
    const ref_exe = b.addExecutable(.{ .name = "http_ref", .root_module = ref_mod });
    step.dependOn(&b.addInstallArtifact(ref_exe, dest).step);
}
