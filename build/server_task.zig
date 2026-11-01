//! Stage 10 step 3a (PLAN.md, Decision 65): the server's data layer --
//! `tools/server/`'s ports of `server-common`, `server-task` and
//! `server-schema` -- its unit tests (`zig build test-server-data`) and the
//! port's half of the `server-task-diff` gate (`zig build
//! server-task-diff-bins`, into `zig-out/server-task-diff/`).
//!
//! **Not a port.** This is build description.
//!
//! The reference half, `harness/server_task_ref.cpp` over upstream's
//! `server-common.cpp`, `server-task.cpp`, `server-schema.cpp` and
//! `server-chat.cpp` compiled unmodified, is linked by
//! `scripts/server-task-diff` against the libraries `zig build tools-ref`
//! installs, as `scripts/chat-msg-diff` links its own.

const std = @import("std");
const builtin = @import("builtin");
const llamacpp = @import("llamacpp.zig");
const upstream = @import("upstream.zig");
const mtmd_build = @import("mtmd.zig");

const root = "llama.cpp";

pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sdk: ?[]const u8,
    reference: llamacpp.Artifacts,
    cuda: ?llamacpp.Cuda = null,
    /// `zigjinja`, for `cli/common_chat/`'s templates, which the data
    /// layer's chat conversion renders through (Stage 10 step 4).
    jinja: ?*std.Build.Module = null,
};

/// The target triple `--version` prints, as `build/tools.zig` formats it.
fn buildTarget(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    return b.fmt("{s}-{s}", .{ @tagName(target.result.cpu.arch), @tagName(target.result.os.tag) });
}

/// The modules the data layer is compiled against.
pub const Modules = struct {
    tools_h: *std.Build.Module,
    tools_common: *std.Build.Module,
    cli_fit: *std.Build.Module,
    json: *std.Build.Module,
    common_chat: *std.Build.Module,
    mtmd: *std.Build.Module,
    data: *std.Build.Module,
};

/// The module tree: `tools_common` and `cli_fit` as the tools have them,
/// `json` (`tools/server/json_module.zig`, `cli_fit`'s `cli/json/` under
/// the name `cli/common_chat/` imports), `common_chat`, libmtmd's port,
/// and the data layer itself rooted at `tools/server/module.zig`.
pub fn modules(b: *std.Build, opts: Options, optimize: std.builtin.OptimizeMode) Modules {
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

    const fit_module = b.createModule(.{ .root_source_file = b.path("cli/fit.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    fit_module.addImport("llama", opts.reference.llama.root_module);

    const common_module = b.createModule(.{ .root_source_file = b.path("tools/common/module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    common_module.addImport("tools_h", tools_h_module);
    common_module.addOptions("build_info", build_info);
    common_module.addImport("cli_fit", fit_module);

    const json_module = b.createModule(.{ .root_source_file = b.path("tools/server/json_module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    json_module.addImport("cli_fit", fit_module);

    // Stage 9 step 3's PEG engine, which `cli/common_chat/chat_peg.zig` imports.
    const peg_module = b.createModule(.{ .root_source_file = b.path("cli/peg/module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    peg_module.addImport("json", json_module);
    peg_module.addImport("llama", opts.reference.llama.root_module);

    const chat_module = b.createModule(.{ .root_source_file = b.path("cli/common_chat/module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    chat_module.addImport("json", json_module);
    chat_module.addImport("peg", peg_module);
    chat_module.addImport("llama", opts.reference.llama.root_module);
    if (opts.jinja) |j| chat_module.addImport("zigjinja", j);

    const mtmd_module = mtmd_build.module(b, .{
        .target = opts.target,
        .optimize = opts.optimize,
        .sdk = opts.sdk,
        .reference = opts.reference,
    });

    const data = b.createModule(.{ .root_source_file = b.path("tools/server/module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    data.addImport("tools_common", common_module);
    data.addImport("tools_h", tools_h_module);
    data.addImport("cli_fit", fit_module);
    data.addImport("llama", opts.reference.llama.root_module);
    data.addImport("json", json_module);
    data.addImport("common_chat", chat_module);
    data.addImport("peg", peg_module);
    data.addImport("mtmd", mtmd_module);
    data.addOptions("build_info", build_info);

    return .{
        .tools_h = tools_h_module,
        .tools_common = common_module,
        .cli_fit = fit_module,
        .json = json_module,
        .common_chat = chat_module,
        .mtmd = mtmd_module,
        .data = data,
    };
}

/// Adds `test-server-data` and `server-task-diff-bins`.
pub fn add(b: *std.Build, opts: Options) void {
    // The data layer's own code is built ReleaseSafe whatever the
    // libraries under it are, as the HTTP layer's harness is.
    const safe: std.builtin.OptimizeMode = if (opts.optimize == .debug) .debug else .safe;
    const m = modules(b, opts, safe);

    const tests = b.addTest(.{ .root_module = m.data });
    llamacpp.addSdkPaths(b, m.data, opts.sdk);
    const test_step = b.step("test-server-data", "Run the server data layer's unit tests (tools/server/common, tokens, oaicompat, task, result, schema)");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    const step = b.step("server-task-diff-bins", "Build the port's server-task-diff harness (Stage 10 step 3a) into zig-out/server-task-diff");
    const port_mod = b.createModule(.{ .root_source_file = b.path("harness/server_task_diff.zig"), .target = opts.target, .optimize = safe, .link_libc = true, .link_libcpp = true });
    port_mod.addImport("server_data", m.data);
    port_mod.addImport("tools_common", m.tools_common);
    port_mod.addImport("tools_h", m.tools_h);
    port_mod.addImport("llama", opts.reference.llama.root_module);
    port_mod.addImport("json", m.json);
    port_mod.addImport("common_chat", m.common_chat);
    port_mod.addImport("mtmd", m.mtmd);
    llamacpp.addSdkPaths(b, port_mod, opts.sdk);
    llamacpp.addLibgcc(b, port_mod);
    llamacpp.addCudaRuntime(b, port_mod, opts.cuda);
    const port_exe = b.addExecutable(.{ .name = "server_task_port", .root_module = port_mod });
    step.dependOn(&b.addInstallArtifact(port_exe, .{ .dest_dir = .{ .override = .{ .custom = "server-task-diff" } } }).step);
}
